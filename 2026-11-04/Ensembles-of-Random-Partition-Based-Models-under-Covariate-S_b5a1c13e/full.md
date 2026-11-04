# Ensembles of Random Partition-Based Models under Covariate Shift<sup>∗</sup>

Mary Versa Clemens-Sewall and Eliza O’Reilly

November 4, 2026

## Abstract

Distribution shifts, including covariate shifts, are ubiquitous in real-world applications of machine learning. Empirical evidence suggests that, across a variety of models with and without downstream adaptation, ensembles tend to be more robust to covariate shift than single models. We extend the theory of covariate shift severity from single models to ensembles, providing theoretical justification for the empirical trends. We introduce new discrepancy measures of covariate shift severity to control the risk of partition-based kernel estimators. We then show that with suficient ensemble size, these models can achieve minimax-optimal risk convergence rates under covariate shift. Our results suggest that modelers can expect to need larger ensemble sizes for more severe covariate shifts.

## 1 Introduction

Distribution shift occurs when some or all of the data used to train a model is sampled from a shifted distribution, meaning one that difers from the distribution of a randomly sampled test point, and it is common in a wide range of machine learning applications [15]. General distribution shift narrows to covariate shift if the nature of the shift is restricted to the features, a.k.a. covariates. Specifically, covariate shift occurs when the marginal distributions of the feature vectors difer yet the conditional distributions of the labels given the features are identical [24].

Recent work has characterized covariate shifts severe enough that they afect convergence rates of nonparametric estimators, yet not so severe that they preclude learning altogether [16, 10, 23]. These studies have defined asymmetric discrepancy measures for source-target pairs of distributions and proven that these discrepancy measures control minimax rates of convergence. Coarser discrepancy measures, especially the transfer exponent [16], preceded finer ones, like the aggregate transfer exponent [10] and integrated transfer exponent [23, 10].

To establish that these covariate shift discrepancy measures control minimax rates, these works have demonstrated that certain single-learner (i.e., not ensembled) local averaging estimators achieve the reported rates, including nearest-neighbor, dyadic tree, and top-hat (a.k.a. na¨ıve [12]) kernel estimators [16, 10, 23]. In this paper, we extend this theory to random partition-based kernel estimators. Our interest in these models stems, in part, from the empirical success of random forest models [5] on many tabular data machine learning problems. Both random forests and random-partition based kernel estimators are ensembles of local averaging estimators where each local averaging estimator is a piecewise constant function over a randomly generated partition. By studying random partition-based kernel estimators, we aim to discover when covariate shift theory developed on single learners applies to models suited to ensembling, and when it does not.

Inspired by the integrated transfer exponent [23, 10], we define a covariate shift discrepancy measure (definition 5) that likewise emerges in the analysis of the estimator’s variance. Our random partition-based discrepancy measure exhibits some advantageous theoretical properties (see proposition 1). Of practical concern, it cannot always be controlled, even when the integrated transfer exponent can be (see proposition 3). Our analysis of a class of doubling measures demonstrates that to control a partition-based estimator’s variance under severe covariate shift, ensembling may be necessary, with larger ensembles needed to achieve minimax-optimal rates on more severe covariate shifts.

Our discrepancy measure is specified to the partitioning scheme used to initialize the single learners in the ensemble. We must choose a partitioning scheme to derive risk convergence rates for the random partitionbased kernel estimator under covariate shift. We choose to study the STIT tessellation as a partitioning scheme because of its theoretical advantages.

The STIT tree and its ensembles, especially the STIT random forest, have been proven to achieve minimax-optimal convergence rates on Lipshitz regression without covariate shift [22]. Indeed, the STIT random forest is the only random forest model to date to have been proven minimax-optimal in arbitrary dimension. This was possible because the stochastic geometry of the STIT tessellation can be described with precision [19, 22], thanks in part to the memorylessness of the exponential distribution and the Poisson process built upon it. However, this same memorylessness gives the stochastic geometry of the tessellation a heavy-tailed (i.e., exponential-tailed) phenomenon that renders the typical cell in a partition quite small. In a recent work, the authors remarked that the small-cell phenomenon complicates the pursuit of statistical inference on Mondrian trees [8], which are a special case of STIT trees. So, too, in our work, the cells of a single Mondrian tree seem to be too small for the single learner to be robust to severe covariate shift (see proposition 3). As with statistical inference [8], we overcome this challenge through ensembling Mondrian trees to suficiently widen the pool of independent training points that contribute to the local average.

We prove that, with a suficient ensemble size, the Mondrian kernel [2] estimator can achieve minimaxoptimal rates on a class of severe covariate shifts (see proposition 4), just like the top-hat kernel estimator can [23]. In doing so, we extend covariate shift theory from simpler, single learners to more complex, ensemble estimators.<sup>1</sup>

## 1.1 Empirical Evidence for the Robustness of Ensembles to Covariate Shift

Empirical studies in the literature report that ensembles outperform single learners under covariate shift. Among popular classification algorithms, random forests consistently experienced less performance degradation under covariate shift than single learners on a wide range of two-dimensional datasets [3]. While individual trees may learn associations that fail to generalize under covariate shift, ensembling them appears to dilute the efect of any one such bias [3, 9].

Many works incorporate ensembling with other interventions to improve the empirical robustness of various models to covariate shift [9, 26, 34, 17]. The results of these studies show that ensembling benefits performance under covariate shift both with and without additional interventions. For instance, image corruption can induce a covariate shift, and Table 1 in [35] reports that ensembles of Bayesian models outperform single models at multiple levels of corruption. Ensembling helped both baseline neural networks and ones that had been adapted to target data after training. Similar trends are reported on a dataset of financial loss given default, where covariate shift is observed between historical and more recent data. Tables 4 and 5 in [34] demonstrate that for all studied models, ensembling improved performance under covariate shift. Ensembles were also preferred for the covariate shifts that emerge in motor imagery related electroencephalography (EEG) signals as a user of an EEG-based brain-computer interface experiences changes in attention, fatigue, and other stimuli [26]. Combined with ensembling, further interventions can ofer benefits, but a straightforward ensemble of a classical model ofers a strong baseline.

To complement and explain these empirical findings, we explore the theoretical advantages of ensembling partition-based models. Our studied model isolates the role of ensembling from interventions that attempt to adapt a trained model to the target distribution. Of practical relevance, a modeler may train our model on a mix of source- and target-distributed data without knowing which distribution generated each data point. By contrast, further interventions, including all of those described in the previous paragraph, typically require the modeler to know which of their data points were generated by the target distribution.

## 1.2 Notation

We refer to probability measures by µ and ν throughout. Where applicable, $\mu$ refers to the target distribution, which generates the test point X for model evaluation and possibly some of the training data points’ feature vectors, while ν refers to the source distribution, which generates the shifted training points’ feature vectors. As such, our source-target pair is $( \nu , \mu )$ . Previous works have referred to the source-target pair as $( P , Q ) \ [ 2 3 ]$ or $( P _ { X } , Q _ { X } )$ [16], to accentuate them being marginal over the covariates. We choose to refer to the target instead by $\mu$ because it is the canonical notation for a measure and we wish to emphasize its importance to the covariate shift problem. The distribution $\mu$ is what the modeler wants to learn about, while ν poses a challenge with which the modeler must contend.

Our notation for measures aims to minimize confusion with ${ \mathcal { P } } _ { : }$ , by which we refer to a partition of $\mathbb { R } ^ { d }$ or of a subset thereof. Uppercase C is an arbitrary cell in a partition. The cell containing the point $x$ is called $Z _ { x }$ . In particular, $Z _ { 0 }$ is the cell containing the origin in $\mathbb { R } ^ { d }$ . Superscripts on $C$ or $Z _ { x }$ record indices or parameters of the partition to which the cell belongs, not powers. Lowercase $c _ { \gamma }$ is a constant depending on the variable $\gamma$ . We use this notation sparingly; in order to avoid confusion with the notation of cell $C ,$ we prefer to write $a \lesssim b$ to elide a constant $c > 0$ in the inequality $a \leq c \cdot b$ . Whenever $\lesssim$ is used, the elided constant’s dependence on other variables is noted in the text.

The norm $\| \cdot \|$ is arbitrary unless otherwise noted and log is the natural logarithm. The notation $\| f \| _ { \infty }$ refers to the maximum absolute value of the real-valued function $f .$ By $\mathbf { 1 } _ { E }$ we refer to the indicator function of event $E .$

## 1.3 Related Work on Covariate Shift Discrepancies

Let $B ( x , r )$ refer to the ball of radius r centered at x. Previously defined covariate shift discrepancy measures control the growth of the probability ratios $\frac { \mu ( B ( x , r ) ) } { \nu ( B ( x , r ) ) }$ as r shrinks. Intuitively, the limit of this ratio, as $r  \infty$ is the density ratio, which can be used to reweight shifted training data and mitigate the efect of covariate shift [24, 31]. By considering only positive, finite values of $r ,$ the transfer exponent and other covariate shift discrepancy measures avoid assuming that the density ratio is well-defined [16].

Definition 1. $\gamma > 0$ is called a transfer exponent for the source-target pair of distributions $( \nu , \mu )$ if there is a constant $c _ { \gamma } > 0$ that depends on $\gamma$ such that, for all $r \in ( 0$ , diam(supp(µ))),

$$
\operatorname* { s u p } _ { x \in \mathrm { s u p p } ( \mu ) } \frac { \mu ( B ( x , r ) ) } { \nu ( B ( x , r ) ) } \leq c _ { \gamma } r ^ { - \gamma } .
$$

Following the introduction of the transfer exponent [16], a refinement was introduced [23] and named the integrated transfer exponent [10]. It controls an integral over the choice of test point $X \sim \mu ,$ , as opposed to the transfer exponent, which controls a supremum over all possible values of $x \in \operatorname { s u p p } ( \mu )$ . The integral itself, for fixed $r ,$ can also be seen as a covariate shift discrepancy; it was so introduced by [23], and we refer to it as the integrated r-transfer discrepancy.

Definition 2. For the source-target pair of distributions $( \nu , \mu )$ and fixed $r > 0$ , the integrated r-transfer discrepancy is defined as

$$
\mathbb { E } _ { X \sim \mu } \left[ \frac { 1 } { \nu ( B ( X , r ) ) } \right] .
$$

Moreover, $\gamma > 0$ is called an integrated transfer exponent for the source-target pair of distributions $( \nu , \mu )$ if there is a constant $c _ { \gamma } > 0$ that depends on $\gamma$ such that, for all $r > 0$ ，

$$
\mathbb { E } _ { X \sim \mu } \left[ \frac { 1 } { \nu ( B ( X , r ) ) } \right] \leq c _ { \gamma } r ^ { - \gamma } .\tag{1}
$$

The expression of the integrated r-transfer discrepancy emerges when analyzing the variance of the Nadaraya-Watson kernel estimator with the top-hat kernel $k ( x , x ^ { \prime } ) = \mathbf { 1 } _ { \| x - x ^ { \prime } \| \leq \tau }$ under covariate shift. The integrated transfer exponent controls the convergence rate as n grows and r correspondingly shrinks, per Theorem 1 in [23].

A closely related discrepancy emerges from analyzing the variance of a dyadic classification tree [10]. The local averaging in a dyadic tree is performed over a leaf, which is a cell in the dyadic partition of feature space, so the cells of such partitions replace $B ( x , r )$ in the definition.

Definition 3. An r-grid Ξ is a partition of $\operatorname { s u p p } ( \nu )$ such that each cell $C \in \Xi$ has a point $x _ { C }$ with $B ^ { \infty } ( x _ { C } , r ) \subseteq C \subseteq B ^ { \infty } ( x _ { C } , 2 r )$ where $B ^ { \infty } ( x , r )$ is the $\ell _ { \infty } .$ -norm ball with center x and radius r. For the source-target pair of distributions $( \nu , \mu )$ and fixed $r > 0$ , the aggregate r-transfer discrepancy is defined as the supremum of

$$
\sum _ { C \in \Xi } { \frac { \mu ( C ) } { \nu ( C ) } }
$$

over all $r { \mathrm { - g r i d s ~ } } \Xi$ . Moreover, $\gamma > 0$ is an aggregate transfer exponent for the source-target pair of measures $( \nu , \mu )$ if there exists a constant $c _ { \gamma } > 0$ such that, for all $r > 0$ and r-grids Ξ,

$$
\sum _ { C \in \Xi } \frac { \mu ( C ) } { \nu ( C ) } \leq c _ { \gamma } r ^ { - \gamma } .\tag{2}
$$

Proposition 9 in [10] proves that, at least for source-target pairs of doubling measures, the infimum integrated and aggregate transfer exponents coincide.

## 1.4 Background on Tessellations and Partitions

Before introducing partition-based covariate shift discrepancy measures, we must define the class of partitions we study. Let $\mathcal { P }$ be a partition of $\mathbb { R } ^ { d }$ , meaning a collection of pairwise disjoint sets whose union is $\mathbb { R } ^ { d }$ . We will generally restrict this partition to supp(ν), the set of all possible data or evaluation points. We denote by $Z _ { x }$ the cell of the partition that contains a point x in $\mathbb { R } ^ { d }$

We restrict our attention to deterministic and randomly generated partitions such that the closures of all the cells $\{ \operatorname { c l } ( C ) : C \in { \mathcal { P } } \}$ form a mosaic. A mosaic in $\mathbb { R } ^ { d }$ is defined as a collection of bounded, convex bodies whose interiors are pairwise disjoint and whose union is $\mathbb { R } ^ { d } \ [ 2 1 ]$ . For example, unit d-dimensional cubes tile $\mathbb { R } ^ { d }$ , forming a mosaic

$$
\{ [ n _ { 1 } , n _ { 1 } + 1 ] \times [ n _ { 2 } , n _ { 2 } + 1 ] \times \cdots \times [ n _ { d } , n _ { d } + 1 ] : n _ { k } \in \mathbb { Z } \} .
$$

To obtain a partition from a mosaic, each boundary between interiors must be assigned to one of the bodies it is contained in. For the grid, this can be accomplished by assigning each face shared between multiple cubes to the uppermost cube in all directions, yielding the partition

$$
\{ [ n _ { 1 } , n _ { 1 } + 1 ) \times [ n _ { 2 } , n _ { 2 } + 1 ) \times \cdots \times [ n _ { d } , n _ { d } + 1 ) : n _ { k } \in \mathbb { Z } \} .
$$

This and a few other hypotheses about the partition are summarized below.

Hypothesis 1 (Regarding the Partition). The closures of the cells $\{ \operatorname { c l } ( C ) : C \in { \mathcal { P } } \}$ form a mosaic in $\mathbb { R } ^ { d }$ implying the interior int(C) is convex for all $C \in \mathcal { P }$ . Also, $\mathcal { P }$ is locally finite, meaning for any compact set $K ,$ , the set $\{ C \cap K : C \in { \mathcal { P } } \}$ has finite cardinality. Lastly, P is generated independently of the test point X.

The union of boundaries of a mosaic is called a tessellation [21]. The boundaries are less of a concern if the mosaic is determined by a random process with a property called stationarity.

Definition 4. A random tessellation $\tau$ is stationary if its distribution is invariant under translation, meaning that for all $x \in \mathbb { R } ^ { d }$ , the distribution of $\tau + x$ is the same as that of $\tau$

A tessellation has Lesbesgue measure of zero, but it could have nonzero $\mu -$ or ν-mass if these measures are not absolutely continuous with respect to Lesbesgue measure. However, if the tessellation is randomly generated and stationary, any fixed point x will almost surely (with respect to the tessellation) fall in the interior of one of the cells, so we can evaluate distributional quantities, like the estimator’s expected prediction error of a randomly generated $( \mathrm { b y } ~ \mu )$ test point X, without concern for the case that X is contained in the tessellation (as opposed to the interior of a partition cell).

## 1.5 Organization

Section 2 defines covariate shift discrepancies for ensembles of partitions. These partition-based discrepancies have useful theoretical properties (e.g., proposition 1) and indicate the utility of ensembling (see lemma 1). Section 3 provides a risk bound for an ensembled partition-based estimator under covariate shift, demonstrating that our partition-based discrepancy controls the variance. Section 4 specializes the risk bound to STIT tessellations and, after presenting a novel bound on the bias (lemma 2), provides a convergence rate for the risk of a STIT kernel estimator. Finally, Section 5 evaluates the partition-based discrepancy for a frequently-studied class of severe covariate shifts and demonstrates that the STIT kernel estimator achieves minimax-optimal convergence rates with a suficiently large ensemble. Three longer proofs are deferred to the appendix.

## 2 Partition-based Covariate Shift Discrepancies

Local averaging estimators can remain consistent under even fairly severe covariate shifts because the shifted training data near a test point have labels similar, on average, to the value of the shifted regression function at the test point, which is the same as the regression function for the test distribution. Covariate shift may erode the estimator’s convergence rate (possibly so dramatically that it renders the estimator inconsistent without unshifted data [16]) by reducing the speed at which the training set accumulates points near the test point and hence increasing the model’s variance; this occurs if the neighborhood around the test point has less probability mass under the shifted (marginal) distribution ν than the target (marginal) distribution $\mu .$ Nevertheless, so long as some distribution generating training data has positive probability on a neighborhood of the test point, eventually enough training data will accumulate near the test point that its partition cell can be refined (or kernel bandwidth shrunk) to reduce the bias while controlling the variance.

Quantitatively, transfer exponents appear in place of (or in addition to) the dimension d in the minimaxoptimal convergence rates [16, 23]. If severe enough to afect minimax rates, a covariate shift has a transfer exponent large enough that d is replaced by a larger number. This makes the learning problem as “hard” $( \mathrm { i . e . } ,$ having a minimax rate as slow) as a problem in higher dimensions than $d .$

To define a transfer exponent and reveal it in a model’s convergence rate, we require a one-dimensional complexity parameter for the model. For instance, the top-hat kernel bandwidth is a complexity parameter that allows for studying the top-hat kernel estimator, a single learner. However, a general ensemble of partitions may not have a single complexity parameter, so we begin by defining “transfer discrepancies” that allow us to derive risk bounds without reference to a model complexity parameter. In Section 4, we study a random partition with a one-dimensional complexity parameter and define a partition-based transfer exponent.

We now define partition-based transfer discrepancies. In anticipation of ensembling random partitionbased estimators, we state these definitions with an ensemble size parameter $M$ , which is always a positive integer. Partitions $\left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M }$ are identically distributed as $\mathcal { P } .$ . If $\mathcal { P }$ is deterministic (i.e., a point-mass in the space of partitions), it follows that the realizations $\left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M }$ are identical. We use the notation $Z _ { x } ^ { m }$ to refer to the cell in ${ \mathcal { P } } ^ { ( m ) }$ that contains the point x.

Definition 5. For a partition ${ \mathcal { P } } ,$ an ensemble size $M \geq 1$ , and a source-target pair of distributions $( \nu , \mu )$ ， the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy is defined as

$$
\mathbb { E } \left[ \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m } \right) } \right] ,
$$

where X is distributed as $\mu .$ If P is random, the expectation is taken over the partitions, too.

In contrast to the integrated r-transfer discrepancy (definition 2), the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy replaces $B ( X , r )$ with the union $\textstyle \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m }$

The aggregate $\mathcal { P } ^ { M }$ -transfer discrepancy handles the ensemble intricately.

Definition 6. For a partition ${ \mathcal { P } } _ { : }$ , an ensemble size $M \geq 1$ , and a source-target pair of distributions $( \nu , \mu )$ the aggregate $\mathcal { P } ^ { M }$ -transfer discrepancy is defined as

$$
\mathbb { E } \left[ \sum _ { ( C ^ { ( m ) } ) _ { m = 1 } ^ { M } \in \prod _ { m = 1 } ^ { M } \mathcal { P } ^ { ( m ) } : } \frac { \mu \left( \bigcap _ { m = 1 } ^ { M } C ^ { ( m ) } \right) } { \nu \left( \bigcup _ { m = 1 } ^ { M } C ^ { ( m ) } \right) } \right] ,
$$

where the sum is taken over length-M tuples $( C ^ { ( m ) } ) _ { m = 1 } ^ { M }$ such that $C ^ { ( m ) } \in \mathcal { P } ^ { ( m ) }$ for each $m$ and also $\textstyle \bigcap _ { m = 1 } ^ { M } C ^ { ( m ) }$ has nonempty intersection with supp(µ). The expectation is taken over the realizations of $\mathcal { P } .$

The aggregate $\mathcal { P } ^ { 1 }$ -transfer discrepancy can be directly compared to the aggregate r-transfer discrepancy (definition 3) when P is almost surely a grid. Since the aggregate $\mathcal { P } ^ { 1 }$ -transfer discrepancy is an average, rather than a supremum, it may be smaller.

Remark 1. Adopt Hypothesis 1. Suppose $\mathcal { P }$ is almost surely an r-grid. Then, the aggregate $\mathcal { P } ^ { 1 }$ -transfer discrepancy is no greater than the aggregate r-transfer discrepancy, so long as both are finite. More precisely, if with probability 1 with respect to $\mathcal { P }$ the realization of $\mathcal { P }$ is an r-grid, then we can compute the aggregate ${ \mathcal { P } } ^ { 1 } .$ -transfer discrepancy by integrating over this probability-1 region only:

$$
\mathbb { P } [ \mathcal { P } \mathrm { ~ i s ~ a n ~ } r \mathrm { - g r i d } ] = 1 \implies \mathbb { E } \left[ \sum _ { \substack { C \in \mathcal { P } : } \nu ( C ) } \frac { \mu ( C ) } { \nu ( C ) } \right] \le \operatorname* { s u p } _ { \substack { \Xi \mathrm { ~ i s ~ a n ~ } r \mathrm { - g r i d } } } \sum _ { \substack { C \in \Xi : } \nu ( C ) } \frac { \mu ( C ) } { \nu \left( C \right) } \cdot
$$

Under certain conditions, it has been shown that the integrated and aggregate transfer exponents are equal [10], but the relation is not known in general. Here we establish equality between the integrated $\mathcal { P } ^ { \bar { M } }$ -transfer discrepancy and the aggregate $\mathcal { P } ^ { \overset { \sim } { M } }$ -transfer discrepancy.<sup>2</sup>

Proposition 1. Adopt Hypothesis 1. For partition P and ensemble size M, the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy and aggregate $\mathcal { P } ^ { M }$ -transfer discrepancy are equal.

Proof. Consider a fixed $x \in \operatorname { s u p p } ( \mu )$ . Each partition ${ \mathcal { P } } ^ { ( m ) }$ has exactly one cell $Z _ { x } ^ { m }$ that contains $x ,$ so the Cartesian product $\textstyle \prod _ { m = 1 } ^ { M } { \mathcal { P } } ^ { ( m ) }$ has exactly one tuple $( C ^ { ( m ) } ) _ { m = 1 } ^ { M }$ such that $\begin{array} { r } { x \in \bigcap _ { m = 1 } ^ { M } C ^ { ( m ) } } \end{array}$ . Moreover, x is in the intersection $\textstyle \bigcap _ { m = 1 } ^ { M } C ^ { ( m ) } \Gamma$ supp $( \mu )$ , hence this intersection is not empty, so the tuple is a member of the subset of the Cartesian product $\textstyle \prod _ { m = 1 } ^ { M } { \mathcal { P } } ^ { ( m ) }$ restricted to $\textstyle \bigcap _ { m = 1 } ^ { M } C ^ { ( m ) } \cap \operatorname { s u p p } ( \mu ) \neq \varnothing$ . Summing an indicator over all such tuples,

$$
\frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m } \right) } = \sum _ { \substack { ( C ^ { ( m ) } ) _ { m = 1 } ^ { M } \in \prod _ { m = 1 } ^ { M } \mathcal { P } ^ { ( m ) } : } } \frac { \mathbf { 1 } _ { x \in \bigcap _ { m = 1 } ^ { M } C ^ { ( m ) } } } { \nu \left( \bigcup _ { m = 1 } ^ { M } C ^ { ( m ) } \right) } .
$$

The sum above is finite because each partition ${ \mathcal { P } } ^ { ( m ) }$ is locally finite and so has finitely many cells with nonempty intersection of the compact set supp(µ). Because X is realized independently from the partitions, we can apply linearity of expectation to the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy, recovering the aggregate

$\mathcal { P } ^ { M }$ -transfer discrepancy as claimed:

$$
\begin{array} { r l } { \mathbb { E } [ \frac { 1 } { \nu } ( \bigcup _ { i = 1 } ^ { H } \mathcal { L } _ { x } ^ { ( m ) } ) ] = \mathbb { E } [ \underset { ( \Gamma _ { u - 1 } ^ { ( m ) } , \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } ) \leq 0 } { \sum } \frac { 1 } { \nu } ( \bigcup _ { i = 1 } ^ { H } \mathcal { L } _ { x } ( \Gamma _ { u - 1 } ^ { ( m ) } , C ^ { ( m ) } ) ) ] } & { = \mathbb { E } [ \underset { ( \Gamma _ { u - 1 } ^ { ( m ) } , \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } ) \leq 0 } { \sum } \frac { 1 } { \nu } ( \bigcup _ { i = 1 } ^ { H } C ( \mathsf { m } ) ) ] } \\ & { = \mathbb { E } _ { [ \nu ^ { ( m ) } \cap \Gamma _ { u - 1 } ^ { ( m ) } ] } [ \underset { ( \Gamma _ { u - 1 } ^ { ( m ) } , \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } ) \leq 0 } { \sum } \frac { 1 } { \nu } [ \underset { ( \Gamma _ { u - 1 } ^ { ( m ) } , C ^ { ( m ) } ) } { \sum } ] } \\ & { = \mathbb { E } [ \underset { ( \Gamma _ { u - 1 } ^ { ( m ) } , \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } } ] \times ( \bigcup _ { m - 1 } ^ { H } C ( \mathsf { m } ) ) } \\ &  = \mathbb { E } [ \underset  ( \Gamma _ { u - 1 } ^ { ( m ) } , \Gamma _ { u } ^ { ( m ) } \cap \Gamma _ { u } ^ { ( m ) } \cap \Gamma  \end{array}
$$

The integrated transfer exponent is a refinement of the transfer exponent because it considers an average test point x, rather than the worst case. The aggregate r-transfer exponent approximates an average by summing probability ratios of cells in a partition of supp(µ), but it considers the worst-case partition. By randomizing the partition, we obtain a notion of an average partition, allowing us to study the average test point and average model.

The form of the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy hints at the utility of ensembling because the discrepancy (weakly) decreases as M increases.

Lemma 1. Adopt Hypothesis 1. For source-target pair $( \nu , \mu )$ and partition ${ \mathcal { P } } _ { : }$ , the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy is decreasing in M.

Proof. Condition on the realizations of the partition. $\mathrm { B y }$ monotonicity of measure,

$$
\frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m } \right) } \leq \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M - 1 } Z _ { X } ^ { m } \right) } .
$$

The relation still holds when we take the expectation of both sides.

The partitions in the ensemble only difer if the partition is randomized, so reductions in the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy due to ensembling are only possible using random partitions. We will show in Section 5 that these reductions can be substantial.

## 3 Risk Bounds for General Partition-based Kernel Estimators

Define the partition-based kernel estimator

$$
\begin{array} { r } { \hat { f } _ { \mathcal { P } , M } ( x ) = \left\{ \begin{array} { l l } { \frac { \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { n } Y _ { i } \mathbf { 1 } _ { X _ { i } \in Z _ { x } } } { \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in Z _ { x } } } } & { \mathrm { ~ i f ~ } \exists m , i : X _ { i } \in Z _ { x } ^ { m } } \\ { 0 } & { \mathrm { ~ i f ~ } \nexists m , i : X _ { i } \in Z _ { x } ^ { m } } \end{array} \right. . } \end{array}
$$

$\hat { f } _ { \mathcal { P } , M }$ is a kernel method whose kernel is called the “kernel based on random forest” in [29]. We refer to it as a kernel estimator for brevity.<sup>3</sup> Note that it also has the form of a random binning features model [25] with bins prescribed by the partitions.

The estimate is learned on the pairwise independent dataset $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ . The feature vectors $\{ X _ { i } \} _ { i = 1 } ^ { n }$ take values in $\mathbb { R } ^ { d }$ , and the labels $\{ Y _ { i } \} _ { i = 1 } ^ { n }$ take values in $\mathbb { R } .$ . The first $n _ { \nu } \geq 0$ data points $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n _ { \nu } }$ are independently and identically distributed as $\nu _ { \mathrm { j o i n t } }$ , a probability measure over $\mathbb { R } ^ { d } \times \mathbb { R }$ . The remaining $n _ { \mu } ~ = ~ n - n _ { \nu } ~ \ge ~ 0$ data points $\{ ( X _ { i } , Y _ { i } ) \} _ { i = n _ { \nu } + 1 } ^ { n }$ are independently and identically distributed as $\mu _ { \mathrm { j o i n t } } ,$ another probability measure over $\mathbb { R } ^ { d } \times \mathbb { R } . ^ { 4 }$ Let $\mu$ and ν refer to the marginal distributions of $\mu _ { \mathrm { j o i n t } }$ and $\nu _ { \mathrm { j o i n t } }$ , respectively, over $\mathbb { R } ^ { d }$ . In other words, the first $n _ { \nu }$ feature vectors $\{ X _ { i } \} _ { i = 1 } ^ { n _ { \nu } }$ are distributed as ν and the remaining $n _ { \mu }$ feature vectors $\{ X _ { i } \} _ { i = n _ { \nu } + 1 } ^ { n }$ are distributed as $\mu .$

Our risk bounds apply to partition-based kernel estimators whose partitions are independent of the training data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ . Random forests with this property are called purely random forests and are a frequent object of study [14, 30, 11].

Hypothesis 2 (Purely Random Partitioning). The partition $\mathcal { P }$ is independent of the training data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$

Dependence on the training data is the primary reason the partitions would have dependence on one another. For simplicity, we assume the partitions are independently realized.

Hypothesis 3 (Independent Partitions). $\left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M }$ are generated independently.

The supports of $\mu$ and $\nu ,$ which are closed by definition, must be nested for the transfer exponent to be finite (Example 1 in [16]). We also assume they are compact.

Hypothesis 4 (Nested and Compact Support). The source and target supports are nested, i.e., $\operatorname { s u p p } ( \mu ) \subseteq$ supp(ν). Also, supp(ν) is bounded. Hence, supp(µ) and $\operatorname { s u p p } ( \nu )$ are compact.

Additionally, we assume the following about the conditional distribution of $Y | X$ when $( X , Y )$ is sampled from either $\mu _ { \mathrm { j o i n t } }$ or $\nu _ { \mathrm { j o i n t } }$ . This is the defining hypothesis of covariate shift, which distinguishes it from general distribution shift. It allows us to focus on the discrepancy between $\mu$ and ν rather than that of $\mu _ { \mathrm { j o i n t } }$ and $\nu _ { \mathrm { j o i n t } }$

Hypothesis 5 (Covariate Shift). The regression function $f ^ { * } : \mathrm { s u p p } ( \nu )  \mathbb { R }$ defined as

$$
f ^ { * } ( x ) : = \mathbb { E } [ Y | X = x ]
$$

is identical for $( X , Y ) \sim \mu _ { \mathrm { j o i n t } }$ and $( X , Y ) \sim \nu _ { \mathrm { j o i n t } }$

Consistent with previous theoretical work on nonparametric regression under covariate shift [23, 10], we study H¨older-continuous regression functions that are Lipshitz or rougher.

Hypothesis 6 (Smoothness of the Regression Function). There is some $\beta \in ( 0 , 1 ]$ such that $f ^ { * }$ is $\beta \mathrm { . }$ H¨oldercontinuous with constant L so that for any $x , x ^ { \prime } \in \operatorname { s u p p } ( \nu )$ 2

$$
| f ^ { * } ( x ) - f ^ { * } ( x ^ { \prime } ) | \leq L \| x - x ^ { \prime } \| ^ { \beta } .
$$

Finally, we impose a second-moment bound on the noise in the training labels. The noise $\xi _ { i } : = Y _ { i } - f ^ { * } ( X _ { i } )$ may depend on the features $X _ { i }$ so long as the variance is uniformly bounded.

Hypothesis $\mathbf { 7 }$ (Label Noise). For every index $i \leq n$ , there is a random variable $\xi _ { i }$ that may depend only on $X _ { i }$ (and no other random elements) such that

$$
Y _ { i } = f ^ { * } ( X _ { i } ) + \xi _ { i } .
$$

Moreover, the second moment of the noise is conditionally bounded by $\sigma ^ { 2 } > 0$ . For all $i \leq n$

$$
\operatorname* { s u p } _ { x \in \mathrm { s u p p } ( \nu ) } \mathbb { E } \left[ \xi _ { i } ^ { 2 } \big | X _ { i } = x \right] \leq \sigma ^ { 2 } .
$$

Note that this implies $\mathbb { E } [ \xi _ { i } | X _ { i } ] = 0$ for all $i .$

We now derive a pointwise risk bound on $\hat { f } _ { \mathcal { P } , M }$

Theorem 1. Adopt Hypotheses 1 and 2 to 7. For fixed $x \in \operatorname { s u p p } ( \mu )$ ，

$$
\begin{array} { r l } & { \mathbb { E } [ ( \hat { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } ] \leq L ^ { 2 } \cdot \mathbb { E } \left[ \underset { m = 1 } { \overset { M } { \operatorname { M a x } } } \mathrm { d i a m } ( Z _ { x } ^ { m } \cap \operatorname { s u p p } ( \nu ) ) ^ { 2 \beta } \right] } \\ & { \qquad + \frac { 1 8 \sigma ^ { 2 } + ( f ^ { * } ( x ) ) ^ { 2 } } { n } \cdot \mathbb { E } \left[ n \wedge \frac { 1 } { \frac { n _ { \nu } } { n } \cdot \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m } \right) + \frac { n _ { \mu } } { n } \cdot \mu \left( \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m } \right) } \right] . } \end{array}
$$

The proof of theorem 1 is provided in Section 7. If the partition is stationary, we can take the expectation with respect to a random test point $X \sim \mu .$ . This yields an integrated risk bound that depends on a truncation of the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy.

Corollary 1. Adopt Hypotheses 1 and 2 to $^ { 7 , }$ and suppose $\mathcal { P }$ is stationary. Then,

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( \hat { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } \big ] \leq L ^ { 2 } \cdot \mathbb { E } \left[ \underset { m = 1 } { \overset { M } { \operatorname { M a x } } } \mathrm { d i a m } ( Z _ { 0 } ^ { m } ) ^ { 2 \beta } \right] } \\ & { \qquad + \frac { 1 8 \sigma ^ { 2 } + \left\| f ^ { * } \right\| _ { \infty } ^ { 2 } } { n } \cdot \mathbb { E } \left[ n \wedge \frac { 1 } { \frac { n _ { \nu } } { n } \cdot \nu \left( { \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m } } \right) + \frac { n _ { \mu } } { n } \cdot \mu \left( { \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m } } \right) } \right] . } \end{array}
$$

Proof. Stationarity means that, for any point x and index $m ,$ , the cell $Z _ { x } ^ { m }$ is distributed as $Z _ { 0 } ^ { m } + x$ . The diameter is a translation-invariant function of the cell, so we can translate $Z _ { x } ^ { m }$ back to the origin without changing the diameter, thus obtaining a bound for the bias term that does not depend on x:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { x } ^ { m } \cap \operatorname { s u p p } ( \nu ) ) ^ { 2 \beta } \right] \leq \mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { x } ^ { m } ) ^ { 2 \beta } \right] = \mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m } ) ^ { 2 \beta } \right] .
$$

We combine this observation with theorem 1 and take the supremum of $f ^ { * }$ to conclude.

Observe that the latter term in corollary 1, which controls the variance, is no greater than the integrated $\mathcal { P } ^ { M }$ -transfer discrepancy for source target pair $\begin{array} { r } { \left( \left( \frac { n _ { \nu } } { n } \cdot \nu + \frac { n _ { \mu } } { n } \cdot \mu \right) , \mu \right) } \end{array}$

## 4 Convergence Rates for STIT Kernel Estimators

For the remainder of the paper, we focus on random partitions induced by independently realized STIT tessellations. In this section, we briefly introduce the tessellation and define a specialized transfer exponent. We then derive a convergence rate for the associated partition-based kernel estimator that depends on this transfer exponent.

## 4.1 STIT Tessellation

For a construction of the STIT tessellation, we refer to [20] and [21]. To state our results, it sufices to present only the defining parameters and characteristics of the STIT tessellation. STIT (“stable under iteration”) tessellations are a class of stationary random tessellations parameterized by deterministic convex bodies called centered zonoids. The support function of a centered zonoid $\Pi ^ { \prime }$ can be expressed as

$$
h ( \Pi ^ { \prime } , v ) : = \operatorname* { s u p } _ { x \in \Pi ^ { \prime } } \langle v , x \rangle = \frac { \lambda } { 2 } \int _ { \mathbb { S } ^ { d - 1 } } | \langle u , v \rangle | \ d \phi ( u ) ,
$$

where $\phi$ is an even probability measure on the sphere $\mathbb { S } ^ { d - 1 }$ and λ is a positive real number. We refer to λ as the lifetime and $\phi$ as the directional distribution. The STIT tessellation gets its name from its property of being stable under iteration, from which follows the scaling property: the $\frac { 1 } { \lambda }$ dilation of the tessellation with lifetime λ and directional distribution $\phi$ has the same distribution as the tessellation of lifetime 1 and directional distribution $\phi .$ The zonoid Π associated to the tessellation of lifetime 1 is called the normalized zonoid associated to all STIT tessellations with its same directional distribution $\phi .$

![](images/1ba8dcc81b469674ebe84ab3cbd257153e1703e670e50558fa522b7008290477.jpg)  
Figure 1: Four realizations of the Mondrian tessellation-induced partition of the square $[ 0 , 1 ] ^ { 2 }$ are shown individually on the left and superimposed on the right. The test point $x = \left( { \textstyle { \frac { 1 } { 2 } } } , { \textstyle { \frac { 1 } { 2 } } } \right)$ is marked in blue, and the cell $Z _ { x }$ is shaded in red. The union of the independently realized partitions’ cells $Z _ { x }$ is larger than any single partition’s, illustrating how ensembling reduces the $\dot { \mathcal { P } } ^ { M }$ -transfer discrepancy (lemma 1).

As such, whenever P is induced by a STIT tessellation, we can fully describe its distribution by a pair (λ, Π). The lifetime λ captures the complexity, with higher values of λ corresponding to finer partitions. The normalized associated zonoid Π captures the relative likelihood of cell boundaries oriented in various directions.

For example, the normalized zonoid associated to the Mondrian tessellation is the d-dimensional axisaligned cube with side length $\textstyle { \frac { d } { 2 } } ;$ , so all cell boundaries in the Mondrian are axis-aligned, and boundaries are equally likely to be aligned to any of the axes. Four realizations of the Mondrian in dimension $d = 2$ (with fixed lifetime λ) are shown in fig. 1, both individually and superimposed. For the test point $\begin{array} { r } { x = \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) } \end{array}$ , the cell $Z _ { x }$ is highlighted in each. The union of these cells is (stochastically) larger than any single one, which is why ensembling reduces the $\mathcal { P } ^ { M } .$ -transfer discrepancy (see lemma 1).

If Π is a Euclidean ball, the STIT tessellation is called isotropic. Figure 2 shows four realizations of the isotropic STIT and their superposition.

The lifetime λ is central to the theoretical analysis of STIT tessellation estimators because tuning λ allows modelers to balance the bias-variance tradeof and achieve optimal convergence rates. The directional distribution plays a subtler role, afecting the dependence of risk bounds on d but not convergence rates in n. Hence, we use λ as a shorthand for the parameterization of a STIT tessellation used in defining estimators or discrepancies. $\mathcal { P } _ { \lambda }$ is our notation for the partition induced by a STIT tessellation parameterized by $( \lambda , \Pi )$ and $Z _ { x } ^ { m , \lambda }$ refers to the cell in the partition $\mathcal { P } _ { \lambda } ^ { ( m ) }$ that contains the point x. The partitions $\left\{ { \mathcal P } _ { \lambda } ^ { ( m ) } \right\} _ { m = 1 } ^ { M }$ are always generated independently of one another and so satisfy Hypothesis 3.

Remark 2. $\mathcal { P } _ { \lambda }$ satisfies Hypotheses 1 and 2 and is stationary.

## 4.2 STIT Covariate Shift Discrepancy Measures

For STIT tessellations, we can define transfer exponents by letting the lifetime λ take the role of the inverse bandwidth $r ^ { - 1 }$

Definition 7. For ensemble size $M \geq 1$ , we call $\gamma \geq 0 \mathrm { ~ a ~ } \mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent if, for all $\lambda > \mathrm { d i a m } ( \operatorname { s u p p } ( \mu ) )$ ),

![](images/b259d4a3fbf1cb1e2f614418e86ffa23f488efc195aff20a6123cee8e16d6559.jpg)  
Figure 2: Four realizations of the isotropic STIT tessellation-induced partition of the square $[ 0 , 1 ] ^ { 2 }$ are shown individually on the left and superimposed on the right. The test point $\begin{array} { r } { x = \left( \frac { 1 } { 2 } , \frac { 1 } { 2 } \right) } \end{array}$ is marked in blue, and the cell $Z _ { x }$ is shaded in red.

with probability 1 with respect to the partitions $\left\{ { \mathcal P } _ { \lambda } ^ { ( m ) } \right\} _ { m = 1 } ^ { M }$ 2

$$
\operatorname* { s u p } _ { x \in \mathrm { s u p p } ( \mu ) } \frac { \mu \left( \bigcap _ { m = 1 } ^ { M } Z _ { x } ^ { m , \lambda } \right) } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m , \lambda } \right) } \lesssim \lambda ^ { \gamma } ,
$$

where $\lesssim$ elides a constant that depends on $\gamma , M ,$ , and Π.

Definition 8. For ensemble size $M \geq 1$ , we call $\gamma > 0$ an integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent if, for all $\lambda > 0$

$$
\mathbb { E } \left[ \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] \lesssim \lambda ^ { \gamma } ,
$$

where the expectation is taken over the independently realized partitions and X, which is distributed as $\mu .$ Here, $\lesssim$ elides a constant that depends on γ, M, and Π.

The $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent can be seen as dimension-free, whereas the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent incorporates dimension in the following sense. If $\gamma > 0$ is a $\mathcal { P } _ { \lambda } ^ { 1 }$ -transfer exponent, then $\gamma + d _ { \mu }$ is an integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent for any ensemble size $M \geq 1$ , where $d _ { \mu }$ is the dimension of supp(µ). However, this relationship is loose; there are source-target pairs with integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponents less than $\gamma + d _ { \mu }$ as we will show in Section 5. A similar relationship holds between the transfer exponent and integrated transfer exponent and is why the latter is seen as a refinement of the former [10, 23].

Proposition 2. Suppose $\gamma > 0$ is a $\mathcal { P } _ { \lambda } ^ { 1 }$ -transfer exponent for source-target pair $( \nu , \mu )$ . Then $\gamma + d _ { \mu }$ is an integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent for any ensemble size $M \geq 1$

Proof. Due to lemma 1 and proposition 1,

$$
\mathbb { E } \left[ \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] \leq \mathbb { E } \left[ \frac { 1 } { \nu \left( Z _ { X } ^ { 1 , \lambda } \right) } \right] = \mathbb { E } \left[ \sum _ { \substack { C \in \mathcal { P } _ { \lambda } ^ { ( 1 ) } : } } \frac { \mu \left( C \right) } { \nu \left( C \right) } \right] ,
$$

and we need only show that $\gamma + d _ { \mu }$ is an aggregate $\mathcal { P } _ { \lambda } ^ { 1 }$ -transfer exponent.

For each $C \in \mathcal { P } _ { \lambda } ^ { ( 1 ) } : C \cap \operatorname { s u p p } ( \mu ) \neq \emptyset$ , there is some $x \in \operatorname { s u p p } ( \mu )$ such that $C = Z _ { x } ^ { 1 , \lambda }$ . Thus, we can apply the definition of the $\mathcal { P } _ { \lambda } ^ { 1 }$ -transfer exponent to bound

$$
\mathbb { E } \left[ \sum _ { \substack { C \cap \operatorname { s u p p } ( \mu ) \neq \emptyset } } \frac { \mu \left( C \right) } { \nu \left( C \right) } \right] \lesssim \mathbb { E } \left[ \sum _ { \substack { C \cap \operatorname { s u p p } ( \mu ) \neq \emptyset } } \lambda ^ { \gamma } \right]
$$

The last line above is a consequence of Lemma 4 in [22], and the $\lesssim$ elides lower-degree terms in λ and coeficients that depend on $\gamma , \Pi$ , and $d _ { \mu }$ but not λ. □

## 4.3 STIT Kernel Estimator Convergence Rate

We next specialize our risk bound in Theorem 1 to the STIT kernel estimator and use the integrated $\mathcal { P } _ { \lambda } ^ { M }$ transfer exponent to control the variance as λ grows with n. First, we obtain a bound for the bias of a STIT kernel estimator. This bound grows logarithmically in the ensemble size M, so it is suitable for moderate-tolarge ensembles but not extremely large ones. The general STIT kernel estimator’s bias bound depends on d through the associated zonoid Π via a constant $c _ { \beta , \Pi }$ . To shed light on this dependence on $d ,$ we separately derive a bias bound for the Mondrian kernel estimator, using the explicit distribution of the zero cell of the Mondrian partition [19].

Lemma 2. There exists $c _ { \beta , \Pi } > 0$ depending on $\beta$ and Π such that for any $x \in \operatorname { s u p p } ( \mu )$ ，

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \dim ( Z _ { x } ^ { m , \lambda } ) ^ { 2 \beta } \right] \lesssim \frac { c _ { \beta , \Pi } } { \lambda ^ { 2 \beta } } \cdot \left( 1 \vee \log ^ { 2 \beta } ( d M ) \right) .
$$

For Mondrian partitions, with $M \geq 2$

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { x } ^ { m , \lambda } ) ^ { 2 \beta } \right] \lesssim \frac { d ^ { \beta } } { \lambda ^ { 2 \beta } } \log ^ { 2 \beta } M .
$$

In both cases, $\lesssim$ elides a universal constant.

Proofs of the two statements in lemma 2 are provided in Section 8 and Section 11. Together, corollary 1 and lemma 2, bound the integrated risk of a STIT kernel estimator.

Theorem 2. Adopt Hypotheses 4 to 7. Then,

$$
\begin{array} { r l } & { \mathbb { E } [ ( \hat { f } _ { \mathcal { P } _ { \lambda } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } ] \lesssim \frac { L ^ { 2 } c _ { \beta , \Pi } } { \lambda ^ { 2 \beta } } \cdot \Big ( 1 \vee \log ^ { 2 \beta } ( d M ) \Big ) } \\ & { \quad \quad \quad \quad + \frac { \sigma ^ { 2 } + \| f ^ { * } \| _ { \infty } ^ { 2 } } { n } \cdot \mathbb { E } \left[ n \wedge \frac { 1 } { \frac { n _ { \nu } } { n } \cdot \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) + \frac { n _ { \mu } } { n } \cdot \mu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] . } \end{array}
$$

Here, $\lesssim$ elides a universal constant.

Now that both the bias and variance are controlled by expressions involving the lifetime parameter λ, we can tune λ to balance the bias and variance, obtaining a convergence rate. Compared to the convergence rate of the STIT tree without covariate shift (Corollary 7 in [22]), this rate has the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent in the role of d.

Corollary 2. $\operatorname { I f } \gamma \geq d$ is an integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent for the source-target pair $\begin{array} { r } { \left( \left( \frac { n _ { \nu } } { n } \cdot \nu + \frac { n _ { \mu } } { n } \cdot \mu \right) , \mu \right) } \end{array}$ then tuning $\lambda \asymp n ^ { \frac { 1 } { 2 \beta + \gamma } }$ causes $\hat { f } _ { \mathcal { P } _ { \lambda } , M }$ to achieve the following convergence rate in n:

$$
{  { \mathbb E } } [ ( \hat { f } _ { \mathcal P _ { \lambda } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } ] \lesssim \frac { 1 } { \lambda ^ { 2 \beta } } + \frac { \lambda ^ { \gamma } } { n } \asymp n ^ { - \frac { 2 \beta } { 2 \beta + \gamma } } .
$$

Here, $\lesssim$ elides constants depending on $\gamma , L , \beta , d , \Pi , \boldsymbol { M } , \sigma ^ { 2 }$ , and $\| f ^ { * } \| _ { \infty }$ but not λ.

![](images/f71d6db809b69980b35ae8bb29faa7385d6abc507bb15b42f751ef6fd9ba4c36.jpg)  
Figure 3: The STIT kernel estimator is used to fit an exponential regression function under covariate shift of severity $\alpha = 2$ . The covariate shift results in insuficient data near 0, so individual trees $( { \mathrm { i . e . } }$ , single learners) have high variance. Ensembling smooths the estimate, which improves its performance, especially near 0.

This convergence rate is minimax-optimal for $n _ { \nu } = n$ and $d = 1$ . A matching lower bound is provided in Theorem 2 of [23], wherein the authors say the proof can be extended to $d > 1$

Remark 3. In our convergence rates, the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent $\gamma$ takes the role of $d .$ If $\gamma <$ $d ,$ that would suggest the covariate-shifted training data confers rate improvements over unshifted data. Intuitively, this could occur if the source ν were closer to the uniform distribution than the target $\mu .$ For further discussion of the integrated transfer exponent being strictly less than $d ,$ see Sections 2.3 and 2.4 in [23].

## 5 STIT Kernel Estimator on Example Covariate Shift

In this section, we demonstrate that ensemble size can significantly afect the integrated $\mathcal { P } _ { \lambda } ^ { M } .$ -transfer exponent. We study a class of covariate shifts that have previously been used to demonstrate properties of covariate shift discrepancies, such as the separation between the transfer exponent and integrated transfer exponent [23, 10]. For covariate shifts in this class, $\mu$ is the uniform measure over the unit cube and ν has a polynomial density. We restrict our focus to $d = 1$ , where the only STIT tessellation is the Mondrian and the distribution of $\textstyle \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda }$ can be expressed directly. As in other works, we suppose all training data is ν-distributed, so $n _ { \nu } = n \ [ 2 3 , 1 0 ]$ . This covariate shift, and the performance of the STIT kernel estimator on ${ \mathrm { i t } } ,$ are illustrated in fig. 3.

Hypothesis 8 (Polynomial Density). $\mu = \mathrm { U n i f } ( [ 0 , 1 ] )$ , and there is an integer $\alpha > 1$ such that ν has density $( \alpha + 1 ) x ^ { \alpha } \mathbf { 1 } _ { x \in [ 0 , 1 ] }$

Remark 4. Under Hypothesis 8, the infimum transfer exponent is $\alpha + 1$ , while the infimum integrated transfer exponent is $\alpha \ [ 2 3 , 1 0 ]$

We show that the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer discrepancy for the Mondrian kernel estimator is infinite for $M \leq \alpha$ yet the Mondrian kernel estimator achieves the minimax-optimal convergence rates for $M \geq \alpha + 2$

Proposition 3. Adopt Hypothesis 8. If $M \leq \alpha$ , the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer discrepancy is infinite.

The proof is provided in Section 13. It uses the distribution of the zero cell of the Mondrian tessellation [19] to relate the integrated $\mathcal { P } _ { \lambda } ^ { M } .$ -transfer discrepancy to an inverse moment of a Gamma-distributed random variable with shape parameter M. When the shape parameter is small, the Gamma random variable has very few finite inverse moments.

A larger ensemble size allows us to better control the variance of the estimator. While the stochastic nature of the partition complicates eforts to bound the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent itself, we can nevertheless use theorem 2 and corollary 2 to guarantee the Mondrian kernel estimator with suficient M achieves minimax-optimal convergence rates.

Proposition 4. Adopt Hypothesis 8. If $M \geq \alpha + 2$ , the Mondrian kernel estimator achieves the minimaxoptimal convergence rate. Namely,

$$
\mathbb { E } [ ( \hat { f } _ { \mathcal { P } _ { \lambda } , M } ( X ) - f ^ { * } ( X ) ) ^ { 2 } ] \lesssim \frac { \log ^ { 2 \beta } M } { \lambda ^ { 2 \beta } } + \frac { \lambda ^ { \alpha } } { n } \left( \frac { M } { \left( 1 - \frac { \alpha + 1 } { M } \right) ^ { \alpha + 1 } } + \frac { 1 } { \left( 1 - \frac { \alpha } { M } \right) ^ { \alpha } } \right) + \frac { M ^ { 2 } e ^ { - \lambda } } { n \lambda } + M ^ { 2 } e ^ { - \frac { \lambda } { 2 } } ,
$$

where $\lesssim$ elides coeficients that depend on $L , \sigma ^ { 2 }$ , and $\| f ^ { * } \| _ { \infty }$

We tune $\lambda \asymp n ^ { - \frac { 1 } { 2 \beta + \alpha } }$ and, so long as M is not too large $( \mathrm { i . e . } ,$ , no more than polynomial in λ), then for suficient n, the risk has the minimax-optimal [23] rate of $n ^ { - \frac { 2 \beta } { 2 \beta + \alpha } }$

The proof is provided in Section 9. In efect, we prove in proposition 4 that for suficient ensemble size, the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent is α with high probability. The truncation at n in E $\left[ n \land \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right]$ allows us to control the risk due to the low-probability event. This principle is generalized in the following corollary to theorem 2.

Corollary 3. Adopt Hypotheses 4 to 7. Suppose there is a number $\gamma \geq d$ and an event $E$ in the joint probability space of $X \sim \mu$ and $\left\{ \mathcal { P } _ { \lambda } ^ { ( m ) } \right\} _ { m }$ such that

$$
\mathbb { E } \left[ \frac { \mathbf { 1 } _ { E } } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] \lesssim \lambda ^ { \gamma } ,
$$

where $\lesssim$ elides coeficients depending on $\gamma$ and M but not λ. Also, suppose E has probability at least $\begin{array} { r } { 1 - \exp \left( - \frac { c \lambda } { \log M } \right) } \end{array}$ for some universal constant $c > 0$ . Finally, suppose λ is tuned to $\lambda \asymp n ^ { \frac { 1 } { 2 \beta + \gamma } }$ , and M is tuned to no greater than a polynomial in λ. Then, the conclusion of corollary 2 holds with $\gamma$ in the role of the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent. Namely, the STIT kernel estimator $\hat { f } _ { \mathcal { P } _ { \lambda } , M }$ has the following integrated risk bound:

$$
\mathbb { E } [ ( \hat { f } _ { \mathcal { P } _ { \lambda } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } ] \lesssim \frac { 1 } { \lambda ^ { 2 \beta } } + \frac { \lambda ^ { \gamma } } { n } + \exp { ( - c \lambda ) } \asymp n ^ { - \frac { 2 \beta } { 2 \beta + \gamma } } .
$$

Here, $\lesssim$ elides constants depending on γ, L, β, d, Π, M, $\sigma ^ { 2 }$ , and $\| f ^ { * } \| _ { \infty }$ but not λ.

## 6 Conclusion

In this paper, we prove theoretical guarantees for partition-based kernel estimators under covariate shift, demonstrating the beneficial efect of ensembling. We begin by modifying previously introduced covariate shift measures [16, 23, 10] to control the variance of partition-based kernel estimators. Our pointwise risk bound applies to general partitions. We then derive an integrated risk bound specifically for STIT tessellations under covariate shift, whose complexity is prescribed by the lifetime parameter λ. Lastly, we analyze a class of well-studied severe covariate shifts, revealing that ensemble size must increase with shift severity to control the partition-based discrepancy measure (with suficiently high probability).

Our conclusions about the importance of ensembling parallel recent insights about Mondrian forests, where ensembling counteracts the small-cell phenomenon and enables statistical inference [8]. We find that ensembling likewise helps under covariate shift to control variance, especially in the STIT kernel estimator.

Empirical evidence suggests ensembles help to transfer learning from one distribution to another, even under more general distribution shifts [9]. To illuminate and explain this phenomenon theoretically, we envision extending and evaluating discrepancy measures for posterior drift [7, 6] to ensemble models. Recent work suggests ensembling might even control variance well enough to explain the success of interpolating models [33], another fruitful avenue for continued theoretical work to illuminate and explain the empirical success of ensembles.

## 7 Proof of Theorem 1

Proof of theorem 1. Conditioning on the training features $\{ X _ { i } \} _ { i = 1 } ^ { n }$ and the partitions $\left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } .$ , we compute the conditional expectation of the kernel estimator:

$$
\begin{array} { r l } & { \bar { f } _ { \mathcal { P } , M } ( x ) : = \mathbb { E } \left[ \left. \hat { f } _ { \mathcal { P } , M } ( x ) \right| \left\{ X _ { i } \right\} _ { i = 1 } ^ { n } , \left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } \right] } \\ & { \qquad = \mathbf { 1 } _ { \exists i , m : X _ { i } \in Z _ { x } ^ { m } } \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left( \sum _ { i = 1 } ^ { n _ { \nu } } f _ { \nu } ^ { * } ( X _ { i } ) \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { m } } + \sum _ { i = n _ { \nu } + 1 } ^ { n } f _ { \mu } ^ { * } ( X _ { i } ) \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { m } } \right) } { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { m } } } .  \end{array}
$$

Conditioned on the training features $\{ X _ { i } \} _ { i = 1 } ^ { n }$ and the partitions $\left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } .$ , we separate out the case in which there is no local data and perform a bias-variance decomposition:

$$
\begin{array} { r l } { \mathbb { E } [ ( \hat { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } ] = \mathbb { E } [ ( ( \hat { f } _ { \mathcal { P } , M } ( x ) - \bar { f } _ { \mathcal { P } , M } ( x ) ) + ( \bar { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) ) ) ^ { 2 } \mathbf { 1 } _ { \exists i , m : X _ { i } \in Z _ { x } ^ { m } }  } \\ & { \qquad + \mathbb { E } [ ( \hat { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } \mathbf { 1 } _ { \bar { \mathcal { H } } , m : X _ { i } \in Z _ { x } ^ { m } } ] } \\ & { = \mathbb { E } [ ( \hat { f } _ { \mathcal { P } , M } ( x ) - \bar { f } _ { \mathcal { P } , M } ( x ) ) ^ { 2 } \mathbf { 1 } _ { \bar { \mathcal { Q } } , m : X _ { i } \in Z _ { x } ^ { m } } ] } \\ & { \qquad + \mathbb { E } [ ( \bar { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } \mathbf { 1 } _ { \bar { \mathcal { Q } } , m : X _ { i } \in Z _ { x } ^ { m } } ] } \\ & { \qquad + \mathbb { E } [ ( f ^ { * } ( x ) ) ^ { 2 } \cdot \mathbb { P } [ \overline { { \hat { \mu } _ { i } } } , m : X _ { i } \in Z _ { x } ^ { m }  ] \{ \mathcal { P } ^ { ( m ) } \} _ { m = 1 } ^ { M } ] ] . } \end{array}\tag{3}
$$

The cross-terms cancel due to a conditioning argument, which hinges on the definition of $\bar { f } _ { \mathcal { P } , M } ( x )$ and the observation that neither it nor $f ^ { * } ( x )$ depend on $\{ \xi _ { i } \} _ { ; }$ . Namely,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left( \hat { f } _ { \mathcal { P } , M } ( x ) - \bar { f } _ { \mathcal { P } , M } ( x ) \right) \left( \bar { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) \right) \mathbf { 1 } _ { \exists i , m : X _ { i } \in Z _ { x } ^ { m } } \right] } \\ & { \quad = \mathbb { E } \left[ \mathbb { E } \left[ \left( \hat { f } _ { \mathcal { P } , M } ( x ) - \bar { f } _ { \mathcal { P } , M } ( x ) \right) \left( \bar { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) \right) \mathbf { 1 } _ { \exists i , m : X _ { i } \in Z _ { x } ^ { m } } \left. \left\{ X _ { i } \right\} _ { i = 1 } ^ { n } , \left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } \right] \right] } \\ & { \quad = \mathbb { E } \left[ \left( \mathbb { E } \left[ \hat { f } _ { \mathcal { P } , M } ( x ) \right. \left\{ X _ { i } \right\} _ { i = 1 } ^ { n } , \left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } \right] - \bar { f } _ { \mathcal { P } , M } ( x ) \right) \left( \bar { f } _ { \mathcal { P } , M } ( x ) - f ^ { * } ( x ) \right) \mathbf { 1 } _ { \exists i , m : X _ { i } \in Z _ { x } ^ { m } } \right] = 0 . } \end{array}
$$

We bound the pointwise bias by a convex combination of the diameters of the cells $Z _ { x } ^ { m }$ , provided at least one of these cells contains training points. Using Jensen’s inequality (over a finite average) and H¨older continuity,

$$
\begin{array} { r l } & { ( \hat { f } _ { P , M } ( x ) - f ^ { * } ( x ) ) ^ { 2 } \mathbf { 1 } _ { 3 , i , m ; N , i \in \mathcal { G } _ { \mathrm { s } } ^ { m } } = ( \frac { \sum _ { i = 1 } ^ { n } ( f ^ { * } ( X _ { i } ) - f ^ { * } ( x ) ) ( \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } ) } { \sum _ { i = 1 } ^ { n } ( \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } ) } ) ^ { 2 } } \\ & { \leq \frac { \sum _ { i = 1 } ^ { n } ( f ^ { * } ( X _ { i } ) - f ^ { * } ( x ) ) ^ { 2 } ( \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } ) } { \sum _ { i = 1 } ^ { n } ( \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } ) } } \\ & { \leq \frac { \sum _ { i = 1 } ^ { n } L ^ { 2 } \| X _ { i } - x \| ^ { 2 / k } ( \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } ) } { \sum _ { i = 1 } ^ { M } ( \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } ) } } \\ &  = L ^ { 2 } \cdot \frac { \displaystyle M }  \sum _ { m = 1 } ^ { M } ( \frac { \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in \mathcal { G } _ { \mathrm { s } } ^ { m } } }  \displaystyle \sum _ { m = 1 } ^ { M } ( \sum _ \end{array}\tag{4}
$$

In the convex combination, the term diam $( Z _ { x } ^ { m } \cap \mathrm { s u p p } ( \nu ) ) ^ { 2 \beta }$ is weighted proportionally to the number of data points that fall in $Z _ { x } ^ { m }$ . The weights, therefore, are random variables depending on the training data and the partitions. Comparatively more weight is given to terms diam $( Z _ { x } ^ { m } \cap \mathop { \mathrm { s u p p } } ( \nu ) ) ^ { 2 \beta }$ for which $Z _ { x } ^ { m }$ contains more data points, and we would expect to see more data points in cells with larger diameters if, for instance, ν were the uniform measure. Since a convex combination is no greater than the maximum quantity being combined,

$$
\sum _ { m = 1 } ^ { M } \left( \frac { \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { m } } } { \sum _ { m ^ { \prime } = 1 } ^ { M } \left( \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { m ^ { \prime } } } \right) } \right) \dim ( Z _ { x } ^ { m } \cap \operatorname { s u p p } ( \nu ) ) ^ { 2 \beta } \leq \frac { M } { m = 1 } \dim ( Z _ { x } ^ { m } \cap \operatorname { s u p p } ( \nu ) ) ^ { 2 \beta } .\tag{5}
$$

We continue the analysis of the kernel estimator’s risk, as decomposed in eq. (3), by bounding its conditional variance. Using the bound on the noise in the training labels,

$$
\begin{array} { r l } & { \mathbb { E } [ ( \hat { f } _ { \mathcal { P } , \mathcal { M } } ( x ) - \bar { f } _ { \mathcal { P } , \mathcal { M } } ( x ) ) ^ { 2 } \mathbf { 1 } _ { \mathcal { I } _ { 1 } , \mathcal { m } ; \mathcal { N } , \zeta \in \mathbb { Z } _ { \mathcal { m } } ^ { m } } \{ \{ X _ { i } \} _ { i = 1 } ^ { n } , \{ \mathcal { P } ( m ) \} _ { m = 1 } ^ { \mathcal { M } } ] } \\ & { \quad = \mathbf { 1 } _ { \mathfrak { I } _ { i } , m ; X _ { i } \in \mathbb { Z } _ { \mathcal { C } } ^ { m } } \cdot \mathrm { V a r } [ \hat { f } _ { \mathcal { P } , \mathcal { M } } ( x ) \bigg | \{ X _ { i } \} _ { i = 1 } ^ { n } , \{ \mathcal { P } ^ { ( m ) } \} _ { m = 1 } ^ { \mathcal { M } } ] } \\ & { \quad \quad = \mathbf { 1 } _ { \mathfrak { I } _ { i } , m ; X _ { i } \in \mathbb { Z } _ { \mathcal { C } } ^ { m } } \cdot \quad \quad \quad \quad \quad \quad \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { n } \mathrm { V a r } [ \xi _ { i } | X _ { i } ] \mathbf { 1 } _ { X _ { i } \in \mathbb { Z } _ { \mathcal { m } } ^ { m } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \leq \mathbf { 1 } _ { \mathfrak { I } _ { i } , m ; X _ { i } \in \mathbb { Z } _ { \mathcal { C } } ^ { m } } \cdot \frac { 1 }  \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in \mathbb { Z } _ { \mathcal { X } } ^ { m } } + \sum _ { m = 1 } ^ { M } \sum _ { i = m _ { n } + 1 } \end{array}\tag{6}
$$

The final inequality is based on the geometric observation that $\begin{array} { r } { \sum _ { m = 1 } ^ { M } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { m } } \geq \mathbf { 1 } _ { X _ { i } \in \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m } } } \end{array}$ . This step is useful in the analysis because it allows us to analyze the denominator as the sum of independent binomial random variables, conditioned on the partitions. Moreover, the variance is intuitively controlled by averaging many independent data points, and while this bound might be loose for large $M$ , it preserves the overal number of independent data points used to form the local average $\hat { f } _ { \mathcal { P } , M } ( x )$ . For notational brevity, let $\textstyle Z _ { x } ^ { [ M ] } : = \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m }$

Due to Hypothesis 2 and independence of $\{ X _ { i } \} _ { i = 1 } ^ { n }$ , the denominator in eq. (6) is the sum of two inde-

pendent binomial random variables, conditioned on the partitions. By lemma 3,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathbf { 1 } _ { \exists i , m : X _ { i } \in Z _ { x } ^ { m } } \frac { \sigma ^ { 2 } } { \sum _ { i = 1 } ^ { n _ { \nu } } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { [ M ] } } + \sum _ { i = n _ { \nu } + 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { [ M ] } } } \Bigg | \left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } \right] } \\ & { \leq \sigma ^ { 2 } \cdot \mathbb { E } \left[ 1 \land \frac { 1 } { \sum _ { i = 1 } ^ { n _ { \nu } } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { [ M ] } } + \sum _ { i = n _ { \nu } + 1 } ^ { n } \mathbf { 1 } _ { X _ { i } \in Z _ { x } ^ { [ M ] } } } \Bigg | \left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } \right] } \\ & { \leq 4 \sigma ^ { 2 } \land \frac { 1 8 \sigma ^ { 2 } } { n _ { \nu } \cdot \nu \left( Z _ { x } ^ { [ M ] } \right) + n _ { \mu } \cdot \mu \left( Z _ { x } ^ { [ M ] } \right) } . } \end{array}\tag{7}
$$

So far, we have bounded the bias and variance in the case that for at least one partition, indexed $m$ there is at least one data point, indexed i, such that $X _ { i } \in Z _ { x } ^ { m }$ , so that $\hat { f } _ { \mathcal { P } , M } ( x )$ is a local average of training labels. If not, then the values of $\hat { f } _ { \mathcal { P } , M } ( x )$ and $\bar { f } _ { \mathcal { P } , M } ( x )$ are identically zero, so the squared error is $( f ^ { * } ( x ) ) ^ { 2 }$ Conditioned on the partitions, by independence of $\{ X _ { i } \} _ { i }$ , the probability of no data falling in the cells is

$$
\begin{array} { r l r } & { } & { ( f ^ { * } ( x ) ) ^ { 2 } \cdot \mathbb { P } \left[ \left. \vec { \mu } i , m : X _ { i } \in Z _ { x } ^ { m } \right| \left\{ \mathcal { P } ^ { ( m ) } \right\} _ { m = 1 } ^ { M } \right] = ( f ^ { * } ( x ) ) ^ { 2 } \left( 1 - \nu \left( Z _ { x } ^ { [ M ] } \right) \right) ^ { n _ { \nu } } \left( 1 - \mu \left( Z _ { x } ^ { [ M ] } \right) \right) ^ { n _ { \mu } } } \\ & { } & { \leq ( f ^ { * } ( x ) ) ^ { 2 } \wedge \displaystyle \frac { ( f ^ { * } ( x ) ) ^ { 2 } } { n _ { \nu } \cdot \nu \left( Z _ { x } ^ { [ M ] } \right) + n _ { \mu } \cdot \mu \left( Z _ { x } ^ { [ M ] } \right) } . } \end{array}\tag{8}
$$

We used the identities $\begin{array} { r } { ( 1 - p _ { 1 } ) ^ { n _ { 1 } } ( 1 - p _ { 2 } ) ^ { n _ { 2 } } \leq \exp ( - ( n _ { 1 } p _ { 1 } + n _ { 2 } p _ { 2 } ) ) \leq \frac { 1 } { n _ { 1 } p _ { 1 } + n _ { 2 } p _ { 2 } | } } \end{array}$ , which hold for $p _ { 1 } , p _ { 2 } \in ( 0 , 1 )$ and integers $n _ { 1 } , n _ { 2 } \geq 0$ . See footnote 3 in [23] for similar usage. We also bounded the probability above by 1. We absorb eq. (8) into the variance term.

Finally, combining eq. (3), eq. (4), eq. (5), eq. (7), and eq. (8),

$$
\begin{array} { r l } { 4 \left. \hat { \mathcal { L } } _ { f , \psi _ { \lambda } } ( s ) - f ^ { * } ( s ) \right. ^ { 2 } } & { = - \left. \hat { \mathcal { L } } _ { f } ^ { \dagger } \mathcal { R } _ { f , \psi _ { \lambda } } ( s ) - f ^ { * } ( s ) \right. ^ { 2 } \frac { 1 } { s } \frac { \left( \hat { \mathcal { L } } _ { f } ^ { ( 0 ) } - f ^ { * } ( s ) \right) ^ { 2 } } { s } + \frac { \left( \hat { \mathcal { L } } _ { f } ^ { ( 0 ) } - f ^ { * } ( s ) \right) ^ { 2 } } { s } \Bigg | \left( \mathcal { R } _ { f } ^ { ( 0 ) } + f ^ { * } ( s ) \right) \Bigg | _ { s = 0 } ^ { s } } \\ & { \quad + \mathbb { E } \Bigg [ \mathbb { E } \Bigg [ \left( \hat { \mathcal { L } } _ { f } ^ { \dagger } \mathcal { R } _ { f } ( s ) - \left. \hat { \mathcal { L } } _ { f } ^ { \dagger } \mathcal { R } _ { f } ( s ) \right. \right) ^ { 2 } \Bigg ] \mathbb { E } _ { s = 0 } \cdot \mathbb { E } _ { s = 0 } ^ { \lambda _ { 0 } } \Bigg [ \left( \mathcal { R } _ { f } ^ { ( 0 ) } \right) \Bigg | _ { s = 0 } \Bigg ] \Bigg | \int _ { s } ^ { s } \Bigg [ \mathcal { R } _ { f } ^ { ( 0 ) } \Bigg | _ { s = 0 } ^ { s } \Bigg ] \Bigg | _ { s = 0 } ^ { s } \Bigg ] \Bigg | _ { s = 0 } ^ { s } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}
$$

## 8 Proof of Lemma 2

Proof of lemma 2. The STIT tessellation has stationarity and the scaling property, so $Z _ { x } ^ { m , \lambda }$ is distributed as $\begin{array} { r } { \frac { 1 } { \lambda } ( Z _ { 0 } ^ { \ ' m , 1 } ) + x } \end{array}$ . Translation does not change a convex body’s diameter. We use these observations to separate

the bias into factors depending on λ and M:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { x } ^ { m , \lambda } ) ^ { 2 \beta } \right] = \mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } \left( \frac { Z _ { 0 } ^ { m , 1 } } { \lambda } \right) ^ { 2 \beta } \right] = \frac { 1 } { \lambda ^ { 2 \beta } } \cdot \mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } ) ^ { 2 \beta } \right] .\tag{9}
$$

To clarify the dependence of E $\left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } ) ^ { 2 \beta } \right]$ on M, we obtain a moment bound for $\mathrm { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } )$ via a previously obtained tail bound on diam $( Z _ { 0 } ^ { m , 1 } )$

We note that we can apply known results about the distribution of the zero cell of the Poisson hyperplane tessellation because the zero cell of the STIT tessellation has the same distribution as that of the Poisson hyperplane tessellation with corresponding parameters, as argued in [22] by citing Corollary 1 of [28] and Theorem 10.4.1 of [27]. We work from the proof of Theorem 6.1.2 in [13], which is provided in the text after the statement of Theorem 6.3.1, a more general result. The authors identify a constant $c _ { \Pi } > 0$ , which depends on the zonoid Π, such that, for any m,

$$
\mathbb { P } \left[ \frac { 1 } { 2 } \dim ( Z _ { 0 } ^ { m , 1 } ) \geq t \right] \leq 2 d \cdot \exp ( - c _ { \Pi } t ) .\tag{10}
$$

More precisely, Theorem 6.1.2 provides a right-sided tail bound for the radius of a sphere containing $Z _ { 0 } ^ { m , 1 }$ hence our inclusion of the $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ in the line above.

For $d M \geq 2 .$ , we follow the proof technique of Proposition 2.7.6 in [32], to derive a tail bound for $\mathrm { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } )$ :

$$
\mathbb { P } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } ) \geq t \right] \leq \sum _ { m = 1 } ^ { M } \mathbb { P } \left[ \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } ) \geq t \right] \leq 2 d M e ^ { - \epsilon \mathrm { \scriptscriptstyle m } \cdot \frac { t } { 2 } } \leq 2 \exp \left( - \frac { c _ { \mathrm { \scriptscriptstyle m } } t } { 6 \log ( d M ) } \right) .\tag{11}
$$

The first inequality in eq. (11) is a union bound, and the second is an application of eq. (10). The third inequality is a result of a numerical argument given in Section 12. From eq. (11) (and eq. (10) in the case $d = M = 1 )$ , we obtain the moment bound

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } ) ^ { 2 \beta } \right] \leq \left( \mathbb { E } \left[ \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } \left( Z _ { 0 } ^ { m , 1 } \right) \right) ^ { 2 } \right] \right) ^ { \beta } \lesssim \frac { 1 } { c _ { \mathtt { M } } ^ { 2 \beta } } \cdot \left( 1 \vee \log ^ { 2 \beta } ( d M ) \right) .\tag{12}
$$

The $\lesssim$ elides a universal constant. Details for the derivation of eq. (12) from eq. (11) and eq. (10) are provided in Section 12. The result for a general STIT tessellation is obtained by substituting eq. (12) into eq. (9). For the statement of the lemma, we refer to $c _ { \Pi } ^ { - 2 \beta }$ as $^ { c _ { \beta , \Pi } }$

The proof for the Mondrian case is provided in Section 11.

## 9 Proof of Proposition 4

Proof of proposition $\it 4 .$ Per corollary 2, it sufices to prove that E $\begin{array} { r } { \bigg [ n \wedge \frac { 1 } { \nu \big ( \bigcup _ { m = 1 } ^ { M } Z _ { x } ^ { m , \lambda } \big ) } \bigg ] } \end{array}$ is on the order of $\lambda ^ { \alpha }$ As in eq. (26) and eq. (27), we begin by rewriting the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer discrepancy in terms unit exponential random variables $\{ T _ { m } ^ { + } , T _ { m } ^ { - } \} _ { m = 1 } ^ { M }$ . From the distribution of the cell of the Mondrian [19],

$$
\mathbb { E } \left[ n \wedge \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] = \mathbb { E } \left[ n \wedge \frac { 1 } { \nu \left( \left[ - \frac { 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } + X , \frac { 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } + X \right] \right) } \right] .\tag{13}
$$

Inspired by the proof technique in Appendix C of [10], we condition on $\{ T _ { m } ^ { + } , T _ { m } ^ { - } \} _ { m = 1 } ^ { M }$ and bifurcate the support of X into a region closer to the ν singularity at 0 and a region further away. To account for the boundaries of the support, we will compute the measure ν of only a subset of each cell. Unlike their proof, however, the extents of the cell on either side of X are random and independent, so there is a nonzero probability that X falls into neither region. We will bound this probability and show that, even when multiplied by the truncation value $n ,$ , suitable values of M and λ can ensure this term quickly approaches zero. For compactness, notate $T _ { [ M ] } ^ { - } : = \mathrm { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - }$ and $T _ { [ M ] } ^ { + } : = \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + }$ . Continuing to analyze eq. (13),

$$
\begin{array} { r l } & { \mathbb { E } [ \pi \wedge \frac { 1 } { w ( [ - \frac { 1 } { \lambda } T _ { | M | } ^ { - 1 } + X _ { 1 } ^ { 1 } T _ { | M | } ^ { + } + X _ { 1 } ^ { 1 } ] ) } ] } \\ & { \leq \mathbb { E } _ { \{ T _ { * } ^ { + } , T _ { * } ^ { - } \} \sim [ \mathbb { E } _ { X } ] } [ \frac { 1 } { w } [ \frac { 1 } { \gamma } x _ { ( [ - \frac { 1 } { \lambda } T _ { | M | } ^ { - 1 } ] \wedge \frac { 1 } { \lambda } T _ { | M | } ^ { + } + X _ { 1 } ^ { - 1 } ] \wedge \frac { 1 } { \lambda } } { \lambda ^ { 1 } ( [ - \frac { 1 } { \lambda } T _ { | M | } ^ { - 1 } ] \wedge \frac { 1 } { \lambda } T _ { | M | } ^ { + } + X _ { 1 } ^ { 1 } ) } + \frac { 1 } { \gamma ( [ - \frac { 1 } { \lambda } T _ { | M | } ^ { - 1 } + X _ { 1 } ^ { 1 } T _ { | M | } ^ { + } + X _ { 1 } ^ { 1 } ] ) } ] ] } \\ & { \qquad + n \cdot \mathbb { P } [ 1 - \frac { 1 } { \lambda } T _ { | M | } ^ { + } < X \leq ( \frac { 2 } { \lambda } T _ { | M | } ^ { - 1 } ) \wedge 1 ] } \\ &  \leq \mathbb { E } _ { \{ T _ { * } ^ { - } , T _ { * } ^ { - } \} \sim [ \mathbb { E } _ { X } ] } [ \frac { 1 } { w } [ \frac { 1 } { \gamma } x _ { ( [ \frac { - 1 } { \lambda } T _ { | M | } ^ { - 1 } ] \wedge \frac { 1 } { \lambda } T _ { | M | } ^ { + } ) }  ( [ X _ { 1 } , \frac { 1 } { \lambda } T _ { | M | } ^ { + } + X _ { 1 } ^ { - 1 } ] \end{array}\tag{14}
$$

The second term in eq. (14) encompasses values of X that are far enough from 0 that we can lower-bound the density of ν on the interval $\left[ - \textstyle { \frac { 1 } { \lambda } } T _ { [ M ] } ^ { - } + X , X \right]$ , which is a subset of the cell $\left[ - { \textstyle \frac { 1 } { \lambda } } T _ { [ M ] } ^ { - } + X , { \textstyle \frac { 1 } { \lambda } } T _ { [ M ] } ^ { + } + X \right]$ and of $\operatorname* { s u p p } ( \nu ) = [ 0 , 1 ]$ . For $x > \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } ,$ , any $x ^ { \prime } \in \left[ - \textstyle { \frac { 1 } { \lambda } } T _ { [ M ] } ^ { - } + x , x \right]$ satisfies

$$
x ^ { \prime } \geq x - \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } = x \left( 1 - \frac { 1 } { x } \cdot \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } \right) \geq x \left( 1 - \frac { 1 } { 2 } \right) = \frac { x } { 2 } ,
$$

where the last inequality holds because $x > \frac { 2 } { \lambda } T _ { [ M ] } ^ { - }$ can be rearranged to $\begin{array} { r } { \frac { 1 } { 2 } > \frac { 1 } { x } \cdot \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } } \end{array}$ . Thus, the density of $\nu$ is at least $\textstyle { \left( \alpha + 1 \right) \left( { \frac { x } { 2 } } \right) ^ { \alpha } }$ at any such $x ^ { \prime }$ . In summary, with X in place of $x ,$

$$
X > \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \implies \nu \left( \left[ - \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } + X , X \right] \right) \geq \frac { 1 } { \alpha + 1 } \left( \frac { X } { 2 } \right) ^ { \alpha } \cdot \frac { 1 } { \lambda } T _ { [ M ] } ^ { - }
$$

We next integrate with respect to $X ,$ deriving an expression for the inner expectation of the second term in $\mathrm { e q . ~ } ( 1 4 )$ . Using the assumption that $\alpha - 1$ is strictly positive,

$$
\begin{array} { r l } & { \mathbb { E } _ { x } \left[ \frac { \mathbf { 1 } _ { X > \left( \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \right) \wedge 1 } } { \nu \left( \left[ - \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } + \lambda , X \right] \right) } \right] = \int _ { \left( \frac { 2 } { \lambda ^ { 2 } \Gamma _ { [ M ] } ^ { - } } \right) \wedge 1 } ^ { 1 } \frac { d x } { \nu \left( \left[ - \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } + x , x \right] \right) } } \\ & { \qquad \leq \int _ { \left( \frac { 2 } { \lambda ^ { 2 } \Gamma _ { [ M ] } ^ { - } } \right) \wedge 1 } \frac { d x } { ( \alpha + 1 ) \left( \frac { 2 } { \lambda ^ { 2 } } \right) ^ { \alpha } \cdot \frac { 1 } { \lambda ^ { 2 } \Gamma _ { [ M ] } ^ { - } } } } \\ & { \qquad = 2 ^ { \alpha } \cdot \frac { 1 } { ( \alpha - 1 ) ( \alpha + 1 ) } \cdot \frac { \lambda } { T _ { [ M ] } ^ { - } } \cdot \left( \frac { 1 } { \left( \left( \frac { 2 } { \lambda ^ { 2 } \Gamma _ { [ M ] } ^ { - } } \right) \wedge 1 \right) ^ { \alpha - 1 } } - 1 \right) } \\ & { \qquad \leq 2 \cdot \frac { 1 } { ( \alpha - 1 ) ( \alpha + 1 ) } \cdot \frac { \lambda ^ { \alpha } } { \left( T _ { [ M ] } ^ { - } \right) ^ { \alpha } } . } \end{array}\tag{15}
$$

The last inequality above holds because if $\frac { 2 } { \lambda } T _ { [ M ] } ^ { - } > 1$ , then the expression on the second-to-last line is 0 and, otherwise, $\left( { \textstyle { \frac { 2 } { \lambda } } } T _ { [ M ] } ^ { - } \right) \wedge 1$ takes the value $\frac { 2 } { \lambda } T _ { [ M ] } ^ { - }$ . In preparation to integrate with respect to $\{ T _ { m } ^ { - } \} _ { m }$ , we lower-bound the random variable $\left( T _ { [ M ] } ^ { - } \right) ^ { \alpha }$ by a scaling of $\left( \sum _ { m = 1 } ^ { M } T _ { m } ^ { - } \right) ^ { \alpha }$ . By definition of $T _ { [ M ] } ^ { - }$

$$
\left( \sum _ { m = 1 } ^ { M } T _ { m } ^ { - } \right) ^ { \alpha } \leq \left( \sum _ { m = 1 } ^ { M } T _ { [ M ] } ^ { - } \right) ^ { \alpha } = M ^ { \alpha } \left( T _ { [ M ] } ^ { - } \right) ^ { \alpha }\tag{16}
$$

$\scriptstyle \sum _ { m = 1 } ^ { M } T _ { m } ^ { - }$ has distribution Gamma $( M , 1 )$ , whose density is known. We compute its moment of order $( - \alpha )$ Since $M \geq \alpha + 1$ ，

$$
\mathbb { E } _ { \{ T _ { m } ^ { - } \} _ { m } } \left[ \frac { 1 } { \left( \sum _ { m = 1 } ^ { M } T _ { m } ^ { - } \right) ^ { \alpha } } \right] = \int _ { 0 } ^ { \infty } { \frac { 1 } { t ^ { \alpha } } } \cdot { \frac { t ^ { M - 1 } e ^ { - t } } { \Gamma ( M ) } } d t = { \frac { 1 } { \Gamma ( M ) } } \int _ { 0 } ^ { \infty } t ^ { M - \alpha - 1 } e ^ { - t } d t = { \frac { ( M - \alpha - 1 ) ! } { ( M - 1 ) ! } } .\tag{17}
$$

Combining eq. (15), eq. (16), and $\mathrm { e q . }$ (17), and noting that $\alpha \geq 2$ by assumption,

$$
\begin{array} { r l } { \mathbb { E } _ { \{ T _ { m } ^ { - } \} _ { m } } \left[ \mathbb { E } _ { X } \left[ \frac { 1 _ { X \geq \left( \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \right) \wedge 1 } } { \nu \left( \left[ - \frac { 1 } { \lambda } T _ { [ M ] } ^ { - } + X , X \right] \right) } \right] \right] \leq \mathbb { E } _ { \{ T _ { m } ^ { - } \} _ { m } } \left[ 2 \cdot \frac { 1 } { ( \alpha - 1 ) ( \alpha + 1 ) } \cdot \frac { \lambda ^ { \alpha } } { \left( T _ { [ M ] } ^ { - } \right) ^ { \alpha } } \right] } & { } \\ { \leq 2 \cdot \frac { 1 } { ( \alpha - 1 ) ( \alpha + 1 ) } \cdot \lambda ^ { \alpha } \cdot M ^ { \alpha } \cdot \mathbb { E } _ { \{ T _ { m } ^ { - } \} _ { m } } \left[ \frac { 1 } { \left( \sum _ { m = 1 } ^ { M } T _ { m } ^ { - } \right) ^ { \alpha } } \right] } & { } \\ { = 2 \cdot \frac { 1 } { ( \alpha - 1 ) ( \alpha + 1 ) } \cdot \lambda ^ { \alpha } \cdot M ^ { \alpha } \cdot \frac { ( M - \alpha - 1 ) ! } { ( M - 1 ) ! } } & { } \\ { \leq \frac { \lambda ^ { \alpha } } { ( 1 - \frac { M } { M } ) ^ { \alpha } } . } \end{array}\tag{18}
$$

This completes an expression for the second term of eq. (14).

We turn now to the first term in $\mathrm { e q . }$ (14). It concerns the region of the support of X that is so close to the singularity of $\nu$ at 0 that we cannot lower-bound the density of ν. Rather, we will use the rate given by the $\mathcal { \bar { P } } _ { \lambda } ^ { M }$ -transfer exponent, which is $\lambda ^ { \alpha + 1 }$ , and show that this region is only on the order of $\frac { 1 } { \lambda }$ in width, recovering the rate $\lambda ^ { \alpha }$ as desired. To begin, because the density of ν is increasing on its support,

$$
\nu \left( \left[ X , \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } + X \right] \right) = \nu \left( \left[ X , \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } + X \right) \wedge 1 \right] \right) \geq \nu \left( \left[ 0 , \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) \wedge ( 1 - X ) \right] \right) .
$$

The numerator of the first term in eq. (14) indicates that $\begin{array} { r } { X \leq \left( \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \right) \wedge \left( 1 - \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) } \end{array}$ . In particular, we have $\left( \textstyle { \frac { 1 } { \lambda } } T _ { [ M ] } ^ { + } \right) \le 1 - X$ and so,

$$
\nu \left( \left[ 0 , \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) \wedge ( 1 - X ) \right] \right) = \nu \left( \left[ 0 , \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right] \right) = \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) ^ { \alpha + 1 } \wedge 1 .
$$

We integrate with respect to X to derive an expression for the inner expectation of the first term of eq. (14). Using independence of $\{ T _ { m } ^ { + } \} _ { m }$ and $\{ T _ { m } ^ { - } \} _ { m }$ (and dropping indicator functions after they have served their purpose),

$$
\mathbb { E } _ { X } \left[ \frac { \mathbf { 1 } _ { X \le \left( \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \right) \wedge \left( 1 - \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) } } { \nu \left( \left[ X , \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } + X \right] \right) } \right] \le \mathbb { E } _ { X } \left[ \frac { \mathbf { 1 } _ { X \le \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } } } { \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) ^ { \alpha + 1 } \wedge 1 } \right] \le \frac { \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } } { \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } \right) ^ { \alpha + 1 } \wedge 1 } .\tag{19}
$$

The numerator and denominator in eq. (19) are independent. A union bound to controls the expectation of the numerator by

$$
\mathbb { E } \left[ \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \right] = \frac { 2 } { \lambda } \int _ { 0 } ^ { \infty } \mathbb { P } \left[ T _ { [ M ] } ^ { - } \geq t \right] ~ d t \leq \frac { 2 } { \lambda } \int _ { 0 } ^ { \infty } \sum _ { m = 1 } ^ { M } \mathbb { P } \left[ T _ { m } ^ { - } \geq t \right] ~ d t = \frac { 2 M } { \lambda } .\tag{20}
$$

To compute the expectation of (the reciprocal of) the denominator in eq. (19), we break into the cases $T _ { [ M ] } ^ { + } \geq \lambda$ and $T _ { [ M ] } ^ { + } < \lambda$ . For the latter case, noting that $M \geq \alpha + 2$ and that $T _ { m } ^ { + }$ and $T _ { m } ^ { - }$ are identically

distributed, we apply eq. (16) and eq. (17) with $\alpha + 1$ in place of α to obtain

$$
\begin{array} { r } { \mathbb { E } [ \frac { 1 } { ( \frac { 1 } { \lambda ^ { \prime } ( \lambda ) } ) ^ { \alpha + 1 } \lambda ^ { 1 } } ] = \mathbb { E } [ \frac { \mathbf { 1 } _ { \Gamma _ { ( i ) } ^ { [ \alpha ] } \geq \lambda } } { ( \frac { 1 } { \lambda ^ { \prime } ( \lambda ) } ) ^ { \alpha + 1 } } ] + \mathbb { E } [ \frac { \mathbf { 1 } _ { \Gamma _ { ( i ) } ^ { [ \alpha ] } \leq \lambda } } { ( \frac { 1 } { \lambda ^ { \prime } ( \lambda ) } ) ^ { \alpha + 1 } \lambda ^ { 1 } } ] } \\ { \leq \mathbb { E } [ \frac { \mathbf { 1 } _ { \Gamma _ { ( i ) } ^ { [ \alpha ] } \geq \lambda } } { \frac { 1 } { \lambda } } ] + \mathbb { E } [ \frac { 1 } { ( \frac { 1 } { \lambda ^ { \prime } ( \lambda ) } ) ^ { \alpha + 1 } } ] } \\ { \leq \frac { M } { \operatorname* { m a x } } \mathbb { P } [ \Gamma _ { = } \geq \lambda ] + \mathcal { X } ^ { \alpha 1 } + \mathcal { Z } [ \frac { M ^ { \alpha + 1 } } { ( \sum _ { m = 1 } ^ { \infty } T _ { n } ^ { \alpha } ) ^ { \alpha + 1 } } ] } \\ { \leq \frac { M } { \operatorname* { m a x } } \mathbb { P } [ \Gamma _ { m } \geq \lambda ] + \mathcal { X } ^ { \alpha 1 } + \mathcal { Z } [ \frac { M ^ { \alpha + 1 } } { ( \sum _ { m = 1 } ^ { \infty } T _ { n } ^ { \alpha } ) ^ { \alpha + 1 } } ] } \\ { \leq \mathcal { M } e ^ { - \lambda } + \mathcal { X } ^ { \alpha + 1 } - \mathcal { M } ^ { \alpha - 1 } \ \frac { ( M - \alpha - 2 ) ! } { ( M - 1 ) ! } } \\  \leq \mathcal { M } e ^ { - \lambda } + \frac { \mathcal { X } ^ { \alpha + 1 } }  ( \frac { \mathcal { X } ^ { \alpha + 1 } }  \ \end{array}
$$

This, together with eq. (19) and eq. (20), gives us a bound on the first term of eq. (14):

$$
\begin{array} { r } { \mathbb { E } _ { \{ T _ { n } ^ { * } , T _ { n } \} _ { m } } \left[ \mathbb { E } _ { X } \left[ \frac { \mathbf { 1 } _ { X \leq \left( \frac { 2 } { X } , \Gamma _ { [ M ] } ^ { * } \right) , ( X + \frac { 1 } { X } , T _ { [ M ] } ^ { * } ) } } { \nu \left( \left[ X _ { \uparrow } , \frac { 1 } { X } T _ { [ M ] } ^ { * } + X \right] \right) } \right] \right] \leq \mathbb { E } _ { \{ T _ { n } ^ { * } , T _ { n } \} _ { m } } \left[ \frac { \frac { 2 } { \lambda } T _ { [ M ] } ^ { * } } { \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { * } \right) ^ { \alpha + 1 } \wedge 1 } \right] } \\ { \leq \mathbb { E } \left[ \frac { 2 } { \lambda } T _ { [ M ] } ^ { * } \right] \cdot \mathbb { E } \left[ \frac { 1 } { \left( \frac { 1 } { \lambda } T _ { [ M ] } ^ { * } \right) ^ { \alpha + 1 } \wedge 1 } \right] } \\ { \leq \frac { 2 M } { \lambda } \left( M e ^ { - \lambda } + \frac { \lambda ^ { \alpha + 1 } } { \left( 1 - \frac { \alpha + 1 } { M } \right) ^ { \alpha + 1 } } \right) } \\ { = 2 M ^ { 2 } \cdot \frac { e ^ { - \lambda } } { \lambda } + \lambda ^ { \alpha } \cdot \frac { 2 M } { \left( 1 - \frac { \alpha + 1 } { \alpha + 1 } \right) ^ { \alpha + 1 } } . } \end{array}\tag{21}
$$

Finally, there is a chance that X falls in neither of these regions but rather in the interval between, namely $\begin{array} { r } { \left( 1 - \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } , \left( \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } \right) \wedge 1 \right] } \end{array}$ . We show that with high probability this interval is empty, meaning the lower endpoint exceeds the upper. By independence of $T _ { [ M ] } ^ { - }$ and $T _ { [ M ] } ^ { + }$

$$
\begin{array} { r l } {  { \mathbb { P } [ 1 - \frac { 1 } { \lambda } T _ { [ M ] } ^ { + } < \frac { 2 } { \lambda } T _ { [ M ] } ^ { - } ] = \mathbb { E } _ { \{ T _ { m } ^ { + } \} _ { m } } \mathbb { E } _ { \{ T _ { m } ^ { - } \} _ { m } } [ \mathbf { 1 } _ { T _ { [ M ] } ^ { - } > \frac { 1 } { 2 } ( \lambda - T _ { [ M ] } ^ { + } ) } ] } \ ~ } & { } \\ & { \leq \mathbb { E } _ { \{ T _ { m } ^ { + } \} _ { m } } [ ( M \cdot \exp ( - \frac { 1 } { 2 } ( \lambda - T _ { [ M ] } ^ { + } ) ) ) \wedge 1 ] } \\ & { \leq M \cdot \exp ( - \frac { \lambda } { 2 } ) \cdot \mathbb { E } [ \exp ( \frac { 1 } { 2 } T _ { [ M ] } ^ { + } ) ] } \\ & { \leq 2 M ^ { 2 } \cdot \exp ( - \frac { \lambda } { 2 } ) . } \end{array}\tag{22}
$$

The last inequality comes from manipulating the moment generating function of the unit exponential distribution, which is $t \mapsto { \frac { 1 } { 1 - t } }$ for $| t | < 1$ , as follows:

$$
\mathbb { E } \left[ \exp \left( \frac { T _ { [ M ] } ^ { + } } { 2 } \right) \right] = \mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \exp \left( \frac { T _ { m } ^ { + } } { 2 } \right) \right] \leq \mathbb { E } \left[ \sum _ { m = 1 } ^ { M } \exp \left( \frac { T _ { m } ^ { + } } { 2 } \right) \right] = \sum _ { m = 1 } ^ { M } \mathbb { E } \left[ \exp \left( \frac { T _ { m } ^ { + } } { 2 } \right) \right] = 2 M .
$$

Altogether, eq. (14), eq. (18), eq. (21), and eq. (22) imply

$$
\mathbb { E } \left[ n \wedge \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] \leq \frac { 2 M ^ { 2 } e ^ { - \lambda } } { \lambda } + \frac { \lambda ^ { \alpha } \cdot 2 M } { \left( 1 - \frac { \alpha + 1 } { M } \right) ^ { \alpha + 1 } } + \frac { \lambda ^ { \alpha } } { \left( 1 - \frac { \alpha } { M } \right) ^ { \alpha } } + 2 n M ^ { 2 } e ^ { - \frac { \lambda } { 2 } } .
$$

Applying theorem 2 and the Mondrian case from lemma 2 gives the result.

## 10 Lemma about Binomial Random Variables

To prove theorem 1, we need the following technical lemma about binomial random variables. It is a special case of Lemma SA3 in [8] with slightly better constants. Lemma 6 in [23] provides a similar result, but it omits the truncation factor, which we need in proposition 4, and we found its proof somewhat unclear.

Lemma 3. Let $N _ { 1 }$ and $N _ { 2 }$ be independent binomial random variables with distributions Binomial $( n _ { 1 } , p _ { 1 } )$ and Binomia $( n _ { 2 } , p _ { 2 } )$ , respectively, for positive integers $n _ { 1 } , n _ { 2 }$ and $p _ { 1 } , p _ { 2 } \in [ 0 , 1 ]$ ]. Then,

$$
\mathbb { E } \left[ 1 \wedge \frac { 1 } { N _ { 1 } + N _ { 2 } } \right] \leq 4 \wedge \frac { 1 8 } { n _ { 1 } p _ { 1 } + n _ { 2 } p _ { 2 } }
$$

Proof. As in Lemma SA3 in [8], we apply Bernstein’s inequality (the one-tailed version of Proposition 2.9.5 in [32]) to a binomial random variable $N \sim$ Binomial(n, p), negated. −N is the sum of negated independent Bernoulli trials, denoted $\{ U _ { 1 } , . . . , U _ { n } \}$ , so Bernstein’s inequality applies to the centered sum $\begin{array} { r } { n p - N = \sum _ { i = 1 } ^ { n } ( p - U _ { i } ) } \end{array}$ . Each $p - U _ { i }$ has mean 0, second moment no greater than $p ,$ and maximum absolute value no greater than 1. Hence, Bernstein’s inequality states that for all positive $t ,$

$$
\mathbb { P } \left[ n p - N \ge t \right] \le \exp \left( - \frac { \frac { 1 } { 2 } t ^ { 2 } } { n p + \frac { 1 } { 3 } t } \right)
$$

Setting t to $\frac { 3 n p } { 4 }$ , we obtain the following bound on the left tail of $N .$

$$
\mathbb { P } \left[ N \leq \frac { n p } { 4 } \right] \leq \exp \left( - \frac { \frac { 1 } { 2 } \left( \frac { 3 n p } { 4 } \right) ^ { 2 } } { n p + \frac { 1 } { 3 } \left( \frac { 3 n p } { 4 } \right) } \right) = \exp \left( - \frac { 9 } { 4 0 } n p \right)
$$

Then, we bifurcate into the cases $\begin{array} { r } { N \leq \frac { n p } { 4 } } \end{array}$ and $\begin{array} { r } { N > \frac { n p } { 4 } } \end{array}$ to analyze the truncated and shifted inverse moment. For any $a \geq 0$ 2

$$
\begin{array} { r l r } {  { \mathbb { E } [ \frac { 1 } { 1 \vee ( N + a ) } ] \le \exp ( - \frac { 9 } { 4 0 } n p ) \cdot \frac { 1 } { 1 \vee a } + \frac { 1 } { 1 \vee ( \frac { n p } { 4 } + a ) } } } \\ & { } & { \le \frac { 1 } { 1 \vee ( \frac { 9 } { 4 0 } n p + a ) } + \frac { 1 } { 1 \vee ( \frac { n p } { 4 } + a ) } } \\ & { } & { \le 2 \cdot \frac { 1 } { 1 \vee ( \frac { 9 } { 4 0 } n p + a ) } } \end{array}
$$

The second inequality above uses the numeric inequality $e ^ { - t } \cdot \frac { 1 } { 1 \vee a } \le 1 \wedge \frac { 1 } { t + a }$ for $a , t \geq 0$ , which follows from the identity $e ^ { t } \leq \frac { 1 } { 1 + t }$ through the following argument. If $a \leq 1$ , then $\begin{array} { r } { e ^ { - t } \cdot \frac { 1 } { 1 \vee a } = e ^ { - t } \le \frac { 1 } { t + 1 } \le \frac { 1 } { t + a } ; } \end{array}$ also, $e ^ { - t } \cdot \frac { 1 } { 1 \vee a } = e ^ { - t } \leq 1$ . On the other hand, i $\mathrm { ~ f ~ } a > 1$ , then $\begin{array} { r } { e ^ { - t } \le \frac { 1 } { 1 + t } \le \frac { 1 } { 1 + \frac { t } { a } } , \ : \mathrm { s o } \ e ^ { - t } \cdot \frac { 1 } { 1 \vee a } = e ^ { - t } \cdot \frac 1 a \le \frac { 1 } { 1 + \frac { t } { a } } \cdot \frac 1 a = \frac { 1 } { t + a } ; } \end{array}$ also $\begin{array} { r } { e ^ { - t } \cdot \frac { 1 } { 1 \vee a } = e ^ { - t } \cdot \frac { 1 } { a } \le 1 } \end{array}$

We use this tail bound for $N _ { 1 }$ and $N _ { 2 }$ in sequence, using independence to evaluate the expectation with

respect to $N _ { 1 }$ while conditioning on $N _ { 2 }$

$$
\begin{array} { r } { \mathbb { E } \left[ 1 \wedge \frac { 1 } { N _ { 1 } + N _ { 2 } } \right] = \mathbb { E } _ { N _ { 2 } } \left[ \mathbb { E } _ { N _ { 1 } } \left[ 1 \wedge \frac { 1 } { N _ { 1 } + N _ { 2 } } \right] \right] } \\ { \leq \mathbb { E } _ { N _ { 2 } } \left[ 2 \cdot \frac { 1 } { 1 \vee \left( \frac { 9 } { 4 0 } n _ { 1 } p _ { 1 } + N _ { 2 } \right) } \right] } \\ { \leq 4 \cdot \frac { 1 } { 1 \vee \left( \frac { 9 } { 4 0 } n _ { 1 } p _ { 1 } + \frac { 9 } { 4 0 } n _ { 2 } p _ { 2 } \right) } } \\ { = 4 \wedge \frac { 1 8 } { n _ { 1 } p _ { 1 } + n _ { 2 } p _ { 2 } } } \end{array}
$$

## 11 Proof of Mondrian Kernel Estimator Bias Bound

Proof of Mondrian case in lemma 2. The Mondrian case can be handled diferently to obtain an explicit dependence on d and $\beta .$

Since $t \mapsto t ^ { \beta }$ is concave, we apply Jensen’s inequality as before to isolate the second moment, which we will subsequently bound.

$$
\begin{array} { r } { \mathbb { E } \left[ L ^ { 2 } \underset { m = 1 } { \overset { M } { \operatorname* { m a x } } } \mathrm { d i a m } ( Z _ { x } ^ { m , \lambda } \cap \operatorname { s u p p } ( \nu ) ) ^ { 2 \beta } \right] \leq L ^ { 2 } \left( \mathbb { E } \left[ \underset { m = 1 } { \overset { M } { \operatorname* { m a x } } } \mathrm { d i a m } ( Z _ { x } ^ { m , \lambda } ) ^ { 2 } \right] \right) ^ { \beta } } \end{array}
$$

For a given $m ,$ the squared diameter of $Z _ { x } ^ { m , \lambda }$ is stochastically upper-bounded by the sum of d squared, independent Gamma(2, λ)-distributed variables, per the proof of Corollary 1 in [19]. Letting $\{ W _ { j , m } \}$ and $\{ W _ { m } \}$ refer to independent Gamma(2, 1) variables, we can bound the second moment of diam $( Z _ { x } ^ { m , \lambda } )$ .

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \dim ( Z _ { x } ^ { m , \lambda } ) ^ { 2 } \right] \leq \mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \sum _ { j = 1 } ^ { d } \frac { W _ { j , m } ^ { 2 } } { \lambda ^ { 2 } } \right] \leq \mathbb { E } \left[ \sum _ { j = 1 } ^ { d } \operatorname* { m a x } _ { m = 1 } ^ { M } \frac { W _ { j , m } ^ { 2 } } { \lambda ^ { 2 } } \right] = \frac { d } { \lambda ^ { 2 } } \mathbf { \cdot E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } ^ { 2 } \right] = \frac { d } { \lambda ^ { 2 } } \mathbf { \cdot E } \left[ \left( \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } \right) ^ { 2 } \right]
$$

Each $W _ { m }$ is Gamma-distributed, hence sub-exponential [32]. We adapt the (second) proof of Proposition 2.7.6 in [32] to show that the maximum of M sub-exponential random variables is sub-exponential with sub-exponential norm on the order of log M. From this we infer a second-moment bound for $\operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } .$ as desired.

For each $m ,$ the variable $W _ { m }$ has sub-exponential norm $\scriptstyle { \frac { 2 } { 2 - { \sqrt { 2 } } } }$ because the moment generating function of $W _ { m }$ , evaluated at the reciprocal of this number, is 2. (See the definition in line (2.25) in [32].)

$$
\mathbb { E } \left[ \exp \left( \frac { W _ { m } } { 2 - \sqrt { 2 } } \right) \right] = \left( 1 - \frac { 2 - \sqrt { 2 } } { 2 } \right) ^ { - 2 } = 2
$$

From this, we derive a similar bound on the moment generating function of ma $\tau _ { m = 1 } ^ { M } W _ { m }$ , after scaling by a logarithmic factor.

$$
\begin{array} { r } { \mathbb { E } \left[ \exp \left( \frac { \displaystyle \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \displaystyle \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) } \right) \right] \leq \left( \mathbb { E } \left[ \exp \left( \frac { \displaystyle M } { \displaystyle m = 1 } \frac { \displaystyle { W _ { m } } } { \displaystyle \frac { 2 } { 2 - \sqrt { 2 } } } \right) \right] \right) ^ { \frac { 1 } { 2 \log ( 2 M ) } } } \\ { = \left( \mathbb { E } \left[ \frac { \displaystyle M } { \displaystyle m = 1 } \exp \left( \frac { W _ { m } } { \displaystyle \frac { 2 } { 2 - \sqrt { 2 } } } \right) \right] \right) ^ { \frac { 1 } { 2 \log ( 2 M ) } } } \\ { \leq \left( \mathbb { E } \left[ \frac { \displaystyle M } { \displaystyle m - 1 } \exp \left( \frac { W _ { m } } { \displaystyle \frac { 2 } { 2 - \sqrt { 2 } } } \right) \right] \right) ^ { \frac { 1 } { 2 \log ( 2 M ) } } } \\ { \leq \left( M \cdot 2 \right) ^ { \frac { 1 } { 2 \log ( 2 M ) } } = \sqrt { e } \leq 2 } \end{array}
$$

The first inequality above is Jensen’s inequality for concave functions; it holds for $M \geq 2$ , where $t \mapsto t ^ { \frac { 1 } { 2 \log ( 2 M ) } }$ is concave.

Adapting the proof of Proposition 2.6.1 in [32], we obtain a sub-exponential tail bound for the random variable $\frac { \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) }$

$$
\mathbb { P } \left[ \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) } > t \right] = \mathbb { P } \left[ \exp \left( \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) } \right) > e ^ { t } \right] \leq e ^ { - t } \cdot \mathbb { E } \left[ \exp \left( \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) } \right) \right] \leq 2 e ^ { - t }
$$

Finally, we derive a second moment bound for $\operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m }$

$$
\begin{array} { r l r } {  { \mathbb { E } [ ( \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) } ) ^ { 2 } ] = \int _ { 0 } ^ { \infty } \mathbb { P } [ \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } W _ { m } } { \frac { 2 } { 2 - \sqrt { 2 } } \cdot 2 \log ( 2 M ) } > t ] \cdot 2 t d t \leq 4 \int _ { 0 } ^ { \infty } t e ^ { - t } d t = 4 } } & { } & \\ & { } & { \mathbb { E } [ ( \frac { M } { m = 1 } W _ { m } ) ^ { 2 } ] = 4 ( \frac { 2 - \sqrt { 2 } } { 2 } \cdot 2 \log ( 2 M ) ) ^ { 2 } = \frac { 3 2 } { 3 - 2 \sqrt { 2 } } \log ^ { 2 } ( 2 M ) } & \end{array}
$$

Substituting this bound into our previous expression completes the proof.

$$
\begin{array} { r } { \mathbb { E } \left[ L ^ { 2 } \underset { m = 1 } { \overset { M } { \operatorname* { m a x } } } \mathrm { d i a m } ( Z _ { x } ^ { m , \lambda } \cap \operatorname { s u p p } ( \nu ) ) ^ { 2 \beta } \right] \leq L ^ { 2 } \left( \mathbb { E } \left[ \underset { m = 1 } { \overset { M } { \operatorname* { m a x } } } \mathrm { d i a m } ( Z _ { x } ^ { m , \lambda } ) ^ { 2 } \right] \right) ^ { \beta } } \\ { \leq L ^ { 2 } \left( \frac { d } { \lambda ^ { 2 } } \cdot \mathbb { E } \left[ \left( \underset { m = 1 } { \overset { M } { \operatorname* { m a x } } } W _ { m } \right) ^ { 2 } \right] \right) ^ { \beta } } \\ { \leq \left( \frac { 3 2 } { 3 - 2 \sqrt { 2 } } \right) ^ { \beta } \cdot \frac { L ^ { 2 } d ^ { \beta } } { \lambda ^ { 2 \beta } } \cdot { \log ^ { 2 } ( 2 M ) } } \end{array}
$$

## 12 Derivation of a Moment Bound from a Tail Bound

We derive a moment bound on the maximum of sub-exponential variables from a tail bound.

Derivation of Equations (11) and (12) from Equation (10). First, we justify the final inequality in eq. (11) via the following numerical argument. If dM is not too large, then the exponential term can absorb the coeficient with a small penalty.

$$
d M \leq \exp \left( \frac { c _ { \Pi } t } { 4 } \right) \implies 2 d M \cdot \exp \left( - \frac { c _ { \Pi } t } { 2 } \right) \leq 2 \exp \left( - \frac { c _ { \Pi } t } { 4 } \right) \leq 2 \exp \left( - \frac { c _ { \Pi } t } { 6 \log ( d M ) } \right)
$$

The last inequality above holds because $d M \geq 2$ . On the other hand, if dM is very large, the claimed tail bound becomes trivial because it exceeds 1.

$$
d M > \exp \left( \frac { c _ { \Pi } t } { 4 } \right) \implies 2 \exp \left( - \frac { c _ { \Pi } t } { 6 \log ( d M ) } \right) > 2 \exp \left( - \frac { 2 } { 3 } \right) > 1
$$

Thus, eq. (11) establishes a tail bound on ma $\mathfrak { x } _ { m = 1 } ^ { M } \mathrm { d i a m } ( Z _ { 0 } ^ { m , 1 } )$ for $d M \geq 2$ , from which we now derive a bound on its kth moment for any $k \geq 1$ , using the proof technique of Proposition 2.6.1 in [32].

$$
\begin{array} { r l r } {  { \mathbb { E } [ ( \operatorname* { m a x } _ { m = 1 } ^ { M } \dim ( Z _ { 0 } ^ { m , 1 } ) ) ^ { k } ] = \int _ { 0 } ^ { \infty } \mathbb { P } [ \operatorname* { m a x } _ { m = 1 } ^ { M } \dim ( Z _ { 0 } ^ { m , 1 } ) \geq t ] \cdot k t ^ { k - 1 } \ d t } } \\ & { } & { \leq \int _ { 0 } ^ { \infty } 2 \exp ( - \frac { c _ { \Pi } t } { 6 \log ( d M ) } ) \cdot k t ^ { k - 1 } \ d t } \\ & { } & { = 2 k ( \frac { 6 \log ( d M ) } { c _ { \Pi } } ) ^ { k } \cdot \Gamma ( k ) } \end{array}
$$

The notation $\Gamma ( \cdot )$ refers to the gamma function.

For our purposes, it sufices to bound the second moment, i.e., $k = 2$

$$
\mathbb { E } \left[ \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \mathrm { d i a m } \left( Z _ { 0 } ^ { m , 1 } \right) \right) ^ { 2 } \right] \leq 1 4 4 \left( \frac { 1 } { c _ { \Pi } } \right) ^ { 2 } \log ^ { 2 } ( d M )\tag{23}
$$

Since $t \mapsto t ^ { \beta }$ is concave, we can apply Jensen’s inequality to isolate the second moment and apply the bound from eq. (23).

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { m = 1 } ^ { M } \dim ( Z _ { 0 } ^ { m , 1 } ) ^ { 2 \beta } \right] \leq \left( \mathbb { E } \left[ \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \dim \left( Z _ { 0 } ^ { m , 1 } \right) \right) ^ { 2 } \right] \right) ^ { \beta } \leq \frac { 1 4 4 ^ { \beta } } { c _ { \Pi } ^ { 2 \beta } } \log ^ { 2 \beta } ( d M )\tag{24}
$$

In the case of $d = M = 1$ , a single tree’s diameter diam $\big ( Z _ { 0 } ^ { 1 , 1 } \big )$ is the maximum diameter ma $\mathrm { x } _ { m = 1 } ^ { M }$ diam $( Z _ { 0 } ^ { m , 1 } )$ , so eq. (10) immediately provides the tail bound we need to repeat the derivation of a moment bound.

$$
\mathbb { E } \left[ \mathrm { d i a m } ( Z _ { 0 } ^ { 1 , 1 } ) ^ { 2 \beta } \right] \leq \left( \mathbb { E } \left[ \left( \mathrm { d i a m } \left( Z _ { 0 } ^ { 1 , 1 } \right) \right) ^ { 2 } \right] \right) ^ { \beta } \leq 1 6 ^ { \beta } \left( \frac { 1 } { c _ { \Pi } } \right) ^ { 2 \beta }\tag{25}
$$

## 13 Proof of Infinite $\mathcal { P } _ { \lambda } ^ { M }$ -transfer Discrepancy

Proof of proposition 3. We begin by expressing the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer discrepancy using the STIT tessellation’s stationarity and eponymous property: stability under iteration [21].

$$
\mathbb { E } \left[ \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } Z _ { X } ^ { m , \lambda } \right) } \right] = \mathbb { E } \left[ \frac { 1 } { \nu \left( \frac { 1 } { \lambda } \left( \bigcup _ { m = 1 } ^ { M } Z _ { 0 } ^ { m , 1 } \right) + X \right) } \right]\tag{26}
$$

Now, the independently realized random partitions are all Mondrian trees with lifetime 1. The zero cell of the partition indexed m is denoted $Z _ { 0 } ^ { \bar { m } , 1 }$ and is distributed as the stochastic interval $[ - T _ { m } ^ { - } , T _ { m } ^ { + } ]$ where $T _ { m } ^ { - }$ and $T _ { m } ^ { + }$ are independent unit exponential random variables [19]. The union of these intervals is $[ - \operatorname * { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } , \operatorname * { m a x } _ { m = 1 } ^ { M } { \bar { T } } _ { m } ^ { + } ]$ . Hence, we can rewrite the integrated $\dot { \mathcal { P } } _ { \lambda } ^ { M }$ -transfer exponent in terms of independent unit exponential random variables.

$$
\mathbb { E } \left[ \frac { 1 } { \nu \left( \frac { 1 } { \lambda } \left( \bigcup _ { m = 1 } ^ { M } Z _ { 0 } ^ { m , 1 } \right) + X \right) } \right] = \mathbb { E } \left[ \frac { 1 } { \nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + X , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) } \right]\tag{27}
$$

Condition on the exponential random variables, and consider the expectation with respect to X restricted to the case $\begin{array} { r } { X \leq \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \frac { T _ { m } ^ { + } } { \lambda } \right) \wedge 1 } \end{array}$ . This case is the most probable one because each $T _ { m } ^ { + }$ has mean 1. We restrict to this case only for convenience in the analysis.

$$
\mathbb { E } _ { X } \left[ \frac { 1 } { \nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + X , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) } \right] \geq \mathbb { E } _ { X } \left[ \frac { \mathbf { 1 } _ { X \leq \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \frac { T _ { m } ^ { + } } { \lambda } \right) \wedge 1 } } { \nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + X , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) } \right]
$$

We demonstrate that the right-hand side of the line above is at least on the order of $\frac { \lambda ^ { \alpha } } { ( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } ) ^ { \alpha } }$ . Since ν has support [0, 1], and in the event that $\begin{array} { r } { X \leq \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \frac { T _ { m } ^ { + } } { \lambda } \right) \wedge 1 } \end{array}$ ，

$$
\nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + X , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) \leq \nu \left( \left[ 0 , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) \leq \nu \left( \left[ 0 , \frac { \lambda + 1 } { \lambda } \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { m = 1 } \right] \right)
$$

and substituting the density for $\nu ,$

$$
\nu \left( \left[ 0 , \frac { \lambda + 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right] \right) = ( \alpha + 1 ) \int _ { 0 } ^ { \left( \frac { \lambda + 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 } x ^ { \alpha } \ d x = \left( \left( \frac { \lambda + 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 \right) ^ { \alpha + 1 }
$$

so that for all $\lambda \geq 1$

$$
\begin{array} { r l } { \mathbb { E } _ { X } \left[ \frac { \mathbf { 1 } _ { X \leq \left( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 } } { \nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + X , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) } \right] = \int _ { 0 } ^ { \left( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 } \frac { d x } { \nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + x , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + x \right] \right) } } \\ & { \geq \int _ { 0 } ^ { ( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } ) \wedge 1 } \frac { d x } { \left( \left( \frac { \lambda + 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 \right) ^ { \alpha + 1 } } } \\ & { = \frac { \left( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 } { \left( \left( \frac { \lambda + 1 } { \lambda } \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 \right) ^ { \alpha + 1 } } } \\ & { \geq \frac { 1 } { 2 ^ { \alpha + 1 } } \cdot \frac { 1 } { \left( \left( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 \right) ^ { \alpha } } } \end{array}
$$

Note that the maximum of independent exponential random variables $\lbrace T _ { m } ^ { + } \rbrace _ { m = 1 } ^ { M }$ is less than their sum, and their sum is a Gamma-distributed random variable with shape parameter M and rate parameter 1.

$$
\Bigl ( \Bigl ( \sum _ { m = 1 } ^ { M } T _ { m } ^ { + } \Bigr ) \wedge 1 \Bigr ) \le \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \le \sum _ { m = 1 } ^ { M } T _ { m } ^ { + } \sim \mathrm { G a m m a } ( M , 1 )
$$

The first inverse moment of the unit exponential random variable is infinite by comparison to the harmonic series on the interval (0, 1). From this, we extrapolate that the Gamma(M, 1) variable’s moment of order −α is infinite for $\alpha \geq M$

$$
\begin{array} { r l } { \mathbb { E } _ { \{ T _ { m } ^ { + } \} _ { m } } \left[ \displaystyle \frac { 1 } { \left( \sum _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) ^ { \alpha } } \right] = \int _ { 0 } ^ { \infty } \displaystyle \frac { 1 } { t ^ { \alpha } } \cdot \frac { 1 } { \Gamma ( M ) } t ^ { M - 1 } e ^ { - t } \ d t } & { } \\ & { = \displaystyle \frac { 1 } { \Gamma ( M ) } \int _ { 0 } ^ { \infty } \displaystyle \frac { 1 } { t ^ { \alpha - M + 1 } } \cdot e ^ { - t } \ d t } \\ & { \geq \displaystyle \frac { 1 } { \Gamma ( M ) } \int _ { 0 } ^ { 1 } \displaystyle \frac { 1 } { t ^ { \alpha - M + 1 } } \cdot e ^ { - t } \ d t } \\ & { \geq \displaystyle \frac { 1 } { \Gamma ( M ) } \int _ { 0 } ^ { 1 } \displaystyle \frac { 1 } { t } \cdot e ^ { - t } \ d t } \\ & { = \infty \ \infty } \end{array}
$$

Hence, the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent is infinite.

$$
\begin{array} { r l } & { \mathbb { E } \left[ \frac { 1 } { \nu \left( \bigcup _ { m = 1 } ^ { M } \mathcal { Z } _ { X } ^ { m , \lambda } \right) } \right] \geq \mathbb { E } _ { \{ T _ { m } ^ { - } , T _ { m } ^ { + } \} _ { m } } \mathbb { E } _ { X } \left[ \frac { 1 _ { X \leq \left( \operatorname* { m a x } _ { m = 1 } ^ { M } \frac { T _ { m } ^ { + } } { \lambda } \right) \wedge 1 } } { \nu \left( \left[ - \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { - } } { \lambda } + X , \frac { \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } } { \lambda } + X \right] \right) } \right] } \\ & { \qquad \geq \mathbb { E } _ { \{ T _ { m } ^ { + } \} _ { m } } \left[ \frac { 1 } { 2 ^ { \alpha + 1 } } \cdot \frac { 1 } { \left( \left( \operatorname* { m a x } _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) \wedge 1 \right) ^ { \alpha } } \right] } \\ & { \qquad \geq \frac { 1 } { 2 ^ { \alpha + 1 } } \cdot \mathbb { E } _ { \{ T _ { m } ^ { + } \} _ { m } } \left[ \frac { 1 } { \left( \sum _ { m = 1 } ^ { M } T _ { m } ^ { + } \right) ^ { \alpha } } \right] } \\ & { \qquad = \infty } \end{array}
$$

## 14 Future Research Directions

We ofer a few research questions that extend from our work.

First, what is the relationship between the partition-based discrepancy measures we introduce and the previously introduced discrepancy measures? We have shown three cases: one in which the aggregate rtransfer discrepancy upper-bounds the aggregate $\mathcal { P } ^ { M }$ -transfer discrepancy (Remark 1), one in which the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer discrepancy is infinite when the integrated r-transfer discrepancy is finite (proposition 3), and one in which the integrated r-transfer exponent coincides with the integrated $\mathcal { P } _ { \lambda } ^ { M }$ -transfer exponent with suficiently high probability (proposition 4). Given the wide variety of partitioning schemes, a general relationship is unlikely to exist, but for partitioning schemes used in modeling, the relationship is important to elucidate because the transfer exponent (and its refinements) appear in risk lower bounds [16, 23]. On the other hand, the relationship among partition-based discrepancies is actually clearer (see proposition 1) than the relationship between integrated and aggregate r-transfer discrepancies [10].

Second, what can this line of theory teach us about the performance of ensembles of interpolating models under covariate shift? Modelers often tune complexity parameters higher than the bias-variance tradeof would recommend, even to the point of interpolating the training data, and then rely on ensembling to control variance [33]. Recent theory demonstrates that consistency is possible even for interpolating models, including certain random forests [4, 1]. Perhaps similar results hold under covariate shift, or else covariate shift might reveal the fragility of such highly parameterized models.

Third, what theoretical guarantees can be provided for ensemble models under more general distribution shifts? In parallel to the development of the transfer exponent, a discrepancy measure (the “relative signal exponent”) for posterior drift was introduced [7] in the context of nonparametric classification and another one for nonparametric regression [6]. The posterior drift setting does not permit covariate shifts with nonzero transfer exponent [16, 6], but the conclusion of the classification paper [7] proposes a method for incorporating the transfer exponent alongside the relative signal exponent to bound the convergence rate of a classifier under a general distribution shift. Hence, it may be possible to combine refinements of the transfer exponent with suitable discrepancy measures for regression to derive upper bounds on the risk of ensemble models under general distribution shifts.

## References

[1] L. Arnould, C. Boyer, and E. Scornet. Is interpolation benign for random forest regression? In F. Ruiz, J. Dy, and J.-W. van de Meent, editors, Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5493–5548. PMLR, 25–27 Apr 2023.

[2] M. Balog, B. Lakshminarayanan, Z. Ghahramani, D. M. Roy, and Y. W. Teh. The mondrian kernel. In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence, UAI’16, page 32–41, Arlington, Virginia, USA, 2016. AUAI Press.

[3] F. Bayram and B. S. Ahmed. A domain-region based evaluation of ml performance robustness to covariate shift. Neural Computing and Applications, 35(24):17555–17577, Aug. 2023.

[4] M. Belkin, A. Rakhlin, and A. B. Tsybakov. Does data interpolation contradict statistical optimality? In K. Chaudhuri and M. Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 1611–1619. PMLR, 16–18 Apr 2019.

[5] L. Breiman. Random forests. Machine Learning, 45(1):5–32, Oct. 2001.

[6] T. T. Cai and H. Pu. Transfer learning for nonparametric regression: Non-asymptotic minimax analysis and adaptive procedure. (arXiv:2401.12272), Jan. 2024. arXiv:2401.12272.

[7] T. T. Cai and H. Wei. Transfer learning for nonparametric classification: Minimax rate and adaptive classifier. The Annals of Statistics, 49(1), Feb. 2021.

[8] M. D. Cattaneo, J. M. Klusowski, and W. G. Underwood. Inference with mondrian random forests. Journal of the Royal Statistical Society Series B: Statistical Methodology, Dec. 2025.

[9] C. Clark, M. Yatskar, and L. Zettlemoyer. Don’t take the easy way out: Ensemble based methods for avoiding known dataset biases. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), page 4067–4080, Hong Kong, China, 2019. Association for Computational Linguistics.

[10] N. R. Galbraith and S. Kpotufe. Classification tree pruning under covariate shift. IEEE Transactions on Information Theory, 70(1):456–481, Jan. 2024.

[11] R. Genuer. Variance reduction in purely random forests. Journal of Nonparametric Statistics, 24(3):543–562, 2012.

[12] L. Gy¨orfi, M. Kohler, A. Krzy˙zak, and H. Walk. A Distribution-Free Theory of Nonparametric Regression. Springer Series in Statistics. Springer New York, New York, NY, 2002.

[13] D. Hug and R. Schneider. Poisson Hyperplane Tessellations. Springer Monographs in Mathematics. Springer Nature Switzerland, Cham, 2024.

[14] J. Klusowski. Sharp analysis of a simple model for random forests. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, page 757–765. PMLR, Mar. 2021.

[15] P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao, T. Lee, E. David, I. Stavness, W. Guo, B. Earnshaw, I. Haque, S. M. Beery, J. Leskovec, A. Kundaje, E. Pierson, S. Levine, C. Finn, and P. Liang. Wilds: A benchmark of in-thewild distribution shifts. In Proceedings of the 38th International Conference on Machine Learning, page 5637–5664. PMLR, 2021.

[16] S. Kpotufe and G. Martinet. Marginal singularity and the benefits of labels in covariate-shift. The Annals of Statistics, 49(6), Dec. 2021.

[17] B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 6405–6416, Red Hook, NY, USA, 2017. Curran Associates Inc.

[18] M. E. Lopes, S. Wu, and T. C. M. Lee. Measuring the algorithmic convergence of randomized ensembles: The regression setting. SIAM Journal on Mathematics of Data Science, 2(4):921–943, Jan. 2020.

[19] J. Mourtada, S. Ga¨ıfas, and E. Scornet. Minimax optimal rates for mondrian trees and forests. The Annals of Statistics, 48(4), Aug. 2020.

[20] W. Nagel and V. Weiss. Limits of sequences of stationary planar tessellations. Advances in Applied Probability, 35(1):123–138, Mar. 2003.

[21] E. O’Reilly and N. M. Tran. Stochastic Geometry to Generalize the Mondrian Process. SIAM Journal on Mathematics of Data Science, 4(2):531–552, June 2022.

[22] E. O’Reilly and N. M. Tran. Minimax Rates for High-Dimensional Random Tessellation Forests. Journal of Machine Learning Research, 25(89):1–32, 2024.

[23] R. Pathak, C. Ma, and M. Wainwright. A new similarity measure for covariate shift with applications to nonparametric regression. In Proceedings of the 39th International Conference on Machine Learning, pages 17517–17530. PMLR, June 2022. ISSN: 2640-3498.

[24] J. Qui˜nonero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence, editors. Dataset Shift in Machine Learning. The MIT Press, Dec. 2008.

[25] A. Rahimi and B. Recht. Random Features for Large-Scale Kernel Machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007.

[26] H. Raza, D. Rathee, S.-M. Zhou, H. Cecotti, and G. Prasad. Covariate shift estimation based adaptive ensemble learning for handling non-stationarity in motor imagery related eeg-based brain-computer interface. Neurocomputing, 343:154–166, May 2019.

[27] R. Schneider and W. Weil. Stochastic and integral geometry. Probability and Its Applications. Springer, Berlin, 2008.

[28] T. Schreiber and C. Th¨ale. Geometry of iteration stable tessellations: Connection with poisson hyperplanes. Bernoulli, 19(5A), Nov. 2013.

[29] E. Scornet. Random forests and kernel methods. IEEE Transactions on Information Theory, 62(3):1485–1500, Mar. 2016.

[30] E. Scornet, G. Biau, and J.-P. Vert. Consistency of random forests. The Annals of Statistics, 43(4), Aug. 2015.

[31] M. Sugiyama, T. Suzuki, and T. Kanamori. Density Ratio Estimation in Machine Learning. Cambridge University Press, 2012.

[32] R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, 2 edition, Feb. 2026.

[33] M. Wu and Q. Sun. Ensemble linear interpolators: The role of ensembling. SIAM Journal on Mathematics of Data Science, 7(2):438–467, 2025.

[34] J. Zhang, G. Andreeva, and Y. Dong. A novel dynamic ensemble learning (DEL) framework to combat the dataset shift: The case of loss given default. Working paper, Credit Research Centre, University of Edinburgh Business School, 2025.

[35] A. Zhou and S. Levine. Training on test data with bayesian adaptation for covariate shift. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc.