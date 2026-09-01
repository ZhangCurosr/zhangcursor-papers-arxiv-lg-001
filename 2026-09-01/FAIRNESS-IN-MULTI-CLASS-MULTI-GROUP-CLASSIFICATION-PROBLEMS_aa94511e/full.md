# FAIRNESS IN MULTI-CLASS MULTI-GROUP CLASSIFICATION PROBLEMS VIA CONTEXTUAL COHERENT RISK MEASURES

Darinka Dentcheva Department of Mathematical Sciences Stevens Institute of Technology Hoboken, NJ 07030, USA ddentche@stevens.edu

Xiangyu Tian   
School of Business   
Stevens Institute of Technology   
Hoboken, NJ 07030, USA   
xtian9@stevens.edu

## ABSTRACT

We propose a new design of fair classifiers for multi-class classification problems in the presence of vector-valued sensitive attributes. In that scenario each sensitive attribute has multiple values and forms several groups relevant to the fairness consideration. Naturally those groups are overlapping and one should also analyze the interaction of factors. Additionally, the decision makers aided by the classification should not violate individual rights at the expense of satisfying fairness metrics at the group level. We propose an approach using the theory and methods of coherent measures of risk aiming at resolving the fairness challenges. Further, we propose a specialized numerical method for solving the resulting optimization problem. The method scales well with the increase of the number of observations. Additionally, we note that the obtained classifier is robust with respect to corrupted data or to situation when data is scarce. We demonstrate the advantages of the proposed framework in comparison to the support-vector machine framework and other methods handling fairness.

Keywords coherent risk measures, contextual risk, systemic risk, stochastic programming, fairness

## 1 Introduction

Fairness has attracted significant attention in the machine-learning community, highlighting concerns about biased outcomes in automated decision-making. The algorithms used in job recruiting, credit risk assessment, and beyond need to be carefully designed to avoid propagating biases.

Different applications motivate different fairness notions. The choice of definition reflects the domain and the harms one aims to prevent. We discuss the choice of fainess metric in due course.

In our paper, we propose a multi-class multi-group method, which directly handles the multi-class classification problem and pays attention regarding the fairness among multiple groups associated with a sensitive attributes including intersectional groups. Furthermore, we can show that our approach results in individual fairness within each group.

Work on algorithmic fairness interventions is often grouped into three approaches, that can be summarized as follows.

• Methods modifying the data [18, 24, 42]. These data-modification (pre-processing) methods enforce fairness by changing the training data distribution before learning, so that standard classifiers trained on the processed data are less likely to reproduce group disparities. Typical strategies include re-weighting or resampling examples to balance group–label combinations, relabeling a small set of borderline cases to reduce observed discrimination, and “repairing” or learning fair representations that remove (or weaken) information correlated with sensitive attributes while preserving predictive signal. In short, they aim to make the dataset itself less biased, so fairness improvements hold across many downstream models.

• Methods modifying the learner/objective [15, 19, 30, 33, 37]. These methods usually enforce fairness inside the optimization problem itself: they keep the training data largely as-is, but change what the model is asked to minimize (or the way SGD “sees” the data). The approaches include adding fairness constraints, fairness regularizers, or fairness-aware training dynamics to the model or training process.

• Methods adjusting the predictions [8, 29]. These methods train the model as usual, and then enforce a fairness notion by transforming its outputs (scores/probabilities/labels)—often using the sensitive attribute at prediction time—via group-specific thresholds, score mappings, or controlled randomization, so the final predictions satisfy a desired constraint.

While the proposed approached have led to substantial progress, most of them do not discuss the potential issue of data corruption and provide robustness. In practice, however, fairness goals are sometimes pursued in settings where data are noisy, limited, or systematically imperfect: labels may be corrupted, sensitive attributes may be missing or inaccurately measured, and minority groups may be underrepresented, making constraint satisfaction unstable. This motivates studies such as [6, 16, 35–37, 39]. These approaches usually aim to incorporate fairness directly into adversarial training, enforce fairness constraints that remain valid under adversarial perturbations, and use distributionally robust optimization to guard fairness guarantees against plausible shifts in the data-generating distribution.

In this paper, our goal is to analyze the classification methods based on coherent measures of risk, which were proposed [12]. Our main focus is to enforce fairness in machine learning and provide robustness at the same time via the notions of contexual risk and systemic risk aggregator measures. We have proposed the concept of contextual risk in [12], where conditional risk are evaluated given contexts. In the context of fairness as associated contexts with the values of the sensitive attributes. We note that coherent measures of risk come with built-in robustness while remaining amenable to efficient computation as we shall elaborate in due course. We use nonlinear risk aggregation, which enhances robustness against inaccurate data while enabling fairness enforcement. We point out that our method provides a structured way to handle multi-class/multi-group fairness problem. Moreover, our work does not focus on hard-enforcement of fairness such as directly constraining the gap between fairness metrics of different groups. Instead, many coherent measures of risk introduce an implicit penalty on the difference of misclassification risks of each group. The effect is that each group is treated fairly during the training process, while still focusing on giving accurate predictions even in presence of corrupted data. Furthermore, we show theoretically and numerically that the outcomes of the proposed framework are consistent with many inequality indices known in economics, that are based on widely accepted axioms. In this way our approach facilitates enforcement of fairness under complex situations, which is confirmed by our numerical experiments. While the extant literature accepts drop of performance for fairness-aware classifiers (see, e.g., [33]), we observe high performance of the proposed classifiers for the same or better fairness outcomes.

## 2 Coherent measures of risk for groups’ and systems’ losses

The risk of a scalar random variable is evaluated by a risk functional, that complies with an accepted set of axioms, which were proposed in [4] ina special case. Extensions, analysis and further theoretical advances are presented in [9, 11, 20, 28, 31, 32] and many other works. We refer to [11] for an extensive treatment of the theory of risk measures, other risk-averse models, and methods for optimization under uncertainty and risk; see also the references ibid.

We work with the space of random vectors with realizations in $\mathbb { R } ^ { M }$ , defined on the probability space $( \Omega , { \mathcal { F } } , P )$ which have finite p-th moments, $p \in [ 1 , \infty )$ , and are indistinguishable on events with zero probability; this space is denoted $\mathcal { L } _ { p } ( \Omega , \mathcal { F } , \mathbf { \tilde { P } } ; \mathbb { R } ^ { M } )$ ). We assume that the random variables represent losses. A counterpart axioms and results exists for random variables representing gains. We focus on the former preferences as we plan to evaluate the risk in classification.

When we deal with scalar-valued random variables $( M ~ = ~ 1 )$ , then a lower semi-continuous functional $\varrho : \cdot \cdot$ $\mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ) \to \mathbb { R } \cup \{ + \infty \}$ is called a coherent risk measure if it is convex, positively homogeneous, monotonic with respect to the a.s. comparison of random variables, and satisfies the following translation property:

$$
\varrho [ Z + a ] = \varrho [ Z ] + a \mathrm { ~ f o r ~ a l l ~ } Z \in \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ) , \ : a \in \mathbb { R } .
$$

If $\varrho [ \cdot ]$ is monotonic, convex, and satisfies the translation property, it is called a convex risk measure. The theory of risk measures for scalar-valued random variables has reached advanced state of development including numerical methods for solving the stochastic optimization problems involving those measures. Less work is associated with high dimensional risks, that is, measures of risk for random vectors, or risk of complex systems with heterogeneous sources of risk. The need for special attention to random vectors arises in part from the challenge that univariate measures of risk are not additive, unlike the expected value functional.

We denote the n-dimensional vector, whose components are all equal to one by 1, and the random vector with realizations equal to 1 by I. We adopt the following definition, introduced in [3, 11]:

Definition 1. A lower semi-continuous functional $\varrho : \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } ) \to \mathbb { R } \cup \{ + \infty \}$ is a systemic coherent risk measure with preference to small outcomes if it satisfies the following properties:

A1. Convexity: For all $Z , Z ^ { \prime } \in \mathcal { L } _ { v } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } )$ and $\alpha \in ( 0 , 1 )$ , the following holds $\varrho [ \alpha Z + ( 1 - \alpha ) Z ^ { \prime } ] \leq \alpha \varrho [ Z ] + ( 1 - \alpha ) \varrho [ Z ^ { \prime } ]$

A2. Monotonicity: For all $Z , Z ^ { \prime } \in \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } )$ , if $Z _ { i } \geq Z _ { i } ^ { \prime }$ for all components $i = 1 , \ldots , M , P \mathrm { - a . s } .$ ., then $\varrho [ Z ] \ge \varrho [ Z ^ { \prime } ]$

A3. Positive homogeneity: For all $Z \in \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } )$ and all constants $t > 0$ , the relation $\varrho [ t X ] = t \varrho [ X ]$ holds.

A4. Translation equivariance: For all $Z \in \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } )$ and for all constants $a \in \mathbb { R }$ , the following relation holds $\varrho [ Z + \dot { a } \mathbb { I } ] = \varrho [ Z ] + a \varrho [ \mathbb { I } ]$

It is shown in [3, 11] that any systemic risk measure $\varrho ,$ which is proper, lower semicontinuous, and satisfies those axioms, can be represented as follows:

$$
\varrho [ Z ] = \operatorname* { s u p } _ { \zeta \in \mathcal { A } _ { \varrho } } \langle \zeta , Z \rangle .\tag{1}
$$

The set $\mathcal { A } _ { \varrho } \subset \mathcal { L } _ { q } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } )$ with $\textstyle { \frac { 1 } { p } } + { \frac { 1 } { q } } = 1$ is defined as:

$$
\begin{array} { r } { \mathcal { A } _ { \varrho } = \Big \{ \zeta \in \mathcal { L } _ { q } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } ) \mid \displaystyle \int _ { \Omega } \zeta ( \omega ) d P ( \omega ) = \mu _ { \zeta } , \zeta \geq 0 \mathrm { a . s . , ~ } \langle \mathbb { I } , \mu _ { \zeta } \rangle = r \Big \} , } \end{array}
$$

where $r \in \mathbb { R }$ is a constant. The set $A _ { \varrho }$ is the convex subdifferential $\partial \varrho ( 0 )$ of the risk measure. A systemic measure of risk ϱ is normalized if $\varrho [ \mathbb { I } ] = 1$ , in which case, $r = 1$ for all $\zeta \in { \mathcal { A } } _ { \varrho } .$ . In the special case when $N = 1$ , we obtain the widely used dual representation of coherent measures of risk for scalar-valued random variables

$$
\ell [ Z ] = \operatorname* { s u p } _ { \frac { d Q } { d P } \in \mathcal { A } _ { \varrho } } \mathbb { E } _ { Q } [ Z ] ,\tag{2}
$$

where $\scriptstyle { \frac { d Q } { d P } }$ is the Radon-Nikodym derivative of the measure $Q$ with respect to the reference measure $P .$ The dual representation (1) shows the link between minimizing a coherent measure $\varrho ( Z )$ to a distributionally robust optimization (DRO) problem. The DRO problem with a loss function $L ( Z ( \theta , \omega ) )$ and an ambiguity set $\mathcal { A } \subset \mathcal { P } ( \Omega )$ is formulated as

$$
\operatorname* { m i n } _ { \theta } \operatorname* { s u p } _ { Q \in \mathcal { A } } \mathbb { E } _ { Q } \big [ L ( Z ( \theta , \omega ) ) \big ] .\tag{3}
$$

A widely used coherent risk measure for scalar-valued random variables is the well-known Mean - Upper Semideviation, which is defined as follows

$$
\varrho ( Z ) = \mathbb { E } [ Z ] + \kappa \big \vert \big \vert ( Z - \mathbb { E } [ Z ] ) _ { + } \big \vert \big \vert _ { p } , \mathrm { w h e r e ~ } \kappa \in [ 0 , 1 ] .
$$

Here $\| \cdot \| _ { p }$ stands for the p-norm in $\mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } )$ and $( a ) _ { + } = \operatorname* { m a x } ( 0 , a )$ . The Mean - Upper Semideviation has the dual representation (2) with the dual set

$$
\begin{array} { r } { { \cal A } _ { \varrho } = \left\{ Q \ll P : \frac { d Q } { d P } = \mathbb { I } + \zeta - \mathbb { E } [ \zeta ] \mathbb { I } , \ \| \zeta \| _ { q } \leq \kappa , \zeta \geq 0 \right\} , } \end{array}
$$

i.e, $Q$ are probability measures with densities w.r.to $P$ given by $\mathbb { I } + \boldsymbol { \zeta } - \mathbb { E } [ \boldsymbol { \zeta } ] \mathbb { I }$ . Notice that the parameter κ controls the risk-averse level of the objective; larger value of κ allows larger weight-differences in ζ, leading to a larger ambiguity set.

Consider a finite probability space $( \Omega _ { M } , \mathcal { F } _ { M } , \pi )$ , where $\Omega _ { M } = \{ 1 , \dots , M \}$ , π is a probability mass function, and ${ \mathcal { F } } _ { M }$ contains all subsets of $\Omega _ { M }$ . Given a collection of M measures of risk $\dot { \varrho } _ { i } : \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ) \to \mathbb { R } , i \in \Omega _ { M }$ , we associate with a random M-dimensional vector Z defined on $( \Omega , { \mathcal { F } } , P )$ , a random variable $R _ { Z }$ on the space $\Omega _ { M }$ as follows. The realizations of $R _ { Z }$ are

$$
R _ { Z } ( i ) = \varrho _ { i } ( Z ^ { i } ) , \quad i \in \Omega _ { M } .
$$

Let a coherent measure of risk $\varrho _ { 0 } : \mathcal { L } _ { \infty } ( \Omega _ { M } , \mathcal { F } _ { M } , \pi ) $ R be chosen as an aggregator. We define the measure of systemic risk $\varrho _ { \mathrm { s y s } } : \mathcal { L } _ { p } ( \Omega , \mathcal { F } , P ; \mathbb { R } ^ { M } ) \to \mathbf { \bar { R } }$ as follows:

$$
\underline { { \varrho } } _ { \mathrm { s y s } } ( Z ) = \varrho _ { 0 } ( R _ { Z } ) .\tag{4}
$$

This type of measure satisfies the axioms in Definition 1 as shown in [3].

We give two examples of systemic risk-measures that will become useful in our further developments.

We may choose as $\varrho _ { 0 }$ to be the mean-upper-semideviation risk measure of order q $( q \ge 1 )$ and evaluate all components of the error vector $Z$ by the same law-invariant coherent measure of risk $\varrho ( \cdot )$ . The description of the total risk evaluation is the following:

$$
\varrho _ { \mathrm { s y s } } ( Z ) = \sum _ { i \in \Omega _ { M } } \pi _ { i } \varrho [ Z ^ { i } ] + \kappa \bigg ( \sum _ { i \in \Omega _ { M } } \pi _ { i } \Big ( \varrho ( Z ^ { i } ) - \sum _ { i \in \Omega _ { M } } \pi _ { j } \varrho ( Z ^ { j } ) \Big ) ^ { q } + \bigg ) ^ { \frac { 1 } { q } }\tag{5}
$$

with $\kappa \in [ 0 , 1 ]$ . When $q = 1$ , the expression simplifies as follows

$$
\begin{array} { r l r } {  { \varrho _ { \mathrm { s y s } } ^ { 1 } ( Z ) = \sum _ { i \in \Omega _ { \cal M } } \pi _ { i } \bigg ( \varrho [ Z ^ { i } ] + \kappa \sum _ { j = 1 } ^ { M } \pi _ { j } ( \varrho [ Z ^ { i } ] - \varrho [ Z ^ { j } ] ) _ { + } \bigg ) } } \\ & { } & { = \displaystyle \sum _ { i \in \Omega _ { \cal M } } \pi _ { i } \varrho [ Z ^ { i } ] + \kappa \sum _ { \stackrel { i , j = 1 } { j \neq i } } ^ { M } \pi _ { i } \pi _ { j } | \varrho [ Z ^ { i } ] - \varrho [ Z ^ { j } ] | } \end{array}\tag{6}
$$

This representation shows that the individual risks of the components are aggregated with an additional penalty on the deviation of the individual risks from that average risk. This property is crucial to our treatment of fairness in classification.

Another example would be to aggregate the individual risks by a convex combination of the expected value and the Average Value-at-Risk at some level $\alpha ;$ again all components of the error vector $Z$ may be evaluated by the same law-invariant coherent measure of risk $\varrho ( \cdot )$ . For any $\kappa \in [ 0 , 1 ]$ ], this systemic measure of risk takes on the form:

$$
\varrho _ { \mathrm { s y s } } ^ { 2 } ( Z ) = ( 1 - \kappa ) \sum _ { i \in \Omega _ { M } } c _ { i } \varrho ( Z ^ { i } ) + \kappa \operatorname* { i n f } _ { \eta \in \mathbb { R } } \Big \{ \eta + \frac { 1 } { \alpha } \sum _ { i \in \Omega _ { M } } c _ { i } \big ( \varrho ( Z ^ { i } ) - \eta \big ) _ { + } \Big \} .\tag{7}
$$

Here, the infimum with respect to $\eta \in \mathbb { R }$ is taken over the individual risks of the components $\varrho ( Z ^ { i } ) , i \in \Omega _ { M }$ . Hence, this method of aggregation imposes additional penalty for the components whose risk exceeds a certain threshold, which can be shown to be the α -quantile of the distribution of $R _ { Z }$

We note that the linear aggregation is a special case of a systemic risk measure because one can choose $\varrho _ { 0 } ( R _ { Z } ) =$ $\begin{array} { r } { \mathbb E _ { \pi } ( R _ { Z } ) = \sum _ { i \in \Omega _ { M } } { \pi } _ { i } \varrho ( \breve { Z } ^ { i } ) } \end{array}$ . The expectation is the simplest coherent measure of risk. In each case, we could also use different risk measure for each component $Z ^ { i }$ , should we have a reason to do so.

## 3 Fairness notions and challenges

In order to discuss fairness, one needs to identify a definition agreeable for the application it is discussed.

Naturally, the fairness definitions and the associated algorithmic ideas are first developed in a binary classification scenario, as it answers most pertinent questions such as will a given loan be re-paid; is a project likely to success; is a treatment effective for the individual, etc.

For a binary decision, we consider fair-sensitive feature with values $\cal S = 1 , \dots , \cal S$ , which is present in the data points to be classified. The labels are $Y = 1$ (favorable) and $Y = 0$ (unfavorablle). The notion of demographic parity requires that the classifier outputs a decision $\hat { Y }$ such that the probability $P ( \hat { Y } = 1 | G = s )$ is the same across $s \in S .$ Here, $P$ is the frequency within the given dataset used as an estimation of the true conditional probability.

Another notion is so called equalized odds, which requires that the true positive rates and false positive rates are both equal, i.e. for all $s _ { 1 } , s _ { 2 } \in S _ { : }$ , the following relations are satisfied.

$$
P ( \hat { Y } = 1 | Y = 1 , G = s _ { 1 } ) = P ( \hat { Y } = 1 | Y = 1 , G = s _ { 2 } )
$$

$$
P ( \hat { Y } = 1 | Y = 0 , G = s _ { 1 } ) = P ( \hat { Y } = 1 | Y = 0 , G = s _ { 2 } ) .
$$

The notion of equal opportunity requires only the first equality, regarding the true positive rates (TPR).

The multi-class setting is less investigated and the binary situation cannot be carried to it in a straightforward manner, as there are no true positive/true negative predictions (or false positive/false negative). The two standard approaches are to define fairness per-class (one-vs-rest, giving a rate for each class × group and then aggregate. In some work the maximum difference across the various groups is taken and then averaged across classes, in others an average difference is calculated. In this way the fairness metric reduces to the classical one when there are two classes and two groups. In [10], the authors extend the definition of approximate fairness based on demographic parity to multi-class classification with two groups, i.e. $S = 2$ . They define an index of unfairness as the maximum difference between $P ( \hat { Y } = k | Y = 1 , G = s _ { 1 } )$ for k varying over the classes. The authors impose demographic-parity constraints and compute multi-class thresholds from the Lagrange multipliers of them.

In [2], the authors modify the notion of demographic parity, equalized odds, and overall accuracy equality into a multi-class, multiple-protected-group setting by post-processing method based on information projection. The most unified treatment of multi-class multi-group setting as far as we could see so far is based on optimal transport. It is established that demographic-parity fair prediction is equivalent to a certain Wasserstein-barycenter problem ( [8, 23], see also [22]). One phenomenon to be aware of when dealing with multiple groups is described in the paper [25]. It illustrates that treating one sensitive attribute at a time might lead to a model fair on each attribute but maximally unfair across intersectional groups. The authors propose auditing algorithms for subgroup fairness over combinatorially many subgroups defined by false-positive, false-negative, and classification rates.

The individual fairness is a recurring critique of group-fairness criteria because a model can satisfy them at the group level while treating individuals within a group inconsistently, arbitrarily, or even worse than before the fairness constraint was imposed. In [17], the authors introduce individual fairness — “treat similar individuals similarly,” formal ized as a Lipschitz condition on a task-specific similarity metric. Their motivating critique of demographic parity is that it constrains only group statistics and so permits within-group abuse: parity can be met by selecting the qualified members of one group and the unqualified members of another, leaving group rates equal but individuals treated incoherently. In [27] it is shown that fairness interventions optimized solely for group fairness and accuracy can exacerbate predictive multiplicity: the existence of many near-equivalent models that disagree on individual predictions. The authors find that some sort of arbitrariness is present in the predictions that could flip on retraining with a different random seed. They argue that “arbitrariness” should be considered alongside accuracy and group fairness.

In [26] the authors propose a metric multifairness framework to treat subpopulations similarly, where the subpopulations range over a rich, possibly overlapping class of comparison sets. This is the individual-fairness analogue of the multicalibration idea presented in [21], which asks for calibration to hold not just marginally per group but simultaneously across a rich family of overlapping subgroups; both are attempts to push a guarantee down from groups toward the individual level.

Measures for within-group unfairness using the inequality-index machinery known in economics are advocated in [34], where a model’s unfairness is quantified with a generalized-entropy index over a per-individual “benefit”. This index has the benefit of additivity, in which the overall individual-level unfairness decomposes into a between-group and a within-group component. Let us emphasize that classical inequality measures in exonomics such as the Gini coefficient are based on axiomatic foundations. The work [34] abuilds on it. The paper [38] proved that every fairness risk measure satisfying natural axioms is a coherent risk measure, establishing a connection to the risk measures used in our training formulation. Despite this progress, most of the fairness literature still reports point estimates of ad hoc metrics without statistical inference.

Finally, there is a discussion about the compatibility of individual and group fairness. In [40], the authors establish conditions under which optimal statistical-parity post-processing is compatible with the Lipschitz condition for individual fairness, introduced by [17] and, when the two conflict, identify the regions of the error–disparity Pareto frontier that still satisfy the individual-fairness requirement.

We start with a general definition. Consider a classifier ψ applied to data with true labels $Y$ taking values in $\{ 1 , \dots , M \}$ and a sensitive attribute G taking values in $S = \{ 1 , \dots , S \}$ . For each data point $X _ { i } ,$ , the classifier produces a prediction $\psi ( X _ { i } )$ . The prediction is a random variable whose distribution may depend on both the true label and the group membership. We say that the classifier $\psi$ is fair with respect to the sensitive attribute $G { \mathrm { i f } } ,$ conditioned on the true label, the distribution of the predictions does not depend on the group. That is, for each point $( Y , X )$ , the classifier ${ \hat { Y } } = \phi ( X )$ satisfies equality (8) for all classes $k , c \in \{ 1 , \ldots , M \}$ and for all group attributes $s \in S$

$$
P [ \hat { Y } = k \mid Y = c , G = s ] = P [ \hat { Y } = k \mid Y = c ] .\tag{8}
$$

In words, knowing which group a data point belongs to provides no additional information about the prediction it receives, beyond what is already known from its true label. This requires not only that each group achieves the same per-class accuracy, but that the entire pattern of correct and incorrect classifications be independent of group membership. This definition generalizes equalized odds from binary to multi-class setting. In comparison, [2] consider the true class Y and the group $s \in S$ and use a probabilistic classifier. They require that the entire per-group conditional confusion distribution match the group-marginal one by using a tolerance level α and setting:

$$
\left| \frac { P ( \hat { Y } = k \mid Y = c , G = s ) } { P ( \hat { Y } = k \mid Y = c ) } - 1 \right| \leq \alpha .
$$

So for each true class c and each predicted class k the probability of predicting k when the truth is c must be nearly the same in a group s as in the population. When the task is binary, this becomes equality of the false-positive and false-negative.

The definition (8) implies equal opportunity as a special case (take $c = 1$ and $k = 1 )$ , but it additionally requires the misclassification patterns to be group-independent. Two groups could satisfy equal opportunity while having very different error distributions across the remaining classes while under (8), this would still constitute unfairness. We note that in binary classification with $M = 2 .$ , the prediction distribution conditioned on $Y = 1$ is fully determined by the TPR, so equal opportunity and (8) conditioned on $Y = 1$ are equivalent.

Frequently, when the sensitive attribute takes $S > 2$ values (e.g., race, age, education, country, etc), the Equal Opportunity is typically computed as a ratio of the smallest True Positive Rate (TPR) over the largest one. This uses information from only the two most extreme groups and ignores what happens in between. Two classifiers could have the same EO-ratio but very different treatment of the intermediate groups. In a M-class setting, the issue is more fundamental: equal recall for each class does not imply (8), since two groups could have the same recall but very different distributions of errors across the other classes. The standard metrics, which summarize each group by a single number per class, cannot detect such differences. Most importantly, these metrics are reported as point estimates without uncertainty quantification. An EO-ratio of 0.85 might indicate a serious fairness violation if computed on 10000 samples, or it might be entirely consistent with a perfectly fair classifier if computed on 50 samples. The work [7] provides a comprehensive discussion of this gap, framing fairness auditing as a multiple hypothesis testing problem and showing that bootstrap-based inference can provide simultaneous confidence bounds across many subgroups.

We note also the work of [33], where it is proposed to measure unfairness via integral probability metrics, connecting fairness to the Wasserstein distance between group-conditional outcome distributions.

In addition to the equalized odds and equality ratios, we use the classical Gini index of inequality. Furthermore, we propose to evaluate the validity of the equalities in (8) by the classical statistical test of goodness of fit. The key observation is that the definition requires the conditional prediction distribution be the same across group. This is precisely the null hypothesis tested by Pearson’s chi-square test of homogeneity. Fix a true class $c \in \{ 1 , \ldots , M \}$ and consider the test samples with true label c. Each sample belongs to one of S groups and receives one of M possible predictions, giving a $\bar { M } \times S$ contingency table where cell $( k , s )$ contains the count $O _ { k s }$ of samples from group s that received prediction k. Under the null hypothesis the classifier map $\psi ( \cdot )$ satisfies:

$$
H _ { 0 } : \quad P [ \psi ( X _ { i } ) = k \mid Y _ { i } = c , G _ { i } = s ] = P [ \psi ( X _ { i } ) = k \mid Y _ { i } = c ] \quad { \mathrm { f o r ~ a l l ~ } } k , s ,
$$

the expected count in cell $( k , s )$ is

$$
E _ { k s } = \frac { R _ { k } \cdot n _ { s } } { n } ,
$$

where $\begin{array} { r } { R _ { k } = \sum _ { s } O _ { k s } } \end{array}$ is the total number of samples predicted as class $\begin{array} { r } { k , n _ { s } = \sum _ { k } O _ { k s } } \end{array}$ is the total number of samples in group s, and $\begin{array} { r } { \dot { N } = \sum _ { k , s } O _ { k s } } \end{array}$ is the grand total. This formula follows from the definition of statistical independence: under $H _ { 0 } .$ , the probability of falling into cell $( k , s )$ equals the product $P [ \psi ( X _ { i } ) = k \mid Y _ { i } = c ] \cdot P [ G _ { i } = s \mid Y _ { i } = c ]$ The Pearson chi-square statistic

$$
\chi ^ { 2 } = \sum _ { k = 1 } ^ { M } \sum _ { s = 1 } ^ { S } \frac { ( O _ { k s } - E _ { k s } ) ^ { 2 } } { E _ { k s } }\tag{9}
$$

follows a $\chi ^ { 2 } \big ( ( M - 1 ) ( S - 1 ) \big )$ distribution asymptotically under $H _ { 0 }$ . A large value indicates that the prediction distributions differ significantly across groups, providing statistical evidence of unfairness.

For a multi-class problem, we construct one contingency table per true class and obtain M chi-square statistics with corresponding p-values. To control the family-wise error rate, we apply the Bonferroni correction: the classifier is declared fair at level α if all M p-values exceed $\alpha / M$ . That being said, if we are defining the fairness in a statistical parity style, where we do not care about a sample’s true label, but only care about the probability of its prediction, the test would be even easier since we only need one test overall instead of one test per class.

## 4 Contextual risk and fairness

We proposed risk-averse formulation based on contextual risk and aggregation of risk by outer risk measure with specific properties. For example, the mean-upper semi-deviation risk measure used as aggregator suggests that fairness within the learning model between the classes is forced. This can be categorized into the modifying learner/objective methods we discussed in the literature review. We have seen models that directly modify the loss function, such as [33], which uses a penalty term on the distance between the distributions of predictions within the two groups. Other works use extra constraints to force fairness, e.g. [15]. Our two-stage model naturally involves an implicit penalty term within the formulation of the outer risk measure, If we use the mean-semideviation risk measure as aggregator, then it penalizes the deviation of the risk for each class from the average risk. When the Average Value-at-Risk at level $\alpha \in ( 0 , 1 )$ is involved in the objective function, then the deviation of the risk for each class from the α-quantile of the risk distribution to classes is penalized. In this way there is some similarity to [33]. However, one issue arises: when considering sensitive groups or attributes, we are indeed assessing the risk with conditions; therefore, the behavior and properties of such conditional risks need to be well-defined.

We consider the scenario in which a fair-sensitive categorical feature with values $s = 1 , \ldots , S$ is present in the data points to be classified. To make the prediction fair with respect to the groups within all classes, we introduce a contextual risk-measure $\varrho _ { c } [ Z ]$ . Assume that $Z = f ( \vartheta , X , Y , \dot { G } )$ , where $\vartheta$ determines the classifier, X stands for a random data point and $Y$ stands for the class label and G indicates the value of said feature. Let the function $f : \mathbb { R } ^ { 2 n } \times \mathcal { I } \times \dot { \mathcal { S } } \to \mathbb { R } ^ { M }$ be convex and monotonically non-decreasing with respect to the second argument for any value of the other arguments. We need to consider the classification error $Z ^ { i }$ in every context $s \in S$ for all classes $i \in \mathcal { I }$ . Consider the probability space $( \Omega , { \mathcal { F } } , P _ { X Y } )$ where $( X , Y )$ are defined. The measure $P _ { X Y }$ is disintegrated into the marginal distribution π of Y and the conditionals $P _ { i }$ for $X | Y = i$ . We consider the spaces

$$
\mathcal { V } _ { i , s } = \big ( \Omega , \mathcal { F } \cap \{ G = s \} , P _ { i s } \big ) ,
$$

where $P _ { i s }$ is the $P _ { i }$ -induced probability measure on ${ \mathcal { F } } \cap \{ G = s \}$ . A contextual risk measure is a set of conditional coherent measures of risk evaluating the risk given a context. An aggregation measure provids the total risk for all contexts. More precisely, we fix a real-valued coherent risk measures $\bar { \rho } _ { i , s } [ Z | Y = i , \bar { G } = s ]$ for each $i \in \mathcal { I }$ and $s \in { \mathcal { S } }$ defined on $\mathcal { D } _ { i , s }$ and choose $\varrho _ { c } [ Z ^ { i } ]$ as an aggregation measure to obtain the total risk of each class over all contexts. Additional aggregation using an outer risk measure $\varrho _ { 0 } [ Z ]$ will aggregate the risk over all the classes. We denote the distribution of all classes in the entire population by π, i.e., π is a vector with M positive components such that $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \pi _ { i } = 1 } \end{array}$

Proposition 2. Let $\rho _ { i , s } [ \cdot ]$ be coherent measures of risk for all $\textit { i } \in \textit { J }$ and all $s \in \ S ,$ , risk measures $\varrho _ { i } :$ $\mathcal { L } _ { \infty } ( S , \mathcal { F } _ { S } , P _ { i } | G = s ) \stackrel { \left. } { \right. }$ R be coherent measures of risk for all $i \in \mathcal { I }$ , and $\varrho _ { 0 } : \mathcal { L } _ { \infty } ( \Omega _ { M } , \mathcal { F } _ { M } , \pi )  \mathbb { R }$ be coherent as well. Denote

$$
V _ { c } ( i , s ) = \rho _ { i , s } [ Z ^ { i } | G = s ] \quad a n d \quad W ( i ) = \varrho _ { i } [ V _ { c } ( i , \cdot ) ] , \quad s \in \mathcal { S } , \ i \in \mathcal { I } .
$$

Then the risk measure $\varrho _ { s y s } [ Z ] = \varrho _ { 0 } [ W ]$ satisfies axioms A1-A4 ofsystemic measure ofrisk.

Proof. (i) Given any $Z _ { 1 } , Z _ { 2 }$ and $\alpha \in ( 0 , 1 )$ , we consider the random vector $Z _ { 3 } = \alpha Z _ { 1 } + ( 1 - \alpha ) Z _ { 2 }$ . It follows that

$$
V _ { c } ^ { 3 } ( i , s ) = \rho _ { i , s } [ Z _ { 3 } ^ { i } | Y = s ] \leq \alpha \rho _ { i , s } [ Z _ { 1 } ^ { i } | Y = s ] + ( 1 - \alpha ) \rho _ { i , s } [ Z _ { 2 } ^ { i } | Y = s ]
$$

$$
= \alpha V _ { c } ^ { 1 } ( i , s ) + ( 1 - \alpha ) V _ { c } ^ { 2 } ( i , s ) .
$$

by the convexity of $\rho _ { i , s } [ \cdot ]$ for all $i \in \mathcal { I }$ and all $s \in S$ . Hence

$$
W ^ { 3 } ( i ) = \varrho _ { i } [ V _ { c } ^ { 3 } ( i , \cdot ) ] \leq \alpha \varrho _ { i } [ V _ { c } ^ { 1 } ( i , \cdot ) ] + ( 1 - \alpha ) \varrho _ { i } [ V _ { c } ^ { 2 } ( i , \cdot ) ] = \alpha W ^ { 1 } ( i ) + ( 1 - \alpha ) W ^ { 2 } ( i ) .
$$

by the same arguments. Analogously,

$$
\varrho _ { \mathrm { s y s } } [ Z _ { 3 } ] = \varrho _ { 0 } [ W ^ { 3 } ] \leq \alpha \varrho _ { 0 } [ W ^ { 1 } ] + ( 1 - \alpha ) \varrho _ { 0 } [ W ^ { 2 } ] = \alpha \varrho _ { \mathrm { s y s } } [ Z _ { 1 } ] + ( 1 - \alpha ) \varrho _ { \mathrm { s y s } } [ Z _ { 2 } ] ,
$$

which establishes the convexity property.

(ii) Suppose the vectors $Z _ { 1 } , Z _ { 2 }$ satisfy $Z _ { 1 } \le Z _ { 2 } \mathrm { a . s }$ . This implies that $Z _ { 1 } ^ { i } \leq Z _ { 2 } ^ { i }$ a.s. Using the monotonicity of the risk measures $\rho _ { i , s } [ \cdot ] , \varrho _ { i } [ \cdot ]$ and $\varrho _ { 0 } [ \cdot ]$ , we infer the monotonicity of $\varrho _ { \mathrm { s y s } } [ \cdot ]$

(iii) The positive homogeneity follows in a straightforward manner from the definition.

(iv) Given a random vector $Z$ and a real constant a, we calculate the risk ${ \cal V } _ { c } ^ { + } ( i , s )$ of the translated error in context s as follows:

$$
V _ { c } ^ { + } ( i , s ) = \rho _ { i , s } [ Z ^ { i } + a | G = s ] = \rho _ { i , s } [ Z ^ { i } | G = s ] + a = V _ { c } ( i , s ) + a \quad \mathrm { f o r ~ a l l ~ } s \in \mathcal { S } , \ i \in \mathcal { I } .
$$

The second equality holds by the translation property of the coherent measures of risk. This implies that $W ^ { + } ( i ) =$ $\varrho _ { i } [ V _ { c } ^ { + } ( i , \cdot ) ] \stackrel { - } { = } \varrho _ { i } [ \dot { V _ { c } } ( i , \cdot ) + \stackrel { . } { a } ] = W ( i ) + a$ by the same argument. Hence, $\varrho _ { 0 } [ W + a ] = \varrho _ { 0 } [ W ] + a$ , concluding that property (A4) holds as well. □

This structure of measuring risk allows us to enforce fairness within each class by choosing the aggregate measures $\varrho _ { i }$ to be such that deviation from the average risk or excessive risk above a certain quantile is penalized.

As a class-level aggregator $\varrho _ { c } ,$ , we use the mean-semideviation risk measure. If applied to two classes: A and $B ,$ with one sensitive context valued in $S = \{ 1 , 2 \}$ , it leads to the representation:

$$
\varrho _ { c } = \pi _ { c , 1 } \varrho _ { c | 1 } + \pi _ { c , 2 } \varrho _ { c | 2 } + \kappa _ { \mathrm { m i d } } \pi _ { c , 1 } \pi _ { c , 2 } | \varrho _ { c | 1 } - \varrho _ { c | 2 } | , \quad c = A , B .
$$

Here $p _ { c , s }$ is the conditional weight of group s within class c and $\kappa _ { \mathrm { m i d } } \in [ 0 , 1 ]$ . This middle-level aggregation penalizes within-class group disparity: when the risks $\varrho _ { c | 1 }$ and $\varrho _ { c | 2 }$ differ, the semideviation term increases the total class risk, discouraging the classifier from achieving low risk for one group at the expense of the other.

For the outer aggregation $\varrho _ { 0 }$ , one could simply use the expected value $\varrho _ { 0 } = \pi _ { A } \varrho _ { A } + \pi _ { B } \varrho _ { B }$ , which balances the group-regularized class risks. However, a linear outer aggregation does not penalize imbalance between the class-level risks — a classifier that achieves low risk for one class but high risk for the other is not penalized beyond the weighted average. To address this, we propose using a non-linear aggregation, $\mathrm { e . g }$ , the mean semideviation, at the outer level a well:

$$
\varrho _ { 0 } = \pi _ { A } W ( A ) + \pi _ { B } W ( B ) + \kappa _ { \mathrm { o u t } } \pi _ { A } \pi _ { B } | W ( A ) - W ( B ) | ,
$$

with $\kappa _ { \mathrm { o u t } } \in [ 0 , 1 ]$ . This gives a fully nonlinear three-level risk structure: sample losses within each scenario are aggregated by the inner risk $\rho _ { i , s }$ , scenario risks within each class are aggregated by the middle MSD with parameter $\kappa _ { \mathrm { m i d } } .$ , and class risks are aggregated by the outer MSD with parameter $\kappa _ { \mathrm { o u t } }$ . By the Proposition above, this three-level composition remains a coherent systemic risk measure.

## 5 Three-stage regularized decomposition

We now describe a regularized decomposition method for soling the three-level nonlinear risk structure introduced above. Recall that all functions and risk measures involved depend on the parameter ϑ of the classifier. The overall objective is

$$
\begin{array} { c } { \rho _ { \mathrm { s y s } } ( \vartheta ) = \varrho _ { 0 } \big [ W ( 1 ; \vartheta ) , \ldots , W ( M ; \vartheta ) \big ] , } \\ { W ( i ; \vartheta ) = \varrho _ { i } \big [ V _ { c } ( i , 1 ; \vartheta ) , \ldots , V _ { c } ( i , S ; \vartheta ) \big ] , \quad V _ { c } ( i , s ; \vartheta ) = \rho _ { i , s } [ Z ^ { i } ( \vartheta ) | G = s ] . } \end{array}
$$

For the outer risk $\varrho _ { 0 }$ and for the class aggregators $\varrho _ { i } [ ( i , \cdot , \vartheta ) ]$ , the dual representations are

$$
\varrho _ { 0 } [ W ; \vartheta ] = \operatorname* { m a x } _ { \mu \in \mathcal { A } _ { 0 } } \sum _ { i = 1 } ^ { M } \mu _ { i } W ( i ; \vartheta ) .\tag{10}
$$

$$
\varrho _ { i } [ V ( i , \cdot ; \vartheta ) ] = \operatorname* { m a x } _ { \zeta ^ { i } \in \mathcal { A } _ { i } } \sum _ { s \in \mathcal { S } } \zeta _ { s } ^ { i } V ( i , s ; \vartheta ) ,\tag{11}
$$

The problem with non-linear aggregation takes on the form:

$$
\operatorname* { m i n } _ { \vartheta , \alpha } \alpha + \sigma \sum _ { i \in \mathcal { I } } \left. v ^ { i } \right. ^ { 2 } \quad \mathrm { s . t } \ \alpha \geq \sum _ { i = 1 } ^ { M } \mu _ { i } W ( i ; \vartheta ) \quad \forall \mu \in \mathcal { A } _ { 0 } ;\tag{12}
$$

$$
W ( i , \vartheta ) = \operatorname* { m i n } _ { q _ { i } } q _ { i } \quad \mathrm { s . t } q _ { i } \geq \sum _ { s \in \mathcal { S } } \zeta _ { s } ^ { i } V ( i , s ; \vartheta ) \quad \forall \zeta ^ { i } \in \mathcal { A } _ { i }\tag{13}
$$

$$
\begin{array} { r l } { V ( i , s , \vartheta ) = \underset { Z ^ { i } } { \operatorname* { m i n } } } & { \varrho _ { i , s } [ Z ^ { i } | G = s ] } \\ { \mathrm { s . t . } } & { z _ { \ell } ^ { i } \geq \psi ( x _ { \ell } ^ { i } ; \vartheta _ { i } ) - \psi ( x _ { \ell } ^ { i } ; \vartheta _ { j } ) + 1 \quad j \in \mathcal { T } ^ { - i } , \ell \in \{ 1 , \ldots , m _ { i } : G = i \} } \\ & { Z ^ { i } \geq 0 . } \end{array}\tag{14}
$$

In the two-stage problem treated in [13], the outer aggregation $\varrho _ { 0 }$ is applied directly to the class risks $R _ { i } ( \vartheta )$ , and the master problem maintained two sets of cuts: objective cuts approximating $\varrho _ { 0 }$ and scenario cuts approximating each $R _ { i } ( \cdot )$ . With the contextual risk structure, we have an additional middle level: the class risks $W ( i ; \bar { \vartheta } )$ are themselves nonlinear aggregations via $\varrho _ { i }$ of the contextual risks $V _ { c } ( i , s ; \vartheta )$

The construction of cuts at each level follows from the dual representation. At each iteration $k ,$ we evaluate the class risks $W ^ { k } ( i ; \vartheta )$ and identify the maximizer $\mu ^ { k } \in \mathcal { A } _ { 0 }$ for the random variable $W ^ { k } ( \cdot ; \vartheta )$ that has realizations $W ^ { k } ( i ; \vartheta )$

with probabilities $\begin{array} { r } { \frac { m _ { i } } { N } , N = \sum _ { i = 1 } ^ { M } m _ { i } } \end{array}$ . This provides the outer cut

$$
\alpha \geq \sum _ { i = 1 } ^ { M } \mu _ { i } ^ { k } q _ { i } ,\tag{15}
$$

where $q _ { i }$ are variables approximating the class risks. This is the same type of cut as the objective cut in [13], but now acting on the class-level variables $q _ { i }$ rather than the scenario-level variables. Similarly, for each class-level aggregator $\varrho _ { i }$ , at each iteration k, we evaluate the scenario risks $R _ { i s } ^ { k } = V ( i , s : \vartheta )$ for $s \in S$ and identify the maximizer $\bar { \zeta } ^ { i , k } \in \mathcal { A } _ { i }$ This provides the class-level cut

$$
q _ { i } \geq \sum _ { s \in \mathcal { S } } \zeta _ { s } ^ { i , k } r _ { i s } ,\tag{16}
$$

where $r _ { i s }$ are variables approximating the contextual risks. These cuts are the new ingredient compared to the method in [13]: they approximate each $\varrho _ { i }$ from below as a function of the scenario risks.

At the bottom level, the scenario cuts are constructed exactly as in [13]. For each scenario $( i , s )$ , we solve the subproblem (14) to obtain the scenario risk $R _ { i s } ^ { k }$ and the subgradient $g _ { i s } ^ { k }$ of $V ( i , s ; \vartheta )$ with respect to ϑ, which provides the cut

$$
r _ { i s } \ge R _ { i s } ^ { k } + \langle g _ { i s } ^ { k } , \vartheta - \vartheta ^ { k } \rangle .\tag{17}
$$

The regularized master problem at iteration k takes the form:

$$
\begin{array} { r l r } { \underset { \vartheta , \alpha } { \operatorname* { m i n } } } & { \alpha + \sigma \| v \| ^ { 2 } + \beta \| \vartheta - w ^ { k } \| ^ { 2 } } \\ { \mathrm { s . t . } } & { \alpha \geq \sum _ { i = 1 } ^ { M } \mu _ { i } ^ { \kappa } \bar { q } _ { i } , } & { \kappa \in \mathcal { K } _ { 0 } , } \\ & { \alpha \geq 0 . } \end{array}\tag{18}
$$

For each $i \in \mathcal { I }$ , we consider the problem

$$
\begin{array} { r l r } { \bar { q } _ { i } = \underset { r , q , \alpha } { \operatorname* { m i n } } } & { q _ { i } } & \\ { \mathrm { s . t . } } & { q _ { i } \geq \sum _ { s \in S } \zeta _ { s } ^ { i , \kappa } r _ { i s } , } & { \kappa \in \mathcal { K } _ { i } , } \\ & { r _ { i s } \geq R _ { i s } ^ { \kappa } + \langle g _ { i s } ^ { \kappa } , \vartheta - \vartheta ^ { \kappa } \rangle , } & { \kappa \in \mathcal { K } _ { i s } , s \in \mathcal { S } , } \\ & { r _ { i s } , q _ { i } \geq 0 \quad s \in \mathcal { S } . } \end{array}\tag{19}
$$

Here $\sigma > 0$ is the regularization parameter, $\beta > 0$ is the proximal parameter, and $w ^ { k }$ is the current stability center.   
The sets $\kappa _ { 0 } , \kappa _ { i }$ , and $\bar { \boldsymbol { \kappa } } _ { i s }$ collect the indices of the outer, class-level, and scenario cuts, respectively.

We propose the following three-level regularized multi-cut method with a parameter $\delta \in ( 0 , 1 )$

Three-stage Classification with Contextual Systemic Risk   
Step 0. Set $k = 1 ,$ . Choose initial decision variable $\vartheta ^ { 1 }$   
Step 1. For each $( s , i ) , s \in S ; i \in \mathcal { I } ,$ solve the third stage subproblem (14). Denote its optimal value by $R _ { i s } ^ { k }$ and   
calculate the subgradient $g _ { i s } ^ { k }$ . Add the new scenario cuts to $\boldsymbol { \kappa } _ { i s }$   
Step 2. For each class $i \in \mathcal { I } ,$ , evaluate the class-level risk $W ^ { k } ( i ) = \varrho _ { i } [ R _ { i s } ^ { k } : s \in \mathcal { S } ]$ by solving (19) and calculate   
$\zeta ^ { i , k }$ using (11). Add the new class-level cuts to $\textstyle \mathcal { K } _ { i } , i \in \mathcal { I }$   
Step 3. Calculate the systemic risk $\varrho _ { \mathrm { s y s } } ^ { k } = \varrho _ { 0 } [ W ^ { k } ( \cdot ) ]$ and calculate $\mu ^ { k }$ using (10). Add the new outer cut to $\kappa _ { 0 }$   
Step 4. Determine the new center $w ^ { k }$ as follows. If $k = 1$ or   
$\varrho _ { \mathrm { s y s } } ^ { k } \leq ( 1 - \delta ) \bar { \varrho } ^ { k - 1 } + \delta \alpha ^ { k - 1 } ,$   
then set $w ^ { k } = \vartheta ^ { k }$ and $\bar { \varrho } ^ { k } = \varrho _ { \mathrm { s y s } } ^ { k }$ (descent step). Otherwise, set $w ^ { k } = w ^ { k - 1 }$ and $\bar { \varrho } ^ { k } = \bar { \varrho } ^ { k - 1 }$ (null step).   
Step 5. Solve the master problem (18). Denote the solution by $\alpha ^ { k } , \vartheta ^ { k }$   
Step 6. If $\bar { \varrho } ^ { k } = \alpha ^ { k }$ , then stop $( w ^ { k }$ is an optimal solution); otherwise continue.   
Step 7. Remove from the sets $\kappa _ { 0 } , \kappa _ { i } ,$ and $\boldsymbol { \kappa } _ { i s }$ the constraints whose Lagrange multipliers are 0 at the solution of   
(19) and (18). Increase k by 1 and go to Step 1.

We observe that the second stage problem only calculates the best approximation of the risk of individual classes and does not change the classification parameter. Using the monotonicity property of the risk measures involved and the preserving the hierarchical structure, we may collapse the first and the second stage into one. In that case, we accumulate all three sets of cuts in one master problem. The regularized master problem at iteration k takes the form:

```latex
$\operatorname* { m i n } _ { \vartheta , r , q , \alpha } \quad \alpha + \sigma \| v \| ^ { 2 } + \beta \| \vartheta - w ^ { k } \| ^ { 2 }$
$\begin{array} { r } { \mathrm { s . t . } \quad \alpha \geq \sum _ { i = 1 } ^ { N } \mu _ { i } ^ { \kappa } q _ { i } , } \end{array}$ $\kappa \in { \cal K } _ { 0 } ,$
$\begin{array} { r } { q _ { i } \geq \sum _ { s \in \mathcal { S } } \zeta _ { s } ^ { i , { \kappa } } r _ { i s } , } \end{array}$ $\kappa \in { \mathcal { K } } _ { i } , i \in { \mathcal { I } } ,$
$r _ { i s } \geq R _ { i s } ^ { \kappa } + ( g _ { i s } ^ { \kappa } ) ^ { \top } ( \vartheta - \vartheta ^ { \kappa } ) , \quad \kappa \in { \mathcal { K } } _ { i s } , \ s \in { \mathcal { S } } , i \in { \mathcal { I } } ,$
$\alpha , r _ { i s } , q _ { i } \ge 0 \quad s \in \mathcal { S } , i \in \mathcal { I } .$
```

(20)

The three-stage method now is modified to the following three-level regularized multi-cut method.

Three-level Classification with Contextual Systemic Risk   
Step 0. Set $k = 1 .$ . Choose initial decision variable $\vartheta ^ { 1 }$   
Step 1. For each $( s , i ) , s \in S ; i \in \mathcal { I } ,$ solve the third stage subproblem (14). Denote its optimal value by $R _ { i s } ^ { k }$ and   
calculate the subgradient $g _ { i s } ^ { k } .$ . Add the new scenario cuts to $\boldsymbol { \kappa } _ { i s }$   
Step 2. For each class $i \in \mathcal { I } ,$ evaluate the class-level risk $W ^ { k } ( i ) = \varrho _ { i } [ R _ { i s } ^ { k } : s \in \mathcal { S } ]$ and calculate $\zeta ^ { c , k }$ using (11).   
Add the new class-level cuts to $\qquad K _ { i } , i \in { \mathcal { I } } .$   
Step 3. Calculate the systemic risk $\varrho _ { \mathrm { s y s } } ^ { k } = \varrho _ { 0 } [ W ^ { k } ( \cdot ) ]$ and calculate $\mu ^ { k }$ using (10). Add the new outer cut to $\kappa _ { 0 }$   
Step 4. Determine the new center $w ^ { k }$ as follows. I $\mathrm { ~ f ~ } k = 1$ or   
$\varrho _ { \mathrm { s y s } } ^ { k } \leq ( 1 - \delta ) \bar { \varrho } ^ { k - 1 } + \delta \alpha ^ { k - 1 } ,$   
then set $w ^ { k } = \vartheta ^ { k }$ and $\bar { \varrho } ^ { k } = \varrho _ { \mathrm { s y s } } ^ { k }$ (descent step). Otherwise, set $w ^ { k } = w ^ { k - 1 }$ and $\bar { \varrho } ^ { k } = \bar { \varrho } ^ { k - 1 }$ (null step).   
Step 5. Solve the master problem (20). Denote the solution by $\alpha ^ { k } , \vartheta ^ { k } , r ^ { k } , q ^ { k }$   
Step 6. If $\bar { \varrho } ^ { k } = \alpha ^ { k } .$ , then stop $( w ^ { k }$ is an optimal solution); otherwise continue.   
Step 7. Remove from the sets $\kappa _ { 0 } , \kappa _ { i } ,$ and $\boldsymbol { \kappa } _ { i s }$ the constraints whose Lagrange multipliers are 0 at the solution of   
(20). Increase k by 1 and go to Step 1.

Observe that the convergence of both methods follows from the convergence theorem of the regularized decomposition method in [12].

## 6 Numerical Experiments with Fair Risk-averse Classification

We now test our method on two datasets, the Adult dataset [5] and the ACSPublicCoverage dataset from the Folktables benchmark suite [14], for two different purposes. On first one, we study what happens when the recorded sensitive attribute is wrong by adding noises to the label. On the second one, since the ACSPublicCoverage data is recorded in different states, we can study what happens when the test set distribution is shifted from the training set. On both datasets we use a binary sensitive attribute (sex) and a multi-group one (race), and then, on the public coverage data, we use an intersectional attribute (sex × race), so we can thoroughly observe the behavior of the methods on different numbers of protected groups.

We compare CNACR against a few baseline methods: a plain linear SVM, a covariance-constrained classifier from [41], a fair empirical risk minimization from [15], the reductions approach from [1] with EO constraints. All of them use a linear SVM as the base classifier, so if we see any difference, we know that it comes from the fairness-enforcing method but not from the classification model. Note that the method of [15] is only defined for two protected groups, so we use it only in the experiments with sex as the sensitive attribute. We sweep through the parameter configurations for each baseline method and report the best one of each family in the tables. For CNACR we use fixed parameters $\kappa _ { \mathrm { i n n e r } } = 0 . 1 , \kappa _ { \mathrm { m i d } } = 1 . 0 , \kappa _ { \mathrm { o u t } } = 0 . 5 .$

We report the EO ratio as one of the fairness metrics. We argued previously that this mainstream metric is limited to reflecting the fairness between the two extreme groups when more than two groups are involved, so we also report the Gini coefficient and the goodness-of-fit test results.

## 6.1 Adult: Corrupted Sensitive Attributes

The Adult dataset has 48842 samples, and the task is to predict whether an individual earns more than \$50K per year. We first use sex as the sensitive attribute, which is binary. Then we use race, grouped into White, Black, and Other with proportions 85%/10%/5%, which allows us to run the experiment in an imbalanced multi-group setting.

Setup. For each of 100 independent runs, we randomly sample 20000 data points from the full dataset proportionally across classes and genders, and split them into 70% training and 30% test. By ’proportionally’, we mean that the sample keeps the same proportion of each class and each group as they are in the whole dataset. To simulate corrupted sensitive attributes, we randomly flip the sensitive attribute of 20% of the training samples and keep the test set clean. For example, in the experiment with sex as the sensitive group, we randomly pick 20% of the training samples, among which males become females and vice versa. All methods are trained on the same contaminated training set and evaluated on the same clean test set in each run, so each comparison is paired. We repeat this for 100 times at 0% and 20% mislabeling respectively, and report both results.

Results. Tables 1 and 2 give the results of the experiment using sex as sensitive attribute. We report the average on all metrics ± standard deviation over the 100 runs, and the best value in each row is bolded. Since each method uses the same train and test set in every run, the results are paired, and we run paired t-tests to see if the differences of the metrics are statistically significant.
<table><tr><td></td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>EO-ratio</td><td> $0 . 8 3 2 8 \pm 0 . 0 5 9$ </td><td> $\mathbf { 0 . 9 5 3 3 \pm 0 . 0 3 2 }$ </td><td> $0 . 9 4 8 0 \pm 0 . 0 3 8$ </td><td> $0 . 9 2 7 0 \pm 0 . 0 5 5$ </td><td> $0 . 9 3 9 1 \pm 0 . 0 4 3$ </td></tr><tr><td>0i o0% Gini(TPR)</td><td> $0 . 0 2 1 9 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 0 6 2 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 0 6 9 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 9 5 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 8 0 \pm 0 . 0 0 6$ </td></tr><tr><td>F1</td><td> $0 . 7 8 1 3 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 7 8 6 6 \pm 0 . 0 0 6 }$ </td><td> $0 . 7 7 5 1 \pm 0 . 0 0 6$ </td><td> $0 . 7 8 0 1 \pm 0 . 0 0 6$ </td><td> $0 . 7 7 8 6 \pm 0 . 0 0 6$ </td></tr><tr><td> $\breve { Z } _ { \chi ^ { 2 } \mathrm { r e j . } }$ </td><td>76.0%</td><td>8.0%</td><td>6.0%</td><td>20.0%</td><td>12.0%</td></tr><tr><td>EO-ratio</td><td> $0 . 8 3 2 8 \pm 0 . 0 5 9$ </td><td> $0 . 9 4 6 3 \pm 0 . 0 4 0$ </td><td> $\mathbf { 0 . 9 4 6 6 \pm 0 . 0 3 9 }$ </td><td> $0 . 9 0 4 7 \pm 0 . 0 7 2$ </td><td> $0 . 9 0 2 5 \pm 0 . 0 6 4$ </td></tr><tr><td>Iise 20% Gini(TPR)</td><td> $0 . 0 2 1 9 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 0 7 1 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 0 0 7 1 \pm 0 . 0 0 5 }$ </td><td> $0 . 0 1 2 4 \pm 0 . 0 0 9$ </td><td></td></tr><tr><td>F1</td><td></td><td></td><td></td><td></td><td> $0 . 0 1 2 7 \pm 0 . 0 0 8$ </td></tr><tr><td></td><td> $0 . 7 8 1 3 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 7 8 7 1 \pm 0 . 0 0 6 }$ </td><td> $0 . 7 7 5 2 \pm 0 . 0 0 6$ </td><td> $0 . 7 8 0 0 \pm 0 . 0 0 6$ </td><td> $0 . 7 7 9 3 \pm 0 . 0 0 7$ </td></tr><tr><td> ${ \frac { \circ } { z } } \chi ^ { 2 } { \mathrm { ~ r e j . } }$ </td><td>76.0%</td><td>13.0%</td><td>8.0%</td><td>33.0%</td><td>33.0%</td></tr></table>

Table 1: Adult with sex as the sensitive attribute (mean ± standard deviation over 100 runs). $\overline { { { \chi } ^ { 2 } { \bf \Psi } ^ { \mathrm { r e j . } } } }$ . is the fraction of runs in which equal TPR across groups is rejected at $\alpha = 0 . 0 5 .$ On this attribute, CNACR and the covarianceconstrained classifier (Zafar) are statistically indistinguishable on both fairness measures; the bolding marks the better mean but with no significant difference.

<table><tr><td></td><td></td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>FL-EOp</td></tr><tr><td rowspan="4">i 00%</td><td>EO-ratio</td><td> $0 . 7 4 3 3 \pm 0 . 0 9 7$ </td><td> $\mathbf { 0 . 8 7 5 6 \pm 0 . 0 7 2 }$ </td><td> $0 . 8 5 8 1 \pm 0 . 0 7 5$ </td><td> $0 . 8 3 2 2 \pm 0 . 0 9 6$ </td></tr><tr><td>Gini(TPR)</td><td> $0 . 0 1 3 9 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 0 6 6 } \pm \mathbf { 0 . 0 0 4 }$ </td><td> $0 . 0 0 8 2 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 9 2 \pm 0 . 0 0 5$ </td></tr><tr><td>F1</td><td> $0 . 7 8 1 3 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 7 8 9 2 \pm 0 . 0 0 6 }$ </td><td> $0 . 7 7 9 8 \pm 0 . 0 0 6$ </td><td> $0 . 7 8 0 2 \pm 0 . 0 0 5$ </td></tr><tr><td> $\chi ^ { 2 } \ \mathrm { r e j . }$ </td><td>54.0%</td><td>10.0%</td><td>12.0%</td><td>16.0%</td></tr><tr><td rowspan="4">Noie 220%</td><td>EO-ratio</td><td> $0 . 7 4 3 3 \pm 0 . 0 9 7$ </td><td> $\mathbf { 0 . 8 7 3 6 \pm 0 . 0 7 3 }$ </td><td> $0 . 8 5 4 8 \pm 0 . 0 7 9$ </td><td> $0 . 7 9 3 8 \pm 0 . 1 1 1$ </td></tr><tr><td>Gini(TPR)</td><td> $0 . 0 1 3 9 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 0 6 7 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 0 8 4 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 1 3 \pm 0 . 0 0 6$ </td></tr><tr><td>F1</td><td> $0 . 7 8 1 3 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 7 8 9 3 \pm 0 . 0 0 5 }$ </td><td> $0 . 7 7 9 7 \pm 0 . 0 0 6$ </td><td> $0 . 7 8 0 4 \pm 0 . 0 0 6$ </td></tr><tr><td> $\chi ^ { 2 } \ \mathrm { r e j . }$ </td><td>54.0%</td><td>18.0%</td><td>14.0%</td><td>35.0%</td></tr></table>

Table 2: Adult with race as the sensitive attribute (mean ± standard deviation over 100 runs). CNACR shows better EO-ratio, Gini and F1 than all baselines with statistical significance $( p < 0 . 0 1 )$

With race as the sensitive attribute (Table 2), CNACR shows significant advantage. On clean data, compared to the plain SVM, it raises the EO-ratio from 0.7433 to 0.8756, more than halves the Gini coefficient from 0.0139 to 0.0066, and greatly reduces the chi-square rejection rate from a random 54% to 10%. In a paired t-test, the advantage is significant with $p < 1 0 ^ { - 4 }$ , with CNACR higher in 92 out of the 100 runs.

The closest baseline is the covariance-constrained classifier (Zafar), and the two behave differently in the two experiments. In the race experiment, the paired differences favor CNACR on the EO-ratio $( + 0 . 0 1 7 5 , p = 0 . 0 0 4 )$ and on the Gini coefficient $( - 0 . { \overset { \cdot } { 0 } } 0 1 7 , p < 1 0 { \overset { \cdot } { - } } 3 )$ . However, the two methods are statistically indistinguishable on both fairness measures in the sex experiment. But against every other baseline, and against the plain SVM, CNACR’s fairness differences are still significant at the 0.05 level on both attributes and at both corruption levels. This indicates that CNACR might show a greater advantage compared to the baseline methods as the number of attributes increases. The same pattern will repeat in future experiments.

This improvement in fairness metrics does not come at a cost in classification quality. CNACR attains a higher F1 score than every baseline on both attributes and at both corruption levels, and each of these differences is significant at $p < 1 0 ^ { - 4 }$

## ACSPublicCoverage: Distribution Shift across States

In this section, we conduct the experiment on the ACSPublicCoverage dataset. We train all the method only on California data and test on all 8 states, for a distributional shift test. ACSPublicCoverage predicts whether an individual has public health insurance coverage. We choose this dataset because it is very different than the Adult dataset. The coverage rates in the raw data are very close across all groups. Same as the Adult dataset experiments, we first conduct the binary-group experiment using sex as the attribute. Then we conduct the multi-group experiment using the 5 race groups in the data: White, Black, Hispanic, Asian, and Other. In addition, we also use the intersection of the two (sex × race) for a 10-sensitive-attribute experiment.

Setup. For each run, we randomly sample $N _ { \mathrm { t r a i n } } = 2 0 0 0 0$ data proportionally from California. Again, the sampling preserves the proportion of each class and attribute. We only train the methods with California data, and then we randomly sample $N _ { \mathrm { t e s t } } = 5 0 0 0$ out-of-sample data from each state, including California, as test sets. We repeat this for 100 runs and report the mean ± standard deviation like before. And again, every method is trained on the same training set and tested on the same test set, allowing us to compare the results by paired t-tests over the 100 runs.

Results. Tables 3–6 give the binary case with sex as the protected attribute. CNACR attains the best EO-ratio and the lowest Gini coefficient on every state. According to the paired t-tests, every advantage is significant at $p < 1 0 ^ { - 4 }$ with CNACR ahead on over 90 of the 100 runs. On the contrary, the three fairness baselines stay very close to the plain SVM on both measures in every state. Table 6 reports the goodness-of-fit test rejection counts at $\alpha = 0 . 0 5$ . CNACR has the lowest or joint-lowest count on 6 of the 8 states. Pooled over all 800 tests, CNACR rejects the null hypothesis of equal TPR in 4.9% of cases, lowest compared to all the other methods. In terms of classification performance, CNACR has the lowest pooled F1 score. However, we can see that the poor performance is mainly in the last three states (FL, GA, TX), and CNACR has the best F1 score out of the five methods in the other five states, with a smaller standard deviation. The three losing states have very different distributions from the rest of the five states, whose coverage rates are much higher. This pattern shows again in the race experiment.

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td></td><td></td><td></td><td></td><td>0.9433 ± 0.042 0.9618 ± 0.0270.9434 ± 0.039 0.9453 ± 0.039 0.9444 ± 0.041</td></tr><tr><td>NY</td><td></td><td></td><td></td><td></td><td>0.9552 ± 0.030 0.9657 ± 0.022 0.9545 ± 0.031 0.9565 ± 0.030 0.9551 ± 0.031</td></tr><tr><td>MA</td><td></td><td></td><td></td><td></td><td> $0 . 9 6 0 4 \pm 0 . 0 3 3 \ { \bf 0 . 9 7 2 6 \pm 0 . 0 1 9 } \ 0 . 9 5 8 7 \pm 0 . 0 3 4 \ 0 . 9 5 8 9 \pm 0 . 0 3 4 \ 0 . 9 6 0 7 \pm 0 . 0 3 4$ </td></tr><tr><td>AZ</td><td></td><td></td><td></td><td></td><td> $0 . 9 5 9 4 \pm 0 . 0 3 0 \ { \bf 0 . 9 7 0 5 \pm 0 . 0 2 0 \ { \mathrm { 0 . 9 5 8 8 } } } \pm 0 . 0 2 9 \ { \mathrm { 0 . 9 6 0 6 \pm 0 . 0 2 8 \ { \mathrm { 0 . 9 6 0 7 } } \pm 0 . 0 3 0 } }$ </td></tr><tr><td>IL</td><td></td><td></td><td></td><td></td><td> $0 . 9 5 2 1 \pm 0 . 0 3 5 \ 0 . 9 6 6 9 \pm 0 . 0 2 8 \ 0 . 9 5 2 7 \pm 0 . 0 3 4 \ 0 . 9 5 3 2 \pm 0 . 0 3 4 \ 0 . 9 5 1 2 \pm 0 . 0 3 7$ </td></tr><tr><td>FL</td><td></td><td></td><td></td><td>0.9327 ± 0.039 0.9628 ± 0.027 0.9335 ± 0.041 0.9347 ± 0.041 0.9335 ± 0.041</td><td></td></tr><tr><td>GA</td><td></td><td></td><td>0.9569 ± 0.029 0.9723 ± 0.020 0.9565 ± 0.029 0.9562 ± 0.032</td><td></td><td> $0 . 9 5 8 2 \pm 0 . 0 3 1$ </td></tr><tr><td>TX</td><td> $0 . 9 4 4 0 \pm 0 . 0 3 8$ </td><td> $\mathbf { 0 . 9 6 4 9 \pm 0 . 0 2 4 }$ </td><td>0.9428± 0.038</td><td> $0 . 9 4 3 7 \pm 0 . 0 3 7$ </td><td> $0 . 9 4 3 3 \pm 0 . 0 3 7$ </td></tr><tr><td colspan="2">Pooled 0.9505</td><td>0.9672</td><td>0.9501</td><td>0.9511</td><td>0.9509</td></tr></table>

Table 3: ACSPublicCoverage, sex: per-state EO-ratio (mean ± std over 100 runs).

We now report the five-group case with race as the sensitive attribute. CNACR attains the best EO-ratio on every test state (Table 7) with the lowest standard deviations, so CNACR’s fairness is not only better on average but also more consistent from run to run. And the paired t-test says the difference over the 100 runs is significant at $p < 1 0 ^ { - 4 }$ against every baseline, with CNACR winning on 96 to 98 of the 100 runs.

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td> $0 . 0 1 4 8 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 0 9 8 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 1 4 8 \pm 0 . 0 1 0$ </td><td> $0 . 0 1 4 3 \pm 0 . 0 1 1$ </td><td> $0 . 0 1 4 5 \pm 0 . 0 1 1$ </td></tr><tr><td>NY</td><td> $0 . 0 1 1 6 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 0 8 8 \pm 0 . 0 0 6 }$ </td><td> $0 . 0 1 1 8 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 1 2 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 1 6 \pm 0 . 0 0 8$ </td></tr><tr><td>MA</td><td> $0 . 0 1 0 2 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 0 0 7 0 \pm 0 . 0 0 5 }$ </td><td> $0 . 0 1 0 7 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 0 6 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 0 2 \pm 0 . 0 0 9$ </td></tr><tr><td>AZ</td><td> $0 . 0 1 0 5 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 0 7 5 \pm 0 . 0 0 5 }$ </td><td> $0 . 0 1 0 6 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 0 1 \pm 0 . 0 0 7$ </td><td> $0 . 0 1 0 1 \pm 0 . 0 0 8$ </td></tr><tr><td>IL</td><td> $0 . 0 1 2 4 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 0 0 8 5 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 1 2 3 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 2 1 \pm 0 . 0 0 9 \ 0 . 0 1 2 7 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td>FL</td><td> $0 . 0 1 7 6 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 0 9 6 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 1 7 4 \pm 0 . 0 1 1$ </td><td> $0 . 0 1 7 1 \pm 0 . 0 1 1$ </td><td> $0 . 0 1 7 4 \pm 0 . 0 1 1$ </td></tr><tr><td>GA</td><td> $0 . 0 1 1 1 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 0 7 1 \pm 0 . 0 0 5 }$ </td><td> $0 . 0 1 1 2 \pm 0 . 0 0 8$ </td><td>0.0113± 0.008</td><td> $0 . 0 1 0 8 \pm 0 . 0 0 8$ </td></tr><tr><td>TX</td><td> $0 . 0 1 4 6 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 0 0 9 0 \pm 0 . 0 0 6 }$ </td><td> $0 . 0 1 4 9 \pm 0 . 0 1 0$ </td><td> $0 . 0 1 4 7 \pm 0 . 0 1 0$ </td><td> $0 . 0 1 4 8 \pm 0 . 0 1 0$ </td></tr><tr><td>Pooled</td><td>0.0128</td><td>0.0084</td><td>0.0130</td><td>0.0127</td><td>0.0128</td></tr></table>

Table 4: ACSPublicCoverage, sex: per-state Gini coefficient on per-group TPRs (mean ± std over 100 runs).

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td> $0 . 6 2 7 0 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 6 5 1 4 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 2 6 2 \pm 0 . 0 0 8$ </td><td> $0 . 6 2 6 6 \pm 0 . 0 0 8$ </td><td> $0 . 6 2 6 8 \pm 0 . 0 0 8$ </td></tr><tr><td>NY</td><td> $0 . 6 4 4 8 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 6 6 9 1 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 4 4 1 \pm 0 . 0 0 8$ </td><td> $0 . 6 4 3 4 \pm 0 . 0 0 8$ </td><td> $0 . 6 4 4 2 \pm 0 . 0 0 8$ </td></tr><tr><td>MA</td><td> $0 . 6 8 8 5 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 7 1 0 0 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 8 7 8 \pm 0 . 0 0 8$ </td><td> $0 . 6 8 6 7 \pm 0 . 0 0 8$ </td><td> $0 . 6 8 7 6 \pm 0 . 0 0 8$ </td></tr><tr><td>AZ</td><td> $0 . 6 6 0 4 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 6 2 0 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 5 9 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 5 9 8 \pm 0 . 0 0 7$ </td><td> $0 . 6 6 0 1 \pm 0 . 0 0 7$ </td></tr><tr><td>IL</td><td> $0 . 6 9 2 6 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 9 8 9 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 9 1 2 \pm 0 . 0 0 7$ </td><td> $0 . 6 9 1 1 \pm 0 . 0 0 8$ </td><td> $0 . 6 9 2 1 \pm 0 . 0 0 7$ </td></tr><tr><td>FL</td><td> $0 . 6 7 7 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 5 4 2 \pm 0 . 0 0 8$ </td><td> $0 . 6 7 8 2 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 7 8 3 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 7 7 9 \pm 0 . 0 0 7$ </td></tr><tr><td>GA</td><td> $0 . 6 9 0 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 5 9 2 \pm 0 . 0 0 8$ </td><td> $0 . 6 9 1 2 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 6 9 1 7 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 9 1 2 \pm 0 . 0 0 7$ </td></tr><tr><td>TX</td><td> $0 . 6 7 4 4 \pm 0 . 0 0 8$ </td><td> $0 . 6 3 7 9 \pm 0 . 0 0 8$ </td><td> $0 . 6 7 4 3 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 7 5 1 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 7 4 6 \pm 0 . 0 0 7$ </td></tr><tr><td>Pooled</td><td>0.6696</td><td>0.6678</td><td>0.6691</td><td>0.6691</td><td>0.6693</td></tr></table>

Table 5: ACSPublicCoverage, sex: per-state F1 score (mean ± std over 100 runs).

<table><tr><td>State</td><td>SVM</td><td>CNACR Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td>10/100 6/100</td><td>8/100</td><td>7/100</td><td>9/100</td></tr><tr><td>NY MA</td><td>5/100 4/100 8/100 2/100</td><td>6/100 7/100</td><td>4/100 7/100</td><td>7/100 9/100</td></tr><tr><td>AZ IL</td><td>2/100 2/100 8/100 10/100</td><td>2/100 6/100</td><td>2/100 5/100</td><td>2/100 9/100</td></tr><tr><td>FL GA</td><td>10/100 7/100 3/100 3/100</td><td>11/100 1/100</td><td>8/100 4/100</td><td>11/100 5/100</td></tr><tr><td>TX Total 6.5%</td><td>6/100 4.9%</td><td>5/100</td><td>7/100 6/100</td><td>5/100</td></tr></table>

Table 6: ACSPublicCoverage, sex: per-state chi-square rejection counts out of 100 runs, at $\alpha = 0 . 0 5$ . The last row pools all 800 (seed, state) settings and is descriptive only, since the eight states within a run share one trained classifier.

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>FL-EOp</td></tr><tr><td>CA</td><td> $0 . 7 3 6 8 \pm 0 . 0 8 9$ </td><td> $\mathbf { 0 . 8 3 1 1 \pm 0 . 0 5 7 }$ </td><td> $0 . 7 5 7 2 \pm 0 . 0 8 7$ </td><td> $0 . 7 7 5 9 \pm 0 . 0 8 9$ </td></tr><tr><td>TX</td><td> $0 . 7 4 7 5 \pm 0 . 1 0 0$ </td><td> $\mathbf { 0 . 8 2 1 6 \pm 0 . 0 8 2 }$ </td><td> $0 . 7 4 7 2 \pm 0 . 0 9 9$ </td><td> $0 . 7 2 0 1 \pm 0 . 1 0 3$ </td></tr><tr><td>NY</td><td> $0 . 7 8 9 1 \pm 0 . 0 9 0$ </td><td> $\mathbf { 0 . 8 3 5 2 \pm 0 . 0 7 8 }$ </td><td> $0 . 7 9 2 3 \pm 0 . 0 9 1$ </td><td> $0 . 7 6 6 3 \pm 0 . 1 0 3$ </td></tr><tr><td>FL</td><td> $0 . 7 0 1 2 \pm 0 . 1 3 2$ </td><td> $\mathbf { 0 . 7 9 0 0 \pm 0 . 0 9 8 }$ </td><td> $0 . 7 0 8 1 \pm 0 . 1 2 9$ </td><td> $0 . 7 0 0 5 \pm 0 . 1 2 8$ </td></tr><tr><td>IL</td><td> $0 . 7 5 1 7 \pm 0 . 0 8 9$ </td><td> $\mathbf { 0 . 8 0 9 8 \pm 0 . 0 7 3 }$ </td><td> $0 . 7 5 8 4 \pm 0 . 0 9 2$ </td><td> $0 . 7 4 6 1 \pm 0 . 0 9 7$ </td></tr><tr><td>MA</td><td> $0 . 7 5 8 3 \pm 0 . 0 9 2$ </td><td> $\mathbf { 0 . 8 1 1 2 \pm 0 . 0 6 8 }$ </td><td> $0 . 7 4 3 2 \pm 0 . 0 9 3$ </td><td> $0 . 6 8 8 1 \pm 0 . 1 1 1$ </td></tr><tr><td>GA</td><td> $0 . 7 6 5 4 \pm 0 . 0 9 9$ </td><td> $\mathbf { 0 . 8 4 4 5 \pm 0 . 0 7 1 }$ </td><td> $0 . 7 7 1 2 \pm 0 . 1 0 0$ </td><td> $0 . 7 4 6 2 \pm 0 . 0 9 6$ </td></tr><tr><td>AZ</td><td> $0 . 7 2 9 1 \pm 0 . 0 9 5$ </td><td> $\mathbf { 0 . 8 0 8 3 \pm 0 . 0 8 1 }$ </td><td> $0 . 7 4 8 0 \pm 0 . 0 9 0$ </td><td> $0 . 7 5 1 4 \pm 0 . 0 9 1$ </td></tr><tr><td>Pooled</td><td>0.7474</td><td>0.8190</td><td>0.7532</td><td>0.7368</td></tr><tr><td>CA</td><td> $0 . 6 2 7 0 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 5 1 4 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 2 6 8 \pm 0 . 0 0 7$ </td><td> $0 . 6 2 6 5 \pm 0 . 0 0 7$ </td></tr><tr><td>TX</td><td> $\mathbf { 0 . 6 7 4 5 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 3 8 8 \pm 0 . 0 0 8$ </td><td> $0 . 6 7 4 5 \pm 0 . 0 0 8$ </td><td> $0 . 6 7 4 2 \pm 0 . 0 0 8$ </td></tr><tr><td>NY</td><td> $0 . 6 4 5 0 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 6 9 2 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 4 4 5 \pm 0 . 0 0 7$ </td><td> $0 . 6 4 3 4 \pm 0 . 0 0 7$ </td></tr><tr><td>FL</td><td> $\mathbf { 0 . 6 7 8 3 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 5 4 4 \pm 0 . 0 0 7$ </td><td> $0 . 6 7 8 6 \pm 0 . 0 0 7$ </td><td> $0 . 6 7 8 2 \pm 0 . 0 0 7$ </td></tr><tr><td>IL</td><td> $0 . 6 9 2 6 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 7 0 0 2 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 9 2 0 \pm 0 . 0 0 7$ </td><td> $0 . 6 9 0 4 \pm 0 . 0 0 7$ </td></tr><tr><td>MA</td><td> $0 . 6 8 7 9 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 7 1 0 9 \pm 0 . 0 0 8 }$ </td><td> $0 . 6 8 7 5 \pm 0 . 0 0 8$ </td><td> $0 . 6 8 7 5 \pm 0 . 0 0 8$ </td></tr><tr><td>GA</td><td> $\mathbf { 0 . 6 9 0 6 \pm 0 . 0 0 8 }$ </td><td> $0 . 6 6 0 0 \pm 0 . 0 0 8$ </td><td> $0 . 6 9 0 9 \pm 0 . 0 0 8$ </td><td> $0 . 6 9 0 0 \pm 0 . 0 0 8$ </td></tr><tr><td>AZ</td><td> $0 . 6 6 0 8 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 6 6 2 8 \pm 0 . 0 0 7 }$ </td><td> $0 . 6 6 0 7 \pm 0 . 0 0 7$ </td><td> $0 . 6 6 0 3 \pm 0 . 0 0 7$ </td></tr><tr><td>Pooled</td><td>0.6696</td><td>0.6685</td><td>0.6694</td><td>0.6688</td></tr></table>

Table 7: ACSPublicCoverage, race: per-state EO-ratio (mean ± std over 100 runs).

Table 8: ACSPublicCoverage, race: per-state F1 score (mean ± std over 100 runs).

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>FL-EOp</td></tr><tr><td>CA</td><td> $0 . 0 3 5 1 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 0 2 1 6 \pm 0 . 0 0 8 }$ </td><td> $0 . 0 3 2 5 \pm 0 . 0 1 2$ </td><td> $0 . 0 2 8 7 \pm 0 . 0 1 1$ </td></tr><tr><td>TX</td><td> $0 . 0 2 5 9 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 0 1 8 2 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 2 6 0 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 9 5 \pm 0 . 0 1 1$ </td></tr><tr><td>NY</td><td> $0 . 0 2 2 7 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 0 1 6 9 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 2 2 0 \pm 0 . 0 0 9$ </td><td> $0 . 0 2 5 5 \pm 0 . 0 1 0$ </td></tr><tr><td>FL</td><td> $0 . 0 2 5 3 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 1 7 6 \pm 0 . 0 0 8 }$ </td><td> $0 . 0 2 4 9 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 7 1 \pm 0 . 0 1 1$ </td></tr><tr><td>IL</td><td> $0 . 0 2 7 4 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 0 1 8 6 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 2 5 2 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 4 1 \pm 0 . 0 0 8$ </td></tr><tr><td>MA</td><td> $0 . 0 2 0 4 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 1 5 1 \pm 0 . 0 0 5 }$ </td><td> $0 . 0 2 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 0 2 6 2 \pm 0 . 0 0 9$ </td></tr><tr><td>GA</td><td> $0 . 0 2 2 7 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 0 1 4 4 } \pm \mathbf { 0 . 0 0 7 }$ </td><td> $0 . 0 2 2 5 \pm 0 . 0 0 9$ </td><td> $0 . 0 3 3 1 \pm 0 . 0 1 4$ </td></tr><tr><td>AZ</td><td> $0 . 0 2 6 7 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 1 7 1 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 2 5 5 \pm 0 . 0 1 1$ </td><td> $0 . 0 2 6 3 \pm 0 . 0 1 1$ </td></tr><tr><td>Pooled</td><td>0.0258</td><td>0.0174</td><td>0.0250</td><td>0.0276</td></tr></table>

Table 9: ACSPublicCoverage, race: per-state Gini coefficient on per-group TPRs (mean ± std over 100 runs).

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>FL-EOp</td></tr><tr><td>CA</td><td>21/100</td><td>8/100</td><td>16/100</td><td>6/100</td></tr><tr><td>TX</td><td>5/100</td><td>2/100</td><td>5/100</td><td>13/100</td></tr><tr><td>NY</td><td>3/100</td><td>4/100</td><td>6/100</td><td>13/100</td></tr><tr><td>FL</td><td>10/100</td><td>6/100</td><td>9/100</td><td>12/100</td></tr><tr><td>IL</td><td>22/100</td><td>17/100</td><td>13/100</td><td>10/100</td></tr><tr><td>MA</td><td>6/100</td><td>5/100</td><td>11/100</td><td>35/100</td></tr><tr><td>GA</td><td>7/100</td><td>5/100</td><td>6/100</td><td>22/100</td></tr><tr><td>AZ</td><td>14/100</td><td>12/100</td><td>10/100</td><td>10/100</td></tr><tr><td>Total</td><td>88/800 (11.0%)</td><td>59/800 (7.4%)</td><td>76/800 (9.5%)</td><td>119/800 (14.9%)</td></tr></table>

Table 10: ACSPublicCoverage, race: per-state chi-square rejection counts, that is, the number of runs (out of 100) in which the test rejects $H _ { 0 }$ (equal TPRs across the 5 race groups) at $\alpha \ : = \ : 0 . 0 5$ . The last row pools across all $1 0 0 \times 8 = 8 0 0$ (seed, state) settings; since the eight states within a run share one trained classifier, this pooled figure is descriptive and is not used for inference.

The F1 difference is small and depends on the state (Table 8). CNACR is the best on CA, NY, IL, MA and AZ, but it is the worst on TX, FL and GA, just like the experiment with the binary sex attribute.

The Gini coefficient results (Table 9) again show that CNACR has the lowest score on all 8 states, and the paired t-test shows the difference is significant against every baseline at $p < 1 0 ^ { - 4 }$ , with CNACR ahead on 98 or 99 of the 100 runs. Table 10 reports the goodness-of-fit test rejection counts at α = 0.05. CNACR has the lowest or joint-lowest count on 4 of the 8 states. Pooled over all 800 (seed, state) settings, CNACR rejects the null hypothesis in 7.4% of cases, which is again the lowest among all the methods.

We can see that the reduction method form the fairness constraint only on the CA training distribution, and once the deployment population changes they make fairness worse than a plain SVM. This shows an overfitting pattern of the fairness constraint to the source distribution. On the other hand, the other two baseline classifiers sit in between, neither helping nor hurting. The coverage rates across the groups are too close; therefore, the constraints of these two methods leave their classifiers almost unchanged. CNACR is the only method that improves the fairness in this situation.

## Intersectional Groups: Sex × Race

The two attributes above are treated separately. We now take their intersection, which gives 10 protected groups. The coverage rates of the ten intersectional groups in California spread from 0.360 to 0.394, slightly wider than the spread ot the five race groups difference. Therefore, this does not introduce any new disparity in the raw data; it is still in the same situation as the previous two experiments. The experiment protocol is identical to the previous experiments: 100 runs, training on only California, $N _ { \mathrm { t r a i n } } = 2 0 0 0 0$ and $\bar { N } _ { \mathrm { t e s t } } = 5 \bar { 0 } 0 0$ per state.

Tables 11–14 report the results.

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td>0.5947</td><td>0.7057</td><td>0.6070</td><td>0.6150</td><td>0.5873</td></tr><tr><td>NY</td><td>0.6251</td><td>0.6956</td><td>0.6064</td><td>0.6019</td><td>0.5748</td></tr><tr><td>MA</td><td>0.5875</td><td>0.6655</td><td>0.5722</td><td>0.5512</td><td>0.5246</td></tr><tr><td>AZ</td><td>0.5659</td><td>0.6436</td><td>0.5780</td><td>0.5687</td><td>0.5566</td></tr><tr><td>IL</td><td>0.5774</td><td>0.6618</td><td>0.5722</td><td>0.5665</td><td>0.5330</td></tr><tr><td>FL</td><td>0.4853</td><td>0.6087</td><td>0.4826</td><td>0.4917</td><td>0.4617</td></tr><tr><td>GA</td><td>0.5573</td><td>0.6835</td><td>0.5592</td><td>0.5635</td><td>0.5220</td></tr><tr><td>TX</td><td>0.5252</td><td>0.6583</td><td>0.5242</td><td>0.5233</td><td>0.4770</td></tr><tr><td colspan="6">Pooled 0.5648 0.6653 0.5627 0.5602 0.5296</td></tr></table>

Table 11: ACSPublicCoverage, sex × race (10 groups): per-state EO-ratio (mean over 100 runs).

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td>0.0519</td><td>0.0347</td><td>0.0499</td><td>0.0492</td><td>0.0502</td></tr><tr><td>NY</td><td>0.0378</td><td>0.0286</td><td>0.0388</td><td>0.0395</td><td>0.0443</td></tr><tr><td>MA</td><td>0.0357</td><td>0.0270</td><td>0.0370</td><td>0.0374</td><td>0.0414</td></tr><tr><td>AZ</td><td>0.0411</td><td>0.0281</td><td>0.0407</td><td>0.0416</td><td>0.0442</td></tr><tr><td>IL</td><td>0.0431</td><td>0.0311</td><td>0.0432</td><td>0.0419</td><td>0.0461</td></tr><tr><td>FL</td><td>0.0441</td><td>0.0298</td><td>0.0447</td><td>0.0439</td><td>0.0485</td></tr><tr><td>GA</td><td>0.0373</td><td>0.0265</td><td>0.0385</td><td>0.0402</td><td>0.0576</td></tr><tr><td>TX</td><td>0.0450</td><td>0.0323</td><td>0.0461</td><td>0.0468</td><td>0.0526</td></tr><tr><td colspan="6">Pooled 0.0420 0.0298 0.0424</td></tr></table>

Table 12: ACSPublicCoverage, sex × race: per-state Gini coefficient on per-group TPRs (mean over 100 runs).

In terms of classification, the F1 score of CNACR is the best on 5 out of 8 states, and it is lossing on the same three states again. The pooled F1 score is very close to the plain SVM, so the fairness improvement is obtained at essentially no cost in classification quality.

On the fairness metrics, CNACR attains the best EO-ratio and the lowest Gini coefficient in each of the eight states. Moreover, we notice that the advantage here is larger than in the five-group race experiment. This is the same pattern showed in [12], where the risk-averse method’s advantage grows with the number of classes. In this case, CNACR’s advantage grows with the number of protected groups.

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td>0.6271</td><td>0.6512</td><td>0.6242</td><td>0.6241</td><td>0.6172</td></tr><tr><td>NY</td><td>0.6450</td><td>0.6681</td><td>0.6408</td><td>0.6403</td><td>0.6299</td></tr><tr><td>MA</td><td>0.6884</td><td>0.7100</td><td>0.6840</td><td>0.6828</td><td>0.6729</td></tr><tr><td>AZ</td><td>0.6610</td><td>0.6627</td><td>0.6583</td><td>0.6579</td><td>0.6480</td></tr><tr><td>IL</td><td>0.6927</td><td>0.6999</td><td>0.6887</td><td>0.6872</td><td>0.6762</td></tr><tr><td>FL</td><td>0.6785</td><td>0.6556</td><td>0.6773</td><td>0.6770</td><td>0.6662</td></tr><tr><td>GA</td><td>0.6914</td><td>0.6620</td><td>0.6916</td><td>0.6903</td><td>0.6743</td></tr><tr><td>TX</td><td>0.6745</td><td>0.6398</td><td>0.6746</td><td>0.6740</td><td>0.6636</td></tr><tr><td colspan="2">Pooled 0.6698</td><td>0.6687</td><td>0.6674</td><td>0.6667</td><td>0.6560</td></tr></table>

Table 13: ACSPublicCoverage, sex × race: per-state F1 score (mean over 100 runs).

<table><tr><td>State</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>CA</td><td>14/100</td><td>10/100</td><td>11/100</td><td>11/100</td><td>5/100</td></tr><tr><td>NY</td><td>4/100</td><td>4/100</td><td>5/100</td><td>5/100</td><td>11/100</td></tr><tr><td>MA</td><td>4/100</td><td>5/100</td><td>5/100</td><td>10/100</td><td>17/100</td></tr><tr><td>AZ IL</td><td>13/100 10/100</td><td>8/100 10/100</td><td>8/100 13/100</td><td>9/100 4/100</td><td>10/100 13/100</td></tr><tr><td>FL</td><td>8/100</td><td>5/100</td><td>8/100</td><td>11/100</td><td>20/100</td></tr><tr><td>GA</td><td>6/100</td><td>4/100</td><td>7/100</td><td>4/100</td><td>33/100</td></tr><tr><td>TX</td><td>5/100</td><td>5/100</td><td>4/100</td><td>5/100</td><td>16/100</td></tr></table>

Table 14: ACSPublicCoverage, sex × race: per-state chi-square rejection counts out of 100 runs, at $\alpha = 0 . 0 5$

The baselines move in the opposite direction. On the Gini coefficient, both two fairness baselines are worse than the plain SVM. The same ordering appears in the goodness-of-fit rejection rate table. It seems that constraining ten group-conditional rates simultaneously, several of which are estimated from very few positive examples, may overfit the constraint.

## Remarks on κ Parameters

CNACR is reported at $\kappa _ { \mathrm { i n n e r } } = 0 . 1 , \kappa _ { \mathrm { m i d } } = 1$ and $\kappa _ { \mathrm { o u t } } = 0 . 5$ throughout the experiments. Note that κ has a range of [0, 1] for the mean-semi-deviation to be a coherent risk measure. Looking at the two outer layers of the CNACR structure, $\kappa _ { \mathrm { o u t } }$ aggregates the risk of each class, whereas $\kappa _ { \mathrm { m i d } }$ aggregates the risk of each sensitive group. This means that, theoretically, $\kappa _ { \mathrm { m i d } }$ should be the parameter that controls the strength of fairness enforcement. To see how the result changes with the choice of κ parameters, we vary one parameter at a time over [0, 1] with the other two held fixed at the default value, using 20 runs.

The experiment shows that the layers contribute very differently, and it depends on the data. When the group base rates differ, the middle level produces the improvement: on Adult dataset, where the raw data carries a relatively great unfairness, raising $\kappa _ { \mathrm { m i d } }$ from 0 to 1 increases the EO-ratio from 0.8508 to 0.9498 in the sex experiment, 0.7640 to 0.8394 in the race experiment. And the Gini coefficient also decreases from 0.0194 to 0.0066 in the sex experiment, 0.0132 to 0.0083 in the race experiment. However, on ACSPubcov dataset, where the base rate are very close across the groups, $\kappa _ { \mathrm { m i d } }$ barely does anything. Under all settings, increasing the value of $\kappa _ { \mathrm { m i d } }$ leads to almost zero change in the fairness metrics.

The outer layer is also contributing to fairness enforcement. Changing $\kappa _ { \mathrm { o u t } }$ from 0 to 1 decreases Gini coefficient from 0.0087 to 0.0060 in the Adult dataset using sex as the attribute, which is a much smaller change compared to changing $\kappa _ { \mathrm { m i d } }$ under the same setting. However, with more groups involved and more scenarios to be aggregated, $\kappa _ { \mathrm { o u t } }$ is contributing much more. In the race experiment with the Adult dataset, increasing $\kappa _ { \mathrm { o u t } }$ decreases Gini coefficient from 0.0113 to 0.0054, which is even more significant than changing $\kappa _ { \mathrm { m i d } }$ . And in the ACSPubcov dataset, where $\kappa _ { \mathrm { m i d } }$ does almost nothing, changing $\kappa _ { \mathrm { o u t } }$ from 0 to 1 decreases Gini coefficient from 0.0145 to 0.0076 in the sex experiment, 0.0311 to 0.0140 in the race experiment.

On the performance part, increasing $\kappa _ { \mathrm { m i d } }$ for stronger fairness enforcement almost never costs anything. Under all settings, increasing $\kappa _ { \mathrm { m i d } }$ decreases the F1 score by a small amount that is indistinguishable from 0. Increasing $\kappa _ { \mathrm { o u t } } ,$ on the other hand, slightly helps with the F1 score in the ACSPubcov experiment. But it does not make any significant changes in the Adult experiment.

## Within-Group Unfairness

<table><tr><td>Attribute</td><td>Noise</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td><td>FL-EOdds</td></tr><tr><td rowspan="2">sex</td><td>0%</td><td>0.08236</td><td>0.08019</td><td>0.08382</td><td>0.08257</td><td>0.08315</td><td>0.08883</td></tr><tr><td>20%</td><td>0.08236</td><td>0.07970</td><td>0.08384</td><td>0.08255</td><td>0.08332</td><td>0.08593</td></tr><tr><td rowspan="2">race</td><td>0%</td><td>0.08269</td><td>0.07820</td><td>0.08317</td><td></td><td>0.08303</td><td>0.08383</td></tr><tr><td>20%</td><td>0.08269</td><td>0.07808</td><td>0.08315</td><td></td><td>0.08288</td><td>0.08370</td></tr></table>

Table 15: Adult: within-group unfairness $I _ { \mathrm { w i t h i n } }$ under the Generalized Entropy decomposition of [34] (mean over 100 runs). Lower is better. The method of [15] is defined for two groups only and is not used in the race experiment.

<table><tr><td>Attribute</td><td>SVM</td><td>CNACR</td><td>Zafar</td><td>Donini</td><td>FL-EOp</td></tr><tr><td>sex</td><td> $0 . 1 4 5 2 6$ </td><td>0.12941</td><td>0.14563</td><td> $0 . 1 4 5 4 9$ </td><td>0.14531</td></tr><tr><td>race</td><td>0.14444</td><td>0.12942</td><td>0.14456</td><td></td><td>0.14498</td></tr></table>

Table 16: ACSPublicCoverage: within-group unfairness $I _ { \mathrm { w i t h i n } } ,$ pooled over the 8 test states (mean over 100 runs). Lower is better.

All the measures used so far compare the true positive rates of the groups against each other, so they only see the group averages. [34] point out that this can hide a second kind of unfairness: a method can equalize the group averages and at the same time make the treatment of the individuals inside a group more uneven. They give each individual a benefit score $b _ { i } = \hat { y } _ { i } - y _ { i } + 1$ , so a false negative receives 0, a correct prediction receives 1 and a false positive receives 2, and measure the inequality of the benefit vector with the Generalized Entropy index at $\alpha = 2$ . This index splits exactly into a between-group part and a within-group part,

$$
I _ { \mathrm { t o t a l } } = I _ { \mathrm { b e t w e e n } } + I _ { \mathrm { w i t h i n } } ,
$$

where $I _ { \mathrm { b e t w e e n } }$ replaces every individual by the mean benefit of their group and $I _ { \mathrm { w i t h i n } }$ combines the internal inequality of each group. We verified the identity numerically on every run and the largest residual was below $1 0 ^ { - 1 5 }$ . The between-group part measures the same thing as the fairness measures already reported in the previous tables, namely how the groups compare on average, so here we report only the within-group part, which the earlier measures cannot see.

Tables 15 and 16 give the results. CNACR is the only in-training method that reduces the within-group unfairness relative to the plain SVM, and it does so in all four settings: by 2.6% and 3.2% on Adult with sex, by 5.4% and 5.6% on Adult with race, and by about 11% and 10% on ACSPublicCoverage. Every constraint-based method increases it, in every setting, although the increases are small. This means that, for CNACR, the fairness improvements reported in the previous sections are not obtained at the cost of treating individuals inside a group more unevenly, which [34] warn about for methods that equalize only the group averages.

## References

[1] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. A reductions approach to fair classification. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of PMLR, pages 60–69, 2018.

[2] Wael Alghamdi, Hsiang Hsu, Haewon Jeong, Hao Wang, P. Winston Michalak, Shahab Asoodeh, and Flavio P. Calmon. Beyond Adult and COMPAS: Fair multi-class prediction via information projection. In Advances in Neural Information Processing Systems, volume 35, pages 38747–38760, 2022.

[3] Aray Almen and Darinka Dentcheva. On risk evaluation and control of distributed multi-agent systems. Journal ofOptimization Theory and Applications, pages 1–30, 2024.

[4] Philippe Artzner, Freddy Delbaen, Jean-Marc Eber, and David Heath. Coherent measures of risk. Mathematical finance, 9(3):203–228, 1999.

[5] Barry Becker and Ronny Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5XW20.

[6] L. Elisa Celis, Anay Mehrotra, and Nisheeth Vishnoi. Fair classification with adversarial perturbations. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 8158–8171, 2021.

[7] John J. Cherian and Emmanuel J. Candès. Statistical inference for fairness auditing, 2023. Preprint at https: //arxiv.org/abs/2305.03712.

[8] Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto, and Massimiliano Pontil. Fair regression with Wasserstein barycenters. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 7321–7331, 2020.

[9] Freddy Delbaen. Coherent risk measures on general probability spaces. Advances in Finance and Stochastics, pages 1–37, March 2000.

[10] Christophe Denis, Romuald Elie, Mohamed Hebiri, and François Hu. Fairness guarantees in multi-class clas sification with demographic parity. Journal of Machine Learning Research, 25:1–46, 2024. Earlier version: arXiv:2109.13642 (2021), “Fairness guarantee in multi-class classification”.

[11] Darinka Dentcheva and Andrzej Ruszczynski. Risk-Averse Optimization and Control Theory and Methods. Springer Series in Operations Research and Financial Engineering. 2024.

[12] Darinka Dentcheva and Xiangyu Tian. Risk-averse fair multi-class classification. arXiv preprint arXiv:2509.05771, 2025.

[13] Darinka Dentcheva and Xiangyu Tian. Risk-averse fair multi-class classification. arXiv preprint arXiv:2509.05771, 2025.

[14] Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. In Advances in Neural Information Processing Systems 34, pages 6478–6490, 2021.

[15] Michele Donini, Luca Oneto, Shai Ben-David, John S. Shawe-Taylor, and Massimiliano Pontil. Empirical risk minimization under fairness constraints. In Advances in Neural Information Processing Systems, volume 31, pages 2791–2801, 2018.

[16] Wei Du and Xintao Wu. Fair and robust classification under sample selection bias. In Proceedings of the 30th acm international conference on information & knowledge management, pages 2999–3003, 2021.

[17] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings ofthe 3rd Innovations in Theoretical Computer Science Conference (ITCS), pages 214–226, 2012.

[18] Michael Feldman, Sorelle A Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. Certifying and removing disparate impact. In proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pages 259–268, 2015.

[19] Benjamin Fish, Jeremy Kun, and Ádám D Lelkes. A confidence-based approach for balancing fairness and accuracy. In Proceedings of the 2016 SIAM international conference on data mining, pages 144–152. SIAM, 2016.

[20] Hans Föllmer and Alexander Schied. Stochastic Finance: An Introduction in Discrete Time, 3rd Edition. 2011.

[21] Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Multicalibration: Calibration for the (computationally-identifiable) masses. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 1939–1948, 2018.

[22] François Hu, Philipp Ratz, and Arthur Charpentier. Fairness in multi-task learning via Wasserstein barycenters. In Machine Learning and Knowledge Discovery in Databases: Research Track (ECML PKDD 2023), Lecture Notes in Computer Science, 2023.

[23] Ray Jiang, Aldo Pacchiano, Tom Stepleton, Heinrich Jiang, and Silvia Chiappa. Wasserstein fair classification. In Ryan P. Adams and Vibhav Gogate, editors, Proceedings of the 35th Uncertainty in Artificial Intelligence Conference (UAI), volume 115 of Proceedings ofMachine Learning Research, pages 862–872, 2020.

[24] Faisal Kamiran and Toon Calders. Data preprocessing techniques for classification without discrimination. Knowledge and information systems, 33(1):1–33, 2012.

[25] Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu. Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In International conference on machine learning, pages 2564–2572. PMLR, 2018.

[26] Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Fairness through computationally-bounded awareness. In Advances in Neural Information Processing Systems, volume 31, 2018. Page range differs across secondary sources; confirm via the URL.

[27] Carol Xuan Long, Hsiang Hsu, Wael Alghamdi, and Flavio P. Calmon. Individual arbitrariness and group fairness. In Advances in Neural Information Processing Systems, volume 36, pages 68602–68624, 2023.

[28] G.Ch. Pflug and W. Römisch. Modeling, Measuring and Managing Risk. World Scientific, Singapore, 2007.

[29] Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q. Weinberger. On fairness and calibration. In Advances in Neural Information Processing Systems, volume 30, pages 5680–5689, 2017.

[30] Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh. Fairbatch: Batch selection for model fairness. arXiv preprint arXiv:2012.01696, 2020.

[31] A. Ruszczynski and A. Shapiro. Optimization of risk measures. In G. Calafiore and F. Dabbene, editors,´ Probabilistic and Randomized Methods for Design under Uncertainty, pages 117–158. Springer-Verlag, London, 2005.

[32] A. Ruszczynski and A. Shapiro. Optimization of convex risk functions. ´ Mathematics of Operations Research, 31:433–452, 2006.

[33] Yves Rychener, Bahar Taskesen, and Daniel Kuhn. Metrizing fairness. arXiv preprint arXiv:2205.15049, 2022.

[34] Till Speicher, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian Weller, and Muhammad Bilal Zafar. A unified approach to quantifying algorithmic unfairness: Measuring individual & group unfairness via inequality indices. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, page 2239–2248, July 2018.

[35] Haipei Sun, Kun Wu, Ting Wang, and Wendy Hui Wang. Towards fair and robust classification. In 2022 IEEE 7th European Symposium on Security and Privacy (EuroS& P), pages 356–376, 2022.

[36] Bahar Taskesen, Viet Anh Nguyen, Daniel Kuhn, and Jose Blanchet. A distributionally robust approach to fair classification. arXiv preprint arXiv:2007.09530, 2020.

[37] Yijie Wang, Viet Anh Nguyen, and Grani A Hanasusanto. Wasserstein robust classification with fairness con straints. Manufacturing & Service Operations Management, 26(4):1567–1585, 2024.

[38] Robert C. Williamson and Aditya Krishna Menon. Fairness risk measures, 2019.

[39] Han Xu, Xiaorui Liu, Yaxin Li, Anil Jain, and Jiliang Tang. To be robust or to be fair: Towards fairness in adversarial training. In International conference on machine learning, pages 11492–11501. PMLR, 2021.

[40] Shizhou Xu and Thomas Strohmer. On the (in)compatibility between group fairness and individual fairness, 2024. Preprint at https://arxiv.org/abs/2401.07174.

[41] Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez-Rodriguez, and Krishna P. Gummadi. Fairness con straints: A flexible approach for fair classification. Journal ofMachine Learning Research, 20(75):1–42, 2019.

[42] Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork. Learning fair representations. In Inter national conference on machine learning, pages 325–333. PMLR, 2013.