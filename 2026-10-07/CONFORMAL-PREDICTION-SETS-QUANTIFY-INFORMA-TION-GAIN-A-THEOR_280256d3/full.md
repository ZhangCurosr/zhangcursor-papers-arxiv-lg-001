# CONFORMAL PREDICTION SETS QUANTIFY INFORMA-TION GAIN: A THEORETICAL PERSPECTIVE

Kevin Zhang<sup>1</sup>

Stephen Bates<sup>1</sup>

## ABSTRACT

Conformal prediction is a popular tool for uncertainty quantification that outputs prediction sets with finite-sample coverage guarantees. While prediction set size is commonly used as a heuristic measure of uncertainty, the information-theoretic basis for this interpretation remains poorly understood. In this work, we provide such a foundation using a decision-theoretic generalization of entropy tailored to set-valued prediction. In particular, we introduce a family of generalized information measures based on the size and coverage of conformal prediction sets. Notably, Shannon mutual information admits an exact integral representation in terms of these measures. We then show that, in standard classification settings, the reduction in conformal set size from additional information (i) is sandwiched between calibration-dependent members of this family and (ii) obeys a data processing inequality, both up to finite-sample calibration and model error terms. Together, our results formally relate conformal prediction to classical informationtheoretic quantities and justify using set-size reduction as an information gain metric. Empirically, we validate our theory across 11 classification settings and show that set-size reduction and Shannon mutual information can rank features differently in a greedy feature selection experiment.

## 1 INTRODUCTION

Conformal prediction is a general procedure for constructing prediction sets with finite-sample and distribution-free coverage guarantees under minimal assumptions (Angelopoulos & Bates, 2021). Recently, the split conformal prediction variant has gained popularity as a lightweight and scalable calibration procedure, with applications in healthcare (Lu et al., 2022), dynamic control (Lindemann et al., 2023), and language modeling (Lu et al., 2023; Quach et al., 2024), among other settings.

Beyond coverage, conformal predictors are commonly evaluated by their inefficiency, or expected set size (Vovk et al., 2016; Sadinle et al., 2019). At the instance level, set size also provides an easyto-compute and operationally meaningful heuristic measure of uncertainty (Cresswell et al., 2024), since more ambiguous inputs typically produce larger sets (Romano et al., 2019). Motivated by this interpretation, we examine conformal set-size reduction as a natural notion of information gain.

The relationship between conformal set size and information-theoretic quantities has been studied both empirically (e.g. Hagos & Lundstrom, 2026) and theoretically by using list-decoding arguments¨ to upper bound the conditional entropy in terms of set size and coverage (Correia et al., 2024). However, these one-sided bounds do not establish whether set-size reduction itself behaves as a principled information measure. In this work, we instead analyze conformal prediction using a decision-theoretic notion of entropy (DeGroot, 1962). This perspective provides new insights into expected conformal set size and its connection to Shannon entropy and mutual information.

Our primary contributions are as follows: (i) we introduce a family of generalized information measures for set-valued prediction that exactly recovers Shannon mutual information via an integral representation; (ii) we prove that, in standard classification settings, conformal set-size reduction approximately lies between calibration-dependent members of this family and obeys a data processing inequality; and (iii) we empirically validate our theory across 11 classification settings and verify that set-size reduction and Shannon mutual information can induce different feature selection orderings in practice. Our results further the theoretical understanding of uncertainty quantification with conformal prediction and motivate set-size reduction as a principled information gain metric.

## 2 PRELIMINARIES

We introduce the notation and relevant background on information theory and conformal prediction. In this work, we consider a classification setting in which $X \in { \mathcal { X } }$ is a generic feature vector and $Y \in \mathcal { Y } : = \{ 1 , . . . , K \}$ is a categorical prediction target.

Generalized entropy and information. Our analysis relies on a decision-theoretic generalization of entropy and information, in which uncertainty is represented using Bayes risk (DeGroot, 1962; Grunwald & Dawid, 2004). Suppose we fix an action space¨ $\mathcal { A }$ and a loss function $\ell : \mathcal { V } \times \mathcal { A }  \mathbb { R }$ The generalized entropy induced by ℓ is defined as

$$
H _ { \ell } ( Y \mid X ) = \operatorname { \mathbb { E } } \left[ \operatorname* { i n f } _ { a \in { \cal A } } \ \operatorname { \mathbb { E } } [ \ell ( Y , a ) \mid X ] \right] .\tag{1}
$$

Given side information $Z ,$ , the generalized mutual information between X and $Y$ quantifies the predictive value of X by the reduction in Bayes risk under the chosen loss.

$$
I _ { \ell } ( X ; Y \mid Z ) = H _ { \ell } ( Y \mid Z ) - H _ { \ell } ( Y \mid X , Z ) \ge 0 .\tag{2}
$$

Generalized mutual information retains several natural information-theoretic properties, including non-negativity and a data processing inequality (Duchi et al., 2018). The marginal versions $H _ { \ell } ( Y )$ and $I _ { \ell } ( X ; Y )$ are defined by taking the respective conditioning variables to generate the trivial σ-algebra (e.g., if X and $Z$ are almost surely constant, respectively).

Example. Shannon entropy $H ( Y ) \mathop { = } \_ { \mathbb { E } } [ \log P _ { Y } ( Y ) ]$ is a special case of the generalized entropy by taking the action space $\begin{array} { r } { \mathbf { \mathcal { A } } = \Delta ^ { K - 1 } } \end{array}$ to be the probability simplex over $\mathcal { V }$ and using the log loss function $\mathcal { \ell } ( y , a ) = - \log a ( y )$ . The Bayes-optimal action is then $a = P _ { Y }$ , giving $H _ { \ell } ( \boldsymbol { \mathsf { V } } ) = H \mathbf { \bar { ( } } Y )$

Conformal prediction. Conformal prediction provides a general framework for converting the output of a predictive model into a prediction set with finite-sample coverage guarantees (Vovk et al., 2005; Shafer & Vovk, 2008; Angelopoulos et al., 2025). In this work, we focus on split conformal prediction (Papadopoulos et al., 2002; Lei et al., 2018), which we briefly review next.

Let $f$ be a pretrained model and ${ \mathcal D } _ { \mathrm { c a l } } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ be a calibration dataset. Given a score function $s : \overline { { \mathcal { X } } } \times \mathcal { Y }  \mathbb { R } \cup \{ \infty \}$ , define the calibration scores $S _ { i } = s ( X _ { i } , Y _ { i } )$ for $i = 1 , \ldots , n$ Fix a nominal target coverage level $1 - \alpha \in ( 0 , 1 )$ , and define the calibrated conformal threshold $\hat { \tau } _ { \alpha } = \operatorname { Q u a n t i l e } \left( ( 1 + 1 / n ) ( 1 - \alpha ) ; \{ S _ { i } \} _ { i = 1 } ^ { n } \right)$ . At inference time, the conformal prediction set for a new test point is $\mathcal { C } _ { \alpha } ( X _ { \mathrm { t e s t } } ) = \{ y \in \mathcal { V } : s ( X _ { \mathrm { t e s t } } , y ) \leq \hat { \tau } _ { \alpha } \}$

A key appeal of split conformal prediction is that the calibration procedure is lightweight and yields a finite-sample marginal coverage guarantee.

Proposition 1 (Conformal coverage guarantee). Suppose that the calibration observations $\mathcal { D } _ { \mathrm { c a l } }$ and test point $( X _ { \mathrm { t e s t } } , Y _ { \mathrm { t e s t } } )$ are exchangeable and that the scores $S _ { i }$ are almost surely distinct. Then,

$$
\begin{array} { r } { 1 - \alpha \le \mathbb { P } \big ( Y _ { \mathrm { t e s t } } \in \mathcal { C } _ { \alpha } ( X _ { \mathrm { t e s t } } ) \big ) \le 1 - \alpha + \frac { 1 } { n + 1 } , } \end{array}\tag{3}
$$

where the probability is taken jointly over the calibration data and the test point.

Proposition 1 holds for any model $f$ and score function s. Nonetheless, the score function determines the size and structure of the conformal sets (Angelopoulos et al., 2021). We review two standard choices for classification. Let $q ( \cdot \mid x ) \in \Delta ^ { K - 1 }$ denote the predicted distribution for an input x.

Under the probability score (Sadinle et al., 2019), $s ( x , y ) = - q ( y \mid x )$ is the negative predicted probability of the label. The resulting conformal prediction set $\mathcal { C } _ { \alpha } ( x ) = \{ y : q ( y \mid x ) \geq - \hat { \tau } _ { \alpha } \}$ is therefore a level set of the predicted class probabilities determined by the calibrated cutoff.

The Adaptive Prediction Sets (APS) score (Romano et al., 2020) instead computes the score by accumulating the predicted probability mass of labels that are at least as likely as the true label. Formally, let $q ( y ^ { ( 1 ) } \mid x ) > q ( \dot { y } ^ { ( 2 ) } \mid x ) \stackrel { . } { > } \dots > q ( y ^ { ( K ) } \mid x )$ denote the ordered predicted class probabilities (assume for simplicity that there are no ties). Let $r _ { x } ( y )$ be the rank of the label $y ,$ $\mathrm { i . e . , } y = y ^ { ( r _ { x } ( y ) ) }$ . The randomized APS score function is

$$
s ( x , y ) = \sum _ { \ell = 1 } ^ { r _ { x } ( y ) - 1 } q ( y ^ { ( \ell ) } \mid x ) + U \cdot q ( y \mid x ) ,\tag{4}
$$

where $U \sim \mathrm { U n i f } [ 0 , 1 ]$ is drawn independently to randomize inclusion of the boundary label. If there are ties among the predicted probabilities, we break them at random following Romano et al. (2020).

## 3 CONFORMAL ENTROPY AND MUTUAL INFORMATION

We introduce a family of generalized entropy and mutual information measures motivated by setvalued prediction. Our main result then establishes a two-sided bound showing that information gain from conformal prediction is bracketed by calibration-dependent members of this family, up to calibration and model error terms. Lastly, we extend the construction to the APS score. All proofs are deferred to Appendix A.

## 3.1 GENERALIZED INFORMATION MEASURES FOR THE PROBABILITY SCORE

We begin with a family of loss functions indexed by $\lambda \in ( 0 , 1 ]$ . For a set-valued prediction $\Gamma \subseteq \mathcal { V }$ define the loss function

$$
\begin{array} { r } { \ell _ { \lambda } ( Y , \Gamma ) = | \Gamma | + \frac 1 \lambda \mathbf { 1 } ( Y \notin \Gamma ) , } \end{array}\tag{5}
$$

which penalizes both the size of Γ and its miscoverage of the true label $Y ,$ . We use $\ell _ { \lambda }$ to define a generalized entropy with action space $A = 2 ^ { \mathcal { V } }$ , the collection of all subsets of $\mathcal { V }$

Definition 1 (Conformal entropy and mutual information). For a fixed $\lambda \in \mathsf { \Gamma } ( 0 , 1 ]$ , the conformal entropy is defined as

$$
H _ { \lambda } ( Y \mid X ) = \operatorname \operatorname { \mathbb { E } } { \Big [ } \operatorname* { i n f } _ { \Gamma \subseteq { \mathcal { V } } } \operatorname { \mathbb { E } } { \big [ } \ell _ { \lambda } ( Y , \Gamma ) \mid X { \big ] } { \Big ] } .\tag{6}
$$

The corresponding conformal mutual information is $I _ { \lambda } ( X ; Y \mid Z ) = H _ { \lambda } ( Y \mid Z ) - H _ { \lambda } ( Y \mid X , Z ) .$

Since the action space is finite, the conformal entropy can equivalently be expressed as an infimum over (measurable) decision rules $\Gamma : \mathcal { X }  \mathcal { A }$ , so that $H _ { \lambda } ( { \dot { Y } } \mid X ) { \dot { = } } \operatorname* { i n f } _ { \Gamma : { \dot { \mathcal { X } } } \to A } \operatorname { \mathbb { E } } \left[ \ell _ { \lambda } ( Y , \Gamma ( X ) ) \right]$ We next characterize the Bayes-optimal actions under $\ell _ { \lambda }$

Proposition 2 (Optimal set-valued predictors). $F i x \lambda \in ( 0 , 1 ]$ and let $p ( \cdot \mid X )$ be the conditional distribution $o f Y$ given X. The decision rule $\Gamma _ { \lambda } ^ { * } ( X ; p ) \ : = \ : \{ y \in \mathcal { y } : p ( y \mid X ) \geq \lambda \}$ minimizes $\mathbb { E } \big [ \ell _ { \lambda } ( Y , \Gamma ( X ) ) \big ]$ over all $\Gamma : \mathcal { X }  2 ^ { \mathcal { Y } }$

Proposition 2 explains the conformal entropy terminology: the Bayes-optimal action $\Gamma _ { \lambda } ^ { * } ( X ; p )$ is exactly the conformal prediction set from an oracle model $f ( X ) \stackrel { \cdot } { = } p ( \cdot \mid X )$ using the probability score at the calibrated threshold $\hat { \tau } _ { \alpha } = - \lambda$ . This connection should be unsurprising since $\dot { \mathbb { E } } [ \ell _ { \lambda } ( Y , \Gamma ) ]$ is the Lagrangian relaxation of minimizing expected set size subject to a coverage constraint, which is precisely the tradeoff underlying the probability score (Sadinle et al., 2019).

In practice, the model predictions may not coincide with the true conditional distribution of $Y .$ . The resulting model imperfection can be quantified using a generalized information divergence<sup>2</sup>.

Definition 2 (Conformal information divergence). For afixed $\lambda \in ( 0 , 1 ]$ , the conformal information divergence between conditional distributions $p ( \cdot \mid X )$ and $q ( \cdot \mid X )$ is defined as

$$
D _ { \lambda } ( p ( \cdot  { | } X )  { | } q ( \cdot  { | } X ) ) = \mathbb { E } _ { X } \bigl [ \mathbb { E } _ { Y \sim p ( \cdot  { | } X ) } [ \ell _ { \lambda } ( Y , \Gamma _ { \lambda } ^ { * } ( X ; q ) ) - \ell _ { \lambda } ( Y , \Gamma _ { \lambda } ^ { * } ( X ; p ) ) ] \bigr ] .\tag{7}
$$

Intuitively, $D _ { \lambda } ( p \parallel q )$ is the excess risk incurred by acting under $q$ when the true distribution is $p .$ The divergence is non-negative by the optimality of $\Gamma _ { \lambda } ^ { * } ( X ; p )$ and vanishes when $p = q$

Using our construction, we can decompose Shannon mutual information into a uniform average of conformal mutual information terms over $\lambda ,$ analogous to the classical decomposition of the log loss into cost-weighted binary losses (Schervish, 1989).

Theorem 1 (Integral representation of Shannon information). Let I denote the Shannon mutual information (in nats). For any random variables $X$ and $Z ,$

$$
I ( X ; Y \mid Z ) = \int _ { 0 } ^ { 1 } I _ { \lambda } ( X ; Y \mid Z ) d \lambda .\tag{8}
$$

This identity connects information gain in conformal prediction to classical information-theoretic quantities. At a high level, Proposition 2 expresses conformal mutual information in terms of differences in set size and miscoverage. Standard coverage guarantees then allow us to bound set-size reduction in terms of $I _ { \lambda }$ at calibrated values of $\lambda ,$ each of which corresponds to a slice of Shannon mutual information by Theorem 1. The following section makes this intuition precise.

![](images/aec4ade7b22825942ffe3db5ed407170174bb25efb214d847b6371e563d27576.jpg)  
Figure 1: We illustrate an example of our setting in (1a), where $Z$ is a low-resolution image and X is its residual. The information gain of X beyond Z can be quantified by $\Delta _ { \alpha }$ . We develop the information hierarchy in (1b), showing that conformal mutual information near the calibrated λ approximates $\Delta _ { \alpha }$ , while Shannon mutual information aggregates the entire family (Theorem 1).

## 3.2 APPROXIMATE INFORMATION GAIN USING CONFORMAL SET-SIZE REDUCTION

We first formalize the information gained from additional features X beyond base features Z with conformal prediction. For notational convenience, let $X ^ { ( 1 ) } = Z$ and $X ^ { ( 2 ) } = \left( X , Z \right)$

For $i \in \{ 1 , 2 \}$ , let $p ( \cdot \mid X ^ { ( i ) } )$ and $\boldsymbol { q } ( \cdot \mid X ^ { ( i ) } )$ denote the true and predicted conditional distributions, respectively. Let $\hat { \tau } _ { \alpha } ^ { \left( i \right) }$ be the calibrated probability-score threshold using $X ^ { ( i ) }$ at nominal coverage level $1 - \alpha$ , obtained from a shared calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ of size n. Let $\mathcal { C } _ { \alpha } ( X ^ { ( i ) } )$ denote the corresponding conformal prediction set. The (training-conditional) expected set-size reduction is

$$
\Delta _ { \alpha } = \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( X ^ { ( 1 ) } ) | \big ] - \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ] ,\tag{9}
$$

where the expectations are over a new test input. This setting is illustrated in Figure 1a.

We emphasize that $\Delta _ { \alpha }$ depends on $\mathcal { D } _ { \mathrm { c a l } }$ . One might expect set-size reduction to be non-negative on average over draws of the calibration data $( \mathrm { i . e . , \mathbb { E } } [ \Delta _ { \alpha } ] \stackrel { - } { \geq } 0 )$ . In the following analysis, we show that an approximate version of this intuition holds.

Theorem 2 (Probability score sandwich bound). Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha .$ . For $i \in \{ 1 , 2 \}$ , let $\hat { \lambda } _ { \alpha } ^ { ( i ) } = - \hat { \tau } _ { \alpha } ^ { ( i ) }$ and assume that $\hat { \lambda } _ { \alpha } ^ { ( i ) } > 0 .$ . Moreover, define

$$
\begin{array} { r } { I ^ { ( i ) } = I _ { \hat { \lambda } _ { \sigma } ^ { ( i ) } } \big ( X ; Y \mid Z \big ) , \qquad D ^ { ( i ) } = D _ { \hat { \lambda } _ { \sigma } ^ { ( i ) } } \big ( p ( \cdot \mid X ^ { ( i ) } ) \mid \mid q ( \cdot \mid X ^ { ( i ) } \big ) \big ) . } \end{array}\tag{10}
$$

Let the training-conditional miscoverage be $\varepsilon ^ { ( i ) } = \mathbb { P } ( Y \notin \mathcal { C } _ { \alpha } ( X ^ { ( i ) } ) ; \mathcal { D } _ { \mathrm { c a l } } )$ , where the probability is over a fresh test point $( X ^ { ( i ) } , Y )$ , and define $\varepsilon ^ { \Delta } = \varepsilon ^ { ( 2 ) } - \dot { \varepsilon } ^ { ( 1 ) }$ . Then,

$$
\begin{array} { r } { I ^ { ( 2 ) } - D ^ { ( 2 ) } + \varepsilon ^ { \Delta } / \hat { \lambda } _ { \alpha } ^ { ( 2 ) } \leq \Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 1 ) } + \varepsilon ^ { \Delta } / \hat { \lambda } _ { \alpha } ^ { ( 1 ) } . } \end{array}\tag{11}
$$

Theorem 2 states that conformal set-size reduction is bracketed by conformal mutual information terms at two different values of λ, up to calibration terms and model error. By standard conformal arguments, the training-conditional miscoverage concentrates around α as n grows, which allows us to control the calibration term in expectation under suitable regularity conditions.

Corollary 1 (Probability score sandwich bound in expectation). Fix a nominal coverage level $1 - \alpha$ Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. For $i \in \{ 1 , 2 \}$ , define $\hat { \lambda } _ { \alpha } ^ { ( i ) } , \ : \dot { I } ^ { ( i ) }$ , and $D ^ { ( i ) }$ as in Theorem 2, and suppose that $s ^ { ( i ) } = \dot { \mathrm { V a r } } ( 1 / \hat { \lambda } _ { \alpha } ^ { ( i ) } ) ^ { 1 / 2 } < \infty .$ . Then,

$$
\begin{array} { r } { \mathbb { E } \big [ I ^ { ( 2 ) } \big ] - \mathbb { E } \big [ D ^ { ( 2 ) } \big ] - s ^ { ( 2 ) } / \sqrt { n } \le \mathbb { E } [ \Delta _ { \alpha } ] \le \mathbb { E } \big [ I ^ { ( 1 ) } \big ] + \mathbb { E } \big [ D ^ { ( 1 ) } \big ] + s ^ { ( 1 ) } / \sqrt { n } , } \end{array}\tag{12}
$$

where all expectations are over the calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$

Since $I _ { \lambda } \geq 0 ,$ , Corollary 1 implies that on average over draws of the calibration data, the expected set-size reduction from adding X is asymptotically non-negative when $\mathbb { E } [ D ^ { ( 2 ) } ] = 0 \mathrm { a n d } s ^ { ( 2 ) }$ remains bounded as n increases.

![](images/c5ba42fd9bc7ea1f4d3826cd0505cb2754e7cc93609e3e626127f3cfbcba5ee8.jpg)

![](images/2ed4ad0c4e3a91d97dabfcd97b3d343f14dd38f54dbf73acbb1c06493e50de43.jpg)  
Figure 2: Conformal mutual information accurately reflects feature utility for reducing set size, whereas Shannon mutual information may not. We show information profiles $\lambda \mapsto I _ { \lambda } ( { \bar { X } } _ { i } ; Y \mid Z )$ for two candidate features $X _ { 1 }$ and $X _ { 2 }$ with similar Shannon mutual information (left), along with their differences in $I _ { \lambda }$ and E $[ \Delta _ { \alpha } ]$ using the probability score (right). Nominal miscoverage levels $\alpha \in \{ 0 . 0 1 , 0 . 1 , \ldots , 0 . 9 , 0 . 9 9 \}$ are indicated along the top axis. Despite having nearly identical Shannon mutual information, $X _ { 1 }$ yields a larger conformal set-size reduction at high coverage $( 1 - \alpha > 0 . 6 )$ , whereas $X _ { 2 }$ reduces set size more at low coverage levels. Each difference in $\mathbb { E } [ \tilde { \Delta _ { \alpha } } ]$ is positioned at the geometric mean of the two calibrated thresholds.

The variance condition in Corollary 1 is mild, requiring only that small predicted probabilities of the true label $q ( Y \mid X ^ { ( i ) } )$ are sufficiently rare. In Appendix $\mathbf { A } ,$ we further relax this to almost-sure positivity using an alternative bound that holds with high probability over the calibration data.

We now revisit the integral representation in Theorem 1. Shannon mutual information uniformly averages $I _ { \lambda }$ over all $\lambda \in ( 0 , 1 \bar { ] }$ , whereas $\Delta _ { \alpha }$ is approximated by members near the calibrated $\hat { \lambda } _ { \alpha } ^ { ( i ) }$ We summarize this relationship in Figure 1b. Since these two quantities operate across different regions of λ, Shannon mutual information can misrepresent the utility of new features for conformal prediction. We demonstrate this behavior in Figure 2 using a long-tailed synthetic setting with two candidate features $X _ { 1 }$ and $X _ { 2 }$ . Details on the data-generating process are described in Appendix B.

## 3.3 DATA PROCESSING INEQUALITY FOR CONFORMAL SET-SIZE REDUCTION

Conformal set-size reduction enjoys additional properties that make it attractive as an approximate notion of mutual information. Here, we highlight one such property by deriving a data processing inequality (DPI) for conformal prediction sets.

Suppose that for each data point we also observe a single realization of a degraded feature $\tilde { X }$ obtained from a (possibly noisy) channel, so that $Y  ( { \overset { \smile } { X } } , Z )  ( { \overset { \vartriangle } { X } } , Z )$ forms a Markov chain. Let $\tilde { X } ^ { ( 2 ) } = ( \tilde { X } , \tilde { Z ) }$ and, similarly to $\Delta _ { \alpha }$ , define the set-size reduction with the degraded features as

$$
\tilde { \Delta } _ { \alpha } = \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( X ^ { ( 1 ) } ) | \big ] - \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) | \big ] .\tag{13}
$$

The following analysis shows that conformal set-size reduction satisfies an approximate version of the DPI by comparing $\Delta _ { \alpha }$ and $\tilde { \Delta } _ { \alpha }$

Theorem 3 (Probability score DPI). Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha$ . Let $\hat { \lambda } _ { \alpha } ^ { ( 2 ) }$ $D ^ { ( 2 ) }$ , and $\varepsilon ^ { ( 2 ) }$ be defined as in Theorem 2 and assume $\hat { \lambda } _ { \alpha } ^ { ( 2 ) } > 0 .$ . Define the trainingconditional miscoverage for the degraded features as $\tilde { \varepsilon } ^ { ( 2 ) } = \mathbb { P } \big ( Y \notin \mathcal { C } _ { \alpha } \big ( \tilde { X } ^ { ( 2 ) } \big ) ; \mathcal { D } _ { \mathrm { c a l } } \big )$ . Then,

$$
\tilde { \Delta } _ { \alpha } \leq \Delta _ { \alpha } + D ^ { ( 2 ) } + \big ( \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } \big ) \big / \hat { \lambda } _ { \alpha } ^ { ( 2 ) } .\tag{14}
$$

As before, a suitable regularity condition controls the finite-sample calibration term in expectation.

Corollary 2 (Probability score DPI in expectation). Fix a nominal coverage level 1−α. Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. Let $\hat { \lambda } _ { \alpha } ^ { ( 2 ) }$ and $D ^ { ( 2 ) }$ be defined as in Theorem 2, and suppose that $s ^ { ( 2 ) } = \mathrm { V a r } ( 1 / \hat { \lambda } _ { \alpha } ^ { ( 2 ) } ) ^ { 1 / 2 } < \infty$ Then,

$$
\mathbb { E } [ \tilde { \Delta } _ { \alpha } ] \leq \mathbb { E } [ \Delta _ { \alpha } ] + \mathbb { E } [ D ^ { ( 2 ) } ] + s ^ { ( 2 ) } / \sqrt { n } .\tag{15}
$$

Corollary 2 establishes the approximate DPI for the conformal set-size reduction: on average over draws of the calibration data, the degraded feature $\tilde { X }$ reduces the set size by no more than $\check { X }$ does, up to model imperfection and an $O ( \stackrel { \smile } { n } ^ { - 1 / 2 } )$ calibration term. However, in practice, these slack terms could make the degraded feature spuriously appear more informative.

## 3.4 EXTENSION OF CONFORMAL MUTUAL INFORMATION FOR APS SCORE

We extend our results to conformal prediction sets with the APS score. Unlike the probability score, whose calibrated threshold acts as a single global cutoff on $q ( \cdot \mid x )$ , the APS score effectively applies an input-dependent threshold with randomization at the boundary. Hence, we make two key extensions in our analysis. First, we index the conformal mutual information by a random $\Lambda \in ( 0 , \dot { 1 } ]$ in place of the fixed λ. Second, we introduce $U \sim \mathrm { U n i f } [ 0 , 1 ]$ independent of $( \dot { \Lambda } , X , Y )$ to randomize the set-valued action. Let Ω denote the space of measurable functions $[ 0 , 1 ] \stackrel { } {  } 2 ^ { y }$

Definition 3 (Conformal entropy and mutual information for random Λ). Let $\Lambda \in ( 0 , 1 ]$ be a random index variable and let $\ell _ { \Lambda } ( Y , \Gamma )$ denote the loss in Equation 5 evaluated using the value ofΛ. Define the conformal entropy under Λ as

$$
H _ { \Lambda } ( Y \mid X ) = \operatorname { \mathbb { E } } { \Bigl [ } \operatorname* { i n f } _ { \Gamma \in \Omega } \operatorname { \mathbb { E } } { \bigl [ } \ell _ { \Lambda } ( Y , \Gamma ( U ) ) \mid X { \bigr ] } { \Bigr ] } ,\tag{16}
$$

where the inner expectation is taken over $Y , U$ , and Λ. The corresponding conformal mutual information is $I _ { \Lambda } ( X ; \bar { Y } \mid Z ) = H _ { \Lambda } ( Y \mid Z ) - H _ { \Lambda } ( Y \mid X , Z )$

The random Λ allows the loss index to depend on the input. Since the action space is finite, we again express the conformal entropy as an infimum over measurable decision rules $\Gamma : \mathcal { X } \times \left. 0 , 1 \right. \ \stackrel { } { \to } \ \mathcal { A }$ and show that conformal prediction sets using the APS score achieve $H _ { \Lambda }$ for suitable choices of Λ.

$$
\Lambda ) .
$$

$$
\Lambda = \xi ( X )
$$

$$
\xi : \mathcal { X } \to ( 0 , \bar { 1 } \bar { 1 }
$$

$$
p ( \cdot \mid X )
$$

$$
\Gamma _ { \Lambda } ^ { \ast } ( X , \dot { U } ; p )
$$

$$
u \in [ 0 , 1 ]
$$

$$
\big \{ y : p ( y \mid X ) > \xi ( X ) \big \} \subseteq \Gamma _ { \Lambda } ^ { * } ( X , u ; p ) \subseteq \big \{ y : p ( y \mid X ) \geq \xi ( X ) \big \} .\tag{17}
$$

Then, $\Gamma _ { \Lambda } ^ { \ast } ( X , U ; p )$ minimizes $\mathbb { E } \left[ \ell _ { \Lambda } ( Y , \Gamma ( X , U ) ) \right]$ ] over all measurable Γ : $\mathcal { X } \times [ 0 , 1 ] \to 2 ^ { \mathcal { Y } }$

The conformal prediction set using an oracle model and the $\mathsf { A P S }$ score is a Bayes-optimal action $\Gamma _ { \Lambda } ^ { \ast } ( X , U ; p )$ , where Λ is the induced input-dependent probability threshold equal to the first predicted probability (in decreasing order) for which the cumulative mass exceeds $\hat { \tau } _ { \alpha } .$ When $\hat { \tau } _ { \alpha } < 1$ this quantity is well-defined and satisfies $\Lambda = \xi ( X ) \in ( 0 , 1 ]$ for some measurable function $\xi .$

We define $\Delta _ { \alpha }$ as in Equation 9, with the expectation now also taken over $U .$ . Furthermore, we overload $\hat { \tau } _ { \alpha } ^ { \left( i \right) }$ to denote the calibrated APS-score threshold using $X ^ { ( i ) }$ and $\varepsilon ^ { ( i ) }$ to be the $X ^ { ( 2 ) }$ -conditional miscoverage. We next formalize the conformal information divergence under a random Λ.

Definition 4 (Conformal information divergence for random Λ). Suppose that $\Lambda = \xi ( X )$ for some fixed measurable function $\xi : \mathcal { X } \to ( 0 , 1 ]$ . Define the conformal information divergence under Λ between conditional distributions $p ( \cdot \mid X )$ and $q ( \cdot \mid X )$ as

$$
D _ { \Lambda } ( p ( \cdot \mid X ) \parallel q ( \cdot \mid X ) ) = \mathbb { E } _ { X } \bigl [ \mathbb { E } _ { Y \sim p ( \cdot \mid X ) , U } [ \ell _ { \Lambda } ( Y , \Gamma _ { \Lambda } ^ { * } ( X , U ; q ) ) - \ell _ { \Lambda } ( Y , \Gamma _ { \Lambda } ^ { * } ( X , U ; p ) ) ] \bigr ] .\tag{18}
$$

In Definition 4, the value of $D _ { \Lambda }$ depends on the boundary randomization choice for $\Gamma _ { \Lambda } ^ { * } ( X , U ; q )$ When Λ is the probability threshold induced by the APS score of $q ( \cdot \mid X )$ at some $\tau < 1$ , we use the convention that $\Gamma _ { \Lambda } ^ { \ast } ( \dot { X } , U ; q )$ is the corresponding conformal prediction set. We are now ready to extend Theorem 2 for the APS score.

Theorem 4 (APS score sandwich bound). Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha . \ F o r i \in \{ 1 , 2 \}$ , assume that $\hat { \tau } _ { \alpha } ^ { ( i ) } < 1$ , and let $\hat { \Lambda } _ { \alpha } ^ { \left( i \right) }$ be the input-dependent random probability threshold induced $b \overset { \cdot } { y } \hat { \tau } _ { \alpha } ^ { ( i ) }$ . Moreover, define

$$
\begin{array} { r } { I ^ { ( i ) } = I _ { \hat { \Lambda } _ { \sigma } ^ { ( i ) } } \big ( X ; Y \mid Z \big ) , \qquad D ^ { ( i ) } = D _ { \hat { \Lambda } _ { \sigma } ^ { ( i ) } } \big ( p \big ( \cdot \mid X ^ { ( i ) } \big ) \mid \mid q \big ( \cdot \mid X ^ { ( i ) } \big ) \big ) . } \end{array}\tag{19}
$$

Let $ { \varepsilon } ^ { ( i ) } = \mathbb { P } ( Y \not \in  { \mathcal C } _ { \alpha } ( X ^ { ( i ) } ) \mid X ^ { ( 2 ) } ;  { \mathcal D } _ { \mathrm { c a l } } )$ , where the probability is taken jointly over a fresh test label Y given $X ^ { ( 2 ) }$ and the APS randomization U. Define $\varepsilon ^ { \Delta } = \dot { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 1 ) }$ . Then,

$$
I ^ { ( 2 ) } - D ^ { ( 2 ) } + \mathbb { E } \big [ \varepsilon ^ { \Delta } / \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } \big ] \leq \Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 1 ) } + \mathbb { E } \big [ \varepsilon ^ { \Delta } / \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } \big ] ,\tag{20}
$$

where the expectations are over $X ^ { ( 2 ) }$ , and both bounds arefinite.

Theorem 4 reduces to Theorem 2 when $\hat { \Lambda }$ is constant. For a random $\hat { \Lambda } ,$ the $\varepsilon ^ { \Delta }$ terms depend on the $X ^ { ( 2 ) }$ -conditional miscoverage, rather than the training-conditional miscoverage. In Appendix A, we derive corresponding bounds in expectation and with high probability over the calibration data, as well as a corresponding data processing inequality. In general, the $\mathsf { A P S }$ results incur additional slack on both sides of the bound, because the conditional miscoverage typically does not concentrate as n grows. For oracle models, this slack vanishes in the upper bound.

## 4 RELATED WORK

We build on the decision-theoretic notion of information introduced by DeGroot (1962). A long line of work has studied generalized entropies in relation to proper scoring rules (Gneiting & Raftery, 2007) and information-theoretic divergences (Reid & Williamson, 2011; Garc´ıa-Garc´ıa & Williamson, 2012). We instead focus on a particular family of generalized information measures tailored to set-valued prediction. The loss in Equation 5 is related to class-selective rejection (Ha, 1996) and classification with rejection (Herbei & Wegkamp, 2006), though these ideas have largely been developed independently of conformal prediction (Chzhen et al., 2021).

A growing body of empirical work studies conformal set size as an operational measure of predictive uncertainty (Fayyad et al., 2024; Portela et al., 2025; Hagos & Lundstrom, 2026). On the theoret-¨ ical side, several works characterize expected set size as a measure of predictive inefficiency using finite-sample estimates (Dhillon et al., 2024) and model-dependent bounds (Zecchin et al., 2024). However, these works do not formally motivate set size as a measure of uncertainty or information.

The work closest to ours is Correia et al. (2024), who use list-decoding arguments to upper bound the conditional entropy in terms of set size and coverage. In contrast, our decision-theoretic approach uses generalized entropy to exactly characterize set size and miscoverage. This formulation yields two-sided bounds on set-size reduction, which we further connect to Shannon mutual information via an integral representation. These results provide a direct information-theoretic interpretation of set-size reduction that is not captured by existing work.

Information gain metrics are widely used in practice for active learning (Houlsby et al., 2011; Kirsch et al., 2019; Bickford Smith et al., 2023) and feature acquisition (Covert et al., 2023). While these approaches typically rely on classical information-theoretic quantities, Neiswanger et al. (2022) study decision-theoretic information to optimize downstream utility. Separately, conformal prediction has been used to design acquisition policies based on prediction set size (Kharazian et al., 2024). Our main contribution is identifying theoretical connections between these notions of information gain.

## 5 EXPERIMENTS

We empirically examine our theoretical results and then probe set-size reduction and Shannon mutual information in a feature selection experiment. Motivated by the behavior in Figure 2, we show that these two criteria can rank features differently in practice. For all experiments, we fix the nominal coverage to be $1 - \alpha = 0 . 9$ , unless noted otherwise.

Datasets. We consider 11 datasets spanning synthetic, image, and tabular domains. We use three synthetic settings. The first (S1) defines the true conditional distribution using a two-layer ReLU network, while the second (S2) uses a Gaussian mixture model with each mixture component representing a class. For the feature selection experiments, we additionally introduce a third setting (S3), where the conditional distribution has a long low-probability tail. All synthetic datasets provide access to the true conditional probabilities, allowing us to evaluate our results using an oracle model.

For tabular domains, we use the Wine Quality (Wine) (Cortez et al., 2009), Human Activity Recognition (HAR) (Reyes-Ortiz et al., 2013), Letter Recognition (Letter) (Slate, 1991), and Covertype (Blackard, 1998) datasets. For image classification, we use FMNIST (Xiao et al., 2017), CIFAR-10, CIFAR-100 (Krizhevsky, 2009), and ImageNet-1k (Deng et al., 2009). Dataset sizes, dimensions, numbers of classes, and preprocessing details are provided in Appendix B.

Feature construction. We partition coordinates of the synthetic data and features of the tabular datasets into disjoint subsets Z and X. For images, we take Z to be a Gaussian-blurred version of the original data and X to be its residual (see Figure 1a). The DPI experiments construct X<sup>˜</sup> by either adding Gaussian noise or applying Gaussian blur to X. Candidate features in the selection experiment are partitions of the input vector for synthetic and tabular datasets or a pretrained dense representation for image datasets. Appendix B provides further details on feature construction.

Models. We train a model for each feature set on the real-world datasets and hold it fixed across subsequent calibration procedures. The tabular datasets use MLPs with two or three hidden layers, ReLU activations, and dropout. For FMNIST, CIFAR-10, and CIFAR-100, we train CNNs with max pooling, while for ImageNet-1k we fine-tune a pretrained ResNet-18 (He et al., 2016).

![](images/7259327beee0fc40f46b7d03b2c14720e8a156a540899e5be81a11046c444c9c.jpg)  
Figure 3: We show the normalized margin, width, and DPI gap based on 100 random draws of the calibration and test sets from the fixed evaluation pool. The margin and width panels report the mean ± standard error, while the DPI gap panel summarizes the distribution using boxplots. Datasets whose evaluation pool is too small for a given calibration size are omitted.

Training. We use the oracle model for the synthetic settings and generate a fixed evaluation pool of 200,000 samples. For the tabular datasets, we allocate 60% of the data to training, 15% to validation, and the remaining 25% to an evaluation pool. The image datasets have dedicated evaluation pools, and the remaining data is split into 80% for training and 20% for validation.

Across all training procedures, we use early stopping based on validation loss and select the learning rate, weight decay, and dropout rate by grid search. In Appendix C, we provide additional details on model architectures, training procedures, hyperparameters, and predictive accuracy.

## 5.1 SANDWICH AND DPI VALIDATION

We first evaluate the sandwich bounds in Theorems 2 and 4. From each evaluation pool, we sample separate calibration and test sets to estimate $\Delta _ { \alpha }$ and $\varepsilon ^ { ( i ) }$ . In the synthetic settings, the true conditional distribution is known, so $I ^ { ( i ) }$ is computable and $D ^ { ( i ) } = 0$ . For the real-world datasets, $I ^ { ( i ) }$ and $D ^ { ( i ) }$ are inaccessible, so we use the plug-in model estimates, which set $D ^ { ( i ) } = 0$ . Moreover, we use a k-NN approximation for $I ^ { ( 2 ) }$ in the APS lower bound since $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) }$ depends on X (see Appendix D). The real-data results are thus an empirical corroboration rather than strict validation of the theory.

Letting L and U denote the lower and upper bounds, we measure the magnitude of violations using the normalized margin $d ( \Delta _ { \alpha } , [ L , U ] ) / \dot { K }$ , where $d ( x , [ L , U ] )$ is the distance from x to $\textstyle \left\lceil L , U \right\rceil$ . We separately quantify the informativeness of the bound using the normalized width $( U - \dot { L } ) / K$

The left column of Figure 3 reports the normalized margin as we vary $n _ { \mathrm { c a l } } = n _ { \mathrm { t e s t } }$ . We limit the synthetic sample sizes to match the real datasets for a controlled comparison. Even with oracle models, the average margin is positive since finite samples introduce Monte Carlo error, but it approaches zero as sample sizes increase. The margins on the real data exhibit similar behavior. The middle column reports the normalized width for a fixed $n _ { \mathrm { t e s t } }$ . The intervals narrow with more calibration data, but the width remains nonzero due to the gap between $I ^ { ( 1 ) }$ and $I ^ { ( 2 ) }$ . At very small $n _ { \mathrm { c a l } }$ and large K, the $1 / \hat { \lambda } _ { \alpha } ^ { ( i ) }$ factor makes the bounds vacuous since the calibrated threshold is typically small.

We next empirically analyze Theorem 3 and its APS analogue (Theorem 5 in Appendix A). The right column of Figure 3 reports the normalized DPI gap $( \Delta _ { \alpha } - \tilde { \Delta } _ { \alpha } ) / K$ as we vary $n _ { \mathrm { c a l } }$ with $n _ { \mathrm { t e s t } }$ fixed. At small calibration sizes, finite-sample effects can produce negative DPI gaps, meaning that the degraded features X<sup>˜</sup> appear to reduce set size more than the original features $X$ . As $n _ { \mathrm { c a l } }$ increases, however, these violations shrink in magnitude and the gap concentrates around non-negative values.

We include additional experimental analyses in Appendix E, including violation rates for Theorems 2 and 4, results across nominal coverage levels, and DPI behavior under varying noise levels.

Table 1: We evaluate feature selection performance using conformal set-size reduction. Each row represents one selection criterion and values are reported as mean ± standard error over 10 random splits of the evaluation pool. Arrows indicate the direction of improvement, and the best-performing method for each metric is bolded. Empirical coverage is reported for reference without bolding.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Criterion</td><td colspan="2">Prob.</td><td colspan="2">APS</td><td colspan="2"> $\operatorname { A c c } .$ </td></tr><tr><td>E[|C|] (↓)</td><td>Cov.</td><td>E[|C|] (↓)</td><td>Cov.</td><td> $\mathrm { { T o p - 1 } \left( \uparrow \right) }$ </td><td> $\mathrm { T o p } { - } 5 \left( \uparrow \right)$ </td></tr><tr><td rowspan="5">(0 = ) SS</td><td> $\Delta _ { \alpha } \left( \mathrm { P r o b . } \right)$ </td><td> $\mathbf { 2 . 2 3 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $2 . 6 2 9 \pm 0 . 0 0 2$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 7 5 0 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 5 2 \pm 0 . 0 0 0 }$ </td></tr><tr><td> $ { \Delta } _ { \alpha } \left(  { \mathrm { A P S } } \right)$ </td><td> $2 . 2 4 3 \pm 0 . 0 0 2$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 2 . 6 1 1 \pm 0 . 0 0 2 }$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 7 5 2 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 9 5 2 \pm 0 . 0 0 0 }$ </td></tr><tr><td>Shannon MI</td><td> $2 . 4 4 7 \pm 0 . 0 0 5$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $2 . 8 7 6 \pm 0 . 0 0 6$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 7 6 7 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 9 5 2 \pm 0 . 0 0 0 }$ </td></tr><tr><td>Accuracy</td><td> $3 . 6 1 3 \pm 0 . 0 2 0$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $4 . 4 8 7 \pm 0 . 0 2 9$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 7 6 8 \pm 0 . 0 0 1 }$ </td><td> $0 . 9 1 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Random</td><td> $3 . 0 4 6 \pm 0 . 1 0 0$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $3 . 5 9 1 \pm 0 . 1 1 6$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 6 7 2 \pm 0 . 0 0 9$ </td><td> $0 . 9 3 1 \pm 0 . 0 0 4$ </td></tr><tr><td rowspan="5">(0 = 0) CIAR</td><td> $\Delta _ { \alpha } \left( \mathrm { P r o b . } \right)$ </td><td> $1 2 . 6 8 3 \pm 0 . 1 0 0$ </td><td> $0 . 9 0 4 \pm 0 . 0 0 2$ </td><td> $1 5 . 6 6 1 \pm 0 . 1 2 7$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> $0 . 4 6 6 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 7 5 3 \pm 0 . 0 0 2 }$ </td></tr><tr><td> $\Delta _ { \alpha } \ ( { \mathrm { A P S } } )$ </td><td> ${ \bf 1 2 . 5 9 4 \pm 0 . 0 8 8 }$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> ${ \bf 1 5 . 4 7 2 \pm 0 . 1 2 9 }$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 4 6 7 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 7 5 3 \pm 0 . 0 0 2 }$ </td></tr><tr><td>Shannon MI</td><td> $1 3 . 7 6 3 \pm 0 . 1 1 5$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> $1 6 . 5 9 7 \pm 0 . 1 4 5$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> $0 . 4 5 8 \pm 0 . 0 0 3$ </td><td> $0 . 7 4 1 \pm 0 . 0 0 2$ </td></tr><tr><td>Accuracy</td><td> $1 2 . 8 7 2 \pm 0 . 1 3 0$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> $1 5 . 8 7 7 \pm 0 . 1 5 8$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 4 6 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 7 5 2 \pm 0 . 0 0 2$ </td></tr><tr><td>Random</td><td> $1 4 . 2 4 3 \pm 0 . 2 5 3$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 3$ </td><td> $1 7 . 1 8 9 \pm 0 . 2 2 4$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 3$ </td><td> $0 . 4 5 4 \pm 0 . 0 0 2$ </td><td> $0 . 7 3 7 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan="6"> $\fallingdotseq \textcircled { \approx }$  6  $\fallingdotseq$  LK</td><td>∆α (Prob.)</td><td> $\mathbf { 2 . 7 5 3 \pm 0 . 0 1 6 }$ </td><td> $0 . 8 9 8 \pm 0 . 0 0 3$ </td><td> $\mathbf { 3 . 3 5 8 \pm 0 . 0 1 9 }$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 7 5 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 2 6 \pm 0 . 0 0 0 }$ </td></tr><tr><td> $\Delta _ { \alpha } \ ( { \mathrm { A P S } } )$ </td><td> $2 . 8 3 5 \pm 0 . 0 3 8$ </td><td> $0 . 8 9 8 \pm 0 . 0 0 2$ </td><td> $3 . 3 9 8 \pm 0 . 0 3 2$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 7 5 8 \pm 0 . 0 0 2$ </td><td> $0 . 9 2 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Shannon MI</td><td> $2 . 8 0 1 \pm 0 . 0 1 9$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 3$ </td><td> $3 . 3 9 9 \pm 0 . 0 2 6$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 7 5 8 \pm 0 . 0 0 1$ </td><td> $0 . 9 2 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Accuracy</td><td> $2 . 8 7 9 \pm 0 . 0 2 0$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 2$ </td><td> $3 . 4 1 5 \pm 0 . 0 2 2$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 7 6 3 \pm 0 . 0 0 1 }$ </td><td> $0 . 9 2 2 \pm 0 . 0 0 1$ </td></tr><tr><td>Random</td><td> $4 . 4 5 9 \pm 0 . 1 8 4$ </td><td> $0 . 8 9 7 \pm 0 . 0 0 2$ </td><td> $5 . 0 5 4 \pm 0 . 1 7 4$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 6 4 2 \pm 0 . 0 1 1$ </td><td> $0 . 8 7 1 \pm 0 . 0 0 7$ </td></tr></table>

## 5.2 FEATURE SELECTION

We now investigate conformal set-size reduction as an information gain metric in a sequential feature selection setting with candidate features $\{ X _ { 1 } , \ldots , X _ { m } \}$ . At each step, we select the subsequent feature by maximizing the conformal set-size reduction or Shannon mutual information (MI), with top-1 accuracy gain and random selection as baselines. The purpose of this experiment is to empirically probe the theoretical differences between these two criteria, as highlighted in Figure 2, on real-world datasets, rather than to propose a competitive active feature acquisition method.

Evaluating the selection criteria requires estimating the label distribution for many subsets of candidate features. We amortize this computation by training a single mask-conditioned classifier that can make predictions from arbitrary subsets of observed features (Yoon et al., 2019; Covert et al., 2021). The evaluation pool is split into two halves to determine the feature ordering and evaluate its performance on held-out data. Further details on the analysis procedure are provided in Appendix D.

We measure the resulting conformal prediction set size and coverage using the probability and APS scores, as well as top-1 and top-5 accuracy. To summarize performance across the selection trajec tory, we average each metric over all non-empty proper prefixes of the feature ordering. Table 1 summarizes results for one dataset from each domain, with the remaining results in Appendix E.

Selecting features according to conformal set-size reduction consistently produces the smallest sets, although the relative performance of the probability-score and APS criteria varies. This advantage is partly expected since $\Delta _ { \alpha }$ directly targets set size, but the gap to Shannon MI shows that it can induce different feature orderings on real data. In an ablation on CIFAR-100, this gap persists when we retrain a separate model for each selected feature subset, indicating that it is not solely an artifact of the shared mask-conditioned model. Accuracy-based selection achieves the highest top-1 accuracy, while $\Delta _ { \alpha }$ matches it on top-5 accuracy. We visualize metric trajectories in Appendix E.

## 6 DISCUSSION

In this work, we provide new theoretical insights relating conformal prediction and information theory. Our main results show that in standard classification settings, $\Delta _ { \alpha }$ is bracketed by calibrationdependent members of the conformal mutual information family, which exactly recovers Shannon mutual information via an integral representation. Moreover, set-size reduction satisfies an approximate data processing inequality, further motivating it as a principled measure of information gain.

Limitations. Our theoretical analysis is limited to classification with the probability and APS scores, while our empirical evaluation considers sequential feature selection with a globally fixed ordering. Future work could extend these connections to other conformal scores and regression settings or develop active feature acquisition methods using $\Delta _ { \alpha }$ as a proxy objective.

## REFERENCES

Anastasios N. Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. CoRR, abs/2107.07511, 2021.

Anastasios N. Angelopoulos, Stephen Bates, Michael I. Jordan, and Jitendra Malik. Uncertainty sets for image classifiers using conformal prediction. In International Conference on Learning Representations, 2021.

Anastasios N. Angelopoulos, Rina Foygel Barber, and Stephen Bates. Theoretical Foundations of Conformal Prediction. Cambridge University Press, 2025.

Freddie Bickford Smith, Andreas Kirsch, Sebastian Farquhar, Yarin Gal, Adam Foster, and Tom Rainforth. Prediction-oriented bayesian active learning. In Proceedings ofthe 26th International Conference on Artificial Intelligence and Statistics, volume 206, 2023.

Jock Blackard. Covertype. UCI Machine Learning Repository, 1998.

Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an alternative to the cifar datasets. CoRR, abs/1707.08819, 2017.

Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, and Titouan Lorieul. Set-valued classification – overview via a unified framework. CoRR, abs/2102.12318, 2021.

Alvaro H. C. Correia, Fabio Valerio Massoli, Christos Louizos, and Arash Behboodi. An information theoretic perspective on conformal prediction. In Advances in Neural Information Processing Systems, volume 37, 2024.

Paulo Cortez, Antonio Cerdeira, Fernando Almeida, Telmo Matos, and Jos ´ e Reis. Wine quality.´ UCI Machine Learning Repository, 2009.

Ian Covert, Scott Lundberg, and Su-In Lee. Explaining by removing: A unified framework for model explanation. Journal ofMachine Learning Research, 22, 2021.

Ian Connick Covert, Wei Qiu, Mingyu Lu, Na Yoon Kim, Nathan J. White, and Su-In Lee. Learning to maximize mutual information for dynamic feature selection. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 6424–6447. PMLR, 2023.

Jesse C. Cresswell, Yi Sui, Bhargava Kumar, and Noel Vouitsis. Conformal prediction sets im- ¨ prove human decision making. In Proceedings of the 41st International Conference on Machine Learning, volume 235, 2024.

Morris H. DeGroot. Uncertainty, information, and sequential experiments. The Annals of Mathematical Statistics, 33, 1962.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, 2009.

Guneet S. Dhillon, George Deligiannidis, and Tom Rainforth. On the expected size of conformal prediction sets. In Proceedings ofthe 27th International Conference on Artificial Intelligence and Statistics, volume 238, 2024.

John C. Duchi, Khashayar Khosravi, and Feng Ruan. Multiclass classification, information, divergence, and surrogate risk. The Annals of Statistics, 46, 2018.

Jamil Fayyad, Shadi Alijani, and Homayoun Najjaran. Empirical validation of conformal prediction for trustworthy skin lesions classification. Computer Methods and Programs in Biomedicine, 253, 2024.

Dario Garc´ıa-Garc´ıa and Robert C. Williamson. Divergences and risks for multiclass experiments. In Proceedings of the 25th Annual Conference on Learning Theory, volume 23, 2012.

Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102, 2007.

Peter D. Grunwald and A. Philip Dawid. Game theory, maximum entropy, minimum discrepancy¨ and robust bayesian decision theory. The Annals ofStatistics, 32, 2004.

Thien M. Ha. An optimum class-selective rejection rule for pattern recognition. In Proceedings of the 13th International Conference on Pattern Recognition, volume 2. IEEE, 1996.

Misgina Tsighe Hagos and Claes Lundstrom. Performance of conformal prediction in capturing ¨ aleatoric uncertainty. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

Radu Herbei and Marten H. Wegkamp. Classification with reject option. The Canadian Journal of Statistics, 34, 2006.

Neil Houlsby, Ferenc Huszar, Zoubin Ghahramani, and M ´ at´ e Lengyel. Bayesian active learning for ´ classification and preference learning. CoRR, abs/1112.5745, 2011.

Zahra Kharazian, Tony Lindgren, Sindri Magnusson, and Henrik Bostrom. Copal: Conformal pre-¨ diction in active learning an algorithm for enhancing remaining useful life estimation in predictive maintenance. In Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications, volume 230. PMLR, 2024.

Andreas Kirsch, Joost van Amersfoort, and Yarin Gal. Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning. In Advances in Neural Information Processing Systems, volume 32, 2019.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. Distributionfree predictive inference for regression. Journal of the American Statistical Association, 113, 2018.

Lars Lindemann, Matthew Cleaveland, Gihyun Shim, and George J. Pappas. Safe planning in dynamic environments using conformal prediction. IEEE Robotics and Automation Letters, 8, 2023.

Charles Lu, Andreanne Lemay, Ken Chang, Katharina H´ obel, and Jayashree Kalpathy-Cramer. Fair¨ conformal predictors for applications in medical imaging. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, 2022.

Charles Lu, Bhawesh Kumar, Gauri Gupta, Anil Palepu, David Bellamy, Ramesh Raskar, and Andrew Beam. Conformal prediction with large language models for multi-choice question answering. In ICML 2023 Workshop on Neural Conversational AI: What’s Left to TEACH (Trustworthy, Enhanced, Adaptable, Capable and Human-centric) Chatbots?, 2023.

Olivier Marchal and Julyan Arbel. On the sub-gaussianity of the beta and dirichlet distributions. Electronic Communications in Probability, 22, 2017.

Willie Neiswanger, Lantao Yu, Shengjia Zhao, Chenlin Meng, and Stefano Ermon. Generalizing bayesian optimization with decision-theoretic entropies. In Advances in Neural Information Processing Systems, volume 35, 2022.

Harris Papadopoulos, Kostas Proedrou, Vladimir Vovk, and Alex Gammerman. Inductive confidence machines for regression. In Machine Learning: ECML 2002. Springer, 2002.

Alberto Portela, Julio R. Banga, and Marcos Matabuena. Conformal prediction for uncertainty quantification in dynamic biological systems. PLOS Computational Biology, 21, 2025.

Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi Jaakkola, and Regina Barzilay. Conformal language modeling. In International Conference on Learning Representations, 2024.

Mark D. Reid and Robert C. Williamson. Information, divergence and risk for binary experiments. Journal of Machine Learning Research, 12, 2011.

Jorge Luis Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, and Xavier Parra. Human activity recognition using smartphones. UCI Machine Learning Repository, 2013.

Yaniv Romano, Evan Patterson, and Emmanuel J. Candes. Conformalized quantile regression. In\` Advances in Neural Information Processing Systems, volume 32, 2019.

Yaniv Romano, Matteo Sesia, and Emmanuel J. Candes. Classification with valid and adaptive\` coverage. In Advances in Neural Information Processing Systems, volume 33, 2020.

Mauricio Sadinle, Jing Lei, and Larry Wasserman. Least ambiguous set-valued classifiers with bounded error levels. Journal ofthe American Statistical Association, 114, 2019.

Mark J. Schervish. A general method for comparing probability assessors. The Annals ofStatistics, 17, 1989.

Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal ofMachine Learning Research, 9, 2008.

Hajin Shim, Sung Ju Hwang, and Eunho Yang. Joint active feature acquisition and classification with variable-size set encoding. In Advances in Neural Information Processing Systems, volume 31, 2018.

David Slate. Letter recognition. UCI Machine Learning Repository, 1991.

Vladimir Vovk, Alex Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World, volume 29. Springer, 2005.

Vladimir Vovk, Valentina Fedorova, Ilia Nouretdinov, and Alex Gammerman. Criteria of efficiency for conformal prediction. In Conformal and Probabilistic Prediction with Applications. Springer, 2016.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: A novel image dataset for benchmarking machine learning algorithms. CoRR, abs/1708.07747, 2017.

Jinsung Yoon, James Jordon, and Mihaela van der Schaar. Invase: Instance-wise variable selection using neural networks. In International Conference on Learning Representations, 2019.

Matteo Zecchin, Sangwoo Park, Osvaldo Simeone, and Fredrik Hellstrom. Generalization and in-¨ formativeness of conformal prediction. In 2024 IEEE International Symposium on Information Theory. IEEE, 2024.

# SUPPLEMENTARY MATERIAL FOR CONFORMAL PREDICTION SETS QUANTIFY INFORMATION GAIN: A THEORETICAL PERSPECTIVE

## A PROOFS

We first establish several useful results about conformal entropy and mutual information, followed by concentration results for the finite-sample miscoverage terms. We then prove our main results on conformal prediction with the probability score in Appendix A.2, and extend these results to the APS score in Appendix A.3.

## A.1 PROPERTIES OF CONFORMAL ENTROPY AND MUTUAL INFORMATION

In this section, we first prove our results on optimal set-valued predictors for conformal entropy and derive the integral representation of Shannon mutual information. We also establish a data processing inequality for generalized mutual information.

## A.1.1 PROPOSITION 2: OPTIMAL SET-VALUED PREDICTORS

Consider the risk for a fixed $\lambda \in ( 0 , 1 ]$ . Using Equation $^ { 5 , }$

$$
\begin{array} { r } { \mathbb { E } \left[ \ell _ { \lambda } ( Y , \Gamma ( X ) ) \right] = \mathbb { E } [ | \Gamma ( X ) | ] + \frac { 1 } { \lambda } \mathbb { P } ( Y \notin \Gamma ( X ) ) . } \end{array}\tag{21}
$$

We can decompose the risk into a sum over the elements of $\Gamma ( X )$ and its complement. Since $p ( \cdot \mid X )$ is the true conditional probability of Y given X,

$$
\mathbb { E } [ | \Gamma ( X ) | ] + \frac { 1 } { \lambda } \mathbb { P } ( Y \notin \Gamma ( X ) ) = \mathbb { E } \left[ \sum _ { y \in \Gamma ( X ) } 1 + \frac { 1 } { \lambda } \sum _ { y \notin \Gamma ( X ) } p ( y \mid X ) \right] .\tag{22}
$$

Using $\begin{array} { r } { \sum _ { y \in { \mathcal { y } } } p ( y \mid X ) = 1 } \end{array}$ , we have

$$
\mathbb { E } _ { X } \left[ \mathbb { E } _ { p ( \cdot | X ) } [ \ell _ { \lambda } ( Y , \Gamma ( X ) ) ] \right] = \frac { 1 } { \lambda } + \mathbb { E } \left[ \sum _ { y \in \Gamma ( X ) } \left( 1 - \frac { 1 } { \lambda } p ( y \mid X ) \right) \right] .\tag{23}
$$

Hence, $\Gamma ( X )$ minimizes the risk pointwise in x if and only if it includes every y with $p ( y \mid x ) > \lambda$ and excludes every $y$ with $p ( y \mid x ) < \lambda .$ . The decision rule that outputs probability superlevel sets

$$
\Gamma _ { \lambda } ^ { * } ( X ; p ) = \{ y \in \mathcal { Y } : p ( y \mid X ) \geq \lambda \}\tag{24}
$$

is one such minimizer for every $x \in \mathcal { X }$ and thus achieves the conformal entropy $H _ { \lambda } ( Y \mid X )$

## A.1.2 THEOREM 1: INTEGRAL REPRESENTATION OF SHANNON INFORMATION

We first establish the corresponding integral representation for the Shannon entropy.

Lemma 1 (Integral representation of Shannon entropy). Let H denote the Shannon entropy (in nats). For any random variable X,

$$
1 + H ( Y \mid X ) = \int _ { 0 } ^ { 1 } H _ { \lambda } ( Y \mid X ) d \lambda .\tag{25}
$$

Proof. Recall that the decision rule $\Gamma _ { \lambda } ^ { * } ( X ; p ) = \{ y : p ( y \mid X ) \geq \lambda \}$ achieves $H _ { \lambda } ( Y \mid X )$ . At this optimum value, each $y \in \mathcal { V }$ either increases the set size by one when $p ( y \mid x ) \geq \lambda$ or increases the miscoverage penalty term by $p ( y \mid x ) / \lambda$ otherwise. Hence,

$$
H _ { \lambda } ( Y \mid X ) = \mathbb { E } _ { X } \left[ \sum _ { y \in \mathcal { Y } } \operatorname* { m i n } \left\{ 1 , \frac { p ( y \mid X ) } { \lambda } \right\} \right] ,\tag{26}
$$

and therefore by Tonelli’s theorem,

$$
\int _ { 0 } ^ { 1 } H _ { \lambda } ( Y \mid X ) d \lambda = \mathbb { E } _ { X } \left[ \sum _ { y \in \mathcal { Y } } \int _ { 0 } ^ { 1 } \operatorname* { m i n } \left\{ 1 , \frac { p ( y \mid X ) } { \lambda } \right\} d \lambda \right] .\tag{27}
$$

Evaluating the inner integral,

$$
\int _ { 0 } ^ { 1 } \operatorname* { m i n } \left\{ 1 , { \frac { p ( y \mid X ) } { \lambda } } \right\} d \lambda = \underbrace { \int _ { 0 } ^ { p ( y \mid X ) } d \lambda } _ { p ( y \mid X ) } + \underbrace { \int _ { p ( y \mid X ) } ^ { 1 } { \frac { p ( y \mid X ) } { \lambda } } d \lambda } _ { - p ( y \mid X ) \log p ( y \mid X ) } .\tag{28}
$$

Substituting Equation 28 into Equation 27,

$$
\int _ { 0 } ^ { 1 } H _ { \lambda } ( Y \mid X ) d \lambda = \operatorname \operatorname { \mathbb { E } } _ { X } \left[ \sum _ { y \in \mathcal { Y } } p ( y \mid X ) - p ( y \mid X ) \log p ( y \mid X ) \right] = 1 + H ( Y \mid X ) .\tag{29}
$$

The integral representation of Shannon information now follows immediately from Lemma 1.

$$
H ( Y \mid Z ) - H ( Y \mid X , Z ) = \int _ { 0 } ^ { 1 } H _ { \lambda } ( Y \mid Z ) d \lambda - \int _ { 0 } ^ { 1 } H _ { \lambda } ( Y \mid X , Z ) d \lambda .\tag{30}
$$

Thus,

$$
I ( X ; Y \mid Z ) = \int _ { 0 } ^ { 1 } I _ { \lambda } ( X ; Y \mid Z ) d \lambda .\tag{31}
$$

## A.1.3 PROPOSITION 3: OPTIMAL SET-VALUED PREDICTORS FOR RANDOM Λ

Consider the risk when $\Lambda = \xi ( X )$ , where $\xi : \mathcal { X } \to ( 0 , 1 ]$ is a fixed measurable function. Using Equation 5 with a fixed value of $u \in [ 0 , 1 ]$

$$
\begin{array} { r } { \mathbb { E } \left[ \ell _ { \Lambda } ( Y , \Gamma ( X , u ) ) \right] = \mathbb { E } _ { X } \left[ \left| \Gamma ( X , u ) \right| \right] + \mathbb { E } _ { X } \left[ \frac { 1 } { \xi ( X ) } \mathbb { P } ( Y \notin \Gamma ( X , u ) \mid X ) \right] . } \end{array}\tag{32}
$$

Since $p ( \cdot \mid X )$ is the true conditional probability of $Y$ given $X$ , the risk decomposition in Equations 22 and 23 gives that

$$
\mathbb { E } \left[ \ell _ { \Lambda } ( Y , \Gamma ( X , u ) ) \right] = \mathbb { E } _ { X } \left[ \frac { 1 } { \xi ( X ) } + \sum _ { y \in \Gamma ( X , u ) } \left( 1 - \frac { 1 } { \xi ( X ) } p ( y \mid X ) \right) \right] .\tag{33}
$$

Hence, $\Gamma ( X , u )$ minimizes the risk pointwise in x if and only if it includes every y with $p ( y \mid x ) > \xi ( x )$ and excludes every y with $p ( y \mid x ) < \xi ( x )$ . Thus, any decision rule $\Gamma _ { \Lambda } ^ { \ast } ( X , U ; p )$ such that, for all u $\mathbf \xi , \in [ 0 , 1 ]$

$$
\big \{ y : p ( y \mid X ) > \xi ( X ) \big \} \subseteq \Gamma _ { \Lambda } ^ { * } ( X , u ; p ) \subseteq \big \{ y : p ( y \mid X ) \geq \xi ( X ) \big \} ,\tag{34}
$$

minimizes $\mathbb { E } \left[ \ell _ { \Lambda } ( Y , \Gamma ( X , U ) ) \right]$ ] and thus achieves the conformal entropy $H _ { \Lambda } ( Y \mid X )$ . The decision rule may choose to include or exclude labels at the boundary $p ( y \mid X ) \dot { = } \xi ( \dot { X } )$ depending on $U$

## A.1.4 LEMMA 2: GENERALIZED INFORMATION DPI

The data processing inequality for conformal prediction sets follows from a more general result for generalized mutual information. For completeness, we include the result here. Let $\bar { \boldsymbol { H _ { \ell } } }$ and $I _ { \ell }$ denote the generalized entropy and mutual information under a loss $\ell : \mathcal { V } \times \mathcal { A }  \mathbb { R }$ , as defined in Section $2 .$ Lemma 2 (Generalized information DPI). Let ℓ be any loss withfinite action space ${ \mathcal { A } } ,$ and suppose $Y  X  { \tilde { X } }$ forms a Markov chain. Then,

$$
H _ { \ell } ( Y \mid X ) \le H _ { \ell } ( Y \mid \tilde { X } ) .\tag{35}
$$

Equivalently, $I _ { \ell } ( \tilde { X } ; Y ) \le I _ { \ell } ( X ; Y )$

Proof. Fix any decision rule $g$ for the $\tilde { X }$ -problem. Since $Y \perp { \tilde { X } } \mid X$

$$
\mathbb { E } \big [ \ell ( Y , g ( \tilde { X } ) ) \mid X , \tilde { X } = \tilde { x } \big ] = \mathbb { E } \big [ \ell ( Y , g ( \tilde { x } ) ) \mid X \big ] \ge \operatorname* { i n f } _ { a \in \mathcal { A } } \mathbb { E } \big [ \ell ( Y , a ) \mid X \big ] .\tag{36}
$$

Taking the joint expectation over $( X , { \tilde { X } } )$

$$
\mathbb { E } \left[ \operatorname* { i n f } _ { a \in \boldsymbol { A } } \mathbb { E } \big [ \ell ( \boldsymbol { Y } , a ) \mid \boldsymbol { X } \big ] \right] \leq \mathbb { E } \big [ \ell ( \boldsymbol { Y } , g ( \tilde { \boldsymbol { X } } ) ) \big ] .\tag{37}
$$

Therefore,

$$
H _ { \ell } ( Y \mid X ) \leq \mathbb { E } { \bigl [ } \ell ( Y , g ( { \tilde { X } } ) ) { \bigr ] } .\tag{38}
$$

Minimizing the right-hand side over g gives $H _ { \ell } ( Y \mid X ) \le H _ { \ell } ( Y \mid \tilde { X } )$ . The equivalent form expressed in terms of mutual information follows by subtracting both sides from $H _ { \ell } ( { \dot { Y } } )$ □

## A.2 PROBABILITY SCORE RESULTS

We now prove our main results for the probability score in Section 3.2.

## A.2.1 THEOREM 2: PROBABILITY SCORE SANDWICH BOUND

We wish to bound the expected conformal set-size reduction

$$
\Delta _ { \alpha } = \mathbb { E } \big [ | { \mathcal { C } } _ { \alpha } ( X ^ { ( 1 ) } ) | \big ] - \mathbb { E } \big [ | { \mathcal { C } } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ]\tag{39}
$$

in terms of the conformal mutual information $I _ { \lambda } ,$ , for a fixed realization of the calibration data $\mathcal { D } _ { \mathrm { c a l } }$ Since $\mathcal { D } _ { \mathrm { c a l } }$ is fixed, we treat the calibrated thresholds $\hat { \lambda } _ { \alpha } ^ { ( 1 ) } , \hat { \lambda } _ { \alpha } ^ { ( 2 ) }$ and the training-conditional miscov erage rates $\varepsilon ^ { ( 1 ) } , \varepsilon ^ { ( 2 ) }$ as constants. A key intermediate quantity in our analysis is

$$
\rho : = H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) - H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) ,\tag{40}
$$

which compares the conformal entropies at the two different calibrated thresholds. We first establish a sandwich bound on $\rho ,$ then convert it into a bound on $\Delta _ { \alpha }$ using the relationship between conformal prediction sets and Bayes-optimal actions.

Lemma 3 (Probability score sandwich bound on $\rho )$ . Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha$ . For $i \in \{ 1 , 2 \}$ , define $\hat { \lambda } _ { \alpha } ^ { ( i ) } , \stackrel { \cdot } { I } ^ { ( i ) } , \stackrel { } { D } ^ { ( i ) }$ , and $\varepsilon ^ { ( i ) }$ as in Theorem 2, and assume that $\hat { \lambda } _ { \alpha } ^ { ( i ) } > 0 .$ . Then,

$$
I ^ { ( 2 ) } - D ^ { ( 1 ) } + \varepsilon ^ { ( 1 ) } \left( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \right) \ \leq \ \rho \ \leq \ I ^ { ( 1 ) } + D ^ { ( 2 ) } + \varepsilon ^ { ( 2 ) } \left( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \right) .\tag{41}
$$

Proof. In both bounds we insert an intermediate conformal entropy term that isolates $I _ { \lambda }$ and leaves a cross-threshold conformal entropy difference that we control with an appropriately chosen predictor. Let $p ( \cdot \mid X )$ be the true conditional distribution and $q ( \cdot \mid X )$ be the predicted conditional distribution. We start with the upper bound.

Upper bound: Adding and subtracting $H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } )$ decomposes ρ as

$$
\rho = I ^ { ( 1 ) } + H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } ) - H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) .\tag{42}
$$

It remains to upper bound the cross-threshold conformal entropy difference. By the definition of the conformal information divergence,

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \big ( Y \mid X ^ { ( 2 ) } \big ) = \mathbb { E } \big [ | \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } \big ( X ^ { ( 2 ) } ; q \big ) | \big ] + \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } \big ( X ^ { ( 2 ) } ; q \big ) \big ) - D ^ { ( 2 ) } .\tag{43}
$$

Under the $\ell _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } }$ loss, the same predictor has risk no smaller than $H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } )$ . Thus,

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \big ( Y \mid X ^ { ( 2 ) } \big ) \leq \mathbb { E } \big [ | \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } \big ( X ^ { ( 2 ) } ; q \big ) | \big ] + \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } \big ( X ^ { ( 2 ) } ; q \big ) \big ) .\tag{44}
$$

By Proposition 2, we have $\Gamma _ { \hat { \lambda } _ { - } ^ { ( 2 ) } } ^ { * } ( X ^ { ( 2 ) } ; q ) = \mathcal { C } _ { \alpha } ( X ^ { ( 2 ) } )$ , so the miscoverage probabilities in Equations 44 and 43 are both $\varepsilon ^ { ( 2 ) }$ . Subtracting Equation 43 from Equation 44 gives

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \big ( Y \mid X ^ { ( 2 ) } \big ) - H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \big ( Y \mid X ^ { ( 2 ) } \big ) \leq D ^ { ( 2 ) } + \varepsilon ^ { ( 2 ) } \bigg ( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) .\tag{45}
$$

Substituting Equation 45 into 42 establishes the upper bound.

Lower bound: Adding and subtracting $H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } )$ instead decomposes $\rho$ as

$$
\rho = I ^ { ( 2 ) } + H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) - H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } ) .\tag{46}
$$

It remains to lower bound the cross-threshold conformal entropy difference. By the definition of the conformal information divergence,

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \big ( Y \mid X ^ { ( 1 ) } \big ) = \mathbb { E } \big [ \big | \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } \big ( X ^ { ( 1 ) } ; q ) \big | \big ] + \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } \big ( X ^ { ( 1 ) } ; q \big ) \big ) - D ^ { ( 1 ) } .\tag{47}
$$

Under the $\ell _ { \hat { \lambda } _ { \alpha } ^ { \left( 2 \right) } }$ loss, the same predictor has risk no smaller than $H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } )$ . Thus,

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \bigl ( Y \mid X ^ { ( 1 ) } \bigr ) \leq \mathbb { E } \bigl [ \bigl | \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } \bigl ( X ^ { ( 1 ) } ; q \bigr ) \bigr | \bigr ] + \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \mathbb { P } \bigl ( Y \notin \Gamma _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } \bigl ( X ^ { ( 1 ) } ; q \bigr ) \bigr ) .\tag{48}
$$

By Proposition 2, we have $\Gamma _ { \hat { \lambda } _ { - } ^ { ( 1 ) } } ^ { * } ( X ^ { ( 1 ) } ; q ) = { \mathcal C } _ { \alpha } ( X ^ { ( 1 ) } )$ , so the miscoverage probabilities in Equations 48 and 47 are both $\varepsilon ^ { ( 1 ) }$ . Subtracting Equation 48 from Equation 47 gives

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \big ( \boldsymbol { Y } \mid \boldsymbol { X } ^ { ( 1 ) } \big ) - H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \big ( \boldsymbol { Y } \mid \boldsymbol { X } ^ { ( 1 ) } \big ) \ge \varepsilon ^ { ( 1 ) } \bigg ( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) - D ^ { ( 1 ) } .\tag{49}
$$

Substituting Equation 49 into Equation 46 establishes the lower bound.

We now convert the two-sided bound on $\rho$ in Lemma 3 into the corresponding bound on $\Delta _ { \alpha }$ in Theorem 2. Using Proposition 2, we can write the conformal entropy at $\hat { \lambda } _ { \alpha } ^ { \scriptscriptstyle ( 1 ) }$ and $\hat { \lambda } _ { \alpha } ^ { ( 2 ) }$ as

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \big ( \boldsymbol { Y } \mid \boldsymbol { X } ^ { ( 1 ) } \big ) = \mathbb { E } \big [ \big | \mathcal { C } _ { \boldsymbol { \alpha } } \big ( \boldsymbol { X } ^ { ( 1 ) } \big ) \big | \big ] + \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \mathbb { P } \big ( \boldsymbol { Y } \not \in \mathcal { C } _ { \boldsymbol { \alpha } } \big ( \boldsymbol { X } ^ { ( 1 ) } \big ) ; \mathcal { D } _ { \mathrm { c a l } } \big ) - D ^ { ( 1 ) } ,\tag{50}
$$

and

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \big ( Y \mid X ^ { ( 2 ) } \big ) = \mathbb { E } \big [ | { \mathcal C } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ] + \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \mathbb { P } \big ( Y \notin { \mathcal C } _ { \alpha } ( X ^ { ( 2 ) } ) ; { \mathcal D } _ { \mathrm { c a l } } \big ) - D ^ { ( 2 ) } .\tag{51}
$$

Recall $\rho = H _ { \hat { \lambda } _ { - } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) - H _ { \hat { \lambda } _ { - } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } )$ , and the miscoverage probabilities in Equations 50 and 51 are $\varepsilon ^ { ( 1 ) }$ and $\varepsilon ^ { ( 2 ) }$ , respectively. Subtracting the two identities,

$$
\Delta _ { \alpha } = \rho - \frac { \varepsilon ^ { ( 1 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } + \frac { \varepsilon ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } + D ^ { ( 1 ) } - D ^ { ( 2 ) } .\tag{52}
$$

We combine Lemma 3 with the identity in Equation 52 to obtain the corresponding bounds on $\Delta _ { \alpha }$ . Upper bound: Substituting the upper bound from Equation 41,

$$
\Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 2 ) } + \varepsilon ^ { ( 2 ) } \Bigg ( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \Bigg ) - \frac { \varepsilon ^ { ( 1 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } + \frac { \varepsilon ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } + D ^ { ( 1 ) } - D ^ { ( 2 ) } .\tag{53}
$$

Therefore,

$$
\Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 1 ) } + \frac { \varepsilon ^ { ( 2 ) } - \varepsilon ^ { ( 1 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } .\tag{54}
$$

Lower bound: Substituting the lower bound from Equation 41,

$$
\Delta _ { \alpha } \geq I ^ { ( 2 ) } - D ^ { ( 1 ) } + \varepsilon ^ { ( 1 ) } \Bigg ( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \Bigg ) - \frac { \varepsilon ^ { ( 1 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } + \frac { \varepsilon ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } + D ^ { ( 1 ) } - D ^ { ( 2 ) } .\tag{55}
$$

Therefore,

$$
\Delta _ { \alpha } \geq I ^ { ( 2 ) } - D ^ { ( 2 ) } + \frac { \varepsilon ^ { ( 2 ) } - \varepsilon ^ { ( 1 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } .\tag{56}
$$

This completes the proof.

## A.2.2 CONCENTRATION OF FINITE-SAMPLE MISCOVERAGE DIFFERENCE

The bounds in Equations 56 and 54 contain the finite-sample calibration terms $\varepsilon ^ { \Delta } / \hat { \lambda } _ { \alpha } ^ { ( i ) } \mathrm { f o r } i \in \{ 1 , 2 \}$ involving the difference in miscoverage rates. We now turn our attention to these terms under a random calibration dataset. Using standard arguments from conformal prediction, the marginal distribution of the coverage $1 - \varepsilon ^ { ( i ) }$ is that of an order statistic of a uniform sample. We make this result precise below and include the proof for completeness.<sup>3</sup>

Lemma 4 (Training-conditional coverage). Fix a nominal coverage level $1 - \alpha \in ( 0 , 1 )$ such that $k = \lceil ( 1 - \alpha ) ( n + \mathbf { \bar { 1 } } ) \rceil \leq n .$ . Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. Then,

$$
1 - \varepsilon ^ { ( i ) } \sim \mathrm { B e t a } ( k , n + 1 - k ) , \qquad f o r i \in \{ 1 , 2 \} .\tag{57}
$$

Proof. The argument is identical for $i \in \{ 1 , 2 \}$ , so we suppress the superscript. Let $S _ { 1 } , \ldots , S _ { n }$ be the non-conformity scores on the calibration data and $S _ { n + 1 }$ be the score of an independent test point. Let G denote the CDF of the score distribution. The test point is covered if and only if $S _ { n + 1 } \leq \hat { \tau } _ { \alpha } = S _ { ( k ) }$ . Hence, the training-conditional coverage is

$$
\begin{array} { r } { 1 - \varepsilon = \mathbb { P } \big ( S _ { n + 1 } \leq \hat { \tau } _ { \alpha } ; \mathcal { D } _ { \mathrm { c a l } } \big ) = G \big ( S _ { ( k ) } \big ) . } \end{array}\tag{58}
$$

If the scores are almost surely distinct, then G is continuous. Thus, $G ( S _ { j } ) \stackrel { \mathrm { \tiny ~ i . i . d . } } { \sim } \mathrm { \ U n i f } ( 0 , 1 )$ , so $G ( S _ { ( k ) } )$ is the k-th order statistic of n uniform samples, which follows Beta $( k , n + 1 - k )$ □

Because Lemma 4 applies for $i \in \{ 1 , 2 \}$ , the two miscoverages share the same mean. Thus,

$$
\mathbb { E } \big [ \varepsilon ^ { ( 1 ) } \big ] = \mathbb { E } \big [ \varepsilon ^ { ( 2 ) } \big ] = 1 - \frac { k } { n + 1 } = \frac { n + 1 - k } { n + 1 } \le \alpha .\tag{59}
$$

Since $\mathbb { E } [ \varepsilon ^ { \Delta } ] = 0$ , one might hope that the calibration term $\varepsilon ^ { \Delta } / \hat { \lambda } _ { \alpha } ^ { ( i ) }$ also vanishes in expectation. Unfortunately, this is not the case. Taking expectations gives

$$
\mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \right] = \mathrm { C o v } \left( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } , \varepsilon ^ { \Delta } \right) ,\tag{60}
$$

which is generally nonzero because the calibrated thresholds are correlated with the miscoverages. We provide two approaches to control $\varepsilon ^ { \Delta } / \hat { \lambda } _ { \alpha } ^ { ( i ) }$ without placing assumptions on the joint distribution of $\left( \bar { \varepsilon } ^ { ( 1 ) } , \varepsilon ^ { ( 2 ) } \right)$ . First, we analyze its expectation when the reciprocal of the calibrated threshold has finite variance. We then derive a high-probability guarantee that only requires the calibrated threshold to be almost surely positive.

Lemma 5 (Expectation bound on the calibration term). Fix a nominal coverage level $1 - \alpha .$ . Assume that the calibration scores are almost surely distinct and that the calibration data and future test samples are i.i.d. For $i \in \{ 1 , 2 \} , i f s ^ { ( i ) } = \mathrm { V a r } ( 1 / \hat { \lambda } _ { \alpha } ^ { ( i ) } ) ^ { 1 / 2 } < \infty ,$ , then

$$
\left| \mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \right] \right| \leq \frac { s ^ { ( i ) } } { \sqrt { n } } .\tag{61}
$$

Proof. Fix $i \in \{ 1 , 2 \}$ . Let $\mathrm { V a r } ( \varepsilon ^ { ( i ) } ) = \sigma ^ { 2 }$ . By Lemma 4,

$$
\sigma ^ { 2 } = \frac { \left( n + 1 - k \right) k } { ( n + 1 ) ^ { 2 } ( n + 2 ) } \leq \frac { 1 } { 4 ( n + 2 ) } \leq \frac { 1 } { 4 n } .\tag{62}
$$

Expanding the variance of $\varepsilon ^ { \Delta } = \varepsilon ^ { ( 2 ) } - \varepsilon ^ { ( 1 ) }$ and applying Cauchy–Schwarz to the cross term,

$$
\operatorname { V a r } ( \varepsilon ^ { \Delta } ) = 2 \sigma ^ { 2 } - 2 \operatorname { C o v } \left( \varepsilon ^ { ( 1 ) } , \varepsilon ^ { ( 2 ) } \right) \leq 2 \sigma ^ { 2 } + 2 { \sqrt { \operatorname { V a r } ( \varepsilon ^ { ( 1 ) } ) \operatorname { V a r } ( \varepsilon ^ { ( 2 ) } ) } } = 4 \sigma ^ { 2 } \leq 1 / n .\tag{63}
$$

Since $\mathbb { E } [ \varepsilon ^ { \Delta } ] = 0$ , Equation 60 and the Cauchy–Schwarz inequality give

$$
\left| \mathbb { E } \bigg [ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \bigg ] \right| = \bigg | \mathrm { C o v } \bigg ( \frac { 1 } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } , \varepsilon ^ { \Delta } \bigg ) \bigg | \leq \sqrt { \mathrm { V a r } ( 1 / \hat { \lambda } _ { \alpha } ^ { ( i ) } ) \mathrm { V a r } ( \varepsilon ^ { \Delta } ) } \leq \frac { s ^ { ( i ) } } { \sqrt { n } } .\tag{64}
$$

Lemma 5 only requires $1 / \hat { \lambda } _ { \alpha } ^ { ( i ) }$ to have finite variance. Under additional smoothness assumptions $( \mathrm { e . g . }$ , Lipschitz continuity of the reciprocal quantile function of the scores), the expectation can instead be bounded at a faster $O ( 1 / n )$ rate. Alternatively, we can relax the condition to almost-sure positivity and control the calibration term with high probability.

Lemma 6 (High-probability bound on the calibration term). Fix a nominal coverage level $1 - \alpha$ and $\delta \in ( 0 , 1 )$ . Assume that the calibration scores are almost surely distinct and that the calibration data and future test samples are i.i.d. Suppose that there exists $\lambda _ { \delta / 2 } > 0$ such that

$$
\begin{array} { r } { \mathbb { P } \big ( \operatorname* { m i n } \{ \hat { \lambda } _ { \alpha } ^ { ( 1 ) } , \hat { \lambda } _ { \alpha } ^ { ( 2 ) } \} \geq \lambda _ { \delta / 2 } \big ) \geq 1 - \frac { \delta } { 2 } . } \end{array}\tag{65}
$$

Then, with probability at least $1 - \delta$ over the draw of $\mathcal { D } _ { \mathrm { c a l } }$

$$
\left| \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \right| \leq \frac { 1 } { \lambda _ { \delta / 2 } } \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } , \qquad f o r b o t h i \in \{ 1 , 2 \} .\tag{66}
$$

Proof. By Lemma 4, we have $\varepsilon ^ { ( i ) } \sim \mathrm { B e t a } ( n + 1 - k , k )$ , and let $\mu$ denote their common mean. The $\mathrm { B e t a } ( \alpha , \bar { \beta } )$ distribution is sub-Gaussian with proxy variance $\sigma _ { p } ^ { 2 } \overset { \cdot } { = } 1 / ( 4 ( \alpha + \beta + 1 ) )$ (Marchal & Arbel, 2017). Hence, by the sub-Gaussian tail bound,

$$
\begin{array} { r } { \mathbb { P } \big ( | \varepsilon ^ { ( i ) } - \mu | \geq t \big ) \leq 2 \exp \big ( - t ^ { 2 } / ( 2 \sigma _ { p } ^ { 2 } ) \big ) = 2 \exp \big ( - 2 ( n + 2 ) t ^ { 2 } \big ) \leq 2 \exp \big ( - 2 n t ^ { 2 } \big ) . } \end{array}\tag{67}
$$

Setting the right-hand side equal to $\delta / 4$ gives $t _ { \delta } ~ = ~ \sqrt { \log ( 8 / \delta ) / ( 2 n ) }$ . By a union bound over $i \in \{ 1 , 2 \}$ , with probability at least $\textstyle 1 - { \frac { \delta } { 2 } }$ we have $| \varepsilon ^ { ( i ) } - \mu | \leq t _ { \delta }$ , and therefore

$$
\left. \varepsilon ^ { \Delta } \right. = \left. \left( \varepsilon ^ { \left( 2 \right) } - \mu \right) - \left( \varepsilon ^ { \left( 1 \right) } - \mu \right) \right. \leq 2 t _ { \delta } = \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } .\tag{68}
$$

By assumption, with probability at least $\textstyle 1 - { \frac { \delta } { 2 } }$ we have $\hat { \lambda } _ { \alpha } ^ { ( i ) } \ge \lambda _ { \delta / 2 }$ for both i. By a union bound, with probability at least $1 - \delta$

$$
\left| \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \right| \le \frac { | \varepsilon ^ { \Delta } | } { \lambda _ { \delta / 2 } } \le \frac { 1 } { \lambda _ { \delta / 2 } } \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } .\tag{69}
$$

□

## A.2.3 COROLLARY 1: PROBABILITY SCORE SANDWICH BOUND IN EXPECTATION

The result follows by applying Lemma 5 to Theorem 2. For $i \in \{ 1 , 2 \}$

$$
- \frac { s ^ { ( i ) } } { \sqrt { n } } \leq \mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \right] \leq \frac { s ^ { ( i ) } } { \sqrt { n } } .\tag{70}
$$

Taking expectations over the calibration data in Theorem 2 gives

$$
\begin{array} { r } { \mathbb { E } [ I ^ { ( 2 ) } ] - \mathbb { E } [ D ^ { ( 2 ) } ] + \mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \right] \ \leq \ \mathbb { E } [ \Delta _ { \alpha } ] \ \leq \ \mathbb { E } [ I ^ { ( 1 ) } ] + \mathbb { E } [ D ^ { ( 1 ) } ] + \mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } \right] . } \end{array}\tag{71}
$$

Substituting the appropriate lower and upper bounds from Equation 70 into 71 completes the proof.

## A.2.4 COROLLARY 3: PROBABILITY SCORE SANDWICH BOUND WITH HIGH PROBABILITY

Corollary 3 (Probability score sandwich bound with high probability). Fix a nominal coverage level $1 - \alpha$ and $\delta \in ( 0 , 1 )$ ). Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. Suppose that there exists $\lambda _ { \delta / 2 } > 0$ such that

$$
\begin{array} { r } { \operatorname { \mathbb { P } } ( \operatorname* { m i n } \{ \hat { \lambda } _ { \alpha } ^ { ( 1 ) } , \hat { \lambda } _ { \alpha } ^ { ( 2 ) } \} \geq \lambda _ { \delta / 2 } ) \geq 1 - \frac { \delta } { 2 } . } \end{array}\tag{72}
$$

Then, with probability at least $1 - \delta$ over the draw of $\mathcal { D } _ { \mathrm { c a l } }$

$$
I ^ { ( 2 ) } - D ^ { ( 2 ) } - \sqrt { \frac { 2 \log ( 8 / \delta ) } { n \lambda _ { \delta / 2 } ^ { 2 } } } \le \Delta _ { \alpha } \le I ^ { ( 1 ) } + D ^ { ( 1 ) } + \sqrt { \frac { 2 \log ( 8 / \delta ) } { n \lambda _ { \delta / 2 } ^ { 2 } } } .\tag{73}
$$

The result follows by applying Lemma 6 to Theorem 2. By Lemma 6, with probability at least $1 - \delta .$

$$
- \sqrt { \frac { 2 \log ( 8 / \delta ) } { n \lambda _ { \delta / 2 } ^ { 2 } } } \le \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } \le \sqrt { \frac { 2 \log ( 8 / \delta ) } { n \lambda _ { \delta / 2 } ^ { 2 } } } .\tag{74}
$$

Since Theorem 2 holds for every fixed calibration dataset with $\hat { \lambda } _ { \alpha } ^ { ( i ) } > 0 .$ , on this event

$$
I ^ { ( 2 ) } - D ^ { ( 2 ) } + \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \ \leq \ \Delta _ { \alpha } \ \leq \ I ^ { ( 1 ) } + D ^ { ( 1 ) } + \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 1 ) } } .\tag{75}
$$

Substituting the appropriate lower and upper bounds from Equation 74 into 75 completes the proof.

## A.2.5 THEOREM 3: PROBABILITY SCORE DPI

We convert the data processing inequality for generalized mutual information into a corresponding approximate result for conformal prediction sets for a fixed calibration dataset. Since $\Delta _ { \alpha }$ and $\tilde { \Delta } _ { \alpha }$ share the term $\mathbb { E } [ | { \mathcal { C } } _ { \alpha } ( X ^ { ( 1 ) } ) | ]$ ],

$$
\begin{array} { r } { \tilde { \Delta } _ { \alpha } - \Delta _ { \alpha } = \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ] - \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) | \big ] . } \end{array}\tag{76}
$$

Using Proposition 2 and the definition of the conformal information divergence, we can write

$$
\mathbb { E } \big [ | { \mathcal C } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ] = H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) - \frac { \varepsilon ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } + { \cal D } ^ { ( 2 ) } .\tag{77}
$$

For the degraded features, $\mathcal { C } _ { \alpha } \big ( \tilde { X } ^ { ( 2 ) } \big )$ is a feasible decision rule under the $\ell _ { \hat { \lambda } _ { \alpha } ^ { \left( 2 \right) } }$ loss, so its risk is no smaller than $H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } )$ . Evaluating the risk gives

$$
H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } ) \leq \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) | \big ] + \frac { \tilde { \varepsilon } ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } .\tag{78}
$$

Combining Equations 78 and 77,

$$
\tilde { \Delta } _ { \alpha } - \Delta _ { \alpha } \leq H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) - H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } ) + D ^ { ( 2 ) } + \frac { \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } .\tag{79}
$$

By Lemma 2, $H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) \leq H _ { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } )$ . Thus,

$$
\tilde { \Delta } _ { \alpha } - \Delta _ { \alpha } \leq D ^ { ( 2 ) } + \frac { \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } .\tag{80}
$$

This completes the proof.

## A.2.6 COROLLARY 2: PROBABILITY SCORE DPI IN EXPECTATION

Since $\tilde { \varepsilon } ^ { ( 2 ) }$ and $\varepsilon ^ { ( 2 ) }$ are both training-conditional miscoverages at coverage level $1 - \alpha ,$ , they share the same marginal distribution by Lemma 4. The result follows by applying Lemma 5 to Theorem 3 with $\varepsilon ^ { \Delta } = \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) }$ . Thus, we get

$$
\mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \right] \leq \frac { s ^ { ( 2 ) } } { \sqrt { n } } .\tag{81}
$$

Taking expectations over the calibration data in Theorem 3 gives

$$
\begin{array} { r } { \mathbb { E } [ \tilde { \Delta } _ { \alpha } ] \ \leq \ \mathbb { E } [ \Delta _ { \alpha } ] + \mathbb { E } [ D ^ { ( 2 ) } ] + \mathbb { E } \left[ \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \right] . } \end{array}\tag{82}
$$

Substituting Equation 81 into Equation 82 completes the proof.

## A.2.7 COROLLARY 4: PROBABILITY SCORE DPI WITH HIGH PROBABILITY

Corollary 4 (Probability score DPI with high probability). Fix a nominal coverage level $1 - \alpha$ and $\delta \in ( 0 , 1 )$ ). Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. Suppose that there exists $\lambda _ { \delta / 2 } > 0$ such that

$$
\begin{array} { r } { \mathbb { P } ( \hat { \lambda } _ { \alpha } ^ { ( 2 ) } \geq \lambda _ { \delta / 2 } ) \geq 1 - \frac { \delta } { 2 } . } \end{array}\tag{83}
$$

Then, with probability at least $1 - \delta$ over the draw of $\mathrm { \dot { \sigma } } _ { \mathrm { c a l } }$

$$
\tilde { \Delta } _ { \alpha } \leq \Delta _ { \alpha } + D ^ { ( 2 ) } + \sqrt { \frac { 2 \log ( 8 / \delta ) } { n \lambda _ { \delta / 2 } ^ { 2 } } } .\tag{84}
$$

The result follows by applying the proof in Lemma 6 to Theorem 3 with $\varepsilon ^ { \Delta } = \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) }$ . Since $\tilde { \varepsilon } ^ { ( 2 ) }$ and $\varepsilon ^ { ( 2 ) }$ are both training-conditional miscoverages at level $1 - \alpha ,$ they share the same marginal distribution by Lemma 4. Then, Lemma 6 gives, with probability at least $1 - \delta$

$$
\frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } \leq \sqrt { \frac { 2 \log ( 8 / \delta ) } { n \lambda _ { \delta / 2 } ^ { 2 } } } .\tag{85}
$$

Since Theorem 3 holds for every fixed calibration dataset with $\hat { \lambda } _ { \alpha } ^ { ( 2 ) } > 0$ , on this event

$$
\tilde { \Delta } _ { \alpha } \ \leq \ \Delta _ { \alpha } + D ^ { ( 2 ) } + \frac { \varepsilon ^ { \Delta } } { \hat { \lambda } _ { \alpha } ^ { ( 2 ) } } .\tag{86}
$$

Substituting the upper bound from Equation 85 into Equation 86 completes the proof.

## A.3 APS SCORE RESULTS

We now prove our results using the APS score from Section 3.4. The arguments mirror the proofs in Appendix A.2, with the fixed calibrated thresholds replaced by the input-dependent thresholds induced by the APS score. As in Section 2, we assume without loss of generality that there are no ties among the predicted probabilities, since any ties can be broken at random following Romano et al. (2020).

## A.3.1 REGULARITY OF THE INDUCED APS THRESHOLD

We first show that $\hat { \tau } _ { \alpha } < 1$ ensures $\hat { \boldsymbol { \Lambda } } _ { \alpha } \in ( 0 , 1 ]$ as well as boundedness of $\mathbb { E } [ 1 / \hat { \Lambda } _ { \alpha } ]$ and $D _ { \hat { \Lambda } _ { c } }$ α

Lemma 7 (APS threshold and divergence bounds). Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha$ . Let $\hat { \tau } _ { \alpha }$ be the calibrated APS threshold and let $\hat { \Lambda } _ { \alpha } = \xi ( X )$ be the induced probability threshold. $I f \hat { \tau } _ { \alpha } < 1$ , thenfor every $x \in \mathcal { X }$

$$
\xi ( x ) \ge \frac { 1 - \hat { \tau } _ { \alpha } } { K } .\tag{87}
$$

Consequently, $\hat { \boldsymbol { \Lambda } } _ { \alpha } \in ( 0 , 1 ]$ and $\mathbb { E } [ 1 / \hat { \Lambda } _ { \alpha } ] \leq K / ( 1 - \hat { \tau } _ { \alpha } )$ . Moreover,

$$
D _ { \hat { \Lambda } _ { \alpha } } \bigl ( p ( \cdot \mid X ) \parallel q ( \cdot \mid X ) \bigr ) \leq K + \frac { K } { 1 - \hat { \tau } _ { \alpha } } .\tag{88}
$$

Proof. Fix x and order the classes as in Section 2. Define $\begin{array} { r } { P _ { j } = \sum _ { \ell < j } q ( y ^ { ( \ell ) } \mid x ) } \end{array}$ with $P _ { 0 } = 0$

Since $P _ { K } = 1 > \hat { \tau } _ { \alpha }$ , there is a first index j with $P _ { j } > \hat { \tau } _ { \alpha } ,$ , and $\xi ( x ) = q ( y ^ { ( j ) } \mid x )$ by definition. Furthermore, we have $P _ { j - 1 } \leq \hat { \tau } _ { \alpha } , \mathrm { s o } P _ { j } = \hat { P } _ { j - 1 } + \xi \bar { ( x ) } \leq \hat { \tau } _ { \alpha } + \xi \bar { ( x ) }$ . Each of the $K - j$ remaining labels has predicted probability at most $\xi ( x )$ , so

$$
1 - { \hat { \tau } } _ { \alpha } - \xi ( x ) \leq 1 - P _ { j } = \sum _ { \ell > j } q ( y ^ { ( \ell ) } \mid x ) \leq ( K - j ) \xi ( x ) .\tag{89}
$$

Rearranging gives $1 - \hat { \tau } _ { \alpha } \le \left( K - j + 1 \right) \xi ( x ) \le K \xi ( x )$ , which is Equation 87. Taking expectations over X gives the bound on $\mathbb { E } [ 1 / \hat { \Lambda } _ { \alpha } ]$

For the divergence bound, the decision rule $\Gamma _ { \hat { \Lambda } _ { \alpha } } ^ { * } ( X , U ; q )$ has risk at most $K + \mathbb { E } [ 1 / \hat { \Lambda } _ { \alpha } ]$ when the conditional distribution is $p ( \cdot \mid X )$ . Equation 88 then follows from the definition of $D _ { \hat { \Lambda } _ { \alpha } }$ □

## A.3.2 THEOREM 4: APS SCORE SANDWICH BOUND

We wish to bound the expected conformal set-size reduction

$$
\Delta _ { \alpha } = \mathbb { E } \big [ | { \mathcal { C } } _ { \alpha } ( X ^ { ( 1 ) } ) | \big ] - \mathbb { E } \big [ | { \mathcal { C } } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ]\tag{90}
$$

in terms of the conformal mutual information $I _ { \Lambda }$ under a random index, for a fixed realization of the calibration data $\mathcal { D } _ { \mathrm { c a l } }$ . For $i \in \{ 1 , 2 \}$ , let $\boldsymbol { \xi } ^ { ( i ) }$ be the fixed measurable map sending an input to its induced probability threshold, so that $\hat { \Lambda } _ { \alpha } ^ { ( i ) } \stackrel { \smile } { = } \xi ^ { ( i ) } \big ( X ^ { ( i ) } \big )$ .

Since $\mathcal { D } _ { \mathrm { c a l } }$ is fixed, the conditional miscoverages $\varepsilon ^ { ( 1 ) } , \varepsilon ^ { ( 2 ) }$ are deterministic functions of $X ^ { ( 2 ) }$ , where $\varepsilon ^ { ( i ) } = \mathbb { P } ( Y \notin \mathcal { C } _ { \alpha } ( X ^ { ( i ) } ) \mid X ^ { ( 2 ) } ; \mathcal { D } _ { \mathrm { c a l } } )$ with the conditional probability taken over the fresh test label $Y \sim p ( \cdot \ | \ X ^ { ' ( 2 ) } )$ and the independent randomization $U \sim \dot { \mathrm { U n i f } } [ 0 , 1 ]$

A key intermediate quantity in our analysis is

$$
\rho : = H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) ,\tag{91}
$$

which compares the conformal entropies at the two different calibrated thresholds. We first establish a sandwich bound on $\rho ,$ then convert it into a bound on $\Delta _ { \alpha }$ by making use of the fact that conformal prediction sets using the APS score satisfy the Bayes-optimality criterion in Proposition 3.

Lemma 8 (APS score sandwich bound on $\rho )$ . Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha$ . For $i \in \{ 1 , 2 \}$ , assume that $\hat { \tau } _ { \alpha } ^ { ( i ) } < 1$ , and define $\hat { \Lambda } _ { \alpha } ^ { ( i ) } , I ^ { ( i ) } , D ^ { ( i ) }$ , and $\varepsilon ^ { ( i ) }$ as in Theorem 4. Then,

$$
I ^ { ( 2 ) } - D ^ { ( 1 ) } + \mathbb { E } \bigg [ \varepsilon ^ { ( 1 ) } \bigg ( \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) \bigg ] \leq \rho \leq I ^ { ( 1 ) } + D ^ { ( 2 ) } + \mathbb { E } \bigg [ \varepsilon ^ { ( 2 ) } \bigg ( \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) \bigg ] ,\tag{92}
$$

where the expectations are over $X ^ { ( 2 ) }$ , and both bounds arefinite.

Proof. The proof closely mirrors the argument in Lemma 3. Let $p ( \cdot \mid X )$ be the true conditional distribution of Y given X and $q ( \cdot \mid X )$ be the predicted conditional distribution. By Lemma 7, we have $\hat { \Lambda } _ { \alpha } ^ { ( i ) } \in ( 0 , 1 \cdot$ ] with $\mathbb { E } [ 1 / \hat { \Lambda } _ { \alpha } ^ { ( i ) } ]$ ] and $\dot { D } ^ { ( i ) }$ finite, so we can make use of Proposition 3 and every term in our analysis is finite.

Upper bound: Adding and subtracting $H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } )$ decomposes $\rho$ as

$$
\rho = I ^ { ( 1 ) } + H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) .\tag{93}
$$

It remains to upper bound the cross-threshold conformal entropy difference. By the definition of the conformal information divergence,

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) = \mathbb { E } \big [ | \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } ( X ^ { ( 2 ) } , U ; q ) | \big ] + \mathbb { E } \bigg [ \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } ( X ^ { ( 2 ) } , U ; q ) \mid X ^ { ( 2 ) } \big ) \bigg ] - D ^ { ( 2 ) } .\tag{94}
$$

Under the $\ell _ { \hat { \Lambda } _ { \alpha } ^ { \left( 1 \right) } }$ loss, the same predictor has risk no smaller than $H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } )$ . Thus,

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } ) \leq \mathbb { E } \big [ | \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } ( X ^ { ( 2 ) } , U ; q ) | \big ] + \mathbb { E } \bigg [ \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ^ { * } ( X ^ { ( 2 ) } , U ; q ) \mid X ^ { ( 2 ) } \big ) \bigg ] ~ .\tag{95}
$$

By Proposition 3, we can choose $\Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ^ { \ast } ( X ^ { ( 2 ) } , U ; q ) = { \mathcal { C } } _ { \alpha } ( X ^ { ( 2 ) } )$ since $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) } = \xi ^ { ( 2 ) } \bigl ( X ^ { ( 2 ) } \bigr )$

Hence, the miscoverage terms in Equations 94 and 95 become E $\big / \big [ \mathcal { E } ^ { ( 2 ) } / \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } \big ]$ and $\mathbb { E } \big [ \varepsilon ^ { ( 2 ) } / \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } \big ]$ , respectively. Subtracting Equation 94 from Equation 95 gives

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 2 ) } ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) \leq D ^ { ( 2 ) } + { \mathbb { E } } \biggl [ \varepsilon ^ { ( 2 ) } \biggl ( \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \biggr ) \biggr ] .\tag{96}
$$

Substituting Equation 96 into 93 establishes the upper bound.

Lower bound: Adding and subtracting $H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } )$ instead decomposes ρ as

$$
\rho = I ^ { ( 2 ) } + H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } ) .\tag{97}
$$

It remains to lower bound the cross-threshold conformal entropy difference. By the definition of the conformal information divergence and the law of iterated expectation,

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) = \mathbb { E } \big [ | \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } ( X ^ { ( 1 ) } , U ; q ) | \big ] + \mathbb { E } \bigg [ \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } ( X ^ { ( 1 ) } , U ; q ) \mid X ^ { ( 2 ) } \big ) \bigg ] - D ^ { ( 1 ) } .\tag{98}
$$

Under the $\ell _ { \hat { \Lambda } _ { \alpha } ^ { \left( 2 \right) } }$ loss, the same predictor has risk no smaller than $H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } )$ . Thus,

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 1 ) } ) \leq \mathbb { E } \big [ \big | \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } ( X ^ { ( 1 ) } , U ; q ) \big | \big ] + \mathbb { E } \bigg [ \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \mathbb { P } \big ( Y \notin \Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ^ { * } ( X ^ { ( 1 ) } , U ; q ) \mid X ^ { ( 2 ) } \big ) \bigg ] ~ .\tag{99}
$$

By Proposition 3, we can choose $\Gamma _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } ^ { \ast } ( X ^ { ( 1 ) } , U ; q ) = { \mathcal C } _ { \alpha } ( X ^ { ( 1 ) } )$ since $\hat { \Lambda } _ { \alpha } ^ { \scriptscriptstyle ( 1 ) } = \xi ^ { \scriptscriptstyle ( 1 ) } \big ( X ^ { \scriptscriptstyle ( 1 ) } \big )$

Hence, the miscoverage terms in Equations 98 and 99 become $\mathbb { E } \big [ \varepsilon ^ { ( 1 ) } / \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } \big ]$ and $\mathbb { E } \big [ \varepsilon ^ { ( 1 ) } / \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } \big ]$ , respectively. Subtracting Equation 99 from Equation 98 gives

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \big ( \cal Y \mid \cal X ^ { ( 1 ) } \big ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \big ( \cal Y \mid \cal X ^ { ( 1 ) } \big ) \geq \mathbb { E } \bigg [ \varepsilon ^ { ( 1 ) } \bigg ( \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) \bigg ] - D ^ { ( 1 ) } .\tag{100}
$$

Substituting Equation 100 into Equation 97 establishes the lower bound.

We now convert the two-sided bound on $\rho$ in Lemma 8 into the corresponding bound on $\Delta _ { \alpha }$ in Theorem 4. Because $\hat { \Lambda } _ { \alpha } ^ { ( i ) } = \xi ^ { ( i ) } \big ( X ^ { ( i ) } \big )$ and $\hat { \tau } _ { \alpha } ^ { ( i ) } ~ < ~ 1$ , Proposition 3 lets us write the conformal entropy at $\hat { \Lambda } _ { \alpha } ^ { \scriptscriptstyle ( 1 ) }$ and $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) }$ as

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \big ( \boldsymbol { Y } \mid \boldsymbol { X } ^ { ( 1 ) } \big ) = \mathbb { E } \big [ \vert \mathcal { C } _ { \boldsymbol { \alpha } } \big ( \boldsymbol { X } ^ { ( 1 ) } \big ) \vert \big ] + \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 1 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \bigg ] - D ^ { ( 1 ) } ,\tag{101}
$$

and

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \big ( \boldsymbol { Y } \mid \boldsymbol { X } ^ { ( 2 ) } \big ) = \mathbb { E } \big [ | \mathcal { C } _ { \boldsymbol { \alpha } } ( \boldsymbol { X } ^ { ( 2 ) } ) | \big ] + \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] - D ^ { ( 2 ) } ,\tag{102}
$$

where each term is finite by Lemma 7. Recall $\rho = H _ { \hat { \Lambda } _ { \sigma } ^ { ( 1 ) } } ( Y \mid X ^ { ( 1 ) } ) - H _ { \hat { \Lambda } _ { \sigma } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } )$ . Subtracting the two identities,

$$
\Delta _ { \alpha } = \rho - \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 1 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \bigg ] + \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] + D ^ { ( 1 ) } - D ^ { ( 2 ) } .\tag{103}
$$

We combine Lemma 8 with the identity in Equation 103 to obtain the corresponding bounds on $\Delta _ { \alpha }$ x • Upper bound: Substituting the upper bound from Equation 92,

$$
\Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 2 ) } + \mathbb { E } \bigg [ \varepsilon ^ { ( 2 ) } \bigg ( \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) \bigg ] - \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 1 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \bigg ] + \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] + D ^ { ( 1 ) } - D ^ { ( 2 ) } .\tag{104}
$$

Therefore,

$$
\Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 1 ) } + \mathbb { E } \Bigg [ \frac { \varepsilon ^ { ( 2 ) } - \varepsilon ^ { ( 1 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \Bigg ] .\tag{105}
$$

Lower bound: Substituting the lower bound from Equation 92,

$$
\Delta _ { \alpha } \geq I ^ { ( 2 ) } - D ^ { ( 1 ) } + \mathbb { E } \bigg [ \varepsilon ^ { ( 1 ) } \bigg ( \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } - \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ) \bigg ] - \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 1 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \bigg ] + \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] + D ^ { ( 1 ) } - D ^ { ( 2 ) } .\tag{106}
$$

Therefore,

$$
\Delta _ { \alpha } \geq I ^ { ( 2 ) } - D ^ { ( 2 ) } + \mathbb { E } \Bigg [ \frac { \varepsilon ^ { ( 2 ) } - \varepsilon ^ { ( 1 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \Bigg ] .\tag{107}
$$

This completes the proof.

## A.3.3 DECOMPOSITION OF THE APS CALIBRATION TERM

Unlike for the probability score, the calibration terms involve a threshold $\hat { \Lambda } _ { \alpha } ^ { \left( i \right) }$ and conditional miscoverage $\varepsilon ^ { ( i ) }$ which vary with the test input $X ^ { ( 2 ) }$ . We next isolate the part of the calibration term that concentrates with n from an input-dependent covariance that does not.

Lemma 9 (Decomposition of the APS calibration term). Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \bar { \alpha } . F o r i \in \{ 1 , 2 \}$ , assume $\hat { \tau } _ { \alpha } ^ { ( i ) } < 1$ and define

$$
\bar { \varepsilon } ^ { ( i ) } = \mathbb { P } \big ( Y \notin \mathcal { C } _ { \alpha } ( X ^ { ( i ) } ) ; \mathcal { D } _ { \mathrm { c a l } } \big ) , \qquad M ^ { ( i ) } = \mathbb { E } \bigg [ \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( i ) } } \bigg ] ,\tag{108}
$$

and let $\bar { \varepsilon } ^ { \Delta } = \bar { \varepsilon } ^ { ( 2 ) } - \bar { \varepsilon } ^ { ( 1 ) }$ . Then,

$$
\mathbb { E } _ { X ^ { ( 2 ) } } \bigg [ \frac { \varepsilon ^ { \Delta } } { \hat { \Lambda } _ { \alpha } ^ { ( i ) } } \bigg ] = \bar { \varepsilon } ^ { \Delta } M ^ { ( i ) } + R ^ { ( i ) } , \qquad w h e r e ~ R ^ { ( i ) } : = \mathrm { C o v } _ { X ^ { ( 2 ) } } \bigg ( \varepsilon ^ { \Delta } , \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( i ) } } \bigg ) .\tag{109}
$$

Moreover, let $\bar { \varepsilon } _ { Z } ^ { ( i ) } = \mathbb { P } ( Y \notin \mathcal { C } _ { \alpha } ( X ^ { ( i ) } ) \mid X ^ { ( 1 ) } ; \mathcal { D } _ { \mathrm { c a l } } )$ denote the $X ^ { ( 1 ) }$ -conditional miscoverage. Define

$$
\kappa ^ { ( 2 ) } = \sqrt { \mathrm { V a r } _ { X ^ { ( 2 ) } } \big ( \mathcal { E } ^ { ( 2 ) } - \mathcal { E } ^ { ( 1 ) } \big ) } , \qquad \bar { \kappa } ^ { ( 1 ) } = \sqrt { \mathrm { V a r } _ { X ^ { ( 1 ) } } \big ( \bar { \varepsilon } _ { Z } ^ { ( 2 ) } - \bar { \varepsilon } _ { Z } ^ { ( 1 ) } \big ) }\tag{110}
$$

and

$$
v ^ { ( 2 ) } = \sqrt { \mathrm { V a r } _ { X ^ { ( 2 ) } } \big ( 1 / \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } \big ) } , \qquad \bar { v } ^ { ( 1 ) } = \sqrt { \mathrm { V a r } _ { X ^ { ( 1 ) } } \big ( 1 / \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } \big ) } .\tag{111}
$$

Then, $| R ^ { ( 1 ) } | \leq \bar { \kappa } ^ { ( 1 ) } \bar { v } ^ { ( 1 ) }$ and $| R ^ { ( 2 ) } | \le \kappa ^ { ( 2 ) } v ^ { ( 2 ) }$

Proof. By the law of iterated expectation, $\bar { \varepsilon } ^ { ( i ) } = \mathbb { E } _ { X ^ { ( 2 ) } } [ \varepsilon ^ { ( i ) } ]$ . Thus, Equation 109 follows from the identity $\mathbb { E } [ A B ] = \mathbb { E } [ A ] \mathbb { E } [ B ] + { \mathrm { C o v } } ( A , B )$ with $A = \varepsilon ^ { \Delta }$ and $B = 1 / \hat { \Lambda } _ { \alpha } ^ { ( i ) }$ , together with Lemma 7. By the law of total covariance,

$$
\operatorname { C o v } _ { X ^ { ( 2 ) } } \left( \varepsilon ^ { \Delta } , \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \right) = \operatorname { C o v } _ { Z } \left( \mathbb { E } _ { X } [ \varepsilon ^ { \Delta } \mid Z ] , \frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( 1 ) } } \right) .\tag{112}
$$

By the law of iterated expectation, $\mathbb { E } _ { X } [ \varepsilon ^ { ( i ) } \mid Z ] = \bar { \varepsilon } _ { Z } ^ { ( i ) }$ . The bounds on $R ^ { ( i ) }$ now follow directly from the Cauchy–Schwarz inequality for covariance, $| \mathrm { C o v } ( A , B ) | \leq \sqrt { \mathrm { V a r } ( A ) \mathrm { V a r } ( B ) }$ , together with Lemma 7 to ensure that $\mathrm { V a r } ( 1 / \hat { \Lambda } _ { \alpha } ^ { ( i ) } )$ is finite. □

The κ factors quantify the spread of the coverage gap across inputs, while the v factors measure the spread of the induced APS threshold. These typically will not shrink with n. However, under an oracle model, $R ^ { ( 1 ) }$ vanishes since the randomized APS score achieves exact conditional coverage. We formalize this next.

Lemma 10 (Oracle conditional coverage). Assume the oracle model $q ( \cdot \mid X ^ { ( i ) } ) = p ( \cdot \mid X ^ { ( i ) } )$ for $i \in \{ 1 , 2 \}$ . Then, for each $i \in \{ 1 , 2 \}$ and every realization o $f \mathcal { D } _ { \mathrm { c a l } }$

$$
\mathbb { P } \big ( \boldsymbol { Y } \notin \mathcal { C } _ { \alpha } ( \boldsymbol { X } ^ { ( i ) } ) \mid \boldsymbol { X } ^ { ( i ) } ; \mathcal { D } _ { \mathrm { c a l } } \big ) = 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } \qquad a l m o s t s u r e l y ,\tag{113}
$$

where the conditional probability is over afresh label $Y \sim p ( \cdot \mid X ^ { ( i ) } )$ and $U \sim \mathrm { U n i f } [ 0 , 1 ]$

Proof. Fix x and order the classes so that $p ( y ^ { ( 1 ) } \mid x ) > \cdots > p ( y ^ { ( K ) } \mid x )$ . Write

$$
P _ { j } = \sum _ { \ell \le j } p ( y ^ { ( \ell ) } \mid x ) ,\tag{114}
$$

with $P _ { 0 } = 0 . { \mathrm { ~ I f ~ } } Y = y ^ { ( j ) }$ , the randomized APS score is $s ( x , Y ) = P _ { j - 1 } + U p ( y ^ { ( j ) } \mid x )$ , which is uniform on $( P _ { j - 1 } , P _ { j } ]$ . Since $\mathbb { P } ( Y = y ^ { ( j ) } \mid X = x ) = P _ { j } - P _ { j - 1 }$ , mixing over all $j$ gives $s ( x , Y ) \mid X = x \sim$ Unif $( 0 , 1 ]$ . Hence,

$$
\mathbb { P } \big ( Y \notin \mathcal { C } _ { \alpha } ( x ) \mid X = x \big ) = \mathbb { P } \big ( s ( x , Y ) > \hat { \tau } _ { \alpha } ^ { ( i ) } \mid X = x \big ) = 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } .\tag{115}
$$

Thus, Equation 113 holds almost surely.

□

Using Lemma 10 with $X ^ { ( 1 ) }$ , we get $\bar { \varepsilon } _ { Z } ^ { ( 1 ) } = 1 - \hat { \tau } _ { \alpha } ^ { ( 1 ) }$ <sup>)</sup>. With $X ^ { ( 2 ) }$ and the law of total probability,

$$
\bar { \varepsilon } _ { Z } ^ { ( 2 ) } = \mathbb { E } \bigl [ \varepsilon ^ { ( 2 ) } \mid X ^ { ( 1 ) } \bigr ] = \mathbb { E } \bigl [ 1 - \hat { \tau } _ { \alpha } ^ { ( 2 ) } \mid X ^ { ( 1 ) } \bigr ] = 1 - \hat { \tau } _ { \alpha } ^ { ( 2 ) } .\tag{116}
$$

Given a fixed calibration dataset, $\bar { \varepsilon } _ { Z } ^ { ( 2 ) } - \bar { \varepsilon } _ { Z } ^ { ( 1 ) } = \hat { \tau } _ { \alpha } ^ { ( 1 ) } - \hat { \tau } _ { \alpha } ^ { ( 2 ) }$ has zero variance and $\bar { \kappa } ^ { ( 1 ) } = 0$ . Thus, $R ^ { ( 1 ) } = 0$ under an oracle model. The same argument does not apply to $R ^ { ( 2 ) }$ , since $\varepsilon ^ { ( 1 ) }$ is conditioned on $X ^ { ( 2 ) }$ , whereas the corresponding predictor only observes $X ^ { ( 1 ) }$

## A.3.4 COROLLARY 5: APS SCORE SANDWICH BOUND IN EXPECTATION

Corollary 5 (APS score sandwich bound in expectation). Fix a nominal coverage level $1 - \alpha$ Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. For $i \in \{ 1 , 2 \}$ , define $I ^ { ( i ) }$ and $D ^ { ( i ) }$ as in Theorem 4 and $\kappa ^ { ( 2 ) } , \bar { \kappa } ^ { ( 1 ) } , v ^ { ( 2 ) } , \bar { v } ^ { ( \bar { 1 } ) }$ , and $M ^ { ( i ) }$ as in Lemma 9. Furthermore, suppose that $\mathbb { E } [ ( 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } ) ^ { - 2 } ] < \infty$ and let $n ^ { ( i ) } = \mathrm { V a r } ( M ^ { ( i ) } ) ^ { 1 / 2 }$ Then,

$$
\mathbb { E } [ I ^ { ( 2 ) } - D ^ { ( 2 ) } - \kappa ^ { ( 2 ) } v ^ { ( 2 ) } ] - \frac { m ^ { ( 2 ) } } { \sqrt { n } } \le \mathbb { E } [ \Delta _ { \alpha } ] \le \mathbb { E } [ I ^ { ( 1 ) } + D ^ { ( 1 ) } + \bar { \kappa } ^ { ( 1 ) } \bar { v } ^ { ( 1 ) } ] + \frac { m ^ { ( 1 ) } } { \sqrt { n } } ,\tag{117}
$$

where all expectations are over the calibration dataset $\mathcal { D } _ { \mathrm { c a l } } ,$ , and both bounds arefinite.

Proof. By Lemma 9,

$$
\mathbb { E } _ { X ^ { ( 2 ) } } \bigg [ \frac { \varepsilon ^ { \Delta } } { \hat { \Lambda } _ { \alpha } ^ { ( i ) } } \bigg ] = \bar { \varepsilon } ^ { \Delta } M ^ { ( i ) } + R ^ { ( i ) } .\tag{118}
$$

By Lemma 4, we have $1 - { \bar { \varepsilon } } ^ { ( i ) } \sim \operatorname { B e t a } ( k , n + 1 - k )$ , where $k = \lceil ( 1 - \alpha ) ( n + 1 ) \rceil$ . Hence, $\mathbb { E } \breve { \varepsilon } ^ { \Delta } \breve { = } 0$ and, following the proof of Lemma $^ { 5 , }$ we have $\mathrm { V a r } ( \bar { \varepsilon } ^ { \Delta } ) \leq 1 / \dot { n }$ . Lemma 7 also gives that $\begin{array} { r } { \dot { M } ^ { ( i ) } \le K / ( 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } ) } \end{array}$ , so $m ^ { ( i ) }$ is finite. By the Cauchy–Schwarz inequality,

$$
\left| \mathbb { E } [ \bar { \varepsilon } ^ { \Delta } M ^ { ( i ) } ] \right| = \left| \mathrm { C o v } \bigl ( \bar { \varepsilon } ^ { \Delta } , M ^ { ( i ) } \bigr ) \right| \leq \frac { m ^ { ( i ) } } { \sqrt { n } } .\tag{119}
$$

We also have $\mathbb { E } [ R ^ { ( 2 ) } ] \ge - \mathbb { E } \big [ \kappa ^ { ( 2 ) } v ^ { ( 2 ) } \big ]$ and $\mathbb { E } [ R ^ { ( 1 ) } ] \le \mathbb { E } \big [ \bar { \kappa } ^ { ( 1 ) } \bar { v } ^ { ( 1 ) } \big ]$ by Lemma 9. Taking expectations over $\mathcal { D } _ { \mathrm { c a l } }$ in Theorem 4 and then substituting Equation 118 together with the bounds on $\mathbb { E } [ R ^ { ( i ) } ]$ gives Equation 117.

It remains to show that all other terms are finite. Since $\mathbb { E } [ ( 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } ) ^ { - 2 } ] < \infty$ , we have $\hat { \tau } _ { \alpha } ^ { ( i ) } < 1$ almost surely, so Theorem 4 applies. Lemma 7 then gives

$$
\frac { 1 } { \hat { \Lambda } _ { \alpha } ^ { ( i ) } } \leq \frac { K } { 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } } , \qquad D ^ { ( i ) } \leq \frac { 2 K } { 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } } ,
$$

where the right-hand sides have finite expectation since $\mathbb { E } [ ( 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } ) ^ { - 2 } ] < \infty . \mathrm { S i n c e } v ^ { ( 2 ) }$ and $\bar { v } ^ { ( 1 ) }$ are standard deviations of $1 / \hat { \Lambda } _ { \alpha } ^ { ( i ) }$ over the test input, they are also bounded by $K / { ( 1 - \hat { \tau } _ { \alpha } ^ { ( i ) } ) }$ . Moreover, $\kappa ^ { ( 2 ) } , \bar { \kappa } ^ { ( 1 ) } \leq 1$ . Thus, both bounds are finite. □

## A.3.5 COROLLARY 6: APS SCORE SANDWICH BOUND WITH HIGH PROBABILITY

Corollary 6 (APS score sandwich bound with high probability). Fix a nominal coverage level $1 - \alpha$ and $\delta \in ( 0 , 1 )$ . Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. Suppose that there exists a constant $\eta _ { \delta / 2 } < 1$ such that

$$
\begin{array} { r } { \mathbb { P } \big ( \operatorname* { m a x } \big \{ \hat { \tau } _ { \alpha } ^ { ( 1 ) } , \hat { \tau } _ { \alpha } ^ { ( 2 ) } \big \} \leq \eta _ { \delta / 2 } \big ) \geq 1 - \frac { \delta } { 2 } . } \end{array}\tag{120}
$$

Let $\begin{array} { r } { c _ { \delta / 2 } = \frac { K } { 1 - \eta _ { \delta / 2 } } } \end{array}$ . Then, with probability at least $1 - \delta$ over the draw of $\mathbf { \nabla } \mathcal { D } _ { \mathrm { c a l } }$

$$
I ^ { ( 2 ) } - D ^ { ( 2 ) } - \sqrt { \frac { 2 c _ { \delta / 2 } ^ { 2 } \log ( 8 / \delta ) } { n } } - \kappa ^ { ( 2 ) } v ^ { ( 2 ) } \leq \Delta _ { \alpha } \leq I ^ { ( 1 ) } + D ^ { ( 1 ) } + \sqrt { \frac { 2 c _ { \delta / 2 } ^ { 2 } \log ( 8 / \delta ) } { n } } + \bar { \kappa } ^ { ( 1 ) } \bar { v } ^ { ( 1 ) } ,\tag{121}
$$

where $I ^ { ( i ) }$ and $D ^ { ( i ) }$ are defined as in Theorem 4 and $\bar { \kappa } ^ { ( 1 ) } , \ \kappa ^ { ( 2 ) } , \ \bar { v } ^ { ( 1 ) }$ , and $v ^ { ( 2 ) }$ are defined as in Lemma 9.

Proof. By Lemma $9 , \mathrm { i f } \hat { \tau } _ { \alpha } ^ { ( i ) } < 1$ , then

$$
\mathbb { E } _ { X ^ { ( 2 ) } } \bigg [ \frac { \varepsilon ^ { \Delta } } { \hat { \Lambda } _ { \alpha } ^ { ( i ) } } \bigg ] = \bar { \varepsilon } ^ { \Delta } M ^ { ( i ) } + R ^ { ( i ) } .\tag{122}
$$

By Lemma $4 , \bar { \varepsilon } ^ { ( i ) } \sim \mathrm { B e t a } ( n + 1 - k , k )$ . Following the proof of Lemma $^ { 6 , }$ each $\bar { \varepsilon } ^ { \left( i \right) }$ is sub-Gaussian with proxy variance $1 / ( 4 ( n + 2 ) )$ , so with probability at least $\textstyle 1 - { \frac { \delta } { 2 } }$

$$
| \bar { \varepsilon } ^ { \Delta } | \leq \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } .\tag{123}
$$

By assumption, with probability at least $1 - { \frac { \delta } { 2 } }$ we have $\hat { \tau } _ { \alpha } ^ { ( i ) } \leq \eta _ { \delta / 2 }$ for both $i ,$ so Lemma 7 gives $M ^ { ( i ) } \le c _ { \delta / 2 } . \mathrm { B y }$ a union bound, with probability at least $1 - \delta ,$

$$
\vert \bar { \varepsilon } ^ { \Delta } M ^ { ( i ) } \vert \leq c _ { \delta / 2 } \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } = \sqrt { \frac { 2 c _ { \delta / 2 } ^ { 2 } \log ( 8 / \delta ) } { n } } .\tag{124}
$$

Moreover, $R ^ { ( 2 ) } \geq - \kappa ^ { ( 2 ) } v ^ { ( 2 ) }$ and $R ^ { ( 1 ) } \leq \bar { \kappa } ^ { ( 1 ) }$ v¯<sup>(1)</sup> by Lemma 9. Since Theorem 4 holds for every fixed calibration dataset with $\hat { \tau } _ { \alpha } ^ { ( i ) } < 1$ , substituting these bounds completes the proof. □

## A.3.6 THEOREM 5: APS SCORE DPI

We extend the data processing inequality to the APS score. As in Section 3.3, suppose that for each data point we also observe a single realization of a degraded feature $\tilde { X }$ obtained from a (possibly noisy) channel, so that $Y  ( { \breve { X } } , Z )  ( { \tilde { X } } , Z )$ forms a Markov chain. Let $\tilde { X } ^ { ( 2 ) } = ( \tilde { X } , Z )$ and define the set-size reduction with the degraded features as

$$
\tilde { \Delta } _ { \alpha } = \mathbb { E } [ | \mathcal { C } _ { \alpha } ( X ^ { ( 1 ) } ) | ] - \mathbb { E } [ | \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) | ] .\tag{125}
$$

Theorem 5 (APS score DPI). Fix a calibration dataset $\mathcal { D } _ { \mathrm { c a l } }$ and nominal coverage level $1 - \alpha$ Define the conditional miscoverage $\tilde { \varepsilon } ^ { ( 2 ) } \underset { \sim } { = } \mathbb { P } ( Y \notin \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) \mid X ^ { ( 2 ) } ; \mathcal { D } _ { \mathrm { c a l } } )$ , where the probability i taken jointly over a noisy realization of X<sup>˜</sup> and fresh test label Y given $\dot { X } ^ { ( 2 ) }$ and the APS randomization U. Moreover, let $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) } , D ^ { ( 2 ) }$ <sup>)</sup>, and $\varepsilon ^ { ( 2 ) }$ be defined as in Theorem $^ { 4 , }$ and assume that $\hat { \tau } _ { \alpha } ^ { ( 2 ) } < 1$ Then,

$$
\begin{array} { r } { \tilde { \Delta } _ { \alpha } \leq \Delta _ { \alpha } + D ^ { ( 2 ) } + \mathbb { E } \big [ \big ( \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } \big ) / \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } \big ] , } \end{array}\tag{126}
$$

where the expectation is over $X ^ { ( 2 ) }$ , and the right-hand side isfinite.

Proof. We convert the data processing inequality for generalized mutual information into a corresponding approximate result for conformal prediction sets for a fixed calibration dataset. Since $\Delta _ { \alpha }$ and $\Delta _ { \alpha }$ share the term $\mathbb { E } [ | \mathcal { C } _ { \alpha } ( X ^ { ( 1 ) } ) | ]$ ],

$$
\begin{array} { r } { \tilde { \Delta } _ { \alpha } - \Delta _ { \alpha } = \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ] - \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) | \big ] . } \end{array}\tag{127}
$$

Because $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) } = \xi ^ { ( 2 ) } \bigl ( X ^ { ( 2 ) } \bigr )$ is a function of $X ^ { ( 2 ) }$ , using Proposition 3 gives

$$
\mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( X ^ { ( 2 ) } ) | \big ] = H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) - \mathbb { E } \bigg [ \frac { \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] + D ^ { ( 2 ) } ,\tag{128}
$$

where each term is finite by Lemma 7. For the degraded features, $\mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } )$ is a feasible decision rule under the $\ell _ { \hat { \Lambda } _ { \alpha } ^ { \left( 2 \right) } }$ loss, so its risk is no smaller than $H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } )$ . Evaluating the risk and using the law of iterated expectation gives

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \big ( Y \mid \tilde { X } ^ { ( 2 ) } \big ) \leq \mathbb { E } \big [ | \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) | \big ] + \mathbb { E } \bigg [ \frac { \tilde { \varepsilon } ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] .\tag{129}
$$

Combining Equations 129 and 128,

$$
\tilde { \Delta } _ { \alpha } - \Delta _ { \alpha } \leq H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } ) + D ^ { ( 2 ) } + \mathbb { E } \bigg [ \frac { \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] .\tag{130}
$$

Since $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) } = \xi ^ { ( 2 ) } \bigl ( X ^ { ( 2 ) } \bigr )$ is a function of $X ^ { ( 2 ) }$ , the loss index is fixed given $X ^ { ( 2 ) }$ and the proof of Lemma 2 applies to $\ell _ { \hat { \Lambda } _ { \alpha } ^ { \left( 2 \right) } }$ , so $H _ { \hat { \Lambda } _ { \sigma } ^ { ( 2 ) } } ( Y \mid X ^ { ( 2 ) } ) \leq H _ { \hat { \Lambda } _ { \sigma } ^ { ( 2 ) } } ( Y \mid \tilde { X } ^ { ( 2 ) } )$ . Thus,

$$
\tilde { \Delta } _ { \alpha } - \Delta _ { \alpha } \leq D ^ { ( 2 ) } + \mathbb { E } \Bigg [ \frac { \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \Bigg ] .\tag{131}
$$

Finiteness of each term on the right-hand side follows from Lemma 7.

As with the sandwich bound in Theorem $^ { 4 , }$ we provide versions of the DPI either in expectation or with high probability over the calibration data.

## A.3.7 COROLLARY 7: APS SCORE DPI IN EXPECTATION

Corollary 7 (APS score DPI in expectation). Fix a nominal coverage level $1 - \alpha .$ . Assume that the calibration scores are almost surely distinct and that the calibration data and future test samples are i.i.d. Define $D ^ { ( 2 ) }$ as in Theorem 4 and $v ^ { ( 2 ) }$ as in Lemma 9, and let $\tilde { \kappa } ^ { ( 2 ) } = \sqrt { \mathrm { V a r } _ { X ^ { ( 2 ) } } \big ( \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } \big ) }$ Moreover, suppose $\hat { \tau } _ { \alpha } ^ { ( 2 ) }$ satisfies the conditions ofCorollary ${ 5 , }$ , with $m ^ { ( 2 ) }$ defined accordingly. Then,

$$
\mathbb { E } [ \tilde { \Delta } _ { \alpha } ] \le \mathbb { E } [ \Delta _ { \alpha } ] + \mathbb { E } [ D ^ { ( 2 ) } ] + \frac { m ^ { ( 2 ) } } { \sqrt { n } } + \mathbb { E } \big [ \tilde { \kappa } ^ { ( 2 ) } v ^ { ( 2 ) } \big ] ,\tag{132}
$$

where all expectations are over the calibration dataset $\mathcal { D } _ { \mathrm { c a l } } ,$ , and the right-hand side isfinite.

Proof. The proof follows the same argument as Corollary 5. Define $\bar { \mathcal { E } } ^ { ( 2 ) } : = \mathbb { P } ( Y \notin \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) ; \mathcal { D } _ { \mathrm { c a l } } )$ and $\bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } : = \bar { \tilde { \varepsilon } } ^ { ( 2 ) } - \bar { \varepsilon } ^ { ( 2 ) }$ . By Lemma 9 with $\bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta }$

$$
\mathbb { E } _ { X ^ { ( 2 ) } } \bigg [ \frac { \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] = \bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } M ^ { ( 2 ) } + R _ { \mathrm { D P I } } ^ { ( 2 ) } , \qquad | R _ { \mathrm { D P I } } ^ { ( 2 ) } | \le \tilde { \kappa } ^ { ( 2 ) } v ^ { ( 2 ) } .\tag{133}
$$

Since $\bar { \tilde { \varepsilon } } ^ { ( 2 ) }$ and $\bar { \varepsilon } ^ { ( 2 ) }$ are both training-conditional miscoverages at coverage level $1 - \alpha$ , Lemma 4 gives that they share the same marginal distribution. Following the argument for Equation 119 with $\bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta }$ in place of $\bar { \varepsilon } ^ { \Delta }$ ,

$$
\left| \mathbb { E } \big [ \bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } M ^ { ( 2 ) } \big ] \right| \leq \frac { m ^ { ( 2 ) } } { \sqrt { n } } .\tag{134}
$$

Taking expectations over $\mathcal { D } _ { \mathrm { c a l } }$ in Theorem 5 and substituting Equations 133 and 134 completes the proof. All terms on the right-hand side are finite using the same argument as in Corollary 5.

## A.3.8 COROLLARY 8: APS SCORE DPI WITH HIGH PROBABILITY

Corollary 8 (APS score DPI with high probability). Fix a nominal coverage level $1 - \alpha$ and $\delta \in ( 0 , 1 )$ . Assume that the calibration scores are almost surely distinct and that the calibration data andfuture test samples are i.i.d. Suppose that there exists a constant $\eta _ { \delta / 2 } < 1$ such that

$$
\begin{array} { r } { \mathbb { P } \big ( \hat { \tau } _ { \alpha } ^ { ( 2 ) } \leq \eta _ { \delta / 2 } \big ) \geq 1 - \frac { \delta } { 2 } . } \end{array}\tag{135}
$$

Let $\begin{array} { r } { c _ { \delta / 2 } = \frac { K } { 1 - \eta _ { \delta / 2 } } } \end{array}$ . Then, with probability at least $1 - \delta$ over the draw of $\mathcal { D } _ { \mathrm { c a l } }$

$$
\tilde { \Delta } _ { \alpha } \leq \Delta _ { \alpha } + D ^ { ( 2 ) } + \sqrt { \frac { 2 c _ { \delta / 2 } ^ { 2 } \log ( 8 / \delta ) } { n } } + \tilde { \kappa } ^ { ( 2 ) } v ^ { ( 2 ) } ,\tag{136}
$$

where $D ^ { ( 2 ) }$ is defined as in Theorem 4 and $\tilde { \kappa } ^ { ( 2 ) }$ and $v ^ { ( 2 ) }$ are defined as in Corollary 7.

Proof. The proof follows the same argument as Corollary 6. Define $\bar { \mathcal { \tilde { E } } } ^ { ( 2 ) } : = \mathbb { P } ( Y \notin \mathcal { C } _ { \alpha } ( \tilde { X } ^ { ( 2 ) } ) ; \mathcal { D } _ { \mathrm { c a l } } )$ and $\bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } : = \bar { \tilde { \varepsilon } } ^ { ( 2 ) } - \bar { \varepsilon } ^ { ( 2 ) }$ as in Corollary 7. By Lemma 9, if $\hat { \tau } _ { \alpha } ^ { ( 2 ) } < 1$ , then

$$
\mathbb { E } _ { X ^ { ( 2 ) } } \bigg [ \frac { \tilde { \varepsilon } ^ { ( 2 ) } - \varepsilon ^ { ( 2 ) } } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \bigg ] = \bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } M ^ { ( 2 ) } + R _ { \mathrm { D P I } } ^ { ( 2 ) } , \qquad | R _ { \mathrm { D P I } } ^ { ( 2 ) } | \le \tilde { \kappa } ^ { ( 2 ) } v ^ { ( 2 ) } .\tag{137}
$$

Since $\bar { \tilde { \varepsilon } } ^ { ( 2 ) }$ and $\bar { \varepsilon } ^ { ( 2 ) }$ are both training-conditional miscoverages at coverage level $1 - \alpha ,$ Lemma 4 gives that they share the same marginal distribution. Following the proof of Lemma 6, with probability at least $1 \textstyle { \dot { - } } { \frac { \delta } { 2 } }$

$$
| \bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } | \leq \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } .\tag{138}
$$

By assumption, with probability at least $1 - \textstyle { \frac { \delta } { 2 } }$ we have $\hat { \tau } _ { \alpha } ^ { ( 2 ) } \leq \eta _ { \delta / 2 } .$ , so Lemma 7 gives $M ^ { ( 2 ) } \le c _ { \delta / 2 }$ By a union bound, with probability at least $1 ^ { - } - \delta .$

$$
\vert \bar { \varepsilon } _ { \mathrm { D P I } } ^ { \Delta } M ^ { ( 2 ) } \vert \leq c _ { \delta / 2 } \sqrt { \frac { 2 \log ( 8 / \delta ) } { n } } = \sqrt { \frac { 2 c _ { \delta / 2 } ^ { 2 } \log ( 8 / \delta ) } { n } } .\tag{139}
$$

Since Theorem 5 holds for every fixed calibration dataset with $\hat { \tau } _ { \alpha } ^ { ( 2 ) } < 1$ , substituting Equation 139 together with $R _ { \mathrm { D P I } } ^ { ( 2 ) } \le \tilde { \kappa } ^ { ( 2 ) } v ^ { ( 2 ) }$ completes the proof. □

## B DATASETS

We provide additional details on the synthetic and real datasets used in our experiments. The dataset statistics, such as sample sizes and feature dimensions, are summarized in Table 2.

## B.1 DATASET DETAILS

We provide a brief overview of the datasets. Throughout, we will use ⊕ as a concatenation operator.

S1. The first synthetic dataset samples the features $Z \oplus X \sim { \mathcal { N } } ( 0 , I _ { d } )$ , where $d = 4$ . The first two coordinates are the base features $Z$ and the remaining two coordinates are the additional features $X$ The label Y is sampled as

$$
Y \mid X , Z \sim { \mathrm { C a t e g o r i c a l } } { \big ( } { \mathrm { s o f t m a x } } ( W _ { 2 } { \mathrm { R e L U } } ( W _ { 1 } ( Z \oplus X ) + b _ { 1 } ) + b _ { 2 } ) { \big ) } ,\tag{140}
$$

with $K = 1 0$ classes and hidden width $h = 1 6$

The weights $W _ { 1 } \in \mathbb { R } ^ { h \times d } , W _ { 2 } \in \mathbb { R } ^ { K \times h }$ and biases $b _ { 1 } \in \mathbb { R } ^ { h } , b _ { 2 } \in \mathbb { R } ^ { K }$ are drawn once, with i.i.d. entries $\bar { W _ { 1 } } \tilde { \mathbf { \Gamma } } \sim \mathcal { N } ( 0 , 1 / d ) , W _ { 2 } \sim \mathcal { N } ( 0 , 1 / h )$ , and $b _ { 1 } , b _ { 2 } \sim \mathcal { N } ( 0 , 0 . 0 1 )$ ). Columns of $W _ { 1 }$ corresponding to X are scaled by a factor of 4. We compute the marginal oracle models $p ( \cdot \mid Z )$ by Gauss–Hermite quadrature, while $p ( \cdot \mid X , Z )$ is available in closed form.

S2. The second synthetic dataset samples a label $Y \sim { \mathrm { U n i f } } \{ 0 , \dots , K - 1 \}$ with $K = 5 0$ classes and features $Z \oplus \dot { X } \mid Y = k \sim \mathcal { N } ( \dot { \mu _ { k } } , I _ { d } )$ , where $d = 2 0$ . The first 10 coordinates are the base features $Z$ and the remaining 10 coordinates are the additional features X.

Each class mean $\mu _ { k } = \mu _ { k } ^ { Z } \oplus \mu _ { k } ^ { X }$ is drawn once by sampling a vector of i.i.d. standard normal entries in each block, then rescaling it to a fixed norm where $\| \mu _ { k } ^ { Z } \| \approx \sqrt { 2 }$ and $\| \mu _ { k } ^ { X } \| \approx 2$ . Since $Y$ is uniform and the features are Gaussian, the oracle posteriors $p ( \cdot \mid Z )$ and $p ( \cdot \mid \bar { X } , Z )$ are available in closed form by Bayes’ rule.

S3. The third synthetic dataset is designed for the feature selection experiment. We use $K = 4 0$ classes separated into a small set of 8 head classes containing 70% of the prior mass and a large set of 32 tail classes containing the remaining 30% of the prior mass. The base feature is

$$
Z \mid Y = k \sim \mathcal { N } ( c _ { k } + \delta _ { k } , I _ { d _ { Z } } ) ,\tag{141}
$$

with $d _ { Z } = 8 ,$ , where $c _ { k } = c _ { \mathrm { h e a d } }$ for head classes and $c _ { k } = c _ { \mathrm { t a i l } }$ for tail classes. Each center $c _ { k }$ has norm 2 and $\delta _ { k }$ is a smaller per-class offset of norm 0.5.

A pool of $m = 5 0$ scalar candidate features $X _ { 1 } , \ldots , X _ { m }$ is observed independently given Y ,

$$
X _ { j } \mid Y = k \sim { \mathcal { N } } ( a _ { j } \dim _ { j } [ k ] , 1 ) ,\tag{142}
$$

where $a _ { j } \in \mathbb { R }$ is a per-candidate amplitude and di $\mathbf { \Psi } _ { : j } \in \mathbb { R } ^ { K }$ is a per-candidate direction vector over the classes. Candidates are split across three roles, which set $a _ { j }$ and $\operatorname { d i r } _ { j }$ as follows:

• 18 head detectors: the 8 head classes are randomly partitioned into two groups of 4. The direction $\operatorname { d i r } _ { j }$ is zero for all non-head classes and takes values $\bar { . } \pm 1 / \sqrt { 8 }$ on the two head groups, with opposite signs for each group, so that dir has unit norm. The amplitude is $a _ { j } = 2 ;$

• 18 tail detectors: the same construction, splitting the 32 tail classes into two groups of 16;

• 14filler features: either dir<sub>j</sub> is a random unit vector with $a _ { j } = 0 . 4 \mathrm { o r } \mathrm { d i r } _ { j } = 0 \mathrm { . }$

The tail detector amplitude is calibrated by bisection so that the average Shannon information $I ( X _ { j } ; Y \mid Z )$ across tail detectors matches that across head detectors. Since the base and candidate features are conditionally independent given Y, the posterior $p ( y \mid Z , X _ { S } )$ is available in closed form via Bayes’ rule for any subset $S \subseteq \overbar { \{ 1 , \ldots , m \} }$

Note that S3 is relatively complex because it is designed to exhibit clear cases where conformal setsize reduction and Shannon mutual information disagree in feature selection. We observe similar behavior on several real-world datasets, albeit to a lesser extent.

An adapted version of S3 is used for Figure 2, with $K = 2 4$ classes split into head, tail, and filler groups and $X _ { 1 } , X _ { 2 } \in \mathbb { R } ^ { 8 }$ . The class-conditional mean of $X _ { 1 }$ is a distinct random vector for each of the 8 tail classes and zero otherwise, and $X _ { 2 }$ is defined analogously for the 4 head classes.

Table 2: Sample sizes, feature dimensions, and number of classes for all datasets. For the synthetic datasets S1–S3, the conditional distributions are known and no training data are required. Here, $n _ { \mathrm { e v a l } }$ denotes the number of samples available for calibration and evaluation. For tabular datasets, the input features are partitioned into X and Z. For image datasets, X and $Z$ correspond to separate image representations with the indicated channel dimensions. The number of candidate features in the feature selection experiments is denoted by m.
<table><tr><td>Dataset</td><td> $n _ { \mathrm { t o t a l } }$ </td><td> $n _ { \mathrm { e v a l } }$ </td><td> $d _ { \mathrm { t o t a l } }$ </td><td> $d _ { X }$ </td><td> $d _ { Z }$ </td><td></td><td>m</td><td>K</td></tr><tr><td>S1</td><td>200,000</td><td>200,000</td><td>4</td><td>2</td><td>2</td><td></td><td></td><td>10</td></tr><tr><td>S2</td><td>200,000</td><td>200,000</td><td>20</td><td>10</td><td>10</td><td></td><td>10</td><td>50</td></tr><tr><td>S3</td><td>200,000</td><td>200,000</td><td>58</td><td>50</td><td>8</td><td></td><td>50</td><td>40</td></tr><tr><td>Wine</td><td>4,898</td><td>1,224</td><td>11</td><td>6</td><td>5</td><td></td><td>11</td><td>7</td></tr><tr><td>HAR</td><td>10,299</td><td>2,575</td><td>561</td><td>348</td><td>213</td><td></td><td>51</td><td>6</td></tr><tr><td>Letter</td><td>20,000</td><td>5,000</td><td>16</td><td>9</td><td>7</td><td></td><td>16</td><td>26</td></tr><tr><td>Covertype</td><td>581,012</td><td>145,253</td><td>54</td><td>10</td><td>44</td><td></td><td>54</td><td>7</td></tr><tr><td>FMNIST</td><td>70,000</td><td>10,000</td><td>2 × 28 × 28</td><td>1 × 28 × 28</td><td>1 × 28 × 28</td><td></td><td>64</td><td>10</td></tr><tr><td>CIFAR-10</td><td>60,000</td><td>10,000</td><td>6 × 32 × 32</td><td>3 × 32 × 32</td><td>3 × 32 × 32</td><td></td><td>64</td><td>10</td></tr><tr><td>CIFAR-100</td><td>60,000</td><td>10,000</td><td>6 × 32 × 32</td><td>3 × 32 × 32</td><td>3 × 32 × 32</td><td></td><td>64</td><td>100</td></tr><tr><td>ImageNet-1k</td><td>1,331,167</td><td>50,000</td><td>6 × 64 × 64</td><td>3 × 64 × 64</td><td>3 × 64 × 64</td><td></td><td>64</td><td>1000</td></tr></table>

Wine Quality (Wine). The Wine Quality dataset (Cortez et al., 2009) contains physicochemical measurements and quality ratings for red and white wines. We use the white wine subset, which contains 4,898 samples with 11 physicochemical features. The prediction target is the wine quality rating ranging from 3 to 9.

Human Activity Recognition (HAR). The Human Activity Recognition dataset (Reyes-Ortiz et al., 2013) contains smartphone sensor measurements collected from participants performing six activities. We use the preprocessed feature representation, consisting of 10,299 samples with 561 features. The prediction target is one of six activities: walking, walking upstairs, walking downstairs, sitting, standing, or laying.

Letter Recognition (Letter). The Letter Recognition dataset (Slate, 1991) contains 20,000 samples of capital English letters, each represented by 16 numerical features extracted from the corresponding character image. The prediction target is one of the 26 letters of the English alphabet.

Covertype. The Covertype dataset (Blackard, 1998) contains 581,012 samples describing forested areas using 54 cartographic features, comprising 10 quantitative features and 44 binary features encoding wilderness area and soil type information. The prediction target is one of seven forest cover types.

Fashion-MNIST (FMNIST). Fashion-MNIST (Xiao et al., 2017) consists of 70,000 grayscale 28 × 28 images of fashion items from 10 classes. The classes are T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot. The dataset is split into 60,000 training samples and 10,000 test samples.

CIFAR-10. CIFAR-10 (Krizhevsky, 2009) consists of 60,000 RGB 32×32 images from 10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset contains 50,000 training samples and 10,000 test samples.

CIFAR-100. CIFAR-100 (Krizhevsky, 2009) consists of 60,000 RGB 32 × 32 images from 100 object classes. The dataset contains 50,000 training samples and 10,000 test samples.

ImageNet-1k. ImageNet-1k (Deng et al., 2009) is a large-scale image classification dataset containing images from 1,000 object classes. We use the 64 × 64 downsampled version of the dataset (Chrabaszcz et al., 2017) containing 1,281,167 training samples and 50,000 test samples.

![](images/b6d10f303f0d9805ee6348753d931ef0264d9f7471cff027923d0cf7f81cb629.jpg)  
Figure 4: Examples of the original image, base, new, and degraded features for each image dataset. The degraded features apply an additional Gaussian blur with standard deviation γσ to X.

## B.2 FEATURE CONSTRUCTION DETAILS

We provide additional details on the construction of the base features Z and additional features X for each dataset. For the synthetic datasets, Z consists of the first $d _ { Z }$ feature coordinates and X consists of the remaining $d _ { X }$ coordinates.

For the tabular datasets, we partition the input features into X and Z based on natural groupings of the feature types, as detailed below.

Wine Quality (Wine). The base features Z consist of five physical properties: chlorides, density, residual sugar, alcohol content, and pH. The additional features X consist of the remaining six acidity- and sulfur-related measurements.

Human Activity Recognition (HAR). The base features Z consist of the 213 gyroscope-derived measurements, including time- and frequency-domain statistics computed from the gyroscope signals. The additional features X consist of the remaining 348 features, including accelerometerderived measurements and the three angle-between-vectors features.

Letter Recognition (Letter). The base features Z consist of seven statistical moments of the x- and y-pixel positions. The additional features X consist of the remaining nine features: five bounding-box features describing position, width, height, and pixel count, and four edge-count and edge-correlation features.

Covertype. The base features Z consist of the 44 binary features encoding wilderness area and soil type. The additional features X consist of the 10 quantitative cartographic variables, including elevation, slope, and distances to hydrology.

For all image datasets, we construct Z by applying a Gaussian blur with standard deviation $\sigma = 3$ to the original image, with the Gaussian kernel truncated at four standard deviations on each side. The additional features X are given by the residual between the original and blurred images. An example of the features for each image dataset is shown in Figure 4.

## B.3 DEGRADED FEATURE CONSTRUCTION DETAILS

We provide additional details on the construction of the degraded feature $\tilde { X }$ used in the data processing inequality experiments (Section 3.3). We apply degradation only to the additional features $X$ , leaving the base features $Z$ unchanged. We consider three degradation levels, low, medium, and high, corresponding to degradation multipliers $\gamma \in \{ 0 . 2 5 , 0 . 5 , 1 . 0 \}$ , respectively. We report results for all three degradation levels in Appendix E. In the main text, we report results only for the medium degradation level, corresponding to $\gamma = 0 . 5$

For the synthetic and tabular datasets, we degrade X by adding independent Gaussian noise to each coordinate. For a degradation level $\gamma ,$ the noise standard deviation for each coordinate is $\gamma$ times its empirical standard deviation, estimated using the training data. The degradation noise is sampled once for each dataset and held fixed across all noise levels and subsequent experimental replicates in the analysis.

For the image datasets, we degrade X by applying an additional Gaussian blur to the residual features, while leaving the already-blurred base features $Z$ unchanged. For a degradation level $\gamma ,$ we use a blur standard deviation of $\gamma \sigma _ { : }$ , where $\sigma = 3$ is the blur scale used to construct Z. An example of the degraded features for each image dataset is shown in Figure 4.

The synthetic datasets (S1 and S2) have known oracle distributions, so we compute the degraded posterior in closed form or using Gauss–Hermite quadrature. For the real-world datasets, we train a separate model on $( \tilde { X } , Z )$ for each degradation level.

## B.4 CANDIDATE FEATURE DETAILS

We describe the candidate features $\{ X _ { 1 } , \ldots , X _ { m } \}$ used in the feature selection experiments. In the synthetic settings, the base features $Z$ are always observed, and each coordinate of the additional features X forms a separate candidate feature. This yields 10 and 50 candidate features for S2 and S3, respectively, for which we can compute oracle conditional probabilities given any subset of features. We exclude S1 from the feature selection experiments due to the low dimensionality of X.

For the tabular datasets, no features are observed initially and each input feature (including those previously in $Z )$ is treated as a separate candidate. The only exception is HAR, which contains 561 input features. To keep the number of candidates manageable, we randomly partition the features into 51 groups of 11, with each group forming a candidate feature.

The image candidates are constructed from the final convolutional representation of the frozen backbone trained on (X, Z). As with the tabular datasets, no features are observed initially. Since the purpose of the feature selection experiment is to empirically probe the differences between set-size reduction and Shannon information, this representation serves to provide a collection of meaningful candidates, rather than physically acquirable features.

On FMNIST, CIFAR-10, and CIFAR-100, we partition the channels of the final convolutional feature map into 64 contiguous groups. Each group contains 2 channels of size $3 \times 3$ on FMNIST and 4 channels of size $4 \times 4$ on CIFAR-10 and CIFAR-100. The ResNet backbone on ImageNet-1k instead produces 512 globally average-pooled channels, which we partition into 64 groups of 8.

The number of candidate features m for each dataset is summarized in Table 2.

Table 3: Selected hyperparameters and training configurations for the classification models. The learning rate, weight decay, and dropout rate are selected via grid search. The maximum number of epochs and patience correspond to the full training procedure after hyperparameter selection.
<table><tr><td>Dataset</td><td>Params.</td><td>Max. epochs</td><td>Patience</td><td>Learning rate</td><td>Weight decay</td><td>Dropout</td></tr><tr><td>Wine</td><td>600</td><td>1,000</td><td>100</td><td> $4 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td></tr><tr><td>HAR</td><td>178K</td><td>1,000</td><td>100</td><td> $4 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.1</td></tr><tr><td>Letter</td><td>106K</td><td>1,000</td><td>100</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.2</td></tr><tr><td>Covertype</td><td>424K</td><td>500</td><td>100</td><td> $4 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td>0.1</td></tr><tr><td>FMNIST</td><td>105K</td><td>1,000</td><td>100</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td><td>0.3</td></tr><tr><td>CIFAR-10</td><td>414K</td><td>500</td><td>100</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.1</td></tr><tr><td>CIFAR-100</td><td>783K</td><td>500</td><td>100</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td></tr><tr><td>ImageNet-1k</td><td>11.7M</td><td>25</td><td>5</td><td> $4 \times 1 0 ^ { - 4 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td><td>0.1</td></tr></table>

## C TRAINING DETAILS

We provide details on the model architectures, training procedures, hyperparameter selection, and final pre-calibration accuracy of our classification models. The synthetic datasets use oracle probabilities and therefore do not require model training.

## C.1 MODEL ARCHITECTURES

For the tabular datasets, we use multi-layer perceptrons (MLPs) with dropout after each hidden layer. The hidden-layer widths are (16, 16) for Wine, (256, 128) for HAR, (256, 256, 128) for Letter, and (512, 512, 256) for Covertype. All non-binary features are independently standardized to zero mean and unit variance using statistics computed from the training data.

For FMNIST, CIFAR-10, and CIFAR-100, we use three-layer CNNs with 3 × 3 convolutional kernels, batch normalization, and ReLU activations. The convolutional layers have channel widths (32, 64, 128) for FMNIST and (64, 128, 256) for CIFAR-10 and CIFAR-100. Each convolutional layer is followed by $2 \times 2$ max pooling. After the final layer, we flatten the spatial representation, apply dropout, and use a linear classification head to produce the predicted probabilities.

For ImageNet-1k, we use the ResNet-18 architecture. We directly fine-tune the pretrained network on the base features Z using standard ImageNet normalization and modify the input layer to accommodate the additional channels when using both X and Z. For all ResNet models, the fina convolutional representation is passed through global adaptive pooling and dropout, followed by a linear classification head.

## C.2 TRAINING OPTIMIZATION

We train all models on the real datasets using the Adam optimizer with weight decay to minimize the standard cross-entropy loss. All newly initialized model parameters use the default PyTorch initialization. For the image datasets, we additionally apply data augmentation consisting of horizontal flips with probability 0.5 and random crops after zero-padding the images by 2 pixels for FMNIST and 4 pixels for CIFAR-10, CIFAR-100, and ImageNet-1k. We also use cosine annealing to decay the learning rate from its initial value to zero over the maximum number of training epochs.

Training uses a batch size of 256 and early stopping based on validation loss, with the checkpoint attaining the lowest validation loss restored at the end of training. Separate models are trained for the base, degraded, and augmented feature sets. We set the maximum number of training epochs to 1,000 for Wine, HAR, Letter, and FMNIST; 500 for CIFAR-10, CIFAR-100, and Covertype; and 25 for ImageNet-1k. We use an early-stopping patience of 100 epochs for all datasets except ImageNet-1k, for which we use a patience of 5 epochs.

Table 4: Top-1 classification accuracy (%) on the held-out evaluation pool for the augmented, degraded, and base feature sets, and for the mask-conditioned model with all candidate features observed. For the synthetic datasets S1 and S2, we report the accuracy of the oracle model.
<table><tr><td>Dataset</td><td> $( X , Z )$ </td><td> $( { \tilde { X } } , Z )$  low noise</td><td> $( { \tilde { X } } , Z )$  medium noise</td><td> $( { \tilde { X } } , Z )$  high noise</td><td> $Z$ </td><td>Mask-conditioned (all)</td></tr><tr><td>S1</td><td>56.52</td><td>54.86</td><td>51.83</td><td>46.69</td><td>39.33</td><td>一</td></tr><tr><td>S2</td><td>50.83</td><td>48.84</td><td>44.00</td><td>34.21</td><td>18.43</td><td></td></tr><tr><td>Wine</td><td>55.15</td><td>54.98</td><td>53.43</td><td>52.78</td><td>50.33</td><td>55.80</td></tr><tr><td>HAR</td><td>98.29</td><td>97.40</td><td>96.54</td><td>93.86</td><td>88.08</td><td>98.72</td></tr><tr><td>Letter</td><td>97.68</td><td>95.52</td><td>92.00</td><td>86.76</td><td>86.52</td><td>97.36</td></tr><tr><td>Covertype</td><td>96.24</td><td>83.11</td><td>74.93</td><td>68.85</td><td>65.16</td><td>94.12</td></tr><tr><td>FMNIST</td><td>93.06</td><td>92.77</td><td>91.43</td><td>89.92</td><td>89.85</td><td>93.11</td></tr><tr><td>CIFAR-10</td><td>86.40</td><td>84.10</td><td>77.55</td><td>70.31</td><td>68.80</td><td>86.30</td></tr><tr><td>CIFAR-100</td><td>60.97</td><td>56.77</td><td>50.14</td><td>42.94</td><td>40.36</td><td>56.79</td></tr><tr><td>ImageNet-1k</td><td>58.79</td><td>57.84</td><td>54.57</td><td>47.06</td><td>43.70</td><td>57.99</td></tr></table>

## C.3 MASK-CONDITIONED MODEL TRAINING

The feature selection experiments require predicted conditional probabilities given arbitrary subsets of the candidate features. We therefore modify the model architectures to accommodate arbitrary conditioning masks. For the tabular datasets, we modify the MLP to take inputs of the form $[ x \odot M , M ]$ , where x is the full feature vector and M is a binary mask indicating which features are observed, following the masked predictor construction in Yoon et al. (2019).

For the image datasets, we use the pretrained CNN or ResNet model previously trained on the full set of features $( X , Z )$ We freeze the backbone through the final convolutional layer, fixing the representation from which the candidate features are constructed. For FMNIST, CIFAR-10, and CIFAR-100, we train a masked MLP on this representation using the same masking construction as for the tabular datasets. We instead train a linear classification head on the masked representation on ImageNet-1k for scalability.

During training, we independently sample a conditioning mask for each example. Specifically, we first sample u ∼ Unif[0, 1] and then independently mask each candidate feature with probability u. To ensure sufficient exposure to the two extreme cases, we additionally mask all candidate features with probability 0.05 and observe all candidate features with probability 0.15.

We double both the maximum number of training epochs and the patience relative to the full-feature training procedure described in Appendix C.2. For early stopping, we average the validation loss across masks generated with fixed masking probabilities $u \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$

## C.4 HYPERPARAMETER SELECTION

We select the learning rate, weight decay, and dropout rate via grid search. We consider learning rates in $( 1 0 ^ { - 3 } , 4 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ , weight decay values in $( 1 0 ^ { - 4 } , \overset { \smile } { 4 } \times 1 0 ^ { - 5 } , 1 0 ^ { - 5 } )$ , and dropout rates in $( 0 . 1 , \dot { 0 } . 2 , 0 . 3 )$ , yielding 27 configurations in total. Hyperparameters are selected using the augmented model with both X and Z. The selected configuration is then shared across the augmented model, the base model using only Z, and the models using degraded features X<sup>˜</sup> and Z.

For each configuration, we train the augmented model using 20% of both the maximum epoch budget and early-stopping patience specified above. The image models use the same cosine annealing schedule during hyperparameter selection. For ImageNet-1k, each configuration is trained for the reduced epoch budget without early stopping.

Among the 27 configurations, we select the one achieving the highest validation accuracy after early stopping. We then train all models using this configuration with their respective epoch budget and early-stopping patience. The selected hyperparameters for each dataset are summarized in Table 3.

![](images/84780e734887a1fd4b07985a386e0dd5c2919ae785c00f4a28facc6d7f2a3b5f.jpg)  
Figure 5: Empirical marginal coverage of the conformal predictors using (X, Z) over 100 random equal calibration and test splits of the evaluation pool. The dashed line indicates the nominal coverage level $1 - \alpha = 0 . 9$

## C.5 ACCURACY AND COVERAGE METRICS

We report the top-1 classification accuracy of all models on the held-out evaluation pool. We evaluate models using the augmented features (X, Z), the degraded features (X, Z<sup>˜</sup> ) at each noise level, the base features Z, as well as the mask-conditioned model with all candidate features observed. The results are summarized in Table 4.

As expected, classification accuracy generally decreases as features are degraded or removed. The mask-conditioned models also obtain comparable performance to models trained on the full feature set (X, Z), despite supporting predictions given arbitrary subsets of features. The models using the augmented features do not achieve state-of-the-art performance on these classification tasks, since we use simplified architectures and training procedures. Our primary goal is not to maximize predictive accuracy, but to evaluate our theoretical results under realistic levels of model imperfection.

We additionally evaluate the marginal coverage of the conformal predictors using (X, Z) at the nominal coverage level $1 - \alpha = 0 . 9$ , which is used throughout most of our experimental results. For each dataset, we randomly split the evaluation pool in half into calibration and test sets over 100 replicates. As shown in Figure 5, both the probability and APS scores achieve empirical coverage close to the nominal level on all datasets, with larger variability on the datasets with smaller evaluation pools, such as Wine and HAR.

## D ANALYSIS DETAILS

We provide additional details on the estimation of the bounds, the evaluation protocols, and the analysis metrics used in our validation and feature selection experiments.

## D.1 ESTIMATION OF THE BOUNDS

To evaluate the conformal mutual information terms in Theorem 2, we estimate $I _ { \lambda } ( X ; Y \mid Z )$ with the plug-in (or oracle) probability distributions on the test set at the calibrated $\ddot { \lambda } _ { \alpha } ^ { ( i ) }$ . For Theorem $^ { 4 , }$ we compute $I ^ { ( 1 ) }$ in the same way, since $\hat { \Lambda } _ { \alpha } ^ { \scriptscriptstyle ( 1 ) }$ depends only on $Z .$ However, the term $I ^ { ( 2 ) }$ in the lower bound cannot be computed using a plug-in estimate. Recall that

$$
I ^ { ( 2 ) } = H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid Z ) - H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid X , Z ) .\tag{143}
$$

The second term can be evaluated separately for each test point, but the first term is problematic because $\hat { \Lambda } _ { \alpha } ^ { ( 2 ) }$ depends on both X and $\bar { Z }$ . In particular,

$$
H _ { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } ( Y \mid Z ) = \mathbb { E } _ { Z } \left[ \sum _ { y } \operatorname* { m i n } \Bigl \{ 1 , \mathbb { E } \left[ \frac { p ( y | X , Z ) } { \hat { \Lambda } _ { \alpha } ^ { ( 2 ) } } \ \Big \vert \ Z \right] \Bigr \} \right] .\tag{144}
$$

Since the conditional expectation over $X \mid Z$ lies inside the minimum, computing it requires multiple draws of X for each $Z .$ . In the synthetic settings, $p ( x \mid z )$ is known, so we estimate it by Monte Carlo sampling with $M = 2 5 6$ draws of $X \mid Z$ for each test point. For the real-world datasets, we instead average $q ( y \mid x _ { j } , z _ { j } ) / \hat { \Lambda } _ { \alpha } ^ { { \scriptscriptstyle ( 2 ) } } ( x _ { j } , z _ { j } )$ over the $k = 1 0$ nearest other neighbors $j$ of each test point in the evaluation pool. Neighbors are computed using Euclidean distance on the base features $Z$ for tabular data and cosine distance on the learned embedding of $Z$ for image data.

In the synthetic settings, we use the oracle conditional distributions, so $D ^ { ( i ) } = 0$ exactly. For the real-world datasets, estimating $D ^ { ( i ) }$ requires the true conditional distribution, so we set $D ^ { ( i ) } = 0 .$ which is consistent with the plug-in estimates of $I ^ { ( i ) }$ . We emphasize that the resulting bounds for the real-world datasets are therefore approximate. These approximations do not undermine our validation, since the synthetic settings evaluate the bounds using the true conditional distributions, while the real-world datasets complement these results under realistic model imperfections.

The terms involving miscoverage rates are estimated by the test-set average of

$$
\frac { \mathbf { 1 } [ Y \notin \mathcal { C } _ { \alpha } ( X ^ { ( 2 ) } ) ] - \mathbf { 1 } [ Y \notin \mathcal { C } _ { \alpha } ( X ^ { ( 1 ) } ) ] } { \hat { \lambda } _ { \alpha } ^ { ( i ) } } .\tag{145}
$$

For the APS bounds, we analogously replace $\hat { \lambda } _ { \alpha } ^ { ( i ) }$ with $\hat { \Lambda } _ { \alpha } ^ { \left( i \right) }$ . Since the random threshold is a function of the input, this estimator is unbiased for the corresponding calibration terms in the bounds, conditional on the calibration set.

## D.2 EVALUATION PROTOCOLS

As described in Section 5, we reserve a dedicated pool of $n _ { \mathrm { e v a l } }$ samples from each synthetic and real-world dataset for calibration and testing. We next describe how this evaluation pool is used in each experiment.

## D.2.1 VALIDATION EXPERIMENTS

We first describe the evaluation protocol for the validation experiments. For each experimental replicate, we sample a calibration dataset of size $n _ { \mathrm { c a l } }$ and a test dataset of size $n _ { \mathrm { t e s t } }$ without replacement from the evaluation pool. The calibration and test datasets are disjoint but do not necessarily partition the entire pool. For a given random seed, the sampled datasets are nested across sample sizes, while different seeds yield independent draws.

For each metric, we consider three regimes: (i) jointly varying $n _ { \mathrm { c a l } } = n _ { \mathrm { t e s t } }$ , (ii) varying $n _ { \mathrm { c a l } }$ with $n _ { \mathrm { t e s t } }$ fixed, and (iii) varying $n _ { \mathrm { t e s t } }$ with $n _ { \mathrm { c a l } }$ fixed. For the normalized margin and violation rate, we use a logarithmically spaced grid of five to seven points ranging from 100 to min $( 1 0 , 0 0 0 , n _ { \mathrm { e v a l } } / 2 )$ inclusive. Because the bound width becomes visually uninformative at larger sample sizes, we instead use a logarithmically spaced grid from 10 to min $( 1 , 0 0 0 , n _ { \mathrm { e v a l } } / 2 )$ . In regimes (ii) and (iii), the fixed sample size is set to the maximum of this grid.

Calibration follows the standard split conformal prediction procedure using either the probability score or APS score. For APS, we use the same score randomization draw $\bar { U } \sim \mathrm { U n i f } [ 0 , \bar { 1 } ]$ between the procedures using $X ^ { ( 1 ) }$ and $X ^ { ( 2 ) }$ . Both procedures are calibrated on the same calibration set and evaluated on the same test set. We estimate $\Delta _ { \alpha }$ by the difference in average prediction set size on the test set under $X ^ { ( 1 ) }$ and $X ^ { ( 2 ) }$

We additionally vary the nominal coverage level over $1 - \alpha \in \{ 0 . 8 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 0 . 9 9 \}$ . For these experiments, $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ are held fixed at the respective maximum sample sizes used for each metric, $\mathrm { i . e . , }$ min $\left( 1 0 , 0 0 0 , n _ { \mathrm { e v a l } } / 2 \right)$ for the normalized margin and violation rate and min $( 1 , 0 0 0 , n _ { \mathrm { e v a l } } / 2 )$ for the bound width.

For the DPI validation experiments, we follow a similar calibration and evaluation procedure by sampling from the same evaluation pool. We fix the test set size at $n _ { \mathrm { t e s t } } =$ min $( 1 0 , 0 0 0 , n _ { \mathrm { e v a l } } / 2 )$ and vary $n _ { \mathrm { c a l } } \in \{ 1 0 , 1 0 0 , 1 , 0 0 0 \}$ , omitting values of $n _ { \mathrm { c a l } }$ that exceed $n _ { \mathrm { e v a l } } / 2$ . Within each replicate, the procedures using the augmented and degraded features share the same calibration and test sets. We intentionally consider small calibration sample sizes to illustrate how the normalized DPI gap concentrates as $n _ { \mathrm { c a l } }$ increases and to demonstrate that finite-sample calibration variability can result in a negative gap at sufficiently small $n _ { \mathrm { c a l } }$

## D.2.2 FEATURE SELECTION EXPERIMENTS

We next describe the evaluation protocol for the feature selection experiments. For each of 10 random seeds, we randomly partition the evaluation pool into three disjoint subsets, using 50% of the samples as a selection set, 25% as a calibration set, and the remaining 25% as a test set. The selection set is used to compute the criteria and determine the feature ordering, while the calibration and test sets are used only to evaluate the resulting sequence. In the synthetic settings, the base features $Z$ are always observed, whereas no features are known initially for the real-world datasets.

Note that this setting contrasts with the active feature acquisition (AFA) literature, which typically selects features to acquire for each individual instance and trains dedicated acquisition policies to optimize predictive performance under a feature budget (Shim et al., 2018). We use a different evaluation protocol because our goal is not to propose a competitive AFA method. Instead, we use a simple protocol to empirically probe how the criteria from our theory rank features in practice.

Let S denote the set of candidate features, initialized as $S = \emptyset$ . At each step, we evaluate a selection criterion for every remaining candidate $X _ { j }$ using the predicted probabilities given $X _ { S \cup \{ j \} }$ , add the highest-scoring candidate to $S _ { \ i }$ , and repeat until all candidates are exhausted. This yields a single global feature ordering for each criterion and seed, which is shared across all test samples. For each prefix of the ordering, we then calibrate the conformal predictor on the calibration set at nominal coverage $1 - \alpha = 0 . 9$ and evaluate set size, coverage, and accuracy on the test set.

We compare the following selection criteria, each computed using the predicted probabilities of the oracle or mask-conditioned model as a plug-in estimate.

$\Delta _ { \alpha }$ (Prob.): the reduction in the average set size using probability score with $X _ { S \cup \{ j \} }$

$\Delta _ { \alpha }$ (APS): the reduction in the average set size using APS score with $X _ { S \cup \{ j \} }$

$I _ { \lambda }$ (Prob.): the conformal mutual information $I _ { \lambda } ( X _ { j } ; Y \mid X _ { S } )$ evaluated at the geometric mean $\lambda = ( \hat { \lambda } _ { \alpha } ^ { ( 1 ) } \hat { \lambda } _ { \alpha } ^ { ( 2 ) } ) ^ { 1 / 2 }$ of the calibrated thresholds using $X _ { S }$ and $X _ { S \cup \{ j \} } ;$

• Shannon MI: the plug-in estimate $\hat { I } ( X _ { j } ; Y \mid X _ { S } ) = \hat { H } ( Y \mid X _ { S } ) - \hat { H } ( Y \mid X _ { S \cup \{ j \} } ) ;$

• Accuracy: the empirical top-1 accuracy gain with $X _ { S \cup \{ j \} } ;$

• Random: a uniformly random ordering of the candidate features.

Since the conformal-based criteria require separate calibration data, we use 5-fold cross-fitting within the selection set. We calibrate the thresholds on four folds and evaluate the criterion on the held-out fold, then average the resulting values across folds. The Shannon mutual information and accuracy criteria do not require calibration and are computed directly on the full selection set.

We do not include an analogous $I _ { \Lambda }$ criterion for the $\mathsf { A P S }$ score since it requires the same k-NN approximation of the conditional expectation over $X _ { j } ~ \vert ~ X _ { S }$ as described above for $I ^ { ( 2 ) }$ . Repeating this approximation for every candidate at every step is computationally expensive and introduces additional approximation error, so we omit this criterion.

## D.3 METRICS

We next formalize the evaluation metrics in further detail.

## D.3.1 VALIDATION EXPERIMENTS

In the validation experiments, we consider four main metrics: normalized margin, violation rate, normalized width, and normalized DPI gap. Except for the violation rate, we normalize all metrics by the number of classes K in order to compare across datasets. We next define each metric formally.

Let $[ L , U ]$ denote the estimated bounds from Theorem 2 for the probability score or Theorem 4 for $\mathsf { A P S }$ . When computing the normalized margin, violation rate, and normalized width, we discard replicates with invalid bounds $( U < L )$ , which occur in roughly $3 \%$ of cases, and average over the remaining random splits of the data. This exclusion does not change our conclusions.

The normalized margin quantifies the magnitude by which the empirical $\Delta _ { \alpha }$ falls outside the theoretical bounds for a particular realization of the calibration and test data. Although the theoretical bounds hold exactly for the population quantities under the assumptions of our results, finite-sample Monte Carlo estimation and model approximation on the real-world datasets can lead to empirical violations. We define

$$
\mathrm { N o r m a l i z e d \ : M a r g i n } = \frac { d ( \Delta _ { \alpha } , [ L , U ] ) } { K } = \frac { \operatorname* { m a x } \{ L - \Delta _ { \alpha } , \Delta _ { \alpha } - U , 0 \} } { K } ,\tag{146}
$$

where $d ( x , \mathcal { T } )$ denotes the distance from x to the interval $\mathcal { T } ,$ with $d ( x , { \mathcal { T } } ) = 0$ whenever $x \in \mathbb { Z } .$

The violation rate examines how frequently the empirical $\Delta _ { \alpha }$ falls outside the theoretical bounds, providing a binary measure of empirical validity that does not account for the magnitude of a violation. We define

$$
\operatorname { V i o l a t i o n } \operatorname { R a t e } = \mathbb { P } \big ( \Delta _ { \alpha } \notin [ L , U ] \big ) ,\tag{147}
$$

where the probability is estimated over random replicates of the calibration and test data. In practice, we expand the lower and upper bounds by a small slack of $1 0 ^ { - 4 }$ to account for floating-point error when computing this metric.

The normalized width measures the informativeness of the bounds through the size of the interval $[ L , U ]$ . We define

$$
\mathrm { N o r m a l i z e d ~ W i d t h } = { \frac { U - L } { K } } .\tag{148}
$$

Since we exclude replicates with $U < L$ , the normalized width is non-negative for every replicate.

Finally, the normalized DPI gap compares the conformal set-size reduction obtained by augmenting the base features $Z$ with either X or its degraded counterpart X<sup>˜</sup> . For each replicate, we use identical test sets to estimate the expected set-size reductions $\Delta _ { \alpha }$ and $\bar { \Delta } _ { c }$ and define

$$
\mathrm { N o r m a l i z e d D P I G a p } = \frac { \Delta _ { \alpha } - \tilde { \Delta } _ { \alpha } } { K } .\tag{149}
$$

We examine the distribution of this gap across random draws of the calibration and test data.

## D.3.2 FEATURE SELECTION EXPERIMENTS

In the feature selection experiments, we evaluate each criterion by the quality of the conformal predictors obtained along its selection trajectory. Let $S _ { j }$ denote the first $j$ candidates acquired under a given criterion. For each j, we compute the average prediction set size and empirical coverage on the test set using $X _ { S _ { i } }$ for both the probability score and the APS score. We additionally report the top-1 and top-5 classification accuracy of the model given $X _ { S _ { j } }$ on the test set.

Let $\mathcal { E } _ { j }$ denote any one of these metrics evaluated using the first $j$ features. To summarize performance across the entire selection trajectory, we report the average

$$
\bar { \mathcal { E } } = \frac { 1 } { m - 1 } \sum _ { j = 1 } ^ { m - 1 } \mathcal { E } _ { j } ,\tag{150}
$$

where we exclude the cases $j = 0$ and $j = m$ since all criteria observe identical features at the endpoints. Intuitively, E<sup>¯</sup> approximates the area under the curve of each metric normalized by m. This summarizes the performance across all steps rather than at a single fixed budget.

## D.4 IMPLEMENTATION DETAILS

The validation experiments require evaluating a large number of calibration and test draws across sample sizes, random seeds, nominal coverage levels, feature sets, and conformity scores. To reduce the computational cost, we presort the predicted class probabilities for the evaluation pool and use binary search to efficiently compute the quantities required for each replicate. This optimization is an exact reformulation of the standard computation and does not affect any reported results.

For an evaluation pool of N samples and K classes, presorting requires O(NK log K) time. Each subsequent replicate requires $O ( n _ { \mathrm { c a l } } \log n _ { \mathrm { c a l } } + n _ { \mathrm { t e s t } } \log K )$ time. This provides a substantial computational speedup over directly recomputing the conformal prediction sets, particularly for datasets with large K such as ImageNet-1k.

The presorting approach does not apply to the feature selection experiments, since the evaluated subsets depend on the selection criterion and the number of possible subsets grows exponentially. Instead, we naively compute the predicted probabilities for each candidate subset at each step.

## E ADDITIONAL EXPERIMENTS

In this section, we further investigate the empirical behavior of our sandwich bounds for set-size reduction using the probability and APS scores. We then provide additional results for the DPI under different noise levels and report feature selection results across the remaining datasets. Details of the evaluation procedures and metrics are described in Appendix D.

## E.1 SANDWICH VALIDATION RESULTS

We provide additional results demonstrating the empirical behavior of Theorems 2 and 4. We first investigate the violation rate of the bounds as we jointly vary $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ . We show these results in Figure 6, together with the normalized margin and width results from Figure 3 for comparison.

Consistent with the normalized margin, the violation rate is nonzero at smaller calibration and test set sizes but decreases toward zero as the sample size increases. The main outlier is the Wine dataset using the probability score, for which the violation rate remains high over the evaluated range. This behavior largely reflects the binary nature of the violation rate. The corresponding normalized margin behaves similarly to those of the other datasets, indicating that the violations themselves are small in magnitude.

Comparing the probability and APS scores, we find that the violation rate is typically lower for APS across datasets and sample sizes. The lower violation rate for APS is consistent with its wider bounds, which are easier to satisfy and less sensitive to finite-sample estimation error.

We next examine the normalized margin, violation rate, and normalized width as we independently and jointly vary $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ in Figures 7–9. Additionally, we investigate the three metrics across different nominal coverage levels in Figure 10.

Normalized margin (Figure 7). The magnitude of the empirical violations is driven primarily by the size of the test set. $\mathbf { A s } n _ { \mathrm { t e s t } }$ increases with $n _ { \mathrm { c a l } }$ fixed, the normalized margin decreases toward zero. In contrast, with $n _ { \mathrm { t e s t } }$ fixed, the margin is largely insensitive to $n _ { \mathrm { c a l } }$ . This suggests that the observed violations primarily arise from finite test-set estimation error.

![](images/d546ea2495b3a5ba7aefd8aac9a321506aca86b6a79813f9d2e6c726651b42a7.jpg)  
Figure 6: We show the normalized margin and violation rate as we jointly vary $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ (left and middle) and the normalized width as we vary $n _ { \mathrm { c a l } }$ while keeping $n _ { \mathrm { t e s t } }$ fixed (right). All panels report the mean ± standard error based on 100 random draws of the calibration and test sets from the fixed evaluation pool.

![](images/076db1a56478f6c594893198ffe1b2b08d0590b57fc62a7bc8db5fe27fcdc9f0.jpg)  
Figure 7: We show the normalized margin as we vary $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ jointly (left) or independently (middle and right). All panels report the mean ± standard error based on 100 random draws of the calibration and test sets from the fixed evaluation pool.

![](images/45864f2cbe9c42a8ffc3b846a56a813d14d38230c6f6b96f629db4a132ed0ca3.jpg)  
Figure 8: We show the violation rate as we vary $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ jointly (left) or independently (middle and right). All panels report the mean ± standard error based on 100 random draws of the calibration and test sets from the fixed evaluation pool.

Violation rate (Figure 8). The violation rate exhibits a similar dependence on $n _ { \mathrm { t e s t } }$ . When $n _ { \mathrm { t e s t } }$ is fixed, the violation rate for APS is low and largely insensitive to $n _ { \mathrm { c a l } }$ , whereas for the probability score it decreases toward zero as $n _ { \mathrm { c a l } }$ increases. The Wine dataset is the sole exception, where the violation rate remains between roughly 60% and 80%.

Normalized width (Figure ${ \bf 9 ) . }$ The normalized width mostly depends on the size of the calibration set and decreases quickly with $n _ { \mathrm { c a l } }$ . This behavior agrees with our theoretical analysis of the finitesample calibration terms. As the calibration size increases, the training-conditional miscoverage rates concentrate around the nominal coverage level, causing the finite-sample calibration terms to shrink and the bounds to tighten.

![](images/966a1211a8bd60f6abf69afda9a143f7a50864c7409779ecb255b5123493e95a.jpg)  
Figure 9: We show the normalized width as we vary $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ jointly (left) or independently (middle and right). All panels report the mean ± standard error based on 100 random draws of the calibration and test sets from the fixed evaluation pool.

![](images/2408c72384c9cea115f658391581413a1238ce02ad426f78060efb920cc8d5f8.jpg)  
Figure 10: We show the normalized margin, violation rate, and normalized width as we vary the nominal coverage level $1 - \alpha \in \{ 0 . 8 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 0 . 9 9 \}$ , with $n _ { \mathrm { c a l } }$ and $n _ { \mathrm { t e s t } }$ fixed at their largest values for each dataset. All panels report the mean ± standard error based on 100 random draws of the calibration and test sets from the fixed evaluation pool.

Nominal coverage levels (Figure 10). The bounds remain well-behaved across nominal coverage levels $1 - \alpha \leq 0 . 9 5$ for most datasets (with exceptions for Wine, FMNIST, and CIFAR-10). However, at $1 - \alpha = 0 . 9 9$ , all three metrics increase sharply for most datasets. We attribute this behavior to sensitivity to the calibrated thresholds $\hat { \lambda } _ { \alpha } ^ { \left( i \right) }$ and $\bar { \Lambda } _ { \alpha } ^ { ( i ) }$ . As the nominal coverage level increases, these thresholds become smaller, which amplifies the finite-sample calibration terms and consequently increases the width of the bounds. The smaller thresholds also increase the variance of the corresponding empirical estimates, leading to larger normalized margins and violation rates at fixed calibration and test sample sizes.

![](images/d62c9fa3a390098776deab7335589cca0b8e35cc8b3c48011d2ed666be393ac8.jpg)  
Figure 11: We show the normalized DPI gap across calibration sample sizes and degradation levels, using boxplots to summarize 100 random draws of the calibration and test sets. Each column corre sponds to a degradation level, and the marker for each dataset is indicated above the corresponding boxplots. For visual clarity, we magnify the y-axis by $2 \times$ and $4 \times$ for $n _ { \mathrm { c a l } } = 1 0 0$ and $n _ { \mathrm { c a l } } = 1 { , } 0 0 0$ respectively. Datasets whose evaluation pool is too small for a given calibration size are omitted.

## E.2 DPI VALIDATION RESULTS

We show the distribution of the normalized DPI gap across calibration sample sizes and degradation levels in Figure 11. The results for the medium degradation level $( \gamma = 0 . 5 )$ correspond to those reported in Section 5.

Across all degradation levels, the realized gap can be negative for particular draws of the calibration and test data. As the calibration sample size increases, however, the variability of the gap decreases and its distribution generally concentrates around a non-negative mean. Increasing the degradation level also generally increases the average gap as expected, since stronger noise removes more of the information present in X. These trends hold for both probability score and APS score.

These results further support the interpretation of conformal set-size reduction as a measure of information gain. Theorems 3 and 5 establish an approximate data processing inequality, with finitesample calibration and model-error terms that allow the DPI gap to be negative. Empirically, the gap is generally non-negative on average, providing evidence of this data processing behavior in practice. At the same time, the negative gaps observed for individual replicates demonstrate that the deviations permitted by our theoretical results do indeed occur in practice.

## E.3 FEATURE SELECTION RESULTS

We report feature selection results for all datasets in Table 5, excluding synthetic setting S1 since it has too few candidate features to meaningfully distinguish the criteria.

Across most settings, selection based on set-size reduction yields the smallest prediction sets while typically achieving the best top-5 accuracy. The main exceptions occur for the APS score on Wine and FMNIST. The two $\Delta _ { \alpha }$ criteria perform similarly to each other but do not always yield the smallest sets under their own score. Since these differences are often within one standard error, they likely reflect estimation noise in the selection set.

We also evaluate selection using $I _ { \lambda } .$ , which performs similarly to $\Delta _ { \alpha }$ under the probability score in the synthetic settings but often yields larger sets on the real-world datasets. We attribute this gap to model imperfection. Theorem 2 brackets $\Delta _ { \alpha }$ by conformal mutual information only up to the model error terms $D ^ { ( i ) }$ , so the plug-in $I _ { \lambda }$ at the geometric mean of the calibrated thresholds becomes a less precise proxy for $\Delta _ { \alpha }$ when these terms are nonzero.

Table 5: We evaluate feature selection performance using conformal set-size reduction. Each row represents one selection criterion and values are reported as mean ± standard error over 10 random splits of the evaluation pool. Arrows indicate the direction of improvement, and the best-performing method for each metric is bolded. Empirical coverage is reported for reference without bolding to indicate that differences in set size are not due to undercoverage.
<table><tr><td rowspan=1 colspan=9>Prob.                         APS                         Acc.Dataset   CriterionE[|C|] (↓)        Cov.        E[|C|] (↓)        Cov.       Top-1 (↑)       Top-5 (↑)</td></tr><tr><td rowspan=1 colspan=9>∆α (Prob.)   11.464 ± 0.032  0.900 ± 0.001  11.952 ± 0.029  0.900 ± 0.001  0.366 ± 0.001  0.722 ± 0.000</td></tr><tr><td rowspan=1 colspan=7>∆α (APS)   11.459 ± 0.033   $0 . 9 0 0 \pm 0 . 0 0 1$    $1 1 . 9 5 0 \pm 0 . 0 2 9$   $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=2>0.366 ± 0.001  0.722 ± 0.001</td></tr><tr><td rowspan=2 colspan=7>Iλ (Prob.)  11.456 ± 0.032   $0 . 9 0 0 \pm 0 . 0 0 1$   ${ \bf 1 1 . 9 4 8 \pm 0 . 0 2 8 }$  0.900 ± 0.001S   Shannon MI  11.473 ± 0.031   $0 . 9 0 0 \pm 0 . 0 0 1$    $1 1 . 9 5 9 \pm 0 . 0 2 7$   $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=2> $0 . 3 6 6 \pm 0 . 0 0 1$   0.722 ± 0.001</td></tr><tr><td rowspan=1 colspan=2> $\mathbf { 0 . 3 6 7 \pm 0 . 0 0 1 }$   0.722 ± 0.001</td></tr><tr><td rowspan=1 colspan=7>Accuracy   11.503 ± 0.029   $0 . 9 0 0 \pm 0 . 0 0 1$    $1 1 . 9 8 6 \pm 0 . 0 2 6$    $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=2> $\mathbf { 0 . 3 6 7 \pm 0 . 0 0 1 }$   0.722 ± 0.001</td></tr><tr><td rowspan=1 colspan=9>Random    12.207 ± 0.085   $0 . 9 0 0 \pm 0 . 0 0 1$    $1 2 . 6 7 0 \pm 0 . 0 8 5$    $0 . 9 0 0 \pm 0 . 0 0 1$    $0 . 3 5 4 \pm 0 . 0 0 2$   0.707 ± 0.002</td></tr><tr><td rowspan=1 colspan=7> $\Delta _ { \alpha }$ (Prob.)   2.239 ± 0.002   $0 . 9 0 0 \pm 0 . 0 0 0$    $2 . 6 2 9 \pm 0 . 0 0 2$   0.900 ± 0.000</td><td rowspan=1 colspan=2> $0 . 7 5 0 \pm 0 . 0 0 1$   0.952 ± 0.000</td></tr><tr><td rowspan=1 colspan=7>∆α (APS)   2.243 ± 0.002   $0 . 9 0 0 \pm 0 . 0 0 0$    $\mathbf { 2 . 6 1 1 \pm 0 . 0 0 2 }$    $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=2> $0 . 7 5 2 \pm 0 . 0 0 0$    $0 . 9 5 2 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan=2 colspan=7>Iλ (Prob.)   2.237 ± 0.003   $0 . 9 0 0 \pm 0 . 0 0 0$    $2 . 6 2 6 \pm 0 . 0 0 2$    $0 . 9 0 0 \pm 0 . 0 0 0$ SShannon MI  2.447 ± 0.005    $0 . 9 0 0 \pm 0 . 0 0 0$    $2 . 8 7 6 \pm 0 . 0 0 6$    $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=2> $0 . 7 5 3 \pm 0 . 0 0 0$    $\mathbf { 0 . 9 5 3 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=2> $0 . 7 6 7 \pm 0 . 0 0 0$    $0 . 9 5 2 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan=1 colspan=7>Accuracy    $3 . 6 1 3 \pm 0 . 0 2 0$    $0 . 9 0 0 \pm 0 . 0 0 0$    $4 . 4 8 7 \pm 0 . 0 2 9$    $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=2> $\mathbf { 0 . 7 6 8 \pm 0 . 0 0 1 }$    $0 . 9 1 4 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan=1 colspan=7>Random     $3 . 0 4 6 \pm 0 . 1 0 0$    $0 . 9 0 0 \pm 0 . 0 0 0$    $3 . 5 9 1 \pm 0 . 1 1 6$    $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=2> $0 . 6 7 2 \pm 0 . 0 0 9$    $0 . 9 3 1 \pm 0 . 0 0 4$ </td></tr><tr><td rowspan=1 colspan=7> $\Delta _ { \alpha }$ (Prob.)   2.265 ± 0.034  0.904 ± 0.008   2.387 ± 0.029   0.905 ± 0.006</td><td rowspan=1 colspan=2>0.553 ± 0.006 0.996 ± 0.001</td></tr><tr><td rowspan=1 colspan=4>∆α (APS)   2.279 ± 0.042   $0 . 9 0 2 \pm 0 . 0 0 8$ </td><td rowspan=1 colspan=3>2.378 ± 0.036   $0 . 9 0 4 \pm 0 . 0 0 7$ </td><td rowspan=1 colspan=2> $0 . 5 4 8 \pm 0 . 0 0 5$   0.995 ± 0.001</td></tr><tr><td rowspan=1 colspan=3>Iλ (Prob.)   2.334 ± 0.050</td><td rowspan=1 colspan=1> $0 . 9 0 3 \pm 0 . 0 1 0$ </td><td rowspan=1 colspan=1> $2 . 4 1 7 \pm 0 . 0 4 0$ </td><td rowspan=1 colspan=2>0.904 ± 0.009</td><td rowspan=1 colspan=2> $0 . 5 2 9 \pm 0 . 0 0 5$   0.995 ± 0.001</td></tr><tr><td rowspan=1 colspan=3>Shannon MI   2.288 ± 0.042</td><td rowspan=1 colspan=1> $0 . 9 0 0 \pm 0 . 0 0 9$ </td><td rowspan=1 colspan=1> $2 . 3 8 3 \pm 0 . 0 3 6$ </td><td rowspan=1 colspan=2> $0 . 9 0 3 \pm 0 . 0 0 8$ </td><td rowspan=1 colspan=1> $0 . 5 5 0 \pm 0 . 0 0 5$ </td><td rowspan=1 colspan=1> $0 . 9 9 5 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan=1 colspan=3>Accuracy    2.284 ± 0.042</td><td rowspan=1 colspan=1> $0 . 9 0 4 \pm 0 . 0 0 9$ </td><td rowspan=1 colspan=1> $\mathbf { 2 . 3 7 1 \pm 0 . 0 3 2 }$ </td><td rowspan=1 colspan=2> $0 . 9 0 4 \pm 0 . 0 0 7$ </td><td rowspan=1 colspan=1> $0 . 5 5 2 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 9 6 \pm 0 . 0 0 1 }$ </td></tr><tr><td rowspan=1 colspan=3>Random     $2 . 4 2 0 \pm 0 . 0 4 6$ </td><td rowspan=1 colspan=1>0.906 ± 0.007</td><td rowspan=1 colspan=1>2.506 ± 0.046</td><td rowspan=1 colspan=2>0.902 ± 0.007</td><td rowspan=1 colspan=1>0.507 ± 0.005</td><td rowspan=1 colspan=1>0.995 ± 0.001</td></tr><tr><td rowspan=1 colspan=4> $\Delta _ { \alpha } \ ( { \mathrm { P r o b . } } )$    0.912 ± 0.005   0.899 ± 0.005</td><td rowspan=1 colspan=1>0.963 ± 0.002</td><td rowspan=1 colspan=2>0.900 ± 0.002</td><td rowspan=1 colspan=1>0.971 ± 0.001</td><td rowspan=1 colspan=1>1.000 ± 0.000</td></tr><tr><td rowspan=1 colspan=3>∆α (APS)   0.909 ± 0.006</td><td rowspan=1 colspan=1>0.896 ± 0.006</td><td rowspan=1 colspan=1>0.960 ± 0.002</td><td rowspan=1 colspan=2>0.899 ± 0.001</td><td rowspan=1 colspan=2>0.970 ± 0.001  1.000 ± 0.000</td></tr><tr><td rowspan=1 colspan=3>Iλ (Prob.)   0.918 ± 0.005</td><td rowspan=1 colspan=1> $0 . 8 9 7 \pm 0 . 0 0 5$ </td><td rowspan=1 colspan=1>0.961 ± 0.001</td><td rowspan=1 colspan=2>0.898 ± 0.002</td><td rowspan=1 colspan=2>0.968 ± 0.002  1.000 ± 0.000</td></tr><tr><td rowspan=1 colspan=3>Shannon MI   0.916 ± 0.005</td><td rowspan=1 colspan=1> $0 . 8 9 8 \pm 0 . 0 0 5$ </td><td rowspan=1 colspan=1>0.960 ± 0.001</td><td rowspan=1 colspan=2>0.899 ± 0.002</td><td rowspan=1 colspan=1>0.968</td><td rowspan=1 colspan=1>± 0.002  1.000 ± 0.000</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Accuracy   0.910 ± 0.005</td><td rowspan=1 colspan=1> $0 . 8 9 8 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=1> $0 . 9 6 1 \pm 0 . 0 0 3$ </td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9</td><td rowspan=1 colspan=1>7 1 \pm 0 . 0 0 1 }$    $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Random    0.999 ± 0.015   $0 . 9 0 0 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=1> $1 . 0 6 4 \pm 0 . 0 1 7$ </td><td rowspan=1 colspan=2> $0 . 8 9 9 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $0 . 9 3 4 \</td><td rowspan=1 colspan=1>pm 0 . 0 0 5$    $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>∆α (Prob.)    $\mathbf { 2 . 7 5 3 \pm 0 . 0 1 6 }$ </td><td rowspan=1 colspan=1> $0 . 8 9 8 \pm 0 . 0 0 3$ </td><td rowspan=1 colspan=1>3.358 ± 0.019</td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 7 5 6 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 2 6 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>∆α (APS)   2.835 ± 0.038</td><td rowspan=1 colspan=1> $0 . 8 9 8 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1>3.398 ± 0.032</td><td rowspan=1 colspan=1> $0 . 9 0 0</td><td rowspan=1 colspan=1>\pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 7 5 8 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1>0.924 ± 0.001</td></tr><tr><td rowspan=1 colspan=3>Iλ (Prob.)    $2 . 8 2 4 \pm 0 . 0 2 2$ </td><td rowspan=1 colspan=1> $0 . 8 9 8 \pm 0 . 0 0 3$ </td><td rowspan=1 colspan=1>3.385 ± 0.028</td><td rowspan=1 colspan=1> $0 . 9 0 1</td><td rowspan=1 colspan=1>\pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $0 . 7 6 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1>0.924 ± 0.001</td></tr><tr><td rowspan=2 colspan=3>Leter   Shannon MI   $2 . 8 0 1 \pm 0 . 0 1 9$ Accuracy     $2 . 8 7 9 \pm 0 . 0 2 0$ </td><td rowspan=1 colspan=1> $0 . 8 9 9 \pm 0 . 0 0 3$ </td><td rowspan=1 colspan=1>3.399 ± 0.026</td><td rowspan=1 colspan=1> $0 . 9 0 1</td><td rowspan=1 colspan=1>\pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 7 5 8 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1>0.924 ± 0.001</td></tr><tr><td rowspan=1 colspan=1> $0 . 8 9 9 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $3 . 4 1 5 \pm 0 . 0 2 2$ </td><td rowspan=1 colspan=1> $0 . 9 0 1</td><td rowspan=1 colspan=1>\pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 6 3 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=1> $0 . 9 2 2 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan=1 colspan=4>Random    4.459 ± 0.184   0.897 ± 0.002</td><td rowspan=1 colspan=1>5.054 ± 0.174</td><td rowspan=1 colspan=1>0.900</td><td rowspan=1 colspan=1>± 0.001</td><td rowspan=1 colspan=1>0.642 ± 0.011</td><td rowspan=1 colspan=1>0.871 ± 0.007</td></tr><tr><td rowspan=1 colspan=4>∆α (Prob.)   $\mathbf { 1 . 0 1 0 \pm 0 . 0 0 1 }$   0.900 ± 0.001</td><td rowspan=1 colspan=1>1.139 ± 0.001</td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=1>0.906 ± 0.001</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>∆α (APS)    $1 . 0 1 1 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1>0.900 ± 0.001</td><td rowspan=1 colspan=1>1.140 ± 0.001</td><td rowspan=1 colspan=2>0.900 ± 0.000</td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 0 6 \pm 0 . 0 0 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=2>Iλ (Prob.)    $\mathbf { 1 . 0 1 0 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.900 ± 0.001</td><td rowspan=1 colspan=1> $\mathbf { 1 . 1 3 8 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=1>0.906 ± 0.001</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=2>Shannon MI   $\mathbf { 1 . 0 1 0 \pm 0 . 0 0 1 }$ </td><td></td><td rowspan=1 colspan=1>0.900 ± 0.001</td><td rowspan=1 colspan=1> $\mathbf { 1 . 1 3 8 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 0 6 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>Accuracy   1.010 ± 0.001</td><td rowspan=1 colspan=2>0.900 ± 0.001    $1 . 1 3 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 0 6 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>Random    1.593 ± 0.035</td><td rowspan=1 colspan=2>0.901 ± 0.001   1.731 ± 0.036</td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td rowspan=1 colspan=1> $0 . 7 1 7 \pm 0 . 0 1 1$ </td><td rowspan=1 colspan=1>0.996 ± 0.000</td></tr><tr><td rowspan=1 colspan=3>∆α (Prob.)   1.020 ± 0.002</td><td rowspan=1 colspan=1> $0 . 9 0 0 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $1 . 2 2 0 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=2> $0 . 8 9 8 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 9 0 3 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>∆α (APS)    $1 . 0 2 8 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $0 . 9 0 1 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $1 . 1 6 7 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=2> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 9 0 1 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1>0.998 ± 0.000</td></tr><tr><td rowspan=1 colspan=3>Iλ (Prob.)    $1 . 0 4 0 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $0 . 9 0 1 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $1 . 1 6 6 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=2> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>Shannon MI    $1 . 0 3 7 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=2> $0 . 9 0 1 \pm 0 . 0 0 2$    $\mathbf { 1 . 1 6 3 \pm 0 . 0 0 2 }$ </td><td rowspan=1 colspan=2> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 9 0 1 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>Accuracy    $1 . 0 2 3 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=2> $0 . 9 0 2 \pm 0 . 0 0 2$    $1 . 2 2 0 \pm 0 . 0 0 3$ </td><td rowspan=1 colspan=2> $0 . 8 9 8 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 0 4 \pm 0 . 0 0 1 }$ </td><td rowspan=1 colspan=1> $0 . 9 9 7 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan=1 colspan=3>Random     $1 . 0 6 4 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=2>0.902 ± 0.002   $1 . 2 5 9 \pm 0 . 0 0 4$ </td><td rowspan=1 colspan=2> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1>0.896 ± 0.001</td><td rowspan=1 colspan=1> $0 . 9 9 7 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan=1 colspan=5>∆α (Prob.)    $\mathbf { 1 . 4 8 9 \pm 0 . 0 0 7 }$    $0 . 8 9 9 \pm 0 . 0 0 3$   1.774 ± 0.007</td><td rowspan=1 colspan=2> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 7 9 5 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1>0.986 ± 0.000</td></tr><tr><td rowspan=1 colspan=5>∆α (APS)    $1 . 5 0 7 \pm 0 . 0 0 6$    $0 . 9 0 0 \pm 0 . 0 0 2$    $\mathbf { 1 . 7 4 2 \pm 0 . 0 0 4 }$ </td><td rowspan=1 colspan=2> $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 7 9 3 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 9 8 6 \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan=1 colspan=5>Ix (Prob.)    $1 . 5 9 2 \pm 0 . 0 1 0$    $0 . 9 0 0 \pm 0 . 0 0 2$    $1 . 7 7 4 \pm 0 . 0 0 8$ </td><td rowspan=1 colspan=2>0.899 ± 0.001</td><td rowspan=1 colspan=1>0.781 ± 0.001</td><td rowspan=1 colspan=1>0.984 ± 0.000</td></tr><tr><td rowspan=1 colspan=7>Shannon MI    $1 . 5 6 1 \pm 0 . 0 0 7$    $0 . 8 9 9 \pm 0 . 0 0 2$    $1 . 7 5 6 \pm 0 . 0 0 5$    $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 7 8 5 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1> $0 . 9 8 4 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan=1 colspan=7>Accuracy     $1 . 5 0 2 \pm 0 . 0 0 8$    $0 . 9 0 1 \pm 0 . 0 0 2$    $1 . 7 8 0 \pm 0 . 0 0 4$    $0 . 9 0 0 \pm 0 . 0 0 1$ </td><td rowspan=1 colspan=1>0.800 ± 0.002</td><td rowspan=1 colspan=1> $0 . 9 8 5 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan=1 colspan=7>Random     $1 . 7 1 1 \pm 0 . 0 1 2$    $0 . 9 0 0 \pm 0 . 0 0 2$    $1 . 9 9 6 \pm 0 . 0 1 5$    $0 . 9 0 0 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $0 . 7 7 2 \pm 0 . 0 0 3$ </td><td rowspan=1 colspan=1>0.978 ± 0.001</td></tr><tr><td rowspan=1 colspan=8>∆α (Prob.)    $1 2 . 6 8 3 \pm 0 . 1 0 0$   0.904 ± 0.002   $1 5 . 6 6 1 \pm 0 . 1 2 7$   $0 . 9 0 2 \pm 0 . 0 0 2$    $0 . 4 6 6 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 5 3 \pm 0 . 0 0 2 }$ </td></tr><tr><td rowspan=1 colspan=7>∆α (APS)   ${ \bf 1 2 . 5 9 4 \pm 0 . 0 8 8 }$  0.902 ± 0.002   $\mathbf { 1 5 . 4 7 2 \pm 0 . 1 2 9 }$   $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 6 7 \pm 0 . 0 0 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 5 3 \pm 0 . 0 0 2 }$ </td></tr><tr><td rowspan=4 colspan=8>CI-R10     Iλ (Prob.)   13.700 ± 0.115  0.901 ± 0.002   $1 6 . 4 3 8 \pm 0 . 1 1 0$   0.901 ± 0.002Shannon MI13.763 ± 0.1150.902 ± 0.002 $1 6 . 5 9 7 \pm 0 . 1 4 5$  $0 . 9 0 2 \pm 0 . 0 0 2$    $0 . 4 5 8 \pm 0 . 0 0 3$ Accuracy12.872 ± 0.130 $0 . 9 0 2 \pm 0 . 0 0 2$  $1 5 . 8 7 7 \pm 0 . 1 5 8$  $0 . 9 0 1 \pm 0 . 0 0 2$  $\mathbf { 0 . 4 6 7 \pm 0 . 0 0 3 }$ Random    14.243 ± 0.253   $0 . 9 0 2 \pm 0 . 0 0 3$    $1 7 . 1 8 9 \pm 0 . 2 2 4$    $0 . 9 0 2 \pm 0 . 0 0 3$   $0 . 4 5 4 \pm 0 . 0 0 2$ </td><td rowspan=1 colspan=1>0.456 ± 0.003</td></tr><tr><td rowspan=1 colspan=1>0.458 ± 0.003</td><td rowspan=1 colspan=1> $0 . 7 4 1 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 7 5 2 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan=1 colspan=1> $0 . 7 3 7 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan=5 colspan=9>∆α (Prob.)   ${ \bf 4 7 . 1 5 9 \pm 0 . 2 9 7 }$   $0 . 9 0 1 \pm 0 . 0 0 1$    $5 8 . 9 9 3 \pm 0 . 3 5 2$   $0 . 9 0 0 \pm 0 . 0 0 1$    $0 . 4 7 6 \pm 0 . 0 0 1$    $0 . 7 1 1 \pm 0 . 0 0 1$ Ia-IK     ∆α (APS)   $4 7 . 2 6 2 \pm 0 . 3 2 2$    $0 . 9 0 1 \pm 0 . 0 0 1$   ${ \bf 5 8 . 1 5 5 \pm 0 . 3 3 7 }$   $0 . 9 0 1 \pm 0 . 0 0 1$    $0 . 4 7 5 \pm 0 . 0 0 1$    $0 . 7 1 1 \pm 0 . 0 0 1$ Iλ (Prob.) $5 1 . 9 5 8 \pm 0 . 4 4 3$  $0 . 9 0 1 \pm 0 . 0 0 1$  $6 3 . 8 8 7 \pm 0 . 3 5 9$  $0 . 9 0 0 \pm 0 . 0 0 1$  $0 . 4 7 2 \pm 0 . 0 0 1$ Shannon MI $5 5 . 9 5 9 \pm 0 . 3 5 8$  $0 . 9 0 0 \pm 0 . 0 0 1$  $6 8 . 6 7 5 \pm 0 . 3 2 0$  $0 . 9 0 0 \pm 0 . 0 0 1$  $0 . 4 7 1 \pm 0 . 0 0 1$ Accuracy51.504 ± 0.494 $0 . 9 0 0 \pm 0 . 0 0 1$  $6 4 . 7 9 6 \pm 0 . 5 8 1$  $0 . 9 0 0 \pm 0 . 0 0 1$  $\mathbf { 0 . 4 8 0 \pm 0 . 0 0 1 }$ Random58.187 ± 0.5220.901 ± 0.001 $7 2 . 6 2 3 \pm 0 . 6 1 6$  $0 . 9 0 0 \pm 0 . 0 0 1$ 0.472 ± 0.0010.704 ± 0.001</td></tr><tr><td rowspan=1 colspan=1>0.711 ± 0.001</td></tr><tr><td rowspan=1 colspan=1>0.707 ± 0.001</td></tr><tr><td rowspan=1 colspan=1>0.705 ± 0.001</td></tr><tr><td rowspan=1 colspan=1>0.713 ± 0.001</td></tr></table>

![](images/b2b41033689fd166f321e45f208866850a4daf2ebbe7c1c584511a1c82ef6c2a.jpg)  
Figure 12: We show the expected set size using the probability score and APS score against the fraction of candidate features selected. Each curve is the mean over 10 random splits of the evaluation pool. We show only the first half of the selection trajectory, after which the curves largely coincide. On S3 and CIFAR-100, the gap between $\Delta _ { \alpha }$ and Shannon MI is concentrated in the early steps, which becomes diluted when averaging over the full trajectory.

The metrics in Table 5 are averaged over all intermediate steps of the feature ordering. Since every criterion eventually selects all candidate features, the set sizes converge as more features are acquired, and averaging over the full trajectory can mask differences that occur early in selection. To examine this, we show the expected set size at each step for the three datasets from Table 1 in Figure 12. The $\Delta _ { \alpha }$ criteria yield noticeably smaller sets than Shannon MI when only a small fraction of the candidates has been selected for S3 and CIFAR-100. On Letter, $\Delta _ { \alpha }$ using the probability score provides only a small improvement throughout the trajectory.

## E.4 SUBSET-SPECIFIC MODEL ABLATION

In the feature selection experiments, the selection criteria and the evaluation both rely on a single mask-conditioned model. A difference between two criteria could therefore reflect how well this model fits particular subsets of features, rather than the information those features carry about Y. To test for this effect, we retrain a subset-specific model for each selected subset on CIFAR-100.

For each criterion and random split, we take the feature ordering from the main experiment and consider its first $k \in \{ 1 , 2 , 4 , 8 \}$ candidates, since the difference between the criteria is concentrated in the early steps of selection (Figure 12). We train a separate classification head on each resulting subset of the frozen backbone representation, using the same architecture and hyperparameters as the mask-conditioned head but without masking. Each subset-specific head is then calibrated and evaluated on the same data as the mask-conditioned model.

Table 6 reports the difference in expected set size between Shannon MI and $\Delta _ { \alpha }$ under both models. For simplicity, each $\Delta _ { \alpha }$ criterion is evaluated under its own score. The subset-specific models yield smaller sets than the mask-conditioned model for every criterion, and the advantage of $\Delta _ { \alpha }$ decreases when averaged over k from 4.07 to 1.89 under the probability score and from 3.84 to 2.11 under the APS score. However, $\Delta _ { \alpha }$ continues to yield smaller sets at every $k ,$ with the largest differences in the first two steps.

Table 6: Difference in expected set size between Shannon MI and $\Delta _ { \alpha }$ on CIFAR-100 when each feature ordering is evaluated with the mask-conditioned model or with subset-specific models. Positive values indicate that $\Delta _ { \alpha }$ yields smaller sets. Values are reported as mean ± standard error over 10 random splits of the evaluation pool.
<table><tr><td rowspan="2"> $k$ </td><td colspan="2">Prob.</td><td colspan="2">APS</td></tr><tr><td>Mask-cond.</td><td>Subset-specific</td><td>Mask-cond.</td><td>Subset-specific</td></tr><tr><td>1</td><td> $3 . 7 5 \pm 0 . 3 6$ </td><td> $2 . 4 9 \pm 0 . 3 2$ </td><td> $3 . 5 9 \pm 0 . 4 0$ </td><td> $2 . 3 3 \pm 0 . 3 5$ </td></tr><tr><td>2</td><td> $5 . 8 1 \pm 0 . 4 6$ </td><td> $4 . 0 6 \pm 0 . 3 5$ </td><td> $5 . 2 4 \pm 0 . 5 4$ </td><td> $3 . 8 7 \pm 0 . 4 6$ </td></tr><tr><td>4</td><td> $3 . 7 1 \pm 0 . 3 3$ </td><td> $0 . 4 2 \pm 0 . 2 4$ </td><td> $2 . 9 8 \pm 0 . 6 7$ </td><td> $1 . 3 2 \pm 0 . 3 1$ </td></tr><tr><td>8</td><td> $3 . 0 0 \pm 0 . 4 6$ </td><td> $0 . 5 8 \pm 0 . 3 2$ </td><td> $3 . 5 6 \pm 0 . 6 4$ </td><td> $0 . 9 1 \pm 0 . 2 8$ </td></tr><tr><td> $\operatorname { A v g } .$ </td><td> $4 . 0 7 \pm 0 . 2 7$ </td><td> $1 . 8 9 \pm 0 . 1 4$ </td><td> $3 . 8 4 \pm 0 . 4 1$ </td><td> $2 . 1 1 \pm 0 . 1 4$ </td></tr></table>

## F COMPUTE RESOURCES

All training and analysis were performed on a single machine running Ubuntu 24.04 with 32 GB of RAM and an NVIDIA GeForce RTX 5070 Ti GPU. The hyperparameter grid search required approximately 10 hours and subsequent model training required approximately 12 hours. The vali dation experiments were run on a single CPU thread without GPU acceleration, whereas the feature selection experiments evaluated the mask-conditioned models on the GPU. All downstream analyses required approximately 4 hours in total.

## G EXPERIMENTAL ASSETS

We obtain the Human Activity Recognition dataset (Reyes-Ortiz et al., 2013) directly from the UCI Machine Learning Repository, available under the CC BY 4.0 license. For ImageNet-1k, we use the 64 × 64 downsampled version of Chrabaszcz et al. (2017), which is made available through the ChocolateDave/imagenet-64 mirror on Hugging Face. The pretrained ResNet-18 weights (He et al., 2016) are the IMAGENET1K V1 weights distributed with torchvision. All remaining datasets are obtained through OpenML or torchvision, with their original sources cited in Section 5 and Appendix B. Code for reproducing our analyses will be released upon publication.