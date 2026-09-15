# Conformal Individual Treatment Efect Estimation under Networked Interference

Matteo Zecchin<sup>†</sup>

Osvaldo Simeone<sup>§</sup>

<sup>†</sup>Communication Systems Department, EURECOM, Sophia Antipolis, France <sup>§</sup>Institute for Intelligent Networked Systems (INSI), Northeastern University London, London, UK zecchin@eurecom.fr, o.simeone@{nulondon.ac.uk, northeastern.edu}

## Abstract

Conformal counterfactual prediction constructs prediction sets with finite-sample coverage guarantees for counterfactual outcomes and individual treatment efects under the no-interference assumption. In this work, we relax this assumption by allowing each unit’s potential outcomes to depend on other units’ treatments and covariates. In this setting, propensity-score reweighting does not restore weighted exchangeability, and existing methods may fail to achieve valid coverage. To address this issue, we develop interference-adjusted weighted conformal prediction that accounts for interference by constructing an observable upper bound on the ideal and unobserved conformal pvalue under the target intervention. The resulting prediction sets provide finite-sample marginal coverage guarantees for counterfactual outcomes and individual treatment effects in both transductive and inductive settings. We also derive a sharper construction when intervention-induced changes in nonconformity scores are bounded. Numerical experiments show that our methods preserve nominal coverage, whereas existing methods may not.

## 1 Introduction

## 1.1 Motivation

Many practical decisions require predicting how a particular unit would have responded to an action it did not receive. A physician may wish to assess what a patient’s outcome would have been under an alternative treatment (Zhao et al., 2012), an online platform may need to evaluate how a user would have responded to an advertising campaign (Gordon et al., 2019), and a network operator may need to assess what would have happened had resources been allocated to a specific user (Bao et al., 2017; Hou et al., 2025). Such decisions require comparing the observed factual outcome under the assigned treatment with the counterfactual outcome that would have been observed under an alternative treatment. The fundamental challenge is that only the factual outcome is observed, while the remaining potential outcomes must be predicted from data (Rubin, 1974, 2005).

The reliability of the resulting decision hinges on faithfully quantifying the uncertainty associated with the counterfactual prediction. Conformal prediction has emerged as a powerful framework for this purpose, providing model-agnostic prediction sets with finite-sample marginal coverage guarantees for missing potential outcomes and individual treatment effects (Shafer and Vovk, 2008). Under the assumption that the treatment assigned to a given unit does not afect the outcomes of other units, counterfactual prediction can be formulated as a prediction problem under covariate shift, and weighted conformal prediction (WCP) (Tibshirani et al., 2019) can be used to construct valid prediction sets via propensity-score weighting (Lei and Cand\`es, 2021).

In many applications, however, a unit’s outcome depends not only on its own treatment but also on the treatments assigned to other units (Hudgens and Hal loran, 2008; Aronow and Samii, 2017; Forastiere et al., 2021). Vaccinating one person can change the infection risk faced by others; showing an advertisement to one user can influence the behavior of their peers; and allocating resources to one user can reduce the resources available to others. In these settings, intervening on the treatment of a target unit can alter the distribution of outcomes of other units. This distributional shift cannot be expressed via unit-level propensity weights and, as a result, WCP may fail to provide valid coverage under interference.

![](images/80c82bb6c2539227c20c1e04eb92968c3417b27b8f32187d09b402df78d96fcf.jpg)  
Figure 1: Efect of interference on weighted conformal prediction (WCP) (Lei and Cand\`es, 2021). The top row shows one realization of a covariate-dependent knearest-neighbor exposure graph $\mathcal { G } ( X _ { 1 : N } )$ for increasing values of k. The bottom row reports the empirical marginal coverage and normalized prediction-set size for a target coverage level of $1 - \alpha = 0 . 8$ . WCP increasingly undercovers as interference between units increases, i.e., as k increases, whereas the proposed interference-adjusted WCP (IA-WCP) maintains the target marginal coverage by returning larger predic tion sets.

This phenomenon is illustrated in Figure 1. The example considers a simple model of local interactions in which each of N units is connected to its k nearest neighbors in covariate space. Treatments are assigned independently according to the units’ covariates, and each unit’s outcome depends on the fraction of its treated neighbors. When $k = 0 .$ , there is no interference, and WCP attains the target coverage. As the number of neighbors k increases, an intervention on the target unit alters the exposures of a growing number of units. Consequently, the discrepancy between the observational and interventional distributions increases, and WCP exhibits increasingly severe undercoverage.

To address this problem, we develop a counterfactual conformal procedure that preserves marginal coverage under networked interference. The proposed interference-adjusted WCP (IA-WCP) is based on a corrected p-value computed from the observed data (Vovk et al., 2005; Shafer and Vovk, 2008) and knowledge of the interference structure. As shown in Figure 1, IA-WCP reduces to WCP in the absence of interference and, unlike WCP, maintains coverage as interference increases, at the cost of wider prediction sets. We develop IA-WCP for both transductive and inductive settings (Clarkson, 2023; H. Zargarbashi et al., 2023; H. Zargarbashi and Bojchevski, 2024) and establish finite-sample marginal coverage guarantees in each case. We also provide an example showing that the correction can be tight with respect to marginal coverage and introduce a sharper variant, interferenceadjusted WCP+ (IA-WCP+), for settings in which intervention-induced changes in the afected nonconformity scores are bounded.

## 1.2 Related Work

WCP provides prediction sets for counterfactual outcomes and individual treatment efects under covariate shift and no interference (Tibshirani et al., 2019; Lei and Cand\`es, 2021). Recent extensions of WCP address hidden confounding and poor overlap while retaining the no-interference assumption (Jin et al., 2023; Chen et al., 2024; Farzaneh et al., 2025; Qchohi et al., 2026). Separately, research on causal inference under interference introduced the notion of exposure mappings to identify and estimate direct and spillover ef fects (Hudgens and Halloran, 2008; Aronow and Samii, 2017; Forastiere et al., 2021). Conformal methods for networked data have primarily targeted factual node labels or links (Clarkson, 2023; Huang et al., 2023a; Zhao et al., 2024). The closely related method of Zhou et al. (2025) constructs conformal prediction sets for ITEs on networked data but assumes no interference. Our work connects these lines of research by constructing prediction sets with finite-sample marginal coverage guarantees for individual potential outcomes and treatment efects under network interference. A more detailed discussion is provided in Appendix A.

## 1.3 Contributions

Our main contributions are as follows:

• We formulate conformal inference for ITEs and treatment efects under interference in both the transductive setting, where the targets are units in the observed population, and the inductive setting, where the goal is to generalize to new units.

• We develop IA-WCP, an interference-adjusted counterfactual conformal procedure that produces prediction sets for the potential outcomes with finite-sample marginal coverage guarantees under interference, and reduces to WCP (Lei and

Cand\`es, 2021) in the absence of interference.

• Under a score-stability condition, we present a sharper version of IA-WCP, referred to as IA-WCP+, that can produce smaller prediction sets while maintaining coverage.

• We provide experimental results showing that WCP can undercover under interference, whereas the proposed IA-WCP and IA-WCP+ preserve marginal coverage.

## 2 Setting

We consider a population of N units indexed by $i \in$ $\{ 1 , \ldots , N \}$ Each unit has covariates $X _ { i } ~ \in ~ { \mathcal { X } }$ and receives a binary treatment $T _ { i } ~ \in ~ \{ 0 , 1 \}$ . We write $\boldsymbol { X } _ { 1 : N } = ( X _ { 1 } , \ldots , X _ { N } )$ and $T _ { 1 : N } = ( T _ { 1 } , \dots , T _ { N } ) $ , and use $X _ { 1 : N } ^ { - i }$ and $T _ { 1 : N } ^ { - i }$ for the corresponding vectors with unit i removed. The covariates are independently sampled from a common population distribution $P _ { X }$ , and, as detailed next, treatments are conditionally independent under a common covariate-dependent policy.

Assumption 1 (Treatment assignment and overlap). There exists a propensity score function $\pi : \mathcal { X }  ( 0 , 1 )$ such that, conditionally on covariates $X _ { 1 : N }$ , the assigned treatments are independent and distributed as

$$
T _ { i } \mid X _ { 1 : N } \sim \operatorname { B e r n o u l l i } ( \pi ( X _ { i } ) ) , \quad i = 1 , \ldots , N .\tag{1}
$$

Moreover, there exists a strictly positive constant $\underline { { \pi } } \in$ (0, 1/2] such that

$$
\underline { { \pi } } \leq \pi ( x ) \leq 1 - \underline { { \pi } } , \qquad \forall x \in \mathcal { X } .\tag{2}
$$

For a given treatment vector $T _ { 1 : N }$ , define the control and treated index sets, respectively, as

$$
\begin{array} { r } { \mathcal { T } _ { 0 } = \{ j \in \{ 1 , \dots , N \} : T _ { j } = 0 \} , } \end{array}\tag{3}
$$

and

$$
{ \mathcal { T } } _ { 1 } = \{ j \in \{ 1 , \ldots , N \} : T _ { j } = 1 \} .\tag{4}
$$

For each unit i, let $( Y _ { i } ( 0 ) , Y _ { i } ( 1 ) )$ ) denote its potential outcomes under treatment $T _ { i } = 0$ and $T _ { i } = 1$ . Departing from the standard SUTVA, we allow these outcomes to depend on covariates and treatments of a subset of units, excluding unit i (Tchetgen and VanderWeele, 2012; Hudgens and Halloran, 2008). Specif ically, for each unit i, we summarize this inter-unit dependence through an exposure variable $\begin{array} { l l } { E _ { i } } & { \in \mathcal { E } } \end{array}$ (Aronow and Samii, 2017) that depends on units in a neighborhood set

$$
\mathcal { N } _ { i } ( X _ { 1 : N } ) \subseteq \{ 1 , \dots , N \} \setminus \{ i \} .\tag{5}
$$

The neighborhood $\mathcal { N } _ { i } ( X _ { 1 : N } )$ generally depends on the covariates $X _ { 1 : N }$ . For example, in Figure 1 the set $\mathcal { N } _ { i } ( X _ { 1 : N } )$ corresponds to the k nearest neighbors to unit i. Furthermore, the neighborhood relation may be directed, meaning that $j \in \mathcal { N } _ { i } ( X _ { 1 : N } )$ need not imply $i \in \mathcal { N } _ { j } ( X _ { 1 : N } )$ . Overall, the neighborhood relations $\{ \mathcal { N } _ { i } ( X _ { 1 : N } ) \} _ { i = 1 } ^ { N }$ define a covariate-dependent exposure (directed) graph $\mathcal { G } ( X _ { 1 : N } )$ with nodes given by the units $\{ 1 , \ldots , N \}$ and edges (i, j) with $j \in \mathcal { N } _ { i } ( X _ { 1 : N } )$

In this work, we restrict attention to permutationequivariant neighborhood rules. For a permutation σ of the set $\{ 1 , \ldots , N \}$ , write

$$
T _ { \sigma ( 1 : N ) } = ( T _ { \sigma ( 1 ) } , \ldots , T _ { \sigma ( N ) } ) ,\tag{6}
$$

and

$$
X _ { \sigma ( 1 : N ) } = ( X _ { \sigma ( 1 ) } , \ldots , X _ { \sigma ( N ) } ) .\tag{7}
$$

Assumption 2 (Permutation-equivariant neighborhood rule). For any permutation σ of the set $\{ 1 , \ldots , N \}$ , and every unit $i \in \{ 1 , \ldots , N \}$ , the neighborhood rule satisfies

$$
\begin{array} { r } { \mathcal { N } _ { i } ( X _ { \sigma ( 1 : N ) } ) = \left\{ j : \sigma ( j ) \in \mathcal { N } _ { \sigma ( i ) } ( X _ { 1 : N } ) \right\} . } \end{array}\tag{8}
$$

Example: Given a distance d : $: \mathcal { X } \times \mathcal { X } \to \mathbb { R } _ { + }$ and a radius $r \geq 0$ , the radius-r neighborhood rule corresponds to $\mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ j \neq i : d ( X _ { i } , X _ { j } ) \leq r \}$

This rule satisfies Assumption 2 since permutations of units preserve pairwise distances. For illustration, consider $X _ { 1 } ~ = ~ 0 , ~ X _ { 2 } ~ = ~ 0 . 5$ , and $X _ { 3 } ~ = ~ 1$ , with $d ( x , x ^ { \prime } ) ~ = ~ | x - x ^ { \prime } | , ~ r ~ = ~ 0 . 6$ , and a permutation $\sigma ( 1 ) = 2 , \sigma ( 2 ) = 3 , \sigma ( 3 ) = 1$

Then, we have $\begin{array} { r c l } { { X _ { \sigma ( 1 : 3 ) } } } & { { = } } & { { ( X _ { 2 } , X _ { 3 } , X _ { 1 } ) } } \end{array}$ and, for $\begin{array} { r l r } { i } & { { } = } & { 1 } \end{array}$ , the neighborhood set is $\begin{array} { r l } { \mathcal { N } _ { 1 } ( X _ { \sigma ( 1 : 3 ) } ) } & { { } = } \end{array}$ $\left\{ j \neq 1 : d ( X _ { \sigma ( 1 ) } , X _ { \sigma ( j ) } ) \leq 0 . 6 \right\} = \{ 2 , 3 \}$ , which can be seen to satisfy Assumption 2. More examples of neighborhood rules satisfying this property are given in $\mathrm { A p \mathrm { - } }$ pendix B.

We define the multiset of treatments and covariates for units in neighborhood $\mathcal { N } _ { i } ( X _ { 1 : N } )$ as

$$
\mathcal { L } _ { i } ( X _ { 1 : N } , T _ { 1 : N } ) = \big \{ \big \{ ( X _ { j } , T _ { j } ) : j \in \mathcal { N } _ { i } ( X _ { 1 : N } ) \big \} \big \} ,\tag{9}
$$

where the doubled braces indicate that repeated covariate–treatment pairs are retained with their multiplicities. Each exposure variable $E _ { i }$ depends only on $X _ { i }$ and variables in the multiset $\mathscr { L } _ { i } ( X _ { 1 : N } , T _ { 1 : N } )$ as detailed next.

Assumption 3 (Exposure generation and observation). Conditionally on $\left( X _ { 1 : N } , T _ { 1 : N } \right)$ , the exposure variables $E _ { 1 } , \dots , E _ { N }$ are independent. Moreover, there exists a common conditional distribution $P _ { E | X , \mathcal { L } }$ such that, for each $i = 1 , \ldots , N$ , we have

![](images/1bfdda295b4276dc95093a33a9c02ea306b3cadf8524cb3d0ab913e6edebfcef.jpg)  
Figure 2: Graphical model of the setting considered in this work. For each unit i, the covariate $X _ { i }$ is independently sampled from a common population distribution $P _ { X }$ , and the treatment $T _ { i }$ is independently assigned according to a common covariate-dependent policy $\pi ( X _ { i } )$ . The exposure $E _ { i }$ depends on the unit’s covariate $X _ { i }$ and on the covariates $X _ { j }$ and treatments $T _ { j }$ of the other units $j \in \mathcal { N } _ { i } ( X _ { 1 : N } )$ . The observed outcome $Y _ { i } ^ { \mathrm { { o b s } } }$ equals the potential outcome corresponding to the assigned treatment, and depends on the unit’s covariate and exposure.

$$
E _ { i } \mid \left( X _ { 1 : N } , T _ { 1 : N } \right) \sim P _ { E | X , \mathcal { L } } \left( \cdot \mid X _ { i } , \mathcal { L } _ { i } ( X _ { 1 : N } , T _ { 1 : N } ) \right) .\tag{10}
$$

Each exposure $E _ { i }$ is either directly observed or can be exactly recovered from $X _ { i }$ and the multiset $\mathscr { L } _ { i } ( X _ { 1 : N } , T _ { 1 : N } )$

The assumption that the exposure variable $E _ { i }$ is observable is natural when the exposure mechanism is known. Examples include the number or fraction of treated neighbors within a known group or cluster and distance-weighted exposures that depend on observed covariates (Aronow and Samii, 2017; Forastiere et al., 2022; Cai et al., 2024).

Assumption 4 (No hidden interference confounding). Conditional on the covariates, exposures, and treatments $( X _ { 1 : N } , E _ { 1 : N } , T _ { 1 : N } )$ where $\begin{array} { l } { { E _ { 1 : N } ~ = ~ ( E _ { 1 } , \ldots , E _ { N } ) } } \end{array}$ , the potential outcome pairs $( Y _ { i } ( 0 ) , Y _ { i } ( 1 ) ) _ { i = 1 } ^ { N }$ are independent across units. Moreover, for $t ~ \in ~ \{ 0 , 1 \}$ , there exists a common conditional distribution $P _ { Y ( t ) \mid X , E }$ such that, for each $i \in$ $\{ 1 , \ldots , N \}$ ,

$$
Y _ { i } ( t ) \mid ( X _ { 1 : N } , E _ { 1 : N } , T _ { 1 : N } ) \sim P _ { Y ( t ) \mid X , E } ( \cdot \mid X _ { i } , E _ { i } ) .\tag{11}
$$

Assumption 5 (Consistency). For each observed unit i, the observation $Y _ { i } ^ { \mathrm { { o b s } } }$ equals the potential outcome evaluated at the realized treatment, i.e.,

$$
Y _ { i } ^ { \mathrm { o b s } } = \left\{ { \begin{array} { l l } { Y _ { i } ( 0 ) , } & { i f T _ { i } = 0 , } \\ { Y _ { i } ( 1 ) , } & { i f T _ { i } = 1 . } \end{array} } \right.\tag{12}
$$

Overall, as a result of the observations above, each observation is distributed as

$$
Y _ { i } ^ { \mathrm { o b s } } \mid ( X _ { 1 : N } , T _ { 1 : N } , E _ { 1 : N } ) \sim P _ { Y ( T _ { i } ) | X , E } ( \cdot \mid X _ { i } , E _ { i } ) ,\tag{13}
$$

and the joint distribution of all relevant variables is specified by the following conditional distributions

$$
X _ { i } \sim P _ { X } ,\tag{14a}
$$

$$
T _ { i } \mid X _ { i } \sim \operatorname { B e r n o u l l i } ( \pi ( X _ { i } ) ) ,\tag{14b}
$$

$$
E _ { i } \mid \left( X _ { 1 : N } , T _ { 1 : N } \right) \sim P _ { E | X , \mathcal { L } } ( \cdot \mid X _ { i } , \mathcal { L } _ { i } ( X _ { 1 : N } , T _ { 1 : N } ) ) ,\tag{14c}
$$

$$
Y _ { i } ( T _ { i } ) \mid ( X _ { 1 : N } , E _ { 1 : N } , T _ { 1 : N } ) \sim P _ { Y ( T _ { i } ) | X , E } ( \cdot \mid X _ { i } , E _ { i } )\tag{14d}
$$

$$
Y _ { i } ^ { \mathrm { o b s } } = Y _ { i } ( T _ { i } )\tag{14e}
$$

Under this joint distribution, variables associated with diferent units are generally dependent.

## 2.1 Problem Formulation

In this section, we formalize the individual treatment efect under interference and state the corresponding counterfactual coverage objectives in the transductive and inductive settings. The primary inferential target is the individual treatment efect (ITE), defined as the diference between a unit’s two potential outcomes at the same exposure,

$$
\tau = Y ( 1 ) - Y ( 0 ) .\tag{15}
$$

Given the observational dataset $\mathcal { D } ,$ to be specified dif ferently for the inductive and transductive settings, and a unit with covariates X and exposure E, the goal is to construct an ITE prediction set $\Gamma ^ { \mathrm { I T E } } ( X , E )$ satisfying the marginal coverage guarantee

$$
\operatorname* { P r } [ \tau \in \Gamma ^ { \mathrm { I T E } } ( X , E ) ] \geq 1 - \alpha ,\tag{16}
$$

for a user-defined miscoverage level $\alpha \in ( 0 , 1 )$ , where the probability is taken over both the observational data D and the test unit. We consider two instantiations of this requirement.

## 2.1.1 Transductive ITE Estimation

As illustrated in the left panel of Figure 3, in the transductive setting we are given a dataset $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $( ( X _ { i } , T _ { i } , E _ { i } , Y _ { i } ^ { \mathrm { o b s } } ) ) _ { i = 1 } ^ { N }$ for all N units, and the goal is to construct a family of prediction sets $\{ \Gamma _ { i } ^ { \mathrm { I T E } } \} _ { i = 1 } ^ { N }$ that cover the ITEs of the observed units on average. Specifically, the transductive ITE coverage requirement is

$$
\operatorname* { P r } _ { I \sim \operatorname { U n i f } ( \{ 1 , \dots , N \} ) } \bigl [ Y _ { I } ( 1 ) - Y _ { I } ( 0 ) \in \Gamma _ { I } ^ { \mathrm { I T E } } ( X _ { I } , E _ { I } ) \bigr ] \geq 1 - \alpha ,\tag{17}
$$

where the probability is taken over the random index $I \sim \operatorname { U n i f } ( \{ 1 , \dots , N \} )$ , as well as over the randomness in the observational dataset D and the potential outcomes.

Since the factual outcome $Y _ { i } ( T _ { i } )$ is observed for every unit, constructing an ITE prediction set reduces to constructing a prediction set for the missing potential outcome $Y _ { i } ( \bar { T } _ { i } )$ , where ${ \bar { T } } _ { i } = 1 - T _ { i }$ denotes the counterfactual treatment. Let $\Gamma _ { i } ^ { \bar { T } _ { i } }$ denote a prediction set for this missing potential outcome. Combining this set with the observed factual outcome yields the sets

$$
\Gamma _ { i } ^ { \mathrm { I T E } } = \left\{ \begin{array} { l l } { \Gamma _ { i } ^ { 1 } - Y _ { i } ( 0 ) , } & { \mathrm { i f } \ T _ { i } = 0 , } \\ { Y _ { i } ( 1 ) - \Gamma _ { i } ^ { 0 } , } & { \mathrm { i f } \ T _ { i } = 1 , } \end{array} \right.\tag{18}
$$

where for sets $A \subset  { \mathbb { R } } ^ { d }$ and $B \subset \mathbb { R } ^ { d }$ , we have defined the diference set $A - B = \{ a - b : a \in A , b \in B \}$ Consequently, if the counterfactual potential-outcome prediction sets satisfy

$$
\operatorname* { P r } _ { I \sim \mathrm { U n i f } ( \{ 1 , \dots , N \} ) } \left[ Y _ { I } ( \bar { T } _ { I } ) \in \Gamma _ { I } ^ { \bar { T } _ { I } } \right] \geq 1 - \alpha ,\tag{19}
$$

then the induced sets $\{ \Gamma _ { i } ^ { \mathrm { I T E } } \} _ { i = 1 } ^ { N }$ satisfy the transductive ITE coverage requirement in (17).

## 2.1.2 Inductive ITE Estimation

As illustrated in the right panel of Figure 3, in the inductive setting we are given a dataset D including tuples $\mathcal { D } = \{ ( \breve { X } _ { i } , T _ { i } , E _ { i } , \breve { Y } _ { i } ^ { \mathrm { o b s } } ) \} _ { i = 1 } ^ { N - 1 }$ for $N - 1$ units, and the goal is to construct a set predictor $\Gamma ^ { \mathrm { I T E } }$ that covers the ITE of a new unit with covariates $X _ { N } \sim P _ { X }$ and an exposure level $E _ { N }$ given by the exposure that this unit would have experienced in the network of N interfering units whose first $N - 1$ units have covariates and treatments as in D.

Formally, the output of inductive ITE estimation is a set predictor $\Gamma ^ { \mathrm { I \hat { T } E } }$ satisfying the marginal coverage guarantee

$$
\begin{array} { r } { \operatorname* { P r } \left[ Y _ { N } ( 1 ) - Y _ { N } ( 0 ) \in \Gamma ^ { \mathrm { I T E } } ( X _ { N } , E _ { N } ) \right] \geq 1 - \alpha , } \end{array}\tag{20}
$$

where the probability is over the dataset D and the test unit $( X _ { N } , E _ { N } , Y _ { N } ( 0 ) , Y _ { N } ( 1 ) )$ ). Unlike in the transductive setting, neither potential outcome is observed for the test unit N. Therefore, the ITE prediction set $\Gamma ^ { \mathrm { I T E } } ( X _ { N } , E _ { N } )$ must be constructed from prediction sets $\Gamma ^ { 0 } ( X _ { N } , E _ { N } )$ and $\Gamma ^ { 1 } ( X _ { N } , E _ { N } )$ for both potential outcomes.

![](images/77ab956abe55ff068f20a16471a8f22b48fbe7e84aa3a8cc707901dbea8a4acb.jpg)  
Figure 3: Illustration of the ITE estimation task in the transductive and inductive settings. In the transductive setting, in the left panel, given a network of N interfering units, the goal is to estimate the potential outcome of a random unit i picked uniformly at random from the observed units. In the inductive setting, in the right panel, given a network of $N - 1$ interfering units, the inferential targets are the two unobserved potential outcomes associated with an additional unit sampled from the marginal distribution $P _ { X }$ and embedded in the network.

In the following, we detail the proposed solution for the transductive setting, and present its counterpart for inductive inference, which follows the same principles, in Appendix F.

## 3 Transductive Interference-Adjusted Weighted Conformal Prediction

In this section, we introduce IA-WCP for the transductive setting. Without loss of generality, we condition on the event that the randomly selected test unit $I \sim \operatorname { U n i f } ( \{ 1 , \dots , N \} )$ is a control unit, i.e., $T _ { I } = 0 .$ and construct a prediction set for its missing treated potential outcome $Y _ { I } ( 1 )$ . In what follows, we suppress the dependence on the target treatment in the notation whenever it is clear from context.

## 3.1 Weighted Conformal Prediction

We first review conventional WCP for counterfactual outcomes (Tibshirani et al., 2019; Lei and Cand\`es, 2021) and explain why it is generally invalid under interference. Following the construction in (Lei and Cand\`es, 2021), let $S : \mathcal { X } \times \mathcal { E } \times \mathcal { Y }  \mathbb { R }$ be a fixed nonconformity scoring function. For every treated calibration unit $j \in \mathcal { T } _ { 1 }$ , define the observational calibration score $V _ { j } ^ { \mathrm { o b s } } = S ( X _ { j } , E _ { j } , Y _ { j } ( 1 ) )$ , and, for a candidate value $\bar { y \in \mathcal { V } }$ and test unit I, define the test score $V _ { I } ( y ) = S ( X _ { I } , E _ { I } , y )$

To correct for the covariate shift induced by treatment assignment, WCP (Tibshirani et al., 2019; Lei and Cand\`es, 2021) assigns to each unit $j \in \mathcal { T } _ { 1 } \cup \{ I \}$ the propensity-ratio-based weight

$$
W _ { j } = \frac { 1 - \pi ( X _ { j } ) } { \pi ( X _ { j } ) } ,\tag{21}
$$

and it defines the observational conformal p-value as

$$
p _ { I , y } ^ { \mathrm { o b s } } = \frac { W _ { I } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { I } ( y ) \} } { W _ { I } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } } .\tag{22}
$$

Finally, the prediction set is constructed as

$$
\Gamma _ { I } ^ { \mathrm { W C P } } = \{ y \in \mathscr { V } : p _ { I , y } ^ { \mathrm { o b s } } > \alpha \} .\tag{23}
$$

As proved in (Lei and Cand\`es, 2021), the marginal coverage guarantee in (19) holds for the missing potential outcome $Y _ { I } ( 1 )$ under the standard SUTVA. In fact, in this case, the discrepancy between the distributions of calibration and test scores is fully accounted for by the covariates, and the weights in (21) compensate for this shift. Under network interference, however, a residual discrepancy may remain through the exposure variables, even after correcting for covariate shift.

## 3.2 Transductive IA-WCP

Define the set of treated calibration units whose neigh borhoods contain the test unit I as

$$
\mathcal { H } _ { I } = \left\{ j \in \mathbb { Z } _ { 1 } : I \in \mathcal { N } _ { j } ( X _ { 1 : N } ) \right\} .\tag{24}
$$

IA-WCP modifies the WCP prediction set (23) as

$$
\Gamma _ { I } ^ { \mathrm { I A - W C P } } = \left\{ y \in \mathcal { V } : p _ { I , y } ^ { \mathrm { o b s } } > ( \alpha - \rho _ { I , y } ) + \right\} ,\tag{25}
$$

where $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ and

$$
\rho _ { I , y } = \frac { \sum _ { j \in \mathcal { H } _ { I } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \} } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } .\tag{26}
$$

The correction $\rho _ { I , y }$ is the fraction of total conformal weight assigned to the afected calibration units whose observational score lies below the candidate test score. Intuitively, this term captures the discrepancy between calibration and test scores caused by the presence of network interference. In particular, note that, in the absence of interference, we have $\mathcal { H } _ { I } = \emptyset$ and $\rho _ { I , y _ { \mathrm { - } \Omega } } =$ 0 for every $y \in \mathcal { V }$ I reduces to the WCP set (Lei and Cand\`es, 2021) in (23).

The following result shows that the correction term (26) is suficient to restore the marginal guarantee (19).

Theorem 1. Under Assumptions 1–5, for a randomly chosen control unit I, the IA-WCP set $\Gamma _ { I } ^ { \mathrm { I A - W C P } }$ in (25) satisfies the inequality

$$
\operatorname* { P r } \Big [ Y _ { I } ( 1 ) \notin \Gamma _ { I } ^ { \mathrm { I A - W C P } } \Big | T _ { I } = 0 \Big ] \leq \alpha .\tag{27}
$$

Proof. See Appendix C.

When $T _ { I } ~ = ~ 1$ , the analogous construction uses the control units as the calibration sample and the reverse propensity-ratio weights $W _ { j } = \pi ( X _ { j } ) / ( 1 - \pi ( X _ { j } ) )$ , ensuring that the resulting set (18) satisfies the desired condition (17).

To explain the rationale behind the correction term (26) introduced by IA-WCP, we outline the main steps in the proof of Theorem 1. First, we condition on the event

$$
\begin{array} { c } { A _ { \mathcal { T } _ { 1 } , I } = \big \{ T _ { j } = 1 \mathrm { ~ f o r ~ } j \in \mathcal { T } _ { 1 } , T _ { I } = 0 , \ } \\ { T _ { j } = 0 \mathrm { ~ f o r ~ } j \not \in \mathcal { T } _ { 1 } \cup \{ I \} \big \} , } \end{array}\tag{28}
$$

which states that unit I is a control unit and that $\mathcal { T } _ { 1 }$ is the set of treated units. As $\mathcal { T } _ { 1 }$ ranges over all subsets of $\{ 1 , \ldots , N \} \setminus \{ I \}$ , the events $A _ { \mathcal { T } _ { 1 } , I }$ form a disjoint partition of $\{ T _ { I } = 0 \}$ . It therefore sufices to establish (27) conditionally on each event $A _ { \mathcal { T } _ { 1 } , I }$ and then apply the law of total probability.

Conditional on $A _ { \mathcal { T } _ { 1 } , I } ,$ as shown in Appendix C.1, the covariates $( X _ { j } ) _ { j \in { \mathbb { Z } } _ { 1 } \cup \{ I \} }$ are weighted exchangeable with the weights in (21). This is the key fact underlying the validity of WCP in the absence of interference.

In the presence of interference, however, this does not imply the weighted exchangeability of the calibration and test scores due to the dependence of the outcomes on the exposure variables. In particular, conditional on $A _ { \mathcal { T } _ { 1 } , I }$ and $X _ { 1 : N }$ , the joint distribution $p \left( ( Y _ { j } ( 1 ) , E _ { j } ) _ { j \in \mathbb { Z } _ { 1 } \cup \{ I \} } \mid A _ { \mathbb { Z } _ { 1 } , I } , X _ { 1 : N } \right)$ need not be invariant under permutations that exchange unit I with a calibration unit.

IA-WCP rests on the observation that this source of non-exchangeability disappears under a modified interventional distribution in which the treatment of unit I is changed to one. In fact, denoting the resulting exposures and potential outcomes by $( \tilde { E } _ { j } ^ { I } ) _ { j \in \mathcal { T } _ { 1 } \cup \{ I \} }$ and $( \tilde { Y } _ { j } ^ { I } ( 1 ) ) _ { j \in \mathbb { Z } _ { 1 } \cup \{ I \} }$ respectively, their conditional joint distribution $p \left( ( \tilde { Y } _ { j } ^ { I } ( 1 ) , \tilde { E } _ { j } ^ { I } ) _ { j \in \mathcal { I } _ { 1 } \cup \{ I \} } \mid A _ { \mathcal { T } _ { 1 } , I } , X _ { 1 : N } , \mathrm { d o } ( T _ { I } = 1 ) \right)$ can be shown to be exchangeable under permutations of test and calibration units (see Appendix C.1). Combined with the weighted exchangeability of the covari ates, this implies that the prediction set

$$
\tilde { \Gamma } _ { I } ^ { \mathrm { W C P } } = \{ y \in \mathcal { V } : p _ { I , y } ^ { \mathrm { i n t } } > \alpha \} ,\tag{29}
$$

built using the interventional conformal p-value

$$
p _ { I , y } ^ { \mathrm { i n t } } = \frac { W _ { I } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { I } ^ { \mathrm { i n t } } ( y ) \} } { W _ { I } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } } ,\tag{30}
$$

with scores $V _ { j } ^ { \mathrm { i n t } } = S ( X _ { j } , \tilde { E } _ { j } ^ { I } , \tilde { Y } _ { j } ^ { I } ( 1 ) )$ for $j \in \mathcal { T } _ { 1 }$ and $V _ { I } ^ { \mathrm { i n t } } ( y ) = S ( X _ { I } , \tilde { E } _ { I } ^ { I } , y )$ satisfies the marginal coverage guarantee.

Lemma 1. Under Assumptions 1–5, for a randomly chosen control unit I, the ideal WCP set $\tilde { \Gamma } _ { I } ^ { \mathrm { W C P } }$ in (29) satisfies the inequality

$$
\mathrm { P r } \left[ Y _ { I } ( 1 ) \notin \tilde { \Gamma } _ { I } ^ { \mathrm { W C P } } \Big | T _ { I } = 0 \right] \leq \alpha .\tag{31}
$$

Proof. See Appendix C.1.

□

The last step of the proof of Theorem 1 is to show that it is possible to construct a coupling of the observational and interventional vectors of nonconformity scores, that is, a joint distribution whose marginals coincide with their respective observational and interventional distributions, such that for every $y \in \mathcal { V }$ , the following bound holds

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } , \quad \mathrm { a . s . }\tag{32}
$$

The correction term $\rho _ { I , y }$ in (26) then accounts for the discrepancy between the observable and interventional conformal p-values, ensuring that the set $\Gamma _ { I } ^ { \mathrm { I A - W C P } }$ includes $\tilde { \Gamma } _ { I } ^ { \mathrm { W C P } }$ almost surely, and completing the proof of Theorem 1.

## 3.3 Transductive IA-WCP+

The correction in (26) is worst-case, as it is derived without imposing any restriction on how the observational and interventional nonconformity scores of the afected units may difer. Appendix D provides an example in which the correction is tight in terms of marginal coverage, in the sense that the prediction sets induced by the ideal interventional and corrected observational p-values have identical marginal coverage. In many applications, however, changing the treatment of one unit induces only a bounded perturbation in the nonconformity scores of the afected units. We denote a bound on the perturbation in the score of unit j caused by changing the treatment of unit I by the scalar $\Delta _ { j } ^ { I }$ , which is formally defined in Appendix E. Under the technical assumptions stated therein, this bound yields a sharper correction that accounts for the magnitude of the score perturbation

$$
\rho _ { I , y } ^ { + } = \frac { \sum _ { j \in \mathcal { H } _ { I } } W _ { j } \mathbb { 1 } \left\{ V _ { I } ( y ) - \Delta _ { j } ^ { I } \leq V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \right\} } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } .\tag{33}
$$

Replacing (26) with (33) yields a stability-aware variant of IA-WCP, which we call IA-WCP+. As shown in Appendix E, IA-WCP+ preserves the coverage requirement (27) under the mentioned technical assumptions.

## 4 Experiments

We evaluate the proposed methods using a synthetic peer-interference model inspired by (Forastiere et al., 2021; Chin, 2019). Appendix H provides additional results for this setting, along with experiments in a wireless trafic-slicing problem (Foukas et al., 2017; Bao et al., 2017).

## 4.1 Setup

We consider a network of $N \ = \ 3 0 0$ units in which each unit i has covariate $X _ { i } = ( U _ { i } , G _ { i } )$ , where $U _ { i } \sim$ Unif[−1, 1] and $G _ { i } \sim$ Unif $\{ 1 , \ldots , G \}$ is a group label. Treatments are drawn independently according to $T _ { i } \mid X _ { i } \sim$ Bernoulli(σ $\left( U _ { i } / 2 \right) ,$ , and the neighborhood of unit i contains its same-group peers, $\mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ j \neq$ $i : G _ { j } = G _ { i } \}$ . The exposure is given by the fraction of treated units in the same group, i.e.,

$$
E _ { i } = \frac { \sum _ { j \in \mathcal { N } _ { i } ( X _ { 1 : N } ) } T _ { j } } { | \mathcal { N } _ { i } ( X _ { 1 : N } ) | } ,\tag{34}
$$

with the convention that $E _ { i } = 0 \mathrm { i f } \mathcal { N } _ { i } ( X _ { 1 : N } ) = \emptyset$ . The potential outcomes are

$$
\begin{array} { r l } & { Y _ { i } ( 0 ) = 0 , } \\ & { Y _ { i } ( 1 ) = \theta U _ { i } + \left[ \sigma ( \lambda ( E _ { i } - 0 . 5 ) ) - \sigma ( - 0 . 5 \lambda ) \right] + \varepsilon _ { i } , } \end{array}\tag{35}
$$

(36)

where $\sigma ( \cdot )$ denotes the sigmoid function and $\varepsilon _ { i } \sim$ $\mathcal { N } ( 0 , \sigma _ { \varepsilon } ^ { 2 } )$ This model captures a simple peer-efect mechanism in which each unit responds to the treatment rate in its group, with the response becoming stronger once peer adoption crosses 50%. Such a mechanism can represent settings with social reinforcement or congestion efects. Unless specified otherwise, we instantiate the model with $G = 1 6$ groups, linear coefficient $\theta = 0 . 2 5$ , sigmoid parameter $\lambda = 1 0$ , and noise standard deviation $\sigma _ { \varepsilon } = 0 . 0 5$

## 4.2 Benchmarks

We use the absolute-residual score and compare WCP (Lei and Cand\`es, 2021), ideal WCP based on the unobservable interventional p-value in (30), and the proposed IA-WCP and IA-WCP+. Ideal WCP serves as an oracle reference, and it is available here because the data-generating mechanism is known. We apply these calibration methods to three diferent base predictors. The first is the interference-free linear predictor $\hat { Y } ( 1 ) = \theta U$ , which returns the mean outcome in the absence of interference. We also consider two fitted linear predictors, a covariate-only predictor $\hat { Y } ( 1 ) = \hat { \theta } _ { U } U + \hat { \theta } _ { 0 }$ , and an exposure-aware predictor $\hat { Y } ( 1 ) = \hat { \theta } _ { E } E + \hat { \theta } _ { U } U + \hat { \theta } _ { 0 }$ . The two linear predictors are fitted on treated units from ten independent training networks.

![](images/4939fda5e754e693383ff9efbf1c37e5c9c74ac552956aa0018233c263e4afe7.jpg)

![](images/404a85a48d7a41b23f74e4d4630a3d9cca9b9d18a7de7a04fcafa708421695ee.jpg)  
IA-WCP IA-WCP+  
Figure 4: Empirical coverage and normalized prediction-set width of WCP, ideal WCP, IA-WCP, and IA-WCP+ as a function of the number of groups G. The target coverage is set to $1 - \alpha = 0 . 8$ . Results are averaged over 200 runs.

For IA-WCP+, the stability margin is given by

$$
\Delta _ { j } ^ { I } = \left( \frac { \lambda } { 4 } + | \hat { \theta } _ { E } | \right) \left| \tilde { E } _ { j } ^ { I } - E _ { j } \right| ,\tag{37}
$$

where $\widehat { \theta } _ { E }$ is set to zero for predictors that do not use the exposure level E.

## 4.3 Results

We first study the efect of neighborhood size by varying the number of groups G. As shown in Figure 4, smaller values of G produce larger neighborhoods and increase the fraction of calibration units afected by the intervention. In this regime, WCP undercovers with the miscoverage gap increasing as the number of groups G decreases. For example, at G = 4, its empirical coverage is approximately 0.75, below the target level 0.8. Across all values of G, ideal WCP and the proposed IA-WCP and IA-WCP+ attain or exceed the desired coverage level. However, because IA-WCP uses a worst-case correction, it becomes highly conservative for small values of G. In contrast, IA-WCP+ performance tracks that of ideal WCP and is substantially more eficient than IA-WCP.

We next investigate how the choice of base predictor afects coverage and eficiency. Figure 5 shows that replacing the interference-free linear predictor with predictors fitted on data generated under interference substantially reduces prediction-set width. It also reduces the empirical miscoverage gap of WCP, whose coverage approaches that of ideal WCP when the predictor accounts for the exposure level. It is important to highlight that this behavior is specific to the present experiment, as conventional WCP lacks a general coverage guarantee under interference. The proposed methods retain their validity across all three predictors. Moreover, IA-WCP+ is noticeably narrower than IA-WCP when using the interference-free linear predictor or covariate-only predictor. When exposure E is included in the predictor, the additional term $| \widehat { \theta } _ { E } |$ enlarges the stability margin (37), causing IA-WCP+ to nearly coincide with IA-WCP.

![](images/c9e3b500e0d9bc4bd05217ac4de536a6d2f213b27321873aea83fa6bc56dcbdd.jpg)

![](images/369979c59bced140c607735ad3259a18520ffe1fe6ddc307128e8e725a5e94b2.jpg)  
Figure 5: Empirical coverage and normalized prediction-set width of WCP, ideal WCP, IA-WCP, and IA-WCP+ when applied to the interference-free linear predictor (IF), the covariate-only predictor (Fitted X), and the exposure-aware predictor (Fitted (X,E)). The target coverage is set to $1 - \alpha \ : = \ : 0 . 8$ Results are averaged over 200 runs.

## 5 Conclusion

We studied conformal counterfactual prediction for individual potential outcomes and treatment efects under interference in the inductive and transductive settings. In either setting, a unit’s potential outcomes may depend on the covariates and treatment assignments of other units, creating a source of nonexchangeability that is not corrected by standard unitlevel propensity weighting and may invalidate existing conformal methods.

To address this challenge, we have developed two interference-adjusted calibration methods. The first, IA-WCP, retains finite-sample marginal coverage under interference by adjusting the observational conformal p-value using a worst-case bound on the discrepancy between observational and interventional cal ibration scores. Although this correction remains valid under arbitrary score perturbations, it can produce conservative prediction sets. When bounds on intervention-induced score perturbations are available, IA-WCP+ uses them to sharpen the correction and improve eficiency while retaining the same coverage guarantee. Our experiments show that conventional WCP can remain close to nominal coverage when interference is weak, but it may undercover as interference becomes stronger. In contrast, IA-WCP and IA-WCP+ maintain the target coverage, with IA-WCP+ generally producing smaller prediction sets.

## References

Aronow, P. M. and Samii, C. (2017). Estimating average causal efects under general interference, with application to a social network experiment. The Annals of Applied Statistics, 11(4):1912–1947.

Bao, Y., Wu, H., and Liu, X. (2017). From predic tion to action: Improving user experience with datadriven resource allocation. IEEE Journal on Selected Areas in Communications, 35(5):1062–1075.

Cai, C., Zhang, X., and Airoldi, E. (2024). Independent-set design of experiments for estimating treatment and spillover efects under network interference. In International Conference on Learning Representations, pages 37362–37377.

Chen, Z., Guo, R., Ton, J.-F., and Liu, Y. (2024). Conformal counterfactual inference under hidden confounding. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pages 397–408.

Chin, A. (2019). Regression adjustments for estimating the global treatment efect in experiments with interference. Journal of Causal Inference, 7(2):20180026.

Clarkson, J. (2023). Distribution free prediction sets for node classification. In International conference on machine learning, pages 6268–6278. PMLR.

Farzaneh, A., Zecchin, M., and Simeone, O. (2025). Synthetic counterfactual labels for eficient conformal counterfactual inference. arXiv preprint arXiv:2509.04112.

Forastiere, L., Airoldi, E. M., and Mealli, F. (2021). Identification and estimation of treatment and interference efects in observational studies on networks. Journal of the American Statistical Association, 116(534):901–918.

Forastiere, L., Mealli, F., Wu, A., and Airoldi, E. M. (2022). Estimating causal efects under network interference with Bayesian generalized propensity scores. Journal of Machine Learning Research, 23(289):1–61.

Foukas, X., Patounas, G., Elmokashfi, A., and Marina, M. K. (2017). Network slicing in 5G: Survey and challenges. IEEE communications magazine, 55(5):94–100.

Gordon, B. R., Zettelmeyer, F., Bhargava, N., and Chapsky, D. (2019). A comparison of approaches to advertising measurement: Evidence from big field experiments at facebook. Marketing Science, 38(2):193–225.

H. Zargarbashi, S., Antonelli, S., and Bojchevski, A. (2023). Conformal prediction sets for graph neural networks. In Proceedings of the 40th International Conference on Machine Learning, pages 12292–12318. PMLR.

H. Zargarbashi, S. and Bojchevski, A. (2024). Conformal inductive graph neural networks. In International Conference on Learning Representations, volume 2024, pages 55175–55197.

Hou, Q., Park, S., Zecchin, M., Cai, Y., Yu, G., and Simeone, O. (2025). What if we had used a diferent app? Reliable counterfactual KPI analysis in wireless systems. IEEE Transactions on Cognitive Communications and Networking, 11(5):3529–3543.

Huang, K., Jin, Y., Cand\`es, E., and Leskovec, J. (2023a). Uncertainty quantification over graph with conformalized graph neural networks. Advances in Neural Information Processing Systems, 36:26699– 26721.

Huang, Q., Ma, J., Li, J., Guo, R., Sun, H., and Chang, Y. (2023b). Modeling interference for individual treatment efect estimation from networked observational data. ACM Transactions on Knowledge Discovery from Data, 18(3):1–21.

Hudgens, M. G. and Halloran, M. E. (2008). Toward causal inference with interference. Journal of the american statistical association, 103(482):832–842.

Jin, Y., Ren, Z., and Cand\`es, E. J. (2023). Sensitivity analysis of individual treatment efects: A robust conformal inference approach. Proceedings of the National Academy of Sciences, 120(6):e2214889120.

Koenker, R. and Bassett Jr, G. (1978). Regression quantiles. Econometrica: journal of the Econometric Society, pages 33–50.

Lei, L. and Cand\`es, E. J. (2021). Conformal inference of counterfactuals and individual treatment effects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(5):911–938.

Ma, Y. and Tresp, V. (2021). Causal inference under networked interference and intervention policy enhancement. In International Conference on Artificial Intelligence and Statistics, pages 3700–3708. PMLR.

Qchohi, A., Cortes, J. M., and Zecchin, M. (2026). Confounding-valid conformal inference for counterfactual KPIs in wireless networks. arXiv preprint arXiv:2609.05073.

Rubin, D. B. (1974). Estimating causal efects of treatments in randomized and nonrandomized studies. Journal of educational Psychology, 66(5):688.

Rubin, D. B. (2005). Causal inference using potential outcomes: Design, modeling, decisions. Journal of the American statistical Association, 100(469):322– 331.

Shafer, G. and Vovk, V. (2008). A tutorial on conformal prediction. Journal of machine learning research, 9.

Strassen, V. (1965). The existence of probability measures with given marginals. The Annals of Mathematical Statistics, 36(2):423–439.

Tchetgen, E. J. T. and VanderWeele, T. J. (2012). On causal inference in the presence of interference. Statistical methods in medical research, 21(1):55–75.

Tibshirani, R. J., Foygel Barber, R., Cand\`es, E., and Ramdas, A. (2019). Conformal prediction under covariate shift. Advances in neural information processing systems, 32.

Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic learning in a random world. Springer.

Yin, M., Shi, C., Wang, Y., and Blei, D. M. (2024). Conformal sensitivity analysis for individual treatment efects. Journal of the American Statistical Association, 119(545):122–135.

Zhao, T., Kang, J., and Cheng, L. (2024). Conformalized link prediction on graph neural networks. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pages 4490–4499.

Zhao, Y., Zeng, D., Rush, A. J., and Kosorok, M. R. (2012). Estimating individualized treatment rules using outcome weighted learning. Journal of the American Statistical Association, 107(499):1106– 1118.

Zhou, X., Ma, J., and Cheng, L. (2025). Adaptive degree-based conformal prediction for individual treatment efect estimation on networked data. In 2025 IEEE International Conference on Big Data (BigData), pages 628–637.

# Conformal Individual Treatment Efect Estimation under Networked Interference: Supplementary Materials

## A Related Work

Conformal Inference for Treatment Efects. Conformal prediction (CP) is a calibration framework that transforms point predictions into prediction sets with finite-sample marginal coverage guarantees, provided that the calibration and test data are exchangeable (Shafer and Vovk, 2008). In counterfactual inference, however, exchangeability is typically violated because outcomes under a given treatment are observed only for units assigned to that treatment, inducing covariate shift between the calibration and test distributions. WCP addresses this shift by reweighting the calibration data (Tibshirani et al., 2019). Under the Stable Unit Treatment Value Assumption (SUTVA), unconfoundedness, and overlap, WCP constructs valid prediction sets for counterfactua outcomes and individual treatment efects (ITEs) under covariate-dependent treatment assignment (Lei and Cand\`es, 2021).

Subsequent work has relaxed these assumptions in several directions. Sensitivity analysis has been introduced to address potential violations of unconfoundedness (Jin et al., 2023; Yin et al., 2024); interventional data have been leveraged to mitigate hidden confounding (Chen et al., 2024; Qchohi et al., 2026); and synthetic-powered prediction has been proposed to reduce the ineficiency caused by poor overlap (Farzaneh et al., 2025). However, all these methods fundamentally rely on the no-interference assumption embedded in SUTVA. In this work, we relax this assumption and construct valid conformal prediction sets for counterfactual outcomes and ITEs under network interference.

Causal inference under interference. The causal-inference literature relaxes the no-interference component of SUTVA by imposing structure on how one unit’s treatment may afect another unit’s outcome. Early work studies direct and spillover efects under partial interference, where interference is restricted to known groups (Hudgens and Halloran, 2008; Tchetgen and VanderWeele, 2012). Exposure mappings extend this framework to richer interference patterns by summarizing the treatment assignments of other units through an exposure variable, thereby enabling the identification and estimation of causal efects under randomized or observationa treatment assignment (Aronow and Samii, 2017; Forastiere et al., 2021). More recent machine-learning approaches exploit network structure to estimate heterogeneous treatment efects, exposure-response functions, and treatment policies in networked populations (Ma and Tresp, 2021; Huang et al., 2023b). These works have primarily focused on defining causal estimands and estimating population-level or heterogeneous efects under interference. In contrast, we focus on constructing model-agnostic, finite-sample prediction sets for individua potential outcomes and treatment efects.

Conformal Prediction on Networked Data. A growing body of literature adapts conformal prediction to graph-structured observations. For node-level prediction, existing methods use graph neighborhoods to reweight nonconformity scores (Clarkson, 2023; H. Zargarbashi et al., 2023), or learn topology-aware corrections that exploit permutation invariance to preserve marginal validity (Huang et al., 2023a). Related work addresses the distribution shift induced when new nodes or edges enter an inductive graph (H. Zargarbashi and Bojchevski, 2024), and extends conformal uncertainty quantification to link prediction (Zhao et al., 2024). In these settings, the graph structure is leveraged to learn node representations and improve calibration eficiency. However, the target remains an unobserved label or edge in the factual graph, rather than a counterfactual outcome under a treatment intervention. The closest work to ours is that of Zhou et al. (2025), which exploits a fixed transductive network to learn representations that account for hidden confounding, adapts conformal calibration to node degree, and estimates ITEs. Nevertheless, the framework in (Zhou et al., 2025) explicitly assumes no network interference, whereas in our setting, treatment spillover across neighboring nodes directly alters the target counterfactual outcomes.

## B Examples of Permutation-equivariant Neighborhood and Exposure Maps

We now list several permutation-equivariant neighborhood rules and deterministic exposure maps satisfying Assumption 2. These maps induce degenerate exposure kernels of the form

$$
P _ { E | X , \mathcal { L } } \left( \cdot \mid X _ { i } , \mathcal { L } _ { i } ( X _ { 1 : N } , T _ { 1 : N } ) \right) = \delta _ { e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) } ( \cdot ) ,\tag{38}
$$

where $e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } )$ denotes the exposure assigned to unit $i ,$ and $\delta _ { x } ( \cdot )$ denotes the Dirac measure centered at x.

Throughout this section we use the same permutation convention as in Assumption 2. For a permutation σ of $\{ 1 , \ldots , N \}$ , we denote the permuted treatment and covariate vectors by

$$
T _ { \sigma ( 1 : N ) } = ( T _ { \sigma ( 1 ) } , \ldots , T _ { \sigma ( N ) } ) , \qquad X _ { \sigma ( 1 : N ) } = ( X _ { \sigma ( 1 ) } , \ldots , X _ { \sigma ( N ) } ) .\tag{39}
$$

Thus, under the relabeled vectors $( T _ { \sigma ( 1 : N ) } , X _ { \sigma ( 1 : N ) } )$ , the unit at position i corresponds to the original unit $\sigma ( i )$ A deterministic exposure map is permutation-equivariant if, for every permutation σ and every $i = 1 , \ldots , N$

$$
e _ { i } ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } ) = e _ { \sigma ( i ) } ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } ) .\tag{40}
$$

1. Global treated-count exposure. Set $\mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ 1 , . . . , N \} \setminus \{ i \}$ and define

$$
e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) = \sum _ { j \neq i } T _ { j } .\tag{41}
$$

Then, for any permutation $\sigma _ { \mathrm { { i } } }$

$$
e _ { i } ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } ) = \sum _ { \ell \neq i } T _ { \sigma ( \ell ) } = \sum _ { j \neq \sigma ( i ) } T _ { j } = e _ { \sigma ( i ) } ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } ) .\tag{42}
$$

2. Fraction treated outside unit i. Again set $\mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ 1 , . . . , N \} \setminus \{ i \}$ and define

$$
e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) = \frac { 1 } { N - 1 } \sum _ { j \neq i } T _ { j } .\tag{43}
$$

Then, for any permutation $\sigma _ { \mathrm { { i } } }$

$$
e _ { i } ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } ) = \frac { 1 } { N - 1 } \sum _ { \ell \neq i } T _ { \sigma ( \ell ) } = \frac { 1 } { N - 1 } \sum _ { j \neq \sigma ( i ) } T _ { j } = e _ { \sigma ( i ) } ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } ) .\tag{44}
$$

3. k-nearest-neighbor treated-count exposure. Let $d : \mathcal { X } \times \mathcal { X } \to \mathbb { R } _ { + }$ be a distance, and let $\mathcal { N } _ { i } ^ { ( k ) } ( X _ { 1 : N } )$ denote the set of indices corresponding to the k nearest neighbors of unit i among $\{ 1 , \ldots , N \} \setminus \{ i \}$ . We assume either that ties occur with probability zero, or that they are resolved by a permutation-equivariant rule. Define

$$
e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) = \sum _ { j \in \mathcal { N } _ { i } ^ { ( k ) } ( X _ { 1 : N } ) } T _ { j } .\tag{45}
$$

Under the permutation convention above, the nearest-neighbor set satisfies

$$
\mathcal { N } _ { i } ^ { ( k ) } ( X _ { \sigma ( 1 : N ) } ) = \{ \sigma ^ { - 1 } ( j ) : j \in \mathcal { N } _ { \sigma ( i ) } ^ { ( k ) } ( X _ { 1 : N } ) \} .\tag{46}
$$

Therefore,

$$
e _ { i } ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } ) = \sum _ { \ell \in \mathcal { N } _ { i } ^ { ( k ) } ( X _ { \sigma ( 1 : N ) } ) } T _ { \sigma ( \ell ) } = \sum _ { j \in \mathcal { N } _ { \sigma ( i ) } ^ { ( k ) } ( X _ { 1 : N } ) } T _ { j } = e _ { \sigma ( i ) } ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } ) .\tag{47}
$$

4. Radius-based treated-count exposure. Let $d : \mathcal { X } \times \mathcal { X } \to \mathbb { R } _ { + }$ be a distance and fix $r > 0$ . Set

$$
\begin{array} { r } { \mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ j \neq i : d ( X _ { i } , X _ { j } ) \leq r \} , } \end{array}\tag{48}
$$

and define

$$
e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) = \sum _ { j \in \mathcal { N } _ { i } ( X _ { 1 : N } ) } T _ { j } .\tag{49}
$$

Then, for any permutation $\sigma ,$

$$
\begin{array} { l } { \displaystyle { e _ { i } \big ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } \big ) = \sum _ { \ell \neq i } T _ { \sigma ( \ell ) } \mathbb { 1 } \big \{ d \big ( X _ { \sigma ( i ) } , X _ { \sigma ( \ell ) } \big ) \leq r \big \} = \sum _ { j \neq \sigma ( i ) } T _ { j } \mathbb { 1 } \{ d \big ( X _ { \sigma ( i ) } , X _ { j } \big ) \leq r \} } } \\ { \displaystyle { = e _ { \sigma ( i ) } \big ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } \big ) . } } \end{array}\tag{50}
$$

5. Weighted exposure. Let $a : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ be a measurable weight function. Set $\mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ 1 , . . . , N \} \backslash \cdot$ {i} and define

$$
e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) = \sum _ { j \neq i } a ( X _ { i } , X _ { j } ) T _ { j } .\tag{51}
$$

Then, for any permutation $\sigma _ { \mathrm { { i } } }$

$$
e _ { i } ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } ) = \sum _ { \ell \neq i } a ( X _ { \sigma ( i ) } , X _ { \sigma ( \ell ) } ) T _ { \sigma ( \ell ) } = \sum _ { j \neq \sigma ( i ) } a ( X _ { \sigma ( i ) } , X _ { j } ) T _ { j } = e _ { \sigma ( i ) } ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } ) .\tag{52}
$$

6. Threshold exposure. Let $m \in \{ 1 , \ldots , N - 1 \}$ , set $\mathcal { N } _ { i } ( X _ { 1 : N } ) = \{ 1 , . . . , N \} \setminus \{ i \}$ , and define

$$
e _ { i } ( X _ { 1 : N } , T _ { 1 : N } ^ { - i } ) = 1 \left\{ \sum _ { j \neq i } T _ { j } \geq m \right\} .\tag{53}
$$

Since the treated count outside unit i is permutation-equivariant,

$$
e _ { i } ( X _ { \sigma ( 1 : N ) } , T _ { \sigma ( 1 : N ) } ^ { - i } ) = \mathbb { 1 } \left\{ \sum _ { \ell \neq i } T _ { \sigma ( \ell ) } \geq m \right\} = \mathbb { 1 } \left\{ \sum _ { j \neq \sigma ( i ) } T _ { j } \geq m \right\} = e _ { \sigma ( i ) } ( X _ { 1 : N } , T _ { 1 : N } ^ { - \sigma ( i ) } ) .\tag{54}
$$

## C Proof of Transductive IA-WCP Validity

We prove the result for the case in which the test unit I satisfies $T _ { I } = 0$ , so that the missing potential outcome is $Y _ { I } ( 1 )$ ). The case $T _ { I } = 1$ is analogous. All probabilities in the remainder of this proof are conditional on $T _ { I } = 0$

We first establish weighted exchangeability of the interventional sample and the validity of the ideal WCP procedure under interference. We then compare the observational and interventional p-values under a joint coupling and conclude the validity proof of IA-WCP.

## C.1 Validity of the ideal WCP procedure

Let $\tilde { T } _ { 1 : N } ^ { I }$ denote the treatment vector obtained by setting the treatment of unit I to one while leaving all other treatment assignments in $T _ { 1 : N }$ unchanged, i.e.,

$$
\tilde { T } _ { I } ^ { I } = 1 , \qquad \tilde { T } _ { j } ^ { I } = T _ { j } , \quad j \ne I .\tag{55}
$$

Under this intervention, let $\tilde { E } _ { j } ^ { I }$ denote the exposure of unit $j ,$ , and let $\tilde { Y } _ { j } ^ { I } ( 1 )$ denote the corresponding treated potential outcome. Define

$$
\tilde { Z } _ { j } = ( X _ { j } , \tilde { E } _ { j } ^ { I } , \tilde { Y } _ { j } ^ { I } ( 1 ) ) , \qquad j = 1 , \dots , N .\tag{56}
$$

We now show that, for any treatment vector $T _ { 1 : N }$ such that $T _ { I } = 0$ , the augmented interventional sample

$$
\tilde { \mathcal { D } } _ { 1 } = ( \tilde { Z } _ { j } ) _ { j \in \mathcal { T } _ { 1 } \cup \{ I \} } ,\tag{57}
$$

is weighted exchangeable.

Lemma 2 (Weighted exchangeability of the interventional sample). Fix an index I and a treated calibration set ${ \mathcal { T } } _ { 1 } \subseteq \left\{ 1 , \ldots , N \right\} \setminus \left\{ I \right\}$ , and let $S = \mathcal { T } _ { 1 } \cup \{ I \}$ . Work conditionally on the observational assignment event

$$
A _ { \mathcal { T } _ { 1 } , I } = \{ T _ { j } = 1 \ f o r \ j \in \mathcal { T } _ { 1 } , T _ { I } = 0 , T _ { j } = 0 \ f o r \ j \not \in \mathcal { S } \} .\tag{58}
$$

The intervention is applied after conditioning on this observational assignment event.

Under Assumptions 1–5, the conditional density of $( \tilde { Z } _ { j } ) _ { j \in \mathcal { S } }$ admits the decomposition

$$
p \big ( ( \tilde { Z } _ { j } ) _ { j \in \mathcal { S } } \mid A _ { { \mathcal { T } } _ { 1 } , I } \big ) = c w ( X _ { I } ) g \big ( ( \tilde { Z } _ { j } ) _ { j \in \mathcal { S } } \big ) ,\tag{59}
$$

where $c > 0$

$$
w ( x ) = \frac { 1 - \pi ( x ) } { \pi ( x ) } ,\tag{60}
$$

and $g$ is invariant under permutations of the arguments indexed by $s$

Proof. By Assumption 1,

$$
\operatorname* { P r } ( A _ { \mathcal { T } _ { 1 } , I } \mid X _ { 1 : N } ) = \prod _ { j \in \mathcal { T } _ { 1 } } \pi ( X _ { j } ) \big ( 1 - \pi ( X _ { I } ) \big ) \prod _ { j \not \in \mathcal { S } } \big ( 1 - \pi ( X _ { j } ) \big ) .\tag{61}
$$

Therefore,

$$
\begin{array} { l } { \displaystyle p ( X _ { 1 : N } \mid A _ { \mathcal { T } _ { 1 } , I } ) = \frac { \mathrm { P r } ( A _ { \mathcal { T } _ { 1 } , I } \mid X _ { 1 : N } ) p _ { X } ^ { \otimes N } ( X _ { 1 : N } ) } { \mathrm { P r } ( A _ { \mathcal { T } _ { 1 } , I } ) } } \\ { \displaystyle \qquad = c _ { 0 } \prod _ { j \in \mathcal { T } _ { 1 } } \pi ( X _ { j } ) p _ { X } ( X _ { j } ) \big ( 1 - \pi ( X _ { I } ) \big ) p _ { X } ( X _ { I } ) \prod _ { j \not \in \mathcal { S } } ( 1 - \pi ( X _ { j } ) \big ) p _ { X } ( X _ { j } ) , } \end{array}\tag{62}
$$

where $c _ { 0 } = 1 / \operatorname* { P r } ( A _ { \mathcal { T } _ { 1 } , I } )$ . Multiplying and dividing the factor associated with I by $\pi ( X _ { I } )$ gives

$$
p ( X _ { 1 : N } \mid A _ { { \bar { \cal U } } _ { 1 } , I } ) = c _ { 0 } w ( X _ { I } ) \prod _ { j \in { \cal S } } \pi ( X _ { j } ) p _ { X } ( X _ { j } ) \prod _ { j \not \in { \cal S } } \bigl ( 1 - \pi ( X _ { j } ) \bigr ) p _ { X } ( X _ { j } ) .\tag{63}
$$

Integrating out the covariates indexed by $S ^ { c }$ contributes only a constant independent of $( X _ { j } ) _ { j \in { \cal S } }$ . The resulting marginal density is therefore proportional to

$$
w ( X _ { I } ) \prod _ { j \in { \cal S } } \pi ( X _ { j } ) p _ { X } ( X _ { j } ) ,\tag{64}
$$

which proves the weighted exchangeability of covariates conditioned on the event $A _ { \mathcal { T } _ { 1 } , I } .$

We now define the interventional variables using the intervened treatment vector $\tilde { T } ^ { I }$ . Conditionally on $( X _ { 1 : N } , \tilde { T } ^ { I } )$ ， Assumption 3 gives

$$
p ( \tilde { E } _ { 1 : N } ^ { I } \mid X _ { 1 : N } , A _ { { \bar { \cal L } } _ { 1 } , I } , \mathrm { d o } ( T _ { I } = 1 ) ) = \prod _ { j = 1 } ^ { N } p _ { E | X , { \mathcal { L } } } \left( \tilde { E } _ { j } ^ { I } \mid X _ { j } , { \mathcal { L } } _ { j } ( X _ { 1 : N } , \tilde { T } _ { 1 : N } ^ { I } ) \right) .\tag{65}
$$

Similarly, by Assumption 4,

$$
p \left( \tilde { Y } _ { 1 : N } ^ { I } ( 1 ) \mid X _ { 1 : N } , \tilde { E } _ { 1 : N } ^ { I } , A _ { \bar { Z } _ { 1 } , I } , \mathrm { d o } ( T _ { I } = 1 ) \right) = \prod _ { j = 1 } ^ { N } p _ { Y ( 1 ) | X , E } \left( \tilde { Y } _ { j } ^ { I } ( 1 ) \mid X _ { j } , \tilde { E } _ { j } ^ { I } \right) .\tag{66}
$$

Combining (63), (65), and (66), and then marginalizing over all variables indexed by $S ^ { c } = \left\{ 1 , \ldots , N \right\} \setminus S .$ yields

$$
p \big ( ( \tilde { Z } _ { j } ) _ { j \in \mathcal { S } } \mid A _ { { \mathbb { Z } } _ { 1 } , I } \big ) = c w ( X _ { I } ) g \big ( ( \tilde { Z } _ { j } ) _ { j \in \mathcal { S } } \big ) ,\tag{67}
$$

for a normalizing constant $c > 0$ , where g collects all remaining factors after the extraction of $w ( X _ { I } )$

It remains to show that $g$ is invariant under permutations of its arguments. Let $\sigma$ be any permutation of the indices in $s ,$ extended to $\{ 1 , \ldots , N \}$ by setting $\sigma ( k ) = k$ for $k \not \in S .$ The factors

$$
\prod _ { j \in \cal { S } } \pi ( X _ { j } ) p _ { X } ( X _ { j } )\tag{68}
$$

and

$$
\prod _ { j \in S } p _ { Y ( 1 ) | X , E } \left( \tilde { Y } _ { j } ^ { I } ( 1 ) \mid X _ { j } , \tilde { E } _ { j } ^ { I } \right)\tag{69}
$$

are invariant under permutations of the triples

$$
( X _ { j } , \tilde { E } _ { j } ^ { I } , \tilde { Y } _ { j } ^ { I } ( 1 ) ) , \qquad j \in \mathcal { S } .\tag{70}
$$

Finally, under $\tilde { T } ^ { I }$ , all units in $s$ have treatment equal to one. Hence a permutation of the labels in $s$ only relabels units with the same intervened treatment. By Assumption 2, the neighborhood rule is permutation equivariant and the exposure kernel is invariant to the ordering of the neighbors, so the joint exposure density is unchanged under the same simultaneous permutation of the covariates, exposures, and outcomes indexed by S. The variables indexed by $S ^ { c }$ are integrated ${ \mathrm { o u t } } ,$ and the integration measure is invariant under this relabeling.

Therefore $g$ is invariant under permutations of the arguments $( \tilde { Z } _ { j } ) _ { j \in \mathcal { S } }$ , proving the claimed decomposition.

By Lemma 2, conditionally on $A _ { \mathcal { T } _ { 1 } , I } { : }$ , the augmented interventional sample is weighted exchangeable. Therefore, the standard weighted conformal validity argument (Tibshirani et al., 2019) gives

$$
\mathrm { P r } \left[ p _ { I , \tilde { Y } _ { I } ^ { I } ( 1 ) } ^ { \mathrm { i n t } } \leq \alpha \Big | A _ { \mathbb { Z } _ { 1 } , I } \right] \leq \alpha .\tag{71}
$$

This establishes conditional validity at the interventional target outcome $\widetilde { Y } _ { I } ^ { I } ( 1 )$ . Because $I \not \in { \mathcal { N } } _ { I } ( X _ { 1 : N } )$ , changing $T _ { I }$ does not alter the exposure law of the target unit. Under Assumptions 3 and 4, conditionally on $A _ { \mathcal { T } _ { 1 } , I } $ , we may therefore couple the target variables so that

$$
( \widetilde E _ { I } ^ { I } , \widetilde Y _ { I } ^ { I } ( 1 ) ) = ( E _ { I } , Y _ { I } ( 1 ) ) \qquad \mathrm { a . s . } ,\tag{72}
$$

while preserving their joint law with the interventional calibration scores. Consequently,

$$
\mathrm { P r } \left[ p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } \leq \alpha \Big | A _ { { \mathbb { Z } } _ { 1 } , I } \right] \leq \alpha .\tag{73}
$$

For each fixed I, the events $A _ { \mathcal { T } _ { 1 } , I }$ , indexed by ${ \mathcal { T } } _ { 1 } \subseteq \left\{ 1 , \ldots , N \right\} \setminus \left\{ I \right\}$ , form a partition of $\{ T _ { I } = 0 \}$ . Applying the law of total probability over this partition and then averaging over the selected index I proves Lemma 1.

## C.2 Coupling comparison

For this proof, we consider the exposure-law afected set

$$
\mathcal { H } _ { I } ^ { \mathrm { e x } } = \Big \{ j \in \mathcal { Z } _ { 1 } : P _ { E | X , \mathcal { L } } \left( \cdot \mid X _ { j } , \mathcal { L } _ { j } \big ( X _ { 1 : N } , T _ { 1 : N } \big ) \right) \neq P _ { E | X , \mathcal { L } } \Big ( \cdot \mid X _ { j } , \mathcal { L } _ { j } \big ( X _ { 1 : N } , \tilde { T } _ { 1 : N } ^ { I } \big ) \Big ) \Big \} ,\tag{74}
$$

and write $\mathcal { U } _ { I } ^ { \mathrm { e x } } = \mathcal { T } _ { 1 } \setminus \mathcal { H } _ { I } ^ { \mathrm { e x } }$ . Note that $\mathcal { H } _ { I } ^ { \mathrm { e x } } \subseteq \mathcal { H } _ { I }$ because the conditional exposure law of unit $j$ is unchanged when $I \not \in { \mathcal { N } } _ { j } ( X _ { 1 : N } )$ . Thus, proving the comparison first with the set $\mathcal { H } _ { I } ^ { \mathrm { e x } }$ yields the main-text bound by enlarging the index set of the nonnegative correction sum from $\mathcal { H } _ { I } ^ { \mathrm { e x } }$ to $\mathcal { H } _ { I }$

Lemma 3. Under Assumptions 3 and $^ { 4 , }$ conditionally on $T _ { I } = 0$ , the observational conformal p-value in (22) and the interventional conformal p-value in (30) admit a coupling such that, simultaneously for every $y \in \mathcal { V }$

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { \mathrm { e x } } \qquad a . s . ,\tag{75}
$$

where

$$
\rho _ { I , y } ^ { \mathrm { e x } } = \frac { \sum _ { j \in \mathcal { H } _ { I } ^ { \mathrm { e x } } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \} } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } .\tag{76}
$$

Proof. We construct the coupling conditionally on $( X _ { 1 : N } , T _ { 1 : N } , I )$ , with $T _ { I } = 0$ . Since the exposure of a unit does not depend on its own treatment,

$$
\begin{array} { r } { E _ { I } \mid X _ { 1 : N } , T _ { 1 : N } \stackrel { d } { = } \tilde { E } _ { I } ^ { I } \mid X _ { 1 : N } , \tilde { T } _ { 1 : N } ^ { I } , } \end{array}\tag{77}
$$

where $\circeq$ denotes equality in distribution. Moreover, for every $j \in \mathcal { U } _ { I } ^ { \mathrm { e x } }$ , the definition of $\mathcal { U } _ { I } ^ { \mathrm { e x } }$ gives

$$
\begin{array} { r } { E _ { j } \mid X _ { 1 : N } , T _ { 1 : N } \stackrel { d } { = } \tilde { E } _ { j } ^ { I } \mid X _ { 1 : N } , \tilde { T } _ { 1 : N } ^ { I } . } \end{array}\tag{78}
$$

We may therefore couple these exposures so that

$$
E _ { I } = \tilde { E } _ { I } ^ { I } , \qquad \mathrm { a n d } \quad E _ { j } = \tilde { E } _ { j } ^ { I } , \qquad j \in \mathcal { U } _ { I } ^ { \mathrm { e x } } ,\tag{79}
$$

almost surely. Conditional on these coupled exposures, the corresponding treated potential outcomes have the same conditional distribution $P _ { Y ( 1 ) | X , E }$ . Hence, they may be coupled so that

$$
Y _ { I } ( 1 ) = { \tilde { Y } } _ { I } ^ { I } ( 1 ) , \quad { \mathrm { ~ a n d ~ } } \quad Y _ { j } ( 1 ) = { \tilde { Y } } _ { j } ^ { I } ( 1 ) , \qquad j \in \mathcal { U } _ { I } ^ { \mathrm { e x } } ,\tag{80}
$$

almost surely. For every $j \in \mathcal { H } _ { I } ^ { \mathrm { e x } } ,$ , choose any coupling with the correct observational and interventional marginals.

By Assumptions 3 and 4, the unit-specific exposure and outcome variables are conditionally independent across units. These couplings can therefore be combined into a product coupling that preserves the observational and interventional score-vector marginals and their joint laws with the corresponding target potential outcome. Under this coupling, for $y \in \mathcal { V }$ ，

$$
V _ { I } ^ { \mathrm { i n t } } ( y ) = V _ { I } ( y ) , \quad \mathrm { ~ a n d ~ } \quad V _ { j } ^ { \mathrm { i n t } } = V _ { j } ^ { \mathrm { o b s } } , \qquad j \in \mathcal { U } _ { I } ^ { \mathrm { e x } } ,\tag{81}
$$

almost surely. The conformal weights are also unchanged because the intervention does not alter the covariates. Consequently, the unafected units cancel from the diference between the two p-values and

$$
p _ { I , y } ^ { \mathrm { i n t } } - p _ { I , y } ^ { \mathrm { o b s } } = \frac { \sum _ { j \in \mathcal { H } _ { I } ^ { \mathrm { e x } } } W _ { j } \left[ \mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { I } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { I } ( y ) \} \right] } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } .\tag{82}
$$

Note that for every $j \in \mathcal { H } _ { I } ^ { \mathrm { e x } }$

$$
\mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { I } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { I } ( y ) \} \leq \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \} ,\tag{83}
$$

and therefore,

$$
p _ { I , y } ^ { \mathrm { i n t } } - p _ { I , y } ^ { \mathrm { o b s } } \leq \frac { \sum _ { j \in \mathcal { H } _ { I } ^ { \mathrm { e x } } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \} } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } \leq \frac { \sum _ { j \in \mathcal { H } _ { I } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \} } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } = \rho _ { I , y } .\tag{84}
$$

Hence,

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { \mathrm { e x } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } \qquad \mathrm { a . s . }\tag{85}
$$

Since the coupling does not depend on $y ,$ this inequality holds simultaneously for every $y \in \mathcal { V }$

## C.3 Coverage of IA-WCP

Use the joint coupling constructed in Lemma 3. It preserves the joint law of the interventional scores and target outcome and satisfies

$$
Y _ { I } ( 1 ) = \tilde { Y } _ { I } ^ { I } ( 1 ) , \qquad p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } = p _ { I , \tilde { Y } _ { I } ^ { I } ( 1 ) } ^ { \mathrm { i n t } }\tag{86}
$$

almost surely.

For each fixed I, the events $A _ { \mathcal { T } _ { 1 } , I }$ , indexed by the possible treated sets ${ \mathcal { T } } _ { 1 } \subseteq \{ 1 , \dots , N \} \setminus \{ I \}$ , partition the event $\{ T _ { I } = 0 \}$ . Averaging (71) over these events and over the randomly selected index I, and using the target equality under the coupling, yields

$$
\mathrm { P r } \left[ p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } \leq \alpha \Big | T _ { I } = 0 \right] \leq \alpha .\tag{87}
$$

By Lemma 3, under the same coupling, almost surely and simultaneously for every $y \in \mathcal { V }$

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } .\tag{88}
$$

Since overlap ensures $p _ { I , y } ^ { \mathrm { o b s } } > 0$ , rejection is impossible when $\rho _ { I , y } \geq \alpha$ . Otherwise, rejection implies $p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } \leq \alpha$ Therefore,

$$
\left\{ p _ { I , y } ^ { \mathrm { o b s } } \le \left( \alpha - \rho _ { I , y } \right) + \right\} \subseteq \left\{ p _ { I , y } ^ { \mathrm { i n t } } \le \alpha \right\} .\tag{89}
$$

Applying this inclusion at $y = Y _ { I } ( 1 )$ gives

$$
\operatorname* { P r } \left[ p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } \le ( \alpha - \rho _ { I , Y _ { I } ( 1 ) } ) _ { + } \Big | T _ { I } = 0 \right] \le \operatorname* { P r } \left[ p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } \le \alpha \Big | T _ { I } = 0 \right] \le \alpha .\tag{90}
$$

By definition of the IA-WCP prediction set in (25),

$$
Y _ { I } ( 1 ) \notin \Gamma _ { I } ^ { \mathrm { I A - W C P } } \quad \iff \quad p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } \leq ( \alpha - \rho _ { I , Y _ { I } ( 1 ) } ) _ { + } .\tag{91}
$$

Hence,

$$
\operatorname* { P r } \left[ Y _ { I } ( 1 ) \notin \Gamma _ { I } ^ { \mathrm { I A - W C P } } | T _ { I } = 0 \right] \le \alpha .\tag{92}
$$

This proves Theorem 1.

## D Coverage Tightness of the Correction Factor

We give an example in which the IA-WCP set in (25) is tight in the sense that it achieves the same marginal coverage as the ideal WCP set in (29). Fix a population size $N \geq 2$ and set the miscoverage level to $\alpha = \alpha _ { N } =$ $1 / N$ . For a constant propensity $\pi \in ( 0 , 1 )$ , covariates and treatments are generated as

$$
X _ { i } \stackrel { \mathrm { i i d } } { \sim } \mathrm { U n i f } ( 0 , 1 ) ,
$$

$$
T _ { i } \stackrel { \mathrm { i i d } } { \sim } \mathrm { B e r n o u l l i } ( \pi ) , \qquad i \in \{ 1 , \ldots , N \} .\tag{93}
$$

Define the permutation-equivariant neighborhood rule

$$
{ \mathcal { N } } _ { i } ( X _ { 1 : N } ) = \{ \ell \neq i : X _ { \ell } < X _ { i } \} .\tag{94}
$$

We consider an exposure mechanism in which the exposure $E _ { i }$ of unit i is determined by the number of control units in this neighborhood, i.e.,

$$
E _ { i } = \sum _ { \ell \neq i } ( 1 - T _ { \ell } ) \mathbb { 1 } \{ X _ { \ell } < X _ { i } \} .\tag{95}
$$

The potential outcomes are defined as

$$
Y _ { i } ( 1 ) = \left\{ { \begin{array} { l l } { X _ { i } , } & { E _ { i } = 0 , } \\ { - 1 , } & { E _ { i } \geq 1 , } \end{array} } \right.
$$

$$
Y _ { i } ( 0 ) = 0 .\tag{96}
$$

The deterministic exposure and outcome kernels also satisfy the conditional-independence requirements in $\mathrm { A s } -$ sumptions 3 and 4. Take $\mathcal { V } = [ - 1 , 1 ]$ and use the nonconformity score

$$
S ( x , e , y ) = y .\tag{97}
$$

We use the ideal WCP set $\tilde { \Gamma } _ { I } ^ { \mathrm { W C P } }$ in (29) and the IA-WCP set $\Gamma _ { I } ^ { \mathrm { I A - W C P } }$ in (25), both evaluated at miscoverage level $\alpha _ { N }$

Proposition 1. Under the construction in (93)–(97), conditionally on $T _ { I } = 0$ , the adjusted and interventional prediction sets have the same marginal coverage,

$$
\operatorname* { P r } \left[ Y _ { I } ( 1 ) \in \tilde { \Gamma } _ { I } ^ { \mathrm { W C P } } \mid T _ { I } = 0 \right] = \operatorname* { P r } \left[ Y _ { I } ( 1 ) \in \Gamma _ { I } ^ { \mathrm { I A - W C P } } \mid T _ { I } = 0 \right] = 1 - \frac { \pi ^ { N - 1 } } { N } \geq 1 - \alpha _ { N } .\tag{98}
$$

Proof. Let I be drawn uniformly from $\{ 1 , \ldots , N \}$ and condition on $T _ { I } = 0$ . Let

$$
A _ { I } = \{ T _ { j } = 1 \mathrm { f o r } \mathrm { e v e r y } j \neq I \}\tag{99}
$$

be the event that the target is the only control. Conditional on $T _ { I } = 0$

$$
\mathrm { P r } ( \mathcal { A } _ { I } \mid T _ { I } = 0 ) = \pi ^ { N - 1 } .\tag{100}
$$

On $A _ { I } ,$ the target exposure is zero, so its true treated score is $V _ { I } ( Y _ { I } ( 1 ) ) = X _ { I }$ . If a treated calibration unit $j$ satisfies $X _ { j } ~ < ~ X _ { I }$ , then its observational and interventional scores both equal $X _ { j } ~ < ~ X _ { I }$ . If $X _ { j } \ > \ X _ { I }$ , its observational exposure is one because of the control target, whereas its interventional exposure is zero after setting $T _ { I } = 1$ . Therefore,

$$
V _ { j } ^ { \mathrm { { o b s } } } = - 1 < X _ { I } < X _ { j } = V _ { j } ^ { \mathrm { { i n t } } } .\tag{101}
$$

Consequently, on $\boldsymbol { \mathcal { A } } _ { I }$

$$
\mathcal { H } _ { I } = \mathcal { H } _ { I } ^ { \mathrm { e x } } = \{ j \neq I : X _ { j } > X _ { I } \} .\tag{102}
$$

Writing $H = | \mathcal { H } _ { I } |$ and using the equality of the weights gives

$$
p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } = \frac { 1 } { N } , \qquad \rho _ { I , Y _ { I } ( 1 ) } = \frac { H } { N } , \qquad p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } = \frac { H + 1 } { N } .\tag{103}
$$

Thus $p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } + \rho _ { I , Y _ { I } ( 1 ) } = p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } }$ at the true outcome. Since $p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } > 0$ , the adjusted acceptance condition is equivalent to

$$
p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } + \rho _ { I , Y _ { I } ( 1 ) } > \alpha _ { N } .\tag{104}
$$

Hence, the ideal and corrected observational procedures make the same coverage decision on $\boldsymbol { \mathcal { A } } _ { I }$ . Moreover, $H + 1$ is the descending rank of $X _ { I }$ among N i.i.d. continuous observations, and is therefore uniform on $\{ 1 , \ldots , N \}$ At $\alpha _ { N } = 1 / N$ , both procedures miscover on $\boldsymbol { \mathcal { A } } _ { I }$ precisely when $H = 0$ , which has conditional probability $1 / N$

On $\mathcal { A } _ { I } ^ { c }$ , there is at least one control in addition to I, so the number of treated calibration units is at most $N - 2 .$ i.e., $| \mathcal { T } _ { 1 } | \le N - 2$ . Both the interventional and observational p-values are bounded below as

$$
p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { o b s } } , p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } \geq \frac { 1 } { 1 + | \mathcal { Z } _ { 1 } | } \geq \frac { 1 } { N - 1 } > \frac { 1 } { N } = \alpha _ { N } .\tag{105}
$$

The ideal and corrected observational procedures therefore both cover on $\mathcal { A } _ { I } ^ { c }$ . Combining this with (100) proves (98). □

The empirical coverage of the ideal WCP, WCP, and the proposed IA-WCP is illustrated in Figure 6. As established in Proposition 1, the ideal and corrected procedures have identical empirical coverage, while the unadjusted observational procedure can substantially undercover for smaller values of $\begin{array} { r } { \bar { N } . } \end{array}$ , although the discrepancy decreases as N grows.

## E Transductive IA-WCP+

As mentioned in Section 3.2, IA-WCP+ leverages bounds on intervention-induced changes in the afected calibration scores to produce a sharper correction than the worst-case correction of IA-WCP. In this section, we formalize the stability condition under which the refined correction $\rho _ { I , y } ^ { + }$ introduced in (33) preserves marginal coverage. Throughout, we use the notation of Section 3, condition on $\overset { \vartriangle } { \boldsymbol { T } _ { I } } = 0$ , and impose the following stability assumption on the afected calibration scores.

![](images/bac16b89801b2ad18bc99b20009b4f3ab4baae4699e81089e63db65a8e0d8e5a.jpg)  
Figure 6: Empirical marginal coverage of the ideal interventional, corrected observational, and unadjusted observational procedures under the setting of Proposition 1, with $\pi = 0 . 9$ and $\alpha _ { N } = 1 / N$ . Error bars represent 95% Monte Carlo confidence intervals. The ideal and corrected procedures have identical empirical coverage, in agreement with Proposition 1. The unadjusted observational procedure can substantially undercover for smaller values of N, although the discrepancy decreases as N grows.

Assumption 6. For every afected calibration unit $j \in \mathcal { H } _ { I }$ , there exists a finite, nonnegative, observable margin $\Delta _ { j } ^ { I }$ that is a measurable function of $( X _ { 1 : N } , T _ { 1 : N } , I , E _ { j } , Y _ { j } ( 1 ) )$ ), but does not depend on the random exposures or outcomes of other units, such that for every $t \in \mathbb { R }$

$$
\operatorname* { P r } \left[ V _ { j } ^ { \mathrm { i n t } } > t \mid X _ { 1 : N } , T _ { 1 : N } , I \right] \le \operatorname* { P r } \left[ V _ { j } ^ { \mathrm { o b s } } + \Delta _ { j } ^ { I } > t \mid X _ { 1 : N } , T _ { 1 : N } , I \right] .\tag{106}
$$

Assumption 6 states that, conditionally on the covariates, treatment assignments, and target index, the interventional score of each afected unit is stochastically dominated by its observational score shifted upward by $\Delta _ { i } ^ { I }$ . As shown in Section E.1, suficient conditions for this assumption include Lipschitz continuity of the nonconformity score and almost-sure bounds on the intervention-induced changes in exposures and potential outcomes.

Using margins satisfying (106) in the correction factor (33), IA-WCP+ defines the prediction set as

$$
\Gamma _ { I } ^ { \mathrm { I A - W C P + } } = \left\{ y \in \mathcal { V } : p _ { I , y } ^ { \mathrm { o b s } } > ( \alpha - \rho _ { I , y } ^ { + } ) _ { + } \right\} , \qquad \alpha \in ( 0 , 1 ) .\tag{107}
$$

Since $0 \le \rho _ { I , y } ^ { + } \le \rho _ { I , y }$ for every y, comparing the acceptance thresholds in (107) and (25) gives

$$
\Gamma _ { I } ^ { \mathrm { I A - W C P + } } \subseteq \Gamma _ { I } ^ { \mathrm { I A - W C P } } .\tag{108}
$$

Thus, IA-WCP+ returns a prediction set no larger than IA-WCP for the same observational sample. Under the assumptions of the following lemma, $p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { + }$ bounds the interventional p-value from above under a suitable coupling.

Lemma 4. Under Assumptions ${ \it 3 , \ 4 , }$ and 6, conditionally on $T _ { I } = 0 ;$ , the observational and interventional conformal p-values admit a coupling such that, simultaneously for every $y \in \mathcal { V }$

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { + } \qquad a . s .\tag{109}
$$

Proof. Work conditionally on $( X _ { 1 : N } , T _ { 1 : N } , I )$ with $T _ { I } ~ = ~ 0$ , and write $\mathcal { U } _ { I } = \mathcal { T } _ { 1 } \setminus \mathcal { H } _ { I }$ . As in Lemma 3, the observational and interventional scores can be coupled so that

$$
V _ { I } ^ { \mathrm { i n t } } ( y ) = V _ { I } ( y ) , \quad \mathrm { ~ a n d ~ } \quad V _ { j } ^ { \mathrm { o b s } } = V _ { j } ^ { \mathrm { i n t } } \qquad \mathrm { a . s . ~ f o r ~ e v e r y ~ } j \in \mathcal { U } _ { I } .\tag{110}
$$

For every $j \in \mathcal { H } _ { I }$ , Assumption 6 and the coupling characterization of stochastic order (Strassen, 1965) imply the existence of a coupling, preserving the conditional joint law of $( V _ { j } ^ { \mathrm { o b s } } , \Delta _ { j } ^ { I } )$ , such that

$$
V _ { j } ^ { \mathrm { i n t } } \leq V _ { j } ^ { \mathrm { o b s } } + \Delta _ { j } ^ { I } \qquad \mathrm { a . s . }\tag{111}
$$

Since, conditional on $( X _ { 1 : N } , T _ { 1 : N } , I )$ , each $\Delta _ { j } ^ { I }$ may depend only on $( E _ { j } , Y _ { j } ( 1 ) )$ and not on the exposures or outcomes of other units, the pair $( V _ { j } ^ { \mathrm { o b s } } , \Delta _ { j } ^ { I } )$ is a function only of the random variables $( E _ { j } , Y _ { j } ( 1 ) )$ . From Assumptions 3 and 4 it then follows that the pairs $( V _ { j } ^ { \mathrm { o b s } } , \Delta _ { j } ^ { I } )$ are conditionally independent across units. The unit couplings can therefore be combined into a product coupling under which (111) holds simultaneously for every $j \in \mathcal { H } _ { I }$

The interventional scores are conditionally independent for the same reason. The product coupling therefore preserves both score-vector marginals. We also retain the identical target coupling $( \tilde { E } _ { I } ^ { I } , \tilde { Y } _ { I } ^ { I } ( 1 ) ) \stackrel {  } { = } \stackrel {  } { ( } E _ { I } , Y _ { I } ( 1 ) )$ from Lemma 3, preserving the joint laws with the target potential outcome.

For every $j \in \mathcal { H } _ { I }$ , (111) implies

$$
\mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { I } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { I } ( y ) \} \leq \mathbb { 1 } \left\{ V _ { I } ( y ) - \Delta _ { j } ^ { I } \leq V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \right\} .\tag{112}
$$

In fact, the left-hand side can be positive only if

$$
V _ { j } ^ { \mathrm { o b s } } < V _ { I } ( y ) \leq V _ { j } ^ { \mathrm { i n t } } \leq V _ { j } ^ { \mathrm { o b s } } + \Delta _ { j } ^ { I } .\tag{113}
$$

Since the unafected scores coincide under the coupling,

$$
p _ { I , y } ^ { \mathrm { i n t } } - p _ { I , y } ^ { \mathrm { o b s } } = \frac { \sum _ { j \in \mathcal { H } _ { I } } W _ { j } \left[ \mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { I } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { I } ( y ) \} \right] } { W _ { I } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } \leq \rho _ { I , y } ^ { + } .\tag{114}
$$

Therefore,

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { + } \qquad \mathrm { a . s . }\tag{115}
$$

Since the coupling does not depend on y, the inequality holds simultaneously for every $y \in \mathcal { V }$

It follows that IA-WCP+ preserves marginal coverage, as formalized in the following theorem.

Theorem 2. Under Assumptions 1, 2, 3, 4, 5, and $\delta ,$ the prediction set in (107) satisfies

$$
\operatorname* { P r } \left[ Y _ { I } ( 1 ) \notin \Gamma _ { I } ^ { \mathrm { I A - W C P + } } \mid T _ { I } = 0 \right] \leq \alpha .\tag{116}
$$

Proof. The interventional superuniformity established in (87) gives

$$
\mathrm { P r } \left[ p _ { I , Y _ { I } ( 1 ) } ^ { \mathrm { i n t } } \leq \alpha \mid T _ { I } = 0 \right] \leq \alpha .\tag{117}
$$

By Lemma 4,

$$
p _ { I , y } ^ { \mathrm { i n t } } \leq p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { + } .\tag{118}
$$

Since overlap ensures $p _ { I , y } ^ { \mathrm { o b s } } > 0$ , rejection is impossible when $\rho _ { I , y } ^ { + } \geq \alpha$ . Otherwise, rejection implies $p _ { I , y } ^ { \mathrm { o b s } } + \rho _ { I , y } ^ { + } \leq \alpha$ Thus, under this coupling,

$$
\left\{ p _ { I , y } ^ { \mathrm { o b s } } \leq \left( \alpha - \rho _ { I , y } ^ { + } \right) _ { + } \right\} \subseteq \left\{ p _ { I , y } ^ { \mathrm { i n t } } \leq \alpha \right\} .\tag{119}
$$

The coupling preserves the joint law with $Y _ { I } ( 1 )$ and holds simultaneously for every y. Evaluating at $y = Y _ { I } ( 1 )$ therefore proves the claim. □

## E.1 Suficient conditions for score stability

Let $d _ { \mathcal { E } } : \mathcal { E } \times \mathcal { E }  [ 0 , \infty )$ be a metric on the exposure space $\mathcal { E } .$ Conditionally on $( X _ { 1 : N } , T _ { 1 : N } , I )$ , suppose that the observational and interventional exposures and outcomes of every afected unit $j \in \mathcal { H } _ { I }$ admit a coupling with the correct conditional marginals satisfying

$$
d _ { \mathcal { E } } ( E _ { j } , \tilde { E } _ { j } ^ { I } ) \leq \Delta _ { E , j } ^ { I } ,\tag{120}
$$

$$
\left| Y _ { j } ( 1 ) - \tilde { Y } _ { j } ^ { I } ( 1 ) \right| \leq \Delta _ { Y , j } ^ { I }\tag{121}
$$

almost surely, where the two bounds are finite, nonnegative, and observable, with the same local dependence allowed for $\Delta _ { j } ^ { I }$ in Assumption 6. Suppose also that the nonconformity score satisfies, for finite constants $L _ { E } , L _ { Y } \geq$ 0,

$$
| S ( x , e , y ) - S ( x , e ^ { \prime } , y ^ { \prime } ) | \leq L _ { E } d _ { \mathcal { E } } ( e , e ^ { \prime } ) + L _ { Y } | y - y ^ { \prime } | .\tag{122}
$$

Then

$$
\left| V _ { j } ^ { \mathrm { i n t } } - V _ { j } ^ { \mathrm { o b s } } \right| \leq L _ { E } \Delta _ { E , j } ^ { I } + L _ { Y } \Delta _ { Y , j } ^ { I }\tag{123}
$$

almost surely. Hence, Assumption 6 holds with

$$
\Delta _ { j } ^ { I } = L _ { E } \Delta _ { E , j } ^ { I } + L _ { Y } \Delta _ { Y , j } ^ { I } .\tag{124}
$$

If the score does not depend on the exposure, the same conclusion holds with

$$
\Delta _ { j } ^ { I } = L _ { Y } \Delta _ { Y , j } ^ { I } .\tag{125}
$$

## F Inductive Interference-Adjusted Weighted Conformal Prediction

In this section, we consider the inductive setting described in Section 2.1.2. Given data D from a network of $N - 1$ units, we construct a prediction set for the ITE of a new unit with covariates $X _ { N } \sim P _ { X }$ after it is embedded in the network. We outline how the reasoning developed for the transductive setting in Section 3 extends to this case. In what follows, we construct a prediction set for the treated potential outcome $Y _ { N } ( 1 )$

As in the transductive setting, a direct application of WCP to the treated calibration units $\mathcal { T } _ { 1 } = \{ j \in \{ 1 , \dots , N -$ $1 \} : T _ { j } = 1 \}$ does not generally guarantee coverage for $Y _ { N } ( 1 )$ . In fact, inverse-propensity weighting with weights

$$
W _ { j } = \frac { 1 } { \pi ( X _ { j } ) } , ~ \mathrm { f o r } ~ j \in \mathcal { T } _ { 1 } \cup \{ N \} ,\tag{126}
$$

accounts for the diference between the covariate distributions of the treated calibration units and the new unit. However, the calibration exposures and outcomes are generated in the original network of N − 1 units, whereas the test exposure $E _ { N }$ and outcome $Y _ { N } ( 1 )$ are generated in the augmented network of N units. Consequently, the calibration scores $V _ { j } ^ { \mathrm { o b s } } = S ( X _ { j } , E _ { j } , Y _ { j } ( 1 ) )$ , for $j \in \mathcal { T } _ { 1 }$ , and the test score $V _ { N } ( Y _ { N } ( 1 ) )$ need not be weighted exchangeable. Thus, the observational conformal p-value

$$
p _ { N , y } ^ { \mathrm { o b s } } = \frac { W _ { N } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { N } ( y ) \} } { W _ { N } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } }\tag{127}
$$

need not be valid.

To correct this procedure, we follow the same interventional reasoning as in the transductive setting. Consider adding unit N with treatment $T _ { N } = 1$ and regenerating the calibration exposures and potential outcomes accord ing to the model in the augmented network. Under this intervention, calibration and test units become weighted exchangeable with weights $W _ { j }$ . Replacing the observational calibration scores in $p _ { N , y } ^ { \mathrm { o b s } }$ with their interventional counterparts yields an ideal WCP procedure with the desired marginal coverage guarantee.

To relate the ideal but unobservable procedure to the observational one, IA-WCP defines the treated calibration units whose neighborhoods change when unit N is added to the network

$$
\begin{array} { r } { \mathcal { H } _ { N } = \{ j \in \mathbb { Z } _ { 1 } : \mathcal { N } _ { j } ( X _ { 1 : N - 1 } ) \neq \mathcal { N } _ { j } ( X _ { 1 : N } ) \} , } \end{array}\tag{128}
$$

and evaluates the correction factor

$$
\rho _ { N , y } = \frac { \sum _ { j \in \mathcal { H } _ { N } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { N } ( y ) \} } { W _ { N } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } } .\tag{129}
$$

The factor (129) accounts for the discrepancy between the observational and interventional conformal p-values and it is used to define the IA-WCP prediction set

$$
\Gamma _ { N } ^ { \mathrm { I A - W C P } } = \left\{ y \in \mathcal { V } : p _ { N , y } ^ { \mathrm { o b s } } > ( \alpha - \rho _ { N , y } ) _ { + } \right\} .\tag{130}
$$

A suitable coupling of the observational and interventional scores ensures that the IA-WCP set contains the ideal WCP set, yielding the following coverage guarantee.

Theorem 3 (Inductive IA-WCP coverage). Under Assumptions 1–5, the $I A - W C P$ set $\Gamma _ { N } ^ { \mathrm { I A - W C P } }$ in (130) satisfies the inequality

$$
\begin{array} { r } { \operatorname* { P r } \left[ Y _ { N } ( 1 ) \in \Gamma _ { N } ^ { \mathrm { I A - W C P } } \right] \geq 1 - \alpha . } \end{array}\tag{131}
$$

Proof. See Appendix $\mathrm { G } .$

For $Y _ { N } ( 0 )$ , the corresponding prediction set is obtained using the control units and weights $1 / ( 1 - \pi ( X _ { j } ) )$ . To distinguish the two constructions, we now make the target treatment explicit in the notation. Let $\mathrm { i } _ { N } ^ { \mathrm { \tiny ~ t , I A - W C P } }$ denote the prediction set for $Y _ { N } ( t )$ constructed at miscoverage level $\alpha _ { t }$ . Choosing $\alpha _ { 1 } + \alpha _ { 0 } = \alpha$ , the ITE prediction set is defined as

$$
\Gamma _ { N } ^ { \mathrm { I T E } } = \left\{ y _ { 1 } - y _ { 0 } : y _ { 1 } \in \Gamma _ { N } ^ { 1 , \mathrm { I A - W C P } } , y _ { 0 } \in \Gamma _ { N } ^ { 0 , \mathrm { I A - W C P } } \right\} ,\tag{132}
$$

and it satisfies (20) by the union bound.

## F.1 Inductive IA-WCP+

As in the transductive setting, the correction can be sharpened when the intervention-induced increase in each afected calibration score is stochastically bounded.

Assumption 7. For every afected calibration unit $j \in \mathcal { H } _ { N }$ , there exists a finite, nonnegative, observable margin $\Delta _ { j } ^ { N }$ that is a measurable function of $( X _ { 1 : N } , T _ { 1 : N } , E _ { j } , Y _ { j } ( 1 ) )$ , but does not depend on the random exposures or outcomes of other units, such that for every $t \in \mathbb { R }$

$$
\begin{array} { r } { \operatorname* { P r } [ V _ { j } ^ { \mathrm { i n t } } > t  { | \begin{array} { l l } { X _ { 1 : N } , T _ { 1 : N } } \end{array} | } \leq \operatorname* { P r } [ V _ { j } ^ { \mathrm { o b s } } + \Delta _ { j } ^ { N } > t  { | \begin{array} { l l } { X _ { 1 : N } , T _ { 1 : N } } \end{array} ] } . } \end{array}\tag{133}
$$

Assumption 7 states that, conditionally on the augmented covariate and treatment vectors, the interventiona score of each afected calibration unit is stochastically dominated by its observational score shifted upward by $\Delta _ { j } ^ { N }$ . Using the stability margins in (133), IA-WCP+ defines the refined correction factor

$$
\rho _ { N , y } ^ { + } = \frac { \sum _ { j \in \mathcal { H } _ { N } } W _ { j } \mathbb { 1 } \left\{ V _ { N } ( y ) - \Delta _ { j } ^ { N } \leq V _ { j } ^ { \mathrm { o b s } } < V _ { N } ( y ) \right\} } { W _ { N } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } .\tag{134}
$$

Under Assumption $^ { 7 , }$ the observational and interventional score vectors admit a coupling with the correct conditional marginals such that, simultaneously for every $y \in \mathcal { V }$

$$
p _ { N , y } ^ { \mathrm { i n t } } \leq p _ { N , y } ^ { \mathrm { o b s } } + \rho _ { N , y } ^ { + } \qquad \mathrm { a . s . }\tag{135}
$$

The correction $\rho _ { N , y } ^ { + }$ counts only the conformal weight of afected calibration units whose observational scores lie within a margin $\bar { \Delta } _ { j } ^ { N }$ below the test score.

Under this coupling, every candidate accepted by the ideal interventional procedure is also accepted by the adjusted observational procedure. This motivates the IA-WCP+ prediction set

$$
\Gamma _ { N } ^ { \mathrm { I A - W C P + } } = \left\{ y \in \mathscr { V } : p _ { N , y } ^ { \mathrm { o b s } } > ( \alpha - \rho _ { N , y } ^ { + } ) _ { + } \right\} .\tag{136}
$$

Because $\rho _ { N , y } ^ { + } ~ \le ~ \rho _ { N , y } .$ , the IA-WCP+ prediction set $\Gamma _ { N } ^ { \mathrm { I A - W C P + } }$ is contained in the IA-WCP prediction set $\Gamma _ { N } ^ { 1 , \mathrm { I A - W C P } }$ in (130).

Theorem 4. Under Assumptions 1–5 and $\gamma ,$ the IA-WCP+ prediction set in (136) satisfies

$$
\mathrm { P r } \left[ Y _ { N } ( 1 ) \notin \Gamma _ { N } ^ { \mathrm { I A - W C P } + } \right] \leq \alpha .\tag{137}
$$

Proof. See Appendix G.4.

## G Proofs of Inductive IA-WCP and Inductive IA-WCP+ Validity

In this section, we provide the proofs of Theorem 3 and Theorem 4. Throughout, the observational network contains N − 1 units and the new unit is indexed by N, as in the main text. Following the same template as the transductive case, we first establish the validity of ideal WCP in the augmented network, then compare its scores with the observational scores through a coupling and establish the validity of IA-WCP. In Appendix G.4 we show that the same reasoning applies to IA-WCP+ under the score stability condition stated in Assumption 7.

## G.1 Validity of the ideal WCP procedure

We consider the case in which the test unit is embedded into the observed network with treatment $T _ { N } = 1$ and we wish to estimate its potential outcome $Y _ { N } ( 1 )$ . Define the augmented covariate and treatment vectors

$$
X _ { 1 : N } = ( X _ { 1 } , \ldots , X _ { N - 1 } , X _ { N } ) , \qquad T _ { 1 : N } = ( T _ { 1 } , \ldots , T _ { N - 1 } , 1 ) ,\tag{138}
$$

where $X _ { N } \sim P _ { X }$ is independent of the observed covariates $X _ { 1 : N - 1 }$ and treatments $T _ { 1 : N - 1 }$ . For each treated calibration unit $j \in \mathcal { I } _ { 1 }$ , let $\tilde { E } _ { j }$ denote the exposure of unit j in the augmented network, and let $\tilde { Y } _ { j } ( 1 )$ denote the corresponding treated potential outcome. Thus,

$$
\tilde { E } _ { j } \mid ( X _ { 1 : N } , T _ { 1 : N } ) \sim P _ { E | X , \mathcal { L } } ( \cdot \mid X _ { j } , \mathcal { L } _ { j } ( X _ { 1 : N } , T _ { 1 : N } ) ) ,\tag{139}
$$

and

$$
\tilde { Y } _ { j } ( 1 ) \mid X _ { j } , \tilde { E } _ { j } \sim P _ { Y ( 1 ) \mid X , E } ( \cdot \mid X _ { j } , \tilde { E } _ { j } ) .\tag{140}
$$

For the test unit, let

$$
E _ { N } \mid \left( X _ { 1 : N } , T _ { 1 : N } \right) \sim P _ { E | X , \mathcal { L } } \left( \cdot \mid X _ { N } , \mathcal { L } _ { N } ( X _ { 1 : N } , T _ { 1 : N } ) \right) ,\tag{141}
$$

and

$$
Y _ { N } ( 1 ) \mid X _ { N } , E _ { N } \sim P _ { Y ( 1 ) \mid X , E } ( \cdot \mid X _ { N } , E _ { N } ) .\tag{142}
$$

We define the interventional nonconformity scores $V _ { j } ^ { \mathrm { i n t } } = S ( X _ { j } , \tilde { E } _ { j } , \tilde { Y } _ { j } ( 1 ) )$ and the triples

$$
\tilde { Z } _ { j } = ( X _ { j } , \tilde { E } _ { j } , \tilde { Y } _ { j } ( 1 ) ) , \qquad j \in \mathcal { T } _ { 1 } ,\tag{143}
$$

and

$$
Z _ { N } = ( X _ { N } , E _ { N } , Y _ { N } ( 1 ) ) .\tag{144}
$$

We first establish the weighted exchangeability of the augmented interventional sample.

Lemma 5 (Inductive weighted exchangeability). Under Assumptions 1, 3, 4, and 2, the joint density of

$$
\big ( ( \tilde { Z } _ { j } ) _ { j \in \mathcal { T } _ { 1 } } , Z _ { N } \big ) ,\tag{145}
$$

conditional on the realized treated set $\mathcal { T } _ { 1 }$ , admits the decomposition

$$
p \left( ( \tilde { Z } _ { j } ) _ { j \in \mathbb { Z } _ { 1 } } , Z _ { N } \mid \mathbb { Z } _ { 1 } \right) = c w ( X _ { N } ) g \left( ( \tilde { Z } _ { j } ) _ { j \in \mathbb { Z } _ { 1 } } , Z _ { N } \right) ,\tag{146}
$$

where $c > 0$ is a normalizing constant,

$$
w ( x ) = \frac { 1 } { \pi ( x ) } ,\tag{147}
$$

and $g ( \cdot )$ is invariant under permutations of its arguments.

Proof. Conditionally on the event defining the treated calibration set $\mathcal { T } _ { 1 }$ , the covariates of treated calibration units have density proportional to $\pi ( x ) p _ { X } ( x )$ . In contrast, the test-unit covariate has density $p _ { X } ( x )$ . Since

$$
p _ { X } ( X _ { N } ) = w ( X _ { N } ) \pi ( X _ { N } ) p _ { X } ( X _ { N } ) , \qquad w ( x ) = { \frac { 1 } { \pi ( x ) } } ,\tag{148}
$$

the covariate density of the augmented sample can be written, up to a normalizing constant, as

$$
w ( X _ { N } ) \prod _ { j \in { \mathcal { Z } } _ { 1 } } \pi ( X _ { j } ) p _ { X } ( X _ { j } ) \pi ( X _ { N } ) p _ { X } ( X _ { N } ) .\tag{149}
$$

After extracting the factor $w ( X _ { N } )$ , the remaining covariate density is symmetric in the augmented covariates $\{ X _ { j } : j \in \mathbb { Z } _ { 1 } \} \cup \{ X _ { N } \}$

Under the augmented treatment vector $T _ { 1 : N }$ , all units in $\mathcal { T } _ { 1 } \cup \{ N \}$ have treatment equal to one. By Assumption 3, the exposure variables are conditionally independent given the augmented covariates and treatments. By Assumption 4, the corresponding treated potential outcomes are conditionally independent given their covariates and exposures, with common conditional distribution $P _ { Y ( 1 ) | X , E }$

It remains to check the invariance of the remaining factor. Since all units in the augmented set ${ \mathcal { T } } _ { 1 } \cup \{ N \}$ have treatment equal to one, permuting these units only relabels their positions. By the permutation equivariance of the neighborhood rule in Assumption 2 and the ordering invariance of the common exposure kernel, the joint exposure distribution is unchanged under a simultaneous permutation of the triples

$$
( X _ { j } , \tilde { E } _ { j } , \tilde { Y } _ { j } ( 1 ) ) , \qquad j \in \mathcal { T } _ { 1 } ,\tag{150}
$$

and the test triple $( X _ { N } , E _ { N } , Y _ { N } ( 1 ) )$ . Covariates and variables of the control units are integrated out; their contribution is unchanged by such a relabeling. Hence the remaining factor $g ( \cdot )$ is invariant under permutations of the augmented sample, proving the claim. □

By Lemma 5, the augmented interventional sample is weighted exchangeable with weights $w ( x ) = 1 / \pi ( x )$ Therefore, the standard weighted conformal validity argument (Lei and Cand\`es, 2021) implies that the ideal interventional conformal p-value

$$
p _ { N , y } ^ { \mathrm { i n t } } = \frac { W _ { N } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { N } ( y ) \} } { W _ { N } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } }\tag{151}
$$

is valid and the ideal prediction set

$$
\tilde { \Gamma } _ { N } ^ { \mathrm { W C P } } = \{ y \in \mathcal { V } : p _ { N , y } ^ { \mathrm { i n t } } > \alpha \}\tag{152}
$$

satisfies

$$
\begin{array} { r } { \operatorname* { P r } [ Y _ { N } ( 1 ) \in \tilde { \Gamma } _ { N } ^ { \mathrm { W C P } } ] \geq 1 - \alpha . } \end{array}\tag{153}
$$

## G.2 Coupling comparison

Recall the definition of the units afected by the addition of unit N to the network,

$$
\mathcal { H } _ { N } = \{ j \in \mathbb { Z } _ { 1 } : \mathcal { N } _ { j } ( X _ { 1 : N - 1 } ) \neq \mathcal { N } _ { j } ( X _ { 1 : N } ) \} ,\tag{154}
$$

and write $\mathcal { U } _ { N } = \mathcal { T } _ { 1 } \setminus \mathcal { H } _ { N }$ . For the proof, we consider the tighter exposure-law afected set

$$
\mathcal { H } _ { N } ^ { \mathrm { e x } } = \Big \{ j \in \mathcal { Z } _ { 1 } : P _ { E | X , \mathcal { L } } \big ( \cdot \big | X _ { j } , \mathcal { L } _ { j } \big ( X _ { 1 : N - 1 } , T _ { 1 : N - 1 } \big ) \big ) \neq P _ { E | X , \mathcal { L } } \big ( \cdot \big | X _ { j } , \mathcal { L } _ { j } \big ( X _ { 1 : N } , T _ { 1 : N } \big ) \big ) \Big \} ,\tag{155}
$$

and let $\mathcal { U } _ { N } ^ { \mathrm { e x } } = \mathcal { T } _ { 1 } \setminus \mathcal { H } _ { N } ^ { \mathrm { e x } }$ . Since equal neighborhoods give equal local configurations before and after augmentation, $\mathcal { H } _ { N } ^ { \mathrm { e x } } \subseteq \mathcal { H } _ { N }$ . The tighter set $\mathcal { H } _ { N } ^ { \mathrm { e x } }$ is used to construct the coupling, after which the index set of the nonnegative correction sum is enlarged from $\mathcal { H } _ { N } ^ { \mathrm { e x } }$ to $\mathcal { H } _ { N }$

Lemma 6. Under Assumptions 3 and $^ { 4 , }$ define the observational conformal p-value by

$$
p _ { N , y } ^ { \mathrm { o b s } } = \frac { W _ { N } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { N } ( y ) \} } { W _ { N } + \sum _ { j \in \mathbb { Z } _ { 1 } } W _ { j } } .\tag{156}
$$

Then this p-value and the interventional conformal p-value in (151) admit a coupling such that, simultaneously for every $y \in \mathcal { V }$

$$
p _ { N , y } ^ { \mathrm { i n t } } \le p _ { N , y } ^ { \mathrm { o b s } } + \rho _ { N , y } \qquad a . s . ,\tag{157}
$$

where $\rho _ { N , y }$ is defined in (129).

Proof. We construct the coupling conditionally on $\left( X _ { 1 : N } , T _ { 1 : N } \right)$ . For every $j \in \mathcal { U } _ { N } ^ { \mathrm { e x } }$ , the definition of $\mathcal { U } _ { N } ^ { \mathrm { e x } }$ gives

$$
P _ { E | X , \mathcal { L } } \left( \cdot \mid X _ { j } , \mathcal { L } _ { j } \left( X _ { 1 : N - 1 } , T _ { 1 : N - 1 } \right) \right) = P _ { E | X , \mathcal { L } } \left( \cdot \mid X _ { j } , \mathcal { L } _ { j } \left( X _ { 1 : N } , T _ { 1 : N } \right) \right) .\tag{158}
$$

Thus, for each $j \in \mathcal { U } _ { N } ^ { \mathrm { e x } }$ , the observational and interventional exposures may be coupled so that

$$
E _ { j } = \tilde { E } _ { j } \qquad \mathrm { a . s . }\tag{159}
$$

Conditional on these coupled exposures, the corresponding treated potential outcomes have the same conditional distribution $P _ { Y ( 1 ) | X , E }$ and may therefore be coupled so that

$$
\begin{array} { r } { Y _ { j } ( 1 ) = \tilde { Y } _ { j } ( 1 ) \qquad \mathrm { a . s . } } \end{array}\tag{160}
$$

For every $j \in \mathcal { H } _ { N } ^ { \mathrm { e x } }$ , choose any coupling with the correct observational and interventional marginals.

By Assumptions 3 and 4, the unit-specific exposure and outcome variables are conditionally independent across units. These couplings can therefore be combined into a product coupling that preserves the observational and interventional score-vector marginals. Under this coupling,

$$
V _ { j } ^ { \mathrm { o b s } } = V _ { j } ^ { \mathrm { i n t } } \qquad \mathrm { a . s . ~ f o r ~ e v e r y ~ } j \in { \mathcal U } _ { N } ^ { \mathrm { e x } } .\tag{161}
$$

The test exposure and outcome have the same joint law with the augmented covariates and treatments in both constructions, and are coupled identically. The two p-values therefore use the same target score $V _ { N } ( y )$ and the same weights, since the weights depend only on the covariates. Consequently,

$$
p _ { N , y } ^ { \mathrm { i n t } } - p _ { N , y } ^ { \mathrm { o b s } } = \frac { \sum _ { j \in \mathcal { H } _ { N } ^ { \mathrm { e x } } } W _ { j } \left[ \mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { N } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { N } ( y ) \} \right] } { W _ { N } + \sum _ { j \in \mathcal { T } _ { 1 } } W _ { j } } .\tag{162}
$$

For every $j \in \mathcal { H } _ { N } ^ { \mathrm { e x } }$

$$
\mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { N } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { N } ( y ) \} \leq \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { N } ( y ) \} .\tag{163}
$$

Therefore,

$$
p _ { N , y } ^ { \mathrm { i n t } } - p _ { N , y } ^ { \mathrm { o b s } } \leq \frac { \sum _ { j \in \mathcal { H } _ { N } ^ { \mathrm { e x } } } W _ { j } \mathbb { I } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { N } ( y ) \} } { W _ { N } + \sum _ { j \in \mathcal { I } _ { 1 } } W _ { j } } \leq \frac { \sum _ { j \in \mathcal { H } _ { N } } W _ { j } \mathbb { I } \{ V _ { j } ^ { \mathrm { o b s } } < V _ { N } ( y ) \} } { W _ { N } + \sum _ { j \in \mathcal { I } _ { 1 } } W _ { j } } = \rho _ { N , y } .\tag{164}
$$

Hence,

$$
p _ { N , y } ^ { \mathrm { i n t } } \leq p _ { N , y } ^ { \mathrm { o b s } } + \rho _ { N , y } \qquad \mathrm { a . s . }\tag{165}
$$

Since the coupling does not depend on y, this inequality holds simultaneously for every $y \in \mathcal { V }$

## G.3 Proof of Theorem 3

By Lemma $6 ,$ there is a coupling preserving the observational and interventional marginals such that $p _ { N , y } ^ { \mathrm { i n t } } \ \leq$ $p _ { N , y } ^ { \mathrm { o b s } } + \rho _ { N , y }$ simultaneously for every $y \in \mathcal { V }$ . If $\rho _ { N , y } < \alpha$ , then $p _ { N , y } ^ { \mathrm { i n t } } > \alpha$ implies $p _ { N , y } ^ { \mathrm { o b s } } > \alpha - \rho _ { N , y }$ . If $\rho _ { N , y } \ge \alpha$ the IA-WCP acceptance threshold is zero, and $p _ { N , y } ^ { \mathrm { o b s } } > 0$ because $W _ { N } > 0$ . Hence,

$$
\tilde { \Gamma } _ { N } ^ { \mathrm { W C P } } \subseteq \Gamma _ { N } ^ { \mathrm { I A - W C P } } \qquad \mathrm { a . s . }\tag{166}
$$

Evaluating this inclusion at the shared target outcome and using the validity of the interventional p-value gives

$$
\mathrm { P r } [ Y _ { N } ( 1 ) \notin \Gamma _ { N } ^ { \mathrm { I A - W C P } } ] \leq \mathrm { P r } [ p _ { N , Y _ { N } ( 1 ) } ^ { \mathrm { i n t } } \leq \alpha ] \leq \alpha ,\tag{167}
$$

which proves Theorem 3.

For the control potential outcome, fix the target treatment at zero, use the control calibration units and weights $1 / ( 1 - \pi ( X _ { j } ) )$ , and repeat the argument. Write $\Gamma _ { N } ^ { t , \mathrm { I A - W C P } }$ for the set targeting $Y _ { N } ( t )$ at miscoverage level $\alpha _ { t }$ Since the neighborhood excludes the unit itself, the target exposure law is unchanged by its own treatment; both potential-outcome guarantees therefore apply to the common target exposure in the ITE definition. If both potential outcomes belong to their respective prediction sets, their diference belongs to $\Gamma _ { N } ^ { \mathrm { I T E } }$ . Thus,

$$
\operatorname* { P r } [ Y _ { N } ( 1 ) - Y _ { N } ( 0 ) \notin \Gamma _ { N } ^ { \mathrm { F T F } } ] \le \operatorname* { P r } [ Y _ { N } ( 1 ) \notin \Gamma _ { N } ^ { 1 , \mathrm { I A - W C P } } ] + \operatorname* { P r } [ Y _ { N } ( 0 ) \notin \Gamma _ { N } ^ { 0 , \mathrm { I A - W C P } } ] \le \alpha _ { 1 } + \alpha _ { 0 } = \alpha .\tag{168}
$$

## G.4 Proof of Theorem 4

For every $j \in \mathcal { U } _ { N }$ , the coupling in the proof of Lemma 6 gives

$$
V _ { j } ^ { \mathrm { { i n t } } } = V _ { j } ^ { \mathrm { { o b s } } } \qquad \mathrm { { a . s . } }\tag{169}
$$

For every $j \in \mathcal { H } _ { N }$ , Assumption 7 establishes the required conditional stochastic ordering. By the coupling characterization of stochastic order (Strassen, 1965), applied conditionally on $\left( X _ { 1 : N } , T _ { 1 : N } \right)$ , there exists a coupling of $V _ { j } ^ { \mathrm { i n t } }$ and $( V _ { j } ^ { \mathrm { o b s } } , \Delta _ { j } ^ { N } )$ with the correct conditional marginals such that

$$
V _ { j } ^ { \mathrm { { i n t } } } \leq V _ { j } ^ { \mathrm { { o b s } } } + \Delta _ { j } ^ { N } \qquad \mathrm { { a . s . } }\tag{170}
$$

Conditionally on $\left( X _ { 1 : N } , T _ { 1 : N } \right)$ , each pair $( V _ { j } ^ { \mathrm { o b s } } , \Delta _ { j } ^ { N } )$ depends only on $( E _ { j } , Y _ { j } ( 1 ) )$ ). Assumptions 3 and 4 therefore imply that these pairs are conditionally independent across units. The interventional scores are conditionally independent for the same reason. Hence, the coordinatewise couplings can be combined into a product coupling under which (170) holds simultaneously for every $j \in \mathcal { H } _ { N }$ , while preserving the observational and interventional score-vector marginals.

For every $j \in \mathcal { H } _ { N }$ , (170) implies

$$
\begin{array} { r } { \mathbb { 1 } \{ V _ { j } ^ { \mathrm { i n t } } \geq V _ { N } ( y ) \} - \mathbb { 1 } \{ V _ { j } ^ { \mathrm { o b s } } \geq V _ { N } ( y ) \} \leq \mathbb { 1 } \left\{ V _ { N } ( y ) - \Delta _ { j } ^ { N } \leq V _ { j } ^ { \mathrm { o b s } } < V _ { N } ( y ) \right\} . } \end{array}\tag{171}
$$

The target exposure and outcome are coupled identically, as in Lemma 6. Since the unafected scores coincide under the coupling, the interventional and observational p-values satisfy, simultaneously for every $y \in \mathcal { V }$

$$
p _ { N , y } ^ { \mathrm { i n t } } \leq p _ { N , y } ^ { \mathrm { o b s } } + \rho _ { N , y } ^ { + } \qquad \mathrm { a . s . }\tag{172}
$$

Because $p _ { N , y } ^ { \mathrm { o b s } } > 0$ , the same threshold argument used in Appendix G.3 gives

$$
\left\{ p _ { N , y } ^ { \mathrm { o b s } } \leq \left( \alpha - \rho _ { N , y } ^ { + } \right) + \right\} \subseteq \left\{ p _ { N , y } ^ { \mathrm { i n t } } \leq \alpha \right\} .\tag{173}
$$

Evaluating this inclusion at $y = Y _ { N } ( 1 )$ yields

$$
\mathrm { P r } \left[ Y _ { N } ( 1 ) \notin \Gamma _ { N } ^ { \mathrm { I A - W C P } + } \right] \leq \alpha .\tag{174}
$$

## H Additional Experiments

## H.1 Trafic Slicing

In this section, we evaluate the proposed methods in a wireless trafic-slicing problem (Foukas et al., 2017; Bao et al., 2017). We consider a wireless-network setting in which a finite amount of radio resources is allocated among multiple users. The users are organized into trafic slices representing services with similar requirements, such as latency-sensitive, high-throughput, or best-efort trafic. Each slice is assigned a resource pool that must be shared among its scheduled users. Resource allocation therefore induces interference, since scheduling one user reduces the resources available to other scheduled users in the same slice. Consequently, a scheduled user’s outcome depends not only on its own channel and trafic characteristics, but also on the number of other scheduled users sharing the same resource pool (Foukas et al., 2017; Bao et al., 2017).

## H.1.1 Setup

We consider a population of $N = 2 0 0$ devices sharing radio resources across $N _ { S }$ slices. Each device i is characterized by a feature vector $X _ { i }$ that includes its channel-quality indicator, trafic characteristics, backlog information, and the slice $Z _ { i }$ to which it is assigned. The binary treatment $T _ { i } \in \{ 0 , 1 \}$ indicates whether device i is scheduled and granted radio resources $( T _ { i } = 1 )$ or not $( T _ { i } = 0 )$ . Scheduling decisions are drawn independently according to

$$
T _ { i } \mid X _ { i } \sim \operatorname { B e r n o u l l i } ( \pi ( X _ { i } ) ) ,\tag{175}
$$

where the propensity score $\pi ( X _ { i } )$ is a function of the device covariates.

The neighborhood of device i is the set of other devices assigned to the same slice,

$$
{ \mathcal { N } } _ { i } ( X _ { 1 : N } ) = \{ j \neq i : Z _ { j } = Z _ { i } \} ,\tag{176}
$$

and the exposure of device i corresponds to the number of users in this neighborhood that are scheduled,

$$
E _ { i } = \sum _ { j \in \mathcal { N } _ { i } ( X _ { 1 : N } ) } T _ { j } .\tag{177}
$$

Thus, the exposure $E _ { i }$ measures the same-slice contention experienced by device $i .$

The potential outcome $Y _ { i } ( t )$ represents the throughput of device i under the scheduling decision t. If device i is not scheduled, $T _ { i } = 0$ , it receives no radio resources and its throughput is zero, $Y _ { i } ( 0 ) = 0$ . When device i is scheduled, its throughput depends on its channel quality and the amount of same-slice contention. Specifically, this is given by

$$
Y _ { i } ( 1 ) = \operatorname* { m i n } \left\{ \frac { \eta ( X _ { i } ) \xi _ { i } } { ( E _ { i } + 1 ) N _ { S } } , R _ { \operatorname* { m a x } } \right\} ,\tag{178}
$$

where $\eta ( X _ { i } )$ is a spectral-eficiency term determined by the channel-quality indicator contained in $X _ { i } ,$ and $\xi _ { i }$ is a log-normal multiplicative noise variable normalized to have mean one. The factor $N _ { S } ^ { - 1 }$ represents the fraction of the total resources assigned to each slice, while the term $( E _ { i } + 1 ) ^ { - 1 }$ models equal resource sharing among all scheduled devices in the slice, including device i. Consequently, greater exposure corresponds to stronger resource contention and lower achievable throughput. The throughput is capped at the maximum supported rate $R _ { \mathrm { m a x } }$

We use gradient-boosted quantile regression as the baseline predictor for the treated potential outcome. The predictor is trained on treated samples from five independently generated network realizations. The resulting lower and upper quantile estimates, $\hat { q } _ { \alpha / 2 } ( X )$ and $\hat { q } _ { 1 - \alpha / 2 } ( X )$ , define the exposure-independent nonconformity score of each treated calibration device as

$$
S ( X _ { i } , E _ { i } , Y _ { i } ( 1 ) ) = \operatorname* { m a x } \left\{ \widehat { q } _ { \alpha / 2 } ( X _ { i } ) - Y _ { i } ( 1 ) , Y _ { i } ( 1 ) - \widehat { q } _ { 1 - \alpha / 2 } ( X _ { i } ) \right\} .\tag{179}
$$

![](images/11a451d1d2d80bcf0b4f70d3dd519c8f6ce6c29b162fd35909a980867c1dcd0a.jpg)  
Figure 7: Empirical coverage (left) and average prediction-set size (right) for QR (Koenker and Bassett Jr, 1978), WCP (Lei and Cand\`es, 2021), ideal WCP based on interventional p-values, and the two proposed interferenceadjusted methods, IA-WCP and IA-WCP+. The prediction sets target the treated potential outcomes of unscheduled devices. Prediction-set sizes are normalized by $R _ { \mathrm { m a x } } ,$ and the target coverage is $1 - \alpha = 0 . 8 .$ . Results are averaged over 250 independently generated network realizations. The inset provides a magnified view of the prediction-set sizes for all methods except the IA-WCP procedure.

## H.1.2 Benchmarks

We consider transductive counterfactual estimation and implement the following methods: (a) uncalibrated quantile-regression interval (QR) (Koenker and Bassett Jr, 1978), given by

$$
\Gamma ^ { \mathrm { Q R } } ( X ) = [ \hat { q } _ { \alpha / 2 } ( X ) , \hat { q } _ { 1 - \alpha / 2 } ( X ) ] ,\tag{180}
$$

(b) conventional WCP (Lei and Cand\`es, 2021), (c) ideal WCP, (d) IA-WCP, and (e) the stability-aware IA-WCP+. The last two methods are proposed in this work, while ideal WCP is the prediction set constructed from the interventional p-value in (30). Ideal WCP is available in simulation because the data-generating mechanism allows us to recompute the interventional exposures and outcomes, and therefore serves as an oracle baseline. For IA-WCP+, intervening to schedule the target device I increases the exposure by one for every treated calibration device in the same slice, while leaving the other devices unafected. Since the quantile-regression nonconformity score is 1-Lipschitz in the outcome, the score-stability condition in Assumption 6 is satisfied with

$$
\Delta _ { j } ^ { I } = \frac { Y _ { j } ( 1 ) } { E _ { j } + 2 } \mathbb { 1 } \{ Z _ { j } = Z _ { I } \} .\tag{181}
$$

## H.1.3 Results

Figure 7 reports the empirical coverage and normalized average prediction-set size at the target miscoverage level $\alpha = 0 . 2$ , as the number of slices varies over $N _ { S } \in \{ 2 , 3 , 4 , 6 , 1 0 , 1 6 , 3 2 \}$ . For each network realization, both quantities are averaged over all unscheduled target devices. A smaller number of slices $N _ { S }$ corresponds to a larger average number of devices in each slice and stronger interference. By varying the number of slices $N _ { S }$ we evaluate the performance of the proposed methods under diferent interference regimes.

For every value of the number of slices $N _ { S }$ , the uncalibrated QR interval fails to attain the target coverage level $1 - \alpha = 0 . 8$ . WCP approaches nominal coverage when interference is weak, attaining coverage of approximately 0.81 at $N _ { S } = 3 2$ . As the number of slices decreases and the level of interference increases, however, its coverage falls below the nominal level, reaching approximately 0.76 at $N _ { S } = 2 $ . In contrast, ideal WCP remains above the target coverage for all values of $N _ { S }$ . The coverage gap between WCP and ideal WCP illustrates the loss of validity caused by calibrating with observational scores.

Both proposed interference-adjusted methods maintain coverage above the target level for all values of $N _ { S }$ IA-WCP uses a worst-case correction and therefore becomes highly conservative when a large fraction of the calibration weight is assigned to devices in the target slice. This conservativeness results in substantially larger prediction sets, particularly in the strongest-interference regimes. In contrast, IA-WCP+ mitigates this loss of eficiency by exploiting the device-specific stability margins in (181), and its coverage and prediction-set size remain close to those of ideal WCP.

![](images/7b9b5ecb23cf76d8856f23e98b0c3edf1f71cdbdd8cfd645982866f6ea5a9227.jpg)  
(a) Transductive setting.

![](images/15b6a5fcaede96abab3de88a45f061060911200167191e2f80c298e699c07d8c.jpg)  
(b) Inductive setting.  
Figure 8: Coverage and normalized width of WCP, ideal WCP, IA-WCP, and IA-WCP+ for transductive and inductive inference as a function of the sigmoid parameter λ. The first row corresponds to target miscoverage $\alpha = 0 . 1$ , while the second row corresponds to α = 0.2.

## H.2 Additional Synthetic Results

In this section, we consider the synthetic model introduced in Section 4 and present results that complement those in the main text. Specifically, we consider both transductive and inductive inference, additional target coverage levels, and experiments that vary the interference strength and outcome noise.

## H.2.1 Interference Strength

Figure 8 reports the empirical coverage and normalized width of prediction sets returned by WCP, ideal WCP, the proposed IA-WCP, and IA-WCP+ in both the transductive and inductive settings as a function of the sigmoid parameter λ. The parameter λ controls the sharpness of the exposure-outcome mechanism in (36). In particular, when λ is close to zero, the exposure-outcome mechanism is flat and peer exposure has little efect on the treated outcome. As λ increases, the treated outcome changes rapidly depending on whether a majority of peers are treated.

For λ = 0, WCP and ideal WCP coincide and attain the target coverage levels. In this regime, the observational calibration scores used by WCP are the same as the interventional calibration scores used by ideal WCP. However, as λ increases, the gap between WCP and ideal WCP becomes more visible, and WCP begins to fall below nominal coverage. This degradation is more pronounced in the transductive setting. IA-WCP and IA-WCP+ remain above the target coverage level across the range of values considered. WCP and ideal WCP remain relatively close in width, while IA-WCP becomes wider as the sigmoid parameter λ grows because it applies a worst-case correction for afected calibration units. In contrast, IA-WCP+ maintains a width close to that of ideal WCP across the range of λ values considered, while still attaining the target coverage level.

## H.2.2 Outcome Noise

Figure 9 studies the efect of the outcome noise level $\sigma _ { \varepsilon }$ . When the noise level is small, the nonconformity scores are determined primarily by the deterministic exposure mechanism, and changes in peer exposure induced by the target intervention can substantially alter the calibration scores. In this regime, WCP falls below the nominal coverage level, especially in the transductive setting.

As $\sigma _ { \varepsilon }$ increases, the contribution of noise to the variability of the nonconformity scores grows. In this regime,

![](images/64a536f2d5a76842963b4f2db6387747dbd1b1293b91d31f939a1bc1ea9e8196.jpg)  
(a) Transductive setting.

![](images/c3765da823909bdaedae3eeb522198e531b5d851b9f843ed8db0a0c2fcb3a370.jpg)  
(b) Inductive setting.  
Figure 9: Coverage and normalized width of WCP, ideal WCP, IA-WCP, and IA-WCP+ for transductive and inductive inference as a function of the noise parameter $\sigma _ { \varepsilon } .$ . The first row corresponds to target miscoverage $\alpha = 0 . 1$ , while the second row corresponds to α = 0.2.

WCP becomes closer to ideal WCP, and its empirical coverage degradation is reduced. IA-WCP and IA-WCP+ maintain coverage above the target level in both the transductive and inductive settings across the range of noise levels considered. Larger noise levels naturally lead to wider prediction sets for all methods, and the gap between IA-WCP and other baselines is most visible in the large-noise regime.

## H.2.3 Interference Locality

Figure 10 varies the number of groups G, which controls the locality of interference. For fixed N, smaller values of G correspond to larger peer groups. In this regime, changing the treatment of a target unit, or embedding a new treated unit in the inductive setting, can afect a larger fraction of the calibration sample. The discrepancy between the observational and interventional calibration scores is therefore larger for smaller values of G.

For small values of G, WCP coverage falls below the nominal level. As G increases, peer groups become smaller and interference becomes more localized. As a result, the observational calibration scores used by WCP become closer to the ideal interventional scores, and the empirical coverage of WCP improves, approaching that of ideal WCP. IA-WCP and IA-WCP+ compensate for the presence of interference, and their coverage levels remain above the nominal level throughout. As G grows, the correction term in IA-WCP becomes smaller, so the conservativeness of the method decreases and its performance approaches that of ideal WCP.

## H.2.4 Base Predictors

We conclude by evaluating the performance of WCP, ideal WCP, IA-WCP, and IA-WCP+ with the diferent base predictors introduced in Section 4.2.

Figure 11 reports the empirical coverage and normalized width of the prediction sets obtained using these methods and predictors. As expected, the mismatched interference-free predictor yields the widest prediction sets, whereas the predictor fitted using both X and E is the most eficient. We also observe that the miscoverage gap of WCP is larger when using the interference-free predictor. In contrast, with the fitted predictors, the coverage achieved by WCP is much closer to the target level, despite the absence of coverage guarantees. The proposed methods, IA-WCP and IA-WCP+, remain valid for all predictors considered, and their eficiency increases with the quality of the base predictor.

![](images/c52ca2b0c751ff4fae787adcf0f4428ba111d38913c42d47c35c4c5d8361c5bf.jpg)

![](images/260e52a90074d5c2e91cb49a800c0f471ac53af9f888ded48e658ecf52d106db.jpg)

![](images/02d3ee49d960b112db477c357c2604cff1c9b1c2e498512050979e529d6c5fa7.jpg)

![](images/a10dc54fee2bb053204b6fe8f1fef2e2764d5f10e41b61a0a4b3d097145b79d2.jpg)

![](images/ad8af5f95858c96ba641f72c413d849b88c657ae81462c6972b4b747609e55a7.jpg)

![](images/9330d48b2be9db851975d378eec424c6323fbf258a64d702cd6d3526b1b3de0b.jpg)

(a) Transductive setting.  
![](images/9fdf6eeeb5241e26348b6da99affd47947825447fd1154134cfa593e2b557f20.jpg)

![](images/11d3b3fb1b72c67348c64d2fa1957fcb2595a494623496444426884bc41570d1.jpg)  
(b) Inductive setting.  
Figure 10: Coverage and normalized width of WCP, ideal WCP, IA-WCP and IA-WCP+ for transductive and inductive inference as a function of the number of groups G. The first row corresponds to target miscoverage $\alpha = 0 . 1$ , while the second row corresponds to $\alpha = 0 . 2$

![](images/fefa35d4e3acccbdfa730fd7c8424462049add369f8dff314a648d87b9463282.jpg)

![](images/e314dab34298a7df2dcc303fe5891d2df4909fb26bd41af68727da439ce6db4f.jpg)

![](images/16d29f2abc09a1c145e238261184dffbf9d3c7c8fed33d81563d491f57269b77.jpg)

![](images/571eb093d51a2f33be1993d5aecbce853e7e078df0a64191ad24029dbf075b43.jpg)

![](images/5eb92c13100828a083de1927610513b91e6a59dd74a11e4c95673370c538e0fe.jpg)

![](images/5756cb1ee52b723e4cd0a22d564186a8c8ddbdec6453b779741eb2f2bf8e523d.jpg)  
(a) Transductive setting.

![](images/3aa9aa6e02b0894c02c038417f54f924bca4edc20116cc5146d7adb6754c25df.jpg)

![](images/91d27207dba4e488f6e8aaa5757cccd10fcbba016edc36d22c06528a75330295.jpg)  
(b) Inductive setting.  
Figure 11: Coverage and normalized width of WCP, ideal WCP, IA-WCP, and IA-WCP+ for transductive and inductive inference for diferent types of base predictors. The first row corresponds to target miscoverage $\alpha = 0 . 1$ while the second row corresponds to $\alpha = 0 . 2$