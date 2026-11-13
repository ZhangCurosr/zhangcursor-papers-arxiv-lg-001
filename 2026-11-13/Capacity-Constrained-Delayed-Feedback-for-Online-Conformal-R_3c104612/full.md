# Capacity-Constrained Delayed Feedback for Online Conformal Risk Control

Bowen Wang, Student Member, IEEE and Osvaldo Simeone, Fellow, IEEE

## Abstract

Online conformal risk control (OCRC) enables reliable uncertainty quantification in machine learning by converting model outputs into prediction sets with long-term risk guarantees. These guarantees, however, hinge on the assumption that the loss of each round is revealed immediately and losslessly. In practice, feedback may be delayed, and limited resources may prevent tracking all pending predictions, causing some feedback to be lost. To address this challenge, we develop capacity-aware OCRC (CA-OCRC), a framework that jointly determines how to allocate limited tracking feedback capacity and how to construct prediction sets. CA-OCRC is based on a randomized tracking policy that samples a proxy delay to decide whether, and for how long, feedback for the current round is tracked. CA-OCRC updates the prediction sets via inverse probability weighting, compensating for the possibility of feedback lost due to delays and limited tracking capacity. We analyze risk bounds for both arbitrary and stochastic delays, revealing a trade-off between receiving more feedback and avoiding capacity saturation. Building on these bounds, we optimize the proxy delay distribution, showing scaling laws for the tracking capacity that ensure the same long-term risk as OCRC with unlimited feedback. Experiments on medical imaging, natural language processing, computer vision, and wireless communications show the effectiveness of CA-OCRC as compared to existing OCRC mechanisms.

## Index Terms

Reliable machine learning, conformal risk control, delayed feedback, online learning.

The work of O. Simeone was supported by an EPSRC Open Fellowship (EP/W024101/1), by the EPSRC (EP/X011852/1), and by the European Research Council (ERC) (Grant No. 101198347). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

Bowen Wang is with the Department of Engineering, King’s College London, London WC2R 2LS, U.K. (e-mail: bowen.wang@kcl.ac.uk).

Osvaldo Simeone is with the Institute for Intelligent Networked Systems (INSI) at Northeastern University London, London E1 8PH, U.K. (e-mail: o.simeone@northeastern.edu).

## I. INTRODUCTION

## A. Context and Motivation

Artificial intelligence (AI) models, such as large language models (LLMs), are increasingly deployed in applications ranging from finance and healthcare to robotics, autonomous systems, and wireless communications [1]–[4]. Particularly in high-stakes settings, their reliable deploy ment requires not only predictive accuracy, but also principled methods to quantify predictive uncertainty and to control the risks associated with predictions [5], [6].

Conformal prediction (CP) [7], [8] quantifies uncertainty by transforming the outputs of a pre-trained model into prediction sets that contain the true label with user-specified probability. However, coverage alone may not adequately capture the task-specific consequences of prediction errors. Conformal risk control (CRC) [9]–[11] generalizes CP from the miscoverage indicator to general loss functions, guaranteeing that the expected loss incurred by the prediction sets does not exceed a user-specified target risk level.

The guarantees provided by CP and CRC rely on the exchangeability of calibration and test data [7], [9], which may be violated in sequential environments with evolving data distributions. Online CP (OCP) [12]–[14] and online CRC (OCRC) [15]–[18] address this problem by adaptively updating the prediction sets based on feedback about the losses incurred in past rounds. This way, OCP and OCRC ensure that the long-term average risk approaches an arbitrary target risk level for any, possibly adversarial, data sequence.

The guarantees of OCP and OCRC hinge on timely feedback, as the loss incurred at each round is assumed to be revealed to the predictor by the end of the round [12]–[18]. In practical applications, however, such feedback may not be immediately available and it may lost. For instance, in healthcare, the outcomes of medical decisions may only become known after subsequent examinations or patient follow-up [19], while in open-domain question answering, the correctness of the answers returned to user questions may only be assessed after a slow human annotation process [20]–[22].

Tracking each prediction until its feedback becomes available may also require substantial monitoring and storage resources. In healthcare, patient follow-up is constrained by limited monitoring capacity [19], while, in question answering, only a limited number of questions can be submitted for human annotation at any time, owing to its high cost [21]. Similar constraints arise in other applications, such as wireless communication systems, which must manage pending transmissions with finite memory and computational resources [23], and online recommender and advertising systems, in which user conversions may occur up to a month after an item is displayed [24]. It may thus be infeasible to track all predictions with pending feedback simultaneously, resulting in the loss of some feedback signals.

![](images/07a4272abd7026cded5ad0ca8f89a8a2add6208068fffba65b19df4dfcb803a5.jpg)  
(b) Example of tracking dynamics with $K = 2$  
Fig. 1. Illustration of sequential prediction with delayed feedback under a finite tracking capacity. (a) Operations within round t: The predictor observes the input $X _ { t }$ and generates the prediction set $\mathcal { C } _ { t } .$ . It then evicts the rounds in the set $\mathcal { E } _ { t } \subseteq \mathcal { K } _ { t - 1 }$ from the current set $\boldsymbol { \mathcal { K } } _ { t - 1 }$ of tracked rounds, and decides whether to admit round t, i.e., $\mathcal { A } _ { t } \in \{ \emptyset , \{ t \} \}$ , subject to the capacity constraint $| K _ { t } ^ { + } | \leq K$ . Finally, the feedback signals $\{ L _ { i } \} _ { i \in { \mathcal { O } } _ { \mathit { i } } }$ of the tracked rounds whose delays expire at round t are received, and the rounds in ${ \mathcal { O } } _ { t }$ are removed from the tracking set. (b) Example with $K = 2 ,$ , showing for each round t the decisions $\mathcal { E } _ { t }$ and $\boldsymbol { A } _ { t } .$ together with the resulting tracking set $\textstyle { \mathcal { K } } _ { t }$ at the end of the round. $\mathrm { ~ A t ~ } t = 3$ , the tracking set is full, so round $t = 3$ is not admitted, and its feedback, arriving at the end of round $^ { 4 , }$ is lost. At $t = 4 ,$ round 1 is evicted to admit round 4. The feedback of round 4 is received immediately $( d _ { 4 } = 0 )$ , whereas that of round 1, arriving at the end of the same round, is lost $( d _ { 1 } = 3 )$ ). Round 2 remains tracked, awaiting its feedback at the end of round 5 $( d _ { 2 } = 3 )$

Motivated by this challenge, as illustrated in Fig. 1, we investigate OCRC, and thus also the special case of OCP, under capacity-constrained delayed feedback. The goal is to jointly design the prediction sets and the feedback tracking policy, which determines which rounds to track and for how long, so that the long-term risk approaches a user-specified target level.

## B. Related Work

1) Online Conformal Prediction with Delayed Feedback: Reference [25] studied delayed feedback in OCP by extending adaptive conformal inference (ACI) [12] to multi-step-ahead timeseries forecasting, in which the feedback of a forecast is observed only after its forecast horizon has elapsed. Subsequently, the work [26] extended multi-level quantile calibration to delayed feedback. More recently, the paper [27] analyzed ACI under fixed delays, establishing long-term coverage bounds by decomposing the delayed feedback into interleaved ACI instances. In all these studies, however, the tracking capacity is implicitly unlimited, since all pending feedback signals are tracked until their arrival. At the opposite extreme, intermittent OCP [28], [29], in which the feedback of each round is either received immediately or lost, can be viewed as a special case of delayed feedback with unit tracking capacity (see Sec. III-B).

2) Online Convex Optimization with Delayed Feedback: Early work on delayed online convex optimization (OCO) [30], [31] showed that online gradient descent (OGD) [32] achieves sublinear regret under fixed delays [33]. This analysis was extended to arbitrary delays in [34]. Subsequent studies investigated dynamic regret in non-stationary environments [35], and unknown delays under general norms [36]. More recently, the work [37] went beyond OGD and used the followthe-regularized-leader (FTRL) approach to obtain improved regret bounds for strongly convex and exp-concave losses. These studies assume that all delayed feedback can eventually be observed, without imposing tracking-capacity constraints.

More closely related to our work, reference [38] introduced capacity-constrained delayed online learning for finite-action problems, characterizing the trade-off between tracking capacity and regret through tracking policies. This framework was subsequently extended to OCO in [39] using randomized scheduling and delayed FTRL. However, these works focus on regret minimization, which does not directly imply long-term risk-control guarantees [40]. The tracking policy proposed in this paper also builds on randomized scheduling, as in [39], but it targets long-term risk control rather than regret minimization.

Overall, existing OCP and OCRC schemes with delayed feedback do not account for a limited tracking capacity, while capacity-constrained OCO methods do not provide risk guarantees. This work addresses both limitations.

## C. Main Contributions

The main contributions of this paper are summarized as follows:

Capacity-aware OCRC: We propose capacity-aware OCRC (CA-OCRC), an online framework for long-term risk control under delayed feedback and finite tracking capacity. CA-OCRC consists of two components: (i) a tracking policy and (ii) a threshold update for the prediction sets. The tracking policy samples a random proxy delay for each round, and it tracks the round, if capacity permits, until its feedback is received or its proxy delay expires. The threshold update strategy employs a two-level inverse probability weighting mechanism to compensate for permanently missing feedback.

Risk guarantees: We first derive bounds on the long-term risk that hold for any delay (Theorem 1), and then extend the analysis to stochastic delays, yielding distribution-dependent bounds (Theorem 2). Both bounds reveal a trade-off in the design of the tracking policy: larger tracking probabilities yield more feedback, but they also increase the tracking load, and hence the probability of capacity saturation.

Proxy delay distribution design: Building on the derived bounds, we optimize the proxy delay distribution. Under arbitrary delays, we show that the risk bound is minimized by a Bernoulli-type proxy delay distribution, under which CA-OCRC tracks each round until the maximum delay with a fixed probability, and does not track it otherwise. Under stochastic delays, we obtain an optimized proxy delay distribution in closed form based on the derived bound. With this distribution, CA-OCRC keeps tracking pending rounds by accounting on which feedback signals are more likely to arrive.

Scaling laws: With the optimized proxy delay distributions, the risk gap of CA-OCRC decays with the time horizon T as $\mathcal { O } ( ( d _ { \operatorname* { m a x } } / T ) ^ { K / ( K + 1 ) } )$ under arbitrary delays, and as $\mathcal { O } ( ( \mathbb { E } [ d _ { t } ] / T ) ^ { K / ( K + 1 ) } )$ under stochastic delays, for any capacity $K \geq 1$ , where $d _ { \mathrm { m a x } }$ is the maximum delay and $\mathbb { E } [ d _ { t } ]$ is the mean delay. Compared with the rate $\mathcal { O } ( d _ { \mathrm { m a x } } / T )$ of OCRC (see Sec. III-A), which tracks all pending feedback, the capacity constraint thus reduces the exponent of the convergence rate from 1 to $K / ( K + 1 )$ . In particular, a capacity of the order of ln $( T / d _ { \mathrm { m a x } } )$ , which may be much smaller than $d _ { \operatorname* { m a x } } .$ , ensures CA-OCRC to obtain the same rate as OCRC.

Experimental validation: We validate CA-OCRC on tumor segmentation and open-domain question answering under stochastic delays, as well as on hierarchical image classification and beam selection under temporally correlated delays. The results show that CA-OCRC achieves risk levels close to the target, with faster convergence as the capacity increases, while keeping the tracking load below the capacity. Compared with capacity-constrained benchmarks, CA-OCRC achieves more reliable risk control, while the distribution-aware proxy design further improves capacity utilization, convergence speed, and prediction-set efficiency under stochastic delays.

## D. Organization

The remainder of this paper is organized as follows. Sec. II formulates the sequential prediction problem with delayed feedback and finite tracking capacity. Sec. III reviews OCRC and intermittent OCRC (I-OCRC). Sec. IV introduces CA-OCRC, and Secs. V and VI give risk guarantees and proxy delay designs for arbitrary and stochastic delays, respectively. Sec. VII presents numerical results, and Sec. VIII concludes.

## II. PROBLEM FORMULATION

In this section, we formulate the problem of sequential prediction with delayed feedback under finite tracking capacity, which is outlined in Fig. 1.

## A. Setting

We consider sequential prediction over rounds $t = 1 , 2 , \dots , T$ . At each round t, the environment generates an input-label pair $( X _ { t } , Y _ { t } ) \in \mathcal { X } \times \mathcal { Y }$ , where and denote the input and label spaces, respectively. The data sequence $\{ ( X _ { t } , Y _ { t } ) \} _ { t = 1 } ^ { T }$ is arbitrary, and we make no probabilistic assumptions such as stationarity or exchangeability.

The predictor has access to a pre-trained model $f ( \cdot )$ . At the beginning of round t, the predictor observes the current input $X _ { t }$ and, using the pre-trained model $f ( \cdot )$ , returns a prediction set $\mathcal { C } _ { t } \subseteq \mathcal { V }$ for the unknown label $Y _ { t }$ . The performance of the set prediction is measured by the loss

$$
L _ { t } = \mathcal { L } ( \mathcal { C } _ { t } , Y _ { t } ) ,\tag{1}
$$

where the loss function $\mathcal { L } ( \mathcal { C } , y ) \in [ 0 , 1 ]$ quantifies the cost incurred by the set $\mathcal { C } \subseteq \mathcal { V }$ when the true label is y. A standard example is the miscoverage indicator

$$
\mathcal { L } ( \mathcal { C } _ { t } , Y _ { t } ) = \mathbb { 1 } \{ Y _ { t } \notin \mathcal { C } _ { t } \} ,\tag{2}
$$

which equals one when the true label $Y _ { t }$ is not contained in the set $\mathcal { C } _ { t } .$ , and zero otherwise. Other examples include the false negative rate (FNR) in multi-label classification problems, and weighted variants of miscoverage [9], [15], [16].

Prior art assumed that feedback about the loss $L _ { t }$ was available with zero delay [12], [13], [15], [16], i.e., at the end of round $t ,$ though it could possibly be lost [28], [29] or distorted [41]. In contrast, we assume that the predictor can access the loss $L _ { t } \colon ( i )$ after $d _ { t } \in \mathbb { Z } _ { \geq 0 }$ time steps, i.e., at the end of round $t + d _ { t } ;$ and (ii) only provided that feedback $L _ { t }$ was tracked throughout the time interval $[ t , t + d _ { t } ]$ . As extreme cases, setting $d _ { t } ~ = ~ 0$ recovers the standard scenario with feedback available at the end of the current round, while the delays $d _ { t } \geq T - t$ represent a feedback free scenario in which feedback does not become available before the end of the considered time horizon $T .$

At any given round t, the predictor can only track at most $K \in \mathbb { Z } _ { > 0 }$ of the feedback signals $\{ L _ { t ^ { \prime } } \}$ corresponding to past time instants $t ^ { \prime } \in \{ 1 , \ldots , t \}$ . As mentioned, feedback signals must be tracked continuously in order to be eligible to be received after the corresponding delay. We focus on the nontrivial case $K \leq d _ { \operatorname* { m a x } } .$ , where $d _ { \mathrm { m a x } }$ is the largest possible feedback delay. Settings with $K \geq d _ { \mathrm { m a x } } + 1$ , in fact, permit tracking all the feedback signals, since at any round t only the $d _ { \operatorname* { m a x } } + 1$ rounds $t - d _ { \operatorname* { m a x } } , \ldots , t$ can have pending feedback.

The tracked feedback signals at the end of round t are defined by a tracking set $K _ { t } \subseteq \{ 1 , . . . , t \}$ which evolves recursively over time via eviction and admission policies to be optimized. To elaborate, initialize the set as $\kappa _ { 0 } = \varnothing$ . As shown in Fig. 1, at any round $t ,$ the tracking set evolves from $\boldsymbol { \mathcal { K } } _ { t - 1 }$ to $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ through the following steps.

1) Eviction: The predictor evicts a subset $\mathcal { E } _ { t } \subseteq { \mathcal { K } } _ { t - }$ <sub>1</sub> of previously tracked rounds, yielding the post-eviction set

$$
\begin{array} { r } { { \cal K } _ { t } ^ { - } = { \cal K } _ { t - 1 } \backslash \mathcal { E } _ { t } , \mathrm { w i t h } \mathcal { E } _ { t } \subseteq { \cal K } _ { t - 1 } . } \end{array}\tag{3}
$$

Once a round $i \in \mathcal { E } _ { t }$ is evicted, its feedback $L _ { i }$ can no longer be received by the predictor.

2) Admission: If capacity permits, i.e., if $| \mathcal { K } _ { t } ^ { - } | < K$ , the predictor can next decide to track the current round t. Formally, the admission set at round $t , A _ { t }$ , is empty if capacity is saturated, i.e., if $| \mathcal { K } _ { t } ^ { - } | = K$ , while it may include t otherwise. Accordingly, the post-eviction tracking set ${ \mathrm { K } } _ { t } ^ { - }$ is updated as

$$
\begin{array} { r } { \mathcal { K } _ { t } ^ { + } = \mathcal { K } _ { t } ^ { - } \cup \mathcal { A } _ { t } \mathrm { ~ w i t h ~ } \mathcal { A } _ { t } \in \{ \varnothing , \{ t \} \} , } \end{array}\tag{4}
$$

where $\mathcal { A } _ { t } = \emptyset \mathrm { ~ i f ~ } | \mathcal { K } _ { t } ^ { - } | = K$

3) Feedback Reception: At the end of round t, the predictor receives all tracked feedback signals $L _ { t ^ { \prime } }$ , with $t ^ { \prime } \in { \cal K } _ { t } ^ { + }$ , for which the delay $d _ { t ^ { \prime } }$ satisfies the equality $t ^ { \prime } + d _ { t ^ { \prime } } = t$ . Accordingly, the set of observed feedback signals is

$$
\mathcal { O } _ { t } = \{ i \leq t : i + d _ { i } = t , \ \mathrm { a n d } \ i \in \mathcal { K } _ { t } ^ { + } \} .\tag{5}
$$

Note that, by this convention, feedback $L _ { t }$ with a zero delay $d _ { t } = 0$ is observed only if round t is admitted, i.e., if $A _ { t } = \{ t \}$ . The rounds in ${ \mathcal { O } } _ { t }$ are then removed from the tracking set $\mathcal { K } _ { t } ^ { + }$ yielding the updated tracking set

$$
{ \cal K } _ { t } = { \cal K } _ { t } ^ { + } \setminus \mathcal { O } _ { t } .\tag{6}
$$

Overall, at the beginning of round t, the predictor has access to the tracking set $\boldsymbol { \mathcal { K } } _ { t - 1 }$ , to the feedback signals $\mathcal { O } _ { 1 : t - 1 } = \bigcup _ { t ^ { \prime } = 1 } ^ { t - 1 } \mathcal { O } _ { t ^ { \prime } }$ received so far, as well as to the input $X _ { t }$ and to the prediction $f ( X _ { t } )$ . The predictor can use this information in an arbitrary way to design the prediction set $\mathcal { C } _ { t } .$ , of the eviction set $\mathcal { E } _ { t }$ , and of the admission set $\boldsymbol { A } _ { t }$

## B. Delay Models

We consider the following two alternative models for the feedback delays.

## Assumption 1. The feedback delays satisfy one of the following alternatives:

(a) Arbitrary Delays: The feedback delays $d _ { 1 } , \ldots , d _ { T }$ are arbitrary, while satisfying the upper bound

$$
0 \leq d _ { t } \leq d _ { \operatorname* { m a x } } , t = 1 , \ldots , T ,\tag{7}
$$

for a known constant $d _ { \mathrm { m a x } } \geq 0$

(b) Stochastic Delays: The feedback delays $d _ { 1 } , \ldots , d _ { T }$ are independent and identically distributed, i.e.,

$$
d _ { t } \sim G ,\tag{8}
$$

according to a survival function

$$
G ( a ) = \operatorname* { P r } _ { G } [ d _ { t } \geq a ] , f o r \ a \in \{ 0 , 1 , \ldots , d _ { \operatorname* { m a x } } \} .\tag{9}
$$

## C. Design Goal

For a target risk level $\alpha \in ( 0 , 1 )$ , the goal is to jointly design a policy that maps the observations $\mathcal { O } _ { 1 : t - 1 }$ , the input $X _ { t }$ , and the prediction $f ( X _ { t } )$ to a prediction set $\mathcal { C } _ { t } .$ , an eviction set $\mathcal { E } _ { t } \subseteq \mathcal { K } _ { t - 1 }$ in (3), and an admission set $\boldsymbol { A } _ { t }$ in (4). The design objective is to ensure the long-term risk gap condition

$$
\operatorname* { l i m } _ { T \to \infty } \left\{ \operatorname { R i s k } \mathrm { G a p } ( T ) = \left| \mathbb { E } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } L _ { t } \right] - \alpha \right| \right\} = 0 ,\tag{10}
$$

where the expectation is taken with respect to the internal randomness of the policy under Assumption 1(a), and also with respect to the stochastic delays under Assumption 1(b). Note that the long-term risk in (10) is evaluated over all prediction rounds, including rounds whose labels are never observed by the predictor.

The following lemma connects the two settings with arbitrary and stochastic delays.

Lemma 1. Suppose that, for some deterministic constant ε, the upper bound

$$
\mathrm { R i s k G a p } ( T ) \le \varepsilon\tag{11}
$$

is achievable under Assumption 1(a). Then, the same bound (11) also holds under Assumption 1(b).

Lemma 1 shows that a uniform guarantee for arbitrary bounded delays automatically yields the same guarantee under bounded stochastic delays. In fact, the same policy that ensures the upper bound (11) under Assumption 1(a) automatically also satisfies the bound (11) under Assumption 1(b). That said, as we will see in Sec. VI, knowledge of the delay distribution G under Assumption 1(b) can probably lead to more efficient, tailored policies, yielding distributiondependent bounds on the risk gap.

## III. BACKGROUND: ONLINE CONFORMAL RISK CONTROL

In this section, we review two existing solutions that address extreme special cases of the setting described in Sec. II, namely online conformal risk control (OCRC) [15], [16], which assumes an infinite-capacity predictor and can handle arbitrary finite delays, and intermittent OCRC (I-OCRC), i.e., the direct extension of intermittent OCP [28], [29] to general losses, which applies to a unit-capacity predictor under stochastic delays.

## A. Online Conformal Risk Control

OCRC assumes no capacity constraints, i.e., $K \geq T$ . Accordingly, it tracks all rounds, carrying out no evictions and admitting every current round, i.e., $\mathcal { E } _ { t } = \emptyset$ and $A _ { t } = \{ t \}$ for all rounds t. The prediction set $\mathcal { C } _ { t }$ is constructed as [12], [13]

$$
\mathcal { C } _ { t } ( X _ { t } ; r _ { t } ) = \{ y \in \mathcal { y } : { S _ { t } } ( X _ { t } , y ) \leq r _ { t } \} ,\tag{12}
$$

where $S _ { t } ( X _ { t } , y ) : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ is a score based on the pre-trained model $f ( \cdot )$ , with a larger score indicating poorer agreement between a candidate label y and the prediction $f ( X _ { t } )$ , and $r _ { t }$ is a threshold. Typical examples include the residual score $S _ { t } ( X _ { t } , y ) = | y - f ( X _ { t } ) |$ for regression and $S _ { t } ( X _ { t } , y ) = 1 - \log f _ { y } ( X _ { t } )$ for classification, where $f _ { y } ( X _ { t } )$ is the predictive probability for label y given input $X _ { t }$ . Starting from an initial threshold $r _ { t } ,$ , the threshold is updated as

$$
r _ { t + 1 } = r _ { t } + \eta \sum _ { t ^ { \prime } : t ^ { \prime } + d _ { t ^ { \prime } } = t } ( L _ { t ^ { \prime } } - \alpha ) ,\tag{13}
$$

where $\eta > 0$ is a learning rate. Note that the original analysis in [12], [13], [15], [16] assumed zero delays, so that the update reduces to $r _ { t + 1 } = r _ { t } + \eta ( L _ { t } - \alpha )$

The update (13) is a direct extension of the standard OCRC [15], [16] to delayed feedback, and its analysis relies on the following assumptions [12], [13], [15].

Assumption 2. The score function satisfies $S _ { t } ( X _ { t } , y ) \in [ 0 , B ]$ for every label $y \in \mathcal { V }$ and every round $t ,$ where $B < \infty$

Assumption 3. The loss function satisfies $\mathcal { L } ( \mathcal { y } , \boldsymbol { y } ) = 0$ and ${ \mathcal { L } } ( \emptyset , y ) = 1$ for every label $y \in \mathcal { V } _ { : }$ $i . e .$ , the full set $\mathcal { C } _ { t } = \mathcal { V }$ incurs no loss, while the empty set $\mathcal { C } _ { t } = \emptyset$ incurs the maximum loss.

By Assumption 2, the prediction set (12) equals the full set  for $r _ { t } \ge B$ and the empty set $\emptyset$ for $r _ { t } < 0$ , so that, by Assumption 3, the corresponding losses equal 0 and 1, respectively, irrespective of the true label $Y _ { t }$ . Under these assumptions, the following guarantee can be proved in a manner similar to [12], [13], [15].

Lemma 2. Under Assumption 1(a), Assumption 2, and Assumption 3 with $r _ { 1 } \in [ 0 , B ]$ , the risk gap of OCRC satisfies the upper bound

$$
\operatorname { R i s k G a p } ( T ) \leq \frac { B } { \eta T } + \frac { d _ { \operatorname* { m a x } } + 1 } { T }\tag{14}
$$

for any sequence of data points and delays.

Proof. See Supplementary Material Appendix A.

Accordingly, the risk gap of OCRC vanishes as $T \to \infty$ whenever the maximum delay $d _ { \mathrm { m a x } }$ is constant or grows sublinearly with T, i.e., $d _ { \operatorname* { m a x } } = o ( T )$

## B. Intermittent OCRC

At the opposite extreme, consider a setting with unit tracking capacity, i.e., $K = 1$ . The eviction and admission sets are chosen as $\mathcal { E } _ { t } = \mathcal { K } _ { t - 1 }$ and $A _ { t } = \{ t \}$ , respectively, so that only the current round is tracked. Under Assumption 1(b), the current feedback is thus received with probability

$$
\operatorname* { P r } _ { G } [ d _ { t } = 0 ] = G ( 0 ) - G ( 1 )\tag{15}
$$

and is otherwise lost. I-OCRC produces the prediction set (12) with the threshold updated as [28], [29]

$$
r _ { t + 1 } = r _ { t } + \eta \frac { \mathbb { 1 } \{ d _ { t } = 0 \} } { G ( 0 ) - G ( 1 ) } ( L _ { t } - \alpha ) .\tag{16}
$$

The factor $( G ( 0 ) - G ( 1 ) ) ^ { - 1 } = ( \operatorname* { P r } _ { G } [ d _ { t } = 0 ] ) ^ { - 1 }$ applies a form of inverse probability weighting [42], accounting for the probability of feedback loss.

Lemma 3 ([28], [29]). Under Assumption $I ( b )$ , Assumption 2, and Assumption 3 with $r _ { 1 } \in [ 0 , B ]$ the risk gap of I-OCRC is upper bounded by

$$
\mathrm { R i s k } \mathrm { G a p } ( T ) \leq \frac { B } { \eta T } + \frac { 1 } { ( G ( 0 ) - G ( 1 ) ) T } .\tag{17}
$$

Proof. See Supplementary Material Appendix A, where the analysis in [28], [29] is extended to general losses. □

Accordingly, the risk gap of I-OCRC vanishes as $T \to \infty$ whenever immediate feedback occurs with a positive probability, i.e., $\mathrm { P r } _ { G } [ d _ { t } = 0 ] > 0$ , although the bound (17) degrades as $\mathrm { P r } _ { G } [ d _ { t } = 0 ]$ decreases.

## IV. CAPACITY-AWARE OCRC

In this section, we introduce capacity-aware online conformal risk control (CA-OCRC), a policy that can satisfy the long-term risk requirement (10) for arbitrary values of the capacity K. CA-OCRC is summarized in Algorithm 1 and detailed next.

Algorithm 1: Capacity-Aware OCRC (CA-OCRC)   
Input: Target risk level α, learning rate η, capacity ${ \overline { { K } } } ,$ and proxy delay distribution ${ \overline { { F } } } .$   
1 Initialize $r _ { 1 } = 0 , \bar { d } = \mathbf { 0 } _ { T }$ , and $\kappa _ { 0 } = \emptyset$   
2 for $t = 1 , 2 , \dots , T$ do   
3 Observe $X _ { t }$ and return $\mathcal { C } _ { t } ( X _ { t } ; r _ { t } )$ via (12)   
4 Set $\mathcal { E } _ { t } = \{ k \in \mathcal { K } _ { t - 1 } : k + \bar { d } _ { k } < t \}$ , where ${ \bar { d } } _ { k } = { \bar { d } } [ k ]$   
5 Set $K _ { t } ^ { - } = K _ { t - 1 } \setminus \mathcal { E } _ { t }$   
6 if $\mathcal { C } _ { t } \in \{ \emptyset , \mathcal { D } \}$ then   
7 Calculate $L _ { t } = \mathbb { 1 } \{ \mathcal { C } _ { t } = \emptyset \}$   
8 Set $\bar { d } _ { t } = - 1$ and $\mathbf { \mathcal { A } } _ { t } = \mathbb { \varnothing }$   
9 else   
10 Sample $\bar { d } _ { t } \sim F$ independently and store $\bar { \pmb { d } } [ t ] = \bar { d } _ { t }$   
11 Update $\boldsymbol { A } _ { t }$ via (21)   
12 end   
13 if $| \mathcal { K } _ { t } ^ { - } | < K$ then $K _ { t } ^ { + } = K _ { t } ^ { - } \cup \mathcal { A } _ { t }$ else ${ \cal K } _ { t } ^ { + } = { \cal K } _ { t } ^ { - }$   
14 Observe feedback $\{ L _ { t ^ { \prime } } \} _ { t ^ { \prime } \in \mathcal { O } _ { t } }$   
15 Set ${ \cal K } _ { t } = { \cal K } _ { t } ^ { + } \setminus \mathcal { O } _ { t }$   
16 Update $r _ { t + 1 }$ via (22)   
17 end

## A. Tracking Policy

CA-OCRC implements randomized eviction and admission policies based on proxy delays generated for each round t. Inspired by [39], the general principle is that, if capacity permits, each round t is tracked until the end of round $t + \bar { d } _ { t }$ , where $\bar { d } _ { t }$ is the random proxy delay for round t. The proxy delays $\bar { d } _ { t }$ are drawn independently for each round t, following a survival function $F ( d ) = \mathrm { P r } _ { F } [ \bar { d } _ { t } \geq d ]$ to be designed. The optimization of the function $F ( d )$ is covered in Sec. V and Sec. VI under Assumption 1(a) and Assumption 1(b), respectively.

At each round t, CA-OCRC first checks whether the prediction set $\mathcal { C } _ { t }$ is trivially empty or full, i.e., if $\mathcal { C } _ { t } \in \{ \emptyset , \mathcal { V } \}$ . In either case, there is no need to track feedback $L _ { t } .$ , since, by Assumption 3, the loss $L _ { t }$ is known to equal 1 for $\mathcal { C } _ { t } = \emptyset$ and 0 for $\mathcal { C } _ { t } = \mathcal { y }$ . In such cases, a proxy delay $\bar { d } _ { t } = - 1$ is selected, indicating that round t is not tracked:

$$
\bar { d } _ { t } = - 1 \mathrm { ~ i f ~ } \mathcal { C } _ { t } \in \{ \varnothing , \mathcal { V } \} .\tag{18}
$$

For a nontrivial prediction set $\mathcal { C } _ { t } \not \in \{ \emptyset , \mathcal { V } \}$ , CA-OCRC instead samples the random proxy delay $d _ { t } ,$ , following the proxy delay distribution F, which is supported on the set $\{ - 1 , 0 , 1 , \dots , \bar { d } _ { \mathrm { m a x } } \}$ where $\bar { d } _ { \mathrm { m a x } } \leq d _ { \mathrm { m a x } }$ is the maximum proxy delay:

$$
\bar { d } _ { t } \sim F \ \mathrm { i f } \ C _ { t } \not \in \{ \emptyset , { \mathcal { V } } \} .\tag{19}
$$

Based on the proxy delays, CA-OCRC evicts the tracked rounds whose proxy delays have expired, i.e.,

$$
\mathcal { E } _ { t } = \{ i \in \mathcal { K } _ { t - 1 } : i + \bar { d } _ { i } < t \} ,\tag{20}
$$

and selects round t for admission only if its proxy delay is nonnegative, i.e.,

$$
\begin{array} { r } { \mathcal { A } _ { t } = \left\{ \begin{array} { l l } { \{ t \} , } & { \mathrm { i f ~ } \bar { d } _ { t } \geq 0 , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{21}
$$

## B. Threshold Update

CA-OCRC produces the prediction set (12) with the threshold updated as

$$
r _ { t + 1 } = r _ { t } + \eta \varphi _ { t } ( L _ { t } - \alpha ) + \eta \sum _ { t ^ { \prime } \in \mathcal { O } _ { t } } \frac { L _ { t ^ { \prime } } - \alpha } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { t ^ { \prime } } ) } ,\tag{22}
$$

where $\varphi _ { t } = \mathbb { 1 } \{ \mathcal { C } _ { t } \in \{ \varnothing , \mathcal { V } \} \}$ equals 1 when $\mathcal { C } _ { t }$ is a trivial prediction set, and 0 otherwise; and

$$
q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) = \mathrm { P r } _ { G } [ d _ { t } \leq \bar { d } _ { \mathrm { m a x } } ] = 1 - G ( \bar { d } _ { \mathrm { m a x } } + 1 )\tag{23}
$$

is the probability that the true feedback delay $d _ { t }$ does not exceed the maximum proxy delay $\bar { d } _ { \mathrm { m a x } }$ . Under Assumption 1(a), i.e., deterministic delays, we set $\bar { d } _ { \mathrm { m a x } } = d _ { \mathrm { m a x } }$ and $q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) = 1$ and thus CA-OCRC is applicable under both Assumption 1(a) and Assumption 1(b).

The OCRC update $( L _ { t } - \alpha )$ in (13) is applied only to rounds with trivial prediction sets $\mathcal { C } _ { t } \in \{ \emptyset , \mathcal { V } \}$ . In contrast, for each past received feedback signal $L _ { t ^ { \prime } }$ with $t ^ { \prime } < t ,$ , the update $\left( L _ { t ^ { \prime } } - \alpha \right)$ is applied at each time t and is scaled by the factor $( q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) F ( d _ { t ^ { \prime } } ) ) ^ { - 1 }$ , thus applying a two-level form of inverse probability weighting [42].

This scaling resembles that applied by the I-OCRC update (16). In particular, the factor $q _ { G } ( \bar { d } _ { \operatorname* { m a x } } )$ −<sup>1</sup> compensates for the randomness of the feedback delays as in I-OCRC, since the feedback signals with delays larger than $\bar { d } _ { \mathrm { m a x } }$ are never received. Moreover, the first-level factor $\left( F ( d _ { t ^ { \prime } } ) \right)$ −<sup>1</sup> compensates for the randomness of the proxy delays, which determine the eviction and admission policies (20) and (21).

## V. THEORETICAL RESULTS UNDER ARBITRARY DELAYS

In this section, we analyze the long-term risk of CA-OCRC under Assumption 1(a), i.e., for arbitrary delays with $\bar { d } _ { \mathrm { m a x } } = d _ { \mathrm { m a x } }$ and $q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) = 1$ (see Sec. IV-B). Based on the analysis, we also propose a design method for the proxy delay distribution F.

## A. Risk Guarantee

Since CA-OCRC evicts the rounds whose proxy delays have expired, a round i can be tracked at a later round $t > i$ only if its proxy delay satisfies $\bar { d } _ { i } \geq t - i .$ . In the absence of capacity constraints, the expected number of previous rounds tracked by CA-OCRC at any round is thus upper bounded by the tracking load

$$
\ell _ { F } = \sum _ { a = 1 } ^ { d _ { \mathrm { m a x } } } F ( a ) = \mathbb { E } _ { F } \left[ ( \bar { d } ) _ { + } \right] ,\tag{24}
$$

where $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . To control the probability of capacity saturation, i.e., $| \mathcal { K } _ { t } ^ { - } | = K$ , we impose the design constraint that the tracking load (24) not exceed the capacity K. This design choice is formalized by the following assumption.

Assumption 4. The proxy delay distribution F satisfies the underload condition $\ell _ { F } \leq K$

The following theorem shows that, under Assumption 4, CA-OCRC satisfies the long-term risk condition (10) up to an additive term that accounts for capacity saturation.

Theorem 1 (Risk bound under arbitrary delays). Under Assumption $I ( a ) ,$ , Assumption 2, Assumption 3, and Assumption 4 with $r _ { 1 } \in [ 0 , B ]$ , the risk gap (10) of CA-OCRC satisfies the upper bound

$$
\mathrm { R i s k } \mathrm { G a p } ( T ) \leq \frac { B } { \eta T } + \frac { K + 1 } { F ( d _ { \mathrm { m a x } } ) T } + e ^ { - K h ( \ell _ { F } / K ) }\tag{25}
$$

for any sequence of data points, where

$$
h ( x ) = x - 1 - \ln { x } .\tag{26}
$$

Proof. See Supplementary Material Appendix B.

The first two terms in the bound (25) are similar to the bound (17) for I-OCRC. In fact, the probability $F ( d _ { \mathrm { m a x } } )$ provides a lower bound on the probability of receiving feedback in any given round. In contrast, the last term in the bound (25) accounts for the probability of capacity

saturation, which is absent in I-OCRC since removing any previously tracked round allows the current round to be tracked. By Lemma 1, the bound (25) also holds under the stochastic delays in Assumption 1(b).

## B. Optimization of the Proxy Delay Distribution

We propose to optimize the proxy delay distribution $F$ by minimizing the upper bound in Theorem 1. This yields the problem

$$
\operatorname* { m i n } _ { \{ F ( a ) \} _ { a = - 1 } ^ { d _ { \operatorname* { m a x } } } } \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) T } + e ^ { - K h ( \ell _ { F } / K ) }\tag{27a}
$$

$$
\mathrm { s . t . } \quad \ell _ { F } = \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } F ( a ) \leq K ,\tag{27b}
$$

$$
1 = F ( - 1 ) \geq F ( 0 ) \geq . . . \geq F ( d _ { \operatorname* { m a x } } ) \geq 0 .\tag{27c}
$$

The following proposition describes a solution to problem (27).

Proposition 1. An optimal solution to problem (27) is given by the proxy delay distribution

$$
F ^ { \star } ( a ) = p ^ { \star } , f o r a \in \{ 1 , \ldots , d _ { \operatorname* { m a x } } \} ,\tag{28}
$$

with any $F ^ { \star } ( 0 ) \in [ p ^ { \star } , 1 ]$ and $F ^ { \star } ( - 1 ) = 1$ , where $p ^ { \star }$ is obtained by solving the one-dimensional problem

$$
p ^ { \star } = \operatorname * { a r g m i n } _ { 0 < p \leq K / d _ { \operatorname* { m a x } } } \frac { K + 1 } { p T } + e ^ { - K h ( d _ { \operatorname* { m a x } } p / K ) } .\tag{29}
$$

Proof. See Supplementary Material Appendix C.

Under the optimized proxy delay distribution in Proposition 1, CA-OCRC tracks each round until the maximum possible delay, i.e., $\bar { d } _ { t } = d _ { \operatorname* { m a x } }$ , with probability $p ^ { \star }$ , while not tracking the round otherwise, i.e., $\bar { d } _ { t } = - 1$

## C. Scaling Law

The next result, which follows from Theorem 1 and Proposition 1, shows that, under the optimized proxy delay distribution in Proposition 1, the risk gap of CA-OCRC vanishes as $T \to \infty$ for any fixed capacity K, whenever the maximum delay scales sublinearly with T, i.e., $d _ { \operatorname* { m a x } } = o ( T )$

Corollary 1. Under Assumption $I ( a ) ,$ , Assumption 2, and Assumption 3 with $r _ { 1 } \in [ 0 , B ]$ and $1 \leq K \leq d _ { \operatorname* { m a x } } \leq T$ , the risk gap of CA-OCRC with the proxy delay distribution in Proposition 1 satisfies the upper bound

$$
\mathrm { R i s k G a p } ( T ) \leq \frac { B } { \eta T } + ( 2 e + 1 ) \left( \frac { d _ { \operatorname* { m a x } } } { T } \right) ^ { \frac { K } { K + 1 } } .\tag{30}
$$

Proof. See Supplementary Material Appendix C.

Based on the bound in (30), the main effect of the capacity constraint is to reduce the exponent of the convergence rate from 1 to $K / ( K + 1 )$ . In particular, even with unit-capacity, i.e., $K = 1$ ， CA-OCRC attains the rate $\mathcal { O } ( \sqrt { d _ { \operatorname* { m a x } } / T } )$ under arbitrary delays, while I-OCRC provides no guarantees under Assumption 1(a) (see Lemma 3). Furthermore, a capacity K of the order of $\ln ( T / d _ { \mathrm { m a x } } )$ suffices for CA-OCRC to attain the same rate $\mathcal { O } ( d _ { \mathrm { m a x } } / T )$ as OCRC as per the upper bound in (14). This capacity may be much smaller than the maximum delay $d _ { \mathrm { m a x } }$ , while OCRC requires a capacity $K \geq d _ { \mathrm { m a x } } + 1$ to accommodate all pending feedback.

## VI. THEORETICAL RESULTS UNDER STOCHASTIC DELAYS

In this section, we analyze the long-term risk of CA-OCRC under Assumption 1(b), i.e., for stochastic delays. By Lemma 1, the guarantee in Theorem 1 also applies to this setting. Here, we show that knowledge of the delay distribution G yields a refined guarantee, and that it can be leveraged to tailor the proxy delay distribution F to the feedback delays.

## A. Risk Guarantee

Recall that a tracked round leaves the tracking set as soon as its feedback is received. Accordingly, a round i can be tracked at a later round $t \ > \ i$ only if both its proxy delay and its feedback delay are no smaller than $t - i ,$ i.e., $\bar { d } _ { i } \geq t - i$ and $d _ { i } \geq t - i$ . Since the proxy delays are drawn independently of the feedback delays, in the absence of capacity constraints, the expected number of previous rounds tracked by CA-OCRC at any round is thus upper bounded by the stochastic tracking load

$$
\ell _ { F , G } = \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } F ( a ) G ( a ) = \mathbb { E } _ { F , G } \left[ \operatorname* { m i n } \{ ( \bar { d } _ { t } ) _ { + } , d _ { t } \} \right] .\tag{31}
$$

The stochastic tracking load (31) is no larger than the tracking load $\ell _ { F }$ in (24), with equality holding when all delays equal $d _ { \mathrm { m a x } }$ , i.e., when $G ( a ) = 1$ for $a \in \{ 1 , \dots , d _ { \operatorname* { m a x } } \}$ . As in Sec. V,

we impose the design constraint that the stochastic tracking load not exceed the capacity K, as formalized by the following assumption.

Assumption 5. The proxy delay distribution F satisfies the stochastic underload condition $\ell _ { F , G } \leq$ K.

Assumption 5 is always satisfied if we have $\bar { d } _ { \mathrm { m a x } } < K$ , since in this case, at most $\bar { d } _ { \mathrm { m a x } } < K$ previous rounds can be tracked, and thus capacity saturation never occurs.

The following theorem provides the risk bound under stochastic delays.

Theorem 2 (Risk bound under stochastic delays). Under Assumption $I ( b )$ , Assumption 2, Assumption 3, and Assumption 5 with $r _ { 1 } \in [ 0 , B ]$ , the risk gap (10) of CA-OCRC satisfies the upper bound

$$
\mathrm { R i s k G a p } ( T ) \leq \frac { B } { \eta T } + \frac { R _ { F , G } } { T } + \mathbb { 1 } \left\{ \bar { d } _ { \operatorname* { m a x } } \geq K \right\} e ^ { - K h \left( \ell _ { F , G } / K \right) }\tag{32}
$$

for any sequence of data points, where $h ( \cdot )$ is defined in (26),

$$
R _ { F , G } = \bar { \alpha } \left( \sum _ { a = 0 } ^ { \bar { d } _ { \operatorname* { m a x } } } a \frac { G ( a ) - G ( a + 1 ) } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) } + 2 \right) + \frac { K \bar { \alpha } ^ { 2 } } { 2 ( 1 - \bar { \alpha } ) } Q _ { F , G }\tag{33}
$$

with $\bar { \alpha } = \operatorname* { m a x } \{ \alpha , 1 - \alpha \}$ , and

$$
Q _ { F , G } = \sum _ { a = 0 } ^ { \bar { d } _ { \operatorname* { m a x } } } \frac { G ( a ) - G ( a + 1 ) } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) ^ { 2 } F ( a ) } .\tag{34}
$$

Proof. See Supplementary Material Appendix D.

Compared with the bound (25) for deterministic delays in Theorem 1, the bound (32) contains similar term $\mathbb { 1 } \{ \bar { d } _ { \operatorname* { m a x } } \geq K \} e ^ { - K h ( \ell _ { F , G } / K ) }$ , which bounds the probability of capacity saturation, being equal to zero if $\bar { d } _ { \mathrm { m a x } } < K$ . It also includes the additional term $R _ { F , G } / T$ , which bounds the range of the threshold iterates.

## B. Optimization of the Proxy Delay Distribution

As in Sec. V, we propose to optimize the proxy delay distribution F by minimizing the upper bound in Theorem 2. To this end, we fix the maximum proxy delay $\bar { d } _ { \mathrm { m a x } }$ to a possible value of the feedback delays, i.e., with $\mathrm { P r } _ { G } [ d _ { t } = \bar { d } _ { \mathrm { m a x } } ] > 0$ , so that $q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) > 0$ , and we optimize $\bar { d } _ { \mathrm { m a x } }$

via a grid search over all such values. Removing the terms that do not depend on $F ,$ we obtain the problem

$$
\operatorname* { m i n } _ { \{ F ( a ) \} _ { a = - 1 } ^ { \overline { { d } } _ { \operatorname* { m a x } } } } \frac { R _ { F , G } } { T } + \mathbb { 1 } \{ \bar { d } _ { \operatorname* { m a x } } \geq K \} e ^ { - K h \left( \ell _ { F , G } / K \right) }\tag{35a}
$$

$$
\mathrm { s . t . } \quad \ell _ { F , G } = \sum _ { a = 1 } ^ { \bar { d } _ { \operatorname* { m a x } } } F ( a ) G ( a ) \leq K ,\tag{35b}
$$

$$
1 = F ( - 1 ) \geq F ( 0 ) \geq . . . \geq F ( \bar { d } _ { \operatorname* { m a x } } ) > 0 .\tag{35c}
$$

Unlike problem (27), whose objective depends on $F$ only through the value $F ( d _ { \mathrm { m a x } } )$ and the tracking load $\ell _ { F } ,$ , the objective of problem (35) depends on the entire proxy delay distribution through the quantity $Q _ { F , G }$ in (34). The following proposition describes a solution to problem (35).

Proposition 2. An optimal solution to problem (35) is given by the proxy delay distribution with $F ^ { \star } ( a ) = 1 f o r \ a \in \{ - 1 , 0 , \ldots , \bar { d } _ { \operatorname* { m a x } } \} \ i f \bar { d } _ { \operatorname* { m a x } } < K ;$ and, if $\dot { d } _ { \mathrm { m a x } } \geq K$ , by the proxy delay distribution

$$
F ^ { \star } ( a ) = \left\{ \begin{array} { l l } { 1 , } & { f o r ~ a \in \{ - 1 , 0 \} , } \\ { \operatorname* { m i n } \left\{ 1 , \theta ^ { \star } \sqrt { \kappa ( a ) } \right\} , } & { f o r ~ a \in \{ 1 , \ldots , \bar { d } _ { \operatorname* { m a x } } \} , } \end{array} \right.\tag{36}
$$

where $\theta ^ { \star } > 0$ is obtained by solving problem (35) over the one-parameter family of distributions of the form (36), and

$$
\kappa ( a ) = \operatorname* { m i n } _ { 1 \leq s \leq a } \operatorname* { m a x } _ { a \leq s ^ { \prime } \leq \bar { d } _ { \operatorname* { m a x } } } \frac { G ( s ) - G ( s ^ { \prime } + 1 ) } { \sum _ { i = s } ^ { s ^ { \prime } } G ( i ) } .\tag{37}
$$

Proof. See Supplementary Material Appendix E.

If $K > \bar { d } _ { \mathrm { m a x } }$ , the proxy delay can be set to $\bar { d } _ { t } = \bar { d } _ { \operatorname* { m a x } }$ for every round t with a nontrivial prediction set, since capacity saturation can never occur in this case. In this setting, for $\bar { d } _ { \mathrm { m a x } } = 0$ we have $q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) = G ( 0 ) - G ( 1 )$ , and the CA-OCRC update (22) recovers I-OCRC in (16). Accordingly, CA-OCRC can attain the same performance as I-OCRC, with the bound (32) matching the bound (17) up to constant factors.

If $\bar { d } _ { \mathrm { m a x } } \geq K$ , instead, CA-OCRC with the distribution (36) tracks a pending round at age a, i.e., a rounds after its prediction, with a probability $F ^ { \star } ( a )$ that increases with $\kappa ( a )$ . Intuitively, the ratio $\kappa ( a )$ compares the probability $G ( s ) - G ( s ^ { \prime } + 1 )$ that the feedback of a round arrives between ages s and $s ^ { \prime }$ with the tracking load $\textstyle \sum _ { i = s } ^ { s ^ { \prime } } G ( i )$ that the round contributes over these ages. CA-OCRC thus keeps tracking a round while this ratio is high, and evicts it otherwise to free capacity for new rounds. The parameter $\theta ^ { \star }$ sets the overall tracking level: a larger $\theta ^ { \star }$ yields more feedback, reducing the quantity $Q _ { F , G }$ , at the cost of a larger tracking load $\ell _ { F , G }$

## C. Scaling Law

The next result, which follows from Theorem 2 and Proposition 2, shows that, under stochastic delays, the risk gap of CA-OCRC with the optimized proxy delay distribution is governed by the mean delay $\mathbb { E } _ { G } [ d _ { t } ]$ , rather than by the maximum delay $d _ { \mathrm { m a x } }$

Corollary 2. Under Assumption 1(b), Assumption 2, and Assumption 3 with $r _ { 1 } \in [ 0 , B ]$ , the risk gap of CA-OCRC with the proxy delay distribution in Proposition 2 and the optimized maximum proxy delay satisfies the upper bound

$$
\mathrm { R i s k } \mathrm { G a p } ( T ) \leq \frac { B } { \eta T } + \frac { 2 \bar { \alpha } } { T } + C _ { \alpha } \left( \frac { \mathbb { E } _ { G } [ d _ { t } ] } { T } \right) ^ { \frac { K } { K + 1 } } .\tag{38}
$$

where $\bar { \alpha } = \operatorname* { m a x } \{ \alpha , 1 - \alpha \}$ and $C _ { \alpha } = 1 + \bar { \alpha } + e \bar { \alpha } ^ { 2 } / ( 2 - 2 \bar { \alpha } )$

Proof. See Supplementary Material Appendix E.

By replacing the maximum delay $d _ { \mathrm { m a x } }$ in Corollary 1 with the mean delay $\mathbb { E } _ { G } [ d _ { t } ]$ , Corollary 2 suggests that random delays can yield significantly lower risk gaps as compared to deterministic delays, since the inequality $\mathbb { E } _ { G } [ d _ { t } ] \leq d _ { \operatorname* { m a x } }$ holds.

## VII. NUMERICAL RESULTS

In this section, we evaluate CA-OCRC on tumor segmentation [43], open-domain question answering [9], hierarchical image classification [44], and wireless beam selection [16] tasks.

## A. Setting

1) Datasets: We consider four different experimental settings.

Tumor Segmentation: We adopt the tumor segmentation setting of [43], in which the input $X _ { t }$ is a medical image, and the label $Y _ { t }$ is the nonempty subset of pixels of the image that belong to a tumor. As in [43], the dataset is a compilation of samples from several open-source online repositories, namely Kvasir, CVC-300, CVC-ColonDB, CVC-ClinicDB, and ETIS-LaribDB. The pre-trained model $f ( \cdot )$ is the PraNet segmentation model [45], which outputs the probability $f _ { y } ( X _ { t } ) \in [ 0 , 1 ]$ that each pixel $y \in \mathcal { V }$ belongs to a tumor. Using the score $S _ { t } ( X _ { t } , y ) = 1 - f _ { y } ( X _ { t } )$ which satisfies Assumption 2 with $B = 1$ , the prediction set (12) is the segmentation mask ${ \mathcal { C } } _ { t } ( X _ { t } ; r _ { t } ) = \{ y \in \mathcal { V } : f _ { y } ( X _ { t } ) \geq 1 - r _ { t } \}$ . Performance is measured by the false negative rate (FNR) loss $\mathcal { L } ( \mathcal { C } _ { t } , Y _ { t } ) = 1 - | \mathcal { C } _ { t } \cap Y _ { t } | / | Y _ { t } |$ , which satisfies Assumption 3 since the label $Y _ { t }$ is nonempty.

Open-Domain Question Answering: We then consider the open-domain question answering setting of [9], in which the input $X _ { t } \in \mathcal { X }$ is a natural-language question, and the label $Y _ { t } \subset \mathcal { V }$ is the set of ground-truth answers to the question. The input domain $\mathcal { X }$ is the set of all token sequences, while the output domain is the set of all token sequences of length at most m. As in [9], we use the 3610 test questions of the Natural Questions (NQ) dataset [20], according to the split released with the dense passage retrieval (DPR) package [46]. The pre-trained model $f ( \cdot )$ is the DPR retriever-reader pipeline, which returns a confidence score $f _ { y } ( X _ { t } ) \in [ - M / 2 , M / 2 ]$ , with $M = 8 0$ , for each answer $y$ in a set $\hat { \mathcal { V } } _ { t } \subset \mathcal { V }$ of candidate answers. Using the score $S _ { t } ( X _ { t } , y ) =$ $0 . 5 - f _ { y } ( X _ { t } ) / M$ for $\boldsymbol y \in \hat { \mathcal y } _ { t }$ and $S _ { t } ( X _ { t } , y ) = 1$ otherwise, which satisfies Assumption 2 with $B = 1$ , the prediction set (12) is given by $\mathcal { C } _ { t } ( X _ { t } ; r _ { t } ) = \{ y \in \hat { \mathcal { D } } _ { t } : f _ { y } ( X _ { t } ) \geq ( 0 . 5 - r _ { t } ) F \}$ for $r _ { t } < 1$ , and by $\mathcal { C } _ { t } ( X _ { t } ; r _ { t } ) = \mathcal { Y }$ for $r _ { t } \ge 1$ . Performance is measured by the smallest F1 loss for elements of the set $\mathcal { C } _ { t } ,$ i.e., $\begin{array} { r } { \mathcal { L } ( \mathcal { C } _ { t } , Y _ { t } ) = 1 - \operatorname* { m a x } _ { y \in \mathcal { C } _ { t } , y ^ { \prime } \in Y _ { t } } \operatorname { F 1 } ( y , y ^ { \prime } ) } \end{array}$ , where $\mathrm { F 1 } ( y , y ^ { \prime } ) \in [ 0 , 1 ]$ is the token-level F1 score between the answers y and $y ^ { \prime }$ [47]. Note that we have $\mathcal { L } ( \emptyset , Y _ { t } ) = 1$ and, since $Y _ { t }$ is a nonempty subset of $\mathcal { V } , \mathcal { L } ( \mathcal { V } , Y _ { t } ) = 0$ , which satisfies Assumption 3.

Hierarchical Image Classification: We also adopt the hierarchical image classification setting of [44], in which the input $X _ { t }$ is an image from the ImageNet validation set [48], the output domain $\mathcal { V }$ is the set of 1000 classes, and the label $Y _ { t } \in \mathcal { V }$ is the ground-truth class. The classes are the leaves of a tree of height $H = 1 4$ in the WordNet hierarchy [49]. The prediction set contains all the classes below a node of the tree, so that nodes closer to the root yield larger prediction sets. The pre-trained model $f ( \cdot )$ is a ResNet-152 [50], which outputs the probability $f _ { y } ( X _ { t } ) \in [ 0 , 1 ]$ of each class $y \in \mathcal { V }$ . We denote by $\hat { y } _ { t }$ the top-1 class, and by $f _ { v } ( X _ { t } )$ the total probability of the classes below a node v. We use the score $S _ { t } ( X _ { t } , y ) = f _ { v _ { t } ( y ) } ( X _ { t } )$ for $y \ne \hat { y } _ { t }$ where $v _ { t } ( y )$ is the lowest common ancestor of the classes y and $\hat { y } _ { t }$ , and $S _ { t } ( X _ { t } , \hat { y } _ { t } ) = 0$ , which satisfies Assumption 2 with $B = 1$ . Accordingly, for $r _ { t } \ge 0$ , the prediction set (12) contains all the classes below the node $v _ { t }$ , with $\boldsymbol v _ { t } = \hat { \boldsymbol y } _ { t }$ if no such node exists. Performance is measured by the hierarchical loss $\mathcal { L } ( \mathcal { C } _ { t } , Y _ { t } ) = \mathrm { d i s t } ( v _ { t } , Y _ { t } ) / H$ , where dist $( v _ { t } , Y _ { t } )$ is the number of levels from the node $v _ { t }$ up to its lowest common ancestor with the class $Y _ { t }$ . Note that we set $\mathcal { L } ( \emptyset , Y _ { t } ) = 1$ and, since the root is an ancestor of every class, we have $\mathcal { L } ( \mathcal { V } , Y _ { t } ) = 0$ , which satisfies Assumption 3.

Beam Selection: We finally consider the wireless beam selection setting of [16], in which a base station with 8 antennas selects a set of candidate beams from a discrete Fourier transform codebook [16] of 11 beams, so that the output domain is $\mathcal { V } = \{ 1 , \ldots , 1 1 \}$ . The input $X _ { t }$ is the position of the user, and the pre-trained model $f ( \cdot )$ is a neural network that outputs a probability $f _ { b } ( X _ { t } ) \in [ 0 , 1 ]$ for each beam $b \in \mathcal { V }$ . With the score $S _ { t } ( X _ { t } , b ) = 1 - f _ { b } ( X _ { t } )$ , which satisfies Assumption 2 with $B = 1$ , the prediction set (12) is given by $\mathcal { C } _ { t } ( X _ { t } ; r _ { t } ) = \{ b \in \mathcal { Y } : f _ { b } ( X _ { t } ) \geq 1 -$ $r _ { t } \}$ . Performance is measured by the SNR loss $\begin{array} { r } { \mathcal { L } ( \mathcal { C } _ { t } , Y _ { t } ) = { 1 } - \operatorname* { m a x } _ { b \in \mathcal { C } _ { t } } { \mathrm { S N R } _ { t , b } } / \operatorname* { m a x } _ { b ^ { \prime } \in \mathcal { V } } { \mathrm { S N R } _ { t , b ^ { \prime } } } } \end{array}$ where $\mathrm { S N R } _ { t , b }$ is the signal-to-noise ratio of beam b at round t. Note that we have $\mathcal { L } ( \emptyset , Y _ { t } ) = 1$ and $\mathcal { L } ( \mathcal { V } , Y _ { t } ) = 0$ , which satisfies Assumption 3.

2) Delay Model: We adopt an autoregressive gamma model for the delays. Gamma distributions including the exponential and Erlang distributions provide a standard model for waiting times in queueing systems [51]–[53], packet delays in communication networks [54]–[56], and conversion delays in online advertising [24], [57]. Specifically, as shown in Fig. 1(b), we assume that round t spans the continuous-time interval $[ t , t + 1 )$ , and the feedback $L _ { t }$ arrives at time $t + D _ { t }$ , where $D _ { t } \geq 0$ is the underlying continuous-time gamma-distributed delay. Therefore, any feedback $L _ { t }$ arriving during round $t + \left\lfloor D _ { t } \right\rfloor$ can only be used at the end of round $t + \left\lfloor D _ { t } \right\rfloor$

Formally, we adopt a beta-gamma first-order autoregressive (AR(1)) process [58], obtaining the discrete-time delays $d _ { t }$ by quantizing the continuous-time delays $D _ { t }$ and clipping them at $d _ { \operatorname* { m a x } } .$ yielding

$$
\left\{ \begin{array} { l l } { D _ { t + 1 } = U _ { t + 1 } D _ { t } + \xi _ { t + 1 } , } \\ { d _ { t } = \operatorname* { m i n } \{ \lfloor D _ { t } \rfloor , d _ { \operatorname* { m a x } } \} , } \end{array} \right.\tag{39}
$$

where $D _ { 1 } \sim \mathrm { G a m m a } ( \nu , \theta )$ follows the gamma distribution with shape $\nu > 0$ and scale $\theta > 0$ and the random variables $U _ { t } \sim \mathrm { B e t a } ( \nu \rho , \nu ( 1 - \rho ) )$ and $\xi _ { t } \sim \operatorname { G a m m a } ( \nu ( 1 - \rho ) , \theta )$ , for $t \geq 2$ , are mutually independent and independent of $D _ { 1 }$ . The process (39) preserves the gamma marginal distribution, i.e., $D _ { t } \sim \mathrm { G a m m a } ( \nu , \theta )$ for all rounds t, while $\rho \in [ 0 , 1 )$ is the correlation coefficient between consecutive continuous-time delays [58].

Accordingly, the discrete-time delays $d _ { t }$ have the survival function

$$
G ( a ) = \mathrm { P r } [ d _ { t } \geq a ] = 1 - \Psi ( a ) ,\tag{40}
$$

for $a \in \{ 0 , 1 , \dots , d _ { \operatorname* { m a x } } \}$ and all rounds $t ,$ where $\Psi ( \cdot )$ is the cumulative distribution function of Gamma $( \nu , \theta )$ . The shape parameter ν controls the skewness of the gamma distribution, with $\nu = 1$ corresponding to the exponential distribution and integer values of $\nu$ yielding the Erlang distribution, while the scale parameter θ stretches the distribution without changing its shape [58].

![](images/dfa849115ec93b83e7c1983b96cdc77e2add2b61c71084354e355882f48521e0.jpg)  
(a)

![](images/25e7899ee4c420ebbcaeaf2d3c581cd4dbe0522c835b59076b3d3a1c5e597e7c.jpg)  
(b)

![](images/3c80a5fd0ec695536fe5bd239eee5bd31e2b2ef361cd90009394ca1f24cc6ed0.jpg)  
(c)  
Fig. 2. FNR as a function of the time horizon T for the tumor segmentation task with $\nu = 4 , \theta = 8 , \rho = 0 , d _ { \mathrm { { m a x } } } = 1 5 0 \mathrm { { , } }$ and target level $\alpha = 0 . 1$ (dashed line), for capacities (a) K = 1, (b) $K = 2 ,$ and (c) $K = 4 .$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.

The mean and variance of the delays are $\nu \theta$ and $\nu \theta ^ { 2 }$ , respectively. For $\rho = 0$ , we set $U _ { t } = 0$ and thus the delays are independent and identically distributed (i.i.d.) as in Assumption 1(b), while for $\rho > 0$ , they are identically distributed but temporally correlated, and realization from process (39) hence satisfy only Assumption 1(a).

3) Benchmarks: We compare CA-OCRC against the following schemes:

The original OCRC [15], [16] without the capacity constraint, i.e., $K = \infty$ (see Sec. III-A), which tracks all rounds, and thus serves as an ideal benchmark;

FIFO-OCRC, which tracks each round t until the end of round $t + K - 1$ , thus managing the tracking set as a first-in first-out (FIFO) queue of the K most recent rounds. Under i.i.d. delays, FIFO-OCRC updates the threshold by scaling received feedback signals with factor $( \operatorname* { P r } _ { G } [ d _ { t } \leq K - 1 ] ) ^ { - 1 }$ , thus extending I-OCRC (see Sec. III-B) from unit capacity to an arbitrary capacity K. Under temporally correlated delays, it instead applies the OCRC update (13) to the received feedback signals.

The evaluated variants of CA-OCRC, which differ in the proxy delay distribution $F ,$ are described in Sec. VII-B for i.i.d. delays and in Sec. VII-D for temporally correlated delays.

4) Performance Metrics: Along with the long-term risk in (10) and the set size (see the dataset descriptions in VII-A1), we report the tracking load, which is given by

$$
\mathrm { T r a c k i n g L o a d } ( T ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | { \cal K } _ { t } ^ { + } | ,\tag{41}
$$

with $\mathcal { K } _ { t } ^ { + }$ in (4). By construction, the tracking load of the capacity-constrained schemes is no larger than the capacity K, while, for OCRC, the set $\boldsymbol { { \cal K } } _ { t } ^ { + }$ contains all rounds with pending feedback.

![](images/4aab348a1841d13783e1dde98bc42b3a589ef6e7bca1238e141b8bee38f01cfd.jpg)  
(a)

![](images/a2afed60ad9aae73e88c6c14ace8ee0b6f96f5b605859e5ea0583b485a4d01f4.jpg)  
(b)

![](images/05a6ec8d9f81342296ee7059fcadf5511d0501ccb2ef023dde7a363fe7707340.jpg)  
(c)  
Fig. 3. (a) FNR, (b) set size, and (c) tracking load as a function of the capacity K for the tumor segmentation task with $\nu = 4 ,$ $\theta = 8 , \rho = 0 , d _ { \mathrm { m a x } } = 1 5 0 $ , target level $\alpha = 0 . 1$ , and time horizon $T = 2 1 5 1$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation. FIFO-OCRC, which fails for $K \leq 2$ (see Fig. 3(a)), is shown only for $K \geq$ 4 in Figs. 3(b) and 3(c).

## B. Tumor Segmentation

In this subsection, we evaluate CA-OCRC on the tumor segmentation task under i.i.d. delays, i.e., $\rho = 0$ . The proposed scheme, referred to as “CA-OCRC (Proposed)”, adopts the proxy delay distribution in Proposition 2. We also consider two alternative proxy delay distributions. The first, referred to as “CA-OCRC (Bernoulli)”, is the distribution in Proposition 1, which does not use knowledge of the delay distribution $G .$ The second, referred to as “CA-OCRC $( F = G ) ^ { \prime }$ is the heuristic choice $F = G$ . We set the maximum delay to $d _ { \operatorname* { m a x } } = 1 5 0$ , the target level to $\alpha = 0 . 1$ , and the time horizon to $T = 2 1 5 1$ rounds. We set $\nu = 4$ and $\theta = 8$ in (39), yielding a mean delay of $\nu \theta = 3 2$ rounds.

1) Impact of the Capacity: Fig. 2 shows the FNR as a function of the time horizon $T$ for capacities $K \in \{ 1 , 2 , 4 \}$ . With unlimited capacity, OCRC approaches the target level quickly by tracking all rounds. In contrast, the FNR of FIFO-OCRC remains close to one for $K = 1$ , and it decreases only slowly for large capacity. In fact, FIFO-OCRC receives the feedback of round t only if $d _ { t } \le K - 1$ , which rarely occurs under the considered delay distribution. All versions of CA-OCRC converge faster as $K$ increases, in line with Corollary 2, with CA-OCRC (Proposed) being the fastest capacity-constrained scheme for all values of $K .$

Fig. 3 shows that, as the capacity K grows, the FNRs of all variants of CA-OCRC approach the target level, while that of FIFO-OCRC only achieves for $K \geq 8$ . The set sizes of CA-OCRC (Proposed) and CA-OCRC (Bernoulli) generally decrease with $K .$ , since more feedback signals become available for calibration, and both schemes keep the tracking load below K. CA-OCRC (Bernoulli), being designed for arbitrary delays, has the lowest tracking load, while CA-OCRC (Proposed) exploits knowledge of G to use it more effectively, yielding smaller masks that match those of OCRC for $K = 3 2 .$ . Note that the small masks of CA-OCRC $( F = G )$ for $K \leq 4$ are not evidence of efficiency, as they come with FNR values above the target level. In fact, stochastic tracking load of CA-OCRC $( F = G )$ in (31) equals $\begin{array} { r } { \ell _ { F , G } = \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } G ( a ) ^ { 2 } \approx 2 3 } \end{array}$ , and exceeds K for $K \leq 1 6$ , violating Assumption 5, and the feedback lost at saturation causes the FNR to move away from the target level.

![](images/3ce50a2d9a6b4afde0135973ec862075ea783d9baf7a0c1b0bab2ded430c4f24.jpg)  
(a)

![](images/c1003b59a5e03b8c04a9f6504f7a1eb4985949b20ffaba801d3e115e669cf336.jpg)  
(b)

![](images/267a85daefea235da978bea60b69eaf1c4f53088aca6aba700deb2febcfb9114.jpg)  
(c)  
Fig. 4. (a) FNR, (b) set size, and (c) tracking load as a function of the shape ν for the tumor segmentation task with $\theta = 8 ,$ $\rho = 0 , d _ { \mathrm { m a x } } = 1 5 0$ , capacity $K = 8 ,$ target level $\alpha = 0 . 1$ , and time horizon $T = 2 1 5 1$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation. FIFO-OCRC, which fails for $\nu \geq 8$ (see Fig. 4(a)), is shown only for $\nu \leq 4$ in Figs. 4(b) and 4(c).

2) Impact of the Delay Distribution: We now fix the capacity to $K = 8 ,$ , and change the shape $\nu$ in (39), with scale $\theta = 8$ . Larger values of the shape parameter $\nu$ yield longer and more concentrated delays, with mean νθ ranging from 8 to 128 rounds. Accordingly, the tracking load of OCRC exceeds the capacity for for all values of ν considered (see Fig. 4(c)). As seen in Fig. 4(a), CA-OCRC (Proposed) and CA-OCRC (Bernoulli) keep the FNR close to the target level for all values of the shape parameter ν, while FIFO-OCRC fails for $\nu \geq 8 .$ . Fig. 4(b) shows that the set sizes of OCRC and CA-OCRC (Proposed) increase with $\nu ,$ since longer delays make the feedback used by the threshold updates more outdated.

The advantage of CA-OCRC (Proposed) over CA-OCRC (Bernoulli) in terms of mask size is largest for short delays, and it shrinks as the shape parameter ν increases (see Fig. 4(b)). In fact, the Bernoulli design in Proposition 1 is tailored to the worst case in which all delays equal $d _ { \operatorname* { m a x } } ,$ a scenario that becomes increasingly representative as the mean delay νθ approaches $d _ { \operatorname* { m a x } } = 1 5 0$ This is in line with Corollary 1 and Corollary 2, whose rates depend on the maximum delay $d _ { \mathrm { m a x } }$ and on the mean delay $\mathbb { E } _ { G } [ d _ { t } ]$ , respectively, so that knowledge of the delay distribution G is most valuable when $\mathbb { E } _ { G } [ d _ { t } ]$ is much smaller than $d _ { \mathrm { m a x } }$

![](images/7b16c9aadf0676a8217a5d17f39154680dcf328bc05371aae5669a45e240e4ed.jpg)  
(a)

![](images/076aa07abb2b0812473f0761ba739455d3f8f18e04ad62506e2744559fb6fdd5.jpg)  
(b)

![](images/cfdade158b532227282e099505c3907c2d71f7f3bc59c551b89156c69a24fe77.jpg)  
(c)  
Fig. 5. F1 risk as a function of the time horizon T for the open-domain question answering task with $\nu = 8 , \theta = 3 . 7 5 , \rho = 0 ,$ $d _ { \operatorname* { m a x } } = 1 0 0$ , and target level $\alpha = 0 . 3$ (dashed line), for capacities (a) $K = 1 ,$ (b) $K = 2 ,$ and (c) $K = 4 .$ The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.

## C. Open-Domain Question Answering

In this subsection, we evaluate CA-OCRC on the open-domain question answering task, again under i.i.d. delays. The benchmark settings are the same as in Sec. VII-B. We set the maximum delay to $d _ { \operatorname* { m a x } } = 1 0 0$ , the target level to $\alpha = 0 . 3 { \mathrm { ~ a s } }$ in [9], and the time horizon to $T = 3 6 1 0$ rounds, with each question appearing once. We set $\nu = 8$ and $\theta = 3 . 7 5$ in (39), yielding a mean delay of $\nu \theta = 3 0$ rounds.

1) Impact of the Capacity: Fig. 5 shows the F1 risk as a function of the time horizon T for capacities $K \in \{ 1 , 2 , 4 \}$ . Since short delays are rare, FIFO-OCRC, which obtains almost no feedback for these capacities, thus, its F1 risk remains close to one. In contrast, all versions of CA-OCRC converge faster as K increases, in line with Corollary 2, with CA-OCRC (Proposed) converging fastest for $K \in \{ 2 , 4 \}$ and performing comparably to CA-OCRC (Bernoulli) for $K = 1$

Fig. 6 shows that, as K grows, the F1 risks of all versions of CA-OCRC approach the target level, while that of FIFO-OCRC does so only for $K \geq 1 6 . { \mathrm { C A - O C R C } }$ (Bernoulli) uses the least amount of capacity, since its design targets the worst case and is thus the most conservative. By exploiting knowledge of the delay distribution G, CA-OCRC (Proposed) yields smaller sets than CA-OCRC (Bernoulli) for $K \geq 4 .$ , and its set size approaches that of OCRC for $K = 3 2$ (see Fig. 6(b)). In contrast, CA-OCRC $( F = G )$ , whose stochastic tracking load $\ell _ { F , G } \approx 2 4$ exceeds K for $K \leq 1 6$ , operates at full capacity in this regime, which results in F1 risks well above the target level for $K \le 2$

![](images/41374a2515cbe89ed5e93434747e1e3c3faefca85142832096b19fca6cc5fb1c.jpg)  
(a)

![](images/9287d984fec211848ec04a990009e6164b4ad6bd51c9628e1f75bb4ae8419b57.jpg)  
(b)

![](images/78a53138c1ac149747cf0a668fbdfa9d94db5b20bd468606f7eccbad6c79f025.jpg)  
(c)

Fig. 6. (a) F1 risk, (b) set size, and (c) tracking load as a function of the capacity K for the open-domain question answering task with $\nu = 8 , \theta = 3 . 7 5 , \rho = 0 , d _ { \operatorname* { m a x } } = 1 0 0 ,$ , target level $\alpha = 0 . 3 ,$ and time horizon $T = 3 6 1 0$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation. FIFO-OCRC, which fails for $K \leq 4$ (see Fig. 6(a)), is shown only for $K \geq 8$ in Figs. 6(b) and 6(c).  
![](images/b0f13185f5dd1216040a64106d72c61af331f43fba6dea178d4905b28a570874.jpg)  
(a)

![](images/649a47e11083059a3f937b5f9ebacdda528230e8094485678bceae015116ac48.jpg)  
(b)

![](images/a19846c9bf62668e798b013bd6ea79537aabad613db7be45cd0aefde068b2360.jpg)  
(c)  
Fig. 7. (a) F1 risk, (b) set size, and (c) tracking load as a function of the scale parameter θ for the open-domain question answering task with $\nu = 8 , \rho = 0 , d _ { \mathrm { m a x } } = 1 0 0 \mathrm { . }$ , capacity $K = 8 ,$ target level $\alpha = 0 . 3$ , and time horizon $T = 3 6 1 0$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation. FIFO-OCRC, which fails for $\theta \geq 5$ (see Fig. 7(a)), is shown only for $\theta \leq 3 . 7 5$ in Figs. 7(b) and 7(c).

2) Impact of the Delay Distribution: Fig. 7 shows the impact of the delay distribution for $K = 8$ . Unlike in Sec. VII-B, we fix the shape to $\nu = 8$ , and change the scale θ in (39), which stretches the delays without changing their relative spread, with mean $\nu \theta$ ranging from 10 to 50 rounds and variance $\nu \theta ^ { 2 }$ ranging from 12.5 to 312.5. CA-OCRC (Proposed) and CA-OCRC (Bernoulli) keep the F1 risk close to the target level for all values of θ, while FIFO-OCRC fails for $\theta \geq 5$ . Due to the concentration of the delays, the tracking load of CA-OCRC (Proposed) decreases sharply from nearly the full capacity to half capacity when θ increases from 1.25 to 2.5, and it then remains nearly constant as θ increases (see Fig. 7(c)).

With a constant tracking load, CA-OCRC (Proposed) receives fewer feedback signals as the delays become longer, and these signals are also more outdated, so that its set size grows with θ.

In contrast, since the probability $p ^ { \star }$ in Proposition 1 does not depend on $\theta ,$ , CA-OCRC (Bernoulli) receives the feedback of about the same number of rounds for all values of $\theta ,$ producing large sets of nearly constant size, while its tracking load grows with the mean delay. Finally, for $\theta \geq 2 . 5$ CA-OCRC $( F = G )$ again produces smaller sets than CA-OCRC (Proposed) at the cost of operating close to saturation, so that its F1 risk moves away from the target level as θ increases. Overall, the distribution-aware design in Proposition 2 uses the capacity more effectively than the Bernoulli design in Proposition 1, yielding faster convergence and smaller sets.

## D. Hierarchical Image Classification

In this subsection, we evaluate CA-OCRC on the hierarchical image classification task under temporally correlated feedback delays, i.e., with $\rho > 0$ in (39), which violate Assumption 1(b), unlike in Sec. VII-B and Sec. VII-C. The proposed scheme, referred to as “CA-OCRC (Proposed)”, thus adopts the proxy delay distribution in Proposition 1, with $\bar { d } _ { \mathrm { m a x } } ~ = ~ d _ { \mathrm { m a x } }$ and $q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) = 1$ in (22), which requires no knowledge of the delay beyond $d _ { \mathrm { m a x } }$ . The oracle benchmark “CA-OCRC (best fixed $p ) ^ { \dag }$ adopts the same proxy delay distribution, but with the value of $p$ that minimizes the risk gap, obtained through a grid search. Since the delays are identically distributed, with the survival function $G$ in (40), but not independent, a natural alternative, referred to as “CA-OCRC (i.i.d. design)”, is to adopt the proxy delay distribution in Proposition 2. However, CA-OCRC (i.i.d. design) is not covered by the guarantee in Theorem 2, which relies on the independence of the delays. We set $d _ { \operatorname* { m a x } } = 1 0 0$ 2 $\alpha = 0 . 0 2$ as in [9], and $T = 2 \times 1 0 ^ { 4 }$ , along with $\nu = 2$ $\theta = 8$ , and $\rho = 0 . 9 9 9$ in (39), yielding a mean delay of 16 rounds and a correlation time $1 / ( 1 - \rho )$ of $1 0 ^ { 3 }$ rounds.

1) Choice of the Admission Probability: Fig. 8 evaluates CA-OCRC with a fixed admission probability $p ,$ i.e., with $F ( a ) = p$ for $a \in \{ 1 , \dots , d _ { \operatorname* { m a x } } \}$ . In line with the bound in Theorem 1, the risk gap first decreases with $p ,$ as more feedback is received, and, for small capacities, it then increases once capacity saturation becomes frequent. The probability $p ^ { \star }$ lies to the left of the minimum, i.e., it is smaller than the value of $p$ minimizing the risk gap. This conservative choice keeps the tracking load well below K (see Fig. 8(c)), so that capacity saturation is rare, at the cost of larger sets (see Fig. 8(b)). Such conservatism is consistent with the worst-case design in Proposition 1.

2) Impact of the Capacity: Fig. 9 reports the hierarchical risk as a function of the time horizon T for capacities $K \in \{ 1 , 2 , 4 \}$ . FIFO-OCRC remains well above the target level for all values of K. CA-OCRC (i.i.d. design), whose design relies on the independence of the delays in Assumption 1(b), also remains above the target level for $K \ \leq \ 2 .$ . In contrast, CA-OCRC (Proposed) approaches the target level for all values of $K .$ , converging faster as K increases, in line with Corollary 1. For $K = 1$ , its convergence is slow, with a large standard deviation, since it admits fewer rounds, each with a large inverse probability weight $1 / p ^ { \star }$ , while, for $K = 4 .$ , it performs comparably to CA-OCRC (best fixed $p )$ and to OCRC.

![](images/e387bb5a07c29bde3bb8c5971679665ddc0683a20080191fb2adeb25a1beeb92.jpg)  
(a)

![](images/23dde876fa9a9fb060a4dddf9237c254f842ccf4cf79c4fb3e6c2b9d2c628da6.jpg)  
(b)

![](images/8b7ae9d9bae00fd3cf16ffc8cffbe03d226ede52ba2001ff44e2be700feabd6c.jpg)  
(c)

Fig. 8. (a) Risk gap, (b) set size, and (c) tracking load of CA-OCRC with a fixed admission probability p as a function of $p$ for the hierarchical image classification task with $\nu = 2 , \theta = 8 , \rho = 0 . 9 9 9 , d _ { \operatorname* { m a x } } = 1 0 0$ , capacities $K \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ , target level $\alpha = 0 . 0 2$ , and time horizon $T = 2 \times 1 0 ^ { 4 }$ . Stars mark the probability $p ^ { \star }$ in Proposition 1, and triangles mark the minimum of the risk gap. The results are averaged over $1 0 ^ { 3 }$ independent trials.  
![](images/bacc8c99ad6b8f7924e4b5920d215e3d9b623a0268802632ded39b4d7379e3e3.jpg)  
(a)

![](images/84f0f38a2e307ff817d9d3988ae6668fc258ff0a516aa5c452d6689522b97ba0.jpg)  
(b)

![](images/7df9a7d31b34468f95e0d301b483002fbc0052d28d724bfabe525046ef3a6228.jpg)  
(c)  
Fig. 9. Hierarchical risk as a function of the time horizon T for the hierarchical image classification task with $\nu = 2 , \theta = 8 .$ $\rho = 0 . 9 9 9 , d _ { \mathrm { m a x } } = 1 0 0$ , and target level $\alpha = 0 . 0 2$ (dashed line), for capacities (a) $K = 1 ,$ , (b) $K = 2 ,$ and (c) $K = 4$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.

Fig. 10 shows that, as K grows, the hierarchical risks of all capacity-constrained schemes decrease towards the target level, with CA-OCRC (Proposed) and CA-OCRC (best fixed $p )$ being the closest to it for all values of K. Note that the small sets of FIFO-OCRC are not evidence of efficiency, as they come with hierarchical risks above the target level. The set size of CA-OCRC (Proposed) decreases as K grows, but it remains larger than that of OCRC for all considered capacities, since it keeps the lower tracking load. In contrast, CA-OCRC (best fixed $p )$ uses most of the capacity for $K \geq 4$ , matching the set size of OCRC.

![](images/ea40420843cf7b0d2c81f186a750348a91e28f62b7022e3a7e3704466c7f0ee5.jpg)  
(a)

![](images/335e97b6ca7839212ec93b7b9b79433b74b4820cb108969c10d9558c9fdc8c1e.jpg)  
(b)

![](images/410638674fc0fb248c711cf9bdb52a7cffe846dc3ccb4f971ed2911ce6e3d21b.jpg)  
(c)

Fig. 10. (a) Hierarchical risk, (b) set size, and (c) tracking load as a function of the capacity K for the hierarchical image classification task with $\nu = 2 ,$ θ = 8, $\rho = 0 . 9 9 9$ $d _ { \operatorname* { m a x } } = 1 0 0 .$ , target level $\alpha = 0 . 0 2 \ :$ , and time horizon $T = 2 \times 1 0 ^ { 4 }$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.  
![](images/022ebd39ba2a47be36549682359205e561f50d38d43b55dc03ace4110748ccfa.jpg)  
(a)

![](images/84bd51a2ae81681d15aa9d0382fd304537a241ba02ed3996c34af8bbd4146264.jpg)  
(b)

![](images/049540464479d6984166a10d605bc9671bbf79529eed5167b82eb9903a2bfaae.jpg)  
(c)  
Fig. 11. (a) Hierarchical risk, (b) set size, and (c) tracking load as a function of the correlation time $1 / ( 1 - \rho )$ for the hierarchical image classification task with $\nu = 2 , \theta = 8 , d _ { \mathrm { m a x } } = 1 0 0 ^ { \circ }$ , capacity $K = 4 ,$ target level $\alpha = 0 . 0 2$ , and time horizon $T = 2 \times 1 0 ^ { 4 }$ The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.

3) Impact ofthe Delay Correlation: We now fix the capacity to $K = 4 ,$ and vary the correlation time $1 / ( 1 - \rho )$ in (39) from 1, i.e., i.i.d. delays, to $1 0 ^ { 4 }$ rounds. CA-OCRC (Proposed) keeps the hierarchical risk close to the target level for all correlation times, in line with Theorem 1, which holds for any sequence of delays. In fact, since CA-OCRC (Proposed) admits each round with probability $p ^ { \star }$ , irrespective of the delays, and tracks it until its feedback is received, its set size and tracking load are also unaffected by the temporal correlation of the delays. In contrast, the hierarchical risk of FIFO-OCRC exceeds the target level for all correlation times, and it increases with the correlation time, as the feedback signals with delays $d _ { t } \le K - 1$ arrive in bursts during periods of short delays. CA-OCRC (i.i.d. design) keeps the hierarchical risk close to the target level for correlation times up to about 10 rounds. For longer correlation times, however, the delays depart further from the i.i.d. model underlying its design, and its risk approaches twice the target level.

![](images/b274c157c570df9f0fc3afd89a057be81d3771e3c9d35b9003fad17f17bcac72.jpg)  
(a)

![](images/d6555adb4c47d32e76fea7966e4de4013858aae61136780a08cca9b0d0f36daf.jpg)  
(b)

![](images/43c76724e11712fc6a24c4e159b46c30d7d72595bc2d97a022ecfc8053b0d0a4.jpg)  
(c)  
Fig. 12. SNR loss as a function of the time horizon T for the beam selection task with $\nu = 2 , \theta = 1 0 , \rho = 0 . 9 9 , d _ { \operatorname* { m a x } } = 1 0 0 ,$ and target level $\alpha = 0 . 2$ (dashed line), for capacities (a) $K = 1 ,$ , (b) $K = 2 ,$ , and (c) $K = 4$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.

## E. Beam Selection

In this subsection, we evaluate CA-OCRC on the beam selection task, also under temporally correlated feedback delays. The benchmark settings are the same as in Sec. VII-D. We set the maximum delay to $d _ { \operatorname* { m a x } } = 1 0 0$ , the target level to $\alpha = 0 . 2$ , and the time horizon to $T = 2 \times 1 0 ^ { 4 }$ rounds, with each user appears once. We set $\nu = 2$ $\theta = 1 0$ , and $\rho = 0 . 9 9$ in (39), yielding a mean delay of 20 rounds.

Fig. 12 shows the SNR loss as a function of the time horizon T for capacities $K \in \{ 1 , 2 , 4 \}$ FIFO-OCRC and CA-OCRC (i.i.d. design) converge slowly, remaining above the target level for all values of K. In contrast, CA-OCRC (Proposed) converges faster as K increases, in line with Corollary 1, approaching the target level for $K \geq 2 .$ . For $K = 1$ , CA-OCRC (best fixed p) converges faster than CA-OCRC (Proposed), while both schemes perform comparably to OCRC for $K = 4$

Fig. 13 shows that FIFO-OCRC exceeds the target level for all considered capacities, while CA-OCRC (i.i.d. design) does so for $K \leq 4$ . In contrast, CA-OCRC (Proposed) approaches the target level as K grows. The set size of CA-OCRC (Proposed) decreases from about 7 beams for $K = 1$ to approximately that of OCRC for $K \geq 8$ (see Fig. 13(b)). Finally, Fig. 13(c) shows that OCRC violates the capacity constraint for all considered values of K, while CA-OCRC (Proposed) keeps the tracking load well below K. Overall, the worst-case design in Proposition 1 enables CA-OCRC to approach the target level irrespective of the correlation time, unlike the distribution-aware design in Proposition 2, however, at the cost of larger sets than the oracle choice of $p .$

![](images/72c9dcaed0bf374e3ac0dc97eae9a92a958d433598184ce834a1ea53f7ce8a4e.jpg)  
(a)

![](images/af5472048d9bcaab061fd6bf46f4b67704852e16db1cd5bf0ce9f84ae6b795e2.jpg)  
(b)

![](images/37c4f697a169e2e94f372d92e466eb308344ea4d273c14a709b75759df3e5c40.jpg)  
(c)  
Fig. 13. (a) SNR loss, (b) set size, and (c) tracking load as a function of the capacity K for the beam selection task with ν = 2, θ = 10, $\rho = 0 . 9 9 ;$ $d _ { \operatorname* { m a x } } = 1 0 0 .$ , target level $\alpha = 0 . 2 ,$ and time horizon $T = 2 \times 1 0 ^ { 4 }$ . The results are averaged over $1 0 ^ { 3 }$ independent trials, with the error bars denoting ±1 standard deviation.

## VIII. CONCLUSION

We have studied OCRC under capacity-constrained delayed feedback, in which at most K rounds with pending feedback can be tracked at any time. For this setting, we have proposed capacity-aware OCRC (CA-OCRC), which tracks each round, if capacity permits, until its feedback is received or a random proxy delay expires, and compensates for the lost feedback via a two-level form of inverse probability weighting. We have analyzed the risk boundsof CA-OCRC under arbitrary and stochastic delays, and we have optimized the proxy delay distribution based on these bounds. With the optimized designs, the risk gap decays as $\mathcal { O } ( ( d _ { \operatorname* { m a x } } / T ) ^ { K / ( K + 1 ) } )$ under arbitrary delays, so that a capacity logarithmic in $T / d _ { \mathrm { m a x } }$ suffices to match the convergence rate of OCRC with unlimited capacity. Under stochastic delays, the maximum delay $d _ { \mathrm { m a x } }$ in this rate is replaced by the mean delay $\mathbb { E } [ d _ { t } ]$ , by exploiting knowledge of the delay distribution.

Experiments on tumor segmentation, open-domain question answering, hierarchical image classification, and beam selection have verified the proposed methods and the theoretical findings. In all tasks, CA-OCRC approached the target risk level while keeping the tracking load below the capacity. Under stochastic delays, the distribution-aware design used the capacity more effectively than the worst-case design, yielding faster convergence and smaller prediction sets.

Future work includes the extension of CA-OCRC to general bounded losses that are nonincreasing [9], the design of adaptive tracking policies that require no prior knowledge of the delay distribution, as well as the development of parameter-free capacity-aware algorithms [30].

[1] W. Wisniewski, D. Lindsay, and S. Lindsay, “Application of conformal prediction interval estimations to market makers net positions,” in Proc. Symp. Conformal Probabilistic Prediction Appl., 2020, pp. 285–301.

[2] J. Lekeufack, A. N. Angelopoulos et al., “Conformal decision theory: Safe autonomous decisions from imperfect predictions,” in Proc. IEEE Int. Conf. Robot. Automat., 2024, pp. 11 668–11 675.

[3] O. Simeone, S. Park, and M. Zecchin, “Conformal calibration: Ensuring the reliability of black-box AI in wireless systems,” IEEE Commun. Mag., vol. 64, no. 8, pp. 179–187, 2026.

[4] Y.-J. Park, C. Sobolewski, and N. Azizan, “Uncertainty quantification in detection transformers: Object-level calibration and image-level reliability,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 11, pp. 13 775–13 788, 2026.

[5] K. Wang, F. Cuzzolin, K. Shariatmadar, D. Moens, and H. Hallez, “A review of uncertainty representation and quantification in neural networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 3, pp. 2476–2495, 2026.

[6] O. Simeone and Y. Romano, “Uncertainty quantification and data efficiency in AI: An information-theoretic perspective,” IEEE J. Sel. Areas Inf. Theory, vol. 7, pp. 290–301, 2026.

[7] A. N. Angelopoulos, R. F. Barber, and S. Bates, “Theoretical foundations of conformal prediction,” arXiv preprint arXiv:2411.11824, 2024.

[8] S. Park, K. M. Cohen, and O. Simeone, “Few-shot calibration of set predictors via meta-learned cross-validation-based conformal prediction,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 1, pp. 280–291, 2024.

[9] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, “Conformal risk control,” in Proc. Int. Conf. Learn. Represent., 2024, pp. 55 198–55 218.

[10] B.-S. Einbinder, L. Ringel, and Y. Romano, “Semi-supervised risk control via prediction-powered inference,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 12, pp. 11 080–11 090, 2025.

[11] V. Blot, A. N. Angelopoulos, M. Jordan, and N. J.-B. Brunel, “Automatically adaptive conformal risk control,” in Proc. Int. Conf. Artif. Intell. Statist., 2025.

[12] I. Gibbs and E. Candes, “Adaptive conformal inference under distribution shift,” in Proc. Adv. Neural Inf. Process. Syst., vol. 34, 2021, pp. 1660–1672.

[13] A. N. Angelopoulos, R. Barber, and S. Bates, “Online conformal prediction with decaying step sizes,” in Proc. Int. Conf. Mach. Learn., 2024, pp. 1616–1630.

[14] C. Xu and Y. Xie, “Conformal prediction for time series,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 10, pp. 11 575–11 587, 2023.

[15] S. Feldman, L. Ringel, S. Bates, and Y. Romano, “Achieving risk control in online learning settings,” Trans. Mach. Learn. Res., 2023.

[16] M. Zecchin and O. Simeone, “Localized adaptive risk control,” in Proc. Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 8165–8192.

[17] M. Zhu, M. Zecchin, S. Park, C. Guo, C. Feng, P. Popovski, and O. Simeone, “Conformal distributed remote inference in sensor networks under reliability and communication constraints,” IEEE Trans. Signal Process., vol. 73, pp. 1485–1500, 2025.

[18] F. Binucci, O. Simeone, and P. Banelli, “Conformal Lyapunov optimization: Optimal resource allocation under deterministic reliability constraints,” IEEE Trans. Signal Process., vol. 73, pp. 4521–4538, 2025.

[19] C. L. Cifra, D. F. Sittig, and H. Singh, “Bridging the feedback gap: a sociotechnical approach to informing clinicians of patients’ subsequent clinical course and outcomes,” BMJ Qual. Saf., vol. 30, no. 7, pp. 591–597, 2021.

[20] T. Kwiatkowski et al., “Natural Questions: A benchmark for question answering research,” Trans. Assoc. Comput. Linguist., vol. 7, pp. 452–466, 2019.

[21] B. Csaba, W. Zhang, M. Muller, S.-N. Lim, M. Elhoseiny, P. H. Torr, and A. Bibi, “Label delay in online continual¨ learning,” in Proc. Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 119 976–120 012.

[22] Q. Pham, C. Liu, and S. C. Hoi, “Continual learning, fast and slow,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 1, pp. 134–149, 2023.

[23] R. Wiesmayr, L. Maggi, S. Cammerer, J. Hoydis, F. A. Aoudia, and A. Keller, “SALAD: Self-adaptive link adaptation,” arXiv preprint arXiv:2510.05784, 2025.

[24] O. Chapelle, “Modeling delayed feedback in display advertising,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2014, pp. 1097–1105.

[25] J. Hallberg Szabadvary, “Adaptive conformal inference for multi-step ahead time-series forecasting online,” in ´ Proc. Symp. Conformal Probabilistic Prediction Appl., 2024, pp. 250–263.

[26] T. Ding, I. Gibbs, and R. J. Tibshirani, “Calibrated multi-level quantile forecasting,” arXiv preprint arXiv:2512.23671, 2025.

[27] L. E. Halabi and A. Brandt, “Adaptive conformal inference under delayed feedback: Coverage guarantees and a delay-tomemory diagnostic,” arXiv preprint arXiv:2609.07251, 2026.

[28] M. Zhao, H. Admoni, R. Simmons, A. Ramdas, and A. Bajcsy, “Conformalized interactive imitation learning: Handling expert shift and intermittent feedback,” in Proc. Int. Conf. Learn. Represent., 2025, pp. 9238–9251.

[29] B. Wang, M. Zecchin, and O. Simeone, “Mirror online conformal prediction with intermittent feedback,” IEEE Signal Process. Lett., vol. 32, pp. 2888–2892, 2025.

[30] F. Orabona, “A modern introduction to online learning,” arXiv preprint arXiv:1912.13213, 2019.

[31] S. Shalev-Shwartz, “Online learning and online convex optimization,” Found. Trends Mach. Learn., vol. 4, no. 2, pp. 107–194, 2025.

[32] M. Zinkevich, “Online convex programming and generalized infinitesimal gradient ascent,” in Proc. Int. Conf. Mach. Learn., 2003, pp. 928–936.

[33] J. Langford, A. Smola, and M. Zinkevich, “Slow learners are fast,” arXiv preprint arXiv:0911.0491, 2009.

[34] K. Quanrud and D. Khashabi, “Online learning with adversarial delays,” in Proc. Adv. Neural Inf. Process. Syst., vol. 28, 2015.

[35] Y. Wan, C. Yao, M. Song, and L. Zhang, “Non-stationary online convex optimization with arbitrary delays,” in Proc. Int. Conf. Mach. Learn., 2024, pp. 49 991–50 011.

[36] P. Wu, H. Huang, and Z. Liu, “Online sequential decision-making with unknown delays,” in Proc. ACM Web Conf., 2024, pp. 4028–4036.

[37] H. Qiu, E. Esposito, and M. Zhang, “Exploiting curvature in online convex optimization with delayed feedback,” in Proc. Int. Conf. Mach. Learn., 2025, pp. 50 448–50 479.

[38] A. Ryabchenko, I. Attias, and D. M. Roy, “Capacity-constrained online learning with delays: Scheduling frameworks and regret trade-offs,” in Proc. Conf. Learn. Theory, 2025, pp. 4973–5014.

[39] ——, “Capacity-constrained online convex optimization with delayed feedback,” arXiv preprint arXiv:2606.11711, 2026.

[40] A. N. Angelopoulos, M. I. Jordan, and R. J. Tibshirani, “Gradient equilibrium in online learning: Theory and applications,” J. Mach. Learn. Res., vol. 26, no. 305, pp. 1–68, 2025.

[41] M. Zecchin, U. K. Ganesan, G. Durisi, P. Popovski, and O. Simeone, “Prediction-powered communication with distortion guarantees,” IEEE J. Sel. Areas Inf. Theory, vol. 7, pp. 33–45, 2026.

[42] V. Elvira and L. Martino, Advances in Importance Sampling. John Wiley & Sons, Ltd, 2021, pp. 1–14.

[43] A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle introduction,” Found. Trends Mach. Learn., vol. 16, no. 4, pp. 494–591, 2023.

[44] J. Deng, J. Krause, A. C. Berg, and L. Fei-Fei, “Hedging your bets: Optimizing accuracy-specificity trade-offs in large scale visual recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2012, pp. 3450–3457.

[45] D.-P. Fan, G.-P. Ji, T. Zhou, G. Chen, H. Fu, J. Shen, and L. Shao, “PraNet: Parallel reverse attention network for polyp segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent., 2020, pp. 263–273.

[46] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, “Dense passage retrieval for open-domain question answering,” in Proc. Conf. Empir. Methods Nat. Lang. Process., 2020, pp. 6769–6781.

[47] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ questions for machine comprehension of text,” in Proc. Conf. Empir. Methods Nat. Lang. Process., 2016, pp. 2383–2392.

[48] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2009, pp. 248–255.

[49] C. Fellbaum, WordNet: An electronic lexical database. MIT press, 1998.

[50] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

[51] F. Yao, J. Wu, G. Venkataramani, and S. Subramaniam, “TS-BatPro: Improving energy efficiency in data centers by leveraging temporal–spatial batching,” IEEE Trans. Cogn. Commun. Netw., vol. 3, no. 1, pp. 236–249, 2019.

[52] W. Chen, R. F. da Silva, E. Deelman, and T. Fahringer, “Dynamic and fault-tolerant clustering for scientific workflows,” IEEE Trans. Cloud Comput., vol. 4, no. 1, pp. 49–62, 2016.

[53] L. Kleinrock, Queueing Systems, Volume I: Theory. New York: Wiley, 1975.

[54] M. Kalman and B. Girod, “Modeling the delays of successively-transmitted Internet packets,” in Proc. IEEE Int. Conf. Multimedia and Expo (ICME), 2004, pp. 2015–2018.

[55] A. Severinson, E. Rosnes, S. El Rouayheb, and A. G. i Amat, “DSAG: A mixed synchronous-asynchronous iterative method for straggler-resilient learning,” IEEE Trans. Commun., vol. 71, no. 2, pp. 808–822, 2022.

[56] Y. Liang and B. Girod, “Network-adaptive low-latency video communication over best-effort networks,” IEEE Trans. Circuits Syst. Video Technol., vol. 16, no. 1, pp. 72–81, 2006.

[57] S. I. Ktena, A. Tejani, L. Theis, P. K. Myana, D. Dilipkumar, F. Huszar, S. Yoo, and W. Shi, “Addressing delayed feedback´ for continuous training with neural networks in CTR prediction,” in Proc. ACM Conf. Recomm. Syst., 2019, pp. 187–195.

[58] P. A. W. Lewis, E. McKenzie, and D. K. Hugus, “Gamma processes,” Communications in Statistics. Stochastic Models, vol. 5, no. 1, pp. 1–30, 1989.

# Capacity-Constrained Delayed Feedback for Online Conformal Risk Control

Bowen Wang, Student Member, IEEE and Osvaldo Simeone, Fellow, IEEE

(Supplementary Material)

APPENDIX A

PROOFS OF OCRC, I-OCRC, AND CA-OCRC

## A. Proof of Lemma 2

Following [S1]–[S3], we continue OCRC over $d _ { \mathrm { m a x } }$ virtual rounds $t = T + 1 , \dots , T + d _ { \operatorname* { m a x } }$ where no new data point is observed, while the feedback of the rounds $t \leq T$ with $t + d _ { t } > T$ is received and the threshold is updated via (13). Accordingly, $r _ { T + 1 } , \cdot \cdot \cdot \ , r _ { T + d _ { \mathrm { m a x } } + 1 }$ are referred to as the virtual thresholds. Summing (13) over rounds $t = 1 , \ldots , T + d _ { \mathrm { m a x } }$ yields

$$
\sum _ { t = 1 } ^ { T } ( L _ { t } - \alpha ) = \frac { r _ { T + d _ { \operatorname* { m a x } } + 1 } - r _ { 1 } } { \eta } .\tag{A-1}
$$

Then, we turn to bounding $\{ r _ { t } \} _ { t = 1 } ^ { T + d _ { \operatorname* { m a x } } + 1 }$ . Specifically, suppose that $r _ { t + 1 } > B$ for some $t \le$ $T + d _ { \operatorname* { m a x } } ,$ , and let $\tau = \operatorname* { m a x } \{ j \leq t : r _ { j } \leq B \}$ . For every round $j$ with $\tau < j \leq \operatorname* { m i n } \{ t , T \}$ , we have $r _ { j } > B$ , and hence $\mathcal { C } _ { j } = \mathcal { y }$ and $L _ { j } = \mathcal { L } ( \mathcal { V } , Y _ { j } ) = 0$ by Assumption 2 and Assumption $^ { 3 , }$ so that the feedback of round j decreases the threshold, while the virtual rounds $j > T$ produce no feedback of their own. Therefore, every upward term $\eta ( L _ { k } - \alpha )$ of the updates (13) applied in rounds $\tau , \ldots , t$ is due to a round $k \leq \tau$ with $k + d _ { k } \ge \tau .$ , i.e., $k \in \{ \tau - d _ { \operatorname* { m a x } } , \dots , \tau \}$ , and it is at most $\eta ( 1 - \alpha )$ , which yields

$$
\begin{array} { r } { r _ { t + 1 } \leq r _ { \tau } + \eta ( 1 - \alpha ) ( d _ { \operatorname* { m a x } } + 1 ) } \\ { \leq B + \eta ( 1 - \alpha ) ( d _ { \operatorname* { m a x } } + 1 ) . } \end{array}\tag{A-2}
$$

Symmetrically, we have the lower bound

$$
r _ { t + 1 } \geq - \eta \alpha ( d _ { \operatorname* { m a x } } + 1 ) .\tag{A-3}
$$

Since $r _ { 1 } \in [ 0 , B ]$ , combining the two bounds gives

$$
| r _ { T + d _ { \operatorname* { m a x } } + 1 } - r _ { 1 } | \leq B + \eta ( d _ { \operatorname* { m a x } } + 1 ) .\tag{A-4}
$$

Finally, dividing (A-1) by T and using (A-4) yields

$$
\left| \frac { 1 } { T } \sum _ { t = 1 } ^ { T } L _ { t } - \alpha \right| \leq \frac { B } { \eta T } + \frac { d _ { \operatorname* { m a x } } + 1 } { T } ,\tag{A-5}
$$

which completes the proof of Lemma 2.

## B. Proof of Lemma 3

Summing the update (16) over rounds $t = 1 , \dots , T$ yields

$$
\sum _ { t = 1 } ^ { T } ( L _ { t } - \alpha ) = \frac { r _ { T + 1 } - r _ { 1 } } { \eta } + \sum _ { t = 1 } ^ { T } \left( 1 - \frac { \mathbb { 1 } \left\{ d _ { t } = 0 \right\} } { \operatorname* { P r } _ { G } \left[ d _ { t } = 0 \right] } \right) ( L _ { t } - \alpha ) .\tag{A-6}
$$

Under Assumption 1(b), each summand of the last sum in (A-6) has zero mean given $d _ { 1 } , \dotsc , d _ { t - 1 }$ and the last sum has zero mean [S4], [S5].

Similarly to the proof of Lemma 2, the upward and downward terms of I-OCRC are $\eta ( 1 -$ $\alpha ) / \mathrm { P r } _ { G } [ d _ { t } = 0 ]$ and $\eta \alpha / \mathrm { P r } _ { G } [ d _ { t } = 0 ]$ , respectively, which yields the bound

$$
\lvert r _ { T + 1 } - r _ { 1 } \rvert \leq B + \frac { \eta } { \mathrm { P r } _ { G } [ d _ { t } = 0 ] } .\tag{A-7}
$$

Taking the expectation of (A-6), dividing by T, and using this bound yields (17) completes the proof of Lemma 3.

## C. Proof of Theorem A-1

We first establish a general risk bound for CA-OCRC, which will be used in Appendix B and Appendix D to prove Theorem 1 and Theorem 2, respectively.

Theorem A-1. Under Assumption 1, for any proxy delay distribution F, the risk gap (10) of CA-OCRC satisfies the upper bound

$$
\mathrm { R i s k G a p } ( T ) \le \frac { \big | r _ { 1 } - \mathbb { E } \left[ r _ { T + d _ { \operatorname* { m a x } } + 1 } \right] \big | } { \eta T } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { P r } \{ | K _ { t } ^ { - } | = K \}\tag{A-8}
$$

for any sequence of data points. The expectation and the probability are taken with respect to the proxy delays under Assumption 1(a), and also with respect to the feedback delays under Assumption 1(b).

$r _ { T + d _ { \operatorname* { m a x } } + 1 }$ is the virtual threshold, as for OCRC in the proof of Lemma 2 in Appendix A. In virtual rounds, CA-OCRC only evicts the tracked rounds, receives the feedback of the rounds in the set $\mathcal { O } _ { t } ,$ , and updates the threshold as

$$
r _ { t + 1 } = r _ { t } + \eta \sum _ { k \in \mathcal { O } _ { t } } \frac { L _ { k } - \alpha } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { k } ) } , f o r t = T + 1 , \dots , T + d _ { \operatorname* { m a x } } .\tag{A-9}
$$

Proof. Summing (22) over $t = 1 , \dots , T$ and (A-9) over the virtual rounds telescopes the threshold to

$$
r _ { T + d _ { \mathrm { m a x } } + 1 } = r _ { 1 } - \eta \sum _ { t = 1 } ^ { T } \varphi _ { t } ( \alpha - L _ { t } ) - \eta \sum _ { t = 1 } ^ { T + d _ { \mathrm { m a x } } } \sum _ { k \in \mathcal { O } _ { t } } \frac { \alpha - L _ { k } } { q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) F ( d _ { k } ) }\tag{A-10a}
$$

$$
= r _ { 1 } - \eta \sum _ { t = 1 } ^ { T } w _ { t } ( \alpha - L _ { t } ) ,\tag{A-10b}
$$

where $w _ { t }$ is the weight eventually assigned to the update $\alpha - L _ { t }$ of round t, given by

$$
w _ { t } = \varphi _ { t } + ( 1 - \varphi _ { t } ) \frac { \mathbb { 1 } \{ | K _ { t } ^ { - } | < K \} \mathbb { 1 } \{ \bar { d } _ { t } \geq d _ { t } \} } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { t } ) } .\tag{A-11}
$$

Rearranging (A-10) gives

$$
\sum _ { t = 1 } ^ { T } ( \alpha - L _ { t } ) = \frac { r _ { 1 } - r _ { T + d _ { \operatorname* { m a x } } + 1 } } { \eta } + \Delta _ { T } ,\tag{A-12}
$$

where

$$
\Delta _ { T } = \sum _ { t = 1 } ^ { T } ( 1 - w _ { t } ) ( \alpha - L _ { t } ) = \sum _ { i \in \cal { I } } \left( 1 - \frac { 1 \{ | K _ { i } ^ { - } | < K \} \mathbb { 1 } \{ \bar { d } _ { i } \geq d _ { i } \} } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { i } ) } \right) ( \alpha - L _ { i } ) ,\tag{A-13}
$$

with $\mathcal { T } = \{ t \in \{ 1 , . . . , T \} : \varphi _ { t } = 0 \}$ being the set of rounds with nontrivial prediction sets.

Noting that $\mathbb { 1 } \{ | K _ { i } ^ { - } | < K \} = 1 - \mathbb { 1 } \{ | K _ { i } ^ { - } | = K \}$ , we split $\Delta _ { T } = \Delta _ { T } ^ { 1 } + \Delta _ { T } ^ { 2 }$ with

$$
\Delta _ { T } ^ { 1 } = \sum _ { i \in \mathbb { Z } } \left( 1 - \frac { \mathbb { 1 } \left\{ \bar { d } _ { i } \geq d _ { i } \right\} } { q _ { G } \left( \bar { d } _ { \operatorname* { m a x } } \right) F ( d _ { i } ) } \right) ( \alpha - L _ { i } ) ,\tag{A-14}
$$

$$
\Delta _ { T } ^ { 2 } = \sum _ { i \in \mathcal { I } } \frac { \mathbb { 1 } \{ \bar { d } _ { i } \geq d _ { i } \} \mathbb { 1 } \{ | K _ { i } ^ { - } | = K \} } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { i } ) } ( \alpha - L _ { i } ) .\tag{A-15}
$$

Let $\mathcal { H } _ { i }$ denote the σ-field generated by the feedback delays $d _ { 1 } , \ldots , d _ { i - 1 }$ and by the proxy delays $\bar { d } _ { 1 } , \ldots , \bar { d } _ { i - 1 }$ of the previous rounds. Then, the quantities $r _ { i } , L _ { i } , | \mathcal { K } _ { i } ^ { - } |$ , and $\mathbb { 1 } \{ i \in \mathbb { Z } \}$ are determined by $\mathcal { H } _ { i }$ , while the proxy delay ${ \bar { d } } _ { i }$ is drawn from F independently of $\mathcal { H } _ { i }$ and of the feedback delay $d _ { i }$ , so that $\mathbb { E } [ \mathbb { 1 } \{ \bar { d } _ { i } \geq d _ { i } \} ~ | ~ \mathcal { H } _ { i } , d _ { i } ] = F ( d _ { i } )$ . Since $F ( d _ { i } ) > 0$ if and only if $d _ { i } \leq \bar { d } _ { \operatorname* { m a x } }$ , and since $d _ { i }$ is independent of $\mathcal { H } _ { i }$ , with $\mathrm { P r } _ { G } [ d _ { i } \leq \bar { d } _ { \operatorname* { m a x } } ] = q _ { G } ( \bar { d } _ { \operatorname* { m a x } } )$ , the two-level inverse probability weighting in (22) satisfies

$$
\mathbb { E } \left[ \frac { \mathbb { 1 } \left\{ \bar { d } _ { i } \geq d _ { i } \right\} } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { i } ) } \Bigg | \mathcal { H } _ { i } \right] = \frac { \operatorname* { P r } _ { G } [ d _ { i } \leq \bar { d } _ { \operatorname* { m a x } } ] } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) } = 1 ,\tag{A-16}
$$

where the ratio inside the expectation is set to zero for $d _ { i } > \bar { d } _ { \operatorname* { m a x } }$ . Under Assumption 1(a), the feedback delay $d _ { i }$ is deterministic, and (A-16) holds since $\bar { d } _ { \mathrm { m a x } } = d _ { \mathrm { m a x } }$ and $q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) = 1$

Accordingly, each summand of (A-14) has zero mean given $\mathcal { H } _ { i }$ , and $\Delta _ { T } ^ { 1 }$ is a sum of martingale differences with respect to $\{ \mathcal { H } _ { i } \}$ . Thus, we have $\mathbb { E } [ \Delta _ { T } ^ { 1 } ] = 0$

Similarly, by (A-16), given $\mathcal { H } _ { i }$ , each summand of (A-15) has mean $\mathbb { 1 } \{ | K _ { i } ^ { - } | = K \} ( \alpha - L _ { i } )$ whose absolute value is at most $\mathbb { 1 } \{ | \mathcal { K } _ { i } ^ { - } | = K \}$ , since $| \alpha - L _ { i } | \leq 1$ for $L _ { i } \in [ 0 , 1 ]$ . Therefore, we have

$$
| \mathbb { E } [ \Delta _ { T } ^ { 2 } ] | \leq \sum _ { t = 1 } ^ { T } \operatorname* { P r } \{ t \in \mathbb { Z } , | { \cal K } _ { t } ^ { - } | = K \} \leq \sum _ { t = 1 } ^ { T } \operatorname* { P r } \{ | { \cal K } _ { t } ^ { - } | = K \} .\tag{A-17}
$$

Finally, taking the expectation of (A-12), dividing by $T ,$ and using $\mathbb { E } [ \Delta _ { T } ^ { 1 } ] = 0$ and (A-17) completes the proof. □

## APPENDIX B

## PROOF OF THEOREM 1

We prove the bound in Theorem 1 by bounding the range of the threshold iterates and the saturation term in the following two subsections, respectively. Throughout this appendix, we have $\bar { d } _ { \mathrm { m a x } } = d _ { \mathrm { m a x } }$ and $q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) = 1$ , as discussed in Sec. IV-B.

## A. Bounding the Range of Threshold Iterates

Lemma B-1. Under Assumption 1(a), Assumption 2, and Assumption 3 with $r _ { 1 } \in [ 0 , B ]$ , the thresholds of CA-OCRC satisfy the upper bound

$$
| r _ { t + 1 } - r _ { 1 } | \leq B + \eta \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) }\tag{B-1}
$$

for all $t \in \{ 1 , \dots , T + d _ { \operatorname* { m a x } } \}$ and for any sequence of data points, feedback delays, and proxy delays.

Proof. Each round $t \leq T$ applies the update $- \eta ( \alpha - L _ { t } )$ if $\mathcal { C } _ { t } \in \{ \emptyset , \mathcal { V } \}$ , and the weighted update $- \eta ( \alpha - L _ { k } ) / F ( d _ { k } )$ for each tracked round k with $k + d _ { k } = t .$ , while the virtual rounds apply only the latter updates. Thus, each upward push is at most $\eta ( 1 - \alpha ) / F ( d _ { \mathrm { m a x } } )$ , since $L \leq 1$ and $F ( d _ { k } ) \geq F ( d _ { \operatorname* { m a x } } )$ . Similarly to the proof of Lemma 2, at most K + 1 upward pushes can occur in any sequence of rounds, which gives

$$
\begin{array} { r } { r _ { t + 1 } \leq r _ { \tau } + \eta ( 1 - \alpha ) \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) } } \\ { \leq B + \eta ( 1 - \alpha ) \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) } . } \end{array}\tag{B-2}
$$

Symmetrically, since ${ \mathcal { C } } _ { j } = \emptyset$ and $L _ { j } = \mathcal { L } ( \emptyset , Y _ { j } ) = 1$ whenever $r _ { j } < 0$ , we have the lower bound

$$
r _ { t + 1 } \geq - \eta \alpha \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) } .\tag{B-3}
$$

Since $r _ { 1 } \in [ 0 , B ]$ , combining the upper and lower bounds yields (B-1).

## B. Bounding the Saturation Term

Lemma B-2. Under Assumption $\boldsymbol { l } ( \boldsymbol { a } )$ and Assumption 4, the average probability of capacity saturation of CA-OCRC satisfies the upper bound

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { P r } _ { F } \{ | \mathcal { K } _ { t } ^ { - } | = K \} \le e ^ { - K h ( \ell _ { F } / K ) }\tag{B-4}
$$

for any sequence of data points and feedback delays, where $h ( \cdot )$ is defined in Theorem 1, and $\begin{array} { r } { \ell _ { F } = \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } F ( a ) = \mathbb { E } _ { F } \left[ ( \bar { d } ) _ { + } \right] } \end{array}$ is the tracking load.

Proof. For a given age a, round $t - a$ belongs to the set ${ \ K } _ { t } ^ { - }$ only if it is admitted to tracking, i.e., $| { \cal K } _ { t - a } ^ { - } | < K$ , its feedback is still pending, i.e., $d _ { t - a } \geq a ,$ , and its proxy delay has not expired, i.e., $\bar { d } _ { t - a } \geq a .$ . Accordingly, we have

$$
\begin{array} { r l r } {  { | \mathcal { K } _ { t } ^ { - } | \le \sum _ { a = 1 } ^ { t - 1 } \mathbb { 1 } \{ | \mathcal { K } _ { t - a } ^ { - } | < K \} \mathbb { 1 } \{ d _ { t - a } \ge a \} \mathbb { 1 } \{ \bar { d } _ { t - a } \ge a \} } } \\ & { } & { \le \sum _ { a = 1 } ^ { t - 1 } \mathbb { 1 } \{ d _ { t - a } \ge a \} \mathbb { 1 } \{ \bar { d } _ { t - a } \ge a \} . } \end{array}\tag{B-5a}
$$

(B-5b)

Let $\bar { d } _ { t - a } ^ { \prime } \sim F$ denote the proxy delay that CA-OCRC draws at round $t - a$ if $\mathcal { C } _ { t - a }$ were nontrivial, so that the variables $\bar { d } _ { 1 } ^ { \prime } , \dots , \bar { d } _ { t - 1 } ^ { \prime }$ are independent, $\bar { d } _ { t - a } = \bar { d } _ { t - a } ^ { \prime }$ for every round $t - a$ with a nontrivial prediction set, and $\mathbb { 1 } \{ \bar { d } _ { t - a } \geq a \} = 0 \leq \mathbb { 1 } \{ \bar { d } _ { t - a } ^ { \prime } \geq a \}$ otherwise. Hence, the bound (B-5b) also holds with $\bar { d } _ { t - a }$ replaced by $\bar { d } _ { t - a } ^ { \prime } .$ For a given sequence of feedback delays, the summands in (B-5b), with $\bar { d } _ { t - a }$ replaced by $\bar { d } _ { t - a } ^ { \prime } .$ , are independent Bernoulli random variables, whose sum has mean

$$
\mathbb { E } _ { F } \left[ \sum _ { a = 1 } ^ { t - 1 } \mathbb { 1 } \left\{ d _ { t - a } \ge a \right\} \mathbb { 1 } \left\{ \bar { d } _ { t - a } ^ { \prime } \ge a \right\} \right] = \sum _ { a = 1 } ^ { t - 1 } \mathbb { 1 } \left\{ d _ { t - a } \ge a \right\} F ( a ) \le \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } F ( a ) = \ell _ { F } .\tag{B-6}
$$

By Assumption 4 and the multiplicative Chernoff bound [S6], we have

$$
\begin{array} { r l } & { \operatorname* { P r } _ { F } \{ | K _ { t } ^ { - } | = K \} \le \operatorname* { P r } _ { F } \left\{ \displaystyle \sum _ { a = 1 } ^ { t - 1 } \mathbb { 1 } \{ d _ { t - a } \ge a \} \mathbb { 1 } \{ \bar { d } _ { t - a } ^ { \prime } \ge a \} \ge K \right\} } \\ & { \qquad \le e ^ { - \ell _ { F } } \left( \displaystyle \frac { e \ell _ { F } } { K } \right) ^ { K } } \\ & { \qquad = \exp \left( - \ell _ { F } + K + K \ln ( \ell _ { F } / K ) \right) } \\ & { \qquad = \exp \left( - K h ( \ell _ { F } / K ) \right) . } \end{array}\tag{B-7}
$$

Averaging this bound over t completes the proof.

## C. Overall Bound

By Lemma B-1 with $t = T + d _ { \operatorname* { m a x } } ,$ , the inequality

$$
| r _ { 1 } - r _ { T + d _ { \operatorname* { m a x } } + 1 } | \leq B + \eta \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) }\tag{B-8}
$$

holds for any realization of the proxy delays. Hence, by Jensen’s inequality [S7], it also holds with $r _ { T + d _ { \operatorname* { m a x } } + 1 }$ replaced by $\mathbb { E } _ { F } [ r _ { T + d _ { \operatorname* { m a x } } + 1 } ]$

Furthermore, Lemma B-2 gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { P r } _ { F } \{ | \mathcal { K } _ { t } ^ { - } | = K \} \le e ^ { - K h ( \ell _ { F } / K ) } .\tag{B-9}
$$

Substituting both bounds into (A-8) completes the proof.

## APPENDIX C

## PROOFS OF PROPOSITION 1 AND COROLLARY 1

## A. Proof of Proposition 1

We first rewrite the optimization problem as

$$
\operatorname* { m i n } _ { \{ F ( a ) \} _ { a = - 1 } ^ { d _ { \operatorname* { m a x } } } } \frac { K + 1 } { F ( d _ { \operatorname* { m a x } } ) T } + e ^ { - K h ( \ell _ { F } / K ) }\tag{C-1a}
$$

$$
\mathrm { s . t . } \quad \ell _ { F } = \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } F ( a ) \leq K ,\tag{C-1b}
$$

$$
1 = F ( - 1 ) \geq F ( 0 ) \geq . . . \geq F ( d _ { \operatorname* { m a x } } ) \geq 0 .\tag{C-1c}
$$

Then, we fix $F ( d _ { \operatorname* { m a x } } ) = p \in ( 0 , 1 ]$ . Thus, the first term of the objective in problem (C-1) is fixed, while the second term depends on F only through the tracking load $\ell _ { F }$ in (C-1b). Since $F$ is nonincreasing, we have $F ( a ) \geq F ( d _ { \operatorname* { m a x } } ) = p$ for all $a \in \{ 1 , \cdots , d _ { \operatorname* { m a x } } \}$ , and hence the tracking load satisfies the lower bound

$$
\ell _ { F } = \sum _ { a = 1 } ^ { d _ { \operatorname* { m a x } } } F ( a ) \geq d _ { \operatorname* { m a x } } p .\tag{C-2}
$$

Furthermore, the value $F ( 0 ) \in [ p , 1 ]$ does not appear in the objective.

Since $h ^ { \prime } ( x ) = 1 - 1 / x < 0$ for $x \in ( 0 , 1 )$ , the function $h ( x )$ is decreasing for $x \in ( 0 , 1 ]$ , and the second term $e ^ { - K h ( \ell _ { F } / K ) }$ is thus nondecreasing in $\ell _ { F } \in ( 0 , K ]$ . Accordingly, by (C-2), for any fixed $p ,$ the objective of problem (C-1) is minimized by the proxy delay distribution with $F ( 1 ) = \cdots = F ( d _ { \operatorname* { m a x } } ) = p ,$ , which yields $\ell _ { F } = d _ { \mathrm { m a x } } p$

Therefore, problem (C-1) reduces to the one-dimensional problem

$$
J ( p ) = \frac { K + 1 } { p T } + e ^ { - K h ( d _ { \mathrm { m a x } } p / K ) } ,\tag{C-3}
$$

which completes the proof of Proposition 1.

## B. Proof of Corollary 1

Using the definition of $J ( p )$ in (C-3), we have

$$
\operatorname { R i s k G a p } ( T ) \leq { \frac { B } { \eta T } } + J ( p ^ { \star } ) .\tag{C-4}
$$

Since $p ^ { \star }$ minimizes $J ( p )$ over $( 0 , K / d _ { \mathrm { m a x } } ]$ , we have $J ( p ^ { \star } ) \leq J ( p )$ for the choice

$$
p = \frac { K } { e d _ { \operatorname* { m a x } } } \left( \frac { d _ { \operatorname* { m a x } } } { T } \right) ^ { \frac { 1 } { K + 1 } } ,\tag{C-5}
$$

which satisfies $p \leq K / d _ { \mathrm { m a x } }$ , since $d _ { \operatorname* { m a x } } \leq T$

For the choice (C-5), the first term of $J ( p )$ equals

$$
\frac { K + 1 } { p T } = \frac { e ( K + 1 ) } { K } \left( \frac { d _ { \operatorname* { m a x } } } { T } \right) ^ { \frac { K } { K + 1 } } \leq 2 e \left( \frac { d _ { \operatorname* { m a x } } } { T } \right) ^ { \frac { K } { K + 1 } } ,\tag{C-6}
$$

where the inequality follows from $( K + 1 ) / K \leq 2$ for $K \geq 1$ . Moreover, using the identity $e ^ { - K h ( x ) } = ( e x ) ^ { K } e ^ { - K x }$ with $x = d _ { \mathrm { m a x } } p / K = ( d _ { \mathrm { m a x } } / T ) ^ { 1 / ( K + 1 ) } / e ,$ , the second term of $J ( p )$ satisfies

$$
e ^ { - K h ( d _ { \mathrm { m a x } } p / K ) } \leq ( e x ) ^ { K } = \left( \frac { d _ { \mathrm { m a x } } } { T } \right) ^ { \frac { K } { K + 1 } } .\tag{C-7}
$$

Substituting (C-6) and (C-7) into (C-4) yields

$$
\mathrm { R i s k G a p } ( T ) \leq \frac { B } { \eta T } + ( 2 e + 1 ) \left( \frac { d _ { \operatorname* { m a x } } } { T } \right) ^ { \frac { K } { K + 1 } } ,\tag{C-8}
$$

which completes the proof of Corollary 1.

## APPENDIX D

## PROOF OF THEOREM 2

Similar to the proof of Theorem A-1, we first bound the range of the threshold iterates and the saturation term, and then combine the results to bound the risk gap.

## A. Bounding the Range of Threshold Iterates

Lemma D-1. Under Assumption $I ( b ) _ { \cdot }$ , Assumption 2, and Assumption 3 with $r _ { 1 } \in [ 0 , B ]$ , the thresholds of CA-OCRC satisfy the upper bound

$$
| \mathbb { E } _ { F , G } [ r _ { t + 1 } ] - r _ { 1 } | \le B + \eta R _ { F , G }\tag{D-1}
$$

for all $t \in \{ 1 , \dots , T + d _ { \operatorname* { m a x } } \}$ and for any sequence of data points, where the expectation is taken with respect to both the proxy delays and the feedback delays,

$$
R _ { F , G } = \bar { \alpha } \left( \sum _ { a = 0 } ^ { \bar { d } _ { \operatorname * { m a x } } } a \frac { G ( a ) - G ( a + 1 ) } { q _ { G } ( \bar { d } _ { \operatorname * { m a x } } ) } + 2 \right) + \frac { K { \bar { \alpha } } ^ { 2 } } { 2 ( 1 - \bar { \alpha } ) } Q _ { F , G } ,\tag{D-2}
$$

with $\bar { \alpha } = \operatorname* { m a x } \{ \alpha , 1 - \alpha \}$ , and

$$
Q _ { F , G } = \sum _ { a = 0 } ^ { \bar { d } _ { \operatorname* { m a x } } } \frac { G ( a ) - G ( a + 1 ) } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) ^ { 2 } F ( a ) } .\tag{D-3}
$$

Proof. Let $\bar { d } _ { k } ^ { \prime } \sim F$ denote the proxy delay that CA-OCRC would draw at round k if $\mathcal { C } _ { k }$ were nontrivial, so that the pairs $( d _ { k } , \bar { d } _ { k } ^ { \prime } )$ are independent across rounds, and let $\mathcal { G } _ { k }$ denote the σ-field generated by the pairs $( d _ { i } , \bar { d } _ { i } ^ { \prime } )$ with $i < k$ . Thus, the loss $L _ { k }$ and the indicator

$$
\beta _ { k } = \mathbb { 1 } \{ { \mathcal { C } } _ { k } \notin \{ \emptyset , { \mathcal { V } } \} \} \mathbb { 1 } \{ | { \mathcal { K } } _ { k } ^ { - } | < K \}\tag{D-4}
$$

are determined by $\mathcal { G } _ { k }$ . Round k is admitted if and only if $\beta _ { k } = 1$ and $\bar { d } _ { k } ^ { \prime } \geq 0$ , and its feedback is then received at round $k + d _ { k }$ if and only if $\bar { d } _ { k } ^ { \prime } \geq d _ { k }$

The feedback of an admitted round k with $L _ { k } > \alpha$ increases the threshold by $\eta ( L _ { k } - \alpha ) \pi _ { k } \leq$ $\eta ( 1 - \alpha ) \pi _ { k }$ , with

$$
\pi _ { k } = \frac { \mathbb { 1 } \{ \bar { d } _ { k } ^ { \prime } \geq d _ { k } \} } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( d _ { k } ) } ,\tag{D-5}
$$

where $\pi _ { k } = 0 { \mathrm { ~ i f ~ } } d _ { k } > { \bar { d } } _ { \operatorname* { m a x } } .$ , since $\bar { d } _ { k } ^ { \prime } \leq \bar { d } _ { \mathrm { m a x } }$ , while the feedback of an admitted round k with $L _ { k } \leq \alpha$ does not increase the threshold.

Fix $t \in \{ 1 , \dots , T + d _ { \operatorname* { m a x } } \}$ , and let $T _ { t } = \operatorname* { m i n } \{ t , T \}$ and $c = \alpha / ( 1 - \alpha )$ . Suppose that $r _ { t + 1 } > B$ and let $\tau = \operatorname* { m a x } \{ j \leq T _ { t } : r _ { j } \leq B \}$ , which is well defined since $r _ { 1 } \le B$ . As in the proof of Lemma B-1, every round j with $\tau < j \leq T _ { t }$ applies the update $- \eta \alpha$ , the virtual rounds apply no update of their own, and every upward push in rounds $\tau , \ldots , t$ is due either to round $\tau$ itself, if $ { \mathcal { C } } _ { \tau } = \emptyset$ , or to the feedback of a round $k \in \mathcal { K } _ { \tau } ^ { + }$ with $L _ { k } > \alpha$ , which is received at round $k + d _ { k } \ge \tau$ . Since $r _ { \tau } \le B$ and $| K _ { \tau } ^ { + } | \le K$ , we have

$$
\frac { ( r _ { t + 1 } - B ) _ { + } } { \eta ( 1 - \alpha ) } \leq 1 + \left[ \sum _ { k \in { \cal K } _ { \tau } ^ { + } } \pi _ { k } - c \left( T _ { t } - \tau \right) \right] _ { + } \leq 1 + \sum _ { k \in { \cal K } _ { \tau } ^ { + } } \left[ \pi _ { k } - \frac { c } { K } ( T _ { t } - \tau ) \right] _ { + } .\tag{D-6}
$$

Since every round $k \in \mathcal { K } _ { \tau } ^ { + }$ has $\beta _ { k } = 1$ , and since $T _ { t } - \tau \geq ( T _ { t } - k - d _ { k } ) .$ )<sub>+</sub> if $\pi _ { k } > 0$ , extending the sum to all rounds $k \leq T _ { t }$ yields

$$
\frac { ( r _ { t + 1 } - B ) _ { + } } { \eta ( 1 - \alpha ) } \leq 1 + \sum _ { k = 1 } ^ { T _ { t } } \beta _ { k } \left[ \pi _ { k } - \frac { c } { K } ( T _ { t } - k - d _ { k } ) _ { + } \right] _ { + } ,\tag{D-7}
$$

which also holds trivially if $r _ { t + 1 } \leq B$

Since $\beta _ { k }$ is determined by $\mathcal { G } _ { k }$ and the pair $( d _ { k } , \bar { d } _ { k } ^ { \prime } )$ is independent of $\mathcal { G } _ { k }$ , the expectation of the kth summand in (D-7) is at most $\begin{array} { r }  \sum _ { a = 0 } ^ { \bar { d } _ { \mathrm { m a x } } } \mathrm { P r } _ { G } [ d _ { t } = a \ | \ d _ { t } \le \bar { d } _ { \mathrm { m a x } } ] [ 1 - c q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) F ( a ) ( T _ { t } - k - \ \end{array}$ $a ) _ { + } / K ] _ { + }$ , since, given $d _ { k } = a \leq \bar { d } _ { \operatorname* { m a x } } .$ , we have $\pi _ { k } = 1 / ( q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) F ( a ) )$ with probability $F ( a )$ and $\pi _ { k } = 0$ otherwise. For each $^ { a , }$ at most $a + 1$ rounds $k \leq T _ { t }$ satisfy $T _ { t } - k - a \leq 0$ , while, for the remaining rounds, the bracket is evaluated at distinct positive integers $m = T _ { t } - k - a$ and $\sum _ { m > 1 } [ 1 - \epsilon m ] _ { + } \leq 1 / ( 2 \epsilon )$ for $\epsilon > 0$ . Summing over k thus gives

$$
\mathbb { E } _ { F , G } \left[ ( r _ { t + 1 } - B ) _ { + } \right] \leq \eta ( 1 - \alpha ) \left( \mathbb { E } _ { G } [ d _ { t } \mid d _ { t } \leq \bar { d } _ { \operatorname* { m a x } } ] + 2 + \frac { K ( 1 - \alpha ) } { 2 \alpha } Q _ { F , G } \right) .\tag{D-8}
$$

Symmetrically, with the roles of $\mathcal { C } _ { j } = \emptyset$ and $\mathcal { C } _ { j } = \mathcal { y }$ exchanged and c replaced by $1 / c$ , we have

$$
\mathbb { E } _ { F , G } \left[ ( - r _ { t + 1 } ) _ { + } \right] \leq \eta \alpha \left( \mathbb { E } _ { G } [ d _ { t } \mid d _ { t } \leq \bar { d } _ { \operatorname* { m a x } } ] + 2 + \frac { K \alpha } { 2 ( 1 - \alpha ) } Q _ { F , G } \right) .\tag{D-9}
$$

Finally, since $r _ { 1 } \in [ 0 , B ]$ , we have $| \mathbb { E } _ { F , G } [ r _ { t + 1 } ] - r _ { 1 } | \le B + \operatorname* { m a x } \{ \mathbb { E } _ { F , G } [ ( r _ { t + 1 } - B ) _ { + } ] , \mathbb { E } _ { F , G } [ ( - r _ { t + 1 } ) _ { + } ] \}$ The right-hand sides of (D-8) and (D-9) equal $\eta g ( 1 - \alpha )$ and $\eta g ( \alpha )$ , respectively, with $g ( x ) =$ $x ( \mathbb { E } _ { G } [ d _ { t } \ | \ d _ { t } \leq \bar { d } _ { \operatorname* { m a x } } ] + 2 ) + K x ^ { 2 } Q _ { F , G } / ( 2 ( 1 - x ) )$ , which is increasing in $x \in ( 0 , 1 )$ , so that the maximum equals $\eta g ( \bar { \alpha } ) = \eta R _ { F , G }$ □

## B. Bounding the Saturation Term

Lemma D-2. Under Assumption 1(b) and Assumption 5, the average probability of capacity saturation of CA-OCRC satisfies the upper bound

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { P r } _ { F , G } \{ | K _ { t } ^ { - } | = K \} \le \mathbb { 1 } \left\{ \bar { d } _ { \operatorname* { m a x } } \ge K \right\} e ^ { - K h ( \ell _ { F , G } / K ) }\tag{D-10}
$$

for any sequence of data points, where $h ( \cdot )$ is defined in Theorem 1, and $\ell _ { F , G }$ is defined in (31).

Proof. Based on the proxy delays, it holds that

$$
| { \cal K } _ { t } ^ { - } | \leq \sum _ { a = 1 } ^ { t - 1 } \mathbb { 1 } \{ d _ { t - a } \geq a \} \mathbb { 1 } \{ \bar { d } _ { t - a } \geq a \} .\tag{D-11}
$$

Since $\mathbb { 1 } \{ \bar { d } _ { t - a } \ \geq \ a \} \ = \ 0$ for $a \ > \ \bar { d } _ { \mathrm { m a x } }$ , the bound (D-11) implies $| \mathcal { K } _ { t } ^ { - } | \ \leq \ \bar { d } _ { \operatorname* { m a x } }$ , and hence $\mathrm { P r } _ { F , G } \{ | \mathcal { K } _ { t } ^ { - } | = K \} = 0 \ \mathrm { i f } \ \bar { d } _ { \mathrm { m a x } } < K$ . In the following, we thus consider the case $\bar { d } _ { \operatorname* { m a x } } \geq K$

As in the proof of Lemma B-2, the bound (D-11) also holds with ${ \bar { d } } _ { t } .$ <sub>−a</sub> replaced by the proxy delay $\bar { d } _ { t - a } ^ { \prime } \sim F$ that CA-OCRC would draw at round t − a if $\mathcal { C } _ { t - a }$ were nontrivial, and, by Assumption 1(b), the resulting summands are independent Bernoulli random variables.

The mean of the right-hand side of (D-11), with this replacement, is $\textstyle \sum _ { a = 1 } ^ { t - 1 } G ( a ) F ( a ) \leq \ell _ { F , G }$ The multiplicative Chernoff bound for sums of independent Bernoulli variables, $\operatorname* { P r } \{ x \geq R \} \leq$ $e ^ { - \nu } ( e \nu / R ) ^ { R }$ , is valid for $R \geq \nu$ and nondecreasing in $\nu \in [ 0 , R ]$ . So, since $\ell _ { F , G } \ \leq \ K$ by Assumption 5, applying the Chernoff bound with $\nu = \ell _ { F , G }$ and $R = K$ gives

$$
\begin{array} { r l } {  { \operatorname* { P r } _ { F , G } \{ | \mathcal { K } _ { t } ^ { - } | = K \} \le e ^ { - \ell _ { F , G } } \Big ( \frac { e \ell _ { F , G } } { K } \Big ) ^ { K } } } \\ & { = \exp \big ( - \ell _ { F , G } + K + K \ln ( \ell _ { F , G } / K ) \big ) } \\ & { = \exp \big ( - K h ( \ell _ { F , G } / K ) \big ) , } \end{array}\tag{D-12}
$$

for every t, and averaging over t completes the proof.

## C. Overall Bound

Lemma D-1 with $t = T + d _ { \mathrm { m a x } }$ and $r _ { 1 } \in [ 0 , B ]$ therefore yields

$$
\left| r _ { 1 } - \mathbb { E } _ { F , G } [ r _ { T + d _ { \operatorname* { m a x } } + 1 } ] \right| \leq B + \eta R _ { F , G } .\tag{D-13}
$$

By Lemma D-2, we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { P r } _ { F , G } \{ | \mathcal { K } _ { t } ^ { - } | = K \} \le \mathbb { 1 } \{ \bar { d } _ { \operatorname* { m a x } } \ge K \} e ^ { - K h \left( \ell _ { F , G } / K \right) }\tag{D-14}
$$

Substituting the above two bounds into Theorem A-1 proves Theorem 2.

## APPENDIX E

## PROOFS OF PROPOSITION 2 AND COROLLARY 2

In this appendix, we prove Proposition 2 and Corollary 2, which characterize the optimized proxy delay distribution and the resulting scaling law under stochastic delays, respectively.

## A. Proof of Proposition 2

If $\bar { d } _ { \mathrm { m a x } } < K$ , the optimization problem is given by

$$
\operatorname* { m i n } _ { F } \quad Q _ { F , G } = \sum _ { a = 0 } ^ { \bar { d } _ { \operatorname* { m a x } } } \frac { \mathrm { P r } _ { G } [ d _ { t } = a ] } { q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) ^ { 2 } F ( a ) }\tag{E-1}
$$

$$
\mathrm { s . t . } \quad 1 = F ( - 1 ) \geq F ( 0 ) \geq \cdots \geq F ( \bar { d } _ { \mathrm { m a x } } ) > 0 ,
$$

whose optimal solution is given by $F ^ { \star } ( a ) = 1$ for $a \in \{ - 1 , 0 , \dots , \bar { d } _ { \operatorname* { m a x } } \}$

In the rest of this proof, we consider the case $\bar { d } _ { \mathrm { m a x } } \geq K$ , where the optimization problem is rewritten as

$$
\operatorname* { m i n } _ { \{ F ( a ) \} _ { a = - 1 } ^ { \overline { { d } } _ { \operatorname* { m a x } } } } \frac { R _ { F , G } } { T } + e ^ { - K h \left( \ell _ { F , G } / K \right) }\tag{E-2a}
$$

$$
\mathrm { s . t . } \quad \ell _ { F , G } = \sum _ { a = 1 } ^ { \bar { d } _ { \operatorname* { m a x } } } F ( a ) G ( a ) \leq K ,\tag{E-2b}
$$

$$
1 = F ( - 1 ) \geq F ( 0 ) \geq . . . \geq F ( \bar { d } _ { \operatorname* { m a x } } ) > 0 .\tag{E-2c}
$$

To solve problem (E-2), we first reformulate it into an equivalent convex optimization problem and then derive its closed-form solution in the following four steps:

1) Transformation: The value $F ( 0 )$ only appears in the objective (E-2a) through $\mathrm { P r } [ d _ { t } =$ $0 ] / ( q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) ^ { 2 } F ( 0 ) )$ in $Q _ { F , G }$ . This term is nonincreasing in $F ( 0 )$ , hence, setting $F ( 0 ) = 1$ is optimal.

Define the vector $\pmb { x } = ( x _ { 1 } , \dots , x _ { \bar { d } _ { \operatorname* { m a x } } } )$ with $x _ { a } = F ( a )$ , which takes values in the convex set

$$
\mathcal { X } = \left\{ x : 1 \ge x _ { 1 } \ge \cdots \ge x _ { \bar { d } _ { \mathrm { m a x } } } > 0 \right\} .\tag{E-3}
$$

Define the functions

$$
f ( \pmb { x } ) = q _ { G } ( \bar { d } _ { \mathrm { m a x } } ) ^ { 2 } Q _ { F , G } = \mathrm { P r } _ { G } [ d _ { t } = 0 ] + \sum _ { a = 1 } ^ { \bar { d } _ { \mathrm { m a x } } } \frac { G ( a ) \bar { \kappa } ( a ) } { x _ { a } } ,\tag{E-4}
$$

$$
g ( { \pmb x } ) = \ell _ { F , G } = \sum _ { a = 1 } ^ { \bar { d } _ { \mathrm { m a x } } } G ( a ) x _ { a } ,\tag{E-5}
$$

where $f ( { \pmb x } )$ is convex and $g ( { \pmb x } )$ is linear over $\mathcal { X } .$

2) Convex Surrogate Problem: For any $\mu > 0$ , consider the following surrogate problem:

$$
\operatorname* { m i n } _ { \pmb { x } } f ( \pmb { x } ) + \mu g ( \pmb { x } )\tag{E-6a}
$$

$$
\mathrm { s . t . } \quad { \pmb x } \in { \pmb \chi } .\tag{E-6b}
$$

Since $f ( { \pmb x } )$ is convex, $g ( { \pmb x } )$ is linear, and $\mathcal { X }$ is convex, problem (E-6) is convex for any fixed $\mu > 0$

3) Closed-Form Solution: For fixed $\mu > 0$ , the objective of problem (E-6) can be written as

$$
f ( \pmb { x } ) = \mathrm { P r } _ { G } [ d _ { t } = 0 ] + \sum _ { a = 1 } ^ { { \bar { d } } _ { \operatorname* { m a x } } } G ( a ) \left( \frac { \bar { \kappa } ( a ) } { x _ { a } } + \mu x _ { a } \right) .\tag{E-7}
$$

Define $z _ { a } = \mu x _ { a } ^ { 2 }$ , which is strictly increasing in $x _ { a } > 0$ . Each summand can be expressed as

$$
G ( a ) \left( \frac { \bar { \kappa } ( a ) } { x _ { a } } + \mu x _ { a } \right) = \sqrt { \mu } G ( a ) \left[ D _ { \Phi } ( \bar { \kappa } ( a ) , z _ { a } ) - \Phi ( \bar { \kappa } ( a ) ) \right] ,\tag{E-8}
$$

where

$$
D _ { \Phi } ( y , z ) = \Phi ( y ) - \Phi ( z ) - \Phi ^ { \prime } ( z ) ( y - z )\tag{E-9}
$$

is the Bregman divergence associated with the strictly convex function $\Phi ( z ) = - 2 { \sqrt { z } } .$

Since $\Phi ( \bar { \kappa } ( a ) )$ does not depend on $z _ { a } .$ , problem (E-6) is equivalent to the weighted Bregman isotonic regression problem

$$
\operatorname* { m i n } _ { z } \ \sum _ { a = 1 } ^ { { \bar { d } } _ { \operatorname* { m a x } } } G ( a ) D _ { \Phi } ( \bar { \kappa } ( a ) , z _ { a } )\tag{E-10a}
$$

$$
\mathrm { s . t . } \quad \mu \geq z _ { 1 } \geq \cdot \cdot \cdot \geq z _ { \bar { d } _ { \operatorname* { m a x } } } > 0 ,\tag{E-10b}
$$

whose optimal solution is given by

$$
z _ { a } = \operatorname* { m i n } \{ \mu , \kappa ( a ) \} ,\tag{E-11}
$$

with

$$
\kappa ( a ) = \operatorname* { m i n } _ { 1 \leq s \leq a } \operatorname* { m a x } _ { a \leq s ^ { \prime } \leq \bar { d } _ { \operatorname* { m a x } } } \frac { G ( s ) - G ( s ^ { \prime } + 1 ) } { \sum _ { i = s } ^ { s ^ { \prime } } G ( i ) } .\tag{E-12}
$$

Thus, the solution to problem (E-6) is

$$
x _ { \mu , a } = \operatorname* { m i n } \left\{ 1 , { \sqrt { \frac { \kappa ( a ) } { \mu } } } \right\} .\tag{E-13}
$$

4) Reduction to One-Dimensional Search: We now show that the family of solutions in (E-13) contains an optimal solution to the original problem.

Note that $G ( a ) \geq \mathrm { P r } _ { G } [ d _ { t } = \bar { d } _ { \operatorname* { m a x } } ] > 0$ for $a = 1 , \ldots , \bar { d } _ { \mathrm { m a x } }$ , which implies that $\bar { \kappa } ( a ) \geq 0$ and $\kappa ( \bar { d } _ { \mathrm { m a x } } ) > 0$ . Based on (E-12), it is straightforward to verify that $\kappa ( a ) > 0$

By (E-13), the tracking load of the surrogate solution is

$$
g ( \pmb { x } _ { \mu } ) = \sum _ { a = 1 } ^ { \bar { d } _ { \operatorname* { m a x } } } G ( a ) \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { \kappa ( a ) } { \mu } } \right\} .\tag{E-14}
$$

This function is continuous and nonincreasing in $\mu > 0$ . Thus, we have

$$
\begin{array} { l l l } { { \displaystyle \operatorname* { l i m } _ { \mu \to 0 ^ { + } } g ( { \pmb x } _ { \mu } ) = \sum _ { a = 1 } ^ { \bar { d } _ { \operatorname* { m a x } } } G ( a ) = \mathbb E _ { G } [ \operatorname* { m i n } \{ d _ { t } , \bar { d } _ { \operatorname* { m a x } } \} ] } , }  \\ { { \displaystyle \operatorname* { l i m } _ { \mu \to \infty } g ( { \pmb x } _ { \mu } ) = 0 } . } \end{array}\tag{E-15}
$$

Besides, for any $\mathbf { \boldsymbol { x } } \in \mathcal { X } .$ , we have $0 < g ( \pmb { x } ) \leq \mathbb { E } _ { G } [ \operatorname* { m i n } \{ d _ { t } , \bar { d } _ { \operatorname* { m a x } } \} ]$ . Therefore, by continuity, there exists $\mu > 0$ such that

$$
g ( { \pmb x } _ { \mu } ) = g ( { \pmb x } ) .\tag{E-16}
$$

Since $\scriptstyle { \pmb { x } } _ { \mu }$ minimizes $f ( { \pmb x } ) + \mu g ( { \pmb x } )$ over $\mathcal { X } ,$ , we have

$$
f ( \pmb { x } _ { \mu } ) + \mu g ( \pmb { x } _ { \mu } ) \leq f ( \pmb { x } ) + \mu g ( \pmb { x } ) .\tag{E-17}
$$

Combining this inequality with (E-16) yields

$$
f ( { \pmb x } _ { \mu } ) \leq f ( { \pmb x } ) .\tag{E-18}
$$

The term $R _ { F , G }$ is nondecreasing in $f ( { \pmb x } )$ , while the exponential term depends on x only through $g ( { \pmb x } )$ . Hence, (E-16) and (E-18) imply

$$
\frac { R _ { F _ { \mu } , G } } { T } + e ^ { - K h ( g ( { \pmb x } _ { \mu } ) / K ) } \le \frac { R _ { F , G } } { T } + e ^ { - K h ( g ( { \pmb x } ) / K ) } .\tag{E-19}
$$

Moreover, if x satisfies the tracking-load constraint $g ( { \pmb x } ) \le K$ , then $\scriptstyle { \pmb { x } } _ { \mu }$ also satisfies this constraint by (E-16). Thus, for every feasible solution to problem (E-2), there exists a feasible member of the surrogate family with the same tracking load and no larger objective value.

Finally, substituting $\theta = 1 / \sqrt { \mu }$ into (E-13) gives

$$
x _ { \theta , a } = \operatorname* { m i n } \{ 1 , \theta { \sqrt { \kappa ( a ) } } \} .\tag{E-20}
$$

Accordingly, problem (E-2) is equivalently reduced to the one-dimensional problem

$$
\operatorname* { m i n } _ { \theta > 0 } \quad \frac { R _ { F _ { \theta } , G } } { T } + e ^ { - K h \left( \ell _ { F _ { \theta } , G } / K \right) }\tag{E-21a}
$$

$$
{ \mathrm { s . t . } } \quad \ell _ { F _ { \theta } , G } \leq K ,\tag{E-21b}
$$

where $F _ { \theta }$ denotes the proxy law in (36) with $\theta ^ { \star } = \theta$

Therefore, a global optimizer $\theta ^ { \star }$ of (E-21) yields a global optimizer of problem (E-2), which completes the proof of Proposition 2.

## B. Proof of Corollary 2

Since the optimized maximum proxy delay $\bar { d } _ { \mathrm { m a x } }$ and the proxy delay distribution in Proposition 2 minimize the bound (32), it suffices to bound (32) for $\bar { d } _ { \mathrm { m a x } } = d _ { \mathrm { m a x } } , F ( 0 ) = 1$ , and $F ( a ) = p$ for $a \in \{ 1 , \dots , d _ { \operatorname* { m a x } } \}$ , with $p \in ( 0 , 1 ]$ and $p \mathbb { E } _ { G } [ d _ { t } ] \leq K$ . For this choice, we have $q _ { G } ( \bar { d } _ { \operatorname* { m a x } } ) = 1 , \mathbb { E } _ { G } [ d _ { t } \mid d _ { t } \leq \bar { d } _ { \operatorname* { m a x } } ] = \mathbb { E } _ { G } [ d _ { t } ] , Q _ { F , G } \leq 1 / p .$ , and $\ell _ { F , G } = p \mathbb { E } _ { G } [ d _ { t } ]$ . Therefore, by Theorem 2, we have

$$
\mathrm { R i s k G a p } ( T ) \leq \frac { B } { \eta T } + \frac { \bar { \alpha } \bigl ( \mathbb { E } _ { G } [ d _ { t } ] + 2 \bigr ) } { T } + \frac { \bar { \alpha } ^ { 2 } } { 2 ( 1 - \bar { \alpha } ) } \frac { K } { p T } + e ^ { - K h ( p \mathbb { E } _ { G } [ d _ { t } ] / K ) } .\tag{E-22}
$$

Similar to the proof of Corollary 1 in Appendix C-B, we choose

$$
p = \frac { K } { e \mathbb { E } _ { G } [ d _ { t } ] } \left( \frac { \mathbb { E } _ { G } [ d _ { t } ] } { T } \right) ^ { \frac { 1 } { K + 1 } } ,\tag{E-23}
$$

which satisfies $p \leq 1$ and $p \mathbb { E } _ { G } [ d _ { t } ] \leq K$ , since $K \leq \mathbb { E } _ { G } [ d _ { t } ] \leq T$ , and which yields

$$
e ^ { - K h ( p \mathbb { E } _ { G } [ d _ { t } ] / K ) } \leq ( e x ) ^ { K } = \left( \frac { \mathbb { E } _ { G } [ d _ { t } ] } { T } \right) ^ { \frac { K } { K + 1 } } .\tag{E-24}
$$

Substituting these bounds into (E-22), together with $\mathbb { E } _ { G } [ d _ { t } ] / T \le ( \mathbb { E } _ { G } [ d _ { t } ] / T ) ^ { K / ( K + 1 ) }$ , which completes the proof of Corollary 2.

## REFERENCES

[S1] H. Qiu, E. Esposito, and M. Zhang, “Exploiting curvature in online convex optimization with delayed feedback,” in Proc. Int. Conf. Mach. Learn., 2025, pp. 50 448–50 479.

[S2] X. Cao, J. Zhang, and H. V. Poor, “Constrained online convex optimization with feedback delays,” IEEE Trans. Autom. Control, vol. 66, no. 11, pp. 5049–5064, 2020.

[S3] B. Wang, M. Zecchin, and O. Simeone, “Distributed dynamic associative memory via online convex optimization,” IEEE J. Sel. Areas Commun., vol. 44, pp. 5872–5887, 2026.

[S4] ——, “Mirror online conformal prediction with intermittent feedback,” IEEE Signal Process. Lett., vol. 32, pp. 2888–2892, 2025.

[S5] M. Zhao, H. Admoni, R. Simmons, A. Ramdas, and A. Bajcsy, “Conformalized interactive imitation learning: Handling expert shift and intermittent feedback,” in Proc. Int. Conf. Learn. Represent., 2025, pp. 9238–9251.

[S6] R. Vershynin, High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, 2018.

[S7] R. Durrett, Probability: theory and examples. Cambridge university press, 2019, vol. 49.