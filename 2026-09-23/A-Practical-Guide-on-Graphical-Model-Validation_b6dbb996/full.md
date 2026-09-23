# A Practical Guide on Graphical Model Validation

Mario V. W¨uthrich<sup>∗</sup>

September 23, 2026

## Abstract

This manuscript formalizes the most popular model validation tools used in general insurance actuarial modeling. These include graphical tools like calibration plots, actual-vs-expected plots, lift charts, Murphy diagrams, as well as classical statistical tools such as Bregman losses, deviance losses, elementary losses, Murphy’s decomposition and Gini scores. Particular emphasis is placed on whether calibration and discrimination are studied under a policy-weighted or an exposure-weighted population measure. This distinction is crucial in ensuring that premium schemes are calibrated on the correct scale.

Keywords. Calibration, resolution, discrimination, risk ranking, actuarial pricing, calibration plot, lift chart, actual-vs-expected plot, Murphy graph, Bregman loss, deviance loss, Murphy’s decomposition, Gini score, population distribution, strictly consistent scoring, exponential dispersion family.

## 1 Introduction

The main purpose of this manuscript is to discuss several methods that are useful for model validation in general insurance actuarial pricing. We present graphical tools like calibration plots, actual-vs-expected plots, lift charts, Murphy diagrams, as well as classical statistical tools such as Bregman losses, deviance losses, elementary losses, Murphy’s decomposition and Gini scores.

In the presentation of these model validation tools, the whole discussion will be centered around calibration, discrimination (resolution) and risk ranking. These are essential features that actuarial pricing models should possess. A crucial point in these considerations is the correct scale and probability measure. We discuss the diference between the policy-weighted population measure and the exposure-weighted population measure. This distinction is important in actuarial pricing, but only the recent paper of Lindholm et al. [43] systematically discusses this distinction. In fact, many papers discuss the theory under a policy-weighted view and then present an applied actuarial example in the exposure-weighted view. The alignment of these two views requires additional assumptions; we discuss these in Section 3.

We emphasize that none of the methods presented in this paper is new. They have been floating around for a long time in actuarial practice and in the applied actuarial literature. However, often, they are not properly mathematically formalized. This is precisely the main contribution of this manuscript. We bring all these diferent methods and concepts on a common mathematical ground which facilitates understanding and comparison. For example, what does it mathematically mean to perform insurance policy binning resulting in bins of equal exposures, and how can we write this under the correct population measure? Having a common mathematical view on these diferent concepts, we will realize that many of them present the similar statistics in slightly diferent views.

Literature overview. We skip the literature review at this stage, but all the important references are cited throughout the manuscript whenever they are relevant.

AI declaration. The author developed the concepts, mathematics, numerical examples, and the original manuscript. ChatGPT-5.6 Sol was used to assist through several iterations of reviewing and revisions of the manuscript.

Acknowledgement. The author kindly thanks Alexej Brauer, Lukasz Delong, Selim Gatti, Mathias Lindholm, Filip Lindskog, Christian Lorentzen, Marco Maggi, Michael Mayer, Ronald Richman and Dimitri Semenovich for discussing and challenging many of the items presented in this manuscript.

## 2 Problem setting

## 2.1 Loss costs per unit exposure

An insurance policy is observed for an exposure period $V > 0$ , and it generates a total loss Z during that period. This yields the loss costs per unit exposure $Y = Z / V$ . Equivalently, we scale the total premium P of that contract resulting in the unit premium $\Pi = P / V$ . This normalization makes insurance contracts of diferent exposure lengths comparable. For general insurance pricing, one is then equipped with the random tuple $( Y , V , \Pi )$

$Y \geq 0$ describes the non-negative loss costs per unit exposure,

$V > 0$ is a strictly positive exposure (sometimes also called case weight), and

$\Pi \geq 0$ is the positive unit premium.

Actuarial modeling considers these normalized quantities, and these yield:

• the total loss $Z = V Y$ , and

• the total premium $P = V \Pi$

The statistical literature is often not considering such a split of the total loss Z into loss costs per unit exposure $Y = Z / V$ and an exposure V . In actuarial modeling, this split is very common and useful to compare insurance policyholders with contracts of diferent exposure lengths. From a mathematical viewpoint, this split introduces some complications that we discuss in this section and in Section 3, below.

Remarks 2.1 • Insurance policies and their premiums are usually characterized by covariates (features) X. The above setting covers this situation, because we can think of the unit premium Π = Π(X) being a measurable function of the covariates (policy characteristics).

• The premiums Π and P are always understood as pure risk premiums in this manuscript. That is, these premiums cover the expected loss costs, and they do not contain any additional margins, e.g., for administrative expenses, solvency costs or a profit margin.

• We treat (Y, V, Π) as a random tuple on an underlying probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ . We interpret P as the population distribution. A realization of (Y, V, Π) then corresponds to a randomly selected insurance policy from that population. Naturally, loss costs Y and unit premiums Π should be positively associated for Π to be a meaningful risk-based pricing rule. The bigger this association the better the premium matches the loss costs, this dependence is implicitly reflected by selecting a suitable population distribution P. This association is formalized and quantified by the resolution term in Murphy’s decomposition in Section 6.

• Throughout, we assume that all considered moments exist and are finite.

## 2.2 The exposure-weighted Q-measure

An insurance policy is described by the random tuple (Y, V, Π) that follows the population distribution P. As emphasized by Lindholm et al. [43], working with exposure-scaled quantities requires to work under a second distribution Q which accounts for the exposure scaling.

=⇒ This manuscript mainly works under the exposure-weighted distribution Q.

Briefly explained: There are two diferent ways of averaging across an insurance portfolio:

• The policy-weighted average assigns the same weight to each policy when computing averages: this is described by the population measure P.

• The exposure-weighted average considers an exposure-weighted average accounting for the diferent exposure lengths: this is described by the population measure Q.

Working with exposure-scaled quantities Y and Π requires ensuring that premium computations capture the dependence between loss costs and exposures. This is achieved by introducing the following exposure-weighted probability measure Q. This step simplifies many of the subsequent considerations.

Let E[·] be the expectation under the population measure P. We define the exposure-weighted population measure $\mathbb { Q }$ by the Radon–Nikodym derivative

$$
{ \frac { \mathrm { d } \mathbb { Q } } { \mathrm { d } \mathbb { P } } } = { \frac { V } { \mathbb { E } [ V ] } } .\tag{2.1}
$$

This exposure-weighted measure $\mathbb { Q } \sim \mathbb { P }$ is an equivalent probability measure and we denote its expectation operator by $\mathbb { E } _ { \mathbb { Q } } [ \cdot ]$ . The Q-probability of an event $A \in { \mathcal { F } }$ is computed as

$$
\mathbb { Q } \left( A \right) = \mathbb { E } _ { \mathbb { Q } } \left[ \mathbf { 1 } _ { A } \right] = \frac { 1 } { \mathbb { E } \left[ V \right] } \mathbb { E } \left[ V \mathbf { 1 } _ { A } \right] .
$$

Our main object of interest are the exposure-weighted loss costs

$$
\mathbb { E } _ { \mathbb { Q } } \left[ Y \right] = \frac { 1 } { \mathbb { E } \left[ V \right] } \mathbb { E } \left[ V Y \right] = \frac { 1 } { \mathbb { E } \left[ V \right] } \mathbb { E } \left[ Z \right] .\tag{2.2}
$$

These expected loss costs per unit exposure $Y$ under $\mathbb { Q } ,$ given by (2.2), are directly related to the total loss $Z = V Y$ under P, and the dependence between $Y$ and $V$ is correctly accounted for in (2.2). Similarly, we have for the total premium P under P

$$
\begin{array} { r } { \mathbb { E } \left[ P \right] = \mathbb { E } \left[ V \Pi \right] = \mathbb { E } \left[ V \right] \mathbb { E } _ { \mathbb { Q } } \left[ \Pi \right] . } \end{array}
$$

Consequently, global unbiasedness of P for Z has the two equivalent formulations

$$
{ \mathbb { E } } \left[ Z \right] = { \mathbb { E } } \left[ P \right] \qquad \Longleftrightarrow \qquad { \mathbb { E } } _ { \mathbb { Q } } \left[ Y \right] = { \mathbb { E } } _ { \mathbb { Q } } \left[ \Pi \right] .\tag{2.3}
$$

Why does the exposure-weighted Q-measure matter? It plays a crucial role in insurance pricing, model fitting and model validation because it correctly accounts for the dependence between the loss costs $Y$ and the exposure $V .$ For example, under positive correlation between $Y$ and $V$ under $\mathbb { P } ,$ , the above computations imply

$$
\mathbb { E } _ { \mathbb { Q } } \left[ Y \right] = \frac { 1 } { \mathbb { E } \left[ V \right] } \mathbb { E } \left[ Z \right] > \mathbb { E } \left[ Y \right] .\tag{2.4}
$$

That is, under positive correlation, the expected loss costs $\mathbb { E } [ Y ]$ systematically underestimate the scaled expected total loss $\mathbb { E } [ Z ] / \mathbb { E } [ V ]$ , and we would charge a too low insurance premium if we used the quantity E[Y ] for pricing.

Example 2.2 (Model fitting under P and Q) The present notes are mainly dedicated to model validation. Nevertheless, we briefly illustrate that the exposure-weighted Q-measure also matters for model fitting. Select the mean squared loss – the general theory of strictly consistent loss functions is presented in Section 5.1, below. Assuming square-integrability, the expected value of Y is found by solving the minimization problem

$$
\mathbb { E } [ Y ] \ = \ \underset { x \in \mathbb { R } } { \arg \operatorname* { m i n } } \ \mathbb { E } \left[ ( Y - x ) ^ { 2 } \right] .
$$

As described in (2.4), this expected value $\mathbb { E } [ Y ]$ is not directly useful for pricing the total loss $Z ,$ because it does not capture the dependence structure of the loss costs Y and the exposure V. One therefore generally considers the minimization problem

$$
\mathbb { E } _ { \mathbb { Q } } [ Y ] \ = \ \underset { x \in \mathbb { R } } { \mathrm { a r g } \mathrm { m i n } } \ \mathbb { E } _ { \mathbb { Q } } \left[ ( Y - x ) ^ { 2 } \right] \ = \ \underset { x \in \mathbb { R } } { \mathrm { a r g } \mathrm { m i n } } \ \mathbb { E } \left[ V \left( Y - x \right) ^ { 2 } \right] .\tag{2.5}
$$

This gives the expected total claim

$$
\mathbb { E } [ V ] \mathbb { E } _ { \mathbb { Q } } [ Y ] = \mathbb { E } [ Z ] .
$$

The right-hand side of minimization (2.5) justifies why in the context of actuarial model fitting and validation, the considered loss function $L ( Y , x )$ is always scaled with the exposure V . Namely, this reflects minimization under the exposure-weighted Q-measure. This completes the example. ■

Below, we also need the conditional $\mathbb { Q } \mathrm { - }$ expectation. For a sub-σ-field ${ \mathcal { A } } \subset { \mathcal { F } }$ , it is given by

$$
\operatorname { \mathbb { E } _ { \mathbb { Q } } } \left[ X \mid { \mathcal { A } } \right] = { \frac { 1 } { \operatorname { \mathbb { E } } [ V \mid { \mathcal { A } } ] } } \operatorname { \mathbb { E } } \left[ V X \mid { \mathcal { A } } \right] ;\tag{2.6}
$$

this is known as Bayes’ rule for conditional expectations, and it reflects prediction under partial information ${ \mathcal { A } } .$ The special case of an A-measurable exposure V yields

$$
\mathbb { E } _ { \mathbb { Q } } \left[ X \mid { \mathcal A } \right] = \mathbb { E } \left[ X \mid { \mathcal A } \right] \qquad \mathrm { ~ f o r ~ } V ~ \mathrm { b e i n g ~ } { \mathcal A } \mathrm { \mathrm { - m e a s u r a b l e } } .\tag{2.7}
$$

Thus, when the exposure V is observable w.r.t. the information ${ \mathcal { A } } ,$ we can equivalently work with the conditional population distribution $\mathbb { P } ( \cdot \mid A )$ or with the conditional exposure-scaled distribution $\mathbb { Q } ( \cdot \mid A )$ . We frequently use this property below.

## 3 Calibration

There are two main properties that non-egalitarian insurance premium schemes should fulfil: Discrimination and Calibration.

• Discrimination and resolution concerns the ability of an insurance pricing scheme to distinguish low-risk insurance policies from high-risk policies. This may result in a fine-grained tarif that properly classifies propensity to claims among the considered insurance policyholders. The topic of discrimination and resolution is studied in Sections 5-6, below. Implicitly, discrimination and resolution also involves a correct risk ranking which is discussed in Section 7, below.

• Calibration considers the question whether the total premium P = VΠ charged is suficient to cover the total loss $Z = V Y$ . A crude answer is to require global unbiasedness (2.3), meaning that on the population level the expected total premium covers the expected total loss. In most of the cases – excluding an egalitarian pricing system – we want to have calibration at a finer resolution so that systematic cross-financing between diferent price cohorts is avoided. This is the topic of this section.

## 3.1 Notions of calibration

Global unbiasedness is a requirement at the population level to ensure that the overall premium level is correct. We typically want stronger properties at a finer granularity.

Definition 3.1 (PZ-calibration) The total premium P is PZ-calibrated for the total loss Z $i f$

$$
P = \mathbb { E } \left[ Z \mid P \right] \quad \quad \quad a . s .\tag{3.1}
$$

P Z-calibration (3.1) tells us that every price cohort $P$ is on average self-financing for their total loss $Z ,$ and there is no systematic cross-financing between diferent price cohorts. This is important, but from an actuarial view point, P Z-calibration (3.1) is not fully satisfactory because it does not disentangle the role of the unit premium Π and the exposure V – low unit premium policies are interpreted as low-risk policies, and high-risk policies have a high unit premium. Only considering the total premium P, we may have low-risk policies with long exposures and high-risk policies with short exposures in the same price cohort P. Naturally, we should also try to disentangle potential within-price-cohort cross-subsidy in that case, because high-risk policies should not be systematically cross-financed by low-risk ones, or vice versa. For this reason, we are interested in understanding calibration at the unit premium level Π. This motivates the following definition.

Definition 3.2 (Q-calibration) The unit premium Π is Q-calibrated for (Y, V ) if

$$
\Pi = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi \right] \quad \quad \quad \quad a . s .
$$

Reformulating Q-calibration yields the equivalent formulations

$$
\Pi = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi \right] \qquad \iff \qquad \mathbb { E } \left[ P \mid \Pi \right] = \mathbb { E } \left[ Z \mid \Pi \right]\tag{3.2}
$$

Because generally the σ-fields generated by P = VΠ and by Π do not coincide, PZ-calibration (3.1) and Q-calibration (3.2) are diferent. The latter says that the expected premium collected for any unit premium cohort Π is on average self-financing to cover their total loss Z.

Q-calibration (3.2) is also diferent from P-calibration which is the classical definition of calibration in statistics. P-calibration is given by

$$
\Pi = \mathbb { E } \left[ Y \mid \Pi \right] \mathrm { ~ a . s . }\tag{3.3}
$$

Q-calibration (3.2) accounts for the dependence in (Π, V ) and in $( Y , V )$ , whereas P-calibration (3.3) does not, see (2.4). Therefore, we are generally not interested in P-calibration (3.3).

There is yet another peculiarity in actuarial pricing, namely, the information about the premium Π, the exposure V and the loss costs Y becomes available at diferent time points. Often, the premium Π and the exposure $V$ are available at contract inception – we call this ex-ante – and the total loss $Z = V Y$ is only available ex-post, when the contract expires. In this situation, one may evaluate calibration on the pair (Π, V ); we further discuss the role of an ex-ante available exposure in Remark 3.8, below. Based on ex-ante information (Π, V), there is the following calibration definition.

Definition 3.3 (Q|<sub>V</sub>-calibration) The unit premium Π is Q|<sub>V</sub>-calibrated for (Y, V ) if

$$
\Pi = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi , V \right] \quad \quad \quad \quad a . s .
$$

Reformulating $\mathbb { Q } | _ { V }$ -calibration (3.4) yields the equivalent formulations

$$
\Pi = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi , V \right] \quad \Longleftrightarrow \quad \Pi = \mathbb { E } \left[ Y \mid \Pi , V \right] \quad \Longleftrightarrow \quad P = \mathbb { E } \left[ Z \mid \Pi , V \right]\tag{3.4}
$$

The first equivalence says that conditional on (Π, V ), the loss costs can either be evaluated under the exposure-weighted measure $\mathbb { Q }$ or the original population measure $\mathbb { P } ;$ this follows directly from (2.7). The second equivalence expresses that in this case it does not matter whether we consider total losses $Z$ or unit loss costs $Y .$ . In this sense, Q| -calibration could also be called exposureconditional calibration because it does not depend on the considered probability measure $\mathbb { P }$ or Q.

Summary 3.4 We summarize the diferent definitions of calibration:

• $P Z .$ -calibration (3.1): $P = \mathbb { E } [ Z \mid P ]$ .

• Q-calibration (3.2): $\Pi = \mathbb { E } _ { \mathbb { Q } } [ Y \mid \Pi ] \iff \mathbb { E } [ P \mid \Pi ] = \mathbb { E } [ Z \mid \Pi ] .$

• P-calibration (3.3): $\Pi = \mathbb { E } [ Y \mid \Pi ]$ .

• Q|<sub>V</sub>-calibration (3.4): $\Pi = \mathbb { E } _ { \mathbb { Q } } [ Y \mid \Pi , V ] \iff \Pi = [ Y \mid \Pi , V ] \iff P = [ Z \mid \Pi , V ] .$

The following properties are proved by the tower-property of conditional expectation.

Proposition 3.5 We have the following three implications:

$$
\begin{array} { r c l } { { \mathbb Q | _ { V ^ { - } } c a l i b r a t i o n ~ ( 3 . 4 ) } } & { { \Longrightarrow } } & { { \mathbb Q \cdot c a l i b r a t i o n ~ ( 3 . 2 ) } } \\ { { \mathbb Q | _ { V ^ { + } } c a l i b r a t i o n ~ ( 3 . 4 ) } } & { { \Longrightarrow } } & { { \mathbb P \cdot c a l i b r a t i o n ~ ( 3 . 3 ) } } \\ { { \mathbb Q | _ { V ^ { - } } c a l i b r a t i o n ~ ( 3 . 4 ) } } & { { \Longrightarrow } } & { { P Z \cdot c a l i b r a t i o n ~ ( 3 . 1 ) } } \end{array}
$$

Opposite implications or further implications require additional assumptions. Note that currently we have not made any model assumptions, except that all considered quantities have finite means and that the exposure is strictly positive, a.s. Thus, Proposition 3.5 holds in full generality. We next make a conditional mean independence assumption – this is our first model assumption – it is further discussed in Remark 3.8, below, and we will abandon it again because it is not generally satisfied in practice.

Assumption 3.6 (Conditional mean independence) Assume that

$$
\operatorname { \mathbb { E } } \left[ Y \mid \Pi , V \right] = \operatorname { \mathbb { E } } \left[ Y \mid \Pi \right] \quad \quad \quad \quad a . s .\tag{3.5}
$$

Under this additional assumption (3.5) we have the following equivalences.

Proposition 3.7 We have the following equivalences:

$$
\begin{array} { r l l } { \mathbb { Q } | _ { V ^ { - } } c a l i b r a t i o n ~ ( 3 . 4 ) } & { \iff } & { \mathbb { Q } \_ c a l i b r a t i o n ~ ( 3 . 2 ) + \ c o n d i t i o n a l ~ m e a n ~ i n d e p e n d e n c e ~ ( 3 . 5 ) } \\ & { \iff } & { \mathbb { P } _ { - c a l i b r a t i o n ~ ( 3 . 3 ) } + \ c o n d i t i o n a l ~ m e a n ~ i n d e p e n d e n c e ~ ( 3 . 5 ) . } \end{array}
$$

Consequently, under conditional mean independence, the three definitions of Q|<sub>V</sub>-calibration, Q-calibration and P-calibration coincide, but this requires Assumption 3.6, otherwise only the implications of Proposition 3.5 hold.

Remark 3.8 (Ex-ante exposures and conditional mean independence) We discuss the two items: (1) ex-ante vs. ex-post exposures – ex-ante exposures are implicitly used in $\mathbb { Q } | _ { V ^ { - } }$ calibration (3.4) to make it a practically meaningful object – and (2) the conditional mean independence assumption (3.5). The question whether items (1) and (2) are realistic assumptions in practice is closely related.

(1) Ex-ante availability is an information-timing condition. Q|<sub>V</sub>-calibration (3.4) is a meaningful consideration if the exposure V is known ex-ante, meaning at contract inception. This is a general assumption made in many actuarial modeling approaches, and it looks reasonable because the insurance contract specifies the insurance term. However, many insurance products contain a lapse option, e.g., a car insurance policy can be terminated in case the policyholder changes their vehicle. Consequently, the realized exposure only becomes available at contract expiry/termination (ex-post); this is a critical point raised and discussed in Lindholm et al. [43]. Thus, early termination and temporary suspension turns the exposure into an ex-post available variable, and in this situation it seems more meaningful to study Q-calibration (3.2) because (Π, V) is not in the available information set at contract inception.

(2) The conditional mean independence is a stochastic relationship. The conditional mean independence assumption (3.5) seems to be even more problematic in practical applications. We highlight some crucial features in the following list:

– Property (3.5) is a conditional mean independence, this is weaker than conditional independence. It says that on each unit premium level Π, there is no systematic efect of the exposure V on the expected loss costs per unit exposure. This is interpreted that Π is mean-suficient for Y , and the exposure V does not proxy a missing risk factor for unit loss cost prediction. In particular, we have a proportional mean behavior in V of the total loss Z.

– It is important to realize that (3.5) is a substantive modeling assumption, and not an automatic consequence of defining $Y = Z / V$ . That is, the unit loss costs can always be defined by $Y = Z / V$ , but this does not imply that the expected loss has a linearity in the exposure; see also the correlation statement (2.4). This linearity is a substantial assumption that can be imposed in the form of (3.5). Most classical actuarial models impose a linearity assumption, e.g., working within the exponential dispersion family (EDF), one typically assumes that conditionally, given the true mean of the unit loss costs, the conditional mean is independent of the exposure and the conditional variance scales inversely proportionally to the exposure; this is also the common assumption in the B¨uhlmann–Straub credibility model [10]. For more discussion in a regression context, we also refer to Lindholm–Nazar [44].

– Property (3.5) may be a reasonable assumption if we have ex-ante exposures (known at contract inception) that do not impact the average loss costs per unit exposures.

Such a property may be violated if, e.g., high-risk profiles systematically sign shorter contracts or if there are seasonal patterns resulting in risk profiles that are not ceteris paribus over the entire insured period (e.g., avalanches are more likely during winter periods).

– If exposures are ex-post, i.e., only known at contract termination, then likely (3.5) is violated if there is an endogenous mechanism between claims and contract termination. A common reason for lapsing a contract is an accident (because, e.g., the insured object is replaced). Such accident-induced lapses directly act on the exposures and on the loss costs, thus, conditional mean independence is not a reasonable assumption in such settings because the exposure contains additional information about the loss.

– The conditional mean independence can be analyzed graphically by a calibration plot which additionally stratifies with exposure; calibration plots are discussed in Section 4.2.1, below. Alternatively, we can consider a two-dimensional kernel smoothed heatmap that considers

$$
( \pi , v ) \ \mapsto \ \mathbb { E } [ Y \mid \Pi = \pi , V = v ] .\tag{3.6}
$$

We can also fit a regression model to (3.6) to understand whether there is a systematic efect of the exposure V on Y, given Π.

Conclusion from Remark 3.8: We should generally doubt the validity of the conditional mean independence assumption (3.5) in general insurance pricing. Consequently, we will not require that the exposure is ex-ante, and our focus is on Q-calibration (3.2) throughout the remainder of this manuscript.

## What does the actuarial literature do?

• No exposures (V ≡ 1): Denuit et al. [14, 16] do not involve exposures, however, in their estimation procedure they use a total premium VΠ-view to restore calibration. Fissler et al. [25] do not involve exposures, however, they discuss estimation in their Section 5.2.2 relating to a Q-view. W¨uthrich [56, 57], Denuit–Trufin [19, 20] and Delong–W¨uthrich [12] do not involve exposures. W¨uthrich–Ziegel [61] do not involve exposures, though their example considers Q-calibration through the application of the isotonic regression with case weights.

• P Z-calibration: W¨uthrich–Merz [59, Formula (7.39)] and Denuit et al. [15].

• P-calibration: Delong et al. [11, 13] and Brauer et al. [6], these papers additionally assume that conditional mean independence holds, thus, P-calibration is equivalent to Qcalibration and Q|<sub>V</sub>-calibration, see Proposition 3.7.

• Q-calibration: Lindholm et al. [43] and Gatti [28, formula (4.3)].

From this list we see that most actuarial literature excludes variable exposures. Under exposures there are either the $P Z .$ -calibration view or the Q-calibration view used (sometimes under the additional conditional mean independence assumption). When it comes to the more applied actuarial literature, it is usually Q-calibration that is studied, though this is not particularly emphasized in the notation, $\mathrm { e . g . }$ , Goldburd et al. [34, Section 7.2.1] compare weighted unit loss cost averages against weighted unit premium averages in price buckets that have roughly the same exposure, this equi-exposure view corresponds to a discretized Q-calibration view, which will be denoted by $G _ { \Pi } ^ { \mathbb { Q } }$ , below.

## 3.2 Recalibrated unit premium

The previous section has introduced diferent versions of calibration, and our focus is on $\mathbb { Q } \mathrm { - }$ calibration (3.2) which has the equivalent definitions

$$
\Pi = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi \right] \qquad \iff \qquad \mathbb { E } \left[ P \mid \Pi \right] = \mathbb { E } \left[ Z \mid \Pi \right]
$$

That is, we stratify w.r.t. the unit premium Π to understand whether the resulting premium cohorts are on average self-financing for their claims.

Definition 3.9 (Recalibrated unit premium) The recalibrated unit premium under the exposureweighted Q-measure is defined by

$$
m _ { \mathbb { Q } } ( \Pi ) = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi \right] ,
$$

and under the original population measure P by

$$
m _ { \mathbb { P } } ( \Pi ) = \mathbb { E } \left[ Y \mid \Pi \right] .
$$

We have the following interesting result, a.s.,

$$
m _ { \mathbb { Q } } ( \Pi ) - m _ { \mathbb { P } } ( \Pi ) = \frac { \operatorname { C o v } \left( V , Y \mid \Pi \right) } { \mathbb { E } \left[ V \mid \Pi \right] } .\tag{3.7}
$$

This result indicates that Assumption 3.6 is suficient to have an identity $m _ { \mathbb { Q } } ( \Pi ) = m _ { \mathbb { P } } ( \Pi )$ , a.s., but necessary is only conditional uncorrelatedness between $V$ and Y, given Π.

Our main interest is in the exposure-weighted Q-calibration, and our goal is to verify

$$
\begin{array} { r } { m _ { \mathbb { Q } } ( \Pi ) = \Pi \qquad \mathrm { a . s . } } \end{array}\tag{3.8}
$$

Remark 3.10 • Since $\mathbb { Q } \sim \mathbb { P }$ are equivalent probability measures, the term a.s. (almost surely) is correct under any of the two population measures.

• The recalibrated unit premiums are calibrated, i.e.,

$$
\begin{array} { r } { m _ { \mathbb { Q } } ( \Pi ) = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid m _ { \mathbb { Q } } ( \Pi ) \right] \qquad \mathrm { ~ a n d ~ } \qquad m _ { \mathbb { P } } ( \Pi ) = \mathbb { E } \left[ Y \mid m _ { \mathbb { P } } ( \Pi ) \right] } \end{array}\tag{3.9}
$$

This directly follows from the tower property of conditional expectation, and it motivates the isotonic recalibration step discussed in Section 4.3.3, below.

## 3.3 Stylized example

This section constructs a stylized example that is P-calibrated but not Q-calibrated. This example will be used throughout the subsequent sections that introduce graphical tools and quantitative statistical methods to analyze calibration and discrimination.

The example is constructed in two parts: Part 1 shows the methodological construction, and Part 2 gives a numerical example.

Part 1 of the stylized example. We construct a stylized example that is P-calibrated but not Q-calibrated. This requires that the conditional mean independence (3.5) is violated, see Proposition 3.7, to obtain a non-zero conditional correlation in (3.7). Select a bounded measurable function $h ( \cdot )$ such that

$$
\begin{array} { l l l } { \operatorname { \mathbb { E } } \left[ Y \mid \Pi , V \right] } & { = } & { \Pi + r ( \Pi , V ) } \\ & { = } & { \Pi + h ( V ) - \operatorname { \mathbb { E } } [ h ( V ) \mid \Pi ] > 0 \qquad \mathrm { ~ a . s . } } \end{array}\tag{3.10}
$$

This implies P-calibration because we have centered residuals $\mathbb { E } [ r ( \Pi , V ) \mid \Pi ] = 0$ , that is,

$$
\begin{array} { l l l } { m _ { \mathbb { P } } ( \Pi ) } & { = } & { \mathbb { E } \left[ Y \mid \Pi \right] = \mathbb { E } \left[ \mathbb { E } \left[ Y \mid \Pi , V \right] \mid \Pi \right] } \\ & { = } & { \mathbb { E } \left[ \Pi + r ( \Pi , V ) \mid \Pi \right] = \Pi \qquad \mathrm { a . s . } } \end{array}\tag{3.11}
$$

Using (3.7), we compute the following covariance term

$$
\begin{array} { l l l } { \operatorname { C o v } \left( V , Y \mid \Pi \right) } & { = } & { \operatorname { C o v } \left( V , \mathbb { E } \left[ Y \mid \Pi , V \right] \mid \Pi \right) } \\ & { = } & { \operatorname { C o v } \left( V , \Pi + r ( \Pi , V ) \mid \Pi \right) } \\ & { = } & { \operatorname { C o v } \left( V , h ( V ) \mid \Pi \right) \qquad \mathrm { a . s . } } \end{array}
$$

Using (3.11), this implies, a.s.,

$$
m _ { \mathbb { Q } } ( \Pi ) - \Pi = m _ { \mathbb { Q } } ( \Pi ) - m _ { \mathbb { P } } ( \Pi ) = \frac { \mathrm { C o v } \left( V , Y \mid \Pi \right) } { \mathbb { E } \left[ V \mid \Pi \right] } = \frac { \mathrm { C o v } \left( V , h ( V ) \mid \Pi \right) } { \mathbb { E } \left[ V \mid \Pi \right] } .
$$

Consequently, if the last correlation term is diferent from zero with positive probability, we cannot have Q-calibration. This is the case if $\mathbb { E } [ V r ( \Pi , V ) \ | \ \Pi ] \neq 0$ . We provide an example. Assume that there exists $v _ { 0 } > 0$ and $\pi _ { 0 } > 0$ such that for $\mathbb { P } _ { \Pi }$ almost every π

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { V | _ { \Pi = \pi } \mathrm { ~ i s ~ n o n ~ d e g e n e r a t e ~ a n d ~ s u p p o r t e d ~ i n ~ } ( 0 , v _ { 0 } ] } & { \mathrm { ~ f o r ~ } \pi \leq \pi _ { 0 } , } \\ { V | _ { \Pi = \pi } \mathrm { ~ i s ~ n o n ~ d e g e n e r a t e ~ a n d ~ s u p p o r t e d ~ i n ~ } ( v _ { 0 } , \infty ) } & { \mathrm { ~ f o r ~ } \pi > \pi _ { 0 } . } \end{array} \right. } \end{array}\tag{3.12}
$$

Hence, the support of the unit premium is partitioned into two parts $\Omega _ { 0 } = \lbrace \Pi \leq \pi _ { 0 } \rbrace$ and $\Omega _ { 1 } = \lbrace \Pi > \pi _ { 0 } \rbrace$ . On $\Omega _ { 0 }$ the unit premiums provide non-degenerate exposure distributions on the interval $( 0 , v _ { 0 } ]$ and on $\Omega _ { 1 }$ on the interval $( v _ { 0 } , \infty )$ . Thus, low unit premiums have low exposures, and high unit premiums high exposures.

Finally, we assume that h is strictly increasing on $( 0 , v _ { 0 } ]$ and strictly decreasing on $( v _ { 0 } , \infty )$ Based on these assumptions we have

$$
\left\{ \begin{array} { l l } { \mathrm { C o v } \left( V , h ( V ) \mid \Pi = \pi \right) > 0 } & { \mathrm { ~ f o r ~ } \pi \leq \pi _ { 0 } , } \\ { \mathrm { C o v } \left( V , h ( V ) \mid \Pi = \pi \right) < 0 } & { \mathrm { ~ f o r ~ } \pi > \pi _ { 0 } . } \end{array} \right.
$$

Consequently,

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { m _ { \mathbb { Q } } ( \Pi ) > m _ { \mathbb { P } } ( \Pi ) = \Pi } & { \mathrm { ~ f o r ~ a . e . ~ } \Pi \leq \pi _ { 0 } , } \\ { m _ { \mathbb { Q } } ( \Pi ) < m _ { \mathbb { P } } ( \Pi ) = \Pi } & { \mathrm { ~ f o r ~ a . e . ~ } \Pi > \pi _ { 0 } . } \end{array} \right. } \end{array}
$$

Henceforth, the total losses are underestimated on small unit premiums and overestimated on large unit premiums. This closes Part 1 of the stylized example.

Part 2 of the stylized example. For the sections on graphical and statistical methods, below, we equip the above example with explicit functions and numerical values. We assume that the unit premium follows a scaled and translated beta distribution under P

$$
\Pi \sim G _ { \Pi } \qquad \mathrm { w i t h } \qquad { \frac { \Pi - 8 0 0 } { 4 0 0 } } \sim \mathrm { B e t a } ( \alpha = 4 , \beta = 4 ) .\tag{3.13}
$$

Thus, Π is supported in (800, 1200) and its density is symmetric around $\pi _ { 0 } : = 1 0 0 0$ . The exposure (3.12) is assumed to behave diferently below and above this critical point $\pi _ { 0 }$ , namely,

$$
V | _ { \Pi = \pi } \sim \left\{ \begin{array} { l l } { \mathrm { U n i f o r m } ( 0 . 5 , 1 ) } & { \mathrm { ~ f o r ~ } \pi \leq \pi _ { 0 } , } \\ { \mathrm { U n i f o r m } ( 1 , 1 . 5 ) } & { \mathrm { ~ f o r ~ } \pi > \pi _ { 0 } . } \end{array} \right.\tag{3.14}
$$

Next, we select the measurable function h that enters (3.10). We set on (0.5, 1.5)

$$
h ( v ) = - 3 0 0 0 ( v - 1 ) ^ { 2 } \ \geq \ - 7 5 0 .
$$

This function is increasing on $I _ { 1 } = ( 0 . 5 , 1 )$ and it is decreasing on $I _ { 2 } = ( 1 , 1 . 5 )$ . We set

$$
\mu ( \Pi , V ) = \operatorname { \mathbb { E } } \left[ Y \mid \Pi , V \right] = \Pi + h ( V ) - \operatorname { \mathbb { E } } [ h ( V ) \mid \Pi ] > 0 \qquad \mathrm { ~ a . s . }
$$

This example provides P-calibration $m _ { \mathbb { P } } ( \Pi ) = \Pi , { \mathrm { a . s . } }$ , see (3.11), and we have miscalibration under the Q-measure

$$
m _ { \mathbb { Q } } ( \Pi ) - \Pi = \frac { \mathrm { C o v } \left( V , Y \mid \Pi \right) } { \mathbb { E } \left[ V \mid \Pi \right] } = \frac { \mathrm { C o v } \left( V , h ( V ) \mid \Pi \right) } { \mathbb { E } \left[ V \mid \Pi \right] } = \left\{ \begin{array} { l l } { 1 0 0 0 / 2 4 } & { \mathrm { f o r ~ } \Pi \le \pi _ { 0 } , } \\ { - 1 0 0 0 / 4 0 } & { \mathrm { f o r ~ } \Pi > \pi _ { 0 } . } \end{array} \right.\tag{3.15}
$$

Lastly, we need to model the unit loss costs Y. We assume the conditional distribution

$$
Y | _ { \Pi , V } \sim \mathrm { G a m m a } \left( \gamma , \gamma / \mu ( \Pi , V ) \right) ,\tag{3.16}
$$

with shape parameter $\gamma > 0 .$ . This conditional distribution has expected value $\mu ( \Pi , V )$ , variance $\mu ( \Pi , V ) ^ { 2 } / \gamma$ and coeficient of variation of $1 / \sqrt \gamma ;$ due to (2.7), this distribution is identical under both population measures $\mathbb { P }$ and $\mathbb { Q } .$

This model specification is going to be used in the examples in the next sections.

## 4 Graphical tools to assess calibration

Our first goal is to assess Q-calibration (3.2) in the stylized example introduced in Section 3.3 using diferent graphical tools. Recall that this example is P-calibrated (which is not of much interest in actuarial pricing), but it is not Q-calibrated (our main interest), see (3.15). Our goal is to introduce graphical tools that allow us to identify this Q-miscalibration.

Our first step in Section 4.1 is to present the distribution of the unit premium Π both under the P-measure and the Q-measure. As will be seen below, the latter distribution under Q is important in an empirical set-up.

The remainder of this section is then divided into two parts:

Part 1 – Population Version: In Section 4.2, we assume that the above (true) data generating model is known. This allows us to study the calibration question under the ground truth. Of course, this is unrealistic in practice, but it helps us to shape ideas and to introduce a clean notation.

Part 2 – Sample Version: In Section 4.3, we work under an unknown population model specification, and we try to answer the calibration question empirically from an observed sample. For this second step, we assume to have an i.i.d. test sample $\mathcal { T } = ( Y _ { i } , V _ { i } , \Pi _ { i } ) _ { i = 1 } ^ { n }$ following the same law as $( Y , V , \Pi )$ . Test sample means that we perform a proper out-of-sample validation that does not consider the learning data that has been used to derive the unit premium rule Π – in this sense, all considerations are understood conditionally on the learning sample, which is kept fixed (a mathematically fully consistent notation would require to have a conditional notation, given the learning sample, for notational convenience we do not do that).

## 4.1 Unit premium distributions under P and $\mathbb { Q }$

The unit premium $\mathrm { I } \sim G _ { \mathrm { I I } }$ has a scaled and translated beta distribution under the population measure P, see (3.13). Figure 1 (lhs) shows this distribution $\Pi \sim G _ { \Pi }$ . Additionally, we illustrate the deciles of $G _ { \Pi }$ by the dotted lines. These are obtained by selecting $K = 1 0$ and setting

$$
G _ { \Pi } ^ { - 1 } ( k / K ) \qquad \mathrm { ~ f o r ~ } k \in \{ 0 , \ldots , K \} ,
$$

where $G _ { \Pi } ^ { - 1 }$ is the generalized left-continuous inverse of $G _ { \Pi }$ given by

$$
G _ { \Pi } ^ { - 1 } ( u ) = \operatorname* { i n f } \left\{ y ; G _ { \Pi } ( y ) \ge u \right\} \qquad { \mathrm { f o r ~ } } u \in [ 0 , 1 ] .
$$

Consequently, each interval

$$
I _ { k } = \left( G _ { \Pi } ^ { - 1 } \left( { \frac { k - 1 } { K } } \right) , G _ { \Pi } ^ { - 1 } \left( { \frac { k } { K } } \right) \right]
$$

has an equal probability $\mathbb { P } ( \Pi \in I _ { k } ) = 1 / K$ for $k \in \{ 1 , \ldots , K \}$ ; note that $G _ { \Pi }$ is absolutely continuous in our case.

For parameter estimation, we prefer intervals that contain roughly equal aggregated exposures. Intuitively, this means that in all intervals there is roughly an equal amount of information available for parameter estimation (this statement would need additional assumptions to be

![](images/65b2b900c5fc1002d1a57db196f35498c6b451532bd2093fbc4bb681a0447ae8.jpg)

![](images/a9fd85d4a1dc19866d7044550bf3c0f06ed7f612055837163293a10b6bf25e33.jpg)  
Figure 1: (lhs) Unit premium distribution $\Pi \sim G _ { \Pi }$ under population distribution $\mathbb { P } ,$ and (rhs) comparison of $G _ { \pi }$ and $G _ { \pi } ^ { \mathbb { Q } }$

made precise). Consequently, we do not want intervals $I _ { k }$ with equal probabilities for Π, but rather with equi-exposures of $V$ in all intervals. This motivates the unit premium distribution under the $\mathbb { Q } \mathrm { - }$ measure

$$
G _ { \Pi } ^ { \mathbb { Q } } ( \pi ) = \mathbb { Q } \left( \Pi \leq \pi \right) = \frac { 1 } { \mathbb { E } [ V ] } \mathbb { E } \left[ V \mathbf { 1 } _ { \{ \Pi \leq \pi \} } \right] \qquad \mathrm { ~ f o r ~ } \pi \in \mathbb { R } ,\tag{4.1}
$$

with generalized left-continuous inverse

$$
( G _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( u ) = \operatorname* { i n f } \left\{ y ; \ G _ { \Pi } ^ { \mathbb { Q } } ( y ) \geq u \right\} \qquad { \mathrm { ~ f o r ~ } } u \in [ 0 , 1 ] .
$$

The new intervals for $k \in \{ 1 , \ldots , K \}$ under the exposure-weighted measure $\mathbb { Q }$ are given by

$$
I _ { k } ^ { \mathbb { Q } } = \left( ( G _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } \left( { \frac { k - 1 } { K } } \right) , ( G _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } \left( { \frac { k } { K } } \right) \right] .
$$

Under this $\mathbb { Q } \mathrm { . }$ -measure set-up, we can compute the average exposure in each interval $I _ { k } ^ { \mathbb { Q } }$

$$
1 / K \approx \mathbb { Q } \left( \Pi \in I _ { k } ^ { \mathbb { Q } } \right) = \frac { 1 } { \mathbb { E } \left[ V \right] } \mathbb { E } \left[ V \mathbf { 1 } _ { \{ \Pi \in I _ { k } ^ { \mathbb { Q } } \} } \right] .\tag{4.2}
$$

The approximation is exact for continuous distributions. Thus, we need to work under the exposure-weighted Q-measure framework for ensuring that all intervals have the same average observed exposure under equi-spaced quantile levels. Figure 1 (rhs) shows the diferences of the distributions $G _ { \Pi }$ and $G _ { \Pi } ^ { \mathbb { Q } }$ of Π under P and Q.

We generally work under the exposure-weighted population measure $\mathbb { Q } ,$ and the subsequent derivations generally use $G _ { \Pi } ^ { \mathbb { Q } }$ to ensure equi-exposures by quantile binning, see (4.2).

## 4.2 Graphical tools: Population Version

In this section, we assume that the true data generating model introduced in Section 3.3 is known. This population version will help us to shape ideas and to properly define all the relevant objects. In Section 4.3, we turn to the real-world situation of an unknown population model. This requires approximation by its empirical counterpart using the test sample T . This latter case is called the sample version.

The first object of core interest is the Q-calibration property (3.2). We present graphical tools to analyze it. We discuss the calibration plot and the $l i f t$ chart which present the identical information, but in a slightly diferent structure.

## 4.2.1 Calibration plot

The calibration plot has many diferent names, e.g., in Gneiting–Resin [33] it is called T-reliability diagram (in our case T is the mean functional), in Pohle [52] and W¨uthrich–Merz [59] it is called auto-calibration plot, and a particular case of a sample version of the calibration plot is the actual-vs-expected plot; see Goldburd et al. [34]; we come back to the actual-vs-expected (AvE) plot in Section 4.3.1, below.

The calibration plot considers the graph<sup>1</sup>

$$
\pi \mapsto m _ { \mathbb { Q } } ( \pi ) = \mathbb { E } _ { \mathbb { Q } } \left[ Y | \Pi = \pi \right] ,\tag{4.3}
$$

for π in the convex hull of the premium range attained by Π. If this plot shows a diagonal line $m _ { \mathbb { Q } } ( \pi ) = \pi$ , we have perfectly Q-calibrated unit premiums Π for Y, otherwise not.

![](images/1c70bcc6bda58c1019989d467684d3787b4f345fa5d7e7ba30a7cc30c05f83d5.jpg)  
Figure 2: Calibration plot (4.3) under the exposure-weighted Q-measure; the orange diagonal line corresponds to perfect Q-calibration.

Figure 2 shows the results of the recalibration step (4.3) under $\mathbb { Q } ,$ which in our modeling set-up results in the exact formula (3.15). The black line shows the Q-recalibrated premiums $m _ { \mathbb { Q } } ( \pi )$ in the unit premium range (800, 1200), and the orange line corresponds to the diagonal. We observe that there is Q-miscalibration: for $\pi \leq \pi _ { 0 } = 1 0 0 0 .$ , the expected unit loss costs are underestimated, for $\pi > \pi _ { 0 }$ , they are overestimated under $\mathbb { Q } .$ Thus, in this example we do not have Q-calibration (3.2) because, a.s.,

$$
\begin{array} { r } { \begin{array} { r l } { \mathbb { E } \left[ Z \mid \Pi \right] > \mathbb { E } \left[ P \mid \Pi \right] \quad } & { \mathrm { f o r } \Pi \leq \pi _ { 0 } , } \\ { \mathbb { E } \left[ Z \mid \Pi \right] < \mathbb { E } \left[ P \mid \Pi \right] \quad } & { \mathrm { f o r } \Pi > \pi _ { 0 } . } \end{array} } \end{array}
$$

This leads to a systematic cross-subsidy from high-unit premium policies to low-unit premium policies. Note that $m _ { \mathbb { Q } } ( \Pi )$ is Q-calibrated, see Remark 3.10.

Remark 4.1 (Risk ranking) Figure 2 shows a situation where Q-calibration of Π for Y fails to hold. In fact, this failure is not only on the level of the fitted unit premiums Π, but it also provides a wrong risk ranking: for Π in a small neighborhood around $\pi _ { 0 } = 1 0 0 0$ , the risk ranking from the x-axis and the y-axis difer, e.g., $m _ { \mathbb { Q } } ( \pi _ { 0 } ) > m _ { \mathbb { Q } } ( \pi _ { 0 } + \varepsilon )$ for small $\varepsilon > 0$ . Risk rankings will be assessed by Gini scores in Section 7, below.

## 4.2.2 Lift chart

The lift chart shows the same statistics as the calibration plot, but it uses a diferent scale on the x-axis that is based on quantile levels. Sample versions of lift charts have been considered, e.g., in Goldburd et al. [34], we come back to this in Section 4.3.2, below.

At the population level, the lift chart considers the two functions

$$
u \in ( 0 , 1 ) \mapsto \left\{ \begin{array} { l } { m _ { \mathbb { Q } } \left( ( G _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( u ) \right) = \mathbb { E } _ { \mathbb { Q } } \left[ Y \Big | \Pi = ( G _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( u ) \right] , } \\ { ( G _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( u ) . } \end{array} \right.\tag{4.4}
$$

The first function gives the Q-recalibrated unit premium, and the second one the unit premium at the quantile levels $u \in ( 0 , 1 )$ . If these two functions are identical, we have Q-calibration (3.2). Figure (3) provides the lift chart at the population level, the black curve gives the first function in (4.4) and the orange line corresponds to the second function in (4.4) highlighting the calibration line. This lift chart is identical to the calibration plot of Figure 2, we only changed the x-axis from the observation scale π to the quantile level scale $u = G _ { \Pi } ^ { \mathbb { Q } } ( \pi )$

The term $l i f t$ refers to the fact that the wider the scale on the y-axis, the bigger the lift from the lowest to the biggest unit premiums. Thus, the lift corresponds to discrimination (studied in Section 5, below) regardless whether this lift is justified by the calibration consideration of Π for Y or not. This lift is then interpreted as the premium discrimination between lowest and highest risk profiles (assigned by Π). If we use the global mean premium $\Pi = \mathbb { E } _ { \mathbb { Q } } [ Y ]$ (egalitarian price), there is calibration but there is no discrimination because every policyholder is charged the identical unit premium and the lift is zero.

Intuitively, if there are two premium schemes $( Y , V , \Pi ^ { a } )$ and $( Y , V , \Pi ^ { b } )$ , then the better calibrated one for Y , encloses a smaller “area between the curves” of the black and the orange graphs in Figures 2 and 3, respectively. There are some dificulties in assessing this calibration accuracy for two diferent premium schemes:

![](images/fd9861731e6be7db72aac5133259211b3811e656ceebe66146f81272c3a0d551.jpg)  
Figure 3: Lift chart considering the two graphs (4.4).

• The two Figures 2 and 3 consider diferent scales on the x-axis, and one premium scheme may enclose a smaller area on one scale and a bigger one on the other scale. That is, there is no universality or a canonical scale for measuring the calibration error. One may though argue that in our case preference should be given to the lift chart because its scale on the x-axis does not depend on the selected premium scheme.

• The graphs in Figures 2 and 3 require that we can compute the recalibrated premium $m _ { \mathbb { Q } } ( \Pi )$ . In most applications, this is not the case and this recalibration step needs to be estimated from observations. Therefore we need to turn from this clean mathematical formulation to empirical sample versions. This is the topic of the next section.

• The “area between the curves” on its own is not a qualitative criteria for prediction accuracy, it only evaluates calibration, but not discrimination. $\mathrm { E . g . }$ , we can have two pricing principles $\Pi ^ { a } = Y + c$ for a constant $c > 0$ and $\Pi ^ { b } = \mathbb { E } _ { \mathbb { Q } } [ Y ]$ . The latter is Q-calibrated, the former not, but the reader will certainly agree that the former one provides a better predictor for Y , if c is small because it is the true claim Y slightly shifted by c, i.e., it has the correct resolution but it is not fully calibrated; we come back to this discussion in Section 6, below.

## 4.3 Graphical tools: Sample Version

The previous graphical results were based on the knowledge of the (true) population model of (Y, V, Π). In most applied situations, this population model is unknown and we need to resolve the calibration question from a sample version. This naturally involves noise, also called irreducible risk. Assume we have an i.i.d. test sample $\mathcal { T } = ( Y _ { i } , V _ { i } , \Pi _ { i } ) _ { i = 1 } ^ { n }$ following the same law as (Y, V, Π). We call T a test sample because it should be independent of the learning sample that has been used to find the unit premium rule Π.

The calibration plot considers the graph (4.3). This has been illustrated in Figure (2) under the knowledge of the population model. In absence of this knowledge we approximate it empirically by using the test sample $\tau .$ . The most crude version is to replace the conditional means $m _ { \mathbb { Q } } ( \pi ) =$ ${ \mathbb E } _ { \mathbb Q } \left[ { \cal Y } | \Pi = \pi \right]$ empirically by the observations $Y _ { i } | \Pi _ { i } = \pi$ . However, irreducible risk (noise) on the response scale $Y$ makes the resulting plot not very expressive.

![](images/7bb5f2d6427f022a793727c9cfdc2e5144b3202a51fd3c85edbfc608b1a33de3.jpg)  
Figure 4: Loss costs $Y _ { i }$ plotted against unit premiums $\Pi _ { i }$ on test sample $\tau$

Figure 4 shows the loss costs $Y _ { i }$ plotted against the unit premiums $\Pi _ { i }$ on the test sample $\mathcal { T } = ( Y _ { i } , V _ { i } , \Pi _ { i } ) _ { i = 1 } ^ { n }$ . Basically, we replace the recalibrated unit premiums $m _ { \mathbb { Q } } ( \Pi _ { i } )$ – black dots in Figure 2 – by the observed responses $Y _ { i } \mathrm { ~ - ~ } \mathrm { b l u e }$ dots in Figure 4. We observe that this latter sample plot is dominated by the irreducible risk in $Y _ { i }$ , given $\Pi _ { i } ,$ and it is hard to judge whether we have calibration of Π for $Y ,$ or not.

## 4.3.1 Actual-vs-expected plot – quantile binning

In order to reduce the irreducible risk in Figure 4, we need to aggregate to benefit from the law of large numbers. This can be achieved by quantile binning on the unit premium scale. For this, we estimate the exposure-weighted unit premium $G _ { \Pi } ^ { \mathbb { Q } }$ distribution by

$$
{ \widehat G } _ { \Pi } ^ { \mathbb { Q } } ( \pi ) = { \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } } \sum _ { i = 1 } ^ { n } V _ { i } \mathbf { 1 } _ { \{ \Pi _ { i } \leq \pi \} } .\tag{4.5}
$$

This approximates $G _ { \Pi } ^ { \mathbb { Q } } ( \pi )$ pointwise in π, a.s., as the sample size $n  \infty$ (by the law of large numbers). In contrast to the classical empirical distribution, we select the step sizes in (4.5) by the exposures $( V _ { i } ) _ { i = 1 } ^ { n }$ . This is the sample version of the unit premium distribution under $\mathbb { Q }$ For quantile binning, select the number of bins $K \in \mathbb N$ . This $_ \mathrm { y }$ ields the equi-distributed quantile binning of the test sample under the empirical exposure-weighted unit premium distribution

$$
\begin{array} { r } { \widehat { \mathcal { T } } _ { k } ^ { \mathbb { Q } } = \left\{ i \in \{ 1 , \dots , n \} \Big | ( \widehat { G } _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( ( k - 1 ) / K ) < \Pi _ { i } \leq ( \widehat { G } _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( k / K ) \right\} , } \end{array}\tag{4.6}
$$

for $k = 1 , \ldots , K$ . As indicated in (4.2), every bin will contain approximately the same aggregated exposure

$$
\frac { V _ { k } ^ { + } } { \sum _ { i = 1 } ^ { n } V _ { i } } : = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \mathbb { Q } } } V _ { i } \approx 1 / K .
$$

This is verified in Figure 5, there is a total exposure of 100, 000, and applying decile binning, K = 10, each of the folds has a total exposure of approximately 10, 000.

![](images/2e1a2f762e29ec53330703111ed4f417c87a5d6f9062046ae322d77a31f7a78e.jpg)  
Figure 5: Total exposures in each bin $\widehat { \cal T } _ { k } ^ { \mathrm { Q } }$ for decile binning K = 10 under $\widehat { G } _ { \Pi } ^ { \mathbb { Q } }$

The idea now is to build the weighted bin averages. For the observed loss costs, we have weighted sample mean in each bin

$$
\overline { { Y } } _ { k } = \frac { 1 } { V _ { k } ^ { + } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \mathbb { Q } } } V _ { i } Y _ { i } = \frac { 1 } { V _ { k } ^ { + } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \mathbb { Q } } } Z _ { i } .\tag{4.7}
$$

Equivalently, we compute the exposure-weighted average unit premium in each bin

$$
\overline { { \Pi } } _ { k } = \frac { 1 } { V _ { k } ^ { + } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \mathbb { Q } } } V _ { i } \Pi _ { i } = \frac { 1 } { V _ { k } ^ { + } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \mathbb { Q } } } P _ { i } .\tag{4.8}
$$

This weighted version relates to the Q-measure introduced in Section 2.2, by considering the corresponding exposure-weighted quantities.

The actual-vs-expected (AvE) plot considers the following sample version of the calibration plot

$$
\begin{array} { r } { \left( \Pi _ { k } , \overline { { Y } } _ { k } \right) \quad a n d \quad \left( \overline { { \Pi } } _ { k } , \overline { { \Pi } } _ { k } \right) \qquad \mathrm { ~ f o r ~ } k = 1 , \ldots , K . } \end{array}\tag{4.9}
$$

The AvE plot is also called actual-vs-predicted plot.

Figure 6 shows the sample results for decile binning K = 10 and percentile binning K = 100. The blue dots in Figure 6 show the (recalibrated) pairs $( \overline { { \Pi } } _ { k } , \overline { { Y } } _ { k } ) _ { k = 1 } ^ { K }$ , and the orange lines in these figures correspond to the diagonal calibration line reflecting the pairs $( \overline { { \Pi } } _ { k } , \overline { { \Pi } } _ { k } ) _ { k = 1 } ^ { K }$ . The blue dots clearly difer from the orange diagonal line in Figure 6. This questions Q-calibration. Naturally, this is not a statistical test or a proof of miscalibration because it only corresponds to a graphical inspection that involves noise (irreducible risk). The noise is bigger for K = 100 and smaller for K = 10 due to the law of large numbers. A proper statistical test would involve a standard deviation estimate that quantifies the magnitude of this noise. However, from Figure

![](images/ccb3fcd8997726c2d5cf392025be3abf89d3bb7521606008dc33e6fb96bff5c2.jpg)

![](images/037f1d177ba3fd43589c78e7f44cc63dccdce014e26fd142faa3bd716383bf7c.jpg)  
Figure 6: AvE plot using decile binning K = 10 (lhs) and percentile binning $K = 1 0 0 \ \mathrm { ( r h s ) }$

6 we see that the blue dots do not seem to randomly fluctuate around the orange diagonal, but there is a systematic pattern. This is a clear indication that Q-calibration is violated.

The black lines (rectangles) in Figure 6 show the quantiles $( \widehat { G } _ { \Pi } ^ { \mathbb { Q } } ) ^ { - 1 } ( k / K ) , k \in \{ 0 , \ldots , K \}$ , these are impacted by the unit premium distribution (which is a beta distribution under P) and by the exposure distribution. We see that there are more decile/percentile bins above the value π<sub>0</sub> because the exposure tends to be bigger for bigger unit premiums due to model assumption (3.14).

## 4.3.2 (Empirical) lift chart – quantile binning

The sample version of the lift chart is then straightforward from (4.9). Instead of plotting the quantile levels on the x-axis, we simply use the bin labels $k \in \{ 1 , \ldots , K \}$ instead.

The (sample version of the) lift chart considers the two graphs

$$
\left( k , { \overline { { Y } } } _ { k } \right) \quad { \mathrm { ~ a n d ~ } } \quad \left( k , { \overline { { \Pi } } } _ { k } \right) \qquad { \mathrm { ~ f o r ~ } } k = 1 , \ldots , K .\tag{4.10}
$$

These two graphs contain precisely the same information as (4.9), but we represent the x-axis on a diferent scale (premium scale vs. quantile level scale).

Figure 7 shows the lift chart (4.10) from decile binning and percentile binning using the bins (4.6). The conclusions are essentially the same as from the AvE plot in Figure 6, though they may look a bit more obvious in the lift chart (underestimation for small unit premiums and overestimation for large unit premiums).

To turn the AvE plot and the lift chart into statistical methods, we would need to estimate the variance of (the uncertainty in) the empirical means $\overline { { Y } } _ { k }$ as well as the impact of the quantile binning bounds. This would then allow us to turn the lift chart of Figure 7 into a $\chi ^ { 2 } \cdot$ -test for the null hypothesis that we have a Q-calibrated model; such approaches have been considered, for example, in Gatti [28], and the binary case is known as the Hosmer–Lemeshow test [37]; see also Henzi et al. [36].

![](images/8215b4df0966b28d0a95e43ad96c423c8ecf195164d20601e98f71f7e5808685.jpg)

![](images/0a0fdc51bb4c231cb6cf694ee70db7c20756cfb9bce9e3a49283259e7ee7b5a3.jpg)  
Figure 7: Sample versions of the lift chart (4.10) using decile binning $K = 1 0$ (lhs) and percentile binning $K = 1 0 0$ (rhs).

## 4.3.3 Actual-vs-expected plot – statistical smoothing methods

Quantile binning replaces the individual observations by a finite number of exposure-weighted bin averages thereby benefiting from the law of large numbers. We could also use more statistically guided methods that try to interpolate observations by regression functions. The two natural candidates are local regression of Loader [47] and isotonic regression of Ayer et al. [1], Brunk et al. [9], Miles [49], Barlow et al. [4], Barlow–Brunk [5], Kruskal [41]. The pool adjacent violators (PAV) algorithm gives a fast implementation solving the isotonic regression numerically; see Leeuw et al. [42]. The local regression is a flexible smoothing method, its disadvantage is that it heavily relies on a good hyper-parameter selection. The isotonic regression is in some sense more crude, it essentially relies on the assumption of having a correct risk ranking in Π. Both methods can be problematic in applications, but they are still the best tools that are currently available.

Exposure-weighted Local Regression. This outline follows W¨uthrich et al. [60, Section 4.2.1]. For a local regression, one selects a bandwidth $\delta ( \Pi ) > 0$ that may depend on the unit premium rule Π. This gives the smoothing window (interval)

$$
\Delta ( \Pi ) = \Big ( \Pi - \delta ( \Pi ) , \Pi + \delta ( \Pi ) \Big ) .
$$

For a local regression around Π only the instances $( Y _ { i } , V _ { i } , \Pi _ { i } )$ in this smoothing window $\Pi _ { i } \in$ $\Delta ( \Pi )$ are considered. Typically, one chooses $\delta ( \Pi )$ such that the smoothing window $\Delta ( \Pi )$ contains 10% or 20% of the available sample. Then, one select a weighting function $w _ { \Pi } : \Delta ( \Pi ) $ $\mathbb { R } _ { + }$ . This weighting function acts as a kernel that weighs observations closer Π more than those that are at the boundary of the smoothing window. A popular choice is a (scaled) tricube weighting function. Finally, one select a class of splines, e.g., quadratic polynomials

$$
x \ \mapsto \ \mu _ { \pmb \vartheta } ( x ; \Pi ) = \vartheta _ { 0 } + \vartheta _ { 1 } ( x - \Pi ) + \vartheta _ { 2 } ( x - \Pi ) ^ { 2 } ,\tag{4.11}
$$

with regression parameter $\pmb { \vartheta } = ( \vartheta _ { 0 } , \vartheta _ { 1 } , \vartheta _ { 2 } ) ^ { \top }$ . This motivates the local regression problem

$$
 { \widehat { \boldsymbol { \vartheta } } } ^ { \mathrm { I I } } = \arg \operatorname* { m i n } _ { \pmb { \vartheta } \in \mathbb { R } ^ { 3 } } \sum _ { i = 1 } ^ { n } V _ { i } \mathbf { 1 } _ { \left\{ \Pi _ { i } \in \Delta ( \Pi ) \right\} } w _ { \mathrm { I I } } \left( \Pi _ { i } \right) \left( Y _ { i } - \mu _ { \pmb { \vartheta } } ( \Pi _ { i } ; \Pi ) \right) ^ { 2 } .
$$

The fitted local regression value in Π is then obtained by setting

$$
\widehat { \mu } ^ { \mathrm { l o c } } ( \Pi ) = \mu _ { \widehat { \vartheta } ^ { \mathrm { I I } } } ( \Pi ; \Pi ) = \widehat { \vartheta } _ { 0 } ^ { \Pi } .
$$

This is the local regression method as implemented in the R package locfit of Loader [47]. The hyper-parameters involved are the bandwidth δ(Π) (usually a nearest neighbor fraction around Π), the weighting function $w _ { \Pi }$ (usually tricube) and the splines, usually the step function (local constant fit), linear, quadratic or cubic polynomials.

The bandwidth controls the bias-variance trade-of. A small δ(Π) follows local variation closely but may produce noisy curves, whereas a large δ(Π) considers more information which results in a better smoothing, which however is more prone to local miscalibration.

![](images/0c82d613e79ef7258af3149c3096b0d2837eb34a0c3518562cbeecd909d47069.jpg)

![](images/6de002f4875085e0a8266ce50e090218af3aa673944117ac74b5d0e40a892786.jpg)  
Figure 8: AvE plot using local regression smoothing with (lhs) quadratic polynomials and (rhs) step function spline, both plots use a nearest neighbor fraction of 10%.

Figure 8 shows the local regression smoothed AvE plots. We use a nearest neighbor fraction of 10% for the smoothing windows $\Delta ( \Pi )$ , the tricube kernel for $w _ { \Pi } .$ , and the left-hand side uses the quadratic regression splines (4.11) and the right-hand side step functions. Note that the nearest neighbor fraction does not account for the exposures, but it is selected under the population measure P – this is implied by the available software package and naturally we would prefer to have a Q-measure option. The step function spline can be seen as a rolling window approach of the quantile binning, though, still using the tricube weighting in this rolling window.

From Figure 8 we observe two wiggly curves that essentially show the right miscalibration picture. However, in a more complicated miscalibration situation it is often dificult to say whether the local regression is wiggly because of the noisy data (bias-variance trade-of) or a miscalibration. Therefore, local regression should be viewed as an indicative rather than definitive diagnostics.

Exposure-weighted Isotonic Regression. A diferent approach assumes that the original premium rule Π gives the correct risk ranking, so that $m _ { \mathbb { Q } } ( \Pi )$ is non-decreasing in Π (or comonotonic in the sense of random variables). Under this ranking assumption, define the isotonic estimator by

$$
\widehat { \pmb { m } } _ { \mathbb { Q } } ^ { \mathrm { i s o } } = \underset { m \in \mathbb { R } ^ { n } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { n } V _ { i } \left( Y _ { i } - m _ { i } \right) ^ { 2 } \qquad \mathrm { s u b j e c t ~ t o : } \ \Pi _ { i } \leq \Pi _ { j } \ \mathrm { i m p l i e s } \ m _ { i } \leq m _ { j } .\tag{4.12}
$$

Thus, the premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ provide a ranking on the instances $i \in \{ 1 , \ldots , n \}$ , and this ranking is preserved by the isotonic regression solution $\widehat { \pmb { m } } _ { \mathbb { Q } } ^ { \mathrm { i s o } } \in \mathbb { R } ^ { n }$ . In between these values $\widehat { \pmb { m } } _ { \mathbb { Q } } ^ { \mathrm { i s o } } \in \mathbb { R } ^ { n }$ we select a step function interpolation.

The step function interpolation implies that the isotonic regression results in a binning with constant recalibrated unit premiums in the bins. In contrast to quantile binning, the amount of data in each bin is not a selected hyper-parameter, but the algorithm (4.12) decides on the optimal bin sizes such that isotonicity w.r.t. the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ is preserved. Since on each bin, $\widehat { \pmb { m } } _ { \mathbb { Q } } ^ { \mathrm { i s o } }$ corresponds to the exposure-weighted sample mean, this automatically implies that we have sample Q-calibration on all bins. Thus, this method performs (a discretized version of) the recalibration step; this was used in an essential way in W¨uthrich–Ziegel [61].

The main advantages of isotonic regression over local regression are that it does not involve any hyper-parameter selection and it results in a calibrated solution that is optimally binned w.r.t. (4.12). The disadvantages are that it relies on a correct risk ranking in $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ , this is not satisfied in our example, and under noisy data, the number of bins can be very small resulting in a very crude step function. Moreover, isotonic regression tends to overfit at the boundary observations of the unit premium. For more discussion, we refer to W¨uthrich et al. [60, Section 4.2.2]. Generally, we recommend isotonic regression for calibration inspection, but the resulting recalibrated regression function is often too crude for insurance pricing because a low signal-to noise ratio typically leads to a crude binning.

Figure 9 shows that isotonic regression solution, on the left-hand side we use the risk ranking of the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and on the right-hand side the (correct) risk ranking of the recalibrated premium $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ . The former does not provide the correct risk ranking, see Remark 4.1, the latter does. Figure 9 is also called CORP (consistent, optimally binned, reproducible and PAV) diagram in Dimitriadis et al. [21] and Gneiting–Resin [33].

We have the following observations:

• Figure 9 (lhs) based on risk ranking $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ detects the calibration issue in Π.

• Figure 9 (rhs) based on risk ranking $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ correctly fluctuates around the orange diagonal, in fact, under infinite sample sizes the isotonic regression should coincide with

AvE plot: Isotonic regression recalibration

![](images/e83e991d6d86d5c0f1f2d3876ac8fcc980d3f90a9ed933dc8e3350ddedf5afb1.jpg)

![](images/600330931fb800ceb03aa5bbccf6c07b46e13bd79765dda01eaeb7ec385dc0af.jpg)  
Figure 9: AvE plot using isotonic regression smoothing with (lhs) risk ranking Π and (rhs) risk ranking m<sub>Q</sub>(Π).

this orange diagonal, i.e., $\widehat { \pmb { m } } _ { \mathbb { Q } } ^ { \mathrm { i s o } }$ should match $m _ { \mathbb { Q } } ( \Pi )$ if the isotonic regression is performed on the correct risk ranking. The diference in Figure 9 (rhs) is a consequence of the noise in the responses $Y$

• The wrong risk ranking in $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ leads to a flat price around the center in Figure 9 (lhs).

• There is some overfitting in the tails, especially, in the lower tail. If the lowest unit premium has the smallest claim, the isotonic regression just reports this claim, i.e., no averaging takes place. For this reason, in real applications, smallest bins should be merged as well as largest bins.

As seen from Figure 9, the isotonic regression gives a natural binning that is optimal according to (4.12). Instead of plotting the AvE plot as in Figure 9, we could also label the bins in increasing order that results in a lift chart (with isotonically optimal bins w.r.t. the initial premium Π). We provide such a plot in the real-data example below, see Figure 20.

## 4.3.4 Double-lift charts

Our main focus in the previous section was on Q-calibration which is one part of a good predictive model. The other part is discrimination (resolution) which is going to be studied in the next section. We have briefly touched upon discrimination in the lift charts of Figures 3 and $7$ to explain the terminology lift. Naturally, if we have two pricing schemes $\Pi ^ { a }$ and $\Pi ^ { b }$ , we would like to compare them w.r.t. the lift and resolution they provide. A disadvantage of the sample version of the lift charts is that the quantile binning is done w.r.t. the pricing schemes $\Pi ^ { a }$ and $\Pi ^ { b }$ , respectively. Consequently, the resulting bins will generally not contain exactly the identical policies $i \in \{ 1 , \ldots , n \}$ , e.g., policy $i = 1$ may be in the second smallest bin for $\Pi ^ { a }$ and in the third smallest one for $\Pi ^ { b }$ . This is an inherent dificulty of binning approaches, and it makes a direct comparison dificult.

The double lift chart presented in Goldburd et al. [34] solves the binning problem by simultaneously considering both premium schemes $\Pi ^ { a }$ and $\Pi ^ { b }$ for the binning. Assume that all considered unit premiums are strictly positive, a.s. We analyze the ratio

$$
\kappa = { \frac { \Pi ^ { b } } { \Pi ^ { a } } } .
$$

Small values of $\kappa$ identify policies for which $\Pi ^ { b }$ is low relative to $\Pi ^ { a }$ , whereas large values of κ identify the reverse. We can then study the distribution $G _ { \kappa } ^ { \mathbb { Q } }$ of this ratio κ under the exposureweighted measure $\mathbb { Q }$ for the quantile binning. Its sample version is given by

$$
{ \widehat G } _ { \kappa } ^ { \mathbb Q } ( t ) = { \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } } \sum _ { i = 1 } ^ { n } V _ { i } \mathbf { 1 } _ { \{ \kappa _ { i } \leq t \} } ,
$$

and this yields the quantile binning for $k = 1 , \ldots , K$

$$
\widehat { \cal Z } _ { k } ^ { \kappa } = \Big \{ i \in \{ 1 , \ldots , n \} \Big | ( \widehat { G } _ { \kappa } ^ { \mathbb { Q } } ) ^ { - 1 } ( ( k - 1 ) / K ) < \kappa _ { i } \leq ( \widehat { G } _ { \kappa } ^ { \mathbb { Q } } ) ^ { - 1 } ( k / K ) \Big \} .
$$

The remaining parts are then completely analogous to the lift chart. We define the exposureweighted averages

$$
\begin{array} { l } { { { \displaystyle V _ { k } ^ { \kappa } = \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \kappa } } V _ { i } } , } } \\ { { \displaystyle \quad \overline { { { Y } } } _ { k } ^ { \kappa } = \frac { 1 } { V _ { k } ^ { \kappa } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \kappa } } V _ { i } Y _ { i } , } } \\ { { \displaystyle \quad \overline { { { \Pi } } } _ { k } ^ { a , \kappa } = \frac { 1 } { V _ { k } ^ { \kappa } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \kappa } } V _ { i } \Pi _ { i } ^ { a } , \qquad \displaystyle \overline { { { \Pi } } } _ { k } ^ { b , \kappa } = \frac { 1 } { V _ { k } ^ { \kappa } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { \kappa } } V _ { i } \Pi _ { i } ^ { b } . } } \end{array}
$$

The double-lift chart considers the three graphs

$$
\begin{array} { r } { \left( k , \overline { { Y } } _ { k } ^ { \kappa } \right) , \quad \left( k , \overline { { \Pi } } _ { k } ^ { a , \kappa } \right) \quad \mathrm { a n d } \quad \left( k , \overline { { \Pi } } _ { k } ^ { b , \kappa } \right) \qquad \mathrm { f o r ~ } k = 1 , \ldots , K , } \end{array}\tag{4.13}
$$

on the same axes.

Figure 10 gives the double lift chart for the ratio $\kappa = \Pi ^ { b } / \Pi ^ { a } = m _ { \mathbb { Q } } ( \Pi ) / \Pi$ using ventile binning $K = 2 0$ . On the left-hand side of the plot there are the policies where the unit premium $\Pi ^ { a } = \Pi$ most severely overestimates the losses relative to $\Pi ^ { b } = m _ { \mathbb { Q } } ( \Pi )$ , and on the right-hand side it most severely underestimates the losses; remark that $m _ { \mathbb { Q } } ( \Pi )$ is Q-calibrated by construction. The conclusion of this plot is that the expected values $\overline { { m _ { \mathbb { Q } } ( \Pi ) } } ^ { b , \kappa }$ match the actuals $\overline { { Y } } _ { k } ^ { \kappa }$ much better than the expected values $\overline { { \Pi } } _ { k } ^ { a , \kappa }$ , thus, there is a clear preference for the recalibrated premiums from this double lift chart.

## 5 Tools to assess discrimination and resolution

The previous graphical methods have mainly served to evaluate the calibration property (3.2). These previous tools did not directly target at comparing two diferent pricing schemes $( Y , V , \Pi ^ { a } )$

![](images/4a3f535aac144f1b20e929162d25e598c148a9eeba104f3d5fb0e935d2924c25.jpg)  
Figure 10: Double lift chart using ventile binning $K = 2 0$ for the ratio $\kappa = m _ { \mathbb { Q } } ( \Pi ) / \Pi$

and $( Y , V , \Pi ^ { b } )$ in terms of discrimination and resolution. The present section introduces a single score (summary statistic) that allows one for selecting among multiple competing pricing schemes mainly w.r.t. discrimination, but there will also be some calibration terms involved. This score is based on strictly consistent loss functions; see Gneiting–Raftery [32] and Gneiting [31]. For interpretation of strictly consistent loss functions, we introduce another graphical tool called the Murphy diagram.

## 5.1 Strictly consistent loss function

## 5.1.1 Bregman divergence

We give a brief introduction and present the main tools that are relevant for our purposes. Consider a loss function $( y , x ) \mapsto L ( y , x )$ that compares responses $y$ and predictions $x ;$ the support of this loss function is a possibly infinite rectangle in $\mathbb { R } ^ { 2 }$ . Assume that the response Y has a finite mean E[Y ], and our goal is to determine this mean through an expected loss minimization

$$
{ \widehat { \mu } } _ { 0 } \ \in \ \operatorname { a r g m i n } _ { x } \mathbb { E } \left[ L ( Y , x ) \right] .\tag{5.1}
$$

Generally, this minimization (5.1) will fail to find the true mean $\mathbb { E } [ Y ]$ , which is the motivation to define strictly consistent scoring for mean estimation.

Definition 5.1 A loss function L is consistent for mean estimation of Y if

$$
\begin{array} { r } { { \mathbb E } \left[ L ( Y , { \mathbb E } [ Y ] ) \right] \ \leq \ { \mathbb E } \left[ L ( Y , x ) \right] \qquad f o r \ a l l \ x \ w h e r e \ t h e \ r i g h t . \ h a n d \ s i d e \ i s \ d e f i n e d . } \end{array}\tag{5.2}
$$

The loss function L is strictly consistent for mean estimation if an equality in (5.2) holds if and only if x is the true mean $x = \mathbb { E } [ Y ]$

Under a strictly consistent loss function L for mean estimation, the solution to (5.1) is the singleton $\widehat { \mu } _ { 0 } = \mathbb { E } [ Y ]$ . This makes it natural to measure the quality of a predictor x for Y by the strictly consistent loss $L ( Y , x )$ – smaller is better, and the (unique) minimum is the true mean E[Y ]. Remark that generally, a loss L does not need to be symmetric in its arguments.

The crucial mathematical result of Savage [54] and Gneiting [31, Theorem 7] yields that under mild technical conditions the (strictly) consistent loss functions are exactly the Bregman divergences [8] with (strictly) convex generators $\varphi _ { : }$ , i.e., the (strictly) consistent loss functions for mean estimation take the Bregman divergence form

$$
L _ { \varphi } ( y , x ) = \varphi ( y ) - \varphi ( x ) - \varphi ^ { \prime } ( x ) ( y - x ) \ \geq \ 0 ,\tag{5.3}
$$

with (sub-)gradient $\varphi ^ { \prime }$ of the convex generator $\varphi .$ This choice is regardless of the underlying probability law P as long as the left-hand side of (5.2) exists. We therefore always work with Bregman losses (5.3) for mean estimation and mean validation. Examples of Bregman losses are the square loss function, the Poisson deviance loss, the gamma deviance loss, and generally, all deviance losses from the EDF.

This result of (strict) consistency for mean estimation carries over to conditional probability laws $\mathbb { P } ( \cdot \mid A )$ , yielding that the conditional expectation $\mathbb { E } [ Y \mid A ]$ can be found by the Bregman loss minimization among the A-measurable predictors; we come back to this in Section 6, below. This makes the expected Bregman losses naturally suited to not only compare deterministic predictors x as in (5.2), but we can equally use it for ranking premiums Π. This motivates the following consideration:

For a given Bregman loss $L _ { \varphi }$ and two competing pricing schemes $( Y , V , \Pi ^ { a } )$ and $( Y , V , \Pi ^ { b } )$ , the one with the smaller expected Bregman loss should be preferred, i.e.,

$$
\mathrm { p r e f e r } \ \Pi ^ { a } \ \mathrm { o v e r } \ \Pi ^ { b } \ \mathrm { f o r } \ Y \qquad \Longleftrightarrow \qquad \mathbb { E } [ L _ { \varphi } ( Y , \Pi ^ { a } ) ] \leq \mathbb { E } [ L _ { \varphi } ( Y , \Pi ^ { b } ) ] .\tag{5.4}
$$

Attention. The dificulty with (5.4) is that this preference order depends on the specific choice of the convex generator $\varphi .$ Preference (5.4) generally does not hold simultaneously for all convex generators $\varphi ;$ the simultaneous dominance under all convex generators (with aligned supports) is called forecast dominance, see Kr¨uger–Ziegel [40, Definition 2.1]. In most practical forecast problems forecast dominance is not expected to hold, basically it is related to one premium rule using more information than the other, if both are calibrated, a precise mathematical statement involves convex orders; see Kr¨uger–Ziegel [40, Theorem 3.1].

Testing preference (5.4) for all convex generators $\varphi$ may not be feasible. Section 5.2 considers the simpler class of elementary losses which are parametrized through one single parameter $\theta \in \mathbb { R }$ This makes it easier to test (5.4) for all elementary losses because we work on a parametrized class of (simple) losses. Moreover, the elementary losses can be seen as the building blocks of Bregman losses, this is explained in Section 5.3, below.

## 5.1.2 Exposure-weighted Bregman divergence

The above consistent scoring introduction considers the classical situation without exposures $V > 0$ . As was highlighted by Lindholm et al. [43], in actuarial modeling, one typically considers

weighted Bregman losses, this connects to Example 2.2. For a strictly convex generator $\varphi ,$ (5.1) is replaced by

$$
\mu _ { \mathbb { Q } } \ = \ \underset { x } { \arg \operatorname* { m i n } } \ \mathbb { E } \left[ V L _ { \varphi } ( Y , x ) \right] = \underset { x } { \arg \operatorname* { m i n } } \ \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , x ) \right] ,\tag{5.5}
$$

the latter uses the measure transformation (2.1). This weighted minimization provides a solution that typically difers from E[Y ], namely, it provides the $\mathbb { Q } \mathrm { - }$ mean of $Y$

$$
\mu _ { \mathbb { Q } } = \operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ Y \right] = { \frac { 1 } { \operatorname { \mathbb { E } } [ V ] } } \operatorname { \mathbb { E } } \left[ V Y \right] = { \frac { 1 } { \operatorname { \mathbb { E } } [ V ] } } \operatorname { \mathbb { E } } \left[ Z \right] .\tag{5.6}
$$

Concluding, in actuarial model fitting (5.5), one targets the so-called realized-exposure target (5.6). This accounts correctly for the dependence between the loss costs $Y$ and the exposure $V .$ we also refer to (2.4), and $Z = V Y$ is the loss that the insurer will cover.

A natural consequence of this weighting is that also the preference order (5.4) may change, and we instead consider for a given Bregman loss $L _ { \varphi }$

$$
\begin{array} { r l } { \mathrm { p r e f e r ~ I I } ^ { a } \mathrm { ~ o v e r ~ I I } ^ { b } \mathrm { ~ f o r ~ } Y } & { \quad \Longleftrightarrow \quad \mathbb { E } [ V L _ { \varphi } ( Y , \Pi ^ { a } ) ] \le \mathbb { E } [ V L _ { \varphi } ( Y , \Pi ^ { b } ) ] } \\ & { \quad \Longleftrightarrow \quad \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ^ { a } ) ] \le \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ^ { b } ) ] . } \end{array}\tag{5.7}
$$

We come back to the recalibration step of Definition 3.9.

Example 5.2 (Recalibration step and forecast dominance) The recalibrated unit premium under the $\mathbb { Q } \mathrm { - }$ measure is given by

$$
m _ { \mathbb { Q } } ( \Pi ) = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid \Pi \right] \ \in \ \underset { x } { \arg \operatorname* { m i n } } \ \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , x ) \mid \Pi \right]
$$

for any convex generator $\varphi$ where the expected values exist. This implies

$$
\mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , m _ { \mathbb { Q } } ( \Pi ) ) \right] \leq \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , \Psi ) \right] ,
$$

for any $\sigma ( \Pi )$ -measurable random variable $\Psi$ . In particular, this applies to $\Psi = \Pi$ , and as a consequence

$$
\mathrm { p r e f e r } \ m _ { \mathbb { Q } } ( \Pi ) \ \mathrm { o v e r } \ \Pi \ \mathrm { f o r } \ Y \ \mathrm { u n d e r } \ L _ { \varphi } .
$$

Since this holds for any convex generator $\varphi ,$ we obtain forecast dominance in the sense of Kr¨uger–Ziegel [40, Definition 2.1].

Thus, the recalibrated unit premium $m _ { \mathbb { Q } } ( \Pi )$ is the most accurate $\sigma ( \Pi )$ -measurable predictor of Y under $\mathbb { Q }$ in the forecast dominance sense.

This concludes the example.

## 5.2 Elementary losses and Murphy diagram

## 5.2.1 Elementary losses

We make a first step towards a better understanding of the Bregman loss and a specific choice of a convex generator $\varphi .$ This is done by discussing the elementary losses. The elementary losses have been introduced by Ehm et al. [24], and they allow for a mixture representation of the Bregman loss. The elementary losses are based on the convex functions, for $\theta \in \mathbb { R }$

$$
x \in \mathbb { R } \ \mapsto \ \varphi _ { \theta } ( x ) = ( x - \theta ) _ { + } ,\tag{5.8}
$$

and they are given by

$$
L _ { \theta } ( y , x ) = ( y - \theta ) _ { + } - ( x - \theta ) _ { + } - ( y - x ) \mathbf { 1 } _ { \{ x > \theta \} } .\tag{5.9}
$$

The last term in (5.9) gives a sub-derivative of $x \mapsto \varphi \theta ( x )$ with a non-diferentiability occurring at $x = \theta$ , an other possible choice is $\mathbf { 1 } _ { \{ x \ge \theta \} }$

Notation: $L _ { \varphi }$ denotes a Bregman loss for a general convex generator $\varphi ,$ and $L _ { \theta } = L _ { \varphi _ { \theta } }$ is an elementary loss with generator $\varphi _ { \theta }$ for fixed $\theta \in \mathbb { R }$

The elementary loss (5.9) has many equivalent reformulations

$$
\begin{array} { l c l } { { { \cal L } _ { \theta } ( y , x ) } } & { { = } } & { { ( y - \theta ) _ { + } - ( x - \theta ) _ { + } - ( y - x ) { \bf 1 } _ { \{ x > \theta \} } } } \\ { { } } & { { = } } & { { ( y - \theta ) { \bf 1 } _ { \{ y > \theta \} } - ( x - \theta ) { \bf 1 } _ { \{ x > \theta \} } - ( y - x ) { \bf 1 } _ { \{ x > \theta \} } } } \\ { { } } & { { = } } & { { ( y - \theta ) { \bf 1 } _ { \{ y > \theta \} } - ( y - \theta ) { \bf 1 } _ { \{ x > \theta \} } } } \\ { { } } & { { = } } & { { ( y - \theta ) \left( { \bf 1 } _ { \{ y > \theta \} } - { \bf 1 } _ { \{ x > \theta \} } \right) } } \\ { { } } & { { = } } & { { ( y - \theta ) { \bf 1 } _ { \{ y > \theta \geq x \} } + ( \theta - y ) { \bf 1 } _ { \{ y \leq \theta < x \} } } } \\ { { } } & { { = } } & { { | y - \theta | \left( { \bf 1 } _ { \{ y \leq \theta < x \} } + { \bf 1 } _ { \{ x \leq \theta < y \} } \right) . } } \end{array}\tag{5.10}
$$

The last expression says that it measures the distance $| y - \theta |$ , whenever $\theta$ is between the response y and the forecast x. This will be illustrated and discussed in an AvE plot in Figure 12, below.

## 5.2.2 Murphy diagram

The elementary loss $L _ { \theta }$ is a Bregman loss and it can be used for preference ordering (5.7), it is a consistent loss function for mean estimation, but it is not strictly consistent because (5.8) is only convex. Similar to the selection of the convex generator $\varphi$ for preference ordering (5.7), we may raise the question about the specific choice of θ in the elementary loss $L _ { \theta }$ selection. The Murphy diagram considers the expected elementary loss as a function of θ

$$
\theta \in \mathbb { R } \mapsto \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \theta } ( Y , \Pi ) \right] = \frac { 1 } { \mathbb { E } \left[ V \right] } \mathbb { E } \left[ V L _ { \theta } ( Y , \Pi ) \right] .\tag{5.11}
$$

We now generally work with the exposure-weighted measure $\mathbb { Q } .$

The sample version of the Murphy diagram on the test sample $\mathcal { T } = ( Y _ { i } , V _ { i } , \Pi _ { i } ) _ { i = 1 } ^ { n }$ is given by

$$
\theta \ \mapsto \ { \widehat { \mathbb { E } } } _ { \mathbb { Q } } [ L _ { \theta } ( Y , \Pi ) ] = { \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } } \sum _ { i = 1 } ^ { n } V _ { i } L _ { \theta } ( Y _ { i } , \Pi _ { i } ) .\tag{5.12}
$$

The Murphy diagram was introduced in Ehm et al. [24], and it has become a popular graphical model validation tool, see, e.g., Dimitriadis et al. [22].

![](images/4e6fae67a799c95e5086941d94cc226ab63d3a9fbb43887d6f4e605f2a5d52c5.jpg)

![](images/2f5ec782c1e74830b1ff563c48254ba927947c3ee96fe3f3c14aae4ed14d94cd.jpg)  
Figure 11: (lhs) Murphy diagrams of the unit premium rules Π and $m _ { \mathbb { Q } } ( \Pi )$ , and (rhs) their diference $\Delta _ { \mathbb { Q } } ^ { \mathrm { M u r p h y } } ( \theta ; \Pi , m _ { \mathbb { Q } } ( \Pi ) )$

Figure 11 shows the (empirical) Murphy diagrams (5.12) of the two premium rules Π and $m _ { \mathbb { Q } } ( \Pi )$ in blue and red on the left-hand side, and the right-hand side shows their diference

$$
\begin{array} { r c l } { { \theta \ \mapsto \ \Delta _ { \mathbb { Q } } ^ { \mathrm { M u r p h y } } \left( \theta ; \Pi , m _ { \mathbb { Q } } ( \Pi ) \right) } } & { { = } } & { { \displaystyle { \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \theta } ( Y , \Pi ) ] - \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \theta } ( Y , m _ { \mathbb { Q } } ( \Pi ) ) ] } } } \\ { { } } & { { = } } & { { \displaystyle \sum _ { i = 1 } ^ { 1 } V _ { i } \sum _ { i = 1 } ^ { n } V _ { i } \left( L _ { \theta } ( Y _ { i } , \Pi _ { i } ) - L _ { \theta } ( Y _ { i } , m _ { \mathbb { Q } } ( \Pi _ { i } ) ) \right) . } } \end{array}\tag{5.13}
$$

In Figure 11 (rhs), we observe positivity of this diference $\Delta _ { \mathbb { Q } } ^ { \mathrm { M u r p h y } } ( \theta ; \Pi , m _ { \mathbb { Q } } ( \Pi ) )$ for most of the values $\theta \in \mathbb { R }$ (up to minor noise perturbations). This suggests to prefer premium rule $m _ { \mathbb { Q } } ( \Pi )$ over Π for forecasting $Y .$ Of course, this is clear in this example because it verifies the forecast dominance statement discussed in Example 5.2.

From a practical point of view, we see the following issues:

• For two general pricing rules $\Pi ^ { a }$ and $\Pi ^ { b }$ , we do not expect such a clear preference picture as in Figure 11 (rhs). First, we do not expect that there is this dominance for all $\theta \in \mathbb { R }$ if we consider two general pricing rules that are roughly equally accurate, for example, comparing a gradient boosting rule $\Pi ^ { a }$ to a neural network rule $\Pi ^ { b }$ . Moreover, the noise in the responses $Y$ will contaminate the Murphy decision diagram.

• Figure 11 shows a situation in which we can explicitly compute the recalibrated premium $m _ { \mathbb { Q } } ( \Pi )$ because we know the population model in our example. Generally, this is not the case, and the empirical recalibration step will add an other major source of uncertainty (inaccuracy) to this decision making problem.

• The Murphy diagram in Figure 11 looks appealing and interpretable. However, it is not so obvious from that graph in which range the premiums fit well and in which part they do not. The elementary loss measures the distance $| Y _ { i } - \theta |$ , whenever θ is between the response $Y _ { i }$ and the premium $\Pi _ { i } ,$ see (5.10). As a result, the sample Murphy diagram (5.12) presents an overlap of diferent pairs $( Y _ { i } , \Pi _ { i } )$ for which the selected threshold θ is in between these two values. We will discuss this a bit further in the next graph.

## AvE plot: Contributions to elementary loss

![](images/c8e92593caeccd429ba3cbe00b712f5c0160cf7b41f342f3f8bd06c32ed40399.jpg)  
Figure 12: AvE plot: Contributions to the elementary loss $L _ { \theta }$ for fixed θ (red horizontal line).

We recall the AvE plot of Section 4.3.1. The AvE plot uses the bin averages (4.9). We replace these bin averages by the individual observations (for actuals $Y _ { i }$ vs. expected $\Pi _ { i } )$

$$
( \Pi _ { i } , Y _ { i } ) \quad a n d \quad ( \Pi _ { i } , \Pi _ { i } ) \qquad \mathrm { ~ f o r ~ } i = 1 , \dots , n ,
$$

i.e., we discard the binning. We then add vertical black segments to connect the blue dots $( \Pi _ { i } , Y _ { i } )$ with the orange diagonal dots $( \Pi _ { i } , \Pi _ { i } )$ in Figure 12. Finally, we select a value θ on the y-axis in Figure 12 and we plot a horizontal red line on this level. Every instance i, for which the black vertical segment between $( \Pi _ { i } , Y _ { i } )$ and $( \Pi _ { i } , \Pi _ { i } )$ intersects the red line, contributes to the sample elementary loss (5.12) at the selected level θ, and the size of the contribution is equal to $| Y _ { i } - \theta |$ , see (5.10). For example, the instance with the highest unit premium (to the very right in Figure 12) does not contribute to the selected level θ. This is the formal procedure of computing the sample elementary loss

$$
\widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \theta } ( Y , \Pi ) ] = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { n } V _ { i } L _ { \theta } ( Y _ { i } , \Pi _ { i } ) = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { n } V _ { i } \left( Y _ { i } - \theta \right) \left( \mathbf { 1 } _ { \{ Y _ { i } > \theta \} } - \mathbf { 1 } _ { \{ \Pi _ { i } > \theta \} } \right) .
$$

From Figure 12 it is dificult to interpret these contributions because there are too many instances $i \in \{ 1 , \ldots , n \}$ in the plot. Therefore, we consider an aggregated version (using quantile binning). This aggregated version does not give the same Murphy diagram, but an interpretable aggregated version.

## 5.2.3 The asymmetrically binned Murphy diagram

Figure 12, being based on individual instances, is hardly interpretable. We therefore build a binned version thereof. This binned version does not reproduce the sample Murphy diagram (5.12), but it is only used as a graphical tool for interpretation. However, it has a useful application that gives much deeper insight into pricing and risk classification, see Remark 5.4 and Section 6.4, below.

Our goal is to compare the two premium rules $\Pi ^ { a }$ and $\Pi ^ { b }$ . For binning we need to select one of the two, $\mathrm { e . g . , H ^ { \alpha } }$ , and the resulting plot will depend on this choice – that is why we call the binned version “asymmetric” when comparing the two premium rules $\Pi ^ { a }$ and $\Pi ^ { b }$ . We call the resulting graph the $\Pi ^ { a }$ -binned Murphy diagram. Select premium rule $\Pi ^ { a }$ for binning, and define the $K \in \mathbb N$ bins w.r.t. $\Pi ^ { a }$ by

$$
\widehat { \cal Z } _ { k } ^ { a } = \Big \{ i \in \{ 1 , \ldots , n \} \Big | ( \widehat { \cal G } _ { \Pi ^ { a } } ^ { \mathbb { Q } } ) ^ { - 1 } ( ( k - 1 ) / K ) < \Pi _ { i } ^ { a } \leq ( \widehat { \cal G } _ { \Pi ^ { a } } ^ { \mathbb { Q } } ) ^ { - 1 } ( k / K ) \Big \} .\tag{5.14}
$$

We compute the weighted average loss costs and the binned exposures

$$
\overline { { Y } } _ { k } ^ { a } = \frac { 1 } { V _ { k } ^ { a } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { a } } V _ { i } Y _ { i } \qquad \mathrm { ~ a n d ~ } \qquad V _ { k } ^ { a } = \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { a } } V _ { i } .
$$

Analogously, this yields the weighted $\Pi ^ { a } .$ -binned unit premiums

$$
\overline { { { \Pi } } } _ { k } ^ { a } = \frac { 1 } { V _ { k } ^ { a } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { a } } V _ { i } \Pi _ { i } ^ { a } \qquad \mathrm { ~ a n d ~ } \qquad \overline { { { \Pi } } } _ { k } ^ { b / a } = \frac { 1 } { V _ { k } ^ { a } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { a } } V _ { i } \Pi _ { i } ^ { b } .
$$

This then motivates the (asymmetrically) $\Pi ^ { a }$ -binned Murphy diagrams

$$
\theta ~ \mapsto ~ \frac { 1 } { \sum _ { k = 1 } ^ { K } V _ { k } ^ { a } } \sum _ { k = 1 } ^ { K } V _ { k } ^ { a } \left( \overline { { Y } } _ { k } ^ { a } - \theta \right) \left( \mathbf { 1 } _ { \{ \overline { { Y } } _ { k } ^ { a } > \theta \} } - \mathbf { 1 } _ { \{ \overline { { \Pi } } _ { k } ^ { a } > \theta \} } \right) ,\tag{5.15}
$$

$$
\theta ~ \mapsto ~ \frac { 1 } { \sum _ { k = 1 } ^ { K } V _ { k } ^ { a } } \sum _ { k = 1 } ^ { K } V _ { k } ^ { a } \left( \overline { { Y } } _ { k } ^ { a } - \theta \right) \left( \mathbf { 1 } _ { \{ \overline { { Y } } _ { k } ^ { a } > \theta \} } - \mathbf { 1 } _ { \{ \overline { { \Pi } } _ { k } ^ { b / a } > \theta \} } \right) .\tag{5.16}
$$

The diference in (5.15)-(5.16) stems from the last indicator considering $\overline { { \Pi } } _ { k } ^ { a }$ and $\overline { { \Pi } } _ { k } ^ { b / a }$ , respectively, this is highlighted in red color.

Remark 5.3 • We emphasize that generally

$$
L _ { \theta } ( \overline { { { Y } } } _ { k } ^ { a } , \overline { { { \Pi } } } _ { k } ^ { a } ) \neq \frac { 1 } { V _ { k } ^ { a } } \sum _ { i \in \widehat { \mathcal { T } } _ { k } ^ { a } } V _ { i } L _ { \theta } ( Y _ { i } , \Pi _ { i } ) .\tag{5.17}
$$

The binned version is only used as a graphical tool to reduce the complexity and the noise in the plots, but it does not serve at computing the expected elementary loss.

• We can exchange the role of the two premium rules in (5.15)-(5.16) which gives the $\Pi ^ { b }$ binned Murphy diagram. We present both binning versions to understand the impact of binning.

![](images/3db467c9697ee3e4ff8948aa2d8a2d9f02cfd73dee399388c6770d20b001802b.jpg)

![](images/14a2b86507c0fc0d4390c168299dc87b68a9c42c8296238f8dbf06896c173195.jpg)  
Figure 13: Asymmetrically binned Murphy diagrams of the unit premium rules Π and $m _ { \mathbb { Q } } ( \Pi )$ (lhs) Π-binned and (rhs) m<sub>Q</sub>(Π)-binned for percentile binning $K = 1 0 0$ ; the y-scale is identical in the two plots.

• If the two premium rules $\Pi ^ { a }$ and $\Pi ^ { b }$ provide the same risk ranking (in terms of the unit premium), they will result in the identical binning, thus, the binning is rank based w.r.t. the premium rule.

Figure 13 shows the asymmetrically binned Murphy diagrams with percentile binning $K =$ 100. The left-hand side uses the unit premium rule Π for binning and the right-hand side the recalibrated version $m _ { \mathbb { Q } } ( \Pi )$ . In this example, the binning choice only marginally afects the binned Murphy diagram of the recalibrated premiums $m _ { \mathbb { Q } } ( \Pi )$ , but it impacts the binned Murphy diagram of the premium Π quite significantly; recall that Π does not provide the correct risk ranking, see Remark 4.1. Based on these binned Murphy diagrams, there is a clear preference of $m _ { \mathbb { Q } } ( \Pi )$ over Π, which empirically verifies once more the results of Example 5.2.

Figure 14 shows the AvE plot using percentile binning w.r.t. $\Pi ^ { a } = \Pi ,$ see (4.9). We again connect the actual $( \overline { { \Pi } } _ { k } ^ { a } , \overline { { Y } } _ { k } ^ { a } )$ with the expected $( \overline { { \Pi } } _ { k } ^ { a } , \overline { { \Pi } } _ { k } ^ { a } )$ by vertical black segments, and if these segments intersect the horizontal red line at level θ, they contribute to the elementary loss $L _ { \theta }$ . The upper panel in Figure 14 shows the situation of the unit premium $\Pi ^ { a } = \Pi$ and the lower panel gives the recalibrated case $\Pi ^ { b } = m _ { \mathbb { Q } } ( \Pi )$ . This recalibrated version is binned w.r.t. $\Pi ^ { a } = \Pi$ , and it considers the actual $( \overline { { \Pi } } _ { k } ^ { b / a } , \overline { { Y } } _ { k } ^ { a } )$ against the expected $( \overline { { \Pi } } _ { k } ^ { b / a } , \overline { { \Pi } } _ { k } ^ { b / a } )$ . Generally, in the upper panel the black segments are longer and the unit premium range that contributes to the elementary loss $L _ { \theta }$ for a fixed θ is bigger for Π than $m _ { \mathbb { Q } } ( \Pi )$ . This indicates the calibration issue of Π. Note that the y-levels of the actuals $\overline { { Y } } _ { k } ^ { a }$ are identical in both panels, only their x-coordinates $\overline { { \Pi } } _ { k } ^ { a }$ and $\overline { { { \Pi } } } _ { k } ^ { b / a } = \overline { { { m _ { \mathbb { Q } } ( \Pi ) } } } _ { k } ^ { a }$ difer.

We conclude that the asymmetrically Π<sup>a</sup>-binning Murphy diagram and the resulting AvE plot of Figure 14 is mainly a tool to graphically understand which premium levels contribute to the elementary losses.

AvE plot: Pi−binning  
![](images/eae1a11f348cbb97190c840b05d407349eaa43829831cc265c8c13e35395e436.jpg)

AvE plot: Pi−binning  
![](images/1d84e548db1e302835cdbe199c7892dd7a601c4f8164db16f5f04dcb1f3ef41d.jpg)  
Figure 14: AvE plot with Π-percentile binning: upper panel $\overline { { Y } } _ { k } ^ { a }$ vs. $\overline { { \Pi } } _ { k } ^ { a }$ and lower panel $\overline { { Y } } _ { k } ^ { a }$ vs. $\overline { { \Pi } } _ { k } ^ { b / a }$ with premium rules $\Pi ^ { a } = \Pi$ and $\Pi ^ { b } = m _ { \mathbb { Q } } ( \Pi )$

Remark 5.4 There is a stronger mathematical justification of the Π-binned version of the Murphy diagram that is related to the discussion in Section 6. Fix the number of bins $K \in \mathbb N$ Denote by B the bin allocation of a randomly selected instance (Y, V, Π) among the K quantile bins under the Q-distribution $G _ { \Pi } ^ { \mathbb { Q } }$ . This allows us to define the recalibration step w.r.t. the bin indicator B

$$
\begin{array} { r } { m _ { \mathbb { Q } } ( B ) = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid B \right] \qquad \mathrm { ~ a n d ~ } \qquad \pi _ { B } = \mathbb { E } _ { \mathbb { Q } } \left[ \Pi \mid B \right] . } \end{array}
$$

This gives us the binned Murphy diagram

$$
\theta \ \mapsto \ \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \theta } ( m _ { \mathbb { Q } } ( B ) , \pi _ { B } ) \right] .
$$

The graph (5.15) is a sample version of this binned Murphy diagram. In fact, we have the following Murphy’s decomposition, see Section 6.4 below for a proper treatment,

$$
\begin{array} { r } { \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \theta } ( Y , \pi _ { B } ) \right] = \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \theta } ( Y , m _ { \mathbb { Q } } ( B ) ) \right] + \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \theta } ( m _ { \mathbb { Q } } ( B ) , \pi _ { B } ) \right] . } \end{array}\tag{5.18}
$$

The first term on the right-hand side is the irreducible within-bin variation produced by $Y _ { i }$ and the second term gives a miscalibration error, by using $\pi _ { B }$ instead of the calibrated version $m _ { \mathbb { Q } } ( B )$ . If $\pi _ { B }$ is $\mathbb { Q } \mathrm { . }$ calibrated, this term vanishes, see Section 6, below.

## 5.3 From elementary losses to Bregman divergences

From (5.8)-(5.9) it is immediately clear that elementary losses are Bregman losses, and aggregating these elementary losses will preserve the Bregman loss property. This leads to the following mathematical result. Under mild regularity conditions, Bregman losses can be written as Lebesgue–Stieltjes integrals over the elementary losses, see Ehm et al. [24, Theorem 1b],

$$
L _ { \varphi } ( y , x ) = \int L _ { \theta } ( y , x ) \mathrm { d } \varphi ^ { \prime } ( \theta ) .\tag{5.19}
$$

Thus, Bregman losses consist of mixtures of elementary losses $L _ { \theta }$ with mixing measure $\mathrm { d } \varphi ^ { \prime } ( \theta )$ Inserting the specific form of the elementary loss, we can rewrite (5.19) as

$$
L _ { \varphi } ( y , x ) = \mathbf { 1 } _ { \{ y > x \} } \int _ { x } ^ { y } ( y - \theta ) \ \mathrm { d } \varphi ^ { \prime } ( \theta ) + \mathbf { 1 } _ { \{ y < x \} } \int _ { y } ^ { x } ( \theta - y ) \ \mathrm { d } \varphi ^ { \prime } ( \theta ) .
$$

This now directly connects to the AvE plots of Figures 12 and 14. These plots show for a given unit premium, say $\Pi _ { i } ,$ the segments from $\Pi _ { i }$ to $Y _ { i }$ by the vertical black lines (for a fixed policy $i )$ . To compute the elementary loss $L _ { \theta } ( Y _ { i } , \Pi _ { i } )$ we obtain the length of the segment $| Y _ { i } - \theta | .$ supposed that θ is between $Y _ { i }$ and $\Pi _ { i }$ , see (5.10). The Bregman loss adds a scaling over the segment from $\Pi _ { i }$ to $Y _ { i }$ which is determined by $\mathrm { d } \varphi ^ { \prime } ( \theta )$ . Assume that $\varphi$ is twice diferentiable, then the scaling is given by $\varphi ^ { \prime \prime } ( \theta )$ , and we compute

$$
L _ { \varphi } ( y , x ) = \mathbf { 1 } _ { \{ y > x \} } \int _ { x } ^ { y } ( y - \theta ) \varphi ^ { \prime \prime } ( \theta ) { \mathrm { ~ d } } \theta + \mathbf { 1 } _ { \{ y < x \} } \int _ { y } ^ { x } ( \theta - y ) \varphi ^ { \prime \prime } ( \theta ) { \mathrm { ~ d } } \theta .\tag{5.20}
$$

In the AvE plot of Figure 12, this scaling $\varphi ^ { \prime \prime } ( \theta )$ acts on the y-axis, and it allows us to downgrade, for example, large loss costs by selecting a monotonically decreasing function for $\theta \mapsto \varphi ^ { \prime \prime } ( \theta )$ Such a downgrading of large losses may make sense, namely, we do not generally expect the mean to capture large losses (tail losses), and henceforth, such large losses should not impact the mean model selection too much (if the mean is not dominated by these large losses, i.e., calibration is given). Remark that this downgrading only applies to the model validation part, but not to the model estimation part. I.e., if we would censor claims during model fitting, we would underestimate the true average losses (and likely calibration is violated).

A popular choice for the convex generator $\varphi$ is the Patton family [51] obtained by selecting a   
fixed $p \in \mathbb R$ and setting   
$\varphi ^ { \prime \prime } ( \theta ) = \theta ^ { - p } ,$ (5.21)   
on $\theta > 0 .$ In fact, one can verify that   
p = 0 relates the square loss (2.5),   
p = 1 relates the Poisson deviance loss,   
p = 2 relates the gamma deviance loss,   
p = 3 relates the inverse Gaussian deviance loss.

All these losses belong to the EDF, and they build a model hierarchy with an increasing variance behavior; see W¨uthrich–Merz [59]. We demonstrate the case of the gamma deviance loss $p = 2$ in Example 6.4, below, and the Poisson case is studied in Section 8.1, below.

## 5.4 Sample Bregman loss and currency invariant preferences

The estimation of the expected Bregman loss $\mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ) ]$ is straightforward from an i.i.d. test sample $\mathcal { T } = ( Y _ { i } , V _ { i } , \Pi _ { i } ) _ { i = 1 } ^ { n }$ . Namely, we set

$$
{ \widehat { \mathbb { E } } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ) ] = { \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } } \sum _ { i = 1 } ^ { n } V _ { i } L _ { \varphi } ( Y _ { i } , \Pi _ { i } ) .\tag{5.22}
$$

This is the loss figure that is typically reported in statistical analysis; see, e.g., W¨uthrich et al. [60]. Based on this sample Bregman loss, we can perform preference ordering (5.7) of diferent unit premium rules. We generally do not expect forecast dominance between comparable unit premium rules that are based on the same set of information. Therefore, we typically select one (single) Bregman loss $L _ { \varphi } { \mathrm { : } }$ , and the discussion around (5.21) can support us in this selection.

There is one more critical point to be considered, namely, the implied preference order (5.7) should be scale invariant, i.e., it should not depend on the currency

$$
\widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ^ { a } ) ] \leq \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ^ { b } ) ] \quad \Longleftrightarrow \quad \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( c Y , c \Pi ^ { a } ) ] \leq \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( c Y , c \Pi ^ { b } ) ] ,\tag{5.23}
$$

for any $c > 0$ . This is to say, that model selection should be unit-free as we want to select the same forecast model under Euros or Dollars. The scale invariance (5.23) is not generally given. However, if we select a loss function from the Patton family [51], given by (5.21), we observe by a change of variable

$$
\begin{array} { r c l } { { { \cal L } _ { \varphi } ( c y , c x ) } } & { { = } } & { { { \bf 1 } _ { \{ c y > c x \} } \displaystyle \int _ { c x } ^ { c y } \frac { c y - \theta } { \theta ^ { p } } ~ \mathrm { d } \theta + { \bf 1 } _ { \{ c y < c x \} } \displaystyle \int _ { c y } ^ { c x } \frac { \theta - c y } { \theta ^ { p } } ~ \mathrm { d } \theta } }  \\ { { } } & { { = } } & { { c ^ { 2 - p } L _ { \varphi } ( y , x ) . } } \end{array}\tag{5.24}
$$

Hence, the Patton family [51] is positively homogeneous of degree $2 - p$ . This guarantees that the preference order (5.23) is preserved under diferent currencies. In fact, this motivation is rather similar to Patton [51, Proposition 4] and it also gives support to consider Tweedie’s dominance of Denuit et al. [14]; Tweedie’s class of deviance losses is a subclass of the Patton family (5.21) because Tweedie’s class does not exist for $p \in ( 0 , 1 )$ ; see Jørgensen [38, Theorem 2].

## 6 Murphy’s decomposition

## 6.1 Murphy’s decomposition and Bregman score

Since the (strict) consistency of the Bregman loss (5.3) does not depend on the specific choice of the probability law, it also carries over to conditional probabilities.

Lemma 6.1 Consider an information set ${ \mathcal { A } } \subset { \mathcal { F } }$ and define the conditional mean

$$
m _ { \mathbb { Q } } ( { \cal A } ) = \mathbb { E } _ { \mathbb { Q } } [ Y \mid { \cal A } ] .
$$

Assume X is A-measurable. There is the conditional Bregman Pythagorean relation,

$$
\operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } \left( Y , X \right) \mid { \mathcal { A } } \right] = \operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } \left( Y , m _ { \mathbb { Q } } ( A ) \right) \mid A \right] + L _ { \varphi } \left( m _ { \mathbb { Q } } ( A ) , X \right) \qquad a . s .\tag{6.1}
$$

This lemma is proved in the appendix.

For a strictly convex generator $\varphi ,$ the last term in (6.1) vanishes if and only if $X = m _ { \mathbb { Q } } ( { \mathcal { A } } )$ We translate this to the Q-calibration considerations (3.2). Assume that the information set $\mathcal { A } = \sigma ( \Pi )$ is generated by the price Π.

The conditional Bregman Pythagorean relation (6.1) yields

$$
\operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } \left( Y , \Pi \right) \mid \Pi \right] = \operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } \left( Y , m _ { \mathbb { Q } } ( \Pi ) \right) \mid \Pi \right] + L _ { \varphi } \left( m _ { \mathbb { Q } } ( \Pi ) , \Pi \right) .\tag{6.2}
$$

This relation is crucial. It measures the accuracy of the price Π for predicting the claim Y under a Bregman loss $L _ { \varphi }$ with strictly convex generator $\varphi .$ This accuracy is decomposed into two terms: (1) the accuracy of the recalibrated price $m _ { \mathbb { Q } } ( \Pi )$ , and (2) the discrepancy between the price Π and its recalibrated version $m _ { \mathbb { Q } } ( \Pi )$ . The first term (1) reflects discrimination and the second term (2) calibration. This second term vanishes for Q-calibrated prices Π. Note that these considerations have already been used in Remark 5.4 where we studied the Π-binned version of the Murphy diagram.

Murphy’s decomposition [50] modifies identity (6.2) in two ways. First, it considers the population version by taking expected values w.r.t. the exposure-weighted measure $\mathbb { Q } .$ . It calibrates the consideration to the global mean $\mu _ { \mathbb { Q } } = \mathbb { E } _ { \mathbb { Q } } [ Y ]$

Corollary 6.2 (Murphy’s decomposition) Select a convex generator $\varphi .$ . Murphy’s decomposition is given by

$$
\begin{array} { r l r } { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ) ] } & { = } & { \underbrace { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) ] } _ { = : \mathrm { U N C } _ { \varphi } } - \underbrace { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \mu _ { \mathbb { Q } } ) ] } _ { = : \mathrm { R E S } _ { \varphi } ( \Pi ) } + \underbrace { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \Pi ) ] } _ { = : \mathrm { M C B } _ { \varphi } ( \Pi ) } . } \end{array}\tag{6.3}
$$

This corollary is proved in the appendix.

The expected Bregman loss on the left-hand side of (6.3) is decomposed into an uncertainty term (UNC), a resolution term (RES) and a miscalibration term (MCB). UNC measures the total fluctuations contained in the response $Y$ (relative to its deterministic mean), RES measures the resolution (discrimination) that can be achieved by the calibrated version of the unit premium Π, and MCB quantifies the calibration error. All three terms are non-negative, and they vanish for a strictly convex generator $\varphi$ if and only if

$$
\begin{array} { r l } { \mathrm { R E S } _ { \varphi } ( \Pi ) = 0 \quad \Longleftrightarrow } & { { } m _ { \mathbb { Q } } ( \Pi ) = \mu _ { \mathbb { Q } } , \qquad \mathrm { a . s . } , } \\ { \mathrm { M C B } _ { \varphi } ( \Pi ) = 0 \quad \Longleftrightarrow } & { { } m _ { \mathbb { Q } } ( \Pi ) = \Pi , \qquad \mathrm { a . s . } } \end{array}
$$

We turn Murphy’s decomposition into a score where bigger means better, and so that it is calibrated to the global mean $\mu _ { \mathbb { Q } }$

Definition 6.3 (Bregman score) We define the Bregman score by

$$
\begin{array} { r c l } { S _ { \varphi } ( Y , \Pi ) } & { = } & { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) ] - \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ) ] } \\ & { = } & { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \mu _ { \mathbb { Q } } ) ] - \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \Pi ) ] } \\ & { = } & { \mathrm { R E S } _ { \varphi } ( \Pi ) - \mathrm { M C B } _ { \varphi } ( \Pi ) . } \end{array}\tag{6.4}
$$

The resulting score system is anchored at $S _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) = 0$ , and it quantifies the two important terms of forecast accuracy:

• Resolution: The resolution term measures how well a premium rule Π can discriminate the response $Y$ by quantifying the accuracy gain of the Q-calibrated version $m _ { \mathbb { Q } } ( \Pi )$ relative to the global mean $\mu _ { \mathbb { Q } }$

• Calibration: The calibration term measures how well the premium rule Π is calibrated to the response Y by quantifying the miscalibration.

We come back to the preference order (5.7).

Assume we have two pricing schemes $( Y , V , \Pi ^ { a } )$ and $( Y , V , \Pi ^ { b } )$ and a given convex generator $\varphi .$ This gives the preference order

$$
\begin{array} { r l r l } { \mathrm { p r e f e r ~ } \Pi ^ { a } \mathrm { ~ o v e r ~ } \Pi ^ { b } \mathrm { ~ f o r ~ } Y } & { \iff } & & { S _ { \varphi } ( Y , \Pi ^ { a } ) \geq S _ { \varphi } ( Y , \Pi ^ { b } ) } \\ & { \iff } & & { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ^ { a } ) ] \leq \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ^ { b } ) ] } \\ & { \iff } & & { \mathbb { E } [ V L _ { \varphi } ( Y , \Pi ^ { a } ) ] \leq \mathbb { E } [ V L _ { \varphi } ( Y , \Pi ^ { b } ) ] . } \end{array}\tag{6.5}
$$

The computation of this preference order can be done by its sample version (5.22) on the test sample T . This yields

$$
\begin{array} { r c l } { \widehat { S } _ { \varphi } ( Y , \Pi ) } & { = } & { \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) ] - \widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \Pi ) ] } \\ & { = } & { \displaystyle \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { n } V _ { i } \left( L _ { \varphi } ( Y _ { i } , \mu _ { \mathbb { Q } } ) - L _ { \varphi } ( Y _ { i } , \Pi _ { i } ) \right) . } \end{array}\tag{6.6}
$$

We assume that the premium rule Π and the global mean $\mu _ { \mathbb { Q } }$ have been determined on an independent learning sample, and (6.6) reflects an out-of-sample score.

Example 6.4 (Gamma deviance scoring) We compute the preference order (6.5) for our synthetic data example introduced in Section 3.3. We compute the sample version (6.6) based on the test sample $\tau$ of sample size $n = 1 0 0 , 0 0 0$ ; this is the identical test data that has been used in Section 4.3. The responses of this data have been generated by conditional gamma distributions (3.16). This makes it natural to use the gamma deviance loss from the Bregman loss family. The gamma deviance loss corresponds to the inverse quadratic scaling $\varphi ^ { \prime \prime } ( \theta ) = 2 / \theta ^ { 2 }$ in (5.21) and it respects the currency invariance (5.23) in preference ordering.

We first derive the gamma deviance loss before providing the numerical results of the premium rules Π and $m _ { \mathbb { Q } } ( \Pi )$ . We select for $x > 0$

$$
\varphi ( x ) = - 2 \log x .\tag{6.7}
$$

This yields first and second derivatives on $\mathbb { R } _ { + }$

$$
\varphi ^ { \prime } ( x ) = - { \frac { 2 } { x } } \qquad \mathrm { a n d } \qquad \varphi ^ { \prime \prime } ( x ) = { \frac { 2 } { x ^ { 2 } } } > 0 .
$$

Thus, we have a strictly convex generator $\varphi$ on $\mathbb { R } _ { + }$ and its Bregman loss is given by

$$
L _ { \varphi } ( y , x ) = - 2 \log y + 2 \log x + { \frac { 2 } { x } } \left( y - x \right) = 2 \left( { \frac { y - x } { x } } - \log \left( { \frac { y } { x } } \right) \right) .\tag{6.8}
$$

This Bregman loss (6.8) is the gamma deviance loss. The gamma deviance loss is obtained from the EDF by selecting the cumulant function $\kappa ( \vartheta ) = - \log ( - \vartheta )$ , for $\vartheta < 0$ . This cumulant function generates the gamma distribution, it has canonical link $( \kappa ^ { \prime } ) ^ { - 1 } ( m ) = - 1 / m .$ , for $m > 0 ,$ and variance function $\mathcal { V } ( m ) = \kappa ^ { \prime \prime } ( ( \kappa ^ { \prime } ) ^ { - 1 } ( m ) ) = m ^ { 2 } ;$ see W¨uthrich–Merz [59, Section 2.2.2]. The power parameter of $p = 2$ of the variance function $\nu$ exactly refers to parameter $p$ of the Patton family (5.21).

<table><tr><td>premium rules</td><td>Bregman loss (6.8)</td><td>Bregman score (6.6) in  $1 0 ^ { - 3 }$ </td></tr><tr><td>global mean model  $\mu _ { \mathbb { Q } }$ </td><td>0.31928</td><td></td></tr><tr><td>premium rule II</td><td>0.31901</td><td>0.2743</td></tr><tr><td>premium rule  $m _ { \mathbb { Q } } ( \Pi )$ </td><td>0.31778</td><td>1.5017</td></tr></table>

Table 1: Gamma Bregman loss $\widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \cdot ) ]$ and Bregman score $\widehat { S } _ { \varphi } ( Y , \cdot )$ of the considered premium rules under the gamma deviance loss choice (6.7) for the generator $\varphi .$

Table 1 presents the resulting scores, and we give clear preference to the recalibrated premium rule $m _ { \mathbb { Q } } ( \Pi ) \ ( 1 . 5 0 1 7 \cdot 1 0 ^ { - 3 } )$ over the original premium rule Π $( 0 . 2 7 4 3 \cdot 1 0 ^ { - 3 } )$ . In view of the previous graphs and results this is not surprising, as Π has a serious calibration issue. In fact, the miscalibration term is zero for the recalibrated price $m _ { \mathbb { Q } } ( \Pi )$ , this follows from (6.2). This motivates the estimation

$$
\widehat { \mathrm { M C B } } _ { \varphi } ( \Pi ) = \widehat { S } _ { \varphi } ( Y , m _ { \mathbb { Q } } ( \Pi ) ) - \widehat { S } _ { \varphi } ( Y , \Pi ) .\tag{6.9}
$$

This immediately yields the miscalibration error estimate of Π for $Y$

$$
\widehat { \mathrm { M C B } _ { \varphi } } ( \Pi ) \ = \ 1 . 2 2 7 4 \cdot 1 0 ^ { - 3 } .
$$

This is the Q-calibration defect of the unit premium rule Π for Y measured under the gamma deviance loss. ■

The previous example leaves us with two open questions:

(1) The previous example essentially benefits from the fact that we can explicitly compute the recalibrated unit premium $m _ { \mathbb { Q } } ( \Pi )$ . Only this allows us to obtain Murphy’s decomposition of the Bregman score into the resolution term and the miscalibration term, in particular, this allows us to compute (6.9). What can we do in a real-world situation where the recalibrated unit premium cannot be computed explicitly?

(2) Table 1 gives a clear preference to one of the two premium rules in terms of (6.5). Can we understand this numerical result graphically, e.g., identifying the weaknesses of the premium rules?

The first question (1) needs estimation of the corresponding split. The second question (2) is already partly answered by the AvE plots of Figure 14 and we will provide a diferent perspective on the same results.

## 6.2 Graphical illustration of the Bregman score

Having two unit premium rules $\Pi ^ { a }$ and $\Pi ^ { b }$ , we aim at better understanding how they form the Bregman score estimate (6.6). We again select one of the two unit premium rules, say $\Pi ^ { a }$ , for an ordering and, thus, the following plots will be asymmetric if applied to both unit premium rules. Let $( [ i ] _ { a } ) _ { i = 1 } ^ { n }$ denote the ordered sequence w.r.t. $( \Pi _ { i } ^ { a } ) _ { i = 1 } ^ { n }$ , that is, $\Pi _ { [ i ] _ { a } } ^ { a } \ \le \ \Pi _ { [ i + 1 ] _ { a } } ^ { a }$ for all $i = 1 , \ldots , n - 1$ . This motivates to consider the paired iterative Bregman score aggregation graphs

$$
m \in \{ 1 , \ldots , n \} \mapsto \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i ] _ { a } } \left( L _ { \varphi } ( Y _ { [ i ] _ { a } } , \mu _ { \mathbb { Q } } ) - L _ { \varphi } ( Y _ { [ i ] _ { a } } , \Pi _ { [ i ] _ { a } } ^ { a } ) \right) ,\tag{6.10}
$$

$$
m \in \{ 1 , \ldots , n \} \mapsto \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i ] _ { a } } \left( L _ { \varphi } ( Y _ { [ i ] _ { a } } , \mu _ { \mathbb { Q } } ) - L _ { \varphi } ( Y _ { [ i ] _ { a } } , \Pi _ { [ i ] _ { a } } ^ { b } ) \right) .\tag{6.11}
$$

The only diference is the last premium rule indicated by red color. Note that we consider a paired graph in (6.10)-(6.11), meaning that the aggregation order is the same for both unit premium rules.

![](images/8be0476b5bf62fc6ad9b27da280c528392e9b930054dfd5f74c1a8c8c29c7b52.jpg)  
Iterative Bregman score aggregation

![](images/0747190d7a6795fa7a3c5690145ef15ed21edf5baf30e81e27243a6be8dcf74f.jpg)  
Figure 15: Paired iterative Bregman score aggregation from smallest to biggest unit premium: (lhs) ordered w.r.t. unit premium $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and (rhs) ordered w.r.t. recalibrated unit premium $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$

Figure 15 shows the paired iterative Bregman score aggregation (6.10)-(6.11), on the left-hand side ordered w.r.t. the unit premium $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and on the right-hand side w.r.t. the recalibrated unit premium $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ . The final value for $m = n$ precisely gives the Bregman scores of Table 1.

These curves are increasing if the unit premium prediction is more accurate than the global mean $m _ { \mathbb { Q } } .$ , and decreasing otherwise. We observe that for $m _ { \mathbb { Q } } ( \Pi )$ the curve is generally increasing with flat pieces where the recalibrated unit premium $m _ { \mathbb { Q } } ( \Pi )$ takes roughly the same value as the global mean $m _ { \mathbb { Q } }$ . On the right-hand side this flat piece is in the middle of the graph, on the left-hand side the flat piece is partitioned into two parts by the wrong risk ordering of Π on the x-axis. On the other hand, the original unit premium Π has many predictions that perform worse than the global mean $m _ { \mathbb { Q } }$ (negative slopes in blue curves), which clearly shows that the unit premium Π is not an accurate predictor. This gives a graphical illustration how the Bregman score of Table 1 is composed across its unit premium range, and the downward slopes allow us to identify premium ranges with inaccurate predictions.

## 6.3 Sample computation of resolution and miscalibration

The next problem that we consider is Murphy’s decomposition into resolution and miscalibration terms of the Bregman score in the case the recalibrated premium $m _ { \mathbb { Q } } ( \Pi )$ is unknown, which is the typical situation in applications. Recall (6.4). This gives us two diferent ways of expressing the miscalibration term

$$
\begin{array} { r c l } { { \mathrm { M C B } _ { \varphi } ( \Pi ) } } & { { = } } & { { \mathbb { E } _ { \mathbb { Q } } [ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \Pi ) ] , } } \\ { { \mathrm { M C B } _ { \varphi } ( \Pi ) } } & { { = } } & { { S _ { \varphi } ( Y , m _ { \mathbb { Q } } ( \Pi ) ) - S _ { \varphi } ( Y , \Pi ) . } } \end{array}
$$

The goal is to use these two representations for deriving an estimation. Assume we have estimates $\widehat { m } _ { \mathbb { Q } } ( \Pi _ { i } )$ , we can use both of the two identities to receive a miscalibration term estimate

$$
\widehat { \mathrm { M C B } } _ { \varphi } ^ { ( 1 ) } ( \Pi ) = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { n } V _ { i } L _ { \varphi } ( \widehat { m } _ { \mathbb { Q } } ( \Pi _ { i } ) , \Pi _ { i } ) ,\tag{6.12}
$$

$$
\widehat { \mathrm { M C B } } _ { \varphi } ^ { ( 2 ) } ( \Pi ) \ = \ \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { n } V _ { i } \left( L _ { \varphi } ( Y _ { i } , \Pi ) - L _ { \varphi } ( Y _ { i } , \widehat { m } _ { \mathbb { Q } } ( \Pi _ { i } ) ) \right) .\tag{6.13}
$$

We present these two diferent formulas (6.12)-(6.13) because they present diferent viewpoints, and they give diferent results on finite samples (they are identical in the population version). The first one (6.12) does not use the responses for given $\widehat { m } _ { \mathbb { Q } } ( \Pi _ { i } )$ , whereas the second one (6.13) does. Consequently, the two estimates are expected to difer. From this viewpoint, one might always prefer the first one because it does not involve the noisy part of the responses. However, the accuracy of (6.12) crucially depends on the accuracy that we can get in the (empirical) recalibration step yielding the estimates $\widehat { m } _ { \mathbb { Q } } ( \Pi _ { i } )$ . This is the critical step in the empirical version of Murphy’s decomposition. Usually, one uses the isotonic regression (4.12) on the test sample T for the recalibration step, which yields estimates

$$
\widehat { m } _ { \mathbb { Q } } ( \Pi _ { i } ) = ( \widehat { \pmb { m } } _ { \mathbb { Q } } ^ { \mathrm { i s o } } ) _ { i } \qquad \mathrm { f o r } \ i = 1 , \dots , n .
$$

The results in the following list illustrate that this isotonic regression step underestimates the miscalibration error (0.9934 vs. 1.2274) in the first version (6.12), because the isotonic regression

estimate is comparably crude and tends to be too close to the unit premium $\Pi _ { i } \colon$

$$
\begin{array} { r l r } { \mathrm { M C B } _ { \varphi } ( \Pi ) } & { = } & { 1 . 2 2 7 4 , } \\ { \widehat { \mathrm { M C B } } _ { \varphi } ^ { ( 1 ) } ( \Pi ) } & { = } & { 0 . 9 9 3 4 , } \\ { \widehat { \mathrm { M C B } } _ { \varphi } ^ { ( 2 ) } ( \Pi ) } & { = } & { 1 . 2 5 9 1 . } \end{array}\tag{6.14}
$$

On the other hand, the second version (6.13) overestimates the miscalibration error (1.2591 vs. 1.2274). This is due to an in-sample bias, because we use the test sample $\tau$ to fit the isotonic regression (4.12) and we use the same observations to evaluate (6.13). Of course, we could mitigate the last dificulty if we had additional (independent) observations or by crossvalidation. In our numerical example (6.14), the true value is in between the two estimates, however, we do not know whether this holds more generally or only in this example.

## 6.4 Illustration of resolution and miscalibration

We come back to Remark 5.4. Consider $K \in \mathbb N$ bins, and denote by B the bin allocation of a randomly selected instance $( Y , V , \Pi )$ among the K quantile bins under the Q-distribution $G _ { \Pi } ^ { \mathbb { Q } }$ Recall

$$
\begin{array} { r } { m _ { \mathbb { Q } } ( B ) = \mathbb { E } _ { \mathbb { Q } } \left[ Y \mid B \right] \qquad \mathrm { ~ a n d ~ } \qquad \pi _ { B } = \mathbb { E } _ { \mathbb { Q } } \left[ \Pi \mid B \right] . } \end{array}
$$

The conditional Bregman Pythagorean relation yields, see Lemma 6.1,

$$
\operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } \left( Y , \pi _ { B } \right) \mid B \right] = \operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } \left( Y , m _ { \mathbb { Q } } ( B ) \right) \mid B \right] + L _ { \varphi } \left( m _ { \mathbb { Q } } ( B ) , \pi _ { B } \right) \qquad \mathrm { a . s . }
$$

Taking the E<sub>Q</sub>-expectation proves (5.18), and it yields the following Murphy’s decomposition, the proof is identical to the one of Corollary 6.2,

$$
\begin{array} { r c l } { { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , \pi _ { B } ) \right] } & { = } & { { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , m _ { \mathbb { Q } } ( B ) ) \right] + { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( m _ { \mathbb { Q } } ( B ) , \pi _ { B } ) \right] } \\ & { = } & { { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) \right] - { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( m _ { \mathbb { Q } } ( B ) , \mu _ { \mathbb { Q } } ) \right] + { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( m _ { \mathbb { Q } } ( B ) , \pi _ { B } ) \right] . } \end{array}
$$

The first term on the first line on the right-hand side is the irreducible within-bin variation produced by the responses Y around $m _ { \mathbb { Q } } ( B )$ , and the second term gives a miscalibration error, by using $\pi _ { B }$ instead of the calibrated version $m _ { \mathbb { Q } } ( B )$ . There is a subtle diference here to Murphy’s decomposition (6.3), namely, we consider B-partitioning and we do not recalibrate $\pi _ { B }$ resulting in the calibrated version $m _ { \mathbb { Q } } ( B )$ . Therefore, the second line does not present the classic Murphy’s decomposition (6.3), but a variant that uses B-binning.

This then inspires the following plots of Semenovich–Dolman [55]:

• Figure 16 (top-lhs) is precisely the sample lift chart illustrated in Figure 7; for better visualization we use ventile binning $K = 2 0$ . Moreover, we adapt the notation to the present section using the binning indicator B, and we add to this lift chart the global mean $\mu _ { \mathbb { Q } } = \mathbb { E } _ { \mathbb { Q } } [ Y ]$ (black horizontal line) as well as the true calibrated bin means $m _ { \mathbb { Q } } ( B )$ (red dots).

• Figure 16 (top-rhs) shows the estimation error ${ \overline { { Y } } } _ { B } - m _ { \mathbb { Q } } ( B )$ comparing the sample means ${ \overline { { Y } } } _ { B }$ on the bins $B \in \{ 1 , \ldots , K \}$ , see (4.7), to their recalibrated population counterparts $m _ { \mathbb { Q } } ( B )$ . This is the unavoidable estimation error of the recalibration step on the bins that results from the irreducible risk, i.e., this is the error we collect under an unknown population distribution.

![](images/dc928183810ec3e1b5453e4ba76ddd9f6febd809a5b6ffc1849a91de35cd01f7.jpg)

![](images/9c20414755c149c630dff21b2b6f8b971465a80644f18c29bb558e07835834df.jpg)

![](images/43950bf584d4f56857c6e8070cdddfe8236754a6ee8ef9bc8b965ad02d80b8dc.jpg)

![](images/bbfb9aa3b9fa70622f040cfdc23ac50b435fe947f78f76cc9d155f1e73f8d633.jpg)  
Figure 16: (top-lhs) Lift chart using ventile binning $K = 2 0$ , (top-rhs) illustration of estimation error ${ \overline { { Y } } } _ { B } - m _ { \mathbb { Q } } ( B )$ , (bottom-lhs) visualization of resolution gain over the global mean model $\mu _ { \mathbb { Q } }$ , and (bottom-rhs) visualization of miscalibration error (raw diferences before they enter the Bregman loss).

• Figure 16 (bottom-lhs) shows the resolution term (raw diferences) comparing the calibrated means $m _ { \mathbb { Q } } ( B )$ to the global mean $\mu _ { \mathbb { Q } }$ . This is the (calibrated) price granularity that we can get on the selected bins, resulting in the corresponding discrimination and risk classification. The red area shows the cross-subsidy if one would use the egalitarian price $\mu _ { \mathbb { Q } }$ instead.

• Figure 16 (bottom-rhs) shows the miscalibration error (raw diferences) if using $\pi _ { B }$ instead

of the calibrated version $m _ { \mathbb { Q } } ( B )$ obtained on the B-binning granularity.

The illustration in Figure 16 shows resolution and calibration in the lift chart. The binning was performed w.r.t. the premium scheme Π. However, the same analysis applies to any binnning of the insurance portfolio, and this relates to the original work in the 1960’s about an optimal risk classification; see Bailey–Simon [3], Bailey [2] and Jung [39]. In a more modern view, such a risk classification should also be void of unfair discrimination; see Lindholm et al. [45].

## 7 Gini score

## 7.1 Introduction: Risk ranking

In the previous section, the discussion has been centered around calibration and discrimination. A couple of times, we made some statements about risk ranking, in particular, the isotonic regression is based on such a (correct) risk ranking. The present section discusses the question of accurate risk rankings. A popular risk ranking measure is the Gini score which goes back to the Gini index [29, 30] in economics that was used to study disparity of wealth distributions within diferent populations; the Gini score is based on the Lorenz curve [48]. This Gini concept has been adopted in statistical modeling for risk ranking assessments, and it has also become a popular model validation tool in credit scoring and actuarial modeling; see Gourieroux–Jasiak [35], Frees et al. [26, 27] and Denuit et al. [17, 18]. The following outline is based on Brauer– W¨uthrich [7], who discussed the Gini score under exposures (case weights) and ties in the data, and we directly focus on the sample version of the Gini score.

The main question we study in this section is whether the order statistics of the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and the observed loss costs $( Y _ { i } ) _ { i = 1 } ^ { n }$ are aligned. This consideration is purely rank based, and any strictly monotonically increasing transformation of the unit premium provides the same result. Thus, the following considerations do not assess calibration and they also are not based on strictly consistent loss functions for mean estimation; see W¨uthrich [56] for more discussion.

## 7.2 Cumulative accuracy profile

We start by solely considering the observations $( Y _ { i } ) _ { i = 1 } ^ { n } { \mathrm { : } }$ , and the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ will be integrated in a second step. This first step constructs the Leimkuhler curve, which is a mirrored version of the Lorenz curve. Both curves consider an order statistics of the observations $( Y _ { i } ) _ { i = 1 } ^ { n }$ the former in decreasing order and the latter in increasing order. Consider the decreasing order statistics of the unit loss costs

$$
Y _ { ( 1 ) } \geq Y _ { ( 2 ) } \geq . . . \geq Y _ { ( n ) } ,\tag{7.1}
$$

with a deterministic rule if there are ties in the observations $( Y _ { i } ) _ { i = 1 } ^ { n }$ . The lower round brackets $( i )$ in (7.1) indicate that we ordered the responses from biggest to smallest. We map precisely this order to the exposures $( V _ { [ i ] } ) _ { i = 1 } ^ { n }$ and we use square brackets $[ i ]$ to indicate that this is the implied order from the responses $( Y _ { ( i ) } ) _ { i = 1 } ^ { n }$ . For example, $V _ { [ 5 ] }$ is the exposure of the fifth biggest unit loss costs in (7.1).

We define the increasing sequences (running totals)

$$
m \in \{ 1 , \ldots , n \} \qquad \mapsto \qquad u _ { m } = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i ] } ,\tag{7.2}
$$

$$
m \in \{ 1 , \ldots , n \} \qquad \mapsto \qquad L _ { m } = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } Y _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i ] } Y _ { ( i ) } ,\tag{7.3}
$$

and we initialize $u _ { 0 } = L _ { 0 } = 0$ for $m = 0$ . Note that $( u _ { m } ) _ { m = 0 } ^ { n }$ is strictly increasing and $( L _ { m } ) _ { m = 0 } ^ { n }$ is non-decreasing, both live in the unit interval.

The first line (7.2) considers the exposure-weighted sample distribution of the unit loss costs because the exposures are ordered w.r.t. $( Y _ { i } ) _ { i = 1 } ^ { n } ;$ see (4.5) for the premium counterpart. There is one diference though in (7.2) compared to (4.5), namely, we disaggregate the ties of $( Y _ { i } ) _ { i = 1 } ^ { n }$ to retain the original cardinality n of the training sample T. Since below we are going to linearly interpolate, the selected suborder in the ties (7.1) will not impact the results. The second line (7.3) is the exposure-weighted Leimkuhler curve that measures the weighted contributions of the decreasing unit loss costs $( Y _ { ( i ) } ) _ { i = 1 } ^ { n }$ to the total loss costs $\textstyle \sum _ { i = 1 } ^ { n } Z _ { i } = \sum _ { i = 1 } ^ { n } V _ { i } Y _ { i }$

The Leimkuhler curve is obtained by linearly interpolating between the points

$$
( u _ { m } , L _ { m } ) \qquad \mathrm { f o r } \ m = 0 , \ldots , n .\tag{7.4}
$$

The Leimkuhler curve is a concave curve in the unit square $[ 0 , 1 ] ^ { 2 }$ that connects the two corners (0, 0) and (1, 1). This Leimkuhler curve is illustrated in cyan color in Figure 17 (lhs). This is the upper benchmark (upper bound) of the Gini score because it considers the perfect ordering of the unit loss costs $( Y _ { ( i ) } ) _ { i = 1 } ^ { n }$ , and our goal is to see whether the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ align with this order.

To compute the Gini score, we construct a second curve called the cumulative accuracy profile (CAP); its mirrored version is also called concentration curve, see Denuit et al. [17]. The construction of the CAP slightly difers from the Leimkuhler curve, because now the suborder in the ties of the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ matters.

The decreasing order statistics of the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ is constructed in two steps. In the first step, we order the unit premiums in decreasing order $\Pi _ { ( 1 ) } \geq \Pi _ { ( 2 ) } \geq . . . \geq \Pi _ { ( n ) }$ . This order statistics may have ties, say, we may have $\Pi _ { ( k ) } = \Pi _ { ( k + 1 ) } = \ldots = \Pi _ { ( k + l ) }$ . In such ties we consider two suborders implied by the corresponding responses. The first suborder

$$
\Pi _ { ( 1 \downarrow ) } \geq \Pi _ { ( 2 \downarrow ) } \geq . . . \geq \Pi _ { ( n \downarrow ) } ,\tag{7.5}
$$

is implied by ordering the ties of the unit premiums in a decreasing order w.r.t. the responses $( Y _ { i } ) _ { i = 1 } ^ { n }$ , and equivalently in increasing order w.r.t. the responses $( Y _ { i } ) _ { i = 1 } ^ { n }$ denoted by

$$
\Pi _ { ( 1 \uparrow ) } \geq \Pi _ { ( 2 \uparrow ) } \geq . . . \geq \Pi _ { ( n \uparrow ) } .\tag{7.6}
$$

The first suborder (7.5) in the ties considers the most favorable suborder to align the ordering of the unit premiums with the responses, and the second suborder (7.6) is the least favorable one. In absence of ties in $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ , the order statistics (7.5) and (7.6) are identical.

We then map these two orderings to the exposures $( V _ { [ i \downarrow ] } ) _ { i = 1 } ^ { n }$ and $( V _ { [ i \uparrow ] } ) _ { i = 1 } ^ { n }$ , and to the unit loss costs $( Y _ { [ i \downarrow ] } ) _ { i = 1 } ^ { n }$ and $( Y _ { [ i \uparrow ] } ) _ { i = 1 } ^ { n }$ . This gives us the running total sequences for $m \in \{ 1 , \ldots , n \}$

$$
u _ { m } ^ { \downarrow } = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i \downarrow ] } \quad \mathrm { ~ a n d ~ } \quad u _ { m } ^ { \uparrow } = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i \uparrow ] } ,\tag{7.7}
$$

$$
C _ { m } ^ { \downarrow } = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } Y _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i \downarrow ] } Y _ { [ i \downarrow ] } \quad \mathrm { ~ a n d ~ } \quad C _ { m } ^ { \uparrow } = \frac { 1 } { \sum _ { i = 1 } ^ { n } V _ { i } Y _ { i } } \sum _ { i = 1 } ^ { m } V _ { [ i \uparrow ] } Y _ { [ i \uparrow ] } ,\tag{7.8}
$$

and we initialize $u _ { 0 } ^ { \downarrow } = u _ { 0 } ^ { \uparrow } = C _ { 0 } ^ { \downarrow } = C _ { 0 } ^ { \uparrow } = 0$ for $m = 0$ . In absence of ties in the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ , the two constructions in (7.7) and in (7.8) coincide because no subordering is necessary. The main diference between the Leimkuhler curve (7.3) and the CAPs (7.7)-(7.8) is that we compute the running totals in diferent orders. The former is ordered w.r.t. the responses $( Y _ { i } ) _ { i = 1 } ^ { n }$ and the latter w.r.t. the unit premiums $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ . We interpret (7.8) as a concordance measure that assesses how well the order (ranking) of $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ is aligned with the one of $( Y _ { i } ) _ { i = 1 } ^ { n }$ . If they have the same order, we obtain the Leimkuhler curve (7.3), and otherwise (7.8) is dominated by the Leimkuhler curve (7.3). The motivation behind the Gini score precisely is to measure this discrepancy. This consideration is fully rank based – by (7.5) and (7.6) – and it is asymmetric in the treatment of $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and $( Y _ { i } ) _ { i = 1 } ^ { n }$

The cumulative accuracy profiles (CAPs) are obtained by linear interpolation between the points

$$
\Bigl ( u _ { m } ^ { \downarrow } , C _ { m } ^ { \downarrow } \Bigr ) \qquad \mathrm { f o r } m = 0 , \ldots , n ,\tag{7.9}
$$

respectively,

$$
\Bigl ( u _ { m } ^ { \uparrow } , C _ { m } ^ { \uparrow } \Bigr ) \qquad \mathrm { f o r } m = 0 , \ldots , n .\tag{7.10}
$$

Again, these two curves are identical in absence of ties in the unit premiums.

![](images/20522fd11cbe0e96bc370e23f195382a80963ff23776323da68abab979f80fab.jpg)

![](images/d448b75498707ac7b94a7cd448e0adeda269d276ef47bb72404dc203d2c296dc.jpg)  
Figure 17: Gini score plots: (lhs) CAPs of $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ and Leimkuhler curve of $( Y _ { i } ) _ { i = 1 } ^ { n } ;$ and (rhs) with subtracted diagonal for better visualization.

Figure 17 (lhs) shows the CAPs (7.9)-(7.10) of the two premium rules $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ and $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ in red and orange color (the two premium rules do not have ties). We see that the red and orange CAPs are almost indistinguishable in Figure 17 (lhs) and they are dominated by the Leimkuhler curve that considers the perfect ordering. The diagonal line in blue reflects the null model predictor $\mu _ { \mathbb { Q } }$ not considering any covariates. To better visualize the results, we subtract these diagonal values from the Leimkuhler and CAP curves, this yields the plot on the righthand side of Figure 17. We can now see that the CAP of $( m _ { \mathbb { Q } } ( \Pi _ { i } ) ) _ { i = 1 } ^ { n }$ in orange dominates the one of $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ in red, reflecting that there is an issue in the risk ranking of $( \Pi _ { i } ) _ { i = 1 } ^ { n }$ . That is, we correctly conclude from this plot that the recalibrated unit premiums provide the better risk ranking than the original unit premiums. From these plots, it seems that the diferences are comparably small, however, the magnitudes of the diferences in this analysis heavily sufer from the low signal-to-noise ratio in the data, which was already reported in Figure 12.

## 7.3 Gini score

The final step is to map the graphs of Figure 17 to a score. The Gini score is obtained by computing the ratio of the area of the CAP curve enclosed with the diagonal, and the area of the Leimkuhler curve enclosed with the diagonal – the diference is highlighted by Delta in Figure 17. If Delta is zero we have a perfect ordering. This step also requires taking care of the suborders in the ties (7.7)-(7.8), and we simply average over the areas of the most favorable and least favorable suborders. We define from the Leimkuhler curve (7.4) the area

$$
B = \sum _ { m = 1 } ^ { n } \left( u _ { m } - u _ { m - 1 } \right) \frac { L _ { m - 1 } + L _ { m } } { 2 } - 1 / 2 ,
$$

and for the CAPs (7.9)-(7.10) the areas

$$
\begin{array} { r c l } { { A ^ { \downarrow } } } & { { = } } & { { \displaystyle \sum _ { m = 1 } ^ { n } \left( u _ { m } ^ { \downarrow } - u _ { m - 1 } ^ { \downarrow } \right) \frac { C _ { m - 1 } ^ { \downarrow } + C _ { m } ^ { \downarrow } } { 2 } - 1 / 2 , } } \\ { { A ^ { \uparrow } } } & { { = } } & { { \displaystyle \sum _ { m = 1 } ^ { n } \left( u _ { m } ^ { \uparrow } - u _ { m - 1 } ^ { \uparrow } \right) \frac { C _ { m - 1 } ^ { \uparrow } + C _ { m } ^ { \uparrow } } { 2 } - 1 / 2 \quad \leq \quad A ^ { \downarrow } . } } \end{array}
$$

These formulas reflect a linear interpolation between the points in (7.4), (7.9) and (7.10), respectively. The area Delta in Figure 17 is the diference of B and A, i.e., Delta = B − A. Remark that $B > 0$ as soon as we have at least two diferent observations.

The Gini score for the risk ranking of Π for Y on the test data T is defined by

$$
\mathrm { G i n i } ( \mathcal { T } ) = \frac { ( A ^ { \downarrow } + A ^ { \uparrow } ) / 2 } { B } \ \leq \ 1 .\tag{7.11}
$$

The implementation of the computation of the Gini score (7.11) is straightforward and it can be found in the appendix of Brauer–W¨uthrich [7].

Table 2 shows the resulting Gini scores (bigger is better), and we give preference to the recalibrated unit premium $m _ { \mathbb { Q } } ( \Pi )$ for providing the better risk ranking. As already mentioned, these numbers are comparably small (as can also be seen from Figure 17) because we have a low signal-to-noise ratio, but we obtain the correct preference order for the risk ranking.

<table><tr><td>premium rules</td><td>Gini score</td></tr><tr><td>global mean model  $\mu _ { \mathbb { Q } }$ </td><td>0.0000</td></tr><tr><td>premium rule I</td><td>0.0627</td></tr><tr><td>premium rule  $m _ { \mathbb { Q } } ( \Pi )$ </td><td>0.0719</td></tr></table>

Table 2: Gini score assessing the risk rankings.

## 8 French motor third party liability example

The running example above was a rather stylized one. In this section, we consider the popular French motor third party liability (MTPL) claims frequency dataset of Dutang–Charpentier [23] as our second example. We first apply the data cleaning procedure as described in W¨uthrich– Merz [59].<sup>2</sup> The dataset is then partitioned into a learning dataset L consisting of 610,206 insurance policies and an independent training dataset T that contains $n = 6 7$ , 801 insurance policies. We fit two unit premium rules $\Pi ^ { \mathrm { P I N } }$ and $\Pi ^ { \mathrm { G B M } }$ on the same learning dataset ${ \mathcal { L } } ,$ and our goal is to validate these two premium rules on the independent test sample T .

The dataset contains nine covariates, a claim counts response $Z \in \mathbb { N } _ { 0 }$ and a time exposure $V > 0$ in yearly units. For the subsequent considerations, we compute the unit losses $Y = Z / V -$ which are claims frequencies in the case of claim counts Z – and our aim is to find accurate expected claim frequency estimates $\Pi = \Pi ( X )$ which are measurable functions of the covariates X, we also refer to Remarks 2.1. Thus, in this example we consider claim frequencies, but to maintain linguistic consistency with the previous sections, we call Π a unit premium rule. Our numerical analysis considers two unit premium rules (expected claim frequency regression estimates):

1. Pair-wise interaction network (PIN) forecast model $\Pi ^ { \mathrm { P I N } }$ . The PIN model was developed in Richman et al. [53], it is a neural network that specifically models pair-wise interactions of covariates. This model was trained on the learning dataset L mentioned above. We use precisely the PIN parametrization that was obtained in Richman et al. [53, last line of Table 2].<sup>3</sup>

2. As a second competing forecast model, we fit a gradient boosting machine (GBM) on the same learning dataset L. We use the LightGBM version of R with the hyper-parameter specification as given in the appendix, Listing 1. This provides us with a second unit premium rule $\Pi ^ { \mathrm { G B M } }$

Our goal is to validate these two unit premium rules $\Pi ^ { \mathrm { P I N } }$ and $\Pi ^ { \mathrm { G B M } }$ on the independent test sample T . We highlight that both of the two unit premium rules are strong claim frequency forecast models on the French MTPL dataset.

## 8.1 Poisson deviance loss

We compare and validate the two unit premium rules $\Pi ^ { \mathrm { P I N } }$ and $\Pi ^ { \mathrm { G B M } }$ in terms of calibration, discrimination and risk ranking. This is done on the independent test sample $\tau$ that consists of $n = 6 7$ , 801 instances. Since we deal with claim frequencies, it is natural to use the Poisson deviance loss for scoring. The Poisson deviance loss has a strictly convex generator with $\varphi ^ { \prime \prime } ( \theta ) =$ $2 / \theta > 0$ on the positive real line. That is, we have an example from the Patton family (5.21) with parameter $p = 1$ . We select for $x > 0$

$$
\varphi ( x ) = 2 \left( x \log ( x ) - x \right) ,
$$

and we extend it to $x = 0$ by setting $\varphi ( 0 ) = 0$ . This yields first and second derivatives on $\mathbb { R } _ { + }$

$$
\varphi ^ { \prime } ( x ) = 2 \log ( x ) \qquad { \mathrm { a n d } } \qquad \varphi ^ { \prime \prime } ( x ) = 2 / x > 0 .
$$

This gives the Poisson deviance loss

$$
L _ { \varphi } ( y , x ) = 2 \left( x - y + y \log ( y / x ) \right) ,\tag{8.1}
$$

with $L _ { \varphi } ( 0 , x ) = 2 x$ in $y = 0 ~ { \mathrm { ( i . e . } }$ , for insurance policies without claims).

Remark 8.1 (Exposures in Poisson deviance losses) The Poisson deviance loss is special in terms of the positive homogeneity property (5.24), namely, it is homogeneous of order 1. The consequence of this property is that we obtain the identical scoring by either considering the total loss $Z$ (claim counts in our example) under the P-measure or the unit loss costs $Y$ (claim frequency) under the Q-measure because of the identity

$$
\begin{array} { r c l } { { { \cal L } _ { \varphi } ( Z , P ) ~ = ~ { \cal L } _ { \varphi } ( V Y , V \Pi ) } } & { { = } } & { { 2 \left( V \Pi - V Y + V Y \log ( Y / \Pi ) \right) } } \\ { { } } & { { } } & { { } } \\ { { } } & { { = } } & { { 2 V \left( \Pi - Y + Y \log ( Y / \Pi ) \right) = V { \cal L } _ { \varphi } ( Y , \Pi ) . } } \end{array}
$$

Thus, under the Poisson deviance loss, we can either use claim counts Z to validate P under P (left-hand side of the previous identity) or we can use the claim frequency $Y = Z / V$ to validate $\Pi = P / V$ under the Q-measure (on the right-hand side of the previous identity). This sometimes leads to confusion in Poisson model fitting, $\mathrm { e . g . } $ in R there is the family poisson which requires claim counts $Z ,$ and there is the family quasipoisson which allows one to use claim frequencies $Y$ resulting in the same forecast model. This equivalence uses the homogeneity of order 1, and it does not carry over to other Bregman losses.

The quasi-Poisson estimation is also a popular method to ensure the (in-sample) balance property under a log-link generalized linear model (GLM); see Lindholm–W¨uthrich [46, 58]. It can be used for any non-negative loss costs $Y .$ , not necessarily being claims frequencies, because the estimation procedure only relies on strictly consistent scoring with the Poisson deviance loss under the log-link choice, but not on any distributional assumptions.

Table 3 reports the Poisson deviance losses on the test sample $\tau$ of the (constant) global mean model $\widehat { \mu } _ { \mathbb { Q } } = \widehat { \mu } _ { \mathbb { Q } , \mathcal { L } }$ (estimated by the observed frequency on the learning sample $\mathcal { L } )$ , a strong generalized linear model (GLM) taken from W¨uthrich–Merz [59], and the PIN and LightGBM models described above. This allows us to compute the sample versions of the Poisson Bregman scores (6.4). From these sample Bregman scores we conclude that the PIN and the LightGBM are clearly stronger than the selected GLM (2.532), and we give preference to the LightGBM (3.595) over the PIN (3.352) unit premium rule in the terms of the sample Bregman scores of Table 3. Performing a paired non-parametric bootstrap (drawing with replacement from the test sample T ), we obtain a bootstrap standard deviation of 0.057 that quantifies the uncertainty in the diference $3 . 5 9 5 - 3 . 3 5 2 = 0 . 2 4 3$ , thus, the Bregman score improvement is significant.

<table><tr><td>premium rules</td><td>Bregman divergence</td><td>Bregman score</td><td>global mean</td></tr><tr><td>global mean model  $\widehat { \mu } _ { \mathbb { Q } } = \widehat { \mu } _ { \mathbb { Q } , \mathscr { L } }$ </td><td>47.967</td><td></td><td> $\overline { { Y } } _ { \mathbb { Q } , \mathcal { T } } = 7 . 3 5 \%$ </td></tr><tr><td>GLM unit premium</td><td>45.435</td><td>2.532</td><td>7.40%</td></tr><tr><td>PIN unit premium  $\Pi ^ { \mathrm { P I N } }$ </td><td>44.615</td><td>3.352</td><td>7.31%</td></tr><tr><td>LightGBM unit premium  $\Pi ^ { \mathrm { G B M } }$ </td><td>44.372</td><td>3.595</td><td>7.38%</td></tr></table>

Table 3: MTPL Bregman divergences $\widehat { \mathbb { E } } _ { \mathbb { Q } } [ L _ { \varphi } ( Y , \cdot ) ]$ and Bregman scores $\widehat { S } _ { \varphi } ( Y , \cdot )$ of the considered premium rules under the Poisson deviance loss choice on the test sample $\tau { ; }$ units are shown in $1 0 ^ { - 2 }$

The main question that we study below is whether we can find more evidence for this preference and whether it can be understood on a more granular level. Remark that the Bregman score preference of Table 3 is mainly based on discrimination.

The last column of Table 3 shows that exposure-weighted average unit premiums over the entire portfolio (test sample T ), the first line giving the observed empirical frequency $\overline { { Y } } _ { \mathbb { Q } , \mathscr { T } } = 7 . 3 5 \%$ on the test sample T. Thus, the last column of the table shows the sample version of (2.3). The numbers indicate that the PIN slightly underestimates the observed frequency of 7.35% and the LightGBM slightly overestimates this observed frequency. Under a Poisson assumption the magnitude of irreducible risk in this frequency estimate is 0.14%, thus, the deviations from the global observed frequency do not seem to be of a systematic nature.

## 8.2 Actual-vs-expected plots and lift charts

We begin by studying the exposure-weighted distributions of the unit premiums $\Pi _ { i } ^ { \mathrm { P I N } }$ and $\Pi _ { i } ^ { \mathrm { G B M } }$ over the entire test sample $i = 1 , \ldots , n$

To receive expressive graphs, most of the figures are plotted on the log-scale.

Figure 18 (lhs) shows the scatter plot of the LightGBM unit premiums $\Pi _ { i } ^ { \mathrm { G B M } }$ against the PIN unit premiums $\Pi _ { i } ^ { \mathrm { P I N } }$ over the test instances $i = 1 , \ldots , n$ . This scatter plot fluctuates around the orange diagonal line which indicates a large similarity between the two unit premium rules. This is also verified by the exposure-weighted sample distributions $\widehat { G } _ { \Pi ^ { \mathrm { P I N } } } ^ { \mathbb { Q } }$ and $\widehat { G } _ { \Pi ^ { \mathrm { G B M } } } ^ { \mathbb { Q } }$ on the righthand side of that figure. The biggest diferences are observed in the range of bigger predictions, however, from Figure 18 it seems that these diferences are small.

![](images/0f0b0f425584856fabbd4cf52ca8df3f2f44569b7abfccc24a15023c81df8474.jpg)

![](images/35ccda0d54195f1be1a47afb3ae460ad475e99b92c042aa048152196251397fb.jpg)  
Figure 18: MTPL unit premiums: (lhs) scatter plot of $\log ( \Pi ^ { \mathrm { G B M } } )$ against $\log ( \Pi ^ { \mathrm { P I N } } )$ , and (rhs) exposure-weighted sample distributions $\widehat { G } _ { \Pi ^ { \mathrm { P I N } } } ^ { \mathbb { Q } }$ and $\widehat { G } _ { \Pi ^ { \mathrm { G B M } } } ^ { \mathbb { Q } }$

Figure 19 presents the AvE plots and the lift charts of the two unit premium rules $\Pi ^ { \mathrm { P I N } }$ and $\Pi ^ { \mathrm { G B M } }$ . For the binning, we select ventile binning $K = 2 0$ . The lift charts are indicating the following properties:

• Discrimination: Both unit premium rules show a similar lift on the ventile binning scale, see lift charts in the lower panels; for a colored version see Figure 27 in the appendix.

• Calibration: It seems that PIN shows a slightly better calibration picture because the GBM looks more systematically biased over the bins 9 to 14; see also Figure 27 in the appendix.

• Risk ranking: In both cases the actuals are non-monotone, may be a slight preference is given to the GBM version.

In Figure 27 in the appendix, we present the same lift charts, but we add the coloring for the (estimated) miscalibration and resolution on this ventile binning granularity.

Figure 20 gives the AvE plots and the lift charts under isotonic regression binning (4.12). Recall that this uses the unit premiums as risk rankings (risk ordering). A higher number of bins may indicate a better risk ranking because wrongly ordered premiums typically lead to bigger bin sizes by correspondingly averaging over larger premium ranges; for the explanation of the functioning of the PAV algorithm to perform the isotonic regression, we refer to the appendix of W¨uthrich–Ziegel [61]. PIN receives 38 isotonic regression bins and GBM 46 bins; these are not reliable statistics, but they give some indication about risk rankings.

The lift of the two unit premium rules is still similar. We merged the two smallest bins to ensure strict positivity of the isotonic recalibration solution. We did not merge the two largest bins, and it seems that the isotonic recalibration step slightly overfits for the largest value in the PIN version of Figure 20. Calibration seems in both plots similar, maybe on the larger unit premiums the losses are slightly under estimated, and there might be a small preference for PIN in terms of calibration because the alignment of actuals and expected in the lift chart in the lower panel seems slightly better for PIN.

GBM: Lift chart with ventile binning  
PIN: Lift chart with ventile binning  
PIN: AvE plot with ventile binning  
![](images/4961cd1ea543d6f26e02095266569a520b0a41f4976c21f3136abf9be765e99c.jpg)  
GBM: AvE plot with ventile binning

![](images/5445ec4059fbdfaeb8a1605779e1d66b9b86895476a3c175ffb3ba1bfe40c2bd.jpg)

![](images/8040d6a2abfbf1048ca4d778e66834a29b3292a2e041be2dc4e0eb4235a1b0ac.jpg)

![](images/8f5d6bf91df9fe9b3e09737ec92725b53658f5a55a2c0a9fbb5472afd6befa52.jpg)  
Figure 19: (top) MTPL AvE plots with ventile binning $K = 2 0 \colon$ (lhs) PIN unit premium $\Pi ^ { \mathrm { P I N } }$ and (rhs) LightGBM unit premium $\Pi ^ { \mathrm { G B M } }$ ; (bottom) MTPL lift charts with ventile binning $K = 2 0$ : (lhs) PIN unit premium and (rhs) GBM unit premium; the axes are on the log-scale.

Figure 21 provides the double lift chart using ventile binning $K = 2 0 .$ . It considers the ratio of the LightGBM unit premium divided by the PIN unit premium. From the double lift chart we conclude that the LightGBM is clearly more accurate than the PIN on this granularity, this holds for both tails. This suggests better discrimination (and/or risk ranking) by the GBM forecast, and this also supports the numerical results found in Table 3.

## 8.3 Murphy diagram

Next, we aim at understanding how the Poisson deviance losses of Table 3 are composed across the unit premium ranges. We therefore start by studying the Murphy diagram (5.12) showing the sample elementary losses as a function of the threshold θ. In particular, this does not apply the Poisson deviance loss weighting $\varphi ^ { \prime \prime } ( \theta ) = 2 / \theta$ , see (5.19).

PIN: AvE plot with isotonic regression  
![](images/ca1f59478bf393be74777358d7f5ce077713bccf302653817d3c504f64ca025d.jpg)  
GBM: AvE plot with isotonic regression

![](images/4a2eeaec098c3bc05d907d24ebdf26ce73be5a05fdb67a28e5605dac69c34b55.jpg)

PIN: Lift chart with isotonic binning  
![](images/2dc2722b8b13fe821fa613919906d38cae5adf1e42fcc92fe1a0ce5fe530bf93.jpg)

GBM: Lift chart with isotonic binning  
![](images/82ee67799fc30448953cd7cd1304978d1a88532b5e1120e949f16a32486ec753.jpg)  
Figure 20: (top) MTPL AvE plots with isotonic regression: (lhs) PIN unit premium with 38 bins and (rhs) GBM unit premium with 46 bins; (bottom) MTPL lift chart using isotonic regression binning: (lhs) PIN unit premium with 38 bins and (rhs) GBM unit premium with 46 bins.

Figure 22 shows the Murphy diagram, the left-hand side gives the sample elementary losses (5.12) of the two premium rules as a function of θ, and the right-hand side considers their diferences $\Delta _ { \mathbb { Q } } ^ { \mathrm { M u r p h y } } ( \theta ; \Pi ^ { \mathrm { P I N } } , \Pi ^ { \mathrm { G B M } } )$ , subtracting the GBM version from the PIN version, see (5.13). From the right-hand side we conclude that the LightGBM performs better than the PIN for the majority of θ-values, confirming the better Bregman score of Table 3 probably over almost the entire unit premium range. Remark that the Poisson deviance loss integrates over these sample elementary losses using the decreasing weight function $\varphi ^ { \prime \prime } ( \theta ) = 2 / \theta$ , that is, for the Poisson Bregman score diference we consider

![](images/ad36a1b8b9cbc7cd2d66d7e4e2b5e92541fbf55d45c7c2c7b05b84729d36147d.jpg)  
Figure 21: MTPL double lift chart with ventile binning $K = 2 0$ considering GBM unit premiums over PIN unit premiums for κ; the y-axis is on the log-scale.

![](images/64a5e877eac1aa2a5be3991c96c1fdd65fbcf7cf71b826e3fd526d0a6417cf26.jpg)

![](images/2a8c6a99cf7b6e824326a06ae20562f2a8e2b2a10644aaab0619fee05e691890.jpg)  
Figure 22: MTPL sample elementary losses of the PIN and LightGBM unit premiums and their diferences $\Delta _ { \mathbb { Q } } ^ { \mathrm { M u r p h y } } ( \theta ; \overline { { \Pi ^ { \mathrm { P I N } } } } , \Pi ^ { \mathrm { G B M } } )$ as a function of the threshold θ.

$$
\mathrm { P o i s s o n S c o r e D i f f e r e n c e } = \int _ { 0 } ^ { \infty } \Delta _ { \mathbb { Q } } ^ { \mathrm { M u r p h y } } ( \theta ; \Pi ^ { \mathrm { P I N } } , \Pi ^ { \mathrm { G B M } } ) \frac { 2 } { \theta ^ { p } } \ \mathrm { d } \theta \ \qquad \mathrm { w i t h } \ p = 1 .
$$

In view of Figure 22 (rhs), we expect the same preference order within the Patton family (5.21) for any $p \geq 0$ , since these weightings are all monotonically decreasing.

Figure 23 shows the asymmetrically binned Murphy diagrams (top) and the resulting diferences (bottom). We perform percentile binning $K = 1 0 0 \mathrm { w . r . t } .$ . the PIN premium rule on the left-hand side and the GBM premium rule on the right-hand side. From the PIN binning plots we do not learn very much. But from the GBM binning plots on the right-hand side, it seems that the GBM unit premium has a superior performance especially in the larger range. Moreover, the diferences in the two binning versions may indicate that one of the two premium rules provides a better risk ranking, because a diferent exposure-weighting binning is only impacted by diferent risk rankings.

![](images/6342f4cb738358f1f0f72ffd673ff78c698ce39e79c3505310bd49578697259f.jpg)  
Figure 23: MTPL asymmetrically binned Murphy diagrams of the PIN and GBM unit premium rules: (lhs) PIN percentile binning and (rhs) GBM percentile binning.

For the next figure, we only consider the GBM binning version, because from the PIN binning version we cannot learn much. Figure 24 shows that AvE plots with asymmetric GBM percentile binning K = 100 on the log-scale for both axes. The figure shows that the vertical segments seem to be generally larger for the PIN version than the GBM one, giving preference to the GBM. The figure may also indicate that the GBM gives a more accurate risk ranking, this is especially true for the upper tail. However, since the binning is asymmetric, these conclusions may not be fully valid.

![](images/2873592e01c9e9cd4e821790d9e355f1559a0c2306caf6436099ecee76935aa1.jpg)

Actual−vs−expected plot: GBM−binning  
![](images/53c4b82f3a3ad4fce46108d701d27ee6a1284e1c2635219475cb62d8bebc22ad.jpg)  
Figure 24: MTPL AvE plots with GBM percentile binning: upper panel $\overline { { Y } } _ { k } ^ { \mathrm { G B M } } \ \mathrm { v s . } \ \overline { { \Pi } } _ { k } ^ { \mathrm { P I N / G B M } }$ and lower panel $\overline { { Y } } _ { k } ^ { \mathrm { G B M } } \mathrm { v s . } \overline { { \Pi } } _ { k } ^ { \mathrm { G B M } }$ ; since we work on the log-scale, this graph shows $| \log Y - \log \theta |$ instead of $\lvert Y - \theta \rvert$ , but this does not qualitatively change the statements.

## 8.4 Bregman score

In the next step, we lift the elementary loss versions to the Bregman scores. For the Poisson deviance loss, we integrate over the elementary losses with weighting function $\varphi ^ { \prime \prime } ( \theta ) = 2 / \theta$ , see (5.19). In view of Figure 22, it is clear that preference is given to the LightGBM unit premiums, because the biggest diferences in that figure stem from small values of θ. This is confirmed by Table 3.

To better understand the Poisson Bregman scores, we study the paired iterative Bregman score aggregation from small to large unit premiums (6.10)-(6.11). The left panel of Figure 25 considers the PIN ordering and the right panel the GBM ordering, i.e., the two panels aggregate to the same totals, but they difer in the order of aggregation. From these graphs we conclude that the Poisson Bregman scores are very similar for the half of the insurance policies that have lower risks (smaller unit premiums), but the GBM outperforms the PIN especially on the third of the insurance policies with the highest unit premiums. That is, the diference of the Bregman scores of 0.243 in Table 3 (3.352 vs. 3.595) can be explained by these biggest unit premium policies.

![](images/9aeb36b1a667465ad804bc9a9a4be72301bd94f5fedf646d00ff894b640db326.jpg)  
Iterative Bregman score aggregation: Difference

![](images/47b0f85cd2d057a5cc75cdb180a19e221ab27202e2466ddd7e97fdc04a014306.jpg)  
Iterative Bregman score aggregation: Difference

![](images/d46801ef611ac313bd520b7206055e6edbbfd19c763a6df38de26f5e7bb6911f.jpg)

![](images/2969eca46583f1406d612ecb5e654ea863b76c28a364cc2ebede4fbf2071feed.jpg)  
Figure 25: MTPL paired iterative Bregman score aggregation: (lhs) PIN-ordered aggregation and (rhs) GBM-ordered aggregation. The upper panel shows the score aggregation and the lower panel the diferences. The Poisson Bregman scores are 3.352 for PIN (in blue) and 3.595 for GBM (in red), the diference is 0.243 (0.057).

## 8.5 Murphy’s decomposition

In the next step of our analysis we compute Murphy’s decomposition that separates the Bregman score into the resolution term and the miscalibration term, see (6.4). As mentioned above, the main dificulty in this analysis concerns the recalibration step to receive the calibrated versions of $\Pi ^ { \mathrm { P I N } }$ and $\Pi ^ { \mathrm { G B M } }$ . We perform this step by isotonic regression, which is precisely the step illustrated in the upper panel of Figure 20. Using these isotonically recalibrated unit premiums, we compute the two empirical versions (6.12)-(6.13) of Murphy’s decomposition.

<table><tr><td>premium rules</td><td>Bregman score</td><td>MCB  $1 \ \& \ 2$ </td><td>RES 1&amp;2</td></tr><tr><td>PIN premium</td><td>3.352 3.352</td><td>0.063 0.180</td><td>3.415 3.532</td></tr><tr><td>LightGBM premium</td><td>3.595 3.595</td><td>0.059 0.168</td><td>3.654 3.763</td></tr></table>

Table 4: Murphy’s decomposition (6.4) using the two empirical versions (6.12)-(6.13); units are shown in $1 0 ^ { - 2 }$

Table 4 shows the results of Murphy’s decomposition using the two versions (6.12)-(6.13) under isotonic recalibration steps. Similar to the stylized example, version 1 seems to underestimate miscalibration and version 2 seems to overestimate miscalibration. The first is attributed to missing accuracy in the isotonic regression step and the latter to in-sample overfitting (this is our interpretation but there is no proof in absence of the ground truth). Overall the miscalibration terms seem rather similar between PIN and GBM, with very slight preference for GBM, but this interpretation may not be valid because the isotonic regression uses the premium itself for the recalibration step (i.e., contamination resulting from a wrong risk ranking cannot be controlled). These similar magnitudes in miscalibration are then mapped to the resolution terms, and the absolute diference between the GBM and the PIN unit premiums is preserved for the resolution validation giving preference to the LightGBM forecast model.

## 8.6 Risk ranking

The final step is to analyze the risk rankings provided by the unit premiums $\Pi ^ { \mathrm { P I N } }$ and $\Pi ^ { \mathrm { G B M } }$ For this we show the CAP and the Leimkuhler curves and we compute the Gini scores (7.11).

Figure 26 shows the CAPs of the two unit premium rules $( \Pi _ { i } ^ { \mathrm { P I N } } ) _ { i = 1 } ^ { n }$ and $( \Pi _ { i } ^ { \mathrm { G B M } } ) _ { i = 1 } ^ { n }$ . From the plot on the right-hand side, it can be seen that the GBM risk ordering dominates the PIN one, and this is also confirmed by the Gini scores in Table 5. That is, the GBM provides the more accurate risk ranking relative to the observed responses $( Y _ { i } ) _ { i = 1 } ^ { n }$ . To verify statistical significance of the Gini score diference of $0 . 3 6 6 8 - 0 . 3 5 6 8 = 0 . 0 1 0 0$ we performed a paired non-parametric bootstrap (drawing with replacement on the test sample), and the bootstrap standard deviation is 0.0033.

<table><tr><td>premium rules</td><td>Gini score</td></tr><tr><td>global mean model  $\mu _ { \mathbb { Q } }$  premium rule PIN premium rule GBM</td><td>0.0000 0.3568</td></tr></table>

Table 5: MTPL Gini scores assessing the risk rankings of the PIN and the GBM unit premium rankings.

![](images/815174af5189b7458f205ce1f7d778f4f05d040b3d799d46fe339ad88b7fe0da.jpg)

![](images/41a689171322551e7da528b1f60bbf29b9d4eeb0c4d8aab658b1a5393f6818e7.jpg)  
Figure 26: MTPL Gini plots: (lhs) CAPs of PIN unit premiums $( \Pi _ { i } ^ { \mathrm { P I N } } ) _ { i = 1 } ^ { n }$ and GBM unit premiums $( ( \Pi _ { i } ^ { \mathrm { G B M } } ) _ { i = 1 } ^ { n }$ and Leimkuhler curve of $( Y _ { i } ) _ { i = 1 } ^ { n } ;$ and (rhs) with subtracted diagonal for better visualization.

## 9 Summary

The main goal of this manuscript was to present graphical and quantitative model validation tools. We presented these in a proper out-of-sample model validation analysis. We emphasize the following points:

• Actuarial statistical modeling considers exposure-weighted quantities to make losses and premiums comparable across insurance policies with diferent exposures. Exposure-scaled quantities generally still depend on that exposure, i.e., by an exposure scaling one does not get rid of the exposure-dependence in the scaled responses (unit losses). For example, a mean-independence of the scaled response requires further assumptions. Such assumptions are made, for instance, when working within the exponential dispersion family. However, such assumptions may not generally be satisfied in real-world applications.

• To properly account for the dependence between the exposure and the exposure-scaled losses, it is important to work under the exposure-weighted distribution. Otherwise one may obtain systematically biased forecasts. This requires a change of measure from the classical policy-weighted population distribution to the exposure-weighted distribution.

• There are diferent notions of calibration. Generally, the exposure-weighted calibration, given the unit premium, is the correct calibration view in actuarial pricing. It is based on the available information at contract inception. The available information often only includes the unit premium, but not the exposure, because early termination and suspension of contracts makes the exposure only an ex-post available variable. Moreover, this calibration view accounts for the dependence between the unit losses and the exposure, and it controls (mitigates) systematic cross-subsidy between diferent unit premium classes.

• Good forecast models perform well in calibration, discrimination (resolution) and risk ranking:

– Calibration can be assessed by studying calibration plots, actual-vs-predicted plots and lift charts. The sample version of Murphy’s decomposition (based on isotonic regression) gives a quantitative measure of calibration. A critical point of this sample version is that the isotonic regression step uses the test data, and the subsequent analysis is not truly out-of-sample leading to biases. This critical point requires future research.

– Discrimination can be assessed by the (double) lift chart, the Murphy diagram and the paired iterative Bregman score aggregation. The Bregman score gives a quantitative tool to assess the overall predictive accuracy, and the sample version of Murphy’s decomposition gives a discrimination and resolution statistics.

– Risk ranking can be assessed by the cumulative accuracy profile yielding the rankbased Gini score.

• Forecast dominance is usually a too strong condition for model selection. The Murphy diagram based on elementary losses may help us to do an informed model selection because it illustrates how the Bregman score is composed for diferent convex generators. The selection of the specific Bregman score is often done within the Patton family for actuarial pricing problems.

Our outline has been based on graphical tools and point estimates. Naturally, the next step is to derive confidence bounds for these point estimates to perform proper statistical testing. We presented two paired bootstrap analyses for the Bregman score diference and the Gini score diference in the applied motor insurance example. Naturally, much more remains to be done. Many questions and tools are still active research problems, we mention the calibration tests of Denuit et al. [16], Gatti [28] or Delong–W¨uthrich [13]. A critical issue in many of these developments is the missing accuracy in the isotonic regression step, methods that attempt to address this limitation consider, e.g., a boosting step; see Gatti [28]. Similar things can be said for discrimination testing, and for the Gini score we mentioned the asymptotic results of Frees et al. [26] and Brauer et al. [6].

## References

[1] Ayer, M., Brunk, H.D., Ewing, G.M., Reid, W.T., Silverman, E. (1955). An empirical distribution function for sampling with incomplete information. Annals of Mathematical Statistics 26, 641-647.

[2] Bailey, R.A. (1963). Insurance rates with minimum bias. Proceedings of the Casualty Actuarial Society 50, 4-11.

[3] Bailey, R.A., Simon, L.J. (1960). Two studies on automobile insurance ratemaking. ASTIN Bulletin - The Journal of the IAA 1, 192-217.

[4] Barlow, R.E., Bartholomew, D.J., Bremmer, J.M., Brunk, H.D. (1972). Statistical Inference under Order Restrictions. John Wiley & Sons.

[5] Barlow, R.E., Brunk, H.D. (1972). The isotonic regression problem and its dual. Journal of the American Statistical Association 67/337, 140-147.

[6] Brauer, A., Menzel, P., W¨uthrich, M.V. (2025). Model monitoring: A general framework with an application to non-life insurance pricing. arXiv:2510.04556.

[7] Brauer, A., W¨uthrich, M.V. (2026). Gini score under ties and case weights. Variance 19.

[8] Bregman, L.M. (1967). The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming. USSR Computational Mathematics and Mathematical Physics 7/3, 200-217.

[9] Brunk, H.D., Ewing, G.M., Utz, W.R. (1957). Minimizing integrals in certain classes of monotone functions. Pacific Journal of Mathematics 7, 833-847.

[10] B¨uhlmann, H., Straub, E. (1970). Glaubw¨urdigkeit f¨ur Schadens¨atze. Bulletin of the Swiss Association of Actuaries 1970, 111-131.

[11] Delong, L, Gatti, S., W¨uthrich, M.V. (2026). Calibration bands for mean estimates within the exponential dispersion family. Statistical Theory and Related Fields, in press.

[12] Delong, L, W¨uthrich, M.V. (2025). Isotonic regression for variance estimation and its role in mean estimation and model validation. North American Actuarial Journal 29/3, 563-591.

[13] Delong, L, W¨uthrich, M.V. (2025). Universal inference for testing calibration of mean estimates within the exponential dispersion family. arXiv:2510.23821.

[14] Denuit, M., Charpentier, A., Trufin, J. (2021). Autocalibration and Tweedie-dominance for insurance pricing with machine learning. Insurance: Mathematics and Economics 101, 485-497.

[15] Denuit, M., Huyghe, J., Simon, P.-A., Trufin, J. (2026). Tweedie dominance for autocalibrated predictors and Laplace transform order. Scandinavian Actuarial Journal 2026/6, 569-583.

[16] Denuit, M., Huyghe, J., Trufin J., Verdebout, T. (2024). Testing for auto-calibration with Lorenz and concentration curves. Insurance: Mathematics and Economics 117, 130-139.

[17] Denuit, M., Sznajder, D., Trufin, J. (2019). Model selection based on Lorenz and concentration curves, Gini indices and convex order. Insurance: Mathematics and Economics 89, 128-139.

[18] Denuit, M., Trufin, J. (2021). Lorenz curve, Gini coeficient, and Tweedie dominance for autocalibrated predictors. LIDAM Discussion Paper ISBA 2021/36.

[19] Denuit, M., Trufin, J. (2023). Model selection with Pearson’s correlation, concentration and Lorenz curves under autocalibration. European Actuarial Journal 13/2, 871-878.

[20] Denuit, M., Trufin, J. (2024). Convex and Lorenz orders under balance correction in nonlife insurance pricing: Review and new developments. Insurance: Mathematics and Economics 118, 123-128.

[21] Dimitriadis, T., Gneiting, T., Jordan, A.I. (2021). Stable reliability diagrams for probabilistic classifiers. Proceedings of the National Academy of Sciences of the United States of America 118, e2016191118.

[22] Dimitriadis, T., Gneiting, T., Jordan, A.I., Vogel, P. (2024). Evaluating probabilistic classifiers: the triptych. International Journal of Forecasting 40/3, 1101-1122.

[23] Dutang, C., Charpentier, A., (2024). Insurance dataset. Recherche Data Gouv. https://github. com/dutangc/CASdatasets

[24] Ehm, W., Gneiting, T., Jordan, A., Kr¨uger, F. (2016). Of quantiles and expectiles: Consistent scoring functions, Choquet representations, and forecast rankings. Journal of the Royal Statistical Society Series B: Statistical Methodology 78/3, 505-562.

[25] Fissler, T., Lorentzen, C., Mayer, M. (2022). Model comparison and calibration assessment: user guide for consistent scoring functions in machine learning and actuarial practice. arXiv:2202.12780.

[26] Frees, E.W., Meyers, G., Cummings, A.D. (2011). Summarizing insurance scores using a Gini index. Journal of the American Statistical Association 106, 1085-1098.

[27] Frees, E.W., Meyers, G., Cummings, A.D. (2013). Insurance ratemaking and a Gini index. Journal of Risk and Insurance 81, 335-366.

[28] Gatti, S. (2026). Assessing model calibration with boosting trees. arXiv:2606.08084.

[29] Gini, C. (1912). Variabilit\`a e Mutabilit\`a. Contributo allo Studio delle Distribuzioni e delle Relazioni Statistiche. C. Cuppini, Bologna.

[30] Gini, C. (1936). On the measure of concentration with special reference to income and statistics. Colorado College Publication, General Series No. 208, 73-79.

[31] Gneiting, T. (2011). Making and evaluating point forecasts. Journal of the American Statistical Association 106/494, 746-762.

[32] Gneiting, T., Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102/477, 359-378.

[33] Gneiting, T., Resin, J. (2023). Regression diagnostics meets forecast evaluation: conditional calibration, reliability diagrams, and coeficient of determination. Electronic Journal of Statistics 17, 3226-3286.

[34] Goldburd, M., Khare, A., Tevet, D., Guller, D. (2020). Generalized Linear Models for Insurance Rating. 2nd edition. CAS Monograph Series, 5.

[35] Gourieroux, C., Jasiak, J. (2007). The Econometrics of Individual Risk: Credit, Insurance and Marketing. Princeton University Press.

[36] Henzi, A., Puke, M., Dimitriadis, T., Ziegel, J. (2024). A safe Hosmer–Lemeshow test. The New England Journal of Statistics in Data Science 2/2, 175-189.

[37] Hosmer, D.W., Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression model. Communications in Statistics - Theory and Methods 9, 1043-1069.

[38] Jørgensen, B. (1987). Exponential dispersion models. Journal of the Royal Statistical Society, Series B 49/2, 127-145.

[39] Jung, J. (1968). On automobile insurance ratemaking. ASTIN Bulletin - The Journal of the IAA 5, 41-48.

[40] Kr¨uger, F., Ziegel, J. (2021). Generic conditions for forecast dominance. Journal of Business & Economic Statistics 39/4, 972-983.

[41] Kruskal, J.B. (1964). Nonmetric multidimensional scaling. Psychometrica 29, 115-129.

[42] Leeuw, de J., Hornik, K., Mair, P. (2009). Isotone optimization in R: pool-adjacent-violators algorithm (PAVA) and active set methods. Journal of Statistical Software 32/5, 1-24.

[43] Lindholm, M., Lindskog, F., Palmquist, J. (2023). Local bias adjustment, duration-weighted probabilities, and automatic construction of tarif cells. Scandinavian Actuarial Journal 2023/10, 946- 973.

[44] Lindholm, M., Nazar, T. (2024). On duration efects in non-life insurance pricing. European Actuarial Journal 14/3, 809-832.

[45] Lindholm, M., Richman, R., Tsanakas, A., W¨uthrich, M.V. (2022). Discrimination-free insurance pricing. ASTIN Bulletin - The Journal of the IAA 52/1, 55-89.

[46] Lindholm, M., W¨uthrich, M.V. (2026). The balance property in insurance pricing. Scandinavian Actuarial Journal 2026, 506-543.

[47] Loader, C. (1999). Local Regression and Likelihood. Springer.

[48] Lorenz, M.O. (1905). Methods of measuring the concentration of wealth. Publications of the American Statistical Association 9/70, 209-219.

[49] Miles, R.E. (1959). The complete amalgamation into blocks, by weighted means, of a finite set of real numbers. Biometrika 46, 317-327.

[50] Murphy, A.H. (1973). A new vector partition of the probability score. Journal of Applied Meteorology 12/4, 595-600.

[51] Patton, A.J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics 160, 246-256.

[52] Pohle, M.-O. (2020). The Murphy decomposition and the calibration-resolution principle: A new perspective on forecast evaluation. arXiv:2005.01835.

[53] Richman, R., Scognamiglio, S., W¨uthrich, M.V. (2026). Tree-like pairwise interaction networks. Annals of Actuarial Science, in press.

[54] Savage, L.J. (1971). Elicitation of personal probabilities and expectations. Journal of the American Statistical Association 66/336, 783-810.

[55] Semenovich, D., Dolman, C. (2020). What makes a good forecast? Lessons for premium rating from meteorology. All Actuaries Summit 2020.

[56] W¨uthrich, M.V. (2023). Model selection with Gini indices under auto-calibration. European Actuarial Journal 13/1, 469-477.

[57] W¨uthrich, M.V. (2025). Auto-calibration tests for discrete finite regression functions. European Actuarial Journal 15/1, 335-341.

[58] W¨uthrich, M.V. (2026). The balance property: The constrained case, with a view on risk sharing. arXiv:2606.07276.

[59] W¨uthrich, M.V., Merz, M. (2023). Statistical Foundations of Actuarial Learning and its Applications. Springer Actuarial. https://link.springer.com/book/10.1007/978-3-031-12409-9

[60] W¨uthrich, M.V., Richman, R., Avanzi, B., Lindholm, M., Maggi, M., Mayer, M., Schelldorfer, J., and Scognamiglio, S. (2025). AI tools for actuaries. SSRN Manuscript, ID 5162304. https: //aitools4actuaries.com/

[61] W¨uthrich, M.V., Ziegel, J. (2024). Isotonic recalibration under a low signal-to-noise ratio. Scandinavian Actuarial Journal 2024/3, 279-299.

## A Mathematical proofs

Proof of Lemma 6.1. Using A-measurability and the functional form of the Bregman loss

$$
\operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , X ) \mid { \mathcal { A } } \right] = \operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ \varphi ( Y ) \mid { \mathcal { A } } \right] - \varphi ( X ) - \varphi ^ { \prime } ( X ) ( m _ { \mathbb { Q } } ( A ) - X ) .
$$

The last term vanishes for $X = m _ { \mathbb { Q } } ( { \mathcal { A } } )$ which gives us

$$
\operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , m _ { \mathbb { Q } } ( A ) ) ~ | ~ A \right] = \operatorname { \mathbb { E } } _ { \mathbb { Q } } \left[ \varphi ( Y ) ~ | ~ A \right] - \varphi ( m _ { \mathbb { Q } } ( A ) ) .
$$

Subtracting the two identities proves the claim.

Proof of Corollary 6.2. We take the expectation over (6.2) providing

$$
\begin{array} { r c l } { \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , \Pi ) \right] } & { = } & { \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , m _ { \mathbb { Q } } ( \Pi ) ) \right] + \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \Pi ) \right] } \\ & { = } & { \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) \right] + \left( \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , m _ { \mathbb { Q } } ( \Pi ) ) \right] - \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( Y , \mu _ { \mathbb { Q } } ) \right] \right) + \mathbb { E } _ { \mathbb { Q } } \left[ L _ { \varphi } ( m _ { \mathbb { Q } } ( \Pi ) , \Pi ) \right] . } \end{array}
$$

The round bracket can again be evaluated with (6.1) by selecting $X = \mu _ { \mathbb { Q } }$ . This proves the claim.

## B LightGBM code

Listing 1: LightGBM hyper-parameter specification.

1 library ( lightgbm )   
2   
3 # Data interface of LightGBM   
4 dtrain <- lgb. Dataset (   
5 X\_train ,   
6 label = y\_train ,   
7 weight = learn\$Exposure ,   
8 params = list ( feature\_pre\_filter = FALSE )   
9 )   
10   
11 # Parameters   
12 params <- list (   
13 learning\_rate = 0.05 ,   
14 objective = " poisson ",   
15 metric = " poisson " ,   
16 num\_leaves = 7,   
17 min\_data\_in\_leaf = 50,   
18 min\_sum\_hessian\_in\_leaf = 0.001 ,   
19 colsample\_bynode = 0.8 ,   
20 bagging\_fraction = 0.8 ,   
21 lambda\_l1 = 3,   
22 lambda\_l2 = 5,   
23 num\_threads = 7   
24 )   
25   
26 fit\_lgb <- lgb . train ( params = params , data = dtrain , nrounds = 3000)

## C Miscalibration and resolution plots

![](images/f85b9d4e4620781dce4b70c9b8f49ad8cd5669e71ef3e8c0caeb038ec396436f.jpg)

![](images/450ec7065e5d9c57febd2fc3d86d57ac9a78b094c8de3b6d67579581a3139986.jpg)

![](images/5a6ad5c560e5868b99b73b3893ea5b7aa97ce0fc6734957c93a890874740c58d.jpg)

![](images/e5a12b992cf44a58d67a330ba452e1d69577fca59fd6a6fda4b8835ab7c3a347.jpg)  
Figure 27: MTPL lift charts of Figure 19, revisited: visualizations of (top) estimated miscalibration of PIN and GBM, (bottom) estimated resolution of PIN and GBM using ventile binning $K = 2 0$ and the bin averages $\overline { { Y } } _ { k }$ for the recalibration estimate ${ \widehat { m } } _ { \mathbb { Q } } ( B )$ ， $B = k$ , we also refer to Figure 16; these graphs consider the raw diferences before entering the Bregman score.