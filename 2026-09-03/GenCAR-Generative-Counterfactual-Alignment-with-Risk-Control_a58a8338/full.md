# GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation

Qianqian Wang<sup>a,b</sup>, Yunshan Li<sup>a</sup>, Jiawen Zeng<sup>c</sup>, Wenwu Gong<sup>a</sup> and Lili Yang<sup>a,b,∗</sup>

<sup>a</sup>Shenzhen Key Laboratory ofSafety and Securityfor Next Generation ofIndustrial Internet, Southern University ofScience and Technology, Shenzhen, China

<sup>b</sup>Department of Statistics and Data Science, Southern University of Science and Technology, Shenzhen, China

<sup>c</sup>School ofEngineering and Applied Science, University ofPennsylvania, Philadelphia, USA

## A R T I C L E I N F O

Keywords:   
Out-of-distribution recommendation   
counterfactual reasoning   
large language models   
conformal prediction   
selective inference

## A BS T RA C T

Serving useful recommendations under distribution shift is crucial for balancing utility and risk in out-of-distribution (OOD) recommendation. However, most existing OOD methods improve ranking or construct counterfactual candidates without controlling the proxy-label false discovery rate (FDR) of the served set. In this work, we formulate OOD serving as the �-Valid Counterfactual Recommendation (�-VCR) problem to retain candidate support learned from counterfactual supervision while controlling proxy-label FDR, and propose GenCAR, which couples preference-grounded counterfactual supervision with calibrated set selection. In particular, GenCAR fixes the stable-preference representation while intervening on the environmental factor, grounds ofline large language model proposals through preference anchors and trust-radius filtering, and uses conformal �-values for Benjamini–Hochberg selection. We theoretically bound conditional counterfactual approximation error and prove finite-sample, distribution-free control of proxy-label FDR under exchangeability and positive regression dependence, with a Benjamini–Yekutieli guarantee under arbitrary dependence. Extensive experiments audit realized proxy false discovery proportions and demonstrate that GenCAR consistently enhances OOD candidate recovery across diverse benchmarks.

## 1. Introduction

Out-of-distribution (OOD) recommender systems serve users across changing environments, where interaction distributions difer between training and serving [1, 2]. Due to changes in exposure, item popularity, and temporal context, ranking patterns learned from logged interactions may not transfer reliably to the serving environment [1, 3, 4]. In such settings, a naive strategy is to enlarge the top-ranked set served to each user, increasing the chance that it contains useful post-shift items. While this strategy can retain more useful items, it can also admit low-quality candidates and lead to recommendation failures as the set grows [5]. Thus, it is essential to determine which and how many candidates to serve for each user, which highlights the utility-risk trade-of in OOD recommendation.

Recent research has proposed OOD recommenders that separate preference-related signals from environmental factors to improve ranking after distribution shift [6, 3, 7]. Distributionally robust objectives and causal difusion extend this direction to broader forms of shift [4, 2, 8]. Yet, these methods are learned from interactions logged in the training environment, which do not directly reveal which items remain useful in the serving environment. To obtain supervision for this unobserved setting, a natural strategy is to construct item-level counterfactuals under an environmental intervention. Generative recommenders provide a practical mechanism for constructing such supervision by modeling interaction distributions, generating oracle-like items, or augmenting training sequences [9–13]. However, their objectives optimize reconstruction or personalized ranking without specifying how stable preference should constrain generation under the intervention. Generated candidates can retain environment-specific patterns and produce recommendation failures after the shift.

Counterfactual generation defines the candidate family, but serving still requires deciding which candidates to retain. Conformal recommendation controls the false discovery rate (FDR), the expected proportion of incorrect items among selected recommendations, for sets returned by a given ranker [5]. Recent work further controls unwanted recommendations and expands selected sets using observed user feedback [14]. These guarantees assume a fixed candidate family and predefined relevance or unwanted-content labels. In our setting, candidates are generated under an environmental intervention, and their alignment with stable preference is represented by proxy labels. Existing methods do not jointly construct such candidates and control their proxy-label FDR during serving. This limitation raises a pivotal issue:

## How can we retain counterfactual candidates while controlling proxy-label FDR in the served set?

In this work, we propose Generative Counterfactual Alignment with Risk-Controlled Selection (GenCAR), a two-stage framework that connects preference-grounded counterfactual supervision with calibrated set selection. We define the serving risk as the pooled FDR of candidate pairs labeled as misaligned by a proxy obtained from an ofline large language model (LLM). We formulate OOD serving as the �-Valid Counterfactual Recommendation (�-VCR) problem (Definition 2), which seeks to retain counterfactual candidate support while keeping proxy-label FDR below a user-specified level �. Overall, GenCAR is preference-grounded, risk-controlled, and LLM-free during online serving, as it transforms ofline LLM supervision into �-valid served sets through stable-preference constraints and conformal calibration.

Theoretically, we establish finite-sample, distributionfree control of pooled proxy-label FDR under targetenvironment exchangeability and positive regression dependence on the proxy nulls (Theorem 1). A Benjamini– Yekutieli (BY) variant extends this guarantee to arbitrary dependence (Corollary 1). We also prove that Benjamini– Hochberg (BH) returns the largest member of its nested step-up family (Lemma 1), showing that calibrated selection retains the largest admissible set within this family. A proxy-transfer bound isolates the error required to relate the proxy-label certificate to causal misalignment (Corollary 2). For counterfactual construction, we bound the conditional approximation error introduced by anchor projection, LLM realization, and trust-radius filtering (Proposition 1).

To evaluate the validity and efectiveness of GenCAR, we conduct studies on MovieLens (ML)-100K [15], Coat [1], and Amazon-Book [16], together with two additional calibration datasets. The results show that realized proxy FDP remains below � on all five evaluated candidate pools and that GenCAR consistently improves OOD candidate recovery. Specifically, compared with the shared causal variational autoencoder (CausalVAE) backbone, GenCAR improves Recall@10 by 11.0%, 19.1%, and 43.5% on the three primary benchmarks. Moreover, GenCAR achieves the highest mean Recall@10 among methods with LLM-free serving on all three benchmarks. Matched ablations show that additional training pairs alone do not explain these gains. On Coat, GenCAR retains sub-millisecond online latency, while inference-time LLM reranking requires approximately 5.4 seconds per user.

Our contributions are summarized as follows:

• We formulate OOD serving as the �-VCR problem, establishing a principled framework to retain counterfactual candidate support while controlling pooled proxy-label FDR below a user-specified level �.

• We propose GenCAR, a two-stage framework that transforms ofline LLM proposals into preferencegrounded counterfactual supervision and calibrated served sets. GenCAR supports variable-size and empty outputs and remains LLM-free during online serving.

• We derive a conditional approximation bound for counterfactual construction and establish finitesample pooled proxy-label FDR control for calibrated selection. We further establish BH-family maximality and quantify the proxy error separating this certificate from causal misalignment.

![](images/c74d37b44856accd84f9950b6c4fc88b8a6af6b4435d2b8691f16454412cd8a3.jpg)  
Fig. 1. Structural causal models for OOD recommendation. (a) A naive predictor entangles user preference with the observed environment. (b) A disentangled model separates the latent factors but does not preserve preference across environments. (c) The target SCM changes the environmental factor $\mathbf { z } _ { e }$ while keeping stable preference $\mathbf { z } _ { c }$ fixed.

## 2. Background

In this section, we introduce OOD recommendation under environmental shift, discuss the misalignment risk associated with serving post-shift candidates, and formally define the �-Valid Counterfactual Recommendation (�-VCR) problem.

## 2.1. Preliminaries

Let U and I denote the user and item spaces. We use $( u , j )$ for a fixed user–item pair and $( U , J )$ for a random pair. For each user $u \in \mathcal V$ , let $x _ { u }$ denote the observed interaction history. The training log $D ^ { \mathrm { t r a i n } } \ = \ \{ ( u , j ) \ : \ y _ { u j } \ = \ 1 \}$ is sampled from $P _ { \mathrm { t r a i n } } ,$ , while serving pairs follow $P _ { \mathrm { t e s t } }$ . We consider an OOD setting in which $P _ { \mathrm { t r a i n } } ~ \ne ~ P _ { \mathrm { t e s t } }$ because the exposure policy, item popularity, or temporal context changes between training and serving [1, 3, 2].

To determine which items remain suitable after an environmental change, we model each interaction with a structural causal model (SCM). The latent state contains stable user preference $\mathbf { z } _ { c } ,$ an environmental factor $\mathbf { z } _ { e } ,$ and idiosyncratic noise �. This factorization separates preference-related information from environment-dependent variation [17, 18]. Fig. 1 contrasts the resulting OOD recommendation problem with a naive predictor and a disentangled model without cross-environment invariance.

Assumption 1 (SCM decomposition). For each user–item pair (�, �), the interaction is generated according to

$$
\mathbf { z } _ { c } \sim p ( \mathbf { z } _ { c } ) , \ \mathbf { z } _ { e } \sim p _ { e } ( \mathbf { z } _ { e } ) , \ \eta \sim p ( \eta ) , \ j \sim p ( j \mid \mathbf { z } _ { c } , \mathbf { z } _ { e } , \eta ) .
$$

Here, $\mathbf { z } _ { c }$ represents stable user preference, $\mathbf { z } _ { e }$ captures the current environment, and � accounts for idiosyncratic variation.

Assumption 2 (Environmental shift). For each environment �, the joint distribution factorizes as

$$
P _ { e } ( \mathbf { z } _ { c } , \mathbf { z } _ { e } , \eta , j ) = p ( \mathbf { z } _ { c } ) p ( \eta ) p _ { e } ( \mathbf { z } _ { e } ) p ( j \mid \mathbf { z } _ { c } , \mathbf { z } _ { e } , \eta ) .
$$

Across training and serving, $p ( \mathbf { z } _ { c } ) , p ( \eta )$ , and the interaction mechanism $p ( j | \mathrm { ~ \bf ~ \underline { ~ } { ~ z ~ } ~ } , \mathbf { z } _ { e } , \eta )$ remain invariant [19], while $p _ { e } ( { \bf z } _ { e } )$ may change.

The environmental-shift assumption localizes the distribution change to $\mathbf { z } _ { e } .$ . For a target environment $\tilde { \mathbf { z } _ { e } } ,$ the intervention do $( \mathbf { z } _ { e } : = \tilde { \mathbf { z } _ { e } } )$ changes the environment while preserving the stable preference of user �. This intervention defines whether an item remains aligned with that user after the shift.

Definition 1 (Causal alignment target). For a user–itempair (�, �), define

$$
G _ { u j } = \mathbb { 1 } \left[ p \big ( j | \mathbf { z } _ { c } ^ { ( u ) } , \mathrm { d o } ( \mathbf { z } _ { e } : = \tilde { \mathbf { z } } _ { e } ) , \eta ^ { ( u ) } \big ) \geq \gamma _ { u } \right] ,
$$

where $\gamma _ { u }$ is a user-specific alignment threshold. The corresponding aligned item set is $\mathcal { G } _ { \tilde { \mathbf { z } _ { o } } } ( u ) = \{ j \in I : G _ { u j } = 1 \}$ The set $\mathcal { G } _ { \tilde { \mathbf { z } } _ { e } } ( u )$ may be empty when no candidate item is aligned with user � in the target environment.

OOD recommendation. A standard OOD recommender relies on a scoring function: $s _ { \theta } : \mathcal { V } \times \mathcal { I } \to \mathbb { R }$ , where $s _ { \theta } ( u , j )$ represents the predicted relevance of item � to user � under the target environment. We write $c _ { u }$ for the top-� ranked items of user � and $\mathcal { Q } ( B ) = \{ ( u , j ) : u \in B , j \in C _ { u } \}$ for the pooled candidate family of a serving batch .

Existing OOD recommenders improve $s _ { \theta }$ by separating preference-related signals from environmental factors or by optimizing performance under distribution shift [6, 3, 7, 2]. Generative recommenders extend this direction by constructing additional supervision or candidate support beyond observed interactions [9–12]. These methods can recover useful items that are absent from the training environment. While efective for candidate recovery, the ranking score does not determine whether a candidate belongs to $\mathcal { G } _ { \tilde { \mathbf { z } } _ { e } } ( u ) . \mathrm { A }$ natural serving strategy is to retain the highest-ranked candidates. Fixed set sizes and uncalibrated thresholds, however, provide no control over the proportion of retained candidates that are misaligned with the target preference. Retaining more candidates can recover additional post-shift support and can also admit more misaligned recommendations. To this end, we formalize OOD serving as pooled risk control over Q(B).

## 2.2. Problem Formulation

Given a ranked candidate family (), we construct a set-valued selection rule Γ that returns a served set as ${ \mathcal { S } } _ { \Gamma } =$ Γ(()). We characterize Γ along two dimensions: proxylabel risk and retained candidate support. The causal null set contains candidate pairs that are not aligned with the target preference:

$$
\mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } = \{ ( u , j ) \in \mathcal { Q } ( \boldsymbol { B } ) : G _ { u j } = 0 \} .
$$

The indicator $G _ { u j }$ is not observed during serving. We therefore construct an operational null set using the LLM-derived alignment proxy:

$\mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } = \{ ( u , j ) \in \mathcal { Q } ( \mathcal { B } ) : ( u , j )$ is labeled as misaligned by the proxy}.

For a served set , define its proxy-label false discovery proportion (FDP) as

$$
\mathrm { F D P } _ { \operatorname { p r o x y } } ( S ) = \frac { \lvert S \cap \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } \rvert } { \operatorname* { m a x } \{ \lvert S \rvert , 1 \} } .
$$

The corresponding pooled false discovery rate is

$$
\mathrm { F D R } _ { \mathrm { p r o x y } } ( S ) = \mathbb { E } \left[ \mathrm { F D P } _ { \mathrm { p r o x y } } ( S ) \right] ,
$$

where the expectation is taken over the target candidate batch and any randomness in the selection rule. The retained candidate support is measured by �[||].

We formulate the goal of retaining candidate support while controlling proxy-label risk as the �-Valid Counterfactual Recommendation (�-VCR) problem.

Definition 2 (�-Valid Counterfactual Recommendation). Given a risk level $\alpha \in ( 0 , 1 )$ , the goal of �-VCR is to find a selection rule $\Gamma _ { \alpha }$ whose served set satisfies the pooled proxy-label FDR constraint:

$$
\mathrm { F D R } _ { \mathrm { p r o x y } } \big ( S _ { \alpha } ( B ) \big ) \leq \alpha .\tag{1}
$$

where $S _ { \alpha } ( B ) = \Gamma _ { \alpha } ( { \mathcal Q } ( B ) )$ . Among valid rules, larger $\mathbb { E } [ | S _ { \alpha } ( B ) | ]$ means that more candidate support is retained.

By bounding the expected proportion of proxymisaligned pairs, �-VCR prevents candidate expansion from relying on uncontrolled misaligned recommendations. The expected served-set size records how much post-shift candidate support remains under this risk constraint.

In practice, the target proxy-null distribution is unknown, so the �-VCR constraint cannot be evaluated exactly. A datadriven solution requires a finite calibration set.

Remark 1 (Interpretation of validity). A rule is �-valid when its pooled proxy-label FDR does not exceed �. An empty set has zero FDP and represents abstention. Corollary 2 isolates the additional error needed to transfer this proxy certificate to causal misalignment.

## 3. GenCAR

In this section, we present Generative Counterfactual Alignment with Risk-Controlled Selection (GenCAR), a two-stage framework for solving the �-VCR problem (Definition 2). GenCAR first constructs preference-grounded counterfactual supervision under a specified environmental intervention. The accepted counterfactuals train a ranker that forms the post-shift candidate family (). GenCAR next calibrates evidence against proxy misalignment and selects a variable-size served set satisfying the proxy-label FDR constraint in Eq. (1). All LLM calls and proxy labeling occur ofline, leaving online serving to the learned ranker and stored calibration artifacts.

![](images/e79520cd547a613beabd84aec97eb38ad12959956e1b69bf5d4184f90072617c.jpg)  
Fig. 2. Overview of GenCAR. GenCAR operates in ofline and online phases. Ofline construction and calibration (left). The backbone separates stable preference from the environmental factor. GenCAR fixes stable preference, intervenes on the environment, and grounds ofline LLM proposals through preference anchors and trust-radius filtering. The accepted items finetune the ranker. A disjoint proxy-null split supplies the calibration scores used to compute conformal evidence. Risk-controlled serving (right). GenCAR forms a ranked candidate family without an LLM call and applies BH or BY at level �. The resulting pooled served set is projected onto user-level lists, with an empty list representing abstention.

As illustrated in Figure 2, GenCAR contains three modules. Preference-grounded counterfactual construction fixes the stable-preference representation, intervenes on the environmental factor, and converts ofline LLM proposals into training supervision. Proxy-label risk calibration fits an alignment predictor and converts a disjoint proxy-null calibration set into conformal evidence. Risk-controlled serving applies BH or BY to the ranked candidate family and returns a pooled served set together with its user-level lists. The first two modules operate ofline, while online serving uses only the stored model and calibration artifacts.

## 3.1. Preference-Grounded Counterfactual Construction

The ranked candidate family () must contain items suitable for the target environment, yet these post-shift interactions are absent from the training log. We construct item-level supervision by changing the environmental factor while preserving the learned stable-preference representation. Given an interaction vector $x _ { u } \in \{ 0 , 1 \} ^ { | T | }$ , the encoder $q _ { \phi }$ produces the factorized posterior

$$
q _ { \phi } ( \mathbf { z } _ { c } , \mathbf { z } _ { e } , \eta \mid x _ { u } ) = q _ { \phi } ( \mathbf { z } _ { c } \mid x _ { u } ) q _ { \phi } ( \mathbf { z } _ { e } \mid x _ { u } ) q _ { \phi } ( \eta \mid x _ { u } ) .
$$

The three factors correspond to the SCM in Assumption 1. The stable-preference representation $\mathbf { z } _ { c }$ determines ranking, while $\mathbf { z } _ { e }$ records environment-dependent variation and � captures residual variation.

Preference–environment factorization. The backbone combines reconstruction, Bayesian personalized ranking (BPR), and factor separation:

$$
{ \mathcal { L } } _ { \mathrm { b a c k b o n e } } = { \mathcal { L } } _ { \mathrm { E L B O } } + \lambda _ { \mathrm { B P R } } { \mathcal { L } } _ { \mathrm { B P R } } + \lambda _ { \mathrm { d i s e n t } } { \mathcal { L } } _ { \mathrm { d i s e n t } } .
$$

Here, $\mathcal { L } _ { \mathrm { E L B O } }$ denotes the negative of the evidence lower bound (ELBO) and $\lambda _ { \mathrm { B P R } } , \lambda _ { \mathrm { d i s e n t } } \ge 0$ . Let $\tau _ { \mathrm { { o b s } } }$ denote the observed training triplets, where $j ^ { + }$ is interacted and $j ^ { - }$ is unobserved for user �.

The ranker scores item � for user � by $\hat { r } _ { u j } = \left. \mathbf { z } _ { c } ^ { ( u ) } , e _ { j } \right.$ and optimizes the BPR objective [20]

$$
\mathcal { L } _ { \mathrm { B P R } } = - \sum _ { ( u , j ^ { + } , j ^ { - } ) \in \mathcal { T } _ { \mathrm { o b s } } } \log \sigma \big ( \hat { r } _ { u j ^ { + } } - \hat { r } _ { u j ^ { - } } \big ) .
$$

The reconstruction and ranking terms preserve predictive information, while ${ \mathcal { L } } _ { \mathrm { d i s e n t } }$ separates $\mathbf { z } _ { c }$ from $\mathbf { z } _ { e }$ . Only $\mathbf { z } _ { c }$ enters the ranking score $\hat { r } _ { u j }$ . This design makes the environmental intervention operational without redefining the learned preference coordinate.

Abduction and environmental intervention. We construct counterfactual supervision through abduction, action, and prediction [21]. For each user �, the encoder infers $( \hat { \mathbf { z } _ { c } } ^ { ( u ) } , \hat { \mathbf { z } _ { e } } ^ { ( u ) } , \hat { \eta } ^ { ( u ) } )$ from $x _ { u } .$ We replace $\hat { \mathbf { z } _ { e } } ^ { ( u ) }$ with the target factor $\hat { \mathbf { z } _ { e } }$ while keeping $\hat { \pmb { z } } _ { c } ^ { ( u ) }$ and $\hat { \eta } ^ { ( u ) }$ fixed. The resulting counterfactual target is

$$
p \Big ( j | \hat { \mathbf { z } _ { c } } ^ { ( u ) } , \mathrm { d o } ( \mathbf { z } _ { e } : = \hat { \mathbf { z } _ { e } } ) , \hat { \eta } ^ { ( u ) } \Big ) .
$$

This distribution describes item suitability in the target environment for the preference state inferred from the observed user history.

Anchor projection and LLM realization. The latent vector $\hat { \pmb { z } } _ { c } ^ { ( u ) }$ cannot be supplied directly to an LLM. We map this vector to the catalog items with the highest preference scores:

$$
\mathcal { A } _ { u } = T ( \hat { \mathbf { z } _ { c } } ^ { ( u ) } ) = \mathrm { T o p K } _ { j \in T } \left. \hat { \mathbf { z } _ { c } } ^ { ( u ) } , \boldsymbol { e } _ { j } \right. .
$$

The item descriptions in $\mathcal { A } _ { u }$ provide a discrete representation of the learned preference neighborhood. Gen-CAR combines these anchors with the target-environment descriptor in the ofline prompt. The LLM returns catalog item identifiers $\widehat { \mathcal { V } } _ { u }$ and an alignment score $a _ { u j } ^ { \mathrm { L L M } } \in [ 0 , 1 ]$ for each proposal.

Trust-radius filtering. Let $E _ { 0 } = \{ e _ { i } ^ { 0 } \} _ { j \in \cal { I } }$ be the backbone item embeddings used for anchor projection and filtering. Invalid catalog identifiers are removed, and each remaining proposal receives the distance $d _ { 0 } ( \tilde { j } , \mathbf { z } _ { c } ^ { ( u ) } ) = 1 - \cos ( e _ { \tilde { i } } ^ { 0 } , \mathbf { z } _ { c } ^ { ( u ) } )$ The accepted counterfactual set is

$$
\mathcal { V } _ { u } ^ { \mathrm { c f } } = \left\{ \widetilde { j } \in \widehat { \mathcal { I } } _ { u } \cap \cal T : d _ { 0 } ( \widetilde { j } , \mathbf { z } _ { c } ^ { ( u ) } ) \leq \delta \right\} .\tag{2}
$$

Equation (2) leaves one ofline supervision object: catalog-valid proposals inside the preference trust radius. Counterfactual fine-tuning. The accepted proposals provide item-level supervision for the target environment. Let $\mathcal { V } _ { \mathrm { c f } }$ denote the users with accepted proposals. For each $u \in \mathcal { V } _ { \mathrm { c f } }$ with accepted items, $\Pi _ { u } ^ { \mathrm { c f } }$ pairs $j ^ { \bar { + } } \in \mathcal { V } _ { u } ^ { \mathrm { c f } }$ with an item $j ^ { - }$ unobserved for that user. We keep the encoder fixed and update only the item embeddings with the observed and counterfactual BPR losses. GenCAR minimizes

$$
\mathcal { L } _ { \mathrm { C F } } ( E ) = - \sum _ { u \in \mathcal { V } _ { \mathrm { c f } } } \mathbb { E } _ { ( j _ { + } , j _ { - } ) \sim \Pi _ { u } ^ { \mathrm { c f } } } \left[ \log \sigma \Big ( \left. \hat { \mathbf { z } } _ { c } ^ { ( u ) } , e _ { j _ { + } } \right. - \left. \hat { \mathbf { z } } _ { c } ^ { ( u ) } , e _ { j _ { - } } \right. \Big ) \right] .\tag{3}
$$

The counterfactual update is defined by $\widehat { E } _ { \mathrm { c f } } \quad \in$ arg min $_ E ^ { \mathcal { L } } \mathrm { _ { C F } } ( E )$ . The encoded preference representations $\hat { \pmb { z } _ { c } } ^ { ( u ) }$ remain fixed, and � is the optimized parameter block in Eq. (3). Below, $e _ { j }$ denotes the corresponding updated embedding in $\widehat { E } _ { \mathrm { c f } }$

Counterfactual fine-tuning changes item placement while preserving the preference representation used to construct the intervention. The updated ranker forms the candidate family (). LLM proposals provide ofline supervision and are never returned directly during serving.

The accepted set $\mathcal { V } _ { u } ^ { \mathrm { c f } }$ supplies ranking supervision, whereas the proxy label introduced in Section 3.2 defines the operational null for risk-controlled selection.

Counterfactual approximation path. The intervention target is not observed directly. We approximate it through anchor projection and LLM realization. For a fixed user and target environment, let $P ^ { \star } , P ^ { A }$ , and $P ^ { M }$ denote the target, anchor-projected, and LLM-realized distributions:

$$
\begin{array} { r l } & { P ^ { \star } ( j ) = p \bigg ( j \mid \hat { \mathbf { z } _ { c } } ^ { ( u ) } , \mathrm { d o } ( \mathbf { z } _ { e } : = \hat { \mathbf { z } _ { e } } ) , \hat { \eta } ^ { ( u ) } \bigg ) , } \\ & { P ^ { A } ( j ) = P _ { M } \bigg ( j \mid T ( \hat { \mathbf { z } _ { c } } ^ { ( u ) } ) , \mathrm { d o } ( \mathbf { z } _ { e } : = \hat { \mathbf { z } _ { e } } ) \bigg ) , } \\ & { P ^ { M } ( j ) = P _ { M } \bigg ( j \mid \mathrm { p r o m p t } ( T ( \hat { \mathbf { z } _ { c } } ^ { ( u ) } ) , \hat { \mathbf { z } _ { e } } ) \bigg ) . } \end{array}\tag{4}
$$

These distributions isolate the approximation introduced by anchor projection and LLM realization. Proposition 1 bounds their conditional discrepancy after trust-radius filtering.

## 3.2. Proxy-Label Risk Calibration

Counterfactual fine-tuning determines the ranked candidate family $\mathcal { Q } ( B )$ , while �-VCR requires a separate decision about which pairs enter the served set. The target proxynull score distribution is unknown, so its FDR constraint cannot be evaluated directly. GenCAR uses a finite, disjoint calibration set to construct conformal evidence against proxy misalignment. This stage returns the pooled served set $S _ { \alpha } ( B )$

Proxy alignment predictor. The ofline LLM alignment score defines the proxy label

$$
Y _ { u j } ^ { \mathrm { p r o x y } } = \mathbb { 1 } \left[ a _ { u j } ^ { \mathrm { L L M } } \geq \tau \right] .\tag{5}
$$

Pairs with $Y _ { u j } ^ { \mathrm { p r o x y } } = 0$ instantiate $\mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } }$ in Definition 2. This operational proxy label remains distinct from the unobserved causal indicator $G _ { u j }$ . GenCAR trains

$$
\hat { h } : ( \hat { { \mathbf { z } _ { c } } } ^ { ( u ) } , \hat { { \mathbf { z } _ { e } } } ^ { ( u ) } , e _ { j } ) \longmapsto [ 0 , 1 ]
$$

to estimate proxy alignment. For brevity, we write $\begin{array} { r l r } { \hat { h } ( u , j ) } & { { } : = } & { \hat { h } ( \hat { \textbf { z } _ { c } } ^ { ( u ) } , \hat { \textbf { z } _ { e } } ^ { ( u ) } , e _ { j } ) } \end{array}$ . The predictor minimizes binary cross-entropy on the alignment split:

$$
\begin{array} { r l } & { \mathcal { L } _ { h } = - \displaystyle \frac { 1 } { | D _ { \mathrm { a l i g n } } | } \sum _ { ( u , j ) \in D _ { \mathrm { a l i g n } } } \left[ Y _ { u j } ^ { \mathrm { p r o x y } } \log \hat { h } ( u , j ) \right. } \\ & { \quad \quad \left. + ( 1 - Y _ { u j } ^ { \mathrm { p r o x y } } ) \log ( 1 - \hat { h } ( u , j ) ) \right] . } \end{array}\tag{6}
$$

The predictor $\hat { h }$ transfers the ofline proxy labels to user– item pairs that appear in the serving candidate family. The encoder remains fixed while ℎ<sup>̂</sup> is trained.

Calibration split. Counterfactual fine-tuning, alignment training, and conformal calibration use disjoint user splits. In particular,

$$
\begin{array} { r } { D _ { \mathrm { p r o x y } } = D _ { \mathrm { a l i g n } } \cup \mathcal { D } _ { \mathrm { c a l } } , \quad \mathcal { V } _ { \mathrm { a l i g n } } \cap \mathcal { V } _ { \mathrm { c a l } } = \emptyset . } \end{array}
$$

Let

$$
\mathcal { D } _ { \mathrm { c a l } } ^ { ( 0 ) } = \left\{ ( u _ { i } , j _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } : Y _ { u _ { i } j _ { i } } ^ { \mathrm { p r o x y } } = 0 \right\}
$$

denote the proxy-null calibration set. No pair in ${ \cal D } _ { \mathrm { c a l } } ^ { ( 0 ) }$ is used to train the encoder, item embeddings, or alignment predictor. Conditional on these learned artifacts, the theoretical guarantee requires the proxy-null calibration scores to be exchangeable with proxy-null scores in the target candidate family.

Conformal evidence. For every user–item pair, we define the nonconformity score $V ( u , j ) = 1 - \hat { h } ( u , j )$ . For a candidate pair $( u , j )$ in the serving batch, its lower-tail conformal �-value is

$$
p ( u , j ) = \frac { 1 + \sum _ { ( u _ { i } , j _ { i } ) \in D _ { \mathrm { c a l } } ^ { ( 0 ) } } \mathbb { 1 } [ V ( u _ { i } , j _ { i } ) \leq V ( u , j ) ] } { 1 + | D _ { \mathrm { c a l } } ^ { ( 0 ) } | } .\tag{7}
$$

A small $p ( u , j )$ provides evidence against the proxymisalignment null. Under conditional target-environment exchangeability, the �-value is super-uniform for every pair in $\mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } }$ . Lemma 2 formalizes this property.

FDR-controlled set selection. For a serving batch $B ,$ let $N = | \mathcal { Q } ( B ) |$ and order the candidate $p \mathrm { - }$ values as $p _ { ( 1 ) } \leq \cdots \leq$ $p _ { ( N ) }$ . BH select

$$
\widehat { k } _ { \alpha } = \operatorname* { m a x } \left\{ k : p _ { ( k ) } \leq \frac { \alpha k } { N } \right\} ,\tag{8}
$$

with $\widehat { k } _ { \alpha } \ = \ 0$ when no index satisfies the condition, and returns

$$
S _ { \alpha } ( B ) = \left\{ ( u , j ) \in \mathcal { Q } ( B ) : p ( u , j ) \leq \frac { \alpha \widehat { k } _ { \alpha } } { N } \right\} .\tag{9}
$$

The BH rule returns the largest member of its nested step-up family that satisfies the BH condition. Lemma 1 establishes this maximality property. Under the exchangeability and dependence conditions of Theorem 1, the resulting set controls pooled proxy-label FDR at level �. When arbitrary dependence is allowed, we replace � with the BY correction $\alpha / H _ { N }$ , where $\begin{array} { r } { H _ { N } = \sum _ { k = 1 } ^ { N } k ^ { - 1 } } \end{array}$

## 3.3. Risk-Controlled Serving

After ofline training and calibration, We apply the learned ranker and stored calibration artifacts to each serving batch. The online state is

$$
\Theta _ { \mathrm { s e r v e } } = \left( \hat { q } _ { \phi } , \hat { e } _ { 1 : | T | } , \hat { h } , \{ V _ { i } \} _ { i \in \mathcal { D } _ { \mathrm { c a l } } ^ { ( 0 ) } } \right) .
$$

The decoder and LLM are not part of the online path.

Candidate ranking. For each user � in a serving batch $B ,$ the encoder infers $( \hat { \mathbf { z } _ { c } } ^ { ( u ) } , \hat { \mathbf { z } _ { e } } ^ { ( u ) } )$ from $x _ { u } .$ The counterfactually fine-tuned ranker uses $\hat { \pmb { z } } _ { c } ^ { ( u ) }$ to form $C _ { u } ^ { K } \ = \ \mathrm { T o p K } _ { j \in { \cal I } } \left. \hat { \bf z _ { c } } ^ { ( u ) } , e _ { j } \right.$ . The batch candidate family is

$$
\mathcal { Q } ( B ) = \left\{ ( u , j ) : u \in B , j \in C _ { u } ^ { K } \right\} .
$$

Certified serving. We evaluate $\hat { h }$ and compute conformal �-values for all pairs in (). Applying Eq. (8) produces the pooled served set $S _ { \alpha } ( B )$ . The list returned to user � is

$$
S _ { \alpha } = \left\{ j \in C _ { u } ^ { K } : ( u , j ) \in S _ { \alpha } ( B ) \right\} ,
$$

in the original ranker order; an empty list is an abstention. The FDR guarantee applies to the pooled set $S _ { \alpha } ( B )$ . Each user-level list inherits the membership decisions of that set. The risk level � is specified for the pooled serving batch. Each user-level list inherits membership decisions from $S _ { \alpha } ( B ) ;$ the certificate does not assign a separate FDR guarantee to each user.

Notably, GenCAR ofers several compelling advantages:

• Preference-grounded. GenCAR fixes $\mathbf { z } _ { c }$ under the environmental intervention and constrains ofline LLM proposals through preference anchors and trust-radius filtering.

• Risk-controlled. Conformal evidence and BH or BY convert the user-specified level � into pooled proxylabel FDR control under the conditions in Section 4.

• LLM-free online serving. LLM generation and proxy labeling occur ofline. Online serving uses the encoder, item embeddings, alignment predictor, and stored proxy-null scores.

Algorithm 1 summarizes the complete ofline and online pipeline. The next section bounds the conditional counterfactual approximation error, establishes maximality within the BH family, and proves that calibrated selection satisfies the proxy-label FDR constraint in �-VCR.

## 4. Theoretical Analysis

In this section, we analyze the construction and selection stages of GenCAR. We first bound the conditional approximation error introduced by anchor projection, LLM realization, and trust-radius filtering (Proposition 1). We next show that the BH candidate sets form a nested family with a largest step-up-admissible member (Lemma 1) and establish finite-sample validity of the proxy-null conformal �-values (Lemma 2). Finally, we prove pooled proxy-label FDR control at level � (Theorem 1), extend the guarantee to arbitrary dependence (Corollary 1), and provide a proxy-tocausal risk transfer bound (Corollary 2).

## 4.1. Counterfactual Approximation and Selection Structure

We first analyze the counterfactual construction stage. For a fixed user $u ,$ let $\mathcal { T } _ { u } ( \delta ) = \{ j \in \mathcal { I } : d _ { 0 } ( j , \mathbf { z } _ { c } ^ { ( u ) } ) \leq \delta \}$ denote the trust-radius acceptance event. Recall that $P ^ { \star } , P ^ { A }$ and $P ^ { M }$ denote the counterfactual target, anchor-projected distribution, and LLM realization defined in Eq. (4). We compare the target and realized distributions after applying the same acceptance event.

Algorithm 1 GenCAR training and risk-controlled serving.   
Input: Training data $\pmb { D } ^ { \mathrm { t r a i n } }$ , counterfactual split $D _ { \mathrm { c f } } ,$ alignment   
split $D _ { \mathrm { a l i g n } } ,$ calibration split $\nu _ { \mathrm { c a l } } ,$ ofline LLM , target in  
tervention $\tilde { \mathbf { z } } _ { e } ,$ , trust radius �, anchor size $K _ { a } ,$ candidate size �   
and risk level �   
Output: Pooled served set $S _ { \alpha } ( B )$ and user lists $S _ { \alpha } ( u )$   
Ofline construction   
1: Train the backbone encoder and embeddings $E _ { 0 }$ on $D ^ { \mathrm { t r a i n } }$   
2: for each user � represented in $\mathcal { D } _ { \mathrm { c f } }$ do   
3: Infer $\mathbf { z } _ { c } ^ { ( u ) }$ and form preference anchors $\mathcal { A } _ { u }$   
4: Generate $\widehat { \mathcal { Y } } _ { u } \gets \mathcal { M } ( \mathcal { A } _ { u } , \tilde { \mathbf { z } } _ { e } )$   
5: Filter proposals by Eq. (2) to obtain $\mathcal { V } _ { u } ^ { \mathrm { c f } }$   
6: Form $\Pi _ { u } ^ { \mathrm { c f } }$ from $j ^ { + } \in \mathcal { V } _ { u } ^ { \mathrm { c f } }$ and user-unobserved $j ^ { - }$   
7: end for   
8: Optimize Eq. (3) to obtain $E _ { \mathrm { c f } }$   
Ofline calibration   
9: Label alignment and calibration pairs by Eq. (5)   
10: Fit ℎ on $\mathcal { D } _ { \mathrm { a l i g n } }$ and form $\mathcal { D } _ { \mathrm { c a l } } ^ { ( 0 ) }$   
11: Store $\{ V _ { i } = \stackrel {  } { 1 } - h ( u _ { i } , j _ { i } ) \} _ { ( u _ { i } , j _ { i } ) \in \mathcal { D } _ { \mathrm { c a l } } ^ { ( 0 ) } }$   
LLM-free serving   
12: Infer $\mathbf { z } _ { c } ^ { ( u ) } , \mathbf { z } _ { e } ^ { ( u ) }$ and rank each $u \in \mathcal Ḋ B Ḍ$ into $c _ { u } ,$ pool Q(B)   
13: Compute $h ( u , j )$ and Eq. (7) for all $( u , j ) \in \mathcal { Q } ( B )$   
14: Apply Eqs. $( 8 ) \AA { - } ( 9 )$   
15: for each � ∈  do   
16: Return $S _ { \alpha } ( u ) = \{ j \in C _ { u } : ( u , j ) \in S _ { \alpha } ( B ) \}$ in ranker order   
17: end for

Proposition 1 (Conditional error propagation under trust-radius filtering). Assume $P ^ { \star } ( \mathcal T _ { u } ( \delta ) ) > 0$ and $P ^ { M } ( \mathcal { T } _ { u } ( \delta ) ) > 0$ Let

$$
\begin{array} { r } { \boldsymbol { P } _ { \delta } ^ { \star } = \boldsymbol { P } ^ { \star } ( \cdot \mid \mathcal { T } _ { \boldsymbol { u } } ( \delta ) ) , \qquad \boldsymbol { P } _ { \delta } ^ { M } = \boldsymbol { P } ^ { M } ( \cdot \mid \mathcal { T } _ { \boldsymbol { u } } ( \delta ) ) , } \end{array}
$$

where TV denotes total variation distance, and let $\epsilon _ { A } ~ = ~ \mathrm { T V } ( P ^ { \star } , P ^ { A } ) , ~ \epsilon _ { M } ~ = ~ \mathrm { T V } ( P ^ { A } , P ^ { M } ) ,$ , and $\rho _ { \delta } \quad = $ min{ $P ^ { \star } ( \mathcal T _ { u } ( \delta ) ) , P ^ { M } ( \mathcal T _ { u } ( \delta ) ) \}$ . Then

$$
\mathrm { T V } ( P _ { \delta } ^ { \star } , P _ { \delta } ^ { M } ) \leq \frac { 2 ( \epsilon _ { A } + \epsilon _ { M } ) } { \rho _ { \delta } } .
$$

For the causal-misalignment event $\mathcal { M } _ { u } = \{ j \in \mathcal { I } : G _ { u j } =$ 0},

$$
P _ { \delta } ^ { M } ( \mathcal { M } _ { u } ) \leq P _ { \delta } ^ { \star } ( \mathcal { M } _ { u } ) + \frac { 2 ( \epsilon _ { A } + \epsilon _ { M } ) } { \rho _ { \delta } } .
$$

The proof is provided in Appendix A.1. Proposition 1 quantifies construction fidelity after trust-radius filtering. The terms $\epsilon _ { A }$ and $\epsilon _ { M }$ isolate the errors from anchor projection and LLM realization, while $\rho _ { \delta }$ records the probability mass retained by the filter. The result links the realized counterfactual distribution to the target intervention on the accepted region.

We next analyze the structure of the calibrated served sets. For a serving batch B, let Q(B) contain N candidate pairs. Write their ordered conformal �-values as $p _ { ( 1 ) } \leq \cdots \leq$ $p _ { ( N ) } .$ and let $S _ { k }$ contain the � pairs with the smallest $p \mathrm { - }$ values. This family provides the candidate sets over which the BH rule operates.

Lemma 1 (Nestedness and step-up maximality). The ordered candidate sets form a nestedfamily: $S _ { 0 } \subseteq S _ { 1 } \subseteq \cdots \subseteq$ $S _ { N }$ . For a risk level $\alpha \in ( 0 , 1 )$ , define

$$
\widehat { k } _ { \alpha } = \operatorname* { m a x } \left\{ k : p _ { ( k ) } \leq \frac { \alpha k } { N } \right\} ,
$$

with $\widehat { k } _ { \alpha } = 0$ when the set is empty. Then $S _ { \alpha } = S _ { \hat { k } _ { \alpha } }$ is the largest member of the nestedfamily satisfying the BH stepup condition. Moreover,for any $0 < \alpha _ { 1 } \le \alpha _ { 2 } < 1$

$$
S _ { \alpha _ { 1 } } \subseteq S _ { \alpha _ { 2 } } .
$$

The proof is given in Appendix A.2. Lemma 1 establishes the size property used by GenCAR. At a fixed level �, the method returns the largest set admitted by the BH stepup family. The nesting in � provides an ordered control over served-set size.

## 4.2. Proxy-Label Risk Control

We now present the main theoretical guarantee of Gen-CAR, and show that conformal evidence computed from a finite proxy-null calibration set controls the expected proxymisalignment fraction in an unseen serving batch.

Let $n _ { 0 } = | \mathcal { D } _ { \mathrm { c a l } } ^ { ( 0 ) } |$ . The same � and � from Eq. (7) are used in the algorithm and guarantee. The first result establishes the finite-sample validity of this evidence under the proxymisalignment null.

Lemma 2 (Finite-sample proxy-null validity). Suppose that the alignment predictor ℎ<sup>̂</sup> and candidate family Q(B) are constructed independently of ${ \mathcal D } _ { \mathrm { c a l } } ^ { ( 0 ) } .$ . For every $( u , j ) \in \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } }$ , assume that its nonconformity score and the $n _ { 0 }$ proxy-null calibration scores are exchangeable, conditional on the learned artifacts and candidate family. The �-value is the weak-rank statistic defined in Eq. (7).

Then, for every $t \in [ 0 , 1 ]$

$$
\operatorname* { P r } \left( p ( u , j ) \leq t \mid \hat { h } , { \mathcal { Q } } ( B ) \right) \leq t .
$$

Thus, every proxy-null conformal �-value is super-uniform.

The proof is provided in Appendix A.3. The weak-rank construction in Eq. (7) is super-uniform under exchangeability and remains conservative when scores contain ties. Lemma 1 and Lemma 2 provide the two ingredients for calibrated selection. Nestedness identifies the largest BHadmissible set, while proxy-null validity supplies the evidence required for FDR control.

Theorem 1 (Proxy-label FDR control). Condition on the learned artifacts and the candidatefamily (). Let

$$
N = | \mathcal { Q } ( B ) | , \quad m _ { 0 } = | \mathcal { Q } ( B ) \cap \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } | .
$$

Assume that the conditions of Lemma 2 hold and that the conformal �-value vector is positive regression dependent on a subset (PRDS) of proxy-null hypotheses, conditional

on the candidate family. Then BH at level � returns $S _ { \alpha } ( B )$ satisfying

$$
\mathbb { E } \left[ \frac { | S _ { \alpha } ( B ) \cap \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } | } { \operatorname* { m a x } \{ | S _ { \alpha } ( B ) | , 1 \} } \Bigg | \hat { h } , Q ( B ) \right] \leq \frac { m _ { 0 } } { N } \alpha \leq \alpha .
$$

Consequently,

$$
\mathrm { F D R } _ { \mathrm { p r o x y } } \left( S _ { \alpha } ( B ) \right) \leq \alpha .
$$

The proof is given in Appendix A.4. Theorem 1 controls the proxy-label risk at any finite calibration size without specifying a parametric null-score distribution. Sharedcalibration marginal conformal �-values satisfy PRDS under jointly independent and continuously distributed null scores [22].

Remark 2 (Interpretation of the guarantee). The certified object in Theorem 1 is the pooled candidate-pair set $S _ { \alpha } ( B )$ Each user-level list inherits the membership decisions ofthis set. The guarantee controls the expectedproxy-misalignment fraction over the serving batch.

Corollary 1 (Arbitrary-dependence control). Under the conditions of Lemma 2, define $\begin{array} { r } { H _ { N } = \sum _ { k = 1 } ^ { N } \frac { 1 } { k } } \end{array}$ . Applying the Benjamini–Yekutieli rule at level $\alpha / H _ { N }$ yields

$$
\mathrm { F D R } _ { \mathrm { p r o x y } } \left( S _ { \alpha } ^ { \mathrm { B Y } } ( { \cal B } ) \right) \le \alpha
$$

under arbitrary dependence among the candidate �-values.

The proof is included in Appendix A.4. Corollary 1 provides the same proxy-label certificate without a dependence restriction. GenCAR uses BH under PRDS and BY under arbitrary dependence.

Corollary 2 (Proxy-to-causal risk transfer). Let $\mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } =$ $\left\{ ( u , j ) \in \mathcal { Q } ( \beta ) : G _ { u j } = 0 \right\}$ denote the causal-misalignment null. Define

$$
\mathrm { F D R } _ { \mathrm { c a u s a l } } ( S ) = \mathbb { E } \left[ \frac { | S \cap \mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } | } { \operatorname* { m a x } \{ | S | , 1 \} } \right]
$$

and

$$
\Delta _ { \mathrm { p r o x y } } ( S ) = \mathbb { E } \left[ \frac { \vert S \cap ( \mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } \setminus \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } ) \vert } { \operatorname* { m a x } \{ \vert S \vert , 1 \} } \right] .
$$

The BH set in Theorem 1 satisfies

$$
\mathrm { F D R } _ { \operatorname { c a u s a l } } \big ( S _ { \alpha } ( B ) \big ) \leq \alpha + \Delta _ { \mathrm { p r o x y } } \big ( S _ { \alpha } ( B ) \big ) .
$$

$I f ~ { \mathcal { H } } _ { 0 } ^ { \mathrm { c a u s a l } } \subseteq { \mathcal { H } } _ { 0 } ^ { \mathrm { p r o x y } }$ , then $\Delta _ { \mathrm { p r o x y } } ( S _ { \alpha } ( B ) ) \quad = \quad 0$ and $\mathrm { F D R } _ { \mathrm { c a u s a l } } \big ( S _ { \alpha } ( B ) \big ) \leq \alpha .$

The proof is provided in Appendix A.5. Corollary 2 isolates the causal nulls missed by the observable proxy. Proposition 1 supports the counterfactual construction stage. Theorem 1 answers the �-VCR question for proxy-label risk, while Corollary 2 translates this certificate into a causal-risk bound through Δ<sub>proxy</sub>.

Dataset statistics for the ranking and calibration evaluations. T, E, and P denote temporal, exposure, and popularity shifts, respectively.
<table><tr><td>Dataset</td><td></td><td>#Users #Items</td><td></td><td>#Inter. Density Shift</td><td></td></tr><tr><td>MovieLens(ML)-100K</td><td>942</td><td>1,447</td><td></td><td>55,375 4.058%</td><td>T</td></tr><tr><td>Coat</td><td>290</td><td>300</td><td></td><td>6,960 8.000%</td><td>E</td></tr><tr><td>Amazon-Book</td><td>10,000</td><td></td><td>89,911 1,550,000 0.172%</td><td></td><td>P</td></tr><tr><td>Amazon-Beauty</td><td>22,363</td><td>12,101</td><td></td><td>198,502 0.073%</td><td>P</td></tr><tr><td>MovieLens(ML)-1M</td><td>6,040</td><td></td><td>3,706 1,000,209 4.468%</td><td></td><td>T</td></tr></table>

## 5. Experiments

In this section, we present the experimental results to validate four claims: (i) calibrated selection keeps realized proxy FDP below � on the evaluated diagnostic pools; (ii) preference-grounded counterfactual supervision improves top-10 OOD candidate recovery; (iii) the gains arise from counterfactual content, preference grounding, and trustradius filtering; and (iv) GenCAR preserves an LLM-free, sub-millisecond serving path under full and partial ofline coverage.

## 5.1. Experimental Setup

Datasets. We evaluate GenCAR under three common forms of recommendation shift. ML-100K [15] represents temporal shift, Coat [1] represents exposure shift, and Amazon-Book [16] represents popularity shift. These three datasets form the main ranking evaluation. We further include MovieLens-1M [15] and Amazon-Beauty [16] to evaluate calibration across larger temporal and popularityshift pools. Steam [23] is used to study the setting in which ofline LLM generation is available for only a subset of users. Table 1 summarizes the datasets and their shift types.

Baselines and evaluation metrics. We compare GenCAR with the standard BPRMF ranker [20], propensity-based methods including CausE [24], MACR [25], and IPS-CN [1], and representation-based OOD recommenders including DICE [6], InvCF [3], and CDR [7]. We also include CausalDifRec [8] as a generative baseline. TallRec [26] provides a separate reference for methods that invoke an LLM during serving. The CausalVAE backbone isolates the contribution of GenCAR’s ofline counterfactual pipeline on ML-100K and Coat. We report Recall@� and normalized discounted cumulative gain (NDCG)@� for $K \in \{ 1 0 , 2 0 \}$ . Recall@10 is the primary metric because GenCAR targets the recovery of useful candidates near the top of the served list. Inverse propensity score (IPS)-weighted evaluation measures the same utility after correcting for exposure frequency. Ranking results are averaged over seeds 2024, 2025, and 2026. All methods use the same shift-specific 80/10/10 interaction split and evaluation protocol. Model selection is performed on the validation split, and the reported results are computed on the held-out test split.

Table 2  
Realized proxy FDP and proxy-alignment rate under BH selection.
<table><tr><td>Dataset</td><td>α</td><td> $n _ { \mathrm { t e s t } }$ </td><td> $n _ { \mathrm { s e l } }$ </td><td> $\widehat { \mathbf { F } \mathbf { D P } }$ </td><td> $1 - \widehat { \mathbf { F D P } }$ </td></tr><tr><td>Coat</td><td>0.30</td><td>1,376</td><td>16</td><td>0.188</td><td>0.812</td></tr><tr><td>ML-100K</td><td>0.30</td><td>16,614</td><td>241</td><td>0.195</td><td>0.805</td></tr><tr><td>ML-1M</td><td>0.30</td><td>172,586</td><td>3,891</td><td>0.089</td><td>0.911</td></tr><tr><td>Beauty</td><td>0.30</td><td>21,840</td><td>1,487</td><td>0.054</td><td>0.946</td></tr><tr><td> $\mathsf { A m a z o n – B o o k }$ </td><td>0.30</td><td>559,658</td><td>7,665</td><td>0.048</td><td>0.952</td></tr></table>

Implementation. All models are implemented in PyTorch and trained on NVIDIA L20 graphics processing units (GPUs). We use Adam with a learning rate of $1 0 ^ { - 3 }$ and weight decay of $1 0 ^ { - 4 }$ The CausalVAE encoder produces factorized posteriors with $\mathrm { d i m } ( { \bf z } _ { c } ) = 6 4 $ and $\mathrm { d i m } ( { \bf z } _ { e } ) \ : = \ : \mathrm { d i m } ( \eta ) \ : = \ : 1 6$ from a shared 256-dimensional hidden representation. We set $\lambda _ { \mathrm { B P R } } = 1 . 0 , \lambda _ { \mathrm { d i s e n t } } = 0 . 0 5$ and the Kullback–Leibler (KL) weight $\lambda _ { \mathrm { K L } } ~ = ~ 0 . 2$ . The disentanglement loss is computed with 256 sampled users per update. The fixkl configuration stabilizes the preference and environment factors before counterfactual generation. We use $K _ { a } \ = \ 5$ preference anchors and set the default trust radius to $\delta \mathrm { ~  ~ { ~ = ~ } ~ } 0 . 7$ . DeepSeek-Chat generates the counterfactual candidates, and prompts are cached for each dataset and seed. The LLM is used only during ofline training. Amazon-Book contains sparse longtail interactions that provide weak item representations. We initialize its item embeddings with BAAI General Embedding (BGE)-small-en text representations projected by principal component analysis. An anchor penalty with $\lambda _ { \mathrm { a n c } } = 5$ keeps the fine-tuned embeddings close to this text prior. The proxy-alignment threshold is $\tau = 0 . 8 5$ , and the reported risk audit uses BH at $\alpha = 0 . 3 0 .$ Alignment fitting and calibration use disjoint users, so no user contributes to both predictor fitting and the null reference distribution.

## 5.2. Risk Control and Calibration

The first experiment audits calibrated selection on five constructed diagnostic pools, separately from the rankerlevel candidate-recovery evaluation in Sec. 5.3. Table 2 evaluates realized proxy FDP and retained support, whereas Sec. 5.3 evaluates top-� OOD recovery. Proxy-alignment fitting and null calibration use disjoint users in every dataset, and BH at $\alpha = 0 . 3 0$ is applied only to the held-out test pool. Table 2 reports both sides of the resulting selection profile: realized proxy FDP and the amount of retained support.

GenCAR keeps realized proxy FDP below � on all five candidate pools. $\mathrm { A t } \alpha = 0 . 3 0$ , Table 2 reports realized proxy FDP values of $0 . 1 8 8 , 0 . 1 9 5 , 0 . 0 8 9 , 0 . 0 5 4$ , and 0.048 on Coat, ML-100K, ML-1M, Amazon-Beauty, and Amazon-Book, respectively. The selector retains between 16 and 7,665 user– item pairs. The corresponding retention rates are 1.16%, 1.45%, 2.25%, 6.81%, and 1.37%, showing how retained support varies with pool scale and calibration support at a fixed risk level.

![](images/58d1b223077a38f1b08aa3cf9eaecefd8b0f363aec79e0e5aa8c3a634056e268.jpg)  
(a) Calibration support and BH pass rate

![](images/2727311bbcd0aa76671c41f2b1e8239cb351c645b398c229f2b314ff2b1f29f6.jpg)  
Fig. 3. Finite-sample calibration diagnostics. (a) BH pass rate as a function of positive calibration support. (b) Empirical proxy alignment relative to the nominal target across runs. The Amazon-Book results compare global and popularity-stratified calibration at $\alpha = 0 . 1 0$

Positive calibration support tracks BH retention. The variation in retained support across pools motivates Fig. 3(a), which relates the observed BH pass rate to the number of positive calibration pairs $n _ { \mathrm { c a l + } }$ . Across the five pools, larger positive support is associated with higher retention at the same �.

Popularity stratification aligns heterogeneous score regions. Fig. 3(b) examines popularity-dependent score heterogeneity without changing the underlying recommendations: global and popularity-stratified calibration are applied to the same Amazon-Book candidate pool at $\alpha = 0 . 1 0$ , with the ranker and candidate pool held fixed. Candidates are grouped by training popularity, calibrated within each group, and pooled after selection. $\mathrm { A t } \alpha = 0 . 1 0 $ , this diagnostic raises empirical proxy alignment from 0.735 to 0.988, exceeding the nominal target of 0.90 with the ranker fixed.

## 5.3. OOD Candidate Recovery

This experiment evaluates whether preference-grounded counterfactual supervision improves OOD candidate recovery under diverse shifts. We compare GenCAR with methods for propensity correction, invariant representation learning, and counterfactual generation. TallRec provides an inference-time LLM reference. Recall@10 is the primary metric because it measures useful candidates near the top of each ranked list. The matched CausalVAE comparison holds the backbone, data splits, and evaluation protocol fixed, leaving the ofline counterfactual pipeline as the experimental diference. Table 3 reports the resulting means over three seeds.

GenCAR achieves the strongest LLM-free top-10 recovery across all three shifts. Table 3 reports means over three seeds. GenCAR ranks first in 10 of the 12 LLM-free dataset– metric comparisons and second in the remaining two. Its Recall@10 improves over the corresponding CausalVAE references from 0.0856 to 0.0950, from 0.0639 to 0.0761, and from 0.0046 to 0.0066, corresponding to gains of 11.0%, 19.1%, and 43.5%. The gain is positive under all three shifts and in all nine dataset–seed comparisons (Fig. 4 and Fig. 6(c)).

Table 3  
Mean OOD ranking performance over three seeds under temporal, exposure, and popularity shifts. Bold and underlined entries denote the best and second-best results among methods without online LLM calls. TallRec is included as an inference-time LLM reference. R@� and N@� denote Recall@� and NDCG@�, respectively; BB denotes the CausalVAE backbone.
<table><tr><td rowspan="2">Method</td><td colspan="4">ML-100K</td><td colspan="4">Coat</td><td colspan="4">Amazon-Book</td></tr><tr><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td></tr><tr><td>BPRMF [20]</td><td>0.0409</td><td>0.0794</td><td>0.0939</td><td>0.0990</td><td>0.0726</td><td>0.1238</td><td>0.0415</td><td>0.0586</td><td>0.0002</td><td>0.0003</td><td>0.0005</td><td>0.0005</td></tr><tr><td>DICE [6]</td><td>0.0445</td><td>0.0707</td><td>0.0929</td><td>0.0942</td><td>0.0730</td><td>0.1285</td><td>0.0403</td><td>0.0587</td><td>0.0001</td><td>0.0003</td><td>0.0005</td><td>0.0004</td></tr><tr><td>IPS-CN [1]</td><td>0.0329</td><td>0.0636</td><td>0.0699</td><td>0.0751</td><td>0.0674</td><td>0.1175</td><td>0.0409</td><td>0.0582</td><td>0.0001</td><td>0.0002</td><td>0.0003</td><td>0.0003</td></tr><tr><td>CausE [24]</td><td>0.0337</td><td>0.0623</td><td>0.0841</td><td>0.0854</td><td>0.0535</td><td>0.0956</td><td>0.0357</td><td>0.0502</td><td>0.0001</td><td>0.0002</td><td>0.0003</td><td>0.0003</td></tr><tr><td>MACR [25]</td><td>0.0013</td><td>0.0027</td><td>0.0032</td><td>0.0036</td><td>0.0157</td><td>0.0317</td><td>0.0119</td><td>0.0177</td><td>0.0001</td><td>0.0002</td><td>0.0003</td><td>0.0003</td></tr><tr><td>InvCF [3]</td><td>0.0322</td><td>0.0638</td><td>0.0748</td><td>0.0774</td><td>0.0641</td><td>0.1183</td><td>0.0374</td><td>0.0554</td><td>0.0001</td><td>0.0002</td><td>0.0003</td><td>0.0003</td></tr><tr><td>CDR [7]</td><td>0.0300</td><td>0.0591</td><td>0.0676</td><td>0.0713</td><td>0.0518</td><td>0.0825</td><td>0.0315</td><td>0.0416</td><td>0.0001</td><td>0.0002</td><td>0.0003</td><td>0.0003</td></tr><tr><td>CausaIDiffRec [8]</td><td>0.0116</td><td>0.0225</td><td>0.0422</td><td>0.0439</td><td>0.0402</td><td>0.0596</td><td>0.0274</td><td>0.0345</td><td>0.0002</td><td>0.0004</td><td>0.0006</td><td>0.0006</td></tr><tr><td colspan="9">LLM-based recommenders (inference-time LLM call)</td><td></td><td></td><td></td><td></td></tr><tr><td>TallRec [26] (Qwen2.5-3B + LoRA)</td><td>0.0675</td><td>0.1231</td><td>0.1746</td><td>0.1669</td><td>0.0637</td><td>0.1194</td><td>0.0446</td><td>0.0606</td><td>0.0070</td><td>0.0124</td><td>0.0188</td><td>0.0177</td></tr><tr><td>CausalVAE (BB)</td><td>0.0856</td><td>0.1409</td><td>0.1808</td><td>0.1842</td><td>0.0639</td><td>0.1197</td><td>0.0417</td><td>0.0603</td><td>0.0046</td><td>0.0086</td><td>0.0116</td><td>0.0116</td></tr><tr><td>GenCAR</td><td>0.0950</td><td>0.1438</td><td>0.1820</td><td>0.1798</td><td>0.0761</td><td>0.1240</td><td>0.0438</td><td>0.0604</td><td>0.0066</td><td>0.0120</td><td>0.0165</td><td>0.0162</td></tr><tr><td>∆ vs BB</td><td>11.0%↑</td><td>2.1%↑</td><td>0.7%↑</td><td>2.4%↓</td><td>19.1%↑</td><td>3.6%↑</td><td>5.0%↑</td><td>0.2%↑</td><td>43.5%↑</td><td>39.5%↑</td><td>42.2%↑</td><td>39.7%↑</td></tr></table>

![](images/75116e2d5bcd004e5b560cb57f6b6e920cbae51ff2eeb3a27fbfff5448b880ec.jpg)  
ML-100K (temporal)

![](images/4d31f96f8ded83f0a359d3e27c29fc71da1efeb3331e0acece1084582499f289.jpg)  
Coat (exposure)

![](images/618ff1b169ad575aa13f450bdad3e2edcc84e0577564c8eeaed3e97d2dc0b410.jpg)  
Amazon-Book (popularity)  
Fig. 4. OOD performance under temporal, exposure, and popularity shifts. Bars report means over three seeds. Percentages show GenCAR’s relative gains over the corresponding CausalVAE backbone

GenCAR preserves an LLM-free online path. The TallRec reference separates ofline counterfactual supervision from LLM access during serving: GenCAR uses embedding-based inference, whereas TallRec invokes an LLM online on the same datasets and ranking metrics. GenCAR exceeds the inference-time TallRec reference on ML-100K (0.0950 versus 0.0675) and Coat (0.0761 versus 0.0637). On Amazon-Book, GenCAR approaches TallRec (0.0066 versus 0.0070) while retaining embedding-based online scoring. The cross-shift recovery gains therefore do not require an online LLM call.

## 5.4. Mechanism and Sensitivity Analysis

The preceding results establish GenCAR’s cross-shift Recall@10 gain. We next examine whether this gain arises from counterfactual content, candidate novelty, trust-radius filtering, or preference factorization. We vary one factor at a time under matched backbones and evaluation protocols. The resulting comparisons address four distinct explanations for the main gain: augmentation volume, candidate novelty, preference compatibility, and run-to-run variation.

Preference-grounded counterfactuals outperform matched augmentation controls. On Coat, random and inverse-popularity augmentation add the same number of training pairs as the LLM counterfactuals, controlling for augmentation volume. Under the shared ablation configuration, LLM counterfactuals achieve a Recall@10 of 0.0720 on Coat, compared with 0.0641 for random augmentation and 0.0653 for inverse-popularity sampling. The 12.3% gain over the matched random control isolates the value of counterfactual content from the number of additional training pairs.

![](images/2ad0e05a4a4cf6faa39253ccafecd2529c5ec214bb61bda62c6c6d13b9c5cc4c.jpg)  
Fig. 5. Efect of the ofline LLM on Coat. Bars report the Recall@10 improvement over the CausalVAE backbone. Circle markers report the fraction of generated candidates absent from the observed training support.

Ofline LLM choice separates novelty from ranking gain. With augmentation volume controlled, candidate novelty remains a separate explanation. Fig. 5 compares the novelty profiles of DeepSeek-Chat, V4-Flash, and V4-Pro under the same Coat backbone with their Recall@10 gains over the backbone. DeepSeek-Chat generates 88.6% novel candidates and improves Recall@10 by 8.0%. V4-Flash and V4-Pro each reach 100% novelty, with gains of 18.7% and 11.5%. Equal novelty with diferent recovery gains shows that support expansion and candidate utility capture distinct properties.

Trust-radius filtering has a stable operating region on the primary ranking benchmarks. Because � determines which generated candidates are treated as preferencecompatible, we sweep $\begin{array} { r } { \delta \in \mathrm { ~  ~ { ~ \{ ~ 0 ~ . 3 , 0 . 5 , 0 . 7 , 0 . 9 , 1 . 0 \} } ~ } } \end{array}$ on ML-100K and Coat with the remaining configuration fixed. Fig. 6(a) shows that ML-100K reaches its largest gain at $\delta = 0 . 5 $ , with gradual variation between � = 0.5 and 0.7. Coat exhibits a similarly stable region around the default value $\delta = 0 . 7 .$ These curves identify a broad operating range for filtering preference-compatible proposals.

KL-clamped preference factorization improves novelty and recovery together. The factorization analysis pairs counterfactual novelty with Recall@10 gain, comparing the original backbone with the KL-clamped (fixkl) configuration. Fig. 6(b) shows that novelty rises from 18.3% to 70.3%, while the Recall@10 gain rises from near zero to approximately 8.0%. This paired change links stable preference factorization to more useful counterfactual supervision. Fig. 6(c) then extends the aggregated comparison to the three individual seeds under each main shift. The figure reports positive Recall@10 gains in all nine dataset–seed comparisons, covering three shifts and three seeds. This consistency supports the repeatability of the main candidate-recovery result.

## 5.5. Qualitative Analysis

The aggregate evaluations quantify pooled proxy-label FDP and OOD candidate recovery. We next examine how GenCAR changes candidate composition for individual users. The analysis traces the environmental intervention, candidate-level proxy evidence, and latent candidate placement.

Environmental intervention changes candidate composition under fixed preference. Fig. 7 follows Coat User 156 from logged missing-not-at-random (MNAR) exposure to the counterfactual missing completely at random (MCAR) intervention: $\mathbf { z } _ { c }$ is fixed, $\mathbf { z } _ { e }$ is replaced by the MCAR target, and counterfactual items found among the held-out positives are marked. Bomber jackets account for 50% of logged interactions and 33% of the counterfactual distribution. The intervention assigns 22%, 22%, and 23% to Packable, Rain, and Other jackets; Packable and Other also occur among the held-out MCAR positives. This case traces the change induced by $\mathrm { d o } ( \mathbf { z } _ { e } : = \bar { \mathbf { z } } _ { e } ^ { \mathrm { M C A R } } )$ while $\mathbf { z } _ { c }$ remains fixed.

Proxy-alignment scores organize candidate-level evidence before conformal calibration. Table 4 complements the pooled risk audit with candidate-level evidence from five Amazon-Book users under popularity shift. Alignment scores are shown above, near, and below $\tau \ = \ 0 . 8 5$ , with held-out positives marked for reference. For User 818, The Sense of an Ending receives a proxyalignment score of 0.90. This item is a held-out positive that is absent from the backbone shortlist. Users 166 and 449 include candidates above, near, and below the proxy threshold. These cases visualize the graded candidatelevel evidence available before conformal calibration. The additional users show the same score ordering across diferent preference histories. For User 974, goal- and mastery-oriented titles receive scores of 0.85–0.95, while the more broadly related Wired to Care receives 0.50. For User 1249, Dark Mirror and Starship Troopers receive 0.90 and 0.85, compared with 0.30 for the cross-genre My Side of the Mountain. Across these cases, the scores provide graded candidate evidence for the specified counterfactual preference before conformal calibration.

We complement the item-level cases with the latentspace view in Fig. 8. For panel (a), the two 64-dimensional representations of all Coat users are stacked into a shared $( 2 n _ { \mathrm { u s e r s } } )$ × 64 matrix and projected jointly using t-distributed stochastic neighbor embedding (t-SNE) [27], with perplexity 20, 1,000 iterations, principal component analysis (PCA) initialization, and random seed 0. Panel (b) overlays the training interactions, backbone top-10, six counterfactual candidates, and held-out positives for User 240 on the 300- item map.

Fig. 8(a) places the preference and environment representations in separated regions of the two-dimensional projection, which is consistent with the factorization used for counterfactual generation. Fig. 8(b) traces Coat User 240. One of six counterfactual candidates matches a held-out positive, and several lie outside the backbone’s top-10 neighborhood. The generated set therefore expands this user’s shortlist with a relevant held-out item.

![](images/22c521a986b84f014444b29d435b0699b69e4f09de248f6b426382c05c280e53.jpg)  
(a) Trust-radius sweep

![](images/fb75a69c42ee41ed030356ca062628499050917b1f9a3fa67e58745f7064ed94.jpg)  
CF novelty rate (%)  
(b) Novelty vs. gain

![](images/ba1b020b73ce77ff2dc8e13e10015a3da9e651a706ed17d634e4067be6881aee.jpg)  
(c) Per-seed gains

Fig. 6. Mechanism and sensitivity analysis. (a) Recall@10 gain as the trust radius varies. (b) Association between counterfactua novelty and ranking gain. (c) Per-seed gains under the main GenCAR configuration.  
![](images/f2f69d38c85876db0da7e2ec6228ef2205b0da6ccc4bbdb33fc3a5d3825505a9.jpg)  
Fig. 7. Environmental intervention for Coat User 156. (a) Logged interactions under MNAR exposure. (b) Counterfactual candidates under $\mathrm { d o } ( \mathbf { z } _ { e } : = \tilde { \mathbf { z } } _ { e } ^ { \mathrm { M C A R } } ) .$ . The preference representation $\mathbf { z } _ { c }$ remains fixed, and filled circles mark items that occur in the held-out MCAR positives. MNAR and MCAR denote missing not at random and missing completely at random, respectively.

## 5.6. Serving Eficiency and Partial Ofline Coverage

Deployment depends on both ofline coverage and online cost, which we report separately in Tables 5 and 6. The partial-coverage study evaluates all 2,664 Steam users, although ofline LLM counterfactuals are available for only 1,020 of them. The comparison includes the CausalVAE+BGE backbone, anchor counterfactual finetuning, and an auxiliary reranking variant that explicitly retains available counterfactual items.

GenCAR retains ranking utility under partial ofline coverage. Table 5 evaluates 2,664 Steam users, of whom 1,020 receive ofline LLM counterfactuals. With 38% ofline coverage, anchor counterfactual fine-tuning remains close

Amazon-Book counterfactual candidates under popularity shift. Blue denotes candidates with proxy-alignment scores at or above $\tau = 0 . 8 5 ,$ gray denotes candidates near the threshold, and yellow denotes candidates with lower scores. A ★ marks held-out positives.
<table><tr><td>User</td><td>Training history</td><td colspan="2">LLM counterfactual items (alignment score)</td><td>Held-out positives</td></tr><tr><td rowspan="3"></td><td rowspan="3">U818 The Dying Animal; Kafka on the Šhore; Charlotte Sim- mons (71 total)</td><td>A Visit from the Goon Squad (0.90) The Sense of an Ending (0.90) ★;</td><td>Year of Wonders (0.85)</td><td rowspan="3">The Sense of an Ending ★; How to Live; Blue Nights (16 total)</td></tr><tr><td>The Road (0.80) ; Stoner (0.75) U 166 L.A. Requiem; Deck the</td><td></td></tr><tr><td>Tuesdays with Morrie (0.95) I Know Why the Caged Bird Sings (0.85)</td><td>The Alchemist (0.90) When She Was Bad; Brim- stone; The Dark Tide (14 total)</td></tr><tr><td rowspan="2">U449</td><td rowspan="2">Eight (56 total) Catechism of the Catholic Church; Dreamcatcher; A Simple Plan (56 total)</td><td>Harry Potter: Prisoner of Azkaban (0.80); A Fine Balance (0.90)</td><td>War and Peace (0.60) John Adams (0.85)</td><td rowspan="2">Killing Floor (Jack Reacher); Cell: A Novel; No Country for Old Men (14 total)</td></tr><tr><td>A Simple Plan (0.85) Atlas Shrugged (0.75)</td><td>Man&#x27;s Search For Meaning (0.80);</td></tr><tr><td rowspan="2"></td><td rowspan="2">U 974 Atlas Shrugged; Capitalism: The Unknown Ideal; Extreme Programming Explained; Good to Great</td><td colspan="2">Put Your Dream to the Test (0.95) Peaks and Valleys (0.90)</td><td rowspan="2">The Case Against the Fed; Buy-In; Your Brain at Work (18 total)</td></tr><tr><td colspan="2">What Got You Here Won&#x27;t Get You There (0.82) Mastery (0.85)</td></tr><tr><td rowspan="3">(71 total)</td><td rowspan="3">U 1249 Starman Jones; Tunnel in</td><td colspan="2">Wired to Care (0.50)</td><td rowspan="3">The Fabulous Riverboat; Nightfall and Other Stories;</td></tr><tr><td>Star Trek The Next Generation: Dark Mirror (0.90)</td></tr><tr><td>Starship Troopers (0.85); Eaters of the Dead (0.80);</td></tr><tr><td colspan="2">the Sky; Imzadi; Dyson Sphere (56 total)</td><td colspan="2">Ringworld (0.75) My Side of the Mountain (0.30)</td></tr></table>

![](images/60327750c5351e39f91f7660ff34b383dc6a33aa4a7cd3ee50899324820ae062.jpg)  
(a) Preference and environment factors

![](images/d7ef500c59d2aba4472e002c17d2a30512224cb426113dce97568bf990d8d1d4.jpg)  
(b) Counterfactual candidate placement  
Fig. 8. Latent geometry and counterfactual candidate placement on Coat. (a) Two-dimensional t-distributed stochastic neighbor embedding (t-SNE) projection of the preference and environment representations. (b) Candidate neighborhood for User 240, including generated candidates, backbone top-10 items, and held-out positives. CF denotes counterfactual.

Table 5  
Ranking performance on Steam with ofline LLM counterfactuals available for 38% of the evaluated users. GenCAR + rerank is the auxiliary explicit-retention variant. R@� and N@� denote Recall@� and NDCG@�, respectively; CF denotes counterfactual.
<table><tr><td>Method</td><td>R@10</td><td>R@20</td><td>N@10</td><td>N@20</td></tr><tr><td>CausalVAE + BGE backbone</td><td>0.0411</td><td>0.0646</td><td>0.0678</td><td>0.0690</td></tr><tr><td>GenCAR anchor CF fine-tune</td><td>0.0409</td><td>0.0628</td><td>0.0677</td><td>0.0686</td></tr><tr><td>GenCAR + rerank</td><td>0.0449</td><td>0.0661</td><td>0.0725</td><td>0.0719</td></tr></table>

## Table 6

Model size and online serving cost on Coat. One-time ofline LLM counterfactual generation of GenCAR is excluded from inference latency. BB denotes the CausalVAE backbone; M denotes millions of parameters; CF denotes counterfactual.
<table><tr><td>Method</td><td>Params (M)</td><td>Training</td><td>Inference LLM (min/seed) (ms/user) Usage</td><td></td></tr><tr><td>BPRMF DICE/InvCF/CDR 0.08–0.12</td><td>0.04</td><td>2 8-12</td><td>&lt;1 &lt;1</td><td>X X</td></tr><tr><td>CausalDiffRec TallRec</td><td>0.85 ~3000</td><td>53 ~30</td><td>~35* ~5400†</td><td>X infer</td></tr><tr><td>CausalVAE (BB)</td><td>0.15</td><td>30</td><td>&lt;1</td><td>X</td></tr><tr><td>GenCAR</td><td>0.15</td><td>30 + offline</td><td>&lt; 1</td><td>Offline</td></tr></table>

<sup>∗</sup>50 denoising steps. <sup>†</sup>LLM inference cost. <sup>‡</sup>One-time ofline CF.

to the CausalVAE+BGE backbone (0.0409 versus 0.0411 in Recall@10). The auxiliary GenCAR + rerank variant explicitly retains available counterfactual items in the ranked list, raising Recall@10 to 0.0449 and NDCG@10 to 0.0725, gains of 9.2% and 6.9%. This variant measures partialcoverage utility; the standard pooled selector remains the certified output.

Ofline counterfactual generation preserves submillisecond online serving. Table 6 reports trainable parameters, training time per seed, per-user inference latency, and online LLM usage on Coat. GenCAR’s onetime generation stage is treated as an ofline cost, while TallRec’s inference-time LLM call is included in online latency. GenCAR and its CausalVAE backbone each contain 0.15 million trainable parameters and require less than one millisecond per user. TallRec requires approximately 5.4 seconds per user in the reported setup. GenCAR’s online path consists of user encoding, inner-product scoring, and calibrated selection. Thus, we add counterfactual supervision while retaining the backbone’s model size and sub-millisecond serving profile.

Together, these results define the deployment profile of GenCAR. Ofline LLM counterfactuals can cover a subset of users, and online serving remains LLM-free and submillisecond.

## 6. Related Work

Our work connects two lines of research: counterfactual candidate construction for OOD recommendation and predictive inference for risk-controlled selection. Table 7 compares the relevant method families in terms of OOD scope, candidate construction, certified target, and online LLM usage.

OOD and counterfactual recommendation. Outof-distribution recommendation seeks to preserve ranking utility when interaction distributions change between training and serving. Reviews organize causal recommendation through potential-outcome, structural causal model, and counterfactual formulations [28, 29]. Causal disentanglement and invariant learning separate preference-related factors from environmental correlations [6, 3, 7] [30, 31]. Distributionally robust objectives and causal difusion extend this direction to broader forms of distribution shift [2, 8]. Contrastive learning with uniform data further reduces exposure bias in learned representations [32]. These methods primarily reweight observed interactions or learn shift-robust representations. Item-level supervision for a specified environmental intervention remains outside their main objective.

Generative recommenders model complex interaction distributions through difusion and preference-aware objectives [9, 10, 33, 12] [34–37]. A recent review places these methods within the broader family of generative recommender systems [13]. Surveys of LLM-enhanced recommendation organize language models by their roles in feature construction, representation learning, ranking, and generation [38, 39]. Individual systems integrate collaborative semantics [40–42], external knowledge [43, 44], or LLM-based ranking [45, 46], further illustrating the role of language knowledge in recommendation [47] [48–50]. Their primary objectives are representation enrichment, ranking, or direct generation. Intervention-specific supervision constrained by stable preference remains a separate requirement. GenCAR uses an ofline LLM to construct item-level counterfactual pairs under an explicit environmental intervention. Stable-preference anchors and trust-radius filtering ground these proposals before ranker training.

Risk-controlled predictive inference. Predictive inference quantifies uncertainty and provides finite-sample guarantees for learned outputs [51]. Conformal prediction constructs distribution-free prediction sets [52, 53]. Conformal risk control extends calibration to the expected values of monotone losses [54], while conformal �-values support selection with FDR control [22, 55].

Recommendation-specific methods apply these guarantees to set selection. Angelopoulos et al. [5] transform pretrained ranker outputs into recommendation sets with finitesample FDR control. Conformal Alignment [56] selects foundation-model outputs using a user-defined alignment criterion. De Toni et al. [14] bound unwanted recommendations and expand selected sets using observed user feedback. Their guarantees apply to candidate families paired with observed relevance, alignment, or unwanted-content labels. GenCAR introduces intervention-specific supervision before ranking and then calibrates the ranked candidate family under proxy labels. The �-VCR formulation makes pooled proxy-label FDR the explicit serving constraint, connecting preference-grounded candidate recovery with calibrated set selection.

Table 7  
Positioning of GenCAR across OOD candidate construction and risk-controlled serving. A dash indicates no finite-sample guarantee for the served set. “Partial” means that only some methods in the family explicitly address distribution shift.
<table><tr><td>Family</td><td>OOD</td><td>Candidate construction</td><td>Certified target</td><td>Online LLM</td></tr><tr><td>Invariant and robust OOD recommendation [3, 7, 2, 8]</td><td>Yes</td><td>Observed interactions and invariant representations</td><td></td><td>No</td></tr><tr><td>Diffusion recommendation [9, 10, 33, 12]</td><td>Partial</td><td>Diffusion-generated interactions or sequences</td><td></td><td>No</td></tr><tr><td>LLM-augmented recommendation [45, 43, 47]</td><td>No</td><td>LLM knowledge or language representations</td><td></td><td>Varies</td></tr><tr><td>Reliable recommendation sets [5]</td><td>No</td><td>Fixed ranked candidates</td><td>Relevance FDR</td><td>No</td></tr><tr><td>Conformal Alignment [56]</td><td>No</td><td>Foundation-model outputs</td><td>Output-alignment FDR</td><td>Model- dependent</td></tr><tr><td>Conformal risk control for unwanted recommendations [14]</td><td>No</td><td>Ranked and previously consumed items</td><td>Expected unwanted fraction</td><td>No</td></tr><tr><td>GenCAR (ours)</td><td>Yes</td><td>Ranked candidates learned from environment-intervened, preference-grounded supervision</td><td>Pooled proxy-label FDR</td><td>No</td></tr></table>

## 7. Conclusion

In this paper, we presented GenCAR, a two-stage framework for preference-grounded counterfactual construction and risk-controlled selection in OOD recommendation. By formulating serving as the �-Valid Counterfactual Recommendation (�-VCR) problem, GenCAR connects environment-intervened candidate construction with FDR-calibrated selection. Theoretically, we bounded conditional counterfactual approximation error, established finite-sample proxy-label FDR control under target-environment exchangeability and PRDS, and characterized the remaining discrepancy between proxy-label and causal risk. Empirically, realized proxy FDP remained below � across five evaluated pools, and GenCAR improved top-10 OOD candidate recovery over the corresponding CausalVAE backbones under diverse shifts. Moving LLM-based counterfactual generation and proxy labeling ofline preserves a sub-millisecond, LLMfree online path. GenCAR places counterfactual OOD serving on an explicit statistical footing, with guarantees scoped to proxy labels and their transfer assumptions.

## CRediT authorship contribution statement

Qianqian Wang: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Writing – original draft. Yunshan Li: Writing – review & editing. Jiawen Zeng: Software, Validation. Wenwu Gong: Writing – review & editing, Methodology, Conceptualization. Lili Yang: Supervision, Resources, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This research was funded by the SUSTech Presidential Postdoctoral Fellowship, the China Postdoctoral Science Foundation (Grant No. 2025M773057), Shenzhen Science and Technology Program (Grant No. ZDSYS20210623092007023), and the Shenzhen Key Laboratory of Safety and Security for Next Generation of Industrial Internet, Southern University of Science and Technology.

## Data availability

The MovieLens, Coat, Amazon-Book, Amazon-Beauty, and Steam datasets used in this study are publicly available from the sources cited in Section 5.1. Dataset sources, access instructions, and licensing restrictions are documented at https://github.com/nikewq/GenCAR-release. The model code, training and evaluation pipelines, trained checkpoints, processed evaluation splits, and complete result artifacts will be released in the same repository after manuscript acceptance.

## Declaration of generative AI use

The authors used generative AI tools only for language polishing and readability improvement. All technical ideas, experiments, analyses, and conclusions were developed and verified by the authors, who take full responsibility for the content of this paper.

## A. Proofs of Theoretical Results

## A.1. Proof of Proposition 1

Proof. The triangle inequality for total variation gives

$$
\mathrm { T V } ( P ^ { \star } , P ^ { M } ) \leq \epsilon _ { A } + \epsilon _ { M } .
$$

Let $P = P ^ { \star } , Q = P ^ { M }$ , and $A = \mathcal { T } _ { u } ( \delta )$ . For any $B \subseteq { \mathcal { I } }$

$$
\begin{array} { r l } & { | P ( B \mid A ) - Q ( B \mid A ) | } \\ & { \leq \frac { \left| P ( B \cap A ) - Q ( B \cap A ) \right| } { P ( A ) } } \\ & { \quad + Q ( B \cap A ) \left| \frac { 1 } { P ( A ) } - \frac { 1 } { Q ( A ) } \right| } \\ & { \leq \frac { 2 \mathrm { T V } ( P , Q ) } { \rho _ { \delta } } . } \end{array}
$$

Taking the supremum over � and combining the two bounds proves the first claim of Proposition 1.

For the event $\mathcal { M } _ { u } .$

$$
\begin{array} { r l } & { \displaystyle { P _ { \delta } ^ { M } ( \mathcal { M } _ { u } ) \leq P _ { \delta } ^ { \star } ( \mathcal { M } _ { u } ) + \left| P _ { \delta } ^ { M } ( \mathcal { M } _ { u } ) - P _ { \delta } ^ { \star } ( \mathcal { M } _ { u } ) \right| } } \\ & { \quad \quad \leq \displaystyle { P _ { \delta } ^ { \star } ( \mathcal { M } _ { u } ) + \frac { 2 ( \epsilon _ { A } + \epsilon _ { M } ) } { \rho _ { \delta } } } . } \end{array}
$$

This proves the second claim of Proposition 1.

## A.2. Proof of Lemma 1

Proof. By construction, $S _ { k + 1 }$ contains all pairs in $S _ { k }$ and one additional pair with the next smallest �-value. Hence

$$
S _ { k } \subseteq S _ { k + 1 }
$$

for every $k \in \{ 0 , \ldots , N - 1 \}$ , which proves the nestedness claim of Lemma 1.

The index $\widehat { k } _ { \alpha }$ is defined as the largest index satisfying the BH inequality. Thus, $S _ { \widehat { k } _ { \alpha } }$ is the largest member of the nested family satisfying the step-up condition.

Let $\alpha _ { 1 } ~ \leq ~ \alpha _ { 2 }$ . Every index satisfying $\begin{array} { r } { p _ { ( k ) } \ \le \ \frac { \alpha _ { 1 } k } { N } } \end{array}$ also satisfies $\begin{array} { r } { p _ { ( k ) } \ \le \ \frac { \alpha _ { 2 } k } { N } } \end{array}$ . It follows that $\widehat { k } _ { \alpha _ { 1 } } \leq \widehat { k } _ { \alpha _ { 2 } }$ . Nestedness of $\{ S _ { k } \} _ { k = 0 } ^ { N } \mathrm { g i v e s } S _ { \alpha _ { 1 } } \subseteq S _ { \alpha _ { 2 } }$ □

## A.3. Proof of Lemma 2

Proof. Fix a proxy-null candidate $( u , j )$ and condition on the learned artifacts and candidate family. Let $R ^ { + }$ be the upper rank of $V ( u , j )$ among the $n _ { 0 }$ calibration scores and the test score. The weak-rank �-value in Eq. (7) satisfies

$$
p ( u , j ) = \frac { R ^ { + } } { n _ { 0 } + 1 } .
$$

Conditional on the multiset of the $n _ { 0 } + 1$ exchangeable scores, at most � indices have upper rank no greater than �. Exchangeability therefore gives

$$
\operatorname* { P r } \left( R ^ { + } \leq r \mid \hat { h } , \mathcal { Q } ( \mathcal { B } ) \right) \leq \frac { r } { n _ { 0 } + 1 }
$$

for every $r \in \{ 0 , \ldots , n _ { 0 } + 1 \}$ . Setting $r = \lfloor t ( n _ { 0 } + 1 ) \rfloor$ yields

$$
\operatorname* { P r } \left( p ( u , j ) \leq t \mid \hat { h } , { \mathcal { Q } } ( B ) \right) \leq t .
$$

Thus the implemented weak-rank �-value is super-uniform. Ties can only increase the upper rank and hence preserve the bound. □

## A.4. Proof of Theorem 1 and Corollary 1

Proof. Condition on the learned artifacts and the candidate family (). Lemma 2 establishes super-uniformity for every proxy-null �-value. Under conditional PRDS on the proxy-null hypotheses, the BH result of Benjamini and Yekutieli [57] gives

$$
\mathbb { E } \left[ \frac { | S _ { \alpha } ( B ) \cap \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } | } { \operatorname* { m a x } \{ | S _ { \alpha } ( B ) | , 1 \} } \Bigg | \hat { h } , Q ( B ) \right] \leq \frac { m _ { 0 } } { N } \alpha \leq \alpha .
$$

Taking expectations over the conditioning variables and applying the tower property gives

$$
\mathrm { F D R } _ { \mathrm { p r o x y } } \big ( S _ { \alpha } ( B ) \big ) \leq \alpha ,
$$

which proves Theorem 1.

Under arbitrary dependence, the Benjamini–Yekutieli procedure replaces � with $\alpha / H _ { N }$ . Applying its generaldependence result conditionally and then using the tower property yields

$$
\mathrm { F D R } _ { \mathrm { p r o x y } } \big ( S _ { \alpha } ^ { \mathrm { B Y } } ( { \boldsymbol { \it B } } ) \big ) \leq \alpha .
$$

This proves Corollary 1.

## A.5. Proof of Corollary 2

Proof. For any selected set ,

$$
\begin{array} { r l } & { \lvert S \cap \mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } \rvert \le \lvert S \cap \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } \rvert } \\ & { \qquad + \lvert S \cap ( \mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } \setminus \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } } ) \rvert . } \end{array}
$$

Dividing both sides by max $\{ | S | , 1 \}$ and taking expectations gives

$$
\mathrm { F D R } _ { \operatorname { c a u s a l } } ( S ) \leq \mathrm { F D R } _ { \operatorname { p r o x y } } ( S ) + \Delta _ { \operatorname { p r o x y } } ( S ) .
$$

Applying Theorem 1 with $S = S _ { \alpha } ( B )$ proves Corollary 2. If $\mathcal { H } _ { 0 } ^ { \mathrm { c a u s a l } } \subseteq \mathcal { H } _ { 0 } ^ { \mathrm { p r o x y } }$ <sup>y</sup>, then <sup>causal</sup> $\ ` \varkappa _ { 0 } ^ { \mathrm { p r o x y } } = \emptyset$ . Thus, 0   
$\Delta _ { \mathrm { p r o x y } } ( \breve { S } _ { \alpha } ( B ) ) = 0 .$ , which proves the final statement. □

## References

[1] T. Schnabel, A. Swaminathan, A. Singh, N. Chandak, T. Joachims, Recommendations as treatments: Debiasing learning and evaluation, in: M. F. Balcan, K. Q. Weinberger (Eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, PMLR, New York, New York, USA, 2016, pp. 1670–1679. URL: https://proceedings.mlr. press/v48/schnabel16.html.

[2] B. Wang, J. Chen, C. Li, S. Zhou, Q. Shi, Y. Gao, Y. Feng, C. Chen, C. Wang, Distributionally robust graph-based recommendation system, in: Proceedings of the ACM Web Conference 2024, WWW ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 3777–3788. URL: https://doi.org/10.1145/3589334.3645598. doi:10. 1145/3589334.3645598.

[3] A. Zhang, J. Zheng, X. Wang, Y. Yuan, T.-S. Chua, Invariant collaborative filtering to popularity distribution shift, in: Proceedings of the ACM Web Conference 2023, WWW ’23, Association for Computing Machinery, New York, NY, USA, 2023, p. 1240–1251. URL: https: //doi.org/10.1145/3543507.3583461. doi:10.1145/3543507.3583461.

[4] Z. Yang, X. He, J. Zhang, J. Wu, X. Xin, J. Chen, X. Wang, A generic learning framework for sequential recommendation with distribution shifts, in: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, Association for Computing Machinery, New York, NY, USA, 2023, p. 331–340. URL: https://doi.org/10.1145/3539618.3591624. doi:10.1145/3539618.3591624.

[5] A. N. Angelopoulos, K. Krauth, S. Bates, Y. Wang, M. I. Jordan, Recommendation systems with distribution-free reliability guarantees, in: H. Papadopoulos, K. A. Nguyen, H. Boström, L. Carlsson (Eds.), Proceedings of the Twelfth Symposium on Conformal and Probabilistic Prediction with Applications, volume 204 of Proceedings of Machine Learning Research, PMLR, 2023, pp. 175–193. URL: https://proceedings.mlr.press/v204/angelopoulos23a.html.

[6] Y. Zheng, C. Gao, X. Li, X. He, Y. Li, D. Jin, Disentangling user interest and conformity for recommendation with causal embedding, in: Proceedings of the Web Conference 2021, WWW ’21, Association for Computing Machinery, New York, NY, USA, 2021, p. 2980–2991. URL: https://doi.org/10.1145/3442381.3449788. doi:10. 1145/3442381.3449788.

[7] W. Wang, X. Lin, L. Wang, F. Feng, Y. Ma, T.-S. Chua, Causal disentangled recommendation against user preference shifts, ACM Trans. Inf. Syst. 42 (2023). URL: https://doi.org/10.1145/3593022. doi:10.1145/3593022.

[8] C. Zhao, E. Yang, Y. Liang, P. Lan, Y. Liu, J. Zhao, G. Guo, X. Wang, Graph representation learning via causal difusion for out-of-distribution recommendation, in: Proceedings of the ACM on Web Conference 2025, WWW ’25, Association for Computing Machinery, New York, NY, USA, 2025, p. 334–346. URL: https: //doi.org/10.1145/3696410.3714849. doi:10.1145/3696410.3714849.

[9] J. Zhao, W. Wenjie, Y. Xu, T. Sun, F. Feng, T.-S. Chua, Denoising difusion recommender model, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 1370–1379. URL: https: //doi.org/10.1145/3626772.3657825. doi:10.1145/3626772.3657825.

[10] Z. Yang, J. Wu, Z. Wang, X. Wang, Y. Yuan, X. He, Generate what you prefer: Reshaping sequential recommendation via guided difusion, in: A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, S. Levine (Eds.), Advances in Neural Information Processing Systems, volume 36, Curran Associates, Inc., 2023, pp. 24247– 24261. URL: https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/4c5e2bcbf21bdf40d75fddad0bd43dc9-Paper-Conference.pdf. doi:10.52202/075280-1054.

[11] Q. Liu, F. Yan, X. Zhao, Z. Du, H. Guo, R. Tang, F. Tian, Difusion augmentation for sequential recommendation, in: Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, CIKM ’23, Association for Computing Machinery, New York, NY, USA, 2023, p. 1576–1586. URL: https://doi.org/ 10.1145/3583780.3615134. doi:10.1145/3583780.3615134.

[12] S. Liu, A. Zhang, G. Hu, H. Qian, T.-S. Chua, Preference difusion for recommendation, in: Y. Yue, A. Garg, N. Peng, F. Sha, R. Yu (Eds.), International Conference on Learning Representations, volume 2025, 2025, pp. 79844–79881. URL: https://proceedings.iclr.cc/paper\_files/paper/2025/file/ c6989f4c36acb6f0e0fdd60f1c12e8a0-Paper-Conference.pdf.

[13] Y. Deldjoo, Z. He, J. McAuley, A. Korikov, S. Sanner, A. Ramisa, R. Vidal, M. Sathiamoorthy, A. Kasirzadeh, S. Milano, A review of modern recommender systems using generative models (gen-recsys), in: Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 6448–6458. URL: https: //doi.org/10.1145/3637528.3671474. doi:10.1145/3637528.3671474.

[14] G. De Toni, E. Purificato, E. Gomez, A. Passerini, B. Lepri, C. Consonni, You don’t bring me flowers: Mitigating unwanted recommendations through conformal risk control, in: Proceedings of the Nineteenth ACM Conference on Recommender Systems, RecSys ’25, Association for Computing Machinery, New York, NY, USA, 2025,

p. 492–502. URL: https://doi.org/10.1145/3705328.3748054. doi:10. 1145/3705328.3748054.

[15] F. M. Harper, J. A. Konstan, The movielens datasets: History and context, volume 5, Association for Computing Machinery, New York, NY, USA, 2015. URL: https://doi.org/10.1145/2827872. doi:10.1145/2827872.

[16] R. He, J. McAuley, Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering, in: Proceedings of the 25th International Conference on World Wide Web, WWW ’16, International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, CHE, 2016, p. 507–517. URL: https: //doi.org/10.1145/2872427.2883037. doi:10.1145/2872427.2883037.

[17] B. Schölkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal, Y. Bengio, Toward causal representation learning, Proceedings of the IEEE 109 (2021) 612–634. doi:10.1109/JPROC.2021. 3058954.

[18] I. Khemakhem, D. Kingma, R. Monti, A. Hyvarinen, Variational autoencoders and nonlinear ica: A unifying framework, in: S. Chiappa, R. Calandra (Eds.), Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings ofMachine Learning Research, PMLR, 2020, pp. 2207– 2217. URL: https://proceedings.mlr.press/v108/khemakhem20a.html.

[19] M. Arjovsky, L. Bottou, I. Gulrajani, D. Lopez-Paz, Invariant risk minimization, arXiv preprint arXiv:1907.02893 (2019).

[20] S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme, Bpr: Bayesian personalized ranking from implicit feedback, arXiv preprint arXiv:1205.2618 (2012).

[21] L. G. Neuberg, Causality: Models, reasoning, and inference, by judea pearl, cambridge university press, 2000, Econometric Theory 19 (2003) 675–685. doi:10.1017/S0266466603004109.

[22] S. Bates, E. Candès, L. Lei, Y. Romano, M. Sesia, Testing for outliers with conformal p-values, The Annals of Statistics 51 (2023) 149 – 178. URL: https://doi.org/10.1214/22-AOS2244. doi:10.1214/ 22-AOS2244.

[23] W.-C. Kang, J. McAuley, Self-attentive sequential recommendation, in: 2018 IEEE International Conference on Data Mining (ICDM), 2018, pp. 197–206. doi:10.1109/ICDM.2018.00035.

[24] S. Bonner, F. Vasile, Causal embeddings for recommendation, in: Proceedings of the 12th ACM Conference on Recommender Systems, RecSys ’18, Association for Computing Machinery, New York, NY, USA, 2018, p. 104–112. URL: https://doi.org/10.1145/3240323. 3240360. doi:10.1145/3240323.3240360.

[25] T. Wei, F. Feng, J. Chen, Z. Wu, J. Yi, X. He, Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system, in: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, Association for Computing Machinery, New York, NY, USA, 2021, p. 1791–1800. URL: https://doi.org/10.1145/3447548.3467289. doi:10. 1145/3447548.3467289.

[26] K. Bao, J. Zhang, Y. Zhang, W. Wang, F. Feng, X. He, Tallrec: An efective and eficient tuning framework to align large language model with recommendation, in: Proceedings of the 17th ACM Conference on Recommender Systems, RecSys ’23, Association for Computing Machinery, New York, NY, USA, 2023, p. 1007–1014. URL: https: //doi.org/10.1145/3604915.3608857. doi:10.1145/3604915.3608857.

[27] L. van der Maaten, G. Hinton, Visualizing data using t-sne, Journal of Machine Learning Research 9 (2008) 2579–2605. URL: http: //jmlr.org/papers/v9/vandermaaten08a.html.

[28] H. Luo, F. Zhuang, R. Xie, H. Zhu, D. Wang, Z. An, Y. Xu, A survey on causal inference for recommendation, The Innovation 5 (2024) 100590. URL: https://doi.org/10.1016/j.xinn.2024.100590. doi:10.1016/j.xinn.2024.100590.

[29] W. Wang, Y. Zhang, H. Li, P. Wu, F. Feng, X. He, Causal recommendation: Progresses and future directions, in: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, Association for Computing Machinery, New York, NY, USA, 2023, p. 3432–3435. URL: https: //doi.org/10.1145/3539618.3594245. doi:10.1145/3539618.3594245.

[30] H. Yoo, R. Qiu, C. Xu, F. Wang, H. Tong, Generalizable recommender system during temporal popularity distribution shifts, in: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, KDD ’25, Association for Computing Machinery, New York, NY, USA, 2025, p. 1833–1843. URL: https://doi.org/ 10.1145/3690624.3709299. doi:10.1145/3690624.3709299.

[31] Y. Liao, Y. Yang, M. Hou, L. Wu, H. Xu, H. Liu, Mitigating distribution shifts in sequential recommendation: An invariance perspective, in: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’25, Association for Computing Machinery, New York, NY, USA, 2025, p. 1603–1613. URL: https://doi.org/10.1145/3726302.3730036. doi:10. 1145/3726302.3730036.

[32] X. Yang, Z. Liu, X. Lu, Y. Yuan, S. Lu, Y. Gao, Contrastive disentangled representation learning for debiasing recommendation with uniform data, in: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 4188–4192. URL: https://doi.org/10.1145/3627673.3679889. doi:10. 1145/3627673.3679889.

[33] Z. Li, A. Sun, C. Li, Difurec: A difusion model for sequential recommendation, ACM Trans. Inf. Syst. 42 (2023). URL: https: //doi.org/10.1145/3631116. doi:10.1145/3631116.

[34] S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost, M. Kula, E. Chi, M. Sathiamoorthy, Recommender systems with generative retrieval, in: A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, S. Levine (Eds.), Advances in Neural Information Processing Systems, volume 36, Curran Associates, Inc., 2023, pp. 10299– 10315. URL: https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf. doi:10.52202/075280-0452.

[35] H. Ma, R. Xie, L. Meng, X. Chen, X. Zhang, L. Lin, Z. Kang, Plugin difusion model for sequential recommendation, Proceedings of the AAAI Conference on Artificial Intelligence 38 (2024) 8886–8894. doi:10.1609/aaai.v38i8.28736.

[36] J. Zhai, L. Liao, X. Liu, Y. Wang, R. Li, X. Cao, L. Gao, Z. Gong, F. Gu, J. He, Y. Lu, Y. Shi, Actions speak louder than words: Trillionparameter sequential transducers for generative recommendations, in: R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, F. Berkenkamp (Eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, PMLR, 2024, pp. 58484–58509. URL: https://proceedings.mlr.press/v235/zhai24a.html.

[37] Z. Cai, S. Wang, V. W. Chu, U. Naseem, Y. Wang, F. Chen, Unleashing the potential of difusion models towards diversified sequential recommendations, in: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’25, Association for Computing Machinery, New York, NY, USA, 2025, p. 1476–1486. URL: https://doi.org/10.1145/ 3726302.3730109. doi:10.1145/3726302.3730109.

[38] J. Lin, X. Dai, Y. Xi, W. Liu, B. Chen, H. Zhang, Y. Liu, C. Wu, X. Li, C. Zhu, H. Guo, Y. Yu, R. Tang, W. Zhang, How can recommender systems benefit from large language models: A survey, ACM Trans. Inf. Syst. 43 (2025). URL: https://doi.org/10.1145/3678004. doi:10. 1145/3678004.

[39] Y. Yao, X. Zheng, Heterogeneous graph learning for code reviewers recommendation with llm’s understanding (2025) 928–932. doi:10. 1109/CAIBDA65784.2025.11183380.

[40] Y. Zhang, F. Feng, J. Zhang, K. Bao, Q. Wang, X. He, Collm: Integrating collaborative embeddings into large language models for recommendation, IEEE Transactions on Knowledge and Data Engineering 37 (2025) 2329–2340. doi:10.1109/TKDE.2025.3540912.

[41] B. Zheng, Y. Hou, H. Lu, Y. Chen, W. X. Zhao, M. Chen, J.-R. Wen, Adapting large language models by integrating collaborative semantics for recommendation, in: 2024 IEEE 40th International Conference on Data Engineering (ICDE), 2024, pp. 1435–1448. doi:10.1109/ICDE60146.2024.00118.

[42] Z. Liu, H. Zhang, K. Dong, Y. Fang, Collaborative cross-modal fusion with large language model for recommendation, in: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 1565–1574. URL: https: //doi.org/10.1145/3627673.3679596. doi:10.1145/3627673.3679596.

[43] Y. Xi, W. Liu, J. Lin, X. Cai, H. Zhu, J. Zhu, B. Chen, R. Tang, W. Zhang, Y. Yu, Towards open-world recommendation with knowledge augmentation from large language models, in: Proceedings of the 18th ACM Conference on Recommender Systems, RecSys ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 12–22. URL: https://doi.org/10.1145/3640457.3688104. doi:10.1145/ 3640457.3688104.

[44] Q. Zhao, H. Qian, Z. Liu, G.-D. Zhang, L. Gu, Breaking the barrier: Utilizing large language models for industrial recommendation systems through an inferential knowledge graph, in: Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 5086–5093. URL: https://doi.org/ 10.1145/3627673.3680022. doi:10.1145/3627673.3680022.

[45] J. Liao, S. Li, Z. Yang, J. Wu, Y. Yuan, X. Wang, X. He, Llara: Large language-recommendation assistant, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 1785–1795. URL: https: //doi.org/10.1145/3626772.3657690. doi:10.1145/3626772.3657690.

[46] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, W. X. Zhao, Large language models are zero-shot rankers for recommender systems, in: N. Goharian, N. Tonellotto, Y. He, A. Lipani, G. McDonald, C. Macdonald, I. Ounis (Eds.), Advances in Information Retrieval, Springer Nature Switzerland, Cham, 2024, pp. 364–381.

[47] L. Sheng, A. Zhang, Y. Zhang, Y. Chen, X. Wang, T.-S. Chua, Language representations can be what recommenders need: Findings and potentials, in: Y. Yue, A. Garg, N. Peng, F. Sha, R. Yu (Eds.), International Conference on Learning Representations, volume 2025, 2025, pp. 91632–91658. URL: https://proceedings.iclr.cc/paper\_files/paper/2025/file/ e4bab1843c8d5a69f5abfd0824593493-Paper-Conference.pdf.

[48] W. Wei, X. Ren, J. Tang, Q. Wang, L. Su, S. Cheng, J. Wang, D. Yin, C. Huang, Llmrec: Large language models with graph augmentation for recommendation, in: Proceedings of the 17th ACM International Conference on Web Search and Data Mining, WSDM ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 806–815. URL: https://doi.org/10.1145/3616855.3635853. doi:10.1145/3616855.3635853.

[49] S. Kim, H. Kang, S. Choi, D. Kim, M. Yang, C. Park, Large language models meet collaborative filtering: An eficient all-round llm-based recommender system, in: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 1395–1406. URL: https://doi.org/10.1145/3637528.3671931. doi:10. 1145/3637528.3671931.

[50] J. Lin, R. Shan, C. Zhu, K. Du, B. Chen, S. Quan, R. Tang, Y. Yu, W. Zhang, Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension in recommendation, in: Proceedings of the ACM Web Conference 2024, WWW ’24, Association for Computing Machinery, New York, NY, USA, 2024, p. 3497–3508. URL: https://doi.org/10.1145/3589334.3645467. doi:10. 1145/3589334.3645467.

[51] D. Prinster, S. Stanton, A. Liu, S. Saria, Conformal validity guarantees exist for any data distribution (and how to find them), in: Proceedings of the 41st International Conference on Machine Learning, ICML’24, JMLR.org, 2024.

[52] G. Shafer, V. Vovk, A Tutorial on Conformal Prediction, volume 9, JMLR.org, 2008.

[53] A. N. Angelopoulos, S. Bates, Conformal prediction: A gentle introduction, Foundations and Trends in Machine Learning 16 (2023) 494–591. URL: https:

//doi.org/10.1561/2200000101. doi:10.1561/2200000101. arXiv:https://www.emerald.com/ftmal/article-pdf/16/4/494/11155920/2200000101en.pdf.

[54] A. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, Conformal risk control, in: B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, Y. Sun (Eds.), International Conference on Learning Representations, volume 2024, 2024, pp. 55198– 55218. URL: https://proceedings.iclr.cc/paper\_files/paper/2024/ file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf.

[55] Y. Jin, E. J. Candes, Selection by prediction with conformal pvalues, Journal of Machine Learning Research 24 (2023) 1–41. URL: http://jmlr.org/papers/v24/22-1176.html.

[56] Y. Gui, Y. Jin, Z. Ren, Conformal alignment: Knowing when to trust foundation models with guarantees, in: A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, C. Zhang (Eds.), Advances in Neural Information Processing Systems, volume 37, Curran Associates, Inc., 2024, pp. 73884–73919. URL: https://proceedings.neurips.cc/paper\_files/paper/2024/ file/870ccde24673d3970a680bb48496ed63-Paper-Conference.pdf. doi:10.52202/079017-2350.

[57] Y. Benjamini, D. Yekutieli, The control of the false discovery rate in multiple testing under dependency, The Annals of Statistics 29 (2001) 1165–1188. URL: http://www.jstor.org/stable/2674075.