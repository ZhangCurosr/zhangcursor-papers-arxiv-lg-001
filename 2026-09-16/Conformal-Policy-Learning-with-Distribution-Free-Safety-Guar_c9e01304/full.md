# Conformal Policy Learning with Distribution-Free Safety Guarantees

Ying Jin<sup>∗1</sup> and Naoki Egami<sup>2</sup>

<sup>1</sup>Department of Statistics and Data Science, University of Pennsylvania <sup>2</sup>Department of Political Science & Statistics and Data Science Center, Massachusetts Institute of Technology

## Abstract

Policy learning aims to determine who should be treated based on individual characteristics. In high-stakes settings such as medicine and public policy where safety is a central concern, improving the average outcomes alone may not be suficient: decision makers may also seek to protect individuals from harm, in line with the Hippocratic principle of “do no harm.”

In this paper, we propose conformal policy learning (CPL), a policy learning procedure with a new distribution-free safety guarantee that controls the probability of assigning treatment to an individual who would be harmed relative to control. CPL views each treatment decision as testing a hypothesis of counterfactual harm and assigns treatment by thresholding conformal p-values. These p-values use observable proxies and selective calibration to address the challenge that the potential outcomes under comparison are never simultaneously observed. For randomized experiments, under standard exchangeability conditions, CPL provides finite-sample safety guarantee at a user-specified level, without imposing any outcome modeling assumptions. Moreover, when the outcome model is consistently estimated, CPL achieves asymptotically optimal welfare subject to the safety constraint. In observational studies, CPL with learn-then-balance weights achieves doubly robust safety guarantees. We evaluate CPL through extensive simulations and apply it to an empirical study of AI-powered interventions designed to reduce conspiracy beliefs.

Keywords: AI safety, Causal inference, Conformal prediction, Policy learning

## 1 Introduction

Policy learning, also known as the treatment choice problem, aims to learn a rule that automatically assigns future treatment options based on individual characteristics (Manski, 2004; Hirano and Porter, 2009; Kitagawa and Tetenov, 2018; Athey and Wager, 2021). It has been the foundation for data-driven decision-making in various domains spanning precision medicine (Murphy, 2003; Qian and Murphy, 2011; Zhao et al., 2012), online advertising and recommendation systems (Li et al., 2010; Dud´ık et al., 2011), political campaigns (Imai and Strauss, 2011), and criminal justice (Kleinberg et al., 2018), among others.

Policy learning methods often aim to maximize the average welfare (expectation of the realized outcome) within a policy class (Manski, 2004; Kitagawa and Tetenov, 2018; Athey and Wager, 2021). While welfare maximization is widely useful, it can be insuficient in high-stakes domains such as medicine and public policy where individual safety is of concern. In such applications, decision makers may seek not only on-average improvement of the outcomes, but also controlling the number of individuals harmed by the intervention (Gadbury et al., 2004; Kallus, 2022; Richens et al., 2022; Ben-Michael et al., 2025), i.e., “do no harm.” As an example, suppose a new intervention benefits 60% of a group while harming the remaining

40% by the same magnitude. A decision-maker who maximizes average welfare would decide to treat the group; this would harm 40% of the population, which can be unacceptable if individual safety is of primary concern.

In this paper, we study policy learning with a distribution-free safety guarantee. Formally, assume access to observed data $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = } ^ { n }$ where $Y _ { i } \in \{ 0 , 1 \}$ is a binary outcome, $T _ { i } \in \{ 0 , 1 \}$ is the binary treatment, and $X _ { i } \in { \mathcal { X } }$ is the observed features for each unit i. Under the potential outcome framework with SUTVA (formalized in Section 2.1), the outcome is $Y _ { i } = Y _ { i } ( T _ { i } )$ , where $( Y _ { i } ( 1 ) , Y _ { i } ( 0 ) )$ are the potential outcomes under treatment and control, respectively. For a new test point with observed features $X _ { n + 1 }$ and unknown potential outcomes $( Y _ { n + 1 } ( 0 ) , Y _ { n + 1 } ( 1 ) )$ , our goal is to learn a policy ${ \hat { \pi } } : { \mathcal { X } }  \{ 0 , 1 \}$ that maps features to treatment assignments with the following safety guarantee: for a pre-specified level $\alpha \in ( 0 , 1 )$ 2

$$
\mathbb { P } \big ( Y _ { n + 1 } ( \hat { \pi } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) \big ) \leq \alpha .\tag{1.1}
$$

This safety guarantee (1.1) means the probability of the realized outcome being worse than the “status-quo” outcome under control is no greater than α, thereby limiting the risk of harming the new individual. When this policy is implemented on m new individuals, (1.1) implies that the expected number of units harmed by the treatment is no greater than mα.

Such guarantees are important for two related reasons. First, negative treatment efects are substantively costly in many settings. Second, treatment rules are increasingly embedded in (semi-)automated decision systems: once a learned rule is deployed, it may assign interventions repeatedly and at scale. In such settings, trust in the system requires explicit control of the probability that it harms the individuals it chooses to treat. We highlight three applications that motivate such explicit harm-rate control.

Example 1 (AI-powered Interventions): Generative AI is emerging as a new class of intervention in the social sciences, with applications designed to change attitudes and behaviors through scalable, personalized interactions (e.g., Costello et al., 2024; Bai et al., 2025). At the same time, recent empirical studies highlight an important risk: while such AI interventions may benefit many individuals and tasks, they may also harm others. For example, a randomized experiment in a global consulting firm (Dell’Acqua et al., 2026) found that access to AI can harm the productivity of high-skilled consultants when working on challenging intellectual tasks. Similarly, Bastani et al. (2025) found that generative AI without guardrails can harm the learning of high school math students. As interventions are powered by black-box AI, controlling the harm is fundamental for safety, public trust, and eficient deployment of AI-powered treatment.

Example 2 (Precision Medicine): Precision medicine—prevention and treatment strategies that account for individual variability—is central in modern medicine (Kosorok and Laber, 2019). The problem of controlling individual risk while maximizing benefit has long been recognized. It is especially relevant when eficacious medications may also lead to a higher risk for certain individuals, such as opioid treatment of chronic pain (Laber et al., 2018) and type-2 diabetes with insulin therapies (Wang et al., 2018).

Example 3 (Criminal Justice): In the US criminal justice system, how to safely use risk scores to evaluate an individual’s likelihood of reofending and identify their criminogenic needs is a major topic of interest (Skeem et al., 2020). For example, Ben-Michael et al. (2025) analyze a field experiment to estimate the causal efect of algorithmic recommendations on judges’ decisions at a criminal first appearance hearing. Here, a treatment rule with the safety guarantee can prevent arrestees from committing a new crime or failing to appear in court, while avoiding unnecessarily harsh decisions.

We aim to achieve the safety guarantee in a model-agnostic fashion—meaning that it holds without strong modeling assumptions on the data distribution or the learning algorithms—and tightly in finite samples, so this framework is widely applicable to various high-stakes settings.

## 1.1 Overview of contributions

We develop conformal policy learning (CPL) to achieve the safety guarantee (1.1). Distinct from standard policy learning methods that maximizes empirical welfare within a policy class, our starting point is to view

![](images/43d248a7315ebb69bfc1c355c9e30c4b9bba1fd2f49ceb74debc24292740c139.jpg)  
Figure 1: Workflow of Conformal Policy Learning (CPL). Given labeled data $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , the goal is to assign safe treatment for a new test unit $X _ { n + 1 }$ with harm rate target $\alpha \in ( 0 , 1 )$ . CPL constructs proxy label $Y _ { i } ^ { \dagger }$ and selects a subset of data for calibration. It then computes a p-value by comparing a test conformity score to the calibration scores. The treatment is determined by whether the conformal p-value is below α, which controls the harm rate below α.

<sup>\$</sup> <sup>\$ \$</sup>    <sup>\$</sup>   <sup>\$</sup>each treatment decision as testing a counterfactual harm event $Y _ { n + 1 } ( 1 ) < Y _ { n + 1 } ( 0 )$ <sup>ective</sup> . A p-value $p _ { n + 1 }$ satisfying

$$
\mathbb { P } \big ( Y _ { n + 1 } ( 1 ) < Y _ { n + 1 } ( 0 ) , p _ { n + 1 } \leq \alpha \big ) \leq \alpha
$$

yields the desired safety guarantee by assigning treatment only when $p _ { n + 1 } \leq \alpha$ . This connects safe policy learning to conformal inference (Vovk et al., 2005; Lei and Cand\`es, 2021; Jin and Cand\`es, 2023b).

The key challenge here—which necessitates novel constructions of powerful conformal p-values—is that the “harm” event involves both potential outcomes and is therefore unobserved even for the labeled data. CPL addresses this dificulty through two techniques. First, it uses observable proxy labels to construct valid conformal p-values. Second, to reduce conservativeness, it selects the labeled observations used in pvalue calibration according to which treatment arm provides a sharper proxy for harm. We call this strategy “selective calibration.” The resulting p-value compares the test conformity score with the selected calibration scores computed using the proxy labels. The CPL workflow is in Figure 1.

For randomized experiments, CPL provides finite-sample, distribution-free safety guarantees. Under the exchangeability conditions ensured by random assignment, we show that CPL controls the harm rate exactly below α in finite samples, without making any assumption about the outcome models used in the conformal p-values and the selective calibration rule. We first establish this result for balanced randomized experiments and then extend it to stratified experiments using weighted conformal inference (Tibshirani et al., 2019).

We next study the sharpness and optimality of CPL. Because individual-level harm, $Y ( 1 ) < Y ( 0 )$ , is not identifiable from observed data, we formulate the optimality under partial identification (Kallus, 2022; Li et al., 2023). We characterize the optimal policy that maximizes the power (probability of treatment) and average welfare subject to the safety constraint for every joint potential-outcome distribution compatible with the observed data. Then, we show that as long as the score function used in the p-values and selective calibration rule converge to oracle ones, CPL attains these global optima asymptotically.

We further extend CPL to observational studies. In this setting, the treatment assignment mechanism is unknown, and the calibration weights must be estimated. We develop a learn-then-balance procedure that estimates these weights and enforces balance on functions tailored to the thresholding decisions and the partially identified harm target. Under suitable regularity conditions, the resulting policy has a doubly robust asymptotic safety guarantee: the excess harm rate vanishes when either the propensity-score model or the outcome models, but not necessarily both, are consistently estimated. Moreover, if both components converge at standard slow nonparametric rates, the excess harm rate is of a parametric order. With consistent outcome models, CPL with observational data attains the same welfare and power optima as in the randomized case. Our excess harm rate bound does not pay the price of policy class complexity common in policy learning.

Finally, we evaluate CPL through simulations and an empirical study. In simulations across randomized experiments, stratified experiments, and observational studies, CPL shows robust harm-rate control and high power and welfare. In the empirical study, CPL assigns an AI-powered intervention designed to reduce conspiracy beliefs while tightly controlling the harm rate.

The rest of the paper is organized as follows. Section 2 introduces the problem setup and connects the safety guarantee to conformal hypothesis testing. Section 3 develops CPL for balanced randomized experiments and studies its finite-sample validity and asymptotic optimality; the framework is then extended to stratified experiments in Section 4 and observational studies in Section 5. Section 6 presents simulation studies, and Section 7 applies CPL to an empirical study of AI-powered interventions. We close the paper with a discussion on extensions and future directions in Section 8.

## 1.2 Related Work

This article lies in the intersection of causal inference, policy learning, and conformal inference. We summarize several important lines of related work below.

Our work is connected to the established literature on policy learning (e.g., Murphy, 2003; Manski, 2004; Hirano and Porter, 2009; Zhao et al., 2012; Kitagawa and Tetenov, 2018; Athey and Wager, 2021; Jin et al., 2025b), which aims to select, among a given class of policies, the one that maximizes an objective (such as average welfare) while optionally respecting a constraint (such as harm rate). Within this literature, this work is closely related to a small but emerging literature on policy learning with safety considerations (where the exact meaning of safety varies). One line of work develops safe policy learning algorithms in settings that necessitate extrapolating beyond the observed labeled data, and the safety refers to not performing worse than a “status quo” policy (Zhang et al., 2022; Ben-Michael et al., 2025; Jia et al., 2025; Wu et al., 2025). Our safety notion of individual harm is conceptually related since it measures the harm relative to the status quo of no treatment, but distinct enough to yield completely diferent techniques. In addition, Ben-Michael et al. (2024) studies policy learning when the objective involves counterfactuals, providing doubly robust algorithms and regret bounds for the learned policy; the dual form of a special case in their framework (with an unknown Lagrange parameter) coincides with the welfare maximization problem subject to harm rate control; this connects with our setting and the method of Li et al. (2023). As standard in policy learning, these methods select a policy that maximizes an empirical objective within a policy class, whose performance is often measured by the regret (the gap between the true objective of the learned policy from the optimal), which is typically bounded by a statistical error term that scales with the complexity of the policy class (such as the VC-dimension). In contrast, CPL leverages conformal inference to achieve finite-sample, distribution-free safety guarantee (without a high-probability excess error bound term) when propensity scores are known; moreover, when propensity scores are unknown and estimated so inexact harm control is inevitable, our excess harm rate bound does not pay the price of the policy class complexity.

Our safety notion follows from a literature in causal inference and policy learning that bounds, estimates, and controls the same harm rate notion as (2.1). A line of work studies its bounds under various assumptions such as monotonicity (Huang et al., 2012) and certain conditional independence conditions (Shen et al., 2013; Yin et al., 2018). Relatedly, due to the non-identifiability, several work focuses on establishing bounds on the conditional harm rate for binary outcomes, including Gadbury et al. (2004) without covariates, Zhang et al. (2013) with covariates, Kallus (2022) on the sharp identification bounds, and Wu et al. (2024) using a sensitivity model for the correlation between the potential outcomes. In addition, a recent independent work of Scauda et al. (2026) studies the population-level optimal welfare subject to harm rate control. Since we aim to control the harm rate without strong modeling assumptions, this work is implicitly tied to the Frech´et–Hoefding bound characterized in Zhang et al. (2013); Kallus (2022). We show CPL attains the optimal welfare subject to safety guarantee among the worst-case distributions in these work.

Our method builds on the conformal p-values proposed in Jin and Cand\`es (2023b) for i.i.d. data and Jin and Cand\`es (2023a) for covariate shift settings, which were extended to model selection (Bai and Jin, 2024) and online settings (Xu and Ramdas, 2024). In that literature, the p-values quantify the confidence in a large, ordinary outcome (i.e., no potential outcomes) exceeding a known threshold. While we borrow the high-level intuitions, the technical route in constructing such p-values for our problem—which is the key contribution here—is sharply distinct since the two outcomes under comparison are never simultaneously observed. The resulting optimality and robustness properties are likewise quite diferent.

The conformal inference approach also connects our work with a line of work on conformal inference for individual treatment efects (ITE) $Y _ { n + 1 } ( 1 ) - Y _ { n + 1 } ( 0 )$ (Lei and Cand\`es, 2021; Jin et al., 2023; Yin et al.,

2024). These work typically focuses on constructing a prediction set for the counterfactual outcome for a unit who has already received treatment or control, where inference for the ITE of a new test point with two unknown outcomes appears particularly challenging. The latter case (with binary outcomes) is the setting we address, and we study the “decision” problem rather than prediction set construction. This involves a thresholding decision rule which necessitates distinct calibration and theoretical analysis techniques.

## 2 Problem Setup and Conceptual Framework

## 2.1 Problem Setup

We assume access to a set of labeled data $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , where $X _ { i } ~ \in ~ { \mathcal { X } }$ is the feature, $T _ { i } ~ \in ~ \{ 0 , 1 \}$ is the treatment assignment, and ${ Y _ { i } } ~ \in ~ \{ 0 , 1 \}$ is the binary outcome. We define the potential outcomes $Y _ { i } ( t )$ for $t \in \{ 0 , 1 \}$ and assume the triplets $\{ ( X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) ) \} _ { i = 1 } ^ { n }$ are independent and identically distributed (i.i.d.) from an unknown super-population $\mathbb { P } _ { X , Y ( 1 ) , Y ( 0 ) }$ . Assuming the Stable Unit Treatment Value Assumption (SUTVA) (Rubin, 1980), the observed outcome is given by $Y _ { i } = Y _ { i } ( T _ { i } )$ The joint super-population $\mathbb { P } _ { X , Y ( 1 ) , Y ( 0 ) }$ is unidentifiable from data because researchers observe only one potential outcome for each unit (Holland, 1986). Finally, the distribution of the labeled data $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ is induced by the unknown super-population and the treatment assignment mechanism and denoted as $\mathbb { P } _ { X , Y , T }$

Throughout, we make the unconfoundedness assumption for the treatment assignment mechanism, which is standard in the causal inference literature (Imbens and Rubin, 2015).

Assumption 2.1 (Unconfoundedness). The treatment assignments $\{ T _ { i } \} _ { i = 1 } ^ { n }$ <sub>1</sub> are mutually independent and obey $T _ { i } \perp \perp \left( Y _ { i } ( 0 ) , Y _ { i } ( 1 ) \right) | X _ { i }$ , with $\mathbb { P } ( T _ { i } = 1 | X _ { i } = x ) : = e ( x )$ , and we call $e ( x ) \in ( 0 , 1 )$ the propensity score.

Our framework covers both randomized experiments (Sections 3 and 4) and observational studies (Section 5). In randomized experiments, Assumption 2.1 is satisfied by design and $e ( x )$ is known. In observational studies, Assumption 2.1 should be evaluated based on domain knowledge; even though it holds, the propensity score $e ( x )$ is unknown and needs to be estimated.

Policy learning uses the labeled data to determine the treatment for a new individual (which we call the test unit/point) with observed feature $X _ { n + 1 }$ and unobserved potential outcomes $Y _ { n + 1 } ( 1 )$ and $Y _ { n + 1 } ( 0 )$ Following the literature (e.g., Manski, 2004; Hirano and Porter, 2009; Zhao et al., 2012; Kitagawa and Tetenov, 2018; Athey and Wager, 2021), we assume it is from the same super-population independently of the labeled data.

Assumption 2.2 (IID). The test point is drawn from $( X _ { n + 1 } , Y _ { n + 1 } ( 1 ) , Y _ { n + 1 } ( 0 ) ) \sim \mathbb { P } _ { X , Y ( 1 ) , Y ( 0 ) }$ independently of the labeled data.

Our framework can be naturally extended to relax the i.i.d. assumption and allow for covariate shift between the labeled data $i \in \{ 1 , \ldots , n \}$ and the test point; see Section 8 for a discussion.

## 2.2 Harm Rate

Recall that our goal is to develop a policy learning algorithm producing a rule ${ \hat { \pi } } : { \mathcal { X } }  \{ 0 , 1 \}$ that maps features X to treatment assignment ${ \hat { \pi } } ( X )$ with the counterfactual safety guarantee:

$$
\mathbb { P } \big ( Y _ { n + 1 } ( \hat { \pi } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) \big ) \leq \alpha\tag{2.1}
$$

for a pre-specified confidence level $\alpha \in ( 0 , 1 )$ . Here, the probability is over the labeled data (based on which ˆπ is built) and the test point. We call the left-handed side probability in (2.1) the “harm $\mathrm { r a t e } ^ { \mathrm { 7 } }$ following Zhang et al. (2013); the same quantity is also referred to as the “fraction of negatively afected” in the literature (Kallus, 2022; Li et al., 2023; Wu et al., 2024). Throughout, we focus on binary outcomes; this setting is most extensively studied (Zhang et al., 2013; Kallus, 2022; Wu et al., 2024).

The safety guarantees can be rewritten as

$$
\begin{array} { r } { \mathbb { P } \big ( Y _ { n + 1 } ( \hat { \pi } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) \big ) \leq \alpha \iff \mathbb { P } \big ( Y _ { n + 1 } ( 1 ) < Y _ { n + 1 } ( 0 ) , \hat { \pi } ( X _ { n + 1 } ) = 1 \big ) \leq \alpha . } \end{array}\tag{2.2}
$$

Thus, harm occurs when the individual treatment efect on the test unit is negative and one decides to treat the unit. Thus, a trivial way to achieve this is to treat the test unit with probability $\alpha ;$ however, such a policy is clearly suboptimal due to limited welfare and power (whose meaning will be made precise later). Instead, we aim for a procedure that achieves this with reasonable, and sometimes optimal, welfare and power. Meanwhile, the above result shows that to control the harm rate, we should detect and avoid treating units that can be harmed by the treatment. This is the intuition for CPL.

## 2.3 Connecting Safety Guarantees to Hypothesis Testing

Our technical route difers from standard policy learning approaches, so it is helpful to begin with the conceptual framework: connect the safety guarantees with hypothesis testing.

Equation (2.2) suggest that to make safe treatments, one needs to identify units that are likely not to be harmed by the treatment, i.e., $Y _ { n + 1 } ( 1 ) \ge Y _ { n + 1 } ( 0 )$ . We view this as testing a random null hypothesis $H _ { 0 } \colon Y _ { n + 1 } ( 1 ) < Y _ { n + 1 } ( 0 )$ . Suppose we can construct a p-value $p _ { n + 1 }$ such that

$$
\mathbb { P } \big ( Y _ { n + 1 } ( 1 ) < Y _ { n + 1 } ( 0 ) , \ p _ { n + 1 } \leq \alpha \big ) \leq \alpha .\tag{2.3}
$$

This is a non-conventional definition of p-values because the truth of $H _ { 0 }$ is itself random, but this notion is suficient for the desired safety guarantee: setting $\hat { \pi } ( X _ { n + 1 } ) = \mathbb { 1 } \{ p _ { n + 1 } \leq \alpha \}$ , the validity (2.3) directly implies the safety guarantee (2.1) via the equivalence relationship in (2.2). Intuitively, a small p-value informs strong evidence against the null, i.e., the test unit is unlikely to be harmed and thus can safely receive the treatment.

The remaining of the paper focuses on addressing two questions. First, how can we construct p-values that satisfy (2.3) in finite sample without making strong modeling assumptions? We address this question by expanding the recent literature of conformal p-values (Vovk et al., 2005; Bates et al., 2021; Jin and Cand\`es, 2023b) to causal inference contexts (Imbens and Rubin, 2015; Lei and Cand\`es, 2021), while respecting the partial-identification nature of the individual-level causal efects (e.g., Heckman et al., 1997). Second, is thresholding p-values optimal in any sense? This is important as it is not clear a priori why our approach may be desired, even if it might achieve the safety guarantee. We will show that with specifically designed p-values, our method achieves optimal power and welfare among all safe policies at the population level.

## 3 Conformal Policy Learning with Balanced Randomized Experiments

To fix ideas, in this section, we introduce our framework in the simple setting of balanced randomized experiments with $e ( x ) = 1 / 2  \mathrm { : }$ ; this will be gradually generalized in Sections 4 and 5. In Section 3.1, we start with a single-arm construction that provides distribution-free safety guarantees. In Section 3.2, we introduce observable proxies and selective calibration that sharpens the procedure while preserving the same guarantee. In Section 3.3, we characterize the welfare-optimal policy under partial identification, followed by an explanation on the sharpness of selective calibration in Section 3.4.

Throughout this section, all learned functions are fitted on a training sample independent of the labeled observations used for calibration and the test unit. In practice, this can be achieved by sample splitting (Lei et al., 2018). For notational simplicity, we use $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ to denote the labeled observations available for calibration; selective calibration introduced in Section 3.2 may retain only a subset of these observations. Conditional on the training sample, the learned functions are treated as fixed.

## 3.1 Warm-up: Direct Application of Existing Conformal p-values

We begin with a simple adaptation of the existing idea to show how conformal p-values can serve as the foundation for safe policy learning. Given that we focus on binary outcomes, we can rewrite (2.3) as finding

a p-value $p _ { n + 1 }$ obeying

$$
\mathbb { P } \big ( Y _ { n + 1 } ( 1 ) = 0 , Y _ { n + 1 } ( 0 ) = 1 , \ p _ { n + 1 } \leq \alpha \big ) \leq \alpha .\tag{3.1}
$$

That is, we would like to find p-values that quantify the confidence in a small treated outcome and a large control outcome. The conformal selection (CS) framework (Jin and Cand\`es, 2023b) provides a natural starting point. Given labeled data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ for an ordinary outcome $Y \in \mathbb { R } { \mathrm { ~ ( i . e . } }$ , no potential-outcome framing) and a known threshold $c \in \mathbb { R }$ , CS proposes conformal $\mathrm { p - }$ values with a similar null property $\mathbb { P } ( Y _ { n + 1 } \leq$ $c , p _ { n + 1 } \leq \alpha ) \leq \alpha$ . Setting aside the issue that we now have two potential outcomes to deal with, one simple idea is to develop a conservative p-value $p _ { n + 1 } ^ { ( 1 ) }$ that satisfy

$$
\mathbb { P } \big ( Y _ { n + 1 } ( 1 ) = 0 , \ p _ { n + 1 } ^ { ( 1 ) } \leq \alpha \big ) \leq \alpha ,\tag{3.2}
$$

which directly implies (3.1). For treated outcome, it means we will use the treated units for calibrating p-values. The conformal p-value from CS takes the form

$$
p _ { n + 1 } ^ { ( 1 ) } = \frac { 1 + \sum _ { i = 1 } ^ { n } T _ { i } \cdot \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ) \leq V ( X _ { n + 1 } , 0 ) \} } { 1 + \sum _ { i = 1 } ^ { n } T _ { i } } ,\tag{3.3}
$$

where V is any score function $V : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ obeying $V ( x , y ) \leq V ( x , y ^ { \prime } )$ whenever $y \le y ^ { \prime }$ for any $x \in \mathcal { X }$

Example (clipped score function). One example is the clipped score function (Jin and Cand\`es, 2023b), $V ( x , y ) = M \mathbb { 1 } \{ y > 0 \} + ( 1 - \hat { \mu } _ { 1 } ( x ) )$ , where ${ \hat { \mu } } _ { 1 } ( x )$ is an estimator for the conditional expectation function $\mathbb { E } [ Y _ { i } \mid T _ { i } = 1 , X _ { i } = x ]$ and $M > 2 \operatorname* { s u p } _ { x } | \hat { \mu } _ { 1 } ( x ) |$ is a suficiently large constant so that (3.3) reduces to

$$
p _ { n + 1 } ^ { ( 1 ) } = \frac { 1 + \sum _ { i = 1 } ^ { n } T _ { i } \cdot \mathbb { 1 } \{ Y _ { i } = 0 \} \cdot \mathbb { 1 } \left\{ \hat { \mu } _ { 1 } ( X _ { i } ) \geq \hat { \mu } _ { 1 } ( X _ { n + 1 } ) \right\} } { 1 + \sum _ { i = 1 } ^ { n } T _ { i } } .
$$

Intuitively, this p-value focuses on potentially unsafe cases $( \mathrm { i . e . , } Y _ { i } ( 1 ) = 0 )$ among treated units and examines whether the test unit is extreme with respect to ${ \hat { \mu } } _ { 1 } ( x )$ . When this p-value is small, the test unit is unlikely to come from the distribution of potentially unsafe units, and thus is likely to be safe. □

To see why the conformal p-value (3.3) satisfies (3.2) for any score function, we outline the theoretical arguments from Jin and Cand\`es (2023b). Let $n _ { 1 } = | \mathcal { T } _ { 1 } |$ be the number of treated units, where $\mathcal { T } _ { 1 } = \{ i \in$ $[ n ] \colon T _ { i } = 1 \}$ . Consider the “oracle” p-value

$$
p _ { n + 1 } ^ { * ( 1 ) } = \frac { 1 + \sum _ { i = 1 } ^ { n } T _ { i } \cdot \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ( 1 ) ) \leq V ( X _ { n + 1 } , Y _ { n + 1 } ( 1 ) ) \} } { 1 + n _ { 1 } }\tag{3.4}
$$

which is not computable since $Y _ { n + 1 } ( 1 )$ is unobserved. The only diference between (3.3) and (3.4) is that $Y _ { n + 1 } ( 1 )$ in the oracle p-value is replaced by 0, and we have $Y _ { i } ( 1 ) = Y _ { i }$ for $i \in \mathcal { T } _ { 1 }$ in (3.4). First, conditional on $\{ T _ { i } \} _ { i = 1 } ^ { n }$ , in this randomized experiment, the data $\{ ( X _ { i } , Y _ { i } ( 1 ) \} _ { i \in \mathbb { Z } _ { 1 } } \cup \{ X _ { n + 1 } , Y _ { n + 1 } ( 1 ) \}$ are exchangeable, which implies $\mathbb { P } ( p _ { n + 1 } ^ { * ( 1 ) } \leq \alpha ) \leq \alpha$ (Vovk et al., 2005). Second, on the event that $Y _ { n + 1 } ( 1 ) = 0$ , the two p-values coincide. The two facts imply $\mathbb { P } ( p _ { n + 1 } ^ { ( 1 ) } \le \alpha , \ Y _ { n + 1 } ( 1 ) = 0 ) = \mathbb { P } ( p _ { n + 1 } ^ { * ( 1 ) } \le \alpha , \ Y _ { n + 1 } ( 1 ) = 0 ) \le \mathbb { P } ( p _ { n + 1 } ^ { * ( 1 ) } \le \alpha ) \le \alpha$

Of course, one may also leverage the control samples and consider the null hypotheses $H _ { 0 } ^ { ( 0 ) } \colon Y _ { n + 1 } ( 0 ) = 1$ We can similarly construct the p-value (using a monotone function V as before)

$$
p _ { n + 1 } ^ { ( 0 ) } = \frac { 1 + \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ T _ { i } = 0 \} \cdot \mathbb { 1 } \{ V ( X _ { i } , 1 - Y _ { i } ) \leq V ( X _ { n + 1 } , 0 ) \} } { 1 + \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ T _ { i } = 0 \} } .
$$

Following exactly the same rationales as above, this p-value is valid in the same sense.

## 3.2 Proposed Method: Conformal Policy Learning

While both p-values in the last section are feasible and valid, they are conservative since $H _ { 0 } ^ { ( 1 ) } : Y _ { n + 1 } ( 1 ) = 0$ and $H _ { 0 } ^ { ( 0 ) } : Y _ { n + 1 } ( 0 ) = 1$ are both strict implications of $H _ { 0 } : Y _ { n + 1 } ( 1 ) = 0 , Y _ { n + 1 } ( 0 ) = 1$ . In this section, we propose the general CPL procedure that improves and subsumes the previous two options as special cases.

Following the conformal inference literature, we call the subset of the labeled data used to compute the p-values the “calibration data”. Then, both p-values in Section 3.1 only use either the treatment or the control group as the calibration data, which can be substantially generalized.

First, consider any inclusion function $\hat { g } \colon \mathcal { X } \times \{ 0 , 1 \} \to [ 0 , 1 ]$ whose training process is independent of the labeled data and the test point. Conditional on data, we draw independent inclusion indicators $G _ { i }$ ∼ Bernoulli $\left( \hat { g } ( X _ { i } , T _ { i } ) \right)$ for each $i \in [ n ]$ , which defines the calibration set

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { c a l i b } } = \{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i \in \mathcal { I } _ { \mathrm { c a l i b } } } , \quad \mathrm { w h e r e } \quad \mathcal { I } _ { \mathrm { c a l i b } } = \{ i \in [ n ] \colon G _ { i } = 1 \} . } \end{array}\tag{3.5}
$$

The inclusion into $\mathcal { D } _ { \mathrm { c a l i b } }$ can depend both on $T _ { i }$ and $X _ { i } ,$ and is allowed to be random, although later we shall see a binary $\hat { g }$ sufices for optimality. Since not all labeled data are used in calibration, we call this technique “selective calibration”.

Second, to allow involving data from both arms in calibration, we combine information from the two treatment arms. Define the computable proxy outcome $Y _ { i } ^ { \dagger } = T _ { i } Y _ { i } + ( 1 - T _ { i } ) ( 1 - Y _ { i } )$ , so $Y _ { i } ^ { \dagger } = 0$ whenever $Y _ { i } = 0$ for a treated sample or $Y _ { i } = 1$ for a control sample. Generalizing the previous ideas, $\dot { Y } _ { i } ^ { \dagger } = 0$ captures potentially unsafe units $( { \mathrm { i . e . } }$ , units for which $H _ { 0 }$ might happen).

With the two techniques in hand, we construct the p-value

$$
p _ { n + 1 } = \frac { 1 + \sum _ { i = 1 } ^ { n } G _ { i } \times \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ^ { \dagger } ) \leq V ( X _ { n + 1 } , 0 ) \} } { 1 + \sum _ { i = 1 } ^ { n } G _ { i } } .\tag{3.6}
$$

This covers the single-arm p-values in Section 3.1: it recovers $p _ { n + 1 } ^ { ( 1 ) } ~ ( \mathrm { r e s p . } ~ p _ { n + 1 } ^ { ( 0 ) } )$ when $G _ { i } = T _ { i }$ (resp. $G _ { i } =$   
$1 - T _ { i } )$ . Another simple case is to take $G _ { i } = 1$ which includes all the labeled data as the calibration set.

With this p-value, our conformal policy learning simply produces a thresholding rule ${ \hat { \pi } } ( X _ { n + 1 } ) : = \mathbb { 1 } \left\{ p _ { n + 1 } \leq \right.$ α}. Algorithm 1 summarizes the procedure.

Algorithm 1 Conformal Policy Learning (Randomized Experiments)   
Input: Labeled data $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } { } _ { : \quad }$ , test data $X _ { n + 1 }$ , target level $\alpha \in ( 0 , 1 )$ , score $V ( \cdot , \cdot )$ , inclusion function $\hat { g } ( \cdot , \cdot )$   
1: Draw independent inclusion indicators $G _ { i } \sim$ Bern $( \hat { g } ( X _ { i } , T _ { i } ) )$ for $i = 1 , \ldots , n .$   
2: Compute $Y _ { i } ^ { \dagger } = T _ { i } Y _ { i } + ( 1 - T _ { i } ) ( 1 - Y _ { i } )$ for $i = 1 , \ldots , n .$   
3: Compute p-value p<sub>n+1</sub> via (3.6).   
4: Compute $T _ { n + 1 } = 1 \{ p _ { n + 1 } \leq \alpha \}$   
Output: Safe treatment $T _ { n + 1 }$

Theorem 3.1 establishes the distribution-free safety guarantee for any score and inclusion functions whose inclusion probability in two groups sums up to a constant. The proof is in Appendix B.1.

Theorem 3.1. Assume Assumption 2.1 with $e ( x ) \equiv 0 . 5$ and Assumption 2.2. Then, for any score function $V \colon \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ that is non-decreasing in the second argument and $\hat { g } \colon \mathcal { X } \times \{ 0 , 1 \} \to [ 0 , 1 ]$ whose training process is independent of $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } \cup \{ X _ { n + 1 } \}$ , and obey $\hat { g } ( x , 1 ) + \hat { g } ( x , 0 ) = a \ f o r \ a$ constant $a > 0$ for any $x \in \mathcal { X } _ { : }$ , it holds that $\mathbb { P } ( Y _ { n + 1 } ( 1 ) = 0 , Y _ { n + 1 } ( 0 ) = 1 , p _ { n + 1 } \leq \alpha ) \leq \alpha$ for any $\alpha \in ( 0 , 1 )$ . That is, the finite-sample safety guarantee (2.1) holds for $\hat { \pi } ( X _ { n + 1 } ) = \mathbb { 1 } \{ p _ { n + 1 } \leq \alpha \}$

Theorem 3.1 states that, in principle, users can use any score function $V ( \cdot , \cdot )$ and any inclusion function $\hat { g } ( \cdot , \cdot )$ , and CPL (Algorithm 1) guarantees a controlled harm rate in finite samples. We emphasize that the two simplifying assumptions, namely $e ( x ) = 1 / 2$ and $\hat { g } ( x , 1 ) + \hat { g } ( x , 0 ) = a$ , are used only to streamline presentation, and will be eliminated in Section 4.

This theorem covers the special cases we discussed in Section 3.1. The one based on the treated (resp. control) samples correspond to $\mathbf { \nabla } \hat { g } ( x , t ) = t \left( \mathrm { r e s p . } \ \hat { g } ( x , t ) = 1 - t \right)$ , both obeying the constant-sum condition with $a = 1$ . As a third example, the conformal p-value using all the labeled data corresponds to $\hat { g } ( x , t ) \equiv 1$ and thus it also satisfies the constant-sum condition with $a = 2$ . Finally, while $\left( 3 . 6 \right)$ represents a restrictive class of policies, we shall see in the next subsection that specific choices of V and ˆg lead to asymptotically optimal safe policy learning algorithm among all safe policies (i.e., beyond the policy class specified by CPL).

The detailed proof of Theorem 3.1 is in Appendix B.1, and we provide a sketch here to clarify the intuitions. We again consider the “oracle” p-value

$$
p _ { n + 1 } ^ { * } = \frac { 1 + \sum _ { i = 1 } ^ { n } G _ { i } \times \mathbb { 1 } \left\{ V ( X _ { i } , Y _ { i } ^ { * } ) \leq V ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } \right\} } { 1 + \sum _ { i = 1 } ^ { n } G _ { i } } ,\tag{3.7}
$$

where $Y _ { i } ^ { * } = \operatorname* { m a x } \{ Y _ { i } ( 1 ) , 1 - Y _ { i } ( 0 ) \}$ so that $Y _ { i } ^ { * } = 0$ if and only if $Y _ { i } ( 1 ) = 0$ and $Y _ { i } ( 0 ) = 1$ , i.e., unit i is harmed. This is again not computable because $Y _ { i } ^ { * }$ is never observed even for the labeled data. The first key fact is that this oracle is valid as long as $\hat { g }$ satisfies the constant-sum condition. Indeed, we can show that in balanced randomized experiments, the data $\{ ( X _ { i } , Y _ { i } ^ { * } ) \} _ { G _ { i } = 1 }$ and $( X _ { n + 1 } , Y _ { n + 1 } ^ { * } )$ are exchangeable conditional on $\{ G _ { i } \} _ { i = 1 } ^ { n } ,$ , which implies $\mathbb { P } ( p _ { n + 1 } ^ { * } \le \alpha | \{ G _ { i } \} _ { i = 1 } ^ { n } ) \le \alpha$ . Second, the observed p-value (3.6) replaces $Y _ { n + 1 } ^ { * }$ in (3.7) with the “null” value 0 and replaces the calibration label $Y _ { i } ^ { * }$ with $Y _ { i } ^ { \dagger }$ , thereby upper bounding the oracle one on the “unsafe” null event. Specifically, the proxy outcome is conservative in the sense that $Y _ { i } ^ { * } \geq Y _ { i } ^ { \dagger } ;$ ; thus, whenever $Y _ { n + 1 } ^ { * } = 0$ , the monotonicity of V implies $p _ { n + 1 } \geq p _ { n + 1 } ^ { * }$ . This gives $\begin{array} { r } { \mathbb { P } \big ( Y _ { n + 1 } ^ { * } = 0 , p _ { n + 1 } \leq \alpha \big | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } \big ) \leq \mathbb { P } \big ( Y _ { n + 1 } ^ { * } = 0 , p _ { n + 1 } ^ { * } \leq \alpha \big | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } \big ) \leq \mathbb { P } \big ( p _ { n + 1 } ^ { * } \leq \alpha \big | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } \big ) \leq \alpha } \end{array}$

Remark 3.2 (Sharpness of conformal p-values). Our conformal p-value is subject to two sources of conservativeness compared with $p _ { n + 1 } ^ { * } \colon { \mathrm { \Omega } } ( i )$ the “oracle” labels $Y _ { i } ^ { * }$ are replaced by the proxy labels $Y _ { i } ^ { \dagger }$ , and (ii) the test label $Y _ { n + 1 } ^ { * }$ is replaced by the threshold 0. As in conformal selection (Jin and Cand\`es, 2023b), the issue (ii) will be addressed by a tailored “clipped” score function. On the other hand, (i) is rooted in the partial identification nature of the harm rate, and specific choices of gˆ makes our p-value sharp; we shall discuss this in more details once the optimality results in the next section are ready.

## 3.3 Optimality of Conformal Policy Learning

Having established the finite-sample safety guarantees for CPL, we proceed to study the optimality. We first analyze the optimal policy among all safe policies (i.e., including policies beyond our framework) under partial identification. We then show that CPL with specific choices of the score and inclusion functions asymptotically achieves global optimality.

To formalize the discussion, we define some additional notations. The unknown, true joint distribution over $( X , Y ( 1 ) , Y ( 0 ) )$ is denoted as $\mathbb { P } _ { X , Y ( 1 ) , Y ( 0 ) }$ , which induces the observable distribution $( X _ { i } , Y _ { i } , T _ { i } ) \sim$ $\mathbb { P } _ { X , Y , T }$ . Also, we denote the collection of distributions P over $( X , Y ( 1 ) , Y ( 0 ) )$ that are compatible with the observable distribution $\mathbb { P } _ { X , Y , T }$ as

$$
\mathcal { P } = \{ P _ { X , Y ( 1 ) , Y ( 0 ) } \colon P _ { X , Y ( T ) , T } = \mathbb { P } _ { X , Y ( T ) , T } \mathrm { ~ f o r ~ } P ( T = 1 | X , Y ( 1 ) , Y ( 0 ) ) = e ( X ) \} ,
$$

that is, the induced distribution of $( X , Y , T )$ under $P$ coincides with $\mathbb { P } _ { X , Y , T }$ on the observables.

For any distribution P and any treatment assignment rule $\pi \colon \mathcal { X }  \{ 0 , 1 \}$ , we write the harm rate as

$$
\begin{array} { r } { \mathrm { E r r } ( \pi ; P ) : = P ( Y _ { n + 1 } ( \pi ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) ) = P ( Y _ { n + 1 } ( 1 ) = 0 , Y _ { n + 1 } ( 0 ) = 1 , \pi ( X _ { n + 1 } ) = 1 ) . } \end{array}
$$

By the tower property, letting $\mathbb { E } _ { P }$ denote the expectation under P, we know

$$
\begin{array} { r } { \mathrm { E r r } ( \pi ; P ) = \mathbb { E } _ { P } \left[ \pi \left( X _ { n + 1 } \right) \cdot f _ { P } \left( X _ { n + 1 } \right) \right] \leq \alpha , \quad \mathrm { w h e r e } \quad f _ { P } ( x ) : = P ( Y _ { n + 1 } ( 1 ) < Y _ { n + 1 } ( 0 ) | X _ { n + 1 } = x ) . } \end{array}
$$

Here we call $f ( x )$ the harm rate function. Similar to the scalar marginal harm rate $\operatorname { E r r } ( \pi , P )$ , the function $f _ { P } ( \cdot )$ is not identifiable from data because $Y ( 1 )$ and $Y ( 0 )$ are never simultaneously observed. Kallus (2022) derived the sharp partial-identification upper bound for $f _ { P } ( x )$ , given by

$$
\gamma ( x ) : = \operatorname* { m i n } \{ 1 - \mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \} ,\tag{3.8}
$$

where $\mu _ { t } ( x ) = \mathbb { E } [ Y ( t ) | X = x ]$ for $t \in \{ 0 , 1 \}$ is the conditional mean function of each potential outcome. Namely, for any super-population $P _ { X , Y ( 1 ) , Y ( 0 ) }$ whose observed distribution $P _ { X , Y , T }$ (induced by the same treatment assignment mechanism) is equal to $\mathbb { P } _ { X , Y , T }$ , its harm rate function obeys $f _ { P } ( x ) \leq \gamma ( x )$ , and there exists one such distribution whose harm rate function coincides with $\gamma ( x )$

The most common objective of policy learning is to maximize the welfare. Following the policy learning literature $( \mathrm { e . g . }$ , Murphy, 2003; Manski, 2004; Hirano and Porter, 2009; Zhao et al., 2012; Kitagawa and Tetenov, 2018; Athey and Wager, 2021), we define the welfare of a policy π : $\mathcal { X }  \{ 0 , 1 \}$ as

$$
\mathrm { W e l f a r e } ( \pi ; \mathbb { P } ) : = \mathbb { E } \big [ Y _ { n + 1 } ( \pi ( X _ { n + 1 } ) ) \big ] ,
$$

where the larger value of outcome Y corresponds to the larger welfare. The welfare-optimal treatment rule subject to the safety guarantee can be defined as

$$
\begin{array} { r l } { \pi _ { \mathrm { { w e l f a r e } } } ^ { * } = \underset { \pi : \mathcal K \to \{ 0 , 1 \} } { \mathrm { a r g m a x } } } & { \mathrm { W e l f a r e } ( \pi ; \mathbb { P } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \underset { P \in \mathcal P } { \operatorname* { m a x } } \mathrm { E r r } ( \pi ; P ) \le \alpha . } \end{array}\tag{3.9}
$$

Here, we optimize the welfare subject to the worst-case safety guarantee. The constraint, max $P { \in } \mathcal { P } ~ \mathrm { E r r } ( \pi ; P ) \leq$ $\alpha ,$ ensures that, regardless of the underlying data-generating process, the harm rate is upper bounded by α.

We define the oracle welfare-optimal score function

$$
s _ { \mathrm { w e l f a r e } } ( x ) = - ( \mu _ { 1 } ( x ) - \mu _ { 0 } ( x ) ) / \gamma ( x ) ,\tag{3.10}
$$

where we use the convention $0 / 0 = 0 , a / 0 = + \infty$ if $a > 0$ and $a / 0 = - \infty$ if $a \ < \ 0$ throughout. The following theorem formally gives the optimal solution $\pi _ { \mathrm { w e l f a r e } } ^ { * }$ under the worst-case safety constraint. It is an implication of the more general Theorem A.5 in Appendix $\mathrm { A . 2 }$

Theorem 3.3. Assume $s _ { \mathrm { w e l f a r e } } ( X )$ has no point mass. Then an optimal solution to (3.9) is

$$
\pi _ { \mathrm { v e l f a r e } } ^ { * } ( x ) = \mathbf { 1 } \{ s _ { \mathrm { w e l f a r e } } ( x ) \leq r ^ { * } \} , \quad w h e r e \quad r ^ { * } : = \operatorname* { s u p } \left\{ \tilde { r } \leq 0 : \mathbb { E } [ \gamma ( X ) \mathbf { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) \leq \tilde { r } \} ] \leq \alpha \right\} .
$$

Moreover, writing $\tau ( x ) = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x ) , \ i f \mathbb { E } [ \gamma ( X ) \mathbf { 1 } \{ \tau ( X ) > 0 \} ] \le \alpha ,$ , then $r ^ { * } = 0$ and $\pi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = \mathbf { 1 } \{ \tau ( x ) >$ 0} almost surely. Otherwise, $r ^ { * } < 0$ and $\mathbb { E } [ \gamma ( X ) \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] = \alpha$

Theorem 3.3 shows that the optimal treatment rule for the welfare maximization is based on a cutof on $s _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } )$ while staying in the region $\tau ( X _ { n + 1 } ) \geq 0$ . The intuition is as follows. We can show that the problem (3.9) that defines $\pi _ { \mathrm { w e l f a r e } } ^ { * }$ can be rewritten as

$$
\begin{array} { r l } { \underset { \pi : \ : \mathcal { X } \to \{ 0 , 1 \} } { \mathrm { m a x i m i z e } } } & { \mathbb { E } [ \pi ( X _ { n + 1 } ) \tau ( X _ { n + 1 } ) ] + \mathrm { c o n s t a n t } } \\ { \newline } & { \mathrm { s u b j e c t ~ t o } \quad \mathbb { E } [ \pi ( X _ { n + 1 } ) \gamma ( X _ { n + 1 } ) ] \leq \alpha , } \end{array}
$$

where $\tau ( x ) = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ captures the conditional average treatment efect (CATE).

The optimal solution thus seeks the feature regions that yield the largest welfare gain $\tau ( X )$ relative to each unit of ${ } ^ { \mathfrak { a } } \mathrm { c o s t } ^ { { \mathfrak { n } } } \ \gamma ( X )$

The next question is whether, and under what conditions, the conformal policy learning method can achieve this optimal welfare. Following Theorem 3.3, we define the conformal p-value

$$
p _ { n + 1 } ^ { \mathrm { w e l f a r e } } = \frac { 1 + \sum _ { i = 1 } ^ { n } G _ { i } \times \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \times \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { s } ( X _ { n + 1 } ) \} } { 1 + \sum _ { i = 1 } ^ { n } G _ { i } } ,\tag{3.11}
$$

for the specific choices

$$
G _ { i } = T _ { i } \mathbf { 1 } \big \{ 1 - \hat { \mu } _ { 1 } ( X _ { i } ) \leq \hat { \mu } _ { 0 } ( X _ { i } ) \big \} + ( 1 - T _ { i } ) \mathbf { 1 } \big \{ 1 - \hat { \mu } _ { 1 } ( X _ { i } ) > \hat { \mu } _ { 0 } ( X _ { i } ) \big \} ,\tag{3.12}
$$

$$
\mathrm { a n d } \qquad \hat { s } ( x ) = - ( \hat { \mu } _ { 1 } ( x ) - \hat { \mu } _ { 0 } ( x ) ) / \hat { \gamma } ( x ) .\tag{3.13}
$$

This corresponds to taking a clipped score $V ( x , y ) = M \mathbb { 1 } \{ y > 0 \} + \hat { s } ( x )$ with a suficiently large constant $M > 0$ in Algorithm 1. Note that $\hat { g } ( x , 1 ) + \hat { g } ( x , 0 ) = 1$ and therefore it also satisfies the condition for the inclusion function specified in Theorem 3.1. The intuition about the expression of the inclusion function $G _ { i }$ is given in the next subsection, where we discuss the sharpness of our results.

Consider the conformal policy learning algorithm with estimated welfare cutof (if necessary)

$$
\begin{array} { r } { \widehat \pi _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) : = \mathbb { 1 } \{ p _ { n + 1 } ^ { \mathrm { w e l f a r e } } \leq \alpha \} \mathbb { 1 } \{ \widehat \mu _ { 1 } ( X _ { n + 1 } ) > \widehat \mu _ { 0 } ( X _ { n + 1 } ) \} . } \end{array}
$$

Theorem 3.4 shows that when conditional expectation functions of potential outcomes are correctly specified, CPL achieves the optimal welfare asymptotically. Its proof is included in Appendix B.2.

Theorem 3.4. Suppose Assumption 2.1 holds with $e ( x ) = 0 . 5$ and Assumption 2.2 holds. Suppose $\| \hat { \mu } _ { t } ( X ) -$ $\mu _ { t } ( X ) \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 )$ as $n \to \infty$ for $t \in \{ 0 , 1 \}$ . Additionally, assume $s _ { \mathrm { w e l f a r e } } ( X )$ has no point mass. Then, $\mathbb { E } [ Y ( \tilde { \pi } _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) ) ]  \mathrm { W e l f a r e } ( \pi _ { \mathrm { w e l f a r e } } ^ { * } ; \mathbb { P } )$ as $n \to \infty$ where $\pi _ { \mathrm { w e l f a r e } } ^ { * }$ is the optimal solution in Theorem 3.3.

The consistency condition on the estimated conditional expectation functions could be achieved by flexible machine-learning-based methods. Moreover, even if the consistency condition fails, the conformal policy learning algorithm based on $p _ { n + 1 } ^ { \mathrm { w e l f a r e } }$ still satisfies the finite-sample safety guarantee as per Algorithm 1.

In Appendix A.1, we present analogous optimality results when researchers are interested in maximizing the $\mathrm { ^ { 6 } p o w e r } ^ { \mathrm { , 5 } }$ , the probability of treating the test unit. The only diference is that we should now use a power-oriented score $\hat { s } ( x ) = \hat { \gamma } ( x )$

## 3.4 Optimality, Sharpness, and Selective Calibration

The optimality result suggests that, in the non-trivial setting where the constraint is binding, conformal policy learning can exhaust the harm rate budget under partial identification, matching the globally optimal rule whose worst-case harm rate is exactly α. We now continue on Remark 3.2 to explain why this sharpness can be achieved. The use of clipped score essentially eliminated the conservativeness of using 0 instead of $Y _ { i } ^ { * }$ We now focus on the conservativeness arising from replacing the “oracle” outcome $Y _ { i } ^ { * } = \operatorname* { m a x } \{ Y _ { i } ( 1 ) , 1 - Y _ { i } ( 0 ) \}$ by the proxy label $Y _ { i } ^ { \dagger } = Y _ { i } T _ { i } + ( 1 - Y _ { i } ) ( 1 - T _ { i } ) \leq Y _ { i } ^ { * }$ , and discuss how this is eliminated by selective calibration.

The use of $Y _ { i } ^ { \dagger }$ is deeply rooted in the partial identification nature of the problem: even for the labeled data, the harm $Y _ { i } ^ { * }$ is not observed. As such, the sharpness hinges on whether, on the population level, calibration with $\dot { Y _ { i } ^ { \dag } }$ matches the worst-case harm-rate bound that relies on $\gamma ( X _ { i } )$ in (3.8).

CPL achieves so through the selective calibration mechanism. Assuming, for intuitions, that the conditional mean functions are perfect, the optimal conformal p-value uses the inclusion indicator $G _ { i } \ =$ $T _ { i } \mathbb { 1 } \{ 1 - \mu _ { 1 } ( X _ { i } ) \leq \mu _ { 0 } ( X _ { i } ) \} + ( 1 - T _ { i } ) \mathbb { 1 } \{ 1 - \mu _ { 1 } ( X _ { i } ) > \mu _ { 0 } ( X _ { i } ) \}$ . Namely, a treated unit $X _ { i }$ enters the calibration set if and only if $1 - \mu _ { 1 } ( X _ { i } ) \leq \mu _ { 0 } ( X _ { i } )$ , in which case

$$
\gamma ( X _ { i } ) = 1 - \mu _ { 1 } ( X _ { i } ) = \mathbb { E } [ \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \vert X _ { i } , T _ { i } = 1 ]
$$

since $Y _ { i } ^ { \dagger } = Y _ { i } ( 1 )$ for $T _ { i } = 1$ . That is, the proxy label $Y _ { i } ^ { \dagger }$ provides “sharp” calibration for the worst-case harm rate $\gamma ( X _ { i } )$ . A similar observation applies to control units.

The above discussion shows that for sharp harm rate control, one should use a labeled unit for calibration if and only if the treatment status matches whether the outcome model is “active” in the worst-case harm rate function $\gamma ( x )$ , i.e., whether a treated unit obeys $\gamma ( X _ { i } ) = 1 - \mu _ { 1 } ( X _ { i } )$ or a control unit obeys $\gamma ( X _ { i } ) = \mu _ { 0 } ( X _ { i } )$ In the simulations, we shall evaluate the “informativeness” of labeled data and the power of CPL.

Finally, we re-emphasize the model-free nature of conformal policy learning. Even though the arm informativeness—through the inclusion function $\hat { g } \mathrm { - } \mathrm { i } \mathrm { s }$ estimated and therefore certainly imperfect, our method provides conservative harm rate control without any modeling assumptions.

## 4 Conformal Policy Learning with Stratified Experiments

In this section, we use weighted conformal inference to generalize our previous results to allow for both arbitrary, known propensity score $e ( x )$ and arbitrary inclusion function ${ \hat { g } } .$ It also serves as the foundation for CPL in observational studies, which we cover in the next section.

The selective calibration process remains the same as Section 3.2. Consider any pre-trained monotone score function $V \colon \mathcal { X } \times \{ 0 , 1 \} \to \mathbb { R }$ and inclusion function $\boldsymbol { \hat { g } } \colon \mathcal { X } \times \{ 0 , 1 \}  [ 0 , 1 ]$ . The calibration set is the same as (3.5) where $G _ { i } \sim \mathrm { B e r n } ( \hat { g } ( X _ { i } , T _ { i } ) )$ are independently drawn inclusion indicators.

When computing conformal p-values for a new test point $X _ { n + 1 } \sim \mathbb { P } _ { X }$ , conditional on $G _ { i } = 1$ , there is a covariate shift between $\mathcal { D } _ { \mathrm { c a l i b } }$ and $X _ { n + 1 }$ that arises from both the treatment assignment and the exogenous random inclusion into the calibration set. To address this shift, we define the function w : $\mathcal { X } \to \mathbb { R } ^ { + }$ by

$$
\boldsymbol { w } ( \boldsymbol { x } ) = \big ( e ( x ) \hat { g } ( x , 1 ) + ( 1 - e ( x ) ) \hat { g } ( x , 0 ) \big ) ^ { - 1 } .\tag{4.1}
$$

Because $e ( x ) : = \mathrm { P r } ( T _ { i } = 1 \mid X _ { i } = x )$ and $\hat { g } ( x , t ) : = \operatorname* { P r } ( G _ { i } = 1 \ | \ X _ { i } = x , T _ { i } = t )$ , one can show that $w ( x ) = \operatorname* { P r } ( G _ { i } = 1 \mid X _ { i } = x ) ^ { - 1 }$ , the inverse probability of being included for calibration given $X _ { i } ~ = ~ x$ regardless of the choice of $e ( x )$ and $\hat { g } .$ . Given the weights, we compute

$$
p _ { n + 1 } ^ { \mathrm { s t r } } = \frac { w ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } G _ { i } w ( X _ { i } ) \Im \{ V ( X _ { i } , Y _ { i } ^ { \dagger } ) \leq V ( X _ { n + 1 } , 0 ) \} } { w ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } G _ { i } w ( X _ { i } ) } .\tag{4.2}
$$

The p-value (3.6) in Section 3 is a special case of (4.2), where the simplifying assumptions $e ( x ) = 0 . 5$ and $\hat { g } ( x , 1 ) + \hat { g } ( x , 0 ) = a$ for a constant $a > 0$ implied constant weights and no reweighting is needed.

Theorem 4.1 states that CPL with the above weighted conformal p-value achieves the distribution-free safety guarantee, whose proof is in Appendix B.3.

Theorem 4.1. Suppose Assumption 2.1 and Assumption 2.2 hold, and $e ( x )$ is known. Then, for any score function $V \colon \mathcal { X } \times \mathcal { Y } \ $ R that is non-decreasing in the second argument and for any inclusion function $\hat { g } \colon \mathcal { X } \times \{ 0 , 1 \} \to [ 0 , 1 ]$ obeying $e ( X ) \hat { g } ( X , 1 ) + ( 1 - e ( X ) ) \hat { g } ( X , 0 ) > 0 , P _ { X } - a . s .$ , it holds that $\mathbb { P } ( Y _ { n + 1 } ( 1 ) =$ $0 , Y _ { n + 1 } ( 0 ) = 1 , p _ { n + 1 } ^ { s t r } \leq \alpha ) \leq \alpha ~ f o r$ any $\alpha \in ( 0 , 1 )$ . That is, (2.1) holds $f o r \hat { \pi } ^ { s t r } ( X _ { n + 1 } ) = \mathbb { 1 } \{ p _ { n + 1 } ^ { s t r } \leq \alpha \}$

As in Section 3, given consistent outcome modeling, CPL achieves optimality with

$$
p _ { n + 1 } ^ { \mathrm { s t r - o p t } } = \frac { w ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } w ( X _ { i } ) G _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { s } ( X _ { n + 1 } ) \} } { w ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } w ( X _ { i } ) G _ { i } } ,\tag{4.3}
$$

where $G _ { i } = T _ { i } \mathbf { 1 } \big \{ 1 - \hat { \mu } _ { 1 } ( X _ { i } ) \leq \hat { \mu } _ { 0 } ( X _ { i } ) \big \} + ( 1 - T _ { i } ) \mathbf { 1 } \big \{ 1 - \hat { \mu } _ { 1 } ( X _ { i } ) > \hat { \mu } _ { 0 } ( X _ { i } ) \big \}$ . The score functions remains the same as the preceding case: $\hat { s } ( x ) = - ( \hat { \mu } _ { 1 } ( x ) - \hat { \mu } _ { 0 } ( x ) ) / \hat { \gamma } ( x )$ for welfare maximization. This is the weighted version of equation (3.11). The proof of Theorem 4.2 is in Appendix B.4.

Theorem 4.2. Suppose Assumption 2.1 and Assumption 2.2 hold, and $e ( x )$ is known. Suppose $\| \hat { \mu } _ { t } ( X ) -$ $\mu _ { t } ( X ) \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } \overset { P } { \to } 0$ as n → ∞ for t ∈ {0, 1} and $s _ { \mathrm { w e l f a r e } } ( X )$ has no point mass. Let

$$
\begin{array} { r } { \widehat { \pi } _ { s t r - o p t } ( X _ { n + 1 } ) : = \mathbb { 1 } \{ p _ { n + 1 } ^ { s t r - o p t } \leq \alpha \} \mathbb { 1 } \{ \widehat { \mu } _ { 1 } ( X _ { n + 1 } ) > \widehat { \mu } _ { 0 } ( X _ { n + 1 } ) \} } \end{array}
$$

where we take the same $G _ { i }$ as (3.12) and $\hat { s } ( \cdot )$ as (3.13). Then $\mathbb { E } [ Y ( \tilde { \pi } _ { s t r - o p t } ( X _ { n + 1 } ) ) ]  \mathrm { W e l f a r e } ( \pi _ { \mathrm { w e l f a r e } } ^ { * } ; \mathbb { P } )$ as $n \to \infty$ , where $\pi _ { \mathrm { w e l f a r e } } ^ { * }$ is the optimal solution in Theorem 3.3.

Analogously, CPL with the same selection indicators $\left\{ G _ { i } \right\}$ as (3.12) and the estimated power-optimal score yields asymptotically optimal power; this result is deferred to Theorem $\mathrm { A . 2 }$ in Appendix A.1.

## 5 Conformal Policy Learning with Observational Data

In this section, we further generalize CPL to observational studies. With observational data, the main challenge is that the propensity score, hence the weight function (4.1), is unknown and needs to be estimated.

While a natural idea is to estimate the weights and plug them into the conformal p-value, it remains unclear how the safety guarantee changes with the estimation quality. We present a learn-then-balance procedure to obtain the estimated weights so the resulting policy learning algorithm enjoys a double robustness property.

## 5.1 Conformal Policy Learning with Learn-then-Balance Weights

We construct p-values of a similar form as (4.2) and (4.3):

$$
p _ { n + 1 } ^ { \mathrm { o b s } } = \frac { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } \hat { w } _ { i } G _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { s } ( X _ { n + 1 } ) \} } { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } \hat { w } _ { i } G _ { i } } ,\tag{5.1}
$$

where the inclusion indicators are $G _ { i } \sim \mathrm { B e r n } ( \hat { g } ( X _ { i } , T _ { i } ) )$ for a function $\boldsymbol { \hat { g } } \colon \mathcal { X } \times \{ 0 , 1 \}  [ 0 , 1 ]$ whose training process is independent of the labeled and unlabeled data. The only diference of (5.1) from (4.3) is that the weights are now estimated. Notably, here we work with a form analogous to (4.3) since our balancing weights are more easily stated in terms of the score function $\hat { s } ( \cdot )$ . It corresponds to (4.2) with the conformity score $V ( x , y ) = M \mathbb { 1 } \{ y > 0 \} + \hat { s } ( x )$ for a suficiently large constant $\begin{array} { r } { M > 2 \operatorname* { s u p } _ { x } { | \hat { s } ( x ) | } } \end{array}$

Denoting ${ \mathcal { T } } _ { \mathrm { c a l i b } } = \{ i \in [ n ] \colon G _ { i } = 1 \}$ , the estimated weights $\{ \hat { w } _ { i } \} _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } \cup \{ n + 1 \} }$ in (5.1) are obtained by a learn-then-balance approach. We begin with two learned functions, one preliminary weight function w˜ : $\mathcal { X } \to \mathbb { R } ^ { + }$ , and one estimated harm-rate function $\hat { \gamma } \colon \mathcal { X }  [ 0 , 1 ]$ . A natural choice is to define $\tilde { w } ( x ) =$ $1 / [ \hat { e } ( x ) \hat { g } ( x , 1 ) + ( 1 - \hat { e } ( x ) \hat { g } ( x , 0 ) ]$ where $\hat { e } ( x )$ is an estimated propensity score function, and $\hat { \gamma } ( x ) = \operatorname* { m i n } \lbrace 1 -$ $\hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ for estimated outcome models $\{ \hat { \mu } _ { t } ( \cdot ) \} _ { t \in \{ 0 , 1 \} }$ . For simplicity, we require the propensity score and outcome models to be trained independently of the calibration and test data (in practice, one can use sample splitting to fit the models (Chernozhukov et al., 2018) and the properties are analogously studied in Jin and Zubizarreta (2025)). Then, define the balancing feature vector

$$
\hat { \phi } ( x ) = \left( \hat { \gamma } ( x ) \mathbb { 1 } \{ \hat { s } ( x ) \leq \hat { t } \} , \tilde { w } ( x ) \right) \in \mathbb { R } ^ { 2 } , \quad \mathrm { w h e r e } \quad \hat { t } = \operatorname* { s u p } \bigg \{ t \in \mathbb { R } : \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq t \} \leq \alpha \bigg \} .
$$

Finally, we let $\{ \hat { w } _ { i } \} _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } }$ be the optimal solution to the following optimization program with $\delta _ { n } = O ( n ^ { - 1 / 2 } )$ :

$$
\operatorname * { a r g m i n } _ { w \ge 0 } \left\{ \| w \| ^ { 2 } : \left| \frac { 1 } { | T _ { \mathrm { c a l i b } } | } \sum _ { i \in \mathcal { I } _ { \mathrm { c a l i b } } } w _ { i } \hat { \phi } _ { \boldsymbol k } ( \boldsymbol X _ { i } ) - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { \phi } _ { \boldsymbol k } ( \boldsymbol X _ { i } ) \right| \le \delta _ { n } , k \in \{ 1 , 2 \} ; \frac { 1 } { | T _ { \mathrm { c a l i b } } | } \sum _ { i \in \mathcal { I } _ { \mathrm { c a l i b } } } w _ { i } = 1 \right\} .\tag{5.2}
$$

We shall see that the post-hoc processing to ensure the finite-sample approximate balancing (5.2) is essential for us to obtain the doubly robust safety guarantees even when the propensity model is misspecified. The first condition in (5.2) follows the balancing weights (Hainmueller, 2012; Zubizarreta, 2015), by enforcing a finite-sample balancing condition inspired by the desired population-level property: for the correct weights $w ( \cdot )$ , it should hold that $\begin{array} { r } { \frac { 1 } { | { \mathcal T } _ { \mathrm { c a l i b } } | } \sum _ { i \in { \mathcal T } _ { \mathrm { c a l i b } } } w ( X _ { i } ) f ( X _ { i } ) \approx \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f ( X _ { i } ) } \end{array}$ for any fixed function $f \colon \mathcal { X }  \mathbb { R }$ . Here, we balance specifically-designed functions to obtain favorable statistical properties under general conditions on the learned functions.

## 5.2 Doubly Robust Safety Guarantees

We now formalize the safety guarantees of CPL (5.1) with weights $\{ \hat { w } _ { i } \} _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } }$ from (5.2). These results will require mild regularity conditions for the learn-then-balance optimization, which we defer to Assumption A.10. These conditions require overlap and boundedness, local stability of the population balancing solution, and regularity of the population cutof.

Under these conditions, and given that either the preliminary weight function or the outcome models are consistent, we obtain asymptotic safety guarantee for CPL. The proof of Theorem 5.1 is included in Appendix B.5, which follows from our general theory with estimated weights in Appendix C.1.

Theorem 5.1 (Model double robustness). Suppose Assumption A.10 holds, and $\delta _ { n } = O ( n ^ { - 1 / 2 } )$ . Then,

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname { l p } ( Y _ { n + 1 } ( \hat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) ) \leq \alpha
$$

under either of the following conditions:

(i). The preliminary weight function is consistent: $\| \tilde { w } - w \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } \bigl ( 1 \bigr )$ for the true weight $w ( \cdot )$ in (4.1). (ii). The outcome models are consistent: $\| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } \bigl ( 1 \bigr )$

Remark 5.2. We remark that the techniques and conditions here difer from existing model double-robustness results in conformal prediction (e.g. Lei and Cand\`es (2021)) where correct outcome model alone is enough to ensure validity; in our setting, because of the thresholding nature of the CPL policy, explicit finite-sample balance turns out to be an important element of our theoretical analysis for double robustness.

Taking a step further, we show that the learn-then-balance weights lead to a product error rate structure, which further yields the $O _ { P } ( n ^ { - 1 / 2 } )$ convergence of the harm rate of CPL given slow, nonparametric convergence rates of the estimated models. The proof of Theorem 5.3 is in Appendix B.6.

Theorem 5.3 (Rate double robustness). Suppose Assumption A.10 holds and $\delta _ { n } = { \cal O } ( n ^ { - 1 / 2 } ) . I f \parallel \tilde { w } -$ $w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } \big ( n ^ { - 1 / 4 } \big )$ and $\| \hat { \gamma } - \gamma \| _ { L _ { 1 } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } )$ , then $\{ \mathbb { P } ( Y _ { n + 1 } ( \hat { \pi } _ { o b s } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) | \mathcal { A } _ { n } ) - \alpha \} _ { + } =$ $O _ { P } ( n ^ { - 1 / 2 } )$ , where ${ \mathcal { A } } _ { n }$ is the σ-field for the randomness in calibration data, training, and selective inclusion.

## 5.3 Optimal Conformal Policy Learning with Observational Studies

Since the population optimization problem only depends on the super-population of the potential outcomes and observed features, the population-level optimal solution remains the same as Theorem 3.3. The following theorem shows that under mild conditions, our method with the optimally chosen score and inclusion function achieves the optimal welfare with observational data. Its proof is in Appendix B.7. The results for power optimality are deferred to Appendix $\mathrm { A . 1 }$

Theorem 5.4 (Welfare optimality with observational data). Let $\hat { \pi } _ { o b s } ( X _ { n + 1 } ) = \mathbb { 1 } \{ p _ { n + 1 } ^ { o b s } \leq \alpha \} \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) <$ 0} where we take the same $G _ { i }$ as (3.12) and sˆ(·) as (3.13). Suppose Assumption A.10 holds, and $\delta _ { n } =$ $O ( n ^ { - 1 / 2 } )$ . Furthermore, assume $\| \hat { g } - g ^ { * } \| _ { L _ { 2 } ( \mathbb { P } _ { X , T } ) } + \| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } + \| \hat { s } - s _ { \mathrm { w e l f a r e } } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 )$ for $g ^ { * } ( x , t ) =$ $t \mathbb { 1 } \{ 1 - \mu _ { 1 } ( x ) \leq \mu _ { 0 } ( x ) \} + ( 1 - t ) \mathbb { 1 } \{ 1 - \mu _ { 1 } ( x ) > \mu _ { 0 } ( x ) \}$ , and $s _ { \mathrm { w e l f a r e } } ( X )$ defined in (3.10) has no point mass. Then $\mathbb { E } [ Y \{ \hat { \pi } _ { o b s } ( X _ { n + 1 } ) \} ]  \mathrm { W e l f a r e } ( \pi _ { \mathrm { w e l f a r e } } ^ { * } ; \mathbb { P } )$ as $n  \infty$ , where $\pi _ { \mathrm { w e l f a r e } } ^ { * }$ is the optimal solution in Theorem 3.3.

As in randomized experiments, optimality requires two convergence conditions: $\hat { g }  g ^ { * }$ , which makes the proxy-label calibration sharp, and $\hat { s }  s _ { \mathrm { w e l f a r e } }$ , which provides the optimal treatment ranking. When $\hat { \boldsymbol g } , \hat { \boldsymbol s } .$ , and $\hat { \gamma }$ are constructed from plug-in outcome model estimates, these conditions follow from consistent outcome models under the usual regularity conditions. In this outcome-correct case, the balancing condition aligns the weighted calibration with the oracle harm-welfare curve, so the weight estimator need not be consistent. However, inheriting the model consistency of Theorem 5.1, one can show that the conclusion also continues to hold under consistent weights, ¯w ∝ w, without requiring $\hat { \gamma } \to \gamma$ , provided that $\hat { g } \to g ^ { * }$ and $\hat { s }  s _ { \mathrm { o p t } }$ still hold. Yet, we do not pursue it here since the latter two convergence conditions usually require consistency of the estimated outcome models which imply the consistency of the harm rate estimation.

## 6 Simulation Studies

## 6.1 Balanced Randomized Experiments

We begin by evaluating the methods in balanced randomized experiments (Section 3). The data in the randomized experiment settings are generated using the general framework below.

Data generating processes. Throughout, we generate the features $X \sim P _ { X }$ for some distribution $P _ { X }$ and i.i.d. treatment indicators $T \sim \mathrm { B e r n } ( 1 / 2 )$ , independent of everything else. The potential outcomes follow $\mathbb { P } ( Y ( 1 ) = 1 | X = x ) = \mu _ { 1 } ( x )$ and $\mathbb { P } ( Y ( 0 ) = 1 | X = x ) = \mu _ { 0 } ( x )$ for some functions $\mu _ { 1 } ( \cdot )$ and $\mu _ { 0 } ( \cdot )$ . The potential outcomes are coupled negatively, meaning that $\mathbb { P } ( Y ( 1 ) = 0 , Y ( 0 ) = 1 | X = x ) =$ min $\{ 1 - \mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \}$ . Note that any coupling would lead to the same observed data distribution $P _ { X , Y , T }$ (and thus the same results for any method), but the negative coupling leads to the worst-case harm rate. The specific choice of $P _ { X }$ and $\mu _ { 1 } ( \cdot ) , \mu _ { 0 } ( \cdot )$ is introduced in each set of simulations below. We first design four diverse settings to compare our methods and baselines, followed by two additional settings to investigate the performance of our methods, including (i) the power of selective calibration by varying the informativeness of arms, and (ii) decisions by heterogeneous subgroups.

## 6.1.1 Harm rate control in diverse settings

The first set of simulations evaluates the harm rate control across diverse settings and modeling choices. We set $X \in \mathbb { R } ^ { 2 0 }$ with $P _ { X } = \mathcal { N } ( 0 , I _ { d } )$ . The conditional mean functions are $\mu _ { t } ( x ) = \exp ( \eta _ { t } ( x ) ) / ( 1 + \exp ( \eta _ { t } ( x ) )$ , $t \in \{ 0 , 1 \}$ for some functions $\eta _ { t } ( x )$ . We design four settings (see Appendix E.1 for detailed DGPs):

• Setting 1: approximately linear, where $\eta _ { t } ( x ) = \beta _ { t } ^ { \top }$ x for some $\beta _ { t } \in \mathbb { R } ^ { 2 0 }$ and fully-observed $X \in \mathbb { R } ^ { 2 0 }$

• Setting 2: approximately linear DGP but the observed covariates are $X _ { 1 : 7 } \in \mathbb { R } ^ { 7 } ;$

• Setting 3: nonlinear $\eta _ { t } ( x )$ involves the first 5 entries of x, with fully observed $X \in \mathbb { R } ^ { 2 0 }$

• Setting 4: nonlinear $\eta _ { t } ( x )$ involves the first 8 entries of $x ,$ but the observed covariates are $X _ { 1 : 5 } \in \mathbb { R } ^ { 5 }$ In settings 2 and 4, missing some covariates typically reduces the accuracy of learned outcome models (finite-sample harm-rate control still holds as per Theorem 3.1). We vary the total labeled sample size $n _ { \mathrm { t o t a l } } \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 \}$ , fixing the number of test data $m = 1 0 0 0$ , and $\alpha = 0 . 1$ . We use $\mathcal { D } _ { \mathrm { l a b e l } }$ to denote the labeled data and $\mathcal { D } _ { \mathrm { t e s t } }$ to denote the test data.

Methods and evaluation. We compare the following three baselines and two CPL methods:

(1) Li et al., Li et al. (2023) with doubly-robust estimator and constrained optimization among depth-2 decision trees via econml python library. The policy is learned on $\mathcal { D } _ { \mathrm { l a b e l } }$ and evaluated on $\mathcal { D } _ { \mathrm { t e s t } }$ . This method requires the distributional assumption about the potential outcomes. In particular, it assumes that the potential outcomes are non-negatively correlated, which is violated in this setup.

(2) Policy-Tree, which maximizes welfare $\mathbb { E } [ Y ( \pi ( X _ { n + 1 } ) ) ]$ among policies represented by depth-2 decision trees with econml. The policy is learned on $\mathcal { D } _ { \mathrm { l a b e l } }$ and evaluated on $\mathcal { D } _ { \mathrm { t e s t } }$ . While this method is not designed for safety control, we include it as the baseline due to its popularity as a standard practice.

(3) Threshold, which uses $\mathcal { D } _ { \mathrm { l a b e l } }$ to obtain an estimator $\hat { \gamma } ( x ) = \mathrm { m i n } \{ 1 - \hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ and treat $\{ j \in$ $[ m ] \colon \widehat { \gamma } ( X _ { n + j } ) \ \leq \ \widetilde { \gamma } \}$ where $\begin{array} { r } { \widetilde { \gamma } = \operatorname* { m a x } \{ \gamma \colon \sum _ { i = 1 } ^ { m } \widehat \gamma ( X _ { n + j } ) \mathbb { 1 } \{ \widehat \gamma ( X _ { n + j } ) \leq \gamma \} \leq 0 . 1 \cdot m \} } \end{array}$ . This heuristic calibration is valid only when the estimator ˆγ is suficiently accurate.

(4) CPL-sel-welfare, the CPL algorithm that maximizes the welfare, i.e., Algorithm 1 with $g ( x , t ) =$ t 1 $\{ 1 - \hat { \mu } _ { 1 } ( x ) \leq \hat { \mu } _ { 0 } ( x ) \} + ( 1 - t ) \mathbb { 1 } \{ 1 - \hat { \mu } _ { 1 } ( x ) > \hat { \mu } _ { 0 } ( x ) \}$ and a clipped score function $V ( x , y ) = M \mathbb { 1 } \{ y >$ $0 \} - \}$ with a suficiently large constant $M > 0$ . This method always satisfies the safety guarantee and asymptotically achieves the optimal welfare if $\mu _ { t } ( x )$ is consistently estimated.

(5) CPL-sel-power, the CPL algorithm that maximizes the power, i.e., Algorithm 1 with $g ( x , t ) = t 1 \{ 1 -$ $\hat { \mu } _ { 1 } ( x ) \leq \hat { \mu } _ { 0 } ( x ) \} + ( 1 - t ) \mathbb { 1 } \{ 1 - \hat { \mu } _ { 1 } ( x ) > \hat { \mu } _ { 0 } ( x ) \}$ and a clipped score function $V ( x , y ) = M \mathbb { 1 } \{ y > 0 \} + \hat { \gamma } ( x )$ with a suficiently large constant $M > 0$ . This method always satisfies the safety guarantee and asymptotically achieves the optimal power if $\mu _ { t } ( x )$ is consistently estimated (Appendix A.1).

For the two CPL variants, the labeled data $\mathcal { D } _ { \mathrm { l a b e l } }$ is randomly split into the training (75%) and calibration (25%) folds $\mathcal { D } _ { \mathrm { t r a i n } }$ and $\mathcal { D } _ { \mathrm { c a l i b } }$ . The training fold is used to fit two models ${ \hat { \mu } } _ { 1 } ( x )$ and ${ \hat { \mu } } _ { 0 } ( x )$ for $\mu _ { 1 } ( x )$ and $\mu _ { 0 } ( x )$ , respectively. The function $\hat { \gamma } ( x ) = \mathrm { m i n } \{ 1 - \hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ then estimates the upper bound on the harm rate function. We adopt two model classes for training the outcome models: logistic regression, and random forest, which are used in both Threshold and CPL methods. Li et al. and Policy-Tree use random forests for nuisance component estimation to be consistent with the tree-based policy class. See Appendix E.1 for additional details on method implementation.

Given the learned treatment $T _ { n + j } : = \hat { \pi } ( X _ { n + j } ) \in \{ 0 , 1 \}$ for $j \in [ m ]$ produced by the methods, we evaluate three metrics: the harm rate by $\begin{array} { r } { \frac { 1 } { m } \sum _ { j = 1 } ^ { m } T _ { n + j } \Im \{ Y _ { n + j } ( 1 ) < Y _ { n + j } ( 0 ) \} } \end{array}$ , the welfare by $\begin{array} { r } { \frac { 1 } { m } \sum _ { j = 1 } ^ { m } Y _ { n + j } ( T _ { n + j } ) } \end{array}$ and the fraction of treatment by $\begin{array} { r } { \frac { 1 } { m } \sum _ { j = 1 } ^ { m } T _ { n + j } } \end{array}$ . All metrics are averaged over 200 independent runs.

![](images/90d9108bd8c3278a6232fe4cd548cff12f45084fafaecae7a4fbc1937c429c41.jpg)  
Figure 2: (a) Empirical harm rate, (b) power (probability of treatment in the test units), (c) average welfare of various methods at level $\alpha = 0 . 1$ in the randomized experiment studies, averaged over 200 independent simulation runs. Each row represents a diferent prediction model (logistic regression, random forest) for $( \hat { \mu } _ { 1 } , \hat { \mu } _ { 0 } )$ , and each column represents a diferent data-generating process. The x-axis is the total labeled sample size n. The blue dashed lines in (b) and (c) show the optimal power and welfare of CPL based on oracle correct models $( \mu _ { 1 } , \mu _ { 0 } )$ without any estimation error.

Simulation results. Figure 2 presents the results for the five methods across various settings. First, the three baselines lead to a drastic violation of the target harm rate. The Threshold method relies on accurate outcome models to ensure consistency of $\hat { \gamma } ( \cdot )$ , which is dificult to satisfy with limited labeled data. Policy-tree, which focuses on welfare maximization can incur large harm rates. Finally, Li et al. achieves harm rate only when the two potential outcomes are non-negatively correlated given the features; as a result, it leads to exceedingly high harm rate due to (i) worst-case negative coupling and (ii) inconsistency due to missing covariates in settings 2 and 4.

On the other hand, the two variants of CPL control the harm rate tightly at the target level, showing both the validity and sharpness of CPL. The two variants based on diferent score functions demonstrate negligible diference: in these settings, the rank of instances based on the two optimal scores does not drastically change the decisions by CPL. In practice, this means that researchers can approximately maximize both power and welfare together via CPL without worrying about the tradeof between the two objectives. Finally, compared with the oracle power and welfare (blue dashed lines), the CPL methods achieve close-to-optimal performance, and the gap shows the impact of estimation error in $\hat { \mu } _ { t }$ functions. Such gap seems to be moderate even for misspecified models (Logistic regression in settings 3 and 4).

![](images/f3490b2e53484dbc6f226f58ed768c81cceac2db9618aab5c74dccd062c1da61.jpg)  
Figure 3: (a) Empirical harm rate, (b) power (probability of treatment) and (c) average welfare of the seven procedures in the arm-informativeness experiments at level $\alpha = 0 . 1$ over 200 independent runs. Within each panel, each column represents a prediction model (logistic regression, random forest) for $( \hat { \mu } _ { 1 } , \hat { \mu } _ { 0 } )$

## 6.1.2 Efectiveness of selective calibration

We now use another set of experiments to dive deeper into the selective calibration mechanism. Following the discussion in Section 3.4, the harm rate control by using the conservative proxy $Y _ { i } ^ { \dagger } = T _ { i } Y _ { i } + ( 1 - T _ { i } ) ( 1 - Y _ { i } )$ is tight if a selected sample satisfies (i) $T _ { i } = 1$ and $\gamma ( X _ { i } ) = 1 - \mu _ { 1 } ( X _ { i } )$ , or (ii) $T _ { i } = 0$ and $\gamma ( X _ { i } ) = \mu _ { 0 } ( X _ { i } )$ . Our selective calibration method in Section 3.2 aims to address this by adaptively selecting the informative arm. In this part, we vary the magnitudes of $\mu _ { 1 } ( x )$ and $\mu _ { 0 } ( x )$ to how this strategy contributes to the statistical eficiency. We vary the proportion of samples obeying $\gamma ( X ) = 1 - \mu _ { 1 } ( X )$ (treated arm is informative) and those obeying $\gamma ( X ) = \mu _ { 0 } ( X )$ (control arm is informative); see Appendix E.2 for the detailed data-generating process. In addition to the two CPL variants evaluated in Section 6.1.1, we evaluate two more procedures:

(6) CPL-treat, our method in Section 3.1 with treated samples in $\mathcal { D } _ { \mathrm { c a l i b } }$ and $V ( x , y ) = M y - \hat { \mu } _ { 1 } ( x )$ with a suficiently large constant $M > 0$

(7) CPL-control, our method in Section 3.1 with control samples in $\mathcal { D } _ { \mathrm { c a l i b } }$ and $V ( x , y ) = M y + \hat { \mu } _ { 0 } ( x )$ with a suficiently large constant $M > 0$

The procedures are evaluated in terms of harm rate, power (fraction of treatment), and welfare, with the metrics averaged over 200 independent runs.

The results are summarized in Figure 3, where the x-axis is the fraction of treated samples being informative $( \mathrm { i . e . , } \gamma ( X ) = 1 - \mu _ { 1 } ( X ) )$ . As before, all variants of CPL control the harm rate below $\alpha = 0 . 1$ . CPL-treat and CPL-control demonstrate clear power tradeofs: CPL-treat is more powerful when the treated arm is more informative (x-axis above 0.5), and the opposite happens otherwise. Importantly, CPL-sel-power and CPL-sel-welfare are often comparable to the more powerful single-arm variants, showing the efec tiveness of selective calibration. This justifies our recommendations for the asymptotically optimal variants (Section 3.3), since in practice it is often unknown which arm might be more powerful.

## 6.1.3 Performance under subgroup heterogeneity

To further inspect the behavior of CPL under treatment efect heterogeneity, we design a setting with three subgroups driven by the first two features in $X \in \mathbb { R } ^ { 2 0 }$ (Figure 4 panel (a)). There is strong cross-group heterogeneity, but the units in the same group are largely similar. Studying the decisions in each group ofers a zoom-in observation of “who gets treated” with diferent scoring functions $s ( \cdot )$ in CPL.

The first group consists of half of the population, whose worst-case harm rate $\gamma ( X )$ is relatively large while the conditional average treatment efect $\tau ( X ) = \mu _ { 1 } ( X ) - \mu _ { 0 } ( X )$ is also large. Groups 2 and 3 consist of $1 / 4$ of the population each, whose worst-case harm rate $\gamma ( X )$ and conditional average treatment efect $\tau ( X )$ are both small; the main diference is that the treated samples are more “informative” in Group 2 $( \mathrm { i . e . , ~ 1 - } \mu _ { 1 } ( X ) < \mu _ { 0 } ( X ) )$ , while the control samples are more informative in Group 3. We follow the same procedures as before to evaluate the four variants of CPL.

The results averaged over 200 independent simulation runs are summarized in Figure 4. In panel (b), we show the fraction of $T _ { n + j }$ within each group based on random forests predictions. Diferent score functions lead to diferent treatment prioritization patterns. CPL-treat concentrates the safe treatment budgets on group 2 (small $\gamma ( X )$ with treatment arm being informative) since it ranks instances based on ${ \hat { \mu } } _ { 1 } ( x )$ , while CPL-control concentrates the budgets on group 3. In contrast, CPL-sel-power distributes the budgets relatively uniformly across groups 2 and 3 (since their harm rates are comparably small) by using the score function ˆγ(x). CPL-sel-welfare, which ranks instances by balancing harm rate and treatment efect size, puts most budgets on Group 1 (large treatment efects) but assigns fewer treatments in general.

![](images/22fad61afb3e28592a356ddd7347212aea7974984c1152565720912505a13233.jpg)

![](images/84bf878dd466647d4d2565a43ddb98481303d4031a5c6f1564ad55cbace2e003.jpg)  
Figure 4: (a) Subgroup setup, (b) Per-group fraction of $T _ { n + 1 } = 1$ for variants of $\mathrm { C P L }$ in the subgroup DGP at level $\alpha = 0 . 0 1$ . Panels (c-d) show random forests (RF) as the outcome model only.

## 6.2 Stratified Experiments and Observational Studies

In this part, we proceed to evaluate CPL in stratified experiments and observational studies. The stratified experiments induce a known covariate shift between the calibration and test data, while in observational studies this covariate shift needs to be estimated.

Simulation settings. We first sample the triplets $\{ ( X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) \} _ { i = 1 } ^ { n + m }$ using data generating processes to be specified later. In the labeled data, the treatment assignments are sampled by $T _ { i } \mid X _ { i } \sim$ Bernoull $. ( e ( X _ { i } ) )$ independently with propensity score function $e ( x )$ to be specified later. The observed outcome is $Y _ { i } = Y _ { i } ( T _ { i } )$

Methods and evaluation. Fixing the confidence level at $\alpha = 0 . 1$ , we compare the following procedures:

(1) Li et al., Li et al. (2023) with a doubly-robust estimator and constrained optimization among depth-2 decision trees via econml python library, where the outcome models and propensity scores are estimated under two-fold cross-fitting. The policy is learned on $\mathcal { D } _ { \mathrm { l a b e l } }$ and evaluated on $\mathcal { D } _ { \mathrm { t e s t } }$ . This method requires that the potential outcomes are non-negatively correlated, which is violated here.

(2) Poli $\mathsf { c y - T r e e }$ , which maximizes welfare $\mathbb { E } [ Y ( \pi ( X _ { n + 1 } ) ) ]$ among depth-2 decision trees with econml. The policy is learned with $\mathcal { D } _ { \mathrm { l a b e l } }$ and evaluated on $\mathcal { D } _ { \mathrm { t e s t } }$ with similar cross-fitting estimation of outcome and propensity score models. While this method is not designed for safety control, we include it as the baseline because of its popularity as a standard practice.

(3) Threshold, the same as in Section 6.1 which calibrates the threshold using cumulative estimated $\hat { \gamma } ( X )$ on the test data. This method is valid only when the estimator $\hat { \gamma }$ is suficiently accurate.

(4) CPL-sel-welfare, the method in Section 5 using the same $g ( x , t )$ and $s ( x )$ functions as in the randomized experiment case. The weights are estimated using learn-then-balance with features ${ \hat { \phi } } ( x ) =$ $( \hat { \gamma } ( x ) \mathbb { 1 } \{ s ( x ) \leq \hat { t } \} , \hat { w } ( x ) )$ , where ˆγ and ˆw are estimated using logistic regression or random forests.

(5) CPL-sel-power, the method in Section 5 using the same $g ( x , t )$ and $s ( x )$ functions as in the randomized experiment case. and the weights are estimated in the same way as CPL-sel-welfare.

(6) Plugin-oracle-sel-welfare, the method in Section 4 using the correct weights and the same $g ( x , t )$ and $s ( x )$ functions as CPL-sel-welfare.

![](images/7903d8c31cd576ab5bd3428ee6b260ff5383e1bc8ef67b3c629ee42509d1b11a.jpg)  
Figure 5: (a) Empirical harm rate, (b) power (probability of treatment in the test units), (c) average welfare at level $\alpha = 0 . 1$ in observational studies, averaged over $N = 2 0 0$ runs. Each row is a prediction model (logistic regression, random forest) for $( \hat { \mu } _ { 1 } , \hat { \mu } _ { 0 } )$ , and each column is a data-generating process. The x-axis is the total labeled sample size n. The Plugin methods use the true weights.

(7) Plugin-oracle-sel-power, the method in Section 4 using the correct weights and the same $g ( x , t )$ and s(x) functions as CPL-sel-power.

Here, the last two plugin-oracle methods essentially evaluate CPL in the stratified experiment setting where the propensity score $e ( x )$ , and hence the covariate shift weights, are known. Our theory implies their finite-sample harm rate control due to weighted exchangeability, no matter the accuracy of the scores and selective calibration functions. On the other hand, CPL-sel-power and CPL-sel-welfare evaluate the robustness of CPL in observational studies with estimated weights.

## 6.2.1 Harm rate control with known or estimated weights

We first evaluate the harm rate control using the same data-generating process as in Section 6.1, together with a linear or nonlinear propensity score. The empirical harm rate, fraction of treatment, and average welfare, averaged over 200 independent simulation runs, are summarized in Figure 5. First, the three baselines drastically violate the harm rate control due to similar reasons as in Section 3, now with additional estimation challenges in observational settings. The two plug-in oracles, Plugin-oracle-sel-power and Plugin-oracle-sel-welfare, confirm the finite-sample harm-rate control of Theorem 4.1. Finally, the two methods with estimated weights, CPL-sel-power and CPL-sel-welfare, also demonstrate robust harm-rate control below the target level. Although the four CPL methods rely on distinct scoring functions, they achieve similar performance in terms of fraction of treatment and average welfare, showing that the two objectives can align.

![](images/e72d68be414066938c657963941d1f21cabf5c05b3807be504e6bb238a2688be.jpg)  
Figure 6: Empirical harm rate in the double robustness experiments; results are averaged over $N = 2 0 0$ independent runs. Each column corresponds to one specification of outcome and propensity score models. Each row corresponds to a total labeled sample size. The x-axis is the strength of confounding.

## 6.2.2 Double robustness

We now additionally examine the double robustness property established in Section 5. We design a datagenerating process where the outcome models and propensity score model are all logistic in some nonlinear transformation of the raw features $X \in \mathbb { R } ^ { 8 }$ ; see Appendix E.3 for details. We sample observational data $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ as the labeled dataset, where a random subset of 75% is used as the training fold for training the models $\mu _ { 1 } ( \cdot ) , \mu _ { 0 } ( \cdot )$ , and $e ( \cdot )$ , while the remaining is used as the calibration data in CPL. We vary a parameter in the propensity score model to control the strength of confounding (the x-axis of Figure 6).

The methods evaluated include CPL-sel-power/welfare based on estimated propensity scores, and two known-propensity baselines Plugin-oracle-sel-power/welfare used as oracle comparison only. The four procedures are implemented in the same way as in the preceding parts. Within each procedure, we vary the model classes of the outcome and propensity score models to demonstrate the double robustness property. A correct logistic model runs logistic regression over the nonlinearly-transformed features, while a misspecified logistic model runs logistic regression over the raw features. By Theorem 5.1, we expect CPL-sel-power/welfare to control the harm rate when either of them is correctly specified. Finally, we also consider that both outcome models and propensity scores are trained via random forests, which typically have slower-than-parametric convergence rates yet are less prone to model misspecification; we expect it to control the harm rate, especially when the labeled sample size is suficiently large.

The empirical harm rate averaged over 200 independent simulation runs at nominal level $\alpha = 0 . 0 5$ is summarized in Figure 6. In the first three columns, when either the outcome models or the propensity score model is well-specified, we observe tight harm-rate control by the two CPL variants, which is also close to the plugin-oracle ones. This validates the double robustness theory. When the outcome models are misspecified, we observe a larger slack in the harm rate than the other two configurations when sample size is small. On the other hand, when both models are misspecified (the fourth column), the two CPL variants can violate the harm rate control, yet the violation is moderate. We note that we design the settings deliberately to fail the both-misspecified procedures. In many other settings, misspecification can create a slack in the conservative calibration through the proxy outcome $Y _ { i } ^ { \dagger }$ , which often compensates the misspecification in the weights and keeps the harm rate below the budget even though both models are wrong. Finally, the nonparametric models in the fifth column yield tight harm rate control, showing the robustness to model misspecification and the quick convergence of the harm rate in CPL.

## 7 Empirical Application to AI-Powered Interventions

In this section, we apply CPL to a real-world dataset in the social sciences, where the AI model, ChatGPT, is used as an intervention to persuade participants out of some conspiracy beliefs, in which events are understood as being caused by secret, malevolent plots involving powerful conspirators (Costello et al., 2024). The original study found that brief conversations with AI could reduce conspiracy beliefs by 20 percent on average, and the efect was durable for at least 2 months. Given the societal concern over widespread conspiracy theories, an increasing number of researchers and policymakers are evaluating similar AI interventions as a scalable solution. If policymakers scale up such AI interventions, it is of significant importance to consider safety, as the treatment efects of AI interventions are likely to be highly heterogeneous for several reasons. First, “people believe a wide range of conspiracies, and the specific evidence brought to bear in support of even a particular conspiracy theory may difer substantially from believer to believer” (page 1, Costello et al., 2024). Second, as AI chatbots treat people with natural texts as the intervention, the content of the treatment itself is heterogeneous and unpredictable. Here, to safely scale up these types of AI-powered interventions, we use CPL to decide who should be treated by AI by controlling the harm rate with the safety guarantee.

In this study, before the experiment, the participants stated a conspiracy theory they believed in and reported a numerical score quantifying their belief in it. They are then randomly assigned to treated and control groups, where treated participants engage in a live conversation with a GPT model that is instructed to talk them out of the conspiracy, while the participants in the control condition engage with a neutra conversation with the GPT model. After the experiment, they again report a numerical score quantifying their belief in the same conspiracy theory.

We take all participants in the raw dataset as the analysis population. We binarize the outcome Y to indicate whether the post-experiment belief score is below 50, a cutof the authors originally used to define their analysis population. The participants are randomly split into training (40%), calibration (40%), and test (20%) folds. There are 416 participants originally treated out of 667 participants in the test fold. The fraction of $Y = 1$ in the treated group is 0.274, while the fraction of $Y = 1$ in the control group is 0.100. We consider a stringent harm rate of $\alpha = 0 . 0 2 5$

We build features based on participants’ demographic covariates (education, age, gender, race, religion), political and psychological covariates, AI-related covariates (familiarity and trust in generative AI), baseline belief state variables and textual embedding for the stated conspiracy. The training fold is used to fit the outcome models and conditional treatment efect models, which are used in a similar way as in the simulations to build welfare-maximizing score functions (except that we truncate on extremely small estimated harm rate for stability). See Appendix E.4 for the detailed implementation.

Empirical welfare and power. We report power (the fraction of treated units in the test data) and (estimated) empirical welfare of the welfare-maximizing and power-maximizing variants of CPL. We also compare the results against three baselines as references: the first is to treat everyone in the test data (All treat), the second is to treat no one in the test data (All control), and the third is to treat 2.5% of test units, which trivially satisfies the safety constraint.

Because we observe the realized outcome in the test data, we can estimate the average welfare and harm rate of the policy as follows. Let $T _ { n + j } ^ { \mathrm { r e a l } } \in \{ 0 , 1 \}$ be the actual treatment assigned by the experiment, and $T _ { n + j } = \hat { \pi } ( X _ { n + j } )$ be the decision produced by CPL. We are interested in the average welfare Welfare(ˆπ) := $\mathbb { E } [ \dot { Y _ { n + j } } ( \hat { \pi } ( X _ { n + j } ) ) ] = \mathbb { E } [ Y _ { n + j } ( 1 ) \cdot \hat { \pi } ( X _ { n + j } ) + Y _ { n + j } ( 0 ) \cdot ( 1 - \hat { \pi } ( X _ { n + j } ) ) ]$ , for which an unbiased estimate is

$$
\begin{array} { r } { \mathrm { W e l f a r e } = \widehat { \mathbb { E } } _ { \mathrm { t e s t } } \big [ Y _ { n + j } \cdot \widehat { \pi } ( X _ { n + j } ) | T _ { n + j } ^ { \mathrm { r e a l } } = 1 \big ] + \widehat { \mathbb { E } } _ { \mathrm { t e s t } } \big [ Y _ { n + j } \cdot ( 1 - \widehat { \pi } ( X _ { n + j } ) ) | T _ { n + j } ^ { \mathrm { r e a l } } = 0 \big ] , } \end{array}\tag{7.1}
$$

where $\hat { \mathbb { E } } _ { \mathrm { t e s t } }$ denotes the empirical average in the test fold. We also estimate the harm rate $\mathbb { P } ( Y _ { n + j } ( 1 ) <$ $Y _ { n + j } ( 0 ) , \hat { \pi } ( X _ { n + j } ) = 1 )$ by the following conservative estimator.

$$
\begin{array} { r l } & { \mathrm { H a r m } = \widehat { \mathbb { E } } _ { \mathrm { t e s t } } \big [ \big ( 1 - Y _ { n + j } \big ) \mathbb { 1 } \{ 1 - \widehat { \mu } _ { 1 } \big ( X _ { n + j } \big ) \leq \widehat { \mu } _ { 0 } \big ( X _ { n + j } \big ) \big \} \widehat { \pi } \big ( X _ { n + j } \big ) \big | T _ { n + j } ^ { \mathrm { r e a l } } = 1 \big ] } \\ & { \qquad + \widehat { \mathbb { E } } _ { \mathrm { t e s t } } \big [ Y _ { n + j } \mathbb { 1 } \{ 1 - \widehat { \mu } _ { 1 } \big ( X _ { n + j } \big ) > \widehat { \mu } _ { 0 } \big ( X _ { n + j } \big ) \big \} \widehat { \pi } \big ( X _ { n + j } \big ) \big | T _ { n + j } ^ { \mathrm { r e a l } } = 0 \big ] . } \end{array}\tag{7.2}
$$

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>All treat</td><td rowspan=1 colspan=1>All control</td><td rowspan=1 colspan=1>Trivially-safe</td><td rowspan=1 colspan=1>CPL-welfare</td><td rowspan=1 colspan=1>CPL-power</td></tr><tr><td rowspan=1 colspan=1>Est. harm rate</td><td rowspan=1 colspan=1>0.0392 (0.0111)</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.001 (0.0002)</td><td rowspan=1 colspan=1>0.0247 (0.0094)</td><td rowspan=1 colspan=1>0.0207 (0.0086)</td></tr><tr><td rowspan=1 colspan=1>Num. treatment</td><td rowspan=1 colspan=1>667</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>605</td><td rowspan=1 colspan=1>615</td></tr><tr><td rowspan=1 colspan=1>Est. welfare</td><td rowspan=1 colspan=1>0.274 (0.0218)</td><td rowspan=1 colspan=1>0.0996 (0.0189)</td><td rowspan=1 colspan=1>0.104 (0.0184)</td><td rowspan=1 colspan=1>0.254 (0.0246)</td><td rowspan=1 colspan=1>0.250 (0.0231)</td></tr></table>

Table 1: Estimated harm rate, number of treatment, and estimated welfare (standard deviation) on the test fold using diferent methods. “All treat” treats all test units; “All control” treats no test units; “Triviallysafe” randomly treats α-fraction of test units. The welfare/harm rate estimates are based on (7.1) and (7.2).

Note that Harm is unbiased and asymptotically normal for the population quantity<sup>ˆ</sup> $\mathbb { E } [ \{ ( 1 - \mu _ { 1 } ( X _ { n + j } ) ) \} \mathbb { 1 } \{ 1 -$ $\hat { \mu } _ { 1 } ( X _ { n + j } ) \leq \hat { \mu } _ { 0 } ( X _ { n + j } ) \} + \mu _ { 0 } ( X _ { n + j } ) { \mathbb 1 } \{ 1 - \hat { \mu } _ { 1 } ( X _ { n + j } ) > \hat { \mu } _ { 0 } ( X _ { n + j } ) \} \} \cdot \hat { \pi } ( X _ { n + j } ) \rfloor$ , which upper bounds the harm rate. This is a valid conservative estimator of the harm rate even if $\hat { \mu } _ { t } ( x )$ is misspecified, and this is a consistent estimator for the sharp upper bound of the harm rate when $\hat { \mu } _ { t } ( x )$ is consistently estimated.

Results. The main results are summarized in Table 1. Several points are worth noting. First, treating everyone (All treat) violates the safety constraint as its harm rate is 3.92% and exceeds 2.5%. In contrast, our proposed CPL methods achieve the tight control of harm rates at 2.5% and 2.1%, respectively. Second, while treating no one (All control) and treating only 2.5% of test units (Trivially-safe), of course, satisfy the safety constraint, they have extremely low power and welfare. In contrast, our proposed methods can treat more than 90% of test units and achieve the high average welfare. While maintaining the safety constraint, our method simultaneously achieves high power and welfare. Finally, it is important to note that the diference between our welfare-maximizing and power-maximizing variants are minimal in practice. It is true that, consistent with our theory, our power-maximizing variant has a slightly higher fraction of treated test units, and our welfare-maximizing variant has a slightly higher average welfare. However, overall, their actual policy decision on who gets treated is similar, which implies that researchers can use either variant in practice and expect to approximately optimize both power and welfare in various applications.

Safe treatments by CPL. We now take a closer look at the decisions produced by the welfare-maximizing varinat of CPL. Figure 7 visualizes the treatment decisions, where panel (a) plots the test points based on the predicted harm rate $\hat { \gamma } ( X )$ and predicted treatment efect ${ \hat { \tau } } ( X )$ , while panel (b) plots the test points based on the predicted outcomes ${ \hat { \mu } } _ { 0 } ( X )$ and ${ \hat { \mu } } _ { 1 } ( X )$ . CPL treats the test units with the largest values of $\hat { \tau } ( X ) / \hat { \gamma } ( X )$ . The decision boundary is plotted in both panels, and the region not treated is indicated in light grey. While the actual harm $\mathbb { 1 } \{ Y _ { n + j } ( 1 ) < Y _ { n + j } ( 0 ) \}$ is not observed, we conservatively estimate it by checking the label $Y _ { n + j } ^ { \dagger } = T _ { n + j } ^ { \mathrm { r e a l } } Y _ { n + j } + ( 1 - T _ { n + j } ^ { \mathrm { r e a l } } ) ( 1 - Y _ { n + j } )$ , where $T _ { n + j } ^ { \mathrm { r e a l } }$ is the actual treatment received by the j-th test unit, among those with $\hat { g } ( X _ { n + j } , T _ { n + j } ^ { \mathrm { r e a l } } ) = 1$ , i.e., either $\hat { \mu } _ { 1 } ( X _ { n + j } ) \leq \hat { \mu } _ { 0 } ( X _ { n + j } )$ and $T _ { n + j } ^ { \mathrm { r e a l } } = 1$ or $\hat { \mu } _ { 1 } ( X _ { n + j } ) > \hat { \mu } _ { 0 } ( X _ { n + j } )$ and $T _ { n + j } ^ { \mathrm { r e a l } } = 0$ . The colored dots are those with $\hat { g } ( X _ { n + j } , T _ { n + j } ^ { \mathrm { r e a l } } ) = 1$ , among which the blue dots are those with $Y _ { n + j } ^ { \dagger } = 1$ and the red dots are those with $Y _ { n + j } ^ { \dagger } = 0$ . We observe that very few red dots in the treated region can possibly be harmed.

We further examine how the treatment decision by CPL varies with participants. Figure 8 plots the moving average of predicted treated and control outcomes, as well as the fraction of safe treatments, for every value of pre-experiment belief scores (smoothed by a moving window of size 10). In general, the two outcome curves indicate that a stronger pre-treatment belief makes it less likely to be persuaded out of the conspiracy in both conditions, but the efect of the AI intervention seems strong for participants with a strong pre-treatment belief. The welfare-maximizing variant of CPL prioritizes a treatment-efect-versus-harm-rate tradeof. It mainly treats units with firm pre-treatment belief for whom the treatment is likely to make a huge diference (the gap between the two blue curves) while the estimated harm rate is relatively low.

Figure 9 similarly reports this information among participants with a specific GenAI familiarity score (panel a) and GenAI trust score (panel b). In this case, however, the outcomes and treatment decisions do not change significantly based on these features, indicating that AI intervention may be similarly safe for users with diferent familiarity with or trust in GenAI.

![](images/34bce5f1190de4f35070c60cc4ac42c49f6ae001d97df5404895116d53c0f26b.jpg)

![](images/bb9feee04d32a77bcebce3a21590e0cf0ee3e4d098be7f125a61658c86611d6f.jpg)

Figure 7: Visualization of treatment decisions by CPL using the welfare-maximizing variant, where outcome models are estimated by causal forests. Colored dots are those with $\hat { g } ( X , T ) = 1$ , which provide a conservative bound for harm; red dots are those who can possibly be harmed, with $Y ^ { \dagger } = 0$  
![](images/fbac429b3b9257a72be19ad2cac38e2796743d70d5e221251e8521b6431c7b50.jpg)  
Figure 8: Fraction of safe treatment (red), average predicted treated (dark blue) and control (light blue) outcomes among test participants within a moving window of self-reported pre-experiment conspiracy belief.

![](images/e27131a3f8eff7a88d7bbe39ecc3aaea5e5dfb7ba540bbb1cdcab984e621918d.jpg)

![](images/62d837de2f4684039da6b0c08ad38082bcdb00838de6b7060bcc216378edf2d8.jpg)  
Figure 9: Fraction of treatment by the welfare-maximizing variant (red), average predicted treated (dark blue) and control (light blue) outcomes among test units stratified by self-reported GenAI-related variables.

## 8 Discussion

In this article, we developed the CPL approach that allows for individualized treatment assignment with a safety guarantee. We prove that when learning a treatment assignment rule from randomized experiments, CPL provides the safety guarantee in a finite sample without imposing any modeling assumption about potential outcomes. Additionally, when the outcome regression model is consistently estimated as assumed in many existing methods, CPL also asymptotically achieves the optimal welfare and power with appropriate choices of score V and inclusion g functions. We then extended our results to observational studies and derived novel asymptotic doubly robust safety guarantees for CPL.

We now briefly discuss several natural extensions of our proposed conformal policy learning. The first concerns external validity settings where the test units may come from a diferent superpopulation than the labeled data and Assumption 2.2 is violated (e.g., Egami and Hartman, 2023; Jin et al., 2025a). The most common and natural strategy is to relax the i.i.d. assumption to the covariate shift assumption, i.e., the distributions of the labeled data and the test data difer only in the covariate distribution. Under this setting, we can generalize CPL by simply multiplying the current conformal weights by additional weights $w _ { Q / P } ( x ) : = \mathrm { d } \mathbb { Q } _ { X } / \mathrm { d } \mathbb { P } _ { X } ( x )$ that account for the covariate shift density ratio where P denotes the distribution of the labeled data and Q represents the distribution of the test data. Assuming both $e ( x )$ and $w _ { Q / P } ( x )$ are known, CPL then proceeds in exactly the same way as in Section 4. When either of them is unknown, one may use similar strategies as in Section 5 to estimate and plug in these quantities.

The second natural extension concerns the control of harm rate among subgroups. In many problems where fairness and equity are stressed, it is desirable to maintain the harm rate control within each subgroup, such as those defined by demographic features (Romano et al., 2020). Following the setup in the main framework, our goal is to find the treatment assignment rule $\hat { \pi } ( X _ { n + 1 } ) ~ \in ~ \{ 0 , 1 \}$ such that, for a group indicator $\mathcal { G } ( \cdot ) , \mathbb { P } ( Y _ { n + 1 } ( \pi ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) | \mathcal { G } ( X _ { n + 1 } ) = 1 )$ . Given a new sample with $\mathcal { G } ( X _ { n + 1 } ) = 1$ , this can be achieved by taking the calibration data from the same subgroup, i.e., $\{ ( X _ { i } , T _ { i } , Y _ { i } ) \colon \mathcal { G } ( X _ { i } ) = 1 \}$ . These data are induced by the full data in the subgroup $\{ ( X _ { i } , Y _ { i } ( 1 ) , Y _ { i } ( 0 ) ) \colon \mathcal { G } ( X _ { i } ) = 1 \}$ which is exchangeable with the new test sample conditional on the group indicator. The same CPL procedures can then be performed within the subgroup for randomized experiments and observational studies.

## Acknowledgments

The authors thank Eli Ben-Michael and Molly Ofer-Westort for helpful discussions at the Online Causal Inference Seminar and the Political Methodology summer meeting, respectively. We also thank seminar participants at Yale Economics, Harvard Applied Stats Workshop, University of Tokyo Applied Stats Seminar, and the Political Methodology summer meeting. Y.J. is partially supported by NSF DMS-2610282.

## References

Athey, S. and Wager, S. (2021). Policy learning with observational data. Econometrica, 89(1):133–161.

Bai, H., Voelkel, J. G., Muldowney, S., Eichstaedt, J. C., and Willer, R. (2025). LLM-generated Messages can Persuade Humans on Policy Issues. Nature Communications, 16(1):6037.

Bai, T. and Jin, Y. (2024). Optimized conformal selection: Powerful selective inference after conformity score optimization. arXiv preprint arXiv:2411.17983.

Bastani, H., Bastani, O., Sungu, A., Ge, H., Kabakcı, O., and Mariman, R. (2025). Generative AI without <sup>¨</sup> Guardrails can Harm Learning: Evidence from high School Mathematics. Proceedings of the National Academy of Sciences, 122(26):e2422633122.

Bates, S., Cand\`es, E., Lei, L., Romano, Y., and Sesia, M. (2021). Testing for outliers with conformal p-values. arXiv preprint arXiv:2104.08279.

Ben-Michael, E., Greiner, D. J., Imai, K., and Jiang, Z. (2025). Safe policy learning through extrapolation: Application to pre-trial risk assessment. Journal of the American Statistical Association, 120(551):1386– 1399.

Ben-Michael, E., Imai, K., and Jiang, Z. (2024). Policy learning with asymmetric counterfactual utilities. Journal of the American Statistical Association, 119(548):3045–3058.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters.

Costello, T. H., Pennycook, G., and Rand, D. G. (2024). Durably Reducing Conspiracy Beliefs through Dialogues with AI. Science, 385(6714):eadq1814.

Dell’Acqua, F., McFowland III, E., Mollick, E. R., Lifshitz-Assaf, H., Kellogg, K., Rajendran, S., Krayer, L., Candelon, F., and Lakhani, K. R. (2026). Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Efects of AI on Knowledge Worker Productivity and Quality. Organization Science.

Dud´ık, M., Langford, J., and Li, L. (2011). Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601.

Egami, N. and Hartman, E. (2023). Elements of external validity: Framework, design, and analysis. American Political Science Review, 117(3):1070–1088.

Gadbury, G. L., Iyer, H. K., and Albert, J. M. (2004). Individual treatment efects in randomized trials with binary outcomes. Journal of Statistical Planning and Inference, 121(2):163–174.

Hainmueller, J. (2012). Entropy balancing for causal efects: A multivariate reweighting method to produce balanced samples in observational studies. Political analysis, 20(1):25–46.

Heckman, J. J., Smith, J., and Clements, N. (1997). Making the Most Out of Programme Evaluations and Social Experiments: Accounting for Heterogeneity in Programme Impacts. The Review of Economic Studies, 64(4):487–535.

Hirano, K. and Porter, J. R. (2009). Asymptotics for Statistical Treatment Rules. Econometrica, 77(5):1683– 1701.

Holland, P. W. (1986). Statistics and Causal Inference. Journal of the American Statistical Association, 81(396):945–960.

Huang, Y., Gilbert, P. B., and Janes, H. (2012). Assessing treatment-selection markers using a potential outcomes framework. Biometrics, 68(3):687–696.

Imai, K. and Strauss, A. (2011). Estimation of heterogeneous treatment efects from randomized experiments, with application to the optimal planning of the get-out-the-vote campaign. Political Analysis, 19(1):1–19.

Imbens, G. W. and Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. Cambridge University Press.

Jia, Z., Ben-Michael, E., and Imai, K. (2025). Bayesian safe policy learning with chance constrained optimization: Application to military security assessment during the vietnam war. Journal of the Royal Statistical Society Series A: Statistics in Society, page qnaf122.

Jin, Y. and Cand\`es, E. J. (2023a). Model-free selective inference under covariate shift via weighted conformal p-values. arXiv preprint arXiv:2307.09291.

Jin, Y. and Cand\`es, E. J. (2023b). Selection by prediction with conformal p-values. Journal of Machine Learning Research, 24(244):1–41.

Jin, Y., Egami, N., and Rothenh¨ausler, D. (2025a). Beyond reweighting: On the predictive role of covariate shift in efect generalization. Proceedings of the National Academy of Sciences, 122(45):e2427181122.

Jin, Y., Ren, Z., and Cand\`es, E. J. (2023). Sensitivity analysis of individual treatment efects: A robust conformal inference approach. Proceedings of the National Academy of Sciences, 120(6):e2214889120.

Jin, Y., Ren, Z., Yang, Z., and Wang, Z. (2025b). Policy Learning “without” Overlap: Pessimism and Generalized Empirical Bernstein’s Inequality. The Annals of Statistics, 53(4):1483–1512.

Jin, Y. and Zubizarreta, J. (2025). Cross-balancing for data-informed design and eficient analysis of observational studies. arXiv preprint arXiv:2511.15896.

Kallus, N. (2022). What’s the harm? sharp bounds on the fraction negatively afected by treatment. Advances in Neural Information Processing Systems, 35:15996–16009.

Kitagawa, T. and Tetenov, A. (2018). Who should be treated? empirical welfare maximization methods for treatment choice. Econometrica, 86(2):591–616.

Kleinberg, J., Lakkaraju, H., Leskovec, J., Ludwig, J., and Mullainathan, S. (2018). Human decisions and machine predictions. The quarterly journal of economics, 133(1):237–293.

Kosorok, M. R. and Laber, E. B. (2019). Precision medicine. Annual review of statistics and its application, 6(1):263–286.

Laber, E. B., Wu, F., Munera, C., Lipkovich, I., Colucci, S., and Ripa, S. (2018). Identifying optimal dosage regimes under safety constraints: An application to long term opioid treatment of chronic pain. Statistics in medicine, 37(9):1407–1418.

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523):1094–1111.

Lei, L. and Cand\`es, E. J. (2021). Conformal inference of counterfactuals and individual treatment efects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(5):911–938.

Li, H., Zheng, C., Cao, Y., Geng, Z., Liu, Y., and Wu, P. (2023). Trustworthy policy learning under the counterfactual no-harm criterion. In International Conference on Machine Learning, pages 20575–20598. PMLR.

Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pages 661–670.

Manski, C. F. (2004). Statistical Treatment Rules for Heterogeneous Populations. Econometrica, 72(4):1221– 1246.

Murphy, S. A. (2003). Optimal Dynamic Treatment Regimes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 65(2):331–355.

Qian, M. and Murphy, S. A. (2011). Performance guarantees for individualized treatment rules. Annals of statistics, 39(2):1180.

Richens, J., Beard, R., and Thompson, D. H. (2022). Counterfactual harm. Advances in Neural Information Processing Systems, 35:36350–36365.

Romano, Y., Barber, R. F., Sabatti, C., and Cand\`es, E. (2020). With malice toward none: Assessing uncertainty via equalized coverage. Harvard Data Science Review, 2(2):4.

Rubin, D. B. (1980). Randomization Analysis of Experimental Data: The Fisher Randomization Test Comment. Journal of the American statistical association, 75(371):591–593.

Scauda, M., Freidling, T., and Zhao, Q. (2026). Counterfactual optimization of policy interventions: Lexical ordering and leapfrogging. arXiv preprint arXiv:2608.20505.

Shen, C., Jeong, J., Li, X., Chen, P.-S., and Buxton, A. (2013). Treatment benefit and treatment harm rate to characterize heterogeneity in treatment efect. Biometrics, 69(3):724–731.

Skeem, J., Scurich, N., and Monahan, J. (2020). Impact of risk assessment on judges’ fairness in sentencing relatively poor defendants. Law and human behavior, 44(1):51.

Tibshirani, R. J., Barber, R. F., Cand\`es, E. J., and Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. In Advances in Neural Information Processing Systems 32, pages 2526–2536.

Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic learning in a random world. Springer Science & Business Media.

Wang, Y., Fu, H., and Zeng, D. (2018). Learning optimal personalized treatment rules in consideration of benefit and risk: with an application to treating type 2 diabetes patients with insulin therapies. Journal of the American Statistical Association, 113(521):1–13.

Wu, P., Ding, P., Geng, Z., and Liu, Y. (2024). Quantifying individual risk for binary outcome. arXiv preprint arXiv:2402.10537.

Wu, P., Jiang, Q., Luo, S., and Geng, Z. (2025). Safe individualized treatment rules with controllable harm rates. arXiv preprint arXiv:2505.05308.

Xu, Z. and Ramdas, A. (2024). Online multiple testing with e-values. In International Conference on Artificial Intelligence and Statistics, pages 3997–4005. PMLR.

Yin, M., Shi, C., Wang, Y., and Blei, D. M. (2024). Conformal sensitivity analysis for individual treatment efects. Journal of the American Statistical Association, 119(545):122–135.

Yin, Y., Liu, L., and Geng, Z. (2018). Assessing the treatment efect heterogeneity with a latent variable. Statistica Sinica, pages 115–135.

Zhang, Y., Ben-Michael, E., and Imai, K. (2022). Safe policy learning under regression discontinuity designs with multiple cutofs. arXiv preprint arXiv:2208.13323.

Zhang, Z., Wang, C., Nie, L., and Soon, G. (2013). Assessing the heterogeneity of treatment efects via potential outcomes of individual patients. Journal of the Royal Statistical Society Series C: Applied Statistics, 62(5):687–704.

Zhao, Y., Zeng, D., Rush, A. J., and Kosorok, M. R. (2012). Estimating Individualized Treatment Rules using Outcome Weighted Learning. Journal of the American Statistical Association, 107(499):1106–1118.

Zubizarreta, J. R. (2015). Stable weights that balance covariates for estimation with incomplete outcome data. Journal of the American Statistical Association, 110(511):910–922.

## A Deferred discussion

## A.1 Power Maximization

We first define a natural notion of power, the probability of getting treated:

$$
\operatorname { P o w e r } ( \pi ; P ) : = P ( \pi ( X _ { n + 1 } ) = 1 ) .
$$

The optimal treatment rule under this power notion is defined as

$$
\begin{array} { r l } { \pi _ { \mathrm { p o w e r } } ^ { * } = \underset { \pi : \mathcal K \to \{ 0 , 1 \} } { \mathrm { a r g m a x } } } & { \mathrm { P o w e r } ( \pi ; \mathbb { P } _ { X , Y ( 1 ) , Y ( 0 ) } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \underset { P \in \mathcal P } { \mathrm { m a x } } \mathrm { E r r } ( \pi ; P ) \le \alpha . } \end{array}\tag{A.1}
$$

We can rewrite this optimization problem as follows.

$$
\begin{array} { r l } { \pi _ { \mathrm { p o w e r } } ^ { * } = \underset { \pi : \mathcal { X }  \{ 0 , 1 \} } { \mathrm { a r g m a x } } } & { \mathbb { E } [ \pi ( X _ { n + 1 } ) ] } \\ { \mathrm { s u b j e c t ~ t o } } & { \mathbb { E } [ \pi ( X _ { n + 1 } ) \gamma ( X _ { n + 1 } ) ] \leq \alpha . } \end{array}
$$

where $\gamma ( x ) = \mathrm { m i n } \{ 1 - \mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \}$ , which is the sharp upper bound for $\mathbb { P } ( Y _ { n + 1 } ( 1 ) = 0 , Y _ { n + 1 } ( 0 ) = 1 \ | $ $X _ { n + 1 } = x )$ . Intuitively, given the linear relaxation, the optimal treatment rule is a thresholding rule based on $\gamma ( X _ { n + 1 } )$ , which acts as the “cost” in this optimization problem.

Theorem A.1 formally establishes that the optimal solution $\pi _ { \mathrm { p o w e r } } ^ { * }$ under the worst-case harm constraint is based on a cutof on $\gamma ( x )$ . It is implied by a more general result in Theorem A.4 in Appendix A.2 with proof in Appendix D.1; we thus omit the proof of Theorem A.1 here.

Theorem A.1 (Power-optimal $\pi _ { \mathrm { p o w e r } } ^ { * } )$ . Assume $\gamma ( X )$ has no point mass. Then, the optimal solution (A.1) is $\pi _ { p o w e r } ^ { * } ( x ) = \mathbb { 1 } \{ \gamma ( x ) \leq \gamma ^ { * } \}$ for the cutof $\gamma ^ { * } = \operatorname* { m a x } \{ \tilde { \gamma } \in [ 0 , 1 ] \colon \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ \gamma ( X ) \leq \tilde { \gamma } ) \} ] \leq \alpha \}$

To achieve the optimal power, we define the power-optimal conformal p-value (for stratified experiments, covering the complete randomized experiments in Section 3 as a special case)

$$
p _ { n + 1 } ^ { \mathrm { s t r - p o w e r } } = \frac { w ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } w ( X _ { i } ) G _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} } { w ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } G _ { i } w ( X _ { i } ) } ,\tag{A.2}
$$

where $G _ { i } = T _ { i } \mathbf { 1 } \{ 1 - \hat { \mu } _ { 1 } ( X _ { i } ) \leq \hat { \mu } _ { 0 } ( X _ { i } ) \} + ( 1 - T _ { i } ) \mathbf { 1 } \{ 1 - \hat { \mu } _ { 1 } ( X _ { i } ) > \hat { \mu } _ { 0 } ( X _ { i } ) \}$ . The choice of the calibration inclusion indicators $G _ { i }$ is the same as the welfare-optimal procedure. This coincides with Algorithm 1 with the clipped score $V ( x , y ) = M \mathbb { 1 } \{ y > 0 \} + \hat { \gamma } ( x )$ using a suficiently large constant $M > 0$

Theorem $\mathrm { A . 2 }$ establishes the optimality of PCL with the power-oriented conformity score. Its proof is in Appendix D.2.

Theorem A.2. Suppose Assumption 2.1 and Assumption 2.2 hold, and $e ( x )$ is known. Suppose $\| \hat { \mu } _ { t } ( X ) -$ $\mu _ { t } ( X ) \lVert \mathbf { \Psi } _ { L _ { 2 } ( \mathbb { P } _ { X } ) } \stackrel { P } { \to } \textbf { 0 }$ as $n  \infty f o r t \in \{ 0 , 1 \}$ , and $\gamma ( X )$ has no point mass. Let $\widehat { \pi } _ { s t r - p o w e r } ( X _ { n + 1 } ) \ : =$ $\mathbb { 1 } \{ p _ { n + 1 } ^ { s t r - p o w e r } \le \alpha \}$ for the p-value in (A.2). Then $\mathbb { E } [ Y ( \hat { \pi } _ { s t r - p o w e r } ( X _ { n + 1 } ) ) ]  P o w e r ( \pi _ { p o w e r } ^ { * } ; \mathbb { P } )$ as $n  \infty$ where $\pi _ { p o w e r } ^ { * }$ is the power-optimal solution in Theorem A.1.

The theorem below establishes asymptotic power optimality under suitable conditions, in parallel to Theorem 5.4. The proof is in Appendix D.3.

Theorem A.3 (Power optimality with observational data). Let $\hat { \pi } _ { o b s } ( X _ { n + 1 } ) = \mathbb { 1 } \{ p _ { n + 1 } ^ { o b s } \leq \alpha \}$ , where we take the same $G _ { i }$ as (3.12) and the score $\hat { s } ( x ) = \hat { \gamma } ( x ) = \operatorname* { m i n } \{ 1 - \hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ . Suppose Assumption A.10 holds, and $\delta _ { n } = O ( n ^ { - 1 / 2 } )$ . Furthermore, assume $\| \hat { g } - g ^ { * } \| _ { L _ { 2 } ( \mathbb { P } _ { X , T } ) } + \| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } + \| \hat { s } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) \ f o r$ $g ^ { * } ( x , t ) = t \mathbb { 1 } \{ 1 - \mu _ { 1 } ( x ) \leq \mu _ { 0 } ( x ) \} + ( 1 - t ) \mathbb { 1 } \{ 1 - \mu _ { 1 } ( x ) > \mu _ { 0 } ( x ) \}$ , and $\gamma ( X )$ has no point mass. Then $\mathbb { E } [ \hat { \pi } _ { o b s } ( X _ { n + 1 } ) ]  \mathrm { P o w e r } ( \pi _ { \mathrm { p o w e r } } ^ { * } ; \mathbb { P } )$ , where $\pi _ { \mathrm { p o w e r } } ^ { * }$ is the power-optimal solution in Theorem A.1.

## A.2 General form of optimality

Theorem A.4 is a general form of Theorem A.1, whose proof is in Appendix D.1.

Theorem A.4 (Power-optimal $\phi ^ { * }$ , general form). Let $\gamma ( x ) = \operatorname* { m i n } \{ \mathbb { P } ( Y ( 1 ) = 0 | X = x ) , \mathbb { P } ( Y ( 0 ) = 1 | X =$ $x ) \}$ $I f \ \mathbb { E } [ \gamma ( X ) ] \ \leq \ \alpha _ { \mathrm { ; } }$ , then an optimal solution to $\mathrm { ( A . 1 ) }$ is $\phi _ { \mathrm { p o w e r } } ^ { * } ( x ) \ \equiv \ 1$ . Otherwise, define $\gamma ^ { \ast } : =$ inf $\{ c \in [ 0 , 1 ] : \mathbb { E } [ \gamma ( X ) \mathbf { 1 } \{ \gamma ( X ) \leq c \} ] \geq \alpha \}$ , and let $\eta ^ { * } \in [ 0 , 1 ]$ be chosen such that $\mathbb { E } [ \gamma ( X ) \mathbf { 1 } \{ \gamma ( X ) < \gamma ^ { * } \} ] +$ $\eta ^ { * } \mathbb { E } [ \gamma ( X ) \mathbf { 1 } \{ \gamma ( X ) = \gamma ^ { * } \} ] = \alpha$ . Then an optimal solution to (A.1) is $\phi _ { \mathrm { p o w e r } } ^ { * } ( x ) = \mathbf { 1 } \{ \gamma ( x ) < \gamma ^ { * } \} + \eta ^ { * } \mathbf { 1 } \{ \gamma ( x ) =$ $\gamma ^ { * } \}$ . In this case, the safety constraint is binding: $\mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { p o w e r } } ^ { * } ( X ) ] { \stackrel { . } { = } } \alpha$

Theorem A.5 is a general form of Theorem 3.3, whose proof is in Appendix D.4.

Theorem A.5. Consider the randomized extension of problem (3.9), where a policy is a measurable function $\phi : \mathcal { X }  [ 0 , 1 ]$ 1], and $\phi ( x )$ denotes the probability of treatment conditional on $X = x$ . For $r \leq 0$ , define $H ( r ) : = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) < r \} ] . \quad I f H ( 0 ) \leq \alpha$ , let $r ^ { * } = 0$ and $\eta ^ { * } = 0$ $I f \ H ( 0 ) \ > \ \alpha ,$ let $r ^ { * } : =$ sup $\left\{ r < 0 : H ( r ) \leq \alpha \right\}$ , and choose $\eta ^ { * } \in [ 0 , 1 ]$ such that $H ( r ^ { * } ) + \eta ^ { * } \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) = r ^ { * } \} ] = \alpha$ . Then an optimal solution to the randomized extension of (3.9) is

$$
\phi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( x ) < r ^ { * } \} + \eta ^ { * } \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( x ) = r ^ { * } \} .
$$

Moreover, $\mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] \leq \alpha$ and $r ^ { * } \cdot \{ \mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] - \alpha \} = 0$ . In particular, $i f \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ \tau ( X ) >$ $0 \} ] \leq \alpha _ { \mathrm { : } }$ , then $r ^ { * } = 0$ and $\phi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = \mathbb { 1 } \{ \tau ( x ) > 0 \}$ . Otherwise, $r ^ { * } < 0$ and the safety constraint is exactly attained.

## A.3 A general theory for doubly robust guarantee with estimated weights

In this section, we present the general theory for CPL with estimated weights. We consider any estimated weights $\{ \hat { w } _ { i } \} _ { i = 1 } ^ { n + 1 }$ , which may come from a pre-trained weighted function or whose estimation may depend on the data. We recall the definition of conformal p-values:

$$
p _ { n + 1 } ^ { \mathrm { o b s } } = \frac { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } \hat { w } _ { i } G _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { s } ( X _ { n + 1 } ) \} } { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } \hat { w } _ { i } G _ { i } } ,
$$

where the inclusion indicators are $G _ { i } \sim \mathrm { B e r n } ( \hat { g } ( X _ { i } , T _ { i } ) )$ for a function $\boldsymbol { \hat { g } } \colon \mathcal { X } \times \{ 0 , 1 \}  [ 0 , 1 ]$ whose training process is independent of the labeled and unlabeled data.

We introduce several high-level conditions that the weight function and other nuisance functions need to satisfy in order to achieve the doubly robust safety guarantee. Importantly, we will later show that our learn-then-balance weights satisfy these conditions under mild model convergence conditions.

First, we require the weights to satisfy the following approximate balancing condition. This will involve an estimator $\hat { \gamma } ( \cdot )$ of the worst-case harm rate function $\gamma ( x ) = \operatorname* { m i n } \{ 1 - \mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \}$ estimated with data independent of the calibration and test data, such as by plugging in independently estimated outcome models.

Assumption A.6 (Approximate balance). Let $r _ { n } \ > \ 0$ be a deterministic sequence obeying $r _ { n } ~ = ~ o ( 1 )$ The weights obey $\begin{array} { r } { \hat { w } _ { i } \ge 0 , \frac { 1 } { | \mathcal { T } _ { c a l i b } | } \sum _ { i \in \mathcal { T } _ { c a l i b } } ( \hat { w } _ { i } - \hat { \omega } ( X _ { i } ) ) ^ { 2 } = O _ { P } ( r _ { n } ^ { 2 } ) } \end{array}$ for some function $\hat { \omega } ( \cdot ) : \mathcal { X } $ R trained independently of the calibration data and test point, and $\begin{array} { r } { \frac { \hat { w } _ { n + 1 } \vee \operatorname* { m a x } _ { i \in \mathcal { T } _ { c a l i b } } \hat { w } _ { i } } { \sum _ { i \in \mathcal { T } _ { c a l i b } } \hat { w } _ { i } } = O _ { P } ( 1 / n ) } \end{array}$ . In addition, for $\begin{array} { r } { \hat { t } = \operatorname* { s u p } \{ t \in \mathbb { R } \colon \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { t } \} \leq \alpha \} } \end{array}$ , it holds that

$$
\begin{array} { r } { \Big | \frac { 1 } { | { \bar { Z } } _ { c a l b } | } \sum _ { i \in { \bar { Z } } _ { c a l i b } } \hat { w } _ { i } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \big \{ \hat { s } ( X _ { i } ) \leq \hat { t } \big \} - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \big \{ \hat { s } ( X _ { i } ) \leq \hat { t } \big \} \Big | = O _ { P } ( r _ { n } ) , } \end{array}
$$

$$
\begin{array} { r l r } { a n d } & { { } } & { \frac { 1 } { | { \mathcal Z } _ { c a l i b } | } \sum _ { i \in { \mathcal Z } _ { c a l i b } } \hat { w } _ { i } = 1 + O _ { P } ( r _ { n } ) . } \end{array}
$$

In Assumption A.6, the first condition requires the estimated weights $\{ \hat { w } _ { i } \}$ to converge to any independently trained weight function with parametric rates. This is easily satisfied if one set $\hat { w } _ { i } = \hat { w } ( X _ { i } )$ ; we shall see that our balancing program also satisfies this general condition. The key balancing condition requires the estimated weights to approximately balance the capped score functions at the critical cutof t<sup>ˆ</sup>, following ideas in the balancing weights literature (Hainmueller, 2012; Zubizarreta, 2015; Jin and Zubizarreta, 2025) but with specific choice of the balancing features to yield favorable statistical properties.

The following theorem shows that, under Assumptions A.6, the conformal policy learning $\widehat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) : =$ $\mathbb { 1 } \{ p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha \}$ achieves the safety guarantees asymptotically if either the outcome conditional expectation function ${ \hat { \mu } } _ { t } ( x )$ or the weight function $\hat { w } ( x )$ , but not necessarily both, is consistently estimated. The proof of Theorem A.7 is in Appendix C.1. Recall that $\mathcal { T } _ { n }$ is the σ-algebra of the training process.

Theorem A.7 (Model double robustness). Suppose Assumption A.6 holds for some $r _ { n } ~ = ~ o ( 1 )$ . Define $p _ { G } : = \mathbb { P } ( G = 1 \mid { \mathcal { T } } _ { n } )$ and $w ^ { \circ } ( x ) : = p _ { G } w ( x )$ . Then,

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname { l p } ( Y _ { n + 1 } ( \hat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) ) \leq \alpha
$$

under either of the following conditions:

(i). The weight $\hat { \omega } ( \cdot )$ obeys $\| \hat { \omega } - w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 )$ , and there exist constants $c , C > 0$ such that $p _ { G } \geq c$ and $\| w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } \leq C$ with probability tending to one.

(ii). The outcome models obey $\| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } \big ( 1 \big )$ , and there exist constants $c , C , \eta > 0$ such that, with probability tending to one, $q ( x ) \geq c , c \leq \hat { \omega } ( x ) \leq C , x \in \mathcal { X }$ , and the function $H _ { n } ( t ) : = \mathbb { E } [ \hat { \gamma } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq$ $t \} \mid T _ { n } \}$ satisfies $H _ { n } ( t _ { n } ^ { \circ } - u ) \leq \alpha - c u$ and $H _ { n } ( t _ { n } ^ { \circ } + u ) \geq \alpha + c$ u for $0 < u \leq \eta$ , where $t _ { n } ^ { \circ } : = \operatorname* { s u p } \{ t \in \mathbb { R }$ : $H _ { n } ( t ) \leq \alpha \}$ , and $\mathbb { P } \{ a < \hat { s } ( X ) \leq b \mid \mathcal { T } _ { n } \} \leq C ( b - a ) ~ f o r ~ e v e r y ~ a < b .$

Notably, the doubly robust safety guarantees do not require the score function $\hat { s } ( \cdot )$ to converge to any true function, which inherits the model-free nature of conformal prediction. The asymptotic safety guarantee hinges on the convergence of $\hat { w } ( \cdot )$ and/or ˆγ(·). In particular, in the case where the weights do not converge to w(·), consistent outcome models and the balancing condition still ensure harm rate control.

We further show that CPL with balanced weights achieves rate double robustness: if both the outcome models and the weight functions are consistently estimated at slow, nonparametric rates $o _ { P } ( n ^ { - 1 / 4 } )$ , the excess harm rate above α is of the parametric order $O ( 1 / \sqrt { n } )$ . The proof of Theorem A.9 is in Appendix C.2. Recall that $p _ { G } : = \mathbb { P } ( G = 1 \mid { \mathcal { T } } _ { n } )$ and $w ^ { \circ } ( x ) : = p _ { G } w ( x )$

Assumption A.8 (Slow convergence). Let $\hat { \omega } ( \cdot )$ be the reference function in Assumption A.6, and Assume $\| \hat { \omega } - w ^ { \circ } \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } ) \ a n d \ \| \hat { \gamma } - \gamma \| _ { L _ { 1 } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } )$

Theorem A.9 (Rate double robustness). Suppose Assumption A.6 holds with $r _ { n } = O ( n ^ { - 1 / 2 } )$ and Assumption A.8 holds. Suppose there exist deterministic constants $c , C , \eta > 0$ such that, with probability tending to one, the following regularity conditions hold:

(i) $q ( x ) \geq c$ and $0 \leq \hat { \gamma } ( x ) \leq 1$ for every $x \in \mathcal { X } ;$

(ii) the function $H _ { n } ( t ) : = \mathbb { E } [ \widehat { \gamma } ( X ) \mathbb { 1 } \left\{ \widehat { s } ( X ) \leq t \right\} | \mathcal { T } _ { n } ]$ satisfies $H _ { n } ( t _ { n } ^ { \circ } - u ) \leq \alpha - c u$ and $H _ { n } ( t _ { n } ^ { \circ } + u ) \geq \alpha + c u$ for all $0 < u \leq \eta$ , where $t _ { n } ^ { \circ } : = \operatorname* { s u p } \{ t \in \mathbb { R } : H _ { n } ( t ) \leq \alpha \}$ ;

(iii) for every $a < b , \mathbb { P } \{ a < \hat { s } ( X ) \leq b \mid \mathcal { T } _ { n } \} \leq C ( b - a )$

Then

$$
\begin{array} { r } { \left[ \mathbb { P } \left\{ Y _ { n + 1 } \big ( \hat { \pi } _ { o b s } ( X _ { n + 1 } ) \big ) < Y _ { n + 1 } ( 0 ) \big | \mathcal { A } _ { n } \right\} - \alpha \right] _ { + } = O _ { P } ( n ^ { - 1 / 2 } ) . } \end{array}
$$

## A.4 Technical conditions for the balancing algorithm

In this section, we provide the technical conditions and proofs that our learn-then-balance weights satisfy the needed conditions in the preceding parts, which together lead to the doubly robust safety guarantees in the main text. We first define some preparatory notations. Let $\mathcal { T } _ { n }$ denote the σ-field generated by the training process, so that $\hat { e } , \hat { g } , \hat { \gamma } , \hat { s } .$ and ˜w are fixed conditional on $\mathcal { T } _ { n }$ . Let G be a generic inclusion indicator satisfying $G \mid X , T , \mathcal { T } _ { n } \sim$ Bernoulli $\{ \hat { g } ( X , T ) \}$ . Define $q ( x ) : = e ( x ) \hat { g } ( x , 1 ) + \{ 1 - e ( x ) \} \hat { g } ( x , 0 ) , p _ { G } : = \mathbb { E } [ q ( X ) \mid \mathcal { T } _ { n } ]$ , and $m _ { n } : = | \mathbb { Z } _ { \mathrm { c a l i b } } |$ . For any $\mathcal { T } _ { n }$ -measurable functions $r \colon \mathcal { X }  [ 0 , 1 ]$ and $u \colon \mathcal { X } \to \mathbb { R } ^ { + }$ , define

$$
h _ { t } ^ { r } ( \boldsymbol { x } ) : = r ( \boldsymbol { x } ) \mathbb { 1 } \{ \hat { s } ( \boldsymbol { x } ) \leq t \} , \qquad \psi _ { t } ^ { r , u } ( \boldsymbol { x } ) : = ( \boldsymbol { 1 } , h _ { t } ^ { r } ( \boldsymbol { x } ) , u ( \boldsymbol { x } ) ) ^ { \top } ,
$$

and

$$
t ^ { \circ } ( r ) : = \operatorname* { s u p } \left\{ t \in \mathbb { R } : \mathbb { E } [ h _ { t } ^ { r } ( X ) \mid \mathcal { T } _ { n } ] \leq \alpha \right\} .
$$

Also, let

$$
M _ { r , u } ( t ) : = \mathbb { E } [ \psi _ { t } ^ { r , u } ( X ) \psi _ { t } ^ { r , u } ( X ) ^ { \top } \mid G = 1 , { \mathcal T } _ { n } ] , \qquad b _ { r , u } ( t ) : = \mathbb { E } [ \psi _ { t } ^ { r , u } ( X ) \mid { \mathcal T } _ { n } ] .
$$

Whenever $M _ { r , u } ( t )$ is invertible, define

$$
\omega _ { t } ^ { r , u } ( x ) : = \psi _ { t } ^ { r , u } ( x ) ^ { \top } M _ { r , u } ( t ) ^ { - 1 } b _ { r , u } ( t ) , \qquad \mathcal { W } _ { n } ( r , u ) : = \omega _ { t ^ { \circ } ( r ) } ^ { r , u } .
$$

For the fitted functions, abbreviate $t ^ { \circ } : = t ^ { \circ } ( \hat { \gamma } ) , M ( t ) : = M _ { \hat { \gamma } , \tilde { w } } ( t ) , b ( t ) : = b _ { \hat { \gamma } , \tilde { w } } ( t )$ , and $\omega _ { t } : = \omega _ { t } ^ { \hat { \gamma } , \tilde { w } }$ . Also, write $\psi _ { t } : = \psi _ { t } ^ { \hat { \gamma } , \tilde { w } }$ and define

$$
\hat { t } : = \operatorname* { s u p } \Big \{ t \in \mathbb { R } : \frac { 1 } { n } \sum _ { i = 1 } ^ { n } h _ { t } ^ { \hat { \gamma } } ( X _ { i } ) \leq \alpha \Big \} , \qquad \hat { M } _ { n } ( t ) : = \frac { 1 } { m _ { n } } \sum _ { i \in \mathcal { I } _ { \mathrm { c a l i b } } } \psi _ { t } ( X _ { i } ) \psi _ { t } ( X _ { i } ) ^ { \top } ,
$$

and $\begin{array} { r } { \bar { \psi } _ { n } ( t ) : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } { \psi _ { t } ( X _ { i } ) } } \end{array}$ . Let $B _ { n } ( t ) : = \left\{ b \in \mathbb { R } ^ { 3 } : b _ { 1 } = 1 , \ | b _ { k } - \bar { \psi } _ { n , k } ( t ) | \leq \delta _ { n } , \ k \in \{ 2 , 3 \} \right\}$ . Whenever $\hat { M } _ { n } ( t )$ is invertible, define, for any $x \in \mathcal { X }$ and $t \in \mathbb { R }$

$$
\hat { b } _ { n } ( t ) \in \operatorname * { a r g m i n } _ { b \in B _ { n } ( t ) } b ^ { \top } \hat { M } _ { n } ( t ) ^ { - 1 } b , \qquad \hat { a } _ { n } ( t ) : = \hat { M } _ { n } ( t ) ^ { - 1 } \hat { b } _ { n } ( t ) , \qquad \hat { \omega } _ { t } ( x ) : = \psi _ { t } ( x ) ^ { \top } \hat { a } _ { n } ( t ) .
$$

Assumption A.10 (Regularity of the balancing program). Recall that $t ^ { \circ } = t ^ { \circ } ( \hat { \gamma } ) , M ( t ) = M _ { \hat { \gamma } , \tilde { w } } ( t )$ , and $\omega _ { t } = \omega _ { t } ^ { \hat { \gamma } , \tilde { w } }$ . There exist deterministic constants $c _ { 0 } , c _ { 1 } , c _ { 2 } , C , \eta > 0$ such that, with probability tending to one over the training process, the following conditions hold.

(i) For every $x \in \mathcal { X } , q ( x ) \geq c _ { 0 } , 0 \leq \hat { \gamma } ( x ) \leq 1$ , and $0 < \tilde { w } ( x ) \le C$

(ii) Let $\mathcal { N } : = \{ t \in \mathbb { R } : | t - t ^ { \circ } | \leq \eta \}$ . The population balancing problem is uniformly nondegenerate and its solution is uniformly interior on N: in $\mathsf { f } _ { t \in \mathcal { N } } \lambda _ { \operatorname* { m i n } } \{ M ( t ) \} \geq c _ { 1 }$ , and inf $\dot { \tau } _ { t \in \mathcal { N } , x \in \mathcal { X } } \omega _ { t } ( x ) \geq c _ { 1 }$

(iii) The population cutof is locally regular: for every $0 < u \leq \eta _ { ; }$ , it holds that ${ \mathbb E } [ h _ { t ^ { \circ } - u } ( X ) \mid { \mathcal T } _ { n } ] \leq \alpha - c _ { 2 } u _ { \mathrm { : } }$ and ${ \mathbb E } \big [ h _ { t ^ { \circ } + u } ( X ) \mid \mathcal { T } _ { n } \big ] \geq \alpha + c _ { 2 } u$

(iv) The conditional distribution $o f \hat { s } ( X )$ has a uniformly bounded density in the sense that, for every $a < b _ { \scriptscriptstyle  { 1 } }$ $\mathbb { P } \{ a < \hat { s } ( X ) \leq b | \mathcal { T } _ { n } \} \leq C ( b - a )$

Under the above regularity conditions, the following lemma shows that the learned weights $\hat { w } _ { i }$ are close to the population balancing rule $W _ { n } ( \hat { \gamma } , \tilde { w } ) ( X _ { i } )$ . Moreover, if the preliminary weight function consistently estimates the oracle weight w, then the population balancing rule consistently estimates the normalized oracle weight $w ^ { \circ } { \mathrm { : } }$ ; under an $O _ { P } ( n ^ { - 1 / 4 } )$ uniform rate, it inherits the same rate. The proof is in Appendix C.3.

Lemma A.11 (Approximation properties of the balancing weights). Suppose $\delta _ { n } = O ( n ^ { - 1 / 2 } )$ and Assumption A.10 holds. Then, with probability tending to one, the balancing program has the unique solution $\hat { w } _ { i } = \hat { \omega } _ { \hat { t } } ( X _ { i } ) \mathrm { ~ } f o r \mathrm { ~ } i \in \mathcal { T } _ { c a l i b }$ . Define $\hat { w } _ { n + 1 } : = \hat { \omega } _ { \hat { t } } ( X _ { n + 1 } )$ . Then

$$
| \hat { l } - l ^ { \circ } | = O _ { P } ( n ^ { - 1 / 2 } ) , \quad \frac { 1 } { m _ { n } } \sum _ { i \in \mathbb { Z } _ { \mathrm { o i s } } } \left\{ \hat { w } _ { i } - \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } ) ( X _ { i } ) \right\} ^ { 2 } = O _ { P } ( n ^ { - 1 / 2 } ) , \quad \frac { \hat { w } _ { n + 1 } \vee \operatorname* { m a x } _ { i \in \mathbb { Z } _ { \mathrm { o i s } } } \hat { w } _ { i } } { \sum _ { i \in \mathbb { Z } _ { \mathrm { o i s } } } \hat { w } _ { i } } = O _ { P } ( n ^ { - 1 } ) .
$$

Moreover, $i f \parallel \tilde { w } - w \parallel _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 )$ , then

$$
\| \mathcal { W } _ { n } ( \widehat { \gamma } , \widetilde { w } ) - w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) ,
$$

where $w ^ { \circ } ( x ) = p _ { G } w ( x )$ . Finally, $i f \| \tilde { w } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } )$ , then

$$
\frac { 1 } { m _ { n } } \sum _ { i \in \mathcal { I } _ { c o l i b } } \left\{ \hat { w } _ { i } - \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } ) ( X _ { i } ) \right\} ^ { 2 } = O _ { P } ( n ^ { - 1 } ) , \qquad \left. \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } ) - w ^ { \circ } \right. _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } ) .
$$

## B Technical proofs

## B.1 Proof of Theorem 3.1

Proof of Theorem 3.1. We aim to show that

$$
\mathbb { P } ( p _ { n + 1 } \leq \alpha , Y _ { n + 1 } ^ { * } = 0 | \{ G _ { i } \} _ { i = 1 } ^ { n } ) \leq \alpha .\tag{B.1}
$$

Note that by definition, $Y _ { i } ^ { \dagger } = T _ { i } Y _ { i } + ( 1 - T _ { i } ) ( 1 - Y _ { i } ) \leq \operatorname* { m a x } \{ Y _ { i } ( 1 ) , 1 - Y _ { i } ( 0 ) \} = Y _ { i } ^ { * }$ . Thus, by the monotonicity of $V ( x , y )$ in y, on the event $\{ Y _ { n + 1 } ^ { * } = 0 \}$ , it holds deterministically that

$$
p _ { n + 1 } \geq p _ { n + 1 } ^ { * } : = \frac { 1 + \sum _ { i = 1 } ^ { n } G _ { i } \cdot \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ^ { * } ) \leq V ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) \} } { 1 + \sum _ { i = 1 } ^ { n } G _ { i } } .
$$

We thus have

$$
\begin{array} { r } { \mathbb { P } ( p _ { n + 1 } \leq \alpha , Y _ { n + 1 } ^ { * } = 0 | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } ) \leq \mathbb { P } ( p _ { n + 1 } ^ { * } \leq \alpha , Y _ { n + 1 } ^ { * } = 0 | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } ) \leq \mathbb { P } ( p _ { n + 1 } ^ { * } \leq \alpha | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } ) . } \end{array}\tag{B.2}
$$

Now, conditional on $\{ G _ { i } \} _ { i = 1 } ^ { n }$ , we denote ${ \mathcal { T } } : = \{ i \in [ n ] : G _ { i } = 1 \}$ as the index set of the selected calibration data. Let $P _ { X , Y } ^ { \mathrm { s u p } }$ be the (unknown) joint distribution of $( X , Y ^ { * } )$ induced by the unknown super-population P<sup>sup</sup> of $( X _ { i } , \dot { Y _ { i } } ( 1 ) , Y _ { i } ( 0 ) )$ . We then have $( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) \sim P _ { X , Y ^ { * } } ^ { \mathrm { s u p } }$ . For any measurable subset A of $\mathcal { X } \times \{ 0 , 1 \}$ , by the Bayes’ rule, we have

$$
\frac { \mathbb { P } ( ( X _ { i } , Y _ { i } ^ { * } ) \in A | G _ { i } = 1 ) } { \mathbb { P } ( ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) \in A ) } = \frac { \mathbb { P } ( G _ { i } = 1 | ( X _ { i } , Y _ { i } ^ { * } ) \in A ) } { \mathbb { P } ( G _ { i } = 1 ) } .
$$

Furthermore, we know

$$
\begin{array} { r l } & { \mathbb { P } ( G _ { i } = 1 | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ) = \mathbb { P } ( G _ { i } = 1 , T _ { i } = 1 | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ) + \mathbb { P } ( G _ { i } = 1 , T _ { i } = 0 | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ) ) } \\ & { \qquad = \mathbb { P } ( G _ { i } = 1 | T _ { i } = 1 , ( X _ { i } , Y _ { i } ^ { * } ) \in A ) \cdot \mathbb { P } ( T _ { i } = 1 | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ) } \\ & { \qquad + \mathbb { P } ( G _ { i } = 1 | T _ { i } = 0 , ( X _ { i } , Y _ { i } ^ { * } ) \in A ) \cdot \mathbb { P } ( T _ { i } = 0 | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ) } \\ & { \qquad = 1 / 2 \cdot \mathbb { E } [ \hat { g } ( X _ { i } , 1 ) | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ] + 1 / 2 \cdot \mathbb { E } [ \hat { g } ( X _ { i } , 0 ) | \left( X _ { i } , Y _ { i } ^ { * } \right) \in A ] . } \end{array}
$$

Since $\hat { g } ( x , 1 ) + \hat { g } ( x , 0 ) \equiv a$ for a constant $a > 0$ , we have

$$
\mathbb { P } ( G _ { i } = 1 | ( X _ { i } , Y _ { i } ^ { * } ) \in A ) = a / 2 ,
$$

and by the tower property,

$$
\mathbb P ( G _ { i } = 1 ) = \mathbb E \big [ \mathbb P ( G _ { i } = 1 | X _ { i } ) \big ] = \mathbb E \big [ \mathbb P ( G _ { i } = 1 , T _ { i } = 1 | X _ { i } ) + \mathbb P ( G _ { i } = 1 , T _ { i } = 0 | X _ { i } ) \big ] = a / 2 .
$$

Putting things together, we know that

$$
\begin{array} { r } { ( X _ { i } , Y _ { i } ^ { * } ) \mid G _ { i } = 1 \stackrel { d } { = } ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) . } \end{array}
$$

Thus, conditional on $\{ G _ { i } \} _ { i = 1 } ^ { n }$ , the samples $\{ ( X _ { i } , Y _ { i } ) \} _ { i \in { \mathcal { T } } _ { 0 } \cup \{ n + 1 \} }$ are exchangeable. Noting that

$$
p _ { n + 1 } ^ { * } = \frac { 1 + \sum _ { i \in \mathbb { Z } _ { 0 } } { \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ^ { * } ) \leq V ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) \} } } { 1 + \left| \mathbb { Z } _ { 0 } \right| } ,
$$

we know that $\mathbb { P } ( p _ { n + 1 } ^ { * } \le \alpha | \{ G _ { i } \} _ { i = 1 } ^ { n } ) \le \alpha$ in (B.2); see, e.g., Jin and Cand\`es (2023b) or Vovk et al. (2005). Marginalizing over $\{ G _ { i } \} _ { i = 1 } ^ { n }$ , we complete the proof of (B.1). Finally, taking $T _ { n + 1 } = \mathbb { 1 } \{ p _ { n + 1 } \leq \alpha \}$ , we have

$$
\begin{array} { r } { \mathbb { P } ( Y _ { n + 1 } ( T _ { n + 1 } ) < Y _ { n + 1 } ( 0 ) ) = \mathbb { P } ( T _ { n + 1 } = 1 , \ Y _ { n + 1 } ^ { * } = 0 ) = \mathbb { P } ( p _ { n + 1 } \leq \alpha , \ Y _ { n + 1 } ^ { * } = 0 ) \leq \alpha , } \end{array}
$$

thereby concluding the proof of Theorem 3.1.

## B.2 Proof of Theorem 3.4

Proof of Theorem 3.4. Write $\tau ( x ) : = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x ) , \hat { \tau } ( x ) : = \hat { \mu } _ { 1 } ( x ) - \hat { \mu } _ { 0 } ( x )$ , and recall that $\gamma ( x ) = \mathrm { { m i n } \{ 1 - \ } $ $\mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \}$ and $\hat { \gamma } ( x ) = \operatorname* { m i n } \{ 1 - \hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ . We simplify the notation to

$$
s ^ { * } ( x ) : = - \frac { \tau ( x ) } { \gamma ( x ) } = s _ { \mathrm { w e l f a r e } } ( x ) , \qquad \hat { s } ( x ) : = - \frac { \hat { \tau } ( x ) } { \hat { \gamma } ( x ) } .
$$

Let $\mathcal { T } _ { n }$ denote the sigma-field generated by the independent training process for $\hat { \mu } _ { 0 }$ and $\hat { \mu } _ { 1 }$ . Conditional on $\mathcal { T } _ { n } .$ , the estimated functions $\hat { \mu } _ { t } , \hat { \gamma } , \hat { s } .$ , and the inclusion rule in (3.10) are fixed. We first show that selective calibration consistently recovers the sharp worst-case harm-rate function. Define

$$
q _ { 1 } ( x ) : = 1 - \mu _ { 1 } ( x ) , q _ { 0 } ( x ) : = \mu _ { 0 } ( x ) ,
$$

and analogously $\hat { q } _ { 1 } ( x ) : = 1 - \hat { \mu } _ { 1 } ( x )$ and $\hat { q } _ { 0 } ( x ) : = \hat { \mu } _ { 0 } ( x )$ . Let $\hat { a } ( x ) \in \{ 0 , 1 \}$ denote the arm selected by (3.10), so that $\hat { a } ( x ) = 1$ when $\hat { q } _ { 1 } ( x ) \leq \hat { q } _ { 0 } ( x )$ and $\hat { a } ( x ) = 0$ otherwise. Define the true proxy-label risk of the selected arm by

$$
\overline { { { \gamma } } } _ { n } ( x ) : = q _ { 1 } ( x ) \mathbb { 1 } \{ \widehat { a } ( x ) = 1 \} + q _ { 0 } ( x ) \mathbb { 1 } \{ \widehat { a } ( x ) = 0 \} .
$$

Since $\gamma ( x ) = \mathrm { m i n } \{ q _ { 0 } ( x ) , q _ { 1 } ( x ) \}$ , we have $\overline { { \gamma } } _ { n } ( x ) \geq \gamma ( x )$ . Moreover, if $a ^ { * } ( x ) \in \arg \operatorname* { m i n } _ { a \in \{ 0 , 1 \} } q _ { a } ( x )$ , the optimality of ${ \hat { a } } ( x )$ for the estimated risks gives

$$
\begin{array} { r l } & { 0 \leq \overline { { \gamma } } _ { n } ( x ) - \gamma ( x ) = q _ { \hat { a } ( x ) } ( x ) - q _ { a ^ { * } ( x ) } ( x ) } \\ & { \qquad \leq \left| q _ { \hat { a } ( x ) } ( x ) - \hat { q } _ { \hat { a } ( x ) } ( x ) \right| + \left| \hat { q } _ { a ^ { * } ( x ) } ( x ) - q _ { a ^ { * } ( x ) } ( x ) \right| } \\ & { \qquad \leq 2 \left\{ \left| \hat { \mu } _ { 1 } ( x ) - \mu _ { 1 } ( x ) \right| + \left| \hat { \mu } _ { 0 } ( x ) - \mu _ { 0 } ( x ) \right| \right\} . } \end{array}
$$

The assumed $L _ { 2 } ( P _ { X } )$ convergence and the Cauchy–Schwarz inequality therefore imply

$$
\| \overline { { \gamma } } _ { n } - \gamma \| _ { L _ { 1 } ( P _ { X } ) } = o _ { P } ( 1 ) .
$$

We next establish convergence of the estimated ranking. The outcome-model consistency implies $\lVert \hat { \boldsymbol { \tau } } - \boldsymbol { \mathbf { \check { \tau } } }$ $\tau \| _ { L _ { 2 } ( P _ { X } ) } = o _ { P } ( 1 )$ and $\| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( P _ { X } ) } = o _ { P } \bigl ( 1 \bigr )$ , where the latter follows from the Lipschitz property of the minimum function. Under the ratio conventions in $( 3 . 1 0 ) , \mathrm { i f } \ \gamma ( X ) = 0 .$ , then $s ^ { * } ( X )$ must take one of the three values $- \infty , 0 , \mathrm { o r } + \infty$ . By the no-point-mass assumption for $s ^ { * } ( X )$ , we have ${ \mathbb P } \{ \gamma ( X ) = 0 \} = 0$ . The continuous mapping theorem then gives $\hat { s } ( X ) - s ^ { * } ( X ) = o _ { P } ( 1 )$ , where $X \sim P _ { X }$ is independent of the training process. The preceding convergence also holds uniformly for threshold indicators. Indeed, for every $\varepsilon > 0$

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { t \in \mathbb { R } } \mathbb { P } ( \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \neq \mathbb { 1 } \{ s ^ { * } ( X ) \leq t \} | \mathcal { T } _ { n } ) } } \\ & { } & { \leq \mathbb { P } ( | \hat { s } ( X ) - s ^ { * } ( X ) | > \varepsilon | \mathcal { T } _ { n } ) + \operatorname* { s u p } _ { t \in \mathbb { R } } \mathbb { P } \{ | s ^ { * } ( X ) - t | \leq \varepsilon \} . } \end{array}
$$

The first term is $o _ { P } ( 1 )$ . The second term converges to zero as $\varepsilon \downarrow 0$ because $s ^ { * } ( X )$ has no point mass. It follows that

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \mathbb { P } ( \mathbb { 1 } \left\{ { \hat { s } } ( X ) \leq t \right\} \neq \mathbb { 1 } \left\{ { s } ^ { * } ( X ) \leq t \right\} | \mathcal { T } _ { n } ) = o _ { P } ( 1 ) .
$$

Now define the empirical curve

$$
F _ { n } ( t ) : = { \frac { 1 + \sum _ { i = 1 } ^ { n } G _ { i } \mathbb { 1 } \left\{ Y _ { i } ^ { \dagger } = 0 \right\} \mathbb { 1 } \left\{ { \hat { s } } ( X _ { i } ) \leq t \right\} } { 1 + \sum _ { i = 1 } ^ { n } G _ { i } } } , \qquad t \in \mathbb { R } ,
$$

so that $p _ { n + 1 } ^ { \mathrm { w e l f a r e } } = F _ { n } \{ \hat { s } ( X _ { n + 1 } ) \}$ . Conditional on $\mathcal { T } _ { n } .$ , the summands are i.i.d. and bounded, thus Lemma F.3 gives

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \bigg | F _ { n } ( t ) - \frac { \mathbb { E } \big [ G \mathbb { 1 } \{ Y ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \big | T _ { n } \big ] } { \mathbb { E } [ G \mid \mathcal { T } _ { n } ] } \bigg | = o _ { P } ( 1 ) .
$$

Under balanced randomization and the given inclusion rule, we know $\mathbb { E } [ G \mid X , { \mathcal { T } } _ { n } ] = 1 / 2$ . Furthermore, conditional on $X = x .$ , the probability that the selected proxy label is zero is $q _ { 1 } ( x )$ when $\hat { a } ( x ) = 1$ and $q _ { 0 } ( x )$ when $\hat { a } ( x ) = 0$ . Consequently,

$$
\mathbb { E } \big [ G \mathbb { 1 } \{ Y ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \big | \mathcal { T } _ { n } \big ] = \frac { 1 } { 2 } \mathbb { E } \big [ \overline { { \gamma } } _ { n } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \big | \mathcal { T } _ { n } \big ] .
$$

Therefore,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \big | F _ { n } ( t ) - \overline { H } _ { n } ( t ) \big | = o _ { \mathbb { P } } ( 1 ) , \qquad \overline { H } _ { n } ( t ) : = \mathbb { E } [ \overline { \gamma } _ { n } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} | \mathcal T _ { n } ] .
$$

Define the oracle harm-cost curve

$$
H ( t ) : = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s ^ { * } ( X ) \leq t \} ] .
$$

Using $0 \leq \gamma \leq 1$ , we have

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \overline { H } _ { n } ( t ) - H ( t ) | \leq \| \overline { \gamma } _ { n } - \gamma \| _ { L _ { 1 } ( P _ { X } ) } + \operatorname* { s u p } _ { t \in \mathbb { R } } \mathbb { E } [ \gamma ( X ) | \mathbf 1 \{ \hat { s } ( X ) \leq t \} - \mathbf I \{ s ^ { * } ( X ) \leq t \} | | T _ { n } ] = o _ { P } ( 1 ) .
$$

Combining the preceding displays gives su $) _ { t \in \mathbb { R } } | F _ { n } ( t ) - H ( t ) | = o _ { P } ( 1 )$ . The function H is continuous because its jump at any $t \in \mathbb { R }$ is $\mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s ^ { * } ( X ) = \bar { t } \} ] = 0$ . Thus, for the independent test point,

$$
\left| p _ { n + 1 } ^ { \mathrm { v e l f a r e } } - H \{ s ^ { * } ( X _ { n + 1 } ) \} \right| \le \operatorname* { s u p } _ { t \in \mathbb { R } } \left| F _ { n } ( t ) - H ( t ) \right| + \left| H \{ \hat { s } ( X _ { n + 1 } ) \} - H \{ s ^ { * } ( X _ { n + 1 } ) \} \right| = 0 _ { P } ( 1 ) .
$$

We next translate this convergence into convergence of treatment decisions. Since $\tau ( X ) = 0$ implies $s ^ { * } ( X ) =$ $0 ,$ the no-point-mass condition gives $\mathbb { P } \{ \tau ( X ) = 0 \} = 0$ . Hence, $\mathbb { 1 } \{ \hat { \tau } ( X _ { n + 1 } ) > 0 \} - \mathbb { 1 } \{ \tau ( X _ { n + 1 } ) > 0 \}  0$ in probability. We also claim that

$$
\mathbb { P } ( H \{ s ^ { * } ( X ) \} = \alpha , \tau ( X ) > 0 ) = 0 .
$$

To see this, let $J _ { \alpha } : = \{ t < 0 : H ( t ) = \alpha \}$ . Since H is nondecreasing, $J _ { \alpha }$ is an interval. If $a \textless b$ lie in the interior of this interval, then $0 = H ( b ) - H ( a ) = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ a < s ^ { * } ( X ) \leq b \} ]$ . Since $\gamma ( X ) > 0$ almost surely, it follows that $\mathbb { P } \{ a < s ^ { * } ( X ) \leq b \} = 0$ . The endpoints of $J _ { \alpha }$ also have probability zero because $s ^ { * } ( X )$ has no point mass. This proves the claim. Recall that $\hat { \pi } _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) = \mathbb { 1 } \{ p _ { n + 1 } ^ { \mathrm { w e l f a r e } } \leq \alpha \} \mathbb { 1 } \{ \hat { \tau } ( X _ { n + 1 } ) > 0 \}$ The convergence $p _ { n + 1 } ^ { \mathrm { w e l f a r e } } - H \{ s ^ { * } ( X _ { n + 1 } ) \} = o _ { \mathbb { P } } ( 1 )$ , the convergence of the treatment-efect indicator, and the preceding zero-probability claim imply $\mathbb { P } \{ \hat { \pi } _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) \neq \pi _ { 0 } ( X _ { n + 1 } ) \}  0 .$ , where

$$
\pi _ { 0 } ( x ) : = 1 \{ H ( s ^ { * } ( x ) ) \leq \alpha \} 1 \{ \tau ( x ) > 0 \} .
$$

It remains to identify $\pi _ { 0 }$ with the oracle policy in Theorem 3.3. Let $r ^ { * } \leq 0$ be the cutof in that theorem. Suppose first that the safety constraint is nonbinding. Then $r ^ { * } = 0$ and, since ${ \mathbb P } \{ \tau ( X ) = 0 \} = 0$

$$
H ( 0 ) = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ \tau ( X ) > 0 \} ] \le \alpha .
$$

For every x such that $\tau ( x ) > 0$ , we have $s ^ { * } ( x ) < 0$ , and therefore $H \{ s ^ { * } ( x ) \} \leq H ( 0 ) \leq \alpha$ . It follows that

$$
\pi _ { 0 } ( X ) = \mathbb { 1 } \{ \tau ( X ) > 0 \} = \mathbb { 1 } \{ s ^ { * } ( X ) \leq 0 \} = \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

Suppose instead that the safety constraint is binding. Then $r ^ { * } < 0$ . By the continuity of H and the definition of $r ^ { * }$ , we have $H ( r ^ { * } ) = \alpha$ . By the monotonicity of H and the preceding zero-probability result for the level set $\{ t < 0 : H ( t ) = \alpha \}$ , we know

$$
\mathbb { 1 } \left\{ H ( s ^ { * } ( X ) ) \leq \alpha \right\} = \mathbb { 1 } \left\{ s ^ { * } ( X ) \leq r ^ { * } \right\} \quad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Moreover, the indicator $\mathbb { 1 } \{ \tau ( X ) > 0 \}$ is redundant on the event $\{ s ^ { * } ( X ) \leq r ^ { * } \}$ because $r ^ { * } < 0$ . Consequently,

$$
\pi _ { 0 } ( X ) = \mathbb { 1 } \{ s ^ { * } ( X ) \leq r ^ { * } \} = \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) \leq r ^ { * } \} = \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

We have therefore shown that $\mathbb { P } \{ \hat { \pi } _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) \neq \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X _ { n + 1 } ) \}  0$ , which implies

$$
\begin{array} { r } { | \mathbb { E } [ Y \{ \hat { \pi } _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) \} ] - \mathrm { W e l f a r e } ( \pi _ { \mathrm { w e l f a r e } } ^ { * } ; P ) | \leq \mathbb { P } \{ \hat { \pi } _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) \neq \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X _ { n + 1 } ) \} \to 0 . } \end{array}
$$

This completes the proof.

## B.3 Proof of Theorem 4.1

Proof of Theorem 4.1. The p-value (4.2) can be equivalently written as

$$
p _ { n + 1 } = \frac { w ( X _ { n + 1 } ) + \sum _ { i \in \mathcal { T } _ { \mathrm { c a l i b } } } w ( X _ { i } ) \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ^ { \dagger } ) \leq V ( X _ { n + 1 } , 0 ) \} } { \sum _ { i \in \mathcal { T } _ { \mathrm { c a l i b } } } w ( X _ { i } ) + w ( X _ { n + 1 } ) } .
$$

By definition, we have $Y _ { i } ^ { \dagger } = T _ { i } Y _ { i } + ( 1 - T _ { i } ) ( 1 - Y _ { i } ) \leq \operatorname* { m a x } \{ Y _ { i } ( 1 ) , 1 - Y _ { i } ( 0 ) \} = Y _ { i } ^ { * }$ . Thus, by the monotonicity of $V ( x , y )$ in y, on the event $\{ Y _ { n + 1 } ^ { * } = 0 \}$ , it holds deterministically that

$$
p _ { n + 1 } \geq p _ { n + 1 } ^ { * } : = \frac { w ( X _ { n + 1 } ) + \sum _ { i \in { \cal Z } _ { \mathrm { c a l i b } } } w ( X _ { i } ) \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ^ { * } ) \leq V ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) \} } { \sum _ { i \in { \cal Z } _ { \mathrm { c a l i b } } } w ( X _ { i } ) + w ( X _ { n + 1 } ) } .
$$

We thus have

$$
\begin{array} { r } { \mathbb { P } ( p _ { n + 1 } \leq \alpha , Y _ { n + 1 } ^ { * } = 0 | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } ) \leq \mathbb { P } ( p _ { n + 1 } ^ { * } \leq \alpha , Y _ { n + 1 } ^ { * } = 0 | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } ) \leq \mathbb { P } ( p _ { n + 1 } ^ { * } \leq \alpha | \left\{ G _ { i } \right\} _ { i = 1 } ^ { n } ) . } \end{array}\tag{B.3}
$$

The goal is then to prove the RHS of (B.3) is upper bounded by $\alpha .$

Conditional on $\{ G _ { i } \} _ { i = 1 } ^ { n }$ , the data $( X _ { i } , Y _ { i } ^ { * } )$ for $i \in \mathcal { T } _ { \mathrm { c a l i b } }$ are i.i.d. and follow the distribution

$$
P ^ { \mathrm { c a l i b } } : \stackrel { d } { = } P _ { ( X _ { i } , Y _ { i } ^ { * } ) | G _ { i } = 1 } ,
$$

whereas the test point $( X _ { i } , Y _ { i } ^ { * } )$ follows the distribution

$$
\begin{array} { r } { P ^ { \mathrm { t e s t } } : = P _ { X _ { i } , Y _ { i } ^ { * } } , } \end{array}
$$

and we recall that $P$ denotes the true underlying super-population distribution. Let $\boldsymbol { p } ( x , y ^ { * } )$ be the density function of $P _ { X , Y * }$ with respect to some base measure. The density ratio between the two distributions is

$$
{ \frac { \operatorname { d } P ^ { \operatorname { t e s t } } } { \operatorname { d } P ^ { \operatorname { c a l i b } } } } ( x , y ) = { \frac { p ( x , y ^ { * } ) } { p ( x , y ^ { * } \mid G = 1 ) } } = { \frac { p ( x ) p ( y ^ { * } \mid x ) } { p ( x \mid G = 1 ) p ( y ^ { * } \mid x , G = 1 ) } } ,
$$

where we define the random variable $G = \hat { g } ( X , T )$ , and $p ( y ^ { * } \mid x , G = 1 )$ is the conditional density of $Y _ { i } ^ { * }$ given $X _ { i } = x$ and $G _ { i } = 1$ , etc. By unconfoundedness, $Y ^ { * }$ is independent of T conditional on X. This leads to

$$
\frac { \mathrm { d } P ^ { \mathrm { t e s t } } } { \mathrm { d } P ^ { \mathrm { c a l i b } } } ( x , y ) = \frac { p ( x ) p ( y ^ { * } \mid x ) } { p ( x \mid G = 1 ) p ( y ^ { * } \mid x ) } = \frac { p ( x ) } { p ( x \mid G = 1 ) } = \frac { { \mathbb P } ( G = 1 ) } { { \mathbb P } ( G = 1 \mid X = x ) } .
$$

Now, by the definition of the $G _ { i } { } ^ { \ ' } \mathrm { s } ,$ we know

$$
\begin{array} { r l } & { \mathbb { P } ( G = 1 | X = x ) = \mathbb { P } ( G = 1 , T = 1 | X = x ) + \mathbb { P } ( G = 1 , T = 0 | X = x ) } \\ & { \qquad = \mathbb { P } ( G = 1 | T = 1 , X = x ) \mathbb { P } ( T = 1 | X = x ) + \mathbb { P } ( G = 1 | T = 0 , X = x ) \mathbb { P } ( T = 0 | X = x ) } \\ & { \qquad = e ( x ) \tilde { g } ( x , 1 ) + ( 1 - e ( x ) ) \tilde { g } ( x , 0 ) . } \end{array}
$$

This implies the density ratio between $P ^ { \mathrm { t e s t } }$ and $P ^ { \mathrm { c a l i b } }$ is given by

$$
{ \frac { \mathrm { d } P ^ { \mathrm { t e s t } } } { \mathrm { d } P ^ { \mathrm { c a l i b } } } } ( x , y ) = \mathbb { P } ( G = 1 ) \cdot w ( x ) .
$$

In other words, $\{ ( X _ { i } , Y _ { i } ^ { * } \} _ { i \in { \mathbb { Z } } _ { \mathrm { c a l i b } } \cup \{ n + 1 \} }$ are weighted exchangeable (Tibshirani et al., 2019), and thus following Tibshirani et al. (2019), we know that

$$
\mathbb { P } \Bigg ( \frac { w ( X _ { n + 1 } ) + \sum _ { i \in \mathcal { T } _ { \mathrm { c a l i b } } } w ( X _ { i } ) \mathbb { 1 } \{ V ( X _ { i } , Y _ { i } ^ { * } ) \leq V ( X _ { n + 1 } , Y _ { n + 1 } ^ { * } ) \} } { \sum _ { i \in \mathcal { T } _ { \mathrm { c a l i b } } } w ( X _ { i } ) + w ( X _ { n + 1 } ) } \leq \alpha \Bigg | \{ G _ { i } \} _ { i = 1 } ^ { n } \Bigg ) \leq \alpha .
$$

This proves the desired upper bound on the RHS of (B.3) hence Theorem 4.1.

## B.4 Proof of Theorem 4.2

Proof of Theorem 4.2. The optimality follows from the asymptotic convergence of the power/welfare due to the law of large numbers, as well as the optimal results in Theorems A.1 and 3.3.

Write $\tau ( x ) : = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ and $\hat { \tau } ( x ) : = \hat { \mu } _ { 1 } ( x ) - \hat { \mu } _ { 0 } ( x )$ , and recall that $\gamma ( x ) = \operatorname* { m i n } \{ 1 - \mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \}$ and $\hat { \gamma } ( x ) = \operatorname* { m i n } \{ 1 - \hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ . Define

$$
s ^ { * } ( x ) : = - \frac { \tau ( x ) } { \gamma ( x ) } = s _ { \mathrm { w e l f a r e } } ( x ) , \qquad \hat { s } ( x ) : = - \frac { \hat { \tau } ( x ) } { \hat { \gamma } ( x ) } .
$$

Let $\mathcal { T } _ { n }$ denote the sigma-field generated by the independent training process. Conditional on $\mathcal { T } _ { n } .$ , all estimated functions are fixed. As in the proof of Theorem 3.4, let $q _ { 1 } ( x ) : = 1 - \mu _ { 1 } ( x )$ and $q _ { 0 } ( x ) : = \mu _ { 0 } ( x )$ , and let $\hat { a } ( x ) \in \{ 0 , 1 \}$ denote the arm selected by (3.12). Thus, $\hat { a } ( x ) = 1$ when $1 - \hat { \mu } _ { 1 } ( x ) \le \hat { \mu } _ { 0 } ( x )$ and $\hat { a } ( x ) = 0$ otherwise. Define

$$
\gamma _ { n } ( x ) : = q _ { 1 } ( x ) \mathbb { 1 } \{ { \hat { a } } ( x ) = 1 \} + q _ { 0 } ( x ) \mathbb { 1 } \{ { \hat { a } } ( x ) = 0 \} .
$$

The optimality of ${ \hat { a } } ( x )$ for the estimated risks gives

$$
0 \leq \gamma _ { n } ( x ) - \gamma ( x ) \leq 2 \left\{ \left| \hat { \mu } _ { 1 } ( x ) - \mu _ { 1 } ( x ) \right| + \left| \hat { \mu } _ { 0 } ( x ) - \mu _ { 0 } ( x ) \right| \right\} ,
$$

and hence $\| \gamma _ { n } - \gamma \| _ { L _ { 1 } ( P _ { X } ) } = o _ { \mathbb { P } } ( 1 )$ . The same argument as in the proof of Theorem 3.4 also gives ${ \hat { s } } ( X ) -$ $s ^ { * } ( X ) = o _ { \mathbb { P } } ( 1 )$ and

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \mathbb { P } ( \mathbb { 1 } \left\{ { \hat { s } } ( X ) \leq t \right\} \neq \mathbb { 1 } \left\{ { s } ^ { * } ( X ) \leq t \right\} | \mathcal { T } _ { n } ) = o _ { \mathbb { P } } ( 1 ) .
$$

We now examine the efect of weighting. Define the conditional inclusion probability

$$
\rho _ { n } ( x ) : = e ( x ) \mathbb { 1 } \{ { \hat { a } } ( x ) = 1 \} + \{ 1 - e ( x ) \} \mathbb { 1 } \{ { \hat { a } } ( x ) = 0 \} ,
$$

so that the weight in (4.1) can be written as $w _ { n } ( x ) = \rho _ { n } ( x ) ^ { - 1 }$ . For covariates X, the inclusion indicator is $G = \mathbb { 1 } \{ T = \hat { a } ( X ) \}$ . Conditional on $( X , \mathcal { T } _ { n } )$ , note the two identities

$$
\mathbb { E } [ G w _ { n } ( X ) \mid X , { \mathcal { T } } _ { n } ] = 1 , \qquad \mathbb { E } \big [ G w _ { n } ( X ) \mathbb { 1 } \{ Y ^ { \dagger } = 0 \} \mid X , { \mathcal { T } } _ { n } \big ] = \gamma _ { n } ( X ) .
$$

Indeed, i $\textrm { f } \hat { a } ( X ) = 1$ , then $G = T , w _ { n } ( X ) = 1 / e ( X )$ , and the second conditional expectation is $1 - \mu _ { 1 } ( X )$ . If $\hat { a } ( X ) = 0$ , then $G = 1 - T , w _ { n } ( X ) = 1 / \{ 1 - e ( X ) \}$ , and the second conditional expectation is $\mu _ { 0 } ( X )$ . Define

$$
F _ { n } ( t ) : = \frac { w _ { n } ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } G _ { i } w _ { n } ( X _ { i } ) \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq t \} } { w _ { n } ( X _ { n + 1 } ) + \sum _ { i = 1 } ^ { n } G _ { i } w _ { n } ( X _ { i } ) } , \qquad t \in \mathbb { R } ,
$$

so that $p _ { n + 1 } ^ { \mathrm { s t r - o p t } } = F _ { n } \{ \hat { s } ( X _ { n + 1 } ) \}$ . The preceding identities imply that the training-conditional population counterpart of $F _ { n }$ is

$$
\overline { { H } } _ { n } ( t ) : = \mathbb { E } [ \gamma _ { n } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} | { \mathcal { T } } _ { n } ] .
$$

We now briefly verify the uniform law of large numbers. Let $Z = G w _ { n } ( X )$ . Conditional on $\mathcal { T } _ { n } , \mathbb { E } [ Z \mid \mathcal { T } _ { n } ] = 1$ Moreover, for every $M > 0$ , we know

$$
\begin{array} { r } { \mathbb { E } [ ( Z - M ) _ { + } \mid \mathcal { T } _ { n } ] = \mathbb { E } [ ( 1 - M \rho _ { n } ( X ) ) _ { + } \mid \mathcal { T } _ { n } ] \leq \mathbb { E } [ ( 1 - M e ( X ) ) _ { + } ] + \mathbb { E } [ ( 1 - M \{ 1 - e ( X ) \} ) _ { + } ] , } \end{array}
$$

which converges to zero as $M  \infty$ because $e ( X ) \ \in \ ( 0 , 1 )$ almost surely. Applying Lemma F.3 to the bounded truncations and then letting $M  \infty$ therefore gives a uniform law of large numbers for the numerator of $F _ { n } { \mathrm { : } }$ ; the denominator follows by the same argument. In addition, $w _ { n } ( X _ { n + 1 } ) / n = o _ { \mathbb { P } } ( 1 )$ because $w _ { n } ( X ) \leq \operatorname* { m a x } \{ e ( X ) ^ { - 1 } , \{ 1 - e ( X ) \} ^ { - 1 } \} < \infty$ almost surely. Consequently,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | F _ { n } ( t ) - \overline { { H } } _ { n } ( t ) | = o _ { \mathbb { P } } ( 1 ) .
$$

Define the oracle harm-cost curve $H ( t ) : = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s ^ { * } ( X ) \leq t \} ]$ . Using $0 \leq \gamma \leq 1$ , we have

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left| \overline { H } _ { n } ( t ) - H ( t ) \right| \leq \| \gamma _ { n } - \gamma \| _ { L _ { 1 } ( P _ { X } ) } + \operatorname* { s u p } _ { t \in \mathbb { R } } \mathbb { E } [ \gamma ( X ) \left| \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} - \mathbb { 1 } \{ s ^ { * } ( X ) \leq t \} \right| | T _ { n } ] = o _ { \mathbb { P } } ( 1 ) .
$$

It follows that

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | F _ { n } ( t ) - H ( t ) | = o _ { \mathbb { P } } ( 1 ) .
$$

The function H is continuous because its jump at any $t \in \mathbb { R }$ is $\mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s ^ { * } ( X ) = t \} ] = 0$ . Hence, for the independent test point, $p _ { n + 1 } ^ { \mathrm { s t r - o p t } } - H \{ s ^ { * } ( X _ { n + 1 } ) \} = o _ { \mathbb { P } } ( 1 )$ . Finally, the outcome-model consistency and ${ \mathbb P } \{ \tau ( X ) = 0 \} = 0$ imply $\lfloor \left\{ \hat { \tau } ( X _ { n + 1 } ) > 0 \right\} - \mathbb { 1 } \{ \tau ( X _ { n + 1 } ) > 0 \} = o _ { \mathbb { P } } ( 1 )$ . As shown in the proof of Theorem 3.4, the no-point-mass condition also implies $\mathbb { P } ( H \{ s ^ { * } ( X ) \} = \alpha , \tau ( X ) > 0 ) = 0$ . Therefore, we have $\mathbb { P } \{ \hat { \pi } _ { \mathrm { s t r - o p t } } ( X _ { n + 1 } ) \neq \pi _ { 0 } ( X _ { n + 1 } ) \} \  \ 0$ , where $\pi _ { 0 } ( x ) : = \mathbb { 1 } \{ H ( s ^ { * } ( x ) ) \leq \alpha \} \mathbb { 1 } \{ \tau ( x ) > 0 \}$ . The binding and nonbinding arguments in the proof of Theorem 3.4 show that $\pi _ { 0 } ( X ) = \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X )$ almost surely. Thus,

$$
\mathbb { P } \{ \hat { \pi } _ { \mathrm { s t r - o p t } } ( X _ { n + 1 } ) \neq \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X _ { n + 1 } ) \}  0 .
$$

With similar arguments as the end of the proof of Theorem 3.4, we complete the proof.

## B.5 Proof of Theorem 5.1

Proof of Theorem 5.1. By Lemma A.11, the learn-then-balance weights satisfy Assumption A.6 with $r _ { n } =$ $n ^ { - 1 / 4 }$ and reference function $\hat { \omega } ( \cdot ) = \mathscr { W } _ { n } ( \hat { \gamma } , \tilde { w } ) ( \cdot )$

Under condition (i), Lemma A.11 gives $\| \hat { \omega } - w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 )$ , where $w ^ { \circ } = p _ { G } w \propto w$ . Hence condition (i) of Theorem A.7 holds.

Under condition (ii), Assumption A.10 verifies the overlap, interiority, local-slope, and bounded-density conditions in condition (ii) of Theorem A.7. The conclusion therefore follows from Theorem A.7 in either case. □

## B.6 Proof of Theorem 5.3

Proof of Theorem 5.3. Set $\hat { \omega } : = \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } )$ . By Lemma A.11, the learn-then-balance weights satisfy Assumption A.6 with $r _ { n } = n ^ { - 1 / 2 }$ and $\| \hat { \omega } - w ^ { \circ } \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } \big ( n ^ { - 1 / 4 } \big )$ , where $w ^ { \circ } ( x ) : = p _ { G } w ( x )$ . Together with the assumed rate for ˆγ, this verifies Assumption A.8. Assumption A.10 verifies the remaining regularity conditions in Theorem A.9, which gives the result. □

## B.7 Proof of Theorem 5.4

Proof of Theorem 5.4. Let

$$
H _ { n } ( t ) : = \mathbb { E } [ \widehat { \gamma } ( X ) \mathbb { 1 } \{ \widehat { s } ( X ) \leq t \} | \mathcal { T } _ { n } ] , \qquad t _ { n } ^ { \circ } : = \operatorname* { s u p } \{ t \in \mathbb { R } : H _ { n } ( t ) \leq \alpha \} ,
$$

and define

$$
H _ { 0 } ( t ) : = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) \leq t \} ] .
$$

By the assumed convergence of $\hat { \gamma }$ and ˆs, the no-point-mass condition on $s _ { \mathrm { w e l f a r e } } ( X )$ , and Lemma F.4,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - H _ { 0 } ( t ) | = o _ { P } ( 1 ) .
$$

Together with the local-crossing condition in Assumption A.10(iii), this implies $t _ { n } ^ { \circ } - t _ { \mathrm { r a w } } ^ { \ast } = o _ { P } ( 1 )$ , where $t _ { \mathrm { r a w } } ^ { * } = \operatorname* { s u p } \{ t : H _ { 0 } ( t ) \leq \alpha \}$ . Indeed, for every fixed $0 < \varepsilon \leq \eta ,$ with probability tending to one, it holds that $H _ { 0 } ( t _ { n } ^ { \circ } - \varepsilon ) < \alpha < H _ { 0 } ( t _ { n } ^ { \circ } + \varepsilon )$ , and hence $t _ { n } ^ { \circ } - \varepsilon \leq t _ { \mathrm { r a w } } ^ { * } \leq t _ { n } ^ { \circ } + \varepsilon$

Set $\hat { \omega } : = \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } )$ and, writing $I _ { t } ( x ) : = \mathbb { 1 } \{ \hat { s } ( x ) \leq t \}$ , define

$$
F _ { n } ^ { \widehat { \gamma } } ( t ) : = \frac { \mathbb { E } [ \widehat { \omega } ( X ) \widehat { g } ( X , T ) \widehat { \gamma } ( X ) I _ { t } ( X ) \mid \mathcal { T } _ { n } ] } { \mathbb { E } [ \widehat { \omega } ( X ) \widehat { g } ( X , T ) \mid \mathcal { T } _ { n } ] } .
$$

By the defining balance equations for $\hat { \omega }$ , we know $F _ { n } ^ { \hat { \gamma } } ( t _ { n } ^ { \circ } ) = H _ { n } ( t _ { n } ^ { \circ } ) = \alpha$ . Moreover, Assumption $\mathrm { A . 1 0 ( i ) { - } ( i i i ) }$ implies that, for some deterministic $\kappa > 0$ , with probability tending to one, $F _ { n } ^ { \hat { \gamma } } ( t _ { n } ^ { \circ } - u ) \leq \alpha - \kappa u$ and $F _ { n } ^ { \hat { \gamma } } ( t _ { n } ^ { \circ } + u ) \geq \alpha + \kappa u$ hold for every $0 < u \leq \eta$ . Define

$$
\hat { F } _ { n } ^ { \mathrm { o b s } } ( t ) : = \frac { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} I _ { t } ( X _ { i } ) } { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } } ,
$$

so that $p _ { n + 1 } ^ { \mathrm { o b s } } = \hat { F } _ { n } ^ { \mathrm { o b s } } \{ \hat { s } ( X _ { n + 1 } ) \}$ . Similar to the arguments in the proof of Theorem A.7, Lemma A.11, Lemmas C.1 and F.4, and $\textstyle \hat { w } _ { n + 1 } \big / \sum _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } } \hat { w } _ { i } = O _ { P } \big ( n ^ { - 1 } \big )$ give

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \big | \hat { F } _ { n } ^ { \mathrm { o b s } } ( t ) - F _ { n } ^ { \dagger } ( t ) \big | = o _ { P } ( 1 ) , \quad F _ { n } ^ { \dagger } ( t ) : = \frac { \mathbb { E } [ \hat { \omega } ( X ) \hat { g } ( X , T ) m ( X , T ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} | T _ { n } ] } { \mathbb { E } [ \hat { \omega } ( X ) \hat { g } ( X , T ) | T _ { n } ] } ,
$$

where we define $m ( x , t ) = \mathbb { P } ( Y ^ { \dagger } = 0 | X = x , T = t ) = t ( 1 - \mu _ { 1 } ( x ) ) + ( 1 - t ) \mu _ { 0 } ( x )$ . Furthermore, by the definition of $g ^ { * }$ , we know $\mathbb { E } [ g ^ { * } ( X , T ) \mathbb { 1 } \{ Y ^ { \dagger } = 0 \} | X ] = \mathbb { E } [ g ^ { * } ( X , T ) m ( X , T ) | X ] = \mathbb { E } [ g ^ { * } ( X , T ) \gamma ( X ) | X ]$ . Thus,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { F } _ { n } ^ { \mathrm { o b s } } ( t ) - F _ { n } ^ { \hat { \gamma } } ( t ) | \leq o _ { P } ( 1 ) + C \| \hat { g } - g ^ { * } \| _ { L _ { 1 } ( \mathbb { P } _ { X , T } ) } + C \| \hat { \gamma } - \gamma \| _ { L _ { 1 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) .
$$

Let

$$
\begin{array} { r } { \hat { \pi } _ { \mathrm { o b s , r a w } } ( X _ { n + 1 } ) : = \mathbb { 1 } \{ p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha \} . } \end{array}
$$

Fix $0 < \varepsilon \le \eta$ . The preceding results imply that, with probability tending to one,

$$
\begin{array} { r l r } { \hat { s } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } - \varepsilon } & { { } \Rightarrow } & { p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha , } \\ { \hat { s } ( X _ { n + 1 } ) \geq t _ { n } ^ { \circ } + \varepsilon } & { { } \Rightarrow } & { p _ { n + 1 } ^ { \mathrm { o b s } } > \alpha . } \end{array}
$$

Therefore, we have

$$
\begin{array} { r } { | \hat { \pi } _ { \mathrm { o b s , r a w } } ( X _ { n + 1 } ) - \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } \} | \leq \mathbb { 1 } \{ | \hat { s } ( X _ { n + 1 } ) - t _ { n } ^ { \circ } | \leq \varepsilon \} + o _ { \mathbb { P } } ( 1 ) . } \end{array}
$$

Assumption $\mathrm { A . 1 0 ( i v ) }$ and the arbitrariness of $\varepsilon > 0$ then imply $\mathbb { E } \big [ | \hat { \pi } _ { \mathrm { o b s , r a w } } ( X _ { n + 1 } ) - \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } \} | \big ]  0 .$ Since multiplication by an indicator cannot increase the absolute diference, the preceding display gives

$$
\begin{array} { r } { \mathbb { E } [ | \hat { \pi } _ { \mathrm { o b s - w e l f a r e } } ( X _ { n + 1 } ) - \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } \} \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) < 0 \} | ]  0 . } \end{array}
$$

We next pass to the population limit. The assumptions $\begin{array} { r } { \big \| \hat { s } - s _ { \mathrm { w e l f a r e } } \big \| _ { L _ { 2 } ( P _ { X } ) } = o _ { \mathbb { P } } ( 1 ) } \end{array}$ and $t _ { n } ^ { \circ } - t _ { \mathrm { r a w } } ^ { \ast } = o _ { \mathbb { P } } ( 1 )$ 2 together with the no-point-mass condition on $s _ { \mathrm { w e l f a r e } } ( X )$ , imply

$$
\begin{array} { r } { \mathbb { E } \Big [ \big | \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } \} \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) < 0 \} - \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) \leq t _ { \mathrm { r a w } } ^ { * } \} \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X _ { n + 1 } ) < 0 \} \big | \Big ] \to 0 . } \end{array}
$$

Indeed, for any $\varepsilon > 0$ , disagreement between the first threshold indicators is contained in the union of $\{ | \hat { s } - s _ { \mathrm { w e l f a r e } } | > \varepsilon \} , ~ \{ | t _ { n } ^ { \circ } - t _ { \mathrm { r a w } } ^ { * } | > \varepsilon \}$ , and $\left\{ \left. s _ { \mathrm { w e l f a r e } } - t _ { \mathrm { r a w } } ^ { \ast } \right. \leq 2 \varepsilon \right\}$ . The corresponding argument at the threshold zero establishes convergence of the second indicators.

It remains to identify the limiting rule with the oracle policy in Theorem 3.3. Let $r ^ { * } \leq 0$ denote the cutof therein. We consider two cases:

• If the safety constraint is binding, then $H _ { 0 } ( 0 ) ~ > ~ \alpha$ , so $t _ { \mathrm { r a w } } ^ { * } ~ < ~ 0$ and $t _ { \mathrm { r a w } } ^ { * } ~ = ~ r ^ { * }$ . In this case, $1 1 \{ s _ { \mathrm { w e l f a r e } } \ < \ 0 \}$ is redundant on $\left\{ s _ { \mathrm { w e l f a r e } } \ \leq \ t _ { \mathrm { r a w } } ^ { * } \right\}$ , and hence $\mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) \leq t _ { \mathrm { r a w } } ^ { * } \} \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) <$ $0 \} = \mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) \leq r ^ { * } \right\} = \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X )$ almost surely.

• If the safety constraint is nonbinding, then $H _ { 0 } ( 0 ) \leq \alpha , \mathrm { s o } r ^ { * } = 0$ and $t _ { \mathrm { r a w } } ^ { * } \geq 0$ . Therefore, $\mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) \leq$ $t _ { \mathrm { r a w } } ^ { * } \} \mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) < 0 \right\} = \mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) < 0 \right\}$ . Since $s _ { \mathrm { w e l f a r e } } ( X )$ has no point mass at zero, this equals $\mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) \le 0 \} = \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X )$ almost surely.

Combining the preceding results gives $\begin{array} { r } { \mathbb { E } \big [ | \hat { \pi } _ { \mathrm { o b s - w e l f a r e } } ( X _ { n + 1 } ) - \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X _ { n + 1 } ) | \big ] \to 0 } \end{array}$ . Finally, since the outcomes are binary, we know

$$
\begin{array} { r } { | \mathbb { E } [ Y \{ \hat { \pi } _ { \mathrm { o b s - w e l f a r e } } ( X _ { n + 1 } ) \} ] - \operatorname { W e l f a r e } ( \pi _ { \mathrm { w e l f a r e } } ^ { * } ; P ) | \leq \mathbb { E } \big [ | \hat { \pi } _ { \mathrm { o b s - w e l f a r e } } ( X _ { n + 1 } ) - \pi _ { \mathrm { w e l f a r e } } ^ { * } ( X _ { n + 1 } ) | \big ] \to 0 , } \end{array}
$$

which completes the proof.

## C Proof of general balancing theory

## C.1 Proof of Theorem A.7

Proof of Theorem A.7. Throughout this proof, we condition on the training process of the functions, so ˆs and ˆg are viewed as fixed, as well as the function $\hat { w } ( \cdot )$ in the first condition of Assumption A.6. Once we prove $\begin{array} { r } { \mathbb { P } ( Y _ { n + 1 } ( \pi _ { \mathrm { o b s } } ( X _ { n + 1 } ) < Y _ { n + 1 } ( 0 ) | \hat { s } , \hat { g } , \hat { w } ) \leq \alpha + o _ { P } ( 1 ) } \end{array}$ conditional on these randomness, since R is uniformly bounded we also have the marginal harm rate $\mathbb { P } ( Y _ { n + 1 } ( \pi _ { \mathrm { o b s } } ( X _ { n + 1 } ) < Y _ { n + 1 } ( 0 ) ) \leq \alpha + o _ { P } ( 1 )$ . For notational simplicity, we shall omit the conditioning in the probability/expectation throughout the proof. Define

$$
\hat { \tau } : = \operatorname* { s u p } \{ t \in \mathbb { R } \colon \hat { F } _ { n } ( t ) \leq \alpha \} , \quad \hat { F } _ { n } ( t ) : = \frac { \hat { w } _ { n + 1 } + \sum _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } } \hat { w } _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} } { \sum _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } } \hat { w } _ { i } + \hat { w } _ { n + 1 } } , \quad \hat { F } _ { n } ( t ) : = \frac { \hat { w } _ { n + 1 } } { 2 } ,
$$

so that $p _ { n + 1 } = \hat { F } _ { n } ( \hat { s } ( X _ { n + 1 } ) )$ . Thus, $p _ { n + 1 } \leq \alpha$ is equivalent to $\hat { s } ( X _ { n + 1 } ) \leq \hat { \tau }$

We now construct a random variable $Y _ { i } ^ { * * }$ whose conditional expectation is the worst-case harm rate function $\gamma ( X _ { i } )$ and upper bounds $Y _ { i } ^ { \dagger }$ . For $t \in \{ 0 , 1 \}$ , define

$$
D _ { i } ( t ) : = t Y _ { i } ( 1 ) + ( 1 - t ) \{ 1 - Y _ { i } ( 0 ) \} , \qquad q _ { t } ( x ) : = \mathbb { P } ( D _ { i } ( t ) = 0 \mid X _ { i } = x ) ,
$$

so that $D _ { i } ( T _ { i } ) = Y _ { i } ^ { \dag } , q _ { 1 } ( x ) = 1 - \mu _ { 1 } ( x ) , q _ { 0 } ( x ) = \mu _ { 0 } ( x )$ , and our previously defined oracle label satisfies

$$
Y _ { i } ^ { * } = \operatorname* { m a x } \{ D _ { i } ( 0 ) , D _ { i } ( 1 ) \} .
$$

Let $\nu ( x ) : = \mathbb { P } ( Y _ { i } ^ { * } = 0 \mid X _ { i } = x )$ . Since $\{ Y _ { i } ^ { * } = 0 \} \subseteq \{ D _ { i } ( t ) = 0 \}$ for each $t \in \{ 0 , 1 \}$ , we have

$$
\nu ( x ) \leq \gamma ( x ) = \operatorname* { m i n } \{ q _ { 0 } ( x ) , q _ { 1 } ( x ) \} \leq q _ { t } ( x ) .
$$

Define

$$
\begin{array} { r } { \rho _ { t } ( x ) : = \left\{ \begin{array} { l l } { \displaystyle \frac { \gamma ( x ) - \nu ( x ) } { q _ { t } ( x ) - \nu ( x ) } , } & { q _ { t } ( x ) > \nu ( x ) , } \\ { 0 , } & { q _ { t } ( x ) = \nu ( x ) . } \end{array} \right. } \end{array}
$$

Then $\rho _ { t } ( x ) \in [ 0 , 1 ] ;$ moreover, if $q _ { t } ( x ) = \nu ( x )$ , then necessarily $\gamma ( \boldsymbol { x } ) = \nu ( \boldsymbol { x } )$ . On an enlarged probability space, let $U _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( 0 , 1 )$ be independent of all existing random variables, and define

$$
Y _ { i } ^ { * * } : = Y _ { i } ^ { * } - \mathbb { 1 } \left\{ Y _ { i } ^ { * } = 1 , D _ { i } ( T _ { i } ) = 0 , U _ { i } \leq \rho _ { T _ { i } } ( X _ { i } ) \right\} .
$$

By construction, $Y _ { i } ^ { * * } \le Y _ { i } ^ { * }$ . Furthermore, if $Y _ { i } ^ { \dagger } = 1$ , then $D _ { i } ( T _ { i } ) = 1$ , so the indicator in the preceding display vanishes and $Y _ { i } ^ { * * } = Y _ { i } ^ { * } = 1$ . Hence

$$
Y _ { i } ^ { \dagger } \leq Y _ { i } ^ { * * } \leq Y _ { i } ^ { * } \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

By unconfoundedness, for each $t \in \{ 0 , 1 \}$ ,

$$
\begin{array} { r l } & { \mathbb { P } \big ( { Y } _ { i } ^ { * * } = 0 \ | \ X _ { i } = x , { T } _ { i } = t \big ) = \nu ( x ) + \rho _ { t } ( x ) \mathbb { P } \big ( { Y } _ { i } ^ { * } = 1 , { D } _ { i } ( t ) = 0 \ | \ X _ { i } = x \big ) } \\ & { \qquad = \nu ( x ) + \rho _ { t } ( x ) \big \{ q _ { t } ( x ) - \nu ( x ) \big \} = \gamma ( x ) . } \end{array}
$$

Thus, in particular, $\mathbb { P } ( Y _ { i } ^ { * * } = 0 \mid X _ { i } ) = \gamma ( X _ { i } )$

To avoid introducing a cutof that depends on the test point, define $\begin{array} { r } { S _ { n } : = \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } } \end{array}$ , and, on the event $\{ S _ { n } > 0 \}$ , define the calibration-only functions

$$
\hat { F } _ { n , 0 } ( t ) : = \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq t \} } { S _ { n } } , \quad \hat { F } _ { n , 0 } ^ { * } ( t ) : = \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } \mathbb { 1 } \{ Y _ { i } ^ { * * } = 0 \} \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq t \} } { S _ { n } } .
$$

Under the assumptions of the theorem, $S _ { n } > 0$ with probability tending to one. Moreover, assuming that the estimated weights are nonnegative, both $\hat { F } _ { n , 0 }$ and $\hat { F } _ { n , 0 } ^ { * }$ are nondecreasing functions taking values in [0, 1].

The observed p-value thus obeys

$$
p _ { n + 1 } ^ { \mathrm { o b s } } = \frac { \hat { w } _ { n + 1 } + S _ { n } \hat { F } _ { n , 0 } \big ( \hat { s } ( X _ { n + 1 } ) \big ) } { \hat { w } _ { n + 1 } + S _ { n } } \geq \hat { F } _ { n , 0 } ( \hat { s } ( X _ { n + 1 } ) )
$$

since $\hat { F } _ { n , 0 } ( t ) \in [ 0 , 1 ]$ for any $t \in \mathbb { R }$ . Furthermore, since $Y _ { i } ^ { \dag } \leq Y _ { i } ^ { * * }$ almost surely, we have

$$
\hat { F } _ { n , 0 } ( t ) \geq \hat { F } _ { n , 0 } ^ { * } ( t ) \qquad \mathrm { f o r ~ e v e r y ~ } t \in \mathbb { R } .
$$

It follows that

$$
\begin{array} { r } { \{ p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha \} \subseteq \big \{ \hat { F } _ { n , 0 } ( \hat { s } ( X _ { n + 1 } ) ) \leq \alpha \big \} \subseteq \big \{ \hat { F } _ { n , 0 } ^ { * } ( \hat { s } ( X _ { n + 1 } ) ) \leq \alpha \big \} . } \end{array}
$$

For later use, define the calibration-only cutofs

$$
\hat { \tau } _ { 0 } : = \operatorname* { s u p } \{ t \in \mathbb { R } : \hat { F } _ { { n , 0 } } ( t ) \leq \alpha \} , \qquad \hat { \tau } _ { 0 } ^ { * } : = \operatorname* { s u p } \{ t \in \mathbb { R } : \hat { F } _ { { n , 0 } } ^ { * } ( t ) \leq \alpha \} .
$$

The preceding pointwise ordering implies $\hat { \tau } _ { 0 } \leq \hat { \tau } _ { 0 } ^ { * }$ . Moreover,

$$
\begin{array} { r } { \{ p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha \} \subseteq \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } \} \subseteq \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} . } \end{array}
$$

Importantly, both $\hat { \tau } _ { 0 }$ and $\hat { \tau } _ { 0 } ^ { * }$ depend only on the calibration data, together with the auxiliary randomness used to construct $Y _ { i } ^ { * * }$ , and do not depend on the test point. Let $A _ { n }$ denote the σ-field generated by these quantities. Conditional on $A _ { n } .$ and writing

$$
R _ { n } : = \mathbb { P } \left( Y _ { n + 1 } ( 1 ) = 0 , Y _ { n + 1 } ( 0 ) = 1 , p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha \mid { \cal A } _ { n } \right) ,
$$

the sharp conditional upper bound $\mathbb { P } ( Y ( 1 ) = 0 , Y ( 0 ) = 1 \ | \ X ) \le \gamma ( X )$ gives

$$
R _ { n } \leq \mathbb { E } \left[ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \left\{ \hat { F } _ { n , 0 } ^ { * } ( \hat { s } ( X _ { n + 1 } ) ) \leq \alpha \right\} \bigg | \mathcal { A } _ { n } \right] \leq \mathbb { E } \left[ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \left\{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \right\} \big | \mathcal { A } _ { n } \right] .
$$

Since $\hat { F } _ { n , 0 } ^ { * } ( t )$ is a right-continuous and non-decreasing step function, letting $\hat { s } ( X _ { [ 1 ] } ) \leq \hat { s } ( X _ { [ 2 ] } ) \leq \cdot \cdot \cdot \leq$ $\hat { s } ( X _ { [ n ] } )$ be the order statistics and $[ 1 ] , [ 2 ] , \ldots , [ n ]$ is a permutation of $( 1 , \ldots , n )$ , we consider two cases:

• When there exists some $k \in [ n ]$ such that $\hat { F } _ { n , 0 } ^ { * } ( \hat { s } ( X _ { [ k ] } ) ) = \alpha$ , we know $\hat { \tau } _ { 0 } ^ { * } = \hat { s } ( X _ { [ k ] } )$ , thus $\hat { F } _ { n , 0 } ^ { * } ( \hat { \tau } _ { 0 } ^ { * } ) = \alpha$

• When there exists some $k \in [ n ]$ such that $\hat { F } _ { n , 0 } ^ { * } ( \hat { s } ( X _ { [ k - 1 ] } ) ) < \alpha < \hat { F } _ { n , 0 } ^ { * } ( \hat { s } ( X _ { [ k ] } ) )$ , we know $\hat { \tau } _ { 0 } ^ { * } = \hat { s } ( X _ { [ k ] } )$ ， and $\begin{array} { r } { \alpha \prec \hat { F } _ { n , 0 } ^ { * } ( \hat { \tau } ^ { * } ) \ \le \ \alpha + \ \hat { F } _ { n , 0 } ^ { * } ( X _ { [ k ] } ) - \hat { F } _ { n , 0 } ^ { * } ( X _ { [ k - 1 ] } ) \ = \ \alpha + \ \frac { \hat { w } _ { [ k - 1 ] } } { \sum _ { i \in \mathcal { T } _ { \mathrm { c o l l h } } , \ } \hat { w } _ { i } + \hat { w } _ { n + 1 } } \ = \ \alpha + O _ { P } ( 1 / n ) } \end{array}$ by Lemma A.11.

Combining the two cases yields $\alpha \leq \hat { F } _ { n , 0 } ^ { * } ( \hat { \tau } _ { 0 } ^ { * } ) \leq \alpha + O _ { P } ( 1 / n )$

The above arguments yield

$$
\begin{array} { r l } & { R _ { n } \leq \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} | \mathcal { A } _ { n } ] + \alpha - \hat { F } _ { n , 0 } ^ { * } ( \hat { \tau } _ { 0 } ^ { * } ) + O _ { P } ( 1 / n ) } \\ & { \quad \leq \alpha + \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} | \mathcal { A } _ { n } ] - \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } \mathbb { 1 } \{ Y _ { i } ^ { * * } = 0 \} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } } \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { \tau } _ { 0 } ^ { * } \} } \end{array} + O _ { P } ( 1 / n ) ,\tag{C.1}
$$

We shall repeatedly invoke the following lemma, whose proof is in Appendix F.1.

Lemma C.1. Under Assumption A.6, for any random variable $\left\{ Z _ { i } \right\}$ such that $( X _ { i } , Y _ { i } , T _ { i } , Z _ { i } )$ are i.i.d. across $i \in [ n ]$ and $\mathbb { E } [ Z ^ { 2 } ] < \infty .$ , and any (possibly random) function $f \colon \mathcal { X } \to \mathbb { R }$ , it holds that

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \bigg | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { w } _ { i } Z _ { i } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { \omega } ( X _ { i } ) Z _ { i } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} \bigg | = O _ { P } ( r _ { n } ) .
$$

Lemma C.1 with $r _ { n } = o ( 1 )$ implies $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) + O _ { P } ( r _ { n } ) } \end{array}$ , and together with $Z _ { i } =$ $G _ { i } 1 \mathbb { \{ } Y _ { i } ^ { * * } = 0 \}$ and $f = \hat { s }$ it implies

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left| \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } \mathbb { 1 } \left\{ Y _ { i } ^ { * * } = 0 \right\} \mathbb { 1 } \left\{ \hat { s } ( X _ { i } ) \leq t \right\} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } } - \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) \mathbb { 1 } \left\{ Y _ { i } ^ { * * } = 0 \right\} \mathbb { 1 } \left\{ \hat { s } ( X _ { i } ) \leq t \right\} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) } \right| = O _ { P } ( r _ { n } ) .
$$

Taking $t = \hat { \tau } _ { 0 } ^ { * }$ in the above display, and following (C.1), we have

$$
R _ { n } \leq \alpha + \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \left\{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \right\} | \mathcal { A } _ { n } ] - \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) \mathbb { 1 } \left\{ Y _ { i } ^ { * * } = 0 , \hat { s } ( X _ { i } ) \leq \hat { \tau } _ { 0 } ^ { * } \right\} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) } + O _ { P } ( r _ { n } + 1 / n ) .\tag{C.2}
$$

Consider $Z _ { i } : = G _ { i } \hat { \omega } ( X _ { i } ) \{ \mathbb { 1 } ( Y _ { i } ^ { * * } = 0 ) - \gamma ( X _ { i } ) \}$ . Since $G _ { i }$ is generated independently of $( Y _ { i } ( 0 ) , Y _ { i } ( 1 ) , U _ { i } )$ conditional on $( X _ { i } , T _ { i } )$ , and $\mathbb { P } ( Y _ { i } ^ { * * } = 0 \mid X _ { i } , T _ { i } ) = \gamma ( X _ { i } )$ , we have

$$
\begin{array} { r l } & { \mathbb { E } [ Z _ { i } \mid X _ { i } ] = \hat { \omega } ( X _ { i } ) \mathbb { E } \left[ \mathbb { E } \left[ G _ { i } \{ 1 ( Y _ { i } ^ { * * } = 0 ) - \gamma ( X _ { i } ) \} \mid X _ { i } , T _ { i } \right] \mid X _ { i } \right] } \\ & { \quad \quad \quad = \hat { \omega } ( X _ { i } ) \mathbb { E } \left[ \hat { g } ( X _ { i } , T _ { i } ) \{ \mathbb { P } ( Y _ { i } ^ { * * } = 0 \mid X _ { i } , T _ { i } ) - \gamma ( X _ { i } ) \} \big | X _ { i } \right] = 0 . } \end{array}
$$

Thus, invoking Lemma F.3 with this $Z _ { i }$ and $s ( \cdot ) = \hat { s } ( \cdot )$ yields

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left| \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) [ \mathbb { 1 } \{ Y _ { i } ^ { * * } = 0 \} - \gamma ( X _ { i } ) ] \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq t \} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) } \right| = O _ { P } ( 1 / \sqrt { n } ) .
$$

Continuing with (C.2), this implies

$$
R _ { n } \leq \alpha + \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} | \mathcal { A } _ { n } ] - \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) \gamma ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { \tau } _ { 0 } ^ { * } \} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) } + O _ { P } ( 1 / \sqrt { n } + r _ { n } ) .\tag{C.3}
$$

Case 1: $\| \hat { \omega } - w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 )$ . Let $I _ { t } ( \boldsymbol { x } ) : = \mathbb { 1 } \{ \hat { s } ( \boldsymbol { x } ) \leq t \}$ . Conditional on $\mathcal { T } _ { n }$ , uniform empirical convergence gives

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \bigg | \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) \gamma ( X _ { i } ) I _ { t } ( X _ { i } ) } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) } - \frac { \mathbb { E } [ q ( X ) \hat { \omega } ( X ) \gamma ( X ) I _ { t } ( X ) \mid \mathcal { T } _ { n } ] } { \mathbb { E } [ q ( X ) \hat { \omega } ( X ) \mid \mathcal { T } _ { n } ] } \bigg | = o _ { P } ( 1 ) .
$$

Since $0 \leq q , \gamma , I _ { t } \leq 1$ , we have

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left| \mathbb { E } [ q ( X ) \{ \hat { \omega } ( X ) - w ^ { \circ } ( X ) \} \gamma ( X ) I _ { t } ( X ) \mid \mathcal { T } _ { n } ] \right| = o _ { P } ( 1 ) ,
$$

and $\mathbb { E } [ q ( X ) \{ \hat { \omega } ( X ) - w ^ { \circ } ( X ) \} \mid { \mathcal { T } } _ { n } ] = o _ { P } ( 1 )$ . Moreover, $q ( x ) w ^ { \circ } ( x ) = p _ { G }$ , and therefore

$$
\frac { \mathbb { E } [ q ( X ) w ^ { \circ } ( X ) \gamma ( X ) I _ { t } ( X ) \mid \mathcal { T } _ { n } ] } { \mathbb { E } [ q ( X ) w ^ { \circ } ( X ) \mid \mathcal { T } _ { n } ] } = \mathbb { E } [ \gamma ( X ) I _ { t } ( X ) \mid \mathcal { T } _ { n } ] .
$$

Since $p _ { G }$ is bounded away from zero, it follows that

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left| \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) \gamma ( X _ { i } ) I _ { t } ( X _ { i } ) } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ( X _ { i } ) } - \mathbb { E } [ \gamma ( X ) I _ { t } ( X ) \mid \mathcal { T } _ { n } ] \right| = o _ { P } ( 1 ) .
$$

Evaluating this display at $t = \hat { \tau } _ { 0 } ^ { * }$ in (C.3) gives $R _ { n } \leq \alpha + o _ { P } ( 1 )$ .

Case 2: $\| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } \bigl ( 1 \bigr )$ . For $\xi \in \{ \hat { \gamma } , \gamma \}$ , define the training-conditional population curves

$$
H _ { n } ^ { \xi } ( t ) : = \mathbb { E } [ \xi ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} | \mathcal { T } _ { n } ] , \qquad F _ { n } ^ { \xi } ( t ) : = \frac { \mathbb { E } [ \hat { \omega } ( X ) \hat { g } ( X , T ) \xi ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} | \mathcal { T } _ { n } ] } { \mathbb { E } [ \hat { \omega } ( X ) \hat { g } ( X , T ) \mid \mathcal { T } _ { n } ] } .
$$

Let $t _ { n } ^ { \circ } : = \operatorname* { s u p } \{ t \in \mathbb { R } : H _ { n } ^ { \widehat { \gamma } } ( t ) \leq \alpha \}$ and recall

$$
{ \hat { Q } } _ { n } ( t ) : = { \frac { \sum _ { i = 1 } ^ { n } G _ { i } { \hat { w } } _ { i } { \hat { \gamma } } ( X _ { i } ) \mathbb { 1 } \left\{ { \hat { s } } ( X _ { i } ) \leq t \right\} } { \sum _ { i = 1 } ^ { n } G _ { i } { \hat { w } } _ { i } } } .
$$

Lemmas C.1 and F.4, applied to the numerators and denominators, give

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { Q } _ { n } ( t ) - F _ { n } ^ { \hat { \gamma } } ( t ) | = o _ { P } ( 1 ) , \quad \operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { F } _ { n , 0 } ^ { * } ( t ) - F _ { n } ^ { \gamma } ( t ) | = o _ { P } ( 1 ) .
$$

Here the second display also uses $\mathbb { P } ( Y _ { i } ^ { * * } = 0 \mid X _ { i } , T _ { i } ) = \gamma ( X _ { i } )$ . Moreover, since the denominator of $F _ { n } ^ { \xi }$ is bounded away from zero,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | F _ { n } ^ { \gamma } ( t ) - F _ { n } ^ { \gamma } ( t ) | + \operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ^ { \gamma } ( t ) - H _ { n } ^ { \gamma } ( t ) | \leq C \| \widehat { \gamma } - \gamma \| _ { L _ { 1 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) .
$$

By Lemma F.4 and the assumed local slope, $\hat { t } - t _ { n } ^ { \circ } = O _ { P } ( n ^ { - 1 / 2 } )$ . The balancing condition in Assumption A.6, together with the definition of t<sup>ˆ</sup> and the fact that the maximal jump of its defining empirical curve is at most $n ^ { - 1 }$ , gives $\hat { Q } _ { n } ( \hat { t } ) = \alpha + o _ { P } ( 1 )$ . Hence $F _ { n } ^ { \hat { \gamma } } ( \hat { t } ) = \alpha + o _ { P } ( 1 )$ and, by the bounded-density condition,

$$
F _ { n } ^ { \hat { \gamma } } ( t _ { n } ^ { \circ } ) = \alpha + o _ { P } ( 1 ) .
$$

The same curve has a positive local slope at $t _ { n } ^ { \circ }$ . Indeed, for $t _ { 1 } < t _ { 2 }$ in a suficiently small neighborhood of $t _ { n } ^ { \circ }$ , the lower bounds on $q$ and ˆω, together with their boundedness, imply

$$
F _ { n } ^ { \hat { \gamma } } ( t _ { 2 } ) - F _ { n } ^ { \hat { \gamma } } ( t _ { 1 } ) \geq c \{ H _ { n } ^ { \hat { \gamma } } ( t _ { 2 } ) - H _ { n } ^ { \hat { \gamma } } ( t _ { 1 } ) \}
$$

for some deterministic $c > 0$ . Therefore, for every fixed $\varepsilon > 0$ , with probability tending to one,

$$
F _ { n } ^ { \gamma } ( t _ { n } ^ { \circ } - \varepsilon ) < \alpha < F _ { n } ^ { \gamma } ( t _ { n } ^ { \circ } + \varepsilon ) .
$$

Combining this strict separation with sup $| \hat { F } _ { n , 0 } ^ { * } ( t ) - F _ { n } ^ { \gamma } ( t ) | = o _ { P } ( 1 )$ and the monotonicity of $\hat { F } _ { n , 0 } ^ { * }$ yields

$$
\hat { \tau } _ { 0 } ^ { * } - t _ { n } ^ { \circ } = o _ { P } ( 1 ) .
$$

Finally, conditional on $A _ { n } .$ the test point is independent of the training and calibration data. Thus, by (C.3) and the preceding uniform convergence,

$$
R _ { n } \leq \alpha + H _ { n } ^ { \gamma } ( \hat { \tau } _ { 0 } ^ { * } ) - F _ { n } ^ { \gamma } ( \hat { \tau } _ { 0 } ^ { * } ) + o _ { P } ( 1 ) .
$$

The bounded-density condition, $\hat { \tau } _ { 0 } ^ { * } - t _ { n } ^ { \circ } = o _ { P } ( 1 )$ , and the uniform diferences between the $\gamma -$ and ˆγ-curves imply $H _ { n } ^ { \gamma } ( \hat { \tau } _ { 0 } ^ { * } ) = \alpha + o _ { P } ( 1 )$ and $F _ { n } ^ { \gamma } ( \hat { \tau } _ { 0 } ^ { * } ) = \alpha + o _ { P } ( 1 )$ . Consequently, $R _ { n } \leq \alpha + o _ { P } ( 1 )$ . Marginalizing over $A _ { n }$ completes the proof in this case. □

## C.2 Proof of Theorem A.9

Proof of Theorem A.9. We reuse the proof of Theorem 5.1 up to (C.3) to prove the rate double robustness result. Note that the results up to (C.3) did not use any assumptions on the fitted models. Define

$$
\hat { \omega } ^ { \dagger } ( x ) : = \frac { \hat { \omega } ( x ) } { p _ { G } } .
$$

Since $q ( x ) \geq c ,$ , we have $p _ { G } = \mathbb { E } [ q ( X ) \mid { \mathcal { T } } _ { n } ] \geq c .$ . Therefore, Assumption A.8 implies

$$
\| \hat { \omega } ^ { \dagger } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = \| \hat { \omega } - w ^ { \circ } \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } / p _ { G } = O _ { P } ( n ^ { - 1 / 4 } ) .
$$

Moreover, replacing ˆω by $\hat { \omega } ^ { \dagger }$ does not change any weighted empirical or population ratio. Recall that $A _ { n }$ is the σ-field that includes the randomness in the labeled data, the training process, and generating $\{ G _ { i } \}$

and we denote the ${ \mathcal { A } } _ { n }$ -conditional harm rate $R _ { n } : = \mathbb { P } ( Y _ { n + 1 } ( \pi _ { \mathrm { o b s } } ( X _ { n + 1 } ) ) < Y _ { n + 1 } ( 0 ) | { \mathcal A } _ { n } )$ . The balancing condition in Assumption $\mathrm { A . 6 }$ with $r _ { n } = O ( n ^ { - 1 / 2 } )$ , together with (C.3), implies

$$
\begin{array} { r l r } {  { R _ { n } \leq \alpha + \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} | \mathcal { A } _ { n } ] - \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) \gamma ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { \tau } _ { 0 } ^ { * } \} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) } } } \\ & { } & { + \frac { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { t } \} } { \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) } - \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { t } \} + O _ { P } ( 1 / \sqrt { n } ) , } \end{array}
$$

where the second and the third terms invoked Lemma C.1 twice. By Assumption ${ \mathrm { A . 6 } } ,$ , we have

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \dot { \omega } ( X _ { i } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \dot { w } _ { i } + O _ { P } ( 1 / \sqrt { n } ) = \frac { m _ { n } } { n } \{ 1 + O _ { P } ( 1 / \sqrt { n } ) \} + O _ { P } ( 1 / \sqrt { n } ) = p _ { G } + O _ { P } ( 1 / \sqrt { n } ) .
$$

Consequently, $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) = 1 + O _ { P } ( 1 / \sqrt { n } ) } \end{array}$ , which further yields

$$
\begin{array} { r l } & { R _ { n } \leq \alpha + \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} | A _ { n } ] - \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { t } \} } \\ & { \qquad + \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { t } \} - \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } \hat { \omega } ^ { \dagger } ( X _ { i } ) \gamma ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { \tau } _ { 0 } ^ { * } \} + O _ { P } ( 1 / \sqrt { n } ) } \\ & { \leq \alpha + \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq \hat { \tau } _ { 0 } ^ { * } \} | A _ { n } ] - \mathbb { E } [ \hat { \gamma } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq \hat { t } \} | A _ { n } ] } \\ &  \qquad + \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { g } ( X _ { i } , T _ { i } ) \hat { \omega } ^ { \dagger } ( X _ { i } ) \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { t } \} - \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { g } ( X _ { i } , T _ { i } ) \hat { \omega } ^ { \dagger } ( X _ { i } ) \gamma ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq \hat { \tau } _ { 0 } ^ { * } \} + O _ { P } ( 1 / \sqrt  n \end{array}
$$

where $X \sim \mathbb { P } _ { X }$ is an independent copy. Here, the second inequality uses Lemma F.3 for $Z _ { i } ~ = ~ ( { \hat { G } } _ { i } ~ -$ $\hat { g } ( X _ { i } , T _ { i } ) ) \hat { \omega } ^ { \dagger } ( X _ { i } ) \gamma ( X _ { i } )$ or $Z _ { i } = ( \hat { G } _ { i } - \hat { g } ( X _ { i } , T _ { i } ) ) \hat { \omega } ^ { \dagger } ( X _ { i } ) \hat { \gamma } ( X _ { i } )$ , as well as Lemma F.3 for $Z _ { i } = \hat { \gamma } ( X _ { i } )$ and $H _ { n } ( t ) = \mathbb { E } [ \hat { \gamma } ( X ) \mathbb { 1 } \left\{ \hat { s } ( X ) \leq t \right\} | { \mathcal { T } } _ { n } ]$ evaluated at $t = \hat { t }$ which is adapted to the σ-field $A _ { n }$

For the oracle weight $w ( X )$ , we know $\mathbb { E } [ G _ { i } w ( X _ { i } ) ] = \mathbb { E } [ \hat { g } ( X , T ) w ( X ) ] = 1$ , and due to the covariate shift between calibration data conditional on $G _ { i } = 1$ and the test data,

$$
\begin{array} { r l } & { \mathbb { E } [ \gamma ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { \tau } _ { 0 } ^ { * } \} | \mathcal { A } _ { n } ] = \mathbb { E } [ \hat { g } ( X , T ) w ( X ) \gamma ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq \hat { \tau } _ { 0 } ^ { * } \} | \mathcal { A } _ { n } ] , } \\ & { \mathbb { E } [ \hat { \gamma } ( X _ { n + 1 } ) \mathbb { 1 } \{ \hat { s } ( X _ { n + 1 } ) \leq \hat { t } \} | \mathcal { A } _ { n } ] = \mathbb { E } [ \hat { g } ( X , T ) w ( X ) \hat { \gamma } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq \hat { t } \} | \mathcal { A } _ { n } ] , } \end{array}
$$

where the expectations are over an independent copy $( X , T ) \sim \mathbb { P } _ { X , T }$ . Thus, denoting

$$
\hat { r } ( x , t ) = \hat { g } ( x , t ) \cdot \left\{ \hat { \gamma } ( x ) \mathbb { 1 } \{ \hat { s } ( x ) \leq \hat { t } \} - \gamma ( x ) \mathbb { 1 } \{ \hat { s } ( x ) \leq \hat { \tau } _ { 0 } ^ { * } \} \right\}
$$

which is a function adapted to $A _ { n }$ and obeys $\hat { r } ( x , t ) \in [ - 1 , 1 ]$ for any value of $( x , t )$ , since $\hat { g } ( x , t ) \in [ 0 , 1 ]$ and $\hat { \gamma } ( x ) , \gamma ( x ) \in [ 0 , 1 ]$ . Then, for an independent copy $( X , T ) \sim \mathbb { P } _ { X , T }$ 2

$$
\begin{array} { r l } & { { \cal R } _ { n } \le \alpha + \displaystyle \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \hat { \omega } ^ { \dagger } ( X _ { i } ) \hat { r } ( X _ { i } , T _ { i } ) - \mathbb { E } [ w ( X ) \hat { r } ( X , T ) \mid { \cal A } _ { n } ] + O _ { P } ( 1 / \sqrt { n } ) } \\ & { \quad \le \alpha + \displaystyle \underbrace { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left\{ \hat { \omega } ^ { \dagger } ( X _ { i } ) - w ( X _ { i } ) \right\} \hat { r } ( X _ { i } , T _ { i } ) } _ { ( \mathrm { a } ) } + \displaystyle \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { i = 1 } ^ { n } \left\{ w ( X _ { i } ) \hat { r } ( X _ { i } , T _ { i } ) - \mathbb { E } [ w ( X ) \hat { r } ( X , T ) \mid { \cal A } _ { n } ] \right\} + O _ { P } ( 1 / \sqrt { n } ) . } \end{array}
$$

We now proceed to control the two terms separately via uniform convergence. Let $\mathcal { T } _ { n }$ denote the σ-field generated by the training processes, conditional on which $\hat { g } , \hat { \gamma }$ , and ˆs are fixed. Writing $P _ { n }$ for the empirical measure of $( X _ { i } , T _ { i } ) _ { i = 1 } ^ { n }$ and $P$ for the expectation over an independent copy, define

$$
f _ { 1 , u } ( \boldsymbol { x } , t ) : = w ( \boldsymbol { x } ) \hat { g } ( \boldsymbol { x } , t ) \hat { \gamma } ( \boldsymbol { x } ) \mathbb { 1 } \{ \hat { s } ( \boldsymbol { x } ) \leq u \} , \qquad f _ { 2 , v } ( \boldsymbol { x } , t ) : = w ( \boldsymbol { x } ) \hat { g } ( \boldsymbol { x } , t ) \gamma ( \boldsymbol { x } ) \mathbb { 1 } \{ \hat { s } ( \boldsymbol { x } ) \leq v \} .
$$

By definition, as the cutofs $\hat { \tau } _ { 0 } ^ { * }$ and $\hat { t }$ in the definition of $\hat { r } ( \cdot , \cdot )$ is fixed given $A _ { n }$ , we know

$$
| \mathrm { ( b ) } | = \left| ( P _ { n } - P ) f _ { 1 , { \hat { t } } } - ( P _ { n } - P ) f _ { 2 , { \hat { r } } _ { 0 } ^ { \star } } \right| \leq \operatorname* { s u p } _ { u \in \mathbb { R } } | ( P _ { n } - P ) f _ { 1 , u } | + \operatorname* { s u p } _ { v \in \mathbb { R } } | ( P _ { n } - P ) f _ { 2 , v } | = O _ { P } ( 1 / { \sqrt { n } } ) ,
$$

where the last equality follows from applying Lemma F.3 twice, provided that $\begin{array} { r l } { \| w \hat { g } \hat { \gamma } \| _ { L _ { 2 } ( \mathbb { P } _ { X , T } ) } + \| w \hat { g } \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X , T } ) } = } & { { } } \end{array}$ $O _ { P } ( 1 )$ based on the boundedness of $w ( \cdot ) , \hat { g } , \gamma ,$ and ˆγ. For term (a), we define for any $u , v \in \mathbb { R }$ the function

$$
r _ { u , v } ( x , t ) : = \hat { g } ( x , t ) \left\{ \hat { \gamma } ( x ) \mathbb { 1 } \{ \hat { s } ( x ) \leq u \} - \gamma ( x ) \mathbb { 1 } \{ \hat { s } ( x ) \leq v \} \right\} ,
$$

so that $\hat { r } = r _ { \hat { t } , \hat { \tau } _ { 0 } ^ { * } }$ . We then have

$$
| \mathrm { ( a ) } | = \left| P _ { n } \{ ( \hat { \omega } ^ { \dagger } - w ) r _ { \hat { t } , \hat { \tau } _ { 0 } ^ { * } } \} \right| \leq \left| P \{ ( \hat { \omega } ^ { \dagger } - w ) r _ { \hat { t } , \hat { \tau } _ { 0 } ^ { * } } \} \right| + \operatorname* { s u p } _ { u , v \in \mathbb { R } } \left| ( P _ { n } - P ) \{ ( \hat { \omega } ^ { \dagger } - w ) r _ { u , v } \} \right| .
$$

By the H¨older’s inequality,

$$
\left. P \{ ( \hat { \omega } ^ { \dagger } - w ) r _ { \hat { t } , \hat { \tau } _ { 0 } ^ { * } } \} \right. \leq \| \hat { \omega } ^ { \dagger } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } \| \hat { r } \| _ { L _ { 1 } ( \mathbb { P } _ { X , T } ) } .
$$

Moreover,

$$
\begin{array} { r l } & { \underset { u , v \in \mathbb { R } } { \operatorname* { s u p } } \left| ( P _ { n } - P ) \{ ( \hat { \omega } ^ { \dagger } - w ) r _ { u , v } \} \right| } \\ & { \leq \underset { u \in \mathbb { R } } { \operatorname* { s u p } } \left| ( P _ { n } - P ) \left[ ( \hat { \omega } ^ { \dagger } - w ) \hat { g } \hat { \gamma } \mathbb { 1 } \{ \hat { s } \leq u \} \right] \right| + \underset { v \in \mathbb { R } } { \operatorname* { s u p } } \left| ( P _ { n } - P ) \left[ ( \hat { \omega } ^ { \dagger } - w ) \hat { g } \gamma \mathbb { 1 } \{ \hat { s } \leq v \} \right] \right| . } \end{array}
$$

Conditional on $\mathcal { T } _ { n }$ , both terms are indexed by nested threshold classes. Since $\hat { g } , \hat { \gamma } , \gamma \in [ 0 , 1 ]$ and $\| \hat { \omega } ^ { \dagger } - \|$ $w \vert \vert _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = O _ { P } ( 1 )$ , applying Lemma F.3 twice gives $\begin{array} { r } { \operatorname* { s u p } _ { u , v \in \mathbb { R } } \left| ( P _ { n } - P ) \{ ( \hat { \omega } ^ { \dagger } - w ) r _ { u , v } \} \right| = O _ { P } ( n ^ { - 1 / 2 } ) } \end{array}$ . Thus,

$$
| ( a ) | \leq \| \hat { \omega } ^ { \dagger } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } \| \hat { r } \| _ { L _ { 1 } ( \mathbb { P } _ { X , T } ) } + O _ { P } ( n ^ { - 1 / 2 } ) .
$$

Combining the above two bounds gives

$$
R _ { n } \leq \alpha + \| \hat { \omega } ^ { \dagger } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } \| \hat { r } \| _ { L _ { 1 } ( \mathbb { P } _ { X , T } ) } + O _ { P } ( n ^ { - 1 / 2 } ) .\tag{C.4}
$$

We now proceed to show $\| \hat { r } \| _ { L _ { 1 } ( \mathbb { P } _ { X , T } ) } = O _ { P } ( n ^ { - 1 / 4 } )$ . By definition, as $\hat { g } \in [ 0 , 1 ]$ and $\gamma \in [ 0 , 1 ]$

$$
\begin{array} { r l } & { \left\| \hat { r } \right\| _ { L _ { 1 } ( { \mathbb { P } } _ { X , T } ) } \leq \left\| \hat { \gamma } ( \cdot ) { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { t } \right\} - \gamma ( \cdot ) { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { \tau } _ { 0 } ^ { * } \right\} \right\| _ { L _ { 1 } ( { \mathbb { P } } _ { X , T } ) } } \\ & { \qquad \leq \left\| \hat { \gamma } ( \cdot ) { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { t } \right\} - \gamma ( \cdot ) { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { t } \right\} \right\| _ { L _ { 1 } ( { \mathbb { P } } _ { X , T } ) } + \left\| \gamma ( \cdot ) { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { t } \right\} - \gamma ( \cdot ) { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { \tau } _ { 0 } ^ { * } \right\} \right\| _ { L _ { 1 } ( { \mathbb { P } } _ { X , T } ) } } \\ & { \qquad \leq \left\| \hat { \gamma } - \gamma \right\| _ { L _ { 1 } ( { \mathbb { P } } _ { X , T } ) } + \left\| { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { t } \right\} - { \bf 1 } \left\{ \hat { s } ( \cdot ) \leq \hat { \tau } _ { 0 } ^ { * } \right\} \right\| _ { L _ { 1 } ( { \mathbb { P } } _ { X , T } ) } \leq O _ { P } ( n ^ { - 1 / 4 } ) + O ( | \hat { \tau } _ { 0 } ^ { * } - \hat { t } | ) , } \end{array}
$$

where the last inequality uses the bounded-density condition on $\hat { s } ( \cdot )$

In the following, we show that $| \hat { \tau } _ { 0 } ^ { * } - \hat { t } | = O _ { P } ( n ^ { - 1 / 4 } )$ under the given conditions. Recall that $H _ { n } ( t ) : =$ $\mathbb { E } [ \hat { \gamma } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \mid { \mathcal { T } } _ { n } ]$ and $t _ { n } ^ { \circ } : = \operatorname* { s u p } \{ t \in \mathbb { R } : H _ { n } ( t ) \leq \alpha \}$ . By Lemma F.4,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \widehat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \widehat { s } ( X _ { i } ) \leq t \} - H _ { n } ( t ) \right| = O _ { P } ( n ^ { - 1 / 2 } ) .
$$

Together with condition (ii) of Theorem A.9, the cutof conclusion of Lemma F.4 gives $| \hat { t } - t _ { n } ^ { \circ } | = O _ { P } { \left( n ^ { - 1 / 2 } \right) }$ Since Assumption A.6 holds with $r _ { n } = O ( n ^ { - 1 / 2 } )$ , Lemmas C.1 and F.4, together with $\mathbb { P } ( Y _ { i } ^ { * * } = 0 \mid$ $X _ { i } , T _ { i } ) = \gamma ( X _ { i } )$ , give

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { F } _ { n , 0 } ^ { * } ( t ) - F _ { n } ^ { \circ } ( t ) | = O _ { P } ( n ^ { - 1 / 2 } ) , \quad \mathrm { w h e r e } \quad F _ { n } ^ { \circ } ( t ) : = \frac { \mathbb { E } \left[ \hat { \omega } ^ { \dagger } ( X ) \hat { g } ( X , T ) \gamma ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \le t \} ~ \big | ~ { \mathcal T } _ { n } \right] } { \mathbb { E } \left[ \hat { \omega } ^ { \dagger } ( X ) \hat { g } ( X , T ) ~ \big | ~ { \mathcal T } _ { n } \right] } .
$$

Write $D _ { n } : = \mathbb { E } \big [ \hat { \omega } ^ { \dagger } ( X ) \hat { g } ( X , T ) | \mathcal { T } _ { n } \big ]$ . For every integrable function h, we have $\mathbb { E } [ w ( X ) \hat { g } ( X , T ) h ( X ) | \mathcal { T } _ { n } ] =$ $\mathbb { E } [ h ( X ) \mid T _ { n } ]$ . It follows that $| D _ { n } - 1 | \leq \| \hat { \omega } ^ { \dagger } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } \big ( n ^ { - 1 / 4 } \big )$ , and

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left. \mathbb { E } \left[ \hat { \omega } ^ { \dagger } ( X ) \hat { g } ( X , T ) \gamma ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \mid \mathcal { T } _ { n } \right] - H _ { n } ( t ) \right. \leq \lVert \hat { \omega } ^ { \dagger } - w \rVert _ { L _ { \infty } ( \mathbb { P } _ { X } ) } + \lVert \hat { \gamma } - \gamma \rVert _ { L _ { 1 } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } ) .
$$

Since $D _ { n } = 1 + o _ { P } ( 1 )$ and $0 \leq H _ { n } ( t ) \leq 1$ , on the event $D _ { n } \geq 1 / 2$ , it holds that $\begin{array} { r } { \operatorname* { s u p } _ { t \in \mathbb { R } } | F _ { n } ^ { \circ } ( t ) - H _ { n } ( t ) | = } \end{array}$ $O _ { P } ( n ^ { - 1 / 4 } )$ . Therefore, by the triangle inequality,

$$
\varepsilon _ { n } : = \operatorname* { s u p } _ { t \in \mathbb { R } } \vert \hat { F } _ { n , 0 } ^ { * } ( t ) - H _ { n } ( t ) \vert \leq \operatorname* { s u p } _ { t \in \mathbb { R } } \vert \hat { F } _ { n , 0 } ^ { * } ( t ) - F _ { n } ^ { \circ } ( t ) \vert + \operatorname* { s u p } _ { t \in \mathbb { R } } \vert F _ { n } ^ { \circ } ( t ) - H _ { n } ( t ) \vert = O _ { P } ( n ^ { - 1 / 4 } ) .
$$

Let ${ \mathcal { E } } _ { n }$ denote the event on which $H _ { n } ( t _ { n } ^ { \circ } - u ) \leq \alpha - c u$ and $H _ { n } ( t _ { n } ^ { \circ } + u ) \geq \alpha + c u$ for every $0 < u \leq \eta$ . By condition (ii) of Theorem A.9, we know $\mathbb { P } ( \mathcal { E } _ { n } )  1$ . Set $\bar { \varepsilon } _ { n } : = \varepsilon _ { n } + n ^ { - 1 }$ and $\begin{array} { r } { u _ { n } : = \frac { 2 \bar { \varepsilon } _ { n } } { c } } \end{array}$ . Since $\varepsilon _ { n } = O _ { P } ( n ^ { - 1 / 4 } )$ , we have $u _ { n } = O _ { P } ( n ^ { - 1 / 4 } )$ and $\mathbb { P } \{ \mathcal { E } _ { n } \cap \{ u _ { n } \leq \eta \} \}  1$ . On this event,

$$
\begin{array} { r l } & { \hat { F } _ { n , 0 } ^ { * } ( t _ { n } ^ { \circ } - u _ { n } ) \leq H _ { n } ( t _ { n } ^ { \circ } - u _ { n } ) + \varepsilon _ { n } \leq \alpha - c u _ { n } + \varepsilon _ { n } < \alpha , } \\ & { \hat { F } _ { n , 0 } ^ { * } ( t _ { n } ^ { \circ } + u _ { n } ) \geq H _ { n } ( t _ { n } ^ { \circ } + u _ { n } ) - \varepsilon _ { n } \geq \alpha + c u _ { n } - \varepsilon _ { n } > \alpha . } \end{array}
$$

Because $\hat { F } _ { n . 0 } ^ { * }$ is non-decreasing, it follows that $t _ { n } ^ { \circ } - u _ { n } \leq \hat { \tau } _ { 0 } ^ { * } \leq t _ { n } ^ { \circ } + u _ { n }$ . Therefore, we have $| \hat { \tau } _ { 0 } ^ { * } - t _ { n } ^ { \circ } | =$ $O _ { P } ( n ^ { - 1 / 4 } )$ , and hence

$$
| \hat { \tau } _ { 0 } ^ { * } - \hat { t } | \leq | \hat { \tau } _ { 0 } ^ { * } - t _ { n } ^ { \circ } | + | \hat { t } - t _ { n } ^ { \circ } | = O _ { P } ( n ^ { - 1 / 4 } ) .
$$

Putting it back to (C.4), we conclude the proof of Theorem A.9.

## C.3 Proof of Lemma A.11

Proof of Lemma A.11. Let $\mathcal { N } : = \{ t \in \mathbb { R } : | t - t ^ { \circ } | \leq \eta \}$ and write $P _ { n , G } f : = m _ { n } ^ { - 1 } \sum _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } } f ( X _ { i } )$ and $P _ { G } f : = \mathbb { E } [ f ( X ) \mid G = 1 , \mathcal { T } _ { n } ]$ . We work conditional on $\mathcal { T } _ { n }$ and on the event in Assumption A.10, whose probability tends to one. Conditional on $\mathcal { T } _ { n }$ , the pairs $( X _ { i } , G _ { i } )$ are i.i.d., with $\mathbb { P } ( G _ { i } = 1 \mid X _ { i } = x , { \mathcal { T } } _ { n } ) = q ( x )$ and $\mathbb { P } ( G _ { i } = 1 \mid \mathcal { T } _ { n } ) = p _ { G }$ . All the conditional stochastic bounds below therefore also hold marginally.

For brevity, throughout the first part of the proof write $\psi _ { t } : = \psi _ { t } ^ { \hat { \gamma } , \tilde { w } } , M ( t ) : = M _ { \hat { \gamma } , \tilde { w } } ( t ) , b ( t ) : = b _ { \hat { \gamma } , \tilde { w } } ( t )$ and $\omega _ { t } : = \omega _ { t } ^ { \hat { \gamma } , \tilde { w } }$ . The coordinates of $\psi _ { t }$ and their pairwise products are uniformly bounded and indexed by the nested sets $\{ \hat { s } ( X ) \leq t \}$ . Applying Lemma F.4 coordinatewise gives

$$
\operatorname* { s u p } _ { t \in \mathcal { N } } \| \bar { \psi } _ { n } ( t ) - b ( t ) \| = O _ { P } ( n ^ { - 1 / 2 } ) , \qquad \frac { m _ { n } } { n } = p _ { G } + O _ { P } ( n ^ { - 1 / 2 } ) ,\tag{C.5}
$$

$$
\operatorname* { s u p } _ { t \in \mathcal { N } } \| \hat { M } _ { n } ( t ) - M ( t ) \| _ { \mathrm { o p } } = O _ { P } ( n ^ { - 1 / 2 } ) , \qquad \operatorname* { s u p } _ { u \in \mathbb { R } } | P _ { n , G } \mathbb { 1 } \{ \hat { s } ( X ) \leq u \} - P _ { G } \mathbb { 1 } \{ \hat { s } ( X ) \leq u \} | = O _ { P } ( n ^ { - 1 / 2 } ) .
$$

Since $p _ { G }$ and in $\mathrm { f } _ { t \in \mathcal { N } } \lambda _ { \operatorname* { m i n } } \{ M ( t ) \}$ are bounded away from $\mathrm { z e r o } , \hat { M } _ { n } ( t )$ is invertible uniformly over $t \in \mathcal N$ with probability tending to one.

Define $\begin{array} { r } { H _ { n } ( t ) : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \hat { \gamma } ( X _ { i } ) \mathbb { 1 } \{ \hat { s } ( X _ { i } ) \leq t \} } \end{array}$ and $H ( t ) : = \mathbb { E } [ \hat { \gamma } ( X ) \mathbb { 1 } \{ \hat { s } ( X ) \leq t \} \mid \mathcal { T } _ { n } ]$ . Lemma F.4 gives $\operatorname* { s u p } _ { t } | H _ { n } ( t ) - H ( t ) | = O _ { P } { \left( n ^ { - 1 / 2 } \right) }$ . The local-slope condition in Assumption A.10 therefore yields

$$
| \hat { t } - t ^ { \circ } | = O _ { P } ( n ^ { - 1 / 2 } ) .\tag{C.6}
$$

In particular, $\hat { t } \in \mathcal N$ with probability tending to one.

We next characterize the optimizer. Fix $t \in \mathcal N$ and let $\Psi _ { t }$ be the $m _ { n } \times 3$ matrix whose row indexed by $i \in \mathcal { T } _ { \mathrm { c a l i b } }$ is $\psi _ { t } ( X _ { i } ) ^ { \top }$ For any $b \in B _ { n } ( t )$ , the unique vector $u \in \mathbb { R } ^ { m _ { n } }$ minimizing $\| u \| ^ { 2 }$ subject to $m _ { n } ^ { - 1 } \Psi _ { t } ^ { \top } u = b$ is $u = \Psi _ { t } \hat { M } _ { n } ( t ) ^ { - 1 } b .$ , with objective value $m _ { n } b ^ { \top } \hat { M } _ { n } ( t ) ^ { - 1 } b$ . Hence the unconstrained solution to the balancing program is $\hat { w } _ { i } ( t ) = \hat { \omega } _ { t } ( X _ { i } )$ . By the definition of $B _ { n } ( t )$ , equation (C.5), and $\delta _ { n } = O ( n ^ { - 1 / 2 } )$ , we know $\begin{array} { r } { \operatorname* { s u p } _ { t \in \mathcal { N } } \| \hat { b } _ { n } ( t ) - b ( t ) \| = O _ { P } \big ( n ^ { - 1 / 2 } \big ) } \end{array}$ . The identity $\hat { M } _ { n } ^ { - 1 } - M ^ { - 1 } = \hat { M } _ { n } ^ { - 1 } ( M - \hat { M } _ { n } ) M ^ { - 1 }$ then gives

$$
\operatorname* { s u p } _ { t \in \mathcal { N } } \| \hat { a } _ { n } ( t ) - M ( t ) ^ { - 1 } b ( t ) \| = O _ { P } ( n ^ { - 1 / 2 } ) , \qquad \operatorname* { s u p } _ { t \in \mathcal { N } , x \in \mathcal { X } } | \hat { \omega } _ { t } ( x ) - \omega _ { t } ( x ) | = O _ { P } ( n ^ { - 1 / 2 } ) .\tag{C.7}
$$

By Assumption $\mathrm { A . 1 0 ( i i ) }$ , we have i $\mathrm { n f } _ { t \in \mathcal { N } , x \in \mathcal { X } } \omega _ { t } ( x ) \geq c _ { 1 }$ . Thus, with probability tending to one, $\hat { \omega } _ { t } ( x ) \geq 0$ uniformly over $t \in \mathcal { N }$ and $x \in \mathcal { X }$ . The unconstrained solution is therefore feasible for the nonnegative balancing program and, by strict convexity, is its unique solution. Taking $t = \hat { t }$ proves $\hat { w } _ { i } = \hat { \omega } _ { \hat { t } } ( X _ { i } )$

We now compare the empirical weights with $\mathcal W _ { n } ( \hat { \gamma } , \tilde { w } ) = \omega _ { t ^ { \circ } }$ . Since $q ( x ) \leq 1$ and $p _ { G }$ is bounded away from zero, Assumption $\mathrm { A . 1 0 ( i v ) }$ implies $P _ { G } \{ a < \hat { s } ( X ) \leq b \} \leq C ( b - a )$ for all $a < b$ . Consequently, uniformly over $t , t ^ { \prime } \in \mathcal { N }$

$$
\| b ( t ) - b ( t ^ { \prime } ) \| + \| M ( t ) - M ( t ^ { \prime } ) \| _ { \mathrm { o p } } \le C | t - t ^ { \prime } | , \qquad \| M ( t ) ^ { - 1 } b ( t ) - M ( t ^ { \prime } ) ^ { - 1 } b ( t ^ { \prime } ) \| \le C | t - t ^ { \prime } | .
$$

Write $M ( t ) ^ { - 1 } b ( t ) = ( a _ { 0 } ( t ) , a _ { h } ( t ) , a _ { w } ( t ) ) ^ { \top }$ . Then

$$
\omega _ { t } ( x ) - \omega _ { t ^ { \prime } } ( x ) = \psi _ { t } ( x ) ^ { \top } \{ M ( t ) ^ { - 1 } b ( t ) - M ( t ^ { \prime } ) ^ { - 1 } b ( t ^ { \prime } ) \} + a _ { h } ( t ^ { \prime } ) \{ h _ { t } ^ { \widehat { \gamma } } ( x ) - h _ { t ^ { \prime } } ^ { \widehat { \gamma } } ( x ) \} ,
$$

and hence

$$
| \omega _ { t } ( x ) - \omega _ { t ^ { \prime } } ( x ) | ^ { 2 } \leq C | t - t ^ { \prime } | ^ { 2 } + C \mathbb { 1 } \{ \operatorname* { m i n } ( t , t ^ { \prime } ) < \hat { s } ( x ) \leq \operatorname* { m a x } ( t , t ^ { \prime } ) \} .\tag{C.8}
$$

Equations (C.5) and (C.6) imply

$$
P _ { n , G } \mathbb { 1 } \{ \operatorname* { m i n } ( \hat { t } , t ^ { \circ } ) < \hat { s } ( X ) \leq \operatorname* { m a x } ( \hat { t } , t ^ { \circ } ) \} = O _ { P } ( n ^ { - 1 / 2 } ) .
$$

It follows from (C.8) that

$$
P _ { n , G } \{ \omega _ { \hat { t } } ( X ) - \omega _ { t ^ { \circ } } ( X ) \} ^ { 2 } = O _ { P } ( n ^ { - 1 / 2 } ) .\tag{C.9}
$$

On the other hand, equation (C.7) gives

$$
P _ { n , G } \{ \hat { \omega } _ { \hat { t } } ( X ) - \omega _ { \hat { t } } ( X ) \} ^ { 2 } = \{ \hat { a } _ { n } ( \hat { t } ) - M ( \hat { t } ) ^ { - 1 } b ( \hat { t } ) \} ^ { \top } \hat { M } _ { n } ( \hat { t } ) \{ \hat { a } _ { n } ( \hat { t } ) - M ( \hat { t } ) ^ { - 1 } b ( \hat { t } ) \} = O _ { P } ( n ^ { - 1 } ) .\tag{C.10}
$$

Combining (C.9)–(C.10) proves

$$
\frac { 1 } { m _ { n } } \sum _ { i \in \mathcal { T } _ { \mathrm { c a l i b } } } \left\{ \hat { w } _ { i } - \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } ) ( X _ { i } ) \right\} ^ { 2 } = O _ { P } ( n ^ { - 1 / 2 } ) .
$$

The boundedness of $\psi _ { t }$ , together with Assumption A.10(ii), implies

$$
\operatorname* { s u p } _ { t \in \mathcal { N } , x \in \mathcal { X } } | \omega _ { t } ( x ) | \leq \operatorname* { s u p } _ { t \in \mathcal { N } , x \in \mathcal { X } } \| \psi _ { t } ( x ) \| \operatorname* { s u p } _ { t \in \mathcal { N } } \| M ( t ) ^ { - 1 } \| _ { \mathrm { o p } } \operatorname* { s u p } _ { t \in \mathcal { N } } \| b ( t ) \| \leq C .
$$

Together with (C.7) and $\mathbb { P } ( \boldsymbol { \hat { t } } \in \mathcal { N } )  1$ , this gives su $\mathrm { p } _ { x \in \mathcal { X } } | \hat { \omega } _ { \hat { t } } ( x ) | = O _ { P } ( 1 )$ . Since $\hat { w } _ { i } = \hat { \omega } _ { \hat { t } } ( X _ { i } )$ for $i \in \mathcal { T } _ { \mathrm { c a l i b } }$ and $\hat { w } _ { n + 1 } = \hat { \omega } _ { \hat { t } } ( X _ { n + 1 } )$ , it follows that $\hat { w } _ { n + 1 } \vee \operatorname* { m a x } _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } } \hat { w } _ { i } = O _ { P } ( 1 )$ . The normalization constraint gives $\textstyle \sum _ { i \in { \mathcal { T } } _ { \mathrm { c a l i b } } } { \hat { w } } _ { i } = m _ { n }$ , while $m _ { n } / n = p _ { G } + O _ { P } ( n ^ { - 1 / 2 } )$ and $p _ { G }$ is bounded away from zero. Therefore,

$$
\frac { \hat { w } _ { n + 1 } \vee \operatorname* { m a x } _ { i \in { \mathcal { Z } _ { \mathrm { c a l i b } } } } \hat { w } _ { i } } { \sum _ { i \in { \mathcal { Z } _ { \mathrm { c a l i b } } } } \hat { w } _ { i } } = O _ { P } ( n ^ { - 1 } ) .
$$

Suppose now that $\| \tilde { w } - w \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } \big ( 1 \big )$ . Let $a ^ { \circ } : = ( 0 , 0 , p _ { G } ) ^ { \top }$ . Since $q ( x ) w ( x ) = 1$ , we have, uniformly over $t \in \mathcal N$ , that $b ( t ) = \mathbb { E } [ q ( X ) \psi _ { t } ( X ) w ( X ) | \mathcal { T } _ { n } ]$ . On the other hand, since $\psi _ { t } ( x ) ^ { \top } a ^ { \circ } = p _ { G } \tilde { w } ( x )$ , we have

$$
M ( t ) a ^ { \circ } = \mathbb { E } [ q ( X ) \psi _ { t } ( X ) \tilde { w } ( X ) \mid \mathcal { T } _ { n } ] .
$$

Consequently, we have $b ( t ) - M ( t ) a ^ { \circ } \ = \ \mathbb { E } [ q ( X ) \psi _ { t } ( X ) \{ w ( X ) - \tilde { w } ( X ) \} \ | \ T _ { n } ]$ . By Assumption $\mathrm { A . 1 0 , ~ } \psi _ { t }$ is uniformly bounded and $M ( t ) ^ { - 1 }$ is uniformly bounded over $t \in \mathcal N$ . Hence, by the Cauchy–Schwarz inequality,

$$
\operatorname* { s u p } _ { t \in \mathcal { N } } \| M ( t ) ^ { - 1 } b ( t ) - a ^ { \circ } \| \leq C \| \tilde { w } - w \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) .
$$

Moreover, since $w ^ { \circ } ( x ) = p _ { G } w ( x )$ , we have

$$
\omega _ { t } ( x ) - w ^ { \circ } ( x ) = \psi _ { t } ( x ) ^ { \top } \{ M ( t ) ^ { - 1 } b ( t ) - a ^ { \circ } \} + p _ { G } \{ \tilde { w } ( x ) - w ( x ) \} .
$$

It follows that $\begin{array} { r } { \operatorname* { s u p } _ { t \in \mathcal { N } } \| \omega _ { t } - w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) } \end{array}$ . Taking $t = t ^ { \circ }$ and recalling that $W _ { n } ( \widehat { \gamma } , \widetilde { w } ) = \omega _ { t ^ { \mathsf { c } } }$ gives

$$
\| \mathcal { W } _ { n } ( \widehat { \gamma } , \widetilde { w } ) - w ^ { \circ } \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) .
$$

Finally, suppose $\varepsilon _ { n } : = \| \tilde { w } - w \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } \big ( n ^ { - 1 / 4 } \big )$ . The preceding argument now yields

$$
\operatorname* { s u p } _ { t \in \mathcal { N } } \| M ( t ) ^ { - 1 } b ( t ) - a ^ { \circ } \| \leq C \varepsilon _ { n } , \quad \operatorname* { s u p } _ { t \in \mathcal { N } } | a _ { h } ( t ) | \leq C \varepsilon _ { n } ,
$$

where $a _ { h } ( t )$ denotes the coeficient on $h _ { t } ^ { \hat { \gamma } }$ in $M ( t ) ^ { - 1 } b ( t )$ . Therefore, $\begin{array} { r } { \operatorname* { s u p } _ { t \in \mathcal { N } } \| \omega _ { t } - w ^ { \circ } \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } \big ( n ^ { - 1 / 4 } \big ) } \end{array}$ and in particular,

$$
\| \mathcal { W } _ { n } ( \widehat { \gamma } , \widetilde { w } ) - w ^ { \circ } \| _ { L _ { \infty } ( \mathbb { P } _ { X } ) } = O _ { P } ( n ^ { - 1 / 4 } ) .
$$

To sharpen the empirical approximation rate, recall that, uniformly over $t , t ^ { \prime } \in \mathcal { N }$

$$
| \omega _ { t } ( x ) - \omega _ { t ^ { \prime } } ( x ) | ^ { 2 } \leq C | t - t ^ { \prime } | ^ { 2 } + C \left\{ \operatorname* { s u p } _ { u \in N } | a _ { h } ( u ) | ^ { 2 } \right\} \mathbb { 1 } \{ \operatorname* { m i n } ( t , t ^ { \prime } ) < \hat { s } ( x ) \leq \operatorname* { m a x } ( t , t ^ { \prime } ) \} .
$$

The bounds already established above imply $| \hat { t } - t ^ { \circ } | = O _ { P } ( n ^ { - 1 / 2 } )$ and

$$
P _ { n , G } \mathbb { 1 } \{ \operatorname* { m i n } ( \hat { t } , t ^ { \circ } ) < \hat { s } ( X ) \leq \operatorname* { m a x } ( \hat { t } , t ^ { \circ } ) \} = O _ { P } ( n ^ { - 1 / 2 } ) .
$$

Since $\begin{array} { r } { \operatorname* { s u p } _ { u \in { \mathcal { N } } } | a _ { h } ( u ) | ^ { 2 } = O _ { P } ( n ^ { - 1 / 2 } ) } \end{array}$ , it follows that $P _ { n , G } \{ \omega _ { \hat { t } } ( X ) - \omega _ { t ^ { \circ } } ( X ) \} ^ { 2 } = O _ { P } ( n ^ { - 1 } )$ . Combining this with $P _ { n , G } \{ \hat { \omega } _ { \hat { t } } ( X ) - \omega _ { \hat { t } } ( X ) \} ^ { 2 } = O _ { P } ( n ^ { - 1 } )$ gives

$$
\frac { 1 } { m _ { n } } \sum _ { i \in \mathcal { Z } _ { \mathrm { c a l i b } } } \left\{ \hat { w } _ { i } - \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } ) ( X _ { i } ) \right\} ^ { 2 } = O _ { P } ( n ^ { - 1 } ) ,
$$

which completes the proof.

## D Technical proof for results in appendix

## D.1 Proof of Theorem A.4

Proof of Theorem $A . 4 .$ Following Kallus (2022) and Li et al. (2023), for any decision rule $\phi \colon \mathcal { X }  \{ 0 , 1 \}$

$$
\operatorname* { m a x } _ { P \in { \mathcal { P } } } \operatorname { E r r } ( \phi ; P ) = \mathbb { E } [ \phi ( X ) \gamma ( X ) ] ,
$$

where $\gamma ( x ) = \operatorname* { m i n } \{ \mathbb { P } ( Y ( 1 ) = 0 | X = x ) , \mathbb { P } ( Y ( 0 ) = 1 | X = x ) \}$ solely relies on the observed distribution, and $\mathbb { E } [ \cdot ]$ is with respect to the observed distribution. On the other hand, the power is the expectation of $\phi ( X )$ under the observed covariate distribution. The optimization problem (with randomized policy) is

$$
\begin{array} { r l } { \underset { \phi : \mathcal { X } \to [ 0 , 1 ] } { \mathrm { m a x i m i z e } } } & { \mathbb { E } [ \phi ( X ) ] } \\ { \mathrm { s u b j e c t ~ t o } } & { \mathbb { E } [ \phi ( X ) \gamma ( X ) ] \le \alpha . } \end{array}
$$

If $\mathbb { E } [ \gamma ( X ) ] \leq \alpha _ { \mathrm { : } }$ , then $\phi _ { \mathrm { p o w e r } } ^ { * } = 1$ maximizes the power since its power is one.

Otherwise, suppose $\mathbb { E } [ \gamma ( X ) ] > \alpha$ . Consider any decision rule $\psi ( \cdot )$ obeying $\mathbb { E } [ \psi ( X ) \gamma ( X ) ] \leq \alpha ,$ , and write $\phi ^ { * } ( \cdot ) = \phi _ { \mathrm { p o w e r } } ^ { * } ( \cdot )$ for notational convenience. Note that

$$
\begin{array} { r l } & { \gamma ^ { * } \cdot \left( \mathbb { E } [ \psi ( X ) ] - \mathbb { E } [ \phi ^ { * } ( X ) ] \right) } \\ & { = \mathbb { E } \left[ \gamma ^ { * } \cdot ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) < \gamma ^ { * } \} \right] } \end{array}
$$

$$
+ \mathbb { E } { \left[ \gamma ^ { * } \cdot \left( \psi ( X ) - \phi ^ { * } ( X ) \right) \mathbb { 1 } \{ \gamma ( X ) > \gamma ^ { * } \} \right] } + \mathbb { E } { \left[ \gamma ^ { * } \cdot \left( \psi ( X ) - \phi ^ { * } ( X ) \right) \mathbb { 1 } \{ \gamma ( X ) = \gamma ^ { * } \} \right] } .
$$

The first term, as $\phi ^ { * } ( X ) = 1$ when $\gamma ( X ) < \gamma ^ { * }$ and hence $\psi ( X ) - \phi ^ { * } ( X ) \leq 0$ , obeys

$$
\mathbb { E } \left[ \gamma ^ { * } \cdot ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) < \gamma ^ { * } \} \right] \leq \mathbb { E } \left[ \gamma ( X ) ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) < \gamma ^ { * } \} \right] .
$$

Similar arguments yield

$$
\begin{array} { r l } & { \mathbb { E } \big [ \gamma ^ { * } \cdot ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) > \gamma ^ { * } \} \big ] \leq \mathbb { E } \big [ \gamma ( X ) ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) > \gamma ^ { * } \} \big ] , } \\ & { \mathbb { E } \big [ \gamma ^ { * } \cdot ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) = \gamma ^ { * } \} \big ] = \mathbb { E } \big [ \gamma ( X ) ( \psi ( X ) - \phi ^ { * } ( X ) ) \mathbb { 1 } \{ \gamma ( X ) = \gamma ^ { * } \} \big ] . } \end{array}
$$

Adding the three terms up, we know

$$
\gamma ^ { * } \cdot \left( \mathbb { E } [ \psi ( X ) ] - \mathbb { E } [ \phi ^ { * } ( X ) ] \right) \leq \mathbb { E } [ \gamma ( X ) \psi ( X ) ] - \mathbb { E } [ \gamma ( X ) \phi ^ { * } ( X ) ] \leq \alpha - \alpha = 0
$$

since $\psi ( X )$ is a feasible solution and the satefy constraint is binding for $\phi ^ { * }$ . Finally, as $\gamma ^ { * } \geq 0$ , this implies $\mathbb { E } [ { \psi } ( X ) ] \leq \mathbb { E } [ { \phi } ^ { * } ( X ) ]$ ]. The arbitrariness of ψ implies the optimality of $\phi ^ { * }$ □

## D.2 Proof of Theorem A.2

Proof of Theorem A.2. Let $\hat { a } ( x ) \in \{ 0 , 1 \}$ denote the arm selected by the inclusion rule, and define

$$
\overline { { { \gamma } } } _ { n } ( x ) : = \{ 1 - \mu _ { 1 } ( x ) \} { \bf 1 } \{ \hat { a } ( x ) = 1 \} + \mu _ { 0 } ( x ) { \bf 1 } \{ \hat { a } ( x ) = 0 \} .
$$

As in the proof of Theorem 4.2, the optimality of ${ \hat { a } } ( x )$ for the estimated proxy-label risks gives

$$
0 \leq \overline { { \gamma } } _ { n } ( x ) - \gamma ( x ) \leq 2 \left\{ \left| \hat { \mu } _ { 1 } ( x ) - \mu _ { 1 } ( x ) \right| + \left| \hat { \mu } _ { 0 } ( x ) - \mu _ { 0 } ( x ) \right| \right\} .
$$

Hence $\| \overline { { \gamma } } _ { n } - \gamma \| _ { L _ { 1 } ( P _ { X } ) } = o _ { \mathbb { P } } ( 1 )$ . The assumed outcome-model consistency also implies $\Vert \hat { \gamma } - \gamma \Vert _ { L _ { 1 } ( P _ { X } ) } = o _ { \mathbb { P } } \ l ( 1 )$

Conditional on the training process, let

$$
\rho _ { n } ( x ) : = e ( x ) \mathbf { 1 } \{ \hat { a } ( x ) = 1 \} + \{ 1 - e ( x ) \} \mathbf { 1 } \{ \hat { a } ( x ) = 0 \} , \qquad w _ { n } ( x ) : = \rho _ { n } ( x ) ^ { - 1 } .
$$

The weighted identities in the proof of Theorem 4.2 give $\mathbb { E } [ G w _ { n } ( X ) \mid X ] = 1$ and $\mathbb { E } \left[ G w _ { n } ( X ) \mathbf { 1 } \{ Y ^ { \dagger } = 0 \} \big | X \right] =$ $\overline { { \gamma } } _ { n } ( X )$ . It follows by the same uniform-law-of-large-numbers argument that the weighted empirical conformal curve converges uniformly to $H ( t ) : = \mathbb { E } [ \gamma ( X ) \mathbf { 1 } \{ \gamma ( X ) \leq t \} ]$ . Consequently,

$$
p _ { n + 1 } ^ { \mathrm { s t r - p o w e r } } - H \{ \gamma ( X _ { n + 1 } ) \} = o _ { \mathbb { P } } ( 1 ) .
$$

Since $\gamma ( X )$ has no point mass, the same threshold argument as in the proof of Theorem 4.2 gives

$$
\begin{array} { r } { \mathbb { P } \big \{ \hat { \pi } _ { \mathrm { s t r - p o w e r } } ( X _ { n + 1 } ) \neq \pi _ { \mathrm { p o w e r } } ^ { * } ( X _ { n + 1 } ) \big \}  0 , } \end{array}
$$

where the binding and nonbinding cases are characterized by Theorem A.4. Therefore, $| \mathbb { E } [ \hat { \pi } _ { \mathrm { s t r - p o w e r } } ( X _ { n + 1 } ) ] -$ Power $( \pi _ { \mathrm { p o w e r } } ^ { * } ; P ) |  0$ , which completes the proof. □

## D.3 Proof of Theorem A.3

Proof of Theorem A.3. Define

$$
H _ { n } ( t ) : = \mathbb { E } [ \widehat { \gamma } ( X ) \mathbb { 1 } \{ \widehat { \gamma } ( X ) \leq t \} | \mathcal { T } _ { n } ] , \qquad t _ { n } ^ { \circ } : = \operatorname* { s u p } \{ t \in \mathbb { R } : H _ { n } ( t ) \leq \alpha \} ,
$$

and

$$
H _ { 0 } ( t ) : = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ \gamma ( X ) \leq t \} ] .
$$

By $\| \hat { \gamma } - \gamma \| _ { L _ { 2 } ( \mathbb { P } _ { X } ) } = o _ { P } \big ( 1 \big )$ , the no-point-mass condition on $\gamma ( X )$ , and Lemma F.4, we have

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - H _ { 0 } ( t ) | = o _ { P } ( 1 ) .
$$

Together with the local-crossing condition in Assumption $\mathrm { A . 1 0 ( i i i ) }$ , this implies $t _ { n } ^ { \circ } - t _ { \mathrm { p o w e r } } ^ { \ast } = o _ { P } \big ( 1 \big )$ , where $t _ { \mathrm { p o w e r } } ^ { * } : = \operatorname* { s u p } \{ t \in \mathbb { R } : H _ { 0 } ( t ) \leq \alpha \}$ . Set $\hat { \omega } : = \mathcal { W } _ { n } ( \hat { \gamma } , \tilde { w } )$ and define

$$
F _ { n } ^ { \hat { \gamma } } ( t ) : = \frac { \mathbb { E } [ \hat { \omega } ( X ) \hat { g } ( X , T ) \hat { \gamma } ( X ) \mathbb { 1 } \{ \hat { \gamma } ( X ) \leq t \} | \mathcal { T } _ { n } ] } { \mathbb { E } [ \hat { \omega } ( X ) \hat { g } ( X , T ) | \mathcal { T } _ { n } ] } .
$$

By the defining population balance equations, we know $F _ { n } ^ { \hat { \gamma } } ( t _ { n } ^ { \circ } ) ~ = ~ H _ { n } ( t _ { n } ^ { \circ } ) ~ = ~ \alpha$ . Moreover, Assumption A.10(i)–(iii) implies that $F _ { n } ^ { \hat { \gamma } }$ crosses α at $t _ { n } ^ { \circ }$ with a slope bounded away from zero, with probability tending to one. Define

$$
\hat { F } _ { n } ^ { \mathrm { o b s } } ( t ) : = \frac { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } \mathbb { 1 } \{ Y _ { i } ^ { \dagger } = 0 \} \mathbb { 1 } \{ \hat { \gamma } ( X _ { i } ) \leq t \} } { \hat { w } _ { n + 1 } + \sum _ { i = 1 } ^ { n } G _ { i } \hat { w } _ { i } } ,
$$

so that $p _ { n + 1 } ^ { \mathrm { o b s } } ~ = ~ \hat { F } _ { n } ^ { \mathrm { o b s } } \{ \hat { \gamma } ( X _ { n + 1 } ) \}$ . By Lemma A.11, Lemmas C.1 and F.4, and $\begin{array} { r } { \frac { \hat { w } _ { n + 1 } } { \sum _ { i \in \mathbb { Z } _ { \mathrm { c a l i b } } } \hat { w } _ { i } } = O _ { P } ( n ^ { - 1 } ) } \end{array}$ we obtain the uniform convergence of the empirical curve $\hat { F } _ { n } ^ { \mathrm { o b s } } ( t )$ to its training-conditional population counterpart. Furthermore, by the definition of $g ^ { * }$ , we know

$$
\begin{array} { r l } & { \mathbb { E } [ g ^ { * } ( X , T ) \mathbf { 1 } \{ Y ^ { \dagger } = 0 \} | X ] } \\ & { = \mathbb { E } \big [ \mathbb { E } [ \mathbf { 1 } \{ Y ^ { \dagger } = 0 \} | X ] T \mathbf { 1 } \{ 1 - \mu _ { 1 } ( X ) \le \mu _ { 0 } ( X ) \} \big | X \big ] + \mathbb { E } \big [ \mathbb { E } [ \mathbf { 1 } \{ Y ^ { \dagger } = 0 \} | X ] ( 1 - T ) \mathbf { 1 } \{ 1 - \mu _ { 1 } ( X ) > \mu _ { 0 } ( X ) \} \big | X \big ] } \\ & { = \mathbb { E } \big [ ( 1 - \mu _ { 1 } ( X ) ) T \mathbf { 1 } \{ 1 - \mu _ { 1 } ( X ) \le \mu _ { 0 } ( X ) \} \big | X \big ] + \mathbb { E } \big [ \mu _ { 0 } ( X ) ( 1 - T ) \mathbf { 1 } \{ 1 - \mu _ { 1 } ( X ) > \mu _ { 0 } ( X ) \} \big | X \big ] } \\ & { = \mathbb { E } [ g ^ { * } ( X , T ) \gamma ( X ) \mid X ] } \end{array}
$$

Therefore,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { F } _ { n } ^ { \mathrm { o b s } } ( t ) - F _ { n } ^ { \hat { \gamma } } ( t ) | \leq o _ { P } ( 1 ) + C \| \hat { g } - g ^ { * } \| _ { L _ { 1 } ( \mathbb { P } _ { X , T } ) } + C \| \hat { \gamma } - \gamma \| _ { L _ { 1 } ( \mathbb { P } _ { X } ) } = o _ { P } ( 1 ) .
$$

Fix $0 < \varepsilon \le \eta$ . The local-crossing and uniform convergence results imply that, with probability tending to one,

$$
\begin{array} { r } { \widehat { \gamma } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } - \varepsilon \quad \Rightarrow \quad p _ { n + 1 } ^ { \mathrm { o b s } } \leq \alpha , \qquad \mathrm { w h e r e a s } \qquad \widehat { \gamma } ( X _ { n + 1 } ) \geq t _ { n } ^ { \circ } + \varepsilon \quad \Rightarrow \quad p _ { n + 1 } ^ { \mathrm { o b s } } > \alpha . } \end{array}
$$

Consequently,

$$
\begin{array} { r } { | \hat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) - \mathbb { 1 } \{ \hat { \gamma } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } \} | \leq \mathbb { 1 } \{ | \hat { \gamma } ( X _ { n + 1 } ) - t _ { n } ^ { \circ } | \leq \varepsilon \} + o _ { P } ( 1 ) . } \end{array}
$$

Assumption A.10(iv) and the arbitrariness of $\varepsilon > 0$ yield

$$
\mathbb { E } [  \hat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) - \mathbb { 1 } \{ \hat { \gamma } ( X _ { n + 1 } ) \leq t _ { n } ^ { \circ } \}  ]  0 .
$$

Finally, since $\hat { \gamma } \to \gamma$ in $L _ { 2 } ( \mathbb { P } _ { X } ) , t _ { n } ^ { \circ } \to t _ { \mathrm { p o w e r } } ^ { * }$ , and $\gamma ( X )$ has no point mass,

$$
\mathbb { E } [ | \hat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) - \mathbb { 1 } \{ \gamma ( X _ { n + 1 } ) \leq t _ { \mathrm { p o w e r } } ^ { * } \} | ]  0 .
$$

By Theorem A.1, we know $\pi _ { \mathrm { p o w e r } } ^ { * } ( x ) = \mathbb { 1 } \{ \gamma ( x ) \leq t _ { \mathrm { p o w e r } } ^ { * } \}$ . Hence

$$
\begin{array} { r } { \left| \mathbb { E } \big [ \hat { \pi } _ { \mathrm { o b s } } \big ( X _ { n + 1 } \big ) \big ] - \mathbb { E } \big [ \pi _ { \mathrm { p o w e r } } ^ { * } \big ( X _ { n + 1 } \big ) \big ] \right| \leq \mathbb { E } \left[ \big | \hat { \pi } _ { \mathrm { o b s } } \big ( X _ { n + 1 } \big ) - \pi _ { \mathrm { p o w e r } } ^ { * } \big ( X _ { n + 1 } \big ) \big | \right] \to 0 , } \end{array}
$$

which proves $\mathbb { E } [ \hat { \pi } _ { \mathrm { o b s } } ( X _ { n + 1 } ) ]  \mathrm { P o w e r } ( \pi _ { \mathrm { p o w e r } } ^ { * } ; \mathbb { P } )$

## D.4 Proof of Theorem 3.3

Proof of Theorem 3.3. The result follows directly from Theorem A.5. In particular, since $s _ { \mathrm { w e l f a r e } } ( X )$ has no point mass, $\mathbb { P } \{ s _ { \mathrm { w e l f a r e } } ( X ) = r ^ { * } \} = 0$ , so the randomization at the cutof in Theorem A.5 is immaterial. Therefore, the optimal randomized rule in Theorem A.5 reduces almost surely to $\pi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( x ) \leq$ $r ^ { * } \}$ . The binding/nonbinding characterizations follow from the two cases in Theorem A.5. □

Proof of Theorem A.5. We first characterize the worst-case safety constraint. For any $P ^ { \prime } \in \mathcal { P }$ , define

$$
f _ { P ^ { \prime } } ( x ) : = P ^ { \prime } \{ Y ( 1 ) = 0 , Y ( 0 ) = 1 \mid X = x \} .
$$

For a randomized policy $\phi ,$ whose randomization is independent of the potential outcomes conditional on $X ,$ its harm rate under $P ^ { \prime }$ is $\operatorname { \mathrm { ? r r } } ( \phi ; P ^ { \prime } ) = \mathbb { E } [ \phi ( X ) f _ { P ^ { \prime } } ( X ) ]$ ]. Since $f _ { P ^ { \prime } } ( x ) \leq P ^ { \prime } \{ Y ( 1 ) = 0 \mid X = x \} = 1 - \mu _ { 1 } ( x )$ and $f _ { P ^ { \prime } } ( x ) \leq P ^ { \prime } \{ Y ( 0 ) = 1 \mid X = x \} = \mu _ { 0 } ( x )$ , we have $f _ { P ^ { \prime } } ( x ) \leq \gamma ( x )$ for every $P ^ { \prime } \in \mathcal { P }$ . This upper bound is sharp. Indeed, for every x, consider the following conditional distribution of the potential outcomes:

$$
\begin{array} { r l } & { P ^ { \prime } \{ Y ( 1 ) = 0 , Y ( 0 ) = 1 \mid X = x \} = \gamma ( x ) , } \\ & { P ^ { \prime } \{ Y ( 1 ) = 0 , Y ( 0 ) = 0 \mid X = x \} = 1 - \mu _ { 1 } ( x ) - \gamma ( x ) , } \\ & { P ^ { \prime } \{ Y ( 1 ) = 1 , Y ( 0 ) = 1 \mid X = x \} = \mu _ { 0 } ( x ) - \gamma ( x ) , } \\ & { P ^ { \prime } \{ Y ( 1 ) = 1 , Y ( 0 ) = 0 \mid X = x \} = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x ) + \gamma ( x ) . } \end{array}
$$

All four probabilities are nonnegative. The first three nonnegativity claims follow directly from $\gamma ( x ) =$ min $\{ 1 - \mu _ { 1 } ( x ) , \mu _ { 0 } ( x ) \}$ , while the last follows from $\gamma ( x ) \geq \mu _ { 0 } ( x ) - \mu _ { 1 } ( x )$ . They sum to one and yield the conditional marginals $P ^ { \prime } \{ Y ( 1 ) = 1 \mid X = x \} = \mu _ { 1 } ( x )$ and $P ^ { \prime } \{ { \cal Y } ( 0 ) = 1 \mid { \cal X } = x \} = \mu _ { 0 } ( x )$ . Combining this conditional coupling with the original distribution of X and the same treatment-assignment mechanism therefore defines a distribution in $\mathcal { P } .$ Consequently, for every randomized policy $\phi ,$

$$
\operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } \mathrm { E r r } ( \phi ; P ^ { \prime } ) = \mathbb { E } [ \gamma ( X ) \phi ( X ) ] .
$$

Write $\tau ( x ) : = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ . The welfare of a randomized policy is

$$
\begin{array} { r l } & { \mathrm { W e l f a r e } ( \phi ; P ) = \mathbb { E } [ \mu _ { 1 } ( X ) \phi ( X ) + \mu _ { 0 } ( X ) \{ 1 - \phi ( X ) \} ] } \\ & { \qquad = \mathbb { E } [ \mu _ { 0 } ( X ) ] + \mathbb { E } [ \tau ( X ) \phi ( X ) ] . } \end{array}
$$

Thus, up to the constant $\mathbb { E } [ \mu _ { 0 } ( X ) ]$ , the randomized extension of (3.9) is equivalent to

$$
\operatorname* { m a x i m i z e } _ { \phi : \mathcal X \to [ 0 , 1 ] } \mathbb { E } [ \tau ( X ) \phi ( X ) ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathbb { E } [ \gamma ( X ) \phi ( X ) ] \leq \alpha .
$$

We next verify that the cutof and randomization probability in the theorem are well defined. Recall that $\begin{array} { r } { s _ { \mathrm { w e l f a r e } } ( x ) = - \frac { \tau ( x ) } { \gamma ( x ) } } \end{array}$ under the stated ratio conventions, and define, for $r \leq 0$

$$
H ( r ) : = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) < r \} ] .
$$

The function H is nondecreasing and left-continuous on $( - \infty , 0 ]$ . Indeed, if $r _ { k } \uparrow r .$ then $\mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) < r _ { k } \right\}$ increases pointwise to $\mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) < r \right\}$ , so the conclusion follows from the dominated convergence theorem. Moreover, $H ( r ) \to 0$ as $r  - \infty$ . To see this, if $s _ { \mathrm { w e l f a r e } } ( X )$ is finite, then $\mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) < r \}  0$ . If $s _ { \mathrm { w e l f a r e } } ( X ) = - \infty .$ , then necessarily $\gamma ( X ) = 0 , \operatorname { s o } \gamma ( X ) \mathbb { 1 } \{ s _ { \mathrm { w e l f a r e } } ( X ) < r \} = 0$ for every r. The dominated convergence theorem therefore gives the claim, and hence the set defining $r ^ { * }$ is nonempty. Suppose first that $H ( 0 ) > \alpha$ . Since $H ( r ) \uparrow H ( 0 )$ as $r \uparrow 0 ,$ , there exists some $r _ { 0 } < 0$ such that $H ( r _ { 0 } ) > \alpha$ . It follows that $r ^ { * } < 0$ . By the definition of $r ^ { * }$ , there exists a sequence $r _ { k } \uparrow r ^ { * }$ such that $H ( r _ { k } ) \le \alpha$ . The left-continuity of H therefore implies $H ( r ^ { * } ) \leq \alpha$ . On the other hand, the right limit of H at $r ^ { * }$ is

$$
\operatorname* { l i m } _ { r \downarrow r ^ { * } } H ( r ) = \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) \leq r ^ { * } \right\} ] = H ( r ^ { * } ) + \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) = r ^ { * } \right\} ] .
$$

This limit must be at least $\alpha .$ . Otherwise, there would exist some $r > r ^ { * }$ suficiently close to $r ^ { * }$ such that $H ( r ) < \alpha$ , contradicting the definition of $r ^ { * }$ as the supremum. Therefore,

$$
H ( r ^ { * } ) \leq \alpha \leq H ( r ^ { * } ) + \mathbb { E } [ \gamma ( X ) \mathbb { 1 } \left\{ s _ { \mathrm { w e l f a r e } } ( X ) = r ^ { * } \right\} ] .
$$

This establishes the existence of $\eta ^ { * } ~ \in ~ [ 0 , 1 ]$ satisfying the equality in the theorem. If the expectation multiplying $\eta ^ { * }$ is zero, the preceding inequalities imply $H ( r ^ { * } ) = \alpha .$ , and we may take $\eta ^ { * } = 0$ . If instead

$H ( 0 ) \le \alpha$ , the theorem sets $r ^ { * } = 0$ and $\eta ^ { * } = 0$ . Under the stated ratio conventions, $s _ { \mathrm { w e l f a r e } } ( x ) < 0$ if and only if $\tau ( x ) > 0$ . Hence, in this case,

$$
\phi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = \mathbb { 1 } \{ \tau ( x ) > 0 \} , \qquad \mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] = H ( 0 ) \leq \alpha .
$$

It follows in both cases that $\phi _ { \mathrm { w e l f a r e } } ^ { * }$ is feasible and satisfies $\left( - r ^ { * } \right) \left\{ \mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] - \alpha \right\} = 0$ by complementary slackness.

It remains to establish optimality. Let $\psi : \mathcal { X } \to [ 0 , 1 ]$ be any feasible randomized policy. By the definition of $\phi _ { \mathrm { w e l f a r e } } ^ { * } ,$ it holds pointwise that

$$
\{ \tau ( x ) + r ^ { * } \gamma ( x ) \} \{ \psi ( x ) - \phi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) \} \le 0 .
$$

Indeed, if $s _ { \mathrm { w e l f a r e } } ( x ) < r ^ { * }$ , then $\tau ( x ) + r ^ { * } \gamma ( x ) \geq 0$ and $\phi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = 1$ , so the second factor is nonpositive. $\mathrm { I f } \ s _ { \mathrm { w e l f a r e } } ( x ) \ > \ r ^ { * }$ , then $\tau ( x ) + r ^ { * } \gamma ( x ) \leq 0$ and $\phi _ { \mathrm { w e l f a r e } } ^ { * } ( x ) = 0$ , so the second factor is nonnegative. If $s _ { \mathrm { w e l f a r e } } ( x ) = r ^ { * }$ , the first factor is zero. The same conclusions continue to hold when $\gamma ( \boldsymbol { x } ) = 0$ under the stated ratio conventions. Consequently,

$$
\begin{array} { r l } & { \mathrm { W e l f a r e } ( \psi ; P ) - \mathrm { W e l f a r e } ( \phi _ { \mathrm { w e l f a r e } } ^ { * } ; P ) } \\ & { \quad = \mathbb { E } [ \tau ( X ) \{ \psi ( X ) - \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) \} ] } \\ & { \quad = \mathbb { E } [ \{ \tau ( X ) + r ^ { * } \gamma ( X ) \} \{ \psi ( X ) - \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) \} ] - r ^ { * } \mathbb { E } [ \gamma ( X ) \{ \psi ( X ) - \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) \} ] } \\ & { \quad \le - r ^ { * } \left\{ \mathbb { E } [ \gamma ( X ) \psi ( X ) ] - \mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] \right\} } \\ & { \quad \le - r ^ { * } \left\{ \alpha - \mathbb { E } [ \gamma ( X ) \phi _ { \mathrm { w e l f a r e } } ^ { * } ( X ) ] \right\} = 0 . } \end{array}
$$

The first inequality follows from the preceding pointwise inequality. The second follows from the feasibility of $\psi$ and the fact that $- r ^ { * } \geq 0$ . The final equality follows from complementary slackness. Since $\psi$ was arbitrary, this proves the optimality of $\phi _ { \mathrm { w e l f a r e } } ^ { * } .$ □

## E Simulation details

## E.1 Details for experiments in Figure 2

This section includes the omitted details that produce Figure 2.

Data generating process. Across all the four settings, we set the feature dimension as $X \in \mathbb { R } ^ { 2 0 }$ . In Settings 1-2, the linear coeficients are obtained by a random i.i.d. draw from Unif[(1, 2)] and fixed before running all the experiments.

In Setting 1, we draw $X \stackrel { \mathrm { i . i . d . } } { \sim } N ( 0 , { \bf I } )$ and set $\mu _ { 1 } ( x ) = \mathrm { l o g i t } ^ { - 1 } ( 0 . 0 8 3 5 x _ { 1 } + 0 . 0 9 7 x _ { 2 } + 0 . 0 8 8 5 x _ { 3 } + 0 . 0 5 7 5 x _ { 4 } +$ $0 . 0 6 7 5 x _ { 5 } + 1 )$ and $\mu _ { 0 } ( x ) = 1 - 0 . 8 \mu _ { 1 } ( x ) - 0 . 2 \log \mathrm { i t } ^ { - 1 } ( - 0 . 8 7 x _ { 4 } - 0 . 5 8 x _ { 5 } - 0 . 6 7 5 x _ { 6 } - 0 . 5 4 x _ { 7 } - 0 . 5 7 x _ { 8 } )$ . Both functions are then truncated to [0.05, 0.95].

In Setting 2, we draw $X \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \Sigma )$ with $\Sigma = 0 . 5 \Im \Im \Im ^ { \top } + 0 . 5 \mathbf { I }$ and set $\mu _ { 1 } ( x ) = \mathrm { l o g i t } ^ { - 1 } ( x ^ { \top } \theta _ { 1 } + 0 . 8 )$ where $[ \theta _ { 1 } ] _ { 1 : 1 0 } = ( 1 . 6 6 , 1 . 7 1 , 1 . 9 3 , 1 . 7 2 , 1 . 9 9 , 1 . 9 8 , 1 . 5 2 , 1 . 5 7 , 1 . 3 7 , 1 . 5 3 ) \times 0 . 0 2$ , and $\mu _ { 0 } ( x ) = 1 - 0 . 9 \mu _ { 1 } ( x ) -$ $0 . 4 \log \mathrm { i t } ^ { - 1 } ( x ^ { \top } \theta _ { 0 } )$ where $[ \theta _ { 0 } ] _ { 4 : 1 3 } = - ( 1 . 7 3 , 1 . 9 2 , 1 . 9 7 , 1 . 7 0 , 1 . 7 8 , 1 . 3 7 , 1 . 7 8 , 1 . 7 6 , 1 . 5 9 , 1 . 3 2 ) \times 0 . 2 ;$ ; both functions are then truncated to $[ 0 . 0 5 , 0 . 9 5 ]$ . Finally, we take $X _ { 1 : 7 }$ as the observed features.

In the nonlinear Setting 3, we draw entries of X i.i.d. from Unif(0, 1) and set $\mu _ { 1 } ( x ) = \mathrm { l o g i t } ^ { - 1 } ( f _ { 1 } ( x ) )$ ， where $f _ { 1 } ( x ) = \mathrm { s i g n a l } \cdot \{ \sin ( 3 \pi x _ { 1 } ) + \cos ( 2 \pi x _ { 2 } ) + x _ { 3 } ^ { 2 } - \mathbb { 1 } ( x _ { 4 } > 0 . 5 ) x _ { 4 } ^ { 2 } + 0 . 5 \ \mathbb { 1 } ( x _ { 5 } > 0 . 3 ) + e ^ { - x _ { 4 } } \} + 1 ;$ we then truncate $\mu _ { 1 } ( x )$ to [0.05, 0.95]. Next, we define $\mu _ { 0 } ( x ) = 1 - 0 . 2 \mu _ { 1 } ( x ) - 0 . 8 \mu _ { 1 } ( x ) ^ { 2 }$

In Setting 4, we draw X from the same correlated Gaussian design as in Setting 2, and set $\mu _ { 1 } ( x ) =$ $\log \mathrm { i t } ^ { - 1 } ( f _ { 1 } ( x ) )$ with $f _ { 1 } ( x ) = 0 . 2 \cdot \left\{ \sin ( 3 \pi x _ { 1 } ) + \cos ( 2 \pi x _ { 8 } ) + 0 . 5 x _ { 3 } ^ { 2 } - \mathbb { 1 } ( x _ { 4 } > 0 . 5 ) x _ { 5 } ^ { 2 } + 0 . 5 ~ \mathbb { 1 } ( x _ { 6 } > 0 . 3 ) + e ^ { - x _ { 7 } } \right\} + 0 . 5 ~ \mathbb { 1 } ( x _ { 8 } > 0 . 3 ) + 0 . 5 ~ \mathbb { 1 } ( x _ { 9 } > 0 . 3 ) + 0 . 5 ~ \mathbb { 1 } ( x _ { 1 } > 0 . 5 )$ $0 . 8 + 0 . 1 x _ { 8 }$ , truncated to [0.05, 0.95]. We then define $\mu _ { 0 } ( x ) = 1 - \mu _ { 1 } ( x ) + 0 . 1 \ \Im \left( x _ { 6 } > 0 . 5 \right) - 0 . 2 x _ { 8 } ^ { 2 } ,$ truncated to [0.08, 0.95]. Finally, after generating the outcomes and treatments, we take $X _ { 1 : 5 }$ as the observed features.

Additional implementation details. For Li et al., we implement the doubly-robust estimator $\hat { u } _ { \mathrm { F N A } } ( \pi )$ in Li et al. (2023, Lemma 6.1) with uniform cost $c ( X ) = 1$ , where the $\mu _ { t } ( \cdot )$ are fitted using the same random forest classifier in scikit-learn Python library with 2-fold cross-fitting. The policy class $\pi \in \Pi$ is depth-2 policy trees from the econml Python library. Given these estimators we fit the optimization problem (2) in Li et al. (2023) via the dual form and a bi-search to find the dual variable. The Policy-Tree baseline follows the standard use of econml library, where we use $\mathcal { D } _ { \mathrm { l a b e l } }$ to learn the policy and apply to $\mathcal { D } _ { \mathrm { t e s t } }$

## E.2 Details for additional experiments in Section 6.1

For all the additional experiments in Section 6.1, we sample $X \ \in \ [ 0 , 1 ] ^ { 2 0 } \ \overset { \mathrm { i . i . d . } } { \sim } \ \operatorname { U n i f } ( [ 0 , 1 ] ^ { 2 0 } )$ and assign treatment independently as $T \sim \mathrm { B e r n o u l l i } ( 1 / 2 )$ . We state the three sets of DGPs below.

Arm informativeness and selective calibration. In this set of experiments, we vary the scale of $\mu _ { 1 } ( x )$ and $\mu _ { 0 } ( x )$ to vary whether the harm rate relies on the treated or control outcome, i.e., whether $\rho ( x ) = 1 - \mu _ { 1 } ( x )$ or $\rho ( x ) = \mu _ { 0 } ( x )$ . Let $q \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ be a parameter that controls the comparison of the two outcome models. We define the latent score $r _ { \mathrm { a r m } } ( x ) = \sin ( \pi x _ { 1 } x _ { 2 } ) + x _ { 3 } - 0 . 5 x _ { 4 } + 0 . 3 ~ 1 \{ x _ { 5 } > 0 . 4 \}$ and $q _ { r } \in \mathbb { R }$ be the q-th population quantile of $r _ { \mathrm { a r m } } ( X )$ . For $r _ { \mathrm { a r m } } ( x ) \leq q _ { r }$ , we set

$$
\mu _ { 1 } ( x ) = 1 - \rho ( x ) , \qquad \mu _ { 0 } ( x ) = \rho ( x ) + \Delta ( x ) ,
$$

whereas for $r _ { \mathrm { a r m } } ( x ) > q _ { r }$ , we set

$$
\mu _ { 0 } ( x ) = \rho ( x ) , \qquad \mu _ { 1 } ( x ) = 1 - \rho ( x ) - \Delta ( x ) .
$$

Thus, the parameter q controls the fraction of units for which the treated arm is the informative arm. Here we use a nonlinear harm rate function $\rho ( x ) = 0 . 0 8 + 0 . 2 2$ expit $\{ \sin ( 2 \pi x _ { 1 } ) + 0 . 7 \cos ( 2 \pi x _ { 2 } ) + 0 . 5 ( x _ { 3 } - 0 . 5 ) ^ { 2 } -$ $0 . 6 \ 1 1 \{ x _ { 4 } > 0 . 6 \} + 0 . 5 e ^ { - x _ { 5 } } \}$ , and the treatment efect function $\Delta ( x ) = \{ 0 . 3 5 + 0 . 3 5 + 0 . 2 5 \mathrm { e x p i t } ( r _ { \delta } ( x ) ) \}$ max $\{ 1 - 2 \rho ( x ) - 0 . 0 2 , 0 . 0 2 \}$ , where $r _ { \delta } ( x ) = \cos ( 2 \pi x _ { 3 } ) - 0 . 6 \sin ( 2 \pi x _ { 5 } ) + 0 . 8 x _ { 6 } - 0 . 4 \Im \{ x _ { 7 } > 0 . 5 \}$ , and $\mathrm { e x p i t } ( u ) = 1 / ( 1 + e ^ { - u } )$

Subgroup heterogeneity. For this experiment we use a three-group construction. Let $\tilde { x } _ { 1 } = x _ { 1 } - 0 . 5$ and $\tilde { x } _ { 2 } = x _ { 2 } - 0 . 5$ , and define

$$
v ( x ) = \sigma ( \sin ( 2 \pi x _ { 3 } ) - 0 . 8 \cos ( 2 \pi x _ { 4 } ) + 0 . 6 x _ { 5 } ) .
$$

We partition the covariate space into three groups: the first group $G _ { 1 }$ consists of samples with $\tilde { x } _ { 1 } > 0 ;$ the second group $G _ { 2 }$ consists of samples with $\tilde { x } _ { 1 } \leq 0$ and $\tilde { x } _ { 2 } \leq 0 ;$ the third group $G _ { 3 }$ consists of samples with $\tilde { x } _ { 1 } \leq 0$ and $\tilde { x } _ { 2 } > 0$ . For $x \in G _ { 1 }$ , we set

$$
\mu _ { 0 } ( x ) = 0 . 3 5 , \qquad \mu _ { 1 } ( x ) = 0 . 6 5 .
$$

For $x \in G _ { 2 } \cup G _ { 3 }$ , we define a small heterogeneous treatment efect $\tau ( x ) = \operatorname* { m a x } \{ 0 . 0 0 4 , \operatorname* { m i n } \{ 0 . 0 3 , 0 . 0 0 8 +$ $0 . 0 1 1 2 5 \bigl ( 0 . 3 + 0 . 7 v ( x ) \bigr ) \cdot$ }. Then for $x \in G _ { 2 }$ , we set

$$
\mu _ { 1 } ( x ) = \operatorname* { m a x } \{ 0 . 9 4 , \operatorname* { m i n } \{ 0 . 9 9 5 , \ 0 . 9 5 + 0 . 0 2 v ( x ) \} \} , \qquad \mu _ { 0 } ( x ) = \operatorname* { m a x } \{ 0 . 9 , \ \operatorname* { m i n } \{ 0 . 9 9 , \ \mu _ { 1 } ( x ) - \tau ( x ) \} \} ,
$$

so both outcomes are near one and the treatment efect is small and positive. Note that in $G _ { 2 } .$ , the harm rate is small, and one recognizes this only when using the treated outcomes for calibration. For $x \in G _ { 3 }$ , we set

$$
\mu _ { 0 } ( x ) = \operatorname* { m a x } \{ 0 . 0 0 5 , \ \operatorname* { m i n } \{ 0 . 0 8 , \ 0 . 0 2 + 0 . 0 3 v ( x ) \} \} \qquad \mu _ { 1 } ( x ) = \operatorname* { m a x } \{ 0 . 0 0 1 , \ \operatorname* { m i n } \{ 0 . 0 6 , \ \mu _ { 0 } ( x ) - \tau ( x ) \} \} ,
$$

so both arms are near zero and the treatment efect is small and negative. In $G _ { 3 }$ , the harm rate is close to zero, and the method recognizes this only when using the control outcomes for calibration.

## E.3 Details for experiments in Section 6.2

Data generating process for Figure 5. In the four settings, the process that generates $( X , Y ( 1 ) , Y ( 0 ) )$ is the same as those in Figure 2 for RCT experiments. Given the features $\{ X _ { i } \}$ , we sample the treatments $\{ T _ { i } \}$ independently from Bernoulli(e(X )) according to a propensity score $e ( x ) = \mathbb { P } ( T = 1 \mid X = x ) =$ $\mathrm { l o g i t } ^ { - 1 } ( \eta _ { e } ( x ) )$ . In the linear settings 1-2, we set $\eta _ { e } ( x ) = 0 . 2 5 \cdot ( 1 . 1 x _ { 1 } + 0 . 9 x _ { 2 } + 0 . 6 x _ { 3 } - 0 . 4 x _ { 4 } + 0 . 5 x _ { 5 } + 0 . 3 x _ { 6 } )$ In the nonlinears 3-4, we set $\eta _ { e } ( x ) = 0 . 2 5 \cdot ( \sin ( 3 \pi x _ { 1 } ) + \cos ( 2 \pi x _ { 8 } ) + 0 . 1 x _ { 3 } ^ { 2 } - 0 . 3 x _ { 4 } )$

Data generating process for double robustness results. We generate covariates $X \in \mathbb { R } ^ { 2 0 }$ with entries i.i.d. from Unif(0, 1). We define three latent threshold indicators by $q _ { 1 } ( x ) = 1 \{ x _ { 1 } > 0 . 5 \} , q _ { 2 } ( x ) = 1 \{ x _ { 3 } >$ $0 . 5 \}$ , and $q _ { 3 } ( x ) = \Im \{ x _ { 5 } > 0 . 5 \}$ . These induce three hidden subgroups: $G _ { 1 } ( x ) = \mathbb { 1 } \{ q _ { 1 } ( x ) = 1 , q _ { 2 } ( x ) = 1 \}$ $G _ { 2 } ( x ) = \mathbb { 1 } \{ q _ { 1 } ( x ) = 1 , q _ { 2 } ( x ) = 0 \}$ , and $G _ { 3 } ( x ) = \mathbb { 1 } \{ q _ { 1 } ( x ) = 0 , q _ { 3 } ( x ) = 1 \}$ . The potential outcomes satisfy $Y ( t ) \mid X = x \sim$ Bernoull $\mathrm { i } ( \mu _ { t } ( x ) )$ for $t \in \{ 0 , 1 \}$ . We set $\mu _ { 0 } ( x ) = 0 . 0 7 + 0 . 0 2 x _ { 7 } + 0 . 0 1 ( x _ { 8 } - 0 . 5 ) + 0 . 3 8 G _ { 1 } ( x ) +$ $0 . 0 3 G _ { 3 } ( x )$ and $\tau ( x ) = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x ) = 0 . 0 3 + 0 . 0 1 ( x _ { 8 } - x _ { 7 } ) - 0 . 4 8 G _ { 1 } ( x ) + 0 . 4 2 G _ { 2 } ( x ) + 0 . 1 4 G _ { 3 } ( x )$ , and then truncate $\mu _ { 0 } ( x )$ to [0.03, 0.82] and set $\mu _ { 1 } ( x ) = \mu _ { 0 } ( x ) + \tau ( x )$ truncated to $[ \mu _ { 0 } ( x ) + 0 . 0 1 , 0 . 9 5 ]$

Treatment is assigned observationally according to the propensity score $e ( x ) = \mathrm { l o g i t } ^ { - 1 } ( - 1 . 2 + \beta G _ { 1 } ( x ) +$ $1 . 2 G _ { 2 } ( x ) + 0 . 6 G _ { 3 } ( x ) )$ ), truncated to [0.1, 0.9], and we sample $T \ | \ X \ = \ x \ \sim$ Bernoulli(e(x)). Here the parameter $\beta \in \{ 2 . 8 , 3 . 4 , 4 . 0 , 4 . 4 \}$ controls the strength of confounding in four levels shown in the x-axis of Figure 6. The observed outcome is $Y = T Y ( 1 ) + ( 1 - T ) Y ( 0 )$ . We generate $( Y ( 1 ) , Y ( 0 ) )$ under the negative coupling scheme.

Methods for double robustness results. To study double robustness, we compare five nuisance-model regimes. In both correct, both the propensity and outcome models are fit by logistic regression on the transformed features $\left( q _ { 1 } , q _ { 2 } , q _ { 3 } , G _ { 1 } , G _ { 2 } , G _ { 3 } , x _ { 7 } , x _ { 8 } \right)$ . In both wrong, both are fit by logistic regression on the raw covariates X. In ps correct only, the propensity model uses the transformed features while the outcome models use the raw covariates, and in outcome correct only the reverse is used. In the rf regime, both the propensity and outcome models are fit by random forests on the raw covariates. The other data splitting and model fitting details are the same as other experiments.

## E.4 Real data analysis details

For the real-data analysis, we define the binary outcome as $Y _ { i } = \mathbb { I } ( \mathrm { P o s t . B e l i e f . S p e c i f i c } _ { i } < 5 0 )$ and the treatment indicator as $T _ { i } = \mathbb { I } ( { \tt m e s s a g e T y p e } _ { i } = { \tt A c t i v e } )$ . Because the study is a randomized experiment, treatment assignment is known by design, so we do not estimate a propensity score model.

We split the sample in two stages. First, we reserve 20% of observations as the test set. We then split the remaining 80% evenly into a 40% training set and a 40% calibration set. All prediction models are fit on the training set only, and predictions are generated for the calibration and test sets.

The feature set includes demographic covariates (Education Cat, AgeYears, Race \*, Gender \*, religion), political and psychological covariates (Extremism, AOT, IH, Party \*), AI-related covariates (genai fam 1, genai use 1, genai trust, Sureness 1), and baseline state variables (Pre Belief Specific, LowConfidence) Missing values in the structured covariates are imputed using mean imputation. These variables are then standardized using scaling parameters estimated on the training set. To incorporate the text information in the stated conspiracy, we use a pre-computed embedding for the (pre-treatment) conspiracy for each observation. We apply principal components analysis (PCA) to the training-sample embeddings and retain the first 20 principal components. The final feature vector is the concatenation of the standardized structured covariates, the standardized dialogue-length variables, and 20 PCA components.

We estimate the conditional mean outcomes under treatment and control separately using the regression forest in econml Python package. Specifically, one regression forest is fit on the treated training subsample to estimate $\mu _ { 1 } ( x ) = \mathbb { E } [ Y \mid X = x , T = 1 ]$ , and a second regression forest is fit on the control training subsample to estimate $\mu _ { 0 } ( x ) = \mathbb { E } [ Y \mid X = x , T = 0 ]$ . These two forests produce predictions ${ \hat { \mu } } _ { 1 } ( x )$ and ${ \hat { \mu } } _ { 0 } ( x )$ on the calibration and test sets. To obtain a direct estimate of treatment heterogeneity, we additionally fit a causal forest via the econml package on the full training sample using the same features. This model produces a direct estimate of the conditional average treatment efect, $\hat { \tau } _ { \mathrm { C F } } ( x )$ , where $\tau ( x ) = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$

Our welfare-based selection rule combines the direct causal-forest estimate of treatment benefit with a conservative proxy for treatment risk. Define $\hat { \rho } ( x ) = \operatorname* { m i n } \{ 1 - \hat { \mu } _ { 1 } ( x ) , \hat { \mu } _ { 0 } ( x ) \}$ . We then define the welfare score as $S _ { \mathrm { w e l f a r e } } ( x ) = - \hat { \tau } _ { \mathrm { C F } } ( x ) / \operatorname* { m a x } \{ \hat { \rho } ( x ) , c \}$ , where $c > 0$ is a small numerical floor to avoid instability when ${ \hat { \rho } } ( x )$ is close to zero. In our implementation, we set $c = 0 . 0 2 5$

Our power-based selection rule directly uses the ${ \hat { \rho } } ( x )$ above in the score.

## F Auxiliary lemmas

Lemma F.1. Suppose the distribution of a random variable $X \in \mathbb { R }$ has no point mass. Then $\operatorname* { s u p } _ { t } \mathbb { P } ( t - \delta <$ $X \leq t )$ as a function of δ converges to zero as $\delta  0$

Proof of Lemma F.1. Since X has no point mass, the c.d.f. $F ( t ) : = \mathbb { P } ( X \leq t )$ is continuous and nondecreasing. Consider any constant $\epsilon > 0$ . There exists constants $a , b \in \mathbb { R }$ such that $F ( a ) \leq \epsilon$ and $1 - F ( b ) \leq \epsilon .$ Second, since F is continuous on the compact set $[ a , b ]$ , the Heine–Cantor theorem implies F is uniformly continuous on $[ a , b ]$ , which means there exists some $\delta > 0$ such that $\begin{array} { r } { \operatorname* { s u p } _ { x , y \in [ a , b ] , | x - y | \leq \delta } | F ( x ) - F ( y ) | \leq \epsilon } \end{array}$ Now we take any sequence

$$
a = t _ { 1 } < t _ { 2 } < \cdot \cdot \cdot < t _ { M } = b ,
$$

such that $t _ { i + 1 } - t _ { i } \leq \delta$ . By the arguments above, we know the following holds:

$$
\mathbb { P } ( t _ { i } < X \le t _ { i + 1 } ) \le \epsilon , \quad \mathbb { P } ( X \le t _ { 1 } ) \le \epsilon , \quad \mathbb { P } ( X > t _ { M } ) \le \epsilon .
$$

Therefore, for any $t \in [ t _ { 1 } + \delta , t _ { M } + \delta ]$ , there exists some i such that $\delta _ { i } \leq t - \delta < t \leq \delta _ { i + 2 }$ and thus $\mathbb { P } ( t - \delta < X \leq t ) \leq 2 \epsilon$ . For any $t \leq t _ { 1 } + \delta _ { \ast }$ , we know $\mathbb { P } ( t - \delta < X \leq t ) \leq \mathbb { P } ( X \leq t _ { 1 } + \delta )$ Therefore, for any $\epsilon > 0$ , we know s $\begin{array} { r } { \mathfrak { l p } _ { t } \mathbb { P } ( X \le t < \hat { s } ( X ) ) \le o _ { P } ( 1 ) + \epsilon } \end{array}$ . The arbitrariness of $\epsilon > 0$ the implies the desired result. □

Lemma F.2. Suppose two fixed functions $s _ { 1 } , s _ { 2 } \colon \mathcal { X }  \mathbb { R }$ obeys $\| s _ { 1 } ( X ) - s _ { 2 } ( X ) \| _ { L _ { 2 } } = o ( 1 )$ and one of them has no point mass. Then for any fixed $t \in \mathbb { R }$ , we have $\mathbb { P } ( s _ { 1 } ( X ) \leq t ) - \mathbb { P } ( s _ { 2 } ( X ) \leq t ) = o ( 1 )$

Proof of Lemma F.2. Without loss of generality we assume $s _ { 1 } ( X )$ has no point mass, so the mapping $t \mapsto$ $\mathbb { P } ( s _ { 1 } ( X ) \leq t )$ is continuous on $t \in \mathbb { R }$ . The $L _ { 2 }$ convergence implies the convergence in probability. That is, $\mathbb { P } ( | s _ { 1 } ( X ) - s _ { 2 } ( X ) | > \epsilon ) \to 0$ for any fixed $\epsilon > 0 .$ . Due to the continuity of $\mathbb { P } ( s _ { 1 } ( X ) \leq t ) \mathrm { ~ i n ~ } t \in \mathbb { R }$ and the above convergence, for any $\delta > 0$ , we can find a suficiently small $\epsilon > 0$ such that $\mathbb { P } ( s _ { 1 } ( X ) \leq t - \epsilon ) \geq \mathbb { P } ( s _ { 1 } ( X ) \leq t ) - \delta$ $\mathbb { P } ( s _ { 1 } ( X ) \leq t + \epsilon ) \leq \mathbb { P } ( s _ { 1 } ( X ) \leq t ) + \delta$ , and $\mathbb { P } ( | s _ { 1 } ( X ) - s _ { 2 } ( X ) | > \epsilon ) \le \delta$ . Therefore,

$$
\begin{array} { r l } & { \mathbb { P } ( s _ { 2 } ( X ) \leq t ) \leq \mathbb { P } ( s _ { 2 } ( X ) \leq t , | s _ { 1 } ( X ) - s _ { 2 } ( X ) | > \epsilon ) + \mathbb { P } ( s _ { 2 } ( X ) \leq t , | s _ { 1 } ( X ) - s _ { 2 } ( X ) | \leq \epsilon ) } \\ & { \qquad \leq \mathbb { P } ( | s _ { 1 } ( X ) - s _ { 2 } ( X ) | > \epsilon ) + \mathbb { P } ( s _ { 1 } ( X ) \leq t + \epsilon ) \leq \mathbb { P } ( s _ { 1 } ( X ) \leq t ) + 2 \delta . } \end{array}
$$

By the same arguments,

$$
\begin{array} { r } { \mathbb { P } ( s _ { 1 } ( X ) \leq t - \epsilon ) \leq \mathbb { P } ( | s _ { 1 } ( X ) - s _ { 2 } ( X ) | > \epsilon ) + \mathbb { P } ( s _ { 2 } ( X ) \leq t ) , } \end{array}
$$

which further implies

$$
\mathbb { P } ( s _ { 2 } ( X ) \leq t ) \geq \mathbb { P } ( s _ { 1 } ( X ) \leq t ) - 2 \delta .
$$

The arbitrariness of $\delta > 0$ thus implies the desired result.

Lemma F.3. Let $\mathcal { T } _ { n }$ be $\textit { a } \sigma { - } f i e l d$ representing a possibly random training process. Conditional on $\mathcal { T } _ { n }$ , let $\{ ( X _ { i } , Z _ { i } ) \} _ { i = 1 } ^ { n }$ be i.i.d. copies of $( X , Z )$ , where $Z \in \mathbb { R }$ , and let $s \colon \mathcal { X }  \mathbb { R }$ be fixed. Suppose $\mathbb { E } [ Z ^ { 2 } \mid { \mathcal { T } } _ { n } ] = O _ { P } ( 1 )$ Define

$$
H _ { n } ( t ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } \mathbb { 1 } \{ s ( X _ { i } ) \leq t \} , \qquad H ( t ) : = \mathbb { E } [ Z \mathbb { 1 } \{ s ( X ) \leq t \} | T _ { n } ] , \qquad \Delta _ { n } : = \operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - H ( t ) | .
$$

Then $\Delta _ { n } ~ = ~ O _ { P } ( n ^ { - 1 / 2 } )$ , where the stochastic order is with respect to both the training process and the evaluation sample. In addition, suppose that $Z \ge 0$ almost surely conditional on $\mathcal { T } _ { n }$ , and define

$$
\hat { \tau } : = \operatorname* { s u p } \bigg \{ t \in \mathbb { R } : \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } \mathbb { 1 } \{ s ( X _ { i } ) \leq t \} \leq \alpha \bigg \} , \qquad \tau ^ { * } : = \operatorname* { s u p } \big \{ t \in \mathbb { R } : \mathbb { E } [ Z \mathbb { 1 } \{ s ( X ) \leq t \} \ | \ T _ { n } ] \leq \alpha \big \} .
$$

Assume that there exist constants $c > 0$ and $\delta > 0$ such that, with probability tending to one over the training process,

$$
\begin{array} { r } { H ( \tau ^ { * } - u ) \leq \alpha - c u , \qquad H ( \tau ^ { * } + u ) \geq \alpha + c u } \end{array}
$$

for every $0 < u \le \delta$ . Then $\hat { \tau } - \tau ^ { * } = O _ { P } ( n ^ { - 1 / 2 } )$

Proof of Lemma F.3. For $t \in \mathbb { R }$ , set $g _ { t } ( x , z ) : = z \mathbb { 1 } \{ s ( x ) \leq t \}$ . Conditional on $\mathcal { T } _ { n }$ , the functions $g _ { t }$ are fixed and $( X _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ are i.i.d. By the conditional symmetrization inequality,

$$
\mathbb { E } [ \Delta _ { n } \mid \mathcal { T } _ { n } ] \leq \frac { 2 } { n } \mathbb { E } \left[ \operatorname* { s u p } _ { t \in \mathbb { R } } \left| \sum _ { i = 1 } ^ { n } \varepsilon _ { i } Z _ { i } \mathbb { 1 } \{ s ( X _ { i } ) \leq t \} \right| \Bigg | \mathcal { T } _ { n } \right] ,
$$

where $\varepsilon _ { 1 } , \ldots , \varepsilon _ { n }$ are i.i.d. Rademacher random variables independent of everything else.

Condition further on $( X _ { i } , Z _ { i } ) _ { i = 1 } ^ { n }$ , reorder the observations so that $s ( X _ { ( 1 ) } ) \leq \cdot \cdot \cdot \leq s ( X _ { ( n ) } )$ , and let $a _ { j } : = Z _ { ( j ) }$ . Since the sets $\{ i : s ( X _ { i } ) \leq t \}$ are nested,

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \left. \sum _ { i = 1 } ^ { n } \varepsilon _ { i } Z _ { i } \Im \left\{ s ( X _ { i } ) \leq t \right\} \right. \leq \operatorname* { m a x } _ { 0 \leq k \leq n } \left. \sum _ { j = 1 } ^ { k } \widetilde { \varepsilon } _ { j } a _ { j } \right. ,
$$

where $( \widetilde { \varepsilon } _ { j } ) _ { j = 1 } ^ { n }$ is again a Rademacher sequence. Setting $\begin{array} { r } { M _ { k } : = \sum _ { j = 1 } ^ { k } \widetilde { \varepsilon } _ { j } a _ { j } } \end{array}$ , Doob’s $L _ { 2 }$ maximal inequality gives

$$
\mathbb { E } _ { \varepsilon } [ \operatorname* { m a x } _ { 0 \leq k \leq n } | M _ { k } | ^ { 2 } | ( X _ { i } , Z _ { i } ) _ { i = 1 } ^ { n } , { \mathcal { T } } _ { n } ] \leq 4 \sum _ { j = 1 } ^ { n } a _ { j } ^ { 2 } .
$$

Consequently,

$$
\mathbb { E } [ \Delta _ { n } \mid \mathcal { T } _ { n } ] \leq \frac { 4 } { n } \mathbb { E } \left[ \left. \left( \sum _ { i = 1 } ^ { n } Z _ { i } ^ { 2 } \right) ^ { 1 / 2 } \right| \mathcal { T } _ { n } \right] \leq \frac { 4 } { \sqrt { n } } \left\{ \mathbb { E } [ Z ^ { 2 } \mid \mathcal { T } _ { n } ] \right\} ^ { 1 / 2 } .
$$

Let $V _ { n } : = \{ \mathbb { E } [ Z ^ { 2 } \mid { \mathcal { T } } _ { n } ] \} ^ { 1 / 2 }$ . For any $K , M > 0$ , conditional Markov’s inequality yields

$$
\mathbb { P } ( { \sqrt { n } } \Delta _ { n } > M ) \leq \mathbb { P } ( V _ { n } > K ) + { \frac { 4 K } { M } } .
$$

Since $V _ { n } = O _ { P } ( 1 )$ , this proves $\Delta _ { n } = O _ { P } ( n ^ { - 1 / 2 } )$

For the second claim, let ${ \mathcal { E } } _ { n }$ denote the event on which the local-slope condition holds, and define

$$
A _ { n } : = \mathcal { E } _ { n } \cap \left\{ \Delta _ { n } \leq \frac { c \delta } { 4 } \right\} .
$$

Then $\mathbb { P } ( A _ { n } ) \to 1$ . On $A _ { n } .$ let $u _ { n } : = 4 \Delta _ { n } / c .$ , so that $u _ { n } \leq \delta$ . If $\Delta _ { n } = 0$ , then $H _ { n } \equiv H$ and hence $\hat { \tau } = \tau ^ { * }$ Otherwise,

$$
H _ { n } ( \tau ^ { * } + u _ { n } ) \geq H ( \tau ^ { * } + u _ { n } ) - \Delta _ { n } \geq \alpha + c u _ { n } - \Delta _ { n } = \alpha + 3 \Delta _ { n } > \alpha ,
$$

whereas

$$
H _ { n } ( \tau ^ { * } - u _ { n } ) \leq H ( \tau ^ { * } - u _ { n } ) + \Delta _ { n } \leq \alpha - c u _ { n } + \Delta _ { n } = \alpha - 3 \Delta _ { n } < \alpha .
$$

Because $Z _ { i } \geq 0$ , the function $H _ { n }$ is nondecreasing. It follows that

$$
\begin{array} { r } { \tau ^ { * } - u _ { n } \leq \hat { \tau } \leq \tau ^ { * } + u _ { n } , } \end{array}
$$

and therefore $\lvert \hat { \tau } - \tau ^ { * } \rvert \leq 4 \Delta _ { n } / c$ on $A _ { n }$ . Since $\Delta _ { n } = O _ { P } ( n ^ { - 1 / 2 } )$ and $\mathbb { P } ( A _ { n } ) \to 1$ , we conclude that $\hat { \tau } - \tau ^ { * } =$ $O _ { P } ( n ^ { - 1 / 2 } )$ □

Lemma F.4. Consider a sequence of random functions ${ \hat { f } } _ { n } \colon { \mathcal { X } } \times { \mathcal { Z } } \to \mathbb { R } ^ { + }$ and $\hat { s } _ { n } \colon \mathcal { X }  \mathbb { R }$ obeying

$$
\begin{array} { r } { \| \hat { f } _ { n } - f \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } = o _ { P } ( 1 ) , \qquad \| \hat { s } _ { n } - s \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } = o _ { P } ( 1 ) , } \end{array}
$$

where $f \in L _ { 2 } ( \mathbb { P } _ { X , Z } )$ is nonnegative and the distribution $o f s ( X )$ has no point masses. Let $\{ ( X _ { i } , Z _ { i } ) \} _ { i = 1 } ^ { n }$ be i.i.d. samples from $\mathbb { P } _ { X , Z }$ and independent of the training processes of ${ \hat { f } } _ { n }$ and $\hat { s } _ { n }$ . Define

$$
\hat { H } _ { n } ( t ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { f } _ { n } ( X _ { i } , Z _ { i } ) \mathbb { 1 } \{ \hat { s } _ { n } ( X _ { i } ) \leq t \} , \quad H _ { n } ( t ) = \mathbb { E } \left[ \hat { f } _ { n } ( X , Z ) \mathbb { 1 } \{ \hat { s } _ { n } ( X ) \leq t \} \right] ,
$$

$$
\begin{array} { r } { \widetilde { H } _ { n } ( t ) = \mathbb { E } \left[ f ( X , Z ) \mathbb { 1 } \{ \hat { s } _ { n } ( X ) \leq t \} \right] , \quad H ( t ) = \mathbb { E } \left[ f ( X , Z ) \mathbb { 1 } \{ s ( X ) \leq t \} \right] , } \end{array}
$$

and

$$
\hat { \Delta } _ { n } = \operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { H } _ { n } ( t ) - H ( t ) | , \qquad \Delta _ { n } = \operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - H ( t ) | .
$$

Here, the expectations defining $H _ { n } , { \widetilde { H } } _ { n } .$ , and H are taken over an independent copy $( X , Z ) \sim \mathbb { P } _ { X , Z }$ . Then

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - \widetilde { H } _ { n } ( t ) | = o _ { P } ( 1 ) , \qquad \operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { H } _ { n } ( t ) - \widetilde { H } _ { n } ( t ) | = o _ { P } ( 1 ) ,
$$

and $\Delta _ { n } = o _ { P } ( 1 ) m \hat { \Delta } _ { n } = o _ { P } ( 1 )$ . In addition, suppose that $H ( t ^ { * } - \varepsilon ) < \alpha < H ( t ^ { * } + \varepsilon )$ holds for every $\varepsilon > 0$ where $\hat { t } = \operatorname* { s u p } \{ t \in \mathbb { R } : \hat { H } _ { n } ( t ) \leq \alpha \}$ and $t ^ { * } = \operatorname* { s u p } \{ t \in \mathbb { R } : H ( t ) \leq \alpha \}$ . Then

$$
\hat { t } - t ^ { * } = o _ { P } ( 1 ) .
$$

Proof of Lemma $F . 4 \cdot$ Write $\begin{array} { r } { P _ { n } g : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } g ( X _ { i } , Z _ { i } ) } \end{array}$ and ${ P g } : = \mathbb { E } [ g ( X , Z ) ]$ , and let $\mathcal { T } _ { n }$ be the $\sigma \cdot$ -field generated by the training processes of ${ \dot { f _ { n } } }$ and $\hat { s } _ { n } .$ Define

$$
A _ { n } : = \operatorname* { s u p } _ { t \in \mathbb { R } } \left| ( P _ { n } - P ) \big ( \widehat { f } _ { n } \mathbb { 1 } \{ \widehat { s } _ { n } \leq t \} \big ) \right| , \quad C _ { n } : = \operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - \widetilde { H } _ { n } ( t ) | , \quad D _ { n } : = \operatorname* { s u p } _ { t \in \mathbb { R } } | \widetilde { H } _ { n } ( t ) - H ( t ) | .
$$

Then $\Delta _ { n } \leq C _ { n } + D _ { n } , \hat { \Delta } _ { n } \leq A _ { n } + C _ { n } + D _ { n }$ , and

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { H } _ { n } ( t ) - \widetilde { H } _ { n } ( t ) | \leq A _ { n } + C _ { n } .
$$

We first control $A _ { n }$ . Conditional on $\mathcal { T } _ { n } ,$ the functions $\hat { f } _ { n }$ and $\hat { s } _ { n }$ are fixed and the evaluation observations are i.i.d. By conditional symmetrization,

$$
\mathbb { E } [ A _ { n } \mid \mathcal { T } _ { n } ] \leq \frac { 2 } { n } \mathbb { E } [ \operatorname* { s u p } _ { t \in \mathbb { R } } | \sum _ { i = 1 } ^ { n } \varepsilon _ { i } \hat { f } _ { n } ( X _ { i } , Z _ { i } ) \mathbb { 1 } \{ \hat { s } _ { n } ( X _ { i } ) \leq t \} | | \mathcal { T } _ { n } ] ,
$$

where $\varepsilon _ { 1 } , \ldots , \varepsilon _ { n }$ are i.i.d. Rademacher random variables independent of everything else. Conditional further on the evaluation sample, reorder the observations so that ${ \hat { s } } _ { n } ( X _ { ( 1 ) } ) ~ \leq ~ \cdot \cdot \cdot \leq { \hat { s } } _ { n } ( X _ { ( n ) } )$ and set $a _ { j } : = \hat { f } _ { n } ( X _ { ( j ) } , Z _ { ( j ) } )$ . Since the sets $\{ i : \hat { s } _ { n } ( X _ { i } ) \leq t \}$ are nested, they are prefixes of this ordering. Thus, by Doob’s $L _ { 2 }$ maximal inequality and Jensen’s inequality,

$$
\mathbb { E } _ { \varepsilon } [ \operatorname* { s u p } _ { t \in \mathbb { R } } | \sum _ { i = 1 } ^ { n } \varepsilon _ { i } { \hat { f } } _ { n } ( X _ { i } , Z _ { i } ) \mathbb { 1 } \{ { \hat { s } } _ { n } ( X _ { i } ) \leq t \} | | ( X _ { i } , Z _ { i } ) _ { i = 1 } ^ { n } , { \mathcal { T } } _ { n } ] \leq 2 ( \sum _ { j = 1 } ^ { n } a _ { j } ^ { 2 } ) ^ { 1 / 2 } .
$$

Consequently,

$$
\mathbb { E } [ A _ { n } \mid { \mathcal { T } } _ { n } ] \leq { \frac { 4 } { n } } \mathbb { E } \left[ \left. \left( \sum _ { i = 1 } ^ { n } { \hat { f } } _ { n } ( X _ { i } , Z _ { i } ) ^ { 2 } \right) ^ { 1 / 2 } \right| { \mathcal { T } } _ { n } \right] \leq { \frac { 4 } { \sqrt { n } } } \| { \hat { f } } _ { n } \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } .
$$

Since

$$
\| \hat { f } _ { n } \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } \leq \| f \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } + \| \hat { f } _ { n } - f \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } = O _ { P } ( 1 ) ,
$$

conditional Markov’s inequality gives $A _ { n } = O _ { P } ( n ^ { - 1 / 2 } ) = o _ { P } ( 1 )$

Next,

$$
C _ { n } = \operatorname* { s u p } _ { t \in \mathbb { R } } \Big | P \big ( ( \widehat { f } _ { n } - f ) \mathbb { 1 } \{ \widehat { s } _ { n } \leq t \} \big ) \Big | \leq P | \widehat { f } _ { n } - f | \leq \| \widehat { f } _ { n } - f \| _ { L _ { 2 } ( \mathbb { P } _ { x , z } ) } = o _ { P } ( 1 ) .
$$

It remains to control $D _ { n }$ . For each $t \in \mathbb { R }$ , let

$$
E _ { n , t } : = \left\{ \mathbb { 1 } \left\{ { \hat { s } } _ { n } ( X ) \leq t \right\} \neq \mathbb { 1 } \left\{ s ( X ) \leq t \right\} \right\} .
$$

By the Cauchy–Schwarz inequality,

$$
D _ { n } \leq \| f \| _ { L _ { 2 } ( \mathbb { P } _ { X , Z } ) } \left( \operatorname* { s u p } _ { t \in \mathbb { R } } P ( E _ { n , t } ) \right) ^ { 1 / 2 } .
$$

For any $\eta > 0 .$

$$
E _ { n , t } \subseteq \{ | s ( X ) - t | \leq \eta \} \cup \{ | \hat { s } _ { n } ( X ) - s ( X ) | > \eta \} ,
$$

and therefore

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } P ( E _ { n , t } ) \leq \omega ( \eta ) + P ( | \hat { s } _ { n } ( X ) - s ( X ) | > \eta ) \leq \omega ( \eta ) + \eta ^ { - 2 } \| \hat { s } _ { n } - s \| _ { L _ { 2 } ( \mathbb { R } _ { X , z } ) } ^ { 2 } ,
$$

where $\begin{array} { r } { \omega ( \eta ) : = \operatorname* { s u p } _ { t \in \mathbb { R } } P ( | s ( X ) - t | \leq \eta ) } \end{array}$ . Since the distribution of $s ( X )$ has no point masses, its distribution function is continuous and hence uniformly continuous, which implies $\omega ( \eta ) \to 0$ as $\eta \downarrow 0$ . For every fixed $\eta > 0$ , the second term is $o _ { P } ( 1 )$ . Letting first $n \to \infty$ and then $\eta \downarrow 0$ yields su $\mathrm { p } _ { t } P ( E _ { n , t } ) = o _ { P } ( 1 )$ , and hence $D _ { n } = o _ { P } ( 1 )$

We have thus shown

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | H _ { n } ( t ) - \widetilde { H } _ { n } ( t ) | = C _ { n } = o _ { P } ( 1 )
$$

and

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } | \hat { H } _ { n } ( t ) - \widetilde { H } _ { n } ( t ) | \leq A _ { n } + C _ { n } = o _ { P } ( 1 ) .
$$

Moreover,

$$
\Delta _ { n } \leq C _ { n } + D _ { n } = o _ { P } ( 1 ) , \qquad \hat { \Delta } _ { n } \leq A _ { n } + C _ { n } + D _ { n } = o _ { P } ( 1 ) .
$$

Finally, fix $\varepsilon > 0$ and define

$$
\gamma _ { \varepsilon } : = \frac { 1 } { 2 } \operatorname* { m i n } \left\{ \alpha - H \left( t ^ { * } - \frac { \varepsilon } { 2 } \right) , H \left( t ^ { * } + \frac { \varepsilon } { 2 } \right) - \alpha \right\} > 0 .
$$

On the event $\{ \hat { \Delta } _ { n } < \gamma _ { \varepsilon } \}$

$$
\hat { H } _ { n } \left( t ^ { * } - \frac { \varepsilon } { 2 } \right) \leq H \left( t ^ { * } - \frac { \varepsilon } { 2 } \right) + \hat { \Delta } _ { n } < \alpha
$$

and

$$
\hat { H } _ { n } \left( t ^ { * } + \frac { \varepsilon } { 2 } \right) \geq H \left( t ^ { * } + \frac { \varepsilon } { 2 } \right) - \hat { \Delta } _ { n } > \alpha .
$$

Since $\hat { f } _ { n } \geq 0$ , the function $\hat { H } _ { n }$ is nondecreasing. Hence

$$
t ^ { \ast } - \frac { \varepsilon } { 2 } \leq \hat { t } \leq t ^ { \ast } + \frac { \varepsilon } { 2 } ,
$$

so

$$
\begin{array} { r } { \mathbb { P } ( | \hat { t } - t ^ { * } | \geq \varepsilon ) \leq \mathbb { P } ( \hat { \Delta } _ { n } \geq \gamma _ { \varepsilon } )  0 . } \end{array}
$$

Therefore, $\hat { t } - t ^ { * } = o _ { P } ( 1 )$

## F.1 Proof of Lemma C.1

Proof of Lemma C.1. By the Cauchy-Schwarz inequality, it holds deterministically for any $t \in \mathbb { R }$ that

$$
\begin{array} { r l } & { \displaystyle \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { w } _ { i } Z _ { i } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} - \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { w } ( X _ { i } ) Z _ { i } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} \right| } \\ & { \displaystyle = \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } ( \hat { w } _ { i } - \hat { w } ( X _ { i } ) ) \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} \right| } \\ & { \displaystyle \leq \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } ^ { 2 } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} } \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \hat { w } _ { i } - \hat { w } ( X _ { i } ) ) ^ { 2 } } \leq \sqrt { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } ^ { 2 } } \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \hat { w } _ { i } - \hat { w } ( X _ { i } ) ) ^ { 2 } } . } \end{array}
$$

Since $\mathbb { E } [ Z _ { i } ^ { 2 } ] < \infty$ , we know $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } ^ { 2 } = O _ { P } ( 1 ) } \end{array}$ by the Markov’s inequali ${ \mathrm { , y , } }$ and therefore

$$
\operatorname* { s u p } _ { t \in \mathbb { R } } \bigg | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { w } _ { i } Z _ { i } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \hat { w } ( X _ { i } ) Z _ { i } \mathbb { 1 } \{ f ( X _ { i } ) \leq t \} \bigg | \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } ^ { 2 } } \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \hat { w } _ { i } - \hat { w } ( X _ { i } ) ) ^ { 2 } } = O _ { P } ( r _ { n } )
$$

by Assumption A.6.