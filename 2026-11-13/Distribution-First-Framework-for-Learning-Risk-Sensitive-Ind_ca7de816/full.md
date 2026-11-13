# Distribution-First Framework for Learning Risk-Sensitive Individualized Treatment Rules

Zhen Fang Department of Statistics and Operations Research University of North Carolina at Chapel Hill zhenfang@unc.edu

Guanting Chen Department of Statistics and Operations Research University of North Carolina at Chapel Hill guanting@unc.edu

Yufeng Liu Department of Statistics University of Michigan yufliu@umich.edu

## Abstract

Learning a risk-sensitive individualized treatment rule (ITR) with patient covariates is challenging when the treatment criterion depends on the full conditional outcome distribution rather than only its conditional mean. Existing methods are largely criterion-specific, making it difficult to compare multiple treatment criteria within a common pipeline, such as quantiles, conditional value-at-risk (CVaR), and complex preference-based objectives such as cumulative prospect theory. We propose a distribution-first framework for risk-sensitive ITR learning that first estimates treatment-specific conditional outcome distributions and then constructs the target ITR by plug-in criterion evaluation. The proposed framework unifies treatment rule learning across multiple criteria and accommodates hard-to-optimize objectives within the same estimation pipeline. On the theoretical side, it yields a general error propagation result for locally Hölder-stable criteria, linking first-stage distribution estimation error to downstream criterion estimation and treatment regret, together with sharper regret bounds under standard margin conditions. Empirically, we show that the proposed framework outperforms direct criterion-specific methods in multiple settings and further evaluate it on the ACTG175 HIV clinical trial, a randomized study comparing several antiretroviral regimens in HIV-infected adults.

## 1 Introduction

Personalized decision making is a central goal in precision medicine, aiming to move beyond the traditional “one size fits all” paradigm and tailor treatment decisions to patient-specific characteristics [1]. A core statistical approach to this problem is learning an individualized treatment rule (ITR), which maps patient covariates to treatment recommendations to optimize a clinically meaningful objective. Over the past two decades, a rich literature has developed around optimal ITR learning, including regression-based and value-based approaches that seek treatment rules with the best expected outcome, leading to substantial progress in individualized treatment learning [2–6].

In many treatment decisions, however, the conditional mean alone does not fully reflect what matters clinically. Two treatments may have similar expected outcomes but substantially different lower-tail behavior, variability, or asymmetry, leading to different risk–benefit profiles for different patients. This limitation has motivated growing interest in risk-sensitive treatment criteria that depend on distributional features beyond the conditional mean, such as conditional quantiles and conditional value-at-risk (CVaR), which are better suited to capture downside risk and tail behavior [7–9]. More broadly, complex risk-sensitive criteria like Cumulative Prospect Theory (CPT) have also been recognized as meaningful in health-related decision making [10, 11]. Originating in economics, CPT combines a value function with probability distortion to represent asymmetric preferences over gains and losses together with nonlinear attitudes toward unlikely events [12].

At the same time, existing risk-sensitive ITR methods face two related challenges. First, they are typically criterion-specific: a method designed for one treatment criterion, such as a conditional quantile or CVaR objective, generally does not transfer directly to another. As a result, adapting treatment recommendations to different patient preferences may require redesigning and retraining a separate learning procedure for each new criterion, which can substantially increase modeling and computational cost in practice. Second, direct criterion-specific learning becomes especially difficult for more complex risk-sensitive objectives. This is particularly clear for CPT-type criteria, whose functional form involves nonlinear transformations of cumulative probabilities rather than a simple expectation of a pointwise loss. Related evidence also appears in other decision-learning settings, such as reinforcement learning and agents [13–16]. Taken together, these observations suggest that the challenge is broader than any single criterion and arises more generally when treatment learning is tied to a specific risk-sensitive objective.

Problem description. These considerations motivate the main question of this paper: how can we learn risk-sensitive ITRs in a unified andflexible way when the relevant treatment criterion depends on the full conditional outcome distribution rather than only its mean? The challenge is not only to optimize a single pre-specified risk-sensitive criterion, but to develop a general approach that can support multiple treatment criteria and remain applicable when the target objective is difficult to optimize directly. Such a formulation would make it possible to compare treatment rules under different risk-sensitive objectives and to accommodate more complex criteria beyond the standard conditional mean-based setting.

Related work and our method. Our goal is to develop a unified framework for risk-sensitive ITR learning under the potential outcome framework. A natural starting point is that many treatment criteria of interest can be written as functionals of the conditional outcome distribution given patient covariates. This suggests a distribution-first strategy: first estimate the treatment-specific conditional outcome distribution, then obtain the target criterion by plug-in evaluation, and finally construct the treatment rule by comparing criterion values across treatments. This perspective is especially appealing when one wishes to compare treatment rules under different criteria or handle objectives that are difficult to optimize directly.

This view is supported by a broad literature on conditional distribution estimation. Classical approaches include kernel-based conditional distribution estimation [17], while more recent work has developed random forest-based methods for conditional quantile and distribution learning [18–20]. There is also growing interest in modern generative approaches to conditional sampling and distribution learning, including deep generative methods such as GCDS [21] and related neural density estimators [22]. Diffusion-based generative models, such as denoising diffusion probabilistic models (DDPMs) [23], offer another flexible approach to distribution learning. Recent work has further extended diffusion models to causal inference, such as DiffPO [24], which models potential outcome distributions while addressing treatment selection bias. Together, these methods provide increasingly rich options for estimating treatment-specific conditional outcome distributions.

To the best of our knowledge, unlike criterion-specific methods [2–6], there has been no work that explicitly proposes a unified framework for risk-sensitive ITR learning. The work closest to ours appears in offline policy assessment, where Huang et al. [25] studied plug-in estimation of Lipschitz criteria by first estimating the target policy’s outcome CDF and then analyzing how CDF estimation error propagates to criterion estimation error. However, their setting differs from ours in several aspects. First, they estimate the policy-induced marginal outcome distribution for policy assessment, whereas we focus on treatment- and covariate-specific conditional outcome distributions. Second, our goal is treatment recommendation, which requires treatment rule comparison and regret analysis rather than criterion estimation alone. Our work builds on this perspective but shifts the focus from criterion estimation to treatment recommendation. Another related work is Näf et al. [26], which develops a distributional forest-based estimator of conditional treatment effects. Its focus is distributional treatment effect estimation and inference rather than criterion-specific treatment rule construction.

Contributions. Our main contributions are as follows. (1) We formulate a distribution-first framework for risk-sensitive ITR learning to provide a unified way to learn treatment rules under multiple criteria and accommodate complex objectives that are difficult to handle by direct criterion-specific optimization. (2) We establish a theoretical analysis for this framework by linking the first-stage distribution estimation error to downstream criterion estimation and treatment regret in the ITR setting. (3) We empirically show that the proposed framework outperforms direct criterion-specific methods in multiple settings, and we further evaluate the framework on a real data application to illustrate a complete pipeline.

## 2 Distribution-First Risk-Sensitive ITR Learning

We observe i.i.d. samples $\{ ( X _ { i } , A _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , where $X _ { i } \in \mathcal { X } \subseteq \mathbb { R } ^ { p }$ is a vector of covariates, $A _ { i } \in$ $\mathcal { A } = \{ - 1 , + 1 \}$ is the treatment assignment, and $Y _ { i } \in \mathbb { R }$ is the observed outcome, with larger values preferred. Let $\operatorname { T } _ { i } ( a )$ denote the potential outcome under treatment $a \in { \mathcal { A } }$ . We make the following standard identification assumptions[27].

Assumption 1 (Identification). For each $a \in \mathcal { A } \colon ( i )$ consistency, $Y = Y ( A ) ; ( i i )$ unconfoundedness, $Y ( a ) { \bar { \perp } } A \mid X ;$ and (iii) positivity, there exists $\eta > 0$ such that $\eta \ \leq \ \mathbb { P } ( A = a \ | \ X = x ) \ \leq$ $1 - \eta , \forall x \in \mathcal { X } .$

Under Assumption 1, the treatment-specific conditional distribution of the potential outcome is identified by ${ \hat { F } } _ { a } ( y \mid x ) = \mathbb { P } \{ Y ( a ) \leq y \mid X = x \} = \mathbb { P } \{ Y \leq y \mid X = x , { \hat { A } } = a \}$ . Let $\tau$ be a real-valued criterion functional on distributions, with larger values preferred. We define the treatmentspecific score $Q _ { a } ^ { T } ( x ) = { \underline { { T } } } \{ F _ { a } ( \cdot \mid x ) \}$ , and the corresponding optimal individualized treatment rule $\pi \tau ( x ) \in$ arg $\mathrm { n a x } _ { a \in \mathcal { A } } Q _ { a } ^ { T } ( x )$ . This formulation makes clear that the target of learning is the full treatment-specific conditional distribution, rather than only the conditional mean, whenever treatment choice depends on distributional features beyond expectation.

Our distribution-first framework proceeds in two stages shown in Algorithm 1. In Stage 1, we estimate the treatment-specific conditional distributions $F _ { a } ( \cdot \mid x )$ for $a \in A .$ Let ${ \widehat { F } } _ { a } ( \cdot \mid x )$ denote a generic first-stage estimator. The framework is agnostic to the particular conditional distribution learner; in our experiments, we consider kernel-based estimators, random forest-based estimators, and modern generative models, which are discussed in Section 2.2. In Stage 2, we evaluate the target treatment criterion by plug-in: $\widehat { Q } _ { a } ^ { \mathcal { T } } ( x ) = \mathcal { T } \{ \widehat { F } _ { a } ( { \cdot } \mid x ) \}$ , and return the estimated treatment rule $\widehat { \pi } _ { \mathcal { T } } ( x ) \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \widehat { Q } _ { a } ^ { \mathcal { T } } ( x )$ . This modular design allows the same first-stage estimate to support multiple downstream criteria without redesigning the learning procedure for each objective.

Algorithm 1 Distribution-First Risk-Sensitive ITR Learning   
Require: Data $\{ ( X _ { i } , A _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , a criterion functional $\tau ,$ and a first-stage conditional distribution   
learner   
1: Estimate treatment-specific conditional distribution ${ \widehat { F } } _ { a } ( \cdot \mid x )$ for each $a \in { \mathcal { A } }$   
2: Compute plug-in scores ${ \widehat { Q } } _ { a } ^ { T } ( x ) = { \mathcal { T } } \{ { \widehat { F } } _ { a } ( { \cdot } \mid x ) \}$   
3: Return $\widehat { \pi } _ { \mathcal { T } } ( x ) \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \widehat { Q } _ { a } ^ { \mathcal { T } } ( x )$

## 2.1 Treatment criteria

A main advantage of the proposed framework is that different treatment criteria can be evaluated from the same estimated conditional distribution. These criteria correspond to different notions of utility, risk sensitivity, and personal preference. In this paper, we focus on four representative criteria: the conditional mean, the conditional quantile, CVaR, and CPT.

Conditional mean. The mean criterion is defined by $\mathcal { T } _ { \mathrm { m e a n } } \{ F _ { a } ( \cdot  { \mid } x ) \} = \mathbb { E } \{ Y ( a ) \ \mid \ X = x \}$ This is the standard objective in much of the classical ITR literature. It corresponds to risk-neutral decision making, in the sense that treatments are compared solely through their average outcomes. While natural and interpretable, the conditional mean criterion ignores variability, tail behavior, and asymmetry of the outcome distribution.

Conditional quantile. For $\tau \in ( 0 , 1 )$ , let $Q _ { \tau } \{ Y ( a ) \mid X = x \} = \operatorname* { i n f } \{ y : F _ { a } ( y \mid x ) \geq \tau \}$ denote the conditional τ-quantile under treatment a. The corresponding criterion is $\mathcal { T } _ { \mathrm { Q } , \tau } \{ F _ { a } ( \cdot \ | \ x ) \} =$

$Q _ { \tau } \{ Y ( a ) \mid X = x \}$ . Quantile-based decision making focuses on a specific location of the outcome distribution rather than its average. When τ is small, the criterion emphasizes lower-tail performance and yields a more conservative treatment rule.

Conditional value-at-risk (CVaR). For $\alpha \in ( 0 , 1 )$ , we define the lower-tail CVaR by $\mathrm { C V a R } _ { \alpha } \{ Y ( a ) \ |$ $\begin{array} { r } { X \ = \ x \} \ = \ \frac { 1 } { \alpha } \int _ { 0 } ^ { \alpha } Q _ { u } \{ Y ( a ) \ | \ X \ = \ x \} d u } \end{array}$ . The associated criterion is $\mathcal { T } _ { \mathrm { C V a R } , \alpha } \{ F _ { a } ( \cdot \ | \ x ) \} =$ $\mathrm { C V a R } _ { \alpha } \{ Y ( a ) \vert X = x \}$ . CVaR summarizes the average outcome over the worst α fraction of the conditional distribution. Compared with a single lower quantile, it captures the overall severity of adverse outcomes in the lower tail and is therefore a more informative measure of downside risk. In personalized treatment learning, CVaR is particularly relevant when severe but relatively rare unfavorable outcomes must be explicitly controlled.

Cumulative prospect theory (CPT). Unlike the previous criteria, CPT is designed to encode personal preferences under risk, including asymmetric attitudes toward gains and losses and nonlinear perception of probabilities. Let $r ( x )$ be a reference point and define the centered outcome $Z ( a ) =$ $\hat { Y } ( a ) \dot { - } r ( x )$ . We use the value function

$$
\begin{array} { r } { u ( z ) = \left\{ \begin{array} { l l } { u ^ { + } ( z ) = z ^ { \eta _ { + } } , } & { z \geq 0 , } \\ { - u ^ { - } ( - z ) = - \lambda ( - z ) ^ { \eta _ { - } } , } & { z < 0 , } \end{array} \right. } \end{array}
$$

where $\lambda > 1$ captures loss aversion, and $\eta _ { + } , \eta _ { - } \in ( 0 , 1 ]$ control diminishing sensitivity over gains and losses. Let $G _ { a } ( \cdot \mid x )$ denote the conditional distribution function of ${ \bar { Z } } ( a ) \mid X { \bar { = } } x$ . Given probability-weighting functions $w ^ { + }$ and $w ^ { - }$ for gains and losses, the CPT criterion is

$$
\mathcal { T } _ { \mathrm { C P T } } \{ F _ { a } ( \cdot \ | \ x ) \} = \int _ { 0 } ^ { \infty } w ^ { + } ( 1 - G _ { a } ( z \ | \ x ) ) \ d u ^ { + } ( z ) - \int _ { 0 } ^ { \infty } w ^ { - } ( G _ { a } ( - z \ | \ x ) ) \ d u ^ { - } ( z ) .
$$

CPT is useful when treatment choice should reflect not only statistical risk but also behavioral preference. Relative to above criteria, CPT can represent richer patient-specific attitudes, such as stronger aversion to losses than attraction to gains, or overweighting of small probabilities. This makes CPT particularly appealing for personalized medicine when patients or clinicians care about more nuanced risk–reward tradeoffs.

## 2.2 First-stage conditional distribution estimation

Our distribution-first framework is modular. Instead of designing a separate learner for each decision criterion, we first estimate the treatment-specific conditional outcome distribution and then evaluate downstream functionals by plug-in.

We consider three broad classes of first-stage estimators: kernel-based methods, random forest-based methods, and generative models. These three estimator classes represent distinct modeling principles for first-stage distribution learning. Kernel methods emphasize local smoothness and interpretability, random forest-based methods emphasize adaptive partitioning and robustness in heterogeneous tabular data, and generative models emphasize flexibility for complex conditional distributions. By studying all three within the same framework, we can assess whether the downstream gains of distribution-first decision learning are robust across substantially different first-stage estimators.

Kernel-based estimators. Kernel methods estimate the conditional distribution by locally weighting observations near the target covariate value. For treatment arm a, a generic estimator is

$$
{ \widehat { F } } _ { a } ( y \mid x ) = { \frac { \sum _ { i = 1 } ^ { n } K _ { h } ( X _ { i } - x ) \mathbf { 1 } ( A _ { i } = a ) \mathbf { 1 } ( Y _ { i } \leq y ) } { \sum _ { i = 1 } ^ { n } K _ { h } ( X _ { i } - x ) \mathbf { 1 } ( A _ { i } = a ) } } ,
$$

with an analogous density estimator obtained by additional smoothing in y.

The main appeal of kernel methods is their simplicity and interpretability [17]. They impose relatively weak structural assumptions and are natural when the conditional distribution varies smoothly with covariates, making them a useful baseline for distribution-first learning. Their main limitation is that performance can degrade quickly in moderate or high dimensions, where local neighborhoods become sparse and estimation becomes sensitive to the choice of metric and bandwidth.

Random Forest-based estimators. Forest-based methods replace fixed metric neighborhoods with data-adaptive neighborhoods learned through recursive partitioning. For a treatment-specific fit, or a joint fit whose weights are zero outside the queried treatment arm, the fitted forest induces nonnegative weights $\alpha _ { i } ^ { ( a ) } ( x )$ satisfying $\textstyle \sum _ { i : A _ { i } = a } \alpha _ { i } ^ { ( a ) } ( x ) = 1$ . The resulting estimator is

$$
\widehat F _ { a } ( y \mid x ) = \sum _ { i : A _ { i } = a } \alpha _ { i } ^ { ( a ) } ( x ) { \bf 1 } ( Y _ { i } \le y ) .
$$

This formulation is closely related to distributional random forests and their causal extensions [18, 20, 28]. The main advantage of this class is adaptivity. Forest methods can capture nonlinearities, interactions, and heterogeneous structure without relying on a fixed distance in the ambient covariate space. This makes them especially attractive for tabular settings with moderate or high-dimensional covariates, where only a subset of variables may be relevant. In addition, the learned weights provide a convenient representation of the full conditional distribution, which aligns naturally with our plug-in framework.

Generative models. A more flexible alternative is to model $Y ( a ) \mid X = x$ generatively and sample $\widetilde { Y } _ { a , 1 } ( x ) , \hdots , \widetilde { Y } _ { a , M } ( x ) \sim \widehat { P } \{ Y ( a ) \ | \ X = x \}$ , then construct an empirical distribution. This class includes GANs, normalizing flows, variational models, and conditional diffusion models [23, 24, 29– 34]. The main strength of this class is expressive power. Generative models can in principle represent complex non-Gaussian, skewed, heavy-tailed, or multimodal conditional outcome distributions that may be difficult to capture with local smoothing or recursive partitioning. This makes them a natural candidate when the outcome law has substantial distributional complexity. At the same time, they typically require heavier computation, more tuning, and larger effective sample sizes for stable training.

A key point is that the best estimator for global distributional fit need not be the best estimator for downstream decision quality under a particular criterion. For example, a method that fits the center of the distribution well may still perform poorly for tail-sensitive objectives if it misses lower-tail behavior. For this reason, we evaluate first-stage estimators primarily through the treatment rules they induce, rather than viewing conditional distribution estimation as an end in itself.

## 3 Theoretical Properties

We establish a general error propagation guarantee for the proposed framework, connecting first-stage conditional distribution estimation error to downstream criterion estimation error and, ultimately, to treatment policy regret. Our focus is on a broad class of Hölder-continuous functionals of the treatment-specific conditional outcome distribution. This formulation covers the main criteria considered in this work under criterion-specific regularity conditions, including the conditional mean, quantile, mean-variance, CVaR, and CPT.

For any treatment rule π, define $V _ { \mathcal { T } } ( \pi ) = \mathbb { E } \left[ Q _ { \pi ( X ) } ^ { \mathcal { T } } ( X ) \right] , \ \mathcal { R } _ { \mathcal { T } } ( \pi ) = V _ { \mathcal { T } } ( \pi _ { \mathcal { T } } ) - V _ { \mathcal { T } } ( \pi )$ . Define $e _ { n } ^ { \mathcal { T } } ( x ) = \mathrm { m a x } _ { a \in \mathcal { A } } | \widehat { Q } _ { \mathfrak { a } } ^ { \mathcal { T } } ( x ) - Q _ { a } ^ { \mathcal { T } } ( x ) |$ , and $\delta _ { n } ^ { \mathcal { T } } ( x ) \_ = \ \operatorname* { m a x } _ { a } d _ { \mathcal { T } } \{ \widehat { F } _ { \underline { { a } } } ( \cdot \ | \ x ) , F _ { a } ( \cdot \ | \ x ) \}$ . So correspondingly, we have $\begin{array} { r } { e _ { n , \infty } ^ { \mathcal { T } } = \operatorname* { s u p } _ { x \in \mathcal { X } } e _ { n } ^ { \mathcal { T } } ( x ) } \end{array}$ and $\begin{array} { r } { \delta _ { n , \infty } ^ { \mathcal { T } } = \operatorname* { s u p } _ { x \in \mathcal { X } } \delta _ { n } ^ { \mathcal { T } } ( x ) } \end{array}$ . We first characterize how perturbations of the estimated conditional distribution propagate to the criterion value.

Assumption 2 (Hölder continuity). There exist $L _ { \mathcal { T } } > 0 , \epsilon _ { \mathcal { T } } > 0 ,$ , and $0 < \rho _ { T } \leq 1$ such that, uniformly over a and x,for every distribution G satisfying $d _ { T } \{ G , F _ { a } ( \cdot \mid x ) \} \le \epsilon _ { T }$ , we have

$$
| { \cal T } ( G ) - { \cal T } \{ F _ { a } ( \cdot  { | } \ x ) \} | \leq L _ { \cal T } d _ { \cal T } \{ G , F _ { a } ( \cdot  { | } \ x ) \} ^ { \rho _ { \cal T } } .
$$

Here, $L _ { T }$ is a criterion-specific stability constant, $\epsilon \tau$ specifies the neighborhood over which the stability condition holds, while $\rho \tau$ is the Hölder exponent characterizing the sensitivity of the criterion to local distributional perturbations. When $\rho \tau = 1$ , the condition gives local Lipschitz stability, related to the CDF-based Lipschitz formulation of Huang et al. [25]. Allowing $\rho _ { T } < 1$ extends the framework to non-Lipschitz distributional criteria, including the CPT specification considered in this work. The semimetric $d _ { T }$ may be chosen according to the criterion and the underlying distribution class, and need not be common across different criteria. Verification of this condition for the criteria considered in this paper is provided in Appendix A.1.

Define the Hölder-induced score-error bound $\bar { e } _ { n } ^ { \mathcal { T } } : = L \tau \big ( \delta _ { n , \infty } ^ { \mathcal { T } } \big ) ^ { \rho _ { \mathcal { T } } }$ . Under Assumption 2, whenever $\delta _ { n , \infty } ^ { T } \leq \epsilon _ { T }$ , the uniform criterion estimation error satisfies

$$
e _ { n , \infty } ^ { \mathcal { T } } \leq \bar { e } _ { n } ^ { \mathcal { T } } .
$$

Thus, $\bar { e } _ { n } ^ { \mathcal { T } }$ summarizes how first-stage distribution estimation error propagates through the criterionspecific Hölder stability condition.

Theorem 1 (Baseline policy regret). The plug-in treatment rule satisfies $\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) \leq 2 e _ { n , \infty } ^ { \mathcal { T } }$ . Under Assumption 2, $i f \delta _ { n , \infty } ^ { \mathcal { T } } \leq \epsilon \tau$ , then $\mathcal { R } \tau ( \widehat { \pi } \tau ) \leq 2 \bar { e } _ { n } ^ { T }$

Theorem 1 establishes a direct link between first-stage conditional distribution estimation and the quality of the resulting treatment rule. A sharper guarantee is available when little probability mass lies near the treatment decision boundary. Let $\begin{array} { r } { \breve { Q } _ { \star } ^ { \mathcal { T } } ( x ) = \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { a } ^ { \mathcal { T } } ( x ) } \end{array}$ , and define the gap to the best strictly suboptimal treatment by $\begin{array} { r } { \Delta _ { \mathcal { T } } ( x ) = \operatorname* { m i n } _ { a : Q _ { a } ^ { \mathcal { T } } ( x ) < Q _ { \star } ^ { \mathcal { T } } ( x ) } \left\{ Q _ { \star } ^ { \mathcal { T } } ( x ) - Q _ { a } ^ { \mathcal { T } } ( x ) \right\} } \end{array}$ , with $\Delta \tau ( x ) = + \infty$ if all treatments are optimal.

Assumption 3 (Margin condition). There are $t _ { 0 } > 0$ and a nondecreasing $\Psi \tau : [ 0 , t _ { 0 } ]  [ 0 , 1 ]$ , with $\Psi \tau ( t ) \stackrel { - } { \to } 0 a s t \downarrow 0 ,$ , such that

$$
\mathbb { P } _ { X } \{ \Delta _ { \mathcal { T } } ( X ) \leq t \} \leq \Psi _ { \mathcal { T } } ( t ) , \qquad 0 < t \leq t _ { 0 } .
$$

The function $\Psi _ { T }$ characterizes the probability mass near the treatment decision boundary, with smaller values corresponding to fewer covariate values for which a strictly suboptimal treatment is nearly optimal. This condition encompasses the standard margin or low-noise condition used to obtain fast rates in plug-in classification and policy learning[35, 36]. In comparison, existing direct-learning approaches obtain sharper rates under method-specific structural conditions, such as approximation-error, sparsity, or function-class assumptions [3, 9].

Theorem 2 (Boundary-refined policy regret). Under Assumptions 2-3, $i f \delta _ { n , \infty } ^ { \mathcal { T } } \leq \epsilon _ { \mathcal { T } }$ and $2 \bar { e } _ { n } ^ { T } \le t _ { 0 } ,$ then $\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) \leq 2 \bar { e } _ { n } ^ { \mathcal { T } } \Psi _ { \mathcal { T } } \big ( 2 \bar { e } _ { n } ^ { \mathcal { T } } \big )$ .

Theorems 1 and 2 separate criterion stability from decision-boundary behavior. The quantity $\bar { e } _ { n } ^ { T }$ summarizes the propagation of distribution estimation error through the criterion, while $\Psi \tau ( 2 \bar { e } _ { n } ^ { \mathcal { T } } )$ provides the boundary-dependent refinement of the baseline bound. Detailed verification for specific criteria, conditional distribution estimators, and explicit convergence-rate consequences under additional assumptions, is provided in Appendix A.

## 4 Simulations

We conduct simulation studies to evaluate the proposed framework. Throughout, we consider binary randomized trials with equal treatment assignment probabilities. We compare our methods with several representative baselines, including $\ell _ { 1 } { \mathrm { - } } { \bar { \mathrm { P L S } } } \left[ 3 \right]$ , RWL [5], CQR [8, 37], and the direct $\ell _ { 2 } { \mathrm { - C V a R } }$ method of Qi et al. [9]. Within our distribution-first framework, we further compare several firststage conditional distribution estimators, including Kernel, Distributional Random Forest (DRF), Conditional DDPM, and GCDS. All results are based on 100 Monte Carlo replications. Kernel, $ { \ell _ { 1 } } \mathrm { - P L S , R W L }$ , and the direct $\ell _ { 2 } { \mathrm { - C V a R } }$ method use cross-validation for tuning; other methods use prespecified or separately selected hyperparameters as provided in Appendix B. Across settings, the outcome is generated as ${ \cal Y } = \dot { 1 } + \dot { \cal X } _ { 1 } + { \cal X } _ { 2 } + \dot { A \delta } ( { \bf X } ) + \epsilon$ We report four main metrics: Mean, $Q _ { 0 . 1 } , \mathrm { { C V a R } _ { 0 . 1 } , \mathrm { { C V a R } _ { 0 . 2 5 } } }$ , Err. Here Mean measures overall policy value, and Err measures decision error relative to the oracle rule. We also report $Q _ { 0 . 1 } , \mathrm { C V a R _ { 0 . 1 } }$ , and $\mathrm { C V a R _ { 0 . 2 5 } }$ to assess lower-tail performance, since the proposed framework is intended to support risk-sensitive decision making. In particular, $Q _ { 0 . 1 }$ captures a lower-tail threshold, whereas $\mathrm { C V a R _ { 0 . 1 } }$ summarizes the average outcome among the worst 10% of cases. These are standard downside-risk summaries and are especially informative in heavy-tailed settings [9, 38]. In the tables, for criterion-specific competitors, each row evaluates one native-trained rule under all reported criteria; for Kernel, DRF, Conditional DDPM, and GCDS, each Value/Err pair uses the plug-in rule targeting its column criterion.

Overall, our framework shows competitive performance among criterion-specific methods, especially when the outcome distribution skews from baseline settings. Among the candidate first-stage estimators, DRF is the most reliable overall choice.

## 4.1 Distribution Shift Examples

We first examine robustness under distribution shift, following the setup of Qi et al. [9]. In both shifted settings, we use $\delta ( { \bf X } ) = X _ { 1 } - X _ { 2 } + X _ { 3 } ,$ , with $n = 2 0 0$ and $p = 2 0$ . In the covariate shift setting, the training covariates are generated from $X _ { j } \sim 0 . 7 N ( 0 , 1 ) + 0 . 3$ LogNormal(0, 2), whereas the test covariates follow $X _ { j } \sim N ( 0 , 1 )$ ; the noise remains $\varepsilon \sim N ( 0 , 1 )$ for both training and test samples. In the outcome shift setting, both training and test covariates follow $\mathrm { U n i f } ( [ - 1 , 1 ] ^ { p } )$ while the training noise is generated from $\varepsilon \sim 0 . 7 N ( 0 , 1 ) + 0 . 3 \mathrm { L o g N }$ ormal(0, 2), and the test noise follows $\varepsilon \sim N ( 0 , 1 )$ . For simplicity, we only compare Kernel and $\ell _ { 2 } { \mathrm { - C V a R } }$ in this subsection. This follows Qi et al. [9], where $\bar { \ell _ { 2 } } \mathrm { - } \bar { \mathrm { C V a R } }$ is already the strongest criterion-specific robust competitor in this experiment.

Table 1: Mean and CVaR performance under distribution shift. Standard errors are reported in parentheses.
<table><tr><td></td><td></td><td colspan="2">Mean</td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td></td><td>Scenario Method</td><td>Value</td><td>Err</td><td>Value</td><td>Err</td><td>Value</td><td>Err</td><td>Value</td><td>Err</td></tr><tr><td></td><td>Oracle</td><td></td><td></td><td></td><td></td><td>2.380(0.003) 0.000(0.000) -1.061 (0.005) 0.000 (0.000) –0.142 (0.004) 0.000 (0.000) 0.773 (0.003) 0.000 (0.000)</td><td></td><td></td><td></td></tr><tr><td>Covariate</td><td>l2-CVaR (0.1)</td><td></td><td></td><td></td><td></td><td>2.092(0.025) 0.198(0.009) –1.695 (0.046) 0.198 (0.009) –0.656(0.039) 0.198 (0.009) 0.360 (0.033) 0.198 (0.009)</td><td></td><td></td><td></td></tr><tr><td>shift</td><td> $\ell _ { 2 } \cdot$ </td><td></td><td></td><td></td><td></td><td>-CVaR (0.25) 2.145(0.019) 0.179(0.008) –1.605(0.038) 0.179 (0.008) –0.573 (0.032) 0.179(0.008) 0.431 (0.027) 0.179(0.008)</td><td></td><td></td><td></td></tr><tr><td></td><td> $\ell _ { 2 } \cdot$  -CVaR (0.5)</td><td></td><td></td><td></td><td></td><td>2.174(0.021) 0.162(0.008)-1.547(0.042) 0.162 (0.008) -0.528 (0.035) 0.162(0.008) 0.471 (0.029) 0.162(0.008)</td><td></td><td></td><td></td></tr><tr><td></td><td>Kernel</td><td></td><td></td><td></td><td></td><td>2.333(0.004) 0.077(0.002) -1.419(0.023) 0.113(0.003) –0.323 (0.011) 0.098 (0.003) 0.671 (0.006) 0.091 (0.003)</td><td></td><td></td><td></td></tr><tr><td></td><td>Oracle</td><td>1.812(0.002) 0.000(0.000) −0.702(0.004) 0.000 (0.000) –0.016(0.003) 0.000 (0.000) 0.673 (0.002) 0.000 (0.000)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Outcome shift</td><td> $\ell _ { 2 } \cdot$  CVaR (0.1)</td><td></td><td></td><td></td><td></td><td>1.430(0.019) 0.314(0.009) -1.349(0.024) 0.314(0.009) -0.594(0.024) 0.314(0.009) 0.169(0.023) 0.314(0.009)</td><td></td><td></td><td></td></tr><tr><td> $\ell _ { 2 } \cdot$ </td><td></td><td></td><td></td><td></td><td></td><td>CVaR (0.25) 1.463 (0.018) 0.298 (0.009) −1.300 (0.024) 0.298 (0.009) –0.550 (0.023) 0.298 (0.009) 0.209 (0.022) 0.298 (0.009)</td><td></td><td></td><td></td></tr><tr><td></td><td> $\ell _ { 2 } \cdot$  CVaR (0.5)</td><td></td><td></td><td></td><td></td><td>1.504(0.018) 0.277(0.008) -1.250(0.023) 0.277 (0.008) -0.497 (0.023) 0.277 (0.008) 0.259 (0.022) 0.277 (0.008)</td><td></td><td></td><td></td></tr><tr><td></td><td>Kernel</td><td>1.514(0.021) 0.260 (0.010) -0.892 (0.014) 0.131 (0.004) -0.114(0.008) 0.105 (0.003) 0.621 (0.005) 0.089 (0.003)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1 shows that Kernel is consistently more stable than $\ell _ { 2 } { \mathrm { - C V a R } }$ under both types of shift. The main reason might be that our method depends more directly on the conditional distribution, instead of a single criterion-specific training objective, and therefore is more robust when the training and test environments differ. The internal comparison within the framework is also informative. In the outcome shift scenario, Kernel plug-in rules based on risk-sensitive criteria substantially outperform the mean-based rule, reflecting that lower-tail criteria are less affected by the heavy-tailed training noise than the conditional mean.

## 4.2 Comparison Across Outcome Regimes

We evaluate the proposed framework further via three additional examples. In the main text, we focus on dimension $p = 1 0$ and $n = 1 0 0 0$ since this setting is close to our real data, with additional results deferred to Appendix B. $X _ { j } \overset { \mathrm { i i d } } { \sim } \mathrm { U n i f } [ - 1 , 1 ]$ . The testing sample size is 10,000. The experimental settings below are from [9]:

Setting 1 (linear Gaussian): $\delta ( X ) = X _ { 1 } - X _ { 2 } + X _ { 3 } , \varepsilon \sim N ( 0 , 1 )$

Setting 2 (nonlinear Gaussian): $\delta ( X ) = \varsigma ( X _ { 1 } ) \varsigma ( X _ { 2 } ) - c _ { 0 } , \varsigma ( x ) = 1 + 1 / ( 1 + \exp \{ - 2 0 ( x -$ $1 / 3 ) \} \bar { \rangle _ { 0 } } , \varepsilon \sim N ( 0 , 1 )$ , with $c _ { 0 } = 1 . 8 7$

Setting 3 (heavy-tail): $\delta ( X ) = X _ { 1 } - X _ { 2 } + X _ { 3 } , \log \varepsilon \sim N ( 0 , 2 ^ { 2 } )$

Setting 1 is the standard sparse linear Gaussian benchmark and is oracle for both $\ell _ { 1 } { \mathrm { - P L S } }$ and CQR. Setting 2 is used to test adaptation to nonlinear decision boundary. Setting 3 introduces heavy right-tailed lognormal noise, which is used to assess the robustness of different ITRs and is also oracle for CQR. Because ε is independent of $( X , A )$ in Settings 1–3, the two treatment-specific conditional outcome distributions differ only by a location shift. All criteria considered here are monotone with respect to first-order stochastic dominance, so they share the same oracle rule sign $\{ \delta ( X ) \}$ . These settings therefore assess how accurately each method recovers this common rule under different outcome regimes. Appendix B.4 considers a treatment-interactive design in which the optimal rule genuinely depends on the criterion.

The results are summarized in Tables 2–4. Overall, our framework shows competitive performance among criterion-specific methods. In Setting 1, the proposed framework remains competitive even in this favorable regime for classical linear baselines. Within the distribution-first family, Conditional DDPM performs best, followed by GCDS and DRF. In Setting 2, DRF becomes the strongest method among all competitors, showing its ability to adapt to nonlinear treatment heterogeneity. Conditional DDPM and GCDS remain competitive with criterion-specific methods. In Setting 3, GCDS, Kernel, and DRF all clearly outper- form the other baselines beyond oracle CQR. In the distribution-first family, DRF has the highest $Q _ { 0 . 1 }$ and $\mathrm { C V a R _ { 0 . 1 } }$ Values, whereas Kernel has the highest $\mathrm { C V a R _ { 0 } } .$ 25 Values and the lowest Err at all four tail criteria. The linear mean-based baselines deteriorate sharply in this regime. Since all criteria share the same oracle rule here, this deterioration reflects the sensitivity of squared-loss estimation to heavy right-tailed noise, whereas quantile- and CVaR-based plug-in rules are more robust to such noise. Together with the treatment-interactive and higher-dimensiona experiments in Appendix B, these results show that DRF is the most reliable first-stage estimator in most settings, whereas Kernel degrades quickly as p grows and the generative estimators are less stable, with one GCDS run failing. We therefore recommend DRF as the default first-stage estimator and regard generative models as promising but exploratory.

Table 2: Setting 1 (linear Gaussian; $n = 1 0 0 0 , p = 1 0 )$ . Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td></td><td colspan="2">Mean</td><td colspan="2"> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td></tr><tr><td>Method</td><td>Value</td><td>Err</td><td>Value</td><td> $\mathrm { E r r }$ </td><td>Value</td><td>Err</td><td>Value</td><td>Err</td></tr><tr><td>Oracle</td><td>1.810(0.002)</td><td>0.000 (0.000)</td><td>-0.042 (0.004)</td><td></td><td>0.000 (0.000) -0.699 (0.004)</td><td>0.000 (0.000)</td><td>-0.015(0.003)</td><td>0.000 (0.000)</td></tr><tr><td>l1-PLS</td><td>1.805 (0.002)</td><td></td><td>0.034 (0.002) −0.050 (0.004)</td><td></td><td>0.034(0.002) -0.712(0.005)</td><td>0.034(0.002)</td><td>)-0.024 (0.004)</td><td>0.034(0.002)</td></tr><tr><td>RWL</td><td>1.800 (0.002)</td><td></td><td>0.051 (0.002) -0.060 (0.004)</td><td></td><td>0.051(0.002)-0.720(0.004)</td><td>0.051 (0.002)</td><td>) -0.032 (0.003)</td><td>0.051 (0.002)</td></tr><tr><td>CQR (0.1)</td><td>1.799 (0.002)</td><td></td><td>0.053 (0.001) -0.063 (0.004)</td><td></td><td>0.053 (0.001) -0.723 (0.004)</td><td>0.053 (0.001)</td><td>)-0.035 (0.003)</td><td>0.053 (0.001)</td></tr><tr><td> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>1.776(0.003)</td><td>0.092 (0.002)</td><td>-0.102 (0.005)</td><td>0.092 (0.002)</td><td>-0.768 (0.005)</td><td>0.092 (0.002)</td><td>-0.075 (0.004)</td><td>0.092(0.002)</td></tr><tr><td> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 2 5 \right)$ </td><td>1.786(0.003)</td><td>0.078 (0.002)</td><td>-0.088 (0.004)</td><td>0.078 (0.002)</td><td>-0.754(0.005)</td><td>0.078 (0.002)</td><td>-0.060(0.004)</td><td>0.078 (0.002)</td></tr><tr><td>Kernel</td><td>1.771 (0.003)</td><td>0.096(0.002)</td><td>-0.221 (0.006)</td><td>0.148 (0.002)</td><td>-0.923 (0.007)</td><td>0.146(0.002)</td><td>-0.144(0.006)</td><td>0.125 (0.002)</td></tr><tr><td>DRF</td><td>1.788 (0.002)</td><td></td><td>0.075 (0.001) -0.096(0.004)</td><td></td><td>0.084(0.001) -0.766(0.005)</td><td>0.085 (0.001)</td><td>-0.062(0.003)</td><td>0.080 (0.001)</td></tr><tr><td>Conditional DDPM1.807 (0.002)</td><td></td><td></td><td></td><td></td><td>0.032 (0.001) −0.050 (0.004) 0.033 (0.001) −0.709 (0.004)</td><td></td><td>0.034(0.001)-0.023(0.003)</td><td>0.033(0.001)</td></tr><tr><td>GCDS</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.798(0.003) 0.055 (0.001) −0.077(0.004) 0.067(0.002) −0.756(0.005) 0.076(0.003) −0.050(0.003) 0.066(0.002)</td><td></td></tr></table>

Table 3: Setting 2 (nonlinear Gaussian; $n = 1 0 0 0 , p = 1 0 )$ . Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td></td><td colspan="2">Mean</td><td colspan="2"> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td></tr><tr><td>Method</td><td>Value</td><td>Err</td><td>Value</td><td>Err</td><td>Value</td><td>Err</td><td>Value</td><td>Err</td></tr><tr><td>Oracle</td><td>1.654(0.003)</td><td>0.000 (0.000)</td><td>-0.044 (0.003)</td><td>0.000 (0.000)</td><td>-0.587 (0.003)</td><td>0.000 (0.000)</td><td>-0.029(0.002)</td><td>0.000(0.000)</td></tr><tr><td>l1-PLS</td><td>1.566(0.003)</td><td></td><td>0.173 (0.001) -0.156(0.003)</td><td></td><td>0.173 (0.001) -0.697 (0.003)</td><td>0.173 (0.001)</td><td>-0.139(0.003)</td><td>0.173 (0.001)</td></tr><tr><td>RWL</td><td>1.592(0.003)</td><td>0.214(0.002)</td><td>-0.118(0.003)</td><td>0.214(0.002)</td><td>)-0.658 (0.003)</td><td>0.214(0.002)</td><td>-0.102(0.003)</td><td>0.214(0.002)</td></tr><tr><td>CQR (0.1)</td><td>1.561 (0.003)</td><td>0.182(0.001)</td><td>-0.166(0.004)</td><td></td><td>0.182 (0.001) -0.709 (0.004)</td><td>0.182(0.001)</td><td>-0.148 (0.003)</td><td>0.182(0.001)</td></tr><tr><td> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>1.559(0.004)</td><td>0.253 (0.003)</td><td>-0.140 (0.004)</td><td>0.253 (0.003)</td><td>-0.679 (0.004)</td><td>0.253 (0.003)</td><td>-0.124(0.003)</td><td>0.253 (0.003)</td></tr><tr><td>l2-CVaR (0.25)</td><td>1.578 (0.003)</td><td>0.242 (0.003)</td><td>-0.125(0.004)</td><td>0.242 (0.003)</td><td>-0.664(0.004)</td><td>0.242 (0.003)</td><td>-0.109(0.003)</td><td>0.242 (0.003)</td></tr><tr><td>Kernel</td><td>1.582(0.003)</td><td>0.231 (0.002)</td><td>-0.237 (0.004)</td><td>0.412(0.003)</td><td>-0.807 (0.004)</td><td>0.448 (0.003)</td><td>-0.219(0.004)</td><td>0.416(0.003)</td></tr><tr><td>DRF</td><td>1.638(0.003)</td><td></td><td>0.066(0.004) -0.076(0.003)</td><td>0.129 (0.007)</td><td>-0.626(0.004)</td><td>0.158(0.008)</td><td>-0.060 (0.003)</td><td>0.124(0.007)</td></tr><tr><td>Conditional DDPM1.596(0.003)</td><td></td><td></td><td>0.200(0.001)-0.121 (0.003)</td><td></td><td>0.195(0.001)-0.663(0.003)</td><td></td><td>0.193(0.001)-0.105(0.002)</td><td>0.195(0.001)</td></tr><tr><td>GCDS</td><td></td><td></td><td></td><td></td><td>1.568(0.003) 0.184(0.001) -0.192(0.007) 0.187(0.001) -0.769(0.011) 0.195 (0.002) -0.173 (0.006)</td><td></td><td></td><td>0.187 (0.001)</td></tr></table>

## 5 Real data analysis

We evaluate the proposed framework on the ACTG175 HIV clinical trial, a randomized study comparing several antiretroviral regimens in HIV-infected adult[39]. We compare ZDV+ddI with $\mathrm { Z D V + Z A L }$ among 1,046 participants, and use the cohort and 12 baseline covariates supplied with Qi et al. [9]. The outcome is the change in CD4 count from baseline to follow-up with larger values preferred. All methods use the same 200 treatment-stratified 80/20 splits. The results show that our framework yields a more flexible treatment rule than the criterion-specific methods, while maintaining strong out-of-sample performance.

We compare $\ell _ { 1 } { \cdot } \mathrm { { P L S } } , \ell _ { 2 } { \cdot } \mathrm { { C V a R } } ( 0 . 2 5 )$ , and CQR(0.25), with three policies derived from our distributionfirst framework: DRF-Mean, DRF-CVaR(0.25), and DRF-CPT. For the proposed approach, we first estimate the conditional outcome distribution under each treatment arm using DRF, and then construct treatment rules by optimizing different decision criteria with respect to the estimated distribution. All learned policies are evaluated on held-out test sets using mean value, $\mathrm { C V a R } _ { 0 . 2 5 } ,$ , and CPT. We also report the proportion of subjects assigned to treatment 1 (ZDV+ddI) and runtime. CPT uses reference point zero (no CD4 change), with η<sub>+</sub> = η<sub>−</sub> = 0.88, λ = 2.25, $\gamma _ { + } = 0 . 6 1$ , and $\gamma _ { - } = 0 . 6 9 [ 1 2 ]$ . This choice is natural in the present setting, where avoiding poor outcomes and accounting for asymmetric gain-loss preferences may both be clinically relevant.

Table 4: Setting 3 (heavy-tailed lognormal; $n = 1 0 0 0 , p = 1 0 )$ . Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td></td><td colspan="2">Mean</td><td colspan="2"> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td colspan="2"> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td></tr><tr><td>Method</td><td>Value</td><td>Err</td><td>Value</td><td> $\mathrm { E r r }$ </td><td>Value</td><td>Err</td><td>Value</td><td>Err</td></tr><tr><td>Oracle</td><td>9.244(0.075)</td><td>0.000(0.000)</td><td>1.167(0.003)</td><td>0.000(0.000)</td><td>0.615(0.003)</td><td>0.000(0.000)</td><td>1.243 (0.002)</td><td>0.000 (0.000)</td></tr><tr><td>l1-PLS</td><td>8.670(0.081)</td><td>0.432(0.011)</td><td>0.267 (0.028)</td><td>0.432(0.011)</td><td>-0.339(0.025)</td><td>0.432(0.011)</td><td>0.357(0.027)</td><td>0.432(0.011)</td></tr><tr><td>RWL</td><td>8.759(0.089)</td><td>0.396(0.009)</td><td>0.348 (0.024)</td><td>0.396(0.009)</td><td>-0.263 (0.023)</td><td>0.396(0.009)</td><td>0.444(0.024)</td><td>0.396(0.009)</td></tr><tr><td>CQR (0.1)</td><td>9.243(0.076)</td><td>0.009(0.000)</td><td>1.167 (0.003)</td><td>0.009 (0.000)</td><td>0.614(0.003)</td><td>0.009(0.000)</td><td>1.242 (0.002)</td><td>0.009 (0.000)</td></tr><tr><td>l2-CVaR (0.1)</td><td>9.074(0.084)</td><td>0.190(0.016)</td><td>0.867 (0.040)</td><td>0.190(0.016)</td><td>0.279(0.041)</td><td>0.190(0.016)</td><td>0.943 (0.039)</td><td>0.190(0.016)</td></tr><tr><td> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 2 5 \right)$ </td><td>9.080(0.086)</td><td>0.186(0.014)</td><td>0.884(0.036)</td><td>0.186(0.014)</td><td>0.289(0.037)</td><td>0.186(0.014)</td><td>0.959(0.035)</td><td>0.186(0.014)</td></tr><tr><td>Kernel</td><td>8.681 (0.082)</td><td>0.416(0.007)</td><td>1.069(0.004)</td><td>0.116(0.001)</td><td>0.468 (0.004)</td><td>0.109(0.001)</td><td>1.168(0.003)</td><td>0.096(0.001)</td></tr><tr><td>DRF</td><td>8.861 (0.081)</td><td>0.372(0.010)</td><td>1.088(0.004)</td><td>0.118(0.002)</td><td>0.516(0.005)</td><td>0.118(0.002)</td><td>1.166(0.003)</td><td>0.114(0.002)</td></tr><tr><td>Conditional DDPM</td><td>8.994(0.081)</td><td>0.266(0.009)</td><td>0.897(0.018)</td><td>0.194(0.007)</td><td>0.267 (0.021)</td><td>0.201 (0.008)</td><td>0.976(0.017)</td><td>0.193 (0.007)</td></tr><tr><td>GCDS</td><td>9.089(0.074)</td><td>0.163(0.008)</td><td>1.024(0.014)</td><td>0.138(0.006)</td><td>0.411 (0.016)</td><td>0.142(0.006)</td><td>1.097(0.013)</td><td>0.136(0.006)</td></tr></table>

Table 5: ACTG175 policy evaluation and local computational cost. Utilities and treatment rates are mean (standard deviation) across 200 paired splits; runtimes are median [Q1, Q3] in seconds. Each row evaluates the same learned rule under all three utilities. The three DRF rules share one distribution-estimation stage, and their combined runtime is reported once. Bold indicates the largest average value in each utility column.
<table><tr><td>Method</td><td>Criterion</td><td>Mean</td><td> $\mathrm { C V a R _ { 0 . 2 5 } }$ </td><td>CPT</td><td>Treat-1 rate</td><td>Runtime (s)</td></tr><tr><td> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Mean</td><td>51.797 (12.731)</td><td>-107.879 (21.484)</td><td>-3.485 (13.110)</td><td>0.887 (0.067)</td><td>0.073 [0.051, 0.083]</td></tr><tr><td>l2-CVaR (0.25)</td><td>CVaR (0.25)</td><td>52.878 (13.061)</td><td>-107.955 (21.109)</td><td>-2.946 (12.631)</td><td>0.921 (0.123)</td><td>193.587 [135.079, 268.785]</td></tr><tr><td>CQR (0.25)</td><td>Quantile (0.25)</td><td>48.280 (12.845)</td><td>-109.971 (20.140)</td><td>-6.121 (12.554)</td><td>0.702 (0.071)</td><td>0.005 [0.004, 0.015]</td></tr><tr><td rowspan="3">DRF (shared fit)</td><td>Mean</td><td>54.042 (12.292)</td><td>-108.808 (21.467)</td><td>-3.058 (12.720)</td><td>0.983 (0.021)</td><td rowspan="3">4.042 [3.238, 4.662]</td></tr><tr><td>CVaR (0.25)</td><td>44.950 (12.863)</td><td>-105.383 (17.219)</td><td>-2.555 (11.481)</td><td>0.575 (0.107)</td></tr><tr><td>CPT</td><td>53.637 (12.150)</td><td>-104.632 (19.734)</td><td>-1.246 (12.363)</td><td>0.861 (0.076)</td></tr></table>

Table 5 reports all three utilities for each fixed learned rule. DRF-Mean has the highest average held-out mean, while DRF-CPT has the highest average marginal CVaR and CPT among the six policies. DRF-CVaR also has a numerically higher average CVaR than the three baselines, but a lower average mean, illustrating a tradeoff between the evaluated utilities. The three DRF-based policies also produce meaningfully different treatment rates, indicating that different downstream criteria can lead to substantively different treatment recommendations even when they are based on the same estimated conditional outcome distribution. This illustrates the practical value of the distribution-first framework that once a flexible conditional distribution model is fitted, one can construct and compare treatment rules tailored to different decision objectives within a single unified pipeline.

Runtime includes split extraction, initialization, parameter selection where applicable, fitting, prediction, and held-out evaluation. The DRF time covers all three policies and their full cross-utility evaluations. The CVaR pipeline includes five-fold cross-validation over 11 regularization parameters, whereas DRF uses fixed settings without a tuning search. With the split-specific regularization parameter already selected, the separately timed post-selection CVaR workflow takes 1.849 [1.380, 8.533] seconds. Thus, DRF’s computational benefit is reusable distribution estimation across criteria, rather than uniformly faster single-criterion learning.

## 6 Discussion

We study risk-sensitive ITR learning through a distribution-first framework. The key idea is to first estimate the treatment-specific conditional outcome distribution and then construct the treatment rule by plug-in criterion evaluation. This yields a unified pipeline for multiple risk-sensitive criteria, including mean, quantile, CVaR, and CPT-type objectives. Our results support this perspective both theoretically and empirically. The theory provides a general error propagation result from firststage distribution estimation to downstream criterion estimation and treatment regret. Empirically, performance depends on the outcome regime and first-stage estimator, while the treatment-interactive and real-data analyses illustrate criterion-dependent decisions using shared distribution estimates.

Several directions remain for future work. First, extending the distribution-first approach to doubly robust estimators is a potential next step, especially for observational settings. Second, the current theory focuses on Hölder-continuous criteria; sharper results for specific criteria may yield stronger and more interpretable guarantees. Third, our experiments suggest that gains from generative models depend on the outcome regime and target criterion. Future work should therefore investigate broader data types and response structures where generative models may be better matched to the underlying outcome distribution.

## Acknowledgments and Disclosure of Funding

The authors thank the reviewers for their helpful and constructive comments, which have improved the presentation of this paper. This research was supported in part by NSF grant DMS-2625396. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. The author declare no competing interests.

## References

[1] Michael R. Kosorok and Eric B. Laber. Precision medicine. Annual Review of Statistics and Its Application, 6:263–286, 2019.

[2] Susan A. Murphy. Optimal dynamic treatment regimes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 65(2):331–355, 2003.

[3] Min Qian and Susan A. Murphy. Performance guarantees for individualized treatment rules. The Annals of Statistics, 39(2):1180–1210, 2011.

[4] Yingqi Zhao, Donglin Zeng, A. John Rush, and Michael R. Kosorok. Estimating individualized treatment rules using outcome weighted learning. Journal ofthe American Statistical Association, 107(499):1106–1118, 2012.

[5] Xin Zhou, Nicole Mayer-Hamblett, Umer Khan, and Michael R. Kosorok. Residual weighted learning for estimating individualized treatment rules. Journal of the American Statistical Association, 112(517):169–187, 2017.

[6] Zhengling Qi, Dacheng Liu, Haoda Fu, and Yufeng Liu. Multi-armed angle-based direct learning for estimating optimal individualized treatment rules with various outcomes. Journal ofthe American Statistical Association, 115(530):678–691, 2020.

[7] Lan Wang, Yu Zhou, Rui Song, and Ben Sherwood. Quantile-optimal treatment regimes. Journal ofthe American Statistical Association, 113(523):1243–1254, 2018.

[8] Wei Xiao, Hao Helen Zhang, and Wenbin Lu. Robust regression for optimal individualized treatment rules. Statistics in Medicine, 38(11):2059–2073, 2019.

[9] Zhengling Qi, Jong-Shi Pang, and Yufeng Liu. On robustness of individualized decision rules. Journal ofthe American Statistical Association, 118(543):2143–2157, 2023.

[10] Arthur E. Attema, Werner B. F. Brouwer, and Olivier l’Haridon. Prospect theory in the health domain: A quantitative assessment. Journal ofHealth Economics, 32(6):1057–1065, 2013.

[11] Stefan A. Lipman, Werner B. F. Brouwer, and Arthur E. Attema. The corrective approach: Policy implications of recent developments in QALY measurement based on prospect theory. Value in Health, 22(7):816–821, 2019.

[12] Amos Tversky and Daniel Kahneman. Advances in prospect theory: Cumulative representation of uncertainty. Journal ofRisk and Uncertainty, 5(4):297–323, 1992.

[13] L. A. Prashanth and Michael C. Fu. Risk-sensitive reinforcement learning via policy gradient search. Foundations and Trends in Machine Learning, 15(5):537–693, 2022.

[14] Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. Off-policy risk assessment for markov decision processes. In International Conference on Artificial Intelligence and Statistics, volume 151, pages 5022–5050, 2022.

[15] Anas Barakat, Souradip Chakraborty, Peihong Yu, Pratap Tokekar, and Amrit Singh Bedi. On the global optimality of policy gradient methods in general utility reinforcement learning. In Advances in Neural Information Processing Systems, 2025.

[16] Mason O Smith and Wenlong Zhang. Risk-sensitive theory of mind: Coordinating with agents of unknown bias using cumulative prospect theory. In International Conference on Machine Learning, volume 267, pages 55999–56017, 2025.

[17] Peter Hall, Rodney C. L. Wolff, and Qiwei Yao. Methods for estimating a conditional distribution function. Journal ofthe American Statistical Association, 94(445):154–163, 1999.

[18] Nicolai Meinshausen. Quantile regression forests. Journal of Machine Learning Research, 7 (35):983–999, 2006.

[19] Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019.

[20] Domagoj Cevid, Lukas Michel, Jeffrey Näf, Peter Bühlmann, and Nicolai Meinshausen. Distri-<sup>´</sup> butional random forests: Heterogeneity adjustment and multivariate distributional regression. Journal ofMachine Learning Research, 23(333):1–79, 2022.

[21] Xingyu Zhou, Yuling Jiao, Jin Liu, and Jian Huang. A deep generative approach to conditional sampling. Journal ofthe American Statistical Association, 118(543):1837–1848, 2023.

[22] Qiao Liu, Jiaze Xu, Rui Jiang, and Wing Hung Wong. Density estimation using deep generative neural networks. Proceedings of the National Academy of Sciences, 118(15):e2101344118, 2021.

[23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[24] Yuchen Ma, Valentyn Melnychuk, Jonas Schweisthal, and Stefan Feuerriegel. DiffPO: A causal diffusion model for learning distributions of potential outcomes. In Advances in Neural Information Processing Systems, volume 37, pages 43663–43692, 2024.

[25] Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. Off-policy risk assessment in contextual bandits. In Advances in Neural Information Processing Systems, volume 34, pages 23714–23726, 2021.

[26] Jeffrey Näf, Junhyung Park, and Herbert Susmann. Causal-DRF: Conditional kernel treatment effect estimation using distributional random forest. arXiv preprint arXiv:2411.08778, 2024.

[27] Donald B Rubin. Estimating causal effects of treatments in randomized and nonrandomized studies. Journal ofEducational Psychology, 66(5):688–701, 1974.

[28] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[29] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27, 2014.

[30] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[31] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.

[32] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[33] Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, and Artem Babenko. TabDDPM: Modelling tabular data with diffusion models. In International Conference on Machine Learning, volume 202, pages 17564–17579, 2023.

[34] Rong Tang, Lizhen Lin, and Yun Yang. Conditional diffusion models are minimax-optimal and manifold-adaptive for conditional distribution estimation. In International Conference on Learning Representations, 2025.

[35] Jean-Yves Audibert and Alexandre B. Tsybakov. Fast learning rates for plug-in classifiers. The Annals ofStatistics, 35(2):608–633, 2007.

[36] Alexander Luedtke and Antoine Chambaz. Faster rates for policy learning. arXiv preprint arXiv:1704.06431, 2017.

[37] Kristin A. Linn, Eric B. Laber, and Leonard A. Stefanski. Interactive Q-learning for quantiles. Journal ofthe American Statistical Association, 112(518):638–649, 2017.

[38] R. Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal ofRisk, 2(3):21–41, 2000.

[39] Scott M Hammer, David A Katzenstein, Michael D Hughes, Holly Gundacker, Robert T Schooley, Richard H Haubrich, Keith WK Henry, Michael M Lederman, John P Phair, Manette Niu, et al. A trial comparing nucleoside monotherapy with combination therapy in hiv-infected adults with cd4 cell counts from 200 to 500 per cubic millimeter. New England Journal of Medicine, 335(15):1081–1090, 1996.

[40] Aad W Van Der Vaart and Jon A Wellner. Weak Convergence and Empirical Processes: With Applications to Statistics. Springer, 1996.

[41] Uwe Einmahl and David M. Mason. Uniform in bandwidth consistency of kernel-type function estimators. The Annals ofStatistics, 33(3):1380–1403, 2005.

## A Additional theoretical details

## A.1 Verification of the Hölder Continuity Condition

Assumption 2 is formulated at the level of criterion stability and allows the semimetric $d _ { T }$ to depend on the distributional criterion and the underlying distribution class. The formulation is closely related to the CDF-based Lipschitz risk-functional framework of Huang et al. [25], which establishes plug-in stability for a broad class of Lipschitz functionals and verifies such properties for criteria including the mean, CVaR, variance, mean–variance, and CPT-type objectives. These results correspond to the special case $\rho \tau = 1$ of Assumption 2.

The Hölder formulation relaxes the global Lipschitz requirement in two directions. First, it requires stability only locally, which accommodates criteria such as quantiles that are not globally Lipschitz but are locally Lipschitz under standard regularity conditions around the target quantile. Second, it allows Hölder exponents $( \rho \tau < 1 )$ ), which accommodates functionals that are locally continuous but not Lipschitz. In particular, the Tversky–Kahneman probability distortions used in our CPT specification exhibit singular behavior near the probability endpoints and are generally not Lipschitz on [0,1]. We show below that these distortions are Hölder continuous, yielding a corresponding Hölder stability bound for the CPT functional.

Write

$$
d _ { \infty } ( F , G ) = \operatorname* { s u p } _ { y \in \mathbb { R } } | F ( y ) - G ( y ) |
$$

and let $F ^ { - 1 } ( u ) = \operatorname* { i n f } \{ y : F ( y ) \geq u \}$ . The next proposition records conditions under which common criteria satisfy Assumption 2. These conditions are not interchangeable: for example, CDF sup-norm convergence on an unbounded support does not by itself control a mean or a tail average.

For CPT, the formal statement below uses a prespecified fixed reference point $r _ { 0 } = 0$ , matching the ACTG175 analysis. A covariate-dependent reference point $r ( x )$ instead defines a family of functionals $\mathcal { T } _ { x } ;$ the same argument applies only when the stability modulus and all transformed-outcome bounds hold uniformly in x.

Proposition 1 (Representative stability moduli). Suppose F and $G$ have common support contained in [ℓ, u], and le $D = u - \ell a n d M = \operatorname* { m a x } \{ | \ell | , | u | \}$

1. For the mean $\textstyle { \mathcal { T } } _ { \operatorname* { m e a n } } ( F ) = \int y d F ( y )$

$$
| \mathcal { T } _ { \mathrm { m e a n } } ( F ) - \mathcal { T } _ { \mathrm { m e a n } } ( G ) | \leq D d _ { \infty } ( F , G ) .
$$

2. For $\lambda \ge 0 a n d \mathcal { T } _ { \mathrm { M V } } ( F ) = \mathcal { T } _ { \mathrm { m e a n } } ( F ) - \lambda \mathcal { T } _ { \mathrm { V a r } } ( F ) ,$

$$
| \mathcal { T } _ { \mathrm { M V } } ( F ) - \mathcal { T } _ { \mathrm { M V } } ( G ) | \le D ( 1 + 4 \lambda M ) d _ { \infty } ( F , G ) .
$$

3. For the lower-tail CVaR of a reward, obtained by sign-converting the usual upper-tail loss convention,

$$
{ \mathcal { T } } _ { \mathrm { C V a R } , \tau } ( F ) = { \frac { 1 } { \tau } } \int _ { 0 } ^ { \tau } F ^ { - 1 } ( v ) d v , \qquad \tau \in ( 0 , 1 ) ,
$$

$$
| \mathcal { T } _ { \mathrm { C V a R } , \tau } ( F ) - \mathcal { T } _ { \mathrm { C V a R } , \tau } ( G ) | \leq \frac { D } { \tau } d _ { \infty } ( F , G ) .
$$

4. Let ${ \mathcal T } _ { Q , \tau } ( F ) = F ^ { - 1 } ( \tau )$ , write $q _ { \tau } = F ^ { - 1 } ( \tau )$ , and suppose $F ( q _ { \tau } ) = \tau$ . Assume that, for some $\dot { \boldsymbol { r } } _ { \tau } , \boldsymbol { c } _ { \tau } > 0 , \boldsymbol { F }$ is absolutely continuous on $[ q _ { \tau } - r _ { \tau } , q _ { \tau } + r _ { \tau } ]$ , with density satisfying $f ( y ) \geq c _ { \tau }$ almost everywhere on this interval. If

$$
d _ { \infty } ( F , G ) < \operatorname* { m i n } \{ \tau , 1 - \tau , c _ { \tau } r _ { \tau } \} ,
$$

then

$$
| \mathcal { T } _ { Q , \tau } ( F ) - \mathcal { T } _ { Q , \tau } ( G ) | \leq c _ { \tau } ^ { - 1 } d _ { \infty } ( F , G ) .
$$

For a bound uniform in treatments and covariates, the same $r _ { \tau } , c _ { \tau }$ , and crossing condition $F _ { a } ( q _ { \tau , a } ( x ) \mid x ) ^ { \cdot } = \tau$ must hold uniformly over a and x.

5. Consider a cumulative-prospectfunctional $I I 2 J$ with a nondecreasing valuefunction v, the fixed reference point normalized to zero so that $v ( 0 ) = 0 ,$ , and transformed gains and losses $\stackrel { \cdot } { v } ^ { + } ( y ) \stackrel { \cdot } { = } \operatorname* { m a x } \bar { \{ v ( y ) , 0 \} }$ and $v ^ { - } ( y ) = \operatorname* { m a x } \{ - v ( y ) , 0 \}$ satisfying $\begin{array} { r } { 0 \leq v ^ { + } \bar { ( Y ) } \leq M _ { + } } \end{array}$ and $0 \leq v ^ { - } ( Y ) \leq M _ { - }$ , and probability distortions satisfying

$$
| w _ { \pm } ( s ) - w _ { \pm } ( t ) | \leq L _ { \pm } | s - t | ^ { \rho _ { \pm } } .
$$

Define its gain-minus-loss Choquet tail representation by

$$
\begin{array} { r l r } {  { \mathcal { T } _ { \mathrm { C P T } } ( F ) = \int _ { 0 } ^ { M _ { + } } w _ { + } \bigl ( \mathbb { P } _ { F } \{ v ^ { + } ( Y ) > z \} \bigr ) \ d z } } \\ & { } & { - \int _ { 0 } ^ { M _ { - } } w _ { - } \bigl ( \mathbb { P } _ { F } \{ v ^ { - } ( Y ) > z \} \bigr ) \ d z . } \end{array}
$$

With $\rho = \operatorname* { m i n } ( \rho _ { + } , \rho _ { - } )$ , this functional satisfies

$$
| \mathcal { T } _ { \mathrm { C P T } } ( F ) - \mathcal { T } _ { \mathrm { C P T } } ( G ) | \leq \left( L _ { + } M _ { + } + L _ { - } M _ { - } \right) d _ { \infty } ( F , G ) ^ { \rho } .
$$

Proof. In one dimension,

$$
W _ { 1 } ( F , G ) = \int _ { \ell } ^ { u } \left| F ( y ) - G ( y ) \right| d y = \int _ { 0 } ^ { 1 } \left| F ^ { - 1 } ( v ) - G ^ { - 1 } ( v ) \right| d v ,
$$

so $W _ { 1 } ( F , G ) \leq D d _ { \infty } ( F , G )$ . The mean is 1-Lipschitz in $W _ { 1 }$ , proving the first claim. Integration by parts for the bounded-variation function $y ^ { 2 }$ gives

$$
\begin{array} { r } { \big | \mathbb { E } _ { F } ( Y ^ { 2 } ) - \mathbb { E } _ { G } ( Y ^ { 2 } ) \big | \le 2 M D d _ { \infty } ( F , G ) . } \end{array}
$$

Moreover,

$$
\begin{array} { r } { \left| \mu ( F ) ^ { 2 } - \mu ( G ) ^ { 2 } \right| \leq 2 M \left| \mu ( F ) - \mu ( G ) \right| \leq 2 M D d _ { \infty } ( F , G ) , } \end{array}
$$

which proves the stated mean-variance bound. For lower-tail CVaR,

$$
\begin{array} { r } { \vert \mathrm { C V a R } _ { \tau } ( F ) - \mathrm { C V a R } _ { \tau } ( G ) \vert \le \tau ^ { - 1 } W _ { 1 } ( F , G ) . } \end{array}
$$

For the quantile statement, let $\delta = d _ { \infty } ( F , G )$ and $q _ { \tau } = F ^ { - 1 } ( \tau )$ . The assumed bound on $\delta$ ensures that $\tau - \delta , \tau + \delta \in ( 0 , 1 )$ and $\delta / c _ { \tau } < r _ { \tau }$ . The generalized inverse inequalities give

$$
F ^ { - 1 } ( \tau - \delta ) \leq G ^ { - 1 } ( \tau ) \leq F ^ { - 1 } ( \tau + \delta )
$$

. Since $F ( q _ { \tau } ) = \tau$ and $f \geq c _ { \tau }$ almost everywhere on the specified neighborhood,

$$
F ( q _ { \tau } - s ) \leq \tau - c _ { \tau } s , \qquad F ( q _ { \tau } + s ) \geq \tau + c _ { \tau } s , \qquad 0 \leq s \leq r _ { \tau } .
$$

Taking $s = \delta / c _ { \tau }$ therefore yields

$$
q _ { \tau } - \frac { \delta } { c _ { \tau } } \leq F ^ { - 1 } ( \tau - \delta ) \leq G ^ { - 1 } ( \tau ) \leq F ^ { - 1 } ( \tau + \delta ) \leq q _ { \tau } + \frac { \delta } { c _ { \tau } } ,
$$

which proves the claim.

For CPT, write

$$
p _ { F } ^ { + } ( z ) = P _ { F } \{ v ^ { + } ( Y ) > z \} , \qquad p _ { F } ^ { - } ( z ) = P _ { F } \{ v ^ { - } ( Y ) > z \} ,
$$

and define $p _ { G } ^ { + } ( z )$ and $p _ { G } ^ { - } ( z )$ analogously. Since v is nondecreasing, each event appearing above is an upper or lower threshold event in Y. Hence, using CDF values or their left limits as appropriate,

$$
| p _ { F } ^ { \pm } ( z ) - p _ { G } ^ { \pm } ( z ) | \le d _ { \infty } ( F , G ) = \delta .
$$

This remains true when v has flat regions.

By the Hölder continuity of the probability distortions,

$$
| w _ { \pm } ( p _ { F } ^ { \pm } ( z ) ) - w _ { \pm } ( p _ { G } ^ { \pm } ( z ) ) | \leq L _ { \pm } \delta ^ { \rho \pm } .
$$

Therefore, using the gain-minus-loss tail-integral representation,

$$
\begin{array} { r l } & { | \mathcal { T } _ { \mathrm { C P T } } ( F ) - \mathcal { T } _ { \mathrm { C P T } } ( G ) | \leq \displaystyle \int _ { 0 } ^ { M _ { + } } | w _ { + } ( p _ { F } ^ { + } ( z ) ) - w _ { + } ( p _ { G } ^ { + } ( z ) ) | d z } \\ & { \qquad + \displaystyle \int _ { 0 } ^ { M _ { - } } | w _ { - } ( p _ { F } ^ { - } ( z ) ) - w _ { - } ( p _ { G } ^ { - } ( z ) ) | d z } \\ & { \leq L _ { + } M _ { + } \delta ^ { \rho _ { + } } + L _ { - } M _ { - } \delta ^ { \rho _ { - } } . } \end{array}
$$

Letting $\rho = \operatorname* { m i n } ( \rho _ { + } , \rho _ { - } )$ , and noting that $0 \leq \delta \leq 1$ , we have $\delta ^ { \rho _ { + } } \leq \delta ^ { \rho } , \delta ^ { \rho _ { - } } \leq \delta ^ { \rho }$ . Thus,

$$
| \mathcal { T } _ { \mathrm { C P T } } ( F ) - \mathcal { T } _ { \mathrm { C P T } } ( G ) | \leq ( L _ { + } M _ { + } + L _ { - } M _ { - } ) \delta ^ { \rho } ,
$$

which proves the claim.

Lemma 1 (Hölder continuity of the standard CPT distortion). For the commonly used Tversky– Kahneman distortion

$$
w _ { \gamma } ( p ) = \frac { p ^ { \gamma } } { \{ p ^ { \gamma } + ( 1 - p ) ^ { \gamma } \} ^ { 1 / \gamma } } , \qquad 0 < \gamma < 1 ,
$$

the endpoint behavior is Hölder. With gain and loss exponents $\gamma _ { + }$ and $\cdot \gamma _ { - } ,$ , respectively, Proposition 1 uses $\rho _ { T } = \operatorname* { m i n } ( \gamma _ { + } , \gamma _ { - } )$ . More generally, for every $0 < \gamma < 1$ , there is a finite $C _ { \gamma }$ such that

$$
| w _ { \gamma } ( p ) - w _ { \gamma } ( q ) | \leq C _ { \gamma } | p - q | ^ { \gamma } , \qquad p , q \in [ 0 , 1 ] .
$$

Proof. Let $D _ { \gamma } ( p ) = p ^ { \gamma } + ( 1 - p ) ^ { \gamma }$ . Because $D _ { \gamma } ( p ) \geq 1$ , direct differentiation on (0, 1) gives

$$
w _ { \gamma } ^ { \prime } ( p ) = D _ { \gamma } ( p ) ^ { - 1 / \gamma - 1 } \left[ ( \gamma - 1 ) p ^ { 2 \gamma - 1 } + \gamma p ^ { \gamma - 1 } ( 1 - p ) ^ { \gamma } + p ^ { \gamma } ( 1 - p ) ^ { \gamma - 1 } \right] .
$$

Since $0 < p < 1$ , the triangle inequality therefore gives

$$
\left. w _ { \gamma } ^ { \prime } ( p ) \right. \leq p ^ { \gamma - 1 } + ( 1 - p ) ^ { \gamma - 1 } .
$$

For $0 \leq p < q \leq 1$ , integration therefore yields

$$
| w _ { \gamma } ( q ) - w _ { \gamma } ( p ) | \leq \frac { 1 } { \gamma } \left[ q ^ { \gamma } - p ^ { \gamma } + ( 1 - p ) ^ { \gamma } - ( 1 - q ) ^ { \gamma } \right] .
$$

The subadditivity of $t ^ { \gamma }$ implies that each bracketed difference is at most $( q - p ) ^ { \gamma }$ . Thus the claimed bound holds with $C _ { \gamma } = 2 \bar { / } \gamma \mathrm { o n } ( 0 , 1 )$ , and continuity of $w _ { \gamma }$ extends it to the endpoints. □

Remark 1 (Unbounded outcomes). For Gaussian, lognormal, or other unbounded outcomes, the bounded-support statements in Proposition 1 require either outcome clipping that is also used by the algorithm, or explicit tail and moment assumptions with a truncation remainder. Alternatively, a criterion-matched transport metric may be used: $W _ { 1 }$ controls the mean and lower-tail CVaR under the required integrability, whereas variance needs second-moment control (for example through W<sub>2</sub> plus uniform moment conditions), and unbounded CPT needs conditions matched to the growth ofits value function. CDF sup-norm convergence alone cannot control a mean when a vanishing amount of mass may move arbitrarily far into the tail. Consequently, the bounded-support CDF theory does not by itselfcover the Gaussian or lognormal simulations.

## A.2 Proofs of the Main Theoretical Results

We first establish a pointwise oracle inequality for the plug-in treatment rule, which directly yields Theorem 1 and serves as the key step in proving Theorem 2.

Lemma 2 (Pointwise oracle inequality). For every x,

$$
0 \leq Q _ { \star } ^ { \mathcal { T } } ( x ) - Q _ { \widehat { \pi } _ { \mathcal { T } } ( x ) } ^ { \mathcal { T } } ( x ) \leq 2 e _ { n } ^ { \mathcal { T } } ( x ) .
$$

This statement permits arbitrary ties among optimal treatments.

Proof. Choose any $a ^ { \star } \in$ arg max<sub>a</sub> $Q _ { a } ^ { \mathcal { T } } ( x )$ . Since $\widehat { Q } _ { \widehat { \pi } \tau ( x ) } ^ { T } ( x ) \geq \widehat { Q } _ { a ^ { \star } } ^ { T } ( x )$

$$
\begin{array} { r l } & { Q _ { \star } ^ { T } ( x ) - Q _ { \widehat { \pi } _ { T } ( x ) } ^ { \mathcal { T } } ( x ) = Q _ { a ^ { \star } } ^ { T } ( x ) - Q _ { \widehat { \pi } _ { T } ( x ) } ^ { \mathcal { T } } ( x ) } \\ & { \qquad \leq \left| \widehat { Q } _ { a ^ { \star } } ^ { T } ( x ) - Q _ { a ^ { \star } } ^ { T } ( x ) \right| + \left| \widehat { Q } _ { \widehat { \pi } _ { T } ( x ) } ^ { \mathcal { T } } ( x ) - Q _ { \widehat { \pi } _ { T } ( x ) } ^ { \mathcal { T } } ( x ) \right| } \\ & { \qquad \leq 2 e _ { n } ^ { \mathcal { T } } ( x ) . } \end{array}
$$

ProofofTheorem 1. By the definition of policy regret and Lemma 2,

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) = \mathbb { E } _ { X } \left[ Q _ { \star } ^ { \mathcal { T } } ( X ) - Q _ { \widehat { \pi } _ { \mathcal { T } } ( X ) } ^ { \mathcal { T } } ( X ) \right] \leq 2 e _ { n , \infty } ^ { \mathcal { T } } .
$$

Under Assumption 2, on the event $\delta _ { n , \infty } ^ { \mathcal { T } } \leq \epsilon \tau , e _ { n , \infty } ^ { \mathcal { T } } \leq L \tau ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho \tau }$ . Combining the two inequalities gives $\mathcal { R } _ { T } ( \widehat { \pi } _ { T } ) \leq 2 L \tau ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho _ { \mathcal { T } } }$ , as claimed. □

The following lemma gives a general transfer from pointwise criterion estimation error to policy regret under the local boundary-mass condition. It serves as the common basis for both Theorem 2 and the integrated-error extension below.

Lemma 3 (General margin transfer). Under Assumption 3, for every realization of the training sample and every $0 < t \leq t _ { 0 }$

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) \leq t \Psi _ { \mathcal { T } } ( t ) + 2 \mathbb { E } _ { X } \left[ e _ { n } ^ { \mathcal { T } } ( X ) \mathbf { 1 } \{ e _ { n } ^ { \mathcal { T } } ( X ) > t / 2 \} \right] .\tag{1}
$$

Proof. Let

$$
G _ { n } ( X ) = Q _ { \star } ^ { \mathcal { T } } ( X ) - Q _ { \widehat { \pi } _ { \mathcal { T } } ( X ) } ^ { \mathcal { T } } ( X ) .
$$

On $\left\{ 0 < G _ { n } \leq t \right\}$ , the selected treatment is strictly suboptimal and $\Delta _ { \mathcal { T } } ( X ) \leq G _ { n } ( X ) \leq t .$ . Hence

$$
\begin{array} { r } { \mathbb { E } [ G _ { n } { \bf 1 } \{ 0 < G _ { n } \leq t \} ] \leq t \mathbb { P } \{ \Delta _ { \mathcal { T } } ( X ) \leq t \} \leq t \Psi _ { \mathcal { T } } ( t ) . } \end{array}
$$

On $\{ G _ { n } > t \}$ , Lemma 2 implies $G _ { n } \leq 2 e _ { n } ^ { \tau } ( X )$ and $e _ { n } ^ { \tau } ( X ) > t / 2$ . Adding the two pieces proves (1). □

ProofofTheorem 2. Under Assumption 2, on the event $\delta _ { n , \infty } ^ { \mathcal { T } } \leq \epsilon \tau , e _ { n , \infty } ^ { \mathcal { T } } \leq L \tau ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho \tau }$

I $\mathrm { f } \delta _ { n , \infty } ^ { \mathcal { T } } = 0$ , Theorem 1 gives zero regret directly. Otherwise, apply Lemma 3 with $t = 2 L _ { \mathcal { T } } ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho _ { \mathcal { T } } }$ By the assumed condition $2 L \tau ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho _ { \tau } } \leq t _ { 0 } .$ , this is an admissible choice of t. Moreover, $e _ { n } ^ { \mathcal { T } } ( X ) \leq$ $e _ { n , \infty } ^ { \mathcal { T } } \leq L \tau ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho _ { \tau } } = t / 2$ for every X. Hence, 1 $\{ e _ { n } ^ { \tau } ( X ) > t / 2 \} = 0 .$ , and the second term in Lemma 3 vanishes. Therefore, $\mathcal { R } _ { T } ( \widehat { \pi } _ { \mathcal { T } } ) \leq 2 L _ { T } ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho _ { \mathcal { T } } } \Psi _ { \mathcal { T } } \left\{ 2 \dot { L } _ { \mathcal { T } } ( \delta _ { n , \infty } ^ { \mathcal { T } } ) ^ { \rho _ { \mathcal { T } } } \right\}$ , which proves Theorem 2. □

The main-text results connect uniform conditional distribution estimation error to policy regret through the Hölder stability condition. Independently of this distribution-to-score transfer, Lemma 3 also yields regret guarantees directly from integrated criterion estimation errors. We record these score-level consequences under the familiar polynomial special case of the boundary-mass condition.

Corollary 1 (Polynomial margin and $L _ { q }$ score errors). Suppose that, for $0 < t \leq t _ { 0 }$

$$
\Psi \tau ( t ) \leq C _ { \Delta } t ^ { \kappa } , \qquad \kappa \geq 0 .\tag{2}
$$

Then the following statements hold.

1. For $1 < q < \infty ,$ , with $m _ { n , q } = \| e _ { n } ^ { T } \| _ { L _ { q } ( P _ { X } ) } ,$

$$
\begin{array} { r } { \mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) \leq C _ { q , \kappa , C _ { \Delta } } m _ { n , q } ^ { q ( 1 + \kappa ) / ( q + \kappa ) } } \end{array}\tag{3}
$$

whenever $m _ { n , q }$ is sufficiently small. Consequently, $m _ { n , q } = O _ { \mathbb { P } } ( \rho _ { n } )$ implies

$$
\begin{array} { r } { \mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) = O _ { \mathbb { P } } \Big ( \rho _ { n } ^ { q ( 1 + \kappa ) / ( q + \kappa ) } \Big ) . } \end{array}
$$

2. $I f 2 m _ { n , \infty } \leq t _ { 0 }$ and $m _ { n , \infty } = \left\| e _ { n } ^ { \mathcal { T } } \right\| _ { L _ { \infty } ( P _ { X } ) } = e _ { n , \infty } ^ { \mathcal { T } } ,$ , then

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) \leq 2 m _ { n , \infty } \Psi _ { \mathcal { T } } ( 2 m _ { n , \infty } ) \leq 2 ^ { 1 + \kappa } C _ { \Delta } m _ { n , \infty } ^ { 1 + \kappa } .\tag{4}
$$

3. With only an $L _ { 1 } ( P _ { X } )$ score bound, the always-valid conclusion remains

$$
\begin{array} { r } { \mathcal { R } _ { T } ( \widehat { \pi } _ { T } ) \leq 2 m _ { n , 1 } . } \end{array}
$$

Proof. For $1 < q < \infty$ , Hölder’s inequality followed by Markov’s inequality gives

$$
\mathbb { E } [ e _ { n } ^ { \mathcal { T } } ( X ) \mathbf { 1 } \{ e _ { n } ^ { \mathcal { T } } ( X ) > t / 2 \} ] \le 2 ^ { q - 1 } m _ { n , q } ^ { q } t ^ { 1 - q } .
$$

Thus Lemma 3 gives

$$
\mathcal { R } \tau ( \widehat { \pi } _ { \mathcal { T } } ) \leq C _ { \Delta } t ^ { 1 + \kappa } + 2 ^ { q } m _ { n , q } ^ { q } t ^ { 1 - q } .
$$

Taking t proportional to $m _ { n , q } ^ { q / ( q + \kappa ) }$ proves (3). For $q = \infty$ , a strictly suboptimal selection implies both $G _ { n } ( X ) \leq 2 m _ { n , \infty }$ and $\bar { \Delta _ { T } } ( X ) \leq 2 m _ { n , \infty }$ , yielding (4). □

Remark 2 (Interpretation and limiting cases). When $\kappa = 0 ,$ , the polynomial condition supplies no rate acceleration, and all displays reduce to thefirst-stage score rate. Under a hard gap, $\Psi \tau ( t ) = 0 f o r$ $t < t _ { 0 } ,$ , a uniformly accurate plug-in rule has zero regret once $2 m _ { n , \infty } < t _ { 0 }$ . Without any restriction on mass near the decision boundary, a score perturbation ofsize ε may change the selected treatment on a region with gaps of order ε, so a general improvement over order ε is impossible. This is the same role played by low-noise assumptions in classical plug-in classification [35].

## A.3 Estimator-Specific Guarantees

## A.3.1 A generic weighted conditional-CDF estimator

Both kernel smoothers and forest estimators can be written as weighted empirical CDFs. For a treatment $^ { a , }$ let ${ \mathcal { T } } _ { a } = \{ i : A _ { i } = a \}$ , and consider

$$
\widehat { F } _ { a , W } ( y \mid x ) = \sum _ { i \in \mathbb { Z } _ { a } } W _ { i , a } ( x ) \mathbf { 1 } ( Y _ { i } \leq y ) , \qquad W _ { i , a } ( x ) \geq 0 , \qquad \sum _ { i \in \mathbb { Z } _ { a } } W _ { i , a } ( x ) = 1 .\tag{5}
$$

This definition is used on $\mathcal { E } _ { n , A } = \{ \mathcal { T } _ { a } \neq \emptyset$ for all $a \}$ ; outside that event, set each unavailable-arm estimate to a prespecified fixed CDF belonging to the domain of $\tau$ . Under overlap,

$$
\mathbb { P } ( \mathcal { E } _ { n , A } ^ { c } ) \leq 2 ( 1 - \eta ) ^ { n } \longrightarrow 0 .
$$

All weighted-estimator statements below are proved on $\mathcal { E } _ { n , A }$ . For bookkeeping, set the weights, localization bias, and squared weight mass to arbitrary bounded values off this event. Its exponentially small probability leaves the unconditional $O _ { \mathbb { P } }$ conclusions unchanged. Let ${ \mathcal { H } } _ { n }$ contain all covariates, treatment labels, algorithmic randomization, structure-sample observations, and other information used to construct the weights, but not the estimation outcomes appearing in (5).

Assumption 4 (Global honesty for the estimation outcomes). Conditionally on $\mathcal { H } _ { n } ,$ , the weights in (5) arefixed, and the estimation outcomes appearing in that equation are independent with

$$
{ \mathbb E } [ { \bf 1 } ( Y _ { i } \leq y ) \mid { \mathcal H } _ { n } ] = F _ { a } ( y \mid X _ { i } ) , \qquad i \in { \mathcal L } _ { a } .
$$

For kernels, Assumption 4 holds because the weights depend only on $X _ { i }$ and $A _ { i }$ . For forests, it holds for a global sample-splitting construction in which a structure sample builds all partitions and an independent estimation sample supplies the weighted outcomes. Standard per-tree honesty does not automatically imply Assumption 4 for the final aggregate weights, because one observation can be used for splitting in some trees and for leaf estimation in others. This fixed-weight conditionalindependence requirement is therefore stronger than the per-tree honesty used in generalized forest constructions [19].

Define the oracle weighted target, localization bias, squared weight mass, and effective sample size:

$$
F _ { a , W } ( y \mid x ) = \sum _ { i \in \mathcal { T } _ { a } } W _ { i , a } ( x ) F _ { a } ( y \mid X _ { i } ) ,
$$

$$
B _ { a , n } ( x ) = \sum _ { i \in \cal { Z } _ { a } } W _ { i , a } ( x ) \operatorname* { s u p } _ { y } | F _ { a } ( y \mid X _ { i } ) - F _ { a } ( y \mid x ) | ,
$$

$$
S _ { a , n } ( x ) = \sum _ { i \in \mathcal { Z } _ { a } } W _ { i , a } ^ { 2 } ( x ) , \qquad n _ { \mathrm { e f f } , a } ( x ) = S _ { a , n } ^ { - 1 } ( x ) .
$$

The exact decomposition

$$
\widehat { F } _ { a , W } - F _ { a } = ( \widehat { F } _ { a , W } - F _ { a , W } ) + ( F _ { a , W } - F _ { a } )\tag{6}
$$

separates stochastic fluctuation from localization bias.

Proposition 2 (Fixed-query weighted-CDF concentration). On $\mathcal { E } _ { n , A }$ , under Assumption 4, for every fixed a, x and everyfinite $1 \leq r < \infty ,$ , there is a constant $C _ { r } < \infty$ such that

$$
\left( \mathbb { E } \bigg [ \operatorname* { s u p } _ { y } \Big | \widehat { F } _ { a , W } ( y \mid x ) - F _ { a , W } ( y \mid x ) \Big | ^ { r } \bigg | \mathcal { H } _ { n } \bigg ] \right) ^ { 1 / r } \leq C _ { r } S _ { a , n } ^ { 1 / 2 } ( x ) .\tag{7}
$$

Consequently,

$$
\operatorname* { s u p } _ { y } \Big | \widehat { F } _ { a , W } ( y \mid x ) - F _ { a } ( y \mid x ) \Big | = O _ { \mathbb { P } } \Big ( B _ { a , n } ( x ) + S _ { a , n } ^ { 1 / 2 } ( x ) \Big )\tag{8}
$$

conditionally on the weight construction.

Proof. The bias part of (6) is bounded by $B _ { a , n } ( x )$ . For the stochastic part, write

$$
Z _ { i } ( y ) = \mathbf { 1 } ( Y _ { i } \leq y ) - F _ { a } ( y \mid X _ { i } ) .
$$

Conditional $L _ { r }$ symmetrization [40] gives

$$
\left\| \operatorname* { s u p } _ { y } \left| \sum _ { i } W _ { i , a } ( x ) Z _ { i } ( y ) \right| \right\| _ { L _ { r } ( \mathbb { P } | \mathcal { H } _ { n } ) } \leq 2 \left\| \operatorname* { s u p } _ { y } \left| \sum _ { i } W _ { i , a } ( x ) \epsilon _ { i } \mathbf { 1 } ( Y _ { i } \leq y ) \right| \right\| _ { L _ { r } ( \mathbb { P } \times \mathbb { P } _ { \epsilon } | \mathcal { H } _ { n } ) } ,
$$

where the $\epsilon _ { i } { ' } \mathrm { { s } }$ are independent Rademacher variables. Conditional on the realized outcomes, sort them increasingly. Threshold sets are prefixes of this ordering (ties only reduce the collection), so the last supremum is bounded by the maximum absolute partial sum of the independent increments $W _ { ( i ) , a } ( \boldsymbol { x } ) \bar { \boldsymbol { \epsilon } } _ { ( i ) }$ . For $r > 1$ , Doob’s maximal inequality followed by the Khintchine inequality bounds its $L _ { r }$ norm by $C _ { r } \{ \sum _ { i } W _ { i , a } ^ { 2 } ( x ) \} ^ { 1 / 2 }$ ; for $r = 1$ , use the $L _ { 2 }$ bound and Jensen’s inequality. This proves (7), and (8) follows from (6). □

Theorem 3 (Integrated weighted-CDF error). Let $X \sim P _ { X }$ be independent of the training data, let

$$
\delta _ { n , W } ( x ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \operatorname* { s u p } _ { y } \left| \widehat { F } _ { a , W } ( y \mid x ) - F _ { a } ( y \mid x ) \right| ,
$$

and, for finite $1 \leq r < \infty ,$ define

$$
b _ { n , r } = \left. \underset { a \in A } { \operatorname* { m a x } } B _ { a , n } \right. _ { L _ { r } ( P _ { X } ) } , \qquad \quad \qquad s _ { n , r } = \left. \underset { a \in A } { \operatorname* { m a x } } S _ { a , n } ^ { 1 / 2 } \right. _ { L _ { r } ( P _ { X } ) } .
$$

If

$$
b _ { n , r } = O _ { \mathbb { P } } ( \bar { b } _ { n , r } ) , \qquad s _ { n , r } = O _ { \mathbb { P } } ( \bar { s } _ { n , r } ) ,
$$

then, under Assumption 4,

$$
\lVert \delta _ { n , W } \rVert _ { L _ { r } ( P _ { X } ) } = O _ { \mathbb { P } } ( \bar { b } _ { n , r } + \bar { s } _ { n , r } ) .\tag{9}
$$

Proof. Apply Proposition 2 conditionally at each x, integrate its r-th moment with respect to $P _ { X }$ and use the finite-action inequality ma $\begin{array} { r } { \mathrm { x } _ { a } \ : z _ { a } ^ { r } \leq \sum _ { a } z _ { a } ^ { r } } \end{array}$ . Conditional Markov’s inequality yields the stochastic order in (9); the localization bias is handled by the triangle inequality. □

Corollary 2 (Weighted estimator regret). Suppose Assumption 2 holds with $d _ { T } = d _ { \infty }$ . Assume additionally that there exists $M \tau < \infty$ such that

$$
e _ { n } ^ { \mathcal { T } } ( x ) \leq M _ { \mathcal { T } }
$$

uniformly in $x ,$ almost surely. Assume that all rate sequences used below converge to zero. Let $1 \leq q < \infty$ satisfy $\rho \tau q \geq 1$ , and apply Theorem 3 with $r = \rho \tau q .$ . Then

$$
\| e _ { n } ^ { \mathcal { T } } \| _ { L _ { q } ( P _ { X } ) } = O _ { \mathbb { P } } \big [ \{ \bar { b } _ { n , \rho _ { \tau } q } + \bar { s } _ { n , \rho _ { \tau } q } \} ^ { \rho _ { \tau } } \big ] .\tag{10}
$$

Without a margin condition, applying Theorem 3 with $r = 1$ , Jensen’s inequality, and $0 \leq \delta _ { n , W } \leq 1$ gives

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) = O _ { \mathbb { P } } \big [ \{ \bar { b } _ { n , 1 } + \bar { s } _ { n , 1 } \} ^ { \rho _ { \mathcal { T } } } \big ] .\tag{11}
$$

Under (2) and $1 < q < \infty$

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ) = O _ { \mathbb { P } } \left[ \{ \bar { b } _ { n , \rho _ { \mathcal { T } } q } + \bar { s } _ { n , \rho _ { \mathcal { T } } q } \} ^ { \rho _ { \mathcal { T } } q ( 1 + \kappa ) / ( q + \kappa ) } \right] .\tag{12}
$$

A uniform-in-x analogue requires a separate entropy or complexity condition for $\{ W _ { a } ( x ) : x \in { \mathcal { X } } \}$ and does notfollow by setting $q = \infty$ in thefinite-r integrated theorem.

Proof. Write $\delta _ { n } = \delta _ { n , W }$ . On $\{ \delta _ { n } ( X ) \leq \epsilon \tau \}$ , Assumption 2 gives

$$
e _ { n } ^ { \tau } ( X ) \leq L \tau \delta _ { n } ( X ) ^ { \rho \tau } .
$$

On the complementary event, use $e _ { n } ^ { \boldsymbol { \tau } } ( \boldsymbol { X } ) \leq M \tau$ . Therefore, whenever $\rho \tau q \geq 1$

$$
\begin{array} { r l } & { \left\| e _ { n } ^ { \mathcal { T } } \right\| _ { L _ { q } ( P _ { X } ) } ^ { q } \leq L _ { \mathcal { T } } ^ { q } \mathbb { E } _ { X } [ \delta _ { n } ( X ) ^ { \rho \tau q } ] + M _ { \mathcal { T } } ^ { q } \mathbb { P } _ { X } \{ \delta _ { n } ( X ) > \epsilon _ { \mathcal { T } } \} } \\ & { \qquad \leq \left( L _ { \mathcal { T } } ^ { q } + M _ { \mathcal { T } } ^ { q } \epsilon _ { \mathcal { T } } ^ { - \rho \tau q } \right) \| \delta _ { n } \| _ { L _ { \rho \tau q } ( P _ { X } ) } ^ { \rho \tau q } . } \end{array}
$$

This proves (10). Moreover, the pointwise oracle inequality, Jensen’s inequality, and Markov’s inequality give

$$
\begin{array} { r l } & { { \mathcal R } _ { \mathcal T } ( \widehat { \pi } _ { \mathcal T } ) \leq 2 \left\| e _ { n } ^ { \mathcal T } \right\| _ { L _ { 1 } ( P _ { X } ) } } \\ & { \qquad \leq 2 L _ { \mathcal T } \left\| \delta _ { n } \right\| _ { L _ { 1 } ( P _ { X } ) } ^ { \rho _ { \mathcal T } } + 2 M _ { \mathcal T } \epsilon _ { \mathcal T } ^ { - 1 } \left\| \delta _ { n } \right\| _ { L _ { 1 } ( P _ { X } ) } , } \end{array}
$$

which proves (11). Finally, combining (10) with Corollary 1 proves (12).

## A.3.2 Kernel conditional-CDF estimation

For $K _ { h } ( u ) = h ^ { - p } K ( u / h )$ , consider the local-constant estimator

$$
{ \widehat { F } } _ { a } ^ { K } ( y \mid x ) = { \frac { \sum _ { i = 1 } ^ { n } K _ { h } ( X _ { i } - x ) \mathbf { 1 } ( A _ { i } = a ) \mathbf { 1 } ( Y _ { i } \leq y ) } { \sum _ { i = 1 } ^ { n } K _ { h } ( X _ { i } - x ) \mathbf { 1 } ( A _ { i } = a ) } } .\tag{13}
$$

If the denominator is zero, define the estimate to be a prespecified fixed CDF belonging to the domain of T . The proof below shows that this fallback is unused uniformly on $\mathcal { X } _ { 0 }$ with probability tending to one.

Assumption 5 (Uniform kernel conditions). The following conditions hold.

1. $\mathcal { X } _ { 0 } ~ \subset ~ \mathcal { X }$ is compact and $\mathbb { P } _ { X } ( X \in \mathcal { X } _ { 0 } ) = 1$ . The support is regular enough that the population kernel denominator is uniformly positive as specified below.

2. For each a, the joint design density $g _ { a } ( x ) = f _ { X } ( x ) e _ { a } ( x )$ is bounded above, and, for some $c _ { D } , h _ { 0 } > 0$

$$
\operatorname* { i n f } _ { a \in A , x \in \mathcal { X } _ { 0 } \atop 0 < h \leq h _ { 0 } } \int K ( v ) g _ { a } ( x + h v ) \mathbf { 1 } ( x + h v \in \mathcal { X } _ { 0 } ) d v \geq c _ { D } .
$$

A density bounded away from zero on a regular compact support is a standard sufficient condition.

3. K is bounded, nonnegative, compactly supported, $\textstyle \int K ( u ) d u = 1$ , and its translates and dilationsform a VC-type class. The corresponding product classes

$$
( X , A , Y ) \mapsto K _ { h } ( X - x ) \mathbf { 1 } ( A = a ) \mathbf { 1 } ( Y \leq y )
$$

and the denominator class without ${ \bf 1 } ( Y \le y )$ are also assumed VC-type uniformly over $( a , x , y , h )$ . This follows from standard permanence properties for product kernels of bounded variation and the threshold class.

4. For some $0 < \beta \leq 1$ and $L _ { F } < \infty ,$

$$
\operatorname* { s u p } _ { a \in A } \operatorname* { s u p } _ { y \in \mathbb { R } } | F _ { a } ( y \mid x ) - F _ { a } ( y \mid x ^ { \prime } ) | \leq L _ { F } \left\| x - x ^ { \prime } \right\| _ { 2 } ^ { \beta }\tag{14}
$$

for all $x , x ^ { \prime } \in \mathcal { X } _ { 0 }$

$$
5 . \ h = h _ { n } \to 0 \ a n d n h _ { n } ^ { p } / \log n \to \infty .
$$

The condition $\beta \leq 1$ corresponds to the local-constant estimator in (13). Exploiting smoothness beyond first order requires an appropriate higher-order or local-polynomial conditional-distribution estimator and its own shape-validity argument.

Theorem 4 (Uniform kernel conditional-CDF rate). Under Assumptions 1 and ${ } ^ { 5 , }$

$$
\operatorname* { m a x } _ { a \in \mathcal { A } } \operatorname* { s u p } _ { x \in \mathcal { X } _ { 0 } } \operatorname* { s u p } _ { y \in \mathbb { R } } \Big | \widehat { F } _ { a } ^ { K } ( y \mid x ) - F _ { a } ( y \mid x ) \Big | = O _ { \mathbb { P } } \left( h _ { n } ^ { \beta } + \sqrt { \frac { \log n } { n h _ { n } ^ { p } } } \right) .\tag{15}
$$

Proof. For each $^ { a , }$ write the numerator and denominator of (13), divided by $n ,$ as $N _ { n , a } ( x , y )$ and $D _ { n , a } ( x )$ . Their population counterparts are

$$
\begin{array} { l } { \displaystyle \bar { N } _ { a , h } ( x , y ) = \int _ { x + h v \in \mathcal { X } _ { 0 } } K ( v ) g _ { a } ( x + h v ) F _ { a } ( y \mid x + h v ) d v , } \\ { \displaystyle \bar { D } _ { a , h } ( x ) = \int _ { x + h v \in \mathcal { X } _ { 0 } } K ( v ) g _ { a } ( x + h v ) d v . } \end{array}
$$

The design-density lower bound and nonnegativity of K imply $\mathrm { i n f } _ { a , x } \bar { D } _ { a , h } ( x ) > c > 0$ . Moreover, (14) gives, uniformly in $a , x , y$

$$
\left| \bar { N } _ { a , h } ( x , y ) / \bar { D } _ { a , h } ( x ) - F _ { a } ( y \mid x ) \right| \le C h ^ { \beta } .
$$

The VC-type assumption and the bounded, compactly supported kernel imply the uniform empirical process bounds

$$
\begin{array} { r } { \displaystyle \operatorname* { s u p } _ { a , x , y } \left| N _ { n , a } ( x , y ) - \bar { N } _ { a , h } ( x , y ) \right| = O _ { \mathbb { P } } \left( \sqrt { \frac { \log n } { n h ^ { p } } } \right) , } \\ { \displaystyle \operatorname* { s u p } _ { a , x } \left| D _ { n , a } ( x ) - \bar { D } _ { a , h } ( x ) \right| = O _ { \mathbb { P } } \left( \sqrt { \frac { \log n } { n h ^ { p } } } \right) . } \end{array}
$$

More explicitly, the corresponding maximal inequality also contributes an envelope term of order log $n / ( n h ^ { p } )$ , which is dominated by the displayed square-root term under $n h ^ { p } / \log n \to \infty$ . These are standard uniform-in-bandwidth kernel empirical-process bounds [41]. The second display and $n h ^ { p } / \log n \to \infty$ ensure that the empirical denominator is uniformly bounded away from zero with probability tending to one. A ratio expansion and the uniform bias bound prove (15). This is the standard bias-stochastic structure for kernel conditional-distribution estimation [17]. □

Corollary 3 (Kernel score and regret rates). Suppose Assumption 2 holds with $d _ { T } = d _ { \infty }$ . Let

$$
r _ { n , K } = h _ { n } ^ { \beta } + \sqrt { \frac { \log n } { n h _ { n } ^ { p } } } .
$$

Then

$$
\begin{array} { r l } {  { \| e _ { n } ^ { \mathcal { T } } \| _ { L _ { \infty } ( P _ { X } ) } = O _ { \mathbb { P } } ( r _ { n , K } ^ { \rho \tau } ) , } } \\ & { \quad \mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ^ { K } ) = O _ { \mathbb { P } } ( r _ { n , K } ^ { \rho \tau } ) \quad w i t h o u t a m a r g i n c o n d i t i o n . } \end{array}
$$

Under (2),

$$
\begin{array} { r } { \mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ^ { K } ) = O _ { \mathbb { P } } ( r _ { n , K } ^ { \rho \tau ( 1 + \kappa ) } ) . } \end{array}
$$

The log-optimized bandwidth and resulting rate are

$$
h _ { n } \asymp \left( \frac { \log n } { n } \right) ^ { 1 / ( 2 \beta + p ) } ,
$$

$$
r _ { n , K } \asymp \left( \frac { \log n } { n } \right) ^ { \beta / ( 2 \beta + p ) } ,
$$

$$
\mathcal { R } _ { T } ( \widehat { \pi } _ { T } ^ { K } ) = O _ { \mathbb { P } } \left[ \left( \frac { \log n } { n } \right) ^ { \beta \rho _ { T } ( 1 + \kappa ) / ( 2 \beta + p ) } \right] .
$$

Remark 3 (An interior-only kernel bound). If Theorem 4 is proved only on $\mathcal { X } _ { 0 }$ but $\mathbb { P } ( X \notin \mathcal { X } _ { 0 } ) > 0 ,$ it does not by itself imply a population regret rate. If all scores are bounded by $M \tau$ , one may instead add the explicit remainder

$$
\mathcal { R } _ { T , \mathcal { X } _ { 0 } } ( \widehat \pi _ { T } ) : = \mathbb { E } _ { X } \left[ \{ Q _ { \star } ^ { \mathcal { T } } ( X ) - Q _ { \widehat \pi _ { T } ( X ) } ^ { \mathcal { T } } ( X ) \} \mathbf { 1 } ( X \in \mathcal { X } _ { 0 } ) \ | \ \mathcal { D } _ { n } \right] ,
$$

and

$$
\mathcal { R } _ { T } ( \widehat { \pi } _ { T } ) \leq \mathcal { R } _ { T , \chi _ { 0 } } ( \widehat { \pi } _ { T } ) + 2 M _ { T } \mathbb { P } ( X \notin \mathcal { X } _ { 0 } ) .
$$

For a sequence $\mathcal { X } _ { 0 , n } ,$ , the outside probability must be negligible relative to the claimed rate.

## A.3.3 Distributional Random Forest

A forest conditional-CDF estimator has the form

$$
\widehat { F } _ { a } ^ { F } ( y \mid x ) = \sum _ { i : A _ { i } = a } w _ { i , a } ( x ) { \bf 1 } ( Y _ { i } \leq y ) , \qquad w _ { i , a } ( x ) \geq 0 , \qquad \sum _ { i : A _ { i } = a } w _ { i , a } ( x ) = 1 .\tag{16}
$$

The formal result below covers treatment-specific fits, or a joint fit whose query at treatment a produces weights that are zero outside arm a and are normalized within that arm. A generic joint fit that assigns cross-arm weights requires a separate formulation on the joint (X, A) space. Once estimated, the treatment-specific distributions can be reused for every downstream criterion without refitting criterion-specific objectives.

Suppose a global structure sample constructs all partitions and an independent estimation sample supplies the outcomes in (16). Then Assumption 4 holds and Theorem 3 applies. In particular, define

$$
\begin{array} { l } { { \displaystyle B _ { a , n } ^ { F } ( x ) = \sum _ { i : A _ { i } = a } w _ { i , a } ( x ) \operatorname* { s u p } _ { y } \left| F _ { a } ( y \mid X _ { i } ) - F _ { a } ( y \mid x ) \right| , } } \\ { { \displaystyle S _ { a , n } ^ { F } ( x ) = \sum _ { i : A _ { i } = a } w _ { i , a } ^ { 2 } ( x ) } . } \end{array}
$$

If (14) holds, the more informative weighted localization bound is

$$
B _ { a , n } ^ { F } ( x ) \leq L _ { F } \sum _ { i : A _ { i } = a } w _ { i , a } ( x ) \left\| X _ { i } - x \right\| _ { 2 } ^ { \beta } .
$$

This is preferable to the maximum radius of all observations with positive weight, which can be driven by a negligible forest weight.

Corollary 4 (Integrated forest regret). Assume global sample splitting, binary treatment, CDFsupnorm criterion stability with exponent $\rho \tau$ , and the integrated weight controls below for the requiredfinite values ofr:

$$
\Bigl \| \operatorname* { m a x } _ { a } B _ { a , n } ^ { F } \Bigr \| _ { L _ { r } ( P _ { X } ) } = O _ { \mathbb { P } } ( \bar { b } _ { n , r } ) , \qquad \Bigl \| \operatorname* { m a x } _ { a } ( S _ { a , n } ^ { F } ) ^ { 1 / 2 } \Bigr \| _ { L _ { r } ( P _ { X } ) } = O _ { \mathbb { P } } ( \bar { s } _ { n , r } ) .
$$

Then Corollary 2 gives an explicit regret bound. In particular, for a Lipschitz criterion and $1 < q <$ $\infty ,$

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ^ { F } ) = O _ { \mathbb { P } } \Big [ \{ \bar { b } _ { n , q } + \bar { s } _ { n , q } \} ^ { q ( 1 + \kappa ) / ( q + \kappa ) } \Big ]
$$

under (2); the no-margin bound is (11) with theforest weight quantities.

The published DRF theory of Cevid et al.<sup>´</sup> [20] establishes a fixed-x polynomial rate for a conditional RKHS mean embedding under subsampling, honesty, regularity, symmetry, and random-splitting conditions. Renaming their subsample exponent to θ, if $s _ { n } ^ { \mathrm { ~ \textsc ~ { ~ } ~ } } \asymp n ^ { \theta ^ { \mathrm { ~ \tiny ~ } } }$ , their Theorem 2 gives

$$
\| \widehat { \mu } _ { n } ( x ) - \mu ( x ) \| _ { \mathcal { H } } = O _ { \mathbb { F } } ( n ^ { - \gamma _ { \mathrm { D R F } } } ) , \qquad \gamma _ { \mathrm { D R F } } = \frac { 1 } { 2 } \operatorname* { m i n } \left\{ 1 - \theta , \frac { \log \left\{ ( 1 - \alpha _ { \mathrm { r e g } } ) ^ { - 1 } \right\} } { \log \left( \alpha _ { \mathrm { r e g } } ^ { - 1 } \right) } \frac { \pi _ { \mathrm { s p l i t } } } { p } \theta \right\} ,
$$

for each fixed test point, under their assumptions on the design density, bounded kernel, conditional embedding, and forest construction. Here $\alpha _ { \mathrm { r e g } }$ and $\pi _ { \mathrm { s p l i t } }$ denote their tree-regularity and random-split constants. Their CDF and quantile corollaries establish consistency at appropriate continuity points. Those results do not directly provide a uniform-in- $( x , y )$ CDF sup-norm rate, nor do they establish automatic adaptation to an unknown intrinsic dimension.

Therefore, if the implementation uses standard per-tree honest DRF rather than the globally split construction above, define the pointwise criterion-matched distribution error by

$$
\delta _ { n , \mathrm { D R F } } ^ { \mathcal { T } } ( x ) : = \operatorname* { m a x } _ { a \in \cal A } d _ { \mathcal { T } } \left\{ \widehat { F } _ { a } ^ { \mathrm { D R F } } ( \cdot  { | } x ) , F _ { a } ( \cdot  { | } x ) \right\} .
$$

A defensible generic assumption, for some finite $1 \leq r < \infty .$ , is

$$
\begin{array} { r } { \left. \delta _ { n , \mathrm { D R F } } ^ { \mathcal { T } } \right. _ { L _ { r } ( P _ { X } ) } = O _ { \mathbb { P } } ( r _ { n , \mathrm { D R F } } ) , \qquad r _ { n , \mathrm { D R F } } \to 0 . } \end{array}
$$

This distribution-error condition does not by itself imply a regret rate. It must first be transferred to a score-error rate through the criterion-stability condition, after which Lemma 2 and Corollary 1 can be applied.

More explicitly, suppose Assumption 2 holds and that the bounded score-error condition used in Corollary 2 is satisfied. If $r \geq \rho \tau$ , the same good-set/bad-set argument gives

$$
\begin{array} { r } { \| e _ { n } ^ { \mathcal { T } } \| _ { L _ { r / \rho _ { \mathcal { T } } } ( P _ { X } ) } = O _ { \mathbb { P } } ( r _ { n , \mathrm { D R F } } ^ { \rho _ { \mathcal { T } } } ) . } \end{array}
$$

Consequently,

$$
\mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ^ { \mathrm { D R F } } ) = O _ { \mathbb { P } } ( r _ { n , \mathrm { D R F } } ^ { \rho _ { \mathcal { T } } } )
$$

without a margin condition. $\mathrm { I f } r > \rho \tau$ and (2) holds, then

$$
\begin{array} { r } { \mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ^ { \mathrm { D R F } } ) = O _ { \mathbb { P } } \Big ( r _ { n , \mathrm { D R F } } ^ { \rho _ { \mathcal { T } } r ( 1 + \kappa ) / ( r + \rho _ { \mathcal { T } } \kappa ) } \Big ) . } \end{array}
$$

A separately established genuine uniform bound

$$
\operatorname* { s u p } _ { x } \delta _ { n , \mathrm { D R F } } ^ { \mathcal { T } } ( x ) = O _ { \mathbb { P } } ( r _ { n , \mathrm { D R F } } )
$$

can instead be combined directly with Theorems 1–2. Under the polynomial margin condition, this gives

$$
\begin{array} { r } { \mathcal { R } _ { \mathcal { T } } ( \widehat { \pi } _ { \mathcal { T } } ^ { \mathrm { D R F } } ) = O _ { \mathbb { P } } \Big ( r _ { n , \mathrm { D R F } } ^ { \rho _ { \mathcal { T } } ( 1 + \kappa ) } \Big ) . } \end{array}
$$

The rate $r _ { n , \mathrm { D R F } }$ should not be replaced by a closed-form intrinsic-dimension expression unless that localization result is separately proved for the actual forest construction.

Remark 4 (Uniform forest bounds need weight-class complexity). Pointwise control through $B _ { a , n } ^ { F } ( x )$ and $S _ { a , n } ^ { F } ( x )$ does not automatically yield a supremum over x. A uniform result additionally requires a complexity boundfor the collection ofweight vectors indexed by x. For example, ifa conditional entropy or log-cardinality $\ell _ { n }$ controls that class, a typical high-level uniform CDF bound has the form

$$
\operatorname* { s u p } _ { a , x , y } \Big | \widehat { F } _ { a } ^ { F } ( y \mid x ) - F _ { a } ( y \mid x ) \Big | = O _ { \mathbb { P } } \Big ( b _ { n } + \sqrt { s _ { n } \ell _ { n } } \Big ) ,
$$

where $b _ { n }$ bounds the uniform localization bias and $s _ { n }$ bounds the uniform squared weight mass. Taking $\ell _ { n } = O ( \log n )$ requires a separate complexity argument; it does not follow from honesty alone.

## B Additional experiment details

## B.1 Compute resource

All simulations were run on UNC’s Longleaf cluster. Non-generative methods were fit in R 4.5.0 (drf 1.3.1, quantreg 6.1, DynTxRegime 4.16, glmnet 4.1-10) with 8 CPU cores and 32 GB of memory per job. Conditional DDPM was trained on a single NVIDIA A100 (40 GB) GPU with PyTorch 2.6.0 (CUDA 12.4), and GCDS was run with 4 CPU cores and 32 GB of memory using TensorFlow 2.16.2; both generative pipelines were driven from R 4.4.0.

The treatment-interactive experiment was run locally on macOS using R 4.4.1. The ACTG175 evaluation and runtime measurements were performed on an Apple M3 Pro using R 4.4.1, with numerical-library and DRF thread counts set to one.

## B.2 Full Experimental Results

Tables 6, 11, and 14 report all five criteria and all CQR and $\ell _ { 2 } { \mathrm { - C V a R } }$ training levels for the same 100 paired replications as the main tables. Higher-dimensional results with $p = 2 0 , 5 0 , 1 0 0 , 3 0 0$ are also reported below.

In Setting $1 , \ell _ { 1 } { \mathrm { - P L S } }$ has the highest Value and lowest Err among non-oracle methods for all reported criteria at $p = 2 0 , 5 0 , 1 0 0 , 3 0 0$ . Within the distribution-first family, Conditional DDPM performs best at $p = 2 0$ , whereas DRF performs best at $p = 5 0 , 1 0 0 , 3 0 0$ . In Setting 2, DRF has the highest Value and lowest Err for all five criteria at $p = 2 0 . { \mathrm { A t } } p = 3 0 0$ , DRF has the highest Mean Value, while $\ell _ { 1 } { \mathrm { - P L S } }$ has the lowest Err for all criteria and the highest quantile and CVaR Values. Within the distribution-first family at $p = 3 0 0$ , DRF has the highest Values and lowest mean-targeted Err, whereas GCDS has the lowest tail-targeted Err among its evaluable repetitions. All methods have 100 evaluable repetitions except GCDS in Setting $2 \mathrm { a t } p = 3 0 0$ , which has 99; the failure record and reporting protocol are described in Appendix B.5. Core code for the experiments is available at https: $/ / \mathsf { g } \dot { \mathbf { 1 } }$ thub.com/zhenf530/Distributio\_Risk\_ITR.

Table 6: Setting 1 (linear Gaussian; $n = 1 0 0 0 , p = 1 0 ) \colon$ full results for Mean, Quanti $\mathrm { l e } _ { 0 . 1 }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.810(0.002)</td><td>-0.042(0.004)</td><td>-0.699(0.004)</td><td>-0.015(0.003)</td><td>0.672(0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.805(0.002)</td><td>-0.050(0.004)</td><td>-0.712(0.005)</td><td>-0.024(0.004)</td><td>0.666(0.003)</td></tr><tr><td>Err</td><td>0.034(0.002)</td><td>0.034(0.002)</td><td>0.034(0.002)</td><td>0.034(0.002)</td><td>0.034(0.002)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.800(0.002)</td><td>-0.060(0.004)</td><td>−0.720(0.004)</td><td>-0.032(0.003)</td><td>0.658(0.003)</td></tr><tr><td>Err</td><td>0.051 (0.002)</td><td>0.051 (0.002)</td><td>0.051 (0.002)</td><td>0.051 (0.002)</td><td>0.051 (0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.799(0.002)</td><td>-0.063 (0.004)</td><td>-0.723 (0.004)</td><td>-0.035(0.003)</td><td>0.656(0.003)</td></tr><tr><td>Err</td><td>0.053(0.001)</td><td>0.053(0.001)</td><td>0.053(0.001)</td><td>0.053(0.001)</td><td>0.053(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.803 (0.003)</td><td>-0.057(0.004)</td><td>-0.717(0.004)</td><td>-0.029(0.003)</td><td>0.661 (0.002)</td></tr><tr><td>Err</td><td>0.043(0.001)</td><td>0.043(0.001)</td><td>0.043(0.001)</td><td>0.043(0.001)</td><td>0.043 (0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.805(0.002)</td><td>-0.054(0.004)</td><td>−0.712(0.004)</td><td>-0.025(0.003)</td><td>0.664(0.002)</td></tr><tr><td>Err</td><td>0.039(0.001)</td><td>0.039(0.001)</td><td>0.039(0.001)</td><td>0.039(0.001)</td><td>0.039(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>Value</td><td>1.776(0.003)</td><td>-0.102(0.005)</td><td>-0.768(0.005)</td><td>-0.075(0.004)</td><td>0.623 (0.004)</td></tr><tr><td>Err</td><td>0.092(0.002)</td><td>0.092(0.002)</td><td>0.092(0.002)</td><td>0.092(0.002)</td><td>0.092(0.002)</td></tr><tr><td rowspan="2">l2-CVaR (0.25)</td><td>Value</td><td>1.786(0.003)</td><td>-0.088(0.004)</td><td>-0.754(0.005)</td><td>-0.060(0.004)</td><td>0.636(0.003)</td></tr><tr><td>Err</td><td>0.078(0.002)</td><td>0.078(0.002)</td><td>0.078(0.002)</td><td>0.078(0.002)</td><td>0.078(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.792(0.003)</td><td>-0.078 (0.004)</td><td>-0.742(0.005)</td><td>-0.049(0.004)</td><td>0.645(0.003)</td></tr><tr><td>Err</td><td>0.069(0.002)</td><td>0.069(0.002)</td><td>0.069(0.002)</td><td>0.069(0.002)</td><td>0.069(0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.771 (0.003)</td><td>-0.221 (0.006)</td><td>-0.923 (0.007)</td><td>-0.144(0.006)</td><td>0.593 (0.004)</td></tr><tr><td>Err</td><td>0.096(0.002)</td><td>0.148 (0.002)</td><td>0.146(0.002)</td><td>0.125(0.002)</td><td>0.111 (0.002)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.788(0.002)</td><td>-0.096(0.004)</td><td>-0.766(0.005)</td><td>−0.062(0.003)</td><td>0.636(0.003)</td></tr><tr><td>Err</td><td>0.075(0.001)</td><td>0.084(0.001)</td><td>0.085(0.001)</td><td>0.080(0.001)</td><td>0.077(0.001)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.807 (0.002)</td><td>-0.050(0.004)</td><td>-0.709(0.004)</td><td>-0.023(0.003)</td><td>0.666(0.002)</td></tr><tr><td>Err</td><td>0.032(0.001)</td><td>0.033(0.001)</td><td>0.034(0.001)</td><td>0.033(0.001)</td><td>0.033(0.001)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.798(0.003)</td><td>-0.077(0.004)</td><td>-0.756(0.005)</td><td>-0.050(0.003)</td><td>0.651 (0.003)</td></tr><tr><td>Err</td><td>0.055(0.001)</td><td>0.067(0.002)</td><td>0.076(0.003)</td><td>0.066(0.002)</td><td>0.058(0.002)</td></tr></table>

## B.3 Discussion about GCDS

We further consider two additional scenarios that are more favorable to GCDS [21]. Since the goal here is internal method selection rather than external benchmarking, we mainly compare these models with DRF, which was the strongest distribution-first estimator in the earlier experiments. Setting 4 uses multiplicative non-Gaussian noise: $Y = \left( 1 + X _ { 1 } + X _ { 2 } + A ( X _ { 1 } - X _ { 2 } + X _ { 3 } ) \right) Z , Z = \exp ( 0 . 5 \bar { \varepsilon } )$ where ε follows a two-component Gaussian mixture. Setting 5 uses a mix Gaussian distribution: $Y \mid X , A \sim 0 . 5 N ( m ( X ) + ^ { - } A \delta ( X ) - 2 , 0 . 5 ^ { 2 } ) + 0 . 5 N ( m ( { \bar { X } } ) + A \delta ( X ) + 2 , 0 . 5 ^ { 2 } )$

Tables 15–18 show that the gains remain local. Under Setting 4, GCDS is only marginally better than DRF at $p = 1 0$ , and this advantage disappears already at $p = 2 0$ . Under Setting 5, GCDS has a clear advantage at $p = 1 0$ , but this advantage again disappears at $p = 2 0$ . Hence, even in scenarios designed to favor generative estimators, the advantage is limited and unstable. Combined with the earlier benchmark results, this leads us to a cautious view of generative estimators in the current tabular ITR problem: they are feasible within the distribution-first framework and may provide gains in selected scenarios, but their behavior is not stable enough to justify using them as default choices. Given their substantially higher training and tuning cost, we currently regard generative models as promising but still exploratory extensions.

Table 7: Setting 1 (linear Gaussian; $n = 1 0 0 0 , p = 2 0 ) \colon$ : full results for Mean, Quantil $\mathrm { . e _ { 0 . 1 } }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.811(0.003)</td><td>-0.040(0.003)</td><td>-0.699(0.004)</td><td>-0.013(0.003)</td><td>0.674(0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.805(0.003)</td><td>-0.052(0.004)</td><td>-0.716(0.004)</td><td>-0.025(0.003)</td><td>0.666(0.003)</td></tr><tr><td>Err</td><td>0.035(0.002)</td><td>0.035(0.002)</td><td>0.035(0.002)</td><td>0.035(0.002)</td><td>0.035(0.002)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.793(0.003)</td><td>−0.072(0.004)</td><td>-0.739(0.004)</td><td>-0.046(0.004)</td><td>0.648 (0.003)</td></tr><tr><td>Err</td><td>0.065(0.002)</td><td>0.065(0.002)</td><td>0.065(0.002)</td><td>0.065(0.002)</td><td>0.065 (0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.788(0.003)</td><td>-0.083 (0.004)</td><td>-0.751 (0.005)</td><td>-0.055(0.004)</td><td>0.641 (0.003)</td></tr><tr><td>Err</td><td>0.074(0.001)</td><td>0.074(0.001)</td><td>0.074(0.001)</td><td>0.074(0.001)</td><td>0.074(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.796(0.003)</td><td>-0.067(0.004)</td><td>-0.732(0.004)</td><td>-0.040(0.003)</td><td>0.653(0.003)</td></tr><tr><td>Err</td><td>0.058(0.001)</td><td>0.058(0.001)</td><td>0.058(0.001)</td><td>0.058(0.001)</td><td>0.058(0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.797(0.003)</td><td>-0.064(0.004)</td><td>-0.729(0.004)</td><td>-0.038(0.003)</td><td>0.655(0.003)</td></tr><tr><td>Err</td><td>0.053(0.001)</td><td>0.053(0.001)</td><td>0.053(0.001)</td><td>0.053(0.001)</td><td>0.053(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } .$  CVaR (0.1)</td><td>Value</td><td>1.750(0.004)</td><td>-0.148 (0.005)</td><td>-0.828(0.006)</td><td>-0.121 (0.005)</td><td>0.587(0.004)</td></tr><tr><td>Err</td><td>0.120(0.002)</td><td>0.120(0.002)</td><td>0.120(0.002)</td><td>0.120(0.002)</td><td>0.120(0.002)</td></tr><tr><td rowspan="2">l2-CVaR (0.25)</td><td>Value</td><td>1.771 (0.003)</td><td>-0.110(0.004)</td><td>-0.790(0.005)</td><td>-0.086(0.004)</td><td>0.617(0.003)</td></tr><tr><td>Err</td><td>0.096(0.002)</td><td>0.096(0.002)</td><td>0.096(0.002)</td><td>0.096(0.002)</td><td>0.096(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.781 (0.003)</td><td>−0.092(0.004)</td><td>-0.764(0.005)</td><td>-0.066(0.003)</td><td>0.632(0.003)</td></tr><tr><td>Err</td><td>0.083 (0.002)</td><td>0.083(0.002)</td><td>0.083(0.002)</td><td>0.083(0.002)</td><td>0.083 (0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.735(0.004)</td><td>-0.342(0.008)</td><td>-1.070(0.009)</td><td>-0.251 (0.008)</td><td>0.527(0.006)</td></tr><tr><td>Err</td><td>0.133(0.003)</td><td>0.200(0.003)</td><td>0.203(0.003)</td><td>0.173(0.003)</td><td>0.153(0.003)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.781 (0.003)</td><td>-0.103 (0.004)</td><td>-0.780(0.005)</td><td>-0.069(0.003)</td><td>0.630(0.003)</td></tr><tr><td>Err</td><td>0.081 (0.001)</td><td>0.090(0.002)</td><td>0.092(0.002)</td><td>0.086(0.001)</td><td>0.083(0.001)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.800(0.003)</td><td>-0.058(0.004)</td><td>-0.722(0.004)</td><td>-0.031 (0.003)</td><td>0.661 (0.003)</td></tr><tr><td>Err</td><td>0.048(0.001)</td><td>0.048(0.001)</td><td>0.049(0.001)</td><td>0.048(0.001)</td><td>0.048(0.001)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.779(0.003)</td><td>-0.099(0.004)</td><td>-0.782(0.006)</td><td>-0.075(0.004)</td><td>0.628(0.003)</td></tr><tr><td>Err</td><td>0.084(0.002)</td><td>0.087(0.002)</td><td>0.090(0.002)</td><td>0.087(0.002)</td><td>0.085(0.002)</td></tr></table>

Table 8: Setting 1 (linear Gaussian; $n = 1 0 0 0 , p = 5 0 ) \colon$ full results for Mean, $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.

<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.814(0.003)</td><td>-0.045(0.003)</td><td>-0.702(0.004)</td><td>-0.019(0.003)</td><td>0.671 (0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.806(0.003)</td><td>−0.062(0.004)</td><td>-0.727 (0.004)</td><td>-0.036(0.003)</td><td>0.658(0.003)</td></tr><tr><td>Err</td><td>0.044(0.002)</td><td>0.044(0.002)</td><td>0.044(0.002)</td><td>0.044(0.002)</td><td>0.044(0.002)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.777(0.003)</td><td>-0.112(0.005)</td><td>-0.787(0.005)</td><td>−0.088(0.004)</td><td>0.616(0.003)</td></tr><tr><td>Err</td><td>0.095(0.002)</td><td>0.095(0.002)</td><td>0.095(0.002)</td><td>0.095(0.002)</td><td>0.095(0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.758(0.003)</td><td>-0.142(0.004)</td><td>-0.824(0.005)</td><td>-0.118(0.004)</td><td>0.591 (0.003)</td></tr><tr><td>Err</td><td>0.115(0.001)</td><td>0.115(0.001)</td><td>0.115(0.001)</td><td>0.115(0.001)</td><td>0.115(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.779(0.003)</td><td>-0.109(0.004)</td><td>-0.780(0.004)</td><td>-0.083(0.003)</td><td>0.619(0.003)</td></tr><tr><td>Err</td><td>0.093(0.001)</td><td>0.093(0.001)</td><td>0.093(0.001)</td><td>0.093(0.001)</td><td>0.093 (0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.784(0.003)</td><td>−0.101 (0.004)</td><td>-0.773(0.004)</td><td>-0.075(0.003)</td><td>0.626(0.003)</td></tr><tr><td>Err</td><td>0.086(0.001)</td><td>0.086(0.001)</td><td>0.086(0.001)</td><td>0.086(0.001)</td><td>0.086(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>Value</td><td>1.695(0.004)</td><td>-0.254(0.006)</td><td>-0.950(0.007)</td><td>-0.226(0.006)</td><td>0.501 (0.005)</td></tr><tr><td>Err</td><td>0.172(0.002)</td><td>0.172(0.002)</td><td>0.172(0.002)</td><td>0.172(0.002)</td><td>0.172(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } \mathrm { - C V a R } \ ( 0 . 2 5 )$ </td><td>Value</td><td>1.732(0.004)</td><td>-0.183(0.006)</td><td>-0.877(0.006)</td><td>-0.161 (0.005)</td><td>0.557 (0.004)</td></tr><tr><td>Err</td><td>0.141 (0.002)</td><td>0.141 (0.002)</td><td>0.141(0.002)</td><td>0.141 (0.002)</td><td>0.141 (0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.754(0.003)</td><td>-0.144(0.005)</td><td>-0.832(0.006)</td><td>-0.123(0.004)</td><td>0.588(0.004)</td></tr><tr><td>Err</td><td>0.119(0.002)</td><td>0.119(0.002)</td><td>0.119(0.002)</td><td>0.119(0.002)</td><td>0.119(0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.664(0.004)</td><td>-0.522(0.007)</td><td>-1.284(0.009)</td><td>-0.411 (0.007)</td><td>0.411 (0.005)</td></tr><tr><td>Err</td><td>0.191 (0.002)</td><td>0.269(0.002)</td><td>0.281 (0.003)</td><td>0.240(0.002)</td><td>0.213(0.002)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.777(0.003)</td><td>-0.123(0.005)</td><td>-0.811(0.007)</td><td>-0.089(0.004)</td><td>0.615(0.003)</td></tr><tr><td>Err</td><td>0.094(0.001)</td><td>0.103(0.002)</td><td>0.105(0.002)</td><td>0.098(0.001)</td><td>0.095(0.001)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.726(0.003)</td><td>-0.197(0.005)</td><td>-0.892(0.006)</td><td>-0.174(0.005)</td><td>0.546(0.004)</td></tr><tr><td>Err</td><td>0.147(0.002)</td><td>0.148(0.002)</td><td>0.149(0.002)</td><td>0.148(0.002)</td><td>0.147(0.002)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.626(0.006)</td><td>−0.388(0.012)</td><td>-1.121 (0.014)</td><td>-0.358(0.011)</td><td>0.403 (0.009)</td></tr><tr><td>Err</td><td>0.213(0.004)</td><td>0.223 (0.004)</td><td>0.230(0.005)</td><td>0.222(0.004)</td><td>0.217(0.004)</td></tr></table>

Table 9: Setting 1 (linear Gaussian; $n = 1 0 0 0 , p = 1 0 0 )$ : full results for Mean, Quantil $\mathrm { e _ { 0 . } } .$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.817(0.003)</td><td>-0.046(0.003)</td><td>-0.700(0.003)</td><td>-0.016(0.003)</td><td>0.673 (0.003)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.807(0.003)</td><td>-0.063(0.004)</td><td>-0.723(0.005)</td><td>-0.033(0.003)</td><td>0.660(0.003)</td></tr><tr><td>Err</td><td>0.046(0.002)</td><td>0.046(0.002)</td><td>0.046(0.002)</td><td>0.046(0.002)</td><td>0.046(0.002)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.744(0.004)</td><td>-0.170(0.005)</td><td>-0.852(0.006)</td><td>-0.141 (0.005)</td><td>0.573 (0.004)</td></tr><tr><td>Err</td><td>0.130(0.002)</td><td>0.130(0.002)</td><td>0.130(0.002)</td><td>0.130(0.002)</td><td>0.130(0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.703 (0.003)</td><td>-0.240(0.005)</td><td>-0.936(0.006)</td><td>-0.212(0.005)</td><td>0.514(0.004)</td></tr><tr><td>Err</td><td>0.166(0.001)</td><td>0.166(0.001)</td><td>0.166(0.001)</td><td>0.166(0.001)</td><td>0.166(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.737(0.003)</td><td>-0.180(0.005)</td><td>-0.863 (0.005)</td><td>-0.151 (0.004)</td><td>0.564(0.003)</td></tr><tr><td>Err</td><td>0.136(0.001)</td><td>0.136(0.001)</td><td>0.136(0.001)</td><td>0.136(0.001)</td><td>0.136(0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.750(0.003)</td><td>-0.162(0.004)</td><td>-0.840(0.004)</td><td>-0.132(0.003)</td><td>0.580(0.003)</td></tr><tr><td>Err</td><td>0.125(0.001)</td><td>0.125(0.001)</td><td>0.125(0.001)</td><td>0.125(0.001)</td><td>0.125(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } .$  -CVaR (0.1)</td><td>Value</td><td>1.632(0.004)</td><td>-0.357(0.006)</td><td>-1.062(0.007)</td><td>-0.325(0.006)</td><td>0.416(0.005)</td></tr><tr><td>Err</td><td>0.215(0.002)</td><td>0.215(0.002)</td><td>0.215(0.002)</td><td>0.215(0.002)</td><td>0.215(0.002)</td></tr><tr><td rowspan="2">l2-CVaR (0.25)</td><td>Value</td><td>1.679(0.004)</td><td>–0.281 (0.006)</td><td>–0.978 (0.006)</td><td>-0.250(0.005)</td><td>0.482(0.005)</td></tr><tr><td>Err</td><td>0.183 (0.002)</td><td>0.183(0.002)</td><td>0.183(0.002)</td><td>0.183(0.002)</td><td>0.183 (0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.707(0.004)</td><td>-0.232(0.006)</td><td>−0.927(0.006)</td><td>-0.205 (0.005)</td><td>0.520(0.004)</td></tr><tr><td>Err</td><td>0.162(0.002)</td><td>0.162(0.002)</td><td>0.162(0.002)</td><td>0.162(0.002)</td><td>0.162(0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.578(0.005)</td><td>-0.676(0.008)</td><td>-1.424(0.009)</td><td>-0.556(0.007)</td><td>0.291 (0.006)</td></tr><tr><td>Err</td><td>0.243 (0.002)</td><td>0.323 (0.002)</td><td>0.334(0.003)</td><td>0.293(0.002)</td><td>0.265 (0.002)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.770(0.003)</td><td>-0.152(0.007)</td><td>-0.850(0.012)</td><td>-0.110(0.006)</td><td>0.604(0.003)</td></tr><tr><td>Err</td><td>0.104(0.002)</td><td>0.117(0.003)</td><td>0.119(0.003)</td><td>0.110(0.002)</td><td>0.108(0.002)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.342(0.014)</td><td>-0.756(0.016)</td><td>-1.471(0.016)</td><td>-0.720(0.016)</td><td>0.052(0.015)</td></tr><tr><td>Err</td><td>0.359(0.006)</td><td>0.360(0.006)</td><td>0.361 (0.006)</td><td>0.360(0.006)</td><td>0.360(0.006)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.689(0.004)</td><td>-0.264(0.005)</td><td>-0.957(0.006)</td><td>-0.233(0.005)</td><td>0.496(0.004)</td></tr><tr><td>Err</td><td>0.176(0.001)</td><td>0.176(0.001)</td><td>0.176(0.001)</td><td>0.176(0.001)</td><td>0.176(0.001)</td></tr></table>

Table 10: Setting 1 (linear Gaussian; $n = 1 0 0 0 , p = 3 0 0 ) \colon$ full results for Mean, $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.

<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.816(0.002)</td><td>-0.044(0.003)</td><td>-0.703(0.004)</td><td>-0.016(0.003)</td><td>0.674(0.003)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.799(0.003)</td><td>-0.075(0.004)</td><td>-0.750(0.005)</td><td>-0.049(0.004)</td><td>0.648(0.003)</td></tr><tr><td>Err</td><td>0.058(0.002)</td><td>0.058(0.002)</td><td>0.058(0.002)</td><td>0.058(0.002)</td><td>0.058(0.002)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.600(0.004)</td><td>-0.404(0.006)</td><td>-1.115(0.006)</td><td>−0.372(0.006)</td><td>0.375(0.005)</td></tr><tr><td>Err</td><td>0.233 (0.002)</td><td>0.233(0.002)</td><td>0.233(0.002)</td><td>0.233 (0.002)</td><td>0.233(0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.518(0.004)</td><td>-0.524(0.006)</td><td>-1.246(0.006)</td><td>-0.494(0.005)</td><td>0.268(0.005)</td></tr><tr><td>Err</td><td>0.276(0.001)</td><td>0.276(0.001)</td><td>0.276(0.001)</td><td>0.276(0.001)</td><td>0.276(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.533 (0.004)</td><td>-0.501 (0.005)</td><td>-1.223(0.005)</td><td>-0.471 (0.005)</td><td>0.288(0.004)</td></tr><tr><td>Err</td><td>0.268(0.002)</td><td>0.268(0.002)</td><td>0.268(0.002)</td><td>0.268(0.002)</td><td>0.268(0.002)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.560(0.004)</td><td>-0.460(0.005)</td><td>-1.177(0.006)</td><td>-0.430(0.005)</td><td>0.323(0.004)</td></tr><tr><td>Err</td><td>0.254(0.001)</td><td>0.254(0.001)</td><td>0.254(0.001)</td><td>0.254(0.001)</td><td>0.254(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>Value</td><td>1.490(0.004)</td><td>-0.563 (0.006)</td><td>-1.284(0.007)</td><td>–0.531 (0.006)</td><td>0.233(0.005)</td></tr><tr><td>Err</td><td>0.290(0.002)</td><td>0.290(0.002)</td><td>0.290(0.002)</td><td>0.290(0.002)</td><td>0.290(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } \mathrm { - C V a R } \ ( 0 . 2 5 )$ </td><td>Value Err</td><td>1.542(0.004)</td><td>-0.490(0.006)</td><td>-1.208(0.006)</td><td>-0.459(0.006)</td><td>0.299(0.005)</td></tr><tr><td></td><td>0.264(0.002)</td><td>0.264(0.002)</td><td>0.264(0.002)</td><td>0.264(0.002)</td><td>0.264(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value Err</td><td>1.592(0.004)</td><td>-0.417(0.006)</td><td>-1.129(0.006)</td><td>-0.387(0.005)</td><td>0.363(0.005)</td></tr><tr><td></td><td>0.238(0.002)</td><td>0.238(0.002)</td><td>0.238 (0.002)</td><td>0.238(0.002)</td><td>0.238(0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.408 (0.005)</td><td>-0.865(0.007)</td><td>-1.595(0.007)</td><td>-0.773 (0.007)</td><td>0.063(0.007)</td></tr><tr><td>Err</td><td>0.330(0.002)</td><td>0.396(0.002)</td><td>0.402(0.002)</td><td>0.375(0.002)</td><td>0.352(0.002)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.751 (0.004)</td><td>–0.240(0.012)</td><td>-1.004(0.019)</td><td>-0.193(0.010)</td><td>0.557(0.005)</td></tr><tr><td>Err</td><td>0.128(0.002)</td><td>0.161 (0.005)</td><td>0.176(0.006)</td><td>0.149(0.004)</td><td>0.138(0.003)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.055(0.006)</td><td>-1.079(0.007)</td><td>-1.774(0.006)</td><td>-1.039(0.006)</td><td>–0.271 (0.007)</td></tr><tr><td>Err</td><td>0.479(0.002)</td><td>0.478(0.002)</td><td>0.478(0.002)</td><td>0.478(0.002)</td><td>0.478(0.002)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.229(0.006)</td><td>-0.900(0.007)</td><td>-1.607(0.007)</td><td>-0.859(0.007)</td><td>-0.083 (0.007)</td></tr><tr><td>Err</td><td>0.407(0.002)</td><td>0.407(0.002)</td><td>0.407 (0.002)</td><td>0.407(0.002)</td><td>0.407(0.002)</td></tr></table>

Table 11: Setting 2 (nonlinear Gaussian; $n = 1 0 0 0 , p = 1 0 )$ : full results for Mean, Quantile , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.654(0.003)</td><td>-0.044(0.003)</td><td>-0.587(0.003)</td><td>−0.029(0.002)</td><td>0.533(0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2">l1-PLS</td><td>Value</td><td>1.566(0.003)</td><td>-0.156(0.003)</td><td>-0.697(0.003)</td><td>-0.139(0.003)</td><td>0.425 (0.002)</td></tr><tr><td>Err</td><td>0.173(0.001)</td><td>0.173(0.001)</td><td>0.173(0.001)</td><td>0.173(0.001)</td><td>0.173(0.001)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.592(0.003)</td><td>-0.118(0.003)</td><td>-0.658(0.003)</td><td>-0.102(0.003)</td><td>0.460(0.002)</td></tr><tr><td>Err</td><td>0.214(0.002)</td><td>0.214(0.002)</td><td>0.214(0.002)</td><td>0.214(0.002)</td><td>0.214(0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.561 (0.003)</td><td>-0.166(0.004)</td><td>-0.709(0.004)</td><td>-0.148 (0.003)</td><td>0.418(0.003)</td></tr><tr><td>Err</td><td>0.182(0.001)</td><td>0.182(0.001)</td><td>0.182(0.001)</td><td>0.182(0.001)</td><td>0.182(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.568(0.003)</td><td>-0.155(0.003)</td><td>-0.696(0.004)</td><td>-0.137(0.003)</td><td>0.428(0.002)</td></tr><tr><td>Err</td><td>0.180(0.001)</td><td>0.180(0.001)</td><td>0.180(0.001)</td><td>0.180(0.001)</td><td>0.180(0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.568(0.003)</td><td>-0.155(0.003)</td><td>-0.696(0.003)</td><td>-0.137(0.003)</td><td>0.427 (0.002)</td></tr><tr><td>Err</td><td>0.178(0.001)</td><td>0.178(0.001)</td><td>0.178(0.001)</td><td>0.178(0.001)</td><td>0.178(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } .$  -CVaR (0.1)</td><td>Value</td><td>1.559(0.004)</td><td>-0.140(0.004)</td><td>-0.679(0.004)</td><td>-0.124(0.003)</td><td>0.436(0.003)</td></tr><tr><td>Err</td><td>0.253(0.003)</td><td>0.253(0.003)</td><td>0.253(0.003)</td><td>0.253(0.003)</td><td>0.253(0.003)</td></tr><tr><td rowspan="2">l2-CVaR (0.25)</td><td>Value</td><td>1.578(0.003)</td><td>-0.125(0.004)</td><td>-0.664(0.004)</td><td>-0.109(0.003)</td><td>0.450(0.003)</td></tr><tr><td>Err</td><td>0.242(0.003)</td><td>0.242(0.003)</td><td>0.242(0.003)</td><td>0.242(0.003)</td><td>0.242(0.003)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.584(0.003)</td><td>-0.120(0.003)</td><td>-0.659(0.003)</td><td>-0.105(0.003)</td><td>0.455(0.002)</td></tr><tr><td>Err</td><td>0.237(0.002)</td><td>0.237(0.002)</td><td>0.237(0.002)</td><td>0.237(0.002)</td><td>0.237(0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.582(0.003)</td><td>-0.237(0.004)</td><td>-0.807(0.004)</td><td>-0.219(0.004)</td><td>0.381 (0.004)</td></tr><tr><td>Err</td><td>0.231 (0.002)</td><td>0.412(0.003)</td><td>0.448(0.003)</td><td>0.416(0.003)</td><td>0.366(0.003)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.638(0.003)</td><td>-0.076(0.003)</td><td>-0.626(0.004)</td><td>-0.060(0.003)</td><td>0.508(0.003)</td></tr><tr><td>Err</td><td>0.066(0.004)</td><td>0.129(0.007)</td><td>0.158(0.008)</td><td>0.124(0.007)</td><td>0.098(0.006)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.596(0.003)</td><td>-0.121 (0.003)</td><td>-0.663 (0.003)</td><td>-0.105(0.002)</td><td>0.460(0.002)</td></tr><tr><td>Err</td><td>0.200(0.001)</td><td>0.195(0.001)</td><td>0.193(0.001)</td><td>0.195(0.001)</td><td>0.197(0.001)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.568 (0.003)</td><td>-0.192(0.007)</td><td>-0.769(0.011)</td><td>-0.173(0.006)</td><td>0.413 (0.004)</td></tr><tr><td>Err</td><td>0.184(0.001)</td><td>0.187(0.001)</td><td>0.195(0.002)</td><td>0.187(0.001)</td><td>0.183(0.001)</td></tr></table>

Table 12: Setting 2 (nonlinear Gaussian; $n = 1 0 0 0$ $p = 2 0 ) $ : full results for Mean, $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.

<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.650(0.003)</td><td>-0.045(0.003)</td><td>-0.590(0.003)</td><td>-0.029(0.002)</td><td>0.531 (0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.558(0.003)</td><td>-0.163(0.003)</td><td>-0.708 (0.003)</td><td>-0.145(0.003)</td><td>0.418(0.002)</td></tr><tr><td>Err</td><td>0.174(0.001)</td><td>0.174(0.001)</td><td>0.174(0.001)</td><td>0.174(0.001)</td><td>0.174(0.001)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.580(0.003)</td><td>-0.134(0.004)</td><td>-0.674(0.004)</td><td>-0.115(0.003)</td><td>0.446(0.003)</td></tr><tr><td>Err</td><td>0.221 (0.002)</td><td>0.221 (0.002)</td><td>0.221(0.002)</td><td>0.221 (0.002)</td><td>0.221 (0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.545(0.003)</td><td>-0.188(0.004)</td><td>-0.737(0.004)</td><td>-0.169(0.003)</td><td>0.399(0.003)</td></tr><tr><td>Err</td><td>0.193 (0.001)</td><td>0.193(0.001)</td><td>0.193(0.001)</td><td>0.193(0.001)</td><td>0.193(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.555(0.003)</td><td>-0.171 (0.003)</td><td>-0.718(0.003)</td><td>-0.153(0.003)</td><td>0.412(0.002)</td></tr><tr><td>Err</td><td>0.185(0.001)</td><td>0.185(0.001)</td><td>0.185(0.001)</td><td>0.185(0.001)</td><td>0.185(0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.557(0.003)</td><td>-0.168(0.003)</td><td>-0.713(0.004)</td><td>-0.150(0.003)</td><td>0.415(0.002)</td></tr><tr><td>Err</td><td>0.184(0.001)</td><td>0.184(0.001)</td><td>0.184(0.001)</td><td>0.184(0.001)</td><td>0.184(0.001)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>Value</td><td>1.515(0.005)</td><td>-0.185(0.005)</td><td>-0.729(0.006)</td><td>-0.167(0.005)</td><td>0.394(0.004)</td></tr><tr><td>Err</td><td>0.269(0.003)</td><td>0.269(0.003)</td><td>0.269(0.003)</td><td>0.269(0.003)</td><td>0.269(0.003)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } \mathrm { - C V a R } \ ( 0 . 2 5 )$ </td><td>Value</td><td>1.550(0.003)</td><td>-0.154(0.004)</td><td>-0.695(0.004)</td><td>-0.136(0.003)</td><td>0.423 (0.003)</td></tr><tr><td>Err</td><td>0.255(0.002)</td><td>0.255(0.002)</td><td>0.255(0.002)</td><td>0.255(0.002)</td><td>0.255(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.564(0.003)</td><td>-0.143(0.003)</td><td>-0.684(0.004)</td><td>-0.126(0.003)</td><td>0.434(0.003)</td></tr><tr><td>Err</td><td>0.247(0.002)</td><td>0.247(0.002)</td><td>0.247(0.002)</td><td>0.247(0.002)</td><td>0.247 (0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.542(0.003)</td><td>-0.305(0.003)</td><td>−0.848(0.003)</td><td>-0.294(0.003)</td><td>0.287(0.003)</td></tr><tr><td>Err</td><td>0.262(0.003)</td><td>0.469(0.002)</td><td>0.487(0.001)</td><td>0.474(0.001)</td><td>0.436(0.002)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.633(0.003)</td><td>−0.083(0.004)</td><td>-0.632(0.004)</td><td>-0.064(0.003)</td><td>0.503(0.003)</td></tr><tr><td>Err</td><td>0.069(0.005)</td><td>0.130(0.008)</td><td>0.165(0.008)</td><td>0.125(0.008)</td><td>0.096(0.006)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.575(0.003)</td><td>-0.148(0.003)</td><td>-0.697(0.004)</td><td>-0.130(0.003)</td><td>0.436(0.002)</td></tr><tr><td>Err</td><td>0.191 (0.001)</td><td>0.187(0.001)</td><td>0.185(0.001)</td><td>0.187(0.001)</td><td></td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.542(0.003)</td><td>-0.162(0.004)</td><td>-0.712(0.006)</td><td>-0.143(0.004)</td><td>0.188(0.001)</td></tr><tr><td>Err</td><td>0.197(0.001)</td><td>0.251 (0.005)</td><td>0.276(0.007)</td><td>0.249(0.005)</td><td>0.420(0.003) 0.223 (0.003)</td></tr></table>

Table 13: Setting 2 (nonlinear Gaussian; $n = 1 0 0 0 , p = 3 0 0 )$ : full results for Mean, $\mathrm { Q u a n t i l e } _ { 0 . 1 } .$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions, except GCDS, which has 99 evaluable repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.650(0.003)</td><td>-0.049(0.003)</td><td>-0.590(0.003)</td><td>-0.031 (0.002)</td><td>0.530(0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>1.551 (0.003)</td><td>-0.175(0.003)</td><td>-0.717(0.003)</td><td>-0.157 (0.003)</td><td>0.408(0.002)</td></tr><tr><td>Err</td><td>0.174(0.001)</td><td>0.174(0.001)</td><td>0.174(0.001)</td><td>0.174(0.001)</td><td>0.174(0.001)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>1.385(0.005)</td><td>-0.358(0.007)</td><td>-0.963(0.009)</td><td>-0.350(0.007)</td><td>0.245 (0.006)</td></tr><tr><td>Err</td><td>0.318(0.002)</td><td>0.318(0.002)</td><td>0.318(0.002)</td><td>0.318(0.002)</td><td>0.318(0.002)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>1.281 (0.003)</td><td>-0.564(0.004)</td><td>-1.235(0.005)</td><td>-0.558(0.004)</td><td>0.088(0.003)</td></tr><tr><td>Err</td><td>0.355(0.001)</td><td>0.355(0.001)</td><td>0.355(0.001)</td><td>0.355(0.001)</td><td>0.355(0.001)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>1.290(0.003)</td><td>-0.552(0.004)</td><td>-1.222(0.005)</td><td>-0.547(0.004)</td><td>0.096(0.003)</td></tr><tr><td>Err</td><td>0.351 (0.001)</td><td>0.351 (0.001)</td><td>0.351(0.001)</td><td>0.351(0.001)</td><td>0.351 (0.001)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>1.313(0.004)</td><td>-0.520(0.005)</td><td>-1.182(0.006)</td><td>-0.515(0.004)</td><td>0.123(0.003)</td></tr><tr><td>Err</td><td>0.340(0.002)</td><td>0.340(0.002)</td><td>0.340(0.002)</td><td>0.340(0.002)</td><td>0.340(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>Value</td><td>1.245 (0.005)</td><td>−0.529(0.007)</td><td>-1.185(0.010)</td><td>-0.525(0.007)</td><td>0.103 (0.005)</td></tr><tr><td>Err</td><td>0.388(0.002)</td><td>0.388(0.002)</td><td>0.388(0.002)</td><td>0.388(0.002)</td><td>0.388 (0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } \mathrm { - C V a R } \ ( 0 . 2 5 )$ </td><td>Value</td><td>1.277(0.004)</td><td>-0.482(0.006)</td><td>-1.128(0.007)</td><td>-0.480(0.005)</td><td>0.139(0.004)</td></tr><tr><td>Err</td><td>0.372(0.002)</td><td>0.372(0.002)</td><td>0.372(0.002)</td><td>0.372(0.002)</td><td>0.372(0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>1.305(0.004)</td><td>−0.421 (0.008)</td><td>-1.040(0.012)</td><td>-0.415(0.008)</td><td>0.187(0.006)</td></tr><tr><td>Err</td><td>0.364(0.002)</td><td>0.364(0.002)</td><td>0.364(0.002)</td><td>0.364(0.002)</td><td>0.364(0.002)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>1.225 (0.004)</td><td>-0.336(0.003)</td><td>-0.860(0.003)</td><td>-0.322(0.002)</td><td>0.207 (0.002)</td></tr><tr><td>Err</td><td>0.416(0.003)</td><td>0.497(0.001)</td><td>0.498(0.001)</td><td>0.498(0.001)</td><td>0.496(0.001)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.604(0.003)</td><td>-0.331 (0.003)</td><td>-0.858(0.003)</td><td>-0.320(0.002)</td><td>0.267 (0.007)</td></tr><tr><td>Err</td><td>0.186(0.009)</td><td>0.498(0.001)</td><td>0.499(0.001)</td><td>0.498(0.001)</td><td>0.465(0.003)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>1.095(0.004)</td><td>-0.411 (0.022)</td><td>–0.965(0.026)</td><td>-0.399(0.021)</td><td>0.154(0.016)</td></tr><tr><td>Err</td><td>0.491 (0.002)</td><td>0.490(0.002)</td><td>0.490(0.002)</td><td>0.490(0.002)</td><td>0.490(0.002)</td></tr><tr><td rowspan="2">GCDS†</td><td>Value</td><td>1.331 (0.005)</td><td>-0.337(0.008)</td><td>-0.920(0.012)</td><td>-0.330(0.008)</td><td>0.246(0.005)</td></tr><tr><td>Err</td><td>0.361 (0.004)</td><td>0.362(0.004)</td><td>0.362(0.004)</td><td>0.362(0.004)</td><td>0.362(0.004)</td></tr></table>

Table 14: Setting 3 (heavy-tailed lognormal; $n = 1 0 0 0 , p = 1 0 ) \colon$ : full results for Mean, Quantile , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>9.244(0.075)</td><td>1.167(0.003)</td><td>0.615(0.003)</td><td>1.243 (0.002)</td><td>1.952(0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2"> $\ell _ { 1 } { \mathrm { - P L S } }$ </td><td>Value</td><td>8.670(0.081)</td><td>0.267(0.028)</td><td>-0.339(0.025)</td><td>0.357(0.027)</td><td>1.193(0.027)</td></tr><tr><td>Err</td><td>0.432(0.011)</td><td>0.432(0.011)</td><td>0.432(0.011)</td><td>0.432(0.011)</td><td>0.432(0.011)</td></tr><tr><td rowspan="2">RWL</td><td>Value</td><td>8.759(0.089)</td><td>0.348(0.024)</td><td>-0.263 (0.023)</td><td>0.444(0.024)</td><td>1.288(0.023)</td></tr><tr><td>Err</td><td>0.396(0.009)</td><td>0.396(0.009)</td><td>0.396(0.009)</td><td>0.396(0.009)</td><td>0.396(0.009)</td></tr><tr><td rowspan="2">CQR (0.1)</td><td>Value</td><td>9.243(0.076)</td><td>1.167 (0.003)</td><td>0.614(0.003)</td><td>1.242(0.002)</td><td>1.951 (0.002)</td></tr><tr><td>Err</td><td>0.009 (0.000)</td><td>0.009 (0.000)</td><td>0.009 (0.000)</td><td>0.009 (0.000)</td><td>0.009 (0.000)</td></tr><tr><td rowspan="2">CQR (0.25)</td><td>Value</td><td>9.252(0.074)</td><td>1.164(0.003)</td><td>0.610(0.003)</td><td>1.240(0.002)</td><td>1.949(0.002)</td></tr><tr><td>Err</td><td>0.021 (0.000)</td><td>0.021 (0.000)</td><td>0.021 (0.000)</td><td>0.021 (0.000)</td><td>0.021 (0.000)</td></tr><tr><td rowspan="2">CQR (0.5)</td><td>Value</td><td>9.200(0.075)</td><td>1.131 (0.004)</td><td>0.559(0.005)</td><td>1.203 (0.003)</td><td>1.922(0.003)</td></tr><tr><td>Err</td><td>0.073 (0.002)</td><td>0.073 (0.002)</td><td>0.073 (0.002)</td><td>0.073 (0.002)</td><td>0.073 (0.002)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 1 \right)$ </td><td>Value</td><td>9.074(0.084)</td><td>0.867(0.040)</td><td>0.279(0.041)</td><td>0.943 (0.039)</td><td>1.701 (0.034)</td></tr><tr><td>Err</td><td>0.190(0.016)</td><td>0.190(0.016)</td><td>0.190(0.016)</td><td>0.190(0.016)</td><td>0.190(0.016)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 2 5 \right)$ </td><td>Value</td><td>9.080(0.086)</td><td>0.884(0.036)</td><td>0.289(0.037)</td><td>0.959(0.035)</td><td>1.721 (0.030)</td></tr><tr><td>Err</td><td>0.186(0.014)</td><td>0.186(0.014)</td><td>0.186(0.014)</td><td>0.186(0.014)</td><td>0.186(0.014)</td></tr><tr><td rowspan="2"> $\ell _ { 2 } { \mathrm { - C V a R } } \left( 0 . 5 \right)$ </td><td>Value</td><td>9.098 (0.090)</td><td>0.882(0.038)</td><td>0.287(0.039)</td><td>0.959(0.037)</td><td>1.719(0.033)</td></tr><tr><td>Err</td><td>0.187(0.015)</td><td>0.187(0.015)</td><td>0.187(0.015)</td><td>0.187(0.015)</td><td>0.187(0.015)</td></tr><tr><td rowspan="2">Kernel</td><td>Value</td><td>8.681 (0.082)</td><td>1.069(0.004)</td><td>0.468(0.004)</td><td>1.168(0.003)</td><td>1.896(0.003)</td></tr><tr><td>Err</td><td>0.416(0.007)</td><td>0.116(0.001)</td><td>0.109(0.001)</td><td>0.096(0.001)</td><td>0.098(0.001)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>8.861 (0.081)</td><td>1.088(0.004)</td><td>0.516(0.005)</td><td>1.166(0.003)</td><td>1.886(0.003)</td></tr><tr><td>Err</td><td>0.372(0.010)</td><td>0.118(0.002)</td><td>0.118(0.002)</td><td>0.114(0.002)</td><td>0.114(0.002)</td></tr><tr><td rowspan="2">Conditional DDPM</td><td>Value</td><td>8.994(0.081)</td><td>0.897(0.018)</td><td>0.267(0.021)</td><td>0.976(0.017)</td><td>1.752(0.013)</td></tr><tr><td>Err</td><td>0.266(0.009)</td><td>0.194(0.007)</td><td>0.201 (0.008)</td><td>0.193 (0.007)</td><td>0.192(0.006)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>9.089(0.074)</td><td>1.024(0.014)</td><td>0.411 (0.016)</td><td>1.097(0.013)</td><td>1.847(0.010)</td></tr><tr><td>Err</td><td>0.163(0.008)</td><td>0.138(0.006)</td><td>0.142(0.006)</td><td>0.136(0.006)</td><td>0.133(0.006)</td></tr></table>

Table 15: Setting 4 (multiplicative non-Gaussian noise; $n = 1 0 0 0 , p = 1 0 )$ : full results for Mean, $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>3.167(0.006)</td><td>0.200(0.001)</td><td>-0.115(0.003)</td><td>0.179(0.001)</td><td>0.530(0.001)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>3.099(0.007)</td><td>0.183(0.001)</td><td>-0.171(0.005)</td><td>0.152(0.002)</td><td>0.505(0.001)</td></tr><tr><td>Err</td><td>0.091 (0.002)</td><td>0.083(0.002)</td><td>0.080(0.001)</td><td>0.080(0.001)</td><td>0.087 (0.002)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>3.130(0.007)</td><td>0.168(0.002)</td><td>-0.257(0.013)</td><td>0.116(0.004)</td><td>0.493 (0.002)</td></tr><tr><td>Err</td><td>0.072(0.002)</td><td>0.129(0.003)</td><td>0.145(0.004)</td><td>0.122(0.003)</td><td>0.097(0.002)</td></tr></table>

Table 16: Setting 4 (multiplicative non-Gaussian noise; $n = 1 0 0 0 , p = 2 0 ) \colon$ full results for Mean, $\mathrm { Q u a n t i l e } _ { 0 . 1 } ,$ and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>3.166(0.007)</td><td>0.199(0.001)</td><td>-0.117(0.003)</td><td>0.178(0.001)</td><td>0.530(0.002)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>3.074(0.008)</td><td>0.176(0.001)</td><td>-0.183(0.004)</td><td>0.145(0.002)</td><td>0.497 (0.002)</td></tr><tr><td>Err</td><td>0.106(0.002)</td><td>0.097(0.002)</td><td>0.097 (0.002)</td><td>0.094(0.002)</td><td>0.100 (0.002)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>3.059(0.008)</td><td>0.152(0.003)</td><td>-0.314(0.010)</td><td>0.079(0.005)</td><td>0.461 (0.004)</td></tr><tr><td>Err</td><td>0.116(0.003)</td><td>0.113(0.003)</td><td>0.114(0.003)</td><td>0.112(0.003)</td><td>0.111 (0.003)</td></tr></table>

Table 17: Setting 5 (additive Gaussian-mixture noise; $n = 1 0 0 0 , p = 1 0 )$ : full results for Mean, $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e } _ { 0 . 1 }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.808 (0.003)</td><td>-1.193(0.004)</td><td>-1.817(0.003)</td><td>-1.106(0.003)</td><td>-0.213(0.003)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.750(0.004)</td><td>-1.267(0.004)</td><td>-1.927(0.007)</td><td>-1.179(0.004)</td><td>-0.284(0.004)</td></tr><tr><td>Err</td><td>0.120(0.003)</td><td>0.113(0.002)</td><td>0.112(0.002)</td><td>0.110(0.002)</td><td>0.122(0.003)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.799(0.004)</td><td>-1.203(0.004)</td><td>-1.832(0.004)</td><td>-1.115(0.003)</td><td>-0.226(0.004)</td></tr><tr><td>Err</td><td>0.049(0.002)</td><td>0.040(0.001)</td><td>0.042(0.002)</td><td>0.040(0.001)</td><td>0.054(0.002)</td></tr></table>

Table 18: Setting 5 (additive Gaussian-mixture noise; $n = 1 0 0 0 , p = 2 0 )$ : full results for Mean, $\mathrm { Q u a n t i l e } _ { 0 . 1 } .$ , and CVaR. Entries are means (Monte Carlo standard errors) over 100 paired repetitions.
<table><tr><td>Method</td><td>Metric</td><td>Mean</td><td> $\mathrm { Q u a n t i l e _ { 0 . 1 } }$ </td><td> $\mathrm { { C V a R } _ { 0 . 1 } }$ </td><td> $\mathrm { C V a R } _ { 0 . 2 5 }$ </td><td> $\mathrm { C V a R } _ { 0 . 5 }$ </td></tr><tr><td rowspan="2">Oracle</td><td>Value</td><td>1.813(0.004)</td><td>-1.195(0.003)</td><td>-1.816(0.003)</td><td>-1.105(0.003)</td><td>-0.210(0.003)</td></tr><tr><td>Err</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td><td>0.000(0.000)</td></tr><tr><td rowspan="2">DRF</td><td>Value</td><td>1.731 (0.005)</td><td>-1.293(0.004)</td><td>-1.964(0.008)</td><td>-1.210(0.004)</td><td>-0.306(0.005)</td></tr><tr><td>Err</td><td>0.144(0.003)</td><td>0.138(0.002)</td><td>0.138(0.003)</td><td>0.136(0.003)</td><td>0.147 (0.003)</td></tr><tr><td rowspan="2">GCDS</td><td>Value</td><td>1.635(0.010)</td><td>-1.513(0.018)</td><td>-2.247 (0.020)</td><td>-1.412(0.016)</td><td>-0.417(0.011)</td></tr><tr><td>Err</td><td>0.207(0.005)</td><td>0.226(0.006)</td><td>0.237(0.007)</td><td>0.224(0.006)</td><td>0.214(0.006)</td></tr></table>

## B.4 Treatment-Interactive Outcome Distributions

To examine criterion-dependent treatment decisions, we adopt the treatment-interactive error design from Simulation Study II of Xiao, Zhang, and Lu (2019). Let ${ \bar { X } } _ { 1 } , X _ { 2 }$ be independent Uniform $[ - 1 , \bar { 1 } ]$ variables, let $B \sim$ Bernoulli(0.5) independently of $X$ , and generate

$$
\begin{array} { r l } & { Y = 1 + 0 . 5 \sin \{ \pi ( X _ { 1 } - X _ { 2 } ) \} + 0 . 2 5 ( 1 + X _ { 1 } + 2 X _ { 2 } ) ^ { 2 } } \\ & { \qquad + ( B - 0 . 5 ) \theta ^ { \top } \widetilde { X } + ( 1 + B d _ { 0 } X _ { 1 } ^ { 2 } ) \epsilon , } \end{array}
$$

where $\begin{array} { r } { \widetilde { X } = ( 1 , X _ { 1 } , X _ { 2 } ) ^ { \top } , \theta = ( 0 . 5 , 2 , - 1 ) ^ { \top } , d _ { 0 } = 5 . } \end{array}$ , and ϵ ∼ Gamma(shape = 1, rate $= 1 ) - 1$ independently of $( X , { \dot { B } } )$ . The implementation uses $A \in \{ - 1 , + 1 \}$ with $B \bar { = } ( A + 1 ) / 2$ . We use $p = 2 , n = 8 0 0$ , 100 Monte Carlo replications, and an independent test sample of size 10,000 per replication. Treatment changes both the conditional location and scale.

Table 19 reports the criterion-matched comparisons for Mean, $Q _ { 0 . 1 } ,$ , and $\mathrm { C V a R } _ { 0 . 1 }$ . Conditional regret is the corresponding oracle value minus the learned-rule value. For the mean criterion, $\ell _ { 1 } { \mathrm { - P L S } }$ performs best among the reported methods, consistent with the linear mean treatment contrast. For $Q _ { 0 . 1 }$ and $\mathrm { C V a R } _ { 0 . 1 }$ , DRF performs best and Kernel shows similar tail-criterion gains. These results illustrate criterion-adaptive decision making with flexible distribution estimation in this nonlinear-tail setting.

Table 19: Treatment-interactive simulation with $n = 8 0 0$ and $p = 2 .$ . Entries are Monte Carlo means (standard errors) over 100 replications.
<table><tr><td>Criterion</td><td>Method</td><td>Value</td><td>Conditional regret</td><td>OverallErr</td></tr><tr><td rowspan="4">Mean</td><td> $\ell _ { 1 } { - } \mathrm { P L } S$ </td><td>2.229(0.001)</td><td>0.011(0.001)</td><td>0.056(0.003)</td></tr><tr><td>RWL-linear</td><td>2.219(0.004)</td><td>0.021 (0.004)</td><td>0.074(0.005)</td></tr><tr><td>Kernel</td><td>2.212(0.002)</td><td>0.028(0.002)</td><td>0.078(0.003)</td></tr><tr><td>DRF</td><td>2.202(0.003)</td><td>0.038(0.003)</td><td>0.104(0.004)</td></tr><tr><td rowspan="3"> $Q _ { 0 . 1 }$ </td><td>CQR (0.1)</td><td>0.455(0.005)</td><td>0.313(0.005)</td><td>0.396(0.004)</td></tr><tr><td>Kernel</td><td>0.729(0.002)</td><td>0.039(0.002)</td><td>0.115(0.002)</td></tr><tr><td>DRF</td><td>0.730(0.003)</td><td>0.038(0.003)</td><td>0.099(0.003)</td></tr><tr><td rowspan="3"> $\mathrm { C V a R } _ { 0 . 1 }$ </td><td>Qi-CVaR (0.1)</td><td>0.461 (0.002)</td><td>0.242(0.002)</td><td>0.296(0.002)</td></tr><tr><td>Kernel</td><td>0.658(0.002)</td><td>0.044(0.002)</td><td>0.134(0.002)</td></tr><tr><td>DRF</td><td>0.666(0.003)</td><td>0.036(0.002)</td><td>0.117(0.003)</td></tr></table>

## B.5 Tuning and Preflight Diagnostics

Non-generative methods. The following protocol applies to Settings 1–3 and their higherdimensional extensions. Kernel selects a bandwidth shared by both treatment arms by ten-fold cross-validation over $\{ 0 . 1 5 , 0 . 2 0 , 0 . 2 5 , 0 . 3 0 , 0 . 4 0 , 0 . 5 0 , 0 . 7 0 , 0 . 9 \dot { 0 } , 1 . 2 0 , 1 . 5 0 , 2 . 0 0 \}$ , minimizing the mean squared CDF prediction error at empirical outcome quantiles for 20 equally spaced probability levels in [0.1, 0.9], using all covariates. $\ell _ { 1 } { \mathrm { - P L S } }$ selects its penalty along the glmnet path by ten-fold cross-validation, maximizing the summed validation-fold policy value. RWL uses a linear kernel and five-fold policy-value cross-validation over $\{ 2 ^ { - 1 0 } , \ldots , 2 ^ { 0 } \} . \ell _ { 2 }$ -CVaR uses the same grid with five-fold cross-validation, CQR initialization within each training fold, and its original validation score, which under $\mu = 0$ is the mean outcome among validation observations whose treatment agrees with the learned rule; it is then refit on the full training sample. CQR fits linear quantile regression with treatment interactions at $\tau \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 \}$ without tuning. DRF uses 500 trees per treatment arm and otherwise the defaults of drf 1.3.1.

Generative methods. Conditional DDPM and GCDS are tuned on independent pilot data. We measure predictive accuracy by the continuous ranked probability score (CRPS; lower is better) on factual validation outcomes, averaged within and then equally across treatment arms, and define $\mathrm { s k i l l } = 1 - \mathrm { C R P S } _ { \mathrm { m o d e l } } / \mathrm { C R P S } _ { \mathrm { r e f } }$ , where the reference is the covariate-free empirical outcome distribution of the corresponding training arm. Each candidate is fit on five pilot datasets with two training seeds each, using treatment-stratified 80/20 training–validation splits and 512 generated draws per validation covariate and arm. A candidate is eligible if at least 9 of its 10 runs succeed and its scores are finite on all five datasets; Conditional DDPM additionally requires mean skill of at least −0.10. Conditional DDPM searches learning rates $\{ 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } \}$ and hidden widths {64, 128}, selecting by a one-standard-error rule that favors smaller networks, with standard errors computed across the five dataset-level averages. GCDS searches generator and discriminator learning rates in $\{ 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \} \times \{ 1 0 ^ { - 4 } , 5 \times \mathbf { \check { l } } 0 ^ { - 4 } \}$ and, since its candidates share one architecture, selects the eligible candidate with the lowest mean CRPS. Checkpoints are assessed every 100 epochs up to 2,000 for Conditional DDPM and every 500 iterations up to 7,500 for GCDS; the final budget is the median selected checkpoint of the chosen candidate, rounded to the checkpoint interval. Formal fits use the full training sample and this budget without further checkpoint selection. Test outcomes and oracle policies are not used for tuning.

Both methods use batch size 128. Conditional DDPM uses 200 diffusion steps, v-prediction, a cosine noise schedule, EMA decay 0.999, and 50-step deterministic DDIM sampling. GCDS uses three-dimensional Gaussian noise, generator hidden width 50, discriminator hidden widths (50, 25), and five discriminator updates per generator update. Selected configurations are listed in Table 20.

Table 20: Selected generative configurations. All Conditional DDPM models use hidden width 64. GCDS learning rates are given as (generator, discriminator). <sup>†</sup>Did not pass the preflight check. –: not evaluated.
<table><tr><td rowspan="2">Setting p</td><td rowspan="2"></td><td colspan="2">Conditional DDPM</td><td colspan="2">GCDS</td></tr><tr><td>LR</td><td>Epochs</td><td>LR</td><td>Iterations</td></tr><tr><td>1</td><td>10</td><td> $1 0 ^ { - 4 }$ </td><td>900</td><td> $( 1 0 ^ { - 3 } , 5 { \times } 1 0 ^ { - 4 } )$ </td><td>500</td></tr><tr><td>2</td><td>10</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>800</td><td> $( 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>1500</td></tr><tr><td>3</td><td>10</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>1600†</td><td> $( 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>2500†</td></tr><tr><td>1</td><td>20</td><td> $1 0 ^ { - 4 }$ </td><td>1000</td><td> $( 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>1500</td></tr><tr><td>1</td><td>50</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>300</td><td> $( 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>1000</td></tr><tr><td>1</td><td>100</td><td> $1 0 ^ { - 4 }$ </td><td>300</td><td> $( 1 0 ^ { - 3 } , 1 0 ^ { - 4 } )$ </td><td>6500†</td></tr><tr><td>1</td><td>300</td><td> $1 0 ^ { - 4 }$ </td><td>200</td><td> $( 1 0 ^ { - 3 } , 1 0 ^ { - 4 } )$ </td><td>3000†</td></tr><tr><td>2</td><td>20</td><td> $1 0 ^ { - 4 }$ </td><td>700</td><td> $( \mathrm { \dot { 5 } } { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>1000</td></tr><tr><td>2</td><td>300</td><td> $1 0 ^ { - 4 }$ </td><td>100†</td><td> $( 1 0 ^ { - 3 } , 1 0 ^ { - 4 } )$ </td><td>5500†</td></tr><tr><td>4</td><td>10</td><td>一</td><td>一</td><td> $( \mathrm { \dot { 5 } } { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>3500</td></tr><tr><td>4</td><td>20</td><td>一</td><td>一</td><td> $( 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>1000</td></tr><tr><td>5</td><td>10</td><td>一</td><td>一</td><td> $( 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>4500</td></tr><tr><td>5</td><td>20</td><td></td><td></td><td> $( 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 4 } )$ </td><td>1000</td></tr></table>

Preflight check. Before formal evaluation, each selected configuration was run 20 times with independent seeds and treatment-stratified 80/20 splits, using no formal test data. It passed if at least 19 runs completed with finite validation CRPS and the median skill was nonnegative. This check assesses execution reliability and predictive quality. Although our protocol originally required passing it before formal evaluation, we evaluated all configurations, including those that did not pass (Table 20), and report all replications without performance-based exclusion. In formal evaluation, all Conditional DDPM runs were evaluable. One of 1,000 GCDS runs (Setting 2, p = 300) failed a generated-sample collapse check and could not be evaluated, so the GCDS entries in Table 13 average over 99 replications. Some fitted policies are also degenerate: in Setting 2 at $p = 3 0 0$ $\mathrm { D R F s \ C V a R _ { 0 . 1 } }$ rule assigns treatment −1 to every test subject in all 100 replications (Err 0.499); Conditional DDPM’s mean-targeted rule is constant in 5 of 100 replications in Setting 1 at $p = 3 0 0$ and in 48 of 100 in Setting 2 at p = 300. All such results are retained in the reported tables.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately summarize the paper’s main contributions: the distribution-first framework, the theoretical analysis, and the empirical comparisons. They also reflect the scope of the paper and the criteria considered.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses several limitations, including the current focus on Höldercontinuous treatment criteria, the plug-in nature of the framework, and the dependence of generative-model gains on the outcome regime and target criterion. It also outlines corresponding future directions, such as doubly robust extensions, sharper theory for specific criteria, and broader data settings.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Each theoretical result is stated under explicit assumptions, and full proofs are provided in the appendix. The main text also summarizes the role of the assumptions and the resulting error propagation and regret bounds.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper specifies the simulation settings, evaluation criteria, competing methods, and the real-data analysis setup needed to reproduce the main empirical results. It also describes the first-stage estimators and the overall distribution-first pipeline used in each experiment.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The paper provides access to the code and specifies the data source, simulation data-generation process, sample settings, and experimental configurations for the main results.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies the simulation settings, sample sizes, data splits, evaluation criteria, and the main implementation choices for each experiment. These details are sufficient to understand how the results were obtained.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Simulation results are reported with Monte Carlo standard errors across repli cations. ACTG175 utilities and treatment rates are reported with standard deviations across 200 paired splits, and runtimes with medians and interquartile ranges; these descriptive summaries do not establish statistical significance.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The appendix describes the compute environment used for the experiments, including the cluster setting, CPU/GPU usage, software environment, and the main factors affecting computational cost. It also clarifies that the reported compute corresponds to the experiments included in the paper.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics and are not aware of any aspect of this work that violates it. The paper uses standard simulation studies and a public clinical trial dataset, with no human-subject interaction or other special ethical concerns beyond those discussed in the paper.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The paper is primarily a foundational methodological study, and we do not identify an immediate direct path to societal harm from the proposed framework itself. The paper discusses its potential positive relevance to precision medicine.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release high-risk models, scraped datasets, or other resources with an obvious dual-use or misuse risk. It is a methodological study based on simulations and an existing clinical trial dataset.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper credits the original sources of the ACTG175 dataset and the existing methods used for comparison through citations in the main text. We use publicly available data and standard research baselines, and no third-party assets are redistributed as part of this work.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not release new datasets, models, or other standalone assets. It is a methodological study based on simulations and existing data sources.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects. It uses simulation studies and an existing clinical trial dataset.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or prospective research with human subjects. It uses simulation studies and an existing de-identified clinical trial dataset.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core methodology of the paper does not use LLMs as an important, original, or non-standard component. Any LLM use was limited to writing or editing assistance and does not affect the scientific contribution of the work.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.