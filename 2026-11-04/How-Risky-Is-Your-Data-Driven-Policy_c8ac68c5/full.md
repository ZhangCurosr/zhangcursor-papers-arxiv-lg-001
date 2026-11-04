# How Risky Is Your Data-Driven Policy?

Omar Besbes and Alon Rieger

## Abstract

Data-driven algorithms are ubiquitous across industries, and there is an extensive literature on analyzing their average performance. The risk profile of such algorithms, while critical, is much less understood. This paper quantifies the risk of data-driven policies and isolates its sources. We evaluate a policy through the conditional value-at-risk (CVaR) of its loss over the training sample, any policy randomization, and the future outcome, at a fixed unknown distribution. Regret against an oracle tha knows the distribution decomposes into expected post-sample regret (PS), the average outcome-only CVaR gap of the selected action, and a sample-risk premium (SR), the additional tail risk omitted by averaging those evaluations. The analogous premium vanishes under expected loss.

We characterize when the premium vanishes and how large it is relative to post-sample regret. For losses with a continuous density near the tail cutof, we identify conditions under which both components decay at the same rate, yet the limiting ratio of the premium to post-sample regret can reach the thousands. When a probability mass at the cutof moves with the action, the premium can decay more slowly and dominate post-sample regret. We establish conditions under which sample average approximation learns the oracle action at rate $n ^ { - 1 / 2 }$ in both settings, but its combined-risk regret is of order $n ^ { - 1 }$ in the first and $n ^ { - 1 / 2 }$ in the second. In finite-sample numerical examples of SAA in both settings, the premium accounts for approximately 74%–97% of regret.

keywords: data-driven decisions, conditional value-at-risk, sample average approximation, regret decomposition

## 1 Introduction

Data-driven decision procedures are now deployed at scale. A retailer fits an order quantity to last year’s sales for each of thousands of products; a cloud provider sizes capacity from recent usage; an insurer sets a retention from a window of historical claims. Each procedure is retrained and redeployed many times, and in each deployment the realized loss depends on two things: the data the procedure was trained on, and the outcome it then faces. Under an expected-loss objective, averaging performance conditional on the training sample recovers the overall expected loss. Under a tail-risk objective, averaging conditional risk evaluations need not recover the risk of the learning procedure. We ask how large this diference can be and when it becomes a leading component of regret.

Tail risk is the natural criterion when the costly events are rare: a stockout that loses a customer, a capacity shortfall that breaches a service level, a reserve that falls short of a claim. Conditional Value-at-Risk (CVaR), the average loss in the worst (1 − �) fraction of cases, is a standard expression of that criterion. When the distribution of the outcome is known, the risk-averse problem is well understood. The action is fixed in advance, and the only way to be unlucky is to draw a bad outcome.

Learning the action from a finite sample changes what unlucky means. In the retailer’s case, a slow year produces a low order, and an ordinary week then arrives as a stockout. The demand was not unusual; the sample was. Conversely, a week of unusually high demand does little harm when the sample ran high as well and the order already covers it. For a learned policy, large losses can reflect a mismatch between the sample that chose the action and the outcome that tests it, and any randomization inside the policy is one more draw of the same kind. If risk means being unlucky, the unlucky event is the combined draw of the sample, any randomization, and the future outcome. We call the CVaR of the loss over that combined draw the combined-risk CVaR, following the total CVaR of Lu et al. (2015). It is what a practitioner estimates by pooling realized losses across deployments, as opposed to the within-deployment CVaR of the action a particular sample happened to produce.

The question becomes precise once the regret is split. The regret of a learning policy against an oracle that knows the distribution is the sum of two nonnegative pieces. The first is the expected post-sample regret: the gap between the outcome-only CVaR of the selected action and the oracle value, averaged over samples and any policy randomization. This component already captures the efect of learning on the quality of the selected action. It is the quantity that most of the data-driven optimization literature studies. The second is a sample-risk premium: the diference between combined-risk CVaR and the average outcome-only CVaR of the selected actions. The analogous premium is zero under an expected-loss criterion. Under CVaR, it can dominate regret even when the selected actions are nearly optimal. The premium can be the whole of the regret. Section 3 gives a two-action example in which every action the policy returns is optimal, so expected post-sample regret is zero, while the combined-risk evaluation still pays a strictly positive premium. The training sample selects diferent actions on diferent draws.

Three questions organize the paper. When is the premium zero, and what exactly is charged when it is not? How do the two pieces scale with the sample size, and which features of the loss, the distribution, and the policy set their relative order? And how does the standard policy, sample average approximation, fare under the combined criterion? Each question is about the learning procedure rather than about any one action it produces. The usual assessment of a data-driven policy fixes the sample and evaluates the action it induced; that assessment sees the first piece and not the second. We answer the three questions for a general loss and a scalar action, at a fixed unknown distribution with no parametric family and no prior.

## 1.1 Main contributions

Quantifying the sample-risk premium. Building on total CVaR and known conditional-risk comparisons, discussed in Section 1.2, we quantify the premium relative to oracle regret for scalar decisions under a general loss, derive its sharp smooth and atomic expansions, and establish the resulting regret rates for sample average approximation. The premium measures the additional tail risk from evaluating those losses together, under one threshold shared across samples, beyond the average risk of the selected actions. It vanishes exactly when there is a threshold that minimizes the variational objective for almost every selected action, and therefore when the action is fixed; under expectation the analogous premium vanishes for every policy. The expected post-sample regret vanishes when the returned action is an oracle action. A nonasymptotic upper bound relates the premium to the dispersion of the induced loss cutofs. The local results below then identify when the premium is comparable to, dominates, or persists beyond post-sample regret.

The order and magnitude of each component. We determine how the two pieces scale with the sample size, and which features of the loss, the distribution, and the policy set their relative order. We characterize four representative regimes at a fixed distribution, under their respective regularity and policy-concentration conditions. The smooth regime is the one in which the cutof moves to first order with the action and the loss has a density there. Both components are then of order $r _ { n } ^ { 2 }$ when the action concentrates at rate $r _ { n }$ The premium is an additional variance penalty: a common tail threshold across samples gives a larger local action curvature than a threshold re-optimized for each selected action. The ratio of these curvatures is $1 + K$ . For centered limiting fluctuations, the ratio of the premium to expected post-sample regret tends to �. For squared loss with lognormal outcomes, Table 1 gives $K \approx 3 9 0 0$ at $\delta = 0 . 9 9$ and $K \approx 5 0 0 0 0$ at $\delta = 0 . 9 9 9$ . These are asymptotic coeficients; they do not specify the premium at a given sample size. The atomic regime is the one in which an atom sits at the cutof and moves with the action, and there the expected post-sample regret is of order $r _ { n } ^ { 2 }$ and the premium is of order $r _ { n }$ . The mixing regime is the one in which the policy keeps visiting several optimal actions with no common minimizing threshold, and there the premium tends to a positive constant. The displaced regime is the one in which the policy stays of a unique well-separated optimum, and there the expected post-sample regret is bounded away from zero.

Sample average approximation. Under the conditions of Proposition 4.2, SAA learns the oracle action at rate $n ^ { - 1 / 2 }$ in both regimes. Nevertheless, its combined-risk regret is $\Theta ( n ^ { - 1 } )$ in the smooth regime and $\Theta ( n ^ { - 1 / 2 } )$ in the atomic regime, where the premium dominates. We identify the limiting variance of the action and the resulting atomic premium constant. Finite-sample examples show that the premium accounts for most regret in both regimes: Monte Carlo estimates give shares of about 74%–94% in bounded smooth squared-loss examples, while calculations from the exact distribution of the uniform-demand newsvendor order give shares of 76% at $n = 1 1 0$ and 95% at $n = 1 6 3 0$ . These results show why the action’s learning rate alone does not describe the policy’s tail-risk performance.

## 1.2 Related literature

Coherent risk measures (Artzner et al. 1999) and the variational representation of CVaR (Rockafellar and Uryasev 2002) are the language of the papers below. The closest connections concern total-CVaR and risk estimation, conditional-risk and quantile-score decompositions, Bayesian and contextual risk, and empirical optimization.

The total-CVaR perspective. A plug-in evaluation treats the estimated distribution as known, so the risk it reports is the risk of the future outcome alone. The papers below bring the risk contributed by the data into the evaluation. Lu et al. (2015) do so by changing the objective. Under the name total CVaR, they apply CVaR to the profit of the historical sample and next-period demand taken together, explicitly contrast that criterion with expected conditional CVaR, and optimize operational statistics for it, for a repeated newsvendor with parametric location-scale demand. Building on this criterion, we compare the sample-risk premium with oracle regret for scalar decisions under a general loss, derive its smooth and atomic expansions, and establish the resulting regret rates for sample average approximation.

Pitera and Schmidt (2018) show that plug-in estimators of value-at-risk and expected shortfall can underestimate risk. They correct it by an acceptance criterion: the reported position has to be acceptable as a risk capital. Francioni and Herzog (2012) require that the next observation exceed the estimated quantile with the nominal probability, for every value of the parameter. The quantile of the fitted distribution fails that requirement, and for the normal distribution a distortion that uses the sampling distribution of the estimate restores it. Gerrard and Tsanakas (2011) study how often a threshold set to a nominal tail probability fails, as in a capital or inventory rule. Parameter error can raise that frequency. For increasing transformations of location-scale families the frequency can be computed without knowing the true parameter. A tighter nominal level for the plug-in threshold and a threshold taken from the Bayesian predictive distribution both achieve a prescribed failure probability; under the prior proportional to 1/�, the predictive threshold meets the nominal level exactly. Bignozzi and Tsanakas (2016) subtract the estimated capital from the loss and measure the risk that remains. That remainder is the further capital that makes the position acceptable. Rules based on the parametric bootstrap and on the predictive distribution reduce it, and for location-scale families a capital estimator can remove it.

The risk-estimation papers above calibrate data-dependent capital, quantile, or threshold rules to a nominal standard, accounting for sampling randomness in that calibration. Our analysis uses the regret decomposition to quantify how much of the excess risk comes from the premium, including when the selected actions approach an oracle action.

The premium also difers from the optimizer’s curse: selection using noisy value estimates can make the selected alternative’s estimated performance optimistic (Smith and Winkler 2006). Here the premium is a diference between true population risk evaluations of a policy, and can be positive even when every selected action is optimal.

Conditional risk and quantile-score decompositions. The nonnegativity of the premium follows from known conditional-risk comparisons. For a coherent distortion risk measure, the risk of an outcome is at least the average of the risks computed after extra information (Iancu et al. 2015). Expected shortfall has the same property because it is concave under mixtures (Embrechts et al. 2021, Pertaia and Uryasev 2019, Tselishchev 2019), and the same comparison follows for optimized certainty equivalents from their infimum-of-expectations representation, under the required integrability conditions (Ben-Tal and Teboulle

2007). Taking the information to be the action makes the diference the premium. The same diference is $( 1 - \delta ) ^ { - 1 }$ times the resolution term of a quantile score, the gain from letting the cutof depend on side information (Bentzien and Friederichs 2014, Ehm and Ovcharov 2017, Pohle 2020). For a fixed policy, this identity follows by taking its realized loss as the forecast target, its action as side information, and the standard pinball loss as the quantile score. The local behavior of quantile-score excess, including quadratic and linear growth under diferent distributional conditions, is also studied in Steinwart and Christmann (2011). Our analysis relates the premium to the movement of the loss cutof, the curvature of the outcome-only objective, and the dispersion of the learned action. Comparing it with post-sample regret yields the smooth and atomic orders and the coeficient � under the respective regularity conditions. The mixing regime further identifies when the premium remains positive as post-sample regret vanishes.

Composite and Bayesian risk. Bayesian approaches condition on the observed sample and represent uncertainty about the outcome distribution through a posterior. Some apply an outer risk measure to a parameter-dependent performance criterion; others evaluate risk under the posterior predictive distribution of future outcomes. These constructions difer in how they aggregate parameter and outcome uncertainty.

Qian et al. (2019) nest two risk measures. The inner measure prices the decision when the parameter of the outcome distribution is known. The outer measure prices the fact that the parameter must be estimated. Stochastic programming, robust optimization, and distributionally robust optimization are the cases in which one of the two measures is an expectation or a worst-case value. Lin et al. (2022b) apply a risk functional, nested over the Bayesian posterior of an unknown transition distribution, to the expected total cost of a finite-horizon Markov decision process, and solve the CVaR case by an approximate dynamic program. Ma et al. (2026) use the same nesting for stochastic control: the inner risk is conditional on a latent environment parameter, the outer risk is taken under the posterior, and the action may depend on that posterior. Bodnar et al. (2020, 2022) and Kim and Kang (2021) compute portfolio value-at-risk and CVaR from the posterior predictive distribution of returns, so the allocation is a function of the observed sample. Jaiswal et al. (2023) replace the expectation in the Bayes risk by an entropic risk over the posterior and give rates for the approximate posterior and for the decision, with the newsvendor among the examples. Sadatsafavi et al. (2022) measure the value of a larger development sample by the net-benefit gap between the predictions of a fitted risk model and the correct predictions.

Our evaluation takes a frequentist perspective before the sample is observed. The distribution � is fixed and unknown, and no prior is placed on it. Combined-risk CVaR is the tail risk of the loss under the combined draw of the sample given �, the action that sample induces, and the future outcome. The distinction is in the conditioning: posterior and predictive evaluations condition on the observed sample, whereas combined-risk CVaR evaluates repeated draws of the sample and the future outcome at the same fixed �. It is distinct from the nested value that first replaces each action by its outcome-only CVaR and then takes the tail of that number: either one can be larger.

Contextual risk. Tao et al. (2025) study when a nested assessment of contextual and outcome uncertainty agrees with one risk evaluation of the two sources taken together. Their revised analysis also considers integrated learning and optimization. Here the information is a training sample, independent of the future outcome at a fixed �, and the distribution of the learned action changes with the sample size. Our focus is on regret against a distribution-knowing oracle and on how its post-sample and sample-risk components compare as the sample grows.

Outcome-only empirical optimization. A large literature approximates an unknown distribution by the sample and then evaluates risk on future outcomes only. Sampling variability enters as estimation error. For CVaR objectives, Meng et al. (2010) show that under continuous distributions, sample-average optimal values converge to the population optimum and limit points of sample-average stationary points are stationary for the true problem. Shapiro (2013) shows that, under mild moment and regularity conditions, limit points of sample-average solutions lie in the set of outcome-only optima. Dentcheva et al. (2017) and Guigues et al. (2018) refine this analysis with limit theorems for the estimation error, and Banholzer et al. (2022) give rates of convergence in mean under Lipschitz and moment conditions. Data-driven inventory applications include expected-profit optimization with an empirical value-at-risk constraint (Lin et al. 2022a), support-vector formulations of CVaR newsvendor criteria (Chen 2024), a feature-based CVaR newsvendor (Liu and Zhu 2024), and high-probability near-optimality guarantees for risk-averse multiperiod inventory (Jiang et al. 2026). Across this literature, the guarantees concern an optimizer of an outcome-only risk. Lee et al. (2020) give high-probability excess population-risk guarantees for empirical optimized-certainty-equivalent minimization, including CVaR, while Soma and Yoshida (2020) directly bound expected excess population CVaR. These results control the post-sample component, directly or after averaging over training samples. Such guarantees account for sampling variability through the population risk of the selected action. They do not by themselves characterize the additional sample-risk premium, which can determine the order of combined-risk regret. Our SAA analysis connects the concentration of the learned action to the rates of both components.

Related work studies the distribution of cumulative regret in sequential learning. Fan and Glynn (2025) and Simchi-Levi et al. (2025) show how expected-regret performance can conceal substantial tail risk arising from the learning process. Here we study the loss of a learned action on an independent future outcome and decompose its combined-risk CVaR regret into expected post-sample regret and a sample-risk premium.

The companion paper (Besbes and Rieger 2026) specializes the decomposition and the asymptotic regimes developed here to the newsvendor. It studies worst-case regret over bounded demand distributions and optimizes policies within a class based on order statistics. For every fixed $\delta \in ( 0 , 1 )$ , the minimax rate over all data-driven policies is $\Theta ( n ^ { - 1 / 4 } )$ , while at $\delta = 0$ it is $\Theta ( n ^ { - 1 / 2 } )$ , and uniform windows of width $n ^ { 3 / 4 }$ attain the rate $\Theta ( n ^ { - 1 / 4 } )$ . The present paper develops the structural and local asymptotic analysis for a general loss at a fixed distribution.

## 2 Problem Formulation

We consider a stochastic optimization problem with a nonempty Borel action set $\chi \subseteq \mathbb { R }$ . The decision maker chooses $x \in \chi$ before a random outcome $\xi \in \mathbb { R }$ is realized. We use $c ( x , \xi )$ to denote the loss incurred by the decision maker.

CVaR. We adopt Conditional Value-at-Risk (CVaR) as our performance criterion, a tail-risk measure that focuses on large losses.

Definition 2.1 (CVaR). For $\delta \in ( 0 , 1 )$ and an integrable random variable � with CDF $F _ { X }$ , define

$$
C V a R _ { \delta } [ X ] ~ = ~ \frac { 1 } { 1 - \delta } \cdot \int _ { \delta } ^ { 1 } F _ { X } ^ { - 1 } \left( u \right) d u .\tag{2.1}
$$

Here, $F _ { X } ^ { - 1 } ( u ) = \operatorname* { i n f } \{ y : F _ { X } ( y ) \geq u \}$ is the $u ^ { t h }$ quantile of �.

The Value-at-Risk at level �, denoted by $\operatorname { V a R } _ { \delta } [ X ]$ , is the corresponding quantile,

$$
\operatorname { V a R } _ { \delta } [ X ] : = F _ { X } ^ { - 1 } ( \delta ) .
$$

Equivalently, it is the smallest loss threshold � such that $\mathbb { P } ( X \leq t ) \geq \delta$

Throughout, $\delta \in ( 0 , 1 )$ . The excluded endpoint $\delta \ : = \ : 0$ is the risk-neutral case, $\operatorname { C V a R } _ { 0 } [ X ] = \mathbb { E } [ X ]$ in which the criterion is the expectation and the premium studied below vanishes identically by the tower property.

Informally, CVaR is the expected loss conditional on the loss lying in the worst $( 1 - \delta )$ -tail of its distribution. When the distribution has no probability mass at the $\delta ^ { t h }$ quantile, CVaR admits the following commonly studied expression $\operatorname { C V a R } _ { \delta } [ X ] = \mathbb { E } [ X \mid X \geq F _ { X } ^ { - 1 } ( \delta ) ]$ . When there is an atom at the cutof, the definition includes only the fraction of that atom needed to obtain a tail of probability $1 - \delta$

An equivalent representation of CVaR was shown by Rockafellar and Uryasev (2002, Theorem 10):

$$
{ \mathrm { C V a R } } _ { \delta } [ X ] = \operatorname* { i n f } _ { t \in \mathbb { R } } \left\{ t + { \frac { 1 } { 1 - \delta } } \cdot \mathbb { E } \left[ ( X - t ) ^ { + } \right] \right\} .\tag{2.2}
$$

For an integrable � and $\delta \in ( 0 , 1 )$ , the infimum is attained at $t = \mathsf { V a R } _ { \delta } [ X ]$ . If several thresholds attain it, this Value-at-Risk is the smallest of them. We call that threshold the cutof of the loss, and the cutof of an action � is the cutof of $c ( x , \xi )$

The same representation applies when the loss is generated by a random action. Below we evaluate it over the training sample, any policy randomization, and the future outcome together.

Data and policies. In the present paper, we consider the case in which the distribution � is unknown to the decision maker, but the decision maker has access to � i.i.d. samples $\xi _ { 1 } ^ { n } : = ( \xi _ { 1 } , \ldots , \xi _ { n } )$ from �. We consider

a class Π of data-driven policies, each a measurable probability kernel from $\mathbb { R } ^ { n }$ to X. A policy maps the sample to a distribution in $\mathcal { P } ( \chi )$ , the set of probability distributions on $\chi$ . We suppress dependence on � in the policy notation. In particular, a policy $\pi \in \Pi$ specifies

$$
\pi : \xi _ { 1 } ^ { n } \mapsto H _ { \xi _ { 1 } ^ { n } } ,
$$

where $H _ { \xi _ { 1 } ^ { n } } \in { \mathcal { P } } ( \chi )$ is the distribution of the action conditional on the sample. For an observed sample $\xi _ { 1 } ^ { n }$ the distribution $\pi ( \xi _ { 1 } ^ { n } )$ specifies a possibly randomized action. We identify a deterministic policy with the point mass at its prescribed action. Note that even for a deterministic policy, the resulting decision $x ^ { \pi }$ is a random variable because it depends on the random sample $\xi _ { 1 } ^ { n }$ . The decision maker commits to the policy before observing the sample.

We let $\xi _ { 1 } ^ { n } \sim F$ generate the sample, $x ^ { \pi } \sim \pi ( \xi _ { 1 } ^ { n } )$ denote the resulting decision, and $\xi \sim F$ denote the future outcome realization, drawn independently of the sample and of any randomization in the policy; the induced loss random variable is $c ( x ^ { \pi } , \xi )$

Oracle benchmark and combined-risk performance. When � is known, the decision maker minimizes $\mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x , \xi ) ]$ over �; we will refer to the optimal value as the oracle value defined by

$$
 \mathrm { O P T } _ { \delta } ( F ) = \operatorname * { i n f } _ { x \in \mathcal { X } } \mathrm { C V a R } _ { \delta , \xi \sim F } \left[ c \left( x , \xi \right) \right] .
$$

Here the decision � is deterministic, so $\mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x , \xi ) ]$ has a single source of randomness—the outcome realization $\xi \sim F$ . Independent randomization over actions cannot improve this value, as the comparison in Section 3 shows. The conditions maintained below are the following.

Assumption 2.2 (Maintained conditions). The loss $c$ is jointly measurable, with $\mathbb { E } _ { \xi \sim F } | c ( x , \xi ) | <$ ∞ for every $x \in { \mathcal { X } }$ and $\mathbb { E } | c ( x ^ { \pi } , \xi ) | < \infty ,$ , the second expectation over the sample, policy randomization, and future outcome. The future outcome $\xi \sim F$ is independent of the sample and of any policy randomization. The oracle value satisfies $O P T _ { \delta } ( F ) > - \infty$

It is convenient to name the fixed-action quantities in which the entire analysis below is expressed. For an action $x \in \chi$ , let

$$
\phi _ { F } ( x , t ) \ : = \ t + \frac { 1 } { 1 - \delta } \cdot \mathbb { E } _ { \xi \sim F } \left[ ( c \left( x , \xi \right) - t ) ^ { + } \right]\tag{2.3}
$$

denote the variational objective appearing in equation (2.2), and let

$$
Q _ { F } ( { \boldsymbol x } ) \ : = \ \mathrm { C V a R } _ { \delta , \boldsymbol { \xi } \sim F } \left[ c \left( { \boldsymbol x } , { \boldsymbol { \xi } } \right) \right] \ = \ \operatorname* { i n f } _ { t \in \mathbb { R } } \phi _ { F } ( { \boldsymbol x } , t ) \ , \qquad q _ { F } ( { \boldsymbol x } ) \ : = \ \mathrm { V a R } _ { \delta , \boldsymbol { \xi } \sim F } \left[ c \left( { \boldsymbol x } , { \boldsymbol { \xi } } \right) \right]\tag{2.4}
$$

denote, respectively, the outcome-only CVaR and the outcome-only VaR of the loss at the fixed action �. The infimum defining $Q _ { F } ( x )$ is attained at $t = q _ { F } ( x )$ . In this notation the oracle value is $\mathrm { { O P T } } _ { \delta } ( F ) = $ in $\dot { \cdot } _ { x \in X } Q _ { F } ( x )$ , and whenever this infimum is attained we write $x _ { F } ^ { * }$ for an optimal action. We state $x _ { F } ^ { * }$ with the subscript when it matters for the understanding, and otherwise write $x ^ { * }$ . Proposition 3.9 treats the case of several optimal actions.

In contrast to the oracle, the data-driven setting carries two layers of randomness: the sample $\xi _ { 1 } ^ { n }$ , which feeds into the action $x ^ { \pi }$ , and the independent future outcome $\xi ,$ both governed by �. Following the total-CVaR perspective of Lu et al. (2015), expressed there as lower-tail CVaR of profit, we evaluate upper-tail CVaR of loss over both sources of randomness. The tail-risk value of a policy under � is therefore

$$
\mathrm { C V a R } _ { \delta } [ c \left( x ^ { \pi } , \xi \right) ] \ : = \ \operatorname* { i n f } _ { t \in \mathbb { R } } \left\{ t + \frac { 1 } { 1 - \delta } \cdot \mathbb { E } _ { \xi _ { 1 } ^ { n } \sim F , x ^ { \pi } \sim \pi \left( \xi _ { 1 } ^ { n } \right) , \xi \sim F } \left[ \left( c \left( x ^ { \pi } , \xi \right) - t \right) ^ { + } \right] \right\} \ .\tag{2.5}
$$

Throughout the paper, $\operatorname { C V a R } _ { \delta } [ \cdot ]$ without a distribution subscript denotes evaluation over the combined randomness of sample, policy, and outcome; outcome-only evaluation at a fixed action is written $\mathrm { C V a R } _ { \delta , \xi \sim F } [ \cdot ]$ equivalently $Q _ { F } ( \cdot )$

Lipschitz condition. Assumption 2.2 is in force throughout. The smooth and atomic results in Section 3 use, in addition, the following Lipschitz condition. There is a nonnegative random variable $L ( \xi )$ with $\mathbb { E } [ L ( \xi ) ] < \infty$ such that

$$
\begin{array} { r l r } { \left| c ( x , \xi ) - c ( y , \xi ) \right| } & { { } \le } & { L ( \xi ) \left| x - y \right| } \end{array}\tag{2.6}
$$

for �-almost every $\xi ,$ simultaneously for all $x , y \in X$ . Additional moment and regularity conditions are stated with the relevant results. The SAA results in Section 4 use their own stated assumptions.

We define the regret of a data-driven policy as the nonnegative diference between the CVaR induced by the policy and the oracle value $\mathrm { { O P T } } _ { \delta } ( F )$ when the distribution is known

$$
{ \mathcal R } _ { \delta , n } \left( \pi , F \right) ~ = ~ { \bf C } { \bf V } \mathrm { a } { \bf R } _ { \delta } [ c \left( x ^ { \pi } , \xi \right) ] - { \bf O } { \bf P } \mathrm { T } _ { \delta } ( F ) ~ .\tag{2.7}
$$

Nonnegativity follows from the comparison in Section 3. This regret evaluates the learning procedure before the sample and any policy randomization are realized. The tail of the combined loss depends on both the sample-induced action and the future outcome, which remain independent. Variation across samples can increase combined-risk CVaR even when every selected action is optimal under its outcome-only CVaR.

For an operational interpretation, consider independent deployments under the same distribution � and policy �, with the sample size � fixed. Each deployment draws a fresh training sample, selects an action, and records the loss from an independent future outcome. As the number of deployments grows, the empirical CVaR of these losses converges to the combined-risk value in equation (2.5). If instead the realized sample and action are held fixed and outcomes are independently redrawn from $F ,$ the empirical CVaR converges to the outcome-only value $Q _ { F } ( x ^ { \pi } )$ .

## 3 Decomposition of combined-risk regret

After the sample is observed and the policy has selected an action, one may instead evaluate tail risk only over the future realization. Fix $\pi \in \Pi$ . Independence of the future outcome gives the post-sample outcome-only CVaR $\Psi _ { n } ^ { \pi } = Q _ { F } ( x ^ { \pi } )$ , with conditional variational representation

$$
\Psi _ { n } ^ { \pi } : = \operatorname* { i n f } _ { t \in \mathbb { R } } \left\{ t + \frac { 1 } { 1 - \delta } \mathbb { E } _ { \xi \sim F } \big [ \left( c \left( x ^ { \pi } , \xi \right) - t \right) ^ { + } \big | \xi _ { 1 } ^ { n } , x ^ { \pi } \big ] \right\} .\tag{3.1}
$$

This value is fixed once the action is selected and random across samples and policy randomization.

Comparison and decomposition. The following comparison and resulting identity isolate the two components of combined-risk regret. Let $X : = c ( x ^ { \pi } , \xi )$ , and let $t ^ { * }$ attain the variational representation of $\operatorname { C V a R } _ { \delta } [ X ]$ in equation (2.2). For every realized $( \xi _ { 1 } ^ { n } , x ^ { \pi } )$

$$
\Psi _ { n } ^ { \pi } \leq t ^ { * } + \frac { 1 } { 1 - \delta } \mathbb { E } \big [ ( X - t ^ { * } ) ^ { + } \big | \xi _ { 1 } ^ { n } , x ^ { \pi } \big ] ,
$$

because the conditional variational formula defining $\Psi _ { n } ^ { \pi }$ minimizes over all thresholds, and $t ^ { * }$ is one feasible threshold. Taking expectations over $( \xi _ { 1 } ^ { n } , x ^ { \pi } )$ yields

$$
\mathbb { E } \xi _ { 1 } ^ { n } , x ^ { \pi } \left[ \Psi _ { n } ^ { \pi } \right] \ \leq \ \mathbf { C V a R } _ { \delta } [ c \left( x ^ { \pi } , \xi \right) ] .\tag{3.2}
$$

Adding and subtracting $\mathbb { E } _ { \xi _ { 1 } ^ { n } , x ^ { \pi } } [ \Psi _ { n } ^ { \pi } ]$ in (2.7) gives

$$
\begin{array} { r l } { \mathcal { R } _ { \delta , n } \left( \pi , F \right) = } & { \underbrace { \left( \mathbb { E } _ { \xi _ { 1 } ^ { n } , x ^ { \pi } } \left[ \Psi _ { n } ^ { \pi } \right] - \mathrm { O P T } _ { \delta } \left( F \right) \right) } _ { \mathrm { P S } _ { \delta , n } \left( \pi , F \right) - \mathrm { e x p e c t e d p o s t - s a m p l e r g e t } } + \underbrace { \left( \mathrm { C V a R } _ { \delta } [ c \left( x ^ { \pi } , \xi \right) ] - \mathbb { E } _ { \xi _ { 1 } ^ { n } , x ^ { \pi } } \left[ \Psi _ { n } ^ { \pi } \right] \right) } _ { \mathrm { S R } _ { \delta , n } \left( \pi , F \right) - \mathrm { s a m p l e - f i s k p r e m i u m } } . } \end{array}\tag{3.3}
$$

Both brackets are nonnegative. For every realized $( \xi _ { 1 } ^ { n } , x ^ { \pi } )$ the induced action is a feasible deterministic action, so $\Psi _ { n } ^ { \pi } \ge \mathrm { O P T } _ { \delta } ( F )$ , and taking expectations establishes nonnegativity of the first. We call it the expected post-sample regret $\mathrm { P S } _ { \delta , n } ( \pi , F )$ : the expected outcome-only tail-risk gap of the selected action relative to the oracle, averaged over samples and policy randomization. The second bracket is nonnegative by equation (3.2), and it measures something the first cannot see. Combined-risk CVaR evaluates the induced loss over the sample, the policy randomization, and the future outcome together. Its excess over the average outcome-only CVaR can be positive even when every selected action is optimal, as Example 3.1 below shows. Under expected cost this excess vanishes by the tower property. We call this additional severity the sample-risk premium $\operatorname { S R } _ { \delta , n } ( \pi , F )$ . For independent deployments under the same �, �, and �, the average of the outcome-only values $\Psi _ { n } ^ { \pi }$ converges to $\mathbb { E } [ \Psi _ { n } ^ { \pi } ]$ . The gap between this limit and the combined-risk CVaR is $\operatorname { S R } _ { \delta , n } ( \pi , F )$

The premium as the price of one shared cutof. The premium admits a form that makes this additional severity explicit. Because the induced action $x ^ { \pi }$ is independent of the future outcome $\xi \sim F$ , both terms in the premium can be expressed through the fixed-action variational objective $\phi _ { F } ( x , t )$ from equation (2.3). Combined-risk CVaR in equation (2.5) optimizes over a single threshold that has to serve every realized action at once, so that it equals inf<sub>�</sub> $\mathbb { E } _ { x ^ { \pi } } [ \phi _ { F } ( x ^ { \pi } , t ) ]$ . The post-sample evaluation minimizes the threshold separately for each realized action, giving $\begin{array} { r } { \Psi _ { n } ^ { \pi } = \operatorname* { i n f } _ { t } \phi _ { F } ( x ^ { \pi } , t ) = Q _ { F } ( x ^ { \pi } ) } \end{array}$ . Subtracting the second from the first,

$$
\operatorname { S R } _ { \delta , n } ( \pi , F ) ~ = ~ \operatorname* { i n f } _ { t \in \mathbb { R } } \mathbb { E } _ { x ^ { \pi } } [ \phi _ { F } ( x ^ { \pi } , t ) ] ~ - ~ \mathbb { E } _ { x ^ { \pi } } \left[ \operatorname* { i n f } _ { t \in \mathbb { R } } \phi _ { F } ( x ^ { \pi } , t ) \right] ~ = ~ \operatorname* { i n f } _ { t \in \mathbb { R } } \mathbb { E } _ { x ^ { \pi } } [ \phi _ { F } ( x ^ { \pi } , t ) ] ~ - ~ \mathbb { E } _ { x ^ { \pi } } [ Q _ { F } ( x ^ { \pi } ) ]\tag{3.4}
$$

The premium is the cost of using a single threshold across the induced loss distributions: it is the gap between the infimum of an expectation and the expectation of the pointwise infimum. The threshold is an auxiliary variable in the CVaR representation. Example 3.1 makes that gap strict. Both actions are optimal for outcome-only CVaR, with minimizing cutofs 1 and 0, so the diference in $_ { ( 3 . 4 ) }$ is positive and the combined-risk regret equals the premium.

Example 3.1 (Premium without a bad action). Fix $\delta = 1 / 2$ , let $X = \{ 0 , 1 \}$ , and let $\xi$ equal 0 or 2 with probabilities $3 / 4$ and $1 / 4 .$ . Set $c ( 0 , \xi ) = 1$ and $c ( 1 , \xi ) = \xi .$ . Both actions have outcome-only CVaR equal to 1, with cutofs 1 and 0. For any $n \geq 1$ , consider the deterministic policy that selects action 0 if the first training observation is 2 and action 1 otherwise. Against an independent future outcome, its losses 0, 1, and 2 have probabilities 9/16, 4/16, and $3 / 1 6 .$ Its combined-risk CVaR is therefore $5 / 4 \div$ expected post-sample regret is zero, while the sample-risk premium is $1 / 4$

The next proposition characterizes when a common threshold minimizes the objective for almost every induced action.

When each component vanishes. The following proposition characterizes when each component vanishes. Before we state it, we write

$$
\mathcal T _ { \delta } ( x ) \ : = \ \arg \operatorname* { m i n } _ { t } \phi _ { F } ( x , t )
$$

for the set of minimizing thresholds at the action �. The conditional tails induced by $x ^ { \pi }$ are aligned if one threshold lies in $\mathcal { T } _ { \delta } ( x ^ { \pi } )$ for almost every realized action, and misaligned otherwise.

Proposition 3.2 (Endpoints of the two components). Suppose that $x ^ { \pi }$ is independent of $\dot { \varsigma } \sim F$ and that

$$
\begin{array} { r } { \mathbb { E } _ { x ^ { \pi } , \xi } | c ( x ^ { \pi } , \xi ) | < \infty . } \end{array}
$$

Then the following hold.

1. SR ${ } _ { \cdot \delta , n } ( \pi , F ) = 0$ if and only if the conditional tails induced by $x ^ { \pi }$ are aligned. In particular, the premium vanishes whenever $x ^ { \pi }$ is degenerate.

2. Suppose in addition that the infimum defining $O P T _ { \delta } ( F )$ is attained, and let $\begin{array} { r } { X _ { F } ^ { * } : = \arg \operatorname* { m i n } _ { x \in X } Q _ { F } ( x ) } \end{array}$ denote the set of oracle actions. Then $\mathrm { P S } _ { \delta , n } ( \pi , F ) = 0$ if and only $i f x ^ { \pi } \in \mathcal { X } _ { F } ^ { * }$ almost surely. If the oracle action is unique, this holds if and only $i f x ^ { \pi } = x _ { F } ^ { * }$ almost surely.

In words: PS vanishes if and only if the policy plays an oracle action almost surely; SR vanishes if and only if the induced loss distributions admit a common minimizing threshold, and a degenerate action is always aligned. Distinct selected VaR values can still yield zero premium when their sets of minimizing thresholds overlap. The proof is in Appendix A.1. A complementary bound controls the premium at every sample size through the distribution of the induced cutof.

Proposition 3.3 (An upper bound on the sample-risk premium). Under the independence and integrability hypotheses of Proposition 3.2, let $Y = q _ { F } ( x ^ { \pi } )$ and $L _ { \delta } = \operatorname* { m a x } \{ 1 , \delta / ( 1 - \delta ) \}$ . Then � is integrable and

$$
0 \leq \mathrm { S R } _ { \delta , n } ( \pi , F ) \leq C V a R _ { \delta } [ Y ] - \mathbb { E } Y \leq L _ { \delta } \operatorname* { i n f } _ { t \in \mathbb { R } } \mathbb { E } | Y - t | .\tag{3.5}
$$

The proof is in Appendix A.2. Equality holds between the premium and $\mathrm { C V a R } _ { \delta } [ Y ] - \mathbb { E } Y$ when the loss is conditionally deterministic. That middle term keeps the asymmetric tail weighting that the absolutedeviation bound discards. Corollary ${ \mathrm { A . 1 } }$ records the rate that follows from the bound: under the Lipschitz condition (2.6), if $x ^ { \pi } = x ^ { * } + r _ { n } Z _ { n } , r _ { n } \longrightarrow 0 , \operatorname* { s u p } _ { n } \mathbb { E } | Z _ { n } | < \infty$ , and $q _ { F }$ is diferentiable at $x ^ { * }$ , then $\mathrm { S R } = O ( r _ { n } )$ Its proof is in Appendix A.3. The atomic regime below attains this upper order. The rate conclusion uses the cutof and moment conditions; other orders remain possible outside the regimes below.

We first establish the orders of the two components for policies concentrating on a unique oracle action. We then quantify their leading coeficients in the smooth and atomic regimes, and examine when a component remains positive in the limit.

## 3.1 The asymptotic order of the components

The curvature of the outcome-only objective determines the local cost of action error. The movement of the cutof and the behavior of the loss distribution around it determine the premium. The following results translate a policy concentration rate $r _ { n }$ into orders of regret under the stated moment and regularity conditions. Section 4 establishes such concentration for SAA under additional hypotheses.

Theorem 3.4 (Smooth regime). Under Assumption 2.2, suppose that equation (2.6) holds. Fix � and $\delta \in ( 0 , 1 )$ , and let $x ^ { * }$ be the unique oracle action. Suppose $x ^ { * }$ is interior and that $Q _ { F }$ is twice diferentiable at $x ^ { * }$ with $0 < Q _ { F } ^ { \prime \prime } ( x ^ { * } ) < \infty$ . Write $q ^ { * } : = q _ { F } ( x ^ { * } )$ . Suppose $c ( x , \xi )$ has a density $g _ { x } ( t )$ for $( x , t )$ near $( x ^ { * } , q ^ { * } )$ , that the map $( x , t ) \mapsto g _ { x } ( t )$ is continuous at $( x ^ { * } , q ^ { * } )$ , and that $0 < g _ { x ^ { * } } ( q ^ { * } ) < \infty .$ . Suppose the policy concentrates on $x ^ { * }$ at rate $r _ { n } ,$

$$
x ^ { \pi } = x ^ { * } + r _ { n } Z _ { n } , \qquad r _ { n } \searrow 0 , \qquad Z _ { n } \Rightarrow Z ,\tag{3.6}
$$

and that $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0 ,$ , so the cutof moves to first order with the action. Suppose $\big \{ | Z _ { n } | ^ { 2 } \big \}$ is uniformly integrable and � is nondegenerate. Then

$$
\begin{array} { r c l } { { \mathrm { P S } _ { \delta , n } ( \pi , F ) } } & { { = } } & { { \Theta \Bigl ( r _ { n } ^ { 2 } \Bigr ) \ , \qquad \mathrm { S R } _ { \delta , n } ( \pi , F ) } } \end{array} = \Theta \Bigl ( r _ { n } ^ { 2 } \Bigr ) \ .
$$

The proof is in Appendix A.4. Theorem 3.4 characterizes the regime in which the policy concentrates around a unique oracle action. The relative order of the two regret components is determined by three features: the local growth of the outcome-only CVaR objective away from the oracle action, the order at which the VaR moves with the action, and the local growth of the variational objective around the VaR. When the outcome-only CVaR objective is locally quadratic, the variational objective is locally quadratic around the VaR, and the VaR moves to first order with the action, the two regret components are of the same order.

A squared loss against a skewed outcome illustrates the smooth regime when the loss density satisfies the theorem’s conditions, the cutof slope is nonzero, and the oracle is unique and interior with positive finite curvature. Table 1 evaluates the local coeficient for this loss. Symmetry can instead make $q _ { F } ^ { \prime } ( x ^ { * } ) = 0$ outside the theorem’s scope. Proposition 3.6 identifies the limiting ratio under these same hypotheses. For centered limiting fluctuations, the coeficient can be thousands even though the two components have the same order.

Under the moving-atom, straddling, and isolation conditions below, the premium is linear in the concentration scale, while post-sample regret remains quadratic.

Theorem 3.5 (Atomic regime). Under Assumption 2.2, suppose that equation (2.6) holds. Let $x ^ { * }$ be the unique interior oracle action, and suppose $Q _ { F }$ is twice diferentiable there with $0 < Q _ { F } ^ { \prime \prime } ( x ^ { * } ) < \infty$ . Suppose the loss places an atom at a level $\ell ( x )$ , continuously diferentiable near $x ^ { * } ,$ , with $\ell ( x ^ { * } ) = q ^ { * } : = q _ { F } ( x ^ { * } )$ and $\ell ^ { \prime } ( x ^ { * } ) \neq 0 ,$ , and that the atom strictly straddles the tail level,

$$
\mathbb { P } \big ( c ( x ^ { * } , \xi ) < q ^ { * } \big ) < \delta < \mathbb { P } \big ( c ( x ^ { * } , \xi ) \leq q ^ { * } \big ) .
$$

As the action approaches the oracle and the margin around $\ell ( x )$ shrinks, the probability strictly above the atom and the probability at or above a level just below it settle at their oracle values,

$$
\begin{array} { r l r } & { } & { \displaystyle \operatorname* { l i m } _ { ( x , u ) \to ( x ^ { * } , 0 ^ { + } ) } \mathbb { P } \big ( c ( x , \xi ) > \ell ( x ) + u \big ) = \mathbb { P } \big ( c ( x ^ { * } , \xi ) > q ^ { * } \big ) , } \\ & { } & { \displaystyle \operatorname* { l i m } _ { ( x , u ) \to ( x ^ { * } , 0 ^ { + } ) } \mathbb { P } \big ( c ( x , \xi ) \geq \ell ( x ) - u \big ) = \mathbb { P } \big ( c ( x ^ { * } , \xi ) \geq q ^ { * } \big ) . } \end{array}\tag{3.7}
$$

These limits are the isolation assumption. Other losses may approach the atom; only the probability in a shrinking band is controlled. Suppose that the policy concentrates on $x ^ { * }$ as in equation (3.6), with $\left\{ \vert Z _ { n } \vert ^ { 2 } \right\}$ uniformly integrable and � nondegenerate. Then

$$
\begin{array} { r c l } { { \mathrm { P S } _ { \delta , n } ( \pi , F ) } } & { { = } } & { { \Theta \Bigl ( r _ { n } ^ { 2 } \Bigr ) \ , \qquad \mathrm { S R } _ { \delta , n } ( \pi , F ) } } \end{array} = \Theta ( r _ { n } ) \ .
$$

The proof is in Appendix A.14. The profit formulation of the newsvendor illustrates an atom created by the loss: when demand meets or exceeds the order, the loss is constant at a level that moves with the order. At the oracle, this atom strictly straddles the tail level; continuous demand gives the isolation condition (Besbes and Rieger 2026). With a unique interior oracle, a demand distribution diferentiable there with a positive finite derivative, and a policy satisfying the theorem’s concentration and moment conditions, the problem lies in the atomic regime. Multiple optimal orders can instead produce the mixing behavior discussed below. More generally, an atom yields the stated orders only under the full hypotheses of Theorem 3.5. Under these hypotheses, the sample-risk premium is the leading component of combined-risk regret: its ratio to expected post-sample regret diverges.

## 3.2 Mean squared error plus a �-weighted variance penalty

In the smooth regime, the common rate of decay does not determine the relative magnitude of the two components. The next proposition shows that PS is asymptotically proportional to the mean squared error of the action about $x ^ { * }$ , while SR is asymptotically proportional to its variance, and identifies the two constants.

Proposition 3.6 (Smooth-regime coeficients). Suppose that the hypotheses of Theorem 3.4 hold. Write $q ^ { * } : = q _ { F } ( x ^ { * } ) , g : = g _ { x ^ { * } } ( q ^ { * } ) , \mathbf { M S E } : = \mathbb { E } [ ( x ^ { \pi } - x ^ { * } ) ^ { 2 } ]$ , and $V : = V a r ( x ^ { \pi } )$ . Then

$$
\begin{array} { r } { \frac { \mathrm { P S } _ { \delta , n } ( \pi , F ) } { \mathrm { M S E } } \longrightarrow \frac { 1 } { 2 } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) , \qquad \frac { \mathrm { S R } _ { \delta , n } ( \pi , F ) } { V } \longrightarrow \frac { g \cdot q _ { F } ^ { \prime } ( x ^ { * } ) ^ { 2 } } { 2 ( 1 - \delta ) } . } \end{array}\tag{3.8}
$$

The proof is in Appendix A.8. The ratio of the two constants is the relative variance penalty,

$$
\boxed { K = \frac { g \cdot q _ { F } ^ { \prime } ( x ^ { * } ) ^ { 2 } } { ( 1 - \delta ) \ Q _ { F } ^ { \prime \prime } ( x ^ { * } ) } } .\tag{3.9}
$$

Together, these limits give the local expansion

$$
\begin{array} { r c l } { \operatorname { P S } _ { \delta , n } ( \pi , F ) + \operatorname { S R } _ { \delta , n } ( \pi , F ) } & { = } & { { \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) \left[ \operatorname { M S E } + K \cdot V \right] + o \Bigl ( r _ { n } ^ { 2 } \Bigr ) ~ . } \end{array}\tag{3.10}
$$

Since $\begin{array} { r } { \mathbf { M S E } = \left( \mathbb { E } { \left[ x ^ { \pi } - x ^ { * } \right] } \right) ^ { 2 } + V , } \end{array}$ , variance receives weight $1 + K$ relative to squared bias in the leading term.

Why the premium can be large. The coeficient compares two local curvatures. Let $A _ { F }$ be the action curvature when the threshold is held at $q ^ { * } = q _ { F } ( x ^ { * } )$ . The smooth local expansion gives

$$
\begin{array} { c } { { \phi _ { F } ( x ^ { * } + h , q ^ { * } ) = Q _ { F } ( x ^ { * } ) + \frac { 1 } { 2 } A _ { F } h ^ { 2 } + o ( h ^ { 2 } ) , } } \\ { { { } } } \\ { { A _ { F } = Q _ { F } ^ { \prime \prime } ( x ^ { * } ) + \displaystyle \frac { g q _ { F } ^ { \prime } ( x ^ { * } ) ^ { 2 } } { 1 - \delta } . } } \end{array}\tag{3.11}
$$

Thus $1 + K = A _ { F } / Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ . Post-sample evaluation allows a separate minimizing threshold for each selected action. Combined evaluation uses one threshold across samples; locally, it adjusts to the mean action but not to each fluctuation. In the leading regret term, squared bias is therefore charged at curvature $Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ and variance at $A _ { F }$ . The additional variance curvature in the premium is $A _ { F } - Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ . For the squared loss with lognormal outcomes in Table 1 at $\delta = 0 . 9 9$ , the optimized and fixed-threshold curvatures are approximately 1.85 and 7234. Appendix A.10 derives the expansion under the existing smooth-regime hypotheses.

To express the relative magnitude of the two components, write

$$
\rho _ { n } \because = \frac { \mathrm { S R } _ { \delta , n } ( \pi , F ) } { \mathrm { P S } _ { \delta , n } ( \pi , F ) }\tag{3.12}
$$

whenever $\mathrm { P S } _ { \delta , n } ( \pi , F ) > 0$ . Combined regret is then $ ( 1 + \rho _ { n } )$ times $\mathrm { P S } _ { \delta , n } ( \pi , F )$ . Under the smooth-regime hypotheses, the ratio converges to a positive finite constant whose magnitude can be large.

Corollary 3.7 (The limiting premium ratio in the smooth regime). Under the hypotheses of Proposition 3.6, suppose in addition that $\mathbb { E } [ Z ] = 0$ . Then

$$
\rho _ { n } \longrightarrow K ,\tag{3.13}
$$

with � the relative variance penalty in equation (3.9). Without the centering assumption the limit is $K \cdot V a r ( Z ) / \mathbb { E } [ Z ^ { 2 } ]$

The proof is in Appendix A.9. A nonzero mean of the limiting scaled error � lowers the limiting ratio by the factor $\mathrm { V a r } ( Z ) / \mathbb { E } [ Z ^ { 2 } ]$ . For fixed $( c , F , \delta )$ , � is the largest limiting ratio under Proposition 3.6, attained when $\mathbb { E } [ Z ] = 0$ . Table 1 evaluates � for squared loss using the formula in Proposition A.6. In the tabulated examples, � increases over the reported tail levels. It reaches the tens of thousands for the lognormal family, while lying below one for the gamma family at lower tail levels.

For centered limiting fluctuations, the premium’s share of combined-risk regret converges to $K / ( 1 + K )$ Thus the lognormal entry $K \approx 3 9 0 0$ at $\delta = 0 . 9 9$ corresponds to a limiting share of approximately 99.97%. The coeficient � describes a local limit; the results do not specify the sample size at which $\rho _ { n }$ is close to it. Finite-sample accuracy depends on both the spread of the learned action and the neighborhood over which the density and cutof expansions are accurate.

The zero-slope endpoint is diferent. If the other hypotheses of Theorem 3.4 hold but $q _ { F } ^ { \prime } ( x ^ { * } ) = 0$ , then $\mathrm { S R } \ = \ o ( r _ { n } ^ { 2 } )$ while $\mathrm { P S } = \Theta ( r _ { n } ^ { 2 } )$ . Proposition A.5 records these orders. Its proof evaluates the common threshold at $q ^ { * }$ . Thus action variability need not create a premium of the same order as post-sample regret.

## 3.3 When the cutof sits on an atom

The reduction in equation (3.10) prices a smooth cutof by mean squared error and a variance. Under the hypotheses of Theorem 3.5, the unique minimizing threshold near $x ^ { * }$ is the atom level, $q _ { F } ( x ) = \ell ( x )$ . Hence $q _ { F } ^ { \prime } ( x ^ { * } ) = \ell ^ { \prime } ( x ^ { * } ) \neq 0$ , and the variational objective grows linearly away from this threshold. The leading premium is then a weighted absolute deviation of the induced cutof. This dispersion is invariant under deterministic translations of the scaled cutof fluctuations, while the leading post-sample term continues to charge mean squared error. Write $p : = \mathbb { P } ( c ( x ^ { * } , \xi ) = q ^ { * } )$ for the mass at the cutof, and

<table><tr><td>Family</td><td> $\overline { { \delta = 0 . 8 0 } }$ </td><td> $\overline { { \delta = 0 . 9 0 } }$ </td><td> $\overline { { \delta = 0 . 9 5 } }$ </td><td> $\overline { { \delta = { \bf 0 . 9 9 } } }$ </td><td> $\overline { { \delta = \mathbf { 0 . 9 9 9 } } }$ </td></tr><tr><td>Exponential</td><td>1.56</td><td>3.93</td><td>8.76</td><td>47.9</td><td>492</td></tr><tr><td>Gamma</td><td>0.228</td><td>0.413</td><td>0.637</td><td>1.33</td><td>2.93</td></tr><tr><td>Lognormal</td><td>34.7</td><td>136</td><td>428</td><td>3900</td><td>49700</td></tr><tr><td>Pareto</td><td>4.46</td><td>13.4</td><td>36.6</td><td>330</td><td>7200</td></tr><tr><td>Weibull</td><td>6.21</td><td>22.1</td><td>70.5</td><td>884</td><td>28700</td></tr></table>

Table 1: The relative variance penalty � at the oracle action for $c ( x , \xi ) = ( \xi - x ) ^ { 2 } / 2$ , from (A.19), with the oracle action and cutof located numerically. For the limiting-ratio interpretation, take a compact action interval with the oracle in its interior and a policy satisfying Theorem 3.4 with centered limiting fluctuations. Then the entries are the limit of $\rho _ { n } = \mathrm { S R } _ { \delta , n } ( \pi , F ) / \mathrm { P S } _ { \delta , n } ( \pi , F )$ . The coeficient is invariant to location– scale changes of the outcome for this squared loss. Exponential has rate 1; gamma has shape 4 and scale $1 / 8 ;$ lognormal has log $\xi \sim \mathcal { N } ( 0 , 1 . 8 ^ { 2 } )$ ; Pareto is Type I with tail index 3 and scale 1; Weibull has shape 0.7 and scale 1.

$$
\tau : = \frac { \delta - \mathbb { P } \big ( c ( x ^ { * } , \xi ) < q ^ { * } \big ) } { p }\tag{3.14}
$$

for the fraction of that mass lying below �. The fraction above � is $1 - \tau$ . For any integrable random variable �, define $\begin{array} { r } { \mathcal { W } ( Y ) : = p \operatorname* { i n f } _ { t \in \mathbb { R } } \mathbb { E } [ \tau ( Y - t ) ^ { + } + ( 1 - \tau ) ( t - Y ) ^ { + } ] } \end{array}$ . A �-quantile attains the infimum. Letting $q _ { \tau }$ be such a quantile of the induced VaR, this gives

$$
\begin{array} { r l r } { \mathcal { W } ( q _ { F } ( x ^ { \pi } ) ) } & { = } & { p \mathbb { E } \left[ \tau \big ( q _ { F } ( x ^ { \pi } ) - q _ { \tau } \big ) ^ { + } + ( 1 - \tau ) \big ( q _ { \tau } - q _ { F } ( x ^ { \pi } ) \big ) ^ { + } \right] ~ . } \end{array}\tag{3.15}
$$

Proposition 3.8 (Atom at the cutof). Suppose that the hypotheses ofTheorem 3.5 hold, and write MSE as in Proposition 3.6. Then

$$
\frac { \mathrm { P S } _ { \delta , n } ( \pi , F ) } { \mathrm { M S E } } \longrightarrow \frac { _ 1 } { ^ 2 } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) \ , \qquad \frac { \mathrm { S R } _ { \delta , n } ( \pi , F ) } { \mathcal { W } ( q _ { F } ( x ^ { \pi } ) ) } \longrightarrow \frac { 1 } { 1 - \delta } .\tag{3.16}
$$

The proof is in Appendix A.16. The induced VaR moves with the action because $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0$ . When the two gaps in the straddling hypothesis are equal, $\tau = 1 / 2$ and $\mathcal { W } ( q _ { F } ( x ^ { \pi } ) )$ is $( p / 2 )$ times the mean absolute deviation of the induced VaR about a median. The two limits have diferent remainders, so they are recorded separately,

$$
\begin{array} { r c l c l } { { \mathrm { P S } _ { \delta , n } ( \pi , F ) } } & { { = } } & { { \frac { 1 } { 2 } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) ~ \mathrm { M S E } + o ( r _ { n } ^ { 2 } ) ~ , } } & { { } } & { { \mathrm { S R } _ { \delta , n } ( \pi , F ) } } & { { = } } & { { \displaystyle \frac { ~ \mathcal { W } ( q _ { F } ( x ^ { \pi } ) ) ~ } { 1 - \delta } + o ( r _ { n } ) ~ . } } \end{array}\tag{3.17}
$$

The error in the premium is of order $o ( r _ { n } )$ . That remainder can exceed the quadratic post-sample term, so the sum is an expansion of total regret up to $o ( r _ { n } )$ . At small action spreads the linear premium dominates the quadratic post-sample term, the same comparison the atomic regime records asymptotically.

The cutof expansion holds in first mean: $q _ { F } ( x ^ { \pi } ) - q ^ { * } = q _ { F } ^ { \prime } ( x ^ { * } ) r _ { n } Z _ { n } + o _ { L ^ { 1 } } ( r _ { n } )$ . Weak convergence and uniform integrability give convergence of the scaled cutof distributions in the Wasserstein-1 metric. The weighted deviation is continuous in this metric, positively homogeneous, and invariant under deterministic translations. Applying it with the same level � to $q _ { F } ^ { \prime } ( x ^ { * } ) Z$ yields

$$
r _ { n } \rho _ { n } \longrightarrow \frac { 2 \mathcal { W } \bigl ( q _ { F } ^ { \prime } ( x ^ { * } ) Z \bigr ) } { ( 1 - \delta ) Q _ { F } ^ { \prime \prime } ( x ^ { * } ) \mathbb { E } [ Z ^ { 2 } ] } .\tag{3.18}
$$

The weights are a property of $( c , F , \delta )$ at the oracle action and carry no sample size. The numerator of $\rho _ { n }$ is homogeneous of degree one and the denominator of degree two, so at a concentration rate $r _ { n }$ the ratio is of order $1 / { r _ { n } }$ , and equation (3.18) names the constant. The premium therefore outgrows $\mathrm { P S } _ { \delta , n } ( \pi , F )$ in order and not merely in constant, which is what Theorem 3.5 records. For the newsvendor loss under the conditions of Theorem 3.5, the ratio of the premium to expected post-sample regret grows like $1 / { r _ { n } }$ . Besbes and Rieger (2026) specialize Proposition 3.8 to that loss: they record the limit of PS over mean squared error and the limit of the premium over a weighted absolute deviation of the order.

## 3.4 Regimes in which a component does not vanish

Uniqueness of $x _ { F } ^ { * }$ , which Theorems 3.4 and 3.5 both assume, excludes a qualitatively diferent regime. When several actions attain the oracle value, a data-driven policy may converge to the set of optimal actions without converging to any single one of them. In that case, the expected post-sample regret may vanish even though the induced action remains nondegenerate. If no threshold minimizes the variational objective for almost every limiting action, the sample-risk premium can converge to a strictly positive constant. The following proposition formalizes this possibility.

Proposition 3.9. Suppose that the unconditional action distribution, over the sample and any policy randomization, satisfies $x ^ { \pi } \Rightarrow x _ { \infty } ^ { \pi }$ , where $x _ { \infty } ^ { \pi }$ is supported on the set of population-optimal actions. Suppose further that $x \mapsto c ( x , \xi )$ is continuous at $x _ { \infty } ^ { \pi }$ almost surely, and that thefamily

$$
\{ | c ( x ^ { \pi } , \xi ) | \} _ { n \geq 1 }
$$

is uniformly integrable. Then

$$
\mathrm { P S } _ { \delta , n } ( \pi , F ) \longrightarrow 0 .
$$

Moreover, if

$$
\begin{array} { r } { \mathbb { P } \big ( t \in \mathcal { T } _ { \delta } ( x _ { \infty } ^ { \pi } ) \big ) < 1 f o r e \nu e r y t \in \mathbb { R } , } \end{array}
$$

then

$$
\mathrm { S R } _ { \delta , n } ( \pi , F ) \longrightarrow \mathrm { S R } _ { \delta , \infty } ( \pi , F ) > 0 .
$$

Consequently,

$$
\mathrm { P S } _ { \delta , n } ( \pi , F ) = o ( 1 ) , \qquad \mathrm { S R } _ { \delta , n } ( \pi , F ) = \Theta ( 1 ) .
$$

The proof is in Appendix A.17. The cost and � determine the optimal actions and their sets of minimizing thresholds. The policy determines whether its limiting action remains nondegenerate on that optimal set. Under the proposition’s hypotheses, this limiting mixture has vanishing post-sample regret and a positive premium precisely when it has no common minimizing threshold. A policy converging to a single optimal action has vanishing premium. For example, in the newsvendor problem, SAA at a particular Bernoull distribution keeps mixing optimal orders with no common minimizing threshold, and its premium tends to a positive constant. A window of order-statistic indices wider than the $\sqrt { n }$ fluctuations of the binomial count concentrates on one limiting optimal order and eliminates that persistent premium (Besbes and Rieger 2026).

The displaced regime keeps a unique oracle action and allows persistent displacement from it.

Proposition 3.10. Fix � and suppose that the population problem has a unique optimal action $x _ { F } ^ { * }$ satisfying

$$
\kappa _ { F } ( \varepsilon ) : = \operatorname* { i n f } _ { \stackrel { x \in X \colon } { | x - x _ { F } ^ { * } | \geq \varepsilon } } \left\{ C V a R _ { \delta , \xi \sim F } [ c ( x , \xi ) ] - O P T _ { \delta } ( F ) \right\} > 0
$$

for every $\varepsilon > 0 .$ . If, for some $\varepsilon > 0$ and $\eta > 0 ;$

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { \mathbb { P } _ { F } } \bigl ( | x ^ { \pi } - x _ { F } ^ { * } | \geq \varepsilon \bigr ) \geq \eta ,
$$

then

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } \mathrm { P S } _ { \delta , n } ( \pi , F ) \geq \eta \kappa _ { F } ( \varepsilon ) > 0 ,
$$

and hence

$$
\mathrm { P S } _ { \delta , n } ( \pi , F ) = \Omega ( 1 ) .
$$

We prove the proposition in Appendix A.19. The same bound yields the form used below: if $x _ { F } ^ { * }$ is well separated and $\mathrm { P S } _ { \delta , n } ( \pi , F ) \to 0$ , then $x ^ { \pi } \to x _ { F } ^ { * }$ in probability.

A policy enters this regime when it remains away from the oracle with probability bounded below. This occurs, for example, if it converges to a surrogate-objective optimum diferent from $x _ { F } ^ { * }$ , or to the best action in a restricted family whose optimum difers from $x _ { F } ^ { * }$ . Shrinking an otherwise consistent action toward a distinct fixed anchor with a weight converging to a positive constant also gives a displaced limit. Confining the policy to actions outside a neighborhood of $x _ { F } ^ { * }$ , while the oracle continues to minimize over X, forces persistent displacement directly. These mechanisms satisfy the proposition when the oracle is well separated.

## 3.5 A summary of the regimes

Theorems 3.4 and 3.5, Proposition 3.9 and Proposition 3.10 provide a spectrum of possibilities for the relative importance of the two regret components. The spectrum covers a wide variety of cost functions and distributions. However, it is not a partition: other combinations of the policy, the cost, and � are possible, and the decomposition is not characterized for every one of them. These regimes are summarized in Table 2.

<table><tr><td>Regime</td><td>Entered when</td><td>premium</td><td>Expected Sample-risk post-sample regret</td></tr><tr><td>Smooth</td><td>Unique optimum; the cutoff moves to first order and the loss has a positive finite density there; the policy concentrates at rate  $r _ { n } \searrow 0 .$ </td><td> $\overline { { \Theta ( r _ { n } ^ { 2 } ) } }$ </td><td> $\overline { { \Theta ( r _ { n } ^ { 2 } ) } }$ </td></tr><tr><td>Atomic</td><td>Unique optimum; an atom at the cutoff moves with the action; the policy concentrates at rate  $r _ { n } \searrow 0 .$ </td><td> $\Theta ( r _ { n } )$ </td><td> $\Theta ( r _ { n } ^ { 2 } )$ </td></tr><tr><td>Mixing</td><td>Several actions are optimal, and the policy keeps mix- ing over them, with no common minimizing threshold.</td><td>Θ(1)</td><td>o(1)</td></tr><tr><td>Displaced</td><td>Unique well-separated optimum, and the policy is per- sistently bounded away from it.</td><td>unrestricted</td><td>Ω(1)</td></tr></table>

Table 2: Structural regimes governing the relative magnitude of the sample-risk premium and expected postsample regret. In the displaced row, unrestricted means that Proposition 3.10 does not constrain the premium. Each row abbreviates the hypotheses of the corresponding result. The smooth and atomic rows also use positive finite curvature, the moment and nondegeneracy conditions, and, in the atomic row, strict straddling and the isolation limits. Concentration at a rate $r _ { n } \ \searrow \ 0$ implies consistency. Persistent displacement is strictly stronger than failure of consistency: an inconsistent policy need only sit away from $x _ { F } ^ { * }$ along a subsequence, which yields lim $\operatorname* { s u p } _ { n } \mathrm { P S } _ { \delta , n } ( \pi , F ) > 0$ rather than the $\mathrm { P S } _ { \delta , n } ( \pi , F ) = \Omega ( 1 )$ recorded in the row.

Under the stated hypotheses of the smooth and atomic regimes, the sample-risk premium is of at least the same order as expected post-sample regret. Equal orders do not imply similar magnitudes. For centered limiting fluctuations in the smooth regime, the premium ratio tends to �, which reaches thousands in Table 1. The atomic regime attains the $O ( \boldsymbol { r } _ { n } )$ upper order from Proposition 3.3 under the stated cutof and moment conditions. Other orders remain possible outside the representative regimes. Section 4 shows that sample average approximation is consistent at a unique well-separated oracle under a compact action set and a loss that is continuous in the action and uniformly bounded. The displaced regime is then avoided. In the atomic regime the sample-risk premium is of larger order than the expected post-sample regret.

## 4 Regret of SAA

We now establish the concentration properties of sample average approximation (SAA) needed to apply Section 3, and use them to determine the rates of its two regret components. Throughout this section, � and $\delta \in ( 0 , 1 )$ are fixed.

Sample average approximation replaces � by the empirical distribution ${ \hat { F } } _ { n }$ and minimizes the resulting CVaR. For $\xi \sim \hat { F } _ { n }$

$$
\operatorname { C V a R } _ { \delta , \xi \sim \hat { F } _ { n } } \left[ c \left( x , \xi \right) \right] ~ = ~ \operatorname* { i n f } _ { t } \left\{ t + \frac { 1 } { 1 - \delta } \cdot \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( c ( x , \xi _ { i } ) - t ) ^ { + } \right\}\tag{4.1}
$$

Sort the sample losses from worst to best as $c _ { [ 1 ] } ( x ) \geq \cdot \cdot \cdot \geq c _ { [ n ] } ( x )$ . The infimum over � reduces to a weighted average of the upper tail, with $m = \lceil ( 1 - \delta ) n \rceil$

$$
\mathrm { C V a R } _ { \delta , \xi \sim \hat { F } _ { n } } \left[ c \left( x , \xi \right) \right] = \frac { 1 } { 1 - \delta } \cdot \left[ \frac { 1 } { n } \cdot \sum _ { i = 1 } ^ { m - 1 } c _ { \left[ i \right] } \left( x \right) + \left( 1 - \delta - \frac { m - 1 } { n } \right) \cdot c _ { \left[ m \right] } \left( x \right) \right] .
$$

An SAA action is a measurable minimizer

$$
x ^ { \mathrm { S A A } } \in \mathop { \mathrm { a r g m i n } } _ { x \in \chi } \mathrm { C V a R } _ { \delta , \xi \sim \hat { F } _ { n } } [ c ( x , \xi ) ] .
$$

We assume a minimizer exists. If several actions attain $\operatorname { i t } , x ^ { \mathrm { S A A } }$ is the one returned by a fixed measurable tie-break.

SAA minimizes an empirical estimate of the outcome-only objective $Q _ { F } ( \cdot )$ . The ex-ante value of the procedure is the CVaR of losses recorded across independent deployments, each with its own training sample. That is the object in which the sample-risk premium appears, and it is not the objective in equation (4.1).

## 4.1 Post-sample consistency and the sample-risk premium

Under the conditions below, the expected outcome-only CVaR of the action selected by SAA converges to the oracle benchmark. Thus the post-sample component in equation (3.3) vanishes.

Consistency and convergence-in-mean results are well established in SAA analysis; see, for example, Banholzer et al. (2022). The proposition below records a bounded, continuous setting in which expected post-sample regret vanishes, providing the starting point for analyzing the two components of combined-risk regret.

## Proposition 4.1. Fix a distribution �. Suppose that:

1. The action set X is nonempty and compact.

2. The mapping $x \mapsto c ( x , \xi )$ is continuous on X for �-almost every $\xi .$

3. The loss is uniformly bounded under �: there exists $M < \infty$ such that

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } | c ( x , \xi ) | \leq M \qquad a l m o s t s u r e l y .
$$

Then

$$
\mathrm { { P S } } _ { \delta , n } ( \pi ^ { \mathrm { { S A A } } } , F ) = \mathbb { E } _ { \xi _ { 1 } ^ { n } \sim F } \bigg [ Q _ { F } \Big ( x ^ { \mathrm { { S A A } } } \Big ) \bigg ] - O P T _ { \delta } ( F ) \longrightarrow 0 .
$$

The proof is in Appendix B. Proposition 4.1 controls the post-sample component. With several optimal actions, persistent mixing without a common minimizing threshold can still retain a positive premium, as Proposition 3.9 shows. If the oracle is unique, continuity on the compact action set makes it well separated,

so Proposition 3.10 gives $x ^ { \mathrm { S A A } } \to x _ { F } ^ { * }$ in probability. Boundedness and almost-sure continuity then give convergence of the induced losses in mean; continuity of CVaR in the $L ^ { 1 }$ metric implies that combined-risk regret, and hence both components, vanish.

## 4.2 Concentration and regret rates

We next establish the concentration rate of SAA and determine how quickly each regret component vanishes. In addition to the structural conditions of the smooth or atomic regime, the rate uses almost-sure action diferentiability at $x _ { F } ^ { * }$ and a local Lipschitz envelope with a moment above two. Specifically, $\partial _ { x } c ( x _ { F } ^ { * } , \xi )$ exists for �-almost every $\xi ,$ and there is a nonnegative $\mu ( \xi )$ with $\mathbb { E } [ \mu ( \xi ) ^ { 2 + \eta } ] < \infty$ for some $\eta > 0$ such that, for �-almost every $\xi ,$ simultaneously for all $x , y$ in a fixed neighborhood of $x _ { F } ^ { * }$ ,

$$
\begin{array} { r l r } { \left| c ( x , \xi ) - c ( y , \xi ) \right| } & { \leq } & { \mu ( \xi ) \left| x - y \right| . } \end{array}\tag{4.2}
$$

These conditions control fluctuations of the empirical criterion near the oracle. For the squared loss $( x - \xi ) ^ { 2 }$ on a compact action interval, the derivative exists and one may take $\mu ( \xi ) = C ( 1 + | \xi | )$ for a suitable constant $C .$ . Bounded outcomes also ensure the uniform loss bound in Proposition 4.1 and all required moments. The unbounded families in Table 1 provide values of $K ;$ applying the SAA result below additionally requires its bounded-loss hypothesis.

Proposition 4.2. Suppose the hypotheses of Proposition 4.1, and let $x _ { F } ^ { * }$ be the unique interior, well-separated oracle action. Assume the hypotheses on $Q _ { F _ { \bot } }$ , the cutof, and the loss distribution in either Theorem 3.4 or Theorem 3.5. Suppose also that the loss satisfies (4.2) with $\mathbb { E } [ \mu ( \xi ) ^ { 2 + \eta } ] < \infty$ for some $\eta > 0 ;$ , and that $\partial _ { x } c ( x _ { F } ^ { * } , \xi )$ exists for �-almost every $\xi .$

Write $q ^ { * } = q _ { F } ( x _ { F } ^ { * } )$ and let $a _ { F } = q _ { F } ^ { \prime } ( x _ { F } ^ { * } )$ in the smooth regime and $a _ { F } = \ell ^ { \prime } ( x _ { F } ^ { * } )$ in the atomic regime. Define

$$
\begin{array} { l } { \displaystyle \psi _ { F } ( \xi ) = a _ { F } + \frac { \partial _ { x } c ( x _ { F } ^ { * } , \xi ) - a _ { F } } { 1 - \delta } \mathbf { 1 } _ { \{ c ( x _ { F } ^ { * } , \xi ) > q ^ { * } \} } , } \\ { \displaystyle \sigma ^ { 2 } = \frac { \mathrm { V a r } ( \psi _ { F } ( \xi ) ) } { Q _ { F } ^ { \prime \prime } ( x _ { F } ^ { * } ) ^ { 2 } } . } \end{array}\tag{4.3}
$$

Then $0 < \sigma ^ { 2 } < \infty$ , and

$$
\sqrt { n } ( x ^ { \mathrm { S A A } } - x _ { F } ^ { * } ) \Rightarrow N ( 0 , \sigma ^ { 2 } ) .
$$

Moreover, $\{ n ( x ^ { \mathrm { S A A } } - x _ { F } ^ { * } ) ^ { 2 } \} _ { n \geq 1 }$ is uniformly integrable. In particular,

$$
x ^ { \mathrm { S A A } } - x _ { F } ^ { * } = \Theta _ { \mathbb { P } } \left( n ^ { - 1 / 2 } \right) .
$$

The proof is in Appendix C. The normal limit and uniform integrability establish the concentration and moment conditions in Section 3 with $r _ { n } = n ^ { - 1 / 2 }$ . Bounded loss supplies the remainder estimates needed in

the proofs of Theorems 3.4 and 3.5, giving

$$
\mathrm { S m o o t h } \colon \mathrm { ~ \mathbb { P S } = \Theta ( \eta ^ { - 1 } ) , ~ \qquad ~ \mathrm { S R } = \Theta ( \eta ^ { - 1 } ) , }
$$

$$
\mathrm { A t o m i c } \mathrm { : \quad } \mathrm { P S } = \Theta ( n ^ { - 1 } ) , \qquad \mathrm { S R } = \Theta ( n ^ { - 1 / 2 } ) .
$$

The action converges at the same rate in both regimes, but combined-risk regret decays more slowly in the atomic regime because of the premium. In the smooth regime, the centered normal limit also gives $\rho _ { n } \to K$ by Corollary 3.7.

The newsvendor loss is piecewise linear in the order with uniformly bounded slopes, so � can be chosen as a deterministic constant. Continuous demand makes the action derivative exist almost surely at the oracle and yields the isolation limits (Besbes and Rieger 2026). On a compact order set with bounded continuous demand, the proposition therefore applies when the structural atomic conditions of Theorem 3.5 hold.

## 4.3 Finite-sample illustrations

We evaluate SAA at finite sample sizes in two smooth squared-loss examples and an atomic newsvendor example.

Smooth losses. Consider $c ( x , \xi ) = ( \xi - x ) ^ { 2 } / 2$ at $\delta \ : = \ : 0 . 9 5$ , with $X ~ = ~ [ 0 , B ]$ . The outcome is either exponential with rate 1, conditioned on $\xi \le 1 2$ , or lognormal with log $\xi \sim \cal N ( 0 , 1 . 8 ^ { 2 } )$ before conditioning on $\xi \leq 2 0 0 0 ;$ in each case � is the upper bound. These bounded examples satisfy Proposition 4.2, as verified in Appendix C.2. Table 3 reports estimates from 12,000 independent training samples for each row. Conditional risk evaluations use analytic truncated moments of the specified outcome distribution.

<table><tr><td>Outcome</td><td>K</td><td>n</td><td>SR/PS</td><td>Premium share (%)</td></tr><tr><td rowspan="3">Exponential</td><td>8.76</td><td>100</td><td>2.90 (0.05)</td><td>74.4 (0.31)</td></tr><tr><td></td><td>400</td><td>3.88 (0.04)</td><td>79.5 (0.16)</td></tr><tr><td></td><td>1600</td><td>5.59 (0.04)</td><td>84.8 (0.09)</td></tr><tr><td rowspan="3">Lognormal</td><td>418.42</td><td>100</td><td>14.08 (0.20)</td><td>93.4 (0.09)</td></tr><tr><td></td><td>400</td><td>12.52 (0.14)</td><td>92.6 (0.08)</td></tr><tr><td></td><td>1600</td><td>14.38 (0.15)</td><td>93.5 (0.06)</td></tr></table>

Table 3: Smooth-regime SAA with the truncated outcome distributions specified in the text. Premium share is $\mathrm { S R } / ( \mathrm { P S } + \mathrm { S R } )$ . Estimates use 12,000 independent training samples per row. Parentheses give approximate Monte Carlo standard errors, in percentage points for shares. The limiting coeficient � is computed from the population distribution. Appendix C.2 describes the computation.

The estimated premium share ranges from 74.4% to 93.5% across these cases. Yet the finite-sample ratios remain below �, especially for the lognormal outcome: at $n = 1 6 0 0$ , the estimated ratio is 14.38, compared with a limit of 418.42. Thus the premium can already account for most regret while its ratio to post-sample regret is far from its local asymptotic value.

An atom at the cutof. Consider the newsvendor loss $c ( x , \xi ) = x / 2 - \operatorname* { m i n } ( x , \xi )$ , with $\xi \sim U [ 0 , 1 ]$ $X \ = \ [ 0 , 1 ]$ , and $\delta \ : = \ : 0 . 9 .$ . The unique oracle order is $x _ { F } ^ { * } = 0 . 0 5$ , and the problem satisfies the atomic hypotheses. With the smallest-minimizer tie-break, SAA returns an order with exact sampling distribution

$$
x ^ { \mathrm { S A A } } = \xi _ { ( k ) } , \qquad k = \lceil n / 2 0 \rceil , \qquad \xi _ { ( k ) } \sim \mathrm { B e t a } ( k , n + 1 - k ) .\tag{4.4}
$$

Table 4 evaluates both regret components using this distribution. The premium accounts for about 76% of regret at $n = 1 1 0$ and 95% at $n = 1 6 3 0$ . The scaled components approach the constants predicted by Propositions 4.2 and 3.8. Appendix C.3 gives the formulas and verifies the assumptions.

<table><tr><td>n</td><td>nPS</td><td> $\overline { { { \sqrt { n } } \operatorname { S R } } }$ </td><td>SR/PS</td><td>Premium share</td></tr><tr><td>110</td><td>0.2548</td><td>0.0781</td><td>3.21</td><td>76.3%</td></tr><tr><td>430</td><td>0.2434</td><td>0.0962</td><td>8.19</td><td>89.1%</td></tr><tr><td>1630</td><td>0.2391</td><td>0.1041</td><td>17.58</td><td>94.6%</td></tr><tr><td>6430</td><td>0.2379</td><td>0.1078</td><td>36.34</td><td>97.3%</td></tr><tr><td>Limit</td><td>0.2375</td><td>0.1112</td><td> $\sim 0 . 4 6 8 3 \sqrt { n }$ </td><td>→ 100%</td></tr></table>

Table 4: Finite-sample SAA regret for uniform demand. Premium share is $\mathrm { S R } / ( \mathrm { P S } + \mathrm { S R } )$ . Values are evaluated numerically from the exact Beta distribution using incomplete Beta moments and a scalar quantile equation, without simulation.

## 5 Conclusion

This paper studies an ex-ante notion of tail risk for data-driven decisions. We evaluate CVaR over the combined randomness of training, policy randomization, and the future outcome, relative to an oracle that knows the distribution. The resulting combined-risk regret decomposes into two nonnegative terms: average post-sample regret and a sample-risk premium. The first averages the outcome-only CVaR gaps of the selected actions; it already captures how learning afects action quality. The second measures the additional tail risk omitted by this averaging. This premium can be positive even when every selected action is individually optimal under outcome-only CVaR.

The regime analysis shows how the premium can become the main component of regret. Under the smooth-regime hypotheses, the two components have the same asymptotic order. For centered limiting fluctuations, the ratio of the premium to post-sample regret tends to �, which can be large. The curvature comparison explains this diference: using one tail threshold across samples adds a variance penalty. The limit alone does not determine the ratio at a given sample size. Under the atomic-regime hypotheses, the premium decays more slowly than average post-sample regret and becomes the leading component. With persistent mixing among optimal actions that admit no common minimizing threshold, the premium can remain positive even as average post-sample regret vanishes.

Sample average approximation illustrates the distinction between action convergence and regret. Under Proposition 4.2, its action converges at rate $n ^ { - 1 / 2 }$ in both the smooth and atomic regimes, but its combinedrisk regret is of order $n ^ { - 1 }$ in the first and $n ^ { - 1 / 2 }$ in the second. The finite-sample examples in Tables 3 and 4 illustrate the importance of the premium in both regimes: it accounts for approximately $7 4 \% - 9 7 \%$ of regret in the reported cases. In the smooth examples, these large premium shares occur while the ratios remain below their limiting coeficients.

These regimes are representative settings at a fixed distribution, not an exhaustive classification. The smooth expansion also identifies a local design tradeof: at leading order, action variance receives weigh $1 + K$ relative to squared bias. Comparing policies through this expansion requires both to satisfy its hypotheses. Finite-sample guarantees for policy changes motivated by this tradeof, and statistical guarantees for estimating � and the cutof-based upper bound, remain questions for further work. Evaluating a learned policy through its average outcome-only risk can omit a leading component of excess tail risk.

## References

Artzner P, Delbaen F, Eber JM, Heath D (1999) Coherent measures of risk. Mathematical Finance 9(3):203–228. 3

Banholzer D, Fliege J, Werner R (2022) On rates of convergence for sample average approximations in the almost sure sense and in mean. Mathematical Programming 191:307–345, URL http://dx.doi.org/10.1007/ s10107-019-01400-4

Ben-Tal A, Teboulle M (2007) An old-new concept of convex risk measures: The optimized certainty equivalent. Mathematical Finance 17(3):449–476, URL http://dx.doi.org/10.1111/j.1467-9965.2007.00311.x. 4

Bentzien S, Friederichs P (2014) Decomposition and graphical portrayal of the quantile score. Quarterly Journal of the Royal Meteorological Society 140:1924–1934, URL http://dx.doi.org/10.1002/qj.2284. 5

Besbes O, Rieger A (2026) Beyond expected profit: How risk aversion alters learning in the data-driven newsvendor. Working paper. 6, 14, 17, 18, 22

Bignozzi V, Tsanakas A (2016) Parameter uncertainty and residual estimation risk. Journal of Risk and Insurance 83(4):949–978, URL http://dx.doi.org/10.1111/jori.12075. 4

Bodnar T, Lindholm M, Niklasson V, Thorsén E (2020) Bayesian quantile-based portfolio selection. Technical Report 2020:11, Department of Mathematics, Stockholm University, arXiv preprint arXiv:2012.01819. 5

Bodnar T, Lindholm M, Niklasson V, Thorsén E (2022) Bayesian portfolio selection using VaR and CVaR. Applied Mathematics and Computation 427:127120, URL http://dx.doi.org/10.1016/j.amc.2022.127120. 5

Chen ZY (2024) Data-driven risk-averse newsvendor problems: Developing the CVaR criteria and support vector machines. International Journal of Production Research 62(4):1221–1238, URL http://dx.doi.org/10. 1080/00207543.2023.2179350. 6

Dentcheva D, Penev S, Ruszczyński A (2017) Statistical estimation of composite risk functionals and risk optimization problems. Annals ofthe Institute ofStatistical Mathematics 69(4):737–760. 6

Ehm W, Ovcharov EY (2017) Bias-corrected score decomposition for generalized quantiles. Biometrika 104(2):473– 480, URL http://dx.doi.org/10.1093/biomet/asx004. 5

Embrechts P, Mao T, Wang Q, Wang R (2021) Bayes risk, elicitability, and the expected shortfall. Mathematical Finance 31(4):1190–1217, URL http://dx.doi.org/10.1111/mafi.12313. 4

Fan L, Glynn PW (2025) The fragility of optimized bandit algorithms. Operations Research 73(6):3173–3198. 6

Francioni I, Herzog F (2012) Probability-unbiased value-at-risk estimators. Quantitative Finance 12(5):755–768, URL http://dx.doi.org/10.1080/14697681003687569. 4

Gerrard RJG, Tsanakas A (2011) Failure probability under parameter uncertainty. Risk Analysis 31(5):727–744, URL http://dx.doi.org/10.1111/j.1539-6924.2010.01549.x. 4

Guigues V, Krätschmer V, Shapiro A (2018) A central limit theorem and hypotheses testing for risk averse stochastic programs. SIAM Journal on Optimization 28(2):1337–1366. 6

Iancu DA, Petrik M, Subramanian D (2015) Tight approximations of dynamic risk measures. Mathematics of Operations Research 40(3):655–682, URL http://dx.doi.org/10.1287/moor.2014.0689. 4

Jaiswal P, Honnappa H, Rao VA (2023) On the statistical consistency of risk-sensitive Bayesian decision-making. Advances in Neural Information Processing Systems, volume 36. 5

Jiang X, Tang LC, Ye Z, Zhang X (2026) Risk aversion in a data-driven multi-period inventory control problem. Production and Operations Management 35(8):3082–3097. 6

Kim D, Kang KH (2021) Conditional value-at-risk forecasts of an optimal foreign currency portfolio. International Journal ofForecasting 37(2):838–861, URL http://dx.doi.org/10.1016/j.ijforecast.2020.09.011. 5

Lee J, Park S, Shin J (2020) Learning bounds for risk-sensitive learning. Advances in Neural Information Processing Systems, volume 33. 6

Lin S, Chen YF, Li Y, Shen ZJM (2022a) Data-driven newsvendor problems regularized by a profit risk constraint. Production and Operations Management 31(4):1630–1644. 6

Lin Y, Ren Y, Zhou E (2022b) Bayesian risk Markov decision processes. Advances in Neural Information Processing Systems, volume 35. 5

Liu C, Zhu W (2024) Newsvendor conditional value-at-risk minimisation: A feature-based approach under adaptive data selection. European Journal ofOperational Research 313(2):548–564. 6

Lu M, Shanthikumar JG, Shen ZJM (2015) Technical note—operational statistics: Properties and the risk-averse case. Naval Research Logistics 62(3):206–214, URL http://dx.doi.org/10.1002/nav.21623. 2, 4, 9

Ma W, Chen Z, Xu H (2026) A Bayesian composite risk approach for stochastic optimal control and Markov decision processes. Mathematical Programming URL http://dx.doi.org/10.1007/s10107-026-02403-8, published online 19 August 2026. 5

Meng F, Sun J, Goh M (2010) Stochastic optimization problems with CVaR risk measure and their sample average approximation. Journal ofOptimization Theory and Applications 146(2):399–418, URL http://dx.doi.org/ 10.1007/s10957-010-9676-3. 6

Pertaia G, Uryasev S (2019) Fitting heavy-tailed mixture models with CVaR constraints. Dependence Modeling 7:365–374, URL http://dx.doi.org/10.1515/demo-2019-0019. 4

Pitera M, Schmidt T (2018) Unbiased estimation of risk. Journal of Banking & Finance 91:133–145, URL http: //dx.doi.org/10.1016/j.jbankfin.2018.04.016. 4

Pohle MO (2020) The Murphy decomposition and the calibration-resolution principle: A new perspective on forecast evaluation. ArXiv preprint arXiv:2005.01835. 5

Qian P, Wang Z, Wen Z (2019) A composite risk measure framework for decision making under uncertainty. Journal of the Operations Research Society of China 7(1):43–68, URL http://dx.doi.org/10.1007/

Rockafellar RT, Uryasev S (2002) Conditional value-at-risk for general loss distributions. Journal ofBanking & Finance 26(7):1443–1471, ISSN 0378-4266. 3, 7

Sadatsafavi M, Lee TY, Gustafson P (2022) Uncertainty and the value of information in risk prediction modeling. Medical Decision Making 42(5):661–671, URL http://dx.doi.org/10.1177/0272989X221078789. 5

Shapiro A (2013) Consistency of sample estimates of risk averse stochastic programs. Journal of Applied Probability 50(2):533–541. 6

Simchi-Levi D, Zheng Z, Zhu F (2025) A simple and optimal policy design with safety against heavy-tailed risk for stochastic bandits. Management Science 71(7):6298–6318, URL http://dx.doi.org/10.1287/mnsc.2022. 03512. 6

Smith JE, Winkler RL (2006) The optimizer’s curse: Skepticism and postdecision surprise in decision analysis. Management Science 52(3):311–322, URL http://dx.doi.org/10.1287/mnsc.1050.0451. 4

Soma T, Yoshida Y (2020) Statistical learning with conditional value at risk. ArXiv preprint arXiv:2002.05826. 6

Steinwart I, Christmann A (2011) Estimating conditional quantiles with the help ofthe pinball loss. Bernoulli 17(1):211– 225, URL http://dx.doi.org/10.3150/10-BEJ267. 5

Tao Y, Delage E, Xu H (2025) Risk-averse decision making with contextual information: Model, sample average approximation, and kernelization. URL http://dx.doi.org/10.48550/arXiv.2502.16607, arXiv preprint arXiv:2502.16607. 6

Tselishchev M (2019) On the concavity of expected shortfall. ArXiv preprint arXiv:1910.00640. 4

van der Vaart AW (1998) Asymptotic Statistics (Cambridge: Cambridge University Press), URL http://dx.doi. org/10.1017/CBO9780511802256. 50

## A Decomposition of regret

## A.1 Proof of Proposition 3.2

For $x \in \chi$ and $t \in \mathbb { R } .$ , write $\phi _ { x } ( t ) : = \phi _ { F } ( x , t )$ , so that

$$
Q _ { F } ( x ) = \operatorname* { m i n } _ { t \in \mathbb { R } } \phi _ { x } ( t ) , \qquad \mathcal { T } _ { \delta } ( x ) = \arg \operatorname* { m i n } _ { t \in \mathbb { R } } \phi _ { x } ( t ) .
$$

Both the minimum and the minimizing set are nonempty for almost every realization of $x ^ { \pi }$ , by integrability.

Part 1. By the independence of $x ^ { \pi }$ and $\xi$ and equation (3.4),

$$
\operatorname { S R } _ { \delta , n } ( \pi , F ) = \operatorname* { i n f } _ { t \in \mathbb { R } } G ( t ) , \qquad G ( t ) : = \mathbb { E } _ { x ^ { \pi } } \left[ \phi _ { x ^ { \pi } } ( t ) - \operatorname* { m i n } _ { s \in \mathbb { R } } \phi _ { x ^ { \pi } } ( s ) \right] ,
$$

and the integrand defining $G ( t )$ is nonnegative for every �. The quantity $\begin{array} { r } { \mathbb { E } _ { x ^ { \pi } } [ \operatorname* { m i n } _ { s } \phi _ { x ^ { \pi } } ( s ) ] = \mathbb { E } _ { x ^ { \pi } } [ Q _ { F } ( x ^ { \pi } ) ] } \end{array}$ is finite by integrability and does not depend on �, so � difers from $t \mapsto \mathbb { E } _ { x ^ { \pi } } [ \phi _ { x ^ { \pi } } ( t ) ]$ by a constant.

Consequently $G$ attains its infimum at any $t ^ { * }$ attaining the variational representation of $\operatorname { C V a R } _ { \delta } [ c ( x ^ { \pi } , \xi ) ]$ and such $\mathrm { ~ a ~ } t ^ { * }$ exists by integrability.

Suppose first that the tails are aligned, so that some $t ^ { \star }$ satisfies $t ^ { \star } \in \mathcal { T } _ { \delta } ( x ^ { \pi } )$ almost surely. Then $\begin{array} { r } { \phi _ { x ^ { \pi } } ( t ^ { \star } ) = \operatorname* { m i n } _ { s } \phi _ { x ^ { \pi } } ( s ) } \end{array}$ almost surely, hence $G ( t ^ { \star } ) = 0$ and the premium vanishes. Conversely, suppose the premium vanishes, so that $G ( t ^ { * } ) = 0$ . The integrand is nonnegative, whence $\begin{array} { r } { \phi _ { x ^ { \pi } } ( t ^ { * } ) = \operatorname* { m i n } _ { s } \phi _ { x ^ { \pi } } ( s ) } \end{array}$ almost surely, that is, $t ^ { \ast } \in \mathcal { T } _ { \delta } ( x ^ { \pi } )$ almost surely, and the tails are aligned. Finally, if $x ^ { \pi } = x _ { 0 }$ almost surely, then any element of $\mathcal { T } _ { \delta } ( x _ { 0 } )$ , a nonempty set, serves as a common threshold; the tails are aligned and the premium vanishes.

## Part 2. By definition,

$$
\begin{array} { r } { \mathrm { P S } _ { \delta , n } ( \pi , F ) = \mathbb { E } _ { x ^ { \pi } } \left[ Q _ { F } ( x ^ { \pi } ) \right] - \mathrm { O P T } _ { \delta } ( F ) = \mathbb { E } _ { x ^ { \pi } } \left[ Q _ { F } ( x ^ { \pi } ) - \mathrm { O P T } _ { \delta } ( F ) \right] , } \end{array}
$$

and $Q _ { F } ( x ) ~ \ge ~ \mathrm { O P T } _ { \delta } ( F )$ for every $x \in \chi .$ , so the integrand is nonnegative. Hence $\mathrm { P S } _ { \delta , n } ( \pi , F ) = 0$ if and only if $Q _ { F } ( x ^ { \pi } ) ~ = ~ \mathrm { O P T } _ { \delta } ( F )$ almost surely. Since the infimum defining $\mathrm { O P T } _ { \delta } ( F )$ is attained, $\{ x \in X : Q _ { F } ( x ) = \mathrm { O P T } _ { \delta } ( F ) \} = X _ { F } ^ { \ast }$ , so this is exactly $x ^ { \pi } \in \chi _ { F } ^ { * }$ almost surely. If the oracle action is unique, $X _ { F } ^ { * } = \{ x _ { F } ^ { * } \}$ and the condition reads $x ^ { \pi } = x _ { F } ^ { * }$ almost surely.

## A.2 Proof of Proposition 3.3

Write $Y = q _ { F } ( x ^ { \pi } )$ . Markov’s inequality gives

$$
| q _ { F } ( x ) | \leq \frac { \mathbb { E } _ { \xi } | c ( x , \xi ) | } { \operatorname* { m i n } \{ \delta , 1 - \delta \} } ,
$$

so $Y$ is integrable. The one-sided threshold derivatives of $\phi _ { F } ( x , t )$ lie in $[ - \delta / ( 1 - \delta ) , 1 ]$ . Since $q _ { F } ( x )$ minimizes this convex function,

$$
0 \leq \phi _ { F } ( x , t ) - Q _ { F } ( x ) \leq ( t - q _ { F } ( x ) ) ^ { + } + \frac { \delta } { 1 - \delta } ( q _ { F } ( x ) - t ) ^ { + } .
$$

Taking expectations and infima in $t ,$ and using the CVaR variational formula, gives

$$
\begin{array} { r l } & { 0 \le \mathrm { S R } \le \operatorname* { i n f } _ { t } \mathbb { E } \left[ ( t - Y ) ^ { + } + \displaystyle \frac { \delta } { 1 - \delta } ( Y - t ) ^ { + } \right] } \\ & { \qquad = \operatorname* { i n f } _ { t } \bigg \{ t + \displaystyle \frac { \mathbb { E } \big [ ( Y - t ) ^ { + } \big ] } { 1 - \delta } \bigg \} - \mathbb { E } Y = \mathrm { C V a R } _ { \delta } [ Y ] - \mathbb { E } Y . } \end{array}
$$

For a conditionally deterministic loss, $Q _ { F } ( x ) = q _ { F } ( x ) = c ( x , \xi )$ almost surely, so ${ \bf S } { \bf R } = { \bf C } { \bf V } { \bf a } { \bf R } _ { \delta } [ Y ] - \mathbb { E } Y$ Since $L _ { \delta } = \operatorname* { m a x } \{ 1 , \delta / ( 1 - \delta ) \}$ ,

$$
( t - Y ) ^ { + } + \frac { \delta } { 1 - \delta } ( Y - t ) ^ { + } \leq L _ { \delta } | Y - t | .
$$

Taking the expectation and the infimum in � gives the last inequality in equation (3.5).

## A.3 A linear rate from the premium bound

Corollary A.1 (Linear rate for the sample-risk premium). Maintain the hypotheses of Proposition 3.3 and the Lipschitz condition (2.6). Let $x ^ { * }$ be such that $q _ { F }$ is diferentiable at $x ^ { * }$ , and suppose $x ^ { \pi } = x ^ { * } + r _ { n } Z _ { n }$ with $\begin{array} { r } { r _ { n }  0 \ : a n d \ : \mathrm { s u p } _ { n } \mathbb { E } | Z _ { n } | < \infty } \end{array}$ . Then $\mathrm { S R } = O ( r _ { n } )$

Proof. Diferentiability at $x ^ { * }$ gives $| q _ { F } ( x ) - q _ { F } ( x ^ { * } ) | \leq C | x - x ^ { * } |$ in a fixed neighborhood. The Lipschitz envelope in (2.6) and Markov’s inequality in the proof of Proposition 3.3 give $| q _ { F } ( x ) - q _ { F } ( x ^ { * } ) | \leq C ^ { \prime } ( 1 +$ $\left| x - x ^ { * } \right| )$ globally. Outside that neighborhood the same comparison is bounded by a constant times $\left| x - x ^ { * } \right|$ Evaluating (3.5) at $t = q _ { F } ( x ^ { * } )$ therefore gives

$$
\mathrm { S R } \leq C ^ { \prime \prime } \mathbb { E } { \left| x ^ { \pi } - x ^ { * } \right| } = C ^ { \prime \prime } r _ { n } \mathbb { E } { \left| Z _ { n } \right| } = O ( r _ { n } ) .
$$

## A.4 Proof of Theorem 3.4

Both $\mathrm { P S } _ { \delta , n } ( \pi , F )$ and $\operatorname { S R } _ { \delta , n } ( \pi , F )$ are expectations, and the hypotheses control them near $x ^ { * }$ . The argument records that local control and the orders it induces, and then checks that the contribution of the event on which the induced action leaves a neighborhood of $x ^ { * }$ is $o \big ( r _ { n } ^ { 2 } \big )$ .

Twice diferentiability of $Q _ { F }$ at the interior minimizer $x ^ { * }$ , with $0 < Q _ { F } ^ { \prime \prime } ( x ^ { * } ) < \infty$ , makes $Q _ { F } ( x ) - Q _ { F } ( x ^ { * } )$ quadratic in $| x - x ^ { * } |$ . The assumption $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0$ makes the cutof move linearly with the action. A positive finite density of the loss, continuous at $( x ^ { * } , q ^ { * } )$ , makes the variational objective from equation (2.3) quadratic in the threshold. The following lemma turns those facts into bounds on a neighborhood of $x ^ { * }$ : the outcome-only CVaR grows like $| x - x ^ { * } | ^ { 2 }$ , the cutof moves linearly with the action, and the variational objective grows like the square of the distance from the cutof.

Lemma A.2 (Smooth exponents). Suppose that the hypotheses ofTheorem 3.4 hold. Write $q ^ { * } : = q _ { F } ( x ^ { * } )$ and $g : = g _ { x ^ { * } } ( q ^ { * } )$ . For every $\theta > 0$ there is a neighborhood $N : = \left[ x ^ { * } - \eta , x ^ { * } + \eta \right]$ of �<sup>∗</sup>, together with constants $0 < c _ { 2 } \leq C _ { 2 } < \infty$ and $0 < c _ { \gamma } \leq C _ { \gamma } < \infty ,$ , a threshold radius $u _ { 0 } > 0 ,$ , and $\varepsilon \geq 0 ,$ , such that for every $x \in N ,$

$$
\begin{array} { r l r } { c _ { 2 } \left| x - x ^ { * } \right| ^ { 2 } } & { \leq } & { Q _ { F } ( x ) - Q _ { F } ( x ^ { * } ) ~ \leq ~ C _ { 2 } \left| x - x ^ { * } \right| ^ { 2 } ~ , } \end{array}\tag{A.1}
$$

$$
\begin{array} { r l r } { \left| q _ { F } ( x ) - q ^ { * } - q _ { F } ^ { \prime } ( x ^ { * } ) \left( x - x ^ { * } \right) \right| } & { { } \le } & { \varepsilon \left| x - x ^ { * } \right| , } \end{array}\tag{A.2}
$$

and such that for every $x \in N$ and every $| u | \leq u _ { 0 }$

$$
c _ { \gamma } | u | ^ { \gamma } ~ \leq ~ \phi _ { F } ( x , q _ { F } ( x ) + u ) - Q _ { F } ( x ) ~ \leq ~ C _ { \gamma } | u | ^ { \gamma } ~ ,\tag{A.3}
$$

with $\gamma = 2$ and $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0$ , with $c _ { 2 }$ and $C _ { 2 }$ within � $o f { \textstyle \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ , with $c _ { \gamma }$ and $C _ { \gamma }$ within � $o f g / ( 2 ( 1 - \delta ) )$ , and with � arbitrarily small.

The proof is in Appendix A.7. The comparisons bound a fixed action. Under the concentration in equation (3.6), one has $x ^ { \pi } - x ^ { * } = r _ { n } Z _ { n }$ , so the post-sample regret and the premium, both expectations of these comparisons at the induced action, are of order $r _ { n } ^ { 2 }$ and $r _ { n } ^ { \gamma }$ once the contribution outside N is of smaller order and the moments of $Z _ { n }$ stay of order one. The passage from the local comparisons to these orders is the same at either exponent, so it is recorded once and cited by Theorem 3.5 at $\gamma = 1$

Lemma A.3 (Orders under local brackets). Suppose $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0 a n d \gamma \geq 1$ . Suppose that on every suficiently small neighborhood $N : = [ x ^ { * } - \eta , x ^ { * } + \eta ]$ the comparisons in equations $( \mathrm { A } . 1 ) { - } ( \mathrm { A } . 3 )$ hold, with the cutof error � in $( \mathrm { A } . 2 )$ arbitrarily small. The nonzero slope keeps a flat cutof from making the premium vanish while the post-sample regret stays quadratic. Write $A : = \{ x ^ { \pi } \in N \}$ and

$$
R _ { \mathrm { P S } } : = \mathbb { E } [ ( Q _ { F } ( x ^ { \pi } ) - Q _ { F } ( x ^ { * } ) ) \mathbf { 1 } _ { A ^ { c } } ] \ , \qquad R _ { \mathrm { S R } } : = \operatorname* { s u p } _ { | t - q ^ { * } | \leq u _ { 0 } / 2 } \mathbb { E } [ ( \phi _ { F } ( x ^ { \pi } , t ) - Q _ { F } ( x ^ { \pi } ) ) \mathbf { 1 } _ { A ^ { c } } ] \ ,
$$

where $u _ { 0 }$ is the threshold radius in (A.3). Both remainders are nonnegative. Let the policy concentrate as in equation (3.6), with $\left\{ \vert Z _ { n } \vert ^ { \operatorname* { m a x } \{ 2 , \gamma \} } \right\}$ uniformly integrable and � nondegenerate, and suppose $R _ { \mathrm { P S } } = o ( r _ { n } ^ { 2 } )$ and $R _ { \mathrm { S R } } = o ( r _ { n } ^ { \gamma } )$ . Then

$$
\begin{array} { r c l } { { \mathrm { P S } _ { \delta , n } ( \pi , F ) } } & { { = } } & { { \Theta \Bigl ( r _ { n } ^ { 2 } \Bigr ) \ , \qquad \mathrm { S R } _ { \delta , n } ( \pi , F ) } } \end{array} = \Theta \bigl ( r _ { n } ^ { \gamma } \bigr ) .\tag{A.4}
$$

The proof is in Appendix A.5. Lemma A.2 gives the comparisons at $\gamma = 2$ . What remains is the remainder hypothesis of Lemma A.3. The two remainders are supported on $\{ x ^ { \pi } \notin N \}$ , and the following bounds control them there.

The remainder $R _ { \mathrm { S R } }$ admits a bound in terms of the induced VaR outside the neighborhood. The variational objective is convex in the threshold. Its right derivative is

$$
\partial _ { t } ^ { + } \phi _ { F } ( x , t ) = 1 - \frac { \mathbb { P } \left( c \left( x , \xi \right) > t \right) } { 1 - \delta } ,\tag{A.5}
$$

and the left derivative replaces $c \left( x , \xi \right) > t \mathrm { ~ b y ~ } c \left( x , \xi \right) \geq t .$ At an atom the two difer. Both are bounded by

$$
\left| \partial _ { t } ^ { \pm } \phi _ { F } ( x , t ) \right| \ \leq \ L _ { \delta } : = \operatorname * { m a x } \left\{ 1 , \frac { \delta } { 1 - \delta } \right\} \ ,
$$

uniformly in � and �, and convexity gives the same Lipschitz bound from either one. Since $q _ { F } ( x )$ minimizes $t \mapsto \phi _ { F } ( x , t )$ , this yields

$$
R _ { \mathrm { S R } } ~ \leq ~ L _ { \delta } \left( \frac { u _ { 0 } } { 2 } \mathbb { P } \left( A ^ { c } \right) + \mathbb { E } \big [ | q _ { F } ( x ^ { \pi } ) - q ^ { * } | \mathbf { 1 } _ { A ^ { c } } \big ] \right) ~ .
$$

The Lipschitz constant $L _ { \delta }$ grows like $1 / ( 1 - \delta )$ . Equation (2.6) gives the companion bound for the post-sample

remainder. Since $\begin{array} { r } { | \mathrm { C V a R } _ { \delta } [ Y ] - \mathrm { C V a R } _ { \delta } [ Y ^ { \prime } ] | \leq ( 1 - \delta ) ^ { - 1 } \mathbb { E } | Y - Y ^ { \prime } | , } \end{array}$

$$
\left| Q _ { F } ( x ) - Q _ { F } ( y ) \right| \ \leq \ \frac { \mathbb { E } [ L ( \xi ) ] } { 1 - \delta } | x - y | ,
$$

and therefore $R _ { \mathrm { P S } } \leq \mathbb { E } [ L ( \xi ) ] ( 1 - \delta ) ^ { - 1 } \mathbb { E } \big [ | x ^ { \pi } - x ^ { * } | \mathbf { 1 } _ { A ^ { c } } \big ]$ . Equation (2.6) likewise bounds the cutof: $| q _ { F } ( x ) -$ $q ^ { * } \vert \leq C \bigl ( 1 + \vert x - x ^ { * } \vert \bigr )$ for a finite constant �, because $c ( x , \xi ) \leq | c ( x ^ { * } , \xi ) | + L ( \xi ) | x - x ^ { * } | .$

Under the concentration in equation (3.6), these bounds have to be turned into $o \big ( r _ { n } ^ { 2 } \big )$ . Write $x ^ { \ast } : = x _ { F } ^ { \ast } ,$ $q ^ { * } : = q _ { F } ( x ^ { * } )$ and $\Delta : = x ^ { \pi } - x ^ { * } = r _ { n } Z _ { n }$ , and for a radius $\eta > 0$ let $N : = \left[ x ^ { * } - \eta , x ^ { * } + \eta \right]$ and $A _ { n } : = \{ x ^ { \pi } \in N \}$ Since $r _ { n } \searrow 0$ and $Z _ { n } \Rightarrow Z$ , we have $\mathbb { P } ( A _ { n } ) \to 1$ for every fixed �.

Escapes. The bounds above and the Lipschitz assumption give, for every fixed radius $\eta > 0$

$$
R _ { \mathrm { P S } } ~ \leq ~ \frac { \mathbb { E } [ L ( \xi ) ] } { 1 - \delta } \mathbb { E } \big [ | \Delta | \mathbf { 1 } _ { A _ { n } ^ { c } } \big ]
$$

and

$$
\begin{array} { r l r } { R _ { \mathrm { S R } } } & { \leq } & { C \Big ( \mathbb { P } \big ( A _ { n } ^ { c } \big ) + \mathbb { E } \Big [ \big | \Delta \big | \mathbf { 1 } _ { A _ { n } ^ { c } } \Big ] \Big ) . } \end{array}
$$

Uniform integrability of $\left\{ \vert Z _ { n } \vert ^ { 2 } \right\}$ forces $\mathbb { E } [ Z _ { n } ^ { 2 } \mathbf { 1 } _ { | Z _ { n } | > \eta / r _ { n } } ]  0$ . Markov’s inequality gives $\mathbb { P } ( A _ { n } ^ { c } ) \le ( r _ { n } / \eta ) ^ { 2 }$ times that expectation, hence $\mathbb { P } ( A _ { n } ^ { c } ) ~ = ~ o ( r _ { n } ^ { 2 } )$ . On the same event $| Z _ { n } | \le ( r _ { n } / \eta ) Z _ { n } ^ { 2 }$ , so $\mathbb { E } [ | \Delta | \mathbf { 1 } _ { A _ { n } ^ { c } } ] \ =$ $r _ { n } \mathbb { E } [ | Z _ { n } | \mathbf { 1 } _ { A _ { n } ^ { c } } ] \leq ( r _ { n } ^ { 2 } / \eta ) \mathbb { E } [ Z _ { n } ^ { 2 } \mathbf { 1 } _ { A _ { n } ^ { c } } ] = o ( r _ { n } ^ { 2 } )$ . Therefore, for every fixed neighborhood,

$$
R _ { \mathrm { P S } } = { \cal O } \left( r _ { n } ^ { 2 } \right) , R _ { \mathrm { S R } } = { \cal O } \left( r _ { n } ^ { 2 } \right) .\tag{A.6}
$$

Equation (A.6) is then the remainder hypothesis of Lemma A.3 at $\gamma = 2 .$ , so $\mathrm { P S } _ { \delta , n } ( \pi , F ) = \Theta ( r _ { n } ^ { 2 } )$ and $\mathrm { S R } _ { \delta , n } ( \pi , F ) = \Theta ( r _ { n } ^ { 2 } ) $

## A.5 Proof of Lemma A.3

The comparisons in equations $( \mathrm { A } . 1 ) { - } ( \mathrm { A } . 3 )$ bound a fixed action. The post-sample regret is the expectation of $Q _ { F } ( x ^ { \pi } ) - Q _ { F } ( x ^ { * } )$ , so on the event $A : = \{ x ^ { \pi } \in N \}$ the first comparison bounds it by a second moment of $\Delta : = x ^ { \pi } - x ^ { * }$ , and the contribution of $A ^ { c }$ is a remainder. The premium is the price of one shared threshold against the action-specific cutofs $q _ { F } ( x ^ { \pi } )$ , so on � it is the distance from $q _ { F } ( x ^ { \pi } )$ to a single constant. For $\gamma \geq 1$ , a random variable � and an event �, let

$$
\mathcal { D } _ { \gamma } ( W ; A ) ~ { : = } ~ \operatorname* { i n f } _ { t \in \mathbb { R } } \mathbb { E } [ | W - t | ^ { \gamma } \mathbf { 1 } _ { A } ] ~ ,\tag{A.7}
$$

and write ${ \mathcal { D } } _ { \gamma } ( W )$ when � is the sure event. This is the best approximation of � by a single constant, measured in �-th absolute moment. For $\gamma = 2$ it is the variance, attained at the mean; for $\gamma = 1$ it is the

mean absolute deviation about the median. The functional is invariant under deterministic shifts,

$$
\begin{array} { r l r } { \mathcal { D } _ { \gamma } ( W + b ; A ) } & { = } & { \mathcal { D } _ { \gamma } ( W ; A ) \qquad \mathrm { ~ f o r ~ e v e r y ~ c o n s t a n t ~ } b , } \end{array}
$$

and it vanishes precisely when � is almost surely constant on �.

The expansion of the cutof carries the distance for $q _ { F } ( x ^ { \pi } )$ over to the same distance for Δ. The resulting bounds hold at a single sample size, and the orders are obtained by scaling them once the remainders are of smaller order. The next lemma records both bounds and this transfer. The exponent $\gamma \geq 1$ is free: the smooth regime takes $\gamma = 2$ , and the atomic regime takes $\gamma = 1$

Lemma A.4 (Local brackets at a fixed sample size). Suppose that on a neighborhood $N : = \left[ x ^ { * } - \eta , x ^ { * } + \eta \right]$ the comparisons (A.1)–(A.3) hold for an exponent $\gamma \geq 1$ , with $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0 ,$ , and constants $0 < c _ { 2 } \leq C _ { 2 } < \infty$ and $0 < c _ { \gamma } \leq C _ { \gamma } < \infty$ . Suppose also that the neighborhood is small enough that the induced cutofs stay inside the threshold radius on which (A.3) applies,

$$
\left( \left| q _ { F } ^ { \prime } ( x ^ { * } ) \right| + \varepsilon \right) \eta \ \leq \ \frac { u _ { 0 } } { 4 } \left( \frac { c _ { \gamma } } { 2 C _ { \gamma } } \right) ^ { 1 / \gamma } \ .\tag{A.8}
$$

Write $\Delta : = x ^ { \pi } - x ^ { * }$ and $A : = \{ x ^ { \pi } \in N \}$ . Then

$$
\begin{array} { r l r } { c _ { 2 } \mathbb { E } \big [ | \Delta | ^ { 2 } \mathbf { 1 } _ { A } \big ] } & { \le } & { \mathrm { P S } _ { \delta , n } ( \pi , F ) \le { \cal C } _ { 2 } \mathbb { E } \big [ | \Delta | ^ { 2 } \mathbf { 1 } _ { A } \big ] + { \cal R } _ { \mathrm { P S } } , } \end{array}\tag{A.9}
$$

and

$$
\begin{array} { r c l } { c _ { \gamma } \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A ) } & { \le } & { \mathrm { S R } _ { \delta , n } ( \pi , F ) \hspace * { \arraycolsep } \le \hspace C _ { \gamma } \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A ) \hspace * { \arraycolsep } + \hspace R _ { \mathrm { S R } } , } \end{array}\tag{A.10}
$$

where $R _ { \mathrm { P S } }$ and $R _ { \mathrm { S R } }$ are the remainders in the statement of Lemma A.3; both are nonnegative. Moreover, the dispersion ofthe induced VaR is itselfcontrolled by that ofΔ,

$$
\begin{array} { r l r } { \Big | \ \mathcal D _ { \gamma } ( q _ { F } ( x ^ { \pi } ) \ ; \ A ) ^ { 1 / \gamma } \ - \ \big | q _ { F } ^ { \prime } ( x ^ { * } ) \big | \cdot \mathcal D _ { \gamma } ( \Delta ; A ) ^ { 1 / \gamma } \ \Big | \ \leq & { \varepsilon } ( \mathbb { E } [ | \Delta | ^ { \gamma } \mathbf { 1 } _ { A } ] ) ^ { 1 / \gamma } \ . } \end{array}\tag{A.11}
$$

The proof is in Appendix A.6. Write $x ^ { * } : = x _ { F } ^ { * } , q ^ { * } : = q _ { F } ( x ^ { * } )$ and $\Delta : = x ^ { \pi } - x ^ { * } = r _ { n } Z _ { n }$ , and for a radius $\eta > 0$ let $N : = \left[ x ^ { * } - \eta , x ^ { * } + \eta \right]$ and $A _ { n } : = \{ x ^ { \pi } \in N \}$ . Since $r _ { n } \searrow 0$ and $Z _ { n } \Rightarrow Z$ , we have $\mathbb { P } ( A _ { n } ) \to 1$ for every fixed $\eta$

The comparisons in equations (A.1)–(A.3) are the hypothesis of Lemma A.4, so the brackets in equations (A.9), (A.10) and (A.11) are available at every sample size, and all that remains is to evaluate them at $\Delta = r _ { n } Z _ { n }$ and pass to the limit. The evaluation is stated at the free exponents of those comparisons. In the smooth regime $\gamma = 2$ , and in the atomic regime $\gamma = 1$ . For these exponents equation (A.6) supplies $R _ { \mathrm { P S } } = o ( r _ { n } ^ { 2 } )$ and $R _ { \mathrm { S R } } = o ( r _ { n } ^ { \gamma } )$ , since $o ( r _ { n } ^ { 2 } ) \mathrm { i s } o ( r _ { n } ^ { \gamma } )$ when $\gamma \leq 2$ . For $\gamma > 2$ the argument uses the hypothesis $R _ { \mathrm { S R } } = o ( r _ { n } ^ { \gamma } )$ of the lemma. The second-moment escape does not give that stronger remainder.

An elementary observation. The one estimate the fixed-� lemma does not supply is that a dispersion functional cannot degenerate along a weakly convergent sequence. Fix $\gamma \geq 1$ and let $Y _ { n } \Rightarrow Y$ with $\{ \vert Y _ { n } \vert ^ { \gamma } \}$ uniformly integrable and � nondegenerate. Then

$$
0 < \operatorname* { l i m } _ { n \to \infty } | \operatorname* { m i n f } _ { \mathcal { D } _ { \gamma } } \mathcal { D } _ { \gamma } ( Y _ { n } ) \leq \operatorname* { l i m } _ { n \to \infty } \mathcal { D } _ { \gamma } ( Y _ { n } ) < \infty .\tag{A.12}
$$

Indeed, the upper bound follows by taking $t \ = \ 0$ in equation (A.7), since uniform integrability gives $\Lambda : = \operatorname* { s u p } _ { n } \mathbb { E } [ | Y _ { n } | ^ { \gamma } ] < \infty$ . If the lower bound failed, there would exist $t _ { n }$ with $\mathbb { E } [ | Y _ { n } - t _ { n } | ^ { \gamma } ]  0$ . Finiteness of Λ makes $\left\{ t _ { n } \right\}$ bounded, so along a convergent subsequence with $t _ { n } \to t$ we would have $Y _ { n } - t _ { n } \to 0$ in probability, and hence $Y = t$ almost surely, contradicting nondegeneracy.

Restricting the functional to a high-probability event cannot destroy this, and the resulting bound is uniform in the event, which is what lets a radius be chosen below without circularity. Let $\{ E _ { n } \}$ be any events with $\mathbb { P } ( E _ { n } ) \to 1$ . Then

$$
\operatorname* { l i m i n f } _ { n \to \infty } { \mathcal { D } } _ { \gamma } ( Y _ { n } ; E _ { n } ) \ \geq \ \operatorname* { l i m i n f } _ { n \to \infty } { \mathcal { D } } _ { \gamma } ( Y _ { n } ) \ .\tag{A.13}
$$

To see this, choose � with su $\begin{array} { r } { \mathfrak { d } _ { n } \mathbb { P } ( | Y _ { n } | > M ) \le 1 / 4 } \end{array}$ , which is possible because $\left\{ Y _ { n } \right\}$ is bounded in probability, and set $S : = M + ( 4 \Lambda ) ^ { 1 / \gamma }$ . For � large enough that $\mathbb { P } ( E _ { n } ) \geq 3 / 4$ we have ${ \mathbb { P } } ( E _ { n } \cap \{ | Y _ { n } | \leq M \} ) \geq 1 / 2$ , so for $| t | > S$

$$
\begin{array} { r } { { \mathbb { E } } \big [ | Y _ { n } - t | ^ { \gamma } \mathbf { 1 } _ { E _ { n } } \big ] \ \geq \ { \frac { 1 } { 2 } } \left( S - M \right) ^ { \gamma } \ = \ 2 \Lambda \ > \ { \mathcal { D } } _ { \gamma } ( Y _ { n } ; E _ { n } ) \ , } \end{array}
$$

the last inequality by the feasible choice $t = 0$ . The infimum in equation (A.7) may therefore be taken over $| t | \leq S$ , with � free of � and of the event. For such �,

$$
\begin{array} { r } { \mathbb { E } \big [ | Y _ { n } - t | ^ { \gamma } \mathbf { 1 } _ { E _ { n } ^ { c } } \big ] \ \leq \ 2 ^ { \gamma - 1 } \left( \mathbb { E } \big [ | Y _ { n } | ^ { \gamma } \mathbf { 1 } _ { E _ { n } ^ { c } } \big ] + S ^ { \gamma } \mathbb { P } \big ( E _ { n } ^ { c } \big ) \right) \ = : \ \varepsilon _ { n } \ \longrightarrow \ 0 \ , } \end{array}
$$

uniformly in �, by uniform integrability and $\mathbb { P } ( E _ { n } ^ { c } ) \to 0$ . Hence $\mathcal { D } _ { \gamma } ( Y _ { n } ; E _ { n } ) \ge \mathcal { D } _ { \gamma } ( Y _ { n } ) - \varepsilon _ { n }$ , which gives equation (A.13).

Scaling, and the choice of radius. The functional $\mathcal { D } _ { \gamma } ( \cdot ; A )$ is positively homogeneous of degree �: replacing � by �� with $c > 0$ reindexes the infimum in equation $( \mathrm { A } . 7 )$ by $t \mapsto c t$ and rescales the objective by $c ^ { \gamma }$ . Since $\Delta = r _ { n } Z _ { n }$

$$
\mathbb { E } \big [ | \Delta | ^ { 2 } { \mathbf { 1 } } _ { A _ { n } } \big ] \ = \ r _ { n } ^ { 2 } \mathbb { E } \big [ | Z _ { n } | ^ { 2 } { \mathbf { 1 } } _ { A _ { n } } \big ] \ , \qquad \mathcal { D } _ { \gamma } ( \Delta ; A _ { n } ) \ = \ r _ { n } ^ { \gamma } \mathcal { D } _ { \gamma } ( Z _ { n } ; A _ { n } ) \ .\tag{A.14}
$$

Uniform integrability of $\{ | Z _ { n } | ^ { \operatorname* { m a x } \{ 2 , \gamma \} } \}$ , weak convergence $Z _ { n } \Rightarrow Z$ and $\mathbb { P } ( A _ { n } ) \to 1$ give

$$
{ \mathbb E } \big [ | Z _ { n } | ^ { 2 } { \mathbf 1 } _ { A _ { n } } \big ] \longrightarrow { \mathbb E } \big [ | Z | ^ { 2 } \big ] \ , \qquad { \mathbb E } \big [ | Z _ { n } | ^ { \gamma } { \mathbf 1 } _ { A _ { n } } \big ] \longrightarrow { \mathbb E } [ | Z | ^ { \gamma } ] \ ,\tag{A.15}
$$

both limits finite, and the first strictly positive because � is nondegenerate. Uniform integrability of $\{ | Z _ { n } | ^ { \operatorname* { m a x } \{ 2 , \gamma \} } \}$ implies that of $\{ | Z _ { n } | ^ { p } \}$ for every $p \leq \operatorname* { m a x } \{ 2 , \gamma \}$ , since the family is bounded in $L ^ { \operatorname* { m a x } \{ 2 , \gamma \} }$ ; in

particular, the same argument gives

$$
\mathbb { E } \bigl [ Z _ { n } \mathbf { 1 } _ { A _ { n } } \bigr ] \longrightarrow \mathbb { E } [ Z ] \ ,\tag{A.16}
$$

which is used only in the proof of Corollary 3.7. Applying equation (A.12) to $Y _ { n } = Z _ { n }$ , which converges weakly to the nondegenerate $Z$ with $\{ | Z _ { n } | ^ { \gamma } \}$ uniformly integrable, the quantity

$$
\varrho : = \left( \frac { \operatorname* { l i m } \operatorname* { i n f } _ { n } \mathcal { D } _ { \gamma } ( Z _ { n } ) } { \mathbb { E } \left[ | Z | ^ { \gamma } \right] } \right) ^ { 1 / \gamma }
$$

lies in (0, 1]; it is at most one because $t = 0$ is feasible in equation (A.7), and it does not depend on the radius $\eta .$ Fix once and for all an � with $0 < \varepsilon < | q _ { F } ^ { \prime } ( x ^ { * } ) | \varrho / 2$ . Choose � small enough that the comparisons in equations $( \mathrm { A } . 1 ) { - } ( \mathrm { A } . 3 )$ hold on N with that � and that equation (A.8) holds. The second-moment escape (A.6) then gives $R _ { \mathrm { P S } } = o ( r _ { n } ^ { 2 } )$ and $R _ { \mathrm { S R } } = o ( r _ { n } ^ { 2 } )$ on the corresponding $A _ { n }$ . The premium remainder required by the lemma is $o ( r _ { n } ^ { \gamma } )$ , which this supplies when $\gamma \leq 2$ and which is otherwise the hypothesis.

Expected post-sample regret. The bracket in equation (A.9) together with equation (A.14) gives

$$
c _ { 2 } r _ { n } ^ { 2 } \mathbb { E } \bigl [ | Z _ { n } | ^ { 2 } \mathbf { 1 } _ { A _ { n } } \bigr ] \ \leq \ \mathrm { P S } _ { \delta , n } ( \pi , F ) \ \leq \ C _ { 2 } r _ { n } ^ { 2 } \mathbb { E } \bigl [ | Z _ { n } | ^ { 2 } \mathbf { 1 } _ { A _ { n } } \bigr ] + R _ { \mathrm { P S } } \ .
$$

The moment converges to a finite positive limit by equation (A.15) and $R _ { \mathrm { P S } } = o ( r _ { n } ^ { 2 } )$ by equation (A.6), so $\mathrm { P S } _ { \delta , n } ( \pi , F ) = \Theta ( r _ { n } ^ { 2 } )$

Sample-risk premium. By equation (A.11), homogeneity and equation (A.14),

$$
\begin{array} { r l r } { \Big | \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) \ ; \ A _ { n } ) ^ { 1 / \gamma } - \big | q _ { F } ^ { \prime } ( x ^ { * } ) \big | r _ { n } \mathcal { D } _ { \gamma } ( Z _ { n } ; \ A _ { n } ) ^ { 1 / \gamma } \Big | } & { \leq } & { \varepsilon r _ { n } \left( \mathbb { E } \big [ | Z _ { n } | ^ { \gamma } \mathbf { 1 } _ { A _ { n } } \big ] \right) ^ { 1 / \gamma } \ . } \end{array}
$$

For the upper bound, $\mathcal { D } _ { \gamma } ( Z _ { n } ; A _ { n } ) \leq \mathbb { E } [ | Z _ { n } | ^ { \gamma } \mathbf { 1 } _ { A _ { n } } ]$ and equation (A.15) give $\mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A _ { n } ) ^ { 1 / \gamma } = O ( r _ { n } )$ For the lower bound, divide by $r _ { n }$ and use equation (A.13) with $Y _ { n } \ = \ Z _ { n }$ and $E _ { n } \ = \ A _ { n }$ , together with equation (A.15) and the definition of $\varrho \colon$

$$
\operatorname* { l i m i n f } _ { n \to \infty } \frac { { \mathcal D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A _ { n } ) ^ { 1 / \gamma } } { r _ { n } } \ \ge \ \big ( \big | q _ { F } ^ { \prime } ( x ^ { * } ) \big | \varrho - \varepsilon \big ) \big ( { \mathbb E } [ | Z | ^ { \gamma } ] \big ) ^ { 1 / \gamma } \ > \ 0 ,
$$

strictly positive by the choice $\varepsilon < | q _ { F } ^ { \prime } ( x ^ { * } ) | \varrho / 2$ . Hence $\mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A _ { n } ) = \Theta ( r _ { n } ^ { \gamma } )$ , and the bracket in equation (A.10),

$$
c _ { \gamma } \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) \ ; \ A _ { n } ) \ \le \ \mathrm { S R } _ { \delta , n } ( \pi , F ) \ \le \ C _ { \gamma } \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) \ ; \ A _ { n } ) + R _ { \mathrm { S R } } \ ,
$$

with $R _ { \mathrm { S R } } ~ = ~ o ( r _ { n } ^ { \gamma } )$ by the hypothesis of the lemma, which equation (A.6) verifies when $\gamma \leq 2$ , gives $\mathbf { S R } _ { \delta , n } ( \pi , F ) = \Theta ( r _ { n } ^ { \gamma } )$ . Dividing the two orders yields $\rho _ { n } = \Theta ( r _ { n } ^ { \gamma - 2 } )$ and completes equation (A.4).

## A.6 Proof of Lemma A.4

Throughout write $x ^ { * } : = x _ { F } ^ { * } , q ^ { * } : = q _ { F } ( x ^ { * } ) , \Delta : = x ^ { \pi } - x ^ { * }$ and $A : = \{ x ^ { \pi } \in N \}$ . For a fixed action � set

$$
h _ { x } ( u ) : = \phi _ { F } ( x , q _ { F } ( x ) + u ) - Q _ { F } ( x ) .
$$

Since $( c ( x , \xi ) - t ) ^ { + }$ is convex in �, the map $t \mapsto \phi _ { F } ( x , t )$ is convex, and hence so is $h _ { x }$ . By definition of $q _ { F } ( x )$ as a minimizer, $h _ { x } \geq 0 = h _ { x } ( 0 )$ . A convex function attaining its minimum at the origin is nonincreasing on $( - \infty , 0 ]$ and nondecreasing on $[ 0 , \infty )$ , so $| u | \mapsto h _ { x } ( u )$ is nondecreasing on each side of the origin. We use this monotonicity twice below. $\operatorname { I f } \mathbb { P } ( A ) = 0$ the terms supported on � vanish, and the inequalities reduce to the remainder bounds $0 \leq \mathrm { P S } _ { \delta , n } ( \pi , F ) \leq R _ { \mathrm { P S } }$ and $0 \leq \mathrm { S R } _ { \delta , n } ( \pi , F ) \leq R _ { \mathrm { S R } }$ . Assume $\mathbb { P } ( A ) > 0$

Geometry of the neighborhood. Comparing the two bounds in (A.3) at any $u \ne 0$ gives $c _ { \gamma } \leq C _ { \gamma }$ . The radius condition (A.8) keeps every induced cutof inside the interval on which those bounds apply. Its right-hand side is at most $u _ { 0 } / 4$ . Set $\begin{array} { r } { \kappa : = ( | q _ { F } ^ { \prime } ( x ^ { * } ) | + \varepsilon ) \eta } \end{array}$ . By the VaR expansion (A.2), for every $x \in N$

$$
\lvert q _ { F } ( x ) - q ^ { * } \rvert \ \leq \ \lvert q _ { F } ^ { \prime } ( x ^ { * } ) \rvert \lvert x - x ^ { * } \rvert + \varepsilon \lvert x - x ^ { * } \rvert \ \leq \ \kappa ,\tag{A.17}
$$

and the radius condition (A.8) gives $\kappa \leq u _ { 0 } / 4$ . Let $T : = [ q ^ { * } - u _ { 0 } / 2 , q ^ { * } + u _ { 0 } / 2 ]$ . Two consequences follow. First, for $t \in T$ and $x \in N$

$$
\begin{array} { r } { | t - q _ { F } ( x ) | \ \le \ | t - q ^ { * } | + | q _ { F } ( x ) - q ^ { * } | \ \le \ \frac { u _ { 0 } } { 2 } + \frac { u _ { 0 } } { 4 } \ \le \ u _ { 0 } , } \end{array}
$$

so the growth bounds (A.3) apply. Second, for $t \notin T$ and $x \in N$

$$
\begin{array} { r } { | t - q _ { F } ( x ) | \ge | t - q ^ { * } | - \kappa > \frac { u _ { 0 } } { 2 } - \frac { u _ { 0 } } { 4 } = \frac { u _ { 0 } } { 4 } . } \end{array}\tag{A.18}
$$

Expected post-sample regret. By Proposition 3.2 the integrand $Q _ { F } ( x ^ { \pi } ) - Q _ { F } ( x ^ { * } )$ is nonnegative, so splitting on �,

$$
\mathrm { P S } _ { \delta , n } ( \pi , F ) = \mathbb { E } [ \left( Q _ { F } ( x ^ { \pi } ) - Q _ { F } ( x ^ { * } ) \right) \mathbf { 1 } _ { A } ] + R _ { \mathrm { P S } } \ ,
$$

with $R _ { \mathrm { P S } } \geq 0 . \ \mathrm { A p p l y i n g \left( A . 1 \right) }$ on � to the first term and discarding $R _ { \mathrm { P S } }$ in the lower bound yields (A.9).

The premium as a threshold problem. By equation (3.4), and because $\mathbb { E } [ Q _ { F } ( x ^ { \pi } ) ]$ is finite and free of $t ,$

$$
\mathrm { S R } _ { \delta , n } ( \pi , F ) = \operatorname* { i n f } _ { t \in \mathbb { R } } \mathbb { E } [ \phi _ { F } ( x ^ { \pi } , t ) - Q _ { F } ( x ^ { \pi } ) ] = \operatorname* { i n f } _ { t \in \mathbb { R } } \left\{ J ( t ) + J ^ { c } ( t ) \right\} ,
$$

where

$$
J ( t ) : = \mathbb { E } \big [ \left( \phi _ { F } ( x ^ { \pi } , t ) - Q _ { F } ( x ^ { \pi } ) \right) \mathbf { 1 } _ { A } \big ] \ , \qquad J ^ { c } ( t ) : = \mathbb { E } \big [ \left( \phi _ { F } ( x ^ { \pi } , t ) - Q _ { F } ( x ^ { \pi } ) \right) \mathbf { 1 } _ { A ^ { c } } \big ] \ .
$$

Both are nonnegative, since $q _ { F } ( x )$ minimizes $t \mapsto \phi _ { F } ( x , t )$ pointwise. Write $G ( t ) : = \mathbb { E } [ | t - q _ { F } ( x ^ { \pi } ) | ^ { \gamma } \mathbf { 1 } _ { A } ]$ so that inf<sub>�</sub> $G ( t ) = \mathcal { D } _ { \gamma } { \big ( } q _ { F } ( x ^ { \pi } ) \ ; A { \big ) }$

Restricting the threshold to �. We first check that $\operatorname { i n f } _ { t \in \mathbb { R } } J ( t ) = \operatorname { i n f } _ { t \in T } J ( t )$ . For $t \notin T .$ , combining (A.18) with the monotonicity of $h _ { x }$ in $| u |$ and the lower bound in (A.3) evaluated at $\vert u \vert = u _ { 0 } / 4$ gives

$$
\begin{array} { r } { J ( t ) \ge c _ { \gamma } \left( \frac { u _ { 0 } } { 4 } \right) ^ { \gamma } \mathbb { P } ( A ) . } \end{array}
$$

On the other hand, taking $t = q ^ { * } \in T$ and using (A.17) with the upper bound in $( \mathrm { A } . 3 )$

$$
J ( q ^ { * } ) \ \leq \ C _ { \gamma } \mathbb { E } \big [ | q ^ { * } - q _ { F } ( x ^ { \pi } ) | ^ { \gamma } \mathbf { 1 } _ { A } \big ] \ \leq \ C _ { \gamma } \kappa ^ { \gamma } \mathbb { P } ( A ) \ \leq \ C _ { \gamma } \big ( \frac { u _ { 0 } } { 4 } \big ) ^ { \gamma } \frac { c _ { \gamma } } { 2 C _ { \gamma } } \mathbb { P } ( A ) \ = \ \frac { c _ { \gamma } } { 2 } \big ( \frac { u _ { 0 } } { 4 } \big ) ^ { \gamma } \mathbb { P } ( A ) \ ,
$$

where the third inequality is the radius condition $( \mathrm { A } . 8 )$ . Since $\mathbb { P } ( A ) > 0$ , the value at $q ^ { * }$ is strictly below the bound available of $T ,$ so no $t \notin T$ can be optimal. The same comparison applies to $G .$ , and in addition the convex map $t \mapsto G ( t )$ has a minimizer in the convex hull of the essential range of $q _ { F } ( x ^ { \pi } )$ on �, which by (A.17) lies in $[ q ^ { * } - \kappa , q ^ { * } + \kappa ] \subseteq T$ . Hence $\begin{array} { r } { \operatorname* { i n f } _ { t \in T } G ( t ) = \operatorname* { i n f } _ { t \in \mathbb { R } } G ( t ) = { \mathcal { D } } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) \ ; \ A ) } \end{array}$ , and the infimum is attained at some $\hat { t } \in T$ because � is continuous on the compact set $T .$

Premium: lower bound. Dropping the nonnegative term $J ^ { c }$ and using the restriction just established,

$$
\mathrm { S R } _ { \delta , n } ( \pi , F ) \geq \operatorname* { i n f } _ { t \in \mathbb { R } } J ( t ) = \operatorname* { i n f } _ { t \in T } J ( t ) \geq c _ { \gamma } \operatorname* { i n f } _ { t \in T } G ( t ) = c _ { \gamma } \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A ) ,
$$

where the second inequality holds because $J ( t ) \geq c _ { \gamma } G ( t )$ pointwise on � by (A.3).

Premium: upper bound. Evaluating the infimum at the feasible threshold $\hat { t } \in T$ and using $J ( \hat { t } ) \leq C _ { \gamma } G ( \hat { t } )$ on $T _ { \ast }$

$$
\begin{array} { r } { \mathrm { S R } _ { \delta , n } ( \pi , F ) \le J ( \widehat t ) + J ^ { c } ( \widehat t ) \le C _ { \gamma } \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) ; A ) + R _ { \mathrm { S R } } , } \end{array}
$$

since $\hat { t } \in T$ and $R _ { \mathrm { S R } }$ is by definition the supremum of $J ^ { c }$ over $T .$ . This establishes (A.10).

Dispersion of the induced VaR. Write $\| Y \| : = ( \mathbb { E } [ | Y | ^ { \gamma } \mathbf { 1 } _ { A } ] ) ^ { 1 / \gamma }$ , which is a seminorm because $\gamma \geq 1$ . On � decompose $q _ { F } ( x ^ { \pi } ) - q ^ { * } = q _ { F } ^ { \prime } ( x ^ { * } ) \Delta + e$ where, by $( \mathrm { A } . 2 ) , | e | \le \varepsilon | \Delta |$ and hence $\| e \| \leq \varepsilon \| \Delta \|$ . Substituting $t = q ^ { * } + s$ and using shift invariance,

$$
\begin{array} { r l r } { \mathcal { D } _ { \gamma } ( q _ { F } ( x ^ { \pi } ) \ ; \ A ) ^ { 1 / \gamma } = \underset { s } { \operatorname* { i n f } } \left. q _ { F } ^ { \prime } ( x ^ { * } ) \Delta + e - s \right. \ , } & { } & { \left| q _ { F } ^ { \prime } ( x ^ { * } ) \right| \mathcal { D } _ { \gamma } ( \Delta ; \ A ) ^ { 1 / \gamma } = \underset { s } { \operatorname* { i n f } } \left. q _ { F } ^ { \prime } ( x ^ { * } ) \Delta - s \right. \ . } \end{array}
$$

For each � the triangle inequality gives $\big | \| q _ { F } ^ { \prime } ( x ^ { * } ) \Delta + e - s \| - \| q _ { F } ^ { \prime } ( x ^ { * } ) \Delta - s \| \big | \leq \| e \|$ , and two infima whose integrands difer by at most $\| e \|$ uniformly difer by at most $\| e \|$ . This is (A.11).

## A.7 Proof of Lemma A.2

Fix $\theta > 0$ . Since $x ^ { * }$ is an interior minimizer of $Q _ { F }$ and $Q _ { F }$ is twice diferentiable at $x ^ { * } , Q _ { F } ^ { \prime } ( x ^ { * } ) = 0$ , and Taylor’s theorem with Peano remainder gives

$$
\begin{array} { r } { Q _ { F } ( x ) - Q _ { F } ( x ^ { * } ) = \frac { 1 } { 2 } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) ( x - x ^ { * } ) ^ { 2 } + o \Big ( ( x - x ^ { * } ) ^ { 2 } \Big ) ~ , } \end{array}
$$

so $( \mathrm { A } . 1 )$ holds with constants within $\theta$ of $\textstyle { \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ once $\eta$ is small enough; positivity of $Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ makes the lower constant positive for $\theta$ small. The expansion (A.2) with $q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0$ and $\varepsilon$ arbitrarily small is the definition of that derivative, and the expansion persists on every smaller neighborhood. Diferentiability of $q _ { F }$ at $x ^ { * }$ supplies continuity of $q _ { F }$ at $x ^ { * }$ . For the threshold direction, diferentiating the variational objective twice in � gives $\partial _ { t } ^ { 2 } \phi _ { F } ( x , t ) = g _ { x } ( t ) / ( 1 - \delta )$ , while $\partial _ { t } \phi _ { F } ( x , q _ { F } ( x ) ) = 0$ by optimality. Joint continuity of $( x , t ) \mapsto g _ { x } ( t )$ at $( x ^ { * } , q ^ { * } )$ , positivity and finiteness of $g _ { \cdot }$ both assumed in Theorem 3.4, and continuity of $q _ { F }$ at $x ^ { * }$ together imply that $g _ { x } ( \tau )$ lies within $\theta ^ { \prime }$ of $g$ for all $x \in N$ and all � between $q _ { F } ( x )$ and $q _ { F } ( x ) + u$ with $| u | \leq u _ { 0 }$ , provided � and $u _ { 0 }$ are small enough. Taylor’s theorem in � then yields

$$
h _ { x } ( u ) = \textstyle { \frac { 1 } { 2 } } { \frac { g _ { x } ( \tau ) } { 1 - \delta } } u ^ { 2 } ,
$$

so (A.3) holds with $\gamma = 2$ and constants within � of $g / ( 2 ( 1 - \delta ) )$ . Choose � small enough that $( \mathrm { A } . 8 )$ holds; shrinking the neighborhood does not disturb these bounds. One consequence of equation (A.3) is that the minimizing set of the variational objective from equation (2.3) is a singleton on the neighborhood: if it contained an interval, the left-hand side of equation (A.3) would be strictly positive at an interior � while the right-hand side would vanish. Hence that set equals $\{ q _ { F } ( x ) \}$ for every $x \in N$

## A.8 Proof of Proposition 3.6

Under Theorem $3 . 4 , \gamma = 2 .$ , so the dispersion is a restricted variance. Minimizing $t \mapsto \mathbb { E } [ ( W - t ) ^ { 2 } \mathbf { 1 } _ { B } ]$ at $t = \mathbb { E } [ W \mathbf { 1 } _ { B } ] / \mathbb { P } ( B )$ gives the identity

$$
\mathcal { D } _ { 2 } ( W ; B ) = \mathbb { E } \big [ W ^ { 2 } \mathbf { 1 } _ { B } \big ] - \frac { ( \mathbb { E } [ W \mathbf { 1 } _ { B } ] ) ^ { 2 } } { \mathbb { P } ( B ) } ,
$$

so (A.15) at $\gamma = 2$ and the first-moment limit (A.16), together with $\mathbb { P } ( A _ { n } ) \to 1$ , upgrade the liminf bound of (A.13) to convergence: ${ \mathcal { D } } _ { 2 } ( Z _ { n } ; A _ { n } ) \to { \mathbb { E } } [ Z ^ { 2 } ] - { \mathbb { E } } [ Z ] ^ { 2 } = \operatorname { V a r } ( Z )$ and $\mathbb { E } [ Z _ { n } ^ { 2 } \mathbf { 1 } _ { A _ { n } } ]  \mathbb { E } [ Z ^ { 2 } ]$ , the latter strictly positive. The same uniform integrability gives the unrestricted moments $\mathbb { E } [ Z _ { n } ^ { 2 } ] \  \ \mathbb { E } [ Z ^ { 2 } ]$ and $\mathbb { i } [ Z _ { n } ] \to \mathbb { E } [ Z ]$ , hence $\operatorname { V a r } ( Z _ { n } ) \to \operatorname { V a r } ( Z )$

Fix $\theta > 0$ and let � be the radius supplied by the smooth case in Appendix A.7 for that �, so that $c _ { 2 }$ and $C _ { 2 }$ lie within � of $\textstyle { \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) , c _ { \gamma }$ and $C _ { \gamma }$ within � of $g / ( 2 ( 1 - \delta ) )$ , and $\varepsilon \leq \theta$ . The bracket (A.11) at $\gamma = 2$ gives

$$
\begin{array} { r } { \biggr \vert \mathcal { D } _ { 2 } ( q _ { F } ( x ^ { \pi } ) \ ; A _ { n } ) ^ { 1 / 2 } - \bigl \vert { q } _ { F } ^ { \prime } ( x ^ { * } ) \bigr \vert r _ { n } \mathcal { D } _ { 2 } ( Z _ { n } ; A _ { n } ) ^ { 1 / 2 } \ \biggr \vert \ \leq \ \theta r _ { n } \left( \mathbb { E } \left[ Z _ { n } ^ { 2 } \mathbf { 1 } _ { A _ { n } } \right] \right) ^ { 1 / 2 } \ , } \end{array}
$$

so that, letting $n \to \infty$ at fixed �, both the limit inferior and the limit superior of $\mathcal { D } _ { 2 } ( q _ { F } ( x ^ { \pi } ) ; A _ { n } ) ^ { 1 / 2 } / r _ { n }$ lie within $\theta \mathbb { E } [ Z ^ { 2 } ] ^ { 1 / 2 } \operatorname { o f } \left| q _ { F } ^ { \prime } ( x ^ { * } ) \right| \operatorname { V a r } ( Z ) ^ { 1 / 2 }$ . No claim is made that the limit exists at fixed $\theta ;$ the bounds below are applied to the two extreme limits and collapse only as $\theta \downarrow 0$

Read separately, before the ratio is formed, the two brackets ofLemma A.4 are the limits in Proposition 3.6. The post-sample bracket, with $R _ { \mathrm { P S } } = o ( r _ { n } ^ { 2 } )$ , pins $\mathrm { P S } _ { \delta , n } ( \pi , F ) / r _ { n } ^ { 2 }$ within � of $\textstyle { \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) \mathbb { E } [ Z ^ { 2 } ]$ once � is large. The premium bracket, fed with the display above, likewise pins $\operatorname { S R } _ { \delta , n } ( \pi , F ) / r _ { n } ^ { 2 }$ within a multiple of � of

$$
\frac { g q _ { F } ^ { \prime } ( x ^ { * } ) ^ { 2 } } { 2 \left( 1 - \delta \right) } \cdot \mathrm { V a r } ( Z ) .
$$

Letting $n \to \infty$ and then $\theta \downarrow 0$ gives

$$
\begin{array} { r } { \frac { \mathrm { P S } _ { \delta , n } ( \pi , F ) } { r _ { n } ^ { 2 } } \longrightarrow \frac { Q _ { F } ^ { \prime \prime } ( x ^ { * } ) } { 2 } \cdot \mathbb { B } \big [ Z ^ { 2 } \big ] , \qquad \frac { \mathrm { S R } _ { \delta , n } ( \pi , F ) } { r _ { n } ^ { 2 } } \longrightarrow \frac { g q _ { F } ^ { \prime } ( x ^ { * } ) ^ { 2 } } { 2 ( 1 - \delta ) } \cdot \mathrm { V a r } ( Z ) . } \end{array}
$$

Since $\mathbf { M S E } = r _ { n } ^ { 2 } \mathbb { E } [ Z _ { n } ^ { 2 } ]$ and $V = r _ { n } ^ { 2 } \operatorname { V a r } ( Z _ { n } )$ , dividing through by these moments is (3.8).

## A.9 Proof of Corollary 3.7

Dividing the two limits in the proof of Proposition 3.6 yields $\rho _ { n }  K \cdot \mathrm { V a r } ( Z ) / \mathbb { E } [ Z ^ { 2 } ]$ , by the definition of � in (3.9). When $\mathbb { E } [ Z ] = 0$ the ratio of the variance to the second moment is one, which is (3.13).

## A.10 The fixed-threshold curvature in equation (3.11)

Let $Q ^ { \prime \prime } ~ = ~ Q _ { F } ^ { \prime \prime } ( x ^ { * } ) , ~ a _ { F } ~ = ~ q _ { F } ^ { \prime } ( x ^ { * } )$ , and $b ~ = ~ g / ( 1 - \delta )$ The smooth hypotheses give $Q _ { F } ( x ^ { * } + h ) \ =$ $\begin{array} { r } { Q _ { F } ( x ^ { * } ) + \frac { 1 } { 2 } Q ^ { \prime \prime } h ^ { 2 } + o ( h ^ { 2 } ) } \end{array}$ and $q _ { F } ( x ^ { * } + h ) = q ^ { * } + a _ { F } h + o ( | h | )$ . Integrating the threshold derivative, using joint continuity of the loss density at $( x ^ { * } , q ^ { * } )$ , gives

$$
\begin{array} { r } { \phi _ { F } ( x , q _ { F } ( x ) + \nu ) - Q _ { F } ( x ) = \frac { 1 } { 2 } b \nu ^ { 2 } + o ( \nu ^ { 2 } ) } \end{array}
$$

as $( x , \nu ) \to ( x ^ { * } , 0 )$ , with a remainder uniform over nearby �. Consequently,

$$
\begin{array} { r } { \phi _ { F } ( x ^ { * } + h , q ^ { * } + u ) = Q _ { F } ( x ^ { * } ) + \frac { 1 } { 2 } Q ^ { \prime \prime } h ^ { 2 } + \frac { 1 } { 2 } b ( u - a _ { F } h ) ^ { 2 } + o ( h ^ { 2 } + u ^ { 2 } ) . } \end{array}
$$

Taking $u = 0$ gives ${ A _ { F } } = { Q ^ { \prime \prime } } + b a _ { F } ^ { 2 }$ and $1 + K = { A _ { F } } / { Q ^ { \prime \prime } }$ . The regret expansion (3.10) can therefore be written

$$
\begin{array} { r } { \mathcal { R } _ { \delta , n } ( \pi , F ) = \frac { 1 } { 2 } \left\{ Q ^ { \prime \prime } \big ( \mathbb { E } [ x ^ { \pi } - x ^ { * } ] \big ) ^ { 2 } + A _ { F } \operatorname { V a r } ( x ^ { \pi } ) \right\} + o ( r _ { n } ^ { 2 } ) . } \end{array}
$$

This argument uses the local expansion, rather than existence of classical second partial derivatives of $\phi _ { F }$ throughout a neighborhood.

## A.11 The zero-slope cutof endpoint

Proposition A.5 (Zero-slope cutof). Suppose that the hypotheses of Theorem 3.4 hold, except that $q _ { F } ^ { \prime } ( x ^ { * } ) =$ 0. Then

$$
\begin{array} { r c l } { { \mathrm { S R } _ { \delta , n } ( \pi , F ) } } & { { = } } & { { o \left( r _ { n } ^ { 2 } \right) ~ , } } \end{array}
$$

while

$$
\begin{array} { r c l } { { \mathrm { P S } _ { \delta , n } ( \pi , F ) } } & { { \sim } } & { { \frac { 1 } { 2 } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) ~ r _ { n } ^ { 2 } \mathbb { E } [ Z ^ { 2 } ] ~ = ~ \Theta \Bigl ( r _ { n } ^ { 2 } \Bigr ) ~ . } } \end{array}
$$

Proof. For any $\varepsilon > 0$ , diferentiability gives $| q _ { F } ( x ) - q ^ { * } | \leq \varepsilon | x - x ^ { * } |$ on a suficiently small neighborhood. Evaluate the common threshold in equation (3.4) at $q ^ { * }$ . The local threshold curvature is bounded, so its contribution is at most $C \varepsilon ^ { 2 } \mathbb { E } [ ( x ^ { \pi } - x ^ { * } ) ^ { 2 } ]$ , with � bounded as the neighborhood shrinks. The escape contribution is $o \big ( r _ { n } ^ { 2 } \big )$ by the same envelope and uniform-integrability argument as in equation (A.6). Hence

$$
0 \leq \operatorname* { l i m } _ { n } \operatorname* { s u p } \frac { \mathrm { S R } _ { \delta , n } ( \pi , F ) } { r _ { n } ^ { 2 } } \leq C \varepsilon ^ { 2 } \mathbb { E } [ Z ^ { 2 } ] .
$$

Letting $\varepsilon \downarrow 0$ gives $\mathrm { S R } _ { \delta , n } ( \pi , F ) = o ( r _ { n } ^ { 2 } ) $ . The post-sample expansion uses only the curvature of $Q _ { F }$ and the moment conditions and still yields $\mathrm { P S } _ { \delta , n } ( \pi , F ) \sim Q _ { F } ^ { \prime \prime } ( x ^ { * } ) r _ { n } ^ { 2 } \mathbb { E } [ Z ^ { 2 } ] / 2$ □

## A.12 The relative variance penalty for squared loss

Table 1 evaluates the relative variance penalty (3.9) for the squared loss $c ( x , \xi ) ~ = ~ ( \xi - x ) ^ { 2 } / 2$ The second derivative of this loss in the deviation $\xi - x$ is the constant 1. In the tail expectation $T ( x ) \ =$ $\mathbb { E } \big [ ( \xi - x ) \mathbf { 1 } _ { \{ | \xi - x | > \sqrt { 2 q _ { F } ( x ) } \} } \big ]$ , the factor $\xi - x$ has action derivative −1, so that contribution is dominated by the constant 1. A finite second moment makes the factor itself integrable. Diferentiating � also moves the tail boundaries; the proof handles that term by Leibniz’s rule.

Proposition A.6 (The relative variance penalty for squared loss). Fix $\delta \in ( 0 , 1 )$ and $c ( x , \xi ) = ( \xi - x ) ^ { 2 } / 2$ and assume $\mathbb { E } [ \xi ^ { 2 } ] < \infty$ . Let �<sup>∗</sup> be an interior oracle action, write $q ^ { * } : = q _ { F } ( x ^ { * } )$ and $d : = \sqrt { 2 q ^ { * } }$ , and let $f _ { + } : = f ( x ^ { * } + d ) , f _ { - } : = f ( x ^ { * } - d )$ . Suppose � admits a density � continuous in a neighborhood of each of $x ^ { * } + d$ and $x ^ { * } - d ,$ both interior to the support of �, with $f _ { + } + f _ { - } > 0$ . Then

$$
K = { \frac { d ( f _ { - } - f _ { + } ) ^ { 2 } } { ( 1 - \delta ) ( f _ { + } + f _ { - } ) + 4 d f _ { + } f _ { - } } } .\tag{A.19}
$$

The two curvatures are

$$
\begin{array} { c } { { Q _ { F } ^ { \prime \prime } ( x ^ { * } ) = 1 + { \displaystyle \frac { 4 d f _ { + } f _ { - } } { ( 1 - \delta ) ( f _ { + } + f _ { - } ) } } \geq 1 , } } \\ { { A _ { F } = 1 + { \displaystyle \frac { d ( f _ { + } + f _ { - } ) } { 1 - \delta } } < \infty . } } \end{array}\tag{A.20}
$$

In particular $K = 0$ if and only $i f f _ { + } = f _ { - }$

## A.13 Proof of Proposition A.6

The loss equals � at the deviations $\pm \sqrt { 2 t }$ , so

$$
\mathbb { P } \big ( c ( x , \xi ) \le t \big ) = F \big ( x + \sqrt { 2 t } \big ) - F \big ( x - \sqrt { 2 t } \big ) .
$$

The left-hand side is 0 at $t = 0$ and the interior-support hypothesis places the �-crossing in the open support. Since $f _ { + } + f _ { - } > 0$ , the right-hand side is strictly increasing there, and $q : = q _ { F } ( x )$ is its unique nonnegative root. Diferentiating in � at $t = q$ , with $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \sqrt { 2 t } = 1 / \sqrt { 2 t } . } \end{array}$ , gives the density of the loss at the cutof,

$$
g \ = \ { \frac { f _ { + } + f _ { - } } { d } } \ .
$$

Set $G ( x , q ) : = F ( x + { \sqrt { 2 q } } ) - F ( x - { \sqrt { 2 q } } ) - \delta$ . Continuity of $f$ near the two critical levels makes � continuously diferentiable near $( x ^ { * } , q ^ { * } )$ , with $\partial _ { x } G = f _ { + } - f _ { - }$ and $\partial _ { q } G = g > 0$ . The implicit function theorem yields

$$
q _ { F } ^ { \prime } ( x ^ { * } ) = { \frac { f _ { - } - f _ { + } } { g } }\tag{A.21}
$$

and makes $q _ { F }$ continuously diferentiable near $x ^ { * }$

Write $\phi ( x , t ) = t + ( 1 - \delta ) ^ { - 1 } \mathbb { E } \big [ \big ( ( \xi - x ) ^ { 2 } / 2 - t \big ) _ { + } \big ]$ . The integrand vanishes at both critical levels, so

$$
\partial _ { x } \phi ( x , t ) \ = \ - \frac { 1 } { 1 - \delta } \mathbb { E } \big [ ( \xi - x ) \mathbf { 1 } _ { \{ c ( x , \xi ) > t \} } \big ] \ .
$$

The envelope theorem at the minimizing threshold $t = q ( x )$ gives $Q ^ { \prime } ( x ) = - ( 1 - \delta ) ^ { - 1 } T ( x )$ , where

$$
T ( x ) \ : = \ \mathbb { E } { \big [ } ( \xi - x ) \mathbf { 1 } _ { \{ | \xi - x | > { \sqrt { 2 q ( x ) } } \} } { \big ] } \ .
$$

The interior oracle condition is $T ( x ^ { * } ) = 0$ . On that tail the derivative of $\xi - x$ in the action is −1. For every action near $x ^ { * }$ this derivative is bounded by the constant 1, which does not depend on the action; $\mathbb { E } [ \xi ^ { 2 } ] < \infty$ makes the integrand integrable. The boundaries $x \pm { \sqrt { 2 q ( x ) } }$ are continuously diferentiable because $q _ { F }$ is, and $f$ is continuous there, so Leibniz’s rule applies:

$$
\begin{array} { r l } & { T ^ { \prime } ( x ^ { * } ) \ = \ - \left( 1 + \frac { q ^ { \prime } } { d } \right) d f _ { + } - \left( 1 - \frac { q ^ { \prime } } { d } \right) d f _ { - } - \mathbb { E } \big [ \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) > q ^ { * } \} } \big ] } \\ & { = \ - d \left( f _ { + } + f _ { - } \right) - ( 1 - \delta ) + q ^ { \prime } ( f _ { - } - f _ { + } ) ~ . } \end{array}
$$

The indicator has probability $1 - \delta$ because the loss has a density at its Value-at-Risk. Since $q ^ { \prime } = ( f _ { - } - f _ { + } ) / g$ and $g = ( f _ { + } + f _ { - } ) / d$ , the expression for $T ^ { \prime }$ gives the first identity in (A.20). It establishes positive finite curvature from the remaining hypotheses. Adding $g q ^ { \prime 2 } / ( 1 - \delta )$ yields the fixed-threshold curvature $A _ { F }$ , and substituting in (3.9) gives (A.19). A positive afine transformation of the loss multiplies both curvatures by the same positive factor. For squared loss, a location–scale change of the outcome replaces � by a positive multiple and $f _ { \pm }$ by the reciprocal multiple, leaving � unchanged.

## A.14 Proof of Theorem 3.5

Write $p : = \mathbb { P } ( c ( x ^ { * } , \xi ) = q ^ { * } )$ and

$$
s _ { + } : = \frac { \mathbb { P } ( c ( x ^ { * } , \xi ) \leq q ^ { * } ) - \delta } { 1 - \delta } , \quad s _ { - } : = \frac { \delta - \mathbb { P } ( c ( x ^ { * } , \xi ) < q ^ { * } ) } { 1 - \delta } ,\tag{A.22}
$$

so that $s _ { + } + s _ { - } = p / ( 1 - \delta )$ . Both are positive under the straddling hypothesis of Theorem 3.5.

Lemma A.7 (Atom exponents). Suppose the hypotheses of Theorem 3.5. For every $\theta > 0$ there is a neighborhood of �<sup>∗</sup> on which (A.1)–(A.3) hold with $\gamma = 1$ and $q _ { F } ^ { \prime } ( x ^ { * } ) = \ell ^ { \prime } ( x ^ { * } ) \neq 0 ,$ , with $\varepsilon \leq \theta ,$ , and with $c _ { \gamma }$ and $C _ { \gamma }$ within � ofmin $\left\{ s _ { + } , s _ { - } \right\}$ and max $\left\{ s _ { + } , s _ { - } \right\}$

Lemma A.7 gives $( \mathrm { A } . 1 ) { - } ( \mathrm { A } . 3 )$ with $\gamma = 1 , q _ { F } ^ { \prime } ( x ^ { * } ) = \ell ^ { \prime } ( x ^ { * } ) \neq 0$ and, for every $\theta > 0$ , with $\varepsilon \leq \theta$ Lemma A.3 at this exponent gives $\mathrm { P S } _ { \delta , n } ( \pi , F ) = \Theta ( r _ { n } ^ { 2 } )$ and $\operatorname { S R } _ { \delta , n } ( \pi , F ) = \Theta ( r _ { n } )$

## A.15 Proof of Lemma A.7

Write $S _ { x } ( t ) : = \mathbb { P } ( c ( x , \xi ) > t )$ for the survival function of the loss at the action �. Throughout, $\theta > 0$ is fixed and � and $u _ { 0 }$ are shrunk finitely many times.

The threshold direction. Diferentiating the variational objective in the threshold, as in equation $( \mathsf { A } . 5 )$ gives the right derivative $\partial _ { t } ^ { + } \phi _ { F } ( x , t ) = 1 - S _ { x } ( t ) / ( 1 - \delta )$ . Where $S _ { x }$ is continuous this is the ordinary derivative, and $t \mapsto \phi _ { F } ( x , t )$ is convex and Lipschitz, so for every $w > 0$

$$
\begin{array} { l } { { \phi _ { F } ( x , \ell ( x ) + w ) - \phi _ { F } ( x , \ell ( x ) ) = \displaystyle \int _ { 0 } ^ { w } \left( 1 - \frac { S _ { x } \left( \ell ( x ) + \nu \right) } { 1 - \delta } \right) d \nu , } } \\ { { \phi _ { F } ( x , \ell ( x ) - w ) - \phi _ { F } ( x , \ell ( x ) ) = \displaystyle \int _ { 0 } ^ { w } \left( \frac { S _ { x } \left( \ell ( x ) - \nu \right) } { 1 - \delta } - 1 \right) d \nu . } } \end{array}
$$

The integrands are monotone in �, since $S _ { x }$ is nonincreasing. Bounding each by its values at $\nu \to 0 ^ { + }$ and at $\nu = w \leq u _ { 0 }$ gives, for every $x \in N$ and every $0 < w \leq u _ { 0 }$

$$
\left( 1 - \frac { S _ { x } \left( \ell ( x ) \right) } { 1 - \delta } \right) w \ \leq \ \phi _ { F } ( x , \ell ( x ) + w ) - \phi _ { F } ( x , \ell ( x ) ) \ \leq \ \left( 1 - \frac { S _ { x } \left( \ell ( x ) + u _ { 0 } \right) } { 1 - \delta } \right) w \ ,
$$

and

$$
\left( \frac { \mathbb { P } \left( c \left( x , \xi \right) \geq \ell \left( x \right) \right) } { 1 - \delta } - 1 \right) w \leq \phi _ { F } ( x , \ell \left( x \right) - w ) - \phi _ { F } ( x , \ell \left( x \right) ) \leq \left( \frac { \mathbb { P } \left( c \left( x , \xi \right) \geq \ell \left( x \right) - u _ { 0 } \right) } { 1 - \delta } - 1 \right) w \mathrm { ~ . }
$$

The continuity of the rest of the loss distribution at the cutof, assumed in Theorem 3.5, is the pair of limits

$$
\begin{array} { r } { \underset { ( x , u )  ( x ^ { * } , 0 ^ { + } ) } { \operatorname* { l i m } } \mathbb { P } ( c ( x , \xi ) > \ell ( x ) + u ) = \mathbb { P } ( c ( x ^ { * } , \xi ) > q ^ { * } ) , } \\ { \underset { ( x , u )  ( x ^ { * } , 0 ^ { + } ) } { \operatorname* { l i m } } \mathbb { P } ( c ( x , \xi ) \geq \ell ( x ) - u ) = \mathbb { P } ( c ( x ^ { * } , \xi ) \geq q ^ { * } ) . } \end{array}\tag{A.23}
$$

The four coeficients converge, as � and $u _ { 0 }$ shrink, to $s _ { + } , s _ { + } , s _ { - }$ <sub>−</sub> and �<sub>−</sub> respectively: this is exactly (3.7) together with the definitions in (A.22). Choose � and $u _ { 0 }$ small enough that all four lie within � of their limits. Since $s _ { + } > 0$ and $s _ { - } > 0$ by the straddling hypothesis, both one-sided increments are strictly positive for � small, so ℓ(�) is the unique minimizer of $t \mapsto \phi _ { F } ( x , t )$ , that is,

$$
q _ { F } ( x ) = \ell ( x ) \qquad { \mathrm { f o r ~ e v e r y ~ } } x \in N .
$$

The two displays are therefore exactly the growth condition $( \mathsf { A } . 3 )$ with $\gamma = 1 , c _ { \gamma } \geq \operatorname* { m i n } \{ s _ { + } , s _ { - } \} - \theta$ and $C _ { \gamma } \leq \operatorname* { m a x } \{ s _ { + } , s _ { - } \} + \theta$ . The slope identity in (A.22) is immediate:

$$
s _ { + } + s _ { - } = \frac { \mathbb { P } \left( c \left( x ^ { * } , \xi \right) \leq q ^ { * } \right) - \mathbb { P } \left( c \left( x ^ { * } , \xi \right) < q ^ { * } \right) } { 1 - \delta } = \frac { \mathbb { P } \left( c \left( x ^ { * } , \xi \right) = q ^ { * } \right) } { 1 - \delta } = \frac { p } { 1 - \delta } ,
$$

and $s _ { + } = s .$ <sub>−</sub> precisely when the two gaps are equal, in which case each equals $p / ( 2 ( 1 - \delta ) )$ .

The action direction. Since $q _ { F } ( x ) = \ell ( x )$ on N and ℓ is continuously diferentiable near $x ^ { * }$ , once � is small enough one has $\left| \ell ^ { \prime } ( x ) - q _ { F } ^ { \prime } ( x ^ { * } ) \right| \leq \varepsilon$ throughout N, with $q _ { F } ^ { \prime } ( x ^ { * } ) = \ell ^ { \prime } ( x ^ { * } ) \neq 0 , \varepsilon \leq \theta$ and $\varepsilon < \left| q _ { F } ^ { \prime } ( x ^ { * } ) \right| / 2$ The mean value theorem gives the expansion (A.2) and the pairwise bound

$$
{ \begin{array} { r l r l } { { \big | } \ell ( x ) - \ell ( y ) - q _ { F } ^ { \prime } ( x ^ { * } ) ( x - y ) { \big | } } & { \leq } & { \varepsilon \ | x - y | \qquad { \mathrm { ~ f o r ~ a l l ~ } } x , y \in N ~ . } \end{array} }
$$

For the growth of the outcome-only CVaR objective, $x ^ { * }$ is an interior minimizer of $Q _ { F }$ and $Q _ { F }$ is twice diferentiable there, so $Q _ { F } ^ { \prime } ( x ^ { * } ) = 0$ and Taylor’s theorem with Peano remainder gives (A.1) with constants within � of $\textstyle { \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ , positive for � small because $Q _ { F } ^ { \prime \prime } ( x ^ { * } ) > 0$ . This part of the argument is identical to the corresponding step in Appendix $\mathrm { A } . 7 ;$ only the threshold direction difers. Choose $\eta$ small enough that (A.8) holds. Its right-hand side is positive because $c _ { \gamma } > 0$

## A.16 Proof of Proposition 3.8

Write $x ^ { * } : = x _ { F } ^ { * } , q ^ { * } : = q _ { F } ( x ^ { * } )$ and $\Delta : = x ^ { \pi } - x ^ { * } = r _ { n } Z _ { n }$ , and for a radius $\eta > 0$ let $N : = \left[ x ^ { * } - \eta , x ^ { * } + \eta \right]$ and $A _ { n } : = \{ x ^ { \pi } \in N \}$ . Since $r _ { n } \searrow 0$ and $Z _ { n } \Rightarrow Z$ , we have $\mathbb { P } ( A _ { n } ) \to$ 1 for every fixed $\eta$

The post-sample limit is the bracket already used for Proposition 3.6. It depends only on the curvature of $\boldsymbol { Q } _ { F } \mathrm { a t } \boldsymbol { x } ^ { * }$ , which the atom hypotheses supply, and with $\mathbf { M S E } = r _ { n } ^ { 2 } \mathbb { E } [ Z _ { n } ^ { 2 } ]$ it is the first limit in (3.16).

For the premium, fix $\theta \in \big ( 0 , \operatorname* { m i n } \{ s _ { + } , s _ { - } \} \big )$ and take the neighborhood of Lemma $\mathrm { A } . 7$ on which $q _ { F } ( x ) =$ ℓ(�) and the one-sided increments of $t \mapsto \phi _ { F } ( x , t )$ lie within � of $s _ { + }$ and $s _ { - }$ . On that neighborhood the excess

$\phi _ { F } ( x , \ell ( x ) + u ) - Q _ { F } ( x )$ therefore lies within $\theta | u | \mathrm { o f } s _ { + } u ^ { + } + s _ { - } ( - u ) ^ { + }$ . The weighted-optimal threshold need not be a median, so the error $\theta | u |$ does not pass through the minimum as � times the unweighted absolute deviation. Write $s _ { \mathrm { m i n } } : = \operatorname* { m i n } \{ s _ { + } , s _ { - } \} > 0$ and

$$
\begin{array} { r l r } { W _ { s } ( t ) } & { : = } & { \mathbb { E } \left[ \left( s _ { + } \big ( t - \ell ( x ^ { \pi } ) \big ) ^ { + } + s _ { - } \big ( \ell ( x ^ { \pi } ) - t \big ) ^ { + } \right) { \bf 1 } _ { A _ { n } } \right] \ . } \end{array}
$$

Let $J ( t )$ be the expected excess of the variational objective on $A _ { n } .$ . The pointwise bound gives $| J ( t ) -$ $W _ { s } ( t ) | \leq \theta \mathbb { E } [ | t - \ell ( x ^ { \pi } ) | \mathbf { 1 } _ { A _ { n } } ]$ . The weighted deviation is at least $s _ { \mathrm { m i n } }$ times that absolute deviation, so $| J ( t ) - W _ { s } ( t ) | \leq ( \theta / s _ { \mathrm { m i n } } ) W _ { s } ( t )$ for every threshold in the interval $T = [ q ^ { * } - u _ { 0 } / 2 , q ^ { * } + u _ { 0 } / 2 ]$ of the proof of Lemma A.4. Shrink the action neighborhood so that $\left( \mathsf { A } . 8 \right)$ holds. That proof shows that both � and $W _ { s }$ are minimized inside �: of � the excess exceeds the value at $q ^ { * }$ . The relative bound is applied for $t \in T$ and the contribution of $A _ { n } ^ { c }$ is added afterwards. The infima of � and of $W _ { s }$ over � therefore stand in the ratio $1 \pm \theta / s _ { \mathrm { { m i n } } }$ . Uniform integrability of $\left\{ \vert Z _ { n } \vert ^ { 2 } \right\}$ gives that of $\left\{ \left. Z _ { n } \right. \right\}$ , and the contribution of $A _ { n } ^ { c }$ is $o ( r _ { n } )$ by (A.6). On $A _ { n }$ one has $q _ { F } ( x ^ { \pi } ) = \ell ( x ^ { \pi } )$ , so the expansion of ℓ gives $q _ { F } ( x ^ { \pi } ) = q ^ { * } + r _ { n } \ell ^ { \prime } ( x ^ { * } ) Z _ { n } + r _ { n } e _ { n }$ with $\mathbb { E } [ | e _ { n } | ]  0$ after $\theta \downarrow 0$ . Since $q _ { F } = \ell$ on the neighborhood, $q _ { F } ^ { \prime } ( x ^ { * } ) = \ell ^ { \prime } ( x ^ { * } )$ . Weak convergence of $Z _ { n }$ and uniform integrability of $\lvert Z _ { n } \rvert$ give convergence of their distributions in Wasserstein-1. The first-mean cutof expansion transfers this convergence to the scaled cutofs. For any coupling, the weighted deviation changes by at most � max $\{ \tau , 1 - \tau \} \mathbb { E } | Y - Y ^ { \prime } |$ , so it is Lipschitz in Wasserstein-1. Homogeneity and translation invariance therefore give $\mathcal { W } ( q _ { F } ( x ^ { \pi } ) ) / r _ { n } \to \mathcal { W } ( q _ { F } ^ { \prime } ( x ^ { * } ) Z )$ . Sending $\theta \downarrow 0$ removes the relative factor and yields

$$
\begin{array} { r } { \frac { \operatorname { S R } _ { \delta , n } ( \pi , F ) } { r _ { n } } \longrightarrow \frac { \mathcal { W } \left( q _ { F } ^ { \prime } ( x ^ { * } ) Z \right) } { 1 - \delta } . } \end{array}
$$

Since $p > 0 , \tau \in ( 0 , 1 ) , q _ { F } ^ { \prime } ( x ^ { * } ) \neq 0$ , and � is nondegenerate, $\mathcal { W } ( q _ { F } ^ { \prime } ( x ^ { * } ) Z ) > 0$ . Dividing produces the second limit in (3.16).

The same uniform integrability gives ${ \mathcal { D } } _ { 1 } ( Z _ { n } ) \to { \mathcal { D } } _ { 1 } ( Z )$ by the argument already used for (A.13), and the expansion of ℓ multiplies that deviation by $\vert \ell ^ { \prime } ( x ^ { \ast } ) \vert$ . The resulting liminf and limsup of $\mathrm { S R } _ { \delta , n } ( \pi , F ) / \mathcal { D } _ { 1 } ( x ^ { \pi } )$ lie between min $\left\{ s _ { + } , s _ { - } \right\} \left| \ell ^ { \prime } ( x ^ { * } ) \right|$ | and $\operatorname* { m a x } \{ s _ { + } , s _ { - } \} \vert \ell ^ { \prime } ( x ^ { * } ) \vert$

The scaled ratio (3.18) is the quotient of the two limits just obtained. The post-sample limit yields $\mathrm { P S } _ { \delta , n } ( \pi , F ) / r _ { n } ^ { 2 } \longrightarrow { \textstyle \frac { 1 } { 2 } } Q _ { F } ^ { \prime \prime } ( x ^ { * } ) \mathbb { E } [ Z ^ { 2 } ]$ , so

$$
r _ { n } \rho _ { n } = \frac { { \bf S R } _ { \delta , n } ( \pi , F ) / r _ { n } } { { \bf P S } _ { \delta , n } ( \pi , F ) / r _ { n } ^ { 2 } } \longrightarrow \frac { 2 \mathcal { W } \big ( q _ { F } ^ { \prime } ( x ^ { * } ) Z \big ) } { ( 1 - \delta ) Q _ { F } ^ { \prime \prime } ( x ^ { * } ) \mathbb { E } [ Z ^ { 2 } ] } .
$$

## A.17 Proof of Proposition 3.9

The proof proceeds by coupling $x ^ { \pi }$ with its weak limit. Continuity and uniform integrability then imply $L ^ { 1 }$ convergence of the induced losses. The $L ^ { 1 }$ continuity of CVaR allows us to pass both terms in the regret decomposition to the limit, after which the strict positivity result follows from Lemma A.8 below.

Convergence of the induced losses. By the Skorokhod representation theorem, we may construct random variables $x _ { n } ^ { \pi }$ and $x _ { \infty } ^ { \pi }$ with the same respective distributions as $x ^ { \pi }$ and $x _ { \infty } ^ { \pi }$ such that

$$
x _ { n } ^ { \pi } \longrightarrow x _ { \infty } ^ { \pi }
$$

almost surely.

Let $\xi \sim F$ be independent of $( x _ { n } ^ { \pi } , x _ { \infty } ^ { \pi } )$ . By the assumed continuity of $x \mapsto c ( x , \xi ) \mathrm { a t } x _ { \infty } ^ { \pi } .$

$$
c ( x _ { n } ^ { \pi } , \xi ) \longrightarrow c ( x _ { \infty } ^ { \pi } , \xi ) \qquad \mathrm { a l m o s t \ s u r e l y } .
$$

The family $\{ | c ( x _ { n } ^ { \pi } , \xi ) | \} _ { n \geq 1 }$ is uniformly integrable because $x _ { n } ^ { \pi }$ has the same distribution as $x ^ { \pi }$ . Hence, by uniform integrability,

$$
\mathbb { E } \left[ | c ( x _ { n } ^ { \pi } , \xi ) - c ( x _ { \infty } ^ { \pi } , \xi ) | \right] \longrightarrow 0 .\tag{A.24}
$$

Expected post-sample regret. For integrable random variables � and $Y ^ { \prime }$

$$
\vert \mathrm { C V a R } _ { \delta } [ Y ] - \mathrm { C V a R } _ { \delta } [ Y ^ { \prime } ] \vert \leq \frac { 1 } { 1 - \delta } \mathbb { E } \vert Y - Y ^ { \prime } \vert .
$$

Applying this inequality conditionally on $( x _ { n } ^ { \pi } , x _ { \infty } ^ { \pi } )$ and then taking expectations, (A.24) gives

$$
{ \mathbb E } \big [ \big | \mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x _ { n } ^ { \pi } , \xi ) ] - \mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x _ { \infty } ^ { \pi } , \xi ) ] \big | \big ] \longrightarrow 0 .
$$

Since $x _ { \infty } ^ { \pi }$ is supported on population-optimal actions,

$$
\begin{array} { r } { \mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x _ { \infty } ^ { \pi } , \xi ) ] = \mathrm { O P T } _ { \delta } ( F ) \qquad \mathrm { a l m o s t ~ s u r e l y } . } \end{array}
$$

Therefore,

$$
\mathrm { P S } _ { \delta , n } ( \pi , F ) \longrightarrow 0 .
$$

Sample-risk premium. Applying the same $L ^ { 1 }$ continuity of CVaR directly to (A.24) yields

$$
\mathrm { C V a R } _ { \delta , x _ { n } ^ { \pi } , \xi } \left[ c ( x _ { n } ^ { \pi } , \xi ) \right] \longrightarrow \mathrm { C V a R } _ { \delta , x _ { \infty } ^ { \pi } , \xi } \left[ c ( x _ { \infty } ^ { \pi } , \xi ) \right] .
$$

Together with the preceding convergence of the expected outcome-only CVaR, this implies

$$
\begin{array} { r } { \mathrm { S R } _ { \delta , n } ( \pi , F ) \longrightarrow \mathrm { C V a R } _ { \delta , x _ { \infty } ^ { \pi } , \xi } [ c ( x _ { \infty } ^ { \pi } , \xi ) ] - \mathbb { E } _ { x _ { \infty } ^ { \pi } } \big [ \mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x _ { \infty } ^ { \pi } , \xi ) ] \big ] = : \mathrm { S R } _ { \delta , \infty } ( \pi , F ) . } \end{array}
$$

The limit is the sample-risk premium of the limiting action. The same comparison holds for any integrable action independent of the future outcome, so the strict inequality is recorded on its own.

Lemma A.8. Under the hypotheses of Proposition 3.2, if the conditional tails induced by $x ^ { \pi }$ are misaligned,

then

$$
C V a R _ { \delta , x ^ { \pi } , \xi } \left[ c ( x ^ { \pi } , \xi ) \right] > \mathbb { E } _ { x ^ { \pi } } \left[ C V a R _ { \delta , \xi \sim F } \left[ c ( x ^ { \pi } , \xi ) \right] \right] .
$$

Consequently, the sample-risk premium of � is strictly positive.

The proof is in Appendix A.18. The condition

$$
\mathbb { P } \big ( t \in \mathcal { T } _ { \delta } ( x _ { \infty } ^ { \pi } ) \big ) < 1 \qquad \mathrm { f o r } \mathrm { e v e r y } t \in \mathbb { R }
$$

is precisely the condition that the conditional tails induced by $x _ { \infty } ^ { \pi }$ are misaligned. Since $c ( x _ { \infty } ^ { \pi } , \xi )$ is integrable and $x _ { \infty } ^ { \pi }$ is independent of $\xi ,$ Lemma A.8 gives

$$
\begin{array} { r } { \mathrm { S R } _ { \delta , \infty } ( \pi , F ) > 0 . } \end{array}
$$

Consequently,

$$
\mathrm { P S } _ { \delta , n } ( \pi , F ) = o ( 1 ) , \qquad \mathrm { S R } _ { \delta , n } ( \pi , F ) = \Theta ( 1 ) .
$$

## A.18 Proof of Lemma A.8

This is the contrapositive of part 1 of Proposition 3.2. In the notation of Appendix A.1, let $t ^ { * }$ attain the variational representation of $\operatorname { C V a R } _ { \delta } [ c ( x ^ { \pi } , \xi ) ]$ , which exists by integrability; as shown there, $t ^ { * }$ also minimizes $G$ . Misalignment gives $\mathbb { P } ( t ^ { * } \notin \mathcal { T } _ { \delta } ( x ^ { \pi } ) ) > 0$ , and on that event $\phi _ { x ^ { \pi } } ( t ^ { * } ) > \mathrm { m i n } _ { s } \phi _ { x ^ { \pi } } ( s )$ . Since the integrand defining $G$ is nonnegative, $\mathrm { S R } _ { \delta , n } ( \pi , F ) = G ( t ^ { * } ) > 0$ , which is the claimed strict inequality.

## A.19 Proof of Proposition 3.10

Write $x _ { F } ^ { * }$ for the unique optimal action. For any policy � and any $\varepsilon > 0$ , part 2 of Proposition 3.2 gives that the integrand

$$
\mathrm { C V a R } _ { \delta , \xi \sim F } [ c ( x ^ { \pi } , \xi ) ] - \mathrm { O P T } _ { \delta } ( F )
$$

is nonnegative everywhere, and by definition of $\kappa _ { F } ( \varepsilon )$ it is at least $\kappa _ { F } ( \varepsilon )$ on the event $\{ | x ^ { \pi } - x _ { F } ^ { * } | \ge \varepsilon \}$ Hence

$$
\begin{array} { r l r } { \mathrm { P S } _ { \delta , n } ( \pi , F ) } & { \geq } & { \kappa _ { F } ( \varepsilon ) \cdot \mathbb { P } \big ( | x ^ { \pi } - x _ { F } ^ { * } | \geq \varepsilon \big ) . } \end{array}
$$

Under the hypothesis $\kappa _ { F } ( \varepsilon ) > 0$ , if for some $\varepsilon > 0$ and $\eta > 0$ one has

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } _ { \mathbb { P } } \bigl ( | x ^ { \pi } - x _ { F } ^ { * } | \geq \varepsilon \bigr ) \geq \eta ,
$$

the displayed inequality gives

$$
\operatorname* { l i m i n f } _ { n \to \infty } \mathrm { P S } _ { \delta , n } ( \pi , F ) \geq \eta \kappa _ { F } ( \varepsilon ) > 0 ,
$$

hence $\mathrm { P S } _ { \delta , n } ( \pi , F ) = \Omega ( 1 )$

For the consistency claim, suppose $\mathrm { P S } _ { \delta , n } ( \pi , F ) \to 0$ . Fix $\varepsilon > 0$ . Then $\kappa _ { F } ( \varepsilon ) > 0$ , and

$$
\begin{array} { r c l } { \mathbb { P } \big ( | x ^ { \pi } - x _ { F } ^ { * } | \ge \varepsilon \big ) } & { \le } & { \frac { \mathrm { P S } _ { \delta , n } ( \pi , F ) } { \kappa _ { F } ( \varepsilon ) } \longrightarrow 0 . } \end{array}
$$

Since $\varepsilon > 0$ was arbitrary, $x ^ { \pi } \to x _ { F } ^ { * }$ in probability.

## B Post-sample consistency of SAA

## B.1 Proof of Proposition 4.1

Write $\widehat { Q } _ { n } ( x ) = \mathrm { C V a R } _ { \delta , \xi \sim \hat { F } _ { n } } [ c ( x , \xi ) ]$ , and write $\phi _ { n } ( x , t )$ for the empirical counterpart of the variational objective $\phi _ { F } ( x , t )$ . The argument is a uniform law of large numbers on the variational objective, passed to the infimum over the threshold and then to the minimizer. Boundedness localizes the threshold. Compactness and almost-sure continuity give the uniform law.

The threshold lies in $[ - M , M ]$ . The envelope $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } | c ( x , \xi ) | \ \le \ M } \end{array}$ holds for �-almost every $\xi ,$ so $c ( x , \xi ) \in [ - M , M ]$ for every $x \in \chi$ , outside a single null set. For $t > M , ( c ( x , \xi ) - t ) ^ { + } = 0 $ , hence $\phi _ { F } ( x , t ) = t > M = \phi _ { F } ( x , M )$ . For $t < - M , ( c ( x , \xi ) - t ) ^ { + } = c ( x , \xi ) - t ,$ , and

$$
\phi _ { F } ( x , t ) = \frac { \mathbb { E } [ c ( x , \xi ) ] } { 1 - \delta } - \frac { \delta } { 1 - \delta } t ,
$$

which is strictly decreasing in �. Thus $\phi _ { F } ( x , t ) > \phi _ { F } ( x , - M )$ whenever $t < - M$ . The same identities hold pathwise for $\phi _ { n }$ , because almost every sample lies in the envelope. Therefore both infima are attained inside $[ - M , M ]$

$$
Q _ { F } ( x ) = \operatorname * { i n f } _ { t \in [ - M , M ] } \phi _ { F } ( x , t ) , \qquad { \widehat { Q } } _ { n } ( x ) = \operatorname * { i n f } _ { t \in [ - M , M ] } \phi _ { n } ( x , t ) .
$$

Uniform convergence of the variational objective. Let $\Theta = \textstyle X \times [ - M , M ]$ , which is compact, and write a generic point as $\theta = \left( x , t \right)$ . Define

$$
f ( \theta , \xi ) = \big ( c ( x , \xi ) - t \big ) ^ { + } .
$$

Almost-sure continuity of� $\mapsto c ( x , \xi )$ makes $\theta \mapsto f ( \theta , \xi )$ continuous for �-almost every $\xi ,$ , and $0 \leq f \leq 2 M$ For $\eta > 0$ let

$$
\omega ( \xi , \eta ) = \operatorname* { s u p } \big \{ | f ( \theta , \xi ) - f ( \theta ^ { \prime } , \xi ) | : \theta , \theta ^ { \prime } \in \Theta , \ | \theta - \theta ^ { \prime } | \le \eta \big \} ,
$$

where the supremum may be taken over a countable dense set of pairs and then agrees with the supremum over all pairs, on the almost-sure set where $f ( \cdot , \xi )$ is continuous. Thus $\omega ( \xi , \eta )$ is measurable, $0 \leq \omega ( \xi , \eta ) \leq 4 M$ and $\omega ( \xi , \eta ) \to 0$ for �-almost every $\xi$ as $\eta  0$ , by uniform continuity on the compact set $\Theta .$ . Dominated convergence gives $\mathbb { E } [ \omega ( \xi , \eta ) ]  0$

Fix $\eta > 0$ and cover Θ by finitely many balls of radius � with centers $\theta _ { 1 } , \ldots , \theta _ { J }$ . For an arbitrary �, choose a center $\tau ( \theta )$ within distance $\eta .$ Splitting the empirical average at that center and using $| f ( \theta , \xi ) - f ( \tau ( \theta ) , \xi ) | \le$ $\omega ( \xi , \eta )$ gives

$$
\left| \phi _ { n } ( x , t ) - \phi _ { F } ( x , t ) \right| \leq \frac { 1 } { 1 - \delta } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \omega ( \xi _ { i } , \eta ) + \operatorname* { m a x } _ { 1 \leq j \leq J } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f ( \theta _ { j } , \xi _ { i } ) - \mathbb { E } [ f ( \theta _ { j } , \xi ) ] \right| + \mathbb { E } [ \omega ( \xi , \eta ) ] \right) .
$$

The right-hand side does not depend on $\theta .$ The strong law sends the average of $\omega ( \cdot , \eta )$ to its expectation and each of the finitely many centered averages to zero. Hence, almost surely,

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } _ { \theta \in \Theta } \Bigl | \phi _ { n } ( x , t ) - \phi _ { F } ( x , t ) \Bigr | \leq \frac { 2 } { 1 - \delta } \mathbb { E } [ \omega ( \xi , \eta ) ] .
$$

The null set can be taken independent of $\eta = 1 / k$ for $k \in \mathbb { N }$ . The left-hand side does not depend on $k ,$ and the right-hand side tends to zero with �, so the limsup is zero. That is,

$$
\operatorname* { s u p } _ { x \in X , t \in [ - M , M ] } \left| \phi _ { n } ( x , t ) - \phi _ { F } ( x , t ) \right| \longrightarrow 0 \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

An infimum over � moves by at most the uniform distance of the two objectives, so

$$
\operatorname * { s u p } _ { x \in \chi } \left| \widehat { Q } _ { n } ( x ) - Q _ { F } ( x ) \right| \longrightarrow 0 \qquad \mathrm { a l m o s t \ s u r e l y } .
$$

Whenever $| ( x , t ) - ( y , s ) | \leq \eta$

$$
\left| \phi _ { F } ( x , t ) - \phi _ { F } ( y , s ) \right| \le \left| t - s \right| + \frac { \mathbb { E } [ \omega ( \xi , \eta ) ] } { 1 - \delta } .
$$

The right-hand side tends to zero with $\eta ,$ so $\phi _ { F }$ is jointly continuous on $\chi \times [ - M , M ]$ . Minimizing over the compact threshold interval gives continuity of $Q _ { F } ( \cdot )$ on $\chi$ . A continuous function on a nonempty compact set attains its minimum, so $\chi _ { _ { F } } ^ { * }$ is nonempty and compact. On the same almost-sure set, each map $x \mapsto c ( x , \xi _ { i } )$ is continuous, so ${ \widehat { Q } } _ { n }$ is continuous and its argmin is nonempty. The smallest minimizer is a measurable selection; the argument below applies to any measurable selection, and $x ^ { \mathrm { S A A } }$ denotes the one fixed in the text.

The post-sample gap. Let $x _ { F } ^ { * } \in X _ { F } ^ { * }$ . Optimality of $x ^ { \mathrm { S A A } }$ for ${ \widehat { Q } } _ { n }$ gives $\widehat { Q } _ { n } ( x ^ { \mathrm { S A A } } ) \leq \widehat { Q } _ { n } ( x _ { F } ^ { * } )$ . Adding and subtracting $\widehat { Q } _ { n } ( x ^ { \mathrm { S A A } } )$ and $\widehat { Q } _ { n } ( x _ { F } ^ { * } )$ and dropping that nonpositive diference,

$$
\begin{array} { r l } & { 0 \leq Q _ { F } \Big ( x ^ { \mathrm { S A A } } \Big ) - \mathrm { O P T } _ { \delta } ( F ) } \\ & { \ = \big ( Q _ { F } \Big ( x ^ { \mathrm { S A A } } \Big ) - \widehat { Q } _ { n } ( x ^ { \mathrm { S A A } } ) \big ) + \big ( \widehat { Q } _ { n } ( x ^ { \mathrm { S A A } } ) - \widehat { Q } _ { n } ( x _ { F } ^ { * } ) \big ) + \big ( \widehat { Q } _ { n } ( x _ { F } ^ { * } ) - Q _ { F } \big ( x _ { F } ^ { * } \big ) \big ) } \\ & { \ \leq 2 \underset { x \in \mathcal { X } } { \operatorname* { s u p } } \big | \widehat { Q } _ { n } ( x ) - Q _ { F } ( x ) \big | . } \end{array}
$$

The right-hand side tends to zero almost surely, so the random post-sample gap vanishes almost surely. The gap is at most 2�. Bounded convergence gives

$$
\mathrm { P S } _ { \delta , n } ( \pi ^ { \mathrm { S A A } } , F ) = \mathbb { E } \big [ Q _ { F } \Big ( x ^ { \mathrm { S A A } } \Big ) \big ] - \mathrm { O P T } _ { \delta } ( F ) \longrightarrow 0 .
$$

## C SAA: asymptotics and finite-sample calculations

For the concentration proof, write $x ^ { * } = x _ { F } ^ { * } , q ^ { * } = q _ { F } ( x ^ { * } )$ , and $Q ^ { \prime \prime } = Q _ { F } ^ { \prime \prime } ( x ^ { * } )$ . Let $a _ { F }$ be the cutof slope defined in (4.3). Write $\Delta _ { n } = x ^ { \mathrm { S A A } } - x ^ { * }$ , and $\widehat { Q } _ { n } ( x ) = \mathrm { C V a R } _ { \delta , \xi \sim \widehat { F } _ { n } } [ c ( x , \xi ) ]$ . Let � be the uniform loss bound in Proposition 4.1. For an integrable function $f ,$ use $P f = \mathbb { E } [ f ( \xi ) ]$ and $\begin{array} { r } { P _ { n } f = n ^ { - 1 } \sum _ { i = 1 } ^ { n } f ( \xi _ { i } ) } \end{array}$

Lemma C.1 (Quadratic gap). Let $x ^ { * }$ be an interior minimizer of $\cdot _ { Q _ { F } , }$ , and suppose $Q _ { F }$ is twice diferentiable there with $0 < Q ^ { \prime \prime } < \infty$ . Then there are a neighborhood N of $\boldsymbol { \cdot } _ { x ^ { * } }$ and constants $0 < c \le C < \infty$ such that, for every $x \in N$

$$
\begin{array} { r l r } { c \lvert x - x ^ { * } \rvert ^ { 2 } } & { \leq } & { Q _ { F } ( x ) - Q _ { F } ( x ^ { * } ) \leq { \cal \ C } \lvert x - x ^ { * } \rvert ^ { 2 } . } \end{array}
$$

Proof. By hypothesis, $x ^ { * }$ is an interior minimizer of $Q _ { F } ( \cdot )$ and $0 < Q ^ { \prime \prime } < \infty$ . The first derivative vanishes at an interior minimizer, and Taylor’s theorem with Peano remainder gives

$$
\begin{array} { r } { Q _ { F } ( x ) - Q _ { F } ( x ^ { * } ) = \frac { 1 } { 2 } Q ^ { \prime \prime } \left( x - x ^ { * } \right) ^ { 2 } + o \left( \left( x - x ^ { * } \right) ^ { 2 } \right) } \end{array}
$$

as $x \to x ^ { * }$ . Positivity and finiteness of the second derivative produce the neighborhood and the constants. □

Lemma C.2 (Parametric rate). Suppose the hypotheses of Proposition 4.2. With $0 < \sigma ^ { 2 } < \infty$ as in (4.3),

$$
\sqrt { n } \Delta _ { n } \Rightarrow N ( 0 , \sigma ^ { 2 } ) , \qquad \{ n \Delta _ { n } ^ { 2 } \} _ { n \geq 1 } i s u n i f o r m l y i n t e g r a b l e .
$$

Consequently, $\Delta _ { n } = \Theta _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ .

Proof. We first identify a centered score with positive variance and bound the local empirical process. This bound localizes the joint SAA minimizer and controls its tails. We obtain asymptotic linearity from the joint M-estimation theorem in the smooth regime and from a direct comparison in the atomic regime, then conclude uniform integrability.

Well separation and the post-sample consistency of SAA give $\Delta _ { n }  0$ in probability.

The slope is a nondegenerate average. We identify the score governing the action fluctuation. By the hypothesis of the proposition the action derivative exists at $x ^ { * }$ for almost every outcome, and equation (4.2) forces $| \partial _ { x } c ( x ^ { * } , \xi ) | \leq \mu ( \xi )$ . Use $\psi = \psi _ { F }$ , with $\psi _ { F }$ as in (4.3), and set $S _ { n } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \psi ( \xi _ { i } )$ and $\nu = \mathrm { V a r } ( \psi ( \xi ) )$ . The bound by a constant multiple of $1 + \mu$ puts $\psi$ in $L ^ { 2 }$ . We verify below that it is centered and $\nu > 0 ;$ centering then gives $\mathbb { E } [ n S _ { n } ^ { 2 } ] = \nu$ for every �.

In the smooth regime, the score in equation (4.3) can be written as

$$
\psi ( \xi ) = \frac { 1 } { ( 1 - \delta ) } \left( \partial _ { x } c ( x ^ { * } , \xi ) \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) > q ^ { * } \} } + a _ { F } \big ( \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) \leq q ^ { * } \} } - \delta \big ) \right) .
$$

There is no atom at the cutof, so $\mathbb { P } ( c ( x ^ { * } , \xi ) > q ^ { * } ) = ( 1 - \delta )$ and the threshold-score term has mean zero. The local envelope permits diferentiation under the expectation at a fixed threshold. Since $\phi _ { F } ( x , q ^ { * } ) - Q _ { F } ( x )$ is nonnegative and vanishes at the interior point $x ^ { * }$ , its derivative there is zero. Thus

$$
\mathbb { E } \psi = \frac { \mathbb { E } \big [ \partial _ { x } c ( x ^ { * } , \xi ) \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) > q ^ { * } \} } \big ] } { ( 1 - \delta ) } = \partial _ { x } \phi _ { F } ( x ^ { * } , q ^ { * } ) = Q _ { F } ^ { \prime } ( x ^ { * } ) = 0 .
$$

On $\{ c ( x ^ { * } , \xi ) < q ^ { * } \}$ , which has probability $\delta > 0$ , the score equals $a _ { F } \neq 0$ . Its variance � is therefore positive. In the atomic regime the cutof sits on the atom, so the score is the derivative of the variational objective along the path that moves the action and the atom level together.

Here $a _ { F }$ is the derivative of the atom level at $x ^ { * }$ . The criterion along that path is $\ell ( x ) + ( c ( x , \xi ) -$ $\ell ( x ) ) ^ { + } / ( 1 - \delta )$ . The isolation limits (3.7) force the action derivative on the atom to equal $a _ { F }$

If, for some $\varepsilon > 0$ , the event $E _ { \varepsilon } : = \{ c ( x ^ { * } , \xi ) = q ^ { * } , \partial _ { x } c ( x ^ { * } , \xi ) - a _ { F } > \varepsilon \}$ has positive probability, diferentiability implies

$$
{ \bf 1 } _ { E _ { \varepsilon } } { \bf 1 } _ { \left\{ c \left( x ^ { * } + h , \xi \right) > \ell \left( x ^ { * } + h \right) + \varepsilon h / 2 \right\} } \longrightarrow { \bf 1 } _ { E _ { \varepsilon } }\tag{ℎ ↓ 0).}
$$

By continuity, the original strict upper-tail outcomes also remain above this boundary asymptotically. Dominated convergence on these two disjoint events gives a limit inferior of the upper-tail probability at least $\mathbb { P } ( c ( x ^ { * } , \xi ) > q ^ { * } ) + \mathbb { P } ( E _ { \varepsilon } )$ , contradicting the first isolation limit along $u = \varepsilon h / 2$ . If the derivative diference is below −� on a positive-probability subset of the atom, the analogous argument removes that mass from the event in the second isolation limit, while outcomes initially below the atom remain below the moving boundary asymptotically. This contradicts the second limit. Hence $\partial _ { x } c = a _ { F }$ almost surely on the oracle atom.

Outcomes of the atom may still cross the moving level. For each increment ℎ, let $C _ { h }$ be the event that a loss initially of the oracle atom crosses the moving level between $x ^ { * }$ and $x ^ { * } + h$ . Of the oracle atom, $c ( x ^ { * } , \xi ) \neq \ell ( x ^ { * } )$ ; continuity of $c ( \cdot , \xi )$ and $\ell$ makes the crossing indicator tend to zero. Dominated convergence therefore gives $\mathbb { P } ( C _ { h } ) \to 0$ . The diference quotients are dominated by a constant multiple of $1 + \mu .$ , so dominated convergence controls both these crossings and the ordinary action-derivative remainder. On the oracle atom, derivative alignment instead gives $c ( x ^ { * } + h , \xi ) - \ell ( x ^ { * } + h ) = o ( | h | )$ almost surely. Thus the moving-path remainder is $o ( | h | )$ in $L ^ { 2 }$

Where the positive part does not switch, the derivative equals $a _ { F }$ on $\{ c \leq q ^ { * } \}$ and $( \partial _ { x } c - \delta a _ { F } ) / ( 1 - \delta )$ on $\{ c > q ^ { * } \}$ . The score is therefore

$$
\psi ( \xi ) ~ = ~ a _ { F } + \frac { \partial _ { x } c ( x ^ { * } , \xi ) - a _ { F } } { ( 1 - \delta ) } \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) > q ^ { * } \} } .
$$

It equals $a _ { F }$ on the atom. Since $a _ { F } \neq 0$ and the atom has positive probability, $\psi$ is not almost surely zero. Dominated convergence under (4.2) passes the derivative under the expectation, so $\psi$ is centered because $Q _ { F } ^ { \prime } ( x ^ { * } ) = 0$ , and a centered square-integrable random variable that is not almost surely zero has variance $\nu > 0 , S _ { n }$ is its sample average.

In both regimes the central limit theorem gives ${ \sqrt { n } } S _ { n } \Rightarrow N ( 0 , \nu )$ with $\nu > 0$ . In particular $S _ { n } \ =$ $O _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ and $S _ { n }$ is not $o _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$

A local empirical-process bound. Write $f _ { x , t } ( \xi ) = t + ( c ( x , \xi ) - t ) ^ { + } / ( 1 - \delta )$ . The local bound (4.2) gives an envelope $\mu _ { f } ( \xi ) = C ( 1 + \mu ( \xi ) )$ with $\mathbb { E } [ \mu _ { f } ^ { 2 + \eta } ] < \infty$ , such that, for all actions $x , y$ in the neighborhood of (4.2) and all thresholds $t , t ^ { \prime }$

$$
| f _ { x , t } - f _ { y , t ^ { \prime } } | \leq \mu _ { f } ( \xi ) ( | x - y | + | t - t ^ { \prime } | ) .
$$

Let $\mathcal { B } _ { \lambda }$ be the square of radius � centered at $( x ^ { * } , q ^ { * } )$ , and define

$$
G _ { n } ( x , t ) = \sqrt { n } ( P _ { n } - P ) ( f _ { x , t } - f _ { x ^ { * } , q ^ { * } } ) .
$$

Local maximal bound. For all suficiently small $\lambda > 0$ and all $n ,$

$$
\mathbb { E } \Bigg [ \operatorname* { s u p } _ { \mathcal { B } _ { \lambda } } | G _ { n } | ^ { 2 + \eta } \Bigg ] \leq C _ { \eta } \lambda ^ { 2 + \eta } \mathbb { E } [ \mu _ { f } ^ { 2 + \eta } ] ,\tag{C.1}
$$

where $C _ { \eta }$ is independent of � and �.

The local Lipschitz envelope makes $G _ { n }$ almost surely continuous, so its supremum equals the supremum over the countable union of dyadic grids and is measurable. To prove the bound, symmetrize $G _ { n }$ and use dyadic grids with spacing $d _ { k } \ = \ \lambda 2 ^ { - k }$ . Connect each point to $( x ^ { * } , q ^ { * } )$ through its successive grid projections. Conditionally on the sample, each Rademacher increment across a grid edge is sub-Gaussian with variance proxy at most $C ( P _ { n } \mu _ { f } ^ { 2 } ) d _ { k } ^ { 2 }$ . There are at most $C 4 ^ { k }$ edges at level $k ,$ so the conditional $L ^ { 2 + \eta }$ norm of their maximum is at most $\dot { C } _ { \eta } d _ { k } \sqrt { k + 1 } ( P _ { n } \mu _ { f } ^ { 2 } ) ^ { 1 / 2 }$ . The Lipschitz bound makes the terminal grid remainder vanish for each fixed $n .$ Minkowski’s inequality, moment symmetrization, and the convergence of $\begin{array} { r } { \sum _ { k } 2 ^ { - k } \sqrt { k + 1 } } \end{array}$ therefore bound the left-hand side of (C.1) by $C _ { \eta } \lambda ^ { 2 + \eta } \mathbb { E } [ ( P _ { n } \mu _ { f } ^ { 2 } ) ^ { ( 2 + \eta ) / 2 } ]$ . Jensen’s inequality gives $\mathbb { E } [ ( P _ { n } \mu _ { f } ^ { 2 } ) ^ { ( 2 + \eta ) / 2 } ] \le \mathbb { E } [ \mu _ { f } ^ { 2 + \eta } .$ , proving the claim.

Localization at the root-� scale. Bounded loss confines every minimizing threshold to $[ - M , M ]$ , as shown in Appendix B. Let $t _ { n } ( x )$ minimize $t \mapsto P _ { n } f _ { x , t }$ and write $t _ { n } = t _ { n } ( x ^ { \mathrm { S A A } } )$ . Since $\widehat { Q } _ { n } ( x ^ { \mathrm { S A A } } ) \leq \widehat { Q } _ { n } ( x ^ { * } ) \leq$ $P _ { n } f _ { x ^ { * } , q ^ { * } }$ ，

$$
P f _ { x ^ { \mathrm { S A A } } , t _ { n } } - P f _ { x ^ { * } , q ^ { * } } \leq n ^ { - 1 / 2 } \big | G _ { n } ( x ^ { \mathrm { S A A } } , t _ { n } ) \big | .
$$

The left-hand side is the quadratic action gap plus the threshold excess above $q _ { F } ( x ^ { \mathrm { S A A } } )$ . The minimizing threshold at $x ^ { * }$ is unique: the local quadratic threshold gap in Lemma $\mathsf { A } . 2 .$ , or strict straddling in Lemma $\mathrm { A } . 7$ together with convexity in the threshold, gives the singleton $\{ q ^ { * } \}$ . Since the oracle action is unique, the joint population minimizer is $( x ^ { * } , q ^ { * } )$ . Continuity on the compact set $\chi \times [ - M , M ]$ and the uniform law proved in Appendix B then give joint consistency of $( x ^ { \mathrm { S A A } } , t _ { n } )$ . Choose a suficiently small fixed neighborhood. There $| q _ { F } ( x ) - q ^ { * } | \leq ( | a _ { F } | + 1 ) | x - x ^ { * } |$ , and the threshold excess is at least a positive multiple of $| t - q _ { F } ( x ) | ^ { \gamma }$ with $\gamma = 2$ in the smooth regime and � = 1 $\gamma = 1$ in the atomic regime. On this bounded neighborhood the linear atomic bound also dominates a quadratic bound. Consequently the joint population gap is at least a positive multiple of

$$
| x - x ^ { * } | ^ { 2 } + | t - q ^ { * } | ^ { 2 }
$$

in both regimes. Write

$$
R _ { n } = \operatorname* { m a x } \big ( | \Delta _ { n } | , | t _ { n } - q ^ { * } | / ( | a _ { F } | + 1 ) \big )
$$

and $\lambda _ { j } = 2 ^ { j } n ^ { - 1 / 2 }$ . On an annulus $\{ \lambda _ { j - 1 } < R _ { n } \leq \lambda _ { j } \}$ inside a suficiently small fixed neighborhood, the basic inequality requires

$$
\operatorname* { s u p } _ { \mathcal { B } _ { C \lambda _ { j } } } | G _ { n } | \geq c \sqrt { n } \lambda _ { j - 1 } ^ { 2 } .
$$

Bound (C.1) and Markov’s inequality make its probability at most $C _ { \eta } 2 ^ { - j ( 2 + \eta ) }$ . Choose a fixed radius � small enough that the bounds hold on all the squares needed to cover $\{ R _ { n } \leq r \}$ . Summing the annuli, including the last annulus intersected with this set, gives

$$
\begin{array} { r } { \mathbb { P } ( R _ { n } > 2 ^ { J } n ^ { - 1 / 2 } , R _ { n } \le r ) \le C _ { \eta } 2 ^ { - J ( 2 + \eta ) } . } \end{array}
$$

Joint consistency controls $\{ R _ { n } > r \}$ . Thus $\Delta _ { n } = O _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ and $t _ { n } - q ^ { * } = O _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$

Asymptotic linearity. Joint localization and $S _ { n } ~ = ~ O _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ place the minimizer and the candidate $h _ { n } = - S _ { n } / Q ^ { \prime \prime }$ in a fixed root-� window with arbitrarily high probability. We prove the expansions on each window $| h | , | u | \leq R _ { 0 } n ^ { - 1 / 2 }$ , with $x = x ^ { * } + h$ and $t = q ^ { * } + u ;$ the joint integrand is locally Lipschitz in these coordinates.

The population half expands in both coordinates together. Lemma C.1 gives $Q _ { F } ( x ^ { * } + h ) - Q _ { F } ( x ^ { * } ) =$ $\scriptstyle { \frac { 1 } { 2 } } Q ^ { \prime \prime } h ^ { 2 } + o ( h ^ { 2 } )$ . In the smooth regime the proof of Lemma A.2 gives threshold curvature $g / ( 2 ( 1 - \delta ) )$ , with $g = g _ { x ^ { * } } ( q ^ { * } )$ , and $u ( h ) : = q _ { F } ( x ^ { * } + h ) - q ^ { * } = a _ { F } h + o ( | h | )$ . Uniformly for $| h |$ and $| u |$ of order $n ^ { - 1 / 2 }$

$$
P f _ { x ^ { * } + h , q ^ { * } + u } = Q _ { F } ( x ^ { * } ) + \textstyle { \frac { 1 } { 2 } } Q ^ { \prime \prime } h ^ { 2 } + \frac { g } { 2 ( 1 - \delta ) } ( u - u ( h ) ) ^ { 2 } + o ( n ^ { - 1 } ) .
$$

In the atomic regime Lemma $\mathrm { A } . 7$ gives $q _ { F } = \ell$ near the oracle and a linear threshold excess of slope at least half of min $\{ s _ { + } , s _ { - } \} > 0$ , with the slopes of (A.22). The quadratic term in ℎ is again the lemma above.

In the smooth regime, apply the minimization form of van der Vaart (1998, Theorem 5.23) to the joint criterion $f _ { x , t }$ . Joint consistency was established above. There is no atom at $q ^ { * }$ , so the criterion is almost surely diferentiable at $( x ^ { * } , q ^ { * } )$ ; the local Lipschitz envelope is a constant multiple of $1 + \mu$ and lies in $L ^ { 2 }$

The population expansion above has positive definite matrix

$$
\mathsf { H } = \left( \begin{array} { c c } { Q ^ { \prime \prime } + b a _ { F } ^ { 2 } } & { - b a _ { F } } \\ { - b a _ { F } } & { b } \end{array} \right) , \qquad b = g / ( 1 - \delta ) , \qquad \mathsf { d e t H } = Q ^ { \prime \prime } b > 0 .
$$

Exact SAA minimization satisfies the theorem’s optimization-error condition. With

$$
\dot { f } ( \xi ) = \binom { \partial _ { x } c ( x ^ { * } , \xi ) \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) > q ^ { * } \} } / ( 1 - \delta ) } { 1 - \mathbf { 1 } _ { \{ c ( x ^ { * } , \xi ) > q ^ { * } \} } / ( 1 - \delta ) } ,
$$

the theorem gives the joint linear representation with influence $- \mathsf { H } ^ { - 1 } \dot { f }$ . Its action coordinate is $- \psi _ { F } / Q ^ { \prime \prime }$ since the first row of $\mathsf { H } ^ { - 1 }$ is $( 1 , a _ { F } ) / Q ^ { \prime \prime }$ . Thus

$$
\sqrt { n } \left( x ^ { \mathrm { S A A } } - x ^ { * } \right) = - \frac { \sqrt { n } S _ { n } } { Q ^ { \prime \prime } } + o _ { \mathbb { P } } ( 1 ) .
$$

Stochastic remainder bound. Suppose $r _ { 0 , 0 } = 0$ , the class $r _ { h , w }$ is locally Lipschitz in $( h , w )$ with envelope $C \mu _ { f }$ , and $\operatorname* { s u p } _ { | h | , | w | \leq \lambda } \| r _ { h , w } \| _ { 2 } = o ( \lambda )$ . For every fixed $0 < R _ { 0 } < \infty$

$$
\operatorname* { s u p } _ { | h | , | w | \leq R _ { 0 } / \sqrt { n } } | ( P _ { n } - P ) r _ { h , w } | = o _ { \mathbb { P } } ( n ^ { - 1 } ) .\tag{C.2}
$$

To see this, put $\lambda = R _ { 0 } / \sqrt { n }$ and normalize the class as $r _ { \lambda \beta _ { 1 } , \lambda \beta _ { 2 } } / \lambda .$ , with $\beta \in [ - 1 , 1 ] ^ { 2 }$ . On any fixed finite grid, the expected maximum of its centered empirical process ${ \sqrt { n } } ( P _ { n } - P )$ tends to zero by the $L ^ { 2 }$ assumption. After symmetrization, the fine-grid tail has expected supremum at most

$$
C \| \mu _ { f } \| _ { 2 } \sum _ { k \geq k _ { 0 } } 2 ^ { - k } \sqrt { k + 1 } ,
$$

uniformly in �, by the same chaining argument. First let � grow on the fixed grid, then let $k _ { 0 }$ grow. Rescaling gives (C.2).

In the atomic regime use relative threshold coordinates:

$$
\widetilde { f } _ { h , w } = f _ { x ^ { * } + h , \ell \left( x ^ { * } + h \right) + w } , \qquad B = 1 - \frac { \mathbf { 1 } _ { \{ c \left( x ^ { * } , \xi \right) > q ^ { * } \} } } { \left( 1 - \delta \right) } , \qquad D = \frac { \mathbf { 1 } _ { \{ c \left( x ^ { * } , \xi \right) = q ^ { * } \} } } { \left( 1 - \delta \right) } .
$$

Retain the threshold kink. The integrand expansion is

$$
\widetilde { f } _ { h , w } - \widetilde { f } _ { 0 , 0 } = h \psi + w B + ( - w ) ^ { + } D + r _ { h , w } .
$$

Of the oracle atom, continuity makes the sign of the hinge constant on a suficiently small square for each outcome. On the atom, derivative alignment gives $c ( x ^ { * } + h , \xi ) - \ell ( x ^ { * } + h ) = o ( | h | )$ ; the Lipschitz property

of the positive part controls the remainder uniformly in �. Diferentiability therefore yields

$$
\operatorname* { s u p } _ { | h | , | w | \leq \lambda } { \frac { | r _ { h , w } ( \xi ) | } { \lambda } } \longrightarrow 0 \quad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

This supremum is bounded by a constant multiple of $1 + \mu$ . Dominated convergence therefore gives

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { | h | , | w | \leq \lambda } | r _ { h , w } | ^ { 2 } \right] = o ( \lambda ^ { 2 } ) , \quad \quad \operatorname* { s u p } _ { | h | , | w | \leq \lambda } \| r _ { h , w } \| _ { 2 } = o ( \lambda ) .
$$

The second bound follows from the first. Since ℓ is continuously diferentiable, the remainder also has a Lipschitz envelope $C \mu _ { f }$ . Bound (C.2) therefore applies.

Keep the population threshold excess exact:

$$
T ( h , w ) = P ( \widetilde { f } _ { h , w } - \widetilde { f } _ { h , 0 } ) \geq c _ { \mathrm { a t } } | w |
$$

locally, for some $c _ { \mathrm { a t } } > 0 .$ , by Lemma A.7. With the centered averages $B _ { n } = ( P _ { n } - P ) B$ and $D _ { n } = ( P _ { n } - P ) D$ the empirical expansion is

$$
P _ { n } \widetilde { f } _ { h , w } - P _ { n } \widetilde { f } _ { 0 , 0 } = \frac { Q ^ { \prime \prime } } { 2 } h ^ { 2 } + h S _ { n } + T ( h , w ) + w B _ { n } + ( - w ) ^ { + } D _ { n } + o _ { \mathbb { P } } ( n ^ { - 1 } ) .
$$

The indicators are bounded, so $| B _ { n } | + | D _ { n } | = O _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ . With probability tending to one, the threshold terms are therefore at least $c _ { \mathrm { a t } } | \boldsymbol { w } | / 2$ . Joint localization and local Lipschitz continuity of ℓ give $w _ { n } = t _ { n } - \ell ( x ^ { \mathrm { S A A } } ) =$ $O _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ . Compare this joint minimizer with the feasible candidate $\left( h _ { n } , 0 \right) = \left( - S _ { n } / Q ^ { \prime \prime } , 0 \right)$ . Optimality and completion of the action square give

$$
\frac { Q ^ { \prime \prime } } { 2 } ( \Delta _ { n } - h _ { n } ) ^ { 2 } + \frac { c _ { \mathrm { a t } } } { 2 } | w _ { n } | \leq o _ { \mathbb { P } } ( n ^ { - 1 } ) .
$$

Thus $\Delta _ { n } - h _ { n } = o _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$ and $w _ { n } = o _ { \mathbb { P } } ( n ^ { - 1 } )$ .

The atomic comparison holds on each fixed root-� window; tightness lets its radius increase after taking � to infinity. Together with the smooth linear representation, this gives in both regimes

$$
\left( \Delta _ { n } + S _ { n } / Q ^ { \prime \prime } \right) ^ { 2 } = o _ { \mathbb { P } } ( n ^ { - 1 } ) .
$$

Hence

$$
\sqrt { n } \Delta _ { n } = - \frac { \sqrt { n } S _ { n } } { Q ^ { \prime \prime } } + o _ { \mathbb { P } } ( 1 ) .
$$

The first term converges in distribution to $N ( 0 , \nu / ( Q ^ { \prime \prime } ) ^ { 2 } ) = N ( 0 , \sigma ^ { 2 } )$ , with $0 < \sigma ^ { 2 } < \infty$ as in (4.3). Since this normal distribution has no atom at zero, $\Delta _ { n } = \Theta _ { \mathbb { P } } ( n ^ { - 1 / 2 } )$

Uniform integrability. Use the annuli from the localization step. On the �-th annulus inside $\{ R _ { n } \leq r \}$ $n \Delta _ { n } ^ { 2 } \leq n R _ { n } ^ { 2 } \leq 4 ^ { j }$ , while its probability is at most $C _ { \eta } 2 ^ { - j ( 2 + \eta ) }$ . Thus the tail contribution from $j \geq J$ is bounded, uniformly in �, by

$$
C _ { \eta } \sum _ { j \ge J } 2 ^ { - j \eta } \longrightarrow 0 .
$$

For the escape event, set

$$
W _ { n } = \operatorname* { s u p } _ { x \in \boldsymbol { X } , | t | \leq M } | ( P _ { n } - P ) f _ { x , t } | .
$$

The bounded uniform law from Appendix B gives $\mathbb { E } [ W _ { n } ]  0$ . Replacing one observation changes $W _ { n }$ by at most $2 M / ( n ( 1 - \delta ) )$ . Bounded diferences therefore give an exponentially small probability that $W _ { n }$ exceeds any fixed positive constant, for all suficiently large �. On the compact joint parameter set the continuous population objective has the unique minimizing pair $( x ^ { * } , q ^ { * } )$ , so $\{ R _ { n } > r \}$ forces such a fixed deviation by the basic inequality. Since X is bounded, $\vec { \mathfrak { s } } [ n \Delta _ { n } ^ { 2 } \mathbf { 1 } _ { \left\{ R _ { n } > r \right\} } ] \to 0$ . Combining this with the annular tail bound proves uniform integrability of $\{ n \Delta _ { n } ^ { 2 } \}$ □

## C.1 Proof of Proposition 4.2 and the regret rates

Lemma C.2 gives the normal limit, uniform integrability, and the stated rate in probability.

The normal limit and uniform integrability give (3.6) with $r _ { n } = n ^ { - 1 / 2 }$ . For a fixed action neighborhood $A _ { n } = \left\{ \left| \Delta _ { n } \right| \leq r \right\}$ , uniform integrability implies

$$
n \mathbb { P } ( A _ { n } ^ { c } ) \leq r ^ { - 2 } \mathbb { E } [ n \Delta _ { n } ^ { 2 } \mathbf { 1 } _ { A _ { n } ^ { c } } ] \longrightarrow 0 .
$$

Bounded loss gives $| Q _ { F } ( x ) - Q _ { F } ( x ^ { * } ) | \leq 2 M$ and $| q _ { F } ( x ) | \le M$ . The escape remainders in Lemma A.3 are therefore $O ( \mathbb { P } ( A _ { n } ^ { c } ) ) = o ( n ^ { - 1 } )$ . The proofs of Theorems 3.4 and 3.5 apply with these bounds replacing the global Lipschitz estimates for those remainders. They give $\mathrm { P S } , \mathrm { S R } = \Theta ( n ^ { - 1 } )$ in the smooth regime, and $\mathrm { P S } = \Theta ( n ^ { - 1 } ) , \mathrm { S R } = \Theta ( n ^ { - 1 / 2 } )$ in the atomic regime. The same replacement applies to the coeficient limits in Propositions 3.6 and 3.8 and hence to Corollary 3.7 in the smooth regime.

## C.2 Finite-sample computation for the smooth examples

Table 3 uses the truncated distributions of an exponential outcome conditional on $\xi \le 1 2$ and a lognormal outcome conditional on $\xi \le 2 0 0 0$ , with the parameters in Section 4.3. There is no mass at the truncation bound. $\mathrm { A t } \ \delta = 0 . 9 5$ , conditioning the lognormal distribution on $\xi \le 2 0 0 0$ changes � from approximately 428.07 for the untruncated distribution in Table 1 to 418.42 in Table 3. For squared loss on $[ 0 , B ]$ bounded outcomes ensure bounded loss and a bounded action-Lipschitz envelope. The objective $Q _ { F }$ is strongly convex, since subtracting $x ^ { 2 } / 2$ from each loss leaves an afine function of � whose CVaR is convex. The computed oracles are $x ^ { * } \approx 1 . 9 7 1 5 1$ and $x ^ { * } \approx 3 0 . 8 1 1 3 4$ , respectively. Their critical outcome pairs $x ^ { * } \pm \sqrt { 2 q ^ { * } }$ are approximately (0.03048, 3.91254) and (0.04188, 61.58080), both strictly inside the respective

supports, with positive unequal densities. Proposition A.6 gives the local curvature and nonzero cutof slope;   
compactness and the smooth squared loss give the remaining hypotheses of Proposition 4.2.

For each outcome distribution and $n \in \{ 1 0 0 , 4 0 0 , 1 6 0 0 \}$ , generate $R = 1 2 { , } 0 0 0$ independent training samples and compute their SAA actions $x _ { 1 } ^ { \mathrm { S A A } } , \ldots , x _ { R } ^ { \mathrm { S A A } }$ . With $1 - \delta = 0 . 0 5 , ( 1 - \delta ) n$ is an integer, so $m = ( 1 - \delta ) n$ in the SAA formula and the empirical CVaR is the average of these � largest sample losses. For a sorted sample, these losses come from the two ends of the sample, so the objective is piecewise quadratic in �. On each piece, its minimizer is the mean of the selected tail outcomes projected onto that piece. Minimizing these quadratic pieces gives the SAA action. All actions are retained individually for risk evaluation. The accompanying computational supplement includes smooth\_saa.py, whose mc mode reproduces these calculations. It initializes one random generator per outcome distribution, with seeds 20261025 for the exponential and 20261026 for the lognormal, and generates the three rows in increasing order of �.

Population evaluation uses analytic truncated moments. For $t ~ \geq ~ 0$ , put $d \ : = \ : \sqrt { 2 t }$ and $m _ { j } ( x , d ) =$ $\mathbb { E } [ \xi ^ { j } \mathbf { 1 } _ { \left\{ | \xi - x | > d \right\} } ] , j = 0 , 1 , 2$ . Then

$$
\phi _ { F } ( x , t ) = t + \frac { m _ { 2 } ( x , d ) - 2 x m _ { 1 } ( x , d ) + ( x ^ { 2 } - 2 t ) m _ { 0 } ( x , d ) } { 2 ( 1 - \delta ) } .
$$

The conditional cutof solves $\mathbb { P } ( | \xi - x | \leq \sqrt { 2 q _ { F } ( x ) } ) = \delta$ , and $Q _ { F } ( x ) = \phi _ { F } ( x , q _ { F } ( x ) )$ . The common threshold b� minimizes $R ^ { - 1 } \textstyle \sum _ { r = 1 } ^ { R } \phi _ { F } ( x _ { r } ^ { \mathrm { S A A } } , t )$ and solves

$$
\frac 1 R \sum _ { r = 1 } ^ { R } \mathbb { P } _ { \xi } \left( | \xi - x _ { r } ^ { \mathrm { S A A } } | \le \sqrt { 2 t } \right) = \delta ,
$$

where each probability is under the truncated outcome distribution with $x _ { r } ^ { \mathrm { S A A } }$ held fixed. With $Q ^ { * } = Q _ { F } ( x ^ { * } )$ , define

$$
a _ { r } = Q _ { F } ( x _ { r } ^ { \mathrm { S A A } } ) - Q ^ { * } , \qquad b _ { r } = \phi _ { F } ( x _ { r } ^ { \mathrm { S A A } } , \widehat t ) - Q _ { F } ( x _ { r } ^ { \mathrm { S A A } } ) .
$$

The Monte Carlo estimates are $\begin{array} { r } { \widehat { \mathrm { P S } } = R ^ { - 1 } \sum _ { r } a _ { r } , \widehat { \mathrm { S R } } = R ^ { - 1 } \sum _ { r } b _ { r } } \end{array}$ , and ${ \widehat { \rho } } = { \widehat { \mathrm { S R } } } / { \widehat { \mathrm { P S } } }$

The standard errors quantify only variation across the simulated training samples. The envelope derivative at the minimizing common threshold and the delta method give the first-order ratio influence values

$$
w _ { r } = \frac { b _ { r } - \widehat { \rho } a _ { r } } { \widehat { \mathrm { P S } } } , \qquad \sum _ { r = 1 } ^ { R } w _ { r } = 0 .
$$

We report the approximate Monte Carlo standard errors

$$
\operatorname { s e } _ { \mathrm { M C } } ( { \widehat { \rho } } ) = \left( { \frac { \sum _ { r = 1 } ^ { R } w _ { r } ^ { 2 } } { R ( R - 1 ) } } \right) ^ { 1 / 2 } , \qquad \operatorname { s e } _ { \mathrm { M C } } ( { \mathrm { s h a r e } } ) = { \frac { \operatorname { s e } _ { \mathrm { M C } } ( { \widehat { \rho } } ) } { ( 1 + { \widehat { \rho } } ) ^ { 2 } } } .
$$

Share standard errors are multiplied by 100 in the table to express them in percentage points.

Minimizing over the common threshold using the same replications makes $\widehat { \mathrm { S R } }$ downward biased. The supplement’s bias\_diagnostic.py estimates the leading term of a second-order expansion of this optimization bias. Across the six rows, its estimated magnitude is $0 . 3 \% { - } 1 . 3 \%$ of the Monte Carlo standard error of $\widehat { \mathrm { S R } }$

## C.3 Finite-sample calculations for the uniform newsvendor

For $\xi \sim U [ 0 , 1 ] , c ( x , \xi ) = x / 2 - \operatorname* { m i n } ( x , \xi )$ , and $1 - \delta = 0 . 1$ , direct integration gives

$$
\begin{array} { r } { Q _ { F } ( x ) = \left\{ \begin{array} { l l } { 5 x ^ { 2 } - x / 2 , } & { 0 \leq x \leq ( 1 - \delta ) , } \\ { x / 2 - ( 1 - \delta ) / 2 , } & { ( 1 - \delta ) \leq x \leq 1 . } \end{array} \right. } \end{array}
$$

Hence $x _ { F } ^ { * } = 0 . 0 5 , Q _ { F } ( x _ { F } ^ { * } ) = - 0 . 0 1 2 5$ , and $Q _ { F } ^ { \prime \prime } ( x _ { F } ^ { * } ) = 1 0$ . Near the oracle, the cutof is $\ell ( x ) = - x / 2$ , with atom mass 1 − �. At the oracle, $p = 0 . 9 5$ and $\tau = 0 . 9 / 0 . 9 5 = 1 8 / 1 9$ . The continuous demand distribution gives the isolation limits. The action set is compact, the loss is bounded and continuous, its slopes have magnitude $1 / 2 .$ , and its action derivative exists almost surely at the oracle. These facts verify the assumptions of Proposition 4.2.

The one-sided derivatives of the empirical objective give the quantile optimality condition ${ \hat { F } } _ { n } ( x - ) \leq$ $( 1 - \delta ) / 2 \leq { \hat { F } } _ { n } ( x )$ . Choosing the smallest such order gives $x ^ { \mathrm { S A A } } = \xi _ { ( k ) } , k = \lceil n / 2 0 \rceil$ , including a consistent selection when the empirical minimizer is not unique. Its Beta distribution yields all calculations in Table 4. Define the truncated moments

$$
M _ { j } ( h ) = \frac { ( k ) _ { j } } { ( n + 1 ) _ { j } } I _ { h } ( k + j , n + 1 - k ) , \qquad T _ { j } ( h ) = \frac { ( k ) _ { j } } { ( n + 1 ) _ { j } } - M _ { j } ( h ) ,
$$

where $( a ) _ { j }$ is the rising factorial, $( a ) _ { 0 } = 1$ , and $I _ { h }$ is the regularized incomplete Beta function. These are $\mathbb { E } [ ( x ^ { \mathrm { S A A } } ) ^ { j } \mathbf { 1 } _ { \{ x ^ { \mathrm { S A A } } \leq h \} } ]$ and $\mathbb { E } [ ( x ^ { \mathrm { S A A } } ) ^ { j } \mathbf { 1 } _ { \{ x ^ { \mathrm { S A A } } > h \} } ]$ . In particular,

$$
\begin{array} { r } { \mathbb { E } Q _ { F } ( x ^ { \mathrm { { S A A } } } ) = 5 M _ { 2 } ( 1 - \delta ) - \frac { 1 } { 2 } M _ { 1 } ( 1 - \delta ) + \frac { 1 } { 2 } T _ { 1 } ( 1 - \delta ) - \frac { 1 - \delta } { 2 } T _ { 0 } ( 1 - \delta ) . } \end{array}
$$

For the sample sizes in Table 4, the combined-loss quantile $t _ { n } ^ { \mathrm { c o m b } }$ is in $( - 1 / 2 , 0 )$ . Conditional on $x ^ { \mathrm { S A A } } = x$ and for � in this interval, the loss CDF is zero when $x < - 2 t$ and is $1 - x / 2 + t$ when $x > - 2 t$ . Thus $t _ { n } ^ { \mathrm { c o m b } }$ solves

$$
\begin{array} { r } { ( 1 + t ) T _ { 0 } ( - 2 t ) - \frac { 1 } { 2 } T _ { 1 } ( - 2 t ) = 0 . 9 . } \end{array}
$$

For $h = - 2 t$ , integrating the conditional positive part gives

$$
\begin{array} { r l } & { J _ { n } ( t ) : = \mathbb { E } \big [ \big ( c ( x ^ { \mathrm { S A A } } , \xi ) - t ) ^ { + } \big ] } \\ & { \qquad = \frac { 1 } { 2 } M _ { 2 } ( h ) - \frac { 1 } { 2 } M _ { 1 } ( h ) - t M _ { 0 } ( h ) } \\ & { \qquad + \frac { 1 } { 8 } T _ { 2 } ( h ) - \frac { 1 } { 2 } t T _ { 1 } ( h ) + \frac { 1 } { 2 } t ^ { 2 } T _ { 0 } ( h ) . } \end{array}
$$

The combined-risk value is $t _ { n } ^ { \mathrm { c o m b } } + J _ { n } ( t _ { n } ^ { \mathrm { c o m b } } ) / ( 1 - \delta )$ . Subtracting $\mathbb { E } Q _ { F } ( x ^ { \mathrm { S A A } } )$ gives SR, while $\mathrm { P S } \ =$ $\mathbb { E } Q _ { F } ( x ^ { \mathrm { S A A } } ) + 0 . 0 1 2 5$ . Numerical evaluation of these special functions and the scalar quantile equation produces the reported finite-sample values.

Finally, the score in (4.3) is $\psi _ { F } ( \xi ) = - 1 / 2 + 1 0 \mathbf { 1 } _ { \{ \xi < 0 . 0 5 \} }$ . It takes the value 9.5 with probability 0.05 and $- 1 / 2$ otherwise, so $\mathrm { V a r } ( \psi _ { F } ( \xi ) ) = 4 . 7 5$ . Since $Q _ { F } ^ { \prime \prime } ( x _ { F } ^ { * } ) = 1 0$

$$
\sigma ^ { 2 } = \frac { \mathrm { V a r } ( \psi _ { F } ( \xi ) ) } { Q _ { F } ^ { \prime \prime } ( x _ { F } ^ { * } ) ^ { 2 } } = 0 . 0 5 ( 0 . 9 5 ) .
$$

Let $\Phi$ and $\varphi$ denote the standard normal distribution function and density. Then

$$
n \mathrm { P S }  0 . 2 3 7 5 , \qquad \sqrt { n } \mathrm { S R }  \frac { 0 . 9 5 ( 1 / 2 ) \sqrt { 0 . 0 5 ( 0 . 9 5 ) } \varphi ( \Phi ^ { - 1 } ( 1 8 / 1 9 ) ) } { 0 . 1 } \approx 0 . 1 1 1 2 1 6 8 .
$$

The limiting ratio is $\rho _ { n } / \sqrt { n }  0 . 1 1 1 2 1 6 8 / 0 . 2 3 7 5 \approx 0 . 4 6 8 2 8 1 3$ , and the premium share tends to one.