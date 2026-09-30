# Corruption-Robust Sparse Linear Contextual Bandits with Knapsack Constraints

Yige Wang<sup>†</sup>, Hanyang Li<sup>†</sup>, Yiming Zong<sup>†</sup>, Wanteng Ma<sup>‡∗</sup>, Jiashuo Jiang<sup>†∗</sup>

† Department of Industrial Engineering and Decision Analytics, The Hong Kong University of Science and Technology ‡ Department of Statistics and Data Science, The Wharton School, University of Pennsylvania

Abstract. We study sparse linear contextual bandits with knapsack constraints under joint reward and consumption corruption. Consumption corruption creates a challenge beyond corrupted rewards: it affects not only statistical estimates, but also the recorded budget, resource prices, and stopping decisions that govern future allocation. We develop Robust Optimistic Primal–Dual (ROPD), an estimator-modular framework that combines corruption-aware confidence widths with online resource prices and a budget-safety rule. With concrete sparse implementation, ROPD achieves regret against a clean population-LP benchmark of $\widetilde { \cal O } ( T ^ { 2 / 3 } + \Gamma T ^ { 1 / 3 } )$ under forced exploration and population-design coverage, and Oe( T + Γ) under on-policy realized-design coverage, for a supplied valid corruption bound Γ under the stated proportional-budget scaling and fixed model/design parameters. When the corruption level is unknown, Shared-Grid adapts confidence radii around common point estimates fitted to a single realized history, incurring explicit initialization and master-comparison costs; its sharper on-policy guarantee additionally requires recommendation coverage. Both methods preserve observed budgets on every realization and bound clean resource violation by cumulative consumption corruption. These results connect corruption-robust sparse estimation with resource accounting, pricing, and stopping in high-dimensional online allocation.

Key words: Contextual Bandits with Knapsacks; Sparse Linear Bandits; Adversarial Corruption; Online Resource Allocation; Primal–Dual Learning

## 1. Introduction

Contextual bandits with knapsack constraints (CBwK) couple learning with resource allocation: after observing a context, a learner chooses an action whose reward must be learned while its consumption depletes finite budgets (Badanidiyuru et al. 2018, Agrawal and Devanur 2016, Agrawal et al. 2016). In highdimensional applications, only a small subset of features may govern reward and consumption, motivating sparse action-specific models (Bastani and Bayati 2020, Hao et al. 2020, Ma et al. 2024).

Corruption of consumption feedback creates a difficulty beyond corrupted rewards. A corrupted reward changes what the learner infers about an action, whereas corrupted consumption also changes the state used to allocate future resources—consumption estimate, recorded budget, resource prices, and potentially stopping time. For example, an advertisement that is truly expensive but repeatedly recorded as inexpensive can be favored while the observed budget account still appears feasible.

A robust estimator alone therefore does not resolve the allocation problem. The analysis must control how corrupted observations affect later predictions, relate observed consumption to the latent clean resource use, and account for reward lost when corrupted resource information changes stopping. In the sparse setting, these guarantees also depend on whether the contexts actually collected for each action are informative.

Table 1 Guarantee regimes and leading terms when corruption is absent.
<table><tr><td>Corruption information Sampling</td><td></td><td>Coverage</td><td>Corruption-free term</td></tr><tr><td>Bound  $\Gamma \geq C _ { Z }$ </td><td>Forced</td><td>Population design</td><td> $\widetilde { \cal O } ( T ^ { 2 / 3 } )$ </td></tr><tr><td>Bound  $\Gamma \geq C _ { Z }$ </td><td>On-policy</td><td>Realized design</td><td> $\widetilde { O } ( \sqrt { T } )$ </td></tr><tr><td>Not supplied</td><td>Forced</td><td>Population design</td><td> $\widetilde { O } ( T ^ { 2 / 3 } ) + \mathrm { R e g } _ { \mathrm { m s t } } ( T )$ </td></tr><tr><td>Not supplied</td><td>On-policy</td><td>Realized design and recommendation coverage</td><td> $\widetilde { O } ( \sqrt { T } ) + \mathrm { R e g } _ { \mathrm { m s t } } ( T )$ </td></tr></table>

Orders assume proportional budgets, fixed model and coverage parameters, dominated initialization and burn-in, and negligible failure costs. $\mathrm { R e g } _ { \mathrm { m s t } } ( T )$ is the master-comparison cost; Corollaries 1 and 4 give the full bounds.

When the corruption level is unknown, candidates calibrated to different levels may recommend different actions even though feedback is observed only for the actions that are actually played.

We address these issues with Robust Optimistic Primal–Dual (ROPD), which combines corruption-aware prediction widths with online resource prices and a budget-safety rule. The allocation analysis is estimatormodular: it uses only pointwise confidence and cumulative uncertainty along the recommended actions. Norm-constrained Lasso provides one concrete sparse implementation. We study both forced exploration and on-policy learning, and for unknown corruption introduce Shared-Grid, which compares confidence calibrations around common point estimates fitted to one realized history.

## 1.1. Main Contributions

1. Joint statistical and resource effects of corruption. We measure joint reward–consumption corruption through the priced budget $C _ { Z }$ and separate its two roles: contaminated samples affect future prediction, while corrupted consumption directly changes resource accounting, price updates, and stopping. This decomposition yields regret guarantees together with exact observed-budget feasibility and an explicit bound on clean resource violation.

2. Estimator-modular robust allocation with two data routes. Given a valid bound $\Gamma \geq C _ { Z }$ , ROPD combines optimistic predictions with online resource prices. Its allocation theorem requires only pointwise confidence and cumulative recommendation-width control; norm-constrained Lasso is one concrete implementation. Forced exploration under population design gives the $\widetilde { \cal O } ( T ^ { 2 / 3 } )$ corruption-free regime, while realized on-policy design gives $\widetilde { O } ( \sqrt { T } )$ under the conditions summarized in Table 1.

3. Adaptation to unknown corruption on shared data. Shared-Grid compares corruption calibrations around common point estimates fitted to the single realized history, rather than analyzing candidates on counterfactual estimator histories. Its guarantee retains initialization and master-comparison costs explicitly; the sharper on-policy result additionally requires coverage of candidate recommendations by realized plays. Relation to closest work. Clean high-dimensional CBwK already combines sparse estimation with primal– dual resource control (Ma et al. 2024), while corruption-robust linear and contextual bandits primarily address corrupted reward feedback without knapsack state (He et al. 2022, Liu et al. 2024). Thus the new difficulty here is not sparse estimation or dual pricing in isolation: corrupted consumption simultaneously affects prediction, resource-price path, recorded budget, and stopping. We analyze these effects against a clean population-LP benchmark and, when the corruption level is unknown, adapt on a single realized estimator history. Section 1.2 gives the broader comparison.

## 1.2. Related Work

Contextual bandits with knapsacks and high-dimensional allocation. The CBwK framework joins bandit learning with finite resource constraints (Badanidiyuru et al. 2018). Agrawal and Devanur (2016) study the closest clean linear contextual model: expected rewards and resource consumptions are linear in observed features, and confidence sets are combined with online resource prices to control the knapsacks. Agrawa et al. (2016) develop a more general oracle-based contextual framework, and primal–dual ideas also connect this literature to online packing and dynamic resource allocation (Agrawal and Devanur 2014, Devanur and Hayes 2009, Agrawal et al. 2014). Our known-corruption method keeps the basic confidence-plus-price structure, but the feedback used by the estimator, the dual update, and the stopping rule can now differ from the clean process used by the benchmark. Our analysis accounts for the difference between clean and observed consumption in the price updates and the reward lost after stopping. Adversarial and non-stationary BwK study changing reward and consumption environments (Immorlica et al. 2022, Liu et al. 2022), while instance-dependent BwK analyses go beyond worst-case guarantees under clean feedback (Sankararaman and Slivkins 2021). Our model instead keeps a latent uncorrupted linear contextual model with i.i.d. contexts and studies corrupted observations of that process, so the goal is to recover performance against a clean population-LP benchmark.

Sparse contextual bandits and clean high-dimensional CBwK. Sparse contextual bandit methods use regularization, forced sampling, Thompson sampling, or hard thresholding so regret depends on a small relevant support (Bastani and Bayati 2020, Hao et al. 2020, Ren and Zhou 2024, Chakraborty et al. 2023). The closest paper to our sparse allocation setting is Ma et al. (2024), which combines an online hard-thresholding estimator with a primal–dual CBwK method and obtains logarithmic dependence on the ambient feature dimension under clean feedback. Their sharper clean regime and our zero-corruption on-policy result both have $\widetilde { O } ( \sqrt { T } )$ horizon dependence, with different action, sparsity, curvature, norm, and resource-price factors. This comparison concerns the horizon exponent, not identical dimension dependence. Our setting adds corrupted reward and consumption observations, a direct resource-accounting term, a sequential charge for the later effect of corrupted action-specific samples, and adaptation when the corruption level is unknown. The retained-design RSC condition in our sharper result plays the familiar statistical role of ensuring that action-specific data identify sparse parameters. The additional Shared-Grid recommendation-coverage condition is different: it controls how often candidate recommendations are supported by realized plays.

Corruption-robust bandits. Corruption-robust bandit work first showed how regret can degrade with an adversarial corruption budget in stochastic multi-armed bandits (Lykouris et al. 2018, Gupta et al. 2019).

Later work studies linear and contextual models. Ding et al. (2022) allow attacked rewards and contexts without requiring the attack budget. For a known played-action reward-corruption budget C, He et al. (2022) obtain ${ \widetilde { O } } ( d { \sqrt { T } } + d C )$ and also study adaptation without a supplied bound. For fixed-parameter linear bandits, Liu et al. (2024) distinguish C from the worst-action budget $C _ { \infty }$ and establish $\widetilde { O } ( d \sqrt { T } + \operatorname* { m i n } \{ d C , \sqrt { d } C _ { \infty } \} )$ with matching lower bounds under the corresponding corruption information. These results concern reward corruption without knapsack constraints and do not by themselves establish optimal dependence on our joint reward–consumption budget. Other work studies linear optimization, Lipschitz bandits, generalized linear models, and explicit poisoning attacks (Li et al. 2019, Kang et al. 2023, Yu et al. 2025, Liu and Shroff 2019). These papers develop robust confidence sets, weighting, elimination, and attack-adaptive exploration for bandit objectives centered on prediction and action selection. Our setting adds a resource signal that changes both statistical estimates and the state of the allocation system. The Z-scaled budget measures corruption in the action score, and the proof also controls observed budget account, dual-price path, and stopping time.

Robust sparse estimation and sequential prediction. Robust sparse regression studies parameter recovery when some responses or samples are corrupted (Chen et al. 2013, Bhatia et al. 2015, Balakrishnan et al. 2017, Diakonikolas et al. 2019, Prasad et al. 2020, Liu et al. 2020). Restricted-eigenvalue and sparse covariance conditions explain when high-dimensional parameters can be identified (Bickel et al. 2009, Raskutti et al. 2010, Negahban et al. 2012, Wainwright 2019, Bühlmann and Van De Geer 2011). These tools provide the statistical base of our estimator module, while an online allocation proof additionally requires the sum of reward and consumption widths along an adaptive action path. We prove this sequential bound by matching estimator sample set to its coverage event and by charging each corrupted observation through the later inverse action counts in which it still appears. Our concrete analysis uses norm-constrained Lasso with candidate-independent regularization. An enlarged-cone argument and an RSC tolerance yield explicit corruption-aware confidence, with a norm bound handling large corruption. The resulting allocation theorems transfer to any sparse estimator that supplies the same predictable pointwise and cumulative-width guarantees.

Unknown corruption and bandit model selection. Corralling and model-selection methods combine candidates or policy classes under partial feedback (Agarwal et al. 2017, Foster et al. 2019, Pacchiano et al. 2020, 2022, Cutkosky et al. 2021, Krishnamurthy and Athey 2021, Ghosh et al. 2021). Agarwal et al. (2017) show that a master must avoid starving a candidate of feedback. Foster et al. (2019) adapt to an unknown contextual model class, and Pacchiano et al. (2022) combine model selection with guarantees for fixed-mean and adversarial rewards. Model selection has also been used for corruption-robust reinforcement learning (Wei et al. 2022). Unknown corruption in sparse CBwK adds a data-design problem. Two independently updated corruption candidates would choose different actions and would need different action-specific context histories, so the confidence set of an inactive candidate can depend on data that the learner never collected. The shared-estimator grid avoids reliance on these unobserved histories: every candidate uses the same realized point estimate, and the master compares only their current recommendations. The remaining candidate-coverage condition in the exploration-free result states when these recommendations receive enough realized plays for their shared confidence widths to be summed.

Our paper combines primal–dual resource control, corruption-robust learning, and sparse action-specific estimation in a setting where resource-consumption feedback is itself corrupted. This requires a relation between clean and observed resource use, stopping control through resource prices, sequential tracking of corrupted retained samples, and shared-estimator adaptation on one realized design.

## 2. Model, Feedback, and Benchmark

## 2.1. Interaction and Feedback

The learner interacts with the environment for T rounds with m resource budgets, $B = ( B _ { 1 } , \ldots , B _ { m } ) ^ { \top }$ with $B _ { i } > 0$ for every resource $i \in [ m ] = \{ 1 , . . . , m \}$ . At round $t ,$ it observes a context $x _ { t } \in \mathbb { R } ^ { d }$ , a vector of features available before choosing an action. It then selects an action $a _ { t }$ from K alternatives, indexed by $\mathcal { A } = [ K ] = \{ 1 , . . . , K \}$ , or chooses the null action null, which gives zero reward and uses no resources. We write $\mathcal { A } _ { 0 } = \mathcal { A } \cup \{ \mathtt { n u l 1 } \}$ for the full action set and call the actions in A non-null actions. The learner receives reward and consumption feedback only for the action it selects; this feedback may be corrupted.

Clean outcomes and observed feedback. We distinguish three quantities: the expected outcome before corruption, its noisy realization, and the feedback observed after corruption. We use clean to mean before adversarial corruption, not without stochastic noise. For action a and context x, let $r ^ { 0 } ( a , x )$ be the clean expected reward and $b _ { i } ^ { 0 } ( a , x )$ the clean expected consumption of resource i. These clean conditional means are linear in the context, with unknown coefficient vectors $\mu _ { a } ^ { \star } , w _ { a , i } ^ { \star } \in \mathbb { R } ^ { d }$

$$
r ^ { 0 } ( a , x ) = \langle \mu _ { a } ^ { \star } , x \rangle , \quad b _ { i } ^ { 0 } ( a , x ) = \langle w _ { a , i } ^ { \star } , x \rangle , \quad i \in [ m ] .\tag{1}
$$

Write $b ^ { 0 } ( a , x ) = ( b _ { 1 } ^ { 0 } ( a , x ) , \ldots , b _ { m } ^ { 0 } ( a , x ) ) ^ { \top }$ for the consumption-mean vector.

The clean realizations $r _ { t } ^ { \mathrm { c l } } ( a )$ ) and b<sup>cl</sup> (a) $b _ { t , i } ^ { \mathrm { c l } } ( a )$ equal these means plus ordinary stochastic noise, specified in Assumption 1 below. The adversary adds raw perturbations $\xi _ { t } ^ { r } ( a )$ and $\xi _ { t , i } ^ { b } ( a )$ to these realizations. The resulting feedback is clipped to [0, 1]:

$$
\begin{array} { r } { r _ { t } ( a ) = \mathrm { c l i p } ( r _ { t } ^ { \mathrm { c l } } ( a ) + \xi _ { t } ^ { r } ( a ) , 0 , 1 ) , \quad b _ { t , i } ( a ) = \mathrm { c l i p } ( b _ { t , i } ^ { \mathrm { c l } } ( a ) + \xi _ { t , i } ^ { b } ( a ) , 0 , 1 ) , } \end{array}
$$

where $\mathrm { c l i p } ( z , 0 , 1 ) = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , z \} \}$ truncates values outside this interval. The actual changes after clipping, rather than the raw perturbations, enter our corruption measures:

$$
\begin{array} { r } { c _ { t } ^ { r } ( a ) = r _ { t } ( a ) - r _ { t } ^ { \mathrm { c l } } ( a ) , \quad c _ { t , i } ^ { b } ( a ) = b _ { t , i } ( a ) - b _ { t , i } ^ { \mathrm { c l } } ( a ) . } \end{array}\tag{2}
$$

The vectors $b _ { t } ^ { \mathrm { c l } } ( a ) , b _ { t } ( a )$ , and $c _ { t } ^ { b } ( a )$ collect the corresponding m resource coordinates. Because both clean and observed outcomes lie in [0, 1], $| c _ { t } ^ { r } ( a ) | \leq 1$ and $\| c _ { t } ^ { b } ( a ) \| _ { \infty } \leq 1$ , where $\| v \| _ { \infty }$ is the largest absolute coordinate of v. For the null action, all clean outcomes, observed outcomes, and corruption terms are zero.

These variables describe each action’s potential outcomes: the outcomes it would produce if selected. They are defined for all actions to permit comparisons, but only $r _ { t } ( a _ { t } )$ and $b _ { t } { \big ( } a _ { t } { \big ) }$ are observed (Table 2). The adversary changes clean realizations, not their underlying means, and its raw perturbations have no probability model. Their total effect is measured by the corruption budget in Section 2.2. Clipping bounds feedback; it does not identify corrupted observations.

Table 2 Three levels of reward and resource-consumption feedback.
<table><tr><td>Quantity</td><td>Meaning</td><td>Observed by the learner?</td></tr><tr><td> $r ^ { 0 } ( a , x ) , b ^ { 0 } ( a , x )$ </td><td>clean conditional means</td><td>No</td></tr><tr><td> $\boldsymbol { r } _ { t } ^ { \mathrm { c l } } ( a ) , b _ { t } ^ { \mathrm { c l } } ( a )$ </td><td>clean noisy realizations before corruption No</td><td></td></tr><tr><td> $r _ { t } ( a _ { t } ) , b _ { t } ( a _ { t } )$ </td><td>clipped feedback after corruption</td><td>Yes, for the chosen action</td></tr></table>

Information and within-round timing. For analysis, let $\mathcal { F } _ { t - 1 }$ be the full history before round t: all earlier contexts, actions, clean potential outcomes, corruption, and learner randomization. The learner’s own history contains earlier contexts, selected actions, observed feedback, and its past randomization, but not latent clean outcomes or corruption. Its decisions use only this observed history, the current context, and fresh independent randomness.

Before observing $x _ { t } ,$ the learner applies the budget-safety check in Section 2.3. A round is active if this check passes; otherwise, it selects null from that round onward. On an active round: (i) the learner observes $x _ { t } ;$ (ii) conditional on $\mathcal { F } _ { t - 1 }$ and $x _ { t } .$ , the environment generates the clean potential outcomes and fixes the corruption for every action; (iii) the learner draws $a _ { t }$ using fresh randomness that, conditional on $\mathcal { F } _ { t - 1 }$ and $x _ { t } .$ , is independent of these potential outcomes and corruption, and observes only $r _ { t } ( a _ { t } )$ and $b _ { t } { \big ( } a _ { t } { \big ) }$ ; (iv) the learner records the observed resource use and updates its estimates and other algorithm variables.

Thus, corruption is non-anticipating: it may depend on $\mathcal { F } _ { t - 1 }$ , the current context, and the generated clean potential outcomes, but not on the learner’s fresh round-t random seed. A sample is retained when it is used to fit the estimates in Section 3.1. Both the action and the sample’s eligibility for retention are determined before the current feedback is revealed.

Distributional and sparsity assumptions. The following conditions use the full pre-round history $\mathcal { F } _ { t - 1 }$ :

ASSUMPTION 1 (Sparse linear contextual model).

(i) Contexts. Given $\mathcal { F } _ { t - 1 } , ~ \boldsymbol { x } _ { t }$ has distribution D and satisfies $\| x _ { t } \| _ { 2 } \leq 1$

(ii) Clean outcomes. For every action and resource, $r _ { t } ^ { \mathrm { c l } } ( a ) = r ^ { 0 } ( a , x _ { t } ) + \eta _ { t } ^ { r } ( a )$ and $b _ { t , i } ^ { \mathrm { c l } } ( a ) = b _ { i } ^ { 0 } ( a , x _ { t } ) +$ $\eta _ { t , i } ^ { b } ( a )$ , where $\mathbb { E } [ \eta _ { t } ^ { r } ( a ) \vert \mathcal { F } _ { t - 1 } , x _ { t } ] = 0$ and $\mathbb { E } [ \eta _ { t , i } ^ { b } ( a ) \vert \mathcal { F } _ { t - 1 } , x _ { t } ] = 0 .$ . The reward noise $\eta _ { t } ^ { r }$ and consumption noise $\eta _ { t , i } ^ { b }$ are conditionally sub-Gaussian with parameters at most $\sigma _ { r }$ and $\sigma _ { b } ,$ , respectively. Clean rewards and every clean consumption coordinate lie in [0, 1].

(iii) Sparse coefficients. For every a, i, $\| \mu _ { a } ^ { \star } \| _ { 0 } , \| w _ { a , i } ^ { \star } \| _ { 0 } \leq s _ { 0 }$ and $\| \mu _ { a } ^ { \star } \| _ { 2 } , \| w _ { a , i } ^ { \star } \| _ { 2 } \leq R$ . Here $\Vert \boldsymbol { v } \Vert _ { 0 }$ counts the nonzero coordinates $o f v , s _ { 0 }$ is the true sparsity level, and $R > 0$ bounds the Euclidean norm of each coefficient vector.

Clean noise has mean zero given the full history and current context, and the sub-Gaussian parameters control its tail fluctuations. With the timing above, this supports concentration of retained clean-noise sums and unbiased importance-weighted comparisons in the unknown-corruption analysis. Conditioning only on the learner’s observations would not suffice for the martingale arguments. The estimator uses a supplied sparsity upper bound $s \geq \operatorname* { m a x } \{ 1 , s _ { 0 } \}$ for confidence and numerical-precision calibration; it does not constrain the Lasso fit to have at most s nonzero coefficients.

## 2.2. Benchmark, Resource Prices, and Effective Corruption

Clean benchmark and regret. We evaluate the learner using clean rewards rather than corrupted feedback. The benchmark is a population linear program $( L P )$ based on the context distribution and clean conditional means. Let $y ( a \mid x )$ be the probability of choosing non-null action a given context x, with remaining probability assigned to null. This rule can depend on the context but is the same across rounds. Writing $\mathbb { E } _ { x }$ for expectation over $x \sim \mathcal { D }$ , the LP maximizes expected reward subject to expected resource budgets:

$$
\begin{array} { r l } { V ^ { \mathrm { U B } } : = \displaystyle \operatorname* { m a x } _ { y } } & { T \mathbb { E } _ { x } \left[ \displaystyle \sum _ { a \in \mathcal { A } } r ^ { 0 } ( a , x ) y ( a \mid x ) \right] } \\ { \mathrm { s . t . } } & { T \mathbb { E } _ { x } \left[ \displaystyle \sum _ { a \in \mathcal { A } } b _ { i } ^ { 0 } ( a , x ) y ( a \mid x ) \right] \leq B _ { i } , \quad i \in [ m ] , } \\ & { y ( a \mid x ) \geq 0 , \quad \displaystyle \sum _ { a \in \mathcal { A } } y ( a \mid x ) \leq 1 , \quad \forall x . } \end{array}\tag{3}
$$

Under the i.i.d. model, this value upper-bounds the clean expected reward attainable under the population resource constraints (Agrawal and Devanur 2016, Agrawal et al. 2016). For the analysis, fix a measurable optimizer $y ^ { \star }$ before interaction. The learner does not know or solve this LP. Its clean expected regret is

$$
\mathrm { R e g } ( T ) : = V ^ { \mathrm { U B } } - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } r ^ { 0 } ( a _ { t } , x _ { t } ) \right] ,\tag{4}
$$

where the expectation includes contexts, clean noise, learner randomness, and any randomized corruption Resource prices. To compare reward with resource use, assign nonnegative weights $\lambda \in \Lambda : = \{ \lambda \in \mathbb { R } _ { + } ^ { m }$ $\| \lambda \| _ { 1 } \leq 1 \}$ and choose an overall scale $Z > 0$ . The weights have total mass at most one, and $Z \lambda _ { i }$ is the price of one unit of resource i in reward units. The clean Lagrangian score is expected reward minus priced expected consumption:

$$
F ^ { 0 } ( a , x ; \lambda ) = r ^ { 0 } ( a , x ) - Z \lambda ^ { \top } b ^ { 0 } ( a , x ) .
$$

ASSUMPTION 2 (Lagrangian scale). Choose a positive scale $Z > 0$ satisfying $Z \geq V ^ { \mathrm { U B } } / B _ { \mathrm { m i n } }$ , where $B _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i } B _ { i }$

The choice $Z = T / B _ { \mathrm { m i n } }$ is computable from the horizon and budgets and satisfies Assumption 2 because $V ^ { \mathrm { U B } } \leq T$ . This scale ensures that, in the stopping argument, depletion of any resource carries enough dual value to offset the remaining benchmark reward. Under proportional budgets, $B _ { \mathrm { m i n } } = \Theta ( T )$ and $B _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i } B _ { i } = O ( T )$ , this choice gives $Z = O ( 1 )$ . Our formal bounds nevertheless keep the dependence on these quantities explicit.

Effective corruption. Reward corruption changes the score directly; consumption corruption changes its resource-cost term. For every $\lambda \in \Lambda$ , the resulting score change satisfies

$$
\begin{array} { r } { \left| c _ { t } ^ { r } ( a ) - Z \lambda ^ { \top } c _ { t } ^ { b } ( a ) \right| \leq | c _ { t } ^ { r } ( a ) | + Z \| c _ { t } ^ { b } ( a ) \| _ { \infty } . } \end{array}\tag{5}
$$

We therefore measure both types of corruption in the same units through the effective corruption budget

$$
C _ { Z } : = \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { a \in \cal { A } } \left\{ | c _ { t } ^ { r } ( a ) | + Z \| c _ { t } ^ { b } ( a ) \| _ { \infty } \right\} .\tag{6}
$$

The maximum covers all non-null actions before the fresh action draw on each round. In the knowncorruption setting, the learner receives a deterministic bound Γ satisfying $C _ { Z } \leq \Gamma$ on every admissible realization; in the unknown-corruption setting, no such bound is supplied. Knowing Γ does not identify corrupted observations. For later analysis, we also measure corruption along the actions actually played:

$$
C _ { Z } ( t ) : = \sum _ { \tau = 1 } ^ { t } \bigl ( | c _ { \tau } ^ { r } ( a _ { \tau } ) | + Z \| c _ { \tau } ^ { b } ( a _ { \tau } ) \| _ { \infty } \bigr ) , \quad C _ { b } ( t ) : = \sum _ { \tau = 1 } ^ { t } \| c _ { \tau } ^ { b } ( a _ { \tau } ) \| _ { \infty } .
$$

Here $C _ { Z } ( t )$ measures both types of corruption in score units, whereas $C _ { b } ( t )$ measures only consumption corruption. Both use the selected actions, not the worst actions as in $C _ { Z } ;$ in particular, $C _ { Z } ( t ) \leq C _ { Z }$

For estimation, let $S _ { a } ^ { \mathrm { e s t } } ( t )$ contain the earlier rounds $\tau < t$ when action a was selected and its feedback retained for fitting (Section 3.1). The corresponding reward and resource-i corruption totals are

$$
C _ { a , \mathrm { e s t } } ^ { r } ( t ) : = \sum _ { \tau \in S _ { a } ^ { \mathrm { e s t } } ( t ) } | c _ { \tau } ^ { r } ( a ) | , \quad C _ { a , i , \mathrm { e s t } } ^ { b } ( t ) : = \sum _ { \tau \in S _ { a } ^ { \mathrm { e s t } } ( t ) } | c _ { \tau , i } ^ { b } ( a ) | .
$$

All these corruption quantities are for analysis: the learner cannot compute them without the clean outcomes.

## 2.3. Observed and Clean Budget Feasibility

Because the learner records observed consumption, its budget account may differ from clean resource use. Choose an operating budget $0 < B ^ { \mathrm { o p } } \leq B$ , with coordinatewise inequalities. This is the resource limit used by the algorithm; a smaller operating budget reserves a buffer against consumption corruption. Let $\begin{array} { r } { U _ { t } = \sum _ { \tau < t } b _ { \tau } ( a _ { \tau } ) } \end{array}$ be the observed use before round t. A non-null action is allowed only if at least one unit remains in every operating budget, enough to cover any observed consumption on that round:

$$
U _ { t , i } \leq B _ { i } ^ { \mathrm { o p } } - 1 , \quad \forall i \in [ m ] .\tag{7}
$$

Otherwise the learner selects null from that round onward. The one-unit safety margin is already included in (7), so it is not subtracted again from $B ^ { \mathrm { o p } }$ . For $[ z ] _ { + } : = \operatorname* { m a x } \{ z , 0 \}$ , define the largest observed and clean budget excesses relative to the original budget vector $B \colon$

$$
\mathrm { V i o l } _ { \mathrm { o b s } } ( T ) : = \operatorname* { m a x } _ { i \in [ m ] } \Bigl [ \sum _ { t = 1 } ^ { T } b _ { t , i } ( a _ { t } ) - B _ { i } \Bigr ] _ { + } , \quad \mathrm { V i o l } _ { \mathrm { c l } } ( T ) : = \operatorname* { m a x } _ { i \in [ m ] } \Bigl [ \sum _ { t = 1 } ^ { T } b _ { t , i } ^ { \mathrm { c l } } ( a _ { t } ) - B _ { i } \Bigr ] _ { + } .
$$

Recall that $C _ { b } ( T )$ is total consumption corruption along the played actions. Below, $\Gamma _ { b }$ is an upper bound on this quantity, when available, and 1 is the all-ones vector. The guarantees hold on every realization.

PROPOSITION 1 (Three feasibility levels). Under (7),for every resource $i \mathrm { : }$

(i) $\textstyle \sum _ { t = 1 } ^ { T } b _ { t , i } ( a _ { t } ) \leq B _ { i } ^ { \mathrm { o p } }$ (exact observed feasibility);

(ii) $\begin{array} { r } { \sum _ { t = 1 } ^ { T } b _ { t , i } ^ { \mathrm { c l } } ( a _ { t } ) \leq B _ { i } ^ { \mathrm { o p } } + C _ { b } ( T ) } \end{array}$ (clean sample-path control);

(iii) $i f C _ { b } ( T ) \le \Gamma _ { b } , B _ { i } > \Gamma _ { b }$ for every i, and $B ^ { \mathrm { o p } } = B - \Gamma _ { b } { \bf 1 }$ , then $\begin{array} { r } { \sum _ { t = 1 } ^ { T } b _ { t , i } ^ { \mathrm { c l } } ( a _ { t } ) \leq B _ { i } } \end{array}$ (strict clean samplepath feasibility).

Proof. Before a non-null action, the rule in (7) and $b _ { t , i } ( a _ { t } ) \leq 1$ imply that the updated observed use is at most ${ \cal B } _ { i } ^ { \mathrm { o p } }$ . Null actions use no resources, which proves (i). Since $b _ { t , i } ^ { \mathrm { c l } } ( a _ { t } ) = b _ { t , i } ( a _ { t } ) - c _ { t , i } ^ { b } ( a _ { t } )$

$$
\sum _ { t = 1 } ^ { T } b _ { t , i } ^ { \mathrm { c l } } ( a _ { t } ) \leq \sum _ { t = 1 } ^ { T } b _ { t , i } ( a _ { t } ) + \sum _ { t = 1 } ^ { T } | c _ { t , i } ^ { b } ( a _ { t } ) | \leq B _ { i } ^ { \mathrm { o p } } + C _ { b } ( T ) ,
$$

which proves (ii). Substituting $B _ { i } ^ { \mathrm { o p } } = B _ { i } - \Gamma _ { b }$ proves (iii).

For the default choice $B ^ { \mathrm { o p } } = B$ , Proposition 1 gives

$$
\mathrm { V i o l } _ { \mathrm { o b s } } ( T ) = 0 , \quad \mathrm { V i o l } _ { \mathrm { c l } } ( T ) \leq C _ { b } ( T ) \leq C _ { z } / Z .
$$

The last inequality follows because the per-round maximum defining $C _ { Z }$ dominates $Z \| c _ { t } ^ { b } ( a _ { t } ) \| _ { \infty }$ . Hence $C _ { Z } \le \Gamma$ also certifies $\mathrm { V i o l } _ { \mathrm { c l } } ( T ) \le \Gamma / Z$ . A corruption buffer protects clean feasibility but leaves fewer resources for reward. Let $V ^ { \mathrm { U B } } ( B ^ { \prime } )$ denote the population-LP value with budget vector $B ^ { \prime }$ . To apply the regret guarantee with a reduced operating budget, first use that budget’s benchmark and a scale $Z \ge$ $V ^ { \mathrm { U B } } ( B ^ { \mathrm { o p } } ) / B _ { \mathrm { m i n } } ^ { \mathrm { o p } }$ , where $B _ { \mathrm { m i n } } ^ { \mathrm { o p } } : = \operatorname* { m i n } _ { i } B _ { i } ^ { \mathrm { o p } }$ ; the computable choice $Z = T / B _ { \mathrm { m i n } } ^ { \mathrm { o p } }$ is sufficient. All quantities that depend on $Z ,$ , including the corruption allowance and confidence radii, use this same scale. Comparison with the original budget then adds $V ^ { \mathrm { U B } } ( B ) - V ^ { \mathrm { U B } } ( B ^ { \mathrm { o p } } )$ ; for the buffered choice, $B ^ { \mathrm { o p } } = B - \Gamma _ { b } { \bf 1 }$

REMARK 1 (CONDITIONAL-MEAN FEASIBILITY). Proposition 1 controls clean realized consumptions $b _ { t , i } ^ { \mathrm { c l } } ( a _ { t } )$ , not their conditional means. To bound $\textstyle \sum _ { t } b _ { i } ^ { 0 } ( a _ { t } , x _ { t } )$ , one must also account for clean stochastic noise. Under Assumption 1, a martingale concentration argument gives an additional buffer of order $\sqrt { T \log ( m / \delta ) }$ , where $\delta \in ( 0 , 1 )$ is the failure probability. Adding this concentration buffer gives the corresponding conditional-mean feasibility statement.

## 2.4. Notation Summary

Table 3 collects the main notation; its final three rows preview quantities defined in Sections 3–5.

Table 3 Model notation and selected quantities introduced later.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $T , K , d , m$ </td><td>Number of rounds, non-null actions, context coordinates, and resources.</td></tr><tr><td> $x _ { t } , a _ { t } ; B$ </td><td>Observed context and selected action at round t; resource-budget vector.</td></tr><tr><td> $r ^ { 0 } , b ^ { 0 }$ </td><td>Clean conditional reward and consumption means used by the benchmark.</td></tr><tr><td> $\boldsymbol r ^ { \mathrm { c l } } , \boldsymbol b ^ { \mathrm { c l } }$ </td><td>Clean noisy realizations before corruption.</td></tr><tr><td> $r , b$ </td><td>Clipped corrupted feedback observed for the played action.</td></tr><tr><td> $c ^ { r } , c ^ { b }$ </td><td>Changes from clean to observed outcomes, measured after clipping.</td></tr><tr><td> $V ^ { \mathrm { U B } } , \mathrm { R e g } ( T )$ </td><td>Clean population-LP value and clean expected regret.</td></tr><tr><td> $Z , \lambda , \Lambda$ </td><td>Price scale, normalized resource-price vector, and its admissible set.</td></tr><tr><td> $C _ { Z } , \Gamma$ </td><td>Global effective corruption with a worst-action maximum on each round; supplied upper bound in the known-corruption setting.</td></tr><tr><td> $C _ { Z } ( t )$   $C _ { b } ( t )$ </td><td>Effective corruption accumulated along the selected actions through round t.</td></tr><tr><td> $B ^ { \mathrm { o p } } , U _ { t }$ </td><td>Consumption corruption accumulated along the selected actions through round t.</td></tr><tr><td> $R ; s _ { 0 } , s$ </td><td>Operating budget and observed resource use before round t.</td></tr><tr><td></td><td>Coefficient-norm bound; true sparsity and supplied calibration upper bound.</td></tr><tr><td colspan="2">Selected notation introduced in the estimation and algorithm sections</td></tr><tr><td> $\kappa _ { \mathrm { F E } } , \kappa _ { \mathrm { o p } }$   $A _ { \mathrm { F E } } , A _ { \mathrm { o p } } ; n _ { 0 }$ </td><td>Curvature bounds for forced-exploration and on-policy designs (Appendix A.3). Coefficients in corruption-free confidence widths; minimum retained-sample count per action (Sec-</td></tr><tr><td> $\chi , n _ { \mathrm { c a n d } } ; t _ { \mathrm { i n i t } }$ </td><td>tion 3.2). Constants comparing recommendation counts with actual plays; safe initialization length (Section 5).</td></tr></table>

## 3. Estimation and Optimistic Primal–Dual Learning

The method has two components: an estimator predicts reward and resource consumption, and an allocation rule uses these predictions and their uncertainty to choose actions and update resource prices. We first describe a norm-constrained Lasso estimator, then state the confidence and coverage requirements that connect estimation to regret, and finally give the allocation and price updates. The allocation guarantees depend on these requirements, not on Lasso itself. Calibration, proofs, and certified optimization details are in Appendix A.

## 3.1. Sparse Estimation and Prediction Confidence

For each non-null action, we fit one reward model and one model for each resource. A retained observation is an observation used in these fits. Forced exploration (FE) retains observations from context-independent uniform exploration; Shared-Grid also retains its fixed initialization samples. On-policy learning (OP) instead retains every non-null observation. All models for the same action use the same retained contexts.

Fix an action a and one response, called a channel: reward with parameter $\theta ^ { \star } = \mu _ { a } ^ { \star }$ , or consumption of resource i with $\theta ^ { \star } = w _ { a , i } ^ { \star }$ . Let $S _ { a } ^ { \mathrm { e s t } } ( t ) = \{ \tau < t : a _ { \tau } = a$ and the sample is retained} be the retained round indices before t, and let $n = N _ { a } ^ { \mathrm { e s t } } ( t ) : = | S _ { a } ^ { \mathrm { e s t } } ( t ) |$ . The matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ has the retained contexts as rows, and $y \in \mathbb { R } ^ { n }$ contains the corresponding observed responses.

For analysis, write $y = X \theta ^ { \star } + \eta + \zeta$ , where η is clean stochastic noise and $\zeta$ collects the post-clipping corruptions. The learner observes X and y, not the separate noise and corruption terms. Whether to retain a sample is decided before its feedback is revealed.

For regularization strength $\rho _ { n } > 0$ , we use the norm-constrained Lasso fit (Tibshirani 1996):

$$
\widehat { \theta } \in \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { \| \theta \| _ { 2 } \leq R } \left\{ \frac { \| y - X \theta \| _ { 2 } ^ { 2 } } { 2 n } + \rho _ { n } \| \theta \| _ { 1 } \right\} , \quad n \geq 1 .\tag{8}
$$

The $\ell _ { 1 }$ penalty encourages sparse estimates, while the $\ell _ { 2 }$ constraint enforces the known coefficient bound R. Neither fixes the support of the estimate. Set $\widehat { \theta } = 0$ when $n = 0$

The prediction guarantee uses a supplied sparsity bound $s \geq \operatorname* { m a x } \{ 1 , s _ { 0 } \}$ , a retained-sample threshold $n _ { 0 } .$ and an estimation failure level $\delta _ { \mathrm { e s t } } \in ( 0 , 1 )$ . The bound s calibrates confidence and numerical precision; it is not a support constraint in the fit. The retained design is the matrix of contexts used by that fit. Its restricted strong convexity (RSC) condition requires those contexts to distinguish changes in the coefficient vector, with an $\ell _ { 1 }$ tolerance for high-dimensional data. The constant $\kappa > 0$ measures the curvature in this condition; the precise inequality is (47).

Appendix A.2 specifies $\rho _ { n } , \kappa , n _ { 0 }$ , and Appendix A.8 specifies how accurately to solve the objective. For a fixed retained history, the fitted coefficients and regularization do not depend on a proposed corruption bound Γ. Such a bound changes the confidence radii, not the fitting objective.

PROPOSITION 2 (Prediction confidence for the concrete estimator). Under Assumption 1 and the calibration specified in Appendix A.2, there is a noise event of probability at least $1 - \delta _ { \mathrm { { e s t } } } ,$ , simultaneous over actions, response channels, and obtained sample prefixes. On this event, whenever a retained prefix $n \geq n _ { 0 }$ satisfies the calibrated RSC condition, any feasible fit meeting the certified optimization tolerance of Appendix A.8 satisfies

$$
\operatorname* { s u p } _ { \| \boldsymbol { x } \| _ { 2 } \le 1 } \big | \langle \widehat \theta - \theta ^ { \star } , \boldsymbol { x } \rangle \big | \le \operatorname* { m i n } \left\{ 2 R , \frac { 7 \rho _ { n } \sqrt { s } } { \kappa } + \frac { 4 C _ { n } } { \kappa n } \right\} , \quad C _ { n } : = \| \boldsymbol { \zeta } \| _ { 1 } .\tag{9}
$$

For an exact minimizer, the coefficient 7 can be replaced by 6.

Here $C _ { n }$ measures corruption only in the retained responses for this action and channel. The term $7 \rho _ { n } \sqrt { s } / \kappa$ is the clean estimation contribution: $\rho _ { n }$ decreases as $n ^ { - 1 / 2 }$ under the stated calibration, $\sqrt { s }$ reflects sparsity, and $1 / \kappa$ reflects design conditioning. The additional term $4 C _ { n } / ( \kappa n )$ measures the effect of retained corruption. The cap 2R follows from the norm constraint, and the coefficient $^ { 7 , }$ rather than 6, allows the certified numerical error. Appendix A.5 gives the proof, including the large-corruption case.

## 3.2. Coverage and the Estimator Interface

The allocation rule needs predictions and confidence radii from estimator. Write $\widehat { \mu } _ { a , t - 1 }$ and $\widehat { w } _ { a , i , t - 1 }$ for the reward and resource fits using observations before round t. At the current context, their clipped predictions are $\widehat { r } _ { a , t } = \mathrm { c l i p } ( \langle \widehat { \mu } _ { a , t - 1 } , x _ { t } \rangle , 0 , 1 )$ and $\widehat { b } _ { a , i , t } = \mathrm { c l i p } ( \langle \widehat { w } _ { a , i , t - 1 } , x _ { t } \rangle , 0 , 1 )$ . The interface imposes two requirements: the radii must cover prediction errors, and the uncertainty charged over the run must be controlled.

Pointwise prediction confidence. A corruption candidate Γ is a proposed upper bound on $C _ { Z } ;$ it is valid when $\Gamma \geq C _ { Z }$ . Let $\beta _ { a , t } ^ { r , \Gamma }$ and $\beta _ { a , i , t } ^ { b , \Gamma }$ be nonnegative error radii for the reward and resource-i predictions. These radii bound errors in clean conditional means, not errors relative to the noisy observed feedback. Predictions and radii use the observed history and $x _ { t }$ and are fixed before the fresh action draw. On a common retained history, candidates share point estimates and differ only in their radii.

Let $\delta _ { \mathrm { c o v } } \in ( 0 , 1 )$ be the failure level for the design/coverage event. On one joint event of failure at most $\delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } }$ , require simultaneously for every valid $\Gamma \geq C _ { Z }$

$$
\begin{array} { r } { | \widehat { r } _ { a , t } - r ^ { 0 } ( a , x _ { t } ) | \leq \beta _ { a , t } ^ { r , \Gamma } , \quad | \widehat { b } _ { a , i , t } - b _ { i } ^ { 0 } ( a , x _ { t } ) | \leq \beta _ { a , i , t } ^ { b , \Gamma } \quad ( a \in \mathcal { A } , i \in [ m ] , \ t \in [ T ] ) . } \end{array}\tag{10}
$$

For Lasso, Proposition 2 and $C _ { Z } \leq \Gamma$ give the radii in Appendix ${ \bf A . 4 ; }$ use unit radii before $n _ { 0 }$ and zero predictions and radii for null. The allocation rule compares reward minus resource cost. Its combined radius therefore adds reward uncertainty to resource uncertainty in the same units. Since $\| \lambda \| _ { 1 } \leq 1$ , the largest resource radius suffices. Write h for the length of the clean-score interval [−Z, 1]:

$$
\beta _ { a , t } ^ { Z , \Gamma } : = \beta _ { a , t } ^ { r , \Gamma } + Z \operatorname* { m a x } _ { i } \beta _ { a , i , t } ^ { b , \Gamma } , \quad h : = 1 + Z .\tag{11}
$$

Cumulative uncertainty. Pointwise confidence alone is not enough for a regret bound: the widths charged over the run must also remain small. Let $\widehat { a } _ { t }$ denote the optimistic recommendation defined in Section 3.3, with the fixed corruption input suppressed in this notation. Let $I _ { t } \in \{ 0 , 1 \}$ indicate whether the pre-action safety test (7) passes; it remains one on an active round even if null is subsequently chosen. Require on the same event

$$
2 \sum _ { t = 1 } ^ { T } I _ { t } \beta _ { \widehat { a } _ { t } , t } ^ { Z , \Gamma } \leq \mathcal { R } _ { \mathrm { e s t } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma ) .\tag{12}
$$

Here $\mathcal { R } _ { \mathrm { e s t } } ^ { 0 }$ bounds cumulative uncertainty from clean estimation, and $\psi _ { \mathrm { e s t } } ( \Gamma )$ bounds the additional width caused by retained corruption. Both are analysis quantities, not algorithm inputs. The optimistic comparison charges uncertainty to the recommended action, so no separate cumulative-width bound is needed along the population-LP policy. Shared-Grid needs the corresponding bound uniformly over valid candidates on one shared history; see (30). Any estimator meeting these confidence and cumulative-width requirements can replace Lasso without changing the allocation analysis.

Design conditions for the concrete rates. The conditions below establish the interface for Lasso; the general allocation theorem does not impose them on another estimator that already meets the interface. FE selects its retained actions independently of the current context on active rounds, allowing a population condition to control the retained designs. $\mathrm { O P }$ uses adaptively selected observations and instead requires a condition on its actual retained designs. Restricted-curvature conditions are standard in sparse estimation (Negahban et al. 2012, Raskutti et al. 2010), but adaptive action selection must also be accounted for (Bastani and Bayati 2020). Appendix A.3 gives the calibrations and proofs.

ASSUMPTION 3 (Population design for the concrete estimator). For a generic context $x \sim \mathcal { D } ,$ , known constants $\kappa _ { x } , L _ { x } > 0$ satisfy

$$
\mathbb { E } [ x x ^ { \top } ] \succeq \kappa _ { x } I _ { d } , \quad \mathbb { E } \exp \left( \frac { \langle u , x \rangle ^ { 2 } } { L _ { x } ^ { 2 } \| u \| _ { 2 } ^ { 2 } } \right) \leq 2 \quad ( u \neq 0 ) .\tag{13}
$$

Here $\kappa _ { x }$ lower-bounds the eigenvalues of the population second-moment matrix $\mathbb { E } [ x x ^ { \top } ]$ , while $L _ { x }$ controls the directional sub-Gaussian scale.

For OP, the relevant information is in the contexts actually retained for each action, rather than in the population distribution alone.

CONDITION 4 (On-policy retained-design coverage). For supplied design parameters $\kappa _ { \mathrm { o p } } , v _ { \mathrm { o p } } > 0$ and threshold $n _ { 0 } ,$ , the action-specific realized retained designs satisfy the calibrated RSC event defined in Appendix A.3, simultaneously after $n _ { 0 }$ samples per action, with probability at least $1 - \delta _ { \mathrm { c o v } }$ . Here $\kappa _ { \mathrm { o p } }$ is the RSC curvature and $\upsilon _ { \mathrm { o p } }$ controls its $\ell _ { 1 }$ tolerance.

This condition concerns the algorithm’s actual adaptive, corrupted, and potentially stopped trajectory; population richness alone does not imply it. Shared-Grid OP also needs the separate recommendationcoverage condition in Section 5, because a candidate may recommend an action that is not played.

Rate constants. The concrete regret bounds use $\ell : = \log \left( 1 6 e K ( m + 1 ) d T / \operatorname* { m i n } \{ \delta _ { \mathrm { e s t } } , \delta _ { \mathrm { c o v } } \} \right)$ for concentration and $H _ { T } : = 1 + \log ( 1 + T )$ for the harmonic factor arising when widths are summed. The retained-design curvatures are $\kappa _ { \mathrm { F E } } : = \kappa _ { x } / 2$ for FE and $\kappa _ { \mathrm { o p } }$ from Condition 4 for OP. The coefficients $A _ { \mathrm { F E } }$ and $A _ { \mathrm { o p } }$ multiply the corresponding corruption-free $n ^ { - 1 / 2 }$ score widths. They include model and design parameters, not just universal constants; Appendix A.3 gives their full dependence.

## 3.3. Optimistic Primal–Dual Rule

The primal step recommends an action using estimated reward minus resource cost, with an uncertainty bonus. The dual step adjusts resource prices using the consumption actually observed.

At prices $\lambda _ { t } \in \Lambda$ , substitute the fitted reward and consumption means into the clean score $F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } )$ This gives the estimated score $\widehat { F } _ { a , t } \ ;$ candidate Γ recommends the action with the largest estimate plus its confidence radius:

$$
\widehat { F } _ { a , t } ( \lambda _ { t } ) : = { \widehat { r } } _ { a , t } - Z \sum _ { i = 1 } ^ { m } \lambda _ { t , i } { \widehat { b } } _ { a , i , t } ,\tag{14}
$$

$$
\widehat { a } _ { t } ^ { \Gamma } \in \mathop { \operatorname { a r g m a x } } _ { a \in \mathcal { A } _ { 0 } } \{ \widehat { F } _ { a , t } ( \lambda _ { t } ) + \beta _ { a , t } ^ { Z , \Gamma } \} ,\tag{15}
$$

with zero score and radius for null and the fixed order null, $1 , \ldots , K$ for ties. $\widehat { a } _ { t } ^ { \Gamma }$ is the optimistic recommendation for candidate Γ, computed on rounds that pass the pre-action safety test (7). The actually played action $a _ { t }$ may differ because of forced exploration or the Shared-Grid master; if the safety test fails, the learner plays null from that round onward.

For the price update, represent $\lambda _ { t }$ by the first m coordinates of a probability vector $q _ { t } \in \mathbb { R } _ { + } ^ { m + 1 } \colon \lambda _ { t } = q _ { t , 1 : m }$ 2 and $q _ { t , m + 1 } = 1 - \| \lambda _ { t } \| _ { 1 }$ . The last coordinate holds unused price mass; it is not another physical resource. Compare observed consumption with the per-round operating-budget target $B ^ { \mathrm { o p } } / T$ . For stepsize $\eta _ { q } > 0$ , set $\widetilde { g } _ { t } = \left( b _ { t } ( a _ { t } ) - B ^ { \mathrm { o p } } / T , 0 \right)$ , with zero in the slack coordinate, and update

$$
q _ { t + 1 , j } \propto q _ { t , j } e ^ { \eta _ { q } \widetilde { g } _ { t , j } } , \quad \lambda _ { t + 1 } = q _ { t + 1 , 1 : m } .\tag{16}
$$

Normalize $q _ { t + 1 }$ to sum to one. The update increases the relative weight of resources with larger excess use. We denote its cumulative price-learning penalty by

$$
\mathcal { D } _ { T } : = Z \left[ \frac { \log ( m + 1 ) } { \eta _ { q } } + \eta _ { q } \left( 1 + \frac { B _ { \operatorname* { m a x } } } { T } \right) ^ { 2 } T \right] .\tag{17}
$$

Choosing $\begin{array} { r } { \eta _ { q } \asymp \frac { \sqrt { \log ( m + 1 ) / T } } { 1 + B \operatorname* { m a x } / T } } \end{array}$ gives $\begin{array} { r } { \mathcal { D } _ { T } = O \left( Z \left( 1 + \frac { B _ { \operatorname* { m a x } } } { T } \right) \sqrt { T \log ( m + 1 ) } \right) } \end{array}$

## 4. Robust Learning with Known Corruption

## 4.1. Algorithm and Inputs

We now assume that the learner receives a deterministic bound $\Gamma \geq C _ { Z }$ that is valid on every admissible realization. On each active round, ROPD forms an optimistic recommendation, selects an action, and updates its estimates, resource prices, and observed-budget account from the played action’s feedback. Forced exploration may replace the recommendation by a uniform non-null action; on-policy learning plays the recommendation.

Using the fitted score and radius from Section 3, write the optimistic score as

$$
\begin{array} { r } { U _ { a , t } ^ { \Gamma } : = \widehat { F } _ { a , t } ( \lambda _ { t } ) + \beta _ { a , t } ^ { Z , \Gamma } , } \end{array}\tag{18}
$$

with $U _ { \mathrm { n u l 1 } , t } ^ { \Gamma } = 0$ . Here $U _ { a , t } ^ { \Gamma }$ is a scalar optimistic score for action a, distinct from the cumulative observedconsumption vector $U _ { t }$

In FE mode, $\epsilon _ { t }$ is the probability of a uniform forced action at round t, and only forced observations are retained for estimation. OP retains every non-null observation and sets $\epsilon _ { t } = 0$ . For a fixed retained history, Γ changes confidence radii but not the fitting objective; it can still affect future data through action selection.

Table 4 lists the inputs, and Algorithm 1 gives the complete procedure. In the algorithm, $\mathcal { H } _ { a , t } ^ { \mathrm { e s t } }$ denotes action a’s retained data through round $t ,$ so $| \mathcal { H } _ { a , t - 1 } ^ { \mathrm { e s t } } | = N _ { a } ^ { \mathrm { e s t } } ( t )$ . The concrete fits are specified in Section 3.1 and Appendix A.2. Another estimator meeting the interface can replace these fits without changing the allocation, price, or safety updates.

```latex
Algorithm 1 Robust Optimistic Primal–Dual Learning with Known Corruption (ROPD)
Require: Horizon $T ;$ actions $\overline { { \mathcal { A } = \left[ K \right] } }$ and $\overline { { \mathcal { A } _ { 0 } = \mathcal { A } \cup \left\{ \mathtt { n u l l } \right\} } }$ ; budgets $0 < B ^ { \mathrm { o p } } \leq B ;$ corruption bound $\Gamma \geq C _ { Z } ;$ scale
$Z ;$ dual stepsize $\eta _ { q } .$
Coverage inputs: mode $\in$ {forced, on-policy} and exploration probabilities $\boldsymbol { \epsilon } _ { t } \in [ 0 , 1 ] \ ( \epsilon _ { t } = 0$ in on-policy mode).
Estimation module: predictions and radii satisfying (10) and $( 1 2 ) .$ , with the module’s required inputs.
Estimator inputs: $s , R , \sigma _ { r } , \sigma _ { b } ,$ , calibrated $( \kappa , v , n _ { 0 } )$ , failure levels $\delta _ { \mathrm { e s t } } , \delta _ { \mathrm { c o v } } .$ , and precision $s \rho _ { n } ^ { 2 } / ( 8 \kappa )$
1: Set $\mathcal { H } _ { a , 0 } ^ { \mathrm { e s t } } = \emptyset$ for all $a \in \mathcal { A } , U _ { 1 } = 0 , q _ { 1 , j } = 1 / ( m + 1 )$ for $j \in [ m + 1 ] ,$ , and $\lambda _ { 1 } = q _ { 1 , 1 : m } .$
2: for $t = 1 , \dots , T$ do
3: if $U _ { t , i } > B _ { i } ^ { \mathrm { o p } } - 1$ for some $i \in [ m ]$ then
4: Play $a _ { t } = \mathtt { n u l l } ;$ ; keep $( U _ { t + 1 } , \stackrel { \cdot } { q _ { t + 1 } } , \lambda _ { t + 1 } ) = ( U _ { t } , q _ { t } , \lambda _ { t } )$ and $\mathcal { H } _ { a , t } ^ { \mathrm { e s t } } = \mathcal { H } _ { a , t - 1 } ^ { \mathrm { e s t } }$ for all $a ;$ continue.
5: Observe $x _ { t } .$
$_ 6 \colon$ for $a \in { \mathcal { A } }$ do
7: Set $n = | \mathcal { H } _ { a , t - 1 } ^ { \mathrm { e s t } } | . \mathrm { I f } \ n = 0$ or no justified calibration is supplied, use zero coefficients; otherwise obtain the
channel fits with Appendix $\mathrm { A } . 8 \mathrm { ^ { * } s }$ certified solver, reusing unchanged fits.
8: Use the supplied calibration and certified Lasso precision. Use unit channel radii if $n < n _ { 0 }$ or calibration is
unavailable; otherwise use (51)–(52).
9: Compute $\beta _ { a , t } ^ { Z , \Gamma }$ and $U _ { a , t } ^ { \Gamma }$ from (18).
10: Set $U _ { \mathrm { n u l 1 } , t } ^ { \Gamma } = 0$ and choose $\widehat { a } _ { t } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } _ { 0 } } U _ { a , t } ^ { \Gamma }$ using the fixed tie-breaking rule.
11: Draw $e _ { t } \sim$ Bernoulli $\left( \epsilon _ { t } \right)$ . If $e _ { t } = 1 ,$ , draw $a _ { t }$ uniformly from $\mathcal { A } ;$ otherwise set $a _ { t } = \widehat { a } _ { t }$
12: if $a _ { t } \neq$ null then
13: Observe $r _ { t } ( a _ { t } )$ and $b _ { t } ( a _ { t } )$
14: else
15: Set $r _ { t } ( \mathtt { n u l 1 } ) = 0$ and $b _ { t } ( \mathtt { n u l 1 } ) = 0 .$
16: Set $\mathcal { H } _ { a , t } ^ { \mathrm { e s t } } = \mathcal { H } _ { a , t - \underline { { 1 } } } ^ { \mathrm { e s t } }$ for every $a \in A .$
17: if $a _ { t } \neq$ null and [mode = on-policy or $e _ { t } = 1 ]$ then
18: Append $( x _ { t } , r _ { t } ( a _ { t } ) , b _ { t } ( a _ { t } ) )$ to $\mathcal { H } _ { a _ { t } , t } ^ { \mathrm { e s t } }$
19: Set $g _ { t } = b _ { t } ( a _ { t } ) - B ^ { \mathrm { o p } } / T$ and $\widetilde { g } _ { t } = ( g _ { t } , 0 ) \in \mathbb { R } ^ { m + 1 } .$
20: Set q<sub>t+1,j</sub> = $q _ { t , j } \exp ( \eta _ { q } \widetilde { g } _ { t , j } )$ $, \forall j \in [ m + 1 ] ;$ then set $\lambda _ { t + 1 } = q _ { t + 1 , 1 : m }$ and $\boldsymbol { U } _ { t + 1 } = \boldsymbol { U } _ { t } + \boldsymbol { b } _ { t } ( \boldsymbol { a } _ { t } )$
$\begin{array} { r l } { ~ } & { { } \sum _ { \ell = 1 } ^ { m + 1 } q _ { t , \ell } \exp ( \eta _ { q } \widetilde { g } _ { t , \ell } ) } \end{array}$
```

Table 4 Inputs to the complete known-corruption method.  
```latex
Input Definition and role
$B ^ { \mathrm { o p } }$ Operating budget used by the observed-consumption safety account; the default is $B .$
Γ Deterministic supplied bound satisfying $C _ { Z } \le \bar { \Gamma }$ on every admissible realization.
$Z$ For $B ^ { \mathrm { o p } } = B ,$ a positive scale satisfying Assumption 2; $Z = T / B _ { \mathrm { m i n } }$ suffices. The reduced-budget
calibration is described below.
$\eta _ { q }$ Multiplicative-weights stepsize for the resource-price distribution.
$\epsilon _ { t }$ Probability of a uniform non-null forced-exploration override; it is zero in on-policy mode.
$n _ { 0 }$ Action-specific retained-sample threshold before nontrivial sparse confidence is used.
$( \kappa , v )$ Supplied RSC calibration justified by (49) or Condition 4.
$s , R , \sigma _ { r } , \sigma _ { b }$ Supplied sparsity, parameter-norm, and noise bounds for Lasso confidence.
$\delta _ { \mathrm { e s t } } , \delta _ { \mathrm { c o v } }$ Failure levels used in ℓ and design calibration.
```

The pre-action check leaves room for one more observed consumption, since $b _ { t , i } ( a _ { t } ) \leq 1$ . It therefore gives exact observed feasibility, $\begin{array} { r } { \sum _ { t = 1 } ^ { T } b _ { t , i } ( a _ { t } ) \leq B _ { i } ^ { \mathrm { o p } } } \end{array}$ . If a valid bound ${ \mathit { C } } _ { b } ( T ) \leq { \Gamma } _ { b }$ is known and $B _ { i } > \Gamma _ { b }$ for every resource, using $B ^ { \mathrm { o p } } = B - \Gamma _ { b } { \bf 1 }$ also gives strict clean sample-path feasibility by Proposition 1.

The regret guarantees below use the default budget $B ^ { \mathrm { o p } } = B ,$ . For a reduced operating budget, first apply the analysis to the reduced-budget benchmark, with $Z \ge V ^ { \mathrm { U B } } ( B ^ { \mathrm { o p } } ) / B _ { \mathrm { m i n } } ^ { \mathrm { o p } }$ , where $B _ { \mathrm { m i n } } ^ { \mathrm { o p } } : = \mathrm { m i n } _ { i } B _ { i } ^ { \mathrm { o p } }$

$Z = T / B _ { \mathrm { m i n } } ^ { \mathrm { o p } }$ is sufficient. Evaluate all scale-dependent quantities, including $C _ { Z }$ and its supplied bound Γ, with this same scale. Returning to the original benchmark then adds $V ^ { \mathrm { U B } } ( B ) - V ^ { \mathrm { U B } } ( B ^ { \mathrm { o p } } )$ , as explained in Appendix B. This scale qualification is needed for the regret analysis, not for the pathwise feasibility statement. For the default-budget price update, the stepsize choice from Section 3.3 gives

$$
\mathcal { D } _ { T } = O \left( Z ( 1 + \frac { B _ { \operatorname* { m a x } } } { T } ) \sqrt { T \log ( m + 1 ) } \right) .\tag{19}
$$

## 4.2. Regret and Resource Guarantees

The first guarantee separates estimation uncertainty from exploration, price learning, direct corruption, and stopping. Set $\delta _ { \mathrm { t o t } } : = \delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } }$ , combining estimator and coverage failure probabilities.

THEOREM 1 (Known corruption: regret and resource control). Suppose Assumptions 1 and 2 hold and the estimator/coverage pair satisfies the interface in (10)–(12)for a deterministic supplied bound Γ with $C _ { Z } \le \Gamma$ on every admissible realization. Algorithm 1 with $B ^ { \mathrm { o p } } = B$ satisfies the following expected-regret bound:

$$
\mathrm { R e g } ( T ) \leq O \left( \mathcal { R } _ { \mathrm { e s t } } ^ { 0 } + h \sum _ { t = 1 } ^ { T } \epsilon _ { t } + \mathcal { D } _ { T } + \Gamma + \psi _ { \mathrm { e s t } } ( \Gamma ) + Z \right) + h \delta _ { \mathrm { t o t } } T .\tag{20}
$$

Moreover, on every realization, $\mathrm { V i o l } _ { \mathrm { o b s } } ( T ) = 0 a n d \mathrm { V i o l } _ { \mathrm { c l } } ( T ) \leq C _ { b } ( T ) \leq C _ { z } / Z \leq \Gamma / Z .$

Proof overview. In (20), $\mathcal { R } _ { \mathrm { e s t } } ^ { 0 }$ measures clean estimation uncertainty, $h \sum _ { t } \epsilon _ { t }$ the cost of forced exploration, and $\mathcal { D } _ { T }$ the cost of learning resource prices. Corruption enters in two distinct ways: Γ accounts for transferring corrupted scores and resource use to their clean counterparts, while $\psi _ { \mathrm { e s t } } ( \Gamma )$ accounts for its accumulated effect on prediction widths. The term Z is the stopping boundary, and $h \delta _ { \mathrm { t o t } } T$ is the failure-event cost. Confidence controls the clean-score gap, the price update controls observed resource imbalance, and $Z B _ { i } \geq V ^ { \mathrm { U B } }$ offsets benchmark reward lost after stopping. Appendix B gives the full argument.

## 4.3. Concrete Rates

The next corollary instantiates the theorem with Lasso. FE collects context-independent exploration samples and pays an exploration cost. OP uses all non-null observations but requires the realized-design condition; its sharper rate is not asserted under the FE assumptions alone.

COROLLARY 1 (Known corruption: concrete rates). Under Assumptions 1–2, run Algorithm 1 with $B ^ { \mathrm { o p } } = B$ , a deterministic valid bound $\Gamma \geq C _ { Z }$ , and the estimator and certified precision of Proposition 2. For a universal C, with the rate constants defined in Section 3.2:

(i) Forced exploration. Under the population-design Assumption 3 (calibration in Appendix A.3), use (49) and a constant forced-exploration probability $\epsilon _ { t } \equiv \epsilon$ chosen by (50). Then

$$
\begin{array} { r l r } & { } & { \mathrm { R e g } ( T ) \le C \bigg [ h ^ { 1 / 3 } A _ { \mathrm { F E } } ^ { 2 / 3 } K ^ { 1 / 3 } T ^ { 2 / 3 } + h \sqrt { K ( n _ { 0 } + \ell ) T } } \\ & { } & { \quad \quad \quad + \left( 1 + \frac { K H _ { T } } { \kappa _ { \mathrm { F E } } \epsilon } \right) \Gamma + \mathcal { D } _ { T } + Z \bigg ] + h \delta _ { \mathrm { t o t } } T . } \end{array}\tag{21}
$$

(ii) On-policy learning. Under the retained-design Condition $^ { 4 , }$ with $\epsilon _ { t } = 0 $ , use the supplied OP calibration and the same certified Lasso rule on all non-null samples. Then

$$
\mathrm { R e g } ( T ) \leq C \bigg [ A _ { \mathrm { o p } } \sqrt { K T } + h K n _ { 0 } + \left( 1 + \frac { K H _ { T } } { \kappa _ { \mathrm { o p } } } \right) \Gamma + \mathcal { D } _ { T } + Z \bigg ] + h \delta _ { \mathrm { t o t } } T .\tag{22}
$$

The terms involving $A _ { \mathrm { F E } }$ or $A _ { \mathrm { o p } }$ measure clean learning, and the $n _ { 0 }$ terms account for the initial period before informative confidence is available. The terms proportional to Γ measure corruption’s additional effect under the corresponding design conditioning. Price learning, stopping, and failure events contribute the same $\mathcal { D } _ { T } + Z + h \delta _ { \mathrm { t o t } } T$ terms as in the theorem.

With proportional budgets, fixed remaining parameters, dominated burn-in, the stepsize following (17), and $\delta _ { \mathrm { t o t } } \leq T ^ { - 2 }$ , the FE and OP bounds simplify to $\widetilde { O } ( T ^ { 2 / 3 } + \Gamma T ^ { 1 / 3 } )$ and $\widetilde { O } ( \sqrt { T } + \Gamma )$ , respectively. Here $\widetilde O$ suppresses polylogarithmic factors in $d , K , m , T$ and inverse failure levels. Appendix B.6 gives full analysis.

Comparison of rates. Reward-corrupted linear bandits without resource constraints achieve near-optimal or minimax corruption dependence (He et al. 2022, Liu et al. 2024), while clean sparse CBwK exhibits $T ^ { 2 / 3 }$ -type learning costs and, under stronger coverage, $\sqrt { T }$ behavior (Ma et al. 2024). Our results extend these CBwK regimes to joint reward and consumption corruption, giving $\widetilde { O } ( T ^ { 2 / 3 } + \Gamma T ^ { 1 / 3 } )$ under FE and $\widetilde { O } ( \sqrt { T } + \Gamma )$ under OP, together with observed-budget feasibility and control of clean resource violation. The rates are not directly comparable across settings and no matching lower bound establishes their optimality.

## 4.4. Exploration Tradeoff and the Clean Case

The preceding FE rate uses a calibrated exploration probability. The next result keeps ϵ explicit, showing the tradeoff between collecting informative samples and the reward cost of exploration.

COROLLARY 2 (Known FE with a calibrated estimator). Under Assumptions 1, 2, and 3, run $A l g o -$ rithm 1 with $B ^ { \mathrm { o p } } = B$ , a deterministic pathwise bound $C _ { Z } \le \Gamma$ , constant $0 < \epsilon \leq 1 / 2 ,$ , and (49). Use the regularization, trivial early radii, and certified numerical precision ofAppendix A.2. For a universal C and $\delta _ { \mathrm { t o t } } = \delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } }$

$$
\begin{array} { r l r } & { } & { \mathrm { R e g } ( T ) \le C \bigg [ \cfrac { h K ( n _ { 0 } + \ell ) } { \epsilon } + A _ { \mathrm { F E } } \sqrt { \cfrac { K T } { \epsilon } } + h \epsilon T } \\ & { } & { + \mathcal { D } _ { T } + \Gamma + \cfrac { K \Gamma H _ { T } } { \kappa _ { \mathrm { F E } } \epsilon } + Z \bigg ] + h \delta _ { \mathrm { t o t } } T . } \end{array}\tag{23}
$$

The bound holds for every $T \geq 1$ , even if coverage is not reached. The choice (50) gives the main-text bound (21), by the calculation in Appendix B.6.

In (23), smaller ϵ reduces the exploration cost hϵT but increases the initial coverage and estimation costs, including the corruption-dependent width. The choice (50) balances the clean terms and does not depend on Γ; the corruption term is evaluated at that choice. The bound remains valid at short horizons, but need not be informative before enough samples are collected.

OP has no forced-exploration parameter. Setting both the actual corruption and the supplied allowance to zero gives the following special case of its bound.

COROLLARY 3 (Clean known-corruption case). Under the on-policy conditions of Corollary 1 with $C _ { Z } = \Gamma = 0$ , the Γ-dependent contribution in (22) vanishes. The burn-in, numerical/statistical width, dual term, stopping boundary, and $h \delta _ { \mathrm { t o t } } T$ remain.

These claims follow by substituting Lemma 7 or Proposition 4 into the modular known-corruption theorem; Appendix B proves the allocation step. In particular, no sparsity-constrained optimizer or empirical sparse-eigenvalue oracle is used.

## 5. Adaptation to Unknown Corruption

Without a supplied corruption bound, Shared-Grid considers several possible bounds, called candidates, and uses a master to combine their recommendations into one action distribution. Candidates share the retained data, fitted coefficients, resource prices, and budget account; only their confidence radii and recommendations differ. They are not separate runs of ROPD: each round produces one played action and at most one retained observation.

## 5.1. Candidates and Shared Estimation

Corruption candidates. A candidate Γ uses the optimistic recommendation (15) with the common fits and its own radius $\beta _ { a , t } ^ { Z , \Gamma }$ . The post-clipping bounds imply $C _ { Z } \le C _ { \mathrm { m a x } , Z } : = T ( 1 + Z )$ , so zero and successive powers of two cover all admissible corruption levels:

$$
\mathcal { G } : = \{ 0 \} \cup \{ 2 ^ { j } : j = 0 , \ldots , \lceil \log _ { 2 } C _ { \operatorname* { m a x } , Z } \rceil \} .\tag{24}
$$

The learner uses every candidate without knowing which are valid. For analysis, define

$$
\Gamma ^ { \star } : = \operatorname* { m i n } \{ \Gamma \in { \mathcal { G } } : \Gamma \geq C _ { Z } \} .\tag{25}
$$

It satisfies

$$
\Gamma ^ { \star } = 0 \mathrm { i f } C _ { Z } = 0 , \qquad C _ { Z } \le \Gamma ^ { \star } \le \mathrm { m a x } \{ 1 , 2 C _ { Z } \} \mathrm { o t h e r w i s e } .\tag{26}
$$

The comparator $\Gamma ^ { \star }$ is determined by the realized path, not supplied to or computed by the learner. Its recommendations use the common realized history, not a counterfactual history from running that candidate alone.

Initialization and retained samples. Begin with a prescribed cyclic schedule over the K non-null actions, of length $t _ { \mathrm { i n i t } } \leq T$ , a nonnegative multiple of K. The schedule is fixed before the current context and feedback, and every round must pass the safety test (7); failure stops non-null play and freezes all states. Initialization observations are retained and update prices and the budget account, but not master weights.

After initialization, forced-exploration (FE) mode overrides the master with probability $\epsilon _ { t }$ and selects uniformly from the non-null actions. Let $e _ { t } = 1$ mark this override; in on-policy (OP) mode, $\boldsymbol { \epsilon } _ { t } = \boldsymbol { e } _ { t } = 0$

Recall that $I _ { t } = 1$ means the safety test passes, even if null is then chosen. The shared retained set for action a before round t is

$$
\mathcal { T } _ { a , t } ^ { \mathrm { e s t } } = \left\{ \begin{array} { l l } { \{ \boldsymbol { \tau } < t : I _ { \tau } = 1 , \ : a _ { \tau } = a , \ : [ \boldsymbol { \tau } \leq t _ { \mathrm { i n i t } } \mathrm { ~ o r ~ } e _ { \tau } = 1 ] \} , } & { \mathrm { m o d e = f o r c e d } , } \\ { \{ \boldsymbol { \tau } < t : I _ { \tau } = 1 , \ : a _ { \tau } = a \} , } & { \mathrm { m o d e = o n - p o l i c y } . } \end{array} \right.\tag{27}
$$

Thus FE retains initialization and forced samples, whereas OP retains every non-null observation. Eligibility is decided before feedback. Each eligible observation is appended once, and all candidates use the same regularization, numerical precision, supplied calibration, and fits.

## 5.2. Master and Complete Procedure

Advice and sampling. The EXP4.P master (Beygelzimer et al. 2011) combines experts, each supplying an action distribution called its advice. Set $K _ { 0 } : = | { \mathcal { A } } _ { 0 } | = K + 1$ and ${ \mathcal { E } } _ { \mathrm { M } } : = { \mathcal { G } } \cup \{ { \mathrm { u n i f } } \}$ , with $M : = | \mathcal { G } | + 1$ Candidate Γ gives advice $\xi _ { t , \Gamma } ( a ) = \mathbf { 1 } \{ \widehat { a } _ { t } ^ { \Gamma } = a \}$ ; the extra expert gives $\xi _ { t , \mathrm { u n i f } } ( a ) = 1 / K _ { 0 }$ , as required by Beygelzimer et al. (2011, Theorem 2). Advice $\xi _ { t , j }$ is distinct from corruption; the extra expert requires no fit. Expert weights $w _ { t , j }$ start at one. Their normalized values $w _ { t , j } / W _ { t }$ , where $\textstyle W _ { t } : = \sum _ { j } w _ { t , j }$ , weight the advice in the master’s action distribution $p _ { t }$ . For failure level $\delta _ { \mathrm { m a s t e r } }$ , use probability floor $p _ { \mathrm { m i n } }$ and confidence-bonus coefficient $\alpha _ { m }$ :

$$
p _ { \operatorname* { m i n } } : = \operatorname* { m i n } \left\{ \frac { 1 } { 2 K _ { 0 } } , \sqrt { \frac { \log M } { K _ { 0 } T } } \right\} , \quad \alpha _ { m } : = \sqrt { \frac { \log ( 3 M / \delta _ { \operatorname* { m a s t e r } } ) } { K _ { 0 } T } } .\tag{28}
$$

The uniform expert supplies advice, smoothing ensures $p _ { t } ( a ) \geq p _ { \mathrm { m i n } } .$ , and forced exploration supplies independent FE samples. The distribution $p _ { t }$ applies only on the non-forced branch.

Payoffs and updates. The master evaluates reward net of current resource cost and rescales this score to [0, 1]:

$$
F _ { t } ^ { \mathrm { o b s } } ( a ; \lambda _ { t } ) : = r _ { t } ( a ) - Z \lambda _ { t } ^ { \top } b _ { t } ( a ) , \quad Y _ { t } ( a ) : = \frac { F _ { t } ^ { \mathrm { o b s } } ( a ; \lambda _ { t } ) + Z } { 1 + Z } , \quad a \in \mathcal { A } _ { 0 } .\tag{29}
$$

Since $F _ { t } ^ { \mathrm { o b s } } \in [ - Z , 1 ] , Y _ { t } \in [ 0 , 1 ]$ . Only the played action’s feedback is observed. On a non-forced active round after initialization, importance weighting divides its payoff by $p _ { t } ( a _ { t } )$ to correct for unequal sampling probabilities. Algorithm 2 defines the action-payoff estimate $\widehat { Y } _ { t } ( a )$ , its advice-weighted expert estimate $\widehat { y } _ { t , j }$ and the inverse-probability bonus term $\widehat { v } _ { t , j }$

The complete procedure below uses the shared estimator, including trivial early radii. Expert weights $w _ { t , j }$ combine advice; price weights $q _ { t }$ govern resources. Initialization and overrides freeze expert weights, but prices and accounting update from observed consumption on every active round.

Algorithm 2 Complete Shared-Grid Procedure for Unknown Corruption   
Require: Horizon $T ;$ initialization $t _ { \mathrm { i n i t } } ;$ actions $\mathcal { A } _ { 0 } \dag ,$ operating budget $B ^ { \mathrm { o p . } } ,$ scale $Z = T / B _ { \mathrm { m i n } } ;$ coverage mode ∈   
{forced, on-policy}; rates $0 \le \epsilon _ { t } \le 1 / 2$ with $\epsilon _ { t } = 0$ on-policy; stepsize $\eta _ { q } ;$ estimator inputs $( s , R , \sigma _ { r } , \sigma _ { b } , \kappa , v , n _ { 0 } ) \colon$   
failure levels $\delta _ { \mathrm { { e s t } } } , \delta _ { \mathrm { { c o v } } } , \delta _ { \mathrm { { m a s t e r } } }$ and the certified Lasso precision.   
Estimation module: shared predictions and radii satisfying (10) and (30), with the module’s required inputs.   
1: Form G and ${ \mathcal { E } } _ { \mathrm { M } }$ as above; set $w _ { 1 , j } = 1 \mathrm { f o r } j \in \mathcal { E } _ { \mathrm { M } } , q _ { 1 , j } = 1 / ( m + 1 ) , \lambda _ { 1 } = q _ { 1 , 1 : m } , U _ { 1 } = 0 ,$ , and initialize one shared   
estimator history per action.   
2: for $t = 1 , \dots , T$ do   
3: if $U _ { t , i } > B _ { i } ^ { \mathrm { o p } } - 1$ for some $i \in$ [m] then   
4: Play $a _ { t } =$ null; set $I _ { t } = e _ { t } = 0 ;$ ; freeze all states; continue.   
5: Set $I _ { t } = 1$ and observe $x _ { t }$   
6: $\mathbf { i f } t \le t _ { \mathrm { i n i t } }$ then   
7: Play the next non-null action in the fixed cyclic schedule; set $e _ { t } = 0$ and keep the master weights fixed.   
8: else   
9: For each action, set $n = \left| \mathcal { I } _ { a , t } ^ { \mathrm { e s t } } \right|$ . Use zero coefficients if $n = 0$ or no justified calibration is supplied; otherwise   
fit or reuse the certified channel estimates. All candidates share these fits.   
10: for $\Gamma \in \mathcal G$ do   
11: Compute $\beta _ { a , t } ^ { Z , \Gamma }$ and $\widehat { a } _ { t } ^ { \Gamma } \in$ arg ma $\underline { { \underline { { \tau } } } } _ { a \in \mathcal { A } _ { 0 } } \{ \widehat { F } _ { a , t } ( \lambda _ { t } ) + \beta _ { a , t } ^ { Z , \Gamma } \}$   
12: Set $\begin{array} { r } { W _ { t } = \sum _ { j \in \mathcal { E } _ { \mathrm { M } } } w _ { t , j } \mathrm { ~ a n d ~ } p _ { t } ( a ) = ( 1 - K _ { 0 } p _ { \operatorname* { m i n } } ) \sum _ { j \in \mathcal { E } _ { \mathrm { M } } } ( w _ { t , j } / W _ { t } ) \xi _ { t , j } ( a ) + p _ { \operatorname* { m i n } } \mathrm { ~ f o r ~ } a \in \mathcal { A } _ { 0 } . } \end{array}$   
13: In forced mode draw $e _ { t } \sim$ Bernoull $\mathsf { i } ( \epsilon _ { t } ) ;$ ; otherwise set $\begin{array} { r } { \dot { e _ { t } } = 0 . } \end{array}$ . If $e _ { t } = 1$ , draw $a _ { t }$ uniformly from $\mathcal { A } ;$ otherwise   
draw $a _ { t } \sim p _ { t }$   
14: Observe $( r _ { t } ( a _ { t } ) , b _ { t } ( a _ { t } ) )$ for $a _ { t } \neq$ null; for $a _ { t } = \mathtt { n u l 1 }$ , set both to zero.   
15: if $a _ { t } \neq$ null and $[ t \leq t _ { \mathrm { i n i t } }$ or mode = on-policy or $e _ { t } = 1 ]$ then   
16: Append $( x _ { t } , r _ { t } ( a _ { t } ) , b _ { t } ( a _ { t } ) )$ once to the shared history for $a _ { t } .$   
17: if $t > t _ { \mathrm { i n i t } }$ and $e _ { t } = 0$ then   
18: Set $Y _ { t } ( a _ { t } ) = [ r _ { t } ( a _ { t } ) - Z \lambda _ { t } ^ { \top } b _ { t } ( a _ { t } ) + Z ] / ( 1 + Z )$ and $\widehat { Y } _ { t } ( a ) = \mathbf { 1 } \{ a = a _ { t } \} Y _ { t } ( a _ { t } ) / p _ { t } ( a _ { t } )$   
19: for $j \in \mathcal { E } _ { \mathrm { M } }$ do   
20: Set $\begin{array} { r } { \widehat { y } _ { t , j } = \sum _ { a \in \mathcal { A } _ { 0 } } \xi _ { t , j } ( a ) \widehat { Y } _ { t } ( a ) } \end{array}$ and $\begin{array} { r } { \widehat { v } _ { t , j } = \sum _ { a \in \mathcal { A } _ { 0 } } \xi _ { t , j } ( a ) / p _ { t } ( a ) } \end{array}$   
21: Update $w _ { t + 1 , j } = w _ { t , j } \exp \{ ( p _ { \mathrm { m i n } } / 2 ) ( \widehat { y } _ { t , j } + \alpha _ { m } \widehat { v } _ { t , j } ) \}$   
22: else   
23: Keep every master weight unchanged.   
24: Update the observed budget account and resource prices: $\boldsymbol { U } _ { t + 1 } = \boldsymbol { U } _ { t } + \boldsymbol { b } _ { t } \big ( \boldsymbol { a } _ { t } \big )$ and $( q _ { t + 1 } , \lambda _ { t + 1 } )$ by (16).

## 5.3. Confidence and Recommendation Coverage

A common confidence event. Because $\Gamma ^ { \star }$ depends on the realized path, the analysis needs one event covering every valid candidate. Shared fits, the time-uniform noise event, and calibrated retained designs give simultaneous pointwise confidence; the cumulative requirement is

$$
2 \sum _ { t = t _ { \mathrm { i n i t } } + 1 } ^ { T } I _ { t } \beta _ { \hat { a } _ { t } ^ { \Gamma } , t } ^ { Z , \Gamma } \leq \mathcal { R } _ { \mathrm { s h } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma ) .\tag{30}
$$

This inequality must hold for all valid $\Gamma \in \mathcal G$ on the same event. Here $\mathcal { R } _ { \mathrm { s h } } ^ { 0 }$ bounds corruption-free recommendation widths after initialization, and $\psi _ { \mathrm { e s t } } ( \Gamma )$ is their additional corruption cost. The common event and radii monotone in Γ permit substitution of $\Gamma ^ { \star }$ after the run; Appendix C gives the argument. Why OP needs recommendation coverage. A candidate may repeatedly recommend an action that the master rarely plays. Retained-design coverage controls the quality of that action’s data, but does not ensure enough observations for its recommendations. Let $\begin{array} { r } { N _ { a } ( t ) : = \sum _ { \tau < t } { \bf 1 } \{ a _ { \tau } = a \} } \end{array}$ count actual plays, including

initialization, and let $\begin{array} { r } { N _ { a } ^ { \mathrm { r e c , \Gamma } } ( t ) : = \sum _ { \tau = t _ { \mathrm { i n i t } } + 1 } ^ { t - 1 } I _ { \tau } { \bf 1 } \{ \widehat { a } _ { \tau } ^ { \Gamma } = a \} } \end{array}$ count post-initialization active recommendations.   
The following condition relates the two counts.

CONDITION 5 (Covered recommendations). For a recommendation-to-play ratio $\chi \geq 1$ , additive count slack $n _ { \mathrm { c a n d } } \geq 0$ , andfailure level $\delta _ { \mathrm { c a n d } } \in ( 0 , 1 )$

$$
\begin{array} { r } { \mathbb { P } \left( N _ { a } ^ { \mathrm { r e c , F } } ( t ) \leq n _ { \mathrm { c a n d } } + \chi N _ { a } ( t ) f o r a l l \Gamma \in \mathcal { G } , a \in \mathcal { A } , a n d t \leq T + 1 \right) \geq 1 - \delta _ { \mathrm { c a n d } } . } \end{array}\tag{31}
$$

The ratio $\chi$ and slack $n _ { \mathrm { c a n d } }$ ensure that frequent recommendations are supported by actual plays. This complements the OP retained-design Condition 4; FE instead obtains candidate-independent samples through uniform exploration and does not need this extra condition. Both OP conditions concern the actual shared trajectory, not a post-run diagnostic. The next lemma gives sufficient sampling probabilities.

LEMMA 1 (Recommendation-aligned sampling implies candidate coverage). Suppose that on every post-initialization active round in on-policy mode, a candidate-recommended action receives conditional sampling probability at least $q _ { \mathrm { c o v } } > 0 .$ : whenever $\widehat { a } _ { t } ^ { \Gamma } = a .$

$$
\mathbb { P } ( a _ { t } = a \mid \mathcal { Q } _ { t } ) \geq q _ { \mathrm { c o v } } ,
$$

where $\mathcal { Q } _ { t }$ contains the history, current context, shared estimates, prices, and all candidate recommendations before the fresh action draw. Then, with probability at least $1 - \delta _ { \mathrm { c a n d } }$ , simultaneously for every $\Gamma \in { \mathcal { G } } , a \in { \mathcal { A } } ,$ and $t \leq T + 1$

$$
N _ { a } ^ { \mathrm { r e c , T } } ( t ) \leq \frac { 2 } { q _ { \mathrm { c o v } } } N _ { a } ( t ) + \frac { 8 } { q _ { \mathrm { c o v } } } \log \frac { 2 | \mathcal { G } | K T } { \delta _ { \mathrm { c a n d } } } .\tag{32}
$$

Consequently, Condition 5 holds with

$$
\chi = { \frac { 2 } { q _ { \mathrm { c o v } } } } , \quad n _ { \mathrm { c a n d } } = { \frac { 8 } { q _ { \mathrm { c o v } } } } \log { \frac { 2 | \mathcal { G } | K T } { \delta _ { \mathrm { c a n d } } } } .
$$

The proof is in Appendix C.4. Default smoothing gives $q _ { \mathrm { c o v } } = p _ { \mathrm { m i n } }$ and hence finite-horizon coverage, but $p _ { \mathrm { m i n } }$ decreases with $T \colon$ it does not establish the horizon-independent ratio used in the sharper OP summary. A horizon-independent lower bound on recommendation-aligned sampling instead gives constant $\chi$ and logarithmic slack. Appendix A.7 illustrates nonempty design and count conditions, without asserting them for every adaptive OP trajectory. Table 5 summarizes the additional notation.

Table 5 Additional notation for Shared-Grid and recommendation coverage.
<table><tr><td>Symbol</td><td>Definition and role</td></tr><tr><td> $t _ { \mathrm { i n i t } }$ </td><td>Prescribed cyclic initialization length, divisible by K and subject to budget safety.</td></tr><tr><td> $| \mathcal G |$ </td><td>Number of corruption candidates in  ${ \mathcal { G } } .$ </td></tr><tr><td> $K _ { 0 }$ </td><td>Number of actions including the null action,  $K _ { 0 } = | \mathcal { A } _ { 0 } | = K + 1 .$ </td></tr><tr><td> $p _ { \mathrm { m i n } }$ </td><td>Probability floor for each action on the master-controlled branch.</td></tr><tr><td> $\alpha _ { m }$ </td><td>EXP4.P confidence-bonus coefficient.</td></tr><tr><td> ${ \cal N } _ { a } ^ { \mathrm { r e c , F } } ( t )$ </td><td>Post-initialization active-round count of candidate Γ recommending action a before t</td></tr><tr><td> $n _ { \mathrm { c a n d } }$ </td><td>Additive slack when comparing recommendation counts with actual plays.</td></tr><tr><td> $\chi$ </td><td>Multiplicative bound on recommendations relative to actual plays.</td></tr></table>

## 5.4. Regret and Resource Guarantees

The master-comparison cost includes override concentration and stochastic observed-to-clean score transfer:

$$
\mathrm { R e g } _ { \mathrm { m s t } } ( T ) : = 2 0 h \sqrt { T K _ { 0 } \log ( 3 M / \delta _ { \mathrm { m a s t e r } } ) } .\tag{33}
$$

Here $h = 1 + Z ;$ direct corruption is charged separately. When the nontrivial-horizon condition for EXP4.P fails, the score-range bound gives the same order. The failure totals are

$$
\delta _ { \mathrm { t o t } } ^ { \mathrm { U , F E } } : = \delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } } + \delta _ { \mathrm { m a s t e r } } , \quad \delta _ { \mathrm { t o t } } ^ { \mathrm { U , O P } } : = \delta _ { \mathrm { t o t } } ^ { \mathrm { U , F E } } + \delta _ { \mathrm { c a n d } } .\tag{34}
$$

Below, $\delta _ { \mathrm { t o t } }$ is the FE or OP total, according to the mode; OP additionally includes the recommendationcoverage event.

THEOREM 2 (Unknown corruption: regret and resource control). Run Algorithm 2 with $B ^ { \mathrm { o p } } = B$ and the master specification of Section 5.2. Suppose Assumptions 1 and 2 hold, and the pointwise and grid-uniform width interfaces (10) and (30) hold on the corresponding estimator-and-coverage event. Then

$$
\begin{array} { r l } & { \displaystyle \mathrm { R e g } ( T ) \leq { \cal O } \biggl ( \mathcal { R } _ { \mathrm { s h } } ^ { 0 } + h t _ { \mathrm { i n i t } } + h \sum _ { t = 1 } ^ { T } \epsilon _ { t } + \mathrm { R e g } _ { \mathrm { m s t } } ( T ) + \mathcal { D } _ { T } + Z } \\ & { \quad \quad \quad \quad + \mathbb { E } [ 3 \Gamma ^ { \star } + \psi _ { \mathrm { e s t } } ( \Gamma ^ { \star } ) ] \biggr ) + h \delta _ { \mathrm { t o t } } T . } \end{array}\tag{35}
$$

On every realization, $\mathrm { V i o l } _ { \mathrm { o b s } } ( T ) = 0$ and $\mathrm { V i o l } _ { \mathrm { c l } } ( T ) \le C _ { b } ( T ) \le C _ { Z } / Z \le \Gamma ^ { \star } / Z .$ . Consequently, $\mathbb { E } [ \mathrm { V i o l } _ { \mathrm { c l } } ( T ) ] \leq \mathbb { E } [ \Gamma ^ { \star } ] / Z$

Interpretation and proof overview. Relative to known corruption, the bound uses shared widths, adds initialization $h t _ { \mathrm { i n i t } }$ and master cost $\mathrm { R e g } _ { \mathrm { m s t } } ( T )$ , and compares with $\Gamma ^ { \star }$ . The term $3 \Gamma ^ { \star }$ accounts for two observed-to-clean score transfers and one resource transfer, separately from the prediction-width cost $\psi _ { \mathrm { e s t } } ( \Gamma ^ { \star } )$ . The comparator is path-dependent, so these costs are averaged; feasibility remains pathwise. Price learning and stopping use the same argument as before. Appendix C gives the proof.

Concrete Lasso rates. Substituting the shared FE or OP widths gives the following rates, with $A _ { \mathrm { F E } } , A _ { \mathrm { o p } } , n _ { 0 } , H _ { T }$ defined in Section 3.2.

COROLLARY 4 (Unknown corruption: concrete shared-data rates). Under Assumptions $I { - } 2 ,$ , run Algorithm 2 with $B ^ { \mathrm { o p } } = B$ , the shared estimator and precision ofProposition 2, and the EXP4.P specification of Section 5.2. With a universal C and the corresponding failure total in (34):

(i) Forced exploration. Under the population-design Assumption 3 (calibration in Appendix A.3), use (49) and a constantforced-exploration probability $\epsilon _ { t } \equiv \epsilon$ chosen by (50). Then

$$
\begin{array} { r l } & { \mathrm { R e g } ( T ) \leq C \left[ h ^ { 1 / 3 } A _ { \mathrm { F E } } ^ { 2 / 3 } K ^ { 1 / 3 } T ^ { 2 / 3 } + h \sqrt { K ( n _ { 0 } + \ell ) T } + h t _ { \mathrm { i n i t } } \right. } \\ & { \qquad \left. + \left( 3 + \frac { K H _ { T } } { \kappa _ { \mathrm { F E } } \epsilon } \right) \mathbb { E } [ \Gamma ^ { \star } ] + \mathrm { R e g } _ { \mathrm { m s t } } ( T ) + \mathcal { D } _ { T } + Z \right] + h \delta _ { \mathrm { t o t } } T . } \end{array}\tag{36}
$$

(ii) On-policy learning. Under Conditions 4 and 5, with $\epsilon _ { t } = 0 ,$ , use the stated initialization, complete grid, and uniform-advice expert. Fit every realized non-null sample using the supplied OP calibration and the same certified Lasso rule. Then

$$
\begin{array} { r l r } {  { \mathrm { R e g } ( T ) \le C \biggl [ \chi A _ { \mathrm { o p } } \sqrt { K T } + \mathrm { R e g } _ { \mathrm { m s t } } ( T ) + h \{ t _ { \mathrm { i n i t } } + K ( n _ { \mathrm { c a n d } } + \chi n _ { 0 } ) \} } } \\ & { } & { \qquad + ( 3 + \frac { K \{ \chi H _ { T } + n _ { \mathrm { c a n d } } \} } { \kappa _ { \mathrm { o p } } } ) \mathbb { E } [ \Gamma ^ { \star } ] + \mathcal { D } _ { T } + Z \biggr ] + h \delta _ { \mathrm { t o t } } T . } \end{array}\tag{37}
$$

Compared with Corollary 1, both modes add initialization and master costs; OP also pays the recommendation-coverage factors $\chi , n _ { \mathrm { c a n d } }$ . Under the same proportional-budget and stepsize regime, fixed remaining parameters, dominated initialization and burn-in, and $\delta _ { \mathrm { t o t } } \leq T ^ { - 2 }$ , the FE and OP summaries are $\widetilde { \cal O } ( T ^ { 2 / 3 } + \mathbb { E } [ \Gamma ^ { \star } ] T ^ { 1 / 3 } )$ and $\widetilde { O } ( \sqrt { T } + \mathbb { E } [ \Gamma ^ { \star } ] )$ . For OP, default smoothing alone gives horizon-dependent coverage parameters; those factors must remain in the explicit bound.

Arbitrary exploration rate. As in Section 4, the unoptimized FE bound exposes the dependence on a constant exploration probability:

COROLLARY 5 (Unknown FE with shared estimation). Under Assumptions 1, 2, and 3, run Algorithm 2 with $B ^ { \mathrm { o p } } = B ,$ , fixed safe cyclic initialization length $t _ { \mathrm { i n i t . } }$ , constant $0 < \epsilon \le 1 / 2$ , and (49). Use the complete grid, uniform-advice master expert, and Appendix A.2’s common regularization and certified precision on the initialization and later forced samples. For a universal C and $\delta _ { \mathrm { t o t } } = \delta _ { \mathrm { t o t } } ^ { \mathrm { U , F E } }$

$$
\begin{array} { r l r } {  { \mathrm { R e g } ( T ) \le C \bigg [ h t _ { \mathrm { i n i t } } + \frac { h K ( n _ { 0 } + \ell ) } { \epsilon } + A _ { \mathrm { F E } } \sqrt { \frac { K T } { \epsilon } } + h \epsilon T } } \\ & { } & { + \mathrm { R e g } _ { \mathrm { m s t } } ( T ) + \mathcal { D } _ { T } + Z + \bigg ( 3 + \frac { K H _ { T } } { \kappa _ { \mathrm { F E } } \epsilon } \bigg ) \mathbb { E } [ \Gamma ^ { \star } ] \bigg ] + h \delta _ { \mathrm { t o t } } T . } \end{array}\tag{38}
$$

Choosing (50) gives (36), including its capped regime, by the calculation in Appendix B.6.

The calibration (50) balances clean learning, burn-in, and exploration without using the unknown corruption level. Appendix B.6 gives the calculation, including the capped regime.

## 6. Numerical Experiments

This section gives finite-sample illustrations of the four mechanisms emphasized by the theory: robustness with a supplied corruption bound, adaptation when the corruption level is unknown, the distinct roles of reward and consumption corruption, and the pathwise resource-accounting guarantee. The experiments use the same clean population-LP benchmark as the analysis and evaluate reward with the clean conditional means, so corrupted feedback affects the learner but not the benchmark. We focus on robustness as the corruption level varies rather than fitting an empirical asymptotic exponent; the latter would mix finite-horizon initialization, master, design, and logarithmic terms that are kept explicit in Corollaries 1 and 4.

## 6.1. Common environment and evaluation protocol

Three-specialist instance. The main experiments use $K = 3$ non-null actions, $m = 2$ resources, dimension $d = 2 0$ , true and calibration sparsity $s _ { 0 } = s = 3 $ , and parameter-radius bound $R = 1 . 2 $ . Budgets are $B = ( 0 . 5 T , 0 . 5 T )$ , hence $Z = 2$ . Draw $q _ { 1 } , q _ { 2 }$ independently and uniformly from $\{ - 1 , - 0 . 5 , 0 . 5 , 1 \}$ and $q _ { 3 } , z _ { 5 } , \dotsc , z _ { 2 0 }$ independently and uniformly from $\{ - 1 , + 1 \}$ , and set

$$
x = ( 0 . 5 0 , 0 . 4 0 q _ { 1 } , 0 . 4 0 q _ { 2 } , 0 . 2 0 q _ { 3 } , 0 . 1 5 z _ { 5 } , \ldots , 0 . 1 5 z _ { 2 0 } ) .\tag{39}
$$

Then $\| x \| _ { 2 } ^ { 2 } \leq 0 . 9 7$ . The clean reward means are

$$
\begin{array} { r } { r _ { 1 } ^ { 0 } ( x ) = 0 . 5 5 + 0 . 0 8 q _ { 1 } + 0 . 0 2 q _ { 3 } , } \\ { r _ { 2 } ^ { 0 } ( x ) = 0 . 5 5 - 0 . 0 8 q _ { 1 } + 0 . 0 2 q _ { 3 } , } \\ { r _ { 3 } ^ { 0 } ( x ) = 0 . 5 7 + 0 . 0 8 q _ { 2 } + 0 . 0 2 q _ { 3 } , } \end{array}\tag{40}
$$

and the clean mean consumptions are

$$
b _ { 1 } ^ { 0 } = ( 0 . 6 0 , 0 . 4 0 ) ^ { \top } , \qquad b _ { 2 } ^ { 0 } = ( 0 . 4 0 , 0 . 6 0 ) ^ { \top } , \qquad b _ { 3 } ^ { 0 } = ( 0 . 5 0 , 0 . 5 0 ) ^ { \top } .\tag{41}
$$

Independent reward and consumption noises are uniform on [−0.02, 0.02]. The exact finite-support population LP has value $V ^ { \mathrm { U B } } = 0 . 6 2 2 5 T$ , action masses $( 5 / 1 6 , 5 / 1 6 , 6 / 1 6 )$ , and resource use (0.5, 0.5); thus all three actions are clean-LP relevant and both resources bind in expectation. The minimum clean reward gap between the best and second-best action over the supported $( q _ { 1 } , q _ { 2 } )$ states is 0.02.

Algorithms and practical calibration. All headline policy comparisons use the on-policy mode from Sections 3.3 and 5. We use the norm-constrained Lasso implementation in Section 3.1 with fixed practical coefficients $c _ { \rho } = 0 . 2 5 , c _ { \beta } = 0 . 7 5 , c _ { C } = 1$ , unit dual multiplier, start count 20, and certified optimization gap $1 0 ^ { - 8 }$ . Shared-Grid uses the complete dyadic grid, the explicit uniform-advice expert, the EXP4.P specification of Section 5, and 201 safe cyclic initialization rounds (67 per non-null action). These settings are held fixed across the reported comparisons.

We compare ROPD with a valid supplied bound, a Reward-Only Robust ablation that protects reward confidence but not the consumption channel, and Non-Robust PD with zero corruption allowance. Shared-Grid receives no corruption bound. Attack maps are functions of the potential contexts and are fixed before the learner’s fresh action draw; compared methods receive the same contexts, clean potential outcomes, and exogenous attack map within each paired seed.

For a policy $\pi ,$ define the reported percentage Relative Regret by

$$
\mathrm { R e l R e g } ( \pi ) : = 1 0 0 \frac { V ^ { \mathrm { U B } } - \sum _ { t = 1 } ^ { T } r ^ { 0 } ( a _ { t } , x _ { t } ) } { V ^ { \mathrm { U B } } } ,\tag{42}
$$

where the null action contributes zero, including after stopping. Each reported cell uses 20 paired replications. Error bars and bands are 95% percentile bootstrap intervals from 2,000 whole-replication resamples; paired

Table 6 Theory-to-experiment map. The experiments are finite-sample illustrations of the mechanisms in the corresponding guarantees; empirical design and master diagnostics do not replace the stated probabilistic coverage conditions.
<table><tr><td>Experiment</td><td>Theory connection</td><td>Finite-sample question</td></tr><tr><td>Figure 1</td><td>Theorem 1, Corollary 1</td><td>How absolute regret changes as a valid known corruption level increases.</td></tr><tr><td>Figure 2</td><td>Theorem 2, Corollary 4</td><td>How much additional regret unknown corruption causes after accounting for each method&#x27;s clean finite-horizon cost.</td></tr><tr><td>Figure 3</td><td>Joint reward-consumption model and the Reward-Only ablation</td><td>How the location of a fixed corruption budget across reward and resource channels changes performance.</td></tr><tr><td>Figure 4</td><td>Theorems 1-2</td><td>Proposition 1 and the pathwise parts of How over- and under-reported consumption affect stopping and clean resource use.</td></tr></table>

differences and corrupted-minus-clean increments are formed within seed before resampling, and rounds are never treated as independent replicates. The final runs used in this section have no excluded seeds, no clipping, and no solver failures.

## 6.2. Known corruption: performance as $C _ { Z }$ increases

We first study the setting of Theorem 1, where the learner is given a deterministic valid bound $\Gamma \geq C _ { Z }$ . Fix $T = 2 0 { , } 0 0 0$ . Corruption targets Action 1 only on contexts where it is uniquely clean-preferred and applies the joint shift

$$
( \Delta r _ { 1 } , \Delta b _ { 1 , 1 } , \Delta b _ { 1 , 2 } ) = ( - 0 . 2 0 , + 0 . 1 0 , + 0 . 1 0 ) .
$$

With $Z = 2 ,$ , each attacked opportunity contributes 0.40 units to $C _ { Z }$ . We vary $C _ { Z } \in \{ 0 , 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 \}$ by attacking 0, 250, 500, 750, 1000 spread eligible opportunities.

Figure 1 reports absolute Relative Regret. All three methods coincide at $C _ { Z } = 0 ( \mathrm { m e a n } 0 . 7 5 6 \% )$ . ROPD has the lowest mean regret at every positive tested corruption level. At $C _ { Z } = 4 0 0$ , mean Relative Regret is 1.333% for ROPD, 1.383% for Reward-Only Robust, and 2.008% for Non-Robust PD. Thus the separation from the non-robust baseline becomes substantial as corruption grows, while the Reward-Only ablation remains between the full robust method and the non-robust policy throughout the main sweep. This is consistent with the corruption-dependent terms in Corollary 1: the corruption-aware policies trade wider confidence sets for reduced sensitivity to contaminated feedback, and protecting both channels is useful when resource feedback can also be corrupted.

## 6.3. Unknown corruption: adaptation on one shared history

We next remove the corruption-bound input and evaluate Shared-Grid from Algorithm 2. Because Shared-Grid pays its initialization and master-comparison costs even on clean data, the primary quantity here is the corruption-induced increase

$$
\Delta \mathrm { R e l R e g } _ { M } ( C _ { Z } ) : = \mathrm { R e l R e g } _ { M } ( C _ { Z } ) - \mathrm { R e l R e g } _ { M } ( 0 )\tag{43}
$$

![](images/1a2307f076dc44219f3517e812a6ee0cb9412f2ebd025da80989c0d0c6769d6f.jpg)  
Figure 1 Known-corruption robustness. At fixed $T = 2 0 , 0 0 0 \colon$ , the effective corruption level $C z$ increases along the horizontal axis. The vertical axis is absolute Relative Regret against the clean population-LP benchmark. Bands are the frozen 95% whole-replication bootstrap intervals.

for method M, formed within paired seeds. This isolates sensitivity to corruption from the method’s clean finite-horizon cost. Shared-Grid may still have larger absolute finite-horizon regret because it also pays initialization and master-comparison costs.

Fix $T = 8 0 { , } 0 0 0$ and vary the unknown corruption budget over $C _ { Z } \in \{ 0 , 1 0 0 0 , 2 0 0 0 , 3 0 0 0 , 4 0 0 0 \}$ . The same joint shift (−0.20, +0.10, +0.10) targets clean-preferred Action 1 opportunities; Shared-Grid is never given $C _ { Z }$ . Figure 2 shows that Shared-Grid incurs a smaller corruption-induced increase at every positive severity. At $C _ { Z } = 4 0 0 0$ , the increase is 1.313 percentage points for Shared-Grid versus 2.574 for Non-Robust PD. The paired Shared-minus-Non-Robust differences are −0.262, −0.770, −0.916, and −1.261 percentage points at $C _ { Z } = 1 0 0 0 , 2 0 0 0 , 3 0 0 0 , 4 0 0 0$ , respectively; each pointwise paired 95% interval lies below zero.

These results illustrate the role of the adaptive grid in Theorem 2: even without a supplied corruption level, the shared-data master is markedly less sensitive to increasing corruption on this instance. The result should be read together with the explicit master term in Corollary 4. In absolute terms Shared-Grid still pays a visible clean finite-horizon master cost; the experiment therefore supports adaptation to corruption rather than an absolute dominance claim over the non-robust policy.

## 6.4. Reward, consumption, and joint corruption

The preceding experiments vary the amount of corruption. We now fix its effective budget and vary where it enters the feedback. Set $T = 4 0 { , } 0 0 0$ and attack the same 1,200 clean-preferred Action 1 opportunities in every corrupted condition. The shifts are

$$
\begin{array} { r l } & { \mathrm { R e w a r d : } \quad ( - 0 . 2 0 , 0 , 0 ) , } \\ & { \mathrm { C o n s u m p t i o n : } ( 0 , + 0 . 1 0 , + 0 . 1 0 ) , } \\ & { \mathrm { J o i n t : } \quad \quad ( - 0 . 1 0 , + 0 . 0 5 , + 0 . 0 5 ) . } \end{array}
$$

![](images/85cbfe19f9dc107308b79c5bcea6c6b4c154584ef0469277f1fdc6b45d895d04.jpg)  
Re-rendered from frozen evidence; original 95% bootstrap intervals  
Figure 2 Unknown-corruption adaptation. At fixed $T = 8 0 , 0 0 0$ , the horizontal axis is the unknown effective corruption level $C _ { Z } ,$ , and the vertical axis is the paired corruption-induced increase in Relative Regret. Bands are 95% paired whole-replication bootstrap intervals.

Each attacked opportunity contributes 0.20 to $C _ { Z }$ , so all three conditions have exactly $C _ { Z } = 2 4 0$ , with the same target, attacked contexts, count, and timing within a paired seed.

Figure 3 reports absolute Relative Regret under the corrupted environments. ROPD has the lowest mean regret in all three channels: 0.416% under reward corruption, 0.907% under consumption corruption, and 0.655% under joint corruption. Reward-Only Robust attains 0.501%, 0.989%, and 0.738%, respectively, while Non-Robust PD attains 0.617%, 1.086%, and 0.840%. The ordering is consistent across the three equal- $. C _ { Z }$ corrupted environments and illustrates why consumption corruption is a separate modeling issue: it changes not only a regression target but also the resource-price and budget-accounting path. ROPD has the lowest absolute finite-horizon Relative Regret in each tested channel; the matched-clean incremental comparison is reported separately below and is more nuanced.

## 6.5. Pathwise resource accounting under corrupted consumption

Finally, we isolate the resource-accounting statement of Proposition 1. This controlled stress test uses the same problem scale $( K = 3 , m = 2 , d = 2 0 , s = 3 )$ and perturbs one frequently used action–resource pair by ±0.10. The goal is not a policy ranking: it is to make the two pathwise effects of consumption corruption visible.

When consumption is over-reported, the observed ledger is depleted too quickly and the safety rule can stop allocation early. Across increasing severities, mean lost rounds rise from zero to 341 at the largest tested perturbation. When consumption is under-reported, the observed ledger remains feasible while clean

![](images/cbdba59270309f99028c6fc766662f3c64f1eb21824b117cdfb6af5e4107482a.jpg)  
Re-rendered from frozen evidence; original 95% bootstrap intervals

Figure 3 Corruption channels. Reward, consumption, and joint attacks use the same 1,200 targeted opportunities and the same effective corruption budget $C z = 2 4 0$ at $T = 4 0 { , } 0 0 0$ . Bars show absolute Relative Regret. Error bars are the frozen 95% whole-replication bootstrap intervals.

resource use can exceed the nominal budget. At the largest severity, mean clean resource-1 excess is 361.243 and mean played consumption corruption is $C _ { b } = 3 6 3 . 6 6 5$ . More importantly, every trajectory satisfies

$$
\mathrm { V i o l } _ { \mathrm { o b s } } ( T ) = 0 , \qquad \mathrm { V i o l } _ { \mathrm { c l } } ( T ) \leq C _ { b } ( T ) \leq C _ { Z } / Z ,\tag{44}
$$

with no additional additive unit. Thus Figure 4 directly illustrates the distinction between observed feasibility and clean feasibility that motivates the resource component of the theory.

## 6.6. Additional checks and interpretation

The main experiments above measure finite-horizon performance under known corruption, unknown corruption, different feedback channels, and corrupted resource accounting. We report three additional checks to clarify the interpretation of these results.

Matched clean controls. For the known-corruption algorithms, supplying a positive bound Γ changes the confidence widths even when the observed feedback is clean. We therefore also compare each corrupted run with a clean run using the same supplied bound. This separates the effect of corrupted feedback from the finite-sample effect of the corruption-aware confidence allowance.

For the sweep in Figure 1, both robust methods have smaller matched corruption increments than Non-Robust PD at the two largest tested levels, $C _ { Z } = 3 0 0$ and 400. At $C _ { Z } = 4 0 0$ , the increments are 1.153 percentage points for ROPD, 1.113 for Reward-Only Robust, and 1.252 for Non-Robust PD. The difference between ROPD and Reward-Only is small in this matched comparison, so Figure 1 is interpreted primarily as an absolute finite-horizon performance comparison.

Resource accounting under selective consumption corruption (T = 20,000)

![](images/45e94c0eafaf865cb3f1b6b6417b953a37c6e361c667b63b0a7248a5229d4c35.jpg)

![](images/d74034c8ee2915516faa332413d0ebeb011fe59279f197841a210b71096e6778.jpg)  
Figure 4 Resource accounting under consumption corruption. Over-reported consumption advances stopping (left), while under-reported consumption can hide clean resource use (right). The diagonal in the right panel is the pathwise upper bound $\mathrm { V i o l } _ { \mathrm { c l } } ( T ) \le C _ { b } ( T )$ . Observed-budget violation is zero on every reported trajectory.

Table 7 Increase in Relative Regret relative to matched clean controls (percentage points). ROPD and Reward-Only Robust use clean controls with the same supplied Γ = 240; Non-Robust PD has no corruption-bound input.
<table><tr><td>Channel</td><td>ROPD</td><td>Reward-Only Robust</td><td>Non-Robust PD</td></tr><tr><td>Reward</td><td>0.0631</td><td>0.0673</td><td>0.0709</td></tr><tr><td>Consumption</td><td>0.5543</td><td>0.5551</td><td>0.5403</td></tr><tr><td>Joint</td><td>0.3019</td><td>0.3034</td><td>0.2939</td></tr></table>

The same control helps interpret Figure 3. With clean feedback and the same supplied Γ = 240, Relative Regret is 0.353% for ROPD and 0.434% for Reward-Only Robust, compared with 0.546% in the ordinary Γ = 0 clean setting. Table 7 reports the resulting channel-specific increments.

The matched-control comparison gives a more refined view of the channel effects. In the consumption and joint conditions, the three incremental regrets are numerically close; Non-Robust PD has slightly smaller increments than ROPD by 0.0140 and 0.0080 percentage points, respectively, in the paired comparisons. This does not conflict with the absolute comparison in Figure 3. The robust methods also have lower matched clean regret when supplied with $\Gamma = 2 4 0$ , and under the corrupted environments themselves ROPD attains the lowest Relative Regret in all three channels. Thus, Figure 3 summarizes finite-horizon performance under corruption, while the matched controls isolate the additional effect of corrupted feedback relative to each method’s corresponding clean baseline.

Horizon and design checks. A separate frozen horizon experiment uses $T \in \{ 4 0 , 0 0 0 , 8 0 , 0 0 0 , 1 6 0 , 0 0 0 \}$ with corruption of order $T ^ { 2 / 3 }$ . The corruption-induced increase for Shared-Grid decreases from 1.612 to 0.920 percentage points over these horizons, while the corresponding increase for Non-Robust PD decreases from 3.530 to 1.851. Clean and corrupted Relative Regret also decrease with T. These results provide a finite-horizon consistency check for the sublinear behavior in Corollaries 1 and 4, rather than an empirical estimate of their asymptotic exponents.

The experiment also has nondegenerate action-specific designs: all three specialist actions have positive clean LP mass, and their population second moments on the true reward supports are strictly positive. For Shared-Grid, candidate recommendations become meaningfully different under corruption; at $C _ { Z } = 4 0 0 0$ grid candidates disagree on about 61% of active post-initialization rounds. These diagnostics confirm that the finite-sample experiment exercises the learning and adaptation mechanisms studied in the theory.

Summary. The experiments illustrate four distinct parts of the analysis. With a supplied corruption bound, ROPD has lower absolute Relative Regret than the compared baselines throughout the tested corruption sweep. Without such a bound, Shared-Grid incurs substantially less corruption-induced degradation than Non-Robust PD across the tested unknown-corruption levels, while retaining its finite-horizon master cost. Across reward, consumption, and joint corruption, ROPD has the lowest absolute Relative Regret in the tested corrupted environments; the matched-clean incremental comparisons are more nuanced and are interpreted separately above. Finally, the accounting experiment directly illustrates exact observed feasibility and the corruption-controlled clean resource bound.

## 7. Conclusion and Limitations

We develop corruption-robust contextual allocation methods that account for both contaminated predictions and errors in resource accounting, pricing, and stopping. ROPD uses a supplied corruption bound, while Shared-Grid adapts confidence radii on one realized history. Both preserve observed budgets pathwise and control clean resource violation by consumption corruption. The estimator interface separates these allocation guarantees from the choice of sparse fitting method. Forced exploration provides one coverage route; the sharper on-policy rates require informative realized designs, with the additional recommendation-coverage condition for Shared-Grid. Obtaining less conservative calibration and weaker trajectory conditions remains a direction for further work.

## References

Alekh Agarwal, Haipeng Luo, Behnam Neyshabur, and Robert E Schapire. Corralling a band of bandit algorithms. In Conference on Learning Theory, pages 12–38. PMLR, 2017.

Shipra Agrawal and Nikhil Devanur. Linear contextual bandits with knapsacks. Advances in neural information processing systems, 29, 2016.

Shipra Agrawal and Nikhil R Devanur. Bandits with concave rewards and convex knapsacks. In Proceedings of the fifteenth ACM conference on Economics and computation, pages 989–1006, 2014.

Shipra Agrawal, Zizhuo Wang, and Yinyu Ye. A dynamic near-optimal algorithm for online linear programming. Operations Research, 62(4):876–890, 2014.

Shipra Agrawal, Nikhil R Devanur, and Lihong Li. An efficient algorithm for contextual bandits with knapsacks, and an extension to concave objectives. In Conference on Learning Theory, pages 4–18. PMLR, 2016.

Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. Journal ofthe ACM (JACM), 65(3):1–55, 2018.

Sivaraman Balakrishnan, Simon S Du, Jerry Li, and Aarti Singh. Computationally efficient robust sparse estimation in high dimensions. In Conference on Learning Theory, pages 169–212. PMLR, 2017.

Hamsa Bastani and Mohsen Bayati. Online decision making with high-dimensional covariates. Operations Research, 68(1):276–294, 2020.

Alina Beygelzimer, John Langford, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandit algorithms with supervised learning guarantees. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings ofMachine Learning Research, pages 19–26. PMLR, 2011.

Kush Bhatia, Prateek Jain, and Purushottam Kar. Robust regression via hard thresholding. Advances in neural information processing systems, 28, 2015.

Peter J Bickel, Ya’acov Ritov, and Alexandre B Tsybakov. Simultaneous analysis of lasso and dantzig selector. The Annals ofStatistics, 37(4):1705–1732, 2009. doi: 10.1214/08-AOS620.

Peter Bühlmann and Sara Van De Geer. Statistics for high-dimensional data: methods, theory and applications. Springer Science & Business Media, 2011.

Sunrit Chakraborty, Saptarshi Roy, and Ambuj Tewari. Thompson sampling for high-dimensional sparse linear contextual bandits. In International Conference on Machine Learning, pages 3979–4008. PMLR, 2023.

Yudong Chen, Constantine Caramanis, and Shie Mannor. Robust sparse regression under adversarial corruption. In International conference on machine learning, pages 774–782. PMLR, 2013.

Ashok Cutkosky, Christoph Dann, Abhimanyu Das, Claudio Gentile, Aldo Pacchiano, and Manish Purohit. Dynamic balancing for model selection in bandits and rl. In International Conference on Machine Learning, pages 2276–2285. PMLR, 2021.

Nikhil R Devanur and Thomas P Hayes. The adwords problem: online keyword matching with budgeted bidders under random permutations. In Proceedings ofthe 10th ACM conference on Electronic commerce, pages 71–78, 2009.

Ilias Diakonikolas, Daniel Kane, Sushrut Karmalkar, Eric Price, and Alistair Stewart. Outlier-robust high-dimensional sparse estimation via iterative filtering. Advances in Neural Information Processing Systems, 32, 2019.

Qin Ding, Cho-Jui Hsieh, and James Sharpnack. Robust stochastic linear contextual bandits under adversarial attacks. In International Conference on Artificial Intelligence and Statistics, pages 7111–7123. PMLR, 2022.

Dylan J Foster, Akshay Krishnamurthy, and Haipeng Luo. Model selection for contextual bandits. Advances in Neural Information Processing Systems, 32, 2019.

Avishek Ghosh, Abishek Sankararaman, and Kannan Ramchandran. Model selection for generic contextual bandits. arXiv preprint arXiv:2107.03455, 2021.

Anupam Gupta, Tomer Koren, and Kunal Talwar. Better algorithms for stochastic bandits with adversarial corruptions. In Conference on Learning Theory, pages 1562–1578. PMLR, 2019.

Botao Hao, Tor Lattimore, and Mengdi Wang. High-dimensional sparse linear bandits. Advances in Neural Information Processing Systems, 33:10753–10763, 2020.

Jiafan He, Dongruo Zhou, Tong Zhang, and Quanquan Gu. Nearly optimal algorithms for linear contextual bandits with adversarial corruptions. Advances in neural information processing systems, 35:34614–34625, 2022.

Nicole Immorlica, Karthik Sankararaman, Robert Schapire, and Aleksandrs Slivkins. Adversarial bandits with knapsacks. Journal ofthe ACM, 69(6):1–47, 2022.

Yue Kang, Cho-Jui Hsieh, and Thomas Chun Man Lee. Robust lipschitz bandits to adversarial corruptions. Advances in Neural Information Processing Systems, 36:10897–10908, 2023.

Sanath Kumar Krishnamurthy and Susan Athey. Optimal model selection in contextual bandits with many classes via offline oracles. Working paper, Stanford Graduate School of Business, 2021.

Yingkai Li, Edmund Y Lou, and Liren Shan. Stochastic linear optimization with adversarial corruption. arXiv preprint arXiv:1909.02109, 2019.

Fang Liu and Ness Shroff. Data poisoning attacks on stochastic bandits. In International Conference on Machine Learning, pages 4042–4050. PMLR, 2019.

Haolin Liu, Artin Tajdini, Andrew Wagenmaker, and Chen-Yu Wei. Corruption-robust linear bandits: Minimax optimality and gap-dependent misspecification. Advances in Neural Information Processing Systems, 37:24277– 24325, 2024.

Liu Liu, Yanyao Shen, Tianyang Li, and Constantine Caramanis. High dimensional robust sparse regression. In International Conference on Artificial Intelligence and Statistics, pages 411–421. PMLR, 2020.

Shang Liu, Jiashuo Jiang, and Xiaocheng Li. Non-stationary bandits with knapsacks. Advances in Neural Information Processing Systems, 35:16522–16532, 2022.

Thodoris Lykouris, Vahab Mirrokni, and Renato Paes Leme. Stochastic bandits robust to adversarial corruptions. In Proceedings of the 50th annual ACM SIGACT symposium on theory of computing, pages 114–122, 2018.

Wanteng Ma, Dong Xia, and Jiashuo Jiang. High-dimensional linear bandits with knapsacks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 34008–34037. PMLR, 2024.

Sahand N Negahban, Pradeep Ravikumar, Martin J Wainwright, and Bin Yu. A unified framework for high-dimensional analysis of m-estimators with decomposable regularizers. Statistical Science, 27(4):538–557, 2012. doi: 10.1214/ 12-STS400.

Aldo Pacchiano, Christoph Dann, Claudio Gentile, and Peter Bartlett. Regret bound balancing and elimination for model selection in bandits and rl. arXiv preprint arXiv:2012.13045, 2020.

Aldo Pacchiano, Christoph Dann, and Claudio Gentile. Best of both worlds model selection. Advances in Neural Information Processing Systems, 35:1883–1895, 2022.

Adarsh Prasad, Arun Sai Suggala, Sivaraman Balakrishnan, and Pradeep Ravikumar. Robust estimation via robust gradient estimation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(3):601–627, 2020.

Garvesh Raskutti, Martin J Wainwright, and Bin Yu. Restricted eigenvalue properties for correlated gaussian designs. The Journal ofMachine Learning Research, 11:2241–2259, 2010.

Zhimei Ren and Zhengyuan Zhou. Dynamic batch learning in high-dimensional sparse linear contextual bandits. Management Science, 70(2):1315–1342, 2024.

Karthik Abinav Sankararaman and Aleksandrs Slivkins. Bandits with knapsacks beyond the worst case. Advances in Neural Information Processing Systems, 34:23191–23204, 2021.

Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1):267–288, 1996.

Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

Chen-Yu Wei, Christoph Dann, and Julian Zimmert. A model selection approach for corruption robust reinforcement learning. In International Conference on Algorithmic Learning Theory, pages 1043–1096. PMLR, 2022.

Qingyuan Yu, Euijin Baek, Xiang Li, and Qiang Sun. Corruption-robust variance-aware algorithms for generalized linear bandits under heavy-tailed rewards. In The 41st Conference on Uncertainty in Artificial Intelligence, 2025.

## Appendix A: Lasso Confidence, Design Calibration, and Cumulative Widths

## A.1. Retained Histories and the Confidence Interface

The regression sample set, not the recommendation count, determines each fit. With $e _ { \tau }$ the independent forcedexploration indicator, the three retained histories are

$$
\begin{array} { r l } & { S _ { a } ^ { \mathrm { F E } } ( t ) = \{ \tau < t : I _ { \tau } e _ { \tau } = 1 , \ a _ { \tau } = a \} , } \\ & { S _ { a } ^ { \mathrm { O P } } ( t ) = \{ \tau < t : a _ { \tau } = a \} , } \\ & { S _ { a } ^ { \mathrm { s h , F E } } ( t ) = \{ \tau \leq t _ { \mathrm { i n i t } } : \tau < t , \ I _ { \tau } = 1 , \ a _ { \tau } = a \} \cup \{ t _ { \mathrm { i n i t } } < \tau < t : I _ { \tau } e _ { \tau } = 1 , \ a _ { \tau } = a \} . } \end{array}\tag{45}
$$

These sets are defined for $a \in { \mathcal { A } } ;$ let $N _ { a } ^ { \mathrm { F E } } ( t ) : = | S _ { a } ^ { \mathrm { F E } } ( t ) |$ and $N _ { a } ^ { \mathrm { e s t } } ( t ) : = | S _ { a } ^ { \mathrm { e s t } } ( t ) |$ for the selected mode. In OP it equals the realized non-null action count $N _ { a } ( t )$ ; in FE it generally does not. The candidate-recommendation count ${ \cal N } _ { a } ^ { \mathrm { r e c , F } } ( t )$ is a different object from either retained count.

## A.2. Regularization and Prediction Calibration

Section 3.1 defines the norm-constrained objective and states Proposition 2. Here we specify its inputs and calibration. Supply $R > 0$ , an integer max $\{ 1 , s _ { 0 } \} \leq s \leq d ,$ and noise bounds $\sigma _ { r } , \sigma _ { b } \ge 0$ . Bounded clean outcomes permit $\sigma _ { r } = \sigma _ { b } =$ $1 / 2$ by Hoeffding’s lemma. The fit does not require a support constraint; s is used for confidence and numerical-precision calibration. For one action and channel, $n , X , y ,$ and $\theta ^ { \star }$ have the local meanings given in Section 3.1; the corresponding estimate $\widehat { \theta }$ is $\widehat { \theta } _ { a , t - 1 }$ . At zero samples set it to zero. Exact fits select the minimum-Euclidean-norm minimizer, while approximate fits use the deterministic certified solver in Appendix A.8.

A design calibration consists of supplied $\kappa , \upsilon > 0$ and an integer $n _ { 0 } .$ . For failure levels $\delta _ { \mathrm { { e s t } } } , \delta _ { \mathrm { { c o v } } } \in ( 0 , 1 )$ and $n \geq 1$ , set

$$
\ell = \log \frac { 1 6 e K ( m + 1 ) d T } { \operatorname* { m i n } \{ \delta _ { \mathrm { e s t } } , \delta _ { \mathrm { c o v } } \} } , \quad \omega _ { n } = \frac { v \ell } { n } , \quad \rho _ { n } = 4 ( \sigma + R \sqrt { \kappa v } ) \sqrt { \frac { 2 \ell } { n } } , \quad n _ { 0 } \ge \left\lceil \frac { 2 5 6 s v \ell } { \kappa } \right\rceil .\tag{46}
$$

Use $\sigma = \sigma _ { r }$ or $\sigma _ { b }$ for the respective channel, writing $\rho _ { n } ^ { r }$ or $\rho _ { n } ^ { b }$ accordingly. The required restricted strong convexity (RSC) event is

$$
\frac { 1 } { n } \| X u \| _ { 2 } ^ { 2 } \geq \kappa \| u \| _ { 2 } ^ { 2 } - \omega _ { n } \| u \| _ { 1 } ^ { 2 } \quad \mathrm { f o r ~ e v e r y ~ } u \in \mathbb { R } ^ { d } \mathrm { ~ a n d ~ e v e r y ~ r e t a i n e d ~ p r e f i x ~ } n \geq n _ { 0 } .\tag{47}
$$

The tolerance is essential when $n < d .$ . The positive regularization floor in (46) controls corruption even if clean noise is zero. It depends on the design calibration, not on a candidate corruption bound.

Under this calibration, Proposition 2 gives simultaneous prediction on every unit-ball context. Its bound holds for all $C _ { n } \geq 0$ , with no corruption-dependent sample threshold, and permits an objective gap of $s \rho _ { n } ^ { 2 } / ( 8 \kappa )$ . Exact optimization replaces the coefficient 7 by 6.

Appendix A.5 proves the sequential noise bound, enlarged-cone argument, and large-corruption case. Appendix A.4 gives the resulting reward and resource radii. All candidates share the retained data, regularization, numerical-precision rule, and fitted coefficients.

Computation and information. Appendix A.8 gives a constrained proximal solver with a computable optimality-gap certificate for exactly (8). Calibration is a separate requirement: the learner uses model-supplied bounds justified in Appendix A.3, not an oracle for an unknown empirical RE constant. The on-policy guarantee remains conditional on supplied trajectory bounds. Convex fitting alone does not certify informative confidence for an arbitrary design. At most $( m + 1 ) T$ changed fits are needed; the grid adds candidate scoring, not candidate-specific fits.

## A.3. Population and On-Policy Design Conditions

Assumption 3 and Condition 4 instantiate the main-text interface for the estimator in Appendix A.2; they are not imposed on another estimator that already meets that interface. Recall $h = 1 + Z$ . For a calibrated pair $( \kappa , v )$ , define the clean-width coefficient

$$
A ( \kappa , v ) : = \frac { 2 8 \sqrt { 2 s \ell } } { \kappa } \left[ \sigma _ { r } + Z \sigma _ { b } + h R \sqrt { \kappa v } \right] , \qquad H _ { T } : = 1 + \log ( 1 + T ) .\tag{48}
$$

This coefficient includes noise, sparsity, norm, design, and numerical-accuracy costs. Horizon summaries use the price stepsize specified after (17).

Assumption 3 is stated in Section 3.2.

The condition imposes a lower bound in all population directions, rather than only in 2s-sparse directions. Under context-independent forced selection, Lemma 5 proves (47) simultaneously for retained prefixes with the conservative calibration

$$
D _ { x } = 4 0 9 6 L _ { x } ^ { 4 } / \kappa _ { x } ^ { 2 } , \quad \kappa _ { \mathrm { F E } } = \kappa _ { x } / 2 , \quad v _ { \mathrm { F E } } = 1 6 \kappa _ { \mathrm { F E } } D _ { x } , \quad n _ { 0 } = \lceil 4 0 9 6 s D _ { x } \ell \rceil .\tag{49}
$$

Write $A _ { \mathrm { F E } } = A ( \kappa _ { \mathrm { F E } } , v _ { \mathrm { F E } } )$ . These formulas do not require $n \geq d ,$ but their constants can be conservative. Since $\| x \| _ { 2 } \leq 1$ necessarily $\kappa _ { x } \leq 1 / d ;$ dimension dependence through curvature must not be suppressed. Appendix A.7 gives a bounded Rademacher-feature instance with $D _ { x }$ independent of d and explains its compatible nonnegative feedback model.

On-policy coverage. Retain all non-null observations and write $\begin{array} { r } { N _ { a } ( t ) = \sum _ { \tau < t } \mathbf { 1 } \{ a _ { \tau } = a \} } \end{array}$ . Let $\mathcal { E } _ { \mathrm { R S C } } ( \kappa _ { \mathrm { o p } } , v _ { \mathrm { o p } } , n _ { 0 } )$ denote (47) for every action-specific realized design with $N _ { a } ( t ) \geq n _ { 0 }$ , using $\omega _ { n } = v _ { \mathrm { o p } } \ell / n$ . Condition 4 in Section 3.2 states the required probability bound for this event.

Set $A _ { \mathrm { o p } } = A ( \kappa _ { \mathrm { o p } } , v _ { \mathrm { o p } } )$ . Population calibration does not imply this condition for adaptive actions: selection, budget stopping, and corruption can change each action’s realized design. A post-run diagnostic is not a learner-supplied lower bound, and a selected-support eigenvalue does not certify (47). Context-independent uniform actions with sufficient budget give a nonempty illustrative trajectory class, not a guarantee that our on-policy algorithm behaves that way (Appendix A.7). Candidate-recommendation coverage remains the separate Condition 5 for Shared-Grid OP.

Exploration calibration. For the concrete forced-exploration implementation, choose a constant probability

$$
\epsilon = \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , \operatorname* { m a x } \left\{ \left( \frac { A _ { \mathrm { F E } } ^ { 2 } K } { h ^ { 2 } T } \right) ^ { 1 / 3 } , \sqrt { \frac { K ( n _ { 0 } + \ell ) } { T } } \right\} \right\} .\tag{50}
$$

This balances clean statistical, burn-in, and exploration costs using supplied calibration only; it is independent of the corruption bound and is also used by Shared-Grid. The full bounds for arbitrary constant $0 < \epsilon \leq 1 / 2$ appear in Sections 4 and 5.

## A.4. Channel Radii and Score Confidence

All point estimates use (8) and (46). At zero samples the estimate is zero and both channel radii are one. For a supplied, justified calibration and $n = N _ { a } ^ { \mathrm { e s t } } ( t ) \geq n _ { 0 }$ , the concrete radii are

$$
\beta _ { a , t } ^ { r , \Gamma } = \operatorname* { m i n } \{ 1 , 7 \rho _ { n } ^ { r } \sqrt { s } / \kappa + 4 \Gamma / ( \kappa n ) \} ,\tag{51}
$$

$$
\beta _ { a , i , t } ^ { b , \Gamma } = \operatorname* { m i n } \{ 1 , 7 \rho _ { n } ^ { b } \sqrt { s } / \kappa + 4 \Gamma / ( Z \kappa n ) \} .\tag{52}
$$

Otherwise they are one. Null-action estimates and radii are zero. These prescriptions are defined even when a calibration event fails; confidence is asserted on that event, not by a data-dependent claim that an uncomputed RSC constant is positive.

Let $\beta ^ { r , 0 }$ and $\beta ^ { b , 0 }$ be the same formulas with $\Gamma = 0$ . The clean combined width and its corruption envelope obey

$$
\beta _ { a , t } ^ { Z , \Gamma } \leq f ( n ) + \frac { 8 \Gamma } { \kappa ( n \vee 1 ) } \mathbf { 1 } \{ n \geq n _ { 0 } \} ,\tag{53}
$$

$$
f ( n ) : = \left\{ { h , \qquad } \right. \qquad n < n _ { 0 } ,
$$

Here $f$ is nonincreasing, $h = 1 + Z$ , and A is defined in (48). The bound follows from min $\{ 1 , u + v \} \leq \operatorname* { m i n } \{ 1 , u \} + v$ and holds on any realized history, independently of confidence validity. Before burn-in, the entire radius is charged to the clean part. The nonnegative envelope $8 \Gamma / ( \kappa n )$ need not itself be clipped.

LEMMA 2 (From channel prediction to score confidence). On the noise-and-design event of Proposition 2, (10) holds simultaneouslyfor all valid Γ, andfor every $\lambda \in \Lambda$

$$
\begin{array} { r } { | \widehat { F } _ { a , t } ( \lambda ) - F ^ { 0 } ( a , x _ { t } ; \lambda ) | \leq \beta _ { a , t } ^ { Z , \Gamma } . } \end{array}\tag{54}
$$

Proof. The clean means are in [0, 1] because the clean outcomes are in that interval and their noise is conditionally mean-zero. Projection onto [0, 1] cannot increase distance to a clean mean. On a retained history,

$$
\sum _ { \tau \in S _ { a } ^ { \mathrm { e s t } } ( t ) } | c _ { \tau } ^ { r } ( a ) | \leq C _ { Z } \leq \Gamma , \qquad \sum _ { \tau \in S _ { a } ^ { \mathrm { e s t } } ( t ) } | c _ { \tau , i } ^ { b } ( a ) | \leq C _ { Z } / Z \leq \Gamma / Z .
$$

These inequalities concern observed-minus-clean deviations after clipping, not the raw perturbations. Apply Proposition 2 to each channel, including the trivial early radii, and use $\| \lambda \| _ { 1 } \leq 1$ . The same fits and noise event work for every valid Γ, including a path-dependent grid choice. No union bound over corruption values is needed.

## A.5. Sequential Noise and the Corrupted Lasso Basic Inequality

LEMMA 3 (Coordinate noise control for retained prefixes). For the pre-feedback retention rules in (45), with probability at least $1 - \delta _ { \mathrm { { e s t } } } ,$ every action, response channel, and obtained prefix of size $1 \leq n \leq T$ satisfies

$$
\| X ^ { \top } \eta / n \| _ { \infty } \leq \sigma \sqrt { 2 \ell / n } \leq \rho _ { n } / 2 .\tag{55}
$$

The claim remains valid for adaptively selected and budget-stopped $O P$ histories.

Proof. Fix an action, channel, and coordinate j. Let $Q _ { t }$ indicate that this action’s observation is retained. Conditional on the past, current context, and the learner’s fresh action/retention seed, $Q _ { t } x _ { t , j }$ is determined and the clean noise still has its stated conditional sub-Gaussian bound: the fresh seed is conditionally independent of the potential outcomes and corruption. The retention rule cannot inspect the current feedback. Thus, for every real u,

$$
\mathbb { E } \left[ \exp \{ u Q _ { t } x _ { t , j } \eta _ { t } - u ^ { 2 } \sigma ^ { 2 } Q _ { t } / 2 \} \mid \mathrm { p a s t } , \mathrm { c o n t e x t } , \mathrm { s e e d } \right] \leq 1 .
$$

Iterated conditioning makes the product an exponential supermartingale in calendar time. Stop it at the nth retained observation or at T, whichever comes first. On the event that the nth observation occurs and its score exceeds $b ,$ its value is at least exp $\left( u b - u ^ { 2 } \sigma ^ { 2 } n / 2 \right)$ . Optional stopping and Markov’s inequality, with $u = b / ( n \sigma ^ { 2 } )$ , give a tail at most $\exp [ - b ^ { 2 } / ( 2 n \sigma ^ { 2 } ) ]$ on this event. Apply both signs and take $b = \sigma { \sqrt { 2 n \ell } }$ . Union over $d K ( m + 1 ) T$ choices gives failure at most $2 d K ( m + 1 ) T e ^ { - \ell } \leq \delta _ { \mathrm { e s t } }$ . When $\sigma = 0 ,$ , the conditional sub-Gaussian property implies $\eta _ { t } = 0$ almost surely and the assertion is immediate. The supermartingale argument retains adaptive selection and stopping throughout.

LEMMA 4 (Shared-regularization Lasso under response corruption). Consider $y = X \theta ^ { \star } + \eta + \zeta$ , with row norms at most one, $\| { \boldsymbol { \theta } } ^ { \star } \| _ { 2 } \leq R ,$ , and $| J | \leq s f o r J = \operatorname { s u p p } ( \theta ^ { \star } )$ . Let $\widehat { \theta }$ be feasible for (8) and have objective suboptimality at most $\varepsilon _ { \mathrm { o p t } } \geq 0$ . Suppose

$$
Q ( u ) : = \| X u \| _ { 2 } ^ { 2 } / n \geq \kappa \| u \| _ { 2 } ^ { 2 } - \omega \| u \| _ { 1 } ^ { 2 } \quad f o r e \nu e r y u ,
$$

$$
\| X ^ { \top } \eta / n \| _ { \infty } \leq \rho / 2 , \qquad 0 < \omega \leq \kappa / ( 2 5 6 s ) , \qquad \rho \geq 4 R \sqrt { \kappa \omega } .
$$

Writing $C _ { n } = \| \zeta \| .$ , one has

$$
\| \widehat { \theta } - \theta ^ { \star } \| _ { 2 } \leq \operatorname* { m i n } \left\{ 2 R , \frac { 6 \rho \sqrt { s } } { \kappa } + \frac { 4 C _ { n } } { \kappa n } + 2 \sqrt { \frac { \varepsilon _ { \mathrm { o p t } } + 4 \omega \varepsilon _ { \mathrm { o p t } } ^ { 2 } / \rho ^ { 2 } } { \kappa } } \right\} .\tag{56}
$$

In particular, $\varepsilon _ { \mathrm { o p t } } \leq s \rho ^ { 2 } / ( 8 \kappa )$ ) gives (9). The error vector may be dense; the proof controls its $\ell _ { 1 }$ norm through the penalized basic inequality.

Proof. Put $\Delta = \widehat { \theta } - \theta ^ { \star } , r = \| \Delta \| _ { 2 } \leq 2 R$ , and $\gamma = C _ { n } / n$ . Feasibility of $\theta ^ { \star }$ and the objective-gap inequality yield the basic inequality

$$
\frac 1 2 Q ( \Delta ) \leq \frac { \eta ^ { \top } X \Delta } { n } + \frac { \zeta ^ { \top } X \Delta } { n } + \rho ( \| \theta ^ { * } \| _ { 1 } - \| \theta ^ { * } + \Delta \| _ { 1 } ) + \varepsilon _ { \mathrm { o p t } } \leq \frac { 3 \rho } { 2 } \| \Delta _ { J } \| _ { 1 } - \frac { \rho } { 2 } \| \Delta _ { J ^ { c } } \| _ { 1 } + \gamma r + \varepsilon _ { \mathrm { o p t } } .\tag{57}
$$

The corruption bound uses $| \langle x _ { j } , \Delta \rangle | \leq r$ , and the penalty difference uses sparsity of $\theta ^ { \star }$ . Optimality compares the penalized objectives, so the penalty difference remains in the basic inequality. Since $Q \geq 0$

$$
\| \Delta _ { J ^ { c } } \| _ { 1 } \leq 3 \| \Delta _ { J } \| _ { 1 } + \frac { 2 \gamma } { \rho } r + \frac { 2 \varepsilon _ { \mathrm { o p t } } } { \rho } , \qquad \| \Delta \| _ { 1 } \leq ( 4 \sqrt { s } + 2 \gamma / \rho ) r + 2 \varepsilon _ { \mathrm { o p t } } / \rho .\tag{58}
$$

Squaring with $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ gives

$$
\| \Delta \| _ { 1 } ^ { 2 } \leq ( 6 4 s + 1 6 \gamma ^ { 2 } / \rho ^ { 2 } ) r ^ { 2 } + 8 \varepsilon _ { \mathrm { o p t } } ^ { 2 } / \rho ^ { 2 } .
$$

First suppose $\gamma \le \rho \sqrt { \kappa / ( 6 4 \omega ) }$ . The two tolerance coefficients are each at most $\kappa / 4$ , so

$$
Q ( \Delta ) \ge \kappa r ^ { 2 } / 2 - 8 \omega \varepsilon _ { \mathrm { o p t } } ^ { 2 } / \rho ^ { 2 } .
$$

Combining with (57) and dropping its negative term gives

$$
\kappa r ^ { 2 } / 4 \leq ( 3 \rho \sqrt { s } / 2 + \gamma ) r + \varepsilon _ { \mathrm { o p t } } + 4 \omega \varepsilon _ { \mathrm { o p t } } ^ { 2 } / \rho ^ { 2 } .
$$

For $a r ^ { 2 } \leq b r + c \operatorname { w i t h } a > 0 , b , c \geq 0 , r \leq b / a + { \sqrt { c / a } } ,$ , proving the second bound in (56) in this case. $\mathrm { I f } \gamma > \rho \sqrt { \kappa / ( 6 4 \omega ) }$ then the regularization floor implies $\gamma > R \kappa / 2$ . Consequently $r \leq 2 R < 4 \gamma / \kappa ,$ , which proves the same bound without absorbing the enlarged cone. This case distinction uses the unknown true corruption only in analysis; neither the algorithm nor burn-in tests it. Finally, when $\varepsilon _ { \mathrm { o p t } } \leq s \rho ^ { 2 } / ( 8 \kappa )$ , one has $4 \omega \varepsilon _ { \mathrm { o p t } } / \rho ^ { 2 } \leq 1$ . The square-root term is at most $2 \sqrt { 2 \varepsilon _ { \mathrm { o p t } } / \kappa } \leq \rho \sqrt { s } / \kappa$ . Cauchy–Schwarz gives simultaneous prediction on every new unit-ball context. With exact optimization that term is zero.

ProofofProposition 2. Use Lemma 3. At every calibrated prefix, (46) implies $\omega _ { n } \leq \kappa / ( 2 5 6 s )$ and $\rho _ { n } \geq 4 R \sqrt { \kappa \omega _ { n } } .$ even when $\sigma { = } 0 . \mathrm { A p p l y }$ Lemma 4 on the actual retained design. The lemma holds for arbitrary retained corruption vectors and therefore simultaneously for all admissible corruption sums on that history.

## A.6. A Primitive RSC Calibration for Forced Samples

LEMMA 5 (Bounded sub-Gaussian design calibration). Under Assumptions 1 and 3, the FE retained histories, including a safe context-independent cyclic initialization, satisfy (47) with (49), simultaneously for all obtained prefixes $n \geq n _ { 0 } ,$ , with failure probability at most $\delta _ { \mathrm { c o v } } / 2 .$

Proof. Write $\Sigma = \mathbb { E } [ x x ^ { \top } ]$ . For a unit vector v, put $W = \langle v , x \rangle ^ { 2 }$ . The exponential-moment assumption gives $\mathbb { E } W ^ { p } \le 2 p ! L _ { x } ^ { 2 p }$ and $\begin{array} { r } { \mathbb E | W - \mathbb E W | ^ { p } \leq 2 ^ { p + 1 } p ! L _ { x } ^ { 2 p } } \end{array}$ for $p \geq 2 .$ . Expanding the exponential power series yields Bernstein’s bound

$$
\mathbb P \left( \left| \frac 1 n \sum _ { j = 1 } ^ { n } W _ { j } - \mathbb E W \right| > z , \ n \mathrm { o b t a i n e d } \right) \le 2 \exp \left( - \frac { n z ^ { 2 } } { 3 2 L _ { x } ^ { 4 } + 4 L _ { x } ^ { 2 } z } \right) .\tag{59}
$$

For independent samples this is the ordinary Chernoff calculation. For the actual retained samples the identical calculation uses the exponential supermartingale with predictable selection and stopping at the nth retained observation, as in Lemma 3. FE selection and cyclic initialization are independent of the current context; safety is determined before that context. Hence the conditional moment bound is unchanged by selection. No conditioning on the final count or on survival is needed.

Jensen’s inequality implies $\kappa _ { x } \leq L _ { x } ^ { 2 } \log 2 \leq L _ { x } ^ { 2 }$ . With $z = \kappa _ { x } / 8$ , the exponent in (59) is at least $n / D _ { x }$ for $D _ { x } =$ $4 0 9 6 L _ { x } ^ { 4 } / \kappa _ { x } ^ { 2 }$ . For each support of size q, a 1/4-net of its unit sphere has size at most $9 ^ { q }$ . Union over supports and the net implies

$$
| v ^ { \top } ( \widehat { \Sigma } _ { n } - \Sigma ) v | \leq ( \kappa _ { x } / 4 ) \| v \| _ { 2 } ^ { 2 } \quad \mathrm { f o r ~ a l l ~ } q \mathrm { - s p a r s e ~ } v
$$

except with probability at most $2 \exp [ - n / D _ { x } + q \log ( 9 e d / q ) ]$ ]. To see the factor two from the net explicitly, for the symmetric restriction M to any support, its operator norm is at most the maximum net quadratic form plus ${ \scriptstyle { \frac { 1 } { 2 } } } \left\| M \right\| _ { \mathrm { o p } }$

For each $n \geq 8 D _ { x } \ell ,$ let $k = \lfloor n / ( 8 D _ { x } \ell ) \rfloor$ and $q = \operatorname* { m i n } \{ 2 k , d \}$ . Since $\log ( 9 e d / q ) \leq \ell$ and $q \leq n / ( 4 D _ { x } \ell )$ , the preceding failure bound is at most $2 e ^ { - 3 n / ( 4 D _ { x } ) } \leq 2 e ^ { - 6 \ell }$ . If $q = d ,$ , the net result gives $\widehat { \Sigma } _ { n } \succeq 3 \kappa _ { x } I _ { d } / 4$ and the desired RSC follows. Otherwise partition any u into disjoint blocks of k coordinates, in decreasing absolute order. On the union of any two blocks the symmetric operator norm of $\widehat { \Sigma } _ { n } - \Sigma$ is at most $\kappa _ { x } / 4$ . Also

$$
\sum _ { j } \| u _ { J _ { j } } \| _ { 2 } \leq \| u \| _ { 2 } + \| u \| _ { 1 } / { \sqrt { k } } .
$$

Indeed, the norm of each block after the first is at most the preceding block’s $\ell _ { 1 }$ norm divided by ${ \sqrt { k } } .$ . Bilinearity and the two-block restriction therefore give

$$
\lvert u ^ { \top } ( \widehat { \Sigma } _ { n } - \Sigma ) u \rvert \leq \frac { \kappa _ { x } } { 4 } \left( \sum _ { j } \lVert u _ { J _ { j } } \rVert _ { 2 } \right) ^ { 2 } \leq \frac { \kappa _ { x } } { 2 } \lVert u \rVert _ { 2 } ^ { 2 } + \frac { \kappa _ { x } } { 2 k } \lVert u \rVert _ { 1 } ^ { 2 } .
$$

It follows that $u ^ { \top } \widehat { \Sigma } _ { n } u \geq \kappa _ { \mathrm { F E } } \Vert u \Vert _ { 2 } ^ { 2 } - \kappa _ { \mathrm { F E } } \Vert u \Vert _ { 1 } ^ { 2 } / k$ . The bound $k \geq n / ( 1 6 D _ { x } \ell )$ proves the asserted tolerance. Union over at most $K T$ obtained prefixes costs at most $2 K T e ^ { - 6 \ell } \le \delta _ { \mathrm { c o v } } / 2$ . The threshold $n _ { 0 } = \lceil 4 0 9 6 s D _ { x } \ell \rceil$ is above $8 D _ { x } \ell$ and satisfies the absorption requirement in (46).

The full population lower bound and all-direction sub-Gaussian envelope are used in the block-extension step above; positivity only on 2s-sparse directions would not justify that step. RSC with a tolerance is a standard framework for high-dimensional regularized estimation (Negahban et al. 2012); the selected-sample concentration and corruption calculation above supply the specific steps used here.

## A.7. A Nonempty Bounded-Context Instance

Let $x = ( 1 , \epsilon _ { 2 } , \ldots , \epsilon _ { d } ) / \sqrt { d } .$ , where the $\epsilon _ { j }$ are independent Rademacher signs. Then $\| x \| _ { 2 } = 1$ and $\mathbb { E } x x ^ { \top } = I _ { d } / d .$ One may take $\kappa _ { x } = 1 / d$ and $L _ { x } = 4 / { \sqrt { d } } .$ To verify the exponential envelope, write ${ \sqrt { d } } \langle u , x \rangle = u _ { 1 } + V$ , where $\begin{array} { r } { V = \sum _ { j \geq 2 } u _ { j } \epsilon _ { j } } \end{array}$ is sub-Gaussian with scale at most $\lVert u \rVert _ { 2 }$ . For $\alpha > 0$ and an independent standard normal g, the identity $\mathbb { E } _ { V , g } e ^ { \sqrt { 2 \alpha } g V } = \mathbb { E } _ { V } e ^ { \alpha V ^ { 2 } }$ implies $\mathbb { E } e ^ { V ^ { 2 } / ( 8 \| u \| _ { 2 } ^ { 2 } ) } \leq ( 1 - 1 / 4 ) ^ { - 1 / 2 }$ . Since $( u _ { 1 } + V ) ^ { 2 } \leq 2 u _ { 1 } ^ { 2 } + 2 V ^ { 2 }$ , the required expectation is at most $e ^ { 1 / 8 } ( 1 - 1 / 4 ) ^ { - 1 / 2 } < 2$ . Thus $D _ { x }$ is independent of d and the sufficient sample threshold is of order $s \ell ,$ albeit with conservative numerical constants. It can be smaller than d when $d / ( s \ell )$ is sufficiently large; no claim of a small practical burn-in follows from these constants.

For compatible sparse means, take $s _ { 0 } = s$ and choose $\theta ^ { \star } = \textstyle \sqrt { d } ( c e _ { 1 } + \sum _ { j = 2 } ^ { s } b _ { j } e _ { j } )$ with $0 < c - \sum | b _ { j } | \leq c + \sum | b _ { j } | < 1$ and $d ( c ^ { 2 } + \textstyle \sum b _ { j } ^ { 2 } ) \le R ^ { 2 }$ . Such choices exist for every $R > 0$ by scaling $c , b _ { j }$ down together. Add independent bounded mean-zero noise of amplitude less than the distance of the means to the endpoints of $[ 0 , 1 ]$ , separately for each channel. This gives sparse parameters with bounded, potentially context-dependent reward and consumption means. In this normalized class curvature is $1 / d ;$ the coefficient A must retain that dependence.

For illustration, context-independent uniform non-null sampling and budgets $B _ { i } \geq T + 1$ prevent safety stopping and give the same design event on all action histories. A time-uniform Chernoff bound gives $N _ { a } ( t ) \geq ( t - 1 ) / ( 2 K ) -$ $c \log ( K T / \delta )$ . Moreover, $N _ { a } ^ { \mathrm { r e c , T } } ( t ) \leq t - 1 \leq 2 K N _ { a } ( t ) + c K \log ( K T / \delta )$ for any recommendations. Thus this sampling regime provides a nonempty illustrative class with informative action-specific designs and recommendation coverage. It is not a guarantee that the proposed OP algorithms generate the same trajectories under corruption and budget stopping.

## A.8. A Reproducible Solver and Certified Numerical Error

For $n \geq 1$ , the following solver minimizes the constrained objective (8). Inputs are the retained matrix $X$ , clipped response vector $y \in [ 0 , 1 ] ^ { n }$ , R, and the channel-specific $\rho _ { n }$ . There is no unpenalized intercept and no automatic centering, feature standardization, or response rescaling. A constant feature, when used, is part of x and subject to both the penalty and norm bound. Any deliberate preprocessing must first be reflected in $R ,$ noise bounds, and design calibration.

Let $g ( \theta ) = \| y - X \theta \| _ { 2 } ^ { 2 } / ( 2 n )$ and $F ( \theta ) = g ( \theta ) + \rho _ { n } \| \theta \|$ . Its gradient is 1-Lipschitz because $\| X ^ { \top } X / n \| _ { \mathrm { o p } } \leq 1$ . Start at $\theta ^ { ( 0 ) } = 0$ and use step size one. For iteration $j$ set

$$
v = \theta ^ { ( j ) } - \nabla g ( \theta ^ { ( j ) } ) , \qquad z = \mathrm { s i g n } ( v ) ( | v | - \rho _ { n } ) _ { + } , \qquad \theta ^ { ( j + 1 ) } = z \operatorname* { m i n } \{ 1 , R / \lVert z \rVert _ { 2 } \} ,\tag{60}
$$

with the last expression interpreted as zero when $z = 0$ . The soft threshold and scaling together are the proximal map of $\rho _ { n } \| \cdot \| _ { 1 } + \iota _ { \{ \| \cdot \| _ { 2 } \leq R \} }$ , where the last term is the convex indicator (zero on the ball, +∞ outside). To check this, set $a = ( \Vert z \Vert _ { 2 } / R - 1 ) _ { + }$ , and choose $w _ { j } = \mathrm { s i g n } ( z _ { j } ) { \mathrm { ~ i f ~ } } z _ { j } \neq 0 .$ , and $w _ { j } = v _ { j } / \rho _ { i }$ otherwise. Then $w \in \partial \big | \big | \theta ^ { ( j + 1 ) } \big | \big | .$ <sub>1</sub>, $v - \theta ^ { ( j + 1 ) } = \rho _ { n } w + a \theta ^ { ( j + 1 ) }$ , and $a \big ( \big \| \theta ^ { ( j + 1 ) } \big \| _ { 2 } - R \big ) = 0$ , which are sufficient convex optimality conditions for the proximal problem.

At the new iterate form $v _ { F } = \nabla g ( \theta ^ { ( j + 1 ) } ) + \rho _ { n }$ w and the computable certificate

$$
G _ { j } = \langle v _ { F } , \theta ^ { ( j + 1 ) } \rangle + R \| v _ { F } \| _ { 2 } \ge F ( \theta ^ { ( j + 1 ) } ) - \operatorname* { m i n } _ { \| \theta \| _ { 2 } \le R } F ( \theta ) \ge 0 .\tag{61}
$$

The first inequality follows by minimizing the supporting affine minorant of F over the ball. Stop at the first certified $G _ { j } \le \varepsilon _ { \mathrm { o p t } , n } : = s \rho _ { n } ^ { 2 } / ( 8 \kappa )$ . This is a deterministic selection convention for approximate fits; it need not return the minimum-norm exact minimizer. The latter convention is well defined because the minimizer set is nonempty, compact,

and convex and the squared Euclidean norm is strictly convex. Measurability of the exact convention follows, for example, by taking limits of the unique minimizers of $F ( \theta ) + \epsilon \| \theta \| _ { 2 } ^ { 2 }$ on the ball as $\epsilon \downarrow 0$ . The finite algorithm uses measurable arithmetic and a first-passage stopping rule.

PROPOSITION 3 (Solver termination and precision accounting). In exact arithmetic the stopping rule above terminates. A sufficient upper bound on the iterations is $1 + \lceil 1 6 R ^ { 2 } / \varepsilon _ { \mathrm { o p t } , n } ^ { 2 } \rceil$ . Each iteration costs $O ( n d )$ arithmetic operations. The returned fit satisfies Proposition 2, including its numerical-error contribution.

Proof. Proximal descent and the 1-Lipschitz gradient give $\begin{array} { r } { F \big ( \theta ^ { ( j ) } \big ) - F \big ( \theta ^ { ( j + 1 ) } \big ) \geq \frac { 1 } { 2 } \| \theta ^ { ( j + 1 ) } - \theta ^ { ( j ) } \| _ { 2 } ^ { 2 } } \end{array}$ . Since $F ( 0 ) \leq$ $1 / 2$ and $F \geq 0 .$ , the sum of these squared steps is at most one. The proximal optimality equation implies

$$
\begin{array} { r } { \| v _ { F } + a \theta ^ { ( j + 1 ) } \| _ { 2 } \le \| \theta ^ { ( j ) } - \theta ^ { ( j + 1 ) } \| _ { 2 } + \| \nabla g ( \theta ^ { ( j + 1 ) } ) - \nabla g ( \theta ^ { ( j ) } ) \| _ { 2 } \le 2 \| \theta ^ { ( j + 1 ) } - \theta ^ { ( j ) } \| _ { 2 } . } \end{array}
$$

Using complementarity and $\| \theta ^ { ( j + 1 ) } \| _ { 2 } \leq R$ in (61) gives $G _ { j } \leq 4 R \| \theta ^ { ( j + 1 ) } - \theta ^ { ( j ) } \| _ { 2 }$ . If all of the first J certificates exceeded $\varepsilon _ { \mathrm { o p t } , n }$ , the squared-step sum would exceed $J \varepsilon _ { \mathrm { o p t } , n } ^ { 2 } / ( 1 6 R ^ { 2 } )$ . This is impossible for the stated J. The certificate implies the objective-gap condition in Lemma 4.

The iteration bound is conservative; it establishes certified termination rather than practical speed. For floating-point implementation, verify feasibility and evaluate an upper bound on (61) using directed rounding or explicitly bounded arithmetic residuals. An approximately valid subgradient with a certified Euclidean discrepancy e adds at most $2 R e$ to the certificate. Accept an iterate only when a certified upper bound on its Euclidean norm is at most R. A fixed iteration limit or small successive-step difference without these error bounds is not the certificate used in our guarantee. If the tolerance is not met, use trivial radii until a certified fit is available; the concrete rates assume the stated certificate is obtained.

For a different certified objective gap $\varepsilon _ { \mathrm { o p t } , a , t } .$ , the general additional prediction term is

$$
E _ { a , t } = 2 \sqrt { \left( \varepsilon _ { \mathrm { o p t } , a , t } + 4 \omega _ { n } \varepsilon _ { \mathrm { o p t } , a , t } ^ { 2 } / \rho _ { n } ^ { 2 } \right) / \kappa } .
$$

Apply this separately by channel and add twice the recommendation-weighted sum of $E ^ { r } + Z$ max $E _ { i } ^ { b }$ to the cumulative width interface. Under the prescribed tolerance these sums are already included in the coefficient 7 and hence in A; there is no omitted fixed numerical error accumulated over T rounds.

Full online cost and calibration. For a fixed channel, all corruption candidates use the same $X , y , \rho _ { n }$ and solver stopping criterion. Only an action receiving a newly retained observation requires a new fit; cached fits for other actions retain their certified objective gaps for the unchanged objectives, since $\rho _ { n }$ depends on $n ,$ not calendar time. There are at most $( m + 1 ) T$ changed fits. A conservative total arithmetic cost is

$$
O \left( ( m + 1 ) d \sum _ { a = 1 } ^ { K } \sum _ { n = 1 } ^ { N _ { a } ^ { \mathrm { e s t } } ( T + 1 ) } n \left[ 1 + \frac { R ^ { 2 } } { \varepsilon _ { \mathrm { o p t } , n } ^ { 2 } } \right] + T \{ K ( m + 1 ) d + K | \mathcal { G } | + m \} \right) ,
$$

using the largest channel-specific iteration bound in the first term. Computing all fitted scores costs $O ( K ( m + 1 ) d )$ per round; scoring the grid costs $O ( K | \mathcal { G } | )$ ), and the master and price updates add $O ( | \mathcal { G } | + K + m )$ . Storing retained raw samples costs $O ( T ( d + m ) ) ,$ ). No delayed refitting or unanalysed stale estimator is used. In FE, $\kappa _ { x } , L _ { x }$ and the model bounds are supplied and (49) is computed directly. In OP, the supplied pair must be justified for the actual trajectory by Condition 4; the method does not compute the smallest RE constant. Without justified bounds, trivial radii remain valid but these informative-width rates do not follow.

## A.9. Counting and Summing Statistical Widths

LEMMA 6 (Sequential scalar charging). For $\begin{array} { r } { z _ { t } \in [ 0 , 1 ] , Z _ { t } = \sum _ { u < t } z _ { u } , } \end{array}$ , and $S { = } \textstyle \sum _ { t } z _ { t }$ , one has

$$
\sum _ { t } \frac { z _ { t } } { Z _ { t } \vee 1 } \leq 2 [ 1 + \log ( 1 + S ) ] , \qquad \sum _ { t } \frac { z _ { t } } { \sqrt { Z _ { t } \vee 1 } } \leq 4 \sqrt { S } .\tag{62}
$$

Ifnonnegative integer counts $A _ { n }$ have cumulative sums $\textstyle \sum _ { j = 0 } ^ { n } A _ { j } \leq b + \chi ( n + 1 )$ , then every nonincreasing nonnegative sequence $f$ satisfies

$$
\sum _ { n = 0 } ^ { N } A _ { n } f ( n ) \leq b f ( 0 ) + \chi \sum _ { n = 0 } ^ { N } f ( n ) .\tag{63}
$$

Proof. For the first sum, the terms with $Z _ { t } < 1$ have total mass less than two and denominator one. For the remaining terms $Z _ { t } \geq 1$ and $z _ { t } / Z _ { t } \le 1$ , so $z _ { t } / Z _ { t } \le 2 \log ( 1 + z _ { t } / Z _ { t } )$ . These logarithms telescope from a starting cumulative mass at least one, giving a sum at most $2 \log ( 1 + S )$ ; the asserted bound follows. For the square-root sum,

$$
{ \sqrt { 1 + v + z } } - { \sqrt { 1 + v } } = { \frac { z } { \sqrt { 1 + v + z } + { \sqrt { 1 + v } } } } ,
$$

whose denominator is at most $4 \sqrt { v \vee 1 }$ . Telescoping gives at most $4 ( { \sqrt { 1 + S } } - 1 ) \leq 4 { \sqrt { S } }$ . Finally apply discrete summation by parts to the cumulative sums of $A _ { n }$ ; all differences $f ( n ) - f ( n + 1 )$ are nonnegative. Substitution of $b + \chi ( n + 1 )$ gives (63).

LEMMA 7 (Forced counts and cumulative widths). Let $0 < \epsilon \leq 1 / 2$ be constant. On an event offailure at most $\delta _ { \mathrm { c o v } } / 2 ,$ every active round with $\epsilon ( t - 1 ) \geq 8 K \ell$ satisfies

$$
N _ { a } ^ { \mathrm { F E } } ( t ) \geq \epsilon ( t - 1 ) / ( 2 K ) \quad ( a \in \mathcal { A } ) .\tag{64}
$$

Together with Lemma 5, this gives the estimator/coverage event of failure at most $\delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } }$ . For an arbitrary sequence ofrecommendations, the FE cumulative interface holds with a universal constant $C$ and

$$
\mathcal { R } _ { \mathrm { e s t } } ^ { \mathrm { 0 } } \leq C \left\lceil \frac { h K ( n _ { 0 } + \ell ) } { \epsilon } + A _ { \mathrm { F E } } \sqrt { \frac { K T } { \epsilon } } \right\rceil , \qquad \psi _ { \mathrm { e s t } } ( \Gamma ) \leq \frac { C K \Gamma H _ { T } } { \kappa _ { \mathrm { F E } } \epsilon } .\tag{65}
$$

For shared FE after initialization the same bounds hold over subsequent rounds, with the separate initialization cost charged in the allocation theorem.

Proof. Generate the independent forced coins and uniform draws for all calendar times, including unused draws after stopping. On the event that round t is active, all earlier rounds were active, so the actual pre-t forced count equals this latent count. A multiplicative Chernoff bound at mean $\epsilon ( t - 1 ) / K$ gives failure at most exp $- \epsilon ( t - 1 ) / ( 8 K ) ]$ ; union over $^ { a , }$ t and the condition $\epsilon ( t - 1 ) \geq 8 K \ell$ prove the first assertion. No distributional claim is made conditional on being active.

All actions have both the count bound and $n \geq n _ { 0 }$ by the deterministic calendar threshold

$$
t _ { \mathrm { c o v } } = \operatorname* { m i n } \left\{ T + 1 , 1 + \left\lceil \frac { 2 K ( n _ { 0 } + 4 \ell ) } { \epsilon } \right\rceil \right\} .\tag{66}
$$

Before this time charge at most 2h per active round. Afterwards (53) and (64) bound the recommended action’s statistical width by $A _ { \mathrm { F E } } \sqrt { 2 K / [ \epsilon ( t - 1 ) ] }$ ] and its corruption envelope by $1 6 K \Gamma / [ \kappa _ { \mathrm { F E } } \epsilon ( t - 1 ) ]$ . The interface contains this recommendation width with coefficient two. Summing $t ^ { - 1 / 2 }$ and $t ^ { - 1 }$ gives the conservative bound (65). If the threshold exceeds $T ,$ , the displayed burn-in term already bounds the whole horizon. For shared FE use elapsed post-initialization time in the count argument. Initialization samples can only increase that count, and Lemma 5 applies to the combined retained prefix; no comparison of two different fitted estimators is used.

LEMMA 8 (On-policy played-action widths). For the concrete on-policy module with the stated calibration, every $\Gamma \geq 0$ satisfies

$$
\sum _ { t = 1 } ^ { T } I _ { t } \beta _ { a _ { t } , t } ^ { Z , \Gamma } \leq C \left\{ h K n _ { 0 } + A _ { \mathrm { o p } } \sqrt { K T } + \frac { K \Gamma H _ { T } } { \kappa _ { \mathrm { o p } } } \right\} .
$$

Proof. Null actions have zero radius. For a fixed non-null action, its successive plays have pre-round retained counts $0 , 1 , \ldots , N _ { a } ( T + 1 ) - 1$ . Apply the envelope (53). The first $n _ { 0 }$ counts cost at most $h n _ { 0 } .$ . Subsequently, $\textstyle \sum _ { n = 1 } ^ { N } n ^ { - 1 / 2 } \leq$ $2 \sqrt { N }$ and $\begin{array} { r } { \sum _ { n = 1 } ^ { N } n ^ { - 1 } \le H _ { T } } \end{array}$ . Summing over actions and using $\begin{array} { r } { \sum _ { a } \sqrt { N _ { a } ( T + 1 ) } \le \sqrt { K T } } \end{array}$ proves the claim. The RSC and noise events justify the confidence radii; no comparison with an LP-policy sample count is needed.

PROPOSITION 4 (Known and shared on-policy cumulative interfaces). For known-corruption $O P ,$ the recommendation equals the played action, and the cumulative interface holds with

$$
\mathcal { R } _ { \mathrm { e s t } } ^ { \mathrm { 0 } } \leq C \{ h K n _ { 0 } + A _ { \mathrm { o p } } \sqrt { K T } \} , \qquad \psi _ { \mathrm { e s t } } ( \Gamma ) \leq \frac { C K \Gamma H _ { T } } { \kappa _ { \mathrm { o p } } } .\tag{67}
$$

For shared $O P ,$ on the coverage event of Condition ${ 5 , }$ uniformly over candidates the post-initialization cumulative interface holds with

$$
\begin{array} { c } { { \mathcal { R } _ { \mathrm { s h } } ^ { \scriptscriptstyle 0 } \leq C \{ h K ( n _ { \mathrm { c a n d } } + \chi n _ { \mathrm { 0 } } ) + \chi A _ { \mathrm { o p } } \sqrt { K T } \} , } } \\ { { \psi _ { \mathrm { e s t } } ( \Gamma ) \leq \displaystyle \frac { C K \Gamma } { \kappa _ { \mathrm { o p } } } \{ \chi H _ { T } + n _ { \mathrm { c a n d } } \} . } } \end{array}\tag{68}
$$

The noise and retained-design events additionally ensure that these radii provide the pointwise confidence interface.

Proof. The known case follows from twice Lemma 8, with constant factors absorbed into $C .$

For the shared case fix $a , \Gamma .$ , and let $A _ { n }$ count active post-initialization recommendations of a made when its pre-round realized count is n. At the time immediately after the last such recommendation with count at most $n ,$ Condition 5 gives

$$
\sum _ { j = 0 } ^ { n } A _ { j } \leq n _ { \mathrm { c a n d } } + \chi ( n + 1 ) , 
$$

because that recommendation round can increase the realized count by at most one. Apply (63) to the nonincreasing clean-width envelope $f .$ Its initial value is at most $h$ and

$$
\sum _ { n = 0 } ^ { N } f ( n ) \leq h ( n _ { 0 } + 1 ) + 2 A _ { \mathrm { o p } } { \sqrt { N } } .
$$

Sum through the final, possibly unplayed count level for each action, then use $\begin{array} { r } { \sum _ { a } \sqrt { N _ { a } ( T + 1 ) } \le \sqrt { K T } } \end{array}$ and $n _ { 0 } \geq 1$ This yields the stated clean-width bound after absorbing universal constants.

For the corruption part, dominate ${ \bf 1 } \{ n \ge n _ { 0 } \} / n$ by the nonincreasing sequence $1 / ( n \vee 1 )$ , whose partial sum is at most $2 H _ { T }$ . The same summation inequality gives a charge of order $n _ { \mathrm { c a n d } } + \chi H _ { T }$ per action. Multiplying by the corruption coefficient in (53), summing over actions, and absorbing universal constants proves the second line. The count event holds simultaneously for all candidates, so the conclusion is uniform on the shared history.

## A.10. Failure Probabilities and Concrete Substitution

For known FE combine the coordinate-noise event, the forced-design event, and the forced-count event; their failure is at most $\delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } }$ . For known $\mathrm { O P }$ combine noise with Condition 4. For shared $\mathrm { O P }$ add the candidate event. None of these events is obtained by conditioning the clean martingale increments on successful estimation. Instead, bound the score gap pathwise on the good event and by h on its complement. This produces the explicit $h \delta _ { \mathrm { t o t } } T$ expected-regret contribution. Mean-zero clean-feedback terms are removed using the original filtration before this event split, as in Appendix B.

Equations (65), (67), and (68) provide every input to the modular allocation theorems. They include the chosen solver’s numerical error through A, retain the actual curvature and tolerance dependence, and have no corruptiondependent burn-in. If κ is small or the calibrated threshold exceeds available data, the bound can be larger than $T ;$ the unconditional range bound $\mathrm { R e g } ( T ) \leq T$ still applies. Neither convexity nor an unsuccessful calibration implies sublinear regret.

## Appendix B: Proof of the Known-Corruption Guarantee

This appendix expands the main-text proof overview into five blocks: score confidence and comparison with the population LP; cumulative widths and overrides; dual control and insertion of the benchmark; the relation between clean and observed resource use and stopping cancellation; and final assembly with concrete substitutions. We use $B ^ { \mathrm { o p } } = B ,$ predictable estimates satisfying the main-text interface, clipped observed feedback, and the clean conditional-mean score $F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } )$ . The estimation objective enters only in the concrete substitutions at the end.

Let $I _ { t }$ indicate that the pre-action safety condition holds at the start of round t. Active rounds form a prefix because the algorithm plays null and freezes its state after the first failure. Let $\textstyle \tau : = \sum _ { t = 1 } ^ { T } I _ { t }$ . For the null action, set $\beta _ { \mathrm { n u l } 1 , t } ^ { Z , \Gamma } = 0$ $\widehat { F } _ { \mathrm { n u l 1 } , t } ( \lambda ) = 0 .$ , and $F ^ { 0 } ( { \tt n u l 1 } , x ; \lambda ) = 0$

Table 8 Notation used in the known-corruption proof.  
Symbol Meaning   
$I _ { t }$ Indicator that round t is active under the pre-action safety rule.   
$\tau$ Number of active rounds, $\textstyle { \tau = \sum _ { t } I _ { t } . }$   
$R _ { D }$ Dual regret against the best fixed resource-price vector in $\Lambda .$   
${ \mathcal E } _ { \mathrm { c o n f } }$ Common event supplying the pointwise confidence interface.   
$\mathcal { R } _ { \mathrm { e s t } } ^ { 0 }$ Clean cumulative recommendation-width bound supplied by the estimator interface.   
$\psi _ { \mathrm { e s t } } ( \Gamma )$ Additional cumulative width caused by retained response corruption.

## B.1. Score Confidence and Population-LP Comparison

LEMMA 9 (Score confidence). On the event ${ \mathcal E } _ { \mathrm { c o n f } } ,$ suppose the reward and consumption estimates satisfy the componentwise interface (10). Then every active round t, action $a \in A _ { 0 } ,$ , and price vector $\lambda _ { t } \in \Lambda$ satisfy

$$
\left| \widehat { F } _ { a , t } ( \lambda _ { t } ) - F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) \right| \leq \beta _ { a , t } ^ { Z , \Gamma } .\tag{69}
$$

Proof. For $a = \mathtt { n u l 1 }$ , both scores are zero. For $a \in A .$ apply the componentwise confidence interface directly. The reward error is at most $\beta _ { a , t } ^ { r , \Gamma }$ , and the priced consumption error is at most

$$
Z \sum _ { i = 1 } ^ { m } \lambda _ { t , i } \beta _ { a , i , t } ^ { b , \Gamma } \leq Z \| \lambda _ { t } \| _ { 1 } \operatorname* { m a x } _ { i } \beta _ { a , i , t } ^ { b , \Gamma } \leq Z \operatorname* { m a x } _ { i } \beta _ { a , i , t } ^ { b , \Gamma } .
$$

Adding the two terms gives $\beta _ { a , t } ^ { Z , \Gamma }$ , as stated.

## B.1.1. Optimistic comparison with the population-LP policy

LEMMA 10 (Optimistic comparison). On ${ \mathcal E } _ { \mathrm { c o n f } } ,$ every active round satisfies

$$
F ^ { 0 } ( \widehat { a } _ { t } , x _ { t } ; \lambda _ { t } ) \geq \sum _ { a \in \mathcal { A } } y ^ { \star } ( a  { | } x _ { t } ) F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) - 2 \beta _ { \widehat { a } _ { t } , t } ^ { Z , \Gamma } .\tag{70}
$$

$\mathnormal { I f e } _ { t }$ is theforced-exploration indicator in Algorithm 1, the played action satisfies

$$
{ \cal F } ^ { 0 } ( a _ { t } , x _ { t } ; \lambda _ { t } ) \geq \sum _ { a \in \mathcal { A } } y ^ { \star } ( a \mid x _ { t } ) { \cal F } ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) - 2 \beta _ { \hat { a } _ { t } , t } ^ { Z , \Gamma } - ( 1 + Z ) e _ { t } .\tag{71}
$$

Proof. Score confidence and maximization imply, for every $a \in A _ { 0 }$

$$
F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) \leq U _ { a , t } ^ { \Gamma } \leq U _ { \widehat { a } _ { t } , t } ^ { \Gamma } \leq F ^ { 0 } ( \widehat { a } _ { t } , x _ { t } ; \lambda _ { t } ) + 2 \beta _ { \widehat { a } _ { t } , t } ^ { Z , \Gamma } .
$$

The last expression is nonnegative because it is at least $U _ { \widehat { a } _ { t } , t } ^ { \Gamma } \geq U _ { \mathrm { n u l l } , t } ^ { \Gamma } = 0$ . Extend $y ^ { \star } ( \cdot | x _ { t } )$ to a distribution on $\mathcal { A } _ { 0 }$ by assigning its remaining mass to null. Multiplying the action-wise inequality by this distribution and summing proves (70), since the null clean score is zero.

A forced action can reduce the clean score by at most $1 + Z ,$ , because all clean scores lie in $[ - Z , 1 ]$ ; when $e _ { t } = 0$ , the played action equals the recommendation. This proves (71) pathwise.

## B.2. Cumulative Widths and Overrides

Define the cumulative clean-score gap on active rounds

$$
\mathcal { E } _ { \mathrm { o p t } } : = \sum _ { t = 1 } ^ { T } { I _ { t } \left[ \sum _ { a \in \mathcal { A } } { y ^ { \star } ( a \mid x _ { t } ) F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) } - F ^ { 0 } ( a _ { t } , x _ { t } ; \lambda _ { t } ) \right] } .\tag{72}
$$

LEMMA 11 (Cumulative clean-score gap). Under the cumulative estimator interface, with joint estimator and coverage failure probability at most $\delta _ { \mathrm { t o t } }$

$$
\mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ] \le \mathcal { R } _ { \mathrm { e s t } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma ) + ( 1 + Z ) \sum _ { t = 1 } ^ { T } \epsilon _ { t } + ( 1 + Z ) \delta _ { \mathrm { t o t } } T .\tag{73}
$$

Here $\mathcal { R } _ { \mathrm { e s t } } ^ { 0 }$ is the clean cumulative-width term, while $\psi _ { \mathrm { e s t } } ( \Gamma )$ is the extra width caused by corrupted retained samples.

Proof. Sum (71) over active rounds and apply the cumulative estimator interface (12) directly. It bounds the width sum by $\mathcal { R } _ { \mathrm { e s t } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma )$ ; no estimator-specific support property is needed here. For the concrete module, (53) charges pre-certification radii to clean burn-in. Lemma 7 in FE and Proposition 4 in OP provide the concrete interface bounds. Finally, $I _ { t }$ is fixed before the fresh exploration coin is drawn, so

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } I _ { t } e _ { t } \right] \leq \sum _ { t = 1 } ^ { T } \epsilon _ { t } .
$$

On the complement of the joint estimator–coverage event, each clean Lagrangian comparison is at most $1 + Z$ . Its contribution is at most $( 1 + Z ) \delta _ { \mathrm { t o t } } T$ . The clean-to-observed resource discrepancy enters later through Lemma 14, separate from the estimator term.

## B.3. Dual Control and Insertion of the LP Benchmark

On an active round, let

$$
g _ { t } : = b _ { t } ( a _ { t } ) - \frac { B } { T } , \qquad \widetilde { g } _ { t } : = ( g _ { t } , 0 ) \in \mathbb { R } ^ { m + 1 } .
$$

Define

$$
R _ { D } : = \operatorname* { m a x } _ { \lambda \in \Lambda } \sum _ { t = 1 } ^ { T } I _ { t } \lambda ^ { \top } g _ { t } - \sum _ { t = 1 } ^ { T } I _ { t } \lambda _ { t } ^ { \top } g _ { t } .\tag{74}
$$

LEMMA 12 (Resource-price update bound). The multiplicative-weights update in Algorithm 1 satisfies

$$
R _ { D } \leq \frac { \log ( m + 1 ) } { \eta _ { q } } + \eta _ { q } \left( 1 + \frac { B _ { \operatorname* { m a x } } } { T } \right) ^ { 2 } T .\tag{75}
$$

So $Z R _ { D } \leq { \mathcal { D } } _ { T }$ for the definition in (17).

Proof. The first m coordinates of $q _ { t }$ equal $\lambda _ { t } .$ , and the last coordinate is the unused mass $1 - \| \lambda _ { t } \| _ { 1 }$ <sub>1</sub>. Every $\lambda \in \Lambda$ corresponds to $q = ( \lambda , 1 - \| \lambda \| _ { 1 } )$ in the (m + 1)-simplex. Also,

$$
\| \widetilde { g } _ { t } \| _ { \infty } \leq 1 + \frac { B _ { \operatorname* { m a x } } } { T } .
$$

Hoeffding’s lemma for the finite distribution $q _ { t }$ bounds its log-moment-generating function by its mean plus $\eta _ { q } ^ { 2 } \| \widetilde { g } _ { t } \| _ { \infty } ^ { 2 } / 2 .$ Summing the log-potential increments yields, for every such q and any $\eta _ { q } > 0$

$$
\sum _ { t = 1 } ^ { T } { I _ { t } ( q - q _ { t } ) ^ { \top } \widetilde { g } _ { t } \leq \frac { \mathrm { K L } ( q \| q _ { 1 } ) } { \eta _ { q } } + \eta _ { q } \sum _ { t = 1 } ^ { T } { I _ { t } \| \widetilde { g } _ { t } \| _ { \infty } ^ { 2 } } } .
$$

The algorithm freezes $q _ { t }$ after stopping, so this is exactly the update on the active prefix. Since $q _ { 1 }$ is uniform, $\begin{array} { r } { \mathrm { K L } ( q \| q _ { 1 } ) \leq } \end{array}$ $\log ( m + 1 )$ . Taking the maximum over λ proves (75).

## B.3.1. Insertion of the population-LP benchmark

LEMMA 13 (Population-LP conversion). The cumulative clean-score gap on active rounds satisfies

$$
\begin{array} { r l } & { \mathrm { R e g } ( T ) \leq \mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ] + V ^ { \mathrm { U B } } \mathbb { E } \Big [ 1 - \frac { \tau } { T } \Big ] } \\ & { \quad \quad \quad + Z \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \lambda _ { t } ^ { \top } \left( \frac { B } { T } - b ^ { 0 } ( a _ { t } , x _ { t } ) \right) \right] . } \end{array}\tag{76}
$$

Proof. Because active rounds form a prefix, $I _ { t }$ is measurable before $x _ { t }$ is drawn. Optimality of $y ^ { \star }$ gives

$$
V ^ { \mathrm { U B } } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } I _ { t } \sum _ { a \in \mathcal { A } } y ^ { \star } ( a \mid x _ { t } ) r ^ { 0 } ( a , x _ { t } ) \right] + V ^ { \mathrm { U B } } \mathbb { E } \left[ 1 - \frac { \tau } { T } \right] .\tag{77}
$$

Expanding (72), using $( 7 7 )$ , and then applying population-LP feasibility yield (76). The key measurability point is that $I _ { t } \lambda _ { t }$ is predictable before $x _ { t }$ . For every resource coordinate,

$$
{ \mathbb E } _ { x } \left[ \sum _ { a } y ^ { \star } ( a \mid x ) b _ { i } ^ { 0 } ( a , x ) \right] \leq \frac { B _ { i } } { T }
$$

can be multiplied by $I _ { t } \lambda _ { t , i }$ and averaged without changing its direction. This step is where population-LP feasibility enters the proof.

## B.4. Relating Resource Use and Controlling Stopping

The population LP uses clean conditional-mean consumption, while the algorithm updates prices and stops using observed consumption. This step connects the two resource processes.

LEMMA 14 (Relating clean and observed resource use). The cumulative clean-score gap on active rounds satisfies

$$
\begin{array} { r l } & { \displaystyle \mathrm { R e g } ( T ) \leq \mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ] + { V } ^ { \mathrm { U B } } \mathbb { E } \Big [ 1 - \frac { \tau } { T } \Big ] } \\ & { \quad \quad \quad \quad + Z \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \lambda _ { t } ^ { \top } \left( \frac { B } { T } - b _ { t } ( a _ { t } ) \right) \right] + \mathbb { E } [ C _ { Z } ] . } \end{array}\tag{78}
$$

Proof. Because $I _ { t } \lambda _ { t }$ is predictable before $x _ { t }$ and $a _ { t }$ is selected before the current feedback using randomness independent of the clean potential outcomes, the conditional mean-zero clean noise has zero expectation. Together with $b _ { t } ( a _ { t } ) = b _ { t } ^ { \mathrm { c l } } ( a _ { t } ) + c _ { t } ^ { b } ( a _ { t } )$ , this implies

$$
\begin{array} { r l } & { Z \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \lambda _ { t } ^ { \top } \left( b _ { t } ( a _ { t } ) - b ^ { 0 } ( a _ { t } , x _ { t } ) \right) \right] } \\ & { \quad = Z \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \lambda _ { t } ^ { \top } c _ { t } ^ { b } ( a _ { t } ) \right] \leq Z \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } I _ { t } \| c _ { t } ^ { b } ( a _ { t } ) \| _ { \infty } \right] \leq \mathbb { E } [ C _ { Z } ] . } \end{array}\tag{79}
$$

Equivalently, pathwise the corruption part is controlled by

$$
Z \sum _ { t = 1 } ^ { T } I _ { t } \left| \lambda _ { t } ^ { \top } \left( b _ { t } ( a _ { t } ) - b _ { t } ^ { \mathrm { c l } } ( a _ { t } ) \right) \right| \leq C _ { Z } .
$$

Substituting the expected inequality into (76) proves (78). This produces one direct $C _ { Z }$ resource-accounting term, separate from the estimator widths.

## B.4.1. Stopping cancellation

LEMMA 15 (Stopping cancellation and expected BwK conversion). Under Assumption 2,

$$
\mathrm { R e g } ( T ) \leq \mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ] + Z \mathbb { E } [ R _ { D } ] + \mathbb { E } [ C _ { Z } ] + Z .\tag{80}
$$

Proof. Let

$$
G : = \sum _ { t = 1 } ^ { T } I _ { t } \left( b _ { t } ( a _ { t } ) - \frac { B } { T } \right) .
$$

The definition of $R _ { D }$ keeps the following maximum:

$$
Z \sum _ { t = 1 } ^ { T } I _ { t } \lambda _ { t } ^ { \top } \left( \frac { B } { T } - b _ { t } ( a _ { t } ) \right) \le Z R _ { D } - Z \operatorname* { m a x } _ { \lambda \in \Lambda } \lambda ^ { \top } G .\tag{81}
$$

The maximum selects the most depleted resource direction and supplies the charge used for the benchmark value after stopping.

If $\tau = T$ , there is no inactive benchmark value and the nonnegative maximum may be dropped. If $\tau < T$ , then the pre-action test first fails after $\tau$ active rounds. For some resource $i ,$

$$
U _ { \tau + 1 , i } = \sum _ { t = 1 } ^ { T } I _ { t } b _ { t , i } ( a _ { t } ) > B _ { i } - 1 ,
$$

and so

$$
G _ { i } = U _ { \tau + 1 , i } - \frac { \tau B _ { i } } { T } > B _ { i } \left( 1 - \frac { \tau } { T } \right) - 1 .\tag{82}
$$

Since both 0 and the coordinate vector $e _ { i }$ belong to Λ,

$$
\operatorname* { m a x } _ { \lambda \in \Lambda } \lambda ^ { \top } G \geq \operatorname* { m a x } \{ 0 , G _ { i } \} .
$$

Assumption 2 gives $V ^ { \mathrm { U B } } \leq Z B _ { \operatorname* { m i n } } \leq Z B _ { i \cdot } \mathrm { I f } \ G _ { i } \geq 0$ , then (82) implies

$$
V ^ { \mathrm { U B } } \left( 1 - \frac { \tau } { T } \right) - Z \operatorname* { m a x } _ { \lambda \in \Lambda } \lambda ^ { \top } G < Z .
$$

If $G _ { i } < 0$ , the same display follows because (82) implies $B _ { i } ( 1 - \tau / T ) < 1$ , while the maximum is nonnegative. In both cases,

$$
V ^ { \mathrm { U B } } \left( 1 - \frac { \tau } { T } \right) - Z \operatorname* { m a x } _ { \lambda \in \Lambda } \lambda ^ { \top } G \leq Z .\tag{83}
$$

Taking expectations in the two pathwise inequalities and combining (78), (81), and (83) proves the stopping-aware conversion. The remaining Z is the stopping boundary. The pre-action rule already gives exact observed budget feasibility.

## B.5. Assembly and Concrete Substitutions

Proof of Theorem 1. Lemma 11 gives

$$
\mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ] \le \mathcal { R } _ { \mathrm { e s t } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma ) + ( 1 + Z ) \sum _ { t = 1 } ^ { T } \epsilon _ { t } + ( 1 + Z ) \delta _ { \mathrm { t o t } } T .
$$

Lemma 12 gives $Z \mathbb { E } [ R _ { D } ] \leq D _ { T }$ , and Lemma 15 gives the direct resource transfer and the stopping boundary. Thus, the assembled bound is

$$
\begin{array} { r l } & { \boxed { \mathrm { R e g } ( T ) \le O \left( \mathcal { R } _ { \mathrm { e s t } } ^ { 0 } + ( 1 + Z ) \sum _ { t = 1 } ^ { T } \epsilon _ { t } + \mathcal { D } _ { T } + \mathbb { E } [ C _ { Z } ] + \psi _ { \mathrm { e s t } } ( \Gamma ) + Z \right) } } \\ & { \qquad + ( 1 + Z ) \delta _ { \mathrm { t o t } } T . } \end{array}\tag{84}
$$

Finally, the supplied bound is pathwise, $C _ { Z } \le \Gamma$ , and therefore $\mathbb { E } [ C _ { Z } ] \le \Gamma$ . Replacing the explicit $\mathbb { E } [ C _ { Z } ]$ in (84) by Γ proves (20). The final display keeps the direct transfer Γ separate from the estimator term $\psi _ { \mathrm { e s t } } ( \Gamma )$

Lemma 7 gives (65). Substituting it into the assembled bound proves (23) and Corollary $2 ;$ the term hϵT is already present in that bound. The exploration balance in Appendix B.6, including its range-bound argument for the capped case, then gives (21) and Corollary 1(i).

For OP, the recommendation is the actual played action and the regression count is $N _ { a } ( t )$ . Substituting (67) from Proposition 4 proves (22) and Corollary 1(ii). These substitutions retain the explicit $A , n _ { 0 } .$ , curvature, and harmonic factors. Setting $\Gamma = 0$ proves Corollary 3.

If the algorithm instead uses a strictly smaller operating budget, the same five proof blocks apply after replacing B by $B ^ { \mathrm { o p } }$ , using a scale valid for the reduced-budget benchmark, and measuring regret first against $V ^ { \mathrm { U B } } ( B ^ { \mathrm { o p } } )$ . Returning to the original benchmark adds exactly $V ^ { \mathrm { U B } } ( B ) - V ^ { \mathrm { U B } } ( B ^ { \mathrm { o p } } )$

## B.6. Exploration Calibration for the Concrete FE Bounds

Exploration calibration. For clarity about the FE choice, put $a = A _ { \mathrm { F E } } \sqrt { K T }$ and $b = h K ( n _ { 0 } + \ell )$ . The three clean terms are $b / \epsilon + a / \sqrt { \epsilon } + h T \epsilon .$ . If the cap in (50) is inactive, its two lower bounds imply

$$
\frac { b } { \epsilon } \le \sqrt { b h T } , \qquad \frac { a } { \sqrt { \epsilon } } \le a ^ { 2 / 3 } ( h T ) ^ { 1 / 3 } , \qquad h T \epsilon \le \sqrt { b h T } + a ^ { 2 / 3 } ( h T ) ^ { 1 / 3 } .
$$

This proves the $T ^ { 2 / 3 }$ term together with the square-root burn-in contribution in (21). For the capped case, write

$$
r _ { 1 } = \left( \frac { A _ { \mathrm { F E } } ^ { 2 } K } { h ^ { 2 } T } \right) ^ { 1 / 3 } , \qquad r _ { 2 } = \sqrt { \frac { K ( n _ { 0 } + \ell ) } { T } } .
$$

If max $\{ r _ { 1 } , r _ { 2 } \} \ge 1 / 2$ , then the sum of the two clean terms in the main bound is $h T ( r _ { 1 } + r _ { 2 } ) \geq h T / 2 \geq T / 2$ . The unconditional range bound $\mathrm { R e g } ( T ) \leq T$ therefore proves the same main-text inequality after choosing $C \geq 2 .$ . Thus it is valid for all horizons, not only the uncapped regime; no informative rate is asserted when these terms are of order T. The argument also proves (36), since its other terms are nonnegative. The rule balances clean costs only and is implementable with either known or unknown corruption; the corruption term is evaluated at the chosen ϵ.

## Appendix C: Proof of the Unknown-Corruption Guarantee

The proof compares recommendations computed from one common realized estimator history. It does not compare counterfactual histories that different candidates would have generated. Direct corruption enters twice when transferring the master comparison to clean scores, and once in the resource conversion. The cumulative confidence term is separate.

## C.1. One Common Confidence and Width Event

LEMMA 16 (Shared estimates and all valid grid points). For the concrete module of Appendices A.2–A.3, under the FE calibration or the OP trajectory condition, one event of failure at most $\delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } }$ gives pointwise confidence for every valid $\Gamma \in { \mathcal { G } } .$ . On this event the FE grid-uniform width interface holds with (65). For OP, intersect also with Condition 5’s event; then (68) supplies that interface, with total failure at most $\delta _ { \mathrm { e s t } } + \delta _ { \mathrm { c o v } } + \delta _ { \mathrm { c a n d } }$

Proof. The estimator sample set (27) is exactly the initialization-augmented FE or all-realized $\mathrm { O P }$ set in (45). Lemma 3 covers its obtained prefixes, including stopping. In FE, Lemma 5 applies directly to the combined initialization and later forced samples, and Lemma 7 supplies the count event. In OP, the retained design is precisely the design in Condition 4. For every response channel, the data, $\rho _ { n } .$ , and certified objective-gap tolerance are common to all candidates. Lemma 4 is deterministic for any corruption vector on the noise event. Therefore Lemma 2 covers all valid candidates simultaneously, without a candidate-dependent noise theorem or a candidate-dependent fit. The FE width bound holds for arbitrary recommendations. The OP width bound uses Proposition 4 and the additional candidate event. Since $C _ { Z } \leq T ( 1 + Z )$ , the grid contains a valid point and the same event permits the post hoc substitution of Γ<sup>⋆</sup>.

Thus, on the appropriate common event,

$$
2 \sum _ { \substack { t = t _ { \mathrm { i n i t } } + 1 } } ^ { T } I _ { t } \beta _ { \hat { a } _ { t } ^ { \Gamma ^ { \star } } , t } ^ { Z , \Gamma ^ { \star } } \leq \mathcal { R } _ { \mathrm { s h } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma ^ { \star } ) .\tag{85}
$$

Initialization is charged separately, so $\mathcal { R } _ { \mathrm { s h } } ^ { 0 }$ is a post-initialization width bound. The parameters of the OP event concern the algorithm’s actual shared trajectory, not the trajectory of a candidate run alone.

## C.2. Master Comparison with a Uniform-Advice Expert

Let $\mathcal { Q } _ { t }$ include the full analysis history, current context, fitted coefficients, prices, and all recommendations before fresh round-t draws. Let $\mathcal { P } _ { t }$ also include the potential observed payoff vector. Non-anticipation fixes this vector before the independent action seed, so, conditional on $\mathcal { P } _ { t }$ and a non-forced branch, $a _ { t } \sim p _ { t }$

LEMMA 17 (Unbiased expert-payoff estimate). On each post-initialization, non-forced active round, every expert $j \in \mathcal { E } _ { \mathrm { M } }$ obeys

$$
\mathbb { E } [ \widehat { y } _ { t , j } \mid \mathcal { P } _ { t } , e _ { t } = 0 ] = \sum _ { a \in \mathcal { A } _ { 0 } } \xi _ { t , j } ( a ) Y _ { t } ( a ) .
$$

For a grid candidate this equals $Y _ { t } ( \widehat { a } _ { t } ^ { \Gamma } )$ ; for the uniform expert it is the uniform average.

Proof. All $p _ { t } ( a ) \geq p _ { \operatorname* { m i n } } > 0$ . Conditional expectation of $\widehat { Y } _ { t } ( a ) = \mathbf { 1 } \{ a _ { t } = a \} Y _ { t } ( a _ { t } ) / p _ { t } ( a _ { t } )$ is $Y _ { t } ( a )$ ; summing against the advice gives the result. The denominator is the actual non-forced sampling probability, not the mixture probability including forced overrides.

LEMMA 18 (Master score regret with overrides). For the parameters (28), let $L _ { \mathrm { M } } = \log ( 3 M / \delta _ { \mathrm { m a s t e r } } )$ . With $f a i l -$ ure at most $2 \delta _ { \mathrm { m a s t e r } } / 3$ , simultaneouslyfor all $\Gamma \in \mathcal G$

$$
\sum _ { t = t _ { \mathrm { i n i t } } + 1 } ^ { T } I _ { t } \big [ F _ { t } ^ { \mathrm { o b s } } ( \widehat { a } _ { t } ^ { \Gamma } ; \lambda _ { t } ) - F _ { t } ^ { \mathrm { o b s } } ( a _ { t } ; \lambda _ { t } ) \big ] \leq 8 h \sqrt { T K _ { 0 } L _ { \mathrm { M } } } + h \sum _ { t = 1 } ^ { T } \epsilon _ { t } .\tag{86}
$$

Proof. Suppose first $T \geq 4 K _ { 0 }$ log M and $L _ { \mathrm { M } } \leq K _ { 0 } T$ . Then the cap on $p _ { \mathrm { m i n } }$ is inactive and (28) matches EXP4.P with confidence $\delta _ { \mathrm { m a s t e r } } / 3$ . The expert class includes an expert whose advice is uniform on every round, as required by Beygelzimer et al. (2011, Theorem 2 and Algorithm 1). The explicit uniform-advice expert is not replaced by action smoothing. Payoffs are in [0, 1], advice and potential payoffs are determined before the fresh master draw, and Lemma 17 verifies importance weighting.

Reindex post-initialization, non-forced active rounds. Their number is at most $T ;$ pad this sequence after its final round with zero-payoff rounds, retaining the uniform expert and continuing the same master update. Padding adds zero payoff difference for every expert and does not alter earlier decisions. This is a length-T non-anticipating expert-advice problem, even though the original subsequence and stopping time depend on the preceding history. The cited theorem gives a normalized regret at most $6 \sqrt { T K _ { 0 } L _ { \mathrm { M } } }$ with failure at most $\delta _ { \mathrm { m a s t e r } } / 3$

Forced rounds leave the master weights unchanged and cost at most h each. Since $I _ { t }$ is predictable and the forced coin is fresh,

$$
\sum _ { t > t _ { \mathrm { i n i t } } } I _ { t } e _ { t } \leq \sum _ { t = 1 } ^ { T } \epsilon _ { t } + \sqrt { 2 T \log ( 3 / \delta _ { \mathrm { m a s t e r } } ) }
$$

except with probability $\delta _ { \mathrm { m a s t e r } } / 3 ,$ , by the exponential bound for bounded martingale differences. Multiplication by h and $K _ { 0 } \geq 2$ give (86). If either horizon condition fails, the total observed-score regret is always at most hT. When $T < 4 K _ { 0 }$ log M this is at most $2 h \sqrt { T K _ { 0 } L _ { \mathrm { M } } } ;$ when $L _ { \mathrm { M } } > K _ { 0 } T$ it is at most $h \sqrt { T K _ { 0 } L _ { \mathrm { M } } }$ . Thus the same bound holds in the capped or trivial regime without invoking the nontrivial-horizon theorem.

LEMMA 19 (Uniform clean-score transfer). Withfailure at most $\delta _ { \mathrm { m a s t e r } } / 3 ,$ , simultaneouslyfor everyfixed $\Gamma \in { \mathcal { G } } ,$

$$
\begin{array} { r l } {  { \sum _ { t > t _ { \mathrm { i n i t } } } I _ { t } \big [ F ^ { 0 } ( \widehat { a } _ { t } ^ { \Gamma } , x _ { t } ; \lambda _ { t } ) - F ^ { 0 } ( a _ { t } , x _ { t } ; \lambda _ { t } ) \big ] } \quad } & { } \\ & { \leq \sum _ { t > t _ { \mathrm { i n i t } } } I _ { t } \big [ F _ { t } ^ { \mathrm { o b s } } ( \widehat { a } _ { t } ^ { \Gamma } ; \lambda _ { t } ) - F _ { t } ^ { \mathrm { o b s } } ( a _ { t } ; \lambda _ { t } ) \big ] + 2 C _ { z } + 2 h \sqrt { 2 T \log ( 3 | \mathcal { G } | / \delta _ { \mathrm { m a s t e r } } ) } . } \end{array}\tag{87}
$$

Proof. For each non-null action,

$$
F _ { t } ^ { \mathrm { o b s } } ( a ; \lambda _ { t } ) - F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) = \eta _ { t } ^ { r } ( a ) - Z \lambda _ { t } ^ { \top } \eta _ { t } ^ { b } ( a ) + c _ { t } ^ { r } ( a ) - Z \lambda _ { t } ^ { \top } c _ { t } ^ { b } ( a ) .
$$

Each of the two action sequences contributes at most $C _ { Z }$ in absolute corruption. For a fixed grid index, the recommendation is pre-feedback measurable and the played action uses an independent fresh seed. Hence the difference of their clean-noise terms is conditionally mean-zero, with absolute value at most 2h. This is a martingale-difference argument under the original history and current context, not under conditioning on the potential observed payoff vector or a successful confidence event. Azuma’s inequality and a union bound over the fixed grid give the displayed remainder. Only after this uniform event is established do we substitute the path-dependent Γ<sup>⋆</sup>.

The preceding two lemmas together imply a clean candidate-to-played comparison bounded by $\mathrm { R e g } _ { \mathrm { m s t } } ( T ) +$ $h \sum _ { t } \epsilon _ { t } + 2 C _ { Z }$ , using the stated choice $\mathrm { R e g } _ { \mathrm { m s t } } ( T ) = 2 0 h \sqrt { T K _ { 0 } \log ( 3 M / \delta _ { \mathrm { m a s t e r } } ) }$

## C.3. Resource Conversion, Stopping, and Assembly

ProofofTheorem 2. Define

$$
\mathcal { E } _ { \mathrm { o p t } } ^ { \mathrm { s h } } = \sum _ { t = 1 } ^ { T } I _ { t } \left[ \sum _ { a } y ^ { \star } ( a \mid x _ { t } ) F ^ { 0 } ( a , x _ { t } ; \lambda _ { t } ) - F ^ { 0 } ( a _ { t } , x _ { t } ; \lambda _ { t } ) \right] .
$$

Split this sum into the initialization prefix, the subsequent population-L $\scriptstyle \mathbf { P - t o - } \Gamma ^ { \star }$ recommendation comparison, and the subsequent Γ<sup>⋆</sup>-to-played-action comparison. Initialization costs at most $h t _ { \mathrm { i n i t } }$ . On the joint confidence and coverage event, Lemma 10 and (85) bound the second part by $\mathcal { R } _ { \mathrm { s h } } ^ { 0 } + \psi _ { \mathrm { e s t } } ( \Gamma ^ { \star } )$ . The master and clean-transfer lemmas bound the third by $\begin{array} { r } { \mathrm { R e g } _ { \mathrm { m s t } } ( T ) + h \sum _ { t } \epsilon _ { t } + 2 C _ { Z } } \end{array}$

For FE the joint event comprises noise, forced design/counts, and the master/transfer events. For OP it instead uses the realized-design event together with candidate-recommendation coverage. Their failure probabilities sum to the corresponding $\delta _ { \mathrm { t o t } }$ in (34). The per-round clean-score gap is always at most $h ,$ , including on the complement. All terms on the good-event upper bound are nonnegative, so taking expectations gives

$$
\mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ^ { \mathrm { s h } } ] \le \mathcal { R } _ { \mathrm { s h } } ^ { 0 } + \mathrm { R e g } _ { \mathrm { m s t } } ( T ) + h t _ { \mathrm { i n i t } } + h \sum _ { t } \epsilon _ { t } + \mathbb { E } [ \psi _ { \mathrm { e s t } } ( \Gamma ^ { \star } ) + 2 C _ { Z } ] + h \delta _ { \mathrm { t o t } } T .\tag{88}
$$

The method uses exactly the known algorithm’s actual observed-consumption price update and pre-context stopping test. Lemmas 13, 14, and 15 therefore apply to this played sequence without any estimator change:

$$
\mathrm { R e g } ( T ) \leq \mathbb { E } [ \mathcal { E } _ { \mathrm { o p t } } ^ { \mathrm { s h } } ] + Z \mathbb { E } [ R _ { D } ] + \mathbb { E } [ C _ { Z } ] + Z .\tag{89}
$$

Here $Z \mathbb { E } [ R _ { D } ] \leq \mathcal { D } _ { T }$ by Lemma 12. The additional $C _ { Z }$ is the clean-to-observed resource transfer, separate from the two master score transfers. Thus their total is $3 \mathbb { E } [ C _ { Z } ] \le 3 \mathbb { E } [ \Gamma ^ { \star } ]$ . This proves the expected-regret claim. Proposition 1 gives the pathwise observed and clean resource claims, independently of all confidence and coverage events.

Proofs ofCorollaries 4 and 5. For FE substitute (65) into (88) and then (89). The combined initialization and forced retained history has the RSC calibration proved in Lemma 5; subsequent forced counts provide a lower bound for its sample size. This is a bound on counts and radii, not an assumed ordering of estimators fitted on different samples. Add $h t _ { \mathrm { i n i t } } , h \epsilon T _ { \mathrm { : } }$ , the master, dual, and stopping costs. This yields (38). Applying the exploration calculation in Appendix B.6 gives (36); its chosen ϵ is independent of the unknown pathwise corruption. For OP use (68). Its proof keeps recommendation and retained counts separate and explicitly charges $h \chi K n _ { 0 }$ before informative recommendation confidence. Substitution, with $\epsilon _ { t } = 0$ , yields (37). Retain the full $h \delta _ { \mathrm { t o t } } T$ contribution in both cases. If $C _ { Z } = 0 , \Gamma ^ { \star } = 0$ removes only the corruption terms; it does not remove initialization, regularization, or the master comparison.

## C.4. Proof of Recommendation Coverage

Proof of Lemma 1. Fix $( \Gamma , a , t )$ and, for post-initialization rounds $\tau < t ,$ , let

$$
\begin{array} { r } { Q _ { \tau } : = I _ { \tau } { \bf 1 } \{ \widehat { a } _ { \tau } ^ { \mathrm { r } } = a \} , \qquad X _ { \tau } : = Q _ { \tau } { \bf 1 } \{ a _ { \tau } = a \} . } \end{array}
$$

Conditional on $\mathcal { Q } _ { \tau } , \mathbb { E } [ X _ { \tau } \mid \mathcal { Q } _ { \tau } ] \geq q _ { \mathrm { c o v } } Q$ <sub>τ</sub> . For every deterministic $u > 0$ , a multiplicative martingale lower-tail bound gives

$$
\operatorname* { P r } \left( \sum _ { \tau < t } X _ { \tau } < \frac { q _ { \mathrm { c o v } } } { 2 } \sum _ { \tau < t } Q _ { \tau } , \quad \sum _ { \tau < t } Q _ { \tau } \geq u \right) \leq \exp \left( - \frac { q _ { \mathrm { c o v } } u } { 8 } \right) .
$$

Set

$$
u = \frac { 8 } { q _ { \mathrm { c o v } } } \log \frac { 2 | \mathcal { G } | K T } { \delta _ { \mathrm { c a n d } } } .
$$

A union bound over candidates, actions, and prefixes implies that, simultaneously for all $( \Gamma , a , t )$ , either $N _ { a } ^ { \mathrm { r e c , } \Gamma } ( t ) < u$ or

$$
\sum _ { \tau < t } X _ { \tau } \geq \frac { q _ { \mathrm { c o v } } } { 2 } N _ { a } ^ { \mathrm { r e c , \Gamma } } ( t ) .
$$

Since $\smash { \sum _ { \tau \leq t } X _ { \tau } \leq N _ { a } ( t ) }$ , in the latter case $N _ { a } ^ { \mathrm { r e c , \Gamma } } ( t ) \leq 2 N _ { a } ( t ) / q _ { \mathrm { c o v } }$ . Combining the two cases proves (32).