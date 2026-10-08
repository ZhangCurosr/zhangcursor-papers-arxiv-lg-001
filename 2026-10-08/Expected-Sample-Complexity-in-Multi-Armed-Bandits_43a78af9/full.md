# Expected Sample Complexity in Multi-Armed Bandits

Nadav Sukenik Technion – Institute of Technology

Nadav Merlis Technion – Institute of Technology

## Abstract

Sample complexity is a widely used metric in sequential decision-making problems, defined as the number of suboptimal decisions during the interaction between the agent and an environment. We study the sample complexity of stochastic multi-armed bandit problems and introduce the expected sample complexity performance measure, analyzing it in a novel framework called approximately correct in expectation (ACE). We show that ACE guarantees imply almost sure convergence to the optimal expected reward, in contrast to high-probability guarantees found in other frameworks, and also show how to convert ACE guarantees into explicit expected regret bounds. We further show that, in contrast to existing measures, deterministic algorithms cannot obtain favorable ACE bounds, and analyze stochastic algorithms in two settings: when the allowed suboptimality level ϵ is known to the algorithm and when it is unknown. In the former, we devise an explorethen-ϵ-greedy algorithm, and in the latter, we analyze the expected sample complexity of Thompson sampling. Finally, we establish nearly matching lower bounds for both settings, showing that the algorithms are tight in ϵ and proving a performance separation between the two regimes.

## 1 INTRODUCTION

The stochastic multi-armed bandit (MAB) problem (Lattimore and Szepesv´ari, 2020) is an important sequential decision making problem, in which an agent picks an action from a set of possible arms and observes a stochastic reward for the chosen arm. A common measure in both MABs and the more general reinforcement learning (RL, Sutton and Barto, 1998) setting is the expected regret, a measure of the diference between the performance of an optimal oracle and our algorithm when interacting with the environment over a finite horizon (Lattimore and Szepesv´ari, 2020).

Another performance metric that is mostly considered in the RL setting is the sample complexity (Kakade, 2003), which measures how many ϵ-suboptimal actions are played during the interaction for a given suboptimality level ϵ. Algorithms for which the sample complexity is bounded with high probability are often called PAC-MDP (Kakade, 2003).

This framework has two notable disadvantages: (1) algorithms for sample complexity optimization are tailored to a specific suboptimality level, which leads to high regret or even infinite sample complexity for smaller suboptimality levels (Dann et al., 2017); and (2) they are tailored to a specific probability of success, which implies a constant probability that catastrophic failure may occur, in the form of infinite suboptimal rounds. While previous works extended the setting to alleviate the first issue (Dann et al., 2017; Liu et al., 2024), the second issue is largely unaddressed.

We tackle both weaknesses by introducing a novel framework for analyzing the sample complexity called approximately correct in expectation (ACE). In this framework, we measure the expected sample complexity: the expected number of rounds in which the policy was expected to act ϵ-suboptimally over an infinite horizon of interaction.

We show that algorithms that have uniformly bounded expected sample complexity – a property we call Uniformly-ACE (U-ACE) – have almost-sure convergence to the optimal expected reward. This comes in stark contrast to the fixed-probability guarantees of other sample complexity frameworks. Moreover, we show that explicit expected regret bounds can be derived from U-ACE guarantees.

Next, we show that deterministic algorithms cannot have a finite expected sample complexity on all instances of the Bernoulli bandit class, which leads us to analyze the expected sample complexity of randomized algorithms. We do so in two regimes: when the suboptimality level ϵ is known and when it is unknown. In the latter setting, we analyze Thompson sampling (Thompson, 1933), a well-known MAB algorithm popular for its strong empirical performance; for the former setting, we suggest a successive-eliminationϵ-greedy hybrid algorithm. We complement the expected sample complexity upper bounds with nearly matching lower bounds, showing that the guarantees possible in the two regimes are inherently diferent. To our knowledge, this is the first such separation result in the sample complexity literature.

## 2 RELATED WORK

The stochastic multi-armed bandits problem (Lattimore and Szepesv´ari, 2020; Slivkins, 2019) is one of the most widely studied problems in sequential decision making, with applications in the fields of advertising, recommendation systems, finance and more (Silva et al., 2022; Bounefouf and Rish, 2019).

A common performance measure in the MAB literature is the expected regret, which is the cumulative diference between the reward of the best action (arm) and the agent’s action (Lattimore and Szepesv´ari, 2020). Much of the literature focuses on the design of algorithms that minimize the expected regret (Auer et al., 2002; Lattimore and Szepesv´ari, 2020). An example of such an algorithm is Thompson sampling (TS, Thompson (1933); Kaufmann et al. (2012)), a randomized algorithm inspired by the Bayesian statistics framework, which grew in popularity due to its demonstrated strong empirical performance (Chapelle and Li, 2011).

Another common performance measure is the sample complexity of exploration (Kakade, 2003; Domingues et al., 2021), referred to as sample complexity, which is defined as the number of ϵ-suboptimal interaction rounds of the agent over an infinite horizon. The sample complexity is usually analyzed in the probably approximately correct (PAC) framework, and the setting is commonly referred to as PAC-MDP. We also refer to this setting as PAC, and emphasize that it should not be conflated with the best arm identification (Audibert et al., 2010) setting, where an algorithm must identify a near-optimal arm with high probability.

Although the sample complexity has been formally defined for randomized policies (Domingues et al., 2021), most works on the metric have largely focused on de terministic algorithms, with no obvious way to generalize their results to randomized policies.

This constitutes a major literature gap, since many widely used algorithms, such as Thompson Sampling (Thompson, 1933), proximal policy optimization (PPO, Schulman et al., 2017), trust region policy optimization (TRPO, Schulman et al., 2015), and many more (Lehmann, 2024; Sutton et al., 1999) are stochastic. In contrast to previous works, our framework makes the distinction between randomized and deterministic algorithms and their possible guarantees.

There are many extensions to the standard PAC framework. A notable example is Uniform-PAC (U-PAC, Dann et al., 2017), which measures PAC uni formly across all suboptimality levels ϵ, resulting in stronger guarantees compared to regular PAC. Another noteworthy extension is uniform last-iterate (ULI, Liu et al., 2024), which instead bounds the instantaneous suboptimality gap of an algorithm with high probability, also giving rise to stronger guarantees than the vanilla PAC framework. We define and discuss these frameworks in detail in Section 3.3. In particular, we explain how both extensions sufer from the same constant probability of potentially catastrophic failure mentioned in Section 1.

Another related metric to the sample complexity is the lenient regret (Merlis and Mannor, 2021), which combines the notions of regret and sample complexity into a single performance measure; we also discuss the relation between ACE and lenient regret in Section 3.2.

## 3 SETTING

## 3.1 Multi-Armed Bandits

We consider the stochastic multi-armed bandit prob lem with K arms, where each arm $a \in [ K ] \triangleq \triangleq$ $\{ 1 , 2 , \ldots , K \}$ is associated with a distribution $\nu _ { a }$ . Let $\boldsymbol { \nu } = \left\{ \nu _ { a } \right\} _ { a = 1 } ^ { K }$ denote the set of all arm distributions. At each round $t \in \mathbb { N } .$ , an agent picks a single arm to pull $A _ { t }$ and observes a reward from only that arm. When pulling arm a for the $n ^ { t h }$ time, the agent observes a reward $X _ { a , n } \sim \nu _ { a }$ bounded in [0, 1], and of expectation $\mu _ { a } \triangleq \mathbb { E } [ X _ { a , n } ]$ . Denote the expectation of an optimal arm by $\mu ^ { * } = \operatorname* { m a x } _ { a } \mu _ { a }$ , and the suboptimality gap of an arm a by $\Delta _ { a } \triangleq \mu ^ { * } - \mu _ { a }$ . Additionally, we denote $\Delta _ { \mathrm { m i n } } \triangleq \operatorname* { m i n } _ { a : \Delta _ { a } > 0 } \Delta _ { a } .$ , and $\Delta _ { \operatorname* { m a x } } \triangleq \operatorname* { m a x } _ { a } \Delta _ { a }$

The decisions of the agent are defined by a sequence of policies $\pi \triangleq ( \pi _ { t } ) _ { t = 1 } ^ { \infty } ,$ where each policy $\pi _ { t }$ is a mapping from histories $\mathcal { F } _ { t - 1 }$ to distributions over the arms $\Delta ( [ K ] )$ . At round t, the agent samples an arm based on the current history $A _ { t } \sim \pi _ { t } ( \cdot \mid \mathcal { F } _ { t - 1 } )$ , and observes a reward $X _ { t } = X _ { A _ { t } , n _ { t } ( A _ { t } ) + 1 }$ , where $n _ { t } ( a ) =$ $\textstyle \sum _ { i = 1 } ^ { t - 1 } 1 \mathbb { \{ } A _ { i } = a \}$ is the number of times an arm a was sampled by the agent up to round t − 1.

Denote the mean of arm a using the first n samples by $\begin{array} { r } { \hat { \mu } _ { a , n } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } X _ { a , k } } \end{array}$ and set $\hat { \mu } _ { a , 0 } \triangleq 0$ . Define the empirical mean up to round t to be $\hat { \mu } _ { t } ( a ) = \hat { \mu } _ { a , n _ { t } ( a ) }$

The history $\mathcal { F } _ { t - 1 }$ contains all the actions and rewards observed before round t, but not the internal randomization of the algorithm. We refer to Appendix A.2 for a formal definition and a detailed explanation.

We say that policy $\pi _ { t }$ is deterministic if for every realization of history $\mathcal { F } _ { t - 1 }$ , there exists an arm $a \in [ K ]$ such that $\pi _ { t } ( a \mid \mathcal F _ { t - 1 } ) = 1$ . Any policy that is not deterministic is called stochastic. Unless stated oth erwise, we let $\pi _ { t } ( a ) \triangleq \pi _ { t } ( a | \mathcal F _ { t - 1 } )$ for brevity. Finally, we say that policy $\pi ^ { * }$ is optimal if $\begin{array} { r } { \sum _ { a : \Delta _ { a } = 0 } \pi ^ { * } ( a ) = 1 } \end{array}$

## 3.2 Expected Regret, Lenient Regret and Sample Complexity

The most common performance measure in the MAB literature is the expected regret (Lattimore and Szepesv´ari, 2020), which measures the cumulative difference in reward between the best possible agent and an agent acting according to policy π when interacting with instance $\nu .$ The expected regret is defined to be

$$
R _ { T } ( \pi , \nu ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { t } \right] ,
$$

where $\Delta _ { t } = \Delta _ { A _ { t } }$ , and the expectation is taken over the interaction of the agent with the environment. When ν is clear from context, we write $R _ { T } ( \pi ) = R _ { T } ( \pi , \nu )$

Another common performance measure is the sample complexity (Kakade, 2003; Domingues et al., 2021). It measures how many suboptimal decisions an agent was expected to perform when using the policy sequence $\pi = ( \pi _ { t } ) _ { t = 1 } ^ { \infty }$ over an infinite horizon of interaction with instance ν. Formally, we define the policy gap at round t to be $\begin{array} { r } { \Delta _ { \pi _ { t } } \triangleq \sum _ { a = 1 } ^ { K } \pi _ { t } ( a ) \Delta _ { a } } \end{array}$ , and define the sample complexity to be

$$
N _ { \epsilon } ( \pi , \nu ) = \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \}\tag{1}
$$

for some suboptimality level $\epsilon > 0 .$ The notion of the policy gap arises naturally since it coincides with the realized gap $\Delta _ { A _ { i } }$ for deterministic algorithms, and measures the suboptimality of our decision rule itself – the policy. Though it is equally meaningful to measure the suboptimality of the realized actions, we show that working with the policy gap has a clear advantage – it facilitates the derivation of many favorable guarantees, including almost sure convergence, expected regret bounds and PAC guarantees.

We note that this definition of the sample complexity is closely related to the lenient regret metric (Merlis and Mannor, 2021), which is defined as

$$
R _ { T } ^ { \epsilon } ( \pi , \nu ) = \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \Delta _ { t } > \epsilon \} .
$$

The notable diference between the two is that sample complexity measures the expected suboptimality of the policy and is mostly analyzed in the PAC setting, while lenient regret measures the suboptimality of the realized actions and is analyzed in expectation.

To relate the policy gap and the suboptimality gap, we show the following:

Claim 1. It holds that $\begin{array} { r } { \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { t } \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { \pi _ { t } } \right] , } \end{array}$

The proof is given in Appendix B.1. While this claim implies that optimizing the expected cumulative policy gap and the expected regret are equivalent, we note that this is not necessarily true for the infinite-horizon lenient regret and the sample complexity.

## 3.3 PAC, Uniform-PAC and Uniform Last-Iterate

For the following definitions, we follow the conventions used in Domingues et al. (2021) to formally extend the sample complexity framework to randomized algo rithms. In doing so, we adopt the notion of the policy gap for the frameworks, and note that all the definitions align with their original statements for deterministic algorithms. We remark that while Domingues et al. (2021) defined the sample complexity to accommodate randomized policies, their sample complexity results only apply to deterministic ones. The most common framework in the literature of sample complexity is the PAC framework:

Definition 1 (Kakade (2003)). A policy $\pi$ is called $( \epsilon , \delta ) { - } P A C$ with a finite sample complexity bound m ${ } ^ { > } A C ( \epsilon , \delta ) ~ i f \operatorname* { P r } ( N _ { \epsilon } ( \pi , \nu ) > m _ { P A C } ( \epsilon , \delta ) ) \leq \delta$

Algorithms in this setting are designed to minimize the sample complexity bound for a given (ϵ, δ) pair, which leads to two notable disadvantages:

1. Algorithms are tailored to the given suboptimality level $\epsilon ,$ resulting in suboptimal performance in terms of expected regret, high-probability regret, and even (ϵ<sup>′</sup>, δ)-PAC for $\epsilon ^ { \prime } < \epsilon$ (Dann et al., 2017). In practice, we favor algorithms that can be agnostic or robust to the parameters $\epsilon , \delta .$

2. Algorithms are designed for the given confidence δ, and as such can fail with constant probability, resulting in infinite sample complexity, a highly undesirable trait.

Extensions have been made to the existing PAC framework to mitigate the first problem, but to our knowledge, none have made progress in the latter issue. One such framework is Uniform-PAC:

Definition 2 ((Dann et al., 2017)). A policy π is called δ-U-PAC for $\delta \ > \ 0$ with sample complexity bound $m _ { U - P A C } ( \epsilon , \delta )$ if

$$
\operatorname* { P r } ( \exists \epsilon > 0 : N _ { \epsilon } ( \pi , \nu ) > m _ { U - P A C } ( \epsilon , \delta ) ) \leq \delta .
$$

It can be shown that algorithms that are δ-U-PAC are also $( \epsilon , \delta ) \mathrm { - P A C }$ for all $\epsilon > 0 .$ and enjoy highprobability regret bounds (Dann et al., 2017), which addresses the major clauses in the first disadvantage. Unfortunately, this framework does not address the second problem, still allowing catastrophic failure to possibly occur with constant probability.

Another noteworthy framework is the Uniform Last-Iterate (ULI) framework (Liu et al., 2024), a stronger notion than U-PAC that gives us sharper bounds.

Definition 3 (ULI). A policy π is called δ-ULI if there exists a function $m _ { U L I } ( \delta , t ) ~ \le ~ p o l y \bigl ( { \textstyle { \frac { 1 } { t } } } , \log { \frac { 1 } { \delta } } \bigr )$ 2 such that $\operatorname* { P r } ( \exists t : \Delta _ { \pi _ { t } } > m _ { U L I } ( \delta , t ) ) \leq \delta .$

While the U-PAC framework allows algorithms to play maximally suboptimal arms for a very large but finite horizon, the ULI framework instead characterizes the actual convergence rate of $\Delta _ { \pi _ { t } }$ . The main theorem in Liu et al. (2024) states that ULI implies U-PAC, and their proof generalizes to the policy gap. As such, any policy which is ULI also sufers from the same disadvantages as U-PAC.

## 3.4 Expected Sample Complexity and the ACE Framework

We introduce a new sample complexity measure, the expected sample complexity, which for a given policy π and instance ν is defined as:

$$
\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] .
$$

When ν is clear from context, we let $N _ { \epsilon } ( \pi ) \triangleq N _ { \epsilon } ( \pi , \nu )$

We also introduce a novel framework for the analysis of the expected sample complexity called the approximately correct in expectation (ACE) framework. We use the following definitions:

Definition 4. A policy $\pi \textit { \textbf { \_ } } i s$ called ϵ-ACE (Approximately Correct in Expectation) on ν with sample complexity bound $m _ { A C E } ( \epsilon , \nu ) < \infty \ \ i$ f:

$$
\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] \le m _ { A C E } ( \epsilon , \nu ) .
$$

Definition 5. A policy π is U-ACE (Uniformly-Approximately Correct in Expectation) on ν if π is $\epsilon { - } A C E$ with bound $m _ { U - A C E } ( \epsilon , \nu )$ for all $\epsilon > 0$

## 4 PROPERTIES OF (U-)ACE ALGORITHMS

In this section, we analyze the properties of ACE and U-ACE policies, including convergence, expected regret bounds, and the necessity of stochastic policies to be U-ACE.

We begin by showing that ACE bounds can be used to prove the convergence of algorithms. In the following proposition, we show that ϵ-ACE implies almost-sure ϵ boundedness of the policy gap

Proposition 1. $L e t \pi = ( \pi _ { t } ) _ { t = 1 } ^ { \infty }$ be ϵ-ACE for some $\epsilon > 0$ . Then lim $\begin{array} { r } { \operatorname* { s u p } _ { t \to \infty } \Delta _ { \pi _ { t } } \leq \epsilon } \end{array}$ almost surely.

The proof of this proposition is given in Appendix B.2 and relies on the Borel–Cantelli lemma. As a corollary, we get that U-ACE implies almost sure convergence of the policy gap to zero:

Corollary 1. Let $\pi ~ = ~ ( \pi _ { t } ) _ { t = 1 } ^ { \infty }$ be U-ACE. Then $\begin{array} { r } { \operatorname* { l i m } _ { t  \infty } \Delta _ { \pi _ { t } } = 0 } \end{array}$ almost surely.

This corollary is proved by applying Proposition 1 to a sequence ϵ ↓ 0; see Appendix B.2 for a formal proof.

This result yields a much stronger guarantee than the high-probability convergence implied by previous frameworks. We do remark that Corollary 1 does not imply convergence of the sequence $\pi _ { t }$ to an optimal policy $\pi ^ { * }$ . In fact, $\pi _ { t }$ may not converge at all; for example, if the policy flips between two distinct optimal policies. Yet, the corollary indicates that the expected reward of the sequence converges to that of an optimal policy, which we believe to be the more meaningful definition for a convergence of a policy to its optimum.

Next, we show that for algorithms with polynomial $m _ { U - A C E }$ , the expected regret is sublinear:

Claim 2. Assume that $\pi \quad { \begin{array} { r l } { i s \quad } & { { } U - A C E } \end{array} }$ with $\begin{array} { r } { m _ { U - A C E } ( \epsilon , \nu ) \ \leq \ \frac { C _ { \nu } } { \epsilon ^ { \alpha } } } \end{array}$ for $C _ { \nu } ~ > ~ 0 , ~ \alpha ~ > ~ 1$ . Then for all T, it holds that $\begin{array} { r } { R _ { T } ( \pi , \nu ) \leq \frac { \alpha } { \alpha - 1 } C _ { \nu } ^ { 1 / \alpha } T ^ { \frac { \alpha - 1 } { \alpha } } } \end{array}$

Moreover, for α = 1, we obtain a logarithmic bound of $\begin{array} { r } { R _ { T } ( \pi , \nu ) \leq C _ { \nu } \Big ( 1 + \operatorname* { m a x } \Big \{ 0 , \ln \Big ( \frac { T } { C _ { \nu } } \Big ) \Big \} \Big ) } \end{array}$

Proof of this claim can be found in Appendix B.3. After showing the favorable properties of the framework, we end this section with an impossibility result for deterministic algorithms: such algorithms cannot be U-ACE for all bandit instances. This will later guide us to only focus on stochastic policies.

Claim 3. Let $\mathcal { E } _ { B e r } ^ { K }$ be the class of K-armed Bernoulli bandits. For any deterministic policy π and $K \geq 2$ there exists $\nu \in \dot { \mathcal { E } } _ { B e r } ^ { K }$ and ϵ > 0 s.t. $\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \infty$

A proof for this claim is given in Appendix B.4 and is based on the following observation: for deterministic algorithms, $\mathrm { i f } \epsilon < \Delta _ { \mathrm { m i n } }$ , the expected sample complexity is the number of rounds in which a suboptimal arm was played. Thus, it sufices to show that there exists an instance in which the expected number of suboptimal plays is infinite. This can be proved using known lower bounds for the bandit problem.

At first glance, Claim 3 seems to imply that U-ACE is not achievable; indeed, a simple application of Markov’s inequality gives a PAC guarantee (albeit with polynomial dependence on $1 / \delta )$ , since

$$
\mathrm { P r } ( N _ { \epsilon } ( \pi , \nu ) > m _ { P A C } ( \epsilon , \delta ) ) \leq \frac { \mathbb { E } [ N _ { \epsilon } ( \pi ) ] } { m _ { P A C } ( \epsilon , \delta ) } .
$$

Thus, for $\begin{array} { l } { \mathit { m } _ { P A C } ( \epsilon , \delta ) = \frac { \mathbb { E } [ N _ { \epsilon } ( \pi ) ] } { \delta } } \end{array}$ we get $( \epsilon , \delta ) { \mathrm { - P A C } } .$ while Claim 2 simultaneously implies an expected regret bound. This seemingly contradicts Theorem 1 of Dann et al. (2017), which states that no algorithm may achieve sublinear regret and finite PAC bounds for arbitrarily small ϵ. Yet, a closer inspection of (Dann et al., 2017) reveals their analysis assumes deterministic policies, and their methods are inapplicable for stochastic ones. This key observation and Claim 3 both encourage us to turn our attention to stochastic algorithms in our pursuit of U-ACE.

In the following section, we show that stochastic algorithms can enjoy both near-optimal expected regret, U-ACE and a finite (ϵ, δ)-PAC bound for all $\epsilon > 0 .$ , a surprising result considering most of the literature on sample complexity studies deterministic algorithms, dating back to the seminal work of Kakade (2003).

## 5 ALGORITHMS FOR THE ACE FRAMEWORK

We now analyze the expected sample complexity of two classes of algorithms: algorithms that receive ϵ as input (non-uniform algorithms), and algorithms that do not (uniform algorithms). We show that the dependence of the expected sample complexity on ϵ is inherently diferent for the diferent classes; to our knowledge, this is the first separation result of this type in the sample complexity setting.

## 5.1 Non-Uniform Algorithms

This setting is the most similar to the vanilla PAC framework; the algorithm receives ϵ as input, and its goal is to achieve ϵ-ACE with minimal $m _ { A C E } ( \epsilon , \nu )$ While algorithms designed for this setting can only support suboptimality levels equal or larger than theirs (in contrast to uniform algorithms), we will show that this regime leads to better ACE bounds.

Before presenting our algorithm for this regime, we explain our design methodology; one can surmise that algorithms for this setting must explore infinitely often to stay ϵ-optimal over an infinite horizon in expectation. We turn our attention to ϵ-greedy (Auer et al., 2002), a simple algorithm that performs biased coin tosses to decide whether to explore or exploit the environment at each round. Formally, for a given sequence of biases $\left( \epsilon _ { t } \right) _ { t } ,$ the ϵ-greedy policy is defined as

$$
\pi _ { t } ^ { \epsilon - g r e e d y } ( a \mid \mathcal { F } _ { t - 1 } ) = \frac { \epsilon _ { t } } { K } + ( 1 - \epsilon _ { t } ) \mathbb { 1 } \{ a = g _ { t } \} ,
$$

where $g _ { t } \in \arg \operatorname* { m a x } _ { a ^ { \prime } } \hat { \mu } _ { t } ( a ^ { \prime } )$ is a maximizer of the empirical means in round t. A simple approach would then be to run ϵ-greedy with $\epsilon _ { t } \triangleq \epsilon / 2$ . Intuitively, after enough rounds, the greedy action is optimal w.h.p., and so the policy gap is small since our exploration probability is also small by our choice of bias.

While such a policy will be $\epsilon { - } \mathrm { A C E }$ , it will be sample ineficient; the probability of exploration is too small, making the number of rounds needed to detect an optimal arm w.h.p. very large, incurring high expected sample complexity even in ‘easy’ instances.

To balance the need for fast exploration at the beginning with more careful exploitation in later rounds, one can start with a more aggressive exploration algorithm and then transition to a balanced exploration exploitation algorithm in later rounds. With this motivation in mind, we propose the Successive Elimination Then ϵ-Greedy (SETϵ-G) algorithm (depicted in Algorithm 1), and analyze its expected sample complexity. The algorithm has two phases: an initial successive elimination phase for exploration and pruning of bad arms, and an ϵ-greedy phase to exploit discovered good arms but keep exploring in case of misidentification. The algorithm enjoys the following ACE bound (see Appendix C.1 for the full proof):

Theorem 1. Let $\epsilon > 0$ and let $\pi ^ { S E T \in - G }$ denote the policy induced by Algorithm 1. Then for all ν:

$$
\mathbb { E } \Big [ N _ { \epsilon } \Big ( \pi ^ { S E T \epsilon - G } , \nu \Big ) \Big ] = \mathcal { O } \left( \sum _ { \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 1 } { \Delta _ { a } ^ { 2 } } \ln \left( \frac { K } { \epsilon ^ { 2 } } \right) \right) .
$$

Proof Sketch. We analyze both phases separately. We define a concentration event for the means over the entire horizon, such that the probability it is violated is $\mathcal { O } ( \epsilon ^ { 2 } )$ . For the initial successive elimination phase, we show that under the good event, suboptimal arms with $\Delta _ { a } > \epsilon$ are eliminated after $\tilde { \mathcal { O } } \big ( 1 / \Delta _ { a } ^ { 2 } \big )$ rounds. Since this phase is deterministic, the expected sample complexity is exactly the number of rounds we play arms with gap larger than ϵ: bounded by $\begin{array} { r } { \tilde { \mathcal { O } } \left( \sum _ { a : \Delta _ { a } > \epsilon } \frac { 1 } { \Delta _ { a } ^ { 2 } } \right) } \end{array}$

For the ϵ-greedy phase, we show that the probability of an arm a with gap $\Delta _ { a } \ > \ { \frac { 3 \epsilon } { 4 } }$ being the empirical maximizer is uniformly bounded across all rounds by $\mathcal { O } ( \epsilon ^ { 2 } )$ , and also by an exponentially decreasing function of the rounds of order $e ^ { - \epsilon \Delta _ { a } ^ { 2 } s }$ , where s is the current round of the phase, giving us $\mathrm { P r } ( g _ { s } = a ) =$ $\mathcal { O } \Big ( \operatorname* { m i n } \Big \{ \epsilon ^ { 2 } , e ^ { - \epsilon \Delta _ { a } ^ { 2 } s } \Big \} \Big )$ . In Lemma 5, we show that $\begin{array} { r } { \mathcal { O } \Big ( \sum _ { j = 0 } ^ { \infty } \operatorname* { m i n } \Bigl \{ \epsilon ^ { 2 } , e ^ { - \epsilon \Delta _ { a } ^ { 2 } s } \Bigr \} \Big ) = \tilde { \mathcal { O } } \Bigl ( \frac { 1 } { \Delta _ { a } ^ { 2 } } \Bigr ) } \end{array}$ , and summing over the bad arms gives us a bound of $\tilde { \mathcal { O } } \left( \sum _ { \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 1 } { \Delta _ { a } ^ { 2 } } \right)$ for the ϵ-Greedy phase. Combining the bounds for both phases yields the result. □

```tcl
Algorithm 1 Successive Elimination Then ϵ-Greedy
1: Input: $\epsilon > 0$
2: Set $\begin{array} { r } { \mathcal { A } _ { 1 } = [ K ] , q = \frac { \epsilon ^ { 2 } } { 1 6 K } } \end{array}$
# Phase I
3: for $\begin{array} { r } { n = 1 , 2 , \ldots , n _ { \epsilon } \triangleq \operatorname* { m i n } \{ n \geq 1 : \beta _ { n } < \frac { \epsilon } { 3 2 } \} } \end{array}$ do
4: Set $\begin{array} { r } { \beta _ { n } = \sqrt { \frac { 1 } { 2 n } \ln \left( \frac { 6 4 K ^ { 2 } n ^ { 2 } } { \epsilon ^ { 2 } } \right) } } \end{array}$
5: for every a $\in { \mathcal { A } } _ { n }$ do
6: Pull arm a once
7: end for
8: Let $\hat { \mu } _ { a , n }$ denote the empirical mean of the first
n samples of every $a \in A _ { n }$
9: Let $\mathcal { A } _ { n + 1 } = \{ a \in \mathcal { A } _ { n }$ : max<sub>b∈</sub> ${ \mathcal A } _ { n } \hat { \mu } _ { b , n } - \hat { \mu } _ { a , n } \le 6 \beta _ { n } \Bigr \}$
10: if $| { \mathcal { A } } _ { n + 1 } | = 1$ then
11: Terminate Phase I
12: end if
13: end for
# Phase II
14: Initialize ${ \hat { \mu } } _ { 1 } ( a ) , S _ { 1 } ( a ) , n _ { 1 } ( a )$ for all $a \in [ K ]$ using
the samples from phase I
15: for remaining rounds $s = 1 , 2 , \ldots$ do
16: Sample $B _ { s } \sim B e r ( \operatorname* { m i n } \{ \epsilon / 4 , 1 \} )$
17: if $B _ { s } = 1$ then
18: Play $a _ { s } \sim u n i ( [ K ] )$ and observe reward $X _ { s }$
19: else
20: Play $a _ { s } ~ \in ~ \arg \operatorname* { m a x } _ { a ^ { \prime } } { \hat { \mu } } _ { s } ( a ^ { \prime } )$ (break ties by
smallest arm index) and observe reward $X _ { s }$
21: end if
22: Update $\begin{array} { r } { n _ { s + 1 } ( a _ { s } ) = n _ { s } ( a _ { s } ) + 1 , } \end{array}$
$\begin{array} { r } { S _ { s + 1 } ( a _ { s } ) = S _ { s } ( a _ { s } ) + X _ { s } , \ \hat { \mu } _ { s + 1 } ( a _ { s } ) = \frac { S _ { s + 1 } ( a _ { s } ) } { n _ { s + 1 } ( a _ { s } ) } } \end{array}$
23: For all a $\neq a _ { s } ,$ set $\begin{array} { r } { n _ { s + 1 } ( a ) = n _ { s } ( a ) . } \end{array}$
$S _ { s + 1 } ( a ) = S _ { s } ( a ) , \hat { \mu } _ { s + 1 } ( a ) = \hat { \mu } _ { s } ( a )$
24: end for
```

We complement the ACE bound for SETϵ-G with an almost-matching lower bound, showing that Algorithm 1 is near-optimal in the non-uniform setting, up to logarithmic factors in K and $1 / \epsilon$

Proposition 2. Let $\delta , \epsilon > 0$ such that max $\left\{ \delta , 3 \epsilon \right\} \leq$ $1 / 8 ,$ and let $K \ge 2$ . Define the bandit class ${ \mathcal { E } } _ { \delta } ~ =$ $\left\{ \nu \in \mathcal { E } _ { B e r } ^ { K } : \Delta _ { \operatorname* { m i n } } ( \nu ) \geq \delta \right\}$ Then for every policy $\pi ,$ there exists $\nu \in { \mathcal { E } } _ { \delta }$ such that:

$$
\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \Omega \left( \frac { K } { \operatorname* { m a x } \{ \delta , \epsilon \} ^ { 2 } } \right) .
$$

Proof Sketch. (see Appendix C.3 for the full proof) Our lower bound design methodology closely follows Theorem 15.2 in Lattimore and Szepesv´ari (2020). We let $\Delta = \operatorname* { m a x } \{ \delta , 3 \epsilon \}$ , fix some $T \in \mathbb { N }$ to be chosen later, and create two instances, the first one ν is defined as

$$
\mu _ { \nu } ( a ) = { \left\{ \begin{array} { l l } { 1 / 2 + \Delta } & { a = 1 } \\ { 1 / 2 } & { a \not = 1 } \end{array} , \right. }
$$

and the second instance $\nu ^ { \prime }$ is such that the arm that is played least on $\nu$ in expectation is set to $1 / 2 + 2 \Delta$ with the rest of the arms remaining unchanged. We also let $\begin{array} { r } { \mathbb { E } [ N _ { \epsilon } ( \pi , \nu , T ) ] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] } \end{array}$ be the T-expected sample complexity. Finally, we define A to be the event that the number of times $\pi _ { t } ( 1 )$ was the dominant probability over the horizon is smaller than $T / 2 .$ This implies that the T-expected sample complexity in $\nu$ is at least $\operatorname { P r } _ { \nu } ( A ) \cdot { \frac { T } { 2 } }$ , and the T-expected sample complexity in $\nu ^ { \prime }$ is at least $\operatorname { P r } _ { \nu ^ { \prime } } ( { \overline { { A } } } ) \cdot { \frac { T } { 2 } }$ . Apply the standard Bretagnolle-Huber approach (Theorem 14.2 in Lattimore and Szepesv´ari (2020)) gives us

$$
\mathrm { n a x } \{ \mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu , T ) ] , \mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) ] \} = \Omega \Big ( T e ^ { - \frac { T \Delta ^ { 2 } } { K } } \Big )
$$

To get the desired result, we note that $\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] \ge$ $\mathbb { E } [ N _ { \epsilon } ( \pi , \nu , T ) ]$ for all $T ,$ and choose $T ^ { * }$ that maximizes the above lower bound. □

Notably, this lower bound holds even for algorithms that obtain both ϵ and δ as input, whereas our algorithm assumes access only to ϵ. Moreover, choosing $\delta = 3 \epsilon .$ , we immediately obtain the following lower bound for when only ϵ is known:

Corollary 2. Let $\epsilon \in ( 0 , 1 / 2 4 ]$ be given. Then for every π, there exists $\nu \in \mathcal { E } _ { B e r } ^ { K }$ such that

$$
\begin{array} { r } { \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \Omega \big ( K / \epsilon ^ { 2 } \big ) . } \end{array}
$$

Proposition 2 highlights a delicate tradeof when ϵ is known. If ϵ is large compared to the instance gaps, one should explore only to ϵ-accuracy. On the other hand, if ϵ is very small, then it is better to aggressively explore to identify optimal arms.

## 5.2 Uniform Algorithms

In contrast to the non-uniform regime, algorithms that do not optimize for a given suboptimality level can enjoy the full benefit of the U-ACE framework, with expected regret guarantees (Claim 2) and almost sure convergence (Corollary 1). We show that these benefits come at the cost of worse expected sample complexity bounds, compared to non-uniform algorithms.

Noting that this regime is strictly harder than the nonuniform regime, the previous lower bounds also hold for uniform algorithms. Here, we present more refined lower bounds for uniform algorithms, showing the setting has an intrinsic polynomial dependence on $1 / \epsilon .$ Recall that a policy π is consistent over bandit class E if for all $\nu \in { \mathcal { E } }$ and $p > 0 \colon$ lim $\begin{array} { r } { r \to \infty \frac { R _ { T } ( \pi , \nu ) } { T ^ { p } } = 0 . } \end{array}$

Proposition 3. Let π be U-ACE and consistent over $\mathcal { E } _ { B e r } ^ { K }$ . Then for any fixed instance $\nu \in \mathcal { E } _ { B e r } ^ { K }$ with means in $[ 1 / 4 , 3 / 4 ] ,$ , it holds that

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \operatorname* { s u p } _ { \epsilon \cdot \mathbb { E } } [ N _ { \epsilon } ( \pi , \nu ) ] \geq \sum _ { a : \Delta _ { a } > 0 } \frac { 3 } { 8 \Delta _ { a } } .
$$

This lower bound shows that the instance-dependent bound for consistent U-ACE algorithms is larger than the proven upper bound for the non-uniform SETϵ- G algorithm. The following proposition shows that a polynomial dependence on 1/ϵ is still present even if we lift the consistency assumption:

Proposition 4. Let π be U-ACE over $\mathcal { E } _ { B e r } ^ { 2 } .$ . Then there exists an instance $\nu \in \mathcal { E } _ { B e r } ^ { 2 }$ such that for any $\alpha \in ( 0 , 1 )$ :

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \operatorname* { s u p } _ { \epsilon } \epsilon ^ { \alpha } \cdot \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \infty .
$$

Proofs for Propositions 3 and 4 can be found in Appendices D.1 and D.2, respectively. These lower bounds show a clear separation from the non-uniform setting.

Next, we analyze the expected sample complexity of Thompson sampling, a widely used algorithm in practice known for its strong empirical performance and robustness, and show that it is U-ACE.

Thompson sampling works by maintaining a prior over the means of all arms, and at each round drawing independent samples from each prior. The arm with the largest sample is chosen, and the observed reward is then used to update the posterior of that arm. The algorithm works well in practice even when the problem is not sampled from the prior distribution, and enjoys near-optimal expected regret guarantees (Agrawal and Goyal, 2017). We show the following:

Theorem 2. Let $\pi ^ { T S ( \mathcal { N } ) }$ be the policy induced by Algorithm 2. Then for all $\epsilon > 0$ and instances ν:

$$
\mathbb { E } \bigg [ N _ { \epsilon } \bigg ( \pi ^ { T S ( \mathcal { N } ) } , \nu \bigg ) \bigg ] = \mathcal { O } \left( \frac { K } { \epsilon } \sum _ { \Delta _ { a } \ge \frac { \epsilon } { K } } \frac { 1 } { \Delta _ { a } } \mathrm { l n } \bigg ( \frac { 3 K \Delta _ { a } } { \epsilon } \bigg ) \right) .
$$

We defer the full proof to Appendix D.3.

Algorithm 2 Thompson sampling with normal priors   
1: Initialize $n _ { 1 } ( a ) = 0 , S _ { 1 } ( a ) = 0 , \hat { \mu } _ { 1 } ( a ) = 0$ for all   
$a \in [ K ]$   
2: for rounds $t = 1 , 2 , \dots$ do   
3: for $a = 1 , \ldots , K$ do   
4: Sample $\begin{array} { r } { \theta _ { t } ( a ) \sim \mathcal { N } \Big ( \hat { \mu } _ { t } ( a ) , \frac { 1 } { 1 + n _ { t } ( a ) } \Big ) } \end{array}$   
5: end for   
6: Play $a _ { t } \in \arg \operatorname* { m a x } _ { a ^ { \prime } } \theta _ { t } ( a ^ { \prime } )$ (break ties arbitrar  
ily) and observe reward X<sub>t</sub>   
7: Update $n _ { t + 1 } ( a _ { t } ) ~ = ~ n _ { t } ( a _ { t } ) + 1 , S _ { t + 1 } ( a _ { t } ) ~ =$   
$\begin{array} { r } { S _ { t } ( a _ { t } ) + X _ { t } , \hat { \mu } _ { t + 1 } ( a _ { t } ) = \frac { S _ { t + 1 } ( a _ { t } ) } { 1 + n _ { t + 1 } ( a _ { t } ) } } \end{array}$   
8: For all $a \neq a _ { t } ,$ set $n _ { t + 1 } ( a ) \stackrel { } { = } n _ { t } ( a ) , S _ { t + 1 } ( a ) \stackrel { } { = }$   
$S _ { t } ( a ) , \hat { \mu } _ { t + 1 } ( a ) = \hat { \mu } _ { t } ( a )$   
9: end for

Proof Sketch. We first aim to manipulate the sample complexity measure to a form that allows using decompositions similar to (Agrawal and Goyal, 2017). In particular, each summand of the expected sample complexity can be decomposed as a sum over the probability of choosing each arm:

## Lemma 8.

$$
\mathrm { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) \le \sum _ { a : \Delta _ { a } \ge \frac { \epsilon } { K } } \mathrm { P r } \bigg ( \pi _ { t } ( a ) > \frac { \epsilon } { K \Delta _ { a } } \bigg ) .
$$

Combined with the Fubini-Tonelli Theorem, this yields

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] \le \sum _ { \Delta _ { a } \ge \frac { \epsilon } { K } } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \biggl ( \pi _ { t } ( a ) > \frac { \epsilon } { K \Delta _ { a } } \biggr ) .
$$

Fixing arm a with $\Delta _ { a } \geq \frac { \epsilon } { K }$ , we decompose the inner sum into 3 terms:

$$
\begin{array} { r l } & { \frac { \infty } { t - 1 } \operatorname* { P r } \bigg ( \pi _ { \mathrm { f } } ( a ) > \frac { \epsilon } { K \Delta _ { a } } \bigg ) } \\ & { \leq \displaystyle \sum _ { \ell = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { \ell } = a , \frac { E _ { a } ^ { \ell } ( t ) | \mathcal { F } _ { \ell - 1 } } { S _ { 1 } } \Big ) > \frac { \epsilon } { 3 K \Delta _ { a } } \bigg ) } \\ & { \quad + \displaystyle \sum _ { \ell = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { \ell } = a , \frac { \mathcal { F } _ { a } ^ { \ell } ( t ) } { S _ { a } } \Big , \frac { \tilde { F } _ { a } ^ { \ell } ( t ) | \mathcal { F } _ { \ell - 1 } } { S _ { 2 } } \Big ) > \frac { \epsilon } { 3 K \Delta _ { a } } \bigg ) } \\ & { \quad + \displaystyle \sum _ { \ell = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { \ell } = a , E _ { a } ^ { \ell } ( t ) , E _ { a } ^ { \ell } ( t ) | \mathcal { F } _ { \ell - 1 } \Big ) > \frac { \epsilon } { 3 K \Delta _ { a } } \bigg ) } \\ & { \quad + \displaystyle \sum _ { \ell = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { \ell } = a , E _ { a } ^ { \ell } ( t ) , E _ { a } ^ { \ell } ( t ) | \mathcal { F } _ { \ell - 1 } \Big ) > \frac { \ell } { 3 K \Delta _ { a } } \bigg ) . } \end{array}
$$

where $\begin{array} { r l r } { E _ { a } ^ { \mu } ( t ) } & { { } \triangleq } & { \{ \hat { \mu } _ { t } ( a ) \leq x _ { a } \} } \end{array}$ and $\begin{array} { r l } { E _ { a } ^ { \theta } ( t ) } & { { } \triangleq } \end{array}$ $\{ \theta _ { t } ( a ) \leq y _ { a } \}$ are the events that the empirical means and the prior samples are not overestimated, respectively. Then, we bound terms $S _ { 1 }$ and $S _ { 2 }$ using similar methods to Agrawal and Goyal (2017). We remark that parts of these derivations require more careful treatment than the original proof, since the sample complexity is defined for an infinite horizon (in contrast to regret); we defer these details to the full proof.

To bound $S _ { 3 }$ , the approach from Agrawal and Goyal (2017) does not trivially generalize to the infinite horizon setting. Moreover, their proof relies on the uniqueness of the optimal arm, which can be assumed w.l.o.g. for regret minimization. Technically, it was proven by showing that adding an optimal arm decreases the regret, since whenever TS plays it, this arm incurs no regret. However, this is no longer the case in our setting; in Appendix D.4, we provide a numerical example in which adding an optimal arm strictly increases the finite-time expected sample complexity. To deal with this, we fix some optimal arm $a ^ { * }$ and define

$$
p _ { a , t } \triangleq \operatorname* { P r } ( \theta _ { t } ( a ^ { * } ) > y _ { a } \vert \mathcal { F } _ { t - 1 } ) , \qquad q _ { a , t } \triangleq \frac { 1 - p _ { a , t } } { p _ { a , t } } .
$$

Then, similarly to Agrawal and Goyal (2017), we show that

$$
\begin{array} { r l } & { \operatorname* { P r } \big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) } \\ & { \qquad \leq q _ { a , t } \operatorname* { P r } \big ( A _ { t } = a ^ { * } , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) . } \end{array}
$$

Applying Markov’s inequality and the above yields

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \Big ( \mathrm { P r } \big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) > \frac { \epsilon } { 3 K \Delta _ { a } } \Big ) } \\ { \displaystyle \qquad \leq \frac { 3 K \Delta _ { a } } { \epsilon } \mathbb { E } \Bigg [ \sum _ { t = 1 } ^ { \infty } q _ { a , t } { 1 } \{ A _ { t } = a ^ { * } \} \Bigg ] , } \end{array}
$$

and noting that ${ { q } _ { a , t } }$ only updates when $A _ { t } = a ^ { * }$ , we decouple our analysis from the rounds t by defining

$$
\begin{array} { l } { \displaystyle \hat { \mu } _ { k } ^ { * } \triangleq \frac { 1 } { k + 1 } \sum _ { s = 1 } ^ { k } X _ { a ^ { * } , s } , } \\ { \displaystyle \tilde { p } _ { a , k } \triangleq \operatorname* { P r } \biggl ( \mathcal { N } \biggl ( \hat { \mu } _ { k } ^ { * } , \frac { 1 } { 1 + k } \biggr ) > y _ { a } \biggl | X _ { a ^ { * } , 1 , \dots , X _ { a ^ { * } , k } } \biggr ) , } \\ { \displaystyle \tilde { q } _ { a , k } \triangleq \frac { 1 - \tilde { p } _ { a , k } } { \tilde { p } _ { a , k } } . } \end{array}
$$

We are then left to show that $\begin{array} { r } { \sum _ { k = 0 } ^ { \infty } \mathbb { E } [ \tilde { q } _ { a , k } ] = \mathcal { O } \Big ( \frac { 1 } { \Delta _ { a } ^ { 2 } } \Big ) } \end{array}$ To do this, we define a sequence of good events $\left\{ H _ { a , k } \right\} _ { k }$ , showing that $\tilde { q } _ { a , k } 1 \{ H _ { a , k } \} \le \mathcal { O } \Bigl ( e ^ { - \Delta _ { a } ^ { 2 } k } \Bigr )$ , and via the Cauchy-Schwarz inequality that

$$
\mathbb { E } \left[ \tilde { q } _ { a , k } \mathbb { 1 } \left\{ \overline { { H _ { a , k } } } \right\} \right] \leq \sqrt { \mathbb { E } \left[ \tilde { q } _ { a , k } ^ { 2 } \right] } \sqrt { \operatorname* { P r } \left( \overline { { H _ { a , k } } } \right) } .
$$

We prove that $\mathrm { P r } \big ( \overline { { H _ { a , k } } } \big ) = \mathcal { O } \Big ( e ^ { - \Delta _ { a } ^ { 2 } k } \Big )$ and show that $\mathbb { E } \left[ \tilde { q } _ { a , k } ^ { 2 } \right]$ is uniformly bounded for all k. This gives us the desired bound on $S _ { 3 }$ . Finally, combining with the bounds for $S _ { 1 }$ and $S _ { 2 }$ , and summing over all arms with $\Delta _ { a } \geq \frac { \epsilon } { K }$ yields the desired result. □

An immediate corollary is an instance-independent upper bound for TS that matches the lower bound in Corollary 2 (up to K factors), which is applicable here since knowing ϵ can only improve our upper bound:

Corollary 3. For all $\epsilon > 0$ , it holds that:

$$
\mathbb { E } \Big [ N _ { \epsilon } \Big ( \pi ^ { T S ( \mathcal { N } ) } \Big ) \Big ] = \mathcal { O } \Big ( \frac { K ^ { 3 } } { \epsilon ^ { 2 } } \Big ) .
$$

The proof is given in Appendix D.3. We attribute part of the $\mathcal { O } \big ( K ^ { 2 } \big )$ factor to Lemma 8, but note that it was key in our analysis. We leave tightening the K-dependence through an alternative approach to fu ture work. This result shows that TS is instanceindependent optimal in ϵ, while Theorem 2 matches the lower bound from Proposition 3 in ϵ, up to logarithmic factors.

## 6 CONCLUSIONS AND FUTURE WORK

This work introduces the expected sample complexity and the ACE framework for its analysis. We showed that U-ACE guarantees imply expected regret bounds and almost-sure convergence to the optimal expected reward, much stronger theoretical guarantees than the popular PAC setting and its various extensions. We also prove that no deterministic policy can be U-ACE uniformly over the K-armed Bernoulli class, revealing a separation between deterministic and stochastic policies that is absent in other frameworks.

We then characterized the behavior of the expected sample complexity, showing an instance-independent rate in ϵ of $\tilde { \Theta } \big ( 1 / \epsilon ^ { 2 } \big )$ both when ϵ is known and unknown. Furthermore, we proved that the instancedependent rates of the two regimes are fundamentally diferent; notably, in the limit of small ϵ, knowing the value of ϵ in advance leads to improved expected sample complexity. To our knowledge, this is the first separation result between known-ϵ and uniform sample complexity bounds.

This work opens up many possible future directions. A natural extension is defining and analyzing the expected sample complexity in other sequential decisionmaking problems; continuing this line of work in tabular RL can open the door to simplifying expected regret analyses together with convergence results for policy-gradient approaches. Another direction would be defining and analyzing the expected sample complexity in the linear bandits model, an extension of the multi-armed bandit problem, in which stochastic policies have been a topic of many recent works.

## AI use statement

The setting, Algorithm 1, and most of the critical results were initially derived by the human authors, in cluding both upper bounds. AI tools were partially used for deriving some of the lower bounds, and some of the original proof arguments were later refined in an iterative interaction with AI tools.

Additionally, we used generative AI tools for the generation of Figure 1 and the code for Figure 2. Finally, AI tools were used to verify the correctness of the proofs and revise the paper before submission. Due to the theoretical nature of our work, the use of AI for generating synthetic data, refining hypotheses, feedback on experiments, translation, dataset cleaning, thematic data analysis, and result interpretation is not applicable.

All LLM-assisted proofs were carefully checked line by line by the authors. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## ACKNOWLEDGMENTS

This research was supported by Israel Science Foundation research grant (ISF’s No. 4118/25) and the Maimonides Fund’s Future Scientists Center.

## References

Milton Abramowitz and Irene A Stegun. Handbook of mathematical functions with formulas, graphs, and mathematical tables, volume 55. US Government printing ofice, 1964.

Shipra Agrawal and Navin Goyal. Near-optimal regret bounds for thompson sampling. Journal of the ACM (JACM), 64(5):1–24, 2017.

Jean-Yves Audibert, S´ebastien Bubeck, and R´emi Munos. Best arm identification in multi-armed bandits. In Conference on learning theory, 2010.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

Djallel Bounefouf and Irina Rish. A survey on practical applications of multi-armed and contextual bandits. arXiv preprint arXiv:1904.10040, 2019.

Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. Advances in neural information processing systems, 24, 2011.

Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. Advances in Neural Information Processing Systems, 30, 2017.

Omar Darwiche Domingues, Pierre M´enard, Emilie Kaufmann, and Michal Valko. Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited. In Algorithmic Learning Theory, pages 578–598. PMLR, 2021.

Aur´elien Garivier, H´edi Hadiji, Pierre Menard, and Gilles Stoltz. Kl-ucb-switch: optimal regret bounds for stochastic bandits from both a distributiondependent and a distribution-free viewpoints. Journal of Machine Learning Research, 23(179):1–66, 2022.

Sham Machandranath Kakade. On the sample complexity of reinforcement learning. University of London, University College London (United Kingdom), 2003.

Emilie Kaufmann, Nathaniel Korda, and R´emi Munos. Thompson sampling: An asymptotically optimal finite-time analysis. In International conference on algorithmic learning theory, pages 199–213. Springer, 2012.

Emilie Kaufmann, Olivier Capp´e, and Aur´elien Garivier. On the complexity of best-arm identification in multi-armed bandit models. The Journal of Machine Learning Research, 17(1):1–42, 2016.

Tze Leung Lai and Herbert Robbins. Asymptotically eficient adaptive allocation rules. Advances in applied mathematics, 6(1):4–22, 1985.

Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020.

Matthias Lehmann. The definitive guide to policy gradients in deep reinforcement learning: Theory, algorithms and implementations. arXiv preprint arXiv:2401.13662, 2024.

Junyan Liu, Yunfan Li, Ruosong Wang, and Lin F Yang. Uniform last-iterate guarantee for bandits and reinforcement learning. Advances in Neural Information Processing Systems, 37:133600–133660, 2024.

Nadav Merlis and Shie Mannor. Lenient regret for multi-armed bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 8950–8957, 2021.

Jan Poland and Marcus Hutter. On the convergence speed of mdl predictions for bernoulli sequences. In International Conference on Algorithmic Learning Theory, pages 294–308. Springer, 2004.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal

policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

N´ıcollas Silva, Heitor Werneck, Thiago Silva, Adriano CM Pereira, and Leonardo Rocha. Multi-armed bandits in recommendation systems: A survey of the state-of-the-art and future directions. Expert Systems with Applications, 197:116669, 2022.

Aleksandrs Slivkins. Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning, 12(1-2):1–286, 2019.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT Press, 1998.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285– 294, 1933.

# Supplementary Materials

## A PREREQUISITES

## A.1 Useful Facts

Fact 1 (Hoefding’s Inequality). Let $X _ { 1 } , \ldots , X _ { n }$ be a sequence of i.i.d. random variables bounded in [0, 1] almost surely with expectation $\mu .$ Denote by $\mu \triangleq \mathbb { E } [ X _ { 1 } ]$ , and define $\begin{array} { r } { S _ { n } = \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ , and $\begin{array} { r } { \hat { \mu } = \frac { S _ { n } } { n } } \end{array}$ . Then for all $t \geq 0$

$$
\operatorname* { P r } ( S _ { n } - n \mu \geq t ) \leq \exp \biggl ( - \frac { 2 t ^ { 2 } } { n } \biggr ) ,
$$

and:

$$
\operatorname* { P r } ( \hat { \mu } - \mu \geq t ) \leq \exp \left( - 2 n t ^ { 2 } \right) .
$$

Fact 2 (Multiplicative Chernof Bound). Let $B _ { 1 } , \ldots , B _ { n }$ be a sequence of independent Bernoulli random variables. Let $\begin{array} { r } { S _ { n } = \sum _ { i = 1 } ^ { n } B _ { i } } \end{array}$ and $\mu = \mathbb { E } [ S _ { n } ]$ . Then for any $\delta \in ( 0 , 1 )$

$$
\operatorname* { P r } ( S _ { n } \leq ( 1 - \delta ) \mu ) \leq \exp \left( - \frac { \delta ^ { 2 } \mu } { 2 } \right) .
$$

Fact 3. For all $x > 0 \colon$

$$
\frac { e ^ { - x } } { 1 - e ^ { - x } } \leq \frac { 1 } { x } , \qquad a n d \qquad \frac { 1 } { 1 - e ^ { - x } } \leq 1 + \frac { 1 } { x } .
$$

Proof. Recall that $1 + x \leq e ^ { x }$ for all $x \in \mathbb { R }$ . Rearrange to get:

$$
{ \frac { 1 } { e ^ { x } - 1 } } \leq { \frac { 1 } { x } } .
$$

Multiplying both numerator and denominator of the L.H.S. by $e ^ { - x }$ gives us the first desired inequality; the second inequality is a direct result of the first, as $\begin{array} { r } { \frac { 1 } { 1 - e ^ { - x } } = \frac { e ^ { - x } } { 1 - e ^ { - x } } + 1 } \end{array}$ □

Fact 4 (Abramowitz and Stegun (1964)). Let $Z \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ be a Gaussian random variable. Then $f o r$ any $t > 0 .$

$$
{ \frac { 1 } { \sqrt { 2 \pi } } } { \frac { t } { 1 + t ^ { 2 } } } e ^ { - t ^ { 2 } / 2 } \leq \operatorname* { P r } ( Z - \mu \geq t \sigma ) \leq { \frac { 1 } { 2 } } e ^ { - t ^ { 2 } / 2 } .
$$

Fact 5 (Lemma 2 of Poland and Hutter (2004)). Let $p , q \in [ 1 / 4 , 3 / 4 ]$ . Then:

$$
D ( B e r ( p ) , B e r ( q ) ) \leq \frac { 8 { ( p - q ) } ^ { 2 } } { 3 } .
$$

Fact 6 (Lai-Robbins lower bound (Lai and Robbins, 1985; Lattimore and Szepesv´ari, 2020)). Let $\mathcal { E } _ { B e r } ^ { K }$ denote the class $o f$ Bernoulli bandits with K arms, and let π be consistent with respect to this class. Then for all $\nu \in \mathcal { E } _ { B e r } ^ { K } .$

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { l n f } _ { \log ( T ) } \geq \sum _ { a : \Delta _ { a } > 0 } \frac { \Delta _ { a } } { d ( \mu _ { a } , \mu ^ { * } ) } ,
$$

where $d ( b , c ) = D ( B e r ( b ) , B e r ( c ) )$ . Consequently, by Fact ${ } ^ { 5 , }$ for instances with means bounded in $[ 1 / 4 , 3 / 4 ]$ :

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { i n f } _ { \log ( T ) } \geq \sum _ { a : \Delta _ { a } > 0 } \frac { 3 } { 8 \Delta _ { a } } ,
$$

Fact 7 (Maximal Hoefding’s inequality). Let $\left( X _ { t } \right) _ { t > 1 }$ be a sequence of random variables adapted to a filtration $\mathcal { F } _ { t }$ such that $Y _ { t } \triangleq X _ { t } - \mathbb { E } [ X _ { t } | \mathcal { F } _ { t - 1 } ] \in [ m _ { t } , M _ { t } ] .$ , with $M _ { t } , m _ { t }$ being F<sub>t−1</sub>-measurable and $M _ { t } - m _ { t } \leq c$ almost surely for all $t , \ f o r$ some absolute constant $c > 0$ . Let $\left( \iota _ { t } \right) _ { t \geq 1 }$ be $\mathcal { F } _ { t - 1 }$ -measurable, and assume that $\iota _ { t } \in \{ 0 , 1 \}$ for all $t \in \mathbb { N }$ . Define $\begin{array} { r } { N _ { t } = \sum _ { s \leq t } \iota _ { s } } \end{array}$ and $\begin{array} { r } { \hat { \mu } _ { t } = \frac { \sum _ { s \leq t } \iota _ { s } Y _ { s } } { N _ { t } \vee 1 } } \end{array}$ . Then for any $n , \epsilon > 0 .$

$$
\operatorname* { P r } ( \exists t : N _ { t } \geq n \ a n d \ | \hat { \mu _ { t } } | \geq \epsilon ) \leq 2 \exp \biggl ( - \frac { 2 n \epsilon ^ { 2 } } { c ^ { 2 } } \biggr ) .
$$

Proof. We note that this is a slightly more general version of Proposition 5 in (Garivier et al., 2022). Let $\begin{array} { r } { Z _ { t } \triangleq \sum _ { s \leq t } \iota _ { s } Y _ { s } } \end{array}$ , and for $x \in \mathbb { R }$ define $\begin{array} { r } { M _ { t } ^ { x } \triangleq \exp \Bigl ( x Z _ { t } - \frac { x ^ { 2 } c ^ { 2 } N _ { t } } { 8 } \Bigr ) } \end{array}$ . We show that $M _ { t } ^ { x }$ is a supermartingale. We have that

$$
\begin{array} { r l } & { \mathbb { E } [ M _ { t } ^ { x } | \mathcal { F } _ { t - 1 } ] = \mathbb { E } \bigg [ \exp \bigg ( x Z _ { t } - \frac { x ^ { 2 } c ^ { 2 } N _ { t } } { 8 } \bigg ) \bigg | \mathcal { F } _ { t - 1 } \bigg ] } \\ & { \quad \quad = \mathbb { E } \left[ \exp \left( x \sum _ { s \leq t } \iota _ { s } Y _ { s } - \frac { x ^ { 2 } c ^ { 2 } N _ { t } } { 8 } \right) \bigg | \mathcal { F } _ { t - 1 } \right] } \\ & { \quad \quad = \exp \left( x \sum _ { s \leq t - 1 } \iota _ { s } Y _ { s } - \frac { x ^ { 2 } c ^ { 2 } N _ { t - 1 } } { 8 } \right) \mathbb { E } \left[ \exp \left( x \iota _ { t } Y _ { t } - \frac { x ^ { 2 } c ^ { 2 } \iota _ { t } } { 8 } \right) \bigg | \mathcal { F } _ { t - 1 } \right] } \\ & { \quad \quad = M _ { t - 1 } ^ { x } \exp \left( - \frac { x ^ { 2 } c ^ { 2 } \iota _ { t } } { 8 } \right) \mathbb { E } [ \exp ( x \iota _ { t } Y _ { t } ) | \mathcal { F } _ { t - 1 } ] , } \end{array}\tag{2}
$$

where the final equality is by the definition of $M _ { t - 1 } ^ { x }$ and since $\iota _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable. Note that $\iota _ { t } Y _ { t } \in [ \iota _ { t } m _ { t } , \iota _ { t } M _ { t } ]$ almost surely, and that $\mathbb { E } [ \iota _ { t } Y _ { t } | \mathcal { F } _ { t - 1 } ] = \iota _ { t } \mathbb { E } [ Y _ { t } | \mathcal { F } _ { t - 1 } ] = 0$ , and so we can apply the conditional version of Hoefding’s lemma on $\iota _ { t } Y _ { t } \in [ \iota _ { t } m _ { t } , \iota _ { t } M _ { t } ]$ to get

$$
\mathbb { E } [ \exp ( x \iota _ { t } Y _ { t } ) | \mathcal { F } _ { t - 1 } ] \leq \exp \biggl ( \frac { x ^ { 2 } \iota _ { t } ^ { 2 } c ^ { 2 } } { 8 } \biggr ) = \exp \biggl ( \frac { x ^ { 2 } \iota _ { t } c ^ { 2 } } { 8 } \biggr ) ,
$$

where the equality holds since $\iota _ { t } \in \{ 0 , 1 \}$ , and thus $\iota _ { t } ^ { 2 } = \iota _ { t }$ . Substituting back into (2), we get that $\mathbb { E } [ M _ { t } ^ { x } | \mathcal { F } _ { t - 1 } ] \le$ $M _ { t - 1 } ^ { x }$ , and so $M _ { t } ^ { x }$ is indeed a supermartingale. As such, it holds that $\mathbb { E } [ M _ { t } ^ { x } ] \leq \mathbb { E } [ M _ { 0 } ^ { x } ] = 1$ ; noting that $M _ { t } ^ { x }$ is also non-negative, we can apply Ville’s inequality and get that for all $u > 0$

$$
\operatorname* { P r } ( \exists t : M _ { t } ^ { x } \geq u ) \leq \frac { \mathbb { E } [ M _ { 0 } ^ { x } ] } { u } \leq \frac { 1 } { u } .\tag{3}
$$

Therefore, for all $\epsilon > 0$ and $n \in \mathbb { N }$ , taking $\textstyle x = { \frac { 4 \epsilon } { c ^ { 2 } } }$ yields

$$
\begin{array} { r l r } { \operatorname* { P r } ( \exists t : Z _ { t } \geq N _ { t } \epsilon , N _ { t } \geq n ) = \operatorname* { P r } \left( \exists t : \exp \left( x Z _ { t } - \frac { x ^ { 2 } c ^ { 2 } N _ { t } } { 8 } \right) \geq \exp \left( x N _ { t } \epsilon - \frac { x ^ { 2 } c ^ { 2 } N _ { t } } { 8 } \right) , N _ { t } \geq n \right) } & \\ & { \qquad = \operatorname* { P r } \left( \exists t : M _ { t } ^ { x } \geq \exp \left( x N _ { t } \epsilon - \frac { x ^ { 2 } c ^ { 2 } N _ { t } } { 8 } \right) , N _ { t } \geq n \right) } & \\ & { \qquad = \operatorname* { P r } \left( \exists t : M _ { t } ^ { x } \geq \exp \left( \frac { 2 } { c ^ { 2 } } N _ { t } \epsilon ^ { 2 } \right) , N _ { t } \geq n \right) } & { \mathrm { ( d e f i n i t i o n ~ o f ~ } x \mathrm { ) } } \\ & { \qquad \leq \operatorname* { P r } \left( \exists t : M _ { t } ^ { x } \geq \exp \left( \frac { 2 } { c ^ { 2 } } n \epsilon ^ { 2 } \right) \right) } & \\ & { \qquad \leq \exp \left( - \frac { 2 n \epsilon ^ { 2 } } { c ^ { 2 } } \right) . } & { \mathrm { ( E q u a t i o n ~ ( 3 ) ) } } \end{array}
$$

We can apply similar arguments to get $\begin{array} { r } { \operatorname* { P r } ( \exists t : Z _ { t } \le - N _ { t } \epsilon , N _ { t } \ge n ) \ \le \ \exp \left( - \frac { 2 n \epsilon ^ { 2 } } { c ^ { 2 } } \right) } \end{array}$ , and finally noting that $\hat { \mu } _ { t } = Z _ { t } / N _ { t }$ gives us that for all $n , \epsilon > 0$

2nϵ<sup>2</sup> Pr(∃t $: N _ { t } \geq n$ and |µˆ<sub>t</sub>| ≥ ϵ) ≤ Pr(∃t : Z<sub>t</sub> ≤ −N<sub>t</sub>ϵ, N<sub>t</sub> ≥ n) + Pr(∃t : Z<sub>t</sub> ≥ N<sub>t</sub>ϵ, N<sub>t</sub> ≥ n) ≤ 2 exp c<sup>2</sup> as desired.

## A.2 Interaction Protocol and Filtrations

One of the distinguishing features of this work is the analysis of the policy itself rather than the realized action.   
This raises the need to carefully define how histories and internal randomization of the algorithm work.

We explicitly represent the internal randomization of the algorithm by a sequence of mutually independent random variables $( U _ { t } ) _ { t \geq 1 }$ , independent of the reward table, such that $U _ { t }$ contains all randomization used by the algorithm in round t before observing the reward. We let

$$
\mathcal { F } _ { t } \triangleq \sigma ( A _ { 1 } , X _ { 1 } , \dots , A _ { t } , X _ { t } ) ,
$$

denote the history up to the end of round $t ,$ with $\mathcal { F } _ { 0 }$ taken to be the trivial sigma-algebra. At the beginning of round t, the policy $\pi _ { t }$ is $\mathcal { F } _ { t - 1 } .$ -measurable, and conditioned on the history,

$$
A _ { t } | { \mathcal { F } } _ { t - 1 } \sim \pi _ { t } ( \cdot | { \mathcal { F } } _ { t - 1 } ) .
$$

Noting that the action $A _ { t }$ is completely determined by $\mathcal { F } _ { t - 1 }$ and $U _ { t } .$ , and since the action space is finite, we may write

$$
A _ { t } = \Phi _ { t } ( \pi _ { t } , U _ { t } )
$$

for some measurable map $\Phi _ { t }$ . For $t \geq 0$ , we define the filtration

$$
\mathcal { G } _ { t } \triangleq \sigma ( \mathcal { F } _ { t } , U _ { 1 } , \dots , U _ { t + 1 } )
$$

to be the history available at round t together with the realized internal randomness. Hence, immediately before observing the reward in round t, the available information is $\mathcal { G } _ { t - 1 }$ . In particular, all random choices determining the action in round t are in $\mathcal { G } _ { t - 1 }$ , and in particular $A _ { t }$ is G -measurable, giving us that $\mathbb { E } [ X _ { t } | \mathcal { G } _ { t - 1 } ] = \mu _ { A _ { t } }$

![](images/44dff2f73e0f1020a71ff8ab02162ea898e188da237c32a0bcc08655a1144e7e.jpg)  
Figure 1: Flow of randomness in round t of the interaction. The policy $\pi _ { t }$ is determined from the past history $\mathcal { F } _ { t - 1 }$ . The internal randomization $U _ { t }$ is then revealed, making the realized action $A _ { t }$ and any other internal choice mechanism predictable before the reward $X _ { t }$ is observed.

## A.3 The Table Model

The sequential nature of the multi-armed bandit problem requires a delicate approach to the analysis of algo rithms. Towards this, in the analysis of all our algorithms, we adopt an infinite table model to generate our samples; details on why a policy is well-defined for such a construction can be found in (Lattimore and Szepesv´ari, 2020, Section 4.6). Before the interaction, we generate $( \tilde { X } _ { a , k } ) _ { a \in [ K ] , k \in \mathbb { N } }$ such that the law of $\tilde { X } _ { a , k }$ is $\nu _ { a }$ (independent of all other samples). During the interaction, we let $X _ { t } = \tilde { X } _ { A _ { t } , n _ { t } ( A _ { t } ) + 1 }$ , noting that $A _ { t } , n _ { t } ( A _ { t } )$ are $\mathcal { G } _ { t - 1 }$ -measurable, thus making this construction valid. We further denote by $\begin{array} { r } { \tilde { \mu } _ { a , k } \triangleq \frac { 1 } { k } \sum _ { s = 1 } ^ { k } \tilde { X } _ { a , s } } \end{array}$ the empirical mean of the first k samples from the row in the table representing arm a.

## B PROPERTIES OF U-ACE – OMITTED PROOFS

## B.1 Proof of claim 1

Claim 1. It holds that $\begin{array} { r } { \mathbb { i } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { t } \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { \pi _ { t } } \right] , } \end{array}$

Proof. First, observe that the expected regret can be rewritten as

$$
R _ { T } ( \pi , \nu ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { t } \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { a \in [ K ] } \Delta _ { t } \mathbb { 1 } \{ A _ { t } = a \} \right] = \sum _ { a \in [ K ] } \Delta _ { a } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ A _ { t } = a \} \right] = \sum _ { a \in [ K ] } \Delta _ { a } \mathbb { E } [ n _ { T + 1 } ( a ) ] .
$$

Furthermore, observe that for all $a \in [ K ] { \mathrm { : } }$

$$
\mathbb { E } [ n _ { T + 1 } ( a ) ] \stackrel { ( 1 ) } { = } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ A _ { t } = a \right\} \right] = \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathbb { 1 } \left\{ A _ { t } = a \right\} ] = \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathrm { P r } ( A _ { t } = a | \mathcal { F } _ { t - 1 } ) ] \stackrel { ( 2 ) } { = } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \pi _ { t } ( a \mid \mathcal { F } _ { t - 1 } ) ] .
$$

where (1) is by definition and (2) is also by definition. Combining the above yields

$$
\sum _ { \alpha \in [ K ] } \Delta _ { \alpha } \mathbb { E } [ n _ { T + 1 } ( a ) ] = \sum _ { \alpha \in [ K ] } \Delta _ { \alpha } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \pi _ { t } ( a \mid { \mathcal F } _ { t - 1 } ) ] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { a \in [ K ] } \Delta _ { a } \pi _ { t } ( a \mid { \mathcal F } _ { t - 1 } ) \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { \pi _ { t } } \right] ,
$$

where the last equality is by definition.

## B.2 Proof of Proposition 1

Proposition 1. Let $\pi = \left( \pi _ { t } \right) _ { t = } ^ { \infty }$ <sub>1</sub> be ϵ-ACE for some $\epsilon > 0$ . Then lim su $\begin{array} { r } { \mathrm { p } _ { t  \infty } \Delta _ { \pi _ { t } } \leq \epsilon } \end{array}$ almost surely.

Proof. First, note that for any $\epsilon > 0 .$ , by Fubini-Tonelli:

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \Im \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] = \sum _ { t = 1 } ^ { \infty } \mathbb { E } [ \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} ] = \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) .
$$

By ϵ-ACE and the above, we have that

$$
\sum _ { t = 1 } ^ { \infty } \mathrm { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) < \infty ,
$$

and so by the first lemma of Borel-Cantelli:

$$
\mathrm { P r } ( \forall \tau , \exists t > \tau \ s . t . \ \Delta _ { \pi _ { t } } > \epsilon ) = 0 ,
$$

which in turn implies that

$$
1 = \operatorname* { P r } ( { \overline { { A } } } ) = \operatorname* { P r } ( { \exists } { \tau } ~ s . t . ~ \forall t > { \tau } : \{ \Delta _ { \pi _ { t } } \leq \epsilon \} ) \leq \operatorname* { P r } ( \operatorname* { l i m s u p } _ { t  \infty } \Delta _ { \pi _ { t } } \leq \epsilon ) = 1 ,
$$

giving us our result.

Corollary 1. Let $\pi = \left( \pi _ { t } \right) _ { t = } ^ { \infty }$ be U-ACE. Then $\begin{array} { r } { \operatorname* { l i m } _ { t  \infty } \Delta _ { \pi _ { t } } = 0 } \end{array}$ almost surely.

Proof. We apply the fact that π is U-ACE for the sequence $\begin{array} { r } { \epsilon _ { k } = \frac { 1 } { k } } \end{array}$ , and combine with Proposition 1 to get

$$
\forall k \in \mathbb { N } : \operatorname* { P r } ( \operatorname* { l i m } _ { t  \infty } \Delta _ { \pi _ { t } } \leq { \frac { 1 } { k } } ) = 1 .
$$

Recalling that a countable intersection of almost-sure events is an almost-sure event, the above gives us

$$
\operatorname* { P r } \left( \bigcap _ { k = 1 } ^ { \infty } \left\{ \operatorname* { l i m } _ { t \to \infty } \Delta _ { \pi _ { t } } \leq { \frac { 1 } { k } } \right\} \right) = 1 .
$$

Noting that $\Delta _ { \pi _ { t } } \geq 0$ by definition, and that

$$
\bigcap _ { k = 1 } ^ { \infty } \Bigl \{ \operatorname* { l i m } _ { t \to \infty } \operatorname* { s u p } _ { \pi _ { t } } \leq \frac { 1 } { k } \Bigr \} \subseteq \Bigl \{ \operatorname* { l i m } _ { t \to \infty } \operatorname* { s u p } _ { \pi _ { t } } \leq 0 \Bigr \} ,
$$

yields $\begin{array} { r } { \operatorname* { P r } ( \operatorname* { l i m } _ { t  \infty } \Delta _ { \pi _ { t } } = 0 ) = 1 } \end{array}$ , as desired.

## B.3 Proof of Claim 2

Claim 2. Assume that π is U-ACE with $\begin{array} { r } { m _ { U - A C E } ( \epsilon , \nu ) \leq \frac { C _ { \nu } } { \epsilon ^ { \alpha } } \ f o r \ C _ { \nu } > 0 , \ \alpha > 1 } \end{array}$ . Then for all T, it holds that $\begin{array} { r } { R _ { T } ( \pi , \nu ) \leq \frac { \alpha } { \alpha - 1 } C _ { \nu } ^ { 1 / \alpha } T ^ { \frac { \alpha - 1 } { \alpha } } } \end{array}$

Moreover, $f o r \alpha = 1$ , we obtain a logarithmic bound of $\begin{array} { r } { R _ { T } ( \pi , \nu ) \leq C _ { \nu } \Big ( 1 + \operatorname* { m a x } \Big \{ 0 , \ln \Big ( \frac { T } { C _ { \nu } } \Big ) \Big \} \Big ) } \end{array}$

Proof. Following a similar decomposition to Claim 1 in Merlis and Mannor (2021):

$$
R _ { T } ( \pi ) = \mathbb { R } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { \tau _ { t } } \right] = \mathbb { K } \left[ \sum _ { t = 1 } ^ { T } \int _ { \epsilon = 0 } ^ { 1 } \mathbb { I } \left\{ \Delta _ { \tau _ { t } } > \epsilon \right\} d \epsilon \right] = \int _ { \epsilon = 0 } ^ { 1 } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { I } \left\{ \Delta _ { \tau _ { t } } > \epsilon \right\} \right] d \epsilon \leq \int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \Bigg \{ T , \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { I } \left\{ \Delta _ { \tau _ { t } } > \epsilon \right\} \right] \Bigg \} d \epsilon ,
$$

which by definition implies that

$$
R _ { T } ( \pi ) \leq \int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon } ( \pi ) ] \} d \epsilon .
$$

We begin with the case that $\alpha > 1$ . Note that for $T \leq C _ { \nu } ,$ the bound holds trivially. For $T > C _ { \nu } ,$ we have that $\left( C _ { \nu } / T \right) ^ { 1 / \alpha } < 1$ , and we are given that $\mathbb { E } [ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} ] \le C _ { \nu } \epsilon ^ { - \alpha }$ for all $\epsilon > 0$ , which yields

$$
\int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon } ( \pi ) ] \} d \epsilon \leq \int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , C _ { \nu } \epsilon ^ { - \alpha } \} d \epsilon \leq \int _ { \epsilon = 0 } ^ { ( C _ { \nu } / T ) ^ { 1 / \alpha } } T d \epsilon + \int _ { \epsilon = ( C _ { \nu } / T ) ^ { 1 / \alpha } } ^ { 1 } C _ { \nu } \epsilon ^ { - \alpha } d \epsilon ,
$$

and a simple computation gives us

$$
\int _ { \epsilon = 0 } ^ { ( C _ { \nu } / T ) ^ { 1 / \alpha } } T d \epsilon + \int _ { \epsilon = ( C _ { \nu } / T ) ^ { 1 / \alpha } } ^ { 1 } C _ { \nu } \epsilon ^ { - \alpha } d \epsilon = C _ { \nu } ^ { 1 / \alpha } T ^ { \frac { \alpha - 1 } { \alpha } } + \frac { C _ { \nu } } { 1 - \alpha } \Big ( 1 - ( C _ { \nu } / T ) ^ { \frac { 1 - \alpha } { \alpha } } \Big ) \leq \frac { \alpha } { \alpha - 1 } C _ { \nu } ^ { 1 / \alpha } T ^ { \frac { \alpha - 1 } { \alpha } } .
$$

For $\alpha = 1$ , we note that if $C _ { \nu } \geq T$ , then $R _ { T } ( \pi ) \leq T \leq C _ { \nu }$ and so the bound holds. For $C _ { \nu } < T$ , we have that:

$$
\int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon } ( \pi ) ] \} d \epsilon \leq \int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , C _ { \nu } \epsilon ^ { - 1 } \} d \epsilon \leq \int _ { \epsilon = 0 } ^ { ( C _ { \nu } / T ) } T d \epsilon + \int _ { \epsilon = ( C _ { \nu } / T ) } ^ { 1 } C _ { \nu } \epsilon ^ { - 1 } d \epsilon ,
$$

and thus

$$
\int _ { \epsilon = 0 } ^ { ( C _ { \nu } / T ) } T d \epsilon + \int _ { \epsilon = ( C _ { \nu } / T ) } ^ { 1 } C _ { \nu } \epsilon ^ { - 1 } d \epsilon \leq C _ { \nu } + C _ { \nu } \log \left( \frac { T } { C _ { \nu } } \right) .
$$

## B.4 Proof of Claim 3

Claim 3. Let $\mathcal { E } _ { B e r } ^ { K }$ be the class of K-armed Bernoulli bandits. For any deterministic policy π and $K \geq 2$ , there exists $\nu \in \mathcal { E } _ { B e r } ^ { K }$ and $\epsilon > 0 ~ s . t . ~ \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \infty$

Proof. We know the policy is a deterministic mapping from $\mathcal { F } _ { t - 1 } \mathrm { ~ t o ~ } [ K ]$ , and as such for any history there exists $a \in [ K ]$ such that $\pi _ { t } ( a | \mathcal { F } _ { t - 1 } ) = 1$ . Further, for any instance ν and for all $\epsilon < \Delta _ { \mathrm { m i n } }$

$$
R _ { T } ( \pi , \nu ) \overset { ( 1 ) } { = } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Delta _ { A _ { t } } \right] \overset { ( 2 ) } { \leq } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \Delta _ { A _ { t } } > \epsilon \} \right] \overset { ( 3 ) } { \leq } \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { A _ { t } } > \epsilon \} \right] \overset { ( 4 ) } { = } \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] = \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] ,
$$

where (1) is by definition, (2) is since each time an arm with $\Delta _ { A _ { t } } \ > \epsilon$ is chosen, it incurs at most 1 regret (since the distributions are Bernoulli), (3) is by the monotone convergence theorem and the non-negativity of the indicators, and (4) is since the policy is deterministic. Using this lower bound, we divide into two cases:

$\mathbf { I f } \ \pi$ is consistent. Fix $\nu \in \mathcal { E } _ { B e r } ^ { K }$ with $\Delta _ { \operatorname* { m i n } } > 0$ and $\mu ^ { * } < 1$ . From the lower bound of Lai-Robbins (Fact 6) we have that:

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { i n f } _ { \mathrm { ~ \ln ( } T ) } > 0 ,
$$

which in turn implies that lim $_ { 1 T  \infty } R _ { T } ( \pi , \nu ) = \infty$ , and thus by the lower bound $\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \infty .$ , proving that π cannot be $\mathrm { \epsilon { - A C E } }$ in this case.

If π is not consistent. Then there exists $p , c > 0$ , instance $\nu \in \mathcal { E } _ { B e r } ^ { K }$ and a strictly increasing sequence of natural numbers $\{ T _ { i } \} _ { i \in \mathbb { N } }$ such that $R _ { T _ { i } } ( \pi , \nu ) \geq c T _ { i } ^ { p }$ . Together with the monotonicity of the expected regret, this gives lim<sub>T</sub> $\ldots { \tilde { R } } _ { T } ( \pi , \nu ) = \infty$ . Thus for all $\epsilon < \Delta _ { \mathrm { m i n } }$ , by the above we will again get $\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \infty .$ , proving our desired result. □

## C NON-UNIFORM ALGORITHMS – OMITTED PROOFS

## C.1 Proof of Theorem 1

Theorem 1. Let $\epsilon > 0$ , and let $\pi ^ { S E T \in - G }$ denote the policy induced by Algorithm 1. Then for all ν:

$$
\mathbb { E } \Big [ N _ { \epsilon } \Big ( \pi ^ { S E T \epsilon - G } , \nu \Big ) \Big ] = \mathcal { O } \left( \sum _ { \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 1 } { \Delta _ { a } ^ { 2 } } \ln \left( \frac { K } { \epsilon ^ { 2 } } \right) \right) .
$$

Proof. For brevity, we write $\pi = \pi ^ { \mathrm { S E T } \epsilon - \mathrm { G } }$ . Note that when $\epsilon \geq 1$ , since $\Delta _ { \pi _ { t } } \le 1$ , we have that $N _ { \epsilon } ( \pi ) = 0 \mathrm { a . s . }$ 2 thus $\mathbb { E } [ N _ { \epsilon } ( \pi ) ] = 0$ , which trivially satisfies the upper bound. We can therefore assume that $\epsilon \in ( 0 , 1 )$ and is known to the algorithm. We denote

$$
\begin{array} { r } { \begin{array} { r l r } { \mathscr { A } ^ { * } \triangleq \{ a \in [ K ] : \Delta _ { a } = 0 \} , \quad } & { { } \mathscr { A } _ { s u b } \triangleq \{ a \in [ K ] : \Delta _ { a } > 0 \} , } \end{array} } \end{array}
$$

noting that if $\mathcal { A } _ { \mathrm { s u b } } = \emptyset$ , then again $N _ { \epsilon } ( \pi ) = 0 \mathrm { a . s . }$ , thus $\mathbb { E } [ N _ { \epsilon } ( \pi ) ] = 0$ , which trivially satisfies the upper bound. We henceforth assume that $\mathcal { A } _ { \mathrm { s u b } } \neq \emptyset$ , let $q \triangleq \frac { \epsilon ^ { 2 } } { 1 6 K }$ and define for all $n \geq 1$ the confidence radius

$$
\beta _ { n } \triangleq \sqrt { \frac { 1 } { 2 n } \ln \biggl ( \frac { 4 K n ^ { 2 } } { q } \biggr ) } = \sqrt { \frac { 1 } { 2 n } \ln \biggl ( \frac { 6 4 K ^ { 2 } n ^ { 2 } } { \epsilon ^ { 2 } } \biggr ) } .
$$

Lemma 1. $\beta _ { n }$ is decreasing in $n .$

Proof of this lemma can be found in Appendix C.2. For $x \in ( 0 , 1 ]$ , define $\begin{array} { r } { n ( x ) \triangleq \operatorname* { m i n } \left\{ n \geq 1 : \beta _ { n } < \frac { x } { 8 } \right\} } \end{array}$ , and let $\begin{array} { l } { n _ { \epsilon } \triangleq n { \left( \frac { \epsilon } { 4 } \right) } } \end{array}$ be the maximal number of successive-elimination epochs. For our analysis, we define the good event to be

$$
G \triangleq \bigcap _ { a \in [ K ] } \bigcap _ { n = 1 } ^ { \infty } \{ | { \hat { \mu } } _ { a , n } - \mu _ { a } | \leq \beta _ { n } \} .
$$

Note that by Hoefding’s inequality (Fact 1) and the definition of $\beta _ { n }$ :

$$
\operatorname* { P r } ( \left| { \hat { \mu } } _ { a , n } - \mu _ { a } \right| > \beta _ { n } ) \leq 2 e ^ { - 2 n \beta _ { n } ^ { 2 } } = 2 \exp \biggl ( - \ln \biggl ( \frac { 4 K n ^ { 2 } } { q } \biggr ) \biggr ) = \frac { q } { 2 K n ^ { 2 } } ,
$$

and so by the union bound,

$$
\operatorname* { P r } ( { \overline { { G } } } ) \leq \sum _ { a = 1 } ^ { K } \sum _ { n = 1 } ^ { \infty } { \frac { q } { 2 K n ^ { 2 } } } = { \frac { q } { 2 } } \sum _ { n = 1 } ^ { \infty } { \frac { 1 } { n ^ { 2 } } } \leq q .
$$

We decompose the expected sample complexity of $\pi$ into the two phases: the successive elimination (SE) phase and the ϵ-Greedy (ϵ-G) phase:

$$
\mathbb { E } [ N _ { \epsilon } ( \pi ) ] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \right] = \underbrace { \mathbb { E } \left[ \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \right] } _ { S E } + \underbrace { \mathbb { E } \left[ \sum _ { t = T _ { S E } + 1 } ^ { \infty } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \right] } _ { \epsilon - G } ,
$$

where $T _ { S E }$ denotes the round in which the SE phase terminated. We analyze each phase of the algorithm separately, starting with the SE phase.

Our analysis follows a simple argument, showing that under the good event, bad arms are eliminated proportionally to their suboptimality, while the probability of the bad event balances the worst-case expected sample complexity in this phase.

Optimal arms are not eliminated under $G$ Let $a ^ { * } \in { \mathcal { A } } ^ { * }$ be some optimal arm. We show that under the event $G , a ^ { * }$ is never eliminated in the SE phase. Suppose $a ^ { * }$ is active at epoch n. For any other active arm $b ,$ it holds that:

$$
\hat { \mu } _ { b , n } \leq \mu _ { b } + \beta _ { n } \leq \mu ^ { * } + \beta _ { n } .
$$

Moreover, note that $\hat { \mu } _ { a ^ { * } , n } \geq \mu ^ { * } - \beta _ { n }$ , and thus $\hat { \mu } _ { b , n } - \hat { \mu } _ { a ^ { * } , n } \leq 2 \beta _ { n } < 6 \beta _ { n } .$ implying that $a ^ { * }$ is not eliminated at the end of epoch n. This is true for any n, and since all arms are active in the first epoch, we have that $a ^ { * }$ is never eliminated in the SE phase under the good event $G .$

Suboptimal arms are eliminated proportionally to their suboptimality under $G$ Now, fix a suboptimal arm $a \in \mathcal { A } _ { s u b }$ . We show that under G, a is eliminated after at most $n ( \Delta _ { a } )$ epochs. Suppose a is active at epoch $n = n ( \Delta _ { a } )$ . Thus, on $G \mathrm { : }$

$$
\hat { \mu } _ { a ^ { * } , n } - \hat { \mu } _ { a , n } \geq \mu ^ { * } - \beta _ { n } - \left( \mu _ { a } + \beta _ { n } \right) = \Delta _ { a } - 2 \beta _ { n } .
$$

By the definition of $n ,$ we have that $\begin{array} { r } { \beta _ { n } < \frac { \Delta _ { a } } { 8 } } \end{array}$ , thus we get

$$
\hat { \mu } _ { a ^ { * } , n } - \hat { \mu } _ { a , n } \geq \Delta _ { a } - \frac { \Delta _ { a } } { 4 } = \frac { 3 \Delta _ { a } } { 4 } .
$$

Since $6 \beta _ { n } \leq \frac { 3 \Delta _ { a } } { 4 }$ , a is therefore eliminated by $a ^ { * }$ after at most $n ( \Delta _ { a } )$ epochs if $n _ { \epsilon } > n ( \Delta _ { a } )$

This implies that on $G ,$ every suboptimal arm a is pulled for at most min $\{ n ( \Delta _ { a } ) , n _ { \epsilon } \}$ epochs during the SE phase. Moreover, since the policy in this phase is deterministic, we have that:

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \mathbb { 1 } \left\{ G \right\} \right] \leq \sum _ { a : \Delta _ { a } > \epsilon } \operatorname* { m i n } \{ n ( \Delta _ { a } ) , n _ { \epsilon } \} \leq \sum _ { a : \Delta _ { a } > \epsilon } n ( \Delta _ { a } ) ,
$$

noting that if all arms have $\Delta _ { a } \leq \epsilon ,$ , then $\Delta _ { \pi _ { t } } \le \epsilon$ and as such $N _ { \epsilon } ( \pi ) = 0$ , aligning with the above. Furthermore, we can trivially bound the sample complexity by noting that each epoch contains at most K rounds, and the total number of epochs is bounded by $n _ { \epsilon } ,$ implying that $\begin{array} { r } { \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \le K n _ { \epsilon } } \end{array}$ almost surely, which gives us that

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \mathbb { 1 } \left\{ \overline { { G } } \right\} \right] \leq \mathrm { P r } ( \overline { { G } } ) K n _ { \epsilon } \leq q K n _ { \epsilon } .
$$

We give a bound for $n ( x )$ in the following lemma:

Lemma 2. For every $x \in ( 0 , 1 ]$

$$
n ( x ) \leq \bigg \lceil \frac { 2 0 4 8 } { x ^ { 2 } } \ln \bigg ( \frac { 1 0 2 4 K } { \epsilon x } \bigg ) \bigg \rceil , \qquad a n d \qquad n _ { \epsilon } \leq \bigg \lceil \frac { 3 2 7 6 8 } { \epsilon ^ { 2 } } \ln \bigg ( \frac { 4 0 9 6 K } { \epsilon ^ { 2 } } \bigg ) \bigg \rceil .
$$

Proof of Lemma 2 can be found in Appendix C.2. Applying this result to the previous bound, we $\mathrm { g e t }$

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \right] } & { = \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \mathbb { 1 } \left\{ G \right\} \right] + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \mathbb { 1 } \left\{ \overline { { G } } \right\} \right] } \\ & { \leq \displaystyle \sum _ { a : \Delta _ { a } > \epsilon } n ( \Delta _ { a } ) + q K n _ { \epsilon } } \\ & { \leq \displaystyle \sum _ { a : \Delta _ { a } > \epsilon } \left[ \displaystyle \frac { 2 0 4 8 } { \Delta _ { a } ^ { 2 } } \ln \left( \displaystyle \frac { 1 0 2 4 K } { \epsilon \Delta _ { a } } \right) \right] + \displaystyle \frac { \epsilon ^ { 2 } } { 1 6 } \left[ \displaystyle \frac { 3 2 7 6 8 } { \epsilon ^ { 2 } } \ln \left( \displaystyle \frac { 4 0 9 6 K } { \epsilon ^ { 2 } } \right) \right] , } \end{array}
$$

implying that

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] = \mathcal { O } \left( \sum _ { a : \Delta _ { a } > \epsilon } \frac { 1 } { \Delta _ { a } ^ { 2 } } \ln \left( \frac { K } { \epsilon \Delta _ { a } } \right) + \ln \left( \frac { K } { \epsilon ^ { 2 } } \right) \right) .\tag{4}
$$

Noting that for $\Delta _ { a } > \epsilon$ we have ln $\left( \frac { K } { \epsilon \Delta _ { a } } \right) \le \ln \left( \frac { K } { \epsilon ^ { 2 } } \right)$ , we can absorb the additional term since $\begin{array} { r } { \frac { 1 } { \Delta _ { a } ^ { 2 } } \geq 1 } \end{array}$ to get the desired bound of $\begin{array} { r } { \mathcal { O } \Big ( \sum _ { a : \Delta _ { a } > \epsilon } \frac { 1 } { \Delta _ { a } ^ { 2 } } \ln \left( \frac { K } { \epsilon ^ { 2 } } \right) \Big ) } \end{array}$

Next, we analyze the ϵ-greedy phase of the algorithm.

Our analysis is again relatively simple; we show that the probability of a bad arm being empirically best decays exponentially in the number of rounds, but is also bounded by $q ,$ which creates a balance between the factors in ϵ and in $\Delta _ { a }$ , which leads us to our desired result.

Let $s = 1 , 2 , . . .$ . denote the time in the ϵ-greedy phase, and $g _ { s }$ denote the greedy arm at round s. By the definition of $\pi _ { s } ,$ we have that

$$
\Delta _ { \pi _ { s } } = \Bigl ( 1 - { \frac { \epsilon } { 4 } } \Bigr ) \Delta _ { g _ { s } } + { \frac { \epsilon } { 4 K } } \sum _ { a \in [ K ] } \Delta _ { a } \le \Delta _ { g _ { s } } + { \frac { \epsilon } { 4 } } ,
$$

which implies that $\{ \Delta _ { \pi _ { s } } > \epsilon \} \subseteq \left\{ \Delta _ { g _ { s } } > \frac { 3 \epsilon } { 4 } \right\}$ . We show the following:

Lemma 3. For all s and for any arm with $\textstyle \Delta _ { a } > { \frac { 3 \epsilon } { 4 } }$ , it holds that

$$
\operatorname* { P r } ( g _ { s } = a ) \leq q .
$$

Proof. We begin by showing that under $G ,$ if an arm survives the SE phase, then it must hold that $\Delta _ { a } \leq \epsilon / 4$

Let a be an arm that is not eliminated in the SE phase. We divide into cases:

If the SE phase did not terminate early, then by definition we terminate at epoch $n _ { \epsilon } .$ , in which we have $\beta _ { n _ { \epsilon } } \leq \frac { \epsilon } { 3 2 }$ Suppose towards contradiction that $\Delta _ { a } > \epsilon / 4 \geq 8 \beta _ { n }$ . For arm a that survived to epoch $n _ { \epsilon }$ , we have that on $G$

$$
\hat { \mu } _ { a ^ { * } , n _ { \epsilon } } - \hat { \mu } _ { a , n _ { \epsilon } } \geq \mu ^ { * } - \beta _ { n } - \left( \mu _ { a } + \beta _ { n } \right) = \Delta _ { a } - 2 \beta _ { n } > 6 \beta _ { n _ { \epsilon } } ,
$$

which implies that a is eliminated, which is a contradiction. Thus $\Delta _ { a } \leq \frac { \epsilon } { 4 }$

If the SE phase did terminate early, then by definition there is only one active arm remaining, which must be the optimal arm, since an optimal arm is never eliminated on $G .$

In either case, for all surviving arms, it holds that $\Delta _ { a } \leq \epsilon / 4$

Next, we show that eliminated arms never become empirical best on $G .$ . Let a be a suboptimal arm which was eliminated at some epoch n. Then there exists an arm b such that $\hat { \mu } _ { b , n } - \hat { \mu } _ { a , n } > 6 \beta _ { n }$

On $G ,$ we have that $\hat { \mu } _ { b , n } \leq \mu _ { b } + \beta _ { n } \leq \mu ^ { * } + \beta _ { n }$ , while $\hat { \mu } _ { a , n } \geq \mu _ { a } - \beta _ { n } = \mu ^ { * } - \Delta _ { a } - \beta _ { n }$ . Combining, we get

$$
\mu ^ { * } + \beta _ { n } - \left( \mu ^ { * } - \Delta _ { a } - \beta _ { n } \right) > 6 \beta _ { n } .
$$

This implies that $\Delta _ { a } / 4 > \beta _ { n }$ . Note that $a ^ { * }$ is still active at epoch $n ,$ thus has at least n samples. Furthermore, arm a also has at least n samples. At every future time $t ,$ their sample count can only increase, and so by Lemma 1 we have that

$$
\beta _ { n _ { t } ( a ) } \leq \beta _ { n } , \qquad \beta _ { n _ { t } ( a ^ { * } ) } \leq \beta _ { n } .
$$

Thus, on $G \mathrm { : }$

$$
\hat { \mu } _ { t } ( a ^ { * } ) - \hat { \mu } _ { t } ( a ) \ge \big ( \mu ^ { * } - \beta _ { n _ { t } ( a ^ { * } ) } \big ) - \big ( \mu _ { a } + \beta _ { n _ { t } ( a ) } \big ) \ge ( \mu ^ { * } - \beta _ { n } ) - ( \mu _ { a } + \beta _ { n } ) = \Delta _ { a } - 2 \beta _ { n } > \Delta _ { a } / 2 > 0 ,
$$

Therefore, eliminated arms will not become empirically best after the SE phase on $G$

We have shown that arms that survive the entire SE phase have $\Delta _ { a } \leq \epsilon / 4 ,$ , and that arms that are eliminated do not become empirically best again, which implies that under G, in round s of the ϵ-greedy phase, the chosen empirical-best arm $g _ { s }$ satisfies $\Delta _ { g _ { s } } \leq \epsilon / 4$ . Taking the complement gives us

$$
\left\{ \Delta _ { g _ { s } } > \frac { 3 \epsilon } { 4 } \right\} \subseteq \left\{ \Delta _ { g _ { s } } > \frac { \epsilon } { 4 } \right\} \subseteq \overline { { G } } ,
$$

yielding the desired result.

Lemma 3 gives a uniform bound on the probability of choosing a bad arm. The next lemma gives us a decaying bound that allows us to sum the tail of the expected sample complexity:

Lemma 4. For all s, and for any arm a with $\textstyle \Delta _ { a } > { \frac { 3 \epsilon } { 4 } }$ , it holds that

$$
\begin{array} { r } { \operatorname* { P r } ( g _ { s } = a ) \le 6 e ^ { - \frac { \epsilon ( s - 1 ) \Delta _ { a } ^ { 2 } } { 3 2 K } } . } \end{array}
$$

Proof. Fix a suboptimal arm a satisfying $\Delta _ { a } > \frac { 3 \epsilon } { 4 }$ , and let $a ^ { * } \in { \mathcal { A } } ^ { * }$ . Denote by $M _ { s } ( b )$ the number of exploration rounds in which arm b was chosen during the first $s - 1$ ϵ-greedy phase rounds, and note that by the definition of the policy

$$
M _ { s } ( a ) , M _ { s } ( a ^ { * } ) \sim B i n \Big ( s - 1 , \frac { \epsilon } { 4 K } \Big ) .
$$

Let $m _ { s } \triangleq { \frac { ( s - 1 ) \epsilon } { 4 K } }$ . Lastly, let $N _ { s } ( b )$ denote the total number of draws of arm b before round s of the ϵ-greedy phase, including all pulls from the SE phase. As such, the empirical mean used in the algorithm for arm b is given by

$$
\hat { \mu } _ { s } ^ { \pi } ( b ) = \frac { 1 } { N _ { s } ( b ) } \sum _ { j = 1 } ^ { N _ { s } ( b ) } \tilde { X } _ { b , j } .
$$

We denote:

$$
B _ { b , s } \triangleq \bigg \{ | \hat { \mu } _ { s } ^ { \pi } ( b ) - \mu _ { b } | \geq \frac { \Delta _ { a } } { 2 } \bigg \} , \qquad D _ { b , s } = \Big \{ M _ { s } ( b ) \geq \frac { m _ { s } } { 2 } \Big \} .
$$

Note that on the event $B _ { a , s } ^ { C } \cap B _ { a ^ { * } , s } ^ { C } .$ , we have that

$$
\hat { \mu } _ { s } ^ { \pi } ( a ) < \mu _ { a } + \frac { \Delta _ { a } } { 2 } = \mu ^ { * } - \frac { \Delta _ { a } } { 2 } < \hat { \mu } _ { s } ^ { \pi } ( a ^ { * } ) ,
$$

implying that arm a is not empirically best at round $s ,$ and specifically $g _ { s } \neq a .$ . Taking the complement and including the events $D _ { a , s } , D _ { a ^ { * } , s } ,$ we get:

$$
\{ g _ { s } = a \} \subseteq B _ { a , s } \cup B _ { a ^ { * } , s } \subseteq \left( D _ { a , s } ^ { C } \cup D _ { a ^ { * } , s } ^ { C } \right) \cup ( B _ { a , s } \cap D _ { a , s } ) \cup ( B _ { a ^ { * } , s } \cap D _ { a ^ { * } , s } ) ,
$$

and thus by the union bound:

$$
\operatorname* { P r } ( g _ { s } = a ) \leq \underbrace { \operatorname* { P r } \bigl ( D _ { a , s } ^ { C } \cup D _ { a ^ { * } , s } ^ { C } \bigr ) } _ { ( * ) } + \underbrace { \operatorname* { P r } ( B _ { a , s } \cap D _ { a , s } ) + \operatorname* { P r } ( B _ { a ^ { * } , s } \cap D _ { a ^ { * } , s } ) } _ { ( * * ) } .
$$

To bound (∗), we apply the union bound and the multiplicative Chernof bound (Fact 2) and get that

$$
\operatorname* { P r } \bigl ( D _ { a , s } ^ { C } \cup D _ { a ^ { * } , s } ^ { C } \bigr ) \leq \operatorname* { P r } \Bigl ( M _ { s } ( a ) < \frac { m _ { s } } { 2 } \Bigr ) + \operatorname* { P r } \Bigl ( M _ { s } ( a ^ { * } ) < \frac { m _ { s } } { 2 } \Bigr ) \leq 2 e ^ { - m _ { s } / 8 } .
$$

To bound (∗∗), we note that $N _ { s } ( b ) \geq M _ { s } ( b )$ and apply Fact 7 on arms $a , a ^ { * }$ with the filtration G to get:

$$
\operatorname* { P r } ( B _ { a , s } \cap D _ { a , s } ) + \operatorname* { P r } ( B _ { a ^ { * } , s } \cap D _ { a ^ { * } , s } ) \leq 2 e ^ { - m _ { s } \Delta _ { a } ^ { 2 } / 4 } + 2 e ^ { - m _ { s } \Delta _ { a } ^ { 2 } / 4 } = 4 e ^ { - m _ { s } \Delta _ { a } ^ { 2 } / 4 } .
$$

Combining these results, we get

$$
\mathrm { P r } ( g _ { s } = a ) \leq 2 e ^ { - m _ { s } / 8 } + 4 e ^ { - m _ { s } \Delta _ { a } ^ { 2 } / 4 } \leq 6 e ^ { - m _ { s } \Delta _ { a } ^ { 2 } / 8 } ,
$$

where the last inequality is since for $\Delta _ { a } \leq 1$ , we have that $m _ { s } \Delta _ { a } ^ { 2 } / 8 \leq m _ { s } / 8$ and $m _ { s } \Delta _ { a } ^ { 2 } / 8 \leq m _ { s } \Delta _ { a } ^ { 2 } / 4$ . Recalling that $\begin{array} { r } { m _ { s } = \frac { \epsilon ( s - 1 ) } { 4 K } } \end{array}$ , we get

$$
\begin{array} { r } { \operatorname* { P r } ( g _ { s } = a ) \le 6 e ^ { - \frac { \epsilon ( s - 1 ) \Delta _ { a } ^ { 2 } } { 3 2 K } } , } \end{array}
$$

as desired.

Combining Lemmas 3 and 4, we get

$$
\begin{array} { r } { \operatorname* { P r } ( g _ { s } = a ) \le \operatorname* { m i n } \biggl \{ q , 6 e ^ { - \frac { \epsilon ( s - 1 ) \Delta _ { a } ^ { 2 } } { 3 2 K } } \biggr \} . } \end{array}
$$

Applying the union bound and taking the above bound, we get

$$
\mathbb { E } \left[ \sum _ { t = T _ { S E } + 1 } ^ { \infty } 1 \left\{ \Delta _ { \pi _ { t } } > \epsilon \right\} \right] \leq \mathbb { E } \left[ \sum _ { s = 1 } ^ { \infty } 1 \left\{ \Delta _ { s _ { s } } > \frac { 3 \epsilon } { 4 } \right\} \right] \leq \sum _ { s = 1 } ^ { \infty } \sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \mathbb { P r } ( g _ { s } = a ) \leq \sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \sum _ { s = 1 } ^ { \infty } 6 \operatorname* { m i n } \left\{ q , e ^ { - \frac { \epsilon ( s - 1 ) \Delta _ { a } ^ { 2 } } { 3 2 K } } \right\} .
$$

We bound the inner sum using the following lemma.

Lemma 5. Let $0 < q < 1$ and $\lambda \in ( 0 , 1 ]$ . Then

$$
\sum _ { j = 0 } ^ { \infty } \operatorname* { m i n } \bigl \{ q , e ^ { - \lambda j } \bigr \} \leq q + \frac { q } { \lambda } \biggl ( \ln \biggl ( \frac { 1 } { q } \biggr ) + 2 \biggr ) .
$$

Proof of this lemma can be found in Appendix C.2. We use lemma 5 with $\begin{array} { r } { q = \frac { \epsilon ^ { 2 } } { 1 6 K } } \end{array}$ and $\begin{array} { r } { \lambda = \frac { \epsilon \Delta _ { a } ^ { 2 } } { 3 2 K } } \end{array}$ to get

$$
\sum _ { s = 1 } ^ { \infty } \operatorname* { m i n } \biggr \{ q , e ^ { - \frac { \epsilon ( s - 1 ) \Delta _ { a } ^ { 2 } } { 3 2 K } } \biggr \} \leq \frac { \epsilon ^ { 2 } } { 1 6 K } + \frac { 2 \epsilon } { \Delta _ { a } ^ { 2 } } \biggl ( \ln \biggl ( \frac { 1 6 K } { \epsilon ^ { 2 } } \biggr ) + 2 \biggr ) ,
$$

which in turn yields

$$
\sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \sum _ { s = 1 } ^ { \infty } 6 \operatorname* { m i n } \biggl \{ q , e ^ { - \frac { \epsilon ( s - 1 ) \Delta _ { a } ^ { 2 } } { 3 2 K } } \biggr \} \leq \sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 6 \epsilon ^ { 2 } } { 1 6 K } + \frac { 1 2 \epsilon } { \Delta _ { a } ^ { 2 } } \biggl ( \ln \biggl ( \frac { 1 6 K } { \epsilon ^ { 2 } } \biggr ) + 2 \biggr ) .
$$

Since $\epsilon < 1$ , we have that

$$
\sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 6 \epsilon ^ { 2 } } { 1 6 K } + \frac { 1 2 \epsilon } { \Delta _ { a } ^ { 2 } } \biggl ( \ln \biggl ( \frac { 1 6 K } { \epsilon ^ { 2 } } \biggr ) + 2 \biggr ) = \mathcal { O } \left( \sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 1 } { \Delta _ { a } ^ { 2 } } \ln \biggl ( \frac { K } { \epsilon ^ { 2 } } \biggr ) \right) .
$$

Combining this with Equation (4), we get that

$$
\mathbb { E } [ N _ { \epsilon } ( \pi ) ] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T _ { S E } } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] + \mathbb { E } \left[ \sum _ { t = T _ { S E } + 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] = \mathcal { O } \left( \sum _ { a : \Delta _ { a } > \frac { 3 \epsilon } { 4 } } \frac { 1 } { \Delta _ { a } ^ { 2 } } \mathrm { l n } \Big ( \frac { K } { \epsilon ^ { 2 } } \Big ) \right) ,
$$

proving the desired bound.

## C.2 Technical Lemmas for the Proof of Theorem 1

Lemma 1. $\beta _ { n }$ is decreasing in n.

Proof. Recall that $\begin{array} { r } { \beta _ { n } = \sqrt { \frac { 1 } { 2 n } \ln \left( \frac { 6 4 K ^ { 2 } n ^ { 2 } } { \epsilon ^ { 2 } } \right) } } \end{array}$ . Since $x \mapsto { \sqrt { x } }$ is monotone increasing for $x > 0 .$ , it is enough to show that $\begin{array} { r } { f ( x ) \triangleq \frac { 1 } { 2 x } \ln \left( \frac { 6 4 K ^ { 2 } x ^ { 2 } } { \epsilon ^ { 2 } } \right) } \end{array}$ is monotone decreasing for $x \ge 1$ . It holds that

$$
f ^ { \prime } ( x ) = - { \frac { 1 } { 2 x ^ { 2 } } } \ln \biggl ( { \frac { 6 4 K ^ { 2 } x ^ { 2 } } { \epsilon ^ { 2 } } } \biggr ) + { \frac { 1 } { x ^ { 2 } } } = { \frac { 2 - \ln \biggl ( { \frac { 6 4 K ^ { 2 } x ^ { 2 } } { \epsilon ^ { 2 } } } \biggr ) } { 2 x ^ { 2 } } } .
$$

Noting that for $K \geq 1 , \epsilon \in ( 0 , 1 )$ and $x \ge 1$ we have that $\frac { 6 4 K ^ { 2 } x ^ { 2 } } { \epsilon ^ { 2 } } \geq 6 4$ , we get

$$
\ln \left( \frac { 6 4 K ^ { 2 } x ^ { 2 } } { \epsilon ^ { 2 } } \right) \ge \ln ( 6 4 ) > 2 ,
$$

and so $f ^ { \prime } ( x ) < 0$ for all $x \geq 1$ . This implies that $f ( x )$ is strictly monotone decreasing for $x \geq 1$ , which in turn gives us that $\beta _ { n }$ is monotone decreasing in n. □

Lemma 2. For every $x \in ( 0 , 1 ]$

$$
n ( x ) \leq \bigg \lceil \frac { 2 0 4 8 } { x ^ { 2 } } \ln \bigg ( \frac { 1 0 2 4 K } { \epsilon x } \bigg ) \bigg \rceil , \qquad a n d \qquad n _ { \epsilon } \leq \bigg \lceil \frac { 3 2 7 6 8 } { \epsilon ^ { 2 } } \ln \bigg ( \frac { 4 0 9 6 K } { \epsilon ^ { 2 } } \bigg ) \bigg \rceil .
$$

Proof. Let $x \in ( 0 , 1 ]$ , and define $L \triangleq \ln \left( { \frac { 1 0 2 4 K } { \epsilon x } } \right)$ and $n _ { 0 } \ { \stackrel { \Delta } { = } } \ { \frac { 2 0 4 8 L } { x ^ { 2 } } }$ . Since $K \ge 1 , \epsilon \le 1$ and $x \le 1$ , we have $L \ge \ln 1 0 2 4$ . Furthermore, since $\begin{array} { r } { \frac { 4 K } { q } = \frac { 6 4 K ^ { 2 } } { \epsilon ^ { 2 } } } \end{array}$ , we have that

$$
\ln \left( \frac { 4 K n _ { 0 } ^ { 2 } } { q } \right) = \ln \left( \frac { 6 4 K ^ { 2 } n _ { 0 } ^ { 2 } } { \epsilon ^ { 2 } } \right) = \ln 6 4 + 2 \ln \left( \frac { K } { \epsilon } \right) + 2 \ln 2 0 4 8 + 2 \ln L + 4 \ln \left( \frac { 1 } { x } \right) .
$$

By the definition of $L$

$$
\ln \left( { \frac { K } { \epsilon } } \right) \leq L , \qquad \ln \left( { \frac { 1 } { x } } \right) \leq L , \qquad \ln L \leq L .
$$

Furthermore, since $L \geq$ ln 1024, it holds that ln $6 4 + 2 \ln 2 0 4 8 \leq 3 L$ , and so we get that ln $\left( \frac { 4 K n _ { 0 } ^ { 2 } } { q } \right) \le 1 1 L$ . By the definition of $\beta _ { n }$ and Lemma 1, it follows that

$$
\beta _ { \lceil n _ { 0 } \rceil } ^ { 2 } \leq \frac { 1 } { 2 n _ { 0 } } \ln \biggl ( \frac { 4 K n _ { 0 } ^ { 2 } } { q } \biggr ) \leq \frac { x ^ { 2 } } { 4 0 9 6 L } \cdot 1 1 L = \frac { 1 1 x ^ { 2 } } { 4 0 9 6 } < \frac { x ^ { 2 } } { 6 4 } ,
$$

thus $\beta _ { \lceil n _ { 0 } \rceil } < \frac { x } { 8 }$ . It follows that $n ( x ) \leq \lceil n _ { 0 } \rceil$ , which proves the first result. Applying the first bound with $x = \epsilon / 4$ gives

$$
n _ { \epsilon } \le \bigg \lceil \frac { 3 2 7 6 8 } { \epsilon ^ { 2 } } \ln \bigg ( \frac { 4 0 9 6 K } { \epsilon ^ { 2 } } \bigg ) \bigg \rceil ,
$$

as desired.

Lemma 5. Let $0 < q < 1$ and $\lambda \in ( 0 , 1 ]$ . Then

$$
\sum _ { j = 0 } ^ { \infty } \operatorname* { m i n } \bigl \{ q , e ^ { - \lambda j } \bigr \} \leq q + \frac { q } { \lambda } \biggl ( \ln \biggl ( \frac { 1 } { q } \biggr ) + 2 \biggr ) .
$$

Proof. Let $\begin{array} { r } { J = \left\lceil \frac { 1 } { \lambda } \ln \left( \frac { 1 } { q } \right) \right\rceil } \end{array}$ . Then

$$
\sum _ { j = 0 } ^ { J - 1 } \operatorname* { m i n } \bigl \{ q , e ^ { - \lambda j } \bigr \} \leq q J \leq q + \frac { q } { \lambda } \ln \biggl ( \frac { 1 } { q } \biggr ) .
$$

Moreover,

$$
\sum _ { j = J } ^ { \infty } \operatorname* { m i n } \bigl \{ q , e ^ { - \lambda j } \bigr \} \leq \sum _ { j = J } ^ { \infty } e ^ { - \lambda j } = \frac { e ^ { - \lambda J } } { 1 - e ^ { - \lambda } } \leq \frac { q } { 1 - e ^ { - \lambda } } .
$$

Since $\lambda \in ( 0 , 1 ]$ , we have that $\begin{array} { r } { 1 - e ^ { - \lambda } \ge \frac { \lambda } { 2 } } \end{array}$ , implying that

$$
\sum _ { j = J } ^ { \infty } e ^ { - \lambda j } \leq { \frac { 2 q } { \lambda } } .
$$

Combining the two inequalities $_ \mathrm { y }$ ields

$$
\sum _ { j = 0 } ^ { \infty } \operatorname* { m i n } \Bigl \{ q , e ^ { - \lambda j } \Bigr \} \le q + \frac { q } { \lambda } \ln \Biggl ( \frac { 1 } { q } \Biggr ) + \frac { 2 q } { \lambda } = q + \frac { q } { \lambda } \Biggl ( \ln \Biggl ( \frac { 1 } { q } \Biggr ) + 2 \Biggr ) ,
$$

giving us the desired bound.

## C.3 Proof of Proposition 2

Proposition 2. Let $\delta , \epsilon > 0$ such that max $\{ \delta , 3 \epsilon \} \ \le \ 1 / 8$ , and let $K \ \geq \ 2$ . Define the bandit class ${ \mathcal { E } } _ { \delta } ~ =$ $\left\{ \nu \in \mathcal { E } _ { B e r } ^ { K } : \Delta _ { \operatorname* { m i n } } ( \nu ) \geq \delta \right\}$ . Then for every policy π, there exists $\boldsymbol \nu \in \mathcal { E } _ { \delta }$ such that:

$$
\mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \Omega \left( \frac { K } { \operatorname* { m a x } \{ \delta , \epsilon \} ^ { 2 } } \right) .
$$

Proof. Fix some policy $\pi .$ Let $T ~ \in ~ \mathbb { N }$ be some constant to be chosen later, and denote $N _ { \epsilon } ( \pi , \nu , T ) =$ $\textstyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \}$ . We construct 2 instances, $\nu , \nu ^ { \prime } \in \mathcal { E } _ { \delta }$ , and show that π must incur high expected sample complexity on at least one of them.

Let $\Delta = \operatorname* { m a x } \{ \delta , 3 \epsilon \}$ , and note that $\Delta \le 1 / 8$ . We define an instance from ${ \mathcal { E } } _ { \delta }$ by a vector, where the $i ^ { t h }$ entry represents the mean of the $i ^ { t h }$ arm. The first instance ν is defined by the mean vector:

$$
\begin{array} { r } { \mu _ { \nu } ( a ) = \left\{ \frac { 1 } { 2 } + \Delta \quad a = 1 \atop \frac { 1 } { 2 } \right.}   \end{array}
$$

Now, let $i = \arg \operatorname* { m i n } _ { \substack { j > 1 } } \mathbb { E } _ { \nu } [ n _ { T + 1 } ( j ) ]$ be the suboptimal arm in ν that the policy π played the least in expectation when running on horizon $T$ . We define the instance $\nu ^ { \prime }$ by the following mean vector:

$$
\mu _ { \nu ^ { \prime } } ( a ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } + \Delta } & { a = 1 } \\ { \frac { 1 } { 2 } + 2 \Delta } & { a = i } \\ { \frac { 1 } { 2 } } & { a \not \in \{ 1 , i \} . } \end{array} \right.
$$

Consider the random variable $\begin{array} { r } { S _ { T } = \sum _ { t = 1 } ^ { T } \mathbb { 1 } \big \{ \pi _ { t } ( 1 ) \geq \frac { \Delta - \epsilon } { \Delta } \big \} } \end{array}$ . Note that by our choice of $\Delta ,$ on instance ν we have that $\Delta _ { \pi _ { t } } = ( 1 - \pi _ { t } ( 1 ) ) \Delta$ , and so

$$
N _ { \epsilon } ( \pi , \nu , T ) = \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ ( 1 - \pi _ { t } ( 1 ) ) \Delta > \epsilon \} = \sum _ { t = 1 } ^ { T } \mathbb { 1 } \bigg \{ \pi _ { t } ( 1 ) < \frac { \Delta - \epsilon } { \Delta } \bigg \} = T - S _ { T } ,
$$

implying that

$$
N _ { \epsilon } ( \pi , \nu , T ) \geq \mathbb { 1 } \left\{ S _ { T } \leq \frac { T } { 2 } \right\} \cdot \frac { T } { 2 } .
$$

Taking the expectation on both sides yields:

$$
\mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu , T ) ] \geq\mathrm { P r } _ { \nu } \bigg ( S _ { T } \leq \frac { T } { 2 } \bigg ) \cdot \frac { T } { 2 } .\tag{5}
$$

Moreover, by the definition of $\nu ^ { \prime } ,$ we have that

$$
N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) \geq \sum _ { t = 1 } ^ { T } \Im \{ \pi _ { t } ( 1 ) \Delta > \epsilon \} \geq \sum _ { t = 1 } ^ { T } \Im \Big \{ \pi _ { t } ( 1 ) > \frac { \epsilon } { \Delta } \Big \} \geq \sum _ { t = 1 } ^ { T } \Im \bigg \{ \pi _ { t } ( 1 ) \geq \frac { \Delta - \epsilon } { \Delta } \bigg \} ,
$$

where the last inequality is since $\begin{array} { r } { \frac { \Delta - \epsilon } { \Delta } > \frac { \epsilon } { \Delta } } \end{array}$ for our choice of $\Delta .$ The above implies that $N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) \ge S _ { T }$ , and so we get

$$
N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) \geq \mathbb { 1 } \left\{ S _ { T } > \frac { T } { 2 } \right\} \cdot \frac { T } { 2 } ,
$$

and taking the expectation on both sides yields:

$$
\mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) ] \geq \operatorname* { P r } _ { \nu ^ { \prime } } \left( S _ { T } > \frac { T } { 2 } \right) \cdot \frac { T } { 2 } .\tag{6}
$$

Combining Equations (5) and (6), we have that:

$$
\begin{array} { r l r } {  { 2 \operatorname* { m a x } \{ \mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu , T ) ] , \mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) ] \} \geq \mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu , T ) ] + \mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) ] } } \\ & { } & { \geq \operatorname* { P r } _ { \nu } \bigg ( S _ { T } \leq \frac { T } { 2 } \bigg ) \cdot \frac { T } { 2 } + \operatorname* { P r } _ { \nu ^ { \prime } } \bigg ( S _ { T } > \frac { T } { 2 } \bigg ) \cdot \frac { T } { 2 } } \\ & { } & { = \bigg ( \operatorname* { P r } _ { \nu } \bigg ( S _ { T } \leq \frac { T } { 2 } \bigg ) + \operatorname* { P r } _ { \nu ^ { \prime } } \bigg ( S _ { T } > \frac { T } { 2 } \bigg ) \bigg ) \cdot \frac { T } { 2 } . } \end{array}
$$

Applying the Bretagnolle-Huber inequality (Theorem 14.2 in Lattimore and Szepesv´ari (2020)) to the above yields:

$$
\left( \operatorname* { P r } _ { \nu } \left( S _ { T } \leq { \frac { T } { 2 } } \right) + \operatorname* { P r } _ { \nu ^ { \prime } } \left( S _ { T } > { \frac { T } { 2 } } \right) \right) \cdot { \frac { T } { 2 } } \geq \left( { \frac { 1 } { 2 } } \exp ( - D ( \operatorname { P r } _ { \nu } , \operatorname { P r } _ { \nu ^ { \prime } } ) ) \right) \cdot { \frac { T } { 2 } } .\tag{7}
$$

We use the divergence decomposition for bandits (Lemma 15.1 in Lattimore and Szepesv´ari (2020)) to get:

$$
D ( \operatorname* { P r } , \operatorname* { P r } _ { \nu ^ { \prime } } ) = \sum _ { a \in [ K ] } \mathbb { E } _ { \nu } [ n _ { T + 1 } ( a ) ] D ( \nu _ { a } , \nu _ { a } ^ { \prime } ) = \mathbb { E } _ { \nu } [ n _ { T + 1 } ( i ) ] D ( B e r ( 1 / 2 ) , B e r ( 1 / 2 + 2 \Delta ) ) .
$$

Since $\begin{array} { r } { \sum _ { a \in [ K ] \backslash \{ 1 \} } n _ { T + 1 } ( a ) \leq T } \end{array}$ , it must hold that $\begin{array} { r } { \mathbb { E } _ { \nu } [ n _ { T + 1 } ( i ) ] \le \frac { T } { K - 1 } } \end{array}$ . We combine this, and the bound on the divergence of Bernoulli RVs (Fact 5) to get:

$$
\mathbb { E } _ { \nu } [ n _ { T + 1 } ( i ) ] D ( B e r ( 1 / 2 ) , B e r ( 1 / 2 + 2 \Delta ) ) \le \frac { 3 2 T \Delta ^ { 2 } } { 3 ( K - 1 ) } \le \frac { 1 1 T \Delta ^ { 2 } } { K - 1 } .
$$

We substitute this back to get:

$$
\Bigg ( \frac { 1 } { 2 } \exp \Big ( { - D \Big ( \operatorname* { P r } _ { \nu } , \operatorname* { P r } _ { \nu ^ { \prime } } \Big ) } \Big ) \Bigg ) \cdot \frac { T } { 2 } \geq \frac { T } { 4 } \exp \Bigg ( { - \frac { 1 1 T \Delta ^ { 2 } } { K - 1 } } \Bigg ) .
$$

Denote by $x \triangleq$ arg $\begin{array} { r } { \operatorname* { m a x } _ { T \in \mathbb { R } _ { + } } \Big \{ \frac { T } { 4 } \exp \Big ( { - \frac { 1 1 T \Delta ^ { 2 } } { K - 1 } } \Big ) \Big \} = \frac { K - 1 } { 1 1 \Delta ^ { 2 } } } \end{array}$ the continuous maximizer of the lower bound. Since $\Delta \le \frac { 1 } { 8 }$ , we have that $\begin{array} { r } { x \ge \frac { 6 4 ( \dot { K _ { - 1 } } ) } { 1 1 } \ge 1 } \end{array}$ . Taking $T ^ { * } = \lfloor x \rfloor \in \mathbb { N }$ gives

$$
f ( \lfloor x \rfloor ) = { \frac { \lfloor x \rfloor } { 4 } } \exp \left( - { \frac { 1 1 \lfloor x \rfloor \Delta ^ { 2 } } { K - 1 } } \right) \geq { \frac { \lfloor x \rfloor } { 4 } } \exp \left( - { \frac { \lfloor x \rfloor } { x } } \right) .
$$

Noting that $\textstyle { \frac { \lfloor x \rfloor } { x } } \leq 1$ , and that $\lfloor x \rfloor \geq { \frac { x } { 2 } }$ for $x \ge 1$ , we get

$$
\frac { \lfloor x \rfloor } { 4 } \exp { \left( - \frac { \lfloor x \rfloor } { x } \right) } \geq \frac { x } { 8 } \exp ( - 1 ) = \frac { K - 1 } { 8 8 e \Delta ^ { 2 } } .
$$

Combining this with Equation (7) and the definition of $\Delta$ yields

$$
\operatorname* { m a x } \{ \mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu , T ) ] , \mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi , \nu ^ { \prime } , T ) ] \} \geq \frac { K - 1 } { 1 7 6 e \Delta ^ { 2 } } = \frac { K - 1 } { 1 7 6 e \operatorname* { m a x } \{ \delta , 3 \epsilon \} ^ { 2 } } \geq \frac { K - 1 } { 1 5 8 4 e \operatorname* { m a x } \{ \delta , \epsilon \} ^ { 2 } } .
$$

Noting that $N _ { \epsilon } ( \pi , \nu , T ) \le N _ { \epsilon } ( \pi , \nu , T + 1 )$ for all $\pi , \nu$ and T implies that for all $T \in \mathbb { N }$

$$
\mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu , T ) ] \le \mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu ) ] .
$$

Finally, taking $T = T ^ { * }$ gives

$$
\operatorname* { m a x } \{ \mathbb { E } _ { \nu } [ N _ { \epsilon } ( \pi , \nu ) ] , \mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi , \nu ^ { \prime } ) ] \} \ge \frac { K - 1 } { 1 5 8 4 e \operatorname* { m a x } \{ \delta , \epsilon \} ^ { 2 } } ,
$$

which yields desired result.

## D UNIFORM ALGORITHMS – OMITTED PROOFS

## D.1 Proof of Proposition 3

Proposition 3. Let π be U-ACE and consistent over $\mathcal { E } _ { B e r } ^ { K }$ . Then for any fixed instance $\nu \in \mathcal { E } _ { B e \tau } ^ { K }$ with means in $[ 1 / 4 , 3 / 4 ]$ , it holds that

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \operatorname* { s u p } _ { \epsilon \cdot \mathbb { E } } [ N _ { \epsilon } ( \pi , \nu ) ] \geq \sum _ { a : \Delta _ { a } > 0 } \frac { 3 } { 8 \Delta _ { a } } .
$$

Proof. Denote $\begin{array} { r } { C _ { \nu } = \sum _ { a : \Delta _ { a } > 0 } \frac { 3 } { 8 \Delta _ { a } } } \end{array}$ , and suppose towards contradiction that

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \operatorname* { s u p } _ { \epsilon } \epsilon \cdot \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] < C _ { \nu } .
$$

If $C _ { \nu } = 0$ , the bound holds trivially. If $C _ { \nu } > 0$ then the above implies that there exists $c < C _ { \nu }$ and $\epsilon _ { 0 } > 0$ such that for all $\epsilon \in ( 0 , \epsilon _ { 0 } )$ ):

$$
\epsilon \cdot \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] < c .
$$

Using the above and recalling the following decomposition from the proof of Claim 2:

$$
R _ { T } ( \pi ) \leq \int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon } ( \pi ) ] \} d \epsilon ,
$$

gives us that

$$
R _ { T } ( \pi ) \leq \int _ { \epsilon = 0 } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon } ( \pi ) ] \} d \epsilon \leq \int _ { \epsilon = 0 } ^ { \epsilon _ { 0 } } \operatorname* { m i n } \Bigl \{ T , \frac { c } { \epsilon } \Bigr \} d \epsilon + \int _ { \epsilon = \epsilon _ { 0 } } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon _ { 0 } } ( \pi ) ] \} d \epsilon .
$$

By the monotonicity of the integral:

$$
\int _ { \epsilon = \epsilon _ { 0 } } ^ { 1 } \operatorname* { m i n } \{ T , \mathbb { E } [ N _ { \epsilon _ { 0 } } ( \pi ) ] \} d \epsilon \leq \mathbb { E } [ N _ { \epsilon _ { 0 } } ( \pi ) ] ,
$$

and so this term is a constant with respect to T. Note that for all suficiently large $T ,$ namely $\begin{array} { r } { T > \frac { c } { \epsilon _ { 0 } } } \end{array}$ , we have that

$$
\int _ { \epsilon = 0 } ^ { \epsilon _ { 0 } } \operatorname* { m i n } \Bigl \{ T , \frac { c } { \epsilon } \Bigr \} d \epsilon \le \int _ { \epsilon = 0 } ^ { \epsilon / T } T d \epsilon + \int _ { \epsilon = \epsilon / T } ^ { \epsilon _ { 0 } } \frac { c } { d \epsilon } d \epsilon = c + c \ln \left( \frac { \epsilon _ { 0 } T } { c } \right) = c ( 1 + \ln ( \epsilon _ { 0 } ) + \ln ( 1 / c ) ) + c \ln ( T ) ,
$$

and so since $\epsilon _ { \mathrm { 0 } }$ and c are constants, we have that

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { l n p } _ { \ln ( T ) } \leq c .
$$

This contradicts the Lai-Robbins lower bound (Fact 6), thus implying our desired result.

## D.2 Proof of Proposition 4

Proposition 4. Let π be U-ACE over $\mathcal { E } _ { B e r } ^ { 2 }$ . Then there exists an instance $\nu \in \mathcal { E } _ { B e r } ^ { 2 }$ such that for any $\alpha \in ( 0 , 1 )$

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \operatorname* { s u p } _ { \epsilon } \epsilon ^ { \alpha } \cdot \mathbb { E } [ N _ { \epsilon } ( \pi , \nu ) ] = \infty .
$$

Proof. Let $\nu = \left( B e r \left( \textstyle { \frac { 1 } { 2 } } \right) , B e r \left( \textstyle { \frac { 1 } { 4 } } \right) \right)$ , and let $\nu ^ { \prime } = \left( B e r \bigl ( \textstyle { \frac { 1 } { 2 } } \bigr ) , B e r \bigl ( \textstyle { \frac { 3 } { 4 } } \bigr ) \right)$ . We prove the following:

Lemma 6. $I f \operatorname* { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 2 ) = \infty ) = 1$ , then $\operatorname* { P r } _ { \nu } ( n _ { \infty } ( 2 ) = \infty ) = 1$

Proof. Assume that $\mathrm { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 2 ) = \infty ) = 1$ . We utilize the following lemma:

Lemma 7 (Lemma 1 from Kaufmann et al. (2016)). Let $\nu , \nu ^ { \prime }$ be two bandit instances with K arms such that for all $^ { a , }$ the distributions of $\nu _ { a }$ and $\nu _ { a } ^ { \prime }$ are mutually absolutely continuous. Then for any almost-surely finite stopping time σ with respect to filtration $( \mathcal { F } _ { t } )$

$$
\sum _ { a \in [ K ] } \mathbb { E } _ { \nu } [ n _ { \sigma + 1 } ( a ) ] D ( \nu _ { a } , \nu _ { a } ^ { \prime } ) \geq \operatorname* { s u p } _ { A \in \mathcal { F } _ { \sigma } } d ( \operatorname* { P r } _ { \nu } ( A ) , \operatorname* { P r } _ { \nu ^ { \prime } } ( A ) ) ,
$$

where $d ( x , y ) \triangleq x \ln ( x / y ) + ( 1 - x ) \ln ( ( 1 - x ) / ( 1 - y ) )$

Note that $\nu _ { 1 } = \nu _ { 1 } ^ { \prime }$ , while $\nu _ { 2 } = B e r ( 1 / 4 )$ and $\nu _ { 2 } ^ { \prime } = B e r ( 3 / 4 )$ are mutually absolutely continuous, since both assign positive probability to 0 and 1. Moreover,

$$
D ( \nu _ { 2 } , \nu _ { 2 } ^ { \prime } ) = D ( B e r ( 1 / 4 ) , B e r ( 3 / 4 ) ) = \frac { \ln ( 3 ) } { 2 } .
$$

Fix $n \in \mathbb { N }$ , and define $\tau _ { n } \triangleq \operatorname* { i n f } \{ t : n _ { t } ( 2 ) = n \}$ . Let $T \in \mathbb { N } ,$ , and define $\sigma _ { n , T } \triangleq \tau _ { n } \wedge T$ to be an almost-surely finite stopping time. We look at the event $E _ { n , T } \triangleq \{ \tau _ { n } \leq T \} = \{ n _ { T } ( 2 ) \geq n \}$ . We can see that $E _ { n , T }$ is in ${ \mathcal { F } } _ { \sigma _ { n , T } } ,$ , which implies by Lemma 7 that

$$
\begin{array} { r } { \mathbb { E } _ { \nu } \left[ n _ { \sigma _ { n , T } + 1 } ( 2 ) \right] D ( \nu _ { 2 } , \nu _ { 2 } ^ { \prime } ) \geq d ( \operatorname* { P r } _ { \nu } ( E _ { n , T } ) , \operatorname* { P r } _ { \nu ^ { \prime } } ( E _ { n , T } ) ) , } \end{array}
$$

but by the definition of $\sigma _ { n , T }$ , we have that $n _ { \sigma _ { n , T } + 1 } ( 2 ) \leq n + 1$ , which implies that $d ( \mathrm { P r } _ { \nu } ( E _ { n , T } ) , \mathrm { P r } _ { \nu ^ { \prime } } ( E _ { n , T } ) ) \leq$ $\frac { ( n + 1 ) \ln ( 3 ) } { 2 } = C _ { n }$ . Since $\mathrm { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 2 ) = \infty ) = 1$ , we have that $\mathrm { P r } _ { \nu ^ { \prime } } ( E _ { n , T } )  1$ as $T \to \infty$

Now, denote $p _ { n , T } \triangleq \operatorname* { P r } _ { \nu } ( E _ { n , T } ) , q _ { n , T } \triangleq \operatorname* { P r } _ { \nu ^ { \prime } } ( E _ { n , T } )$ , and note that the sequence of events $E _ { n , T }$ is increasing in T, thus implying that $p _ { n , T } \uparrow p _ { n } = \mathrm { P r } _ { \nu } ( n _ { \infty } ( 2 ) \geq n )$ , and that $q _ { n , T } \uparrow \operatorname* { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 2 ) \geq n ) = 1$ by our assumption. Suppose towards contradiction that $p _ { n } < 1$ , and let $\begin{array} { r } { \eta = \frac { 1 - p _ { n } } { 2 } > 0 . } \end{array}$ Since $p _ { n , T } \uparrow p _ { n }$ , we have that $p _ { n , T } \leq 1 - \eta$ which gives us $1 - p _ { n , T } \ge \eta$ . Recall the bound $d ( p , q ) \geq ( 1 - p ) \ln ( 1 / ( 1 - q ) ) - \ln ( 2 )$ , which implies that

$$
d ( p _ { n , T } , q _ { n , T } ) \geq ( 1 - p _ { n , T } ) \ln ( 1 / ( 1 - q _ { n , T } ) ) - \ln ( 2 ) \geq \eta \ln ( 1 / ( 1 - q _ { n , T } ) ) - \ln ( 2 ) .
$$

But since $q _ { n , T } \to 1$ , we have that $\eta \ln ( 1 / ( 1 - q _ { n , T } ) ) - \ln ( 2 ) \to \infty$ , contradicting the fact that $d ( p _ { n , T } , q _ { n , T } )$ is bounded by $C _ { n }$ for all T. This in turn implies that $p _ { n , T } \uparrow 1$ , or equivalently:

$$
\forall n \in \mathbb { N } : \quad \operatorname* { P r } _ { \nu } ( n _ { \infty } ( 2 ) \geq n ) = 1 .
$$

Note that the sequence $A _ { n } = \{ n _ { \infty } ( 2 ) \geq n \}$ is decreasing in n, thus by continuity and the above

$$
\operatorname* { P r } _ { \nu } ( n _ { \infty } ( 2 ) = \infty ) = \operatorname* { P r } _ { \nu } \left( \bigcap _ { n = 1 } ^ { \infty } A _ { n } \right) = \operatorname* { l i m } _ { n \to \infty } \operatorname* { P r } _ { \nu } ( A _ { n } ) = 1 ,
$$

as desired.

We continue by showing that on at least one of the instances $\nu , \nu ^ { \prime } ,$ we have that $\mathbb { E } [ \sum _ { t = 1 } ^ { \infty } \Delta _ { \pi _ { t } } ] = \infty$ . Notice it sufices to show that either $\mathrm { P r } _ { \nu } ( n _ { \infty } ( 2 ) = \infty ) > 0 \ \mathrm { o r } \ \mathrm { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 1 ) = \infty ) > 0$ , since this would imply the above by applying Claim 1 for all finite T and MCT.

As such, if $\mathrm { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 1 ) = \infty ) > 0 .$ , we are done, and if $\operatorname* { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 1 ) = \infty ) = 0$ , then $\operatorname* { P r } _ { \nu ^ { \prime } } ( n _ { \infty } ( 2 ) = \infty ) = 1$ , which by Lemma 6 implies that $\mathrm { P r } _ { \nu } ( n _ { \infty } ( 2 ) = \infty ) = 1 > 0$

This shows that there exists an instance such that $\mathbb { E } [ \sum _ { t = 1 } ^ { \infty } \Delta _ { \pi _ { t } } ] = \infty$ , which implies that on that instance

$$
\int _ { 0 } ^ { \Delta _ { \operatorname* { m a x } } } \mathbb { E } [ N _ { \epsilon } ( \pi ) ] d \epsilon = \int _ { 0 } ^ { \Delta _ { \operatorname* { m a x } } } \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] d \epsilon = \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \int _ { 0 } ^ { \Delta _ { \operatorname* { m a x } } } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} d \epsilon \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \Delta _ { \pi _ { t } } \right] = \infty .
$$

Fix some $\epsilon _ { 0 } \in ( 0 , \Delta _ { \operatorname* { m a x } } )$ . π is U-ACE, and so $\mathbb { E } [ N _ { \epsilon _ { 0 } } ( \pi ) ] < \infty$ . Moreover, since $\mathbb { E } [ N _ { \epsilon } ( \pi ) ]$ is non-increasing in $\epsilon ,$ we have that

$$
\int _ { \epsilon _ { 0 } } ^ { \Delta _ { \operatorname* { m a x } } } \mathbb { E } [ N _ { \epsilon } ( \pi ) ] d \epsilon \leq ( \Delta _ { \operatorname* { m a x } } - \epsilon _ { 0 } ) \mathbb { E } [ N _ { \epsilon _ { 0 } } ( \pi ) ] < \infty
$$

Thus implying that $\begin{array} { r } { \int _ { 0 } ^ { \epsilon _ { 0 } } \mathbb { E } [ N _ { \epsilon } ( \pi ) ] d \epsilon = \infty } \end{array}$ . Let $\alpha \in ( 0 , 1 )$ , and suppose towards contradiction there exists $C > 0$ such that for all $\epsilon \in ( 0 , \epsilon _ { 0 } )$

$$
\mathbb { E } [ N _ { \epsilon } ( \pi ) ] \leq \frac { C } { \epsilon ^ { \alpha } } .
$$

By the monotonicity of the integral and the fact that $\int _ { 0 } ^ { \epsilon _ { 0 } } \frac { C } { \epsilon ^ { \alpha } } d \epsilon < \infty$ , we have that

$$
\int _ { 0 } ^ { \epsilon _ { 0 } } \mathbb { E } [ N _ { \epsilon } ( \pi ) ] d \epsilon \leq \int _ { 0 } ^ { \epsilon _ { 0 } } \frac { C } { \epsilon ^ { \alpha } } d \epsilon < \infty
$$

In contradiction to $\begin{array} { r } { \int _ { 0 } ^ { \epsilon _ { 0 } } \mathbb { E } [ N _ { \epsilon } ( \pi ) ] d \epsilon = \infty } \end{array}$ . Therefore, for all $C > 0$ there exists $\epsilon \in ( 0 , \epsilon _ { 0 } )$ such that

$$
\mathbb { E } [ N _ { \epsilon } ( \pi ) ] \geq \frac { C } { \epsilon ^ { \alpha } } .
$$

The claim follows by noting that we can take $\epsilon _ { 0 } \downarrow 0$ and $C  \infty$

## D.3 Proof of Theorem 2

Theorem 2. $L e t \ \pi ^ { T S ( \mathcal { N } ) }$ be the policy induced by Algorithm 2. Then for all $\epsilon > 0$ and instances ν:

$$
\mathbb { E } \bigg [ N _ { \epsilon } \Big ( \pi ^ { T S ( \mathcal { N } ) } , \nu \Big ) \bigg ] = \mathcal { O } \left( \frac { K } { \epsilon } \sum _ { \Delta _ { a } \ge \frac { \epsilon } { K } } \frac { 1 } { \Delta _ { a } } \mathrm { l n } \bigg ( \frac { 3 K \Delta _ { a } } { \epsilon } \bigg ) \right) .
$$

Proof. Our proof is inspired by the methods used in Agrawal and Goyal (2017). We adopt their choices of events and decomposition, but we difer in two main ways:

1. Analyzing the sample complexity requires diferent techniques than bounding the regret, since our objective measures the conditional distribution of the policy itself, thus requiring a more subtle approach. Moreover, our analysis is over an infinite horizon, requiring delicate tracking of some of the events, which is unnecessary when interacting for a finite horizon.

2. In contrast to previous works, we cannot apply a monotonicity argument to reduce to a unique optimal arm. The intuition is that there is no guarantee that playing an optimal arm will conditionally decrease the policy gap at the next time-step, with an empirical counterexample given in Appendix D.4. This adds an additional layer to our analysis, requiring us to design events that are agnostic to the number of optima arms, and also to the optimal representative chosen in our proofs.

Since $\pi ^ { T S ( \mathcal { N } ) }$ is the only policy we deal with in this proof, we refer to it as π for brevity. We denote

$$
\begin{array} { r } { A ^ { * } \triangleq \{ a \in [ K ] : \Delta _ { a } = 0 \} , \qquad \mathcal { A } _ { s u b } \triangleq \{ a \in [ K ] : \Delta _ { a } > 0 \} , \qquad M \triangleq | A _ { s u b } | . } \end{array}
$$

Note that if $M = 0 ,$ , then all arms are optimal and $N _ { \epsilon } ( \pi ) = 0$ almost surely, giving us the desired result. We henceforth assume $M \geq 1$ , and address the case of multiple optimal arms when needed. For every suboptimal arm $a \in \mathcal { A } _ { s u b }$ , we define

$$
x _ { a } \triangleq \mu _ { a } + \frac { \Delta _ { a } } { 3 } , \qquad y _ { a } \triangleq \mu ^ { * } - \frac { \Delta _ { a } } { 3 } ,
$$

and the events

$$
E _ { a } ^ { \mu } ( t ) \triangleq \{ \hat { \mu } _ { t } ( a ) \leq x _ { a } \} , \qquad E _ { a } ^ { \theta } ( t ) \triangleq \{ \theta _ { t } ( a ) \leq y _ { a } \} .
$$

We prove the following:

Lemma 8. For all $t ,$ we have that

$$
\operatorname* { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) \leq \sum _ { a : \Delta _ { a } \geq \frac { \epsilon } { M } } \operatorname* { P r } \biggl ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \biggr ) .
$$

Proof. Assume that $\Delta _ { \pi _ { t } } > \epsilon .$ and assume for the sake of contradiction that $\begin{array} { r } { \pi _ { t } ( a ) \leq \frac { \epsilon } { M \cdot \Delta _ { a } } } \end{array}$ for all $a \in \mathcal { A } _ { s u b }$ . We get that

$$
\Delta _ { \pi _ { t } } = \sum _ { a \in [ K ] } \pi _ { t } ( a ) \Delta _ { a } \leq \sum _ { a \in A _ { s u b } } \frac { \epsilon } { M \Delta _ { a } } \Delta _ { a } = \frac { M } { M } \cdot \epsilon = \epsilon ,
$$

which is a contradiction, implying that $\begin{array} { r } { \{ \Delta _ { \pi _ { t } } > \epsilon \} \subseteq \bigcup _ { a \in \mathcal { A } _ { s u b } } \Bigl \{ \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \Bigr \} } \end{array}$ . This gives us

$$
\operatorname* { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) \leq \operatorname* { P r } \biggl ( \bigcup _ { a \in A _ { s u b } } \biggl \{ \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \biggr \} \biggr ) \leq \sum _ { a : \Delta _ { a } \geq \frac { \epsilon } { M } } \operatorname* { P r } \biggl ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \biggr ) ,
$$

where the last inequality is by the union bound and the fact that for suboptimal arms with $\Delta _ { a } < \epsilon / M$ , we get that the event $\begin{array} { r } { \left\{ \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \right\} } \end{array}$ is empty. □

We therefore have that

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] \stackrel { ( 1 ) } { = } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) \stackrel { ( 2 ) } { \leq } \sum _ { t = 1 } ^ { \infty } \sum _ { \Delta _ { \omega } \geq \frac { \epsilon } { M } } \operatorname* { P r } \bigg ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) \stackrel { ( 3 ) } { = } \sum _ { \Delta _ { \omega } \geq \frac { \epsilon } { M } } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) ,\tag{8}
$$

where (1) is by MCT, (2) is by lemma 8 and (3) is by Fubini-Tonelli. Since $\pi _ { t } ( a ) = \operatorname* { P r } ( A _ { t } = a | \mathcal { F } _ { t - 1 } )$ , we have that

$$
\sum _ { \Delta _ { a } \geq \frac { \epsilon } { M } } \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) = \sum _ { \Delta _ { a } \geq \frac { \epsilon } { M } } \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \mathrm { P r } ( A _ { t } = a | \mathcal { F } _ { t - 1 } ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) .
$$

For the rest of our analysis, we fix an arm $a \in \mathcal { A } _ { s u b }$ with $\Delta _ { a } \geq \epsilon / M$ . By the law of total probability

$$
\begin{array} { r } { \operatorname* { P r } ( A _ { t } = a | \mathcal { F } _ { t - 1 } ) = \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) + \operatorname* { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) + \operatorname* { P r } \big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) , } \end{array}
$$

which by the union bound implies that

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) } \\ & { \displaystyle = \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , \frac { \overline { { E _ { a } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } } { A ( t ) } \Big ) + \operatorname* { P r } \Big ( A _ { t } = a , \frac { E _ { a } ^ { \mu } ( t ) } { A _ { a } ^ { \nu } ( t ) } | \mathcal { F } _ { t - 1 } \Big ) + \operatorname* { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) } \\ & { \displaystyle \leq \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , \frac { \overline { { E _ { a } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } } { A ( t ) } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) + \underbrace { \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \beta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) } _ { \displaystyle \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , \frac { E _ { a } ^ { \mu } ( t ) } { A _ { a } } \Big ) } } \\ &  \displaystyle \quad + \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _  t - \end{array}
$$

We bound each sum $S _ { 1 } , S _ { 2 } , S _ { 3 }$ separately. We begin with $S _ { 1 }$ :

Lemma 9. It holds that:

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \biggl ( \operatorname* { P r } \Bigl ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Bigr ) > \frac { \epsilon } { 3 M \Delta _ { a } } \biggr ) \leq \frac { 3 3 M } { 2 \epsilon \Delta _ { a } } .
$$

Proof. By Markov’s inequality, we have that:

$$
\mathbb { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { \alpha } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { \alpha } } \bigg ) \leq \frac { \mathbb { E } _ { \mathcal { F } _ { t - 1 } } \left[ \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { \alpha } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) \right] } { \epsilon / 3 M \Delta _ { \alpha } } = \frac { 3 M \Delta _ { \alpha } \mathbb { E } _ { \mathcal { F } _ { t - 1 } } \left[ \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { \alpha } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) \right] } { \epsilon } ,
$$

where $\mathbb { E } _ { \mathcal { F } _ { t } } [ \cdot ]$ denotes the expectation over the randomness of the history. By the tower rule:

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { u } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) \leq \sum _ { t = 1 } ^ { \infty } \frac { 3 M \Delta _ { a } \operatorname { E } _ { \mathcal { F } _ { t - 1 } } \Big [ \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { u } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) \Big ] } { \epsilon } = \frac { 3 M \Delta _ { a } } { \epsilon } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { u } ( t ) } } \Big ) .
$$

We analyze $\textstyle \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \Bigl ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } \Bigr )$ . Note that $\overline { { E _ { a } ^ { \mu } ( t ) } } = \{ \hat { \mu } _ { t } ( a ) > x _ { a } \}$ , an event dependent only on $\hat { \mu } _ { t } ( a )$ , which changes only when arm a is pulled. This implies that

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } \Big ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ A _ { t } = a \} \mathbb { 1 } \{ \hat { \mu } _ { t } ( a ) > x _ { a } \} \right] \stackrel { ( * ) } { \leq } \mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } \mathbb { 1 } \{ \tilde { \mu } _ { a , k } > x _ { a } \} \right] \leq 1 + \sum _ { k = 1 } ^ { \infty } \operatorname* { P r } ( \tilde { \mu } _ { a , k } > x _ { a } ) ,
$$

where (∗) is since $n _ { \infty } ( a ) \leq \infty$ , and on the $( k + 1 ) ^ { t h }$ pull of arm $^ { a , }$ the empirical mean is exactly $\frac { k } { k + 1 } \tilde { \mu } _ { a , k } \leq \tilde { \mu } _ { a , k }$ Applying Hoefding’s inequality (Fact 1) and Fact 3 yields

$$
1 + \sum _ { k = 1 } ^ { \infty } \operatorname* { P r } ( \tilde { \mu } _ { a , k } > x _ { a } ) \leq 1 + \sum _ { k = 1 } ^ { \infty } \exp \Bigl ( - 2 k \bigl ( \Delta _ { a } / 3 \bigr ) ^ { 2 } \Bigr ) = 1 + \frac { e ^ { - 2 \Delta _ { a } ^ { 2 } / 9 } } { 1 - e ^ { - 2 \Delta _ { a } ^ { 2 } / 9 } } \leq 1 + \frac { 9 } { 2 \Delta _ { a } ^ { 2 } } \leq \frac { 1 1 } { 2 \Delta _ { a } ^ { 2 } } ,
$$

where the last inequality is since $\Delta _ { a } \leq 1$ . Combining, we get that

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) \leq \frac { 3 M \Delta _ { a } } { \epsilon } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } \Big ) \leq \frac { 3 M \Delta _ { a } } { \epsilon } \cdot \frac { 1 1 } { 2 \Delta _ { a } ^ { 2 } } = \frac { 3 3 M } { 2 \epsilon \Delta _ { a } } .
$$

To bound $S _ { 2 } .$ , we show the following:

Lemma 10. It holds that:

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \biggl ( \operatorname* { P r } \Bigl ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Bigr ) > \frac { \epsilon } { 3 M \Delta _ { a } } \biggr ) \leq \frac { 1 1 4 M } { \epsilon \Delta _ { a } } \ln \biggl ( \frac { 3 M \Delta _ { a } } { \epsilon } \biggr ) .
$$

Proof. Let $\begin{array} { r } { L _ { a } \triangleq \frac { 1 8 } { \Delta _ { a } ^ { 2 } } \ln \left( \frac { 3 M \Delta _ { a } } { \epsilon } \right) } \end{array}$ , and define a

$$
\begin{array} { r } { Z _ { t } ^ { < } \triangleq \operatorname* { P r } \Big ( A _ { t } = a , n _ { t } ( a ) < L _ { a } , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) , \quad Z _ { t } ^ { \geq } \triangleq \operatorname* { P r } \Big ( A _ { t } = a , n _ { t } ( a ) \geq L _ { a } , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) , } \end{array}
$$

noting that $\mathrm { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) = Z _ { t } ^ { < } + Z _ { t } ^ { \geq }$ for all $t \in \mathbb N$ . Since both $n _ { t } ( a )$ and $\hat { \mu } _ { t } ( a )$ are $\mathcal { F } _ { t - 1 ^ { - } }$ measurable, we have that

$$
\begin{array} { r l } & { Z _ { t } ^ { \geq } = \operatorname* { P r } \Bigl ( A _ { t } = a , n _ { t } ( a ) \geq L _ { a } , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Bigr ) } \\ & { \mathrm { ~ \ ~ \ } = \mathbb { E } [ \mathbb { 1 } \{ A _ { t } = a \} \mathbb { 1 } \{ n _ { t } ( a ) \geq L _ { a } \} \mathbb { 1 } \{ \hat { \mu } _ { t } ( a ) \leq x _ { a } \} \mathbb { 1 } \{ \theta _ { t } ( a ) > y _ { a } \} | \mathcal { F } _ { t - 1 } ] } \\ & { \leq \mathbb { 1 } \{ n _ { t } ( a ) \geq L _ { a } \} \mathbb { 1 } \{ \hat { \mu } _ { t } ( a ) \leq x _ { a } \} \mathbb { E } [ \mathbb { 1 } \{ \theta _ { t } ( a ) > y _ { a } \} | \mathcal { F } _ { t - 1 } ] . } \end{array}
$$

Furthermore, on the event $E _ { a } ^ { \mu } ( t )$ , the conditional posterior sample

$$
\theta _ { t } ( a ) | \mathcal { F } _ { t - 1 } \sim \mathcal { N } \bigg ( \hat { \mu } _ { t } ( a ) , \frac { 1 } { 1 + n _ { t } ( a ) } \bigg )
$$

is stochastically dominated by $\begin{array} { r } { \mathcal { N } \Big ( x _ { a } , \frac { 1 } { 1 + n _ { t } ( a ) } \Big ) } \end{array}$ , and so by Fact 4 we get

$$
\begin{array} { r l } & { \mathbb { 1 } \{ n _ { t } ( a ) \geq L _ { a } \} \mathbb { 1 } \{ \hat { \mu } _ { t } ( a ) \leq x _ { a } \} \mathbb { E } [ \mathbb { 1 } \{ \theta _ { t } ( a ) > y _ { a } \} | \mathcal { F } _ { t - 1 } ] \leq \mathbb { 1 } \{ n _ { t } ( a ) \geq L _ { a } \} \operatorname* { P r } \left( \mathcal { N } \left( x _ { a } , \frac { 1 } { 1 + n _ { t } ( a ) } \right) > y _ { a } | \mathcal { F } _ { t - 1 } \right) } \\ & { \qquad \leq \operatorname* { P r } \left( \mathcal { N } ( 0 , 1 ) > \sqrt { L _ { a } + 1 } ( y _ { a } - x _ { a } ) \right) } \\ & { \qquad \leq \frac { 1 } { 2 } \exp \left( \frac { - L _ { a } \Delta _ { a } ^ { 2 } } { 1 8 } \right) } \\ & { \qquad = \frac { \epsilon } { 6 M \Delta _ { a } } . } \end{array}
$$

The above shows that $\begin{array} { r } { Z _ { t } ^ { \geq } \leq \frac { \epsilon } { 6 M \Delta _ { a } } } \end{array}$ for all $t \in \mathbb N$ , which implies that

$$
\left\{ \operatorname* { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \right\} = \bigg \{ Z _ { t } ^ { < } + Z _ { t } ^ { \geq } > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg \} \subseteq \bigg \{ Z _ { t } ^ { < } > \frac { \epsilon } { 6 M \Delta _ { a } } \bigg \} ,
$$

and as such

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( \operatorname* { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) \leq \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigg ( Z _ { t } ^ { < } > \frac { \epsilon } { 6 M \Delta _ { a } } \bigg ) .
$$

By Markov’s inequality:

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( Z _ { t } ^ { < } > \frac { \epsilon } { 6 M \Delta _ { a } } \bigg ) \leq \frac { 6 M \Delta _ { a } } { \epsilon } \sum _ { t = 1 } ^ { \infty } \mathbb { E } \big [ Z _ { t } ^ { < } \big ] } \\ { \displaystyle \leq \frac { 6 M \Delta _ { a } } { \epsilon } \mathbb { E } \bigg [ \sum _ { t = 1 } ^ { \infty } \mathbb { E } \big [ \mathbb { I } \left\{ A _ { t } = a \right\} \mathbb { I } \left\{ n _ { t } ( a ) < L _ { a } \right\} \mathbb { I } \left\{ \hat { \mu } _ { t } ( a ) \leq x _ { a } \right\} \mathbb { I } \left\{ \theta _ { t } ( a ) > y _ { a } \right\} \big | { \mathcal F } _ { t - 1 } \big ] \bigg ] } \\ { \displaystyle \leq \frac { 6 M \Delta _ { a } } { \epsilon } \mathbb { E } \big [ \big [ L _ { a } \big ] \big ] } \\ { \displaystyle = \frac { 6 M \Delta _ { a } } { \epsilon } [ L _ { a } ] , } \end{array}
$$

where the last inequality is since for any history, the indicators 1 $\{ A _ { t } = a \} \mathbb { 1 } \{ n _ { t } ( a ) < L _ { a } \}$ can happen at most $\lceil L _ { a } \rceil$ times since each time $\mathbb { 1 } \{ A _ { t } = a \}$ occurs, $n _ { t } ( a )$ increases by 1. Combining and substituting in the definition of $L _ { a }$ yields

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \biggr ( \operatorname* { P r } \biggr ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \biggr ) > \frac { \epsilon } { 3 M \Delta _ { a } } \biggr ) \leq \frac { 6 M \Delta _ { a } } { \epsilon } \bigl [ L _ { a } \bigr ] \leq \frac { 1 1 4 M } { \epsilon \Delta _ { a } } \ln \biggr ( \frac { 3 M \Delta _ { a } } { \epsilon } \biggr ) ,
$$

where the last inequality is since ln $( 3 M \Delta _ { a } / \epsilon ) \geq 1$ for $\Delta _ { a } \geq \epsilon / M$

□

Finally, we bound $S _ { 3 } { \mathrm { : } }$

Lemma 11. There exists a universal constant $C > 0$ such that

$$
\sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \mathrm { P r } \big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) \leq \frac { C M } { \epsilon \Delta _ { a } } .
$$

Proof. Fix an arbitrary optimal arm $a ^ { * } \in { \mathcal { A } } ^ { * }$ , and define

$$
p _ { a , t } \triangleq \operatorname* { P r } ( \theta _ { t } ( a ^ { * } ) > y _ { a } \vert \mathcal { F } _ { t - 1 } ) , \qquad q _ { a , t } \triangleq \frac { 1 - p _ { a , t } } { p _ { a , t } } .
$$

Lemma 12. For every round t, we have that

$$
\begin{array} { r } { \operatorname* { P r } \big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) \leq q _ { a , t } \operatorname* { P r } \big ( A _ { t } = a ^ { * } , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \big ) . } \end{array}
$$

Proof. We begin by noting that $E _ { a } ^ { \mu } ( t )$ is $\mathcal { F } _ { t - 1 } \mathrm { - m e a s u r a b l e } ,$ and thus on any instantiation of $\mathcal { F } _ { t - 1 }$ such that $E _ { a } ^ { \mu } ( t )$ is false, the probability on the left-hand side is 0 and thus the inequality holds trivially. We let $F _ { t - 1 }$ be an instantiation of $\mathcal { F } _ { t - 1 }$ on which $E _ { a } ^ { \mu } ( t )$ is true, and define

$$
B _ { a , t } \triangleq \{ \theta _ { t } ( a ) \leq y _ { a } \} \cap \{ \theta _ { t } ( a ) \geq \theta _ { t } ( b ) , \forall b \neq a ^ { * } \} .
$$

Note that $B _ { a , t }$ is conditionally independent of $\boldsymbol { \theta } _ { t } ( \boldsymbol { a } ^ { * } )$ given $F _ { t - 1 } .$ and so by the definition of $E _ { a } ^ { \theta } ( t )$

$$
\begin{array} { r l } { \operatorname* { P r } \bigl ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } = F _ { t - 1 } \bigr ) \leq \operatorname* { P r } ( \{ \theta _ { t } ( a ) \leq y _ { a } \} \cap \{ \forall b \in [ K ] : \theta _ { t } ( a ) \geq \theta _ { t } ( b ) \} | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) } & { } \\ { = \operatorname* { P r } ( B _ { a , t } , \theta _ { t } ( a ^ { * } ) \leq \theta _ { t } ( a ) | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) } & { } \\ { \leq \operatorname* { P r } ( B _ { a , t } , \theta _ { t } ( a ^ { * } ) \leq y _ { a } | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) } & { } \\ { \leq ( 1 - p _ { a , t } ) \operatorname* { P r } ( B _ { a , t } | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) . } & { } \end{array}
$$

Moreover, we have that

$$
\begin{array} { r l } & { \operatorname* { P r } \big ( A _ { t } = a ^ { * } , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } = F _ { t - 1 } \big ) \geq \operatorname* { P r } ( \{ \theta _ { t } ( a ) \leq y _ { a } \} \cap \{ \forall b \in [ K ] : \theta _ { t } ( a ^ { * } ) \geq \theta _ { t } ( b ) \} | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) } \\ & { \qquad \geq \operatorname* { P r } ( B _ { a , t } , \theta _ { t } ( a ^ { * } ) \geq \theta _ { t } ( a ) | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) } \\ & { \qquad \geq \operatorname* { P r } ( B _ { a , t } , \theta _ { t } ( a ^ { * } ) > y _ { a } | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) } \\ & { \qquad = p _ { a , t } \operatorname* { P r } ( B _ { a , t } | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) . } \end{array}
$$

Rearranging and combining with the above yields

$$
\mathrm { P r } \Big ( A _ { \varepsilon } = a , E _ { \alpha } ^ { \mu } ( t ) , E _ { \alpha } ^ { \beta } ( t ) | \mathcal { F } _ { t - 1 } = F _ { t - 1 } \Big ) \leq ( 1 - p _ { \alpha , t } ) \mathrm { P r } ( B _ { \alpha , t } | \mathcal { F } _ { t - 1 } = F _ { t - 1 } ) \leq \frac { 1 - p _ { \alpha , t } } { p _ { \alpha , t } } \mathrm { P r } \Big ( A _ { \varepsilon } = a ^ { * } , E _ { \alpha } ^ { \mu } ( t ) , E _ { \alpha } ^ { \beta } ( t ) | \mathcal { F } _ { t - 1 } = F _ { t - 1 } \Big ) ,
$$

which shows that the inequality holds on histories where $E _ { a } ^ { \mu } ( t )$ is true, concluding the proof.

In order to bound the sum $S _ { 3 }$ , we bound $\mathbb { E } [ q _ { a , t } ]$ and then combine with the above to get our final result. Denote:

$$
\hat { \mu } _ { k } ^ { * } \triangleq \frac { 1 } { k + 1 } \sum _ { s = 1 } ^ { k } X _ { a ^ { * } , s } , \qquad \tilde { p } _ { a , k } \triangleq \operatorname* { P r } \bigg ( \mathcal { N } \bigg ( \hat { \mu } _ { k } ^ { * } , \frac { 1 } { 1 + k } \bigg ) > y _ { a } \bigg | X _ { a ^ { * } , 1 } , \ldots , X _ { a ^ { * } , k } \bigg ) , \qquad \tilde { q } _ { a , k } \triangleq \frac { 1 - \tilde { p } _ { a , k } } { \tilde { p } _ { a , k } } .
$$

We show the following:

Lemma 13. There exists an absolute constant $C _ { 1 } > 0$ such that

$$
\sum _ { k = 0 } ^ { \infty } \mathbb { E } [ \tilde { q } _ { a , k } ] \leq \frac { C _ { 1 } } { \Delta _ { a } ^ { 2 } } .
$$

Proof. Let $\begin{array} { r } { H _ { a , k } \triangleq \left\{ \hat { \mu } _ { k } ^ { * } \geq y _ { a } + \frac { \Delta _ { a } } { 6 } \right\} } \end{array}$ . We decompose $\tilde { q } _ { a , k }$ as

$$
\tilde { q } _ { a , k } = \tilde { q } _ { a , k } \mathbb { 1 } \{ H _ { a , k } \} + \tilde { q } _ { a , k } \mathbb { 1 } \{ \overline { { H _ { a , k } } } \} ,
$$

and bound each part separately. We begin by looking at $\begin{array} { r } { \tilde { q } _ { a , k } \Im \{ H _ { a , k } \} = \frac { 1 - \tilde { p } _ { a , k } } { \tilde { p } _ { a , k } } \Im \{ H _ { a , k } \} } \end{array}$ . Note that on $H _ { a , k }$ we have $\begin{array} { r } { \hat { \mu } _ { k } ^ { * } - y _ { a } \ge \frac { \Delta _ { a } } { 6 } } \end{array}$ , which by Fact 4 implies that:

$$
\begin{array} { r l } & { ( 1 - \tilde { p } _ { a , k } ) { \mathbb 1 } \{ H _ { a , k } \} = \operatorname* { P r } \biggl ( \Lambda ^ { \ast } \biggl ( \hat { \mu } _ { k } ^ { \ast } , \frac { 1 } { 1 + k } \biggr ) \leq y _ { a } \biggl | X _ { a ^ { \ast } , 1 } , \dots , X _ { a ^ { \ast } , k } \biggr ) { \mathbb 1 } \{ H _ { a , k } \} } \\ & { \qquad = \operatorname* { P r } \Bigl ( { \mathcal N } ( 0 , 1 ) \leq - \sqrt { k + 1 } ( \hat { \mu } _ { k } ^ { \ast } - y _ { a } ) \Bigl | X _ { a ^ { \ast } , 1 } , \dots , X _ { a ^ { \ast } , k } \Bigr ) { \mathbb 1 } \{ H _ { a , k } \} } \\ & { \qquad \leq \operatorname* { P r } \biggl ( { \mathcal N } ( 0 , 1 ) > \frac { \Delta _ { a } \sqrt { k + 1 } } { 6 } \biggl | X _ { a ^ { \ast } , 1 } , \dots , X _ { a ^ { \ast } , k } \biggr ) { \mathbb 1 } \{ H _ { a , k } \} } \\ & { \qquad \leq \frac { 1 } { 2 } \exp \biggl ( - \frac { \Delta _ { a } ^ { 2 } ( k + 1 ) } { 7 2 } \biggr ) . } \end{array}
$$

Let $\begin{array} { r } { x = \exp \left( { - \frac { \Delta _ { a } ^ { 2 } ( k { + } 1 ) } { 7 2 } } \right) } \end{array}$ . We showed that $( 1 - \tilde { p } _ { a , k } ) \mathbb { 1 } \{ H _ { a , k } \} \leq \frac { x } { 2 }$ , thus $\begin{array} { r } { \tilde { p } _ { a , k } \Im \{ H _ { a , k } \} \ge \left( 1 - \frac { x } { 2 } \right) \Im \{ H _ { a , k } \} } \end{array}$ , which gives us

$$
\tilde { q } _ { a , k } \mathbb { 1 } \{ H _ { a , k } \} = \frac { 1 - \tilde { p } _ { a , k } } { \tilde { p } _ { a , k } } \mathbb { 1 } \{ H _ { a , k } \} \leq \frac { \frac { x } { 2 } } { 1 - \frac { x } { 2 } } \mathbb { 1 } \{ H _ { a , k } \} .
$$

We use the fact that ${ \frac { { \frac { 1 } { 2 } } x } { 1 - { \frac { 1 } { 2 } } x } } \leq x$ for all $x \in [ 0 , 1 ]$ to get

$$
\tilde { q } _ { a , k } \Im \{ H _ { a , k } \} \le \exp \biggl ( - \frac { \Delta _ { a } ^ { 2 } ( k + 1 ) } { 7 2 } \biggr ) \Im \{ H _ { a , k } \} ,
$$

bounding $\tilde { q } _ { a , k }$ under the event $H _ { a , k }$ . We now analyze the complementary summand. We show that for all k:

$$
\mathrm { P r } \left( \overline { { { H _ { a , k } } } } \right) \leq 2 \exp \left( - \frac { k \Delta _ { a } ^ { 2 } } { 7 2 } \right) .
$$

For $k = 0$ , this bound is trivial. Hence assume $k \geq 1$ . Using the definition of $\hat { \mu } _ { k } ^ { * }$ and the fact that $\mu ^ { * } \leq 1$ , we have that

$$
\overline { { H _ { o , k } } } = \left\{ \widehat { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { \Delta _ { a } } { 6 } \right\} \subseteq \left\{ \frac { k } { k + 1 } \widehat { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { \Delta _ { a } } { 6 } \right\} \subseteq \left\{ \widehat { \mu } _ { k } ^ { * } < \mu ^ { * } + \frac { \mu ^ { * } } { k } - \frac { k + 1 } { k } \cdot \frac { \Delta _ { a } } { 6 } \right\} ^ { \mu ^ { * } \overset { * } { \subseteq } 1 } \left\{ \widehat { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { \Delta _ { a } } { 6 } + \frac { 1 } { k } \right\}
$$

We look at two cases:

For $\begin{array} { r } { k \geq \frac { 1 2 } { \Delta _ { a } } } \end{array}$ , we have that $\begin{array} { r l r } & { } & { \left\{ \tilde { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { \Delta _ { a } } { 6 } + \frac { 1 } { k } \right\} \subseteq \left\{ \tilde { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { \Delta _ { a } } { 1 2 } \right\} } \end{array}$ , which by Hoefding’s inequality (Fact 1) implies that

$$
\mathrm { P r } \big ( \overline { { H _ { a , k } } } \big ) \leq \mathrm { P r } \bigg ( \tilde { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { \Delta _ { a } } { 1 2 } \bigg ) \leq \exp \bigg ( { - \frac { k \Delta _ { a } ^ { 2 } } { 7 2 } } \bigg ) ,
$$

Thus satisfying the required inequality.

For $\begin{array} { r } { k < \frac { 1 2 } { \Delta _ { a } } } \end{array}$ , we have that $\begin{array} { r } { \frac { k \Delta _ { a } ^ { 2 } } { 7 2 } < \frac { 1 } { 6 } } \end{array}$ , thus yielding

$$
\mathrm { P r } \bigl ( \overline { { H _ { a , k } } } \bigr ) \leq 1 = e ^ { \frac { 1 } { 6 } } e ^ { - \frac { 1 } { 6 } } \leq e ^ { \frac { 1 } { 6 } } \exp \biggl ( { - \frac { k \Delta _ { a } ^ { 2 } } { 7 2 } } \biggr ) ,
$$

thus we have that $\begin{array} { r } { \mathrm { P r } \big ( \overline { { H _ { a , k } } } \big ) \leq 2 \exp \Big ( - \frac { k \Delta _ { a } ^ { 2 } } { 7 2 } \Big ) } \end{array}$ for all k. By Cauchy-Schwarz:

$$
\mathbb { E } \big [ \tilde { q } _ { a , k } \mathbb { 1 } \big \{ \overline { { H _ { a , k } } } \big \} \big ] \leq \sqrt { \mathbb { E } \Big [ \tilde { q } _ { a , k } ^ { 2 } \Big ] \operatorname* { P r } \big ( \overline { { H _ { a , k } } } \big ) } \leq \sqrt { 2 \mathbb { E } \Big [ \tilde { q } _ { a , k } ^ { 2 } \Big ] } \exp \bigg ( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \bigg ) .\tag{9}
$$

We now show that $\mathbb { E } \left[ \tilde { q } _ { a , k } ^ { 2 } \right]$ is bounded by an absolute constant for all k. Let $\begin{array} { r } { Y _ { i } ^ { k } | \hat { \mu } _ { k } ^ { * } \sim \mathcal N \Big ( \hat { \mu } _ { k } ^ { * } , \frac { 1 } { 1 + k } \Big ) } \end{array}$ be i.i.d. samples from the posterior of the optimal arm after k draws. Define $G _ { k } | \hat { \mu } _ { k } ^ { * } \triangleq \operatorname* { i n f } \left\{ i \geq 1 : Y _ { i } ^ { k } > y _ { a } \right\}$ to be the number of samples needed from $\begin{array} { r } { \mathcal { N } \Big ( \hat { \mu } _ { k } ^ { * } , \frac { 1 } { 1 + k } \Big ) } \end{array}$ to obtain a sample larger than $y _ { a } .$ , noting that $G _ { k } | \hat { \mu } _ { k } ^ { * } \sim G e o ( \tilde { p } _ { a , k } )$ We show that there exists a constant $C _ { G }$ such that

$$
\operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \big [ G _ { k } ^ { 2 } \big ] \leq C _ { G } ,
$$

since by the formula for the second moment of a geometric RV:

$$
\mathbb { E } \big [ G _ { k } ^ { 2 } | \hat { \mu } _ { k } ^ { * } \big ] = \frac { 2 - \tilde { p } _ { a , k } } { \tilde { p } _ { a , k } ^ { 2 } } \overset { \tilde { p } _ { a , k } \leq 1 } { \geq } \frac { 1 } { \tilde { p } _ { a , k } ^ { 2 } } ,
$$

which implies that

$$
\operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \big [ \tilde { q } _ { a , k } ^ { 2 } \big ] = \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \left[ \frac { ( 1 - \tilde { p } _ { a , k } ) ^ { 2 } } { \tilde { p } _ { a , k } ^ { 2 } } \right] \leq \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \left[ \frac { 1 } { \tilde { p } _ { a , k } ^ { 2 } } \right] \leq \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \big [ \mathbb { E } \big [ G _ { k } ^ { 2 } \big | X _ { a ^ { * } , 1 } , \dots , X _ { a ^ { * } , k } \big ] \big ] \leq \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \big [ G _ { k } ^ { 2 } \big ] \leq C _ { G } .
$$

For $k = 0$ , the result follows since $1 \geq \mu ^ { * } \geq y _ { a }$ , which implies that $\tilde { p } _ { a , 0 } \geq \operatorname* { P r } ( \mathcal { N } ( 0 , 1 ) > 1 ) > 0 . 1$ , and so $\begin{array} { r } { \mathbb { E } \big [ G _ { 0 } ^ { 2 } \big ] = \frac { 2 - \tilde { p } _ { a , 0 } } { \tilde { p } _ { a , 0 } ^ { 2 } } < } \end{array}$ 200 by the formula for the second moment of a geometric random variable. Let $k \geq 1$ , let $r \geq 2$ and set $z _ { r } \triangleq \sqrt { \frac { 3 } { 2 } \ln ( r ) }$ . Since $y _ { a } \le \mu ^ { * }$ , we have that

$$
\big \{ G _ { k } \leq r \big \} \supseteq \Bigg \{ \operatorname* { m a x } _ { j \in [ r ] } Y _ { j } ^ { k } > \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } \Bigg \} \cap \bigg \{ \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } \geq \mu ^ { * } \bigg \} .
$$

Taking the complement and applying the union bound, we get

$$
\begin{array} { r } { \operatorname* { P r } ( G _ { k } > r ) \leq \operatorname* { P r } \biggr ( \biggr \{ \underset { j \in [ r ] } { \operatorname* { m a x } } Y _ { j } ^ { k } \leq \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } \biggr \} \cup \biggr \{ \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } < \mu ^ { * } \biggr \} \biggr ) } \\ { \leq \underbrace { \operatorname* { P r } \biggr ( \underset { j \in [ r ] } { \operatorname* { m a x } } Y _ { j } ^ { k } \leq \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } \biggr ) } _ { ( A ) } + \underbrace { \operatorname* { P r } \biggr ( \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } < \mu ^ { * } \biggr ) } _ { ( B ) } . } \end{array}
$$

To bound (A), we note that the sequence of samples $Y _ { i } ^ { k }$ is i.i.d, giving us

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { j \in [ r ] } Y _ { j } ^ { k } \leq \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } \right) = ( 1 - \operatorname* { P r } ( \mathcal { N } ( 0 , 1 ) > z _ { r } ) ) ^ { r } \leq \exp ( - r \operatorname* { P r } ( \mathcal { N } ( 0 , 1 ) > z _ { r } ) ) .
$$

Applying the lower bound on the tail of a Gaussian RV (Fact 4) yields

$$
\exp ( - r \operatorname* { P r } ( \mathcal { N } ( 0 , 1 ) > z _ { r } ) ) \leq \exp \left( - r \frac { 1 } { \sqrt { 2 \pi } } \frac { z _ { r } } { 1 + z _ { r } ^ { 2 } } e ^ { - z _ { r } ^ { 2 } / 2 } \right) .
$$

Substituting back the definition of $z _ { r }$

$$
\exp \Biggl ( - r \frac { 1 } { \sqrt { 2 \pi } } \frac { z _ { r } } { 1 + z _ { r } ^ { 2 } } e ^ { - z _ { r } ^ { 2 } / 2 } \Biggr ) = \exp \Biggl ( - r \frac { 1 } { \sqrt { 2 \pi } } \frac { z _ { r } } { 1 + z _ { r } ^ { 2 } } e ^ { - \left( \sqrt { \frac { 3 } { 2 } \ln ( r ) } \right) ^ { 2 } / 2 } \Biggr ) \leq \exp \Biggl ( - r ^ { 1 / 4 } \frac { 1 } { \sqrt { 2 \pi } } \frac { z _ { r } } { 1 + z _ { r } ^ { 2 } } \Biggr ) .
$$

Note that for $r \geq 2$ , we have $z _ { r } \geq 1$ and thus $\begin{array} { r } { \frac { z _ { r } } { 1 + z _ { r } ^ { 2 } } \geq \frac { 1 } { 2 z _ { r } } } \end{array}$ , which gives us

$$
\exp \left( - r ^ { 1 / 4 } \frac { 1 } { \sqrt { 2 \pi } } \frac { z _ { r } } { 1 + z _ { r } ^ { 2 } } \right) \leq \exp \left( - r ^ { 1 / 4 } \frac { 1 } { 2 \sqrt { 2 \pi } z _ { r } } \right) \leq \exp \left( - \frac { r ^ { 1 / 4 } } { 2 \sqrt { 3 \pi } \ln ( r ) } \right) .
$$

Note that $\ln ( r ) ^ { 3 / 2 } = o \left( r ^ { 1 / 4 } \right)$ , implying that there exists a constant $r _ { 0 } > 0$ such that for all $r \geq r _ { 0 }$ , we have that

$$
{ \frac { r ^ { 1 / 4 } } { 2 { \sqrt { 3 \pi \ln ( r ) } } } } \geq 3 \ln ( r ) .
$$

Thus, for all $r \geq \operatorname* { m a x } \{ 2 , r _ { 0 } \}$

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { j \in [ r ] } Y _ { j } ^ { k } \leq \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } \right) \leq \exp \left( - \frac { r ^ { 1 / 4 } } { 2 \sqrt { 3 \pi \ln ( r ) } } \right) \leq \exp ( - 3 \ln ( r ) ) \leq \frac { 1 } { r ^ { 3 } } .
$$

To bound (B), we let $\tilde { \mu } _ { k } ^ { \ast } \triangleq \tilde { \mu } _ { a ^ { \ast } , k } .$ and note that $\begin{array} { r } { \hat { \mu } _ { k } ^ { * } = \frac { k } { k + 1 } \tilde { \mu } _ { k } ^ { * } } \end{array}$ , and thus:

$$
\left\{ { \hat { \mu } } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } < \mu ^ { * } \right\} \subseteq \left\{ \frac { k } { k + 1 } { \tilde { \mu } } _ { k } ^ { * } < \mu ^ { * } - \frac { z _ { r } } { \sqrt { k + 1 } } \right\} \subseteq \left\{ { \tilde { \mu } } _ { k } ^ { * } < \frac { k + 1 } { k } \mu ^ { * } - \frac { k + 1 } { k } \frac { z _ { r } } { \sqrt { k + 1 } } \right\} \subseteq \left\{ { \tilde { \mu } } _ { k } ^ { * } < \mu ^ { * } - \frac { z _ { r } - 1 } { \sqrt { k } } \right\} .
$$

Therefore by Hoefding (Fact 1):

$$
\operatorname* { P r } \biggl ( \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } < \mu ^ { * } \biggr ) \leq \operatorname* { P r } \biggl ( \tilde { \mu } _ { k } ^ { * } < \mu ^ { * } - \frac { z _ { r } - 1 } { \sqrt { k } } \biggr ) \leq \exp \Bigl ( - 2 ( z _ { r } - 1 ) ^ { 2 } \Bigr ) = \exp \bigl ( - 2 z _ { r } ^ { 2 } + 4 z _ { r } - 2 \bigr ) .
$$

Opening the exponent and substituting in the definition of $z _ { r }$ yields

$$
\exp \Bigl ( - 2 ( z _ { r } - 1 ) ^ { 2 } \Bigr ) = \exp \bigl ( - 2 z _ { r } ^ { 2 } + 4 z _ { r } - 2 \bigr ) = \exp \Biggl ( - 2 \frac { 3 } { 2 } \ln ( r ) + 4 \sqrt { \frac { 3 } { 2 } \ln ( r ) } - 2 \Biggr ) = \frac { 1 } { r ^ { 3 } } \exp \Biggl ( 4 \sqrt { \frac { 3 } { 2 } \ln ( r ) } - 2 \Biggr ) .
$$

Since $4 \sqrt { \textstyle { \frac { 3 } { 2 } } \ln ( r ) } - 2 = o ( \ln ( r ) )$ , there exists $r _ { 1 } > 0$ such that for all $r > r _ { 1 }$ , we have that

$$
4 { \sqrt { \frac { 3 } { 2 } \ln ( r ) } } - 2 < { \frac { 1 } { 2 } } \ln ( r ) .
$$

Thus, for all $r > r _ { 1 }$

$$
\operatorname* { P r } \biggr ( \hat { \mu } _ { k } ^ { * } + \frac { z _ { r } } { \sqrt { k + 1 } } < \mu ^ { * } \biggr ) \leq \frac { 1 } { r ^ { 3 } } \exp \biggr ( 4 \sqrt { \frac { 3 } { 2 } \ln ( r ) } - 2 \biggr ) \leq \frac { 1 } { r ^ { 5 / 2 } } .
$$

Combining, we get that for all $k ,$ , and for all $r \geq R \triangleq \operatorname* { m a x } \{ 2 , r _ { 0 } , r _ { 1 } \}$

$$
\operatorname* { P r } ( G _ { k } > r ) \leq \frac { 1 } { r ^ { 3 } } + \frac { 1 } { r ^ { 5 / 2 } } \leq \frac { 2 } { r ^ { 5 / 2 } } .\tag{10}
$$

Recalling that any $n \in \mathbb { N }$ can be written as $\begin{array} { r } { n ^ { 2 } = \sum _ { r = 0 } ^ { n - 1 } ( 2 r + 1 ) } \end{array}$ , we have that for any RV X with support in N:

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big [ X ^ { 2 } \big ] = \sum _ { n = 1 } ^ { \infty } n ^ { 2 } \operatorname* { P r } ( X = n ) = \sum _ { n = 1 } ^ { \infty } \operatorname* { P r } ( X = n ) \sum _ { r = 0 } ^ { n - 1 } ( 2 r + 1 ) } \\ { \displaystyle \qquad = \sum _ { n = 1 } ^ { \infty } \operatorname* { P r } ( X = n ) \sum _ { r = 0 } ^ { \infty } ( 2 r + 1 ) \mathbb { 1 } \big \{ n > r \big \} = \mathbb { E } \left[ \sum _ { r = 0 } ^ { \infty } ( 2 r + 1 ) \mathbb { 1 } \big \{ X > r \big \} \right] , } \end{array}
$$

giving us that

$$
\mathbb { E } \big [ G _ { k } ^ { 2 } \big ] = \mathbb { E } \left[ \sum _ { r = 0 } ^ { \infty } ( 2 r + 1 ) \mathbb { 1 } \{ G _ { k } > r \} \right] .
$$

Splitting the sum into $r \leq R$ and $r > R$ and using Equation (10) gives:

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \displaystyle \sum _ { r = 0 } ^ { \infty } ( 2 r + 1 ) \mathbb { 1 } \left\{ G _ { k } > r \right\} \right] = \mathbb { E } \left[ \displaystyle \sum _ { r = 0 } ^ { R } ( 2 r + 1 ) \mathbb { 1 } \left\{ G _ { k } > r \right\} + \displaystyle \sum _ { r = R + 1 } ^ { \infty } ( 2 r + 1 ) \mathbb { 1 } \left\{ G _ { k } > r \right\} \right] } & { } \\ { \displaystyle \le \sum _ { r = 0 } ^ { R } ( 2 r + 1 ) + \mathbb { E } \left[ \displaystyle \sum _ { r = R + 1 } ^ { \infty } ( 2 r + 1 ) \mathbb { 1 } \left\{ G _ { k } > r \right\} \right] } & { } \\ { \displaystyle \le ( R + 1 ) ^ { 2 } + \displaystyle \sum _ { r = R + 1 } ^ { \infty } \frac { 4 r + 2 } { r ^ { 5 / 2 } } . } \end{array}
$$

Since $\scriptstyle \sum _ { r = R + 1 } ^ { \infty } { \frac { 4 r + 2 } { r ^ { 5 / 2 } } }$ is a convergent sum, we can let $\begin{array} { r } { C _ { G } \triangleq \operatorname* { m a x } \{ 2 0 0 , ( R + 1 ) ^ { 2 } + \sum _ { r = R + 1 } ^ { \infty } \frac { 4 r + 2 } { r ^ { 5 / 2 } } \} < \infty } \end{array}$ be a universal constant, since R is a universal constant, and so by the above we have that for all $k \geq 0 \colon$

$$
\mathbb { E } \big [ G _ { k } ^ { 2 } \big ] \leq C _ { G } .
$$

We recall that this implies that $\begin{array} { r } { \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \Big [ \tilde { q } _ { a , k } ^ { 2 } \Big ] \leq C _ { G } } \end{array}$ , which by Equation (9) implies that

$$
\mathbb { E } \left[ \tilde { q } _ { a , k } \mathbb { 1 } \left\{ \overline { { H _ { a , k } } } \right\} \right] \leq \sqrt { 2 \mathbb { E } \left[ \tilde { q } _ { a , k } ^ { 2 } \right] } \exp \left( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \right) \leq \sqrt { 2 C _ { G } } \exp \left( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \right) .
$$

We already showed that $\begin{array} { r } { { \mathbb { E } } [ \tilde { q } _ { a , k } \mathbf { 1 } \{ H _ { a , k } \} ] \le \exp \left( - \frac { \Delta _ { a } ^ { 2 } ( k + 1 ) } { 7 2 } \right) } \end{array}$ , thus yielding

$$
\sum _ { k = 0 } ^ { \infty } \mathbb { E } [ \tilde { q } _ { a , k } ] = \sum _ { k = 0 } ^ { \infty } ( \mathbb { E } [ \tilde { q } _ { a , k } ~ \mathbf { 1 } \left\{ H _ { a , k } \right\} ] + \mathbb { E } \left[ \tilde { q } _ { a , k } ~ \mathbf { 1 } \left\{ \overline { { H _ { a , k } } } \right\} \right] ) \leq \sum _ { k = 0 } ^ { \infty } \left( \exp \left( - \frac { \Delta _ { a } ^ { 2 } ( k + 1 ) } { 7 2 } \right) + \sqrt { 2 C _ { G } } \exp \left( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \right) \right) .
$$

Noting that $\begin{array} { r } { \exp \Bigl ( - \frac { \Delta _ { a } ^ { 2 } ( k + 1 ) } { 7 2 } \Bigr ) \le \sqrt { 2 C _ { G } } \exp \Bigl ( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \Bigr ) } \end{array}$ for all $k \geq 0$ and applying the formula for the sum of an infinite geometric series, we get that

$$
\sum _ { k = 0 } ^ { \infty } \biggl ( \exp \biggl ( - \frac { \Delta _ { a } ^ { 2 } ( k + 1 ) } { 7 2 } \biggr ) + \sqrt { 2 C _ { G } } \exp \biggl ( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \biggr ) \biggr ) \leq 2 \sqrt { 2 C _ { G } } \sum _ { k = 0 } ^ { \infty } \exp \biggl ( - \frac { k \Delta _ { a } ^ { 2 } } { 1 4 4 } \biggr ) \leq \sqrt { 8 C _ { G } } \frac { 1 } { 1 - e ^ { - \frac { \Delta _ { a } ^ { 2 } } { 1 4 4 } } } .
$$

Finally, using Fact 3 and taking $C _ { 1 } = 1 4 5 \sqrt { 8 C _ { G } }$ , we get that

$$
\sum _ { k = 0 } ^ { \infty } \mathbb { E } [ \tilde { q } _ { a , k } ] \leq \sqrt { 8 C _ { G } } \frac { 1 } { 1 - e ^ { - \frac { \Delta _ { a } ^ { 2 } } { 1 4 4 } } } \leq \sqrt { 8 C _ { G } } \bigg ( 1 + \frac { 1 4 4 } { \Delta _ { a } ^ { 2 } } \bigg ) \leq \frac { C _ { 1 } } { \Delta _ { a } ^ { 2 } } ,
$$

where the last inequality is since $\Delta _ { a } ^ { 2 } \leq 1$

Combining the above, we get

$$
\sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \biggl ( \operatorname* { P r } \bigl ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \bigr ) > \frac { \epsilon } { 3 M \Delta _ { a } } \biggr )
$$

$$
\leq \frac { 3 M \Delta _ { a } } { \epsilon } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \bigl ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) \bigr )
$$

(Markov’s inequality)

$$
\leq \frac { 3 M \Delta _ { a } } { \epsilon } \mathbb { E } \Bigg [ \sum _ { t = 1 } ^ { \infty } q _ { a , t } \mathbb { 1 } \{ A _ { t } = a ^ { * } \} \mathbb { 1 } \{ E _ { a } ^ { \mu } ( t ) \} \mathbb { 1 } \{ E _ { a } ^ { \theta } ( t ) \} \Bigg ]\tag{lemma 12}
$$

$$
\leq \frac { 3 M \Delta _ { a } } { \epsilon } \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } q _ { a , t } \mathbb { 1 } \left\{ A _ { t } = a ^ { * } \right\} \right] .
$$

Note that in round t where arm $a ^ { * }$ is pulled for the $( k + 1 ) ^ { t h }$ time, we have $q _ { a , t } = \tilde { q } _ { a , k }$ . Using this, we get that

$$
\frac { 3 M \Delta _ { a } } { \epsilon } \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } q _ { a , t } \mathbb { 1 } \{ A _ { t } = a ^ { * } \} \right] \leq \frac { 3 M \Delta _ { a } } { \epsilon } \mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } \tilde { q } _ { a , k } \right] = \frac { 3 M \Delta _ { a } } { \epsilon } \sum _ { k = 0 } ^ { \infty } \mathbb { E } [ \tilde { q } _ { a , k } ] \leq \frac { 3 M \Delta _ { a } } { \epsilon } \frac { C _ { 1 } } { \Delta _ { a } ^ { 2 } } ,
$$

where the last inequality is by lemma 13. Taking $C = 3 C _ { 1 }$ yields the desired inequality.

We combine Lemmas 9, 10 and 11 to get

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \bigg ) } \\ { \displaystyle \leq \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \mathrm { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) + \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \mathrm { P r } \Big ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , \overline { { E _ { a } ^ { \theta } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) } \\ { + \sum _ { t = 1 } ^ { \infty } \mathrm { P r } \bigg ( \mathrm { P r } \Big ( A _ { t } = a , \overline { { E _ { a } ^ { \mu } ( t ) } } | \mathcal { F } _ { t - 1 } \Big ) > \frac { \epsilon } { 3 M \Delta _ { a } } \bigg ) } \\ { \displaystyle \leq \frac { C M } { \epsilon \Delta _ { a } } + \frac { 1 1 4 M } { \epsilon \Delta _ { a } } \ln \bigg ( \frac { 3 M \Delta _ { a } } { \epsilon } \bigg ) + \frac { 3 3 M } { 2 \epsilon \Delta _ { a } } . } \end{array}
$$

The above is true for all arms with $\Delta _ { a } \geq \frac { \epsilon } { M }$ , which in turn implies that

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] = \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } ( \Delta _ { \pi _ { t } } > \epsilon ) } & { } \\ { \displaystyle \leq \sum _ { \Delta _ { a } \geq \frac { \epsilon } { M } } \sum _ { t = 1 } ^ { \infty } \operatorname* { P r } \left( \pi _ { t } ( a ) > \frac { \epsilon } { M \Delta _ { a } } \right) } & { } \\ { \displaystyle \leq \sum _ { \Delta _ { a } \geq \frac { \epsilon } { M } } \left( \frac { C M } { \epsilon \Delta _ { a } } + \frac { 1 1 4 M } { \epsilon \Delta _ { a } } \ln \left( \frac { 3 M \Delta _ { a } } { \epsilon } \right) + \frac { 3 3 M } { 2 \epsilon \Delta _ { a } } \right) . } \end{array}
$$

Defining $C _ { 2 } \triangleq 3 \operatorname* { m a x } \{ C , 1 1 4 \}$ , and noting that for $\Delta _ { a } \geq \frac { \epsilon } { M }$ we have that ln $\left( \frac { 3 M \Delta _ { a } } { \epsilon } \right) \ge 1$ , we get

$$
\sum _ { \Delta _ { a } \geq \frac { c } { M } } \left( \frac { C M } { \epsilon \Delta _ { a } } + \frac { 1 1 4 M } { \epsilon \Delta _ { a } } \ln \left( \frac { 3 M \Delta _ { a } } { \epsilon } \right) + \frac { 3 3 M } { 2 \epsilon \Delta _ { a } } \right) \leq \sum _ { \Delta _ { a } \geq \frac { c } { M } } \frac { C _ { 2 } M } { \epsilon \Delta _ { a } } \ln \left( \frac { 3 M \Delta _ { a } } { \epsilon } \right) .
$$

Noting that $M \leq K$ , we get our desired instance-dependent upper bound.

Corollary 3. For all $\epsilon > 0$ , it holds that:

$$
\mathbb { E } \Big [ N _ { \epsilon } \Big ( \pi ^ { T S ( \mathcal { N } ) } \Big ) \Big ] = \mathcal { O } \Big ( \frac { K ^ { 3 } } { \epsilon ^ { 2 } } \Big ) .
$$

Proof. Let $b > 0$ , and consider the function $\begin{array} { r } { f ( x ) = \frac { 1 } { x } \ln ( b x ) } \end{array}$ on the interval $( 0 , \infty )$ . It holds that:

$$
f ^ { \prime } ( x ) = - { \frac { \ln ( b x ) } { x ^ { 2 } } } + { \frac { 1 } { x ^ { 2 } } } = { \frac { 1 } { x ^ { 2 } } } ( 1 - \ln ( b x ) ) ,
$$

Setting the derivative to 0 yields $\begin{array} { r } { x ^ { * } = \frac { e } { b } } \end{array}$ , and noting that $f ^ { \prime \prime }$ is negative at $x ^ { * }$ implies that $f$ is maximized at that point. Applying this to our instance-dependent bound from Theorem 2 with $\begin{array} { r } { b = \frac { 3 K } { \epsilon } } \end{array}$ , we get

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { \infty } \mathbb { 1 } \big \{ \Delta _ { \pi _ { t } } > \epsilon \big \} \right] \leq \frac { C K } { \epsilon } \displaystyle \sum _ { \Delta _ { a } \geq \frac { \epsilon } { K } } \frac { 1 } { \Delta _ { a } } \ln \left( \frac { 3 K \Delta _ { a } } { \epsilon } \right) } \\ & { \quad \quad \quad \leq \frac { C K } { \epsilon } \displaystyle \sum _ { \Delta _ { a } \geq \frac { \epsilon } { K } } \frac { 1 } { \frac { \epsilon \epsilon } { 3 K } } \ln \left( \frac { 3 K \frac { \epsilon \epsilon } { 3 K } } { \epsilon } \right) } \\ & { \quad \quad \quad \leq \frac { 3 C K ^ { 3 } } { \epsilon \epsilon ^ { 2 } } , } \end{array}
$$

Giving us the desired instance-independent bound.

## D.4 Adding an optimal arm can increase the finite-horizon expected sample complexity

We use simulations to demonstrate that adding an additional optimal arm can strictly increase the T-expected sample complexity $\begin{array} { r } { \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \{ \Delta _ { \pi _ { t } } > \epsilon \} \right] } \end{array}$ . This illustrates why, in contrast to standard regret analyses of Thompson sampling, we cannot assume that the quantity we analyze improves monotonically with the number of optima arms.

Consider the two-armed Bernoulli instance

$$
\nu = ( B e r ( 1 ) , B e r ( 0 ) ) ,
$$

and the augmented three-armed instance

$$
\nu ^ { \prime } = ( B e r ( 1 ) , B e r ( 0 ) , B e r ( 1 ) ) .
$$

Thus, arm 1 is the unique optimal arm in $\nu ,$ while arms 1 and 3 are optimal in $\nu ^ { \prime } .$ . In both instances, arm 2 is the only suboptimal arm and satisfies $\Delta _ { 2 } = 1$

Consider the policy $\pi ^ { T S ( \mathcal { N } ) }$ as defined in Theorem 2. Since the rewards of all arms are deterministic in both instances, the state of the algorithm at the beginning of a round is completely determined by the number of times each arm has been pulled. In particular, if an optimal arm a has been pulled n times, then $S _ { t } ( a ) = n$ while for the suboptimal arm 2 we have $S _ { t } ( 2 ) = 0$ . Therefore, after n pulls of arm $^ { a , }$ its Gaussian sampling distribution is

$$
\theta ( a ) \sim \left\{ \begin{array} { l l } { \displaystyle N \bigg ( \frac { n } { n + 1 } , \frac { 1 } { n + 1 } \bigg ) , } & { a \mathrm { ~ i s ~ o p t i m a l , } } \\ { \displaystyle N \bigg ( 0 , \frac { 1 } { n + 1 } \bigg ) , } & { a = 2 . } \end{array} \right.
$$

Since arm 2 is the only suboptimal arm in both instances and $\Delta _ { 2 } = 1$ , the policy gap is simply the conditional probability of selecting arm 2:

$$
\Delta _ { \pi _ { t } } = \pi _ { t } ( 2 ) = \mathrm { P r } \bigg ( \theta _ { t } ( 2 ) > \operatorname* { m a x } _ { a \neq 2 } \theta _ { t } ( a ) \bigg | \mathcal { F } _ { t - 1 } \bigg ) .
$$

For the two-armed instance $\nu ,$ if the pull counts at the beginning of round t are $n _ { 1 } , n _ { 2 }$ , the conditional probability of selecting the suboptimal arm admits the closed form

$$
\pi _ { t } ( 2 ) = \Phi \left( - { \frac { \frac { n _ { 1 } } { n _ { 1 } + 1 } } { \sqrt { { \frac { 1 } { n _ { 1 } + 1 } } + { \frac { 1 } { n _ { 2 } + 1 } } } } } \right) .
$$

For the three-armed instance $\nu ^ { \prime } ,$ if the pull counts are $n _ { 1 } , n _ { 2 } , n _ { 3 }$ , the analogous probability is

$$
\pi _ { t } ( 2 ) = \int _ { - \infty } ^ { \infty } \phi ( z ) \Phi \Bigg ( \sqrt { n _ { 1 } + 1 } \Bigg ( \frac { z } { \sqrt { n _ { 2 } + 1 } } - \frac { n _ { 1 } } { n _ { 1 } + 1 } \Bigg ) \Bigg ) \Phi \Bigg ( \sqrt { n _ { 3 } + 1 } \Bigg ( \frac { z } { \sqrt { n _ { 2 } + 1 } } - \frac { n _ { 3 } } { n _ { 3 } + 1 } \Bigg ) \Bigg ) d z ,
$$

where $\phi$ and Φ denote the standard Gaussian PDF and CDF, respectively. We simulated 100, 000 interactions with horizon $T = 1 0 0$ for Thompson sampling over the two instances. For the instance $\nu ,$ we calculated the policy gap via the closed formula, and for $\nu ^ { \prime }$ we used high-order Gauss-Hermite quadrature to estimate the above integral. The results of this experiment are presented in Figure 2. We have empirically demonstrated that the

![](images/438b1e4060b718ad7467712a1940f29997aaebbf6e9cb1fa8d81af0519f86a40.jpg)  
Figure 2: The X-axis is number of rounds, the Y-axis is the estimated expected sample complexity with $\epsilon = 0 . 0 1 8$ up to round $t ,$ using 100, 000 simulations of the interaction of $\pi ^ { T S ( \mathcal { N } ) }$ with the environment. The 99% CI for the expected sample complexity at round 100 is [55.076538, 55.452762] for $\mathbb { E } _ { \nu } \left[ N _ { \epsilon } \big ( \pi ^ { T S ( \mathcal { N } ) } , 1 0 0 \big ) \right]$ , and [64.676313, 65.170247] for $\mathbb { E } _ { \nu ^ { \prime } } [ N _ { \epsilon } ( \pi ^ { T S ( \mathcal { N } ) } $ , 100

addition of more optimal arms strictly increases the expected sample complexity after enough rounds.

Our intuition for the example is that although the additional optimal arm directly competes with the suboptimal arm, it also splits the observations between the two optimal arms. Consequently, the individual posteriors of the optimal arms concentrate more slowly, and this loss of concentration can outweigh the benefit of the additional optimal arm.

This example seeks to demonstrate that a point-wise monotonicity reduction argument, similar to that used in previous works that analyze TS (Agrawal and Goyal, 2017; Merlis and Mannor, 2021), is unavailable in our analysis, but nevertheless we overcame this hindrance in Lemma 11 using a careful choice of event that is agnostic to optimal arms, both in their number and in the arbitrary choice of optimal representative for the analysis.