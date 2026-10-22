# Approximate Solomonof Induction: On the Robustness of Learned Universal Predictors

Qian Li Shenzhen Research Institute of Big Data liqian.ict@gmail.com

## Abstract

Shang-Hua Teng University of Southern California shanghua@usc.edu

Why large language models trained via log-loss minimization over massive and diverse datasets exhibit broad capabilities and generalizable skills is not yet fully understood. Grau-Moya et al. (ICML 2024) ofer a conceptual explanation through the lens of meta-learning: if sequence-generating environments � are drawn from a prior $w _ { \nu } ,$ minimizing population logloss recovers the corresponding Bayesian mixture predictor $\begin{array} { r } { \xi = \sum _ { \nu } w _ { \nu } \nu } \end{array}$ . In the idealized limit in which � is a universal prior, the optimal predictor corresponds to Solomonof universal induction. Practical machine learning algorithms, however, generally produce only approximately optimal predictors. This raises a natural question: Does the prediction performance remain robust under approximation? In this paper, we first provide a tight bound on the worst-case efect of approximation on prediction error. We show any predictor $\rho$ satisfying $D _ { K L } ( \xi \| \rho ) \le \eta$ retains bounded cumulative-expected-squared prediction error. However, while the Bayesian mixture predictor � has error �(log(1/� )) in an environment $\mu ,$ we establish a tight $\Theta ( \eta / w _ { \mu } )$ bound on the excess error of � relative to �. Thus, approximation can substantially degrade performance in environments assigned small prior probability. In sharp contrast, when environments are averaged according to prior �, we prove that the excess prediction error is only �(�). Furthermore, we establish a distribution-shift theorem for dominated shifts from a training prior � to a test prior $w ^ { \prime } ,$ interpolating our average-case and worst-case guarantees. Finally, we extend our robustness analysis to a finite-sampling setting relevant to learning universal predictors with neural networks and Transformers, where $D _ { K L } ( \xi \| \rho )$ could be unbounded. Our theory extends the classical theory of Bayesian and Solomonof Induction to approximate predictors and characterizes a broad spectrum of prediction behavior. It helps explain the sharp gap between strong empirical benchmark performance and poor worst-case performance in rare environments, while providing quantitative justification for fine tuning and synthetic data generation designed to protect important but underrepresented task families.

## 1 Introduction

Large language models (LLMs) trained for sequential prediction have acquired a remarkable range of capabilities from vast and diverse data. In recent years, scaling model capacity, expanding the volume and breadth of training data, increasing computational resources [KMH<sup>+</sup>20], and incorporating real-world human interaction [OWJ<sup>+</sup>22] have driven substantial progress, transforming LLMs into increasingly general-purpose systems for problem solving and induction. Given the substantial and rapidly growing resources required to train frontier models, understanding the benefits and limitations of further scaling has become more important than ever, raising a range of fundamental research questions.

Classical learning frameworks, such as semi-supervised learning, PAC learning, and reinforcement learning, typically analyze learning relative to a specified task, data distribution, and hypothesis class. In contrast, a single LLM trained on heterogeneous data can perform a wide range of tasks—including translation, writing, coding, and mathematical reasoning—without task-specific retraining, guided only by instructions or a few demonstrations provided in context. Although pretraining is largely driven by minimizing next-token log loss, the resulting models exhibit capabilities involving generation, abstraction, reasoning, and planning, often without explicit supervision for each capability. Moreover, once trained, such a model can be deployed at a comparatively low marginal cost per inference [Dee25]. LLM training thus distills enormous amounts of data and computation into a reusable neural network whose fixed parameters support adaptive behavior across contexts and tasks. This phenomenon suggests that learning to predict suficiently rich and heterogeneous sequences can give rise to broad knowledge and generalizable computational abilities. Precisely how such capabilities emerge from a seemingly simple prediction objective, however, remains incompletely understood, and is central to characterizing the fundamental power and limitations of the current LLM paradigm.

Motivating Prior Work. Meta-learning, or learning to learn, provides a conceptual framework for understanding this phenomenon [GGMW<sup>+</sup>26]. Meta-learning treats the learning procedure itself as an object of optimization, using experience accumulated across tasks to improve the learning of new tasks [HAMS21]. It involves two levels of learning, either explicitly or implicitly: an inner process learns a particular task from a specified dataset and objective, while an outer process optimizes aspects of the inner process–such as the model class, initialization, or optimization rule–with the aim of reducing the data or computation required to solve new tasks.

One approach is memory-based meta-learning (MBML) [SBB<sup>+</sup>16], in which a memory-augmented model–an LLM whose context serves as memory, for example–is trained on data drawn from a distribution over tasks using a single optimization loop. The two levels of learning are realized through diferent mechanisms: the outer process updates the model’s parameters across training tasks, whereas the inner process adapts to an individual task through changes in the model’s memory state. At test time, the parameters remain fixed, and adaptation proceeds through memory dynamics without task-specific parameter updates. In-context learning thus emerges as a learned adaptation procedure implemented through the model’s forward computation.

This MBML framework admits a precise Bayesian characterization [OWR<sup>+</sup>19, XRLM22]. For a fixed horizon $n ,$ suppose that each training sequence is generated by first sampling a task � from a prior distribution $w ,$ then sampling data $x _ { 1 : n }$ from �. Suppose a suficiently expressive model $\rho _ { \theta ^ { * } }$ is trained to minimize the population log-loss objective

$$
\mathcal { L } _ { n } ( \rho _ { \theta } ) = \mathbb { E } _ { \nu \sim w } \mathbb { E } _ { x \sim \nu } \left[ - \sum _ { t = 1 } ^ { n } \log _ { 2 } \rho _ { \theta } ( x _ { t } \mid x _ { < t } ) \right]
$$

Then $\rho \theta ^ { * }$ behaves as the Bayesian mixture predictor $\begin{array} { r } { \xi ( { \boldsymbol x } ) = \sum _ { \boldsymbol \nu } { \boldsymbol w } _ { \boldsymbol \nu } \cdot { \boldsymbol \nu } ( { \boldsymbol x } ) } \end{array}$ . In particular,

$$
\rho _ { \theta ^ { * } } ( x _ { t } \mid x _ { < t } ) = \xi ( x _ { t } \mid x _ { < t } ) = \sum _ { \nu } w _ { \nu } ( x _ { < t } ) \nu ( x _ { t } \mid x _ { < t } ) ,
$$

where $w _ { \nu } ( x _ { < t } ) \propto w _ { \nu } \cdot \nu ( x _ { < t } )$ is the posterior probability of �. Thus, conditioned on an observed prefix $x _ { < t } ,$ the Bayesian predictor combines the predictions of the candidate tasks according to their posterior probabilities, reflecting both their prior weights and their consistency with the observations. The appeal of the Bayesian perspective lies in its strong prediction guarantee [Hut05]: at test time, $\rho _ { \theta ^ { * } }$ can achieve eficient in-context adaptation to any task $\mu$ with positive prior weight $w _ { \mu } > 0 \colon$

$$
S _ { n } ( \mu , \rho _ { \theta ^ { * } } ) : = \sum _ { t = 1 } ^ { n } \mathbb { E } _ { x _ { < t } \sim \mu } \sum _ { x _ { t } } \left( \mu ( x _ { t } \mid x _ { < t } ) - \rho _ { \theta ^ { * } } ( x _ { t } \mid x _ { < t } ) \right) ^ { 2 } = O \left( \log { \frac { 1 } { w _ { \mu } } } \right) .
$$

This provides a principled explanation of in-context learning as amortized Bayesian prediction: training amortizes learning across tasks by encoding an adaptive inference procedure into the forward computation of a neural network. In particular, consider the class of all computable environments<sup>1</sup> endowed with the universal prior $w _ { \nu } = \Theta ( 2 ^ { - K ( \nu ) } )$ , where $K ( \nu )$ denotes the Kolmogorov complexity of $\nu .$ The resulting single Bayesian predictor becomes Solomonof universal induction [Sol78, Sol86], which achieves cumulative expected squared prediction error $O ( K ( \mu ) )$ simultaneously in every computable environment $\mu ,$ independently of the prediction horizon. Building on this connection, [GGH<sup>+</sup>24] proposed learning increasingly general predictors by meta-training neural networks on synthetic data generated by computationally bounded environments, represented by universal Turing machines with bounded program lengths and runtimes. Progressively relaxing these bounds while increasing model capacity and training data ofers an principled approach toward universal prediction, provided that the corresponding prediction objectives can be optimized suficiently well. The paper [GGMW<sup>+</sup>26] placed this framework within the broader context of algorithmic compression: minimizing sequential log loss can be understood as learning to compress observations by approximating and amortizing Bayesian prediction over their underlying generative mechanisms. Under a universal prior, the resulting predictor favors simpler computational explanations while averaging over all computable hypotheses according to both their simplicity and their consistency with the observations.

Our Main Contributions. The classical theory [Sol78, Sol86, Hut05] focuses on the exact predictor that attains the minimum population log loss. In practice, however, limited model capacity and imperfect optimization–even when training data are abundant–mean that learned models generally only approximate the ideal Bayesian predictor. As emphasized by [GGMW<sup>+</sup>26], extending the Bayesian view of meta-learning beyond exact realizability and perfect optimization remains an important open direction. Motivated by this line of work, we take a first step toward understanding the fundamental gap between ideal Bayesian prediction and practical learned predictors by addressing the following central question:

To what extent do the guarantees of the ideal Bayesian predictor continue to hold when practical learning architectures can only approximate it?

In this work, we develop a theory of Bayesian and universal sequential prediction under approximation, characterizing performance and robustness across a broad range of regimes:

Environment-Wise Worst-Case Robustness. We show in Theorem 1 that every approximate predictor $\rho$ with excess population log loss at most $\eta ,$ equivalently $D _ { \mathrm { K L } } ( \boldsymbol { \xi } \| \rho ) \leq \eta ,$ , retains a finite bound on its cumulative expected squared prediction error. Specifically, for any environment $\mu ,$ the error of the ideal Bayesian predictor $\xi$ is ${ \cal O } ( \log ( 1 / w _ { \mu } ) )$ ), whereas that of $\rho$ is bounded by �  log $1 / w _ { \mu } + \eta / w _ { \mu } )$ We further show in Theorem 3 that the approximation-dependent term is tight. In particular, for a universal mixture, an error of order $\eta 2 ^ { K ( \bar { \mu } ) }$ is attained for infinitely many computable environments $\mu .$ We also extend the analysis to two other fundamental measures of prediction error: cumulative KL divergence and cumulative squared Hellinger distance.

Typical Average-Case Robustness: In sharp contrast, we prove in Theorem 4 that, when performance is averaged over environments according to the prior �–the canonical average-case measure–the excess prediction error due to approximation is only $O ( \eta )$ . We show that an analogous average-case bound holds for cumulative KL divergence but not for cumulative squared Hellinger distance.

Distribution-Shift Robustness: Furthermore, we establish a distribution-shift theorem (Theorem 5) for dominated shifts from a training prior � to a test prior $w ^ { \prime } .$ , where $w _ { \nu } ^ { \prime } \leq C w _ { \nu }$ for every environment � and some $C \geq 1$ . The resulting error bound is �(log� + ��), thereby interpolating between our average-case and worst-case guarantees.

Finite-Data Robustness: Finally, in Section 5, we study predictors learned from finitely many training sequences. We show exponential upper-tail bounds for population KL error, yet its expectation can remain infinite when unseen events receive zero probability. Nevertheless, by using a weaker alternative measure of approximation, we extend our robustness analysis to this setting. In particular, we bound the expected average excess prediction error by the representation and optimization errors plus a finite-sample term that decays exponentially with the number of training sequences.

Taken together, these results provide a broad characterization of robustness to approximation in sequential prediction. They explain the sharp contrast between strong aggregate benchmark performance and poor worst-case performance on rare environments. This sharp distinction between worst-case and average-case behavior helps explain why approximation may have a much milder efect in practice. It may also shed light on the strong empirical performance of learned universal predictors [GGH<sup>+</sup>24, GGMW<sup>+</sup>26] and, more broadly, of LLMs across a wide range of tasks and benchmarks. Our distribution-shift theorem has several implications beyond basic transfer principle for sequential prediction. It provides a quantitative rationale for targeted sampling and synthetic data generation designed to protect important but underrepresented families of tasks. In particular, it provides a theoretical rationale for expecting that augmenting real-world datasets with Universal-Turing-Machine-generated (UTM-generated) data may improve prediction performance on practically important task families, as suggested by [GGH<sup>+</sup>24].

## 2 Notations and Preliminaries

We assume familiarity with the basic notions in probability theory and algorithmic information theory, including conditional probability and Turing machines. See, for example, [LV08] for background. Throughout this paper, log denotes the base-2 logarithm, while ln denotes the natural logarithm. Entropy and KL divergence are measured in bits.

Sequence measures and mixtures. Let X be a finite alphabet with $| X | \ge 2$ . Suppose $\mu$ is a probability measures on $\chi ^ { \infty }$ (representing a sequence generative model). We use $\mu ( x _ { 1 : n } )$ to denote the probability that a sequence sampled from $\mu$ begins with $x _ { 1 : n } ,$ and $\mu _ { 1 : n }$ to denote the induced distribution on $X ^ { n }$ . We denote $x _ { < t } : = x _ { 1 : t - 1 }$ . Let $M = \left\{ \nu _ { 1 } , \nu _ { 2 } , . . . \right\}$ be a countable class of measures. Let $\{ w _ { i } > 0 \}$ be weights satisfying $\begin{array} { r } { \sum _ { i } w _ { i } = 1 } \end{array}$ . Then, define the mixture $\begin{array} { r } { \xi : = \sum _ { i = 1 } { w _ { i } \nu _ { i } } . } \end{array}$ . When $\mu = \nu _ { i }$ is a selected measure from $M ,$ we write $w _ { \mu } : = w _ { i }$ . Note that for every finite string $x , \xi ( x ) \ge w _ { \mu } \mu ( x )$

The universal mixture. Solomonof [Sol78, Sol86] introduced a universal prior for sequence prediction. Unless stated otherwise, we fix a universal monotone Turing machine � throughout the paper. We say a measure $\nu$ is estimable if there exists a monotone Turing machine that takes an infinite sequence of independent, uniformly random bits and generates a sequence according to �. Let �(�) denote the length of the shortest program for � on this machine.

Let $M _ { U }$ denote the class consisting of all estimable measures. Set

$$
Z : = \sum _ { \nu \in \mathcal { M } _ { U } } 2 ^ { - K ( \nu ) } , \qquad w _ { \nu } : = \frac { 2 ^ { - K ( \nu ) } } { Z } , \qquad \xi _ { U } : = \sum _ { \nu \in \mathcal { M } _ { U } } w _ { \nu } \nu .
$$

The programs are described in a prefix-free fashion, so Kraft’s inequality gives $0 < Z \leq 1$ , and $w _ { \nu } = \Theta ( \bar { 2 } ^ { - K ( \nu ) } )$ . The fair-coin measure belongs to $M _ { U }$ , so $\xi _ { U } ( x ) > 0$ for every finite string �. We also use $K ( x )$ for the usual prefix-free Kolmogorov complexity of a finite string or integer �.

KL divergence. For two measures $\mu$ and $\rho$ and $n \geq 1$ , define

$$
D _ { \mathrm { K L } } ^ { n } ( \mu \| \rho ) : = \sum _ { x _ { 1 : n } \in X ^ { n } } \mu ( x _ { 1 : n } ) \log \frac { \mu ( x _ { 1 : n } ) } { \rho ( x _ { 1 : n } ) } , \qquad D _ { \mathrm { K L } } ( \mu \| \rho ) : = \operatorname* { l i m } _ { n \to \infty } D _ { \mathrm { K L } } ^ { n } ( \mu \| \rho ) .
$$

We use the convention that 0 log $\left( 0 / q \right) = 0$ and $p \log ( p / 0 ) = + \infty$ for $p > 0$ . The finite-horizon divergences are nondecreasing in $n ,$ so the supremum is also their limit. The chain rule gives that:

$$
D _ { \mathrm { K L } } ^ { n } ( \mu \| \rho ) = \sum _ { t = 1 } ^ { n } \mathbb { E } _ { x _ { < t } \sim \mu } \left[ D _ { \mathrm { K L } } ( \mu ( { \boldsymbol { \cdot } } \mid x _ { < t } ) \| \rho ( { \boldsymbol { \cdot } } \mid x _ { < t } ) ) \right] .\tag{1}
$$

Cumulative prediction error. As in [Sol78], we use cumulative squared prediction error as our primary criterion for evaluating the prediction performance of $\rho \colon$

$$
S _ { n } ( \mu , \rho ) : = \sum _ { t = 1 } ^ { n } \mathbb { E } _ { x _ { < t } \sim \mu } \sum _ { a \in \mathcal { X } } \left( \mu ( a \mid x _ { < t } ) - \rho ( a \mid x _ { < t } ) \right) ^ { 2 } , \qquad S ( \mu , \rho ) : = \operatorname* { l i m } _ { n \to \infty } S _ { n } ( \mu , \rho ) .
$$

To examine the scope of our guarantees across diferent error criteria, we also consider cumulative squared Hellinger error:

$$
\mathrm { H e l } _ { n } ( \mu , \rho ) : = \sum _ { t = 1 } ^ { n } \mathbb { E } _ { \boldsymbol { x } _ { < t } \sim \mu } \sum _ { a \in X } \left( \sqrt { \mu ( a \mid \boldsymbol { x } _ { < t } ) } - \sqrt { \rho ( a \mid \boldsymbol { x } _ { < t } ) } \right) ^ { 2 } , \qquad \mathrm { H e l } ( \mu , \rho ) : = \operatorname* { l i m } _ { n \to \infty } \mathrm { H e l } _ { n } ( \mu , \rho ) .
$$

and cumulative KL error:

$$
D _ { n } ( \mu , \rho ) : = \sum _ { t = 1 } ^ { n } \mathbb E _ { x _ { < t } \sim \mu } \left[ D _ { \mathrm { K L } } ( \mu ( \cdot \mid x _ { < t } ) \| \rho ( \cdot \mid x _ { < t } ) ) \right] = D _ { \mathrm { K L } } ^ { n } ( \mu \| \rho ) , \qquad D ( \mu , \rho ) : = \operatorname* { l i m } _ { n \to \infty } D _ { n } ( \mu , \rho ) .
$$

The following inequalities relate these error criteria.

Lemma 1 (Theorem 3.2.3 in [HQC24]). Let $\mu$ and $\rho$ be probability measures on $\chi ^ { \infty }$ . For every $n \geq 1$

$$
S _ { n } ( \mu , \rho ) \leq ( \ln 2 ) D _ { n } ( \mu \| \rho ) , \qquad \mathrm { H e l } _ { n } ( \mu , \rho ) \leq ( \ln 2 ) D _ { n } ( \mu \| \rho ) , \qquad S _ { n } ( \mu , \rho ) \leq 2 \mathrm { H e l } _ { n } ( \mu , \rho ) .
$$

The same inequalities hold for the corresponding infinite-horizon errors �, Hel, and $D .$

## 3 Robustness under Approximation: Task-wise Guarantee

In this section, we establish a tight task-wise prediction guarantee for a trained model $\rho$ that approximates the Bayesian predictor � with KL gap $D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta$ . The KL divergence provides the most natural measure of approximation quality because the population log-loss of $\rho$ at horizon � admits the following decomposition, where � denotes the Shannon entropy:

$$
\mathcal { L } _ { n } ( \rho ) : = \mathbb { E } _ { x _ { 1 : n } \sim \xi } \left[ - \log \rho ( x _ { 1 : n } ) \right] = H ( \xi _ { 1 : n } ) + D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho ) ,
$$

Moreover, $\xi$ minimizes the population loss over all probability measures on $\chi ^ { \infty }$ , attaining the minimum value $H ( \xi _ { 1 : n } )$ . Thus, $D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta \implies D _ { \mathrm { K I . } } ^ { n } ( \xi \| \rho ) \le \eta \iff \mathcal { L } _ { n } ( \rho ) \le \operatorname* { m i n } _ { \pi } \mathcal { L } _ { n } ( \pi ) + \eta$

## 3.1 Environment-Wise Guarantee: Upper Bound

We first extend the classical analysis of [Hut05] by showing that, under a finite KL-divergence, the cumulative squared prediction error remains finite for every environment in M.

Theorem 1 (Environment-wise upper bound). $\begin{array} { r } { f D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta . } \end{array}$ , then for every $\mu \in { \cal M }$

$$
S ( \mu , \rho ) \leq \frac { e \ln 2 } { e - 1 } \left( \log \frac { 1 } { w _ { \mu } } + \frac { \eta } { w _ { \mu } } \right) .
$$

The proof is given in Appendix ${ \mathrm { A } } ,$ which bounds the cumulative KL error for every � by

$$
D _ { n } ( \mu \| \rho ) \leq \frac { e } { e - 1 } \left( \log \frac { 1 } { w _ { \mu } } + \frac { \eta } { w _ { \mu } } \right) .
$$

Theorem 1 then follows from Lemma 1 that $S _ { n } ( \mu , \rho ) \leq ( \ln 2 ) D _ { n } ( \mu \| \rho )$ and ${ \mathrm { H e l } } _ { n } ( \mu , \rho ) \leq ( \ln 2 ) D _ { n } ( \mu \| \rho )$ We thus also prove that, up to constant factors, this upper bound extends the cumulative squared Hellinger distance $\mathrm { H e l } ( \mu , \rho )$

Specializing to the universal mixture $\xi _ { U }$ and using the relation $w _ { \boldsymbol { \mu } } = \Theta ( 2 ^ { - K ( \boldsymbol { \mu } ) } )$ , we obtain the following environment-wise prediction-error bound in terms of the Kolmogorov complexity $K ( \mu )$

Theorem 2 (Upper bound for universal prediction). $I f D _ { \mathrm { K L } } ( \xi _ { U } \| \rho ) \leq \eta .$ , then for every $\mu \in M _ { U }$

$$
S ( \mu , \rho ) = O \left( K ( \mu ) + \eta \cdot 2 ^ { K ( \mu ) } \right) ,
$$

where the implicit constant depends only on thefixed universal Turing machine.

## 3.2 Environment-Wise Guarantee: Tight Worst-Case Lower Bound

Our next theorem shows that the approximation-dependent term $\eta / w _ { \mu }$ in Theorem 1 is unavoidable and tight up to a constant factor. Remarkably, for the universal mixture, this tightness holds for infinitely many environments in $M _ { U }$ . Throughout this subsection, we restrict attention to the binary alphabet $X = \{ 0 , 1 \}$

Theorem 3 (Tight lower bound for universal prediction). For every $\eta > 0$ , there are infinitely many estimable measures $\mu$ such that

$$
\operatorname* { s u p } _ { \rho \colon D _ { \operatorname { K L } } ( \xi _ { U } \| \rho ) \leq \eta } S ( \mu , \rho ) \geq \Omega \left( K ( \mu ) + \eta 2 ^ { K ( \mu ) } \right) ,
$$

where the implicit constant depends only on the fixed universal Turing machine.

Thus, even an arbitrarily small KL-approximation gap can coexist with a cumulative squared prediction error that grows exponentially with the Kolmogorov complexity of the environment.

We now describe the high-level idea behind the proof. The complete proof is given in $\mathsf { A p - }$ pendix C and formalizes the following rare-prefix construction.

Rare-prefix construction. The lower bound exploits prefixes that are unlikely under the Bayesian mixture $\xi ,$ but occur with probability one under a particular environment. For $z \in \{ 0 , 1 \} ^ { \infty } ,$ , let $\delta _ { z }$ denote the deterministic environment that outputs �. Suppose $\mu = \delta _ { z }$ and that a finite prefix $y$ of $z$ has mixture probability $q : = \xi ( y )$ . We construct $\rho$ by replacing the next $N$ conditional predictions along � after the prefix � with the uniform distribution, while leaving all other predictions unchanged.

Each modified history has probability at most $q$ under $\xi ,$ and replacing its next-symbol distribution with the uniform distribution incurs at most one bit of conditional KL divergence. Hence, the total KL cost is at most $q N$ . Under $\mu ,$ however, every modified history is visited with probability one, and each modified prediction contributes exactly $1 / 2$ to the squared prediction error. Taking $N = \lfloor \eta / q \rfloor$ yields the following lemma.

Lemma 2 (Lower bound from a rare prefix). Let $\mu = \delta _ { z } \in { \mathcal { M } }$ , and let $y \prec z$ be a finite prefix $o f z$ and $q : = \xi ( y )$ . For every $\eta > 0$ , there exists a probability measure $\rho$ with $D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta$ such that

$$
S ( \mu , \rho ) \geq S _ { | y | } ( \mu , \xi ) + \frac { 1 } { 2 } \left\lfloor \frac { \eta } { q } \right\rfloor .
$$

When $q = O ( w _ { \boldsymbol { \mu } } )$ , it yields an $\Omega ( \eta / w _ { \mu } )$ lower bound whenever $\eta / w _ { \mu }$ is suficiently large. The proof of Theorem 3 identifies infinitely many deterministic environments having prefixes that satisfy this condition. Since $w _ { \boldsymbol { \mu } } = \Theta ( 2 ^ { - K ( \mu ) } )$ , the resulting approximation penalty is $\bar { \Omega ( \eta 2 ^ { K ( \mu ) } ) }$ ).

Finally, we can show that Lemma 2 also extends to cumulative KL and squared Hellinger errors. Then by Lemma 1, we have, for every $n \ge 1 , D _ { n } ( \mu \| \rho ) \ge S _ { n } ( \mu , \rho ) / \mathrm { l n } 2$ and ${ \mathrm { H e l } } _ { n } ( \mu , \rho ) \geq S _ { n } ( \mu , \rho ) / 2$ Thus, our construction can establish similar lower bounds as in Theorem $^ 3$ for cumulative KL and squared Hellinger errors.

## 4 Robustness under Approximation: Average-Case Analysis

In this section, we show that the environment-wise prediction guarantee under approximation can be sharply improved when performance is averaged across environments. We then establish a distribution-shift theorem for moving from the training prior $\boldsymbol { w }$ to a dominated test prior $w ^ { \prime }$ . The result interpolates between our average-case and environment-wise guarantees. Before stating the theorems and discussing their implications, we explain the motivation for developing an average-case theory of robustness under approximation. Section 3.2 shows that the approximationdependent term $\eta / w _ { \mu }$ in Theorem 1 is unavoidable in the worst case. For an environment with a small prior weight $w _ { \mu } ,$ this term can overwhelm the classical $O ( \log ( 1 / w _ { \mu } ) )$ prediction guarantee for the ideal Bayesian predictor.

Such pessimistic worst-case guarantees, however, stand in contrast to the strong empirical performance of learned universal predictors [GGH<sup>+</sup>24, GGMW<sup>+</sup>26] and, more broadly, to the outstanding performance of LLMs across a wide range of tasks and benchmarks. This contrast raises the question of why approximation has a milder efect in practice. Empirical evaluations typically aggregate performance across many tasks or environments, whereas an environment-wise guarantee treats each environment individually. Accordingly, we first study prediction performance averaged over the same prior � that defines the training mixture $\xi .$ We show that, in sharp contrast to the worst-case setting, approximation has a substantially milder efect under this average-case criterion. Our distribution-shift theorem can then be used to extend this guarantee to other test distributions.

## 4.1 Sharp Contrast between Typical and Task-Wise Prediction Guarantees

Theorem 4 (Approximation robustness: average-case upper bound). $\begin{array} { r } { I f D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta , } \end{array}$ , then

$$
{  { \mathbb E } } _ { \mu \sim w } \left[ S _ { n } ( \mu , \rho ) - S _ { n } ( \mu , \xi ) \right] \le ( \ln 2 ) \eta .
$$

This theorem is a consequence of the following exact average-case decomposition:

Proposition 1 (Exact average-case decomposition). For every predictor $\rho$ and every $n \geq 1$

$$
\mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \rho ) = \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \xi ) + S _ { n } ( \xi , \rho ) .
$$

Theorem 4 matches the exact prediction-error baseline of the ideal mixture $\xi \colon$ the average cost of replacing $\xi$ with $\rho$ is $S _ { n } ( \xi , \rho )$ . Together with Theorem $^ { 3 , }$ this result shows that approximation can preserve aggregate performance while substantially degrading performance on individual tasks.

Remark 1. Note that Theorem 4 cannot be obtained simply by averaging the bound in Theorem 1. Instead, the proof exploits the mixture structure to identify the average excess error exactly. To see this, consider the case when M is infinite, the following average diverges:

$$
\mathbb { E } _ { \mu \sim w } \frac { 1 } { w _ { \mu } } = \sum _ { \mu \in \mathcal { M } } w _ { \mu } \frac { 1 } { w _ { \mu } } = \sum _ { \mu \in \mathcal { M } } 1 = + \infty .
$$

A fundamental diferentiation: The role of approximation in Theorem 4 also highlights an interesting distinction between exact Bayesian prediction and its approximate counterpart. The classical theory applies to cumulative squared prediction error, KL error, and Hellinger error. As we show in Section 3.1, the task-wise guarantees under approximation extend to all three error criteria. By contrast, the average-case decomposition in Proposition 1 holds only for cumulative squared prediction error and cumulative KL error. For KL error, the chain rule immediately yields the analogous decomposition. $\mathbb { E } _ { \mu \sim w } D _ { n } ( \mu \| \rho ) = \mathbb { E } _ { \mu \sim w } D _ { n } ( \mu \| \xi ) + D _ { n } ( \xi \| \rho )$ . Hence, if $D _ { \mathrm { K L } } ( \boldsymbol { \xi } \| \rho ) \leq \eta ,$ , then $\mathbb { E } _ { \mu \sim w } D _ { n } ( \mu \| \rho ) \le \mathbb { E } _ { \mu \sim w } D _ { n } ( \mu \| \xi ) + \eta$ . For cumulative squared Hellinger error, neither the analogous decomposition nor a uniform additive $O ( \eta )$ excess-error bound relative to $\xi$ holds in general. The mixture predictor � need not minimize average squared Hellinger error; a counterexample is given in Appendix E.

Remark 2. Prior work has shown that the average gains ofknowledge distillation can coexist with degraded performance on underrepresented subgroups and tail classes [LBMK22, WNZ<sup>+</sup>23]. By viewing $\xi$ as the teacher model and $\rho$ as the student model, our results provide a complementary explanation in Bayesian sequential prediction: a small population KL distillation loss controls the student’s average excess prediction error, but can coexist with substantial degradation relative to the teacher in rare environments.

## 4.2 Approximation Robustness under Distribution Shift

The preceding analysis evaluates performance under the training prior �. We now consider a diferent test prior $w ^ { \prime }$ . Our next theorem shows that the average-case guarantee remains robust when the test prior is dominated by the training prior, in the sense that $w _ { \nu } ^ { \prime } \leq C w _ { \nu }$ for every environment � and some $C \geq 1$ . This result has several implications beyond establishing a basic transfer principle for sequential prediction and interpolating between the worst-case and averagecase analyses of robustness under approximation. One potentially significant implication is that the theory provides basic support for the argument that augmenting real-world datasets with Universal-Turing-Machine-generated (UTM-generated) data can improve prediction performance on practically important families of tasks, as suggested by [GGH<sup>+</sup>24].

Theorem 5 (Approximation robustness under distribution shift). Let $w ^ { \prime }$ be a probability distribution on M satisfying $w _ { \nu } ^ { \prime } \leq C w _ { \nu }$ for every $\nu \in \mathcal { M }$ and some $C \geq 1$ . Define the corresponding mixture

$$
\xi ^ { \prime } : = \sum _ { \nu \in \mathcal { M } } w _ { \nu } ^ { \prime } \nu .
$$

$I f D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta .$ , then, for every $n \geq 1$

$$
\begin{array} { r } { \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \rho ) \leq \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) + ( \ln 2 ) \left( \sqrt { \log C } + \sqrt { C \eta } \right) ^ { 2 } } \\ { \leq \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ) + ( \ln 2 ) \left( \sqrt { \log C } + \sqrt { C \eta } \right) ^ { 2 } . } \end{array}\tag{2}
$$

In particular, the average excess error relative to $\xi ^ { \prime }$ is $O ( \log C + C \eta )$

Cumulative KL error satisfies a distribution-shift bound of the same order. Indeed, the KL decomposition and the domination $\xi ^ { \prime } \leq C \xi ,$ , together with the KL transfer inequality (Lemma 3) used in the proof of Theorem 1, give

$$
\mathbb { E } _ { \mu \sim w ^ { \prime } } D _ { n } ( \mu \| \rho ) \leq \mathbb { E } _ { \mu \sim w ^ { \prime } } D _ { n } ( \mu \| \xi ^ { \prime } ) + \frac { e } { e - 1 } \left( \log C + C \eta \right) .
$$

Implication I: Rare task families and targeted sampling. Theorem $5$ interpolates between average-case and environment-wise guarantees. Taking $ { w ^ { \prime } } =  { w }$ and $C = 1$ recovers the average-case upper bound in Theorem 4. At the other extreme, if $w ^ { \prime }$ is concentrated on a single environment $\mu ,$ we may take $C = 1 / w _ { \mu }$ . Then $\xi ^ { \prime } = \mu ,$ and

$$
S _ { n } ( \mu , \rho ) \leq ( \ln 2 ) \left( \sqrt { \log \frac { 1 } { w _ { \mu } } } + \sqrt { \frac { \eta } { w _ { \mu } } } \right) ^ { 2 } = O \left( \log \frac { 1 } { w _ { \mu } } + \frac { \eta } { w _ { \mu } } \right) .
$$

Thus, the theorem also recovers the order of the environment-wise bound in Theorem 1.

More generally, let $A \subseteq M$ be a family of environments with prior mass $w ( A ) > 0$ , and consider evaluation under $w ^ { \prime } = w ( \cdot \mid A )$ . Since $w _ { \nu } ^ { \prime } \leq w _ { \nu } / w ( A )$ , we may take $C = 1 / w ( A )$ and obtain

$$
{ \mathbb E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \rho ) \le { \mathbb E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) + O \left( \log \frac { 1 } { w ( A ) } + \frac { \eta } { w ( A ) } \right) .\tag{3}
$$

For this conditional test distribution, the guarantee depends on the total prior mass of the task family.

This suggests a way to improve coverage of an important but underrepresented family. Let $q$ be a probability distribution supported on that family, and define

$$
\widetilde w : = ( 1 - \lambda ) w + \lambda q , \qquad \xi _ { q } : = \sum _ { \nu \in M } q _ { \nu } \nu , \qquad \widetilde \xi : = \sum _ { \nu \in M } \widetilde w _ { \nu } \nu = ( 1 - \lambda ) \xi + \lambda \xi _ { q } ,
$$

where $\lambda \in ( 0 , 1 )$ is the fraction of training mass allocated to $q .$ Since $q _ { \nu } \leq \widetilde { w } _ { \nu } / \lambda ,$ any predictor with $D _ { \mathrm { K L } } ( \widetilde { \xi } \| \rho ) \le \eta$ satisfies

$$
\mathbb { E } _ { \mu \sim q } S _ { n } ( \mu , \rho ) \le \mathbb { E } _ { \mu \sim q } S _ { n } ( \mu , \xi _ { q } ) + O \left( \log \frac { 1 } { \lambda } + \frac { \eta } { \lambda } \right) .
$$

The approximation guarantee for this family therefore depends on its assigned training mass $\lambda ,$ independently of its rarity under the original prior �.

Within this model, the bound provides a quantitative rationale for increasing the training probability assigned to important but underrepresented task families, for example through targeted sampling or synthetic data generation. In particular, it supports augmenting UTM-generated training data with real-world datasets to improve prediction performance on practically important task families, as suggested by [GGH<sup>+</sup>24].

Implication II: Fine-tuning with replay. Replaying pretraining data during adaptation can mitigate forgetting $\left[ \mathrm { B G B } ^ { + } 2 5 , \mathrm { I T G } ^ { + } 2 4 \right]$ . Theorem 5 can be used to give a population-level analysis of this strategy. Interpret � as the target-task distribution and � as the original pretraining distribution. During fine-tuning, draw a fraction � of sequences from $\xi _ { q }$ and a fraction $1 - \lambda$ from $\xi .$ The resulting population training distribution is $\widetilde { \xi } = ( 1 - \lambda ) \xi + \lambda \xi _ { q }$

Suppose the fine-tuned predictor $\rho _ { \mathrm { f t } }$ approximately minimizes the population log-loss, in that

$$
D _ { \mathrm { K L } } ( \widetilde { \xi } \Vert \rho _ { \mathrm { f t } } ) \leq \eta .
$$

Both task distributions are dominated by the fine-tuning prior: $\begin{array} { r } { q _ { \nu } \ \leq \ \frac { \widetilde { w } _ { \nu } } { \lambda } , \ w _ { \nu } \ \leq \ \frac { \widetilde { w } _ { \nu } } { 1 - \lambda } } \end{array}$ . Applying Theorem 5 to each distribution yields the simultaneous guarantees

$$
\begin{array} { r l } & { \mathbb { E } _ { \mu \sim q } S _ { n } ( \mu , \rho _ { \mathrm { f t } } ) \le \mathbb { E } _ { \mu \sim q } S _ { n } ( \mu , \xi _ { q } ) + O \left( \log \frac { 1 } { \lambda } + \frac { \eta } { \lambda } \right) , } \\ & { \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \rho _ { \mathrm { f t } } ) \le \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \xi ) + O \left( \log \frac { 1 } { 1 - \lambda } + \frac { \eta } { 1 - \lambda } \right) . } \end{array}
$$

The first bound controls prediction error on the target distribution, while the second controls error on the original distribution. Since $\xi$ minimizes average squared prediction error under $w ,$ the second bound also limits the possible increase in average error relative to the predictor before fine-tuning.

For fixed approximation accuracy �, increasing � tightens the target-distribution bound while weakening the original-distribution bound. The mixing parameter � therefore controls the tradeof between the performance guarantees on the target and original task distributions. Finally, we remark that these guarantees assume that the fine-tuned predictor achieves the stated population approximation accuracy �.

Implication III: Choice of reference universal Turing machine. Let � and $U ^ { \prime }$ be two universal Turing machines. By the invariance theorem, there is a constant $c _ { U , U ^ { \prime } } \geq 0$ such that

$$
\left| K _ { U } ( \nu ) - K _ { U ^ { \prime } } ( \nu ) \right| \le c _ { U , U ^ { \prime } } \qquad \mathrm { f o r ~ e v e r y ~ e s t i m a b l e ~ } \nu .
$$

Consider the universal priors $w _ { \nu } = 2 ^ { - K _ { U } ( \nu ) } / Z _ { U } , ~ w _ { \nu } ^ { \prime } = 2 ^ { - K _ { U ^ { \prime } } ( \nu ) } / Z _ { U ^ { \prime } }$ , where $\begin{array} { r } { Z _ { V } : = \sum _ { \nu \in \mathcal { M } } 2 ^ { - K _ { V } ( \nu ) } } \end{array}$ Then we have $w _ { \nu } ^ { \prime } / w _ { \nu } = 2 ^ { K _ { U } ( \nu ) - K _ { U ^ { \prime } } ( \nu ) } \cdot Z _ { U } / Z _ { U ^ { \prime } } \le 4 ^ { c _ { U , U ^ { \prime } } } . \mathrm { { I f ~ t r a i n i n g } }$ uses the mixture $\xi$ induced by � and $D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta .$ , then test under the prior induced by $U ^ { \prime }$ satisfies

$$
\mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \rho ) \le \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) + O \big ( c _ { U , U ^ { \prime } } + 4 ^ { c _ { U , U ^ { \prime } } } \eta \big ) .
$$

Thus, changing the reference machine preserves the form of the average-case guarantee, with constants depending only on the pair of machines, independently of the environment and prediction horizon. These constants may nevertheless be large because their dependence on the compiler overhead is exponential.

## 5 Robustness of Learning and Prediction with Finite Data

We now examine whether our prediction guarantees remain robust when the predictor is learned from finitely many samples. Fix a prediction horizon �, and let $\rho _ { m }$ be learned from � independent training sequences $\mathscr { D } _ { m } \sim ( \xi _ { 1 : n } ) ^ { \otimes m }$ by minimizing the empirical log-loss over the hypothesis class H up to an additive optimization error $\eta _ { \mathrm { o p t } } \geq 0$ . In addition to this optimization error, the model class incurs a representation error $\begin{array} { r } { \eta _ { \mathrm { r e p } } : = \operatorname* { m i n } _ { \rho \in { \mathcal { H } } } D _ { \mathrm { K I . } } ^ { n } ( { \xi } | | \rho ) } \end{array}$ . We consider $\eta _ { \mathrm { r e p } } + \eta _ { \mathrm { o p t } } > 0$

We show an exponential upper-tail bound on the population KL error: for every $\gamma > 0$ , there exist constants $c _ { \gamma } , C _ { \gamma } > 0$ such that<sup>2</sup>, for all $m \geq 1$

$$
\operatorname* { P r } _ { \mathcal { D } _ { m } } \left\{ D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho _ { m } ) > \eta _ { \mathrm { r e p } } + \eta _ { \mathrm { o p t } } + \gamma \right\} \le C _ { \gamma } e ^ { - c _ { \gamma } m } .\tag{4}
$$

Nevertheless, this high-probability guarantee need not imply a finite expected KL error. A learned predictor may assign zero probability to an event that has positive population probability but is absent from the training data. Precisely, it is possible that

$$
\mathbb { E } _ { \mathcal { D } _ { m } } D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho _ { m } ) = + \infty \qquad \mathrm { f o r ~ e v e r y ~ f i n i t e } \ m .\tag{5}
$$

Despite this dificulty, we show that our squared prediction error guarantees remain meaningful in expectation. It follows from two observations. First, if we characterize the approximation quality with a weaker condition $S _ { n } ( \xi , \rho _ { m } ) \leq \eta$ instead of $D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho _ { m } ) \leq \eta .$ , then there exist constants � and � such that for any $m \geq 1$

$$
\begin{array} { r } { \mathbb { E } _ { \mathcal { D } _ { m } } S _ { n } ( \xi , \rho _ { m } ) \le \eta _ { \mathrm { r e p } } + \eta _ { \mathrm { o p t } } + C \cdot n e ^ { - c \cdot m } , } \end{array}\tag{6}
$$

which means that the finite-sample contribution to the approximation error decays exponentially with the training sample size. Moreover, one can check that all of our theorems extend from the KL approximation condition to the weaker condition $S _ { n } ( \xi , \rho ) \leq \eta ,$ , with corresponding changes to the constants. In particular, Theorem 4 and (6) together imply: for any $m \geq 1$ 1

$$
\begin{array} { r } { \mathbb { E } _ { \mathcal { D } _ { m } } \big [ \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \rho _ { m } ) - \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \xi ) \big ] \le \eta _ { \mathrm { r e p } } + \eta _ { \mathrm { o p t } } + C \cdot n e ^ { - c \cdot m } . } \end{array}\tag{7}
$$

## 6 Conclusion

We characterize how approximation afects Bayesian prediction and Solomonof induction. Our results show that approximation can preserve average performance while substantially degrading performance on individual tasks. The analysis also extends to dominated shifts from the training prior to the test prior and is robust in the finite-sample setting.

An important direction for future work is to develop mitigation strategies that provide stronger guarantees across all tasks simultaneously. We explore initial approaches in Appendices H and I.

Another open direction is length generalization. Our finite-sample analysis considers training and test sequences of the same length; extending these guarantees to longer test sequences remains a challenge. One possibility is to combine pretrained predictors with continual learning, using minimum description length principle [Grü07, PH05] to guide adaptation as new observations become available.

## Acknowledgments

We thank Xinyu Mao and Guangxu Yang for early discussions, and Marcus Hutter for his comments. We used AI tools for polishing the language, searching the literature, and developing and drafting all proofs. We have checked and revised all proofs and take full responsibility for the whole content of this work.

## References

[BGB<sup>+</sup>25] Louis Béthune, David Grangier, Dan Busbridge, Eleonora Gualdoni, Marco Cuturi, and Pierre Ablin. Scaling laws for forgetting during finetuning with pretraining data injection. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien,

Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Forty second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. 10

[Dee25] DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models, 2025. 2

[GGH<sup>+</sup>24] Jordi Grau-Moya, Tim Genewein, Marcus Hutter, Laurent Orseau, Grégoire Delé- tang, Elliot Catt, Anian Ruoss, Li Kevin Wenliang, Christopher Mattern, Matthew Aitchison, and Joel Veness. Learning universal predictors. In International Conference on Machine Learning, 2024. 3, 4, 8, 9, 10

[GGMW<sup>+</sup>26] Tim Genewein, Jordi Grau-Moya, Li Kevin Wenliang, Laurent Orseau, and Marcus Hutter. Algorithmic compression via pretrained neural networks. Entropy, 28(6):596, 2026. 2, 3, 4, 8

[Grü07] Peter D Grünwald. The minimum description length principle. The MIT press, 2007. 12

[HAMS21] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Metalearning in neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(9):5149–5169, 2021. 2

[HQC24] Marcus Hutter, David Quarel, and Elliot Catt. An introduction to universal artificial intelligence. CRC Press, 2024. 6

[Hut05] Marcus Hutter. Universal artificial intelligence: Sequential decisions based on algorithmic probability. Springer Science & Business Media, 2005. 3, 6, 18

[ITG<sup>+</sup>24] Adam Ibrahim, Benjamin Thérien, Kshitĳ Gupta, Mats L. Richter, Quentin Gregory Anthony, Eugene Belilovsky, Timothée Lesort, and Irina Rish. Simple and scalable strategies to continually pre-train large language models. Trans. Mach. Learn. Res., 2024, 2024. 10

[KMH<sup>+</sup>20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020. 1

[LBMK22] Michal Lukasik, Srinadh Bhojanapalli, Aditya Krishna Menon, and Sanjiv Kumar. Teacher’s pet: understanding and mitigating biases in distillation. Trans. Mach. Learn. Res., 2022, 2022. 9

[LV08] Ming Li and Paul Vitányi. An Introduction to Kolmogorov Complexity and Its Applications. Texts in Computer Science. Springer, New York, NY, 3 edition, 2008. 4, 18

[OWJ<sup>+</sup>22] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022. 1

[OWR<sup>+</sup>19] Pedro A. Ortega, Jane X. Wang, Mark Rowland, Tim Genewein, Zeb Kurth-Nelson, Razvan Pascanu, Nicolas Heess, Joel Veness, Alexander Pritzel, Pablo Sprechmann, Siddhant M. Jayakumar, Tom McGrath, Kevin J. Miller, Mohammad Gheshlaghi Azar, Ian Osband, Neil C. Rabinowitz, András György, Silvia Chiappa, Simon Osindero, Yee Whye Teh, Hado van Hasselt, Nando de Freitas, Matthew M. Botvinick, and Shane Legg. Meta-learning of sequential strategies. CoRR, abs/1905.03030, 2019. 2

[PH05] Jan Poland and Marcus Hutter. Asymptotics of discrete mdl for online prediction. IEEE Transactions on Information Theory, 51(11):3780–3795, 2005. 12

[SBB<sup>+</sup>16] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, page 1842–1850. JMLR.org, 2016. 2

[Sol78] Ray Solomonof. Complexity-based induction systems: comparisons and convergence theorems. IEEE Transactions on Information Theory, 24(4):422–432, 1978. 3, 5

[Sol86] Ray Solomonof. The application of algorithmic probability to problems in artificial intelligence. In Machine Intelligence and Pattern Recognition, volume 4, pages 473–491. 1986. 3, 5

[WNZ<sup>+</sup>23] Serena Wang, Harikrishna Narasimhan, Yichen Zhou, Sara Hooker, Michal Lukasik, and Aditya Krishna Menon. Robust distillation for worst-class performance: on the interplay between teacher and student objectives. In Robin J. Evans and Ilya Shpitser, editors, Uncertainty in Artificial Intelligence, UAI 2023, July 31 - 4 August 2023, Pittsburgh, PA, USA, volume 216 of Proceedings of Machine Learning Research, pages 2237–2247. PMLR, 2023. 9

[XRLM22] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 2

## A Proof of Theorem 1

Lemma 3 (KL transfer under domination). Let $P , Q , R$ be probability distributions on the same finite space Ω. Suppose that $Q \geq w \cdot P$ for some $w \in ( 0 , 1 ]$ . Then

$$
D _ { \mathrm { K L } } ( P \| R ) \leq \frac { e } { e - 1 } \left( \log \frac { 1 } { w } + \frac { 1 } { w } \cdot D _ { \mathrm { K L } } ( Q \| R ) \right) .
$$

Proof. The claim is immediate when $w = 1$ , because then $Q \ = \ P .$ . We may therefore assume $0 < w < 1$ . The conclusion is also immediate if $D _ { \mathrm { K L } } ( Q \Vert R ) = \infty$

Write $Q = w P + ( 1 - w ) V$ for a probability distribution $V ,$ and define

$$
\alpha ( x ) : = \frac { w P ( x ) } { Q ( x ) }
$$

on the support of $Q ,$ with $\alpha ( x ) = 0$ elsewhere. Then $0 \leq \alpha \leq 1$ and $\mathbb { E } _ { Q } \alpha = w$ . Introduce an auxiliary binary variable $Z .$ . Under ${ \widetilde { Q } } ,$ let $X \sim Q$ and

$$
Z \mid X = x \sim \operatorname { B e r n } ( \alpha ( x ) ) .
$$

Thus ${ \widetilde Q } ( X \in \cdot \mid Z = 1 ) = P$ . Define $\widetilde { R }$ using the same conditional law of $Z$ given $X ,$ but with $X \sim R ,$ and put

$$
r : = \mathbb { E } _ { R } \alpha = \operatorname* { P r } _ { \tilde { R } } [ Z = 1 ] .
$$

Finiteness of $D _ { \mathrm { K L } } ( Q \| R )$ implies $0 < r < 1$ . Let $R _ { 1 }$ denote the conditional law of � under $\widetilde { R }$ given $Z = 1$ , and define $R _ { 0 }$ analogously. One can check that

$$
R _ { 1 } ( x ) = \frac { \alpha ( x ) } { r } \cdot R ( x ) , \quad R _ { 0 } ( x ) = \frac { 1 - \alpha ( x ) } { 1 - r } \cdot R ( x ) , \quad \mathrm { a n d } \ R = r R _ { 1 } + ( 1 - r ) R _ { 0 } .
$$

The conditional kernel from � to � is the same under the two joint distributions. Hence

$$
D _ { \mathrm { K L } } ( \widetilde { Q } \| \widetilde { R } ) = D _ { \mathrm { K L } } ( Q \| R ) .
$$

Applying the chain rule in the other direction gives

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( Q | | R ) = d _ { \mathrm { B } } ( w | | r ) + w D _ { \mathrm { K L } } ( P | | R _ { 1 } ) + ( 1 - w ) D _ { \mathrm { K L } } ( V | | R _ { 0 } ) } \\ & { \qquad \geq d _ { \mathrm { B } } ( w | | r ) + w D _ { \mathrm { K L } } ( P | | R _ { 1 } ) , } \end{array}\tag{8}
$$

where

$$
d _ { \mathrm { B } } ( u \| v ) : = u \log \frac { u } { v } + ( 1 - u ) \log \frac { 1 - u } { 1 - v }
$$

is binary KL divergence. Moreover,

$$
\frac { R _ { 1 } ( x ) } { R ( x ) } = \frac { \alpha ( x ) } { r } ,
$$

and therefore

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( P \Vert R ) = D _ { \mathrm { K L } } ( P \Vert R _ { 1 } ) + \mathbb { E } _ { P } [ \log \alpha ] - \log r } \\ & { \qquad \le D _ { \mathrm { K L } } ( P \Vert R _ { 1 } ) - \log r . } \end{array}
$$

Together with (8), this yields

$$
D _ { \mathrm { K L } } ( P \Vert R ) \leq \frac { D _ { \mathrm { K L } } ( Q \Vert R ) - d _ { \mathrm { B } } ( w \Vert r ) } { w } - \log r .\tag{9}
$$

We next use the inequality

$$
- \log r \le \frac { e } { e - 1 } \left( \log \frac { 1 } { w } + \frac { 1 } { w } d _ { \mathrm { B } } ( w \| r ) \right) , \qquad 0 < w , r < 1 .\tag{10}
$$

To verify it, first observe that

$$
\log \frac { 1 } { w } + \frac { 1 } { w } d _ { \mathrm { B } } ( w \| r ) = - \log r + \frac { 1 - w } { w } \log \frac { 1 - w } { 1 - r } .
$$

If $r \ \geq \ w ,$ the second term is nonnegative, and (10) follows. If $r \ < \ w ,$ the desired inequality is equivalent to

$$
e ( 1 - w ) \log { \frac { 1 - r } { 1 - w } } \leq - w \log r .
$$

For fixed $w ,$ let

$$
F _ { w } ( r ) : = - w \log r - e ( 1 - w ) \log { \frac { 1 - r } { 1 - w } } .
$$

This function is strictly convex on $( 0 , w )$ , and its stationary point is

$$
r _ { * } : = \frac { w } { e - ( e - 1 ) w } .
$$

Writing $a : = 1 - w ,$ direct substitution gives

$$
F _ { w } ( r _ { * } ) = G ( a ) : = [ 1 + ( e - 1 ) a ] \log [ 1 + ( e - 1 ) a ] - ( 1 - a ) \log ( 1 - a ) - e a \log e .
$$

Now $G ( 0 ) = G ( 1 ) = 0$ and

$$
G ^ { \prime } ( a ) = ( e - 1 ) \log [ 1 + ( e - 1 ) a ] + \log ( 1 - a ) ,
$$

while

$$
G ^ { \prime \prime } ( a ) = \frac { 1 } { \ln 2 } \left( \frac { ( e - 1 ) ^ { 2 } } { 1 + ( e - 1 ) a } - \frac { 1 } { 1 - a } \right) .
$$

The numerator of $G ^ { \prime \prime }$ after taking a common denominator is afine and strictly decreasing, positive at $a = 0$ and negative at $a = 1$ . Thus $G ^ { \prime }$ first increases and then decreases. Since $G ^ { \prime } ( 0 ) = 0$ and $G ^ { \prime } ( a )  - \infty$ as � ↑ 1, it is initially positive and then crosses zero exactly once. Hence � first increases and then decreases to $G ( 1 ) = 0$ . It follows that $G ( a ) \geq 0 ,$ proving (10).

Let $c : = e / ( e - 1 )$ ). Substituting (10) into (9), and using $d _ { \mathrm { B } } ( w \| r ) \leq D _ { \mathrm { K L } } ( Q \| R )$ , gives

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } ( P \| R ) \leq \displaystyle \frac { D _ { \mathrm { K L } } ( Q \| R ) } { w } + ( c - 1 ) \frac { d _ { \mathrm { B } } ( w \| r ) } { w } + c \log \frac { 1 } { w } } \\ & { \qquad \leq c \left( \log \displaystyle \frac { 1 } { w } + \frac { 1 } { w } D _ { \mathrm { K L } } ( Q \| R ) \right) . \ d \Omega } \end{array}
$$

Proof of Theorem 1. Fix �, and apply Lemma 3 to

$$
P = \mu _ { 1 : n } , \qquad Q = \xi _ { 1 : n } , \qquad R = \rho _ { 1 : n } .
$$

Since $\xi \ge w _ { \mu } \mu ,$

$$
D _ { \mathrm { K L } } ^ { n } ( \mu \| \rho ) \leq \frac { e } { e - 1 } \left( \log \frac { 1 } { w _ { \mu } } + \frac { D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho ) } { w _ { \mu } } \right) \leq \frac { e } { e - 1 } \left( \log \frac { 1 } { w _ { \mu } } + \frac { \eta } { w _ { \mu } } \right) .
$$

Applying Lemma 1 and taking the supremum over � gives the conclusion.

## B Proof of Lemma 2

Let $m : = | y |$ and set $N : = \lfloor \eta / q \rfloor$ . Note that

$$
q = \xi ( y ) \geq w _ { \mu } \mu ( y ) = w _ { \mu } > 0 .
$$

For $0 \leq \ell < N ,$ , write

$$
h _ { \ell } : = z _ { 1 : m + \ell } .
$$

We construct a measure $\rho$ on $\{ 0 , 1 \} ^ { \infty }$ as follows: set

$$
\rho ( \cdot \mid h _ { \ell } ) : = \mathrm { B e r n } ( 1 / 2 ) , \qquad 0 \le \ell < N ,
$$

and let $\rho$ agree with $\xi$ at every other history of positive �-probability. At histories of zero $\xi -$ probability, choose an arbitrary conditional distribution.

For any distribution � on {0, 1},

$$
D _ { \mathrm { K L } } ( \boldsymbol { p } \| \mathrm { B e r n } ( 1 / 2 ) ) = 1 - H ( \boldsymbol { p } ) \le 1 ,
$$

where $H ( p )$ is the binary entropy. Hence, for every $n ,$ the chain rule gives

$$
D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho ) \le \sum _ { 0 \le \ell < N } \xi ( h _ { \ell } ) D _ { \mathrm { K L } } \left( \xi ( \cdot  { | } h _ { \ell } )  { \| } \mathrm { B e r n } ( 1 / 2 ) \right) \le \sum _ { 0 \le \ell < N } \xi ( h _ { \ell } ) \le N q .
$$

Taking the supremum over � yields

$$
D _ { \mathrm { K L } } ( \boldsymbol { \xi } | | \rho ) \leq \eta .
$$

The modified conditionals occur only after the prefix $y .$ Therefore, the first � predictions along � are unchanged, and

$$
S _ { m } ( \mu , \rho ) = S _ { m } ( \mu , \xi ) .
$$

At each history $h _ { \ell } ,$ the next-symbol distribution under $\mu = \delta _ { z }$ is a point mass, whereas $\rho ( \cdot \mid h _ { \ell } ) =$ Bern(1/2). Thus

$$
( \mu ( 1 \mid h _ { \ell } ) - \rho ( 1 \mid h _ { \ell } ) ) ^ { 2 } + ( \mu ( 0 \mid h _ { \ell } ) - \rho ( 0 \mid h _ { \ell } ) ) ^ { 2 } = { \frac { 1 } { 2 } } .
$$

Consequently,

$$
S ( \mu , \rho ) \geq S _ { m + N } ( \mu , \rho ) = S _ { m } ( \mu , \xi ) + \frac { N } { 2 } .
$$

This finishes the proof.

## C Proof of Theorem 3

We remark Lemma 2 already gives a nearly exponential lower bound in the universal setting. Specifically, for every $n ,$ since $\begin{array} { r } { \sum _ { h \in \{ 0 , 1 \} ^ { n } } \xi _ { U } ( h ) = 1 } \end{array}$ , there is a string $h _ { n } \in \{ 0 , 1 \} ^ { n }$ such that $\xi _ { U } ( h _ { n } ) \leq 2 ^ { - n }$ Let $\mu _ { n } : = \delta _ { h _ { n } 0 ^ { \infty } }$ . We have

$$
n - O ( 1 ) \leq K ( \mu _ { n } ) \leq n + O ( \log n ) .
$$

The first inequality is by combining (i) $\xi _ { U } ( h _ { n } ) \leq 2 ^ { - n }$ and (ii) $\xi _ { U } ( h _ { n } ) \ge w _ { \mu _ { n } } = \Theta ( 2 ^ { - K ( \mu _ { n } ) } )$ . Lemma 2 therefore yields, for all suficiently large $n ,$

$$
\operatorname* { s u p } _ { \rho : D _ { \mathrm { K L } } ( \xi _ { U } \| \rho ) \le \eta } S ( \mu _ { n } , \rho ) = \Omega \left( \eta \frac { 2 ^ { K ( \mu _ { n } ) } } { K ( \mu _ { n } ) ^ { O ( 1 ) } } \right) .
$$

This matches the upper bound in Theorem 1 up to a polynomial factor. We next show that this factor can be removed.

We will make use of the following coding theorem.

Lemma 4 (Corollary 4.3.1 in $\left[ \mathrm { L V 0 8 } \right] )$ . If � is a lower semi-computable semi-measure on $\{ 0 , 1 \} ^ { * }$ with $\begin{array} { r } { \sum _ { y \in \{ 0 , 1 \} ^ { * } } P ( y ) \leq 1 } \end{array}$ , then there is a constant $c _ { P } = K ( P ) + O ( 1 )$ such that $K ( x ) \leq - \log P ( x ) + c _ { P }$

Proof of Theorem 3. For $m \geq 1 .$ , define

$$
C _ { m } : = \{ 1 ^ { m } 0 s : s \in \{ 0 , 1 \} ^ { m } \} , \qquad C : = \bigcup _ { m \geq 1 } C _ { m } .
$$

Note that the set C is prefix-free. Since $| C _ { m } | = 2 ^ { m }$ , a counting argument gives a string $y _ { m } \in C _ { m }$ satisfying $K ( y _ { m } ) \geq m$ . Here

$$
K ( y ) : = \operatorname* { m i n } \{ | p | : U ( p ) = y \}
$$

is the prefix Kolmogorov complexity of the string �. Denote $\mu _ { m } : = \delta _ { y _ { m } 0 ^ { \infty } }$ the deterministic measure outputting $y _ { m } 0 ^ { \infty }$ . Note that $\mu _ { m }$ is estimable, as it can hard-code the self-delimiting program generating $y _ { m }$ and then output zeros forever. In addition, we have $K ( \mu _ { m } ) \le K ( y _ { m } ) + O ( 1 )$ .

Let $M _ { U }$ be the Solomonof prior

$$
M _ { U } ( y ) : = \sum _ { p : U ( p ) = y * } 2 ^ { - | p | } ,
$$

which is semicomputable. Since C is prefix-free, $\begin{array} { r } { \sum _ { y \in C } M _ { U } ( y ) \leq 1 } \end{array}$ . By letting ${ \cal P } ( y ) = { \cal M } _ { U } ( y ) \cdot { \bf 1 } _ { y \in { \cal C } } ,$ Lemma 4 therefore gives

$$
M _ { U } ( y ) = O \left( 2 ^ { - K ( y ) } \right) \qquad { \mathrm { f o r ~ e v e r y ~ } } y \in C ,
$$

Furthermore, a standard fact in algorithmic information theory (see e.g. Section 1.2.5 in [Hut05]) says that $\xi _ { U } ( y ) = O ( M _ { U } ( y ) )$ . Thus,

$$
\xi _ { U } ( y ) = O \left( 2 ^ { - K ( y ) } \right) \qquad { \mathrm { f o r ~ e v e r y ~ } } y \in C ,
$$

In particular,

$$
\xi _ { U } ( y _ { m } ) = O \left( 2 ^ { - K ( y _ { m } ) } \right) = O \left( 2 ^ { - K ( \mu _ { m } ) } \right) .
$$

Fix $\eta > 0$ . Applying Lemma 2 with $\mu = \mu _ { m }$ and $y = y _ { m }$ gives: there exists a universal constant $C _ { U } > 0 _ { \prime }$ , such that

$$
\operatorname* { s u p } _ { \rho \colon D _ { \mathrm { K L } } ( \xi _ { U } \| \rho ) \le \eta } S ( \mu _ { m } , \rho ) \ge \frac 1 2 \left\lfloor \frac { \eta 2 ^ { K ( \mu _ { m } ) } } { C _ { U } } \right\rfloor \ge \frac { 1 } { 4 C _ { U } } \left( K ( \mu _ { m } ) + \eta 2 ^ { K ( \mu _ { m } ) } \right) ,
$$

for infinitely many �.

## D Proof of Proposition 1

For a history ℎ with $\xi ( h ) > 0$ , let $w ( \mu \mid h ) : = w _ { \mu } \mu ( h ) / \xi ( h )$ denote the posterior weight of $\mu .$ For any symbol $a \in X$

$$
\xi ( a \mid h ) = \sum _ { \mu \in \mathcal { M } } w ( \mu \mid h ) \mu ( a \mid h ) .
$$

The variance decomposition gives

$$
\sum _ { \mu \in { \mathcal M } } w ( \mu \mid h ) { \big ( } \mu ( a \mid h ) - \rho ( a \mid h ) { \big ) } ^ { 2 } = \sum _ { \mu \in { \mathcal M } } w ( \mu \mid h ) { \big ( } \mu ( a \mid h ) - \xi ( a \mid h ) { \big ) } ^ { 2 } + { \big ( } \xi ( a \mid h ) - \rho ( a \mid h ) { \big ) } ^ { 2 } .
$$

Multiplying by $\xi ( h )$ and using $\xi ( h ) w ( \mu \mid h ) = w _ { \mu } \mu ( h )$ yields

$$
\sum _ { \mu \in M } w _ { \mu } \mu ( h ) { \big ( } \mu ( a \mid h ) - \rho ( a \mid h ) { \big ) } ^ { 2 } = \sum _ { \mu \in M } w _ { \mu } \mu ( h ) { \big ( } \mu ( a \mid h ) - \xi ( a \mid h ) { \big ) } ^ { 2 } + \xi ( h ) { \big ( } \xi ( a \mid h ) - \rho ( a \mid h ) { \big ) } ^ { 2 } .
$$

Summing over all histories $h \in { \mathcal { X } } ^ { t - 1 }$ , all symbols $a \in { \mathcal { X } } ,$ and all $1 \leq t \leq n$ gives

$$
\mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \rho ) = \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \xi ) + S _ { n } ( \xi , \rho ) .
$$

## E A counterexample for squared Hellinger error

Let $\mu _ { 1 } ~ = ~ \delta _ { 1 ^ { \infty } }$ and $\mu _ { 0 } ~ = ~ \delta _ { 0 ^ { \infty } }$ , with prior weights $w _ { 1 } ~ = ~ 3 / 4$ and $w _ { 0 } \ = \ 1 / 4 ,$ , respectively. Write $q _ { r } = r \mu _ { 1 } + ( 1 - r ) \mu _ { 0 }$ , so that $\xi = q _ { 3 / 4 }$ . After the first symbol $x _ { 1 , }$ , every $q _ { r }$ with $r \in ( 0 , 1 )$ predicts the remaining sequence perfectly. Thus its averaged cumulative squared Hellinger error is

$$
R ( r ) : = \mathbb { E } _ { \mu \sim w } \mathrm { H e l } ( \mu , q _ { r } ) = 2 - 2 \big [ w _ { 1 } \sqrt { r } + w _ { 0 } \sqrt { 1 - r } \big ] .
$$

This is minimized at

$$
r _ { * } = \frac { w _ { 1 } ^ { 2 } } { w _ { 1 } ^ { 2 } + w _ { 0 } ^ { 2 } } = \frac { 9 } { 1 0 } \neq \frac { 3 } { 4 } .
$$

Hence $\xi$ is not a minimizer, ruling out the analogous decomposition into its average error and a nonnegative approximation term.

Moreover, let $\rho _ { u } = q _ { 3 / 4 - u }$ for $u > 0$ suficiently small. Since $\rho _ { u }$ and $\xi$ difer only in their first-step predictions,

$$
\eta _ { u } : = D _ { \mathrm { K L } } ( \xi | | \rho _ { u } ) = w _ { 1 } \log \frac { w _ { 1 } } { w _ { 1 } - u } + w _ { 0 } \log \frac { w _ { 0 } } { w _ { 0 } + u } = \Theta ( u ^ { 2 } ) .
$$

However,

$$
R ( 3 / 4 - u ) - R ( 3 / 4 ) = \frac { \sqrt { 3 } - 1 } { 2 } u + O ( u ^ { 2 } ) = \Theta ( \sqrt { \eta _ { u } } ) .
$$

Thus, no finite constant � can make $R ( 3 / 4 - u ) \le R ( 3 / 4 ) + C \eta _ { u }$ hold for all suficiently small �.

## F Proof of Theorem 5

By Proposition 1, we have

$$
\mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \rho ) = \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) + S _ { n } ( \xi ^ { \prime } , \rho ) ,\tag{11}
$$

and

$$
\begin{array} { r } { \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ) = \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) + S _ { n } ( \xi ^ { \prime } , \xi ) \ge \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) . } \end{array}
$$

In the following, we will show that

$$
S _ { n } ( \xi ^ { \prime } , \rho ) \leq ( \ln 2 ) \left( \sqrt { \log C } + \sqrt { C \eta } \right) ^ { 2 } ,\tag{12}
$$

which concludes the proof.

The assumption $w _ { \nu } ^ { \prime } \leq C w _ { \nu }$ implies that, for every finite history ℎ,

$$
\xi ^ { \prime } ( h ) = \sum _ { \nu \in \mathcal { M } } w _ { \nu } ^ { \prime } \nu ( h ) \leq C \sum _ { \nu \in \mathcal { M } } w _ { \nu } \nu ( h ) = C \xi ( h ) .
$$

In particular, $D _ { \mathrm { K L } } ^ { n } ( \xi ^ { \prime } | | \xi ) \leq \log C .$ . Hence, by Lemma 1, we have

$$
S _ { n } ( \xi ^ { \prime } , \xi ) \leq ( \ln 2 ) \cdot D _ { \mathrm { K L } } ^ { n } ( \xi ^ { \prime } \| \xi ) \leq ( \ln 2 ) \cdot \log C .\tag{13}
$$

Next, define

$$
B _ { n } : = \sum _ { t = 1 } ^ { n } \sum _ { h \in X ^ { t - 1 } } \xi ^ { \prime } ( h ) \sum _ { a \in X } \left( \xi ( a \mid h ) - \rho ( a \mid h ) \right) ^ { 2 } .
$$

Using $\xi ^ { \prime } ( h ) \leq C \xi ( h )$ and Lemma 1, we obtain

$$
B _ { n } \leq C \sum _ { t = 1 } ^ { n } \sum _ { h \in X ^ { t - 1 } } \xi ( h ) \sum _ { a \in X } \left( \xi ( a \mid h ) - \rho ( a \mid h ) \right) ^ { 2 } = C \cdot S _ { n } ( \xi , \rho ) \leq \left( C \ln 2 \right) \cdot D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho ) \leq \left( \ln 2 \right) \cdot C \eta .\tag{14}
$$

Expanding the square, we obtain

$$
S _ { n } ( \xi ^ { \prime } , \rho ) = S _ { n } ( \xi ^ { \prime } , \xi ) + B _ { n } + 2 T _ { n } ,
$$

where

$$
T _ { n } : = \sum _ { t = 1 } ^ { n } \sum _ { h \in \mathcal { X } ^ { t - 1 } } \xi ^ { \prime } ( h ) \sum _ { a \in \mathcal { X } } \left( \xi ^ { \prime } ( a \mid h ) - \xi ( a \mid h ) \right) \left( \xi ( a \mid h ) - \rho ( a \mid h ) \right) .
$$

By the Cauchy–Schwarz inequality,

$$
\left| T _ { n } \right| \leq { \sqrt { S _ { n } ( \xi ^ { \prime } , \xi ) B _ { n } } } .
$$

It follows that

$$
\begin{array} { c } { { S _ { n } ( \xi ^ { \prime } , \rho ) \leq S _ { n } ( \xi ^ { \prime } , \xi ) + B _ { n } + 2 \sqrt { S _ { n } ( \xi ^ { \prime } , \xi ) B _ { n } } } } \\ { { { } } } \\ { { = \left( \sqrt { S _ { n } ( \xi ^ { \prime } , \xi ) } + \sqrt { B _ { n } } \right) ^ { 2 } . } } \end{array}
$$

Together with (13) and (14), this yields (12).

## G Proofs for Section 5

Fix a horizon �. Write $P : = \xi _ { 1 : n } , H : = H ( P )$ , and $a : = \eta _ { \mathrm { r e p } } + \eta _ { \mathrm { o p t } }$ . Let $\boldsymbol { \rho } ^ { * } \in \mathcal { H }$ denote the predictor achieving $D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho ^ { * } ) = \eta _ { \mathrm { r e p } }$ . Write ${ \widehat { R } } _ { m }$ for the empirical log-loss, so that $\widehat { R } _ { m } ( \rho _ { m } ) \leq \widehat { R } _ { m } ( \rho ^ { * } ) + \eta _ { \mathrm { o p t } }$

Proof of (4). Let ${ \widehat { P } } _ { m }$ be the empirical distribution of the training sequences. For $0 < \tau < 1$ , define

$$
E _ { m , \tau } : = \Bigl \{ \exists x \in \mathrm { s u p p } ( P ) : | \widehat { P } _ { m } ( x ) - P ( x ) | > \tau P ( x ) \Bigr \} .
$$

Write $d : = | \operatorname { s u p p } ( P ) |$ and

$$
p _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { x \in \mathrm { s u p p } ( P ) } P ( x ) > 0 ,
$$

Hoefding’s inequality and a union bound give

$$
\begin{array} { r } { \operatorname* { P r } [ E _ { m , \tau } ] \leq 2 d \exp ( - 2 m \tau ^ { 2 } p _ { \mathrm { m i n } } ^ { 2 } ) . } \end{array}
$$

Conditioned on $\overline { { E } } _ { m , \tau }$ , the nonnegativity of log-loss and approximate empirical optimality imply

$$
\begin{array} { r } { ( 1 - \tau ) \big [ H + D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho _ { m } ) \big ] \leq \widehat { R } _ { m } ( \rho _ { m } ) \leq ( 1 + \tau ) ( H + \eta _ { \mathrm { r e p } } ) + \eta _ { \mathrm { o p t } } . } \end{array}
$$

Thus,

$$
D _ { \mathrm { K L } } ^ { n } ( \boldsymbol { \xi } | | \rho _ { m } ) \leq a + \frac { \tau } { 1 - \tau } \big [ 2 ( H + \eta _ { \mathrm { r e p } } ) + \eta _ { \mathrm { o p t } } \big ] .
$$

For any $\gamma > 0 ,$ choose $\tau > 0$ small enough that the second term is at most $\gamma .$ The claim follows with $C _ { \gamma } = 2 d$ and $c _ { \gamma } = 2 \tau ^ { 2 } p _ { \mathrm { m i n } } ^ { 2 } .$

Example establishing (5). Take $n = 1 , P = \mathrm { B e r n o u l l i } ( 1 / 2 )$ , and H consisting of $\rho _ { 0 } : = \delta _ { 0 }$ and $\rho _ { 1 } : = \mathrm { B e r n o u l l i } ( 1 / 4 )$ . On the event that all � training observations are zero, which has probability $2 ^ { - m }$ , exact empirical log-loss minimization selects $\rho _ { m } = \rho _ { 0 }$ . Since $D _ { \mathrm { K L } } ( P \| \rho _ { 0 } ) = + \infty ,$ the expected KL error is infinite for every finite �. Padding deterministic zeros extends the example to any fixed �.

Proof of (6). Recall $a = \eta _ { \mathrm { r e p } } + \eta _ { \mathrm { o p t } } ,$ , let $\gamma _ { a } : = ( 1 / \ln 2 - 1 ) a$ . By $0 \leq S _ { n } \leq 2 n$ and $S _ { n } ( \xi , \rho _ { m } ) \ \leq$ $( \ln 2 ) D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho _ { m } )$ , we have

$$
\begin{array} { r } { \mathbb { E } _ { \mathcal { D } _ { m } } \big [ ( S _ { n } ( \xi , \rho _ { m } ) - a ) _ { + } \big ] \leq 2 n \operatorname* { P r } \{ D _ { \mathrm { K L } } ^ { n } ( \xi \| \rho _ { m } ) > a + \gamma _ { a } \} \leq 2 n C _ { \gamma _ { a } } e ^ { - c _ { \gamma _ { a } } m } . } \end{array}
$$

Theorems $4 , 5 ,$ and 1 under squared-error approximation. Suppose $S _ { n } ( \xi , \rho ) \overset { } { \underset { } { = } } \eta$ . By Proposition 1, we have

$$
\begin{array} { r } { \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \rho ) = \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \xi ) + S _ { n } ( \xi , \rho ) \le \mathbb { E } _ { \mu \sim w } S _ { n } ( \mu , \xi ) + \eta . } \end{array}
$$

In the proof of Theorem $5 ,$ by almost the same argument except adjusting (14) to $B _ { n } \leq C \eta$ , we will get

$$
\mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \rho ) \le \mathbb { E } _ { \mu \sim w ^ { \prime } } S _ { n } ( \mu , \xi ^ { \prime } ) + \left( \sqrt { ( \ln 2 ) \log C } + \sqrt { C \eta } \right) ^ { 2 } .
$$

Letting $w ^ { \prime }$ concentrate on a single environment $\mu ,$ we recover Theorem 1, up to constants:

$$
S _ { n } ( \mu , \rho ) \leq \left( \sqrt { \left( \ln 2 \right) \log \frac { 1 } { w _ { \mu } } } + \sqrt { \frac { \eta } { w _ { \mu } } } \right) ^ { 2 } = O \left( \log \frac { 1 } { w _ { \mu } } + \frac { \eta } { w _ { \mu } } \right) .
$$

Theorem 3 under squared-error approximation. By Lemma 1,

$$
\{ \rho : D _ { \mathrm { K L } } ( \xi _ { U } \| \rho ) \leq \eta / \ln 2 \} \subseteq \{ \rho : S ( \xi _ { U } , \rho ) \leq \eta \} .
$$

Thus Theorem 3, applied with budget $\eta / \ln 2 ,$ implies the same infinite-horizon lower bound under squared-error approximation, up to constants.

## H Mitigation With Category Information

The worst-case bounds of order $\Theta ( \eta / w _ { \mu } )$ in Theorem 1 arise when approximation is controlled only through a single aggregate KL bound $D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta$ . When available, training-time task category information enables separate control of approximation errors within each task category, potentially improving category-wise prediction performance. Notably, the learned predictor does not require category tags at test time.

Let $\mathcal { T } = \{ T _ { 1 } , . . . , T _ { q } \}$ be a finite set of categories, with tag $: \mathcal { M }  \mathcal { T }$ defining the partition $M _ { j } : = \{ \mu \in M : \mathrm { t a g } ( \mu ) = T _ { j } \}$ . Write

$$
p _ { j } : = \sum _ { \mu \in { \cal M } _ { j } } w _ { \mu } , \qquad \xi _ { j } : = \sum _ { \mu \in { \cal M } _ { j } } \frac { w _ { \mu } } { p _ { j } } \mu .
$$

Thus $\begin{array} { r } { \xi = \sum _ { j = 1 } ^ { g } p _ { j } \xi _ { j } } \end{array}$ . Recall that, by Theorem 1, an aggregate approximation guarantee $D _ { \mathrm { K L } } ( \xi \| \rho ) \le \eta$ gives

$$
S _ { n } ( \xi _ { j } , \rho ) = O \left( \log \frac { 1 } { p _ { j } } + \frac { \eta } { p _ { j } } \right) .
$$

Direct control of the category-wise KL error avoids this inverse-category-mass amplification. The following guarantee follows immediately from Lemma 1 and Proposition 1.

Proposition 2 (Category-wise approximation). $I f D _ { \mathrm { K L } } ^ { n } ( \xi _ { j } \| \rho ) \leq \eta _ { j }$ , then

$$
\begin{array} { r } { \mathbb { E } _ { \mu \sim w ( \cdot \vert j ) } S _ { n } ( \mu , \rho ) - \mathbb { E } _ { \mu \sim w ( \cdot \vert j ) } S _ { n } ( \mu , \xi _ { j } ) = S _ { n } ( \xi _ { j } , \rho ) \le ( \ln 2 ) \eta _ { j } . } \end{array}
$$

In particular, if the tag identifies the generating task and $D _ { \mathrm { K L } } ^ { n } ( \xi _ { \mu } | | \rho ) \leq \eta _ { \mu } ,$ , then

$$
S _ { n } ( \mu , \rho ) = O \left( \eta _ { \mu } \right) .
$$

To control the category-wise approximation errors $D _ { \mathrm { K L } } ^ { n } ( \xi _ { j } | | \rho )$ , one may minimize the following worst-category excess log-loss, or a suitable surrogate:

$$
\operatorname* { m i n } _ { \rho } \operatorname* { m a x } _ { 1 \leq j \leq g } D _ { \mathrm { K L } } ^ { n } ( \xi _ { j } \| \rho ) .
$$

## I Higher moments of sequence log-loss

One may instead minimize the following sample-only training objective:

$$
\begin{array} { r } { \mathcal { L } _ { p } ( \rho ) : = \mathbb { E } _ { x \sim \xi } \left[ \left( - \log \rho ( x ) \right) ^ { p } \right] , \quad p \ge 2 . } \end{array}
$$

Fix a horizon �. Let $q$ be the minimizer of $\mathcal { L } _ { p }$ among all probability distributions on $X ^ { n }$ . Suppose the predictor $\rho$ satisfies

$$
\begin{array} { r } { \mathcal L _ { p } ( \rho _ { 1 : n } ) \leq \mathcal L _ { p } ( q ) + \eta . } \end{array}
$$

Lemma 5. For every $\mu \in { \cal M } ,$

$$
S _ { n } ( \mu , \rho ) = O _ { p , | { \cal X } | } \left( \log \frac { 1 } { w _ { \mu } } + \log ( n + 1 ) + \left( \frac { \eta } { w _ { \mu } } \right) ^ { 1 / p } \right) .
$$

Here, the implicit constant in $O _ { p , | X | } ( \cdot )$ may depend on � and $| \chi |$ , but not on any other parameters.

Proof. Let $\Omega _ { n } : = \{ x \in X ^ { n } : \xi ( x ) > 0 \}$ be the support of $\xi _ { 1 : n } .$ . Since $\xi \ge w _ { \mu } \mu ,$ the distribution $\mu _ { 1 : n }$ is also supported on $\Omega _ { n }$ . The minimizer � assigns all its mass to $\Omega _ { n } ,$ since otherwise, renormalizing its restriction to $\Omega _ { n }$ would strictly decrease the objective. If $\left| \Omega _ { n } \right| \ = \ 1$ , then the result follows immediately from

$$
\left( - \log \rho ( x ) \right) ^ { p } \leq \eta
$$

on the unique point $x \in \Omega _ { n }$ . We henceforth assume $\left| \Omega _ { n } \right| \geq 2$ . In this case, $q ( x ) \in ( 0 , 1 )$ for every $x \in \Omega _ { n }$

Write $\ell ( x ) : = - \log q ( x )$ . Consider the Lagrangian

$$
\mathcal { L } ( q , \lambda ) : = \sum _ { x \in \Omega _ { n } } \xi ( x ) \big ( - \log _ { 2 } q ( x ) \big ) ^ { p } + \lambda \left( \sum _ { x \in \Omega _ { n } } q ( x ) - 1 \right) .
$$

Since the minimizer is an interior point, the first-order condition gives

$$
\frac { \partial \mathcal { L } } { \partial q ( x ) } = - \frac { \mathcal { P } \xi ( x ) \ell ( x ) ^ { p - 1 } } { q ( x ) \ln 2 } + \lambda = 0 , \qquad x \in \Omega _ { n } .
$$

Thus, setting $c _ { n } : = \lambda$ ln $2 / p ,$ , we obtain

$$
\xi ( x ) \ell ( x ) ^ { p - 1 } = c _ { n } q ( x ) , \qquad x \in \Omega _ { n } .\tag{15}
$$

Since $\xi ( x ) > 0 , \ell ( x ) > 0 .$ , and $q ( x ) > 0 ,$ , we also have $c _ { n } > 0 .$ . Summing (15) over � gives

$$
c _ { n } = \mathbb { E } _ { x \sim \xi } \left[ \ell ( x ) ^ { p - 1 } \right] .
$$

Since $u \mapsto u ^ { ( p - 1 ) / p }$ is concave, Jensen’s inequality gives

$$
\begin{array} { r } { c _ { n } \leq \left( \mathbb { E } _ { x \sim \xi } \left[ \ell ( x ) ^ { P } \right] \right) ^ { ( p - 1 ) / p } = \mathcal { L } _ { p } ( q ) ^ { ( p - 1 ) / p } \leq \mathcal { L } _ { p } \big ( \mathrm { U n i f } ( \Omega _ { n } ) \big ) ^ { ( p - 1 ) / p } \leq \big ( n \log | X | \big ) ^ { p - 1 } . } \end{array}
$$

We next show that � dominates �. If $\ell ( x ) \geq 1$ , then (15) gives

$$
{ \frac { q ( x ) } { \xi ( x ) } } = { \frac { \ell ( x ) ^ { p - 1 } } { c _ { n } } } \geq { \frac { 1 } { { \bigl ( } n \log | X | { \bigr ) } ^ { p - 1 } } } .
$$

If $\ell ( x ) < 1$ , then $q ( x ) > 1 / 2$ , and hence

$$
\frac { q ( x ) } { \xi ( x ) } \geq q ( x ) > \frac { 1 } { 2 } ,
$$

The two cases imply

$$
q ( x ) \geq { \frac { \xi ( x ) } { C _ { n , p } } } , \qquad C _ { n , p } : = 2 { \bigl ( } n \log | \chi | { \bigr ) } ^ { p - 1 } .\tag{16}
$$

We now use the approximate optimality of $\rho .$ . Define $z ( x ) : = \log ( q ( x ) / \rho ( x ) )$ . Since $\mathcal { L } _ { p } ( \rho _ { 1 : n } ) < \infty ,$ it has that $\rho ( x ) > 0 \mathrm { o n } \Omega _ { n } ,$ and thus

$$
- \log \rho ( x ) = \ell ( x ) + z ( x ) \geq 0 .
$$

For $p \ge 2$ and $u , v \geq 0 ,$ , we have

$$
v ^ { p } \geq u ^ { p } + p u ^ { p - 1 } ( v - u ) + | v - u | ^ { p } .\tag{17}
$$

Indeed, when $\boldsymbol { v } = \boldsymbol { u } + \boldsymbol { s }$ with $s \geq 0 .$ , subtracting the right-hand side from the left-hand side gives a function $F _ { + } ( s )$ satisfying $F _ { + } ( 0 ) = 0$ and

$$
F _ { + } ^ { \prime } ( s ) = { \cal { P } } \left( ( u + s ) ^ { p - 1 } - u ^ { p - 1 } - s ^ { p - 1 } \right) \geq 0 .
$$

When $\boldsymbol { v } = \boldsymbol { u } - \boldsymbol { s }$ with $0 \leq s \leq u ,$ the corresponding function satisfies $F _ { - } ( 0 ) = 0$ and

$$
F _ { - } ^ { \prime } ( s ) = { \ o } \left( u ^ { p - 1 } - ( u - s ) ^ { p - 1 } - s ^ { p - 1 } \right) \geq 0 .
$$

Applying (17) with

$$
u = \ell ( x ) , \qquad v = - \log \rho ( x ) = \ell ( x ) + z ( x ) ,
$$

we obtain

$$
\begin{array} { r } { \mathcal { L } _ { p } ( \rho ) - \mathcal { L } _ { p } ( q ) = \mathbb { E } _ { x \sim \xi } \left[ ( \ell ( x ) + z ( x ) ) ^ { p } - \ell ( x ) ^ { p } \right] \geq \mathbb { E } _ { x \sim \xi } \left[ \vert z ( x ) \vert ^ { p } \right] + p \mathbb { E } _ { x \sim \xi } \left[ \ell ( x ) ^ { p - 1 } z ( x ) \right] . } \end{array}
$$

By (15),

$$
\mathbb { E } _ { { x } \sim { \xi } } \left[ \ell ( { x } ) ^ { p - 1 } z ( { x } ) \right] = c _ { n } \sum _ { { x } \in \Omega _ { n } } q ( { x } ) \log \frac { q ( { x } ) } { \rho ( { x } ) } = c _ { n } D _ { \mathrm { K L } } ( q \| \rho _ { 1 : n } ) \ge 0 .
$$

It follows from the approximation guarantee that

$$
\mathbb { E } _ { x \sim \xi } \left[ \left| \log \frac { q ( x ) } { \rho ( x ) } \right| ^ { p } \right] \leq \eta .\tag{18}
$$

Since $\xi ( x ) \ge w _ { \mu } \mu ( x )$ , (16) and (18) imply

$$
q ( x ) \geq \frac { w _ { \mu } } { C _ { n , p } } \mu ( x )
$$

and

$$
\mathbb { E } _ { x \sim \mu } \left[ | z ( x ) | ^ { \mathfrak { p } } \right] \leq \frac { \eta } { w _ { \mu } } .
$$

Therefore, by Holder’s inequality,

$$
\begin{array} { l } { { \displaystyle { D _ { \mathrm { K L } } ^ { n } ( \mu \| \rho ) = \mathbb { E } _ { x \sim \mu } \left[ \log \frac { \mu ( x ) } { q ( x ) } + z ( x ) \right] \leq \log \frac { C _ { n , \rho } } { w _ { \mu } } + \mathbb { E } _ { x \sim \mu } | z ( x ) | \leq \log \frac { C _ { n , \rho } } { w _ { \mu } } + \left( \frac { \eta } { w _ { \mu } } \right) ^ { 1 / \ p } } } } \\ { { \displaystyle ~ = O _ { p , \vert X \vert } \left( \log \frac { 1 } { w _ { \mu } } + \log ( n + 1 ) + \left( \frac { \eta } { w _ { \mu } } \right) ^ { 1 / p } \right) . } } \end{array}
$$

By applying Lemma 1, we obtain the lemma.