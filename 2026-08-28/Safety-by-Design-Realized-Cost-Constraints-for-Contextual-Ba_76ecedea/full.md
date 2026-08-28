# Safety by Design: Realized-Cost Constraints for Contextual Bandits with Continuous Actions

Spyros Dragazis<sup>1</sup> and Aldo Pacchiano<sup>1,2</sup>

<sup>1</sup>Boston University <sup>2</sup>Broad Institute of MIT and Harvard

August 28, 2026

## Abstract

Contextual bandits are a standard framework for sequential decision-making under uncertainty, with applications in clinical trials, dosage selection, recommendation systems, and autonomous systems. Safety is central in many of these applications, since a single unsafe decision in settings such as dosage selection or autonomous driving can have catastrophic consequences. A common way to model safety in bandit problems is to associate each action with both a reward signal and a cost signal, and to optimize reward subject to constraints on cost. Most existing safety-constrained bandit models enforce safety by requiring the expected cost of each action to remain below a prescribed threshold. However, this may be insuficient in heteroscedastic settings, where the chosen action afects not only the expected reward and cost, but also the variability of the observed outcomes. We study contextual bandits with one-dimensional continuous actions and stage-wise high-probability constraints on the realized cost. We propose High-Probability Constrained UCB, an optimistic-pessimistic algorithm that explores for reward while conservatively estimating the <sub>safe action set. For linear reward and cost models, we prove a tight O</sub>˜<sub>(d</sub>√<sub>T) regret bound, and</sub> we extend the analysis to general function classes using the eluder dimension. Experiments show that enforcing realized-cost safety substantially reduces violations compared with expected-cost constrained baselines.

## 1 Introduction

Bandit algorithms [33, 3, 19] have been applied across a wide range of domains, including reinforcement learning with human feedback (RLHF) [11, 40], clinical trials [23, 21, 4, 35, 7], recommendation systems [38], dynamic pricing [25], large language models [8], and fair allocation [29]; see [9] for a broader overview. In these domains, a learner interacts sequentially with an unknown environment, aiming to learn from observations while maximizing cumulative reward through its decisions. Recently, there has been increasing interest in contextual bandits [13, 19], where the learner observes a context before selecting an action from a potentially large or continuous action space.

Within this general framework, constrained bandit algorithms are particularly relevant when applications involve resource limitations or safety requirements [12, 36]. Such constraints can take diferent forms, including budget-type constraints as in knapsack bandits [5], fairness constraints [16], and models with simultaneous reward and cost signals [2, 27, 28]. The latter formulation is especially natural in applications such as advertising and drug administration, where decisions must balance utility against risk. In drug dosage problems, for example, treatments generate an eficacy signal, interpreted as reward, and a toxicity signal, interpreted as cost; the goal is to maximize therapeutic benefit while ensuring that toxicity remains below a prescribed threshold τ, as in Phase II clinical trials. For example, related work by [21] considers a binary reward–cost model that enforces constraints

on the average cost over time.

While cumulative or averaged cost control is suficient in some applications, many safety-critical settings require stage-wise constraints, namely control of the cost at each time step. To address this requirement, researchers have studied the well-established safe linear bandit problem [15, 28, 24], in which, at each round t, the selected action generates both a reward signal $\boldsymbol { r } _ { t } \in \mathbb { R }$ and a cost signal $c _ { t } \in \mathbb { R }$ . The objective is to maximize expected cumulative reward while ensuring that the expected cost remains below a known safety threshold τ at every round, with this constraint holding with high probability.

Guarantees on the expected value of the cost signal are natural in homoscedastic models, where the level of noise is not afected by the chosen action. However, in applications such as autonomous driving, robotic control, and drug treatment, the variability of the outcome may itself depend on the action level. For example, a faster robotic motion may lead to a wider distribution of possible next positions due to friction or sensor noise, a larger velocity in autonomous driving may be less predictable than a smaller one, and the efect of a drug may vary across dosage levels [18]. In such heteroscedastic settings, satisfying a constraint only in expectation does not preclude catastrophic outcomes at individual rounds. This motivates algorithms that control the realized cost signal itself at each time step, rather than only its expectation. With this motivation, we consider the following question.

## Is it possible to design constrained low-regret algorithms

## that control realized cost at every round with high probability?

In this work, we address this question in a heteroscedastic reward–cost model with one-dimensional continuous actions [2, 28, 22]. Our goal is to control the realized cost signal $C _ { t }$ at each time step by requiring it to satisfy the constraint $\mathbb { P } ( C _ { t } \leq \tau ) \geq 1 - \delta$ , where $\delta > 0$ is a user-specified confidence parameter. As in prior work, the resulting guarantee accounts for both the noise in the cost observations and the internal randomization of the algorithm.

To achieve this objective, we propose a UCB-like algorithm termed High Probability Constrained UCB. Our method is based on the Optimism in the Face of Uncertainty (OFUL) principle [3, 1] and does not require prior knowledge of an initial safe action, in contrast to several existing approaches [28]. The algorithm applies in both stochastic and adversarial contextual settings, under the standard assumption of sub-Gaussian noise for both reward and cost signals. Under this assumption, we construct a high-probability constraint event that ensures the realized cost remains below the safety threshold at every round.

We show that the proposed algorithm achieves a T-round regret bound of order $\tilde { \mathcal { O } } ( d \sqrt { T } )$ , which is minimax optimal in its dependence on both the dimension d and the time horizon T for linear bandits [19]. In addition, we empirically validate the performance of our approach on both synthetic and real-world datasets. Finally, we extend our analysis to settings with non-linear reward and cost functions, characterizing the regret in terms of the eluder dimension [31], a complexity measure for general function classes.

The main contributions of this work are as follows:

• We introduce a safety framework for reward–cost constrained bandits in a heteroscedastic setting with one-dimensional continuous actions.

• We derive algorithms with tight regret guarantees for both linear and non-linear reward and cost models.

• We validate our method on both synthetic and real-world datasets, showing the importance of high-probability realized-cost guarantees in heteroscedastic settings.

## 2 Model and Problem Formulation

Notation. We use the following notation throughout the paper. For vectors $x , y \in \mathbb { R } ^ { d }$ , we write $\langle x , y \rangle = x ^ { \top } y$ and $\langle x , y \rangle _ { \mathbf { A } } = x ^ { \top } \mathbf { A } y$ for the standard and weighted inner products, respectively, where $\mathbf { A } \in \mathbb { R } ^ { d \times d }$ is positive definite. Similarly, we define $\| x \| _ { 2 } = { \sqrt { \langle x , x \rangle } }$ and $\| x \| _ { \mathbf { A } } = { \sqrt { \langle x , x \rangle _ { \mathbf { A } } } }$ as the $\ell _ { 2 }$ and weighted $\ell _ { 2 }$ norms of x. We denote the indicator function by $\mathbf { 1 } \{ \cdot \}$ . Upper-case letters denote random variables, $\mathrm { e . g . , } X$ , and lower-case letters denote their realizations, $\mathrm { e . g . } , X = x$ . We write $[ T ] = \{ 1 , \dots , T \}$ . Finally, $\widetilde { \mathcal { O } }$ denotes $\mathrm { b i g } _ { - } \mathcal { O }$ notation up to logarithmic factors.

Inspired by bandit algorithms for adaptive dosage allocation [23, 21, 4], we consider the following formulation of actions, rewards, and costs. At each round $t \in [ T ]$ , the learner observes a d-dimensional context vector $X _ { t } \in \mathbb { R } ^ { d }$ , which may represent structured measurements, such as medical test results, or an embedding of unstructured information, such as text descriptions generated by a language model [6]. We impose no distributional assumptions on $X _ { t }$ ; the contexts may be generated either stochastically or adversarially. The learner then selects a one-dimensional continuous action $\alpha _ { t } \in [ 0 , 1 ]$ [22], which can be viewed as a continuous generalization of a finite set of dosage levels. The environment generates reward and cost signals according to

$$
R _ { t } : = g ( \alpha _ { t } ) \cdot \left( r ( X _ { t } ) + \xi _ { t } ^ { r } \right) ,\tag{1}
$$

$$
C _ { t } : = g ( \alpha _ { t } ) \cdot \left( c ( X _ { t } ) + \xi _ { t } ^ { c } \right) ,\tag{2}
$$

where $\xi _ { t } ^ { r }$ and $\xi _ { t } ^ { c }$ are sub-Gaussian noise terms, and $r ( X _ { t } )$ and $c ( X _ { t } )$ denote the underlying reward and cost functions. The function $g : [ 0 , 1 ]  [ 0 , 1 ]$ is known and strictly increasing, with $g ( 0 ) = 0$ and $g ( 1 ) = 1$ . In dosage applications, g can be interpreted as a covariate-independent dose-response curve, such as a population-level eficacy or toxicity profile that can be estimated from prior studies. <sup>1</sup> The normalization $g ( 1 ) = 1$ is without loss of generality, since any constant scale can be absorbed into the reward and cost functions. This formulation allows for nearly flat regions that model saturation or diminishing returns, while preserving invertibility. In dosage-like applications, eficacy and toxicity are commonly assumed to increase with the dosage level; see [4, 21, 20] and references therein. In the linear setting, we assume $r ( X _ { t } ) = \langle X _ { t } , \theta ^ { * } \rangle$ and $c ( X _ { t } ) = \langle X _ { t } , \bar { \mu ^ { * } } \rangle$ for unknown parameters $\theta ^ { * } , \mu ^ { * } \in \mathbb { R } ^ { d }$ In more general settings, we require only that $r ( X _ { t } )$ and $c ( X _ { t } )$ are bounded. The interaction protoco is summarized in Figure 1.

To illustrate the linear case, consider a medical treatment scenario in which the context $X _ { t } \in \mathbb { R } ^ { d }$ represents a patient feature vector encoding medical attributes such as blood pressure, oxygen saturation, or laboratory test results. In this model, there exist unknown parameters $\theta ^ { \star } , \mu ^ { \star } \in \mathbb { R } ^ { d }$ such that

$$
\begin{array} { r } { R _ { t } = g ( \alpha _ { t } ) \cdot ( \langle X _ { t } , \theta ^ { \star } \rangle + \xi _ { t } ^ { r } ) , } \\ { C _ { t } = g ( \alpha _ { t } ) \cdot ( \langle X _ { t } , \mu ^ { \star } \rangle + \xi _ { t } ^ { c } ) . } \end{array}
$$

The vector $\theta ^ { \star }$ characterizes how patient-specific features influence treatment eficacy: a positive inner product $\langle X _ { t } , \theta ^ { \star } \rangle$ indicates high expected benefit from treatment. Similarly, the vector $\mu ^ { \star }$ captures how the same features relate to adverse efects or toxicity, with larger values of $\boldsymbol { \left. X _ { t } , \mu ^ { \star } \right. }$ indicating increased risk of side efects. The noise terms $\xi _ { t } ^ { r }$ and $\xi _ { t } ^ { c }$ represent stochastic variability not captured by the linear model, such as unobserved physiological factors or measurement errors.

The action $\alpha _ { t } \in [ 0 , 1 ]$ represents the administered dosage level. Through the known response function $^ { g , }$ the dosage scales both the reward and cost signals, allowing the learner to trade of eficacy against toxicity. For instance, a patient with elevated inflammatory markers, corresponding to a high value of $\left. X _ { t } , \theta ^ { \star } \right.$ ⟩, may benefit substantially from treatment. However, if the same patient exhibits impaired kidney function, corresponding to a high value of $\langle X _ { t } , \mu ^ { \star } \rangle$ , a reduced dosage may be preferable to mitigate adverse outcomes. This formulation is consistent with standard monotonicity assumptions on eficacy–toxicity dose-response curves [37, 20], whereby increasing the dose tends to increase both therapeutic and toxic efects.

Safe Bandit protocol   
Input: Horizon T   
For rounds $t = 1 , 2 , \dots , T \colon$   
1. $\mathcal { S } _ { t } \in \mathbb { R } ^ { d }$  % context   
2. α<sub>t</sub>∈[0,1]  % action selection   
3. R<sub>t</sub>,C<sub>t</sub>∈R  % reward, cost signals  
Figure 1: Safe Bandit protocol.

The same action-dependent scaling also makes the model heteroscedastic. Indeed, the variability of the observed reward and cost depends on the selected dosage through $g ( \alpha _ { t } )$ . This captures the fact that aggressive treatments may lead to less predictable responses [30]. In this sense, the model allows for stochastic departures from deterministic monotonicity, such as saturation or instability in biological responses.

As a result, the learner must select actions $\alpha _ { t }$ not only to maximize expected cumulative reward, but also to control the variability induced by the treatment. Our model belongs to the broader class of heteroscedastic bandits [41, 26, 39, 14], in which the noise distribution may depend on both the current context and the chosen action. In more general formulations, the noise variance or sub Gaussian parameter may scale as a quadratic form $\phi ( X _ { t } , A _ { t } ) ^ { \top } \Sigma ^ { \star } \phi ( X _ { t } , A _ { t } )$ , where $\phi ( \cdot , \cdot ) : \mathcal { X } \times \mathcal { A }  \mathbb { R } ^ { d }$ denotes a joint context–action embedding and $\Sigma ^ { \star }$ is an unknown positive definite matrix. While our model adopts a simpler structure, it captures monotone dose-response behavior commonly observed in reward–cost systems and remains interpretable, a property that is particularly valuable in medical decision-making contexts.

## 2.1 Assumptions

Before presenting our analysis and results, we state the assumptions used throughout the paper. These assumptions are standard in the literature on constrained contextual bandits; see [15, 28, 24] and references therein.

Assumption 2.1 (Sub-Gaussian noise). For all $t \in [ T ]$ , the reward and cost noise random variables $\xi _ { t } ^ { r }$ and $\xi _ { t } ^ { c }$ are conditionally sub-Gaussian. That is, for all $\lambda \in \mathbb { R }$ , there exist constants $\kappa _ { 1 } , \kappa _ { 2 } > 0$ such that

$$
\begin{array} { r l } & { \mathbb { E } [ \xi _ { t } ^ { r } \mid \mathcal { H } _ { t - 1 } ] = 0 , \qquad \mathbb { E } [ \exp ( \lambda \xi _ { t } ^ { r } ) \mid \mathcal { H } _ { t - 1 } ] \le \exp ( \lambda ^ { 2 } \kappa _ { 1 } ^ { 2 } / 2 ) , } \\ & { \mathbb { E } [ \xi _ { t } ^ { c } \mid \mathcal { H } _ { t - 1 } ] = 0 , \qquad \mathbb { E } [ \exp ( \lambda \xi _ { t } ^ { c } ) \mid \mathcal { H } _ { t - 1 } ] \le \exp ( \lambda ^ { 2 } \kappa _ { 2 } ^ { 2 } / 2 ) , } \end{array}
$$

where $\mathcal { H } _ { t } : = \sigma ( X _ { 1 } , \alpha _ { 1 } , R _ { 1 } , C _ { 1 } , \ldots , X _ { t } , \alpha _ { t } , R _ { t } , C _ { t } )$ . is the filtration containing the history up to the end of round t.

Assumption 2.2 (Bounded parameters). There exists a known constant $\kappa _ { 3 } > 0$ such that $\lVert { \boldsymbol { \theta } } ^ { \star } \rVert _ { 2 } \leq \kappa _ { 3 }$ and $\| \mu ^ { \star } \| _ { 2 } \leq \kappa _ { 3 }$

Assumption 2.3 (Bounded contexts). The ℓ<sub>2</sub>-norms of all contexts are bounded by a known constant $\kappa _ { 4 } > 0 , i . e .$ 2

$$
\operatorname* { m a x } _ { t \in [ T ] } \| X _ { t } \| _ { 2 } \leq \kappa _ { 4 } .
$$

Assumption 2.4 (Positive toxicity threshold). The toxicity threshold satisfies $\tau > 0$

## 2.2 Constraint Formulation

The learner’s goal is to maximize expected cumulative reward over $T$ rounds,

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } g ( \alpha _ { t } ) \langle X _ { t } , \theta ^ { \star } \rangle \right] ,
$$

while ensuring that the realized cost remains below a known safety threshold with high probability. Our algorithm takes as input a confidence level δ, which controls the probability of violating the constraint due to stochastic fluctuations in the cost signal. Thus, unlike formulations that constrain only the expected cost, our safety requirement depends explicitly on the tail behavior of the realized cost.

If the noise distribution were fully known, for instance Gaussian with known variance, one could potentially enforce the constraint directly using the corresponding quantiles. In our setting, however, the noise distribution is unknown and we assume only a sub-Gaussian upper bound. We therefore rely on distribution-free high-probability bounds and formulate the following suficient condition. If +k

Lemma 1. If the selected action $\alpha _ { t }$ satisfies

$$
g ( \alpha _ { t } ) \left( \langle X _ { t } , \mu ^ { \star } \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \right) \leq \tau ,
$$

then $\mathbb { P } ( C _ { t } \leq \tau \mid \mathcal { H } _ { t } ) \geq 1 - \delta$

The proof is provided in Section A.1 and follows directly from a concentration inequality for sub-Gaussian random variables; see [19], Chapter 5, or [34], Chapter 2.

The key diference from standard expected-cost formulations is conceptual rather than merely algebraic: the constraint is imposed on the realized cost, and the confidence parameter δ specifies the tolerated probability of an unsafe realization. In our action-scaled heteroscedastic model, the se lected action controls both the mean cost and the magnitude of the cost fluctuations, making this realized-cost requirement a meaningful decision constraint.

Accordingly, we define the feasible action set at round t as

$$
\begin{array} { r } { \mathcal { A } _ { t } ^ { f } ( \delta ) : = \left\{ \alpha \in [ 0 , 1 ] : g ( \alpha ) \left( \langle X _ { t } , \mu ^ { \star } \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \right) \leq \tau \right\} . } \end{array}\tag{3}
$$

Any action $\alpha _ { t } \in \mathcal { A } _ { t } ^ { f } ( \delta )$ ensures that $\mathbb { P } ( C _ { t } \leq \tau \mid \mathcal { H } _ { t } ) \geq 1 - \delta$

Given the feasible action set, let

$$
\alpha _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { \alpha \in \boldsymbol { A } _ { t } ^ { f } ( \delta ) } \boldsymbol { g } ( \alpha ) \boldsymbol { r } ( X _ { t } )
$$

denote an optimal feasible action at round $t . ^ { 2 }$ We measure the learner’s performance through the $T \cdot$ -round constrained pseudo-regret

$$
\mathcal { R } _ { \mathcal { C } } ( T ) : = \sum _ { t = 1 } ^ { T } \left[ g ( \alpha _ { t } ^ { \star } ) r ( X _ { t } ) - g ( \alpha _ { t } ) r ( X _ { t } ) \right] .\tag{4}
$$

Our algorithm guarantees that the selected action belongs to an estimated subset of $\mathcal { A } _ { t } ^ { f } ( \delta )$ with probability at least $1 - \delta ^ { \prime } .$ , where $\delta ^ { \prime }$ is the confidence level associated with estimating $\mu ^ { \star }$ . We keep $\delta$ and $\delta ^ { \prime }$ separate to distinguish two sources of uncertainty: the randomness of the realized cost, governed by the noise $\xi _ { t } ^ { c }$ , and the statistical uncertainty in the estimate of $\mu ^ { \star }$ , governed by the observed history $\mathcal { H } _ { t }$ . One may combine these events by a union bound, but we state them separately to make the two roles explicit.

We now address safety in the initial round, where no observations are available to estimate $\theta ^ { \star }$ or $\mu ^ { \star }$ In contrast to prior work [28, 24], our model does not require prior knowledge of an initially safe action. Because $\tau > 0$ and $g ( 0 ) = 0$ , the zero action is always safe. Moreover, using the boundedness assumptions on $\mu ^ { \star }$ and $X _ { t } .$ , we can characterize a nontrivial initial safe interval. Specifically, since $\langle X _ { 1 } , \mu ^ { \star } \rangle \leq \kappa _ { 3 } \kappa _ { 4 }$ by Cauchy–Schwarz, any action satisfying

$$
g ( \alpha _ { 1 } ) \leq \frac { \tau } { \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } + \kappa _ { 3 } \kappa _ { 4 } } .\tag{5}
$$

is safe. Equivalently, since g is strictly increasing, an initial safe interval is given by

$$
\mathcal { A } _ { 1 } ^ { f } ( \delta ) = \left[ 0 , g ^ { - 1 } \left( \operatorname* { m i n } \left\{ g ( 1 ) , \frac { \tau } { \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } + \kappa _ { 3 } \kappa _ { 4 } } \right\} \right) \right] .
$$

## 3 Proposed Algorithm

Our algorithm follows the principle of optimism in the face of uncertainty for the reward signal, while using pessimistic estimates for the cost signal. This asymmetric treatment encourages exploration for reward maximization while maintaining conservative estimates of the safe action set.

At each round, we construct two regularized least-squares estimators, one for $\theta ^ { \star }$ and one for $\mu ^ { \star }$ . For a regularization parameter $\lambda > 0$ , define the regularized covariance matrix at round t as

$$
\Sigma _ { t } = \lambda I + \sum _ { \stackrel { s < t } { g ( \alpha _ { s } ) > 0 } } X _ { s } X _ { s } ^ { \top } .\tag{6}
$$

Our theoretical analysis holds for any fixed constant $\lambda > 0 ;$ in our experiments, we set $\lambda = 1$ . Using Equation (6), we define

$$
\widehat { \theta } _ { t } = \Sigma _ { t } ^ { - 1 } \sum _ { \substack { s < t } } \frac { R _ { s } } { g ( \alpha _ { s } ) } X _ { s } , \qquad \widehat { \mu } _ { t } = \Sigma _ { t } ^ { - 1 } \sum _ { \substack { s < t } } \frac { C _ { s } } { g ( \alpha _ { s } ) } X _ { s } .\tag{7}
$$

Since

$$
\frac { R _ { s } } { g ( \alpha _ { s } ) } = \langle X _ { s } , \theta ^ { \star } \rangle + \xi _ { s } ^ { r } , \qquad \frac { C _ { s } } { g ( \alpha _ { s } ) } = \langle X _ { s } , \mu ^ { \star } \rangle + \xi _ { s } ^ { c } ,
$$

the transformed observations follow the standard linear regression model whenever $g ( \alpha _ { s } ) > 0$ . Rounds with $g ( \alpha _ { s } ) = 0$ provide no information about the unknown parameters and are therefore excluded.

To construct a UCB-style algorithm, we define high-probability confidence sets centered at $\widehat { \theta } _ { t }$ and $\hat { \mu } _ { t }$ These sets bound the estimation error with respect to the true parameters $\theta ^ { \star }$ and $\mu ^ { \star }$ and follow from standard self-normalized concentration inequalities. Let

$$
N ( t ) = | \{ 1 \leq s < t : g ( \alpha _ { s } ) > 0 \} |
$$

denote the number of informative samples collected before round t.

Theorem 1 (Theorem 2 in 1). For fixed $\delta ^ { \prime } \in ( 0 , 1 )$ , define

$$
\beta _ { t } ^ { r } ( \delta ^ { \prime } , d ) = \kappa _ { 1 } \sqrt { d \log \left( \frac { 1 + N ( t ) \kappa _ { 4 } ^ { 2 } / \lambda } { \delta ^ { \prime } } \right) } + \sqrt { \lambda } \kappa _ { 3 } ,
$$

$$
\beta _ { t } ^ { c } ( \delta ^ { \prime } , d ) = \kappa _ { 2 } \sqrt { d \log \left( \frac { 1 + N ( t ) \kappa _ { 4 } ^ { 2 } / \lambda } { \delta ^ { \prime } } \right) } + \sqrt { \lambda } \kappa _ { 3 } .
$$

Then, with probability at least $1 - \delta ^ { \prime }$

$$
\begin{array} { r } { \| \hat { \theta } _ { t } - \theta ^ { \star } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { r } ( \delta ^ { \prime } , d ) , \qquad \| \hat { \mu } _ { t } - \mu ^ { \star } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { c } ( \delta ^ { \prime } , d ) . } \end{array}
$$

We use Theorem 1 to define the confidence ellipsoids

$$
\begin{array} { r l } & { \mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) = \left\{ \theta \in \mathbb { R } ^ { d } : \| \theta - \hat { \theta } _ { t } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { r } ( \delta ^ { \prime } , d ) \right\} , } \\ & { \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) = \left\{ \mu \in \mathbb { R } ^ { d } : \| \mu - \hat { \mu } _ { t } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { c } ( \delta ^ { \prime } , d ) \right\} . } \end{array}\tag{8}
$$

When clear from context, we omit the dependence of $\beta _ { t } ^ { r }$ and $\beta _ { t } ^ { c }$ on $\delta ^ { \prime }$ and $d .$ By a union bound, the events $\theta ^ { \star } \in \mathcal { C } _ { t } ^ { r }$ and $\mu ^ { \star } \in \mathcal { C } _ { t } ^ { c }$ hold with probability at least $1 - 2 \delta ^ { \prime }$

We choose an optimistic estimate of the reward parameter by solving

$$
\tilde { \theta } _ { t } \in \arg \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t } ^ { r } } \langle X _ { t } , \theta \rangle .\tag{9}
$$

This is a linear optimization problem over an ellipsoid and admits a closed-form solution from the KKT conditions [10]; see Section C for details.

For the cost parameter, we take the opposite approach. We select a pessimistic estimate $\tilde { \mu } _ { t }$ that yields a conservative estimate of the feasible action set. Since the feasible set is an interval in $[ 0 , 1 ]$ , this amounts to choosing the parameter in the confidence set that makes the upper endpoint of the safe interval as small as possible. This optimistic–pessimistic principle underlies Algorithm 1.

We have stated the algorithm using the sub-Gaussian parameters $\kappa _ { 1 }$ and $\kappa _ { 2 }$ . If these parameters are unknown, they can be replaced by valid upper confidence estimates or upper bounds using standard techniques. In particular, the confidence intervals of [17] can be incorporated without changing the structure of the algorithm. Moreover, when reward and cost observations are almost surely bounded, a sub-Gaussian upper bound follows directly from standard concentration results [34].

```latex
Algorithm 1 High Probability Constrained UCB
Require: Constraint threshold $\tau > 0 .$ , safety parameter $\delta ,$ confidence parameter $\delta ^ { \prime }$ , sub-Gaussianity
constants $\kappa _ { 1 } , \kappa _ { 2 }$
1: $\begin{array} { r } { \alpha _ { 1 } \gets g ^ { - 1 } \bigg ( \operatorname* { m i n } \bigg \{ 1 , \frac { \tau } { \kappa _ { 2 } \sqrt { 2 \log ( 1 / \delta ) } + \kappa _ { 3 } \kappa _ { 4 } } \bigg \} \bigg ) } \end{array}$
2: Select action $\alpha _ { 1 }$ and store $( X _ { 1 } , R _ { 1 } , \tilde { C _ { 1 } } ) ^ { \prime }$
3: for $t = 2 , 3 , \dots , T$ do
4: Compute $\hat { \theta } _ { t } , \hat { \mu } _ { t }$ according to $( 7 )$
5: Use $\hat { \mu } _ { t }$ to solve (10) and compute $\hat { \mathcal { A } } _ { t } ^ { f }$ using (12).
6: Use $\widehat { \theta } _ { t }$ to solve (9) and compute $\tilde { \theta } _ { t }$
7: Compute
$\alpha _ { t } \in \arg \operatorname* { m a x } _ { \alpha \in \hat { \mathcal { A } } _ { t } ^ { f } } g ( \alpha ) \langle X _ { t } , \tilde { \theta } _ { t } \rangle$
8: Select action $\alpha _ { t }$
9: if $g ( \alpha _ { t } ) > 0$ then
10: Store $\left( X _ { t } , R _ { t } , C _ { t } \right)$
11: end if
12: end for
```

## 3.1 Construction of the Estimated Feasible Set

The estimated feasible set $\hat { \mathcal { A } } _ { t } ^ { f }$ is constructed from a pessimistic estimate of the cost parameter. First, we compute the least-squares estimator $\hat { \mu } _ { t }$ , which defines a confidence ellipsoid that contains $\mu ^ { \star }$ with high probability. Among all parameters in this ellipsoid, we choose the one that makes the safe action set most conservative.

More precisely, we construct a surrogate feasible set $\hat { \mathcal { A } } _ { t } ^ { f }$ such that

$$
\mathbb { P } \Big ( \hat { \mathcal { A } } _ { t } ^ { f } \subseteq \mathcal { A } _ { t } ^ { f } \Big ) \geq 1 - \delta ^ { \prime }
$$

for all t. We compute $\hat { \mu } _ { t }$ according to (7) and solve the following optimization problem:

$$
\begin{array} { r l l } & { \underset { \mu } { \operatorname* { m a x } } } & { \langle X _ { t } , \mu \rangle } \\ & { \mathrm { s u b j e c t ~ t o } } & { \| \mu - \hat { \mu } _ { t } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { c } , } \\ & { } & { \langle X _ { t } , \mu \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \displaystyle \frac { 1 } { \delta } \right) } \geq 0 . } \end{array}\tag{10}
$$

The last constraint ensures that the denominator defining the safe upper endpoint is nonnegative. Let

$$
\mathcal { K } \left( \hat { \mu } _ { t } \right) _ { t } = \left\{ \mu \in \mathbb { R } ^ { d } : \| \mu - \hat { \mu } _ { t } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { c } , \langle X _ { t } , \mu \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \geq 0 \right\} .\tag{11}
$$

If K $\left( \hat { \mu } _ { t } \right) _ { t } \neq \varnothing$ , let

$$
\tilde { \mu } _ { t } \in \arg \operatorname* { m a x } _ { \mu \in \mathcal { K } ( \hat { \mu } _ { t } ) _ { t } } \langle X _ { t } , \mu \rangle .
$$

We then define the estimated feasible set as

$$
\hat { A } _ { t } ^ { f } ( \delta ) = \left\{ \begin{array} { l l } { [ 0 , 1 ] , } & { \mathrm { i f ~ } K \left( \hat { \mu } _ { t } \right) _ { t } = 0 , } \\ { \left[ 0 , g ^ { - 1 } \left( \operatorname* { m i n } \left\{ 1 , \frac { \tau } { \left. X _ { t } , \tilde { \mu } _ { t } \right. + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } } \right\} \right) \right] , } & { \mathrm { i f ~ } K \left( \hat { \mu } _ { t } \right) _ { t } \neq 0 . } \end{array} \right.\tag{12}
$$

Combining Theorem 1 with the pessimistic construction above yields the following guarantee; its proof is provided in the supplementary material.

Lemma 2. For all $t \in [ T ]$

$$
\mathbb { P } \left( \hat { \mathcal { A } } _ { t } ^ { f } ( \delta ) \subseteq \mathcal { A } _ { t } ^ { f } ( \delta ) \mid \mathcal { H } _ { t } \right) \geq 1 - \delta ^ { \prime } .
$$

## 4 The “Good Event”

A common analysis technique in multi-armed bandit algorithms is to condition on a high-probability “good event.” We define this event as the conjunction of the concentration and feasibility guarantees required by our algorithm. Specifically, let ε denote the intersection of the following sub-events:

$$
\bullet \ \mathcal { E } _ { \theta } = \{ \theta ^ { \star } \in \mathcal { C } _ { t } ^ { r } , \forall t \in [ T ] \} .
$$

• ε<sub>µ</sub> = {µ<sup>⋆</sup> ∈ C<sup>c</sup><sub>t</sub> , ∀t ∈ [T]}.

$$
\bullet \ \mathcal { E } _ { A } = \{ \hat { A } _ { t } ^ { f } ( \delta ) \subseteq \mathcal { A } _ { t } ^ { f } ( \delta ) , \ \forall t \in [ T ] \} .
$$

The first two events ensure that the confidence sets for $\theta ^ { \star }$ and $\mu ^ { \star }$ contain the true parameters, as established in Theorem 1. The third event ensures that the estimated feasible action set is contained in the true feasible action set at every round.

Lemma 3. Let $\mathcal { E } = \mathcal { E } _ { \theta } \cap \mathcal { E } _ { \mu } \cap \mathcal { E } _ { A }$ . Then,

$$
\mathbb { P } ( \mathcal { E } ) \geq 1 - 3 \delta ^ { \prime } .
$$

## 5 Regret Analysis – Linear Case

We remind the definition of the T-round constrained pseudo-regret as defined in Equation $( 4 )$ . In the linear setting, the expected reward obtained by action $\alpha _ { t }$ is $g ( \alpha _ { t } ) \langle X _ { t } , \theta ^ { \star } \rangle$ ⟩. We therefore define

$$
\mathcal { R } _ { \mathcal { C } } ( T ) = \sum _ { t = 1 } ^ { T } \left( r ^ { * } ( X _ { t } ) - r _ { t } \right) ,\tag{13}
$$

where

$$
r ^ { * } ( X _ { t } ) = \operatorname* { m a x } _ { \alpha \in \mathcal { A } _ { t } ^ { f } } g ( \alpha ) \langle X _ { t } , \theta ^ { \star } \rangle
$$

denotes the optimal feasible reward at round t , and

$$
r _ { t } = g ( \alpha _ { t } ) \langle X _ { t } , \theta ^ { \star } \rangle
$$

is the expected reward obtained by the algorithm.

Since $g$ is strictly increasing, the optimal feasible action depends on the sign of $\left. X _ { t } , \theta ^ { \star } \right.$ . If $\left. X _ { t } , \theta ^ { \star } \right. \geq 0 .$ the optimal action is the largest feasible action. If $\left. X _ { t } , \theta ^ { \star } \right. < 0$ , the optimal action is $\alpha _ { t } = 0$ . Let $\alpha _ { t } ^ { \star }$ denote an optimal feasible action at round t.

Then

$$
\begin{array} { c l } { \mathcal { R } _ { \mathcal { C } } ( T ) = \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { \alpha \in A _ { t } ^ { f } } g ( \alpha ) \langle X _ { t } , \theta ^ { \star } \rangle - g ( \alpha _ { t } ) \langle X _ { t } , \theta ^ { \star } \rangle } \\ { \displaystyle } & { = \displaystyle \sum _ { t = 1 } ^ { T } \left( g ( \alpha _ { t } ^ { \star } ) - g ( \alpha _ { t } ) \right) \langle X _ { t } , \theta ^ { \star } \rangle . } \end{array}\tag{14}
$$

We adopt a regret decomposition similar to those commonly used in the literature on constrained linear bandits [2, 27, 28]. Define the auxiliary feasible action

$$
\tilde { \alpha } _ { t } \in \arg \operatorname* { m a x } _ { \alpha \in \mathcal { A } _ { t } ^ { f } } g ( \alpha ) \langle X _ { t } , \tilde { \theta } _ { t } \rangle ,\tag{15}
$$

where $\tilde { \theta } _ { t }$ is the optimistic reward parameter defined in Equation (9). Using this definition, we decompose the regret as

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathcal { C } } ( T ) = \displaystyle \sum _ { t = 1 } ^ { T } \left( g ( \alpha _ { t } ^ { \star } ) - g ( \alpha _ { t } ) \right) \langle X _ { t } , \theta ^ { \star } \rangle } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \left( g ( \alpha _ { t } ^ { \star } ) - g ( \tilde { \alpha } _ { t } ) \right) \langle X _ { t } , \theta ^ { \star } \rangle } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \left( g ( \tilde { \alpha } _ { t } ) - g ( \alpha _ { t } ) \right) \langle X _ { t } , \theta ^ { \star } \rangle . } \end{array}\tag{16}
$$

## 5.1 Analysis of the Regret

Lemma 4. For all $t \in [ T ]$ , on the event $\varepsilon _ { \theta }$ , the first term in the regret decomposition satisfies

$$
\begin{array} { r } { ( g ( \alpha _ { t } ^ { \star } ) - g ( \tilde { \alpha } _ { t } ) ) \langle X _ { t } , \theta ^ { \star } \rangle \leq g ( \tilde { \alpha } _ { t } ) \langle X _ { t } , \tilde { \theta } _ { t } - \theta ^ { \star } \rangle . } \end{array}
$$

The proof is provided in the supplementary material. This bound is the analogue of the standard optimism argument for linear bandits [1], applied to the transformed action value $g ( \alpha )$ . The main additional step is to control the second term, which captures the loss due to using a conservative estimate of the feasible action set.

Lemma 5. For all $t \in [ T ]$ , on the event $\varepsilon _ { \mu }$ , the second term in the regret decomposition satisfies

$$
\left( g ( \tilde { \alpha } _ { t } ) - g ( \alpha _ { t } ) \right) \left. X _ { t } , \theta ^ { \star } \right. \leq \kappa _ { 3 } \kappa _ { 4 } \cdot \frac { \left. X _ { t } , \tilde { \mu } _ { t } - \mu ^ { \star } \right. } { \tau } .
$$

By combining Lemma 4 and Lemma 5 with the regret decomposition in Equation (16), we obtain the following bound.

Theorem 2. With probability at least $1 - \delta ^ { \prime }$ , the regret of the High Probability Constrained UCB algorithm satisfies

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathcal { C } } ( T ) = \mathcal { O } \Big ( \Big ( 1 + \frac { \kappa _ { 3 } \kappa _ { 4 } } { \tau } \Big ) \cdot \operatorname* { m a x } \{ \beta _ { T } ^ { r } ( \delta ^ { \prime } , d ) , \beta _ { T } ^ { c } ( \delta ^ { \prime } , d ) \} } \\ & { \qquad \times \sqrt { d T \log \left( 1 + \frac { T \kappa _ { 4 } ^ { 2 } } { d } \right) } \Big ) . } \end{array}
$$

All proofs are deferred to the supplementary material. Notably, the bound does not depend on $\delta ,$ since the benchmark is the clairvoyant policy that knows $\theta ^ { \star }$ and $\mu ^ { \star }$ but must satisfy the same realized-cost constraint as the learner.

## 6 Non-linear Rewards and Costs

We now extend the analysis beyond linear reward and cost models. Instead of assuming that the reward and cost functions are linear in unknown parameters, we consider general function classes and express the regret bound in terms of the eluder dimension [31].

At each round t, the learner observes a context $X _ { t }$ and selects an action $\alpha _ { t } \in [ 0 , 1 ]$ . The reward and cost signals are given by

$$
R _ { t } = g ( \alpha _ { t } ) \left( \theta ^ { \star } ( X _ { t } ) + \xi _ { t } ^ { r } \right) , \qquad C _ { t } = g ( \alpha _ { t } ) \left( \mu ^ { \star } ( X _ { t } ) + \xi _ { t } ^ { c } \right) ,
$$

where $\theta ^ { \star } \in \mathcal { F } ^ { r }$ and $\mu ^ { \star } \in \mathcal { F } ^ { c }$ are the unknown mean reward and cost functions, respectively. The function classes ${ \mathcal { F } } ^ { r }$ and ${ \mathcal { F } } ^ { c }$ are known to the learner. We assume that all functions in these classes are bounded in [−1, 1]. This relaxes the common normalization to [0, 1]; the important property for the analysis is boundedness. As in the linear case, the noise variables $\xi _ { t } ^ { r }$ and $\xi _ { t } ^ { c }$ are conditionally sub-Gaussian.

The realized-cost feasible action set at round t is

$$
\begin{array} { r } { \boldsymbol A _ { t } ^ { f } ( X _ { t } ) = \left\{ \alpha \in [ 0 , 1 ] : g ( \alpha ) \left( \mu ^ { \star } ( X _ { t } ) + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \right) \leq \tau \right\} . } \end{array}\tag{17}
$$

The agent selects actions from an estimated subset of this feasible set.

For a subset ${ \mathcal { \tilde { F } } } \subseteq { \mathcal { F } }$ , we define its width at a context X as

$$
w _ { \tilde { \mathcal { F } } } ( X ) = \operatorname* { s u p } _ { \underline { { f } } , \overline { { f } } \in \widetilde { \mathcal { F } } } \left( \overline { { f } } ( X ) - \underline { { f } } ( X ) \right) .\tag{18}
$$

This quantity measures the uncertainty of the function class at the current context and plays the same role as confidence widths in the linear analysis.

The $T -$ round constrained pseudo-regret is

$$
\mathcal { R } ( T ) = \sum _ { t = 1 } ^ { T } \left[ g ( \alpha _ { t } ^ { \star } ) \theta ^ { \star } ( X _ { t } ) - g ( \alpha _ { t } ) \theta ^ { \star } ( X _ { t } ) \right] ,\tag{19}
$$

where

$$
\alpha _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { \alpha \in \mathcal { A } _ { t } ^ { f } ( X _ { t } ) } g ( \alpha ) \theta ^ { \star } ( X _ { t } )
$$

is the optimal feasible action at round t.

We define the dataset available at the beginning of round t as

$$
\mathcal { D } _ { t } = \left\{ \left( X _ { s } , \frac { R _ { s } } { g ( \alpha _ { s } ) } , \frac { C _ { s } } { g ( \alpha _ { s } ) } \right) : s < t , \ g ( \alpha _ { s } ) > 0 \right\} .
$$

Thus, whenever $g ( \alpha _ { s } ) > 0$

$$
\frac { R _ { s } } { g ( \alpha _ { s } ) } = \theta ^ { \star } ( X _ { s } ) + \xi _ { s } ^ { r } , \qquad \frac { C _ { s } } { g ( \alpha _ { s } ) } = \mu ^ { \star } ( X _ { s } ) + \xi _ { s } ^ { c } .
$$

Rounds with $g ( \alpha _ { s } ) = 0$ provide no information about the unknown functions and are excluded. For any function $f ,$ let

$$
\| f \| _ { \mathcal { D } _ { t } } = \sqrt { \sum _ { ( X _ { s } , \cdot , \cdot ) \in \mathcal { D } _ { t } } f ^ { 2 } ( X _ { s } ) }
$$

denote the empirical norm induced by the observed contexts.

At each round, we construct least-squares estimates $\widehat { \theta } _ { t }$ and $\hat { \mu } _ { t }$ over the classes ${ \mathcal { F } } ^ { r }$ and ${ \mathcal { F } } ^ { c }$ using the transformed observations in $\mathcal { D } _ { t }$ . We then define the confidence sets

$$
\mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) = \left\{ \theta \in \mathcal { F } ^ { r } : \left. \theta - \hat { \theta } _ { t } \right. _ { \mathcal { D } _ { t } } ^ { 2 } \leq \rho ^ { r } ( t , \delta ^ { \prime } ) \right\} ,\tag{20}
$$

$$
\mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) = \left\{ \mu \in \mathcal { F } ^ { c } : \| \mu - \hat { \mu } _ { t } \| _ { \mathcal { D } _ { t } } ^ { 2 } \leq \rho ^ { c } ( t , \delta ^ { \prime } ) \right\} .\tag{21}
$$

For finite function classes, one may take

$$
\rho ^ { r } ( t , \delta ^ { \prime } ) = 8 \kappa _ { 1 } ^ { 2 } \log \left( \frac { | \mathcal { F } ^ { r } | } { \delta ^ { \prime } } \right) , \qquad \rho ^ { c } ( t , \delta ^ { \prime } ) = 8 \kappa _ { 2 } ^ { 2 } \log \left( \frac { | \mathcal { F } ^ { c } | } { \delta ^ { \prime } } \right) .
$$

More generally, the regret bound below is stated in terms of the corresponding confidence radii and eluder dimensions.

To construct the estimated feasible action set, we choose the most conservative cost function in the confidence set at the current context. Specifically, we solve

$$
\begin{array} { r l } { \underset { \mu \in \mathcal { F } ^ { c } } { \operatorname* { m a x } } } & { \mu ( X _ { t } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \mu \in \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) , } \\ & { \mu ( X _ { t } ) + \kappa _ { 2 } \sqrt { 2 \log \left( \displaystyle \frac { 1 } { \delta } \right) } \geq 0 . } \end{array}\tag{22}
$$

If this optimization problem has no feasible solution, we set

$$
\hat { \mathcal { A } } _ { t } ^ { f } = [ 0 , 1 ] .
$$

Otherwise, let $\tilde { \mu } _ { t }$ denote a solution of (22). The estimated feasible set is then

$$
\hat { A } _ { t } ^ { f } = \left[ 0 , g ^ { - 1 } \left( \operatorname* { m i n } \left\{ 1 , \frac { \tau } { \tilde { \mu } _ { t } ( X _ { t } ) + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } } \right\} \right) \right] .\tag{23}
$$

For the reward function, we use optimism and select

$$
\tilde { \theta } _ { t } ( X _ { t } ) = \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) } \theta ( X _ { t } ) .
$$

The action is then chosen as

$$
\alpha _ { t } \in \arg \operatorname* { m a x } _ { \alpha \in \hat { \mathcal { A } } _ { t } ^ { f } } g ( \alpha ) \tilde { \theta } _ { t } ( X _ { t } ) .
$$

Since $g$ is strictly increasing, this optimization reduces to selecting the largest action in $\hat { \mathcal { A } } _ { t } ^ { f }$ whenever $\tilde { \theta } _ { t } ( X _ { t } ) \geq 0$ , and selecting $\alpha _ { t } = 0$ otherwise.

```latex
Algorithm 2 Non-Linear High Probability Constrained UCB
Require: Constraint threshold $\tau > 0 ,$ , safety parameter $\delta ,$ confidence parameter $\delta ^ { \prime }$ , sub-Gaussianity
constants $\kappa _ { 1 } , \kappa _ { 2 }$
1: $\begin{array} { r } { \alpha _ { 1 }  g ^ { - 1 } \bigg ( \operatorname* { m i n } \bigg \{ 1 , \frac { \tau } { \kappa _ { 2 } \sqrt { 2 \log ( 1 / \delta ) } + 1 } \bigg \} \bigg ) } \end{array}$
2: Select action $\alpha _ { 1 }$ and store $( X _ { 1 } , R _ { 1 } , { \mathrm { ~ \small ~ \mathscr ~ { ~ C ~ } ~ } } _ { 1 } )$
3: for $t = 2 , 3 , \dots , T$ do
4: Compute $\hat { \theta } _ { t } , \hat { \mu } _ { t }$ using least-squares estimators over ${ \mathcal { F } } ^ { r }$ and ${ \mathcal { F } } ^ { c } .$
5: Construct $\mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } )$ and $\mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } )$ using (20).
6: Compute $\hat { \mathcal { A } } _ { t } ^ { f }$ using (22) and (23).
7: Compute $\tilde { \theta } _ { t } ( X _ { t } ) = \mathrm { m a x } _ { \theta \in \mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) } \theta ( X _ { t } ) \ :$
8: Compute
$\alpha _ { t } \in \arg \operatorname* { m a x } _ { \alpha \in \hat { \mathcal { A } } _ { t } ^ { f } } g ( \alpha ) \tilde { \theta } _ { t } ( X _ { t } ) .$
9: Select action $\alpha _ { t } .$
10: if $g ( \alpha _ { t } ) > 0$ then
11: Store $\left( X _ { t } , R _ { t } , C _ { t } \right)$
12: end if
13: end for
```

Let $d _ { \mathrm { e l u d e r } } ^ { r }$ and $d _ { \mathrm { e l u d e r } } ^ { c }$ denote the eluder dimensions of the reward and cost function classes, respectively. The following theorem gives the regret guarantee for the non-linear setting.

Theorem 3. With probability at least $1 - \bar { \delta ^ { \prime } }$ , the regret of the Non-Linear High Probability Constrained UCB algorithm satisfies

$$
\begin{array} { r l } & { \mathcal { R } ( T ) = \mathcal { O } \Big ( \sqrt { d _ { \mathrm { e l u d e r } } ^ { r } T \rho ^ { r } ( T , \delta ^ { \prime } / 3 ) } } \\ & { \quad \quad \quad + \frac { 1 } { \tau } \sqrt { d _ { \mathrm { e l u d e r } } ^ { c } T \rho ^ { c } ( T , \delta ^ { \prime } / 3 ) } \Big ) . } \end{array}
$$

## 7 Experimental Results

We evaluate the performance of Algorithm 1 on both synthetic and real-world datasets. A natural application of our algorithm is dosage selection, where the action corresponds to the administered treatment level. However, access to large-scale medical datasets with rich patient context is limited, and we leave a thorough empirical study in clinical settings for future work. Instead, we assess the performance of our method on the NASA Li-ion Battery Aging Datasets. In this setting, we interpret the current level as the dosage, the voltage change as the reward, and the temperature as the cost.

We first report in Table 1 the violation ratio of HPUCB and compare it with a baseline algorithm that enforces the cost constraint only in expectation, as in [28]. We observe that HPUCB achieves a violation ratio of 0.02%, whereas the expected-cost baseline exhibits violation rates between 12% and 23%. Such violation rates are undesirable in safety-critical applications, where individual unsafe outcomes may be unacceptable.

We begin with synthetic experiments. We generate $\theta ^ { * }$ and $\mu ^ { * }$ by sampling from a d-dimensional standard normal distribution and normalizing them to unit norm. Contexts $X _ { t }$ are generated similarly from a multivariate normal distribution and normalized. We consider dimensions $d \in \{ 5 , 1 0 \}$ and multiple threshold values $\tau ,$ and run each configuration for $1 0 ^ { 4 }$ rounds over 5 random seeds. In the linear case, we use the closed-form solutions derived for Equations (9) and (10), avoiding the need to solve convex programs at each round. The total running time across all seeds is below two minutes for all thresholds. All experiments were run on a MacBook Air with an Apple M3 chip and 24 GB RAM using the JAX framework. The dominant computational cost comes from Cholesky decompositions, which scale as $\mathcal { O } ( d ^ { 3 } )$ . JAX substantially accelerates these linear algebra operations, especially when executed on a GPU. This becomes particularly relevant in the real-world experiment, where the feature dimension is 2048.

Table 1: Comparison of HP-UCB and On-Expectation across diferent values of τ, with 95% confidence intervals.
<table><tr><td>τ</td><td></td><td>HP-UCB (95% CI)</td><td></td><td>On-Expectation (95% CI)</td></tr><tr><td>0.1</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.2392, [0.2354, 0.2430]</td></tr><tr><td>0.2</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.2313, [0.2272, 0.2355]</td></tr><tr><td>0.3</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.2224, [0.2179, 0.2269]</td></tr><tr><td>0.4</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.2126, [0.2077, 0.2176]</td></tr><tr><td>0.5</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.2016, [0.1964, 0.2067]</td></tr><tr><td>0.6</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.1890, [0.1834, 0.1946]</td></tr><tr><td>0.7</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.1763, [0.1705, 0.1821]</td></tr><tr><td>0.8</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.1621, [0.1563, 0.1678]</td></tr><tr><td>0.9</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.1472, [0.1419, 0.1525]</td></tr><tr><td>1.0</td><td></td><td>0.0002, [0.0001, 0.0002]</td><td></td><td>0.1324, [0.1274, 0.1374]</td></tr></table>

In Figure 2, we report representative learning curves for $d \in \{ 5 , 1 0 \}$ with threshold $\tau = 0 . 5$ . We compare our algorithm with a variant of ε-Greedy that plays the safe action $\alpha _ { 1 } .$ , as defined in Equation (5), with probability ε, and otherwise follows the policy of Algorithm 1. The main advantage of this variant is its lower computational cost, especially in higher dimensions, while still collecting approximately $\varepsilon T$ informative samples that help tighten its confidence intervals. Empirically, we find that $\varepsilon = 0 . 5$ provides a favorable trade-of between computational eficiency and statistical performance, achieving comparable regret while avoiding dependence on T in the parameter tuning.

A direct comparison with the method of [28] is not entirely appropriate, as their algorithm enforces the cost constraint only in expectation, whereas our setting requires high-probability constraint satisfaction for the realized cost. As shown in Table 1, this diference leads to substantially higher violation rates for the expected-cost baseline.

![](images/c80aac9cc87ee1b4e5f5dae62b11ff29a8af6d6732900764c04aea79166ca06d.jpg)  
(a) $d = 5 , \tau = 0 . 5$

![](images/f7e09c4c06d071ca4a251168eba57a880d0bf3781e760bd2a82db3bc1b181a5d.jpg)  
(b) d = 10, τ = 0.5  
Figure 2: Synthetic experiments comparing HPUCB and ε-Greedy for $\tau = 0 . 5$

We now describe the experimental setup for the NASA Li-ion Battery Aging Datasets. The dataset contains measurements from four types of batteries (RW9, RW10, RW11, RW12) operating under both charging and discharging modes, across a range of current levels. For each cycle, the dataset records the voltage change, temperature, textual metadata describing the operating conditions, and a time series corresponding to the duration of the cycle.

To construct contextual features, we use embeddings generated by the language model Gemma [32]. At each round, the input to the language model consists of all available information from the previous five time steps, formatted as a JSON object. The resulting embedding vector has dimension 2048 and serves as the context $X _ { t }$

To satisfy the realizability assumption and obtain ground-truth parameters $\theta ^ { * }$ and $\mu ^ { * }$ , we split the dataset into four parts and use approximately half of the data, about 100k samples with dimension 2048 + 3, to perform a least-squares fit. The fitted parameters are then treated as the true underlying model, and we evaluate the performance of Algorithm 1 on the remaining portion of the dataset.

Due to the high dimensionality of the problem, all real-world experiments were conducted on a compute cluster equipped with an NVIDIA L40 GPU. The total running time was approximately 30 minutes for HPUCB and about two minutes for the baseline algorithm, which has access to the true reward and cost parameters. We report the cumulative regret of HPUCB and ε-Greedy for threshold values τ ∈ {0.1, 0.2}.

![](images/8d950170907ceafd45fca59015f6960c08a72dea767e95b1e64abe25aa14715c.jpg)  
(a) $\tau = 0 . 1$

![](images/d905ec00301cc789b4704c5264cabe616013044031f1f2d5c315236748136c1f.jpg)  
(b) τ = 0.2  
Figure 3: NASA Li-ion Battery Aging Dataset experiments for diferent safety thresholds.

Finally, we note that the guarantees of our algorithm hold for arbitrary context sequences $X _ { t } ,$ including both stochastic and adversarial settings. This property makes the proposed approach particularly appealing for applications such as battery charging and control problems, where an explicit dynamical model of the system may be unavailable or dificult to specify. In such scenarios, our algorithm provides a principled way to perform safe decision-making directly from observed data.

## 8 Conclusions

We studied contextual bandits with continuous actions under stage-wise high-probability constraints on the realized cost, motivated by applications in which the variability of the outcome depends on the selected action. We introduced an optimistic–pessimistic algorithm that balances reward exploration with conservative estimation of the feasible action set, and established regret guarantees for both linear and general reward–cost models. Our experimental results further illustrate the importance of accounting for heteroscedasticity when safety is imposed on realized, rather than expected, costs.

Several directions remain open. An important extension is to consider more general forms of heteroscedasticity beyond the action-scaled structure studied here, as well as Bayesian approaches to the same realized-cost constrained problem. Another promising direction is a more extensive empirical evaluation in safety-critical real-world applications, particularly adaptive clinical dosage selection.

## References

[1] Yasin Abbasi-Yadkori, D´avid P´al, and Csaba Szepesv´ari. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011. 2, 7, and 10

[2] Sanae Amani, Mahnoosh Alizadeh, and Christos Thrampoulidis. Linear stochastic bandits under safety constraints. Advances in Neural Information Processing Systems, 32, 2019. 1, 2, 5, and 10

[3] Peter Auer. Using confidence bounds for exploitation-exploration trade-ofs. Journal of Machine Learning Research, 3(Nov):397–422, 2002. 1 and 2

[4] Maryam Aziz, Emilie Kaufmann, and Marie-Karelle Riviere. On multi-armed bandit designs for dose-finding trials. Journal of Machine Learning Research, 22(14):1–38, 2021. 1 and 3

[5] Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. Journal of the ACM (JACM), 65(3):1–55, 2018. 1

[6] Ali Baheri and Cecilia O Alm. Llms-augmented contextual bandit. arXiv preprint arXiv:2311.02268, 2023. 3

[7] Hamsa Bastani and Mohsen Bayati. Online decision making with high-dimensional covariates. Operations Research, 68(1):276–294, 2020. 1

[8] Djallel Bounefouf and Rapha¨el F´eraud. A tutorial on multi-armed bandit applications for large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 6412–6413, 2024. 1

[9] Djallel Bounefouf and Irina Rish. A survey on practical applications of multi-armed and contextual bandits. arXiv preprint arXiv:1904.10040, 2019. 1

[10] Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004. 7 and 22

[11] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 1

[12] Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773, 2023. 1

[13] Dylan J Foster and Alexander Rakhlin. Foundations of reinforcement learning and interactive decision making. arXiv preprint arXiv:2312.16730, 2023. 1

[14] Ping-Chun Hsieh, Xi Liu, Anirban Bhattacharya, and PR Kumar. Stay with me: Lifetime maximization through heteroscedastic linear bandits with reneging. In International Conference on Machine Learning, pages 2800–2809. PMLR, 2019. 4

[15] Spencer Hutchinson, Berkay Turan, and Mahnoosh Alizadeh. Directional optimism for safe linear bandits. In International Conference on Artificial Intelligence and Statistics, pages 658–666. PMLR, 2024. 2 and 4

[16] Matthew Joseph, Michael Kearns, Jamie H Morgenstern, and Aaron Roth. Fairness in learning: Classic and contextual bandits. Advances in neural information processing systems, 29, 2016. 1

[17] Kwang-Sung Jun and Jungtaek Kim. Noise-adaptive confidence sets for linear bandits and application to bayesian optimization. arXiv preprint arXiv:2402.07341, 2024. 7

[18] Emmanuel Dodzi Kpeglo, Michael Jackson Adjabui, and Jakperik Dioggban. Identification of minimum efective dose based on ratio of a normally distributed data under heteroscedasticity. Afrika Statistika, 16(2):2719–2731, 2021. 2

[19] Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020. 1, 2, 5, and 20

[20] Christophe Le Tourneau, J Jack Lee, and Lillian L Siu. Dose escalation methods in phase i cancer clinical trials. JNCI: Journal of the National Cancer Institute, 101(10):708–720, 2009. 3 and 4

[21] Hyun-Suk Lee, Cong Shen, James Jordon, and Mihaela Schaar. Contextual constrained learning for dose-finding clinical trials. In International Conference on Artificial Intelligence and Statistics, pages 2645–2654. PMLR, 2020. 1 and 3

[22] Maryam Majzoubi, Chicheng Zhang, Rajan Chari, Akshay Krishnamurthy, John Langford, and Aleksandrs Slivkins. Eficient contextual bandits with continuous actions. Advances in Neural Information Processing Systems, 33:349–360, 2020. 2 and 3

[23] Kentaro Matsuura, Junya Honda, Imad El Hanafi, Takashi Sozu, and Kentaro Sakamaki. Optimal adaptive allocation using deep reinforcement learning in a dose-response study. Statistics in Medicine, 41(7):1157–1171, 2022. 1 and 3

[24] Ahmadreza Moradipari, Sanae Amani, Mahnoosh Alizadeh, and Christos Thrampoulidis. Safe linear thompson sampling. arXiv preprint arXiv:1911.02156, 2019. 2, 4, and 6

[25] Jonas W Mueller, Vasilis Syrgkanis, and Matt Taddy. Low-rank bandit methods for highdimensional dynamic pricing. Advances in Neural Information Processing Systems, 32, 2019. 1

[26] Subhojyoti Mukherjee, Qiaomin Xie, Josiah P Hanna, and Robert Nowak. Speed: Experimental design for policy evaluation in linear heteroscedastic bandits. In International Conference on Artificial Intelligence and Statistics, pages 2962–2970. PMLR, 2024. 4

[27] Aldo Pacchiano, Mohammad Ghavamzadeh, Peter Bartlett, and Heinrich Jiang. Stochastic bandits with linear constraints. In International conference on artificial intelligence and statistics, pages 2827–2835. PMLR, 2021. 1 and 10

[28] Aldo Pacchiano, Mohammad Ghavamzadeh, and Peter Bartlett. Contextual bandits with stagewise constraints. arXiv preprint arXiv:2401.08016, 2024. 1, 2, 4, 5, 6, 10, 13, and 14

[29] Ariel D Procaccia, Benjamin Schifer, and Shirley Zhang. Honor among bandits: No-regret learning for online fair division. arXiv preprint arXiv:2407.01795, 2024. 1

[30] Michael D Rawlins. Variability in response to drugs. Br Med J, 4(5936):91–94, 1974. 4

[31] Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems, 26, 2013. 2 and 10

[32] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi\`ere, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024. 15

[33] William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3-4):285–294, 1933. 1

[34] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018. 5 and 7

[35] Sof´ıa S Villar, Jack Bowden, and James Wason. Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges. Statistical science: a review journal of the Institute of Mathematical Statistics, 30(2):199, 2015. 1

[36] Akifumi Wachi and Yanan Sui. Safe reinforcement learning in constrained markov decision processes. In International Conference on Machine Learning, pages 9797–9806. PMLR, 2020. 1

[37] Nolan A Wages, Mark R Conaway, and John O’Quigley. Dose-finding design for multi-drug combinations. Clinical Trials, 8(4):380–389, 2011. 4

[38] Hao Wang, Yifei Ma, Hao Ding, and Yuyang Wang. Context uncertainty in contextual bandits with applications to recommender systems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 8539–8547, 2022. 1

[39] Justin Weltz, Tanner Fiez, Alexander Volfovsky, Eric Laber, Blake Mason, Lalit Jain, et al. Experimental designs for heteroskedastic variance. Advances in Neural Information Processing Systems, 36:65967–66005, 2023. 4

[40] Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit q\*-approximation for sample-eficient rlhf. arXiv preprint arXiv:2405.21046, 2024. 1

[41] Ruitu Xu, Yifei Min, and Tianhao Wang. Noise-adaptive thompson sampling for linear contextual bandits. Advances in Neural Information Processing Systems, 36:23630–23657, 2023. 4

## Appendix

## Contents

1 Introduction 1   
2 Model and Problem Formulation 3   
2.1 Assumptions 4   
2.2 Constraint Formulation 5   
3 Proposed Algorithm 6   
3.1 Construction of the Estimated Feasible Set 8   
4 The “Good Event” 9   
5 Regret Analysis – Linear Case 9   
5.1 Analysis of the Regret . 10   
6 Non-linear Rewards and Costs 10   
7 Experimental Results 13   
8 Conclusions 15   
Supplementary Material 19   
A Constraint Formulation 20   
A.1 Proof of Lemma 1 20   
A.2 Proof of Lemma 2 20   
B Good Event 21   
B.1 Proof of Lemma 3 21   
C Analytical Solutions to Equations (9) and (10) 22   
D Regret Analysis – Linear Case 23   
D.1 Proof of Lemma 4 23   
D.2 Proof of Lemma 5 24   
D.3 Proof of Theorem 2 25   
E Non-linear Rewards and Costs 26   
E.1 Concentration for Non-linear Least Squares 26   
E.2 Non-linear Regret Analysis 28   
E.3 Proof of Theorem 3 . 30

## A Constraint Formulation

## A.1 Proof of Lemma 1

By Assumption 2.1, the cost noise $\xi _ { t } ^ { c }$ is conditionally κ<sub>2</sub>-sub-Gaussian given the history $\mathcal { H } _ { t }$ . Let $z _ { t } : = g ( \alpha _ { t } )$ . If $z _ { t } = 0$ , then $C _ { t } = 0$ , and since $\tau > 0$ the constraint is satisfied trivially. We therefore consider the case $z _ { t } > 0$

Theorem 4 (Sub-Gaussian concentration bounds – Theorem 5.3 in [19]). If X is σ-sub-Gaussian, then for any $\epsilon > 0$

$$
\mathbb { P } ( X \geq \epsilon ) \leq \exp \left( - \frac { \epsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .
$$

Lemma 1. If the selected action $\alpha _ { t }$ satisfies

$$
g ( \alpha _ { t } ) \left( \langle X _ { t } , \mu ^ { \star } \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \right) \leq \tau ,
$$

then $\mathbb { P } ( C _ { t } \leq \tau \mid \mathcal { H } _ { t } ) \geq 1 - \delta$

Proof. Since

$$
C _ { t } = z _ { t } \left( \langle X _ { t } , \mu ^ { \star } \rangle + \xi _ { t } ^ { c } \right) ,
$$

we have

$$
\frac { C _ { t } - z _ { t } \langle X _ { t } , \mu ^ { \star } \rangle } { z _ { t } } = \xi _ { t } ^ { c } .
$$

Thus, conditional on $\mathcal { H } _ { t }$ ,

$$
\begin{array} { r l r } & { } & { \mathbb { P } ( C _ { t } \geq \tau \mid \mathcal { H } _ { t } ) = \mathbb { P } \left( \xi _ { t } ^ { c } \geq \frac { \tau - z _ { t } \langle X _ { t } , \mu ^ { \star } \rangle } { z _ { t } } \Bigg \vert \mathcal { H } _ { t } \right) } \\ & { } & { \leq \exp \left( - \frac { \left( \frac { \tau - z _ { t } \langle X _ { t } , \mu ^ { \star } \rangle } { z _ { t } } \right) ^ { 2 } } { 2 \kappa _ { 2 } ^ { 2 } } \right) , } \end{array}
$$

where the last step follows from Theorem 4. Therefore, it is suficient to require

$$
\kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \leq \frac { \tau - z _ { t } \langle X _ { t } , \mu ^ { \star } \rangle } { z _ { t } } ,
$$

which is equivalent to

$$
g ( \alpha _ { t } ) \left( \langle X _ { t } , \mu ^ { \star } \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } \right) \leq \tau .
$$

Under this condition, $\mathbb { P } ( C _ { t } \leq \tau \mid \mathcal { H } _ { t } ) \geq 1 - \delta .$

## A.2 Proof of Lemma 2

Lemma 2. For all $t \in [ T ]$ ，

$$
\mathbb { P } \left( \hat { \mathcal { A } } _ { t } ^ { f } ( \delta ) \subseteq \mathcal { A } _ { t } ^ { f } ( \delta ) \mid \mathcal { H } _ { t } \right) \geq 1 - \delta ^ { \prime } .
$$

Proof. Let

$$
B _ { t } ( \mu ) : = \langle X _ { t } , \mu \rangle + \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } .
$$

On the event $\{ \mu ^ { \star } \in \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) \}$ , which holds with probability at least $1 - \delta ^ { \prime }$ by Theorem 1, we show that

$$
{ \hat { \mathcal { A } } } _ { t } ^ { f } ( \delta ) \subseteq { \mathcal { A } } _ { t } ^ { f } ( \delta ) .
$$

We consider two cases.

Case 1: $\mathcal { K } \left( \hat { \mu } _ { t } \right) _ { t } = \varnothing$ . Since $\mu ^ { \star } \in { \mathcal { C } } _ { t } ^ { c } ( \delta ^ { \prime } )$ , the only way the feasible set of Equation (10) can be empty is if

$$
B _ { t } ( \mu ^ { \star } ) < 0 .
$$

Then for every $\alpha \in [ 0 , 1 ]$

$$
g ( \alpha ) B _ { t } ( \mu ^ { \star } ) \leq 0 \leq \tau .
$$

Hence $\boldsymbol { \mathcal { A } } _ { t } ^ { f } ( \delta ) = [ 0 , 1 ]$ . By construction, $\hat { \mathcal { A } } _ { t } ^ { f } ( \delta ) = [ 0 , 1 ]$ as well, so the inclusion holds.

Case 2: $\boldsymbol { \mathcal { K } } \left( \hat { \mu } _ { t } \right) _ { t } \neq \boldsymbol { \emptyset }$ . Let $\tilde { \mu } _ { t }$ be the solution of Equation (10). If $B _ { t } ( \mu ^ { \star } ) < 0$ , then again $\boldsymbol { \mathcal { A } } _ { t } ^ { f } ( \delta ) = [ 0 , 1 ]$ and the inclusion is immediate.

It remains to consider the case $B _ { t } ( \mu ^ { \star } ) \geq 0$ . Since $\mu ^ { \star } \in \mathcal { K } \left( \hat { \mu } _ { t } \right) _ { t }$ and $\tilde { \mu } _ { t }$ maximizes $\langle X _ { t } , \mu \rangle$ over K $\left( \hat { \mu } _ { t } \right) _ { t }$ we have

$$
\langle X _ { t } , \mu ^ { \star } \rangle \leq \langle X _ { t } , \tilde { \mu } _ { t } \rangle ,
$$

and therefore

$$
B _ { t } ( \mu ^ { \star } ) \leq B _ { t } ( \tilde { \mu } _ { t } ) .
$$

Now take any $\alpha \in \hat { \mathcal { A } } _ { t } ^ { f } ( \delta )$ . By definition,

$$
g ( \alpha ) B _ { t } ( { \tilde { \mu } } _ { t } ) \leq \tau .
$$

Since $g ( \alpha ) \geq 0$ and $B _ { t } ( \mu ^ { \star } ) \leq B _ { t } ( \tilde { \mu } _ { t } )$ , it follows that

$$
g ( \alpha ) B _ { t } ( \mu ^ { \star } ) \leq \tau .
$$

Thus $\alpha \in \mathcal { A } _ { t } ^ { f } ( \delta )$ . Therefore,

$$
{ \hat { \mathcal { A } } } _ { t } ^ { f } ( \delta ) \subseteq { \mathcal { A } } _ { t } ^ { f } ( \delta ) .
$$

Conditioning on $\{ \mu ^ { \star } \in \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) \}$ gives the desired probability bound.

## B Good Event

## B.1 Proof of Lemma 3

Lemma 3. Let $\mathcal { E } = \mathcal { E } _ { \theta } \cap \mathcal { E } _ { \mu } \cap \mathcal { E } _ { A }$ . Then,

$$
\mathbb { P } ( \mathcal { E } ) \geq 1 - 3 \delta ^ { \prime } .
$$

Proof. We prove that each component of the good event holds with probability at least $1 - \delta ^ { \prime }$ , and then apply a union bound. By Theorem 1,

$$
\mathbb { P } ( \mathcal { E } _ { \theta } ) \geq 1 - \delta ^ { \prime } , \qquad \mathbb { P } ( \mathcal { E } _ { \mu } ) \geq 1 - \delta ^ { \prime } .
$$

Moreover, Lemma 2 gives

$$
\mathbb { P } ( \mathcal { E } _ { A } ) \geq 1 - \delta ^ { \prime } .
$$

Therefore,

$$
\mathbb { P } ( \mathcal { E } ) = \mathbb { P } ( \mathcal { E } _ { \theta } \cap \mathcal { E } _ { \mu } \cap \mathcal { E } _ { A } ) \geq 1 - 3 \delta ^ { \prime } ,
$$

as claimed.

## C Analytical Solutions to Equations (9) and (10)

In this section, we derive closed-form solutions to Equations (9) and (10) in the linear case using the Karush–Kuhn–Tucker (KKT) conditions [10]. This avoids solving a convex program at each round and substantially reduces the computational cost of the algorithm.

Lemma 6 (Linear maximization over an ellipsoid). Let $c \in \mathbb { R } ^ { d }$ with $c \neq 0$ , let $\Sigma \in \mathbb { R } ^ { d \times d }$ be symmetric positive definite, and let $\beta > 0$ . Consider

$$
\operatorname* { m a x } _ { \boldsymbol { x } \in \mathbb { R } ^ { d } } ~ \langle \boldsymbol { c } , \boldsymbol { x } \rangle ~ s u b j e c t ~ t o ~ \boldsymbol { x } ^ { \top } \Sigma \boldsymbol { x } \leq \beta ^ { 2 } .
$$

Then the unique optimal solution is

$$
x ^ { \star } = \beta \frac { \Sigma ^ { - 1 } c } { \| c \| _ { \Sigma ^ { - 1 } } } ,
$$

and the optimal value is

$$
\operatorname* { m a x } _ { x ^ { \top } \Sigma x \leq \beta ^ { 2 } } \langle c , x \rangle = \beta \left\| c \right\| _ { \Sigma ^ { - 1 } } .
$$

Proof. Since $\Sigma \succ 0$ , the feasible set is nonempty, compact, and convex. The objective is continuous and linear, so an optimum exists. The KKT conditions are necessary and suficient because Slater’s condition holds. The Lagrangian is

$$
{ \mathcal { L } } ( x , \lambda ) = \langle c , x \rangle - \lambda ( x ^ { \top } \Sigma x - \beta ^ { 2 } ) , \qquad \lambda \geq 0 .
$$

Stationarity gives

$$
c - 2 \lambda \Sigma x = 0 , \qquad \mathrm { h e n c e } \qquad x = { \frac { 1 } { 2 \lambda } } \Sigma ^ { - 1 } c .
$$

Since $c \neq 0$ , the optimum lies on the boundary, so $x ^ { \top } \Sigma x = \beta ^ { 2 }$ . Substituting the expression for x gives

$$
\beta ^ { 2 } = \frac { 1 } { 4 \lambda ^ { 2 } } c ^ { \top } \Sigma ^ { - 1 } c .
$$

Thus

$$
\lambda = \frac { 1 } { 2 \beta } \sqrt { c ^ { \top } \Sigma ^ { - 1 } c } .
$$

Substituting back yields

$$
x ^ { \star } = \beta \frac { \Sigma ^ { - 1 } c } { \sqrt { c ^ { \top } \Sigma ^ { - 1 } c } } = \beta \frac { \Sigma ^ { - 1 } c } { \| c \| _ { \Sigma ^ { - 1 } } } .
$$

The optimal value follows immediately. Uniqueness follows from strict convexity of the ellipsoid boundary in the direction of the nonzero linear functional. □

Proposition 1 (Closed form for $\tilde { \theta } _ { t } )$ . The solution of Equation (9) is uniquely given $b y$

$$
\widetilde { \theta } _ { t } = \widehat { \theta } _ { t } + \beta _ { t } ^ { r } ( \delta ^ { \prime } , d ) \frac { \Sigma _ { t } ^ { - 1 } X _ { t } } { \Vert X _ { t } \Vert _ { \Sigma _ { t } ^ { - 1 } } } .
$$

Moreover,

$$
\operatorname* { m a x } _ { \theta \in \mathscr { C } _ { t } ^ { r } ( \delta ^ { \prime } ) } \langle X _ { t } , \theta \rangle = \langle X _ { t } , \hat { \theta } _ { t } \rangle + \beta _ { t } ^ { r } ( \delta ^ { \prime } , d ) \left. X _ { t } \right. _ { \Sigma _ { t } ^ { - 1 } } .
$$

Proof. Apply Lemma 6 with $c = X _ { t }$ and $x = \theta - \hat { \theta } _ { t }$

Proposition 2 (Closed form for $\tilde { \mu } _ { t } )$ . The solution of

$$
\operatorname* { m a x } _ { \mu \in \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) } \langle X _ { t } , \mu \rangle
$$

is uniquely given by

$$
\tilde { \mu } _ { t } = \hat { \mu } _ { t } + \beta _ { t } ^ { c } ( \delta ^ { \prime } , d ) \frac { \Sigma _ { t } ^ { - 1 } X _ { t } } { \| X _ { t } \| _ { \Sigma _ { t } ^ { - 1 } } } .
$$

Moreover,

$$
\operatorname* { m a x } _ { \mu \in \mathscr { C } _ { t } ^ { c } ( \delta ^ { \prime } ) } \langle X _ { t } , \mu \rangle = \langle X _ { t } , \hat { \mu } _ { t } \rangle + \beta _ { t } ^ { c } ( \delta ^ { \prime } , d ) \left\| X _ { t } \right\| _ { \Sigma _ { t } ^ { - 1 } } .
$$

Proof. Apply Lemma 6 with $c = X _ { t }$ and $x = \mu - \hat { \mu } _ { t }$

Lemma 7 (Feasibility of the pessimistic cost program). Let

$$
\mathcal { C } _ { t } ^ { c } = \left\{ \mu \in \mathbb { R } ^ { d } : \| \mu - \hat { \mu } _ { t } \| _ { \Sigma _ { t } } \leq \beta _ { t } ^ { c } \right\} ,
$$

and define

$$
b _ { \delta } = \kappa _ { 2 } \sqrt { 2 \log { \left( \frac { 1 } { \delta } \right) } } .
$$

Consider

$$
\operatorname* { m a x } _ { \mu \in \mathcal { C } _ { t } ^ { c } } \langle X _ { t } , \mu \rangle \qquad s u b j e c t \ t o \qquad \langle X _ { t } , \mu \rangle + b _ { \delta } \geq 0 .
$$

Let

$$
\bar { v } _ { t } = \operatorname* { m a x } _ { \mu \in \mathcal { C } _ { t } ^ { c } } \langle X _ { t } , \mu \rangle = \langle X _ { t } , \hat { \mu } _ { t } \rangle + \beta _ { t } ^ { c } \left\| X _ { t } \right\| _ { \Sigma _ { t } ^ { - 1 } } .
$$

Then the program is infeasible $i f$ and only if $\mathit { \bar { v } } _ { t } + b _ { \delta } < 0 . \mathit { \Delta } I f \mathit { \bar { v } } _ { t } + b _ { \delta } \geq 0$ , then the unconstrained maximizer over $\mathcal { C } _ { t } ^ { c }$ is feasible and is also the optimizer of the constrained problem.

Proof. If $\bar { v } _ { t } + b _ { \delta } < 0$ , then for every $\mu \in \mathcal { C } _ { t } ^ { c }$

$$
\left. X _ { t } , \mu \right. + b _ { \delta } \leq \bar { v } _ { t } + b _ { \delta } < 0 .
$$

Thus the constraint cannot be satisfied, and the program is infeasible.

If $\bar { v } _ { t } + b _ { \delta } \geq 0$ , then the unconstrained maximizer satisfies the additional constraint. Since it already maximizes $\langle X _ { t } , \mu \rangle$ over the larger set $\mathcal { C } _ { t } ^ { c }$ , it also maximizes the same objective over the constrained feasible subset. □

## D Regret Analysis – Linear Case

As shown in Section 5, we decompose the regret into two terms: the cost of estimating $\theta ^ { \star }$ and the cost of estimating $\mu ^ { \star }$ . We now bound these terms in terms of the problem parameters.

## D.1 Proof of Lemma 4

Lemma 4. For all $t \in [ T ]$ , on the event $\varepsilon _ { \theta . }$ , the first term in the regret decomposition satisfies

$$
\begin{array} { r } { ( g ( \alpha _ { t } ^ { \star } ) - g ( \tilde { \alpha } _ { t } ) ) \langle X _ { t } , \theta ^ { \star } \rangle \leq g ( \tilde { \alpha } _ { t } ) \langle X _ { t } , \tilde { \theta } _ { t } - \theta ^ { \star } \rangle . } \end{array}
$$

Proof. Condition on the event $\varepsilon _ { \theta } .$ , so that $\theta ^ { \star } \in \mathcal { C } _ { t } ^ { r }$ for all $t \in [ T ]$ . Since $g ( \alpha _ { t } ^ { \star } ) \geq 0$ , we have

$$
\begin{array} { r l r } {  { g ( \alpha _ { t } ^ { \star } ) \langle X _ { t } , \theta ^ { \star } \rangle \le \underset { \theta \in \mathcal { C } _ { t } ^ { r } } { \operatorname* { m a x } } g ( \alpha _ { t } ^ { \star } ) \langle X _ { t } , \theta \rangle } } \\ & { } & { \le \underset { \alpha \in A _ { t } ^ { f } } { \operatorname* { m a x } } \operatorname* { m a x } g ( \alpha ) \langle X _ { t } , \theta \rangle } \\ & { } & { = \underset { \alpha \in A _ { t } ^ { f } } { \operatorname* { m a x } } g ( \alpha ) \langle X _ { t } , \tilde { \theta } _ { t } \rangle } \\ & { } & { = g ( \tilde { \alpha } _ { t } ) \langle X _ { t } , \tilde { \theta } _ { t } \rangle . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { ( g ( \alpha _ { t } ^ { \star } ) - g ( \tilde { \alpha } _ { t } ) ) \langle X _ { t } , \theta ^ { \star } \rangle \leq g ( \tilde { \alpha } _ { t } ) \langle X _ { t } , \tilde { \theta } _ { t } - \theta ^ { \star } \rangle . } \end{array}
$$

## D.2 Proof of Lemma 5

Lemma 5. For all $t \in [ T ]$ , on the event $\varepsilon _ { \mu } .$ , the second term in the regret decomposition satisfies

$$
\left( g ( \tilde { \alpha } _ { t } ) - g ( \alpha _ { t } ) \right) \left. X _ { t } , \theta ^ { \star } \right. \leq \kappa _ { 3 } \kappa _ { 4 } \cdot \frac { \left. X _ { t } , \tilde { \mu } _ { t } - \mu ^ { \star } \right. } { \tau } .
$$

Proof. Let

$$
b _ { \delta } = \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } , \qquad B _ { t } ^ { \star } = \langle X _ { t } , \mu ^ { \star } \rangle + b _ { \delta } , \qquad \widetilde { B } _ { t } = \langle X _ { t } , \tilde { \mu } _ { t } \rangle + b _ { \delta } .
$$

On the event $\varepsilon _ { \mu } .$ , the pessimistic construction gives

$$
B _ { t } ^ { \star } \le \widetilde B _ { t }
$$

whenever $B _ { t } ^ { \star } \geq 0$ . If $B _ { t } ^ { \star } < 0$ , then the true feasible set is all of [0, 1].

Since g is strictly increasing, maximizing

$$
g ( \alpha ) \langle X _ { t } , \tilde { \theta } _ { t } \rangle
$$

over an interval is equivalent to choosing the largest feasible action when $\langle X _ { t } , \tilde { { \theta } } _ { t } \rangle \geq 0$ , and choosing $\alpha = 0$ otherwise. If $\langle X _ { t } , \tilde { { \theta } } _ { t } \rangle < 0$ , then both $\tilde { \alpha } _ { t }$ and $\alpha _ { t }$ are zero, and the claim is immediate.

Assume now that $\langle X _ { t } , \tilde { { \theta } } _ { t } \rangle \geq 0$ . Define the efective actions

$$
\tilde { z } _ { t } : = g ( \tilde { \alpha } _ { t } ) , \qquad z _ { t } : = g ( \alpha _ { t } ) .
$$

The true and estimated feasible sets in the efective action variable are intervals with upper endpoints

$$
\bar { z } _ { t } ^ { \star } = \left\{ \begin{array} { l l } { 1 , } & { B _ { t } ^ { \star } \le 0 , } \\ { \operatorname* { m i n } \left\{ 1 , \displaystyle \frac { \tau } { B _ { t } ^ { \star } } \right\} , } & { B _ { t } ^ { \star } > 0 , } \end{array} \right.
$$

and

$$
\widehat { z } _ { t } = \left\{ \begin{array} { l l } { 1 , } & { \widetilde { B } _ { t } \le 0 , } \\ { \operatorname* { m i n } \left\{ 1 , \frac { \tau } { \widetilde { B } _ { t } } \right\} , } & { \widetilde { B } _ { t } > 0 . } \end{array} \right.
$$

Thus,

$$
\begin{array} { r } { \tilde { z } _ { t } = \bar { z } _ { t } ^ { \star } , \qquad z _ { t } = \widehat { z } _ { t } . } \end{array}
$$

We claim that

$$
\bar { z } _ { t } ^ { \star } - \widehat { z } _ { t } \leq \frac { \widetilde { B } _ { t } - B _ { t } ^ { \star } } { \tau } = \frac { \langle X _ { t } , \tilde { \mu } _ { t } - \mu ^ { \star } \rangle } { \tau } .
$$

If $B _ { t } ^ { \star } \leq 0$ , then $\bar { z } _ { t } ^ { \star } = 1$ . If $\widehat { z } _ { t } = 1$ , the claim is trivial. Otherwise, $\widehat { z } _ { t } = \tau / \widetilde { B } _ { t }$ , which implies $\begin{array} { r } { \widetilde { B } _ { t } \geq \tau . } \end{array}$ Then

$$
1 - \frac { \tau } { \widetilde { B } _ { t } } = \frac { \widetilde { B } _ { t } - \tau } { \widetilde { B } _ { t } } \leq \frac { \widetilde { B } _ { t } - B _ { t } ^ { \star } } { \tau } ,
$$

where we used $B _ { t } ^ { \star } \le 0 \le \tau$ and $\widetilde { B } _ { t } \ge \tau$

If $B _ { t } ^ { \star } > 0$ , then by pessimism $B _ { t } ^ { \star } \le \widetilde { B } _ { t }$ . If either upper endpoint equals 1, the desired inequality is immediate. Otherwise,

$$
\bar { z } _ { t } ^ { \star } - \widehat { z } _ { t } = \frac { \tau } { B _ { t } ^ { \star } } - \frac { \tau } { \widetilde { B } _ { t } } = \frac { \tau ( \widetilde { B } _ { t } - B _ { t } ^ { \star } ) } { B _ { t } ^ { \star } \widetilde { B } _ { t } } .
$$

In this subcase both upper endpoints are strictly below 1, hence $B _ { t } ^ { \star } \geq \tau$ and $\widetilde { B } _ { t } \ge \tau$ . Therefore,

$$
\widetilde { z } _ { t } ^ { \star } - \widehat { z } _ { t } \leq \frac { \widetilde { B } _ { t } - B _ { t } ^ { \star } } { \tau } .
$$

Combining the cases gives

$$
g ( \tilde { \alpha } _ { t } ) - g ( \alpha _ { t } ) \leq \frac { \langle X _ { t } , \tilde { \mu } _ { t } - \mu ^ { \star } \rangle } { \tau } .
$$

By Cauchy–Schwarz and Assumptions 2.2 and 2.3,

$$
\left. X _ { t } , \theta ^ { \star } \right. \leq \left\| X _ { t } \right\| _ { 2 } \left\| \theta ^ { \star } \right\| _ { 2 } \leq \kappa _ { 3 } \kappa _ { 4 } .
$$

Thus,

$$
\left( g ( \tilde { \alpha } _ { t } ) - g ( \alpha _ { t } ) \right) \left. X _ { t } , \theta ^ { \star } \right. \leq \kappa _ { 3 } \kappa _ { 4 } \cdot \frac { \left. X _ { t } , \tilde { \mu } _ { t } - \mu ^ { \star } \right. } { \tau } .
$$

## D.3 Proof of Theorem 2

As usual in linear bandit analysis, we use the elliptic potential lemma. The only diference is that our algorithm collects informative samples only in rounds for which $g ( \alpha _ { t } ) > 0$ . Since the number of such rounds is at most T, the standard bound applies.

Lemma 8 (Elliptic potential lemma). Let $X _ { 1 } , \ldots , X _ { N ( T ) }$ be the contexts collected by Algorithm 1 in rounds with $g ( \alpha _ { t } ) > 0$ . For any regularization parameter $0 < \lambda < 1$

$$
\sum _ { t = 1 } ^ { T } \left\| X _ { t } \right\| _ { \Sigma _ { t } ^ { - 1 } } ^ { 2 } \leq d \log \left( 1 + \frac { T \kappa _ { 4 } ^ { 2 } } { d } \right) .
$$

By Cauchy–Schwarz,

$$
\sum _ { t = 1 } ^ { T } \left. X _ { t } \right. _ { \Sigma _ { t } ^ { - 1 } } \leq \sqrt { d T \log \left( 1 + \frac { T \kappa _ { 4 } ^ { 2 } } { d } \right) } .
$$

Theorem 2. With probability at least $1 - \delta ^ { \prime }$ , the regret of the High Probability Constrained UCB algorithm satisfies

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathcal { C } } ( T ) = \mathcal { O } \Big ( \Big ( 1 + \frac { \kappa _ { 3 } \kappa _ { 4 } } { \tau } \Big ) \cdot \operatorname* { m a x } \{ \beta _ { T } ^ { r } ( \delta ^ { \prime } , d ) , \beta _ { T } ^ { c } ( \delta ^ { \prime } , d ) \} } \\ & { \qquad \times \sqrt { d T \log \left( 1 + \frac { T \kappa _ { 4 } ^ { 2 } } { d } \right) } \Big ) . } \end{array}
$$

Proof. On the good event,

$$
\begin{array} { r l } {  { \boldsymbol { \nabla } _ { { \mathbf { C } } } ( \gamma ) - \sum _ { k = 1 } ^ { T } ( \phi ( \hat { \alpha } _ { k } ^ { * } ) - \beta ( \hat { \alpha } _ { k } ^ { * } ) ) ( \boldsymbol { \Lambda } _ { \mathbf { C } } \theta ^ { * } ) ^ { * } \mid \sum _ { t = 1 } ^ { T } ( \phi ( \hat { \alpha } _ { k } ^ { * } ) - \beta ( ( \alpha _ { k } ) ) ) ( \boldsymbol { X } _ { \mathbf { C } } \theta ^ { * } ) } } \\ & { \leq \sum _ { k = 1 } ^ { T } \phi ( \hat { \alpha } _ { k } ) ( \boldsymbol { X } _ { \mathbf { C } } \theta _ { k } - \theta ^ { * } ) + \frac { R _ { \mathbf { C } } R _ { \mathbf { C } } } { T } \sum _ { k = 1 } ^ { T } \boldsymbol { \nabla } _ { \mathbf { C } } \boldsymbol { \mathcal { R } } _ { \mathbf { C } } \theta _ { k } - \mu ^ { * } ) } \\ & { \leq \sum _ { k = 1 } ^ { T } \phi ( \hat { \alpha } _ { k } ) ( \boldsymbol { X } _ { \mathbf { C } } ) \mid \mathrm { s } _ { k } ] \mathrm { s } _ { k } , } \\ { \leq \sum _ { k = 1 } ^ { T } \phi ( \hat { \alpha } _ { k } ) ( \boldsymbol { X } _ { \mathbf { C } } ) \mid \mathrm { s } _ { k } ) , } \\ & { \leq 2 \sum _ { k = 1 } ^ { T } \phi ( \hat { \alpha } _ { k } ^ { * } ) \mid \mathrm { s } _ { k } \mid \mathrm { s } _ { k } , } \\ & { \leq 2 \sum _ { k = 1 } ^ { T } \phi ( \hat { \alpha } _ { k } ^ { * } ) , \mathrm { i n } \times \mathrm { i n } \times \mathrm { i n } \times \mathrm { i n } \times \mathrm { i n } \times \mathrm { i n } \times \mathrm { i n } \times \mathrm { i n } \mathrm { ~ s } _ { k } ^ { T } } \\ &  \leq 2 \sum _ { k = 1 } ^ { T } \phi ( \hat { \alpha } \end{array}
$$

In the third inequality, we used $g ( \tilde { \alpha } _ { t } ) \leq 1$ and the fact that both $\tilde { { \boldsymbol { \theta } } } _ { t } , { \boldsymbol { \theta } } ^ { \star }$ and $\tilde { \mu } _ { t } , \mu ^ { \star }$ lie in their corresponding confidence sets. □

## E Non-linear Rewards and Costs

## E.1 Concentration for Non-linear Least Squares

For the non-linear case, define the informative set

$$
S _ { t } = \{ s \leq t : g ( \alpha _ { s } ) > 0 \} .
$$

The least-squares estimators are

$$
\hat { \theta } _ { t } : = \underset { \theta \in \mathcal { F } ^ { r } } { \arg \operatorname* { m i n } } \sum _ { s \in \mathcal { S } _ { t - 1 } } \left( \theta ( X _ { s } ) - \frac { R _ { s } } { g ( \alpha _ { s } ) } \right) ^ { 2 } ,\tag{24}
$$

and

$$
\hat { \mu } _ { t } : = \underset { \mu \in \mathcal { F } ^ { c } } { \arg \operatorname* { m i n } } \sum _ { s \in \mathcal { S } _ { t - 1 } } \left( \mu ( X _ { s } ) - \frac { C _ { s } } { g ( \alpha _ { s } ) } \right) ^ { 2 } .\tag{25}
$$

Whenever $g ( \alpha _ { s } ) > 0$

$$
\frac { R _ { s } } { g ( \alpha _ { s } ) } = \theta ^ { \star } ( X _ { s } ) + \xi _ { s } ^ { r } , \qquad \frac { C _ { s } } { g ( \alpha _ { s } ) } = \mu ^ { \star } ( X _ { s } ) + \xi _ { s } ^ { c } .
$$

Thus, after normalization by $g ( \alpha _ { s } )$ , the regression observations have the same conditionally sub-Gaussian noise as in the original model.

For any function $f ,$ define

$$
\| f \| _ { \mathcal { D } _ { t } } = \sqrt { \sum _ { s \in \mathcal { S } _ { t - 1 } } f ^ { 2 } ( X _ { s } ) } .
$$

The confidence sets are

$$
\mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) = \left\{ \theta \in \mathcal { F } ^ { r } : \left. \theta - \hat { \theta } _ { t } \right. _ { \mathcal { D } _ { t } } ^ { 2 } \leq \rho ^ { r } ( t , \delta ^ { \prime } ) \right\} ,\tag{26}
$$

$$
\mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) = \left\{ \mu \in \mathcal { F } ^ { c } : \| \mu - \hat { \mu } _ { t } \| _ { \mathcal { D } _ { t } } ^ { 2 } \leq \rho ^ { c } ( t , \delta ^ { \prime } ) \right\} .\tag{27}
$$

For finite function classes, we take

$$
\rho ^ { r } ( t , \delta ^ { \prime } ) = 8 \kappa _ { 1 } ^ { 2 } \log \left( \frac { | \mathcal { F } ^ { r } | } { \delta ^ { \prime } } \right) , \qquad \rho ^ { c } ( t , \delta ^ { \prime } ) = 8 \kappa _ { 2 } ^ { 2 } \log \left( \frac { | \mathcal { F } ^ { c } | } { \delta ^ { \prime } } \right) .
$$

Lemma 9. For any sequence of real-valued random variables $( Z _ { t } ) _ { t \leq T }$ adapted to a filtration $( \mathcal { F } _ { t } ) _ { t \leq T }$ it holds with probability at least $1 - \delta$ that, for all $T ^ { \prime } \leq T$ ，

$$
\sum _ { t = 1 } ^ { T ^ { \prime } } Z _ { t } \leq \sum _ { t = 1 } ^ { T ^ { \prime } } \log \left( \mathbb { E } _ { t - 1 } \left[ e ^ { Z _ { t } } \right] \right) + \log ( \delta ^ { - 1 } ) ,
$$

where $\mathbb { E } _ { t - 1 } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t - 1 } ]$

Proof. Define

$$
M _ { \tau } = \exp \left( \sum _ { t = 1 } ^ { \tau } Z _ { t } - \sum _ { t = 1 } ^ { \tau } \log \left( \mathbb { E } _ { t - 1 } \left[ e ^ { Z _ { t } } \right] \right) \right) .
$$

Then $( M _ { \tau } ) _ { \tau \le T }$ is a nonnegative martingale with $M _ { 0 } = 1$ . Ville’s inequality gives

$$
\mathbb { P } \left( \exists \tau \leq T : M _ { \tau } > \frac { 1 } { \delta } \right) \leq \delta .
$$

Taking logarithms and complements yields the claim.

Proposition 3. For any fixed $\eta \in \mathbb { R }$ , with probability at least $1 - \delta ,$ for all $\theta \in \mathcal { F } ^ { r }$ and all $T ^ { \prime } \leq T$

$$
\eta \sum _ { t = 1 } ^ { T ^ { \prime } } \xi _ { t } ^ { r } \left( \theta ^ { \star } ( X _ { t } ) - \theta ( X _ { t } ) \right) \leq \frac { \eta ^ { 2 } \kappa _ { 1 } ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T ^ { \prime } } \left( \theta ^ { \star } ( X _ { t } ) - \theta ( X _ { t } ) \right) ^ { 2 } + \log \left( \frac { | \mathcal { F } ^ { r } | } { \delta } \right) .
$$

Similarly, with probability at least $1 - \delta ,$ for all $\mu \in \mathcal { F } ^ { c }$ and all $T ^ { \prime } \leq T$

$$
\eta \sum _ { t = 1 } ^ { T ^ { \prime } } \xi _ { t } ^ { c } \left( \mu ^ { \star } ( X _ { t } ) - \mu ( X _ { t } ) \right) \leq \frac { \eta ^ { 2 } \kappa _ { 2 } ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T ^ { \prime } } ( \mu ^ { \star } ( X _ { t } ) - \mu ( X _ { t } ) ) ^ { 2 } + \log \left( \frac { | \mathcal { F } ^ { c } | } { \delta } \right) .
$$

Proof. Fix $\theta \in \mathcal { F } ^ { r }$ and apply Lemma 9 with

$$
Z _ { t } = \eta \xi _ { t } ^ { r } \left( \theta ^ { \star } ( X _ { t } ) - \theta ( X _ { t } ) \right) .
$$

By conditional sub-Gaussianity,

$$
\log \mathbb { E } _ { t - 1 } \left[ e ^ { Z _ { t } } \right] \leq \frac { \eta ^ { 2 } \kappa _ { 1 } ^ { 2 } } { 2 } \left( \theta ^ { \star } ( X _ { t } ) - \theta ( X _ { t } ) \right) ^ { 2 } .
$$

Thus, for this fixed $\theta ,$ the desired inequality holds with probability at least $1 - \delta$ . A union bound over $\theta \in \mathcal { F } ^ { r }$ gives the stated reward result. The cost result is identical, replacing θ by $\mu$ and $\kappa _ { 1 }$ by $\kappa _ { 2 } . ~ \boxed { \begin{array} { r l } \end{array} }$

Lemma 10 (Concentration for non-linear rewards and costs). Let $\delta ^ { \prime } \in ( 0 , 1 )$ . With probability at least $1 - 2 \delta ^ { \prime }$ , for all $t \leq T$

$$
\theta ^ { \star } \in \mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) , \qquad \mu ^ { \star } \in \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) .
$$

Proof. We prove the reward statement; the cost statement is identical. By the definition of $\widehat { \theta } _ { t }$ as a least-squares estimator,

$$
\sum _ { s \in S _ { t - 1 } } \bigg ( \hat { \theta } _ { t } ( X _ { s } ) - \frac { R _ { s } } { g ( \alpha _ { s } ) } \bigg ) ^ { 2 } \leq \sum _ { s \in S _ { t - 1 } } \bigg ( \theta ^ { \star } ( X _ { s } ) - \frac { R _ { s } } { g ( \alpha _ { s } ) } \bigg ) ^ { 2 } .
$$

Since

$$
\frac { R _ { s } } { g ( \alpha _ { s } ) } = \theta ^ { \star } ( X _ { s } ) + \xi _ { s } ^ { r } ,
$$

this implies

$$
\begin{array} { r } { \displaystyle \sum _ { s \in S _ { t - 1 } } \Big ( \hat { \theta } _ { t } ( X _ { s } ) - \theta ^ { \star } ( X _ { s } ) \Big ) ^ { 2 } \leq 2 \displaystyle \sum _ { s \in S _ { t - 1 } } \xi _ { s } ^ { r } \Big ( \hat { \theta } _ { t } ( X _ { s } ) - \theta ^ { \star } ( X _ { s } ) \Big ) . } \end{array}
$$

Applying Proposition 3 with $\theta = \widehat { \theta } _ { t }$ and choosing $\eta = 1 / ( 2 \kappa _ { 1 } ^ { 2 } )$ yields

$$
\sum _ { s \in S _ { t - 1 } } \left( \hat { \theta } _ { t } ( X _ { s } ) - \theta ^ { \star } ( X _ { s } ) \right) ^ { 2 } \leq 8 \kappa _ { 1 } ^ { 2 } \log \left( \frac { | \mathcal { F } ^ { r } | } { \delta ^ { \prime } } \right) .
$$

Thus $\theta ^ { \star } \in { \mathcal { C } } _ { t } ^ { r } ( \delta ^ { \prime } )$ . The same argument gives $\mu ^ { \star } \in { \mathcal { C } } _ { t } ^ { c } ( \delta ^ { \prime } )$ . A union bound gives probability at least $1 - 2 \delta ^ { \prime }$ □

## E.2 Non-linear Regret Analysis

Define the nonlinear good event

$$
\mathcal { E } _ { \mathrm { n l } } : = \left\{ \theta ^ { \star } \in \mathcal { C } _ { t } ^ { r } ( \delta ^ { \prime } ) , \mu ^ { \star } \in \mathcal { C } _ { t } ^ { c } ( \delta ^ { \prime } ) , \hat { \mathcal { A } } _ { t } ^ { f } \subseteq \mathcal { A } _ { t } ^ { f } , \forall t \in [ T ] \right\} .
$$

By the preceding concentration result and the same feasible-set argument as in Lemma 2,

$$
\mathbb { P } ( \mathcal { E } _ { \mathrm { n l } } ) \geq 1 - 3 \delta ^ { \prime } .
$$

For a subset ${ \mathcal { \tilde { F } } } \subseteq { \mathcal { F } } .$ , recall the confidence width

$$
w _ { \tilde { \mathcal { F } } } ( X ) = \operatorname* { s u p } _ { \underline { { f } } , \overline { { f } } \in \widetilde { \mathcal { F } } } \left( \overline { { f } } ( X ) - \underline { { f } } ( X ) \right) .
$$

Lemma 11. Under the nonlinear good event ${ \varepsilon _ { \mathrm { n l } } }$

$$
\sum _ { t = 1 } ^ { T } [ g ( \alpha _ { t } ^ { \star } ) \theta ^ { \star } ( X _ { t } ) - g ( \tilde { \alpha } _ { t } ) \theta ^ { \star } ( X _ { t } ) ] \leq \sum _ { t = 1 } ^ { T } w _ { \mathcal { C } _ { t } ^ { r } } ( X _ { t } ) .
$$

Proof. For each t, since $\theta ^ { \star } \in \mathcal { C } _ { t } ^ { r }$ and $g ( \alpha _ { t } ^ { \star } ) \geq 0$

$$
\begin{array} { r l } { g ( \alpha _ { t } ^ { \star } ) \theta ^ { \star } ( X _ { t } ) \leq \displaystyle \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t } ^ { r } } g ( \alpha _ { t } ^ { \star } ) \theta ( X _ { t } ) } & { } \\ { \leq \displaystyle \operatorname* { m a x } _ { \alpha \in \mathcal { A } _ { t } ^ { f } } \displaystyle \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t } ^ { r } } g ( \alpha ) \theta ( X _ { t } ) } & { } \\ { = g ( \tilde { \alpha } _ { t } ) \tilde { \theta } _ { t } ( X _ { t } ) . } \end{array}
$$

Therefore,

$$
g ( \alpha _ { t } ^ { \star } ) \theta ^ { \star } ( X _ { t } ) - g ( \tilde { \alpha } _ { t } ) \theta ^ { \star } ( X _ { t } ) \leq g ( \tilde { \alpha } _ { t } ) \left( \tilde { \theta } _ { t } ( X _ { t } ) - \theta ^ { \star } ( X _ { t } ) \right) .
$$

Since $g ( \tilde { \alpha } _ { t } ) \leq 1$ and both $\tilde { { \boldsymbol { \theta } } } _ { t } , { \boldsymbol { \theta } } ^ { \star } \in \mathcal { C } _ { t } ^ { r }$

$$
g ( \tilde { \alpha } _ { t } ) \left( \tilde { \theta } _ { t } ( X _ { t } ) - \theta ^ { \star } ( X _ { t } ) \right) \leq w _ { \mathcal { C } _ { t } ^ { r } } ( X _ { t } ) .
$$

Summing over t proves the claim.

Lemma 12. Under the nonlinear good event ${ \varepsilon _ { \mathrm { n l } } }$

$$
\sum _ { t = 1 } ^ { T } \left[ g ( \tilde { \alpha } _ { t } ) \theta ^ { \star } ( X _ { t } ) - g ( \alpha _ { t } ) \theta ^ { \star } ( X _ { t } ) \right] \leq \frac { 1 } { \tau } \sum _ { t = 1 } ^ { T } w _ { \mathcal { C } _ { t } ^ { c } } ( X _ { t } ) .
$$

Proof. Fix a round t and define

$$
b _ { \delta } = \kappa _ { 2 } \sqrt { 2 \log \left( \frac { 1 } { \delta } \right) } , \qquad B _ { t } ^ { \star } = \mu ^ { \star } ( X _ { t } ) + b _ { \delta } , \qquad \widetilde { B } _ { t } = \widetilde { \mu } _ { t } ( X _ { t } ) + b _ { \delta } .
$$

By construction of the pessimistic cost estimate and on the good event,

$$
B _ { t } ^ { \star } \le \widetilde { B } _ { t }
$$

whenever $B _ { t } ^ { \star } \geq 0 .$

If the optimistic reward estimate at $X _ { t }$ is negative, then both $\tilde { \alpha } _ { t }$ and $\alpha _ { t }$ are zero, and the contribution to regret is nonpositive. We therefore consider the case where the optimistic reward estimate is nonnegative. Since $g$ is strictly increasing, the selected actions correspond to the largest feasible efective actions.

Let

$$
\tilde { z } _ { t } = g ( \tilde { \alpha } _ { t } ) , \qquad z _ { t } = g ( \alpha _ { t } ) .
$$

The true and estimated safe efective-action upper endpoints are

$$
\bar { z } _ { t } ^ { \star } = \left\{ \begin{array} { l l } { 1 , } & { B _ { t } ^ { \star } \le 0 , } \\ { \operatorname* { m i n } \left\{ 1 , \displaystyle \frac { \tau } { B _ { t } ^ { \star } } \right\} , } & { B _ { t } ^ { \star } > 0 , } \end{array} \right.
$$

and

$$
\widehat { z } _ { t } = \left\{ \begin{array} { l l } { 1 , } & { \widetilde { B } _ { t } \le 0 , } \\ { \operatorname* { m i n } \left\{ 1 , \frac { \tau } { \widetilde { B } _ { t } } \right\} , } & { \widetilde { B } _ { t } > 0 . } \end{array} \right.
$$

Thus,

$$
\begin{array} { r } { \tilde { z } _ { t } = \bar { z } _ { t } ^ { \star } , \qquad z _ { t } = \widehat { z } _ { t } . } \end{array}
$$

The same endpoint calculation as in Lemma 5 gives

$$
g ( \tilde { \alpha } _ { t } ) - g ( \alpha _ { t } ) = \tilde { z } _ { t } - z _ { t } \leq \frac { \widetilde { B } _ { t } - { B } _ { t } ^ { \star } } { \tau } = \frac { \tilde { \mu } _ { t } ( X _ { t } ) - \mu ^ { \star } ( X _ { t } ) } { \tau } .
$$

Since $\theta ^ { \star } ( X _ { t } ) \leq 1$ by boundedness,

$$
g ( \tilde { \alpha } _ { t } ) \theta ^ { \star } ( X _ { t } ) - g ( \alpha _ { t } ) \theta ^ { \star } ( X _ { t } ) \leq \frac { \tilde { \mu } _ { t } ( X _ { t } ) - \mu ^ { \star } ( X _ { t } ) } { \tau } .
$$

Finally, both $\tilde { \mu } _ { t }$ and $\mu ^ { \star }$ belong to $\mathcal { C } _ { t } ^ { c }$ on the good event, so

$$
\tilde { \mu } _ { t } ( X _ { t } ) - \mu ^ { \star } ( X _ { t } ) \leq w _ { \mathcal { C } _ { t } ^ { c } } ( X _ { t } ) .
$$

Summing over t gives the result.

Lemma 13 (Eluder-dimension width bound). Let $\mathcal { F }$ be a function class with eluder dimension $d _ { \mathrm { e l u d e r } } = d _ { \mathrm { e l u d e r } } ( \mathcal { F } , 1 / T ^ { 2 } )$ , and let $\mathcal { C } _ { t }$ be confidence sets with radius $\rho ( T , \delta ^ { \prime } )$ . Then

$$
\sum _ { t = 1 } ^ { T } w _ { \mathcal { C } _ { t } } ( X _ { t } ) = \mathcal { O } \left( \sqrt { d _ { \mathrm { e l u d e r } } T \rho ( T , \delta ^ { \prime } ) } \right) .
$$

## E.3 Proof of Theorem 3

Theorem 3. With probability at least $1 - \delta ^ { \prime }$ , the regret of the Non-Linear High Probability Constrained UCB algorithm satisfies

$$
\begin{array} { r l } & { \mathcal { R } ( T ) = \mathcal { O } \Big ( \sqrt { d _ { \mathrm { e l u d e r } } ^ { r } T \rho ^ { r } ( T , \delta ^ { \prime } / 3 ) } } \\ & { \quad \quad \quad + \frac { 1 } { \tau } \sqrt { d _ { \mathrm { e l u d e r } } ^ { c } T \rho ^ { c } ( T , \delta ^ { \prime } / 3 ) } \Big ) . } \end{array}
$$

Proof. Using Lemmas 11 and 12, we have

$$
\begin{array} { l } { { \displaystyle { \mathcal R } _ { { \mathcal C } } ( T ) = \sum _ { t = 1 } ^ { T } [ g ( \alpha _ { t } ^ { \star } ) \theta ^ { \star } ( X _ { t } ) - g ( \tilde { \alpha } _ { t } ) \theta ^ { \star } ( X _ { t } ) ] + \sum _ { t = 1 } ^ { T } [ g ( \tilde { \alpha } _ { t } ) \theta ^ { \star } ( X _ { t } ) - g ( \alpha _ { t } ) \theta ^ { \star } ( X _ { t } ) ] } } \\ { { \displaystyle \qquad \leq \sum _ { t = 1 } ^ { T } w _ { { \mathcal C } _ { t } ^ { r } } ( X _ { t } ) + \frac { 1 } { \tau } \sum _ { t = 1 } ^ { T } w _ { { \mathcal C } _ { t } ^ { \mathrm { c } } } ( X _ { t } ) } . } \end{array}
$$

Applying Lemma 13 to the reward and cost confidence classes gives

$$
\sum _ { t = 1 } ^ { T } w _ { \mathcal { C } _ { t } ^ { r } } ( X _ { t } ) = \mathcal { O } \left( \sqrt { d _ { \mathrm { e l u d e r } } ^ { r } T \rho ^ { r } ( T , \delta ^ { \prime } / 3 ) } \right) ,
$$

and

$$
\sum _ { t = 1 } ^ { T } w _ { \mathcal { C } _ { t } ^ { c } } ( X _ { t } ) = \mathcal { O } \left( \sqrt { d _ { \mathrm { e l u d e r } } ^ { c } T \rho ^ { c } ( T , \delta ^ { \prime } / 3 ) } \right) .
$$

Therefore,

$$
\mathcal { R } _ { \mathcal { C } } ( T ) = \mathcal { O } \left( \sqrt { d _ { \mathrm { e l u d e r } } ^ { r } T \rho ^ { r } ( T , \delta ^ { \prime } / 3 ) } + \frac { 1 } { \tau } \sqrt { d _ { \mathrm { e l u d e r } } ^ { c } T \rho ^ { c } ( T , \delta ^ { \prime } / 3 ) } \right) .
$$