# Achieving Minimax-Optimal First-Order and Best-of-Both-Worlds Regret Simultaneously in Multi-Armed Bandits

Mengxiao Zhang University of Iowa mengxiao-zhang@uiowa.edu

## Abstract

We study first-order and best-of-both-worlds regret guarantees in multi-armed bandits. Pogodin and Lattimore (2019) proposed an algorithm that combines minimax-optimal worst-case regret with a first-order guarantee whose loss-dependent term is $\mathcal { O } ( \sqrt { K L ^ { \star } } \log T )$ , where T denotes the horizon, $L ^ { \star }$ denotes the smallest expected cumulative loss among fixed arms, and K denotes the number of arms. However, a logarithmic gap remains between this loss-dependent term and the optimal first-order scale. In this paper, we remove this logarithmic factor by proposing an efficient and parameter-free algorithm that achieves the minimax optimal first-order regret guarantee $\mathcal { O } ( \sqrt { K L ^ { \star } } )$ . Technically, our algorithm is built upon FTRL with a Tsallis entropy regularizer (Zimmert and Seldin, 2021), with a learning rate adapted to the learner’s cumulative loss and a sampling distribution determined by a scalar fixed point. This construction controls the combined estimation and stability cost in terms of the learner’s loss. Moreover, we show that the same algorithm achieves a best-of-both-worlds guarantee, with $\begin{array} { r } { \mathcal { O } ( \sum _ { i \neq i ^ { \star } } \frac { \log T } { \Delta _ { i } } ) } \end{array}$ ) regret for i.i.d. losses with a unique best arm i<sup>⋆</sup>, where $\Delta _ { i }$ denotes the suboptimality gap of arm i.

## 1 Introduction

We study first-order and best-of-both-worlds regret guarantees in multi-armed bandits (MAB). In adversarial MAB, a learner has K arms available and interacts with the environment over a horizon $T > 2$ where the loss of each arm is arbitrarily decided. At each round, the learner chooses an arm and observes only the loss of the chosen arm. The objective is to minimize the learner’s expected cumulative loss relative to the best fixed arm. While the minimax regret is $\Theta ( \sqrt { K T } )$ (Auer et al., 2002b; Audibert and Bubeck, 2009), this worst-case guarantee does not adapt to easier loss sequences.

Indeed, one can achieve a much better regret guarantee by exploiting structure in the loss sequence. For example, when the losses are drawn i.i.d. from a fixed distribution, algorithms can achieve logarithmic gap-dependent regret (Auer et al., 2002a). First-order bounds, also called small-loss bounds, provide another form of adaptation by replacing the leading dependence on the horizon with a dependence on the smallest expected cumulative loss $L ^ { \star }$ among fixed arms. When $L ^ { \star } \ll T$ , such a guarantee can be substantially smaller than the minimax bound. In fact, multiple approaches achieve near-optimal first-order guarantees (Allenberg et al., 2006; Neu, 2015b; Wei and Luo, 2018; Pogodin and Lattimore, 2019; Ito, 2021). Specifically, Pogodin and Lattimore (2019) combine minimax-optimal worst-case regret with a first-order guarantee whose lossdependent term is $\mathcal { O } ( \sqrt { K L ^ { \star } } \log T )$ , together with additional terms depending on K and log $T . ^ { 1 }$ <sup>1</sup> However, none of these guarantees removes the logarithmic factor from the leading first-order term, so these first-order bounds alone do not recover the minimax-optimal $\mathcal { O } ( \sqrt { K T } )$ ) regret when $L ^ { \star }$ is of order T, while the lowerbound construction of Gerchinovitz and Lattimore (2016) only shows an $\Omega ( \sqrt { K L ^ { \star } } )$ first-order regret lower bound. Via a Bayesian analysis of Thompson sampling and the minimax theorem, Bubeck and Sellke (2020) show that some algorithm attains $\mathcal { O } ( \sqrt { K \bar { L } } + K \log ^ { 2 } \bar { L } )$ regret on every oblivious loss sequence with $L ^ { \star } \leq \bar { L }$ However, this argument is non-constructive and requires $\bar { L }$ and $T$ in advance. This leaves open whether an explicit algorithm can achieve the minimax-optimal first-order guarantee without such prior knowledge. We therefore ask:

Can one design an efficient algorithm that achieves the minimax-optimal first-order guarantee in adversarial MAB? Moreover, can the same algorithm achieve a best-of-both-worlds guarantee?

In this paper, we answer this question affirmatively by designing a single algorithm with leading first-order regret $\mathcal { O } ( \sqrt { K L ^ { \star } } )$ in the adversarial environment and $\begin{array} { r } { \dot { \mathcal { O } } ( \sum _ { i \neq i ^ { \star } } \frac { \log T } { \Delta _ { i } } ) } \end{array}$ pseudo-regret for i.i.d. losses with a unique best arm $i ^ { \star }$ , where $\Delta _ { i }$ denotes the mean-loss gap of arm i from $i ^ { \star }$ . Technically, we couple the sampling distribution of Tsallis FTRL (Zimmert and Seldin, 2021) with its implicit exploration estimator (Kocák et al., 2014; Neu, 2015a) through a scalar fixed point. This controls the combined cost of estimation bias and stability in terms of the learner’s loss, and a regularization scale based on observed cumulative loss gives the optimal leading first-order term in Section 3. In Section 4, a refined stability analysis bounds the optimal arm’s contribution through suboptimal-arm probabilities, while the estimation bias cancels the gap-dependent part of the stability bound. The resulting regret bound depends on suboptimal-arm probabilities, allowing the same observed-loss scale and the self-bounding identity to yield logarithmic gap-dependent regret. To our knowledge, this is the first efficient algorithm that achieves the minimax optimal first-order regret guarantee for MAB without knowing $L ^ { \star }$ or $T ,$ , and it also achieves a best-of-both-worlds guarantee.

## 1.1 Related work

First-order guarantees refine the worst-case bandit bounds of Auer et al. (2002b); Audibert and Bubeck (2009) by replacing the horizon dependence with $L ^ { \star }$ (Allenberg et al., 2006; Neu, 2015b). On the upper-bound side, GREEN (Allenberg et al., 2006), FPL-TRIX (Neu, 2015b), log-barrier mirror descent with a learning rate tuned using $L ^ { \star }$ (Foster et al., 2016), and BROAD-OMD (Wei and Luo, 2018) achieve first-order guarantees with loss-dependent terms of $\widetilde { \mathcal { O } } ( \sqrt { K L ^ { \star } } ) . ^ { 2 }$ <sup>2</sup> In particular, FPL-TRIX attains $\mathcal { O } ( \sqrt { K L ^ { \star } \log K } )$ without prior knowledge, but its worst-case bound $\mathcal { O } ( \sqrt { K T \log K } )$ is not minimax optimal, while the log-barrier methods above incur $\mathcal { O } ( \sqrt { K L ^ { \star } \log T } )$ The INF/log-barrier FTRL algorithm of Pogodin and Lattimore (2019) combines minimax-optimal $\mathcal { O } ( \sqrt { K T } )$ worst-case regret with a first-order guarantee whose loss-dependent term is $\mathcal { O } ( \sqrt { K L ^ { \star } } \log T )$ , but a logarithmic factor remains in this term. First-order guarantees have also been studied in richer models with partial feedback, including bandits with feedback graphs (Lykouris et al., 2018; Lee et al., 2020b), contextual bandits (Allen-Zhu et al., 2018; Foster and Krishnamurthy, 2021; Olkhovskaya et al., 2023; van Erven et al., 2025), linear bandits (Lee et al., 2020a; Ito et al., 2020), and episodic Markov decision processes (MDPs) (Lee et al., 2020a; Li et al., 2026).

Best-of-both-worlds guarantees require a single algorithm to achieve $\widetilde { \mathcal { O } } ( \sqrt { K T } )$ adversarial pseudo-regret and (poly)logarithmic gap-dependent stochastic pseudo-regret without knowing the environment. SAO initiated this direction (Bubeck and Slivkins, 2012), followed by EXP3++ (Seldin and Slivkins, 2014; Seldin and Lugosi, 2017) and SAPO (Auer and Chiang, 2016). Tsallis-INF achieves $\mathcal { O } ( \sqrt { K T } )$ adversarial pseudo regret and $\begin{array} { r } { \mathcal { O } ( \sum _ { i \neq i ^ { \star } } \frac { \log T } { \Delta _ { i } } ) } \end{array}$ stochastic pseudo-regret with an anytime learning rate (Zimmert and Seldin, 2021). Ito (2021) showed that Tsallis-INF retains logarithmic stochastic regret without a unique optimal arm, and Jin et al. (2023) extended this to general regularizers. Follow-the-perturbed-leader (FTPL) also achieves best-of-both-worlds guarantees for multi-armed bandits (Honda et al., 2023; Lee et al., 2024; Chen et al., 2025; Lee et al., 2025, 2026). This objective has been extended beyond standard MAB to combinatorial semi-bandits (Zimmert et al., 2019; Chen et al., 2026), bandits with delayed feedback (Masoudian et al.,

2022; Schlisselberg et al., 2025), linear bandits, contextual bandits, and bandits with feedback graphs (Dann et al., 2023; Ito et al., 2022a, 2024; Kato and Ito, 2025), as well as episodic MDPs (Jin and Luo, 2020; Jin et al., 2021; Ito et al., 2025; Li et al., 2026). Simultaneous first-order and best-of-both-worlds guarantees have been obtained using log-barrier OMD and adaptive FTRL (Wei and Luo, 2018; Ito, 2021; Ito et al., 2022b; Nguyen et al., 2025). Our result removes the logarithmic factor from the leading adversarial first-order term while achieving logarithmic gap-dependent stochastic regret.

## 2 Preliminaries

We study multi-armed bandits with $K \geq 2$ arms over a horizon $T > 2$ . Denote by $[ K ] \triangleq \{ 1 , \dots , K \}$ the set of arms and by $\mathcal { F } _ { t - 1 }$ the σ-field generated by the learner’s past actions, observed losses, and internal randomness before round t. At round t, the environment selects a loss vector $\ell _ { t } \in [ 0 , 1 ] ^ { K }$ , and the learner selects an $\mathcal { F } _ { t - 1 }$ -measurable distribution $q _ { t }$ over the arms. The learner draws $I _ { t } \sim q _ { t }$ , incurs loss $\ell _ { t , I _ { t } }$ , and observes only this loss.

Our performance criterion is pseudo-regret against the best fixed arm in expectation. The expectation includes learner and environment randomization. Denote by $L _ { T , k } \triangleq \mathbb { E } [ \sum _ { t = 1 } ^ { T } \ell _ { t , k } ]$ the expected cumulative loss of arm k and by $L ^ { \star } \triangleq \operatorname* { m i n } _ { k \in [ K ] } L _ { T , k }$ the smallest such loss. For a fixed arm $k ,$ denote by $\mathsf { R e g } _ { T , k } \triangleq$ $\begin{array} { r } { \mathbb { E } [ \sum _ { t = 1 } ^ { T } \ell _ { t , I _ { t } } ] - L _ { T , k } } \end{array}$ the corresponding pseudo-regret. The pseudo-regret against the best fixed arm is

$$
\mathsf { R e g } _ { T } \triangleq \mathbb { E } \left[ \sum _ { \ell = 1 } ^ { T } \ell _ { t , I _ { t } } \right] - L ^ { \star } = \operatorname* { m a x } _ { k \in [ K ] } \mathsf { R e g } _ { T , k } .
$$

For a deterministic loss sequence, $L ^ { \star }$ is the usual cumulative loss of the best arm in hindsight.

In the stochastic environment, the loss vectors are i.i.d. with means $\mu _ { i } \triangleq \mathbb { E } [ \ell _ { t , i } ]$ and a unique best arm $i ^ { \star }$ Denote by $\Delta _ { i } \triangleq \mu _ { i } - \mu _ { i ^ { \star } } > 0$ the gap of arm $i \neq i ^ { \star }$ . The coordinates within a loss vector may be dependent. Independence across rounds gives the identity

$$
\mathsf { R e g } _ { T } = \sum _ { i \neq i ^ { \star } } \Delta _ { i } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } q _ { t , i } \right] .\tag{1}
$$

Our goal is a guarantee with the optimal leading first-order term $\mathcal { O } ( \sqrt { K L ^ { \star } } )$ in the adversarial environment, together with logarithmic gap-dependent regret in the stochastic environment.

Notation. Denote by $\Delta _ { K } \ \triangleq \ \{ x \ \in \ \mathbb { R } _ { + } ^ { K } \ : \ \sum _ { i } x _ { i } \ = \ 1 \}$ the probability simplex and by int $\left( \Delta _ { K } \right)$ its relative interior, consisting of distributions with strictly positive coordinates. Denote by $e _ { i }$ the i-th standard basis vector, by 1 the all-ones vector, and by $\mathbb { I } \{ \cdot \}$ the indicator function. For $u , v \in \mathbb { R } ^ { K }$ , denote by $\textstyle \langle u , v \rangle \triangleq \sum _ { i = 1 } ^ { K } { \mathrm { \overline { { u } } } } _ { i } v _ { i }$ the inner product. Denote by $\mathbb { E } _ { t } [ \cdot ] \triangleq \mathbb { E } [ \cdot \mid \mathcal { F } _ { t - 1 } , \ell _ { t } ]$ conditional expectation over the fresh draw at round t. For a differentiable convex function Φ, denote by $D _ { \Phi } ( x , y ) \triangleq \Phi ( x ) - \Phi ( y ) - \langle \nabla \Phi ( y ) , x - y \rangle$ its Bregman divergence.

## 3 Minimax Optimal First-Order Regret via Fixed-Point Sampling

In this section, we derive our minimax optimal first-order algorithm from FTRL with Tsallis entropy (Zimmert and Seldin, 2021). In Section 3.1, we identify why the usual stability bound gives minimax regret but does not adapt to the learner’s loss, and in Section 3.2, we derive a sampling distribution that controls estimation bias and stability together through a scalar fixed point, then choose the learning rate from observed cumulative loss.

## 3.1 From FTRL with Tsallis Entropy to a Loss-Dependent Remainder

Our starting point is the $\textstyle { \frac { 1 } { 2 } }$ -Tsallis regularizer used by Tsallis-INF (Zimmert and Seldin, 2021). Denote by $\Psi ( x ) \triangleq - 2 \sum _ { i } { \sqrt { x _ { i } } }$ the negative $\mathrm { \frac { 1 } { 2 } \mathrm { - T s a l l i s } }$ entropy regularizer. Fix a learning rate $\eta > 0$ , and denote by $\begin{array} { r } { \widehat { L } _ { t } \triangleq \sum _ { s = 1 } ^ { t } \widehat { \ell } _ { s } } \end{array}$ the cumulative loss estimate, with $\widehat { L } _ { 0 } = 0$ . The FTRL distribution before round t is

$$
x _ { t } = \arg \operatorname* { m i n } _ { x \in \Delta _ { K } } \left\{ \left. \widehat { L } _ { t - 1 } , x \right. + \frac { 1 } { \eta } \Psi ( x ) \right\} .
$$

We first consider direct sampling $I _ { t } \sim x _ { t }$ with the unbiased estimate $\widehat { \ell } _ { t , i } = \ell _ { t , I _ { t } } \mathbb { I } \{ I _ { t } = i \} / x _ { t , i }$ . The standard FTRL analysis (Zimmert and Seldin, 2021) gives

$$
\sum _ { t = 1 } ^ { T } { \left. x _ { t } - e _ { k } , \widehat { \ell } _ { t } \right. } \leq \frac { 2 \sqrt { K } } { \eta } + \eta \sum _ { t = 1 } ^ { T } { \sum _ { i = 1 } ^ { K } { x _ { t , i } ^ { 3 / 2 } \widehat { \ell } _ { t , i } ^ { 2 } } } .\tag{2}
$$

Taking conditional expectations under direct sampling therefore yields

$$
\mathsf { R e g } _ { T , k } \leq \mathcal { O } \left( \frac { \sqrt { K } } { \eta } + \eta \mathbb { E } \left[ \sum _ { { t = 1 } } ^ { T } \sum _ { i = 1 } ^ { K } \sqrt { x _ { { t } , i } } \ell _ { { t } , i } ^ { 2 } \right] \right) .
$$

Bounding the inner sum by $\sqrt { K } \cdot T$ gives $\mathcal { O } ( \sqrt { K T } )$ worst-case regret after picking $\eta = 1 / { \sqrt { T } } .$ . A typical route to a first-order guarantee is to control the quadratic cost $\textstyle \sum _ { t = 1 } ^ { T } \bar { \sum } _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } \ell _ { t , i } ^ { 2 }$ through the learner’s loss $\textstyle \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { K } x _ { t , i } \ell _ { t , i }$ . However, when $\ell _ { t } = e _ { i }$ , the inner sum is $\sqrt { \boldsymbol { x } _ { t , i } }$ , while the learner’s expected loss is $x _ { t , i }$ Their ratio is unbounded as $\boldsymbol { x } _ { t , i }$ approaches zero, so this term cannot be controlled uniformly by the learner’s loss.

To resolve this mismatch, we keep the FTRL update and change its sampling distribution from $x _ { t }$ to $q _ { t }$ We make two choices jointly: the loss estimator and the played distribution. First, after drawing $I _ { t } \sim q _ { t }$ , we use the implicit-exploration (IX) estimator (Kocák et al., 2014; Neu, 2015a), with positive offsets $\rho _ { t , i } > 0$ to be specified later:

$$
\widehat { \ell } _ { t , i } = \frac { \ell _ { t , I _ { t } } \mathbb { I } \{ I _ { t } = i \} } { q _ { t , i } + \rho _ { t , i } } ,\tag{3}
$$

for every arm i. The FTRL update still uses only $\widehat { L } _ { t } = \widehat { L } _ { t - 1 } + \widehat { \ell } _ { t }$ . Under sampling from $q _ { t }$ , its conditional mean is $\mathbb { E } _ { t } [ \widehat { \ell } _ { t , i } ] = q _ { t , i } \ell _ { t , i } / ( q _ { t , i } + \rho _ { t , i } ) \leq \ell _ { t , i }$ , so the estimator is biased downward. Its second moment is $\mathbb { E } _ { t } [ \widehat { \ell } _ { t , i } ^ { 2 } ] = q _ { t , i } \ell _ { t , i } ^ { 2 } / ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 }$ . Second, we construct $q _ { t }$ from $x _ { t }$ so that the cost of this change and the quadratic term in Eq. (2) can be controlled by the played loss. Taking expectations in Eq. (2) and using the fact that $\mathbb { E } _ { t } [ \widehat { \ell } _ { t , k } ] \leq \ell _ { t , k }$ for all $k \in [ K ]$ gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( \mathbb { E } _ { t } \left[ \left. x _ { t } , \widehat { \ell } _ { t } \right. \right] - \eta \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \mathbb { E } _ { t } \left[ \widehat { \ell } _ { t , i } ^ { 2 } \right] \right) \right] \leq L _ { T , k } + \mathcal { O } \left( \frac { \sqrt { K } } { \eta } \right) .
$$

Rearranging this inequality to bound $- L _ { T , k }$ and adding $\mathbb { E } [ \sum _ { t } \left. q _ { t } , \ell _ { t } \right. ]$ to both sides yields

$$
\mathsf { R e g } _ { T , k } \leq \mathcal { O } \left( \frac { \sqrt { K } } { \eta } \right) + \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( \langle q _ { t } , \ell _ { t } \rangle - \mathbb { E } _ { t } \left[ \left. x _ { t } , \widehat { \ell } _ { t } \right. \right] + \eta \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \mathbb { E } _ { t } \left[ \widehat { \ell } _ { t , i } ^ { 2 } \right] \right) \right] .\tag{4}
$$

Thus the two choices must control the combined remainder, including both estimation bias and the quadratic cost. We aim to construct $q _ { t }$ and an $\mathcal { F } _ { t - 1 }$ -measurable $Z _ { t } \in ( 0 , 1 )$ satisfying

$$
\mathbb { E } _ { t } \left[ \left. x _ { t } , \widehat { \ell } _ { t } \right. \right] - \eta \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \mathbb { E } _ { t } \left[ \widehat { \ell } _ { t , i } ^ { 2 } \right] \geq Z _ { t } \left. q _ { t } , \ell _ { t } \right. ,\tag{5}
$$

$$
1 - Z _ { t } = \mathcal { O } ( \eta \sqrt { K } ) .
$$

The first condition bounds the remainder in Eq. (4) by $\left( 1 - Z _ { t } \right) \left. q _ { t } , \ell _ { t } \right.$ , and the second makes this quantity bounded by $\mathcal { O } ( \eta \sqrt { K } ) \left. q _ { t } , \ell _ { t } \right.$ . Using $\begin{array} { r } { \mathbb E [ \sum _ { t } \left. q _ { t } , \ell _ { t } \right. ] = L _ { T , k } + \mathsf { R e g } _ { T , k } } \end{array}$ gives $\mathsf { R e g } _ { T , k } \le \mathcal { O } ( \sqrt { K } / \eta ) +$ $\mathcal { O } ( \eta \sqrt { K } ) ( L _ { T , k } + \mathsf { R e g } _ { T , k } )$ . As long as $\eta \sqrt { K }$ is sufficiently small, rearranging the terms yields $\mathsf { R e g } _ { T , k } =$ $\mathcal { O } ( \sqrt { K } / \eta + \eta \sqrt { K } L _ { T , k } )$ . It remains to construct the offset $\rho _ { t , i }$ and sampling distribution $q _ { t }$ so that both conditions in Eq. (5) hold without observing $\ell _ { t }$

## 3.2 Deriving the Sampling Distribution

We first leave $\rho _ { t , i }$ unspecified and derive the conditions needed for Eq. (5). Substituting the conditional moments of Eq. (3) gives

$$
\mathbb { E } _ { t } \left[ \left. x _ { t } , \widehat { \ell } _ { t } \right. \right] - \eta \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \mathbb { E } _ { t } \left[ \widehat { \ell } _ { t , i } ^ { 2 } \right] \geq \sum _ { i = 1 } ^ { K } q _ { t , i } \ell _ { t , i } \left( \frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { \eta x _ { t , i } ^ { 3 / 2 } } { ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 } } \right) ,\tag{6}
$$

where the inequality uses $\ell _ { t , i } ^ { 2 } \leq \ell _ { t , i }$ . Thus, to control the remainder by $\left( 1 - Z _ { t } \right) \left. q _ { t } , \ell _ { t } \right.$ for every loss vector, it suffices to make

$$
\frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { \eta x _ { t , i } ^ { 3 / 2 } } { ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 } } \geq Z _ { t } ,\tag{7}
$$

for every coordinate with $q _ { t , i } > 0$ , while keeping $1 - Z _ { t } = \mathcal { O } ( \eta \sqrt { K } )$ . This is the joint design requirement on the estimator offset and the sampling distribution.

To obtain a sampling rule $q _ { t }$ from Eq. (7), we aim to find a reciprocal lower bound on its coefficient. Suppose that $q _ { t , i } + \rho _ { t , i } \geq 2 \eta \sqrt { x _ { t , i } }$ . The scalar inequality $1 / d - a / d ^ { 2 } \geq 1 / ( d + 2 a )$ for $d \geq 2 a > 0$ therefore gives

$$
\frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { \eta x _ { t , i } ^ { 3 / 2 } } { ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 } } \geq \frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } + 2 \eta \sqrt { x _ { t , i } } } .\tag{8}
$$

Therefore, an offset satisfying $\rho _ { t , i } \geq 2 \eta \sqrt { x _ { t , i } }$ makes this comparison valid uniformly over the sampling probabilities. We impose this sufficient condition for now and choose the exact offset after checking the normalization cost.

To further match the right hand side of Eq. (8) to $Z _ { t } .$ , a positive coordinate of $q _ { t }$ must satisfy $q _ { t , i } =$ $x _ { t , i } / Z _ { t } - \rho _ { t , i } - 2 \eta \sqrt { x _ { t , i } }$ . Denote by $[ v ] _ { + } \stackrel { \triangle } { = }$ max{v, 0} the positive part. We therefore consider

$$
q _ { i } ( Z ) = \Bigl [ \frac { x _ { t , i } } { Z } - \rho _ { t , i } - 2 \eta \sqrt { x _ { t , i } } \Bigr ] _ { + } ,\tag{9}
$$

and determine $Z _ { t }$ from $\begin{array} { r } { \sum _ { i = 1 } ^ { K } q _ { i } ( Z _ { t } ) = 1 } \end{array}$ , with $q _ { t , i } = q _ { i } ( Z _ { t } )$ . In order to ensure that $q _ { t } \in \Delta _ { K }$ , it suffices to solve the following scalar fixed-point equation

$$
Z _ { t } = \sum _ { i = 1 } ^ { K } \left[ x _ { t , i } - Z _ { t } \big ( \rho _ { t , i } + 2 \eta \sqrt { x _ { t , i } } \big ) \right] _ { + } .\tag{10}
$$

It couples the played probabilities to the estimator that uses those same probabilities in its denominator. The fixed-point equation has a unique solution in $( 0 , 1 ) \colon \sum _ { i } q _ { i } ( Z )$ is continuous and strictly decreasing wherever positive, diverges as $Z$ approaches 0, and is less than one at $Z = 1$

It remains to show that $1 - Z _ { t }$ satisfies the required bound in Eq. (5). By Eq. (10), we have

$$
\begin{array} { r l } & { 1 - Z _ { t } = \displaystyle \sum _ { i = 1 } ^ { K } \Big ( x _ { t , i } - \big [ x _ { t , i } - Z _ { t } \big ( \rho _ { t , i } + 2 \eta \sqrt { x _ { t , i } } \big ) \big ] _ { + } \Big ) } \\ & { \qquad = \displaystyle \sum _ { i = 1 } ^ { K } \operatorname* { m i n } \big \{ x _ { t , i } , Z _ { t } \big ( \rho _ { t , i } + 2 \eta \sqrt { x _ { t , i } } \big ) \big \} \leq \sum _ { i = 1 } ^ { K } \rho _ { t , i } + 2 \eta \sqrt { K } , } \end{array}\tag{11}
$$

Algorithm 1: Follow-the-Regularized-Leader with Tsallis Entropy and Fixed-Point Sampling   
Input: number of arms $\begin{array} { r } { K \geq 2 , \Psi ( x ) = - 2 \sum _ { i = 1 } ^ { K } \sqrt { x _ { i } } . } \end{array}$   
1 Initialize $\widehat { L } _ { 0 } = 0$   
2 for $t = 1 , 2 , \dots$ . do   
3 Compute $\begin{array} { r } { x _ { t } = \arg \operatorname* { m i n } _ { x \in \Delta _ { K } } \Big \{ \Big \langle x , \widehat { L } _ { t - 1 } \Big \rangle + \frac { 1 } { \eta _ { t } } \Psi ( x ) \Big \} } \end{array}$ with $\begin{array} { r } { \eta _ { t } = \frac { 1 } { \sqrt { K + \sum _ { \tau < t } \ell _ { \tau , I _ { \tau } } } } } \end{array}$   
4 Define $\rho _ { t , i } = 2 \eta _ { t } \sqrt { x _ { t , i } }$ for every i and find $Z _ { t } \in ( 0 , 1 )$ such that $\begin{array} { r } { \sum _ { i } [ x _ { t , i } / Z _ { t } - 2 \rho _ { t , i } ] _ { + } = 1 } \end{array}$   
5 Set $q _ { t , i } = [ x _ { t , i } / Z _ { t } - 2 \rho _ { t , i } ] _ { + }$ for every $i \in [ K ]$   
6 Draw $I _ { t } \sim q _ { t }$ and observe $\ell _ { t , I _ { t } }$   
7 Construct $\begin{array} { r } { \widehat { \ell } _ { t , i } = \frac { \ell _ { t , I _ { t } } \mathbb { I } \{ I _ { t } = i \} } { q _ { t , i } + \rho _ { t , i } } } \end{array}$ for every $i \in [ K ]$   
$\underset {  } { q _ { t , i } + \rho _ { t , i } }$   
8 Set $\widehat { L } _ { t } = \widehat { L } _ { t - 1 } + \widehat { \ell _ { t } }$

where the inequality uses $Z _ { t } \le 1$ and $\begin{array} { r } { \sum _ { i } \sqrt { x _ { t , i } } \le \sqrt { K } } \end{array}$ . It therefore suffices to choose offsets satisfying the sufficient coordinate condition in Eq. (8) and $\begin{array} { r } { \sum _ { i = 1 } ^ { K } { \rho _ { t , i } } = \mathcal { O } ( \eta \sqrt { K } ) } \end{array}$ to control the normalization defect. We now choose $\rho _ { t , i } = 2 \eta \sqrt { x _ { t , i } }$ . Its sum is at most $2 \eta \sqrt { K }$ , so it meets both requirements. This choice simplifies $\operatorname { E q . } \left( 9 \right)$ to $\begin{array} { r } { q _ { i } ( Z ) = \left[ \frac { x _ { t , i } } { Z } - 2 \rho _ { t , i } \right] _ { + } } \end{array}$ , and the fixed-point equation becomes $\begin{array} { r } { { \cal Z } _ { t } = \sum _ { i = 1 } ^ { K } [ x _ { t , i } - 2 { \cal Z } _ { t } \rho _ { t , i } ] _ { + } } \end{array}$ . Recall that $q _ { t , i } = q _ { i } ( Z _ { t } )$ for all $i \in [ K ] . \operatorname { I f } q _ { t , i } > 0 .$ , then $x _ { t , i } = Z _ { t } ( q _ { t , i } + 2 \rho _ { t , i } )$ . Consequently, for every $i \in [ K ]$

$$
\frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + 2 \rho _ { t , i } } = Z _ { t } q _ { t , i } ,\tag{12}
$$

where the equality also holds trivially when $q _ { t , i } = 0$ . Combining Eq. (6), Eq. (8), and Eq. (12) proves the first target in Eq. (5). Moreover, $\begin{array} { r } { 1 - Z _ { t } = \sum _ { i = 1 } ^ { K } } \end{array}$ min $\begin{array} { r } { \{ x _ { t , i } , 2 Z _ { t } \rho _ { t , i } \} \le 2 Z _ { t } \sum _ { i = 1 } ^ { K } \rho _ { t , i } = \mathcal { O } ( \eta \sqrt { K } ) } \end{array}$ , which proves the second target in Eq. (5) and also gives $Z _ { t } \ge 1 / ( 1 + 4 \eta \sqrt { K } )$ . Picking $\begin{array} { r } { \eta = \Theta ( \frac { 1 } { \sqrt { K + L ^ { \star } } } ) } \end{array}$ then gives the target leading first-order term $\mathcal { O } ( \sqrt { K L ^ { \star } } )$ . To further achieve this target without knowing $L ^ { \star }$ , we now use an adaptive learning rate choice $\begin{array} { r } { \eta _ { t } = \frac { 1 } { \sqrt { K + \sum _ { s < t } \ell _ { s , I _ { s } } } } } \end{array}$ to adapt the regularization scale to the learner’s past observed losses. It turns out that this adaptive learning rate tuning also achieves the best-of-both-worlds guarantee as we will show in Section 4.

Based on the above analysis, our formal algorithm is given in Algorithm 1. At the beginning of round $t ,$ we compute $x _ { t }$ by FTRL with Tsallis entropy and the adaptive learning rate (Line 3). The fixed-point calculation in Line 4 then determines the sampling distribution $q _ { t }$ from $x _ { t }$ . Both $x _ { t }$ and $Z _ { t }$ involve solving one-dimensional monotone equations, in the simplex multiplier and in $Z .$ , respectively, so each round takes $\mathcal { O } ( K \log ( 1 / \epsilon ) )$ time by bisection to accuracy $\epsilon .$ After sampling $I _ { t }$ from $q _ { t }$ , we construct the IX-estimator $\widehat { \ell } _ { t , i }$ with $\rho _ { t , i } = 2 \eta _ { t } \sqrt { x _ { t , i } }$ and update the cumulative estimate. The following theorem gives the complete first-order guarantee.

## Theorem 3.1. Algorithm 1 guarantees that $\mathsf { R e g } _ { T , k } = \mathcal { O } ( \sqrt { K L _ { T , k } } + K )$ , for all $k \in [ K ]$

To our knowledge, this is the first explicit and efficient algorithm with a minimax optimal first-order guarantee $\mathcal { O } ( \sqrt { K L ^ { \star } } )$ for MAB, improving upon the $\mathcal { O } ( \operatorname* { m i n } \{ \sqrt { K T } , \sqrt { K L ^ { \star } } \log T \} )$ guarantee in Pogodin and Lattimore (2019).

## 3.3 Proof of Theorem 3.1

In this subsection, we provide a proof of Theorem 3.1.

ProofofTheorem 3.1. Via a standard analysis of FTRL with Tsallis entropy (included in Lemma $\mathbf { A . } 1 )$ , we know that

$$
\sum _ { t = 1 } ^ { T } \left. x _ { t } - u , \widehat { \ell } _ { t } \right. \leq \frac { \Psi ( u ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( u ) - \Psi ( x _ { t + 1 } ) \right) + \sum _ { t = 1 } ^ { T } \eta _ { t } \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \widehat { \ell } _ { t , i } ^ { 2 } ,\tag{13}
$$

for every $u \in \Delta _ { K }$ , where we define the last-round learning rate as $\eta _ { T + 1 } = 1 / \sqrt { K + \sum _ { t = 1 } ^ { T } \ell _ { t , I _ { t } } }$ . For $u = e _ { k }$ the first two terms on the right-hand side of Eq. (13) satisfy

$$
\begin{array} { r l } & { \frac { \Psi ( e _ { k } ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( e _ { k } ) - \Psi ( x _ { t + 1 } ) \right) } \\ & { \le 2 \sqrt { K } \left( \displaystyle \frac { 1 } { \eta _ { 1 } } + \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \right) = \frac { 2 \sqrt { K } } { \eta _ { T + 1 } } , } \end{array}
$$

where the inequality uses $\begin{array} { r } { \Psi ( e _ { k } ) - \Psi ( x ) = 2 ( \sum _ { i = 1 } ^ { K } \sqrt { x _ { i } } - 1 ) \le 2 \sqrt { K } } \end{array}$ for every $x \in \Delta _ { K }$ , and $\eta _ { t + 1 } ^ { - 1 } - \eta _ { t } ^ { - 1 } \geq 0$ because the observed losses are nonnegative. Taking expectations and using the fact that $\mathbb { E } _ { t } [ \widehat { \ell } _ { t , k } ] \leq \ell _ { t , k }$ therefore gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( \left. x _ { t } , \widehat { \ell } _ { t } \right. - \eta _ { t } \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \widehat { \ell } _ { t , i } ^ { 2 } ) \right] \leq L _ { T , k } + 2 \sqrt { K } \mathbb { E } \left[ \frac { 1 } { \eta _ { T + 1 } } \right] .
$$

According to the definition of $\widehat { \ell } _ { t , i } .$ , we have $\begin{array} { r } { \mathbb { E } _ { t } [ \widehat { \ell } _ { t , i } ] = \frac { q _ { t , i } \ell _ { t , i } } { q _ { t , i } + \rho _ { t , i } } } \end{array}$ and $\begin{array} { r } { \mathbb { E } _ { t } [ \widehat { \ell } _ { t , i } ^ { 2 } ] = \frac { q _ { t , i } \ell _ { t , i } ^ { 2 } } { ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 } } } \end{array}$ , which gives

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \left[ \left. x _ { t } , \widehat { \ell } _ { t } \right. \right] - \eta _ { t } \displaystyle \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \mathbb { E } _ { t } \left[ \widehat { \ell } _ { t , i } ^ { 2 } \right] \geq \displaystyle \sum _ { i = 1 } ^ { K } q _ { t , i } \ell _ { t , i } \left( \frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { \eta _ { t } x _ { t , i } ^ { 3 / 2 } } { \left( q _ { t , i } + \rho _ { t , i } \right) ^ { 2 } } \right) } \\ & { \qquad \geq \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } q _ { t , i } \ell _ { t , i } } { q _ { t , i } + 2 \rho _ { t , i } } = Z _ { t } \left. q _ { t } , \ell _ { t } \right. , } \end{array}
$$

where the first inequality uses $\ell _ { t , i } ^ { 2 } \ \leq \ \ell _ { t , i }$ , the second follows from the scalar inequality in $\operatorname { E q . } \ ( 8 )$ with $q _ { t , i } + \rho _ { t , i } \geq 2 \eta _ { t } \sqrt { x _ { t , i } }$ , and the equality follows from the sampling rule in Line 5. Since $\mathbb { E } _ { t } [ \ell _ { t , I _ { t } } ] = \langle q _ { t } , \ell _ { t } \rangle$ these inequalities imply $\begin{array} { r } { \mathbb { E } [ \sum _ { t } Z _ { t } \ell _ { t , I _ { t } } ] \le L _ { T , k } + 2 \sqrt { K } \mathbb { E } [ \eta _ { T + 1 } ^ { - 1 } ] } \end{array}$ . Moreover, the fixed-point calculation in Line 4 gives $\begin{array} { r } { Z _ { t } = \sum _ { i = 1 } ^ { K } [ x _ { t , i } - 2 Z _ { t } \rho _ { t , i } ] _ { + } } \end{array}$ . Therefore, we have

$$
\begin{array} { r l r } {  { 1 - Z _ { t } = \sum _ { i = 1 } ^ { K } ( x _ { t , i } - [ x _ { t , i } - 2 Z _ { t } \rho _ { t , i } ] _ { + } ) = \sum _ { i = 1 } ^ { K } \operatorname* { m i n } \{ x _ { t , i } , 2 Z _ { t } \rho _ { t , i } \} } } \\ & { } & { \leq 2 Z _ { t } \displaystyle \sum _ { i = 1 } ^ { K } \rho _ { t , i } = 4 \eta _ { t } Z _ { t } \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } \leq 4 \eta _ { t } \sqrt { K } , } \end{array}\tag{14}
$$

where the last inequality uses $Z _ { t } ~ \le ~ 1$ and $\textstyle \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } \ \leq \ \sqrt { K }$ . Adding the remaining learner’s loss E $\left[ \textstyle \sum _ { t = 1 } ^ { T } ( 1 - Z _ { t } ) \ell _ { t , I _ { t } } \right]$ and applying Eq. (14) yields

$$
\mathsf { R e g } _ { T , k } \leq 2 \sqrt { K } \mathbb { E } \left[ \frac { 1 } { \eta _ { T + 1 } } \right] + \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( 1 - Z _ { t } ) \ell _ { t , I _ { t } } \right]
$$

$$
\leq 2 \sqrt { K } \mathbb { E } \left[ \frac { 1 } { \eta _ { T + 1 } } \right] + 4 \sqrt { K } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \eta _ { t } \ell _ { t , I _ { t } } \right] .
$$

The learning rate in Line 3 gives $\ell _ { t , I _ { t } } = ( \eta _ { t + 1 } ^ { - 1 } - \eta _ { t } ^ { - 1 } ) ( \eta _ { t + 1 } ^ { - 1 } + \eta _ { t } ^ { - 1 } )$ . Consequently, we obtain that

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } \ell _ { t , I _ { t } } = \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( 1 + \frac { \eta _ { t } } { \eta _ { t + 1 } } \right) \leq 3 \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) = 3 \left( \frac { 1 } { \eta _ { T + 1 } } - \frac { 1 } { \eta _ { 1 } } \right) ,
$$

where the inequality uses $\eta _ { t } / \eta _ { t + 1 } = \sqrt { 1 + \eta _ { t } ^ { 2 } \ell _ { t , I _ { t } } } \leq 2$ , since $\eta _ { t } \leq 1$ and $\ell _ { t , I _ { t } } \leq 1$ . Substituting this bound into the regret inequality gives

$$
\mathsf { R e g } _ { T , k } \leq \mathcal { O } \left( \sqrt { K } \mathbb { E } \left[ \sqrt { K + \sum _ { t = 1 } ^ { T } \ell _ { t , I _ { t } } } \right] \right) .\tag{15}
$$

Jensen’s inequality and $\begin{array} { r } { \mathbb E [ \sum _ { t } \ell _ { t , I _ { t } } ] = L _ { T , k } + \mathsf { R e g } _ { T , k } } \end{array}$ imply $\mathsf { R e g } _ { T , k } \le C \sqrt { K ( K + L _ { T , k } + \mathsf { R e g } _ { T , k } ) }$ for some universal constant $C > 0$ . If the regret is negative, Theorem 3.1 follows immediately. Otherwise, applying the AM-GM inequality gives $C \sqrt { K \mathsf { R e g } _ { T , k } } \le \mathsf { R e g } _ { T , k } / 2 + C ^ { 2 } K / 2$ , which further gives $\mathsf { R e g } _ { T , k } \leq \mathsf { R e g } _ { T , k } / 2 +$ $\mathcal { O } ( \sqrt { K L _ { T , k } } + K )$ . Rearranging the terms gives $\mathsf { R e g } _ { T , k } \leq \mathcal { O } ( \sqrt { K L _ { T , k } } + K )$ , proving Theorem 3.1.

## 4 Logarithmic Regret in the Stochastic Environment

In this section, we prove that the same algorithm in Algorithm 1 also achieves logarithmic gap-dependent regret in the stochastic environment. Together with Theorem 3.1, this gives a best-of-both-worlds guarantee with the optimal leading first-order term in the adversarial environment. Formally, we have the following theorem.

Theorem 4.1. For $K \geq 2$ and $T > 2 ,$ , suppose that the loss vectors in $[ 0 , 1 ] ^ { K }$ are i.i.d. with a unique best arm $i ^ { \star } .$ . Algorithm 1 guarantees that

$$
\mathsf { R e g } _ { T } = \mathcal { O } \left( \sum _ { i = 1 , i \neq i ^ { \star } } ^ { K } \frac { \log T } { \Delta _ { i } } \right) .
$$

To prove Theorem 4.1, we follow the approach of Zimmert and Seldin (2021) and derive a regret bound in terms of suboptimal-arm probabilities, which allows us to apply the self-bounding identity in Eq. (1). Starting from the FTRL bound in Lemma A.1, we account for the estimation bias and the difference between the FTRL update $x _ { t }$ and the decision $q _ { t }$ to obtain the following lemma.

## Lemma 4.2. Under the assumptions ofTheorem 4.1, Algorithm 1 guarantees that

$$
\begin{array} { r l r } & { } & { \mathsf { R e g } _ { T } \leq \frac { \Psi \left( e _ { i ^ { \star } } \right) - \Psi \left( x _ { 1 } \right) } { \eta _ { 1 } } + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { t + 1 } ) \right) \right] } \\ & { } & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ ( 1 - Z _ { t } ) \displaystyle \sum _ { i = 1 } ^ { K } \Delta _ { i } q _ { t , i } \right] + 1 4 \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \eta _ { t } \mu _ { i ^ { \star } } \displaystyle \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } \right] . } \end{array}\tag{16}
$$

We first show how Lemma 4.2 implies Theorem 4.1 and then sketch the proof of Lemma 4.2. The detailed conditional-expectation calculations are given in Appendix B.2.

ProofofTheorem 4.1. We first analyze the first two terms in Eq. (16). Since $x _ { 1 }$ is uniform and $\eta _ { 1 } = 1 / \sqrt { K }$ direct calculation gives that

$$
\frac { \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } = \sqrt { K } \left( - 2 + 2 \sqrt { K } \right) \leq 2 K .
$$

For the second term, we show in Lemma B.1 that $x _ { t + 1 , i } \leq 6 x _ { t , i }$ for all $i \in [ K ]$ ]. The learning rate in Line 3 also satisfies $\eta _ { t + 1 } \leq \eta _ { t }$ and $\begin{array} { r } { \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } = \frac { \ell _ { t , I _ { t } } } { \frac { 1 } { \eta _ { t + 1 } } + \frac { 1 } { \eta _ { t } } } = \frac { \eta _ { t } \eta _ { t + 1 } \ell _ { t , I _ { t } } } { \eta _ { t } + \eta _ { t + 1 } } \leq \frac { \eta _ { t } \ell _ { t , I _ { t } } } { 2 } } \end{array}$ . We therefore obtain

$$
\begin{array} { l } { \displaystyle \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) ( \Psi ( e _ { i \star } ) - \Psi ( x _ { t + 1 } ) ) } \\ { = \displaystyle 2 \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \displaystyle \sum _ { i = 1 } ^ { K } \sqrt { x _ { t + 1 , i } } - 1 \right) \le 2 \sqrt { 6 } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \displaystyle \sum _ { i \ne i } ^ { K } \sqrt { x _ { t , i } } \le 3 \eta _ { t } \ell _ { t , I _ { t } } \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } , } \end{array}
$$

where the first inequality uses $\sqrt { x _ { t + 1 , i ^ { \star } } } \leq 1$ and Lemma B.1, and the second uses $1 / \eta _ { t + 1 } - 1 / \eta _ { t } \leq \eta _ { t } \ell _ { t , I _ { t } } / 2$ Summing and taking expectations with $\mathbb { E } [ \ell _ { t , I _ { t } } \mid \mathcal { F } _ { t - 1 } ] = \langle q _ { t } , \mu \rangle$ further gives

$$
\begin{array} { l } { \displaystyle \frac { \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \mathbb { E } \left[ \sum _ { { t = 1 } } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { t + 1 } ) \right) \right] } \\ { \displaystyle \le 2 K + 3 \sum _ { { t = 1 } } ^ { T } \mathbb { E } \left[ \eta _ { t } \langle q _ { t } , \mu \rangle \sum _ { i = 1 \atop i \ne i ^ { \star } } ^ { K } \sqrt { x _ { t , i } } \right] . } \end{array}\tag{17}
$$

For the third term of Eq. (16), we consider two cases separately. When $\eta _ { t } > 1 / ( 8 \sqrt { K } )$ , the learning-rate definition gives $\begin{array} { r } { \sum _ { \tau < t } \ell _ { \tau , I _ { \tau } } < 6 3 K } \end{array}$ . These rounds form an initial interval, and its final loss increment is at most one. Consequently, on every realization, $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \mathbb { I } \left\{ \eta _ { t } > \frac { 1 } { 8 \sqrt { K } } \right\} \ell _ { t , I _ { t } } \leq 6 3 K + 1 } \end{array}$ . The contribution of these rounds satisfies

$$
\begin{array} { r } { \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mathbb { I } \left\{ \eta _ { t } > \frac { 1 } { 8 \sqrt { K } } \right\} ( 1 - Z _ { t } ) \displaystyle \sum _ { i = 1 \atop i \neq i ^ { * } } ^ { K } \Delta _ { i } q _ { t , i } \right] \leq \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mathbb { I } \left\{ \eta _ { t } > \frac { 1 } { 8 \sqrt { K } } \right\} \mathbb { E } [ \ell _ { t , I _ { t } } \ \big \vert \ \mathcal { F } _ { t - 1 } ] \right] } \\ { = \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { I } \left\{ \eta _ { t } > \frac { 1 } { 8 \sqrt { K } } \right\} \ell _ { t , I _ { t } } \right] \leq 6 3 K + 1 , } \end{array}
$$

where the first inequality uses $\begin{array} { r } { \sum _ { i \neq i ^ { \star } } \Delta _ { i } q _ { t , i } \leq \langle q _ { t } , \mu \rangle = \mathbb { E } [ \ell _ { t , I _ { t } } \mid \mathcal { F } _ { t - 1 } ] } \end{array}$ , the equality uses the tower property and $\mathcal { F } _ { t - 1 }$ -measurability of $\eta _ { t }$ , and the last inequality uses the preceding cumulative-loss bound. When $\eta _ { t } \leq 1 / ( 8 \sqrt { K } ) , \mathrm { E q }$ . (14) gives $1 - Z _ { t } \le 4 \eta _ { t } \sqrt { K } \le 1 / 2$ . Thus

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mathbb { I } \left\{ \eta _ { t } \leq \frac { 1 } { 8 \sqrt { K } } \right\} \left( 1 - Z _ { t } \right) \sum _ { \stackrel { i = 1 } { i \neq i } ^ { 1 } } ^ { K } \Delta _ { i } q _ { t , i } \right] \leq \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mathbb { I } \left\{ \eta _ { t } \leq \frac { 1 } { 8 \sqrt { K } } \right\} \sum _ { \stackrel { i = 1 } { i \neq i } ^ { 1 } } ^ { K } \Delta _ { i } q _ { t , i } \right] \leq \frac { \mathsf { R e g } _ { T } } { 2 } ,
$$

where the last inequality follows from Eq. (1) and the nonnegativity of the gaps. Adding the two cases yields

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \left[ ( 1 - Z _ { t } ) \sum _ { \stackrel { i = 1 } { i \neq i ^ { \star } } } ^ { K } \Delta _ { i } q _ { t , i } \right] \leq \frac { \mathsf { R e g } _ { T } } { 2 } + 6 3 K + 1 .\tag{18}
$$

Substituting Eq. (17) and Eq. (18) into Eq. (16), and using $\mu _ { i ^ { \star } } \leq \langle q _ { t } , \mu \rangle$ , gives

$$
\mathsf { R e g } _ { T } \leq \frac { \mathsf { R e g } _ { T } } { 2 } + \mathcal { O } ( K ) + 1 7 \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \eta _ { t } \left. q _ { t } , \mu \right. \sum _ { i = 1 \atop i \neq i ^ { \star } } ^ { K } \sqrt { x _ { t , i } } \right] .\tag{19}
$$

To express this bound through the sampling probability $q _ { t } ,$ , note that according to the definition of $q _ { t }$ in Line $5 ,$ we have

$$
q _ { t , i } = \left[ \frac { x _ { t , i } } { Z _ { t } } - 4 \eta _ { t } \sqrt { x _ { t , i } } \right] _ { + } \geq \frac { x _ { t , i } } { Z _ { t } } - 4 \eta _ { t } \sqrt { x _ { t , i } } \geq x _ { t , i } - 4 \eta _ { t } \sqrt { x _ { t , i } } ,
$$

where the second inequality uses $Z _ { t } \leq 1$ . Completing the square gives $( \sqrt { x _ { t , i } } - 2 \eta _ { t } ) ^ { 2 } \leq q _ { t , i } + 4 \eta _ { t } ^ { 2 }$ , and hence $\sqrt { x _ { t , i } } \le 2 \eta _ { t } + \sqrt { q _ { t , i } + 4 \eta _ { t } ^ { 2 } } \le \sqrt { q _ { t , i } } + 4 \eta _ { t }$ . Substituting this into Eq. (19) and rearranging the terms yields

$$
\mathsf { R e g } _ { T } \leq \mathcal O ( K ) + 3 4 \sum _ { t = 1 } ^ { T } \mathbb E \left[ \eta _ { t } \left. q _ { t } , \mu \right. \sum _ { i = 1 } ^ { K } \sqrt { q _ { t , i } } \right] + 4 \cdot 3 4 ( K - 1 ) \sum _ { t = 1 } ^ { T } \mathbb E \left[ \eta _ { t } ^ { 2 } \left. q _ { t } , \mu \right. \right] .\tag{20}
$$

To bound the last term in Eq. (20), direct calculation shows that

$$
\sum _ { t = 1 } ^ { T } \eta _ { t } ^ { 2 } \ell _ { t , I _ { t } } \leq 2 \sum _ { t = 1 } ^ { T } \log \left( \frac { \eta _ { t } ^ { 2 } } { \eta _ { t + 1 } ^ { 2 } } \right) = 2 \log \left( \frac { \eta _ { 1 } ^ { 2 } } { \eta _ { T + 1 } ^ { 2 } } \right) \leq 2 \log \left( 1 + \frac { T } { K } \right) \leq 2 \log T ,
$$

where the first inequality applies $u \leq 2 \log ( 1 + u )$ with $u = \eta _ { t } ^ { 2 } \ell _ { t , I _ { t } } \leq 1 / 2$ and $1 + u = \eta _ { t } ^ { 2 } / \eta _ { t + 1 } ^ { 2 }$ , and the last two inequalities use $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \ell _ { t , I _ { t } } \leq T , K \geq 2 } \end{array}$ , and $T > 2$ . Since $\eta _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable, we have

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \eta _ { t } ^ { 2 } \langle q _ { t } , \mu \rangle \right] = \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \eta _ { t } ^ { 2 } \mathbb { E } [ \ell _ { t , I _ { t } } \mid \mathcal { F } _ { t - 1 } ] \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \eta _ { t } ^ { 2 } \ell _ { t , I _ { t } } \right] \leq 2 \log T .\tag{21}
$$

Applying Eq. (21) to Eq. (20) gives

$$
\mathsf { R e g } _ { T } \leq \mathcal O ( K \log T ) + 3 4 \sum _ { \ell = 1 } ^ { T } \sum _ { \stackrel { i = 1 } { i \neq i ^ { \star } } } ^ { K } \mathbb E [ \eta _ { t } \left. q _ { t } , \mu \right. \sqrt { q _ { t , i } } ] .\tag{22}
$$

For each $i \neq i ^ { \star }$ , the AM-GM inequality gives 34η<sub>t</sub> $\begin{array} { r } { \left. { q _ { t } , \mu } \right. \sqrt { q _ { t , i } } \le \frac { \Delta _ { i } q _ { t , i } } { 2 } + \frac { 3 4 ^ { 2 } \eta _ { t } ^ { 2 } \left. { q _ { t } , \mu } \right. ^ { 2 } } { 2 \Delta _ { i } } \le \frac { \Delta _ { i } q _ { t , i } } { 2 } + \frac { 3 4 ^ { 2 } \eta _ { t } ^ { 2 } \left. { q _ { t } , \mu } \right. } { 2 \Delta _ { i } } } \end{array}$ Summing these bounds and applying Eq. (1) and Eq. (21) yields

$$
3 4 \sum _ { t = 1 } ^ { T } \sum _ { i = 1 \atop i \neq i } ^ { K } \mathbb { E } [ \eta _ { t } \left. q _ { t } , \mu \right. \sqrt { q _ { t , i } } ] \leq \frac { \mathsf { R e g } _ { T } } { 2 } + \frac { 3 4 ^ { 2 } } { 2 } \sum _ { i = 1 } ^ { K } \frac { 1 } { \Delta _ { i } } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \eta _ { t } ^ { 2 } \left. q _ { t } , \mu \right. ] \leq \frac { \mathsf { R e g } _ { T } } { 2 } + 3 4 ^ { 2 } \sum _ { i = 1 } ^ { K } \frac { \log T } { \Delta _ { i } } .
$$

Since $0 < \Delta _ { i } \le 1$ and $K \ge 2 , \mathcal { O } ( K \log T )$ is bounded by $\begin{array} { r } { \mathcal { O } ( \sum _ { i \neq i ^ { \star } } \frac { \log T } { \Delta _ { i } } ) } \end{array}$ . Substituting the preceding inequality into Eq. (22) therefore gives $\begin{array} { r } { \mathsf { R e g } _ { T } \leq \mathsf { R e g } _ { T } / 2 + \mathcal { O } ( \sum _ { i \neq i ^ { \star } } \frac { \log T } { \Delta _ { i } } ) } \end{array}$ . Rearranging the terms proves Theorem 4.1. □

## 4.1 Proof Sketch of Lemma 4.2

In this section, we provide a proof sketch of Lemma 4.2, with the complete proof deferred to Appendix B.2. By adapting the FTRL analysis with Tsallis entropy to our loss estimator, we show in Lemma A.1 that

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \left. x _ { t } - e _ { i ^ { \star } } , \widehat { \ell } _ { t } \right. \leq \frac { \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { t + 1 } ) \right) } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , t _ { t } } ^ { 3 / 2 } , \displaystyle \sum _ { i = 1 , i \neq I _ { t } } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \widehat { \ell } _ { t , I _ { t } } ^ { 2 } . } \end{array}\tag{23}
$$

Taking the conditional expectation of the final summand, with some calculation, one can obtain that

$$
\mathbb { E } \left[ \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , I _ { t } } ^ { 3 / 2 } , \otimes \sum _ { i = 1 , i \neq l _ { t } } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \widehat { \ell } _ { t , I _ { t } } ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right] \leq Z _ { t } \sum _ { i = 1 , i \neq i ^ { \star } } ^ { K } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } + 1 0 \eta _ { t } \mu _ { i ^ { \star } } \sum _ { i = 1 , i \neq i ^ { \star } } ^ { K } \sqrt { x _ { t , i } } .\tag{24}
$$

In order to transform the left hand side of Eq. (23) to $\mathsf { R e g } _ { T }$ , we need to account for the estimator and the change from $x _ { t }$ to $q _ { t }$ . Using Eq. (12) and $\mu _ { i } = \mu _ { i ^ { \star } } + \Delta _ { i }$ , one can show that

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle q _ { t } - e _ { i ^ { \star } } , \ell _ { t } \rangle - \left. x _ { t } - e _ { i ^ { \star } } , \widehat { \ell } _ { t } \right. \bigm | \mathcal { F } _ { t - 1 } \right] } \\ & { = ( 1 - Z _ { t } ) \displaystyle \sum _ { i = 1 } ^ { K } \Delta _ { i } q _ { t , i } + \mu _ { i ^ { \star } } \left( \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { \left( 1 - x _ { t , i ^ { \star } } \right) \rho _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } \right) - Z _ { t } \displaystyle \sum _ { i = 1 } ^ { K } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } . } \end{array}\tag{25}
$$

Combining Eq. (23), Eq. (24), and Eq. (25), noticing that the negative term Eq. (25) cancels the first term in Eq. (24), we can bound $\mathsf { R e g } _ { T }$ as follows:

$$
\begin{array} { r l } & { \mathsf { R e g } _ { T ^ { \prime } } = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. x _ { t } - e _ { t } , \widehat { \varepsilon } _ { t } \right. \right] + \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. q _ { t } - e _ { t } , \widehat { \varepsilon } _ { t } \right. - \left. x _ { t } - e _ { t } , \widehat { \varepsilon } _ { t } \right. \right] } \\ & { \qquad \le \displaystyle \frac { \mathbb { P } \left( e _ { t } \right) - \mathbb { P } \left( x _ { 1 } \right) } { \eta _ { t } } + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( e _ { t } * ) - \Psi ( x _ { t + 1 } ) \right) \right] } \\ & { \qquad + \displaystyle 1 0 \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \eta _ { t } \mu _ { t } , \displaystyle \sum _ { i = 1 , i \neq j } ^ { K } \sqrt { x _ { t , i } } \right] } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ ( 1 - Z _ { t } ) \displaystyle \sum _ { i = 1 } ^ { K } \Delta _ { q , i } { e _ { i } } _ { i } + \mu _ { t } \left( \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { i , i } \rho _ { i , i } } { \xi _ { t , i } } - \displaystyle \frac { ( 1 - x _ { t , i } * ) \rho _ { i , i } * } { q _ { t } , i * } \right) \right] , } \end{array}
$$

Moreover, as $\begin{array} { r } { q _ { t , i } = [ \frac { x _ { t , i } } { Z _ { t } } - 2 \rho _ { t , i } ] _ { + } \geq \frac { x _ { t , i } } { Z _ { t } } - 2 \rho _ { t , i } } \end{array}$ , we know that

$$
\frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \leq \frac { Z _ { t } ( q _ { t , i } + 2 \rho _ { t , i } ) \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \leq 2 Z _ { t } \rho _ { t , i } \leq 4 \eta _ { t } \sqrt { x _ { t , i } } ,
$$

since $Z _ { t } \le 1$ and $\rho _ { t , i } = 2 \eta _ { t } \sqrt { x _ { t , i } }$ . Applying the preceding bound and dropping the non-positive terms lead to Eq. (16).

## 5 Conclusion

In this paper, we design an algorithm with the minimax-optimal leading first-order regret term $\mathcal { O } ( \sqrt { K L ^ { \star } } )$ in the adversarial environment. To our knowledge, this is the first efficient algorithm to achieve this minimax optimal first-order scale without a logarithmic factor. Moreover, the same algorithm simultaneously achieves $\begin{array} { r } { \hat { \mathcal { O } } ( \sum _ { i \neq i ^ { \star } } \frac { \log T } { \Delta _ { i } } ) } \end{array}$ regret in the stochastic environment with a unique best arm, giving a best-of-both-worlds guarantee. Technically, we build on FTRL with Tsallis entropy and use a scalar fixed point to couple sampling with implicit exploration. This coupling controls estimation bias and stability jointly, while a learning rate based on the learner’s observed cumulative loss adapts to both environments. Future work includes extending this approach to richer partial-feedback models and obtaining high-probability guarantees.

## AI Disclosure

The authors used GPT-6 Astra to assist with exploring proof strategies and checking mathematical derivations. The authors also use Opus 5.5 and GPT-6 Astra for identifying relevant literature and polishing the manuscript. The authors directed the research, critically examined and revised the proposed arguments, and iteratively refined the algorithm, proofs, and exposition. All decisions about the results and the presentation were made by the authors, who are responsible for the correctness, originality, and attribution of all content.

## References

Zeyuan Allen-Zhu, Sébastien Bubeck, and Yuanzhi Li. Make the minority great again: First-order regret bound for contextual bandits. In International Conference on Machine Learning, 2018.

Chamy Allenberg, Peter Auer, László Györfi, and György Ottucsák. Hannan consistency in on-line learning in case of unbounded losses under partial monitoring. In Algorithmic Learning Theory, 2006.

Jean-Yves Audibert and Sébastien Bubeck. Minimax policies for adversarial and stochastic bandits. In Proceedings of the 22nd Annual Conference on Learning Theory, 2009.

Peter Auer and Chao-Kai Chiang. An algorithm with nearly optimal pseudo-regret for both stochastic and adversarial bandits. In Proceedings ofthe 29th Conference on Learning Theory, 2016.

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 2002a.

Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 2002b.

Sébastien Bubeck and Mark Sellke. First-order Bayesian regret analysis of Thompson sampling. In International Conference on Algorithmic Learning Theory, 2020.

Sébastien Bubeck and Aleksandrs Slivkins. The best of both worlds: Stochastic and adversarial bandits. In Proceedings of the 25th Annual Conference on Learning Theory, 2012.

Botao Chen, Jongyeong Lee, and Junya Honda. Geometric resampling in nearly linear time for followthe-perturbed-leader with best-of-both-worlds guarantee in bandit problems. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

Botao Chen, Jongyeong Lee, Chansoo Kim, and Junya Honda. A further efficient algorithm with best-ofboth-worlds guarantees for m-set semi-bandit problem. arXiv preprint arXiv:2603.11764, 2026.

Christoph Dann, Chen-Yu Wei, and Julian Zimmert. A blackbox approach to best of both worlds in bandits and beyond. In Proceedings ofthe 36th Conference on Learning Theory, 2023.

Dylan J. Foster and Akshay Krishnamurthy. Efficient first-order contextual bandits: Prediction, allocation, and triangular discrimination. In Advances in Neural Information Processing Systems, 2021.

Dylan J. Foster, Zhiyuan Li, Thodoris Lykouris, Karthik Sridharan, and Éva Tardos. Learning in games: Robustness of fast convergence. In Advances in Neural Information Processing Systems, 2016.

Sébastien Gerchinovitz and Tor Lattimore. Refined lower bounds for adversarial bandits. In Advances in Neural Information Processing Systems, 2016.

Junya Honda, Shinji Ito, and Taira Tsuchiya. Follow-the-perturbed-leader achieves best-of-both-worlds for bandit problems. In Proceedings of the 34th International Conference on Algorithmic Learning Theory, 2023.

Shinji Ito. Parameter-free multi-armed bandit algorithms with hybrid data-dependent regret bounds. In Proceedings of the 34th Conference on Learning Theory, 2021.

Shinji Ito, Shuichi Hirahara, Tasuku Soma, and Yuichi Yoshida. Tight first- and second-order regret bounds for adversarial linear bandits. In Advances in Neural Information Processing Systems, 2020.

Shinji Ito, Taira Tsuchiya, and Junya Honda. Nearly optimal best-of-both-worlds algorithms for online learning with feedback graphs. In Advances in Neural Information Processing Systems, 2022a.

Shinji Ito, Taira Tsuchiya, and Junya Honda. Adversarially robust multi-armed bandit algorithm with variance-dependent regret bounds. In Proceedings ofthe 35th Conference on Learning Theory, 2022b.

Shinji Ito, Taira Tsuchiya, and Junya Honda. Adaptive learning rate for follow-the-regularized-leader: Competitive analysis and best-of-both-worlds. In Proceedings of the 37th Conference on Learning Theory, 2024.

Shinji Ito, Kevin G. Jamieson, Haipeng Luo, Arnab Maiti, and Taira Tsuchiya. Adapting to stochastic and adversarial losses in episodic MDPs with aggregate bandit feedback. In Advances in Neural Information Processing Systems, 2025.

Tiancheng Jin and Haipeng Luo. Simultaneously learning stochastic and adversarial episodic MDPs with known transition. In Advances in Neural Information Processing Systems, 2020.

Tiancheng Jin, Longbo Huang, and Haipeng Luo. The best of both worlds: Stochastic and adversarial episodic MDPs with unknown transition. In Advances in Neural Information Processing Systems, 2021.

Tiancheng Jin, Junyan Liu, and Haipeng Luo. Improved best-of-both-worlds guarantees for multi-armed bandits: FTRL with general regularizers and multiple optimal arms. In Advances in Neural Information Processing Systems, 2023.

Masahiro Kato and Shinji Ito. LC-Tsallis-INF: Generalized best-of-both-worlds linear contextual bandits. In Proceedings ofthe 28th International Conference on Artificial Intelligence and Statistics, 2025.

Tomáš Kocák, Gergely Neu, Michal Valko, and Rémi Munos. Efficient learning by implicit exploration in bandit problems with side observations. In Advances in Neural Information Processing Systems, 2014.

Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, and Mengxiao Zhang. Bias no more: High-probability datadependent regret bounds for adversarial bandits and MDPs. In Advances in Neural Information Processing Systems, 2020a.

Chung-Wei Lee, Haipeng Luo, and Mengxiao Zhang. A closer look at small-loss bounds for bandits with graph feedback. In Proceedings of the 33rd Conference on Learning Theory, 2020b.

Jongyeong Lee, Junya Honda, Shinji Ito, and Min-hwan Oh. Follow-the-perturbed-leader with Fréchet-type tail distributions: Optimality in adversarial bandits and best-of-both-worlds. In Proceedings ofthe 37th Conference on Learning Theory, 2024.

Jongyeong Lee, Junya Honda, Shinji Ito, and Min-hwan Oh. Revisiting follow-the-perturbed-leader with unbounded perturbations in bandit problems. In Advances in Neural Information Processing Systems, 2025.

Jongyeong Lee, Junya Honda, Shinji Ito, and Chansoo Kim. Adaptive learning rates with surrogate probability for follow-the-perturbed-leader. In Proceedings ofthe 39th Conference on Learning Theory, 2026.

Mingyi Li, Taira Tsuchiya, and Kenji Yamanishi. Data- and variance-dependent regret bounds for online tabular MDPs. In Proceedings ofthe 43rd International Conference on Machine Learning, 2026.

Thodoris Lykouris, Karthik Sridharan, and Éva Tardos. Small-loss bounds for online learning with partial information. In Proceedings of the 31st Conference on Learning Theory, 2018.

Saeed Masoudian, Julian Zimmert, and Yevgeny Seldin. A best-of-both-worlds algorithm for bandits with delayed feedback. In Advances in Neural Information Processing Systems, 2022.

Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. In Advances in Neural Information Processing Systems, 2015a.

Gergely Neu. First-order regret bounds for combinatorial semi-bandits. In Proceedings of the 28th Conference on Learning Theory, 2015b.

Quan Nguyen, Shinji Ito, Junpei Komiyama, and Nishant A. Mehta. Data-dependent bounds with T-optimal best-of-both-worlds guarantees in multi-armed bandits using stability-penalty matching. In Proceedings of the 38th Conference on Learning Theory, 2025.

Julia Olkhovskaya, Jack Mayo, Tim van Erven, Gergely Neu, and Chen-Yu Wei. First- and second-order bounds for adversarial linear contextual bandits. In Advances in Neural Information Processing Systems, 2023.

Roman Pogodin and Tor Lattimore. On first-order bounds, variance and gap-dependent bounds for adversarial bandits. In Proceedings ofthe 35th Conference on Uncertainty in Artificial Intelligence, 2019.

Ofir Schlisselberg, Tal Lancewicki, Peter Auer, and Yishay Mansour. Improved best-of-both-worlds regret for bandits with delayed feedback. In Advances in Neural Information Processing Systems, 2025.

Yevgeny Seldin and Gábor Lugosi. An improved parametrization and analysis of the EXP3++ algorithm for stochastic and adversarial bandits. In Proceedings ofthe 30th Conference on Learning Theory, 2017.

Yevgeny Seldin and Aleksandrs Slivkins. One practical algorithm for both stochastic and adversarial bandits. In Proceedings ofthe 31st International Conference on Machine Learning, 2014.

Tim van Erven, Jack Mayo, Julia Olkhovskaya, and Chen-Yu Wei. An improved algorithm for adversarial linear contextual bandits via reduction. In Advances in Neural Information Processing Systems, 2025.

Chen-Yu Wei and Haipeng Luo. More adaptive algorithms for adversarial bandits. In Proceedings of the 31st Conference on Learning Theory, 2018.

Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal ofMachine Learning Research, 2021.

Julian Zimmert, Haipeng Luo, and Chen-Yu Wei. Beating stochastic and adversarial semi-bandits optimally and simultaneously. In Proceedings of the 36th International Conference on Machine Learning, 2019.

## A Omitted Details in Section 3

In this section, we prove the FTRL bound used in Section 3 and Section 4, which follows the analysis of FTRL with Tsallis entropy with an adaptation to our IX-estimator.

Lemma A.1. Algorithm 1 guarantees that

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \left. x _ { t } - u , \widehat { \ell } _ { t } \right. \leq \frac { \Psi ( u ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( u ) - \Psi ( x _ { t + 1 } ) \right) } \\ & { \quad \quad \quad + \displaystyle \sum _ { t = 1 } ^ { T } \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , I _ { t } } ^ { 3 / 2 } , 8 \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \widehat { \ell } _ { t , I _ { t } } ^ { 2 } , } \end{array}
$$

for every $u \in \Delta _ { K }$

Proof. Denote by

$$
\Phi _ { t } \triangleq \operatorname* { m i n } _ { z \in \Delta _ { K } } \left\{ \left. z , \widehat { L } _ { t - 1 } \right. + \frac { \Psi ( z ) } { \eta _ { t } } \right\} = \left. x _ { t } , \widehat { L } _ { t - 1 } \right. + \frac { \Psi ( x _ { t } ) } { \eta _ { t } }
$$

the FTRL potential at round t. The optimality condition for $x _ { t }$ gives $\left. \widehat { L } _ { t - 1 } + \nabla \Psi ( x _ { t } ) / \eta _ { t } , z - x _ { t } \right. = 0$ for every $z \in \Delta _ { K }$ . Expanding the Bregman divergence and using $\widehat { L } _ { t } = \widehat { L } _ { t - 1 } + \widehat { \ell } _ { t }$ therefore gives the identity

$$
\Phi _ { t + 1 } - \Phi _ { t } = \left. x _ { t + 1 } , \widehat { \ell } _ { t } \right. + D _ { \eta _ { t } ^ { - 1 } \Psi } ( x _ { t + 1 } , x _ { t } ) + \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \Psi ( x _ { t + 1 } ) .
$$

Rearranging the terms and bounding the expression at $x _ { t + 1 }$ by its maximum over the simplex yields

$$
\left. x _ { t } , \widehat { \ell } _ { t } \right. \leq \Phi _ { t + 1 } - \Phi _ { t } - \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \Psi ( x _ { t + 1 } ) + \operatorname* { m a x } _ { z \in \Delta _ { K } } \left\{ \left. x _ { t } - z , \widehat { \ell } _ { t } \right. - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \right\} .
$$

Since $\Phi _ { 1 } = \Psi ( x _ { 1 } ) / \eta _ { 1 }$ and $\Phi _ { T + 1 } \le \left. u , \widehat { L } _ { T } \right. + \Psi ( u ) / \eta _ { T + 1 }$ , summing this inequality and subtracting $\left. u , \widehat { L } _ { T } \right.$ gives

$$
\sum _ { t = 1 } ^ { T } \left. x _ { t } - u , \widehat { \ell } _ { t } \right. \leq \frac { \Psi ( u ) } { \eta _ { T + 1 } } - \frac { \Psi ( x _ { 1 } ) } { \eta _ { 1 } } - \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \Psi ( x _ { t + 1 } )
$$

$$
+ \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { z \in \Delta _ { K } } \Big \{ \Big \langle x _ { t } - z , \widehat { \ell } _ { t } \Big \rangle - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \Big \} .\tag{26}
$$

Using $\begin{array} { r } { 1 / \eta _ { T + 1 } = 1 / \eta _ { 1 } + \sum _ { t = 1 } ^ { T } ( 1 / \eta _ { t + 1 } - 1 / \eta _ { t } ) } \end{array}$ gives the first two terms in the lemma.

It remains to bound the maximization term in Eq. (26). Fix a round t. For $z _ { I _ { t } } \geq x _ { t , I _ { t } }$ , this term is nonpositive. For $z _ { I _ { t } } < x _ { t , I _ { t } }$ , we have

$$
D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \geq \frac { 1 } { 2 \eta _ { t } } \int _ { z _ { I _ { t } } } ^ { x _ { t , I _ { t } } } \frac { s - z _ { I _ { t } } } { s ^ { 3 / 2 } } d s \geq \frac { ( x _ { t , I _ { t } } - z _ { I _ { t } } ) ^ { 2 } } { 4 \eta _ { t } x _ { t , I _ { t } } ^ { 3 / 2 } } ,
$$

where the first inequality uses separability of $\Psi$ and nonnegativity of the other coordinate divergences, and the second uses $s \leq x _ { t , I _ { t } }$ throughout the integral. Completing the square gives

$$
\left. x _ { t } - z , \widehat { \ell } _ { t } \right. - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \le \widehat { \ell } _ { t , I _ { t } } ( x _ { t , I _ { t } } - z _ { I _ { t } } ) - \frac { ( x _ { t , I _ { t } } - z _ { I _ { t } } ) ^ { 2 } } { 4 \eta _ { t } x _ { t , I _ { t } } ^ { 3 / 2 } } \le \eta _ { t } x _ { t , I _ { t } } ^ { 3 / 2 } \widehat { \ell } _ { t , I _ { t } } ^ { 2 } .\tag{27}
$$

For the second bound, denote by $z _ { t } \triangleq \operatorname { a r g m a x } _ { z \in \Delta _ { K } } \left\{ \left. x _ { t } - z , \widehat { \ell } _ { t } \right. - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \right\}$ the maximizer. Direct calculation shows that

$$
\operatorname* { m a x } _ { z \in \Delta _ { K } } \Big \{ \Big \langle x _ { t } - z , \widehat { \ell } _ { t } \Big \rangle - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \Big \} \leq \Big \langle x _ { t } - z _ { t } , \widehat { \ell } _ { t } \Big \rangle = \widehat { \ell } _ { t , I _ { t } } ( x _ { t , I _ { t } } - z _ { t , I _ { t } } ) = \widehat { \ell } _ { t , I _ { t } } \sum _ { \underset { i \neq I _ { t } } { i = 1 } } ^ { K } ( z _ { t , i } - x _ { t , i } ) ,\tag{28}
$$

where the inequality uses nonnegativity of the Bregman divergence, the first equality uses $\widehat { \ell } _ { t , i } = 0$ for $i \neq I _ { t }$ and the second uses $\begin{array} { r } { \sum _ { i } x _ { t , i } = \sum _ { i } z _ { t , i } = 1 } \end{array}$ . Consider two cases. If ${ { I } _ { t } } \mathrm { ~ \ne ~ }$ argmax<sub>i∈[K]</sub> x<sub>t,i</sub>, then the first bound Eq. (27) implies the second because $\begin{array} { r } { x _ { t , I _ { t } } ^ { 3 / 2 } \leq \sum _ { i \neq I _ { t } } x _ { t , i } ^ { 3 / 2 } } \end{array}$ . Suppose now that $I _ { t } = \operatorname { a r g m a x } _ { i \in [ K ] } x _ { t , i } .$ The KKT conditions for z<sub>t</sub> give

$$
\begin{array} { r l r } {  { - \widehat { \ell } _ { t , i } - \frac { \nabla _ { i } \Psi ( z _ { t } ) - \nabla _ { i } \Psi ( x _ { t } ) } { \eta _ { t } } + \lambda = 0 , } } \\ & { } & { \frac { 1 } { \eta _ { t } \sqrt { x _ { t , i } } } - \frac { 1 } { \eta _ { t } \sqrt { z _ { t , i } } } = \lambda - \widehat { \ell } _ { t , I _ { t } } \mathbb { I } \{ i = I _ { t } \} , } \end{array}
$$

for every $i \in [ K ]$ , where λ is the simplex multiplier and the second equality uses $\nabla _ { i } \Psi ( x ) = - 1 / \sqrt { x _ { i } }$ . The simplex constraint gives $0 \leq \lambda \leq \widehat { \ell } _ { t , I _ { t } } ;$ outside this interval, every coordinate would strictly decrease or strictly increase. Consequently,

$$
0 \leq \eta _ { t } \lambda \sqrt { x _ { t , i } } \leq \frac { \eta _ { t } \sqrt { x _ { t , i } } } { q _ { t , I _ { t } } + 2 \eta _ { t } \sqrt { x _ { t , I _ { t } } } } \leq \frac { \sqrt { x _ { t , i } } } { 2 \sqrt { x _ { t , I _ { t } } } } \leq \frac { 1 } { 2 } ,
$$

for $i \neq I _ { t } .$ , where the first upper bound uses $\lambda \leq \widehat { \ell } _ { t , I _ { t } } , \ell _ { t , I _ { t } } \leq 1$ , and $\rho _ { t , I _ { t } } = 2 \eta _ { t } \sqrt { x _ { t , I _ { t } } }$ , the second uses $q _ { t , I _ { t } } \geq 0$ , and the last uses maximality of $x _ { t , I _ { t } }$ . Solving the KKT equation for these coordinates therefore gives

$$
z _ { t , i } - x _ { t , i } = \frac { x _ { t , i } } { ( 1 - \eta _ { t } \lambda \sqrt { x _ { t , i } } ) ^ { 2 } } - x _ { t , i } = x _ { t , i } \frac { 2 \eta _ { t } \lambda \sqrt { x _ { t , i } } - ( \eta _ { t } \lambda \sqrt { x _ { t , i } } ) ^ { 2 } } { ( 1 - \eta _ { t } \lambda \sqrt { x _ { t , i } } ) ^ { 2 } } \leq 8 \eta _ { t } \lambda x _ { t , i } ^ { 3 / 2 } ,
$$

where the inequality uses $0 \leq \eta _ { t } \lambda \sqrt { x _ { t , i } } \leq 1 / 2$ . Substituting into Eq. (28) and using $\lambda \le \widehat { \ell } _ { t , I _ { t } }$ gives

$$
\operatorname* { m a x } _ { z \in \Delta _ { K } } \Big \{ \Big < x _ { t } - z , \widehat { \ell } _ { t } \Big > - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \Big \} \leq 8 \eta _ { t } \lambda \widehat { \ell } _ { t , I _ { t } } \sum _ { \underset { i \neq I _ { t } } { i = 1 } } ^ { K } x _ { t , i } ^ { 3 / 2 } \leq 8 \eta _ { t } \widehat { \ell } _ { t , I _ { t } } ^ { 2 } \sum _ { \underset { i \neq I _ { t } } { i = 1 } } ^ { K } x _ { t , i } ^ { 3 / 2 } .
$$

Combining the two bounds yields

$$
\operatorname* { m a x } _ { z \in \Delta _ { K } } \left\{ \left. x _ { t } - z , \widehat { \ell } _ { t } \right. - D _ { \eta _ { t } ^ { - 1 } \Psi } ( z , x _ { t } ) \right\} \leq \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , I _ { t } } ^ { 3 / 2 } , 8 \sum _ { \stackrel { i = 1 } { i \neq I _ { t } } } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \widehat { \ell } _ { t , I _ { t } } ^ { 2 } .\tag{29}
$$

Substituting Eq. (29) into the FTRL decomposition proves the lemma.

## B Omitted Details in Section 4

In this section, we prove Lemma B.1, which compares consecutive iterates after the loss estimate and learning rate are updated. We then provide the complete proof for Lemma 4.2, which gives the detailed calculation of the conditional-expectation used in Section 4.1.

## B.1 Multiplicative Stability

The following lemma proves the multiplicative stability of the consecutive iterates after both the loss estimate and the learning rate are updated.

Lemma B.1. Algorithm 1 satisfies that $x _ { t + 1 , i } \leq 6 x _ { t , i } , f o r e \nu e r y t \in [ T ]$ and $i \in [ K ]$

Proof. Fix a round t. Denote by $\lambda _ { t }$ and $\lambda _ { t + 1 }$ the simplex multipliers for the two consecutive FTRL updates. Their KKT equations give

$$
\begin{array} { r l } & { \widehat { L } _ { t - 1 , i } + \lambda _ { t } = \frac { 1 } { \eta _ { t } \sqrt { x _ { t , i } } } , } \\ & { \widehat { L } _ { t , i } + \lambda _ { t + 1 } = \frac { 1 } { \eta _ { t + 1 } \sqrt { x _ { t + 1 , i } } } , } \end{array}
$$

for every $i \in [ K ]$ . Subtracting the first equation from the second and using $\widehat { L } _ { t } = \widehat { L } _ { t - 1 } + \widehat { \ell } _ { t }$ , we obtain that

$$
\frac { 1 } { \eta _ { t + 1 } \sqrt { x _ { t + 1 , i } } } = \frac { 1 } { \eta _ { t } \sqrt { x _ { t , i } } } + \widehat { \ell } _ { t , i } + \lambda _ { t + 1 } - \lambda _ { t } = \frac { 1 + \eta _ { t } \sqrt { x _ { t , i } } ( \widehat { \ell } _ { t , i } + \lambda _ { t + 1 } - \lambda _ { t } ) } { \eta _ { t } \sqrt { x _ { t , i } } } ,
$$

where the numerator $1 + \eta _ { t } \sqrt { x _ { t , i } } ( \widehat { \ell } _ { t , i } + \lambda _ { t + 1 } - \lambda _ { t } )$ is positive as the left-hand side is positive. Taking reciprocals and squaring therefore gives

$$
x _ { t + 1 , i } = \frac { \left( \frac { \eta _ { t } } { \eta _ { t + 1 } } \right) ^ { 2 } x _ { t , i } } { \left( 1 + \eta _ { t } \sqrt { x _ { t , i } } ( \widehat { \ell } _ { t , i } + \lambda _ { t + 1 } - \lambda _ { t } ) \right) ^ { 2 } } .
$$

The learning-rate update implies $( \eta _ { t } / \eta _ { t + 1 } ) ^ { 2 } = 1 + \eta _ { t } ^ { 2 } \ell _ { t , I _ { t } } \leq 3 / 2$ , since $\ell _ { t , I _ { t } } \leq 1 , \eta _ { t } ^ { 2 } \leq 1 / K$ , and $K \geq 2$ . If $\lambda _ { t + 1 } \geq \lambda _ { t } .$ , the denominator in the coordinate formula is at least one because $\widehat { \ell } _ { t , i } \geq 0$ , giving $x _ { t + 1 , i } \leq 3 x _ { t , i } / 2$ for every i.

It remains to consider $\lambda _ { t + 1 } < \lambda _ { t }$ . For every $i \neq I _ { t }$ , we have $\widehat { \ell } _ { t , i } = 0$ , so the coordinate formula and $\eta _ { t + 1 } \leq \eta _ { t }$ imply $x _ { t + 1 , i } > x _ { t , i }$ . Since both iterates sum to one, this also gives $x _ { t + 1 , I _ { t } } \leq x _ { t , I _ { t } }$ . For $i \neq I _ { t }$ with $x _ { t , i } \geq x _ { t , I _ { t } }$ , we know that

$$
x _ { t + 1 , i } - x _ { t , i } \leq \sum _ { j = 1 \atop j \neq I _ { t } } ^ { K } ( x _ { t + 1 , j } - x _ { t , j } ) = x _ { t , I _ { t } } - x _ { t + 1 , I _ { t } } \leq x _ { t , I _ { t } } \leq x _ { t , i } ,
$$

where the first inequality uses the nonnegativity of every summand, and the equality uses the simplex constraints. Thus, $x _ { t + 1 , i } \leq 2 x _ { t , i }$ for these coordinates.

To bound the remaining coordinates, we first show that $\lambda _ { t + 1 } \geq \lambda _ { t } - \widehat { \ell } _ { t , I _ { t } }$ . Otherwise, $\widehat { \ell } _ { t , i } + \lambda _ { t + 1 } - \lambda _ { t } < 0$ for every $i \in [ K ]$ , so the coordinate formula and $\eta _ { t + 1 } ~ \leq ~ \eta _ { t }$ would imply $x _ { t + 1 , i } ~ > ~ x _ { t , i }$ for every i, contradicting the simplex constraints. Consequently, for $i \neq I _ { t }$ with $x _ { t , i } < x _ { t , I _ { t } }$ , we have

$$
\eta _ { t } \sqrt { x _ { t , i } } ( \lambda _ { t } - \lambda _ { t + 1 } ) \leq \eta _ { t } \sqrt { x _ { t , i } } \widehat { \ell } _ { t , I _ { t } } \leq \frac { \eta _ { t } \sqrt { x _ { t , i } } } { \rho _ { t , I _ { t } } } = \frac { \sqrt { x _ { t , i } } } { 2 \sqrt { x _ { t , I _ { t } } } } < \frac { 1 } { 2 } ,
$$

where the second inequality uses $\widehat { \ell } _ { t , I _ { t } } = \ell _ { t , I _ { t } } / ( q _ { t , I _ { t } } + \rho _ { t , I _ { t } } ) \leq 1 / \rho _ { t , I _ { t } }$ , and the equality uses $\rho _ { t , I _ { t } } = 2 \eta _ { t } \sqrt { x _ { t , I _ { t } } }$ Substituting this bound into the coordinate formula gives

$$
x _ { t + 1 , i } = \frac { \left( \frac { \eta _ { t } } { \eta _ { t + 1 } } \right) ^ { 2 } x _ { t , i } } { \left( 1 - \eta _ { t } \sqrt { x _ { t , i } } ( \lambda _ { t } - \lambda _ { t + 1 } ) \right) ^ { 2 } } \leq 4 \left( \frac { \eta _ { t } } { \eta _ { t + 1 } } \right) ^ { 2 } x _ { t , i } \leq 6 x _ { t , i } .
$$

Combining the bounds for all coordinates proves the lemma.

## B.2 Proof of Lemma 4.2

Proof. We first apply Lemma A.1 with $u = e _ { i ^ { \star } }$ , which gives Eq. (23):

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \left. x _ { t } - e _ { i ^ { \star } } , \widehat { \ell } _ { t } \right. \leq \frac { \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \displaystyle \sum _ { t = 1 } ^ { T } \left( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } \right) \left( \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { t + 1 } ) \right) } \\ & { \quad \quad \quad + \displaystyle \sum _ { t = 1 } ^ { T } \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , I _ { t } } ^ { 3 / 2 } , 8 \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \widehat { \ell } _ { t , I _ { t } } ^ { 2 } . } \end{array}
$$

We then prove the conditional bound in Eq. (24) for the last term in Eq. (23), then derive the difference between true and estimated regret in Eq. (25), and finally combine these bounds to obtain Eq. (16).

All conditional expectations below are conditioned on $\mathcal { F } _ { t - 1 }$ . Direct calculation shows that

$$
\begin{array} { r l } & { \mathbb { E } [ \widehat { \ell } _ { t , i } \mid \mathcal { F } _ { t - 1 } ] = \frac { q _ { t , i } \mu _ { i } } { q _ { t , i } + \rho _ { t , i } } , } \\ & { } \\ & { \mathbb { E } [ \widehat { \ell } _ { t , i } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] = \frac { q _ { t , i } \mathbb { E } [ \ell _ { t , i } ^ { 2 } ] } { ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 } } \leq \frac { q _ { t , i } \mu _ { i } } { ( q _ { t , i } + \rho _ { t , i } ) ^ { 2 } } , } \end{array}
$$

where the equalities use independence of the current loss vector from the history and the fresh action. The sampling rule in Line 5 also gives

$$
\frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \leq Z _ { t } \frac { q _ { t , i } + 2 \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \leq 2 Z _ { t } ,
$$

where the first inequality follows from $q _ { t , i } = [ x _ { t , i } / Z _ { t } - 2 \rho _ { t , i } ] _ { + } \geq x _ { t , i } / Z _ { t } - 2 \rho _ { t , }$ . Hence, for every suboptimal arm $i \neq i ^ { \star }$ , we have

$$
\begin{array} { r l } & { \eta _ { t } x _ { t , i } ^ { 3 / 2 } \mathbb { E } [ \widehat { \ell } _ { t , i } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] \leq \frac { \eta _ { t } x _ { t , i } ^ { 3 / 2 } q _ { t , i } \mu _ { i } } { \left( q _ { t , i } + \rho _ { t , i } \right) ^ { 2 } } } \\ & { \qquad = \frac { \rho _ { t , i } } { 2 } \frac { x _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \frac { q _ { t , i } \mu _ { i } } { q _ { t , i } + \rho _ { t , i } } } \\ & { \qquad \leq Z _ { t } \frac { q _ { t , i } \rho _ { t , i } \mu _ { i } } { q _ { t , i } + \rho _ { t , i } } } \\ & { \qquad = Z _ { t } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } + Z _ { t } \frac { q _ { t , i } \rho _ { t , i } \mu _ { i } } { q _ { t , i } + \rho _ { t , i } } } \\ & { \qquad \leq Z _ { t } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } + 2 \eta _ { t } \mu _ { i } \setminus \sqrt { x _ { t , i } } , } \end{array}
$$

where the first equality uses $\rho _ { t , i } = 2 \eta _ { t } \sqrt { x _ { t , i } }$ , the second inequality uses $x _ { t , i } / ( q _ { t , i } + \rho _ { t , i } ) \leq 2 Z _ { t }$ , and the last inequality uses $Z _ { t } \le 1$ and $q _ { t , i } / ( q _ { t , i } + \rho _ { t , i } ) \leq 1$

For the best arm, if $i ^ { \star } \neq \operatorname { a r g m a x } _ { i \in [ K ] } x _ { t , i }$ , the same calculation gives

$$
\eta _ { t } x _ { t , i ^ { \star } } ^ { 3 / 2 } \mathbb { E } [ \widehat \ell _ { t , i ^ { \star } } ^ { 2 } \ | \ \mathcal { F } _ { t - 1 } ] \leq \mu _ { i ^ { \star } } \rho _ { t , i ^ { \star } } = 2 \eta _ { t } \mu _ { i ^ { \star } } \sqrt { x _ { t , i ^ { \star } } } \leq 2 \eta _ { t } \mu _ { i ^ { \star } } \sum _ { i = 1 \atop i \neq i ^ { \star } } ^ { K } \sqrt { x _ { t , i } } ,
$$

where the last inequality follows because another coordinate is at least $x _ { t , i ^ { \star } }$ . Otherwise, we have

$$
\frac { q _ { t , i } } { x _ { t , i } } = \left[ \frac { 1 } { Z _ { t } } - \frac { 4 \eta _ { t } } { \sqrt { x _ { t , i } } } \right] _ { + } \leq \left[ \frac { 1 } { Z _ { t } } - \frac { 4 \eta _ { t } } { \sqrt { x _ { t , i ^ { \star } } } } \right] _ { + } = \frac { q _ { t , i ^ { \star } } } { x _ { t , i ^ { \star } } } ,
$$

for every $i \in [ K ]$ , where the equalities follow from Line 5 and the inequality uses $x _ { t , i } \leq x _ { t , i ^ { \star } }$ . Multiplying by $x _ { t , i }$ and summing gives

$$
1 = \sum _ { i = 1 } ^ { K } q _ { t , i } = \sum _ { i = 1 } ^ { K } x _ { t , i } { \frac { q _ { t , i } } { x _ { t , i } } } \leq \sum _ { i = 1 } ^ { K } x _ { t , i } { \frac { q _ { t , i ^ { \star } } } { x _ { t , i ^ { \star } } } } = { \frac { q _ { t , i ^ { \star } } } { x _ { t , i ^ { \star } } } } .
$$

Thus $q _ { t , i ^ { \star } } \geq x _ { t , i ^ { \star } }$ , and we have

$$
\begin{array} { r l } {  { 8 \eta _ { t } \sum _ { i = 1 } ^ { K } x _ { t , i } ^ { 3 / 2 } \mathbb { E } [ \widehat \ell _ { t , i ^ { \star } } ^ { 2 } \ | \ \mathcal { F } _ { t - 1 } ] \leq \frac { 8 \eta _ { t } q _ { t , i ^ { \star } } \mu _ { i ^ { \star } } } { ( q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } ) ^ { 2 } } \sum _ { i \neq i } ^ { K } x _ { t , i } ^ { 3 / 2 } } } \\ & { \leq 8 \eta _ { t } \mu _ { i ^ { \star } } \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } } { q _ { t , i ^ { \star } } } \sqrt { x _ { t , i } } \leq 8 \eta _ { t } \mu _ { i ^ { \star } } \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } , } \end{array}
$$

where the last inequality uses $x _ { t , i } \leq x _ { t , i ^ { \star } } \leq q _ { t , i ^ { \star } }$ . Combining the suboptimal-arm bounds with the two cases for arm $i ^ { \star }$ yields

$$
\mathbb { E } \left[ \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , I _ { t } } ^ { 3 / 2 } , 8 \sum _ { i = 1 , i \neq I _ { t } } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \widehat { \ell } _ { t , I _ { t } } ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right]
$$

$$
\begin{array} { r l } & { \le \displaystyle \sum _ { i \ne i } ^ { K } \eta _ { t } x _ { t , i } ^ { 3 / 2 } \mathbb { E } [ \widehat \ell _ { t , i } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] + \eta _ { t } \operatorname* { m i n } \left\{ x _ { t , i ^ { \star } } ^ { 3 / 2 } , 8 \displaystyle \sum _ { i \ne i } ^ { K } x _ { t , i } ^ { 3 / 2 } \right\} \mathbb { E } [ \widehat \ell _ { t , i ^ { \star } } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] } \\ & { \le Z _ { t } \displaystyle \sum _ { i = 1 } ^ { K } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } + 1 0 \eta _ { t } \mu _ { i ^ { \star } } \displaystyle \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } , } \end{array}\tag{30}
$$

where the first inequality uses $\widehat { \ell } _ { t , i } = 0$ for $i \neq I _ { t }$ and the $\mathcal { F } _ { t } .$ <sub>−1</sub>-measurability of the coefficients, and the second uses the preceding coordinate bounds. This proves Eq. (24).

We next compute the difference between true and estimated regret. Expanding the conditional means gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle q _ { t } - e _ { i ^ { \star } } , \ell _ { t } \rangle - \left. x _ { t } - e _ { i ^ { \star } } , \widehat { \ell } _ { t } \right. | \mathcal { F } _ { t - 1 } \right] } \\ & { = \langle q _ { t } , \mu \rangle - \mu _ { i ^ { \star } } - \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } q _ { t , i } \mu _ { i } } { q _ { t , i } + \rho _ { t , i } } + \frac { q _ { t , i ^ { \star } } \mu _ { i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } } \\ & { = \displaystyle \sum _ { i = i ^ { \star } } ^ { K } \Delta _ { i } \left( q _ { t , i } - \frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \right) + \mu _ { i ^ { \star } } \left( \frac { q _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } - \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \right) , } \end{array}\tag{31}
$$

where the last equality substitutes $\mu _ { i } = \mu _ { i ^ { \star } } + \Delta _ { i }$ for $i \neq i ^ { \star }$ and uses $\textstyle \sum _ { i = 1 } ^ { K } q _ { t , i } = 1$ . For the first summation, using Eq. (12), we can obtain that

$$
\frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + \rho _ { t , i } } = \frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + 2 \rho _ { t , i } } \frac { q _ { t , i } + 2 \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } = Z _ { t } q _ { t , i } + Z _ { t } \frac { q _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } .
$$

Substituting this identity into the first term in Eq. (31) yields

$$
\sum _ { i = 1 \atop i \ne i ^ { * } } ^ { K } \Delta _ { i } \left( q _ { t , i } - \frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + \rho _ { t , i } } \right) = ( 1 - Z _ { t } ) \sum _ { i = 1 \atop i \ne i ^ { * } } ^ { K } \Delta _ { i } q _ { t , i } - Z _ { t } \sum _ { i = 1 \atop i \ne i ^ { * } } ^ { K } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } .
$$

For the second term in Eq. (31), direct calculation shows that

$$
\begin{array} { r l } { \frac { q _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } - \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } q _ { t , i } } { q _ { t , i } + \rho _ { t , i } } = 1 - \frac { \rho _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } - \displaystyle \sum _ { i = 1 } ^ { K } x _ { t , i } + \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } } & { { } } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { \left( 1 - x _ { t , i ^ { \star } } \right) \rho _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } , } & { { } } \end{array}
$$

where the first equality uses $q _ { t , i } / ( q _ { t , i } + \rho _ { t , i } ) = 1 - \rho _ { t , i } / ( q _ { t , i } + \rho _ { t , i } )$ and the second uses $\textstyle \sum _ { i = 1 } ^ { K } x _ { t , i } = 1$ and separates the term indexed by $i ^ { \star } .$ . Combining the two calculations gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle q _ { t } - e _ { i ^ { \star } } , \ell _ { t } \rangle - \left. x _ { t } - e _ { i ^ { \star } } , \widehat { \ell } _ { t } \right. | \mathcal { F } _ { t - 1 } \right] } \\ & { = ( 1 - Z _ { t } ) \displaystyle \sum _ { i = 1 } ^ { K } \Delta _ { i } q _ { t , i } - Z _ { t } \displaystyle \sum _ { i = 1 } ^ { K } \frac { q _ { t , i } \rho _ { t , i } \Delta _ { i } } { q _ { t , i } + \rho _ { t , i } } + \mu _ { i ^ { \star } } \left( \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { ( 1 - x _ { t , i ^ { \star } } ) \rho _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } \right) , } \end{array}\tag{32}
$$

which proves Eq. (25).

We now substitute these bounds into the regret decomposition. Applying Eq. (23), Eq. (30), and Eq. (32) gives

$$
\begin{array} { r l } { \log _ { 2 } \nu _ { \infty } } & { \displaystyle - \sum _ { k = 1 } ^ { 2 } \mathbb { E } [ ( \mathcal { Z } _ { \mathcal { U } _ { k } } - \sigma _ { \mathcal { U } _ { k } } \hat { \mathcal { R } } _ { k } ^ { k } ) ] + \sum _ { t = 2 } ^ { 2 \nu _ { \infty } } \mathbb { E } [ ( \frac { \eta _ { t } } { \hat { \sigma } _ { t } } - \sigma _ { \mathcal { U } _ { k } } , \hat { \rho } _ { k } ) - ( \mathcal { Z } _ { \mathcal { U } _ { k } } - \sigma _ { \mathcal { U } _ { k } } , \hat { \rho } _ { k } ^ { k } ) ] } \\ & { \leq \frac { \Phi ( \varepsilon _ { \mathcal { U } _ { k } } ^ { k } + 1 ) - \Phi ( \tau _ { \mathcal { U } _ { k } } ) } { \eta _ { t } } + \mathbb { E } [ [ \frac { \sum _ { t = 1 } ^ { n } } { k - 1 } ( \frac { 1 } { \eta _ { t } } - 1 ) ( \Phi ( \sigma _ { s - 1 } ) - \Phi ( \sigma _ { \mathcal { U } _ { k } - 1 } ) ) ] } \\ & { \quad + \displaystyle \sum _ { k = 1 } ^ { 2 \nu _ { \infty } } [ 2 \int _ { s _ { 0 } } \mathbb { E } _ { \eta _ { k } } \frac { \Phi ( \varepsilon _ { \mathcal { U } _ { k } } , \hat { \rho } _ { k } , \hat { \sigma } _ { t } , \Delta s _ { t } ) } { \eta _ { t } + \theta _ { \mathcal { U } _ { k } } } \quad \log _ { \mathcal { U } _ { k } } \frac { R } { \sqrt { \eta _ { t } } } ] } \\ &  \quad + \displaystyle \sum _ { k = 1 } ^ { 2 \nu _ { \infty } } [ ( 1 - Z _ { \mathcal { U } _ { k } } ) \sum _ { k = 1 } ^ { K } \Delta _ { \mathcal { U } _ { k } , k } - Z _ { \mathcal { U } _ { k } } \frac  R \end{array}
$$

The two sums involving $Z _ { t } q _ { t , i } \rho _ { t , i } { \Delta } _ { i } / ( q _ { t , i } + \rho _ { t , i } )$ cancel. For the last expectation, we have

$$
\begin{array} { r l r } {  { \mu _ { i ^ { \star } } ( \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } - \frac { ( 1 - x _ { t , i ^ { \star } } ) \rho _ { t , i ^ { \star } } } { q _ { t , i ^ { \star } } + \rho _ { t , i ^ { \star } } } ) \leq \mu _ { i ^ { \star } } \displaystyle \sum _ { i = 1 } ^ { K } \frac { x _ { t , i } \rho _ { t , i } } { q _ { t , i } + \rho _ { t , i } } } } \\ & { } & { \leq 2 Z _ { t } \mu _ { i ^ { \star } } \displaystyle \sum _ { i = 1 } ^ { K } \rho _ { t , i } \leq 4 \eta _ { t } \mu _ { i ^ { \star } } \displaystyle \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } , } \end{array}
$$

where the first inequality drops the nonpositive term, the second uses $x _ { t , i } / ( q _ { t , i } + \rho _ { t , i } ) \leq 2 Z _ { t }$ , and the last uses $Z _ { t } \leq 1$ and $\rho _ { t , i } = 2 \eta _ { t } \sqrt { x _ { t , i } }$ . Substituting this bound into the preceding regret inequality gives

$$
\begin{array} { r l r } {  { \mathsf { R e g } _ { T } \leq \frac { \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { 1 } ) } { \eta _ { 1 } } + \mathbb { E } [ \sum _ { t = 1 } ^ { T } ( \frac { 1 } { \eta _ { t + 1 } } - \frac { 1 } { \eta _ { t } } ) ( \Psi ( e _ { i ^ { \star } } ) - \Psi ( x _ { t + 1 } ) ) ] } } \\ & { } & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } [ ( 1 - Z _ { t } ) \sum _ { i = 1 } ^ { K } \Delta _ { i } q _ { t , i } ] + 1 4 \sum _ { t = 1 } ^ { T } \mathbb { E } [ \eta _ { t } \mu _ { i ^ { \star } } \sum _ { i = 1 } ^ { K } \sqrt { x _ { t , i } } ] , } \\ & { } & { \qquad + \displaystyle \frac { 1 } { i \neq i ^ { \star } } } \end{array}
$$

which finishes the proof.