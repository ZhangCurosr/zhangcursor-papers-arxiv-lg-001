# CoDrift: Compositional Drifting for Offline Reinforcement Learning

Xiewei Ni, Ruofeng Mei, Xiangyu Xu<sup>∗</sup>

Xi’an Jiaotong University

<sup>∗</sup>Corresponding author

Ofline reinforcement learning is intrinsically multi-objective: a policy must remain compatible with the behavioral support of a fixed dataset while preferentially selecting high-value actions. We recast these objectives in a common form by viewing each as an action-space motion field that specifies how generated actions should move. This perspective enables heterogeneous learning objectives to be combined directly through field composition. Inspired by drifting models, we propose CoDrift, a compositional framework for one-step generative policy learning. CoDrift combines three objectivelevel fields into a unified policy field. The conditional field preserves state-dependent behavioral structure, while the marginal field pools actions across states to provide a more stable generative signal in the single-positive-sample regime of continuous-control ofline RL. The value field moves generated actions toward higher-value regions. The composed field is absorbed into a stochastic generator that produces an action with a single forward pass at deployment. We evaluate CoDrift on 73 tasks from OGBench and D4RL in both ofline and ofline-to-online settings. CoDrift compares favorably with state-of-the-art methods and achieves the best average rank in both settings.

## 1 Introduction

Ofline reinforcement learning (RL) aims to learn efective policies entirely from previously collected experience, without further interaction with the environment (Levine et al., 2020). A central challenge is that policy optimization must satisfy multiple objectives simultaneously. On the one hand, the learned policy should remain compatible with the behavioral support of the ofline dataset, since actions far outside the data distribution may lead to unreliable value estimates and poor decisions. On the other hand, merely reproducing the behavior policy is insuficient: the policy must preferentially select high-value actions in order to improve return. Ofline RL can therefore be viewed as an intrinsically multi-objective policy-learning problem, balancing behavioral fidelity with value maximization.

Generative policies provide a powerful way to model the behavioral side of this problem. In particular, difusion-based policies have demonstrated good capability to represent complex and multimodal action distributions (Wang et al., 2023; Hansen-Estruch et al., 2023), while flow-matching approaches provide an even stronger generative policy class (Park et al., 2025b). These successes suggest that expressive generative modeling is a natural foundation for ofline policy learning. We build on this perspective, but take a diferent view of how the multiple objectives of ofline RL should be incorporated into a generative policy.

Our starting point is simple: diferent learning objectives can all be interpreted as specifying how a generated action should move in action space. A behavioral objective moves generated actions toward regions supported by the ofline data, whereas a value objective moves them toward directions of higher predicted return. Once expressed as action-space displacements, these heterogeneous objectives share a common representation and can thereby be composed naturally. This motivates the central principle of our approach: ofline policy learning can be formulated as the composition of action-space motion fields, with each field corresponding to a distinct learning objective.

Our formulation is inspired by drifting models (Deng et al., 2026), a recent class of generative models that learns stochastic generators from displacement fields acting directly on generated samples. In the original drifting formulation, a displacement field is constructed from two elementary components: attraction toward positive data samples and repulsion from negative generated samples. For our setting, it suggests a broader principle: while the original attraction and repulsion components jointly realize a single distribution-matching objective, diferent learning objectives can themselves be represented as displacement fields and composed directly in action space.

![](images/8c7f455a10e3c8156c5d1f7eeb29ca7663a097b188dce1409636e10433540ad1.jpg)  
Figure 1 Overview of CoDrift. Conditional and marginal behavioral fields keep generated actions within the support of the ofline data at two statistical scales, while the value field moves them toward higher-return regions. The three are composed additively into a single policy field $V _ { \mathrm { C o D r i f t } }$

Based on this principle, we propose CoDrift, a compositional drifting framework for ofline RL. CoDrift instantiates three complementary action-space fields as illustrated in Figure 1. First, a conditional behavioral field preserves state-dependent behavioral support by attracting candidate actions toward actions observed at the corresponding states while maintaining stochasticity through repulsion among generated actions. Second, a marginal behavioral field captures global structure in the action distribution by pooling data and generated actions across states within a minibatch. This marginal field is particularly useful in continuous-control ofline RL, where essentially every state is paired with only a single observed action. Consequently, the conditional field is estimated from a single Monte Carlo sample at each state, whereas marginal field can exploit multiple actions across the batch, yielding a more stable, lower-variance generative signal. Finally, a value field pushes generated actions toward higher-value regions using action gradients from learned critics. These fields pla complementary roles and are combined additively into a single generative policy field.

Our contributions are summarized as follows:

• We introduce CoDrift, a compositional drifting framework for generative policy learning in ofline RL, where heterogeneous learning objectives are represented as additive action-space fields. CoDrift achieves expressive stochastic generation while requiring only a single forward pass at deployment.

• We identify the single-positive-sample problem in conditional drifting for ofline RL. We address this issue with marginal drifting, which pools actions across states to provide a more stable training signal.

• We conduct extensive experiments across 73 tasks from OGBench and D4RL in both ofline and oflineto-online settings. CoDrift compares favorably with state-of-the-art approaches, achieving the best average rank in both settings.

## 2 Preliminaries

## 2.1 Offline RL

We consider a Markov decision process (MDP) $\mathcal { M } = ( S , \mathcal { A } , P , r , \rho , \gamma )$ (Levine et al., 2020), where S is the state space, and $\mathcal { A } = [ - 1 , 1 ] ^ { d }$ is the d-dimensional continuous action space. $P ( \cdot \mid s , a ) : S \times \mathcal { A }  \Delta ( S )$ is the transition dynamics, where $\Delta ( \mathcal { X } )$ denotes the set of probability distributions over a space X. The reward function is $r : \mathcal { S } \times \mathcal { A } \to \mathbb { R } , \rho \in \Delta ( \mathcal { S } )$ is the initial state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. A policy $\pi _ { \theta } ( \cdot \mid s ) : { \mathcal { S } } \to \Delta ( { \mathcal { A } } )$ induces a trajectory distribution $p ^ { \pi _ { \theta } } ( \zeta )$ under the dynamics P and the initial state distribution $\rho ,$ and is evaluated by its expected discounted return

$$
J ( \pi _ { \theta } ) = \mathbb { E } _ { \zeta \sim p ^ { \pi _ { \theta } } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] ,\tag{1}
$$

with associated action-value function

$$
Q ^ { \pi _ { \theta } } ( s , a ) = \mathbb { E } _ { \zeta \sim p ^ { \pi _ { \theta } } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \Bigg | s _ { 0 } = s , a _ { 0 } = a \right] .
$$

Ofline RL seeks to maximize Eq. 1 using only a fixed dataset $\mathcal { D } = \{ ( s , a , r , s ^ { \prime } ) \}$ of transitions collected by an unknown behavior policy $\pi _ { \beta }$ , without further interaction with the environment. In this work, we also consider the ofline-to-online setting, where the ofline pre-trained policy is further fine-tuned with a modest amount of online environment interactions. A fundamental dificulty in ofline RL is distribution shift: because D covers only state–action regions visited by $\pi _ { \beta }$ , a critic fit on $\mathcal { D }$ cannot be expected to extrapolate reliably outside the support of the behavior distribution, and unconstrained value maximization against such a critic drives the actor toward out-of-distribution actions (Levine et al., 2020).

Behavior-regularized actor–critic methods. Wu et al. (2019); Fujimoto $\&$ Gu (2021); Tarasov et al. (2023) address this issue by balancing value maximization with an explicit constraint that keeps the learned policy close to the ofline behavior. A critic $Q _ { \varphi }$ is typically trained by Bellman regression,

$$
\begin{array} { r } { \mathcal { L } _ { Q } ( \varphi ) = \mathbb { E } _ { ( s , a , r , s ^ { \prime } ) \sim \mathcal { D } , } \left[ \left( Q _ { \varphi } ( s , a ) - r - \gamma Q _ { \bar { \varphi } } ( s ^ { \prime } , a ^ { \prime } ) \right) ^ { 2 } \right] , } \\ { a ^ { \prime } { \sim } \pi _ { \theta } ( \cdot | s ^ { \prime } ) \ } \end{array}
$$

where $Q _ { \bar { \varphi } }$ denotes a target network (Mnih et al., 2015). The actor is then commonly optimized using an objective of the form

$$
\begin{array} { r } { \mathcal { L } _ { \pi } ( \theta ) = \mathbb { E } _ { \ s \sim \mathcal { D } , \ } \left[ - Q _ { \varphi } ( s , a ) + \alpha \mathcal { R } ( \pi _ { \theta } ) \right] , } \\ { a \sim \pi _ { \theta } ( \cdot | s ) } \end{array}
$$

where $\mathcal { R } ( \pi _ { \theta } )$ regularizes the policy toward the ofline behavior distribution and α controls the trade-of between behavioral fidelity and value maximization.

CoDrift follows the same fundamental principle of balancing behavioral fidelity and value maximization, but realizes it in a diferent form. Rather than combining a critic objective with an explicit behavioral penalty at the loss level, CoDrift represents the learning objectives as action-space displacement fields and composes them into a joint policy field, as introduced in the following sections.

## 2.2 Drifting Models

Drifting models (Deng et al., 2026) are a recent approach to one-step generative modeling. Let $p : = p _ { \mathrm { d a t a } }$ be the target distribution on $\mathbb { R } ^ { d }$ and $p _ { \varepsilon } : = \mathcal { N } ( 0 , I )$ a prior on $\mathbb { R } ^ { m }$ . A one-step generator $f _ { \theta } : \mathbb { R } ^ { m }  \mathbb { R } ^ { d }$ maps noise $\varepsilon \sim p _ { \varepsilon }$ to a sample $x = f _ { \theta } ( \varepsilon )$ , inducing the pushforward distribution $q _ { \theta } : = [ f _ { \theta } ] _ { \# } p _ { \varepsilon }$ , and the goal is to make $q _ { \theta }$ match $p .$ Unlike difusion and flow models, which transport samples through a sequence of intermediate states at inference time, drifting models shift this iterative transport process to training. The generator itself remains a single forward mapping, while its induced distribution $q _ { \theta }$ is progressively moved toward p over the course of optimization.

The transport direction is given by a distribution-dependent drifting field $V _ { p , q } ( x ) = V _ { p } ^ { + } ( x ) - V _ { q } ^ { - } ( x )$ , whose two components are kernel-normalized mean-shift vectors:

$$
\begin{array} { r } { V _ { p } ^ { + } ( x ) = \frac { \mathbb { E } _ { u ^ { + } \sim p } \left[ \kappa _ { \tau } ( x , u ^ { + } ) ( u ^ { + } - x ) \right] } { \mathbb { E } _ { u ^ { + } \sim p } \left[ \kappa _ { \tau } ( x , u ^ { + } ) \right] } , } \\ { V _ { q } ^ { - } ( x ) = \frac { \mathbb { E } _ { u ^ { - } \sim q } \left[ \kappa _ { \tau } ( x , u ^ { - } ) ( u ^ { - } - x ) \right] } { \mathbb { E } _ { u ^ { - } \sim q } \left[ \kappa _ { \tau } ( x , u ^ { - } ) \right] } , } \end{array}\tag{2}
$$

where $\kappa _ { \tau } ( \cdot , \cdot )$ is a positive similarity kernel. Throughout this work, we adopt the Laplace kernel $\kappa _ { \tau } ( x , u ) =$ $\exp ( - \| x - u \| _ { 2 } / \tau )$ , which assigns larger weights to nearby samples, with τ controlling the efective neighborhood size. Because the kernel weights are normalized, each term in $\operatorname { E q . 2 }$ represents a displacement vector from x to the weighted centroid of a reference set. Accordingly, $V _ { p } ^ { + }$ acts as an attractive component, pulling generated samples toward the target distribution, whereas subtracting $V _ { q } ^ { - }$ produces a repulsive component that pushes generated samples away from one another and discourages mode collapse.

An important property of the drifting field is its antisymmetry: $V _ { p , q } ( x ) = - V _ { q , p } ( x )$ . In particular, when $q = p ,$ , the attractive and repulsive components cancel and $V _ { p , q } ( x ) = 0$ , so matching the target distribution corresponds to an equilibrium of the drifting dynamics. Training moves the generator toward this equilibrium by constructing a displaced target for each generated sample. Given $x = f _ { \theta } ( \varepsilon )$ , the target is $\tilde { x } = x + V _ { p , q } ( x )$ which is treated as fixed when updating the generator. The resulting regression objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d r i f t } } ( \theta ; p , q ) = \mathbb { E } _ { \varepsilon \sim p _ { \varepsilon } } \Big [ \big \| f _ { \theta } ( \varepsilon ) - \mathrm { s g } \left( f _ { \theta } ( \varepsilon ) + V _ { p , q } ( f _ { \theta } ( \varepsilon ) ) \right) \big \| _ { 2 } ^ { 2 } \Big ] , } \end{array}\tag{3}
$$

where $\operatorname { s g } ( \cdot )$ is the stop-gradient operator. Thus, rather than explicitly integrating the drifting field at inference time, training repeatedly regresses the one-step generator toward samples displaced by the current field.

In practice, the expectations in $\operatorname { E q . 2 }$ are approximated with minibatch samples. Given references $u _ { 1 } , \ldots , u _ { N }$ each component takes the form $\textstyle \sum _ { i = 1 } ^ { N } w _ { i } ( u _ { i } - x )$ with normalized weights $\begin{array} { r } { w _ { i } = \kappa _ { \tau } ( x , u _ { i } ) / \sum _ { i = 1 } ^ { N } \kappa _ { \tau } ( x , u _ { j } ) } \end{array}$ . For the repulsive component, the references are other samples generated in the same training step, with the query sample excluded from its own reference set.

For later use, we write $V ( x ; \mathcal { V } , \mathcal { Z } )$ for the minibatch drifting field evaluated at $x ,$ with positive reference set Y and negative reference set $\mathcal { Z } ,$ , which can be seen as a Monte Carlo approximation of $V _ { p , q } ( x )$ . Throughout, a query particle is excluded from its own negative references, allowing $\mathcal { Z }$ to denote the full set of generated particles without additional notation.

## 3 Method

We introduce CoDrift, a compositional drifting framework for ofline reinforcement learning. CoDrift represents diferent learning objectives as displacement fields acting directly on generated actions and composes them into a single policy field. Specifically, CoDrift contains three objective-level fields: a conditional behavioral $~ f i e l d ,$ a marginal behavioral field, and a value $~ \ f e l d .$ The first two preserve behavioral structure at complementary statistical scales, while the third pushes generated actions toward higher-value regions. Figure 1 provides an overview.

We parameterize the policy as a one-step stochastic generator $f _ { \theta }$ . Given a state s and noise $\varepsilon \sim p _ { \varepsilon } = \mathcal { N } ( 0 , I )$

$$
\hat { a } = f _ { \theta } ( s , \varepsilon ) , \quad \pi _ { \theta } ( \cdot \mid s ) = \big ( f _ { \theta } ( s , \cdot ) \big ) _ { \# } p _ { \varepsilon } ,\tag{4}
$$

where $f _ { \theta }$ uses tanh to ensure that $\hat { a } \in [ - 1 , 1 ] ^ { d }$ . Sampling from $\pi _ { \theta }$ therefore requires only one noise draw and one forward pass.

## 3.1 Conditional Behavioral Field

The conditional behavioral field preserves the state-dependent action distribution of the ofline data. At the population level, it instantiates the drifting field of Sec. 2.2 between the behavioral conditional distribution

$p _ { \mathcal { D } } ( a \mid s )$ and the policy $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ , where $p _ { \mathcal { D } } ( a \mid s )$ plays the role of the target distribution $p _ { \mathrm { d a t a } } .$ , while $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ plays the role of the generated distribution $q _ { \theta }$

Given a minibatch $\boldsymbol { B } = \{ ( s _ { i } , a _ { i } ) \} _ { i = 1 } ^ { B }$ , we draw K independent noise samples for each state,

$$
\varepsilon _ { i , k } \sim { \mathcal { N } } ( 0 , I ) , \qquad { \hat { a } } _ { i , k } = f _ { \theta } ( s _ { i } , \varepsilon _ { i , k } ) , \qquad k = 1 , \ldots , K .
$$

For each state $s _ { i } ,$ the positive and generated reference sets are

$$
\mathcal { Y } _ { i } ^ { \mathrm { c o n d } } = \{ a _ { i } \} , \qquad \mathcal { Z } _ { i } ^ { \mathrm { c o n d } } = \{ \hat { a } _ { i , 1 } , \dots , \hat { a } _ { i , K } \} .\tag{5}
$$

Using the minibatch drifting notation introduced in Sec. 2.2, we define

$$
V _ { \mathrm { c o n d } } ( \hat { a } _ { i , k } ) : = V \left( \hat { a } _ { i , k } ; \mathcal { X } _ { i } ^ { \mathrm { c o n d } } , \mathcal { Z } _ { i } ^ { \mathrm { c o n d } } \right) .
$$

Because the positive set $\mathcal { V } _ { i } ^ { \mathrm { c o n d } }$ contains only the single action paired with $s _ { i } ,$ the attractive component has weight one and reduces to

$$
V _ { \mathrm { c o n d } } ^ { + } ( \hat { a } _ { i , k } ) = a _ { i } - \hat { a } _ { i , k } .\tag{6}
$$

The repulsive component is computed from the other generated actions at the same state,

$$
V _ { \mathrm { c o n d } } ^ { - } ( \hat { a } _ { i , k } ) = \sum _ { l \neq k } w _ { k l } ^ { \mathrm { c o n d } } \big ( \hat { a } _ { i , l } - \hat { a } _ { i , k } \big ) ,
$$

where the normalized afinities $\begin{array} { r } { w _ { k l } ^ { \mathrm { c o n d } } = \frac { \kappa _ { \tau } ( \hat { a } _ { i , k } , \hat { a } _ { i , l } ) } { \sum _ { j \neq k } \kappa _ { \tau } ( \hat { a } _ { i , k } , \hat { a } _ { i , j } ) } } \end{array}$ following $\operatorname { E q . 2 } ,$ and $\hat { a } _ { i , k }$ is excluded from its own negative references. Thus,

$$
V _ { \mathrm { c o n d } } = V _ { \mathrm { c o n d } } ^ { + } - V _ { \mathrm { c o n d } } ^ { - } .
$$

A distinctive issue arises in continuous-control ofline RL: an exact state is essentially observed only once, and hence $p _ { \mathcal { D } } ( a \mid s _ { i } )$ is represented by only one observed action. Consequently, the attractive component in Eq. 6 is efectively a one-sample Monte Carlo estimate of the population conditional drifting field, which can have high variance. This motivates a complementary field that can exploit distributional information pooled across states.

## 3.2 Marginal Behavioral Field

The same stochastic generator also induces a marginal action distribution when states are drawn from the ofline state distribution. Recall from Eq. 4 that, for a fixed state $s , f _ { \theta } ( s , \cdot )$ pushes the noise distribution $p _ { \varepsilon }$ forward to the conditional policy distribution $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s )$ . If we additionally draw $s \sim p _ { D } ( s )$ , then the joint mapping

$$
( s , \varepsilon ) \mapsto f _ { \theta } ( s , \varepsilon )
$$

pushes the product distribution $p _ { \mathcal { D } } ( s ) p _ { \varepsilon } ( \varepsilon )$ forward to the policy-induced marginal action distribution

$$
q _ { \theta } = ( f _ { \theta } ) _ { \# } \bigl ( p _ { \mathcal { D } } \otimes p _ { \varepsilon } \bigr ) ,
$$

where $\otimes$ denotes product distribution. Equivalently,

$$
q _ { \theta } ( a ) = \int p _ { \mathcal { D } } ( s ) \pi _ { \theta } ( a \mid s ) d s .
$$

The corresponding marginal action distribution in the ofline data is

$$
p _ { \mathcal { D } } ( a ) = \int p _ { \mathcal { D } } ( s ) p _ { \mathcal { D } } ( a \mid s ) d s .
$$

This yields a second distribution-matching objective at the marginal level. In particular, if

$$
\pi _ { \theta } ( a \mid s ) = p _ { \mathcal { D } } ( a \mid s ) ,\tag{7}
$$

then marginalizing over s immediately gives

$$
q _ { \theta } ( a ) = p _ { \mathcal { D } } ( a ) .\tag{8}
$$

Hence matching the marginal action distribution in Eq. 8 is a necessary condition for matching the conditional behavioral distribution in Eq. 7.

More importantly, the marginal distribution is much better sampled in an ofline minibatch. For each predicted action $\hat { a } _ { i , k }$ , we pool actions across states:

$$
\mathcal { Y } ^ { \mathrm { m a r g } } = \bigl \{ a _ { 1 } , \ldots , a _ { B } \bigr \} , \qquad \mathcal { Z } _ { k } ^ { \mathrm { m a r g } } = \bigl \{ \hat { a } _ { 1 , k } , \ldots , \hat { a } _ { B , k } \bigr \} .
$$

The marginal behavioral field acting on $\hat { a } _ { i , k }$ is

$$
V _ { \mathrm { m a r g } } ( \hat { a } _ { i , k } ) : = V ( \hat { a } _ { i , k } ; \mathcal { V } ^ { \mathrm { m a r g } } , \mathcal { Z } _ { k } ^ { \mathrm { m a r g } } ) .
$$

It decomposes into attractive and repulsive components,

$$
V _ { \mathrm { m a r g } } = V _ { \mathrm { m a r g } } ^ { + } - V _ { \mathrm { m a r g } } ^ { - } ,
$$

where the attraction is computed from all B ofline actions in the minibatch and the repulsion from generated actions pooled across states.

The crucial diference from conditional drifting is that state–action pairing is deliberately removed at the marginal level. States used to generate policy actions are sampled from $p _ { { D } } ( s )$ , while positive reference actions are sampled independently from $p _ { \mathcal { D } } ( a )$ . Therefore, instead of estimating a conditional expectation from the single action observed at one state $( \mathcal { V } _ { i } ^ { \mathrm { c o n d } }$ in Eq. 5), marginal drifting is able to estimate the corresponding distributional field using B positive samples in ${ \tt y m a r g }$ . Consequently, it provides a more stable, lower-variance generative signal while constraining a distribution that must also match whenever the full conditional behavior is matched. The conditional and marginal fields are thus complementary: the former preserves state–action correspondence, whereas the latter supplies population-level action-distribution information.

## 3.3 Value Field

Behavioral matching alone does not solve ofline RL: among actions supported by the dataset, the policy should favor those with higher expected return. We therefore express value improvement as a third action-space field. We maintain two critics, $Q _ { \varphi _ { 1 } }$ and $Q _ { \varphi _ { 2 } } ,$ , together with target networks $Q _ { \bar { \varphi } _ { 1 } }$ and $Q _ { \bar { \varphi } _ { 2 } }$ obtained through Polyak averaging (Fujimoto et al., 2018). For a transition $( s , a , r , s ^ { \prime } ) \sim \mathcal { D }$ , we sample

$$
a ^ { \prime } = f _ { \theta } ( s ^ { \prime } , \varepsilon ^ { \prime } ) , \qquad \varepsilon ^ { \prime } \sim \mathcal { N } ( 0 , I ) ,
$$

and form the Bellman target

$$
y = r + \gamma \frac { Q _ { \bar { \varphi } _ { 1 } } ( s ^ { \prime } , a ^ { \prime } ) + Q _ { \bar { \varphi } _ { 2 } } ( s ^ { \prime } , a ^ { \prime } ) } { 2 } .
$$

The critics minimize

$$
\mathcal { L } _ { \mathrm { c r i t i c } } = \mathbb { E } \left[ \frac { 1 } { 2 } \sum _ { n = 1 } ^ { 2 } \left( Q _ { \varphi _ { n } } ( s , a ) - y \right) ^ { 2 } \right] .
$$

When updating the actor, the critics are held fixed. Let

$$
Q ( s , a ) = \frac { Q _ { \varphi _ { 1 } } ( s , a ) + Q _ { \varphi _ { 2 } } ( s , a ) } { 2 } .
$$

The value field is simply the action gradient of the critic,

$$
V _ { \mathrm { v a l u e } } ( s , a ) = \nabla _ { a } Q ( s , a ) .
$$

It specifies the local action-space direction of increasing predicted return. Unlike the conditional and marginal fields, which arise from distribution matching, the value field is not associated with a reference distribution. Nevertheless, all three objects are displacement fields in the same action space and can therefore be composed directly.

## 3.4 Compositional Drifting

We now combine the three objective-level fields into a single policy field. For every generated particle $\hat { a } _ { i , k }$ define

$$
V _ { \mathrm { C o D r i f t } } ( s _ { i } , \hat { a } _ { i , k } ) = \eta _ { c } V _ { \mathrm { c o n d } } ( \hat { a } _ { i , k } ) + \eta _ { m } V _ { \mathrm { m a r g } } ( \hat { a } _ { i , k } ) + \eta _ { v } V _ { \mathrm { v a l u e } } ( s _ { i } , \hat { a } _ { i , k } ) ,\tag{9}
$$

where $\eta _ { c } , \eta _ { m } , \eta _ { v } \ge 0$ control the relative contributions of conditional behavioral matching, marginal behavioral matching, and value maximization. At a finer level, $\operatorname { E q . 9 }$ contains five displacement components:

$$
\begin{array} { r } { V _ { \mathrm { C o D r i f t } } = \eta _ { c } \left( V _ { \mathrm { c o n d } } ^ { + } - V _ { \mathrm { c o n d } } ^ { - } \right) + \eta _ { m } \left( V _ { \mathrm { m a r g } } ^ { + } - V _ { \mathrm { m a r g } } ^ { - } \right) + \eta _ { v } V _ { \mathrm { v a l u e } } . } \end{array}
$$

The first four components define two complementary behavioral fields, while the last introduces task-directed value optimization. Similar to the training of drifting models in Eq. 3, the joint field directly defines a displaced target for every generated action:

$$
\boldsymbol { \tilde { a } } _ { i , k } = \mathrm { s g } [ \hat { a } _ { i , k } + V _ { \mathrm { C o D r i f t } } ( s _ { i } , \hat { a } _ { i , k } ) ] .
$$

The actor is finally trained with

$$
\mathcal { L } _ { \mathrm { C o D r i f t } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { B K } \sum _ { i = 1 } ^ { B } \sum _ { k = 1 } ^ { K } { \lVert \hat { a } _ { i , k } - \tilde { a } _ { i , k } \rVert } _ { 2 } ^ { 2 } \right] ,
$$

which learns the one-step stochastic policy by moving toward the composed target.

In our implementation, we parameterize the field weights as

$$
\eta _ { v } = 1 , \qquad \eta _ { c } = \alpha \lambda , \qquad \eta _ { m } = \alpha ( 1 - \lambda ) ,
$$

where α controls the overall strength of behavioral regularization and $\lambda \in [ 0 , 1 ]$ balances conditional and marginal behavioral matching.

## 4 Experiments

We evaluate CoDrift in both ofline and ofline-to-online reinforcement learning settings across a broad collection of continuous-control benchmarks. Our experiments are designed to answer two main questions: 1) whether compositional drifting provides a competitive generative policy-learning framework across diverse ofline RL tasks, and 2) whether marginal drifting improves over conditional drifting alone.

## 4.1 Experimental Setup

Benchmarks. We follow the benchmark suite and evaluation setting of existing works (Park et al., 2025b; Mu, 2026). Specifically, we evaluate on OGBench (Park et al., 2025a), which contains diverse long-horizon continuous-control tasks spanning navigation, locomotion, and manipulation. The datasets of OGBench are collected by task-agnostic policies, and we use their standard single-task variants, where a fixed evaluation goal is specified, and semi-sparse task rewards are relabeled from the original trajectories. We consider ten state-based datasets across six domains: AntMaze, HumanoidMaze, AntSoccer, Cube, Scene, and Puzzle, with five evaluation tasks per dataset, together with five pixel-based visual manipulation tasks. We additionally evaluate on D4RL (Fu et al., 2020), including six AntMaze navigation tasks and twelve Adroit manipulation tasks. Altogether, the ofline evaluation comprises 73 tasks spanning diferent state and action dimensions, observation modalities, reward structures, and control problems. For ofline-to-online RL, we follow the same 15-task protocol used by FQL and DeFlow (Park et al., 2025b; Mu, 2026), including five OGBench tasks, six D4RL AntMaze tasks, and four D4RL Adroit tasks. The policy is first trained ofline and subsequently fine-tuned using online interactions without changing the learning objective.

Table 1 Offline RL results. CoDrift is evaluated against representative Gaussian-, difusion-, and flow-based ofline RL methods across OGBench and D4RL. Results for CoDrift are averaged over 8 seeds (4 for pixel-based tasks), and baseline values are taken from prior work (Tarasov et al., 2023; Hansen-Estruch et al., 2023; Chen et al., 2024; Park et al., 2025b; Mu, 2026). The pixel-based row is excluded from ranking as several baselines are unavailable. Lower average rank is better.
<table><tr><td></td><td colspan="3">Gaussian Policies</td><td colspan="3">Diffusion Policies</td><td colspan="5">Flow Policies</td><td>Drift Policy</td></tr><tr><td>Task Category</td><td>BC</td><td>IQL</td><td>ReBRAC</td><td>IDQL</td><td>SRPO</td><td>CAC</td><td>FAWAC</td><td>FBRAC</td><td>IFQL</td><td>FQL</td><td>DeFlow</td><td>CoDrift</td></tr><tr><td>OGBench antmaze-large</td><td>11 ±1</td><td>53 ±3</td><td>81 ±5</td><td>21 ±5</td><td>11 ±4</td><td>33 ±4</td><td>6±1</td><td>60 ±6</td><td>28 ±5</td><td>79 ±3</td><td>81 ±3</td><td>84±3</td></tr><tr><td>OGBench antmaze-giant</td><td>0±0</td><td>4±1</td><td>26 ±8</td><td>0±0</td><td>0±0</td><td>0±0</td><td>0±0</td><td>4±4</td><td>3±2</td><td>9±6</td><td>12±5</td><td>46 ±22</td></tr><tr><td>OGBench humanoidmaze-medium</td><td>2±1</td><td>33 ±2</td><td>22±8</td><td>1±0</td><td>1 ±1</td><td>53±8</td><td>19±1</td><td>38 ±5</td><td>60±14</td><td>58 ±5</td><td>48±4</td><td>83±5</td></tr><tr><td>OGBench humanoidmaze-large</td><td>1 ±0</td><td>2±1</td><td>2±1</td><td>1±0</td><td>0±0</td><td>0±0</td><td>0±0</td><td>2 ±0</td><td>11 ±2</td><td>4±2</td><td>5 ±2</td><td>15±5</td></tr><tr><td>OGBench antsoccer-arena</td><td>1 ±0</td><td>8±2</td><td>0±0</td><td>12 ±4</td><td>1±0</td><td>2 ±4</td><td>12±0</td><td> $1 6 \pm { } 1$ </td><td>33 ±6</td><td>60 ±2</td><td>67±3</td><td>66 ±5</td></tr><tr><td>OGBench visual manipulation</td><td></td><td>42 ±4</td><td>60 ±2</td><td></td><td></td><td></td><td></td><td>22±2</td><td>50±5</td><td>65 ±2</td><td>69 ±2</td><td>61 ±7</td></tr><tr><td>OGBench puzzle-3x3</td><td>2±0</td><td>9±1</td><td>21 ±1</td><td>10±2</td><td>18±1</td><td>19±0</td><td>6±2</td><td>14±4</td><td>19±1</td><td>30±1</td><td>43±4</td><td>82±13</td></tr><tr><td>OGBench puzzle-4x4</td><td>0±0</td><td>7±1</td><td>14±1</td><td>29±3</td><td>10±3</td><td>15±3</td><td>1±0</td><td>13±1</td><td>25 ±5</td><td>17±2</td><td>11 ±2</td><td>22 ±4</td></tr><tr><td>OGBench cube-single</td><td>5±1</td><td>83 ±3</td><td>91 ±2</td><td>95 ±2</td><td>80±5</td><td>85 ±9</td><td>81 ±4</td><td>79±7</td><td>79 ±2</td><td>96 ±1</td><td>96 ±2</td><td>92±3</td></tr><tr><td>OGBench cube-double</td><td>2±1</td><td>7±1</td><td>12±1</td><td>15±6</td><td>2±1</td><td>6±2</td><td>5±2</td><td>15±3</td><td>14±3</td><td>29 ±2</td><td>40±5</td><td>49±10</td></tr><tr><td>OGBench scene</td><td>5±1</td><td>28 ±1</td><td>41 ±3</td><td>46 ±3</td><td>20±1</td><td>40±7</td><td>30±3</td><td>45 ±5</td><td>30 ±3</td><td>56 ±2</td><td>51 ±3</td><td>59 ±2</td></tr><tr><td>D4RL antmaze</td><td>17</td><td>57</td><td>78</td><td>79</td><td>74</td><td>30 ±3</td><td>44±3</td><td>64±7</td><td>65±7</td><td>84±3</td><td>83 ±2</td><td>73±6</td></tr><tr><td>D4RL adroit</td><td>48</td><td>53</td><td>59</td><td>52 ±1</td><td>51±1</td><td>43±2</td><td>48±1</td><td>50±2</td><td>52 ±1</td><td>52±1</td><td>52 ±1</td><td>50 ±3</td></tr><tr><td>Average rank ↓ (12 categories)</td><td>10.96</td><td>7.38</td><td>5.29</td><td>6.21</td><td>9.42</td><td>8.04</td><td>9.79</td><td>6.58</td><td>5.58</td><td>3.08</td><td>3.13</td><td>2.54</td></tr></table>

Baselines. We compare CoDrift against eleven representative ofline RL methods spanning several policy classes. Gaussian-policy baselines include behavior cloning (BC), Implicit Q-Learning (IQL; Kostrikov et al., 2022), and ReBRAC (Tarasov et al., 2023). Difusion-based baselines include IDQL (Hansen-Estruch et al., 2023), SRPO (Chen et al., 2024), and Consistency Actor-Critic (CAC; Ding & Jin, 2024). Flow-based baselines include FQL (Park et al., 2025b), DeFlow (Mu, 2026), and three flow-policy variants introduced as baselines by Park et al. (2025b): FAWAC, FBRAC, and IFQL, which are flow counterparts of AWAC (Nair et al., 2020), Difusion-QL (Wang et al., 2023), and IDQL (Hansen-Estruch et al., 2023), respectively. In the ofline-to-online setting, we additionally compare with Cal-QL (Nakamoto et al., 2023) and RLPD (Ball et al., 2023). Baseline results are taken from prior work under the same benchmark protocols.

Evaluation protocol. We follow the evaluation protocol of Park et al. (2025b) to ensure direct comparability with previously reported results. CoDrift is trained for a fixed number of gradient steps and evaluated periodically using 50 rollout episodes. Following Kurenkov & Kolesnikov (2022), we do not select the bestperforming evaluation checkpoint, which can introduce selection bias. For OGBench, we report the average over the final three evaluation epochs and measure task success rate. For D4RL, we report the final evaluation epoch, using success rate for AntMaze and normalized return for Adroit (Fu et al., 2020). CoDrift results are averaged over eight random seeds, except for pixel-based tasks, which use four seeds. For the ofline-to-online experiments, we report performance after ofline pretraining at 10<sup>6</sup> steps and after the subsequent online fine-tuning phase at $2 \times 1 0 ^ { 6 }$ total steps.

Following prior work (Park et al., 2025b; Mu, 2026), values within 95% of the best result in each task category are highlighted as near-best. For average rank, where lower is better, only the best result is highlighted.

## 4.2 Comparison with the State of the Art

Ofline RL performance. Table 1 summarizes the ofline RL results. Across the full benchmark suite, CoDrift achieves the best overall average rank among the compared methods, indicating strong performance across diverse task domains. CoDrift performs particularly strongly on OGBench tasks, while remaining competitive on D4RL. These results show that the proposed compositional drifting formulation provides a strong alternative to existing Gaussian-, difusion-, and flow-based policy classes across a diverse collection of ofline RL problems.

Ofline-to-online performance. Table 2 reports the ofline-to-online results on the 15 fine-tuning tasks. CoDrift can be fine-tuned online without introducing a separate online-stage objective: newly collected transitions are added to the replay bufer, and training proceeds using the same approach as in the ofline phase. Across the 15 tasks, CoDrift attains the best overall average rank among the compared methods. These results show that the same compositional drifting formulation remains efective when transitioning from purely ofline training to online policy improvement, without requiring an algorithmic change between the two phases.

Table 2 Offline-to-online RL results. Performance before and after online fine-tuning on the 15-task benchmark suite. CoDrift results are averaged over 8 seeds, and baseline values are taken from prior work. Lower average rank is better.
<table><tr><td>Task</td><td>IQL</td><td>ReBRAC</td><td>Cal-QL</td><td>RLPD</td><td>IFQL</td><td>FQL</td><td>DeFlow</td><td>CoDrift</td></tr><tr><td>humanoidmaze-medium-navigate-singletask-v0</td><td> $2 1 _ { \pm 1 3 } ~  ~ 1 6 _ { \pm 8 }$ </td><td> $1 6 \pm 2 0  1 \pm 1$ </td><td> $0 \pm \mathrm { { 0 }  0 \pm \mathrm { { 0 } } }$ </td><td> $0 \pm 0  8 \pm 1 0$ </td><td> $5 6 \pm \imath 5 \to 8 2 \pm 2 0$ </td><td> $1 2 \pm \tau  2 2 \pm 1 2$ </td><td> $1 3 \pm 5  6 5 \pm 1 2$ </td><td> ${ \bf 9 4 \pm 5 }  { \bf 9 9 } _ { \pm 1 }$ </td></tr><tr><td>antsoccer-arena-navigate-singletask-v0</td><td> $2 \pm 1  0 \pm 0$ </td><td> $0 \pm \mathrm { { 0 }  0 \pm \mathrm { { 0 } } }$ </td><td> $0 \pm \mathrm { { o }  0 \pm \mathrm { { o } } }$ </td><td> $0 \pm \mathrm { o }  0 \pm \mathrm { o }$ </td><td> $2 6 \pm 1 5 \  \ 3 9 \pm 1 0$ </td><td> $2 8 \pm 8  8 6 \pm \mathrm { s }$ </td><td> $4 4 \pm \mathbf { s } \  \ \mathbf { 8 6 } \pm \mathbf { 3 }$ </td><td> $5 2 \pm \tau  8 6 \pm \mathrm { s }$ </td></tr><tr><td>cube-double-play-singletask-v0</td><td> $0 \pm { } 1  0 \pm { } 0$ </td><td> $\yen 123,456$ </td><td> $0 \pm \mathrm { o }  0 \pm \mathrm { o }$ </td><td> $0 \pm \mathrm { o }  0 \pm \mathrm { o }$ </td><td> $1 2 \pm 9  4 0 \pm { \mathrm { s } }$ </td><td> $4 0 \pm \imath 1 \to 9 2 \pm 3$ </td><td>48 ±13 → 93 ±4</td><td> ${ \bf 6 5 \pm } { \bf 1 0 }  { \bf 1 0 0 } _ { \pm 0 }$ </td></tr><tr><td>scene-play-singletask-v0</td><td> $1 4 \pm 1 1  1 0 \pm 9$ </td><td> $\mathbf { 5 5 } \pm \mathbf { 1 0 }  \mathbf { 1 0 0 } \pm \mathbf { 0 }$ </td><td> $1 \pm 2  5 0 \pm { \mathrm { s } } 3$ </td><td> ${ \bf 0 } _ { \pm 0 }  { \bf 1 0 0 } _ { \pm 0 }$ </td><td> $0 \pm 1  6 0 \pm 3 9$ </td><td> $\mathbf { 8 2 } \pm \mathbf { 1 1 }  {  }  { \mathbf { 1 0 0 } } _ { \pm 1 }$ </td><td> $\mathbf { 5 8 _ { \pm 1 4 } }  {  } \mathbf { 1 0 0 _ { \pm 1 } }$ </td><td> $9 2 \pm 6  {  } 1 0 0  { \pm }  { 0 }$ </td></tr><tr><td>puzzle-4x4-play-singletask-v0</td><td> $5 \pm 2  1 \pm 1$ </td><td> $8 \pm 4  1 4 \pm 3 5$ </td><td> $0 \pm \mathrm { o }  0 \pm \mathrm { o }$ </td><td>0 ±0 → 100 ±1</td><td> $2 3 \pm \delta  1 9 \pm 3 3$ </td><td> $8 \pm 3  3 8 \pm 5 2$ </td><td> $\mathbf { 4 } \pm \mathbf { 4 } \  \ \mathbf { 1 0 0 } _ { \pm 0 }$ </td><td> $1 2 \pm 5  8 8 \pm 3 2$ </td></tr><tr><td>antmaze-umaze-v2</td><td> ${ 7 7 }  \mathbf { 9 6 }$ </td><td> $9 8  7 5$ </td><td> ${ \bf 7 7 }  { \bf 1 0 0 }$ </td><td> $\mathbf { 0 \pm o { 0 } }  \mathbf { 9 8 \pm { 3 } }$ </td><td> $9 4 \pm \textrm { s }  \ \mathbf { 9 6 } _ { \pm 2 }$ </td><td> $9 7 \pm \mathbf { \hat { \mathbf { 2 } } } \  \ \mathbf { 9 9 } \pm \mathbf { \hat { \mathbf { 1 } } }$ </td><td> $9 6 \pm 3 \  \ 9 9 \pm 2$ </td><td> $9 0 \pm \mathfrak { s } \  \ \mathbf { 9 9 } \pm \mathfrak { s }$ </td></tr><tr><td>antmaze-umaze-diverse-v2</td><td> $6 0  6 4$ </td><td> ${ 7 4  9 8 }$ </td><td> $3 2  { \bf 9 8 }$ </td><td> $0 \pm \mathrm { { o }  9 4 \pm \mathrm { { s } } }$ </td><td> $6 9 \pm 2 0  9 3 \pm { \mathfrak { s } }$ </td><td> $\mathbf { 7 9 } \pm \mathbf { 1 6 }  {  }  { \mathbf { 1 0 0 } } _ { \pm 1 }$ </td><td> $\mathbf { 8 7 } \pm \mathbf { 6 } \  \ \mathbf { 9 9 } \pm \mathbf { \scriptscriptstyle \pm 1 }$ </td><td> $6 6 \pm \imath 3 \to 9 8 \pm 2$ </td></tr><tr><td>antmaze-medium-play-v2</td><td> $7 2  9 0$ </td><td> ${ \bf 8 8 }  { \bf 9 8 }$ </td><td> ${ \bf 7 2 }  { \bf 9 9 }$ </td><td> $\mathbf { 0 } _ { \pm 0 }  \mathbf { 9 8 } _ { \pm 2 }$ </td><td> $5 2 \pm 1 9  9 3 \pm 2$ </td><td> $7 7 \pm \tau \  \ 9 7 \pm 2$ </td><td> $7 6 \pm \textrm { s }  \textrm { 9 7 } \pm 2$ </td><td> $6 0 \pm \imath \imath \to 9 8 \pm 2$ </td></tr><tr><td>antmaze-medium-diverse-v2</td><td> $6 4  9 2$ </td><td> $8 5  { \bf 9 9 }$ </td><td> ${ \bf 6 2 }  { \bf 9 8 }$ </td><td> $\mathbf { 0 } _ { \pm 0 }  \mathbf { 9 7 } _ { \pm 2 }$ </td><td> $4 4 \pm 2 6 \to 8 9 \pm 4$ </td><td> ${ \bf 5 5 \pm } { \bf 5 } \Rightarrow { \bf 9 7 \pm } { \bf 3 }$ </td><td> ${ \bf 6 1 } _ { \pm 9 }  { \bf 9 8 } _ { \pm 2 }$ </td><td> $5 9 \pm 1 5 \  \ 9 7 \pm 2$ </td></tr><tr><td>antmaze-large-play-v2</td><td>38 → 64</td><td>68 → 32</td><td>32 → 97</td><td> ${ \bf 0 } _ { \pm 0 }  { \bf 9 3 } _ { \pm 5 }$ </td><td> $6 4 \pm 1 4  8 0 \pm { \mathrm { s } }$ </td><td> $6 6 \pm 4 0  8 4 \pm 3 0$ </td><td> ${ \bf 7 6 } _ { \pm 6 }  { \bf 9 5 } _ { \pm 4 }$ </td><td> $4 1 \pm 1 7  9 2 \pm 3$ </td></tr><tr><td>antmaze-large-diverse-v2</td><td> $2 7  6 4$ </td><td> $6 7  7 2$ </td><td> $4 4  9 2$ </td><td> $\mathbf { 0 } _ { \pm 0 }  \mathbf { 9 4 } _ { \pm 3 }$ </td><td> $6 9 \pm 6  8 6 \pm { \mathfrak { s } }$ </td><td> ${ \bf 7 5 } _ { \pm 2 4 }  { \bf 9 4 } _ { \pm 3 }$ </td><td> ${ 7 6 \pm 9  9 6 \pm 2 }$ </td><td> ${ \bf 7 } \pm \imath { \bf 9 }  { \bf 9 2 } \pm { \bf 4 }$ </td></tr><tr><td>pen-cloned-v1</td><td> $8 4  1 0 2$ </td><td> $7 4  1 3 8$ </td><td> $- 3  - 3$ </td><td> $\mathrm { 3 \pm 2 \ \to \ 1 2 0 \pm 1 0 }$ </td><td> $7 7 \pm \tau  1 0 7 \pm 1 0$ </td><td> $\mathbf { 5 3 _ { \pm 1 4 } }  \mathbf { 1 4 9 _ { \pm 6 } }$ </td><td> $6 4 \pm \imath 3  1 4 2 _ { \pm 7 }$ </td><td> $5 0 \pm \infty  \mathbf { 1 4 3 } \pm \mathbf { 4 }$ </td></tr><tr><td>door-cloned-v1</td><td> $1  2 0$ </td><td> $0  1 0 2$ </td><td> $- 0  - 0$ </td><td> $0 \scriptstyle \pm 0 \ \to \ 1 0 2 \scriptstyle \pm 7$ </td><td> $3 \pm 2  5 0 \pm 1 5$ </td><td> $0 \pm 0  1 0 2 \pm { \mathrm { s } }$ </td><td> $0 { \scriptstyle \pm 0 } \ \to \ 9 9 { \scriptstyle \pm 2 }$ </td><td> ${ \bf 0 } _ { \pm 0 }  { \bf 1 0 8 } _ { \pm 4 }$ </td></tr><tr><td>hammer-cloned-v1</td><td> $1  5 7$ </td><td> $7  1 2 5$ </td><td> $0  0$ </td><td> $0 \pm 0  1 2 8 \pm 2 9$ </td><td> $4 \pm 2 \  \ 6 0 \pm 1 4$ </td><td> $0 \pm 0  1 2 7 \pm { _ { 1 7 } }$ </td><td> $7 \pm 3  1 0 6 \pm 1 0$ </td><td> $\mathbf { 0 \pm 0 }  \mathbf { 1 3 9 } _ { \pm 4 }$ </td></tr><tr><td>relocate-cloned-v1</td><td> $0  0$ </td><td> $1  7$ </td><td> $- 0  - 0$ </td><td> $0 \pm 0  2 \pm 2$ </td><td> $- 0 \pm \mathrm { o } \  \ 5 \pm 3$ </td><td> $\bf { 0 \pm 1 }  \bf { 6 2 } \pm \bf { s }$ </td><td> $1 \pm 0  3 5 \pm 8$ </td><td> $0 \pm \ t { 0 }  2 3 \pm \tau$ </td></tr><tr><td>Average rank ↓ (15 tasks)</td><td>7.10</td><td>4.90</td><td>5.47</td><td>4.30</td><td>5.63</td><td>3.07</td><td>2.83</td><td>2.70</td></tr></table>

<table><tr><td>Task</td><td> $\mathrm { C o D r i f t }$ </td><td>w/o Marg</td><td>∆</td></tr><tr><td>antmaze-large</td><td> ${ \bf 8 4 } \pm \bf { 3 }$ </td><td> $7 7 \pm 6$ </td><td> $+ 7$ </td></tr><tr><td>humanoidmaze-medium</td><td> ${ \bf 8 3 \pm 5 }$ </td><td> $7 7 \pm 5$ </td><td> $+ 6$ </td></tr><tr><td>cube-single</td><td> $\mathbf { 9 2 \pm 3 }$ </td><td> $8 6 \pm 3$ </td><td> $+ 6$ </td></tr><tr><td>puzzle-3x3</td><td> ${ \bf 8 2 \pm 1 3 }$ </td><td> $7 4 \pm 4$ </td><td> $+ 8$ </td></tr><tr><td>Average</td><td>85.3</td><td>78.5</td><td> $+ 6 . 8$ </td></tr></table>

Table 3 Effectiveness of marginal drifting. Success rates on four representative tasks, averaged over 8 seeds (± standard deviation). w/o Marg removes the marginal behavioral field. ∆ denotes the resulting gain from marginal drifting.

## 4.3 Effectiveness of Marginal Drifting

As introduced in Sec. 3.2, conditional drifting estimates its attractive component from the single action observed at each state, whereas marginal drifting pools actions across states and therefore constructs its drifting field from a substantially larger positive sample. The resulting batch-level signal complements the paired state–action supervision provided by the conditional field.

To isolate the contribution of marginal drifting, we compare the full CoDrift model with a variant that removes the marginal behavioral field by setting λ = 1, so that all behavioral regularization is assigned to conditional drifting. Table 3 reports results on four representative tasks drawn from diferent OGBench domains. Adding marginal drifting improves performance on all four tasks, with gains ranging from 6 to 8 percentage points and an average improvement of 6.8 points. This result supports the role of marginal drifting as a complementary behavioral signal beyond per-state conditional drifting.

## 5 Related Work

Ofline RL and behavior regularization. Ofline RL seeks to maximize return using a fixed dataset while avoiding extrapolation errors caused by out-of-distribution actions (Levine et al., 2020; Fujimoto et al., 2019). Existing approaches include value regularization (Kumar et al., 2020; An et al., 2021), in-sample learning (Kostrikov et al., 2022; Xu et al., 2023), and policy regularization. The latter retains an actor–critic formulation while constraining the learned policy toward the behavior distribution, either through explicit penalties (Wu et al., 2019; Fujimoto & Gu, 2021; Tarasov et al., 2023) or weighted regression (Peng et al., 2019; Nair et al., 2020). CoDrift follows the same general principle of balancing behavioral fidelity and value maximization, but represents these objectives as action-space displacement fields. In particular, it constrains behavior at both the conditional and marginal levels and composes these behavioral fields directly with a critic-guided value field.

Generative policies. Expressive generative models provide a natural policy class for representing complex and multimodal action distributions. Existing generative policies have explored several modeling paradigms, including autoregressive models (Kim et al., 2024; Wang et al., 2026a), generative adversarial networks (Vuong et al., 2022), difusion models (Wang et al., 2023; Chi et al., 2023), and flow matching (Park et al., 2025b; Chang et al., 2026; Wang et al., 2026b; Espinosa-Dice et al., 2025). CoDrift takes a diferent approach based on drifting models (Deng et al., 2026). Drifting models learn a stochastic generator from displacement fields acting directly on generated samples and support native one-step generation. More importantly for ofline RL, their displacement-based formulation provides a natural interface for combining heterogeneous learning objectives: once diferent objectives are expressed as action-space fields, their efects can be composed additively before training the generator. Concurrent with our work, Koo et al. (2026) also applies drifting models to one-step policy learning, but uses a single drifting field toward a value-reweighted target. In contrast, CoDrift explicitly composes multiple objective-level fields within a unified one-step generative framework.

## 6 Concluding Remarks

We introduced CoDrift, a compositional drifting framework for generative policy learning in ofline reinforcement learning. The central idea is to represent heterogeneous learning objectives in a common language of actionspace displacement fields and compose them into a single policy field. For ofline RL, CoDrift instantiates this principle with three complementary objective-level fields: a conditional behavioral field that preserves statedependent behavioral structure, a marginal behavioral field that captures population-level action structure, and a critic-guided value field that promotes higher-return actions. The resulting joint field directly defines the regression target of a one-step stochastic actor.

Empirically, CoDrift achieves strong performance across a broad collection of OGBench and D4RL tasks in both ofline and ofline-to-online settings, attaining the best overall average rank among the compared methods. These results demonstrate that compositional drifting provides a competitive alternative to existing Gaussian-, difusion-, and flow-based policy classes, while retaining native one-step stochastic generation at deployment.

More broadly, we view the main advantage of the drifting formulation as its objective compositionality. Diferent requirements need not share the same loss construction; it is suficient that each can be translated into a displacement in action space. Their interaction can then be expressed through the additive composition of the corresponding fields and absorbed into a single generative policy. We hope this perspective provides a useful foundation for incorporating additional objectives and constraints into future one-step generative policies for reinforcement learning.

## References

Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based ofline reinforcement learning with diversified q-ensemble. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Philip J. Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Eficient online reinforcement learning with ofline data. In International Conference on Machine Learning (ICML), 2023.

Jianlei Chang, Ruofeng Mei, Wei Ke, and Xiangyu Xu. Eficientflow: Eficient equivariant flow policy learning for embodied ai. In AAAI Conference on Artificial Intelligence (AAAI), 2026.

Huayu Chen, Cheng Lu, Zhengyi Wang, Hang Su, and Jun Zhu. Score regularized policy optimization through difusion behavior. In International Conference on Learning Representations (ICLR), 2024.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Difusion policy: Visuomotor policy learning via action difusion. In Robotics: Science and Systems (RSS), 2023.

Mingyang Deng, He Li, Tianhong Li, Yilun Du, and Kaiming He. Generative modeling via drifting. arXiv:2602.04770, 2026.

Zihan Ding and Chi Jin. Consistency models as a rich and eficient policy class for reinforcement learning. In International Conference on Learning Representations (ICLR), 2024.

Nicolas Espinosa-Dice, Yiyi Zhang, Yiding Chen, Bradley Guo, Owen Oertell, Gokul Swamy, Kianté Brantley, and Wen Sun. Scaling ofline RL via eficient and expressive shortcut models. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv:2004.07219, 2020.

Scott Fujimoto and Shixiang Gu. A minimalist approach to ofline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning (ICML), 2018.

Scott Fujimoto, David Meger, and Doina Precup. Of-policy deep reinforcement learning without exploration. In International Conference on Machine Learning (ICML), 2019.

Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine. IDQL: Implicit q-learning as an actor-critic method with difusion policies. arXiv:2304.10573, 2023.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv:2406.09246, 2024.

Juil Koo, Mingue Park, Jiwon Choi, Yunhong Min, and Minhyuk Sung. Drifting field policy: A one-step generative policy via Wasserstein gradient flow. arXiv:2605.07727, 2026.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Ofline reinforcement learning with implicit q-learning. In International Conference on Learning Representations (ICLR), 2022.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for ofline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Vladislav Kurenkov and Sergey Kolesnikov. Showing your ofline reinforcement learning work: Online evaluation budget matters. In International Conference on Machine Learning (ICML), 2022.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Ofline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv:2005.01643, 2020.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Zhancun Mu. DeFlow: Decoupling manifold modeling and value maximization for ofline policy extraction. arXiv:2601.10471, 2026.

Ashvin Nair, Murtaza Dalal, Abhishek Gupta, and Sergey Levine. AWAC: Accelerating online reinforcement learning with ofline datasets. arXiv:2006.09359, 2020.

Mitsuhiko Nakamoto, Yuexiang Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-QL: Calibrated ofline RL pre-training for eficient online fine-tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking ofline goal-conditioned rl. In International Conference on Learning Representations (ICLR), 2025a.

Seohong Park, Qiyang Li, and Sergey Levine. Flow q-learning. In International Conference on Machine Learning (ICML), 2025b.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable of-policy reinforcement learning. arXiv:1910.00177, 2019.

Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov. Revisiting the minimalist approach to ofline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Quan Vuong, Aviral Kumar, Sergey Levine, and Yevgen Chebotar. Dasco: Dual-generator adversarial support constrained ofline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Ruiheng Wang, Shuanghao Bai, Haoran Zhang, Badong Chen, and Xiangyu Xu. Blockvla: Accelerating autoregressive vla via block difusion finetuning. arXiv:2605.13382, 2026a.

Zeyuan Wang, Da Li, Yulin Chen, Ye Shi, Liang Bai, Tianyuan Yu, and Yanwei Fu. One-step generative policies with Q-learning: A reformulation of MeanFlow. In AAAI Conference on Artificial Intelligence (AAAI), 2026b.

Zhendong Wang, Jonathan J. Hunt, and Mingyuan Zhou. Difusion policies as an expressive policy class for ofline reinforcement learning. In International Conference on Learning Representations (ICLR), 2023.

Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized ofline reinforcement learning. arXiv:1911.11361, 2019.

Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan, and Xianyuan Zhan. Ofline rl with no ood actions: In-sample learning via implicit value regularization. In International Conference on Learning Representations (ICLR), 2023.