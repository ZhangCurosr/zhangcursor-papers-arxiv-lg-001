# One-Step Generative Surrogate Models via Block-Triangular Joint Drifting

Nicholas Geissler Courant Institute of Mathematical Sciences, New York University

Shreya Jha Courant Institute of Mathematical Sciences, New York University

Ricardo Baptista

Statistical Sciences, University of Toronto

Benjamin Peherstorfer

Institute of Mathematics, EPFL

## Abstract

Drifting provides a direct route to one-step generative models, but applying it directly to stochastic transition modeling requires multiple samples of the next state conditioned on the same current state. Standard trajectory data, however, typically provide only one realized next state for each observed current state and therefore do not provide an empirical approximation of the corresponding conditional distribution over possible next states. We introduce block-triangular joint drifting, which instead applies a projected drift field to the empirically accessible joint distribution of consecutive states. Importantly, the block-triangular architecture preserves the current-state marginal while making its second component a direct sampler of the conditional distribution of possible next states. The resulting surrogate generates stochastic trajectories with one model evaluation per time step, without auxiliary generative steps between time steps. Numerical experiments demonstrate accurate marginal and trajectory-dependent statistics and favorable accuracy-cost tradeofs compared with deterministic, difusion-, flow-, and distillation-based generative surrogate models.

Keywords: model reduction, stochastic modeling, generative modeling, dynamical systems AMS: 65N22, 65N30, 65F55, 65D40

## 1 Introduction

Learning low-cost reduced models of stochastic systems is a ubiquitous task for enabling uncertainty quantification, ensemble forecasting, design, and other outer-loop applications. While for deterministic systems it is suficient to learn a function that maps the current state x(t) to the state x(t + 1) at the next time step, see, e.g., [37, 20, 33, 29, 39, 7], for stochastic systems, one is typically interested in learning a model to rapidly sample from the transition law represented as a conditional distribution $p _ { t } ( \cdot | x ( t ) )$ given a realization $x ( t )$ of the current state X(t). Once such a model is available, stochastic trajectories can be generated autoregressively.

Difusion- and flow-based generative modeling provides flexible mechanisms for modeling complex conditional distributions such as transition laws [46, 24, 48, 44, 31, 52, 1]. These approaches have been extended to probabilistic forecasting and simulation of dynamical systems [14, 13, 28], where a generative model conditioned on the current state can represent a distribution over possible future states. Their main drawback is computational costs of generating trajectories. Generating a new sample from the transition law typically requires multiple denoising or flow-integration steps. Thus, a rollout of T time steps with K generative steps per time step requires $K T$ model evaluations, which becomes costly when generating large trajectory ensembles.

This has motivated work on one- and few-step generative models, often building on difusionand flow-based modeling. For example, progressive distillation compresses multi-step difusion samplers into students that use increasingly fewer steps [41]; consistency models enable one- or few-step sampling through either distillation or direct training [47]; and ReFlow iteratively straightens generative trajectories to reduce the number of generation steps needed per time step [32]. Other approaches such as inductive moment matching [56], MeanFlow [21], flow-map self-distillation [12], and stochastic lifting [10] instead target one- or few-step generation directly, without requiring a pretrained teacher, even though MeanFlow can be applied as a distillation method as well. These methods reduce generation costs but when used as teacher-based distillation schemes for stochastic dynamics, they require first learning a multi-step conditional generative model and subsequently compressing it. Additionally, inherent to the distillation is that accuracy is traded for runtime speedups.

A complementary line of work learns deterministic reduced dynamics for stochastic systems, enabling fast rollouts without an auxiliary generative sampling process. Action Matching [35, 8] and DICE [11] learn population dynamics consistent with the evolving time marginals, while, $\mathrm { e . g . }$ , [43, 9] extend this viewpoint to more general, non-gradient transport fields. Because time marginals do not determine the stochastic transition law, such population models cannot in general recover trajectory-dependent statistics. Yet another complementary line of work learns reduced models of stochastic or uncertain dynamics by identifying lowdimensional stochastic diferential equations with learned drift and difusion operators [19] or by combining dimensionality reduction with probabilistic latent dynamics to construct generative reduced-order models [15].

In this work we instead build on drifting, a generative modeling concept for learning one-step generative models directly [16, 23, 53, 30]. Drifting starts from the distribution induced by the current model and iteratively transports its samples toward samples from a target distribution using a distribution-dependent drift field. The transport is performed during training. After convergence, the model itself maps reference noise to a target sample in one model evaluation. In particular, drifting does not require numerical integration of an auxiliary generative dynamics when generating samples.

However, applying drifting directly to a transition law $p _ { t } ( \cdot \mid x ( t ) )$ raises a dificulty that does not arise in the same way for conditional difusion or flow-matching objectives. Such methods can be trained from paired samples $( X ( t ) , X ( t + 1 ) )$ , each pair providing a sample for a regression objective for the conditional score or velocity field, and repeated realizations of $X ( t + 1 )$ at exactly the same conditioning state are not required. Drifting is diferent because its update is defined through a distribution-dependent field and therefore requires an empirical sample representation of the target distribution. If one sets $p _ { t } ( \cdot | X ^ { ( i ) } ( t ) )$ ) as target, trajectory data provide only the single successor $X ^ { ( i ) } ( t + 1 )$ , so the corresponding empirical target is the point mass $\delta _ { X ^ { ( i ) } ( t + 1 ) }$ . It contains no direct information about the spread or shape of the transition law.

Our key step is to change the distribution to which drifting is applied. Rather than drifting toward the transition law directly, which is a conditional distribution, we lift the problem to the joint law $p _ { t } ( X ( t ) , X ( t + 1 ) )$ , for which the observed transition pairs provide samples directly. Learning joint distributions in order to obtain conditional generators has a long history in measure transport. In particular, triangular transport maps expose conditional distributions through their structure and have been developed for sample-based Bayesian inference, density estimation, and nonlinear state-space models [34, 50, 49, 5]. Closest to our construction, block-triangular transport methods learn conditional samplers from samples of a joint distribution, thereby sharing information across conditioning values without requiring repeated samples at each value [4]. Joint generative models can also be conditioned at inference, including in difusion-based scientific forecasting and simulation-based inference [45, 22]. Related conditional optimal-transport and flow-matching approaches likewise exploit joint or block-triangular constructions for conditional generation [26, 55]. Joint models of consecutive states have also been proposed for generative forecasting, with conditional predictions extracted from collections of generated joint samples [54].

Our use of the joint law difers in its role for drifting and in the resulting inference procedure. We constrain the joint generator to the block-triangular form and project the joint drifting field onto the second block. The joint law thus supplies the empirical target needed by drifting during training, while the triangular block structure makes the learned second component a direct sampler of the transition law. After training, no conditioning procedure, reverse difusion, flow integration, or search over joint samples is required; one neural-network evaluation produces a sample of the transition law.

We demonstrate on numerical experiments with stochastically forced Burgers’ and Navier-Stokes equations that the proposed approach generates diverse and accurate trajectories with one (neural-network) model evaluation per time step. In particular, we show that our approach matches or improves upon more expensive conditional flow and autoregressive difusion baselines, as well as upon one- and few-step MeanFlow and ReFlow-based distilled models.

## 2 Preliminaries and problem formulation

We recapitulate preliminaries and state the problem of applying drifting to transition laws.

## 2.1 Setup

Consider an n-dimensional stochastic process $X ( t )$ that is defined over $\mathbb { R } ^ { n }$ and discrete time $t \in \{ 0 , \ldots , T \} \subset \mathbb { N }$ . We denote the corresponding transition law as $X ( t + 1 ) \sim p _ { t } ( \cdot | X ( t ) )$ with initial $X ( 0 ) \sim \mu _ { 0 }$ . Note that the transition law can depend on time $t ,$ i.e., we are not restricting the following to time-homogeneous transition laws. The joint law of $X ( t )$ and $X ( t + 1 )$ is denoted as

$$
p _ { t } ( X ( t ) , X ( t + 1 ) ) = \mu _ { t } ( X ( t ) ) p _ { t } ( X ( t + 1 ) | X ( t ) ) ,
$$

with the time marginal $\mu _ { t }$ recursively defined as $\begin{array} { r } { \mu _ { t + 1 } ( y ) = \int p _ { t } ( y | x ) \mu _ { t } ( x ) \mathrm { d } x } \end{array}$ . In the following, we have access to training data obtained from the process $X ( t )$

$$
\mathcal D = \{ X ^ { ( i ) } ( t ) | i = 1 , \ldots , N , t = 0 , \ldots , T \} \subset \mathbb { R } ^ { n } ,\tag{1}
$$

which consists of $i = 1 , \ldots , N$ trajectories $X ^ { ( i ) } ( 0 ) , \ldots , X ^ { ( i ) } ( T )$ over $t = 0 , \ldots , T$ time steps.

## 2.2 One-step generative surrogate modeling

We seek a map $g : \mathbb { R } ^ { n } \times \mathbb { R } ^ { d } \times \mathbb { N } \to \mathbb { R } ^ { n }$ such that for $\mu _ { t } { - } \mathrm { a . e . } ~ x _ { t }$ , we have

$$
g ( x ( t ) , \cdot , t ) _ { \sharp } \pi = p _ { t } ( \cdot | x ( t ) ) ,\tag{2}
$$

where $\pi$ is a suitable reference distribution such as a standard normal. The condition (2) implies that if $z \sim \pi$ is a sample from the reference $\pi ,$ then evaluating the map g at $z ,$

$$
g ( x ( t ) , z , t ) \sim p _ { t } ( \cdot | x ( t ) ) ,
$$

gives a sample of the transition law $p _ { t } ( \cdot | x ( t ) )$ . Correspondingly, we refer to g as a one-step map because a single evaluation $g ( x ( t ) , z , t )$ produces a sample from the transition law at the next time point. In particular, no numerical integration of auxiliary dynamics or a sequence of intermediate generative steps are required.

## 2.3 Drifting schemes for time marginals

One approach for learning one-step generators to sample from a target distribution η is given by drifting schemes, first introduced in [16], and further explored in, e.g., [23, 53, 30]. For a parametrized function $f _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ , where θ denotes the vector of, e.g., neural-network weights, define the model distribution induced by the pushforward $q _ { \theta } = ( f _ { \theta } ) _ { \sharp } \pi$ . Drifting schemes iteratively update the parameters $\theta _ { j }$ of $f _ { \theta _ { j } }$ over iterations $j = 0 , 1 , 2 , . . .$ . such that the model distribution $q _ { \theta _ { j } } = ( f _ { \theta _ { j } } ) _ { \sharp } \pi$ improves the match to the target distribution $\eta .$ The iterative updating is achieved via a drift field $V _ { \eta , q _ { \theta _ { i } } } : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ , which determines how samples $x \sim q _ { \theta _ { j } }$ should be transported,

$$
T _ { h , q } ( x ) = x + h V _ { \eta , q } ( x ) ,
$$

where $h > 0$ is a step size. The drifting fields must satisfy $V _ { \eta , \eta } = 0$ so that the target distribution η is a fixed point $T _ { h , \eta } ( x ) = x$ of $q \mapsto T _ { h , q }$ . On the parameter level, if at iteration $j$ we have $x = f _ { \theta _ { i } } ( z )$ with a sample $z \sim \pi .$ , then the transported sample is

$$
\tilde { x } = T _ { h , q _ { \theta _ { j } } } ( f _ { \theta _ { j } } ( z ) ) = f _ { \theta _ { j } } ( z ) + h V _ { \eta , q _ { \theta _ { j } } } ( f _ { \theta _ { j } } ( z ) ) .
$$

Because at iteration $j$ we have $f _ { \theta _ { j } } ( z ) \sim q _ { \theta _ { j } }$ , the distribution of the transported samples is $\tilde { q } _ { j + 1 } = ( T _ { h , q _ { \theta _ { j } } } ) _ { \sharp } q _ { \theta _ { j } }$ . This defines an iteration: draw samples from the current ${ { q } _ { \theta _ { j } } }$ with $f _ { \theta _ { j } }$ transport the samples with the drift field, and then fit $\theta _ { j + 1 }$ so that $f _ { \theta _ { j + 1 } }$ generates samples close to the transported samples, which is achieved with the loss

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ; \boldsymbol { \theta } _ { j } ) = \mathbb { E } _ { z \sim \pi } \left[ \| f _ { \boldsymbol { \theta } } ( z ) - \left( f _ { \boldsymbol { \theta } _ { j } } ( z ) + h V _ { \eta , q _ { \boldsymbol { \theta } _ { j } } } \left( f _ { \boldsymbol { \theta } _ { j } } ( z ) \right) \right) \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{3}
$$

In this manner, each gradient step approximates one fixed-point iteration, with $f _ { \theta _ { j } }$ parameterizing the current iterate.

## 2.4 Problem formulation: Conditional drifting from trajectory data

The drifting field $V _ { \eta , q _ { \theta _ { i } } }$ depends on the target distribution $\eta ;$ however, in the data-driven setting, which we consider here, the drifting field is constructed from a collection of samples of the target $\eta$ together with samples from the current model distribution $q _ { \theta _ { j } }$ . In particular, the samples from the target distribution provide the empirical approximation of $\eta$ used by the drifting procedure. If we now consider the direct application of drifting to learn how to sample from the transitional law, this would mean to condition on a state $x ( t )$ and take $p _ { t } ( \cdot | x ( t ) )$ to be the target distribution $\eta .$ The dificulty is that the trajectory data in (1) do not provide multiple samples from $p _ { t } ( \cdot | x ( t ) )$ . Instead, for each observed conditioning state $x ( t ) = X ^ { ( i ) } ( t )$ , the training data contain only its single realized successor $X ^ { ( i ) } ( t + 1 )$ Consequently, conditioning the empirical training data on an observed state $X ^ { ( i ) } ( t )$ does not provide the drifting procedure with samples that characterize the corresponding conditional $p _ { t } ( \cdot | X ^ { ( i ) } ( t ) )$ . In particular, just having access to one realization of $X ^ { ( i ) } ( t + 1 )$ for a given $X ^ { ( i ) } ( t )$ provides no information about the spread, shape, or possible multimodality of the conditional distribution. We therefore seek a drifting scheme that can learn the transition law from trajectory data without requiring multiple samples from the conditional distribution for a given conditioning state.

## 3 Block-Triangular Joint Drifting

We propose to apply drifting to the joint transition law, for which transition-pair samples are available, while restricting the generator to a block-triangular form from which the desired conditional one-step map can be extracted.

## 3.1 Targeting the joint law

For a fixed time $t ,$ the training data (1) contains the transition pairs $\{ ( X ^ { ( i ) } ( t ) , X ^ { ( i ) } ( t { + } 1 ) ) \} _ { i = 1 } ^ { N } ,$ which are samples from the joint law $p _ { t } ( \cdot , \cdot )$ . Thus, the training data contains multiple samples from the joint law $p _ { t } ( \cdot , \cdot )$ for a fixed t as long as $N > 1$ . Rather than separating the transition pairs at time t via conditioning, we use the joint law $p _ { t }$ as the target distribution of the drifting procedure. In this way, all available transition pairs at time t contribute samples to the same drifting target. We introduce the corresponding parametrized map $f _ { \theta } : \mathbb { R } ^ { n } \times \mathbb { R } ^ { d } \times \mathbb { N } \to \mathbb { R } ^ { 2 n }$ and model joint distribution

$$
q _ { \theta } ( t ) = f _ { \theta } ( \cdot , \cdot , t ) _ { \sharp } ( \mu _ { t } \times \pi ) .\tag{4}
$$

Note that the model joint distribution depends on time t because $f$ is dependent on time.

## 3.2 A block-triangular parametrization

Applying drifting directly with $f _ { \theta }$ while using as target the joint law $p _ { t }$ means that $f _ { \theta }$ produces samples $( X ( t ) , X ( t + 1 ) )$ jointly, but does not in general provide a way to fix an arbitrary current state $x ( t )$ and sample from the transition law $p _ { t } ( \cdot | x ( t ) )$ . To eficiently sample from the transition law, we consider a specific parametrization of $f _ { \theta }$ , so that when $f _ { \theta }$ has learned to sample from the joint law, we additionally obtain a one-step map for the conditional law. We therefore follow the block-triangular construction of [4] and parametrize $f _ { \theta }$ as

$$
f _ { \theta } ( x ( t ) , z , t ) = ( x ( t ) , g _ { \theta } ( x ( t ) , z , t ) ) ,\tag{5}
$$

where the conditioning variable $x ( t ) \sim \mu _ { t }$ is a sample from the time marginal $\mu _ { t }$ and $z \sim \pi$ is a reference sample, which is sampled independently from the conditioning variable. The parametrization $( 5 )$ has a block triangular form because its first component is independent of z. In particular, the first marginal induced by it is $\mu _ { t }$ for every value of $\theta ,$ while only the second component $g _ { \theta }$ depends on $\theta .$ The block-triangular structure ensures that matching the joint law recovers the desired conditional law. In particular, if

$$
f _ { \theta } ( \cdot , \cdot , t ) _ { \sharp } ( \mu _ { t } \times \pi ) = p _ { t } ,
$$

then

$$
g _ { \theta } ( x ( t ) , \cdot , t ) _ { \sharp } \pi = p _ { t } ( \cdot | x ( t ) )
$$

for $\mu _ { t } { \mathrm { - a l m o s t } }$ every $x ( t ) ;$ see Theorem 2.4 of [4]. Thus, by parametrizing $f _ { \theta }$ as in $( 5 )$ , it is suficient to train the block-triangular map $f _ { \theta }$ to match the joint law $p _ { t }$ , and one obtains a one-step map for the transition law via $g _ { \theta }$

## 3.3 Drifting field

We now derive a drifting field that is compatible with the joint law as target and with the block-triangular parametrization as model $f _ { \theta }$ . We begin with the drifting field introduced in [23] and applying it to the joint law $p _ { t }$ over the state space $\mathbb { R } ^ { 2 n }$

$$
V _ { p _ { t } , q _ { \theta } ( t ) } ^ { \varepsilon } ( y ) = - \nabla _ { y } \frac { \delta S _ { \varepsilon } \big ( q _ { \theta } ( t ) , p _ { t } \big ) } { \delta q _ { \theta } ( t ) } ( y ) , \qquad y \in \mathbb { R } ^ { 2 n } ,\tag{6}
$$

where $S _ { \varepsilon }$ denotes the Sinkhorn divergence with entropic regularization parameter $\varepsilon > 0 ;$ see, $\mathrm { e . g . }$ , [18]. The first variation with respect to the model distribution is denoted as $\delta / \delta q _ { \theta } ( t )$

The field in (6) acts on both components of the joint state. Our block-triangular parametrization (5), however, fixes the first component and therefore cannot realize motion in the dimensions corresponding to the first block of the state space of the joint law. With $y = ( y _ { 1 } , y _ { 2 } ) \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { n }$ , we decompose the joint field (6) as

$$
V _ { p _ { t } , q _ { \theta } ( t ) } ^ { \varepsilon } ( y ) = ( V _ { 1 } ^ { \varepsilon } ( y ) , V _ { 2 } ^ { \varepsilon } ( y ) ) \mathrm { , }
$$

where $V _ { 1 } ^ { \varepsilon } , V _ { 2 } ^ { \varepsilon } : \mathbb { R } ^ { 2 n }  \mathbb { R } ^ { n }$ denote joint field’s first and second components, respectively. We then restrict the joint field to the directions compatible with the block-triangular parametrization. Let $P$ denote the orthogonal projection onto the second block. We define

$$
P V _ { p _ { t } , q _ { \theta } ( t ) } ^ { \varepsilon } ( y ) = \left( 0 , V _ { 2 } ^ { \varepsilon } ( y ) \right) ,\tag{7}
$$

to obtain the projected field, which is the drifting field that we use to update the conditional component $g _ { \theta }$

## 3.4 Loss formulation

For a fixed time $t ,$ we insert the projected field (7) into the drifting loss (3), which yields

$$
\begin{array} { r l } & { \mathcal { L } _ { t } ( \theta ; \theta _ { j } ) = } \\ & { \mathbb { E } _ { x ( t ) \sim \mu _ { t } , z \sim \pi } \Big [ \big \| f _ { \theta } ( x ( t ) , z , t ) - \big ( f _ { \theta _ { j } } ( x ( t ) , z , t ) + h P V _ { p _ { t } , q _ { \theta _ { j } } ( t ) } ^ { \varepsilon } \big ( f _ { \theta _ { j } } ( x ( t ) , z , t ) \big ) \big ) \big \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{8}
$$

Using the block-triangular parametrization (5) and the projected field (7), the transported target appearing in (9) can be written as

$$
\begin{array} { r l r } { \mathrm {  { \left. \int { \theta _ { j } } ( x ( t ) , z , t ) + h P V _ { p t , \mathrm { q } _ { j } } ^ { \varepsilon } ( t ) \big ( f _ { \theta _ { j } } ( x ( t ) , z , t ) \big ) = \right. } } } & { } & \\ { \left. \big ( x ( t ) , g _ { \theta _ { j } } ( x ( t ) , z , t ) + h V _ { 2 } ^ { \varepsilon } \big ( f _ { \theta _ { j } } ( x ( t ) , z , t ) \big ) \big ) \right. } & { } & \end{array}\tag{9}
$$

Since $f _ { \theta } ( x ( t ) , z , t )$ has first component $x ( t )$ by (5), the first block of the residual in (9) is identically zero. Therefore, (9) reduces to

$$
\begin{array} { r } { \mathcal { L } _ { t } ( \theta ; \theta _ { j } ) = \mathbb { E } _ { x ( t ) \sim \mu _ { t } , z \sim \pi } \left[ \left| \left| g _ { \theta } ( x ( t ) , z , t ) - \left( g _ { \theta _ { j } } ( x ( t ) , z , t ) + h V _ { 2 } ^ { \varepsilon } \big ( f _ { \theta _ { j } } ( x ( t ) , z , t ) \big ) \right) \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{10}
$$

Finally, we average the loss over the available time points,

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ; \boldsymbol { \theta } _ { j } ) = \mathbb { E } _ { t \sim \mathcal { U } ( \{ 0 , \dots , T - 1 \} ) } \left[ \mathcal { L } _ { t } ( \boldsymbol { \theta } ; \boldsymbol { \theta } _ { j } ) \right] . } \end{array}\tag{11}
$$

For each $t ,$ the drifting field is constructed from the corresponding target $p _ { t }$ and current model distribution $q _ { \theta _ { j } } ( t )$ , while the parameters $\theta _ { j }$ are shared across time.

## 3.5 Theoretical properties

Because the joint field (6) is the Sinkhorn drifting field considered in [23], we can directly build on the well-posedness and other theoretical results established therein. In particular, the projected field (7) inherits the regularity conditions required in [23]. Indeed, since $P$ is an orthogonal projection,

$$
\| P V _ { p _ { t } , q } ^ { \varepsilon } ( y ) - P V _ { p _ { t } , q } ^ { \varepsilon } ( y ^ { \prime } ) \| _ { 2 } \le \| V _ { p _ { t } , q } ^ { \varepsilon } ( y ) - V _ { p _ { t } , q } ^ { \varepsilon } ( y ^ { \prime } ) \| _ { 2 } \qquad \mathrm { f o r ~ a l l ~ } y , y ^ { \prime } \in \mathbb { R } ^ { 2 n } ,
$$

so the growth and Lipschitz bounds satisfied by the joint field are preserved under projection.   
Thus, the well-posedness result of [23] applies also to the projected field.

We additionally note that $q = p _ { t }$ being a (possibly non-unique) fixed point of $P V _ { p _ { t } , q } ^ { \varepsilon }$ follows from the work in [23, 18] under suitable assumptions. In particular, it is established in [18] that for distributions with compact support, $q = p _ { t } \implies S _ { \varepsilon } ( q , p _ { t } ) = 0$ , and by regularity and non-negativity of the Sinkhorn divergence, $S _ { \varepsilon } ( q , p _ { t } ) = 0 \implies V _ { p _ { t } , q _ { \theta } ( t ) } ^ { \varepsilon } \equiv 0$ . Because $P$ is a projection onto the second component of $V _ { p _ { t } , q _ { \theta } ( t ) } ^ { \varepsilon } , p _ { t }$ remains a fixed-point of the projected velocity field.

For drifting, we would like the reverse direction to hold as well, so that we characterize the drift field as

$$
P V _ { p _ { t } , q } ^ { 0 } = 0 \qquad q _ { } { \mathrm { - a . e . } } \qquad \Longleftrightarrow \qquad q = p _ { t }
$$

within the class of joint distributions whose first marginal is $\mu _ { t }$ . This ensures that the projection does not introduce additional zeros of the drift field corresponding to joint distributions diferent from the target distribution. This implication is not immediate, because projecting a vector field can eliminate nonzero components. Hence it is possible in principle that the projected field vanishes even though the original field does not. We show that this cannot occur for (7) when the model distribution $q$ and the target distribution p<sub>t</sub> have the same first marginal $\mu _ { t }$ and the distributions are supported compactly and suitably regular. Importantly, the following statement is restricted to the unregularized drift field (6), i.e., $\varepsilon = 0$ , and thus the following theorem should be viewed as a statement for an idealized setting. The proof proceeds by representing the unregularized drift field through the quadratic-cost optimal transport map from $q$ to $p _ { t }$ . We then show that the condition $P V _ { p _ { t } , q } ^ { 0 } = 0$ forces both components of this transport map to coincide with those of the identity map. Consequently, the optimal transport map is the identity q-almost everywhere, which implies $q = p _ { t }$

Theorem 1. Set $\varepsilon = 0$ in the drift field (6) and fix a time t. Let $p _ { t } , q \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { 2 n } )$ have the same first marginal $\mu _ { t } ,$ where $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { 2 n } )$ denotes the space of probability measures over $\mathbb { R } ^ { 2 n }$ with finite second moment. Assume that q and $p _ { t }$ are supported on the closure of $\Omega _ { 1 } \times \Omega _ { 2 }$ , where $\Omega _ { 1 } , \Omega _ { 2 } \subset \mathbb { R } ^ { n }$ are bounded, open, convex sets. Assume further that q and $p _ { t }$ are absolutely continuous with respect to Lebesgue measure and that there exist constants $0 < \underline { { C } } \le \overline { { C } } < \infty$ such that their densities satisfy Lebesgue-almost everywhere

$$
{ \underline { { C } } } \leq q \leq { \overline { { C } } } \quad o n \ \Omega _ { 1 } \times \Omega _ { 2 } , \qquad { \underline { { C } } } \leq p _ { t } \leq { \overline { { C } } } \quad o n \ \Omega _ { 1 } \times \Omega _ { 2 } .\tag{12}
$$

Then

$$
{ \cal P } V _ { p _ { t } , q } ^ { 0 } = 0 \qquad q \cdot a . e . \qquad \Longleftrightarrow \qquad q = p _ { t } .\tag{13}
$$

Proof. Because $q$ is absolutely continuous with respect to the Lebesgue measure, Brenier’s theorem implies that the quadratic-cost optimal transport from $q$ to $p _ { t }$ is induced by a map $T _ { q } ^ { p _ { t } } : \mathbb { R } ^ { 2 n }  \mathbb { R } ^ { 2 n }$ that is unique q-almost everywhere and of the form $T _ { q } ^ { p _ { t } } = \nabla u$ , where $u : \mathbb { R } ^ { 2 n }  \mathbb { R }$ is convex [3, Theorem 2.26]. By assumption (12), the densities $q$ and $p _ { t }$ are bounded above and bounded away from zero on $\Omega _ { 1 } \times \Omega _ { 2 }$ . Therefore, by the regularity theorem for quadratic optimal transport [3, Theorem 2.27], the optimal transport map $T _ { q } ^ { p _ { t } }$ admits a Hölder-continuous, and hence continuous, representative on $\Omega _ { 1 } \times \Omega _ { 2 }$ . We use this continuous representative in the following. Furthermore, since u is convex and $\nabla u = T _ { q } ^ { p _ { t } }$ almost everywhere on $\Omega _ { 1 } \times \Omega _ { 2 }$ , the continuity of this representative implies that u is continuously diferentiable and $\nabla u = T _ { q } ^ { p _ { t } }$ everywhere on $\Omega _ { 1 } \times \Omega _ { 2 }$

We now relate $T _ { q } ^ { p _ { t } }$ to the unregularized drifting field used in (6). For the quadratic cost $c ( y , \bar { y } ) = \textstyle { \frac { 1 } { 2 } } \| y - \bar { y } \| _ { 2 } ^ { 2 }$ , a source Kantorovich potential associated with the Brenier potential u is

$$
\Phi _ { q } ( y ) = \frac { 1 } { 2 } \| y \| _ { 2 } ^ { 2 } - u ( y ) ,
$$

up to an additive constant; see [42, Proposition 1.21] and the discussion following that proposition. Now note that the lower bound (12) implies that the support of $q$ is the closure $\overline { { \Omega _ { 1 } \times \Omega _ { 2 } } }$ , and likewise for $p _ { t }$ . Additionally, the quadratic cost belongs to $C ^ { 1 } ( \overline { { \Omega _ { 1 } \times \Omega _ { 2 } } } \times$ $\overline { { \Omega _ { 1 } \times \Omega _ { 2 } } } )$ . Hence the assumptions of [42, Proposition 7.18] are satisfied, which shows that the Kantorovich potential is unique up to an additive constant.

Now recall that for $\varepsilon = 0$ , the Sinkhorn divergence $S _ { \varepsilon }$ used in the definition of the drift field (6) becomes $\begin{array} { r } { S _ { 0 } ( q , p _ { t } ) = \frac { 1 } { 2 } W _ { 2 } ^ { 2 } ( q , p _ { t } ) } \end{array}$ ; see, e.g., [23]. By [42, Proposition 7.17], the Kantorovich potential is a subgradient of the functional $q \mapsto { \textstyle \frac { 1 } { 2 } } W _ { 2 } ^ { 2 } ( q , p _ { t } )$ and when it is unique up to additive constants, it represents its first variation,

$$
\frac { \delta S _ { 0 } ( q , p _ { t } ) } { \delta q } = \Phi _ { q }
$$

up to an additive constant. Consequently, for $q \mathrm { . }$ -almost every $y \in \Omega _ { 1 } \times \Omega _ { 2 }$ ，

$$
V _ { p _ { t } , q } ^ { 0 } ( y ) = - \nabla \Phi _ { q } ( y ) = \nabla u ( y ) - y = T _ { q } ^ { p _ { t } } ( y ) - y .
$$

Hence the unregularized drifting field is precisely the displacement field of the quadratic optimal transport from q to $p _ { t }$

Now we are ready to prove the implication

$$
q = p _ { t } \quad \Longrightarrow \quad P V _ { p _ { t } , q } ^ { 0 } = 0 \quad q \mathrm { - a l m o s t ~ e v e r y w h e r e } .
$$

Assume that $q = p _ { t }$ . Since the identity map transports $q$ to itself with zero quadratic cost, $W _ { 2 } ( q , q ) = 0$ . Because $T _ { q } ^ { q }$ is an optimal transport map from $q$ to itself,

$$
\frac { 1 } { 2 } \int _ { \mathbb { R } ^ { 2 n } } \| T _ { q } ^ { q } ( y ) - y \| _ { 2 } ^ { 2 } q ( \mathrm { d } y ) = \frac { 1 } { 2 } W _ { 2 } ^ { 2 } ( q , q ) = 0 .
$$

The integrand is nonnegative, and therefore $T _ { q } ^ { q } ( y ) = y$ for q-almost every $y .$ It follows from $V _ { q , q } ^ { 0 } = T _ { q } ^ { q } - \mathrm { I d }$ that $V _ { q , q } ^ { 0 } = 0$ q-almost everywhere, and hence $P V _ { q , q } ^ { 0 } = 0$ q-almost everywhere. We now prove the converse. Assume that $P V _ { p _ { t } , q } ^ { 0 } = 0$ q-almost everywhere. The strategy is now to show that under this assumption, the map $T _ { q } ^ { p _ { t } }$ must be the identity map. We do this component-wise, starting with the second component. Write $y = ( y _ { 1 } , y _ { 2 } ) \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { n }$ and decompose the optimal transport map as

$$
T _ { q } ^ { p _ { t } } ( y _ { 1 } , y _ { 2 } ) = \left( T _ { 1 } ( y _ { 1 } , y _ { 2 } ) , T _ { 2 } ( y _ { 1 } , y _ { 2 } ) \right) .
$$

Recalling that $P ( v _ { 1 } , v _ { 2 } ) = ( 0 , v _ { 2 } )$ , the identity $V _ { p _ { t } , q } ^ { 0 } = T _ { q } ^ { p _ { t } } -$ Id gives

$$
P V _ { p _ { t } , q } ^ { 0 } ( y _ { 1 } , y _ { 2 } ) = \left( 0 , T _ { 2 } ( y _ { 1 } , y _ { 2 } ) - y _ { 2 } \right) .
$$

Consequently,

$$
T _ { 2 } ( y _ { 1 } , y _ { 2 } ) = y _ { 2 } \qquad q \mathrm { - a . e . } ~ ( y _ { 1 } , y _ { 2 } ) \in \Omega _ { 1 } \times \Omega _ { 2 } .\tag{14}
$$

We next show that (14) holds everywhere on $\Omega _ { 1 } \times \Omega _ { 2 }$ . By $( 1 2 ) , q ( y ) \geq \underline { { { C } } } > 0$ for Lebesguealmost every $y \in \Omega _ { 1 } \times \Omega _ { 2 }$ . Hence, for every measurable set $A \subset \Omega _ { 1 } \times \Omega _ { 2 }$

$$
q ( A ) = \int _ { A } q ( y ) \mathrm { d } y \geq \underline { { C } } | A | ,
$$

where $| A |$ denotes the Lebesgue measure of A. Consequently, $q ( A ) = 0$ implies $| { \cal A } | = 0$ . Thus, (14) implies

$$
T _ { 2 } ( y _ { 1 } , y _ { 2 } ) = y _ { 2 } \qquad \mathrm { ~ f o r ~ L e b e s g u e - a l m o s t ~ e v e r y ~ } ( y _ { 1 } , y _ { 2 } ) \in \Omega _ { 1 } \times \Omega _ { 2 } .
$$

Since $T _ { q } ^ { p _ { t } }$ is continuous on $\Omega _ { 1 } \times \Omega _ { 2 }$ , the map $( y _ { 1 } , y _ { 2 } ) \mapsto T _ { 2 } ( y _ { 1 } , y _ { 2 } ) - y _ { 2 }$ is continuous there. A continuous function that vanishes Lebesgue-almost everywhere on an open set must vanish everywhere on that set. Therefore

$$
T _ { 2 } ( y _ { 1 } , y _ { 2 } ) = y _ { 2 } \qquad \mathrm { f o r ~ e v e r y ~ } ( y _ { 1 } , y _ { 2 } ) \in \Omega _ { 1 } \times \Omega _ { 2 } .\tag{15}
$$

Let us now consider the first component of $T _ { q } ^ { p _ { t } }$ and show that it agrees with the identity q-almost everywhere. Fix an arbitrary reference point $y _ { 2 } ^ { 0 } \in \Omega _ { 2 }$ and define $h : \Omega _ { 1 } \to \mathbb { R }$ by

$$
h ( y _ { 1 } ) = u ( y _ { 1 } , y _ { 2 } ^ { 0 } ) - \frac { 1 } { 2 } \| y _ { 2 } ^ { 0 } \| _ { 2 } ^ { 2 } .
$$

Fix arbitrary $y _ { 1 } \in \Omega _ { 1 }$ and $y _ { 2 } \in \Omega _ { 2 }$ . Since $\Omega _ { 2 }$ is convex, the line segment

$$
y _ { 2 } ( r ) = ( 1 - r ) y _ { 2 } ^ { 0 } + r y _ { 2 } , \qquad r \in [ 0 , 1 ] ,
$$

is contained in $\Omega _ { 2 }$ . Define $\gamma ( r ) = u ( y _ { 1 } , y _ { 2 } ( r ) )$ . Since u is continuously diferentiable, the chain rule gives

$$
\gamma ^ { \prime } ( r ) = \nabla _ { y _ { 2 } } u ( y _ { 1 } , y _ { 2 } ( r ) ) \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) = y _ { 2 } ( r ) \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) ,
$$

where we used $\nabla _ { y _ { 2 } } u ( y _ { 1 } , y _ { 2 } ) = y _ { 2 }$ on $\Omega _ { 1 } \times \Omega _ { 2 }$ which follows from (15) with $T _ { q } ^ { p _ { t } } = \nabla u$ Therefore,

$$
\begin{array} { l } { { u ( y _ { 1 } , y _ { 2 } ) - u ( y _ { 1 } , y _ { 2 } ^ { 0 } ) = \displaystyle \int _ { 0 } ^ { 1 } y _ { 2 } ( r ) \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) \mathrm { d } r } } \\ { { \ \displaystyle = \int _ { 0 } ^ { 1 } \big ( ( 1 - r ) y _ { 2 } ^ { 0 } + r y _ { 2 } \big ) \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) \mathrm { d } r } } \\ { { \ \displaystyle = y _ { 2 } ^ { 0 } \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) \int _ { 0 } ^ { 1 } ( 1 - r ) \mathrm { d } r + y _ { 2 } \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) \int _ { 0 } ^ { 1 } r \mathrm { d } r } } \\ { { \ \displaystyle = \frac { 1 } { 2 } y _ { 2 } ^ { 0 } \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) + \frac { 1 } { 2 } y _ { 2 } \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) } } \\ { { \ \displaystyle = \frac { 1 } { 2 } ( y _ { 2 } + y _ { 2 } ^ { 0 } ) \cdot ( y _ { 2 } - y _ { 2 } ^ { 0 } ) = \frac { 1 } { 2 } \| y _ { 2 } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| y _ { 2 } ^ { 0 } \| _ { 2 } ^ { 2 } . } } \end{array}
$$

Hence

$$
u ( y _ { 1 } , y _ { 2 } ) = h ( y _ { 1 } ) + { \frac { 1 } { 2 } } \| y _ { 2 } \| _ { 2 } ^ { 2 }\tag{16}
$$

for every $( y _ { 1 } , y _ { 2 } ) \in \Omega _ { 1 } \times \Omega _ { 2 }$ . Since u is convex, the restriction $y _ { 1 } \mapsto u ( y _ { 1 } , y _ { 2 } ^ { 0 } )$ is convex, and subtracting the constant $\frac { 1 } { 2 } \| y _ { 2 } ^ { 0 } \| _ { 2 } ^ { 2 }$ shows that h is convex as well. Since u is continuously diferentiable, h is also continuously diferentiable. Diferentiating (16) therefore gives

$$
T _ { q } ^ { p _ { t } } ( y _ { 1 } , y _ { 2 } ) = \nabla u ( y _ { 1 } , y _ { 2 } ) = { \big ( } \nabla h ( y _ { 1 } ) , y _ { 2 } { \big ) } ,
$$

everywhere on $\Omega _ { 1 } \times \Omega _ { 2 }$

It remains to determine the first component $\nabla h ( y _ { 1 } )$ . Let $P _ { 1 } : \mathbb { R } ^ { 2 n }  \mathbb { R } ^ { n }$ denote the projection $P _ { 1 } ( y _ { 1 } , y _ { 2 } ) = y _ { 1 }$ . By assumption, $q$ and $p _ { t }$ have the same first marginal, so

$$
( P _ { 1 } ) _ { \sharp } q = \mu _ { t } = ( P _ { 1 } ) _ { \sharp } p _ { t } .
$$

Since $( T _ { q } ^ { p _ { t } } ) _ { \sharp } q = p _ { t }$ , we obtain

$$
\begin{array} { r l } & { \mu _ { t } = ( P _ { 1 } ) _ { \sharp } p _ { t } = ( P _ { 1 } ) _ { \sharp } \big ( ( T _ { q } ^ { p _ { t } } ) _ { \sharp } q \big ) = ( P _ { 1 } \circ T _ { q } ^ { p _ { t } } ) _ { \sharp } q } \\ & { \quad = ( \nabla h \circ P _ { 1 } ) _ { \sharp } q = ( \nabla h ) _ { \sharp } \big ( ( P _ { 1 } ) _ { \sharp } q \big ) = ( \nabla h ) _ { \sharp } \mu _ { t } , } \end{array}
$$

which shows that ∇h transports $\mu _ { t }$ to itself. We already established that $h ( y _ { 1 } ) = u ( y _ { 1 } , y _ { 2 } ^ { 0 } ) -$ $\begin{array} { r l r } { \frac { 1 } { 2 } \| y _ { 2 } ^ { 0 } \| _ { 2 } ^ { 2 } } \end{array}$ is convex. Thus, the map $\nabla h$ is an optimal transport map for the quadratic cost $[ 3 ,$ Theorem 2.13]. The identity map also transports $\mu _ { t }$ to itself and has zero quadratic cost. Hence the optimal transport cost from $\mu _ { t }$ to itself is zero. Since $\nabla h$ is optimal, it must also attain zero cost. Because the quadratic cost is nonnegative and vanishes only when source and target points coincide, it follows that

$$
\nabla h ( y _ { 1 } ) = y _ { 1 } \qquad \mu _ { t } { \mathrm { - a l m o s t ~ e v e r y w h e r e } } .
$$

Since $( P _ { 1 } ) _ { \sharp } q = \mu _ { t }$ , the identity $\nabla h ( y _ { 1 } ) = y _ { 1 } \ \mu _ { t }$ -almost everywhere implies, together with the representation $T _ { q } ^ { p _ { t } } ( y _ { 1 } , y _ { 2 } ) = ( \nabla h ( y _ { 1 } ) , y _ { 2 } )$ , that

$$
T _ { q } ^ { p _ { t } } ( y _ { 1 } , y _ { 2 } ) = ( y _ { 1 } , y _ { 2 } )
$$

for q-almost every $( y _ { 1 } , y _ { 2 } ) \in \Omega _ { 1 } \times \Omega _ { 2 }$ . Thus $T _ { q } ^ { p _ { t } } = \mathrm { I d }$ q-almost everywhere. Because $( T _ { q } ^ { p _ { t } } ) \sharp q = p _ { t }$ , it follows that

$$
p _ { t } = ( T _ { q } ^ { p _ { t } } ) _ { \sharp } q = ( \operatorname { I d } ) _ { \sharp } q = q .
$$

This proves the converse implication and hence (13).

## 3.6 Empirical loss and computational procedure

We train with the trajectory data (1) by replacing the distributions in (11) with empirical distributions constructed from mini-batches. At each training iteration j, we sample a set

$$
\mathcal { T } = \{ t _ { 1 } , \dots , t _ { n _ { t } } \} \subset \{ 0 , \dots , T - 1 \}
$$

$n _ { t }$ time indices uniformly. For each $t \in \mathcal { T }$ , we draw $n _ { B }$ transition pairs

$\{ ( x ^ { b } ( t ) , x ^ { b } ( t + 1 ) ) \} _ { b = 1 } ^ { n _ { B } }$ from the trajectories in (1). These samples define the empirical target distribution

$$
\widehat { p _ { t } } = \frac { 1 } { n _ { B } } \sum _ { b = 1 } ^ { n _ { B } } \delta _ { ( x ^ { b } ( t ) , x ^ { b } ( t + 1 ) ) } .\tag{17}
$$

To approximate the current joint model distribution (4), we independently sample conditioning states $\{ \bar { x } ^ { b } ( t ) \} _ { b = 1 } ^ { n _ { B } } \sim \mu _ { t }$ from the observed states at time t and reference samples $\{ z ^ { b } \} _ { b = 1 } ^ { n _ { B } } \sim \pi$ We then compute

$$
y _ { j } ^ { b } ( t ) = f _ { \theta _ { j } } ( \bar { x } ^ { b } ( t ) , z ^ { b } , t ) , \qquad b = 1 , \ldots , n _ { B } ,
$$

and define

$$
\widehat { q } _ { \theta _ { j } } ( t ) = \frac { 1 } { n _ { B } } \sum _ { b = 1 } ^ { n _ { B } } \delta _ { y _ { j } ^ { b } ( t ) } .\tag{18}
$$

We further draw an independent second batch $\{ \bar { x } ^ { \prime b } ( t ) \} _ { b = 1 } ^ { n _ { B } } \sim \mu _ { t }$ and $\{ z ^ { \prime b } \} _ { b = 1 } ^ { n _ { B } } \sim \pi$ , and form

$$
y _ { j } ^ { \prime b } ( t ) = f _ { \theta _ { j } } ( \bar { x } ^ { \prime b } ( t ) , z ^ { \prime b } , t ) , \qquad \widehat { q } _ { \theta _ { j } } ^ { \prime } ( t ) = \frac { 1 } { n _ { B } } \sum _ { b = 1 } ^ { n _ { B } } \delta _ { y _ { j } ^ { \prime b } ( t ) } .
$$

For each $t \in \tau$ , we compute the empirical joint field $\widehat { V } _ { t , j } ^ { \varepsilon }$ from $\widehat { q } _ { \theta _ { j } } ( t ) , \widehat { p } _ { t }$ and the independent empirical model distribution $\widehat { q } _ { \theta _ { i } } ^ { \prime } ( t )$ using the Sinkhorn barycentric projections of [23]. We then project this field according to (7),

$$
P \widehat { V } _ { t , j } ^ { \varepsilon } ( y ) = \left( 0 , \widehat { V } _ { 2 , t , j } ^ { \varepsilon } ( y ) \right) .
$$

The empirical counterpart of (10) is therefore

$$
\widehat { \mathcal { L } } ( \theta ; \theta _ { j } ) = \frac { 1 } { n _ { t } n _ { B } } \sum _ { t \in \mathcal { T } } \sum _ { b = 1 } ^ { n _ { B } } \left. g _ { \theta } ( \bar { x } ^ { b } ( t ) , z ^ { b } , t ) - \left( g _ { \theta _ { j } } ( \bar { x } ^ { b } ( t ) , z ^ { b } , t ) + h \widehat { V } _ { 2 , t , j } ^ { \varepsilon } ( y _ { j } ^ { b } ( t ) ) \right) \right. _ { 2 } ^ { 2 } .\tag{19}
$$

A gradient step on (19) then updates $\theta _ { j }$ to $\theta _ { j + 1 }$ . After training, given an initial state $\widehat { X } ( 0 ) \sim \mu _ { 0 }$ , a sample trajectory is generated autoregressively by

$$
\begin{array} { r } { z ( t ) \sim \pi , \qquad \widehat { X } ( t + 1 ) = g _ { \theta } \big ( \widehat { X } ( t ) , z ( t ) , t \big ) , \qquad t = 0 , \dots , T - 1 . } \end{array}\tag{20}
$$

Thus, the block-triangular map $f _ { \theta }$ is used only implicitly during training to construct the empirical joint drifting field, while the second component $g _ { \boldsymbol { \theta } }$ of $f _ { \theta }$ is the one-step transition map used at inference time.

## 4 Experiments

To demonstrate the performance of our algorithm for stochastic transition modeling, we assess our approach on four problems and a range of baselines.

Table 1: Dufing oscillator: BTJD achieves the lowest reported marginal and trajectory-QoI errors for both random and fixed initial conditions.
<table><tr><td rowspan="2"></td><td colspan="2">Duffing  $( \mathrm { I C } \colon X ( 0 ) \sim \mathcal { N } ( [ 0 , - 1 0 ] ^ { \top } , I _ { 2 } ) )$ </td><td colspan="2">Duffing  $( \mathrm { I C } \colon X ( 0 ) = [ 0 , - 1 0 ] ^ { \top } )$ </td></tr><tr><td> $\mathrm { d i s t . \ e r r . } \ ( W _ { 2 } )$ </td><td> $\operatorname { t r a j . } \operatorname { Q o I } \operatorname { e r r . }$ </td><td> $\mathrm { d i s t . ~ e r r . ~ } ( W _ { 2 } )$ </td><td> $\operatorname { t r a j . } \operatorname { Q o I } \operatorname { e r r . }$ </td></tr><tr><td>T. stepper [36]</td><td> $3 . 4 8 \mathrm { e } { - } 0 1 ( \pm 2 . 0 0 \mathrm { e } { - } 0 1 )$ </td><td> $3 . 6 8 \mathrm { e } { - } 0 2 ( \pm 3 . 3 0 \mathrm { e } { - } 0 4 )$ </td><td></td><td></td></tr><tr><td>DICE [11]</td><td> $5 . 8 0 \mathrm { e } { - } 0 1 \left( \pm 5 . 0 0 \mathrm { e } { - } 0 1 \right)$ </td><td> $8 . 3 0 \mathrm { { e } - 0 2 \mathrm { { ( \pm 5 . 8 0 \mathrm { { e } - 0 4 ) } } } }$ </td><td></td><td></td></tr><tr><td>Marginal diff. [24]</td><td> $8 . 1 0 \mathrm { { e } - 0 2 ( \pm 2 . 5 0 \mathrm { { e } - 0 2 ) } }$ </td><td> $1 . 0 7 \mathrm { e } { - } 0 1 ( \pm 5 . 0 0 \mathrm { e } { - } 0 4 )$ </td><td></td><td></td></tr><tr><td>SDE learning [17]</td><td> $8 . 3 0 \mathrm { { e } - 0 2 ( \pm 2 . 8 0 \mathrm { { e } - 0 2 ) } }$ </td><td> $5 . 7 0 \mathrm { e } { - } 0 3 ( \pm 4 . 5 0 \mathrm { e } { - } 0 4 )$ </td><td> $1 . 0 8 \mathrm { { e } - 0 1 ( \pm 8 . 0 0 \mathrm { { e } - 0 2 ) } }$ </td><td> $6 . 8 0 \mathrm { { e } - 0 3 ( \pm 4 . 5 0 \mathrm { { e } - 0 4 ) } }$ </td></tr><tr><td>SDE matching [6]</td><td> $5 . 4 9 \mathrm { e } { - } 0 1 ( \pm 2 . 0 0 \mathrm { e } { - } 0 1 )$ </td><td> $4 . 4 7 \mathrm { e } { - } 0 2 ( \pm 6 . 2 0 \mathrm { e } { - } 0 4 )$ </td><td> $3 . 9 4 \mathrm { e } { - } 0 1 \dot { ( \pm 2 . 2 0 \mathrm { e } { - } 0 1 ) }$ </td><td> $9 . 0 7 \mathrm { e } - 0 2 ( \pm 5 . 6 0 \mathrm { e } - 0 4 )$ </td></tr><tr><td>BTJD (ours)</td><td> $4 . 8 0 \mathrm { { e - } } { \bf 0 2 } ( \pm 1 . 6 0 \mathrm { { e - } } 0 2 )$ </td><td> $\mathbf { 2 . 7 5 e - 0 3 ( \pm 4 . 2 1 e - 0 4 ) }$ </td><td> $\mathbf { 5 . 8 0 e - 0 2 ( \pm 2 . 2 0 e - 0 2 ) }$ </td><td> $\mathbf { 2 . 4 9 \mathrm { e } - 0 . 3 ( \pm 4 . 2 4 \mathrm { e } - 0 . 4 ) }$ </td></tr></table>

Table 2: Rayleigh-Bénard convection: BTJD achieves the lowest reported marginal and rotational-current errors at the unseen Rayleigh parameter.
<table><tr><td></td><td>dist. err.  $( W _ { 2 } )$ </td><td>traj. QoI err.</td></tr><tr><td>T. stepper [36]</td><td> $2 . 6 9 \mathrm { e } + 0 0 ( \pm 1 . 6 0 \mathrm { e } + 0 0 )$ </td><td> $5 . 8 0 \mathrm { e } { - } 0 3 ( \pm 1 . 4 6 \mathrm { e } { - } 0 3 )$ </td></tr><tr><td>DICE [11]</td><td>2.00e−01(±1.20e−01)</td><td>9.30e−02(±1.46e−03)</td></tr><tr><td>Marginal diffusion [24]</td><td> $1 . 1 1 \mathrm { { e } - 0 1 ( \pm 6 . 1 0 \mathrm { { e } - 0 2 ) } }$ </td><td>6.10e−01(±1.15e−02)</td></tr><tr><td>SDE learning [17]</td><td> $5 . 4 0 \mathrm { e } - 0 2 ( \pm 2 . 3 0 \mathrm { e } - 0 2 )$ </td><td> $2 . 2 0 \mathrm { { e } - 0 2 ( \pm 1 . 6 6 \mathrm { { e } - 0 3 ) } }$ </td></tr><tr><td>SDE matching [6]</td><td> $2 . 2 7 \mathrm { e } { - } 0 1 ( \pm 1 . 3 0 \mathrm { e } { - } 0 1 )$ </td><td> $1 . 9 0 \mathrm { { e } - 0 2 ( \pm 1 . 6 7 \mathrm { { e } - 0 3 ) } }$ </td></tr><tr><td>BTJD (ours)</td><td> $\mathbf { 4 . 6 0 e - 0 2 ( \pm 3 . 1 0 e - 0 2 ) }$ </td><td> $\mathbf { 3 . 0 0 } \mathrm { { e - } } \mathbf { 0 . 4 } ( \pm 1 . 3 6 \mathrm { { e - } } 0 3 )$ </td></tr></table>

## 4.1 Baselines

We compare BTJD with deterministic surrogate models, marginal-matching methods, learned stochastic diferential equations, multi-step conditional generative models, and one- or few-step distilled generative models. Baseline implementations and training setups follow [25].

Deterministic surrogate models. We compare against deterministic surrogate models that learn a single successor state from the current state. We consider a learned deterministic time stepper [36] for the low-dimensional problems and a field-to-field surrogate akin to operator learning [51] for the PDE problems. Because these models return a single successor for a given state, they cannot represent the intrinsic stochastic variability of the dynamics.

Marginal-matching methods. We compare against methods that learn the evolution of the time marginals without identifying the stochastic transition law between consecutive states. These include DICE [11], which learns deterministic population dynamics consistent with the observed marginals, and a difusion-based marginal matching approach [24], which trains a time-conditioned difusion model to generate samples from the marginal distribution $\mu _ { t }$ at each physical time t, without conditioning on the preceding state. Thus, such models can sample from time-marginal distributions at individual times, but the marginals alone do not determine trajectory-dependent statistics.

Learned stochastic diferential equations Stochastic surrogate models based on learned SDEs explicitly represent random state evolution through learned drift and difusion terms. One approach fits these coeficients from consecutive trajectory observations using an Euler– Maruyama transition model [17], while SDE Matching [6] learns a generative SDE using a simulation-free matching objective. In both cases, trajectories are obtained by simulating the learned stochastic dynamics.

Conditional difusion and flow models. Conditional generative models directly target the transition law, but typically require an auxiliary sampling procedure at every physical time step. We consider autoregressive difusion models (ARDM), following [28] and building on denoising difusion probabilistic models [24], which generate each successor through reverse difusion, and conditional flow matching (CFM) [2, 31], which generates successors by integrating a learned conditional flow. Their inference cost therefore grows with the number of denoising or flow-integration steps used per transition.

One- and few-step generative models. One- and few-step generative methods reduce the inference cost of conditional generative models after training. We focus on MeanFlow-based distillation [21] and use it to compress a pretrained conditional flow into an average-velocity model, while ReFlow [32] progressively straightens the generative flow and then compresses it into a one-step sampler. We note that in contrast BTJD directly learns the one-step transition map without first training and compressing a multi-step conditional generator.

## 4.2 Dufing oscillator

We first consider a dufing oscillator with stochastic forcing.

## 4.2.1 Duffing oscillator: Setup and training data

The dufing oscillator that we consider is governed by

$$
\begin{array} { l } { { d X _ { 1 } ( \tau ) = X _ { 2 } ( \tau ) d \tau , } } \\ { { d X _ { 2 } ( \tau ) = ( - 2 \xi \omega X _ { 2 } ( \tau ) + \omega ^ { 2 } X _ { 1 } ( \tau ) - \omega ^ { 2 } \gamma X _ { 1 } ( \tau ) ^ { 3 } ) d \tau + \sigma d W ( \tau ) , } } \end{array}\tag{21}
$$

(22)

where the first equation determines position and the second equation describes the dynamics of the velocity. The variable ξ is a damping parameter, $\gamma$ determines the strength of the cubic term, and ω controls the stifness of the linear dynamics. The strength of the Brownian motion is set by σ. We set these parameters to

$$
\xi = 0 . 2 , \quad \gamma = 0 . 2 , \quad \omega = 1 , \quad \sigma = 0 . 5 .\tag{23}
$$

The SDE (21) is numerically integrated with the Euler-Maruyama scheme with step size $\Delta \tau = 0 . 0 1$ on the time interval $[ 0 , \tau _ { \mathrm { e n d } } ] = [ 0 , 1 2 ]$ , yielding T = 1200 time steps. The training data (1) are generated with initial conditions sampled from $\mathcal { N } ( [ 0 , - 1 0 ] ^ { \top } , I _ { 2 } )$ , where $I _ { 2 } \in \mathbb { R } ^ { 2 \times 2 }$ is the identity matrix. We generate $N = 5 0 0 0$ training trajectories. The block-triangular map is parametrized as an MLP. The MLP is a 2-hidden layer MLP with width 512 per layer and SiLU activations. Time is concatenated as a scalar to the input. The ε in the Sinkhorn loss is set to 0.01, h = 0.1, and 50 Sinkhorn iterations are performed per step. The model trains by sampling $5 1 2 \times 4 = 2 0 4 8$ particles per step, 512 particles from 4 independent timesteps, for 100k gradient steps with AdamW at a learning rate of 1e−3.

## 4.2.2 Duffing oscillator: Test initial conditions

To generate test data, we consider two diferent initial conditions. For the first test data set, we draw initial conditions from $\mathcal { N } ( [ 0 , - 1 0 ] ^ { \top } , I _ { 2 } )$ as well, which is meant to assess the approach’s ability to generalize to unseen initial conditions. For the second test data set, we have a deterministic (fixed) initial condition $X ( 0 ) = [ 0 , - 1 0 ] ^ { \top }$ . Having a deterministic initial condition helps to assess how well the approach generates diferent paths from the same initial condition.

We demonstrate the performance of our approach based on two error measures. First, we compute the sliced Wasserstein-2 distance between 5000 generated samples and 5000 test-set samples at each timestep $t ,$ started from the same initial condition. This yields 1200 sliced Wasserstein-2 distances, one for each time step, which we then average and report, along with the standard deviation of this distance over timesteps. This metric is meant to assess agreement between the true marginal distribution and marginal distribution predicted by our method. Second, we compute an error measure of a trajectory-dependent quantity in order to assess whether the model captures the trajectory-dependent dynamics of the system. We consider the quantity of interest (QoI)

$$
Q _ { \phi } = \mathbb { E } \left[ \int _ { 0 } ^ { \tau _ { \mathrm { e n d } } } \phi ( \tau , X ( \tau ) ) \circ d X ( \tau ) \right]\tag{24}
$$

for the smooth test function

$$
\begin{array} { r } { \phi ( x ) = \left[ \frac { 1 } { \sqrt { 2 \pi } } \exp ( \frac { - x _ { 1 } ^ { 2 } } { 2 } ) \operatorname { t a n h } ( x _ { 2 } ) \right] . } \end{array}\tag{25}
$$

The quantity describes a velocity-weighted, smoothed counting of the crossings over the barrier $x _ { 1 } = 0$ of each sample $X ^ { ( i ) }$ over the time interval. We estimate (24) from samples via

$$
\begin{array} { l } { \displaystyle \hat { q } _ { \phi } ( X ^ { ( i ) } ) = \sum _ { k = 0 } ^ { K - 1 } \phi \left( \frac { t _ { k } + t _ { k + 1 } } { 2 } , \frac { X ^ { ( i ) } ( t _ { k } ) + X ^ { ( i ) } ( t _ { k + 1 } ) } { 2 } \right) ^ { \top } \left( X ^ { ( i ) } ( t _ { k + 1 } ) - X ^ { ( i ) } ( t _ { k } ) \right) , } \\ { \displaystyle \hat { Q } _ { \phi } ( \mathcal D ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { q } _ { \phi } ( X ^ { ( i ) } ) . } \end{array}\tag{26}
$$

Due to the variance of this quantity across trajectories, we evaluate the diference between this quantity in 200,000 generated and test-set trajectories from the same initial condition. We report the mean relative error of (26) obtained with generated trajectories versus groundtruth trajectories as well as its standard error, i.e. for ground truth realizations X and generated trajectories $\hat { X }$

$$
\frac { \Bigl ( \mathrm { V a r } \left[ \hat { q } _ { \phi } ( X ) - \hat { q } _ { \phi } ( \hat { X } ) \right] \Bigr ) ^ { \frac { 1 } { 2 } } } { \sqrt { N } \left| \hat { Q } _ { \phi } ( \mathcal { D } ) \right| } .\tag{27}
$$

## 4.2.3 Duffing oscillator: Results

Figure 1 compares ground-truth trajectories with BTJD rollouts from the same deterministic initial condition. Despite starting from a single state, BTJD generates a diverse ensemble with the qualitative variability of the ground-truth trajectories. This demonstrates that the model captures stochasticity in the transition dynamics rather than relying on variability in the initial condition.

![](images/df051e02e4624cb567f783e0e66cd46ff322bb2e1da50159171cd3d932ba7607.jpg)  
(a) ground truth

![](images/f45a51a9578737f9bac1999f486de5b6a1d6c144814f7a09f514dc202da7b9c1.jpg)  
(b) BTJD (ours)  
Figure 1: Dufing oscillator: From the same deterministic initial state, BTJD generates diverse trajectories that capture well the stochastic variability of the dufing oscillator.

Quantitatively, Table 1 shows that BTJD achieves the lowest sliced- $W _ { 2 }$ error among the tested methods for both random and deterministic test initial conditions. Thus, the improved agreement of the time marginals persists both for unseen initial conditions drawn from the training distribution and when all trajectories start from the same state. The phase-space snapshots in Figure 2 further show agreement between the generated and ground-truth marginal distributions throughout the time integration. BTJD also achieves the lowest error in the trajectory-dependent QoI (24) for both test initial condition distributions; see Table 1. Hence, the improved performance achieved by BTJD is not limited to matching time marginals; BTJD also predicts statistics that depend on the temporal evolution of individual trajectories.

## 4.3 Rayleigh-Bénard convection

We now consider the 9-dimensional Rayleigh-Bénard convection model [40] with an additive stochastic forcing.

## 4.3.1 Rayleigh-Bénard: Setup

The process

$$
X ( \tau ) = ( X _ { 1 } ( \tau ) , X _ { 2 } ( \tau ) , \ldots , X _ { 9 } ( \tau ) ) \in \mathbb { R } ^ { 9 }
$$

![](images/64a2d0a69c27384e963121a6db6ec6922205b44f69526110ed286baacc97ac41.jpg)  
(a) $\tau = 3$

![](images/d36456129de4f8b9563845abfc0f5d6cabb34bd8fcd1ca8d07d36c2f1376ff12.jpg)  
(b) $\tau = 6$

![](images/c9350f45bad1962e8597a8292f23a4c5c0b99760949b7d70103aa22efdcc6f20.jpg)  
(c) τ = 9

![](images/bb425e19de295344b903b1310706100d6a4762a9363bfb8da75054a9c72d49e1.jpg)  
(d) $\tau = 1 2$  
Figure 2: Dufing oscillator: BTJD tracks the evolving phase-space distribution throughout the rollout; ground-truth particles (orange) and BTJD (ours) particles (blue) at four rollout times.

is governed by

$$
d X ( \tau ) = f _ { \mu } ( X ( \tau ) ) d \tau + \sigma d W ( \tau )\tag{28}
$$

where $f _ { \mu }$ is the deterministic vector field of [40], determined by a Rayleigh-type parameter µ. We set $\sigma = 0 . 0 5$ . For training initial conditions, we draw 20000 samples from $\mathcal { N } ( 0 , 0 . 0 0 0 4 \cdot I _ { 9 } )$ and generate the corresponding trajectories via Euler-Maruyama with $\Delta \tau = 0 . 0 1$ on the interval [0, 20] for each control parameter $\mu \in \{ 1 3 . 5 , 1 3 . 6 , 1 3 . 7 , . . . , 1 4 . 2 \}$

We generate a test data set by drawing samples from the same distribution of initial conditions. For the test data set, the Rayleigh control parameter $\mu$ is set to $\mu = 1 3 . 6 5$ to evaluate the model’s ability to generalize to unseen parameters. Analogous to Section 4.2.2, we compute sliced Wasserstein-2 distance to assess the time marginals, reporting the mean distance and its standard error across timesteps. Additionally, we compute the relative absolute errors of estimating the quantity (24) but with test function

$$
\phi ( x ) = Q ^ { \top } R Q x , \quad Q x = ( x _ { 6 } , x _ { 7 } ) , \quad R = { \left[ \begin{array} { l l } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} \right] } ~ .\tag{29}
$$

The quantity measures the portion of probability mass motion which is aligned with counterclockwise rotation in the $( x _ { 6 } , x _ { 7 } )$ plane. The system has a probability current with persistent rotation in low-dimensional projections, making this a well-defined quantity to assess trajectory-dependent dynamics. As in Section 4.2.2, we report the mean relative error across trajectories as well as the standard error over trajectories given by (27).

We train a 3-layer MLP with width of 512 neurons per layer. Time is embedded as a sinusoidal embedding with 50 frequencies and then concatenated to the input, along with $\mu .$ For this experiment, we train with $\varepsilon = 0 . 1$ ， $h = 0 . 1$ , 20 Sinkhorn iterations per step. Each gradient step samples $1 0 2 4 \times 8 = 8 1 9 2$ particles, 1024 particles per timestep and control parameter, with 4 timesteps sampled for 2 control parameters in each gradient step. We train for 100k gradient steps using AdamW at a learning rate of 1e−3.

## 4.3.2 Rayleigh-Bénard: Results

We evaluate the BTJD model at the unseen control parameter $\mu = 1 3 . 6 5$ , which lies between parameter values used during training. We plot in Figure 3 histograms of the marginals corresponding to dimension 1, 2, 4, 8, and 9. Our BTJD is accurately approximating the ground-truth marginal distribution. As reported in Table 2, BTJD achieves the lowest ${ \mathrm { s l i c e d } } { \mathrm { - } } W _ { 2 }$ error among all methods. The accuracy of the predicted time marginals therefore extends to interpolation in the Rayleigh control parameter. BTJD also achieves the lowest error in the rotational-current QoI (29); see Table 2. This quantity depends on the direction of probability-mass motion in the $( x _ { 6 } , x _ { 7 } )$ plane, so the result shows that BTJD captures trajectory-dependent dynamics that are not determined by the time marginals alone.

## 4.4 Stochastically forced Burgers equation

The preceding experiments establish transition-law and trajectory-level fidelity in systems where these quantities can be diagnosed directly. We now ask whether BTJD retains its advantage for high-dimensional stochastic dynamics stemming from stochastic PDEs, where inference costs become important. We first consider the Burgers equation with stochastic forcing.

## 4.4.1 Burgers: Setup and training data

We first consider the one-dimensional Burgers equation with periodic boundary conditions on the spatial domain [0, 1),

$$
\partial _ { \tau } u ( \tau , x ) = \nu \partial _ { x x } u ( \tau , x ) - u ( \tau , x ) \partial _ { x } u ( \tau , x )\tag{30}
$$

with viscosity parameter ${ \nu = 0 . 0 0 7 }$ controlling the width of shocks. We then discretize (30) in space on a uniformly spaced 64-point grid. This yields an ODE in $\mathbb { R } ^ { 6 4 }$ , to which we add stochastic forcing. The forcing considered is discrete-in-space and not a standard Brownian motion, but colored noise. Specifically, for grid location $x _ { j }$

$$
d W _ { j } ( \tau ) = \sigma \sum _ { \iota = 1 } ^ { 1 0 } \frac { 1 } { \iota } ( a _ { \iota } ( x _ { j } ) d B _ { \iota } ^ { ( 1 ) } ( \tau ) + b _ { \iota } ( x _ { j } ) d B _ { \iota } ^ { ( 2 ) } ( \tau ) )\tag{31}
$$

where each $B _ { \iota } ^ { ( 1 ) } ( \tau ) , B _ { \iota } ^ { ( 2 ) } ( \tau )$ represent independent Brownian motions and $a _ { \iota } , b _ { \iota }$ represent sine and cosine modes. The noise amplitude is set to $\sigma = 0 . 0 4$

Training initial conditions are drawn from Gaussian bumps with noise of the same structure as (31),

$$
u ( 0 , x ) = \exp ( - 2 0 ( x - 0 . 5 ) ^ { 2 } ) + 0 . 0 1 5 W ( 0 , x ) .\tag{32}
$$

We solve the forced (30) by evolving from initial conditions (32) using a method-of-lines discretization. The solutions are computed on a uniformly spaced 64-point grid. The spatial derivatives are approximated by second-order centered finite diferences. The solutions are then evolved in time via Euler-Maruyama with $\Delta \tau = 5 \times 1 0 ^ { - 4 }$ , on the interval [0, 4] yielding

![](images/6df93616823469e6ab5816aed114e594358575b2acbbfa1fe1cdec8f5b2fcaf0.jpg)  
Figure 3: Rayleigh–Bénard convection: marginal histograms of selected state-vector components $X _ { d } ( \tau )$ at five rollout times. BTJD (ours, blue fill) closely tracks the ground-truth distribution (orange outline).

8000 step trajectories. The solution is then downsampled to 800 steps. We generate N = 4096 training trajectories of this form.

We train a generator operating in a latent space. As such, a convolutional autoencoder is trained compressing the 64-dimensional field into a 16-dimensional state vector. The generator is then a convolutional neural network (CNN) of 3 FiLM-conditioned residual blocks [38] of channel width 32 with circular padding and SiLU activations. Time is injected through a learned embedding, and noise is concatenated channel-wise with the latent state. Additionally, we inject noise into the conditioning with a magnitude of $1 \%$ of the standard deviation of the latent space feature magnitude. For this experiment, we train with $\varepsilon = 0 . 1 , h = 0 . 1$ , and 20 Sinkhorn iterations per step. Each gradient step 1024 particles per timestep are drawn for four randomly chosen timesteps. We train for 100k gradient steps using AdamW with learning rate 1e−3 and cosine schedule.

## 4.4.2 Burgers: Test initial conditions and evaluation metrics

For test initial conditions, we draw an additional 4096 test trajectories of the form (32). We evaluate two quantities of interest on the stochastic Burgers trajectories, the energy $E ( \tau )$ and enstrophy $Z ( \tau )$ at each time point,

$$
E ( \tau ) = \frac 1 2 \int _ { 0 } ^ { 1 } | u ( \tau , x ) | ^ { 2 } d x , \qquad Z ( \tau ) = \frac 1 2 \int _ { 0 } ^ { 1 } | \partial _ { x } u ( \tau , x ) | ^ { 2 } d x .\tag{33}
$$

Both integrals are approximated using the trapezoidal rule, with $\partial _ { x } u ( \tau , x )$ approximated using centered second-order finite diferences on the 64-point grid. We assess generated samples via the relative absolute errors of the energy and enstrophy averaged over all time steps, and also report the standard error of the mean relative error across time steps.

## 4.4.3 Burgers: Results

Figure 4 compares BTJD samples with ground-truth solutions of the stochastic Burgers equation at several time steps. Despite using a single model evaluation for each physical time step, BTJD accurately predicts the stochastic variability of the solution ensemble together with the sharp spatial structures generated by the nonlinear dynamics. The quantitative results in Table 3 further support that BTJD generates accurate trajectories. BTJD achieves the lowest reported mean errors in both energy and enstrophy while requiring only one neuralnetwork function evaluation (NFE) per physical time step, compared with 20 evaluations for conditional flow matching (CFM) and 50–100 denoising steps for the autoregressive difusion models (ARDM). Its enstrophy error is less than half that of the next-best tested method.

## 4.5 Stochastically forced two-dimensional turbulence

We now consider a stochastically forced two-dimensional incompressible flow, providing a high-dimensional problem with chaotic multiscale dynamics.

## 4.5.1 Turbulence: Setup and training data

We consider the two-dimensional incompressible Navier-Stokes equations on the periodic domain $\Omega = [ 0 , 2 \pi ) ^ { 2 }$ , adapting the setup of [27]. In vorticity form, we have the equation,

$$
\partial _ { \tau } \omega + \left( u \cdot \nabla \omega \right) = \left( \nu \Delta \omega - \alpha \omega \right) , \qquad u = \nabla ^ { \perp } \Delta ^ { - 1 } \omega ,\tag{34}
$$

where $\nu = 1 0 ^ { - 3 }$ is the viscosity and $\alpha = 0 . 1$ is a linear drag coeficient. We discretize on a uniform grid, yielding an ODE, to which we introduce stochastic forcing through

![](images/7a00ae62b6d94ba666a2fda7dc2fa90054cf5a6143902f7b31f6fea446dfc458.jpg)  
(a) τ = 0.5

![](images/89ecad8d85e389c54256cfdc6ffeefabc1565a7d21078185c4ca99c79f6286d8.jpg)  
(b) $\tau = 1 . 5$

![](images/cd61c5ef0de87202c8d2c77d8fad9a787520cf201944cb28705a9dae07c6fbe0.jpg)  
(c) τ = 2.5

![](images/4645ad26ce114c06ee7373ddcc16b00cca24b7c2cf011e803627d23d81e3bba9.jpg)  
(d) $\tau = 3 . 5$  
Figure 4: Stochastic Burgers: BTJD generates trajectories that capture both stochastic variability and sharp spatial structures over time; ground-truth samples (vermillion) and BTJD (ours) samples (blue).

Table 3: Stochastic Burgers: With one model evaluation per time step, BTJD achieves the lowest energy and enstrophy errors, with more than a twofold reduction in enstrophy error over the next-best method.
<table><tr><td>Method</td><td>error energy</td><td>error enstrophy</td><td>NFEs/time step</td></tr><tr><td>Operator learning [51]</td><td>2.21e−2 (± 1.54e−2)</td><td>2.55e−1 (± 2.34e−1)</td><td>1</td></tr><tr><td>ARDM 50 steps[28]</td><td>1.46e−2 (± 3.39e−3)</td><td>2.49e−1 (± 1.43e−1)</td><td>50</td></tr><tr><td>ARDM 75 steps [28]</td><td>1.36e−2 (± 3.49e−3)</td><td>2.30e−1 (± 1.26e−1)</td><td>75</td></tr><tr><td>ARDM 100 steps [28]</td><td>1.24e−2 (± 3.12e−3)</td><td>2.11e−1 (± 1.14e−1)</td><td>100</td></tr><tr><td>CFM 20 steps [2, 31]</td><td>2.71e−3 (± 1.96e−3)</td><td>1.53e−1 (± 1.28e−1)</td><td>20</td></tr><tr><td>MeanFlow 1 step[21]</td><td>8.22e−1 (± 2.88e−1)</td><td>3.59e+2 (± 2.78e+2)</td><td>1</td></tr><tr><td>MeanFlow 2 steps [21]</td><td>1.69e−1 (± 5.77e−2)</td><td>9.81e+1 (±8.05e+1)</td><td>2</td></tr><tr><td>ReFlow+Distill 1 step[32]</td><td>2.78e−3 (± 2.10e−3)</td><td>1.40e−1 (± 1.31e−1)</td><td>1</td></tr><tr><td>BTJD (ours)</td><td>2.22e−3 (± 2.47e−3)</td><td>5.87e−2 (± 1.01e−1)</td><td>1</td></tr></table>

low-frequency Fourier modes, at location $x _ { j }$ , for ${ \mathcal { K } } _ { \mathrm { s t o } } = \{ k \in \mathbb { Z } ^ { 2 } : 0 < | k | \leq 4 \}$

$$
d W _ { j } ( \tau ) = \sigma \sum _ { \kappa \in \mathcal { K } _ { \mathrm { s t o } } } w _ { \kappa } \left( a _ { \kappa } ( x _ { j } ) d B _ { \kappa } ^ { ( 1 ) } ( \tau ) + b _ { \kappa } ( x _ { j } ) d B _ { \kappa } ^ { ( 2 ) } ( \tau ) \right) ,\tag{35}
$$

where $a _ { \kappa } = \cos ( \kappa \cdot x )$ and $b _ { \kappa } = \sin ( \kappa \cdot x )$ are sine and cosine modes, $w _ { \kappa } \propto | \kappa | ^ { - \frac { 1 } { 2 } }$ are the spectral weights, and the Brownian motions are mutually independent. We set $\sigma = 0 . 3$ and add this forcing to each grid point at each step of the integration.

Initial conditions are sampled from the ensemble used in [27] and rescaled so that the maximum vorticity is approximately seven as in [27]. We solve (34) on a $2 5 6 \times 2 5 6$ grid using a pseudo-spectral method with $2 / 3$ -rule dealiasing. The linear terms are integrated with Crank-Nicolson and the nonlinear term with a fourth-order Runge-Kutta scheme. We generate $N = 2 0 4 8$ training trajectories using $\Delta \tau = 0 . 0 0 1$ and 25,000 fine time steps. Each trajectory is temporally subsampled to 250 states and spectrally subsampled to a $6 4 \times 6 4$ grid for training.

![](images/5f0b739c3d18e4e596c1e29804c0075aa143ccb519c488eb3736341a03551efb.jpg)  
Figure 5: Turbulence: Our BTJD generates statistically representative turbulent trajectories with only one neural-network function evaluation per time step.

For generation on high dimensional data, we follow the latent space embedding and Masked Autoencoder (MAE) feature extraction of [16, 23]. Both the MAE and latent space feature extractor are autoencoders, and the generator is a 2D U-Net with channel widths of 256 and 512. Time is injected via a learned embedding and noise is concatenated channel-wise. Additionally, the conditioning input is perturbed by Gaussian noise of 2% magnitude of the standard deviation of the latent space magnitude as in 4.4.1. We train with $\varepsilon = 0 . 0 7$ , which is then scaled by the dimension of the feature as well as the mean inter-particle distance as in [16]. We set h = 1 and use 20 Sinkhorn iterations per-step. Each gradient step samples 96 particles per timestep for 64 sampled timesteps. We train for 65, 000 gradient steps of AdamW with learning rate 1e−4 and 5000-step warmup.

## 4.5.2 Turbulence: Test initial conditions and evaluation metrics

We generate 1024 additional test trajectories from independently sampled initial conditions following the same construction as the training data. We assess the predicted dynamics using the kinetic energy and enstrophy,

$$
E ( \tau ) = \frac 1 2 \int _ { \Omega } | u ( \tau , x ) | ^ { 2 } d x , \qquad Z ( \tau ) = \frac 1 2 \int _ { \Omega } | \omega ( \tau , x ) | ^ { 2 } d x .\tag{36}
$$

These quantities characterize the evolution of the kinetic energy and the strength of the vortical structures, respectively. We report their mean relative errors over all timesteps along with the standard error of this quantity over timesteps as in Section 4.4.2.

## 4.5.3 Turbulence: Results

Figures 5 and 6 show that BTJD generates accurate individual turbulent trajectories as well as diverse ensembles of stochastic realizations. Thus, the direct one-step sampler remains expressive enough to represent the variability and multiscale structures of the stochastically forced flow dynamics. The quantitative comparison in Table 4 highlights that BTJD achieves the lowest reported mean errors in both kinetic energy and enstrophy, while requiring only a single neural-network function evaluation (NFE) per time step. Relative to the next-best reported mean errors, BTJD reduces the energy error by approximately a factor of four and the enstrophy error by approximately a factor of three. It therefore outperforms both multi-step difusion and flow baselines and one-step distilled models.

Table 4: Turbulence: With one neural-network function evaluation (NFE) per time step, BTJD reduces energy error by about a factor four and enstrophy error by almost a factor of three over the next-best baseline.
<table><tr><td>Method</td><td>error energy</td><td>error enstrophy</td><td>NFEs/time step</td></tr><tr><td>Operator learning [51]</td><td>2.13e−1(±1.58e−1)</td><td>1.92e−1(±1.48e−1)</td><td>1</td></tr><tr><td>ARDM 50 steps [28]</td><td>3.14e−1(±1.48e−1)</td><td>2.82e−1(±9.60e−2)</td><td>50</td></tr><tr><td>ARDM 75 steps [28]</td><td>2.75e−1(±1.40e−1)</td><td>1.71e−1(±9.40e−2)</td><td>75</td></tr><tr><td>ARDM 100 steps [28]</td><td>1.16e−1(±1.09e−1)</td><td>1.34e−1(±8.46e−2)</td><td>100</td></tr><tr><td>CFM 20 steps [2, 31]</td><td>1.39e−1(±6.30e−2)</td><td>1.04e−1(±7.70e−2)</td><td>20</td></tr><tr><td>MeanFlow 1 step [21]</td><td>2.46e−1(±1.25e−1)</td><td>4.37e−1(±2.78e−1)</td><td>1</td></tr><tr><td>MeanFlow 2 steps [21]</td><td>1.40e−1(±6.61e−2)</td><td>9.81e−2(±2.97e−2)</td><td>2</td></tr><tr><td>MeanFlow 4 steps [21]</td><td>1.22e−1(±8.45e−2)</td><td>6.74e−2(±3.66e−2)</td><td>4</td></tr><tr><td>ReFlow+Distill 1 step [32]</td><td>8.60e−2(±4.41e−2)</td><td>7.16e−2(±6.36e−2)</td><td>1</td></tr><tr><td>BTJD (ours)</td><td>2.14e−2(±1.93e−2)</td><td>2.52e−2(±1.91e−2)</td><td>1</td></tr></table>

## 5 Conclusions

We developed BTJD for learning one-step generative surrogate models of stochastic dynamics from trajectory data. By drifting the joint law of consecutive states while preserving the current-state marginal, the method turns available transition pairs into a direct conditional sampler and enables stochastic rollouts with one model evaluation per time step. The underlying idea extends beyond the particular drifting scheme considered here. More generally, for generative procedures that require the target distribution to enter the learning update through an empirical distribution given by samples, lifting the learning problem to the joint space can make the target distribution accessible from trajectory data, provided that the model is endowed with suficient structure to expose the desired conditional after training. The role of the block-triangular parametrization in BTJD is precisely to provide this structure.

## References

[1] M. S. Albergo, N. M. Bofi, and E. Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and difusions. Journal of Machine Learning Research, 26(209):1–80, 2025.

![](images/d08605ba4b71b5335a0e119710e4538e074b77af6fc871cb407c2e9f3f2ae32a.jpg)  
step 0 fixed IC  
step 10 ¿ = 1:0  
step 20 ¿= 2:0  
step 30 ¿= 3:0  
step 40 ¿ = 4:0  
step 50 ¿= 5:0  
Figure 6: Turbulence: BTJD captures the intrinsic stochasticity of the dynamics. Starting from the same initial condition, repeated BTJD rollouts separate and produce distinct turbulent realizations, as seen in the three generated trajectories. In contrast, a deterministic surrogate such as given by operator learning returns the same trajectory from the same initial condition and therefore cannot represent this stochastic variability.

[2] M. S. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.

[3] L. Ambrosio and N. Gigli. A user’s guide to optimal transport. In Modelling and Optimisation of Flows on Networks: Cetraro, Italy 2009, volume 2062 of Lecture Notes in Mathematics, pages 1–155. Springer, 2013.

[4] R. Baptista, B. Hosseini, N. B. Kovachki, and Y. M. Marzouk. Conditional sampling with monotone GANs: From generative models to likelihood-free inference. SIAM/ASA Journal on Uncertainty Quantification, 12(3):868–900, 2024.

[5] R. Baptista, Y. M. Marzouk, and O. Zahm. On the representation and learning of monotone triangular transport maps. Foundations of Computational Mathematics, 24(6):2063–2108, 2024.

[6] G. Bartosh, D. Vetrov, and C. A. Naesseth. SDE matching: Scalable and simulationfree training of latent stochastic diferential equations. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 3054–3070. PMLR, 2025.

[7] P. Batlle, M. Darcy, B. Hosseini, and H. Owhadi. Kernel methods are competitive for operator learning. Journal of Computational Physics, 496:112549, 2024.

[8] J. Berman, T. Blickhan, and B. Peherstorfer. Parametric model reduction of meanfield and stochastic systems via higher-order action matching. In Advances in Neural Information Processing Systems, volume 37, pages 56588–56618, 2024.

[9] J. Berman, T. Blickhan, and B. Peherstorfer. Leveraging gauge freedom for learning non-gradient population dynamics of stochastic systems. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. PMLR, 2026.

[10] J. Berman, T. Blickhan, and B. Peherstorfer. Stochastic lifting for generating trajectories of stochastic physical systems. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. PMLR, 2026.

[11] T. Blickhan, J. Berman, A. M. Stuart, and B. Peherstorfer. DICE: Discrete inverse continuity equation for learning population dynamics, 2025.

[12] N. M. Bofi, M. S. Albergo, and E. Vanden-Eijnden. How to build a consistency model: Learning flow maps via self-distillation. In Advances in Neural Information Processing Systems, volume 38, 2025.

[13] S. R. Cachay, B. Zhao, H. Joren, and R. Yu. DYfusion: A dynamics-informed difusion model for spatiotemporal forecasting. In Advances in Neural Information Processing Systems, volume 36, 2023.

[14] Y. Chen, M. Goldstein, M. Hua, M. S. Albergo, N. M. Bofi, and E. Vanden-Eijnden. Probabilistic forecasting with stochastic interpolants and Föllmer processes. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 6728–6756. PMLR, 2024.

[15] P. Conti, J. Kneifl, A. Manzoni, A. Frangi, J. Fehr, S. L. Brunton, and J. N. Kutz. VENI, VINDy, VICI: A generative reduced-order modeling framework with uncertainty quantification. Neural Networks, 198:108543, 2026.

[16] M. Deng, H. Li, T. Li, Y. Du, and K. He. Generative modeling via drifting, 2026.

[17] N. Dridi, L. Drumetz, and R. Fablet. Learning stochastic dynamical systems with neural networks mimicking the Euler–Maruyama scheme. In 2021 29th European Signal Processing Conference (EUSIPCO), pages 1990–1994. IEEE, 2021.

[18] J. Feydy, T. Séjourné, F.-X. Vialard, S.-i. Amari, A. Trouvé, and G. Peyré. Interpolating between optimal transport and MMD using sinkhorn divergences. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 2681–2690. PMLR, 2019.

[19] M. A. Freitag, J. M. Nicolaus, and M. Redmann. Learning stochastic reduced models from data: A nonintrusive approach. SIAM Journal on Scientific Computing, 47(5):A2851– A2880, 2025.

[20] S. Fresca, L. Dede’, and A. Manzoni. A comprehensive deep learning-based approach to reduced order modeling of nonlinear time-dependent parametrized pdes. Journal of Scientific Computing, 87(2):61, Apr 2021.

[21] Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He. Mean flows for one-step generative modeling. In Advances in Neural Information Processing Systems, volume 38, 2025.

[22] M. Gloeckler, M. Deistler, C. D. Weilbach, F. Wood, and J. H. Macke. All-in-one simulation-based inference. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 15735–15766. PMLR, 2024.

[23] J. Han, P. Li, Q. Guo, R. Xu, S. Ermon, and E. J. Candès. One-step generative modeling via wasserstein gradient flows, 2026.

[24] J. Ho, A. N. Jain, and P. Abbeel. Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, 2020.

[25] S. Jha, T. Schorlepp, N. Geissler, J. Berman, and B. Peherstorfer. First-order trajectory matching: Fast ensemble predictions of chaotic, turbulent, stochastic systems. arXiv, 2606.11138, 2026.

[26] G. Kerrigan, G. Migliorini, and P. Smyth. Dynamic conditional optimal transport through simulation-free flows. In Advances in Neural Information Processing Systems, volume 37, pages 93602–93642, 2024.

[27] D. Kochkov, J. A. Smith, A. Alieva, Q. Wang, M. P. Brenner, and S. Hoyer. Machine learning–accelerated computational fluid dynamics. Proceedings of the National Academy of Sciences, 118(21):e2101784118, 2021.

[28] G. Kohl, L.-W. Chen, and N. Thuerey. Benchmarking autoregressive conditional difusion models for turbulent flow simulation. Neural Networks, 199:108641, 2026.

[29] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, and A. Anandkumar. Neural operator: Learning maps between function spaces with applications to pdes. Journal of Machine Learning Research, 24(89):1–97, 2023.

[30] C.-H. Lai, B. Nguyen, N. Murata, Y. Takida, T. Uesaka, Y. Mitsufuji, S. Ermon, and M. Tao. A unified view of drifting and score-based models, 2026.

[31] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[32] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

[33] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, Mar 2021.

[34] Y. Marzouk, T. Moselhy, M. Parno, and A. Spantini. Sampling via measure transport: An introduction. In R. Ghanem, D. Higdon, and H. Owhadi, editors, Handbook of Uncertainty Quantification, pages 785–825. Springer, 2017.

[35] K. Neklyudov, R. Brekelmans, D. Severo, and A. Makhzani. Action matching: Learning stochastic dynamics from samples. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 25858–25889. PMLR, 2023.

[36] K. Otness, A. Gjoka, J. Bruna, D. Panozzo, B. Peherstorfer, T. Schneider, and D. Zorin. An extensible benchmark suite for learning to simulate physical systems. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2021.

[37] B. Peherstorfer and K. Willcox. Data-driven operator inference for nonintrusive projection-based model reduction. Computer Methods in Applied Mechanics and Engineering, 306:196–215, 2016.

[38] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), volume 32, pages 3942–3951, 2018.

[39] F. Regazzoni, S. Pagani, M. Salvador, L. Dede’, and A. Quarteroni. Learning the intrinsic dynamics of spatio-temporal processes through latent dynamics networks. Nature Communications, 15(1):1834, Feb 2024.

[40] P. Reiterer, C. Lainscsek, F. Schürrer, C. Letellier, and J. Maquet. A nine-dimensional lorenz system to study high-dimensional chaos. Journal of Physics A: Mathematical and General, 31(34):7121–7139, 1998.

[41] T. Salimans and J. Ho. Progressive distillation for fast sampling of difusion models. In The Tenth International Conference on Learning Representations, 2022.

[42] F. Santambrogio. Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling, volume 87 of Progress in Nonlinear Diferential Equations and Their Applications. Birkhäuser, 2015.

[43] P. Schwerdtner, T. Blickhan, and B. Peherstorfer. Two-parameter flows for learning population dynamics of physical systems. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. PMLR, 2026.

[44] Y. Shi, V. De Bortoli, G. Deligiannidis, and A. Doucet. Conditional simulation using difusion Schrödinger bridges. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pages 1792–1802. PMLR, 2022.

[45] A. Shysheya, C. Diaconu, F. Bergamin, P. Perdikaris, J. M. Hernández-Lobato, R. E. Turner, and E. Mathieu. On conditional difusion models for PDE simulations. In Advances in Neural Information Processing Systems, volume 37, pages 23246–23300, 2024.

[46] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265. PMLR, 2015.

[47] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 32211–32252. PMLR, 2023.

[48] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Scorebased generative modeling through stochastic diferential equations. In The Ninth International Conference on Learning Representations, 2021.

[49] A. Spantini, R. Baptista, and Y. M. Marzouk. Coupling techniques for nonlinear ensemble filtering. SIAM Review, 64(4):921–953, 2022.

[50] A. Spantini, D. Bigoni, and Y. M. Marzouk. Inference via low-dimensional couplings. Journal of Machine Learning Research, 19(66):1–71, 2018.

[51] K. Stachenfeld, D. B. Fielding, D. Kochkov, M. Cranmer, T. Pfaf, J. Godwin, C. Cui, S. Ho, P. Battaglia, and Á. Sánchez-González. Learned coarse models for eficient turbulence simulation. In The Tenth International Conference on Learning Representations, 2022.

[52] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, pages 1–34, 2024.

[53] E. Turan and M. Ovsjanikov. Generative drifting is secretly score matching: A spectral and variational perspective, 2026.

[54] P. Wyrod, A. Chattopadhyay, and D. Venturi. Generative forecasting with joint probability models. Journal of Computational Physics, 563:115109, 2026.

[55] P. S. Zhai, S. Jeong, and V. Ročková. Conditional flow matching for bayesian posterior inference. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 2044–2052. PMLR, 2026.

[56] L. Zhou, S. Ermon, and J. Song. Inductive moment matching. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 78651–78686. PMLR, 2025.