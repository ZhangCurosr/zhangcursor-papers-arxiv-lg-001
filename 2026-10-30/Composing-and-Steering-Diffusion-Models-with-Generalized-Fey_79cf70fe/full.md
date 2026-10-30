# Composing and Steering Diffusion Models with Generalized Feynman-Kac Correctors

Wael Mattar ∗

D<sub>an</sub>i<sub>e</sub>l F<sub>ree</sub>d<sub>man</sub> ∗

## Abstract

Difusion models have become the leading paradigm in generative modeling, yet many applications require sampling from distributions that are not those of any single pretrained model, but are instead defined through combinations of several models, possibly tilted toward high values of a downstream reward. Naively combining the scores of the models yields incorrect samples, while existing corrections are restricted to specific families of targets. In this work, we introduce the generalized Feynman–Kac correctors (GFKC), a principled framework for sampling from an arbitrary target function of several pretrained difusion models. We show that the geometric interpolation between an arbitrary initial function and the target function of the models satisfies a Feynman–Kac partial diferential equation, and derive its potential in closed form. To evaluate this potential during generation, we further generalize the log-probability ODE to track the density of each model along trajectories driven by a diferent velocity field. The resulting algorithm operates entirely in the log domain, incorporates reward-based steering at no additional conceptual cost, and corrects the generation on the fly, without retraining or fine-tuning the models. We validate the method on analytic Gaussian models, where it closely matches the ground-truth targets, and on difusion models trained on MNIST, where it samples from non-trivial combinations of the models and enables efective steering.

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Difusion models [1, 2, 3] have become the leading paradigm in generative modeling. By learning to reverse a gradual noising process, they produce samples of remarkable quality and diversity, and they are now the method of choice across a wide range of domains, including image synthesis [4, 5, 6, 7], audio [8] and video generation [9], inverse problems [10], and the design of molecules and proteins [11, 12, 13]. Beyond their empirical success, difusion models rest on a solid mathematical foundation: in the continuous-time formulation of [3], generation amounts to simulating a stochastic or ordinary diferential equation driven by a learned score, whose marginals evolve according to an explicit partial diferential equation. This structure makes it possible to reason about, and to modify, the generative process in a principled way.

In many applications, however, the distribution of interest is not that of a single pretrained model. One may wish to combine several models, each trained on diferent data or capturing a diferent concept, and to sample from a distribution that is defined in terms of all of them: their product, which retains only what all models agree on; their mixture; or more elaborate combinations, such as distributions favoring the regions where one model assigns high probability and another does not. Training a new model for every such target is costly and often impractical, and a natural alternative is to reuse the pretrained models at inference time [14]. Yet, naive approaches that simply combine the scores of the models do not reproduce the correct intermediate distributions, and therefore yield biased samples [15]. Recently, [16] showed that this bias can be corrected exactly by means of the Feynman–Kac formula [17]: the generated particles are assigned weights that evolve along the generation, and are periodically resampled within a sequential Monte Carlo scheme [18]. These Feynman–Kac correctors, however, are derived separately for specific families of targets, namely annealed distributions, products of experts, and the geometric averages underlying classifier-free guidance [19].

A complementary need arises when the generated samples are required not only to be realistic, but also to optimize a downstream objective, such as the stability or binding afinity of a molecule, or the similarity of an image to a prescribed reference. This is the task of steering, or reward-guided generation [20]. A principled formulation consists in maximizing the expected reward, regularized by the Kullback–Leibler divergence to the pretrained distribution, whose solution is the pretrained distribution exponentially tilted by the reward. One approach is to fine-tune the pretrained model toward this tilted distribution [21, 22], which, however, requires additional training for every reward. Inference-time methods, such as classifier guidance [4] and SMC-based guidance [23, 24], instead approximate the tilted distribution during generation, without modifying the model, but are typically formulated for a single pretrained model.

In this paper, we unify and generalize these two lines of work. We introduce the generalized Feynman–Kac correctors (GFKC), which enable sampling from an arbitrary target function of several pretrained difusion models, with an optional reward-based tilt. Our main result, Theorem 1, shows that the geometric interpolation between an arbitrary initial function and an arbitrary terminal function of the models satisfies a Feynman–Kac PDE, and provides its potential in closed form. To evaluate this potential along the generation, we derive a generalization of the logprobability ODE [25, 26], which tracks the density of every model along trajectories generated by a diferent velocity field. The resulting algorithm operates entirely in the log domain, performs the correction on the fly during the generation process, and requires neither retraining nor fine-tuning of the models. The framework of [16] is recovered as a special case.

The remainder of the paper is organized as follows. Section 2 reviews continuous-time difusion models, the Feynman– Kac correctors, and reward-guided tilting. Section 3 presents our main result, the generalized log-probability ODE, the incorporation of steering, and the resulting GFKC algorithm. Section 4 illustrates the method in two settings of diferent nature: an analytic setting of overlapping Gaussian distributions, where the generated samples are compared with exact ground-truth densities, and a high-dimensional setting of difusion models trained on the MNIST dataset [27], where we demonstrate both the sampling from non-trivial combinations of the models and the efec of steering. All numerical experiments are fully reproducible, and the code, together with the pretrained models, is publicly available at https://github.com/WaelMattar/GFKC.

## 2 P<sub>re</sub>li<sub>m</sub>i<sub>nar</sub>i<sub>es</sub>

We review continuous-time difusion models (Section 2.1), the Feynman–Kac correctors (Section 2.2), and rewardguided tilting (Section 2.3), which together form the basis of our method.

## 2<sub>.</sub>1 C<sub>o</sub>ntin<sub>uous</sub>-tim<sub>e</sub> dif<sub>us</sub>i<sub>o</sub>n m<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

Difusion models [1, 2, 3] generate data by learning to reverse a gradual noising process. In the continuous-time formulation of [3], the noising process is described by a forward stochastic diferential equation (SDE)

$$
\mathrm { d } x _ { u } = f ( x _ { u } , u ) \mathrm { d } u + \sigma _ { \mathrm { { d i f f } } } ( u ) \mathrm { d } W _ { u } , \qquad u \in [ 0 , 1 ] , \qquad x _ { 0 } \sim p _ { \mathrm { d a t a } } ,\tag{1}
$$

where $f$ is a drift, $\sigma _ { \mathrm { d i f f } }$ a scalar difusion coeficient and $W _ { u }$ a standard Brownian motion in $\mathbb { R } ^ { d }$ . The coeficients are chosen so that the marginal distribution $q _ { u }$ of $x _ { u }$ approaches a tractable prior as $u \to 1$ . A widely used choice, which we adopt in Section $^ { 4 , }$ is the variance-preserving (VP) SDE, with $\begin{array} { r } { f ( x , u ) = - \frac { 1 } { 2 } \beta ( u ) } \end{array}$ x and $\sigma _ { \mathrm { d i f f } } ( u ) = \sqrt { \beta ( u ) }$ for a positive noise schedule $\beta .$

Generation proceeds backward in u. To be consistent with the convention used throughout this paper, we reparametrize time as $t = 1 - u$ and denote by $p ( x , t ) : = q _ { 1 - t } ( x )$ the corresponding marginals, so that $p ( \cdot , 0 ) \stackrel { \cdot } { \approx } \dot { \mathcal { N } } ( 0 , I _ { d } )$ corresponds to noise and $p ( \cdot , 1 ) = p _ { \mathrm { d a t a } }$ to data. By the classical time-reversal result of [28], the marginals $p ( \cdot , t )$ are reproduced by the family of generative dynamics

$$
\begin{array} { r } { \mathrm { d } x _ { t } = \Big [ - f ( x _ { t } , 1 - t ) + \frac { 1 + \xi } { 2 } \sigma _ { \mathrm { d i f f } } ( 1 - t ) ^ { 2 } s ( x _ { t } , t ) \Big ] \mathrm { d } t + \sqrt { \xi } \sigma _ { \mathrm { d i f f } } ( 1 - t ) \mathrm { d } W _ { t } , \qquad \xi \geq 0 , } \end{array}\tag{2}
$$

where $s ( x , t ) = \nabla$ log $p ( x , t )$ is the score. The case $\xi = 1$ recovers the reverse-time SDE, whereas $\xi = 0$ yields the deterministic probability flow ODE [3]

$$
\frac { \mathrm { d } x _ { t } } { \mathrm { d } t } = v ( x _ { t } , t ) , \qquad v ( x , t ) = - f ( x , 1 - t ) + \frac { 1 } { 2 } \sigma _ { \mathrm { d i f f } } ( 1 - t ) ^ { 2 } s ( x , t ) .\tag{3}
$$

Intermediate values of $\dot { \boldsymbol { \xi } }$ interpolate between the two regimes; see also [5]. Regardless of $\mid \xi ,$ the marginals satisfy the continuity equation

$$
\begin{array} { r } { \partial _ { t } p ( x , t ) = - \big \langle \nabla , p ( x , t ) v ( x , t ) \big \rangle . } \end{array}\tag{4}
$$

In practice, the score is unknown and is approximated by a neural network trained via denoising score matching [29, 30]. A common parametrization, which we also adopt, is the noise-prediction objective of [2]: a network $\hat { \varepsilon } ( x , u )$ is trained to minimize $\mathbb { E } \| \hat { \varepsilon } ( x _ { u } , u ) - \varepsilon \| ^ { 2 }$ , where $x _ { u } = \bar { \alpha } ( u ) x _ { 0 } + \sigma ( u ) \varepsilon$ with $\varepsilon \sim \dot { \mathcal { N } } ( 0 , I _ { d } )$ , and the score is recovered as $s ( x , t ) \approx - \hat { \varepsilon } ( x , t ) / \sigma ( t )$ . Finally, the probability flow ODE (3) also enables likelihood evaluation: along its trajectories, the instantaneous change-of-variables formula [25, 26] gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } \log p ( x _ { t } , t ) = - \big \langle \nabla , v ( x _ { t } , t ) \big \rangle ,\tag{5}
$$

where, in high dimension, the divergence is typically estimated with the Hutchinson trace estimator [31]. As discussed in Section 3, equation (5) only holds when the particle is transported by the velocity field of the model whose density is evaluated, and must be generalized in our multi-agent setting. This equation will become essential in deriving our main algorithm.

## 2.2 Fe<sub>y</sub>nman–Kac correctors

Many distributions of practical interest are not the marginals of a single trained difusion model, even when they are explicitly defined in terms of such models. Typical examples include the annealed distribution $p ^ { \beta }$ , the product $p _ { 1 } p _ { 2 }$ of two models, and the geometric average ${ p } _ { 1 } ^ { 1 - \bar { w } } { p } _ { 2 } ^ { w }$ , which underlies classifier-free guidance [19]. A natural heuristic is to simulate (2) with the correspondingly combined scores, $\mathrm { e . g . , } \beta s$ or $s _ { 1 } + s _ { 2 }$ . However, the resulting dynamics do not reproduce the desired intermediate distributions, and the generated samples are therefore biased [15].

To address this issue, [16] proposed the Feynman–Kac correctors (FKC). Their key observation is that, for each of the aforementioned targets, the time derivative of the desired path of distributions can be written as a continuity equation of the form (4), augmented with a term of the form $p g ,$ , where $g$ is commonly referred to as the potential function, see [17]. In other words, it satisfies a Feynman–Kac partial diferential equation (PDE), with a velocity field and a potential that are explicitly computable from the pretrained scores. By the Feynman–Kac formula [17], such a PDE can be simulated by a population of weighted particles: the particles are transported by the velocity field, while their log-weights accumulate the potential along the trajectories. The weights are then used within a sequential Monte Carlo (SMC) scheme [17, 18], in which particles are periodically resampled in proportion to their weights, which prevents weight degeneracy and focuses the computational efort on the relevant regions of space. This principle is formalized in the following lemma, which will become essential to proving our main result.

Lemma 1. Let $p ( x , t ) : \mathbb { R } ^ { d } \times [ 0 , 1 ]  \mathbb { R } _ { + }$ be a function with $p ( \cdot , t ) \in L ^ { 1 } ( \mathbb R ^ { d } )$ for every $t \in [ 0 , 1 ]$ . Assume that $\boldsymbol { p } ( \boldsymbol { x } , t )$ adheres to the following Feynman–Kac PDE

$$
\partial _ { t } p ( x , t ) = - \big \langle \nabla , p ( x , t ) v ( x , t ) \big \rangle + p ( x , t ) g ( x , t )\tag{6}
$$

for some vector field $\boldsymbol { v } ( \boldsymbol { x } , t )$ in $\mathbb { R } ^ { d }$ , and a real-valued potential function $g ( x , t )$ . Then, samples from the normalized density pe can be drawn by simulating the weighted SDE

$$
\mathrm { d } x _ { t } = v ( x _ { t } , t ) \mathrm { d } t \quad a n d \quad \mathrm { d } w _ { t } = \overline { { g } } ( x _ { t } , t ) \mathrm { d } t ,\tag{7}
$$

where

$$
\overline { { g } } ( x , t ) = g ( x , t ) - \int _ { \mathbb { R } ^ { d } } g ( x ^ { \prime } , t ) \widetilde { p } ( x ^ { \prime } , t ) \mathrm { d } x ^ { \prime } .\tag{8}
$$

We provide the proof of Lemma 1 in Appendix A for completeness. In practice, the weights in (7) relate the simulated particles to the normalized density $\widetilde { p } ( \cdot , t )$ as follows [16]. Denoting by $\begin{array} { r } { w _ { t } = \int _ { 0 } ^ { t } g ( x _ { s } , s ) } \end{array}$ ds the log-weight accumulated along a trajectory, one has, for any bounded test function $\phi : \mathbb { R } ^ { d }  \mathbb { R }$

$$
\mathbb { E } _ { \boldsymbol { x } \sim \widetilde { \boldsymbol { p } } ( \cdot , t ) } \big [ \phi ( \boldsymbol { x } ) \big ] = \frac { 1 } { Z _ { t } } \mathbb { E } \big [ e ^ { w _ { t } } \phi ( x _ { t } ) \big ] , \qquad Z _ { t } = \mathbb { E } \big [ e ^ { w _ { t } } \big ] ,
$$

where the expectations on the right-hand side are taken over the trajectories of (7). Given $K \in \mathbb N$ simulated particles $\{ ( x _ { t } ^ { ( k ) } , w _ { t } ^ { ( k ) } ) \} _ { k = 1 } ^ { K }$ , replacing both expectations by empirical averages yields the self-normalized importance sampling estimator

$$
\mathbb { E } _ { x \sim \widetilde { p } ( \cdot , t ) } \left[ \phi ( x ) \right] \approx \sum _ { k = 1 } ^ { K } \frac { e ^ { w _ { t } ^ { ( k ) } } } { \sum _ { \ell = 1 } ^ { K } e ^ { w _ { t } ^ { ( \ell ) } } } \phi \bigl ( x _ { t } ^ { ( k ) } \bigr ) ,
$$

which converges to the exact expectation as $K  \infty$ . Accordingly, approximate samples from $\widetilde { p } ( \cdot , t )$ are obtained by drawing particles from the categorical distribution with probabilities proportional to $\{ e ^ { w _ { t } ^ { ( k ) } } \} _ { k = 1 } ^ { K }$ , which is precisely the resampling step of Algorithm 1.

Lemma 1 is a direct consequence of the Feynman–Kac formula [17], and underlies the constructions of [16]. Note that when $g \equiv 0$ , the PDE (6) reduces to the continuity equation (4), the weights remain constant, and the lemma recovers standard sampling with the velocity field v. Moreover, the centering term in (8) is common to all particles at a given time, and therefore cancels upon normalization of the weights. Consequently, in practice one may accumulate the uncentered potential g and normalize the weights only when they are used, as is done in Algorithm 1. Finally, we emphasize that, in [16], the potential g is derived separately for each specific family of targets. Theorem 1 below removes this restriction by providing the potential for an arbitrary combination of the agents.

## 2.3 Reward-<sub>g</sub>uided tiltin<sub>g</sub>

A complementary line of work concerns the adaptation of a pretrained difusion model to maximize a downstream reward $r : \mathbb { R } ^ { d }  \mathbb { R }$ at inference time, without fine-tuning [20]. Maximizing the reward alone would collapse the generation onto its maximizers and discard the information encoded in the pretrained model. The standard remedy is to regularize the objective by the Kullback–Leibler divergence to the pretrained distribution $p _ { \mathrm { p r } \epsilon }$ e

$$
q ^ { \star } = \arg \operatorname* { m a x } _ { q } \Big \{ \mathbb { E } _ { x \sim q } \big [ r ( x ) \big ] - \kappa \mathrm { K L } \big ( q \| p _ { \mathrm { p r e } } \big ) \Big \} , \qquad \kappa > 0 .\tag{9}
$$

This problem admits the closed-form solution

$$
\begin{array} { r } { q ^ { \star } ( x ) = \displaystyle \frac { 1 } { Z _ { \kappa } } p _ { \mathrm { p r e } } ( x ) e ^ { r ( x ) / \kappa } , \qquad Z _ { \kappa } = \int _ { \mathbb { R } ^ { d } } p _ { \mathrm { p r e } } ( x ) e ^ { r ( x ) / \kappa } \mathrm { d } x , } \end{array}\tag{10}
$$

which follows from the identity

$$
\begin{array} { r } { \mathbb { E } _ { q } [ r ] - \kappa \mathrm { K L } ( q \parallel p _ { \mathrm { p r e } } ) = \kappa \log Z _ { \kappa } - \kappa \mathrm { K L } ( q \parallel q ^ { \star } ) , } \end{array}\tag{11}
$$

valid for any q with $\mathrm { K L } ( q \| p _ { \mathrm { p r e } } ) < \infty$ . Indeed, the right-hand side is maximized if and only if $q = q ^ { \star }$ . The parameter κ balances the two objectives: as $\kappa  \infty ,$ the tilted distribution $q ^ { \star }$ reduces to $p _ { \mathrm { p r e } } ,$ whereas as $\kappa  0 ^ { + }$ , it concentrates on the maximizers of r within the support of $p _ { \mathrm { p r e } }$

As reviewed in [20], existing inference-time methods, such as classifier guidance [4], SMC-based guidance [23] and value-based sampling, can be viewed as approximations of the optimal denoising process that samples from (10). A central ingredient of these methods is a look-ahead function, the soft value function, which predicts, from an intermediate state $x _ { t } ,$ , the reward attained at the end of the generation. Since the soft value function is generally intractable, it is commonly approximated by evaluating the reward at an estimate $\widehat { x } _ { 1 }$ of the terminal state, obtained for instance from the posterior mean via Tweedie’s formula [10] or by simulating the generative dynamics up to the terminal time. We adopt the latter approach in Section 4.

In this work, we apply the same principle to targets composed of several difusion models. Given a terminal function $h _ { 1 }$ of the agents, we denote by $\widetilde { h } _ { 1 }$ its normalization, and consider the tilted target (16), which, by (10), is the solution of (9) with $p _ { \mathrm { p r e } } = \widetilde { h } _ { 1 }$

Our contribution. The FKC framework [16] provides exact weighted sampling for specific families of targets built from pretrained models, while reward-guided methods [20] tilt the distribution of a single pretrained model toward high values of a reward. In this paper, we unify and generalize both approaches. We derive the Feynman–Kac potential for an arbitrary target function $h _ { 1 }$ of N difusion models, interpolated geometrically from an arbitrary initial function $h _ { 0 } .$ , with an optional reward-based tilt. The resulting GFKC algorithm performs the correction on the fly, during the generation process, and requires neither retraining nor fine-tuning of the agents.

## 3 Derivation of the <sub>g</sub>eneralized Fe<sub>y</sub>nman-Kac correctors al<sub>g</sub>orithm

In this section, we present our main result, which generalizes the Feynman–Kac correctors of[16] from specific families of targets to arbitrary target functions of several pretrained difusion models, and incorporates the reward-based tilting of Section 2.3 into the same framework.

We consider $N \geq 1$ pretrained difusion models, henceforth referred to as agents, with marginals $p _ { 1 } ( x , t ) , \dotsc , p _ { N } ( x , t )$ scores $s _ { i } = \nabla \log p _ { i }$ and velocity fields $v _ { i } , i = 1 , \ldots , N$ . As in Section 2.1, each agent is assumed to satisfy the continuity equation (4) with its own velocity field, namely $\partial _ { t } p _ { i } = - \langle \nabla , p _ { i } v _ { i } \rangle$ . Moreover, we assume that all agents share a common, tractable initial distribution at $t = 0$ , as is the case for difusion models with the same forward process, $p _ { i } ( \cdot , 0 ) = \mathcal { N } ( 0 , I _ { d } )$

Our goal is to sample from a target distribution that is defined pointwise in terms of the agents. Specifically, given a positive and diferentiable terminal function $h _ { 1 } : \mathbb { R } _ { + } ^ { N }  \mathbb { R } _ { + } ^ { \bullet }$ , we wish to sample from the normalized density proportional to

$$
h _ { 1 } \big ( p _ { 1 } ( x , 1 ) , \dots , p _ { N } ( x , 1 ) \big )
$$

where the agents are evaluated at the terminal time $t = 1 , \mathrm { i . e . , }$ , at their data distributions. This formulation encompasses, for instance, products of experts, geometric averages underlying classifier-free guidance [19], and annealed distributions, which are the targets considered in [16]. It also includes targets that are not of product form, such as the mixtures and ratios studied in Section 4. We refer to a terminal function as admissible whenever the resulting density is both diferentiable and integrable, so that its normalization is well defined.

Since the target is generally not the terminal marginal of any single agent, we construct a path of distributions connecting a tractable initial distribution to the target. To this end, we introduce an initial function $h _ { 0 } : \mathbb { R } _ { + } ^ { N } \to \mathbb { R } _ { + }$ chosen so that $h _ { 0 } \big ( p _ { 1 } ( \cdot , 0 ) , \ldots , p _ { N } ( \cdot , 0 ) \big )$ is easy to sample from. For instance, when the agents share a common prior $p _ { 0 }$ , any $h _ { 0 }$ with $h _ { 0 } ( p _ { 0 } , \ldots , p _ { 0 } ) = p _ { 0 }$ , such as the geometric mean, is a valid choice. A natural and flexible family of initial functions is given by

$$
h _ { 0 } ( p _ { 1 } , \ldots , p _ { N } ) = \phi ^ { - 1 } \Big ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \phi ( p _ { i } ) \Big ) ,
$$

where $\phi : \mathbb { R } _ { + } \to \mathbb { R }$ is continuous and strictly monotone. This family includes the arithmetic mean for $\phi ( z ) = z$ , the geometric mean for $\phi ( z ) = \log z ,$ , and, more generally, the power means for $\phi ( z ) = z ^ { \alpha }$ with $\alpha \neq 0$ , among which the harmonic mean corresponds to $\alpha = - 1$ . Every member of the family satisfies $h _ { 0 } ( p _ { 0 } , \ldots , p _ { 0 } ) = p _ { 0 }$ , and is therefore a valid initial function whenever the agents share a common prior $p _ { 0 }$ . The choice of ϕ thus ofers considerable freedom in designing the initial distribution of the path (13), without afecting the target $h _ { 1 } \mathrm { a t } t = 1$ . In our experiments in Section 4, we use the geometric mean

The two functions are then interpolated geometrically by means of diferentiable schedules $a _ { 0 } , a _ { 1 } : [ 0 , 1 ] \to [ 0 , 1 ]$ satisfying $a _ { 0 } ( 0 ) = a _ { 1 } ( 1 ) = 1$ and $a _ { 0 } ( 1 ) \stackrel { \scriptscriptstyle } { = } a _ { 1 } ( 0 ) = 0$ . Throughout, we write $\partial _ { \log p _ { i } }$ log $h : = p _ { j } \partial _ { p _ { j } }$ log h for the derivative of log h with respect to log $p _ { j }$ , which is the natural quantity in the log domain.

The following theorem shows that the resulting path satisfies a Feynman–Kac PDE of the form (6), with an explicit potential, so that Lemma 1 applies.

Theorem 1. Let $\{ p _ { i } ( x , t ) \} _ { i = 1 } ^ { N }$ be a set ofdifusion models with corresponding vectorfields $v _ { i } .$ Let $h _ { 0 } : \mathbb { R } ^ { N } $ R be an initial annealing function, and denote by $h _ { 1 } : \mathbb { R } ^ { N } $ R the target distribution function. Then, simulating (7) with an arbitrary unifying vector field $v ( x , t ) = V { \big ( } v _ { 1 } ( x , t ) , \dots , v _ { N } ( x , t ) { \big ) }$ and

$$
\begin{array} { l } { { \displaystyle g = \dot { a } _ { 0 } \log h _ { 0 } + \dot { a } _ { 1 } \log h _ { 1 } + \left. \nabla , v \right. } } \\ { { \displaystyle ~ - \sum _ { j = 1 } ^ { N } \left( \left. \nabla , v _ { j } \right. + \left. v _ { j } - v , s _ { j } \right. \right) \left( a _ { 0 } \partial _ { \log p _ { j } } \log h _ { 0 } + a _ { 1 } \partial _ { \log p _ { j } } \log h _ { 1 } \right) } , } \end{array}\tag{12}
$$

amounts to sampling from the trajectory

$$
p ( x , t ) = h _ { 0 } \big ( p _ { 1 } ( x , t ) , \ldots , p _ { N } ( x , t ) \big ) ^ { a _ { 0 } ( t ) } h _ { 1 } \big ( p _ { 1 } ( x , t ) , \ldots , p _ { N } ( x , t ) \big ) ^ { a _ { 1 } ( t ) } ,\tag{13}
$$

which geometrically interpolates between $h _ { 0 }$ at $t = 0$ and $h _ { 1 }$ at $t = 1$

Proof. As in Equation (12), we omit the variables x and t throughout the proof for compactness. In the logarithmic domain, the trajectory distribution p of (13) becomes

$$
\log p = a _ { 0 } \log h _ { 0 } ( p _ { 1 } , \dots , p _ { N } ) + a _ { 1 } \log h _ { 1 } ( p _ { 1 } , \dots , p _ { N } ) .
$$

Consequently, diferentiating log p with respect to t and multiplying the result with $p$ we get $\partial _ { t } p \colon$

$$
\partial _ { t } p = p \left[ \dot { a } _ { 0 } \log h _ { 0 } + a _ { 0 } \sum _ { i = 1 } ^ { N } ( \partial _ { p _ { i } } \log h _ { 0 } ) \left( \partial _ { t } p _ { i } \right) + \dot { a } _ { 1 } \log h _ { 1 } + a _ { 1 } \sum _ { i = 1 } ^ { N } ( \partial _ { p _ { i } } \log h _ { 1 } ) \left( \partial _ { t } p _ { i } \right) \right] = : p \mathcal { C } .\tag{14}
$$

We now simplify the term $\partial _ { t } p _ { i }$ in $\mathcal { C }$ using the agents’ continuity equations (4) and get

$$
\begin{array} { r l } & { \partial _ { t } p _ { i } = - \langle \nabla , p _ { i } v _ { i } \rangle } \\ & { \qquad = - p _ { i } \langle \nabla , v _ { i } \rangle - \langle v _ { i } , \nabla p _ { i } \rangle } \\ & { \qquad = - p _ { i } \big ( \langle \nabla , v _ { i } \rangle + \langle v _ { i } , s _ { i } \rangle \big ) . } \end{array}
$$

Here we used the fact that $\nabla p _ { i } = p _ { i } \nabla \log p _ { i } = p _ { i } s _ { i }$ for all $i = 1 , \ldots , N$ . Before proceeding to the final step of the proof, we note that for an arbitrary unifying vector field v, the term $\langle \nabla , p v \rangle$ can be rewritten as

$$
\begin{array} { r } { \langle \nabla , p v \rangle = p \langle \nabla , v \rangle + \langle v , \nabla p \rangle = p \big ( \langle \nabla , v \rangle + \langle v , \nabla \log p \rangle \big ) = : p \mathcal { D } , } \end{array}
$$

while the term $\mathcal { D }$ can be simplified by directly computing $\nabla$ log p:

$$
\nabla \log p = \nabla { \big ( } a _ { 0 } \log h _ { 0 } + a _ { 1 } \log h _ { 1 } { \big ) } = \sum _ { i = 1 } ^ { N } { \big ( } a _ { 0 } \partial _ { p _ { i } } \log h _ { 0 } + a _ { 1 } \partial _ { p _ { i } } \log h _ { 1 } { \big ) } p _ { i } s _ { i } ,
$$

which overall gives

$$
\mathcal { D } = \langle \nabla , v \rangle + \sum _ { i = 1 } ^ { N } p _ { i } \langle v , s _ { i } \rangle \big ( a _ { 0 } \partial _ { p _ { i } } \log h _ { 0 } + a _ { 1 } \partial _ { p _ { i } } \log h _ { 1 } \big ) .\tag{15}
$$

Finally, combining (14) with (15) we get

$$
\begin{array} { r l } & { \partial _ { t } p = p \mathscr { C } - \langle \nabla , p v \rangle + \langle \nabla , p v \rangle } \\ & { \qquad = - \langle \nabla , p v \rangle + ( \mathscr { C } + \mathscr { D } ) p } \\ & { \qquad = - \langle \nabla , p v \rangle + g p , } \end{array}
$$

which coincides precisely with the Feynman-Kac equation (6) with the potential function g as in (12). □

Theorem 1 establishes an interpolation between an initial function $h _ { 0 }$ and a desired terminal function $h _ { 1 }$ along the path given in (13). The sampling process consists of simulating the SDE (7) with a suitable choice of the function $g ,$ which is constructed so that, at the terminal time $t = 1$ , the reweighted samples are distributed according to $h _ { 1 }$ . The choice of V is discussed before the derivation of our main algorithm.

To evaluate the density of a sample under a flow model, one typically relies on the instantaneous change-of-variables formula, commonly known as the log-probability ODE. In our setting, however, this ODE cannot be applied directly, for reasons we now explain, and must instead be suitably modified.

Although the particles evolve under a velocity field $\boldsymbol { v } ( \boldsymbol { x } , t )$ , we must compute, for every particle, the densities $p _ { j } ( x , t )$ associated with each model $j .$ . In other words, even though a particle is transported by a velocity field $v \neq v _ { j }$ , its density must still be evaluated under every model $j .$ . This is achieved by a generalization of the log-probability ODE (5) as follows.

$$
\begin{array} { r l } & { \frac { \mathrm d } { \mathrm d t } \log p _ { j } ( x _ { t } , t ) = \cfrac { 1 } { p _ { j } ( x _ { t } , t ) } \frac { \mathrm d } { \mathrm d t } p _ { j } ( x _ { t } , t ) } \\ & { \qquad = \cfrac { 1 } { p _ { j } ( x _ { t } , t ) } \left( \frac { \partial } { \partial t } p _ { j } ( x _ { t } , t ) + \left. \nabla p _ { j } ( x _ { t } , t ) , \frac { \mathrm d x _ { t } } { \mathrm d t } \right. \right) } \\ & { \qquad = \cfrac { 1 } { p _ { j } ( x _ { t } , t ) } \left( - \langle \nabla , p _ { j } ( x _ { t } , t ) v _ { j } ( x _ { t } , t ) \rangle + \left. \nabla p _ { j } ( x _ { t } , t ) , \frac { \mathrm d x _ { t } } { \mathrm d t } \right. \right) } \\ & { \qquad = p _ { j } ^ { - 1 } \big ( - p _ { j } \langle \nabla , v _ { j } \rangle - \langle v _ { j } , \nabla p _ { j } \rangle + \langle v , \nabla p _ { j } \rangle \big ) } \\ & { \qquad = - \langle \nabla , v _ { j } ( x _ { t } , t ) \rangle - \langle v _ { j } ( x _ { t } , t ) - v ( x _ { t } , t ) , s _ { j } ( x _ { t } , t ) \rangle . } \end{array}
$$

Here, the second line follows by expanding the total time derivative into the partial time derivative and the contribution from the time derivative of the trajectory $x _ { t } ;$ the third line uses the continuity equation for $p _ { j } ;$ the fourth line uses $\mathrm { d } x _ { t } / \mathrm { d } t = v ( x _ { t } , t )$ , since the trajectory is generated by the velocity field $v ;$ and the fifth line uses the identity $\nabla p _ { j } = p _ { j } s _ { j }$ . Note that when $v = v _ { j }$ , the second term vanishes and we recover the standard log-probability ODE as one would expect.

We now examine how the inclusion of steering afects the framework. Suppose now that we wish to steer the generative process toward the tilted target

$$
{ h } _ { 1 } ^ { \kappa } \bigl ( p _ { 1 } ( x , t ) , \ldots , p _ { N } ( x , t ) \bigr ) \propto { h } _ { 1 } \bigl ( p _ { 1 } ( x , t ) , \ldots , p _ { N } ( x , t ) \bigr ) e ^ { r ( x ) / \kappa }\tag{16}
$$

which is the equivalent of (10), solving the optimization problem in (9), where $\kappa > 0$ is the steering parameter, and $r ( x )$ is a reward function to be maximized. The only modification required in the algorithm is the addition of the term $\dot { a } _ { 1 } ( t ) r ( x ) / \kappa$ to the function $g$ in (12). This additional term alters the evolution of the distribution and tilts the path p toward the maxima of the reward function r.

The steering parameter κ controls the strength of the bias toward maximizing r throughout the generation process. In the limit $\kappa  \infty$ , the additional term becomes negligible, and the original path of distributions is recovered. In contrast, as $\kappa  0 ^ { + }$ , the steering term becomes increasingly dominant and ultimately governs the generative process, as illustrated in Figure 4.

Building on Theorem 1 and the mathematical evaluations presented in this section, we derive a practical algorithm, henceforth referred to as the GFKC algorithm, for sampling from any admissible target function $h _ { 1 } ,$ , with optional reward-based steering. For clarity of presentation, the algorithm refers to the update equations $( \mathrm { A } ) { - } ( \mathrm { G } )$ , which are stated below.

Although Theorem 1 allows for an arbitrary unifying vector field $v = V ( v _ { 1 } , \ldots , v _ { N } )$ , Algorithm 1 relies on a particular choice of V . The particles are divided into $N$ equally sized sub-populations, and for the i-th sub-population we take $V$ to be the projection onto its i-th argument, $V ( v _ { 1 } , \ldots , v _ { N } ) = \bar { v _ { i } } \mathrm { f o r } i = 1 , \ldots , N$ , so that each particle is transported by the velocity field of its own agent. Theorem 1 is thus applied with $v = v _ { i }$ , which yields the potential $g _ { i }$ in (E). In this way, each sub-population is driven by a genuine generative field and explores the high-density regions of its agent, while the terms $u _ { i j }$ in (C) correct for the discrepancy with the other agents. Since the path (13) and its normalizing constant do not depend on the choice of V , all sub-populations target the same normalized density, and their particles can be pooled into a single weighted population for estimation and resampling. After resampling, each particle retains the label of its agent.

The quantities appearing in Algorithm 1 are defined as follows.

$$
q ( x _ { t } , t ) = { \dot { a } } _ { 0 } ( t ) \log h _ { 0 } ( \log p _ { 1 } ( x _ { t } , t ) , \dots , \log p _ { n } ( x _ { t } , t ) )
$$

$$
+ \dot { a } _ { 1 } ( t ) \log h _ { 1 } ( \log p _ { 1 } ( x _ { t } , t ) , \dots , \log p _ { n } ( x _ { t } , t ) )\tag{A}
$$

$$
b _ { j } ( \boldsymbol { x } _ { t } , t ) = a _ { 0 } ( t ) \partial _ { \log p _ { j } } \log h _ { 0 } ( \log p _ { 1 } ( \boldsymbol { x } _ { t } , t ) , \dots , \log p _ { n } ( \boldsymbol { x } _ { t } , t ) )
$$

$$
+ a _ { 1 } ( t ) \partial _ { \log p _ { j } } \log h _ { 1 } ( \log p _ { 1 } ( x _ { t } , t ) , \dots , \log p _ { n } ( x _ { t } , t ) )\tag{B}
$$

$$
u _ { i j } ( x _ { t } , t ) = \langle \nabla , v _ { j } ( x _ { t } , t ) \rangle + \langle v _ { j } ( x _ { t } , t ) - v _ { i } ( x _ { t } , t ) , s _ { j } ( x _ { t } , t ) \rangle\tag{C}
$$

$$
\frac { d } { d t } \log p _ { j } ( x _ { t } , t ) = - u _ { i j } ( x _ { t } , t )\tag{D}
$$

$$
g _ { i } ( x _ { t } , t ) = \langle \nabla , v _ { i } ( x _ { t } , t ) \rangle + q ( x _ { t } , t ) - \sum _ { j = 1 } ^ { N } u _ { i j } ( x _ { t } , t ) b _ { j } ( x _ { t } , t )\tag{E}
$$

$$
d w _ { t } = g _ { i } ( x _ { t } , t ) d t\tag{F}
$$

$$
d x _ { t } = v _ { i } ( x _ { t } , t ) d t\tag{G}
$$

We emphasize that Equations (A)–(G) are expressed entirely in terms of the log-densities log $p _ { j }$ and of derivatives with respect to log $; p _ { j }$ , rather than the densities $p _ { j }$ themselves. Accordingly, all computations in the GFKC algorithm are carried out in the log domain. This is essential in practice, since in high-dimensional spaces the densities $p _ { j }$ may take extremely large or extremely small values, which cannot be represented in floating-point arithmetic. Moreover, in some practical scenarios, the log-densities may additionally be clipped to prevent numerical overflow, as we illustrate in Section 4.

Algorithm 1: The generalized Feynman-Kac correctors (GFKC) algorithm   
Input: Number of agents $N ;$ velocity fields $v _ { 1 } , \ldots , v _ { N }$ and scores $s _ { 1 } , \ldots , s _ { N } ;$ initial and target functions $h _ { 0 } , h _ { 1 } ;$   
schedules $a _ { 0 } , a _ { 1 } ;$ reward function r and steering parameter $\kappa ;$ total number of particles $M N ;$ space   
dimension d; time step $\Delta t ;$ set of resampling times $\tau$   
Output: Weighted particles $\left\{ \left( x _ { 1 } , w _ { 1 } \right) \right\}$ approximating the target $\displaystyle h _ { 1 } e ^ { r / \kappa }$   
I<sub>n</sub>iti<sub>a</sub>li<sub>za</sub>ti<sub>on:</sub>   
Assign an equal number of particles $M \in \mathbb { N }$ to each agent $i = 1 , \ldots , N$   
$t  { 0 }$   
foreach particle do   
Sample $x _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$   
f<sub>or</sub> $j = 1$ to N do   
$\log p _ { j } ( x _ { 0 } , 0 ) \gets - \frac { 1 } { 2 } \left( \| x _ { 0 } \| ^ { 2 } + d \log ( 2 \pi ) \right)$   
<sub>en</sub>d   
$w _ { 0 } \gets 0$   
<sub>en</sub>d   
E<sub>vo</sub>l<sub>u</sub>ti<sub>on:</sub>   
<sub>w</sub>hil<sub>e</sub> $t < 1$ d<sub>o</sub>   
f<sub>or</sub> $i = 1$ to N do   
foreach particle $x _ { t }$ belonging to agent i do   
Compute $q ( x _ { t } , t )$ according to $( \mathrm { A } )$   
if steering is True then   
Estimate clean samples $\widehat { x } _ { 1 }$ from the intermediate states $x _ { t }$   
$q ( x _ { t } , t ) \gets q ( x _ { t } , t ) \dot { + } \dot { a } _ { 1 } r ( \widehat { x } _ { 1 } ) / \kappa$   
<sub>en</sub>d   
f<sub>or</sub> $j = 1$ to N do   
Compute $b _ { j } ( x _ { t } , t )$ according to (B)   
Compute $\dot { u _ { i j } } ( x _ { t } , t )$ according to (C)   
log $\bar { p _ { j } } ( x _ { t + \Delta t } , t + \Delta t ) \gets \bar { \log p _ { j } } ( x _ { t } , t ) - \Delta t u _ { i j } ( x _ { t } , t )$ Eq. (D)   
<sub>en</sub>d   
Compute $g _ { i } ( x _ { t } , t )$ according to (E)   
$\boldsymbol { w _ { t + \bar { \Delta } t } }  \boldsymbol { w _ { t } } + \Delta t g _ { i } ( x _ { t } , t ) \mathrm { \bar { E } q . } ( \mathrm { F } )$   
$x _ { t + \Delta t } \gets x _ { t } + \Delta t v _ { i } ( x _ { t } , t )$ Eq. (G)   
<sub>en</sub>d   
<sub>en</sub>d   
$t \gets t + \Delta t$   
if $t \in \mathcal T$ th<sub>en</sub>   
Resample the particles from the multinomial distribution with weights proportional to $\{ \exp ( w _ { t } ) \}$   
<sub>en</sub>d   
<sub>en</sub>d

## 4 N<sub>umer</sub>i<sub>ca</sub>l ill<sub>us</sub>t<sub>ra</sub>ti<sub>ons</sub>

We illustrate the performance of Algorithm 1 in two settings of inherently diferent nature: the first involves analytic generative models for two overlapping bivariate Gaussian distributions, while the second consists of difusion models trained on the MNIST dataset [27].

## 4.1 Overla<sub>pp</sub>in<sub>g</sub> bivariate normal distributions

We consider a two-dimensional synthetic setting in which all quantities of interest are available in closed form, which allows the output of the GFKC algorithm 1 to be compared against exact ground-truth densities. The setting consists of $N = 2$ agents, each given by a difusion model whose data distribution is an isotropic bivariate Gaussian with unit

covariance,

$$
p _ { 1 } = { \mathcal { N } } ( \mu _ { 1 } , I _ { 2 } ) \quad { \mathrm { a n d } } \quad p _ { 2 } = { \mathcal { N } } ( \mu _ { 2 } , I _ { 2 } ) ,\tag{17}
$$

with means $\mu _ { 1 } = ( 3 , 3 )$ and $\mu _ { 2 } = ( 3 . 5 , 3 . 5 )$ . Since $\lVert \mu _ { 1 } - \mu _ { 2 } \rVert = \sqrt { 2 } / 2 \approx 0 . 7 1$ is smaller than the unit standard deviation of each component, the two distributions overlap substantially, and their high-density regions are dificult to separate by sampling from either model alone.

Recall that we adopt the convention that $t = 0$ corresponds to noise and $t = 1$ to data. Each agent $j \in \{ 1 , 2 \}$ defines a path of marginals $p _ { j } ( \cdot , t ) = \mathcal { N } ( t \mu _ { j } , I _ { 2 } ) , t \in [ 0 , 1 ]$ , which interpolates between the standard Gaussian at $t = 0$ and the data distribution $p _ { j }$ at $t = 1$ . In particular, all agents share the same initial distribution $\mathcal { N } ( 0 , I _ { 2 } )$ , and the scores $s _ { j } ( x , t ) = - ( x - t \mu _ { j } )$ are known exactly. Consequently, no learned approximation enters the experiment, and any discrepancy between the generated and target distributions can be attributed to the sampling procedure itself rather than to model error.

The interpolation between the initial and terminal functions follows the path (13) with linear schedules $a _ { 0 } ( t ) = 1 - t$ and $a _ { 1 } ( t ) = t .$ . As the initial function we take the geometric mean of the two models

$$
h _ { 0 } ( p _ { 1 } , p _ { 2 } ) = \sqrt { p _ { 1 } p _ { 2 } } ,
$$

which at $t = 0$ coincides with the common standard Gaussian prior. This choice ensures that the interpolation starts from the distribution from which the particles are initialized.

For the terminal function $h _ { 1 }$ we consider two cases; the first one is designed to emphasize the region in which one model disagrees with the other, while the second one is its symmetrized counterpart. In particular, the first is the asymmetric ratio

$$
h _ { 1 } ( p _ { 1 } , p _ { 2 } ) = \frac { p _ { 1 } } { 1 + C p _ { 2 } } ,\tag{18}
$$

for some $C > 0$ , which favors regions of high density under $p _ { 1 }$ and low density under $p _ { 2 }$ . The second one is

$$
h _ { 1 } ( p _ { 1 } , p _ { 2 } ) = \frac { 1 } { 2 } \left( \frac { p _ { 1 } } { 1 + C p _ { 2 } } + \frac { p _ { 2 } } { 1 + C p _ { 1 } } \right) ,\tag{19}
$$

which favors regions where exactly one of the two models assigns high density. In both cases we set $C = 1 0 0$ , so that the denominators strongly penalize regions in which the competing model is non-negligible. This choice of C produces a clear yet moderate shift of mass. Nevertheless, the results do not depend on this particular choice. Owing to the substantial overlap between $p _ { 1 }$ and $p _ { 2 }$ , the resulting targets are concentrated on relatively small regions of the plane, which makes them challenging to sample from directly.

We consider 100,000 particles per agent so that the generated samples cover the whole support of the target density, which allows an accurate comparison with the ground truth across the entire domain. We emphasize that this large population is needed only for visualizing and evaluating the full density: drawing individual samples from the target requires far fewer particles, as in the MNIST experiments of Section 4.2.

Since the setting is low-dimensional and the models are available in closed form, we compare the generated samples directly with the ground-truth target distributions. We first present the results for the two choices of $h _ { 1 }$ , then report the convergence behavior of the algorithm, and finally illustrate the efect of steering.

Figure 2 reports the convergence of the GFKC algorithm for the two experiments of Figure 1 as the number of particles per agent increases. The error is measured in two ways: the Kullback–Leibler divergence $\mathrm { K L } ( h _ { 1 } \parallel \hat { h } _ { 1 } )$ between the ground-truth target $h _ { 1 }$ and its empirical approximation $\hat { h } _ { 1 }$ produced by the algorithm, and the $\ell _ { 2 }$ distance between the two.

Finally, for the steered variant of the algorithm, we use the reward function $r ( x ) = - \| x \| _ { 2 }$ , so that the steered target $\boldsymbol { h _ { 1 } } ~ e ^ { \boldsymbol { r / \kappa } }$ of (16) tilts the terminal distribution toward points closer to the origin, with the strength of this tilt controlled by the steering parameter $\kappa > 0$

To examine the steering efect, we consider the symmetric terminal function $h _ { 1 }$ defined in (19), together with the prescribed reward function and steering parameter $\kappa = 1$ . This choice creates a competition between two objectives: sampling from the symmetric target, which is bimodal as shown in Figure 1, and favoring samples that lie close to the origin.

Since the mode associated with the first agent, centered around $\mu _ { 1 } = ( 3 , 3 )$ , is closer to the origin, the steered target naturally assigns more mass to this mode at the expense of the other. Figure 3 illustrates the efect of steering for $\kappa = 1$ and compares the result with the unsteered target $h _ { 1 }$ . Figure 4 then shows the steered distributions for several values of $\kappa ,$ confirming that the steering efect strengthens as κ decreases, in agreement with the theory.

![](images/0dfcd1ede34c9156c5d3d3804975510c0c02e5e6585837dc11beb89e7c0cb57e.jpg)

Figure 1: Sampling with the GFKC algorithm in the bivariate Gaussian setting. Each row corresponds to one choice of the terminal function $h _ { \mathrm { 1 } } \colon$ the asymmetric ratio of (18) on top and its symmetrized counterpart (19) on bottom. Left column: the ground-truth target densities, obtained by applying $h _ { 1 }$ directly to the analytic densities (17). Middle column: the empirical densities generated by the GFKC algorithm, using 100,000 particles per agent and simulating up to time $t = 1$ . Right column: the relative pointwise error $| h _ { 1 } - \hat { h } _ { 1 } | /$ max<sub>x</sub> $h _ { 1 }$ between the ground-truth and the generated densities. Note the close agreement between the left and middle columns.  
![](images/f91535efd52710aa54ba5153e50d54610b228b37ccc1da755a00d94cfe91276e.jpg)  
Figure 2: Convergence of the GFKC algorithm as a function of the number of particles per agent, measured by $\mathrm { K L } ( h _ { 1 } \parallel \hat { h } _ { 1 } )$ and the $\ell _ { 2 }$ distance, for both choices of $h _ { 1 }$

![](images/552cffce62dbf4371c23346b9da421bfa523c4f58271a3fcdbde8e6e4c7ea5f0.jpg)

![](images/3a40f958b2f709782a52dd3b427a62f9619bb3e884266c9e5450fe67411c7620.jpg)

![](images/eb20b82ad1a3cb50faf206d206ec10d8e32554b40de0b5be74321d5eb7db8de9.jpg)  
Figure 3: Efect of steering on the symmetric target $h _ { 1 }$ of (19), with reward $r ( x ) = - \| x \|$ and $\kappa = 1$ . Left: the analytic unsteered target density $h _ { 1 }$ . Middle: the empirical density generated by the GFKC algorithm with steering, using 100,000 particles per agent and simulating up to time $t = \bar { 1 }$ . Right: the pointwise diference between the steered empirical density and the unsteered target. Note the shift of mass toward the mode closer to the origin.

During the steered GFKC algorithm, the reward r must be evaluated at the terminal state of each particle, which is not available at intermediate times. We therefore estimate the terminal sample $\widehat { x } _ { 1 }$ from the current state $x _ { t }$ as follows: starting from $x _ { t }$ , we continue the denoising process up to time $t = 1$ using the difusion model of the agent to which the particle belongs, and take the resulting endpoint as $\widehat { x } _ { 1 }$ . This projection is performed independently of the algorithm, in the sense that the particle itself, its log-densities log $p _ { j }$ and its weight are left unchanged. The reward $\widetilde { r } ( \widehat { x } _ { 1 } )$ is then incorporated into $( \mathrm { A } )$ at the equispaced steering times $t _ { k } = 0 . 0 \hat { 2 } k$ for $k = 0 , \ldots , 5 \bar { 0 }$ , which form a dense grid of the time interval and are evenly distributed across the generation process.

![](images/206e07072149b770a24f81094104d476659f03c04fd1398b1c1eef068b09f718.jpg)  
= 1e + 06

![](images/57e3476722313776f1d787afa43b8499f7bcb9a1bfd4e1c91986e3c8e383b2cf.jpg)  
= 10

![](images/61ac639f73de6565317b2880d9c2b0a594f97abf1f7d8e1ff413706c64a6d3a2.jpg)  
= 3

![](images/dff123b695e4de52fe57aecaf53330a21e11560532742711b5009cdb87c32ab0.jpg)  
= 1

![](images/51d844f3e7c25b97b6e6e3d3b27b26caffa58fd3df9fc92e93b2285ccea88aea.jpg)  
= 0:5  
Figure 4: Efect of the steering parameter κ on the symmetric target $h _ { 1 }$ of (19), with reward $r ( x ) = - \| x \|$ . Each panel shows the density generated by the GFKC algorithm for a diferent value of $\kappa .$ As κ decreases, the steering efect becomes increasingly pronounced, and mass is progressively transferred toward the mode closer to the origin. Conversely, for very large $\kappa ,$ here $\kappa = 1 0 ^ { 6 }$ , the steering term is negligible and the generated density coincides with the unsteered target $h _ { 1 }$ , as expected.

## 4<sub>.</sub>2 Ex<sub>pe</sub>rim<sub>e</sub>ntin<sub>g w</sub>ith th<sub>e</sub> MNIST d<sub>a</sub>t<sub>ase</sub>t

We now move from the analytic Gaussian setting to a high-dimensional setting in which the agents are learned difusion models. Whereas in the Gaussian experiments all quantities were available in closed form, here each agent is a difusion model trained on a subset of the MNIST dataset [27], the data space has dimension $d = 2 8 \times 2 8 = 7 8 4$ and neither the scores nor the densities $p _ { j }$ for $j = 1 , 2$ are known analytically. The two agents are trained on two sets of digit classes that share exactly one class. The shared class thus plays the role of the region where the agents agree, while the two remaining classes, each seen by only one agent, correspond to the regions in which they disagree.

To make the distinction between the shared and the agent-specific classes as clear as possible, we select the classes so that the three digits involved are pairwise far apart. To this end, we view each image as a probability measure on the uniform $2 8 \times 2 8$ grid in $[ 0 , 1 ] ^ { \frac { \cdot } { 2 } }$ , obtained by normalizing its pixel intensities, and compare images through the Wasserstein-1 distance $W _ { 1 }$ with Euclidean ground cost, computed exactly by solving the corresponding optimal transport problem. Distances between classes are measured in two complementary ways: as the $W _ { 1 }$ distance between the class-mean images, and as the average $W _ { 1 }$ distance over 1,000 randomly drawn pairs of images from the two classes. Among all triples of digits $\{ a , b , c \}$ , we then select the one maximizing the smallest of the three pairwise distances, and assign to the shared class b the digit whose distances to the other two are largest. This procedure lead to the choice

$$
S _ { 1 } = \{ 1 , 6 \} , \qquad S _ { 2 } = \{ 1 , 7 \}
$$

so that the digit 1 is shared by both agents, while the digits 6 and 7 are seen exclusively by the first and second agent, respectively.

Each agent is trained on the images of its two classes from the MNIST training and test sets, with pixel intensities rescaled to $[ - 1 , 1 ]$ and images flattened into vectors in $\mathbb { R } ^ { 7 8 4 }$ . Since the classes of MNIST are not equally represented, each training batch is drawn with equal probability from the two classes of the agent, so that the resulting model assigns comparable mass to both digits. As in the Gaussian setting, we adopt the convention that $t = 0$ corresponds to noise and $t = 1$ to data. The forward noising process is the variance-preserving difusion with a linear noise schedule, $\beta ( u ) = \beta _ { 0 } + ( \beta _ { 1 } - \beta _ { 0 } )$ u with $\beta _ { 0 } = 0 . 1$ and $\beta _ { 1 } = 2 0$ , where $\overset { \vartriangle } { \boldsymbol { u } } = 1 - t$ denotes the forward time. The corresponding marginals are $x _ { u } = \alpha ( u ) x _ { 0 } + \sigma ( u ) \varepsilon$ , with $\varepsilon \sim \mathcal { N } ( 0 , I _ { d } )$ , log $\begin{array} { r } { \alpha ( u ) = - \frac 1 2 \beta _ { 0 } u - \frac 1 4 ( \beta _ { 1 } - \beta _ { 0 } ) u ^ { 2 } } \end{array}$ and $\sigma ( u ) = \sqrt { 1 - \alpha ( u ) ^ { 2 } }$ . Each network is trained with the standard noise-prediction objective, from which the score is recovered as $s _ { j } ( x , t ) = - \hat { \varepsilon } _ { j } ( x , t ) / \sigma ( t )$

Both agents share the same architecture, a fully connected network acting directly on the flattened images. The difusion time is encoded through a 64-dimensional sinusoidal embedding, followed by a linear layer and a SiLU activation. The embedding is concatenated with the input image and passed through three hidden layers of width 2,048 with SiLU activations, followed by a linear output layer of dimension 784. Each agent is trained for 300,000 iterations using the Adam optimizer with learning rate $2 \times \mathrm { { i } 0 ^ { - 4 } }$ and batch size 512, with the difusion time sampled uniformly from $[ 1 0 ^ { - 3 } , 1 - \dot { 1 } 0 ^ { - 3 } ]$

Unlike in the Gaussian setting, the target $h _ { 1 }$ cannot be evaluated in closed form. The results in this section are therefore assessed qualitatively, through the generated samples and their weights, rather than by comparison with a ground-truth density. Several adjustments to the GFKC algorithm are required in this setting:

1. Divergence estimation. In the Gaussian experiments, the divergences $\langle \nabla , v _ { j } \rangle$ appearing in (C) and (E) were computed exactly. In dimension $d = 7 8 4 _ { \mathrm { : } }$ , this is prohibitively expensive, as it requires d backward passes per agent and per time step. We therefore replace it with the Hutchinson trace estimator [31]

$$
\langle \nabla , v _ { j } \rangle \approx \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \eta _ { \ell } ^ { \top } \nabla v _ { j } \eta _ { \ell } ,
$$

where $\nabla \boldsymbol { v } _ { j }$ is the Jacobian matrix of $v _ { j }$ with respect to $x ,$ and $L = 1 0$ Gaussian probe vectors $\eta _ { \ell } \sim \mathcal { N } ( 0 , I _ { d } )$

2. Stochastic sampler. The particles are evolved with a stochastic sampler, namely an Euler–Maruyama discretization with 100 time steps and stochasticity parameter $\xi = 4$

3. Double precision. To mitigate the accumulation of round-of errors, all computations during sampling are carried out in double precision.

4. Clipping of the log-densities. The accumulated log-densities log $p _ { j }$ are clipped from above at every time step, which prevents overflow in the evaluation of $\mathrm { \Delta } h _ { 0 } , h _ { 1 }$ and their derivatives.

5. Recentering of the log-weights. The log-weights are recentered by their mean at every time step, which prevents them from attaining extreme values. This recentering is valid since the normalized weights are obtained by applying a softmax to the log-weights, and the softmax is invariant under the addition of a constant to all of its elements.

Before applying the GFKC algorithm, we first examine the two agents in isolation. Figure 5 shows images generated independently by each agent using its own difusion sampler, without any reweighting or resampling. As expected, the first agent produces images of the digits 1 and 6, while the second produces images of the digits 1 and 7. These samples serve as a reference for the results of the GFKC algorithm presented below.

To demonstrate the GFKC algorithm in this setting, we consider the ratio target function $h _ { 1 } ( p _ { 1 } , p _ { 2 } )$ with (18), with $C = 1 0 0$ , which favors regions of high density under the first agent and low density under the second, and hence the digit $S _ { 1 } \ : \backslash \ : S _ { 2 } = \ : \{ 6 \}$ , which only the first agent has seen. In a parallel experiment, we use the same function with its arguments swapped, $h _ { 1 } ( p _ { 2 } , p _ { 1 } )$ , which by symmetry favors the digit $\hat { S _ { 2 } } \setminus S _ { 1 } = \{ 7 \}$ , exclusive to the second agent. Figure 6 shows the results of these two experiments, while Figure 7 shows the histograms of the resulting log-weights.

We next consider a target function of a diferent nature, namely $h _ { 1 } = p _ { 1 } + p _ { 2 }$ . Since the digit 1 is assigned high density by both agents, $\mathsf { \bar { S } } _ { 1 } \cap S _ { 2 } = \{ 1 \}$ , this target places the largest mass on the shared digit, in contrast to the ratio target, which suppressed it. To further emphasize this efect, we sharpen the target by raising it to a power,

$$
h _ { 1 } = ( p _ { 1 } + p _ { 2 } ) ^ { \gamma } , \qquad \gamma \geq 1 ,\tag{20}
$$

![](images/82c2150a37bb77d9c69d581ce77a85214568b4ae538d70b0590e7d5f077f3930.jpg)

![](images/9035e9d012bf11b8203ac24ec3ccf5986365d61538ed6342a1c6f8598a7c4bd9.jpg)  
Figure 5: Samples generated independently by each agent, without the GFKC algorithm. Left: 100 images generated by the first agent, trained on the digits $S _ { 1 }$ . Right: 100 images generated by the second agent, trained on the digits $S _ { 2 }$

![](images/bfeaa4811c2aaed42683aabc151d4c493ab737dc400272a200dbac59cf6553cd.jpg)

![](images/0a6fa21c7e5381beebc0610cdfceb307e1bbbd8c1692071cdea0b5dfff7b243e.jpg)  
Figure 6: Results of the GFKC algorithm with the ratio target (18). Left: $h _ { 1 } ( p _ { 1 } , p _ { 2 } )$ . Right: $h _ { 1 } ( p _ { 2 } , p _ { 1 } )$ . Each panel displays the 100 generated particles, 50 per agent, sorted by their final log-weights in decreasing order, from the top-left to the bottom-right corner. Particles from the first agent (digits 1 and 6) are shown in cyan, and particles from the second agent (digits 1 and 7) in magenta. In the left panel, the highest weights are assigned to the digit 6, exclusive to the first agent, followed by the shared digit 1, while the digit 7 receives the lowest weights. The righ panel exhibits the symmetric behavior, with the digit 7 ranked highest and the digit 6 lowest.

![](images/879428b56d01c94ef273d2f00e34e2c63b7a59c7d903acf4d202eadee111425e.jpg)

![](images/e6747b26391c1db783ca65b0ed45a317463499954fd04ab7dc355255f238cecf.jpg)  
Figure 7: Histograms of the final log-weights of the particles generated by the GFKC algorithm, with the ratio target $h _ { 1 } ( p _ { 1 } , p _ { 2 } )$ (left) and the swapped target $h _ { 1 } ( p _ { 2 } , p _ { 1 } )$ (right). Colors correspond to the agents as in Figure 6: cyan for the first agent (digits 1 and 6) and magenta for the second agent (digits 1 and 7). Solid curves show kernel density estimates of each histogram. In both cases, the log-weights of the two agents are clearly separated, with the agent holding the favored exclusive digit receiving the higher weights. Since the histograms are over log-weights, diferences of several thousands translate into ratios of order $e ^ { 1 0 0 0 }$ or more between the weights themselves, so that, after normalization, essentially all the mass is carried by the particles of the favored agent.

and examine how the distribution of the log-weights evolves as γ increases.

Figure 8 shows the results of the GFKC algorithm for $\gamma = 2 ,$ while Figure 9 illustrates how the histogram of the log-weights evolves for increasing values of γ. As γ grows, the log-weights spread over an increasingly wide range, and the peak associated with the shared digit 1 shifts toward larger values. This behavior follows directly from the structure of the target: since log $h _ { 1 } = \gamma \log ( p _ { 1 } + p _ { 2 } )$ , the contribution of $h _ { 1 }$ to the weight increments in (A), (B) and (E) is scaled by $\gamma .$ Consequently, diferences in log-weight between particles are amplified roughly in proportion to γ, and after normalization the weights concentrate increasingly on the particles of highest density under $p _ { 1 } + p _ { 2 }$

We now demonstrate the efect of steering in the MNIST setting. To this end, we use the reward function

$$
r ( \boldsymbol { x } ) = - \| \mathcal { R } ( \boldsymbol { x } ) - \bigstar \bigstar \| _ { 2 } ,\tag{21}
$$

where R denotes the rotation of an image by π radians, and $\left\lceil 9 \right\rceil$ is the mean image of the digit 9 over the MNIST training set, with pixel intensities rescaled to [−1, 1] as for the training data. As in the Gaussian setting, the terminal sample $\widehat { x } _ { 1 }$ entering the steered term in (A) is estimated by projecting each particle to time $t = 1$ with the difusion model of its agent, at the same steering times $t _ { k } = 0 . 0 2$ k for $k = 0 , \ldots , 5 \bar { 0 }$ . As the terminal function, we take the sum target $h _ { 1 } = p _ { 1 } + p _ { 2 } , \mathrm { i } . \mathbf { e } . , \gamma = 1$ , which, as shown above, favors the shared digit 1.

Since a 6 rotated by π radians closely resembles a 9, the reward function r of (21) favors images of the digit $6 ,$ in competition with the target $h _ { 1 } .$ , which favors the digit 1. We therefore expect that, for a suficiently small steering parameter $\kappa ,$ the highest-weighted particles will no longer depict the digit 1, but rather the digit 6.

We begin by illustrating the steering efect for a small value of $\kappa = 1 0 ^ { - 6 }$ , corresponding to strong steering. As shown in Figure 10, although the target $h _ { 1 }$ favors the digit 1, the steering term dominates, and the highest-weighted particles depict the digit 6. As the weights decrease, the generated 6’s become increasingly deformed and their rotations bear less resemblance to 9 . Figure 11 compares the highest-weighted sample, rotated by π radians, with 9 .

To further examine the relation between the reward and the weights, Figure 12 displays the reward values of the 100 particles of Figure 10 against their final log-weights. The cluster in the top-right corner, combining the highest rewards with the highest log-weights, corresponds to the images of the digit 6 generated by the first agent.

We conclude this experiment by illustrating the steering efect for diferent values of κ. Analogously to Figure 4 in the Gaussian setting, Figure 13 shows the efect of steering as κ varies. However, since the density cannot be visualized in high dimension, we display, for each value of $\kappa ,$ the image with the highest log-weight. The values of κ were chosen deliberately to capture the transition between the two competing efects, namely sampling the digit 1, favored by the target $h _ { 1 } .$ , and steering toward the digit 6, favored by the reward. We observe that this transition occurs around the critical value $\kappa \approx 0 . 0 \bar { 0 } 1 5$

![](images/24e05f9820464fc3d933eadf54edc3ecaf08708f88410eb679a84747e45014b7.jpg)  
Figure 8: Results of the GFKC algorithm with the annealed sum target $h _ { 1 } = ( p _ { 1 } + p _ { 2 } ) ^ { 2 } ( \mathrm { i . e . , } \gamma = 2 )$ . The 100 generated particles, 50 per agent, are sorted by their final log-weights in decreasing order, from the top-left to the bottom-right corner, and colored by agent as in Figure 6: cyan for the first agent (digits 1 and 6) and magenta for the second agent (digits 1 and 7). The highest weights are assigned to the shared digit 1, generated by both agents, while the exclusive digits 6 and 7 are interleaved among the lower weights.

![](images/edfa76a8575ba1822a4d6ec33fa65452ae41743ea0db94d0d8873d59929ef941.jpg)

![](images/fb0db09585521ad046b9474cbc888d28233e918323b010d78bd21a069d041d95.jpg)

![](images/7a49c05b790b5b9a4d9d59655527ba57841bf3b6c57cb575025745bbc7b2ea33.jpg)

![](images/a350c3994bf27f174f302f354b4c7f1e81dd11866cf6145531ef3f4c75f1ceb0.jpg)  
Figure 9: Histograms of the final log-weights for the sharpened sum target $h _ { 1 } = ( p _ { 1 } + p _ { 2 } ) ^ { \gamma }$ , with $\gamma \in \{ 1 , 2 , 3 , 1 0 \}$ colored by agent as in Figure 6. As γ increases, the range of the log-weights widens and the peak associated with the shared digit 1 shifts toward larger values, so that the separation between the highest-weighted particles and the rest becomes increasingly pronounced. In contrast to the ratio target, the log-weights of the two agents overlap throughout, and the highest-weighted particles originate sometimes from the first agent and sometimes from the second, reflecting that both agents contribute to the shared digit.

![](images/50063e5771608d891e83a6b7af4c7d1dff24421cf008273db89328d18aef32c2.jpg)  
Figure 10: Results of the steered GFKC algorithm with the sum target $h _ { 1 }$ of (20) with $\gamma = 1$ , the reward r of (21) and a small steering parameter $\kappa = 1 0 ^ { - 6 }$ . The 100 generated particles are sorted by their final log-weights in decreasing order, from the top-left to the bottom-right corner, and colored by agent as in Figure 6. Although $h _ { 1 }$ favors the shared digit 1, the steering term dominates, and the highest weights are assigned to images of the digit 6, all generated by the first agent. As the weights decrease, the generated 6’s become increasingly deformed, and their π radians rotations bear progressively less resemblance to the mean image 9 . They are followed by the digit 1, while the particles of the second agent, which cannot generate the digit 6, receive the lowest weights.

![](images/1143625b92c1efc873590b8882ac3d43c65c6c6dd3795ae19a91da61c8abfc5c.jpg)

![](images/04c1ec0d8f342040667d151a6515eb7e7617c19089f6bab4b131f04f93982082.jpg)  
Figure 11: Left: the highest-weighted particle of the steered GFKC algorithm, a digit 6 generated by the first agent, rotated by π radians. Right: the mean image 9 of the digit 9 over the MNIST training set, which defines the reward r of (21). The rotated sample closely resembles the mean image, illustrating how the steering guides the generation toward high values of the reward.

![](images/e244a6fa615e7c8b7dc4f88ed6e536292a6c9aa673ae34bcc502d06cd0d1c69e.jpg)  
Figure 12: The reward values against the final log-weights for the 100 particles of Figure 10, colored by agent. The two quantities are strongly positively correlated, reflecting the dominance of the steering term for the small $\dot { \kappa } = 1 0 ^ { - 6 }$ The cluster of roughly 30 points in the top-right corner, combining the highest rewards with the highest log-weights, corresponds to the images of the digit 6, all generated by the first agent. The particles of the second agent, which cannot generate the digit $6 ,$ are concentrated in the lower-left region.

![](images/d634ea113b890b382692a5994a0fdbaf9d14b26ba4d0f0f7cf49d7cf61d2e270.jpg)  
= 100

![](images/1fe72e6f33b500f954c3d933e61643bdba7fe93be183c23cd5027ec6d33c57bf.jpg)  
= 0:003

![](images/21ea65242df7c20fd92db3c64bb95311bc062bfed55bcf4d0622329d3208330e.jpg)  
= 0:002

![](images/7f68c63a2925f40a5f8d228568d2ce2cd45988fc47b32c9323a6e88f3f60ddad.jpg)  
= 0:001

![](images/4551ed8e8ab45283260baa6cc5a57feb0414c717f85f4f30ef99bb99960bd549.jpg)  
= 0:0001

![](images/a12436987839300606edd6482f741948f18707b66ab6e2c690deb8117b686377.jpg)  
= 1e 06  
Figure 13: Highest-weighted image of the steered GFKC algorithm for decreasing values of κ. For large κ, the top image depicts the digit 1, favored by $h _ { 1 } .$ while for small κ the steering dominates and the digit 6 emerges. The critical shift occurs between the third and fourth panels, at around $\kappa \approx 0 . 0 \bar { 0 } 1 5$

## 5 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

In this work, we introduced the generalized Feynman–Kac correctors, a principled framework for sampling from arbitrary target functions of several pretrained difusion models, with optional reward-based steering. Our main result in Theorem 1 provides, in closed form, the Feynman–Kac potential of the geometric interpolation between an arbitrary initial and terminal function of the agents, and, combined with a generalized log-probability ODE, leads to Algorithm 1, which is a practical algorithm that operates entirely in the log domain and corrects the generation on the fly, without retraining or fine-tuning. The numerical experiments confirm that the method accurately recovers non-trivial targets in an analytic setting, and that it scales to learned difusion models in high dimension, where it isolates, combines and steers the concepts captured by the individual agents. Several directions remain open, including the reduction of weight degeneracy in high dimension, for instance through adaptive resampling or lower-variance divergence estimators, cheaper look-ahead estimates for steering, and the application of the framework to large-scale models and scientific domains such as molecular design

## R<sub>e</sub>f<sub>erences</sub>

[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265, 2015.

[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[3] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021.

[4] Prafulla Dhariwal and Alexander Nichol. Difusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, volume 34, pages 8780–8794, 2021.

[5] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of difusion-based generative models. In Advances in Neural Information Processing Systems, volume 35, pages 26565–26577, 2022.

[6] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[7] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image difusion models with deep language understanding. In Advances in Neural Information Processing Systems, volume 35, pages 36479–36494, 2022.

[8] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DifWave: A versatile difusion model for audio synthesis. In International Conference on Learning Representations, 2021.

[9] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video difusion models. In Advances in Neural Information Processing Systems, volume 35, pages 8633–8646, 2022.

[10] Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, and Jong Chul Ye. Difusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations, 2023.

[11] Emiel Hoogeboom, Víctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant difusion for molecule generation in 3D. In International Conference on Machine Learning, pages 8867–8887, 2022.

[12] Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E. Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, et al. De novo design of protein structure and function with RFdifusion. Nature, 620(7976):1089–1100, 2023.

[13] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016):493–500, 2024.

[14] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum. Compositional visual generation with composable difusion models. In European Conference on Computer Vision, pages 423–439, 2022.

[15] Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Grathwohl. Reduce, reuse, recycle: Compositional generation with energybased difusion models and MCMC. In International Conference on Machine Learning, 2023.

[16] Marta Skreta, Tara Akhound-Sadegh, Viktor Ohanesian, Roberto Bondesan, Alán Aspuru-Guzik, Arnaud Doucet, Rob Brekelmans, Alexander Tong, and Kirill Neklyudov. Feynman-kac correctors in difusion: Annealing, guidance, and product of experts. In International Conference on Machine Learning, 2025.

[17] Pierre Del Moral. Feynman-Kac Formulae: Genealogical and Interacting Particle Systems with Applications. New York. NY: Springer, 2004.

[18] Arnaud Doucet, Nando de Freitas, and Neil Gordon, editors. Sequential Monte Carlo Methods in Practice. Springer, 2001.

[19] Jonathan Ho and Tim Salimans. Classifier-free difusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

[20] Masatoshi Uehara, Yulai Zhao, Chenyu Wang, Xiner Li, Aviv Regev, Sergey Levine, and Tommaso Biancalani. Inference-time alignment in difusion models with reward-guided generation: Tutorial and review. arXiv preprint arXiv:2501.09685, 2025.

[21] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training difusion models with reinforcement learning. In International Conference on Learning Representations, 2024.

[22] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint matching: Fine-tuning flow and difusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations, 2025.

[23] Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, and John P. Cunningham. Practical and asymptotically exact conditional sampling in difusion models. In Advances in Neural Information Processing Systems, volume 36, pages 31372–31403, 2023.

[24] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of difusion models. In International Conference on Machine Learning, 2025.

[25] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary diferential equations. In Advances in Neural Information Processing Systems, volume 31, 2018.

[26] Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. FFJORD: Freeform continuous dynamics for scalable reversible generative models. In International Conference on Learning Representations, 2019.

[27] Li Deng. The MNIST database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.

[28] Brian D. O. Anderson. Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3): 313–326, 1982.

[29] Aapo Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal ofMachine Learning Research, 6:695–709, 2005.

[30] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7): 1661–1674, 2011.

[31] Michael F. Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics – Simulation and Computation, 18(3):1059–1076, 1989.

## A<sub>pp</sub>endix A

Here we provide the proof of Lemma 1. We rewrite Equation (6) for the unnormalized without all of the explicit x and t dependencies:

$$
\begin{array} { r } { \partial _ { t } p = - \langle \nabla , p v \rangle + p g . } \end{array}
$$

Now, call the corresponding partition function (i.e. normalization factor) of the unnormalized distribution $Z ( t )$ , and call the normalized version of the distribution ${ \widetilde { p } } .$ In this case,

$$
\widetilde { p } ( x , t ) = \frac { p ( x , t ) } { Z ( t ) } \quad \Rightarrow \quad p ( x , t ) = Z ( t ) \widetilde { p } ( x , t ) .
$$

Plugging this into our equation gives $\partial _ { t } ( Z \tilde { p } ) = - \langle \nabla , Z \tilde { p } v \rangle + Z \tilde { p } g$ , and therefore

$$
\begin{array} { r } { Z \partial _ { t } \widetilde { p } + \widetilde { p } \partial _ { t } Z = - Z \langle \nabla , \widetilde { p } v \rangle + Z \widetilde { p } g \quad \mathrm { a n d } \quad \partial _ { t } \widetilde { p } = - \langle \nabla , \widetilde { p } v \rangle + \widetilde { p } \left( g - \frac { 1 } { Z } \partial _ { t } Z \right) . } \end{array}
$$

Now, using the fact that $\textstyle Z ( t ) = \int p ( x , t ) \mathrm { d } x$ , we have that

$$
\frac { \partial _ { t } Z } { Z } = \frac { 1 } { Z } \int ( \partial _ { t } p ) ( x , t ) \mathrm { d } x = \frac { 1 } { Z } \int \big ( - \langle \nabla , p v \rangle + p g \big ) \mathrm { d } x = - \int \langle \nabla , \widetilde { p } v \rangle \mathrm { d } x + \int \widetilde { p } g \mathrm { d } x .
$$

Note that due to the Divergence Theorem and the fact that $\widetilde { p } ( x , t )  0 \mathrm { a s } x  \infty .$ , the first term is 0. Thus, we have that

$$
\frac { \partial _ { t } Z } { Z } = \int \widetilde { p } ( x , t ) g ( x , t ) \mathrm { d } x ,
$$

which overall gives

$$
\begin{array} { r } { g ( x , t ) - \frac { 1 } { Z } \partial _ { t } Z = g ( x , t ) - \displaystyle \int \widetilde { p } ( x , t ) g ( x , t ) { \mathrm { d } } x = : \overline { { g } } ( x , t ) . } \end{array}
$$

Plugging this into the equation for $\partial _ { t } \widetilde { p }$ gives

$$
\begin{array} { r } { \partial _ { t } \widetilde { p } = - \langle \nabla , \widetilde { p } v \rangle + \widetilde { p } \overline { { g } } . } \end{array}
$$

However, we know that such an equation for the normalized distribution may be sampled from via the weighted SDE (7). This result is derived in Section 2 of [16].