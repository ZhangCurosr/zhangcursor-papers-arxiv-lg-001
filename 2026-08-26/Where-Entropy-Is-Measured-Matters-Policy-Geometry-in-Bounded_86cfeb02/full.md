# Where Entropy Is Measured Matters: Policy Geometry in Bounded Continuous-Control PPO

Yiyang He<sup>∗</sup> Department of Engineering, Lancaster University, Lancaster, UK

Zhichun Zhou<sup>∗</sup> Department of Engineering, Beihang University, Beijing, China

Ziwei Wang Department of Engineering, Lancaster University, Lancaster, UK

y.he20@lancaster.ac.uk

Department of Automation, Tsinghua University, Beijing, China

Haolin Fei

Department of Engineering, Lancaster University, Lancaster, UK

zy2332222@buaa.edu.cn

z.wang82@lancaster.ac.uk

xuetao16@mail.tsinghua.edu.cn

haolin.fei@ieee.org

## Abstract

Many continuous-control policies are optimized as unbounded Gaussians and then mapped into bounded actions. We show that the space in which entropy is measured changes the geometry that proximal policy optimization (PPO) learns. The study begins with an 80- muscle MyoLeg walking task, where a clipped Gaussian executes 89.07% of actions within 5% of a bound. A counterfactual decomposition on the same visited states shows that this is not an efect of variance alone. Setting the variance to zero still leaves 83.83% of actions near a bound, centering the mean while keeping the learned variance gives 77.51%, and 82.12% of state-conditioned means lie outside the executable interval. Replacing clipping by a smooth tanh map does not remove the high-variance regime. For a latent Gaussian, the latent entropy $H ( u )$ has zero gradient with respect to the mean and a constant variance-increasing gradient of the entropy loss. For the executed-action entropy H(a), the transformation Jacobian adds $2 c _ { \mathrm { e n t } } \mathbb { E } [ \operatorname { t a n h } u _ { i } ]$ to the entropy-loss gradient on the mean; gradient descent therefore pulls the mean inward. Across three matched MyoLeg seeds, total near-boundary occupancy is 71.42%, 29.76%, and 18.83% under latent entropy, no entropy, and executed-action entropy. No entropy gives the lowest occupancy when only variance is retained, at the 5% reference margin, yet $H ( a )$ produces the most interior policy mean, so low learned dispersion is not equivalent to interior policy geometry. A 38-dimensional Dog-Stand replication with an independent, CleanRL-based PPO implementation reproduces the ordering in mean geometry, which also survives evaluation on shared states and boundary margins from $1 \%$ to 10%. Direct mean penalties reproduce or exceed the centering produced by $H ( a )$ , so interior means are not unique to executed entropy. However, an L2-regularized policy and $H ( a )$ reach nearly identical mean geometry while difering by 0.600 in mean log standard deviation and by 129 return points in Dog-Stand. Entropy measurement space is therefore a coupled mean–variance design choice, and task return alone does not characterize the geometry of a bounded policy.

## 1 Introduction

Continuous-control reinforcement learning often uses a Gaussian policy even when every actuator is bounded. The policy therefore lives in two spaces. In the latent space, an action is sampled from an unbounded distribution. In the executed space, that sample is clipped or transformed before it reaches the simulator or robot. This distinction is easy to ignore when return is the only evaluation metric. It becomes important when the learned controller spends much of its time near the action limits.

We encountered this problem in an 80-muscle MyoLeg walking task. An of-the-shelf PPO configuration produced high-return locomotion, yet the muscle policy was strongly boundary dominated: roughly 89% of executed muscle commands were within 5% of either 0 or 1. A 14-actuator motor policy on the same model showed only about 9.5% near-boundary occupancy, measured with the same absolute 0.05 margin from either bound (Section 3). The immediate explanations were physical: perhaps muscle activation filtering, asymmetric activation/deactivation time constants, or the non-negative muscle action range caused the switching. They did not. Removing the activation filter or making the time constants symmetric left the regime essentially unchanged.

The more important question is statistical: is the policy near the bounds because its variance is large, or because its state-conditioned mean has moved there? We answer this with a same-state decomposition. In the trained clipped-Gaussian muscle policy, setting the variance to zero still leaves 83.8% near-boundary occupancy, and 82.1% of state-conditioned means lie outside the executable interval. Large dispersion is also suficient, but it is not the whole explanation. The policy mean itself is strongly extruded. We use mean extrusion to denote this movement of a state-conditioned latent mean into regions that the bounded execution map sends close to an action limit.

This observation points to the entropy term. For a diagonal latent Gaussian $u \sim \mathcal N ( \mu , \sigma ^ { 2 } )$ ，

$$
H ( u ) = \sum _ { i } \left( \log \sigma _ { i } + \textstyle { \frac { 1 } { 2 } } \log ( 2 \pi e ) \right) ,\tag{1}
$$

so latent entropy has no direct gradient on the mean and gives the same variance-increasing entropy-loss gradient in every dimension. If entropy is measured after a bounded transform, the change-of-variables Jacobian adds a geometric term. For the tanh map, this term pulls the mean inward and changes the variance gradient from a constant into a distribution-dependent quantity.

We test this mechanism with three matched tanh policies: latent entropy $H ( u )$ , no entropy, and executedaction entropy $H ( a )$ . The intervention changes only the entropy objective; sampling and the PPO importance ratio remain in the same latent Gaussian family. Across three MyoLeg seeds, the three policies converge to sharply diferent mean–variance regimes. Latent entropy has the largest dispersion and the most extruded means. Removing entropy drives dispersion much lower but leaves the mean substantially closer to the bounds than $H ( a )$ . Thus the policy with the smallest exploration-driven boundary occupancy does not have the smallest total boundary occupancy.

We then repeat the experiment on the 38-dimensional DeepMind Control Dog-Stand task using an independent, CleanRL-based PPO implementation. The same ordering in mean geometry appears across all three seeds and three late checkpoints. It also survives two robustness checks that target natural alternative explanations: every policy is scored on shared state bufers, and the near-boundary margin is swept from 1% to 10%. The same Dog study also changes the interpretation of $H ( a )$ . Two direct mean penalties produce equally or more interior means. In particular, a pre-tanh L2 penalty nearly matches the trained mean distribution of $H ( a )$ yet the two policies end in very diferent variance and return regimes. Executed-action entropy is therefore not the only route to an interior mean; it is a coupled mean–variance regularizer.

The contribution is mechanistic rather than a new entropy formula. Tanh-transformed densities and Jacobiancorrected log probabilities are standard, including in Soft Actor-Critic (SAC) (Haarnoja et al., 2018). Our contribution is to connect the choice of entropy measurement space to trained PPO policy geometry, separate mean placement from dispersion experimentally, and test the resulting mechanism with direct gradients and an independent external replication.

The main contributions are:

• A counterfactual decomposition evaluated on the same states, showing that mean placement and dispersion are each suficient to produce high boundary occupancy in trained 80-muscle PPO, and that the policy mean itself is strongly extruded.

• Controlled ablations showing that muscle activation filtering, activation/deactivation asymmetry, and hard clipping are not necessary explanations for the boundary-dominated regime.

• A three-condition intervention on entropy measurement space, replicated across three MyoLeg seeds, together with analytic gradients and measurements on frozen bufers that directly distinguish latent from executed entropy.

• An independent 38-dimensional Dog-Stand replication under CleanRL PPO, followed by an audit on shared states and a sweep of boundary margins from 1% to 10%.

• Controls with direct penalties on the policy mean, showing that interior means are not unique to H(a), while closely matched mean geometry can coexist with very diferent variance and performance regimes.

Figure 1 summarizes this chain of reasoning.  
![](images/c2c3a62cc42359d53cac166c2820a9b61e9bf7143d2bfef50f7fb3b6a904764d.jpg)  
Figure 1: Study logic. The paper starts from an actuator-level phenomenon, separates mean placement from dispersion, changes only the space in which entropy is measured, measures the resulting gradients, and then tests the mechanism in an independent PPO implementation.

## 2 Related Work

## 2.1 Bounded continuous-control policies

The mismatch between an unbounded Gaussian and a bounded action space is well known. Beta policies place probability directly on a bounded interval (Chou et al., 2017), including prior PPO-specific evaluations of Beta policies on bounded action spaces (Petrazzini & Antonelo, 2021). Clipped-action policy gradient corrects the policy-gradient treatment of clipping (Fujita & Maeda, 2018), marginal policy gradients unify a family of estimators for clipped and directional bounded-action policies (Eisenach et al., 2019), and tanh-squashed Gaussians are standard in SAC (Haarnoja et al., 2018). These methods establish that action parameterization matters. Our question is diferent: when PPO uses a latent Gaussian and a bounded execution map, how do the entropy term, mean placement, and variance jointly shape the trained controller?

SAC is particularly relevant because its squashed-Gaussian actor evaluates the transformed action density, including the tanh Jacobian (Haarnoja et al., 2018). In the terminology of this paper, its entropy term is evaluated in the executed action space. We do not claim the change-of-variables identity or H(a) as new. We use the identity to expose a geometric consequence of a common PPO convention: evaluating the entropy of the latent Gaussian before the action is bounded. The convention is common but not universal: the Brax training stack estimates its PPO entropy bonus on the tanh-squashed action, including the change-of-variables correction (Freeman et al., 2021).

Boundary-preferring policies can also perform well. Seyde et al. (2021) showed that bang-bang and Bernoulli policies solve a range of continuous-control tasks. That result makes return an incomplete description of a controller. We focus on how a Gaussian PPO policy acquires boundary-heavy geometry rather than on whether boundary control can be optimal for a task. From a similar starting point, Lin (2025) argues that squashing a Gaussian into a bounded interval distorts the geometry of the action space and creates gradient pathologies near the saturated boundaries, and responds by replacing the distribution itself with a spherical direction–concentration parameterization. We keep the squashed-Gaussian parameterization fixed instead and intervene only on the space in which the PPO entropy bonus is measured; the same-state decomposition then attributes the boundary-heavy executed distribution to mean placement and dispersion separately.

## 2.2 Entropy, variance, and structured exploration

PPO (Schulman et al., 2017) often uses an entropy bonus to maintain stochasticity, a device that dates to REINFORCE-style policy search (Williams & Peng, 1991) and became standard practice through asynchronous actor–critic methods (Mnih et al., 2016). A separate line of work treats entropy not as a bonus but as part of the objective itself, in the maximum-entropy formulation (Ziebart et al., 2008; Haarnoja et al., 2017) of which SAC is the continuous-control instance (Haarnoja et al., 2018). Large empirical studies show that implementation choices such as state-independent versus state-dependent standard deviations can materially afect on-policy learning (Andrychowicz et al., 2021). Code-level details of PPO can likewise dominate its reported advantage over alternatives (Engstrom et al., 2020). Our result is of that kind: the choice of coordinate system for the entropy term is an implementation convention with a specific, measurable geometric consequence. Zhou et al. (2026) study Gaussian variance in PPO and propose an explicit variancemanagement procedure. Other work changes the structure of exploration rather than only its margina variance. Generalized state-dependent exploration (gSDE) introduces state-dependent, temporally persistent exploration for robotic control (Rafin et al., 2022). Pink-noise exploration introduces temporal correlation in continuous-action noise (Eberhard et al., 2023), and Lattice adds structured correlation in time and actuator space for overactuated systems (Chiappa et al., 2023). Our experiments hold these choices fixed and isolate a diferent variable: whether the entropy objective is evaluated before or after the bounded action transform.

## 2.3 Action regularization and smooth control

Action regularization provides another route to changing controller geometry. CAPS explicitly regularizes the policy to produce smoother state-to-action mappings and control sequences (Mysore et al., 2021). Our tanhsquared and pre-tanh L2 penalties are not proposed as new smooth-control methods. They are deliberately simple, per-state mean regularizers used as mechanism controls: they test whether the mean-centering efect of executed entropy can be reproduced without changing the entropy definition. Unlike CAPS, they do not penalize temporal action diferences or local state-to-action sensitivity.

## 2.4 Musculoskeletal control

MyoSuite and related neuromusculoskeletal benchmarks expose many bounded muscle activations on one articulated body (Caggiano et al., 2022; Kidziński et al., 2018; Wang et al., 2025). Prior work addresses exploration and control in these high-dimensional systems with specialized exploration and control architectures (Schumacher et al., 2023; Chiappa et al., 2023; Lee et al., 2019). Muscle activation dynamics are also known to afect optimized trajectories (Anderson & Pandy, 2001; van den Bogert, 2025). We therefore test the muscle dynamics directly before attributing the observed boundary behavior to the RL objective.

## 3 Policy Geometry in a Bounded Action Space

## 3.1 Latent and executed actions

Let PPO produce a diagonal Gaussian latent action

$$
\begin{array} { r } { u = \mu _ { \theta } ( s ) + \sigma \odot \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{2}
$$

with state-independent log σ in the implementations studied here. The environment executes $a = f ( u )$ . We use either hard clipping or a tanh map. For the muscle action interval [0, 1],

$$
a = { \frac { \operatorname { t a n h } u + 1 } { 2 } } .\tag{3}
$$

A 5% near-boundary margin, $a \leq 0 . 0 5$ or $a \ge 0 . 9 5$ , is equivalent to

$$
\begin{array} { r l } & { | \operatorname { t a n h } u | \geq 0 . 9 \Longleftrightarrow | u | \geq u ^ { \star } , } \\ & { \qquad u ^ { \star } = \operatorname { a t a n h } ( 0 . 9 ) = 1 . 4 7 2 2 1 9 4 8 9 6 . } \end{array}\tag{4}
$$

Dog-Stand uses the normalized coordinate $y = \operatorname { t a n h } u \in ( - 1 , 1 )$ and the same criterion $| y | \geq 0 . 9$

Two margin conventions appear in this paper and are not interchangeable on a symmetric action box. The clipped-Gaussian battery of Section 5.1 measures an absolute distance of 0.05 from either bound of the configuration’s own box, which is 5% of the muscle range [0, 1] and 2.5% of the motor range [−1, 1]. Every tanh and Dog-Stand quantity instead uses a fraction of the action range, giving $| y | \geq 1 - 2 m$ with $m = 0 . 0 5$ by default. The two coincide on [0, 1], so all muscle numbers are directly comparable. The motor reference in Table 1 is the one row measured at a stricter relative threshold; it is reported this way because it is the interior reference rather than part of the mechanism comparison.

## 3.2 Same-state mean–variance decomposition

Near-boundary occupancy can come from the mean, the variance, or both. We evaluate four counterfactual cells on the same visited states:

$$
P _ { 1 1 } : ~ ( \mu , \sigma ) \mathrm { a c t u a l } ,\tag{5}
$$

$$
P _ { 1 0 } : ( \mu , 0 ) \mathrm { m e a n } \mathrm { o n l y } ,\tag{6}
$$

$$
P _ { 0 1 } : ( \mu _ { c } , \sigma ) \mathrm { v a r i a n c e o n l y } ,\tag{7}
$$

$$
P _ { 0 0 } : ( \mu _ { c } , 0 ) \mathrm { c e n t e r e d b a s e l i n e } ,\tag{8}
$$

where $\mu _ { c } = 0 . 5$ for clipping to [0, 1] and $\mu _ { c } = 0$ for the symmetric tanh latent coordinate. For tanh,

$$
P _ { 1 1 } = \Phi \left( \frac { - u ^ { \star } - \mu } { \sigma } \right) + 1 - \Phi \left( \frac { u ^ { \star } - \mu } { \sigma } \right) ,\tag{9}
$$

computed componentwise and averaged over states and action dimensions. The mean-only cell is ${ \bf 1 } ( | \mu | \geq u ^ { \star } )$ The variance-only cell uses the same formula with $\mu = 0$

For clipping to [0, 1], the near-boundary thresholds are 0.05 and 0.95. We separately report the exact atom mass, which uses thresholds 0 and 1. The interaction

$$
I = P _ { 1 1 } - P _ { 1 0 } - P _ { 0 1 } + P _ { 0 0 }\tag{10}
$$

is large and negative when mean and variance are redundant under the 100% ceiling. We therefore report the four cells directly rather than treating an additive attribution as unique.

## 3.3 Latent and executed entropy

For the latent Gaussian,

$$
\frac { \partial H ( u ) } { \partial \mu _ { i } } = 0 , \qquad \frac { \partial H ( u ) } { \partial \log \sigma _ { i } } = 1 .\tag{11}
$$

With entropy loss $L _ { \mathrm { e n t } } = - c _ { \mathrm { e n t } } H$ , latent entropy contributes

$$
\frac { \partial { \cal L } _ { \mathrm { e n t } } } { \partial \log \sigma _ { i } } = - c _ { \mathrm { e n t } }\tag{12}
$$

for every action dimension, independent of the current variance. It contributes no direct mean force.

For an invertible diferentiable transform $a = f ( u )$ ，

$$
H ( a ) = H ( u ) + \mathbb { E } _ { u } [ \log | \operatorname* { d e t } J _ { f } ( u ) | ] .\tag{13}
$$

For the componentwise map $a _ { i } = ( \operatorname { t a n h } { u _ { i } } + 1 ) / 2$ , reparameterization gives

$$
\frac { \partial H ( a ) } { \partial \mu _ { i } } = - 2 \operatorname { \mathbb { E } } [ \operatorname { t a n h } u _ { i } ] ,\tag{14}
$$

$$
\frac { \partial H ( a ) } { \partial \log \sigma _ { i } } = 1 - 2 \sigma _ { i } ^ { 2 } \mathbb { E } [ \operatorname { s e c h } ^ { 2 } u _ { i } ] .\tag{15}
$$

Thus the gradient of the entropy loss with respect to the mean is $+ 2 c _ { \mathrm { e n t } } \mathbb { E } [ \operatorname { t a n h } u _ { i } ]$ . Gradient descent moves positive means down and negative means up: the direct efect is inward. The variance term is finite and distribution dependent rather than constant.

The bounded transform does not need to appear in the PPO importance ratio. At a stored latent sample $u ,$

$$
\frac { \pi _ { \mathrm { n e w } } ( a ) } { \pi _ { \mathrm { o l d } } ( a ) } = \frac { p _ { \mathrm { n e w } } ( u ) / | \operatorname* { d e t } J _ { f } ( u ) | } { p _ { \mathrm { o l d } } ( u ) / | \operatorname* { d e t } J _ { f } ( u ) | } = \frac { p _ { \mathrm { n e w } } ( u ) } { p _ { \mathrm { o l d } } ( u ) } .\tag{16}
$$

For the ideal invertible tanh transform, the Jacobian cancels exactly at the stored latent samples. Our $H ( a )$ intervention therefore changes the entropy calculation but not the latent Gaussian sampling rule or $\mathrm { P P O }$ ratio; the MyoLeg implementation adds an inert ±10 latent interface clip, quantified in Section 4. Any fixed afine rescaling applied after the bounded map changes $H ( a )$ only by an additive constant that is independent of the policy parameters, so it leaves every entropy gradient unchanged.

## 4 Experimental Design

## 4.1 MyoLeg task and PPO training

The primary task is a hybrid MyoLeg model based on MyoSuite (Caggiano et al., 2022) and MuJoCo (Todorov et al., 2012). The model contains 80 muscle actuators and 14 motor actuators on the same skeleton. Muscle commands lie in $[ 0 , 1 ] ^ { 8 0 }$ ; motor commands lie in $[ - 1 , 1 ] ^ { 1 4 }$ . The observation has 101 dimensions. The physics timestep is 1 ms and the environment applies a frame skip of 4, so the control frequency is 250 Hz; both numbers were read back from the live environment and are recorded in the release manifest. Episodes are limited to 1000 control steps and can terminate early on a pelvis-height or orientation violation.

The reference motion is CMU clip 02\_01.c3d (Carnegie Mellon University Graphics Lab, 2003), with a 133-frame gait cycle at 120 Hz and a walking speed of approximately 1.187 $\mathrm { m } / \mathrm { s } .$ . The implemented reward is

$$
\begin{array} { l } { { r = 2 0 r _ { \mathrm { e e } } \left( 0 . 7 5 r _ { q } + 0 . 1 0 r _ { v } \right) , } } \\ { { \displaystyle r _ { q } = \exp \left[ - 0 . 8 5 7 \sum _ { j = 1 } ^ { 1 4 } ( q _ { j } ^ { \star } - q _ { j } ) ^ { 2 } \right] , } } \\ { { \displaystyle r _ { \mathrm { e e } } = \exp \left[ - 4 0 \sum _ { k } ( e _ { k } ^ { \star } - e _ { k } ) ^ { 2 } \right] , \qquad r _ { v } \equiv 1 . } } \end{array}\tag{17}
$$

The velocity subreward is constant because the environment implementation sets the velocity-diference variable used by that term to zero. Thus it contributes no velocity-tracking information; it contributes $2 0 \times 0 . 1 0 r _ { \mathrm { e e } } = 2 r _ { \mathrm { e e } }$ to the scalar reward, and the per-step reward ceiling is $2 0 ( 0 . 7 5 + 0 . 1 0 ) = 1 7$ when the tracking terms equal one. This is an environment-definition quirk rather than an experimental condition. It can afect the task being learned, but the same reward code is used in every MyoLeg entropy condition and therefore cannot explain their matched between-condition diferences.

The MyoLeg PPO experiments use Stable-Baselines3 (Rafin et al., 2021). The primary tanh comparison uses separate 512–512 ReLU policy and value networks, learning rate $1 0 ^ { - 4 }$ , rollout length 2048 per environment, minibatch size 128, 10 epochs, $\gamma = 0 . 9 9$ , generalized advantage estimation (Schulman et al., 2016) with $\lambda = 0 . 9 9$ , clip range 0.2, and 12 parallel environments. Both implementations optimize with Adam (Kingma & Ba, 2015). The latent action interface is widened to $[ - 1 0 , 1 0 ] ^ { 8 0 }$ ; clipping can still occur in the saturated tails under $H ( u )$ , but its efect on the executed action is numerically negligible, as quantified next. Measured on canonical late-training rollouts, a Gaussian sample exceeds the box in $3 . 9 \% \pm 0 . 7 \%$ of components, and at least one of the 80 components exceeds it on $9 5 . 9 \% \pm 2 . 5 \%$ of control steps $( n = 3$ seeds; analytic tail probabilities and empirical counts agree to three decimals). With no entropy the component-level rate is at most $1 . 1 \times 1 0 ^ { - 9 }$ in any seed, and under $H ( a )$ it is zero at double precision; in particular, the one condition whose entropy objective contains the transformation Jacobian never evaluates it at a clipped sample. Where the clip does occur it is inert for two reasons. First, Stable-Baselines3 clips only the copy of the action passed to the environment; the rollout bufer stores the unclipped latent sample, so the importance ratio and every entropy quantity are computed exactly as analyzed. Second, tanh $1 0 = 1 - 4 . 1 \times 1 0 ^ { - 9 }$ , so a clipped component’s executed activation difers from the unclipped ideal by at most $2 . 1 \times 1 0 ^ { - 9 }$ on the [0, 1] scale, four orders of magnitude below every threshold used in this paper. All three primary conditions are trained for 145,022,976 environment steps with training seeds 0, 1, and 2. Appendix D gives the full training and network configuration for both implementations.

The three entropy conditions are:

1. Latent entropy $H ( u ) \colon c _ { \mathrm { e n t } } = 1 0 ^ { - 3 }$ and the ordinary Gaussian entropy.

2. No entropy: $c _ { \mathrm { e n t } } = 0 .$

3. Executed-action entropy $H ( a ) \colon c _ { \mathrm { e n t } } = 1 0 ^ { - 3 }$ and Eq. (13), estimated with four antithetic reparameterized samples per state.

The latent Gaussian family and PPO ratio are otherwise identical.

Before the tanh intervention, five configurations isolate physical and distributional explanations: a 14-actuator motor policy; an 80-muscle activation-override policy; the native muscle activation filter; a symmetric activation-time-constant variant; and a bounded Beta policy (Chou et al., 2017). These experiments use three training seeds each and matched canonical evaluation. Their role is diagnostic: they establish which ingredients are not necessary for the original clipped-Gaussian regime.

## 4.2 Dog-Stand external replication

The external task is dm\_control/dog-stand-v0 (Tunyasuvunakool et al., 2020), accessed through the Gymnasium API (Towers et al., 2024) via the Shimmy compatibility layer (Tai et al., 2023), with a 223-dimensional observation and a 38-dimensional bounded action. We start from CleanRL’s single-file continuous-action PPO implementation (Huang et al., 2022). For the three primary entropy conditions, the action transform, entropy objective, and diagnostic logging are the only algorithmic changes. The clipped surrogate, generalized advantage estimation (Schulman et al., 2016), value loss, advantage normalization, minibatching, Adam optimizer (Kingma & Ba, 2015), and the remaining hyperparameters retain the CleanRL configuration listed in Appendix D. The controls with direct mean penalties, introduced after the primary experiment, add the stated actor-loss penalty and otherwise use the same configuration.

The policy samples $u \sim \mathcal { N } ( \mu , \sigma )$ , executes $y = \operatorname { t a n h } { u } .$ and then applies the environment’s afine action scaling. The three primary entropy conditions use the same $c _ { \mathrm { e n t } }$ values as MyoLeg without tuning. Each condition is trained from scratch for 3,000,320 steps with seeds 0, 1, and 2. Dog-Stand episodes last 15 s, or 1000 control steps at 0.015 s per step (66.67 Hz).

The canonical evaluator runs 30 stochastic and 30 deterministic episodes. Reset seeds and action-noise seeds are paired across policies; the checkpoint’s observation normalizer is restored and frozen; returns are raw environment returns. Same-state $P _ { 1 1 } , P _ { 1 0 }$ , and $P _ { 0 1 }$ are evaluated on states visited by the stochastic policy. Deterministic-rollout occupancy is reported separately because deterministic execution visits a diferent state distribution.

Three geometry criteria were specified before the final Dog runs. First, the late-training $H ( u )$ variance should remain above no entropy and $H ( a )$ . Second, $H ( a )$ should reduce mean extrusion relative to both alternatives. Third, no entropy should have lower $P _ { 0 1 }$ than $H ( a )$ while retaining higher total occupancy. Return was not a criterion.

## 4.3 Direct mean regularization controls

After the primary Dog experiment, we added two controls to ask whether $H ( a )$ is simply a complicated way to center the mean:

$$
L _ { \mathrm { t a n h } ^ { 2 } } = \frac { \lambda _ { \mathrm { t a n h } ^ { 2 } } } { 2 } \mathbb { E } _ { s } \left[ \sum _ { i } \operatorname { t a n h } ^ { 2 } \mu _ { i } ( s ) \right] ,\tag{18}
$$

$$
L _ { \mu ^ { 2 } } = \frac { \lambda _ { \mu ^ { 2 } } } 2 \mathbb { E } _ { s } \left[ \sum _ { i } \mu _ { i } ( s ) ^ { 2 } \right] .\tag{19}
$$

The batch operation is a mean over states and a sum over action dimensions. Both are trained with no entropy. Their coeficients are fixed by matching the magnitude of the $H ( a )$ entropy-loss gradient with respect to the mean, taken in the zero-dispersion limit at the 5% boundary threshold:

$$
2 c _ { \mathrm { e n t } } \operatorname { t a n h } { u ^ { \star } } = 1 . 8 \times 1 0 ^ { - 3 } ,\tag{20}
$$

which gives $\lambda _ { \mathrm { t a n h } ^ { 2 } } = 0 . 0 1 0 5 2 6 3 1 5 8$ and $\lambda _ { \mu ^ { 2 } } = 0 . 0 0 1 2 2 2 6 4 3 8$ . Each control is trained for 3,000,320 steps with the same three Dog seeds and evaluated with the same canonical protocol.

## 4.4 Gradient diagnostics

For MyoLeg, frozen-gradient decomposition is performed on the late-training seed-0 checkpoint of each entropy condition using 10 independently collected on-policy bufers per checkpoint. We decompose the actor loss into the PPO surrogate and entropy terms, separately for $\mu$ and log σ. Bufer fingerprints are checked for uniqueness before cross-bufer statistics are computed. The entropy-side mean force is summarized by $\langle g _ { \mu , \mathrm { e n t } } , \mu \rangle$ and, when the gradient is nonzero, by its cosine with $\mu .$ Under $H ( u )$ and no entropy the entropy mean-gradient vector is exactly zero, so the cosine is undefined rather than zero.

For Dog, the inward mean mechanism is evaluated on 30 independent bufers per late-training checkpoint and seed. The terminal log-variance decomposition is also measured, but its confidence intervals span zero under all three primary conditions; we therefore use it only to bound the interpretation of the long-horizon variance trajectories.

## 4.5 Robustness analyses

The common-state audit tests whether the ordering in mean geometry is caused by the policies visiting diferent states. For each Dog seed, all five policies donate 10 episodes of raw pre-normalization states. Every trained policy is scored on every donor set and on an equal-mixture pool. Each evaluated policy applies its own frozen observation normalizer before the forward pass.

The boundary-margin analysis recomputes $P _ { 1 1 } , P _ { 1 0 } $ , and $P _ { 0 1 }$ analytically at margins of 1%, 2.5%, 5%, and 10% on one freshly collected 30-episode on-policy state set per checkpoint. All four thresholds use the same states within a sweep. Both tasks use the same NumPy action-noise convention as their frozen canonica evaluators, and both reproduce the corresponding 5% cells of Tables 2 and 3 exactly. Appendix A reports the sweep values.

## 4.6 Statistical units

The training seed is the unit for method-level comparisons. Tables report the mean and sample standard deviation over three trained policies. Evaluation episodes characterize a fixed policy and are paired by reset and action-noise seeds where applicable; they are not treated as independent training replicates. Frozen gradient bufers are the unit only for local gradient estimation. With three training seeds we emphasize efect sizes and direction agreement rather than small-sample p-values. In particular, Dog return comparisons are descriptive: we report the three-seed mean, sample SD, and matched-seed direction, but make no population-level superiority claim from $n = 3$

## 4.7 Code and evaluation artifacts

The training and analysis code should be treated as part of the experimental specification. An archival code package accompanies this paper, containing the MyoLeg/SB3 and Dog/CleanRL training implementations, the $H ( a )$ estimator and its numerical checks, and the canonical evaluators used for the reported tables. The package also includes the audit on shared states, the sweep over boundary margins, gradient diagnostics, frozen result JSONs, and manifests with checkpoint and script SHA-256 hashes. The same tagged release, together with the Dog checkpoints shipped in the package and the hashed MyoLeg checkpoints, will be maintained in a permanent public repository. The release will also include an environment lockfile and system-version manifest; exact package versions are not inferred from checkpoint files. Third-party model and motion assets will be redistributed only where their licenses permit, otherwise the release will provide acquisition and checksum instructions. The release manifest identifies the canonical evaluator that is the source of record for each table rather than asking readers to reproduce results with ad-hoc rollout scripts.

## 5 Results

## 5.1 The original boundary regime is not a muscle-dynamics efect

Table 1 summarizes the initial mechanism-isolation experiments. The motor policy remains mostly interior, whereas all three clipped-Gaussian muscle variants spend roughly 87–89% of their actions near a bound. Bypassing activation dynamics does not remove the regime, and making activation and deactivation time constants symmetric does not remove it either. The bounded Beta policy has much lower boundary occupancy and competitive stochastic return, showing that exact clipping atoms are not required for successful control on the task.

## 5.2 Mean extrusion and dispersion are both suficient

Figure 2B separates the two sources of boundary occupancy on the same Filter states. The actual policy gives $P _ { 1 1 } = 8 9 . 0 6 5 \% \pm 0 . 1 0 2 \%$ . Setting the variance to zero still gives $P _ { 1 0 } = 8 3 . 8 2 5 \% \pm 1 . 1 0 3 \%$ . Centering the mean while retaining the learned variance gives $P _ { 0 1 } = 7 7 . 5 1 5 \% \pm 0 . 4 2 8 \%$ . The interaction is about −72 pp, so the two causes are strongly redundant near the ceiling.

The mean is not merely close to the interval endpoints. Across the three Filter seeds, $8 2 . 1 2 \% \pm 1 . 2 6 \%$ of state-conditioned means lie outside [0, 1]. Consequently, reducing exploration variance alone cannot make this controller interior. Hard clipping adds exact atoms, but the latent policy geometry is already boundary oriented before clipping.

Table 1: Initial MyoLeg mechanism-isolation configurations. Values are mean ± sample SD over three training seeds. Near-boundary occupancy uses an absolute margin of 0.05 from either bound of the configuration’s own action box, which is 5% of the muscle range [0, 1] but 2.5% of the motor range $[ - 1 , 1 ]$ ; the motor row is therefore measured at a stricter relative threshold than the muscle rows, and a like-for-like 5%-of-range motor threshold is $| a | \geq 0 . 9 0 ;$ the same canonical protocol gives $1 0 . 1 2 \% \pm 0 . 0 6 \%$ at that threshold. All five configurations use the same entropy coeficient $c _ { \mathrm { e n t } } = 1 0 ^ { - 3 }$ : the four Gaussian configurations on their latent Gaussian entropy $H ( u )$ , and the Beta control on the entropy of its Beta distribution. The motor reference uses its prescribed 78,000,000-step budget. For the Beta policy, log disp. is an efective log-dispersion: the state-averaged log of the per-dimension Beta standard deviation implied by the learned concentrations. Beta uses an efective log-dispersion and is not numerically identical to Gaussian log σ.
<table><tr><td>Condition</td><td> $R _ { \mathrm { s t o c h } }$ </td><td>Near bound (%)</td><td>log disp.</td></tr><tr><td>Motor, 14D</td><td> $1 2 6 3 7 \pm 9 4 1$ </td><td></td><td> $9 . 4 9 \pm 0 . 1 0 - 1 . 5 9 4 \pm 0 . 0 3 0$ </td></tr><tr><td>Override, 80D</td><td> $1 3 4 3 7 \pm 7 3 0$ </td><td></td><td> $8 7 . 3 8 \pm 0 . 2 9 + 0 . 8 3 2 \pm 0 . 0 3 7$ </td></tr><tr><td>Filter, 80D</td><td> $1 2 3 8 9 \pm 4 3 5$ </td><td></td><td> $8 9 . 0 7 \pm 0 . 1 6 + 0 . 7 9 8 \pm 0 . 0 2 9$ </td></tr><tr><td>Symmetric τ, 80D</td><td> $1 2 0 7 2 \pm 2 5 0$ </td><td></td><td> $8 8 . 7 3 \pm 0 . 4 5 + 0 . 8 5 2 \pm 0 . 0 1 8$ </td></tr><tr><td>Beta, 80D</td><td> $1 3 6 9 8 \pm 2 9 4$ </td><td></td><td> $1 5 . 2 8 \pm 0 . 3 0 - 1 . 9 3 8 \pm 0 . 0 0 8$ </td></tr></table>

![](images/170c100c7bddf0472d3709e6ae7ec00bc23374d0c53053bf1d197b7bf3b23a44.jpg)

![](images/db34566011d8f589667d557c823ef654c2ef3b5176f5efa05794725d1316d329.jpg)

![](images/9927195a55e78642f021e5aceefabf43c41703dc7c270d19dce2c998d92f218b.jpg)  
C

82.1% of means lie outside [0, 1] and are exactly clipped; mean-only near-boundary occupancy = 83.8%  
![](images/b35490943db0a4b4c2d6d82d31a83071024c746badc5ecaa11ae139d4cc56910.jpg)  
Figure 2: The original MyoLeg phenomenon and its decomposition. A: Fixed-reset-seed traces for the first 300 control steps; no segment was selected after inspection. Motor actions are already in $[ - 1 , 1 ]$ . Muscle actions $a \in [ 0 , 1 ]$ are mapped to $y = 2 a - 1$ only for display, so $| y | \geq 0 . 9$ is the same 5% boundary margin. The 10% and 89% annotations describe these single traces, not the multi-seed canonical aggregates. The Motor annotation is therefore at the like-for-like 5%-of-range threshold $( | a | \geq 0 . 9 0 )$ noted in the Table 1 caption, not at the absolute 0.05 margin used in that table; the canonical aggregate at this matching threshold is $1 0 . 1 2 \% \pm 0 . 0 6 \%$ , so the closeness of the single-trace value to the table’s stricter-threshold 9.49% is coincidental. B: Same-state Filter decomposition at 145M steps, $n = 3$ training seeds. Bars show near-boundary occupancy: $P _ { 1 1 } = 8 9 . 0 7 \%$ $P _ { 1 0 } = 8 3 . 8 3 \%$ , and $P _ { 0 1 } = 7 7 . 5 1 \%$ . Black diamonds show exact clipping-atom mass: 87.89%, 82.12%, and 75.61%. The large negative interaction reflects overlap under the 100% ceiling. C: 82.1% of state-conditioned Filter means lie outside [0, 1] and are exactly clipped; the mean-only cell $P _ { 1 0 }$ is 83.8%.

## 5.3 Entropy measurement space separates the trained tanh policies

Hard clipping is not required for the high-variance latent regime. After replacing clipping by a smooth tanh action map, all three $H ( u )$ runs end in a much higher variance regime than their matched no-entropy and $H ( a )$ counterparts. We do not use a clip-versus-tanh slope ratio because the available cross-parameterization slope comparison is not resolved beyond seed variation.

Figure 3 follows the three matched tanh conditions over training. Under $H ( u )$ , mean log σ rises to $+ 0 . 8 8 3 5 \pm$ 0.0499. With no entropy it falls $\mathrm { t o \ - 0 . 6 9 2 0 { \pm } 0 . 0 1 1 0 }$ . Under $H ( a )$ it falls early and reaches $- 0 . 5 7 4 8 \pm 0 . 0 0 6 9$ at 145M, with little late change. These trajectories are consistent with the direct entropy gradients in Eqs. (11)– (15), while the local frozen-bufer decomposition constrains rather than fully explains the long-horizon training dynamics.

![](images/1d0acf25ec7d617648f14ed403e004f06d32dc0ed5190e7fa17843a6d421cd0c.jpg)

![](images/52b0f2859dc09e64b096a7bbb163be77a9bfcaa05240c30e4ec3621c83055415.jpg)  
Thick line and shading: three-seed mean and min-max, drawn only where all three raw training logs overlap. Thin lines: individual seeds. Open circles at 145M: terminal canonical values for all three seeds, unaffected by log truncation.  
Variance minimisation does not produce interior geometry: B no entropy has the lowest $P _ { 0 1 }$ yet a higher $P _ { 1 1 }$ than H(a)  
Decomposition at 145M C seed 0, 10 buffers

![](images/fce6a645616ef7fc7d0d498bdac0a5ca2e2ce35b82af10a0056761f5eebb4ef8.jpg)

![](images/e600722a252be2a7460f233676bd13c853f09aea6428add74437b0981a2506ff.jpg)  
Panel C: positive lowers σ, negative raises it. Vertical lines are 95% t-Cls over buffers; the buffer is the statistical unit, and these are ten buffers of one trained policy, not ten training seeds. Entropy force on the mean: under H(u) and under no entropy $g _ { \mu , \mathrm { \scriptsize ~ e n t } } \equiv 0 ,$ so its cosine with μ is undefinec rather than zero; under H(a) the cosine ${ \bar { 1 } } 5 \approx + 0 . 9 7 9$ with essentially every component inward.

Figure 3: Matched MyoLeg tanh experiment. A: Mean log σ during training. Thick lines and shading show the three-seed mean and min–max envelope over the full logged range; thin lines are individual seeds. Open circles are the three terminal 145M canonical values. A2: Separate late-training view of no entropy and $H ( a )$ . B: Late-training same-state decomposition. No entropy has lower $P _ { 0 1 }$ than $H ( a )$ but higher $P _ { 1 0 }$ and total occupancy. C: Frozen seed-0 gradient decomposition over 10 independent bufers. Positive loss gradient lowers $\sigma ;$ negative raises it. Under $H ( u )$ and no entropy, $g _ { \mu , \mathrm { e n t } } \equiv 0$ , so its cosine with $\mu$ is undefined. Under $H ( a )$ , the entropy mean-force cosine is approximately +0.979, with essentially every evaluated state–action component receiving an inward contribution.

Table 2 shows the late-training geometry. The strongest comparison is between no entropy and $H ( a )$ . No entropy has the lowest $P _ { 0 1 }$ , 1.53% versus 2.43% for $H ( a )$ . Yet its total occupancy is much higher, 29.76% versus 18.83%, because its $P _ { 1 0 }$ is 25.67% instead of 11.62%. At the 5% reference margin, the condition with the smallest exploration-driven boundary occupancy does not have the smallest total occupancy.

Table 2: Late-training MyoLeg tanh policies at 145,022,976 steps. Mean ± sample SD over three training seeds. $P _ { 1 1 }$ and $P _ { 1 0 }$ are analytic probabilities (Section 3) on states visited by the stochastic policy; $P _ { 0 1 }$ depends only on the state-independent σ and involves no state set. The realized closed-loop occupancy agrees with $P _ { 1 1 }$ within 0.03 percentage points.
<table><tr><td>Condition</td><td> $\overline { { \log \sigma } }$ </td><td> $R _ { \mathrm { s t o c h } }$ </td><td> $P _ { 1 1 }$ </td><td> $P _ { 1 0 }$ </td><td> $P _ { 0 1 }$ </td><td> $\mathbb { E } | \mu |$ </td><td> $R _ { \mathrm { d e t } }$ </td><td>Det. near bound</td></tr><tr><td> $H ( u )$ </td><td> $+ 0 . 8 8 3 5 \pm 0 . 0 4 9 9$ </td><td> $1 3 8 2 9 \pm 6 3 5$ </td><td> $7 1 . 4 2 \pm 1 . 2 9$ </td><td> $6 0 . 7 2 \pm 2 . 2 3$ </td><td> $5 1 . 8 0 \pm 1 . 4 8$ </td><td> $2 . 1 4 2 \pm 0 . 1 4 8$ </td><td> $8 9 9 \pm 1 6 3$ </td><td> $5 0 . 0 9 \pm 1 . 7 2$ </td></tr><tr><td> $\mathrm { N o \ e n t r o p y }$ </td><td> $- 0 . 6 9 2 0 \pm 0 . 0 1 1 0$ </td><td> $\mathbf { 1 6 0 1 1 } \pm \mathbf { 5 5 }$ </td><td> $2 9 . 7 6 \pm 2 . 5 4$ </td><td> $2 5 . 6 7 \pm 2 . 6 2$ </td><td> ${ \bf 1 . 5 3 \pm 0 . 2 7 }$ </td><td> $1 . 0 3 5 \pm 0 . 0 7 0$ </td><td> $\mathbf { 9 1 5 4 } \pm \mathbf { 2 4 8 0 }$ </td><td> $2 4 . 7 7 \pm 1 . 7 1$ </td></tr><tr><td> $H ( a )$ </td><td> $- 0 . 5 7 4 8 \pm 0 . 0 0 6 9$ </td><td> $1 5 5 3 9 \pm 3 7$ </td><td> ${ \bf 1 8 . 8 3 \pm 0 . 2 8 }$ </td><td> ${ \bf 1 1 . 6 2 \pm 0 . 5 6 }$ </td><td> $2 . 4 3 \pm 0 . 1 0$ </td><td> $\mathbf { 0 . 7 1 6 \pm 0 . 0 0 2 }$ </td><td> $5 0 0 1 \pm 7 8 1$ </td><td> $\mathbf { 1 0 . 3 5 \pm 1 . 0 1 }$ </td></tr></table>

The return ordering is diferent from the geometry ordering. No entropy has the highest stochastic return in all three matched MyoLeg seeds, while $H ( a )$ has the most interior mean geometry. Under deterministic execution, $H ( a )$ again has the lowest near-boundary action rate, but no entropy has the highest return in all three seeds. More interior action geometry is therefore not the same as higher task return, under either stochastic or deterministic execution.

## 5.4 The direct gradient on the mean is measurable

The analytic contrast is exact: $H ( u )$ has no entropy gradient on $\mu ,$ whereas $H ( a )$ does. The frozen MyoLeg bufers reproduce this diference. Under $H ( a )$ , the entropy contribution satisfies

$$
\langle g _ { \mu , \mathrm { e n t } } , \mu \rangle = 5 . 9 8 \times 1 0 ^ { - 4 }\tag{21}
$$

with cosine +0.979; essentially all evaluated components receive an inward entropy-gradient contribution.   
Under $H ( u )$ and no entropy the entropy gradient with respect to the mean is exactly zero.

The PPO surrogate has no comparably aligned radial direction, but its small radial component is not symmetric either. Its mean gradient has a near-zero cosine with $\mu$ and a componentwise inward fraction near one half, and its projection is condition dependent: $\langle g _ { \mu , \mathrm { s u r r } } , \mu \rangle$ averages $- 1 . 0 6 \times 1 0 ^ { - 3 }$ over the ten no-entropy bufers, with a 95% interval $\left[ - 2 . 1 0 , - 0 . 0 2 \right] \times 1 0 ^ { - 3 }$ that excludes zero, $- 4 . 6 \times 1 0 ^ { - 4 }$ under $H ( a )$ with an interval spanning zero, and $+ 1 . 5 \times 1 0 ^ { - 4 }$ under $H ( u )$ . Mean extrusion is therefore not a strong universal outward push from the surrogate; it is a weakly outward-leaning, heterogeneous training outcome that latent entropy leaves unconstrained and that the $H ( a )$ entropy term opposes with a consistently inward projection of comparable magnitude.

For log σ, the late-training seed-0 bufer means are informative but local. No entropy has a positive net loss gradient, $+ 1 . 0 1 5 \times 1 0 ^ { - 3 }$ with a 95% interval excluding zero, consistent with continued variance reduction. Under $H ( u )$ , the $\mathrm { e x a c t - 1 0 ^ { - 3 } }$ entropy term ofsets a positive surrogate contribution, giving a negative point estimate whose interval spans zero. Under $H ( a )$ , the entropy term is $- 5 . 5 2 \times 1 0 ^ { - 4 }$ and the net point estimate is near zero, again with an interval spanning zero. These local measurements support the relative MyoLeg variance regimes without implying that one late-training snapshot explains an entire training trajectory (Appendix C).

## 5.5 External replication in Dog-Stand

The Dog-Stand result reproduces the central ordering in a diferent task and a diferent PPO codebase. At 3M steps, $P _ { 1 0 }$ is 36.99% under $H ( u )$ , 22.25% with no entropy, and 1.91% under $H ( a )$ . Total occupancy follows the same order: 45.57%, 23.64%, and 5.26% (Figure 4A). The ordering holds in all three seeds and at all three audited late checkpoints—nine of nine checkpoint–seed comparisons (Figure 4B).

The Dog variance trajectories also sharpen the wording of the mechanism. $H ( u )$ ends at a much higher variance than no entropy, −0.164 versus −1.313, but its absolute variance decreases late in training. Latent entropy therefore contributes a constant variance-increasing gradient and shifts the learned variance regime upward; it does not force absolute variance to rise in every task. The terminal Dog variance-gradient intervals, measured on frozen bufers, span zero, so the final local balance is not resolved.

In contrast, the inward mean mechanism of $H ( a )$ is reproduced directly. Across the three Dog seeds, the entropy mean-force cosine is approximately +0.987 (Figure 4C). The surrogate’s projection mirrors MyoLeg:

$\langle g _ { \mu , \mathrm { s u r r } } , \mu \rangle$ is negative in every no-entropy seed $( - 4 . 3 , - 4 . 7 , - 3 . 4 \times 1 0 ^ { - 3 } ;$ ; two of the three $9 5 \%$ intervals exclude zero) and negative but unresolved under $H ( a ) \ ( - 0 . 3 , - 2 . 2 , - 0 . 3 \times 1 0 ^ { - 3 }$ , all intervals spanning zero), where the entropy term contributes $+ 3 . 0 ~ \mathrm { t o } + 3 . 5 \times 1 0 ^ { - 4 }$ inward in every seed. Under $H ( u )$ and no entropy the entropy gradient with respect to the mean is exactly zero. The main cross-task result is therefore asymmetric: the direct inward mean regularizer is cleanly reproduced, while the detailed long-horizon variance dynamics remain task dependent.

External replication: DMC Dog-Stand, 38-dim action space, independent CleanRL PPO

![](images/808e80e1acf5cc03082fa2bbc29a4c7c2e5babcf58b9c8dff0b2ded3813f45a6.jpg)

![](images/0455d812e6b8a4864cd976d52138ba2a98c5c39bc313e988b5527bbf792a51e5.jpg)

![](images/407db074931843df3266d8067079df25d8c74393e57489e5d054353bd6491ee3.jpg)  
Figure 4: External replication on 38-dimensional DMC Dog-Stand with an independent, CleanRL-based PPO implementation. A: Canonical same-state geometry at 3M steps. The variance-only diference between no entropy and $H ( a )$ is small; the large contrast is in mean placement. B: $P _ { 1 1 }$ ordering is unchanged at 2.015M, 2.519M, and 3.000M for every training seed (nine of nine checkpoint–seed comparisons). This is late-training stability of the measured geometry, not an asymptotic convergence claim. C: Direct entropy force on the mean. Under $H ( u )$ and no entropy the entropy mean-gradient vector is exactly zero and the cosine is not defined; crosses mark these zero-vector conditions. Under $H ( a )$ , the cosine is approximately +0.987 across the three seeds.

## 5.6 State visitation and the 5% threshold do not explain the primary ordering

The common-state audit scores each trained Dog policy on every policy’s raw visited states, using the evaluated policy’s own frozen observation normalizer (Appendix B). On the equal-mixture pool,

$$
\begin{array} { r l } { P _ { 1 0 } : } & { 3 9 . 6 9 \% ( H ( u ) ) > 2 6 . 9 9 \% \ \mathrm { ( n o n e ) } } \\ & { \ > 4 . 4 7 \% ( H ( a ) ) . } \end{array}\tag{22}
$$

The same ordering holds on all six evaluated state sets (the five donor sets plus the equal-mixture pool) and all three training seeds: 18 of 18 state-set–seed comparisons. The largest donor-induced span in seed-averaged $P _ { 1 0 }$ for any one policy is 6.37 percentage points, while the pooled $H ( u ) – H ( a )$ separation is 35.23 points. Policy-specific state visitation changes the magnitude but does not explain the ordering.

The same conclusion holds across boundary margins from $1 \%$ to 10%. For both Dog-Stand and MyoLeg, $P _ { 1 0 }$ and $P _ { 1 1 }$ retain the order H(u) > no entropy $> H ( a )$ at $1 \% , 2 . 5 \% , 5 \%$ , and 10% margins, for every training seed—the complete three-condition ordering holds in all 48 task × metric × margin × seed cells. The small variance-only reversal $P _ { 0 1 } ( \mathrm { n o n e } ) < P _ { 0 1 } ( H ( a ) )$ is less robust. In MyoLeg, one no-entropy seed reverses this relation at the 1% and 2.5% margins. We therefore use the $P _ { 0 1 }$ comparison only at the 5% reference margin and place the main weight on the much larger diferences in mean geometry, which are robust across thresholds.

## 5.7 Direct mean regularization reproduces centering but not the same operating regime

The direct mean penalties change an important part of the interpretation. Before running them, we expected the tanh-squared penalty to be too weak once the mean entered the tanh tail, expected both penalties to end near the no-entropy variance, and expected the L2 penalty to trade return for interior geometry. None of these working predictions was supported.

Table 3 shows the canonical five-condition comparison. The tanh-squared penalty reduces $P _ { 1 0 }$ to 0.09%, and pre-tanh L2 reduces it to 1.60%, both below $H ( a )$ at 1.91%. Interior means are therefore not unique to the intervention on entropy measurement space. The penalties also end at lower variance than no entropy, even though their raw loss has no derivative with respect to log σ. Regularizing the mean alone can indirectly change the long-horizon variance regime through the coupled PPO optimization dynamics.

Table 3: Dog-Stand canonical comparison at 3,000,320 steps. Mean ± sample SD over three training seeds. The two direct mean regularizers were added after the primary three-condition experiment as mechanistic controls.
<table><tr><td>Condition</td><td>log σ</td><td> $R _ { \mathrm { s t o c h } }$ </td><td> $R _ { \mathrm { d e t } }$ </td><td> $P _ { 1 1 }$ </td><td> $P _ { 1 0 }$ </td><td> $P _ { 0 1 }$ </td><td> $\mathbb { E } | \mu |$ </td></tr><tr><td> $H ( u )$ </td><td> $- 0 . 1 6 4 \pm 0 . 1 1 3$ </td><td> $4 7 0 . 5 \pm 4 7 . 2$ </td><td> $4 9 6 . 3 \pm 5 9 . 3$ </td><td> $4 5 . 5 7 \pm 1 . 7 1$ </td><td> $3 6 . 9 9 \pm 1 . 4 6$ </td><td> $1 7 . 7 4 \pm 2 . 1 3$ </td><td> $1 . 3 0 2 \pm 0 . 0 4 3$ </td></tr><tr><td>No entropy</td><td> $- 1 . 3 1 3 \pm 0 . 0 7 1$ </td><td> $4 6 4 . 1 \pm 7 3 . 5$ </td><td> $4 7 0 . 2 \pm 8 2 . 5$ </td><td> $2 3 . 6 4 \pm 1 . 4 7$ </td><td> $2 2 . 2 5 \pm 1 . 2 6$ </td><td> $0 . 0 5 \pm 0 . 0 5$ </td><td> $0 . 9 6 4 \pm 0 . 0 2 8$ </td></tr><tr><td> $\mathrm { T a n h } ^ { 2 }$  mean penalty</td><td> $- 1 . 4 5 9 \pm 0 . 0 4 8$ </td><td> $4 8 3 . 9 \pm 4 8 . 2$ </td><td> $4 5 5 . 4 \pm 5 2 . 1$ </td><td> $0 . 2 0 \pm 0 . 1 5$ </td><td> ${ \bf 0 . 0 9 \pm 0 . 1 0 }$ </td><td> $0 . 0 5 \pm 0 . 0 4$ </td><td> $\mathbf { 0 . 2 0 8 \pm 0 . 0 0 7 }$ </td></tr><tr><td>Pre-tanh L2</td><td> $- 1 . 5 1 8 \pm 0 . 0 4 0$ </td><td> $4 6 5 . 1 \pm 3 9 . 4$ </td><td> $4 7 3 . 8 \pm 3 2 . 3$ </td><td> $2 . 4 9 \pm 0 . 2 3$ </td><td> $1 . 6 0 \pm 0 . 2 5$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 4 7 5 \pm 0 . 0 1 5$ </td></tr><tr><td> $H ( a )$ </td><td> $- 0 . 9 1 8 \pm 0 . 0 8 6$ </td><td> ${ \bf 5 9 4 . 4 \pm 3 4 . 0 }$ </td><td> ${ \bf 6 4 3 . 2 \pm 5 8 . 4 }$ </td><td> $5 . 2 6 \pm 1 . 4 6$ </td><td> $1 . 9 1 \pm 0 . 6 6$ </td><td> $0 . 2 9 \pm 0 . 1 2$ </td><td> $0 . 4 8 9 \pm 0 . 0 3 0$ </td></tr></table>

![](images/270d63d43a8b4f8892637f900e76ef3c10479ff99b9ccc5fb2dc6126842656fa.jpg)  
Small markers are individual training seeds; large markers show the three-seed mean and the bars its sample SD. Labels give canonical stochastic return. The comparison dissociates mean placement from the final variance and performance regime; it does not establish that retained dispersion causes the return difference.

Figure 5: Dog-Stand mean–dispersion operating regimes. Small markers are individual training seeds; large markers are three-seed means and bars show sample SD. The horizontal axis is the canonical mean-only cell $P _ { 1 0 }$ at the 5% margin; the vertical axis is mean log σ. Labels give canonical stochastic return. L2 and $H ( a )$ have nearly matched mean geometry but substantially diferent variance and return regimes. The dashed connection highlights this comparison; it does not imply that retained dispersion causes the return diference.

In Dog-Stand, $H ( a )$ has the highest stochastic return among all five conditions in every matched training seed. The three matched-seed diferences are all positive, but $n = 3$ does not support a population-level performance inference; we therefore treat the return ordering as descriptive sign evidence. This is not a general performance claim: in MyoLeg, no entropy has the highest stochastic return in every matched seed. The consistent cross-task finding is geometric, not a universal return ordering.

## 5.8 Boundary occupancy and temporal action structure are related but distinct

Dog-Stand also lets us examine temporal action structure without early terminations. Figure 6 reports three secondary metrics. Under stochastic execution, the opposite-boundary flip rate is 4.711 Hz for $H ( u )$ , 0.821 Hz for no entropy, and 0.055 Hz for $H ( a )$ . Under deterministic execution the rates are 2.273, 0.682, and 0.011 Hz. Thus sampling noise alone does not explain the diference. Deterministic mean absolute first diference is 0.7140, 0.6357, and 0.4678, respectively.

Dog-Stand temporal execution (secondary, exploratory)

![](images/890b94502399fcc7bf0c7899bd40a3d0d9da1497887d6ec30de63c1892020616.jpg)

![](images/278073cbe252b5980ebed93cb5c5d4f717468503ed9ac5bcb34ea500c78eb3b5.jpg)

![](images/b4265b1efcad3050c0d481c6cd137f265f2a40db56369832c621006639524f35.jpg)  
boundary-to-boundary switching rather than sustained occupancy. Lower variation is consistent with smoother simulated action sequences; hardware consequences are untested

Figure 6: Dog-Stand temporal execution structure, reported as secondary exploratory evidence. A: Oppositeboundary flip rate. A flip is a direct transition between $y \geq 0 . 9$ and $y \le - 0 . 9$ on consecutive control steps; the control period is 0.015 s (66.67 Hz). B: Mean dwell time inside the near-boundary set during stochastic rollout. C: Deterministic action variation, $\mathbb { E } | y _ { t + 1 } - y _ { t } |$ . Lower variation is consistent with smoother simulated action sequences; hardware consequences are not evaluated.

These metrics should not be conflated with occupancy. A controller can remain near one bound for a long time and have high occupancy but few flips. Here, however, H(u) combines high occupancy with rapid transitions between opposite boundaries. H(a) reduces both. These are diagnostics of the present policies, not a comparison with dedicated smooth-control regularizers such as CAPS (Mysore et al., 2021) or structured-exploration methods such as gSDE (Rafin et al., 2022) and colored-noise exploration (Eberhard et al., 2023). We add no temporal smoothness loss and use independent Gaussian action sampling in the studied PPO policies. The result is an actuator-level consequence of the learned geometry in simulation, not evidence about wear or hardware safety.

## 6 Discussion

## 6.1 What entropy measurement changes

The central result holds in both studied tasks and does not rest on either one alone. Latent Gaussian entropy changes the variance objective but is blind to the placement of the state-conditioned mean. Executed-action entropy includes the bounded transform and therefore acts on both. In the tanh case, the mean force has a simple direction: it is inward for any nonzero mean. This direct diference appears in the frozen gradients in both MyoLeg and Dog-Stand.

The long-horizon outcome is not determined by the entropy gradient alone. Dog-Stand is the clearest counterexample: $H ( u )$ has a constant variance-increasing entropy contribution, yet its absolute variance decreases late in training. The correct interpretation is relative. Latent entropy biases the learned policy toward a higher variance regime than matched no-entropy training. Executed entropy adds an inward mean regularizer and a state-dependent variance term. The rest of PPO—the surrogate, visited states, optimizer history, and shared gradient clipping—still participates in the final operating point.

## 6.2 Executed entropy is not the only way to center the mean

The controls with direct mean penalties prevent an overly strong conclusion. A simple explicit penalty can produce an interior policy mean, and the tanh-squared control has lower Dog $P _ { 1 0 }$ than $H ( a )$ . Executed entropy is therefore not uniquely capable of preventing mean extrusion. The two penalties are mechanism controls, not claims of a new regularization family. CAPS, for example, explicitly regularizes smoothness of the state-to-action mapping and temporal actions (Mysore et al., 2021); our controls penalize only the current state’s policy mean and do not target action-rate smoothness.

What distinguishes $H ( a )$ empirically is the coupled regime it produces. The pre-tanh L2 policy provides an unusually clean comparison because its late-training mean distribution is almost the same as $H ( a )$ under both on-policy and shared-state evaluation. The two policies nonetheless end with substantially diferent dispersion and Dog return. This dissociates mean geometry from the final variance/performance regime. It does not show that higher retained variance causes higher return; the two training objectives also change the entire optimization trajectory.

## 6.3 Return does not identify policy geometry

The two tasks make this point in complementary ways. In MyoLeg, no entropy gives the highest return among the three tanh conditions, whereas $H ( a )$ produces the most interior geometry. In Dog-Stand, $H ( a )$ gives the highest return among all five conditions. The geometric ordering is consistent across tasks, whereas the return ordering is not. A method selected only by return would therefore lead to diferent conclusions about action geometry on the two tasks.

The gap between stochastic and deterministic execution is also separable from boundary geometry. In MyoLeg, deterministic return is much lower than stochastic return under all three tanh conditions. In Dog-Stand, the deterministic-to-stochastic return ratio is at or above one for four of the five conditions, including the highly boundary-oriented $H ( u )$ policy; only the tanh-squared control falls slightly below one (Table 3). Boundary-heavy geometry is not suficient to cause a performance gap between stochastic and deterministic execution.

## 6.4 Relation to SAC and transformed distributions

The executed-action entropy used here is standard change-of-variables probability. SAC’s tanh-squashed Gaussian already evaluates the transformed log density, including the Jacobian (Haarnoja et al., 2018), and it inherits that treatment from the maximum-entropy formulation it instantiates (Ziebart et al., 2008; Haarnoja et al., 2017). The novelty is not the formula. The contribution is the mechanistic diagnosis of what is lost when PPO measures entropy only in the latent Gaussian. The entropy term can then increase variance while exerting no direct constraint on where the mean sits relative to a bounded action map.

This distinction also explains why “bounded Gaussian mismatch” is too broad a description of the result. Hard clipping is unnecessary for the high-variance latent regime, and direct mean penalties can repair mean placement without using an entropy objective. The relevant design choice is how the training objective treats both mean placement and dispersion before the bounded execution map.

## 6.5 Practical diagnostic

The four-cell decomposition gives a simple way to diagnose bounded-policy behavior. Report $P _ { 1 0 }$ and $P _ { 0 1 }$ in addition to return and total near-boundary occupancy. If $P _ { 0 1 }$ is large while $P _ { 1 0 }$ is small, the problem is mainly dispersion and variance tuning may be suficient. If $P _ { 1 0 }$ is already large, reducing variance alone cannot produce an interior mean; the policy needs either a geometric term in the executed action space or an explicit mean regularizer.

Whether $H ( a )$ is worth using depends on the objective. If task return is the only target, neither task supports a universal recommendation. If near-boundary mean placement, rapid boundary switching, or simulated action variation matter, $H ( a )$ is a principled option because it regularizes the same bounded geometry used for execution. Explicit mean penalties are another option and may produce a diferent variance regime.

## 6.6 Scope

The experiments identify a reproducible regime, not the minimal conditions required for it. Both tasks use PPO, a diagonal Gaussian latent policy with a state-independent learned log σ, bounded actions, and MuJoCo-family simulation. State-dependent standard deviations are a common alternative implementation choice (Andrychowicz et al., 2021), and gSDE goes further by making exploration state dependent and temporally persistent (Rafin et al., 2022); neither is tested here. High action dimensionality may amplify the phenomenon, but no dimension sweep isolates that factor. Neither PPO specificity nor Gaussian-policy specificity is tested. The entropy coeficient is fixed at $1 0 ^ { - 3 }$ in the primary entropy conditions; the analytic gradient direction is known, but the relation between efect size and coeficient is not measured.

The Dog conclusions are also tied to the attained performance regime. Canonical stochastic returns span roughly 464–594 on 1000-step episodes. Whether substantially stronger training, a diferent optimizer schedule, or another algorithm preserves the same geometry remains open. No hardware experiment is performed. These scope limits do not change the matched comparisons reported here; they define the next questions rather than additional experiments required for the present mechanism claim.

## 7 Conclusion

PPO with a diagonal Gaussian policy and bounded actions has, in the settings studied here, a policy-geometry problem that return alone can hide. In MyoLeg, a boundary-dominated Gaussian policy is produced jointly by mean extrusion and dispersion; the mean by itself is already suficient for most of the occupancy. Measuring entropy in the latent Gaussian raises the relative variance regime without constraining mean placement. Measuring entropy after the bounded transform adds a direct inward mean regularizer and changes the variance objective. The resulting ordering in mean geometry is reproduced in a 38-dimensional Dog-Stand task, survives evaluation on shared states, and is unchanged across boundary margins from 1% to 10%.

Executed entropy is not the only way to obtain interior means. Direct mean penalties can match or exceed its centering. Their late-training variance and performance regimes can still difer substantially, showing that mean placement, dispersion, and task return are distinct properties of a learned controller. For bounded-action PPO, all three should be measured explicitly.

## References

Frank C. Anderson and Marcus G. Pandy. Dynamic optimization of human walking. Journal of Biomechanical Engineering, 123(5):381–390, 2001. doi: 10.1115/1.1392310.

Marcin Andrychowicz, Anton Raichuk, Piotr Stańczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, and Olivier Bachem. What matters for on-policy deep actor-critic methods? a large-scale study. In International Conference on Learning Representations, 2021.

Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, and Vikash Kumar. Myosuite: A contact-rich simulation suite for musculoskeletal motor control. In Proceedings of the 4th Annual Learning for Dynamics and Control Conference, volume 168 of Proceedings of Machine Learning Research, pp. 492–507, 2022.

Carnegie Mellon University Graphics Lab. CMU Graphics Lab Motion Capture Database. https://mocap. cs.cmu.edu/, 2003. Subject 2, trial 1 used as 02\_01.c3d; accessed Aug. 17, 2026.

Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Huang, and Alexander Mathis. Latent exploration for reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023.

Po-Wei Chou, Daniel Maturana, and Sebastian Scherer. Improving stochastic policy gradients in continuous control with deep reinforcement learning using the Beta distribution. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 834–843, 2017.

Onno Eberhard, Jakob Hollenstein, Cristina Pinneri, and Georg Martius. Pink noise is all you need: Colored noise exploration in deep reinforcement learning. In International Conference on Learning Representations, 2023.

Carson Eisenach, Haichuan Yang, Ji Liu, and Han Liu. Marginal policy gradients: A unified family of estimators for bounded action spaces with applications. In International Conference on Learning Representations (ICLR), 2019.

Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. Implementation matters in deep policy gradients: A case study on PPO and TRPO. In International Conference on Learning Representations, 2020.

C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax – a diferentiable physics engine for large scale rigid body simulation. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2021. arXiv:2106.13281.

Yasuhiro Fujita and Shin-ichi Maeda. Clipped action policy gradient. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1597–1606, 2018.

Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1352–1361, 2017.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Of-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1861–1870, 2018.

Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, Jef Braga, Dipam Chakraborty, Kinal Mehta, and João G. M. Araújo. Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. Journal of Machine Learning Research, 23(274):1–18, 2022.

Łukasz Kidziński, Sharada Prasanna Mohanty, Carmichael F. Ong, Zhewei Huang, Shuchang Zhou, Anton Pechenko, et al. Learning to run challenge solutions: Adapting reinforcement learning methods for neuromusculoskeletal environments. In The NIPS ’17 Competition: Building Intelligent Systems, pp. 121–153. Springer, 2018. doi: 10.1007/978-3-319-94042-7\_7.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Seunghwan Lee, Moonseok Park, Kyoungmin Lee, and Jehee Lee. Scalable muscle-actuated human simulation and control. ACM Transactions on Graphics, 38(4):73:1–73:13, 2019. doi: 10.1145/3306346.3322972.

Zhihao Lin. Beyond distributions: Geometric action control for continuous reinforcement learning. arXiv preprint arXiv:2511.08234, 2025.

Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 1928–1937, 2016.

Siddharth Mysore, Bassel Mabsout, Renato Mancuso, and Kate Saenko. Regularizing action policies for smooth control with reinforcement learning. In IEEE International Conference on Robotics and Automation, pp. 1810–1816, 2021. doi: 10.1109/ICRA48506.2021.9561138.

Irving G. B. Petrazzini and Eric A. Antonelo. Proximal policy optimization with continuous bounded action space via the Beta distribution. In 2021 IEEE Symposium Series on Computational Intelligence, pp. 1–8, 2021. doi: 10.1109/SSCI50451.2021.9660123.

Antonin Rafin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.

Antonin Rafin, Jens Kober, and Freek Stulp. Smooth exploration for robotic reinforcement learning. In Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pp. 1634–1644, 2022.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Pierre Schumacher, Daniel F. B. Haeufle, Dieter Büchler, Syn Schmitt, and Georg Martius. Dep-rl: Embodied exploration for reinforcement learning in overactuated and musculoskeletal systems. In International Conference on Learning Representations, 2023.

Tim Seyde, Igor Gilitschenski, Wilko Schwarting, Bartolomeo Stellato, Martin Riedmiller, Markus Wulfmeier, and Daniela Rus. Is bang-bang control all you need? solving continuous control with bernoulli policies. In Advances in Neural Information Processing Systems, volume 34, pp. 27209–27221, 2021.

Jun Jet Tai, Mark Towers, and Elliot Tower. Shimmy: Gymnasium and PettingZoo wrappers for commonly used environments. https://github.com/Farama-Foundation/Shimmy, 2023.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033, 2012. doi: 10.1109/ IROS.2012.6386109.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U. Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, Rodrigo Perez-Vicente, Andrea Pierré, Sander Schulhof, Jun Jet Tai, Hannah Tan, and Omar G. Younis. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel, Tom Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa. dm\_control: Software and tasks for continuous control. Software Impacts, 6:100022, 2020. doi: 10.1016/j.simpa.2020.100022.

Antonie J. van den Bogert. Strange efects of activation dynamics on musculoskeletal trajectory optimization. bioRxiv, 2025. doi: 10.1101/2025.01.30.635759. Preprint; not peer reviewed.

Cheryl Wang, Chun Kwang Tan, Balint Hodossy, Eric Lyu, Pierre Schumacher, James Heald, Kai Biegun, Samo Hromadka, Maneesh Sahani, Gunwoo Park, Beomsoo Shin, Jonghyun Park, Seungbum Koo, Chenhui Zuo, Chengtian Ma, Yanan Sui, Nicklas Hansen, Stone Tao, Yuan Gao, Hao Su, Seungmoon Song, Letizia Gionfrida, Massimo Sartori, Guillaume Durandau, Vikash Kumar, and Vittorio Caggiano. MyoChallenge 2024: A new benchmark for physiological dexterity and agility in bionic humans. In Advances in Neural Information Processing Systems, volume 38, 2025. Datasets and Benchmarks Track.

Ronald J. Williams and Jing Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268, 1991. doi: 10.1080/09540099108946587.

Ruikai Zhou, Taihong Zhong, Wenbo Zhu, Shuai Han, and Shuai Lü. Influence of gaussian distribution on performance metrics in continuous reinforcement learning. Information Processing & Management, 63(2): 104428, 2026. doi: 10.1016/j.ipm.2025.104428.

Brian D. Ziebart, Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey. Maximum entropy inverse reinforcement learning. In Proceedings of the 23rd AAAI Conference on Artificial Intelligence, pp. 1433– 1438, 2008.

## A Boundary Margin Sensitivity

Table 4 reports the seed-mean $P _ { 1 1 } / P _ { 1 0 }$ values used for the margin-robustness statement. For each checkpoint, one fresh 30-episode on-policy state set is collected and all four margins are evaluated analytically on those same states; no actions are resampled for the boundary probabilities. Both sweeps use the same $\mathrm { N u m P y }$ action-noise convention as their frozen canonical evaluators, so each checkpoint’s state set coincides with its canonical one and the 5% cells reproduce Tables 2 and 3 exactly.

Table 4: Boundary-margin sensitivity. Entries are $P _ { 1 1 } / P _ { 1 0 }$ in percent, averaged over three training seeds. The primary ordering $H ( u ) >$ no entropy $> H ( a )$ holds for both quantities in every seed at every margin. Both 5% rows reproduce their frozen canonical values (Tables 2 and 3) exactly.
<table><tr><td>Task</td><td>Margin</td><td> $H ( u )$ </td><td>No entropy</td><td> $H ( a )$ </td></tr><tr><td>Dog</td><td>1%</td><td>24.47 /15.83</td><td>6.63 / 5.88</td><td>0.28 /0.05</td></tr><tr><td>Dog</td><td>2.5%</td><td>35.34 / 26.31</td><td>14.18 /13.02</td><td>1.63 3/ 0.44</td></tr><tr><td>Dog</td><td>5%</td><td>45.57 /36.99</td><td>23.64 /22.25</td><td>5.26 / 1.91</td></tr><tr><td>Dog</td><td>10%</td><td>57.77/ 50.46</td><td>37.58 /36.10</td><td>14.69 / 7.44</td></tr><tr><td>MyoLeg</td><td>1%</td><td>56.40 40.46</td><td>10.30 /7.69</td><td>2.77 /0.82</td></tr><tr><td>MyoLeg</td><td>2.5%</td><td>64.70 51.62</td><td>19.51 15.81</td><td>9.28 /4.75</td></tr><tr><td>MyoLeg</td><td>5%</td><td>71.42 60.72</td><td>29.76 /25.67</td><td>18.83 / 11.62</td></tr><tr><td>MyoLeg</td><td>10%</td><td>78.61 / 70.54</td><td>43.55 /39.42</td><td>33.50 / 23.19</td></tr></table>

The 5% regression check uses a 0.05-percentage-point tolerance as a guard against state-timing, normalization, threshold, and cell-definition errors; in the released runs the reproduction is exact. Table 2 remains the source of record for the primary 5% MyoLeg values, and Table 4 for the four-margin sweep.

The variance-only comparison between no entropy and $H ( a )$ is not threshold invariant in MyoLeg. At 1% and 2.5%, no-entropy seed 2 has $P _ { 0 1 } = 0 . 2 6 1 0 \%$ and 0.7309%, above $H ( a )$ at 0.1204% and 0.6857%. At the 5% reference margin the same seed has 1.8284% versus 2.3493%, and at 10% it has 5.2824% versus 7.5610%. This small comparison is therefore not used as a margin-robust headline result.

## B Common-State Dog Audit

Table 5 reports $P _ { 1 0 }$ when every trained Dog policy is evaluated on every policy’s raw donor states. Each evaluated policy applies its own frozen observation normalizer. The equal-mixture pool contains an equal number of states from each donor condition.

Table 5: Dog common-state $P _ { 1 0 }$ (%), donor state set in rows and evaluated policy in columns, mean over three training seeds.
<table><tr><td>Donor</td><td> $H ( u )$ </td><td>No entropy</td><td> $H ( a )$ </td><td> $\mathrm { T a n h } ^ { 2 }$  penalty</td><td>Pre-tanh L2</td></tr><tr><td> $H ( u )$ </td><td>37.05</td><td>28.31</td><td>4.56</td><td>0.59</td><td>5.30</td></tr><tr><td>No entropy</td><td>38.00</td><td>22.26</td><td>5.15</td><td>0.96</td><td>4.23</td></tr><tr><td> $H ( a )$ </td><td>42.06</td><td>28.62</td><td>1.88</td><td>0.57</td><td>4.52</td></tr><tr><td> $\mathrm { T a n h } ^ { 2 }$  penalty</td><td>42.46</td><td>27.31</td><td>5.29</td><td>0.09</td><td>5.13</td></tr><tr><td>Pre-tanh L2</td><td>38.89</td><td>28.52</td><td>5.38</td><td>0.59</td><td>1.60</td></tr><tr><td>Equal-mixture pool</td><td>39.69</td><td>26.99</td><td>4.47</td><td>0.56</td><td>4.13</td></tr></table>

The primary ordering $H ( u ) > \mathrm { n o }$ entropy $> H ( a )$ holds on all six evaluated state sets (the five donor sets plus the equal-mixture pool) in every seed: 18 of 18 state-set–seed comparisons. At the seed-averaged level all five policies have their lowest $P _ { 1 0 }$ on their own donor distribution; this tendency holds in 14 of 15 individual policy–seed comparisons. We treat this as descriptive evidence of policy–state-distribution co-adaptation, not as a causal result.

## C Gradient Details

Table 6 reports the local actor-loss gradient decomposition for the late-training MyoLeg seed-0 checkpoint of each entropy condition. Each entry is the mean per-dimension gradient with respect to log σ over 10 independently collected frozen-policy bufers.

Table 6: MyoLeg frozen-bufer log σ gradient decomposition, 10 independent bufers per condition. Intervals are 95% t-intervals over bufers. Positive loss gradient lowers σ.
<table><tr><td>Condition</td><td>Entropy</td><td>Surrogate</td><td>Net</td></tr><tr><td>No entropy</td><td>0</td><td> $\phantom { 0 0 0 } { + 1 . 0 1 5 } \times 1 0 ^ { - 3 }$ </td><td> $\phantom { 0 0 0 } { + 1 . 0 1 5 } \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="2">H(u)</td><td></td><td> $[ 0 . 3 6 3 , 1 . 6 7 ] \times 1 0 ^ { - 3 }$ </td><td>same</td></tr><tr><td> $- 1 . 0 0 0 \times 1 0 ^ { - 3 }$ </td><td> $+ 0 . 6 1 \dot { 0 } \times 1 0 ^ { - 3 }$ </td><td> $- 0 . 3 9 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="2">H(a)</td><td>exact</td><td> $[ 0 . 0 5 2 , 1 . 1 7 ] \times 1 0 ^ { - 3 }$ </td><td> $[ - 0 . 9 4 9 , 0 . 1 6 8 ] \times 1 0 ^ { - 3 }$ </td></tr><tr><td> $- 0 . 5 5 2 \times 1 0 ^ { - 3 }$ </td><td> $\phantom { 0 } { + 0 . 6 0 4 } \times 1 0 ^ { - 3 }$ </td><td> $+ 0 . 0 5 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td> $[ - 0 . 1 7 5 , 1 . 3 8 ] \times 1 0 ^ { - 3 }$  1</td><td> $- 0 . 7 2 7 , 0 . 8 3 1 ] \times 1 0 ^ { - 3 }$ </td></tr></table>

Only the net interval under no entropy excludes zero. The local decomposition is therefore used to constrain the late-training MyoLeg mechanism, not as a complete causal explanation of the full trajectories.

## D Training Configuration and Executed-Entropy Estimator

Tables 7 and 8 give the verified training configuration and network architecture used by the two primary implementations. The MyoLeg source does not override several Stable-Baselines3 PPO fields. For those fields, Table 7 reports the resolved values read back from the trained checkpoints by the release manifest, not defaults assumed from library documentation; the manifest also records the environment lockfile.

Table 7: Training configuration for the primary tanh experiments. The Dog controls with direct mean penalties use the Dog column with $c _ { \mathrm { e n t } } = 0$ plus the stated penalty.
<table><tr><td>Setting</td><td>MyoLeg / Stable-Baselines3 PPO</td><td>Dog-Stand / CleanRL PPO</td></tr><tr><td></td><td>Observation / action dimen- 101 / 80 latent muscle actions</td><td>223 / 38 latent actions</td></tr><tr><td>sions Parallel environments</td><td>12, SubprocVecEnv</td><td>1, synchronous vector environment</td></tr><tr><td>Rollout length</td><td>2048 per environment</td><td>2048</td></tr><tr><td>Rollout batch</td><td>24,576 transitions</td><td>2,048 transitions</td></tr><tr><td>Minibatch size</td><td>128</td><td>64 (32 minibatches)</td></tr><tr><td>Update epochs</td><td>10</td><td>10</td></tr><tr><td>Optimizer</td><td>Adam, Stable-Baselines3 implementation</td><td>Adam,  $\epsilon = 1 0 ^ { - 5 }$ </td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 } ,$  constant</td><td> $3 \times 1 0 ^ { - 4 }$  , linearly annealed over the prescribed</td></tr><tr><td></td><td></td><td>run</td></tr><tr><td>Discount γ GAE λ</td><td>0.99 0.99</td><td>0.99</td></tr><tr><td>PPO clip coefficient</td><td>0.2</td><td>0.95 0.2</td></tr><tr><td>Value-loss coefficient</td><td>0.5</td><td>0.5</td></tr><tr><td>Value clipping</td><td>disabled (clip_range_vf=None)</td><td>enabled with clip coefficient 0.2</td></tr><tr><td>Advantage normalization</td><td>enabled</td><td>enabled</td></tr><tr><td>State-dependent exploration / gSDE</td><td>not used; diagonal state-independent logσ</td><td>not used; diagonal state-independent log σ</td></tr><tr><td>Target KL</td><td>none (target_kl=None)</td><td>none</td></tr><tr><td>Maximum gradient norm</td><td>0.5</td><td>0.5 0</td></tr><tr><td>Initial log σ</td><td>0</td><td></td></tr><tr><td>Standard deviation</td><td>learned state-independent vector</td><td>learned state-independent vector</td></tr><tr><td>Observation normalization</td><td>none</td><td>running mean/variance, then clipped to [−10, 10]; frozen during canonical evaluation running reward normalization and [-10, 10] clip-</td></tr><tr><td>Reward normalization</td><td>none</td><td>ping during training; canonical evaluation re- ports raw environment return</td></tr><tr><td>Action mapping</td><td>exposed latent box  $[ - 1 0 , 1 0 ] ^ { 8 0 }$  , with measured tion 4 (tanh 10 = 1 − 4.1 × 10−9); a = (tanh u + affine map to physical bounds 1)/2</td><td>unbounded exposed latent box; no clip of any clip incidence and its inertness reported in Sec- kind precedes the transform; y = tanh u, then</td></tr><tr><td>Executed-entropy samples</td><td>10−3 for  $H ( u ) / H ( a ) ,$  0 for no entropy K = 4, antithetic reparameterized samples</td><td>10−3 for  $H ( u ) / H ( a ) ,$  0 for no entropy  $K = 4 ,$  antithetic reparameterized samples</td></tr><tr><td>Training seeds</td><td>0, 1, 2</td><td>0, 1, 2</td></tr><tr><td>Prescribed environment steps</td><td></td><td>3,000,320</td></tr><tr><td></td><td>145,022,976</td><td></td></tr></table>

Table 8: Policy and value network architecture. Both implementations use separate actor and critic MLPs and a state-independent learnable log σ vector.
<table><tr><td>Component</td><td>MyoLeg</td><td>Dog-Stand</td></tr><tr><td>Actor mean</td><td> $1 0 1  5 1 2  5 1 2  8 0 .$  ReLU hidden layers</td><td> $2 2 3  6 4  6 4  3 8$  tanh hidden layers</td></tr><tr><td>Critic</td><td> $1 0 1  5 1 2  5 1 2  1 ,$  ReLU hidden layers</td><td> $2 2 3 \to 6 4 \to 6 4 \to 1 .$  tanh hidden layers</td></tr><tr><td>Actor dispersion</td><td>80 learned log σi parameters, initialized to zero</td><td>38 learned log  $\sigma _ { i }$  parameters, initialized to zero</td></tr><tr><td>Initialization</td><td>Stable-Baselines3 MlpPolicy initialization</td><td>orthogonal; hidden gain  ${ \sqrt { 2 } } ,$  actor-output gain 0.01, critic-output gain 1.0, zero biases</td></tr><tr><td>Trainable parameters</td><td>671,393</td><td>39,565</td></tr></table>

## D.1 Four-sample antithetic estimator for $H ( a )$

For a minibatch of states, the executed-entropy implementation uses the following pathwise estimator. The MyoLeg afine map to [0, 1] is included in the Jacobian; for Dog, the later afine physical-action scaling is constant in the policy parameters and can be omitted from gradients.

1. Compute $\sigma = \exp ( \log \sigma )$ and the exact latent entropy $\begin{array} { r } { H _ { u } = \sum _ { i } [ \log \sigma _ { i } + \frac { 1 } { 2 } \log ( 2 \pi e ) ] } \end{array}$

2. Draw two independent $\epsilon _ { 1 } , \epsilon _ { 2 } \sim \mathcal { N } ( 0 , I )$ and form the four-sample antithetic set $\{ \epsilon _ { 1 } , - \epsilon _ { 1 } , \epsilon _ { 2 } , - \epsilon _ { 2 } \}$

3. For each $\epsilon _ { k } ,$ form the reparameterized latent action $u _ { k } = \mu + \sigma \odot \epsilon _ { k }$

4. Evaluate the stable componentwise log-Jacobian. For $a = ( \operatorname { t a n h } { u } + 1 ) / 2 .$

$$
\ell _ { [ 0 , 1 ] } ( u ) = \log 2 - 2 u - 2 \mathrm { ~ s o f t p l u s } ( - 2 u ) .\tag{23}
$$

For the normalized Dog action y = tanh u,

$$
\ell _ { [ - 1 , 1 ] } ( u ) = \log 4 - 2 u - 2 \mathrm { \ s o f t p l { u s } } ( - 2 u ) .\tag{24}
$$

## 5. Return

$$
\widehat { H } ( a ) = H _ { u } + \frac { 1 } { 4 } \sum _ { k = 1 } ^ { 4 } \sum _ { i } \ell ( u _ { k , i } ) ,\tag{25}
$$

and add $- c _ { \mathrm { e n t } }$ times the minibatch mean of this quantity to the actor loss.

The Gaussian entropy term is analytic; Monte Carlo noise enters only through the expected log-Jacobian. Reparameterization provides pathwise gradients in both $\mu$ and log σ. The antithetic construction forces the sampled noise mean to zero within each pair and removes first-order sampling imbalance around the current mean, but it does not eliminate higher-order Monte Carlo variance and is not claimed to be variance-optimal. K = 4 is fixed for every $H ( a )$ training run in both tasks, so estimator noise is not a between-condition tuning variable. The numerical self-check in the released step2\_train\_muscle\_tanh\_Ha.py (–check –skip\_env) sweeps K from 1 to 32 and confirms that the estimator mean does not drift with K while its standard deviation shrinks at the expected Monte Carlo rate. The implementation’s numerical check compares estimator bias and variance across several K values and verifies the stable Jacobian form; no paper claim relies on treating the four samples as statistical replicates.

## D.2 Release statement and canonical evaluators

An archival code package accompanies this paper, containing both training implementations, the exact canonical evaluators, analysis scripts, and frozen JSON outputs. A machine-readable manifest maps every reported table to its evaluator revision, checkpoint SHA-256 hash, protocol name, and random-seed convention. The same tagged release will be maintained in a permanent public repository. The canonical evaluators, not training-time rolling diagnostics, are the source of record for reported returns and policy-geometry tables. Where normalization is used, the evaluator restores and freezes the checkpoint-specific observation statistics. The release will also include the common-state and margin scripts so that the state-distribution and threshold robustness claims can be reproduced directly. Large checkpoints and third-party motion-capture assets will be released when licensing permits; otherwise the manifest will provide acquisition instructions and cryptographic hashes.

## E Reproducibility Notes

The final analysis uses step-matched checkpoints and versioned canonical evaluators. Several defects discovered during the audit changed earlier reported values or statistical interpretation. Reward-selected checkpoints were not compute matched, and a motor boundary test used the muscle action interval. Beta parameters could fall through a silent fallback, and a mapping could be overwritten by directory order. The first multi-bufer gradient analysis unintentionally replayed the same rollout, because loading the PPO checkpoint reseeded the environments. Finally, Dog evaluator revisions initially mishandled terminal versus periodic checkpoint names. The final scripts add explicit checkpoint metadata, numeric step parsing, regression tests, and bufer fingerprints. These corrections are part of the evidence chain rather than additional experimental conditions.

All experiments were run on Windows 10 with Python 3.9.25, PyTorch 2.8.0 (CUDA 12.8, cuDNN 9.10.2), NumPy 2.0.2, Gymnasium 0.29.1, Stable-Baselines3 2.7.1, MuJoCo 3.3.0, MyoSuite 2.11.6, Shimmy 1.3.0, and SciPy 1.13.1, on a single NVIDIA GeForce RTX 5090 (32 GiB) with a 32-thread CPU. A machine-generated release manifest reproduces this list together with a full pip freeze, the driver report, per-checkpoint SHA-256 digests, and the resolved Stable-Baselines3 field values read back from the checkpoints rather than assumed. The manifest is produced by a single script and is included with the code archive.

The MyoLeg control loop was verified from the live environment rather than assumed: the physics timestep is 1 ms and the environment applies a frame skip of 4, giving a 4 ms control period and a control frequency of 250 Hz. The Dog-Stand control period is fixed by dm\_control at 0.015 s (66.67 Hz).

Training provenance. The no-entropy and H(a) seed-0 runs resume 30M-step pilot checkpoints of the same configurations; the symmetric-τ seeds 1 and 2 continue 100M-step runs to the matched budget; and the filter and override seed-0 policies were trained by earlier single-run drivers that are included in the release alongside the seeded multi-run scripts.