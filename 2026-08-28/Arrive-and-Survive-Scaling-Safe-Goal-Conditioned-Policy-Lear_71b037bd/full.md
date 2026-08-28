# Arrive and Survive: Scaling Safe Goal-Conditioned Policy Learning from One-Bit Failure Signals

Guopeng Li, Yiyang Duan, Chengcheng Xu Southeast University Nanjing, China {guopengli5, duanyiyang, xuchengcheng}@seu.edu.cn

Yiru Jiao 2030 Lab, Yinwang Intel. Tech. Co. Ltd. Shanghai, China yiru.jiao@studyinger.space

## ABSTRACT

Contrastive reinforcement learning (CRL) scales effectively in goal-conditioned tasks by casting policy learning into a self-supervised contrastive objective. However, in a failure-terminated Markov decision process, established CRL considers pre-failure future goals only when constructing positive samples, without accounting for the probability mass removed by failure termination. Our theoretical analysis shows that this omission induces a systematic overestimation bias in goal-reaching values. Consequently, near-failure trajectories provide disproportionately strong supervision of success despite retaining little future occupancy. Unsafe actions can thereby be reinforced through catastrophic failure bootstrapping, leading to failed policy learning and unsustainable goal-reaching behaviours. To address this problem, we introduce two minimal yet strong corrections: mass-weighted InfoNCE corrects the overweighting of short surviving futures in critic learning, and a log-survival-mass score restores the missing survival mass in policy optimization. The resulting method, Safe Contrastive Reinforcement Learning (Safe-CRL), requires only the one-bit signal provided by failure termination to scale safe goal-conditioned policy learning. Across twelve failure-prone robot navigation and locomo tion tasks, Safe-CRL consistently improves survival and substantially outperforms the Scaling-CRL baseline in goal-reaching performance. Additionally, deep Safe-CRL policies exhibit complex failure avoidance behaviours. This study completes the CRL theory under failure termination and provides a scalable safe RL framework. The code is available via https://github.com/RomainLITUD/safe-crl.

Keywords Self-Supervised Reinforcement Learning · Contrastive Learning · Safe Policy Learning · Robot Control

![](images/26bb3ca446d9d188dfbd161185e42a76707fc4ff042d2302bffbfb370f7fa561.jpg)

![](images/77f0cc7df2ed3352736c2acb10e53a890a1e1e130a30cdaa2c15f945c85cc00d.jpg)

![](images/abe30c962ba39804f16f34abdeab378a8ceb8763a20aae715858ad05bff8b1ad.jpg)

![](images/17f3b6f538529947c554d79a32c721d3ee571ed12305ecbd54e9cd07d63bebab.jpg)

![](images/49af785a3c063d6c0c2af4d4116fbf08ae8bfbfe3fc69b7d140f085767674abc.jpg)

![](images/671d2d9db840d6cef74345048c7b9f2e8daa5dc995679c34411a085f354ebc45.jpg)

![](images/311b34cab00fe9e70f9ee83c62114bc1a2bf0b07365cd72553d944effd0c1192.jpg)

![](images/a56b30251b095b2757826b3cc7f6a47ffe1e844f983bfb0f93e27f37739da275.jpg)

![](images/1cbe0fd7cb2c04fbfe74701ade3dc3245f6464fb598ac86927ee647e984713ff.jpg)

![](images/70c5bc604e09248f4e43a3a50e52eabaddd5a79d7505ba9716ef05bc63cbe178.jpg)

![](images/f09938d2f6f3f09aa94865708e43ec7bd64db83fbf33a8d2c9bbb66b31a01efd.jpg)

![](images/5d28b636ce208dc7b9c06d2c10d69b58ffec60865c4e05c8fcaf20d34b891448.jpg)  
Figure 1: Comparison of time at goal between Scaling-CRL and the proposed Safe-CRL method in failure-prone navigation and locomotion tasks (mean ± standard deviation over five random seeds). Both methods use 64-layer actor and critic networks (8-layer for Point Goal and Car Goal). Safe-CRL achieves higher or on-par mean time at goal across all tasks, with pronounced gains in nine.

## 1 Introduction

Scaling model capacity has become an important route toward more capable reinforcement learning (RL) policies [1, 2]. In goal-conditioned RL, contrastive reinforcement learning (CRL) is particularly promising. It turns sparse goalreaching reward into dense self-supervision through goal relabelling [3, 4], and learns to reach commanded goals via a contrastive objective. Recent work on Scaling-CRL has shown that this approach can effectively exploit deeper actor and critic networks to improve performance, establishing CRL as a scalable policy learning framework [5].

In practice, many goal-conditioned tasks involvefailure termination – unsafe events like falls or collisions immediately end the episode. For example, robot locomotion requires reaching the goal position without falls [6, 7]; navigation and autonomous driving require chasing moving goals while avoiding collisions [8, 9]. In these safety-critical settings, failure prevents the goal from being reached or terminates the episode shortly after the agent reaches the goal. Therefore, a safe goal-conditioned policy needs to learn how to arrive and to survive in a scalable manner. A desirable solution should preserve the self-supervised scalability of CRL without introducing additional safety annotations or task-specific cost signals. Ideally, learning should rely only on the one-bit failure signal already provided by termination.

However, failure termination interacts with the established CRL method subtly: failure truncates the future states available for relabelling, but established CRL renormalizes the remaining pre-failure future goals back to unit mass [3, 5]. For example, from the same state, consider two actions that lead to similar observed pre-failure future goals: one is followed by a long stable trajectory, whereas the other leads to failure quickly after reaching the goal. Relabelling the valid futures of each trajectory without accounting for their different trajectory lengths gives the two actions comparable positive supervision despite different safety performance. In other words, CRL preserves information about which pre-failure future goals are reachable, but loses information about how much discounted future-goal occupancy survives, which we term the survival mass.

The missing survival mass induces a systematic bias in goal-reaching value and can have severe consequences. The first one is what we refer to as catastrophic failure bootstrapping. In failure-prone tasks, the replay buffer is filled with short near-failure trajectories during the early stages of training. Repeatedly reinforcing such trajectories through self-supervision can trap learning in failure-prone behaviour. Another consequence is unsustainable goal reaching: the agent aggressively reaches a commanded goal but collides, overshoots, or falls shortly afterwards, as illustrated in Figure 2. These two failure modes can substantially degrade CRL performance in safety-critical tasks.

The discussion above raises a critical research question: How should CRL accountforfailure termination to realize scalable and safe goal-conditioned policy learning, when only a one-bitfailure signal is available?

![](images/4d9b2b78ecbfd9351313a35e267efd5142d2cc303775b9e0ec4ae940cbf5c15c.jpg)  
Figure 2: Examples of unsustainable goal reaching under Scaling-CRL. (a) The Point agent reaches the goal region but subsequently collides with a moving obstacle; (b) the Ant reaches the goal at high speed and overshoots into the wall; (c–d) locomotion agents reach the goal but fall immediately afterwards. These examples illustrate that successful short-term goal reaching does not necessarily lead to a stable trajectory under failure termination.

In this paper, we first identify and formalize the systematic overestimation bias in established CRL under failure termination. This bias can trap policy learning in unsafe, near-failure regions, or cause unsustainable goal reaching. To address this problem, we introduce mass-weighted InfoNCE and a log-survival-mass correction, yielding Safe-CRL as a minimal yet strong extension of Scaling-CRL [5]. Safe-CRL substantially improves survival and goal-reaching performance in failure-prone environments while preserving Scaling-CRL’s depth scalability.

## 2 The missing survival mass in CRL

We now formalize how failure termination absorbs discounted future-goal occupancy and explain why established CRL has an overestimation bias caused by discarding this information. We then show how the bias affects critic learning and actor optimization.

## 2.1 Problem formulation and CRL preliminaries

We consider a goal-conditioned MDP with failure termination,

$$
\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , \rho _ { 0 } , \gamma , \mathcal { G } , \phi , F ) ,\tag{1}
$$

where $s$ and A denote the state and action spaces, $P ( s ^ { \prime } \mid s , a )$ is the state transition kernel, $\rho _ { 0 }$ is the initial-state distribution, and $\gamma \in ( 0 , 1 )$ is the discount factor. The goal space is ${ \mathcal { G } } .$ , and $\phi : S  \mathcal { G }$ maps a state to the achieved goal. A goal-conditioned policy $\pi ( \boldsymbol { a } \mid s , g )$ acts toward a commanded goal $g \in { \mathcal { G } }$ . The binary function $F : \mathcal { S } { \times } \mathcal { A } { \times } \mathcal { S }  \{ { \bar { 0 } } , 1 \}$ indicates whether a transition leads to a failure that immediately terminates the episode. We denote $T _ { f }$ as the random variable representing the failure horizon measured from an anchor $\left( { { s _ { t } } , { a _ { t } } } \right)$

CRL uses a contrastive critic together with a goal-conditioned actor and operates on anchor-goal pairs. The critic learns a contrastive score $f _ { \boldsymbol { \theta } } ( x , g )$ between a state-action anchor $x = ( s , a )$ and a goal g. Positive goals are obtained by hindsight future-goal relabelling [3, 4]: a horizon H is sampled from a discounted geometric distribution $q _ { \gamma } ( h ) = ( \bar { 1 } - \gamma ) \gamma ^ { \bar { h } - 1 }$ for $h \geq 1$ , and the observed achieved goal $g _ { t + H } = \phi ( S _ { t + H } )$ is the positive goal for this anchor. Goals associated with other anchors in a mini-batch are considered negatives. Thus, the InfoNCE loss is used to train the critic to distinguish the positive goals from negative goals by learning higher scores. As the positive goal is actually reached from the anchor, this learning process is interpreted as maximizing the likelihood of reachability in CRL [5].

At the population optimum, the contrastive score represents a positive-to-negative density ratio. Let $p ^ { + } ( g \mid x )$ denote the positive goal distribution from anchor x and consider an anchor-independent negative proposal $p ^ { - } ( g )$ . The contrastive score estimates [3]

$$
f _ { \theta } ( x , g ) \to \log \frac { p ^ { + } ( g \mid x ) } { p ^ { - } ( g ) } + c ( x ) ,\tag{2}
$$

where $c ( x )$ is an anchor-dependent offset. Because $x = ( s , a )$ , row-wise InfoNCE alone does not make scores directly comparable across candidate actions. CRL therefore uses score calibration to constrain this offset [3]. Under the calibrated score convention, for a fixed commanded goal $^ { g , }$ the negative proposal is action-independent, and actor optimization reduces to maximizing the conditional positive-goal likelihood, where $f _ { \theta }$ denotes the calibrated contrastive score:

$$
\arg \operatorname* { m a x } _ { a } \bar { f } _ { \theta } ( x , g ) = \arg \operatorname* { m a x } _ { a } \log p ^ { + } ( g \mid x ) .\tag{3}
$$

## 2.2 Overestimation bias in established CRL

Next, we show how failure termination and pre-failure-only relabelling in CRL create a systematic overestimation bias. Following CRL [3], throughout the occupancy analysis below, π denotes the replay-induced trajectory distribution underlying contrastive future-goal sampling; when replay contains trajectories collected under different commanded goals, it is understood as the corresponding goal-marginalized replay-induced trajectory distribution. For an MDP with infinite horizon (no failure termination), Eysenbach et al. [3] established the equivalence between the goal-reaching action-value function (Q-function) and the so-called discounted state occupancy, defined by

$$
\mu _ { \infty } ^ { \pi } ( g \mid x ) = \sum _ { h = 1 } ^ { \infty } q _ { \gamma } ( h ) p ^ { \pi } \left( \phi ( s _ { t + h } ) = g \mid x _ { t } = x \right) ,\tag{4}
$$

Proposition 1 of Eysenbach et al. [3] shows

$$
Q _ { g } ^ { \pi } ( s , a ) = \mu _ { \infty } ^ { \pi } ( g \mid x ) .\tag{5}
$$

The occupancy measure of Eq. (4) forms a valid positive goal distribution with unit probability mass for Eq. (3), and maximizing the contrastive score is equivalent to maximizing the goal-reaching Q-function.

Under failure termination, however, post-failure states are no longer valid future goals. In such settings, we propose:

Proposition 1 (Q-function is equivalent to failure-aware discounted future-goal occupancy) Let $T _ { f }$ denote the failure horizon. We define the failure-aware discounted future-goal occupancy as

$$
\mu _ { f } ^ { \pi } ( g \mid x ) = \sum _ { h = 1 } ^ { \infty } q _ { \gamma } ( h ) p ^ { \pi } \left( \phi ( s _ { t + h } ) = g , T _ { f } > h \mid x _ { t } = x \right) ,\tag{6}
$$

Then we have the following equivalence:

$$
Q _ { g } ^ { \pi } ( s , a ) = \mu _ { f } ^ { \pi } ( g \mid x ) .\tag{7}
$$

The proof is provided in Appendix A.1. Unlike the infinite-horizon occupancy in $\mathrm { E q . } ( 4 ) , \mu _ { f } ^ { \pi }$ is NOT a valid goal distribution. It is generally a sub-probability measure because failure removes part of the discounted future occupancy. Its total mass is

$$
Z ^ { \pi } ( x ) \triangleq \int _ { \mathcal { G } } \mu _ { f } ^ { \pi } ( g \mid x ) d g = \sum _ { h = 1 } ^ { \infty } q _ { \gamma } ( h ) \operatorname* { P r } ^ { \pi } \left( T _ { f } > h \mid x \right) = \mathbb { E } _ { H \sim q _ { \gamma } } \left[ \operatorname* { P r } ^ { \pi } \left( T _ { f } > H \mid x \right) \right] \leq 1 .\tag{8}
$$

We call $Z ^ { \pi } ( x )$ survival mass. Intuitively, $Z ^ { \pi } ( x )$ is the probability that a future horizon sampled from $q _ { \gamma }$ still lies before failure.

However, established CRL uses only the valid pre-failure futures as positives and renormalizes their remaining occupancy to unit mass before entering the contrastive objective. Failure therefore changes which futures remain, but the amount of discounted occupancy removed by failure is lost after normalization. Let τ denote the observed future trajectory from anchor $x _ { t } = x ,$ and let $L _ { \tau }$ be the number of valid future states before failure. The realized failure-aware occupancy and its realized survival mass are

$$
\hat { \mu } _ { \tau } ( g \mid x ) = \sum _ { h = 1 } ^ { L _ { \tau } } q _ { \gamma } ( h ) \mathbf { 1 } \{ \phi ( s _ { t + h } ) = g \} , \quad \hat { Z } _ { \tau } = \sum _ { h = 1 } ^ { L _ { \tau } } q _ { \gamma } ( h ) = 1 - \gamma ^ { L _ { \tau } } .\tag{9}
$$

CRL samples positive goals from this realized trajectory after normalizing its valid future occupancy, so the trajectorylevel positive distribution is ${ \hat { \mu } } _ { \tau } ( g \mid x ) / { \hat { Z } } _ { \tau }$ . By construction, on average,

$$
\mathbb { E } _ { \tau \mid x } [ \hat { \mu } _ { \tau } ( g \mid x ) ] = \mu _ { f } ^ { \pi } ( g \mid x ) , \quad \mathbb { E } _ { \tau \mid x } [ \hat { Z } _ { \tau } ] = Z ^ { \pi } ( x ) .\tag{10}
$$

The population positive-goal distribution is

$$
p _ { \mathrm { C R L } } ^ { + } ( { \boldsymbol g } \mid { \boldsymbol x } ) = \mathbb { E } _ { \boldsymbol \tau \mid { \boldsymbol x } } \left[ \frac { \hat { \mu } _ { \boldsymbol \tau } ( { \boldsymbol g } \mid { \boldsymbol x } ) } { \hat { Z } _ { \boldsymbol \tau } } \right] .\tag{11}
$$

Because $Z _ { \tau } \in ( 0 , 1 ]$ , if we denote by $\hat { Q } _ { \mathrm { C R L } , g } ^ { \pi } ( s , a )$ the goal-reaching value implicitly represented by CRL, we have:

$$
\boxed { \hat { Q } _ { \mathrm { C R L } , g } ^ { \pi } ( s , a ) = p _ { \mathrm { C R L } } ^ { + } ( g \mid x ) = \mathbb { E } _ { \tau \mid x } \left[ \frac { \hat { \mu } _ { \tau } ( g \mid x ) } { \hat { Z } _ { \tau } } \right] \geq \mathbb { E } _ { \tau \mid x } [ \hat { \mu } _ { \tau } ( g \mid x ) ] = \mu _ { f } ^ { \pi } ( g \mid x ) = Q _ { g } ^ { \pi } ( s , a ) . }\tag{12}
$$

Eq. (12) establishes a non-negative overestimation bias, while the trajectory-level amplification factor $1 / \hat { Z } _ { \tau }$ increases as $L _ { \tau }$ decreases. This theoretical analysis provides a mechanism that leads to catastrophic failure bootstrapping and unsustainable goal reaching.

## 2.3 Biased critic and actor learning in CRL

Figure 3 conceptualizes two distinct consequences of the same normalization operation. First, normalizing each realized trajectory before averaging distorts the conditional goal distribution learned by the critic. Second, even after this distortion is corrected, the contrastive objective in Eq. (3) still represents a normalized distribution and therefore omits its action-dependent total survival mass.

![](images/02575a89db5916b90b671ae08a35c5f10bc82360becf061b3e8a8763f42fffed.jpg)  
Figure 3: Failure termination removes part of the discounted future-goal occupancy. Established CRL normalizes the remaining pre-failure futures, whereas Safe-CRL corrects the trajectory-wise normalization distortion in critic learning and restores the missing survival mass in actor updates.

• On critic learning: Normalizing each trajectory to unit mass before averaging gives short trajectories too much relative weight compared with long trajectories. Therefore, the positive distribution learned by established CRL generally differs from the normalized population failure-aware occupancy,

$$
p _ { \mathrm { C R L } } ^ { + } ( g \mid x ) \neq \bar { \mu } _ { f } ^ { \pi } ( g \mid x ) \triangleq \frac { \mu _ { f } ^ { \pi } ( g \mid x ) } { Z ^ { \pi } ( x ) } .\tag{13}
$$

Short near-failure trajectories are over-weighted relative to long safe trajectories, distorting the conditional goal distribution learned by the critic.

• On actor optimization: Correcting the critic-side distortion is not sufficient. Contrastive learning can recover only the normalized average conditional occupancy $\bar { \mu } _ { f } ^ { \pi }$ , whereas the true failure-aware goal-conditioned value factorizes as

$$
Q _ { g } ^ { \pi } ( s , a ) \triangleq \mu _ { f } ^ { \pi } ( g \mid x ) = Z ^ { \pi } ( x ) \bar { \mu } _ { f } ^ { \pi } ( g \mid x ) .\tag{14}
$$

The missing factor is exactly the action-dependent survival mass $Z ^ { \pi } ( x )$ . Actions with similar conditional goal distributions but very different survival masses can receive similar contrastive scores.

In summary, trajectory-wise normalization causes the critic-side trajectory-wise normalization distortion, while contrastive normalization removes the actor-side survival mass required for correct goal-conditioned action evaluation. We next rigorously correct the overestimation bias in CRL by restoring $Z ^ { \pi } ( x )$

## 3 Safe-CRL by restoring the survival mass

The two effects identified above suggest two corresponding corrections, as illustrated in the right part of Figure 3.

## 3.1 Critic-side correction: mass-weighted InfoNCE

Since each realized trajectory is normalized by its realized survival mass $\hat { Z } _ { i }$ when constructing positive goals, we restore its relative contribution by weighting the complete InfoNCE row by the same $\hat { Z } _ { i }$ . Let $\ell _ { i } ^ { \mathrm { N C E } }$ denote the complete row-wise InfoNCE loss [10, 11] for anchor i. This gives the mass-weighted InfoNCE (MW-InfoNCE):

$$
\mathcal { L } _ { \mathrm { M W - N C E } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \hat { Z } _ { i } \ell _ { i } ^ { \mathrm { N C E } } ,\tag{15}
$$

where B is the batch size. This gives the following proposition:

Proposition 2 (Mass-weighted InfoNCE corrects trajectory-wise normalization distortion) Eq. (15) induces, at the population level, the normalized conditional distribution associated with the failure-aware occupancy,

$$
p _ { \mathrm { W } } ^ { + } ( g \mid x ) = { \frac { \mu _ { f } ^ { \pi } ( g \mid x ) } { Z ^ { \pi } ( x ) } } = { \bar { \mu } } _ { f } ^ { \pi } ( g \mid x ) .\tag{16}
$$

The proof is provided in Appendix A.2. Intuitively, multiplying each row by $\hat { Z } _ { i }$ cancels the trajectory-wise normalization in expectation.

## 3.2 Actor-side correction: log-survival-mass correction

Using the factorization in Eq. (14), once the critic recovers $\bar { \mu } _ { f } ^ { \pi }$ , the normalized contrastive score differs from the desired failure-aware occupancy score by exactly − log $Z ^ { \pi } ( x )$

Corollary 1 (Restoring survival mass in actor scoring) Under the calibrated CRL score convention in Eq. (3), the optimal mass-weighted contrastive score satisfies

$$
\bar { f } _ { W } ^ { * } ( x , g ) = \log \frac { \bar { \mu } _ { f } ^ { \pi } ( g \mid x ) } { p ^ { - } ( g ) } = \log \mu _ { f } ^ { \pi } ( g \mid x ) - \log Z ^ { \pi } ( x ) - \log p ^ { - } ( g )\tag{17}
$$

Because log $p ^ { - } ( g )$ is action-independent,

$$
\arg \operatorname* { m a x } _ { a } \left[ \bar { f } _ { W } ^ { * } ( x , g ) + \log Z ^ { \pi } ( x ) \right] = \arg \operatorname* { m a x } _ { a } \mu _ { f } ^ { \pi } ( g \mid x ) = \arg \operatorname* { m a x } _ { a } Q _ { g } ^ { \pi } ( s , a ) .\tag{18}
$$

The derivation is provided in Appendix ${ \mathrm { A } } . 3 .$ . This corollary says that adding the log $Z$ term restores the missing survival mass in the actor score and recovers the same action ranking as the failure-aware goal-conditioned Q-value.

## 3.3 Implementation of corrections

Safe-CRL implements these two corrections with a lightweight Z-encoder $Z _ { \psi } ( x ) \in ( 0 , 1 )$ . For an anchor $x _ { t } .$ , its realized survival mass $\hat { Z } _ { t }$ weights the InfoNCE row as in Eq. (15). To train the Z-encoder, we independently sample $H _ { t } \sim q _ { \gamma }$ and use the binary survival label $Y _ { t } = \mathbf { 1 } \{ T _ { f } > H _ { t } \}$ . If $H _ { t }$ extends beyond the available trajectory without an observed failure, including time-limit truncation, we treat it as a survived sample and set $Y _ { t } = 1$ ; if failure is observed at or before $H _ { t } , Y _ { t } = \mathbf { \bar { 0 } }$ . The Z-encoder is trained by

$$
\mathcal { L } _ { Z } ( \boldsymbol { \psi } ) = - \mathbb { E } \left[ Y _ { t } \log Z _ { \boldsymbol { \psi } } ( x _ { t } ) + ( 1 - Y _ { t } ) \log ( 1 - Z _ { \boldsymbol { \psi } } ( x _ { t } ) ) \right] .\tag{19}
$$

For the uncensored process, $\mathbb { E } [ Y _ { t } \mid \tau _ { t } ] = \hat { Z } _ { t } \mathrm { ~ a n d ~ } \mathbb { E } [ Y _ { t } \mid x _ { t } ] = Z ^ { \pi } ( x _ { t } )$ , so Eq. (19) is a one-sample Monte Carlo estimator of the corresponding survival-mass BCE; see Appendix A.4.

For actor updates, we add the log Z term predicted by the Z-encoder to the actor objective:

$$
f _ { \mathrm { c o r r e c t e d } } ( s , a , g ) = f _ { \boldsymbol \theta } ( s , a , g ) + \log Z _ { \psi } ( s , a ) .\tag{20}
$$

In the implementation, we retain the log-sum-exp score regularization of Scaling-CRL to constrain anchor-dependent score offsets, and apply the log-Z correction under the same score convention.

In practice, log $Z _ { \psi }$ is evaluated from the Z-encoder logit output using log $\sigma ( z _ { \psi } ) = - \mathrm { s o f t p l u s } ( - z _ { \psi } )$ for numerical stability. During the actor update, the Z-encoder parameters are fixed while gradients through its action input are retained. All remaining Scaling-CRL components, including the ResNet-like network architecture, stay unchanged [5]. We call the resulting method Safe Contrastive Reinforcement Learning (Safe-CRL).

Why not model failure as an explicit future outcome? In principle, formulating failure as an absorbing outcome would preserve the missing mass. However, this changes the established CRL outcome space: failure is shared across many anchors and therefore requires special treatment in contrastive negative sampling, while it is not a valid commanded goal for actor learning. Safe-CRL instead retains the original achieved-goal space and restores the missing survival mass directly.

## 4 Experiments

![](images/a05f454b2bec9093d484a5e4a96cca4cf51975d920ef22a16d34df80d8654ed5.jpg)  
Figure 4: Illustrations of the experimental environments. The goal is represented by a bright green cylinder (for Point and Car Goal) or a meshed sphere (for Ant and Humanoid locomotion tasks). The walls in Maze environments are concrete, while other obstacles, hazards (pitfalls), and moving gremlins (purple spheres in Point Goal and Car Goal) are non-colliding (ghost objects). Contacting maze walls is not a failure; entering a designated task-specific failure region triggers failure termination. The goal is randomly respawned only for Point Goal and Car Goal. The other environments use a fixed commanded goal in each episode.

We evaluate Safe-CRL on 12 failure-prone robot navigation and locomotion tasks built in Brax [12], as illustrated in Figure 4. Point Goal and Car Goal are adopted from the level-2 Safety-Gymnasium navigation tasks [13], using the same environmental parameters. Entering an obstacle, hazard (pitfall), or moving-gremlin region causes failure, and the goal respawns after being reached. The remaining Ant and Humanoid tasks are adopted from Scaling-CRL [5]. Falling is failure in Goal and Maze tasks, while Pitfall tasks additionally terminate when the robot enters a hazard region. Locomotion goals do not respawn after being reached. We use the MJX backend [14, 15] and make locomotion more failure-prone by reducing ground friction and, for Humanoid, reducing selected actuator gears. All methods share these modified dynamics. Following the lidar-style observations used in Safety-Gymnasium [13], a 2D lidar is added to the observation to detect walls and obstacles. Occlusion is also considered in the lidar observation. Appendix B provides detailed environment and failure specifications.

Baselines and implementations We use Scaling-CRL [5] as the primary baseline because Safe-CRL directly extends its contrastive objective under failure termination, enabling a controlled comparison between Scaling-CRL and the proposed survival-mass correction. An additional hindsight-relabelling baseline, SAC-HER [16, 4], is reported in Appendix C. For the main benchmark, both methods use 64-layer actor and critic networks, except for Point Goal and Car Goal, where 8-layer networks are used. Safe-CRL additionally uses a 4-layer Z-encoder, while all other training settings are identical to Scaling-CRL [5]. We set γ = 0.99 and the episode time limit to 1000 steps for all environments.

Evaluation metrics We report three complementary metrics: (1) time at goal, the number of episode steps spent within the commanded goal region, jointly reflecting goal attainment and persistence near the goal [5, 11]; (2) survival time, the number of steps before failure termination or the episode time limit; and (3) goal coverage, the percentage of episodes in which the commanded goal is reached by the robot at least once. Comparing time at goal and goal coverage indicates whether higher time at goal is obtained at the expense of reaching fewer commanded goals. Each metric is averaged over 128 parallel evaluation environments. We report the mean and standard deviation over five independent random seeds for the main benchmark and depth-scaling experiments, and over three seeds for the remaining experiments and ablations, following Scaling-CRL [5].

## 5 Results and discussion

![](images/929f943f7ca609dd0be1f952f77a810e566e436e662256f914d7903db5f2fe62.jpg)  
Figure 5: Survival time (top two rows) and goal coverage (bottom two rows) on the 12-task benchmark (mean ± std).

Main benchmark Figure 1 shows time at goal, while Figure 5 compares survival time and goal coverage; Table D.1 provides the corresponding values. Safe-CRL achieves higher mean time at goal than Scaling-CRL in all 12 tasks. The gains are modest on Point Goal, Car Goal, and Ant H-Maze, but pronounced on the remaining nine tasks. For example, average time at goal increases from 276.8 to 684.0 steps on Ant Goal, from 22.2 to 266.4 on Humanoid Goal, and from 8.2 to 474.1 on Humanoid Big Pitfall. Safe-CRL also achieves higher mean survival time in all 12 tasks. Goal coverage increases in seven tasks and decreases in the remaining five. Despite these decreases, time at goal increases substantially in the affected environments, indicating that the gains are not simply obtained by sacrificing goal-reaching coverage.

Comparing the three metrics provides further evidence for the two failure modes discussed in Section 1. In Ant Goal, Ant Big Maze, and Ant Hardest Pitfall, differences in goal coverage are much smaller than the corresponding differences in time at goal and survival time. This pattern indicates that Scaling-CRL can reach the commanded goal but often fails to remain there safely. This is unsustainable goal reaching. For the more failure-prone Humanoid environments, Scaling-CRL’s time at goal and goal coverage grow much more slowly than Safe-CRL, while its survival time remains low throughout training. This behaviour is consistent with catastrophic failure bootstrapping, where repeated reinforcement of near-failure trajectories prevents the policy from escaping failure-prone behaviours. Overall, these learning patterns of Scaling-CRL are consistent with the two failure modes discussed in Section 1. Safe-CRL mitigates both failure modes and improves both goal-reaching performance and survival, with the largest gains in Humanoid tasks.

Safe-CRL preserves depth scalability A key finding of Scaling-CRL is that goal-reaching performance improves with deeper networks [5]. To examine whether Safe-CRL preserves this depth scalability, we fix the 4-layer Z-encoder and scale the actor and critic jointly from 4 to 64 layers, further extending to 256 layers for the more challenging Humanoid U-Maze and Humanoid Big Pitfall tasks. Figure 6 reports time at goal and survival time over five random seeds, with 64-layer Scaling-CRL included as a reference.

Figure 6 shows that time at goal and survival time generally improve as network depth increases. Their performance often saturates around 8–16 layers in the easier Ant tasks. Harder Humanoid tasks benefit from greater depth: the improvement on Humanoid Big Pitfall is pronounced up to approximately 64 layers, and Humanoid U-Maze continues to improve up to 256 layers. Notably, 8-layer Safe-CRL matches or outperforms 64-layer Scaling-CRL across all tasks evaluated in this depth-scaling study. These results show that Safe-CRL preserves CRL’s depth scalability and further improves sample efficiency.

![](images/bdaa4301b3c44c6f1170ee0a0583334a5da0c7777eafdff11458b839c75c497c.jpg)  
Figure 6: Time at goal and survival time for Safe-CRL using different actor and critic depths. The Z-encoder depth is 4. Point Goal and Car Goal are omitted from the depth-scaling study because they are easier than locomotion.

Z-encoder learns meaningful survival mass with little overhead We next examine whether the lightweight Zencoder learns a meaningful survival-mass estimate. Figure 7 compares predicted survival mass with observed survival time during training on the four Humanoid tasks. The quantities evolve with closely aligned trends: as predicted surviva mass increases, evaluated survival time generally increases as well. The aligned trends indicate that the predicted survival mass tracks failure-dependent trajectory quality during training.

The Z-encoder is computationally lightweight. Using the default 4-layer architecture increases the total number of model parameters by only approximately 4.2% (64-layer actor and critic networks) and the average wall-clock training time by 2.8% (Table D.2). Thus, the corrections introduce only modest computational overhead.

![](images/1f2acf484c75c8e64a3c2bfd4720dc4cdf17009b910c98277499f1700a6b7d99.jpg)  
Figure 7: Predicted survival mass versus survival time in 128 evaluation runs. Actor and critic depths are 64.

![](images/43b5665626b5441bc0f18538fd91097ded206c6d438a8f340a3e9881c8e96194.jpg)  
Figure 8: Visualization of the policy learned by Safe-CRL. Each row contains 5 snapshots captured from one evaluation episode, showing the key failure-avoidance behaviours of the agent.

Deeper Safe-CRL policies exhibit complex failure-avoidance behaviours We finally visualize representative rollouts of the learned 64-layer policies (8-layer for Point Goal and Car Goal) in Figure 8. Despite receiving only a one-bit failure signal, Safe-CRL learns diverse failure-avoidance behaviours while reaching the commanded goal. On Point Goal, the robot brakes and replans an S-shaped trajectory to pass between moving gremlins after the goal respawns. On Humanoid Goal, the robot walks backward toward a goal behind it and recovers from near-falls and overshooting after reaching the goal. On Humanoid Loop Pitfall, the humanoid adjusts its body orientation to avoid nearby hazard regions, while on the more difficult pitfall tasks, the humanoid and ant exhibit longer-horizon behaviours such as taking detours or slowing down to pass safely between hazards. These rollouts illustrate non-trivial failure-avoidance and recovery behaviours learned from a sparse one-bit failure signal. Additional rollout visualizations are available on the project page https://github.com/RomainLITUD/safe-crl.

Additional experimental results are provided in Appendix D.

## 6 Ablations

In this section, we mainly ablate the two corrections in Safe-CRL and examine the sensitivity to the coefficient of the log Z term. Additional ablations on Z-encoder depth, TD-based survival-mass estimation, and the effect of goal respawning are provided in Appendix D.

Core component ablations Table 1 isolates the contributions of mass-weighted InfoNCE and the log-survival-mass correction. MW-InfoNCE alone produces only modest changes relative to Scaling-CRL, whereas the log-Z correction accounts for the dominant improvement across the four tasks. For example, on Humanoid Goal, the log-Z-only variant increases time at goal from 13.8 to 288.7 steps and survival time from 113.9 to 397.0 steps, while mass-weighted InfoNCE alone achieves 28.3 and 146.1 steps, respectively. A similar pattern is observed on Ant Big Maze and Ant Hardest Pitfall. This observation is reasonable because the log Z term influences the policy learning directly, while the critic-side MW-InfoNCE corrects the distortion in contrastive learning but does not restore the missing survival mass for policy update, as explained in Section 2.3.

Table 1: Core component ablation of Safe-CRL. MW denotes MW-InfoNCE. Results are averaged over the final 10% of training and reported as mean ± standard deviation over three random seeds. The actor and critic depths are 16 (8 for Car Goal), and the Z-encoder depth is 4.
<table><tr><td rowspan="2">Method</td><td rowspan="2">MW</td><td rowspan="2">log Z</td><td colspan="2">Car Goal</td><td colspan="2">Humanoid Goal</td><td colspan="2">Ant Big Maze</td><td colspan="2">Ant Hardest Pitfall</td></tr><tr><td>Time at goal</td><td>Survival time</td><td>Time at goal</td><td>Survival time</td><td>Time at goal</td><td>Survival time</td><td>Time at goal</td><td>Survival time</td></tr><tr><td>Scaling-CRL</td><td></td><td>x</td><td>2.2±0.1</td><td> $4 6 4 . 9 { \pm } 1 1 . 2 $ </td><td>13.8±1.0</td><td>113.9±5.3</td><td>89.7±34.2</td><td>261.1±33.1</td><td>98.5±17.1</td><td>381.1±11.6</td></tr><tr><td>MW only</td><td>x √</td><td>x</td><td>2.2±0.1</td><td> $4 7 1 . 4 { \pm } 1 3 . 3 $ </td><td>28.3±0.7</td><td>146.1±10.2</td><td>91.4±31.1</td><td>291.6±51.6</td><td>113.2±19.3</td><td> $4 1 1 . 0 { \pm } 2 7 . 1 $ </td></tr><tr><td>log Z only</td><td>x</td><td>√</td><td>2.4±0.1</td><td> $5 3 7 . 1 { \pm } 6 1 . 3 $ </td><td>288.7±21.6</td><td>397.0±21.8</td><td>337.0±28.9</td><td>693.5±33.7</td><td>271.2±28.3</td><td>722.9±32.6</td></tr><tr><td>Safe-CRL</td><td>√</td><td>√</td><td>2.4±0.2</td><td>555.5±49.1</td><td>297.0±19.4</td><td>413.4±23.9</td><td>352.9±11.1</td><td>725.7±31.7</td><td>313.0±50.7</td><td>787.9±51.1</td></tr></table>

Ablations on log $Z ^ { \bullet } \mathbf { s }$ weight Under the score convention used in Section 3, the failure-aware occupancy derivation fixes the coefficient of log Z to one. To examine deviations from this value, we introduce an analysis coefficient β and compare 0.2, 1.0, and 5.0 in Figure 9. Increasing β generally increases survival time, whereas its effect on goal-reaching performance is environment-dependent. Safe-CRL fixes $\beta = 1$ because it is theory-derived rather than tuned as a safety–goal trade-off parameter; the experiment does not assume that this value maximizes every empirical metric in every environment.

![](images/3c7753e805ef7e2eda194e235c3f5a1c5f8dfbd5c0253cdcf9545f7a209bb422.jpg)  
Figure 9: Ablations on the coefficient β of log Z. The left four panels report time at goal, and the right four panels report survival time. We set the actor and critic depths to 16 layers (8 for Car Goal) and vary $\beta .$

## 7 Conclusion

This paper studied contrastive reinforcement learning under failure termination and identified a systematic overestimation bias caused by the missing survival mass of the failure-aware discounted future-goal occupancy. The analysis of this bias led to two minimal corrections, mass-weighted InfoNCE for critic learning and a log-survival-mass correction for policy optimization, which together form Safe-CRL. Across failure-prone navigation and locomotion tasks, Safe-CRL improves survival and goal-reaching performance while preserving the depth scalability of CRL, using only the one-bit failure signal provided by termination. More broadly, our results suggest that under failure termination, scalable self-supervised policy learning should preserve not only the relative structure of observed futures, but also the total occupancy mass removed by termination.

Limitations We point out three limitations here: First, Safe-CRL is designed for failure-terminated goal-conditioned control and does not directly address non-terminating safety costs or constrained-RL objectives. Second, time-limit truncation also introduces right censoring. In this study, we retain CRL’s finite-support treatment for future relabelling and treat Z-encoder horizons without an observed failure as survived. Future work could incorporate the right-censoring correction proposed in SVL [17]. Finally, Safe-CRL retains the exploration limitations of CRL, which can result in uneven goal-space coverage in difficult tasks (see an analysis in Figure D.4, Appendix D). Addressing these settings is also left for future work.

## 8 Related work

Finally, we briefly review the work most closely related to this study. Goal-conditioned reinforcement learning (GCRL) commonly uses achieved future states through hindsight relabelling, transforming sparse goal-reaching experience into dense self-supervision [4]. C-learning [18] formulates goal-conditioned value estimation as classification between future and independently sampled states, while contrastive reinforcement learning (CRL) [3] connects contrastive density-ratio estimation to discounted future-state occupancy and goal-conditioned value learning. More recently, Scaling-CRL [5] showed that this self-supervised objective can exploit substantially deeper networks, establishing CRL as a promising regime for scalable self-supervised RL.

For CRL specifically, several studies have examined whether distributions constructed from observed future outcomes preserve all information required for goal-conditioned control. Distributional Distance Classifiers (DDC) [19] separate horizon-dependent goal-reaching quantities from reachability probability. USHER [20] corrects the Hindsight-selection bias under stochastic dynamics, where conditioning on observed outcomes can under-represent unfavourable transitions and bias goal-conditioned estimates. These works show that conditioning or learning from observed future outcomes can distort goal-reaching quantities in ways that matter for control. Their focus is horizon dependence or stochastic hindsight selection, whereas our work studies a different methodological issue: the loss of total discounted future-goal occupancy mass when failure-terminated pre-failure futures are normalized. Recent work also extends single-goal contrastive RL to safety-aware exploration by exploiting its representation-induced exploration mechanism [21].

A complementary line of recent studies reformulates GCRL through survival analysis. Survival Value Learning (SVL) [17] models time to goal using event and right-censored trajectories, while Survival Reinforcement Learning (SRL) [22] extends this formulation to online self-supervised learning and introduces goal “dwell-time” to encourage stable long-horizon behaviour. These works are conceptually adjacent in highlighting temporal information overlooked by standard goal-reaching objectives, but address a different formulation: SVL and SRL explicitly model time-toevent distributions, whereas Safe-CRL retains the CRL occupancy formulation and identifies a normalization-induced overestimation bias caused by missing survival mass under failure termination.

It is useful to mention a separate line of safe reinforcement learning that formulates safety through constrained Markov decision processes (CMDPs), where a policy maximizes task reward subject to explicit constraints on expected cumulative costs [23]. Representative methods include Constrained Policy Optimization (CPO), which directly enforces policy constraints during optimization [24], Lyapunov-based safe RL [25], and Lagrangian methods [26]. These methods address an important but different safety setting from ours: they assume an explicit cost or constraint signal and optimize a constrained objective. An “unsafe event” does NOT necessarily terminate the episode. Safe-CRL instead considers failure-terminated goal-conditioned learning, where termination provides the only one-bit failure signal.

In summary, existing work has studied scalable CRL, stochastic and horizon-dependent goal reachability, and survivalbased goal-conditioned learning. To our knowledge, prior work has not identified the systematic overestimation bias that arises when established CRL normalizes pre-failure futures and discards the survival mass intrinsic to the failure-aware discounted future-goal occupancy, nor the accompanying trajectory-wise normalization distortion in critic learning.

## References

[1] Max Schwarzer, Johan Samir Obando Ceron, Aaron Courville, Marc G Bellemare, Rishabh Agarwal, and Pablo Samuel Castro. Bigger, better, faster: Human-level Atari with human-level efficiency. In International Conference on Machine Learning, pages 30365–30380. PMLR, 2023.

[2] Hojoon Lee, Dongyoon Hwang, Donghu Kim, Hyunseung Kim, Jun Jet Tai, Kaushik Subramanian, Peter R. Wurman, Jaegul Choo, Peter Stone, and Takuma Seno. SimBa: Simplicity bias for scaling up parameters in deep reinforcement learning. In The Thirteenth International Conference on Learning Representations, 2025.

[3] Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, and Ruslan Salakhutdinov. Contrastive learning as goalconditioned reinforcement learning. In Advances in Neural Information Processing Systems, volume 35, 2022.

[4] Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Advances in Neural Information Processing Systems, volume 30, 2017.

[5] Kevin Wang, Ishaan Javali, Michał Bortkiewicz, Tomasz Trzcinski, and Benjamin Eysenbach. 1000 layer´ networks for self-supervised RL: Scaling depth can enable new goal-reaching capabilities. In Advances in Neural Information Processing Systems, volume 38, 2025.

[6] Glen Berseth, Daniel Geng, Coline M. Devin, Nicholas Rhinehart, Chelsea Finn, Dinesh Jayaraman, and Sergey Levine. SMiRL: Surprise minimizing reinforcement learning in unstable environments. In International Conference on Learning Representations, 2021.

[7] Deepali Jain, Ken Caluwaerts, and Atil Iscen. From pixels to legs: Hierarchical learning of quadruped locomotion. In Proceedings of the 2020 Conference on Robot Learning, volume 155 of Proceedings of Machine Learning Research, pages 91–102. PMLR, 2021.

[8] Fethi Belkhouche, Boumediene Belkhouche, and Parviz Rastgoufard. Line of sight robot navigation toward a moving goal. IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics, 36(2):255–267, 2006.

[9] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst. In Robotics: Science and Systems, 2019.

[10] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[11] Michał Bortkiewicz, Władysław Pałucki, Vivek Myers, Tadeusz Dziarmaga, Tomasz Arczewski, Łukasz Kucinski,´ and Benjamin Eysenbach. Accelerating goal-conditioned reinforcement learning algorithms and research. In International Conference on Learning Representations, 2025.

[12] C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax: A differentiable physics engine for large scale rigid body simulation. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

[13] Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Juntao Dai, and Yaodong Yang. Safety-Gymnasium: A unified safe reinforcement learning benchmark. In Advances in Neural Information Processing Systems, volume 36, pages 18964–18993, 2023.

[14] Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. IEEE, 2012.

[15] Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder A. Kahrs, Carmelo Sferrazza, Yuval Tassa, and Pieter Abbeel. MuJoCo Playground. arXiv preprint arXiv:2502.08844, 2025.

[16] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 1861–1870. PMLR, 2018.

[17] Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, and Justin Carpentier. SVL: Goal-conditioned reinforcement learning as survival learning. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research. PMLR, 2026.

[18] Benjamin Eysenbach, Ruslan Salakhutdinov, and Sergey Levine. C-learning: Learning to achieve goals via recursive classification. In International Conference on Learning Representations, 2021.

[19] Ravi Tej Akella, Benjamin Eysenbach, Jeff Schneider, and Ruslan Salakhutdinov. Distributional distance classifier for goal-conditioned reinforcement learning. In ICML 2023 Workshops: Frontiers4LCD, 2023.

[20] Liam Schramm, Yunfu Deng, Edgar Granados, and Abdeslam Boularias. USHER: Unbiased sampling for hindsight experience replay. In Proceedings ofThe 6th Conference on Robot Learning, volume 205 of Proceedings ofMachine Learning Research, pages 2073–2082. PMLR, 2023.

[21] Mahsa Bastankhah, Grace Liu, Dilip Arumugam, Thomas L. Griffiths, and Benjamin Eysenbach. Demystifying the mechanisms behind emergent exploration in goal-conditioned RL. In The Fourteenth International Conference on Learning Representations, 2026.

[22] Franki Nguimatsia-Tiofack, Fabian Schramm, Théotime Le Hellard, and Justin Carpentier. Survival reinforcement learning: Toward scalable self-supervised RL. arXiv preprint arXiv:2605.31273, 2026.

[23] Eitan Altman. Constrained Markov Decision Processes. Chapman & Hall/CRC, Boca Raton, FL, 1999.

[24] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 22–31. PMLR, 2017.

[25] Yinlam Chow, Ofir Nachum, Edgar A. Duéñez-Guzmán, and Mohammad Ghavamzadeh. A Lyapunov-based approach to safe reinforcement learning. In Advances in Neural Information Processing Systems, volume 31, pages 8103–8112, 2018.

[26] Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive safety in reinforcement learning by PID Lagrangian methods. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceeding ofMachine Learning Research, pages 9133–9143. PMLR, 2020.

[27] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 1587–1596. PMLR, 2018.

## A Proofs and technical details

This appendix provides the detailed derivations for the theoretical results in Sections 2 and 3. We first show that the failure-aware discounted future-goal occupancy is the correct goal-conditioned value, then prove that survival-mass weighting in the InfoNCE loss corrects the trajectory-wise normalization distortion induced by future relabelling. Finally, we derive the log-survival-mass correction for actor optimization.

## A.1 Proof of Proposition 1

We prove Proposition 1 using measurable goal sets, which avoids point-mass notation for continuous goal spaces. For an anchor $x = ( s , a )$ and a measurable goal set $B \subseteq { \mathcal { G } }$ , define the failure horizon as

$$
T _ { f } \triangleq \operatorname* { i n f } \left\{ h \geq 1 : F ( s _ { t + h - 1 } , a _ { t + h - 1 } , s _ { t + h } ) = 1 \right\} ,\tag{A.1}
$$

with $T _ { f } = \infty$ if no failure occurs. Therefore, $T _ { f } > h$ means that $S _ { t + h }$ lies on the valid pre-failure portion of the trajectory. The corresponding failure-aware discounted future-goal occupancy is

$$
\mu _ { f } ^ { \pi } ( B \mid x ) = \sum _ { h = 1 } ^ { \infty } q _ { \gamma } ( h ) \operatorname* { P r } ^ { \pi } \left( \phi ( S _ { t + h } ) \in B , T _ { f } > h \mid x _ { t } = x \right) , \quad q _ { \gamma } ( h ) = ( 1 - \gamma ) \gamma ^ { h - 1 } .\tag{A.2}
$$

Consider the normalized goal-reaching transition reward

$$
r _ { B } ( s , a , s ^ { \prime } ) = ( 1 - \gamma ) \mathbf { 1 } \left\{ F ( s , a , s ^ { \prime } ) = 0 , \phi ( s ^ { \prime } ) \in B \right\} .\tag{A.3}
$$

The corresponding discounted action value is

$$
Q _ { B } ^ { \pi } ( s , a ) = \mathbb { E } ^ { \pi } \left[ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r _ { B } ( s _ { t + k } , a _ { t + k } , s _ { t + k + 1 } ) \mid s _ { t } = s , a _ { t } = a \right] .\tag{A.4}
$$

Because failure immediately terminates the episode, a valid goal-reaching reward at horizon $k + 1$ is obtained only when $T _ { f } > k + 1$ . Substituting the reward definition gives

$$
Q _ { B } ^ { \pi } ( s , a ) = \sum _ { k = 0 } ^ { \infty } ( 1 - \gamma ) \gamma ^ { k } \operatorname* { P r } ^ { \pi } \left( \phi ( s _ { t + k + 1 } ) \in B , T _ { f } > k + 1 \mid x _ { t } = x \right) .\tag{A.5}
$$

Letting $h = k + 1$ directly yields

$$
Q _ { B } ^ { \pi } \big ( s , a \big ) = \sum _ { h = 1 } ^ { \infty } q _ { \gamma } ( h ) \overset { \pi } { \operatorname* { P r } } \left( \phi ( S _ { t + h } ) \in B , T _ { f } > h \mid x _ { t } = x \right) = \mu _ { f } ^ { \pi } \big ( B \mid x \big ) .\tag{A.6}
$$

This proves Proposition 1. For discrete goals, or equivalently using density notation for continuous goal spaces, this gives the pointwise relation used in the main text,

$$
Q _ { g } ^ { \pi } ( s , a ) = \mu _ { f } ^ { \pi } ( g \mid x ) .\tag{A.7}
$$

Taking $B = \mathcal { G }$ further gives

$$
Z ^ { \pi } ( x ) = \mu _ { f } ^ { \pi } ( \mathcal { G } \mid x ) = \sum _ { h = 1 } ^ { \infty } q _ { \gamma } ( h ) \operatorname* { P r } ^ { \pi } ( T _ { f } > h \mid x ) ,\tag{A.8}
$$

which is exactly the defined survival mass.

## A.2 Proof of Proposition 2

Consider an anchor $x _ { t } = x = ( s , a )$ and a realized trajectory $\tau$ from this anchor. Let $L _ { \tau }$ denote the number of valid future states before failure. For a measurable goal set ${ \bar { B } } \subseteq { \mathcal { G } }$ , define the realized failure-aware discounted future-goal measure as

$$
\hat { \mu } _ { \tau } ( B \mid x ) = \sum _ { h = 1 } ^ { L _ { \tau } } q _ { \gamma } ( h ) \mathbf { 1 } \{ \phi ( S _ { t + h } ) \in B \} .\tag{A.9}
$$

Its total mass is

$$
\hat { Z } _ { \tau } = \hat { \mu } _ { \tau } ( \mathcal { G } \mid x ) = \sum _ { h = 1 } ^ { L _ { \tau } } q _ { \gamma } ( h ) = 1 - \gamma ^ { L _ { \tau } } .\tag{A.10}
$$

Conditioned on the realized trajectory τ , established CRL samples positive goals only from these valid pre-failure futures. Therefore, for $\hat { Z } _ { \tau } > 0$ , the corresponding normalized positive-goal distribution is

$$
r _ { \tau } ( B \mid x ) = \frac { \hat { \mu } _ { \tau } ( B \mid x ) } { \hat { Z } _ { \tau } } .\tag{A.11}
$$

Without mass weighting, averaging this distribution over trajectories gives the established CRL positive distribution,

$$
p _ { \mathrm { C R L } } ^ { + } ( B \mid x ) = \mathbb { E } _ { \tau \mid x } \left[ r _ { \tau } ( B \mid x ) \right] = \mathbb { E } _ { \tau \mid x } \left[ \frac { \hat { \mu } _ { \tau } ( B \mid x ) } { \hat { Z } _ { \tau } } \right] .\tag{A.12}
$$

In general,

$$
\mathbb { E } _ { \tau | x } \left[ \frac { \hat { \mu } _ { \tau } ( B \mid x ) } { \hat { Z } _ { \tau } } \right] \neq \frac { \mathbb { E } _ { \tau | x } \left[ \hat { \mu } _ { \tau } ( B \mid x ) \right] } { \mathbb { E } _ { \tau | x } \left[ \hat { Z } _ { \tau } \right] } .\tag{A.13}
$$

Thus, trajectory-wise normalization does not generally recover the normalized population failure-aware occupancy. We now consider the mass-weighted InfoNCE objective. Multiplying the complete contrastive row associated with trajectory $\tau$ by $\hat { Z } _ { \tau }$ induces the weighted positive measure

$$
\nu _ { W } ( B \mid x ) \triangleq \mathbb { E } _ { \tau \mid x } \left[ { \hat { Z } } _ { \tau } r _ { \tau } ( B \mid x ) \right] .\tag{A.14}
$$

Using the definition of $r _ { \tau }$ , the trajectory-wise normalization cancels exactly:

$$
\hat { Z } _ { \tau } r _ { \tau } ( B \mid x ) = \hat { \mu } _ { \tau } ( B \mid x ) .\tag{A.15}
$$

Therefore,

$$
\nu _ { W } ( B \mid x ) = \mathbb { E } _ { \tau \mid x } \left[ { \hat { \mu } } _ { \tau } ( B \mid x ) \right] = \mu _ { f } ^ { \pi } ( B \mid x ) .\tag{A.16}
$$

Taking $B = \mathcal { G }$ gives the total weighted mass

$$
\nu _ { W } ( \mathcal { G } \mid x ) = \mathbb { E } _ { \tau \mid x } \left[ \hat { Z } _ { \tau } \right] = Z ^ { \pi } ( x ) .\tag{A.17}
$$

${ \mathrm { S o } } ,$ the normalized positive distribution induced by mass-weighted InfoNCE is

$$
p _ { W } ^ { + } ( B \mid x ) = \frac { \nu _ { W } ( B \mid x ) } { \nu _ { W } ( \mathcal { G } \mid x ) } = \frac { \mu _ { f } ^ { \pi } ( B \mid x ) } { Z ^ { \pi } ( x ) } .\tag{A.18}
$$

Equivalently, using the density notation adopted in the main text,

$$
p _ { W } ^ { + } ( g \mid x ) = \frac { \mu _ { f } ^ { \pi } ( g \mid x ) } { Z ^ { \pi } ( x ) } = \bar { \mu } _ { f } ^ { \pi } ( g \mid x ) .\tag{A.19}
$$

This proves Proposition 2. Mass weighting therefore corrects the trajectory-wise normalization distortion and recover the correct conditional shape of the failure-aware occupancy. However,

$$
\int _ { \mathcal { G } } p _ { W } ^ { + } ( g \mid x ) d g = 1 ,\tag{A.20}
$$

so the contrastive distribution remains normalized, and the total survival mass $Z ^ { \pi } ( x )$ is still absent from the critic score.   
This remaining factor is restored in actor optimization by the log-survival-mass correction derived in Appendix A.3.

For anchors with $Z ^ { \pi } ( x ) = 0 { \mathrm { . } }$ , the normalized distribution above is undefined, but such anchors receive zero weight in the mass-weighted population objective; this case is also discussed separately in Appendix A.5.

Finite-minibatch estimator For a mini-batch of B independently sampled anchor-trajectory pairs, the implemented critic loss is

$$
\widehat { \mathcal { L } } _ { \mathrm { M W - N C E } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \hat { Z } _ { i } \ell _ { i } ^ { \mathrm { N C E } } ,\tag{A.21}
$$

where $\ell _ { i } ^ { \mathrm { { N C E } } }$ denotes the complete row-wise InfoNCE loss for anchor i. Its expectation satisfies

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { \mathcal { L } } _ { \mathrm { M W - N C E } } \right] = \mathbb { E } \left[ \hat { Z } \ell ^ { \mathrm { N C E } } \right] . } \end{array}\tag{A.22}
$$

Thus, the unnormalized mini-batch average is a direct Monte Carlo estimator of the population mass-weighted contrastive risk. No normalization of the weights within each mini-batch is required.

In particular, the self-normalized alternative

$$
\frac { \sum _ { i = 1 } ^ { B } \hat { Z } _ { i } \ell _ { i } ^ { \mathrm { N C E } } } { \sum _ { i = 1 } ^ { B } \hat { Z } _ { i } }\tag{A.23}
$$

contains a random denominator and is not the same finite-minibatch estimator as the objective used in Safe-CRL.

The negative goals appearing in each InfoNCE row remain sampled from the ordinary mini-batch goal marginal, as in established CRL. Row weighting changes the contribution of the anchor-positive pair to the population risk but does not require the negative proposal to be re-weighted. The resulting negative proposal therefore remains anchor-independent, as required by the density-ratio interpretation used in Appendix $\mathbf { \bar { A } } . 3$

## A.3 Proof of Corollary 1

Let $p ^ { - } ( g )$ denote the anchor-independent negative proposal. Under the support condition stated in Appendix A.5, the population optimum of InfoNCE identifies the positive-to-negative density ratio up to an anchor-dependent additive offset:

$$
f _ { W } ^ { * } ( x , g ) = \log \frac { p _ { W } ^ { + } ( g \mid x ) } { p ^ { - } ( g ) } + c ( x ) .\tag{A.24}
$$

Because $x = ( s , a )$ , the offset $c ( x )$ may in general depend on the action. Therefore, the row-wise InfoNCE objective alone does not make scores directly comparable across different actions. To remove this arbitrary offset, define the calibrated contrastive score

$$
\bar { f } _ { W } ( x , g ) = f _ { W } ( x , g ) - \log \mathbb { E } _ { g ^ { \prime } \sim p ^ { - } } \left[ \exp f _ { W } ( x , g ^ { \prime } ) \right] .\tag{A.25}
$$

At the population optimum,

$$
\mathbb { E } _ { g ^ { \prime } \sim p ^ { - } } \left[ \exp f _ { W } ^ { * } ( x , g ^ { \prime } ) \right] = \mathbb { E } _ { g ^ { \prime } \sim p ^ { - } } \left[ \exp \left( \log \frac { p _ { W } ^ { + } ( g ^ { \prime } \mid x ) } { p ^ { - } ( g ^ { \prime } ) } + c ( x ) \right) \right] .\tag{A.26}
$$

Therefore,

$$
\mathbb { E } _ { g ^ { \prime } \sim p ^ { - } } \left[ \exp f _ { W } ^ { * } ( x , g ^ { \prime } ) \right] = e ^ { c ( x ) } \int _ { \mathcal { G } } p ^ { - } ( g ^ { \prime } ) \frac { p _ { W } ^ { + } ( g ^ { \prime } \mid x ) } { p ^ { - } ( g ^ { \prime } ) } d g ^ { \prime } .\tag{A.27}
$$

Because $p _ { W } ^ { + } ( \cdot \mid x )$ is a normalized distribution,

$$
\int _ { \mathcal { G } } p _ { W } ^ { + } ( g ^ { \prime } \mid x ) d g ^ { \prime } = 1 ,\tag{A.28}
$$

and therefore

$$
\begin{array} { r } { \mathbb { E } _ { g ^ { \prime } \sim p ^ { - } } \left[ \exp f _ { W } ^ { * } ( x , g ^ { \prime } ) \right] = e ^ { c ( x ) } . } \end{array}\tag{A.29}
$$

Substituting this result into the calibrated score gives

$$
\bar { f } _ { W } ^ { * } ( x , g ) = \log \frac { p _ { W } ^ { + } ( g \mid x ) } { p ^ { - } ( g ) } .\tag{A.30}
$$

Using Proposition 2, we have:

$$
p _ { W } ^ { + } ( g \mid x ) = \frac { \mu _ { f } ^ { \pi } ( g \mid x ) } { Z ^ { \pi } ( x ) } ,\tag{A.31}
$$

so that

$$
\bar { f } _ { W } ^ { * } ( x , g ) = \log \frac { \mu _ { f } ^ { \pi } ( g \mid x ) } { Z ^ { \pi } ( x ) p ^ { - } ( g ) } .\tag{A.32}
$$

Equivalently,

$$
\begin{array} { r } { \bar { f } _ { W } ^ { * } ( x , g ) = \log \mu _ { f } ^ { \pi } ( g \mid x ) - \log Z ^ { \pi } ( x ) - \log p ^ { - } ( g ) . } \end{array}\tag{A.33}
$$

Adding the log survival mass therefore yields

$$
\bar { f } _ { W } ^ { * } ( x , g ) + \log Z ^ { \pi } ( x ) = \log \mu _ { f } ^ { \pi } ( g \mid x ) - \log p ^ { - } ( g ) .\tag{A.34}
$$

For a fixed state s and commanded goal g, the negative proposal $p ^ { - } ( g )$ is independent of the candidate action a. So, we have:

$$
\arg \operatorname* { m a x } _ { a } \left[ \bar { f } _ { W } ^ { * } ( s , a , g ) + \log Z ^ { \pi } ( x ) \right] = \arg \operatorname* { m a x } _ { a } \log \mu _ { f } ^ { \pi } ( g \mid x ) .\tag{A.35}
$$

Because the logarithm is strictly increasing,

$$
\arg \operatorname* { m a x } _ { a } \left[ \bar { f } _ { W } ^ { * } ( s , a , g ) + \log Z ^ { \pi } ( x ) \right] = \arg \operatorname* { m a x } _ { a } \mu _ { f } ^ { \pi } ( g \mid x ) .\tag{A.36}
$$

Finally, Proposition 1 gives

$$
\mu _ { f } ^ { \pi } ( g \mid x ) = Q _ { g } ^ { \pi } ( s , a ) ,\tag{A.37}
$$

and

$$
\arg \operatorname* { m a x } _ { a } \left[ \bar { f } _ { W } ^ { * } ( s , a , g ) + \log Z ^ { \pi } ( x ) \right] = \arg \operatorname* { m a x } _ { a } Q _ { g } ^ { \pi } ( s , a ) .\tag{A.38}
$$

This proves Corollary 1. The coefficient of the log-Z correction is exactly 1.0 because the failure-aware occupancy factorizes multiplicatively as

$$
\mu _ { f } ^ { \pi } ( g \mid x ) = Z ^ { \pi } ( x ) p _ { W } ^ { + } ( g \mid x ) .\tag{A.39}
$$

So, log $Z ^ { \pi } ( x )$ is not introduced as a trade-off term, but follows directly from restoring the total mass removed by contrastive normalization.

## A.4 Population target of the Z-encoder

The Z-encoder is trained from a one-sample Monte Carlo survival target. For a trajectory τ, we sample $H \sim q _ { \gamma }$ and define $Y = \mathbf { 1 } \{ T _ { f } > H \}$ . For the uncensored process,

$$
\mathbb { E } _ { H } [ Y \mid \tau ] = \sum _ { h = 1 } ^ { L _ { \tau } } q _ { \gamma } ( h ) = \hat { Z } _ { \tau } , \qquad \mathbb { E } _ { \tau , H } [ Y \mid x ] = \mathbb { E } _ { \tau } [ \hat { Z } _ { \tau } \mid x ] = Z ^ { \pi } ( x ) .\tag{A.40}
$$

For a fixed anchor x, the corresponding binary cross-entropy risk is

$$
\begin{array} { r } { \mathcal { L } _ { x } ( z ) = - \mathbb { E } \left[ Y \log z + ( 1 - Y ) \log ( 1 - z ) ~ | ~ x \right] , \qquad z \in ( 0 , 1 ) . } \end{array}\tag{A.41}
$$

Its population minimizer is

$$
Z _ { \psi } ^ { * } ( x ) = \mathbb { E } [ Y \mid x ] = Z ^ { \pi } ( x ) .\tag{A.42}
$$

Because binary cross-entropy is affine in its target label, sampling one $Y$ provides an unbiased Monte Carlo estimator of the soft-target BCE based on $\hat { Z } _ { \tau }$ . Under time-limit truncation, horizons without an observed failure are treated as survived, yielding the finite-support approximation discussed in Appendix A.5.

## A.5 Technical conditions

The theoretical results above rely on the following conditions.

(1) Replay-induced trajectory distribution The notation π denotes the replay-induced trajectory distribution represented by the data used for contrastive learning. When replay contains trajectories collected under different commanded goals, π is the corresponding goal-marginalized replay-induced trajectory distribution; the same replayinduced trajectory distribution must be used consistently in the definitions of $\mu _ { f } ^ { \pi }$ and $\dot { Z } ^ { \pi }$

(2) Support of the negative proposal For the contrastive density ratio to be well-defined, the negative proposal satisfies

$$
p ^ { - } ( g ) > 0 \qquad \mathrm { w h e n e v e r } \qquad p _ { \mathrm { W } } ^ { + } ( g | x ) > 0 .\tag{A.43}
$$

This is the standard support condition for contrastive density-ratio estimation.

(3) Continuous goal spaces For continuous goal spaces, expressions such as $\mu _ { f } ^ { \pi } ( g | x )$ and $p _ { \mathrm { W } } ^ { + } ( g | x )$ denote densities with respect to a common dominating measure. Equivalently, all results can be stated directly for measurable goal sets without assuming the existence of densities.

(4) Zero-mass trajectories If a trajectory contains no valid future state for an anchor, then

$$
L _ { \tau } = 0 , \qquad \hat { Z } _ { \tau } = 0 .\tag{A.44}
$$

Such an anchor has zero weight in the mass-weighted contrastive objective and therefore does not require a valid positive goal in the population formulation. In implementation, rows with $L _ { \tau } = 0$ are masked from the InfoNCE objective before positive-goal sampling.

(5) Failure termination versus right censoring The theoretical quantities $\mu _ { f } ^ { \pi }$ and $Z ^ { \pi }$ are defined for the uncensored failure-terminated process. In implementation, following Scaling-CRL [5, 11], futures beyond the available replay support or time limit are marked as invalid, and we do not apply an explicit right-censoring correction. When this support ends because of time-limit truncation rather than failure, the trajectory is right-censored in survival-analysis terminology [17]. Consequently, $\hat { Z } _ { t }$ should be interpreted operationally as the realized survival mass of the future support available to the sampler. Importantly, the sample-wise identity underlying Proposition 2 remains exact on this observed support,

$$
\hat { Z } _ { \tau } r _ { \tau } ( \boldsymbol { g } \mid \boldsymbol { x } ) = \hat { \mu } _ { \tau } ( \boldsymbol { g } \mid \boldsymbol { x } ) .\tag{A.45}
$$

Therefore, mass weighting still exactly cancels the trajectory-wise normalization introduced by the implemented future sampler. For Z-encoder training, we instead sample $H \sim q _ { \gamma }$ and use $Y = \mathbf { 1 } \{ T _ { f } > H \}$ . Horizons extending beyond the available trajectory without an observed failure are treated as valid survival samples with $Y = 1 ;$ horizons crossing an observed failure remain valid with $Y = 0$ . Without censoring, $\mathbb { E } [ Y \mid x ] = { \dot { Z } } ^ { \pi } ( x )$ . Under time-limit truncation, treating horizons without an observed failure as survived yields a finite-support approximation that can overestimate the uncensored $Z ^ { \pi } ( x )$ . We do not apply an explicit censoring correction.

## B Environments

We evaluate Safe-CRL on 12 failure-prone goal-conditioned robot control tasks implemented in Brax [12] with the MJX backend [14, 15], providing richer friction and contact models. In all environments, failure produces a one-bit failure signal and immediately terminates the episode. Time-limit truncation is not treated as failure.

Navigation tasks Point Goal and Car Goal are adopted from the level-2 Safety-Gymnasium navigation tasks [13]. Goals respawn after being reached, and entering a designated obstacle, hazard, or moving-gremlin region triggers failure termination. The number of obstacles, hazards, gremlins, and their sizes, layout generation rules, playground sizes, 3-channel 16-bin 2D lidar observations, and ego-robot states are all identical to the original Safety-Gymnasium package.

Locomotion tasks The Ant and Humanoid tasks are adapted from Scaling-CRL [5] and JaxGCRL [11]. Goals remain fixed after being reached. Falling terminates all locomotion tasks, while Pitfall variants additionally terminate when the robot enters a designated hazard region. Maze-wall contact itself is not treated as failure. For locomotion tasks, we also equip the agents with a 16-bin, one-channel 2D lidar to detect surrounding walls and pitfalls.

Table B.1 summarizes the task-specific termination rules. To make the locomotion tasks sufficiently failure-prone, we reduce the ground sliding friction coefficient from 1.0 to 0.6, creating a slippery ground. In particular, for the Humanoid robot, the actuator gears of the abdomen, lower limbs, and arms are reduced, as listed in Table B.2. These modified dynamics are shared by both the Scaling-CRL baseline and the proposed Safe-CRL.

## C SAC-HER baseline

We additionally compare Safe-CRL with SAC-HER, combining Soft Actor-Critic [16] with Hindsight Experience Replay [4]. The implementation is based on JaxGCRL [11]. Scaling-CRL has previously shown that contrastive selfsupervision substantially outperforms conventional HER-based goal-conditioned RL in large-scale policy learning [5]. Here, SAC-HER serves only as an additional reference for conventional hindsight-based goal-conditioned RL in the same failure-terminated setting. This comparison is not intended to test the CRL-specific missing-survival-mass mechanism identified in Section 2. As shown in Figure C.1, SAC-HER performs substantially worse than Safe-CRL on the four representative failure-prone tasks, particularly on the more challenging locomotion environments.

## D Supplementary results and ablations

This appendix provides additional quantitative results and analyses that complement the main experiments.

Table B.1: Summary of the experimental environments and failure mechanisms.
<table><tr><td>Environment</td><td>Agent</td><td>Goal Respawn</td><td>Failure Termination</td></tr><tr><td>Point Goal</td><td>Point</td><td>Yes</td><td>Enter obstacle, hazard, or gremlin region</td></tr><tr><td>Car Goal</td><td>Car</td><td>Yes</td><td>Enter obstacle, hazard, or gremlin region</td></tr><tr><td>Ant Goal</td><td>Ant</td><td>No</td><td>Fall</td></tr><tr><td>Humanoid Goal</td><td>Humanoid</td><td>No</td><td>Fall</td></tr><tr><td>Ant H-Maze</td><td>Ant</td><td>No</td><td>Fall</td></tr><tr><td>Ant Cross Maze</td><td>Ant</td><td>No</td><td>Fall</td></tr><tr><td>Ant Big Maze</td><td>Ant</td><td>No</td><td>Fall</td></tr><tr><td>Humanoid U-Maze</td><td>Humanoid</td><td>No</td><td>Fall</td></tr><tr><td>Ant Big Pitfall</td><td>Ant</td><td>No</td><td>Fall or enter hazard region</td></tr><tr><td>Ant Hardest Pitfall</td><td>Ant</td><td>No</td><td>Fall or enter hazard region</td></tr><tr><td>Humanoid Loop Pitfall</td><td>Humanoid</td><td>No</td><td>Fall or enter hazard region</td></tr><tr><td>Humanoid Big Pitfall</td><td>Humanoid</td><td>No</td><td>Fall or enter hazard region</td></tr></table>

Table B.2: Humanoid actuator gears under the default setting and our experiments.
<table><tr><td>Actuator gear</td><td>default</td><td>ours</td></tr><tr><td>abdomen_y</td><td>350</td><td>100</td></tr><tr><td>abdomen_z</td><td>350</td><td>100</td></tr><tr><td>abdomen_x  $\mathtt { r i g h t \_ h i p \_ x }$ </td><td>350 350</td><td>100 100</td></tr><tr><td> $\mathtt { r i g h t \_ h i p \_ z }$ </td><td>350</td><td>100</td></tr><tr><td> $\mathtt { r i g h t \_ h i p \_ y }$  right_knee</td><td>350</td><td>300</td></tr><tr><td> $\mathtt { l e f t \_ h i p \_ x }$ </td><td>350 350</td><td>200 100</td></tr><tr><td> $\mathtt { l e f t \_ h i p \_ z }$ </td><td>350</td><td>100</td></tr><tr><td> $\mathtt { l e f t \_ h i p \_ y }$ </td><td>350</td><td>300</td></tr><tr><td>left_knee</td><td>350</td><td></td></tr><tr><td>right_shoulder1</td><td></td><td>200</td></tr><tr><td></td><td>100</td><td>25</td></tr><tr><td>right_shoulder2</td><td>100</td><td>25</td></tr><tr><td>right_elbow</td><td>100</td><td>25</td></tr><tr><td>left_shoulder1</td><td></td><td></td></tr><tr><td></td><td>100</td><td>25</td></tr><tr><td>left_shoulder2</td><td>100</td><td>25</td></tr><tr><td>left_elbow</td><td>100</td><td>25</td></tr></table>

![](images/632dab0b1ceab283d7c46d04c23edb2c17d51abaa60997243b4647164b42f14a.jpg)  
Figure C.1: Comparison with SAC-HER on four representative failure-prone tasks. Curves report time at goal and survival time. Safe-CRL uses 16 layers for the actor and critic networks (8 for Car Goal) and 4 layers for the Z-encoder.

Quantitative results for the main benchmark Table D.1 reports the complete numerical results corresponding to Figures 1 and 5.

Effect of goal respawning Point Goal and Car Goal respawn a new commanded goal after the previous goal is reached, so the policy must continue acting and can still encounter failure afterwards. We compare this default setting with a fixed-goal variant in Figure D.1. Safe-CRL retains an advantage in both settings, but the improvement in survival time is larger when goals respawn: +13.8% versus +5.0% for Point Goal and +18.3% versus +2.2% for Car Goal.

Table D.1: Final performance on the main benchmark, reported as mean ± standard deviation over five random seeds.
<table><tr><td></td><td colspan="2">Time at goal</td><td colspan="2">Survival time</td><td colspan="2">Goal coverage (%)</td></tr><tr><td>Environment</td><td> $\mathbf { S c a l i n g - C R L }$ </td><td> $_ \mathrm { S a f e - C R L }$ </td><td> $\mathbf { S c a l i n g - C R L }$ </td><td> $_ \mathrm { S a f e - C R L }$ </td><td> $\mathbf { S c a l i n g - C R L }$ </td><td> $_ \mathrm { S a f e - C R L }$ </td></tr><tr><td>Point Goal</td><td> $1 . 9 4 \pm 0 . 1 1$ </td><td> $\mathbf { 1 . 9 8 \pm 0 . 0 9 }$ </td><td> $5 5 5 . 1 \pm 1 2 . 3$ </td><td> ${ \bf 6 3 1 . 5 \pm 2 1 . 0 }$ </td><td> $8 1 . 6 \pm 3 . 4$ </td><td> ${ \bf 8 3 . 0 \pm 1 . 1 }$ </td></tr><tr><td>Car Goal</td><td> $2 . 1 7 \pm 0 . 0 4$ </td><td> ${ \bf 2 . 3 9 \pm 0 . 0 6 }$ </td><td> $4 6 4 . 9 \pm 6 . 5$ </td><td> $\mathbf { 5 5 0 . 1 \pm 1 5 . 9 }$ </td><td> $7 4 . 2 \pm 1 . 9$ </td><td> ${ \bf 7 9 . 7 \pm 0 . 7 }$ </td></tr><tr><td>Ant Goal</td><td> $2 7 6 . 8 3 \pm 1 2 . 8 1$ </td><td> $\mathbf { 6 8 4 . 0 0 \pm 1 4 . 5 7 }$ </td><td> $4 2 2 . 1 \pm 1 6 . 2$ </td><td> $\mathbf { 9 1 0 . 8 \pm 8 . 0 }$ </td><td> ${ \bf 8 7 . 2 \pm 0 . 9 }$ </td><td> $8 1 . 0 \pm 1 . 6$ </td></tr><tr><td>Humanoid Goal</td><td> $2 2 . 1 9 \pm 2 . 6 4$ </td><td> $\mathbf { 2 6 6 . 4 3 \pm 2 0 . 7 0 }$ </td><td> $1 4 5 . 7 \pm 1 2 . 9$ </td><td> $\mathbf { 3 8 6 . 3 \pm 1 9 . 9 }$ </td><td> $6 0 . 9 \pm 0 . 7$ </td><td> ${ \bf 7 8 . 5 \pm 0 . 9 }$ </td></tr><tr><td>Ant H-Maze</td><td> $3 9 0 . 6 0 \pm 5 2 . 8 0$ </td><td> $\mathbf { 4 1 2 . 9 6 \pm 1 7 . 9 2 }$ </td><td> $5 8 6 . 6 \pm 6 6 . 6$ </td><td> ${ \bf 7 9 5 . 6 \pm 1 9 . 6 }$ </td><td> ${ \bf 5 8 . 3 \pm 4 . 6 }$ </td><td> $5 4 . 0 \pm 2 . 5$ </td></tr><tr><td>Ant Cross Maze</td><td> $2 1 5 . 5 5 \pm 7 2 . 0 7$ </td><td> $\mathbf { 3 1 4 . 4 6 \pm 2 7 . 8 9 }$ </td><td> $4 2 5 . 6 \pm 9 2 . 7$ </td><td> ${ \bf 7 8 5 . 2 \pm 5 . 6 }$ </td><td> ${ \bf 5 6 . 1 \pm 1 . 2 }$ </td><td> $4 6 . 0 \pm 3 . 4$ </td></tr><tr><td>Ant Big Maze</td><td> $1 2 2 . 1 2 \pm 3 6 . 2 3$ </td><td> $\mathbf { 3 7 0 . 5 1 \pm 1 4 . 7 3 }$ </td><td> $3 0 0 . 2 \pm 2 9 . 0$ </td><td> ${ \bf 7 6 8 . 4 \pm 1 1 . 3 }$ </td><td> ${ \bf 6 2 . 4 \pm 2 . 5 }$ </td><td> $5 1 . 8 \pm 1 . 7$ </td></tr><tr><td>Humanoid U-Maze</td><td> $0 . 5 6 \pm 0 . 5 6$ </td><td> $\mathbf { 2 0 0 . 2 9 \pm 8 2 . 3 1 }$ </td><td> $1 3 0 . 4 \pm 4 2 . 4$ </td><td> $\mathbf { 3 7 3 . 9 \pm 1 0 2 . 0 }$ </td><td> $5 . 6 \pm 5 . 6$ </td><td> ${ \bf 3 1 . 9 \pm 1 0 . 3 }$ </td></tr><tr><td>Ant Big Pitfall</td><td> $1 1 6 . 8 4 \pm 5 . 2 3$ </td><td> $\mathbf { 4 5 2 . 0 3 \pm 4 2 . 6 1 }$ </td><td> $2 9 4 . 9 \pm 9 . 4$ </td><td> ${ \bf 8 2 1 . 1 \pm 1 6 . 1 }$ </td><td> ${ \bf 7 2 . 9 \pm 5 . 1 }$ </td><td> $6 2 . 7 \pm 4 . 6$ </td></tr><tr><td>Ant Hardest Pitfall</td><td> $1 0 8 . 5 4 \pm 1 1 . 0 2$ </td><td> $\mathbf { 3 4 8 . 0 7 \pm 5 6 . 7 5 }$ </td><td> $4 9 1 . 0 \pm 6 . 7$ </td><td> ${ \bf 8 4 6 . 9 \pm 1 5 . 7 }$ </td><td> $4 1 . 8 \pm 2 . 4$ </td><td> ${ \bf 5 1 . 3 \pm 8 . 8 }$ </td></tr><tr><td>Humanoid Loop Pitfall</td><td> $1 4 . 1 8 \pm 2 . 4 3$ </td><td> $\mathbf { 4 9 2 . 9 5 \pm 1 9 . 9 1 }$ </td><td> $1 2 0 . 5 \pm 2 2 . 3$ </td><td> ${ \bf 6 3 7 . 9 \pm 1 8 . 7 }$ </td><td> $5 3 . 7 \pm 1 9 . 3$ </td><td> ${ \bf 6 8 . 0 \pm 1 . 7 }$ </td></tr><tr><td>Humanoid Big Pitfall</td><td> $8 . 2 1 \pm 3 . 6 9$ </td><td> $\mathbf { 4 7 4 . 1 2 \pm 1 8 . 9 1 }$ </td><td> $1 1 1 . 8 \pm 2 3 . 0$ </td><td> ${ \bf 6 4 6 . 5 \pm 1 7 . 4 }$ </td><td> $3 0 . 8 \pm 1 4 . 1$ </td><td> ${ \bf 6 6 . 7 \pm 1 . 9 }$ </td></tr></table>

This result is consistent with the survival-mass interpretation: when control continues after reaching a goal, failure removes additional future occupancy and therefore has a larger effect on subsequent goal-conditioned behaviour.

![](images/19bfab600887f82102526e68e187c505e1df85150f7566b4666802dba1e07982.jpg)  
Figure D.1: Comparison of survival time with respawned and fixed goals in Point Goal and Car Goal. Percentages show the mean advantage of Safe-CRL over Scaling-CRL during the final 10% of training.

Z-encoder depth We compare 4-layer and 64-layer Z-encoders on four representative tasks while keeping the actor and critic depths fixed. Increasing the Z-encoder depth provides no consistent improvement in either time at goal or survival time, suggesting that a lightweight estimator is sufficient in these experiments.

![](images/8859a66d862896386bd220091bb7230f30993b89bee9479e930980636034649f.jpg)  
Figure D.2: Ablation on Z-encoder depth. Actor and critic depths are fixed at 16 layers.

TD-based survival-mass estimation Safe-CRL trains the Z-encoder from $Y _ { t } = \mathbf { 1 } \{ T _ { f } > H _ { t } \}$ with $H _ { t } \sim q _ { \gamma }$ . We additionally examine whether $Z ( s , a )$ can instead be learned through temporal-difference (TD) learning. We replace the Z-encoder objective with a clipped double-Q formulation [27] that estimates a goal-independent survival-mass function.

As shown in Figure D.3, TD-based estimation produces mixed results. It substantially degrades goal-reaching and survival performance in Car Goal and Humanoid Goal, while remaining competitive on the selected Ant tasks. We therefore use direct Monte Carlo survival-mass estimation as the default because it is simpler and provides more consistent performance across the evaluated environments.

Computational overhead Table D.2 reports the wall-clock training time. Humanoid locomotion tasks are tested on modified NVIDIA RTX 4090 (48 GB) and the others are tested on modified NVIDIA RTX 4080 (32 GB). For each environment, Scaling-CRL and Safe-CRL are measured on the same hardware configuration. With 64-layer actor and critic networks, adding the 4-layer Z-encoder increases the average training time from 9.07 to 9.32 hours per 100M environment steps, corresponding to an average overhead of only 2.8%.

![](images/a1f53662ebaed1bff713afbac769f25ddf4b01b0830d0b510009e01f78593368.jpg)  
Figure D.3: Ablations on TD-learned Z. Actor and critic depths are set to 16 layers (8 for Car Goal).

Table D.2: Average wall-clock training time per 100M environment steps.
<table><tr><td>Environment</td><td>Scaling-CRL</td><td>Safe-CRL (ours)</td></tr><tr><td>Point Goal</td><td>2.92 h</td><td>3.07 h</td></tr><tr><td>Car Goal</td><td>8.73 h</td><td>8.94 h</td></tr><tr><td>Ant Goal</td><td>6.65 h</td><td>6.71 h</td></tr><tr><td>Humanoid Goal</td><td>7.28 h</td><td>7.74 h</td></tr><tr><td>Ant H-Maze</td><td>10.20 h</td><td>10.58 h</td></tr><tr><td>Ant Cross Maze</td><td>10.48 h</td><td>10.64 h</td></tr><tr><td>Ant Big Maze</td><td>10.49 h</td><td>10.70 h</td></tr><tr><td>Humanoid U-Maze</td><td>11.09 h</td><td>11.13 h</td></tr><tr><td>Ant Big Pitfall</td><td>9.85 h</td><td>10.01 h</td></tr><tr><td>Ant Hardest Pitfall</td><td>10.42 h</td><td>10.66 h</td></tr><tr><td>Humanoid Loop Pitfall</td><td>10.55 h</td><td>11.01 h</td></tr><tr><td>Humanoid Big Pitfall</td><td>10.21 h</td><td>10.70 h</td></tr><tr><td>Mean</td><td>9.07 h</td><td>9.32 h (↑ 2.8%)</td></tr></table>

Exploration remains a limitation Safe-CRL corrects the occupancy information used for critic and policy learning, but does not modify the exploration mechanism of CRL. Figure D.4 visualizes goal-reaching success rates across differ ent goal locations. The learned success distributions can remain spatially uneven, including asymmetric performance at geometrically similar goal locations. For example, for Hum. Loop Pitfall, the top-left corner and the bottom-left corner represent the left-back and right-back side of the humanoid robot, respectively. The right-back side (with lighter colors) is better explored and learned than the left-back side. This pattern indicates non-uniform exploration and shows that restoring survival mass does not by itself guarantee uniform goal-space coverage. If the robot’s initial position is near a corner, then exploration is restricted. The success rate therefore depends on the distance between the robot and the goal.

![](images/7d4e3a595ab8e7bec3a8737f847998edb914945b9bde05bd3597ee176f7338f0.jpg)

![](images/46918545d21bc7a4a7a0c1f3c09b842b36fc7098e6857924df191d71112f7c1d.jpg)

![](images/403a57dbadc8e65e07e115e0c935779a28deb953936a2ddcd8025bea9a42bed5.jpg)

![](images/573269853c3b29ff29cfe28300ab172812bc3867fb3e7cb8304928105ccabd22.jpg)

![](images/236a75ebded41af6f3ba6577d0264db1f0a12cdd9c0e475d50e133f8445ddf55.jpg)  
Figure D.4: Success rate for goals at different places in the playground. Darker colors indicate lower goal-reaching success rates. The robot initially faces the +x axis direction (→). The diamond marks the initial position of the robot.