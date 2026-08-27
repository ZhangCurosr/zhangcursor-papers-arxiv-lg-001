# AERIS: Offline Policy Improvement for Multi-UAV Integrated Sensing and Communication

Ziyuan Wang<sup>1,#</sup>, Yifan Sui<sup>2,#</sup>, Wei Wei<sup>3</sup>, Wenjie Xin<sup>4</sup>,

Zekai Zhang<sup>1</sup>, Xiangwang Hou<sup>1</sup>, Xiao-Ping (Steven) Zhang<sup>1,5</sup>

<sup>1</sup>Tsinghua University, Beijing, China <sup>2</sup>Shanghai Jiao Tong University, China

<sup>3</sup>The University of Hong Kong, Hong Kong <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou), China

<sup>5</sup>Toronto Metropolitan University, Canada

<sup>#</sup>Equal contribution.

Abstract—Unmanned aerial vehicle (UAV)-enabled integrated sensing and communication (ISAC) is a promising 6G paradigm, but dynamic multi-UAV ISAC control must jointly balance communication quality, sensing reliability, and flight safety under stochastic mobility. Existing optimization methods often require repeated global non-convex solving, while online reinforcement learning (RL) depends on risky trial-and-error flights that may cause sensing loss or collision-risk events.

This paper proposes AERIS, an offline policy improvement framework for multi-UAV ISAC. AERIS learns from fixed flight logs under centralized training and decentralized execution, so each UAV acts from local histories while training uses logged global information to assess team-level effects. We further design STAR-CRDT, an offline multi-agent RL algorithm that performs support-aware local action rectification and distills only trusted improvements into the decentralized actor. We prove an offlinesupport policy improvement guarantee. Experiments show that STAR-CRDT improves the main ISAC objective return by 29.3% over the strongest baseline. It further improves communication sum rate, sensing pass rate, and sensing margin by 3.4%, 4.8%, and 69.1%, while reducing collision-risk events by 54.2%. On unseen real-road maps built from OpenStreetMap data, STAR-CRDT still obtains the best return.

Index Terms—ISAC, UAV networks, offline MARL, stochastic optimization

## I. INTRODUCTION

Integrated sensing and communication (ISAC) has become a key direction for sixth-generation (6G) networks because the same wireless infrastructure can deliver data services and sense the physical environment [1]–[3]. This joint use of spectrum, antennas, and waveforms is especially valuable for applications that need both connectivity and environmental awareness, such as mobile target monitoring, emergency response, and temporary hotspot service. Unmanned aerial vehicles (UAVs) further enlarge this design space. By adjusting their positions and altitudes, UAVs can rapidly create line-of-sight links, approach sensing targets, and serve areas where fixed infrastructure is unavailable or overloaded [4]–[6]. Recent UAVenabled ISAC studies therefore jointly optimize trajectory, beamforming, user association, and sensing quality [7]–[10].

The same flexibility also makes multi-UAV ISAC control difficult. A local movement of one UAV changes air-to-ground channels, beam directions, inter-user interference, sensing margins, and the distance to other UAVs. Ground users and sensing targets move stochastically, which turns the problem into a long-horizon control task. Classical trajectory and resource optimization methods usually rely on instantaneous global geometry, channel, and association information. Under stochastic mobility, they may also require repeated non-convex optimization or receding-horizon updates, which creates heavy computation and coordination overhead for distributed and fast UAV response [11]–[13]. These properties make reinforcement learning (RL) and multi-agent RL (MARL) a natural tool for dynamic wireless control, because they can learn long-term policies from data without solving a full non-convex program at every slot [14]–[19]. Offline RL further extends this datadriven formulation to fixed logs, so experience collected from human-operated flights, legacy controllers, or conservative behavior policies can be reused for policy improvement [20], [21]. Representative offline methods have been developed for behavior-constrained value learning, conservative actor updates, and sequence-model-based control [20]–[29].

Although RL-based control for UAV networking and multi-UAV ISAC is gaining momentum, we observe that current RL solutions still have fundamental limitations that can make multi-UAV sensing and communication inefficient or unreliable. In the online setting, policy improvement still depends on trial rollouts. For multi-UAV ISAC, such rollouts are not just sampling costs. An immature or exploratory policy can lower sensing beam gain, degrade user service, or violate the inter-UAV safety distance before it becomes useful. Offline RL is attractive because it keeps the long-horizon policylearning ability of RL while moving improvement to fixed data. Yet existing offline designs do not directly resolve the information and support structure of multi-UAV ISAC. A centralized learner can evaluate global state and joint actions, but a deployed UAV can only observe local history. Purely local or imitation-based learners may stay close to behavior logs, but they cannot reliably judge how a local action changes global interference, sensing margin, service quality, and pairwise safety, so residual ISAC violations may persist. More aggressive actor-critic rectification may leave the logged jointaction support and suffer from critic overestimation, which can further degrade sensing and safety [30], [31].

These limitations call for a new offline MARL design for multi-UAV ISAC. Such a design should improve beyond imperfect logs, keep each local correction support-aware, and allow distributed UAVs to improve task policies offline before deployment. To address this need, we propose Aerial

Experience-guided Reinforcement for ISAC Systems (AERIS), a novel offline distributed policy improvement framework for multi-UAV ISAC. AERIS first turns logged flight experience into a centralized training and decentralized execution (CTDE) offline MARL problem, which avoids additional risky exploration while preserving local UAV execution. It then uses centralized training information to evaluate whether a local action change improves the team-level communication, sensing, and safety objective. To improve beyond behavior imitation without leaving reliable support, we design Supportaware Trust-gated Actor Rectification for Critic-Regularized Decision Transformer (STAR-CRDT). STAR-CRDT keeps a shared local-history actor for execution, searches for candidate corrections near logged and actor actions, scores them with a centralized critic, and distills a correction only when its critic gain and behavior proximity are reliable. Together, these choices turn centralized critic feedback into a support-aware teacher for the local actor, allowing AERIS to improve beyond behavior logs without degenerating into pure imitation or unsupported critic maximization.

Our contributions are summarized as follows.

• We propose AERIS, a novel offline policy improvement framework for stochastic long-horizon multi-UAV ISAC control. AERIS enables distributed UAVs to refine communication, sensing, and safety decisions from existing flight experience, reducing the need for risky online trial flights before deployment.

• We design STAR-CRDT, a new offline MARL algorithm for support-aware local policy correction. By combining centralized critic scoring with trust-gated distillation, STAR-CRDT improves the decentralized actor while controlling support shift in fixed-log training.

• We provide a proof of offline-support policy improvement and evaluate AERIS against mainstream offline RL/MARL baselines. The results show strong offline policy improvement, consistent system-scale gains, and the best return on unseen real-road maps built from OpenStreetMap (OSM) data.

## II. BACKGROUND AND MOTIVATION

This section presents two diagnostics that motivate offline policy improvement for multi-UAV ISAC. We first show why direct online policy improvement is risky, then examine whether existing fixed-log methods can remove this risk, and finally summarize the design requirements that guide AERIS.

## A. Online Trial Risk

Multi-UAV ISAC requires a controller to trade off communication rate, sensing feasibility, sensing robustness, and flight safety. Existing studies model these objectives through coupled trajectory and beamforming variables [7]–[10]. Online RL can adapt to the resulting stochastic dynamics, but it must collect trial rollouts. Safe and constrained RL studies have long recognized that unsafe exploration should be measured as a constraint cost rather than hidden inside average reward [32]–[34].

![](images/6ec93c28e25f2e4ac7705c9b8a611ad9167705078a0955c0031ffaf5ca467023.jpg)  
(a) Risk during online improvement.

![](images/a2fd1257cd8b2c11345e3fa21a72fb690582278b28ca7aae276aedb32f937923.jpg)  
(b) Support shift of offline baselines.  
Fig. 1. Motivation diagnostics. Online trial actions can create sensing and safety risk. Generic offline baselines face an imitation and out-of-distribution (OOD) dilemma under fixed joint-action support.

Fig. 1(a) reports an online-learning diagnostic under an evaluation model aligned with recent multi-UAV ISAC trajectory and beamforming studies [7]–[10]. Intermediate Twin Delayed Deep Deterministic Policy Gradient (TD3) checkpoints trigger collision-risk events before the controller becomes useful [35]. After a behavior controller is obtained, additional Gaussian action noise keeps the communication sum rate roughly stable but increases sensing-threshold violations. Thus, a rollout can look acceptable from the communication metric while still damaging sensing reliability and safety. This makes continued online improvement undesirable for physical multi-UAV ISAC deployment.

## B. Fixed-Log Baseline Diagnostic

Offline RL is a natural candidate for avoiding new exploratory flights because it improves policies from a fixed dataset. Representative offline RL solutions have been widely adopted as fixed-log baselines in control and networking stud ies [20]–[22], [27]–[29]. However, multi-UAV ISAC requires an offline update to improve a locally executed action while its effect is judged by the coupled team-level communication, sensing, and safety objective. This creates an information mismatch between centralized evaluation and decentralized execution. Conservative or imitation-based methods may inherit residual ISAC violations from the behavior policy, whereas aggressive actor rectification may leave the logged joint-action support and amplify critic error [30], [31].

Fig. 1(b) illustrates this tradeoff with representative fixedlog baselines. The bars report ISAC violation cost, which combines sensing-threshold violations and collision-risk events under the same penalties used by the reward. The line reports the joint-action OOD ratio measured by a k-nearest-neighbor support test against logged joint actions. Conservative methods remain closer to the log but still leave non-negligible ISAC violations, while aggressive rectification moves farther outside support and incurs larger violation cost. Therefore, directly porting state-of-the-art offline RL/MARL methods does not simultaneously provide policy improvement and reliable support control for multi-UAV ISAC.

## C. Motivation and Design Requirements

The two diagnostics lead to three requirements for offline multi-UAV ISAC control.

• Avoid repeated exploratory flights. Policy improvement should avoid repeated exploratory flights after an initial log is available. If no public flight log exists, limited online learning can be used only to construct a behavior log, after which the dataset should be frozen and all policy refinement should happen offline.

• Preserve distributed UAV execution. The trained controller must remain executable by distributed UAVs. Each UAV should act from its local history, while training may use logged global states and joint actions to evaluate team-level communication, sensing, and safety effects.

• Improve within reliable offline support. The offline update should improve beyond imperfect behavior logs without trusting unsupported actions. This requires local corrections that are selected by a centralized critic but filtered by their proximity to the offline data support.

Taken together, these requirements motivate AERIS and highlight the main difficulty: offline policy improvement should use centralized training signals to identify team-level gains, yet convert them into a decentralized UAV controller without degenerating into pure imitation or unsupported critic maximization.

## III. SYSTEM AND PROBLEM FORMULATION

We adopt a standard UAV-enabled ISAC trajectory-andbeamforming model and formulate a generic communicationsensing optimization problem [7], [8].

## A. System Model

The system model describes the physical control environment in which moving UAVs serve communication users and sensing targets. Each movement or beam choice affects communication quality, sensing reliability, and safety separation over time.

We consider M UAVs serving K communication users and L sensing targets in a $L _ { x } \times L _ { y }$ area. UAV m has position $\mathbf { q } _ { m } ( t ) = [ x _ { m } ( t ) , y _ { m } ( t ) , z _ { m } ( t ) ] ^ { \breve { T } }$ . During log collection and random-mobility evaluation, users and targets follow bounded Brownian-style random walks over the full region [36]. This map-agnostic prior represents high-randomness mobility rather than a fixed road topology. During road-map deployment, entities still move stochastically, but every horizontal update is restricted to OSM road segments and Geographic Information System (GIS) layers. Dense grids induce frequent route branching, arterial corridors concentrate demand along a few directions, and sparse local roads create uneven service opportunities [37], [38].

Each UAV uses an $N _ { t }$ -element transmit array for downlink communication and sensing. The air-to-ground channel $\mathbf { h } _ { m , k } ( t )$ captures path loss, low-altitude attenuation, and array steering, as widely used for UAV links [11], [39]. With beamformer $\mathbf { w } _ { m , k } ( t )$ steered from UAV m to user $k ,$ the received signal-to-interference-plus-noise ratio (SINR) is

![](images/f48b201342d3a86d2f2d171bdb95c4d22d6ce763b8de86176ad383219aa969f2.jpg)  
Fig. 2. Dynamic multi-UAV ISAC control under local execution.

$$
\gamma _ { m , k } ( t ) = \frac { \vert \mathbf { h } _ { m , k } ^ { H } ( t ) \mathbf { w } _ { m , k } ( t ) \vert ^ { 2 } } { \sigma ^ { 2 } + \sum _ { ( i , j ) \neq ( m , k ) } \vert \mathbf { h } _ { i , k } ^ { H } ( t ) \mathbf { w } _ { i , j } ( t ) \vert ^ { 2 } } ,\tag{1}
$$

where $\sigma ^ { 2 }$ is the noise power. The communication rate is $\begin{array} { r c l } { R _ { m , k } ( t ) } & { = } & { B \log _ { 2 } ( 1 + \gamma _ { m , k } ( t ) ) } \end{array}$ , and $\begin{array} { r l } { U _ { \mathrm { c o m m } } ( t ) } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { m } \sum _ { k \in \mathcal { C } _ { m } ( t ) } R _ { m , k } ( t ) } \end{array}$ is the system sum-rate. This standard metric directly improves spectral efficiency in UAVenabled ISAC trajectory and beamforming designs [7], [8]. For sensing, let $\begin{array} { r c l } { \mathbf { R } _ { m } ( t ) } & { = } & { \sum _ { k \in \mathcal { C } _ { m } ( t ) } \mathbf { w } _ { m , k } ( t ) \mathbf { w } _ { m , k } ^ { H } ( t ) } \end{array}$ be the transmit covariance. The beampattern gain $G _ { m , \ell } ( t ) ~ =$ ${ \mathbf { a } } _ { m , \ell } ^ { H } ( t ) { \mathbf { R } } _ { m } ( t ) { \mathbf { a } } _ { m , \ell } ( t )$ measures energy focused toward target ℓ, and $\Delta _ { m , \ell } ^ { \mathrm { s e n } } ( t ) = G _ { m , \ell } ( t ) - \Gamma _ { m , \ell } ( t )$ is its threshold margin. The sensing utility is

$$
\begin{array} { l }  \displaystyle { U _ { \mathrm { s e n } } ( t ) = \alpha _ { p } \sum _ { m , \ell } \mathbb { I } \{ \Delta _ { m , \ell } ^ { \mathrm { s e n } } ( t ) \geq 0 \} } \\ { \displaystyle { + \alpha _ { m } \sum _ { m , \ell } \mathrm { t a n h } \bigg ( \frac { \Delta _ { m , \ell } ^ { \mathrm { s e n } } ( t ) } { \operatorname* { m a x } ( \Gamma _ { m , \ell } ( t ) , \varepsilon ) } \bigg ) , } } \end{array}\tag{2}
$$

where the two terms reward target-threshold satisfaction and normalized beampattern margin. Optimizing this form keeps targets detectable and preserves robustness against mobilityinduced geometry changes [7]–[9].

## B. Optimization Problem

The optimization asks each UAV to decide where to fly and how to form beams over time, so that users receive highrate service, sensing targets remain detectable, and UAVs stay safely separated.

Under distributed execution, the local information available to UAV m includes its own three-dimensional (3-D) position, relative geometry to associated communication users and sensing targets, and local channel and sensing steering estimates, without instantaneous global state exchange. The decision variables are the UAV trajectories q and beamformers w over horizon T. The objective is to maximize a weighted communication and sensing utility that evaluates the locally executed decisions by communication rate, sensing feasibility, and sensing robustness [1], [7], [8],

$$
\begin{array} { r l } { \mathbf { P 0 : } } & { ~ \displaystyle \operatorname* { m a x } _ { \{ \mathbf { q } , \mathbf { w } \} } ~ \sum _ { t = 0 } ^ { T - 1 } \bigl [ \lambda _ { c } U _ { \mathrm { c o m m } } ( t ) + \lambda _ { s } U _ { \mathrm { s e n } } ( t ) \bigr ] } \\ & { ~ \mathrm { s . t . } ~ G _ { m , \ell } ( t ) \geq \Gamma _ { m , \ell } ( t ) , ~ \forall m , \ell , t , } \\ & { ~ \displaystyle \sum _ { k \in \mathcal { C } _ { m } ( t ) } \| \mathbf { w } _ { m , k } ( t ) \| _ { 2 } ^ { 2 } \leq P _ { \operatorname* { m a x } } , ~ \forall m , t , } \\ & { ~ \| \mathbf { q } _ { m } ( t ) - \mathbf { q } _ { n } ( t ) \| _ { 2 } \geq d _ { \operatorname* { m i n } } , ~ \forall m \neq n , t , } \end{array}\tag{3}
$$

where the constraints enforce sensing-threshold satisfaction, per-UAV transmit power, and inter-UAV safety separation. Sensing-threshold violations can make returned data too weak to maintain target awareness [7], [8]. A collision-risk event occurs when pairwise distance falls below $d _ { \mathrm { m i n } }$ , and standard UAV kinematic limits are enforced during trajectory generation [11], [12].

## C. Key Challenges

Solving P0 presents three key challenges.

• Challenge 1: Stochastic non-convex ISAC control. The SINR and beampattern terms jointly depend on UAV positions, beamformers, interference, and geometrydependent associations. User and target mobility further makes P0 a long-horizon stochastic control problem, so repeatedly solving a centralized trajectory-andbeamforming program is impractical for fast distributed UAV response.

• Challenge 2: Online policy-improvement risk. A learning controller for P0 would normally require trial rollouts to estimate long-term rewards and constraint effects. In physical multi-UAV ISAC, such exploratory flights can consume flight resources, reduce communication service, weaken sensing returns, or create collision-risk events before the policy converges.

• Challenge 3: Fixed-log distributed improvement. Freezing a flight log removes new exploration, but it also makes policy improvement support-sensitive. Training can use logged global states and joint actions to evaluate team-level communication, sensing, and safety effects, whereas each deployed UAV must act from local histories. Thus, an offline learner must convert centralized training evidence into decentralized policy updates without imitating residual violations or trusting unsupported joint actions.

## IV. AERIS DESIGN

AERIS addresses these challenges through a fixed-log CTDE design. It reformulates the stochastic non-convex control problem as sequential local action selection, preserving distributed execution while allowing training to evaluate teamlevel ISAC rewards. It then freezes flight experience collected by a conservative behavior controller, so policy refinement does not require additional exploratory flights. Finally, it uses STAR-CRDT to turn centralized critic feedback into supportaware updates of the local-history actor. Thus, AERIS defines the policy-improvement pipeline, while STAR-CRDT is the offline MARL optimizer inside it.

## A. Problem Transformation and Offline Dataset

AERIS transforms P0 from direct trajectory-andbeamforming optimization into distributed agent action selection. At slot t, the local observation $o _ { t } ^ { m }$ of UAV m corresponds to the local information described in Section III-B. A short local history $h _ { t } ^ { m }$ stacks recent observations, previous local actions, and return-conditioning signals. The decentralized actor outputs $\begin{array} { r c l } { a _ { t } ^ { m } } & { = } & { \pi _ { \theta } ( h _ { t } ^ { m } ) } \end{array}$ , where $a _ { t } ^ { m }$ controls UAV displacement and beam residuals, and $\begin{array} { c c l } { \mathbf { a } _ { t } } & { = } & { ( a _ { t } ^ { 1 } , \dots , a _ { t } ^ { M } ) } \end{array}$ is the joint action. This conversion preserves distributed execution because each UAV acts only from local information, while the training process may still use global logged states to evaluate the joint effect of the selected actions.

Under this RL reformulation, AERIS optimizes

$$
\mathbf { P 1 } \colon \ \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \Big [ \sum _ { t = 0 } ^ { T - 1 } r _ { t } \Big ] ,\tag{4}
$$

with per-slot reward

$$
\begin{array} { r } { r _ { t } = \lambda _ { r } U _ { \mathrm { c o m m } } ( t ) + \lambda _ { p } P _ { \mathrm { p a s s } } ( t ) + \lambda _ { m } M _ { \mathrm { s e n } } ( t ) } \\ { - \zeta N _ { t } ^ { \mathrm { v i o } } - c _ { \mathrm { c o l } } N _ { t } ^ { \mathrm { c o l } } , \qquad } \end{array}\tag{5}
$$

where the positive terms are slot-level slices of P0. $U _ { \mathrm { c o m m } } ( t )$ rewards communication spectral efficiency, $P _ { \mathrm { p a s s } } ( t )$ counts sensing-threshold passes, and $M _ { \mathrm { s e n } } ( t )$ measures normalized sensing margin. The negative terms relax the constraints in P0, where $N _ { t } ^ { \mathrm { v i o } }$ counts sensing-threshold violations and $N _ { t } ^ { \mathrm { c o l } }$ counts collision-risk events. Such reward-and-penalty reformulation is standard in RL-based networking and constrained RL [14]–[16], [33]. It keeps the communication, sensing, and safety meanings of P0, while making the problem suitable for fixed-log policy learning.

There is no public flight-log dataset for multi-UAV ISAC with joint communication, sensing, and safety labels. We therefore collect a fixed behavior dataset before offline training, following common offline RL practice [20], [21]. The behavior policy uses TD3 under the same action and reward semantics as P1. TD3 is suitable here because it is a strong continuous-control actor-critic method, and TD3 has been widely used as a continuous-control backbone for communication and networking optimization problems [16], [35]. With clipped target noise $\epsilon ,$ TD3 uses

$$
y _ { t } ^ { \mathrm { T D 3 } } = r _ { t } + \gamma \operatorname* { m i n } _ { j = 1 , 2 } Q _ { \bar { \phi } _ { j } } \left( x _ { t + 1 } , \pi _ { \bar { \theta } } ( x _ { t + 1 } ) + \epsilon \right) ,\tag{6}
$$

to fit twin critics and update the actor by delayed deterministic policy-gradient steps [35]. During collection, each UAV uses only local geometry, service information, neighbor separation, and recent rewards, so the resulting log respects the distributed information pattern.

After the preliminary controller is obtained, AERIS treats its rollouts as surrogate flight logs. They play the role of human-pilot or legacy-controller records, which contain useful behavior but may retain sensing and safety violations. The dataset is then frozen. Each trajectory $\tau \in \mathcal { D }$ stores $\tau =$ $\{ \left( s _ { t } , \mathbf { a } _ { t } , r _ { t } , s _ { t + 1 } , \{ h _ { t } ^ { m } \} _ { m = 1 } ^ { M } \right) \} _ { t = 0 } ^ { T - 1 }$ , where $s _ { t }$ and $\mathbf { a } _ { t }$ support centralized critic training and $h _ { t } ^ { m }$ supports decentralized actor learning. TD3 only constructs the log, whereas all reported AERIS policy improvement comes from offline learning.

![](images/e29c92b1d880329db4a6b20d979e954d9b9ee7c62042c99e69881db470d1443e.jpg)

Fig. 3. Workflow of STAR-CRDT. The actor executes locally, while the centralized critic selects and filters local corrections during offline training.  
Algorithm 1 STAR-CRDT Offline Training   
1: Input: offline dataset $\mathcal { D } ,$ shared actor $\pi _ { \theta } ,$ , twin critics   
$Q _ { \phi _ { 1 } } , Q _ { \phi _ { 2 } }$ , value network $V _ { \psi }$   
2: Output: decentralized STAR-CRDT actor $\pi _ { \theta }$ and the best   
validation checkpoint   
3: for each training epoch e do   
4: sample mini-batches of fixed-length local-history win  
dows from D   
5: run the shared actor on each local history to obtain   
${ \widehat { a } } _ { i } = \pi _ { \theta } ( h _ { i } )$   
6: update $Q _ { \phi _ { 1 } } , Q _ { \phi _ { 2 } }$ by minimizing $( 7 )$   
7: update $V _ { \psi }$ using the expectile value loss   
8: compute the backbone actor loss $\mathcal { L } _ { \mathrm { b a s e } } ^ { ( e ) }$   
9: for each selected token and UAV i do   
10: construct $\mathcal { U } _ { i } ( s )$ by (9)   
11: evaluate behavior and actor contexts by the central  
ized critic   
12: select the rectified action $a _ { i } ^ { \star }$ by (10)   
13: choose $a _ { i } ^ { \mathrm { a n c } }$ and construct $\widetilde { a } _ { i }$ by (11)   
14: end for   
15: build teacher targets and compute $\mathcal { L } _ { T } ^ { ( e ) }$   
16: compute the centralized Q-regularizer $\mathcal { R } _ { Q } ^ { ( e ) }$   
17: update the actor with (12)   
18: update target networks and record the best checkpoint   
19: end for

## B. STAR-CRDT Offline Learning

STAR-CRDT follows one principle. A UAV should change a logged action only when the centralized critic predicts a global ISAC improvement and the change remains close to reliable offline support. Fig. 3 illustrates this procedure. The actor first learns a local sequence policy from the log, the centralized critic evaluates local corrections in joint-action contexts, and the trust gate decides how much correction can be distilled back into the decentralized actor.

Local sequence actor and centralized critic. Each UAV uses the same return-conditioned causal Transformer actor $\pi _ { \theta } ( h _ { t } ^ { m } )$ , inspired by Decision Transformer [29]. Parameter sharing lets all UAV logs train a common local rule across different local geometries. The actor is deliberately local, since a globally conditioned actor would require real-time global information exchange. However, local execution alone cannot judge whether a local correction improves the team objective. STAR-CRDT therefore trains centralized twin critics $Q _ { \phi _ { 1 } } ( s _ { t } , \mathbf { a } _ { t } )$ and $Q _ { \phi _ { 2 } } ( s _ { t } , \mathbf { a } _ { t } )$ , together with a value network $V _ { \psi } ( s _ { t } )$ , using logged global states and joint actions. This follows the CTDE principle [17]–[19]. The critic evaluates global communication, sensing, and safety effects during training, while deployment uses only the local actor in Fig. 3.

Centralized implicit-Q evaluation. The next issue is conservative value estimation under fixed data. Directly maximizing a learned offline critic can select unsupported actions with overestimated values [22], [25]. STAR-CRDT instead learns a ranking signal on logged joint actions. For each transition, STAR-CRDT minimizes

$$
\mathcal { L } _ { Q } = \frac { 1 } { 2 } \mathbb { E } _ { \mathcal { D } } \sum _ { j = 1 } ^ { 2 } \bigl ( Q _ { \phi _ { j } } \bigl ( s _ { t } , \mathbf { a } _ { t } \bigr ) - y _ { t } \bigr ) ^ { 2 } ,\tag{7}
$$

where $y _ { t } = r _ { t } + \gamma V _ { \psi } ( s _ { t + 1 } )$ . With $\bar { Q } _ { t } \ =$ min<sub>j</sub> $Q _ { \phi _ { j } } ( s _ { t } , \mathbf { a } _ { t } )$ the value loss is $\mathcal { L } _ { V } = \mathbb { E } _ { \mathcal { D } } [ \rho _ { \kappa } ( \bar { Q } _ { t } - V _ { \psi } ( s _ { t } ) ) ]$ , where $\rho _ { \kappa } ( u ) =$ $| \kappa - \mathbb { I } \{ u < 0 \} | u ^ { 2 }$ follows implicit Q-learning [26]. The twincritic minimum suppresses optimistic error, and the expectile value provides an in-support baseline.

Behavior-preserving backbone. Before critic-guided correction, the actor first learns a feasible local behavior. For a logged action $\mathbf { a } ^ { \beta }$ , STAR-CRDT computes

$$
A _ { \mathrm { I Q L } } ( s , { \bf a } ^ { \beta } ) = \operatorname* { m i n } _ { j } Q _ { \phi _ { j } } ( s , { \bf a } ^ { \beta } ) - V _ { \psi } ( s ) ,\tag{8}
$$

and $w _ { \mathrm { I Q L } } = \mathrm { m i n } \{ \mathrm { e x p } ( \beta _ { \mathrm { I Q L } } [ A _ { \mathrm { I Q L } } ] _ { + } ) , w _ { \mathrm { m a x } } ^ { \mathrm { I Q L } } \}$ . The backbone loss mixes plain behavior cloning and advantage-weighted behavior cloning. This anchors the sequence actor near logged support before any correction is introduced.

Support-aware local rectification. A behavior-preserving actor can still inherit residual ISAC violations. STAR-CRDT therefore searches for local corrections only near the actor and behavior actions as follows.

$$
\begin{array} { c } { { \mathcal { U } _ { i } ( s ) = \{ \hat { a } _ { i } , a _ { i } ^ { \beta } \} \cup \{ \mathrm { c l i p } ( \hat { a } _ { i } + \sigma _ { \pi } \xi _ { k } ) \} _ { k = 1 } ^ { N _ { r } } } } \\ { { \cup \{ \mathrm { c l i p } ( a _ { i } ^ { \beta } + \sigma _ { \beta } \zeta _ { k } ) \} _ { k = 1 } ^ { N _ { r } } , } } \end{array}\tag{9}
$$

where $\xi _ { k }$ and $\zeta _ { k }$ are local perturbations around the actor and behavior actions. Each candidate u is tested in two centralized contexts. Let $\mathbf { a } _ { u } ^ { \beta }$ replace only the i-th behavior action and $\mathbf { a } _ { u } ^ { \pi }$ replace only the i-th actor action. With $Q _ { \phi } = \operatorname* { m i n } _ { j } Q _ { \phi _ { j } }$

$$
a _ { i } ^ { \star } \in \arg \operatorname* { m a x } _ { u \in \mathcal { U } _ { i } ( s ) } \operatorname* { m a x } _ { \chi \in \{ \beta , \pi \} } Q _ { \phi } ( s , \mathbf { a } _ { u } ^ { \chi } ) .\tag{10}
$$

Thus, a local correction is accepted only after the centralized critic evaluates its effect on the joint ISAC return.

Trust-gated distillation. A critic-improving candidate can still be unreliable if its gain is small or its distance from support is large. STAR-CRDT therefore distills a conservative teacher. Let $a _ { i } ^ { \mathrm { a n c } }$ be the anchor action and $\Delta Q _ { i } = Q _ { \phi } ( s , \mathbf { a } ^ { \star } ) -$ $Q _ { i } ^ { \mathrm { { f l o o r } } } ( s )$ . The trust coefficient is

$$
\tau _ { i } = \tau _ { \mathrm { m a x } } \big ( 1 - e ^ { - \beta _ { \tau } [ \Delta Q _ { i } - m _ { \tau } ] _ { + } } \big ) e ^ { - \| a _ { i } ^ { \star } - a _ { i } ^ { \mathrm { a n c } } \| _ { 2 } ^ { 2 } / \sigma _ { \tau } } .\tag{11}
$$

The teacher is $\widetilde { \boldsymbol { a } } _ { i } = ( 1 - \tau _ { i } ) a _ { i } ^ { \mathrm { a n c } } + \tau _ { i } a _ { i } ^ { \star }$ . The actor is updated by

$$
\mathcal { L } _ { \mathrm { a c t o r } } ^ { ( e ) } = \mathcal { L } _ { \mathrm { b a s e } } ^ { ( e ) } + \lambda _ { T } ^ { ( e ) } \mathcal { L } _ { T } - \lambda _ { Q } ^ { ( e ) } \mathcal { R } _ { Q } ,\tag{12}
$$

where $\mathcal { L } _ { T }$ distills the teacher and $\mathcal { R } _ { Q }$ rewards predicted joint actions only when they improve over logged behavior. Algorithm 1 summarizes the workflow. During training, STAR-CRDT repeatedly samples fixed-log local histories, updates centralized critics and value estimates, constructs supportaware local teachers, and distills them into the shared actor. After training, only $\pi _ { \theta } ( h _ { t } ^ { m } )$ is deployed on UAVs. All critic, value, candidate-search, and trust-gate modules are removed from runtime control, so each UAV can make autonomous real-time decisions from its own local history.

## C. Policy Improvement Proof

We prove the trust-gated local correction on the offlinesupport distribution, rather than claiming global optimality outside the dataset. Let $d _ { \beta }$ be the state distribution induced by the behavior log and let $\mathbf { a } _ { - i } ^ { \mathrm { c t x } }$ be the fixed context actions when only UAV i is varied. The local rectification gain is

$$
\begin{array} { r } { \Delta _ { i } ( s ) = Q _ { \phi } ( s , ( a _ { i } ^ { \star } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) - Q _ { \phi } ( s , ( a _ { i } ^ { \mathrm { a n c } } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) . } \end{array}\tag{13}
$$

Following standard offline actor-critic and smooth optimization analyses [22], [25], [26], [40], we use four local assumptions.

Assumption 1. The critic correctly ranks candidates in $\mathcal { U } _ { i } ( s )$ , and the anchor satisfies $a _ { i } ^ { \mathrm { a n c } } \in \mathcal { U } _ { i } ( s )$

Assumption 2. The local critic slice

$$
g _ { i } ( u ; s ) = Q _ { \phi } ( s , ( u , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) ,\tag{14}
$$

is concave on the segment between $a _ { i } ^ { \mathrm { a n c } }$ and $a _ { i } ^ { \star }$ .

Assumption 3. The post-update actor $\begin{array} { r c l } { { a _ { i } ^ { + } } } & { { = } } & { { \pi _ { \theta ^ { + } } ( h _ { i } ) } } \end{array}$ satisfies $\begin{array} { r } { \mathbb { E } _ { s \sim d _ { \delta } } \| a _ { i } ^ { + } - \widetilde { a } _ { i } \| _ { 2 } \leq \varepsilon } \end{array}$

Assumption 4. For all relevant u and $v ,$

$$
| g _ { i } ( u ; s ) - g _ { i } ( v ; s ) | \leq L _ { Q } \| u - v \| _ { 2 } .\tag{15}
$$

Assumption 1 is the usual local ranking condition for using a learned offline critic only within supported candidates. Assumptions 2 and 4 are local regularity conditions on the short segment used by the trust gate, not global claims about the non-convex wireless objective. Assumption 3 states that supervised distillation tracks the teacher with bounded approximation error.

Lemma 1 (Nonnegative rectification gain). For every offline-support state s, $\Delta _ { i } ( s ) \geq 0$

Proof. The selected action $a _ { i } ^ { \star }$ maximizes the critic value over $\mathcal { U } _ { i } ( s )$ , and $a _ { i } ^ { \mathrm { a n c } } \in \mathcal { U } _ { i } ( s )$ by Assumption 1. Hence the selected action cannot have a smaller critic value than the anchor, which gives (13). □

Lemma 1 shows why local corrections are evaluated in a joint-action context. The candidate accepted by STAR-CRDT is never worse than the anchor under the centralized critic.

Proposition 1 (Gain of the trust-gated teacher). For $\widetilde { a } _ { i } =$ $( 1 - \tau _ { i } ) a _ { i } ^ { \mathrm { a n c } } + \tau _ { i } a _ { i } ^ { \star } \mathrm { a n d } 0 \leq \tau _ { i } \leq 1 ,$

$$
Q _ { \phi } ( s , ( \widetilde { a } _ { i } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) \geq Q _ { \phi } ( s , ( a _ { i } ^ { \mathrm { a n c } } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) + \tau _ { i } \Delta _ { i } ( s ) .\tag{16}
$$

Proof. By concavity on the segment between $a _ { i } ^ { \mathrm { a n c } }$ and $a _ { i } ^ { \star }$ , Jensen’s inequality gives

$$
\begin{array} { r l } & { Q _ { \phi } \bigl ( s , ( \widetilde { a } _ { i } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) \bigr ) \geq \bigl ( 1 - \tau _ { i } \bigr ) Q _ { \phi } \bigl ( s , \bigl ( a _ { i } ^ { \mathrm { a n c } } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } \bigr ) \bigr ) } \\ & { \qquad + \tau _ { i } Q _ { \phi } \bigl ( s , \bigl ( a _ { i } ^ { \star } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } \bigr ) \bigr ) , } \end{array}\tag{17}
$$

which is equivalent to (16).

□

Proposition 1 explains the trust gate. A partial move toward the rectified action preserves a proportional critic gain, so the teacher can improve the anchor while remaining conservative. Theorem 1 (Offline-support policy improvement). If $\begin{array} { r } { \mathbb { E } _ { s \sim d _ { \beta } } \| a _ { i } ^ { + } - \widetilde { a } _ { i } \| _ { 2 } \leq \varepsilon , } \end{array}$ , then

$$
\begin{array} { r } { \mathbb { E } _ { d _ { \beta } } Q _ { \phi } ( s , ( a _ { i } ^ { + } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) \geq \mathbb { E } _ { d _ { \beta } } Q _ { \phi } ( s , ( a _ { i } ^ { \mathrm { a n c } } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) } \\ { + \mathbb { E } _ { d _ { \beta } } [ \tau _ { i } \Delta _ { i } ( s ) ] - L _ { Q } \varepsilon . } \end{array}\tag{18}
$$

Thus, when $\mathbb { E } _ { d _ { \beta } } [ \tau _ { i } \Delta _ { i } ( s ) ] > L _ { Q } \varepsilon$ , the actor update improves the centralized-critic surrogate in expectation.

Proof. By Assumption $^ { 4 , }$

$$
Q _ { \phi } ( s , ( a _ { i } ^ { + } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) \geq Q _ { \phi } ( s , ( \widetilde { a } _ { i } , \mathbf { a } _ { - i } ^ { \mathrm { c t x } } ) ) - L _ { Q } \Vert a _ { i } ^ { + } - \widetilde { a } _ { i } \Vert _ { 2 } .\tag{19}
$$

Taking expectation over $d _ { \beta } ,$ , applying Assumption 3, and using Proposition 1 proves (18). □

Candidate search first produces a nonnegative centralizedcritic gain over the anchor action, the trust gate admits only the supported portion of this gain, and teacher distillation transfers the resulting correction to the local actor. Under the stated local ranking, regularity, and distillation-error conditions, if the expected trusted rectification gain exceeds $L _ { Q } \varepsilon$ , the postupdate actor has a larger centralized-critic surrogate value than the anchor in expectation over $d _ { \beta }$ . This gives the offlinesupport policy-improvement guarantee used by AERIS.

## V. EXPERIMENTAL EVALUATION

## A. Evaluation Setup

We evaluate AERIS under random-mobility and road-map deployment protocols. The first isolates fixed-log offline improvement under the bounded Brownian model used for log collection, while the second freezes each learned policy and transfers it without road-specific tuning to OSM roadconstrained mobility. Thus, random mobility supplies a highrandomness source log, while road-map deployment mimics the common mismatch between available offline logs and actual application scenarios.

The evaluation instantiates the model in Sections III-A and III-B with the parameters in Table I. By default, $M = 3 \mathrm { U A V s }$ serve $K = 6$ users and $L = 3$ targets, with two users and one target per UAV. Service sets use quota-based nearest-UAV association, assigning each user or target to the closest UAV that has not reached its user or target cap [7]–[9]. The scale study varies M from 3 to 6 UAVs to test coordination as the fleet size increases [9], [10]. For high-risk low-altitude urban operation, $d _ { \operatorname* { m i n } } = 3 0$ m follows conservative multi-UAV ISAC/RadCom safety settings [9], [10]. Other values follow UAV-link, ISAC beamforming, mobility/safety, and OSM/GIS road-network studies [7], [8], [11], [36]–[39], [41].

The fixed log is collected once by the preliminary TD3 behavior controller and then held fixed. All offline methods receive the same trajectories, rewards, local histories, global

$$
\Phi \mathrm { S T A R \ll \mathrm { T D 3 + B C \equiv \mathrm { C R D T \Phi \Phi \mathrm { ~ D T \Phi ~ + ~ { \cal ~ C R R } \mp \Phi ~ \mathrm { C R I G A } \mathrm { ~ \mathcal { ~ B C ~ \neq ~ { \cal ~ O M A R } ~ } } } } } }
$$

![](images/5635f955b9d861d8f2a94c074da8263a701916acafacb361d9c421919187ecf6.jpg)  
TD3 training steps (k)

![](images/19d86075395d1fd23dd6973b1876f814df6a7cabcb20bdacea0c4cf43164365a.jpg)

![](images/007aab1998013f880f93d66a42aaad0dd127e172bcd1f76faa052e1bac9725e6.jpg)

![](images/7d7abe727df57784c190d98ec7e4789393642d98566e2162b3b56b6ff03511c2.jpg)  
Fig. 4. Training and ablation results under random mobility. The first three panels show TD3 behavior-policy training, offline policy evaluation curves, and best-checkpoint return comparison. The fourth panel compares CRDT and STAR-CRDT in joint-action OOD ratio and ISAC violation cost. STAR-CRDT reduces violations while keeping critic-corrected actions near useful support.

![](images/b9b49bb591a0c8bfa5961199969c7ae62c2cce827d6b578c4dc2d7e951d06d31.jpg)  
(a) Return

![](images/61df71f1ecdfd5c9da492622657c378a51ac81eae3372469839f29acea988d22.jpg)  
(b) Sum rate

![](images/40acec44279993e869aabe1ee1b295b91c834a87f210010da74aede4215ffafa.jpg)  
(c) Margin

![](images/5515282f4dbfb4b4f32c0c8504ffe3ed27705349f45adc869ce67f4ce48aabf8.jpg)  
(d) Pass rate

![](images/68bfcf44d9b1dd850e63792607eb0a0fa7cba6c844a4d8eab7eb7839738b1197.jpg)  
(e) Coll. risk  
Fig. 5. Random-mobility offline evaluation. The compact five-panel row reports return, communication sum rate, sensing margin, sensing pass rate, and collision-risk count. STAR-CRDT obtains the best overall tradeoff.

![](images/b8b83a544ea06483a435b71f5457630389b144648e04e0e62094c1fd52afb0b9.jpg)  
(a) Return

![](images/0f3ad8ba01e6baafe498cf9f41ac91b75f09ed7662074eac93933a7bba4208ad.jpg)  
(b) Sum rate

![](images/3249afd59ed79dacc050642daec58c5d0f0041c6cef39039bf7a6725e7c46859.jpg)  
(c) Margin

![](images/5c03c435817baec7872328a01419e37b5124e30e3ead5dc1da113043158f87c4.jpg)  
(d) Pass rate

![](images/56dfd698174fe655e6adbe2a8ad98f6a2780675738ad8d05904ccbf4abff2bb5.jpg)  
UAVs (e) Coll. risk  
Fig. 6. System-scale evaluation under random mobility. The compact five-panel row follows Fig. 5 and reports return, communication sum rate, sensing margin, sensing pass rate, and collision-risk count. STAR-CRDT remains strongest when the trained shared actor is evaluated with more UAVs.

states for centralized training, and decentralized observations, so the comparison measures improvement from the same support without extra online interaction or road-map adaptation.

Baselines. We compare with representative offline RL and offline MARL methods spanning imitation, conservative updates, sequence modeling, and multi-agent actor rectification [16], [21], [22]. The dataset, training budget, model capacity where applicable, and evaluation episodes are kept consistent.

• Behavior cloning (BC) [42] tests whether pure supervised imitation of logged actions is sufficient.

• DT is a Decision Transformer baseline [29] that learns return-conditioned decentralized actions from local histories.

• TD3+BC [27] and CRR (Critic Regularized Regression) [28] represent conservative offline actor updates.

CRDT is our sequence-model ablation. It keeps the local Decision-Transformer actor [29] and centralized critic regularization, but removes support-aware candidate search and trust-gated distillation. Comparing CRDT with STAR-CRDT isolates whether trust-gated local rectification is necessary beyond critic regularization alone.

• OMIGA [31] and OMAR [30] are offline MARL base-• Online TD3 [35] is reported as the behavior-policy reference rather than an additional online competitor.

TABLE I KEY PARAMETERS
<table><tr><td colspan="2">System parameters</td></tr><tr><td>Service area  $L _ { x } \times L _ { y }$ </td><td> $\overline { { 5 0 0 \times 5 0 0 ~ \mathrm { m } ^ { 2 } } }$ </td></tr><tr><td>Episode horizon T</td><td>200 slots</td></tr><tr><td>Main topology  $( M , K , L )$ </td><td>(3, 6, 3)</td></tr><tr><td>Scale sweep M</td><td>3 to 6UAVs</td></tr><tr><td>UAV altitude  $[ z _ { \operatorname* { m i n } } , z _ { \operatorname* { m a x } } ]$ </td><td>[100,150] m</td></tr><tr><td>UAV speed  $V _ { \mathrm { m a x } }$ </td><td>5 m/slot</td></tr><tr><td>Ground speed  $\overline { { v _ { \alpha . } ^ { \mathrm { m a x } } } }$  u</td><td>0.5 m/slot</td></tr><tr><td>Safety distance  $d _ { \mathrm { m i n } }$ </td><td>30 m</td></tr><tr><td>Array size  $N _ { t }$ </td><td>4 antennas</td></tr><tr><td>Power budget  $P _ { \mathrm { m a x } }$ </td><td>20 dBm</td></tr><tr><td>Noise power  $\sigma ^ { 2 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Bandwidth B</td><td>1 normalized Hz</td></tr><tr><td>Sensing threshold Γ</td><td> $2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Log size</td><td>1000 episodes, 200k transitions</td></tr><tr><td>Held-out evaluation</td><td>50 episodes</td></tr><tr><td colspan="2">STAR-CRDT parameters</td></tr><tr><td>Actor context length</td><td> $\overline { { K = 5 0 ~ \mathrm { s l o t s } } }$ </td></tr><tr><td>Actor hidden size</td><td>256</td></tr><tr><td>Transformer blocks</td><td> $6 \mathrm { ~ l a y e r s , ~ 8 ~ h e a d s }$ </td></tr><tr><td>Critic hidden size</td><td>512</td></tr><tr><td>Discount and expectile</td><td> $\gamma = 0 . 9 9 , \kappa = 0 . 8$ </td></tr><tr><td>Q regularization  $\lambda _ { Q }$ </td><td> $_ { 0 . 1 8 }$ </td></tr><tr><td>Rectification samples  $N _ { r }$ </td><td> $4 { \mathrm { ~ p e r ~ a n c h o r } }$ </td></tr><tr><td>Rectification std.</td><td> $\sigma _ { \pi } = 0 . 1 8 , \sigma _ { \beta } = 0 . 0 6$ </td></tr><tr><td>Trust range</td><td> $0 . 3 5 ~ \mathrm { t o } ~ 0 . 4 5$ </td></tr></table>

lines that respectively emphasize global-to-local value regularization and actor rectification.

Evaluation Metrics. We report return $\begin{array} { r } { J } { { \bf \Delta } = { \bf \sum } _ { t = 0 } ^ { I - 1 } r _ { t } , } \end{array}$ average communication sum rate $\begin{array} { r } { R _ { \mathrm { s u m } } = T ^ { - 1 } \sum _ { t } U _ { \mathrm { c o m m } } ( t ) } \end{array}$ sensing pass rate (fraction of sensing links satisfying thresholds), sensing margin (average normalized excess over thresholds), and collision-risk count $\begin{array} { r } { C _ { \mathrm { c o l } } = T ^ { - 1 } \sum _ { t } N _ { t } ^ { \mathrm { c o l } } } \end{array}$ , where lower is better. Effective improvement should jointly improve return, communication, sensing, and safety.

## B. Random-Mobility Offline Policy Improvement

We first test whether offline training can refine the initial behavior controller without reintroducing the online exploration risk in Section II. Fig. 4 shows that TD3 forms an initial controller, but its online curve remains noisy because value learning still depends on trial rollouts. After the dataset is frozen, imitation and conservative updates improve only part of the objective, whereas STAR-CRDT reaches the highest and most stable offline plateau. This supports the intended use of AERIS, where online interaction stops after an initial log exists and policy refinement continues offline.

Ablation study. We use CRDT as the main ablation of STAR-CRDT, as it removes support-aware candidate search and trust-gated distillation from the full design. The fourth panel of Fig. 4 compares CRDT and STAR-CRDT using the same joint-action OOD ratio and ISAC violation cost as Fig. 1(b). CRDT remains closer to the log, but still leaves a higher violation cost because critic regularization alone does not identify which local correction is globally beneficial. Compared with CRDT and the baselines in Fig. 1(b), STAR-CRDT lowers violation cost while avoiding the large OOD shift caused by aggressive rectification, since it permits only a slightly larger but controlled support shift through criticimproving candidate selection and trusted-teacher distillation.

Together with Fig. 5, where CRDT is stronger than plain DT but still trails STAR-CRDT, this ablation shows that centralized critic feedback must be coupled with supportaware rectification and trust-gated distillation to obtain reliable offline improvement.

The random-mobility results further verify that STAR-CRDT does more than avoid sensing-threshold and collisionrisk violations. In Fig. 5, STAR-CRDT improves return by 29.3% over TD3+BC, the strongest non-STAR baseline. It also improves the best baseline communication sum rate by 3.4%, sensing pass rate by 4.8%, and sensing margin by 69.1%, while reducing collision-risk count by 54.2%. These gains align with P0 because the policy improves communication and sensing while reducing sensing-threshold violations and collision-risk events.

The remaining baselines confirm the imitation and OOD dilemma in Section II-B. BC and DT remain close to the log and cannot reliably remove residual sensing violations. TD3+BC and CRR are safer than aggressive actor maximization, but their behavior regularization limits improvement once the logged controller is suboptimal. OMAR tends to rectify actions more aggressively and obtains poor return, which is consistent with the danger of leaving joint-action support. Thus, the comparison supports the offline-support improvement mechanism proved in Section IV-C.

## C. System-Scale Evaluation

We then test zero-shot system-scale transfer. The trained shared local actors are evaluated with M = 3 to 6 UAVs without additional offline retraining, while keeping two communication users and one sensing target per UAV. Thus, local observation/action semantics stay fixed but joint coordination becomes harder. STAR-CRDT ranks first at every scale. At $M = 6 ,$ it improves return by 39.3% over TD3+BC and by 60.8% over CRR, while maintaining the best sensing pass rate and sensing margin.

The scale trend shows that the local actor has learned a rule that remains useful as the joint-action space and pairwise safety constraints grow. Larger teams create more interference, coupled sensing beams, and collision-risk events. STAR-CRDT evaluates such team-level effects through the centralized critic before distillation, whereas conservative baselines inherit residual violations and aggressive rectification becomes less stable. This supports the CTDE choice in AERIS, where centralized information selects corrections during training and only the local sequence actor is deployed.

## D. Zero-Shot Road-Map Deployment

Road-map deployment tests whether policies trained on the fixed random-mobility log can transfer to road-constrained urban mobility without adaptation. We deploy frozen policies on real OSM road graphs and Geofabrik OSM/GIS data extracts [37], [38], [41] using three 500 $\mathrm { m } \times 5 0 0 \mathrm { \cdot }$ m Hong Kong scenes: a dense grid (HK-OSM-S1, $7 2 . 6 \mathrm { k m / k m ^ { 2 } } )$ , an arterial corridor (HK-OSM-S2, 15.5 km/km<sup>2</sup>), and a sparse local-road scene (HK-OSM-S3, $9 . 8 \mathrm { k m / k m ^ { 2 } } )$ . No road-scene rollout is used for training, early stopping, checkpoint selection, or hyperparameter tuning. We report both M = 3 and M = 6 to test increasing interference and safety coupling.

TABLE II  
ZERO-SHOT ROAD-MAP DEPLOYMENT EVALUATION. METRICS ARE SHOWN BY ROW FOR EACH ROAD SCENE. BEST VALUES ARE BOLD RED, SECOND-BEST VALUES ARE UNDERLINED BOLD BLUE, AND LOWER IS BETTER ONLY FOR COLL. RISK.
<table><tr><td colspan="2">Scene Metric</td><td colspan="8">M = 3</td><td colspan="8">M = 6</td></tr><tr><td></td><td colspan="3">STAR TD3+BC</td><td>DT</td><td>CRR</td><td>OMIGA</td><td>BC</td><td>TD3</td><td>OMAR</td><td>STAR</td><td>TD3+BC CRDT</td><td></td><td>DT</td><td>CRR</td><td>OMIGA</td><td>BC TD3</td><td>OMAR</td></tr><tr><td>S1</td><td>Ret. Comm.</td><td>648.3 3.63</td><td>570.9 576.8</td><td>512.3</td><td>516.5</td><td>384.9</td><td>395.8</td><td>410.0</td><td>-675.0</td><td>1288.1 6.76</td><td>942.3</td><td>272.7</td><td>372.1</td><td>329.4</td><td>387.6</td><td>319.9 351.8</td><td>-1569.2</td></tr><tr><td></td><td></td><td>3.27</td><td>3.37</td><td>3.29</td><td>3.49</td><td>3.25</td><td>3.29</td><td>3.24 0.668</td><td>3.37 0.310</td><td>5.74</td><td>5.65</td><td>5.55</td><td>5.82</td><td>5.74</td><td>5.63</td><td>5.64</td><td>5.37</td></tr><tr><td>Pass</td><td>0.707</td><td>0.715</td><td>0.707</td><td>0.689</td><td>0.685</td><td>0.661</td><td>0.659</td><td></td><td>0.753</td><td>0.749</td><td>0.677</td><td>0.681</td><td>0.678</td><td>0.673</td><td>0.674</td><td>0.676</td><td>0.396</td></tr><tr><td>Margin</td><td>0.0366</td><td>0.0341</td><td>0.0267</td><td>0.0237</td><td>0.0213</td><td>0.0194</td><td>0.0211</td><td>0.0181</td><td>-0.0574</td><td>0.0572 0.0585</td><td>0.0311</td><td>0.0351</td><td>0.0341</td><td>0.0339</td><td>0.0316</td><td>0.0318</td><td>-0.0401</td></tr><tr><td>Coll. risk</td><td>0.028</td><td>0.063</td><td>0.073</td><td>0.058</td><td>0.080</td><td>0.126</td><td>0.118</td><td>0.116</td><td>0.106</td><td>0.333 0.424</td><td>0.766</td><td>0.655</td><td>0.742</td><td>0.632</td><td>0.700</td><td>0.677</td><td>0.878</td></tr><tr><td>S2</td><td></td><td>520.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-608.8</td><td>-592.4</td><td>-622.6</td><td></td><td></td><td></td><td></td></tr><tr><td>Ret.</td><td></td><td>-54.3</td><td>97.8</td><td>58.3 2.92</td><td>-336.5</td><td>-123.8 2.82</td><td>-29.9 2.84</td><td>-69.7 2.81</td><td>-166.8 3.61</td><td>389.7 4.39</td><td></td><td></td><td>-1367.7</td><td>-667.0</td><td>-629.3</td><td>-761.4</td><td>-1341.3 4.72</td></tr><tr><td>Comm.</td><td>3.33</td><td>2.74 0.576</td><td>3.05 0.602</td><td>0.580</td><td>2.50 0.486</td><td>0.534</td><td>0.547</td><td>0.550</td><td>0.513 0.712</td><td>3.63 0.642</td><td>4.07 0.647</td><td>3.91 0.630</td><td>3.57 0.535</td><td>3.91 0.613</td><td>3.92 0.623</td><td>3.81 0.606</td><td>0.587</td></tr><tr><td>Pass Margin</td><td>0.695 0.0555</td><td>0.0066</td><td>0.0363</td><td>0.0356</td><td>-0.0056</td><td>0.0225</td><td>0.0230</td><td>0.0240</td><td>0.0156 0.0814</td><td>0.0583</td><td>0.0654</td><td>0.0650</td><td>0.0348</td><td>0.0608</td><td>0.0645</td><td>0.0597</td><td>0.0481</td></tr><tr><td>Coll. risk</td><td>0.127</td><td>0.133</td><td>0.159</td><td>0.112</td><td>0.119</td><td>0.154</td><td>0.092</td><td>0.139</td><td>0.261</td><td>0.732 0.896</td><td>1.175</td><td>1.070</td><td>1.159</td><td>1.024</td><td>1.075</td><td>1.080</td><td>1.694</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S3 Ret.</td><td></td><td>531.8 258.4</td><td>119.9</td><td>-38.3</td><td>74.1</td><td>-190.1</td><td>-139.2</td><td>-142.5</td><td>-539.5</td><td>873.8</td><td>376.5</td><td>-316.1</td><td>-407.9</td><td>-263.3</td><td>-492.7</td><td>-607.5 -428.7</td><td>-1557.7</td></tr><tr><td>Comm.</td><td>3.47</td><td>2.90</td><td>2.77</td><td>2.41</td><td>2.57</td><td>2.45</td><td>2.44</td><td>2.55</td><td>3.14</td><td>6.10 5.07</td><td>4.89</td><td>4.64</td><td>4.77</td><td>4.47</td><td>4.47</td><td>4.55</td><td>5.35</td></tr><tr><td></td><td>0.683</td><td>0.635</td><td>0.599</td><td>0.576</td><td>0.589</td><td>0.503</td><td>0.529</td><td>0.517</td><td>0.404</td><td>0.720 0.665</td><td>0.616</td><td>0.587</td><td>0.604</td><td>0.570</td><td>0.553</td><td>0.570</td><td>0.448</td></tr><tr><td>Pass</td><td></td><td>0.0402 0.0075</td><td>0.0013</td><td>-0.0056</td><td>0.0013</td><td>-0.0130</td><td>-0.0142</td><td>-0.0130</td><td>-0.0310</td><td>0.0689 0.0292</td><td>0.0161</td><td>0.0112</td><td>0.0145</td><td>0.0063</td><td>0.0006</td><td>0.0042</td><td>-0.0154</td></tr><tr><td></td><td>Margin</td><td></td><td>0.089</td><td>0.116</td><td>0.062</td><td>0.068</td><td>0.092</td><td>0.079</td><td>0.179</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.613</td><td>1.105</td></tr><tr><td></td><td>Coll. risk</td><td>0.018</td><td>0.057</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.409</td><td>0.325</td><td>0.868</td><td>0.717</td><td>0.677</td><td>0.657</td><td>0.691</td><td></td></tr></table>

![](images/16e364dd35064dcf9e77fa2522d0dd9b194025f10a4c9c4c94452e3101eaed40.jpg)  
Fig. 7. Representative STAR-CRDT rollout under road-map mobility. The left and right panels show top-down and 3-D views, respectively.

The representative rollout in Fig. 7 shows coordinated UAV motion around road-constrained users and targets. The 3-D view confirms feasible altitude control, and the trajectories balance communication coverage, sensing target tracking, and safety separation. Because road maps can concentrate users and targets along shared corridors, STAR-CRDT must learn a joint ISAC response rather than a purely communicationoriented motion.

Table II shows that STAR-CRDT obtains the best return in all six scene-scale settings. Across the 30 metric rows, it is best in 24 rows and remains top-two in 29 rows. At M = 6, it improves return by 36.7% in the dense grid and 132.1% in the sparse-road scene over the strongest baseline. Its only non-top-two case is collision risk in arterial M = 3, where BC has fewer risk events but much lower return, while OMAR sometimes gains communication rate at the cost of sensing and safety.

These results separate robustness from overfitting to random mobility. The three road structures create different route changes, demand concentrations, and coverage gaps, yet STAR-CRDT remains strongest because the centralized critic trains support-aware corrections instead of memorizing one mobility geometry. The transfer gains suggest that STAR-CRDT learns reusable local ISAC responses, including coverage recovery, sensing-margin preservation, and safety-aware separation. Overall, AERIS improves communication, sensing, and safety from fixed logs under decentralized execution.

## VI. RELATED WORKS

Multi-UAV ISAC and RL-based wireless control. ISAC integrates communication and sensing, and UAVs add controllable aerial links with line-of-sight opportunities [1]–[4], [6]. Existing multi-UAV ISAC studies mainly optimize trajectories, beamforming, and sensing quality through modelbased optimization or online learning [7]–[10]. RL and MARL have also been used for adaptive wireless networking and resource allocation [14]–[19]. These works provide online or model-based baselines, while AERIS focuses on fixed-log improvement when new exploration is costly.

Offline RL and offline MARL. Offline RL learns from fixed datasets [20]–[22]. Main families include imitation such as BC, support-constrained value learning such as Batch-Constrained deep Q-learning (BCQ) and Bootstrapping Error Accumulation Reduction (BEAR) [23], [24], conservative or implicit value learning such as Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) [25], [26], actorregularized updates such as TD3+BC and CRR [27], [28], and sequence models such as DT [29]. Offline MARL extends fixed-dataset learning to coordinated agents with centralized training and decentralized policies. OMIGA uses implicit global-to-local value regularization, while OMAR searches critic-improving actor corrections [30], [31]. Yet such methods may inherit imperfect logs or rely on unsupported critic gains. STAR-CRDT addresses both issues by searching for local corrections near logged actions and distilling them only when the centralized critic predicts reliable team-level improvement.

## VII. CONCLUSION

This paper presented AERIS, a novel offline policy improvement framework for distributed multi-UAV ISAC. AERIS reformulates trajectory-and-beamforming control as a fixed-log CTDE offline MARL problem, enabling local UAV execution while using logged global information to assess team-level communication, sensing, and safety effects. To solve this problem, we developed STAR-CRDT, a new algorithm that distills support-aware teacher corrections selected by centralized critic feedback. This lets AERIS improve beyond imperfect logs without degenerating into pure imitation or unsupported critic maximization. With an offline-support improvement guarantee and evaluations under random mobility, system-scale transfer, and unseen OSM road maps, AERIS consistently improves return, communication, sensing, and safety from fixed flight logs before deployment.

## REFERENCES

[1] F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, “Integrated sensing and communications: Toward dual-functional wireless networks for 6g and beyond,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 6, pp. 1728–1767, Jun. 2022.

[2] J. A. Zhang, F. Liu, C. Masouros, R. W. Heath, Z. Feng, L. Zheng, and A. Petropulu, “An overview of signal processing techniques for joint communication and radar sensing,” IEEE Journal of Selected Topics in Signal Processing, vol. 15, no. 6, pp. 1295–1315, Nov. 2021.

[3] F. Liu, C. Masouros, A. P. Petropulu, H. Griffiths, and L. Hanzo, “Joint radar and communication design: Applications, state-of-the-art, and the road ahead,” IEEE Transactions on Communications, vol. 68, no. 6, pp. 3834–3862, Jun. 2020.

[4] Y. Zeng, R. Zhang, and T. J. Lim, “Wireless communications with unmanned aerial vehicles: Opportunities and challenges,” IEEE Communications Magazine, vol. 54, no. 5, pp. 36–42, May 2016.

[5] Y. Zeng, J. Lyu, and R. Zhang, “Cellular-connected uav: Potential, challenges, and promising technologies,” IEEE Wireless Communications, vol. 26, no. 1, pp. 120–127, Feb. 2019.

[6] M. Mozaffari, W. Saad, M. Bennis, Y.-H. Nam, and M. Debbah, “A tutorial on uavs for wireless networks: Applications, challenges, and open problems,” IEEE Communications Surveys and Tutorials, vol. 21, no. 3, pp. 2334–2360, 2019.

[7] Z. Lyu, G. Zhu, and J. Xu, “Joint maneuver and beamforming design for uav-enabled integrated sensing and communication,” IEEE Transactions on Wireless Communications, vol. 22, no. 4, pp. 2424–2440, Apr. 2023.

[8] K. Meng, Q. Wu, S. Ma, W. Chen, and T. Q. S. Quek, “Uav trajectory and beamforming optimization for integrated periodic sensing and communication,” IEEE Wireless Communications Letters, vol. 11, no. 6, pp. 1211–1215, Jun. 2022.

[9] Q. Gao, R. Zhong, H. Shin, and Y. Liu, “Marl-based uav trajectory and beamforming optimization for isac system,” IEEE Internet of Things Journal, vol. 11, no. 24, pp. 40 492–40 505, Dec. 2024.

[10] S. Cheng, X. Lin, X. Li, and J. Wang, “Joint uav trajectory and radcom task schedule for ivns: A game-embedding multi-agent deep reinforcement learning approach,” IEEE Transactions on Wireless Communications, vol. 24, no. 1, pp. 181–196, Jan. 2025.

[11] Y. Zeng, J. Xu, and R. Zhang, “Energy minimization for wireless communication with rotary-wing uav,” IEEE Transactions on Wireless Communications, vol. 18, no. 4, pp. 2329–2345, Apr. 2019.

[12] M. Mozaffari, W. Saad, M. Bennis, and M. Debbah, “Mobile unmanned aerial vehicles (uavs) for energy-efficient internet of things communications,” IEEE Transactions on Wireless Communications, vol. 16, no. 11, pp. 7574–7589, Nov. 2017.

[13] Q. Wu, Y. Zeng, and R. Zhang, “Uav-enabled wireless powered communication networks,” IEEE Transactions on Wireless Communications, vol. 17, no. 7, pp. 4851–4865, Jul. 2018.

[14] H. Mao, M. Alizadeh, I. Menache, and S. Kandula, “Resource management with deep reinforcement learning,” in Proceedings of the 15th ACM Workshop on Hot Topics in Networks, 2016, pp. 50–56.

[15] Z. Xu, J. Tang, J. Meng, W. Zhang, Y. Wang, C. H. Liu, and D. Yang, “Experience-driven networking: A deep reinforcement learning based approach,” in Proceedings of the IEEE Conference on Computer Communications, 2018, pp. 1871–1879.

[16] N. C. Luong, D. T. Hoang, S. Gong, D. Niyato, P. Wang, Y.-C. Liang, and D. I. Kim, “Applications of deep reinforcement learning in communications and networking: A survey,” IEEE Communications Surveys and Tutorials, vol. 21, no. 4, pp. 3133–3174, 2019.

[17] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, “Multiagent actor-critic for mixed cooperative-competitive environments,” in Proceedings of Advances in Neural Information Processing Systems, 2017, pp. 6379–6390.

[18] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson, “Counterfactual multi-agent policy gradients,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2018, pp. 2974–2982.

[19] T. Rashid, M. Samvelyan, C. Schroeder, G. Farquhar, J. Foerster, and S. Whiteson, “Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning,” in Proceedings of the International Conference on Machine Learning, 2018, pp. 4295–4304.

[20] J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine, “D4rl: Datasets for deep data-driven reinforcement learning,” arXiv preprint arXiv:2004.07219, 2020.

[21] C. Gulcehre, Z. Wang, A. Novikov, T. Paine, S. G. Colmenarejo, K. Zolna, R. Agarwal, J. Merel, D. Mankowitz, C. Paduraru, G. Dulac-Arnold, J. Li, M. Norouzi, M. Hoffman, N. Heess, and N. de Freitas, “Rl unplugged: Benchmarks for offline reinforcement learning,” in Proceedings of Advances in Neural Information Processing Systems, 2020.

[22] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv preprint arXiv:2005.01643, 2020.

[23] S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in Proceedings of the International Conference on Machine Learning, 2019, pp. 2052–2062.

[24] A. Kumar, J. Fu, M. Soh, G. Tucker, and S. Levine, “Stabilizing offpolicy q-learning via bootstrapping error reduction,” in Proceedings of Advances in Neural Information Processing Systems, 2019, pp. 11 784– 11 794.

[25] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative q-learning for offline reinforcement learning,” in Proceedings of Advances in Neural Information Processing Systems, 2020, pp. 1179–1191.

[26] I. Kostrikov, A. Nair, and S. Levine, “Offline reinforcement learning with implicit q-learning,” in Proceedings of the International Conference on Learning Representations, 2022.

[27] S. Fujimoto and S. Gu, “A minimalist approach to offline reinforcement learning,” in Proceedings of Advances in Neural Information Processing Systems, 2021.

[28] Z. Wang, A. Novikov, K. Zolna, J. T. Springenberg, S. Reed, B. Shahriari, N. Siegel, J. Merel, C. Gulcehre, N. Heess, and M. Riedmiller, “Critic regularized regression,” in Proceedings of Advances in Neural Information Processing Systems, 2020.

[29] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, “Decision transformer: Reinforcement learning via sequence modeling,” in Proceedings ofAdvances in Neural Information Processing Systems, 2021, pp. 15 084–15 097.

[30] L. Pan, L. Huang, T. Ma, and H. Xu, “Plan better amid conservatism: Offline multi-agent reinforcement learning with actor rectification,” in Proceedings of the International Conference on Machine Learning, 2022, pp. 17 221–17 237.

[31] X. Wang, H. Xu, Y. Zheng, and X. Zhan, “Offline multi-agent reinforcement learning with implicit global-to-local value regularization,” arXiv preprint arXiv:2307.11620, 2023.

[32] J. Garcia and F. Fernandez, “A comprehensive survey on safe reinforcement learning,” Journal of Machine Learning Research, vol. 16, no. 1, pp. 1437–1480, 2015.

[33] J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” in Proceedings of the International Conference on Machine Learning, 2017, pp. 22–31.

[34] Y. Chow, O. Nachum, E. Duenez-Guzman, and M. Ghavamzadeh, “A lyapunov-based approach to safe reinforcement learning,” in Proceedings of Advances in Neural Information Processing Systems, 2018, pp. 8092–8101.

[35] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proceedings of the International Conference on Machine Learning, 2018, pp. 1587–1596.

[36] T. Camp, J. Boleng, and V. Davies, “A survey of mobility models for ad hoc network research,” Wireless Communications and Mobile Computing, vol. 2, no. 5, pp. 483–502, 2002.

[37] OpenStreetMap contributors, “Copyright and License,” https://www. openstreetmap.org/copyright, 2026, accessed: Jun. 11, 2026.

[38] Geofabrik GmbH, “OpenStreetMap Data in GIS Formats Free,” https://download.geofabrik.de/osm-data-in-gis-formats-free.pdf, 2026, accessed: Jun. 11, 2026.

[39] A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal lap altitude for maximum coverage,” IEEE Wireless Communications Letters, vol. 3, no. 6, pp. 569–572, Dec. 2014.

[40] S. Boyd and L. Vandenberghe, Convex Optimization. Cambridge, U.K.: Cambridge University Press, 2004.

[41] G. Boeing, “Osmnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks,” Computers, Environment and Urban Systems, vol. 65, pp. 126–139, 2017.

[42] D. A. Pomerleau, “ALVINN: An autonomous land vehicle in a neural network,” in Proceedings ofAdvances in Neural Information Processing Systems, 1989, pp. 305–313.