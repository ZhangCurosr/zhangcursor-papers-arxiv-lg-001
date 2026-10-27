# MULTI-HORIZON FUTURE DISCOVERY FOR PHYSICS-INFORMED EXPLORATION

Shuwei Wang1,2,3 Qiang Guan1,2,3,† Brian Qiu1 Haoxiang Huang4 Haoxi Wang1,2 Chenyu Li1,2 Bo Xu1,2,3,† Sen Cui5,†

¹University of Chinese Academy of Sciences 2Institute of Automation, Chinese Academy of Sciences (CASIA) 3School of Artificial Intelligence, University of Chinese Academy of Sciences 4University of Science and Technology of China 5Tsinghua University

†Corresponding authors

## ABSTRACT

Exploration in physical control environments requires agents to identify actions whose consequences reveal previously unexplored aspects of the underlying dynamics. However, many intrinsic motivation methods rely on immediate prediction errors, state novelty, or local physical changes, which may overlook informative outcomes that emerge across extended temporal horizons. We investigate whether learning structured representations of reachable futures can provide a more effective basis for exploration. To this end, we introduce Hamiltonian Temporal Contrastive Curiosity (HTC²), a framework that combines multi-horizon temporal representation learning with Hamiltonian phase-space structure. A horizon-conditioned contrastive critic learns temporal relationships from future-state pairs sampled within the same episode. The distance between positive-pair representations is then used as an intrinsic reward, encouraging the agent to explore futures that remain insufficiently organized in the learned representation. To incorporate information about physical dynamics, we additionally modulate this reward using bounded multi-step Hamiltonian variation, rather than directly rewarding changes in the Hamiltonian itself. We integrate HTC2 with both Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO), and evaluate it across six DeepMind Control Suite tasks. Comparisons with ICM, RIDE, E3B, Plan2Explore, and a single-step Hamiltonian curiosity baseline are complemented by state-coverage evaluations, component ablations, and phase-space diagnostics to characterize the contributions of temporal learning and physical structure to exploratory behavior.

## 1 INTRODUCTION

Exploration is a central challenge in reinforcement learning (RL), especially in physical control where useful behaviors often emerge only through temporally extended interactions with the environment (Burda et al., 2018b; Pathak et al., 2017; Thrun, 1992; Stadie et al., 2015; Sutton et al., 1998; Mohamed et al., 2026). In such settings, an agent must discover not only unfamiliar states, but also how its actions shape future system dynamics (Toth et al., 2019; Garcia et al., 2025). For example, locomotion depends on coordinated changes in balance, contact, velocity, and momentum over multiple time steps rather than merely visiting visually different postures (Arzani et al., 2025; Noda et al., 2025; Toth et al., 2019; Garcia et al., 2025). Physical exploration therefore requires more than generic state-space coverage (Pathak et al., 2017; Mohamed et al., 2026; Burda et al., 2018b; Raileanu & Rocktäschel, 2020): the agent must identify action-conditioned futures that reveal informative dynamics.

Existing curiosity-driven methods promote exploration through prediction error, novelty, episodic uncertainty, or representation displacement (Pathak et al., 2017; Burda et al., 2018b; Raileanu & Rocktäschel, 2020; Henaff et al., 2022). These signals are often defined from local states or short transitions: prediction error can be affected by irrelevant stochasticity, novelty may encourage visitation without distinguishing dynamical relevance, and representation displacement does not necessarily separate coherent physical evolution from arbitrary feature change. Temporal contrastive learning provides a way to capture longer-range relations by learning representations between current state-action pairs and their future states (Ziarko et al., 2025; Hansen et al., 2022). Such relations can identify futures that remain poorly captured across multiple temporal scales, but they do not by themselves encode structured variation in the underlying physical dynamics.

![](images/b0c8e3952028244220880c842fa91c6d170618308bdf8bb0e3e4a6d61eae1a74.jpg)  
Figure 1: Overview of Hamiltonian Temporal Contrastive Curiosity (HTC2). Observations are encoded into Hamiltonian phase states; same-episode future pairs train a horizon-conditioned temporal contrastive critic; and the resulting temporal discovery signal is modulated by bounded multi-step Hamiltonian variation.

Physical inductive biases provide a natural way to structure this temporal information. Hamiltonian world models represent dynamics through coordinate-like and momentum-like latent variables and organize their evolution with a learned Hamiltonian (Greydanus et al., 2019; Toth et al., 2019; Tang et al., 2026). Such phase-space representations describe how actions alter latent dynamics over time. DreamSAC exploits this structure by rewarding the change in the learned Hamiltonian across a single transition (Tang et al., 2026). However, single-step Hamiltonian-change curiosity remains local in time: a large instantaneous variation does not necessarily indicate that an action reaches an informative future over longer horizons. This motivates exploration that identifies informative action-conditioned futures across multiple temporal scales while accounting for structured variation in the learned dynamics.

To address this challenge, we introduce Hamiltonian Temporal Contrastive Curiosity (HTC2), which formulates exploration as multi-step future discovery in a learned Hamiltonian phase space. As illustrated in Figure 1, HTC² trains a horizon-conditioned temporal contrastive critic on same-episode future pairs, using positive-pair representation distance as the primary curiosity signal. Bounded multi-step Hamiltonian variation then modulates this signal according to structured changes in the learned phase dynamics, while temporal novelty remains the exploration objective. A latent RND (Burda et al., 2018b) warm start stabilizes the initial learning stage before transitioning to the HTC2 signal.

## Our contributions are summarized as follows:

1. We formulate physical exploration as multi-step action-conditioned future discovery in a learned Hamiltonian phase space, moving beyond curiosity based only on local novelty or single-step Hamiltonian change.

2. We introduce HTC², using horizon-conditioned temporal contrastive distance as the primary curiosity signal and bounded multi-step Hamiltonian variation as structural modulation of learned phase dynamics.

3. We provide comprehensive empirical validation on DeepMind Control Suite tasks with SAC and PPO against ICM, RIDE, E3B, Plan2Explore, and a single-step Hamiltonian-change variant, covering control performance, physical-state coverage, parametric OOD robustness, ablations, and Hamiltonian diagnostics.

## 2 RELATED WORK

Exploration in Reinforcement Learning. Exploration remains a central challenge in reinforcement learning, especially under sparse or delayed rewards (Sutton et al., 1998; Thrun, 1992; Mohamed et al., 2026). Intrinsic-motivation methods address this problem by rewarding novelty, uncertainty, or prediction difficulty (Houthooft et al., 2016; Ostrovski et al., 2017; Burda et al., 2018b; Pathak et al., 2017; Burda et al., 2018a; Lee et al., 2019; Guo et al., 2022). Count-based approaches favor rarely visited states (Bellemare et al., 2016; Tang et al., 2017; Ostrovski et al., 2017), but scale poorly to high-dimensional continuous domains. Prediction-based methods instead use learned errors or uncertainty: ICM relies on forward prediction error (Pathak et al., 2017), RND measures error against a fixed random target (Burda et al., 2018b), and related methods exploit representation change, model disagreement, information gain, or episodic uncertainty (Raileanu & Rocktäschel, 2020; Pathak et al., 2019; Houthooft et al., 2016; Henaff et al., 2022). Such local novelty signals can be affected by irrelevant stochasticity, as illustrated by the “noisy-TV"problem (Pathak et al., 2019; Mavor-Parker et al., 2022). More recently, model-based exploration has leveraged learned world models to plan toward informative future states and improve sample efficiency (Sekar et al., 2020). However, such models are primarily learned as predictive or generative dynamics models and do not explicitly impose Hamiltonian structure or related physics-informed priors on the latent dynamics.

World Models in Reinforcement Learning. World models learn predictive representations of environment dynamics and are widely used in model-based reinforcement learning (Ha & Schmidhuber, 2018b; Hafner et al., 2020; Lyu et al., 2025; Hafner et al., 2023; 2025; Xu et al., 2022; Liu et al., 2023; Ha & Schmidhuber, 2018a; Hafner et al., 2019b; Zhang et al., 2026). Latent world models compress high-dimensional observations and predict future latent states for planning, imagination, or policy learning (Hafner et al., 2019a; Assran et al., 2025; 2023). The Dreamer family uses recurrent latent dynamics with predictive reconstruction objectives (Hafner et al., 2020; 2023; 2025), whereas TD-MPC-style methods emphasize task-relevant latent prediction for efficient control (Hansen et al., 2024; 2022; Janner et al., 2019). Learned dynamics are also used for intrinsic exploration through prediction error, inverse dynamics, and uncertainty (Pathak et al., 2017; 2019; Raileanu & Rocktäschel, 2020; Henaff et al., 2022). However, standard world models mainly capture statistical regularities and need not explicitly encode physical structure (Hafner et al., 2020; 2023; 2025; 2019a; Assran et al., 2025; 2023), nor does predictive modeling alone specify which future interactions are useful for exploration (Sekar et al., 2020; Tang et al., 2026). HTC2 uses a Hamiltonian phase representation as the space for measuring temporal future relations, turning the world model into a structured substrate for intrinsic exploration.

Hamiltonian Dynamics for World Models. Physics-structured world models introduce inductive biases from classical mechanics to improve dynamics learning and generalization (Greydanus et al., 2019; Cranmer et al., 2020; Zhong et al., 2019; Toth et al., 2019; Arzani et al., 2025; Troch et al., 2025). Hamiltonian neural networks and symplectic models represent dynamics through coordinatelike and momentum-like variables governed by gradients of a learned Hamiltonian (Toth et al., 2019; Greydanus et al., 2019; Zhong et al., 2019). Recent work extends these priors to reinforcement learning and pixel-based world models (Schiewer et al., 2024; Deng et al., 2025; Tang et al., 2026). Their use for exploration, however, remains less developed; directly rewarding learned energy change can favor local perturbations rather than temporally extended discovery (Tang et al., 2026). HTC² instead uses Hamiltonian variation only as a bounded modulation, while the primary exploration signal is the temporal contrastive distance between current phase state-action pairs and their multi-step futures.

## 3 METHOD

We propose Hamiltonian Temporal Contrastive Curiosity $( \mathbf { H T C } ^ { 2 } )$ , an intrinsic exploration method for reinforcement learning with Hamiltonian latent world models. $\mathrm { H T C ^ { 2 } }$ encourages the agent to discover future phase states that remain difficult to organize under the current temporal representation while using learned Hamiltonian variation to provide a bounded physics-guided modulation. As illustrated in Figure 1, $\mathrm { H T C ^ { 2 } }$ consists of a Hamiltonian phase-space world model, a horizon-conditioned temporal contrastive critic, and a temporal discovery reward with latent RND warm start.

## 3.1 PRELIMINARIES

Controlled Hamiltonian Dynamics. We consider latent dynamics with a phase state ${ \boldsymbol { z } } _ { t } = ( q _ { t } , p _ { t } )$ where $q _ { t }$ and $p _ { t }$ are learned coordinate-like and momentum-like latent variables (Toth et al., 2019). A controlled Hamiltonian formulation separates the internal latent dynamics from the effect of external actions (Zhong et al., 2019). Given a learned Hamiltonian $H _ { \varphi } ( q , p )$ and a learned control input mapping $g _ { \varphi } ( q )$ , the continuous-time dynamics are

$$
\frac { d q } { d t } = \frac { \partial H _ { \varphi } ( q , p ) } { \partial p } , \qquad \frac { d p } { d t } = - \frac { \partial H _ { \varphi } ( q , p ) } { \partial q } + g _ { \varphi } ( q ) a _ { t } .\tag{1}
$$

The first component describes the internal Hamiltonian flow, while $g _ { \varphi } ( q ) a _ { t }$ represents the actiondependent control contribution to the momentum-like dynamics.

For an autonomous Hamiltonian system without external control, $a _ { t } = 0 .$ , the Hamiltonian is constant along the continuous trajectory. With the control term in Equation 1, its time derivative becomes

$$
\frac { d H _ { \varphi } } { d t } = \nabla _ { p } H _ { \varphi } ( q , p ) ^ { \top } g _ { \varphi } ( q ) a _ { t } .\tag{2}
$$

Thus, Hamiltonian variation separates the effect of external control from the internally conservative Hamiltonian flow (Toth et al., 2019). $\mathrm { H T C ^ { 2 } }$ does not directly maximize this variation; instead, it uses the corresponding multi-step quantity as a bounded modulation of temporal future discovery.

Finite-Horizon Future Distribution. Temporal contrastive learning requires a distribution over future phase states reached from the current state-action pair (Ho & Ermon, 2016; Torabi et al., 2018; Eysenbach et al., 2020). Let $K = \{ 1 , \ldots , K \}$ denote the set of admissible future horizons. We define the finite-horizon future distribution as

$$
p _ { \mathcal K } ( z _ { f } \mid z _ { t } , a _ { t } ) = \sum _ { k = 1 } ^ { K } \omega _ { k } p _ { \mathcal T } \left( z _ { t + k } = z _ { f } \mid z _ { t } , a _ { t } \right) ,\tag{3}
$$

where $p \tau$ denotes the trajectory distribution and $\omega _ { k }$ is the probability of sampling horizon k.

$\mathrm { H T C ^ { 2 } }$ uses a truncated geometric horizon distribution,

$$
\omega _ { k } = \frac { ( 1 - \gamma _ { \mathrm { T } } ) \gamma _ { \mathrm { T } } ^ { k - 1 } } { 1 - \gamma _ { \mathrm { T } } ^ { K } } , \qquad k \in \mathcal { K } ,\tag{4}
$$

which satisfies $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \omega _ { k } = 1 } \end{array}$ . The parameter $\gamma _ { \mathrm { T } } \in ( 0 , 1 )$ controls preference for nearer futures while retaining explicitly multi-step positive pairs. Each pair is constrained to lie within the same episode.

Hamiltonian Phase-Space World Model. $\mathrm { H T C ^ { 2 } }$ uses a recurrent Hamiltonian latent world model to construct the phase representation on which temporal discovery is performed. Given an observation $x _ { t }$ and recurrent deterministic state $h _ { t } .$ the posterior model infers

$$
q _ { \varphi } ( z _ { t } \mid h _ { t } , x _ { t } ) , \qquad z _ { t } = ( q _ { t } , p _ { t } ) .\tag{5}
$$

The stochastic phase state is divided into coordinate-like and momentum-like components. In implementation, both components are represented by categorical stochastic variables; their posterior expectations are used whenever continuous phase values are required by the Hamiltonian transition or the temporal critic.

The deterministic state is updated recurrently from the previous phase state and action. The controlled Hamiltonian dynamics in Equation 1 then provide a physics-structured phase prediction, which is used together with the recurrent context to parameterize the next stochastic prior:

$$
p _ { \varphi } \left( z _ { t + 1 } \mid h _ { t + 1 } , z _ { t } , a _ { t } \right) .\tag{6}
$$

The Hamiltonian transition is integrated using a leapfrog step, while a learned residual prior retains sufficient flexibility for high-dimensional visual control. Detailed transition equations are given in Appendix B.

The world model is optimized with

$$
{ \mathcal { L } } _ { \mathrm { W M } } = { \mathcal { L } } _ { \mathrm { r e c } } + \alpha { \mathcal { L } } _ { \mathrm { d y n } } + ( 1 - \alpha ) { \mathcal { L } } _ { \mathrm { r e p } } + { \mathcal { L } } _ { \mathrm { i n v } } + \lambda _ { \mathrm { p h y s } } { \mathcal { L } } _ { \mathrm { p h y s } } .\tag{7}
$$

Here, $\mathcal { L } _ { \mathrm { r e c } }$ grounds the posterior latent state in the observation space; $\mathcal { L } _ { \mathrm { d y n } }$ and $\mathcal { L } _ { \mathrm { r e p } }$ provide balanced prior-posterior regularization; ${ \mathcal { L } } _ { \mathrm { i n v } }$ preserves action-relevant information; and $\mathcal { L } _ { \mathrm { p h y s } }$ aligns the Hamiltonian prediction with the posterior phase transition.

For $\mathrm { H T C ^ { 2 } }$ , the world model provides posterior phase trajectories $( z _ { t } , z _ { t + 1 } , \ldots )$ for temporal pair construction and the learned Hamiltonian $H _ { \varphi } ( z _ { t } )$ for evaluating multi-step Hamiltonian variation. Temporal positives are formed from actually reached same-episode posterior futures.

## 3.2 HAMILTONIAN TEMPORAL CONTRASTIVE CURIOSITY

$\mathrm { H T C ^ { 2 } }$ uses horizon-conditioned temporal contrastive mismatch as its primary curiosity signal and bounded multi-step Hamiltonian variation to modulate this signal.

Horizon-Conditioned Temporal Contrastive Critic. Contrastive representation learning brings positive pairs closer while separating them from negative samples (Chopra et al., 2005; Oord et al., 2018; Wang & Isola, 2020; Saunshi et al., 2019; Hassani & Khasahmadi, 2020). In $\mathrm { H T C ^ { 2 } }$ , the anchor is $\left( z _ { t } , a _ { t } , k \right)$ , the reached same-episode future $z _ { t + k }$ forms the positive pair, and other minibatch futures serve as negatives. The horizon $k \sim \omega _ { k }$ is sampled according to Equation 4, enabling the critic to learn future relations over multiple temporal scales.

The critic consists of an anchor encoder and a future encoder:

$$
u _ { t , k } = \phi _ { \theta } ( z _ { t } , a _ { t } , e _ { k } ) , \qquad v _ { t + k } = \psi _ { \theta } ( z _ { t + k } ) .\tag{8}
$$

where $e _ { k }$ is a learnable horizon embedding

The temporal score is defined by the negative representation distance:

$$
C _ { \theta } ( ( z _ { t } , a _ { t } , k ) , z _ { f } ) = - \left\| \phi _ { \theta } ( z _ { t } , a _ { t } , e _ { k } ) - \psi _ { \theta } ( z _ { f } ) \right\| _ { 2 } .\tag{9}
$$

Given a minibatch of B same-episode positive pairs, we optimize the temporal critic with the InfoNCE objective (Oord et al., 2018):

$$
\mathcal { L } _ { \mathrm { T C } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( C _ { \theta } ( ( z _ { t _ { i } } ^ { i } , a _ { t _ { i } } ^ { i } , k _ { i } ) , z _ { t _ { i } + k _ { i } } ^ { i } ) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( C _ { \theta } ( ( z _ { t _ { i } } ^ { i } , a _ { t _ { i } } ^ { i } , k _ { i } ) , z _ { t _ { j } + k _ { j } } ^ { j } ) / \tau \right) } .\tag{10}
$$

The matched future forms the positive pair, whereas the remaining minibatch futures act as negatives.   
The temperature τ controls the sharpness of the contrastive classification.

The policy does not directly optimize $\mathcal { L } _ { \mathrm { T C } }$ . Instead, $\mathrm { H T C ^ { 2 } }$ extracts the positive-pair distance

$$
D _ { \theta } ( t , k ) = \| \phi _ { \theta } ( z _ { t } , a _ { t } , e _ { k } ) - \psi _ { \theta } ( z _ { t + k } ) \| _ { 2 } .\tag{11}
$$

A large $D _ { \theta } ( t , k )$ indicates that the reached future remains poorly aligned under the current temporal representation. As these transitions are revisited, the temporal critic learns to better organize them through Equation 10, making the temporal mismatch an adaptive exploration signal.

Hamiltonian-Modulated Temporal Curiosity. For the same sampled future horizon $k , \mathrm { H T C ^ { 2 } }$ evaluates the multi-step Hamiltonian variation

$$
\Delta H _ { t , k } = \left| H _ { \varphi } ( z _ { t + k } ) - H _ { \varphi } ( z _ { t } ) \right| .\tag{12}
$$

Rather than directly maximizing $\Delta H _ { t , k } , \mathrm { H T C ^ { 2 } }$ converts it into the bounded modulation

$$
G _ { H } ( t , k ) = 1 + \alpha _ { H } \operatorname { t a n h } \left( \operatorname { c l i p } \left( \frac { \Delta H _ { t , k } } { \sigma _ { H } + \epsilon } , 0 , c _ { H } \right) \right) ,\tag{13}
$$

where $\sigma _ { H }$ denotes a running scale estimate, € provides numerical stability, αH controls the modulation strength, and $c _ { H }$ is the clipping threshold. Since the baseline value of the gate is one and its amplification is bounded, Hamiltonian variation reweights rather than replaces temporal discovery.

The raw $\mathrm { H T C ^ { 2 } }$ reward is

$$
r _ { \mathrm { H T C ^ { 2 } } } ^ { \mathrm { r a w } } ( t ) = D _ { \theta } ( t , k ) G _ { H } ( t , k ) - \lambda _ { \mathrm { s m } } \left\| a _ { t } - a _ { t - 1 } \right\| _ { 2 } ^ { 2 } ,\tag{14}
$$

where $\lambda _ { \mathrm { s m } }$ controls an action-smoothness penalty. The reward is normalized before being combined with the cold-start signal.

Early in training, both the phase representation and the temporal critic are still being learned. We therefore use normalized latent RND, $\bar { r } _ { t } ^ { \mathrm { R N D } }$ , as a warm-start exploration signal. Let n denote the global environment step. The transition weight is

$$
w _ { n } = \mathrm { c l i p } \left( { \frac { n - T _ { \mathrm { w a r m } } } { T _ { \mathrm { a n n e a l } } } } , 0 , 1 \right) .\tag{15}
$$

The final intrinsic reward is

$$
r _ { t } ^ { \mathrm { i n t } } = ( 1 - w _ { n } ) \bar { r } _ { t } ^ { \mathrm { R N D } } + w _ { n } \bar { r } _ { t } ^ { \mathrm { H T C ^ { 2 } } } ,\tag{16}
$$

where $\bar { r } _ { t } ^ { \mathrm { H T C ^ { 2 } } }$ denotes the normalized $\mathrm { H T C ^ { 2 } }$ signal. The reward used by the policy optimizer is therefore

$$
\widetilde { r } _ { t } = r _ { t } ^ { \mathrm { e x t } } + \beta r _ { t } ^ { \mathrm { i n t } } ,\tag{17}
$$

where $\beta$ is the global intrinsic-reward scale. Optimizer-specific training procedures for SAC and PPO are provided in Appendix B.6.

## 4 EXPERIMENTS

We evaluate $\mathrm { H T C ^ { 2 } }$ on the DeepMind Control Suite (DMC) (Tunyasuvunakool et al., 2020) through five complementary studies. First, we compare control performance under SAC and PPO to assess whether $\mathrm { \bar { H } T C ^ { 2 } }$ improves exploration across policy optimizers. Second, we measure physical-state coverage to examine how broadly each method explores the underlying state space. Third, we evaluate zero-shot retention under parametric dynamics shifts to test policy robustness. Fourth, component ablations isolate the contribution of the main $\mathrm { H T C ^ { 2 } }$ design choices. Finally, Hamiltonian diagnostics examine whether the learned phase representation exhibits the intended energy and latent structure. Additional results and diagnostics are reported in Appendix C.

## 4.1 EXPERIMENTAL SETUP

Benchmarks, baselines, and evaluation. We evaluate $\mathrm { H T C ^ { 2 } }$ on six tasks: walker\_walk, walker\_run, walker\_stand, hopper\_hop, hopper\_stand, and cheetah\_run. HTC² is evaluated with both SAC (Haarnoja et al., 2018) and PPO (Schulman et al., 2017), and compared with four representative exploration methods: ICM (Pathak et al., 2017), RIDE (Raileanu & Rocktäschel, 2020), E3B (Henaff et al., 2022), and Plan2Explore (Sekar et al., 2020). Within each policy optimizer, all methods use the same training budget and evaluation settings. Results are reported as mean ± standard deviation, with the number of random seeds specified for each experiment. Detailed numerical results and complete learning curves are provided in Appendix C.1.

## 4.2 CONTROL PERFORMANCE ON DMC

Figure 2 shows that $\mathrm { H T C ^ { 2 } }$ maintains strong control performance under both SAC and PPO in comparison with ICM, RIDE, E3B, and Plan2Explore. Across the twelve tasks, $\mathrm { H T C ^ { 2 } }$ achieves the highest mean final return in eleven settings. Under SAC, HTC² ranks first on five of the six tasks and performs nearly identically to the strongest baseline on wal ker\_st and; under PPO, it achieves the highest mean return on all six tasks. The consistency across the two optimization regimes indicates that the performance of $\mathrm { H T C ^ { 2 } }$ is not specific to either off-policy replay or on-policy updates.

![](images/9a97d7064a55baf2c79ea56e631fd43eb9b846e69ed7bfdbe06f9d54c0b82b84.jpg)  
Figure 2: Control performance of $\mathbf { H T C ^ { 2 } }$ on six DMC tasks with SAC and PPO. The figure summarizes episode-return performance across five random seeds for ICM, RIDE, E3B, Plan2Explore, and $\mathrm { H T C ^ { 2 } }$ . Numerical results are reported as mean ± standard deviation in Appendix C.1.

The gains are particularly visible on tasks requiring coordinated multi-step transitions, such as walker\_walk and hopper\_hop. This trend is consistent with the design of $\mathrm { H T C ^ { 2 } } ;$ exploration is driven by the mismatch between the current state-action representation and its reached multi-step future, while Hamiltonian variation only provides a bounded modulation of this temporal signal Detailed per-task returns and complete SAC/PPO learning curves are reported in Appendix C.1.

## 4.3 PHYSICAL-STATE EXPLORATION COVERAGE

We evaluate physical-state coverage on three DMC tasks using low-dimensional privileged simulator states, which are used only for evaluation and are not provided to the policy or intrinsic-reward module. For environment $\mathcal { E } ,$ let $s _ { t } ^ { \mathrm { p h y } }$ denote the task-specific physical state and $b \varepsilon ( \cdot )$ its fixed grid discretization. The cumulative coverage at training step t is

$$
\mathcal { C } _ { t } = \left| \left\{ b _ { \mathcal { E } } ( s _ { \tau } ^ { \mathrm { p h y } } ) ~ : ~ 0 \leq \tau \leq t \right\} \right| ,\tag{18}
$$

i.e., the number of unique discretized physical states visited.

For cartpole\_swingup, we use the cart position and wrapped pole angle, $( x _ { \mathrm { c a r t } } , \theta _ { \mathrm { p o l e } } )$ . For bal1\_in\_cup\_cat ch, we use the ball position relative to the cup, $( x _ { \mathrm { b a l l } } - x _ { \mathrm { c u p } } , z _ { \mathrm { b a l l } } - z _ { \mathrm { c u p } } )$ For $\mathtt { f i n g e r \_ s p i n }$ , we use the wrapped object angle and clipped angular velocity, $( \theta _ { \mathrm { o b j } } , \dot { \theta } _ { \mathrm { o b j } } )$ Discretization details are provided in Appendix C.2.

Figure 3 shows distinct coverage patterns across the three environments. On cartpole\_swingup, HTC² continues expanding its visited-state set after the other methods begin to saturate, reaching the highest final coverage and mean episode reward. On bal1\_in\_cup\_cat ch, RIDE attains the highest coverage, while $\mathrm { H T C ^ { 2 } }$ ranks second in coverage AUC, final reward, and reward AUC. On finger\_spin, $\mathrm { H T C ^ { 2 } }$ achieves the highest final coverage and final reward, with a coverage AUC close to E3B.

Overall, $\mathrm { H T C ^ { 2 } }$ exhibits strong physical-state coverage on cartpole\_swingup and $\mathtt { f i n g e r \_ s p i n }$ , while remaining competitive on bal $\cdot 1 \_ \mathrm { i n \_ c u p \_ c a t c h }$ . Full coverage and reward statistics are reported in Appendix C.2.

## 4.4 HAMILTONIAN MECHANISM VALIDATION

Beyond policy learning and exploration coverage, we examine whether the learned Hamiltonian phase representation exhibits the structure used by $\mathrm { H T C ^ { 2 } }$ . Specifically, we test whether $H _ { \varphi } ( q , p )$ remains stable under passive evolution, responds to policy-induced control, and varies coherently over the coordinate-like and momentum-like latent subspaces.

![](images/e5ec5a4115764b7cae7790e90a4bc38c0a16f4f81b7b991f127ee334f17022e2.jpg)  
Figure 3: Physical-state coverage and episode reward during training. Top: cumulative unique discretized states; bottom: episode reward on cartpole\_swingup, ball\_in\_cup\_catch, and finger\_spin. Curves show mean ± one standard deviation over four random seeds.  
Hamiltonian Energy Dynamics and Phase-Subspace Geometry

![](images/134241faa4f976d3d9e3aa83d42307d8f7a4fc5d2e62e213cda95c3651ed3011.jpg)

![](images/e116f1892cffb99bdc8cb24e5882e4a5edc3444b6eae7176e82a848709d7876b.jpg)

![](images/edb2369f1443af15c2b08314c682106854b595d07af439c6b2affdfdae60c7ff.jpg)  
Figure 4: Hamiltonian energy dynamics and phase-subspace geometry. A: open-loop learned Hamiltonian evolution under zero, random, and policy actions. B-C: PCA projections of the coordinate-like latent q and momentum-like latent p, colored by $H _ { \varphi } ( q , p )$

(A) Controlled Hamiltonian energy response. Figure 4A compares the learned Hamiltonian under three action regimes. Under zero action, $H _ { \varphi } ( q , p )$ remains nearly flat over the rollout horizon. consistent with approximate energy preservation under autonomous Hamiltonian evolution. Random actions produce similarly bounded deviations around the zero-action trajectory. In contrast, policy actions induce a clear and sustained energy displacement with greater long-horizon variability, showing that the learned Hamiltonian responds differently to structured control.

(B) Energy organization in the q subspace. Figure 4B projects the coordinate-like latent q into two principal components and colors each state by $\bar { H _ { \varphi } ( q , p ) }$ . The projected trajectory exhibits a clear energy-ordered structure: high-energy states are concentrated near one end of the curve, while lower-energy states occupy a different region of the projected subspace. This smooth variation indicates that the learned Hamiltonian is systematically coupled to the q component of the phase representation. Therefore, the energy term used by $\mathrm { H } \mathrm { { \bar { I } C ^ { 2 } } }$ is computed over an organized phase subspace rather than over an unstructured latent cloud.

(C) Energy organization in the $p$ subspace. Figure 4C shows the corresponding projection for the momentum-like latent $p .$ The learned energy again varies smoothly across the projected trajectory, with a compact low-energy region and a separate high-energy branch. This indicates that $H _ { \varphi } ( q , p )$ depends on both components of the learned phase state. Together, the $q$ and $p$ visualizations support the design of $\mathrm { \dot { H } T C ^ { 2 } } ;$ temporal future discovery is performed in a Hamiltonian phase space whose energy varies coherently over the learned latent dynamics, and whose controlled evolution is modulated by the policy actions.

![](images/3fc733b7385511dc73f2fccdb2e1792ec321b89650707b0dc6fca3fed15b6c61.jpg)  
Figure 5: Zero-shot policy retention under parametric physical shifts. Bars and error bars show mean ± standard deviation over four random seeds. Higher values indicate greater return retention.

We further analyze the relation between the q and p subspaces in Appendix C.3. The dimensionwise coupling maps show that q and p are dynamically related but not uniformly coupled: only selected cross-dimensional pairs exhibit strong correlations. This supports a non-collapsed phase representation in which coordinate-like and momentum-like variables share the same trajectory dynamics while preserving distinct representational roles.

## 4.5 PARAMETRIC OOD ROBUSTNESS

We further evaluate the zero-shot robustness of SAC policies trained with $\mathrm { H T C ^ { 2 } }$ under parametric dynamics shifts. The final policies trained on the original DMC environments are evaluated on variants of walker\_walk, hopper\_stand, and cheetah\_run with modified gravity $( 1 . 5 \times )$ friction (2.0×), mass (1.5×), or damping (2.0×). We report return retention, defined as the OOD return divided by the corresponding in-distribution return for the same method and seed.

Figure 5 shows that $\mathrm { H T C ^ { 2 } }$ maintains high retention under several physical shifts. On walker\_walk, $\mathrm { H } \mathrm { { \bar { T } C } ^ { 2 } }$ achieves the highest retention under gravity, friction, and mass perturbations, with an average retention of 89.7% across the four variants. On hopper\_st and, it performs strongly under the mass shift and remains competitive under friction and damping, while RIDE obtains the highest retention under the gravity shift. On cheet ah\_run, $\mathrm { { H T C ^ { \frac { \cdot } { 2 } } } }$ achieves the highest retention under gravity and mass perturbations, with mass-shift retention exceeding 100%.

Across all three tasks and four parameter shifts, $\mathrm { H T C ^ { 2 } }$ obtains the highest overall average retention (82.3%), compared with RIDE (80.6%), E3B (72.5%), and ICM (69.4%). Detailed per-shift results are reported in Appendix C.4.

## 4.6 ABLATION STUDY

We ablate the main components of $\mathrm { H T C ^ { 2 } }$ on walker\_walk, hopper\_hop, and cheetah\_run. Specifically, w/o $\mathcal { L } _ { \mathrm { p h y s } }$ removes the physical-consistency loss; w/o $G _ { H }$ removes the Hamiltonian modulation; one-step $\dot { \Delta } H$ replaces the temporal contrastive reward with one-step Hamiltonian change; w/o RND removes the warm-start schedule; and w/o repr. norm disables contrastive representation normalization. All variants use SAC with the same training budget and are evaluated over four random seeds.

Table 1 shows that the full $\mathrm { H T C ^ { 2 } }$ configuration achieves the highest mean final return on all three tasks. Removing $\mathcal { L } _ { \mathrm { p h y s } }$ or $G _ { H }$ lowers the mean return across all environments, with larger reductions from removing $\mathcal { L } _ { \mathrm { p h y s } }$ on walker\_walk and cheetah\_run. Replacing the temporal contrastive reward with one-step $\Delta H$ produces the largest reduction on hopper\_hop, where the final return drops to $1 . 8 4 3 \pm 1 . 1 8 3$

Removing the RND warm start or representation normalization also reduces performance across all three tasks. The reduction from removing representation normalization is particularly pronounced on hopper\_hop, whereas removing the RND warm start produces a larger absolute decrease on

Table 1: $\mathbf { H T C ^ { 2 } }$ ablations. Final return $( { \mathrm { m e a n } } \pm { \mathrm { s t d } } ,$ four seeds).
<table><tr><td>Variant</td><td>Walker</td><td>Hopper</td><td>Cheetah</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { p h y s } }$ </td><td> $6 9 3 . 1 6 6 \pm 2 0 3 . 9 6 7$ </td><td> $2 4 5 . 4 1 6 \pm 1 4 . 6 5 1$ </td><td> $5 6 6 . 2 0 9 \pm 1 5 9 . 8 7 1$ </td></tr><tr><td>w/o  $G _ { H }$ </td><td> $7 1 8 . 9 4 1 \pm 1 6 . 1 2 2$ </td><td> $2 3 5 . 2 0 3 \pm 1 . 8 9 6$ </td><td> $6 0 9 . 1 0 8 \pm 1 6 . 1 8 5$ </td></tr><tr><td>one-step  $\Delta H$ </td><td> $7 6 1 . 0 5 1 \pm 1 9 5 . 3 2 4$ </td><td> $1 . 8 4 3 \pm 1 . 1 8 3$ </td><td> $5 1 0 . 0 9 3 \pm 1 2 8 . 2 6 6$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  RND</td><td> $7 8 5 . 0 6 9 \pm 9 3 . 8 0 9$ </td><td> $2 2 7 . 7 7 4 \pm 0 . 7 1 9$ </td><td> $5 7 8 . 8 1 6 \pm 9 . 9 1 0$ </td></tr><tr><td>w/o repr. norm</td><td> $7 5 7 . 7 5 8 \pm 1 1 4 . 1 1 7$ </td><td> $1 4 4 . 4 6 6 \pm 8 5 . 1 6 9$ </td><td> $5 9 4 . 6 9 7 \pm 1 6 . 8 7 6$ </td></tr><tr><td> $\mathbf { H T C } ^ { 2 }$  full</td><td> $\mathbf { 8 1 4 . 3 4 2 \pm 1 6 2 . 8 5 7 }$ </td><td> $\mathbf { 2 5 0 . 8 6 5 \pm 9 . 5 9 8 }$ </td><td> $\mathbf { 6 5 9 . 8 1 0 \pm 3 3 . 5 2 5 }$ </td></tr></table>

cheet ah\_run. Across the three tasks, the complete configuration yields the strongest mean final performance. Full learning curves are reported in Appendix C.5.

## 5 CONCLUSION

We presented Hamiltonian Temporal Contrastive Curiosity $( { \bf H T C } ^ { 2 } )$ , an intrinsic exploration method for discovering temporally informative futures in a learned Hamiltonian phase space. $\mathrm { H T C ^ { 2 } }$ uses a horizon-conditioned temporal contrastive critic to identify poorly organized multi-step futures and modulates this signal with bounded Hamiltonian variation to emphasize structured changes in the learned phase dynamics. Across DeepMind Control Suite tasks, $\mathrm { H T C ^ { 2 } }$ shows strong control performance with both SAC and PPO, competitive physical-state coverage, and robust retention under parametric dynamics shifts. Ablations show performance degradation when removing physical consistency, Hamiltonian modulation, warm start, or representation normalization, while the singlestep Hamiltonian-change reward performs substantially worse on hopper\_hop. Hamiltonian diagnostics further reveal energy-organized phase representations with distinct responses to policyinduced control. Overall, these results characterize $\mathrm { H T C ^ { 2 } }$ as a physics-structured approach to long-horizon exploration in physical control.

## AI USE STATEMENT

During the preparation of this manuscript, the authors used ChatGPT to assist with language polishing, including grammar correction and phrasing improvements. ChatGPT was not used for research ideation. The authors reviewed all AI-assisted text and take full responsibility for the final content of this work.

## REFERENCES

Amirhossein Arzani, Lingxiao Yuan, Pania Newell, and Bei Wang. Interpreting and generalizing deep learning in physics-based problems with functional linear models. Engineering with Computers, 41(1):135–157, 2025.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629. IEEE, 2023.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems, 29, 2016.

Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A Efros. Large-scale study of curiosity-driven learning. arXiv preprint arXiv:1808.04355, 2018a.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018b.

Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), volume 1, pp. 539–546. IEEE, 2005.

Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, and Shirley Ho. Lagrangian neural networks. arXiv preprint arXiv:2003.04630, 2020.

Congyue Deng, Brandon Y Feng, Cecilia Garraffo, Alan Garbarz, Robin Walters, William T Freeman, Leonidas Guibas, and Kaiming He. Denoising hamiltonian network for physical reasoning. arXiv preprint arXiv:2503.07596, 2025.

Benjamin Eysenbach, Ruslan Salakhutdinov, and Sergey Levine. C-learning: Learning to achieve goals via recursive classification. arXiv preprint arXiv:2011.08909, 2020.

Alejandro Castaneda Garcia, Jan Warchocki, Jan Van Gemert, Daan Brinks, and Nergis Tomen. Learning physics from video: Unsupervised physical parameter estimation for continuous dynamical systems. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27924–27933, 2025.

Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian neural networks. Advances in neural information processing systems, 32, 2019.

Zhaohan Guo, Shantanu Thakoor, Miruna Pîslar, Bernardo Avila Pires, Florent Altché, Corentin Tallec, Alaa Saade, Daniele Calandriello, Jean-Bastien Grill, Yunhao Tang, et al. Byol-explore: Exploration by bootstrapped prediction. Advances in neural information processing systems, 35:31855–31870, 2022.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018a.

David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3):440, 2018b.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. Pmlr, 2018.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019a.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555–2565. PMLR, 2019b.

Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, 2025.

Nick Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, volume 2024, pp. 47376–47405, 2024.

Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. arXiv preprint arXiv:2203.04955, 2022.

Kaveh Hassani and Amir Hosein Khasahmadi. Contrastive multi-view representation learning on graphs. In International conference on machine learning, pp. 4116–4126. PMLR, 2020.

Mikael Henaff, Roberta Raileanu, Minqi Jiang, and Tim Rocktäschel. Exploration via elliptical episodic bonuses. Advances in Neural Information Processing Systems, 35:37631–37646, 2022.

Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.

Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. Vime: Variational information maximizing exploration. Advances in neural information processing systems, 29, 2016.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. Advances in neural information processing systems, 32, 2019.

Lisa Lee, Benjamin Eysenbach, Emilio Parisotto, Eric Xing, Sergey Levine, and Ruslan Salakhutdinov. Efficient exploration via state marginal matching. arXiv preprint arXiv:1906.05274, 2019.

Yuren Liu, Biwei Huang, Zhengmao Zhu, Honglong Tian, Mingming Gong, Yang Yu, and Kun Zhang. Learning world models with identifiable factorization. Advances in Neural Information Processing Systems, 36: 31831–31864, 2023.

Jiangran Lyu, Ziming Li, Xuesong Shi, Chaoyi Xu, Yizhou Wang, and He Wang. Dywa: Dynamics-adaptive world action model for generalizable non-prehensile manipulation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11058–11068. IEEE, 2025.

Augustine Mavor-Parker, Kimberly Young, Caswell Barry, and Lewis Griffin. How to stay curious while avoiding noisy tvs using aleatoric uncertainty estimation. In International conference on machine learning, pp. 15220–15240. PMLR, 2022.

Faisal Mohamed, Catherine Ji, Benjamin Eysenbach, and Glen Berseth. Temporal representations for exploration: Learning complex exploratory behavior without extrinsic rewards. arXiv preprint arXiv:2603.02008, 2026.

Kohei Noda, Araki Wakiuchi, Yoshihiro Hayashi, and Ryo Yoshida. Advancing extrapolative predictions of material properties through learning to learn using extrapolative episodic training. Communications Materials, 6(1):36, 2025.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Georg Ostrovski, Marc G Bellemare, Aäron Oord, and Rémi Munos. Count-based exploration with neural density models. In International conference on machine learning, pp. 2721–2730. PMLR, 2017.

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by selfsupervised prediction. In International conference on machine learning, pp. 2778–2787. PMLR, 2017.

Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Self-supervised exploration via disagreement. In International conference on machine learning, pp. 5062–5071. PMLR, 2019.

Roberta Raileanu and Tim Rocktäschel. Ride: Rewarding impact-driven exploration for procedurally-generated environments. arXiv preprint arXiv:2002.12292, 2020.

Nikunj Saunshi, Orestis Plevrakis, Sanjeev Arora, Mikhail Khodak, and Hrishikesh Khandeparkar. A theoretical analysis of contrastive unsupervised representation learning. In International conference on machine learning, pp. 5628–5637. PMLR, 2019.

Robin Schiewer, Anand Subramoney, and Laurenz Wiskott. Exploring the limits of hierarchical world models in reinforcement learning. Scientific Reports, 14(1):26856, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak. Planning to explore via self-supervised world models. In International conference on machine learning, pp. 8583–8592. PMLR, 2020.

Bradly C Stadie, Sergey Levine, and Pieter Abbeel. Incentivizing exploration in reinforcement learning with deep predictive models. arXiv preprint arXiv:1507.00814, 2015.

Richard S Sutton, Andrew G Barto, and Andrew Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.

Jinzhou Tang, Fan Feng, Minghao Fu, Wenjun Lin, Jing Yang, Biwei Huang, and Keze Wang. Dreamsac: Learning hamiltonian world models via symmetry exploration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15188–15198, 2026.

Sebastian B Thrun. Efficient exploration in reinforcement learning. Carnegie Mellon University, 1992.

Faraz Torabi, Garrett Warnell, and Peter Stone. Generative adversarial imitation from observation. arXiv preprint arXiv:1807.06158, 2018.

Peter Toth, Danilo Jimenez Rezende, Andrew Jaegle, Sébastien Racanière, Aleksandar Botev, and Irina Higgins. Hamiltonian generative networks. arXiv preprint arXiv:1909.13789, 2019.

Arne Troch, Kevin Mets, and Siegfried Mercelis. Action-conditioned hamiltonian generative networks (ac-hgn) for supervised and reinforcement learning. In 7th Annual Learning for Dynamics & Control Conference, 04-06 June, 2025, Ann Arbor, Michigan, USA, pp. 310–322, 2025.

Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel, Tom Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa. dm\_control: Software and tasks for continuous control. Software Impacts, 6:100022, 2020.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pp. 9929–9939. PMLR, 2020.

Yingchen Xu, Jack Parker-Holder, Aldo Pacchiano, Philip Ball, Oleh Rybkin, Stephen Roberts, Tim Rocktäschel, and Edward Grefenstette. Learning general world models in a handful of reward-free deployments. Advances in Neural Information Processing Systems, 35:26820–26838, 2022.

Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal Patel, Paul Liang, et al. World-in-world: World models in a closed-loop world. In International Conference on Learning Representations, volume 2026, pp. 55660–55699, 2026.

Yaofeng Desmond Zhong, Biswadip Dey, and Amit Chakraborty. Symplectic ode-net: Learning hamiltonian dynamics with control. arXiv preprint arXiv:1909.12077, 2019.

Alicja Ziarko, Michał Bortkiewicz, Michał Zawalski, Benjamin Eysenbach, and Piotr Miłoś. Contrastive representations for temporal reasoning. In The Thirty-ninth Annual Conference on Neural Information ProcessingSystems, 2025. URL https://openreview.net/forum?id=UlkcH5Ccrk.

## A DETAILED METHODOLOGY

This section provides additional derivations for the controlled Hamiltonian relations introduced in Equation 1 and 2, followed by the leapfrog discretization used by the Hamiltonian transition core. We then derive a continuous-time interpretation of the multi-step Hamiltonian variation defined in Equation 12.

## A.1 CONTROLLED HAMILTONIAN DYNAMICS AND ENERGY BALANCE

Autonomous Hamiltonian conservation. Starting from the controlled dynamics in Equation 1, consider first the autonomous case $a _ { t } = 0$ . For a Hamiltonian with no explicit time dependence, its derivative along the trajectory is

$$
\begin{array} { r l } & { \frac { d H _ { \varphi } } { d t } = \left( \frac { \partial H _ { \varphi } } { \partial q } \right) ^ { \top } \dot { q } + \left( \frac { \partial H _ { \varphi } } { \partial p } \right) ^ { \top } \dot { p } } \\ & { \qquad = \left( \frac { \partial H _ { \varphi } } { \partial q } \right) ^ { \top } \frac { \partial H _ { \varphi } } { \partial p } - \left( \frac { \partial H _ { \varphi } } { \partial p } \right) ^ { \top } \frac { \partial H _ { \varphi } } { \partial q } } \\ & { \qquad = 0 . } \end{array}\tag{19}
$$

This is the standard conservation relation of an autonomous Hamiltonian system. In classical mechanics, it is associated with time-translation invariance through Noether's theorem: in the absence of explicit time dependence and external forcing, the Hamiltonian remains constant along the continuous trajectory.

Energy balance under control. Restoring the action-dependent term in Equation 1 gives

$$
\begin{array} { c } { \displaystyle \frac { d H _ { \varphi } } { d t } = \left( \frac { \partial H _ { \varphi } } { \partial q } \right) ^ { \top } \frac { \partial H _ { \varphi } } { \partial p } } \\ { \displaystyle + \left( \frac { \partial H _ { \varphi } } { \partial p } \right) ^ { \top } \left( - \frac { \partial H _ { \varphi } } { \partial q } + g _ { \varphi } ( q ) a _ { t } \right) } \\ { \displaystyle = \left( \frac { \partial H _ { \varphi } } { \partial p } \right) ^ { \top } g _ { \varphi } ( q ) a _ { t } . } \end{array}\tag{20}
$$

Using $\dot { q } = \partial H _ { \varphi } / \partial p$ , the last term can equivalently be written as $\dot { q } ^ { \top } g _ { \varphi } ( q ) a _ { t }$ . Thus, Equation 20 recovers the controlled energy-balance relation in Equation 2.

## A.2 SYMPLECTIC INTEGRATION OF THE CONTROLLED DYNAMICS

The controlled continuous-time dynamics in Equation 1 must be discretized to construct the Hamiltonian transition prior. Our implementation uses a leapfrog integrator with step size $\Delta t = 0 . 1$ , with $a _ { t }$ held fixed over one integration step.

The update follows the kick-drift–kick structure. Starting from $\left( q _ { t } , p _ { t } \right)$ , the first momentum half-step is

$$
p _ { t + \frac { 1 } { 2 } } = p _ { t } - \frac { \Delta t } { 2 } \frac { \partial H _ { \varphi } } { \partial q } ( q _ { t } , p _ { t } ) + \frac { \Delta t } { 2 } g _ { \varphi } ( q _ { t } ) a _ { t } .\tag{21}
$$

The intermediate momentum-like state is then used to advance the coordinate-like variable over a full step:

$$
q _ { t + 1 } = q _ { t } + \Delta t \frac { \partial H _ { \varphi } } { \partial p } \left( q _ { t } , p _ { t + \frac { 1 } { 2 } } \right) .\tag{22}
$$

Finally, the momentum update is completed at the new position:

$$
p _ { t + 1 } = p _ { t + \frac { 1 } { 2 } } - \frac { \Delta t } { 2 } \frac { \partial H _ { \varphi } } { \partial q } \left( q _ { t + 1 } , p _ { t + \frac { 1 } { 2 } } \right) + \frac { \Delta t } { 2 } g _ { \varphi } ( q _ { t + 1 } ) a _ { t } .\tag{23}
$$

The three updates constitute the Hamiltonian physics step used in the world model. Gradients of $H _ { \varphi }$ with respect to q and p are evaluated through automatic differentiation, while the learned control term is incorporated into both momentum half-steps.

For the autonomous Hamiltonian component, leapfrog provides a symplectic discretization and is designed to preserve the geometric structure of Hamiltonian dynamics under repeated integration Symplectic integration does not imply exact conservation of $H _ { \varphi }$ at every discrete step. The complete world-model prior also contains the learned control contribution and a stochastic residual parameterization; accordingly, the leapfrog step is used as a structure-preserving inductive bias for the Hamiltonian transition core.

## A.3 TEMPORAL HAMILTONIAN VARIATION

The controlled energy-balance relation in Equation 2 provides a continuous-time interpretation of the multi-step Hamiltonian variation used by $\mathrm { \bar { H T C ^ { 2 } } }$ . Integrating it over the interval [t, t + τ] gives

$$
H _ { \varphi } ( z _ { t + \tau } ) - H _ { \varphi } ( z _ { t } ) = \int _ { t } ^ { t + \tau } \left( { \frac { \partial H _ { \varphi } } { \partial p } } ( z _ { s } ) \right) ^ { \top } g _ { \varphi } ( q _ { s } ) a _ { s } d s .\tag{24}
$$

For a sampled discrete horizon $k , { \mathrm { H T C ^ { 2 } } }$ uses the endpoint magnitude $\Delta H _ { t , k }$ defined in Equation 12. Hence, the multi-step term summarizes the Hamiltonian variation across the reached temporal interval rather than only an adjacent one-step change.

$\mathrm { H T C ^ { 2 } }$ does not optimize this quantity independently. The endpoint variation is transformed by the bounded gate in Equation 13 and modulates the temporal mismatch $D _ { \theta } ( t , k )$ defined in Equation 11. The resulting intrinsic signal is given in Equation 14. Thus, temporal future discovery remains the primary exploration signal, while Hamiltonian variation provides a bounded structural modulation over the same reached multi-step transition.

## B IMPLEMENTATION DETAILS

We provide implementation details of the Hamiltonian latent world model, its optimization objectives, the temporal contrastive critic, and the intrinsic-reward schedule used in $\mathrm { \ddot { H T C ^ { 2 } } }$ . Algorithms 1 and 2 summarize the integration of $\mathrm { H T C ^ { 2 } }$ with SAC and PPO, respectively.

## B.1 HAMILTONIAN LATENT WORLD MODEL

At time t, the recurrent latent state consists of a deterministic state $h _ { t }$ and a phase representation

$$
{ z } _ { t } = [ q _ { t } , p _ { t } ] ,\tag{25}
$$

where $q _ { t }$ and $p _ { t }$ denote coordinate-like and momentum-like latent variables. The deterministic state is updated by the recurrent core,

$$
h _ { t + 1 } = \mathrm { G R U } \left( F _ { \mathrm { r n n } } ( \left[ q _ { t } , p _ { t } , a _ { t } \right] ) , h _ { t } \right) ,\tag{26}
$$

where $F _ { \mathrm { r n n } }$ denotes the input projection preceding the GRU.

Let

$$
c _ { t } = E _ { \varphi } ( x _ { t } )\tag{27}
$$

denote the observation token produced by the visual encoder $E _ { \varphi }$ . The posterior network predicts categorical distributions over the two phase components:

$$
Q _ { q , t } = q _ { \varphi } ( q _ { t } \mid h _ { t } , c _ { t } ) ,\tag{28}
$$

$$
Q _ { p , t } = q _ { \varphi } ( p _ { t } \mid h _ { t } , c _ { t } ) .\tag{29}
$$

Their continuous posterior expectations are

$$
\mu _ { q , t } = \mathbb { E } _ { Q _ { q , t } } [ q _ { t } ] , \qquad \mu _ { p , t } = \mathbb { E } _ { Q _ { p , t } } [ p _ { t } ] .\tag{30}
$$

For compactness, $q _ { t }$ and $p _ { t }$ below denote these continuous phase values whenever the Hamiltonian transition or temporal critic is evaluated.

The Hamiltonian transition follows the controlled dynamics in Equation 1. The learned input mapping

$$
g _ { \varphi } ( q _ { t } ) \in \mathbb { R } ^ { d _ { p } \times d _ { a } }\tag{31}
$$

maps the action $a _ { t } \in \mathbb { R } ^ { d _ { a } }$ to the momentum-like control term $g _ { \varphi } ( q _ { t } ) a _ { t } \in \mathbb { R } ^ { d _ { p } }$ . Using the leapfrog updates in Equation 21 and 23, one deterministic Hamiltonian step is written compactly as

$$
\left( \bar { q } _ { t + 1 } , \bar { p } _ { t + 1 } \right) = \Phi _ { \Delta t } ^ { \mathrm { L F } } \left( q _ { t } , p _ { t } , a _ { t } ; H _ { \varphi } , g _ { \varphi } \right) ,\tag{32}
$$

where $\Phi _ { \Delta { t } } ^ { \mathrm { L F } }$ denotes one controlled leapfrog step and $\Delta t = 0 . 1$

The continuous prediction is subsequently mapped to categorical prior logits. We denote by $\mathcal { G } _ { \sigma _ { \ell } } ( \cdot )$ the deterministic Gaussian-shaped logit projection with width parameter $\sigma _ { \ell } .$ The prior logits are

$$
\ell _ { q , t + 1 } ^ { \mathrm { p r i o r } } = \mathcal G _ { \sigma _ { \ell } } ( \bar { q } _ { t + 1 } ) + R _ { q } ( \chi _ { t } ) ,\tag{33}
$$

$$
\ell _ { p , t + 1 } ^ { \mathrm { p r i o r } } = \mathcal G _ { \sigma _ { \ell } } ( \bar { p } _ { t + 1 } ) + R _ { p } ( \chi _ { t } ) ,\tag{34}
$$

where

$$
\begin{array} { r } { \chi _ { t } = [ h _ { t + 1 } , q _ { t } , p _ { t } , a _ { t } ] . } \end{array}\tag{35}
$$

For $\nu \in \{ q , p \}$ , the gated residual branch is

$$
R _ { \nu } ( \chi ) = s \sigma ( G _ { \nu } ( \chi ) ) \odot \operatorname { t a n h } ( F _ { \nu } ( \chi ) ) ,\tag{36}
$$

where $F _ { \nu }$ and $G _ { \nu }$ denote the residual-value and gate networks, respectively, $\sigma ( \cdot )$ is the sigmoid function,  denotes element-wise multiplication, and s is the residual scale. We use $s = 1 . 0$ The residual output is zero-initialized and the gate is initialized negatively so that the Hamiltonian prediction dominates early training.

The categorical prior distributions used below are obtained as

$$
P _ { q , t + 1 } = \mathrm { C a t } \left( \mathrm { s o f t m a x } ( \ell _ { q , t + 1 } ^ { \mathrm { p r i o r } } ) \right) ,\tag{37}
$$

$$
P _ { p , t + 1 } = \mathrm { C a t } \left( \mathrm { s o f t m a x } ( \ell _ { p , t + 1 } ^ { \mathrm { p r i o r } } ) \right) .\tag{38}
$$

## B.2 WORLD MODEL LOSS FUNCTIONS

The complete world-model objective is defined in Equation 7. We use $\alpha = 0 . 8$ and $\lambda _ { \mathrm { p h y s } } = 1 . 0$ This section specifies the individual loss terms used to optimize the posterior representation and the Hamiltonian dynamics prior.

Reconstruction Loss $\scriptstyle ( { \mathcal { L } } _ { \mathrm { r e c } } )$ . Define the posterior state

$$
s _ { t } ^ { \mathrm { p o s t } } = [ h _ { t } , \mu _ { q , t } , \mu _ { p , t } ] .\tag{39}
$$

The decoder $D _ { \varphi }$ reconstructs the next observation from the posterior state:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \mathbb { E } \left[ \left\| D _ { \varphi } ( s _ { t + 1 } ^ { \mathrm { p o s t } } ) - x _ { t + 1 } \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{40}
$$

Dynamics and Representation Losses $( \mathcal { L } _ { \mathrm { d y n } } , \mathcal { L } _ { \mathrm { r e p } } )$ . The dynamics loss updates the prior while stopping gradients through the posterior distributions:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d y n } } = \mathbb { E } \Big [ D _ { \mathrm { K L } } \left( \mathrm { s g } [ Q _ { q , t + 1 } ] \Vert P _ { q , t + 1 } \right) } \\ & { ~ + ~ D _ { \mathrm { K L } } \left( \mathrm { s g } [ Q _ { p , t + 1 } ] \Vert P _ { p , t + 1 } \right) \Big ] . } \end{array}\tag{41}
$$

Conversely, the representation loss updates the posterior while stopping gradients through the prior:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e p } } = \mathbb { E } \Big [ D _ { \mathrm { K L } } \left( Q _ { q , t + 1 } \| \mathrm { s g } [ P _ { q , t + 1 } ] \right) } \\ & { ~ + D _ { \mathrm { K L } } \left( Q _ { p , t + 1 } \| \mathrm { s g } [ P _ { p , t + 1 } ] \right) \Big ] . } \end{array}\tag{42}
$$

Here, $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operator. In the implementation, the $q -$ and $p { \mathrm { - K I } }$ , terms are summed before applying standard free-nats stabilization and batch averaging.

Inverse-Dynamics Loss $( \mathcal { L } _ { \mathrm { i n v } } )$ . The inverse-dynamics model predicts the executed action from adjacent posterior states,

$$
\begin{array} { r } { \hat { a } _ { t } = F _ { \mathrm { i n v } } \left( [ s _ { t } ^ { \mathrm { p o s t } } , s _ { t + 1 } ^ { \mathrm { p o s t } } ] \right) , } \end{array}\tag{43}
$$

and is optimized by

$$
\mathcal { L } _ { \mathrm { i n v } } = \mathbb { E } \left[ \| \hat { \boldsymbol { a } } _ { t } - \boldsymbol { a } _ { t } \| _ { 2 } ^ { 2 } \right] .\tag{44}
$$

Hamiltonian Consistency Loss $( \mathcal { L } _ { \mathrm { p h y s } } )$ . The continuous Hamiltonian prediction from Equation 32 is aligned with the posterior phase expectation:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p h y s } } = \mathbb { E } \bigg [ \| \bar { q } _ { t + 1 } - \mathrm { s g } [ \mu _ { q , t + 1 } ] \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left\| \bar { p } _ { t + 1 } - \mathrm { s g } [ \mu _ { p , t + 1 } ] \right\| _ { 2 } ^ { 2 } \bigg ] . } \end{array}\tag{45}
$$

## B.3 TEMPORAL CONTRASTIVE CRITIC

The temporal critic operates on posterior phase trajectories inferred from collected experience. Future horizons are sampled according to the truncated geometric distribution in Equation 4, with $K = 1 6$ and $\gamma _ { \mathrm { T } } = 0 . 9 8$ in the reported configuration. Only anchor-future pairs belonging to the same episode are retained.

The anchor and future representations follow Equation 8. Both representations are $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ before evaluating the score in Equation 9. The critic uses a representation dimension of 64 and a horizon-embedding dimension of 16. The temperature is initialized to 0.1 and optimized jointly with the critic.

The temporal critic is trained using $\mathcal { L } _ { \mathrm { T C } }$ in Equation 10 with a batch size of 256 and AdamW learning rate $3 \times 1 0 ^ { - 4 }$ . Gradients are stopped at the phase-state input in the default configuration, so this objective updates the temporal critic without directly propagating gradients into the Hamiltonian world model.

At reward computation time, the temporal mismatch is evaluated using Equation 11, while the Hamiltonian modulation and the raw $\mathrm { \dot { H } T C ^ { 2 } }$ reward are computed according to Equation 13 and 14, respectively. The $\mathrm { H T C ^ { 2 } }$ reward is then normalized using expected-distance normalization. For $\ell _ { 2 } \cdot$ -normalized representations and Euclidean distance, the reference distance is ${ \sqrt { 2 } } .$

## B.4 COLD START AND REWARD SCHEDULING

The cold-start RND (Burda et al., 2018b) module operates on the posterior phase representation $z _ { t } .$ It consists of a trainable predictor $f _ { \xi }$ and a fixed randomly initialized target network ${ \bar { f } } .$ The raw RND signal is

$$
r _ { \mathrm { R N D } } ^ { \mathrm { r a w } } ( t ) = \left. f _ { \xi } ( z _ { t } ) - \bar { f } ( z _ { t } ) \right. _ { 2 } ^ { 2 } .\tag{46}
$$

The predictor and target are MLPs with hidden dimension 256 and output dimension 128. The predictor is optimized with AdamW at learning rate $3 \times 1 0 ^ { - 4 }$

The RND signal is normalized by running statistics before reward mixing. The $\mathrm { R N D - t o - H T C ^ { 2 } }$ transition follows Equation 15] and 16, with $T _ { \mathrm { w a r m } } = 5 0 , 0 0 0$ and $T _ { \mathrm { a n n e a l } } \stackrel { - } { = } 2 0 0 { , } 0 0 0$ environment steps. The resulting intrinsic reward is combined with the extrinsic reward according to Equation 17.

## B.5 HYPERPARAMETERS

The core exploration hyperparameters are summarized in Table 2, while the policy-optimizer configurations are reported in Table 3.

## B.6 TRAINING PSEUDOCODE

Algorithms 1 and 2 summarize the integration of $\mathrm { H T C ^ { 2 } }$ with off-policy SAC and on-policy PPO respectively. Standard SAC and PPO optimization steps are omitted for clarity.

Table 2: Core hyperparameters for $\mathrm { H T C ^ { 2 } }$ and intrinsic-reward baselines.
<table><tr><td colspan="2"> $\mathbf { H T C } ^ { 2 }$ </td><td colspan="4">Intrinsic-reward baselines</td></tr><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>ICM</td><td>E3B</td><td>RIDE</td></tr><tr><td>Intrinsic scale  $\beta$ </td><td>0.05</td><td>Intrinsic scale  $\beta$ </td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Cold-start reward</td><td>RND</td><td>Latent dim</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Warmup steps</td><td>50K</td><td>Reward lr</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Annealing steps</td><td>200K</td><td>Batch size</td><td>256</td><td>256</td><td>256</td></tr><tr><td>World-model lr</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>Update proportion</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>World-model batch size</td><td>256</td><td>Reward normalization</td><td>RMS</td><td>RMS</td><td>RMS</td></tr><tr><td>Deterministic state dim</td><td>256</td><td>E3B ridge</td><td></td><td>0.1</td><td></td></tr><tr><td>Stochastic latent dim</td><td>32</td><td>RIDE k</td><td></td><td></td><td>10</td></tr><tr><td> $q / p$  dim</td><td>16 / 16</td><td>Kernel cluster distance</td><td></td><td></td><td>0.008</td></tr><tr><td>Categorical classes</td><td>32</td><td>Kernel epsilon</td><td></td><td></td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Sequence chunk</td><td>32</td><td>RIDE c</td><td></td><td></td><td>0.001</td></tr><tr><td>Integrator  $\mathbf { \xi } / \mathbf { \Xi } \Delta t$ </td><td>Leapfrog / 0.1</td><td>RIDE  $s _ { m }$ </td><td></td><td></td><td>8.0</td></tr><tr><td>Logit projection std.</td><td>0.1</td><td>κ</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Residual scale</td><td>1.0</td><td>γ</td><td>一</td><td></td><td></td></tr><tr><td>KL balance α</td><td>0.8</td><td></td><td></td><td></td><td></td></tr><tr><td>Free-nats stabilization</td><td>1.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Physics loss weight</td><td>1.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Temporal horizon k</td><td>1-16</td><td></td><td></td><td></td><td></td></tr><tr><td>Horizon sampling</td><td>Geometric (0.98)</td><td></td><td></td><td></td><td></td></tr><tr><td>Temporal critic lr / batch</td><td> $3 \times 1 0 ^ { - 4 } / 2 5 6$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Critic hidden / repr. dim</td><td> $2 5 6 / 6 4$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Horizon embedding dim</td><td>16</td><td></td><td></td><td></td><td></td></tr><tr><td>Temperature init.</td><td>0.1 (learned)</td><td></td><td></td><td></td><td></td></tr><tr><td>Similarity</td><td> $\mathrm { N e g a t i v e } \ell _ { 2 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Representation normalization</td><td>Yes</td><td></td><td></td><td></td><td></td></tr><tr><td>Energy gate  $\alpha _ { H } / c _ { H }$ </td><td>0.1 / 5.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Smoothness  $\lambda _ { \mathrm { s m } }$ </td><td>0.01</td><td></td><td></td><td></td><td></td></tr></table>

Table 3: SAC and PPO hyperparameters
<table><tr><td>Hyperparameter</td><td>SAC</td><td>PPO</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Feature dim</td><td>50</td><td>512</td></tr><tr><td>Hidden dim</td><td>1024</td><td>1024</td></tr><tr><td>Batch size</td><td>256</td><td>256</td></tr><tr><td>Discount γ</td><td>0.99</td><td>0.99</td></tr><tr><td>Initial random steps</td><td>5000</td><td></td></tr><tr><td>Replay size</td><td>50000</td><td></td></tr><tr><td>Initial temperature</td><td>0.1</td><td></td></tr><tr><td>Target update τ</td><td>0.005</td><td></td></tr><tr><td>Actor / target update freq.</td><td>1/2</td><td></td></tr><tr><td>Rollout steps</td><td></td><td>256</td></tr><tr><td>PPO epochs</td><td></td><td>15</td></tr><tr><td>Clip range / value clip</td><td></td><td>0.2 / 0.1</td></tr><tr><td>Entropy coefficient</td><td></td><td>0.1</td></tr><tr><td>Value coefficient</td><td></td><td>0.5</td></tr><tr><td>Max gradient norm</td><td></td><td>0.5</td></tr><tr><td></td><td></td><td></td></tr></table>

Algorithm $\bf { 1 H T C ^ { 2 } }$ with Soft Actor-Critic   
1: Params: $\pi _ { \eta } , Q _ { \omega _ { 1 } } , Q _ { \omega _ { 2 } } , p _ { \varphi } , ( \phi _ { \theta } , \psi _ { \theta } ) , f _ { \xi }$ policy, critics, world model, TC critic, RND   
2: Init: replay buffer $\mathcal { D } ,$ target critics, fixed RND target f ▶ initialization   
3: while not converged do   
4: $a _ { t } \sim \pi _ { \eta } ( \cdot \mid \bar { x _ { t } } ) , \quad \mathcal { D } \gets \mathcal { D } \cup \{ x _ { t } , a _ { t } , r _ { t } ^ { \mathrm { e x t } } , x _ { t + 1 } , d _ { t } \}$ environment interaction   
5: ${ \cal B } _ { \mathrm { s e q } } \sim \stackrel { \large - } { \cal D } , \quad z _ { 1 : T }  q _ { \varphi } ^ { \mathrm { p o s t } } ( \vec { B } _ { \mathrm { s e q } } )$ posterior phase trajectory   
6: $k _ { i } \sim \omega _ { k } , ~ z _ { i } ^ { + }  z _ { t _ { i } + k _ { i } }$ Equation 4   
7: $\theta  \theta - \eta _ { \mathrm { T C } } \mathbf { \dot { \nabla } } _ { \theta } \mathcal { L } _ { \mathrm { T C } }$ Equation 10   
8: $r _ { i } ^ { \mathrm { H T C ^ { 2 } } } \gets \mathrm { N o r m } \left( D _ { \theta } ( t _ { i } , k _ { i } ) G _ { H } ( t _ { i } , k _ { i } ) - \lambda _ { \mathrm { s m } } \| a _ { t _ { i } } - a _ { t _ { i } - 1 } \| _ { 2 } ^ { 2 } \right)$ Equation 14   
9: $r _ { i } ^ { \mathrm { \tilde { R N D } } } \gets \operatorname { N o r m } \left( \left| | f _ { \xi } ( z _ { t _ { i } } ) - \bar { f } ( z _ { t _ { i } } ) | \right| _ { 2 } ^ { 2 } \right)$ latent RND   
10: $\begin{array} { r } { w _ { n } \gets \mathrm { c l i p } \left( \frac { n - T _ { \mathrm { w a r m } } } { T _ { \mathrm { a n n e a l } } } , 0 , 1 \right) } \end{array}$ Equation 15   
11: $r _ { i } ^ { \mathrm { i n t } }  ( 1 -  { \mathrm { \omega } } _ { n } ) r _ { i } ^ { \mathrm { R N D } } + w _ { n } r _ { i } ^ { \mathrm { H T C } ^ { 2 } }$ Equation 16   
12: $\tilde { r } _ { i } \gets r _ { i } ^ { \mathrm { e x t } } + \beta r _ { i } ^ { \mathrm { i n t } }$ Equation 17   
13: $\mathcal { B } _ { \mathrm { R L } }  \{ ( x _ { t _ { i } } , \dot { a } _ { t _ { i } } , \tilde { r } _ { i } , x _ { t _ { i + 1 } , d _ { t _ { i } } } ) \} _ { i = 1 } ^ { B }$ anchor transitions   
14: $( \eta , \omega _ { 1 } , \omega _ { 2 } ) \gets \mathrm { S A C U p d a t e } ( \mathcal { B } _ { \mathrm { R L } } )$ standard SAC update   
15: $\varphi  \varphi - \eta _ { \mathrm { W M } } \nabla _ { \varphi } \mathcal { L } _ { \mathrm { W M } }$ Equation 7   
16: $\xi  \xi - \eta _ { \mathrm { { R N D } } } \nabla _ { \xi } \| f _ { \xi } ( z ) - \bar { f } ( z ) \| _ { 2 } ^ { 2 }$ RND predictor update

Algorithm 2 HTC² with Proximal Policy Optimization   
1: Params: $\pi _ { \eta } , V _ { \omega } , p _ { \varphi } , ( \phi _ { \theta } , \psi _ { \theta } ) , f _ { \xi }$ policy, value, world model, TC critic, RND   
2: Init: rollout buffer ${ \mathcal { R } } ,$ fixed RND target f initialization   
3: for each rollout iteration do   
4: $\mathcal { R }  \{ x _ { t } , a _ { t } , r _ { t } ^ { \mathrm { e x t } } , x _ { t + 1 } , d _ { t } \} _ { t = 1 } ^ { T _ { \mathrm { r o l l } } }$ on-policy rollout   
5: $z _ { 1 : T } \gets q _ { \varphi } ^ { \mathrm { p o s t } } ( \bar { \mathcal { R } } )$ posterior phase trajectory   
6: $k _ { i } \sim \omega _ { k } , ~ z _ { i } ^ { + }  z _ { t _ { i } + k _ { i } }$ Equation 4   
7: $\theta  \theta - \eta _ { \mathrm { T C } } \nabla _ { \theta } \mathcal { L } _ { \mathrm { T C } }$ Equation 10   
8: $r _ { i } ^ { \mathrm { H T C ^ { 2 } } } \gets \mathrm { N o r m } \left( D _ { \theta } ( t _ { i } , k _ { i } ) G _ { H } ( t _ { i } , k _ { i } ) - \lambda _ { \mathrm { s m } } \| a _ { t _ { i } } - a _ { t _ { i } - 1 } \| _ { 2 } ^ { 2 } \right)$ Equation 14   
9: $r _ { i } ^ { \mathrm { R N D } } \gets \mathrm { N o r m } \left( \lVert f _ { \xi } ( z _ { t _ { i } } ) - \bar { f } ( z _ { t _ { i } } ) \rVert _ { 2 } ^ { 2 } \right)$ latent RND   
10: $\begin{array} { r } { w _ { n } \gets \mathrm { c l i p } \left( \frac { n - T _ { \mathrm { w a r m } } } { T _ { \mathrm { a n n e a l } } } , 0 , 1 \right) } \end{array}$ Equation 15   
11: $r _ { i } ^ { \mathrm { i n t } }  ( 1 -  { w _ { n } } ) r _ { i } ^ { \mathrm { R N D } } +  { w _ { n } } r _ { i } ^ { \mathrm { H T C ^ { 2 } } }$ Equation 16   
12: $\widetilde { r _ { i } } \gets r _ { i } ^ { \mathrm { e x t } } + \beta r _ { i } ^ { \mathrm { i n t } }$ ▶ Equation 17   
13: $( \hat { A } _ { i } , R _ { i } ) \gets \mathrm { G A E } ( \mathcal { R } , \tilde { r } , V _ { \omega } )$ advantages and returns   
14: $( \eta , \omega ) \gets \mathrm { P P O U p d a t e } ( \mathcal { R } , \hat { A } , R )$ standard clipped PPO update   
15: $\varphi  \varphi - \eta _ { \mathrm { W M } } \nabla _ { \varphi } \mathcal { L } _ { \mathrm { W M } }$ Equation 7   
16: $\xi  \xi - \eta _ { \mathrm { R N D } } \dot { \nabla _ { \xi } } \| f _ { \xi } ( z ) - \bar { f } ( z ) \| _ { 2 } ^ { 2 }$ RND predictor update

## C ADDITIONAL EXPERIMENTS AND DIAGNOSTICS

## C.1 DETAILED DMC CONTROL RESULTS

This section provides the complete numerical results and learning curves corresponding to the DMC control benchmark in Section 4.2. All reported final returns are computed over five random seeds and are shown as mean ± standard deviation. Table 4 reports the final control performance for ICM, RIDE, E3B, and $\mathrm { H T C ^ { 2 } }$ under SAC and PPO, with Plan2Explore additionally included in the PPO comparison. Figures 6 and 7 provide the corresponding SAC and PPO learning curves.

With SAC, HTC² achieves the highest mean final return on five of the six tasks. On wal ker\_st and, $\mathrm { H T C ^ { 2 } }$ obtains $9 6 6 . 5 \pm 3 . 8 .$ , compared with 967.1 ± 3.3 for RIDE. The largest absolute difference relative to the strongest competing baseline occurs on walker\_walk, where $\mathrm { H T C ^ { 2 } }$ reaches 827.0 ± 115.2, compared with $7 1 7 . 9 \pm 6 \bar { 7 } . 5$ for RIDE.

Table 4: Final control performance on six DMC tasks. Episode return over five random seeds, reported as mean ± standard deviation. Within each optimizer block and task, red, orange, and yellow denote the first, second, and third-ranked exploration methods, respectively. Higher is better.
<table><tr><td>RL</td><td>Exploration</td><td>Walker Walk</td><td>Walker Run</td><td>Walker Stand</td><td>Hopper Hop</td><td>Hopper Stand</td><td>Cheetah Run</td></tr><tr><td rowspan="4">SAC</td><td>ICM</td><td> $6 7 7 . 8 { \pm } 2 3 0 . 2$ </td><td> $3 5 6 . 7 { \pm } 5 2 . 9$ </td><td> $9 3 6 . 9 { \pm } 3 1 . 7 \ $  _</td><td> $2 3 7 . 5 { \pm } 9 . 1 $  _</td><td> $8 6 3 . 3 { \pm } 2 9 . 5 $ </td><td> $6 3 1 . 9 { \pm } 4 5 . 4$ </td></tr><tr><td>RIDE</td><td> $7 1 7 . 9 { \pm } 6 7 . 5$  </td><td> $3 6 4 . 9 { \pm } 1 1 7 . 7$  1</td><td> $9 6 7 . 1 { \pm } 3 . 3 $ </td><td> $2 3 9 . 0 { \pm } 1 1 . 3 $  </td><td> $8 7 1 . 3 { \pm } 2 2 . 1 $ </td><td> $6 2 4 . 7 { \pm } 4 4 . 0$ </td></tr><tr><td>E3B</td><td> $6 7 9 . 2 { \pm } 1 9 9 . 4 $ </td><td> $4 0 1 . 6 { \pm } 9 2 . 1 \ $ </td><td> $9 6 1 . 3 { \pm } 8 . 1 $ </td><td> $2 3 5 . 4 { \pm } 2 4 . 7$ </td><td> $8 5 8 . 0 { \pm } 2 2 . 1 $ </td><td> $6 1 9 . 5 { \pm } 4 0 . 2 $ </td></tr><tr><td> $\mathrm { H T C } ^ { 2 } \ \mathrm { ( o u r s ) }$ </td><td> $8 2 7 . 0 { \pm } 1 1 5 . 2 $ </td><td> $4 1 5 . 8 { \pm } 1 1 1 . 4$ </td><td> $9 6 6 . 5 { \pm } 3 . 8 $ </td><td> $2 4 8 . 8 { \pm } 6 . 1$ </td><td> $8 8 3 . 8 { \pm } 1 2 . 5 $  </td><td> $6 5 6 . 5 { \pm } 4 5 . 0 $ </td></tr><tr><td rowspan="5">PPO</td><td>ICM</td><td> $6 0 1 . 9 { \pm } 1 7 9 . 6 \ $  _</td><td> $2 6 9 . 1 { \pm } 4 7 . 1$  </td><td> $9 2 1 . 0 { \pm } 2 5 . 4 $  _</td><td> $8 9 . 5 { \pm } 6 1 . 2 \ $  一</td><td> $6 3 7 . 6 { \pm } 5 5 . 6 $  </td><td> $4 5 1 . 5 { \pm } 6 9 . 2 $ </td></tr><tr><td>RIDE</td><td> $6 4 1 . 1 { \pm } 1 5 3 . 5$ </td><td> $2 5 4 . 2 { \pm } 2 3 . 1 $ </td><td> $8 7 7 . 4 { \pm } 8 2 . 0 $ </td><td>_  $8 3 . 4 { \pm } 6 8 . 5 $ </td><td> $5 6 5 . 8 { \pm } 1 7 4 . 0$ </td><td> $4 2 5 . 8 { \pm } 4 3 . 8 $ </td></tr><tr><td>E3B</td><td> $6 9 8 . 5 { \pm } 1 2 9 . 7 \ $ </td><td> $2 7 7 . 5 { \pm } 1 6 . 4 $  </td><td> $8 6 5 . 7 { \pm } 1 0 0 . 1 $ </td><td> $5 5 . 3 { \pm } 6 5 . 4 $ </td><td> $6 2 8 . 1 { \pm } 5 3 . 9$ </td><td> $4 4 5 . 9 { \pm } 3 3 . 3 \ \qquad $ </td></tr><tr><td> $\mathrm { P l a n 2 E x p l o r e }$ </td><td> $6 4 7 . 8 { \pm } 4 7 . 3$ </td><td> $2 3 9 . 2 { \pm } 3 8 . 4 $ </td><td> $9 2 9 . 2 { \pm } 6 . 2 $ </td><td> $7 4 . 2 { \pm } 6 8 . 6 $ </td><td> $6 2 0 . 7 { \pm } 5 1 . 8 $ </td><td> $4 3 5 . 2 { \pm } 3 7 . 6 $ </td></tr><tr><td> $\mathrm { H T C ^ { 2 } }$  (ours)</td><td> $7 3 3 . 6 { \pm } 1 6 0 . 7$ </td><td> $3 0 7 . 9 { \pm } 2 2 . 8 $ </td><td> $9 3 0 . 6 { \pm } 2 0 . 1 \ $ </td><td> $1 0 0 . 7 { \pm } 5 8 . 8 $ </td><td> $6 7 0 . 2 { \pm } 2 5 . 1 $  –</td><td> $4 5 8 . 0 { \pm } 2 7 . 4 $ </td></tr></table>

![](images/5202b56e7db6382528acb3e42ac68f26acf848e97375e87477b60d6d2077f045.jpg)  
Figure 6: SAC learning curves on six DMC tasks. Curves show the mean episode return over five random seeds, with shaded regions indicating one standard deviation, for SAC combined with ICM, RIDE, E3B, and $\mathrm { H T C ^ { 2 } }$

Under PPO, $\mathrm { H T C ^ { 2 } }$ achieves the highest mean final return on all six tasks among the evaluated methods, including Plan2Explore. On $\mathsf { w a l k e r \_ s t a n d , H T C ^ { 2 } }$ reaches $9 3 0 . 6 \pm 2 0 . 1$ , while Plan2Explore obtains $9 2 9 . 2 \pm 6 . 2 $ showing very similar mean performance in this high-return regime. On hopper\_hop, $\mathrm { H T C ^ { 2 } }$ obtains $1 0 0 . 7 \pm 5 8 . 8$ , compared with $8 9 . 5 \pm 6 1 . 2$ for the strongest competing baseline, ICM.

Overall, the SAC and PPO results show that the strong mean final-return performance of $\mathrm { H T C ^ { 2 } }$ spans multiple control tasks rather than being driven by a single environment. The complete learning curves further show how these differences evolve over training.

## C.2 DETAILED PHYSICAL-STATE COVERAGE RESULTS

We provide the detailed physical-state discretization and numerical statistics corresponding to Figure 3.   
All statistics are computed over four random seeds and reported as mean ± standard deviation.

Coverage-state construction. For cartpole\_swingup, we use

$$
s _ { t } ^ { \mathrm { p h y } } = \left[ c _ { \mathrm { c a r t } , t } , \mathrm { w r a p } ( \theta _ { \mathrm { p o l e } , t } ) \right] ,\tag{47}
$$

with $4 6 0 \times 7 2$ grid over $x _ { \mathrm { c a r t } } \in [ - 0 . 5 0 , 0 . 5 0 ]$ and $\theta _ { \mathrm { p o l e } } \in [ - \pi , \pi ]$

![](images/9b361e6a16dc683af1ca3fe05cb69ebedab35a7149b860c37311b085f38a006d.jpg)  
Figure 7: PPO learning curves on six DMC tasks. Curves show the mean episode return over five random seeds, with shaded regions indicating one standard deviation, for PPO combined with ICM, RIDE, E3B, Plan2Explore, and HTC2.

Table 5: Physical-state coverage and downstream control performance on three DMC environments. Final performance and area under the training curve (AUC) are reported as mean ± standard deviation over four random seeds. For each row, the best and second-best mean values are highlighted in green and yellow, respectively. The highlighting reflects ranking by the reported means. Higher values are better for all metrics.
<table><tr><td>Environment</td><td>Metric</td><td>Statistic</td><td>HTC2 (Ours) + SAC</td><td>ICM + SAC</td><td>RIDE + SAC</td><td>E3B + SAC</td></tr><tr><td rowspan="4">Cartpole Swingup</td><td rowspan="2">Visited States</td><td>Final</td><td>4320.000 ± 0.000</td><td>3862.250 ± 486.344</td><td>3759.500 ± 249.116</td><td>3384.750 ± 399.951</td></tr><tr><td>AUC</td><td>3525.652 ±71.624</td><td>3187.527 ± 163.460</td><td>3200.982 ± 70.494</td><td>2976.313 ± 208.753</td></tr><tr><td rowspan="2">Episode Reward</td><td>Final</td><td>579.550 ± 273.080</td><td>126.855 ± 253.710</td><td>0.000 ± 0.000</td><td>0.000 ± 0.000</td></tr><tr><td>AUC</td><td>134.662±87.884</td><td>15.452±30.791</td><td>0.058 ± 0.042</td><td>0.017 ± 0.013</td></tr><tr><td rowspan="4">Ball in Cup Catch</td><td rowspan="2">Visited States</td><td>Final</td><td>2655.000 ± 55.154</td><td>2680.500 ± 3.536</td><td>2731.500 ± 14.849</td><td>2664.500 ± 12.021</td></tr><tr><td>AUC</td><td>2415.110 ± 30.751</td><td>2406.878 ± 69.883</td><td>2499.878±14.418</td><td>2334.732 ± 19.627</td></tr><tr><td rowspan="2">Episode Reward</td><td>Final</td><td>958.730±2.447</td><td>939.715 ± 29.013</td><td>964.390 ± 1.909</td><td>945.705 ± 13.513</td></tr><tr><td>AUC</td><td>708.565±87.088</td><td>544.440 ± 262.849</td><td>716.214±27.763</td><td>399.726 ± 85.061</td></tr><tr><td rowspan="4">Finger Spin</td><td rowspan="2">Visited States</td><td>Final</td><td>5541.500 ± 34.648</td><td>5428.500 ± 58.690</td><td>5412.000 ± 156.978</td><td>5513.500± 41.719</td></tr><tr><td>AUC</td><td>5317.105±4.489</td><td>5202.047 ± 88.931</td><td>5195.651 ± 213.415</td><td>5317.360±34.418</td></tr><tr><td rowspan="2">Episode Reward</td><td>Final</td><td>965.350±6.604</td><td>957.740 ± 0.905</td><td>956.460 ± 3.281</td><td>927.550 ± 41.111</td></tr><tr><td>AUC</td><td>877.415 ± 11.824</td><td>880.650±12.565</td><td>881.395±3.270</td><td>846.177 ± 28.606</td></tr></table>

For bal1\_in\_cup\_cat ch, the physical state is

$$
s _ { t } ^ { \mathrm { p h y } } = \left[ x _ { \mathrm { b a l l } , t } - x _ { \mathrm { c u p } , t } , z _ { \mathrm { b a l l } , t } - z _ { \mathrm { c u p } , t } \right] ,\tag{48}
$$

which is discretized using a $6 4 \times 6 4$ grid, with both dimensions restricted to [—0.35, 0.35].

For finger\_spin, we use

$$
s _ { t } ^ { \mathrm { p h y } } = \left[ \mathrm { w r a p } ( \theta _ { \mathrm { o b j } , t } ) , \mathrm { c l i p } ( \dot { \theta } _ { \mathrm { o b j } , t } , - 2 5 , 2 5 ) \right] ,\tag{49}
$$

with a $7 2 \times 8 0$ grid over $\theta _ { \mathrm { o b j } } \in [ - \pi , \pi ]$ and $\dot { \theta } _ { \mathrm { o b j } } \in [ - 2 5 , 2 5 ]$ . These task-specific physical states are used only for evaluation and do not affect policy learning.

Table 5 reports the complete coverage and reward statistics for the three environments. On cartpole\_swingup, HTC² reaches 4320.0 ± 0.0 visited states at the end of training and obtains a coverage AUC of 3525.7 ± 71.6, both the highest mean values among the four methods. Its final episode reward and reward AUC are 579.6 ± 273.1 and $1 3 4 . 7 \pm 8 7 . 9 ,$ respectively, which are also the highest reported means for this task.

On ball\_ $\mathtt { \_ m \_ c u p \_ c a t c h }$ , RIDE achieves the highest final coverage, $2 7 3 1 . 5 \pm 1 4 . 8$ , followed by ICM at $2 6 8 0 . 5 \pm 3 . 5 , \mathrm { E } 3 \mathrm { B }$ at $2 6 6 4 . 5 \pm 1 2 . 0$ , and $\mathrm { H } \mathrm { { \bar { T } C } ^ { 2 } }$ at $2 6 5 5 . 0 \pm 5 5 . 2$ In terms of coverage AUC, HTC² obtains the second-highest mean value of $2 4 1 5 . 1 \pm 3 0 . 8$ , behind RIDE at $2 4 9 9 . 9 \pm 1 \bar { 4 } . 4 .$ For downstream control, $\mathrm { H T C ^ { 2 } }$ reaches a final reward of $9 5 8 . 7 \pm 2 .$ 4 and a reward AUC of $7 0 8 . 6 \pm 8 7 . 1$ both ranking second among the evaluated methods and remaining close to the corresponding RIDE values of $9 \bar { 6 4 } . 4 \pm 1 . 9$ and $\bar { 7 } 1 6 . 2 \pm 2 7 . 8$

On finger\_spin, $\mathrm { H T C ^ { 2 } }$ obtains the highest final coverage, $5 5 4 1 . 5 \pm 3 4 . 6$ , followed by E3B at $5 5 1 3 . 5 \pm 4 1 . 7$ . The coverage AUC values of $\mathrm { H T C ^ { 2 } }$ and E3B are $5 3 1 7 . 1 \pm 4 . 5$ and $5 3 1 7 . 4 \pm 3 4 . 4 ,$ respectively. $\mathrm { H T C ^ { 2 } }$ also achieves the highest final reward, $9 6 5 . 4 \pm 6 . 6$ . For reward AUC, RIDE and ICM obtain $8 8 1 . 4 \pm 3 . 3$ and $8 8 0 . 7 \pm 1 2 . 6 ,$ while $\mathrm { H T C ^ { 2 } }$ reaches $8 7 7 . 4 \pm 1 1 . 8$

Across the three tasks, $\mathrm { H T C ^ { 2 } }$ achieves the highest final visited-state count on cartpole\_swingup and $\mathtt { f i n g e r \_ s p i n . }$ while RIDE achieves the highest final coverage on $\mathtt { b a l l \_ i n \_ c u p . }$ \_catch. The corresponding reward statistics show the same diversity across environments, with $\mathrm { H T C ^ { 2 } }$ attaining the highest final reward on $\mathtt { c a r t p o l e \_ m }$ swingup and $\mathtt { f i n g e r \_ s p i n }$ , and the second-highest final reward on $\mathtt { b a l l \_ i n \_ c u p \_ c a t c h }$

## C.3 ADDITIONAL HAMILTONIAN LATENT DIAGNOSTICS

This section provides additional diagnostics for the learned Hamiltonian phase representation. The main text shows that the learned scalar $H _ { \varphi } ( q , p )$ is stable under passive evolution, responds to policy actions, and varies smoothly over the projected $q$ and $p$ subspaces. Here, we further inspect the internal organization of the phase latents from three complementary perspectives: temporal phase trajectories, action-conditioned phase updates, and dimension-wise $q \mathrm { - } p$ coupling.

Phase trajectory organization. Figure 8 visualizes the coordinate-like latent q and the momentumlike latent p using separate PCA bases. For each subspace, we show three views: coloring by the learned Hamiltonian energy, coloring by normalized episode time, and projected trajectory traces. Both q and p exhibit smooth energy variation along the projected trajectory, while the normalized-time coloring shows an ordered temporal progression. The trajectory traces further indicate that the learned phase state evolves along coherent low-dimensional paths. This supports the use of posterior phase trajectories for same-episode future pair sampling in $\mathrm { \ddot { H } T C ^ { 2 } }$

![](images/937005e7ad7185f1dc22091c30a29ddc007d80f0cde4bafc19984fc9f5e15d56.jpg)

![](images/1c2b1fd47cf17ae20de0d2c046731e48a2d7fb3bf4afe760f20fa3c3befc3424.jpg)

![](images/2fa0b3c423f3480a96517ee88e515c12204a3684d2cc55d8d55d2bd37cbbab19.jpg)

![](images/10dd69974aa6b7a8520bd719842becc0c0bc71e50ac268b4e862c424d28ef6e1.jpg)

![](images/cfd0495c98a715a725cbddeacd6192a58631cdce4da7e399f3b2801e5fcd79fb.jpg)

![](images/21ad6cafd7efe355ae5785c13d521e3d34fe96a6c33edf4eda38eb6a406c9285.jpg)  
Figure 8: Separate PCA projections of coordinate-like $q$ and momentum-like p latents on walker\_walk. Each subspace is visualized by learned Hamiltonian energy $H _ { \varphi } ( q , p )$ , normalized episode time, and projected trajectory traces. The learned phase representation exhibits smooth energy organization and temporally coherent evolution in both subspaces.

Action-conditioned phase evolution. Figure 9 analyzes how actions modulate the learned phase dynamics. The phase PCA colored by action magnitude and learned control push shows that controlrelated quantities vary systematically over the phase trajectory rather than being uniformly distributed. The one-step phase updates further separate low-control and high-control transitions: high-control samples produce larger and more directionally distinct phase updates. The phase-step magnitude statistics also increase for the high-control group. These results indicate that the learned phase state is responsive to action-conditioned control, which is the transition structure used by $\mathrm { H T C ^ { 2 } }$ to construct horizon-conditioned future pairs.

![](images/671554e0a11651ff151644a24b34f3c04e7c302e56419cd1986513e41082699e.jpg)  
Figure 9: Action-conditioned phase geometry on walker\_walk. The phase PCA is colored by action magnitude and learned control push. One-step phase updates are grouped by control strength, and the high-control group produces larger phase-step magnitudes. This shows that the learned phase dynamics are modulated by action-conditioned control.

Dimension-wise $q \mathrm { - } p$ coupling. Figure 10 reports the dimension-wise absolute correlation between the coordinate-like subspace $q$ and the momentum-like subspace $p$ on walker\_walk, hopper\_hop, and cheetah\_run. Instead of treating $q$ and $p$ as two monolithic vectors, this analysis inspects all $1 6 \times 1 6$ cross-dimensional pairs. The heatmaps show that the $q \mathrm { - } p$ relation is not uniformly coupled across all dimensions: strong correlations appear only for selected dimension pairs, while many other pairs remain weakly or moderately correlated. This pattern indicates that the learned phase representation does not collapse $q$ and $p$ into a single mixed latent block. Rather, the two subspaces share the same trajectory-level dynamics while preserving a degree of dimension-wise separation.

This result is consistent with the intended phase-space structure of $\mathrm { H T C ^ { 2 } }$ . The coordinate-like and momentum-like components should not be fully independent, because they are inferred from the same controlled trajectory and jointly determine the learned Hamiltonian $H _ { \varphi } ( q , p )$ . At the same time, they should not become identical or uniformly entangled. The observed local and dimension-selective coupling therefore supports the use of a structured phase representation for Hamiltonian temporal contrastive exploration.

Summary. These diagnostics provide a finer view of the Hamiltonian phase representation used by $\mathrm { H T C ^ { 2 } }$ . The separate $q / p \operatorname { P C A }$ plots show energy-ordered and time-coherent phase trajectories; the action-conditioned analysis shows that phase updates are modulated by control strength; and the crosstask $q { - } p$ coupling maps show trajectory-level dependence without uniform dimension-wise collapse Together, these results support the representational structure assumed by $\mathrm { H T C ^ { 2 } }$ : exploration is driven by temporal future discovery in a Hamiltonian phase space whose components are dynamically related yet not reduced to a single entangled latent cloud.

![](images/99edb19689c86a6e983541b01f6e0b66fbeb480dd61015f22c82d13b8c3056f7.jpg)  
Figure 10: Dimension-wise q-p coupling across DMC tasks. Each heatmap shows the absolute correlation between the 16 dimensions of coordinate-like q and the 16 dimensions of momentum-like $p .$ The correlations are local and dimension-selective rather than uniformly high, indicating that q and p remain dynamically related but do not collapse into a single fully mixed latent representation.

## C.4 DETAILED PARAMETRIC OOD RESULTS

This section reports the detailed numerical values for the parametric OOD policy-retention experiment in Section 4.5. Retention is computed as

$$
\mathrm { R e t e n t i o n } = 1 0 0 \times { \frac { R _ { \mathrm { O O D } } } { R _ { \mathrm { I D } } } } ,
$$

where $R _ { \mathrm { O O D } }$ is the zero-shot return under the perturbed environment and $R _ { \mathrm { I D } }$ is the return in the original environment for the same method and seed. We evaluate four physical shifts: gravity (1.5×), friction (2.0×), mass $( 1 . 5 \times )$ , and damping $( 2 . 0 \times )$

Table 6: Parametric OOD zero-shot policy retention on walker\_walk. Values report return retention relative to the base environment (%, mean ± std).
<table><tr><td>Model</td><td>Gravity 1.5×</td><td>Friction 2.0×</td><td>Mass 1.5×</td><td> $\mathrm { D a m p i n g } \ : 2 . 0 \times$ </td></tr><tr><td> $\mathbf { S } \mathbf { A } \mathbf { C } { + } \mathbf { I } \mathbf { C } \mathbf { M }$ </td><td> $6 9 . 5 8 \pm 3 . 0 2$ </td><td> $9 4 . 1 0 \pm 6 . 0 5$ </td><td> $6 4 . 6 4 \pm 1 3 . 1 9$ </td><td> $1 0 1 . 4 5 \pm 0 . 0 8$ </td></tr><tr><td> $\mathrm { S A C + R I D E }$ </td><td> $5 8 . 7 9 \pm 8 . 0 1$ </td><td> $9 5 . 8 5 \pm 0 . 2 1$ </td><td> $6 2 . 8 0 \pm 1 3 . 6 6$ </td><td> $\mathbf { 1 0 1 . 7 3 \pm 0 . 1 3 }$ </td></tr><tr><td> $\mathrm { S A C + E 3 B }$ </td><td> $7 0 . 0 8 \pm 1 6 . 2 6$ </td><td> $9 7 . 3 6 \pm 1 . 5 6$ </td><td> $7 0 . 6 5 \pm 1 4 . 1 6$ </td><td> $1 0 0 . 4 6 \pm 1 . 7 9$ </td></tr><tr><td> $\mathrm { S A C { + } H T C ^ { 2 } }$ </td><td> ${ \bf 7 8 . 7 0 \pm 1 1 . 9 4 }$ </td><td> $\mathbf { 1 0 0 . 0 1 \pm 0 . 8 3 }$ </td><td> $\mathbf { 8 2 . 3 8 \pm 1 0 . 4 0 }$ </td><td> $9 7 . 6 8 \pm 3 . 0 5$ </td></tr></table>

Table 7: Parametric OOD zero-shot policy retention on hopper\_st and. Values report return retention relative to the base environment (%, mean ± std).
<table><tr><td>Model</td><td>Gravity 1.5×</td><td>Friction 2.0×</td><td>Mass 1.5×</td><td>Damping 2.0×</td></tr><tr><td>SAC+ICM</td><td> $5 . 4 2 \pm 6 . 9 5$ </td><td> $9 9 . 1 1 \pm 1 . 2 1$ </td><td> $4 0 . 7 6 \pm 0 . 9 0$ </td><td> $9 9 . 4 5 \pm 1 . 3 7$ </td></tr><tr><td> $\mathrm { S A C + R I D E }$ </td><td> ${ \bf 5 2 . 3 7 \pm 1 4 . 8 4 }$ </td><td> $9 6 . 5 5 \pm 1 . 2 8$ </td><td> $6 4 . 3 2 \pm 1 5 . 8 3$ </td><td> $\mathbf { 1 0 0 . 4 4 \ : \pm 0 . 3 0 }$ </td></tr><tr><td> $\mathrm { S A C + E 3 B }$ </td><td> $4 . 2 3 \pm 3 . 4 8$ </td><td> $9 4 . 8 5 \pm 2 . 7 1 $ </td><td> $3 2 . 2 8 \pm 1 7 . 3 9$ </td><td> $9 9 . 6 0 \pm 0 . 2 3 $ </td></tr><tr><td> $\mathrm { S A C + H T C ^ { 2 } }$ </td><td> $3 0 . 2 1 \pm 5 . 6 3$ </td><td> ${ \bf 9 9 . 1 5 \pm 1 . 3 6 }$ </td><td> ${ \bf 8 5 . 7 3 \pm 1 0 . 9 9 }$ </td><td> $9 7 . 6 5 \pm 3 . 0 6$ </td></tr></table>

Discussion.The detailed results show that $\mathrm { H T C ^ { 2 } }$ obtains the best retention on seven out of twelve OOD settings. The strongest pattern appears under mass perturbations, where $\mathrm { H T C ^ { 2 } }$ achieves the highest retention on all three tasks. It also achieves the best gravity-shift retention on walker\_walk and cheet ah\_run. These results are consistent with the objective of $\mathrm { H T C ^ { 2 } } ;$ exploration is guided by multi-step temporal discovery in a Hamiltonian phase representation, which encourages policies to experience physically informative transitions during training. Friction and damping shifts show more task-dependent behavior, indicating that different intrinsic rewards can favor different robustness profiles.

## C.5 ABLATION LEARNING CURVES

We further analyze the sample efficiency of the ablation variants in Figure 11. The curves compare full $\mathrm { H T C ^ { 2 } }$ with variants that remove the physical consistency loss, remove the Hamiltonian energy gate, replace temporal contrastive reward with one-step $\Delta H$ , remove the RND (Burda et al., 2018b) warm start, or remove representation normalization.

Table 8: Parametric OOD zero-shot policy retention on cheetah\_run. Values report return retention relative to the base environment (%, mean ± std).
<table><tr><td>Model</td><td>Gravity 1.5×</td><td>Friction 2.0×</td><td>Mass 1.5×</td><td>Damping 2.0×</td></tr><tr><td> $\mathbf { S } \mathbf { A } \mathbf { C } { + } \mathbf { I } \mathbf { C } \mathbf { M }$ </td><td> $7 0 . 9 3 \pm 2 . 4 6$ </td><td> $5 4 . 8 4 \pm 2 5 . 5 5$ </td><td> $8 9 . 3 3 \pm 1 7 . 9 8$ </td><td> $4 3 . 7 4 \pm 9 . 2 9$ </td></tr><tr><td> $\mathrm { S A C + R I D E }$ </td><td> $7 7 . 7 2 \pm 9 . 1 6$ </td><td> ${ \bf 9 2 . 3 1 \pm 8 . 8 8 }$ </td><td> $1 0 0 . 9 0 \pm 7 . 7 8$ </td><td> $6 3 . 7 0 \pm 4 . 6 8$ </td></tr><tr><td> $\mathrm { S A C + E 3 B }$ </td><td> $7 4 . 4 7 \pm 2 7 . 4 6$ </td><td> $6 1 . 4 3 \pm 7 . 6 2$ </td><td> $9 6 . 4 3 \pm 1 8 . 1 8$ </td><td> ${ \pm } 7 . 6 3 \pm 6 . 5 2$ </td></tr><tr><td> $\mathrm { S A C { + } H T C ^ { 2 } }$ </td><td> ${ \bf 8 4 . 1 8 \pm 2 . 1 5 }$ </td><td> $6 4 . 6 7 \pm 3 8 . 4 4$ </td><td> ${ \bf 1 0 7 . 8 5 \pm 8 . 0 9 }$ </td><td> $5 9 . 8 9 \pm 4 . 7 1$ </td></tr></table>

![](images/7d091bbe63f46f21d0691bfe95c7958f874117c340ff75663884e957828ab695.jpg)  
Figure 11: Ablation learning curves. Curves report mean performance over 4 seeds, and shaded regions denote standard deviation. Full $\mathrm { H T C ^ { 2 } }$ is compared with variants removing the physical consistency loss, Hamiltonian energy gate, RND warm start, representation normalization, and with a one-step $\dot { \Delta H }$ reward.

On walker\_walk, full $\mathrm { H T C ^ { 2 } }$ improves steadily and reaches the highest final return. Several variants learn useful behaviors, but the full method maintains the strongest late-stage performance.

On hopper\_hop, the difference between variants is more pronounced: full $\mathrm { H T C ^ { 2 } }$ reaches high return and remains stable, whereas the one-step $\Delta H$ reward fails to learn an effective hopping behavior. The variant without representation normalization also suffers a large drop, indicating that stable contrastive distances are important for this underactuated task.

On cheetah\_run, full $\mathrm { H T C ^ { 2 } }$ again achieves the strongest final performance. The variants without $\mathcal { L } _ { \mathrm { p h y s } } ,$ , without the energy gate, or without RND (Burda et al., 2018b) warm start improve during training but converge to lower returns. These learning curves are consistent with the final-return ablation in Table 1: the complete $\mathrm { H T C ^ { 2 } }$ design improves both sample efficiency and final performance by combining Hamiltonian phase learning with multi-step temporal contrastive curiosity

## C.6 POLICY ROLLOUT VISUALIZATION

To qualitatively inspect the behavior induced by the final $\mathrm { H T C ^ { 2 } }$ policy, we perform additional environment rollouts from the trained checkpoint and sample temporally ordered observations from six DMC tasks. Figure 12 shows twelve sampled frames $( t = 0 , \ldots , 1 1 )$ for walker\_walk, walker\_run, hopper\_hop, hopper\_stand, cheetah\_run, and walker\_stand.

The sampled trajectories exhibit distinct task-dependent motion patterns. On walker\_walk, the agent remains upright while its leg configuration evolves across the rollout, whereas walker\_run exhibits substantially larger changes in body and limb configuration. The hopper remains close to an upright posture on hopper\_stand, while hopper\_hop shows greater temporal variation in its articulated configuration. On cheet ah\_run, the torso and limbs evolve through a sequence of locomotion configurations. For walker\_st and, the rollout progresses from an initially inclined configuration toward a more upright posture, followed by continued balancing adjustments.

![](images/77a8bfef6505cd99f242be80219207f565d615316831bf7728a5499c6233df7f.jpg)  
Figure 12: $\mathbf { H T C ^ { 2 } }$ policy rollouts on six DMC tasks. Each row corresponds to one task, and columns show twelve temporally ordered sampled frames $( t = 0 , \ldots , 1 1 )$ from the final trained policy checkpoint. The visualization is used only as a qualitative inspection of policy behavior.