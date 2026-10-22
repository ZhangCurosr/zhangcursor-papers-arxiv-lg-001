# DriveMLP: When Training Strategy Outperforms Model Complexity in Autonomous Driving

Alessandro Canevaro<sup>∗,1,2</sup> Julian Schmidt<sup>1</sup> Hang Yu<sup>1,3</sup> Ye Jin<sup>1,4</sup>

Sajad Marvi<sup>1,5</sup> Georg Martius<sup>2</sup> Julian jordan<sup>1</sup>

Abstract: Learning motion planning policies that remain reliable under closedloop execution is a central challenge in Autonomous Driving (AD). Imitation Learning (IL) scales well to large AD datasets but suffers from distribution shift, causal confusion, and compounding errors under closed-loop execution. In contrast, Reinforcement Learning (RL) typically requires handcrafted rule-based rewards and massive simulation. We study a simple closed-loop IL framework that uses RL purely as an optimization tool, guided by a trajectory-distance reward derived from logged expert data. We refer to this training strategy as Expert-Guided RL (EGRL). Through a controlled empirical study we find evidence that the training procedure, rather than model size and complexity, accounts for a substantial part of the performance gains. We then introduce DriveMLP, a lightweight MLP policy trained with EGRL, as a simple instantiation of these findings. It achieves a score of 86.51 on the nuPlan Val14 Reactive benchmark, a +3.2 point improvement over the previous best IL planner (Flow Planner), with > 15× faster inference. Code is available at:

https://github.com/alessandro-canevaro/DriveMLP.

Keywords: Autonomous Driving, Closed-Loop Training, Planning

## 1 Introduction

Autonomous Driving (AD) planning requires policies that remain reliable in complex multi-agent interactions and during rare, safety-critical events [1]. Despite rapid progress in large-scale learningbased planners, a fundamental challenge remains: how to train models that are robust under closedloop execution without sacrificing efficiency or scalability.

In practice, Imitation Learning (IL) has emerged as the dominant paradigm because it scales well to large open-loop datasets of human driving. Recent methods improve closed-loop performance through auxiliary objectives and data augmentation, alongside increasingly expressive architectures. These include transformer-based [2] planners such as PlanTF [3] and PLUTO [4], as well as generative approaches such as Diffusion Planner [5] and Flow Planner [6].

However, despite the advances, IL fundamentally suffers from distribution shift and causal confu sion: models are trained in an open-loop setting, where spurious correlations in the data may be learned, and are then deployed in closed-loop, where compounding errors can push the policy into previously unseen states [7, 8]. A telling example of causal confusion is that Diffusion Planner loses nearly 10 percentage points in closed-loop performance when provided with additional ego-state inputs [5], suggesting that open-loop training can turn useful causal cues into spurious shortcuts. More broadly, under mild test-time distribution shifts, such as the nuPlan reactive benchmark, where surrounding agents are controlled by rule-based IDM [9] policies instead of replayed logs, open-loop IL methods exhibit substantial performance degradation. These observations motivate training objectives that expose the policy to its own induced state distribution.

Reinforcement Learning (RL) offers this closed-loop training signal, but applying it to AD remains challenging [10, 11]. Strong results often depend either on heavily engineered rule-based reward functions, as in CaRL [12], or on simpler, though still rule-based, rewards paired with extremely large-scale simulation, as in GIGAFLOW [13].

A natural alternative is to derive supervision from expert demonstrations rather than hand-designed rules. Online Inverse Reinforcement Learning (IRL) methods such as GAIL [14] and AIRL [15] learn rewards via adversarial training, while non-adversarial alternatives such as IQ-Learn [16] rely on occupancy matching. However, both families are computationally expensive and difficult to stabilize at scale [17]. Recent hybrid IL-RL approaches reduce or eliminate adversarial reward learning by analytically deriving demonstration-based rewards, but introduce other forms of complexity. CarPlanner [18] uses RL as a trajectory candidate generator, requiring a secondary rule-based post processing module for final selection and refinement. Peng et al. [19] leverage expert-guided trajectory signals for multi-agent traffic simulation, employing a two-stage training strategy where RL fine-tunes a pre-trained autoregressive policy. Another approach is BC-SAC [8], which alternates BC and RL updates using safety-related rewards, treating RL primarily as a corrective component.

In summary, recent ego-planning methods rely on demonstration-based supervision using complex architectures that sacrifice inference speed or, orthogonally, on engineered rule-based supervision.

These approaches motivate an empirical question: how competitive can a simple planning policy become when trained in closed-loop simulation with demonstration-derived supervision? We investigate a practical training recipe and evaluate its performance against SOTA methods on Highway and nuPlan benchmarks. We further assess whether this recipe enables a lightweight MLP to compete with recent generative imitation-based planners.

Additional related work and references are discussed in Appendix A.

In this work, we study closed-loop imitation through DriveMLP, a lightweight MLP planner trained using standard RL in a non-reactive log-replay simulator. Training is guided by a reward derived from trajectory deviations from the logged expert, a strategy we name Expert-Guided RL (EGRL). Our main contributions are as follows: (1) Through a controlled empirical analysis, we show that performance gains stem largely from the training procedure, rather than increased model size or complexity alone. (2) We introduce DriveMLP, a lightweight MLP policy trained with EGRL, as a simple instantiation of these findings. On the evaluated nuPlan reactive splits, it achieves SOTA performance among purely learned imitation-based planners while running > 15× faster than recent generative planners. (3) We propose an exponential reward formulation for AD build-

![](images/cf5fbc61d999d1b2923f130bf40775d79c66e781d950129289f5b02400ffa309.jpg)  
Figure 1: DriveMLP combines strong reactive performance with low inference cost. The plot shows nuPlan Val14-Reactive scores against inference frequency, with circle size indicating parameter count. DriveMLP, a lightweight MLP trained with EGRL, achieves higher scores and faster inference than the compared SOTA planners.

ing on [20] that provides dense and stable supervision, improving over the Peng et al. [19] formulation. (4) We show that through closed-loop RL exploration, our model is substantially more robust to action noise and data efficient compared to SOTA open-loop IL.

## 2 Method

This section describes DriveMLP and its EGRL training procedure. We first formalize the problem and non-reactive simulation setup, then introduce the trajectory-distance reward and the policy architecture.

## 2.1 Problem Formulation

We formulate AD planning as a Markov Decision Process defined by the tuple $( S , { \mathcal { A } } , { \mathcal { T } } , { \mathcal { R } } , { \gamma } )$ where S is the state space, A is the action space, $\mathcal { T } : \mathcal { S } \times \mathcal { A } \times \mathcal { S }  [ 0 , 1 ]$ is the transition function, $\mathcal { R } : \mathcal { S } \times \mathcal { A } $ R is the reward function, and $\gamma \in [ 0 , 1 ]$ is the discount factor.

State Space. The state $s _ { t } \in S$ consists of: (1) vectorized map representation including lane geometry, traffic light status, and route information, (2) surrounding dynamic agents with their positions, velocities, and headings, and (3) static objects in the environment.

Unlike most IL models, we additionally provide the agent with ego-vehicle dynamic state information (velocity, acceleration, steering angle, and yaw rate). All spatial information is represented in the ego-centric coordinate frame.

Action Space. We adopt the low-level control formulation, where actions ${ \mathbf a } _ { t } \in \mathcal A$ consist of normalized steering angle and acceleration commands: $\mathbf { a } _ { t } = ( \phi _ { t } , a _ { t } ) \in [ - 1 , 1 ] ^ { 2 }$ . The actions are then scaled to the vehicle model’s steering and acceleration ranges by the simulator itself.

Expert Demonstrations. We assume access to a dataset $\mathcal { D } = \{ \tau ^ { i } \} _ { i = 1 } ^ { N }$ of N expert driving state trajectories, where each trajectory $\tau ^ { i } = ( s _ { 0 } ^ { i } , \dots , s _ { T _ { i } } ^ { i } )$

## 2.2 Non-Reactive Simulation

A key design principle of our approach is training in a non-reactive simulation environment. Unlike standard RL environments where all agents respond dynamically to the ego vehicle’s actions, our environment operates as follows (Figure 2):

Ego vehicle: Controlled by the learning agent, following the dynamic model constraints with actions sampled from the policy $\pi _ { \theta }$

Other agents: Follow their pre-recorded trajectories from the dataset, independently of the ego vehicle’s behavior.

Static environment: Road geometry, traffic lights, and static obstacles remain fixed as in the original recorded scenario.

![](images/3178ae83cc5179201677aebefffcd63141b7f83d95e034307e00318e7a706951.jpg)  
Figure 2: Working principle of the non-reactive simulator. The teal rectangles represent the logged expert ego trajectory. The gray rectangles represent other agents or obstacles moving according to recorded data, while the dark blue rectangle shows the ego rollout following the actions of the learned policy. Orange arrows show the position distance between the ground truth and the ego rollout which is then used in the reward computation.

Our reward formulation (Section 2.3) is not applicable in fully reactive environments as it would not be well-defined. Deviations by the ego vehicle would causally alter the behavior of other agents, invalidating the logged expert trajectory as a reference and leading to ambiguous or misleading supervision. By replaying surrounding agents from logs, the expert ego trajectory remains a consistent reference target, enabling dense, step-wise supervision without adversarial reward learning or expert policy access.

## 2.3 Trajectory-Based Reward Function

Given access to expert trajectories, we define a simple reward function that measures the agent’s deviation from expert behavior based on position and speed. Unlike rule-based RL rewards that encode task-specific driving heuristics, our reward directly reflects expert behavior. The only additional constraints that we impose are termination conditions that define the domain in which replay-based imitation remains valid, rather than preferences over driving behavior.

Minimizing trajectory deviations can be interpreted as maximizing the likelihood of expert state visitation under a Gaussian noise model on system dynamics. In this sense, our reward implements a closed-loop analog of BC, where supervision is applied to realized state trajectories, induced by the policy itself, rather than one-step expert actions. As a result, errors compound differently and the policy is explicitly trained to recover from its own deviations. Appendix B provides a deeper discussion on closed-loop regret bounds.

At each timestep t, let $p _ { t } ^ { \mathrm { e g o } } \in \mathbb { R } ^ { 2 }$ denote the ego vehicle’s position and $p _ { t } ^ { \mathrm { e x p e r t } } \in \mathbb { R } ^ { 2 }$ denote the corresponding expert position. We define the distance between ego and expert at timestep t as:

$$
\delta _ { t } = \lVert p _ { t } ^ { \mathrm { e g o } } - p _ { t } ^ { \mathrm { e x p e r t } } \rVert _ { 2 }\tag{1}
$$

We then map the normalized distance to $[ 0 , 1 ]$ using an exponential function:

$$
r _ { t } ^ { p o s } = \exp \left( - \alpha _ { 1 } \frac { \delta _ { t } } { \kappa _ { t } ^ { \mathrm { e x p e r t } } + \epsilon } \right) , \quad \mathrm { w i t h } \quad \kappa _ { t } ^ { \mathrm { e x p e r t } } = \| p _ { t } ^ { \mathrm { e x p e r t } } - p _ { t - 1 } ^ { \mathrm { e x p e r t } } \| _ { 2 } ,\tag{2}
$$

where ϵ is a small constant that prevents division by zero when the expert vehicle is stationary. We use the expert’s displacement ${ \kappa } _ { t } ^ { \mathrm { e x p e r t } }$ as a normalization factor to account for speed-dependent tolerance, since higher speeds naturally lead to larger positional deviations for the same control error. Similarly, velocity deviations between the ego and the expert are penalized via:

$$
r _ { t } ^ { v e l } = \exp \left( - \alpha _ { 2 } \big \| v _ { t } ^ { \mathrm { e g o } } - v _ { t } ^ { \mathrm { e x p e r t } } \big \| _ { 2 } \right) ,\tag{3}
$$

where $v _ { t } \in \mathbb { R }$ denotes scalar longitudinal speed (m/s). The full reward expression is then:

$$
r _ { t } = \alpha _ { 3 } r _ { t } ^ { p o s } + ( 1 - \alpha _ { 3 } ) r _ { t } ^ { v e l }\tag{4}
$$

where $\alpha _ { 1 , 2 , 3 }$ are hyperparameters that regulate the scale of the different reward terms.

Episode Terminations. The trajectory-distance reward is meaningful only within a neighborhood of the expert trajectory, where interactions between the ego and other agents remain approximately consistent with the logged data. We enforce termination conditions that define the domain in which the imitation reward is valid. When the ego vehicle deviates beyond this domain, the episode is terminated and assigned a fixed penalty $r _ { p }$ . Specifically, the termination criteria are: (1) the ego vehicle goes outside the drivable area, (2) the ego vehicle bounding box overlaps with the ones of other agents or obstacles, or (3) the ego vehicle deviates significantly from the expert trajectory. Further details are reported in the Appendix C.3.

## 2.4 Policy Learning

The policy network $\pi _ { \theta } : { \mathcal { S } }  \Delta ( { \mathcal { A } } )$ and the value function $V _ { \phi } : { \cal { S } } $ R are parameterized as separate multi-layer perceptrons (MLPs).

To encode variable-length inputs (e.g., neighboring agents and map polylines), we use permutationinvariant set encoders [21]. Specifically, we employ dedicated encoders for the ego state, map, route, neighboring agents, and static obstacles. Each encoder is composed of a shared MLP applied to each input element, followed by mean pooling to obtain a fixed-dimensional representation and a second MLP to further process the aggregated latent features. The resulting encoder outputs are concatenated and mapped to a joint latent representation using an additional MLP, which is then fed to the policy and value heads. The policy head parameterizes a diagonal Gaussian distribution over the continuous action space, predicting both the mean and (diagonal) variance.

The model’s parameters can be optimized with standard RL algorithms. As later shown in the Experiments section, we tested two established ones: Soft Actor-Critic (SAC) [22] and Proximal Policy Optimization (PPO) [23].

## 3 Experiments

We evaluate DriveMLP on two benchmarks to assess whether our method improves closed-loop performance and robustness. We describe the experimental setup and then report quantitative results.

## 3.1 Experimental Setup

We distinguish between three evaluation settings. First, in controlled comparisons (Section 3.2) we train identical models on the same data with different learning objectives, isolating the effect of training strategy itself. Second, we apply our training strategy to recent SOTA models to assess its applicability to different architectures (Section 3.3). Lastly, we evaluate whether a lightweight model trained via our closed-loop optimization is sufficient to outperform SOTA nuPlan IL baselines (Section 3.4). To ensure a fair comparison of learning paradigms, we focus our analysis on methods trained primarily from human demonstrations and leave out rule-based and RL planners that mainly optimize traffic-rule-based reward or cost functions (e.g., speed limits, lane keeping, comfort, or progress terms).

Benchmarks. We evaluate on two AD benchmarks of increasing complexity: (1) Highwayenv [24]: A lightweight multi-lane highway driving simulator featuring dense traffic and lane changes. We use the default configuration for observations and the provided continuous control action space. (2) nuPlan [25, 26]: A large-scale, closed-loop planning benchmark built on 1,500+ hours of real-world human driving logs collected across 4 cities (Las Vegas, Pittsburgh, Boston, and Singapore), making it one of the strongest simulation proxies for real-world urban driving currently available [26, 12]. We train on scenarios from the training split and evaluate on three standard test sets: Val14 [27], Test14-Hard [3], and Test14 [3]. Additional results on the InterPlan [28] benchmark are available in Appendix D.

Implementation and evaluation. To isolate the effectiveness of our closed-loop RL training with trajectory-based rewards, we intentionally avoid common training augmentations in our method, including auxiliary losses, curriculum learning, multi-source training data, and advanced model architectures. For nuPlan, we follow the official closed-loop evaluation protocol [25] and report the overall score under both Reactive (R) and Non-Reactive (NR) settings. Full implementation details, hyperparameters, dataset preparation, and evaluation metrics are provided in Appendix C.

## 3.2 Controlled Training Strategy Comparison

Baselines. We compare our training strategy against BC, AIRL [15], and BC-RL [8] while using the same network architecture for all methods. For all training strategies except BC, we use SAC as the backbone RL optimizer on Highway-Env and PPO on nuPlan.

Results. Table 1 reports performance on both benchmarks. On Highway-Env, our method substantially outperforms BC across all metrics and achieves performance comparable to AIRL and BC-RL. AIRL takes three times more training steps to converge compared to our approach, and each iteration is further slowed down by the reward learning.

On the large-scale nuPlan benchmark, the differences between training strategies become even more pronounced. In particular, AIRL remains the most challenging method to scale due to its intrinsic instability. To mitigate this, we incorporated discriminator regularization techniques commonly used in GAN training, specifically one-sided label smoothing [29] and gradient penalty [30].

Table 1: Controlled training strategy comparison on Highway-Env (left) and nuPlan (right). Highway-Env results averaged over 3 seeds. NR = Non-Reactive, R = Reactive. All training methods use the same DriveMLP policy architecture.
<table><tr><td></td><td colspan="3">Highway-env</td><td colspan="2">Val14</td><td colspan="2">Test14-Hard</td><td colspan="2">Test14</td></tr><tr><td>Method</td><td>Reward ↑</td><td>Ep. Len ↑</td><td>Term. ↓</td><td>NR</td><td>R</td><td>NR</td><td>R</td><td>NR</td><td>R</td></tr><tr><td>BC</td><td> $1 7 . 2 0 \pm 3 . 2 6$ </td><td> $2 3 . 5 6 \pm 4 . 2 8$ </td><td> $4 8 . 7 \% \pm 2 1 . 4 8$ </td><td>53.64</td><td>32.36</td><td>17.06</td><td>19.50</td><td>30.36</td><td>33.01</td></tr><tr><td>AIRL</td><td> $2 0 . 3 7 \pm 0 . 4 0$ </td><td> $2 8 . 4 8 \pm 0 . 4 6$ </td><td> $9 . 3 \% \pm 2 . 8 7$ </td><td>51.74</td><td>58.74</td><td>43.00</td><td>52.09</td><td>48.88</td><td>61.15</td></tr><tr><td>BC-RL</td><td> $2 0 . 2 8 \pm 0 . 9 2$ </td><td> $2 8 . 7 0 \pm 1 . 1 5$ </td><td> $2 0 . 0 \% \pm 1 7 . 3 8$ </td><td>77.19</td><td>78.05</td><td>66.05</td><td>65.35</td><td>77.17</td><td>77.23</td></tr><tr><td>EGRL</td><td> ${ \bf 2 1 . 0 2 \pm 0 . 7 9 }$ </td><td> $\mathbf { 2 9 . 1 7 \pm 1 . 0 1 }$ </td><td> $7 . 3 \% \pm 3 . 5 5$ </td><td>90.38</td><td>86.51</td><td>76.68</td><td>74.50</td><td>86.38</td><td>87.06</td></tr></table>

## 3.3 Transfer Across Planner Backbones

To assess whether the benefits extend beyond MLP policies, we use EGRL to train policies based on feature encoders adapted from PlanTF [3], PLUTO [4], and Diffusion Planner [5].

We adapt these models for closed-loop RL training by (1) modifying the input representation to include ego-state features, (2) adjusting the output space from multi-step trajectory prediction to single-step control actions, and (3) removing open-loop-specific auxiliary heads and objectives such as joint multi-agent prediction. These adaptations are necessary to interface existing SOTA IL planners with our on-policy closed-loop training framework while preserving the underlying transformer backbone. Extending to iterative generative heads remains an interesting direction for future work as it requires online RL through computationally intensive multi-step denoising loops.

Table 2 shows that applying the proposed training objective to encoders derived from existing planners consistently improves reactive performance, matching or exceeding the original pipelines.

Table 2: EGRL training strategy applied to other SOTA backbones. \*Flow Planner uses a similar encoder structure as Diffusion Planner.
<table><tr><td>Model</td><td>Original (Val14-R)</td><td>+EGRL</td><td>∆%</td><td>Params</td></tr><tr><td>PlanTF</td><td>76.95</td><td>78.26</td><td>+1.70</td><td>2.0M</td></tr><tr><td>PLUTO w/o ref.</td><td>78.11</td><td>83.41</td><td>+6.78</td><td>2.4M</td></tr><tr><td>Diff. Planner*</td><td>82.80</td><td>86.05</td><td>+3.92</td><td>2.3M</td></tr></table>

Additional experiments across MLP capacities and with a reduced transformer policy are reported in Appendix E.

## 3.4 Comparison with State of the Art

Baselines. We evaluate against SOTA IL approaches, including PlanTF [3], PLUTO [4], and the more recently proposed diffusion-based [5] and flow-matching-based [6] planners that incorporate generative modeling architectures.

Results. In particular, our training strategy enables a simple MLP model with approximately 2.3M parameters to outperform several SOTA IL baselines in the reactive settings, as shown in Table 3. Our approach exhibits a significantly smaller performance gap between non-reactive and reactive simulation settings, indicating improved robustness and generalization to interactive environments.

We hypothesize that open-loop IL methods overfit to the non-reactive training distribution: by learning to imitate the expert conditioned on the observed agent configurations, these methods may implicitly rely on correlations between agent states and correct actions that hold when agents follow their logged trajectories, but break when agents react to the ego instead. DriveMLP is also trained in a non-reactive simulator, but its closed-loop objective may encourage the policy to rely less on such scene-level correlations and more on ego dynamics. This interpretation is consistent with ego-state ablation (Appendix F.3), which shows that ego-state features help DriveMLP but hurt open-loop IL methods, suggesting that closed-loop training helps establish more correct causal relationships between ego dynamics and planning decisions.

Table 3: Overall scores on nuPlan benchmark. NR = Non-Reactive, R = Reactive simulation.
<table><tr><td></td><td colspan="2">Val14</td><td colspan="2">Test14-Hard</td><td colspan="2">Test14</td></tr><tr><td>Method</td><td>NR</td><td>R</td><td>NR</td><td>R</td><td>NR</td><td>R</td></tr><tr><td>PlanTF</td><td>84.27</td><td>76.95</td><td>69.70</td><td>61.61</td><td>85.62</td><td>79.58</td></tr><tr><td>PLUTO w/o refine</td><td>88.89</td><td>78.11</td><td>70.03</td><td>59.74</td><td>89.90</td><td>78.62</td></tr><tr><td>Diffusion planner</td><td>89.87</td><td>82.80</td><td>75.99</td><td>69.22</td><td>89.19</td><td>82.93</td></tr><tr><td>Flow Planner</td><td>90.43</td><td>83.31</td><td>76.47</td><td>70.42</td><td>89.88</td><td>82.93</td></tr><tr><td>DriveMLP</td><td>90.38</td><td>86.51</td><td>76.68</td><td>74.50</td><td>86.38</td><td>87.06</td></tr></table>

## 4 Ablations and Further Analysis

We analyze DriveMLP’s robustness and efficiency through targeted ablations on reward formulation, training dynamics curves, action noise, data efficiency with reduced demonstrations, and inference latency. Additional parameter sensitivity analysis, including reward ablations, action frequency, and ego-state effects is provided in Appendix F.

## 4.1 Reward Formulation Ablation

Table 4 summarizes the impact of removing individual components of our reward formulation. We ablate (1) the speed term $r _ { t } ^ { v e l }$ , (2) the exponential normalization by replacing it with the inverse distance, and (3) the normalization by directly using the raw distance $\delta _ { t } .$ , following Peng et al. [19]. The results show that each component contributes to performance, with normalization being particularly important for stable and effective training.

Table 4: Exponential reward ablation: NuPlan Val14.
<table><tr><td>Reward</td><td>NR</td><td>R</td></tr><tr><td>Base</td><td>90.38</td><td>86.51</td></tr><tr><td>w/o speed reward</td><td>87.44</td><td>84.92</td></tr><tr><td>w/o exp.</td><td>84.87</td><td>82.74</td></tr><tr><td>w/o normalization</td><td>50.49</td><td>47.58</td></tr></table>

## 4.2 Training Dynamics

![](images/b1d4789cf6ff4afc8f2c8f648106c55bd2534070ea2e5486153951ca72ee73b3.jpg)

![](images/18062385377f5c8315f5fa67678edec79f5b08da5ba0213f6cd0527d93468b89.jpg)  
Figure 3: Training dynamics and validation performance over training steps. Left: Episode return mean. Right: Val14 R Score. Horizontal dashed lines indicate the BC, AIRL, and BC-RL baselines.

Figure 3 illustrates the training dynamics of DriveMLP trained using PPO on the nuPlan benchmark. The agent achieves rapid initial learning within the first 20M steps, surpassing the BC baseline early in training. The training reward continues to improve until plateauing around 80M steps, while the validation performance plateaus earlier at approximately 50M steps, ultimately surpassing the SOTA baseline.

## 4.3 Mitigating Distribution Shifts

To study the robustness benefits of closed-loop training, we evaluate DriveMLP and Diffusion Planner on nuPlan Val14 while injecting Gaussian noise into the actions passed to the vehicle dynamics model.

While this form of perturbation does not represent a general semantic distribution shift, it captures a critical and practically relevant failure mode in AD, where small execution errors can accumulate over time and lead to substantial trajectory drift.

Figure 4 summarizes performance under varying noise magnitudes. Although Diffusion Planner already leverages data augmentation during training to improve robustness to perturbations, DriveMLP exhibits a sub-

![](images/bf0ddc78a6783aec03de3223edb5a1c92bf36c345041fb7753b6bf3d46f014c5.jpg)  
Figure 4: NuPlan score on Val14 Non-Reactive (solid bars) and Reactive (striped bars) benchmark for DriveMLP (light blue) and Diffusion Planner (dark blue) with Gaussian noise on acceleration $( \sigma _ { \mathrm { a c c } } )$ and on steering $( \mu _ { \mathrm { s t e e r } } , \sigma _ { \mathrm { s t e e r } } )$

stantially smaller degradation in closed-loop performance across all noise settings. Appendix G provides a visual comparison of representative trajectories.

## 4.4 Data Efficiency Analysis

A key hypothesis of our approach is that RLbased exploration can compensate for limited expert demonstrations. Figure 5 illustrates performance as a function of training dataset size on nuPlan.

DriveMLP maintains strong performance even when trained on significantly reduced demonstration sets, indicating its ability to generate informative training data through exploration. This property is particularly valuable for long-tail and rare driving scenarios where expert data is scarce.

![](images/a015cac7995148ae442f6b57b943dc1f9948eb34c55e523945c095f9b3be490e.jpg)  
Figure 5: NuPlan score on Val14 Non-Reactive (solid bars) and Reactive (striped bars) benchmark for DriveMLP (light blue) and Diffusion Planner (dark blue) with reduced training dataset sizes.

## 4.5 Inference and Training Cost

Table 5 compares inference latency across methods. Thanks to its lightweight architecture, DriveMLP is significantly faster than all other SOTA methods, achieving > 15× speedup over Flow Planner while delivering higher Val14-R performance.

In terms of training, DriveMLP introduces additional data-collection overhead arising from closed-loop RL rollouts. It requires approximately 112 NVIDIA L4

Table 5: Inference speed (NVIDIA P6000), model size, and nuPlan Val14-R performance.
<table><tr><td>Method</td><td>Time (ms) ↓</td><td>Params (M) ↓</td><td>Val14-R ↑</td></tr><tr><td>PlanTF</td><td>8.30</td><td>2.02</td><td>76.95</td></tr><tr><td>PLUTO w/o refine</td><td>17.33</td><td>4.06</td><td>78.11</td></tr><tr><td>Diffusion Planner</td><td>72.34</td><td>6.04</td><td>82.80</td></tr><tr><td>Flow Planner</td><td>42.31</td><td>14.27</td><td>83.31</td></tr><tr><td>DriveMLP</td><td>2.76</td><td>2.33</td><td>86.51</td></tr></table>

GPU hours, ∼4.7× BC. For reference, large-scale IL models such as Diffusion Planner require approximately 240 NVIDIA A100 GPU-hours in our reproduced experiments. Normalizing by TFLOPS, DriveMLP achieves better NuPlan reactive performance with ${ \sim } 3  – 5 \times$ less compute than Diffusion Planner. Full hardware and software details including simulator throughput are provided in Appendix H.

## 5 Limitations

Like all demonstration-guided methods, the trajectory-distance reward anchors learning to the qual ity of the expert demonstrations. Nonetheless, our data-efficiency ablation (Section 4.4) shows that closed-loop RL exploration partially compensates for scarce demonstrations, maintaining strong performance even with substantially reduced datasets. Moreover, EGRL training requires access to a non-reactive log-replay simulator. While simpler than a fully reactive environment, such a simulator still requires a dynamic vehicle model for the ego.

Although nuPlan provides a high-fidelity closed-loop benchmark grounded in over 1,500 hours of real-world driving data [26], Sim2Real transfer for object-level planning policies remains challenging. While Sim2Real has been successfully demonstrated in other robotics domains [31, 32, 33], as well as for sensor-based driving systems [34], bridging this gap for object-level planners in AD has yet to be fully realized [12]. Several promising directions have been explored in prio work [35, 36, 37]. Given DriveMLP’s low inference latency and robustness, we see integrating it into such pipelines as a natural next step toward real-world deployment.

Appendix I illustrates representative failure cases including overshooting turns, late braking, and missed gap acceptance, providing further insight into the method’s current limitations.

## 6 Conclusion

We presented a simple yet effective approach for AD planning that combines the data efficiency of IL with the robustness of RL. By training RL agents in non-reactive simulation with trajectorybased rewards, we reduce the distribution shift problems of BC and the reward engineering burden of standard RL.

Our experiments demonstrate that EGRL (1) outperforms other training strategies such as BC, AIRL and BC-RL, (2) improves reactive performance of other SOTA models, and (3) DriveMLP scales to the challenging nuPlan benchmark with SOTA performance on the reactive evaluation while using a simpler architecture and achieving > 15× faster inference compared to the previous best IL model.

Additionally, DriveMLP exhibits superior data efficiency compared to the SOTA Diffusion Planner, maintaining strong performance even with significantly reduced demonstration sets. By inherently exposing the model to execution-time deviations during training, our method effectively mitigates the accumulation of small execution errors that typically lead to degraded performance of open-loop trained models in closed-loop settings.

We envision DriveMLP as an easy-to-implement, competitive baseline that encourages re-examining the community’s current emphasis on model complexity while highlighting the importance of training techniques.

## Acknowledgments

This work is a result of the joint research project STADT:up (19A22006O). The project is supported by the German Federal Ministry for Economic Affairs and Energy (BMWE), based on a decision of the German Bundestag. The authors are solely responsible for the content of this publication.

## References

[1] K. Muhammad, A. Ullah, J. Lloret, J. D. Ser, and V. H. C. de Albuquerque. Deep learning for safe autonomous driving: Current challenges and future directions. IEEE Transactions on Intelligent Transportation Systems, 22(7):4316–4336, 2021. doi:10.1109/TITS.2020.3032227.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In NeurIPS, 2017. URL https://arxiv.org/pdf/1706. 03762.pdf.

[3] J. Cheng, Y. Chen, X. Mei, B. Yang, B. Li, and M. Liu. Rethinking imitation-based planner for autonomous driving, 2023.

[4] J. Cheng, Y. Chen, and Q. Chen. Pluto: Pushing the limit of imitation learning-based planning for autonomous driving. arXiv preprint arXiv:2404.14327, 2024.

[5] Y. Zheng, R. Liang, K. ZHENG, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, and J. Liu. Diffusion-based planning for autonomous driving with flexible guidance. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=wM2sfVgMDH.

[6] T. Tan, Y. Zheng, R. Liang, Z. Wang, K. ZHENG, J. Zheng, J. Li, X. Zhan, and J. Liu. Flow matching-based autonomous driving planning with advanced interactive behavior modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=MQVyuWBfSH.

[7] P. de Haan, D. Jayaraman, and S. Levine. Causal confusion in imitation learning. NeurIPS, 2019.

[8] Y. Lu, J. Fu, G. Tucker, X. Pan, E. Bronstein, R. Roelofs, B. Sapp, B. White, A. Faust, S. Whiteson, D. Anguelov, and S. Levine. Imitation is not enough: Robustifying imitation with reinforcement learning for challenging driving scenarios. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7553–7560, 2023. doi: 10.1109/IROS55552.2023.10342038.

[9] M. Treiber, A. Hennecke, and D. Helbing. Congested traffic states in empirical observations and microscopic simulations. Physical Review E, 62:1805–1824, 02 2000. doi: 10.1103/PhysRevE.62.1805.

[10] W. B. Knox, A. Allievi, H. Banzhaf, F. Schmitt, and P. Stone. Reward (mis)design for autonomous driving. Artif. Intell., 316(C), Mar. 2023. ISSN 0004-3702. doi:10.1016/j.artint. 2022.103829. URL https://doi.org/10.1016/j.artint.2022.103829.

[11] P. R. Wurman, S. Barrett, K. Kawamoto, J. MacGlashan, K. Subramanian, T. J. Walsh, R. Capobianco, A. Devlic, F. Eckert, F. Fuchs, L. Gilpin, P. Khandelwal, V. Kompella, H. Lin, P. MacAlpine, D. Oller, T. Seno, C. Sherstan, M. D. Thomure, H. Aghabozorgi, L. Barrett, R. Douglas, D. Whitehead, P. Durr, P. Stone, M. Spranger, and H. Kitano. Outracing cham-¨ pion Gran Turismo drivers with deep reinforcement learning. Nature, 602(7896):223–228, Feb. 2022. doi:10.1038/s41586-021-04357-7.

[12] B. Jaeger, D. Dauner, J. Beißwenger, S. Gerstenecker, K. Chitta, and A. Geiger. CaRL: Learning scalable planning policies with simple rewards. In 9th Annual Conference on Robot Learning, 2025. URL https://openreview.net/forum?id=1otaE496Vm.

[13] M. F. Cusumano-Towner, D. Hafner, A. Hertzberg, B. Huval, A. Petrenko, E. Vinitsky, E. Wi jmans, T. W. Killian, S. Bowers, O. Sener, P. Kraehenbuehl, and V. Koltun. Robust autonomy emerges from self-play. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=yOXoJpG6Qy.

[14] J. Ho and S. Ermon. Generative adversarial imitation learning. CoRR, abs/1606.03476, 2016. URL http://arxiv.org/abs/1606.03476.

[15] J. Fu, K. Luo, and S. Levine. Learning robust rewards with adverserial inverse reinforcement learning. In International Conference on Learning Representations, 2018. URL https:// openreview.net/forum?id=rkHywl-A-.

[16] D. Garg, S. Chakraborty, C. Cundy, J. Song, and S. Ermon. Iq-learn: Inverse soft-q learning for imitation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 4028–4039. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/ 2021/file/210f760a89db30aa72ca258a3483cc7f-Paper.pdf.

[17] H. Zhao, X. Yu, D. M. Bossens, I. Tsang, and Q. Gu. Beyond-expert performance with limited demonstrations: Efficient imitation learning with double exploration. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=FviefuxmeW.

[18] D. Zhang, J. Liang, K. Guo, S. Lu, Q. Wang, R. Xiong, Z. Miao, and Y. Wang. Carplanner: Consistent auto-regressive trajectory planning for large-scale reinforcement learning in autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17239–17248, 2025. doi:10.1109/CVPR52734.2025.01607.

[19] Z. Peng, W. Luo, Y. Lu, T. Shen, C. Gulino, A. Seff, and J. Fu. Improving agent behaviors with RL fine-tuning for autonomous driving. In European Conference on Computer Vision (ECCV), 2024.

[20] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne. Deepmimic: example-guided deep reinforcement learning of physics-based character skills. ACM Trans. Graph., 37(4), July 2018. ISSN 0730-0301. doi:10.1145/3197517.3201311. URL https://doi.org/10.1145/ 3197517.3201311.

[21] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and A. J. Smola. Deep sets. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/ 2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf.

[22] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In ICML, volume 80 of Proceedings of Machine Learning Research, pages 1856–1865. PMLR, 2018.

[23] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://dblp.uni-trier.de/db/ journals/corr/corr1707.html#SchulmanWDRK17.

[24] E. Leurent. An environment for autonomous driving decision-making. https://github. com/eleurent/highway-env, 2018.

[25] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. M. Wolff, A. H. Lang, L. Fletcher, O. Beijbom, and S. Omari. Nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles. In CVPR ADP3 workshop, 2021.

[26] N. Karnchanachari, D. Geromichalos, K. S. Tan, N. Li, C. Eriksen, S. Yaghoubi, N. Mehdipour, G. Bernasconi, W. K. Fong, Y. Guo, and H. Caesar. Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

[27] D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta. Parting with misconceptions about learning-based vehicle motion planning. In Conference on Robot Learning (CoRL), 2023.

[28] M. Hallgarten, J. Zapata, M. Stoll, K. Renz, and A. Zell. Can vehicle motion planning generalize to realistic long-tail scenarios? In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5388–5395, 2024. doi:10.1109/IROS58592.2024. 10803052.

[29] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. In Proceedings of the 30th International Conference on Neural Information Processing Systems, NIPS’16, page 2234–2242, Red Hook, NY, USA, 2016. Curran Associates Inc. ISBN 9781510838819.

[30] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. Improved training of wasserstein gans. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[31] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Muller, V. Koltun, and D. Scaramuzza.¨ Champion-level drone racing using deep reinforcement learning. Nature, 620(7976):982–987, Aug 2023. ISSN 1476-4687. doi:10.1038/s41586-023-06419-4. URL https://doi.org/ 10.1038/s41586-023-06419-4.

[32] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics, 7(62):eabk2822, 2022. doi:10.1126/scirobotics.abk2822. URL https://www.science.org/doi/abs/10. 1126/scirobotics.abk2822.

[33] T. Lin, K. Sachdev, L. Fan, J. Malik, and Y. Zhu. Sim-to-real reinforcement learning for vision-based dexterous manipulation on humanoids. arXiv:2502.20396, 2025.

[34] M. Mueller, A. Dosovitskiy, B. Ghanem, and V. Koltun. Driving policy transfer via modularity and abstraction. In A. Billard, A. Dragan, J. Peters, and J. Morimoto, editors, Proceedings of The 2nd Conference on Robot Learning, volume 87 of Proceedings of Machine Learning Research, pages 1–15. PMLR, 29–31 Oct 2018. URL https://proceedings.mlr.press/ v87/mueller18a.html.

[35] N. Hanselmann, S. Doll, M. Cordts, H. P. Lensch, and A. Geiger. Emperror: A flexible generative perception error model for probing self-driving planners. IEEE Robotics and Automation Letters (RA-L), 2025. doi:10.1109/LRA.2025.3562789.

[36] D. Chen, B. Zhou, V. Koltun, and P. Krahenb¨ uhl. Learning by cheating. In L. P. Kaelbling,¨ D. Kragic, and K. Sugiura, editors, Proceedings of the Conference on Robot Learning, volume 100 of Proceedings ofMachine Learning Research, pages 66–75. PMLR, 30 Oct–01 Nov 2020. URL https://proceedings.mlr.press/v100/chen20a.html.

[37] X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li. Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In ICCV, 2023.

[38] D. Zeng, L. Zheng, X. Yang, and Y. Li. Reinforced imitative trajectory planning for urban automated driving, 2025. URL https://arxiv.org/abs/2410.15607. Accepted by Automotive Innovation.

[39] J. Booher, K. Rohanimanesh, J. Xu, V. Isenbaev, A. Balakrishna, I. Gupta, W. Liu, and A. Petiushko. CIMRL: Combining imitation and reinforcement learning for safe autonomous driving. arXiv preprint arXiv:2406.08878, 2024.

[40] Z. Zhang, P. Karkus, M. Igl, W. Ding, Y. Chen, B. Ivanovic, and M. Pavone. Closed-loop supervised fine-tuning of tokenized traffic models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[41] G. Cole, F. Justin, L. Wenjie, T. George, B. Eli, L. Yiren, H. Jean, P. Xinlei, W. Yan, C. Xiangyu, C.-R. John D., A. Rishabh, R. Rebecca, L. Yao, M. Nico, M. Paul, Y. Zoey, W. Brandyn, F. Aleksandra, M. Rowan, A. Dragomir, and S. Benjamin. Waymax: An accelerated, datadriven simulator for large-scale autonomous driving research. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2023.

[42] M. Harmel, A. Paras, A. Pasternak, N. Roy, and G. Linscott. Scaling is all you need: Autonomous driving with JAX-accelerated reinforcement learning. arXiv preprint arXiv:2312.15223, 2023.

[43] V. Charraut, T. Tournaire, W. Doulazmi, and T. Buhet. V-max: Making RL practical for autonomous driving. In Reinforcement Learning Conference, 2025. URL https: //openreview.net/forum?id=1nqgQ6Y5tC.

[44] M. Bansal, A. Krizhevsky, and A. Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst. In Proceedings ofRobotics: Science and Systems, 2019.

[45] M. Vecer´ık, T. Hester, J. Scholz, F. Wang, O. Pietquin, B. Piot, N. Heess, T. Rothorl, T. Lampe,¨ and M. A. Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards. CoRR, abs/1707.08817, 2017. URL http://arxiv.org/ abs/1707.08817.

[46] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. In Proceedings ofRobotics: Science and Systems (RSS), 2018.

[47] S. Fujimoto and S. Gu. A minimalist approach to offline reinforcement learning. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=Q32U7dzWXpc.

[48] S. Reddy, A. D. Dragan, and S. Levine. Sqil: Imitation learning via reinforcement learning with sparse rewards. arXiv preprint arXiv:1905.11108, 2019.

[49] B. D. Ziebart, A. Maas, J. A. Bagnell, and A. K. Dey. Maximum entropy inverse reinforcement learning. In Proceedings of the 23rd National Conference on Artificial Intelligence - Volume 3, AAAI’08, page 1433–1438. AAAI Press, 2008. ISBN 9781577353683.

[50] M. Wulfmeier, P. Ondruska, and I. Posner. Deep inverse reinforcement learning. CoRR, abs/1507.04888, 2015. URL http://arxiv.org/abs/1507.04888.

[51] I. Kostrikov, O. Nachum, and J. Tompson. Imitation learning via off-policy distribution matching. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=Hyg-JC4FDr.

[52] S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.

[53] E. Greensmith, P. L. Bartlett, and J. Baxter. Variance reduction techniques for gradient estimates in reinforcement learning. Journal of Machine Learning Research, 5:1471–1530, 2004.

[54] B. Piot, M. Geist, and O. Pietquin. Boosted and reward-regularized classification for apprenticeship learning. In Proceedings ofthe 2014 International Conference on Autonomous Agent and Multi-Agent Systems (AAMAS), 2014.

[55] E. Wijmans, A. Kadian, A. Morcos, S. Lee, I. Essa, D. Parikh, M. Savva, and D. Batra. Ddppo: Learning near-perfect pointgoal navigators from 2.5 billion frames. In ICLR, 2020. URL https://arxiv.org/pdf/1911.00357.

[56] Z. Wu, E. Liang, M. Luo, S. Mika, J. E. Gonzalez, and I. Stoica. RLlib flow: Distributed reinforcement learning is a dataflow problem. In Conference on Neural Information Processing Systems (NeurIPS), 2021. URL https://proceedings.neurips.cc/paper/2021/file/ 2bce32ed409f5ebcee2a7b417ad9beed-Paper.pdf.

[57] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal ofMachine Learning Research, 22 (268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html.

[58] A. Gleave, M. Taufeeque, J. Rocamonde, E. Jenner, S. H. Wang, S. Toyer, M. Ernestus, N. Belrose, S. Emmons, and S. Russell. imitation: Clean imitation learning implementations. arXiv:2211.11972v1 [cs.LG], 2022. URL https://arxiv.org/abs/2211.11972.

[59] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, May 1992. ISSN 1573-0565. doi: 10.1007/BF00992696. URL https://doi.org/10.1007/BF00992696.

[60] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In M. F. Balcan and K. Q. Weinberger, editors, Proceedings ofThe 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pages 1928–1937, New York, New York, USA, 20–22 Jun 2016. PMLR. URL https://proceedings.mlr.press/ v48/mniha16.html.

[61] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

[62] S. Sharma, A. S. Lakshminarayanan, and B. Ravindran. Learning to repeat: Fine grained action repetition for deep reinforcement learning. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=B1GOWV5eg.

[63] E. Wijmans, A. Kadian, A. Morcos, S. Lee, I. Essa, D. Parikh, M. Savva, and D. Batra. Ddppo: Learning near-perfect pointgoal navigators from 2.5 billion frames. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=H1gX8C4YPr.

[64] S. Kazemkhani, A. Pandya, D. Cornelisse, B. Shacklett, and E. Vinitsky. Gpudrive: Datadriven, multi-agent driving simulation at 1 million fps. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/ 2408.01584.

[65] M. Luo, J. Yao, R. Liaw, E. Liang, and I. Stoica. Impact: Importance weighted asynchronous architectures with clipped target networks. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=BJeGlJStPr.

[66] S. K. Lam, A. Pitrou, and S. Seibert. Numba: a llvm-based python jit compiler. In Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC, LLVM ’15, New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450340052. doi:10.1145/ 2833157.2833162. URL https://doi.org/10.1145/2833157.2833162.

## Appendix

## A Extended Related Work

This section provides further comparisons with related methods. Table 6 provides a structured comparison of DriveMLP against the most closely related methods, highlighting that while individual design choices appear in prior work, the specific combination of all components is novel to DriveMLP.

Table 6: Comparison of key design components across related methods. CL Train = closed-loop training; Demo Sup. = Supervision primarily provided by expert demonstrations (no traffic rules); No Adv. = no adversarial/iterative reward learning; Lightweight = ≤5M params, no generative or autoregressive architecture; No Post-Proc. = no rule-based post-processing at train or test time; Single Stage = Single training stage; nuPlan = evaluated on nuPlan or similar large-scale egoplanning benchmarks. ✓ = yes, × = no.
<table><tr><td>Method</td><td>CL Train</td><td>Demo Sup.</td><td>No Adv.</td><td>Lightweight</td><td>No Post-Proc.</td><td>Single Stage</td><td>nuPlan</td></tr><tr><td>PDM-Open [27] / PlanTF [3] / PLUTO [4] w/o refine</td><td>X</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Diffusion [5] / Flow Planner [6]</td><td>X</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td></td></tr><tr><td>AIRL [15] / GAIL [14]</td><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td><td>×</td></tr><tr><td>CaRL [12] / GIGAFLOW [13]</td><td>V</td><td>X</td><td>V</td><td>V</td><td>√</td><td>√</td><td></td></tr><tr><td>BC-RL [8]</td><td>√</td><td>√</td><td>V</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>CarPlanner [18]</td><td>√</td><td>V</td><td>√</td><td>×</td><td>×</td><td>√</td><td></td></tr><tr><td>Peng et al. [19]</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>RITP [38]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>X</td><td>√</td></tr><tr><td>CIMRL [39]</td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>CAT-K [40]</td><td>X</td><td>√</td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td></tr><tr><td>DriveMLP (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## A.1 RL Benchmarks and Baselines for Autonomous Driving

Several recent works have established RL baselines within AD simulation platforms. Waymax [41] provides a GPU-accelerated simulator built on the Waymo Open Dataset and includes DQN base lines with collision and off-road penalties; notably, these RL baselines underperform BC in thei evaluation. Scaling Is All You Need [42] demonstrates JAX-accelerated RL for AD but requires BC pre-training and 2.5 billion agent steps, using multi-component handcrafted rewards. V-Max [43] proposes an RL framework with reward functions encoding traffic rules such as speed limits, lanekeeping, and comfort. In all these cases, the supervision signal is fundamentally different from our single trajectory-distance reward, which does not encode any explicit traffic rules.

## A.2 Data Augmentation as an Alternative to Closed-Loop Training

ChauffeurNet [44] pioneered data augmentation via trajectory perturbations, a strategy adopted by recent methods including Diffusion Planner [5]. While augmentation is substantially cheaper per sample (it does not require querying the current policy), it generates fixed, pre-defined perturbation patterns. In contrast, RL exploration is adaptive to the current policy’s weaknesses. Our noise robustness analysis (Section 4.3) supports this: under significant action noise, DriveMLP maintains > 82 Val14-R while Diffusion Planner, which uses 50% augmented samples with perturbed-state recovery, drops to nearly 40.

## A.3 Supervised Fine-Tuning

Zhang et al. [40] propose Closest-Among-Top-K (CAT-K), which selects the best trajectory from K rollouts of a tokenized model. This approach differs fundamentally from ours: CAT-K operates in a discrete action space of pre-computed trajectories, uses a two-stage training pipeline relying on imaginary rollouts within a learned model, and employs an autoregressive transformer with a GMMparameterized policy. Furthermore, their ego-policy variant is not conditioned on route information, making direct comparison on nuPlan non-trivial.

## A.4 Other Hybrid Approaches

CIMRL [39] combines IL and RL with safety constraints. Their approach uses an IL pre-trained model to generate motion priors which are used to constrain RL exploration. Similarly, RITP [38] combines RL and IL by using a critic to score sampled candidate trajectories, training a transformerbased planner with imitation targets derived from the critic, and learning a Bayesian reward mode with uncertainty penalization. It further employs a hybrid post-optimization stage with rule-based trajectory refinement at test time that boosts performance by more than 60%.

In contrast, we directly optimize a lightweight control policy with standard RL without requiring pre-trained models or rule-based refinement.

## A.5 Other Related Approaches

DDPGfD [45] and DAPG [46] use demonstrations to warm-start RL but still rely on hand-designed task rewards. TD3+BC [47] regularizes offline RL with imitation losses but cannot use closed-loop rollouts to address distribution shift. SQIL [48] uses binary imitation rewards (r = 1 for expert data, r = 0 otherwise), providing no gradient information about the degree of proximity to expert states, whereas our formulation provides dense, distance-sensitive feedback. MaxEnt IRL [49], Deep Max-Ent IRL [50], and ValueDICE [51] require iterative reward learning or occupancy-matching objectives that remain computationally prohibitive at the scale of nuPlan.

## B Theoretical discussion

This section provides a formal analysis of why closed-loop training with our trajectory-distance reward addresses key limitations of open-loop BC. We draw connections to DAgger [52], inverse RL, and regularized BC, and analyze properties of the exponential reward formulation.

## B.1 Compounding Error in Open-Loop BC

Standard BC minimizes a single-step loss over expert demonstrations:

$$
\begin{array} { r } { J _ { \mathrm { B C } } ( \theta ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } \left[ \ell ( \pi _ { \theta } ( s ) , a ) \right] , } \end{array}\tag{5}
$$

where D contains state-action pairs drawn from the expert’s state distribution $d ^ { \pi ^ { * } }$ . At deployment, however, the policy $\pi _ { \theta }$ induces its own distribution $d ^ { \pi _ { \theta } }$ , which diverges from $d ^ { \pi ^ { * } }$ due to imperfect actions. Ross et al. [52] show that the resulting performance gap satisfies $J ( \pi ^ { * } ) - J ( \pi _ { \theta } ) \leq 2 T ^ { 2 } \epsilon _ { \mathrm { B C } }$ where T is the planning horizon and $\epsilon _ { \mathrm { B C } }$ is the per-step error under $d ^ { \pi ^ { * } }$ . The quadratic dependence on T is particularly severe for long planning horizons.

A further limitation is that the BC objective is entirely agnostic to the transition dynamics $\tau { : }$ actions are evaluated in isolation from their consequences. This means the learned value function (implicit or explicit) may assign arbitrary values to off-distribution states, providing no corrective signal when the agent drifts [48].

## B.2 On-Policy Optimization Mitigates the Distribution Mismatch

Our framework addresses the distribution shift problem by optimizing the policy through closedloop rollouts:

$$
J _ { \mathrm { C L } } ( \theta ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t } \right] ,\tag{6}
$$

where $\tau$ is generated by rolling out $\pi _ { \theta }$ in the non-reactive simulator and $r _ { t }$ is our trajectory-distance reward. The policy gradient is computed under $d ^ { \pi _ { \theta } }$ , meaning the policy receives learning signal at exactly the states it encounters during deployment. This reduces the distribution mismatch by construction, directly minimizing the performance gap rather than a surrogate loss under the wrong distribution.

## B.3 Connection to DAgger

DAgger [52] addresses BC’s distribution shift by collecting on-policy states from π<sub>θ</sub> and querying the expert $\pi ^ { * }$ for corrective labels, achieving a linear bound $J ( \pi ^ { * } ) - J ( \pi _ { \theta } ) \leq T \epsilon _ { \mathrm { D A g g e r } }$ Our method achieves an analogous benefit without interactive expert access. The key insight is that in the non-reactive simulation setting, surrounding agents follow their recorded trajectories, so the logged expert ego trajectory remains a consistent reference. The trajectory-distance reward $r _ { t } = $ $f ( \mathbf { \phi } \mathbf { \bar { \phi } } \mathbf { \bar { \phi } } \mathbf { \bar { \phi } } - \mathbf { \phi } p _ { t } ^ { \mathrm { e x p e r t } } \mathbf { \bar { \phi } } \mathbf { \phi } \mathbf { \phi } \mathbf { \bar { \phi } } )$ evaluates the policy’s realized position against this fixed reference at every timestep, providing dense per-step supervision under $d ^ { \pi _ { \theta } }$

## B.4 Relationship to Inverse RL

Adversarial IRL methods such as AIRL [15] optimize a divergence between the policy’s state occupancy measure $\rho ^ { \pi _ { \theta } }$ and the expert’s $\rho ^ { \pi ^ { * } }$ , learning the reward via a discriminator. Our trajectorydistance reward serves as an analytically defined surrogate for occupancy matching: it is maximized when $\rho ^ { \pi _ { \theta } }$ concentrates on states near the expert trajectory. In the limit where the reward sharpness $\alpha _ { 1 }  \infty$ , the positional reward $r _ { t } ^ { \mathrm { p o s } } = \mathrm { e x p } ( \bar { - } \alpha _ { 1 } \delta _ { t } \bar { / } ( \kappa _ { t } ^ { \mathrm { e x p \bar { \mathrm { e r t } } } } + \epsilon ) )$ reduces to an indicator ${ \mathbb H } [ \delta _ { t } = 0 ]$ recovering exact state matching. For finite $\alpha _ { 1 } ,$ , the reward provides a smooth, dense signal that acts as a soft surrogate for occupancy matching. This avoids the adversarial min-max optimization that makes GAIL and AIRL computationally expensive and unstable at scale.

Compared to SQIL [48], which assigns binary rewards (r = 1 for expert states, r = 0 otherwise), our formulation provides continuous gradient information proportional to the deviation magnitude. From a policy-gradient perspective, dense per-timestep rewards distribute learning signal across all rollout steps, reducing gradient variance compared to the sparse signal of SQIL, where most updates carry no reward information [53].

## B.5 Interpretation as Dynamics-Aware Regularized BC

Following the framework of Reddy et al. [48] and Piot et al. [54], who showed that RL with imitation rewards is equivalent to BC regularized by a Bellman-consistency prior, we observe that our closedloop objective can also be decomposed into two functional components: (1) an imitation term, where the trajectory-distance reward encourages actions that keep the policy close to the expert; and (2) a dynamics regularization term, where multi-step rollouts enforce that actions are evaluated by their consequences under the true transition dynamics T. The second component is entirely absent in BC, where action quality is assessed independently at each timestep.

## B.6 Properties of the Exponential Reward

The exponential formulation $r _ { t } ^ { \mathrm { p o s } } = \mathrm { e x p } ( - \alpha _ { 1 } \delta _ { t } / ( \kappa _ { t } ^ { \mathrm { e x p e r t } } + \epsilon ) )$ provides a more suitable reward landscape for trajectory tracking compared to the inverse distance $\tilde { r } _ { t } = { \kappa _ { t } ^ { \mathrm { e x p e r t } } } / ( \delta _ { t } + \epsilon )$ or the linear penalty $\tilde { r } _ { t } = - \delta _ { t }$ used by Peng et al. [19].

In driving scenarios with multiple valid trajectory modes (e.g., passing an obstacle on either side), a policy that averages between modes produces trajectories that follow neither. For two modes separated by distance $2 d ,$ the reward ratio between committing to one mode and averaging between them is $\exp ( \alpha _ { 1 } \cdot d / ( \kappa _ { t } ^ { \mathrm { e x p e r t } } + \epsilon ) )$ ) under our formulation. This ratio grows exponentially with mode separation, creating a strong incentive for the policy to commit to a single consistent trajectory. Under the inverse distance reward, the same ratio grows linearly, providing a weaker preference for commitment, while the linear reward $- \delta _ { t }$ , which is additionally unnormalized, leads to unstable optimization in practice.

Our empirical ablation supports this analysis: the exponential reward outperforms the inverse distance by +5.5 NR / +3.8 R points and the unnormalized linear penalty by +39.9 NR / +38.9 R points on nuPlan Val14 (Table 4).

## C Experimental Setup

## C.1 Implementation Details

To enable scalable distributed RL training [55] on nuPlan we employ the Ray RLlib [56] implementation of PPO. For Highway-Env, we use SAC [57] to demonstrate the generality of our approach across different RL algorithms. BC and AIRL baselines are implemented using the Imitation library [58], with the same model architectures as our method.

During training, the policy is executed at 1 Hz by repeating each selected action for 10 consecutive simulation steps. This action repetition promotes greater diversity of state transitions within each training batch (see Section F for an ablation). Evaluation is carried out at 10 Hz strictly following nuPlan settings.

The ego transition model follows nuPlan’s default kinematic bicycle model, using a Numbaaccelerated implementation during training.

PPO improves upon strict on-policy methods (e.g., REINFORCE [59] and A2C [60]) by enabling multiple gradient updates per iteration while using clipping to control policy divergence. We perform approximately 2,000 updates per iteration, substantially improving sample reuse and data efficiency.

We train our approach for approximately 100K environment steps on Highway-Env and for 100M environment steps on nuPlan. We set the discount factor $\gamma = 0 . 7$ , which corresponds to a planning horizon of approximately 3.3 seconds, similar to the horizon used by other planners.

## C.2 Evaluation Metrics

For Highway-Env, we report: (1) mean cumulative reward, (2) mean episode length, and (3) termi nation rate due to collisions or off-road events. All methods are evaluated over 100 episodes in the reactive simulator, with results averaged across three random seeds.

For nuPlan, we follow the official closed-loop evaluation protocol [25] and report the overall score under both reactive and non-reactive settings for three standard splits: Val14 [27], Test14-Hard [3], and Test14 [3].

## C.3 Hyperparameters

Table 7 reports the main training hyperparameters used in the experiments. Hyperparameters not reported here use the default value of the library implementation.

Table 7: Hyperparameters used in Nuplan and Highway env experiments
<table><tr><td rowspan="2">Name</td><td colspan="4">Highway-Env</td><td colspan="4">nuPlan</td></tr><tr><td>BC</td><td>AIRL</td><td>BC-RL</td><td>DriveMLP</td><td>BC</td><td>AIRL</td><td>BC-RL</td><td>DriveMLP</td></tr><tr><td>Algorithm</td><td>BC</td><td>AIRL (SAC)</td><td>BC+SAC</td><td>SAC</td><td>BC</td><td>AIRL (PPO)</td><td>BC+PPO</td><td>PPO</td></tr><tr><td>Training steps</td><td></td><td>300K</td><td>100K</td><td>100K</td><td></td><td>100M</td><td>100M</td><td>100M</td></tr><tr><td>Learning rate</td><td>0.001</td><td>0.0003</td><td>0.0003</td><td>0.0003</td><td>0.0001</td><td>0.00005</td><td>0.00005</td><td>0.00005</td></tr><tr><td>Learning rate schedule</td><td>exponential decay</td><td></td><td></td><td></td><td>exponential decay</td><td>cosine decay</td><td>cosine decay</td><td>cosine decay</td></tr><tr><td>Rollout buffer size</td><td></td><td>1M</td><td>1M</td><td>1M</td><td></td><td>98K</td><td>98K</td><td>98K</td></tr><tr><td>Batch size</td><td>64</td><td>256</td><td>256</td><td>256</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Epochs</td><td>80</td><td></td><td></td><td>一</td><td>100</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Discount factor</td><td></td><td>0.7</td><td>0.7</td><td>0.7</td><td></td><td>0.7</td><td>0.7</td><td>0.7</td></tr><tr><td>Gae lambda</td><td></td><td></td><td>=</td><td></td><td></td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>BC/RL update ratio</td><td></td><td>1</td><td>8</td><td></td><td></td><td>=</td><td>8</td><td></td></tr><tr><td>Discriminator updates</td><td></td><td>4</td><td></td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td>Discriminator learning rate</td><td></td><td>1e-2</td><td></td><td></td><td></td><td>3e-5</td><td></td><td></td></tr></table>

Unlike traditional rule-based rewards in AD which often involve conflicting objectives $( \mathrm { e . g . }$ progress vs. safety), our trajectory-based reward terms are inherently aligned. Both $\dot { r } _ { t } ^ { p o s }$ and $r _ { t } ^ { v e l }$ are maximized at the expert’s state, meaning they share the same global optimum. Consequently, the hyperparameters $\alpha _ { 1 } , \alpha _ { 2 }$ , and $\alpha _ { 3 }$ function as scaling factors that regulate the smoothness of the reward surface rather than shifting the task’s objective. We used the same values across all our experiments: $\alpha _ { 1 } = 1 / 3 , \alpha _ { 2 } = 1 / 4 , \alpha _ { 3 } = 0 . 7 .$ We verified that the proposed reward is not sensitive to scaling: setting $\scriptstyle \alpha _ { 1 } = \alpha _ { 2 } = 1$ , and removing $\alpha _ { 3 }$ yields a Val14-R score of 85.87. The terminal penalty value is $r _ { p } = - 1$ . The deviation-based terminal condition is triggered when ego velocity is not within a 100% tolerance margin of the expert velocity, namely when $| v _ { t } ^ { \mathrm { e g o } } / v _ { t } ^ { \mathrm { e x p e r t } } - 1 | > \overline { { 1 . 0 } }$ for $v _ { t } ^ { \mathrm { e x p e r t } } > 1 m / s$

## C.4 Datasets

For Highway-env, we construct an expert demonstration set by rolling out the built-in IDM policy in the reactive simulator for 1000 episodes of 30s each. We discard episodes that terminate due to collisions or off-road events, yielding approximately 750 optimal trajectories used for training.

For nuPlan, we follow standard preprocessing protocols to enable direct comparison with prior work. For open-loop BC, we use the same scenario extraction procedure as Diffusion Planner [5], resulting in 1M training scenarios. For closed-loop training, we adopt the CaRL preprocessing setup [12]: we select all temporally non-overlapping scenarios of 15s duration from the nuPlan training split that contain valid route information, resulting in approximately 270K scenarios.

## D InterPlan Results

We additionally evaluate on InterPlan [28], which augments nuPlan scenarios with challenging longtail interactions (Table 8). DriveMLP exceeds the reported PlanTF and Diffusion Planner scores on both splits. On the official 80-scenario split, its mean score is close to Flow Planner’s but below PLUTO’s; on the full 335-scenario set, it is close to PLUTO’s and below Flow Planner’s. These results provide additional evidence of transfer to challenging interactive scenarios.

Table 8: Interplan benchmark results. <sup>†</sup>PLUTO incorporates centerline reference paths as an additional input.
<table><tr><td>Split</td><td>PlanTF</td><td>PLUTO w/o ref.†</td><td>Diff. Planner</td><td>Flow Planner</td><td>DriveMLP</td></tr><tr><td>80 scenarios</td><td>30.53</td><td>49.27</td><td>24.71</td><td>36.59</td><td> $3 7 . 0 7 { \scriptstyle \pm 2 . 3 6 }$ </td></tr><tr><td>335 scenarios</td><td>47.70</td><td>58.47</td><td>52.90</td><td>61.82</td><td>58.43±2.74</td></tr></table>

## E Model Capacity Analysis

This section complements the backbone comparisons in Section 3.3 by examining MLP capacity scaling and a reduced transformer policy.

## E.1 MLP Capacity Scaling

To examine the effects of model capacity and training strategy, we vary the capacity of the MLP architecture used by DriveMLP from 112K to 8.7M parameters, comparing EGRL with BC at each capacity. Table 9 reports reactive performance, improvements over BC, inference latency, and training cost.

Table 9: MLP capacity scaling under EGRL. Improvements over BC are reported in score points.
<table><tr><td>Size</td><td>112K</td><td>247K</td><td>590K</td><td>1.1M</td><td>2.3M</td><td>4.6M</td><td>8.7M</td></tr><tr><td>Val14-R</td><td>82.22</td><td>84.39</td><td>84.96</td><td>85.74</td><td>86.51</td><td>86.18</td><td>86.33</td></tr><tr><td>Δ vs. BC</td><td>+53.31</td><td>+54.54</td><td>+53.20</td><td>+53.70</td><td>+54.15</td><td>+53.85</td><td>+54.22</td></tr><tr><td>Infer. (ms)</td><td>1.86</td><td>1.95</td><td>2.27</td><td>2.39</td><td>2.76</td><td>2.99</td><td>3.47</td></tr><tr><td>Train (L4-h)</td><td>56</td><td>60</td><td>74</td><td>87</td><td>112</td><td>155</td><td>185</td></tr></table>

Increasing capacity under BC yields marginal gains, whereas EGRL improves reactive performance over BC by approximately 53–55 points across the tested capacities. EGRL performance increases up to the 2.3M-parameter model, while larger models provide no further improvement in the observed score despite higher inference latency and training cost.

## E.2 Reduced Transformer Policy

Starting from the Diffusion Planner encoder [5], we reduce the policy capacity to approximately 500K parameters to obtain inference latency comparable to DriveMLP. We use the encoder and control-head adaptations described in Section 3.3, without the diffusion generation process or auxiliary multi-agent prediction objective. As shown in Table 10, EGRL achieves approximately +21/+47 (NR/R) points over the BC baseline. This provides additional evidence that the benefits extend beyond MLP policies, including a reduced transformer architecture.

Table 10: Reduced transformer policy (500K parameters) on nuPlan Val14. NR = Non-Reactive, R = Reactive simulation.
<table><tr><td>Method</td><td>NR</td><td>R</td></tr><tr><td>BC</td><td>61.25</td><td>32.44</td></tr><tr><td>EGRL</td><td>82.01</td><td>79.68</td></tr></table>

## F Ablations and Parameter Analysis

## F.1 Reward Components and Hyperparameter Sensitivity

Table 11 summarizes the effects of removing individual episode termination conditions. These conditions improve performance while restricting rollouts to safe and approximately log-consistent states. Collision termination prevents rollouts from continuing through physically invalid vehicle overlaps, while off-road termination restricts exploration to the drivable area. Deviation termination limits exploration in states where interactions with surrounding agents, which continue to replay their logged trajectories independently of the ego, may become inconsistent with the recorded scenario. This differs from constructing a dense reward by weighting traffic-rule objectives such as speed-limit compliance, comfort, lane keeping, and progress. Such objectives often require contextdependent trade-offs. For example, yielding to an emergency vehicle may require leaving the current lane or, where legally permitted, proceeding past a red light; comfort may also need to be subordinated to the maneuver. Encoding these exceptions can require scenario-specific reward engineering and weighting. Our approach instead derives dense behavioral supervision from expert trajectories, while retaining boolean termination conditions to constrain training rollouts.

Table 11: Episode termination ablations for DriveMLP on nuPlan Val14.
<table><tr><td>Component</td><td>NR</td><td>R</td></tr><tr><td>Base</td><td>90.38</td><td>86.51</td></tr><tr><td>w/o off-road</td><td>80.76</td><td>78.67</td></tr><tr><td>w/o collision</td><td>80.08</td><td>79.14</td></tr><tr><td>w/o deviation</td><td>86.92</td><td>84.11</td></tr></table>

In Table 12 we analyze the sensitivity to the discount factor and in Table 13 to the terminal penalty. As reported by [12], AD rewards exhibit significant sensitivity to mini-batch size variations with benchmark scores dropping near zero. Table 14 shows that our trajectory-based reward formulation is substantially more robust to this hyperparameter.

## F.2 Action Repetition Ablation

We train the policy using the nuPlan simulator at its native 10 Hz control frequency, but execute actions at a reduced rate of 1 Hz by repeating each selected action for 10 consecutive simulation steps. At evaluation time, we use the standard nuPlan setup and run the policy at 10 Hz without action repetition (i.e., each action is applied once per simulator step).

This action repetition introduces a trade-off between spatial coverage and control fidelity. Lower control frequency increases the displacement between consecutive decisions, broadening the state distribution within each training batch. It also reduces the effective RL horizon (e.g., 15 steps for a 15-second scenario), which simplifies credit assignment and improves training stability, analogous to frame-skipping in Atari [61]. Conversely, higher control frequencies more closely match the 10 Hz evaluation setting. Table 15 compares performance at 0.5, 1, and 2 Hz, showing that 1 Hz provides the best balance.

Table 12: nuPlan Val14 score at varying magnitude of the discount factor.  
Table 13: nuPlan Val14 score at varying magnitude of the terminal penalty.
<table><tr><td>Discount factor</td><td>: R Score</td></tr><tr><td> $\gamma = 0 . 6$ </td><td>85.08</td></tr><tr><td> $\gamma = 0 . 7$ </td><td>86.51</td></tr><tr><td> $\gamma = 0 . 8$ </td><td>84.48</td></tr></table>

Table 14: nuPlan Val14 score at varying mini-batch size.
<table><tr><td>Terminal penalty</td><td>R Score</td></tr><tr><td> $r _ { p } = - 0 . 5$ </td><td>85.57</td></tr><tr><td> $r _ { p } = - 1$ </td><td>86.51</td></tr><tr><td> $r _ { p } = - 2$ </td><td>84.39</td></tr></table>

<table><tr><td>Mini-batch size</td><td>R Score</td></tr><tr><td>128</td><td>84.02</td></tr><tr><td>512</td><td>86.51</td></tr><tr><td>2048</td><td>82.49</td></tr></table>

Table 15: Action frequency ablation: nuPlan Val14.
<table><tr><td>Frequency</td><td>NR</td><td>R</td></tr><tr><td>0.5 Hz</td><td>75.23</td><td>77.61</td></tr><tr><td>1 Hz</td><td>90.38</td><td>86.51</td></tr><tr><td>2 Hz</td><td>85.13</td><td>82.28</td></tr></table>

While our uniform 1 Hz repetition strategy proves highly effective, a promising future direction is to dynamically parameterize this temporal abstraction using frameworks like FiGAR [62]. Incorporating variable-rate repetition could allow the policy to adaptively scale its control frequency, deploying high-frequency adjustments during critical maneuvers while maintaining a sparse, sample-efficient temporal horizon in nominal driving scenarios.

## F.3 Ego-state Ablation

Table 16 reports an ego-state ablation. Ego-state information improves DriveMLP by +11 NR points, but hurts open-loop trained planners: both the BC MLP and BC Transformer degrade when ego-state features are included, consistent with Diffusion Planner’s reported performance drop of more than 10 points [5]. This is consistent with the hypothesis that closed-loop RL training better captures the interplay between ego dynamics and planning decisions, whereas open-loop BC may learn spurious correlations from ego-state features.

Table 16: Ego-state ablation: nuPlan Val14 NR.
<table><tr><td>Architecture</td><td>Training Method</td><td>w/o Ego State</td><td>w/ Ego State</td></tr><tr><td>DriveMLP</td><td>BC</td><td>54.90</td><td>53.64</td></tr><tr><td>Transformer</td><td>BC</td><td>61.25</td><td>54.76</td></tr><tr><td>DriveMLP</td><td>EGRL</td><td>78.79</td><td>90.38</td></tr></table>

## G Noise Robustness Visualization

Figure 6 provides a qualitative comparison of representative trajectories. Despite the injected noise, our method produces trajectories that remain closely aligned with the expert, whereas Diffusion Planner exhibits a gradual drift away from the reference trajectory.

![](images/368ddf0a50a1cf63bd6efa742491f2ed7cad88acd0c67a8ba69bc14d646fce37.jpg)  
(a)

![](images/c918e09304c2dea44506d4d0b5dc608997dc289fabed14907c7103a771f67017.jpg)  
(b)

![](images/453fa5300eb062b050fe44a7339075eb7ee6ef586fc25f595c4ba2e791f51b7b.jpg)  
(c)

![](images/773f75380410ef9d04dc11f99c7549700e507d4bb80c7d18253cc49b87d5270c.jpg)  
(d)  
Figure 6: Noise brings the Diffusion Planner close to a neighboring agent, with no corrective re-<sup>Executed</sup> <sub>trajectory</sub>sponse to avoid the collision (a), whereas DriveMLP remains close to the expert trajectory (b). Similarly, noise can cause the Diffusion Planner to miss a turn (c), while our planner remains largely unaffected (d).

## Expert H Reproducibility and Simulation Infrastructure

Hardware and Software. We used Python 3.9 together with PyTorch 2.7.1 and CUDA 12.6 for GPU acceleration. All nuPlan experiments were run on a single compute node with 4 NVIDIA L4 GPUs and 48 CPU cores.

Training Parallelism and Runtime. For the nuPlan experiments, we parallelize rollouts across 45 environments using Ray’s asynchronous vectorization interface, which is well suited to nonhomogeneous environments [63]. Training runs for 100M environment steps and completes in approximately 30 hours end-to-end; with our current setup, PPO spends less than half its wall-clock time on data collection. This is made feasible by a domain-specific simulation runtime that achieves more than 100× higher throughput than a naive nuPlan wrapper with vectorized observation features (e.g., CaRL nuPlan Gym Env. with Diffusion Planner observation builder), as shown in Table 17. As recent progress in GPU-accelerated simulation has shown [64], and through asynchronous policy optimization such as APPO [65], there is significant headroom for further speedup.

Table 17: Single env. reset/step functions performance
<table><tr><td>Env</td><td>Reset</td><td>Step</td></tr><tr><td>Naive stack (ms)</td><td>559.4</td><td>366.27</td></tr><tr><td>Ours (ms)</td><td>3.9</td><td>1.59</td></tr><tr><td>Speedup</td><td>143x</td><td>230x</td></tr></table>

Simulation Optimizations. The speedup comes from five main engineering changes. (1) We replace the generic nuPlan/CaRL orchestration stack with a domain-specific runtime optimized for low-latency stepping. This removes unnecessary planner-state reconstruction, bypasses generalpurpose map interfaces, and reduces the reset and step overhead. (2) Expensive scenario parsing and map preprocessing are moved offline. During training, the simulator operates on preconverted scenarios and cached map artifacts, avoiding repeated reconstruction of runtime state. (3) Pointerheavy Python object structures are replaced with contiguous NumPy buffers, which improve memory locality and reduce allocation overhead. (4) Online retrieval operations use spatial indexing and candidate pruning instead of full-set sorting. Broad-phase culling is applied before exact geometric checks. (5) Latency-critical components, including lane ranking, coordinate transforms, vehicle propagation, collision checks, and off-road detection, are implemented using batched linear algebra and Numba-JIT kernels [66].

Despite these optimizations, the entire simulation stack remains implemented in pure Python, with out C/C++ extensions, preserving ease of modification and integration into existing research workflows.

Inference Benchmarking Protocol. Inference latency in Table 5 is measured on an Intel Xeon 6136 CPU and an NVIDIA Quadro P6000 GPU. All methods are evaluated with the same batch size and timing procedure under identical conditions.

## I Failure CasesExpert

Despite the overall improvements, challenges remain. Figure 7 illustrates representative failure scenarios that reveal current limitations and opportunities for future development. The observed failure modes include overshooting turns and invading the opposite lane, failure to progress through narrow time windows at intersections, side collisions with other vehicles, and late braking for obstacles.

![](images/64e47536e863b5ea18b298e0379809040864283524b1a3c6b09f043ca2cceb39.jpg)  
(a)

![](images/2100b5d24aa68c6695c21813d0740f3f53bca9403f3ef35ea7088f1f247e5cd9.jpg)  
(b)

![](images/c0d459132eab51d0c435238023d44c3dbdd5cc0b3dbbd18f6a27eb2f2cfd072d.jpg)  
(c)

![](images/05d2322e6bd1940abc80c9e0db8b57a074c2dc060b8a3e3a849e0cf022026db6.jpg)  
(d)  
Figure 7: Representative failure cases of DriveMLP. Expert trajectory is shown in orange, while the trajectory executed by ego is in blue. Failure modes include (a) overshooting turns and invading the opposite lane, (b) failure to progress due to a limited time window in which the intersection is clear, (c) side collisions with other vehicles, and (d) late braking for obstacles in the lane. All scenarios are taken from the Val14-Reactive evaluation split.

Table 18 provides a per-metric and per-scenario breakdown comparing DriveMLP against Diffusion Planner [5] (reproduced) on Val14 and Test14-Hard. DriveMLP surpasses Diffusion Planner on TTC and Drivable Area Compliance across all splits, but lags slightly behind on Driving Direction Compliance.

The Starting Left Turn scenario is not present in the nuPlan training distribution, making it a natural test of out-of-distribution generalization. DriveMLP matches Diffusion Planner’s almost exactly: 88.55/74.08 (NR/R) vs. 86.71/72.49 on Val14, and 65.36/72.21 vs. 65.73/67.01 on Test14-Hard. This parity indicates that model complexity alone does not determine generalization to unseen maneuvers: a lightweight policy trained in closed-loop can achieve similar out-of-distribution performance to a significantly more complex generative model.

On Val14-Reactive, DriveMLP also exceeds the score obtained by replaying the logged expert trajectory (86.51 vs. 82.02).

Table 18: Performance Comparison: DriveMLP vs. Diffusion Planner across nuPlan Splits
<table><tr><td rowspan="3">Metrics</td><td colspan="4">DriveMLP</td><td colspan="4">Diffusion Planner</td></tr><tr><td colspan="2">Val14</td><td colspan="2">Test-Hard</td><td colspan="2">Val14</td><td colspan="2">Test-Hard</td></tr><tr><td colspan="2">NR R</td><td colspan="2">NR R</td><td colspan="2">NR R</td><td colspan="2">NR R</td></tr><tr><td></td><td></td><td>86.51</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Overall Score</td><td>90.39</td><td>99.82</td><td>76.69</td><td>74.51</td><td>89.86 97.94</td><td>82.95 97.85</td><td>75.73 95.59</td><td>68.94</td></tr><tr><td>Drivable Area Compliance</td><td>99.82</td><td>99.42</td><td>97.43</td><td>97.79</td><td>99.51</td><td></td><td></td><td>94.85</td></tr><tr><td>Driving Direction Comp.</td><td>99.37</td><td>88.10</td><td>97.79</td><td>97.98 87.13</td><td>92.49</td><td>99.82 89.36</td><td>98.90 91.91</td><td>98.90 86.40</td></tr><tr><td>Ego is Comfortable Ego is Making Progress</td><td>87.39 98.93</td><td>98.57</td><td>86.03 95.22</td><td>94.49</td><td>99.91</td><td>96.69</td><td>98.53</td><td>92.28</td></tr><tr><td>Ego Progress vs Expert</td><td>91.27</td><td>88.02</td><td>82.40</td><td>77.19</td><td>95.29</td><td>85.98</td><td>89.77</td><td>76.60</td></tr><tr><td>No At-Fault Collisions</td><td>97.27</td><td>94.45</td><td>92.10</td><td>90.26</td><td>96.20</td><td>93.11</td><td>87.68</td><td>86.21</td></tr><tr><td>Speed Limit Compliance</td><td>97.75</td><td>98.28</td><td>97.03</td><td>97.18</td><td>96.52</td><td>98.29</td><td>96.44</td><td>97.39</td></tr><tr><td>TTC Within Bound</td><td>94.28</td><td>89.71</td><td>83.09</td><td>83.82</td><td>91.59</td><td>88.01</td><td>77.57</td><td>80.51</td></tr><tr><td>Scenario Scores</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Behind Long Vehicle</td><td>99.17</td><td>99.60</td><td>90.63</td><td>91.14</td><td>99.13</td><td>98.58</td><td>91.08</td><td>86.42</td></tr><tr><td>Changing Lane</td><td>89.07</td><td>88.80</td><td>73.27</td><td>73.59</td><td>85.64</td><td>84.57</td><td>64.79</td><td>76.63</td></tr><tr><td>Following Lane w/ Lead</td><td>97.33</td><td>88.74</td><td>88.84</td><td>89.42</td><td>95.65</td><td>80.70</td><td>90.36</td><td>79.69</td></tr><tr><td>High Lateral Accel.</td><td>84.39</td><td>83.90</td><td>59.64</td><td>58.25</td><td>85.71</td><td>83.26</td><td>58.44</td><td>51.05</td></tr><tr><td>High Magnitude Speed</td><td>96.13</td><td>93.40</td><td>91.31</td><td>84.62</td><td>92.84</td><td>90.54</td><td>88.18</td><td>78.94</td></tr><tr><td>Low Magnitude Speed</td><td>88.55</td><td>88.67</td><td>47.25</td><td>54.91</td><td>91.36</td><td>82.98</td><td>50.75</td><td>48.01</td></tr><tr><td>Near Multiple Vehicles</td><td>95.01</td><td>83.69</td><td>77.52</td><td>60.06</td><td>95.47</td><td>83.33</td><td>66.74</td><td>50.52</td></tr><tr><td>Starting Left Turn</td><td>88.55</td><td>74.08</td><td>65.36</td><td>72.21</td><td>86.71</td><td>72.49</td><td>65.73</td><td>67.01</td></tr><tr><td>Starting Right Turn</td><td>83.40</td><td>76.38</td><td>69.79</td><td>57.30</td><td>84.07</td><td>68.61</td><td>67.27</td><td>56.11</td></tr><tr><td>Intersection Traversal</td><td>93.98</td><td>87.66</td><td>83.93</td><td>80.64</td><td>93.65</td><td>82.15</td><td>88.56</td><td>89.07</td></tr><tr><td>Stationary in Traffic</td><td>97.54</td><td>94.72</td><td>96.47</td><td>83.11</td><td>98.17</td><td>93.42</td><td>97.66</td><td>71.52</td></tr><tr><td>Stopping w/ Lead</td><td>96.11</td><td>98.47</td><td>98.19</td><td>98.12</td><td>95.34</td><td>94.61</td><td>99.07</td><td>97.61</td></tr><tr><td>Traversing Pickup/Dropoff</td><td>78.94</td><td>80.00</td><td>67.38</td><td>68.96</td><td>76.29</td><td>75.76</td><td>59.41</td><td>70.03</td></tr><tr><td>Waiting for Pedestrian</td><td>91.88</td><td>87.92</td><td>69.20</td><td>76.97</td><td>91.48</td><td>83.30</td><td>78.14</td><td>47.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>