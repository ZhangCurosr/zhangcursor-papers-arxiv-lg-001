# Efficient Test-Time Adaptation For Robot Policies

Motasem Alfarra, Pietro Mazzaglia, Markus Peschl, Daniel Dijkman, Christos Louizos Qualcomm AI Research, Amsterdam, the Netherlands malfarra@qti.qualcomm.com

## Abstract

Reinforcement-learning policies for robot locomotion can fail at deployment when operating conditions shift (e.g., rough or slippery terrain, payload changes, sensing degradation), and retraining or manual recalibration is often infeasible under on-device latency and memory constraints. We study test-time adaptation (TTA) for robotic control: adapting a pretrained policy online and self-supervised during execution, without access to rewards. Our starting point is a lightweight baseline that updates only the actor using a frozen pretrained critic as a deployment-time performance surrogate. Building on this, we introduce TEMPO, a test-time entropyregularized, memory-guided policy optimization method that stabilizes critic-driven updates under severe shift. TEMPO combines (i) an entropy-promoting regularizer on the critic’s predictive distribution to mitigate overconfident extrapolation, and (ii) a score-based memory, inspired by prioritized replay, that prioritizes low-value (“hard”) observations and applies importance sampling to focus limited computation on the most informative deployment conditions. Across Go1, Go2, and G1 in MuJoCo Playground under four representative shifts and multiple architectures, TEMPO yields consistent gains, reaching up to 80% improvement in a single-device setting. We further validate on a physical Unitree Go2 with an added calf payload, where fully on-device adaptation improves performance over the non-adapted baseline across real-world episodes.

## 1 Introduction

Robots deployed in the real world must contend with operating conditions that can diverge sharply from those encountered during training [Kumar et al., 2021]. These discrepancies arise from multiple sources: terrain variability (e.g., rocks, low-friction surfaces) [Sun et al., 2025], hardware changes (e.g., localized payloads or partial actuator degradation) [Pouroullis et al., 2025], emergent situations (e.g., a weight stuck on one arm/calf) [Zhang et al., 2025, Chen et al., 2023], and the broader sim-toreal gap between simulated and physical environments [Peng et al., 2018]. In practice, such factors rarely occur in isolation: a deployed robot may simultaneously experience altered contact dynamics, sensor noise, actuator wear, and previously unseen environmental configurations. These distribution shifts can substantially degrade policies that perform well under nominal conditions (refer to Figure 1 left), making online adaptation central to robust autonomy.

A common response to combat this problem is to apply large-scale domain randomization during reinforcement learning to improve zero-shot generalization [Tobin et al., 2017, Peng et al., 2018]. While effective in some cases, this approach is fundamentally constrained by what can be specified in advance: it presumes one can enumerate relevant sources of variability, model them with sufficient fidelity in simulation, and collect enough data to cover their interactions. When robots encounter truly novel conditions, adaptation is performed offline. This interrupts operation to gather new data and fine-tune from scratch, which is misaligned with real-time, on-device compute and reliability constraints. Architectural add-ons for adaptation can help but often increase system complexity and reduce portability across tasks, embodiments, and deployment settings [Kumar et al., 2021].

![](images/fb7fc6b8de8ad7b8ddd033b3eb52a47da5c10fa2a70df22ef27f63503c8317c0.jpg)

![](images/1e909ab768794a4b8d536a54ac96383619f080f963c5abd369a6362594c464bc.jpg)

![](images/3940ac0f9fc6e5f19ca42a3116e4de608deb6b3667ac4e4a023b347b9d323872.jpg)  
Figure 1: Motivation and pipeline. Left: Pretrained locomotion policies degrade as shift severity increases. Middle: Off-the-shelf TTA and prior RL adaptation baselines fail under a rocky-terrain shift. Right: TEMPO loop: execute π<sub>θ</sub> , update memory B, and periodically adapt via Eq. (2).

We study Test-Time Adaptation (TTA) [Wang et al., 2020, Sun et al., 2020] as a practical alternative for robotic control under distribution shift. TTA is well-suited for tackling this problem as it adapts the pretrained model at test time in an online and self-supervised manner, alleviating the need to disrupt the inference stream for retraining and requiring no labeled data. Thus, our starting point is a simple, general baseline: we update only the actor (policy) parameters at deployment while freezing the critic, using a compact on-device memory of recent observations. At periodic intervals during execution, the actor is adapted by maximizing the critic’s estimated action value on stored observations—treating the critic as a learned performance surrogate and requiring no online rewards. Unlike standard deployments where the critic is used primarily for training and often discarded at inference, here we actively leverage the pretrained critic during runtime. While distribution shift also affects the critic, we find it still provides a sufficiently informative signal to guide test-time actor updates [Chen et al., 2023]; nonetheless, naive value maximization can overfit to spurious, overconfident estimates [Kumar et al., 2020] and under-explore the loss landscape.

In this work, and to address these limitations, we introduce an enhanced framework with two complementary elements. First, we add an entropy-promoting regularizer on the critic’s predictive distribution during adaptation, which tempers overconfidence under shift and biases updates toward regions that are simultaneously high-value and high-uncertainty. This yields more informative gradients due to the stability and exploration benefits of maximum-entropy objectives in RL [Eysenbach and Levine, 2021], which we adapt to the critic’s uncertainty at test time. Second, and in the spirit of prioritized experience replay [Schaul et al., 2016], we employ a score-based memory that prioritizes observations inversely proportional to the critic’s predicted value, coupled with importance sampling to focus updates on the most informative deployment conditions under tight memory and latency budgets. We evaluate on Go1, Go2, and G1 in Mujoco Playground simulation [Zakka et al., 2025], spanning quadruped and humanoid morphologies under four representative shifts (mass on torso, mass on calf, rocky terrain, slippery terrain) and three different architectures, and observe up to 50% performance gains very efficiently in a single-device setting. We further validate our approach on a physical Unitree GO2 robot by attaching a {3, 2} kg calf payload and performing fully on-device TTA , observing trends consistent with simulation. We can summarize our contributions in four folds:

• A lightweight, on-device TTA framework for RL control. We adapt only the actor parameters while freezing the critic, maintain a compact memory of recent observations, and perform periodic, self-supervised updates that fit strict control-loop latency and compute budgets.

• Entropy-regularized critic-driven adaptation. We introduce an objective that maximizes the critic’s estimated action value while promoting higher-entropy critic outputs during test-time updates—mitigating overconfident value extrapolation and preserving local action coverage; the procedure is architecture-agnostic and supervision-free.

• Score-based memory with importance sampling. We propose a memory mechanism that assigns each observation a score inversely proportional to the critic’s predicted value, stores harder samples, and employs importance sampling to focus on more informative samples under memory constraints.

• Comprehensive validation in simulation and on-device. We show consistent gains on GO1/GO2/G1 embodiments and on four deployment shifts, and confirm the same trends on a physical Unitree GO2 via fully on-device adaptation with a {3, 2} kg calf payload.

## 2 Robot Adaptation as Test Time Adaptation

## 2.1 Problem Formulation and Evaluation Protocol

RL setting and distribution shift. We consider locomotion as a Markov decision process (MDP) $\mathcal { M } = ( \mathcal { O } , \bar { \mathcal { A } } , P , \gamma )$ , where $\mathcal { O } \subseteq \mathbb { R } ^ { d }$ denotes the observation space, $\mathcal { A } \subseteq \mathbb { R } ^ { p }$ the action space, P the (unknown) transition dynamics over observations,<sup>1</sup> and $\gamma \in ( 0 , 1 )$ ) the discount factor. A policy $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { o } )$ maps an observation $o _ { t } \in \mathcal { O }$ (e.g., proprioception, IMU, joint states, contact signals) to an action $a _ { t } \in \mathcal { A } \left( \mathrm { e . g } \right.$ ., target joint positions/torques). During training, $\pi _ { \theta }$ is learned in a source training setup, inducing a training distribution over the source observations. At deployment, the robot operates under a shifted test-time distribution $( \mathrm { e . g . , }$ altered friction, payload changes, rocky terrain), which can be viewed as changes in the observations (i.e from o to oˆ) encountered by the agent and in the induced observation-transition process (i.e. from $P$ to $\hat { P } )$ . We assume the deployed policy is pretrained using an actor–critic framework [Haarnoja et al., 2018, Fujimoto et al., 2018]. In particular, alongside the pretrained policy $\pi _ { \theta _ { 0 } }$ , training produces a critic $Q _ { \psi _ { 0 } } ( o , a )$ that estimates the action-value of executing action a when observing o. In contrast with the common asymmetric setup [Nahrendra et al., 2023] where the critic is fed privileged information from the underlying simulator state, we consider a critic that consumes the same set of observations as the actor.

Test-time adaptation as an online stream problem. At test time, the robot receives a stream of unlabeled observations $\{ o _ { t } \} _ { t \ge 0 }$ governed by the actions taken by the policy $\pi _ { \theta _ { t } }$ and the transition dynamics ${ \hat { P } } .$ A Test-Time Adaptation (TTA) method is an update rule $g$ that uses only quantities available online at inference time to update either the model parameters, the received observation, or both. This formulation is intentionally general: $g$ may adapt the policy parameters $\theta ,$ the critic parameters $\psi ,$ , both, or a subset of either.

Online evaluation protocol. Following the online/streaming evaluation paradigm [Cai et al., 2021, Ghunaim et al., 2023, Alfarra et al., 2024], we simulate the interaction between the deployed agent and the test stream at each time step t as:

1. The environment reveals an observation $o _ { t } \sim \hat { P } ( o _ { t - 1 } , a _ { t - 1 } )$

2. The agent selects and executes an action $a _ { t } \sim \pi _ { \theta _ { t } } ( o _ { t } )$

3. The adaptation rule updates parameters: $( \theta _ { t + 1 } , \psi _ { t + 1 } )  g ( \theta _ { t } , \psi _ { t } ) .$

Notably, this protocol does not assume access to rewards or labels at test time. Although rewards are not used by the adaptation mechanism, performance is evaluated using task-level metrics computed offline or by the benchmark (e.g., episodic return in simulation, distance traveled, tracking error, stability/fall rate, task success). These metrics are used solely for reporting and comparison, not as supervision for adaptation.

## 2.2 Constructing a Baseline

Test-time adaptation for robot control is challenging for two reasons. (i) As opposed to massively parallel simulation-based training [Li et al., 2023], deployment typically occurs in a single environment with limited diversity over short time horizons. Consecutive observations are strongly correlated and provide a narrow view of the shifted dynamics. (ii) The distribution shift can be abrupt (e.g., payload changes or slippery terrain), yet the adaptation budget is small and must fit within on-device latency constraints. These constraints make standard TTA baselines $( \mathrm { e . g } $ ., Tent [Wang et al., 2020]) as well as RL-based alternatives (e.g., OPEX [Park et al., 2024]) ineffective in our setting (see Fig. 1, middle) For simplicity, we assume throughout that the policy is deterministic (i.e. $a _ { i } = \pi _ { \theta _ { t } } ( o _ { i } ) )$

Critic-guided policy update QMAX One naive update strategy is to utilize the critic’s action value as a surrogate self-supervised performance measure. With that, one could potentially update the policy parameters $( i . e . \ \theta )$ to maximize the performance measure on received observations at test-time (i.e. minimizing ${ \mathcal { L } } = - Q _ { \psi _ { 0 } } \bigl ( o _ { i } , a _ { i } \bigr ) ;$ ). While intuitive, the aforementioned factors (i, ii) make this naive per-step updates brittle and motivate aggregating evidence over time via a small memory buffer.

FIFO buffer. We maintain a bounded on-device FIFO buffer B of capacity N that stores the most recent observations. At each time step, we append $o _ { t }$ and evict the oldest element if $| B | > N$

Every K steps, we sample a minibatch $\{ o _ { i } \} _ { i = 1 } ^ { B }$ uniformly from B to form a more stable and diverse adaptation batch than a single observation. We note that employing an episodic memory for TTA was also proposed by Yuan et al. [2023].

Critic-guided policy update with buffer (QMAX). Given a pretrained actor–critic pair $( \pi _ { \theta _ { 0 } } , Q _ { \psi _ { 0 } } )$ we define a simple self-supervised adaptation objective is to maximize the critic’s predicted action value on buffered observations:

$$
\mathcal { L } _ { \operatorname { Q M a x } } ( \theta ; \mathcal { B } ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } Q _ { \psi _ { 0 } } \bigl ( o _ { i } , a _ { i } \bigr ) , \qquad a _ { i } = \pi _ { \theta } \bigl ( o _ { i } \bigr ) .\tag{1}
$$

We then perform a gradient step $\theta _ { t + 1 } \gets \theta _ { t } - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { Q M A X } } | _ { \theta = \theta _ { \mathrm { r } } }$ with step size $\eta > 0$ while freezing the critic’s parameters. While distribution shift can affect the critic’s calibration, we empirically find that $Q _ { \psi _ { 0 } }$ still provides a useful surrogate signal for guiding test-time updates in practice. This baseline is lightweight and fully self-supervised, requiring only test-time observations and critic predictions (no rewards or labels). Nevertheless, we found that QMAX is pretty effective in recovering significant portion of the performance drop at test time.

## 3 Test-time Entropy regularized, Memory guided Policy Optimization

In the previous section, we formulated online test-time adaptation for a pretrained policy under distribution shift and introduced QMAX, which adapts the policy by maximizing a pretrained critic on a small buffer of recent observations. As we demonstrate empirically in Section 5.1, QMAX can be effective, but it may become unstable under severe shifts: when the critic is queried out of distribution, directly ascending $Q _ { \psi _ { 0 } } ( o , a _ { t } )$ may yield overly sharp updates, reduced action coverage, and brittle closed-loop behavior. We propose TEMPO (Test-time Entropy-regularized, Memoryguided Policy Optimization), which improves stability and sample-efficiency by combining two lightweight modifications: (i) An entropy-regularized critic objective, and (ii) a score-based memory that prioritizes informative test-time observations. As described in Section 2.1, TEMPO updates only the policy parameters θ at test time.

## 3.1 Entropy Maximization

While action perturbation broadens the local neighborhood explored by each update, it does not explicitly discourage overconfident critic-driven collapse. TEMPO therefore augments critic maximization with an entropy term defined on the critic’s predictive distribution, encouraging updates that are high-value yet avoid overly peaky solutions.

Distributional critic and entropy. Following Seo et al. [2025], we assume the actor–critic pretraining provides a distributional critic $Z _ { \psi _ { 0 } } ( o , a ) \in \mathcal { P } ( \mathcal { Z } )$ where $\mathcal { P } ( \mathcal { Z } )$ is the probability simplex over a discretized value support ${ \mathcal { Z } } ,$ , with $Q _ { \psi _ { 0 } } ( o , a ) = \mathbb { E } _ { z \sim Z _ { \psi _ { 0 } } ( o , a ) } [ z ]$ . To avoid the high confidence regions over the critic’s prediction we propose an Entropy-regularized objective. TEMPO optimizes the following entropy-regularized loss on buffered observations with:

$$
\mathcal { L } _ { \mathrm { T E M P O } } ( \theta ; \mathcal { B } ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ Q _ { \psi _ { 0 } } \bigl ( o _ { i } , a _ { i } \bigr ) \ + \ \lambda \ \mathcal { H } ( Z _ { \psi _ { 0 } } ( o _ { i } , a _ { i } ) ) \right] ,\tag{2}
$$

where $\lambda > 0$ controls the strength of entropy regularization. The entropy term tempers overconfident maximization by favoring updates that remain robust in regions where the critic’s predictions are less concentrated, improving stability under large shifts.

## 3.2 Score-Based Memory

Next, we ask the following question: given the memory and compute constraints for on-device adaptation, which samples shall we store? And which one shall we adapt on? FIFO buffers treat all recent observations as equally useful, even though under distribution shift only a subset of encountered states may be responsible for performance degradation (e.g., slip events or destabilizing contacts). TEMPO therefore replaces FIFO with a score-based memory that preferentially stores and samples the most informative observations.

Scoring and buffer maintenance. Given the current policy and pretrained critic, we assign each observation a score

$$
S ( o ) = \frac { 1 } { Q _ { \psi _ { 0 } } ( o , \pi _ { \theta } ( o ) ) + \varepsilon } ,\tag{3}
$$

with a small constant $\varepsilon > 0$ for numerical stability. High scores correspond to low predicted value and thus “hard” states. TEMPO maintains a buffer B of capacity N storing pairs $( o , S ( o ) )$ . When a new observation $o _ { t }$ arrives, we compute $S ( o _ { t } )$ and insert it if it exceeds the minimum score currently in the buffer; in that case, the lowest-scoring element is evicted. This yields a compact memory that tracks the most challenging deployment observations under the current policy.

Importance sampling for adaptation. During each adaptation step, TEMPO samples a minibatch from B using probabilities proportional to the stored scores:

$$
p _ { j } ~ = ~ { \frac { \hat { S } _ { j } } { \sum _ { k } \hat { S } _ { k } } } , \qquad { \mathrm { w h e r e ~ } } \hat { S } _ { j } = S _ { j } - \operatorname* { m i n } _ { i } S _ { i } + \varepsilon\tag{4}
$$

where $p _ { j }$ is the probability of sampling the $j ^ { t h }$ element from the memory, and $S _ { j }$ is computed according to Eq. (3). TEMPO then draws B observations from B without replacement according to $\{ p _ { j } \}$ . This focuses limited on-device adaptation compute on states that are most associated with degraded performance, improving sample-efficiency compared to uniform FIFO sampling.

Putting All Together. TEMPO combines three lightweight ingredients into a single on-device adaptation loop. At each control step, the robot executes the current deterministic policy $a _ { t } = \pi _ { \theta _ { t } } ( o _ { t } )$ and records the observation in a bounded memory. Unlike FIFO, TEMPO maintains a score-based buffer that preferentially keeps “hard” observations (Eq. (3)) and samples adaptation minibatches via importance sampling (Eq. (4)). Every K steps, TEMPO performs a single gradient update on θ (with $Q _ { \psi _ { 0 } }$ frozen) by minimizing the entropy-regularized objective in Eq. (2). Figure 1 (right) summarizes the pipeline. A detailed algorithm for adapting with TEMPO is in Appendix A.

## 4 Related Work

Domain Shifts on Robots Robots deployed outside controlled training conditions face domain shifts such as changes in contact dynamics, inertial properties, and sensing, which can substantially degrade locomotion policies. A common response is to harden policies in simulation via accurate actuator modeling [Tan et al., 2018, Hwangbo et al., 2019, Zhu et al., 2026, Shin et al., 2025], domain randomization [Peng et al., 2018, Lee et al., 2020] and massively parallel training [Rudin et al., 2022, Zakka et al., 2025]. This improves zero-shot robustness within the training envelope. To better handle unstructured environments, many approaches additionally incorporate exteroception (e.g., elevation maps or images) to anticipate terrain-induced shifts [Miki et al., 2022, Lai et al., 2023, He et al., 2025, Yu et al., 2024, Dong et al., 2025], or auxiliary mechanisms (e.g., asymmetric training, distillation, learned estimators) that infer latent factors [Chen et al., 2020, Kumar et al., 2021, Ji et al., 2022, Choi et al., 2023]. However, when deployment conditions drift beyond the training distribution, these approaches often still require retraining or additional structure, motivating lightweight deployment-time adaptation under strict hardware constraints. Adapting at deployment time has been explored in adjacent settings: Park et al. [2024] analyze offline RL bottlenecks and propose simple test-time policy improvement methods, while Chen et al. [2023] adapt on-the-fly in single-life deployment by selecting and modulating a repertoire of pretrained behaviors using value estimates. In contrast, we target lightweight on-device test-time adaptation for locomotion under strict latency and memory budgets, by adapting a single deployed actor via critic-guided updates.

Test-Time Adaptation Test-Time Adaptation (TTA) mitigates the performance drop of pretrained models under test-time distribution shift by adapting at inference using only unlabeled test data [Mancini et al., 2018, Kojima et al., 2022]. Methods either update model parameters online [Liang et al., 2020, Wang et al., 2020] or adapt the test input to better match the training distribution [Gao et al., 2022]. Early approaches focused on updating Batch Normalization statistics [Li et al., 2016], while later work introduced stronger self-supervised objectives, including entropy minimization and stabilized variants [Wang et al., 2020, Niu et al., 2022, 2023], information maximization [Liang et al., 2020], and consistency/teacher–student updates [Yuan et al., 2023], as well as auxiliary-task-driven adaptation [Alfarra et al., 2025]. Most progress has been demonstrated on perception benchmarks—primarily classification under synthetic corruptions and covariate shift [Hendrycks and Dietterich, 2019]—with fewer works on structured prediction [Mirza et al., 2023] or other modalities such as missing modality [Ramazanova et al., 2025]. Translating TTA to robotics is challenging because adaptation must preserve stable closed-loop control under strict latency/memory budgets and highly correlated on-policy streams; our work addresses this by tailoring lightweight, critic-guided TTA to on-device locomotion.

![](images/c2d3760866ff79ceb8459da1c5a1c2154f5285e8c5101e977208d3a01936cd98.jpg)  
Figure 2: Main results on Go2 under deployment-time distribution shift. Top: episodic return. Bottom: average forward linear velocity. Shifts (left→right): 1−friction, calf mass, rough terrain, torso mass (severity increases along x-axis). As the severity of the domain shift increases, performance degrades more rapidly. Naive adaptation via QMAX recovers part of the performance drop. Our proposed TEMPO consistently provides the largest performance gain, especially under severe shifts.

## 5 Experiments

Environments and pretrained policies. We evaluate TTA on three simulated locomotion tasks from MuJoCo Playground [Zakka et al., 2025]: GO1-JOYSTICK (quadruped), GO2-JOYSTICK (quadruped), and G1-JOYSTICK (humanoid). Since Section 2 places no assumptions on the policy architecture, we consider three representative pretrained policy families: FAST-TD3 [Seo et al., 2025], SIMBAV2 [Lee et al., 2025], and Rapid Motor Adaptation (RMA) [Kumar et al., 2021].

Training protocol and distribution shifts. To disentangle test-time robustness from training-time regularization, we train π on aflat terrain without domain randomization, and evaluate (and adapt) under a single fixed test environment that emulates a realistic deployment condition. We consider four shift types parameterized by a severity scalar δ: (i) Rough terrain, with height variation δ; (ii) Calfmass perturbation, attaching a mass of δ kg to one calf; (iii) Torso mass perturbation, attaching a mass of δ kg to the torso; and (iv) Slippery terrain, setting the ground friction coefficient to 1 − δ. Unless otherwise stated, each test episode contains a single shift type (i.e., no mixed corruptions).

Adaptation protocol. Across all methods, we use Adam with learning rate $5 \times 1 0 ^ { - 6 }$ . We adapt all learnable parameters of $\pi _ { \theta }$ while freezing the critic(s). We maintain a replay buffer of size $| B | = 1 6 \mathrm { k }$ and use batches of size $B = 2 \mathrm { k }$ . Note that with 49-dimensional observations and $| B | = 1 6 \mathrm { k }$ , the FIFO buffer footprint is merely ≈ 3 MiB; score-based variants require ≈ 3.05 MiB. Adaptation is performed online with period K = 128: every K environment steps, we sample a batch from B, compute gradients, and take a single optimizer step. For TEMPO we set the entropy weight to $\lambda = 0 . 5 .$ . In the following sections, we will provide a comprehensive evaluation demonstrating the benefits of TTA under distribution shift and the consistent performance gains of TEMPO. Unless stated otherwise, we fix the policy architecture to Fast-TD3 and the environment to Go2 on Flat terrain without domain randomization. Further experimental details can be found in Appendix B.1.

## 5.1 Effectiveness of TEMPO

We first evaluate whether lightweight test-time adaptation (TTA) can recover locomotion performance under realistic deployment shifts, and whether TEMPO provides robust gains beyond critic maximization and general-purpose TTA baselines. To do so, we test on four shift families with

Table 1: Real-world deployment on Unitree Go2 under a calf-mass shift. TEMPO significantly improves average reward over the non-adapted baseline for various input commands.
<table><tr><td> $v _ { x }$ </td><td> $w _ { z }$ </td><td>No adaptation</td><td>TEMPO</td></tr><tr><td>0.5</td><td>0.0</td><td> $9 3 . 2 \pm 1 . 3$ </td><td> ${ \bf 9 5 . 2 \pm 0 . 6 }$ </td></tr><tr><td>-0.5</td><td>0.0</td><td> $9 8 . 7 \pm 1 . 6 $ </td><td> ${ \bf 1 1 3 . 5 \pm 0 . 7 }$ </td></tr><tr><td>-0.7</td><td>0.0</td><td> $8 8 . 1 \pm 1 . 1$ </td><td> ${ \bf 9 6 . 5 \pm 1 . 0 }$ </td></tr><tr><td>0.7</td><td>0.6</td><td> $8 8 . 4 \pm 3 . 3 $ </td><td> ${ \bf 1 0 1 . 6 \pm 1 . 4 }$ </td></tr><tr><td>0.7</td><td>-0.6</td><td> $8 0 . 5 \pm 1 . 0$ </td><td> ${ \bf 8 6 . 9 \pm 1 . 2 }$ </td></tr><tr><td colspan="2">Average</td><td> $8 9 . 8$ </td><td> $\mathbf { 9 8 . 7 }$ </td></tr></table>

Figure 3: Real-world distribution shift on Unitree Go2. Weight plates mounted on a calf to induce an inertial perturbation during on-device adaptation.  
![](images/dcdda3e4b02fed556baf74ae2110ab1f1bec1aff7b1d6b1126bb0da5cd8d0832.jpg)  
increasing severity: slippery terrain, calf mass perturbation, rough terrain, and torso mass perturbation. Adaptation is performed online during execution using only test-time observations; rewards are never used for updating and are only used for offline evaluation. All results are averaged across six seeds.

Figure 2 reports episodic return (top row) and average forward linear velocity (bottom row), capturing both task-level performance and the quality of locomotion. Across all shifts, the pretrained policy without adaptation degrades rapidly as severity increases, indicating that even strong policies trained under nominal conditions are brittle when contact dynamics or inertial properties change at deployment. Introducing TTA reliably mitigates this drop: the simple critic-guided baseline QMAX recovers performance across a wide range of severities, confirming that a frozen critic can provide an informative self-supervised signal at test time. However, QMAX often saturates under harder shifts. In contrast, TEMPO is consistently best across all shift families and throughout the severity range, with the clearest advantages emerging in the most challenging regimes. Importantly, these gains are mirrored in forward velocity: TEMPO maintains faster, more continuous locomotion under shift, suggesting that improvements reflect better stabilized behavior rather than reward-only artifacts.

## 5.2 Real World Deployment

We validate TEMPO on a physical Unitree GO2 quadruped to assess whether the improvements observed in simulation translate to real hardware. To induce a controlled deployment shift, we attach weight plates to both the front left and rear left calves using a custom 3d-printed rigid mount (Fig. 3), altering the leg’s inertial properties and contact dynamics while keeping the policy and controller stack unchanged. Adaptation is performed once on-device for 4 minutes and using the same streaming protocol as in simulation: we continuously update the actor parameters online while freezing the critic, using only real observations and critic predictions.

Table 1 reports the environment reward across for various velocity commands cmd $\mathbf { \Psi } = \left( v _ { x } , w _ { z } \right)$ where $v _ { x }$ is the desired forward linear velocity and $w _ { z }$ is the desired angular velocity. The evaluation for each command consists of 5 trials spanning a 3 second window (150 steps each). To define a useful metric for real world performance, we evaluate the undiscounted sum of the per-timestep linear-velocity tracking rewards $r _ { t } = \exp \left( - \| \mathbf { v } _ { x y } ^ { \mathrm { c m d } } - \mathbf { v } _ { x y } ^ { \mathrm { l o c } } \| ^ { 2 } \right)$ . Thus, each timestep contributes a value in (0, 1], resulting in a maximum possible value of 150. TEMPO outperforms the non-adapted baseline for each of the tested commands, improving the mean score from 89.8 to 98.7. Gains are consistent across command types, with the largest improvements under walking backwards at medium speed $( v _ { x } = - 0 . 5 )$ , where TEMPO increases the score by 15% from 98.7 to 113.5. Overall, these results provide evidence that TEMPO remains effective under real-world distribution shift and can improve locomotion performance in a fully on-device self-supervised setting.

## 5.3 Mixed Corruption Setting

Real deployments rarely involve a single isolated shift; instead, multiple factors can co-occur (e.g., payload changes together with altered contact dynamics). We therefore evaluate mixed corruptions by combining a payload perturbation with a terrain shift within the same episode. Specifically, we attach a fixed mass either to the calf (2 kg) or the torso (5 kg), and simultaneously test on either slippery terrain (1−friction) or rough terrain (rocky heightfield), each at two severities.

Table 2: Performance Comparison Under Mixed Corruptions. We report the environment reward (six seeds) for the non-adapted policy, QMAX, and our proposed TEMPO when the policy is subjected to a composition of corruptions (e.g. mass on torso and rocky terrain simultaneously). Our proposed TEMPO consistently provides the best performance by a significant margin.
<table><tr><td rowspan="3"></td><td colspan="4">Mass on Calf = 2 kg</td><td colspan="4">Mass on Torso = 5 kg</td></tr><tr><td colspan="2">Floor friction</td><td colspan="2">Rocky terrain</td><td colspan="2">Floor friction</td><td colspan="2">Rocky terrain</td></tr><tr><td>0.90</td><td>0.85</td><td>0.03</td><td>0.06</td><td>0.90</td><td>0.85</td><td>0.03</td><td>0.06</td></tr><tr><td>No Adapt</td><td> $9 . 2 _ { \pm 0 . 1 }$ </td><td> $1 2 . 9 _ { \pm 0 . 2 }$ </td><td> $1 9 . 9 { \scriptstyle \pm 0 . 1 }$ </td><td> $1 4 . 5 { \scriptstyle \pm 0 . 2 }$ </td><td> $4 . 2 _ { \pm 0 . 2 }$ </td><td> $6 . 2 { \scriptstyle \pm 0 . 1 }$ </td><td> $1 4 . 7 _ { \pm 0 . 3 }$ </td><td> $7 . 8 _ { \pm 0 . 2 }$ </td></tr><tr><td>QMAX</td><td> $9 . 7 _ { \pm 0 . 2 }$ </td><td> $1 3 . 4 _ { \pm 0 . 4 }$ </td><td> $2 0 . 7 _ { \pm 0 . 1 }$ </td><td> $1 5 . 9 { \scriptstyle \pm 0 . 2 }$ </td><td> $4 . 8 _ { \pm 0 . 1 }$ </td><td> $7 . 0 { \scriptstyle \pm 0 . 3 }$ </td><td> $1 7 . 3 _ { \pm 0 . 5 }$ </td><td> $9 . 9 _ { \pm 0 . 2 }$ </td></tr><tr><td>TEMPO</td><td> ${ \bf 1 0 . 4 } _ { \pm 0 . 2 }$ </td><td> ${ \pm } 4 . 4 _ { \pm 0 . 1 }$ </td><td> $2 1 . 2 _ { \pm 0 . 3 }$ </td><td> $1 7 . 5 _ { \pm 0 . 2 }$ </td><td> ${ \pm } 0 . 6 _ { \pm 0 . 2 }$ </td><td> ${ \bf 9 . 0 _ { \pm 0 . 4 } }$ </td><td> ${ \bf 2 0 . 1 _ { \pm 0 . 2 } }$ </td><td> $\pm 4 . 6 _ { \pm 0 . 2 }$ </td></tr></table>

Table 2 summarizes the results. Across all 8 mixed conditions, TEMPO is best: averaged over the table, it improves the environment reward from 11.18 (NO ADAPT) and 12.48 (QMAX) to 14.09, i.e., a gain of +2.91 (+26.1%) over NO ADAPT and +1.61 (+12.9%) over QMAX. The advantage is especially pronounced for the harder torso-payload compositions, where TEMPO raises the average reward from 8.23 to 12.33 $( + 4 . 1 0 , + 4 9 . 8 \hat { \% } )$ , and from 10.00 to 12.33 versus QMAX $( + 2 . 3 3 , \bar { + } 2 3 . 3 \% )$ . The largest single-case improvement occurs under torso mass with rocky terrain at the highest severity, where TEMPO reaches 14.6 compared to 10.5 for QMAX (+4.1, +39.0%) and 7.8 without adaptation $( + 6 . 8 , + 8 7 . 2 \% )$ , demonstrating clear benefits under compounding shifts.

## 5.4 Continual TTA

So far, we evaluated adaptation under a fixed deployment condition. In practice, however, distribution shift is often non-stationary: terrain properties can change gradually, payloads can vary over time, and contact dynamics can drift during prolonged operation. We therefore consider a continual TTA [Wang et al., 2022] setting in which the deployed policy experiences a sequence of increasingly difficult conditions, implemented by continually increasing the severity of the domain shift during evaluation.

Figure 4 reports average return as adaptation progresses in this non-stationary stream. The results highlight a clear separation between methods: TEMPO exhibits faster recovery after the shift intensifies and reaches a higher performance plateau than QMAX. This gap is most evident as the stream becomes more challenging, where naive critic maximization improves initially but saturates earlier, while TEMPO continues to make progress and maintains stronger steady-state performance. Overall, these results indicate that TEMPO is not only effective under a fixed test condition, but also well-suited to continual deployment where the test distribution evolves over time and adaptation must remain stable under compounding shift severity.

Performance on Source Domain. A key concern in continual test-time adaptation is forgetting: gains under shifted conditions should not degrade behavior when the environment returns to nominal. To evaluate this, we append the source domain (flat terrain without domain shift) at the end of the continual sequence and directly assess the adapted policies (Appendix B.4, Fig. 9). We observe that both TEMPO and QMAX recover source-domain performance instantly, matching the original policy without any transient degradation or additional adaptation, indicating no observable forgetting.

![](images/7c247723c1a03b93cf99a69c21b02c7e057a60ce6d3466feeccefce8090451c9.jpg)

![](images/640385f047f5d114fb679dfa131fcf6e94e5d7d25fc687b4c483c6a116886315.jpg)

![](images/c2433580e1f799d92d0d78dd8ed0d0ff869b51239db2157efe310a33793b431b.jpg)  
Figure 4: Continual TTA under progressively increasing shift severity. We report environment rewards versus adaptation steps as the deployment shift intensifies over time. TEMPO recovers faster and converges to a higher performance plateau than QMAX across all corruptions.

Table 3: Ablating TEMPO. Average episodic return under increasing shift severity for three corruption families. Adding entropy regularization and score-based memory each improves over QMAX, and combining both (TEMPO) yields the best robustness across all severities.
<table><tr><td>Corruption</td><td colspan="2">Mass on Calf (kg)</td><td colspan="2">Mass on Torso (kg)</td><td colspan="2">Rocky Terrain (cm)</td></tr><tr><td>Severity Method</td><td>4</td><td>5</td><td>5</td><td>7.5</td><td>7.5</td><td>10</td></tr><tr><td>No Adapt</td><td>15.3</td><td>13.0</td><td>17.5</td><td>12.7</td><td>14.0</td><td>7.0</td></tr><tr><td>QMAX</td><td>16.5</td><td>14.4</td><td>19.7</td><td>15.4</td><td>16.6</td><td>9.0</td></tr><tr><td>+ Entropy Max.</td><td>17.0</td><td>15.3</td><td>21.0</td><td>17.0</td><td>17.3</td><td>10.6</td></tr><tr><td>+ Score-Based Mem.</td><td>17.5</td><td>15.9</td><td>21.0</td><td>17.1</td><td>17.8</td><td>11.3</td></tr><tr><td>+ ALL (TEMPO)</td><td>17.9</td><td>16.0</td><td>21.5</td><td>17.6</td><td>18.0</td><td>11.5</td></tr></table>

## 5.5 Ablating Components of TEMPO

Table 3 ablates the two components of TEMPO—entropy regularization and the score-based memory—across increasing severities of payload and terrain shifts. Starting from QMAX, adding entropy regularization yields consistent gains of +0.5 to +1.6 return across the six settings. Replacing FIFO with the score-based memory provides larger improvements of +1.0 to +2.3. Combining both (TEMPO) gives the best results in every column, improving over QMAX by +1.4 to +2.5. This indicates that the design of the memory can play a crucial role in boosting the adaptation performance. Relative to NO ADAPT, TEMPO improves by +2.6 to +4.9 per setting, indicating that both components contribute and their combination is complementary.

## 5.6 RMA as an Adaptation-Aware Backbone

Rapid Motor Adaptation (RMA) [Kumar et al., 2021] explicitly targets fast adaptation to dynamics changes by conditioning the policy on a learned latent embedding that is inferred online during deployment. Unlike the policies considered in previous sections, RMA is therefore not a static pretrained controller, but already incorporates an internal adaptation mechanism, making it a strong test for whether our proposed TEMPO can provide additional gains beyond architecture-specific adaptation.

![](images/fb5261e25977a4a1d497c14000f703c5134bbec3052d5390f95f7a69aa6340b0.jpg)  
Figure 5: RMA ablation. Frozen RMA vs native RMA adaptation vs RMA+TEMPO. TEMPO provides consistent and significant additional gains.

We evaluate three variants under the same shifted test streams: (i) No Adaptation (freeze RMA at deployment), (ii) RMA (enable RMA’s native online adaptation), and (iii) RMA + TEMPO (apply TEMPO’s critic-guided test-time updates on top of the deployed RMA policy). Figure 5 shows that native RMA adaptation improves over the frozen baseline, while augmenting RMA with TEMPO yields the best performance across the evaluated shift severities, demonstrating that TEMPO complements (rather than replaces) RMA’s built-in adaptation. Further experiments on different architectures, and robustness analyses of TEMPO can be found in Appendix B.

## 6 Conclusions

In this work, we studied reward-free, on-device test-time adaptation for robot locomotion under deployment-time distribution shift. Starting from a lightweight baseline that updates only the actor using a frozen pretrained critic, we introduced TEMPO, which stabilizes critic-guided adaptation through entropy regularization and a score-based memory that prioritizes informative test-time observations. Across simulated locomotion benchmarks and multiple shift types, architectures, and environments, TEMPO consistently improves over no adaptation and naive critic maximization, with the clearest gains under severe shifts. We further confirm the same trend on a physical Unitree Go2 via fully on-device adaptation under a calf-payload perturbation.

## References

Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. RMA: Rapid Motor Adaptation for Legged Robots. In Proceedings of Robotics: Science and Systems, Virtual, July 2021. doi: 10.15607/RSS.2021.XVII. 011.

Peng Sun, Qi Li, Hao Hu, Junjie Qiang, Weiwei Wu, and Xin Luo. Proprioceptive slip detection and state estimation of multi-legged robots in slippery scenarios. Frontiers ofMechanical Engineering, 20(5):36, 2025.

Alexandros Pouroullis, David Blore, Michael Scott, Julius Smith, Sindiso Mkhatshwa, and Geoff Nitschke. Automating damage recovery in a legged robot. In 2025 IEEE Symposium on Computational Intelligence in Artificial Life and Cooperative Intelligent Systems (ALIFE-CIS), pages 1–7. IEEE, 2025.

Yang Zhang, Buqing Nie, Zhanxiang Cao, Yangqing Fu, and Yue Gao. Disturbance-aware adaptive compensation in hybrid force-position locomotion policy for legged robots. arXiv preprint arXiv:2506.00472, 2025.

Annie S Chen, Govind Chada, Laura Smith, Archit Sharma, Zipeng Fu, Sergey Levine, and Chelsea Finn. Adapt on-the-go: Behavior modulation for single-life robot deployment. arXiv preprint arXiv:2311.01059, 2023.

Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 3803–3810, 2018. doi: 10.1109/ICRA.2018.8460528.

Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23–30. IEEE, 2017.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020.

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In International conference on machine learning, pages 9229–9248. PMLR, 2020.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems, 33:1179–1191, 2020.

Benjamin Eysenbach and Sergey Levine. Maximum entropy rl (provably) solves some robust rl problems. arXiv preprint arXiv:2103.06257, 2021.

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay, 2016. URL https://arxiv.org/abs/1511.05952.

Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder A Kahrs, et al. Mujoco playground. arXiv preprint arXiv:2502.08844, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, 2018. URL https://arxiv.org/abs/1801. 01290.

Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods, 2018. URL https://arxiv.org/abs/1802.09477.

I Made Aswin Nahrendra, Byeongho Yu, and Hyun Myung. Dreamwaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 5078–5084. IEEE, 2023.

Zhipeng Cai, Ozan Sener, and Vladlen Koltun. Online continual learning with natural distribution shifts: An empirical study with visual data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8281–8290, 2021.

Yasir Ghunaim, Adel Bibi, Kumail Alhamoud, Motasem Alfarra, Hasan Abed Al Kader Hammoud, Ameya Prabhu, Philip HS Torr, and Bernard Ghanem. Real-time evaluation in online continual learning: A new paradigm. arXiv preprint arXiv:2302.01047, 2023.

Motasem Alfarra, Hani Itani, Alejandro Pardo, Shyma Alhuwaider, Merey Ramazanova, Juan C Pérez, Zhipeng Cai, Matthias Müller, and Bernard Ghanem. Evaluation of test-time adaptation under computational time constraints. International Conference on Machine Learning, 2024.

Zechu Li, Tao Chen, Zhang-Wei Hong, Anurag Ajay, and Pulkit Agrawal. Parallel q-learning: Scaling off-policy reinforcement learning under massively parallel simulation, 2023. URL https://arxiv.org/abs/2307. 12983.

Seohong Park, Kevin Frans, Sergey Levine, and Aviral Kumar. Is value learning really the main bottleneck in offline rl? Advances in Neural Information Processing Systems, 37:79029–79056, 2024.

Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15922–15932, 2023.

Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin, and Pieter Abbeel. Fasttd3: Simple, fast, and capable reinforcement learning for humanoid control. arXiv preprint arXiv:2505.22642, 2025.

Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, and Vincent Vanhoucke. Sim-to-real: Learning agile locomotion for quadruped robots. In Proceedings of Robotics: Science and Systems, Pittsburgh, Pennsylvania, June 2018. doi: 10.15607/RSS.2018.XIV.010.

Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics, 4(26):eaau5872, 2019.

Chengrui Zhu, Zhen Zhang, Weiwei Liu, Siqi Li, and Yong Liu. Learning accurate and robust velocity tracking for quadrupedal robots. Journal ofField Robotics, 43(1):76–88, 2026. doi: https://doi.org/10.1002/rob.70028. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.70028.

Young-Ha Shin, Tae-Gyu Song, Gwanghyeon Ji, and Hae-Won Park. Reinforcement learning for high-speed quadrupedal locomotion with motor operating region constraints: Mitigating motor model discrepancies through torque clipping in realistic motor operating region. IEEE Robotics & Automation Magazine, 32(2): 49–59, 2025. doi: 10.1109/MRA.2024.3487322.

Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. Science Robotics, 5(47):eabc5986, 2020. doi: 10.1126/scirobotics. abc5986. URL https://www.science.org/doi/abs/10.1126/scirobotics.abc5986.

Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes using massively parallel deep reinforcement learning. In Conference on robot learning, pages 91–100. PMLR, 2022.

Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics, 7(62):eabk2822, 2022. doi: 10.1126/scirobotics.abk2822. URL https://www.science.org/doi/abs/10.1126/scirobotics. abk2822.

Hang Lai, Weinan Zhang, Xialin He, Chen Yu, Zheng Tian, Yong Yu, and Jun Wang. Sim-to-real transfer for quadrupedal locomotion via terrain transformer. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 5141–5147, 2023. doi: 10.1109/ICRA48891.2023.10160497.

Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz Bächer, and Marco Hutter. Attention-based map encoding for learning generalized legged locomotion. Science Robotics, 10(105):eadv3604, 2025. doi: 10.1126/scirobotics.adv3604. URL https://www.science.org/doi/abs/10.1126/scirobotics. adv3604.

Alan Yu, Ge Yang, Ran Choi, Yajvan Ravan, John Leonard, and Phillip Isola. Learning visual parkour from generated images. In 8th Annual Conference on Robot Learning, 2024. URL https://openreview.net/ forum?id=cGswIOxHcN

Yinzhao Dong, Ji Ma, Liu Zhao, Wanyue Li, and Peng Lu. Marg: Mastering risky gap terrains for legged robots with elevation mapping. IEEE Transactions on Robotics, 41:6123–6139, 2025. doi: 10.1109/TRO.2025. 3619041.

Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by cheating. In Conference on robot learning, pages 66–75. PMLR, 2020.

Gwanghyeon Ji, Juhyeok Mun, Hyeongjun Kim, and Jemin Hwangbo. Concurrent training of a control policy and a state estimator for dynamic and robust legged locomotion. IEEE Robotics and Automation Letters, 7(2): 4630–4637, 2022.

Suyoung Choi, Gwanghyeon Ji, Jeongsoo Park, Hyeongjun Kim, Juhyeok Mun, Jeong Hyun Lee, and Jemin Hwangbo. Learning quadrupedal locomotion on deformable terrain. Science Robotics, 8(74):eade2256, 2023. doi: 10.1126/scirobotics.ade2256. URL https://www.science.org/doi/abs/10.1126/scirobotics. ade2256.

Massimiliano Mancini, Hakan Karaoguz, Elisa Ricci, Patric Jensfelt, and Barbara Caputo. Kitting in the wild through online domain adaptation. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1103–1109. IEEE, 2018.

Takeshi Kojima, Yutaka Matsuo, and Yusuke Iwasawa. Robustifying vision transformer without retraining from scratch by test-time class-conditional feature alignment. arXiv preprint arXiv:2206.13951, 2022

Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In International Conference on Machine Learning, pages 6028–6039. PMLR, 2020.

Jin Gao, Jialing Zhang, Xihui Liu, Trevor Darrell, Evan Shelhamer, and Dequan Wang. Back to the source: Diffusion-driven test-time adaptation. arXiv preprint arXiv:2207.03442, 2022.

Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Revisiting batch normalization for practical domain adaptation. arXiv preprint arXiv:1603.04779, 2016.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In International conference on machine learning, pages 16888–16905. PMLR, 2022.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. International Conference on Learning Representations, 2023.

Motasem Alfarra, Alvaro Correia, Bernard Ghanem, and Christos Louizos. Test-time adaptation with source based auxiliary tasks. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https: //openreview.net/forum?id=XWAXcxNg4n.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. Proceedings ofthe International Conference on Learning Representations, 2019.

Muhammad Jehanzeb Mirza, Pol Jané Soneira, Wei Lin, Mateusz Kozinski, Horst Possegger, and Horst Bischof. Actmad: Activation matching to align distributions for test-time-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24152–24161, 2023.

Merey Ramazanova, Alejandro Pardo, Bernard Ghanem, and Motasem Alfarra. Test-time adaptation for combating missing modalities in egocentric videos. In International Conference on Learning Representations, volume 2025, pages 82755–82774, 2025.

Hojoon Lee, Youngdo Lee, Takuma Seno, Donghu Kim, Peter Stone, and Jaegul Choo. Hyperspherical normalization for scalable deep reinforcement learning. arXiv preprint arXiv:2502.15280, 2025.

Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7201–7211, 2022.

Ross Hartley, Maani Ghaffari Jadidi, Jessy W Grizzle, and Ryan M Eustice. Contact-aided invariant extended kalman filtering for legged robot state estimation. arXiv preprint arXiv:1805.10410, 2018.

## A Test-time Entropy regularized, Memory guided Policy Optimization

In Section 3, we proposed TEMPO, an efficient, on-device friendly TTA method for adapting robotic policies. In the construction of TEMPO, we followed Seo et al. [2025] in accessing the distributional critic. We note that the output of the distributional critic is nothing but the output of the penultimate layer of the critic network. Refer to the works of Seo et al. [2025], Lee et al. [2025] for further details. Due to the space constraints, we did not provide a full explicit algorithm to describe how one can implement TEMPO in practice. For completeness, we provide these algorithms in this Section.

Algorithm 1 summarizes the high level interaction between the test stream and the pretrained policy equipped with our proposed TEMPO. Given an observation vector $o _ { t } ,$ the policy decides on the action $a _ { t }$ . TEMPO leverages this observation vector and its corresponding action in updating its memory bank B according to Algorithm 2. When there are enough samples in the memory to conduct an adaptation step, TEMPO samples a batch with importance sampling according to Eq. 4 and runs a single update step with the loss in Eq 2. We note that for efficiency purposes, TEMPO conducts a single update step every K received observations to keep the additional compute requirement light.

Algorithm 2 describes the score based memory update that we proposed in section 3.2. In a nutshell, as long as the memory bank not full, we store all received observation accompanied by their corresponding score. When hitting the capacity limit, we check if the newly received observation has a better score than the sample with the lowest score in memory, then we evict that sample and store the newly received one. We note that for additional implementation efficiency purposes, one can leverage directly the action from step 4 in Algorithm 1 instead of recomputing it in step 1 in Algorithm 2 (one less forward pass).

## B Experiments

In Section 5, we provided comprehensive evaluation of how TEMPO is effective under different scenarios in both simulation and real world setting. In this section, we provide additional details about the experimental setup for full reproducibility, and additional analysis on TEMPO.

## B.1 Setup.

The policy observation consists of proprioceptive and command-related signals commonly used in legged locomotion. Specifically, it includes IMU measurements in the form of angular velocity (gyro) and a gravity-aligned vector, base linear velocity, joint angles expressed relative to a default pose, joint velocities, the previous action, and a commanded motion vector $( v _ { x } , v _ { y } , w _ { z } )$ specifying desired forward/lateral linear velocities and yaw rate. The critic uses a symmetric, non-privileged observation space identical to the policy input, which enables on-device test-time adaptation directly from robot sensors. Since our real-world Go2 robot does not enable linear velocity estimation out of the box, we utilize an Extended Kalman Filter to calculate velocities on the fly using the built-in foot force sensors [Hartley et al., 2018].

Algorithm 1: TEMPO (high-level online loop) Algorithm 2: UpdateMemory (Score Based)   
Input: pretrained $\pi _ { \boldsymbol { \theta } _ { 0 } } ;$ frozen critic $Q _ { \psi _ { 0 } } ;$ step size η. Input: observation o; memory bank $\boldsymbol { B } ;$   
Output: actions $\{ a _ { t } \} _ { t \ge 0 }$ and adapted parameters $\{ \theta _ { t } \} _ { t \geq 0 } .$ Output: updated memory bank B   
1: Initialize $\theta _ { 0 } , B  \varnothing$ 1: $a  \pi _ { \theta } ( o ) _ { . }$   
2: for $t = 0 , 1 , 2 , \ldots$ do 2: $S \gets \frac { 1 } { Q _ { \psi _ { 0 } } ( o , a ) + \varepsilon }$ {Eq. 3}   
3: Observe $o _ { t } \sim \hat { P } ( o _ { t - 1 } , a _ { t - 1 } )$ 3: if $| B | < N$ then   
4: Act $a _ { t } \gets \pi _ { \theta _ { t } } ( o _ { t } )$ 4: Insert $( o , S )$ into B   
5: $\boldsymbol { \mathcal { B } } \gets \mathrm { U P D A T E M E M O R Y } \left( \boldsymbol { \mathcal { B } } , \boldsymbol { o _ { t } } \right)$ 5: else   
6: if SHOUL $\boldsymbol { \mathbf { \ell } } _ { \mathrm { { D A D A P T } } } ( t , \boldsymbol { B } )$ then 6: $( o _ { \operatorname* { m i n } } , S _ { \operatorname* { m i n } } ) \gets \arg \operatorname* { m i n } _ { ( o ^ { \prime } , S ^ { \prime } ) \in B } S ^ { \prime }$   
7: $\theta _ { t + 1 } \gets \mathrm { T e m p o U p d a t e } ( \theta _ { t } , B , Q _ { \psi _ { 0 } } )$ 7: if $S > S _ { \mathrm { m i n } }$ then   
8: else 8: Remove $( o _ { \operatorname* { m i n } } , S _ { \operatorname* { m i n } } )$ from B   
9: $\theta _ { t + 1 } \gets \theta _ { t }$ 9: Insert $( o , S )$ into B   
10: end if 10: end if   
11: end for 11: end if  
SHOULDADAPT: true every K steps when $| B | \geq B .$  
TEMPOUPDATE: Sample with Eq.4 and optimize Eq 2.

![](images/6f8ee8888070384010866ac5d1f3bbd57aa0d8e74e46a2427074c805c7eaea01.jpg)

![](images/d39346786f3835dd515e6b01d29e5b769bdd965058792752b0efe84ff0cf5756.jpg)

![](images/771fe1dd4064b8cfd73399adbb84dc1c071268f2cce1ec8167ab16097b50cfb6.jpg)  
Figure 6: Generalization across architectures and embodiments. Rocky-terrain results (left→right): Fast-TD3 trained with domain randomization, SimBA-V2, and the G1 humanoid. TEMPO transfers without architecture-specific modifications and remains effective across settings.

## B.2 Generalization to Different Architectures and Environments

A central goal of our approach is portability: test-time adaptation should improve a deployed controller without relying on architecture-specific components, privileged signals, or task-tailored objectives. We therefore evaluate whether TEMPO transfers beyond our default setting (Fast-TD3 on Go2) to substantially different pretrained policy families and to a different embodiment, without modifying the algorithm or introducing additional modules.

Figure 6 summarizes this evaluation on the rocky terrain test environment. On the left, we consider a Fast-TD3 policy trained with domain randomization, representing a stronger robustness-oriented pretraining baseline. Table 4 reports results on Go1 and G1 under mass perturbations. Even in this regime, TEMPO continues to provide consistent improvements, indicating that its gains are not limited to policies trained without robustness regularization and that it can complement domain randomization at deployment. In the middle, we adapt a policy trained with the SIMBA-V2 architecture, which differs in inductive biases and learning dynamics from Fast-TD3; TEMPO remains effective without any architecture-specific tuning, supporting the claim that our objective and update rule are broadly applicable to actor–critic policies. Finally, on the right, we evaluate on G1, a humanoid with markedly different morphology and contact dynamics. Despite this shift in embodiment, TEMPO still improves performance on rocky terrain, suggesting that the method generalizes beyond quadrupeds and can operate in more challenging control regimes.

## B.3 Sensitivity to Hyperparameters, Memory Size, Batch Size, and Adaptation Interval

On-device test-time adaptation must operate under strict memory and compute budgets; accordingly, we study how TEMPO behaves as we vary the three deployment-critical hyperparameters that directly control resource usage: replay memory size |B| (storage footprint), minibatch size B (compute per update), and adaptation interval K (update frequency, i.e., one gradient step every K environment steps). Figure 7 reports performance under distribution shift across these ablations.

Overall, TEMPO is robust across a broad range of settings and remains the best-performing method throughout the three sweeps. Importantly, this robustness translates into practical efficiency on device. First, TEMPO maintains strong performance even with relatively small |B|, implying that effective adaptation does not require storing large histories; this keeps the on-device memory footprint small and compatible with embedded constraints. Second, TEMPO performs well with small minibatches, showing that adaptation can be carried out with limited per-update compute (and correspondingly lower latency), rather than relying on large, expensive updates. Third, TEMPO remains effective for larger adaptation intervals K, meaning updates can be applied infrequently—reducing the total number of backward passes per unit time and making the method applicable even when compute budgets are very tight.

Table 4: Additional results on Go1 and G1 under payload perturbations. MC denotes mass on calf and MT denotes mass on torso.
<table><tr><td rowspan="2">Method</td><td colspan="4">Gol</td><td colspan="3">G1</td></tr><tr><td>MC=4</td><td>MC=5</td><td>MT=7.5</td><td>MT=10</td><td>MT=2</td><td>MT=3</td><td>MT=5</td></tr><tr><td>No Adapt</td><td>18.9</td><td>16.3</td><td>13.6</td><td>6.8</td><td>21.3</td><td>17.5</td><td>10.0</td></tr><tr><td>QMAX</td><td>19.6</td><td>17.6</td><td>13.6</td><td>6.9</td><td>24.1</td><td>22.1</td><td>17.4</td></tr><tr><td>TEMPO</td><td>20.5</td><td>18.6</td><td>16.9</td><td>11.0</td><td>24.5</td><td>23.4</td><td>18.5</td></tr></table>

![](images/72de39ee9f537d27a90af61f84edd977215668a4e49240f95e43b15e362426f6.jpg)

![](images/512a994c3990ef43b7230a7405a5bd6816ef027b66cf8d11fda3d4cc313a7e92.jpg)

![](images/7340e3fa503a770b999653bfcca8ab6556a56617ddb1a9a46f10bc3f45da35d9.jpg)  
Figure 7: Robustness to on-device adaptation hyperparameters. TEMPO is robust and performs best across different memory sizes (left), batch sizes (middle), and adaptation intervals (right). All experiments are run with Fast-TD3 on Go2 with Mass on Torso Corruption with severity of 7.5 kg.

As expected, extreme settings reveal intuitive trade-offs. Very small memories reduce state diversity and can limit coverage of informative “hard” states, while very large memories yield diminishing returns given a fixed update schedule. Similarly, very small batches increase gradient noise, whereas very large batches offer limited additional benefit relative to their higher compute cost. Finally, K exposes a responsiveness–efficiency trade-off: smaller K increases adaptation frequency but raises compute load, while larger K lowers overhead at the cost of slower reaction to abrupt changes. Taken together, the results indicate that TEMPO provides stable improvements without delicate tuning and can be configured to meet stringent on-device constraints via small |B|, small B, and large K.

At last, we assess the performance of TEMPO when varying its only hyperparameter; λ. We follow the experimental setup in Figure 7 and vary $\bar { \lambda } \in \lbrace 0 . 1 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 0 . 9 \rbrace$ and report the results in Figure 8. We observe a stable behavior under different values of λ.

![](images/55678c6c85ba19b749fe262a23be7b15840c08b2af11a2d1e5ffdc7405d7118e.jpg)  
Figure 8: Ablating λ.

## B.4 Performance on Source Domain

A key concern in continual test-time adaptation is catastrophic forgetting: parameters adapted to severe deployment shifts may degrade performance once the environment returns to nominal conditions. This risk is particularly acute in on-device, online settings, where updates are applied continuously without explicit supervision, replay from the source domain, or reset mechanisms. To assess whether critic-guided test-time adaptation induces such forgetting, we evaluate both QMAX and TEMPO after prolonged exposure to increasingly severe distribution shifts, followed by a return to the source domain (flat terrain without perturbations). Specifically, we adopt the continual TTA protocol from Section 5.4 and append a final evaluation phase in which the environment dynamics match the original training conditions, while keeping the adapted policy parameters fixed—without additional fine-tuning or recovery steps.

Figure 9 reports performance for two representative shift families (mass on calf and mass on torso) using the Fast-TD3 architecture. In both cases, QMAX and TEMPO immediately recover sourcedomain performance, matching the non-adapted policy without any transient degradation. This indicates that critic-guided test-time updates do not overwrite nominal behaviors, and that adaptation remains localized to state regions associated with deployment-time shifts. Overall, these results suggest that both methods—particularly TEMPO—support stable continual adaptation without observable forgetting, making them well suited for long-horizon deployment scenarios in which operating conditions drift temporarily but may later revert to nominal regimes.

Further, and to complement the previous experiments, We evaluate the trained policy under the source domain (flat terrain, no shift) with no adaptation, QMAX, and TEMPO, across all three reward signals used in the paper (averaged across six seeds). We observe that all three methods are numerically indistinguishable, and both velocity-tracking signals sit within run-to-run noise (environment reward of 25.6, Linear Velocity tracking reward of 864, and Angular Velocity Tracking reward of 457).

![](images/8e76670bede3b85e4123352a1746d8d73502cd359dc4180eccb1659bdc295a57.jpg)

![](images/bd0fc6beed0d256e9fa4b8246d5fa316aeeb0c4f33b694cd376122ebfe7c9b5e.jpg)  
Figure 9: Continual TTA followed by return to the source domain. Performance under a nonstationary deployment stream in which the severity of the domain shift increases over time and is then reset to nominal conditions. Left: mass-on-calf shift. Right: mass-on-torso shift. Architecture: Fast-TD3. Both QMAX and TEMPO recover source-domain performance immediately after the shift is removed, indicating no catastrophic forgetting despite prolonged adaptation under shift.

## B.5 Improving Angular Velocity Tracking

While the main paper focuses on task-level performance and forward locomotion metrics, stable angular velocity tracking is critical for practical deployment, especially under distribution shifts that alter contact dynamics or inertial properties. Poor tracking of angular velocity often manifests as oscillatory turning, delayed heading response, or loss of stability, which may not be fully reflected in episodic return alone. To examine this aspect more closely, we evaluate the angular velocity tracking performance of QMAX and TEMPO on Go2 under the same deployment-time distribution shifts considered in Section 5.1 For each method, we measure the tracking angular velocity reward, and report results averaged across 6 random seeds.

Figure 10 summarizes the results across four shift families with increasing severity. Without adaptation, the pretrained policy exhibits rapidly degrading angular velocity tracking as the shift intensifies. Introducing critic-guided test-time adaptation substantially mitigates this degradation: QMAX consistently improves tracking accuracy relative to no adaptation, while TEMPO provides the most stable and accurate tracking across all severities. The gains are most pronounced under harder shifts, such as rocky terrain and torso mass perturbations, where overconfident critic-driven updates can otherwise lead to unstable turning behavior. Overall, these results indicate that the benefits of TEMPO extend beyond forward velocity and reward metrics to finer-grained control objectives, demonstrating improved stability and control fidelity under deployment-time distribution shift.

## B.6 Does TEMPO Improve Local Action Coverage?

To analyze the impact of TEMPO on the local action coverage, we measure the critic’s distributional entropy over the course of adaptation. On the severe mixed-corruption setting (Mass on Torso=5, Rough Terrain=0.06; rightmost column of Table 2), starting from an initial entropy of 1.15, QMAX drives the entropy of $Z _ { \psi _ { 0 } } ( o , \pi _ { \theta } ( o )$ down to 1.05 while TEMPO raises it to 1.41 — a 34% relative gap that is temporally aligned with the return gap (9.9 vs 14.0, +41% relative). Elevated critic entropy preserves broader gradient support and thus wider local action coverage, exactly the mechanism the regularizer is designed to induce. This is also consistent with our results in Table 3, where removing the entropy term reduces performance.

![](images/18ca7c50e97e6e57dc9d18baeef11fddd4a803cde1b0994b8dfed38c4c1f575f.jpg)

![](images/fc7fdeb1cf78f5614cf8ddaf1a71573e9a719500bdf6561d6d982090eca0bd4d.jpg)

![](images/b6d3704376752d928a5685e304144ce30234fce4549fc4cb34c3d995289416c5.jpg)

![](images/c7626a0be27a1fe8d7732bd4e904bb1025a165896839b4e06370f9ffdb5f055e.jpg)  
Figure 10: Angular velocity tracking on Go2 under deployment-time distribution shift. We report angular velocity tracking error as a function of shift severity for four corruption families. Results are averaged over five seeds. Critic-guided test-time adaptation improves tracking accuracy relative to no adaptation, with TEMPO achieving the most stable and accurate angular velocity tracking, particularly under severe distribution shifts.

## B.7 RMA Setup

Due to the space constraints, we breifly outlined the experimental setup of RMA in Section 5.6. Here we provide the remaining training details for full reproducibility. RMA policy is trained on Go2JoystickFlatTerrain with MuJoCo Playground domain randomization enabled. The training randomization covers floor friction U(0.4,1.0), joint friction scale U(0.9,1.1), armature scale U(1.0,1.05), center-of-mass shift U(-0.05,0.05), link-mass scale U(0.9,1.1), torso mass offset U(-1,1) kg, and initial joint jitter U(-0.05,0.05). The RMA target conditions are deliberately outside these ranges: e.g., floor friction 0.1–0.2, torso payload 5–10 kg, calf payload 3–5 kg, and rough-terrain height 0.05–0.1, where terrain height is not part of the flat-terrain training randomization.

## B.8 Refreshing Scores of Memory

Under long hoizon streams, scores in the memory bank for previously observed observations could become stale hindering the positive impact of adaptation. A natural fix to this problem is to integrate refreshing the scores for the observations in the memory bank. For that, we equipped TEMPO with either of the following two score refreshing variants: (i) periodic score refresh, and (ii) age-decayed scoring, where Eq. (3) is replaced by $\hat { S } ( o ) = S ( o ) / ( 1 + \gamma \cdot \mathrm { a g e } ( o ) )$ with age(o) the number of steps o has resided in the memory. We swept $\stackrel { \prime } { \gamma } \in 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 .$ , 100. We observe that both variants added per-step compute without improving returns over the current minimum-score replacement rule. We attribute this to the limited state diversity of our simulation; in real-world, long-horizon deployment with richer diversity, we expect stale-score handling to matter more.

## B.9 Comparison to Test-Time Training

Recently, Park et al. [2024] proposed OPEX and Test Time Training as adaptation methods for policies subjected to shifts. In Figure1, we analyzed the efficacy of OPEX. For completeness, we discuss here the performance of TTT. In particular, one can express the adaptation loss for TTT as: $\mathcal { L } _ { \mathrm { T T T } } = \mathcal { L } _ { \mathrm { Q M A X } } + \beta \mathrm { K L } ( \pi _ { \theta } \| \pi _ { \theta _ { 0 } } )$ . Thus, we compare our proposed baseline; QMAX versus TTT. We evaluate on Go2 under the mass-on-calf shift and sweep the regularization coefficient $\beta .$

We observe a monotonic decrease in performance as $\beta$ increases. For MC=4, the return drops from 16.8 at β = 0 (QMAX) to 16.3, 16.1, 15.7, and 15.6 for $\beta = \{ 0 . 0 2 , 0 . 0 4 , 0 . 0 6 , 0 . 0 8 \}$ , respectively. Similarly, for MC=5, performance decreases from 14.7 at $\beta = 0$ to 14.4, 13.7, 13.6, and 13.3. Thus, the strongest TTT configuration corresponds to $\beta = 0$ , which exactly recovers QMAX.

## C Compute Requirements, Limitations, and Broader Impact

Compute Requirements. All our experiments that were conducted on simulation were run on NVIDIA V100 GPUs including training, evaluation, and TTA (each experiment was run on a single GPU). Regarding the real world experiments, we ran our TTA pipeline on the Unitree Go2 robot leveraging four CPU cores (2 for inference stream and 1 for adaptation and 1 for odometry).

Limitations. Our study focuses on reward-free, critic-guided test-time adaptation for locomotion policies under deployment-time distribution shifts, and some limitations remain. First, while TEMPO is designed to be lightweight and on-device friendly, it still relies on access to a pretrained critic that is reasonably informative under shift; if the critic is severely miscalibrated or trained on a highly mismatched domain, adaptation quality may degrade. Second, TEMPO induces an additional, albeit small, memory and compute cost during inference.

Broader Impact. This work aims to improve the robustness of robotic systems under deploymenttime distribution shifts, reducing the need for manual recalibration and improving autonomy. However, adaptive systems may introduce risks if deployed without adequate testing and oversight, especially in safety-critical environments. We emphasize that our method is intended to complement, not replace, careful system design, and we encourage responsible use in real-world settings.