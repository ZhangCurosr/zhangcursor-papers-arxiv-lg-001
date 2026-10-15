# Adversarial Reinforcement Learning for Robust Planetary Landing

Miguel Leiva-Vélez<sup>∗1,2</sup>, Nicolas Gaston-Rozado<sup>2</sup>, Andrea Scorsoglio<sup>3</sup>, Hodei Urrutxua<sup>4</sup>, Víctor Rodríguez-Fernández<sup>5</sup>

<sup>1</sup>ETSI Aeronáutica y del Espacio, Universidad Politécnica de Madrid, Madrid, Spain <sup>2</sup>Indra Sistemas, Madrid, Spain

<sup>3</sup>Department of Systems and Industrial Engineering, University of Arizona, Tucson, USA

<sup>4</sup>Escuela de Ingeniería de Fuenlabrada, Universidad Rey Juan Carlos, Fuenlabrada, Spain

<sup>5</sup>Department ofComputer Systems Engineering, Universidad Politécnica de Madrid, Madrid, Spain

Deep reinforcement learning (DRL) policies for planetary landing and other aerospace guidance, navigation, and control tasks are commonly trained under randomized disturbances to improve robustness. However, typical stochastic perturbation sampling assumes independent variations, whereas real mission disturbances often exhibit spatial and temporal correlation and cannot be reproduced through independent random draws. This work investigates adversarial reinforcement learning to discover challenging of-nominal conditions. A secondary agent generates physically constrained perturbations that degrade the performance of a landing controller, forming an asymmetric sequential zero-sum game. Training proceeds in three stages: baseline controller learning, robustness enhancement with stochastic perturbations, and an iterative adversarial phase in which the controller and disturbance generator are alternately optimized. Preliminary experiments in a three-degree-of-freedom landing environment show that adversarial disturbances reveal vulnerabilities missed by conventional randomization and provide early indications that, with a more extensive training campaign, the approach can yield improved robustness and stronger generalization under previously unseen perturbation regimes.

## 1 Introduction

Autonomous planetary landing is a safety-critical guidance, navigation, and control (GNC) problem characterized by nonlinear dynamics, limited control authority, and signif icant environmental and parametric uncertainty. During descent, the vehicle must satisfy strict terminal constraints while maintaining safety and fuel eficiency under adverse conditions, making robust performance essential.

Deep reinforcement learning (DRL) has emerged as a promising framework for autonomous landing, enabling direct learning of closed-loop control policies in simulated environments and handling nonlinear dynamics and longhorizon objectives. To improve robustness, most DRL approaches introduce stochastic perturbations or domain ran domization during training.

While efective for improving average-case performance, stochastic sampling does not necessarily capture missionrelevant disturbance patterns, which often evolve sequentially, interact with vehicle dynamics and flight phase, and exhibit spatial or temporal correlation. Because such structured scenarios occupy only a small subset of the disturbance space, they are unlikely to be discovered through independent random sampling, leaving potential failure modes unexplored.

This work proposes adversarial reinforcement learning to explore challenging disturbance scenarios in planetary landing. A secondary agent learns to generate temporally correlated perturbation sequences that degrade controller performance. The interaction between the landing controller and disturbance generator is formulated as an asymmetric sequential zero-sum game, allowing the adversary to search a high-dimensional disturbance space for dificult yet physically plausible scenarios.

The approach is evaluated in a three-degree-of-freedom lunar landing environment. Early results indicate that adversarially generated disturbances expose vulnerabilities not captured by conventional stochastic randomization and potentially lead to policies that remain efective under previously unseen and more severe perturbation regimes.

## 2 Related Work

Planetary landing GNC remains dominated in practice by classical guidance and model-based optimal control due to their interpretability and reliability. ZEM/ZEV-type guidance laws provide compact solutions for nominal scenarios [1], while convex optimization-based powered descent enables real-time trajectory generation with explicit constraint handling and strong theoretical guarantees [2, 3].

Recent academic work has explored DRL as a complementary approach for handling nonlinear dynamics and uncertain environments. Prior studies have demonstrated DRL-based powered descent control, algorithm comparisons, adaptive recurrent policies, hybrid classical-learning guidance architectures, transfer across planetary environments, and landing site selection under divert constraints [1, 4–8]. While promising, these approaches remain sensitive to reward design and distribution shift.

As already stated, most robustness-oriented DRL methods rely on stochastic perturbation sampling or domain randomization [5–7]. Although efective for improving average-case resilience, such approaches may undersample rare but mission-relevant disturbance sequences. Adversarial reinforcement learning addresses this limitation by modeling disturbance generation as a sequential decision process that actively searches for performancedegrading scenarios [9]. This work investigates the application of this idea to DRL-based planetary landing.

## 3 Problem Formulation

We consider autonomous planetary landing in a threedegree-of-freedom setting, where the lander is modeled as a point mass in a target-centered inertial frame with state

$$
\mathbf { x } ( t ) = { \left[ \begin{array} { l } { \mathbf { r } ( t ) } \\ { \mathbf { v } ( t ) } \\ { m ( t ) } \end{array} \right] } ,\tag{1}
$$

where $\mathbf { r } \in \mathbb { R } ^ { 3 }$ is position, $\mathbf { v } \in \mathbb { R } ^ { 3 }$ is velocity, and $m \in \mathbb { R } ^ { + }$ is mass. The translational dynamics are

$$
\dot { \mathbf { r } } = \mathbf { v } \ ,\tag{2}
$$

$$
\dot { \mathbf { v } } = \mathbf { g } ( \mathbf { r } ) + \frac { 1 } { m } \mathbf { T } _ { \mathrm { e f f } } + \mathbf { d } _ { \mathrm { e x t } } ( \mathbf { x } , t ) ,\tag{3}
$$

$$
{ \dot { m } } = - { \frac { \| \mathbf { T } \| } { I _ { \mathrm { s p } } g _ { 0 } } } ,\tag{4}
$$

where T is the commanded thrust, $\mathbf { g } ( \mathbf { r } )$ is the local gravity field, and $I _ { \mathrm { s p } }$ is the specific impulse. Internal actuation imperfections are captured through

$$
\mathbf { T } _ { \mathrm { e f f } } = \mathbf { T } + \mathbf { d } _ { \mathrm { i n t } } ( \mathbf { x } , \mathbf { T } , t ) ,\tag{5}
$$

where $\mathbf { d } _ { \mathrm { { i n t } } }$ models efects such as thrust bias, misalignment, or degradation, and $\mathbf { d } _ { \mathrm { { e x t } } }$ represents external disturbances. The objective is to learn a policy that achieves soft landing conditions,

$$
\mathbf { r } ( t _ { f } ) \approx \mathbf { 0 } , \qquad \mathbf { v } ( t _ { f } ) \approx \mathbf { 0 } ,\tag{6}
$$

while respecting mission and control constraints through out descent.

The controller operates under uncertainty and partial observability. At each time step, it receives a noisy observation

$$
\begin{array} { r } { \mathbf { o } _ { t } = h ( \mathbf { x } _ { t } ) + \pmb { \eta } _ { t } , } \end{array}\tag{7}
$$

where $h ( \cdot )$ is the observation model and $\pmb { \eta } _ { t }$ is sensor noise. The landing task is therefore modeled as a partially observable Markov decision process,

$$
( \mathcal { S } , \mathcal { A } , \mathcal { O } , P , R , \gamma ) ,\tag{8}
$$

with continuous thrust actions and reward function R designed to promote safe, accurate, and fuel-eficient descent. The controller seeks a policy maximizing the expected return

$$
\mathbb { E } \left[ \sum _ { t = 0 } ^ { t _ { f } } \gamma ^ { t } R ( \mathbf { x } _ { t } , \mathbf { T } _ { t } ) \right] .\tag{9}
$$

Perturbation generation is modeled as a sequential decision process in which, alongside the landing controller, a second agent acts as an adversary and selects physically admissible disturbances over time. Their interaction is formulated as an asymmetric zero-sum game: the protagonist seeks successful landing, while the adversary aims to maximally degrade performance through bounded perturbation sequences. Denoting the returns of the controller and adversary by $R _ { A }$ and $R _ { B }$ , respectively,

$$
R _ { A } + R _ { B } = 0 .\tag{10}
$$

## 4 Methodology

We study robust lunar landing in a simplified three-degreeof-freedom (3-DOF) environment, adopted here as an intermediate step toward subsequent extension to full sixdegree-of-freedom dynamics. Training is organized as a staged curriculum to promote stable policy acquisition under progressively more challenging conditions. A baseline controller is first optimized in the nominal environment with randomized initial conditions. This policy is then fine-tuned under stochastic perturbations of increasing intensity. In the final stage, an adversarial agent is introduced and trained iteratively against the robustified controller to generate disturbance sequences that maximally deteriorate landing performance.

The lander policy produces a continuous fourdimensional control action,

$$
\mathbf { a } _ { L } \in [ - 1 , 1 ] ^ { 4 } ,\tag{11}
$$

whose first three components specify thrust direction, subsequently normalized to unit norm, and whose fourth component sets thrust magnitude. During adversarial training, a second agent acts in the disturbance space by controlling three perturbation channels: the magnitude $F _ { \mathrm { e x t } }$ and direction $\hat { \mathbf { f } } _ { \mathrm { e x t } }$ of an external force, the thrust misalignment angle $\theta _ { \mathrm { m i s } } ,$ and the specific-impulse degradation $\ell _ { \mathrm { i s p } }$ . These perturbations are constrained to the same bounds used during stochastic robustness training. Rather than selecting absolute disturbance values at each time step, the adversary applies bounded increments, allowing perturbations to evolve smoothly over time and thereby preserving transient dynamics while avoiding unrealistic step changes.

Guidance is induced through a position-dependent reference velocity field $\mathbf { v } _ { \mathrm { r e f } } ( \mathbf { r } )$ , which steers the vehicle toward the landing site and transitions to a near-vertical descent in the terminal phase. At each step, reward is based on progress projected along the local streamline direction,

$$
\hat { \mathbf { v } } _ { \mathrm { r e f } } = \frac { \mathbf { v } _ { \mathrm { r e f } } ( \mathbf { r } _ { t - 1 } ) } { \Vert \mathbf { v } _ { \mathrm { r e f } } ( \mathbf { r } _ { t - 1 } ) \Vert } ,\tag{12}
$$

with instantaneous progress

$$
p _ { t } = \left( \mathbf { r } _ { t } - \mathbf { r } _ { t - 1 } \right) \cdot \hat { \mathbf { v } } _ { \mathrm { r e f } } ,\tag{13}
$$

and admissible progress

$$
p _ { t } ^ { \star } = \operatorname* { m a x } \left( \| \mathbf { v } _ { \mathrm { r e f } } ( \mathbf { r } _ { t } ) \| , v _ { \mathrm { f l o o r } } \right) \Delta t .\tag{14}
$$

The progress term is defined as

$$
r _ { t } ^ { \mathrm { p r o g } } = \left\{ \begin{array} { l l } { p _ { t } , } & { p _ { t } \leq p _ { t } ^ { \star } , } \\ { 2 p _ { t } ^ { \star } - p _ { t } , } & { p _ { t } > p _ { t } ^ { \star } , } \end{array} \right.\tag{15}
$$

so that motion beyond the admissible local rate is penalized. The resulting lander reward is

$$
r _ { t } ^ { L } = \frac { r _ { t } ^ { \mathrm { p r o g } } } { p _ { t } ^ { \star } + \varepsilon } - \lambda _ { T } \frac { \| \mathbf { T } _ { t } \| } { T _ { \operatorname* { m a x } } } .\tag{16}
$$

The adversary, by contrast, is trained using only a sparse terminal objective proportional to final landing error,

$$
R _ { \mathrm { a d v } } = \| \mathbf { r } ( t _ { f } ) - \mathbf { r } _ { \mathrm { p a d } } \| ,\tag{17}
$$

which concentrates learning on degradation of terminal performance without inheriting the protagonist reward shaping.

The observation vector supplied to the lander comprises vehicle and guidance variables $( \mathbf { r _ { \mathrm { ~ \scriptsize ~ - ~ } } r _ { \mathrm { p a d } } } , \mathbf { v _ { \mathrm { ~ \scriptsize ~ - ~ } } } \mathbf { v } _ { \mathrm { r e f } } , \| \mathbf { T } _ { t - 1 } \| , \hat { \mathbf { T } } _ { t - 1 } )$ , internal perturbation states $( \theta _ { \mathrm { m i s } } , \ell _ { \mathrm { i s p } } )$ , and external perturbation variables $\left( \mathbf { a } _ { \mathrm { p e r t } } , \lVert \mathbf { a } _ { \mathrm { p e r t } } \rVert \right)$ . All channels are normalized prior to network input. The adversary receives the same information, but encoded through a recurrent state to capture temporal correlations across the trajectory.

Performance is assessed under two perturbation regimes: in-distribution (ID) evaluation, in which disturbance magnitudes remain within the training range, and out-ofdistribution (OOD) evaluation, in which larger unseen perturbations are applied to test generalization. Robustness is analyzed by systematically sweeping the principal disturbance axes (external force magnitude, thrust misalignment, and specific-impulse eficiency loss) both independently and in combination, thereby probing the controller’s sensitivity to individual perturbations as well as to compounded degradation efects. For each configuration, disturbance directions are randomized at the beginning of each episode to avoid directional bias. Evaluation is performed over one hundred Monte Carlo rollouts per scenario to capture statistical variability in the resulting trajectories and landing outcomes, enabling consistent estimates of performance under progressively increasing perturbation levels and across multi-axis disturbance conditions.

Algorithmically, Proximal Policy Optimization (PPO) [10] is the selected DRL paradigm both for the main policy and the adversarial one.

## 5 Results

We first analyze learning behavior in the nominal environment. As shown in Figure 1, the baseline policy trained with PPO exhibits fast convergence and stable learning despite randomized initial conditions. This indicates that the reward formulation provides a consistent training signal and enables the policy to generalize across the nominal state distribution.

![](images/2cec428fe768554ccbaa27ffdbaec01eeebb5790ef9d7493c6487885f4c43ee5.jpg)  
Figure 1: Training curves across the curriculum

Representative evaluation trajectories are shown in Figure 2. The policy learns a stable descent geometry and consistently reaches the landing region under nominal conditions.

![](images/6260f9d60d7695191352341ed518f209358e2abfb5d5def0e1211daa6f4140df.jpg)  
Figure 2: Representative 3D landing trajectories.

During stochastic fine-tuning, the agent rapidly compensates for the baseline-policy performance degradation introduced by perturbations, which explains the relatively small variation visible in subplot b) of Figure 1. Nevertheless, increasing disturbance levels still reduce landing precision, particularly in the terminal phase.

Introducing the adversarial agent exposes disturbance sequences that significantly degrade the stochastically finetuned frozen policy, even within the same perturbation bounds used during stochastic training. The alternating dynamics visible subplot c) in Figure 1 show the expected exploit-adapt pattern, where the adversary first degrades the policy and the protagonist subsequently learns to counteract the discovered perturbation strategy.

This efect is illustrated in Figure 3, where the stochastically fine-tuned policy shows degraded landing precision when evaluated against the adversary, if compared to the adversarially-hardened one.

![](images/fb2c0e981a4894ed2b6cd98671e5f0cbf7ffd1a81ae7345dadc99ba2886b9515.jpg)  
Figure 3: Violin plot comparing landing performance of baseline, fine-tuned, and adversarially trained policies against adversarial disturbances.

Then, robustness and generalizability across perturbation levels is evaluated through a Monte Carlo campaign where each policy is tested under progressively increasing disturbances applied individually and jointly. The results are summarized in Figure 4. The baseline controller degrades immediately once perturbations are introduced. The stochastically fine-tuned policy shows a clear improvement across most perturbation conditions. The adversarially trained controller demonstrates stronger robustness at higher disturbance levels, although some degradation appears in lower-perturbation regions.

## 6 Discussion

These results highlight three main points. First, the nominal PPO policy learns a stable and generalizable descent strategy under randomized initial conditions, indicating that the proposed reward formulation is suficient to shape the landing behavior without requiring additional guidance heuristics. This is an important starting point, since the subsequent robustness analyses depend on the existence of a well-behaved baseline controller.

Second, stochastic perturbation training clearly improves robustness relative to the nominal policy, but it does not fully characterize the vulnerability space of the controller. Although the fine-tuned agent performs well under randomly sampled disturbances, the adversarial experiments show that temporally correlated perturbation sequences within the same admissible bounds can still induce substantial degradation. This suggests that robustness assessed only through random sampling may overestimate the true resilience of the policy, particularly in safety-critical phases such as terminal descent.

![](images/48d6620c31c2aa4bb35b38888af82ca9d0683f33cb6c0093872bb1c17b977121.jpg)  
Figure 4: Monte Carlo evaluation of baseline, stochastic finetuned, and adversarially trained policies under increasing perturbation levels applied individually and jointly.

Third, the initial adversarial campaign demonstrates both the promise and the limitation of adversarial reinforcement learning in this setting. On one hand, adversarial training exposes failure modes that stochastic training alone does not reveal, and improves performance under stronger perturbations. On the other hand, the current results also indicate partial specialization to the adversarial regime, with some loss of precision in milder conditions. This points to the need for a more balanced training procedure in which adversarial, stochastic, and nominal scenarios are continuously interleaved to avoid overfitting to a narrow subset of disturbance realizations.

Overall, this study should be interpreted as a preliminary but encouraging step toward robust RL-based planetary landing. The results support the use of adversarial agents as active disturbance discoverers, but also indicate that robust policy synthesis will likely require broader training schemes, such as mixed-distribution curricula or league-based training inspired by large-scale multi-agent reinforcement learning systems such as AlphaStar [11], to achieve uniform performance across the full perturbation envelope.

## References

1. Furfaro, R., Scorsoglio, A., Linares, R. & Massari, M. Adaptive generalized ZEM-ZEV feedback guidance for planetary landing via a deep reinforcement learning approach. Acta Astronautica 171, 156–171 (2020).

2. Blackmore, L., Açikmeşe, B. & Scharf, D. P. Minimum-landing-error powered-descent guidance for Mars landing using convex optimization. Journal of guidance, control, and dynamics 33, 1161–1171 (2010).

3. Açıkmeşe, B., Carson, J. M. & Blackmore, L. Lossless convexification of nonconvex control bound and pointing constraints of the soft landing optimal control problem. IEEE transactions on control systems technology 21, 2104–2113 (2013).

4. Gaudet, B., Linares, R. & Furfaro, R. Deep reinforcement learning for six degree-of-freedom planetary powered descent and landing. arXiv preprint arXiv:1810.08719 (2018).

5. Xu, X., Chen, Y. & Bai, C. Deep reinforcement learning-based accurate control of planetary soft landing. Sensors 21, 8161 (2021).

6. Gaudet, B. & Linares, R. Adaptive guidance with reinforcement meta-learning. arXiv preprint arXiv:1901.04473 (2019).

7. Ciabatti, G., Daftry, S. & Capobianco, R. Autonomous planetary landing via deep reinforcement learning and transfer learning in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021), 2031–2038.

8. Iiyama, K., Tomita, K., Jagatia, B. A., Nakagawa, T. & Ho, K. Deep reinforcement learning for safe landing site selection with concurrent consideration of divert maneuvers. arXiv preprint arXiv:2102.12432 (2021).

9. Pinto, L., Davidson, J., Sukthankar, R. & Gupta, A. Robust adversarial reinforcement learning in International conference on machine learning (2017), 2817–2826.

10. Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

11. Vinyals, O. et al. Grandmaster level in StarCraft II using multi-agent reinforcement learning. nature 575, 350–354 (2019).