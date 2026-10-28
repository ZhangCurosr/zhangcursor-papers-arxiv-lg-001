# Multi-Agent Reinforcement Learning of Co-Adaptive Hip Exoskeleton Control (MARL-Exo) for Metabolic Energy Reduction

Yifei Yuan, Jakob Wolf, Ghaith Androwis, and Xianlian Zhou

Abstract—Objective: This study aims to determine whether a hip-exoskeleton policy trained in simulation, by co-adapting with a musculoskeletal human model under a metabolic-energy reward, reduces the metabolic cost of walking in new users without per-user tuning, and how mechanical-power shaping affects the learned assistance. Methods: We developed MARL-Exo, a multi-agent reinforcement learning framework in which a full-body human model (304 muscle actuators; randomized anatomy, gait, and speed) and a bilateral hip-exoskeleton policy are trained concurrently in two stages. The exoskeleton observes only delayed hip kinematics and its past commands; its reward includes muscle metabolic power from a physiologically based energetics model, alone (Metabolic) or with mechanical-power shaping (Metabolic + Power). The Metabolic policy was deployed and tested without retraining in eight healthy adults walking at 1.25 m/s, using indirect calorimetry. Results: In simulation, the Metabolic policy reduced modeled muscle metabolic power by 13.8% relative to no-device walking, versus 8.8% for Metabolic + Power. On hardware, MARL-Exo reduced net metabolic rate by 18.3 ± 6.6% relative to Passive in all eight participants. Relative to No-Exo, the mean reduction was $\mathbf { 7 . 4 \pm 1 0 . 0 \% } ;$ six participants more than offset the 13.1% device-wearing penalty, walking up to 24.2% below their No-Exo cost. Conclusion: A metabolically informed policy coadapted in simulation transferred to users without per-user tuning and consistently reduced the metabolic cost of walking with the device, while explicit power shaping reduced the simulated benefit. Significance: Unlike per-user human-in-theloop optimization, this approach yields a single population-level controller and allows assistance objectives to be screened in simulation before human testing.

Index Terms—Exoskeletons, physical human–robot interaction, reinforcement learning, co-adaptation, metabolic cost, wearable robotics.

## I. INTRODUCTION

P <sup>OWERED</sup> <sup>lower-limb</sup> <sup>exoskeletons</sup> <sup>are</sup> <sup>being</sup> <sup>developed</sup>f f walking [1], [2]. Among them, hip exoskeletons are especially attractive because the hip musculature performs a large fraction of the positive work of level locomotion, so assistance there can offload muscles that do much of the work of walking. Delivering that benefit efficiently is challenging because the wearer is not a passive plant. Once a device applies torque at a joint, the mechanics of gait change, and the neuromuscular system responds by recruiting muscles differently and re-coordinating the limbs [3]. Each side of the interaction thus changes the operating conditions of the other, and the controller is tuned against a wearer who is still adapting. The problem is compounded by the choice of objective: active devices can shape the assistance profile flexibly, but the benefit they deliver depends strongly on what the controller is optimized for, and on how well that objective reflects the wearer’s response. Human-in-the-loop optimization tunes the objective directly on the wearer, but it demands long peruser experiments that must be repeated for every user and condition [4], [5].

Learning the controller in musculoskeletal simulation offers a way around those experiments, since the wearer’s response can be simulated rather than measured. A neural network control policy for a full lower limb exoskeleton coupled to a musculoskeletal model was first used to learn robust assistance for walking [6], and a similar idea was used for joint human–exoskeleton training with sim-to-real transfer that produced a control policy for a hip exoskeleton that lowered metabolic cost without subject-specific tuning [7]. The paradigm has since been extended to speed- and slopeaware assistance through muscle-synergy priors [8], to reflexbased human models paired with end-to-end hip policies [9], to human-aligned simulation of assistance parameters [10], and to asymmetric gait [11], squatting [12], sit-to-stand [13], and running [14].

Two gaps remain in this literature. First, the mutual adaptation of human and device is mostly left implicit: when both policies are optimized at once, each shifts the data distribution seen by the other, and training can drift toward oscillatory torque, mistimed assistance, or divergence. Second, the effort terms used in training are typically proxies such as muscle activation or work. Luo et al. [7] explicitly chose activation over a metabolic-rate term because of calibration and rewardsmoothness concerns, and other controllers use kinematic tracking or activation-based effort [8], [12]. What a wearer ultimately experiences is metabolic energy, which a musculoskeletal simulation can compute directly but which, to our knowledge, has not been used as a term of the exoskeleton policy reward. The closest examples are of two kinds. Exo-Plore rewards a simplified activation-weighted energy expression rather than a physiological muscle-energetics model [10]. Barati et al. [9] optimize their reflex-based human model with Umberger’s energetics as the objective; however, this objective was not incorporated into the exoskeleton policy reward, which instead used an activation-based effort term.

Curriculum learning is a natural remedy for the first gap, since decomposing a hard learning problem into ordered stages improves stability [15] and physiological plausibility [16], [17]. We previously proposed Staged Multi-Agent Training (SMAT), a four-stage curriculum trained on a compact 26-muscle model [18], [19]. Here we ask two further questions. Can co-adaptive training remain stable, and transfer to real users, when the human is a far richer full-body model with a population of body sizes and gaits? And how does physiological energetics, with and without explicit mechanicalpower shaping, influence the assistance learned through human–exoskeleton co-adaptation, and can the resulting policy transfer to human users?

This study addresses both questions with MARL-Exo, a multi-agent framework built on a full-body model with 304 Hill-type muscle actuators. Its human policy outputs residual joint targets adapted to a reference gait trajectory, which a learned muscle-activation layer converts into muscle activations. Its exoskeleton policy sees only intentionally delayed hip kinematics and its own recent commands, so that it can run on inexpensive wearable sensing. Training uses a twostage schedule (human-only gait, then joint co-adaptation), and the exoskeleton reward includes the metabolic power of the hip muscles, evaluated with a muscle-energetics model, either alone (Metabolic) or together with explicit mechanicalpower shaping (Metabolic + Power). The Metabolic policy was deployed on the same hardware and eight-participant cohort of the SMAT evaluation [19]. The contributions are as follows:

1) the use of a physiologically based muscle-energetics model as a term of the exoskeleton control policy’s reward, within a two-stage multi-agent framework that couples a full-body human model and a hip exoskeleton, through regularized constraints and conditions the policy on randomized body and gait parameters;

2) a simulation comparison showing that, with the metabolic term retained, removing explicit positivepower shaping lowered modeled metabolic power across the analyzed muscle set by 13.8 % relative to no-device walking, against 8.8 % when power shaping was included; and

3) a human validation in eight healthy adults, in which the deployed policy lowered net metabolic rate by 18.3 % relative to the passive device in all participants (adjusted $\begin{array} { r l r } { p } & { { } < } & { 0 . 0 0 1 ) } \end{array}$ , more than offset the device’s 13.1 % wearing penalty in six of eight participants (mean 7.4 % and up to 24.2 % below no-device walking), enlarged hip range of motion by 36.8 %, and delivered predominantly positive estimated assistance power (98.5 %).

The rest of this paper is organized as follows. Section II describes the simulation model and training framework, Section III the hardware and human experiment, Section IV the simulation and experimental results, and Sections V and VI discuss and conclude.

## II. COOPERATIVE MULTI-AGENT TRAINING

## A. Human–Exoskeleton Coupling

The human and the hip exoskeleton are modeled as two separate multibody systems in a shared physical simulation world simulated with the DART engine [20]; each has its own segment, joint, and inertia definitions. The human musculoskeletal model has 50 joint degrees of freedom (DOFs) (excluding the six global or root DOFs) and is driven by 304 Hill-type muscle actuators (line-of-action elements, several of which can represent one anatomical muscle) [21], [22]. The hip exoskeleton weighs 5.94 kg and has 4 DOFs, including two passive abduction/adduction DOFs and two actuated flexion/extension DOFs. The hip exoskeleton is rigidly attached to the human model through three permanent bilateral constraints resolved by the constraint solver: a weld fusing the pelvis to the exo backpack (all six DOFs locked) and a ball joint pinning each femur to the corresponding exo thigh segment (the three translational degrees of freedom locked). The constraints are regularized rather than ideally rigid: DART enforces them through finite constraint-force-mixing and errorreduction parameters, so the interface carries solver-level compliance while remaining much stiffer and cheaper to solve than explicit elastic or bushing connections [23], [6]. Because the ball joints leave hip rotation free, the wearer’s hip degrees of freedom are not over-constrained; we expect the regularization to affect mainly transient load transmission rather than the net sagittal-plane assistive torque. Exoskeleton motor torques (peak at ±18 N m) act on the device’s hip joints and are transmitted to the human pelvis and thighs through these constraints, altering the muscle effort required to produce the gait.

## B. Reinforcement Learning Framework

Hip assistance is posed as a multi-agent reinforcement learning (MARL) problem [24], [25] with two cooperating agents that share one simulated world (Fig. 1): a human policy $\pi _ { H }$ that drives the musculoskeletal model and an exoskeleton policy $\pi _ { E }$ that commands hip torque. Each agent has its own observation, reward, value function (critic), and PPO update [26], and the two learn from one shared batch of rollouts. This framework shares its multi-agent formulation with SMAT [18] but differs from it in the human model (304 rather than 26 muscle actuators, with randomized anatomy and gait), the action pathway, the curriculum (two stages rather than four), the exoskeleton reward, and separate agentspecific rewards and critics (Table I).

Observations. The human observation $s _ { t } ^ { H }$ is a full-body proprioceptive vector: the center-of-mass (COM) height; each body node’s COM position relative to the whole-body COM $\textstyle ( \in \mathbb { R } ^ { 3 } )$ and orientation (first two rows of its world rotation matrix, $\in \mathbb { R } ^ { 6 } ) ;$ each body node’s linear velocity relative to the COM velocity, the COM linear velocity, and each angular velocity (scaled by 0.1); the gait phase $\phi _ { t }$ (with the local phase appended when the time-warp is active); the longitudinal COM-to-next-foothold distance; the scaled joint state (current and target angles, ×0.008); and the sampled or randomized anatomical parameters (gait, skeleton-scale, and muscle-force factors) within their sampled ranges. These parameters include whole-body and per-segment skeleton scales (±10% global, ±5% per segment), the maximum isometric force of each muscle (±25%), and gait stride and cadence (±25%). During training, the target walking speed was sampled from 0.65 m/s to 2.0 m/s, and the height and weight of the human model were randomized between [53.6 kg, 1.55 m] and [97.1 kg, 1.94 m]. Since these sampled parameters are appended to the policy’s observation, the controller is explicitly conditioned on the body and gait it drives rather than trained to be invariant. This parameterconditioned domain randomization enables a single training to produce appropriate gaits across the sampled population. The resulting human observation has 649 dimensions, and the human policy outputs 51 actions: 50 residual joint-target offsets and one phase time-warp term, low-pass filtered with a 6 Hz Butterworth filter at the 50 Hz control rate. The residuals are added to a reference trajectory consisting of a single symmetrized, height-calibrated level-walking cycle which acts as a motion prior; no motion-imitation reward is used.

![](images/fe5c809c62cdab77f3f7095cace55418180f89d6f73d780d3e9e7fd4f3ca83db.jpg)  
Fig. 1. Overview of MARL-Exo. (a) Training in simulation: a human policy $\pi _ { H }$ and an exoskeleton policy $\pi _ { E }$ act on a shared physics simulation; human actions pass through stable proportional-derivative (SPD) control and an auxiliary muscle network $\Phi ^ { \psi }$ , and the exoskeleton reward is computed by a muscle metabolic energy model. Stage 1 trains $\pi _ { H }$ alone, and Stage 2 adds the exoskeleton and co-adapts both policies. Multi-agent PPO updates both policies from a shared rollout batch $\mathcal { D } ,$ while $\Phi ^ { \psi }$ is trained by supervised regression to reproduce the desired joint torques $\tau _ { \mathrm { d e s } }$ through the muscle moment-arm mapping $J ^ { \top } A$ (line 22 of Algorithm 1). (b) The trained $\pi _ { E }$ is deployed on a Raspberry Pi 5 without subject-specific tuning and validated in eight adults.

TABLE I  
MAIN DIFFERENCES BETWEEN MARL-EXO (THIS WORK) AND SMAT [18], [19].
<table><tr><td>SMAT</td><td></td><td>MARL-Exo</td></tr><tr><td>Human model</td><td>imitation reward</td><td>26 muscles (13 per side), 50-DOF body, 304 Hill- type muscle actuators, gait reward</td></tr><tr><td>Human action</td><td>Muscle excitations</td><td>Residual joint targets on a reference trajectory → PD torque → learned muscle</td></tr><tr><td>Randomization</td><td>None</td><td>network Anatomy, strength, gait, speed in observation</td></tr><tr><td>Curriculum Critic</td><td>Four stages Shared by both actors</td><td>Two stages</td></tr><tr><td>Exo reward</td><td>ness</td><td>One per agent Power sign and smooth- Smoothness, magnitude,</td></tr><tr><td></td><td></td><td>hip-muscle metabolic power, and weak human- reward coupling (deployed</td></tr><tr><td rowspan="2">Online filtering</td><td colspan="2">policy) None 3 Hz angle and velocity</td></tr><tr><td colspan="2">Experimental data Same cohort and session; rest, No-Exo, and Passive recordings shared; active trials order randomized</td></tr></table>

The exo observation $s _ { t } ^ { E }$ is compact and delayed: for each of the two hip motors it stacks the joint angle and angular velocity over the last $n _ { \mathsf { h i s t } } { = } 3$ frames, delayed to two control frames (i.e. 40 ms) and low-pass filtered with a second-order Butterworth filter at 3 Hz, concatenated with the last $n _ { a } { = } 3$ exo actions (18 dimensions total). It outputs normalized bilateral hip torque commands $\hat { \mathbf { u } } \in [ - 1 , 1 ] ^ { 2 }$ , scaled by torque limits $\tau _ { \mathrm { m a x } }$ to produce physical assistance, through an MLP with hidden sizes [128, 64]. In all exoskeleton training, an auxiliary symmetry loss (weight 10) added to the PPO objective penalized differences between the policy’s output and its mirrored output for left–right mirrored observations.

## C. Two-Stage Training

Training both agents jointly from scratch can often become unstable, as the exoskeleton optimizes against a human policy that has not yet acquired stable gait. We therefore separate training into two sequential stages: locomotion is first learned without the device, and the mechanically coupled exoskeleton is introduced only in the second stage, where both policies are optimized jointly. Algorithm 1 describes the training loop used in both stages; the stages differ in the presence of the exoskeleton and in the active reward terms.

Human Baseline Gait Learning. Stage 1 trains the human policy alone, without the device, to walk stably at varied speeds, and the resulting policy is the starting point for coadaptation. With a gait-reward gain $w _ { g } = 2$ , the human reward multiplies locomotion, velocity, and foot-placement terms and adds an activation-based effort term plus auxiliary regularizers:

Algorithm 1 Cooperative MARL training of a human policy   
and a hip-exoskeleton policy with an auxiliary muscle actuator   
net $\Phi ^ { \psi }$   
Require: PPO hyperparameter configurations; iterations $I ;$   
workers $N ;$ muscle moment-arm matrix $J ^ { \top } A ;$ exo motor   
torque limit $\tau _ { \mathrm { m a x } } ;$ reference joint trajectory $q ^ { \mathsf { r e f } } ( \phi )$ (gait   
phase $\phi )$ for the residual policy; action scale $c _ { a } = 0 . 0 4$   
1: Init human policy $\pi _ { H } ^ { \theta _ { H } }$ and exo policy $\pi _ { E } ^ { \theta _ { E } }$   
2: Init muscle net $\Phi ^ { \psi } ;$ broadcast $\psi$ to all workers   
3: for i = 1 to I do   
4: for worker $n = 1$ to N do in parallel   
5: while episode not terminated do   
6: s<sup>H</sup> ← GETHUMANSTATE   
7: $s _ { t } ^ { \check { E } } \gets @ \mathsf { E T E } \mathsf { x } 0 \mathsf { S } \mathsf { T A T E }$   
8: $\mathring { a _ { t } ^ { H } } \sim \pi _ { H } ( \cdot \vert s _ { t } ^ { H } ) ; \ : a _ { t } ^ { E } \sim \pi _ { E } ( \cdot \vert s _ { t } ^ { E } )$   
9: target angles $q _ { t } ^ { \mathsf { t a r g e t } } \gets q _ { t } ^ { \mathsf { r e f } } ( \phi ) + \bar { c } _ { a } a _ { t } ^ { H }$   
10: desired human joint torque $\tau _ { \mathsf { d e s } } \gets \mathsf { S P D } ( q _ { t } ^ { \mathsf { t a r g e t } } )$   
11: muscle activations $\pmb { \alpha }  \Phi ^ { \psi } ( J ^ { \top } A , \ \tau _ { \mathsf { d e s } } ) \qquad \triangleright$   
applied each substep   
12: exo torque $\tau _ { E } \gets a _ { t } ^ { E } \times \tau _ { \mathrm { m a x } }$   
13: ENV.STEP ▷ advance one control step (ϕ)   
(n physics substeps, and advance phase(ϕ))   
14: r<sup>H</sup> ← GETHUMANREWARD   
15: r E ← GETEXOREWARD   
16: store $( s _ { t } ^ { H } , a _ { t } ^ { H } , r _ { t } ^ { H } ) , \ : ( s _ { t } ^ { E } , a _ { t } ^ { E } , r _ { t } ^ { E } )$ in $\mathcal { D } ;$ cache   
$\left( J ^ { \top } A , \ \tau _ { { \mathsf { d e s } } } \right)$   
17: terminate on end-of-episode / NaN   
18: end while   
19: end for   
20: compute per-agent GAE advantages from D   
21: $\theta _ { H } \stackrel { \cdot } {  } \theta _ { H } + \eta \stackrel { \cdot } { \nabla } _ { \theta _ { H } } J _ { H } ^ { \mathrm { c l i p } } ; \theta _ { E }  \bar { \theta _ { E } } + \eta \nabla _ { \theta _ { E } } J _ { E } ^ { \mathrm { c l i p } }$   
// Auxiliary supervised muscle learning   
22: α ← Φ<sup>ψ</sup>(J<sup>⊤</sup>A, τ<sub>des</sub>); ψ ← arg min $_ { \psi } \parallel ( J ^ { \top } A ) \alpha -$   
$\tau _ { \mathrm { d e s } } \| ^ { 2 } + \lambda \| \alpha \| ^ { 2 }$ (Adam); broadcast ψ   
23: $\bar { r } ^ { H } , \bar { r } ^ { E } \gets$ mean episode reward of the human and   
exo agent over the rollouts of iteration i   
24: end for   
25: return $\pi _ { H } ^ { \theta _ { H } } , \ \pi _ { E } ^ { \theta _ { E } } , \ \Phi ^ { \psi }$

$$
\begin{array} { r } { r _ { t } ^ { H } = w _ { g } r _ { \mathsf { l o c o } } r _ { \mathsf { a v g } } r _ { \mathsf { s t e p } } + w _ { \mathsf { e f f } } r _ { \mathsf { e f f } } + \sum _ { k } r _ { \mathsf { a u x } } ^ { k } , } \end{array}\tag{1}
$$

$$
r _ { \mathrm { l o c } 0 } = e ^ { - 4 \| \Delta \dot { x } _ { \mathrm { h e a d } } \| ^ { 2 } } \ e ^ { - 4 \Delta \theta _ { \mathrm { h e a d } } ^ { 2 } } , \quad r _ { \mathrm { a v g } } = e ^ { - 6 \| \bar { v } _ { \mathrm { c o m } } - v ^ { * } \hat { z } \| ^ { 2 } } ,\tag{2}
$$

$$
r _ { \mathsf { s t e p } } = e ^ { - 2 \| \Delta _ { \mathsf { t o o t } } \| ^ { 2 } / n } , \qquad r _ { \mathsf { e f f } } = e ^ { - 8 \overline { { \| \alpha \| ^ { 2 } } } / N _ { m } } ,
$$

where $\Delta \dot { x } _ { \mathrm { h e a d } }$ and $\Delta \theta _ { \mathrm { h e a d } }$ are the head linear-acceleration and orientation deviations, $\bar { v } _ { { \mathsf { c o m } } }$ is the current COM velocity, $v ^ { * } = \ell _ { \mathsf { s t r i d e } } / T _ { \mathsf { s t r i d e } }$ is the target COM speed, zˆ is the forward direction vector, $\Delta _ { \mathrm { f o o t } }$ is the dead-zone-clipped swing-foot error, $\overline { { \| \pmb { \alpha } \| ^ { 2 } } }$ is the mean squared muscle activation over the control interval $( N _ { m }$ muscles), and $w _ { \mathsf { e f f } } { = } 0 . 2$ . The term ${ r } _ { \mathtt { e f f } }$ is an activation proxy for human effort and is distinct from the physiological energetics term of the exoskeleton reward below. The auxiliary terms $r _ { \mathrm { a u x } } ^ { k }$ penalize joint-limit violations and insufficient foot clearance and reward smooth joint torques (formulas and constants in Supplementary Section I).

Human–Exoskeleton Co-Adaptation. This stage begins the joint training of the human and exoskeleton policies. The human control policy is loaded from the human baseline training.

The exo reward sums an action-smoothness penalty, an action-magnitude penalty, an optional joint-power term, a muscle metabolic-power term, and a small coupling to the human reward:

(3)

$$
\begin{array} { r l } & { r _ { t } ^ { E } = \mathrm { ~ - ~ } \big ( 0 . 2 \| \Delta a ^ { E } \| ^ { 2 } + \| \Delta ^ { 2 } a ^ { E } \| ^ { 2 } \big ) - 0 . 0 1 \| a ^ { E } \| ^ { 2 } } \\ & { ~ + ~ w _ { \mathsf { p } } r _ { \mathsf { p o w e r } } + r _ { \mathsf { e n e r g y } } + 0 . 0 2 r _ { t } ^ { H } , } \\ & { ~ r _ { \mathsf { p o w e r } } = \displaystyle \sum _ { j } \big [ \operatorname* { m a x } ( 0 , P _ { j } ) - 5 \operatorname* { m a x } ( 0 , - P _ { j } ) \big ] , } \\ & { ~ \quad ~ r _ { \mathsf { e n e r g y } } = \operatorname* { m a x } \Big ( 1 - \frac { \dot { E } _ { \mathsf { U o h i d a } } } { \dot { E } _ { 0 } } , ~ - 5 \Big ) , } \end{array}\tag{4}
$$

where $\Delta a ^ { E } , \ \Delta ^ { 2 } a ^ { E }$ are the first/second temporal differences of the exo action, $\begin{array} { r } { P _ { j } ~ = ~ \tau _ { j } ^ { E } \dot { q } _ { j } } \end{array}$ is the mechanical power at hip motor j (so positive/assistive power is rewarded and negative/resistive power penalized $5 \times )$ , and $\dot { E } _ { \mathrm { U c h i d a } } \left( \mathsf { W } \right)$ is the Uchida-model metabolic power of the hip muscles detailed below, normalized by a constant $\dot { E } _ { 0 } = 1 5 0 \mathsf { W } ;$ the floor at −5 $( \mathsf { i . e . , ~ \dot { E } _ { U c h i d a } } \geq 6 \dot { E } _ { 0 } )$ guards against extreme values caused by simulation instabilities. Two exoskeleton controllers were trained whose rewards differ only in $w _ { \mathsf { p } } \mathrm { . }$ : Metabolic + Power $( w _ { \mathsf { p } } ~ = ~ 0 . 0 0 5 )$ and Metabolic $( w _ { \mathsf { p } } \ = \ 0 )$ . The names refer to the terms being compared, not the entire objective; both retain the smoothness, magnitude, metabolic, and humancoupling terms. Both controllers were obtained within the coadaptation stage. The Metabolic+ Power controller was coadapted from the Stage-1 human policy until its exoskeleton reward plateaued. The Metabolic controller was then initialized from the converged best checkpoint and co-adaptation continued with the power term removed, so the deployed policy is a reward refinement within Stage 2 rather than an independently trained controller. The Metabolic controller is the one deployed on hardware.

Physiologically Based Metabolic Energy Model. $\dot { E } _ { \mathrm { U c h i d a } }$ follows the muscle-energetics model of Umberger et al. [27] as implemented by Uchida et al. [28], summed over the muscle actuators that span the hip flexion/extension degree of freedom of either leg, i.e., all actuators with a nonzero moment arm about that axis. In the present model these are 46 actuators per side (92 in total), representing adductor brevis, longus, and magnus, gluteus maximus, medius, and minimus, iliacus, psoas major, pectineus, tensor fasciae latae, sartorius, gracilis, rectus femoris, the hamstrings (biceps femoris long head, semimembranosus, semitendinosus), and the short external rotators (piriformis, gemelli, obturators, quadratus femoris). Each muscle’s metabolic rate is the sum of a heat rate and a mechanical work rate, scaled by the mass $m _ { i }$ of muscle i:

$$
\begin{array} { l } { { \displaystyle \dot { E } _ { \mathsf { U c h i d a } } = \sum _ { i } \left( \dot { H } _ { i } + \dot { W } _ { i } \right) m _ { i } } , } \\ { { \displaystyle \dot { H } _ { i } = \operatorname* { m a x } \left( \dot { H } _ { \mathsf { A M } , i } + \dot { H } _ { \mathsf { S } , i } , \ 1 \right) , } } \end{array}\tag{5}
$$

where $\dot { H } _ { i }$ (in W/kg, floored at 1) combines the activation– maintenance heat $\dot { H } _ { \mathsf { A M } }$ and the shortening/lengthening heat $\dot { H } _ { 5 } ,$ and $\dot { W }$ is the mechanical work rate. With muscle activation $a \in [ 0 , 1 ]$ , active force–length factor $F _ { \mathrm { i s o } } ,$ normalized fiber velocity $\tilde { v } = v _ { c e } / \ell _ { m } ^ { 0 }$ , where $v _ { c e }$ is the muscle lengthening rate, $\ell _ { m } ^ { 0 }$ is the optimal muscle fiber length, and aerobic scale $\varphi = 1 . 5$ , the components are (muscle index omitted)

TABLE II  
PPO TRAINING HYPERPARAMETERS. THE SAME CONFIGURATION IS SHARED BY THE HUMAN POLICY $\pi _ { H }$ AND THE EXOSKELETON POLICY $\pi _ { E } ;$ THE AUXILIARY MUSCLE NETWORK IS TRAINED BY SUPERVISED REGRESSION.
<table><tr><td colspan="2">Parameter Value</td></tr><tr><td colspan="2">Policy optimization (PPO, per agent)</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>PPO clip range €</td><td>0.2</td></tr><tr><td>Value-function clip</td><td>100</td></tr><tr><td>Value loss coefficient</td><td>1.0</td></tr><tr><td>Discount  $\gamma$ </td><td>0.99</td></tr><tr><td>GAE λ</td><td>0.99</td></tr><tr><td>SGD iters per update</td><td>4</td></tr><tr><td>Train batch size</td><td>8192</td></tr><tr><td>Minibatch size</td><td>512</td></tr><tr><td>KL coefficient (initial)</td><td>0.01</td></tr><tr><td>Target KL</td><td>0.01</td></tr><tr><td>Entropy coefficient</td><td>0</td></tr><tr><td>Gradient clipping</td><td>none</td></tr><tr><td>Batch mode</td><td>truncate episodes</td></tr><tr><td>Horizon</td><td>10000</td></tr><tr><td>Parallel workers (1 env each)</td><td>16</td></tr><tr><td colspan="2">Auxiliary muscle network (supervised)</td></tr><tr><td>Learning rate</td><td> $\mathrm { i } \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Epochs per update</td><td>10</td></tr><tr><td>Minibatch size</td><td>512</td></tr></table>

$$
\begin{array} { c } { \dot { H } _ { \mathsf { A M } } = \varphi a ^ { 0 . 6 } \zeta ( F _ { \mathsf { i s o } } ) , \quad \dot { H } _ { \mathsf { S } } = \varphi a ^ { p } \sigma ( \tilde { v } ) , } \\ { \dot { W } = - \frac { F ^ { \mathrm { a c t } } v _ { c e } } { m } , } \end{array}\tag{6}
$$

where $p { = } 2$ for shortening $( \tilde { v } \le 0 )$ and $p { = } 1$ for lengthening, $\begin{array} { r c l } { F ^ { \mathsf { a c t } } } & { = } & { a f _ { 0 } F _ { \mathsf { i s o } } } \end{array}$ is the active fiber force $( f _ { 0 } { \mathrm { : } }$ maximum isometric force), and $\zeta , \sigma$ are the fiber-length and fiber-velocity heat scalings, which also depend on the recruited fiber-type composition.

The PPO hyperparameters, shared by both agents and both stages, are listed in Table II. The physics ran at 500 Hz and both policies at 50 Hz (10 physics substeps per control step), with human joint SPD (Stable Proportional Derivative [21]) gains $k _ { p } = 2 0 0$ and $k _ { v } = 4 0$ . Training ran on a workstation with two 16-core Intel Xeon Gold 5218 CPUs (2.3 GHz) and an NVIDIA Quadro RTX 5000 GPU, using 16 rollout workers; the co-adaptation runs took about 24 h for 62.6 M control steps for the Metabolic+ Power training and 14 h for 38.9 M control steps for the Metabolic training until convergence.

## D. Simulation Evaluation

Simulations used a full-body musculoskeletal model with a total body mass of 72.9 kg. Walking was simulated at approximately 1.14 m/s with a mean gait-cycle duration of ∼1.10 s. Three conditions were analyzed: unassisted walking without the device (NoExo, a human-only model); an active hip exoskeleton whose controller was trained with both the positive power term $r _ { \mathsf { p o w e r } }$ and the metabolic term r<sub>energy</sub> (Metabolic + Power, labeled PowerExo); and an active hip exoskeleton whose controller was trained with $r _ { \mathsf { p o w e r } }$ removed while retaining $r _ { \tt e n e r g y }$ (Metabolic, labeled EnergyExo). The EnergyExo controller is the policy deployed on hardware and is referred to as MARL-Exo in the human experiment.

The analysis covers 31 muscle actuators per leg, representing 15 anatomical muscles of the hip, knee, and ankle. They were grouped by joint and direction into six functional groups (hip extension and flexion, knee flexion and extension, ankle plantar flexion and dorsiflexion) and by joint into three joint groups (hip, knee, and ankle, each summing both directions). Group membership is listed in Supplementary Section II (Table S2). Biarticular muscles belong to groups at more than one joint.

For each condition we computed muscle-contributed joint torques at the lower limb joints; cycle-averaged muscle activation; and metabolic power per muscle (Watts), aggregated by functional group and summed over all analyzed muscles (W and W/kg), reported per leg as the left–right average. Group activation is the unweighted mean over the member actuators, whereas group metabolic power is their sum. Each condition was evaluated in one deterministic episode at the default body and gait parameters with about 18 complete gait cycles. All quantities are phase-normalized to the gait cycle (0–100%), reported bilaterally (left–right combined), and shown as the cycle mean with a ±1 SD band across cycles. The analyzed muscle set is sagittal-plane only, so absolute metabolic values underestimate whole-body cost. Because the same muscle set is used throughout, the conditions can be compared consistently on this partial-muscle total.

## III. HUMAN EXPERIMENT

## A. Hip Exoskeleton Hardware and Deployment

The trained MARL-Exo control policy was deployed to a custom bilateral hip exoskeleton, whose mechanical design, size adjustments, and full specifications are given in [19]; only the key elements needed are repeated here. The device weighs 5.94 kg and provides one actuated sagittal flexion/extension degree of freedom per side (MyActuator X8- 25, 25 N·m peak) together with a passive abduction/adduction hinge for fitting and lateral compliance. Bilateral hip angle was taken as the pelvis–femur difference measured by three wireless IMUs (LPMS-B2, LP-Research, Tokyo, Japan) mounted on the pelvis and both thighs and zeroed in a standing posture before each trial, and hip angular velocity as the corresponding gyroscope difference about the flexion axis. The trained MARL-Exo policy ran on an onboard Raspberry Pi 5 and reproduced the simulated observation and action interfaces exactly. At each 50 Hz control step the actor consumed the 18- dimensional observation defined in Section II-B and emitted a normalized torque in [−1, 1] per hip, which was scaled by the preset per-hip torque limit and transmitted to the actuators over the CAN bus. A second-order Butterworth low-pass filter with a 3 Hz cutoff, designed for the 50 Hz loop rate, was applied online to the measured hip angle and angular velocity, matching the observation filter used in training. Trials were run under a software torque limit and an emergency stop, with an operator supervising at all times.

## B. Human Validation Framework

Eight healthy adults (5 male, 3 female; age $2 7 . 0 \pm 7 . 8 \mathsf { y r } ,$ range 18–44; mass 70.4 ± 15.0 kg; height $1 . 7 3 \pm 0 . 0 6 \mathsf { m } )$ participated under IRB approval (Protocol No. 2305033091R003) with written informed consent. These participants and this session also supplied the data of the SMAT evaluation [19]. The standing-rest, no-exoskeleton, and passive recordings are therefore common to both studies and are reused here as the reference conditions (Table IV and Fig. 5); the MARL-Exo trials and every analysis involving them are new.

Before data collection, each participant completed a treadmill familiarization session, walking for 5 min with the device worn in passive mode (zero commanded torque) and for 10 min under each of the two active controllers (MARL-Exo and SMAT), after which the fit was left unchanged for the remainder of the experiment. Data collection followed a symmetric protocol with randomized active-controller order, designed to separate the metabolic penalty of wearing the device from the benefit of its active assistance: standing rest → no-exoskeleton walking → passive walking → two active trials (MARL-Exo and SMAT, order randomized; MARL-Exo first in 3 of 8 participants) → passive walking → noexoskeleton walking → standing rest (Fig. 2). The passive condition retained the full device mass with commanded torque held at zero and the actuators left back-drivable, isolating the mechanical burden of the hardware from the assistance it delivers.

Both active trials used identical hardware and differed only in the deployed policy. MARL-Exo is the controller referred to as Ctrl-B in [19], where it was run in the same session but not analyzed. MARL-Exo was commanded at a 12 N·m torque limit. The SMAT policy saturates its normalized output near 0.8 rather than unity [19] and was therefore commanded at 15 N·m, so that its realized peak $( 0 . 8 \times 1 5 \approx 1 2 \mathsf { N } { \cdot } \mathsf { m } )$ matched that of MARL-Exo; both controllers thus operated at a common ∼12 N·m level of assistance. Metabolic and kinematic data were recorded identically in both active trials.

Placing the no-exoskeleton, passive, and standing-rest conditions both before and after the active block helps assess fatigue and drift over the session, and the two repeats of each were averaged before analysis; averaging reduces, but does not eliminate, the influence of nonlinear drift or carryover between active trials. Every walking trial lasted approximately 6 min on a treadmill at 1.25 m/s, and expired gases were sampled breath-by-breath with a portable indirect calorimetry system (COSMED K5, COSMED, Italy). Between conditions, participants rested seated for at least 5 min.

Metabolic rate was computed from breath-by-breath oxygen uptake $\dot { V } _ { O _ { 2 } }$ and carbon-dioxide output $\dot { V } _ { C O _ { 2 } }$ (mL/min) with the Brockway relation [29]. The respiratory exchange ratio (RER, $\dot { V } _ { C O _ { 2 } } / \dot { V } _ { O _ { 2 } } )$ was checked in every walking trial, with values below 1 confirming predominantly aerobic metabolism. Each condition was represented by the mean of the last 2 min of its trial, the two repeats of the No-Exo, Passive, and rest conditions were averaged, and subtracting the standing-rest value gave the net rate $\dot { E } _ { \mathrm { { n e t } } }$ . The contrasts of interest are the device penalty (Passive vs. No-Exo) and the assistance benefit (MARL-Exo vs. Passive and No-Exo).

## C. Biomechanical and Statistical Analysis

Hip angle, angular velocity, and commanded torque were logged at 50 Hz, and analysis used a steady-state window of the last 2 min of each trial. The logged velocity was refiltered offline with a zero-phase fourth-order Butterworth low-pass at 6 Hz, which adds no further phase delay; it cannot, however, remove any delay already present in a causally filtered logged signal (Section III-A). Cycles were segmented at peak hip flexion, since no ground-reaction-force sensors were used.

Hip range of motion (ROM) was the difference between peak flexion and peak extension in each cycle. Cadence was computed from the mean cycle duration, and stride length was derived from it at the fixed belt speed, so the two are reciprocal by construction. Kinematic metrics were computed per cycle, then averaged within each subject and across the two hips.

Mechanical output was summarized with the metrics of [30], also used in [19]: RMS torque (τ<sub>RMS</sub>), mean peak torque $( \tau _ { \mathrm { M A X } } ^ { \mathrm { f l e x } } , \ \tau _ { \mathrm { M A X } } ^ { \mathrm { e x t } } )$ , and the mean positive and negative power over the n samples of a cycle,

$$
{ \mathsf { M P P } } = \frac { 1 } { n } \sum _ { \tau _ { i } { \dot { q } } _ { i } > 0 } \tau _ { i } { \dot { q } } _ { i } , \quad { \mathsf { M N P } } = \frac { 1 } { n } \sum _ { \tau _ { i } { \dot { q } } _ { i } < 0 } \tau _ { i } { \dot { q } } _ { i } ,\tag{7}
$$

with $\textit { P } = \tau \dot { q }$ the estimated assistance power, computed from the commanded torque and the IMU-derived hip angular velocity converted to rad/s. The positive-power ratio $R _ { \mathsf { p o s } } = \mathsf { M P P } / ( \mathsf { M P P } + \mathsf { | M N P | } )$ is the positive estimated work divided by the total absolute estimated work over the same interval; it describes the timing of assistance relative to hip motion and is not a measure of metabolic efficiency. These were averaged like the kinematic metrics.

Percentage changes in net metabolic rate were computed per subject and then averaged, and comparisons used twotailed paired t-tests (n = 8) after Shapiro–Wilk screening of the paired differences, Bonferroni-corrected within each outcome family at $\alpha = 0 . 0 5$ . The metabolic family has three planned comparisons (Passive vs. No-Exo, MARL-Exo vs. Passive, MARL-Exo vs. No-Exo), and the kinematic family has six (MARL-Exo vs. No-Exo for hip ROM, peak flexion, peak extension, cadence, and stride length, and Passive vs. No-Exo for hip ROM). Confidence intervals are unadjusted paired 95 % intervals. The participant was the inferential unit throughout; gait cycles were averaged within participants before testing.

## IV. RESULTS

## A. Simulation Evaluation Results

1) Hip joint kinematics and kinetics: In the simulation evaluation, both active controllers (PowerExo and EnergyExo) reduced peak hip extension relative to NoExo, with PowerExo producing the larger reduction, and slightly deepened peak hip flexion (most under PowerExo), consistent with the device delivering a flexion assist during swing initiation (Fig. 3). Overlaying the biological reference (Fig. 3) shows qualitative agreement in hip kinematics: the simulated hip trajectory and range follow the biological curve. At the hip, both controllers reduced the peak flexion torque, with EnergyExo producing the larger reduction (Fig. 3); hip-torque range was reduced by both. Plotted in mass-normalized units, the simulated joint moments agree with the biological reference in overall shape and timing at the hip joint: both show the hip extensor-toflexor transition through stance. Fig. 3 also shows the gaitcycle-averaged assistance torque and power. Compared to the PowerExo assistance, the EnergyExo assistance provided slightly higher τ<sub>RMS</sub> (13.62 vs. 13.12 N·m) but slightly lower MPP (19.98 vs. 20.64 W) with $\tau _ { \mathrm { m a x } } = 1 8 \mathsf { N } { \cdot } \mathsf { m }$ , both computed from the phase-averaged assistance curve in the figure.

![](images/decceb548c8b227bb57482cc670dd670efac65650994b1158ebfe9c67495c903.jpg)  
Fig. 2. Experimental protocol. Each participant completed standing rest, no-exoskeleton walking (No-Exo), and passive walking (Passive), then the two active controllers (MARL-Exo and SMAT) in randomized order, followed by the same reference conditions in reverse order. Each walking trial lasted approximately 6 min on a treadmill at 1.25 m/s, with metabolic rate measured by indirect calorimetry throughout.

![](images/007e5f3c4039cab17df34b453f528174c3d21ed7a1600845b1920ad2b628a6b0.jpg)  
Fig. 3. Hip joint angles, muscle-contributed joint moment (mass-normalized), assistance torque, and assistance power over the gait cycle for the simulated conditions (PowerExo: Metabolic + Power; EnergyExo: Metabolic) and an independent biological reference (Experiment, teal dashed, mean ± 1 SD) [31]. Muscle-contributed moments are not identical to inverse-dynamics net moments.

2) Muscle activation and metabolic energy by functional group: Total cycle-averaged metabolic power summed over the analyzed muscles fell under both controllers, with EnergyExo achieving the larger reduction (−13.8%, 1.830 → 1.578 W/kg per leg) and PowerExo an intermediate reduction $( - 8 . 8 \% ,  1 . 6 7 0 \mathsf { W } / \mathsf { k g }$ per leg) (Table III). Summed by joint, the savings were concentrated proximally: metabolic power of the hip muscles fell by 29.8% under EnergyExo and 16.0% under PowerExo, that of the knee muscles by 12.8% and 11.9%, and that of the ankle muscles by only 3.9% and 2.5%. The two controllers therefore differed mainly at the hip, where EnergyExo used 4.2 W less than PowerExo, whereas their knee and ankle totals differed by less than 1.3 W. Because biarticular muscles are counted at every joint they span, the joint rows overlap, and each controller’s hip and knee savings together exceed its total reduction. Within each joint, the savings were unevenly distributed between directions. At the hip, both directions contributed, with larger reductions in Hip Extension (EnergyExo −35.4%, PowerExo −19.2%) than in Hip Flexion (−24.0%, −12.6%). At the knee, the reduction came almost entirely from Knee Extension (−36.2%, −26.5%), while Knee Flexion changed little (−4.3%, −6.6%). At the ankle, the plantar-flexor group was preserved near baseline $( - 1 . 7 \% , \ + 0 . 6 \% )$ and the small ankle reduction came mainly from the dorsiflexor (−12.7%, −14.9%). Both controllers thus secured broad proximal savings while leaving ankle plantar-flexor energy essentially unchanged.

The activation curves (Fig. 4, top row) show that both controllers reduce hip-extensor and knee-extensor activation while hip-flexor activation rises and falls during the gait cycle. The energy curves (Fig. 4, bottom row) show that all three groups lose metabolic energy under both controllers, with EnergyExo generally below PowerExo. The hip flexors thus show an activation–energy divergence: group mean activation rises while summed group energy falls (−24.0%, −12.6%; Table III). Because activation is an unweighted group mean, whereas metabolic power is a sum weighted by muscle mass (Eq. 5; see also [10]), this divergence may partly reflect aggregation and redistribution among member muscles.

## B. Metabolic Cost of Walking from Experiment

Net metabolic rate across the three conditions is summarized in Fig. 5 and Table IV. The RER, averaged over the same final 2 min of each trial, remained below unity in all 40 analyzed walking trials (five per participant: two No-Exo, two Passive, and one MARL-Exo; overall $0 . 8 3 8 \pm 0 . 0 3 2$ , range 0.785–0.916; No-Exo: 0.831 ± 0.035, Passive: 0.839 ± 0.030, MARL-Exo: $0 . 8 4 7 \pm 0 . 0 2 9 )$ , so the Brockway relation was applied throughout within its aerobic range. The two repeats of the no-exoskeleton and passive conditions differed by $6 . 9 \pm 5 . 2 \%$ and $3 . 5 \pm 2 . 7 \%$ of gross metabolic rate respectively, with no consistent direction across subjects, so we found no evidence of a systematic trend over the session; averaging the repeats reduces, but does not remove, this variability.

![](images/e816ded6205461afee32725266be42ff02c8ff893c2dd4f7bb3973880ccf19e1.jpg)

TABLE III  
CYCLE-AVERAGED METABOLIC POWER BY FUNCTIONAL GROUP (SUM ACROSS MEMBER MUSCLES) AND TOTAL ACROSS ALL INVOLVED MUSCLE ACTUATORS, ALL VALUES PER LEG (LEFT–RIGHT AVERAGED; A PARTIAL-MUSCLE SUM, NOT WHOLE-BODY METABOLISM). POWEREXO: METABOLIC + POWER REWARD; ENERGYEXO: METABOLIC REWARD (DEPLOYED). ∆ IN WATTS; PERCENTAGES SIGNED VS. NOEXO. GROUPS OVERLAP THROUGH BIARTICULAR MUSCLES, SO THE SIX GROUP TOTALS EXCEED THE TOTAL; THE TOTAL ROW IS THE TRUE NON-OVERLAPPING SUM. THE HIP, KNEE, AND ANKLE ROWS SUM BOTH DIRECTIONS AT EACH JOINT (NO MUSCLE BELONGS TO BOTH DIRECTIONS OF THE SAME JOINT); BIARTICULAR MUSCLES STILL APPEAR IN MORE THAN ONE JOINT ROW. GROUP MEMBERSHIP IS GIVEN IN SUPPLEMENTARY TABLE S2. W/KG IS THE PER-LEG POWER DIVIDED BY BODY MASS (72.9 KG).
<table><tr><td>Muscle group</td><td>NoExo W</td><td>PowerExo W</td><td>EnergyExo W</td><td>NoExo W/kg</td><td>PowerExo—NoExo ∆(%)</td><td>EnergyExo—NoExo ∆ (%)</td></tr><tr><td>Hip Extension</td><td>15.47</td><td>12.50</td><td>9.99</td><td>0.212</td><td>-2.97 (-19.2%)</td><td>-5.48 (-35.4%)</td></tr><tr><td>Hip Flexion</td><td>14.84</td><td>12.97</td><td>11.27</td><td>0.204</td><td>-1.87 (-12.6%)</td><td>-3.57 (-24.0%)</td></tr><tr><td>Knee Flexion</td><td>56.07</td><td>52.39</td><td>53.64</td><td>0.769</td><td>-3.68 (-6.6%)</td><td>-2.43 (-4.3%)</td></tr><tr><td>Knee Extension</td><td>20.32</td><td>14.94</td><td>12.97</td><td>0.279</td><td>-5.38 (-26.5%)</td><td>-7.35 (-36.2%)</td></tr><tr><td>Ankle PlantarFlex</td><td>69.60</td><td>70.02</td><td>68.40</td><td>0.955</td><td>+0.42 (+0.6%)</td><td>-1.20 (−1.7%)</td></tr><tr><td>Ankle DorsiFlex</td><td>17.31</td><td>14.73</td><td>15.12</td><td>0.237</td><td>-2.58 (-14.9%)</td><td>-2.19 (−12.7%)</td></tr><tr><td>Hip (Ext. + Flex.)</td><td>30.30</td><td>25.47</td><td>21.26</td><td>0.416</td><td>-4.84 (-16.0%)</td><td>-9.04 (-29.8%)</td></tr><tr><td>Knee (Flex. + Ext.)</td><td>76.40</td><td>67.34</td><td>66.61</td><td>1.048</td><td>-9.06 (-11.9%)</td><td>-9.79 (-12.8%)</td></tr><tr><td>Ankle (PF+ DF)</td><td>86.91</td><td>84.75</td><td>83.52</td><td>1.192</td><td>-2.16 (-2.5%)</td><td>-3.39 (-3.9%)</td></tr><tr><td>TOTAL</td><td>133.43</td><td>121.75</td><td>115.01</td><td>1.830</td><td>-11.68 (—8.8%)</td><td>-18.42 (-13.8%)</td></tr></table>

Fig. 4. Muscle activation (top row) and metabolic energy rate (bottom row) over the gait cycle for the three functional groups where the controllers produce their largest reductions: Hip Extension, Hip Flexion, and Knee Extension (group mean activation; group summed energy; mean ± 1 SD). Hip-flexor activation rises during stance phase but falls during swing phase while hip-flexor energy mostly falls under both controllers.

Carrying the unpowered device raised net metabolic rate by $1 3 . 1 \pm 4 . 9 \%$ above unassisted walking $( t _ { 7 } = 7 . 8 0 _ { \div }$ , corrected $p < 0 . 0 0 1 )$ , a penalty consistent with the added mass and the constraint the structure imposes, although the present data do not separate these causes. This penalized condition, rather than unassisted walking, isolates the contribution of the assistance itself.

Against that baseline, active MARL-Exo assistance lowered net metabolic rate by $1 8 . 3 \pm 6 . 6 \%$ , from $3 . 5 1 4 \pm 0 . 4 1 4$ to $2 . 8 8 8 \pm 0 . 5 3 8 \mathsf { W } / \mathsf { k g } \ ( t _ { 7 } \ = \ - 9 . 2 2$ , corrected $p ~ < ~ 0 . 0 0 1 )$ . The effect was present in every participant, with individual reductions spanning 7.2 % to 27.7 %, so the assistance more than recovered the cost of wearing the hardware.

Relative to walking without the device, six of the eight participants walked at a lower net cost with the assisted device than without any device, with individual reductions of up to 24.2 % and above 20 % in two participants; in these six, the assistance recovered between 1.1 and 7.5 times the metabolic penalty of wearing the hardware. Across the cohort the mean reduction was $7 . 4 \pm 1 0 . 0 \%$ (paired difference −0.221 W/kg, unadjusted 95 % CI [−0.474, +0.033], $t _ { 7 } = - 2 . 0 5 ,$ , corrected $p = 0 . 2 4 )$ , with the interval reflecting the between-subject spread in an eight-participant cohort.

## C. Hip Biomechanics and Assistance Power Delivery

Fig. 6 compares hip kinematics and delivered assistance power across the No-Exo, Passive, and MARL-Exo conditions. Hip kinematics were derived from the three body-mounted IMUs, whose standing-posture zero offsets each subject’s absolute joint angles. Range of motion is taken as the primary kinematic outcome, while the individual flexion and extension peaks are reported as descriptive quantities. The paired differences satisfied the Shapiro–Wilk normality check for every variable except peak flexion $( W = 0 . 8 0 9 , p = 0 . 0 3 6 )$ for which a Wilcoxon signed-rank test is reported alongside the parametric result.

TABLE IV  
PER-SUBJECT NET METABOLIC RATE AND REDUCTIONS UNDER MARL-EXO AND SMAT (n = 8).
<table><tr><td></td><td colspan="4">Net metabolic rate (W/kg)</td><td colspan="2">MARL-Exo reduction (%)</td><td colspan="2">SMAT reduction (%)</td></tr><tr><td>Subject</td><td>No-Exo</td><td>Passive</td><td>MARL-Exo</td><td>SMAT</td><td>vs Passive</td><td>vs No-Exo</td><td>vs Passive</td><td>vs No-Exo</td></tr><tr><td>S1</td><td>3.008</td><td>3.326</td><td>2.405</td><td>2.738</td><td>27.7</td><td>20.1</td><td>17.7</td><td>9.0</td></tr><tr><td>S2</td><td>3.575</td><td>4.081</td><td>3.376</td><td>3.221</td><td>17.3</td><td>5.6</td><td>21.1</td><td>9.9</td></tr><tr><td>S3</td><td>2.941</td><td>3.051</td><td>2.228</td><td>2.368</td><td>27.0</td><td>24.2</td><td>22.4</td><td>19.5</td></tr><tr><td>S4</td><td>2.794</td><td>3.243</td><td>2.747</td><td>2.711</td><td>15.3</td><td>1.7</td><td>16.4</td><td>3.0</td></tr><tr><td>S5</td><td>3.734</td><td>4.222</td><td>3.916</td><td>3.571</td><td>7.2</td><td>-4.9</td><td>15.4</td><td>4.4</td></tr><tr><td>S6</td><td>2.876</td><td>3.301</td><td>2.678</td><td>2.458</td><td>18.9</td><td>6.9</td><td>25.5</td><td>14.5</td></tr><tr><td>S7</td><td>2.852</td><td>3.444</td><td>2.884</td><td>2.802</td><td>16.3</td><td>-1.1</td><td>18.6</td><td>1.8</td></tr><tr><td>S8</td><td>3.088</td><td>3.445</td><td>2.870</td><td>2.744</td><td>16.7</td><td>7.1</td><td>20.3</td><td>11.2</td></tr><tr><td>Mean ±SD</td><td> $3 . 1 1 \pm 0 . 3 5$ </td><td>3.51±0.41</td><td>2.89±0.54</td><td>2.83±0.39</td><td>18.3±6.6</td><td>7.4±10.0</td><td>19.7±3.3</td><td>9.1 ±6.0</td></tr></table>

Reduction = $( \dot { E } _ { \mathsf { r e f } } - \dot { E } _ { \mathsf { a c t i v e } } ) / \dot { E } _ { \mathsf { r e f } }$ , positive when the active condition costs less, computed per subject and then averaged. The SMAT data are from [19].

![](images/c07a6691f3b030edfe794cd2e21eb8cfa4716beb5c4ee10e2c36639ca87c71a2.jpg)  
Fig. 5. Net metabolic rate $( \mathsf { W } \mathsf { k } \mathsf { g } ^ { - 1 } )$ across conditions $( n = 8 ) .$ . Half-violin distributions, box plots, and individual subjects (connected lines) for No-Exo, Passive, and MARL-Exo. The No-Exo and Passive data are shared with [19]. Brackets show the group-mean percentage change between conditions (+13.1 %, −18.3 %, −7.4 %). Paired t-test with Bonferroni correction: $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1 ; \mathsf { n . s . } p \ge 0 . 0 5$ . All eight subjects reduced net cost under assistance relative to the passive condition.

Active assistance markedly enlarged hip range of motion, from $4 4 . 0 \pm 4 . 3 ^ { \circ }$ without the device to $6 0 . 2 \pm 6 . 6 ^ { \circ }$ under MARL-Exo, an increase of 36.8 % $( t _ { 7 } ~ = ~ 5 . 9 7 ,$ corrected $p = 0 . 0 0 3 )$ , with the passive condition falling between the two $( 4 8 . 7 \pm 4 . 0 ^ { \circ } , + 1 0 . 6 \% , t _ { 7 } = 3 . 9 3$ , corrected $p = 0 . 0 3 4 )$ . The enlargement was carried almost entirely by the flexion phase: peak flexion rose by 20.1<sup>◦</sup>, from $3 3 . 6 \pm 8 . 2 ^ { \circ }$ to $5 3 . 6 \pm 6 . 9 ^ { \circ }$ a 59.8 % increase $( t _ { 7 } = 5 . 3 8$ , corrected $p = 0 . 0 0 6 )$ , whereas peak extension was statistically unchanged $( - 1 0 . 5 \pm 6 . 9 ^ { \circ }$ to $- 6 . 6 \pm 8 . 4 ^ { \circ } , \ t _ { 7 } \ = \ 1 . 1 9$ , corrected $p \ > \ 0 . 9 9 )$ . Both range of motion and peak flexion increased in all eight participants without exception. For peak flexion, whose paired differences were not normally distributed, the nonparametric test agreed in direction and remained significant after correction $( W = 0 ,$ exact $p = 0 . 0 0 8$ , corrected $p = 0 . 0 4 7 )$ ; with $n = 8$ the exact test cannot resolve below $p = 0 . 0 0 8$ , so this value reflects the sample size rather than the magnitude of the effect, which the unanimous direction and the $2 0 . 1 ^ { \circ }$ shift characterize more directly.

The spatiotemporal structure of the gait shifted far less than its amplitude. Cadence rose from $1 1 0 . 3 { \pm } 8 . 2 $ to $1 1 5 . 7 \pm$ 6.3 steps/min under assistance, a 4.8 % increase that did not survive correction $( t _ { 7 } = 2 . 9 2 $ , corrected $p = 0 . 1 3 ) $ ; at the fixed belt speed this is equivalent to a shortening of stride length from $1 . 3 6 6 { \pm } 0 . 0 9 8$ to $1 . 3 0 0 { \pm } 0 . 0 7 2 \mathsf { m }$ , the two being reciprocal by construction. Assistance thus amplified hip excursion, while the changes in cadence and stride length were smaller and not significant after correction.

The exoskeleton delivered torque in both directions of the stride, with a flexion-assist peak in late swing and an extension-assist phase through mid-stance. Table V reports the per-subject mechanical output. Mean per-cycle peak torque was 11.79±0.39 N·m in flexion and $- 1 1 . 8 1 \pm 0 . 4 2 \harpoonright$ N·m in extension, magnitudes that did not differ significantly $( t _ { 7 } = 1 . 0 7 , \ p = 0 . 3 2 )$ , indicating that the policy saturated symmetrically against the 12 N·m commanded limit in both directions; RMS torque was $9 . 4 5 \pm 0 . 4 5 \mathsf { N } { \cdot } \mathsf { m }$ . Mean positive estimated power was $1 9 . 1 9 \pm 2 . 7 4 \mathsf { W }$ while mean negative estimated power remained near zero $( - 0 . 3 0 { \pm } 0 . 1 4 \mathsf { W } )$ , giving a positive-power ratio $R _ { \mathsf { p o s } }$ of $0 . 9 8 5 \pm 0 . 0 0 6$ . Because the deployed policy was trained without the explicit power term, this predominantly positive power emerged from the metabolic and regularization terms rather than being directly rewarded. The spread across subjects was small in all four metrics, indicating that the single deployed policy produced consistent mechanical assistance without subject-specific tuning.

## V. DISCUSSION

A central design choice of this work is to include a physiologically detailed metabolic model in the exoskeleton reward. In simulation, with the metabolic term retained in both controllers, removing explicit positive-power shaping lowered modeled metabolic power by a further 5 percentage points (−13.8 % vs. −8.8 %) and changed where the savings fell. Further training of the Metabolic + Power controller over a comparable number of steps did not improve it, which points to the reward change, not the additional training, as the source of the gain. Power shaping thus steers the learned strategy away from energetically cheaper solutions that the metabolic term alone finds. The divergence of hip-flexor activation and energy noted above suggests a mechanism: an energeticsbased reward can favor strategies that an activation-based reward would penalize. Because energy models disagree with one another [32], the simulated ranking of controllers and the savings by muscle group are specific to the Uchida model used here and may differ under alternatives such as that of Bhargava et al. [33]. The reward otherwise follows trajectory-optimization practice, where energy is one term of a composite cost because energy alone can predict unrealistic movement and its rate expressions are non-smooth [34], [35].

![](images/d6dabf7adfb82002201270136afeb54fdc3b489bfd32450054ff3244220295e7.jpg)

Fig. 6. Treadmill walking at 1.25 m/s under No-Exo (black), Passive (blue), and MARL-Exo (green) (n = 8; per-subject means of both hips). Left: gait-cycle hip angle (flexion positive) with peak flexion and extension; labels give the change in ROM. Middle: cadence and stride length. Right: hip angular velocity together with the exoskeleton torque and mechanical power delivered by MARL-Exo. Bands and error bars show mean ±1 SD across subjects. The No-Exo and Passive data are shared with [19].  
TABLE V  
PER-SUBJECT EXOSKELETON MECHANICAL OUTPUT, MARL-EXO CONDITION (n = 8, BOTH HIPS AVERAGED).
<table><tr><td>Subject</td><td>TRMS (N·m)</td><td> $\tau _ { \mathrm { M A X } } ^ { \mathrm { f l e x } }$  (N·m)</td><td> $\tau _ { \mathsf { M A X } } ^ { \mathsf { e x t } } \left( \mathsf { N } \cdot \mathsf { m } \right)$ </td><td>MPP (W)</td><td>MNP (W)</td><td> $R _ { \mathsf { p o s } }$ </td></tr><tr><td>S1</td><td>9.62</td><td>11.94</td><td>-11.97</td><td>24.04</td><td>-0.54</td><td>0.978</td></tr><tr><td>S2</td><td>9.76</td><td>11.97</td><td>-11.98</td><td>21.02</td><td>-0.24</td><td>0.989</td></tr><tr><td>S3</td><td>9.36</td><td>11.92</td><td>-11.94</td><td>16.79</td><td>-0.20</td><td>0.988</td></tr><tr><td>S4</td><td>10.10</td><td>11.99</td><td>-11.99</td><td>21.53</td><td>-0.33</td><td>0.985</td></tr><tr><td>S5</td><td>9.42</td><td>11.97</td><td>-11.97</td><td>18.40</td><td>-0.39</td><td>0.979</td></tr><tr><td>S6</td><td>9.68</td><td>11.80</td><td>-11.96</td><td>18.26</td><td>-0.05</td><td>0.997</td></tr><tr><td>S7</td><td>8.75</td><td>10.84</td><td>-10.78</td><td>16.16</td><td>-0.29</td><td>0.982</td></tr><tr><td>S8</td><td>8.89</td><td>11.86</td><td>-11.88</td><td>17.29</td><td>-0.33</td><td>0.981</td></tr></table>

τ : RMS torque; $\tau _ { \mathrm { M A X } } ^ { \mathrm { f l e x } } , \tau _ { \mathrm { M A X } } ^ { \mathrm { e x t } } \mathrm { . }$ mean per-cycle peak torque in the flexionand extension-assist directions; MPP/MNP: mean positive/negative power (Eq. 7); $R _ { \mathsf { p o s } } = { \mathsf { M P P } } / ( { \mathsf { M P P } } + | { \mathsf { M N P } } | )$ : positive-power ratio. All metrics were computed per gait cycle and averaged across cycles within each subject, with the two hips averaged.

On hardware, the central result is that assistance beat the unpowered device in every one of the eight participants $( - 1 8 . 3 \pm 6 . 6 \% , p < 0 . 0 0 1 )$ , with reductions from 7.2 % to 27.7 %, even though the policy was neither retrained nor tuned for any of them. Because the passive condition carries the same hardware, this contrast isolates what the controller itself contributes, and we treat it as the primary outcome. On a common scale, active assistance removed 0.63 W/kg relative to the passive device, about 1.5 times the 0.41 W/kg that wearing the unpowered device added. The net result relative to walking without any device therefore depends on the hardware as much as on the controller. Six of eight participants walked at a lower net cost than without the device, two by more than 20 %, and the cohort mean was $- 7 . 4 \pm 1 0 . 0 \%$

The savings relative to no-device walking reported for other simulation-trained hip controllers, 24.3 % [7] and 15.2 % [9], combine the benefit of each controller with the penalty of its particular device. Isolated from the hardware by the passive condition, the benefit of MARL-Exo (18.3 %) exceeds the nodevice saving reported in [9], and it was achieved under less favorable hardware conditions than either study. Our device weighs 5.94 kg, about twice the 3.2 kg [7] and 2.9 kg [9] of those devices, and it was operated at a 12 N·m limit, lower than the peak torque reported in [7]. The smaller no-device saving is therefore consistent with the 13.1 % penalty of our hardware rather than with an inefficient controller. Accordingly, a lighter device and a higher torque limit would be expected to shift the no-device contrast in favor of assistance.

The delivered assistance is consistent with this outcome. Torque was applied in both directions of the stride, with matching peaks in flexion and extension, and almost all of the estimated assistance power acted with the wearer’s hip motion (Table V), the mechanical signature of well-timed assistance. Hip range of motion grew markedly, almost entirely through flexion, while cadence and stride length did not change significantly after correction. This pattern is consistent with an assistance profile that helps swing initiation, and with the simulated reduction of hip-flexor and hip-extensor energy. The larger excursion at the same belt speed is an altered gait response, so part of the metabolic benefit may come from the changed gait as well as from reduced muscle workload.

Because the two deployed controllers (SMAT and MARL-Exo) were tested in the same participants and session at a matched ∼12 N·m peak, they can be compared directly. Relative to the passive device, SMAT lowered net metabolic rate by 19.7 % and MARL-Exo by 18.3 %, and per subject MARL-Exo was on average 0.061 W/kg higher in net cost (unadjusted 95 % CI $[ - 0 . 1 1 6 , + 0 . 2 3 9 ] \mathsf { W } / \mathsf { k g } ;$ $t _ { 7 } ~ = ~ 0 . 8 2$ uncorrected $p \ = \ 0 . 4 4 )$ . Both controllers thus reduced cost relative to the passive device by similar amounts, and this exploratory comparison detected no difference between them. The mechanical pattern was also nearly identical: flexion assistance in late swing and extension assistance through stance, the same positive-power ratio (0.985 vs. 0.984), similar mean positive power (19.2 vs. 18.1 W), and similar gains in hip range of motion $( + 3 6 . 8 \% { \sf v } { \sf s } . + 4 0 . 8 \% )$ . MARL-Exo had a slightly higher RMS torque (9.45 vs. 8.84 N·m) with a slightly lower mean peak (11.79 vs. 12.39 N·m), which suggests a flatter profile held near the limit for longer.

The similar metabolic benefit may partly reflect the shared hardware limit. Both controllers drove the same device, saturated against the same ∼12 N·m cap, and delivered similar positive mechanical power, and in a rigid hip exoskeleton the metabolic saving is expected to scale with the positive work returned to the wearer. Under a common torque ceiling, two policies that both learn to time assistance with hip motion may therefore converge on similar benefit, and the flatter, higher-RMS profile of MARL-Exo did not lower cost further, as a broad optimum or a ceiling would predict. Testing both controllers at higher torque limits, on lighter hardware, and at more than one speed would show where their differences in training become visible.

Several sim-to-real details deserve mention. The observation was built from IMU-derived hip angle and velocity, so it did not depend on motor encoders, and it was low-pass filtered online at 3 Hz, as in training. The policy also ran at 50 Hz on a Raspberry Pi 5, and the observation latency modeled in simulation (40 ms) is what allows such a modest processor to be used. Soft-tissue compliance and strap looseness still make IMU-derived angles an approximation of true hip motion, and the phase alignment of assistance might be improved by more accurate thigh and pelvis sensing [36]. Finally, hardware was commanded at a 12 N·m limit, lower than the ±18 N·m simulated peak, to keep the device comfortable and to avoid wobble and excessive relative movement between the exoskeleton and the wearer, which the rigidly constrained simulation does not capture. The deployed assistance is therefore weaker than the one that produced the simulated savings, and higher torque with a better-fitting interface may yield larger benefit.

The scope of the study defines the next steps. The human study evaluated eight healthy adults walking on a treadmill at a single speed after 10 min of familiarization with each controller, with reference conditions shared with our SMAT study. Larger cohorts would resolve the contrast with unassisted walking (corrected $p \ : = \ : 0 . 2 4$ here), longer or multiday exposure would show whether the benefit grows as wearers adapt to the assistance, and testing across speeds, slopes, and gait-impaired populations would exercise the body and gait randomization already present in training. Adding EMG and full lower-limb motion capture to the hip kinematics recorded here would allow the simulated redistribution of muscle effort, and the gait changes that accompany assistance, to be checked in humans. Finally, a lighter device with a betterfitting interface would reduce the measured wearing penalty and allow the higher torque used in training to be applied without relative motion at the cuffs.

## VI. CONCLUSION

We presented MARL-Exo, a two-stage multi-agent reinforcement learning framework in which a full-body musculoskeletal human policy and a hip-exoskeleton policy are coadapted in simulation, with a physiologically based muscleenergetics term in the exoskeleton reward. Co-adaptation remained stable with a 304-actuator human model and randomized anatomy, gait, and speed, and the choice of reward shaped the learned strategy: with the metabolic term retained, omitting explicit positive-power shaping lowered modeled muscle metabolic power by 13.8 % rather than 8.8 %, with the additional savings concentrated at the hip and knee. The learned policy was deployed on a hip exoskeleton in eight healthy adults, without per-user tuning, and it reduced net metabolic rate by $1 8 . 3 \pm 6 . 6 \%$ relative to the passive device, enlarged hip range of motion by 36.8 %, and delivered almost entirely positive estimated assistance power. Six of eight participants walked below their no-device cost, by up to 24.2 %, with a cohort-mean reduction of $7 . 4 \pm 1 0 . 0 \%$ . The benefit was close to that of our earlier SMAT controller at the same torque limit, despite a different human model, curriculum, and reward. These results show that physiologically informed coadaptive training can produce an exoskeleton controller that transfers from simulation to human users without per-user tuning, and that candidate reward designs can be compared in simulation before human testing. Larger and more diverse cohorts, lighter hardware, and evaluation across speeds, slopes, and gait-impaired populations are the next steps toward population-level assistance.

## ACKNOWLEDGMENT

This work was partially supported by the National Institute on Disability, Independent Living, and Rehabilitation Research (NIDILRR) funded Rehabilitation Engineering Research Center Grant 90REGE0025-01-00 and NSF Award #2524089.

## REFERENCES

[1] A. Rodr´ıguez-Fernandez´ et al., “Systematic review on wearable lowerlimb exoskeletons for gait training in neuromuscular impairments,” Journal of NeuroEngineering and Rehabilitation, vol. 18, no. 1, p. 22, 2021.

[2] R. Baud et al., “Review of control strategies for lower-limb exoskeletons to assist gait,” Journal of NeuroEngineering and Rehabilitation, vol. 18, no. 1, p. 119, 2021.

[3] K. L. Poggensee and S. H. Collins, “How adaptation, training, and customization contribute to benefits from exoskeleton assistance,” Science Robotics, vol. 6, no. 58, p. eabf1078, 2021.

[4] J. Zhang et al., “Human-in-the-loop optimization of exoskeleton assistance during walking,” Science, vol. 356, no. 6344, pp. 1280–1284, 2017.

[5] Y. Ding et al., “Human-in-the-loop optimization of hip assistance with a soft exosuit during walking,” Science Robotics, vol. 3, no. 15, p. eaar5438, 2018.

[6] S. Luo et al., “Robust walking control of a lower limb rehabilitation exoskeleton coupled with a musculoskeletal model via deep reinforcement learning,” Journal of NeuroEngineering and Rehabilitation, vol. 20, no. 1, p. 34, 2023.

[7] S. Luo et al., “Experiment-free exoskeleton assistance via learning in simulation,” Nature, vol. 630, no. 8016, pp. 353–359, 2024.

[8] I. Park et al., “Learning hip exoskeleton control policy via predictive neuromusculoskeletal simulation,” arXiv preprint arXiv:2603.04166, 2026.

[9] H. Barati et al., “End-to-end policy learning for hip exoskeleton via reinforcement learning and reflex-based musculoskeletal simulation,” IEEE Robotics and Automation Letters, 2026.

[10] G. Leem et al., “Exo-Plore: Exploring exoskeleton control space through human-aligned simulation,” International Conference on Learning Representations (ICLR) Proceedings., 2026.

[11] Y. Yuan et al., “Gait asymmetry from unilateral weakness and improvement with ankle assistance: A reinforcement learning based simulation study,” Proceedings of 2026 11th IEEE RAS/EMBS International Conference on Biomedical Robotics and Biomechatronics (BioRob), 2026.

[12] N. Ratnakumar et al., “Reinforcement-learning-based assistance reduces squat effort with a modular hip–knee exoskeleton,” Proceedings of 2026 11th IEEE RAS/EMBS International Conference on Biomedical Robotics and Biomechatronics (BioRob), 2026.

[13] N. Ratnakumar et al., “Predicting sit-to-stand motions with a deep reinforcement learning based controller under idealized exoskeleton assistance,” Multibody System Dynamics, vol. 66, no. 4, pp. 835–852, 2026.

[14] M. Simos et al., “Reinforcement learning-based motion imitation for physiologically plausible musculoskeletal motor control,” arXiv preprint arXiv:2503.14637, 2025.

[15] Y. Bengio et al., “Curriculum learning,” in Proceedings of the 26th Annual International Conference on Machine Learning, 2009, pp. 41–48.

[16] A. S. Chiappa et al., “Acquiring musculoskeletal skills with curriculumbased reinforcement learning,” Neuron, vol. 112, no. 23, pp. 3969–3983, 2024.

[17] V. Caggiano et al., “MyoSuite–A contact-rich simulation suite for musculoskeletal motor control,” arXiv preprint arXiv:2205.13600, 2022.

[18] Y. Yuan et al., “SMAT: Staged multi-agent training for co-adaptive exoskeleton control,” Proceedings of 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems, arXiv preprint arXiv:2603.07618, 2026.

[19] Y. Yuan et al., “Staged multi-agent training for human–exoskeleton coadaptation: Simulation-based learning and real-world metabolic and biomechanical validation,” Manuscript under review, arXiv preprint arXiv:2608.00715, 2026.

[20] J. Lee et al., “DART: Dynamic animation and robotics toolkit,” The Journal of Open Source Software, vol. 3, no. 22, p. 500, 2018.

[21] J. Park et al., “Generative GaitNet,” in ACM SIGGRAPH 2022 Conference Proceedings, 2022, pp. 1–9.

[22] J. Park et al., “Bidirectional GaitNet: A bidirectional prediction model of human gait and anatomical conditions,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023, pp. 1–9.

[23] X. Zhou and X. Chen, “Design and evaluation of torque compensation controllers for a lower extremity exoskeleton,” Journal of Biomechanical Engineering, vol. 143, no. 1, p. 011007, 2021.

[24] C. K. Tan et al., “MyoAssist 0.1: MyoSuite for dexterity and agility in bionic humans,” in 2025 International Conference on Rehabilitation Robotics (ICORR). IEEE, 2025, pp. 437–442.

[25] R. Lowe et al., “Multi-agent actor-critic for mixed cooperative-competitive environments,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[26] J. Schulman et al., “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[27] B. R. Umberger et al., “A model of human muscle energy expenditure,” Computer Methods in Biomechanics and Biomedical Engineering, vol. 6, no. 2, pp. 99–111, 2003.

[28] T. K. Uchida et al., “Stretching your energetic budget: how tendon compliance affects the metabolic cost of running,” PLOS ONE, vol. 11, no. 3, p. e0150378, 2016.

[29] J. Brockway, “Derivation of formulae used to calculate energy expenditure in man.” Human Nutrition. Clinical Nutrition, vol. 41, no. 6, pp. 463–471, 1987.

[30] B. Lim et al., “Parametric delayed output feedback control for versatile human-exoskeleton interactions during walking and running,” IEEE Robotics and Automation Letters, vol. 8, no. 8, pp. 4497–4504, 2023.

[31] J. Camargo et al., “A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions,” Journal of Biomechanics, p. 110320, 2021.

[32] R. H. Miller, “A comparison of muscle energy models for simulating human walking in three dimensions,” Journal of Biomechanics, vol. 47, no. 6, pp. 1373–1381, 2014.

[33] L. J. Bhargava et al., “A phenomenological model for estimating metabolic energy consumption in muscle contraction,” Journal of Biomechanics, vol. 37, no. 1, pp. 81–88, 2004.

[34] M. Ackermann and A. J. van den Bogert, “Optimality principles for model-based prediction of human gait,” Journal of Biomechanics, vol. 43, no. 6, pp. 1055–1060, 2010.

[35] A. D. Koelewijn et al., “A metabolic energy expenditure model with a continuous first derivative and its application to predictive simulations of gait,” Computer Methods in Biomechanics and Biomedical Engineering, vol. 21, no. 8, pp. 521–531, 2018.

[36] T. Seel et al., “IMU-based joint angle measurement for gait analysis,” Sensors, vol. 14, no. 4, pp. 6891–6909, 2014.

# Supplementary Material

# Multi-Agent Reinforcement Learning of Co-Adaptive Hip Exoskeleton Control (MARL-Exo) for Metabolic Energy Reduction

Yifei Yuan, Jakob Wolf, Ghaith Androwis, and Xianlian Zhou

## I. AUXILIARY TERMS OF THE HUMAN REWARD

The human reward contains several auxiliary terms $\sum _ { k } r _ { \mathsf { a u x } } ^ { k }$ that regularize the learned gait. Three terms are used, with identical settings in the human baseline stage and in both co-adaptation runs (Metabolic and Metabolic + Power):

$$
\sum _ { k } r _ { \mathsf { a u x } } ^ { k } = r _ { \mathsf { I i m } } + r _ { \tau } + r _ { \mathsf { c l r } } .\tag{S1}
$$

Their constants are listed in Table S1.

## A. Knee Joint-Limit Penalty

Knee hyperextension is penalized quadratically once the knee angle crosses a soft limit placed a margin δ inside the joint’s lower (extension) limit:

$$
r _ { \tt I i m } = - k _ { \tt I i m } \sum _ { j \in \{ \tt L , \tt R \} } \operatorname* { m a x } \bigl ( 0 , \mathrm { ~ } ( \theta _ { j } ^ { \mathrm { m i n } } + \delta ) - \theta _ { j } \bigr ) ^ { 2 } ,\tag{S2}
$$

where $\theta _ { j }$ is the flexion angle of knee $j$ (flexion positive). With the knee lower limit $\theta ^ { \mathrm { m i n } } = - 0 . 1$ rad and $\delta = 0 . 1 \mathsf { r a d }$ the penalty is active for any hyperextension $( \theta _ { j } < 0 )$ . The upper (flexion) limit is not penalized.

## B. Joint-Torque Smoothness

Smooth joint actuation is encouraged with a bounded term that decreases as the desired joint torque changes more abruptly between control steps:

$$
r _ { \tau } = e ^ { - c \| \nabla \pmb { \tau } _ { t } \| ^ { 2 } } + 0 . 5 e ^ { - c \| \nabla ^ { 2 } \pmb { \tau } _ { t } \| ^ { 2 } } + 0 . 2 e ^ { - c \| \nabla ^ { 3 } \pmb { \tau } _ { t } \| ^ { 2 } } ,\tag{S3}
$$

where $\tau _ { t }$ is the vector of SPD desired joint torques (N·m) at control step $t , c = 1 0 ^ { - 3 }$ , and $\nabla ^ { n }$ denotes the n-th backward difference over consecutive control steps:

$$
\begin{array} { r l } & { \nabla \pmb { \tau } _ { t } = \pmb { \tau } _ { t } - \pmb { \tau } _ { t - 1 } , } \\ & { \nabla ^ { 2 } \pmb { \tau } _ { t } = \pmb { \tau } _ { t } - 2 \pmb { \tau } _ { t - 1 } + \pmb { \tau } _ { t - 2 } , } \\ & { \nabla ^ { 3 } \pmb { \tau } _ { t } = \pmb { \tau } _ { t } - 3 \pmb { \tau } _ { t - 1 } + 3 \pmb { \tau } _ { t - 2 } - \pmb { \tau } _ { t - 3 } . } \end{array}\tag{S4}
$$

The term lies in [0, 1.7] and is applied once four control steps of torque history are available.

## C. Foot-Clearance Penalty

Foot scuffing is penalized when a reference point on the foot is below a clearance height while moving horizontally:

$$
r _ { \mathtt { C } \mid \tau } = - \ k _ { \mathtt { C } \mid \tau } \sum _ { j \in \{ \mathsf { L } , \mathsf { R } \} } \operatorname* { m i n } \big ( 0 , \ y _ { j } - h _ { c } \big ) ^ { 2 } \big \| \mathbf { v } _ { j } ^ { x z } \big \| ,\tag{S5}
$$

where $y _ { j }$ is the height of a fixed point on foot j (0.15 m along the foot segment from its origin), $\mathbf { v } _ { j } ^ { x z }$ is that point’s horizonta velocity, and $h _ { c } = 0 . 0 3 \mathsf { m }$ . The penalty is zero when the point is above $h _ { c }$ or stationary on the ground, so it acts mainly during swing. For example, a point 1 cm below $h _ { c }$ moving at 1 m/s gives $r _ { \mathsf { c l r } } = - 0 . 0 1$

TABLE S1  
CONSTANTS OF THE AUXILIARY HUMAN REWARD TERMS.
<table><tr><td>Term</td><td>Symbol</td><td>Value</td><td>Description</td></tr><tr><td> $r _ { \mathrm { l i m } }$ </td><td> $k _ { \mathrm { l i m } }$ </td><td>100</td><td>penalty gain</td></tr><tr><td></td><td>θmin δ</td><td>-0.1 rad 0.1 rad</td><td>knee lower (extension) limit soft-limit margin</td></tr><tr><td> $r _ { \tau }$ </td><td>C</td><td> $1 0 ^ { - 3 } ( \mathsf { N } { \cdot } \mathsf { m } ) ^ { - 2 }$ </td><td>exponential scale</td></tr><tr><td> $r _ { \mathsf { C l r } }$ </td><td> $k _ { \mathsf { C l r } }$ </td><td> $1 0 0 \mathsf { s m } ^ { - 3 }$ </td><td></td></tr><tr><td></td><td></td><td></td><td>penalty gain</td></tr><tr><td></td><td> $h _ { c }$ </td><td>0.03 m</td><td>clearance height</td></tr></table>

## II. MUSCLE GROUPS USED IN THE SIMULATION ANALYSIS

The simulation analysis of the main text uses 31 muscle actuators per leg (line-of-action elements of the 304-actuator model), representing 15 anatomical muscles that act primarily in the sagittal plane. Each actuator is logged for the left and right legs, and all reported values are left–right averages (per leg). The actuators are assigned to six functional groups by joint and direction: hip extension (HE), hip flexion (HF), knee flexion (KF), knee extension (KE), ankle plantar flexion (PF), and ankle dorsiflexion (DF). Three joint groups combine both directions at each joint: hip (HE + HF), knee (KF + KE), and ankle (PF + DF). Table S2 lists the membership.

No actuator belongs to both directions of the same joint, so each joint group is the disjoint union of its two direction groups. Biarticular muscles (biceps femoris long head, semimembranosus, semitendinosus, rectus femoris, and the gastrocnemius) belong to groups at two joints, so the direction groups and the joint groups overlap across joints and their sums exceed the total over all 31 actuators. Group activation is the unweighted mean over member actuators, and group metabolic power is their sum.

TABLE S2  
MEMBERSHIP OF THE ANALYZED MUSCLE ACTUATORS (PER LEG) IN THE FUNCTIONAL AND JOINT GROUPS. • MARKS MEMBERSHIP. THE BOTTOM ROWS GIVE THE NUMBER OF ACTUATORS IN EACH GROUP.
<table><tr><td>Anatomical muscle</td><td>Actuators</td><td>Hip HE HF</td><td>Knee KF KE</td><td>Ankle PF DF</td></tr><tr><td>Biceps femoris, long head</td><td>1</td><td></td><td></td><td></td></tr><tr><td>Biceps femoris, short head</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Gastrocnemius, lateral head</td><td>1</td><td></td><td></td><td></td></tr><tr><td>Gastrocnemius, medial head</td><td>1</td><td></td><td></td><td></td></tr><tr><td>Gluteus maximus</td><td>5</td><td></td><td></td><td></td></tr><tr><td>Iliacus</td><td>3</td><td></td><td></td><td></td></tr><tr><td>Psoas major</td><td>3</td><td></td><td></td><td></td></tr><tr><td>Rectus femoris</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Semimembranosus</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Semitendinosus</td><td>1</td><td></td><td></td><td></td></tr><tr><td>Soleus</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Tibialis anterior</td><td>1</td><td></td><td></td><td></td></tr><tr><td>Vastus intermedius</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Vastus lateralis Vastus medialis</td><td>2 3</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Actuators per direction group Actuators per joint group</td><td>31</td><td>9 8 17</td><td>8 9 17</td><td>4 1 5</td></tr></table>