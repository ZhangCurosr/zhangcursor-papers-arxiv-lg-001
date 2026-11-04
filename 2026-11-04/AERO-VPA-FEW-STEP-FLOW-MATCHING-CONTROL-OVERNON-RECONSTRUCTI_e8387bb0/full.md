# AERO-VPA: FEW-STEP FLOW-MATCHING CONTROL OVERNON-RECONSTRUCTIVE JEPA LATENTS FOR AGILE FLIGHT ONEDGE HARDWARE

Yeonseok Lee<sup>∗</sup> SLING AI Inc. Incheon, Republic of Korea ylee@sling.ai.kr

Code: https://github.com/ylee-sling/Aero-VPA Trained policy (72.6 MB checkpoint): https://huggingface.co/ylee-sling/aero-vpa

## ABSTRACT

Vision-Language-Action (VLA) models generalize across tasks, but their autoregressive decoders need sequential forward passes in proportion to the action horizon, each bounded below on embedded aerial platforms by streaming billions of parameters from memory. We present Aero-VPA, a lightweight Vision-Predictor-Action architecture for 3D navigation on edge hardware. A compact RGB-D Vision Transformer and a five-head ensemble predictor, trained without pixel reconstruction, forecast the next latent state; a Conditional Flow Matching solver generates a velocity and yaw-rate chunk in two Euler steps, so each decision takes five sequential network evaluations; and the ensemble dispersion drives an uncertainty fallback. Version 1 specified the architecture, deployment and protocol without results. This version reports the first closed-loop evaluation: PX4 software-in-the-loop simulation of quadrotor missions in procedurally generated forests. Behaviour cloning failed on all 60 missions despite open-loop errors of 0.02–0.04 m/s: the policy copies its last command, a copycat form of causal confusion. DAgger with a privileged, safety-gated supervisor gave the first successes but plateaued at 11.1–21.1% across six checkpoints. A map-free runtime shield with a 15 s depth memory, barrier constraints, a cap on motion outside the 73° field of view and backoff along the flown path raised the first DAgger policy’s success on 180 paired missions from 21.1% to 52.2% (95% CI [45.0, 59.4]; exact McNemar $\stackrel { \cdot } { p } = \stackrel { \cdot } { 1 } . 2 \times 1 0 ^ { - 1 0 } )$ and cut its crashes from 72.2% to 22.2%, while timeouts rose from 2.8% to 23.3%. All six checkpoints gained 30.0–34.4 percentage points, at mean speeds of 1.16–1.36 m/s. The code (https://github.com/ylee-sling/Aero-VPA), the trained policy (72.6 MB, https://huggingface.co/ylee-sling/aero-vpa) and the full benchmark dataset are public.

Keywords Aerial Robotics · Joint-Embedding Predictive Architecture · Flow Matching · Edge Computing · Safe Learning-Based Control · Imitation Learning · Safety Filter

## 1 Introduction

Navigating cluttered three-dimensional environments at speed requires a micro aerial vehicle (MAV) to convert onboard sensing into motor commands within a fraction of the time it needs to reach the nearest obstacle. Modular pipelines that chain visual-inertial odometry, volumetric mapping, and trajectory optimization [1–4] remain the dominant engineering solution, but each stage adds latency and couples the system to the accuracy of an explicit map; learned sensorimotor policies have accordingly been shown to fly faster and more robustly in the wild [5] and to reach champion-level performance in drone racing [6, 7]. In parallel, Vision-Language-Action (VLA) models [8–10] have demonstrated broad semantic generalization by casting control as sequence modeling. Most VLAs, however, emit actions through an autoregressive language-model decoder: an H-step chunk of $d _ { a }$ -dimensional actions discretized into q tokens per component requires $\bar { N } \overset { = } { = } q H d _ { a }$ dependent forward passes. At batch size one, every pass must stream the decoder weights from memory, so the latency is bounded below by $N | W _ { \mathrm { L L M } } | b / B W$ , where $\vert W _ { \mathrm { L L M } } \vert$ is the number of parameters, b the bytes per parameter, and $B \bar { W }$ the memory bandwidth. On an NVIDIA Jetson Orin Nano 8 GB, whose LPDDR5 interface provides 68 GB/s in its original configuration and 102 GB/s in the JetPack $6 . 2 \cdots \mathrm { s u p e r } ^ { \prime \prime }$ configurations [11], a 3B-parameter model in 16-bit precision (6 GB, three quarters of the device memory) needs at least 88 ms per pass at 68 GB/s. A single four-component velocity command $( q = 1 , H = 1 , d _ { a } = 4 )$ then takes at least 0.35 s (at most 2.8 Hz), and a 16-step chunk at least 5.6 s, during which a vehicle flying at 5 m/s travels 28 m. These bounds are optimistic: they ignore compute, attention over the context, and memory shared with perception and middleware.

The Vision-Predictor-Action (VPA) framework [12] removes the language decoder from the control loop. A compact encoder maps observations into a latent space learned by a Joint-Embedding Predictive Architecture (JEPA) [13–15], which predicts future embeddings rather than future pixels and is therefore not required to model task-irrelevant appearance; a predictor forecasts the latent consequence of the selected skill; and a Conditional Flow Matching (CFM) solver [16–18] transports Gaussian noise to an entire action chunk by integrating a learned ordinary differential equation (ODE). Denoising diffusion policies [19] generate chunks by iterative denoising, commonly with 10 to 100 network evaluations per chunk; the CFM sampler is instead a deterministic ODE map from a sampled initial condition, and straight transport paths admit accurate integration in very few steps; the number of sequential network evaluations per decision is $K + 3$ for K integration steps, independent of the horizon H [12]. A reference implementation with fewer than 20M trainable parameters ran a decision step in 10.5–14.2 ms on an Apple M4 and succeeded in 40.0% of LIBERO-10 manipulation episodes [12]. Flight, however, differs from tabletop manipulation in three respects that this paper addresses: the platform is underactuated and cannot pause in free space, so the controller needs a safe action whenever its predictions become unreliable; commands must be streamed to a flight controller at a fixed rate under strict failsafe timing [20]; and the compute budget is that of an embedded module rather than a workstation.

Aero-VPA is an autonomous navigation framework incorporating a JEPA-based latent world model. The JEPA predictor serves as the latent dynamics world model predicting forward state transitions, while a flow-matching policy acts as the downstream controller. This paper formalizes Aero-VPA, a specialization of VPA to aerial navigation whose long-term goal is agile flight on edge hardware, specifies the protocol by which it is evaluated, and reports its first closed-loop evaluation. The architecture has three parts: a high-level generative actor, the primitive selector $\pi _ { \phi } ^ { h }$ and the CFM solver $v _ { \theta } ;$ a lightweight latent dynamics world model that conditions the actor, the JEPA encoder $E _ { \psi }$ and the five-head ensemble predictor $P _ { \omega } ,$ whose ensemble mean enters the actor’s conditioning and whose dispersion drives an uncertainty fallback, and which is not used for planning; and, added in this version, a deterministic low-level runtime safety shield.

From offline error to closed-loop failure. Version 1 of this preprint reported no experiments. In PX4 software-inthe-loop (SITL) simulation, the policy trained by behaviour cloning (BC) on about 5.5 million frames of a privileged expert matched the expert almost perfectly offline—on held-out expert frames, the median error of its first velocity knot was 0.02 m/s—but succeeded in none of 60 missions. Input-swap tests (Section 6.2) show why. The command follows the last dispatched setpoint ${ { \bar { a } } _ { t } } ,$ , which is part of the proprioceptive input $S _ { t } \colon$ replacing ${ { \bar { a } } _ { t } }$ by that of another frame changed the command by 0.85 m/s and moved it to within 0.31 m/s of the other frame’s label (medians; two frames’ labels differ by a median of 1.13 m/s), whereas replacing the image or the goal changed it by 0.01 m/s. On expert data, ${ { \bar { a } } _ { t } }$ lies within about 0.02 m/s of the next label, so repeating it is nearly optimal offline and useless in closed loop, where nothing corrects the drift. This is the copycat problem of behaviour cloning [21], a form of causal confusion [22]. Withholding ${ { \bar { a } } _ { t } }$ moved the shortcut to the measured velocity. DAgger [23], stage (iii) of the training procedure (Section 3.8), produced the first successes, but neither a second round nor four round-3 recipes aimed at the shortcut (input removal, state dropout, keyframe weighting) improved on the first round, and the DAgger policies still copied ${ { \bar { a } } _ { t } }$ on expert frames.

Safety under a narrow field of view. The front camera sees $7 3 ^ { \circ }$ horizontally, and in simulation the vehicle responded to a velocity setpoint after a dead time of 0.24 s with a first-order lag of 0.5 s. Across our evaluations, 26.2–36.4% of the policies’ decisions were blind—the commanded direction lay outside the field of view, or the depth there was invalid—compared with 3.7% for the expert on its own held-out flights. The braking envelope of the learned safety layer (Section 3.6) checks only the current frame, and with the identified response its stopping guarantee does not apply. A runtime safety layer therefore has to remember what the camera has seen, and it must not itself move blindly: in a pilot of our first shield version, 10 of its 11 crashes happened while the shield was flying the vehicle, and 8 of its 10 collisions were with obstacles beside or behind the camera.

Runtime shield. We keep the learned components, train them with DAgger against a privileged supervisor that both labels the visited states and takes over when a safety gate fires, and add a runtime shield with no map and no learned parameters (Section 3.9). The shield keeps a 15 s voxel memory of depth returns, so that obstacles beside and behind the camera stay known; projects the commanded velocity onto barrier constraints in the manner of control barrier functions [24]; caps horizontal motion outside the field of view; enforces an altitude floor; holds the vehicle when decisions stop; and, when the vehicle comes too close to a remembered point, retraces its own flown path and then nudges it toward the waypoint. The filtered setpoint is written back as the policy’s input ${ { \bar { a } } _ { t } }$

## Contributions.

• Aero-VPA formulation. A goal-conditioned partially observable formulation with chunks of heading-frame velocity and yaw-rate setpoints $A _ { t } = [ v _ { x } , v _ { y } , v _ { z } , \dot { \psi } ] _ { t : t + H - 1 } ;$ an RGB-D Vision Transformer encoder trained jointly with a primitive- and command-conditioned JEPA predictor under VICReg regularization [25]; and a CFM solver sampled with $K = 2$ Euler steps, giving a sequential depth of five network evaluations per decision. We add a noise-persistence rule and prove a Lipschitz bound on the change between consecutive chunks that it enables (Proposition 3.1).

• Latent-uncertainty safety layer. An epistemic uncertainty score computed from the normalized dispersion of a predictor ensemble [26] without access to future observations, an asymmetric filter, a monotone velocityattenuation law α(¯σ), an uncertainty-dependent chunk validity horizon, and a minimum-jerk braking transition to hover. A model-free braking envelope complements the learned signal, and we state the assumptions under which it guarantees a stop within the measured free space (Proposition 3.2) and why the evaluated configuration does not meet them (Remark 3.2).

• Asynchronous edge deployment. A two-rate architecture for the Jetson Orin Nano (8 GB, 15 W) and a PX4 flight controller connected through micro-XRCE-DDS [27], comprising latency-compensated chunk indexing, continuous chunk blending, minimum-acceleration setpoint interpolation, an analytic parameter and memory-bandwidth budget, and an adaptation to quadruped robots. The embedded deployment is specified, not yet built; the evaluation runs the same executor logic on a cloud GPU (Section 4.6).

• Benchmark protocol. A software-in-the-loop (SITL) and hardware-in-the-loop (HITL) protocol with baselines, metric definitions, falsifiable hypotheses with acceptance criteria, a statistical analysis plan, and an ablation matrix, all fixed in version 1 before any experiment was executed (Section 5). Section 6 executes a first part of it, and Section 8.3 lists every deviation.

• A closed-loop diagnosis of copycat imitation and of DAgger’s plateau. BC failed on all 60 missions despite open-loop errors of 0.02–0.04 m/s. Swap tests on expert frames and on the policies’ own states identify the shortcut—copying ${ { \bar { a } } _ { t } } ,$ or the measured velocity once ${ { \bar { a } } _ { t } }$ is withheld—and show that neither input removal, state dropout nor keyframe weighting removes it. DAgger with a privileged, safety-gated supervisor gave the first successes (21.1% against 0% for BC; exact McNemar $p = 0 . 0 0 0 5$ on the 60 shared missions), then plateaued: the second round was worse than the first (12.8%, $p \ = \ 0 . 0 3 1 5 )$ , no round-3 recipe beat the first round (11.1–15.6%), and the error on the policies’ own states stayed near 0.5 m/s across rounds (Sections 6.2–6.3).

• A map-free runtime shield and its failure analysis. On 180 paired test missions the shield raised success by 30.0–34.4 percentage points for all six DAgger checkpoints, to 52.2% [45.0, 59.4] for the first-round policy $( p = 1 . 2 \times \mathsf { \bar { 1 0 } ^ { - 1 0 } } )$ , and lowered crashes from 72.2–82.2% to 22.2–42.8%; its memory update took 2.9–3.0 ms per frame at the 95th percentile. The gain held on the 144 missions never used to revise the shield (first-round policy: 52.1%), and for the first-round policy it was largest in the densest worlds $( 6 . 7 \%  4 6 . 7 \% )$ . The shield moved failures from collisions to timeouts (2.8% → 23.3% for the first-round policy), and its design iteration moved its own collisions from obstacles beside or behind the camera (8 of 10 in the first version) to obstacles in view (6 of 7 in the second) (Sections 6.4–6.8 and 7).

• Open release. The trained policies are public on Hugging Face (GPL-3.0): the six DAgger checkpoints, each a single file of 72.6–72.7 MB, the three behaviour-cloning baselines and the two stage-(i) checkpoints. So are the code (GPL-3.0-only, release v1.1.0) [28] and the benchmark dataset of every closed-loop mission record, with the decision traces, the 36 test worlds and the analysis reports (CC BY 4.0) [29] (Code and Data Availability).

Changes in version 2. Version 1 (26 October 2026) specified the architecture, the deployment and the benchmark protocol and reported no results. Version 2 keeps Sections 2–5 and changes them only where the evaluated system differs or the evaluation required an addition: the predictor is in residual form (Eq. (7)); $\sigma _ { \mathrm { c r i t } }$ is calibrated from per-flight maxima (Section 3.5); Remark 3.2 states that the evaluated braking envelope does not satisfy Assumption 3.1; the DAgger stage is specified with the supervisor that was used (Section 3.8); the runtime shield is new (Section 3.9, Figure 1); Section 4.6 describes the implementation used in the evaluation; and Section 5 keeps the protocol of version 1 with its execution status. Sections 6–9 and the appendices are new; Section 2 adds work on causal confusion, interactive imitation and safety filters. The networks and their sizes are unchanged (18.13M trainable parameters).

Scope. All results are from PX4 SITL with Gazebo in static, procedurally generated forests, with one training seed per recipe. The shielded mean speeds were 1.16–1.36 m/s, far below the agile flight that remains the goal, and timing was measured on an NVIDIA L4 GPU, not on the Jetson Orin Nano of Section 4. The embedded deployment, the hardware-in-the-loop stage, the baselines and most ablations of Section 5 have not been executed.

## 2 Related Work

## 2.1 Visual Predictive World Models and Non-Reconstructive Latents

Generative world models learn dynamics by reconstructing observations and have achieved strong results in reinforce ment learning [30], but reconstruction obliges the model to account for every pixel, including illumination, texture, and sensor noise that are irrelevant to control. Joint-embedding predictive architectures [13] instead predict the embedding of a target from the embedding of a context: I-JEPA [14] predicts masked image regions in representation space, V-JEPA [15] extends this to video, and V-JEPA 2 [31] shows that an action-conditioned latent predictor post-trained on robot interaction data supports zero-shot manipulation through planning in representation space, at the cost of iterative optimization at inference time. TD-MPC2 [32] likewise learns a decoder-free latent model for continuous control. Without a reconstruction target, joint-embedding objectives admit collapsed solutions; VICReg [25] prevents them with variance and covariance penalties on the embedding batch. In robot learning, ACT-JEPA [33] uses latent prediction to regularize imitation policies, VLA-JEPA [34] adds a latent world model to a VLA backbone, and VPA [12, 35] couples a VICReg-regularized JEPA predictor with a discrete skill selector and a flow-matching action head without a language decoder. Aero-VPA retains this structure and adapts its inputs, conditioning, and safety logic to flight.

## 2.2 Continuous Normalizing Flows and Flow Matching for Robot Action Generation

Continuous normalizing flows define a generative model by integrating a learned vector field [36]. Flow matching [16], rectified flow [17], and stochastic interpolants [18] train such fields by regression onto the velocity of a prescribed probability path, avoiding simulation during training; straight paths permit few-step integration. In robotics, action chunking with transformers (ACT) [37] showed that predicting multi-step chunks mitigates compounding error, Diffusion Policy [19] models multimodal chunk distributions by iterative denoising, and consistency distillation reduces the number of denoising steps [38]. $\pi _ { 0 }$ [10] attaches a flow-matching action expert to a multi-billion-parameter visionlanguage model, and Octo [39] attaches a diffusion head to a transformer backbone of 27M–93M parameters. Real-time chunking [40] addresses the discontinuities that arise when chunks are replanned asynchronously by inpainting the portion of the next chunk that overlaps the current one. Aero-VPA uses a CFM head with K = 2, conditions it on forecast latents rather than on the hidden states of a language backbone, and addresses inter-chunk consistency by noise persistence and continuous blending (Section 3.4).

Closest to our work is Qantara [41], which also couples a JEPA latent model with flow matching on actions. Qantara trains its encoder and a shared block-causal transformer predictor jointly from scratch: the next latent is learned with a Brownian-bridge objective anchored at the current latent, the action with a noise-to-data flow-matching objective, and SIGReg serves as the only anti-collapse term. By sampling separate noise levels for the action and latent axes, $( \tau ^ { a } , \tau ^ { z } )$ in their notation, along the edges and the diagonal of the unit square, a single 21M-parameter checkpoint serves latent planning with the cross-entropy method, behaviour cloning, and a video-inverse path that predicts the next latent and then infers the connecting action. Aero-VPA operates at a similar scale, also samples actions with two Euler steps, and follows the same predict-then-act chain as that last path, but resolves the design trade-off differently: it commits to a single goal- and primitive-conditioned inference path and decouples its components. The VICReg-regularized encoder and ensemble predictor are trained first and frozen, which fixes the latent scale that normalizes the uncertainty signal of Section 3.5. The CFM head is trained afterwards on velocity chunks, so it can be retrained per embodiment and compiled as an independent inference engine. Qantara is evaluated in simulated navigation, manipulation, and reaching tasks, with timing measured on an H100 GPU. Aero-VPA instead targets agile three-dimensional flight under the fixed-rate setpoint interface of PX4 on a 15 W embedded module, and couples action generation to an uncertainty-gated safety layer.

## 2.3 Edge Autonomous Flight and Computational Constraints in Aerial and Legged Systems

Classical agile-flight stacks combine visual-inertial state estimation [1, 42], occupancy mapping [2], and polynomial or gradient-based planning [3, 4], and have been scaled to swarms in unstructured environments [43]. Learning-based approaches replace part of this stack: Loquercio et al. [5] trained a depth-based policy by imitating a privileged planner and executed it on an embedded GPU, Kaufmann et al. [6] trained racing policies with deep reinforcement learning, and Agilicious [44] provides open hardware and software with onboard GPU compute for vision-based agile flight. These systems flew far faster than the 1.16–1.36 m/s reported in Section 6. Learned residual dynamics address aerodynamic effects such as ground effect [45] and strong wind [46]. General navigation models such as GNM [47] and NoMaD [48] transfer across ground robots but target lower speeds. In legged locomotion, hierarchical designs pair a high-level command policy with a low-level controller trained by reinforcement learning [49–51]; this separation of time scales motivates the cross-embodiment discussion in Section 4.5. Across these platforms the binding constraints are size, weight, and power (SWaP), memory bandwidth, and the fixed-rate interfaces of the low-level controller [20, 52], which together determine the design choices of Section 4.

## 2.4 Causal Confusion and Interactive Imitation

De Haan et al. [22] showed that giving an imitation policy more information can hurt it, because it may rely on nuisance correlates of the expert’s action. Wen et al. [21] described the copycat problem of policies that observe their own history and repeat their previous action, and later reweighted training toward keyframes where the action changes [53]; our $\mathrm { D } _ { 3 , \mathrm { K W } }$ recipe follows that idea. Codevilla et al. [54] described the inertia problem of driving policies that condition on their speed, and ChauffeurNet [55] dropped out past motion during training, the precedent of our state-dropout recipe $\mathrm { D } _ { 3 , \mathrm { S D } }$ . In Aero-VPA the copied quantity is the last dispatched velocity setpoint ${ { \bar { a } } _ { t } } .$ , and withholding it moves the shortcut to the measured velocity.

DAgger [23] labels the states the learner visits. Learning by Cheating [56] first trains a privileged agent and then distils it into a sensorimotor student, and SafeDAgger [57] and HG-DAgger [58] let a learned safety policy or a human expert decide when the expert takes over. Our supervisor combines these roles: a privileged planner labels every visited state and takes control when a safety gate fires, softly when it can keep flying its plan.

## 2.5 Safety Filters, Shields, and Barrier Functions

Shields [59] override a learning agent’s actions that would violate a specification, and safety filters [60, 61] minimally modify a nominal controller’s command, often through control barrier functions [24]. NanoMap [62] answers proximity queries against a history of depth frames, so that a vehicle with a narrow field of view can account for obstacles it no longer sees. Our runtime shield combines such a memory with a barrier-function-style velocity constraint enforced by sequential projection and with recovery manoeuvres. It provides no formal guarantee; Section 3.9 states the conditions under which its constraint is invariant.

## 3 Problem Formulation and Proposed Architecture

## 3.1 Goal-Conditioned POMDP with Velocity Chunks

We model goal-directed flight as a goal-conditioned partially observable Markov decision process (POMDP)

$$
\begin{array} { r } { \mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { O } , \mathcal { T } , \Omega , \mathcal { G } , R , \gamma _ { \mathrm { d i s c } } ) , } \end{array}\tag{1}
$$

whose components are defined as follows. A state $s = ( p , v , \mathbf { R } , \omega , w , \mathcal { X } ) \in \mathcal { S }$ comprises the position $p \in \mathbb { R } ^ { 3 }$ and velocity $v \in \mathbb { R } ^ { 3 }$ of the vehicle in a local north-east-down (NED) world frame W, its attitude $\bar { \mathbf { R } } \in S O ( 3 )$ , its body angular rate ω $\in \mathbb { R } ^ { 3 }$ , the ambient wind $w \in \mathbb { R } ^ { 3 }$ , and the occupied set $\mathcal { X } \subset \mathbb { R } ^ { 3 }$ of the environment. The vehicle observes $o _ { t } \overset { \cdot } { = } ( I _ { t } , S _ { t } ) \in \mathcal { O }$ at decision step $t ,$ captured at wall-clock time $\tau _ { t } .$ , where $I _ { t } \in \mathbb { R } ^ { H _ { \mathrm { i m g } } \times W _ { \mathrm { i m g } } \times 4 }$ is an RGB-D frame $( H _ { \mathrm { i m g } } = 1 2 8 , W _ { \mathrm { i m g } } = 1 6 0 )$ and

$$
S _ { t } = \left( \hat { v } _ { t } ^ { \mathcal { H } } , \hat { n } _ { t } ^ { \mathcal { B } } , \hat { \omega } _ { t } ^ { \mathcal { B } } , \bar { a } _ { t } \right) \in \mathbb { R } ^ { d _ { s } } , \qquad d _ { s } = 1 3 ,\tag{2}
$$

is the proprioceptive state estimated by the PX4 EKF2: the velocity $\hat { v } _ { t } ^ { \mathcal { H } } \in \mathbb { R } ^ { 3 }$ expressed in the heading frame $\mathcal { H } _ { t }$ (the gravity-aligned forward-right-down frame obtained by rotating W about its down axis by the estimated yaw $\hat { \psi } _ { t } )$ , the unit gravity direction $\hat { n } _ { t } ^ { B } \in \mathbb S ^ { 2 }$ in the body frame $B ,$ , which encodes roll and pitch, the body rates $\hat { \omega } _ { t } ^ { B } \in \mathbb { R } ^ { 3 }$ , and the setpoint $\bar { a } _ { t } \in \mathbb { R } ^ { 4 }$ dispatched most recently to the flight controller, expressed in $\bar { \mathcal { H } } _ { t }$ . The observation model Ω is induced by image formation, $I _ { t } = \mathcal { T } ( s ( \tau _ { t } ) , \eta _ { t } )$ , where $\eta _ { t }$ collects nuisance factors such as illumination, texture, motion blur, and depth noise.

A mission $G = ( x , ( w ^ { ( m ) } ) _ { m = 1 } ^ { M } ) \in \mathcal { G }$ consists of an optional natural-language instruction x and an ordered sequence of M waypoints $\boldsymbol { w } ^ { ( m ) } \in \mathbb { R } ^ { 3 }$ in W. With the active waypoint index $m _ { t }$ (Section 3.3) and the estimated position $\hat { p } _ { t } .$ , the

![](images/a5c9149301bbc871ead2e207bf31fe586660be192e9c7191e22597773a064b38.jpg)  
Figure 1: Aero-VPA deployment architecture. The inference loop (top) runs once per camera frame and produces a velocity and yaw-rate chunk $\mathcal { C } _ { t }$ through five sequential network evaluations; the safety filter applies the uncertaintydependent attenuation $\alpha ( \bar { \sigma } )$ , the braking envelope $v _ { \mathrm { s a f e } }$ , and the validity horizon $H _ { t }$ . The executor (bottom) evaluates the newest valid chunk at its chunk-local time, converts it to NED setpoints, and streams them to PX4 over micro-XRCE-DDS; it engages the braking fallback when a fallback is requested or no valid chunk is available. The runtime shield (orange, added in version 2) filters every setpoint against a 15 s depth memory or flies recovery manoeuvres itself; the setpoint it dispatches is fed back as ${ { \bar { a } } _ { t } }$ (dashed). In the evaluation of Section 6 the companion-computer software ran on a cloud GPU against PX4 SITL (Section 4.6).

policy receives the goal feature

$$
g _ { t } = \left( \frac { r _ { t } } { \operatorname* { m a x } ( \lVert r _ { t } \rVert _ { 2 } , \epsilon _ { r } ) } , \frac { \operatorname* { m i n } ( \lVert r _ { t } \rVert _ { 2 } , D _ { \operatorname* { m a x } } ) } { D _ { \operatorname* { m a x } } } \right) \in \mathbb { R } ^ { 4 } , \qquad r _ { t } = \mathbf { R } _ { z } ( \hat { \psi } _ { t } ) ^ { \top } \big ( w ^ { ( m _ { t } ) } - \hat { p } _ { t } \big ) ,\tag{3}
$$

where $\mathbf { R } _ { z } ( \psi ) \in S O ( 3 )$ is the rotation about the down axis by ψ, $D _ { \mathrm { m a x } }$ is a saturation distance, and $\epsilon _ { r } > 0$ avoids division by zero.

The per-step action $a = ( v _ { x } , v _ { y } , v _ { z } , \dot { \psi } ) \in A _ { 0 } = [ - \bar { v } _ { h } , \bar { v } _ { h } ] ^ { 2 } \times [ - \bar { v } _ { z } , \bar { v } _ { z } ] \times [ - \bar { \dot { \psi } } , \bar { \dot { \psi } } ]$ is a velocity setpoint and a yaw-rate setpoint, with speed limits $\bar { v } _ { h } , \bar { v } _ { z }$ , and $\dot { \psi }$ . At each decision step the controller outputs an open-loop chunk of H knots spaced $\Delta$ seconds apart,

$$
A _ { t } = [ v _ { x } , v _ { y } , v _ { z } , \dot { \psi } ] _ { t : t + H - 1 } = \left[ a _ { t , 0 } ; a _ { t , 1 } ; \ldots ; a _ { t , H - 1 } \right] \in \mathcal { A } _ { 0 } ^ { H } \subset \mathbb { R } ^ { H \times d _ { a } } , \qquad d _ { a } = 4 ,\tag{4}
$$

where knot $a _ { t , k }$ applies at time $\tau _ { t } + k \Delta$ and its velocity components are expressed in the heading frame $\mathcal { H } _ { t }$ frozen at the capture time $\tau _ { t }$ . We use $\Delta = 2 0$ ms and $H = 1 6 ,$ so that a chunk spans 0.32 s. The transition kernel $\mathcal { T } ( s ^ { \prime } \mid s , a )$ is induced by the PX4 velocity and attitude controllers, the rigid-body and rotor dynamics, and the wind. The reward

$$
R ( s , G ; m ) = - \left\| p - w ^ { ( m ) } \right\| _ { 2 } - \lambda _ { \mathrm { c o l } } \mathcal { k } [ \mathrm { d i s t } ( p , \mathcal { X } ) < r _ { v } ] ,\tag{5}
$$

with vehicle radius $r _ { v }$ and collision weight $\lambda _ { \mathrm { c o l } } > 0 .$ , and the discount $\gamma _ { \mathrm { d i s c } } \in [ 0 , 1 )$ define the task for analysis and for the reinforcement-learning baselines of Section 5.3. Aero-VPA itself is trained by self-supervised latent prediction and by imitation of a privileged expert (Section 3.8).

## 3.2 Non-Reconstructive JEPA World Model

A Vision Transformer [63] $E _ { \psi }$ with patch size 16 divides the four-channel frame into $N _ { p } = ( 1 2 8 / 1 6 ) ( 1 6 0 / 1 6 ) = 8 0$ patches, processes them together with a class token through six blocks of width 384, and projects the class token to the latent

$$
z _ { t } = E _ { \psi } ( I _ { t } ) \in \mathcal { Z } \subset \mathbb { R } ^ { d } , \qquad d = 2 5 6 .\tag{6}
$$

The depth channel is clipped to the sensor range and normalized before patch embedding. When an instruction x is given, a frozen CLIP text encoder [64] computes $c _ { \mathrm { t e x t } } \in \mathbb { R } ^ { 5 1 2 }$ once per mission; otherwise $c _ { \mathrm { t e x t } }$ is a learned null embedding.

The predictor $P _ { \omega }$ forecasts the latent state $T _ { \nu } = 0 . 1 \ : \mathrm { s }$ s ahead, conditioned on the current latent, the active primitive $u _ { t } \in \mathcal { U }$ (Section 3.3), and the executing setpoint ${ { \bar { a } } _ { t } }$ . It consists of $N _ { e } = 5$ independently initialized heads $P _ { \omega } ^ { ( i ) }$ in residual form: head i adds the output of an MLP $f _ { \omega } ^ { ( i ) } ( 2 9 2  5 1 2  5 1 2  2 5 6 )$ to the current latent,

$$
\hat { z } _ { t + 1 } ^ { ( i ) } = z _ { t } + f _ { \omega } ^ { ( i ) } \big ( z _ { t } , \mathrm { E m b e d } ( u _ { t } ) , \bar { a } _ { t } \big ) , \qquad \hat { z } _ { t + 1 } = \frac { 1 } { N _ { e } } \sum _ { i = 1 } ^ { N _ { e } } \hat { z } _ { t + 1 } ^ { ( i ) } ,\tag{7}
$$

where Embed : $\mathcal { U } \to \mathbb { R } ^ { d _ { u } }$ is a learned primitive embedding $( d _ { u } = 3 2 )$ and the subscript $t + 1$ abbreviates the time $\tau _ { t } + T _ { \nu }$ . Version 1 wrote the heads without the residual connection; every evaluated policy except ${ \mathrm { B C } } _ { \mathrm { N R } }$ (Table 4) uses the residual form. Conditioning on ${ { \bar { a } } _ { t } } ,$ , which is known before the decision, makes the predictor action-conditioned without placing the solver output on its input, so the dependency chain of a decision step remains acyclic. The predictor is not used for planning, imagined rollouts or model-predictive control: its ensemble mean conditions the solver (Eq. (15)), and its dispersion drives the uncertainty layer (Section 3.5). The encoder and predictor are trained jointly with

$$
\mathcal { L } _ { \mathrm { J E P A } } ( \psi , \omega ) = \mathbb { E } _ { t } \left[ \frac { 1 } { N _ { e } } \sum _ { i = 1 } ^ { N _ { e } } \left. \hat { z } _ { t + 1 } ^ { ( i ) } - \mathrm { s g } \big ( { E _ { \bar { \psi } } ( I _ { t + 1 } ) } \big ) \right. _ { 2 } ^ { 2 } \right] + \lambda _ { v } v ( Z ) + \lambda _ { c } c ( Z ) ,\tag{8}
$$

where $I _ { t + 1 }$ is the frame captured closest to $\tau _ { t } + T _ { \nu } , \mathrm { s g } ( \cdot )$ is the stop-gradient operator, $E _ { \bar { \psi } }$ is a momentum copy of the encoder updated as $\bar { \psi }  \mu \bar { \psi } + ( 1 - \mu ) \psi$ with $\mu \in ( 0 , 1 )$ , and $Z \in \mathbb { R } ^ { B \times d }$ is the batch of online latents $z _ { t }$ . The variance and covariance penalties of VICReg [25] are

$$
v ( Z ) = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \operatorname* { m a x } \Bigl ( 0 , \gamma - \sqrt { \mathrm { V a r } ( Z _ { \cdot , j } ) + \epsilon } \Bigr ) ,\tag{9}
$$

$$
c ( Z ) = \frac { 1 } { d } \sum _ { i \neq j } \left[ C ( Z ) \right] _ { i , j } ^ { 2 } , \qquad C ( Z ) = \frac { 1 } { B - 1 } \sum _ { b = 1 } ^ { B } \left( Z _ { b , \cdot } - \bar { Z } \right) ^ { \top } \left( Z _ { b , \cdot } - \bar { Z } \right) ,\tag{10}
$$

with margin $\gamma > 0$ , numerical constant $0 < \epsilon < \gamma ^ { 2 }$ , and batch mean $\bar { Z } \in \mathbb { R } ^ { 1 \times d }$ . The variance hinge counteracts collapse to a constant embedding and the covariance term counteracts dimensional collapse; the conditions under which the constant encoder is excluded from the global minimizers, and the limits of this guarantee, are analyzed in [12]. During training we track the realized latent scale

$$
\bar { \gamma } _ { n } = \left( 1 - \alpha _ { \gamma } \right) \bar { \gamma } _ { n - 1 } + \alpha _ { \gamma } \Big ( \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \mathrm { V a r } \left( Z _ { \cdot , j } ^ { ( n ) } \right) + \epsilon \Big ) ^ { 1 / 2 } ,\tag{11}
$$

where $Z ^ { ( n ) }$ is the batch at iteration n and $\alpha _ { \gamma } \in ( 0 , 1 ]$ . After this stage $E _ { \psi }$ and $P _ { \omega }$ are frozen, which fixes the scale $\bar { \gamma } ^ { \star }$ and the latent mean $\bar { \mu } _ { Z }$ used below. The evaluated system uses $\lambda _ { v } \overset { \_ } { = } 2 5 , \lambda _ { c } = 1$ and $\mu = 0 . 9 9 6$ (Appendix A).

## 3.3 Primitive Selection and Waypoint Progression

The selector $\pi _ { \phi } ^ { h }$ is a multilayer perceptron that chooses one of $N _ { u } ~ = ~ 6$ flight primitives $\mathcal { U } ~ = ~ \{ { \bf C } \mathrm { R U I S E } ,$ LATERALAVOID, VERTICALAVOID, YAWSCAN, APPROACH, BRAKEHOVER}:

$$
u _ { t } = \underset { u \in \mathcal { U } } { \arg \operatorname* { m a x } } \pi _ { \phi } ^ { h } \big ( u \mid z _ { t } , g _ { t } , S _ { t } , c _ { \mathrm { t e x t } } \big ) ,\tag{12}
$$

trained with the cross-entropy loss

$$
\mathcal { L } _ { \mathrm { C E } } ( \phi ) = - \mathbb { E } _ { t } \left[ \log \pi _ { \phi } ^ { h } ( u _ { t } ^ { \star } \mid z _ { t } , g _ { t } , S _ { t } , c _ { \mathrm { t e x t } } ) \right] ,\tag{13}
$$

where $u _ { t } ^ { \star }$ is a label derived from the expert command by fixed rules applied in priority order (BRAKEHOVER if the expert speed is below a hover threshold; APPROACH within a distance of the final waypoint; VERTICALAVOID, LATERALAVOID, or YAWSCAN if the corresponding command component exceeds a threshold; CRUISE otherwise). These rules are a stand-in for annotated or learned segmentations and are not part of the method. The selector output is restricted to $u ,$ and BRAKEHOVER doubles as the target of the safety override (Section 3.5).

The active waypoint advances by the symbolic rule

$$
m _ { t + 1 } = { \left\{ \begin{array} { l l } { m _ { t } + 1 , } & { { \mathrm { i f ~ } } \left\| { \hat { p } } _ { t } - w ^ { ( m _ { t } ) } \right\| _ { 2 } < \tau _ { w } { \mathrm { ~ a n d ~ } } m _ { t } < M , } \\ { m _ { t } , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{14}
$$

with acceptance radius $\tau _ { w } > 0 .$ . In [12] the corresponding test is applied to latent distances to goal images, and its soundness depends on a lower co-Lipschitz bound of the encoder that cannot be enforced architecturally. Because flight waypoints are geometric, Eq. (14) tests the state estimate instead: if $\begin{array} { r } { \| \hat { p } _ { t } - p _ { t } \| _ { 2 } \le \epsilon _ { p } , } \end{array}$ a transition implies $\left\| p _ { t } - w ^ { ( m _ { t } ) } \right\| _ { 2 } < \tau _ { w } + \epsilon _ { p }$ by the triangle inequality, without any assumption on $E _ { \psi }$

## 3.4 Conditional Flow-Matching Policy Solver

The solver conditions on the standardized vector

$$
e _ { t } = \mathrm { C o n c a t } \left( \mathrm { E m b e d } ( u _ { t } ) , \ \tilde { z } _ { t } , \ \tilde { \hat { z } } _ { t + 1 } , \ \tilde { S } _ { t } , \ g _ { t } , \ c _ { \mathrm { t e x t } } \right) \in \mathbb { R } ^ { d _ { c } } , \qquad d _ { e } = d _ { u } + 2 d + d _ { s } + 4 + 5 1 2 = 1 0 7 3 ,\tag{15}
$$

where $\tilde { z } = ( z - \bar { \mu } _ { Z } ) / \bar { \gamma } ^ { \star }$ is applied to both latents and $\tilde { S } _ { t }$ is $S _ { t }$ standardized with dataset statistics. Actions are mapped to $[ - 1 , 1 ]$ componentwise by the box bounds of ${ \mathcal { A } } _ { 0 } ;$ we keep the symbol $A _ { t }$ for the normalized chunk. The vector field $v _ { \theta } : \mathbb { R } ^ { H \times d _ { a } } \times \mathbf { \bar { \rho } } [ 0 , 1 ] \times \mathbb { R } ^ { d _ { e } } \to \mathbb { R } ^ { H \times d _ { a } }$ is a four-block transformer of width 256 that attends over the H knot tokens and a conditioning token. With integration time $\rho \in [ 0 , 1 ]$ and Gaussian noise $\xi \sim \mathcal { N } ( 0 , I ) , \xi \in \mathbb { R } ^ { H \times d _ { a } }$ , the linear (optimal-transport) path

$$
A _ { t } ^ { ( \rho ) } = \rho A _ { t } + ( 1 - \rho ) \xi\tag{16}
$$

has constant conditional velocity $A _ { t } - \xi ,$ , and $v _ { \theta }$ is trained by the flow-matching regression

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , \xi , \rho \sim \mathcal { U } [ 0 , 1 ] } \left[ \Big | \Big | \boldsymbol { v } _ { \theta } \big ( \boldsymbol { A } _ { t } ^ { ( \rho ) } , \rho \mid \boldsymbol { e } _ { t } \big ) - \big ( \boldsymbol { A } _ { t } - \xi \big ) \Big | \Big | _ { F } ^ { 2 } \right] ,\tag{17}
$$

where $\left\| \cdot \right\| _ { F }$ is the Frobenius norm and the expectation over t runs over the training data. At deployment, a chunk is obtained by integrating ${ \mathrm { d } } A / { \mathrm { d } } \rho = v _ { \theta } ( A , \rho \mid e _ { t } )$ from $A ^ { ( 0 ) } = \xi _ { t }$ with K Euler steps,

$$
A ^ { ( \rho _ { k + 1 } ) } = A ^ { ( \rho _ { k } ) } + \frac { 1 } { K } v _ { \theta } \big ( A ^ { ( \rho _ { k } ) } , \rho _ { k } \bigm | e _ { t } \big ) , \qquad \rho _ { k } = \frac { k } { K } , \quad k = 0 , \dots , K - 1 ,\tag{18}
$$

which for $K = 2$ reads

$$
\begin{array} { r } { \hat { A } _ { t } = \xi _ { t } + \frac { 1 } { 2 } v _ { \theta } ( \xi _ { t } , 0 \mid e _ { t } ) + \frac { 1 } { 2 } v _ { \theta } \Big ( \xi _ { t } + \frac { 1 } { 2 } v _ { \theta } ( \xi _ { t } , 0 \mid e _ { t } ) , \frac { 1 } { 2 } \Big | e _ { t } \Big ) . } \end{array}\tag{19}
$$

The chunk is then denormalized and projected componentwise onto $\mathcal { A } _ { 0 }$ . Although each conditional path in Eq. (16) is straight, the learned marginal field can be curved where the action distribution is multimodal, so the accuracy of two-step integration is an empirical property to be measured (ablation A1, Section 5.7).

Noise persistence. In [12], frequent replanning with fresh noise reduced manipulation success, plausibly because consecutive chunks switched between distinct but individually valid modes. Because the sampler of Eq. (18) is deterministic given $\xi _ { t }$ , we reuse the initial noise across decisions,

$$
\begin{array} { r } { \xi _ { t } = \left\{ \begin{array} { l l } { \xi _ { t - 1 } , } & { \mathrm { i f ~ } u _ { t } = u _ { t - 1 } \mathrm { ~ a n d ~ } t \not \equiv 0 \pmod { N _ { \xi } } , } \\ { \mathrm { a ~ f r e s h ~ d r a w ~ f r o m ~ } \mathcal { N } ( 0 , I ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{20}
$$

so that the noise is redrawn when the primitive changes and at least every $N _ { \xi }$ decisions. The following proposition shows what persistence buys.

Proposition 3.1 (Chunk coherence under noise persistence). Suppose that, on a set containing all Euler iterates, v<sub>θ</sub> is $L _ { A }$ -Lipschitz in its first argument and $L _ { e } – L i p s c h i t z$ in its conditioning argument with respect to $\left\| \cdot \right\| _ { F } a n d \left\| \cdot \right\| _ { 2 } ,$ respectively, with $L _ { A } > 0$ . Let $\breve { \Phi } _ { K } ( \xi ; e )$ denote the output of Eq. (18) for initial condition ξ and conditioning e. Then, for any ξ and any $e , e ^ { \prime } ,$

$$
\| \Phi _ { K } ( \xi ; e ) - \Phi _ { K } ( \xi ; e ^ { \prime } ) \| _ { F } \le \frac { L _ { e } } { L _ { A } } \Big [ \big ( 1 + \frac { L _ { A } } { K } \big ) ^ { K } - 1 \Big ] \ \| e - e ^ { \prime } \| _ { 2 } \le \frac { L _ { e } } { L _ { A } } \big ( e ^ { L _ { A } } - 1 \big ) \ \| e - e ^ { \prime } \| _ { 2 } .\tag{21}
$$

Proof. Let $\delta _ { k } = \left. A ^ { ( \rho _ { k } ) } - A ^ { \prime ( \rho _ { k } ) } \right. _ { F }$ for the iterates under e and $e ^ { \prime } { \mathrm { . } }$ , so $\delta _ { 0 } = 0$ . By Eq. (18) and the triangle inequality, $\delta _ { k + 1 } \leq ( 1 + L _ { A } / K ) \delta _ { k } + ( L _ { e } / K ) \| e - e ^ { \prime } \| _ { 2 }$ . Unrolling the recursion gives $\begin{array} { r } { \delta _ { K } \leq ( L _ { e } / K ) \| e - e ^ { \prime } \| _ { 2 } \sum _ { j = 0 } ^ { K - 1 } ( 1 + } \end{array}$ $L _ { A } / K ) ^ { j } = ( L _ { e } / L _ { A } ) [ ( 1 + L _ { A } / K ) ^ { K } - 1 ] \parallel e - e ^ { \prime } \parallel _ { 2 } .$ , and $( 1 + L _ { A } / K ) ^ { K } \le e ^ { L _ { A } }$ □

With a shared $\xi ,$ the change between consecutive chunks is therefore bounded in proportion to the change of the conditioning vector, which is small between decisions 33 ms apart; with independent draws no such bound exists, since $\xi _ { t }$ and $\xi _ { t - 1 }$ may be transported to different modes. The constants $L _ { A }$ and $L _ { e }$ are not controlled by training, and dot-product attention is not globally Lipschitz [65], so Eq. (21) explains the mechanism rather than providing a certified bound; its practical effect is tested in ablation A5.

## 3.5 Epistemic Uncertainty and Velocity Attenuation

At deployment the future latent is unobserved, so the reliability of $\hat { z } _ { t + 1 }$ must be assessed from quantities available at time $\tau _ { t }$ . Following deep ensembles [26], we use the disagreement of the heads in Eq. (7), normalized by the frozen latent scale so that it is dimensionless:

$$
\sigma _ { t + 1 } = \frac { 1 } { \sqrt { d } \bar { \gamma } ^ { \star } } \left( \frac { 1 } { N _ { e } } \sum _ { i = 1 } ^ { N _ { e } } \left\| \hat { z } _ { t + 1 } ^ { ( i ) } - \hat { z } _ { t + 1 } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } .\tag{22}
$$

An asymmetric filter reacts immediately to increases and decays slowly,

$$
\bar { \sigma } _ { t + 1 } = \operatorname* { m a x } \big \{ \sigma _ { t + 1 } , ( 1 - \alpha _ { \sigma } ) \bar { \sigma } _ { t } + \alpha _ { \sigma } \sigma _ { t + 1 } \big \} , \qquad \alpha _ { \sigma } \in ( 0 , 1 ] ,\tag{23}
$$

which prevents rapid fluctuations of $\sigma _ { t + 1 }$ from modulating the command at the decision rate. The filtered score governs three mechanisms. First, the translational components of the chunk are attenuated by the monotone, piecewise-linear law

$$
\alpha ( \bar { \sigma } ) = \left\{ \begin{array} { l l } { 1 , } & { \bar { \sigma } \leq \sigma _ { \mathrm { l o } } , } \\ { 1 - ( 1 - \alpha _ { \mathrm { m i n } } ) \displaystyle \frac { \bar { \sigma } - \sigma _ { \mathrm { l o } } } { \sigma _ { \mathrm { c r i t } } - \sigma _ { \mathrm { l o } } } , } & { \sigma _ { \mathrm { l o } } < \bar { \sigma } < \sigma _ { \mathrm { c r i t } } , } \end{array} \right.\tag{24}
$$

with $0 < \alpha _ { \mathrm { m i n } } \le 1$ and $0 \leq \sigma _ { \mathrm { l o } } < \sigma _ { \mathrm { c r i t } } ;$ α is non-increasing and Lipschitz with constant $( 1 - \alpha _ { \mathrm { m i n } } ) / ( \sigma _ { \mathrm { c r i t } } - \sigma _ { \mathrm { l o } } )$ and the yaw rate is not attenuated so that the vehicle can keep reorienting its camera. Second, each chunk carries the validity horizon

$$
H _ { t } = \operatorname* { m a x } \Big ( H _ { \operatorname* { m i n } } , ~ \big \lfloor H _ { \operatorname* { m a x } } \exp ( - \beta \bar { \sigma } _ { t + 1 } ) \big \rfloor \Big ) , \qquad 1 \le H _ { \operatorname* { m i n } } \le H _ { \operatorname* { m a x } } = H ,\tag{25}
$$

with sensitivity $\beta > 0 :$ the executor (Section 3.7) may use the chunk only while its age satisfies $\tau - \tau _ { t } \leq H _ { t } \Delta .$ . In the synchronous execution of [12], $H _ { t }$ is the number of knots executed before replanning; in the asynchronous design adopted here, newer chunks preempt older ones and $H _ { t }$ bounds how stale a chunk may become, for instance when inference stalls, before the fallback is engaged. Third, if $\bar { \sigma } _ { t + 1 } \geq \sigma _ { \mathrm { c r i t } }$ <sub>t</sub>, the primitive is overridden, $u _ { t } \gets \mathbf { B R A K E H O V E R }$ and the executor performs the braking manoeuvre of Section 3.6.

The thresholds are calibrated on in-distribution validation flights: $\sigma _ { \mathrm { l o } }$ is the empirical 0.9-quantile of ${ \bar { \sigma } } ,$ , and $\sigma _ { \mathrm { c r i t } }$ is its $\left( 1 - p _ { \mathrm { f a } } \right)$ -quantile, with $p _ { \mathrm { f a } }$ chosen such that the expected number of spurious fallbacks per mission, $p _ { \mathrm { f a } } f _ { \mathrm { d e c } } T _ { \mathrm { m i s } } ,$ , does not exceed a target $n _ { \mathrm { f a } }$ , where $f _ { \mathrm { d e c } }$ is the decision rate and $T _ { \mathrm { m i s } }$ the mission duration. For $f _ { \mathrm { d e c } } = 3 0 \mathrm { H z } , T _ { \mathrm { m i s } } = 6 0 \mathrm { s } .$ and $n _ { \mathrm { f a } } = 0 . 0 5$ , this gives $p _ { \mathrm { f a } } \approx 2 . 8 \times 1 0 ^ { - 5 }$ , which requires a large validation set because consecutive values of σ¯ are strongly correlated. The trade-off between safety and conservatism that these choices induce is mapped by ablation A2. The evaluated system calibrates $\sigma _ { \mathrm { c r i t } }$ differently, from per-flight maxima (the episode-maximum method): with $\alpha _ { \sigma } = 0 . 1$ and $\sigma _ { \mathrm { l o } }$ the 0.9-quantile of $\bar { \sigma }$ on 360 validation flights, $\sigma _ { \mathrm { c r i t } }$ is calibrated per checkpoint from the maxima of σ¯ of the individual flights with a Poisson model, for about 0.05 spurious fallbacks per 60 s of flight, and the attenuation factor stays at or above 0.5.

## 3.6 Braking Envelope and Hover Fallback

Ensemble disagreement is a heuristic: it can remain low on inputs that the frozen encoder maps into familiar regions of $\mathcal { Z } .$ . We therefore add a model-free bound derived from the depth channel. Let $d _ { t }$ be the smallest range measured in an image region R whose back-projection covers the volume swept by a sphere of radius $r _ { s } \geq r _ { \tau }$ moving along the commanded direction $\hat { u } _ { t } = v _ { t , 0 } / \bar { | | } v _ { t , 0 } \| _ { 2 }$ out to the sensor range, where $v _ { t , k }$ denotes the translational part of $\boldsymbol { a } _ { t , k }$ . If $\hat { u } _ { t }$ lies outside the field of view, or if the fraction of invalid depth pixels in $\mathcal { R } _ { t }$ exceeds a threshold, the speed is limited to a constant $v _ { \mathrm { b l i n d } }$

Braking profile. When the fallback is engaged at speed $v _ { 0 }$ (taken here with zero initial acceleration), the executor commands the velocity profile that minimizes the integrated squared jerk $\int _ { 0 } ^ { T _ { b } } \| \ddot { v } ( \tau ) \| _ { 2 } ^ { 2 }$ dτ subject to $v ( 0 ) = v _ { 0 }$ $\dot { v } ( 0 ) = 0 , v ( T _ { b } ) = 0$ , and $\dot { v } ( T _ { b } ) = 0$ , with the stopping position left free. The Euler–Lagrange condition $\boldsymbol { v } ^ { ( 4 ) } = 0$ makes the minimizer cubic, and the boundary conditions give

$$
v _ { \mathrm { b r k } } ( \tau ) = v _ { 0 } \left( 1 - 3 \Big ( \frac { \tau } { T _ { b } } \Big ) ^ { 2 } + 2 \Big ( \frac { \tau } { T _ { b } } \Big ) ^ { 3 } \right) , \qquad T _ { b } = \frac { 3 \| v _ { 0 } \| _ { 2 } } { 2 a _ { \mathrm { m a x } } } ,\tag{26}
$$

whose peak deceleration, attained at $\tau = T _ { b } / 2$ , equals $1 . 5 \| v _ { 0 } \| _ { 2 } / T _ { b } = a _ { \mathrm { m a x } }$ and whose stopping distance is $\begin{array} { r } { \int _ { 0 } ^ { T _ { b } } \left\| v _ { \mathrm { b r k } } \right\| _ { 2 } \mathrm { d } \tau = \left\| v _ { 0 } \right\| _ { 2 } T _ { b } / 2 = 3 \left\| v _ { 0 } \right\| _ { 2 } ^ { 2 } / ( 4 a _ { \mathrm { m a x } } ) } \end{array}$ . After the stop, the executor holds the stopping point with a position setpoint, and yaw scanning is permitted so that new observations can lower σ¯ and release the vehicle.

Speed envelope. Let $T _ { r }$ upper-bound the time from the capture of $I _ { t }$ to the onset of braking if braking becomes necessary; it comprises the worst-case end-to-end latency $T _ { \mathrm { e 2 e } } ^ { \mathrm { m a x } }$ , the chunk validity $H _ { t } \Delta$ of Eq. (25), and the velocitytracking lag $T _ { \mathrm { t r k } }$ of the flight controller. With depth error bound $\epsilon _ { d } .$ , define

$$
v _ { \mathrm { s a f e } } ( d ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 a _ { \mathrm { m a x } } } { 3 } \left( - T _ { r } + \sqrt { T _ { r } ^ { 2 } + \frac { 3 ( d - \epsilon _ { d } - r _ { s } ) } { a _ { \mathrm { m a x } } } } \right) , } & { d > \epsilon _ { d } + r _ { s } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{27}
$$

The commanded translational velocity of every knot is scaled by the common factor

$$
s _ { t } = \operatorname* { m i n } \left\{ \alpha ( \bar { \sigma } _ { t + 1 } ) , \ \frac { v _ { \mathrm { s a f e } } ( d _ { t } ) } { \operatorname* { m a x } \big ( \operatorname* { m a x } _ { k } \| v _ { t , k } \| _ { 2 } , \epsilon _ { v } \big ) } \right\} , \qquad a _ { t , k } ^ { \mathrm { c m d } } = \big ( s _ { t } v _ { t , k } , \ \dot { \psi } _ { t , k } \big ) ,\tag{28}
$$

where $\epsilon _ { v } > 0$ guards the division; scaling preserves the direction of every knot.

Assumption 3.1 (Braking envelope). Over the interval from $\tau _ { t }$ to the end of a possible braking manoeuvre: (i) the obstacles intersecting the swept volume are static; (ii) every range measured in $\mathcal { R } _ { t }$ exceeds the true range by at most $\epsilon _ { d } ;$ (iii) the vehicle moves along $\hat { u } _ { t }$ with speed at most $v _ { \mathrm { s a f e } } ( d _ { t } )$ until braking begins, and braking begins within $T _ { r }$ of $\tau _ { t }$ with zero acceleration; (iv) the flight controller tracks Eq. (26), which requires a deceleration of at most $a _ { \mathrm { m a x } } .$

Proposition 3.2 (Stop within measured free space). Under Assumption 3.1, the vehicle comes to rest without its center entering the r<sub>s</sub>-neighbourhood of any obstacle point imaged in $\mathcal { R } _ { t }$

Proof. Let $v \ \leq \ v _ { \mathrm { s a f e } } ( d _ { t } )$ be the speed during the reaction interval. The distance travelled before braking is at most $v T _ { r }$ and the braking distance is $3 v ^ { 2 } / ( 4 a _ { \mathrm { m a x } } )$ by Eq. (26). For $d _ { t } > \epsilon _ { d } + r _ { s } , v _ { \mathrm { s a f e } } ( d _ { t } )$ is the positive root of $\begin{array} { r } { \frac { 3 } { 4 a _ { \mathrm { m a x } } } v ^ { 2 } + T _ { r } v - ( d _ { t } - \epsilon _ { d } - r _ { s } ) = 0 } \end{array}$ , and the left-hand side is increasing in $v \geq 0$ , so $v T _ { r } + 3 v ^ { 2 } / ( 4 a _ { \operatorname* { m a x } } ) \leq d _ { t } - \epsilon _ { d } - r _ { s }$ By (ii), the true range of every obstacle point in $\mathcal { R } _ { t }$ is at least $d _ { t } - \epsilon _ { d } ;$ the total travel therefore ends at least $r _ { s }$ short of it. For $d _ { t } \leq \epsilon _ { d } + r _ { s } , v _ { \mathrm { s a f e } } = 0$ and the vehicle is commanded to hold.

Remark 3.1. The guarantee is conditional. It does not cover dynamic obstacles, which motivate fast replanning and the uncertainty signal, nor obstacles outside the field of view, which motivate $v _ { \mathrm { b l i n d } }$ . If braking starts with acceleration $a _ { 0 }$ along $\hat { u } _ { t } ,$ , the cubic profile matching $\dot { v } ( 0 ) = a _ { 0 }$ stops after $\| v _ { 0 } \| _ { 2 } T _ { b } / 2 + a _ { 0 } T _ { b } ^ { 2 } / 1 2$ , and the excess can be absorbed into $r _ { s } .$ . For illustration, with $a _ { \mathrm { m a x } } = 5 \mathrm { m } / \mathrm { s } ^ { 2 } , T _ { r } = 0 . 4 \mathrm { s } .$ , and $d _ { t } - \epsilon _ { d } - r _ { s } = 6 \bf { m }$ , Eq. (27) gives $v _ { \mathrm { s a f e } } \approx 5 . 1$ m/s. The envelope is a simplified braking-feasibility constraint in the spirit of control barrier functions [24].

Remark 3.2 (Evaluated configuration). The evaluated system uses $r _ { s } = 0 . 5 \mathrm { { m } , } a _ { \mathrm { { m a x } } } = 5 \mathrm { { m } / \mathrm { { s } ^ { 2 } , } \epsilon _ { d } = 0 . 2 \mathrm { { m } , } v _ { \mathrm { { b l i n d } } } = }$ 1 m/s and $T _ { r } = 0 . 0 6 + 0 . 0 2 \ : \dot { H _ { t } } + 0 . 1 5 \ : s$ , which is at most 0.53 s. This reaction time is shorter than the response identified in simulation, a dead time of $0 . 2 4 \mathrm { s }$ followed by a first-order lag of 0.5 s (Section $4 . 6 ) _ { ; }$ , so Assumption 3.1(iii)–(iv) does not hold for the evaluated system, Proposition 3.2 does not apply, and we treat the envelope as a heuristic. The envelope also checks only the current frame; obstacles that have left the field of view are handled by the runtime shield of Section 3.9.

## 3.7 Receding-Horizon Control Algorithm

Algorithm 1 summarizes the two loops of Aero-VPA. The inference loop runs once per camera frame whenever the previous decision has finished; the executor runs at a fixed rate $f _ { c } = 1 0 0$ Hz and evaluates the newest valid chunk at its chunk-local time $\tau - \tau _ { t }$ . Because the chunk is indexed by elapsed time since capture rather than by arrival, the knots that correspond to the inference and transport latency are skipped automatically, an approach related to the frozen prefix of real-time chunking [40].

The sequential structure of a decision step is that of [12]: the chain $I _ { t }  z _ { t }  u _ { t }  ( \hat { z } _ { t + 1 } , \sigma _ { t + 1 } )  e _ { t }  A ^ { ( \rho _ { 1 } ) } $ $\cdots \to A ^ { ( \rho _ { K } ) }$ contains one evaluation each of $E _ { \psi } , \pi _ { \phi } ^ { h }$ , and $P _ { \omega }$ (whose heads are independent and can be batched) and $K$ evaluations of $v _ { \theta }$ . The sequential depth is therefore $D _ { \mathrm { V P A } } = K + 3 = 5$ for $K = 2$ , independent of H and $d _ { a }$ , and the network time of a decision step is

$$
T _ { \mathrm { V P A } } = t _ { E } + t _ { h } + t _ { P } + K t _ { v } ( H ) ,\tag{29}
$$

where $t _ { E } , t _ { h } , t _ { P }$ , and $t _ { v } ( H )$ are the wall-clock times of the encoder, selector, predictor, and one solver evaluation. The arithmetic work of $t _ { v } ( H )$ grows with $H d _ { a } ;$ ; approximate independence of H holds only while one solver evaluation fits within the available parallelism, which on an Apple M4 was the case up to $H = 6 4 [ 1 2 ]$ and must be re-measured on the target hardware (Section 5.2). The safety computations of Eqs. (22)–(28) add only vector and image-region reduction and no network evaluation. In version $^ { 2 , }$ the runtime shield of Section 3.9 filters every setpoint of the executor before it is sent to PX4, and the setpoint it dispatches is the ${ { \bar { a } } _ { t } }$ of the next decision.

Algorithm 1 Aero-VPA asynchronous receding-horizon control with safety filter   
Input: frozen $E _ { \psi } , P _ { \omega } , \bar { \gamma } ^ { \star } , \bar { \mu } _ { Z } ;$ trained $\pi _ { \phi } ^ { h } ,$ v ; mission $G ; c _ { \mathrm { t e x t } }$ (computed once); $K = 2$   
Inference loop (event-driven, one decision per cameraframe)   
1: procedure DECIDE $( I _ { t } , S _ { t } , \tau _ { t } )$   
2: $z _ { t } \gets E _ { \psi } ( I _ { t } ) ;$ update m<sub>t</sub> by Eq. (14); compute $g _ { t }$ by Eq. (3)   
3: $u _ { t } \gets$ arg m $\operatorname { a x } _ { u } \pi _ { \phi } ^ { h } ( u \mid z _ { t } , g _ { t } , S _ { t } , c _ { \mathrm { t e x t } } )$ ▷ Eq. (12)   
4: $\{ \hat { z } _ { t + 1 } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { e } }  P _ { \omega } ( z _ { t } ,$ Embed $( u _ { t } ) , \bar { a } _ { t } ) ; \hat { z } _ { t + 1 } \gets$ ensemble mean ▷ Eq. (7)   
5: compute $\sigma _ { t + 1 }$ and $\bar { \sigma } _ { t + 1 }$ ▷ Eqs. (22), (23)   
6: if $\bar { \sigma } _ { t + 1 } \geq \sigma _ { \mathrm { c r i t } }$ then   
7: $u _ { t } \gets$ BRAKEHOVER; REQUESTFALLBACK and return   
8: end if   
9: update $\xi _ { t }$ by Eq. (20); form $e _ { t }$ by Eq. (15)   
10: $\bar { A }  \xi _ { t }$   
11: for $k = 0 , \ldots , K - 1$ do ▷ Eq. (18)   
12: $\begin{array} { r } { A \gets A + \frac { 1 } { K } v _ { \theta } ( A , k / K \mid e _ { t } ) } \end{array}$   
13: end for   
14: $\hat { A } _ { t } \gets \Pi _ { \mathcal { A } _ { 0 } }$ (denormalize(A))   
15: $d _ { t } \gets$ minimum valid range in $\mathcal { R } _ { t } \mathrm { ; }$ compute $s _ { t }$ by Eq. (28); $H _ { t }$ by Eq. (25)   
16: publish chunk $\mathcal { C } _ { t } = ( \tau _ { t } , \hat { \psi } _ { t } , \{ ( s _ { t } v _ { t , k } , \dot { \psi } _ { t , k } ) \} _ { k = 0 } ^ { H - 1 } , H _ { t } )$ to the buffer   
17: end procedure   
Executor (fixed rate $f _ { c } = 1 0 0 H z )$   
18: procedure TICK(τ)   
19: $c \gets$ newest chunk in buffer; $\Delta \tau  \tau - \tau c$   
20: if fallback requested or no chunk or $\Delta \tau > H _ { \cal { C } } \Delta$ or odometry stale then   
21: $( v _ { \mathrm { r e f } } ^ { \mathcal { W } } , \dot { v } _ { \mathrm { r e f } } ^ { \mathcal { W } } )$ ← braking profile Eq. (26), then hold position   
22: else   
23: interpolate knots at $\Delta \tau \left( \mathrm { E q . } \left( 3 9 \right) \right)$ ; blend with previous chunk (Eq. (41)); rate-limit (Eq. (40))   
24: rotate to NED: $v _ { \mathrm { r e f } } ^ { \mathcal { W } }  { \bf R } _ { z } ( \hat { \psi } _ { \mathcal { C } } ) v _ { \mathrm { r e f } } ^ { \mathcal { H } }$ ▷ Eq. (38)   
25: end if   
26: pass the setpoint through the runtime shield, which filters it or replaces it by its own (Algorithm 2)   
27: publish OffboardControlMode and TrajectorySetpoint ${ } _ { ; ( v _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \mathcal { W } } , \dot { \psi } _ { \mathrm { r e f } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { \psi } _ { \mathrm { r e f } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal { W } } } , \dot { v } _ { \mathrm { r e f } } ^ { \bar { \mathcal } { V } } , \dot \dot { v } _  \mathrm { r e f } ^ { \mathcal } { \xi } _ { \xi } _ { \xi \xi _ { \mathrm { r e f } } } ^ { \xi } _ { \xi } \xi _  \xi _ { \xi \xi } _ { \xi } { \xi } _ { \xi \xi } _ { \xi } _ { \xi \xi } _  \xi \xi _ { \xi } { \xi } _ { \xi \xi } _ { \xi } \xi _ { \xi \xi } _  \xi \xi \xi _ { } \xi \xi _ { \xi \xi } _ { \xi \xi \xi } _  \xi \xi \xi \xi _ { \xi } \xi \xi \xi _ { \xi \xi \xi } \xi _ \xi \xi \xi \xi _  \xi \xi \xi \xi \xi \xi \xi \xi _ \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi _ \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi \xi$   
28: a¯ ← dispatched setpoint, stored for the next decision   
29: end procedure

## 3.8 Training Procedure and DAgger Supervisor

Training follows the staged procedure of [12], adapted to flight. (i) Representation learning: $E _ { \psi }$ and $P _ { \omega }$ are trained with Eq. (8) on flight logs, then frozen, fixing $\bar { \gamma } ^ { \star }$ and $\bar { \mu } _ { Z }$ . (ii) Policy learning: with latents cached, the selector and solver minimize

$$
\mathcal { L } _ { \mathrm { p o l } } ( \phi , \theta ) = \mathcal { L } _ { \mathrm { C F M } } ( \theta ) + \lambda _ { h } \mathcal { L } _ { \mathrm { C E } } ( \phi ) ,\tag{30}
$$

with weight $\lambda _ { h } \ > \ 0$ , on expert chunks expressed in the frozen heading frame of their capture time. (iii) Dataset aggregation: because imitation from expert states alone suffers from covariate shift, Aero-VPA is rolled out in simulation under a mixture of its own and the expert’s commands, the privileged expert relabels the visited states, and stage (ii), optionally also stage (i), is repeated on the aggregated data, following DAgger [23]. The expert is a planner with access to the ground-truth occupied set $x ,$ in the manner of [5].

Evaluated training runs. In the evaluated system, one stage-(ii) run took 153,528 steps at batch size 1024, about 2 h 40 min on one NVIDIA L4 GPU. Inputs that a recipe withholds (Table 4) enter $\pi _ { \phi } ^ { h }$ and $e _ { t }$ at their training mean, while $P _ { \omega }$ always receives ${ { \bar { a } } _ { t } }$ . Stage (iii) was realized with the supervisor described next.

Expert and demonstrations. The expert plans with $\mathbf { A } ^ { * }$ on the ground-truth occupancy (1 m inflation) and a minimumsnap trajectory [3]. Demonstrations come from 200 training worlds (seeds 0–199), with the speed limit of each mission drawn from {3, 5, 7} m/s: about 5.5 million RGB-D frames. Another 360 expert flights on calibration worlds (seeds 1000–1011) serve validation and the calibration of $\sigma _ { \mathrm { c r i t } }$ . The label of knot k follows the expert’s tracking law from the path point $\tau ^ { \star }$ nearest the vehicle,

$$
\begin{array} { r } { a _ { k } ^ { \star } = w ( e ) \ v ^ { \star } ( \tau ^ { \star } + k \Delta ) + k _ { p } e _ { p } e ^ { - k _ { p } k \Delta } , \qquad k _ { p } = 1 . 5 \mathrm { s } ^ { - 1 } , } \end{array}\tag{31}
$$

![](images/7ab03df7b16b047c87c387401385935227dc9dac027465a856bc18ba2ebbd89a.jpg)  
Figure 2: Mode automaton of the runtime shield. In FLY the policy’s executor runs and the shield filters each setpoint. In BACKOFF, NUDGE and HOLD (orange) the executor is paused, and the shield’s own setpoint, passed through the same filters, is dispatched and fed back as ${ { \bar { a } } _ { t } }$ . The radius $r = 0 . 5 5$ m is the barrier’s zero; a backoff that ends without an active waypoint returns to FLY.

where $v ^ { \star }$ is the velocity of the expert’s reference trajectory, $e _ { p }$ the vector from the vehicle to the path point and $e = \| e _ { p } \|$ on the path $( e \le 0 . 3  { \mathrm { m } } ) w = 1$ , further off $w \to 0$ and the horizontal speed limit falls to 2 m/s at $e \geq 1 { \mathfrak { m } }$ , so that the expert rejoins its path first.

Gate and takeover. The policy flies until one of six triggers fires: an obstacle within 0.5 m + 0.06 m per m/s of speed of the straight look-ahead along the current velocity (0.3 s of reaction plus the braking distance, at least 1.5 m, sampled every 0.1 m); an altitude below 0.8 m; more than 1.5 m off the path; a heading more than $7 5 ^ { \circ }$ from the path; a missed waypoint; or $5 s$ without progress. The expert then takes over. In a soft takeover, which keeps recording frames, it chooses between flying on, slowing down along its path (a time-scaled reference) and a straight stop by the clearance predicted in a rollout of the delayed closed loop (0.24 s dead time, 0.5 s first-order lag). A hard takeover brakes, backs off and re-plans. Control returns to the policy after 2 s on track. Appendix B gives the details.

Rounds and recipes. Each round flies 201 missions over the training worlds, and the expert labels every visited state. In stage (ii), DAgger frames are weighted $\times 3$ relative to demonstrations. Round 1 was flown by the behaviourcloning policy BC, round 2 by $\mathrm { D } _ { 1 }$ and round 3 by D<sub>2</sub> (Table 6). Four round-3 recipes target the copycat shortcut of Section 6.2 (Table 4): $\mathrm { D } _ { 3 - \bar { a } }$ withholds ${ { \bar { a } } _ { t } }$ from $\pi _ { \phi } ^ { h }$ and $e _ { t } ; \mathbf { D } _ { 3 - \bar { a } , v }$ also withholds $v _ { t } ^ { \mathcal { H } } ; \bar { \mathsf { D } _ { 3 , \mathrm { S D } } }$ replaces $v _ { t } ^ { \mathcal { H } }$ and ${ { \bar { a } } _ { t } }$ by their training mean in 50% of training frames (state dropout); and $\mathrm { D } _ { 3 , \mathrm { K W } }$ weights each frame in proportion to min $\left( 1 + \Vert A _ { t } [ 0 ] - \bar { a } _ { t } \Vert / 0 . 1 , 1 0 \right)$ , the error of repeating the last command (keyframe weighting), with DAgger weight ×1.

## 3.9 Runtime Safety Shield

The runtime shield, added in version 2 after the evaluation of Section 6 exposed the limits of the learned safety layer, sits between the policy’s executor and the flight controller (Figure 1). It is deterministic, has no learned parameters and uses no map or privileged information: its inputs are the depth image, the odometry estimate (position $\hat { p } ,$ velocity vˆ, yaw $\hat { \psi } ,$ attitude ${ \hat { R } } ) .$ , the active waypoint and the decision timestamps. It runs at every executor tick.

Depth memory. For each new frame, the shield takes the nearest valid return $z _ { b } \in [ 0 . 2 , 8 ]$ m of every $4 \times 4$ pixel block b and back-projects it,

$$
\boldsymbol { x } _ { b } = \hat { p } _ { t } + \hat { R } _ { t } \left( z _ { b } \boldsymbol { r } _ { b } \right) ,\tag{32}
$$

where $r _ { b }$ is the ray through the block centre in the body frame, scaled to unit forward component (depth is measured along the optical axis). Points less than 0.25 m above the ground are dropped. The memory $\mathcal { M } _ { t }$ is the union of all points no older than $T _ { M } = 1 5$ s and within 10 m horizontally, voxelised at 0.15 m with the newest point kept per voxel, so that obstacles beside and behind the camera remain known after they leave the field of view. The barrier uses the local set

$$
\mathcal { N } _ { t } = \{ x \in \mathcal { M } _ { t } : \| x - \hat { p } _ { t } \| \leq 3 . 5 \mathrm { m } \} .\tag{33}
$$

Algorithm 2 One step of the runtime shield (every executor tick, 100 Hz). The memory M and the local set $\mathcal { N }$ are   
updated per frame (Eqs. (32)–(33)).   
Input: estimate $( \hat { p } , \hat { v } , \hat { \psi } ) ;$ active waypoint w; time t; time $t _ { f }$ of the last frame; breadcrumbs B   
1: add pˆ to B if it is ≥ 0.25 m from the last breadcrumb (not while backing off); update the progress toward w   
2: $c \gets \operatorname* { m i n } _ { x \in \mathcal { N } } \| x - \hat { p } \|$ ▷ clearance to the memory   
3: if mode = FLY and $c < r _ { \mathrm { h a r d } }$ then   
4: mode ← BACKOFF ▷ retrace at most 1 m   
5: else if mode = FLY and no 0.3 m of progress for $5 s$ then   
6: mode ← NUDGE   
7: end if   
8: if mode = BACKOFF then   
9: u ← FILTER $\mathrm { { ( 0 . 3 m / s } }$ toward the previous breadcrumb)   
10: if retraced 1 m or $c \ge r + 0 . 4$ m or $\Vert u \Vert < 0 . 0 7 5$ m/s then   
11: mode ← N   
12: else   
13: return u as an external setpoint ▷ becomes ${ { \bar { a } } _ { t } }$   
14: end if   
15: end if   
16: if mode = NUDGE then   
17: if the nudge is over then   
18: mode ← FLY   
19: else   
20: $\dot { \psi } \gets \mathrm { c l i p } ( 1 . 5 \Delta \psi , \pm 0 . 8 ) ; u \gets 0$ while $| \Delta \psi | > 3 6 . 5 ^ { \circ }$ , else 0.6 cos $( \Delta \psi )$ m/s toward w for 1 s   
21: return (FILTER(u), ψ<sup>˙</sup>) as an external setpoint ▷ becomes ${ { \bar { a } } _ { t } }$   
22: end if   
23: end if   
24: if $\dot { \mathrm { \Delta } t } - \dot { t } _ { f } > 0 . 3 \mathrm { s }$ then   
25: mode ← HOLD; return zero velocity as an external setpoint   
26: end if   
27: mode ← FLY; $( u , a _ { \mathrm { f f } } , \dot { \psi } )$ ← one executor tick of the policy   
28: $( u , a _ { \mathrm { f f } } ) \gets \mathrm { F I L T E R } ( u , a _ { \mathrm { f f } } )$ ; if it changed, write $( u , \dot { \psi } )$ back as ${ { \bar { a } } _ { t } }$   
29: return $( u , a _ { \mathrm { f f } } , \dot { \psi } )$   
30: function FILTER(u, a<sub>f</sub>)   
31: altitude floor; barrier: two sweeps of Eq. (35) over the 24 points of $\mathcal { N }$ nearest to $q = \hat { p } + 0 . 4$ ˆv; blind cap,   
Eq. (36)   
32: return $( u , a _ { \mathrm { f f } } )$   
33: end function

Barrier. Let $u \in \mathbb { R } ^ { 3 }$ be the commanded velocity, $a _ { \mathrm { f f } }$ its acceleration feed-forward, and $q = \hat { p } + \hat { v } T _ { \mathrm { l a } }$ the position $T _ { \mathrm { l a } } = 0 . 4 \ : \mathrm { s }$ ahead at the flown velocity, a horizon that covers the 0.24 s dead time. For each of the 24 points $x _ { i } \in \mathcal { N } _ { t }$ nearest to $q ,$ let $n _ { i } = ( x _ { i } - q ) / \left\| x _ { i } - q \right\|$ and $h _ { i } = \| x _ { i } - q \| - r$ with $r = 0 . 5 5 \mathrm { m }$ . The shield enforces

$$
n _ { i } ^ { \top } u \leq \beta ( h _ { i } ) , \qquad \beta ( h ) = \left\{ \operatorname* { m i n } \big ( k h , ~ \sqrt { 2 a _ { b } h } \big ) , \quad h > 0 , \right.\tag{34}
$$

with $k = 1 . 5 \mathrm { s } ^ { - 1 } , a _ { b } = 3 \mathrm { m } / \mathrm { s } ^ { 2 }$ and $v _ { \mathrm { p u s h } } = 0 . 5$ m/s, by two sequential projection sweeps over the 24 points: whenever $n _ { i } ^ { \top } u > \beta ( h _ { i } )$

$$
\begin{array} { r l r } { u \gets u - \left( n _ { i } ^ { \top } u - \beta ( h _ { i } ) \right) n _ { i } , } & { { } } & { a _ { \mathrm { f f } } \gets a _ { \mathrm { f f } } - \operatorname* { m a x } \left( 0 , n _ { i } ^ { \top } a _ { \mathrm { f f } } \right) n _ { i } , } \end{array}\tag{35}
$$

which removes the approaching components of the velocity and of the feed-forward and keeps tangential motion past the obstacle. With $h ( p ) = \| x - p \| - r$ and ${ \dot { p } } = u$ , the constraint reads $\dot { h } \ge - \beta ( h )$ , a control-barrier-function condition $[ 2 4 ] ;$ $\beta$ is continuous and non-decreasing with $\beta ( 0 ) = 0$ . The square-root term caps the approach speed at the speed from which a deceleration $a _ { b }$ stops within the distance $h ;$ the linear term is the smaller one for $h < 2 \dot { a } _ { b } / k ^ { 2 } \approx 2 . 7$ m; and for $h \leq 0$ the constraint pushes the vehicle away at up to $v _ { \mathrm { p u s h } }$ . The projection is sequential, not a quadratic program: it is neither optimal nor guaranteed to satisfy all constraints jointly.

Proposition 3.3 (Invariance under exact tracking). Let $x \in \mathbb { R } ^ { 3 }$ be a static point, and on an interval [0, T] let the vehicle position p be absolutely continuous with $p ( t ) \neq x$ and ${ \dot { p } } ( t ) = u ( t )$ , where $u ( t )$ satisfies $n ( t ) ^ { \top } u ( t ) \leq \beta ( h ( t ) )$ for $n = ( x - p ) / \| x - p \|$ and $h = \| x - p \| - r ( E q$ . (34) evaluated at $q = p )$ . Then (i) $i f h ( 0 ) \geq 0 , h ( t ) \geq h ( 0 ) e ^ { - k t } \geq 0$ for all $t \in [ 0 , T ] ,$ ; and (ii) $i f h ( 0 ) < 0 ,$ , h is strictly increasing while it is negative, with $\dot { h } \geq \operatorname* { m i n } ( k | h | , v _ { \mathrm { p u s h } } )$ almost everywhere.

Proof. Wherever p $\begin{array} { r } { \neq \boldsymbol { x } , \frac { d } { d t } \left. \boldsymbol { x } - \boldsymbol { p } \right. = - \boldsymbol { n } ^ { \top } \dot { \boldsymbol { p } } , } \end{array}$ so almost everywhere $\dot { h } = - \boldsymbol { n } ^ { \top } \boldsymbol { u } \ge - \beta ( h )$ . For $h \geq 0 , \beta ( h ) \leq k h$ hence $\dot { h } \geq - k h .$ ; for $h < 0 , \beta ( h ) = \operatorname* { m a x } ( k h , - v _ { \mathrm { p u s h } } ) < 0$ , hence $\dot { h } \geq$ min $( k | h | , v _ { \mathrm { p u s h } } ) > 0$ , which is (ii). For (i), h cannot become negative: at a first time $t _ { 0 }$ with $h ( t _ { 0 } ) = 0$ after which $h < 0 ,$ h would have to decrease on an interval where (ii) makes it increase. On [0, T] we therefore have $h \geq 0$ and $\begin{array} { r } { \frac { d } { d t } \left( h e ^ { k t } \right) = e ^ { k t } ( \dot { h } + k h ) \ge 0 } \end{array}$ almost everywhere, so $h ( t ) e ^ { k t } \geq h ( 0 )$

Remark 3.3. Proposition 3.3 concerns one remembered point, exact velocity tracking and the constraint evaluated at the current position. The implemented filter departs from it in four ways, so the shield carries no safety guarantee. (1) It evaluates the constraint at the predicted position $q ,$ a heuristic for the dead time, while the vehicle tracks u with a 0.24 s dead time and a 0.5 s lag. (2) When several constraints are active, the sequential projection may leave some of them violated, and the blind cap (below), applied after the projection, scales the horizontal command without re-checking them. (3) Only remembered points are constrained: never-observed obstacles, points forgotten after 15 s or beyond 10 m, points outside the 3.5 m local set or beyond the 24 nearest, and depth errors are not. (4) Obstacles are assumed static.

Blind cap, altitude floor and hold. After the barrier, the horizontal command is capped where the camera cannot see:

$$
u _ { x y } \gets \frac { v _ { b } } { \lVert u _ { x y } \rVert } u _ { x y } \quad \mathrm { i f } \quad \lVert u _ { x y } \rVert > v _ { b } \mathrm { a n d } \ \left| \mathrm { w r a p } \left( \mathrm { a t a n 2 } ( u _ { y } , u _ { x } ) - \hat { \psi } \right) \right| > \frac { 1 } { 2 } \mathrm { H F O V } ,\tag{36}
$$

with $v _ { b } = 0 . 7$ m/s and ${ \textstyle \frac { 1 } { 2 } } \mathrm { H F O V } = 3 6 . 5 ^ { \circ }$ , unless $u _ { x y }$ retraces the flown path (it points within $3 5 ^ { \circ }$ of a breadcrumb 0.3–2.5 m away); the feed-forward is scaled alike. Because the cap follows the barrier, a slide around a trunk that leaves the view is capped as well. Before the barrier, an altitude floor sets the upward speed to at least 0.5 m/s below 0.8 m of height and removes downward feed-forward. If no decision has arrived for $0 . 3 \mathrm { s } ,$ the shield commands zero velocity until the next one. An uncertainty cap (speed limit while $\bar { \sigma } > \sigma _ { \mathrm { l o } } )$ exists but is disabled $( v _ { \sigma } = \infty ) \colon$ in the first shield version it held the vehicle at 1 m/s for 18.0 s per mission and turned flights into timeouts. The filter order is altitude floor, barrier, blind cap.

Recovery: backoff and nudge. The shield drops a breadcrumb every 0.25 m along the flown path, which is known free space. When the nearest remembered point is closer than $r _ { \mathrm { h a r d } } = 0 . 4 5 \mathrm { m }$ , it backs off: it retraces the breadcrumbs at 0.3 m/s, with the barrier, the blind cap and the altitude floor still applied, until it has retraced 1 m, the clearance to the memory reaches $r + 0 . 4  { \mathrm { m } }$ , or the filtered command falls below 0.075 m/s because the way back is blocked. It then nudges the vehicle toward the active waypoint: it yaws toward the waypoint at $\dot { \psi } = \mathrm { c l i p } ( 1 . 5 \Delta \psi , \pm 0 . 8 \mathrm { r a d / s } )$ , where $\Delta \psi$ is the bearing of the waypoint relative to the heading, without translating while the waypoint is outside the field of view, and then moves toward it at $0 . 6 \cos ( \Delta \psi )$ m/s for 1 s, filtered like any other command. A nudge also follows a stall: 5 s without 0.3 m of progress toward the waypoint. The nudge ensures that a policy that copies its last command does not remain at rest after an intervention.

Coupling to the policy. While the shield only filters (mode FLY), the filtered setpoint is dispatched and written back as ${ { \bar { a } } _ { t } }$ , so the policy conditions on what was actually sent. While it flies itself (BACKOFF, NUDGE, HOLD), the executor is paused, the shield’s setpoint is dispatched as an external setpoint, and it becomes ${ { \bar { a } } _ { t } }$ . Figure 2 shows the mode automaton and Algorithm 2 one shield step.

## 4 Asynchronous System Integration and Edge Deployment

## 4.1 Compute and Memory Budget on the Jetson Orin Nano

The target module is the Jetson Orin Nano 8 GB with a 1024-core Ampere GPU, 8 GB of LPDDR5 shared by CPU and GPU, and 68 GB/s of memory bandwidth in its original configuration (102 GB/s in the JetPack 6.2 Super configurations) [11]; we use the 15 W power mode. Unlike the Orin NX and AGX Orin, the Orin Nano has no Deep Learning Accelerator, so all networks run on the GPU as TensorRT engines. Table 1 lists the configuration and its parameter count, computed analytically from the layer dimensions. Since every evaluation at batch size one reads the weights of the evaluated network, a decision step streams at least

$$
M _ { \mathrm { d e c } } = b \left( | W _ { E } | + | W _ { h } | + | W _ { P } | + K | W _ { v } | \right)\tag{37}
$$

Table 1: Aero-VPA network configuration and parameter budget (analytic counts for $H = 1 6 , d = 2 5 6 , N _ { u } = 6 ; \mathrm { F P 1 6 }$ weights). The frozen text encoder is evaluated once per mission and lies outside the per-decision dependency chain.
<table><tr><td>Module</td><td>Configuration</td><td>Params</td><td>Evals./decision</td><td>FP16 weights</td></tr><tr><td>Encoder  $E _ { \psi }$ </td><td>ViT, 4× 128× 160 input, patch 16, width 384, depth 6, 6 heads</td><td>11.17M</td><td>1</td><td>22.3 MB</td></tr><tr><td>Selector  $\pi _ { \phi } ^ { h }$ </td><td> $\mathrm { M L P 7 8 5 {  } 5 1 2 {  } 5 1 2 {  } 6 }$ </td><td>0.67M</td><td>1</td><td>1.3 MB</td></tr><tr><td>Predictor  $\scriptstyle { \dot { P } } _ { \omega }$ </td><td> $5 { \mathrm { ~ h e a d s , ~ M L P ~ 2 9 2 } } \to 5 1 2 \to 5 1 2 \to 2 5 6$ </td><td>2.72M</td><td>1 (heads batched)</td><td>5.4MB</td></tr><tr><td>Solver vθ</td><td>Transformer, width 256, depth 4, 4 heads,  $H { + } 1 = 1 7 ~ \mathrm { t o k e n s }$ </td><td>3.57M</td><td>K = 2</td><td>7.1 MB</td></tr><tr><td>Trainable total</td><td></td><td>18.13M</td><td>K+3 = 5 sequential</td><td>36.3 MB</td></tr><tr><td>Text encoder (frozen)</td><td>CLIP ViT-B/32 text tower</td><td>63.4M</td><td>once per mission</td><td></td></tr></table>

bytes, where $| W . |$ are the parameter counts of the four networks and b = 2 for FP16. For the configuration of Table 1, $\dot { M } _ { \mathrm { d e c } } = 4 3 . 4 \dot { \bf M } \dot { \bf B }$ , a lower bound of 0.64 ms at 68 GB/s, compared with 88 ms for a single pass of the 3B-parameter decoder discussed in Section 1. At this scale the latency is expected to be dominated by kernel launch overhead and arithmetic (about 1.0 giga multiply-accumulates per decision) rather than by weight streaming, which is why the latency targets of Section 5.5 are hypotheses to be measured rather than consequences of Eq. (37). The released policy checkpoints are single files of 72.6–72.7 MB, under 1% of the module’s 8 GB of memory; their timing on the module has not been measured (Section 4.6).

## 4.2 Hierarchical Software Stack

The high level runs on the Jetson under ROS 2 [66]. A camera driver publishes time-stamped, hardware-synchronized RGB and aligned depth frames; a preprocessing node resizes and normalizes them on the GPU; the inference node executes $E _ { \psi } , \bar { \pi } _ { \phi } ^ { h } , P _ { \omega } ,$ , and $v _ { \theta }$ as TensorRT FP16 engines on a single CUDA stream with preallocated buffers, so that no memory is allocated on the critical path; and the executor node runs on a real-time-prioritized timer at $f _ { c } = 1 0 0 \mathrm { H z } .$ The two loops communicate only through a lock-free single-slot chunk buffer. Each chunk carries the capture stamp $\tau _ { t } .$ which the executor uses for latency compensation, so the decision rate $f _ { \mathrm { d e c } } = \operatorname* { m i n } \{ f _ { \mathrm { c a m } } , 1 / T _ { \mathrm { i n f } } \}$ , with camera rate $f _ { \mathrm { c a m } }$ and inference time $T _ { \mathrm { i n f } }$ , can vary without affecting the setpoint rate. A watchdog monitors chunk age, odometry age, and link status and requests the fallback when any of them exceeds its bound.

## 4.3 PX4 Interface and Minimum-Acceleration Setpoint Interpolation

The low level is a PX4 flight controller [52] (e.g., an FMUv6X-class Pixhawk) connected to the Jetson through the uXRCE-DDS bridge, in which a client on PX4 exchanges uORB topics with an agent on the companion computer over a serial link (e.g., 921 600 baud) or a dedicated Ethernet link [27]. In offboard mode, PX4 requires a continuous proof-of-life stream of at least 2 Hz, provided by OffboardControlMode, and triggers its configured failsafe if the stream is interrupted for longer than COM\_OF\_LOSS\_T (default 1.0 s) [20]; we publish both this message and TrajectorySetpoint at $f _ { c } = 1 \bar { 0 } 0 \mathrm { H z } ,$ so that PX4’s failsafe acts only as the last of three layers behind the executor fallback and the watchdog. TrajectorySetpoint is expressed in the local NED frame, with unused fields set to NaN, so the heading-frame velocities of a chunk are rotated with the yaw frozen at its capture time,

$$
v _ { \mathrm { r e f } } ^ { \mathcal { W } } ( \tau ) = \mathbf { R } _ { z } ( \hat { \psi } _ { t } ) v _ { \mathrm { r e f } } ^ { \mathcal { H } _ { t } } ( \tau ) , \qquad \dot { \psi } _ { \mathrm { r e f } } ( \tau ) = \dot { \psi } _ { \mathrm { r e f } } ^ { \mathcal { H } _ { t } } ( \tau ) ,\tag{38}
$$

where the yaw rate maps directly onto the NED yawspeed field because the down axes of $\mathcal { H } _ { t }$ and W coincide.

Interpolation. The executor runs faster than the knot rate $1 / \Delta$ and also supplies an acceleration feedforward, which PX4 expects to be kinematically consistent with the velocity setpoint. Among all absolutely continuous velocity references that pass through the knots, $v _ { \mathrm { r e f } } ( k \Delta ) = v _ { t , k }$ , the one minimizing the integrated squared acceleration $\begin{array} { r } { \int \| \dot { v } _ { \mathrm { r e f } } ( \tau ) \| _ { 2 } ^ { 2 } } \end{array}$ dτ satisfies $\ddot { v } _ { \mathrm { r e f } } = 0$ between knots by the Euler–Lagrange equation and is therefore piecewise linear:

$$
v _ { \mathrm { r e f } } ( \tau ) = v _ { t , k } + \frac { \tau - k \Delta } { \Delta } \big ( v _ { t , k + 1 } - v _ { t , k } \big ) , \qquad \dot { v } _ { \mathrm { r e f } } ( \tau ) = \frac { v _ { t , k + 1 } - v _ { t , k } } { \Delta } , \qquad \tau \in [ k \Delta , ( k + 1 ) \Delta ) ,\tag{39}
$$

with τ measured from the capture time $\tau _ { t } ;$ the yaw rate is interpolated in the same way. The acceleration feedforward is piecewise constant and exactly consistent with the velocity reference. To keep it feasible, consecutive knots are rate-limited before interpolation,

$$
\begin{array} { r } { v _ { t , k + 1 }  v _ { t , k } + \mathrm { c l i p } _ { a _ { \mathrm { m a x } } \Delta } ( v _ { t , k + 1 } - v _ { t , k } ) , \qquad \mathrm { c l i p } _ { c } ( y ) = y \mathrm { m i n } \big ( 1 , c / \| y \| _ { 2 } \big ) , } \end{array}\tag{40}
$$

which bounds $\lVert \dot { v } _ { \mathrm { r e f } } \rVert _ { 2 }$ by a<sub>max</sub>.

Chunk blending. When a new chunk $\mathcal { C } _ { t ^ { \prime } }$ replaces $\mathcal { C } _ { t }$ at time $\tau _ { s }$ , the executor cross-fades the two references over a window $T _ { \mathrm { b l } }$

$$
v _ { \mathrm { r e f } } ( \tau ) = \lambda ( \tau ) v _ { \mathrm { r e f } } ^ { ( t ^ { \prime } ) } ( \tau ) + \left( 1 - \lambda ( \tau ) \right) v _ { \mathrm { r e f } } ^ { ( t ) } ( \tau ) , \qquad \lambda ( \tau ) = \mathrm { m i n } \Big ( 1 , \frac { \tau - \tau _ { s } } { T _ { \mathrm { b l } } } \Big ) ,\tag{41}
$$

where both references are expressed in W and evaluated at their own chunk-local times. Since $\lambda ( \tau _ { s } ) = 0$ and both references are continuous, $v _ { \mathrm { r e f } }$ is continuous at $\tau _ { s } ;$ if $\mathcal { C } _ { t }$ expires within the window, its last knot is held. The acceleration feedforward of the blended reference is the corresponding derivative of Eq. (41). Inpainting-based real-time chunking [40] is an alternative evaluated in ablation A8.

Link budget. A TrajectorySetpoint carries 64 bytes of payload. Assuming at most 128 bytes per serialized message including framing, the setpoint stream and an odometry stream of equal rate occupy about 28% of the 92 160 B/s capacity of a 921 600-baud 8N1 link at 100 Hz each; the dedicated Ethernet link removes this constraint.

## 4.4 Latency Budget

We decompose the end-to-end latency from photon capture to the setpoint received by PX4 as

$$
T _ { \mathrm { e 2 e } } = t _ { \mathrm { c a p } } + t _ { \mathrm { p r e } } + \underbrace { t _ { E } + t _ { h } + t _ { P } + K t _ { v } ( H ) } _ { T _ { \mathrm { V P A } } } + t _ { \mathrm { s a f e } } + t _ { \mathrm { e x e c } } + t _ { \mathrm { d d s } } ,\tag{42}
$$

where $t _ { \mathrm { c a p } }$ is the exposure, readout, and driver delay, $t _ { \mathrm { p r e } }$ the preprocessing time, $t _ { \mathrm { s a f e } }$ the time of the safety computations, $t _ { \mathrm { e x e c } } \leq 1 / f _ { c }$ the wait until the next executor tick, and $t _ { \mathrm { d d s } }$ <sub>s</sub> the transport time to the PX4 client. We do not adopt a numerical value for $t _ { \mathrm { d d s } }$ from the literature; every term of Eq. (42) is measured in the HITL protocol (Section 5.2), and the measured worst case $T _ { \mathrm { e 2 e } } ^ { \mathrm { m a x } }$ enters $T _ { r }$ in Eq. (27).

## 4.5 Cross-Embodiment Adaptation to Quadrupeds

The architecture transfers to legged robots through the same separation of time scales. For a quadruped (Legged-VPA), the chunk consists of planar base twists, elements of se(2),

$$
A _ { t } ^ { \mathrm { l e g } } = [ v _ { x } , v _ { y } , \omega _ { z } ] _ { t : t + H - 1 } \in \mathbb { R } ^ { H \times 3 } ,\tag{43}
$$

where $v _ { x } , v _ { y }$ are base-frame linear velocities and $\omega _ { z }$ is the yaw rate. Because leg phase and contact state make single-step proprioception non-Markovian, the proprioceptive input is a history of $L = 3 0$ steps of joint positions $q _ { j } \in \mathbb { R } ^ { 1 2 }$ , joint velocities $\dot { q } _ { j } \in \mathbb { R } ^ { 1 2 }$ , and projected gravity ${ \bf \dot { \varphi } } _ { n } ^ { B } \in \mathbb { R } ^ { 3 }$

$$
S _ { t } ^ { \mathrm { l e g } } = \left( q _ { j , t - \ell } , \dot { q } _ { j , t - \ell } , n _ { t - \ell } ^ { \mathcal { B } } \right) _ { \ell = 0 } ^ { L - 1 } \in \mathbb { R } ^ { L ( 1 2 + 1 2 + 3 ) } = \mathbb { R } ^ { 8 1 0 } ,\tag{44}
$$

which a linear layer projects before it enters $e _ { t }$ . The executor samples Eq. (39) at 50 Hz and passes the twist to a low-level locomotion policy trained by reinforcement learning, such as Walk These Ways [51], whose joint targets are tracked by a proportional-derivative controller at 200 Hz. The safety layer carries over with α(¯σ) applied to $( v _ { x } , v _ { y } )$ , the braking envelope recomputed with the deceleration limit of the gait, and BRAKEHOVER replaced by a stop-and-stand primitive. Whether $E _ { \psi }$ and $P _ { \omega }$ can be shared across aerial and legged embodiments is an open question: it requires pretraining data from both, and camera height, ego-motion, and motion blur differ substantially between them. We treat sharing as a hypothesis, whereas $v _ { \theta }$ and the executor are embodiment-specific by construction.

## 4.6 Implementation Used in the Evaluation

The evaluation of Section 6 does not use the Jetson stack of Sections 4.1–4.3. The networks run in PyTorch on an NVIDIA L4 GPU of a cloud virtual machine, and one ROS 2 node per simulated vehicle runs the inference loop, the executor and the runtime shield against PX4 SITL over uXRCE-DDS [27]. Each camera frame triggers one decision, a chain of $K + 3 = 5$ sequential network evaluations. A 100 Hz executor evaluates the newest chunk with latency compensation and a cross-fade of $T _ { \mathrm { b l } } = 0 . 0 4 \mathrm { s } \left( \mathrm { E q . } \left( 4 1 \right) \right)$ and sends offboard velocity setpoints with acceleration feed-forward; the shield filters every setpoint. Unlike the two-rate design of Section 4.2, decisions and executor ticks share one single-threaded ROS 2 executor, so a slow decision delays the setpoints. With the shield, the 95th percentile of a decision’s compute time was 32.0–33.1 ms and that of its age from capture to commit 43.5–44.6 ms (medians over missions), and the longest setpoint gap of a mission had a median of 0.10 s and a maximum of 0.13–0.18 s. These values were measured on the L4 and do not bear on Hypothesis 1, which concerns the Jetson. A fit of the simulated vehicle’s horizontal response to its setpoints gave a dead time of 0.24 s, a first-order lag of 0.5 s and a feed-forward share of 1.0 (RMSE 0.273 m/s).

## 5 Experimental Design and Benchmark Protocol

Status. Version 1 of this preprint fixed this protocol before any experiment was executed: it specifies the environments, data, hardware, baselines, metrics, hypotheses, and statistical analysis, so that results can be judged against criteria fixed in advance. Version 2 executes a first part of it in software-in-the-loop simulation (Section 6) and lists every deviation in Section 8.3; none of Hypotheses 1–5 has been tested yet. The rest of this section is unchanged from version 1 and remains the plan for the experiments not yet executed. Target values in Tables 2 and 3 are acceptance criteria, not results.

## 5.1 Simulation Setup

Platforms. The primary environment is PX4 SITL coupled to Gazebo [67] in its Harmonic long-term-support release (Gazebo Garden reached end of life in November 2024 and is not supported by current PX4 releases), using the PX4 quadrotor model with a front-facing depth camera (gz\_x500\_depth) augmented with an aligned RGB stream. For photorealistic tests of visual robustness, the same vehicle is simulated in NVIDIA Isaac Sim [68] with PX4 SITL attached through a bridge such as the Pegasus Simulator [69]. A single PX4 release is pinned for all runs and recorded with the results.

Environment classes. Worlds are generated procedurally from integer seeds. (E1) Dense forest: trees placed by Poisson-disc sampling at three densities, with trunk diameters and canopy heights drawn from fixed distributions. (E2) Collapsed indoor structure: rooms and corridors with rubble, tilted slabs, hanging cables, and openings whose minimum width ranges from 1.5 to 3 times the vehicle diameter, under low and non-uniform illumination. (E3) Dynamic clutter: E1 and E2 with moving obstacles crossing the nominal path at speeds of 0.5–2 m/s. Missions consist of 3–6 waypoints with straight-line spacing of 10–40 m. Each episode runs at a commanded speed limit $\bar { v } _ { h } \in \{ 3 , 5 , 7 \}$ m/s.

Perturbations. Physical perturbations comprise steady wind and gusts up to 5 m/s injected mid-flight through the simulator’s wind model. Visual perturbations comprise illumination changes, unseen textures, moving background elements, motion blur, and depth noise and dropout following a range-dependent stereo error model. Training uses domain randomization [70] over a subset of these factors; held-out factors define the out-of-distribution (OOD) test set.

Splits. Training, validation, and test worlds use disjoint seed ranges. The test set comprises 10 held-out worlds per environment class and difficulty level, an in-distribution (ID) condition, and an OOD condition with held-out visual factors and wind.

Data. Demonstrations are produced by a privileged expert that plans on the ground-truth occupancy and tracks the resulting minimum-snap trajectories [3]; expert chunks are recorded at the knot rate 1/∆ = 50 Hz together with the RGB-D stream and EKF2 estimates. We plan 2,000 expert flights over 200 training worlds followed by three DAgger rounds of 500 flights each (Section 3.8); dataset sizes and the expert’s own success rate are reported.

## 5.2 Hardware-in-the-Loop Specification

Configuration. The trained pipeline is exported to ONNX and compiled with TensorRT in FP16 (INT8 with a calibration cache as ablation A9) on a Jetson Orin Nano 8 GB developer kit in the 15 W mode, with clock settings fixed and reported. A simulation host with a discrete GPU runs PX4 SITL and the simulator; rendered RGB-D frames are streamed to the Jetson over a dedicated Gigabit Ethernet link, and the uXRCE-DDS agent on the Jetson connects to the SITL client over UDP. In a second stage, PX4 runs on a physical flight controller in PX4’s HITL mode, connected to the Jetson over a serial link at 921 600 baud, so that the serial transport and flight-controller scheduling are exercised.

Timing validity. Host and Jetson clocks are synchronized with the Precision Time Protocol. The simulator runs in real time, and episodes whose real-time factor falls below 0.98 at any point are discarded and reported. The image transport from host to Jetson is not present on a vehicle; its latency is measured separately and reported both included in and excluded from $T _ { \mathrm { e 2 \epsilon } }$ <sub>e</sub>.

Instrumentation. Every term of Eq. (42) is logged per decision from synchronized timestamps: capture, preprocessing, the four network times (CUDA events), safety filter, executor wait, and DDS transport (round trip halved and cross-checked against synchronized one-way stamps). Power and utilization are logged with tegrastats at 100 ms intervals, including module input power (VDD\_IN), the combined CPU–GPU–CV rail (VDD\_CPU\_GPU\_CV), GPU and memory-controller utilization, and temperatures. On this module the GPU is not metered on a separate rail, so the combined rail is reported as the GPU-power metric.

## 5.3 Baselines

All baselines receive the same observations (RGB-D, EKF2 state, mission waypoints), run on the same Jetson in the same power mode, and command PX4 through the same interface with the same speed limits; imitation-learning baselines use the same expert data and DAgger budget, and hyperparameters are tuned with equal search budgets on validation worlds.

$\mathbf { B 1 , }$ classical: VINS-Fusion [42] stereo-inertial odometry, an OctoMap [2] at 0.1 m resolution, $\mathbf { A } ^ { * }$ search on the occupancy grid, and minimum-snap trajectory generation [3] tracked through PX4 position setpoints.

• B2, end-to-end reinforcement learning: vision-based PPO [71] and SAC [72] policies with an encoder of the same size as $E _ { \psi }$ and an asymmetric critic with privileged state, trained on the reward of $\operatorname { E q . } \left( 5 \right)$ with a progress term in a GPU-parallel simulator [73] with matched vehicle parameters and depth rendering, and evaluated in the common SITL protocol. The training budget is reported in environment steps.

• B3, generative policies: a downscaled Diffusion Policy [19] with the encoder of Table 1, a solver of equal size, and DDIM sampling with 10 and 100 denoising steps; Octo-Small (27M parameters) [39], fine-tuned with a four-dimensional velocity action head; and Qantara [41], trained from scratch on the same data with its published objective, with RGB-D input and the waypoint feature $g _ { t }$ of Eq. (3) added to its conditioning so that it receives the same mission information as the other methods. Qantara is evaluated through its behaviour-cloning and video-inverse paths with two Euler steps; its latent-planning path is excluded from the real-time protocol because its reported planning time of 0.4–1.2 s per step on an H100 GPU exceeds the control period by more than an order of magnitude.

## 5.4 Evaluation Metrics

For N episodes, let $\mathcal { E } _ { i } ^ { \mathrm { s u c c } }$ denote the event that episode i reaches the final waypoint within $\tau _ { w } = 1$ m before the time limit $T _ { \mathrm { m a x } }$ without collision, and $\mathcal { E } _ { i } ^ { \mathrm { c o l } }$ the event that dist ${ \displaystyle \langle p ( \tau ) , \mathcal { X } \rangle < r _ { v } }$ at some time τ during the episode. The primary metrics are

$$
\mathrm { S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { k } \bigl [ \mathcal { E } _ { i } ^ { \mathrm { s u c c } } \bigr ] , \qquad \mathrm { C R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { k } \bigl [ \mathcal { E } _ { i } ^ { \mathrm { c o l } } \bigr ] ,\tag{45}
$$

$$
\bar { v } = \frac { 1 } { | \mathcal { T } _ { \mathrm { s u c c } } | } \sum _ { i \in \mathcal { T } _ { \mathrm { s u c c } } } \frac { \ell _ { i } } { T _ { i } } ,\tag{46}
$$

$$
T _ { \mathrm { e 2 e } } ^ { ( q ) } = \mathrm { t h e ~ } q \mathrm { - q u a n t i l e ~ o f ~ } \{ T _ { \mathrm { e 2 e } , t } \} _ { t } , \quad q \in \{ 0 . 5 , 0 . 9 5 , 0 . 9 9 \} ,\tag{47}
$$

$$
\bar { P } _ { \mathrm { r a i l } } = \frac { 1 } { T } \int _ { 0 } ^ { T } P _ { \mathrm { r a i l } } ( \tau ) \mathrm { d } \tau , \qquad E _ { \mathrm { d e c } } = \frac { \bar { P } _ { \mathrm { V D D \_ I N } } } { f _ { \mathrm { d e c } } } ,\tag{48}
$$

where $\mathcal { T } _ { \mathrm { s u c c } }$ is the set of successful episodes, $\ell _ { i }$ and $T _ { i }$ are the flown path length and duration of episode i, $T _ { \mathrm { e 2 e } , t }$ is the latency of Eq. (42) for decision $t , P _ { \mathrm { r a i l } }$ is the power of the rail VDD\_CPU\_GPU\_CV or VDD\_IN over a logging window of length T, and $E _ { \mathrm { d e c } }$ is the module energy per decision. Secondary metrics are the decision rate $f _ { \mathrm { d e c } } ;$ the deadline-miss rate, i.e., the fraction of executor ticks served by the fallback because no valid chunk was available; the root-mean-square jerk of the flown trajectory; the magnitude of velocity-reference discontinuities at chunk switches before blending; and, for the safety layer, the area under the receiver operating characteristic curve

$$
\mathrm { A U R O C } = \mathrm { P r } \left( \bar { \sigma } _ { t ^ { + } } > \bar { \sigma } _ { t ^ { - } } \right) ,\tag{49}
$$

where $t ^ { + }$ and $t ^ { - }$ are decision steps drawn uniformly from those followed and not followed, respectively, by a collision or a near-miss within $T _ { h } = 1 \mathrm { s }$ , together with the rates of true and spurious fallbacks and the mean speed reduction caused by $s _ { t }$ in perturbation-free segments, which quantifies conservatism.

## 5.5 Hypotheses and Statistical Protocol

Hypothesis 1 (Latency). On the Jetson Orin Nano in the 15 W mode with TensorRT FP16, $T _ { \mathrm { V P A } }$ for $K = 2$ and $H = 1 6$ has a 99th percentile of at most 20 ms, and $T _ { \mathrm { e 2 e } } ^ { ( 0 . 9 9 ) }$ excluding the simulator image transport is at most 50 ms; $t _ { v } ( H )$ remains within 10% of its value at $H = 8$ for $\bar { H } \leq 6 4$

Hypothesis 2 (Task performance). At each speed limit, the SR of Aero-VPA is non-inferior to that of the best B3 baseline with a margin of 5 percentage points, while its $T _ { \mathrm { e 2 e } } ^ { ( 0 . 9 5 ) }$ is lower; and its SR decreases less than that of B1 between $\bar { v } _ { h } = 3$ and 7 m/s.

Table 2: Target comparison schema (template; no results). Each block is reported per environment class (E1–E3), condition (ID/OOD), and speed limit $\bar { v } _ { h } \in \{ 3 , 5 , 7 \}$ m/s. “–” denotes a quantity to be measured under the protocol of Section $5 ; { \bar { P } } _ { \mathrm { C G C } }$ is the mean power of the VDD\_CPU\_GPU\_CV rail. The last row states the acceptance criteria for Aero-VPA (V1) and the hypothesis each criterion tests.
<table><tr><td>ID</td><td>Method</td><td>Params</td><td>SR (%)</td><td>CR (%)</td><td>u (m/s)</td><td> $T _ { \mathrm { e 2 e } } { \mathsf { p } } 5 0 / { \mathsf { p } } 9 5$  (ms)</td><td> $\bar { P } _ { \mathrm { C G C } }$  (W)</td><td> $E _ { \mathrm { d e c } }$  (mJ)</td></tr><tr><td>B1</td><td>VINS-Fusion + OctoMap + min-snap</td><td></td><td></td><td></td><td></td><td>-1-</td><td></td><td></td></tr><tr><td>B2a</td><td>Vision PPO</td><td>matched</td><td></td><td></td><td></td><td>-1-</td><td></td><td></td></tr><tr><td>B2b</td><td>Vision SAC</td><td>matched</td><td></td><td></td><td></td><td>-1-</td><td></td><td></td></tr><tr><td>B3a</td><td>Diffusion Policy (DDIM 10 / 100)</td><td>matched</td><td></td><td></td><td></td><td>-1-</td><td></td><td></td></tr><tr><td>B3b</td><td>Octo-Small, fine-tuned</td><td>27M</td><td></td><td></td><td></td><td>-1-</td><td></td><td></td></tr><tr><td>B3c</td><td>Qantara (behaviour cloning / video-inverse)</td><td>≈22M</td><td></td><td></td><td>一</td><td>-1-</td><td></td><td>一</td></tr><tr><td>V0</td><td>Aero-VPA without safety layer</td><td>18.13M</td><td>一</td><td>一</td><td>一</td><td>-1-</td><td>一</td><td>一</td></tr><tr><td>V1</td><td>Aero-VPA, full (K = 2)</td><td>18.13M</td><td></td><td></td><td></td><td>-1-</td><td></td><td></td></tr><tr><td colspan="2">Acceptance criteria for V1 (hypothesis tested)</td><td></td><td> $\geq \mathbf { B } 3 ^ { * } - 5$  (H2)</td><td>&lt; V0 (H3)</td><td>report vs. B1</td><td> $\mathsf { p } 9 9 \leq 5 0$  (H1)</td><td>15W mode</td><td>report</td></tr></table>

B3<sup>∗</sup>: best B3 baseline at the same speed limit (non-inferiority margin 5 percentage points). Network-only p99 target: $T _ { \mathrm { V P A } } \leq 2 0 \mathrm { m s } .$

Hypothesis 3 (Uncertainty and safety). Under OOD conditions, σ¯ predicts collisions and near-misses within 1 s with $\mathrm { A U R O C } \geq 0 . 7 5$ , and the full safety layer reduces CR relative to the unfiltered policy while increasing the mean mission time by at most 15%.

Hypothesis 4 (Representation). Under held-out visual perturbations, the relative SR decrease of Aero-VPA is smaller than that of an otherwise identical policy whose encoder is trained with a pixel-reconstruction objective.

Hypothesis 5 (Chunk coherence). Noise persistence (Eq. (20)) reduces the median velocity-reference discontinuity at chunk switches and does not reduce SR in environment class E3.

Statistical analysis. Each method is trained with three seeds. For each seed, environment class, condition, and speed limit, 500 episodes are run (10 test worlds × 50 start states), with identical world seeds and start states across methods. Proportions are reported with 95% Wilson intervals and continuous metrics with bootstrap intervals over episodes, together with seed-level means and standard deviations. Paired binary outcomes are compared with McNemar’s exact test, and the non-inferiority test of Hypothesis 2 uses the lower bound of a one-sided 95% interval on the paired SR difference; p-values within each hypothesis family are corrected with the Holm–Bonferroni procedure. For an unpaired two-sided two-proportion z-test at significance level $\alpha _ { s }$ and power $1 - \beta _ { s }$ , the number of episodes per method needed to detect a difference between success probabilities $p _ { 1 }$ and $p _ { 2 }$ is

$$
n = \frac { \Big ( z _ { 1 - \alpha _ { s } / 2 } \sqrt { 2 \bar { p } ( 1 - \bar { p } ) } + z _ { 1 - \beta _ { s } } \sqrt { p _ { 1 } ( 1 - p _ { 1 } ) + p _ { 2 } ( 1 - p _ { 2 } ) } \Big ) ^ { 2 } } { ( p _ { 1 } - p _ { 2 } ) ^ { 2 } } , \qquad \bar { p } = \frac { p _ { 1 } + p _ { 2 } } { 2 } ,\tag{50}
$$

where $z _ { q }$ is the standard normal q-quantile. For $p _ { 1 } = 0 . 6 , p _ { 2 } = 0 . 7 , \alpha _ { s } = 0 . 0 5$ , and $1 - \beta _ { s } = 0 . 8 ,$ , Eq. (50) gives n ≈ 356; with 500 episodes the power is about 0.91, and the paired design is expected to increase it further.

## 5.6 Target Comparison Schema

Table 2 is the template in which the main comparison will be reported, for each environment class, condition, and speed limit. All measured cells are empty by design.

## 5.7 Ablation Matrix

Ablations vary one factor at a time around the default configuration (V1), except A2, which is a full grid because its purpose is to trace the safety–conservatism Pareto front in the plane of collision rate and normalized mission time. Table 3 lists the factors, levels, readouts, and the hypothesis each ablation addresses.

## 6 Closed-Loop Evaluation in Simulation

This section reports the first executed part of the protocol of Section 5: closed-loop waypoint missions through procedurally generated forests, environment class E1 of Section 5.1, in PX4 SITL with Gazebo Harmonic. It evaluates

Table 3: Planned ablation matrix. Defaults are underlined; all other factors are held at their defaults. “Pareto” denotes the front of collision rate versus mission time normalized by that of V0.
<table><tr><td>ID</td><td>Factor</td><td>Levels</td><td>Primary readouts</td><td>Tests</td></tr><tr><td>A1</td><td>CFM integration steps</td><td> $K \in \{ 1 , 2 , 4 , 8 \}$ </td><td> $K t _ { v } , T _ { \mathrm { V P A } } , \mathrm { S R } , \mathrm { C R } .$  chunk error vs. K = 64 reference</td><td>H1, H2</td></tr><tr><td>A2</td><td>Uncertainty threshold × attenuation floor</td><td> $p _ { \mathrm { f a } } \in \{ 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , \underline { { { 2 . 8 \times 1 0 ^ { - 5 } } } } \} \ \mathrm { o r }$   $\mathrm { f a l l b a c k o f f } ; \times \alpha _ { \mathrm { m i n } } \overline { { \in \{ 0 . 2 5 , \underline { { { 0 } } } . 5 , 1 \} } }$ </td><td>Pareto, spurious and true fallback rates</td><td>H3</td></tr><tr><td>A3</td><td>Attenuation law</td><td>none, piecewise linear (Eq. (24)) exponential</td><td>CR, v, jerk</td><td>H3</td></tr><tr><td>A4</td><td>Braking envelope</td><td>off, on</td><td>CR in E1/E2, v, conservatism</td><td>H3</td></tr><tr><td>A5</td><td>Noise persistence</td><td>fresh noise, persistent,  $N _ { \xi } = 3 0 ,$  persistent,  $N _ { \xi } = 1 0$ </td><td>discontinuity at switches, SR in E3</td><td>H5</td></tr><tr><td>A6</td><td>Latent predictor</td><td>removed (reactive CFM), JEPA + VICReg, JEPA without</td><td>SR, effective rank of C(Z), AUROC</td><td>H4</td></tr><tr><td>A7</td><td>Encoder objective</td><td> ${ \overline { { v ( Z ) , c ( Z ) } } }$  JEPA, pixel reconstruction, frozen pretrained DINOv2-S [74]</td><td>relative SR drop under visual OOD</td><td>H4</td></tr><tr><td>A8</td><td>Execution mode</td><td>asynchronous + blending, asynchronous + inpainting [40], synchronous prefix with fixed  $\dot { H } _ { t } \in \{ 4 , 8 , \dot { 1 6 } \}$ </td><td>SR, deadline-miss rate, discontinuity</td><td>H5</td></tr><tr><td>A9</td><td>Numerical precision</td><td> $H _ { t }$  FP32, FP16, INT8</td><td> $T _ { \mathrm { V P A } , E _ { \mathrm { d e c } } }$  , chunk deviation</td><td>H1</td></tr><tr><td>A10</td><td>Chunk horizon</td><td> $H \in \{ 8 , \underline { { 1 6 } } , 3 2 , 6 4 \}$ </td><td>from FP32 tv(H), SR, jerk</td><td>H1</td></tr></table>

behaviour cloning, six DAgger checkpoints and the runtime shield of Section 3.9; Section 8.3 lists how the evaluation departs from the protocol, and the appendices give the full results.

## 6.1 Setup

Vehicle. The vehicle is a PX4 x500 quadrotor model with a front RGB-D camera of 160 × 128 pixels, a horizontal field of view of $7 3 ^ { \circ }$ and a depth range of $0 . 1 { - } 1 0 \mathrm { m }$ , which provides the frame $I _ { t }$ of Section 3.1; PX4 runs in SITL, and the companion-computer software of Section 4.6 commands it in offboard mode.

Worlds and missions. The test worlds are 36 procedurally generated forests, seeds 2000–2011 at three densities (Poisson-disc trunk spacing of 7, 5 and 3.5 m for sparse, medium and dense), disjoint from the training (0–199) and calibration (1000–1011) worlds. Each world has five missions, 180 per policy; mission k of a world is seeded by (world seed, k), so every policy flies identical missions and all comparisons are paired.<sup>2</sup> A mission takes off to 2.5 m and visits 3–6 waypoints placed 10–40 m apart within a time limit of 1.5 $T _ { \mathrm { r e f } } + 1 5 \mathrm { s } ,$ where $T _ { \mathrm { r e f } }$ is the duration of the privileged expert’s reference plan at 3 m/s. The policies are not speed-conditioned.

Endings. A mission ends in success when the final waypoint is reached within $\tau _ { w } = 1$ m before the time limit; in a collision when the privileged distance from the vehicle centre to a trunk surface (trunks are vertical cylinders in the world file) falls below 0.35 m; in ground contact below 0.3 m of altitude; in a timeout at the limit; in a camera timeout after 0.5 s without a decision; or out ofbounds. A crash is a collision or a ground contact.

Policies. Table 4 lists the nine policies. The three BC policies flew 60 missions (12 of the 36 test worlds) before a start-up warm-up fix; the six DAgger checkpoints flew all 180 missions without and with the shield.

Compute. Three PX4 SITL and Gazebo Harmonic [67] instances ran headless in parallel on one virtual machine with 16 vCPUs and an NVIDIA L4 GPU.

Metrics. Besides the endings we report the share of all mission waypoints reached; the median duration and path length; the ratio of duration to the expert’s reference time and of path length to mission length (medians over successful missions); the mean speed (mean over missions of the per-mission mean); the smallest clearance (median over missions of the per-mission minimum privileged distance); the share of blind decisions, whose commanded direction lay outside the field of view or whose depth was invalid; fallback onsets per mission; and, for the shield, interventions per mission, seconds per mission with each rule active, and the clearance to the memory (median over missions of the minimum distance from the estimated position to a remembered return). These metrics differ from those planned in Section 5.4: the mean speed averages over all missions rather than over successful ones as in Eq. (46), the latency quantiles of Eq. (47) and the power and energy of Eq. (48) were not measured, and Section 4.6 reports the decision timing on the L4 instead.

Table 4: Evaluated policies. Except ${ \mathrm { B C } } _ { \mathrm { N R } }$ , all use the frozen residual encoder $E _ { \psi }$ and predictor $P _ { \omega } ;$ all are calibrated with the episode-maximum method of Section 3.5. Withheld inputs enter $\pi _ { \phi } ^ { h }$ and $e _ { t }$ at their training mean; $P _ { \omega }$ always receives ${ \bar { a } } _ { t } .$ “Demos”: demonstrations from the 200 training worlds.
<table><tr><td>Policy</td><td>Training data</td><td>Inputs and recipe</td></tr><tr><td> $\mathbf { B C } _ { \mathrm { N R } }$  BC</td><td>demos</td><td>non-residual stage (i); all inputs residual stage (i); all inputs</td></tr><tr><td> $\mathrm { B C } _ { - \bar { a } }$ </td><td>demos demos</td><td>as BC, with āt withheld</td></tr><tr><td> $\mathrm { D } _ { 1 }$ </td><td>demos + DAgger round 1 (flown by BC)</td><td>all inputs</td></tr><tr><td> $\mathrm { D } _ { 2 }$ </td><td>demos + rounds 1–2 (round 2 flown by  $\mathbf { D } _ { 1 } )$ </td><td>all inputs</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>demos + rounds 1–3 (round 3 flown by</td><td> ${ { \bar { a } } _ { t } }$  withheld</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } , v }$ </td><td> $\mathbf { D } _ { 2 } )$   $\mathrm { D } _ { 3 - \bar { a } }$  as</td><td> ${ { \bar { a } } _ { t } }$  and  $v _ { t } ^ { \mathcal { H } }$  withheld</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td> $\mathrm { a s } \mathrm { D } _ { 3 - \bar { a } }$ </td><td>all inputs; state dropout:  $v _ { t } ^ { \mathcal { H } }$  and  ${ { \bar { a } } _ { t } }$  replaced by their training mean in 50% of training frames</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>as  $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>all inputs; keyframe-weighted sampling, frames drawn ∝ min  $\left( 1 + \mathrm { \bar { \| } } A _ { t } [ 0 ] - \bar { a } _ { t } \mathrm { \bar { \| } } / 0 . 1 , \ 1 \mathrm { \hat { 0 } } \right)$  , the error of</td></tr></table>

Statistics. Success and crash rates carry Wilson 95% intervals [75]. Paired comparisons use the two-sided exact McNemar test on the discordant pairs [76]; tables give the counts of missions in which both, only one or neither of the two policies succeeded. N is stated in every caption. The primary family of tests is the six shielded-against-unshielded success comparisons; with Holm’s correction [77] its largest adjusted $p$ is $2 . 4 \times 1 0 ^ { - 1 0 }$ . All other tests are secondary and unadjusted. Each recipe was trained once, so the tests capture variation between missions, not between training runs.

## 6.2 Behaviour Cloning Fails: The Copycat Shortcut

BC succeeded in none of the 60 missions it flew (Appendix D): BC crashed in 83% of them, BC in 51.7% with timeouts in the rest, and $\mathrm { B C } _ { - \bar { a } }$ in $7 0 . 0 \%$ , mostly into the ground. Offline, the same policies looked nearly perfect (Table 5, top): on 5000 held-out expert frames, the median error of BC’s first knot was 0.02 m/s $( \mathbf { B C } _ { - \bar { a } } \colon 0 . 0 4 \mathbf { m } / \mathrm { s } )$ , and its selector agreed with the expert’s primitive in 98.0% of frames (93.6%).

The swap test explains the gap. It replaces one input by that of another frame and measures the change of the first knot; the labels of two frames differ by a median of 1.13 m/s. For $\mathrm { B C } ,$ swapping ${ { \bar { a } } _ { t } }$ changed the command by 0.85 m/s, moved it to within 0.31 m/s of the donor frame’s label, and changed the primitive in 35.5% of frames, whereas swapping the image or the goal changed the command by 0.01 m/s. With ${ { \bar { a } } _ { t } }$ withheld, $\mathrm { B C } _ { - \bar { a } }$ copied the measured velocity instead (1.17 m/s, 0.21 m/s from the donor’s label). On expert data the last command is within about 0.02 m/s of the next label, so this shortcut fits the demonstrations almost exactly and no offline metric reveals it.

DAgger did not remove the shortcut on expert frames. $\mathrm { D } _ { 1 }$ and $\mathrm { D } _ { 2 }$ followed the donor’s ${ { \bar { a } } _ { t } }$ (0.96 and 0.93 m/s; 0.15 and 0.27 m/s from its label) and ignored the image and the goal $( \leq 0 . 0 2 \mathrm { { m } / \mathrm { { s } ) } }$ ; their velocity swaps changed the command by medians of 0.17 and 0.31 m/s but by 1.24 and 1.62 m/s at the 90th percentile, a growing dependence rather than a copy. With $\bar { a } _ { t } = 0 , \mathrm { D } _ { 1 } \ '$ first knot was still 1.42 m/s (BC: 0.57 m/s). The round-3 recipes moved the shortcut rather than removing it. $\mathrm { D } _ { 3 - \bar { a } }$ copied velocity (1.10 m/s, 0.28 m/s from the donor’s label). $\mathrm { D } _ { 3 - { \bar { a } } , v }$ copied nothing—image and goal swaps moved its command by 0.43 and 0.42 m/s, with donor errors of at least 1.04 m/s—but underfitted (open-loop error 0.31 m/s median and 1.19 m/s at the 90th percentile; selector agreement 78.3%), and ${ { \bar { a } } _ { t } }$ reached its command through the predictor’s forecast (0.33 m/s, 0.86 m/s from the donor’s label against 0.57 m/s from its own). State dropout $\left( \mathrm { D } _ { 3 , \mathrm { S D } } \right)$ reduced the velocity dependence (0.16 m/s) but not the ${ { \bar { a } } _ { t } }$ dependence (0.83 m/s, 0.28 m/s from the donor’s label against 0.84 m/s from its own), and keyframe weighting $( \mathrm { D } _ { 3 , \mathrm { K W } } )$ left both (0.49 and 0.88 m/s; Appendix E).

Table 5: Copycat diagnostics. Top: 5000 held-out expert frames (validation flights). A swap replaces one input by that of another frame; entries are the median change of the first knot $v _ { 0 }$ (m/s), with the error to the donor frame’s label in parentheses (the median label difference between frames is 1.13 m/s). Open loop: median $| v _ { 0 } - v _ { 0 } ^ { \star } |$ and selector agreement with the expert’s primitive. Bottom: the policies’ own DAgger states, labelled by the expert and held out for the policies shown; $\mathrm { ^ { 6 6 } > 4 5 ^ { \circ , \bar { \nu } } }$ : share of frames whose command points more than $4 5 ^ { \circ }$ from the label. “–”: withheld input, or not measured.
<table><tr><td colspan="10">Expert frames</td></tr><tr><td>Policy</td><td> ${ \bar { a } } _ { t } { \ } { \mathrm { \ s w a p } }$ </td><td> $v _ { t } ^ { \mathcal { H } }$  swap</td><td>image</td><td>goal</td><td>all of  $S _ { t }$ </td><td> ${ { \bar { a } } _ { t } }$  to  $P _ { \omega }$  only</td><td> $| v _ { 0 } - v _ { 0 } ^ { \star } |$ </td><td>selector</td></tr><tr><td>BC</td><td>0.85 (0.31)</td><td>0.31 (0.86)</td><td>0.01</td><td>0.01</td><td>1.14</td><td>0.00</td><td>0.02</td><td>98.0%</td></tr><tr><td> $\mathrm { B C } _ { - \bar { a } }$ </td><td></td><td>1.17 (0.21)</td><td>0.03</td><td>0.02</td><td>1.14</td><td>0.00</td><td>0.04</td><td>93.6%</td></tr><tr><td>D1</td><td>0.96 (0.15)</td><td>0.17 (1.02)</td><td>0.02</td><td>0.01</td><td>一</td><td></td><td></td><td></td></tr><tr><td>D2</td><td>0.93 (0.27)</td><td>0.31 (0.99)</td><td>0.02</td><td>0.01</td><td>一</td><td></td><td>一</td><td>一</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td></td><td>1.10 (0.28)</td><td></td><td></td><td>1</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { D } _ { 3 - { \overline { { a } } } , v }$ </td><td></td><td></td><td>0.43</td><td>0.42</td><td></td><td>0.33 (0.86)</td><td>0.31</td><td>78.3%</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>0.83 (0.28)</td><td>0.16 (0.98)</td><td>0.04</td><td>0.03</td><td>1.09</td><td>0.02</td><td>0.04</td><td>91.8%</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>0.88 (0.43)</td><td>0.49 (0.97)</td><td>0.03</td><td>0.02</td><td>1.14</td><td>0.00</td><td>0.03</td><td>92.9%</td></tr><tr><td>Policies&#x27; own states</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>States</td><td>Policy</td><td>error to label</td><td> $> 4 5 ^ { \circ }$ </td><td>selector</td><td>swap on these states</td><td></td><td></td><td></td></tr><tr><td>round 2 (flown by D1)</td><td>BC</td><td>0.50</td><td>22.2%</td><td>56.8%</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>D1</td><td>0.49</td><td>24.2%</td><td>61.2%</td><td>image 0.17 (p90 1.41)</td><td></td><td></td><td></td></tr><tr><td>round 3 (flown by D₂)</td><td>D1</td><td>0.51</td><td>22.0%</td><td>62.7%</td><td> ${ { \bar { a } } _ { t } }$  1.38</td><td></td><td></td><td></td></tr><tr><td></td><td>D2</td><td>0.47</td><td>21.8%</td><td>62.9%</td><td> ${ { \bar { a } } _ { t } }$  1.19</td><td></td><td></td><td></td></tr></table>

Table 6: DAgger rounds under the privileged supervisor $( N = 2 0 1$ missions per round over the training worlds). These flights are gated and partly flown by the expert, so their success and crash rates measure the supervised system, not autonomous performance.
<table><tr><td>Round</td><td>Flown by</td><td>Missions</td><td>Success</td><td>Crash</td><td>Policy&#x27;s share of the flight</td></tr><tr><td>1</td><td>BC</td><td>201</td><td>148 (73.6%)</td><td>28 (13.9%)</td><td>≈50%</td></tr><tr><td>2</td><td>D1</td><td>201</td><td>166 (82.6%)</td><td>5 (2.5%)</td><td>63%</td></tr><tr><td>3</td><td> $\mathrm { D } _ { 2 }$ </td><td>201</td><td>189 (94.0%)</td><td>10 (5.0%)</td><td>64%</td></tr></table>

On the policies’ own states the picture differs (Table 5, bottom). On the states $\mathrm { D } _ { 1 }$ flew in round 2, labelled by the expert and held out for both policies shown, BC and $\mathrm { D } _ { 1 }$ erred by 0.50 and 0.49 m/s, pointed more than $4 5 ^ { \circ }$ away from the label in 22.2% and 24.2% of frames, and agreed with the expert’s primitive in only 56.8% and 61.2%; there $\mathrm { D } _ { 1 }$ did use the image (swap 0.17 m/s median, 1.41 m/s at the 90th percentile). On $\mathrm { D } _ { 2 } \mathbf { \ ' } _ { \mathbf { s } }$ round-3 states, $\mathrm { D } _ { 1 }$ and $\mathbf { D } _ { 2 }$ erred by 0.51 and 0.47 m/s, and swapping ${ { \bar { a } } _ { t } }$ still changed their commands by 1.38 and 1.19 m/s. More rounds with ${ { \bar { a } } _ { t } }$ available did not converge. The expert itself was blind in 3.7% of its held-out frames but in 34% of the frames on $\mathrm { D } _ { 1 } \mathrm { \dot { s } }$ round-2 states.

## 6.3 DAgger Gives the First Successes, Then Plateaus

Under the supervisor, success rose from 73.6% in round 1 to 94.0% in round 3 (Table 6), but these flights were gated and partly flown by the expert. Autonomously, without the shield, $\mathrm { D } _ { 1 }$ succeeded in 38 of the 180 test missions (21.1%, 95% CI [15.8, 27.6]; Table 8); on the 60 missions shared with BC, D alone succeeded in 12 and BC alone in none $( p = 0 . 0 0 0 5 )$ . No later checkpoint improved on it (Table 7): $\mathbf { D } _ { 2 } ,$ trained on two DAgger rounds, succeeded in 12.8% $[ 8 . 7 , 1 8 . 4 ] ( p = 0 . 0 3 1 5$ against $\mathbf { D } _ { 1 } )$ , and the round-3 recipes in $1 1 . 1 { - } 1 5 . 6 \%$ , with $\mathrm { D } _ { 3 - \bar { a } }$ significantly worse than $\mathrm { D } _ { 1 }$ $( p = 0 . 0 0 5 1 )$ and the others not significantly different from it $( p = 0 . 0 7 0 4 \ – 0 . 1 8 4 )$ or from D $( p = 0 . 4 7 3  – 1 )$ . Crashes ended 72.2–82.2% of the unshielded missions, and the median smallest clearance was 0.35 m for all six checkpoints, the collision distance, because most missions ended in a collision. Mission outcomes varied strongly between checkpoints: $\mathrm { D } _ { 1 }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ shared only 12 of their 38 and 28 successes, where independent draws at those rates would share about 6. Together with the errors of about 0.5 m/s on the policies’ own states (Section 6.2), we read this as a plateau: the unshielded DAgger policies stayed at 11.1–21.1% success, and neither input removal, state dropout nor keyframe weighting beat the first round.

Table 7: Paired success comparisons without the shield $( N = 1 8 0$ paired test missions; last row $N = 6 0$ , the missions the BC policies flew). Discordant pairs and the two-sided exact McNemar p.
<table><tr><td>Comparison (A vs B)</td><td>Only A succeeded</td><td>Only B succeeded</td><td>p</td></tr><tr><td> $\mathbf { D } _ { 1 } \ \mathbf { v s } \ \mathbf { D } _ { 2 }$ </td><td>29</td><td>14</td><td>0.0315</td></tr><tr><td> $\mathrm { D } _ { 1 } \ \mathrm { v s } \ \mathrm { D } _ { 3 - \bar { a } }$ </td><td>28</td><td>10</td><td>0.0051</td></tr><tr><td> $\mathrm { D } _ { 1 } \mathrm { v s } \mathrm { D } _ { 3 - \bar { a } , v }$ </td><td>33</td><td>19</td><td>0.0704</td></tr><tr><td> $\mathrm { D } _ { 1 } \ \mathrm { v s } \ \mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>28</td><td>18</td><td>0.184</td></tr><tr><td> $\mathrm { D } _ { 1 } \ \mathrm { v s } \ \mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>26</td><td>16</td><td>0.164</td></tr><tr><td> $\mathrm { D } _ { 2 } { \mathrm { ~ v s ~ } } \mathrm { D } _ { 3 - \bar { a } }$ </td><td>19</td><td>16</td><td>0.736</td></tr><tr><td> $\mathrm { D } _ { 2 } \mathrm { \ v s } \mathrm { D } _ { 3 - \bar { a } , v }$ </td><td>17</td><td>18</td><td>1</td></tr><tr><td> $\mathrm { D } _ { 2 } \ \mathrm { v s } \ \mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>14</td><td>19</td><td>0.487</td></tr><tr><td>D2 vs D3,KW</td><td>13</td><td>18</td><td>0.473</td></tr><tr><td> $\mathrm { D } _ { 1 }$  vs BC (60 missions)</td><td>12</td><td>0</td><td>0.0005</td></tr></table>

Table 8: Main result: the six DAgger checkpoints without and with the runtime shield on the same $N = 1 8 0$ paired test missions. Success counts with Wilson 95% intervals; $\Delta \colon$ change in percentage points; paired: missions in which both, only the unshielded (U), only the shielded (S), or neither run succeeded, with the two-sided exact McNemar p. Rows follow the training lineage, not a ranking: no shielded checkpoint differs significantly from $\mathrm { D } _ { 1 } + \mathrm { s h i e l d }$ (Table 17).
<table><tr><td></td><td colspan="2">Success</td><td></td><td colspan="2">Paired success</td><td>Crash (%)</td><td>Timeout (%)</td></tr><tr><td>Checkpoint</td><td>U</td><td>S [95% CI]</td><td> $\Delta \left( \mathsf { p p } \right)$ </td><td>both / U / S / neither</td><td>p</td><td> $\mathrm { U } \to \mathsf { S }$ </td><td> $\mathrm { U } \to \mathrm { S }$ </td></tr><tr><td> $\mathrm { D } _ { 1 }$ </td><td>38 (21.1%)</td><td>94 (52.2%) [45.0, 59.4]</td><td>+31.1</td><td>26 / 12 / 68 / 74</td><td> $1 . 2 0 \times 1 0 ^ { - 1 0 }$ </td><td> $7 2 . 2  2 2 . 2$ </td><td> $2 . 8  2 3 . 3$ </td></tr><tr><td>D2</td><td>23 (12.8%)</td><td>77 (42.8%) [35.8, 50.1]</td><td>+30.0</td><td>14 / 9 / 63 / 94</td><td> $4 . 1 8 \times 1 0 ^ { - 1 1 }$ </td><td> $8 0 . 0  3 8 . 3 $ </td><td> $1 . 1  1 6 . 7$ </td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>20 (11.1%)</td><td>79 (43.9%) [36.8, 51.2]</td><td>+32.8</td><td>11 /9/68 / 92</td><td> $2 . 4 5 \times 1 0 ^ { - 1 2 }$ </td><td> $8 2 . 2  3 6 . 1 $ </td><td> $2 . 8  1 4 . 4$ </td></tr><tr><td> $\mathrm { D } _ { 3 - { \bar { a } } , \ v }$ </td><td>24 (13.3%)</td><td>81 (45.0%) [37.9, 52.3]</td><td>+31.7</td><td>11 / 13 / 70 / 86</td><td> $1 . 3 3 \times 1 0 ^ { - 1 0 }$ </td><td> $7 7 . 2  4 2 . 8 $ </td><td>3.3 → 10.0</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>28 (15.6%)</td><td>82 (45.6%) [38.4, 52.8]</td><td>+30.0</td><td>19 / 9 / 63 / 89</td><td> $4 . 1 8 \times 1 0 ^ { - 1 1 }$ </td><td> $7 6 . 1  3 5 . 0$ </td><td>3.3 → 16.1</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>28 (15.6%)</td><td>90 (50.0%) [42.8, 57.2]</td><td>+34.4</td><td> $1 6 / 1 2 / 7 4 / 7 8$ </td><td> $4 . 6 9 \times 1 0 ^ { - 1 2 }$ </td><td>80.6 → 26.1</td><td> $1 . 7  2 1 . 7$ </td></tr></table>

## 6.4 Shielded Benchmark

The shield raised success for every checkpoint (Table 8, Figure 3): for D by 31.1 percentage points, from 21.1% to 52.2% [45.0, 59.4] (only shielded 68, only unshielded 12; $p \overset { \cdot } { = } 1 . 2 \times 1 0 ^ { - 1 0 } )$ , and for the other five by 30.0–34.4 points, to 42.8–50.0%. All six paired comparisons gave $p \leq 1 . 3 3 \times 1 0 ^ { - 1 0 }$ . Crashes fell from 72.2–82.2% to 22.2–42.8%; for $\mathbf { D } _ { 1 }$ from 130 to 40 missions (only unshielded 98, only shielded $8 ; p = 8 . 0 8 \times 1 0 ^ { - 2 1 }$ ; Appendix C). Every shielded checkpoint also succeeded more often than the best unshielded one, $\mathbf { D } _ { 1 } .$ , on the same missions $( p \leq 1 . 6 \overset { \cdot } { 9 } \times 1 0 ^ { - 5 } )$ Paired with $\mathrm { D } _ { 1 } + \mathrm { s h i e l d }$ , however, no shielded checkpoint differed significantly $\begin{array} { r } { ( p = 0 . 0 7 1 2 - 0 . 7 2 4 ; } \end{array}$ Table 17), so we do not rank them.

Failures moved from crashes to timeouts: $\mathrm { D } _ { 1 }$ ’s timeouts rose from 2.8% to 23.3% and $\mathrm { D } _ { 3 , \mathrm { K W } } \mathrm { \ ' } _ { \mathrm { s } }$ from 1.7% to 21.7%. The flight itself slowed only slightly: $\mathrm { D } _ { 1 } \mathrm { \ ' } _ { \mathrm { s } }$ mean speed went from 1.24 to 1.18 m/s, while its median smallest clearance rose from 0.35 to 0.52 m; its successful missions took 1.26 times the expert’s reference time and flew 1.13 times the mission length (medians). The learned fallback started 1.04–1.53 times per shielded mission, in 30.0–36.7% of missions, and braked or held for 0.1–0.3 s per mission (Appendix C).

## 6.5 Success by Obstacle Density

For D the shield’s gain was largest in the dense worlds, where success rose from 4 to 28 of 60 missions $( 6 . 7 \% $ 46.7%; Table 9). Without the shield every checkpoint succeeded in at most 4 of the 60 dense missions; with it, in 17–28. In percentage points, the largest gain fell in the dense worlds for $\mathrm { D } _ { 1 }$ and $\mathrm { D } _ { 3 - \bar { a } } .$ , in the medium worlds for $\mathrm { D } _ { 2 } , \mathrm { D } _ { 3 - \bar { a } , \ i }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ , and in the sparse worlds for $\mathrm { D } _ { 3 , \mathrm { S D } }$ . Shielded crashes were most frequent in the dense worlds for every checkpoint except $\mathrm { D } _ { 3 - \bar { a } }$ (Appendix C).

## 6.6 Missions Never Used to Revise the Shield

Mission 0 of every test world was the shield pilots’ set (Section 6.7), so missions 1–4 of each world, 144 per checkpoint, were never used to revise the shield. On them the result is essentially unchanged (Table 10): $\mathbf { D } _ { 1 }$ succeeded in 52.1% [44.0, 60.1], against 52.2% on all 180 missions, $\mathrm { D } _ { 3 , \mathrm { K W } }$ in 54.2% [46.0, 62.1], and the other four in 43.1–46.5%; every

![](images/20a63e0d66b24a9d8fac85c58e0730d10d67479e22c8d55b25d0162fd8644aea.jpg)  
Figure 3: Mission endings of the six DAgger checkpoints without and with the runtime shield $( N = 1 8 0$ paired test missions each); the number at the right of each bar is its success rate. The shield turns most collisions into successes or timeouts. Exact counts are in Table 15. Six first-person-view (FPV) videos on the model card (Code and Data Availability) show $\mathrm { D } _ { 1 } + \mathrm { s h i e l d }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ + shield on one mission per density that both completed in the benchmark; they are re-flights for illustration, and the shares shown come from the benchmark runs.

Table 9: Success by obstacle density, unshielded (U) and shielded (S), as successes of $N = 6 0$ missions per density (12 worlds $\times 5$ missions), with Wilson 95% intervals (%) for the shielded runs. Trunk spacing: dense 3.5 m, medium 5 m, sparse 7 m.
<table><tr><td></td><td colspan="2">Dense</td><td colspan="2">Medium</td><td colspan="2">Sparse</td></tr><tr><td>Checkpoint</td><td>U</td><td>S [95% CI]</td><td>U</td><td>S [95% CI]</td><td>U</td><td>S [95% CI]</td></tr><tr><td>D1</td><td>4 (6.7%)</td><td>28 (46.7%) [35, 59]</td><td>17 (28.3%)</td><td>29 (48.3%) [36, 61]</td><td>17 (28.3%)</td><td>37 (61.7%) [49, 73]</td></tr><tr><td> $\mathrm { D } _ { 2 }$ </td><td>0 (0.0%)</td><td>21 (35.0%) [24, 48]</td><td>9 (15.0%)</td><td>32 (53.3%) [41, 65]</td><td>14 (23.3%)</td><td>24 (40.0%) [29, 53]</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>2 (3.3%)</td><td>23 (38.3%) [27, 51]</td><td>6 (10.0%)</td><td>25 (41.7%) [30, 54]</td><td>12 (20.0%)</td><td>31 (51.7%) [39, 64]</td></tr><tr><td> $\mathrm { D } _ { 3 - { \bar { a } } , \ v }$ </td><td>1 (1.7%)</td><td>17 (28.3%) [19, 41]</td><td>6 (10.0%)</td><td>28 (46.7%) [35, 59]</td><td>17 (28.3%)</td><td>36 (60.0%) [47, 71]</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>2 (3.3%)</td><td>17 (28.3%) [19, 41]</td><td>9 (15.0%)</td><td>26 (43.3%) [32, 56]</td><td>17 (28.3%)</td><td>39 (65.0%) [52, 76]</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>1 (1.7%)</td><td>22 (36.7%) [26, 49]</td><td>10 (16.7%)</td><td>33 (55.0%) [42, 67]</td><td>17 (28.3%)</td><td>35 (58.3%) [46, 70]</td></tr></table>

paired comparison with the unshielded flights of the same missions gave $p \leq 5 . 2 6 \times 1 0 ^ { - 8 }$ . These are the same worlds, so this is a partial check, not a test on fresh worlds; the planned fresh-world development set was not flown.

## 6.7 Shield Design Iteration and Failure Locations

The shield was developed with $\mathrm { D } _ { 1 }$ on mission 0 of the 36 test worlds (Table 11). Shield v1 raised success from 7 to 12 of 36 and lowered crashes from 27 to 11, but its own manoeuvres failed: 10 of its 11 crashes happened while the shield was flying its own setpoint (backing off, nudging or holding); of its 10 collisions, 4 were with obstacles behind the camera, 4 with obstacles beside it and 2 with obstacles in view; and in all 8 crashes with a measurable direction of motion the vehicle was moving more than $3 6 ^ { \circ }$ off the camera axis, outside the field of view. Its uncertainty cap was active for 18.0 s per mission and turned flights into timeouts. Shield $\mathbf { v } 2$ therefore passes the backoff through the barrier and the altitude floor at 0.3 m/s (shield v1: 0.5 m/s), ends it in a nudge when the way back is blocked, applies the blind cap after the barrier, and disables the uncertainty cap. On the same $\bar { 3 6 }$ missions shield $\mathbf { v } 2$ succeeded in 20 (both 5, only unshielded 2, only shielded 15; $p = 0 . 0 0 2 3 )$ and crashed in 9, 7 of them while the shield was flying. Its failure pattern moved into the field of view: 6 of its 7 collisions were with obstacles in view, and the vehicle was moving outside the view in 2 of 7. Its 2 ground contacts descended at 2.8–3.3 m/s, above the 2 m/s vertical action bound, at total speeds of 4.0–4.4 m/s, too fast for the 0.8 m altitude floor.

Table 10: Missions never used to revise the shield: missions 1–4 of every test world (N = 144 paired missions per checkpoint; mission 0 of every world was the shield pilots’ set). Same worlds as the full benchmark, so this is a partial check, not a fresh-world test. Paired success counts and exact McNemar $p ;$ crashes unshielded → shielded with the exact McNemar p of the paired crash comparison.
<table><tr><td></td><td colspan="2">Success</td><td colspan="2">Paired success</td><td colspan="2">Crashes</td></tr><tr><td>Checkpoint</td><td>unshielded</td><td>shielded [95% CI]</td><td>both / only U / only S / neither</td><td>p</td><td>U → S</td><td>p</td></tr><tr><td> $\mathrm { D } _ { 1 }$ </td><td>31</td><td>75 (52.1%) [44.0, 60.1]</td><td>21 / 10 / 54 / 59</td><td> $2 . 0 0 \times 1 0 ^ { - 8 }$ </td><td> $1 0 3  3 3$ </td><td> $1 . 5 2 \times 1 0 ^ { - 1 5 }$ </td></tr><tr><td>D2</td><td>20</td><td>64 (44.4%) [36.6, 52.6]</td><td>13 / 7 / 51 / 73</td><td> $2 . 4 0 \times 1 0 ^ { - 9 }$ </td><td> $1 1 2  5 3$ </td><td> $3 . 8 5 \times 1 0 ^ { - 1 3 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>15</td><td>62 (43.1%) [35.3, 51.2]</td><td>8 / 7 / 54 / 75</td><td> $4 . 3 2 \times 1 0 ^ { - 1 0 }$ </td><td> $1 2 0  5 1$ </td><td> $9 . 4 3 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 - { \bar { a } } , v }$ </td><td>22</td><td>67 (46.5%) [38.6, 54.7]</td><td>10 / 12 / 57 / 65</td><td> $3 . 7 4 \times 1 0 ^ { - 8 }$ </td><td> $1 0 9  6 2$ </td><td> $3 . 8 1 \times 1 0 ^ { - 8 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>24</td><td>65 (45.1%) [37.2, 53.3]</td><td>15 / 9 / 50 / 70</td><td> $5 . 2 6 \times 1 0 ^ { - 8 }$ </td><td> $1 0 7  5 0$ </td><td> $1 . 2 6 \times 1 0 ^ { - 1 2 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>22</td><td>78 (54.2%) [46.0, 62.1]</td><td>15 / 7 / 63 / 59</td><td> $2 . 2 8 \times 1 0 ^ { - 1 2 }$ </td><td> $1 1 7  3 2$ </td><td> $3 . 0 9 \times 1 0 ^ { - 2 1 }$ </td></tr></table>

Table 11: Shield design iteration on the pilot missions: mission 0 of every test world flown by $\mathrm { D } _ { 1 } \left( N = 3 6 \right)$ . Paired: missions in which both, only the unshielded or only the shielded run succeeded, with the exact McNemar p. Crash locations give the bearing of the nearest trunk at the collision relative to the camera axis: in view within 36.5°, beside up to 100°, behind beyond. “Moving outside the view”: horizontal motion more than $3 6 . 5 ^ { \circ }$ off the camera axis, among collisions with a measurable direction (horizontal speed above 0.2 m/s). <sup>†</sup>From a separate traced flight of the same 36 missions without the shield (6 successes, 25 collisions, 2 ground contacts, 2 timeouts, 1 camera timeout).
<table><tr><td></td><td> $\mathrm { D } _ { 1 } ,$  unshielded</td><td>+ shield v1</td><td>+ shield v2</td></tr><tr><td>Success [95% CI]</td><td>7 (19.4%) [10, 35]</td><td>12 (33.3%) [20, 50]</td><td>20 (55.6%) [40, 70]</td></tr><tr><td>Crashes (collision / ground)</td><td>27 (23 / 4)</td><td>11 (10  1)</td><td>9 (7 / 2)</td></tr><tr><td>Timeouts / camera timeouts</td><td>0/2</td><td>12 /1</td><td>7/0</td></tr><tr><td>Paired with the unshielded run (both / only U / only S)</td><td></td><td> $2 / 5 / 1 0 , p = 0 . 3 0$ </td><td>5 / 2 / 15, p = 0.0023</td></tr><tr><td>Crashes while the shield flew</td><td></td><td>10 of 11</td><td>7 of 9</td></tr><tr><td>Collisions in view / beside / behind</td><td>13 / 12 / 0 of 25†</td><td>2 / 4 / 4 of 10</td><td>6 / 1 / 0 of 7</td></tr><tr><td>Moving outside the view</td><td>5 of 24†</td><td>8 of 8</td><td>2 of 7</td></tr><tr><td>Median speed at the crash (m/s)</td><td>0.95†</td><td>0.77</td><td>0.69</td></tr><tr><td>LATERALAvOID active at the crash</td><td>14 of 27†</td><td>5 of 11</td><td>3 of 9</td></tr><tr><td>Learned fallback active in the last second</td><td>0 of 27†</td><td>1 of 11</td><td>2 of 9</td></tr><tr><td>Per mission: backoffs / nudges / stalls / holds</td><td></td><td>0.78 / 4.03 / 3.44 / 4.61</td><td>0.89 / 2.53 / 1.83 / 3.39</td></tr><tr><td>Seconds per mission: σ cap / blind / barrier</td><td></td><td>18.0 / 17.1 / 12.4</td><td>0.0 / 14.4 / 13.6</td></tr><tr><td>Seconds per mission: nudge / backoff</td><td></td><td>5.6 / 2.4</td><td>3.5 / 3.8</td></tr></table>

A separately traced flight of the same 36 missions without the shield (6 successes, 25 collisions, 2 ground contacts, 2 timeouts, 1 camera timeout) gives the unshielded policy’s crash locations. Of its 25 collisions, 13 were with obstacles in view and 12 with obstacles beside the camera, none behind; the vehicle was moving outside the view in 5 of the 24 collisions with a measurable direction; the median speed at the crash was 0.95 m/s; LATERALAVOID was the active primitive in 14 of the 27 crashes; and the learned fallback was active in the last second before none of the 27. The unshielded policy thus crashed mostly while moving within the field of view, into obstacles in view or beside the camera—consistent with a policy that follows its last command more than its image—whereas the first shield’s own failures came from motion outside the field of view (all 8 cases with a measurable direction), which the second version reduced to 2 of its 7 collisions.

Mission 0 was flown twice for $\mathrm { D } _ { 1 }$ in both conditions: without the shield it succeeded in 7 of 36 missions in the evaluation and in 6 of 36 in the traced flight; with shield v2 in 20 in the pilot and 19 in the benchmark (94 successes on all missions minus 75 on missions 1–4). The aggregate counts agreed within one mission; we did not compute agreement mission by mission.

Table 12: Shield activity in the shielded benchmark $( N = 1 8 0$ missions per checkpoint): interventions per mission (means), seconds per mission with each rule active (means), clearance to the memory (median over missions of the per-mission minimum) and the memory update’s 95th percentile (median over missions). The uncertainty cap was disabled (0.0 s for every checkpoint).
<table><tr><td></td><td colspan="4">Per mission</td><td colspan="6">Seconds per mission</td><td rowspan="2">Memory</td><td rowspan="2">Update p95 (ms)</td></tr><tr><td>Checkpoint</td><td>backoffs</td><td>nudges</td><td>stalls</td><td>holds</td><td>barrier</td><td>blind</td><td>backoff</td><td>nudge</td><td>altitude</td><td>stale clear. (m)</td></tr><tr><td> $\mathrm { D } _ { 1 }$ </td><td>0.87</td><td>2.05</td><td>1.35</td><td>2.42</td><td>12.9</td><td>13.6</td><td>3.7</td><td>3.0</td><td>0.3</td><td>0.1</td><td>0.42</td><td>2.9</td></tr><tr><td>D2</td><td>1.03</td><td>1.98</td><td>1.26</td><td>1.59</td><td>11.8</td><td>14.1</td><td>3.7</td><td>3.1</td><td>0.1</td><td>0.1</td><td>0.32</td><td>2.9</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>1.06</td><td>2.07</td><td>1.32</td><td>1.66</td><td>12.6</td><td>17.5</td><td>4.1</td><td>3.1</td><td>0.1</td><td>0.1</td><td>0.37</td><td>2.9</td></tr><tr><td> $\mathrm { D } _ { 3 - { \overline { { a } } } , v }$ </td><td>0.91</td><td>1.72</td><td>1.11</td><td>1.35</td><td>11.1</td><td>14.4</td><td>2.6</td><td>2.6</td><td>0.0</td><td>0.0</td><td>0.33</td><td>3.0</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>0.96</td><td>2.22</td><td>1.49</td><td>1.19</td><td>11.9</td><td>16.5</td><td>4.0</td><td>3.2</td><td>0.0</td><td>0.0</td><td>0.36</td><td>2.9</td></tr><tr><td>D3,KW</td><td>0.98</td><td>2.20</td><td>1.42</td><td>1.59</td><td>13.5</td><td>16.9</td><td>4.5</td><td>3.2</td><td>0.0</td><td>0.1</td><td>0.39</td><td>2.9</td></tr></table>

Table 13: What the shielded missions did, by ending, for $\mathrm { D } _ { 1 }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ (N = 180 missions each; $n \mathrm { : }$ missions with that ending). Waypoint fraction, duration (time), mean speed and smallest privileged clearance are medians over the missions; interventions and the seconds with the barrier and the blind cap active are means per mission.
<table><tr><td colspan="3"></td><td colspan="4">Medians</td><td colspan="3">Per mission</td><td>Seconds</td></tr><tr><td></td><td>Ending</td><td>n</td><td>waypoints</td><td></td><td>time (s) speed (m/s)</td><td>clear. (m)</td><td>backoffs nudges</td><td></td><td>stalls</td><td>barrier / blind</td></tr><tr><td rowspan="5"> $\mathrm { D } _ { 1 }$ </td><td>success</td><td>94</td><td>1.00</td><td>83.1</td><td>1.24</td><td>0.60</td><td>0.50</td><td>1.36</td><td>0.86</td><td>11.7 / 11.4</td></tr><tr><td>timeout</td><td>42</td><td>0.71</td><td>120.3</td><td>0.85</td><td>0.53</td><td>1.17</td><td>4.26</td><td>3.10</td><td>17.7 / 25.0</td></tr><tr><td>collision</td><td>33</td><td>0.25</td><td>50.3</td><td>1.13</td><td>0.35</td><td>1.52</td><td>1.27</td><td>0.67</td><td>11.4/7.7</td></tr><tr><td>ground</td><td>7</td><td>0.50</td><td>56.0</td><td>1.08</td><td>0.40</td><td>1.43</td><td>2.57</td><td>1.14</td><td>12.8 / 8.1</td></tr><tr><td>camera timeout</td><td>4</td><td>0.17</td><td>28.2</td><td>1.26</td><td>1.50</td><td>0.00</td><td>0.50</td><td>0.50</td><td>1.7 /5.0</td></tr><tr><td rowspan="6"> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>success</td><td>90</td><td>1.00</td><td>81.5</td><td>1.28</td><td>0.63</td><td>0.51</td><td>1.50</td><td>0.99</td><td>12.7 / 16.1</td></tr><tr><td>timeout</td><td>39</td><td>0.67</td><td>112.5</td><td>0.87</td><td>0.48</td><td>1.56</td><td>4.26</td><td>2.72</td><td>21.0 / 26.8</td></tr><tr><td>collision</td><td>39</td><td>0.33</td><td>46.6</td><td>1.09</td><td>0.35</td><td>1.49</td><td>1.79</td><td>1.21</td><td>9.9 / 11.5</td></tr><tr><td>ground</td><td>8</td><td>0.12</td><td>24.4</td><td>1.25</td><td>0.39</td><td>1.25</td><td>2.12</td><td>0.88</td><td>7.7 / 6.9</td></tr><tr><td>camera timeout</td><td>2</td><td>0.08</td><td>39.4</td><td>1.05</td><td>0.96</td><td>0.00</td><td>0.50</td><td>0.50</td><td>3.2/ 6.3</td></tr><tr><td>out of bounds</td><td>2</td><td>0.53</td><td>46.6</td><td>1.68</td><td>0.50</td><td>0.50</td><td>3.50</td><td>3.00</td><td>3.0/ 13.8</td></tr></table>

## 6.8 Shield Activity, Cost, and What the Timeouts Did

Per $\mathrm { D } _ { 1 }$ mission (Table 12), the barrier modified the command for 12.9 s and the blind cap for 13.6 s; the shield backed off 0.87 times and nudged 2.05 times, 1.35 of them after a stall, for 3.7 and $3 . 0 \mathrm { s } ;$ it held 2.42 times, for 0.1 s in total; and the altitude floor acted for 0.3 s. The six checkpoints drew similar interventions: barrier 11.1–13.5 s and blind cap 13.6–17.5 s per mission. The memory update cost 2.9–3.0 ms at the 95th percentile, and the 95th percentile of decision compute was 2.6–3.7 ms higher than in the unshielded runs (separate runs, so this difference is approximate). The clearance to the memory, from the estimated position to the nearest remembered depth return (one per 0.15 m voxel), was 0.32–0.42 m, for each checkpoint 0.08–0.11 m below the privileged clearance to the modelled trunk surfaces (0.42–0.52 m); the two are measured to different targets (remembered returns from the estimated position, modelled cylinders from the true position) and are not directly comparable.

The timeouts were not hovering (Table 13). When the time limit ended them, $\mathrm { D } _ { 1 } \ ' _ { \mathrm { ~ S ~ } } 4 2$ timed-out missions had reached a median 71% of their waypoints, at a median mean speed of 0.85 m/s, with 3.10 stalls, 4.26 nudges and 25.0 s under the blind cap per mission, against 0.86 stalls, 1.36 nudges and 11.4 s for its successes. $\mathrm { D } _ { 3 , \mathrm { K W } } \mathbf { \dot { s } }$ 39 timeouts were similar (67% of waypoints, 0.87 m/s, 2.72 stalls, 4.26 nudges, 26.8 s blind). They ended without collision at the time limit, after slow progress that the shield repeatedly interrupted. Collisions came earlier (median 50.3 s for $\mathrm { D } _ { 1 } ,$ 46.6 s for $\mathrm { D } _ { 3 , \mathrm { K W } } )$ and after more backoffs (1.52 and 1.49 per mission).

## 7 Discussion

## 7.1 What the Evaluation Shows

The gain is uniform across recipes. The shield raised every checkpoint by a similar margin, 30.0–34.4 percentage points, although the recipes differ in what they copy (Section 6.2) and in their unshielded success (11.1–21.1%). It removed the failure that dominated all of them, collision with a trunk, whatever the recipe. The remaining differences between shielded checkpoints are within mission-level noise: none differed significantly from $\mathrm { D } _ { 1 } + \mathrm { s h i e l d } .$ , and even D and $\mathrm { D } _ { 3 , \mathrm { K W } }$ , both shielded, disagreed on 72 of the 180 missions (38 succeeded only with $\mathrm { D } _ { 1 } , 3 4$ only with $\mathrm { D } _ { 3 , \mathrm { K W } } )$ . In this study it was the runtime shield, not further imitation, that raised closed-loop success.

Failures move from collisions to timeouts. The shield is conservative by design: per $\mathrm { D } _ { 1 }$ mission the barrier acted for 12.9 s and the blind cap for 13.6 s, and the timeouts rose from 2.8% to 23.3%. The timeouts were slow, repeatedly interrupted progress rather than hovering: they reached a median 71% of their waypoints at 0.85 m/s with 4.26 nudges per mission, and spent 25.0 s per mission under the blind cap, more than twice as long as the successes (11.4 s). This is the price of map-free safety under a narrow field of view: the shield slows the vehicle wherever it cannot see where it is going. A policy that yawed toward its direction of travel, or a shield that turned the camera before moving sideways, may convert some timeouts into successes; we have not tested either.

Does the shield steer the copycat? Because the dispatched setpoint is written back as ${ { \bar { a } } _ { t } }$ , a policy that repeats ${ { \bar { a } } _ { t } }$ continues whatever the shield made it do: after a projection, its next command already points past the obstacle. We hypothesised that this write-back drives part of the gain. The data do not require it: $\mathrm { D } _ { 3 - \bar { a } }$ and $\mathrm { D } _ { 3 - \bar { a } , v }$ , whose actors do not receive ${ \bar { a } } _ { t } .$ , gained as much (32.8 and 31.7 points) as the checkpoints that do. $\mathrm { D } _ { 3 - \bar { a } }$ still copies the measured velocity, which the shield also changes, but $\mathrm { D } _ { 3 - \bar { a } , v } \mathrm { ~ \bar { s } ~ }$ actor receives neither quantity. Disabling the write-back (experiment F3, Section 8.4) would settle the question.

Remaining failure modes. With the shield, 22.2–42.8% of missions still ended in a crash, so the shield mitigates the narrow field of view and the actuation latency; it does not resolve them. In the shield $\mathbf { v } 2$ pilot, 7 of the 9 crashes happened while the shield was flying its own setpoint (backoff, nudge or hold) and 6 of the 7 collisions were with obstacles in view: the nudge moves toward the waypoint once the camera faces it, and the barrier is evaluated at the position 0.4 s ahead, which covers the dead time but not the 0.5 s lag. The two ground contacts descended at 2.8–3.3 m/s, faster than the vertical action bound, so the altitude floor engaged too late; in the benchmark, 5–13 of each checkpoint’s 180 shielded missions ended in ground contact. An altitude floor on the predicted height and a cap on the descent speed near the ground are the next changes.

Latent conditioning and actor sensitivity. In every evaluated policy the JEPA predictor has two operational roles: its ensemble mean $\hat { z } _ { t + 1 }$ conditions the solver (Eq. (15)), and its ensemble dispersion drives the uncertainty fallback (Section 3.5); it is not used for planning. The swap tests show how the actor uses these signals. Its command is far less sensitive to the predictor’s action conditioning than to direct proprioception: replacing ${ { \bar { a } } _ { t } }$ in the predictor’s input moved the command by 0.00–0.02 m/s, whereas replacing it in $S _ { t }$ alone moved it by 0.74 and 0.88 m/s in $\mathrm { D } _ { 3 , \mathrm { S D } }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ The predictor’s action pathway is nonetheless active: in $\mathrm { D } _ { 3 - \bar { a } , v } ,$ , whose actor receives neither ${ { \bar { a } } _ { t } }$ nor $v _ { t } ^ { \mathcal { H } } , \bar { a } _ { t }$ reached the command through the forecast (0.33 m/s). The swap test thus places the copycat shortcut in the actor’s proprioceptive input, where the round-3 recipes intervened, rather than in the latent forecast. The fallback is calibrated for about 0.05 spurious triggers per 60 s; at that threshold it braked or held for 0.1–0.3 s per shielded mission and was inactive in the last second before each of the 27 traced unshielded crashes, so the gain of Section 6.4 came from the runtime shield Ablation A6 (Section 8.4) is planned to measure the predictor’s contribution to closed-loop success.

Implications for the design. The proprioceptive input $S _ { t }$ of $\operatorname { E q . }$ (2) contains the last dispatched setpoint ${ { \bar { a } } _ { t } }$ so that the predictor is action-conditioned without placing the solver output on its input (Section 3.2). The evaluation shows that the same input gives the actor a shortcut: withholding ${ { \bar { a } } _ { t } }$ from the actor moved the shortcut to the measured velocity, and withholding both left an underfitted actor (Section 6.2). DAgger on the policy’s own states gave the first successes without removing the shortcut (Section 6.3). How the actor can use proprioception without copying it therefore remains a design question for Aero-VPA, separate from the shield, which only constrains what the actor commands.

## 7.2 Expected Engineering Hurdles

Sim-to-real gap in depth and appearance. Active stereo depth degrades outdoors when sunlight overwhelms the projector, on thin structures such as branches and cables, and at range, where the error grows roughly quadratically with distance. These effects bias both the encoder input and the braking envelope through $\epsilon _ { d } .$ . We plan range-dependent noise and dropout models in training, a conservative range-dependent $\epsilon _ { d }$ in Eq. (27), and fine-tuning of the stage-(ii) modules on a small amount of real flight data while keeping $E _ { \psi }$ frozen.

Aerodynamic effects. Ground effect near the floor and rubble [45], wall and ceiling effects in narrow openings, rotor downwash recirculating in confined rooms, and gusts [46] all perturb velocity tracking. They enlarge $T _ { \mathrm { t r k } }$ and may violate Assumption 3.1(iv), so the tracking lag and the attainable deceleration will be identified on the physical airframe and entered into $T _ { r }$ and $a _ { \mathrm { m a x } }$ with margins, and the flight envelope will be restricted near surfaces.

Embedded execution. Converting attention layers to FP16 can overflow or reorder reductions; we will verify the parity of the TensorRT engines with the PyTorch reference to a fixed tolerance on recorded inputs before any flight. In an enclosed airframe, the module may throttle thermally even within the 15 W mode, which changes $T _ { \mathrm { V P A } }$ over a mission; temperature and clock logging during long HITL runs will determine whether active cooling is required. Rolling-shutter distortion and motion blur at speed alter the input distribution in ways that simulation reproduces only approximately.

## 7.3 Planned Physical Flight Trials

Physical trials will proceed only if the HITL stage meets Hypothesis 1 and the safety layer meets Hypothesis 3 in simulation. They are staged as follows: (i) tethered and then free flight in a netted indoor arena with motion-capture ground truth, with the speed limit raised from 1 to 3 and then 5 m/s only after each level meets its criteria; (ii) static and moving foam obstacles in the arena to test the braking envelope and the fallback, including deliberately induced inference stalls and link interruptions; and (iii) outdoor flight in sparse and then dense forest with RTK-GNSS ground truth. Throughout, a safety pilot holds a radio override, the arena or field is geofenced, PX4’s offboard-loss failsafe remains active, and every flight is logged with the instrumentation of Section 5.2. Neither condition is met yet: Hypotheses 1 and 3 have not been tested, and 22.2–42.8% of the shielded missions in simulation still ended in a crash (Section 6.4). The runtime shield of Section 3.9, or a successor, would be part of any physical trial; on hardware it depth memory also depends on the drift of the state estimate over the 15 s horizon.

## 8 Limitations, Threats to Validity, and Protocol Deviations

## 8.1 Limitations of the Design

The ensemble of Eq. (22) measures disagreement among predictor heads that share a frozen encoder. It therefore captures predictor uncertainty but can miss failures of the encoder itself: an unfamiliar image mapped to a familiar latent yields confident, agreeing heads. In the evaluation, the fallback braked or held for 0.1–0.3 s per shielded mission and was active in the last second before none of the 27 traced unshielded crashes (Sections 6.4 and 6.7). The braking envelope partially mitigates this risk for static obstacles; latent-density scores and input-space OOD detectors are candidate complements to be compared within ablation A6. The guarantees stated here are conditional: Proposition 3.1 depends on Lipschitz constants that are not controlled, and Proposition 3.2 depends on static obstacles, bounded depth error, and accurate tracking, which the evaluated system does not provide (Remark 3.2); Proposition 3.3 concerns one remembered point under exact tracking, and the runtime shield carries no safety guarantee (Remark 3.3). The primitive dictionary bounds the repertoire of behaviours, and the rule-based labels of Section 3.3 may segment expert behaviour imperfectly. Finally, the uncertainty-adaptive horizon did not improve success in the manipulation experiments of [12]; whether it is beneficial in dynamic flight, where it acts as a staleness bound rather than an execution prefix, is an open question that ablation A8 is designed to answer. The runtime shield constrains the command only against remembered, static points, and it is conservative by design: it turned part of the crashes into timeouts (Section 7.1).

## 8.2 Threats to Validity

1. The shield was revised on test missions. Shield v2 was revised after the shield v1 pilot flew mission 0 of every test world, 36 of the 180 test missions. The 144 missions never used for revision give essentially the same result (D 52.1%, D 54.2%; Section 6.6), but they come from the same worlds. The fresh-world development set was not flown, so we make no claim of generalisation to unseen worlds.

2. Selection among six checkpoints. All six were evaluated on the same missions. We report $\mathrm { D } _ { 1 }$ throughout because it is the first DAgger checkpoint and the best unshielded one; it also had the highest shielded success, so its shielded number is subject to selection, although the paired tests against D + shield show that no checkpoint differs significantly from it.

3. One training seed, and chance. Each recipe was trained once and each evaluation flown once. The flowmatching noise is not seeded per mission and simulator timing varies, so mission outcomes have a large chance component (Section 6.3). Two repeated flights of mission 0 agreed in aggregate within one success (Section 6.7), but we did not measure test–retest agreement mission by mission.

4. A narrow scenario. Simulation only (PX4 SITL with Gazebo Harmonic); static, procedurally generated forests of vertical trunks; no wind, no moving obstacles and no visual distribution shift. The mean speeds of 1.16–1.36 m/s are far below the demonstrations’ speed limits of 3–7 m/s.

5. Privileged definitions. A collision is a privileged centre-to-surface distance below 0.35 m, and a ground contact an altitude below 0.3 m; neither is a physical contact model.

6. Compute. Timing was measured on an NVIDIA L4 GPU, not on the Jetson Orin Nano targeted in Section 4.

7. The BC baselines. The BC policies flew 12 of the 36 test worlds (60 missions), before the start-up warm-up fix, and ${ \mathrm { B C } } _ { - { \bar { a } } } { \mathrm { \Delta } } ^ { \circ } { \mathrm { s } }$ camera timeouts happened while four simulators shared the machine.

## 8.3 Deviations from the Protocol of Section 5

The evaluation departs from the protocol of Section 5, fixed in version 1, in eight ways. (i) Only the forest environment class was flown; the collapsed-indoor and dynamic-clutter classes, the out-of-distribution conditions and wind were not. (ii) Each policy flew 180 missions with one training seed, instead of 500 episodes per condition with three seeds. (iii) Results are not stratified by speed limit: the policies are not speed-conditioned, and the time limit uses a 3 m/s reference. (iv) The classical, reinforcement-learning and generative baselines B1–B3 were not run. (v) The embedded and hardware-in-the-loop latency of Hypothesis 1 was not measured; timing was measured on a cloud GPU (Section 4.6). (vi) The runtime shield is a post-hoc addition outside the pre-registered design, revised on test missions. (vii) Holm–Bonferroni correction across hypothesis families was not applied; instead we name one primary family of tests (Section 6.1) and report all others as secondary and unadjusted. (viii) Each DAgger round flew 201 missions under the safety-gated supervisor of Section 3.8, instead of the 500 flights per round planned in Section 5.1. Consequently none of Hypotheses 1–5 has been tested: the evaluation has no Jetson timing (Hypothesis 1), no baselines and no stratification by speed limit (Hypothesis 2), no out-of-distribution condition and no AUROC of σ¯ (Hypothesis 3), no reconstruction-trained encoder (Hypothesis 4) and no noise-persistence ablation (Hypothesis 5).

## 8.4 Experiments for Future Work

Experiments F1–F6 address the main threats above.

• F1: the shield with a non-learned goal-seeking controller (straight-to-waypoint velocity at about 1.2 m/s, yawing to face the waypoint). It isolates the learned actor’s contribution, which matters because all six shielded checkpoints landed within 42.8–52.2%.

• F2: the privileged expert alone on the 180 test missions, as an upper bound.

• F3: shield ablations: memory off (current frame only), blind cap off, backoff off, nudge off, and the write-back of the dispatched setpoint into ${ { \bar { a } } _ { t } }$ off.

• F4: the classical map-based stack and the Diffusion Policy baseline of Section 5.3 (B1 and B3a).

• F5: the world-model ablation A6 of Table 3 and the area under the ROC curve of σ¯ for collisions (Hypothesis 3).

• F6: several training seeds per recipe, and the fresh-world development set.

## 9 Conclusion

Aero-VPA replaces autoregressive action decoding with a short, fixed chain of small networks: a non-reconstructive JEPA world model, a discrete primitive selector, and a two-step flow-matching solver that produces a full velocity chunk in five sequential evaluations. It couples this chain to a safety layer driven by latent ensemble disagreement and by a model-free braking envelope, and to an asynchronous executor designed for the timing contract of the PX4 offboard interface. We have stated which properties follow from the architecture and which depend on assumptions about the learned components or the environment.

The first closed-loop evaluation, in simulation, showed that behaviour cloning learned a copycat shortcut, repeating its last command, that offline metrics could not detect, and that DAgger with a privileged, safety-gated supervisor produced the first successes but plateaued at 11.1–21.1%. A deterministic, map-free runtime shield raised success by 30.0–34.4 percentage points for all six checkpoints, to 52.2% for the first-round DAgger policy, and lowered its crashes from 72.2% to 22.2%, at the cost of more timeouts; the gain held on the missions never used to revise the shield. Whether the learned actor contributes under the shield, how the system performs in fresh worlds, at higher speeds and on the embedded module, and whether it transfers to hardware remain open; Section 5 and the experiments of Section 8.4 state how each will be tested, and their results, positive or negative, will determine whether the latency advantage of the approach translates into safer and faster flight.

## Code and Data Availability

The trained policy, the code and the benchmark dataset are public. All released results are from simulation (PX4 SITL with Gazebo Harmonic).

• Trained policy and model weights (GPL-3.0):

https://huggingface.co/ylee-sling/aero-vpa 88

The six DAgger checkpoints of Table 4, each a single file of 72.6–72.7 MB (the networks have 18.13M trainable parameters, Table 1); the three behaviour-cloning baselines; the two stage-(i) checkpoints; and six FPV videos of re-flights for illustration.

• Code (GPL-3.0-only), release v1.1.0 [28]:

https://github.com/ylee-sling/Aero-VPA

release page: https://github.com/ylee-sling/Aero-VPA/releases/tag/v1.1.0

The runtime shield evaluated here, sim/shield.py, is unchanged from commit 2aefde1 to this release.

• Benchmark dataset (CC BY 4.0) [29]:

https://huggingface.co/datasets/ylee-sling/aero-vpa-benchmark https://huggingface.co/datasets/ylee-sling/aero-vpa-benchmark

Every closed-loop mission record (a Parquet table and the raw records), the decision traces, the 36 test worlds with their obstacle maps, and the analysis reports.

• Version 1 of this preprint (method and pre-registered protocol): https://doi.org/10.5281/zenodo.23633375

With the dataset downloaded into bench/, the command

$$
\mathrm { p y t h o n 3 ~ \ s i m / p a i r e d \_ s t a t s . p y ~ \ s - r o o t ~ \ b e n c h / r e c o r d s }
$$

run in the repository recomputes the paired comparisons of the shielded benchmark: Tables 8, 10, 13, 15, 16 and 20, and the comparisons with D + shield in Table 17. The other numbers of Section 6 and of the appendices are in the analysis reports of the dataset. The expert demonstrations and validation flights are not released, so the swap tests of Tables 5 and 22 can be checked only against those reports.

## Acknowledgments

The author used an AI assistant (Claude, Anthropic) to help draft and edit the text and to check the reported statistics against the flight logs; the author reviewed all content and takes full responsibility for it.

## References

[1] Tong Qin, Peiliang Li, and Shaojie Shen. VINS-Mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics, 34(4):1004–1020, 2018.

[2] Armin Hornung, Kai M. Wurm, Maren Bennewitz, Cyrill Stachniss, and Wolfram Burgard. OctoMap: An efficient probabilistic 3D mapping framework based on octrees. Autonomous Robots, 34(3):189–206, 2013.

[3] Daniel Mellinger and Vijay Kumar. Minimum snap trajectory generation and control for quadrotors. In IEEE International Conference on Robotics and Automation (ICRA), pages 2520–2525, 2011.

[4] Xin Zhou, Zhepei Wang, Hongkai Ye, Chao Xu, and Fei Gao. EGO-Planner: An ESDF-free gradient-based local planner for quadrotors. IEEE Robotics and Automation Letters, 6(2):478–485, 2021.

[5] Antonio Loquercio, Elia Kaufmann, René Ranftl, Matthias Müller, Vladlen Koltun, and Davide Scaramuzza. Learning high-speed flight in the wild. Science Robotics, 6(59):eabg5810, 2021.

[6] Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Müller, Vladlen Koltun, and Davide Scaramuzza. Champion-level drone racing using deep reinforcement learning. Nature, 620:982–987, 2023.

[7] Yunlong Song, Angel Romero, Matthias Müller, Vladlen Koltun, and Davide Scaramuzza. Reaching the limit in autonomous racing: Optimal control versus reinforcement learning. Science Robotics, 8(82):eadg1462, 2023.

[8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[9] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[10] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[11] NVIDIA. NVIDIA JetPack 6.2 brings Super Mode to NVIDIA Jetson Orin Nano and Jetson Orin NX modules. NVIDIA Technical Blog, https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super -mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/, 2025. Accessed: 2026-10-25.

[12] Yeonseok Lee. Vision-Predictor-Action: Towards efficient embodied AI via neuro-symbolic JEPA predictors. arXiv preprint arXiv:2610.15556, 2026.

[13] Yann LeCun. A path towards autonomous machine intelligence. OpenReview preprint, version 0.9.2, 2022.

[14] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15619–15629, 2023.

[15] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

[16] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[17] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR), 2023.

[18] Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations (ICLR), 2023.

[19] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal ofRobotics Research, 44(10–11):1684–1704, 2025.

[20] PX4 Autopilot. Offboard mode (generic/all frames). PX4 User Guide, https://docs.px4.io/main/en/flig ht\_modes/offboard, 2026. Accessed: 2026-10-25.

[21] Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman, and Yang Gao. Fighting copycat agents in behavioral cloning from observation histories. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 2564–2575, 2020.

[22] Pim de Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[23] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), pages 627–635, 2011.

[24] Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada. Control barrier functions: Theory and applications. In European Control Conference (ECC), pages 3420–3431, 2019.

[25] Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations (ICLR), 2022.

[26] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[27] PX4 Autopilot. uXRCE-DDS (PX4–ROS 2/DDS bridge). PX4 User Guide, https://docs.px4.io/main/en middleware/uxrce\_dds, 2026. Accessed: 2026-10-25.

[28] SLING AI Inc. Aero-VPA: reference implementation, version v1.1.0. GitHub, https://github.com/ ylee-sling/Aero-VPA, 2026. Release page: https://github.com/ylee-sling/Aero-VPA/releases/ tag/v1.1.0. License: GPL-3.0-only.

[29] SLING AI Inc. Aero-VPA closed-loop benchmark (PX4 SITL, procedurally generated forests). Hugging Face, https://huggingface.co/datasets/ylee-sling/aero-vpa-benchmark, 2026. Dataset. License: CC BY 4.0.

[30] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025.

[31] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[32] Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations (ICLR), 2024.

[33] Aleksandar Vujinovic and Aleksandar Kovacevi ˇ c. ACT-JEPA: Novel joint-embedding predictive architecture for´ efficient policy representation learning. IEEE Access, 2026.

[34] Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. VLA-JEPA: Enhancing vision-language-action model with latent world model. arXiv preprint arXiv:2602.10098, 2026.

[35] Yeonseok Lee. NS-JEPA: Towards efficient embodied AI via neuro-symbolic joint-embedding predictive architec ture. Zenodo, https://doi.org/10.5281/zenodo.22493915, 2026.

[36] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018.

[37] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[38] Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency policy: Accelerated visuomotor policies via consistency distillation. In Robotics: Science and Systems (RSS), 2024.

[39] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems (RSS), 2024.

[40] Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2506.07339.

[41] Ruslan Rakhimov, George Bredis, Yuriy Maksyuta, and Daniil Gavrilov. Qantara: Bridge-flow training for multi-paradigm JEPA inference. arXiv preprint arXiv:2607.04978, 2026.

[42] Tong Qin, Jie Pan, Shaozu Cao, and Shaojie Shen. A general optimization-based framework for local odometry estimation with multiple sensors. arXiv preprint arXiv:1901.03638, 2019.

[43] Xin Zhou, Xiangyong Wen, Zhepei Wang, Yuman Gao, Haojia Li, Qianhao Wang, et al. Swarm of micro flying robots in the wild. Science Robotics, 7(66):eabm5954, 2022.

[44] Philipp Foehn, Elia Kaufmann, Angel Romero, Robert Penicka, Sihao Sun, Leonard Bauersfeld, et al. Agilicious: Open-source and open-hardware agile quadrotor for vision-based flight. Science Robotics, 7(67):eabl6259, 2022.

[45] Guanya Shi, Xichen Shi, Michael O’Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung. Neural lander: Stable drone landing control using learned dynamics. In IEEE International Conference on Robotics and Automation (ICRA), pages 9784–9790, 2019.

[46] Michael O’Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, and Soon-Jo Chung. Neural-Fly enables rapid learning for agile flight in strong winds. Science Robotics, 7(66):eabm6597, 2022.

[47] Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose, and Sergey Levine. GNM: A general navigation model to drive any robot. In IEEE International Conference on Robotics and Automation (ICRA), 2023.

[48] Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey Levine. NoMaD: Goal masked diffusion policies for navigation and exploration. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

[49] Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. Science Robotics, 5(47):eabc5986, 2020.

[50] Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. RMA: Rapid motor adaptation for legged robots. In Robotics: Science and Systems (RSS), 2021.

[51] Gabriel B. Margolis and Pulkit Agrawal. Walk these ways: Tuning robot control for generalization with multiplicity of behavior. In Conference on Robot Learning (CoRL), 2022.

[52] Lorenz Meier, Dominik Honegger, and Marc Pollefeys. PX4: A node-based multithreaded open source robotics framework for deeply embedded platforms. In IEEE International Conference on Robotics and Automation (ICRA), pages 6235–6240, 2015.

[53] Chuan Wen, Jierui Lin, Jianing Qian, Yang Gao, and Dinesh Jayaraman. Keyframe-focused visual imitation learning. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of PMLR, pages 11123–11133, 2021.

[54] Felipe Codevilla, Eder Santana, Antonio M. López, and Adrien Gaidon. Exploring the limitations of behavior cloning for autonomous driving. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 9329–9338, 2019.

[55] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst. In Robotics: Science and Systems (RSS), 2019.

[56] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by cheating. In Conference on Robot Learning (CoRL), volume 100 of PMLR, pages 66–75, 2019.

[57] Jiakai Zhang and Kyunghyun Cho. Query-efficient imitation learning for end-to-end simulated driving. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.

[58] Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. HG-DAgger: Interactive imitation learning with human experts. In IEEE International Conference on Robotics and Automation (ICRA), pages 8077–8083, 2019.

[59] Mohammed Alshiekh, Roderick Bloem, Rüdiger Ehlers, Bettina Könighofer, Scott Niekum, and Ufuk Topcu. Safe reinforcement learning via shielding. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 2669–2678, 2018.

[60] Kim P. Wabersich, Andrew J. Taylor, Jason J. Choi, Koushil Sreenath, Claire J. Tomlin, Aaron D. Ames, and Melanie N. Zeilinger. Data-driven safety filters: Hamilton-Jacobi reachability, control barrier functions, and predictive methods for uncertain systems. IEEE Control Systems Magazine, 43(5):137–177, 2023.

[61] Kai-Chieh Hsu, Haimin Hu, and Jaime F. Fisac. The safety filter: A unified view of safety-critical control in autonomous systems. Annual Review ofControl, Robotics, and Autonomous Systems, 7:47–72, 2024.

[62] Peter R. Florence, John Carter, Jake Ware, and Russ Tedrake. NanoMap: Fast, uncertainty-aware proximity queries with lazy search over local 3D data. In IEEE International Conference on Robotics and Automation (ICRA), 2018.

[63] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[64] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), pages 8748–8763, 2021.

[65] Hyunjik Kim, George Papamakarios, and Andriy Mnih. The Lipschitz constant of self-attention. In International Conference on Machine Learning (ICML), pages 5562–5571, 2021.

[66] Steven Macenski, Tully Foote, Brian Gerkey, Chris Lalancette, and William Woodall. Robot Operating System 2: Design, architecture, and uses in the wild. Science Robotics, 7(66):eabm6074, 2022.

[67] Nathan Koenig and Andrew Howard. Design and use paradigms for Gazebo, an open-source multi-robot simulator. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2149–2154, 2004.

[68] NVIDIA. Isaac Sim. https://developer.nvidia.com/isaac/sim, 2026. Accessed: 2026-10-25.

[69] Marcelo Jacinto, João Pinto, Jay Patrikar, et al. Pegasus Simulator: An Isaac Sim framework for multiple aerial vehicles simulation. In International Conference on Unmanned Aircraft Systems (ICUAS), 2024.

[70] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 23–30, 2017.

[71] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[72] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning (ICML), pages 1861–1870, 2018.

[73] Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, et al. Orbit: A unified simulation framework for interactive robot learning environments. IEEE Robotics and Automation Letters, 8(6):3740–3747, 2023.

[74] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[75] Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal ofthe American Statistical Association, 22(158):209–212, 1927.

[76] Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

[77] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979.

## A Hyperparameters

Table 14 lists every value used in the reported flights.

## B DAgger Supervisor Details

Labels. Every frame of a DAgger round, flown by the policy or by the expert, is labelled with the expert’s tracking law, Eq. (31), on its minimum-snap plan of the mission, from the path point τ<sup>⋆</sup> nearest the vehicle. The speed limit of each mission is drawn from $\{ 3 , 5 , { \dot { 7 } } \}$ m/s, as in the demonstrations. On the path $( e \le 0 . 3  { \mathrm { m } } )$ the law is the demonstrations law $( w = 1 )$ ; further off, the expert rejoins first $( w  0$ , horizontal speed limit falling to 2 m/s at $e \geq 1 { \mathfrak { m } } )$

Gate. The policy flies until one of six triggers fires: (1) obstacle: the straight look-ahead along the current velocity, over 0.3 s of reaction plus the braking distance and at least 1.5 m, sampled every 0.1 m, comes within 0.5 m + 0.06 m per m/s of speed of a trunk; (2) low: altitude below 0.8 m; (3) off path: more than 1.5 m from the path; (4) heading: facing more than 75° away from the path; (5) a missed waypoint; (6) stall: 5 s without progress.

Takeover and hand-back. In a soft takeover, which keeps recording frames, the expert chooses how to fly on from $\tau ^ { \star } \colon$ along its plan as is, slowing down along its path with a time-scaled reference, or with a straight stop, by the clearance that a rollout of the delayed closed loop predicts for each option. The rollout replays the setpoints still in flight over a dead time of 0.24 s and models the vehicle’s response as first order with a 0.5 s lag. A missed waypoint, or a predicted conflict that only a straight stop resolves, leads to a hard takeover: the expert brakes, backs off and re-plans; these frames are not recorded. Control returns to the policy after 2 s on track.

Rounds. Each round flies 201 missions over the training worlds; stage (ii) then trains on the demonstrations and all rounds so far, with DAgger frames weighted $\times 3 \ : ( \times 1$ for $\mathrm { D } _ { 3 , \mathrm { K W } } )$ . Round 1 was flown by BC, round 2 by $\mathrm { D } _ { 1 }$ and round 3 by $\operatorname { D } _ { 2 } ;$ the round-3 recipes $\mathrm { D } _ { 3 - \bar { a } } , \mathrm { D } _ { 3 - \bar { a } , v } , \mathrm { D } _ { 3 , \mathrm { S D } }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ were trained on rounds 1–3 (Table 4).

## C Full Results

Table 15 gives the exact ending counts of the twelve evaluations, Table 16 the paired crash comparisons, and Table 17 the shielded checkpoints paired with unshielded $\mathrm { D } _ { 1 }$ and with $\mathrm { D } _ { 1 }$ + shield. Tables 18 and 19 give the secondary metrics, and Table 20 the shielded endings by density.

## D Behaviour-Cloning Results

Table 21 gives the endings of the three BC policies on 60 missions (12 of the 36 test worlds, 5 missions each). They were flown before a start-up warm-up fix: the first decision of a fresh process delayed the setpoint stream at the start of the first mission of each world. ${ \bf B } \bar { \bf C } _ { - \bar { a } } \mathrm { ^ { ~ s ~ } }$ camera timeouts happened while four simulators shared the machine. The open-loop errors and selector agreement of BC and $\mathrm { B C } _ { - \bar { a } }$ on expert frames are in Table 5.

## E Input-Sensitivity Tests of the Round-3 Recipes

Table 22 gives the complete swap tests of $\mathrm { D } _ { 3 , \mathrm { S D } }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ on the 5000 held-out expert frames. Both still follow the donor’s ${ { \bar { a } } _ { t } } \mathrm { { : } }$ their command after the swap is closer to the donor frame’s label (0.28 and 0.43 m/s) than to their own

Table 14: Values used in every reported flight.
<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td>Observation</td><td>RGB-D frame; camera</td><td> $4 \times 1 2 8 \times 1 6 0 ; \mathrm { { H F O V } } 1 . 2 7 4 { \mathrm { r a d } } ( 7 3 ^ { \circ } ) , \mathrm { { d e p t h } }$   $0 . 1 { - } 1 0 \mathrm { m }$   $\mathbb { R } ^ { 1 3 } ; \mathbb { R } ^ { 4 } .$ </td></tr><tr><td>Action</td><td>proprioception  $S _ { t } ;$  goal  $g _ { t }$  waypoint acceptance  $\tau _ { w } ;$  instruction chunk  $H ,$  knot spacing  $\Delta$ </td><td>, distance saturated at 20 m 1 m; CLIP ViT-B/32 text, 512-d (null embedding) 16 knots, 0.02 s (0.32 s)</td></tr><tr><td> $E _ { \psi } \left( 1 1 . 1 7 \mathbf { M } \right)$ </td><td>bounds ViT</td><td>7 m/s horizontal, 2 m/s vertical, 1.5 rad/s patch 16, 80 tokens + CLS, width 384, depth</td></tr><tr><td> $P _ { \omega } \left( 2 . 7 2 \mathbf { M } \right)$ </td><td>ensemble</td><td>heads, z ∈ R256  ${ 5 } \mathrm { M L P }$  heads  $2 9 2  5 1 2  5 1 2  2 5 6 .$  residual,</td></tr><tr><td></td><td></td><td> $\mathrm { E m b e d } ( u ) \in \mathbb { R } ^ { 3 2 } , 0 . 1$  s ahead  $\mathrm { M L P 7 8 5 \to 5 1 2 \to 5 1 2 \to 6 }$ </td></tr><tr><td> $\pi _ { \phi } ^ { h } ~ ( 0 . 6 7 \mathbf { M } )$  vθ (3.57M)</td><td>selector solver</td><td>pre-LN Transformer, 4 layers, width 256, 4 heads, 17</td></tr><tr><td></td><td>sampling</td><td>tokens;  $e _ { t } \in \mathbb { R } ^ { 1 0 7 3 }$  linear path,  $K = 2$  Euler steps, noise persistence</td></tr><tr><td>Stage (i)</td><td>VICReg weights; EMA momentum</td><td> $N _ { \xi } = 3 0$   $\lambda _ { v } = 2 5 , \lambda _ { c } = 1 ; 0 . 9 9 6$ </td></tr><tr><td>Stage (ii)</td><td>steps, batch, time</td><td>153,528 steps, batch 1024, about 2 h 40 min on one</td></tr><tr><td></td><td></td><td>L4</td></tr><tr><td></td><td>DAgger frame weight state dropout  $\left( \mathrm { D } _ { 3 , \mathrm { S D } } \right)$ </td><td> $\times 3 \left( \times 1 \mathrm { f o r } \mathrm { D } _ { 3 , \mathrm { K W } } \right)$   $v ^ { \mathcal { H } }$  and ā set to their training mean in 50% of frames</td></tr><tr><td></td><td>keyframe weighting  $( \mathrm { D } _ { 3 , \mathrm { K W } } )$ </td><td> $\propto \mathrm { m i n } ( 1 + \left. A _ { t } [ 0 ] - \bar { a } _ { t } \right. / 0 . 1 , 1 0 )$ </td></tr><tr><td>Learned safety</td><td>uncertainty filter; thresholds</td><td> $\alpha _ { \sigma } = 0 . 1 ; \sigma _ { \mathrm { l o } } \colon$  0.9-quantile;  $\sigma _ { \mathrm { c r i t } } \mathrm { : }$  per-flight</td></tr><tr><td></td><td></td><td>maxima on 360 validation flights, about 0.05 spurious fallbacks per 60 s</td></tr><tr><td></td><td>attenuation; braking envelope</td><td> $\mathrm { f a c t o r } \geq 0 . 5 ; r _ { s } = 0 . 5 \mathrm { m } , a _ { \mathrm { m a x } } = 5 \mathrm { m } / \mathrm { s } ^ { 2 } ,$   $\epsilon _ { d } = 0 . 2 \mathbf { m } , T _ { r } = 0 . 0 6 + 0 . 0 2 H _ { t } + 0 . 1 5 \mathrm { s } ,$ </td></tr><tr><td>Execution</td><td>executor; cross-fade</td><td> $v _ { \mathrm { b l i n d } } = 1 \mathrm { m } / \mathrm { s }$  100 Hz; 0.04 s</td></tr><tr><td>Shield memory</td><td>blocks; returns; ground</td><td> $4 \times 4$  pixels, nearest return; 0.2–8 m; points below</td></tr><tr><td></td><td>voxel; age; extent; local set</td><td>0.25 m dropped 0.15 m (newest point kept); 15 s; 10 m horizontally;</td></tr><tr><td></td><td></td><td>3.5 m</td></tr><tr><td>Barrier</td><td>look-ahead; points; radius  $\beta ;$  enforcement</td><td>0.4 s at the flown velocity; 24 nearest;  $r = 0 . 5 5 \mathrm { m }$   $k = 1 . 5 \mathrm { s } ^ { - 1 } , a _ { b } = 3 \mathrm { m / s } ^ { 2 } , v _ { \mathrm { p u s h } } = 0 . 5 \mathrm { m / s } ;$  two</td></tr><tr><td></td><td></td><td>projection sweeps 0.7 m/s beyond  $3 6 . 5 ^ { \circ }$  of the camera axis, unless</td></tr><tr><td>Caps and holds</td><td>blind cap</td><td>within  $3 5 ^ { \circ } \mathrm { o f }$  a breadcrumb 0.3–2.5 m away</td></tr><tr><td></td><td>altitude floor; stale hold; uncertainty cap</td><td>climb  $\mathrm { a t } \geq 0 . 5 \mathrm { m / s }$  below 0.8 m; zero velocity after 0.3 s without a decision; off  $( v _ { \sigma } = \infty ; { \mathrm { s h i e l d v 1 } } ;$ </td></tr><tr><td>Recovery</td><td>breadcrumbs; backoff</td><td>1 m/s) every 0.25 m; below 0.45 m clearance, retrace up to</td></tr><tr><td></td><td></td><td>1 m at 0.3 m/s, end if blocked  $( \Vert u \Vert < 0 . 0 7 5$  m/s) or at clearance  $\geq r + 0 . 4  { \mathrm { m } }$   $\dot { \psi } = \mathrm { c l i p } ( 1 . 5 \Delta \psi , \pm 0 . 8 \mathrm { r a d } / \mathrm { s } )$  , then 0.6 cos ∆ψ m/s</td></tr><tr><td></td><td>nudge; stall</td><td> $\mathrm { f o r } 1 \mathrm { s } ; 5 \mathrm { s }$  without 0.3 m of progress</td></tr><tr><td>DAgger</td><td>label law</td><td> $k _ { p } = 1 . 5 \mathrm { s } ^ { - 1 }$  ; on path within 0.3 m; 2 m/s horizontal limit at ≥ 1 m off every</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>2.5 m; 3–6, 10–40 m apart  $1 . 5 T _ { \mathrm { r e f } } + 1 5 \mathrm { s } , T _ { \mathrm { r e f } }$  at 3 m/s</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>gate</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>look-ahead  $0 . 3 \mathrm { s } + $  braking distance  $( \geq 1 . 5 \ : \mathrm { m } ,$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $0 . 1 \mathrm { m } ) , 0 . 5 \mathrm { m } + 0 . 0 6$  m per m/s; altitude 0.8 m; off</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>path 1.5 m; heading  $7 5 ^ { \circ } ;$  stall 5 s</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.24 s dead time, 0.5 s lag; 2 s on track; 201 missions</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>rollout; hand-back; round size</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Missions</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>take-off; waypoints</td><td></td></tr></table>

Table 15: Exact ending counts of the six DAgger checkpoints, unshielded (U) and shielded (S), on the same $N = 1 8 0$ test missions.
<table><tr><td>Checkpoint</td><td></td><td>Success</td><td>Collision</td><td>Ground</td><td>Timeout</td><td>Camera timeout</td><td>Out of bounds</td></tr><tr><td rowspan="2"> $\mathrm { D } _ { 1 }$ </td><td>U</td><td>38</td><td>114</td><td>16</td><td>5</td><td>7</td><td>0</td></tr><tr><td>S</td><td>94</td><td>33</td><td>7</td><td>42</td><td>4</td><td>0</td></tr><tr><td rowspan="2"> $\mathrm { D } _ { 2 }$ </td><td>U</td><td>23</td><td>133</td><td>11</td><td>2</td><td>10</td><td>1</td></tr><tr><td>S</td><td>77</td><td>64</td><td>5</td><td>30</td><td>4</td><td>0</td></tr><tr><td rowspan="2"> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>U</td><td>20</td><td>131</td><td>17</td><td>5</td><td>7</td><td>0</td></tr><tr><td>S</td><td>79</td><td>55</td><td>10</td><td>26</td><td>7</td><td>3</td></tr><tr><td rowspan="2"> $\mathrm { D } _ { 3 - { \bar { a } } , v }$ </td><td>U</td><td>24</td><td>126</td><td>13</td><td>6</td><td>11</td><td>0</td></tr><tr><td>S</td><td>81</td><td>64</td><td>13</td><td>18</td><td>4</td><td>0</td></tr><tr><td rowspan="2"> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>U</td><td>28</td><td>129</td><td>8</td><td>6</td><td>8</td><td>1</td></tr><tr><td>S</td><td>82</td><td>51</td><td>12</td><td>29</td><td>5</td><td>1</td></tr><tr><td rowspan="2"> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>U</td><td>28</td><td>136</td><td>9</td><td>3</td><td>4</td><td>0</td></tr><tr><td>S</td><td>90</td><td>39</td><td>8</td><td>39</td><td>2</td><td>2</td></tr></table>

Table 16: Paired crash comparisons, shielded against unshielded $( N = 1 8 0$ paired missions): crashes in each condition, discordant pairs and the two-sided exact McNemar $p .$
<table><tr><td>Checkpoint</td><td>Crashes U</td><td>Crashes S</td><td>Only U crashed</td><td>Only S crashed</td><td> $p$ </td></tr><tr><td>D1</td><td>130</td><td>40</td><td>98</td><td>8</td><td> $8 . 0 8 \times 1 0 ^ { - 2 1 }$ </td></tr><tr><td> $\mathrm { D } _ { 2 }$ </td><td>144</td><td>69</td><td>83</td><td>8</td><td> $7 . 5 8 \times 1 0 ^ { - 1 7 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>148</td><td>65</td><td>93</td><td>10</td><td> $5 . 2 0 \times 1 0 ^ { - 1 8 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 - { \bar { a } } , v }$ </td><td>139</td><td>77</td><td>78</td><td>16</td><td> $5 . 7 8 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>137</td><td>63</td><td>83</td><td>9</td><td> $3 . 9 2 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>145</td><td>47</td><td>107</td><td>9</td><td> $2 . 0 0 \times 1 0 ^ { - 2 2 }$ </td></tr></table>

(0.84 and 0.89 m/s), almost entirely through $S _ { t }$ rather than through the predictor. On the same frames, the expert’s commanded direction was blind in 3.7% of frames and the braking envelope throttled it in 1.4%; for $\mathrm { D } _ { 3 , \mathrm { S D } }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ the shares were 5.5% and 5.1% blind and 8.5% and 8.8% throttled.

Table 17: Each shielded checkpoint paired with unshielded $\mathrm { D } _ { 1 }$ and with $\mathrm { D } _ { 1 }$ + shield (N = 180 paired missions; both / only $\mathrm { D } _ { 1 } /$ only the other / neither, two-sided exact McNemar p).
<table><tr><td rowspan="2">Shielded checkpoint</td><td colspan="2">Against unshielded  $\mathrm { D } _ { 1 }$ </td><td colspan="2">Against D1 + shield</td></tr><tr><td>counts</td><td> $p$ </td><td>counts</td><td>p</td></tr><tr><td> $\mathrm { D } _ { 1 }$ </td><td> $2 6 / 1 2 / 6 8 / 7 4$ </td><td> $1 . 2 0 \times 1 0 ^ { - 1 0 }$ </td><td></td><td></td></tr><tr><td>D2</td><td> $1 7 / 2 1 / 6 0 / 8 2$ </td><td> $1 . 6 9 \times 1 0 ^ { - 5 }$ </td><td> $4 6 / 4 8 / 3 1 / 5 5$ </td><td>0.0712</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>17 / 21 / 62 / 80</td><td> $7 . 5 1 \times 1 0 ^ { - 6 }$ </td><td> $5 0 / 4 4 / 2 9 / 5 7$ </td><td>0.101</td></tr><tr><td> $\mathrm { D } _ { 3 - { \bar { a } } , v }$ </td><td>19 / 19 / 62 / 80</td><td> $1 . 7 7 \times 1 0 ^ { - 6 }$ </td><td> $5 2 / 4 2 / 2 9 / 5 7$ </td><td>0.154</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>22 / 16 / 60 / 82</td><td> $3 . 8 5 \times 1 0 ^ { - 7 }$ </td><td> $4 7 / 4 7 / 3 5 / 5 1$ </td><td>0.224</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>22 / 16 / 68 / 74</td><td> $8 . 5 4 \times 1 0 ^ { - 9 }$ </td><td> $5 6 / 3 8 / 3 4 / 5 2$ </td><td>0.724</td></tr></table>

Table 18: Secondary metrics without the shield (N = 180 missions per checkpoint). Duration, path, time and path ratios, smallest clearance and timing are medians over missions (time and path ratios over successful missions); waypoints and blind decisions are pooled shares; fallback figures and mean speed are means over missions.

<table><tr><td>Metric</td><td> $\mathrm { D } _ { 1 }$ </td><td> $\mathrm { D } _ { 2 }$ </td><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td> $\mathrm { D } _ { 3 - { \bar { a } } , \ v }$ </td><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>D3,KW</td></tr><tr><td>Waypoints reached</td><td>38.5%</td><td>28.9%</td><td>27.6%</td><td>31.3%</td><td>29.8%</td><td>31.9%</td></tr><tr><td>Duration / path (s, m)</td><td>38.7 / 46.1</td><td>28.9 / 35.1</td><td>30.6 / 33.3</td><td>32.7 / 43.5</td><td>30.1 / 41.4</td><td>35.5 / 38.6</td></tr><tr><td>Fallback onsets per mission</td><td>0.77</td><td>0.51</td><td>1.04</td><td>0.68</td><td>0.69</td><td>0.86</td></tr><tr><td>Missions with a fallback</td><td>21.1%</td><td>21.1%</td><td>30.6%</td><td>23.3%</td><td>26.1%</td><td>31.1%</td></tr><tr><td>Braking or holding (s per mission)</td><td>0.1</td><td>0.1</td><td>0.2</td><td>0.1</td><td>0.1</td><td>0.2</td></tr><tr><td>Time / expert reference</td><td>1.15</td><td>1.00</td><td>1.26</td><td>1.05</td><td>1.10</td><td>1.10</td></tr><tr><td>Path / mission length</td><td>1.09</td><td>1.12</td><td>1.24</td><td>1.16</td><td>1.12</td><td>1.13</td></tr><tr><td>Mean speed (m/s)</td><td>1.24</td><td>1.21</td><td>1.21</td><td>1.37</td><td>1.26</td><td>1.21</td></tr><tr><td>Smallest clearance (m)</td><td>0.35</td><td>0.35</td><td>0.35</td><td>0.35</td><td>0.35</td><td>0.35</td></tr><tr><td>Blind decisions</td><td>26.2%</td><td>29.5%</td><td>33.2%</td><td>31.9%</td><td>31.3%</td><td>34.0%</td></tr><tr><td>Decision compute p95 (ms)</td><td>29.4</td><td>28.8</td><td>29.1</td><td>29.4</td><td>29.4</td><td>29.6</td></tr><tr><td>Decision age p95 (ms)</td><td>41.1</td><td>40.4</td><td>41.3</td><td>41.0</td><td>40.6</td><td>41.3</td></tr><tr><td>Longest setpoint gap, median / max (s)</td><td>0.10 / 0.15</td><td>0.10 / 0.15</td><td>0.10 / 0.15</td><td>0.10 / 0.16</td><td>0.09 / 0.15</td><td>0.10 / 0.16</td></tr></table>

Table 19: Secondary metrics with the shield (N = 180 missions per checkpoint; definitions as in Table 18).
<table><tr><td>Metric</td><td> $\mathrm { D } _ { 1 }$ </td><td> $\mathrm { D } _ { 2 }$ </td><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td> $\mathrm { D } _ { 3 - { \bar { a } } , \ v }$ </td><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>D3,KW</td></tr><tr><td>Waypoints reached</td><td>76.6%</td><td>68.4%</td><td>65.9%</td><td>62.2%</td><td>68.9%</td><td>72.9%</td></tr><tr><td>Duration / path (s, m)</td><td>86.3 / 95.4</td><td>77.9 / 88.0</td><td>76.1 / 90.5</td><td>60.7 / 81.2</td><td>77.6 / 90.0</td><td>82.2 / 93.5</td></tr><tr><td>Fallback onsets per mission</td><td>1.53</td><td>1.13</td><td>1.39</td><td>1.04</td><td>1.32</td><td>1.25</td></tr><tr><td>Missions with a fallback</td><td>30.0%</td><td>32.2%</td><td>34.4%</td><td>33.9%</td><td>36.7%</td><td>34.4%</td></tr><tr><td>Braking or holding (s per mission)</td><td>0.3</td><td>0.2</td><td>0.2</td><td>0.1</td><td>0.2</td><td>0.2</td></tr><tr><td>Time / expert reference</td><td>1.26</td><td>1.22</td><td>1.27</td><td>1.12</td><td>1.22</td><td>1.22</td></tr><tr><td>Path / mission length</td><td>1.13</td><td>1.14</td><td>1.19</td><td>1.18</td><td>1.19</td><td>1.16</td></tr><tr><td>Mean speed (m/s)</td><td>1.18</td><td>1.17</td><td>1.16</td><td>1.36</td><td>1.20</td><td>1.20</td></tr><tr><td>Smallest clearance (m)</td><td>0.52</td><td>0.42</td><td>0.45</td><td>0.43</td><td>0.46</td><td>0.50</td></tr><tr><td>Blind decisions</td><td>28.8%</td><td>32.6%</td><td>36.4%</td><td>30.7%</td><td>35.8%</td><td>35.8%</td></tr><tr><td>Decision compute p95 (ms)</td><td>33.1</td><td>32.4</td><td>32.2</td><td>32.0</td><td>32.1</td><td>32.4</td></tr><tr><td>Decision age p95 (ms)</td><td>44.6</td><td>43.8</td><td>43.6</td><td>43.5</td><td>43.8</td><td>43.8</td></tr><tr><td>Longest setpoint gap, median / max (s)</td><td>0.10 / 0.18</td><td>0.10 / 0.16</td><td>0.10 / 0.15</td><td>0.10 / 0.13</td><td>0.10 / 0.17</td><td>0.10 / 0.16</td></tr></table>

Table 20: Shielded endings by density: successes / crashes / timeouts of N = 60 missions per density; the remainder of each 60 are camera timeouts or out-of-bounds endings.
<table><tr><td>Checkpoint</td><td>Dense</td><td>Medium</td><td>Sparse</td></tr><tr><td> $\mathrm { D } _ { 1 }$ </td><td>28 / 17 / 15</td><td>29 / 16 / 15</td><td>37 / 7 / 12</td></tr><tr><td>D2</td><td>21 / 29 / 10</td><td>32 / 19 / 9</td><td>24 / 21 / 11</td></tr><tr><td> $\mathrm { D } _ { 3 - \bar { a } }$ </td><td>23 / 24 / 12</td><td>25 / 28 / 6</td><td>31 / 13 / 8</td></tr><tr><td> $\mathrm { D } _ { 3 - { \bar { a } } , v }$ </td><td>17 / 31 / 12</td><td>28 / 30 / 2</td><td>36 / 16 / 4</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td>17 / 31 / 12</td><td>26 / 21 / 11</td><td>39 / 11 / 6</td></tr><tr><td> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td><td>22 / 20 / 18</td><td>33 / 12 / 15</td><td>35 / 15 / 6</td></tr></table>

Table 21: Behaviour-cloning policies in closed loop $( N = 6 0$ missions each: 12 of the 36 test worlds, 5 missions per world, flown before the warm-up fix). Crash shares are split into collision and ground contact.
<table><tr><td>Policy</td><td>Stage (i)</td><td>Success</td><td>Crash (collision / ground)</td><td>Other endings</td></tr><tr><td> ${ \mathrm { B C } } _ { \mathrm { N R } }$ </td><td>non-residual</td><td>0/60</td><td>83% (48% / 35%)</td><td>10 missions, not broken down</td></tr><tr><td>BC</td><td>residual</td><td>0/60</td><td>51.7% (42% / 10%)</td><td>timeout 48.3%</td></tr><tr><td> $\mathrm { B C } _ { - \bar { a } }$ </td><td>residual</td><td>0/60</td><td>70.0% (22% / 48%)</td><td>timeout 8.3%, camera timeout 21.7%</td></tr></table>

Table 22: Swap tests of $\mathrm { D } _ { 3 , \mathrm { S D } }$ and $\mathrm { D } _ { 3 , \mathrm { K W } }$ on $N = 5 0 0 0$ held-out expert frames: change of the first knot $v _ { 0 }$ (median / p90, m/s), share of frames whose primitive changed, and the error to the frame’s own label and to the donor frame’s label (m/s). $\mathbf { \tilde { \Delta } } \bar { a } _ { t } = 0 \mathbf { \vec { \Delta } } :$ : the last command set to zero (first-knot speed afterwards in parentheses). Open loop on the same frames: $| v _ { 0 } - v _ { 0 } ^ { \star } |$ median / p90 0.04 / 0.10 m/s $\left( \mathrm { D } _ { 3 , \mathrm { S D } } \right)$ and $0 . { \overset { - } { 0 } } 3 / 0 . 0 7$ m/s $( \mathrm { D } _ { 3 , \mathrm { K W } } ) ;$ selector agreement 91.8% and 92.9%; direction error above $4 5 ^ { \circ }$ in 0.6% and 0.4% of frames.
<table><tr><td rowspan="2">Swapped input</td><td colspan="4"> $\mathrm { D } _ { 3 , \mathrm { S D } }$ </td><td colspan="4"> $\mathrm { D } _ { 3 , \mathrm { K W } }$ </td></tr><tr><td>change</td><td>primitive</td><td>own</td><td>donor</td><td>change</td><td>primitive</td><td>own</td><td>donor</td></tr><tr><td>none</td><td>0.00 / 0.00</td><td>0.0%</td><td>0.04</td><td>1.14</td><td>0.00 / 0.00</td><td>0.0%</td><td>0.03</td><td>1.14</td></tr><tr><td>image  $z _ { t }$ </td><td>0.04 / 0.13</td><td>12.2%</td><td>0.04</td><td>1.14</td><td>0.03 / 0.08</td><td>10.4%</td><td>0.03</td><td>1.15</td></tr><tr><td>goal  $g _ { t }$ </td><td>0.03 / 0.40</td><td>16.9%</td><td>0.05</td><td>1.09</td><td>0.02 / 0.09</td><td>9.0%</td><td>0.03</td><td>1.14</td></tr><tr><td>velocity  $v _ { t } ^ { \mathcal { H } }$ </td><td>0.16 / 1.54</td><td>12.9%</td><td>0.18</td><td>0.98</td><td>0.49 / 1.85</td><td>28.9%</td><td>0.50</td><td>0.97</td></tr><tr><td>attitude  $n _ { t } ^ { B }$ </td><td>0.05 / 0.89</td><td>17.3%</td><td>0.07</td><td>1.16</td><td>0.04 / 0.57</td><td>12.3%</td><td>0.05</td><td>1.16</td></tr><tr><td>body rates</td><td>0.05 / 0.87</td><td>20.8%</td><td>0.07</td><td>1.16</td><td>0.02 / 0.27</td><td>11.2%</td><td>0.04</td><td>1.15</td></tr><tr><td>last command  ${ { \bar { a } } _ { t } }$ </td><td>0.83 / 2.21</td><td>30.1%</td><td>0.84</td><td>0.28</td><td>0.88 / 2.27</td><td>43.1%</td><td>0.89</td><td>0.43</td></tr><tr><td> ${ { \bar { a } } _ { t } }$  in  $S _ { t }$  only</td><td>0.74 / 2.05</td><td>30.1%</td><td>0.75</td><td>0.34</td><td>0.88 / 2.26</td><td>43.1%</td><td>0.89</td><td>0.43</td></tr><tr><td> ${ { \bar { a } } _ { t } }$  to  $P _ { \omega }$  only</td><td>0.02 / 0.12</td><td>0.0%</td><td>0.05</td><td>1.11</td><td>0.00 / 0.01</td><td>0.0%</td><td>0.03</td><td>1.14</td></tr><tr><td> $\bar { a } _ { t } = 0$ </td><td>0.84 / 2.19</td><td>22.0%</td><td>0.86</td><td>(1.31)</td><td>0.83 / 2.06</td><td>40.1%</td><td>0.83</td><td>(1.47)</td></tr><tr><td>all of  $S _ { t }$ </td><td>1.09 / 2.89</td><td>40.6%</td><td>1.09</td><td>0.05</td><td>1.14 / 2.96</td><td>44.0%</td><td>1.13</td><td>0.03</td></tr></table>