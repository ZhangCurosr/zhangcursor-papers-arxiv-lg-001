# Data-Driven Precision Control of Soft Helicoid Robots

(submitted to npj Robotics)

Roshan S. Kaundinya<sup>1</sup>, Mark Leone<sup>2</sup>, Roberto Sabella<sup>3</sup>, Hugo Buurmeijer<sup>2</sup>, Luis Pabon<sup>2</sup>, John Irvin Alora<sup>2</sup>, Marco Pavone<sup>2\*</sup>, George Haller<sup>1\*</sup>

<sup>1</sup>Institute for Mechanical Systems, ETH Z¨urich, Leonhardstrasse 21, 8092, Z¨urich, Switzerland.

<sup>2</sup>Department of Aeronautics and Astronautics, Stanford University, 496 Lomita Mall, Stanford, 94305, CA, USA.

<sup>3</sup>Department of Industrial Engineering, University of Naples Federico II, P.le V. Tecchio, 80, 80125, Naples, Italy.

\*Corresponding author(s). E-mail(s): pavone@stanford.edu; georgehaller@ethz.ch;

## Abstract

A purely data-driven method grounded in the theory of adiabatic spectral submanifolds (aSSMs) was recently used to model and control nonlinear soft robotic simulators. Here, we extend the methodology to learn aSSM-reduced models from data collected from hardware soft robotic platforms. Specifically, we learn a five dimensional (5D) aSSM-reduced nonlinear ordinary diferential equation (ODE) that accurately models the dynamics of a soft-helicoid trunk robot. We then use this ODE in a model predictive control scheme to demonstrate real-time precision control across various challenging closed-loop trajectorytracking tasks, ranging from 2D to 3D trajectories with static and dynamic constraints. We find that our methodology enhances accuracy and precision across all control tasks compared to available data-driven linear methods and modelfree reinforcement learning. While we focus here on a soft-helicoid trunk robot, our methodology relies solely on hardware data and is therefore applicable to a broader class of soft robotic systems.

Keywords: spectral submanifold, soft robotics, data-driven control, nonlinear dynamics

Modeling and control of soft robots has been a grand challenge in the recent decade (see Mengaldo et al. [31] and Yasa et al. [49]). As the design and fabrication of soft robotic systems have advanced, novel soft robotic architectures with complex geometries have become readily attainable. Once such soft robots are in motion, the dynamics that dictate their evolution are nonlinear. This nonlinear behavior of soft robot dynamics, while challenging to model and control, ofers operational advantages in safely navigating sensitive regions routinely encountered during surgical, undersea, and space missions.

On the hardware side, realizing these objectives requires compliant actuators that grant flexibility and dexterity, and robust controllers that ensure real-time tracking precision. Accurately modeling the nonlinear dynamics of a soft robot is a necessary step to inform both hardware fronts as they converge towards mission-ready soft robots. A classical modeling approach is to derive the governing equations from (typically idealized) physical principles. This leads to partial diferential equations, which, for numerical use, are further discretized using finite elements (see Faure et al. [12]), yielding a high-dimensional nonlinear mechanical system.

Classical control schemes exist for tracking tasks in nonlinear mechanical systems, either by linearizing the system along the target track (see Sastry [40]) or by deriving analytic closed-loop nonlinear control formulas for the forces (see Udwadia [46]). While the exactness of such formulas is appealing, the derived forces do not automatically correspond to admissible control inputs for the soft robotic system, which, by design, are underactuated. Also, in this classical approach, one has to make assumptions about the material parameters, the elasticity law, the finite-element geometry, and the geometric nonlinearities arising from boundary conditions.

In practice, one has to reduce the full dynamics of the soft robot to a lowdimensional model for enabling real-time control. Most such reduction strategies employ POD-Galerkin methods (see Goury and Duriez [16], Katzschmann et al. [26] and Tonkens et al. [45]), which are only fully justified if the system’s dynamics are linear. As an alternative, equation-based methods tend to approximate the shape profiles of soft robot geometries using either piecewise constant curvature (see Katzschmann et al. [25]) or strain (see Renda et al. [37]), or Cosserat-rod models (see Naughton et al. [34]). These have low fidelity compared to finite-element methods and may only apply to certain types of soft robotic geometries. As a result, most existing classical approaches are tedious to set up and, in some cases, even impossible to automate for general soft robotic hardware.

Dynamic data from motion capture systems or lasers is now routinely processed and used in simple feedback control. Hence, a data-driven route to modeling soft robot dynamics has become practically viable. The simplest and most popular data-driven method is to fit a linear, time-invariant system to dynamic data from soft robots. Proctor et al. [36] first implemented this method, called Dynamic Mode Decomposition with Control (DMDc), and tested it on simple problems. Recent work by Adibnazari et al. [1] performed a useful case study, comparing various linear data-driven methods, including DMDc, for shape control of a soft robotic fish.

A linear model yields fast controllers but will perform poorly for inherently nonlinear systems. To address this limitation, Rowley et al. [38] sought to fit linear dynamics to nonlinear functions of observables, which has lead to the idea of extended DMD with Controls (or EDMDc; see Bruder et al. [8]). Another fundamentally linear idea, Koopman analysis with static pregain, replaces the static input-output relation of a soft robot with a linear function (see Haggerty et al. [19]). Comparisons on simulations and hardware consistently show, however, that linear models perform poorly in soft robotics applications involving nontrivial trajectories (see Alora et al. [2, 3] and Kaundinya et al. [27], and upcoming experiments in this paper).

A natural next step beyond fitting linear models to data is to fit a neural network that is constrained by the soft robot’s physical laws. Gao et al. [15] learned the forcing due to actuation on the soft robot as a neural network, given the control inputs, assuming that the forcing is diferentiable. Bern et al. [4] pursue a similar approach but constrain it to follow soft-robotic kinematics. St¨olzle and Della Santina [42] learn the dynamics as a neural network by shaping potential energy functions they postulate for the soft robotics system, while Valadas et al. [47] assumes a form for the strain relations. For soft robotics systems with possible memory-dependent behavior, a recent method (see Catalasan and He [10]) uses the Koopman-Mori-Zwanzig framework (see Lin et al. [29] and Gupta et al. [18]) to account for memory dynamics via memory kernel terms, which are learned by fitting neural networks. Although these methods account for some of the underlying physics, they are black-box approximations and do not directly target the control task of interest. As a consequence, their accuracy is limited, as we will see in examples of the present paper.

Reinforcement learning (RL) directly targets the control task of interest by providing a black-box, model-free methodology that learns a control policy as a neural network optimized for the intended control objectives. RL approaches have found success in rigid robotics for tasks involving repetitive control actions that aid locomotion (see Hwangbo et al. [23]) and autonomous driving (see Levine et al. [28]). Even in the rigid robotic setting, however, an extremely large dataset spanning the robot’s workspace is needed to guarantee a convergent RL control policy. This is a physically demanding challenge for soft robotic systems: it prevents extended onrobot data collection due to excessive actuator wear, and also hysteretic damage to the soft robot geometry. For this reason, most RL methodologies for soft robotics rely on simulators (see Thuruthel et al. [44], Naughton et al. [34] and Jitosho et al. [24]) that mimic hardware to generate data for training a reliable RL policy or learn an efective model (see Zheng et al. [51]) that guarantees convergence to a policy by actively exploring the workspace. These model-based RL policies, however, sufer from the same issues as the classical and black-box approaches we have outlined.

A common feature of all soft continuum mechanical systems has emerged in recent years: their dominant dynamics converge to a low-dimensional, attracting invariant manifold in the phase space. Specifically, invariant manifolds known as spectral submanifolds (SSMs) have established themselves as powerful tools for mathematically justified model reduction (see Haller [20]). They have been successfully used in structural dynamics and fluid mechanics problems to predict forced response under previously unseen external forcing (see Cenedese et al. [11], Morsy et al. [32], and Bettini et al. [5]). For SSM construction in soft robotic systems, an additional timescale dichotomy is relevant: the internal vibration rates of the robot decay much faster than its operational speed along tracks in realistic applications. In that case, the invariant structure of interest that contains the global dominant dynamics is an adiabatic spectral submanifold (aSSM) (see Haller and Kaundinya [21] and Kaundinya et al. [27]). An aSSM is an attracting invariant manifold composed of trajectories that approach the set of possible operational trajectories of the robot at the slowest rate.

Reducing the full dynamics onto aSSMs ofers a rigorous approach to modeling the originally infinite-dimensional soft robot dynamics via a low-dimensional ODE. As a first step in this direction, Alora et al. [3] used a zeroth-order (i.e., near-equilibrium) approximation of aSSMs to model and control a hardware diamond robot on 2D target tasks. Kaundinya et al. [27] extended this by identifying low-dimensional aSSMs for several soft-trunk simulators and demonstrating accurate control across diverse 3D target tasks. They also performed a systematic comparison of various approximations of aSSMs. Bettini et al. [6] ofered a further reduction of the aSSM onto a slow manifold to enable modeling and control of one-dimensional artificial muscles. While these have been promising applications of the aSSM-based soft robot control, a hardware demonstration of the aSSM methodology in fully 3D soft robot control has not been available.

We focus here on addressing this hardware gap in the current aSSM control literature. We consider a soft robot made from the sustainable elastomeric resin EPU 46 (see Carbon, Inc. [9]), with a 3D geometry resembling an elephant trunk. Specifically, the soft trunk robot’s architecture is composed of trimmed helicoids, inspired by the recent work of Guan et al. [17]. These architected robots are designed to replicate the compliance of the biological morphologies that inspire them. Controlling soft-helicoid trunk robots has recently attracted significant interest. For instance Guan et al. [17] use inverse kinematics for closed-loop control under the assumption of piecewise-constant curvature. They propose a strategy to infer the correct tendon lengths that correspond to actuation forces. Related recent work by Zhang et al. [50] provides design guidelines for the 3D printing of a trimmed helicoid, with results from open-loop control. Also, Wong et al. [48] uses control barrier functions for whole-body avoidance; their strategy relies on a form of the equations of motion for their helicoidal trunk setup. Nan et al. [33] perform model-based RL policy learning for a soft-trunk helicoid using a neural network as a dynamics model. They then perform closed-loop tracking of 2D targets over short tracks compared to the length of the trunk robot.

Despite these advances, a data-driven closed-loop control method has not been put forward for precision control of soft helicoid trunk robots for dynamic target tasks spanning a global workspace that can also simultaneously perform obstacle avoidance.

Contribution: Our main contribution here is a fully data-driven, hardware implementation of aSSM-based model-predictive control (MPC). Specifically:

• [Systematic aSSM model learning from data] Section 3 presents a systematic methodology for learning accurate, low-dimensional aSSM-reduced models from limited observable data. We demonstrate this methodology with a 5D aSSM-reduced model of our soft helicoid trunk robot.

• [Practical use of aSSM in MPC] We integrate the aSSM-reduced model into an MPC scheme that exploits the full aSSM geometry. In contrast to prior work by Kaundinya et al. [27], the present aSSM-reduced model does not explicitly split the control input into slow and fast components and also does not rely on an inverse-kinematics map to generate guesses for solving the MPC. As a consequence, the model can be readily used in standard control optimization algorithms.

• [Closed-loop benchmarking] Sections 1.4–1.5 report closed-loop trajectory tracking results for aSSM-based MPC, benchmarked against state-of-theart linear baselines: DMDc, the Koopman-pregain method, a memory-based Koopman–Mori-Zwanzig model, and model-free reinforcement learning via proximal policy optimization (PPO).

• [Hardware validation] We provide hardware demonstrations, with video, for all closed-loop control tasks, including obstacle avoidance along 3D tracks.

Throughout this work, we assume no information about the equations of motion in any form. Accordingly, we only choose baselines in our comparison that are agnostic to the physics of the robot.

## 1 Results

In this section, we describe the closed-loop performance of our data-driven methodology based on the theory of aSSMs on a soft helicoid trunk robot. We first briefly discuss the design, digital fabrication, and morphology of the robot. We then present our data-driven modeling approach to capture the dynamics of the robot and evaluate the model’s performance on previously unseen hardware data. We next present real-time, closed-loop results for target tracking across three tracks, two of them 3D, with varying sizes, complexities, and speeds. We close this section by presenting real-time, constraint-aware performance of the soft helicoid robot for 3D static and dynamic obstacles in its workspace.

Table 1: Comparison of data-driven aSSM-based modeling and control against linear and learning-based baselines for closed-loop 3D target tracking. The maximum error represents the maximum relative pointwise error obtained across all closed-loop tasks presented in Section 1.4.
<table><tr><td>Method</td><td>Model dim. d</td><td>Data collection time</td><td>Training time</td><td>Maximum error [cm]</td><td>Constraints handling</td><td>Controller freq. [Hz]</td></tr><tr><td>aSSM (ours)</td><td>5</td><td>2 hours</td><td>5 minutes</td><td>5</td><td>Yes</td><td>50</td></tr><tr><td>Koopman-pregain [19]</td><td>6</td><td>2 hours</td><td>30 seconds</td><td>38</td><td>No</td><td>100</td></tr><tr><td>DMDc [36]</td><td>6</td><td>2 hours</td><td>30 seconds</td><td>18</td><td>Yes</td><td>100</td></tr><tr><td>MZ-Koopman [10]</td><td>33</td><td>2 hours</td><td>15 minutes</td><td>17</td><td>Yes</td><td>20</td></tr><tr><td>RL (PPO) [41]</td><td>NA</td><td>6 hours</td><td>6 hours</td><td>10</td><td>No</td><td>100</td></tr></table>

All of our closed-loop results for 3D target tracking have a video demonstration of the soft helicoid robot and also include comparisons with available linear datadriven modeling baselines: Koopman-pregain (Haggerty et al. [19]), and Dynamic Mode Decomposition with Control (DMDc) (Proctor et al. [36]), a memory-based baseline Mori-Zwanzig-Koopman (MZ-Koopman) (Catalasan and He [10]), and a model-free online reinforcement learning (RL) baseline trained using the proximal policy optimization (PPO) approach (Schulman et al. [41]). We summarize the model specifications, closed-loop tracking performance, and controller frequency in Table 1. We refer to the Supplementary Section 6 for details on learning the baselines from data.

## 1.1 Hardware: Design and setup

We consider a bio-inspired, tendon-driven, soft robotic platform, modeled after an elephant trunk, to validate model-based and model-free control methods. Our design targets two objectives: (i) maximizing the reachable workspace via large bending angles while minimizing parasitic axial compression and viscoelastic hysteresis, and (ii) preserving a wide, unobstructed central lumen to reduce material use and provide protected internal routing for power/signal lines, enabling future end-efector tooling (e.g., a camera or gripper).

## 1.1.1 Morphology and fabrication

Homogeneous hollow-cylinder morphologies are prone to radial expansion and buckling under tendon loads, which severs the monotonic tendon-displacement-to-curvature mapping and precludes reliable control (morphology comparison in Supplementary Section 1). We instead adopt the Trimmed Helicoid (TH) architecture [17], which decouples axial $( K _ { a x i a l } )$ and bending $( K _ { b e n d } )$ stifness, and can be tuned to obtain a structure that is compliant in bending yet stifer in the axial direction. This enables a larger workspace over the whole range of tendon displacement.

The platform is fabricated via Digital Light Synthesis (DLS™) and Continuous Liquid Interface Production $( \mathrm { C L I P ^ { \top \mathrm { M } } } )$ in EPU 46 (Elastomeric Polyurethane, 40% bio-based, see Carbon, Inc. [9]), yielding monolithic, isotropic parts with injectionmolding-like quality and negligible hysteresis (< 1.5%). This manufacturing process was chosen over more common ones (e.g. FDM), which introduce defects unsuitable for cyclic tendon loading (details in Supplementary Section 1).

## 1.1.2 Optimization and final platform

As no prior characterization exists for this TH–DLS–EPU 46 hardware combination, we used an agile prototyping approach, empirically iterating over sixteen morphological variants to converge on parameters P1–P4 which we highlight in the

<table><tr><td>ID</td><td>Parameter</td><td>Value</td></tr><tr><td></td><td>Outer Radius,  $R _ { e x t }$ </td><td>20.0 mm</td></tr><tr><td></td><td>Height, H</td><td>135.0 mm</td></tr><tr><td></td><td>Number of helices,  $N _ { h e l }$ </td><td>6</td></tr><tr><td>P1</td><td>Helix thickness, T</td><td>2.2 mm</td></tr><tr><td>P2</td><td>Width over radius,  $W / R _ { e x t }$ </td><td>0.35</td></tr><tr><td>P3</td><td>Number of revolutions,  $N _ { r e v }$ </td><td>1.5</td></tr><tr><td>P4</td><td>Microstructures, V</td><td>Absent</td></tr></table>

(a)

![](images/a74d582cf04952ba49a3a85545fc5bfae92984a09269ec4b9524b33fefe0d87a.jpg)  
(c)

![](images/d8fb8fbb32b641ce2e7b61288120725de4867e79d442bcaea07d3727936a0792.jpg)  
(b)  
Fig. 1: (a) Final optimized geometric parameters of the Trimmed Helicoid module. (b) The three-module soft helicoid trunk robot is vertically mounted. The end-efector and midpoint positions are motion-tracked using OptiTrack markers (gray). (c) Design space: Helix thickness $( P 1 )$ , Trimming ratio $( P 2 )$ , Revolutions (P3), Microstructures $( P 4 )$

Table. 1a (see Supplementary Section 1 for more detail). The optimized single-module geometry (Fig. 1b) achieves bending angles $> 9 0 ^ { \circ }$ with axial compression $< 1 5 \%$ . Our use of microstructure parameters $P 1 { - } P 4$ for optimization is informed by Sabella et al. [39], which reduces cost, mass, and material waste, thereby supporting environmental sustainability.

The final platform comprises three identical modules in series with a total length of 42 [cm], joined by rigid interface disks that also mount retro-reflective markers (see Fig. 1c). The actuation system comprises of 12 tendons driven by 6 Dynamixel XM540- W150-R motors (2 per module, each controlling an antagonistic tendon pair for singleplane bending), with 12 pretensioning mechanisms to remove slack and deadzones. Module tips are tracked in real-time using 4 OptiTrack Flex 3 cameras (see Fig. 1c).

## 1.2 Adiabatic spectral submanifolds for precision control

Motivated by general hardware setups for robotics, we formulate our control problem in the robot’s workspace, defined by $\mathbf { z } \in \mathbb { R } ^ { n _ { w } }$ , under the admissible control inputs $\mathbf { u } \in \mathbf { R } ^ { n _ { u } }$ . The goal is to find optimal control inputs $\mathbf { u } ^ { * } ( t )$ that force the robot to track a dynamic target track $\mathbf { { \cal T } } ( t )$ defined in its workspace. This objective can be formulated as the following finite-horizon optimal control problem

$$
\mathbf { u } ^ { * } ( t ) = \mathrm { m i n i m i z e } _ { \mathbf { u } ( \cdot ) } \int _ { 0 } ^ { t } \Big ( | | \mathbf { z } ( s ) - \mathbf { \Gamma } ( s ) | | _ { \mathbf { Q } _ { \mathbf { z } } } ^ { 2 } + | | \mathbf { u } ( s ) | | _ { \mathbf { R } _ { \mathbf { u } } } ^ { 2 } + | | \dot { \mathbf { u } } ( s ) | | _ { \mathbf { R } _ { \dot { \mathbf { u } } } } ^ { 2 } \Big ) d s ,\tag{1}
$$

subject to the dynamic evolution of the robot, state, and actuation constraints. Here, $\mathbf { Q _ { z } } , \mathbf { R _ { u } } ,$ , and $\mathbf { R } _ { \dot { \mathbf { u } } }$ are tunable positive semi-definite control cost matrices that penalize the target tracking, the controls, and control rates, ensuring that the optimal control problem is well-posed.

Our objective is to achieve precision control, which means solving eq. (1) accurately with a reliable dynamic model of the robot. For soft robotic hardware, the exact governing equations are generally unknown, and only limited observational data are available. We denote this observable vector as $\textbf { y } \in \mathbb { R } ^ { n _ { y } }$ , which in our soft helicoid system corresponds to its module ends tracked by the OptiTrack system (see Section 1.1).

To infer a reliable dynamic model, we observe that most soft robotic systems are heavily damped and hence exhibit an inherent dichotomy in time scales between the fast internal vibrations of the soft robot and its substantially slower trajectory speed during typical control tasks. This practical observation turns out to provide a representation of the soft robot’s dynamics on slow invariant manifolds in its phase space. The underlying dynamics, when reduced onto these low-dimensional slow invariant manifolds, ofer a rigorously justified modeling method to capture the core dynamics of the soft robot.

More generally, spectral submanifolds (SSMs) and adiabatic spectral submanifolds (aSSMs) are low-dimensional attracting invariant surfaces in the phase space of the robot. Trajectories synchronize exponentially fast with their internal dynamics which therefore act as reduced-order models for the full system. SSMs are constructed locally near fixed points of the uncontrolled robot, while aSSMs can be constructed globally in the presence of the time-scale separation we assume. Recent work by Kaundinya et al. [27] demonstrates clear performance gains when an aSSM-reduced model used in MPC instead of an SSM-reduced model.

The theory of aSSM reduction applies to general control systems of the form

$$
\begin{array} { r } { \dot { \mathbf { x } } = \mathbf { F } \left( \mathbf { x } , \mathbf { u } ( t ) \right) , } \end{array}\tag{2}
$$

where $\mathbf { x } \in \mathbb { R } ^ { n _ { x } }$ denotes points in the phase space of the robot, $\mathbf { u } \in \mathbb { R } ^ { n _ { u } }$ the control inputs and F is a smooth nonlinear function in its arguments. We assume that system eq. (2) satisfies the following:

$( A I )$ The control input ${ \bf \delta u } ( t )$ admits a splitting of the form

$$
\mathbf { u } ( t ) \equiv \mathbf { u } ^ { s } ( \epsilon t ) + \mathbf { u } ^ { d } ( t ) , \quad 0 < \epsilon \ll 1 , \quad \lVert \mathbf { u } ^ { d } ( t ) \rVert < \Delta \ll \lVert \mathbf { u } ^ { s } ( \epsilon t ) \rVert ,\tag{3}
$$

for all times and for some constant $\Delta > 0$

$( A \mathcal { Q } )$ System (2) admits a normally attracting critical manifold $( \mathrm { i . e . }$ , manifold of fixed points), $\mathcal { L } _ { 0 }$ , in the limit $\dot { { \mathbf { u } } } ( t ) \equiv 0$ of static control inputs:

$$
\mathcal { L } _ { 0 } = \Big \{ \mathbf { x } \in \mathbb { R } ^ { n _ { x } } : \mathbf { x } = \mathbf { x } _ { 0 } ( \mathbf { u } ) ; ~ \mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ) , \mathbf { u } ) \equiv 0 , ~ \mathbf { u } \in \mathcal { U } \Big \} ,\tag{4}
$$

for some open set $\mathcal { U } \subset \mathbb { R } ^ { n _ { u } }$

$( A \mathcal { 3 } )$ The d slowest eigenvalues of the spectrum of the linear operator family $\mathbf { D _ { x } F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ) , \mathbf { u } )$ , evaluated along the critical manifold $\mathcal { L } _ { 0 }$ , satisfy non-resonance conditions detailed in Haller and Kaundinya [21].

Then eq. (2) can rigorously be reduced to an $( n _ { u } + d )$ -dimensional aSSM-reduced model for ϵ and $\Delta$ small enough (see Haller and Kaundinya [21] and Kaundinya et al. [27]). The geometry of this $( n _ { u } + d )$ -dimensional aSSM is a fiber bundle whose base, in the frozen limit $( \dot { \mathbf { u } } ( t ) \equiv \mathbf { 0 } )$ , is the $n _ { u }$ -dimensional critical manifold, and whose fibers are the d-dimensional slow SSMs anchored at each point of this critical manifold. In practice, these results extend to larger values of ϵ and $\Delta$ as well; the smallness of these parameters is required only for the rigorous existence proofs of the aSSM.

The above reduction assumes access to the full state x. For soft robotic hardware, access to the full state space is absent $( n _ { y } ~ \ll ~ n _ { x } )$ . In fact, most motion capture systems record only positions and orientations from rigid-body markers attached to the soft robot. In such situations, we seek embeddings of the aSSM and its dynamics in the robot’s observable space, $\mathbf { y } \in \mathbb { R } ^ { n _ { 3 } }$ , from data. Since direct velocity observations are unavailable, we need to use delay embeddings of the position observables, and seek the aSSM in this delay-embedded observable space. Takens’ embedding theorem (see Takens [43]) guarantees that a d-dimensional fiber in the aSSM needs at least $\begin{array} { r } { p = \left\lfloor \frac { 2 d + 1 } { n _ { y } } \right\rfloor } \end{array}$ delays of a nondegenerate observable y to be generically embeddable in a $( p + \bar { 1 } ) n _ { y }$ delay-embedded observable space. We define this p delay embedding function with delay parameter τ as

$$
\begin{array} { r } { \pmb \mu ( \mathbf { y } ( t ) ) = \left( \begin{array} { c } { \mathbf { y } ( t ) } \\ { \mathbf { y } ( t - \tau ) } \\ { \vdots } \\ { \mathbf { y } ( t - p \tau ) } \end{array} \right) . } \end{array}\tag{5}
$$

As we will show in Section 3.1, the $( n _ { u } + d )$ -dimensional aSSM-reduced model in this delay embedded observable space is of the form

$$
\begin{array} { r l } & { \mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ) \pmb { \mu } ( \mathbf { y } - \mathbf { y } _ { 0 } ( \mathbf { u } ) ) , \qquad \mathbf { u } \in \mathcal { U } , } \\ & { \dot { \mathbf { r } } = \mathbf { R } ( \mathbf { r } , \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) ) + \mathbf { B } \left( \mathbf { u } ( t ) \right) \mathbf { D } _ { \mathbf { u } } \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } , } \\ & { \mathbf { y } ( t ) = \mathbf { W } ( \mathbf { r } ( t ) , \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) ) + \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) . } \end{array}\tag{6}
$$

Here $\textbf { r } \in \ \mathbb { R } ^ { d }$ are the reduced coordinates of the aSSM transverse to the critical manifold; $\mathbf { V } ( \cdot ) \in \mathbb { R } ^ { ( p + 1 ) n _ { y } \times d }$ is the chart map whose column vectors span the tangent space of the aSSM; $\mathbf { y } _ { 0 } ( \cdot )$ is the parametrization of the critical manifold in the observable space; $\mathbf { R } ( \mathbf { r } , \mathbf { y } _ { 0 } ( \cdot ) )$ is the leading-order reduced dynamics on the aSSM; B( ) is a calibration matrix for the higher-order terms of the aSSM-reduced dynamics; and ${ \bf W } ( { \bf r } , { \bf y } _ { 0 } ( \cdot ) )$ is the leading-order aSSM parametrization. In Section 3.1, we provide a self-contained derivation of the aSSM-reduced model for this setting, which difers in the choice of parametrization of the aSSM from Haller and Kaundinya [21] and Kaundinya et al. [27].

The functions V, y R, B, and W are all learnable from data. To this end, we need to pass control inputs to the robot that probe aSSM geometry. The frozen aSSM geometry $( \epsilon = 0$ and ${ \mathbf { u } } ^ { d } ( t ) \equiv 0$ in eq. (3)) comprises the critical manifold $\mathcal { L } _ { 0 }$ and the static slow SSM bundle, $\scriptstyle A _ { 0 } .$ attached to it (see Fig. 2a). This frozen aSSM geometry is directly learnable from data. Specifically, passing random step control inputs to the robot allows one to probe $\mathcal { L } _ { 0 }$ and $\mathcal { A } _ { \mathrm { 0 } }$ . The leading-order aSSM-reduced model is obtained by passing dynamic control inputs u(t) to the learned frozen aSSM model (see Fig. 2b).

In Section 3.2, we provide details on learning the leading-order aSSM-related functions from random step control input data, $\{ \mathbf { Y } _ { \mathrm { s t e p } } , \mathbf { U } _ { \mathrm { s t e p } } \}$ . We also provide details to learn the higher-order calibration function B from random control input data, $\{ \mathbf { Y } _ { \mathrm { c a l i b } } , \mathbf { U } _ { \mathrm { c a l i b } } \}$ , and the learned leading-order aSSM-related functions.

We use the aSSM-reduced model (3) learned from data to make predictions on previously unseen general control inputs ${ \bf \delta u } ( t )$ given initial conditions in the robot’s observable space obtained from motion capture. The aSSM-reduced model obtained in this way accurately captures the robot’s dynamics and hence can be used as a constraint in a model predictive control (MPC) scheme with an objective function defined in eq. (1). For MPC, we also define the state ( ) and control ( ) constraint sets, and a mapping from the observable space to the workspace,

![](images/51cfcb3d3db0bab5650c027efea3c052ce116ba4c659c21a9dfbd6476f602dbb.jpg)  
Fig. 2: Schematic for (a) the frozen limit of the aSSM geometry and (b) the dynamic aSSM geometry for time-varying control inputs.

$$
\mathbf { z } = \mathbf { C } ( \mathbf { y } ) , \quad \mathbf { y } \in \mathcal { Y } , \quad \mathbf { u } \in \mathcal { U } .\tag{7}
$$

Solving eq. (1) subject to constraints eq. (6)-(7) provides us with a closed-loop controller for target tracking tasks. In Section 3.3, we provide the techniques to solve this optimization problem in real-time.

## 1.3 Modeling the soft trunk dynamics via aSSM

The camera setup for our soft trunk platform can record the end efector and the mid-section of the trunk. We henceforth denote our observable space of the soft trunk as $\mathbf { y } = \{ x _ { e e } , y _ { e e } , z _ { e e } , x _ { }$ , y , z . Our workspace comprises the end-efector positions $\textbf { z } = \{ x _ { e e } , y _ { e e } , z _ { e e } \}$ , for which we will plan trajectory tracking tasks. These tracks are executed via a total of six motors that tension the tendons along the trunk’s arm. This amounts to $n _ { u } = 6$ control inputs.

The soft trunk decay profiles indicate a strongly damped system. The symmetry of the trunk with respect to the $x _ { e e }$ and $y _ { e e }$ axes yields a near 1 1 resonance between the slowest modes of vibration near the origin $\mathbf y = \mathbf 0$ . We note the resonance is allowed by SSM theory, as we recall in Haller and Ponsioen [22]. The soft trunk’s geometry also allows for at least one decaying mode that is present in the $z _ { e e }$ direction. Therefore, depending on the nature of the decays (overdamped or underdamped), we conclude that at least a $d = 5$ or $d = 6$ —dimensional fiber in the aSSM can capture the dynamics of the soft trunk robot. We note that these observations also applied to the aSSMreduced model derived for trunk simulators with similar geometries by Kaundinya et al. [27]. From our analysis in Section 1.2, we expect these d = 5 or d = 6 dimensional fibers of the aSSM to be embedded in a $6 ( p + 1 )$ ) dimensional delay-embedded observed space with $p \geq 2$

As a first step in building the aSSM from data, we pass random step control inputs to the soft trunk robot for 30 minutes. These step inputs are sampled to ensure that the observable space of the trunk is uniformly probed (see Section 3.2). Next, we collect 1 hour 30 minutes of slow, smooth, random control inputs, with the sampling done via a Perlin noise scheme (see Perlin [35]). These experimental runs provide us with the datasets $\{ \mathbf { Y } _ { \mathrm { s t e p } } , \mathbf { U } _ { \mathrm { s t e p } } \}$ and $\{ \mathbf { Y } _ { \mathrm { c a l i b } } , \mathbf { U } _ { \mathrm { c a l i b } } \}$

In Fig. 3a, we plot the collected trajectories in the observable space. For learning aSSM-reduced models, we also need to post-process the decaying dataset $\{ \mathbf { Y } _ { \mathrm { s t e p } } , \mathbf { U } _ { \mathrm { s t e p } } \}$ dataset as outlined in Section 3.2. The post-processed dataset isolates decays and static states. For this dataset, we isolated $N = 9 9 2$ static states, each with 10 clean decays (see Fig. 3b). We further split these datasets randomly into 90% for training and 10% for testing. We use the training dataset and follow the steps outlined in Section 3.2.1 for learning the critical manifold ${ \bf y } _ { 0 } ( { \bf u } )$ and in Section 3.2.2 for learning the aSSM-reduced model.

We learn a $( n _ { u } + d ) = ( 6 + 5 ) -$ dimensional aSSM-reduced model (see eq. (30)-(35)) with parameters $k _ { r } ^ { W } = 2$ and $k _ { y } ^ { W } = 2$ for the aSSM parametrization, $k _ { r } ^ { R } = 2$ and ${ \dot { k } } _ { y } ^ { R } =$ 2 for the reduced dynamics, and $k _ { u } ^ { B } = 2$ for the calibration in a $6 ( p + 1 )$ - dimensional delay-embedded observer space with $p = 6$ . The normalized mean trajectory error on unseen test decaying data for the leading-order aSSM-reduced model is 9.6% (see Fig. 3c for centered predictions). We also test the aSSM-reduced model with various approximations of the critical manifold $\mathcal { L } _ { 0 }$ . We find that a linear radial basis function with $n _ { c } = 6 , 0 0 0$ centers yields the lowest error compared to linear, polynomial, or neural network models (see Fig. 3d) on unseen control input data. Hence, we identify a $( n _ { u } + d ) \ = \ ( 6 + 5 )$ dimensional aSSM-reduced model with a critical manifold approximated by linear radial basis functions to model the soft trunk’s dynamics accurately. Since $n _ { u }$ is the dimension of the control inputs that parametrize the model, and the reduced ODE is d-dimensional, we hereafter denote the aSSM-reduced model as aSSM (dD).

## 1.4 Closed-loop results for target tracking

We use our data-driven aSSM-reduced model for the closed-loop control of target tracking in the workspace of the robot. The model is justified with full mathematical rigor when the target speed is slow compared to the robot’s internal dynamics. To quantify such regimes of applicability, we recall from Kaundinya et al. [27] the nondimensional relative slowness measure $r _ { s }$ defined for a target track Γ(t) over a finite time interval [0, T] as

$$
r _ { s } = \frac { \frac { 1 } { T } \int _ { 0 } ^ { T } \| \dot { \bf { \cal T } } ( s ) \| d s } { \frac { 1 } { t _ { d } } \int _ { 0 } ^ { t _ { d } } \| \dot { \bf z } _ { \mathrm { f r e e } } \operatorname * { d e c a y s } ( s ) \| d s } ,\tag{8}
$$

Here $\mathbf { z } _ { \mathrm { f r e e ~ d e c a y s } } ( t )$ are the intrinsic vibrations of the soft robot near its static states; the overbar denotes averaging over various initial conditions in the workspace of the robot; and $t _ { d }$ is the total duration of the non-zero part of the decaying signals. We expect accurate target tracking when $r _ { s } < 1$ , as this implies strict time-scale separation, a key assumption for the theory of aSSMs to hold. Note that low values of $r _ { s }$ correspond to slowly varying tracks relative to the robot’s intrinsic dynamics, such tracks do not necessarily indicate targets that are perceived as slow by a human observer.

In Section 3.3, we provide details for the closed-loop controller for target tracking with an aSSM-reduced model. We found that choices $\mathbf { R _ { u } } = 1 0 ^ { - 4 } { \cdot } \mathbb { I } _ { 6 }$ and $\mathbf { R _ { \dot { u } } } = 1 0 ^ { - 3 } { \cdot } \mathbb { I } _ { 6 }$ for control and control rate costs in eq. (37) led to physically relevant control input magnitudes for the aSSM-reduced model. For 2D target tracking tasks, we set $\mathbf { Q _ { z } } =$ diag (1000, 1000, 0) and for 3D target tracking tasks, we set $\mathbf { Q _ { z } } = \mathrm { d i a g } \left( 1 0 0 , 1 0 0 , 1 0 0 \right)$ For MPC horizon lengths up to $M = 2 0$ , we can operate the aSSM-based controller at 50 Hz. We used $M = 1 0$ for the target tracking task and $M = 2 0$ for the obstacle avoidance tasks.

We also present here results for four baselines: a 6D DMDc model (see Proctor et al. [36]), a 6D Koopman-pregain model (see Haggerty et al. [19]), a 33D MZ-Koopman model (see Catalasan and He [10]), and an RL (PPO) control policy (see

![](images/88826608b1c9e1f660e99ce2e8c0852770839f80d7f3e3f6cc15cea6290a4778.jpg)  
(a)

![](images/a97daa9482abbb22a3f6e0dd089aa64795d5140cb8c5ee89b95e41f5195be364.jpg)  
(b)

aSSM model testing on previously unseen decays  
![](images/6b1cd335f35364ab3e19f2d35201fc068aafd6b78390e8ff743ad519520a12e7.jpg)  
(c)

aSSM (5D) model with various approximations of y<sub>0</sub>(u)  
![](images/8c02e40dea58dd7d1a1a5554da9baacec16d8e27a3402c105a5eafafc318618a.jpg)  
(d)  
Fig. 3: (a) Step and random input sequence responses in the soft trunk’s observable space. (b) Filtered decay plots after pre-processing the step input dataset $\mathbf { Y } _ { \mathrm { s t e p } }$ . (c) Predictions on previously unseen decay data of the leading-order aSSM-reduced model for static inputs. (d) Prediction of the aSSM-reduced model for unseen random control inputs using various approximations of the critical manifold $\mathcal { L } _ { 0 }$

Schulman et al. [41]). We report the optimal controller settings for these models in the Supplementary Section 6.

Our interest is to evaluate the dynamic precision of the various methods for several closed-loop tasks. During closed-loop hardware operations, various delays can occur for all models at the software and hardware levels. These delays result in errors between the target and the closed-loop prediction $\mathbf { z } _ { \mathrm { p r e d } }$ at the time instant t. To this end, to measure dynamic accuracy as well as spatial accuracy, we adopt the sliding-windowtype relative pointwise error metric

$$
\mathrm { P o i n t w i s e ~ r e l a t i v e ~ e r r o r } ( t ) = \operatorname* { m i n } _ { \tau \in [ t - \delta t , t + \delta t ] } \| \mathbf { z } _ { \mathrm { p r e d } } ( t ) - \mathbf { \Gamma } ( \tau ) \| ,\tag{9}
$$

with the half-window spacing set to $\delta t = 1 \ [ \mathrm { s } ]$

We also note that our current motor actuation mechanisms allow for reproducible dynamic responses with a mean maximum error of 3 [cm] observed for the end-efector. Hence, we expect our model predictions to perform well if they are within an uncertain tube of this radius around the target track. We will plot such tubes around our target tracks in the workspace and also show these as lines in the pointwise relative error plots.

## 1.4.1 Figure 8 track

We consider here a large Figure-8 track defined in the $x _ { e e } - y _ { e e }$ plane. The maximum width and length span of the Figure-8 is 0.2 [m]. This is six times larger than the Figure-8 track considered by Alora et al. [3] for the hardware diamond robot, and it is nearly half the length of the soft trunk robot (0.42 [m]). We run closed-loop experiments for a total duration of 30 [s] with two diferent target speeds: (i) a Figure-8 track with the mean instantaneous speed $v _ { \Gamma } = 8 ~ [ \mathrm { c m / s } ]$ and slowness measure $r _ { s } =$ 0.17 (ii) a 1.5 faster varying track with the mean instantaneous speed $v _ { \Gamma } = 1 3 ~ [ \mathrm { c m / s } ]$ and slowness measure $r _ { s } = 0 . 2 5$

The closed-loop results for the target speed case (i) for all models are presented in the left panel in Fig. 4 and for the faster speed case (ii) in the right panel of the same figure. Qualitatively, we observe across both target speeds, only the aSSMreduced MPC scheme tracks the Figure-8 within the gray uncertainty region. The Koopman-pregain comes second, with a visible Figure-8, but the plots are misaligned in orientation and hence exit the uncertainty region. The RL (PPO) policy’s performance resembles a Figure-8, but it jitters quite a bit around the traced track. Compared with the Koopman-pregain method, it shows a larger misalignment from the original track. The MZ-Koopman controller also predicts a Figure-8 track, but it is of-center, exhibits jitter, and is significantly smaller than the original target track. The DMDc model-based controller performs poorly, and the resulting track hardly resembles the Figure-8.

We also provide quantitative results with pointwise error plots for the speed case (i) in Fig. 5 and the speed case (ii) in Fig. 6. We find that the aSSM-reduced MPC scheme ofers consistent performance across the target track and track speeds, with pointwise relative error remaining mostly below the tolerance line at 3 [cm]. Unlike the gray tube in Fig. 4, which only checks whether the trajectory stays close to the target track, the pointwise error is evaluated at each time instance and therefore also penalizes delays and deviations from the reference. This explains the three instances where the pointwise error of the aSSM exceeds the tolerance, even though the trajectory remains within the tube. All other models consistently cross the tolerance line. We also see that overall errors increase as the target speed is increased. We also note that on the faster track, there are regions where the aSSM-reduced model has consistently low errors that are comparable to its performance on the regular speed track.

![](images/a6faa40fd4edbcfee705a07186d3e0b14665468707eb0f21a9514e43da35fa0e.jpg)  
Fig. 4: Closed-loop tracking results for a Figure-8 target (black) with the uncertainty band of width 3 [cm] plotted in gray.

![](images/cf436ecc3f4230d483ffc3f0c488b6c7013ee6612316b492fb9c90ef0f0fb99d.jpg)  
Fig. 5: Pointwise relative error for closed-loop tracking of a Figure-8 track with speed $9 \ [ \mathrm { c m / s } ]$

![](images/0c6497d755bc32feba86aa7730b05b6a735faa6b01f8864d3225fdec71dafd80.jpg)  
Fig. 6: Pointwise relative error for closed-loop tracking of a faster Figure-8 track.

## 1.4.2 3D Random track

Our second target track is a challenging 30-second-long 3D Random target track in the trunk’s workspace, generated by passing a smooth random control input sequence. This is not seen in the training control data used to learn the various models and the RL policy. The mean instantaneous speed of the target is $v _ { \Gamma } = 5 ~ [ \mathrm { c m / s } ]$ with a slowness measure $r _ { s } = 0 . 1$

![](images/a61a3ac17a9662b35704a1c708f3b993022b79eb927be072b624e72913bbcfc7.jpg)

![](images/ccbdcfebf3ce70830935b9fba08ad5a14f1f86267a91376ed9b7e09d63e66146.jpg)

Koopman-pregain (6D)  
![](images/ebbfceb8b0a53749cf895150ba9d1ef3ff4fe5561fb896c47ff45f9073dea10e.jpg)

![](images/20a5711879d9d31dc76547ec0ea1ef556cd2d142374edf2b068ce8c255c90b5d.jpg)

![](images/7fb214d174ffa1f66be11eb8e77f8718538f9ce76c5deacc60b5548c1b70046b.jpg)  
Fig. 7: Closed-loop tracking results for a 3D Random track (black) with the uncertainty tube of radius 3 [cm] plotted in gray.

Along the 3D Random track, we reach a maximum height of 0.1 [m], and along the $x _ { e e } - y _ { e e }$ plane, we have a maximum width of 0.4 [m]. This track explores the extremal boundaries of the trunk robot’s workspace.

![](images/3d4e8811484c4e7e6fe38b798b381813a67b1724da5ec09128a88fc163da6568.jpg)  
Fig. 8: Hardware snapshots during closed-loop tracking of the 3D Random track. See Supplementary Movie for all closed-loop hardware demonstrations.

The closed-loop experimental results are plotted in Fig. 7 for all the model-based controllers and the RL policy. The 3D tracks require the use of complex nonlinear bending shapes for the trunk robot to achieve low tracking errors. Our aSSM-reduced controller achieves precisely this objective, as shown in the closed-loop prediction plots, and the snapshot of the trunk in Fig. 8 validates our expectation.

The linear methods (Koopman-pregain and DMDc) fail to achieve this objective. The workspace plots indicate sharp deviations from the target track, and the hardware snapshot plots (see Fig. 8) show that both models cause the trunk to bend linearly, sending the trunk’s end efector of the intended track and outside the tube of uncertainty.

![](images/7fd7c7ae951732d3ec86a320d64b9fda65a03f770af454d69bcf18b2d0e47fb4.jpg)  
Fig. 9: Pointwise relative error for closed-loop tracking of a 3D Random track.

The memory-based MZ-Koopman controller sufers from an of-centering issue, as observed in the Figure-8 track. It is a sizeable model, and hence real-time MPC struggles to correct for the errors as we traverse the target track. We attribute the model’s of-centered performance to the ad hoc manner in which the observables are chosen. The burden of overfitting of the model’s many parameters leads to an incorrect prediction.

The RL policy has even more parameters than the MZ-Koopman model and hence is susceptible to the same problems. Indeed, the RL policy performs very poorly. Despite longer training on similar random target tracks, it fails to generalize to an unseen random target track. We support all these observations quantitatively in Fig. 9 with pointwise relative error plots.

## 1.4.3 3D Pacman track

We conclude this section with closed-loop tracking for a 3D Pacman track. This is a significantly harder task than the 3D Random track, as it is a apriori unclear if the track is physically feasible for the soft trunk to traverse, and it also contains sharp, rapid jumps. The 3D Pacman track rises to a height of 0.05 [m] and has a radius of 0.1 [m]. For a total duration of 30 [s], we perform closed-loop experiments for three diferent target speeds: (i) mean instantaneous speed $v _ { \Gamma } = 3 ~ [ \mathrm { c m / s } ]$ with slowness measure $r _ { s } = 0 . 0 5 ~ \mathrm { ( i i ) }$ twice as fast with slowness measure $r _ { s } = 0 . 1 2 ~ \mathrm { ( i ) } ~ \mathrm { ( i i i ) }$ four times faster than (i) with slowness measure $r _ { s } = 0 . 2$

v<sub>Γ</sub> = 3 [cm/s] aSSM (5D)  
![](images/83a70302e1df681e058d5d1d5ba64eb782f7732b79f9ea9d6d6f4ef9bffc66d2.jpg)

![](images/10d03ff133d1c297493f867ee35914c405e17e1ebff7de2efad6b863863eb941.jpg)  
DMDc (6D)

DMDc (6D)  
![](images/caf13896e84d27eefda88d110d538440e0ad7b1d5970d2d1fdb0cdb691ba0624.jpg)

![](images/a1f07b05db3a7b98f8607693f4f6ed41ce468d043c15e0fc52463ede28118392.jpg)

Koopman-pregain (6D)  
![](images/d2d7f69dca8286685f500860d40ffcafa8f21fa4f9d67adc4e151691adef5204.jpg)

Koopman-pregain (6D)  
![](images/0cc9eec4ec06b5af9779956d2f39402f420ae63239edd712fe4c831332f42419.jpg)  
MZ-Koopman (33D)

MZ-Koopman (33D)  
![](images/01b4d22a9d3693258b4ba60989346275bbf3c5702f12099e17f2a641776c9b97.jpg)

![](images/4775200426e9e10300e2507be71b8f98f56ad24ecc486532c08a00b111435cf5.jpg)

RL (PPO)  
![](images/5a0d569ef6faa04dc552ca99de63add4d6ba90617593e903257d3c8012ffeb24.jpg)  
yee [m]

RL (PPO)  
![](images/d1a932cdf8c8be989f73de26067bdbe0855d66df0c3217a3897d4485e1d1ca59.jpg)  
yee [m]  
Fig. 10: Closed-loop tracking results for a 3D Pacman track (black) with the uncertainty tube of radius 3 [cm] plotted in gray. The left panel shows results for the average speed along the target track of 3 [cm/s], and the right panel shows results for a speed that is 2 times faster.

We plot the closed-loop predictions for all the models in the workspace in Fig. 10, with the left panel for target speed case (i) and the right panel for the speed case (ii). Only the aSSM-reduced model achieves the maximal target height and tracks the 3D Pacman across both speeds. The linear methods fail to lift of from the origin state. Either they are completely of from the target or they underestimate the height as the controllers based on them have no nonlinear information to access the large heights in the trunk’s workspace. They also linearly extrapolate, which causes them to stray outside the tolerance tube.

The MZ-Koopman model provides similar closed-loop predictions as before, but in this problem, it is unable to reproduce even the qualitative shape of the 3D Pacman trajectory. The RL policy at best reaches the necessary height, but it is unable to trace out the track and starts oscillating rapidly as it tries to align with the track. These qualitative visual assessments of the closed-loop result are corroborated with pointwise relative error plots in Fig. 11 for case (i) and in Fig. 12 for case (ii).

![](images/899a02d08609df8651a488d0984b3ef6cd325e7a02d41f3399f098e2bf4257d7.jpg)  
Fig. 11: Pointwise relative error for closed-loop tracking of a 3D Pacman track with speed 3 [cm/s].

![](images/29ab2c025cf39e459a13fb35e97273b06e5d358c57325bf3941eeeb4fb04a3a3.jpg)  
Fig. 12: Pointwise relative error for closed-loop tracking of a faster 3D Pacman track.

In the Supplementary Section 2, we present the results for case (iii) with error plots. Despite the track being extremely fast, the aSSM-reduced model still provides consistent performance with lower errors across the target track than the other baselines.

## 1.5 Closed-loop results for obstacle avoidance

We conclude the results section 1 with closed-loop experiments conducted in the presence of static and dynamic obstacles within the trunk’s observable space. Here, we will set additional constraints for the MPC problem defined in eq. (37). For the static obstacle task, we fix the target Γ(t) to be the 3D Random target track used in Section 1.4.2. For the dynamic obstacle tasks, we focus only on avoiding a moving obstacle; hence, we set the target-tracking costs to zero, $\mathbf { Q _ { z } } = \mathrm { d i a g } ( 0 , 0 , 0 )$ . In all obstacle-avoidance tracking problems, we only record the center points of the trunk’s end-efector and mid-section positions. To safely account for the trunk’s radial width when avoiding three-dimensional objects, we therefore add an efective bufer length of $r _ { \mathrm { e f f } } = 0 . 2 5$ cm.

We also note that none of our obstacles are real physical objects. They are virtual spatial constraints designed by the user. However, we still expect our system to work seamlessly when the constraints are physical, provided we have enough motion capture cameras to accurately observe the trunk’s end efector and the obstacle’s position. At present, we do not have the camera system to carry out this task. Hence, we resort to using virtual obstacles for a proof of concept.

## 1.5.1 Avoiding static spherical obstacles

We add static spherical constraints to the MPC by enforcing the condition,

$$
\| \mathbf { z } - \mathbf { c } \| \geq r + r _ { \mathrm { e f f } }\tag{10}
$$

for the trunk’s end efector under closed-loop tracking. We pick centers c for the sphere locations as random nearby points along the target track. This will ensure that the track is non-trivially obstructed our goal is to track the target as precisely as possible while strictly avoiding the spherical obstacles. We denote the radius of the spherical obstacle as r and always plot it in dark red. We also plot a red translucent spherical shell of width $r _ { \mathrm { e f f } }$ centered at c.

We demonstrated closed-loop experiments for three spherical obstacles of diferent sizes: (i) a small obstacle of radius $r = 0 . 0 5$ [cm], (ii) a medium obstacle of radius $r = 0 . 1$ [cm], and (iii) a large obstacle of radius $r = 0 . 2$ . All these spherical obstacles are placed at random locations close to the 3D Random track. We plot temporal snapshots in which the trunk must avoid the spherical obstacle. See Fig. 13a for case (i), see Fig. 13b for case (ii), and Fig. 14a for case (iii).

We compare our aSSM-reduced model with the constraint condition switched of, to ensure that the avoidance is a non-trivial occurrence. Across all figures, the aSSMreduced controller without constraints passes through the obstacle, whereas with constraints active, the model’s closed-loop predictions steer the trunk’s end efector away from it. For the small- and medium-sized spherical obstacles, the aSSM-reduced controller avoids the entire spherical constraint, including the bufer zone. The large spherical constraint is more challenging. In that task, the aSSM-reduced controller steers the end-efector away from the dark red spherical obstacle but is still trapped in the bufer region.

We also compare these results with those from the DMDc and MZ-Koopman models, the two benchmarks that theoretically allow for constraint handling. The DMDc model operates at a diferent height, avoiding the spherical obstacles, but ultimately misses the equally important task of tracking the target. The MZ-Koopman model’s predictions across all tasks consistently blow $\mathrm { u p } .$ , leading to the deployment of the controller’s safety codes to preserve the trunk hardware. The reason for this poor performance is that a 33D model with constraints is not fast enough to run in real time and is also not accurate, leading to poor guesses by the optimization solver and, consequently, longer solve times.

![](images/94b9ae1aa437d0b275c556ea2240b23e15a096de1a0fa70731b267e37f39fb5b.jpg)  
(a)

t = 12 [s]  
![](images/b2dfd4e1f5c1d7a988ee3ccf435bb812ca99b898a499dd276c3ecd4401d34822.jpg)  
(b)  
Fig. 13: Snapshots for closed-loop avoidance of static spherical obstacles while simultaneously tracking a 3D Random track for (a) a small-sized spherical obstacle, and (b) a medium-sized spherical obstacle. Each plot contains closed-loop results for the 5D aSSM-reduced (green solid line), the 6D DMDc model (purple), the 33D MZ-Koopman model (yellow), and the 5D aSSM-reduced with deactivated constraints (green dotted line). The uncertainty tube of radius 3 [cm] is plotted in gray around the target track. The dark red sphere is the obstacle, and the translucent sphere shell surrounding it is a safety bufer zone with a width of 2.5 [cm].

![](images/173800b18dc3d0eeb5cf20c8dd6ade62492235ae87a1452d15b2826b9f3e2c66.jpg)  
(a)

Relative error for a large-sized spherical obstacle  
![](images/f6b42dba40ab9a701d115f19eaf4782db3380e591e4c87e0544bb96198ae495d.jpg)  
(b)  
Fig. 14: (a) Snapshot for avoiding a large spherical obstacle while tracking the 3D Random target. (b) Pointwise relative error plots for the large-sized spherical obstacle avoidance task. The tolerance line accounts for the sphere’s location in real time as the target track is traversed.

We collected episodes to train RL policies for almost 8 hours, but the method did not converge and caused the trunk’s tendons to snap, resulting in extensive downtime for maintenance. For these reasons, we have not been able to construct an efective RL control policy to handle constraints. In Fig. 14b, we provide error plots that quantitatively assess whether we avoid the large spherical constraints for all models. The error plots for the small- and large- sized obstacles can be found in Supplementary Section 3.

## 1.5.2 Avoiding dynamic cylindrical obstacles

Our final closed-loop experiment involves the trunk avoiding a moving cylindrical obstacle. We present results only for our aSSM-reduced model. The MZ-Koopman model has longer solve times and cannot handle a complex moving obstacle, causing the controller to stop. In contrast, the linear DMDc closed-loop controller produces erratic initial predictions that violate the moving constraint, leading the controller to stop abruptly, as shown in the previous section.

![](images/5593d1f450800b25adfc95d48e03c7c08f956b1102f75a47ad580ed6eacbe53d.jpg)

(a)  
![](images/4c4eb5e9e6542f02044f1eb947f2847e0a52530805887c6804e9c1f4ab1541b0.jpg)  
(b)  
Fig. 15: Snapshots of the trunk main body avoiding a moving cylindrical obstacle at $\mathrm { ( a ) } ~ t = 5 ~ [ \mathrm { s } ]$ as it approaches the rest state and (b) t = 15 [s] as the obstacle rotates near the rest state via aSSM-based closed-loop control. The trunk body is green, and the cylindrical obstacle is yellow.

Here, we will need to use the full trunk geometry to avoid the obstacle. The trunk’s observable space allows us to approximate its geometry between the mid-position and the end-efector as a cylinder with radius $r _ { \mathrm { e f f } } = 0 . 2 5 ~ [ \mathrm { c m } ]$ . We use this cylinder approximation of the trunk in the observable space to set up a constraint that avoids

a moving cylindrical obstacle, given by,

$$
\operatorname* { m i n } _ { s , u \in [ 0 , 1 ] } \Big \| ( ( 1 - s ) \mathbf { z } _ { \operatorname* { m i d } } ( t ) + s \mathbf { z } ( t ) ) - \Big ( \mathbf { c } ( t ) + \big ( u - \frac { 1 } { 2 } \big ) l \mathbf { d } ( t ) \Big ) \Big \| \geq r _ { \mathrm { e f f } } + r .\tag{11}
$$

Here the observable space is split into mid and end-efector positions $\mathbf { y } = \{ \mathbf { z } , \mathbf { z } _ { \mathrm { m i d } } \} ; l$ and r, denote the length and the radius of the cylindrical obstacle, c(t) is the moving centerline of the obstacle, and d(t) is the orientation of the cylinder axis.

![](images/a3c1cfa810cfb61fd8cfd65f2c59f6a17befb4455f03b3d88940a269cd689137.jpg)

![](images/fa37b3444e1adb55de78dd823cd55ca734c8c30764a169521a3f3a6cde91b376.jpg)  
(b)  
Fig. 16: Snapshots at (a) t = 1 [s] and (b) $t = 3 ~ [ \mathrm { s } ]$ for the same closed-loop avoidance task same as Fig. (15), with a faster moving cylindrical obstacle.

In Fig. 15, we plot snapshots of the trunk avoiding the moving cylindrical obstacle under aSSM-based closed-loop control. We set the length of the cylindrical obstacle to $l = 0 . 2 5$ [m] and its radius to $r = 0 . 0 3 1$ [m]. The centerline path c(t) moves the cylindrical obstacle inwards towards the trunk’s rest position from a height of 0.2 [m]. The cylindrical obstacle reaches the trunk’s rest position in 8 [s], after which it rotates its orientation axes d(t) clockwise in the $x _ { e e } - y _ { e e }$ plane for the next 10 [s]. We show that the trunk body (plotted in green) avoids the cylindrical obstacle during dynamic variation along the centerline path (see Fig. 15a) and dynamic changes in cylinder orientation (see Fig. 15b).

Finally, in Fig. 16, we plot the trunk body avoiding the cylindrical obstacle for a dramatically faster case of the task described before. We see that in a span of 3 [s], the trunk quickly moves away from the fast-approaching cylindrical object. We also describe in Supplementary Section 4 a closed-loop result for a more complex centerline path.

## 2 Discussion

We have developed a hardware-compatible, data-driven control method for soft robotic systems based on the theory of adiabatic spectral submanifolds (aSSMs) presented in Section 1.2. We first discussed and verified the existence of an inherent timescale separation in typical operational use of soft robots. This separation allows for reducing the soft robot dynamics onto low-dimensional attracting aSSMs. Specifically, in Section 3.2, we presented a methodology to extract aSSMs from limited observational data. We validated this data-driven method on a hardware soft helicoid trunk robot by learning a 5D aSSM-reduced model only from camera observation data of the trunk’s end-efector and mid positions (see Section 1.3). We then constructed a general-purpose controller using the learned aSSM-reduced model to perform dynamic target tracking and obstacle avoidance (see Section 3.3). Our modeling and control method identifies dominant dynamics of the robot without making assumptions about its underlying physics. As a consequence, our methodology transfers directly across morphologies and actuation systems.

Our aSSM-based controller was then stress-tested on a variety of trajectory tracking control tasks ranging in spatial complexity and speed. We also included systematic comparisons with other available linear baselines (DMDc and Koopmanpregain), a recent memory-based baseline (MZ-Koopman), and a routinely used reinforcement learning (RL) based control policy. In this case study, we documented that the aSSM-based controller provided consistent precision performance compared to the other baselines that failed to generalize (see Section 1.4). We further tested the generalizability of our aSSM-based controller for obstacle-avoidance tasks (see Section 1.5). Across both static and dynamic obstacles, our aSSM-based controller successfully avoided the obstacles while simultaneously tracking other user-defined objectives.

The performance quality of the aSSM-based controller is achieved with only a modest training data exploration of the trunk’s workspace to learn the aSSM-reduced model. These requirements are practically realizable compared to online training of RL control policies directly on hardware. While the linear methods tested here are data-eficient, they end up extrapolating into unsafe regions, ultimately degrading precision. Thus, our aSSM-based controller meets practical trade-of requirements for hardware operation (see Table 1).

On the practical side, future work will include testing the aSSM-based controller with real physical obstacles and performing interactive demos that demonstrate the accuracy of shape control for the soft robot. We also expect to demonstrate our method on diferent soft robotic actuation systems (beyond tendon-driven), such as embodied actuation systems with tactile sensing or pressure-driven systems. On the theory side, developing an extension that systematically learns higher-order corrections to the aSSM-reduced model from data and explores their use in control is highly desirable. Another remaining objective is to extend aSSM-reduced modeling to account for external forcing that commonly occurs during environmental interactions. Such interactions include soft robotic hands grasping a ball or a trunk end-efector carrying payloads.

## 3 Methods

## 3.1 Adiabatic spectral submanifolds

Soft robots are infinite-dimensional dynamical systems satisfying partial diferential equations (PDEs). Available equation-based reduced-order modeling methods approximate the solutions of these PDEs using finite elements or Cosserat rods. We assume here that the controlled soft robot dynamics can be represented by a high-dimensional, first-order dynamical system,

$$
\begin{array} { r } { \dot { \mathbf { x } } = \mathbf { F } ( \mathbf { x } , \mathbf { u } ( t ) ) , } \end{array}\tag{12}
$$

where $\mathbf { x } \in \mathbb { R } ^ { n } , \mathbf { u } \in \mathbb { R } ^ { n _ { u } }$ are the phase- and actuation- space of the soft robot, and F is a smooth nonlinear function that governs the geometric and material deformation rate of the soft robot.

For static control inputs $\mathbf { u } ( t ) \equiv \mathbf { u } _ { 0 }$ , the soft robot will deform into static configurations. We denote this family of static states by ${ \bf x } _ { 0 } ( { \bf u } _ { 0 } )$ . These states are fixed points of the system 12 and hence satisfy

$$
\mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } _ { 0 } ) , \mathbf { u } _ { 0 } ) = \mathbf { 0 } .\tag{13}
$$

We further assume that for all static control inputs $\mathbf { u } _ { 0 }$ in a compact neighborhood $\mathcal { U } \subset \mathbb { R } ^ { n _ { u } }$ , the corresponding static configurations are stable hyperbolic fixed points of the dynamical system (12). This implies in that in the autonomous limit u $\mathbf { \rho } _ { ( t ) } \equiv \mathrm { c o n s t } .$ system (12) has a normally attracting critical manifold $\mathcal { L } _ { 0 }$ (see Fenichel [13]),

$$
\mathcal { L } _ { 0 } = \{ ( \mathbf { x } , \mathbf { u } _ { 0 } ) \in \mathbb { R } ^ { n } \times \mathcal { U } \quad | \quad \mathbf { x } = \mathbf { x } _ { 0 } ( \mathbf { u } _ { 0 } ) \} .\tag{14}
$$

The spectrum of each stable fixed point of the critical manifold, denoted by Spec $( { \bf D _ { x } F ( x _ { 0 } ( u _ { 0 } ) , u _ { 0 } ) } ) = \{ \lambda _ { \nu } ( { \bf u } _ { 0 } ) \} _ { \nu = 0 } ^ { n }$ , is further assumed to have a strictly nonzero spectral gap at $\nu = d$ for all $\mathbf { u } _ { 0 } \in \mathcal { U }$ , i.e.,

$$
\operatorname { R e } ( \lambda _ { n } ( \mathbf { u } _ { 0 } ) ) \leq \dots \leq \operatorname { R e } ( \lambda _ { d + 1 } ( \mathbf { u } _ { 0 } ) ) < \operatorname { R e } ( \lambda _ { d } ( \mathbf { u } _ { 0 } ) ) \leq \dots \leq \operatorname { R e } ( \lambda _ { 1 } ( \mathbf { u } _ { 0 } ) ) < 0 .\tag{15}
$$

This allows us to define the slow spectral subbundle $E ( \mathbf { u } _ { 0 } )$ associated to the eigespaces of the d slowest eigenvalues,

$$
E ( \mathbf { u } _ { 0 } ) = \bigoplus _ { \nu = 1 } ^ { d } E _ { \nu } ( \mathbf { u } _ { 0 } ) .\tag{16}
$$

If further non-resonance conditions hold, this slow subbundle of the linearized problem survives as a slow spectral submanifold (SSM) bundle $\mathcal { A } _ { 0 } = \mathrm { ~ \bigcup ~ } \mathcal { W } ( E ( \mathbf { u } _ { 0 } ) )$ (see Haller u<sub>0</sub>∈U and Kaundinya [21]). This slow SSM bundle is normally attracting and anchored to the critical manifold. Haller and Kaundinya [21] further proves that for relatively slow variations of the control input $\mathbf { u } ~ \equiv ~ \mathbf { u } ^ { s } ( \epsilon t ) , ~ 0 ~ < ~ \epsilon ~ < < ~ 1$ , the slow SSM subbundle $\mathcal { A } _ { \mathrm { 0 } }$ survives as a nearby difeomorphic invariant manifold $\boldsymbol { A } _ { \epsilon }$ , which we call the adiabatic spectral submanifold (aSSM). The critical manifold also survives into a nearby invariant slow manifold $\mathcal { L } _ { \epsilon } ,$ with parametrization $\mathbf { x } _ { \epsilon } ( \mathbf { u } ( \epsilon t ) )$ . Details pertaining to using this for controls can be found in Bettini et al. [6].

In Kaundinya et al. [27], for control inputs of the form

$$
\begin{array} { r } { \mathbf { u } ( t ) \equiv \mathbf { u } ^ { s } ( \epsilon t ) + \mathbf { u } ^ { d } ( t ) , \quad \lVert \mathbf { u } ^ { d } ( t ) \rVert < \Delta , \quad \Delta > 0 , } \end{array}\tag{17}
$$

the survival of the $\mathrm { a S S M } , A _ { \epsilon }$ , as a nearby invariant manifold, $\tilde { \mathcal { A } } _ { \epsilon }$ , was discussed. They also provided explicit formulas of the aSSM-reduced model accounting for dynamics restricted to $\tilde { \mathcal { A } } _ { \epsilon }$ . While their formulas accounted only for the leading-order correction from the data, here we extend their results by deriving an aSSM-reduced model with higher-order correction terms that can be learned from data.

To that end, we note in soft robotic experiments, data representing the full phase space dynamics is unavailable. In fact, most motion capture systems record only positions and orientations from rigid-body markers attached to the soft robot. In such situations, we seek embeddings of the aSSM and its dynamics using data from the soft robot’s observable space, $\mathbf { y } \in \mathbb { R } ^ { n _ { y } } , n _ { y } \ll n$ . Specifically, we use the delay embedding function $\mathrm { e q . ~ ( 5 ) }$ , with $\begin{array} { r } { p = \left\lfloor \frac { 2 d + 1 } { n _ { y } } \right\rfloor } \end{array}$ , to generate a delay embedded observable space by y that guarantees the reconstruction of the d dimensional aSSM $\mathcal { A } _ { \epsilon }$ . We further denote the critical manifold parametrization in the observable space as $\mathbf { y } _ { 0 } ( \cdot )$

We provide in Supplementary Section 5 a detailed derivation for the higher-order aSSM-reduced model in the phase space coordinates, x. The derivation in that case parametrized the aSSM as a graph over the time-varying slow spectral subbundle $E ( \mathbf { u } ^ { s } ( \epsilon t ) )$ . We similarly seek the aSSM as a graph over the corresponding $E ( \mathbf { u } ^ { s } ( \epsilon t ) )$ in the delay embedding observable space. To that end, we denote a time-varying matrix $\mathbf { V } ( \mathbf { u } ^ { s } ( \epsilon t ) )$ whose columns comprise the vectors spanning $E ( \mathbf { u } ^ { s } ( \epsilon t ) )$ in the delay embedding observable space. Using these together with the general form of the aSSMreduced model eq. (15) in the phase space, we obtain the d-dimensional aSSM-reduced model in the delay-embedded observable space as:

$$
\begin{array} { r l } & { \mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ) \pmb { \mu } ( \mathbf { y } - \mathbf { y } _ { 0 } ( \mathbf { u } ) ) , \qquad \mathbf { u } \in \mathcal { U } , } \\ & { \dot { \mathbf { r } } = \mathbf { R } ( \mathbf { r } , \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) ) + \mathbf { B } \left( \mathbf { u } ( t ) \right) \mathbf { D } _ { \mathbf { u } } \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } , } \\ & { \mathbf { y } = \mathbf { W } ( \mathbf { r } , \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) ) + \mathbf { y } _ { 0 } ( \mathbf { u } ( t ) ) . } \end{array}\tag{18}
$$

Here $\mathbf { r } \in \mathbb { R } ^ { d }$ is the reduced coordinate obtained via an orthogonal projection onto $E ( \mathbf { u } ^ { s } ( \epsilon t ) ) , \mathbf { R } ( \cdot )$ is the leading-order aSSM reduced dynamics, $\mathbf { B } ( \cdot )$ is calibration matrix that is accounting for a higher-order term in the aSSM reduced dynamics dependent on u˙ , W( ) is the leading-order aSSM parametrization.

Note that in the delay-embedded observable space, the spectral subspace families of the linearized system about the critical manifold are generally non-normal (see Bettini et al. [5]). Recent work by Bettini et al. [5] provides methods to recover general oblique projections for SSMs from data. For aSSMs, this methodology is still missing, hence we resort to orthogonal projections but account for the projection errors by learning a calibration matrix B (u(t)) for the reduced dynamics.

## 3.2 Data-driven identification of the aSSM-reduced model

We collect the robot’s response to step control input data in the observable space. The step control inputs are generated by performing jumps held for $t _ { \mathrm { d e c a y } } { - } \mathrm { l o n g }$ seconds between the m most distant control inputs for each element in the static control input set $\{ \mathbf { U } _ { \mathrm { s t a t i c } } ^ { i } \} _ { i = 1 } ^ { N }$ . The parameter $t _ { \mathrm { d e c a y } }$ is chosen based on the time required for the robot to settle into a static configuration, and m is chosen to sample the observable space well. We set $m = 4$ for our applications. The set of static control inputs is sampled using the Latin hypercube sampling (LHS) technique (see McKay et al. [30]). We denote this dataset as $\{ \mathbf { Y } _ { \mathrm { s t e p } } , \mathbf { U } _ { \mathrm { s t e p } } \}$

We further split this dataset into static

$$
\{ \mathbf { Y } _ { \mathrm { s t a t i c } } ^ { i } , \mathbf { U } _ { \mathrm { s t a t i c } } ^ { i } \} _ { i = 1 } ^ { N } ,\tag{19}
$$

and decaying

$$
\left\{ \begin{array} { l l } { \begin{array} { r l r } & { \mathbf { Y } _ { \mathrm { d e c a y } } ^ { i } \in \mathbb { R } ^ { n _ { y } \times T } , } & { \mathbf { U } _ { \mathrm { d e c a y } } ^ { i } \in \mathbb { R } ^ { n _ { u } \times T } } \end{array} } \end{array} \right\} _ { i = 1 } ^ { m N } ,\tag{20}
$$

datasets, where $T$ is the temporal length of the decaying trajectory. We also collect the robot’s response to random smooth control data generated with Perlin noise (Perlin [35]). We denote this response by $\{ \mathbf { Y } _ { \mathrm { c a l i b } } , \mathbf { U } _ { \mathrm { c a l i b } } \}$

## 3.2.1 Learning the critical manifold

We learn the critical manifold parametrization $\mathbf { y } _ { 0 } ( \mathbf { u } _ { 0 } )$ in the observable space as a nonlinear static mapping fit to the static dataset. The corresponding nonlinear regression problem takes the form

$$
\mathbf { y } _ { 0 } ^ { \star } ( \mathbf { u } ) = \underset { \mathbf { y } _ { 0 } : \mathcal { U } \subset \mathbb { R } ^ { n _ { u } } \to \mathbb { R } ^ { n _ { y } } } { \arg \operatorname* { m i n } } \sum _ { i } ^ { N } \big \Vert \mathbf { Y } _ { \mathrm { s t a t i c } } ^ { i } - \mathbf { y } _ { 0 } ( \mathbf { U } _ { \mathrm { s t a t i c } } ^ { i } ) \big \Vert ^ { 2 } .\tag{21}
$$

We may seek the critical manifold $\mathcal { L } _ { 0 }$ in the observable space in the following three diferent ways:

(1) A polynomial expansion with polynomial order $k _ { u }$ ,

$$
\mathbf { y } _ { 0 } ( \mathbf { u } ) = \sum _ { 0 \leq | \alpha | \leq k _ { u } } \mathbf { c } _ { \alpha } \mathbf { u } ^ { \alpha } ,\tag{22}
$$

where $_ { \pmb { \alpha } }$ is a multi-index and ${ \bf c } _ { \alpha }$ are the polynomial coeficients.

(2) A radial basis function expansion,

$$
\mathbf { y } _ { 0 } ( \mathbf { u } ) = \sum _ { j = 1 } ^ { n _ { c } } \mathbf { w } _ { j } \phi \big ( r _ { j } \big ) , \qquad r _ { j } = \sqrt { \| \mathbf { u } - \mathbf { c } _ { u } ^ { j } \| ^ { 2 } + \epsilon } ,\tag{23}
$$

where $\mathbf { c } _ { u } ^ { j } , j = 1 , \ldots , n _ { c } ,$ are the centers, ${ \bf w } _ { j }$ are the corresponding weights, and ϵ = $1 0 ^ { - 7 }$ is a tolerance added to regularize against the singularity at $r _ { j } = 0$ . The kernel of $\phi ( \cdot )$ is chosen as either a linear kernel,

$$
\phi ( r ) = r ,\tag{24}
$$

or a Gaussian kernel,

$$
\phi ( r ) = \exp \left( - \frac { r ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{25}
$$

with parameter σ controlling the width of each Gaussian basis function.

(3) A neural network,

$$
\mathbf { y } _ { 0 } ( \mathbf { u } ) = \mathcal { N } _ { \pmb { \theta } } ( \mathbf { u } ) ,\tag{26}
$$

where $\mathcal { N } _ { \theta }$ denotes the network with a trainable parameter vector $\pmb { \theta } .$

For the first two parameterizations, the expansion is linear in the unknown coeficients $\left( \mathbf { c } _ { \alpha } \ \mathrm { o r } \ \mathbf { w } _ { j } \right)$ . Hence, the regression problem reduces to a linear leastsquares problem that admits a quick, closed-form solution. The neural network

parameterization, in contrast, is optimized using stochastic gradient descent due to its nonlinear dependence on θ.

To fit the critical manifold outlined above, we are not limited to static datasets generated solely by step inputs. We can also use slowly varying control inputs that span the robot’s observable space, in a quasi-static sense, to enrich the static datasets.

## 3.2.2 Learning the aSSM paramterization and reduced dynamics

We center each decaying observable data about the static configuration they converge to we also truncate the initial part of the decaying signal at time time $t _ { 0 }$

$$
{ \bf Y } _ { \mathrm { c e n t e r e d } } ^ { i } = ( { \bf Y } _ { \mathrm { d e c a y } } ^ { i } ) _ { : , t _ { 0 } : T } - ( { \bf Y } _ { \mathrm { d e c a y } } ^ { i } ) _ { : , T } .\tag{27}
$$

The truncation parameter $t _ { 0 }$ is set to remove the initial transients that are still far from the dominant slow SSM. We then infer the minimal dimension $d$ of the slow SSM slices that capture the centered decaying data, based on possible symmetries of the robot and a spectrogram analysis of the decaying data for underdamped modes (see Kaundinya et al. [27] for details). Using the inferred dimension $d ,$ we delay embed the centered data $p$ times, where $( p + 1 ) n _ { y } \geq 2 d + 1$ to guarantee a delay embedded observable to embed the slow SSM slices (see Takens [43]). We construct the delay-embedded dataset as follows,

$$
\begin{array} { r } { \mathbf { Y } _ { \mathrm { d e l a y } } ^ { i } = \left[ \begin{array} { l } { ( \mathbf { Y } _ { \mathrm { c e n t e r e d } } ^ { i } ) _ { : , p + 1 : T } } \\ { ( \mathbf { Y } _ { \mathrm { c e n t e r e d } } ^ { i } ) _ { : , p : T - 1 } } \\ { \qquad \vdots } \\ { ( \mathbf { Y } _ { \mathrm { c e n t e r e d } } ^ { i } ) _ { : , 1 : T - p } } \end{array} \right] \in \mathbb { R } ^ { ( p + 1 ) n _ { y } \times ( T - p ) } . } \end{array}\tag{28}
$$

Equipped with the delay embedded dataset $\{ \mathbf { Y } _ { \mathrm { d e l a y } } ^ { i } \} _ { i = 1 } ^ { m N }$ , and grouping m of the decays corresponding to the same static configuration, we perform an economy singular value decomposition on the grouped dataset to obtain a parameter set of chart maps $\left\{ { \bf V } ^ { i } \in \mathbb { R } ^ { ( p + 1 ) n _ { y } \times d } \right\} _ { i = 1 } ^ { N }$ . We learn interpolants $\mathbf { V } \left( \cdot \right)$ for the chart maps over the static set of inputs $\mathbf { U } _ { \mathrm { s t a t i c } }$ . In some robotics applications, the charts for the near configurations around each static configuration can be very similar. In that case, we can learn a global chart map $\mathbf { V } _ { g } \ \in \ \bar { \mathbb { R } } ^ { ( p + 1 ) n _ { y } \times d }$ by performing the singular value decomposition on the entire dataset. Using the learned chart maps, we construct data lying on the reduced space of the aSSM,

$$
\begin{array} { r } { \mathbf { \mathcal { R } } ^ { i } = ( \mathbf { V } ^ { i } ) ^ { \mathrm { T } } \mathbf { Y } _ { \mathrm { d e l a y } } ^ { i } , } \end{array}\tag{29}
$$

and via finite diferencing on $\mathcal { R } ^ { i }$ we obtain reduced dynamics data on the aSSM, which we label as $\dot { \pmb R } ^ { i }$

We seek the leading-order aSSM parametrization and the aSSM-reduced dynamics as polynomial expressions, given by

$$
\mathbf { W } ( \mathbf { r } , \mathbf { y } ) = \sum _ { \substack { 1 \leq | \alpha | \leq k _ { r } ^ { W } 0 \leq | \beta | \leq k _ { y } ^ { W } } } \mathbf { W } _ { \alpha \beta } \mathbf { r } ^ { \alpha } \mathbf { y } ^ { \beta } ,\tag{30}
$$

and the aSSM-reduced dynamics as

$$
\mathbf { R } ( \mathbf { r } , \mathbf { y } ) = \sum _ { 1 \leq | \alpha | \leq k _ { r } ^ { R } 0 \leq | \beta | \leq k _ { y } ^ { R } } \mathbf { R } _ { \alpha \beta } \mathbf { r } ^ { \alpha } \mathbf { y } ^ { \beta } ,\tag{31}
$$

where α and $\beta$ are multi-indices over r and y respectively, $k _ { r } ^ { W } , k _ { y } ^ { W }$ and $k _ { r } ^ { R } , k _ { y } ^ { R }$ denote the corresponding polynomial orders, and $\mathbf { w } _ { \alpha \beta } , \ \mathbf { r } _ { \alpha \beta }$ are the unknown polynomial coeficients.

We solve for the polynomial coeficients by minimizing the reconstruction and dynamics regression losses

$$
\mathbf { W } ^ { \star } ( \mathbf { r } , \mathbf { y } ) = \underset { \mathbf { W } } { \arg \operatorname* { m i n } } \sum _ { i } \big \| \mathbf { Y } _ { \mathrm { c e n t e r e d } } ^ { i } - \mathbf { W } \big ( \mathcal { R } ^ { i } , \big ( \mathbf { Y } _ { \mathrm { d e c a y } } ^ { i } \big ) _ { : , T } \big ) \big \| ^ { 2 } ,\tag{32}
$$

$$
\mathbf { R } ^ { \star } ( \mathbf { r } , \mathbf { y } ) = \underset { \mathbf { R } } { \arg \operatorname* { m i n } } \sum _ { i } \big \| \dot { \pmb { \mathcal { R } } } ^ { i } - \mathbf { R } \big ( \mathcal { R } ^ { i } , \big ( \mathbf { Y } _ { \mathrm { d e c a y } } ^ { i } \big ) _ { : , T } \big ) \big \| ^ { 2 } .\tag{33}
$$

Finally, using the learned leading-order aSSM parametrization $\mathbf { W } ^ { \star }$ , the aSSMreduced dynamics $\mathbf { R } ^ { \star }$ , and the critical manifold parametrization ${ \bf y } _ { 0 } ( { \bf u } )$ , we infer the calibration matrix B(u) as a polynomial expression,

$$
\mathbf { B } ( \mathbf { u } ) = \sum _ { 0 \leq | \alpha | \leq k _ { u } ^ { B } } \mathbf { B } _ { \alpha } \mathbf { u } ^ { \alpha } ,\tag{34}
$$

by minimizing the following regression problem using the calibration dataset:

$$
\mathbf { B } ^ { \star } ( \mathbf { u } ) = \underset { \mathbf { B } ( \mathbf { u } ) } { \arg \operatorname* { m i n } } \left\| \dot { \mathcal { R } } _ { \mathrm { c a l i b } } - \mathbf { R } ^ { \star } \big ( \mathcal { R } _ { \mathrm { c a l i b } } , \mathbf { y } _ { 0 } ^ { \star } ( \mathbf { U } _ { \mathrm { c a l i b } } ) \big ) - \right.\tag{35}
$$

$$
\mathbf { B } \big ( \mathbf { U } _ { \mathrm { c a l i b } } \big ) \mathbf { D } _ { \mathbf { u } } \mathbf { y } _ { 0 } ^ { \star } \big ( \mathbf { U } _ { \mathrm { c a l i b } } \big ) \dot { \mathbf { U } } _ { \mathrm { c a l i b } } \big \| ^ { 2 } ,\tag{36}
$$

Here $\pmb { \mathcal { R } } _ { \mathrm { c a l i b } } = \mathbf { V } ^ { \mathrm { T } } ( \mathbf { U } _ { \mathrm { c a l i b } } ) \mathbf { Y } _ { \mathrm { c a l i b } }$ and $\dot { \mathcal { R } } _ { \mathrm { c a l i b } }$ is obtained by finite-diferencing $\mathcal { R } _ { \mathrm { c a l i b } }$ The choice of all parameters involved in deducing these terms is optimized to reduce the normalized mean trajectory error on test data. We also test the predictive power of the learned aSSM-reduced model on unseen control inputs ${ \bf \delta u } ( t )$ before deploying it for closed-loop control.

## 3.3 Model predictive control using aSSM-reduced model

We use the aSSM-reduced model in eq. (18) in a model predictive control (MPC) scheme to track a target track $\mathbf { \boldsymbol { \Gamma } } ( t )$ . The MPC is formulated as a finite-horizon optimization problem. The finite horizon is set up from time $t _ { j }$ recorded on the camera sensors monitoring the robot, with a user-defined window uniformly spaced in the space window width $M \Delta t$ , with time spacing ∆t. We also take into account motion capture $\mathbf { y } ( t _ { j } )$ . We then solve for the optimal control inputs that will follow the target track in this window,

$$
\begin{array} { r l } { \mathrm { m i n i m i z e } _ { { \mathbf { u } } ( \cdot ) } } & { \displaystyle \int _ { t _ { j } } ^ { t _ { j } + M \Delta t } \Big ( | | { \mathbf { z } } ( s ) - \Gamma ( s ) | | _ { { \mathbf { Q } } _ { \mathbf { z } } } ^ { 2 } + | | { \mathbf { u } } ( s ) | | _ { { \mathbf { R } } _ { \mathbf { u } } } ^ { 2 } + | | \dot { { \mathbf { u } } } ( s ) | | _ { { \mathbf { R } } _ { \mathrm { i } } } ^ { 2 } \Big ) d s , } \\ { \mathrm { s u b j e c t ~ t o } } & { { \mathbf { r } } ( t _ { j } ) = { \mathbf { V } } ^ { { \mathrm { T } } } ( { \mathbf { u } } ( t _ { j } ) ) \mu ( { \mathbf { y } } ( t _ { j } ) - { \mathbf { y } } _ { 0 } ( { \mathbf { u } } ( t _ { j } ) ) ) , } \\ & { \dot { \mathbf { r } } = { \mathbf { R } } ( \mathbf { r } , { \mathbf { u } } ( t ) ) + { \mathbf { B } } ( { \mathbf { u } } ( t ) ) { \mathbf { D } } _ { { \mathbf { u } } { \mathbf { y } } _ { 0 } } ( { \mathbf { u } } ( t ) ) \dot { { \mathbf { u } } } ( t ) , } \\ & { \mathbf { y } ( t ) = \mathbf { W } ( { \mathbf { r } } ( t ) , { \mathbf { u } } ( t ) ) + { \mathbf { y } } _ { 0 } ( { \mathbf { u } } ( t ) ) , } \\ & { \mathbf { y } ( t ) \in \mathcal { V } , \quad \mathbf { u } ( t ) \in \mathcal { U } , } \\ & { \mathbf { z } ( t ) = { \mathbf { C } } ( { \mathbf { y } } ( t ) ) . } \end{array}\tag{37}
$$

(38)

The non-convex optimization problem (37) is solved using the GuSTO algorithm of Bonalli et al. [7], which recasts the problem as a sequence of convex local optimal control problems (LOCPs). At each iteration, the resulting LOCP is a quadratic program, and we solve it using the high-performance interior-point solver (HPIPM) by Frison and Diehl [14]. To enable real-time operation, only the first two control inputs of the optimal sequence are applied to the robot, after which the prediction horizon is shifted forward by the same amount and the LOCP is resolved, yielding a receding-horizon implementation.

## Acknowledgments

This work has been partially supported by grant 200021 214908 from the Swiss National Science Foundation (SNF).

## Author Contributions

G.H., M.P., J.A. and R.K. designed the research. R.K. and M.L carried out the research. M.L. and R.S. designed the hardware. R.K., M.L., H.B. and L.P. developed the software. R.K. wrote the paper. G.H., M.P. and R.K. led the research team. M.P. provided funding for hardware resources.

## Competing Interests

All authors declare no financial or non-financial competing interests.

## References

[1] I. Adibnazari, H. Sharma, M. Park, J. Cervera-Torralba, B. Kramer, and M. T. Tolley. Dynamic shape control of soft robots enabled by data-driven model reduction, 2025. URL https://arxiv.org/abs/2511.03931. arXiv:2511.03931.

[2] J. Alora, M. Cenedese, E. Schmerling, G. Haller, and M. Pavone. Data-driven spectral submanifold reduction for nonlinear optimal control of high-dimensional robots. Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pages 2627–2633, 2023.

[3] J. I. Alora, M. Cenedese, G. Haller, and M. Pavone. Discovering dominant dynamics for nonlinear continuum robot control. npj Robotics, 3(1):5, 2025. doi: 10.1038/s44182-025-00021-8. URL https://doi.org/10.1038/s44182-025-00021-8.

[4] J. M. Bern, Y. Schnider, P. Banzet, N. Kumar, and S. Coros. Soft robot control with a learned diferentiable model. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 417–423, 2020. doi: 10.1109/RoboSoft48309.2020.9115993. Relevant for soft robot control and simulation, applicable to soft silicone modeling and nonlinear dynamics in robotic simulators.

[5] L. Bettini, A. Kazemipour, R. K. Katzschmann, and G. Haller. General oblique projections for model reduction via spectral submanifolds. Chaos, 36(5):053122, 2026.

[6] L. Bettini, A. Kazemipour, R. K. Katzschmann, and G. Haller. Nonlinear spectral modeling and control of soft-robotic muscles from data, 2026. URL https://arxiv. org/abs/2601.03247. arXiv:2601.03247.

[7] R. Bonalli, A. Cauligi, A. Bylard, and M. Pavone. Gusto: Guaranteed sequential trajectory optimization via sequential convex programming. In Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pages 6741–6747. IEEE Press, 2019. doi: 10.1109/ ICRA.2019.8794205.

[8] D. Bruder, X. Fu, R. Gillespie, C. Remy, and R. Vasudevan. Data-driven control of soft robots using Koopman operator theory. IEEE Trans. Robot., 37(3):948–961, 2021.

[9] Carbon, Inc. EPU 46: Elastomeric polyurethane. https://www.carbon3d.com/ materials/epu-46, 2026. Accessed: 2026-09-27.

[10] L. Catalasan and T. He. Memory-dependent dynamic modeling of cable-driven soft robots using the mori-zwanzig koopman operator. Nonlinear Dyn., 114(2): 77, 2026.

[11] M. Cenedese, J. Ax˚as, B. B¨auerlein, K. Avila, and G. Haller. Data-driven modeling and prediction of non-linearizable dynamics via spectral submanifolds. Nat. Commun., 13(1):872, 2022.

[12] F. Faure, C. Duriez, H. Delingette, J. Allard, B. Gilles, S. Marchesseau, H. Talbot, H. Courtecuisse, G. Bousquet, I. Peterlik, et al. Sofa: A multi-model framework for interactive physical simulation. Soft tissue biomechanical modeling for computer assisted surgery, pages 283–321, 2012.

[13] N. Fenichel. Geometric singular perturbation theory for ordinary diferential equations. J. Difer. Equ., 31(1):53–98, 1979.

[14] G. Frison and M. Diehl. HPIPM: A high-performance quadratic programming framework for model predictive control. IFAC-PapersOnLine, 53(2):6563–6569, 2020.

[15] J. Gao, M. Y. Michelis, A. Spielberg, and R. K. Katzschmann. Sim-to-real of soft robots with learned residual physics. IEEE Robot. Autom. Lett., 9(10):8745–8752, August 2024. doi: 10.1109/LRA.2024.3445379.

[16] O. Goury and C. Duriez. Fast, generic, and reliable control and simulation of soft robots using model order reduction. IEEE Trans. Robot., 34(6):1565–1576, 2018.

[17] Q. Guan, F. Stella, C. Della Santina, J. Leng, and J. Hughes. Trimmed helicoids: an architectured soft structure yielding soft robots with high precision, large workspace, and compliant interactions. npj Robotics, 1(1):4, 2023.

[18] P. Gupta, P. J. Schmid, D. Sipp, T. Sayadi, and G. Rigas. Mori–zwanzig latent space koopman closure for nonlinear autoencoder. Proc. R. Soc. A, 481(2313), 2025.

[19] D. A. Haggerty, M. J. Banks, E. Kamenar, A. B. Cao, P. C. Curtis, I. Mezi´c, and E. W. Hawkes. Control of soft robots with inertial dynamics. Sci. Robot., 8(81): eadd6864, 2023.

[20] G. Haller. Modeling Nonlinear Dynamics from Equations and Data — with Applications to Solids, Fluids, and Controls. SIAM, Philadelphia, PA, 2025.

[21] G. Haller and R. S. Kaundinya. Nonlinear model reduction to temporally aperiodic spectral submanifolds. Chaos, 34(4):043152, Apr. 2024. ISSN 1054-1500. doi: 10.1063/5.0187080. URL https://doi.org/10.1063/5.0187080.

[22] G. Haller and S. Ponsioen. Nonlinear normal modes and spectral submanifolds: existence, uniqueness and use in model reduction. Nonlinear Dyn., 86(3):1493– 1534, 2016.

[23] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter. Learning agile and dynamic motor skills for legged robots. Sci. Robot., 4(26):eaau5872, 2019.

[24] R. Jitosho, T. G. W. Lum, A. Okamura, and K. Liu. Reinforcement learning enables real-time planning and control of agile maneuvers for soft robot arms. In J. Tan, M. Toussaint, and K. Darvish, editors, Proc. Conf. Robot Learn. (CoRL), volume 229 of Proceedings of Machine Learning Research, pages 1131–1153. PMLR, 06–09 Nov 2023. URL https://proceedings.mlr.press/v229/jitosho23a. html.

[25] R. K. Katzschmann, C. Della Santina, Y. Toshimitsu, A. Bicchi, and D. Rus. Dynamic motion control of multi-segment soft robots using piecewise constant curvature matched with an augmented rigid body model. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 454–461, 2019. doi: 10.1109/ROBOSOFT. 2019.8722799.

[26] R. K. Katzschmann, M. Thiefry, O. Goury, A. Kruszewski, T.-M. Guerra, C. Duriez, and D. Rus. Dynamically closed-loop controlled soft robotic arm using a reduced order finite element model with state observer. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 717–724, 2019.

[27] R. S. Kaundinya, J. I. Alora, J. G. Matt, L. A. Pabon, M. Pavone, and G. Haller. Data-driven soft robot control via adiabatic spectral submanifolds. Int. J. Robot. Res., 2026. doi: 10.1177/02783649261461630.

[28] S. Levine, C. Finn, T. Darrell, and P. Abbeel. End-to-end training of deep visuomotor policies. J. Mach. Learn. Res., 17(39):1–40, 2016.

[29] Y. T. Lin, Y. Tian, D. Livescu, and M. Anghel. Data-driven learning for the mori–zwanzig formalism: A generalization of the koopman learning framework. SIAM J. Appl. Dyn. Syst., 20(4):2558–2601, 2021.

[30] M. D. McKay, R. J. Beckman, and W. J. Conover. A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2):239–245, 1979. ISSN 00401706.

[31] G. Mengaldo, F. Renda, S. L. Brunton, M. B¨acher, M. Calisti, C. Duriez, G. S. Chirikjian, and C. Laschi. A concise guide to modelling the physics of embodied intelligence in soft robotics. Nat. Rev. Phys., 4(9):595–610, 2022.

[32] A. A. Morsy, Z. Xu, P. Tiso, and G. Haller. Data-driven reduction of the finiteelement model of a Tribomechadynamics benchmark problem. Nonlinear Dyn., 113(18):24539–24555, 2025. doi: 10.1007/s11071-025-11378-8.

[33] F. Nan, H. Ma, Q. Guan, J. Hughes, M. Muehlebach, and M. Hutter. Eficient model-based reinforcement learning for robot control via online optimization, 2025. URL https://arxiv.org/abs/2510.18518. arXiv:2510.18518.

[34] N. Naughton, J. Sun, A. Tekinalp, T. Parthasarathy, G. Chowdhary, and M. Gazzola. Elastica: A compliant mechanics environment for soft robotic control. IEEE Robot. Autom. Lett., 6(2):3389–3396, 2021. doi: 10.1109/LRA. 2021.3063698.

[35] K. Perlin. An image synthesizer. In Proc. Annu. Conf. Comput. Graph. Interact. Tech. (SIGGRAPH), SIGGRAPH ’85, pages 287–296, New York, NY, 1985.

Association for Computing Machinery. ISBN 0897911660. doi: 10.1145/325334. 325247.

[36] J. L. Proctor, S. L. Brunton, and J. N. Kutz. Dynamic mode decomposition with control. SIAM J. Appl. Dyn. Syst., 15(1):142–161, 2016.

[37] F. Renda, F. Boyer, J. Dias, and L. Seneviratne. Discrete cosserat approach for multisection soft manipulator dynamics. IEEE Trans. Robot., 34(6):1518–1533, 2018.

[38] C. Rowley, I. Mezi´c, S. Bagheri, P. Schlachter, and D. Henningson. Spectral analysis of nonlinear flows. J. Fluid Mech., 641:115–127, 2009.

[39] R. Sabella, B. M. V. Ostuni, T. Caporaso, S. Grazioso, and A. Lanzotti. Indices for environmental sustainability in pneumatic soft robotic actuators: A systematic review. In International Conference of the Italian Association of Design Methods and Tools for Industrial Engineering, pages 601–609. Springer, 2023.

[40] S. Sastry. Nonlinear Systems: Analysis, Stability, and Control. Springer, New York, NY, 1999.

[41] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347. arXiv:1707.06347.

[42] M. St¨olzle and C. Della Santina. Input-to-state stable coupled oscillator networks for closed-form model-based control in latent space. In Adv. Neural Inf. Process. Syst. (NeurIPS), volume 37, pages 82010–82059, Red Hook, NY, 2024. Curran Associates, Inc.

[43] F. Takens. Detecting strange attractors in turbulence. In D. Rand and L. Young, editors, Dynamical Systems and Turbulence, Warwick 1980, pages 366–381. Springer, Berlin, Heidelberg, 1981.

[44] T. G. Thuruthel, E. Falotico, F. Renda, and C. Laschi. Model-based reinforcement learning for closed-loop dynamic control of soft robotic manipulators. IEEE Trans. Robot., 35:124–134, 2019. URL https://api.semanticscholar.org/CorpusID: 59618928.

[45] S. Tonkens, J. Lorenzetti, and M. Pavone. Soft robot optimal control via reduced order finite element models. In Proc. IEEE Int. Conf. Robot. Autom. (ICRA), pages 12010–12016, 2021. doi: 10.1109/ICRA48506.2021.9560999.

[46] F. E. Udwadia. A new perspective on the tracking control of nonlinear structural and mechanical systems. Proc. R. Soc. A, 459(2035):1783–1800, 2003.

[47] R. Valadas, M. St¨olzle, J. Liu, and C. Della Santina. Learning low-dimensional strain models of soft robots by looking at the evolution of their shape with application to model-based control. In Proc. IEEE Int. Conf. Soft Robot. (RoboSoft), pages 1–8, 2025. doi: 10.1109/RoboSoft63089.2025.11020829.

[48] K. Wong, M. St¨olzle, W. Xiao, and D. Rus. A closed-form CLF-CBF controller for whole-body continuum soft robot collision avoidance, 2026. URL https:// arxiv.org/abs/2603.19424. arXiv:2603.19424.

[49] O. Yasa, Y. Toshimitsu, M. Y. Michelis, L. S. Jones, M. Filippi, T. Buchner, and R. K. Katzschmann. An overview of soft robotics. Annu. Rev. Control Robot.

Auton. Syst., 6(Volume 6, 2023):1–29, 2023. ISSN 2573-5144. doi: https://doi. org/10.1146/annurev-control-062322-100607. URL https://www.annualreviews. org/content/journals/10.1146/annurev-control-062322-100607.

[50] A. Zhang, H. Matusik, M. Flores-Acton, E. R. Sologuren, J. Jacob, and D. Rus. Printed helicoids with embedded air channels make sensorized segments for soft continuum robots, 2026. URL https://arxiv.org/abs/2602. 23457. arXiv:2602.23457.

[51] H. Zheng, B. Sukhija, C. Li, K. Iten, A. Krause, and R. K. Katzschmann. Learning soft robotic dynamics with active exploration, 2025. URL https://arxiv.org/abs/ 2510.27428. arXiv:2510.27428.

# Supplementary Material for Data-Driven Precision Control of Soft Helicoid Robots

## 1 Soft helicoid design: Morphology and fabrication trade-ofs

## 1.1 Morphology comparison

Once buckling initiates in a homogeneous hollow-cylinder morphology, the monotonic relationship between tendon displacement and backbone curvature is broken, causing the trunk to collapse rather than bend smoothly. A structural bifurcation occurs, rendering the kinematic mapping invalid and precluding accurate control. This motivated the choice of the Trimmed Helicoid (TH) architecture (Section 1.1), which independently tunes axial and bending stifness to avoid this failure mode.

## 1.2 Fabrication method comparison

Manual casting was discarded due to the complexity of the multi-stage molding process required for the internal topology, which introduces alignment errors and limits the geometric intricacy that can be realized. Extrusion-based methods were rejected because their layer-by-layer anisotropy creates stress concentrations that are prone to delamination under high-frequency cyclic loading (see Yap et al. [9] for a review). Digital Light Synthesis (DLS™), driven by Continuous Liquid Interface Production $\left( \mathrm { C L I P ^ { \mathsf { T M } } } \right)$ , avoids both issues: an oxygen-permeable window inhibits polymerization at the interface (dead zone), enabling continuous part generation and monolithic, isotropic structures. EPU 46 was selected for this process for its high resilience and tear strength alongside its sustainability (40% bio-based).

## 1.3 Design-space exploration

Existing theoretical models for TH structures are restricted to small-deformation regimes and do not capture the viscoelastic behavior of the printed elastomer. Also, the lack of prior characterization of the TH–DLS–EPU 46 hardware triad made predictive simulation unreliable. We therefore iterated through sixteen morphological variations (Fig. 1a) to empirically converge on parameters P1–P4. Microstructures (P4) reduced weight but introduced stochastic friction, degrading control transparency, so a solid-wall topology was selected instead.

## 2 Closed-loop results for a faster varying 3D Pacman track

We present here closed-loop experimental results for the 3D Pacman track (appearing in the main text in Section 1.4) in Fig. 1. This track is speeded up to have a mean instantaneous speed of $v _ { \Gamma } = 1 2 ~ [ \mathrm { c m / s } ]$ with slowness measure $r _ { s } ~ = ~ 0 . 2$ . This is a challenging 3D track, as within a span of a second, the trunk must make a sharp liftof. We find that, despite delays and motor hardware uncertainty, the 5D aSSM-based controller provides the most consistent performance among all baselines considered. The closed-loop errors start to compound for the RL policy and the MZ-Koopman controller, leading to highly inaccurate tracking performance. The linear methods performs without delay because it is fast, but ultimately fail to lift the trunk and maintain the height. We support these results with pointwise relative error plots in Fig. (2).

aSSM (5D)  
![](images/1774fb61e9f1e48ff5e9c3aa3a2156f0eec16b7f7614dc0e9a22a4d97303d395.jpg)

DMDc (6D)  
![](images/8669956119920dae8e3b4c233ff43be87f36082cf4f1f64bea826b5bd6d710d8.jpg)

Koopman-pregain (6D)  
![](images/139a0fe2fdc96047305178c2e570c467529496210035b87a2da8cc64c92b8a61.jpg)

![](images/83f52a9248688d0296ac7cef1332446cb2c1ac0fd0f561ffb97e9e1a7c3b01f9.jpg)

![](images/91568d09ccdfb0496350db4959f7359be4e7aaf93b814437e450009b8370ce6b.jpg)  
Fig. 1: Closed-loop tracking results for an extremely fast 3D Pacman track (black) with the uncertainty tube of radius 3 [cm] plotted in gray. The average instantaneous speed along the target track is 12 [cm/s].

![](images/049e8a2dcbd04cc263cd544ce3550d60fedc3e629f7fb48fcb75ce3fd79f9083.jpg)  
Fig. 2: Pointwise relative error plots for closed-loop tracking for a 3D Pacman track with speed 12 [cm/s]

![](images/e1463a5f2022b44fd5d2ab872d4c7bb78b0f8da8dc39b0422b3bcd1bca698a85.jpg)  
(a)

![](images/50108d1646f708f9bf9758e259e5f3122da55600806241b25328707fa9e984a9.jpg)  
(b)  
Fig. 3: Pointwise relative error plots for (a) the small-sized spherical obstacle and (b) the medium-sized spherical obstacle avoidance task. The tolerance line accounts for the sphere’s location in real time as the target track is traversed.

## 3 Relative error plots for spherical obstacle avoidance

For completeness, we present here the pointwise relative error plots for the smallsized (see Fig. (3a)) and medium-sized (see Fig. (3b)) spherical obstacle avoidance tasks discussed in Section 1.5. Both plots show that the aSSM-based controller with constraints activated grazes past the spherical obstacles, whereas the one without constraints minimizes the tracking error to the target. The DMDc controller does not track the target closely and avoids the spherical obstacle by a large distance. The DMDc model lacks precision information to avoid the spheres delicately and bruteforces its way out of the region. The MZ-Koopman model fails to solve the optimization problem in real time. As a result, it stops abruptly at t = 3 [s] for the medium-sized spherical obstacle task and at t = 10 [s] for the small-sized spherical obstacle task.

## 4 Soft trunk avoiding a moving cylindrical obstacle along a complex path

In this section, we provide closed-loop results for the soft trunk avoiding a moving cylinder. This is a continuation of Section 1.5.2 for a complex centerline trajectory path of the cylindrical obstacle. The path starts similarly to the ones described in the Section 1.5.2, but after performing the rotation, it rises vertically in the z direction.

![](images/56d64a3b35b7fd614de5bc4a40de300b40a461964e0fd74becd308693be32e82.jpg)  
(a)

![](images/f24246502bc8faadf8d471a707549b1d91f090b281107249716a25faaa0e51a6.jpg)  
(b)  
Fig. 4: Snapshots of the trunk main body avoiding a moving cylindrical obstacle at $\mathrm { ( a ) } ~ t = 1 5 ~ [ \mathrm { s ] }$ as it approaches the rest state and (b) $t = 3 0 \ [ \mathrm { s } ]$ as the obstacle moves up in the vertical direction. The trunk body is green, and the cylindrical obstacle is yellow.

We expect the trunk body to react by avoiding the moving obstacle by conforming to a nonlinear bending shape. We precisely see these in the snapshot plots in Fig. 4a and Fig. 4b. The trunk body rises to a height of 0.25 [m] to avoid the moving cylindrical obstacle.

## 5 Derivation of the higher-order aSSM-reduced model

Here, we derive the aSSM-reduced model and Using the coordinate change

$$
\zeta = { \bf x } - { \bf x } _ { 0 } ( { \bf u } ) ,\tag{1}
$$

we monitor the error between the phase space coordinate and the critical manifold’s prediction for general control inputs eq. (17). Diferentiating eq. (1), we obtain

$$
\dot { \boldsymbol { \zeta } } ( t ) = \dot { \mathbf { x } } ( t ) - \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } ,\tag{2}
$$

$$
\begin{array} { r } { \dot { \boldsymbol \zeta } ( t ) = \mathbf { F } \left( \boldsymbol \zeta ( t ) + \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) , \mathbf { u } ( t ) \right) - \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } , } \end{array}\tag{3}
$$

Taylor expanding eq. (3) about ${ \bf x } _ { 0 } ( { \bf u } ( t ) )$ then gives,

$$
\begin{array} { r l } & { \dot { \zeta } ( t ) = \mathbf { F } \left( \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) , \mathbf { u } ( t ) \right) + \mathbf { D } _ { \zeta } \mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) , \mathbf { u } ( t ) ) \zeta + \dots } \\ & { \qquad + \mathbf { F } _ { 0 } ( \zeta , \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) , \mathbf { u } ( t ) ) - \mathbf { D } _ { u } \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } , \quad \mathbf { F } _ { 0 } = o ( | \zeta | ) . } \end{array}\tag{4}
$$

Using the critical manifold definition from eq. (13) of the main text, we find that the above equation simplifies to,

$$
\dot { \zeta } ( t ) = \mathbf { D } _ { \zeta } \mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) , \mathbf { u } ( t ) ) \zeta + \mathbf { F } _ { 0 } ( \zeta , \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) , \mathbf { u } ( t ) ) - \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } .\tag{5}
$$

We substitute the form in eq. (17) for the general control inputs ${ \bf \delta u } ( t )$ and expand eq. (5) about the trajectory $( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { \mathbf { s } } ( \epsilon t ) ) , \mathbf { u } ^ { \mathbf { s } } ( \epsilon t ) )$ in the phase space. This results in the following set of equations (with higher-order terms written out):

$$
\dot { \zeta } ( t ) = \mathbf { D } _ { \zeta } \mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , \mathbf { u } ^ { s } ( \epsilon t ) ) \zeta + \mathbf { F } _ { 0 } ( \zeta , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , \mathbf { u } ^ { s } ( \epsilon t ) )\tag{6}
$$

$$
- { \bf D _ { u } } { \bf x } _ { 0 } ( { \bf u } ^ { s } ( \epsilon t ) ) \dot { \bf u } + O \left( ( \epsilon + \delta ) ^ { 2 } , \epsilon | \boldsymbol { \zeta } | , \delta | \boldsymbol { \zeta } | \right) .\tag{7}
$$

We further make the assumed time-scale separation explicit in eq. (6) by introducing a relatively slowly varying parameter α,

$$
\begin{array} { r l } & { \dot { \zeta } ( t ) = \mathbf { D } _ { \zeta } \mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) , \mathbf { u } ^ { s } ( \alpha ) ) \boldsymbol { \zeta } + \mathbf { F } _ { 0 } ( \boldsymbol { \zeta } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) , \mathbf { u } ^ { s } ( \alpha ) ) } \\ & { \qquad - \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) \dot { \mathbf { u } } + O \left( ( \epsilon + \delta ) ^ { 2 } , \epsilon | \boldsymbol { \zeta } | , \delta | \boldsymbol { \zeta } | \right) , } \end{array}\tag{8}
$$

In the frozen-time limit, $\epsilon = 0$ and $\dot { \bf u } = 0 .$ the decomposition in eq. (16) of the main paper applies, which allows us to represent the family of the d slowest right eigenvectors of the linear operator $\mathbf { D } _ { \zeta } \mathbf { F } ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) , \mathbf { u } ^ { s } ( \alpha ) )$ as columns of the matrix $\mathbf { V } ( \mathbf { u } ^ { s } ( \alpha ) ) \in \mathbb { R } ^ { n \times d }$ . This enables us to define a local coordinate along the slow spectral subbundle ${ \bf E } ( { \bf u } ^ { s } ( \alpha ) ,$ ),

$$
\mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ^ { s } ( \alpha ) ) \boldsymbol { \zeta } , \quad \mathbf { r } \in \mathbb { R } ^ { d } .\tag{9}
$$

Using this, we can define the aSSM fiber parametrization,

$$
\zeta ( t ) = \mathbf { W } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ) + \tilde { \mathbf { W } } ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , t ) + O \left( ( \epsilon + \delta ) ^ { 2 } , \epsilon | \mathbf { r } | , \delta | \mathbf { r } | \right) ,\tag{10}
$$

and the aSSM reduced dynamics,

$$
\dot { \mathbf { r } } = \mathbf { R } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ) + \tilde { \mathbf { R } } ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , t ) + O \left( ( \epsilon + \delta ) ^ { 2 } , \epsilon | \mathbf { r } | , \delta | \mathbf { r } | \right) .\tag{11}
$$

where we seek them as Taylor expansions in $\epsilon , \delta .$ , and r, and the terms $\tilde { \mathbf { W } }$ and R<sup>˜</sup> represent the terms at $O ( \epsilon + \delta )$

The aSSM parameterization and the reduced dynamics satisfy the following invariance equations,

$$
\begin{array} { r l } & { \mathbf { D } _ { \zeta } \mathbf { F } \big ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) , \mathbf { u } ^ { s } ( \alpha ) \big ) \mathbf { W } \big ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \big ) } \\ & { \quad + \mathbf { F } _ { 0 } \big ( \mathbf { W } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ) , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) , \mathbf { u } ^ { s } ( \alpha ) \big ) } \\ & { = \mathbf { D } _ { \mathbf { r } } \mathbf { W } \big ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \big ) \mathbf { R } \big ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \big ) , } \end{array}\tag{12}
$$

at leading order. At higher order, we obtain

$$
\begin{array} { r l } & { \mathbf { D } _ { \zeta } \mathbf { F } \big ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) , \mathbf { u } ^ { s } ( \alpha ) \big ) \tilde { \mathbf { W } } \big ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , t \big ) - \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \alpha ) ) \dot { \mathbf { u } } } \\ & { \mathbf { \Psi } = \mathbf { D } _ { \mathbf { r } } \mathbf { W } \big ( \mathbf { 0 } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \big ) \tilde { \mathbf { R } } \big ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , t \big ) + \dot { \tilde { \mathbf { W } } } \big ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , t \big ) . } \end{array}\tag{13}
$$

Eq. (12) is solved by seeking Taylor expansions for the aSSM parameterization and the aSSM-reduced dynamics in the reduced coordinates. Haller and Kaundinya [4] derive recursive algebraic solutions for these Taylor coeficients with initial condition $\mathbf { D _ { r } W } ( \mathbf { 0 } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ) = \mathbf { V } ( \mathbf { u } ^ { s } ( \epsilon t ) )$ ). We adopt the same choices to derive the higherorder terms.

Note that we still need to choose either of the unknowns in eq. (13) to solve it. We choose that the higher-order reduced dynamics has the form $\tilde { \mathbf { R } } ( \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) , t ) =$ $- \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \mathbf { D } \mathbf { u } \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \dot { \mathbf { u } }$ , this is consistent with existing choices (see Breunung and Haller [1], Jain and Haller [5] and) where the higher-reduced dynamics term appears as the projection of the purely time-dependent term in eq. (6) onto $\mathbf { E } ( \mathbf { u } ^ { s } ( \epsilon t ) )$

We further assume that control term $\mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \dot { \mathbf { u } }$ is nearly tangential to the aSSM. We have reason to expect this to be true, as the aSSM captures the most dominant slowest directions near the critical manifold, and the term of interest is a measure of the deviation from the critical manifold, which should lie close to the aSSM. Hence, we can safely ignore the contribution of the higher-order term to the aSSM parameterization. Accounting for these approximations, the d-dimensional aSSM-reduced model takes the final form,

$$
\begin{array} { r l } & { \mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ( \mathbf { x } - \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) ) , } \\ & { \dot { \mathbf { r } } = \mathbf { R } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ) - \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) \dot { \mathbf { u } } + O \left( ( \epsilon + \delta ) ^ { 2 } , \epsilon | \mathbf { r } | , \delta | \mathbf { r } | \right) , } \\ & { \mathbf { x } = \mathbf { W } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) ) + \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) ) + O \left( ( \epsilon + \delta ) , \epsilon | \mathbf { r } | , \delta | \mathbf { r } | \right) . } \end{array}\tag{14}
$$

Remark 1. [aSSM-reduced model without control splitting] In trajectory tracking applications, the slowly varying control input $\mathbf { u } ^ { s } ( \epsilon t )$ is typically dominant in magnitude over the faster deviation control forces ${ \mathbf { u } } ^ { d } ( t )$ at all times. Under this assumption, the critical manifold $\mathcal { L } _ { 0 }$ and the chart mappings to the aSSM, evaluated along $\mathbf { u } ^ { s } ( \epsilon t )$ , remain consistent with those evaluated on the general control input $\mathbf { u } ( t ) = \mathbf { u } ^ { s } ( \epsilon t ) + \mathbf { u } ^ { d } ( t )$ . This justifies expressing the aSSM-reduced model (see eq. (14)) with only the general control input u(t):

$$
\begin{array} { r l } & { \mathbf { r } = \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ( t ) ) ( \mathbf { x } - \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) ) , } \\ & { \dot { \mathbf { r } } = \mathbf { R } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) ) - \mathbf { V } ^ { \mathrm { T } } ( \mathbf { u } ( t ) ) \mathbf { D } _ { \mathbf { u } } \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) \dot { \mathbf { u } } , } \\ & { \mathbf { x } = \mathbf { W } ( \mathbf { r } , \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) ) + \mathbf { x } _ { 0 } ( \mathbf { u } ( t ) ) . } \end{array}\tag{15}
$$

Remark 2. [Related aSSM-reduced model derivations] Haller and Kaundinya [4] provides proofs for the existence of aSSMs and derives the aSSM-reduced model for the same setting as in eq. (12) with $n _ { u } = 1$ and $\mathbf { u } ( t ) \equiv \mathbf { u } ^ { s } ( \epsilon t )$ . Kaundinya et al. [6] presented this derivation for general control inputs assuming the survival of the aSSM defined by Haller and Kaundinya $[ 4 ]$ under additional fast uniformly bounded deviation control inputs.

Both their derivations difer in the choice of coordinate change ζ in eq. (1). Translating the choice of Haller and Kaundinya [4] to our setting results in $\zeta ( t ) =$ $\mathbf { x } ( t ) - \mathbf { x } _ { \epsilon } ( \mathbf { u } ^ { s } ( \epsilon t ) )$ , which measures the distance of the robot’s response from the slow manifold $\mathcal { L } _ { \epsilon }$ . From the data, one does not readily have access to the slow manifold $\mathcal { L } _ { \epsilon }$ but will have access to the critical manifold $\mathcal { L } _ { 0 }$ . This observation prompted Kaundinya et al. [6] to choose $\zeta ( t ) = \mathbf { x } ( t ) - \mathbf { x } _ { 0 } ( \mathbf { u } ^ { s } ( \epsilon t ) )$ as the coordinate for their derivation. Our choice of derivation, eq. (1), takes this a step further by evaluating the critical manifold for general control inputs, as would be done in practice.

# 6 Details for the linear and RL baselines used in our comparisons

## 6.1 DMDc

We identify a linear time-invariant model directly in the delay-embedded observable space $\mathbf { y } ,$ using the same 2 hours of static $\{ \mathbf { Y } _ { \mathrm { s t e p } } ^ { i } , \mathbf { U } _ { \mathrm { s t e p } } ^ { i } \}$ and random smooth data $\{ \mathbf { Y } _ { \mathrm { c a l i b } } , \mathbf { U } _ { \mathrm { c a l i b } } \}$ collected to learn the aSSM-reduced model. We denote these as snapshot matrices Y, U and their one-step shifted snapshot matrix as $\overline { { \mathbf { Y } } }$ , we solve

$$
\left\{ { \bf A } ^ { \star } , { \bf B } ^ { \star } \right\} = \underset { { \bf A } , { \bf B } } { \arg \operatorname* { m i n } } \left\| \overline { { \mathbf { Y } } } - { \bf A } { \bf Y } - { \bf B } { \bf U } \right\| ^ { 2 }\tag{16}
$$

via least-squares regression (see Proctor et al. [7]), and obtain the linear time-invariant model

$$
\mathbf { y } _ { k + 1 } = \mathbf { A } ^ { * } \mathbf { y } _ { k } + \mathbf { B } ^ { * } \mathbf { u } _ { k } .\tag{17}
$$

We then plug eq. (17) as the dynamics constraint in the MPC formulation eq. (37) to obtain a DMDc-based controller. Specifically, we tuned the controller to obtain decent closed-loop performance with cost matrices given by $\mathbf { R _ { u } } = 0 . 0 0 1 \cdot \mathbb { I } _ { 6 } , \mathbf { R _ { \dot { \mathbf { u } } } } = 0 . 0 0 1 \cdot \mathbb { I } _ { 6 }$ and $\mathbf { Q _ { z } } = \mathrm { d i a g } ( 1 0 , 1 0 , 1 0 )$ . We used horizon lengths $M = 1 0$ for target-tracking tasks and $M = 2 0$ for obstacle-avoidance tasks. The model solve times were very fast, hence we could set the controller frequency to 100 Hz.

## 6.2 Koopman-pregain

We fit a Koopman-pregain model (see Haggerty et al. [3]) to the workspace coordinates z. We collect only 30 minutes of random step-response data $\{ \mathbf { Z } , \mathbf { U } \}$ for this baseline. We then delay-embed this once to obtain a delayed embedded dataset Q. We first fit a dynamic Koopman operator to $\mathbf { Q }$

$$
\left\{ { \bf A } ^ { \star } , { \bf B } ^ { \star } \right\} = \underset { { \bf A } , { \bf B } } { \arg \operatorname* { m i n } } \left\| \overline { { \bf Q } } - { \bf A } { \bf Q } - { \bf B } { \bf U } \right\| ^ { 2 } .\tag{18}
$$

This dynamic part alone underperformed, so Haggerty et al. [3] added a static correction using the static equilibria $\{ \mathbf { Z } _ { \mathrm { s t a t i c } } ^ { i } , \mathbf { U } _ { \mathrm { s t a t i c } } ^ { i } \}$ obatiend from the sampled dataset. We fit a linear map G from steady-state workspace positions to the static input that produced them,

$$
\mathbf { G } ^ { \star } = \underset { \mathbf { G } } { \arg \operatorname* { m i n } } \sum _ { i } \left\| \mathbf { U } _ { \mathrm { s t a t i c } } ^ { i } - \mathbf { G } \mathbf { Z } _ { \mathrm { s t a t i c } } ^ { i } \right\| ^ { 2 } .\tag{19}
$$

They reformulate the optimization problem in eq. (37) as a linear quadratic regulator (LQR) problem subject to a linear time-invariant dynamical system. They then approximate the finite-horizon problem by treating it as an infinite-horizon one, and obtain the optimal control inputs needed to track the target trajectory $\mathbf { { \cal T } } ( t )$ by solving the algebraic Riccati equation for the pregain matrix K . This yields the control law

$$
\mathbf { u } ^ { * } ( t ) = - \mathbf { K } ( \mathbf { z } - \mathbf { r } ( t ) ) .\tag{20}
$$

Because this version produced inaccurate tracking, they introduced a static feedforward correction term, resulting in

$$
\mathbf { u } ^ { * } ( t ) = - \mathbf { K } ( \mathbf { z } - \mathbf { r } ( t ) ) + \mathbf { G } \mathbf { r } ( t ) .\tag{21}
$$

Haggerty et al. [3] observed accurate tracking with the added static term, and their results showed that the dynamic component contributed very little. This approach can reasonably be expected to work only when the target trajectory varies slowly, and the robot’s dynamics are purely linear. In practice, they rely on these assumptions by first driving the robot along slowly varying circular paths, then gradually increasing speed once it has latched onto the target. This method would fail if the Koopman-based LQR controller were required to track a rapidly accelerating target from the beginning.

## 6.3 Mori-Zwanzig (MZ) Koopman approach

The MZ-Koopman baseline by Catalasan and He [2] is trained on the same 2 hours of static and random control data used for learning the aSSMs and DMDc models. Closely following the procedure by Catalasan and He [2], we lift the observable state y using Gaussian radial basis functions (RBF) and include a one-step memory term. This results in the following MZ-Koopman dynamic map,

$$
\mathbf { q } _ { k + 1 } = \mathbf { A } _ { 0 } \mathbf { q } _ { k } + \mathbf { A } _ { 1 } \mathbf { q } _ { k - 1 } + \mathbf { B } _ { 0 } \mathbf { u } _ { k } + \mathbf { B } _ { 1 } \mathbf { u } _ { k - 1 } , \qquad \mathbf { q } = \phi ( \mathbf { y } ) .\tag{22}
$$

The RBF map $\phi$ is built from $N _ { c } = 3 3$ unique centers $\{ \mathbf { c } _ { j } \} _ { j = 1 } ^ { 3 3 }$ spanning the observable space of trunk uniformly, and chosen from the training data. The map ϕ is given by

$$
\phi _ { j } ( \mathbf { y } ) = \mathrm { e x p } \left( - \frac { \| \mathbf { y } - \mathbf { c } _ { j } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \qquad j = 1 , \dots , N _ { c } .\tag{23}
$$

We used the same choice for the number of centers as in Catalasan and He [2], as a large number of centers led to larger linear dynamic models that slow down MPC. We also observed that the model’s performance did not quantitatively improve with 33 centers vs. 50 centers. The standard deviation σ is kept fixed as in Catalasan and He [2]. The authors do not provide a systematic method to compute this. Hence, we set the parameter $\sigma$ by sweeping over values and selecting the one that yielded the lowest prediction error on an unseen test dataset. Similar to DMDc, the set $\{ { \bf A } _ { 0 } , { \bf A } _ { 1 } , { \bf B } _ { 0 } , { \bf B } _ { 1 } \}$ is solved by least-squares regression on the lifted snapshots Q and control inputs U.

For MPC deployment, we absorbed the memory term into an augmented state ${ \pmb \xi } _ { k } = \{ { \bf q } _ { k } , { \bf q } _ { k - 1 } , { \bf u } _ { k - 1 } \}$ , and recovered a linear time invariant system that can readily be used with standard MPC schemes. We found that tuning $\mathbf { R _ { u } } = 0 . 0 0 0 1 \cdot \mathbb { I _ { 6 } } , \mathbf { R _ { \dot { \mathbf { u } } } } =$ $0 . 0 0 1 \cdot \mathbb { I } _ { 6 } .$ and $\mathbf { Q _ { z } } = \mathrm { d i a g } ( 1 0 , 1 0 , 1 0 )$ led to reasonable closed-loop performance. We used horizon lengths M = 10 for target tracking tasks, and for obstacle avoidance tasks, sometimes $M = 2 0$ . The model solve times were very slow, hence we could only set the controller frequency to 20 Hz to achieve good tracking performance.

## 6.4 Reinforcement Learning

We train a model-free RL baseline using proximal policy optimization (PPO) as discussed by Schulman et al. [8]. We adopt the neural network architecture from their work and train it to update its parameters directly on hardware. The policy is trained to minimize the same cost structure as in the aSSM-reduced MPC objective, combining target-tracking error with a control-rate penalty. Part of this cost structure is the reward,

$$
\operatorname { R e w a r d } ( t ) = - \Big ( \| \mathbf { z } ( t ) - \mathbf { \boldsymbol { \Gamma } } ( t ) \| ^ { 2 } \Big ) ,\tag{24}
$$

used as the negative reward at each control step. In addition to penalizing the control rate, we ensured that the guessed control inputs were generated continuously to prevent abrupt movements that could cause the trunk robot to break. The target trajectory $\Gamma ( t )$ is chosen in the trunk’s workspace from randomly sampling smooth 10- second-long trajectories from the collected smooth random training data $\mathbf { Y } _ { \mathrm { c a l i b } }$ and $\mathbf { U } _ { \mathrm { c a l i b } }$ for the other models. We did this to avoid forcing policy to conform to physical targets that might be unfeasible to reach physically.

![](images/5c056d06b273dbc0e1cb3d3d274b97b9132cc76c9d3abe269200de97c493f3a1.jpg)  
Fig. 5: Training reward curve for proximal policy optimization (PPO) on hardware. The solid gray line indicates the allowed uncertainty of 17.5, computed by summing the maximum allowed positional uncertainty (3 cm tube radius, measured on hardware) across the 10-second episode.

In Fig. (5), we show the resulting training reward curve. The solid gray line marks the maximum allowed uncertainty of 17.5 in the reward. We obtain it by summing the maximum allowed positional uncertainty due to the motors (the 3 cm uncertainty tube radius measured in hardware tests) over the 10-second episode. We observe that the RL policy converges below this uncertainty threshold after 2 hours of training, and further reduction in jittery behavior was observed after training for 2 hours more. We selected the best-performing converged policy during this period for closed-loop deployment. The execution of the RL policy was extremely fast, and we operated it at 100 Hz.

## Supplementary References

[1] T. Breunung and G. Haller. Explicit backbone curves from spectral submanifolds of forced-damped nonlinear mechanical systems. Proc. R. Soc. A, 474:20180083, 2018.

[2] L. Catalasan and T. He. Memory-dependent dynamic modeling of cable-driven soft robots using the mori-zwanzig koopman operator. Nonlinear Dyn., 114(2):77, 2026.

[3] D. A. Haggerty, M. J. Banks, E. Kamenar, A. B. Cao, P. C. Curtis, I. Mezi´c, and E. W. Hawkes. Control of soft robots with inertial dynamics. Sci. Robot., 8(81): eadd6864, 2023.

[4] G. Haller and R. S. Kaundinya. Nonlinear model reduction to temporally aperiodic spectral submanifolds. Chaos, 34(4):043152, Apr. 2024. ISSN 1054-1500. doi: 10.1063/5.0187080. URL https://doi.org/10.1063/5.0187080.

[5] S. Jain and G. Haller. How to compute invariant manifolds and their reduced dynamics in high-dimensional finite-element models? Nonlinear Dyn., 107:1417– 1450, 2022.

[6] R. S. Kaundinya, J. I. Alora, J. G. Matt, L. A. Pabon, M. Pavone, and G. Haller. Data-driven soft robot control via adiabatic spectral submanifolds. Int. J. Robot. Res., 2026. doi: 10.1177/02783649261461630.

[7] J. L. Proctor, S. L. Brunton, and J. N. Kutz. Dynamic mode decomposition with control. SIAM J. Appl. Dyn. Syst., 15(1):142–161, 2016.

[8] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347. arXiv:1707.06347.

[9] Y. L. Yap, S. L. Sing, and W. Y. Yeong. A review of 3d printing processes and materials for soft robotics. Rapid Prototyp. J., 26(8):1345–1361, 2020.