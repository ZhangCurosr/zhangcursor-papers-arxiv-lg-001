# A Decade of Learning-Based Control: From Theoretic Foundations to Practical Deployment

Angela P. Schoellig<sup>1</sup>, Matthias A. Muller¨ <sup>2</sup>, Sebastian Trimpe<sup>3</sup>, Melanie N. Zeilinger<sup>4</sup>, Mahathi Anand<sup>1</sup>, Andrea Carron<sup>4</sup>, Oliver Hausdorfer¨ <sup>1</sup>, Victor G. Lopez<sup>2</sup>, Friedrich Solowjow<sup>3</sup>, Haocheng Zhao<sup>1</sup>, and Siqi Zhou<sup>5</sup>

Abstract— Motivated by ten years of Learning-Based Control Invited Sessions at the IEEE Conference on Decision and Control (CDC), this tutorial paper reviews key advances at the intersection of machine learning and control. We focus on methods that combine the flexibility of machine learning techniques with the structure and theoretical foundations of control to design autonomous systems that are capable, reliable, and safe. We first introduce core notions for establishing stability and safety guarantees, then review approaches that learn dynamics models to improve control under uncertainty, and finally discuss data-driven methods that directly synthesize or learn policies, including reinforcement learning, imitation learning, and Bayesian optimization. We further discuss representative applications and open challenges. The goal is to provide an accessible entry point to the field and to support continued progress toward reliable learning-based control in real-world systems. To support learning and practical exploration, we share implementations and illustrative examples along with our discussion, together with additional resources on our website: https://learning-based-control.github.io/tutorial/.

## I. INTRODUCTION

Over the past decade, the integration of machine learning and control theory has evolved from a growing research interest into a practical approach for many real-world systems. Machine learning provides powerful tools for extracting patterns from data, while control theory offers the mathematical foundation needed to use such data-driven methods in closedloop systems. Principled combinations of these perspectives can extend autonomous systems to more challenging operating domains, broader system classes, and more complex tasks, while preserving the closed-loop properties required for reliable deployment.

Why machine learning? With the widespread availability of data and accelerated computing, machine learning methods have achieved remarkable success across many domains, especially in computer vision and natural language processing. From early milestones in image recognition [1], [2] to recent large-scale language models [3]–[5], these methods have demonstrated a strong capability to capture complex input-output relationships directly from unstructured data. A similar shift is taking place in control through deep reinforcement learning (RL). Enabled by high-quality synthetic data and massively parallelized compute power [6]–[8], deep RL has expanded our ability to address challenging control problems involving high-dimensional, non-smooth, and complex dynamical systems [9], [10]. More broadly, leveraging data, whether synthetic or real, offers new opportunities to expand what autonomous systems can achieve.

![](images/2e3750294c970a03a8b868ded7e016720c8f14a4efab72d301bbbe6e38f33be9.jpg)  
Fig. 1: Scope of the tutorial paper. We review learning-based control methods that combine machine learning with control-theoretic structure to improve system capabilities while preserving desired closed-loop properties. The paper first introduces foundations for stability and safety guarantees (green), then covers methodological paradigms ranging from structured model-based learning to increasingly data-driven policy learning (orange), and finally discusses representative application domains (grey).

Why control? The flexibility that makes machine learning powerful also introduces challenges when learning-based methods are deployed in physical systems. Purely datadriven models can be highly expressive, but often struggle to satisfy strict safety and reliability requirements, and may require large amounts of data that are costly or infeasible to collect on real systems. Control provides structural priors and theoretical tools for bridging this gap. By embedding control principles into the learning design, one can guide the learning process toward methods that are more sample efficient, highperforming, and safe for practical applications [11]–[14]. The use of data to model the unknown has long been central to control, for instance, in system identification and adaptive control. However, the scale, expressiveness, and opacity of modern learning methods create new challenges in sample efficiency, verification, and reliable deployment. Addressing these challenges requires a tight and principled coupling of machine learning and control-theoretic approaches.

Learning-Related Keywords in Title of Control Conferences  
![](images/4002bda0bfcd2827e52683a4ca713de99048bf507f6917d746421ada0344b256.jpg)  
Fig. 2: Evolution of recurring research threads in the CDC Learning-Based Control Invited Sessions from 2016 through 2025. The heatmap shows the annual frequency of representative concept phrases extracted from paper titles and abstracts. Details of the keyword extraction and analysis procedure are provided in the appendix.

Where do these fields converge? To foster principled approaches that integrate machine learning and control theory beyond the standard tools of either discipline alone, the Invited Session Series on Learning-Based Control has been held at the IEEE Conference on Decision and Control (CDC) since $2 0 1 6 ^ { 1 }$ . Over the past decade, this series has hosted 30 dedicated sessions featuring 182 accepted papers, drawing contributions from over 100 distinct institutions. As reflected in Figure 2, the series captures the increasing incorporation of expressive learning paradigms into control, while maintaining a strong emphasis on principled methods with desired closed-loop properties. Recurring themes include (deep) neural networks, (deep) RL, and Gaussian processes (GPs), combined with control tools such as model predictive control (MPC), Hamilton-Jacobi (HJ) reachability, control Lyapunov functions (CLFs), and control barrier functions (CBFs) to address performance, stability, and safety. The steady growth in citations over the past decade (Figure 3) further reflects the sustained interest in these topics and the expanding body of work contributed by the community.

This ten-year milestone of the CDC Learning-Based Control Invited Sessions does not exist in isolation. The broader effort to bridge control and learning for safe and scalable autonomy is also reflected in dedicated conferences such as the Learning for Dynamics and Control Conference, as well as in the increasing presence of learning-based methods in core control conferences, as shown in Figure 4.

In this tutorial paper, we highlight key developments and emerging trends in learning-based control over the past decade, spanning control-theoretic foundations, dynamics learning, direct data-driven control, and learning-oriented policy optimization (see Figure 1). Our intention is to draw connections among these different families of methods by highlighting how they address uncertainty, which theoretical tools are compatible with them, and what types of guarantees can be established (see Table I). Rather than providing an exhaustive survey, we focus on representative methodological directions motivated by the CDC Learning-Based Control In vited Sessions. With the discussions and examples along the way, we hope this tutorial paper serves to contextualize the progress of the past decade and illustrate how learning and control can be combined to enable more capable, reliable, and safe autonomous systems.

![](images/3f82ecabc59ef1bf34e5b848e348b3445dd40457de06371cd2f369e4e589cb7b.jpg)  
Fig. 3: Citation statistics for papers presented in the CDC Learning-Based Control Invited Sessions. The bars show the total number of citations received per year, while the line indicates cumulative citations based on Google Scholar data collected in May 2026.

![](images/445120833a8f5e64679bf31a72d48b56f0416cf0f77d7a5ecdb054a30e2b608e.jpg)  
Fig. 4: The percentage of papers with the term “learning” in their titles across three control conferences (CDC, ACC, and ECC) from 2010 to 2025. Pertinent data was obtained from the DBLP database in May 2026.

The paper is organized as follows. We begin by formulating the general learning-based control problem (Section II). We then review control-theoretic foundations for certifying desired system properties and guarantees (Section III), followed by learning-based control approaches that learn dynamics and improve performance under uncertainty (Section IV). We next discuss direct data-driven control and learning-oriented methodologies, including RL, imitation learning (IL), and Bayesian optimization (BO), that synthesize or optimize policies from data while preserving desired closed-loop properties (Section V and Section VI). We conclude by highlighting representative applications at the intersection of learning and control (Section VII), followed by takeaways and future research directions (Section VIII).

The methods and perspectives discussed in this tutorial are drawn from diverse subfields across the machine learning and control communities. For a deeper dive into specific subareas, we refer readers to dedicated surveys, including learningbased MPC [12], model-based and safe RL [16], [17], direct data-driven control [18], BO for controller tuning and robot learning [19], HJ reachability [13], CBF certification [14], safety filters [20], [21], and safe learning in robotics [11].

TABLE I: An overview of the learning-based control method families reviewed in Section IV-Section VI. For each method family, we detail the specific learning objective, requisite prior knowledge, and the data-driven learning mechanism. Additionally, we summarize how each method quantifies estimation errors, and how those uncertainty models pair with compatible controller synthesis techniques to yield closed-loop guarantees discussed in Section III. Note that the control methods and closed-loop guarantees presented in the table serve as representative examples, and are as such non-exhaustive.
<table><tr><td rowspan=1 colspan=1>Method Family</td><td rowspan=1 colspan=1>What is Learned?</td><td rowspan=1 colspan=1>Prior KnowledgeRequired</td><td rowspan=1 colspan=1>Learning from Data</td><td rowspan=1 colspan=1>Estimation Errors</td><td rowspan=1 colspan=1>Compatible ControllerSynthesis Methods</td><td rowspan=1 colspan=1>Resulting Closed-LoopGuarantees</td></tr><tr><td rowspan=1 colspan=1>Dynamics learning for contr</td><td rowspan=1 colspan=1>ol (Section IV). All methods ca</td><td rowspan=1 colspan=1>n additionally include a kn</td><td rowspan=1 colspan=1>own nominal model.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>Linear regression(Section IV-B)Neural networks(Section IV-C)Bayesian linearregression (Section IV-D,Example 2)</td><td rowspan=1 colspan=1>point estimates ofcoefficients in a linear fixed-basis nonlineardynamics model</td><td rowspan=1 colspan=1>(locally) linear structure/ (nonlinear) basisfunctions</td><td rowspan=1 colspan=1>closed-form least-squaressolution based on(offline) input-state data</td><td rowspan=1 colspan=1>norm bounds orellipsoidal uncertaintysets for modelcoefficients</td><td rowspan=1 colspan=1>certainty-equivalent androbust LQR; MPC</td><td rowspan=1 colspan=1>robust stability; robustconstraint satisfaction;probabilistic constraintsatisfaction</td></tr><tr><td rowspan=2 colspan=1>point estimates ofparameters in a nonlinearneural-network dynamicsmodelposterior distribution overcoefficients in a fixed-basisnonlinear dynamics model</td><td rowspan=1 colspan=1>network architecture andactivation functions;optional physical ordynamical structure (e.g.,Lipschitz regularities)</td><td rowspan=1 colspan=1>gradient-based trainingbased on (offline)input-state data</td><td rowspan=1 colspan=1>deterministic error oroutput bounds based onregularities ; possibly,predictive uncertaintyestimates viaapproximate (Bayesian)inference</td><td rowspan=1 colspan=1>nonlinear MPC ; Lyapunovand small-gain design</td><td rowspan=1 colspan=1>stability via Lipschitz or localsector bounds; ISS/δ-ISS ofrecurrent models</td></tr><tr><td rowspan=1 colspan=1>(nonlinear) basisfunctions, priorparameter distribution,and noise model</td><td rowspan=1 colspan=1>closed-form posteriorinference based on(offline) input-state dataunder Gaussian prior andi.i.d. noise assumptions</td><td rowspan=1 colspan=1>probabilistic predictiveuncertainty boundsderived from theposterior parameterdistribution</td><td rowspan=1 colspan=1>chance-constrained MPC withconstraint tightening; robustMPC with conservative errorbounds</td><td rowspan=1 colspan=1>probabilistic constraintsatisfaction or worst-casebounds; overconfident if thebasis is misspecified</td></tr><tr><td rowspan=1 colspan=1>Gaussian processes(Section IV-D,Example 2, Example 3)</td><td rowspan=1 colspan=1>posterior distribution overthe dynamics functionrepresented by anonparametric model</td><td rowspan=1 colspan=1>prior mean function,covariance kernel [15],and noise model</td><td rowspan=1 colspan=1>closed-form posteriorinference based on(offline) input-state dataunder Gaussian prior andi.i.d. noise assumptions</td><td rowspan=1 colspan=1>probabilistic predictiveuncertainty boundsderived from theposterior distributionover the dynamicsfunction</td><td rowspan=1 colspan=1>chance-constrained MPC;robust MPC using confidenceregions; CLF-based synthesis</td><td rowspan=1 colspan=1>probabilistic constraintsatisfaction; stability via robustinvariant terminal sets andworst-case safety fromconfidence regions</td></tr><tr><td rowspan=1 colspan=1>Direct data-driven control (</td><td rowspan=1 colspan=1>Section V)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Data-based LMI design</td><td rowspan=1 colspan=1>feedback gain computedfrom input/state orinput/output data</td><td rowspan=1 colspan=1>LTI system; set of basisfunctions in thenonlinear case</td><td rowspan=1 colspan=1>rank condition on thecollected offline data</td><td rowspan=1 colspan=1>noise bounds can beused for robust controldesign</td><td rowspan=1 colspan=1>state- or output-feedbackcontrollers (including LQR,robust control, etc.)</td><td rowspan=1 colspan=1>(robust) stability; computableregions of attraction</td></tr><tr><td rowspan=1 colspan=1>Data-driven predictivecontrol (Example 4)</td><td rowspan=1 colspan=1>data-based systemrepresentation (Willems&#x27;lemma)</td><td rowspan=1 colspan=1>sufficiently longhorizon L; suitableregularization tuning</td><td rowspan=1 colspan=1>rank condition on data;potentially online updateof data</td><td rowspan=1 colspan=1>prediction error boundsdepending on noisemagnitude</td><td rowspan=1 colspan=1>linear (and nonlinear) MPC</td><td rowspan=1 colspan=1>closed-loop stability androbustness; constraintsatisfaction</td></tr><tr><td rowspan=1 colspan=1>Direct policy learning and co</td><td rowspan=1 colspan=1>ntroller tuning (Section VI)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Model-free RL(Example 5)Model-based RL</td><td rowspan=1 colspan=1>policy πθ and/oraction-value function Q</td><td rowspan=1 colspan=1>generally little: rewardfunction and state/actiondefinition, no knowledgeof the dynamics</td><td rowspan=1 colspan=1>interaction with asimulator or the system(data hungry); offline RLfrom a fixed dataset</td><td rowspan=1 colspan=1>not modelled explicitly;stochastic policies</td><td rowspan=1 colspan=1>policy-gradient andactor-critic updates</td><td rowspan=1 colspan=1>none intrinsic</td></tr><tr><td rowspan=1 colspan=1>dynamics model pθM,policy, and/or action-valuefunction Q</td><td rowspan=1 colspan=1>same as model-free RL;optional structure in themodel</td><td rowspan=1 colspan=1>additional interactionwith the learned model;typically, fewer samplesthan model-free RL</td><td rowspan=1 colspan=1>uncertainty quantificationof learned dynamicsmodel, e.g., viaprobabilistic ensembles</td><td rowspan=1 colspan=1>Dyna-style rollouts, ordecision-time planning akin toMPC</td><td rowspan=1 colspan=1>safety via Lyapunov functions,barrier functions, safety filters,or CMDP soft constraints</td></tr><tr><td rowspan=1 colspan=1>Imitation learning andapproximate MPC</td><td rowspan=1 colspan=1>policy πθ fromdemonstrations (behaviourcloning) or a reward(inverse RL)</td><td rowspan=1 colspan=1>an expert: humandemonstrations or anMPC to be approximated</td><td rowspan=1 colspan=1>dataset of state-actionpairs, generated fromexpert; supervisedlearning</td><td rowspan=1 colspan=1>approximation error ofthe learned policy</td><td rowspan=1 colspan=1>supervised fitting of the MPClaw; constrained generativepolicies</td><td rowspan=1 colspan=1>constraint satisfaction despiteapproximation error viarobustness margins orprojection</td></tr><tr><td rowspan=1 colspan=1>BO controller tuning(Example 6)</td><td rowspan=1 colspan=1>low-dimensional controllerparameters θ through asurrogate of J(θ)</td><td rowspan=1 colspan=1>controller structure (e.g.,PID, LQR, MPC); GPprior on J</td><td rowspan=1 colspan=1>few closed-loopevaluations, directly onhardware</td><td rowspan=1 colspan=1>GP posterior over theperformance J</td><td rowspan=1 colspan=1>applicable to tuning ofessentially any controller,acquisition-functionmaximization</td><td rowspan=1 colspan=1>properties inherited from thecontroller structure; safe tuningunder performance and otherconstraints</td></tr></table>

We hope that this tutorial helps contextualize the progress of the past decade and provides a foundation for addressing the challenges that arise as learning-based control continues to scale toward increasingly complex real-world systems.

## II. PROBLEM FORMULATION

Learning-based control seeks to address general nonlinear control design problems; different families of methods may differ in the specifics of the control problem they address. In the following, we provide a general problem formulation, specializing it in subsequent sections where necessary.

We consider a general, nonlinear dynamical system

$$
\begin{array} { c } { { x _ { k + 1 } = f ( x _ { k } , u _ { k } , w _ { k } ) , } } \\ { { y _ { k } = h ( x _ { k } , v _ { k } ) , } } \end{array}\tag{1a}
$$

(1b)

where $k \in \mathbb { Z } _ { \geq 0 }$ is the discrete-time index, $x _ { k } \ \in \ \mathbb { R } ^ { n _ { \mathrm { x } } }$ is the system state, $u _ { k } \in \mathbb { R } ^ { n _ { \mathrm { u } } }$ is the control input, $y _ { k } \in \mathbb R ^ { n _ { \mathrm { y } } }$ is the measured output, $w _ { k } ~ \in ~ \mathbb { R } ^ { n _ { \mathrm { w } } }$ is process noise or disturbance, and $\begin{array} { r c l } { v _ { k } } & { \in } & { \mathbb { R } ^ { n _ { \mathrm { v } } } } \end{array}$ is measurement noise. The dynamics function $f$ describes the evolution of the state over time, while h describes how sensor measurements are obtained. Depending on the setting, different specializations of (1) are considered. Common configurations include state feedback $( { \mathrm { i . e . , ~ } } y _ { k } \ = \ x _ { k } )$ , stochastic noise $( \mathrm { i . e . , } \ w _ { k } \ \sim \ P _ { \mathrm { w } }$ and $v _ { k } \sim P _ { \mathrm { v } }$ with $P _ { \mathrm { { w } } }$ and $P _ { \mathrm { { v } } }$ being appropriately defined distributions), and bounded uncertainties $( \mathrm { i } . \mathrm { e } . , w _ { k } \in \mathcal { W }$ and $v _ { k } ~ \in ~ \mathcal { V }$ with $\mathcal { W } \subset \mathbb { R } ^ { n _ { \mathrm { w } } }$ and $\gamma ~ \subset ~ \mathbb R ^ { n _ { \mathrm { v } } }$ being bounded sets). While some methods specialize to linear dynamics, the strengths of modern machine learning typically unfold in the nonlinear setting. The discrete-time formulation (1) is predominant in learning-based control and adopted throughout; some methods instead build on continuous-time descriptions, where (1a) is replaced by a differential equation.

To control the system, we seek a feedback controller, also called a policy,

$$
u _ { k } = \pi _ { k } ( x _ { k } , r _ { k } ) \quad { \mathrm { o r } } \quad u _ { k } = \pi _ { k } ( y _ { k } , r _ { k } ) ,\tag{2}
$$

which maps the current state $x _ { k }$ (or output $y _ { k } )$ and possibly a reference signal $r _ { k }$ to the control input u . The subscript k allows for time-varying policies; stationary policies (i.e., $\pi _ { k } \equiv \pi )$ arise as a common special case.

The control objective is often cast as an optimization problem,

$$
\operatorname* { m i n } _ { \pi } J ,\tag{3}
$$

where the objective function J encodes the desired task such as reference tracking, energy minimization, or disturbance attenuation. It commonly depends on states, inputs, and the reference signal, with its specific arguments depending on the problem. The system may additionally be subject to state and input constraints,

$$
g ( x _ { k } , u _ { k } ) \leq 0 ,\tag{4}
$$

where $g : \mathbb { R } ^ { n _ { \mathrm { x } } } \times \mathbb { R } ^ { n _ { \mathrm { u } } } \to \mathbb { R } ^ { n _ { \mathrm { g } } }$ denotes the constraint function. In practice, the system model (1), the objective (3), and the constraints (4) may be only partially known or entirely unknown. To address these unknowns, learning-based control leverages machine learning from empirical data, possibly in combination with partial a priori knowledge, to synthesize a policy (2) that reliably achieves the desired objective while satisfying desired safety constraints. The empirical data used in learning-based control often consists of input, state, and/or output measurements:

$$
U = \left[ { { u _ { 0 } ^ { d } } \quad { u _ { 1 } ^ { d } } \quad \cdots \quad { u _ { N - 1 } ^ { d } } } \right] \in { \mathbb { R } ^ { n _ { \mathbf { u } } \times N } } ,\tag{5a}
$$

$$
X = \left[ x _ { 0 } ^ { d } \quad x _ { 1 } ^ { d } \quad \cdots \quad x _ { N - 1 } ^ { d } \right] \in \mathbb { R } ^ { n _ { \mathbf { x } } \times N } ,\tag{5b}
$$

$$
Y = \left[ y _ { 0 } ^ { d } \quad y _ { 1 } ^ { d } \quad \cdots \quad y _ { N - 1 } ^ { d } \right] \in \mathbb { R } ^ { n _ { \mathbf { y } } \times N } ,\tag{5c}
$$

where N denotes the length of the data sequence and the superscript $\ " d \ "$ denotes the (pre-)recorded dataset. For convenience, we further define $X ^ { + }$ as the state data shifted by one time step:

$$
X ^ { + } = \left[ x _ { 1 } ^ { d } \quad x _ { 2 } ^ { d } \quad \cdots \quad x _ { N } ^ { d } \right] \in \mathbb { R } ^ { n _ { \mathbf { x } } \times N } .\tag{6}
$$

Many problems in learning-based control assume that state measurements are available and use input and state data, i.e., (5a) and (5b), for learning. However, particularly challenging learning problems also arise when only output measurements are available, i.e., (5a) and (5c). Such problems often give rise to (learning-based) combinations of state estimation and system identification [22]–[28], which also relate to latent-space modeling in machine learning.

While the focus of this tutorial paper is on learning with state feedback and learning-based control, problems based on output measurements and state estimation are discussed at selected points throughout. Likewise, while we focus on offline learning, where the phase of data collection and learning is typically separate from controller operation, the connection to online learning, i.e., updating learned models or policies during operation, is discussed at relevant places.

## III. STABILITY AND SAFETY

We begin by introducing the control-theoretic foundations that provide the desired closed-loop properties of stability and constraint satisfaction. We first review these concepts in the nominal setting, in which the dynamics are known exactly, and then extend the discussion to robust and stochastic formulations that explicitly account for uncertainty. The latter settings are particularly relevant in learning-based control, where the dynamics are only approximately known from data and the resulting model error must be incorporated into the analysis either as a bounded disturbance or as a probabilistic uncertainty description. Figure 5 illustrates the main concepts of this section.

We first recall the standard comparison function classes used throughout this section [29]. A function $\alpha : \mathbb { R } _ { \geq 0 } $ $\mathbb { R } _ { \geq 0 }$ is of class K if it is continuous, strictly increasing, and satisfies $\alpha ( 0 ) = 0$ . It is of class $\kappa _ { \infty }$ if additionally $\alpha ( r ) \to \infty$ as r → ∞. A function $\beta : \mathbb { R } _ { \ge 0 } \times \mathbb { R } _ { \ge 0 } \to \mathbb { R } _ { \ge 0 }$ is of class $\kappa \mathcal { L }$ if $\beta ( \cdot , s ) \in \mathcal { K }$ for each fixed s, and $\beta ( r , s )  0$ as $s \to \infty$ for each fixed r.

## A. Nominal Stability

Stability concerns the asymptotic behaviour of a system around an equilibrium. Throughout this section, unless otherwise stated, the properties considered are local, i.e., they hold in a neighbourhood of the equilibrium. When the dynamics f are known, stability concerns an equilibrium x¯ of system (1a) in closed loop with a policy $u _ { k } = \pi ( x _ { k } )$

Definition 1 (Nominal stability [29]): Let x¯ be an equilibrium of system (1a) under the policy $u _ { k } \ : = \ : \pi ( x _ { k } )$ and with $w _ { k } \equiv 0$ . The equilibrium x¯ is

• stable if there exist $\gamma \in \mathcal { K }$ and $r > 0$ such that

$$
\| x _ { k } - \bar { x } \| \leq \gamma ( \| x _ { 0 } - \bar { x } \| )
$$

for all $k \geq 0$ and all initial conditions satisfying $\lVert x _ { 0 } -$ $\bar { x } \| < r ;$

• asymptotically stable if there exist $\beta \in \mathcal { K L }$ and $r > 0$ such that

$$
\| x _ { k } - \bar { x } \| \leq \beta ( \| x _ { 0 } - \bar { x } \| , k )
$$

for all $k \geq 0$ and all initial conditions satisfying $\lVert x _ { 0 } -$ $\displaystyle | { \bar { x } } | | < r .$

Lyapunov’s direct method certifies stability through a pointwise decrease condition on a scalar energy-like function.

Theorem 1 (Lyapunov’s direct method [29]): Let x¯ be an equilibrium and D a neighbourhood of x¯. Suppose there exists a function $V : \mathcal { D } $ R satisfying

$$
\alpha _ { 1 } ( \| x - { \bar { x } } \| ) \leq V ( x ) \leq \alpha _ { 2 } ( \| x - { \bar { x } } \| )
$$

for some $\alpha _ { 1 } , \alpha _ { 2 } \in { \mathcal { K } } _ { \infty } $ , and let

$$
\Delta V ( x _ { k } ) : = V ( x _ { k + 1 } ) - V ( x _ { k } )
$$

denote its one-step change along the closed-loop trajectories. Then:

1) if $\Delta V ( x _ { k } ) \le 0$ on D, then x¯ is stable;

2) if there exists $\alpha _ { 3 } \in \mathcal { K } _ { \infty }$ such that

$$
\Delta V ( x _ { k } ) \leq - \alpha _ { 3 } ( \| x _ { k } - \bar { x } \| )
$$

on D, then x¯ is asymptotically stable.

## B. Robust Stability

In practice, however, the dynamics model used for control design is rarely exact. Learning-based controllers rely on models identified from finite data and therefore inevitably incur approximation error. This motivates robust stability notions that explicitly account for uncertainty and disturbances.

radius γ(sup<sub>j</sub> ∥w<sub>j</sub> ∥) or γ(tr Σ) (b) Robust and stochastic stability

![](images/385c408bb20933b7416d62372e9a8a55cf7942a3f23bc63e26be5a6d274697a4.jpg)  
(a) Nominal stability

![](images/4cacedf0474ea9ed39794a6615c319809fe2cdad7b554230d4b13e5008b140e2.jpg)

![](images/4ed4c1108361c455f996ff211daadad328780c67ec6dac93d026d3903c2444e8.jpg)  
forward invariance (c) Invariance and safety

![](images/17a5fc65e587735c6480afe8edccbbdf32dd68df5f1df7045a4964ceeba67c30.jpg)  
(d) Hamilton-Jacobi reachability

![](images/d92bdf62dd01f100236bd3dd290e28c86135b57ae1c58e7d0fe769c2036fb4dd.jpg)  
(e) Control barrier function

![](images/6765e8830a056b51ad74325cf23ca0c52c3493ee9d70c95e0ebdbe81df17d894.jpg)  
(f) Predictive safety filter  
Fig. 5: Illustration of the stability and safety notions of this section. (a) Nominal stability: a Lyapunov function V decreases along closed-loop trajectories (level sets in grey) and the state converges to the equilibrium x¯. (b) Robust and stochastic stability: under disturbances w, the state converges to a neighbourhood of x¯ (grey), whose size depends on the disturbance bound or the noise covariance. (c) Safety: constraint satisfaction at one time instant does not imply safety (red dashed arrow); safety is guaranteed by an invariant set S (blue) contained in the constraint set (white). (d)–(f) Safety filters: th desired input $u _ { \mathrm { d e s } , k }$ would violate the constraint and is modified to $u _ { k } ^ { \star }$ to stay inside of the constraint set (white). HJ reachability computes the maximal safe set S offline, the largest invariant set within the constraint set. A CBF certifies invariance of $\{ b ( x ) \geq 0 \}$ by the one-step condition on $b ( x _ { k + 1 } )$ . A PSF does not compute the safe set but certifies safety by a backup trajectory (dotted) into a smaller terminal set $\boldsymbol { S } _ { f }$

In the robust setting, the disturbance is assumed bounded, i.e., $w ~ \in ~ \mathcal { W }$ , with $\mathcal { W } \subset \mathbb { R } ^ { n _ { \mathrm { w } } }$ a compact set. In the following, the model error and external disturbances are collected in the disturbance term w of (1a). A commonly used robust stability notion is then input-to-state stability (ISS), which characterizes the effect of the disturbance on the state trajectory.

Definition 2 (Input-to-state stability [29]): Let x¯ be an equilibrium of the system with $w \equiv \{ 0 \}$ , and consider (1a) in closed loop with $u _ { k } = \pi ( x _ { k } )$ and disturbance sequence $w = \{ w _ { k } \} , w _ { k } \in \mathcal { W }$ . The closed-loop system is input-tostate stable if there exist $\beta \in \mathcal { K L }$ and $\gamma \in \kappa$ such that, for every initial state $x _ { 0 }$ and every bounded disturbance sequence w,

$$
\| x _ { k } - \bar { x } \| \leq \beta ( \| x _ { 0 } - \bar { x } \| , k ) + \gamma \Big ( \operatorname* { s u p } _ { 0 \leq j < k } \| w _ { j } \| \Big ) , \qquad k \geq 0 .
$$

The ISS property separates the influence of the initial condition from that of the disturbance. The first term decays with time, whereas the second quantifies the residual error induced by uncertainty. As in the nominal case, ISS admits a Lyapunov characterization.

Theorem 2 (ISS-Lyapunov function [29]): Suppose there exists a function V and $\alpha _ { 1 } , \alpha _ { 2 } \in { \mathcal { K } } _ { \infty }$ with

$$
\alpha _ { 1 } ( \| x - \bar { x } \| ) \leq V ( x ) \leq \alpha _ { 2 } ( \| x - \bar { x } \| ) ,
$$

whose one-step change along the closed-loop trajectories satisfies

$$
\begin{array} { r } { V ( x _ { k + 1 } ) - V ( x _ { k } ) \le - \alpha _ { 3 } ( \| x _ { k } - \bar { x } \| ) + \sigma ( \| w _ { k } \| ) } \end{array}
$$

for some $\alpha _ { 3 } \in { \cal K } _ { \infty }$ and $\sigma \in { \mathcal { K } }$ . Then the closed-loop system is input-to-state stable in the sense of Definition 2.

When $w = \{ 0 \}$ , the ISS property reduces to

$$
\| x _ { k } - \bar { x } \| \leq \beta ( \| x _ { 0 } - \bar { x } \| , k ) ,
$$

thus recovering the asymptotic stability of the nominal system.

A related robustness notion is incremental stability [30], which studies the convergence between pairs of trajectories rather than toward a single equilibrium. This concept extends to incremental input-to-state stability (δ-ISS). Such properties can be used in order to compare nominal and perturbed trajectories in the presence of model mismatch.

## C. Stochastic Stability

When the disturbance $w _ { k }$ in (1a) is modelled probabilistically, with $w _ { k } \sim P _ { w } ,$ , the closed-loop system becomes a stochastic process. In this setting, stability is commonly formulated either in probability or in expectation.

Definition 3 (Stochastic stability [31], [32]): Let x¯ be an equilibrium for the system with $w \equiv \{ 0 \}$ , and consider (1a) in closed loop with $\begin{array} { r } { u _ { k } \ = \ \pi ( x _ { k } ) } \end{array}$ and independent and identically distributed (i.i.d.), zero-mean disturbance $w _ { k } \sim$ $P _ { w }$ with covariance Σ and finite second moment. The equilibrium x¯ is

• mean-square bounded if

$$
\operatorname* { l i m } _ { k \to \infty } \ \mathbb { E } \left\| x _ { k } - { \bar { x } } \right\| ^ { 2 } < \infty .
$$

• robustly asymptotically stable in expectation if there exist $\beta \in \mathcal { K L }$ and $\gamma \in \mathcal { K }$ such that

$$
\begin{array} { r } { \mathbb { E } \| x _ { k } - \bar { x } \| \leq \beta ( \| x _ { 0 } - \bar { x } \| , k ) + \gamma ( \mathrm { t r } \Sigma ) , \quad \forall k \geq 0 . } \end{array}
$$

• asymptotically stable in expectation if it is robustly asymptotically stable in expectation with $\gamma \equiv 0 ,$ , i.e., there exists $\beta \in \mathcal { K L }$ such that

$$
\begin{array} { r } { \mathbb { E } \| x _ { k } - \bar { x } \| \leq \beta ( \| x _ { 0 } - \bar { x } \| , k ) , \quad \forall k \geq 0 . } \end{array}
$$

As in the deterministic setting, stochastic stability can be established through a Lyapunov function satisfying an expected decrease condition.

Theorem 3 (Stochastic Lyapunov function [31], [32]): Suppose there exist a function V and $\alpha _ { 1 } , \alpha _ { 2 } \in { \mathcal { K } } _ { \infty }$ with

$$
\alpha _ { 1 } ( \| x - \bar { x } \| ) \leq V ( x ) \leq \alpha _ { 2 } ( \| x - \bar { x } \| ) ,
$$

a positive-definite stage cost $\ell ,$ and a constant $c \geq 0$ such that

$$
\mathbb { E } [ V ( x _ { k + 1 } ) \mid x _ { k } ] - V ( x _ { k } ) \leq - \ell ( x _ { k } ) + c .\tag{7}
$$

Then, the closed loop is mean-square bounded and satisfies

$$
\operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \sum _ { k = 0 } ^ { N - 1 } \mathbb { E } \ell ( x _ { k } ) \le c ,
$$

and the equilibrium x¯ is robustly asymptotically stable in expectation. If moreover $c = 0$ , the equilibrium x¯ is asymptotically stable in expectation.

The constant c in (7) measures the residual effect of stochastic uncertainty and plays a role analogous to the disturbance gain in ISS. When $c = 0$ there is no residual stochastic term and x¯ is recovered as an asymptotically stable equilibrium in expectation. Otherwise, the state remains bounded in mean square within a neighbourhood of x¯, while the long-term average performance remains bounded by c.

## D. Invariance and Safety Filters

While stability concerns convergence toward equilibria, safety concerns the satisfaction of state and input constraints (4) over time. Note that constraint satisfaction at a single time instant does not guarantee constraint satisfaction at future times, since there exist states satisfying the constraints from which a future constraint violation cannot be avoided under the system dynamics. Safety therefore requires identifying sets from which the constraints remain satisfied for all future times. The central notion underlying safety (in terms of constraint satisfaction) is thus invariance.

Definition 4 (Forward invariance [33]): Consider system (1a) in closed loop with the policy $u _ { k } = \pi ( x _ { k } )$ , and let $S \subseteq \mathbb { R } ^ { n _ { \mathbf { x } } }$ . The set S is said to be forward invariant if, for all initial conditions $x _ { 0 } \in S$ , the corresponding trajectory satisfies $x _ { k } \in S$ for all $k \geq 0$

A forward invariant set guarantees that trajectories initialized within it remain inside for all future times; safety then follows whenever such a set lies within the constraints. When the input is a decision variable, the relevant notion is control invariance, i.e., for every $x \in S$ , the existence of an input u satisfying (4) and keeping the state in S.

Invariant sets can either be integrated directly as constraints in the control problem, as e.g., common practice in MPC, or can be used to define a safety filter. Safety filters are independent of the control policy and merely verify the safety of a proposed control action, and otherwise implement a safety-ensuring modification. We focus on the safety filter formulation in the following. For a tutorial overview of common safety filter formulations and their datadriven extensions, see also e.g., [20].

Classical approaches to safety characterize invariant sets through reachability analysis. In particular, HJ reachability methods [13] compute the set of states from which trajectories can be kept within the constraints under the system dynamics by propagating backward the set of constraintviolating states. Such methods provide a global characterization of safety and typically yield tight approximations of the safe set under the assumed dynamics. However, their computational complexity scales poorly with the dimension of the state space, which limits their applicability to highdimensional systems and has motivated various approximate and learning-based extensions [13], [20], [34].

An alternative approach relies on CBFs [14], which enforce safety locally online through tractable conditions on the closed-loop dynamics. Analogously to Lyapunov functions for stability, barrier functions certify safety through conditions that guarantee forward invariance of a safe set. Consider a safe set of the form

$$
S : = \{ x \in \mathbb { R } ^ { n _ { \mathbf { x } } } : b ( x ) \geq 0 \} ,
$$

where $b : \mathbb { R } ^ { n _ { \mathbf { x } } }  \mathbb { R }$ . In discrete time, given a valid CBF, one can impose conditions on the control input such that

$$
b ( f ( x _ { k } , u _ { k } ) ) - b ( x _ { k } ) \geq - \alpha ( b ( x _ { k } ) )
$$

for some $\alpha ~ \in ~ \kappa$ satisfying $\alpha ( r ) ~ \leq ~ r$ for all $r \geq 0 .$ This condition ensures that trajectories cannot leave the safe set S, thereby rendering the set forward invariant. In discrete time, the barrier condition directly provides a one-step-ahead safety filter. A common optimization problem is given by

$$
\begin{array} { r l } { u _ { k } ^ { \star } : = \underset { u _ { k } } { \arg \operatorname* { m i n } } } & { \| u _ { k } - u _ { \mathrm { d e s } , k } \| ^ { 2 } } \\ { \mathrm { s . t . } } & { b ( f ( x _ { k } , u _ { k } ) ) - b ( x _ { k } ) \geq - \alpha ( b ( x _ { k } ) ) , } \end{array}
$$

where $u _ { \mathrm { d e s } , k }$ denotes the desired input generated by the nominal controller or learned policy at time k.

When comparing the computational cost of the two approaches, the construction of the safety certificate has to be distinguished from its online use. HJ reachability computes the value function on a grid over the state space. This is expensive and scales exponentially with the state dimension, but yields the (approximate) maximal safe set. A CBF only requires a function b satisfying the barrier condition, which is cheaper to obtain but generally provides an inner approximation of the safe set and is in general difficult to construct for nonlinear systems. Once the certificate is available, both filters are cheap to evaluate online, since the HJ value function can be used in the same way as a barrier function [35].

Predictive safety filters (PSFs) [36], [37] do not require the construction of a large safe set, but enforce safety by means of finite-horizon trajectory planning. Rather than explicitly computing a global invariant set, PSFs certify safety by ensuring the existence of a feasible backup trajectory that steers the system to a local invariant set. Given the current state and a desired input proposed by a controller or learning-based policy, a PSF solves a constrained finitehorizon optimal control problem of the form

$$
\begin{array} { r l } { \underset { \{ u _ { \{ i \} k } \} } { \operatorname* { m i n } } } & { \| u _ { 0 | k } - u _ { \mathrm { d e s } , k } \| ^ { 2 } } \\ { \mathrm { s . t . } } & { x _ { 0 | k } = x _ { k } , } \\ & { x _ { i + 1 | k } = f ( x _ { i | k } , u _ { i | k } ) , \quad \forall i \in \{ 0 , \ldots , T - 1 \} , } \\ & { g ( x _ { i | k } , u _ { i | k } ) \leq 0 , \quad \forall i \in \{ 0 , \ldots , T - 1 \} , } \\ & { x _ { T | k } \in S _ { f } , } \end{array}
$$

where $T$ is the horizon and $ { \boldsymbol { S } } _ { f }$ denotes a terminal safe set for which safety can be guaranteed beyond the prediction horizon through a local controller. Here, $x _ { i \mid k }$ and $u _ { i | k }$ denote the state and input predicted i steps ahead of time k, with $x _ { 0 | k } = x _ { k }$

The filter applies the first input $u _ { 0 | k } ^ { \star }$ of the optimal sequence: whenever the desired input admits a feasible backup trajectory, it is passed through unmodified; otherwise, it is minimally adjusted. Recursive feasibility and closedloop constraint satisfaction are guaranteed, provided that the terminal set $ { \boldsymbol { S } } _ { f }$ is (control) invariant. In contrast to HJ reachability and CBFs, the safe set is not computed explicitly. The terminal set $ { \boldsymbol { S } } _ { f }$ can be small and simple, e.g., an invariant set of a local linear controller, and the region from which safety is certified is enlarged by the online planning over the horizon T. The computational effort is thereby shifted from the offline set computation to the online optimization.

A closely related approach is shielding [38], [39], originally proposed in the context of RL, in which a supervisory mechanism overrides the learned policy whenever it would lead to unsafe behaviour. In particular, model predictive shielding can be interpreted as a simulation-based variant of the PSF: rather than solving an optimal control problem online, it forward-simulates a fixed backup policy and verifies that from the successor state there exists a feasible trajectory under the backup policy that ensures future constraint satisfaction. If this is not the case, the backup policy is applied instead.

The three approaches thus differ with respect to the representation of the safe set: it is computed explicitly by HJ reachability, characterized by a barrier function in the case of a CBF, and defined implicitly by the existence of a feasible backup trajectory in the case of a PSF.

The complexity of the online problem differs between the approaches. For HJ reachability, the filter only requires the evaluation of the precomputed value function [35]. The CBF filter has a single one-step constraint. It is convex if $b \circ f$ is concave in $u _ { k } , \ \mathrm { e . g . }$ , for linear dynamics and a concave barrier function; otherwise, it may be a small nonconvex program. In continuous time and for control-affine dynamics, the barrier constraint is affine in the input and the filter is a quadratic program (QP) [14]. The PSF is a finite-horizon optimal control problem with T stages. It is a QP for linear dynamics with polytopic constraints and terminal set, and a nonconvex nonlinear program for nonlinear dynamics.

## E. Robust and Stochastic Invariance

The notion of invariance extends to uncertain systems, where it must hold for all admissible disturbances in robust settings [33], and in probability or in expectation in stochastic ones [40], [41].

In the robust case, HJ reachability extends to worst-case analysis by computing invariant sets under adversarial disturbances [42], [43], robust CBFs incorporate the uncertainty directly into the barrier conditions [44], [45], and PSFs account for it through model-based constraint tightening mechanisms [36], [37]. In the stochastic case, stochastic reachability methods characterize the probability of remaining within the safe set [46], barrier-based approaches admit stochastic variants with prescribed confidence levels [47], and PSFs extend through chance-constrained or risk-based optimization formulations [48].

## F. Relaxed Safety Formulations

An alternative formulation of safety, particularly common in RL, replaces hard constraint satisfaction by bounds on expected safety costs. This leads to constrained Markov decision processes (CMDPs) [49], in which the control objective is optimized subject to constraints of the form

$$
\mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } c ( x _ { k } , u _ { k } ) \right] \leq d ,
$$

where c measures safety violations and d denotes an allowable constraint budget. Such formulations trade strict invariance guarantees for long-term probabilistic or averageperformance guarantees.

In practice, strict constraint satisfaction may be infeasible or excessively conservative in the presence of uncertainty and disturbances. For this reason, many safety formulations employ soft constraints [50], which permit temporary constraint violations while preserving optimization feasibility.

Consider the constraints (4). A soft-constrained formulation introduces a nonnegative slack vector $\varepsilon _ { k } \geq 0$ and relaxes the constraints to

$$
g ( x _ { k } , u _ { k } ) \leq \varepsilon _ { k } ,
$$

where each component of $\varepsilon _ { k }$ specifies the permitted violation of the corresponding constraint. The resulting optimization problem penalizes constraint violations through an augmented objective of the form

$$
\begin{array} { r } { J _ { \mathrm { s o f t } } = J + \lambda _ { 1 } \| \varepsilon \| _ { 1 } + \lambda _ { 2 } \| \varepsilon \| _ { 2 } ^ { 2 } , } \end{array}
$$

where $\varepsilon$ denotes the slack vectors collected over the optimization horizon, and $\lambda _ { 1 } , \lambda _ { 2 } \geq 0$ determine the tradeoff between performance and constraint violation. Linear and quadratic penalties are typically combined: the linear term keeps the penalty exact, while the quadratic term penalizes large violations more strongly. In convex settings, choosing $\lambda _ { 1 }$ sufficiently large makes the penalty exact, i.e., the soft-constrained problem recovers the solution of the original hard-constrained problem whenever the constraints can be satisfied, and violations occur only when the hard-constrained problem is infeasible.

## IV. DYNAMICS LEARNING FOR CONTROL

Many control methods use a dynamics model to synthesize a controller for the problem formulated in Section II and to analyze the resulting closed-loop properties, as described in Section III. The achievable performance and guarantees depend on the accuracy of the model, and on how the associated uncertainty is accounted for. In practice, dynamics models may be constructed from our prior knowledge (e.g., firstprinciples physical models) for some applications; for others, such models may be unavailable or may not adequately capture complex dynamics. This has motivated a large body of work in learning-based control that incorporates measured data to learn or refine dynamics models, thereby complementing incomplete prior knowledge. Within this body of work, the learned models are leveraged to improve control performance, while control-theoretic analysis provides the foundation for retaining desired closed-loop properties.

To facilitate our discussion, we distinguish between direct and indirect data-driven control. In indirect data-driven control, a system model is distilled based on data collected from the real system, and this model is then used for controller design [72]. Direct data-driven control approaches, on the other hand, map system observations to control actions based on past data, without explicitly defining a model. In this section, we focus on the former and provide an introduction to common dynamics learning approaches, together with how they have been integrated with control frameworks. Methods for direct data-driven control are described in greater depth in Section V. In the following subsections, we first formulate the dynamics learning problem and then review common methodological threads, ranging from linear dynamics learning to deterministic nonlinear dynamics learning, and to probabilistic nonlinear dynamics learning approaches.

## A. Dynamics Learning Problem

We consider a discrete-time system of the form (1a), for which the dynamics can be partially known or fully unknown. Our goal is to infer a model of the dynamics from a sequence of measured inputs and states of length N, as defined in (5a), (5b), and (6). For the discussion below, we consider systems subject to additive process noise and approximate the true dynamics with a model $\hat { f }$ of the form

$$
\hat { f } ( x , u ) = \hat { f } _ { n } ( x , u ) + \hat { f } _ { l } ( x , u ) ,\tag{8}
$$

where $\hat { f } _ { n }$ is a nominal model that captures available prior knowledge, and $\hat { f } _ { l }$ is the component to be learned from data. When the system dynamics are fully unknown, the nominal model ${ \hat { f } } _ { n }$ can be set to the zero function. For convenience in the following discussion, we define an augmented vector z = $[ x ^ { \top } , u ^ { \top } ] ^ { \top }$ , which is the input to both model components.

Over the years, many works have considered dynamics learning problems of the form (8). Beyond incorporating a prior dynamics model, dynamics learning can also leverage structural knowledge of the system [73]–[77] to improve learning efficiency or model accuracy. We categorize the resulting frameworks according to the formulation of the learned model, distinguishing between linear and nonlinear models as well as deterministic and probabilistic formulations. The chosen formulation influences how model uncertainty can be characterized and accounted for during closed loop operation. In the following, we provide an overview of representative methods, with particular emphasis on the learning framework, the controller synthesis methodology, and the guarantees that can be provided under possible dynamics uncertainties. A summary of the different learning methods, along with their advantages, disadvantages, compatible control methods, and the stability/safety guarantees they enable, is given in Table I. Note that the control methods provided in the table are not exhaustive, but are only intended to serve as a starting point for those new to the field.

## B. Learning Linear System Models

We begin our discussion with linear dynamics learning problems, where the true dynamics can be written as

$$
x _ { k + 1 } = A x _ { k } + B u _ { k } + w _ { k }\tag{9}
$$

with A and B being system matrices of appropriate dimensions. For simplicity, we first suppose that the nominal component is absent, and the approximated dynamics model analogously takes a linear form: $x _ { k + 1 } = { \hat { A } } x _ { k } + { \hat { B } } u _ { k }$ . While our focus here is on linear dynamics, for nonlinear systems, linear models can also be used to approximate the local dynamical behaviour around an operating point. Such linear representations can simplify controller synthesis, which can be desirable in certain applications where the approximation error is sufficiently small.

The model parameters $\Theta \ = \ [ \hat { A } , \hat { B } ]$ are typically determined using least-squares regression (or its variants). Specifically, the learning problem may be formulated as the following optimization problem: arg min<sub>Θ</sub> $\| X ^ { + } - \Theta M \| _ { F } ^ { 2 }$ , where $\bar { M } = [ X ^ { \top } , ~ U ^ { \top } ] ^ { \top }$ . If $M M ^ { \dagger }$ is invertible, we can find the closed-form solution $\Theta ~ = ~ { \cal X } ^ { + } M ^ { \top } ( M M ^ { \top } ) ^ { - 1 }$ This formulation can be related to the maximum likeli hood estimation problem (i.e., maximizing the probability of observing the data given the model parameters) under appropriate noise assumptions [78]. To generalize to the case when the nominal model $\hat { f } _ { n }$ is available, the closed-form solution can be obtained by adapting the datasets to capture the residual dynamics $x _ { k + 1 } - { \hat { f } } _ { n } ( x _ { k } , u _ { k } )$

Note that the computational complexity of linear regression is linear in the number of data samples. However, even for systems with true linear dynamics, convergence to true parameters can require a large amount of data; it was shown in [79] that, for certain cases, the worst-case sample complexity is exponential. Nevertheless, finite-sample error

Stability and safety filters. The relevant notion of stability depends on the assumptions on the system and its uncertainty. Nominal, robust and stochastic settings therefore lead to different stability guarantees, which are, however, only valid under the respective assumptions. In learning-based control, the characterization of the model error determines the setting, since the error enters the analysis either as a bounded disturbance or as a probabilistic uncertainty description.

By means of a safety filter, an arbitrary control law, e.g., a learned policy, can be augmented with guarantees on constraint satisfaction. Hamilton-Jacobi (HJ) reachability, control barrier functions (CBFs) and predictive safety filters (PSFs) are all based on the notion of invariance, but differ with respect to conservativeness, scalability and computational effort. The choice among the three approaches is dependent on the particular problem: HJ reachability is attractive when the state dimension is small, CBFs when a valid barrier function can be constructed and PSFs when the computational effort can be shifted to the online optimization. Recent results focus on combining these techniques in order to leverage their individual benefits and to address their limitations using learning [20]. A detailed design example and comparison of the three different safety filter concepts for an inverted pendulum problem can be found in [20]. Note that filtering concepts can also incorporate stability requirements by incorporating suitable Lyapunov decrease conditions, see e.g. [51], [52].

Various software implementations are available for the presented safety filter concepts. In the following, a selection of commonly used tools is given, which is, however, not intended to be complete. For HJ reachability, the value function can be

bounds can be estimated [80], [81]. These can be normbounded estimation errors (e.g., $\| \Delta _ { A } \| \le e _ { A }$ and $\| \Delta _ { B } \| \le$ $e _ { B }$ with $\Delta _ { A } = A - { \hat { A } }$ and $\Delta _ { B } = B - { \hat { B } } \ )$ , or in the form of confidence ellipsoidal sets where

$$
\mathrm { t r a c e } \left( \left[ \Delta _ { A } \quad \Delta _ { B } \right] Q _ { p } \left[ \Delta _ { A } ^ { \top } \right] \right) \leq c
$$

holds with probability at least $( 1 - \delta )$ for some positive definite matrix $Q _ { p }$ and positive scalar c that depends on a user-chosen confidence parameter δ.

The primary objective of the uncertainty bounds obtained above is to inform the closed-loop controller synthesis, which may be performed for linear systems using well-established techniques such as linear-quadratic regulator (LQR) and MPC. When using LQR, it is shown that a stabilizing controller can be determined with a suboptimal guarantee with estimated dynamics [82]. Alternatively, if the errors are large, then one can also use robust LQR tools (e.g., systemlevel synthesis [83]). For MPC-based controller synthesis in the presence of norm-bounded error estimates, one can rewrite the system dynamics as $\begin{array} { r } { x _ { k + 1 } = \hat { A } x _ { k } + \hat { B } u _ { k } + \tilde { w } _ { k } , } \end{array}$ where $\tilde { w } _ { k } = ( \Delta _ { A } ) x _ { k } + ( \Delta _ { B } ) u _ { k } + w _ { k }$ . If $x _ { k } \in \mathcal X \subseteq \mathbb R ^ { n _ { \mathbf { x } } }$ $u _ { k } \ \in \mathcal U \ \subseteq \ \mathbb R ^ { n _ { \mathrm { u } } } ,$ and $w _ { k } \in \mathcal W \subseteq \mathbb R ^ { n _ { \mathrm { w } } }$ , an aggregate disturbance set Wf can be constructed such that $\Delta _ { A } \mathcal { X }$ ⊕ $\Delta _ { B } u \oplus \mathcal { W } \subseteq \tilde { \mathcal { W } }$ . The resulting set can then be accounted for using constraint tightening or tube-based robust MPC methods [84]. Similarly, when confidence bounds for the learned dynamics are available, robust MPC methods can be extended to ensure constraint satisfaction in the probabilistic computed on a grid of the state space using, $\mathrm { e . g . }$ , the Level Set Toolbox [53], hj reachability [54] or OptimizedDP [55]. DeepReach [56] approximates the value function by means of a neural network and is thereby applicable to systems of higher state dimension. Reachable sets based on set representations can be computed with, e.g., CORA [57] or JuliaReach [58].

The online evaluation of a CBF filter amounts to the solution of a small optimization problem, which, if it can be formulated as a QP, can be solved using standard solvers such as HPIPM [59], OSQP [60], Clarabel [61] or qpOASES [62]. Note that the construction of a valid barrier function is in general not automated. In the following, some tools are given which support this step. For polynomial systems, barrier certificates can be obtained by means of sum-of-squares programming [63] using, e.g., SOS-TOOLS [64]. FOSSIL [65] synthesizes Lyapunov functions and barrier certificates represented by neural networks and formally verifies the resulting candidates, whereas refineCBF [66] refines a given candidate function by means of HJ reachability such that a valid CBF is obtained. CBFkit [67] provides implementations of CBF-based safety filters for robotic applications and, in addition, allows a candidate barrier function to be fitted to labeled data by means of a neural network, the validity of which has to be verified separately.

The online problem of a PSF is a finite-horizon optimal control problem, which can be formulated using, e.g., CasADi [68] and solved by a general-purpose nonlinear programming solver such as IPOPT [69]. Alternatively, frameworks that are specifically designed for real-time optimal control, see, e.g., acados [70] and FORCESPRO [71], can be applied.

sense or can be combined with stochastic MPC. This will be further discussed when dealing with probabilistic models in Section IV-D.

Safety filters may also be employed (e.g., [36]) to ensure robust constraint satisfaction during closed-loop operation. For unknown linear systems subject to (sub-)Gaussian noise, learning-based safety filters use estimates of the dynamics and noise covariance to ensure constraint satisfaction with high probability [85], [86]. Related efforts investigate when multi-step prediction can reduce compounding model errors [87] and develop finite-sample guarantees for learning and control [88].

## C. Learning Deterministic Nonlinear Models

Having introduced linear dynamics learning, we now turn to nonlinear systems. While local linear approximations can be effective in some applications, their accuracy may be limited for systems with strongly nonlinear dynamics. Learning nonlinear dynamics models from data provides a means to capture more complex system behaviour for further improving closed-loop performance. Learning general nonlinear models of the form (8) without any prior structural parameterization can be inefficient. We can often leverage a priori knowledge of the system (e.g., leveraging differential flatness, control-affine structures, or polynomial dynamics) to facilitate the learning process.

One common set of approaches in the literature involves learning residual dynamics with linear regression:

$$
x _ { k + 1 } = { \hat { f } } _ { n } ( z _ { k } ) + \Theta \phi ( z _ { k } ) ,\tag{10}
$$

where $\phi ( z _ { k } )$ is a vector of basis functions capturing system nonlinearities, and Θ is the parameter matrix to be learned. Note that, while the model in (10) is nonlinear in $z _ { k }$ , it is linear in the parameter matrix Θ; consequently, under the corresponding rank condition, the least-squares estimate of Θ admits a closed-form solution, similar to the case in Section IV-B.

When prior knowledge of the system is limited, nonlinear dynamics can instead be represented using neural networks, which are often adopted because of their universal approximation capabilities [89]. In a typical setting (e.g., [90], [91]), the learnable model $\dot { f } _ { l , \theta }$ is a feedforward network and is defined as a sequence of affine and nonlinear transformations [92]:

$$
\left\{ \begin{array} { l l } { z ^ { 0 } = z , } \\ { z ^ { i + 1 } = \arctan ( W ^ { i } z ^ { i } + b ^ { i } ) , \quad i \in \{ 0 , \ldots , l - 1 \} } \\ { z ^ { l + 1 } = W ^ { l } z ^ { l } + b ^ { l } . } \end{array} \right.\tag{11}
$$

Here, l denotes the number of hidden layers in the neural network, $z ^ { 0 } = z$ is the input of the network, $z ^ { l + 1 }$ is the output of the network, θ denotes the augmented vector of the learnable parameters $W ^ { i }$ and $b ^ { i }$ across all layers, and act(·) is a nonlinear activation function (e.g., ReLU or tanh). The goal of the training process is to reduce the error between the predicted output and the target in the dataset. One commonly used objective is the mean-squared error:

$$
\operatorname* { m i n } _ { \theta } \frac { 1 } { N } \sum _ { j = 0 } ^ { N - 1 } \left\| x _ { j + 1 } - \left( \hat { f } _ { n } ( z _ { j } ) + \hat { f } _ { l , \theta } ( z _ { j } ) \right) \right\| ^ { 2 } ,\tag{12}
$$

where the subscript j denotes the j-th data point with $z _ { j } =$ $[ x _ { j } ^ { \top } , u _ { j } ^ { \top } ] ^ { \top }$ and $x _ { j + 1 }$ being the corresponding successor state. Beyond standard feedforward neural networks, nonlinear dynamics learning may also employ frameworks such as neural ODEs [93], physics-informed neural networks [94], [95], and neural autoregressive models [96] to incorporate richer dynamical structure, physical knowledge, or temporal dependencies. Learned neural network dynamics models have been combined with nonlinear MPC [97]–[100] and safety filters [101].

While neural networks are expressive, their black-box nature makes it more difficult to provide closed-loop guarantees. Nevertheless, recent advances have attempted to address this challenge by exploiting the inherent properties of learned models. One property commonly leveraged in closed-loop analysis is Lipschitz continuity. For feedforward neural networks with a finite number of layers and globally Lipschitz-continuous activation functions, there exists a constant $L \ge 0$ such that $\lVert \hat { f } _ { l , \theta } ( z ) - \hat { f } _ { l , \theta } ( z ^ { \prime } ) \rVert \leq L \lVert z - z ^ { \prime } \rVert ,$ $\forall z , z ^ { \prime } \in \mathbb { R } ^ { n _ { \mathrm { x } } + n _ { \mathrm { u } } }$ . Intuitively, the Lipschitz property allows us to bound the change in the network output induced by variations or perturbations in its input. While estimating the Lipschitz constants can be non-trivial [102], Lipschitz constraints can be encouraged or enforced through training procedures [103]–[105] or directly through the network architecture design [106]. A representative training-based approach is spectral normalization [103], in which the weight matrix of each layer is rescaled during training to constrain its spectral norm. If the activation functions are 1-Lipschitz with respect to the Euclidean norm, as in the case of many common activation functions (e.g., ReLU and tanh), then an upper bound on the network Lipschitz constant is given by $\textstyle \prod _ { i = 0 } ^ { l } \| W ^ { i } \| _ { 2 }$ . Beyond spectral normalization and related weight-projection methods, Lipschitz continuity can also be imposed through network architectures that combine norm-constrained linear mappings with gradientnorm-preserving activation functions [106]. Both spectrally normalized networks and Lipschitz-constrained architectures have been integrated with control-theoretic tools, including Lyapunov-based nonlinear control and small-gain arguments, to establish closed-loop stability guarantees for learning-enhanced systems in practical applications [107], [108]. Besides Lipschitz continuity, neural networks can also be characterized through local sector bounds or quadratic constraints on their activation functions; these descriptions preserve more information about the network structure than a single global Lipschitz bound and can be incorporated into Lyapunov- and robust-control analyses (e.g., [109], [110]) to establish desired closed-loop properties.

While the preceding approaches are well-suited to analyzing feedforward neural networks as static input-output mappings, many learning approaches utilize autoregressive models, recurrent neural networks, and neural ODEs, which naturally define dynamical models with internal states or temporal dependencies induced by state-input sequences of the form (5a), (5b), and (6). For these models, it is important to analyze how their internal states and output trajectories evolve over time. This can be done by studying properties such as ISS and δ-ISS and using them in the closed-loop analysis to establish convergence of the system trajectories to the desired behaviour or to a bounded neighbourhood thereof in the presence of model mismatch (see Section III). In this regard, sufficient conditions for ISS and δ-ISS properties have been obtained for neural autoregressive models [111] and recurrent neural networks [112], [113]. Related contractivity and dissipativity guarantees have been developed for neural ODEs [114], while stable-by-design recurrent networks are proposed in [115]. These model properties can be incorporated directly into controller analysis and synthesis. For instance, in [97], δ-ISS properties are leveraged for the derivation of MPC laws ensuring stability and recursive feasibility. This has also been extended to offset-free tracking control in [98].

## D. Learning Probabilistic Nonlinear Models

Although nonlinear dynamics learning methods are more flexible than their linear counterparts, this flexibility can come at the cost of conservatism due to the assumed worst-case bounds when handling model uncertainty. While methods such as approximate Bayesian inference [116], [117], approximate Cramer–Rao lower bound [118], and´ deep ensembles [119] exist, accurately and reliably estimating predictive uncertainty for highly expressive function approximators, such as neural networks, remains a non-trivial problem. For closed-loop control, one may alternatively leverage probabilistic learning methods, which allow us to systematically bookkeep model uncertainties. Such probabilistic characterizations enable the use of chance constraints or risk-sensitive formulations in place of purely worst-case constraints, providing a principled means to trade off control performance and confidence in constraint satisfaction.

To make the problem tractable, i.i.d. Gaussian assumptions are often imposed on the process or measurement noise. For a model that is linear in the parameters, e.g., of the form (10), and with a Gaussian prior $\theta \sim \mathcal { N } ( \mu _ { \theta } , \Sigma _ { \theta } )$ on the unknown parameter vector $\theta \in \mathbb { R } ^ { n _ { \phi } }$ for a single output dimension, the posterior distribution over θ is again Gaussian and available in closed form [92]; this is known as Bayesian linear regression (BLR). In the single-output case, the posterior mean and covariance are $\hat { \mu } _ { \boldsymbol { \theta } } = \bar { \hat { \Sigma } } _ { \boldsymbol { \theta } } \left( \Sigma _ { \boldsymbol { \theta } } ^ { - 1 } \mu _ { \boldsymbol { \theta } } + \sigma _ { \boldsymbol { w } } ^ { - 2 } \bar { M _ { \phi } } X _ { + } ^ { \top } \right)$ and $\begin{array} { r } { \hat { \Sigma } _ { \theta } ^ { - 1 } = } \end{array}$ $\Sigma _ { \theta } ^ { - 1 } + \sigma _ { w } ^ { - 2 } M _ { \phi } M _ { \phi } ^ { \top }$ , where $M _ { \phi } = [ \phi ( z _ { 0 } ) \ldots \phi ( z _ { N - 1 } ) ] \in$ $\mathbb { R } ^ { n _ { \phi } \times N }$ . The multi-output case follows by applying these expressions independently to each row of Θ.

Apart from BLR, nonlinear dynamics can also be represented using non-parametric probabilistic models such as GPs [120]. When approximating $\hat { f } _ { l }$ with a GP, we assume that its function values are jointly Gaussian, with covariance specified by a kernel function κ. For illustration, we consider a single output and define the residual training targets as $\Gamma ^ { \top } \stackrel { - } { = } X ^ { + } - \hat { F } _ { n } ( M )$ ), where $\hat { F } _ { n } ( M )$ denotes the nominalmodel predictions evaluated at the training inputs. Then, for a new test point $z = [ x ^ { \top } , u ^ { \top } ] ^ { \top }$ , we have

$$
\left[ \underset { \ b { \hat { f } } _ { l } ( z ) } { \Gamma } \right] \sim \mathcal { N } \bigg ( 0 , \left[ \begin{array} { c c } { \ b { K } + \Sigma ^ { w } } & { \ b { K } _ { z } } \\ { \ b { K } _ { z } ^ { \top } } & { \kappa ( z , z ) } \end{array} \right] \bigg ) ,\tag{13}
$$

where K is the Gram matrix computed from data $( X , U )$ using $\kappa , K _ { z }$ contains the kernel evaluations between the training inputs and the test input z, and $\Sigma ^ { w }$ denotes the covariance matrix of the noise term. Conditioning on the observed residual targets yields $p \left( { \hat { f } } _ { l } ( z ) \mid Z , \Gamma \right) = { \mathcal { N } } ( { \hat { \mu } } , { \hat { \Sigma } } )$ , where $\hat { \mu } =$ $K _ { z } ^ { \top } \left( K + \Sigma ^ { w } \right) ^ { - 1 } \Gamma$ and $\hat { \Sigma } = \kappa ( z , z ) - K _ { z } ^ { \top } \left( K + \Sigma ^ { w } \right) ^ { - 1 } K _ { z }$ The posterior mean of the full dynamics can then be expressed as $\hat { f } _ { n } ( z ) + \hat { \mu } ( z )$

In practice, BLR and GPs offer complementary advantages and limitations. For both model classes, reliable predictive uncertainty estimates depend on appropriate prior structure, encoded through the choice of basis functions in BLR and the kernel and its hyperparameters in GPs. This dependence is illustrated in Example 2. Moreover, computationally, fitting a GP typically scales cubically with respect to the number of data samples, while BLR scales linearly, offering an advantage over GP [121], at the cost of representational flexibility. While this computational tradeoff may not be critical in offline learning settings, online learning settings often require a careful balance between fast adaptation and model capacity.

It should be noted that the learned probabilistic models only make one-step predictions rather than trajectory predictions. This can be challenging in closed-loop control methods such as MPC, where dynamics uncertainty must be propagated over the prediction horizon. Common approaches approximate the distributions of subsequent predicted states using methods such as linearization [123], moment matching [124], and sigma-point propagation [125]. Depending on the formulation, additional simplifying assumptions may be introduced, such as neglecting correlations between successive GP evaluations. Starting from a state distribution at time k with mean $x _ { k } ^ { \mu }$ and covariance $\Sigma _ { k } ^ { x }$ , these methods yield approximate predicted state distributions characterized by $( x _ { i \mid k } ^ { \mu } , \Sigma _ { i \mid k } ^ { x } )$ for future prediction steps $i > k$

Closed-loop guarantees can then be pursued by solving a stochastic optimal control problem with chance constraints, both in the case of BLR and GP [121], [126]. For example, a joint chance constraint over a finite prediction horizon may be written as $p \big ( g ( x _ { i \mid k } , u _ { i \mid k } ) \ : \le \ : 0 , \ : \ : \forall i \big ) \ : \ge \ : \alpha$ , where α is a user-defined confidence level. To obtain a tractable controller synthesis problem, such probabilistic constraints are commonly replaced by deterministic reformulations or conservative approximations of the form $g ( x _ { i \vert k } ^ { \mu } , u _ { i \vert k } ) \ +$ $\nu ( x _ { i | k } ^ { \mu } , u _ { i | k } , \Sigma _ { i | k } ^ { x } ) \leq 0$ , where $\nu ( \cdot )$ represents a constraint tightening determined by the predicted state uncertainty, the assumed probability distribution, and the selected confidence level. The resulting optimization problem can then be solved (e.g., via MPC) as

$$
\operatorname* { m i n } _ { u _ { i \mid k } } \quad J
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { { } ~ x _ { 0 \mid k } ^ { \mu } = x _ { k } , } \end{array}\tag{14a}
$$

$$
\Sigma _ { 0 | k } ^ { x } = 0 ,\tag{14b}
$$

(14c)

$$
x _ { i + 1 | k } ^ { \mu } = \rho ( x _ { i | k } ^ { \mu } , \Sigma _ { i | k } ^ { x } , u _ { i | k } ) ,\tag{14d}
$$

$$
\Sigma _ { i + 1 | k } ^ { x } = \Lambda ( x _ { i | k } ^ { \mu } , \Sigma _ { i | k } ^ { x } , u _ { i | k } ) ,\tag{14e}
$$

$$
g ( x _ { i | k } ^ { \mu } , u _ { i | k } ) + \nu ( x _ { i | k } ^ { \mu } , u _ { i | k } , \Sigma _ { i | k } ^ { x } ) \leq 0 ,\tag{14f}
$$

$$
\forall i \in \{ 0 , \ldots , T - 1 \} ,
$$

where $x _ { k }$ is the current state, $\rho ( \cdot )$ and $\Lambda ( \cdot )$ are the propagation of the predicted state mean and covariance under the chosen uncertainty-propagation method. Note that the covariance propagation significantly increases the computational complexity of the optimization problem. The computational effort can be reduced by employing SQP methods with tailored Jacobian approximations [127], for which an efficient open-source implementation is provided in [128]. Moreover, an ancillary controller can be introduced to reduce deviations from the predicted nominal trajectory, thereby reducing uncertainty growth over the prediction horizon and the associated constraint tightening. A review of MPC with GP models can be found in [126], and Example 3 shows an example of GP-MPC.

An alternative to the probabilistic uncertainty-propagation methods above is to use robust set-based propagation to construct ellipsoidal confidence regions for the predicted states [129]. In this setting, the state x is assumed to belong to an ellipsoidal uncertainty set $\mathcal { E } ( x _ { p } , Q _ { p } )$ centred at $x _ { p } .$ with shape determined by $Q _ { p }$ . The system dynamics robustly propagate from one ellipsoidal set to another, and the sum

Learning the unknown. When the system dynamics are not fully known, learning can help identify unknown components and thereby improve control performance. Reliable learning depends on both informative data and appropriate prior structure. Observed data constrain the model within sampled regions, while prior structure shapes its generalization beyond them. With two probabilistic learning approaches, this example illustrates how data and structure jointly influence predictive accuracy and the reliability of the associated uncertainty estimates. The implementation for this example is available in [122] (BLR vs. GP).

We consider a car traversing unknown terrain, represented by an elevation profile $h ( p )$ where $p$ denotes the car’s horizontal position and $h ( p )$ denotes the corresponding terrain height. For this example, the true terrain profile is $h ( p ) \stackrel { \textstyle } { = } 0 . 0 1 \cos ( 1 8 p )$ and is observed through noisy measurements $\mathcal { \bar { D } } = \{ ( \boldsymbol { p } ^ { ( d ) } , \boldsymbol { h } ^ { ( d ) } ) \} _ { d = 1 } ^ { D }$ where the superscript $( d )$ denotes the data index, $\bar { h } ^ { ( d ) } = h ( p ^ { ( \bar { d } ) } ) \bar { + }$ $\varepsilon ^ { ( d ) }$ , and $\varepsilon ^ { ( \hat { d } ) } \sim \dot { \mathcal { N } } ( \mu _ { \varepsilon } , \sigma _ { \varepsilon } ^ { 2 } )$ is i.i.d. Gaussian noise with mean $\mu _ { \varepsilon } = 0$ and variance $\sigma _ { \varepsilon } ^ { 2 } = 2 . 5 \times 1 0 ^ { - 5 }$ . Our goal is to learn the function $h ( p )$ using the dataset D.

Figure 6 compares the learning performance of BLR and GP, when different structural assumptions were made. From the topleft panel, we see that, with exactly matching periodic basis functions, the predictive mean and variance of the BLR (red) generalize reliably, even across the unsampled data gap (grey). As further illustrated in the bottom-left panel, when such taskspecific structure is unavailable, a GP, with its hyperparameters tuned via maximizing the log marginal likelihood, can be a flexible alternative, though with increased uncertainty bounds over the

of these ellipsoids gives the multi-step predictions of the system, i.e., they approximate the reachable sets. Then, e.g., a tube-based MPC can be formulated with set-based system dynamics for robust satisfaction of constraints.

While these robust tools for closed-loop control guarantees provide worst-case constraint satisfaction, the confidence regions are often difficult to obtain, or, when available, too conservative in practice. A solution to this is to use samplingbased approaches for constraint tightening [130], [131] or using scenario-based MPC [132]. Separately, problems with GPs can also be solved using LQR via local linearizations [133], [134].

An important concern for closed-loop control with GPs is stability. Within the context of MPC, this can be guaranteed by imposing additional assumptions, for example, the existence of robust positively invariant sets [129], and requires careful selection of terminal constraints. Constructing them is often difficult in practice, so they are handled via samplingbased approaches, back-up controllers or soft constraints. Stability can also be addressed beyond the receding-horizon setting via Lyapunov functions [73]. Here, GP dynamics are slightly modified to ensure a known equilibrium, i.e., $f ( 0 ) = 0 $ , and fidelity on model knowledge is determined through uncertainty bounds. These uncertainty bounds are then incorporated to determine a CLF and a stabilizing feedback law through dynamic programming. Lyapunov-based constructions have also been used for tracking control [135] and backstepping control design [136].

unsampled data region. Overall, both methods rely on structural assumptions (e.g., basis functions for BLR and the kernel and its hyperparameters for GPs). The right-hand panels illustrate the consequences of model misspecification. An incorrect basis or kernel hyperparameters can bias the predictive mean and produce overconfident predictions. Reliable generalization depends on appropriate structural assumptions, especially when only a limited number of observations is available.

![](images/bb9177f575eb1f26af7fd4cb8c61bdbcd6a595dfbfd332db0caf980c59b43522.jpg)  
Fig. 6: Effect of prior structure on learning from incomplete data. The grey region contains no measurements; the dashed blue and solid red curves denote the true profile and posterior mean, respectively, and the shaded bands correspond to model predictive uncertainties.

## E. Data Collection for Offline Versus Online Learning

The dynamics learning methods discussed in this section are performed predominantly in an offline manner, i.e., a suitable model is learned based on a fixed, input-output dataset, and the control is designed based on the learned model prior to closed-loop deployment. The performance of such methods is tied to the assumption that the collected data is informative enough for estimation. In the machine learning literature, similar challenges have been addressed through active learning methods, which enable systematic data collection by actively querying the most informative data points [137], and such techniques have been successfully integrated into learning-based control frameworks to improve sample efficiency and model accuracy [138]–[140]. In this context, common information criteria include (i) mutual information, which quantifies the reduction in uncertainty about the unknown dynamics from the observations [141], [142], (ii) entropy, which targets the regions in the state space where the system dynamics are the most uncertain [143] and (iii) criteria based on the Fisher information matrix [144], including D-optimality, which minimizes the volume of the parameter-confidence ellipsoid, and A-optimality, which minimizes the average parameter variance.

In contrast to offline learning, online learning approaches pair closed-loop control with online model adaptation through information acquisition. Although this is not the focus of our discussion, we note that active-learning methods also extend to online settings. Designing controllers that simultaneously achieve closed-loop performance and gen-

Acting under uncertainty. The previous example focused on learning the unknown terrain profile and quantifying its predictive uncertainty. We now close the loop by embedding the learned GP model within an MPC controller. This example illustrates how the resulting dynamics uncertainty can inform control decisions and constraint tightening, enabling a systematic trade-off between closed-loop performance and confidence in constraint satisfaction. The implementation of this example can be found in [122] (GP-MPC).

Building on the terrain model learned in Example $^ { 2 , }$ we introduce the vehicle dynamics and control task; an illustration is shown in Figure 7. The vehicle has state $x = [ p , v ] ^ { \top } \in \mathbb { R } ^ { 2 }$ comprising its horizontal position p and velocity v, and a scalar control input $u \in [ - 8 , 8 ]$ ]. The vehicle dynamics are given by

$$
\dot { p } = v , \dot { v } = u \cos \big ( \alpha ( p ) \big ) - g \sin \big ( \alpha ( p ) \big ) \cos \big ( \alpha ( p ) \big ) ,\tag{15}
$$

where $g$ denotes the gravitational acceleration and $\alpha ( p ) \ =$ arctan $\left( d h ( p ) / d p \right)$ is the local terrain inclination angle. The task is to reach a target $p _ { \mathrm { g o a l } }$ subject to a speed limit $| v | \leq \bar { v } _ { \operatorname* { m a x } } = 0 . 6 .$ The objective function is defined as $\begin{array} { r } { J = \sum _ { k } \dot { 5 } ( p _ { k } - p _ { \mathrm { g o a l } } ) ^ { 2 } + } \end{array}$ $0 . 0 1 u _ { k } ^ { 2 }$ . We formulate the control problem in the form of (14) and solve it in a receding horizon manner. The nominal predictions $\rho ( \cdot )$ are obtained based on the GP posterior mean prediction, while the state covariance $\Lambda ( \cdot )$ is propagated through a linearized covariance recursion under an ancillary feedback controller. The predicted velocity uncertainty is incorporated by tightening the constraint as $| v _ { i | k } | + \beta \sigma _ { i | k } ^ { v } \le v _ { \mathrm { m a x } } ,$ , with $\beta \geq 0$ controlling the degree of constraint tightening. In general, a larger $\beta$ provides a larger margin against model uncertainty, leading to greater confidence in constraint satisfaction.

The bottom-left panel of Figure 7 shows the predicted state uncertainty and the corresponding constraint tightening at $t = 0$ for $\beta = 2$ . The tightened constraint keeps the predicted mean velocity below $v _ { \mathrm { m a x } }$ by a margin proportional to its predicted standard deviation. The grey ellipses show that, without ancillary feedback, the uncertainty grows more rapidly and would therefore require more restrictive constraint tightening. The bottomright panel shows the trade-off between closed-loop cost and

erate informative data for reducing model uncertainty is a central challenge in dual control [145]–[147]. Intuitively, the objective can often be decomposed as $J = J _ { p } + J _ { i } ,$ where $J _ { p }$ is the performance objective (e.g., reference tracking costs) and $J _ { i }$ is an information objective. Minimizing $J _ { i }$ encourages probing control inputs that steer the system along trajectories expected to reduce uncertainty in the learned dynamics. Through a suitable choice of information objective, the controller can balance task performance with informative data collection and thereby improve the learning efficiency.

Many online-learning methods perform model adaptation at each control instant or prediction horizon. However, learning updates can be computationally costly, and adaptation at every step may be unaffordable or unnecessary. Approaches to mitigating this challenge include the use of a datadictionary in combination with sparse GP methods [148], the exploitation of Kalman filters for online parameter updates [77], spatio-temporal GPs [149], and event-triggered learning (ETL) [150]–[153], in which the learning task is performed when a triggering condition indicates that the current model is no longer sufficiently accurate.

empirical constraint violations as $\beta$ is varied. Without constraint tightening $( \beta \ = \ 0 )$ , the nominal controller exceeds the speed limit. Increasing β to 2 mitigates the observed violations at the expense of a higher closed-loop cost. The green points further show that adding training data within the previously unsampled region reduces model uncertainty, allowing the learning-based MPC controller to achieve a lower closed-loop cost at $\beta = 2 ,$ without constraint violations.

![](images/cebad2ce089020ba8d50308801ef12e59cb4d015db505e6af16279bb7f67f0fc.jpg)  
Fig. 7: Acting under model uncertainty with learning-based MPC. The top panel shows the setup for the car terrain traversal example. The bottomleft panel shows the prediction with constraint tightening at $t = 0$ for $\beta = 2 ,$ while the bottom-right panel shows the trade-off between closedloop cost and constraint violation for different β choices.

## V. DIRECT DATA-DRIVEN CONTROL

An alternative approach to first learning the model of the system and then using it in a model-based control scheme is to directly use the data to design the desired controller. This approach is known as direct data-driven control, as it avoids the intermediate step of system identification. Although diverse methods for direct data-driven control have been studied in the literature, most of the recent success in this area is based $_ { 0 \mathrm { n - } } \mathrm { o r }$ at least inspired by—the result that is now known as Willems’ fundamental lemma [154]. In this section, we focus exclusively on describing this approach.

In this section, we use the following notation. Given a sequence $\boldsymbol { u } = \{ u _ { k } \} _ { k = 0 } ^ { N - 1 }$ , we define $\begin{array} { r } { \overline { { u _ { [ a , b ] } } } : = [ u _ { a } ^ { \top } ~ . . . ~ u _ { b } ^ { \top } ] ^ { \top } } \end{array}$ for $0 \leq a \leq b \leq N - 1$ . Moreover, the Hankel matrix of depth L of the sequence is

$$
H _ { L } ( \boldsymbol { u } ) : = \left[ \begin{array} { c c c c } { u _ { 0 } } & { u _ { 1 } } & { \cdot \cdot \cdot } & { u _ { N - L } } \\ { u _ { 1 } } & { u _ { 2 } } & { \cdot \cdot \cdot } & { u _ { N - L + 1 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { u _ { L - 1 } } & { u _ { L } } & { \cdot \cdot } & { u _ { N - 1 } } \end{array} \right] .
$$

## A. Willems’ Fundamental Lemma

Willems’ fundamental lemma describes how every inputoutput trajectory of a controllable discrete-time linear timeinvariant (LTI) system can be characterized in terms of a single persistently excited trajectory measured from such a system. This result was first described using the behavioural approach to systems theory in [154], and has been replicated in the state space framework in [155]–[157]. Willems’ lemma has provided valuable new insights regarding the information contained in measured data, as well as in how to exploit it for system analysis and control design. In particular, a discretetime LTI system of the form

$$
x _ { k + 1 } = A x _ { k } + B u _ { k } ,\tag{16a}
$$

$$
y _ { k } = C x _ { k } + D u _ { k } ,\tag{16b}
$$

are considered<sup>2</sup>. Then, a definition of persistence of excitation was proposed in [154] as follows.

Definition 5 (Persistently exciting sequence): Consider a discrete-time input sequence $u ~ \stackrel {  } { = } ~ \{ \bar { u _ { k } } \} _ { k = 0 } ^ { N - 1 } , ~ u _ { k } ~ \in ~ \mathbb { R } ^ { n _ { \mathrm { u } } }$ The sequence u is persistently exciting (PE) of order L if its Hankel matrix of depth L has full row rank, i.e., rank $( H _ { L } ( u ) ) = n _ { \mathrm { u } } L$

Applying a PE input to a linear system (16) allows one to collect input-output data that can be used to describe every other trajectory admitted by the system. In other words, a data-based system representation is obtained, as presented in the following theorem.

Theorem 4 (Willems’ fundamental lemma [154]): Let system (16) be controllable and apply an input sequence $\dot { u } = \{ u _ { k } ^ { d } \} _ { k = 0 } ^ { N - 1 }$ that is persistently exciting of order $L + n _ { \mathrm { x } }$ Let $y = \{ y _ { k } ^ { d } \} _ { k = 0 } ^ { N - 1 }$ be the measured output trajectory of (16). A pair of sequences $\{ \bar { u } _ { k } \} _ { k = 0 } ^ { L - 1 } , \ \{ \bar { y } _ { k } \} _ { k = 0 } ^ { L - 1 }$ is an input-output trajectory of (16) if and only if the expression

$$
\left[ { H } _ { L } ( u ^ { d } ) \right] \alpha = \left[ { \bar { u } } \right]\tag{17}
$$

holds for some vector $\begin{array} { r l r } { \alpha } & { { } \in } & { \mathbb { R } ^ { N - L + 1 } } \end{array}$ , with $\bar { u } \quad = \quad$ $[ \bar { u } _ { 0 } ^ { \top } \cdot \cdot \cdot \bar { u } _ { L - 1 } ^ { \top } ] ^ { \top }$ and $\bar { y } = [ \bar { y } _ { 0 } ^ { \top } \cdot \cdot \cdot \bar { y } _ { L - 1 } ^ { \top } ] ^ { \top }$

Now, (17) is a data-based representation of system (16) in the sense that any trajectory u,¯ y¯ of length L that can be generated by (16) can also be expressed in terms of the measured data $u ^ { d } , y ^ { d }$ , by finding a suitable vector α. System (16) being controllable, the sufficient condition in this theorem is the application of an input that is PE of sufficiently large order as in Definition 5. This excitation condition guarantees that the measured data is rich enough in information about the system dynamics and can be used, for example, for control design as described in the subsequent subsections. This fact has raised the question of whether persistence of excitation is a necessary condition for datadriven control. The answer to this question is in fact negative, and the study of sufficient and necessary conditions for datadriven system analysis and control has established the area of research of data informativity [158].

As described above, Willems’ lemma was originally developed for discrete-time systems. In recent years, different approaches have been followed to obtain similar data-based representations for continuous-time systems. In [159], an expression similar to (17) was used to express continuoustime input-output trajectories, but using time-varying Hankel matrices, and where the vector α satisfies a state-dependent differential equation. Since the use of state information is undesirable, in [160] an alternative system representation was obtained by exploiting input-output trajectories and their derivatives. A different approach was followed in [161], where orthogonal bases are used to obtain approximate representations of the continuous-time trajectories. More recently, [162] developed a system representation by computing Gramians of the measured data. Although we point out below some works that have used these results for databased control of continuous-time systems, we focus most of the following exposition on the discrete-time case.

Before presenting data-driven control strategies, in the following subsection we briefly describe some of the extensions of the fundamental lemma that have been developed to allow its application to a wider class of dynamical systems or that relax the requirements on the data.

## B. Extensions to the Fundamental Lemma

Willems’ fundamental lemma requires the application of a persistently exciting input to a controllable system in order to obtain the data-based representation in (17). However, in practice it may not always be possible to arbitrarily excite a system due to, for example, safety constraints. Moreover, this approach excludes considering uncontrollable systems that may be of interest in certain applications. In [163], a generalized persistence of excitation condition for the data was presented, consisting of the rank condition

$$
\mathrm { r a n k } \left( \left[ { \frac { H _ { L } ( u ^ { d } ) } { H _ { L } ( y ^ { d } ) } } \right] \right) = n _ { \mathrm { u } } L + n _ { \mathrm { x } }\tag{18}
$$

with $L > l ,$ and where l is the lag, or observability index, of (16). It is shown in [163] that (17) is a data-based representation of system (16) if and only if (18) holds.

Another important extension of Theorem 4 is relaxing the requirement of collecting a single, long enough PE trajectory from the system. In some applications, it may be inconvenient to collect long trajectories from (16), for example if the open-loop system is unstable. Moreover, the user may face intermittent loss of data. In [156], the fundamental lemma is extended to the case where multiple trajectories are available. First, the concept of collective persistence of excitation was presented by considering q different input sequences, $u ^ { d , i } \stackrel { \cdot } { = } \{ u _ { k } ^ { d , i } \} _ { k = 0 } ^ { N _ { i } - 1 } , i \stackrel { \cdot } { = } 1 , \ldots , \stackrel { \cdot } { q }$ , selected such that rank $( [ H _ { L } ( u ^ { d , 1 } ) \stackrel {  } { H _ { L } ( u ^ { d , 2 } ) } \cdot \cdot \cdot H _ { L } ( u ^ { d , q } ) ] ) = n _ { \mathrm { u } } L$ . To satisfy this rank condition, it is not necessary that any one of the input sequences is PE as in Definition 5. Using the data collected from the system, the data-based representation now becomes

$$
\begin{array} { r l } { \left[ H _ { L } ( u ^ { d , 1 } ) \quad H _ { L } ( u ^ { d , 2 } ) } & { \cdots \quad H _ { L } ( u ^ { d , q } ) \right] \boldsymbol \alpha = \left[ \bar { u } \right] . } \\ { \left[ H _ { L } ( y ^ { d , 1 } ) \quad H _ { L } ( y ^ { d , 2 } ) \quad \cdots \quad H _ { L } ( y ^ { d , q } ) \right] \boldsymbol \alpha = \left[ \bar { y } \right] . } \end{array}\tag{19}
$$

Besides the LTI systems described above, the fundamental lemma has also been extended to other classes of dynamical systems including linear time-varying systems [164], linear parameter-varying systems [165], descriptor systems [166], and multiagent systems [167]. Stochastic systems were considered in [168], and representations from noisy data were described in [169]. In [170], the case when only irregularly measured data is available is studied. Links between the fundamental lemma and kernel regression were investigated in [171].

Data-based representations for classes of nonlinear systems have also been obtained [157], [172]–[175]. These results often exploit knowledge about the nonlinear terms in the system model, for example in the form of a dictionary of basis functions, and then they take advantage of the known results for linear systems. For example, Hankel matrices are used as in (17), but with nonlinearities in their entries.

## C. Data-Based Control Design

In the following subsections, we briefly describe some basic approaches for direct data-driven control in the existing literature. Many more works and extensions than those presented here can be found in the surveys [18], [158] as well as the book [176].

Most of the control design methods in the literature consider state-feedback controllers. Hence, we begin by considering a linear system represented only by the equation (16a), such that the state x can be measured. Later, we describe some results that do not require this assumption. Hence, assume throughout this subsection that system (16a) is controllable. After collecting N + 1 samples of input and state data, $\{ u _ { k } ^ { d } \} _ { k = 0 } ^ { N } , \ \{ x _ { k } ^ { d } \} _ { k = 0 } ^ { N } ,$ define the matrices U, X, and $X ^ { + }$ in (5a), (5b), and (6), respectively. Notice that, by definition, the relation $X ^ { + } = A X + B U$ holds. These matrices were used in [155] to formulate a simple data-based method to design stabilizing controllers using the following linear matrix inequality (LMI)

$$
\left[ \begin{array} { c c } { X \Theta } & { X ^ { + } \Theta } \\ { ( X ^ { + } \Theta ) ^ { \top } } & { X \Theta } \end{array} \right] \succ 0 .\tag{20}
$$

The objective in (20) is to find any $\boldsymbol { \Theta } \in \mathbb { R } ^ { N \times n _ { \operatorname* { m } _ { \operatorname* { m } } } }$ <sup>x</sup> that satisfies the LMI. Using any such matrix, a state-feedback gain can be computed as

$$
K = U \Theta \left( X \Theta \right) ^ { - 1 } .\tag{21}
$$

It is shown in [155] that, if the data satisfy the rank condition

$$
{ \mathrm { r a n k } } \left( { \mathord { \left[ { \begin{array} { l } { U } \\ { X } \end{array} } \right] } } \right) = n _ { \mathrm { u } } + n _ { \mathrm { x } } ,\tag{22}
$$

then the control input $\ u \ : = \ : K \ d { x }$ with K as in (21), and where Θ satisfies (20), stabilizes the system (16a). That is, the closed-loop matrix A+BK is Schur stable, meaning that all its eigenvalues are inside the unit circle. Conversely, any stabilizing feedback gain K for system (16a) can be written as (21), where Θ satisfies (20). These facts can be seen by noticing that (20) implies that XΘ is symmetric and positive definite, and hence we can define $P : = X \Theta$ . Moreover, notice that $X ^ { + } \Theta = ( A X + B U ) \Theta = ( A + B K ) P$ , where we used $U \Theta = K \mathcal { I }$ XΘ from (21). By the Schur complement, (20) also implies $X \Theta - X ^ { + } \Theta ( X \Theta ) ^ { - 1 } ( X ^ { + } \Theta ) ^ { \top } \succ \bar { 0 } .$ , which yields the Lyapunov condition $( A { + } B K ) P ( A { + } B K ) ^ { \top } { - } P \prec$ 0 that shows stability. Note that computing a stabilizing controller from (20)-(21) does not require any knowledge of the system model. The condition in (22) can be guaranteed by applying a PE input of order $n _ { \mathrm { x } } + 1$ to the system, according to Willems’ fundamental lemma [154], [155].

More sophisticated data-based control methods have been developed, considering requirements for robustness, optimality, and other performance characteristics. The case in which the measured data is corrupted by noise has been addressed with the development of robust controllers [177]– [179]. Different approaches to solve the data-based optimal control problem have been investigated [155], [180], [181]. A model-reference controller was proposed in [182] for the system to emulate the behavior of a desired model. The pole placement problem was considered in [183], [184]. The data-based control of continuous-time systems has also been investigated [185]–[188]. Of current interest is the extension of these methods to (classes of) nonlinear systems. In this regard, diverse results have already been obtained using different approaches [189]–[194].

We finally comment on control design results that use only input-output data, and no state information is required. Already in [155] it was noted that an alternative, nonminimal state for a linear system of the form (16) can be constructed using past measurements of input-output data as $\begin{array} { r l } { \xi _ { k } } & { { } = } \end{array}$ $[ u _ { [ k - n _ { \mathrm { x } } , k - 1 ] } ^ { \top } \quad y _ { [ k - n _ { \mathrm { x } } , k - 1 ] } ^ { \top } ] ^ { \top } \in \mathring { \mathbb { R } } ^ { ( n _ { \mathrm { u } } + n _ { \mathrm { y } } ) n _ { \mathrm { x } } }$ . Although $\xi _ { k }$ is a valid state of the system, its use for data-based control is limited to systems for which the relation $n _ { \mathrm { y } } l = n _ { \mathrm { x } }$ holds [179], [195]. However, for general linear systems it holds that $n _ { \mathrm { y } } l \geq n _ { \mathrm { x } }$ and, if the strict equality does not hold, the rank condition (similar to (22)) that is used to obtain guarantees for the controller is not satisfied. In [195], this problem is solved by instead defining the state

$$
z _ { k } = \left[ \Gamma ^ { u _ { [ k - l , k - 1 ] } } \right] \in \mathbb { R } ^ { n _ { \mathrm { u } } l + n _ { \mathrm { x } } } ,\tag{23}
$$

where the matrix Γ can be computed from input-output data. It is shown that $z _ { k }$ is also a valid state of system (16), and it can be used for control design of general controllable linear systems. Indeed, any of the data-based strategies that have been developed for state-feedback control can be used to obtain a (dynamic) output-feedback controller of the form $u _ { k } = K z _ { k }$ . Other methods for data-based output-feedback control have also been proposed [196], [197]. Besides control design, data-driven approaches for the dual problem of state estimation have been developed as well [198]–[201].

## D. Data-Driven Predictive Control

One particular direct data-driven control method that has received significant attention in the literature is data-driven predictive control (DDPC). In DDPC, at each time t, given the past $T _ { \mathrm { i n i } }$ inputs $u _ { [ t - T _ { \mathrm { i n i } } , t - 1 ] }$ and outputs $y _ { [ t - T _ { \mathrm { i n i } } , t - 1 ] }$ with

$T _ { \mathrm { i n i } } \geq l ,$ the following optimization problem is solved:

$$
\operatorname* { m i n i m i z e } _ { \bar { u } ( t ) , \bar { y } ( t ) , \alpha ( t ) } \sum _ { k = 0 } ^ { L - 1 } \ell ( \bar { u } _ { k } ( t ) , \bar { y } _ { k } ( t ) )\tag{24a}
$$

$$
\mathrm { s . t . } \quad [ \begin{array} { l } { H _ { L + T _ { \mathrm { i n i } } } ( u ^ { d } ) \Big ] } \\ { H _ { L + T _ { \mathrm { i n i } } } ( y ^ { d } ) \Big ] } \end{array} \alpha ( t ) = [ \begin{array} { l } { \bar { u } ( t ) } \\ { \bar { y } ( t ) \Big ] } \end{array} ]\tag{24b}
$$

$$
\left[ \bar { u } _ { [ - T _ { \mathrm { i n i } } , - 1 ] } ( t ) \right] = \left[ \begin{array} { c } { { \bar { u } _ { [ - T _ { \mathrm { i n i } } , t - 1 ] } } } \\ { { \bar { y } _ { [ - T _ { \mathrm { i n i } } , - 1 ] } ( t ) } } \end{array} \right]\tag{24c}
$$

$$
\bar { u } _ { k } ( t ) \in \mathbb { U } , \ \bar { y } _ { k } ( t ) \in \mathbb { Y } , \ 0 \leq k \leq L - 1 .\tag{24d}
$$

In problem (24), the decision variables u¯(t) and $\bar { y } ( t )$ correspond to the predicted input and output sequences, predicted at time t. These are of length $L + T _ { \mathrm { i n i } }$ and the first $T _ { \mathrm { i n i } }$ elements of these sequences are fixed to the past $T _ { \mathrm { i n i } }$ inputs and outputs via the constraint (24c). This ensures that the predicted future output sequence is correctly initialized according to the current state of the system. Constraint (24b) replaces the system model that is typically used in classical, model-based MPC and uses the data-driven description (17) from the fundamental lemma instead. Moreover, (24d) specifies the input and output constraints that the system should satisfy. Finally, the stage cost ℓ in (24a) is typically chosen as the classical quadratic cost function in the DDPC literature, i.e.,

$$
\ell ( u , y ) = ( u - u ^ { s } ) ^ { \top } R ( u - u ^ { s } ) + ( y - y ^ { s } ) ^ { \top } Q ( y - y ^ { s } )
$$

with $R ~ > ~ 0$ and $Q \ \geq \ 0$ and $( u ^ { s } , y ^ { s } )$ being a desired input/output setpoint to be stabilized, but also other cost functions can be used depending on the desired control objective. As usual in predictive control, problem (24) is solved and then the first element of the optimal input sequence $\bar { u } ^ { * } ( t )$ is applied to the system, i.e., $u _ { t } : = \bar { u } _ { 0 } ( t )$ before the procedure is repeated at the next time instant t+1.

DDPC based on problem (24) has been introduced under the name data-enabled predictive control (DeePC) in [202] and since then, many properties, extensions and related DDPC variants have been studied in the literature. In case of noisy measurements (both offline and online), one cannot exactly predict future trajectories via (24b) and fixing the initial condition via (24c) might be inconsistent. To this end, one typically employs additional slack variables together with suitable regularization terms in (24). In particular, y¯(t) in (24b) is replaced by $\bar { y } ( t ) + \sigma ( t )$ , where the slack variable $\sigma ( t )$ ensures that the constraints (24b)–(24c) are feasible despite noisy data. Moreover, regularization terms $g _ { 1 } ( \sigma ( t ) ) ,$ and $g _ { 2 } ( \alpha ( t ) )$ are added to the cost function (24a). Typical choices are $\ell _ { 1 }$ and $\ell _ { 2 }$ regularizations, as proposed, $\mathrm { e . g . }$ in [202], [203], but also other choices have been used. These regularization terms robustify the DDPC scheme and avoid overfitting to the noisy data. A detailed study of different regularizations as well as further interpretations of their effects can be found in [204]–[207]. Further stochastic DDPC schemes for settings including disturbances in the dynamics and noisy data have been proposed in, e.g., [208]–[210].

Another aspect that has been studied in the DDPC literature is sample and computational efficiency. Namely, the standard formulation (24) is such that the required minimum number of offline data points $u ^ { d }$ and $y ^ { d }$ is given by $N \geq$ $( n _ { \mathrm { u } } + 1 ) ( L + T _ { \mathrm { i n i } } + n _ { \mathrm { x } } ) - 1$ in order to satisfy the PE condition in Theorem 4, scaling linearly with the prediction horizon L. Moreover, the dimension of $\alpha ( t )$ scales linearly with the number N of collected data points. To address these issues, different DDPC variants have been proposed that modify the data-driven prediction model (24b) to reduce the amount of required offline data and/or the number of optimization variables of problem (24), see, e.g., [211]–[213].

In order to study closed-loop properties of DDPC such as stability, safety, and robustness, one can conceptually use the same tools as in classical model-based MPC, in particular adding suitable terminal constraints or using a long enough prediction horizon L. On the other hand, the details in deriving these closed-loop properties are quite different than in model-based MPC, in particular in case of noisy (offline and online) data where the prediction model (24b) is not exact anymore. The first results on closed-loop stability and robustness have appeared in [203]; a recent review on available methods and systems-theoretic guarantees in DDPC can be found in [214].

One major challenge and currently very active line of research is to develop DDPC approaches for nonlinear systems. Existing works exploit extensions of Willems’ lemma for specific system classes and/or knowledge of a set of suitable basis functions such as [215]–[218], or use online data in the Hankel matrices such that the prediction model (24b) serves as a local affine approximation of the nonlinear system [219].

## VI. REINFORCEMENT LEARNING, IMITATION LEARNING, AND BAYESIAN OPTIMIZATION

This section focuses on learning-based control methods that directly learn a policy (2) from data, often through iterative interaction with the environment, without first identifying a model of the system dynamics (Section IV) or leveraging data-driven control techniques building on Willems fundamental lemma (Section V). Typically, these methods employ a parametrized policy $\pi _ { \boldsymbol { \theta } } ( \cdot )$ , where the parameters θ may belong to a function approximator (often a neural network) or to a classical controller structure such as PID, LQR, or MPC. At a high level, learning a policy (2) to optimize performance (3) can then be formulated as a parametric optimization problem

$$
\operatorname* { m i n } _ { \theta } J ( \theta ) .\tag{25}
$$

Depending on the method and the available prior knowledge, $\theta$ can differ substantially in both dimension and interpretation. When θ are the parameters of a high-capacity function approximator such as a neural network, (25) represents a general, expressive problem formulation, capable of representing a wide class of policies without imposing strong structural assumptions on π. For this class of problems, we focus on state-of-the-art RL and imitation learning approaches. In

Controlling without a model. Willems’ fundamental lemma allows us to work with collected data instead of a parametric model. In this didactic example, this is illustrated by using the standard DDPC algorithm based on (24) and examining how data length and regularization affect closed-loop performance in non-ideal settings. However, note that as explained above, other DDPC variants have been developed to specifically address nonlinear systems. The implemented framework can be found in [122] (DeePC).

We consider the terrain setup as in Example 2 and Example 3 and assume that we have access to position measurements only $( \mathrm { i . e . , } y = p )$ . We solve the goal-reaching problem with $\ell ( u , y ) =$ $( y - \check { p } ^ { \mathrm { s } } ) ^ { 2 } + 0 . 0 2 u ^ { 2 } , L = \bar { 2 0 }$ and $T _ { \mathrm { i n i } } = \bar { 4 } .$ To collect the offline data, a uniformly random input of length N is applied.

In the ideal flat-terrain case, the system is LTI, with the datalength bound $N ~ \geq ~ 5 1$ required to satisfy the PE condition from Theorem 4. Figure 8 shows that (i) with a much smaller dataset, problem (24) is infeasible, and (ii) once the bound is reached, the closed-loop behaviour coincides with that of MPC using the exact model. Further increasing the data length beyond the theoretic bound does not result in better performance.

![](images/8ba5aaea6230444a52223cbec4b00aab270abc8d3b5804f2a342caa637405b46.jpg)  
Fig. 8: DeePC on flat terrain, where LTI assumption holds. Left: Deviation of the DeePC controller from the full-state MPC versus data length $N ,$ where the shaded region corresponds to guaranteed PE. Right: trajectories for the circled N cases.

We then consider bumpy terrain with measurement noise, where the LTI assumption no longer holds. The top panels

particular, we discuss how to address key challenges for realworld deployment, including safety, robustness, stability, and sample efficiency.

When instead specifying a fixed controller structure as named above, prior knowledge about the system or desired closed-loop behaviour is typically built directly into π, and θ is correspondingly lower dimensional and often (more) interpretable. In this case, problem (25) is also referred to as controller tuning. Here, we focus on Bayesian optimization methods for safely optimizing controller parameters directly from input-output data collected from the system.

Table I (bottom) provides an overview of the methods discussed in this section and their selected main characteristics.

## A. Model-Free Reinforcement Learning

RL studies how an agent can learn to make goal-directed sequential decisions through interaction with an environment. In the context of the learning-based control problem from Section II, the agent can be viewed as an adaptive of Figure 9 show that more data are needed to stabilize the system, while a nonzero error remains due to the non-ideal setting. The bottom panels of Figure 9 show how regularization balances fitting the measured history against biasing the predictor; more information about the effects and different types of regularization can be found in [204]–[207]. With suitable tuning, DeePC approaches the performance of MPC based on the imperfect flatterrain model.

![](images/2f2f9357882d9ad8d577270a07df6b0ad0f3d415e6d112ff9d6ee7847c57bb39.jpg)

![](images/80ec737f054b05a80647696ccb02cc16782f547bc866edb9d6dd12c0044a21d5.jpg)

![](images/9afef11465b87630a4da58186344920798a4e98d5b6fd0837e42dd9155879dda.jpg)

![](images/6f58a0e7656d0ecc212c2e40deec2305806b477198ba2f8409d6ee422f3f0daa.jpg)  
Fig. 9: DeePC with bumpy terrain and measurement noise. Top-Left: Deviation from the full-state MPC (median and 10-90 % range computed over ten trials). Top-Right: trajectories for the circled N cases. Bottom-Left: Deviation from the full-state MPC with the bumpy-terrain model for different regularization parameters $\lambda _ { \alpha }$ and $\lambda _ { \sigma } ,$ against the MPC with the flat-terrain model. Bottom-Right: trajectories for the circled settings.

controller that learns a policy (2) from the interaction data, which then determines the control action. The environment encompasses the controlled system together with all relevant external factors. Formally, the environment is often modeled as a Markov decision process (MDP) $( S , \mathcal { A } , p , r , \gamma )$ , where S is the state space, A the action space, $p ( x _ { k + 1 } \mid x _ { k } , u _ { k } )$ the transition dynamics, $r ( \boldsymbol { x } _ { k } , \boldsymbol { u } _ { k } )$ the reward function, and $\gamma ~ \in ~ [ 0 , 1 )$ a discount factor. Even though the notation differs slightly and we maximize rewards instead of minimizing costs, the MDP formulation and the state-space formulation (1a) are equivalent. The MDP transition kernel $p ( x _ { k + 1 } | x _ { k } , u _ { k } )$ is the conditional distribution over successor states induced by the dynamics f and disturbances w<sub>k</sub>. The goal is to learn a policy $\pi ( u _ { k } | x _ { k } )$ that maximizes the expected cumulative reward

$$
\operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \left[ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r ( x _ { k } , u _ { k } ) \right] ,\tag{26}
$$

where π is a conditional distribution and $\mathbb { E } _ { \pi }$ denotes the expectation under the policy π for generating actions.

Classical RL approaches are mostly focused on tabular settings, i.e., with finite and discrete state and action spaces. For control-oriented tasks, we require continuous state-action spaces and thus, focus on modern deep RL, where the policy π is usually represented by a deep neural network, which leads to the notation $\pi _ { \theta } .$ , where θ are the weights of the neural network. We can now identify our RL objective (26) with the general formulation (25):

$$
\operatorname* { m i n } _ { \theta } J ( \theta ) = - \operatorname* { m a x } _ { \theta } \mathbb { E } _ { \pi _ { \theta } } \left[ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r ( x _ { k } , u _ { k } ) \right] .\tag{27}
$$

The basic idea of RL is to learn a competent control policy by interacting with the environment: repeatedly trying out control inputs (actions) and observing the resulting state transitions and incurred costs (rewards). The data is thus typically composed of state (5b), input (5a), and reward data. Different categories of RL emerge depending on what objects are learned and how the search for an optimal policy is structured. We first consider model-free RL, i.e., methods that do not construct a model of the system dynamics as part of solving the RL problem. The main categories within model-free RL are (i) policy-based, (ii) value-function-based, and (iii) actor-critic-based approaches, which we discuss next. Afterwards, we discuss why learning a transition model alongside the policy can nonetheless be beneficial for RL.

Policy-based approaches are the closest to our optimiza tion objective (26) and yield iterative updates of the type

$$
\theta _ { k + 1 } = \theta _ { k } - \alpha { \widehat { \nabla J ( \theta _ { k } } } ) ,\tag{28}
$$

where α is the learning rate or step size and $\widehat { \nabla J ( \theta _ { k } ) }$ is an estimate of the gradient of J with respect to θ. Clearly, the challenge lies in estimating $\widehat { \nabla J } ( \widehat { \theta _ { k } } )$ from data. The policy gradient theorem provides the theoretical basis for this estimation: it expresses the gradient of the expected return as an expectation over trajectories sampled from the current policy, involving only the gradients of the logpolicy (i.e., the log probabilities of the chosen actions) rather than the unknown system dynamics, thus allowing the gradient to be estimated purely from sampled data. A didactic overview including the canonical policy gradient algorithm REINFORCE is provided in [220, Sec.13].

Value-based methods center around the action value function

$$
Q _ { \pi } ( x _ { k } , u _ { k } ) = \mathbb { E } _ { \pi } \left[ \sum _ { l = 0 } ^ { \infty } \gamma ^ { l } r ( x _ { k + l } , u _ { k + l } ) \ \middle | \ x _ { k } , u _ { k } \right] .\tag{29}
$$

Intuitively, the Q-function quantifies the value of state-action pairs. The size of the action space is usually very limited. The famous deep Q-network (DQN) paper [221] varies the valid actions between 4 and 18 depending on the specific Atari game that is considered. The optimal policy can be obtained via $\pi ^ { * } ( x _ { k } ) = \arg \operatorname* { m a x } _ { u } Q ^ { * } ( x _ { k } , u )$ . Modern methods learn the Q-function via deep learning approaches to scale to large state-spaces. Accordingly, in pure value-based methods, we consider the parameters $\theta _ { Q }$ that are used to learn the $Q -$ function and not a policy.

Actor-critic methods combine policy gradients with valuebased methods, leveraging strengths of both worlds. They typically maintain an estimate of the policy π (the actor) with parameters θ and a value function (the critic) with other parameters $\theta _ { Q }$ . Intuitively, the value estimate leads to normalized learning signals that usually result in more stable and efficient algorithms than pure policy-gradient methods. Notable examples include asynchronous advantage actorcritic (A3C) [222], trust-region policy optimization (TRPO) [223], proximal policy optimization (PPO) [224], and twindelayed deep deterministic policy gradient (TD3) [225].

Actor-critic methods are among the model-free RL approaches that are most suitable for control problems and require very little prior knowledge. At the same time, they are typically extremely data hungry and come without any safety guarantees. Data efficiency is a main motivation for model-based RL, which is discussed next.

The above mentioned RL approaches generally assume that the learning agent can actively sample the environment, i.e., query it by applying actions and observing the resulting next state and reward. In this sense, they are sometimes referred to as online RL. The environment may be either a simulator or a physical system, such as a robot. In robotics, a common approach is to learn a policy in simulation during the design phase and subsequently deploy it on physical hardware, which often requires techniques such as domain randomization and sim-to-real adaptation to bridge the gap between simulated and real environments [226]. Some approaches instead learn directly from hardware interaction [227] or even adapt the policy during deployment [228]. In contrast, offline RL considers the setting in which the agent learns solely from a fixed dataset of previously collected interactions, without actively querying the environment. This is particularly relevant when interaction with the environment is undesirable or impractical, as it enables learning from historical data alone. A key challenge in offline RL is distribution shift: the learned policy may encounter state-action pairs that are poorly represented in the fixed dataset [229].

## B. Model-Based Reinforcement Learning

At the core of model-based RL is a dynamics model that predicts future state transitions and enables planning and policy improvement. This model $\hat { p } _ { \theta _ { M } } ( x _ { k + 1 } | x _ { k } , u _ { k } )$ is typically learned from data and does not need to be known a priori. Clearly, this adds the parameters $\theta _ { M }$ to the overall optimization problem.

Next, we distinguish between two major lines of research in model-based RL: (i) Dyna-style or background planning and (ii) decision-time planning. Dyna-style methods primarily use the model to improve learning efficiency by generating additional training data. Modern methods utilize ensembles of probabilistic networks to learn the transition

![](images/832fb947fcf02cb1d151a1a7f7754993c3ec903d1e9be950c286faa076601b6f.jpg)

Reinforcement learning. We consider the standard cart-pole system, where the objective is to swing the pole into the upright position and balance it there. We use a DQN agent [221], adapted from [230] (REINFORCE on CartPole). In RL, the control objective is formalized as the reward. In this case, the reward function is defined as $\begin{array} { r } { r ( x ) = \frac { 1 + \cos \theta } { 2 } \cdot \big ( 1 - \big ( \frac { p } { p _ { \mathrm { b o u n d } } } \big ) ^ { 2 } \big ) \cdot \frac { 1 + 1 0 ^ { - ( \dot { \theta } / 5 ) ^ { 2 } } } { 2 } } \end{array}$ where p denotes the cart position, θ the pole angle, and $\dot { \theta }$ the angular velocity. That is, the reward expresses the control goal of swinging the pole upright, keeping the cart within its position bounds, and minimizing angular velocity to stabilize the pole. An episode terminates when the cart position leaves the interval [−p<sub>bound</sub>, p<sub>bound</sub>]. The control input is discrete and consists of applying a fixed force to the left, applying a fixed force to the right, or applying no force.

Figure 10 illustrates several characteristic properties of RL applied to a control task. After training, the learned policy achieves a fast swing-up and successfully balances the pole near the upright position. In contrast to supervised learning, the performance during training is highly stochastic and can temporarily deteriorate. This is partly a consequence of the ϵ-greedy exploration strategy used by the DQN agent: with probability $\epsilon ,$ the agent selects a random action rather than the action predicted by the current policy. Although such exploratory actions can intermediately lead to poor trajectories and low rewards, exploration is essential for discovering potentially better control actions. A further characteristic of RL is its reliance on repeated interactions with the environment. The agent must collect a large number of transitions through trial and error before it can learn a good approximation of the Q-function. The neural network parameters are subsequently optimized based on these interactions.

DQN [221] was one of the first highly successful deep RL algorithms, though limited to discrete actions, and it triggered the development of many follow-up methods, including approaches for continuous action spaces. Further implementations of tabular and deep RL algorithms can be found in [230]. Additional material and a comparison of PPO and MPC on a mountain car example can be found in [122] (PPO).

Although specific reward design can encourage safe behaviour, standard RL does not inherently guarantee satisfaction of constraints. Safety filters can address this limitation by minimally modifying the agent’s potentially unsafe actions to enable safe exploration during training [37], [231]–[234]. To illustrate this approach, we impose the cart-position constraint $p \in [ - 1 , 2 . 4 ]$ and incorporate a CBF-based safety filter [232] into the DQN

probabilities [235] and consider model uncertainty for improved rollouts [236], [237]. Decision-time planning methods leverage the model to improve action selection through online look-ahead search. Conceptually, decision-time planning can be viewed as an instance of MPC [238].

While both approaches can yield gains in sample efficiency, the model serves fundamentally different roles: in the former, it supports policy learning, while in the latter, it directly guides decision making on the next action $u _ { k }$ . In traditional control, dynamics models are typically grounded in physical principles and used to establish theoretical guarantees, such as stability and safety. In model-based RL, by contrast, models are primarily learned or employed to improve sample efficiency. Despite these distinct motivations, the two paradigms exhibit strong conceptual and methodagent’s interactions with the environment throughout training. An example implementation of the safety filter is provided in [122] (Safe RL). As shown in Figure 11, the filtered agent satisfies the constraint and achieves high episode returns with fewer environment interactions, illustrating how incorporating safety into exploration can also improve learning efficiency.

![](images/0a12e5671f2261a657564d062c3ffbf09a49b8e6ae2d287431d7ebaa2d24cf80.jpg)

![](images/b870ebbf89390a81486854fc85ed2e43ad6377d15075994b41eda67a89c7e7df.jpg)  
Fig. 10: Example behaviour of a DQN agent learning to swing up and balance a pole on a cart-pole system. The left plot shows two trajectories at different stages of the learning process, while the right plot shows the corresponding learning curve. The solid line represents the mean performance across runs, while the shaded area indicates the corresponding standard deviation.  
Fig. 11: Comparison of DQN with safety filter (green) and vanilla DQN (grey). The left plot shows representative pole-tip trajectories from the two agents, with the red shaded regions indicating forbidden cart positions. The right plot shows the corresponding learning curves. In this example, the safety filter enforces the safety constraints while improving learning efficiency, enabling the agent to achieve high episode returns with fewer environment interactions.

ological overlap, enabling fruitful cross-pollination and the transfer of advances between the fields [239], [240].

In particular, access to a model and the integration of control-theoretic techniques (see Section III) provide a means to incorporate different types of safety guarantees in RL. Depending on the specific notion of safety considered, representative approaches include the use of soft constraints [241], [242], Lyapunov functions [243], barrier-functionbased methods [244], and safety filters that modify or restrict policy actions to ensure constraint satisfaction [233], [234], [245].

For the above methods, we consider full state observations, i.e., the true state $x _ { k }$ is assumed to be measured and provided to the policy. In practice, however, one often feeds all available measurements into the policy without explicit knowledge of whether they constitute the full state, and may additionally include historical observations to capture temporal dependencies. Beyond this, there is work that looks at latent state space models (sometimes also called world models) considering output measurements (1b). This type of approach is particularly relevant when acting on image data and gave rise to the Dreamer architecture [246] and other influential methods such as temporal-difference MPC (TD-MPC) [247]. Safety remains an important consideration in this setting as well [248].

## C. Imitation Learning

The previous approaches focus on discovering optimal behaviour through trial-and-error via the reward signal. Instead, we can learn a policy by observing and mimicking an expert’s behaviour, which results in a supervised learning problem.

This expert could, for example, be a human. Behaviour cloning and inverse RL form two major classes of imitation learning methods [249]. In behaviour cloning, we leverage a dataset of state-action pairs $\mathcal { D } = \{ ( x _ { k } , u _ { k } ) \}$ , sometimes also with an additional context variable. We cast the problem as a supervised learning problem and approximate the policy π that has created the control inputs via a neural network $\pi _ { \theta }$ Inverse RL, on the other hand, learns the underlying reward function, and we obtain $R _ { \theta _ { R } }$ , which can afterwards be used to find an optimal policy. Current active topics, especially in robotics, are large behaviour models [250], vision-languageaction models such as $\pi _ { 0 }$ [251], and foundation models in general. A core component of training such models is imitation learning, often in combination with RL. The number of parameters of the underlying transformer models ranges easily in the magnitude of billions.

As an alternative to learning from humans, the expert could also be a machine or an expensive-to-evaluate policy, such as a sophisticated modern MPC. Essentially, the MPC is used to create state-input data offline, which can be used as training data for a neural network to approximate the MPC via supervised learning. The imitation learning policy is often much faster to evaluate during deployment than the original MPC and with constant runtime, which are core motivations for this approach in practice. This approach of approximate MPC (see [252] for a recent survey) gives rise to research questions on constraint satisfaction despite approximation errors [253]–[257], jointly optimizing the MPC and neural network policy [258], [259], how to approximate set-valued solution maps with generative models [260]–[263], adapting policies to parameter changes [264], [265], or even learning iterative solvers [266], to name just a few. Despite great theoretical efforts, there are only a few applications in hardware, for example, in power electronics [267], robotics [261], [263], [268], [269], or automotive production-grade electronic control units [255].

## D. Controller Tuning with Bayesian Optimization

In many practical applications, established controller structures such as PID controllers, LQR, or MPC with predefined cost formulations are often preferred due to their interpretability, robustness, or ease of implementation. However, even when the controller structure is fixed, its parameters must still be selected to achieve good performance on the actual system. This process is commonly referred to as controller tuning. At a high level, controller tuning coincides with optimization problem (25), where the goal is to identify parameter values that optimize a given performance objective. A key distinction, however, is that the parameter space in controller tuning is typically much lowerdimensional than the policy parameter spaces considered in deep RL. In many cases, tuning involves only a few to several tens of parameters. This lower dimensionality enables the use of specialized optimization methods that often require substantially less data and may provide stronger theoretical or practical guarantees compared with general-purpose deep RL approaches.

Controller tuning is a long-standing problem and many methods exist, including PID auto-tuners [270], extremum seeking control [271], virtual reference feedback control [272], iterative feedback tuning [273], and evolutionary search [274], among many others. Here, we focus on BO [275], which has gathered substantial momentum over the last decade and matured into a powerful framework for controller tuning [19], [276]. Its sample efficiency and blackbox formulation make it particularly attractive for tuning directly on physical systems, where evaluations can be costly and accurate simulation models may be unavailable.

At the heart of BO for controller tuning lies the optimization problem (25). We assume that the function J is unknown, can be evaluated through closed-loop experiments, and represents a control performance measure that depends on controller parameters θ, which are typically lowdimensional. To guide the search for optimal parameters, BO constructs a probabilistic surrogate model of $J ,$ usually in the form of a Gaussian process, $J \sim \mathcal { G P } ( m ( \theta ) , k ( \theta , \theta ^ { \prime } ) )$ where $m ( \cdot )$ denotes the mean and $k ( \cdot , \cdot )$ the covariance function (cf. Section IV-D). Alternatively, other probabilistic machine learning models can be used as surrogates, such as neural networks [277]. Based on the posterior predictions of the surrogate model, an acquisition function $\alpha ( \theta , { \mathcal { D } } )$ where D contains the past evaluations, determines which controller parameters are evaluated next. It quantifies the utility of candidate points by balancing exploration (e.g., testing regions with high uncertainty) and exploitation (i.e., refining solutions that are predicted to perform well).

Depending on the specific problem properties, many different approaches have been developed. Important research directions include controllers with higher-dimensional controller parameters [278], [279], safety during controller tuning [77], [280]–[283], and settings where changes in the system must be detected and the controller retuned accordingly [284], [285]. A broader overview can be found in [19].

## VII. APPLICATIONS

The developments reviewed in this tutorial have enabled learning-based control methods to be applied across a

Bayesian optimization. We consider a pole-balancing task adapted from [19] (CartPole). For illustrative purposes, we tune only one parameter of a static feedback controller while keeping the remaining controller structure fixed. In BO, we choose $\bar { \boldsymbol { J } }$ to capture the control objective, in this example, through a trajectory-averaged penalty on absolute state-tracking errors and control effort: $\begin{array} { r } { \bar { \boldsymbol { J } } ( \boldsymbol { \theta } ) \hat { = } \frac { 1 } { N } \sum _ { k = 0 } ^ { N - 1 } ( 1 0 | e _ { k } ( \boldsymbol { \theta } ) | _ { 1 } + | u _ { k } ( \boldsymbol { \bar { \theta } } ) | ) } \end{array}$ , where $e _ { k } = [ x _ { k } - r _ { k } , \dot { x } _ { k } , \phi _ { k } , \dot { \phi } _ { k } ] ^ { \top }$ denotes the state-tracking error, $r _ { k }$ the cart-position reference, and $u _ { k }$ the control input. That is, $J$ is evaluated from state and input trajectory data collected during a closed-loop experiment with a chosen gain $\theta ,$ using 30 seconds of data. BO seeks the gain that minimizes this cost.

In Figure 12, we illustrate the behaviour of a BO algorithm using expected improvement as the acquisition function. Rather than evaluating the controller performance exhaustively, BO uses the available observations to construct a GP surrogate model of the cost function (top plot) and selects the next controller parameter based on the acquisition function (bottom plot). The acquisition function therefore does more than simply search for the parameter with the highest uncertainty. It evaluates how promising each candidate is by considering both the predicted cost and the uncertainty of the prediction. This allows BO to focus evaluations on parameters that are likely to improve the current solution, while still exploring regions where the surrogate model is uncertain. Each observation requires applying the corresponding controller parameter to the physical system and measuring the resulting cost J.

By exploiting the prior control structure and tuning only a single parameter, the BO algorithm identifies a near-optimal parameter after only five iterations. This example highlights a key advantage of BO for controller tuning: it can efficiently optimize expensive-to-evaluate objectives while requiring only a small number of system experiments. While this example tunes a

wide range of practical systems. Examples from the CDC Learning-Based Control Invited Sessions span healthcare, robotics and autonomous systems, energy and power systems, process and industrial control, and networked systems. Representative applications include adaptive medication tapering [286], autonomous driving and racing [287]–[290], quadrotor control [75], [90], [91], [139], manipulation [135], [291], [292], multiagent coordination [293]–[297], power electronics [298] and energy management [299], process control [97], and wireless sensor networks [300]. Across these domains, learning-based methods have been used to improve prediction under uncertainty, adapt controllers from data, reduce conservatism, and enhance performance while preserving desired closed-loop properties.

Robotics has been a particularly prominent application area. Robotic systems expose many of the central challenges in learning-based control, including nonlinear and uncertain dynamics, complex task specifications, limited data availability, real-time computation, perception and actuation uncertainty, and strict safety requirements. They therefore provide useful testbeds for understanding the strengths and limitations of different methodological paradigms, as illustrated, for example, by existing benchmarks [11], [301].

Robotic applications also illustrate how learning-oriented and control-oriented approaches can complement one ansingle parameter, standard BO is typically well-suited to tuning a modest number of parameters. Advanced BO variants have been developed targeting specific desiderata of the tuning problem, such as safety considerations, high dimensionality, and timevarying objectives [19].

![](images/dc93679fec7ac12dfedd617018e6c738f2a4a918c80fc6ae9e18886466b93b96.jpg)

![](images/e950aa82a8e4447ed9be5a4a3ae680b3e984c1e6769584cff2465f54360253a5.jpg)  
Fig. 12: Controller tuning with BO. The top plot shows the GP surrogate model of the cost function J, evaluated at different controller parameters θ. The bottom plot shows the corresponding acquisition function (expected improvement), which quantifies the utility of evaluating each candidate controller parameter next.

other in practice. Learning-oriented methods are flexible in two respects. They can optimize policies for complex sequential decision-making tasks, as in RL for autonomous drone racing [10] and quadruped locomotion [9], and they can model poorly known dynamics from data, as in learning terrain-dependent disturbance models for off-road driving [302]. Their performance on real systems, however, depends strongly on the coverage and quality of the training data. Robustification strategies such as domain randomization can compensate for part of the mismatch between training and deployment conditions, but the task performance sacrificed for this robustness is difficult to characterize in advance. Control-oriented methods, in contrast, exploit prior knowledge of system dynamics and structure, which can improve data efficiency and support the analysis of closed-loop properties. For instance, differential flatness can substantially reduce the learning burden for quadrotor control [75] and imposing Lipschitz regularities informed by control theory can support formal closed-loop stability analysis [107], [108].

Recent work extends this complementarity along three axes: higher-dimensional systems, more expressive policy classes, and richer safety specifications. For instance, controlguided deep RL has scaled from quadrotors, where safety filtering is incorporated into training [231], to humanoids, where CLFs shape rewards for dynamic locomotion [303].

Learned policies have likewise progressed from feedforward networks that approximate a single-valued MPC law [253], [268] to diffusion models that represent multimodal distributions over MPC solutions [260], [261]. Beyond more complex systems and more expressive policies, safety certifications are also expanding beyond geometric constraints such as collision avoidance to semantically-driven commonsense specifications required for unstructured, everyday scenarios [304]–[306]. Together, these developments illustrate a common direction: theory lends structure to learning as systems and policies grow more complex, while learning broadens the classes of systems, the range of tasks, and the types of safety specifications that control can accommodate.

## VIII. CONCLUSIONS AND OUTLOOK

This tutorial paper reviewed key methodological directions in learning-based control that have emerged over the past decade, motivated by the CDC Learning-Based Control Invited Sessions. Across the surveyed approaches, a common theme is that machine learning expands what can be modeled, optimized, and adapted, while control theory provides structure for ensuring stability, safety, robustness, and reliable closed-loop behaviour. As autonomous systems move toward increasingly complex, uncertain and safetycritical real-world settings, incorporating data-based methods directly into the closed loop will be critical for achieving reliable high-performance behaviour.

We highlight several directions that we believe have the potential to drive further practical impact.

• Reliably scaling up learning-based control: In this tutorial, we touched upon families of methods that address how uncertainty can be accounted for in learned models and how policies can be optimized without explicit system identification, including through direct data-driven control, RL, IL, and BO (Table I). There are several important directions we did not dive into details but are equally important for reliably scaling these methods to complex real-world problems. For instance, systems must be able to learn online and continually as their dynamics, environments, and objectives change, while avoiding catastrophic forgetting, satisfying real-time computational and memory constraints, and preserving closed-loop guarantees. It is often also desirable to actively identify and explore poorly understood components, while acquiring reliable uncertainty estimates under continual updates. Finally, learningbased control can extend beyond the time-invariant, discrete-time systems considered here to hybrid, timevarying, and distributed dynamics and possibly more tightly couple perception, state estimation, learning, and control within a unified optimal decision-making framework. Addressing these challenges jointly will be essential for developing learning-based control systems that are capable and scalable without sacrificing reliability required for real-world deployment.

• Leveraging control for learning: As discussed throughout this tutorial, learning-based control has expanded

CDC LBC Invited Session Papers with Open-Source Code

![](images/09daf6ab032f01af0ca2d9e1ed26acc071465d15540a2b82c49df2d85e104348.jpg)  
Fig. 13: Number of papers that released open-source code throughout the years for the CDC Learning-Based Control Invited Sessions. In total, 38 of 182 papers, or 20.9% of the papers, released open-source code. Details of the data source and analysis procedure are included in the appendix.

from structured approaches that integrate learned linear or nonlinear models into established control frameworks to direct data-driven methods for synthesizing policies, as well as flexible optimization regimes that require progressively less prior knowledge. These methods have also demonstrated increasing success on real-world systems. This includes emerging massively parallelized RL in synthetic world with zero-shot transferability, IL from diverse multimodal data sources, as well as visionlanguage action and world action models in robotics for acquiring generalist skills. As we scale up learning, a central question is how control-theoretic structure can more deeply inform the learning process—for example, through objective and reward design, constraint enforcement, policy parameterization, and guided exploration. Just as regularization and inductive biases are essential in supervised learning, desirable closed-loop properties—such as stability, safety, robustness, and constraint satisfaction—can serve as principled regularizers when learning is embedded in feedback. Better leveraging these properties during learning may further improve data efficiency, reliability, and generalization.

• Benchmarks, open-source implementations, and reproducibility: The field needs benchmark problems and evaluation metrics that reflect the requirements of deployment. The need for such benchmarks is becoming increasingly apparent as the discussion around the benefits of RL-oriented versus control-oriented approaches grows (e.g., [301], [307], [308]). Such benchmarks should compare not only task performance, but also sample complexity, required prior knowledge, inference time, robustness, safety, and generalization. As discussed in [309], useful benchmarks should be sufficiently complex to expose challenges in controller design, while remaining simple enough to enable meaningful comparisons. Open-source implementations are equally important for reproducibility and crosscommunity progress.

As shown in Figure 13, 20.9% of the invited-session contributions released open-source code. This is an encouraging basis, but also indicates substantial room for improvement. As learning-based control methods become more computationally involved, communicating ideas through both theory and implementable algorithms will be essential for making new methods transparent, reproducible, and extensible.

For readers interested in contributing to the field of learning-based control, a growing collection of open-source educational materials and software tools is available, including several resources developed by the authors [230], [301], [310]–[314]. We hope that this tutorial provides an accessible entry point to the field and a foundation for future work toward safe, reliable, and capable autonomous systems, pushing the boundaries of learning-based control for realworld autonomy.

## ACKNOWLEDGEMENTS

APS, MA, OH, and HZ acknowledge support from the Alexander von Humboldt Foundation and the Robotics Institute Germany (RIG) funded by the German Federal Ministry of Education and Research (BMBF) under grant 16ME0997K. MAM and VGL acknowledge support by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) as part of the Research Unit “Active Learning for Systems and Control (ALeSCo)” – project number 535860958. ST and FS acknowledge support from the Robotics Institute Germany (RIG); the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant TR 1433/4-1 and Germany’s Excellence Strategy EXC-2023 Internet of Production (390621612); as well as a KI-Starter grant by the state of NRW for FS. SZ acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC) through Discovery Grant RGPIN-2026-07696.

The authors thank H. Hose, P. Kruse, P. Brunzema, J. Menn, D. Subhasish, and A. Eisele for fruitful discussions and feedback on the manuscript, and the anonymous reviewers for helpful comments and suggestions.

## REFERENCES

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248–255.

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), 2012.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), 2017.

[4] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “GPT-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[5] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar \` et al., “LLaMA: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[6] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg, “Orbit: A unified simulation framework for interactive robot learning environments,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3740–3747, 2023.

[7] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012, pp. 5026–5033.

[8] M. Schuck, M. P. Rath, Y. Hua, A. Goudar, S. Zhou, and A. P. Schoellig, “Crazyflow: An accurate, GPU-accelerated, differentiable drone simulator in JAX,” arXiv preprint arXiv:2606.01478, 2026.

[9] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk in minutes using massively parallel deep reinforcement learning,” in Proc. of the Conf. on Robot Learning (CoRL), 2022, pp. 91–100.

[10] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Muller, V. Koltun,¨ and D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” Nature, vol. 620, no. 7976, pp. 982–987, 2023.

[11] L. Brunke, M. Greeff, A. W. Hall, Z. Yuan, S. Zhou, J. Panerati, and A. P. Schoellig, “Safe learning in robotics: From learning-based control to safe reinforcement learning,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 5, no. 1, pp. 411–444, 2022.

[12] L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger, “Learning-based model predictive control: Toward safe learning in control,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 3, no. 1, pp. 269–296, 2020.

[13] S. Bansal, M. Chen, S. Herbert, and C. J. Tomlin, “Hamilton-Jacobi reachability: A brief overview and recent advances,” in Proc. of the IEEE Conf. on Decision and Control (CDC), 2017, pp. 2242–2253.

[14] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in Proc. of the European Control Conf. (ECC), 2019, pp. 3420–3431.

[15] D. Duvenaud, J. Lloyd, R. Grosse, J. Tenenbaum, and G. Zoubin, “Structure discovery in nonparametric regression through compositional kernel search,” in Proc. of the International Conference on Machine Learning (ICML), 2013, pp. 1166–1174.

[16] T. M. Moerland, J. Broekens, A. Plaat, and C. M. Jonker, “Modelbased reinforcement learning: A survey,” Foundations and Trends in Machine Learning, vol. 16, no. 1, pp. 1–118, 2023.

[17] J. Garcıa and F. Fernandez, “A comprehensive survey on safe rein-´ forcement learning,” Journal of Machine Learning Research, vol. 16, no. 1, pp. 1437–1480, 2015.

[18] I. Markovsky and F. Dorfler, “Behavioral systems theory in data-¨ driven analysis, signal processing, and control,” Annual Reviews in Control, vol. 52, pp. 42–64, 2021.

[19] D. Stenger, P. Brunzema, J. Menn, A. von Rohr, A. P. Schoellig, and S. Trimpe, “A decade of Bayesian optimization for controller tuning and robot learning: Tutorial, review, and future prospects,” arXiv preprint arXiv:2609.09403, 2026.

[20] K. P. Wabersich, A. J. Taylor, J. J. Choi, K. Sreenath, C. J. Tomlin, A. D. Ames, and M. N. Zeilinger, “Data-driven safety filters: Hamilton-Jacobi reachability, control barrier functions, and predictive methods for uncertain systems,” IEEE Control Systems Magazine, vol. 43, no. 5, pp. 137–177, 2023.

[21] K.-C. Hsu, H. Hu, and J. F. Fisac, “The safety filter: A unified view of safety-critical control in autonomous systems,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 7, pp. 47–72, 2024.

[22] D. Masti and A. Bemporad, “Learning nonlinear state-space models using deep autoencoders,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2018, pp. 3862–3867.

[23] A. Doerr, C. Daniel, M. Schiegg, N.-T. Duy, S. Schaal, M. Toussaint, and S. Trimpe, “Probabilistic recurrent state-space models,” in Proc. of the International Conference on Machine Learning (ICML), 2018, pp. 1280–1289.

[24] D. Gedon, N. Wahlstrom, T. B. Sch ¨ on, and L. Ljung, “Deep ¨ state space models for nonlinear system identification,” IFAC-PapersOnLine, vol. 54, no. 7, pp. 481–486, 2021.

[25] M. Buisson-Fenet, V. Morgenthaler, S. Trimpe, and F. Di Meglio, “Joint state and dynamics estimation with high-gain observers and gaussian process models,” IEEE Control Systems Letters, vol. 5, no. 5, pp. 1627–1632, 2021.

[26] M. Bin and L. Marconi, “Model identification and adaptive state observation for a class of nonlinear systems,” IEEE Transactions on Automatic Control, vol. 66, no. 11, pp. 5292–5307, 2021.

[27] T. D. Barfoot, State estimation for robotics. Cambridge University Press, 2024.

[28] G. Pillonetto, A. Aravkin, D. Gedon, L. Ljung, A. H. Ribeiro, and T. B. Schon, “Deep networks for system identification: a survey,” ¨ Automatica, vol. 171, p. 111907, 2025.

[29] H. K. Khalil and J. W. Grizzle, Nonlinear systems. Prentice Hall Upper Saddle River, NJ, 2002, vol. 3.

[30] D. Angeli, “A Lyapunov approach to incremental stability properties,” IEEE Trans. on Automatic Control, vol. 47, no. 3, pp. 410–421, 2002.

[31] R. D. McAllister and J. B. Rawlings, “Nonlinear stochastic model predictive control: Existence, measurability, and stochastic asymptotic stability,” IEEE Transactions on Automatic Control, vol. 68, no. 3, pp. 1524–1536, 2023.

[32] D. Chatterjee and J. Lygeros, “On stability and performance of stochastic predictive control techniques,” IEEE Transactions on Automatic Control, vol. 60, no. 2, pp. 509–514, 2015.

[33] F. Blanchini, “Set invariance in control,” Automatica, vol. 35, no. 11, pp. 1747–1767, 1999.

[34] M. Chen and C. J. Tomlin, “Hamilton–Jacobi reachability: Some recent theoretical advances and applications in unmanned airspace management,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 1, no. Volume 1, 2018, pp. 333–358, 2018.

[35] J. Borquez, K. Chakraborty, H. Wang, and S. Bansal, “On safety and liveness filtering using Hamilton-Jacobi reachability analysis,” IEEE Transactions on Robotics, vol. 40, pp. 4235–4251, 2024.

[36] K. P. Wabersich and M. N. Zeilinger, “Linear model predictive safety certification for learning-based control,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2018, pp. 7130–7135.

[37] ——, “A predictive safety filter for learning-based control of constrained nonlinear dynamical systems,” Automatica, vol. 129, p. 109597, 2021.

[38] M. Alshiekh, R. Bloem, R. Ehlers, B. Konighofer, S. Niekum, and¨ U. Topcu, “Safe reinforcement learning via shielding,” in Proc. of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[39] O. Bastani, “Safe reinforcement learning with nonlinear dynamics via model predictive shielding,” in Proc. of the American Control Conference (ACC), 2021, pp. 3488–3494.

[40] E. Kofman, J. A. De Dona, and M. M. Seron, “Probabilistic set´ invariance and ultimate boundedness,” Automatica, vol. 48, no. 10, pp. 2670–2676, 2012.

[41] L. Hewing, A. Carron, K. P. Wabersich, and M. N. Zeilinger, “On a correspondence between probabilistic and robust invariant sets for linear systems,” in Proc. of the European Control Conference (ECC), 2018, pp. 1642–1647.

[42] J. J. Choi, D. Lee, B. Li, J. P. How, K. Sreenath, S. L. Herbert, and C. J. Tomlin, “A forward reachability perspective on robust control invariance and discount factors in reachability analysis,” arXiv preprint arXiv:2310.17180, 2023.

[43] H. Seo, D. Lee, C. Y. Son, I. Jang, C. J. Tomlin, and H. J. Kim, “Real-time robust receding horizon planning using Hamilton-Jacobi reachability analysis,” IEEE Transactions on Robotics, vol. 39, no. 1, pp. 90–109, 2022.

[44] X. Xu, P. Tabuada, J. W. Grizzle, and A. D. Ames, “Robustness of control barrier functions for safety critical control,” IFAC-PapersOnLine, vol. 48, no. 27, pp. 54–61, 2015.

[45] J. J. Choi, D. Lee, K. Sreenath, C. J. Tomlin, and S. L. Herbert, “Robust control barrier-value functions for safety-critical control,” in Proc. ofthe IEEE Conference on Decision and Control (CDC), 2021, pp. 6814–6821.

[46] A. Abate, M. Prandini, J. Lygeros, and S. Sastry, “Probabilistic reachability and safety for controlled discrete time stochastic hybrid systems,” Automatica, vol. 44, no. 11, pp. 2724–2734, 2008.

[47] R. K. Cosner, P. Culbertson, A. J. Taylor, and A. D. Ames, “Robust safety under stochastic uncertainty with discrete-time control barrier functions,” in Proc. of Robotics: Science and Systems (RSS), 2023.

[48] K. P. Wabersich, L. Hewing, A. Carron, and M. N. Zeilinger, “Probabilistic model predictive safety certification for learning-based control,” IEEE Transactions on Automatic Control, vol. 67, no. 1, pp. 176–188, 2021.

[49] E. Altman, Constrained Markov decision processes. Routledge, 2021.

[50] J. Nocedal and S. J. Wright, Numerical optimization. Springer, 2006.

[51] A. Didier, A. Zanelli, K. P. Wabersich, and M. N. Zeilinger, “Predictive stability filters for nonlinear dynamical systems affected by disturbances,” in Proc. of the IFAC Conference on Nonlinear Model Predictive Control (NMPC), vol. 58, no. 18, 2024, pp. 200–207.

[52] E. Milios, K. P. Wabersich, F. Berkel, and L. Schwenkel, “Stability mechanisms for predictive safety filters,” in 2024 IEEE 63rd Conference on Decision and Control (CDC), 2024, pp. 2409–2416.

[53] I. M. Mitchell, “A toolbox of level set methods,” Department of Computer Science, University of British Columbia, Tech. Rep. TR-2007-11, 2007.

[54] E. Schmerling, “hj reachability: Hamilton-Jacobi reachability analysis in JAX,” https://github.com/StanfordASL/hj reachability, 2021.

[55] M. Bui, G. Giovanis, M. Chen, and A. Shriraman, “OptimizedDP: An efficient, user-friendly library for optimal control and dynamic programming,” arXiv preprint arXiv:2204.05520, 2022.

[56] S. Bansal and C. J. Tomlin, “DeepReach: A deep learning approach to high-dimensional reachability,” in Proc. of the IEEE International Conf. on Robotics and Automation (ICRA), 2021, pp. 1817–1824.

[57] M. Althoff, “An introduction to CORA 2015,” in Proc. of the Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH), 2015, pp. 120–151.

[58] S. Bogomolov, M. Forets, G. Frehse, K. Potomkin, and C. Schilling, “JuliaReach: a toolbox for set-based reachability,” in Proc. of the ACM International Conference on Hybrid Systems: Computation and Control (HSCC), 2019, pp. 39–44.

[59] G. Frison and M. Diehl, “HPIPM: a high-performance quadratic programming framework for model predictive control,” IFAC-PapersOnLine, vol. 53, no. 2, pp. 6563–6569, 2020.

[60] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, “OSQP: an operator splitting solver for quadratic programs,” Math. Programming Computation, vol. 12, no. 4, pp. 637–672, 2020.

[61] P. J. Goulart and Y. Chen, “Clarabel: An interior-point solver for conic programs with quadratic objectives,” arXiv preprint arXiv:2405.12762, 2024.

[62] H. J. Ferreau, C. Kirches, A. Potschka, H. G. Bock, and M. Diehl, “qpOASES: A parametric active-set algorithm for quadratic programming,” Mathematical Programming Computation, vol. 6, no. 4, pp. 327–363, 2014.

[63] S. Prajna and A. Jadbabaie, “Safety verification of hybrid systems using barrier certificates,” in Proc. of the Intl. Workshop on Hybrid Systems: Computation and Control (HSCC), 2004, pp. 477–492.

[64] A. Papachristodoulou, J. Anderson, G. Valmorbida, S. Prajna, P. Seiler, P. A. Parrilo, M. M. Peet, and D. Jagt, “SOSTOOLS version 4.00 sum of squares optimization toolbox for MATLAB,” arXiv preprint arXiv:1310.4716, 2021.

[65] A. Abate, D. Ahmed, A. Edwards, M. Giacobbe, and A. Peruffo, “FOSSIL: a software tool for the formal synthesis of Lyapunov functions and barrier certificates using neural networks,” in Proc. of the ACM International Conference on Hybrid Systems: Computation and Control (HSCC), 2021, pp. 1–11.

[66] S. Tonkens and S. Herbert, “Refining control barrier functions through Hamilton-Jacobi reachability,” in Proc. of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 13 355–13 362.

[67] M. Black, G. Fainekos, B. Hoxha, H. Okamoto, and D. Prokhorov, “CBFkit: A control barrier function toolbox for robotics applications,” in Proc. of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

[68] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, “CasADi: a software framework for nonlinear optimization and optimal control,” Mathematical Programming Computation, vol. 11, no. 1, pp. 1–36, 2019.

[69] A. Wachter and L. T. Biegler, “On the implementation of an interior-¨ point filter line-search algorithm for large-scale nonlinear programming,” Mathematical Programming, vol. 106, no. 1, pp. 25–57, 2006.

[70] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren, A. Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, “acados - a modular open-source framework for fast embedded optimal control,” Mathematical Programming Computation, vol. 14, no. 1, pp. 147–183, 2022.

[71] A. Zanelli, A. Domahidi, J. Jerez, and M. Morari, “FORCES NLP: an efficient implementation of interior-point methods for multistage nonlinear nonconvex programs,” International Journal of Control, vol. 93, no. 1, pp. 13–29, 2020.

[72] L. Ljung, “System identification,” in Signal Analysis and Prediction, A. Prochazka, J. Uhl´ ´ıˇr, P. W. J. Rayner, and N. G. Kingsbury, Eds., 1998, pp. 163–173.

[73] J. Umlauft, L. Pohler, and S. Hirche, “An uncertainty-based control¨ Lyapunov approach for control-affine systems modeled by Gaussian process,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2018, pp. 483–488.

[74] G. Evangelisti and S. Hirche, “Physically consistent learning of conservative Lagrangian systems with Gaussian processes,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2022, pp. 4078–4085.

[75] M. Greeff and A. P. Schoellig, “Exploiting differential flatness for robust learning-based tracking control using Gaussian processes,” in Proc. ofthe IEEE Conference on Decision and Control (CDC), 2021, pp. 1121–1126.

[76] A. Carron, E. Arcari, M. Wermelinger, L. Hewing, M. Hutter, and M. N. Zeilinger, “Data-driven model predictive control for trajectory tracking with a robotic arm,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3758–3765, 2019.

[77] L. P. Frohlich, C. K¨ uttel, E. Arcari, L. Hewing, M. N. Zeilinger,¨ and A. Carron, “Contextual tuning of model predictive control for autonomous racing,” in Proc. of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 10 555–10 562.

[78] G. R. Dolby, “Generalized least squares and maximum likelihood estimation of non-linear functional relationships,” Journal of the Royal Statistical Society: Series B (Methodological), vol. 34, no. 3, pp. 393–400, 1972.

[79] A. Tsiamis and G. J. Pappas, “Linear systems can be hard to learn,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2021, pp. 2903–2910.

[80] S. Dean, H. Mania, N. Matni, B. Recht, and S. Tu, “On the sample complexity of the linear quadratic regulator,” Foundations of Computational Mathematics, vol. 20, no. 4, pp. 633–679, 2020.

[81] Y. Abbasi-Yadkori and C. Szepesvari, “Regret bounds for the adaptive´ control of linear quadratic systems,” in Proc. of the Conference on Learning Theory (COLT), 2011, pp. 1–26.

[82] H. Mania, S. Tu, and B. Recht, “Certainty Equivalence is Efficient for Linear Quadratic Control,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[83] J. Anderson, J. C. Doyle, S. H. Low, and N. Matni, “System level synthesis,” Annual Reviews in Control, vol. 47, pp. 364–393, 2019.

[84] D. Q. Mayne, M. M. Seron, and S. V. Rakovic, “Robust model predic-´ tive control of constrained linear systems with bounded disturbances,” Automatica, vol. 41, no. 2, pp. 219–224, 2005.

[85] F. Farokhi, A. S. Leong, M. Zamani, and I. Shames, “Learning safety filters for unknown discrete-time linear systems,” IEEE Control Systems Letters, vol. 7, pp. 1843–1848, 2023.

[86] D. Tranos, A. Russo, and A. Proutiere, “Self-tuning tube-based model predictive control,” in 2023 American Control Conference (ACC), 2023, pp. 3626–3632.

[87] A. Somalwar, B. D. Lee, G. J. Pappas, and N. Matni, “Learning with imperfect models: When multi-step prediction mitigates compounding error,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2025, pp. 82–89.

[88] A. Tsiamis, I. Ziemann, N. Matni, and G. J. Pappas, “Statistical learning theory for control: A finite-sample perspective,” IEEE Control Systems Magazine, vol. 43, no. 6, pp. 67–97, 2023.

[89] K. Hornik, M. Stinchcombe, and H. White, “Multilayer feedforward networks are universal approximators,” Neural Networks, vol. 2, no. 5, pp. 359–366, 1989.

[90] S. Bansal, A. K. Akametalu, F. J. Jiang, F. Laine, and C. J. Tomlin, “Learning quadrotor dynamics using neural network for flight control,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2016, pp. 4653–4660.

[91] S. Zhou, M. K. Helwa, and A. P. Schoellig, “Design of deep neural networks as add-on blocks for improving impromptu trajectory tracking,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2017, pp. 5201–5207.

[92] C. M. Bishop and N. M. Nasrabadi, Pattern recognition and machine learning. Springer, 2006, vol. 4, no. 4.

[93] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018.

[94] T. X. Nghiem, J. Drgona, C. Jones, Z. Nagy, R. Schwan, B. Dey,ˇ A. Chakrabarty, S. Di Cairano, J. A. Paulson, A. Carron et al., “Physics-informed machine learning for modeling and control of dynamical systems,” in Proc. of the American Control Conference (ACC), 2023, pp. 3735–3750.

[95] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and

inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019.

[96] J. Connor, L. Atlas, and D. Martin, “Recurrent networks and NARMA modeling,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), vol. 4, 1991.

[97] L. B. Armenio, E. Terzi, M. Farina, and R. Scattolini, “Model predictive control design for dynamical systems learned by echo state networks,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2019, pp. 1044–1049.

[98] F. Bonassi, J. Xie, M. Farina, and R. Scattolini, “An offset-free nonlinear MPC scheme for systems learned by neural NARX models,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2022, pp. 2123–2128.

[99] L. B. de Giuli, A. L. Bella, M. Farina, and R. Scattolini, “Modeling and predictive control of networked systems via physics-informed neural networks,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 3005–3010.

[100] D. Ravasio, M. Farina, and A. Ballarino, “LMI-based design of a robust model predictive controller for a class of recurrent neural networks with guaranteed properties,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 1126–1131.

[101] H. Wang, J. Borquez, and S. Bansal, “Providing safety assurances for systems with unknown dynamics,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 1108–1113.

[102] M. Fazlyab, A. Robey, H. Hassani, M. Morari, and G. J. Pappas, “Efficient and accurate estimation of Lipschitz constants for deep neural networks,” in Proc. of the International Conference on Neural Information Processing Systems (NeurIPS), Red Hook, NY, USA, 2019.

[103] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, “Spectral normalization for generative adversarial networks,” in Proc. of the International Conference on Learning Representations (ICLR), 2018.

[104] P. Pauli, A. Koch, J. Berberich, P. Kohler, and F. Allgower, “Train-¨ ing robust neural networks using Lipschitz bounds,” IEEE Control Systems Letters, vol. 6, pp. 121–126, 2022.

[105] L. Bungert, R. Raab, T. Roith, L. Schwinn, and D. Tenbrinck, “CLIP: Cheap lipschitz training of neural networks,” in Proc. of the International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), 2021, pp. 307–319.

[106] C. Anil, J. Lucas, and R. Grosse, “Sorting out lipschitz function approximation,” in Proc. of the International Conference on Machine Learning (ICML), 2019, pp. 291–301.

[107] G. Shi, X. Shi, M. O’Connell, R. Yu, K. Azizzadenesheli, A. Anandkumar, Y. Yue, and S.-J. Chung, “Neural lander: Stable drone landing control using learned dynamics,” in Proc. of the IEEE Intl. Conf. on Robotics and Automation (ICRA), 2019, pp. 9784–9790.

[108] S. Zhou, K. Pereida, W. Zhao, and A. P. Schoellig, “Bridging the model-reality gap with Lipschitz network adaptation,” IEEE Robotics and Automation Letters, vol. 7, no. 1, pp. 642–649, 2022.

[109] P. Pauli, D. Gramlich, J. Berberich, and F. Allgower, “Linear systems¨ with neural network nonlinearities: Improved stability analysis via acausal Zames-Falb multipliers,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2021, pp. 3611–3618.

[110] H. Yin, P. Seiler, and M. Arcak, “Stability analysis using quadratic constraints for systems with neural network controllers,” IEEE Transactions on Automatic Control, vol. 67, no. 4, pp. 1980–1987, 2022.

[111] F. Bonassi, M. Farina, and R. Scattolini, “Stability of discrete-time feed-forward neural networks in NARX configuration,” in Proc. of the IFAC Symposium on System Identification (SYSID), vol. 54, no. 7, 2021, pp. 547–552.

[112] F. Bonassi, E. Terzi, M. Farina, and R. Scattolini, “LSTM neural networks: Input to state stability and probabilistic safety verification,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2020, pp. 85–94.

[113] W. D’Amico, A. La Bella, and M. Farina, “An incremental inputto-state stability condition for a class of recurrent neural networks,” IEEE Transactions on Automatic Control, vol. 69, no. 4, pp. 2221– 2236, 2024.

[114] D. Martinelli, C. L. Galimberti, I. R. Manchester, L. Furieri, and G. Ferrari-Trecate, “Unconstrained parametrization of dissipative and contracting neural ordinary differential equations,” in Proc. of the IEEE Conf. on Decision and Control (CDC), 2023, pp. 3043–3048.

[115] M. Revay, R. Wang, and I. R. Manchester, “Recurrent equilibrium networks: Flexible dynamic models with guaranteed stability and

robustness,” IEEE Transactions on Automatic Control, vol. 69, no. 5, pp. 2855–2870, 2024.

[116] Y. Bao, J. M. Velni, and M. Shahbakhti, “Epistemic Uncertainty Quantification in State-Space LPV Model Identification Using Bayesian Neural Networks,” IEEE Control Systems Letters, vol. 5, no. 2, pp. 719–724, 2021.

[117] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proc. ofthe Intl. Conference on Machine Learning (ICML), 2016, pp. 1050–1059.

[118] D. Cohn, “Neural network exploration using optimal experiment design,” in Proc. of the Advances in Neural Information Processing Systems (NeurIPS), vol. 6, 1993.

[119] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in Proc. of the International Conference on Neural Information Processing Systems (NeurIPS), 2017, pp. 6405–6416.

[120] C. K. Williams and C. E. Rasmussen, Gaussian processes for machine learning. MIT Press, Cambridge, MA, 2006, vol. 2, no. 3.

[121] C. D. McKinnon and A. P. Schoellig, “Learning probabilistic models for safe predictive control in unknown environments,” in Proc. of the European Control Conference (ECC), 2019, pp. 2472–2479.

[122] A. P. Schoellig, S. Zhou, H. Zhao, and L. Brunke, “Control for Robotics: From Optimal Control to Reinforcement Learning,” 2026, course Jupyter notebook. [Online]. Available: https://learnsyslab. github.io/cfr/programming/

[123] L. Hewing, E. Arcari, L. P. Frohlich, and M. N. Zeilinger, “On simu- ¨ lation and trajectory prediction with Gaussian process dynamics,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2020, pp. 424–434.

[124] M. Opper, “A Bayesian approach to on-line learning,” in On-line learning in neural networks, USA, 1999, pp. 363–378.

[125] J. Hardy, F. Havlak, and M. Campbell, “Multiple-step prediction using a two stage gaussian process model,” in Proc. of the American Control Conference (ACC), 2014, pp. 3443–3449.

[126] A. Scampicchio, E. Arcari, A. Lahr, and M. N. Zeilinger, “Gaussian processes for dynamics learning in model predictive control,” Annual Reviews in Control, vol. 60, p. 101034, 2025.

[127] A. Lahr, A. Zanelli, A. Carron, and M. N. Zeilinger, “Zero-order optimization for gaussian process-based model predictive control,” European Journal of Control, vol. 74, p. 100862, 2023.

[128] A. Lahr, J. Naf, K. P. Wabersich, J. Frey, P. Siehl, A. Carron,¨ M. Diehl, and M. N. Zeilinger, “L4acados: Learning-based models for acados, applied to gaussian process-based predictive control,” IEEE Transactions on Control Systems Technology, 2026.

[129] T. Koller, F. Berkenkamp, M. Turchetta, and A. Krause, “Learningbased model predictive control for safe exploration,” in Proc. of the IEEE Conf. on Decision and Control (CDC), 2018, pp. 6059–6066.

[130] M. Prajapat, A. Lahr, J. Kohler, A. Krause, and M. N. Zeilinger,¨ “Towards safe and tractable Gaussian process-based MPC: Efficient sampling within a sequential quadratic programming framework,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 7458–7465.

[131] E. Bradford, L. Imsland, and E. A. del Rio-Chanona, “Nonlinear model predictive control with explicit back-offs for gaussian process state space models,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2019, pp. 4747–4754.

[132] F. Micheli and J. Lygeros, “Scenario-based stochastic mpc for systems with uncertain dynamics,” in Proc. of the European Control Conference (ECC), 2022, pp. 833–838.

[133] F. Berkenkamp and A. P. Schoellig, “Safe and robust learning control with gaussian processes,” in Proc. of the European Control Conference (ECC), 2015, pp. 2496–2501.

[134] A. von Rohr, M. Neumann-Brosig, and S. Trimpe, “Probabilistic robust linear quadratic regulators with Gaussian processes,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2021, pp. 324–335.

[135] T. Beckers, J. Umlauft, D. Kulic, and S. Hirche, “Stable Gaussian process based tracking control of Lagrangian systems,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2017, pp. 5180–5185.

[136] A. Capone and S. Hirche, “Backstepping for partially unknown nonlinear systems using Gaussian processes,” IEEE Control Systems Letters, vol. 3, no. 2, pp. 416–421, 2019.

[137] B. Settles, Active learning. Morgan & Claypool Publishers, 2012.

[138] S. Bansal, R. Calandra, T. Xiao, S. Levine, and C. J. Tomlin, “Goaldriven dynamics learning via Bayesian optimization,” in Proc. of the IEEE Conf. on Decision and Control (CDC), 2017, pp. 5168–5173.

[139] S. Zhou and A. P. Schoellig, “Active training trajectory generation for inverse dynamics model learning with deep neural networks,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2019, pp. 1784–1790.

[140] B. D. Lee, I. Ziemann, G. J. Pappas, and N. Matni, “Active learning for control-oriented identification of nonlinear systems,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 3011–3018.

[141] A. Capone, G. Noske, J. Umlauft, T. Beckers, A. Lederer, and S. Hirche, “Localized active learning of Gaussian process state space models,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2020, pp. 490–499.

[142] B. Sukhija, L. Treven, C. Sancaktar, S. Blaes, S. Coros, and A. Krause, “Optimistic active exploration of dynamical systems,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), 2023.

[143] M. Buisson-Fenet, F. Solowjow, and S. Trimpe, “Actively learning Gaussian process dynamics,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2020, pp. 5–15.

[144] A. Wagenmaker and K. Jamieson, “Active learning for identification of linear dynamical systems,” in Proc. of Conference on Learning Theory (COLT), 2020, pp. 3487–3582.

[145] A. A. Feldbaum, “Dual control theory I,” Avtomatika i Telemekhanika, vol. 21, no. 9, pp. 1240–1249, 1960.

[146] E. Arcari, L. Hewing, M. Schlichting, and M. Zeilinger, “Dual stochastic mpc for systems with parametric and structural uncertainty,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2020, pp. 894–903.

[147] E. Arcari, L. Hewing, and M. N. Zeilinger, “An approximate dynamic programming approach for dual stochastic model predictive control,” IFAC-PapersOnLine, vol. 53, no. 2, pp. 8105–8111, 2020.

[148] L. Hewing, J. Kabzan, and M. N. Zeilinger, “Cautious model predictive control using Gaussian process regression,” IEEE Transactions on Control Systems Technology, vol. 28, no. 6, pp. 2736–2743, 2019.

[149] L. Bartels, A. Lahr, A. Carron, and M. N. Zeilinger, “Real-time online learning for model predictive control using a spatio-temporal gaussian process approximation,” arXiv preprint arXiv:2603.17632, 2026.

[150] F. Solowjow and S. Trimpe, “Event-triggered learning,” Automatica, vol. 117, p. 109009, 2020.

[151] J. Umlauft and S. Hirche, “Feedback linearization based on Gaussian processes with event-triggered online learning,” IEEE Transactions on Automatic Control, vol. 65, no. 10, pp. 4154–4169, 2020.

[152] H. Schluter, F. Solowjow, and S. Trimpe, “Event-triggered learning ¨ for linear quadratic control,” IEEE Transactions on Automatic Control, vol. 66, no. 10, pp. 4485–4498, 2021.

[153] K. Zheng, D. Shi, Y. Shi, and J. Wang, “Nonparameteric eventtriggered learning with applications to adaptive model predictive control,” IEEE Transactions on Automatic Control, vol. 68, no. 6, pp. 3469–3484, 2023.

[154] J. C. Willems, P. Rapisarda, I. Markovsky, and B. L. De Moor, “A note on persistency of excitation,” Systems & Control Letters, vol. 54, no. 4, pp. 325–329, 2005.

[155] C. De Persis and P. Tesi, “Formulas for data-driven control: Stabilization, optimality, and robustness,” IEEE Transactions on Automatic Control, vol. 65, no. 3, pp. 909–924, 2020.

[156] H. J. van Waarde, C. De Persis, M. K. Camlibel, and P. Tesi, “Willems’ fundamental lemma for state-space systems and its extension to multiple datasets,” IEEE Control Systems Letters, vol. 4, no. 3, pp. 602–607, 2020.

[157] J. Berberich and F. Allgower, “A trajectory-based framework for data-¨ driven system analysis and control,” in Proc. of the European Control Conference (ECC), 2020, pp. 1365–1370.

[158] H. J. Van Waarde, J. Eising, M. K. Camlibel, and H. L. Trentelman, “The informativity approach: To data-driven analysis and control,” IEEE Control Systems Magazine, vol. 43, no. 6, pp. 32–66, 2023.

[159] V. G. Lopez and M. A. Muller, “On a continuous-time version of¨ Willems’ lemma,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2022, pp. 2759–2764.

[160] V. G. Lopez, M. A. Muller, and P. Rapisarda, “An input-output¨ continuous-time version of Willems’ lemma,” IEEE Control Systems Letters, vol. 8, pp. 916–921, 2024.

[161] P. Rapisarda, M. C¸ amlibel, and H. van Waarde, “A “fundamental lemma” for continuous-time systems, with applications to data-driven simulation,” Systems & Control Letters, vol. 179, p. 105603, 2023.

[162] P. Schmitz, T. Faulwasser, P. Rapisarda, and K. Worthmann, “A continuous-time fundamental lemma and its application in datadriven optimal control,” Systems & Control Letters, vol. 194, p. 105950, 2024.

[163] I. Markovsky and F. Dorfler, “Identifiability in the behavioral setting,” ¨ IEEE Transactions on Automatic Control, vol. 68, no. 3, pp. 1667– 1677, 2023.

[164] B. Nortmann and T. Mylvaganam, “Direct data-driven control of linear time-varying systems,” IEEE Transactions on Automatic Control, vol. 68, no. 8, pp. 4888–4895, 2023.

[165] C. Verhoek, R. Toth, S. Haesaert, and A. Koch, “Fundamental lemma´ for data-driven analysis of linear parameter-varying systems,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2021, pp. 5040–5046.

[166] P. Schmitz, T. Faulwasser, and K. Worthmann, “Willems’ fundamental lemma for linear descriptor systems and its use for data-driven output-feedback MPC,” IEEE Control Systems Letters, vol. 6, pp. 2443–2448, 2022.

[167] V. G. Lopez and M. A. Muller, “Data-based system representation¨ and synchronization for multiagent systems,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 6736–6742.

[168] G. Pan, R. Ou, and T. Faulwasser, “On a stochastic fundamental lemma and its use for data-driven optimal control,” IEEE Transactions on Automatic Control, vol. 68, no. 10, pp. 5922–5937, 2023.

[169] M. Yin, A. Iannelli, and R. S. Smith, “Maximum likelihood estimation in data-driven modeling and control,” IEEE Transactions on Automatic Control, vol. 68, no. 1, pp. 317–328, 2023.

[170] M. Alsalti, I. Markovsky, V. G. Lopez, and M. A. Muller, “Data-¨ based system representations from irregularly measured data,” IEEE Transactions on Automatic Control, vol. 70, no. 1, pp. 143–158, 2025.

[171] O. Molodchyk and T. Faulwasser, “Exploring the links between the fundamental lemma and kernel regression,” IEEE Control Systems Letters, vol. 8, pp. 2045–2050, 2024.

[172] J. G. Rueda-Escobedo and J. Schiffer, “Data-driven internal model control of second-order discrete Volterra systems,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2020, pp. 4572– 4579.

[173] I. Markovsky, “Data-driven simulation of generalized bilinear systems via linear time-invariant embedding,” IEEE Transactions on Automatic Control, vol. 68, no. 2, pp. 1101–1106, 2023.

[174] M. Alsalti, V. G. Lopez, J. Berberich, F. Allgower, and M. A.¨ Muller, “Data-based control of feedback linearizable systems,”¨ IEEE Transactions on Automatic Control, vol. 68, no. 11, pp. 7014–7021, 2023.

[175] X. Shang, J. Cortes, and Y. Zheng, “Willems’ fundamental lemma for´ nonlinear systems with Koopman linear embedding,” IEEE Control Systems Letters, vol. 8, pp. 3135–3140, 2024.

[176] H. J. van Waarde, M. K. Camlibel, and H. L. Trentelman, Data-Based Linear Systems and Control Theory, 1st ed. Kindle Direct Publishing, 2025.

[177] A. Bisoffi, C. De Persis, and P. Tesi, “Data-driven control via Petersen’s lemma,” Automatica, vol. 145, p. 110537, 2022.

[178] H. J. van Waarde, M. K. Camlibel, and M. Mesbahi, “From noisy data to feedback controllers: Nonconservative design via a matrix S-lemma,” IEEE Transactions on Automatic Control, vol. 67, no. 1, pp. 162–175, 2022.

[179] J. Berberich, C. W. Scherer, and F. Allgower, “Combining prior¨ knowledge and data for robust controller design,” IEEE Transactions on Automatic Control, vol. 68, no. 8, pp. 4618–4633, 2023.

[180] F. Dorfler, P. Tesi, and C. De Persis, “On the certainty-equivalence¨ approach to direct data-driven LQR design,” IEEE Transactions on Automatic Control, vol. 68, no. 12, pp. 7989–7996, 2023.

[181] V. G. Lopez, M. Alsalti, and M. A. Muller, “Efficient off-policy Q-¨ learning for data-based discrete-time LQR problems,” IEEE Transactions on Automatic Control, vol. 68, no. 5, pp. 2922–2933, 2023.

[182] V. Breschi, C. D. Persis, S. Formentin, and P. Tesi, “Direct datadriven model-reference control with Lyapunov stability guarantees,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2021, pp. 1456–1461.

[183] G. Bianchin, “Data-driven exact pole placement for linear systems,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2023, pp. 685–690.

[184] A. Bisoffi, C. De Persis, and P. Tesi, “Learning controllers for performance through LMI regions,” IEEE Transactions on Automatic Control, vol. 68, no. 7, pp. 4351–4358, 2023.

[185] P. Rapisarda, H. J. van Waarde, and M. C¸ amlibel, “Orthogonal polynomial bases for data-driven analysis and control of continuoustime systems,” IEEE Transactions on Automatic Control, vol. 69, no. 7, pp. 4307–4319, 2024.

[186] J. Eising and J. Cortes, “When sampling works in data-driven control: ´ Informativity for stabilization in continuous time,” IEEE Transactions on Automatic Control, vol. 70, no. 1, pp. 565–572, 2025.

[187] V. G. Lopez and M. A. Muller, “Data-based control of continuous-¨ time linear systems with performance specifications,” Automatica, vol. 189, p. 113002, 2026.

[188] C. Possieri, “Derivative free data-driven stabilization of continuoustime linear systems from input-output data,” IEEE Control Systems Letters, vol. 10, pp. 67–72, 2026.

[189] C. De Persis and P. Tesi, “Learning controllers for nonlinear systems from data,” Annual Reviews in Control, vol. 56, p. 100915, 2023.

[190] Z. Hu, C. De Persis, and P. Tesi, “Enforcing contraction via data,” IEEE Trans. on Auto. Control, vol. 70, no. 12, pp. 8076–8091, 2025.

[191] Z. Yuan and J. Cortes, “Data-driven optimal control of bilinear´ systems,” IEEE Control Systems Letters, vol. 6, pp. 2479–2484, 2022.

[192] C. Verhoek, P. J. W. Koelewijn, S. Haesaert, and R. Toth, “Direct´ data-driven state-feedback control of general nonlinear systems,” in Proc. ofthe IEEE Conference on Decision and Control (CDC), 2023, pp. 3688–3693.

[193] T. Martin, T. B. Schon, and F. Allg¨ ower, “Guarantees for data-driven¨ control of nonlinear systems using semidefinite programming: A survey,” Annual Reviews in Control, vol. 56, p. 100911, 2023.

[194] M. Alsalti, C. De Persis, V. G. Lopez, and M. A. Muller, “Data- ¨ driven stabilization of nonlinear systems via descriptor embedding,” arXiv preprint arXiv:2511.01457, 2025.

[195] M. Alsalti, V. G. Lopez, and M. A. Muller, “Notes on data-driven¨ output-feedback control of linear MIMO systems,” IEEE Transactions on Automatic Control, vol. 70, no. 9, pp. 6143–6150, 2025.

[196] H. J. van Waarde, J. Eising, M. K. Camlibel, and H. L. Trentelman, “A behavioral approach to data-driven control with noisy input–output data,” IEEE Transactions on Automatic Control, vol. 69, no. 2, pp. 813–827, 2024.

[197] L. Li, A. Bisoffi, C. De Persis, and N. Monshizadeh, “Controller synthesis from noisy-input noisy-output data,” Automatica, vol. 183, p. 112545, 2026.

[198] M. S. Turan and G. Ferrari-Trecate, “Data-driven unknown-input observers and state estimation,” IEEE Control Systems Letters, vol. 6, pp. 1424–1429, 2022.

[199] J. Shi, Y. Lian, and C. N. Jones, “Data-driven input reconstruction and experimental validation,” IEEE Control Systems Letters, vol. 6, pp. 3259–3264, 2022.

[200] W. Liu, G. Wang, J. Sun, F. Bullo, and J. Chen, “Learning robust data-based LQG controllers from noisy data,” IEEE Transactions on Automatic Control, vol. 69, no. 12, pp. 8526–8538, 2024.

[201] T. M. Wolff, V. G. Lopez, and M. A. Muller, “Robust data-driven¨ moving horizon estimation for linear discrete-time systems,” IEEE Trans. on Automatic Control, vol. 69, no. 8, pp. 5598–5604, 2024.

[202] J. Coulson, J. Lygeros, and F. Dorfler, “Data-enabled predictive¨ control: In the shallows of the DeePC,” in Proc. of the European Control Conference (ECC), 2019, pp. 307–312.

[203] J. Berberich, J. Kohler, M. A. M¨ uller, and F. Allg¨ ower, “Data-driven¨ model predictive control with stability and robustness guarantees,” IEEE Trans. on Auto. Control, vol. 66, no. 4, pp. 1702–1717, 2021.

[204] J. Coulson, J. Lygeros, and F. Dorfler, “Regularized and distribu-¨ tionally robust data-enabled predictive control,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2019, pp. 2696–2701.

[205] F. Dorfler, J. Coulson, and I. Markovsky, “Bridging direct and indirect¨ data-driven control formulations via regularizations and relaxations,” IEEE Trans. on Automatic Control, vol. 68, no. 2, pp. 883–897, 2023.

[206] V. Breschi, A. Chiuso, and S. Formentin, “Data-driven predictive control in a stochastic setting: a unified framework,” Automatica, vol. 152, p. 110961, 2023.

[207] M. Kladtke and M. S. Darup, “Towards explainable data-driven¨ predictive control with regularizations,” Automatisierungstechnik, vol. 73, no. 6, pp. 365–382, 2025.

[208] G. Pan, R. Ou, and T. Faulwasser, “On data-driven stochastic outputfeedback predictive control,” IEEE Transactions on Automatic Control, vol. 70, no. 5, pp. 2948–2962, 2025.

[209] S. Kerz, J. Teutsch, T. Brudigam, M. Leibold, and D. Wollherr, “Data-¨ driven tube-based stochastic predictive control,” IEEE Open Journal of Control Systems, vol. 2, pp. 185–199, 2023.

[210] M. Yin, A. Iannelli, and R. S. Smith, “Stochastic data-driven predictive control: Regularization, estimation, and constraint tightening,” in Proc. of the IFAC Symposium on System Identification (SYSID), vol. 58, no. 16, 2024, pp. 250–257.

[211] E. O’Dwyer, E. C. Kerrigan, P. Falugi, M. Zagorowska, and N. Shah, “Data-driven predictive control with improved performance using segmented trajectories,” IEEE Transactions on Control Systems Technology, vol. 31, no. 3, pp. 1355–1365, 2023.

[212] K. Zhang, Y. Zheng, C. Shang, and Z. Li, “Dimension reduction for efficient data-enabled predictive control,” IEEE Control Systems Letters, vol. 7, pp. 3277–3282, 2023.

[213] M. Alsalti, M. Barkey, V. G. Lopez, and M. A. Muller, “Robust¨ and efficient data-driven predictive control,” Automatica, vol. 191, p. 113108, 2026.

[214] J. Berberich and F. Allgower, “An overview of systems-theoretic¨ guarantees in data-driven model predictive control,” Annual Review of Control, Robotics, and Auto. Systems, vol. 8, pp. 77–100, 2025.

[215] Y. Lian and C. N. Jones, “Nonlinear data-enabled prediction and control,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), ser. Proc. of Machine Learning Research, vol. 144, 2021, pp. 523–534.

[216] M. Alsalti, V. G. Lopez, J. Berberich, F. Allgower, and M. A. M¨ uller,¨ “Data-driven nonlinear predictive control for feedback linearizable systems,” in Proc. of the IFAC World Congress, vol. 56, no. 2, 2023, pp. 617–624.

[217] M. Lazar, “Basis-functions nonlinear data-enabled predictive control: Consistent and computationally efficient formulations,” in Proc. of the European Control Conference (ECC), 2024, pp. 888–893.

[218] C. Verhoek, J. Berberich, S. Haesaert, R. Toth, and H. S. Abbas, “A´ linear parameter-varying approach to data predictive control,” IEEE Trans. on Automatic Control, vol. 71, no. 4, pp. 2512–2527, 2026.

[219] J. Berberich, J. Kohler, M. A. M¨ uller, and F. Allg¨ ower, “Linear¨ tracking MPC for nonlinear systems—Part II: The data-driven case,” IEEE Trans. on Auto. Control, vol. 67, no. 9, pp. 4406–4421, 2022.

[220] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.

[221] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[222] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous methods for deep reinforcement learning,” in Proc. of the International Conference on Machine Learning (ICML), 2016, pp. 1928–1937.

[223] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust region policy optimization,” in Proc. of the International Conference on Machine Learning (ICML), 2015, pp. 1889–1897.

[224] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[225] S. Fujimoto, H. Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proc. of the International Conference on Machine Learning (ICML), 2018, pp. 1587–1596.

[226] F. Muratore, F. Ramos, G. Turk, W. Yu, M. Gienger, and J. Peters, “Robot learning from randomized simulations: A review,” Frontiers in Robotics and AI, vol. 9, p. 799893, 2022.

[227] M. Bloesch, J. Humplik, V. Patraucean, R. Hafner, T. Haarnoja, A. Byravan, N. Y. Siegel, S. Tunyasuvunakool, F. Casarini, N. Batchelor et al., “Towards real robot learning in the wild: A case study in bipedal locomotion,” in Proc. of the Conference on Robot Learning (CoRL). PMLR, 2022, pp. 1502–1511.

[228] N. Hansen, R. Jangir, Y. Sun, G. Alenya, P. Abbeel, A. A. Efros,\` L. Pinto, and X. Wang, “Self-supervised policy adaptation during deployment,” in Proc. of the International Conference on Learning Representations (ICLR), 2021.

[229] S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in Proc. of the International Conference on Machine Learning (ICML), 2019, pp. 2052–2062.

[230] B. Frauenknecht, E. Cramer, A. Eisele, P. Kruse, L. Kesper, J. Hertrampf, R. Sabirov, J. Patra, J. Berger, P. Brunzema, F. Solowjow, and S. Trimpe, “RLLBC-Lib: An Educational Code Library for

Reinforcement Learning and Learning-Based Control,” in European Workshop on Reinforcement Learning, 2026.

[231] F. P. Bejarano, L. Brunke, and A. P. Schoellig, “Safety filtering while training: Improving the performance and sample efficiency of reinforcement learning agents,” IEEE Robotics and Automation Letters, vol. 10, no. 1, pp. 788–795, 2024.

[232] L. Yang, B. Werner, M. de Sa, and A. D. Ames, “Cbf-rl: Safety filtering reinforcement learning in training with control barrier functions,” arXiv preprint arXiv:2510.14959, 2026.

[233] A. Eisele, B. Frauenknecht, F. Solowjow, and S. Trimpe, “Dyna-style safety augmented reinforcement learning: Staying safe in the face of uncertainty,” arXiv preprint arXiv:2604.25508, 2026.

[234] B. Frauenknecht, L. Kesper, D. Mayfrank, H. Hose, and S. Trimpe, “Uncertainty-aware predictive safety filters for probabilistic neural network dynamics,” in Proc. of the Reinforcement Learning Conference (RLC), 2026.

[235] M. Janner, J. Fu, M. Zhang, and S. Levine, “When to trust your model: Model-based policy optimization,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), 2019.

[236] B. Frauenknecht, A. Eisele, D. Subhasish, F. Solowjow, and S. Trimpe, “Trust the model where it trusts itself-model-based actor-critic with uncertainty-aware rollout adaption,” in Proc. of the International Conference on Machine Learning (ICML), 2024, pp. 13 973–14 005.

[237] B. Frauenknecht, D. Subhasish, F. Solowjow, and S. Trimpe, “On rollouts in model-based reinforcement learning,” in Proc. of the International Conference on Learning Representations (ICLR), 2025.

[238] K. Chua, R. Calandra, R. McAllister, and S. Levine, “Deep reinforcement learning in a handful of trials using probabilistic dynamics models,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), 2018.

[239] H. Xue, C. Pan, Z. Yi, G. Qu, and G. Shi, “Full-order sampling-based mpc for torque-level locomotion control via diffusion-style annealing,” in Proc. of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 4974–4981.

[240] R. Reiter, J. Hoffmann, D. Reinhardt, F. Messerer, K. Baumgartner,¨ S. Sawant, J. Bodecker, M. Diehl, and S. Gros, “Synthesis of¨ model predictive control and reinforcement learning: Survey and classification,” Annual Reviews in Control, vol. 61, p. 101045, 2026.

[241] J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” in Proc. of the International Conference on Machine Learning (ICML), 2017, pp. 22–31.

[242] Y. As, L. Treven, C. Sferrazza, S. Coros, and A. Krause, “Actsafe: Active exploration with safety constraints for reinforcement learning,” in Proc. of the International Conference on Learning Representations (ICLR), vol. 2025, 2025, pp. 29 752–29 776.

[243] F. Berkenkamp, M. Turchetta, A. Schoellig, and A. Krause, “Safe model-based reinforcement learning with stability guarantees,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.

[244] Y. Wang, S. S. Zhan, R. Jiao, Z. Wang, W. Jin, Z. Yang, Z. Wang, C. Huang, and Q. Zhu, “Enforcing hard constraints with soft barriers: Safe reinforcement learning in unknown stochastic environments,” in Proc. of the International Conference on Machine Learning (ICML), 2023, pp. 36 593–36 604.

[245] W. Lavanakul, J. Choi, K. Sreenath, and C. Tomlin, “Safety filters for black-box dynamical systems by learning discriminating hyperplanes,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2024, pp. 1278–1291.

[246] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in Proc. of the International Conference on Learning Representations (ICLR), 2019.

[247] N. Hansen, X. Wang, and H. Su, “Temporal difference learning for model predictive control,” in Proc. of the International Conference on Machine Learning (ICML), 2022.

[248] W. Huang, J. Ji, C. Xia, B. Zhang, and Y. Yang, “Safedreamer: Safe reinforcement learning with world models,” in Proc. of the International Conference on Learning Representations (ICLR), vol. 2024, 2024, pp. 53 839–53 869.

[249] T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, and J. Peters, “An algorithmic perspective on imitation learning,” Foundations and Trends® in Robotics, vol. 7, no. 1-2, pp. 1–179, 2018.

[250] T. L. Team, “A careful examination of large behavior models for

multitask dexterous manipulation,” Science Robotics, vol. 11, no. 113, p. eaea6201, 2026.

[251] K. Black, N. Brown, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman et al., “π<sub>0</sub>: A vision-languageaction flow model for general robot control,” in Proc. of Robotics: Science and Systems (RSS), 2025.

[252] C. Gonzalez, H. Asadi, L. Kooijman, and C. P. Lim, “Neural networks for fast optimisation in model predictive control: A review,” arXiv preprint arXiv:2309.02668, 2023.

[253] M. Hertneck, J. Kohler, S. Trimpe, and F. Allg¨ ower, “Learning¨ an approximate model predictive controller with guarantees,” IEEE Control Systems Letters, vol. 2, no. 3, pp. 543–548, 2018.

[254] H. Hose, J. Kohler, M. N. Zeilinger, and S. Trimpe, “Approximate ¨ nonlinear model predictive control with safety-augmented neural networks,” IEEE Transactions on Control Systems Technology, vol. 33, no. 6, pp. 2490–2497, 2025.

[255] X. Zhang, M. Bujarbaruah, and F. Borrelli, “Near-optimal rapid mpc using neural networks: A primal-dual policy learning framework,” IEEE Transactions on Control Systems Technology, vol. 29, no. 5, pp. 2102–2114, 2020.

[256] A. D. Bonzanini, J. A. Paulson, D. B. Graves, and A. Mesbah, “Toward safe dose delivery in plasma medicine using projected neural network-based fast approximate NMPC,” IFAC-PapersOnLine, vol. 53, no. 2, pp. 5279–5285, 2020.

[257] B. Karg and S. Lucia, “Efficient representation and approximation of model predictive control laws via deep learning,” IEEE Transactions on Cybernetics, vol. 50, no. 9, pp. 3866–3878, 2020.

[258] J. Drgona, K. Kiˇ s, A. Tuor, D. Vrabie, and M. Klauˇ co, “Differentiableˇ predictive control: Deep learning alternative to explicit model predictive control for unknown nonlinear systems,” Journal of Process Control, vol. 116, pp. 80–92, 2022.

[259] Y. Cao and R. B. Gopaluni, “Deep neural network approximation of nonlinear model predictive control,” IFAC-PapersOnLine, vol. 53, no. 2, pp. 11 319–11 324, 2020.

[260] R. Romer, A. von Rohr, and A. Schoellig, “Diffusion predictive¨ control with constraints,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2025, pp. 791–803.

[261] P. M. Julbe, J. Nubert, H. Hose, S. Trimpe, and K. J. Kuchenbecker, “Diffusion-based approximate MPC: Fast and consistent imitation of multi-modal action distributions,” in Proc. ofthe IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2025, pp. 5633–5640.

[262] M. Okamoto, J. Ren, Q. Mao, J. Liu, and Y. Cao, “Deep learningbased approximation of model predictive control laws using mixture networks,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 2909–2922, 2024.

[263] A. Reske, J. Carius, Y. Ma, F. Farshidian, and M. Hutter, “Imitation learning from mpc for quadrupedal multi-gait control,” in Proc. of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 5014–5020.

[264] H. Hose, A. Grafe, and S. Trimpe, “Parameter-adaptive approximate ¨ mpc: Tuning neural-network controllers without retraining,” in Proc. of the Annual Learning for Dynamics and Control Conference (L4DC), 2024, pp. 349–360.

[265] H. Iqbal, X. Li, T. Ingebrand, A. Thorpe, K. Kumar, U. Topcu, and J. Drgona, “Zero-shot function encoder-based differentiable predic-ˇ tive control,” in Proc. of the Learning for Dynamics and Control Conference (L4DC), 2026.

[266] L. Luken and S. Lucia, “Learning iterative solvers for accurate and¨ fast nonlinear model predictive control via unsupervised training,” in Proc. of the European Control Conf. (ECC), 2024, pp. 1843–1850.

[267] S. Lucia, D. Navarro, B. Karg, H. Sarnago, and O. Lucia, “Deep learning-based model predictive control for resonant power converters,” IEEE Transactions on Industrial Informatics, vol. 17, no. 1, pp. 409–420, 2020.

[268] J. Nubert, J. Kohler, V. Berenz, F. Allg ¨ ower, and S. Trimpe, “Safe and ¨ fast tracking on a robot manipulator: Robust MPC and neural network control,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3050–3057, 2020.

[269] H. Hose, J. Weisgerber, and S. Trimpe, “The mini wheelbot: A testbed for learning-based balancing, flips, and articulated driving,” in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 1339–1346.

[270] K. J. Astrom and T. Hagglund, “Advanced pid control,” ¨ IEEE Control Systems, vol. 26, no. 1, pp. 98–101, 2006.

[271] K. B. Ariyur and M. Krstic, Real-time optimization by extremumseeking control. John Wiley & Sons, 2003.

[272] M. C. Campi, A. Lecchini, and S. M. Savaresi, “Virtual reference feedback tuning: a direct method for the design of feedback controllers,” Automatica, vol. 38, no. 8, pp. 1337–1346, 2002.

[273] H. Hjalmarsson, “Iterative feedback tuning—an overview,” International Journal of Adaptive Control and Signal Processing, vol. 16, no. 5, pp. 373–395, 2002.

[274] P. J. Fleming and R. C. Purshouse, “Evolutionary algorithms in control systems engineering: a survey,” Control engineering practice, vol. 10, no. 11, pp. 1223–1241, 2002.

[275] R. Garnett, Bayesian Optimization. Cambridge University Press, 2023.

[276] J. A. Paulson, F. Sorourifar, and A. Mesbah, “A tutorial on derivativefree policy learning methods for interpretable controller representations,” in Proc. of the American Control Conference (ACC), 2023, pp. 1295–1306.

[277] P. Brunzema, M. Jordahn, J. Willes, S. Trimpe, J. Snoek, and J. Harrison, “Bayesian optimization via continual variational last layer training,” in Proc. of the International Conference on Learning Representations (ICLR), vol. 2025, 2025, pp. 56 480–56 509.

[278] D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek, “Scalable global optimization via local Bayesian optimization,” in Proc. of the Conference on Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[279] S. Muller, A. von Rohr, and S. Trimpe, “Local policy search with¨ bayesian optimization,” in Proc. of the Conf. on Advances in Neural Information Processing Systems (NeurIPS), 2021, pp. 20 708–20 720.

[280] F. Berkenkamp, A. P. Schoellig, and A. Krause, “Safe controller optimization for quadrotors with Gaussian processes,” in Proc. of the IEEE Intl. Conf. on Robotics and Auto. (ICRA), 2016, pp. 491–496.

[281] F. Berkenkamp, A. Krause, and A. P. Schoellig, “Bayesian optimization with safety constraints: safe and automatic parameter tuning in robotics,” Machine learning, vol. 112, no. 10, pp. 3713–3747, 2023.

[282] C. Fiedler, J. Menn, L. Kreiskother, and S. Trimpe, “On safety in safe¨ Bayesian optimization,” Trans. on Machine Learn. Research, 2024.

[283] A. G. Puigjaner, M. Prajapat, A. Carron, A. Krause, and M. N. Zeilinger, “Performance-driven constrained optimal auto-tuner for mpc,” IEEE Robotics and Automation Letters, vol. 10, no. 5, pp. 4698–4705, 2025.

[284] P. Brunzema, A. von Rohr, F. Solowjow, and S. Trimpe, “Eventtriggered time-varying bayesian optimization,” Transactions on Machine Learning Research, 2025.

[285] I. Bogunovic, J. Scarlett, and V. Cevher, “Time-varying Gaussian process bandit optimization,” in Proc. of the International Conf. on Artificial Intelligence and Statistics (AISTATS), 2016, pp. 314–323.

[286] P. Gradu and B. Recht, “Online control for adaptive tapering of medications,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2023, pp. 4067–4074.

[287] A. Bajcsy, S. Bansal, E. Bronstein, V. Tolani, and C. J. Tomlin, “An efficient reachability-based framework for provably safe autonomous navigation in unknown environments,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2019, pp. 1758–1765.

[288] A. Jain and M. Morari, “Computing the racing line using Bayesian optimization,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2020, pp. 6192–6197.

[289] J. Menn, P. Pelizzari, M. Fleps-Dezasse, and S. Trimpe, “Lipschitz safe Bayesian optimization for automotive control,” in Proc. of the IEEE Conf. on Decision and Control (CDC), 2024, pp. 3782–3788.

[290] B. Tearle, K. P. Wabersich, A. Carron, and M. N. Zeilinger, “A predictive safety filter for learning-based racing control,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 7635–7642, 2021.

[291] E. Arcari, A. Scampicchio, A. Carron, and M. N. Zeilinger, “Bayesian multi-task learning using finite-dimensional models: A comparative study,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2021, pp. 2218–2225.

[292] P. B. G. Dohmann, A. Lederer, M. Dißemond, and S. Hirche, “Distributed Bayesian online learning for cooperative manipulation,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2021, pp. 2888–2895.

[293] E. L. Zhu, Y. R. Sturz, U. Rosolia, and F. Borrelli, “Trajectory¨ optimization for nonlinear multi-agent systems using decentralized learning model predictive control,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2020, pp. 6198–6203.

[294] A. Hock and A. P. Schoellig, “Distributed iterative learning control for a team of quadrotors,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2016, pp. 4640–4646.

[295] N. Suriyarachchi, E. Noorani, F. M. Tariq, and J. S. Baras, “Multiagent deep reinforcement learning for shock wave detection and dissipation using vehicle-to-vehicle communication,” in Proc. of the IEEE Conf. on Decision and Control (CDC), 2022, pp. 4072–4077.

[296] M. J. Sorocky, S. Zhou, and A. P. Schoellig, “To share or not to share? performance guarantees and the asymmetric nature of cross-robot experience transfer,” IEEE Control Systems Letters, vol. 5, no. 3, pp. 923–928, 2021.

[297] R. Rickenbach, J. Kohler, A. Scampicchio, M. N. Zeilinger, and ¨ A. Carron, “Active learning-based model predictive coverage control,” IEEE Transactions on Automatic Control, vol. 69, no. 9, pp. 5931–5946, 2024.

[298] D. Xu, R. Aerts, P. Karamanakos, and M. Lazar, “Constraintsinformed neural-Laguerre approximation of nonlinear MPC with application in power electronics,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 7466–7471.

[299] M. Degner, R. Soloperto, M. N. Zeilinger, J. Lygeros, and J. Kohler, ¨ “Adaptive economic model predictive control for linear systems with performance guarantees,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2024, pp. 7490–7496.

[300] J. Beuchert, F. Solowjow, J. Raisch, S. Trimpe, and T. Seel, “Hierarchical event-triggered learning for cyclically excited systems with application to wireless sensor networks,” in Proc. of the IEEE Conference on Decision and Control (CDC), 2020, pp. 103–108.

[301] Z. Yuan, A. W. Hall, S. Zhou, L. Brunke, M. Greeff, J. Panerati, and A. P. Schoellig, “Safe-control-gym: A unified benchmark suite for safe learning-based control and reinforcement learning in robotics,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 142– 11 149, 2022.

[302] C. D. McKinnon and A. P. Schoellig, “Experience-based model selection to enable long-term, safe control for repetitive tasks under changing conditions,” in Proc. ofthe IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 2977–2984.

[303] K. Li, Z. Olkin, Y. Yue, and A. D. Ames, “CLF-RL: Control lyapunov function guided reinforcement learning,” IEEE Robotics and Automation Letters, vol. 11, no. 3, pp. 3230–3237, 2026.

[304] G. Bahati, R. M. Bena, M. Wilkinson, P. Mestres, R. K. Cosner, and A. D. Ames, “Risk-aware safety filters with Poisson safety functions and Laplace guidance fields,” in Proc. of the American Control Conference (ACC), 2026, pp. 110–117.

[305] Z. Feng, H. Zhang, and S. Bansal, “From words to safety: Languageconditioned safety filtering for robot navigation,” arXiv preprint arXiv:2511.05889, 2025.

[306] L. Brunke, Y. Zhang, R. Romer, J. Naimer, N. Staykov, S. Zhou,¨ and A. P. Schoellig, “Semantically safe robot manipulation: From semantic scene understanding to motion safeguards,” IEEE Robotics and Automation Letters, vol. 10, no. 5, pp. 4810–4817, 2025.

[307] Y. Song, A. Romero, M. Muller, V. Koltun, and D. Scaramuzza,¨ “Reaching the limit in autonomous racing: Optimal control versus reinforcement learning,” Science Robotics, vol. 8, no. 82, p. eadg1462, 2023.

[308] S. Weichwald, S. W. Mogensen, T. E. Lee, D. Baumann, O. Kroemer, I. Guyon, S. Trimpe, J. Peters, and N. Pfister, “Learning by doing: Controlling a dynamical system using causality, control, and reinforcement learning,” in Proc. of the NeurIPS 2021 Competitions and Demonstrations Track, 2022, pp. 246–258.

[309] J. P. How, “Benchmarks,” IEEE Control Systems Magazine, vol. 35, no. 1, pp. 6–7, 2015.

[310] M. Zeilinger, J. Sieber, A. Didier, and R. Rickenbach, “AMPyC: Advanced Model Predictive Control in Python,” 2025. [Online]. Available: https://github.com/IntelligentControlSystems/ampyc

[311] B. Frauenknecht, E. Cramer, A. Eisele, P. Kruse, L. Kesper, J. Hertrampf, R. Sabirov, J. Patra, J. Berger, P. Brunzema, F. Solowjow, and S. Trimpe, “Reinforcement Learning and Learning-Based Control Algorithm Library,” 2024. [Online]. Available: https: //github.com/Data-Science-in-Mechanical-Engineering/RLLBC

[312] S. Trimpe and P. Brunzema, “RWTHx: Reinforcement Learning,” 2025. [Online]. Available: https://www.edx.org/learn/ computer-science/rwth-aachen-university-reinforcement-learning-2

[313] ——, “RWTHx: Learning-Based Control,” 2026. [Online]. Available: https://www.edx.org/learn/computer-science/ rwth-aachen-university-learning-based-control

[314] A. P. Schoellig and S. Zhou, “Control for Robotics: From Optimal Control to Reinforcement Learning,” 2026. [Online]. Available: https://learnsyslab.github.io/cfr/

## APPENDIX

The dataset comprises 182 papers from the Learning-Based Control invited sessions at CDC (2016–2025). The paper list and author affiliations were compiled from the session index maintained by ETH Zurich<sup>1</sup>, with abstracts retrieved via the OpenAlex API and citation counts from Google Scholar. For Figure 2, a fixed set of 25 domainrelevant concept phrases was determined by an LLM scan of all papers, and searched across each paper’s title and abstract using case-insensitive matching. Semantically related phrases were then manually checked and consolidated into groups (e.g., reinforcement learning, deep reinforcement learning, and safe reinforcement learning are counted jointly). For Figure 13, for each paper, open-source code availability was determined by scraping GitHub and project-page URLs from its abstract, then querying the GitHub API for repositories referencing the paper’s arXiv ID (obtained via OpenAlex) in their README. A manual check was conducted on the existence of the repositories. Information required for Figure 4 for related venues (i.e., CDC, ACC, and ECC) was obtained from the DBLP API. All required information was obtained in May 2026.