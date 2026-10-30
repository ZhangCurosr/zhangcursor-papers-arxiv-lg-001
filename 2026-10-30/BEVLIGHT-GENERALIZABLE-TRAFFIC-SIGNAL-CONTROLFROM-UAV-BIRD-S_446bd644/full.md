# BEVLIGHT: GENERALIZABLE TRAFFIC SIGNAL CONTROLFROM UAV BIRD’S-EYE VIEWS

A PREPRINT

Maonan Wang<sup>1,2</sup>\*, Yuxin Cai<sup>3</sup>\*, Junming Liu<sup>2</sup>, Zhengyan Huang<sup>2</sup>, Yi Yu<sup>2</sup>, Ding Wang<sup>2</sup>, Yifeng Zhang<sup>4</sup>, Yirong Chen<sup>2†</sup>

<sup>1</sup> The Chinese University of Hong Kong, Shenzhen <sup>2</sup> Shanghai AI Lab <sup>3</sup> Nanyang Technological University <sup>4</sup> National University of Singapore

## ABSTRACT

Learning-based traffic signal control (TSC) has achieved promising performance, but most methods rely on structured traffic states that must be estimated from sensor observations in real deployments. Visual TSC learns signal decisions directly from images, yet existing methods depend on fixed roadside cameras and remain tied to specific intersection configurations and visual domains. UAVs offer on-demand bird’s-eye-view (BEV) observations that jointly capture intersection topology and traffic dynamics, making them a promising sensing platform for visual TSC. We present BEVLight, a generalizable TSC framework that directly maps BEV observations to signal actions. BEVLight grounds visual features onto individual lanes and composes them along the junction topology into movement and phase representations, allowing a shared policy to score candidate phases across heterogeneous intersection layouts and signal plans. To train the policy, we develop a web-based virtual teleoperation platform and collect over 5K human control decisions together with their underlying traffic states. Simulator-provided traffic states first supervise lane- and movement-level representations, after which human phase selections and rankings supervise the phase scorer. Re-rendering the recorded trajectories under diverse appearances preserves both forms of supervision and encourages robustness to visual changes. Closed-loop experiments demonstrate zero-shot generalization across traffic demands, signal plans, and intersections: BEVLight reduces average waiting time by 7–9% relative to perception-to-control visual baselines and matches or outperforms structured-state RL controllers that receive ground-truth traffic states, achieving the lowest waiting time in 12 of 16 scenarios. An offline study further shows that BEVLight transfers to real UAV observations without target-domain adaptation.

## 1 Introduction

Traffic signal control (TSC) requires decisions that adapt to evolving traffic conditions across intersections with different geometries. Recent controllers, from RL policies to LLM agents, have shown promising results, but most operate on structured traffic states such as vehicle counts, queue lengths, and vehicle positions [1, 2, 3, 4, 5, 6]. These states are readily available in simulation but must be estimated in practice, requiring additional perception pipelines whose errors propagate into control.

Visual TSC instead learns signal decisions directly from images [7, 8, 9], removing the need for explicit traffic-state estimation. Existing approaches, however, typically consider a fixed or limited set of intersections and signal plans, and obtain observations from roadside cameras that require site-specific installation and provide fixed coverage. UAVs offer an alternative: they can be deployed on demand without permanent roadside infrastructure [10, 11], and their elevated viewpoints provide bird’s-eye-view (BEV) observations that jointly capture intersection topology and traffic conditions, as illustrated in Figure 1(a).

![](images/ae735e67aba720d555ba3187ee5c8addcc0e09ea27046573e875006bf9a12d39.jpg)  
(a)

![](images/5f3e1d1fd45f3fe38771d30ed37a80a913bc8a53580b5fafcfbdeb41febbe1ed.jpg)  
(b)  
Figure 1: (a) UAV-based visual TSC: BEVLight maps BEV observations to signal phases and is trained from human demonstrations that select and rank feasible phases. (b) The three generalization dimensions considered in this work: structure, demand, and domain.

A key challenge is therefore to learn a single UAV-based visual TSC policy that generalizes across deployment conditions. As illustrated in Figure 1(b), we consider three dimensions of generalization. Structure covers junction geometry and signal configuration, from different signal plans at the same intersection to unseen intersections with different lane topologies and feasible phase sets; across these, the same action index can correspond to different movements, so the policy cannot rely on a fixed action layout. Demand reflects variations in traffic flow that produce different congestion and queue patterns at the same intersection. Domain captures appearance changes from weather, illumination, and the gap between simulated and real UAV imagery. Simulation can expose a policy to combinations of these variations that are impractical to collect at real intersections, provided the policy is not tied to any structure, demand, or appearance.

To address these challenges, we propose BEVLight, a generalizable UAV-based visual TSC framework that maps BEV observations directly to signal actions. BEVLight grounds visual features onto individual lanes and composes them along the junction topology into phase representations, so that a shared scorer ranks candidate phases across layouts and signal plans. Training such a scorer requires supervision on how competing phases should be prioritized, not only which phase is chosen. We therefore learn the policy from human demonstrations that rank all feasible phases at each decision, a preference structure that neither scalar RL rewards nor single-action labels provide. To collect them, we develop a virtual teleoperation platform in which participants control simulated intersections from a BEV view, selecting and ranking feasible phases while the platform records the traffic trajectories. Simulator-provided traffic states first supervise lane- and movement-level representations, after which human selections and rankings supervise the phase scorer. Re-rendering the recorded trajectories under diverse appearances preserves both forms of supervision and encourages appearance-invariant representations.

The contributions of our work are summarized as follows:

• We propose BEVLight, a topology-grounded framework for TSC from UAV BEV observations. By composing visual lane features into movement and phase representations, a single policy handles different intersection layouts and signal plans without retraining.

• We learn signal decisions from human demonstrations that rank all feasible phases, providing preference supervision that a shared phase scorer exploits across signal plans with different phase sets. A virtual teleoperation platform supports collecting over 5K such ranked decisions and re-rendering them under varied visual appearances.

• We demonstrate zero-shot generalization across traffic demands, signal plans, and intersections. BEVLight reduces average waiting time by 7–9% relative to perception-to-control visual baselines, and matches or outperforms structured-state RL methods that receive ground-truth traffic states, achieving the lowest waiting time in 12 of 16 scenarios. An offline study further shows transfer to real UAV videos without target-domain adaptation.

![](images/7195a1487546c24bd11ca6fb792b6a910699099d700138f65e50df51db8e3c7f.jpg)  
Figure 2: Illustration of a four-way intersection under two phase configurations, each comprising four traffic signal phases.

## 2 Preliminaries and Problem Formulation

## 2.1 Traffic Signal Control Preliminaries

Figure 2 illustrates a four-way intersection under two phase configurations.

Lane. Incoming lanes ${ \mathcal { L } } ^ { \mathrm { i n } }$ carry vehicles toward the intersection and outgoing lanes $\mathcal { L } ^ { \mathrm { { o u t } } }$ carry them away, with $\mathcal { L } = \mathcal { L } ^ { \mathrm { i n } } \cup \mathcal { L } ^ { \mathrm { o u t } }$ ; in Figure 2, $\ell _ { 1 } { - } \ell _ { 5 }$ are incoming and $\ell _ { 2 1 } - \ell _ { 2 4 }$ are outgoing.

Movement. A movement $m \subseteq \mathcal { L } ^ { \mathrm { i n } } \times \mathcal { L } ^ { \mathrm { o u t } }$ groups admissible lane connections that serve the same directional flow and share signal control, e.g., the opposing through movements $\begin{array} { r l r } { m _ { 1 } } & { { } = } & { \left\{ ( \ell _ { 2 } , \ell _ { 2 3 } ) , ( \ell _ { 3 } , \ell _ { 2 2 } ) \right\} } \end{array}$ and $\begin{array} { r c l } { m _ { 5 } } & { = } & { \{ ( \ell _ { 1 9 } , \ell _ { 6 } ) , ( \ell _ { 1 8 } , \bar { \ell } _ { 7 } ) \} } \end{array}$ . We denote the movement set by M.

Phase. A phase $p \subseteq { \mathcal { M } }$ contains non-conflicting movements that receive the right of way simultaneously, and a phase configuration defines the feasible phase set $\mathcal { P } = \{ p _ { 1 } , . . . , p _ { K } \}$ . Plan A and Plan B in Figure 2 group the same movements differently: Plan A releases $\{ m _ { 1 } , m _ { 5 } \}$ as Phase-3, whereas Plan B pairs them with other movements.

## 2.2 UAV Bird’s-Eye-View TSC

We characterize a junction by its configuration

$$
\mathcal { C } = ( \mathbf { M } _ { \mathcal { L } } , \mathcal { M } , \mathcal { P } ) , \qquad \mathbf { M } _ { \mathcal { L } } = \{ \mathbf { M } _ { \ell } : \ell \in \mathcal { L } \} ,\tag{1}
$$

where $\mathbf { M } _ { \ell } \in \{ 0 , 1 \} ^ { H _ { \mathrm { B E V } } \times W _ { \mathrm { B E V } } }$ is the binary BEV mask of lane $\ell , \mathcal { M }$ is the movement set, and $\mathcal { P }$ is the feasible phase set of the current signal plan. At decision step t, the controller receives

$$
o _ { t } = \left( \mathbf { I } _ { t } ^ { \mathrm { B E V } } , \mathcal { C } , k _ { t } , \tau _ { t } \right) , \qquad \mathbf { I } _ { t } ^ { \mathrm { B E V } } = \left( I _ { t - H + 1 } ^ { \mathrm { B E V } } , \cdot \cdot \cdot , I _ { t } ^ { \mathrm { B E V } } \right) ,\tag{2}
$$

where $\mathbf { I } _ { t } ^ { \mathrm { B E V } }$ collects the latest H UAV images, $k _ { t } \in \{ 1 , \ldots , K \}$ indexes the current phase $p _ { k _ { t } }$ , and $\tau _ { t } \geq 0$ is its elapsed duration. The policy selects a feasible target phase $a _ { t } = \pi _ { \theta } ( o _ { t } ) \in \mathcal P$ . Our goal is a single policy π<sub>θ</sub> that generalizes across junction layouts, signal plans, and traffic demands, and transfers from simulated BEV sequences to real UAV observations.

## 3 Method

BEVLight learns a shared visual policy across junction geometries and signal configurations. Since BEV observations cannot directly map to junction-specific action indices, BEVLight grounds visual features onto lanes and uses traffic topology to construct movement and phase representations with consistent semantics. As illustrated in Figure 3, the framework comprises BEV-to-lane grounding, topology-grounded composition and scoring, and human-supervised policy learning. The policy is trained from human traffic-control demonstrations together with traffic-state supervision encouraging visual representations to preserve control-relevant information.

## 3.1 Topology-Grounded BEV-to-Phase Representation

A consistent mapping from BEV observations to signal actions must account for junction-specific geometries and signal plans, under which the same action index can correspond to different movements (Figure 2). BEVLight resolves this ambiguity by grounding visual features onto individual lanes and following the junction topology to progressively compose lane representations into movement and phase representations.

![](images/f5efe8b04ac8f3926d5de89d102e2045297e2c8f3702fbea4b2c397b7817f833.jpg)  
Figure 3: Overview of BEVLight. Visual patch features are grounded onto individual lanes using lane masks with content-adaptive pooling (BEV-to-Lane Grounding). Lane representations are composed along the junction topology into movement and phase representations, and a shared scorer ranks candidate phases (Topology-Grounded Composition & Score). Simulator-provided traffic states supervise lane and movement representations, while human phase selections and rankings supervise the scorer; recorded trajectories are re-rendered under diverse appearances with unchanged supervision (Human Demonstration for Training).

BEV-to-Lane Grounding. For each $I _ { j } ^ { \mathrm { B E V } }$ in $\mathbf { I } _ { t } ^ { \mathrm { B E V } }$ , the visual encoder produces U patch features,

$$
\mathbf F _ { j } = E _ { \mathrm { v i s } } ( I _ { j } ^ { \mathrm { B E V } } ) = [ \mathbf f _ { 1 } ^ { j } , \dots , \mathbf f _ { U } ^ { j } ] ,\tag{3}
$$

where $\mathbf { f } _ { u } ^ { j }$ denotes the feature of patch u. To associate these patches with lane $\ell ,$ we derive two geometric attributes from its lane mask $\mathbf { M } _ { \ell } \colon$ the coverage coefficient $c _ { \ell , u } \in [ 0 , 1 ]$ , which measures the fraction of patch u covered by the lane, and the normalized longitudinal offset $o _ { \ell , u } = d _ { \ell , u } / L _ { \ell }$ , where $d _ { \ell , u }$ is the distance from patch u to the stop line along the lane and $L _ { \ell }$ is the lane length.

Rather than uniformly pooling the patches overlapping a lane, BEVLight assigns content-adaptive weights by combining a mask-derived geometric prior with a learned correction:

$$
a _ { \ell , u } ^ { j } = \underbrace { \mathrm { l o g } ( c _ { \ell , u } + \epsilon ) } _ { \mathrm { g e o m e t r i c ~ m a s k ~ p r i o r } } + \underbrace { \mathrm { M L P } _ { \mathrm { p o o l } } \left( \mathrm { c o n c a t } [ \mathbf { f } _ { u } ^ { j } , c _ { \ell , u } , o _ { \ell , u } ] \right) } _ { \mathrm { c o n t e n t - d e p e n d e n t ~ c o r r e c t i o n } } ,\tag{4}
$$

where $\epsilon \mathrm { ~  ~ { ~ \gamma ~ } ~ } > \mathrm { ~  ~ { ~ 0 ~ } ~ }$ ensures numerical stability. A softmax over the patches overlapping $\mathbf { M } _ { \ell }$ gives $\begin{array} { r l } { \alpha _ { \ell , u } ^ { j } } & { { } = } \end{array}$ $\mathrm { S o f t m a x } _ { u : c _ { \ell , u } > 0 } ( a _ { \ell , u } ^ { j } )$ . The frame-level lane token is then

$$
\mathbf { v } _ { \ell } ^ { j } = \sum _ { u } \alpha _ { \ell , u } ^ { j } \left( \mathbf { f } _ { u } ^ { j } + \mathrm { F o u r i e r } ( o _ { \ell , u } ) \right) .\tag{5}
$$

The longitudinal offset, therefore, influences both the adaptive pooling weights and the positional information retained in the resulting lane representation.

Given the lane tokens from consecutive BEV frames, we further model their short-term temporal evolution and interactions across lanes, producing contextualized lane representations $\widetilde { \mathbf { H } } _ { \mathcal { L } } = [ \widetilde { \mathbf { h } } _ { \ell } ^ { \mathrm { L } } ] _ { \ell \in \mathcal { L } }$ . The complete temporal and inter-lane modeling procedure is detailed in Appendix B.1.

Topology-Grounded Composition. We next propagate the lane representations through the junction topology. Three fixed binary incidence matrices derived from the junction configuration C, $\mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { i n } } , \mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { o u t } }$ , and $\mathbf { B } _ { \mathcal { M }  \mathcal { P } }$

encode incoming lane-to-movement, outgoing lane-to-movement, and movement-to-phase relations, respectively. For each movement, we first aggregate its incoming and outgoing lane context:

$$
\mathbf { H } _ { \mathcal { M } } ^ { \mathrm { i n } } = \mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { i n } } \widetilde { \mathbf { H } } _ { \mathcal { L } } , \qquad \mathbf { H } _ { \mathcal { M } } ^ { \mathrm { o u t } } = \mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { o u t } } \widetilde { \mathbf { H } } _ { \mathcal { L } } .\tag{6}
$$

For movement $m ,$ , a shared encoder combines the incoming and outgoing summaries together with their difference,

$$
\bar {  { \mathbf { h } } } _ { m } ^ { \mathrm { M } } = \mathrm { M L P } _ { M } \left( \mathrm { c o n c a t } [  { \mathbf { h } } _ { m } ^ { \mathrm { M , i n } } ,  { \mathbf { h } } _ { m } ^ { \mathrm { M , o u t } } ,  { \mathbf { h } } _ { m } ^ { \mathrm { M , i n } } -  { \mathbf { h } } _ { m } ^ { \mathrm { M , o u t } } ] \right) ,\tag{7}
$$

where the incoming–outgoing difference provides a pressure-like cue for traffic imbalance. We further contextualize the movement representations via self-attention across movements to obtain $\mathbf { h } _ { m } ^ { \mathrm { M } }$ (Appendix A.1).

We then compose the movements associated with each feasible phase according to $\mathbf { B } _ { \mathcal { M }  \mathcal { P } }$ . For phase $p ,$ let ${ \bf { H } } _ { p } = { \bf { \Psi } }$ $[ \mathbf { h } _ { m } ^ { \mathrm { M } } ] _ { m \in p } .$ A shared learnable phase query summarizes the relative composition of its constituent movements,

$$
\mathbf { c } _ { p } = \mathrm { C r o s s A t t n } \left( { \bf q } _ { P } , \mathbf { H } _ { p } , \mathbf { H } _ { p } \right) .\tag{8}
$$

Since normalized attention primarily captures relative composition, we additionally preserve aggregate traffic magnitude through an additive summary,

$$
{ \bf r } _ { p } = \log \left( { \bf 1 } + \sum _ { m \in p } \mathrm { s o f t p l u s } \left( \mathrm { M L P } _ { \mathrm { a d d } } ( { \bf h } _ { m } ^ { \mathrm { M } } ) \right) \right) .\tag{9}
$$

The final phase representation combines these complementary summaries:

$$
{ \bf h } _ { p } ^ { \mathrm { P } } = \mathrm { M L P } _ { P } \left( \mathrm { c o n c a t } [ { \bf c } _ { p } , { \bf r } _ { p } ] \right) .\tag{10}
$$

$\mathbf { h } _ { p } ^ { \mathrm { P } }$ thus encodes both the movement composition of phase $p$ and its aggregate traffic magnitude.

Candidate Phase Scoring. To accommodate different numbers of feasible phases across junctions and signal plans, BEVLight applies the same scorer to every candidate. Let $p _ { k _ { t } }$ denote the current phase and define $\Delta \mathbf { h } _ { p } ^ { \mathrm { P } } = \mathbf { \check { h } } _ { p } ^ { \mathrm { P } } - \mathbf { \check { h } } _ { p _ { k _ { t } } } ^ { \mathrm { P } }$ The score of candidate phase p is

$$
s _ { p } = \mathrm { M L P } _ { D } \left( \mathrm { c o n c a t } [ \mathbf { h } _ { p } ^ { \mathrm { P } } , \mathbf { h } _ { p _ { k _ { t } } } ^ { \mathrm { P } } , \Delta \mathbf { h } _ { p } ^ { \mathrm { P } } , \mathrm { F o u r i e r } \left( \tau _ { t } / \tau _ { \mathrm { m a x } } \right) ] \right) ,\tag{11}
$$

where $\tau _ { t }$ is the elapsed duration of the current phase and $\tau _ { \mathrm { m a x } }$ is the maximum phase duration. The policy selects the highest-scoring feasible phase, $a _ { t } = \arg \operatorname* { m a x } _ { p \in \mathcal { P } } s _ { p } .$ . Because both phase construction and candidate scoring are shared across junctions, junction-specific signal configurations are handled through the topology rather than junctionspecific output heads.

## 3.2 Human Demonstration Collection

To supervise the shared phase scorer, we collect human traffic-control demonstrations through a web-based virtual teleoperation platform. At each decision step, participants observe the evolving junction through a continuous BEV view, select the preferred phase from the feasible set $\mathbf { \dot { \mathcal { P } } } _ { }$ , and rank all candidate phases. Each demonstration therefore provides an observation $o _ { t }$ , a selected phase $a _ { t } ^ { * }$ , and a preference ranking $r _ { t } ( p )$ over $p \in \mathcal P$ . The ranking distinguishes among non-selected phases that a single action label would treat as equally undesirable.

The simulator additionally records the position of every vehicle, allowing the same traffic evolution to be re-rendered under different visual appearances while preserving the original human preferences. These recorded states also provide traffic supervision for learning control-relevant lane representations. Details of the annotation interface, demonstration protocol, data collection, and processing are provided in Appendix C.

## 3.3 Policy Training

BEVLight is trained in two stages. The first stage learns traffic-aware lane and movement representations from simulator-provided supervision, while the second stage learns the signal-control policy from human demonstrations.

Stage I: Traffic-Aware Representation Pretraining. We attach auxiliary prediction heads to the lane and movement representations to supervise control-relevant traffic information. At the lane level, the model predicts queue length, vehicle count, and vehicle positions along the lane; at the movement level, it predicts queue length and vehicle count. The first-stage objective is

$$
\mathcal { L } _ { \mathrm { s t a g e } 1 } = \left( \mathcal { L } _ { q } ^ { \mathcal { L } } + \mathcal { L } _ { n } ^ { \mathcal { L } } + \mathcal { L } _ { \mathrm { C h } } ^ { \mathcal { L } } \right) + \left( \mathcal { L } _ { q } ^ { \mathcal { M } } + \mathcal { L } _ { n } ^ { \mathcal { M } } \right) .\tag{12}
$$

This supervision encourages the learned representations to retain traffic information relevant to downstream signal decisions. Detailed definitions of Stage I are provided in Appendix A.2.

![](images/3f3c4044db66ccf42bab24faec88b5683dbbb830779ae2563c44ec5ad19d0a83.jpg)  
Figure 4: Eight junctions, each with two signal plans. Blue borders mark plans used for training, while purple borders mark held-out plans.

Stage II: Human-Supervised Policy Learning. Starting from the pretrained representations, we train the phase scorer using both the human-selected phase and the ranking over all feasible candidates. Let $\pi _ { t }$ denote the policy distribution obtained by softmax-normalizing the candidate scores in Equation 11. We construct a soft ranking target $\mathbf { y } _ { t }$ from the human ranking $r _ { t } ( p )$ following the top-one listwise formulation of ListNet [12]. With $\mathbf { e } _ { a _ { t } ^ { * } }$ denoting the one-hot target for the selected phase, the second-stage objective is

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathrm { C E } ( \mathbf { e } _ { a _ { t } ^ { * } } , \pmb { \pi } _ { t } ) + \lambda _ { \mathrm { s o f t } } \mathrm { C E } ( \mathbf { y } _ { t } , \pmb { \pi } _ { t } ) .\tag{13}
$$

The first term learns the human-selected phase, while the second preserves the relative preferences among all feasible candidates. The construction of the ranking target and the individual training objectives are detailed in Appendix A.2.

## 4 Experiments

## 4.1 Experimental Setup

Scenarios. We conduct simulation experiments in TranSimHub [13], which uses SUMO [14] to simulate vehicle movements and provides an interface for rendering BEV observations (Appendix B.1). Figure 4 shows the eight junctions used in our experiments. Each junction has two signal plans, A and B, which share the same layout but differ in their phase configurations (Appendix B.2). We consider five traffic demand patterns (Appendix B.3) and evaluate three cumulative generalization settings by progressively introducing unseen demand patterns, signal plans, and junctions. For offline simulation-to-real evaluation, we use real UAV observations from the Songdo dataset [11].

Compared Methods and Metrics. We compare BEVLight with two conventional controllers, FixedTime and Max-Pressure [15], and four structured-state RL controllers: UniTSA [4], IntelliLight [1], AttendLight [16], and MP-Light [3]. We also include two visual perception-to-control baselines. YOLO-based Count (YOLO-C) assigns detected vehicles to lanes and uses the resulting lane-level counts as inputs to AttendLight, while BEVLight-State (BEV-S) feeds BEVLight-estimated lane states into AttendLight. BEVLight directly predicts a signal phase from BEV observations and junction topology. We report average vehicle waiting time, where lower values indicate better performance. Further details on the compared methods are provided in Appendix B.5.

## 4.2 Closed-Loop Generalization

Table 1 reports closed-loop average vehicle waiting time under the three generalization settings. BEVLight achieves the lowest waiting time in 12 of 16 evaluation scenarios, demonstrating that human-supervised control from visual observations can match or even outperform RL-based TSC methods supplied with ground-truth traffic states. Human demonstrations provide more than a single preferred action. They rank feasible phases, expressing relative preferences among alternatives with different traffic conditions, as illustrated in Appendix C.3. This richer supervision guides the policy in comparing candidate phases rather than only reproducing the selected action.

Table 1: Closed-loop average vehicle waiting time (s) across three generalization settings. Values are mean ± sample standard deviation over the two held-out demand patterns. Lower is better. <sup>∗</sup>, <sup>†</sup>, and <sup>‡</sup> denote the first-, second-, and third-best results in each row, respectively.
<table><tr><td></td><td colspan="2">Conventional</td><td colspan="6">Structured-State RL</td><td colspan="6">Vision-Based</td></tr><tr><td>Junction / Plan FixedTime MaxPressure</td><td></td><td></td><td>UniTSA</td><td></td><td>IntelliLight</td><td>AttendLight</td><td></td><td>MPLight</td><td>YOLO-C</td><td></td><td>BEV-S</td><td></td><td>BEVLight</td><td></td></tr><tr><td colspan="9">Unseen Demand</td><td colspan="6"></td></tr><tr><td>Beishahe A</td><td>18.64 ± 0.39</td><td>8.02 ± 1.87</td><td>11.67 ± 0.62</td><td></td><td> $7 . 4 5 ^ { \dagger } \pm 0 . 0 6$ </td><td>8.08 ±0.68</td><td></td><td>8.26±1.24</td><td>8.01 ± 2.07</td><td></td><td>7.73</td><td>± 0.68</td><td>7.40*</td><td>± 0.23</td></tr><tr><td>Beishahe B</td><td>18.30 ± 1.24</td><td>8.01 ± 1.81</td><td> $1 0 . 2 1 \pm 1 . 3 8$ </td><td></td><td> $6 . 6 8 ^ { \ddagger } \pm 1 . 3 2$ </td><td>6.83 ± 1.22</td><td></td><td>6.67† ± 0.93</td><td>7.91 ± 1.76</td><td></td><td>7.57 ± 2.07</td><td></td><td>6.62*</td><td>± 0.60</td></tr><tr><td>Gaojiaoyuan A 26.09 ± 3.99</td><td></td><td>10.20 ± 2.78</td><td>13.04±4.15</td><td></td><td> $1 0 . 5 5 \pm 3 . 2 5$ </td><td>10.28 ± 4.90</td><td></td><td>10.71 ± 4.38</td><td>9.55† ± 4.06</td><td></td><td>10.08‡ ± 4.45</td><td></td><td>9.12*</td><td>± 2.31</td></tr><tr><td>Gaojiaoyuan B</td><td>25.12 ± 3.01</td><td>10.37 ± 1.38</td><td>11.69 ±4.13</td><td></td><td>9.34 ± 0.43</td><td>9.18* ±2.06</td><td></td><td>10.18 ±1.81</td><td>9.24† ± 0.25</td><td></td><td>10.07 ± 0.85</td><td></td><td>9.24†</td><td>±1.21</td></tr><tr><td>Zhijingdao A</td><td>25.65 ± 0.66</td><td>10.21 ± 0.90</td><td>11.89 ± 2.05</td><td>9.51 ± 1.09</td><td></td><td>9.42†</td><td>± 0.56</td><td>9.61 ± 0.71</td><td>9.79± 0.07</td><td></td><td>9.95 ± 1.00</td><td></td><td>8.89*</td><td>± 0.50</td></tr><tr><td>Yau Ma Tei A</td><td>21.00 ± 0.11</td><td>7.45 ± 1.03</td><td>8.66 ± 1.20</td><td></td><td>6.60‡ ± 0.28</td><td>6.39†</td><td>± 0.58</td><td>6.95 ± 0.68</td><td>6.75 ± 1.06</td><td></td><td>7.04 ± 0.37</td><td></td><td>6.09*</td><td>± 0.74</td></tr><tr><td>Pinganli A</td><td>25.21 ± 0.05 10.23† ± 0.14</td><td></td><td>13.80 ± 0.98</td><td></td><td>11.14 ± 2.33</td><td>10.44‡</td><td>± 1.01</td><td>11.08±1.27</td><td>10.52 ± 0.81</td><td></td><td>10.75 ± 0.80</td><td></td><td>10.10*</td><td>± 0.84</td></tr><tr><td>Yongrunlu A</td><td>27.27 ± 2.06</td><td>10.85 ± 0.06</td><td>12.14 ± 0.10</td><td>10.29‡</td><td>± 0.08</td><td>10.24†</td><td>± 0.88</td><td>10.75 ± 0.67</td><td>10.45 ± 0.47</td><td></td><td>10.41 ± 0.84</td><td></td><td>9.54*</td><td>± 0.46</td></tr><tr><td colspan="9">Unseen Signal Plan + Demand</td><td colspan="6"></td></tr><tr><td>Zhijingdao B</td><td>26.22 ± 5.35</td><td> $1 0 . 1 3 ^ { \dagger } \ \pm 1 . 1 1$ </td><td>12.62 ± 1.76</td><td></td><td>9.96* ± 1.13</td><td>10.31 ± 0.60</td><td></td><td>11.00 ± 0.37</td><td>10.44 ± 0.28</td><td></td><td>10.60 ± 0.19</td><td></td><td>10.09†</td><td>± 1.14</td></tr><tr><td>Yau Ma Tei B</td><td>36.86± 1.51</td><td>21.21 ± 6.29</td><td>24.68 ± 7.49</td><td>21.40 ± 11.05</td><td></td><td>20.02‡</td><td>±7.85</td><td>19.44† ± 3.71</td><td>20.84± 5.00</td><td></td><td>21.30 ± 9.20</td><td></td><td>16.52*</td><td>±4.67</td></tr><tr><td>Pinganli B</td><td>25.62 ± 0.63</td><td>9.86± 0.78</td><td>11.29 ± 0.69</td><td></td><td>10.40 ± 1.14</td><td>9.65* ± 0.64</td><td></td><td>9.81 ± 0.13</td><td>9.74 ± 0.69</td><td></td><td>10.24± 1.39</td><td></td><td>9.73†</td><td>± 0.56</td></tr><tr><td>Yongrunlu B</td><td>25.37 ± 3.07</td><td>11.14 ± 1.13</td><td>11.53 ± 0.08</td><td>10.59 ± 0.22</td><td></td><td>10.24 ± 0.95</td><td></td><td>10.51 ± 0.18</td><td>10.23</td><td>± 0.23 10.22†</td><td></td><td>± 0.75 10.08*</td><td></td><td>± 0.15</td></tr><tr><td colspan="9">Unseen Junction + Plan + Demand</td><td colspan="6"></td></tr><tr><td>Guanghua A</td><td>22.49 ± 1.03</td><td>9.19 ± 1.01</td><td>13.39 ± 5.87</td><td></td><td>8.91 ± 0.04</td><td>8.48†</td><td>± 0.27</td><td>8.85 ± 1.15</td><td>8.84±0.98</td><td></td><td>9.59 ± 1.14</td><td></td><td>7.95*</td><td>± 0.18</td></tr><tr><td>Guanghua B</td><td>23.69 ± 0.23</td><td>9.12 ± 1.15</td><td>12.38 ± 0.51</td><td></td><td>8.51* ± 0.45</td><td>9.02† ± 0.94</td><td></td><td>9.31 ± 0.22</td><td>9.38 ± 0.22</td><td></td><td>10.04 ± 0.90</td><td></td><td>9.04</td><td>± 0.62</td></tr><tr><td>Songdo A</td><td>30.16± 0.58</td><td>13.38† ± 0.21</td><td>17.23 ± 3.93</td><td></td><td> $1 3 . 4 7 ^ { \ddagger } \ \pm 0 . 1 3$ </td><td></td><td> $1 3 . 8 7 \pm 1 . 3 1$ </td><td>14.93 ± 1.80</td><td>14.46± 1.09</td><td></td><td>14.31 ± 0.75</td><td></td><td>13.08*</td><td>± 1.14</td></tr><tr><td>Songdo B</td><td>30.28 ± 0.90</td><td>19.47 ± 5.32</td><td>18.52† ± 2.64</td><td></td><td>19.45 ± 7.67</td><td></td><td> $1 9 . 1 6 \pm 7 . 9 3$ </td><td>23.43 ± 9.87</td><td>19.43 ± 7.16</td><td></td><td></td><td>19.15‡ ± 4.83 18.10*</td><td></td><td>± 7.13</td></tr></table>

Unseen demand. BEVLight achieves the lowest waiting time in seven of eight scenarios and closely matches AttendLight on Gaojiaoyuan B (9.24 s versus 9.18 s). Despite receiving accurate simulator-provided quantities such as queue lengths, the structured-state RL methods do not consistently outperform the human-supervised visual policy. This suggests that the information available to the controller and how it is used to prioritize phases both matter for control performance. Visual observations retain vehicle positions and spatial queue patterns, while human rankings supervise how the competing phases should be prioritized. Averaged across this setting, BEVLight reduces waiting time by 7.2% relative to YOLO-C and 9.0% relative to BEV-S.

Unseen signal plans and demand. Across signal plans, the same action index can correspond to different movement combinations, making fixed preferences over action indices insufficient. BEVLight compares phases through the traffic served by their constituent movements, using the topology-grounded representation. On Yau Ma Tei B, BEVLight reduces waiting time from 19.44 s for MPLight, the strongest baseline in this scenario, to 16.52 s, a 15.0% improvement. On Zhijingdao B and Pinganli B, it remains within 0.13 s and 0.08 s of the best baseline, respectively.

Unseen junctions, signal plans, and demand. Finally, we evaluate BEVLight under jointly unseen junction layouts, signal plans, and demand patterns. BEVLight achieves the lowest waiting time on Guanghua A, Songdo A, and Songdo B, while remaining close to AttendLight on Guanghua B (9.04 s versus 9.02 s). It also outperforms the twostage visual baselines in all four scenarios, with average reductions of 7.6% over YOLO-C and 9.3% over BEV-S. This advantage suggests that directly learning decision-relevant visual cues may alleviate the propagation of intermediate state-estimation errors into control decisions. These results demonstrate that BEVLight can transfer phase preferences learned from human rankings across changes in demand, signal plans, and junction layouts, achieving competitive closed-loop control from visual observations even against RL-based methods using ground-truth traffic states.

![](images/e2d539cf8361e973422eea1406329b2f4856cd9c4916d434bebe7f477046616a.jpg)

![](images/d22f026806a0a09d68f9777b6eb0c9288e294c66072c5ba44e2dc0f403d1488c.jpg)  
Songdo, H Junction

Figure 5: Phase predictions on real UAV imagery from Songdo intersections.

## 4.3 Ablation Study

Table 2 examines the contributions of adaptive pooling, visual pretraining, and soft ranking at Guanghua and Songdo.   
Combining all three components consistently improves control performance across these unseen settings.

Adaptive pooling alone yields mixed results, suggesting that its additional parameters are difficult to learn from action supervision without explicit perception guidance. Adding visual pretraining improves performance across all four settings, supporting the importance of perception supervision for learning useful lane-level features. Soft 6 T=21ranking further reduces waiting time in all scenarios. These modest gains are consistent with its role in capturing relative phase preferences when multiple actions are plausible, providing richer guidance than a single preferred action.

Table 2: Ablation results in average waiting time (s).
<table><tr><td colspan="3">Components</td><td colspan="2">Guanghua</td><td colspan="2">Songdo</td></tr><tr><td>Adaptive Pooling</td><td>Visual Pretraining</td><td>Soft Loss</td><td>A</td><td>B</td><td>A</td><td>B</td></tr><tr><td>X</td><td>X</td><td>X</td><td>9.20</td><td>10.73</td><td>13.96</td><td>23.79</td></tr><tr><td>√</td><td>X</td><td>X</td><td>9.38</td><td>10.91</td><td>13.08</td><td>29.03</td></tr><tr><td>√</td><td>V</td><td>X</td><td>8.78</td><td>9.12</td><td>12.86</td><td>18.20</td></tr><tr><td>√</td><td>√</td><td>√</td><td>7.95</td><td>9.04</td><td>12.77</td><td>18.10</td></tr></table>

## 4.4 Offline Simulation-to-Real Case Study

We evaluate BEVLight on recorded UAV imagery from four real intersections in Songdo. Figure 5 illustrates BEV-Light’s zero-shot transfer to real-world traffic scenes through plausible phase decisions without real-world fine-tuning.

At Junction O, BEVLight selects the east-west through phase at $T = 3 0 \mathrm { ~ s } . \mathrm { ~ B y ~ } T = 3 3 \mathrm { ~ s } ,$ after east-west traffic has discharged, the model favors the north-south through phase, shifting priority to the remaining demand. Similarly, at Junction H, the model initially favors east-west through movements and then switches to the west-approach left-turn movement as through traffic subsides. These predictions qualitatively align with human control intuition, suggesting that the learned phase preferences remain meaningful under real-world visual conditions.

## 4.5 Inference Time

We evaluate inference efficiency on an NVIDIA RTX 4090 with batch size 1. As shown in Table 3, BEVLight contains only 1.26M parameters, approximately 53% fewer than YOLO11n. Its full inference latency is 8.59 ms, well below the 10 s control interval, indicating sufficient inference speed on the evaluated hardware.

BEVLight’s compact design reflects its focus on control-relevant representations: it scores feasible phases from latent traffic features without reconstructing individual vehicle locations during inference. Although its FLOPs exceed those of YOLO11n, its latency is approximately 53% lower, highlighting that operation counts alone do not determine runtime.

Table 3: Model size and full inference latency.
<table><tr><td colspan="3">Params</td><td rowspan="2">Latency (ms) ↓</td></tr><tr><td>Method</td><td>(M)</td><td>GFLOPs</td></tr><tr><td>YOL011n</td><td>2.662</td><td>26.74</td><td>18.44</td></tr><tr><td>BEV-S</td><td>1.260</td><td>37.61</td><td>8.60</td></tr><tr><td>BEVLight</td><td>1.260</td><td>37.61</td><td>8.59</td></tr></table>

## 5 Related Work

Learning-based TSC has advanced from RL policies [2, 17, 3, 18, 19] to LLM agents [5, 6, 20, 21]. Despite their different architectures, most methods rely on structured traffic states, such as vehicle counts, queue lengths, speeds, and pressure [22]. These states are directly available in simulation but require estimation from sensor data in real deployments, introducing errors that complicate transfer. Within this setting, generalizable policies transfer across traffic demands, road topologies, and signal configurations [16, 23, 4, 5, 20], while demonstration-based methods reduce trial-and-error training using data from conventional controllers, virtual experts, or traffic engineers [24, 25, 26, 27, 28]. These advances, however, remain grounded in structured states rather than learning control from images.

Vision-based TSC maps images to signal decisions without predefined traffic states. Existing studies explore endto-end RGB control, synthetic visual data, and vision-language reasoning [7, 8, 9], demonstrating the potential of visual control but generally considering a fixed or limited set of junction and signal configurations. UAVs offer a flexible platform for wide-area traffic observation [10], yet existing studies primarily focus on traffic monitoring and trajectory extraction rather than direct signal control [11]. BEVLight learns a shared visual TSC policy from human demonstrations on simulated UAV observations, generalizes across heterogeneous junction geometries and signal configurations, and exhibits plausible zero-shot phase decisions on real UAV imagery.

## 6 Conclusion

We presented BEVLight, a visual traffic signal control framework that maps UAV bird’s-eye-view observations directly to signal actions. BEVLight grounds visual features onto lanes and composes them along the junction topology into phase representations, so that a single shared scorer ranks candidate phases across intersections with different layouts and signal plans. The scorer is trained from human demonstrations that select and rank feasible phases, on top of representations pretrained with simulator-provided traffic states and re-rendered under diverse appearances. In closed-loop experiments, BEVLight generalizes zero-shot to unseen demands, signal plans, and junctions, matching or outperforming structured-state RL methods that receive ground-truth traffic states, and transfers to real UAV observations without target-domain adaptation. Future work includes robustness to UAV altitude and viewpoint changes and coordinated control over multiple junctions.

## References

[1] Hua Wei, Guanjie Zheng, Huaxiu Yao, and Zhenhui Li. Intellilight: A reinforcement learning approach for intelligent traffic light control. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2496–2505, 2018.

[2] Hua Wei, Chacha Chen, Guanjie Zheng, Kan Wu, Vikash Gayah, Kai Xu, and Zhenhui Li. Presslight: Learning max pressure control to coordinate traffic signals in arterial network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1290–1298, 2019.

[3] Chacha Chen, Hua Wei, Nan Xu, Guanjie Zheng, Ming Yang, Yuanhao Xiong, Kai Xu, and Zhenhui Li. Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pages 3414–3421, 2020.

[4] Maonan Wang, Xi Xiong, Yuheng Kan, Chengcheng Xu, and Man-On Pun. UniTSA: A universal reinforcement learning framework for v2x traffic signal control. IEEE Transactions on Vehicular Technology, 73(10):14354– 14369, 2024.

[5] Siqi Lai, Zhao Xu, Weijia Zhang, Hao Liu, and Hui Xiong. LLMlight: Large language models as traffic signal control agents. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 2335–2346, 2025.

[6] Zirui Yuan, Siqi Lai, and Hao Liu. CoLLMLight: Cooperative large language model agents for network-wide traffic signal control. In International Conference on Learning Representations, volume 2026, pages 66687– 66715, 2026.

[7] Deepeka Garg, Maria Chli, and George Vogiatzis. Fully-autonomous, vision-based traffic signal control: From simulation to reality. In Proceedings ofthe 21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’22, page 454–462, Richland, SC, 2022. International Foundation for Autonomous Agents and Multiagent Systems.

[8] Tiejin Chen, Prithvi Shirke, Bharatesh Chakravarthi, Arpitsinh Vaghela, Longchao Da, Duo Lu, Yezhou Yang, and Hua Wei. SynTrac: a synthetic dataset for traffic signal control from traffic monitoring cameras. In 2024

IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), pages 2386–2391. IEEE, 2024.

[9] Maonan Wang, Yirong Chen, Aoyu Pang, Yuxin Cai, Chung Shue Chen, Yuheng Kan, and Man On Pun. VLM-Light: Safety-critical traffic signal control via vision-language meta-control and dual-branch reasoning architecture. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 39590–39621. Curran Associates, Inc., 2025.

[10] Hamid Menouar, Ismail Guvenc, Kemal Akkaya, A Selcuk Uluagac, Abdullah Kadri, and Adem Tuncer. Uavenabled intelligent transportation systems for the smart city: Applications and challenges. IEEE Communications Magazine, 55(3):22–28, 2017.

[11] Robert Fonod, Haechan Cho, Hwasoo Yeo, and Nikolas Geroliminis. Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery. Transportation Research Part C: Emerging Technologies, 178:105205, 2025.

[12] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. Learning to rank: from pairwise approach to listwise approach. In Proceedings of the 24th international conference on Machine learning, pages 129–136, 2007.

[13] Maonan Wang, Yirong Chen, Yuxin Cai, Aoyu Pang, Yuejiao Xie, Zian Ma, Chengcheng Xu, Kemou Jiang, Ding Wang, Laurent Roullet, et al. TranSimHub: A unified air-ground simulation platform for multi-modal perception and decision-making. arXiv preprint arXiv:2510.15365, 2025.

[14] Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann, Yun-Pang Fl”otter”od, Robert Hilbrich, Leonhard L”ucken, Johannes Rummel, Peter Wagner, and Evamarie Wiessner. Microscopic traffic simulation using SUMO. In 2018 21st International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2018.

[15] Pravin Varaiya. Max pressure control of a network of signalized intersections. Transportation Research Part C: Emerging Technologies, 36:177–195, 2013.

[16] Afshin Oroojlooy, Mohammadreza Nazari, Davood Hajinezhad, and Jorge Silva. Attendlight: Universal attention-based reinforcement learning model for traffic signal control. Advances in Neural Information Processing Systems, 33:4079–4090, 2020.

[17] Hua Wei, Nan Xu, Huichu Zhang, Guanjie Zheng, Xinshi Zang, Chacha Chen, Weinan Zhang, Yanmin Zhu, Kai Xu, and Zhenhui Li. Colight: Learning network-level cooperation for traffic signal control. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1913–1922, 2019.

[18] Liang Zhang, Qiang Wu, Jun Shen, Linyuan Lu, Bo Du, and Jianqing Wu. Expression might be enough: Repre-¨ senting pressure and demand for reinforcement learning based traffic signal control. In International Conference on Machine Learning, pages 26645–26654. PMLR, 2022.

[19] Maonan Wang, Yirong Chen, Yuheng Kan, Chengcheng Xu, Michael Lepech, Man-On Pun, and Xi Xiong. Traffic signal cycle control with centralized critic and decentralized actors under varying intervention frequencies. IEEE Transactions on Intelligent Transportation Systems, 25(12):20085–20104, 2024.

[20] Xingchen Zou, Yuhao Yang, Zheng Chen, Xixuan Hao, Yiqi Chen, Chao Huang, and Yuxuan Liang. Traffic-R1: Reinforced llms bring human-like reasoning to traffic signal control systems. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21823–21838, 2026.

[21] Aoyu Pang, Maonan Wang, Man-On Pun, Chung Shue Chen, and Xi Xiong. iLLM-TSC: Integration reinforcement learning and large language model for traffic signal control policy improvement. IEEE Transactions on Vehicular Technology, 75(8):15762–15776, 2026.

[22] Hua Wei, Guanjie Zheng, Vikash Gayah, and Zhenhui Li. A survey on traffic signal control methods. arXiv preprint arXiv:1904.08117, 2019.

[23] Franc¸ois-Xavier Devailly, Denis Larocque, and Laurent Charlin. Model-based graph reinforcement learning for inductive traffic signal control. IEEE Open Journal ofIntelligent Transportation Systems, 5:238–250, 2024.

[24] Yuanhao Xiong, Guanjie Zheng, Kai Xu, and Zhenhui Li. Learning traffic signal control from demonstrations. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 2289–2292, 2019.

[25] Min Wang, Libing Wu, Jianxin Li, Dan Wu, and Chao Ma. Urban traffic signal control with reinforcement learning from demonstration data. In 2022 International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2022.

[26] Yutong Ye, Yingbo Zhou, Jiepin Ding, Ting Wang, Mingsong Chen, and Xiang Lian. Initlight: Initial model generation for traffic signal control using adversarial inverse reinforcement learning. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 4949–4958, 2023.

[27] Xiaoshuang Li, Fenghua Zhu, and Fei-Yue Wang. Deep behavioral cloning for traffic control with virtual expert demonstration under a parallel learning framework. IFAC-PapersOnLine, 53(5):176–181, 2020.

[28] Xiaoshuang Li, Peijun Ye, Junchen Jin, Fenghua Zhu, and Fei-Yue Wang. Data augmented deep behavioral cloning for urban traffic control operations under a parallel learning framework. IEEE Transactions on Intelligent Transportation Systems, 23(6):5128–5137, 2022.

[29] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stablebaselines3: Reliable reinforcement learning implementations. Journal ofmachine learning research, 22(268):1– 8, 2021.

[30] Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024.

In this appendix, we provide supplementary material for BEVLight, covering implementation and training details, experimental settings, human demonstration collection, additional experimental results, and broader discussion. The appendix is organized as follows:

• Appendix A provides implementation and training details of BEVLight.

• Appendix B describes the experimental setup and comparison methods.

• Appendix C describes the collection and processing of human traffic-control demonstrations.

• Appendix D presents additional real-world case studies.

• Appendix E discusses practical considerations, broader impact, and limitations of BEVLight.

## A BEVLight Model Architecture and Algorithm

This section provides additional details of the BEVLight architecture and training objectives introduced in Sec. 3. We first describe the complete computation flow of the topology-grounded BEV-to-phase representation, and then elaborate on the traffic-aware and human-supervised objectives used in the two-stage training procedure.

## A.1 BEVLight Architecture

This subsection provides the complete architectural formulation of the topology-grounded BEV-to-phase representation introduced in Sec. 3.1. Starting from a sequence of BEV observations, BEVLight first grounds visual patch features onto individual lanes, models their temporal evolution and interactions across lanes, and then follows the junction topology to progressively construct movement and phase representations. The resulting candidate phases are finally evaluated by a shared scoring function.

BEV-to-Lane Grounding. For each frame $I _ { j } ^ { \mathrm { B E V } }$ in the observation sequence $\mathbf { I } _ { t } ^ { \mathrm { B E V } }$ , the visual encoder produces U patch features,

$$
\mathbf F _ { j } = E _ { \mathrm { v i s } } ( I _ { j } ^ { \mathrm { B E V } } ) = [ \mathbf f _ { 1 } ^ { j } , \dots , \mathbf f _ { U } ^ { j } ] ,\tag{14}
$$

where $\mathbf { f } _ { u } ^ { j }$ denotes the feature of patch u.

For each lane ℓ, its mask $\mathbf { M } _ { \ell }$ defines the geometric correspondence between the lane and the visual patches. We characterize each lane–patch pair using two quantities. The coverage coefficient $c \ell , u \in [ 0 , 1 ]$ measures the fraction of patch u covered by $\mathbf { M } _ { \ell } .$ , while the normalized longitudinal offset

$$
o _ { \ell , u } = \frac { d _ { \ell , u } } { L _ { \ell } }\tag{15}
$$

specifies its position along the lane, where $d _ { \ell , u }$ is the patch-to-stop-line offset and $L _ { \ell }$ is the lane length.

Rather than assigning fixed weights according to the lane mask, BEVLight predicts a content-dependent correction for each lane–patch pair. Specifically, a two-layer MLP takes the patch feature together with its geometric attributes and combines the resulting correction with a mask-derived geometric prior:

$$
a _ { \ell , u } ^ { j } = \underbrace { \log ( c _ { \ell , u } + \epsilon ) } _ { \mathrm { g e o m e t r i c ~ m a s k ~ p r i o r } } + \underbrace { \mathrm { M L P } _ { \mathrm { p o o l } } \left( \mathrm { c o n c a t } [ \mathbf { f } _ { u } ^ { j } , c _ { \ell , u } , o _ { \ell , u } ] \right) } _ { \mathrm { c o n t e n t - d e p e n d e n t ~ c o r r e c t i o n } } ,\tag{16}
$$

where $\epsilon > 0$ ensures numerical stability. The scores are normalized only over patches that overlap with lane ℓ,

$$
\begin{array} { r } { \alpha _ { \ell , u } ^ { j } = \mathrm { S o f t m a x } _ { u : c _ { \ell , u } > 0 } \left( a _ { \ell , u } ^ { j } \right) . } \end{array}\tag{17}
$$

We then aggregate the position-aware patch features into a frame-level lane token,

$$
\mathbf { v } _ { \ell } ^ { j } = \sum _ { u } \alpha _ { \ell , u } ^ { j } \left( \mathbf { f } _ { u } ^ { j } + \mathrm { F o u r i e r } ( o _ { \ell , u } ) \right) .\tag{18}
$$

The longitudinal offset therefore influences the lane representation in two complementary ways: it conditions the adaptive pooling weights in Eq. 16, and its Fourier encoding preserves within-lane positional information in the aggregated feature.

Temporal and Inter-Lane Modeling. Applying the grounding operation to all H frames gives a token history $[ \mathbf { v } _ { \ell } ^ { t - \tilde { H } + 1 } , \ldots , \mathbf { v } _ { \ell } ^ { t } ]$ for each lane ℓ. To explicitly expose short-term traffic changes, we compute the difference between adjacent lane tokens,

$$
\Delta \mathbf { v } _ { \ell } ^ { j } = \mathbf { v } _ { \ell } ^ { j } - \mathbf { v } _ { \ell } ^ { j - 1 } , \qquad j = t - H + 2 , \ldots , t ,\tag{19}
$$

and set $\Delta \mathbf { v } _ { \ell } ^ { t - H + 1 } = \mathbf { 0 }$

A temporal Transformer independently summarizes the history of each lane:

$$
\begin{array} { r } { \mathbf { h } _ { \ell } ^ { \mathrm { L } } = \mathrm { T r a n s f o r m e r } _ { \mathrm { t e m p } } \left( [ \mathrm { c o n c a t } [ \mathbf { v } _ { \ell } ^ { j } , \Delta \mathbf { v } _ { \ell } ^ { j } ] ] _ { j = t - H + 1 } ^ { t } \right) . } \end{array}\tag{20}
$$

Collecting the temporal representations of all lanes gives

$$
\mathbf { H } _ { \mathcal { L } } = [ \mathbf { h } _ { \ell } ^ { \mathrm { L } } ] _ { \ell \in \mathcal { L } } .\tag{21}
$$

We subsequently apply multi-head self-attention across lanes,

$$
\widetilde { \bf H } _ { \mathcal { L } } = \mathrm { S e l f A t t n } _ { L } \left( { \bf H } _ { \mathcal { L } } \right) ,\tag{22}
$$

allowing each lane to incorporate information from other traffic streams at the junction. The row $\widetilde { \mathbf { h } } _ { \ell } ^ { \mathrm { L } }$ of $\widetilde { \mathbf { H } } _ { \mathcal { L } }$ denotes the resulting contextualized representation of lane ℓ.

Lane-to-Movement Composition. We next propagate the contextualized lane representations through the junction topology. The incoming and outgoing lane-to-movement relations are encoded by two fixed binary incidence matrices, $\mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { { \bar { m } } } }$ and $\mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { o u t } } ,$ respectively. Each entry indicates whether a lane belongs to the incoming or outgoing side of a movement.

The corresponding movement-level summaries are obtained through

$$
\mathbf { H } _ { \mathcal { M } } ^ { \mathrm { i n } } = \mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { i n } } \widetilde { \mathbf { H } } _ { \mathcal { L } } , \qquad \mathbf { H } _ { \mathcal { M } } ^ { \mathrm { o u t } } = \mathbf { B } _ { \mathcal { L }  \mathcal { M } } ^ { \mathrm { o u t } } \widetilde { \mathbf { H } } _ { \mathcal { L } } .\tag{23}
$$

For movement $m ,$ , the rows $\mathbf { h } _ { m } ^ { \mathrm { M , i n } }$ and $\mathbf { h } _ { m } ^ { \mathrm { M , o u t } }$ summarize its incoming and outgoing lane context.

A shared MLP combines these two summaries with their difference:

$$
\bar {  { \mathbf { h } } } _ { m } ^ { \mathrm { M } } = \mathrm { M L P } _ { M } \left( \mathrm { c o n c a t } [  { \mathbf { h } } _ { m } ^ { \mathrm { M , i n } } ,  { \mathbf { h } } _ { m } ^ { \mathrm { M , o u t } } ,  { \mathbf { h } } _ { m } ^ { \mathrm { M , i n } } -  { \mathbf { h } } _ { m } ^ { \mathrm { M , o u t } } ] \right) .\tag{24}
$$

The difference term provides a pressure-like cue that reflects the imbalance between upstream and downstream traffic conditions associated with the movement.

We further allow different movements to exchange information through self-attention:

$$
\mathbf { H } _ { \mathcal { M } } = \mathrm { S e l f A t t n } _ { M } \left( \bar { \mathbf { H } } _ { \mathcal { M } } \right) , \qquad \bar { \mathbf { H } } _ { \mathcal { M } } = [ \bar { \mathbf { h } } _ { m } ^ { \mathrm { M } } ] _ { m \in \mathcal { M } } .\tag{25}
$$

The row $\mathbf { h } _ { m } ^ { \mathrm { M } }$ of $\mathbf { H } _ { \mathcal { M } }$ is the resulting contextualized representation of movement m.

Movement-to-Phase Composition. The movement-to-phase membership is encoded by the fixed binary incidence matrix $\mathbf { B } _ { \mathcal { M }  \mathcal { P } }$ . For each feasible phase $p \in \mathcal P$ , let

$$
\mathbf { H } _ { p } = [ \mathbf { h } _ { m } ^ { \mathrm { M } } ] _ { m \in p }\tag{26}
$$

collect the representations of the movements served by phase $p .$

A shared learnable phase query $\mathbf { q } _ { P }$ attends to these constituent movements:

$$
\mathbf { c } _ { p } = \mathrm { C r o s s A t t n } \left( { \bf q } _ { P } , \mathbf { H } _ { p } , \mathbf { H } _ { p } \right) .\tag{27}
$$

The attention summary $\mathbf { c } _ { p }$ captures the relative composition of the movements within phase $p .$ However, since the attention weights are normalized to sum to one, it does not directly preserve their aggregate traffic magnitude.

To retain this complementary information, each movement representation is projected to a non-negative vector,

$$
{ \bf r } _ { m } = \mathrm { s o f t p l u s } \left( \mathrm { M L P _ { a d d } ( { \bf h } _ { { m } } ^ { M } ) } \right) ,\tag{28}
$$

which is accumulated over the movements belonging to phase p:

$$
\mathbf { r } _ { p } = \log \left( \mathbf { 1 } + \sum _ { m \in p } \mathbf { r } _ { m } \right) .\tag{29}
$$

The element-wise logarithm controls the scale of the additive summary.

Finally, the attention-based composition summary and additive traffic summary are combined to form the phase representation:

$$
{ \bf h } _ { p } ^ { \mathrm { P } } = \mathrm { M L P } _ { P } \left( \mathrm { c o n c a t } [ { \bf c } _ { p } , { \bf r } _ { p } ] \right) .\tag{30}
$$

The resulting $\mathbf { h } _ { p } ^ { \mathrm { P } }$ therefore captures both the relative composition of the movements served by phase $p$ and their aggregate traffic state.

Candidate Phase Scoring. BEVLight applies the same scoring function to every feasible candidate phase, allowing the policy to operate with signal plans containing different numbers of phases.

Let $p _ { k _ { t } }$ denote the currently active phase. For candidate phase $p ,$ we compute

$$
\Delta \mathbf { h } _ { p } ^ { \mathrm { P } } = \mathbf { h } _ { p } ^ { \mathrm { P } } - \mathbf { h } _ { p _ { k _ { t } } } ^ { \mathrm { P } } ,\tag{31}
$$

which describes the representation change associated with switching from the current phase to the candidate.

The shared scorer jointly considers the candidate representation, the current phase representation, their difference, and the elapsed duration of the current phase:

$$
s _ { p } = \mathrm { M L P } _ { D } \left( \mathrm { c o n c a t } [ \mathbf { h } _ { p } ^ { \mathrm { P } } , \mathbf { h } _ { p _ { k _ { t } } } ^ { \mathrm { P } } , \Delta \mathbf { h } _ { p } ^ { \mathrm { P } } , \mathrm { F o u r i e r } \left( \tau _ { t } / \tau _ { \mathrm { m a x } } \right) ] \right) ,\tag{32}
$$

where $\tau _ { t }$ denotes the elapsed duration of the current phase and $\tau _ { \mathrm { m a x } }$ is the maximum phase duration.

At each decision step, BEVLight selects

$$
a _ { t } = \arg \operatorname* { m a x } _ { p \in \mathcal { P } } s _ { p } .\tag{33}
$$

Because the lane-to-movement and movement-to-phase relations are supplied through junction-specific topology while all learnable transformations are shared, the same architecture can process junctions with different lane layouts, movement sets, and feasible signal phases.

## A.2 Training Objectives and Details

This subsection provides additional details of the two-stage training procedure introduced in Sec. 3.3. Stage I uses simulator-provided traffic states to learn traffic-aware lane and movement representations, while Stage II trains the phase-selection policy from human traffic-control demonstrations.

Stage I: Traffic-Aware Representation Pretraining. We apply auxiliary prediction heads to the contextualized lane representations $\widetilde { \mathbf { h } } _ { \ell } ^ { \mathrm { L } }$ and movement representations $\mathbf { h } _ { m } ^ { \mathrm { M } }$ . These objectives encourage the intermediate representations to preserve traffic information directly relevant to downstream signal-control decisions.

At the lane level, BEVLight predicts queue length and vehicle count. Let $q _ { \ell }$ and $n _ { \ell }$ denote their ground-truth values, and $\hat { q } _ { \ell }$ and $\hat { n } _ { \ell }$ the corresponding predictions. We use L2 regression objectives,

$$
\mathcal { L } _ { q } ^ { \mathcal { L } } = \frac { 1 } { \vert \mathcal { L } \vert } \sum _ { \ell \in \mathcal { L } } \Vert \hat { q } _ { \ell } - q _ { \ell } \Vert _ { 2 } ^ { 2 } , \qquad \mathcal { L } _ { n } ^ { \mathcal { L } } = \frac { 1 } { \vert \mathcal { L } \vert } \sum _ { \ell \in \mathcal { L } } \Vert \hat { n } _ { \ell } - n _ { \ell } \Vert _ { 2 } ^ { 2 } .\tag{34}
$$

The lane head additionally predicts slotwise vehicle presence and longitudinal position. Summing the predicted presence values yields the corresponding vehicle-count estimate. To preserve fine-grained spatial information, we supervise the predicted and observed vehicle positions using a bidirectional Chamfer objective $\mathcal { L } _ { \mathrm { C h } } ^ { \mathcal { L } }$ . Each observed vehicle position is matched to its nearest predicted position and vice versa, with the contribution of each predicted position weighted by its predicted presence. This objective encourages $\widetilde { \mathbf { h } } _ { \ell } ^ { \mathrm { L } }$ to retain the locations of individual vehicles relative to the stop line rather than only aggregate traffic statistics.

At the movement level, BEVLight predicts queue length and vehicle count from $\mathbf { h } _ { m } ^ { \mathrm { M } }$ . Let $q _ { m }$ and $n _ { m }$ denote their ground-truth values, with predictions $\hat { q } _ { m }$ and $\hat { n } _ { m }$ . The corresponding objectives are

$$
\mathcal { L } _ { q } ^ { \mathcal { M } } = \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } \| \widehat { q } _ { m } - q _ { m } \| _ { 2 } ^ { 2 } , \qquad \mathcal { L } _ { n } ^ { \mathcal { M } } = \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } \| \widehat { n } _ { m } - n _ { m } \| _ { 2 } ^ { 2 } .\tag{35}
$$

The complete first-stage objective is

$$
\mathcal { L } _ { \mathrm { s t a g e } 1 } = \left( \mathcal { L } _ { q } ^ { \mathcal { L } } + \mathcal { L } _ { n } ^ { \mathcal { L } } + \mathcal { L } _ { \mathrm { C h } } ^ { \mathcal { L } } \right) + \left( \mathcal { L } _ { q } ^ { \mathcal { M } } + \mathcal { L } _ { n } ^ { \mathcal { M } } \right) .\tag{36}
$$

The simulator records the position of every vehicle at each time step. These recorded states provide the supervision above and also allow the same traffic evolution to be rendered under different visual appearances while retaining identical traffic-state targets.

Stage II: Human-Supervised Policy Learning. Starting from the representations learned in Stage I, we train the phase-selection policy using the human demonstrations described in Sec. 3.2. Each demonstration provides the preferred phase $a _ { t } ^ { * }$ together with a ranking $r _ { t } ( p )$ over every feasible phase $p \in \mathcal P$ , where rank one denotes the most preferred phase.

Given the candidate scores $s _ { p }$ from Eq. 11, we first obtain the policy distribution

$$
\pi _ { t } ( p ) = \frac { \exp ( s _ { p } ) } { \sum _ { p ^ { \prime } \in \mathcal { P } } \exp ( s _ { p ^ { \prime } } ) } .\tag{37}
$$

The selected phase provides a hard supervision target $\mathbf { e } _ { a _ { t } ^ { * } }$ . To additionally preserve the preferences among the remaining candidates, we construct a soft target from the complete human ranking. Following the top-one listwise formulation of ListNet [12], we define

$$
y _ { t } ( p ) = \frac { \exp \left( - r _ { t } ( p ) / | \mathcal { P } | \right) } { \sum _ { p ^ { \prime } \in \mathcal { P } } \exp \left( - r _ { t } ( p ^ { \prime } ) / | \mathcal { P } | \right) } .\tag{38}
$$

Scaling the ranks by the number of feasible phases keeps the range of the target logits comparable across signal plans containing different numbers of candidates.

The second-stage objective jointly supervises the selected phase and the complete phase ranking:

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathrm { C E } \left( \mathbf { e } _ { a _ { t } ^ { * } } , \pmb { \pi } _ { t } \right) + \lambda _ { \mathrm { s o f t } } \mathrm { C E } \left( \mathbf { y } _ { t } , \pmb { \pi } _ { t } \right) .\tag{39}
$$

The first term directly learns the human-selected phase, while the second transfers the relative preferences among all feasible candidates.

Because the ranking supervision is defined over the feasible phases of each individual signal plan, the same objective can be applied across junctions with different numbers and compositions of signal phases. Moreover, re-rendered observations corresponding to the same recorded traffic evolution retain the original human ranking because the un derlying traffic state is unchanged.

## B Experimental Setup

## B.1 Experimental Settings

We conduct closed-loop experiments in TranSimHub [13], with microscopic vehicle dynamics simulated by SUMO [14]. All simulated roads have a speed limit of 13.9 m/s (approximately 50 km/h). Vehicle speeds follow a Gaussian distribution with a mean of 10 m/s and a standard deviation of 3 m/s, representing typical urban traffic conditions.

For the visual encoder, we employ a lightweight ResNet trained from scratch. It consists of a stride-4 convolutional stem followed by two residual blocks and produces 128-channel feature maps at 1/4 input resolution. Each observation contains five consecutive 448×448 RGB BEV frames.

All experiments are conducted on a workstation running Ubuntu 22.04 with an Intel Xeon Gold 6133 CPU, 128 GB of memory, and an NVIDIA GeForce RTX 4090 GPU. Structured-state RL baselines are implemented with Stable-Baselines3 [29] and trained for $1 0 ^ { 6 }$ environment steps. UniTSA [4] is trained with PPO, while IntelliLight [1], At tendLight [16], and MPLight [3] are trained with DQN. Their optimization settings are summarized below.

PPO. For UniTSA [4], we use 16 parallel environments with 128 steps per rollout, resulting in 2,048 transitions per update. Each rollout is optimized for 10 epochs with a minibatch size of 64. The learning rate is $3 \times 1 0 ^ { - 4 } ,$ , with a discount factor of 0.99 and a GAE parameter of 0.95. We use a policy clipping range of $0 . 2 ,$ disable value-function clipping, and enable advantage normalization. The entropy and value-loss coefficients are set to 0 and 0.5, respectively, with a maximum gradient norm of 0.5.

DQN. For IntelliLight, AttendLight, and MPLight [1, 16, 3], we use a learning rate of $1 0 ^ { - 4 }$ and a discount factor of 0.99. The replay buffer stores up to $1 0 ^ { 6 }$ transitions, with learning starting after 100 transitions. Each update performs 16 gradient steps with a batch size of 32. We use one-step returns and hard target-network updates $( \tau = 1 )$ every $1 0 { , } 0 0 0$ transitions. The exploration rate is linearly annealed from 1.0 to 0.05 over the first 10% of training. Gradients are clipped to a maximum norm of 10.

![](images/dc99d014cf00bcb2f657b69f3cb2e4f3be858ac2a672b42b1e2686aa845f2a9f.jpg)  
Figure 6: BEV observations and lane masks for the eight simulated intersections. For each scene, the lane masks indicate the visible lane regions used by BEVLight, with different colors denoting individual lanes.

Table 4: Spatial coverage of the eight simulated intersections.
<table><tr><td></td><td colspan="2">Ground coverage</td><td colspan="3">Visible lane length (m)</td></tr><tr><td>Intersection</td><td>Footprint (m × m)</td><td>Area (m²)</td><td>Min.</td><td>Mean</td><td>Max.</td></tr><tr><td>Gaojiaoyuan</td><td>140.86 × 140.86</td><td>19,842</td><td>58.13</td><td>60.00</td><td>62.09</td></tr><tr><td>Beishahe</td><td>131.48 × 131.48</td><td>17,287</td><td>50.82</td><td>57.90</td><td>62.87</td></tr><tr><td>Pinganli</td><td>147.89 × 147.89</td><td>21,871</td><td>55.22</td><td>58.27</td><td>60.67</td></tr><tr><td>Yongrunlu</td><td>147.50 × 147.50</td><td>21,756</td><td>56.59</td><td>59.09</td><td>61.20</td></tr><tr><td>Zhijingdao</td><td>111.95 × 111.95</td><td>12,533</td><td>56.01</td><td>60.13</td><td>64.10</td></tr><tr><td>Yau Ma Tei</td><td>148.48 × 148.48</td><td>22,046</td><td>46.29</td><td>56.89</td><td>61.69</td></tr><tr><td>Guanghua</td><td>157.07 × 157.07</td><td>24,671</td><td>57.74</td><td>60.41</td><td>64.05</td></tr><tr><td>Songdo</td><td>155.51 × 155.51</td><td>24,183</td><td>42.76</td><td>58.07</td><td>69.98</td></tr></table>

## B.2 Simulation Scenarios

We construct eight simulation scenes from real-world intersection layouts across Beijing, Tianjin, Hong Kong, Chengdu, and Songdo, covering diverse three- and four-arm junction geometries and lane configurations. The scenes represent different urban contexts, ranging from dense inner-city intersections, such as Yau Ma Tei in Hong Kong, to newly developed districts with wider road layouts, such as Songdo.

Figure 6 visualizes the rendered BEV observation and corresponding lane masks for each intersection. The masks define the lane regions used by BEVLight and illustrate the visible extent of both incoming and outgoing lanes under the UAV viewpoint. Table 4 further reports the ground coverage and median visible lane length at a simulated UAV height of 100 m.

## B.3 Traffic Demand

For each scene, we construct five demand patterns with different traffic patterns. Three patterns are used for training: off-peak, characterized by low and relatively stable inflow; peak-hour, characterized by sustained high inflow; and commuter fluctuation, which introduces structured variations over time. For evaluation, we introduce two held-out

![](images/1a4dc42324d93f39117b64fcf05f10861dd4650298e27e3c494139b982c21a71.jpg)  
Figure 7: Traffic-demand profiles for the eight simulated scenes.

Table 5: Traffic-demand statistics across the eight simulated scenes. Each entry reports the mean ± standard deviation of the vehicle inflow over five time intervals, in vehicles per minute.
<table><tr><td></td><td colspan="3">Training</td><td colspan="2">Held-out Evaluation</td></tr><tr><td>Scene</td><td></td><td>Off-peak Commuter fluctuation Peak-hour Demand build-up Irregular traffic</td><td></td><td></td><td></td></tr><tr><td>Gaojiaoyuan</td><td> $2 0 . 6 \pm 1 . 5$ </td><td> $3 3 . 0 \pm { 3 . 0 }$ </td><td> $4 8 . 0 \pm 1 . 4$ </td><td> $3 9 . 8 \pm 9 . 8$ </td><td> $3 4 . 4 \pm 3 . 3$ </td></tr><tr><td>Beishahe</td><td> $2 1 . 0 \pm 1 . 4$ </td><td> $3 4 . 0 \pm 6 . 0$ </td><td> $5 0 . 4 \pm 1 . 9$ </td><td> $4 4 . 0 \pm 1 1 . 2$ </td><td> $4 2 . 0 \pm 7 . 4$ </td></tr><tr><td>Pinganli</td><td> $2 6 . 0 \pm 2 . 1$ </td><td> $3 8 . 8 \pm 3 . 7$ </td><td> $5 4 . 8 \pm 3 . 0$ </td><td> $4 3 . 6 \pm 1 0 . 3$ </td><td> $4 3 . 6 \pm 3 . 9$ </td></tr><tr><td>Yongrunlu</td><td> $2 2 . 6 \pm 1 . 0$ </td><td> $3 9 . 8 \pm 3 . 3$ </td><td> $5 6 . 0 \pm 1 . 3$ </td><td> $3 9 . 8 \pm 9 . 8$ </td><td> $3 7 . 6 \pm 3 . 9$ </td></tr><tr><td>Zhijingdao</td><td> $2 0 . 8 \pm 1 . 6$ </td><td> $3 1 . 4 \pm 2 . 5$ </td><td> $4 5 . 6 \pm 1 . 5$ </td><td> $3 3 . 2 \pm 7 . 7$ </td><td> $3 6 . 0 \pm 3 . 6$ </td></tr><tr><td>Yau Ma Tei</td><td> $2 6 . 8 \pm 2 . 0$ </td><td> $4 0 . 4 \pm 4 . 0$ </td><td> $6 6 . 8 \pm 6 . 6$ </td><td> $6 4 . 6 \pm 1 5 . 6$ </td><td> $4 7 . 6 \pm 5 . 0$ </td></tr><tr><td>Guanghua</td><td> $1 7 . 8 \pm 2 . 6$ </td><td> $3 1 . 2 \pm 3 . 3$ </td><td> $4 6 . 4 \pm 1 . 0$ </td><td> $3 6 . 0 \pm 9 . 6$ </td><td> $3 7 . 0 \pm 3 . 6$ </td></tr><tr><td>Songdo</td><td> $4 2 . 2 \pm 1 . 3$ </td><td> $6 4 . 2 \pm 5 . 0$ </td><td> $8 5 . 8 \pm 2 . 7$ </td><td> $7 4 . 4 \pm 1 7 . 9$ </td><td> $7 1 . 6 \pm 7 . 7$ </td></tr></table>

demand patterns that are not observed during training. Demand build-up progressively increases the inflow over time, while irregular traffic introduces stochastic temporal variations.

Figure 7 shows the temporal profiles of the five demand patterns across the eight scenes. Although the absolute inflow varies across scenes, the demand patterns preserve their intended temporal characteristics. Table 5 further summarizes the mean and standard deviation of each demand profile over the five time intervals.

## B.4 Real-UAV Data

For real-world evaluation, we use UAV videos from the dataset introduced by Fonod et al. [11], selecting four intersections in Songdo, denoted as O, P, T, and H. The videos are captured at approximately 140–150 m above ground and provide bird’s-eye views of diverse intersection geometries and lane configurations. These scenes are used to evaluate the transfer of BEVLight from simulation to real UAV observations.

Figure 8 shows representative UAV observations and the corresponding lane masks for the four intersections. The masks indicate the visible lane regions used by BEVLight and illustrate how the lane-level representation is transferred from simulation to real UAV imagery. Table 6 further reports the visible lane lengths in each scene.

## B.5 Compared Methods

We compare BEVLight with three groups of traffic signal methods: two conventional methods, four structured-state RL methods, and two vision-based two-stage methods. The conventional and RL methods directly access structured traffic states provided by the simulator, whereas the vision-based methods first extract lane-level traffic information from BEV observations before applying an RL controller.

![](images/f1c0566c1e0273e577f235448a09ea11d6b43d3104af5369de74d3c5f2d9a005.jpg)  
Figure 8: Real UAV observations and corresponding lane masks for intersections O, P, T, and H

Table 6: Spatial coverage of the four real-UAV intersections.
<table><tr><td rowspan="2">Intersection</td><td colspan="2">Ground coverage</td><td colspan="3">Visible lane length (m)</td></tr><tr><td>Footprint (m × m)</td><td>Area (m²)</td><td>Min.</td><td>Mean</td><td>Max.</td></tr><tr><td>Songdo, O</td><td>118.5 × 118.5</td><td>14,050</td><td>25.61</td><td>30.85</td><td>37.90</td></tr><tr><td>Songdo, P</td><td>119.3 × 119.3</td><td>14,239</td><td>22.28</td><td>27.11</td><td>32.90</td></tr><tr><td>Songdo, T</td><td>120.7 × 120.7</td><td>14,568</td><td>14.53</td><td>25.53</td><td>43.24</td></tr><tr><td>Songdo, H</td><td>121.2 × 121.2</td><td>14,695</td><td>17.54</td><td>25.06</td><td>51.38</td></tr></table>

Conventional Controllers. We include two commonly used traffic-responsive controllers:

• FixedTime follows a predefined signal schedule and holds each phase for 20 s in our experiment, independent of the observed traffic conditions.

• MaxPressure [15] selects the signal phase with the highest traffic pressure, computed from the imbalance between upstream and downstream traffic states. It provides a strong adaptive baseline without requiring policy learning.

Structured-State RL Methods. These methods receive lane-level traffic states directly from the simulator and learn signal-control policies through RL:

• UniTSA [4] introduces a junction-matrix representation to accommodate different intersection layouts and uses traffic-state augmentation to expose the policy to diverse intersection structures and traffic conditions.

• IntelliLight [1] learns adaptive signal control from structured intersection states using a value-based RL formulation.

• AttendLight [16] employs attention mechanisms to aggregate traffic information across lanes and candidate signal phases, allowing a shared policy to operate across different intersection configurations.

• MPLight [3] incorporates traffic-pressure information into RL-based signal control and models competition among candidate phases.

Vision-Based Two-Stage Methods. We further construct two baselines that first recover traffic states from BEV observations and then use AttendLight for signal control:

• YOLO-Count uses YOLOv11 [30] to detect vehicles from BEV images with oriented bounding boxes and assigns the detections to individual lanes to obtain lane-level vehicle counts. The detector is fine-tuned on synthetic images generated from the same training scenes used by BEVLight, ensuring a fair comparison without introducing additional visual training data. The resulting lane-level states are then provided to AttendLight for phase selection.

• BEVLight-State+AttendLight uses BEVLight’s auxiliary lane-level prediction heads to estimate traffic states directly from BEV observations. These predicted states are subsequently passed to AttendLight for signal control, providing a two-stage alternative that does not rely on explicit object detection.

In contrast to the two vision-based baselines, BEVLight directly maps BEV observations and junction topology to signal actions without an intermediate vehicle-detection or structured-state reconstruction stage.

![](images/1cab3f8415927426a6823c228417ab848c5c6d2c35e1e1eba3807de5db370cdc.jpg)  
Figure 9: Web-based virtual TSC platform for collecting human demonstrations. The interface displays the current bird’s-eye-view traffic scene and all feasible signal actions. Hovering over an action highlights its corresponding lanes and permitted movements on the intersection.

## C Human Demonstrations

## C.1 Web-Based Human Demonstration Collection

We develop a virtual teleoperation platform to collect human traffic-control demonstrations. Eight graduate researchers with backgrounds in computer science or transportation, including master’s and PhD students, participated in the data collection. As illustrated in Fig. 9, the interface presents a bird’s-eye view of the current intersection together with al feasible signal actions. To make the control semantics of each action explicit, hovering over an action highlights the corresponding lanes and permitted traffic movements directly on the intersection view.

Participants make a control decision every 10 s and rank all feasible actions using arrow-based controls. Rather than recording only the selected action, the ranking provides relative preference supervision among feasible signal phases, indicating not only which action is preferred but also how the remaining alternatives are ordered. This provides richer supervision for learning the phase-scoring function and avoids treating all non-selected actions as equally undesirable. For each decision, we record the complete simulator state, including the positions of all vehicles, together with the human-provided action ranking. The recorded vehicle states are subsequently used to reconstruct the same traffic scene under different visual appearances, as described in Appendix C.2.

To ensure the quality of the demonstrations, we use MaxPressure as a reference controller and perform filtering at the episode level. Each human-controlled episode is compared with the MaxPressure controller under the same traffic scenario, and only episodes achieving a lower average waiting time are retained. After filtering, the dataset contains more than 5K human decisions, corresponding to approximately 14 hours of traffic-control demonstrations.

## C.2 Appearance-Diverse Scene Rendering

For each human decision, we record the underlying simulator state rather than storing only the rendered observation. In particular, the recorded state contains the positions of all vehicles at the decision time, allowing the same traffic configuration to be reconstructed independently of its original visual appearance. The associated human action ranking is preserved throughout this process. In this way, a single human demonstration can provide supervision for multiple visually different observations while retaining exactly the same traffic state and control preference.

We render each recorded state under five visual conditions: Morning, Noon, Dusk, Dusk-Rain, and Afternoon-Rain. The rendering process changes illumination, weather, and scene appearance while keeping vehicle positions, junction geometry, and control supervision unchanged. As illustrated in Figs. 10–13, each example additionally includes a visualization of the recorded vehicle configuration before rendering. The figures cover all eight intersections used in our experiments: Beishahe, Beijing, Gaojiaoyuan, Beijing, Pinganli, Beijing, Yongrunlu, Beijing, Guanghua, Chengdu, Yau Ma Tai, Hongkong, Songdo, SouthKorea, and Zhijingdao, Tianjin.

![](images/40767d14d42ef7814865bec43375fd941bea9a368465eb0e9e349fdbede8d032.jpg)  
(b) Gaojiaoyuan, Beijing  
Figure 10: Appearance-diverse rendering of recorded human demonstration states. For each intersection, the top-left panel visualizes the recorded vehicle configuration, while the remaining panels show the same traffic state under five visual conditions: Morning, Noon, Dusk, Dusk-Rain, and Afternoon-Rain. The underlying traffic state and human action ranking remain unchanged across all rendered variants. The examples shown here correspond to Beishahe, Beijing and Gaojiaoyuan, Beijing.

![](images/1e0fd70010a2387137c452233738405dd7d48a8fe403176def2b85b4d4932cfb.jpg)  
(d) Yongrunlu, Beijing  
Figure 11: Appearance-diverse rendering examples for Pinganli, Beijing and Yongrunlu, Beijing.

![](images/97129e8deed04986fbbb00bf96d3813e099a7bacbe3d091e9c8886a983445a08.jpg)

(e) Guanghua, Chengdu  
![](images/1a13a8fbacdf94d60bbe46e20714ba0203955e2ff9e1ae0852854ba725ae0ce1.jpg)  
(f) Yau Ma Tei, Hongkong  
Figure 12: Appearance-diverse rendering examples for Guanghua, Chengdu and Yau Ma Tei, Hongkong.

![](images/832608ac58c87bd3858c2e23cf30068b0c96ba156b1e8bb5c8ad4b596d3fed6f.jpg)  
(h) Zhijingdao, Tianjin  
Figure 13: Appearance-diverse rendering examples for Songdo, SouthKorea and zhijingdao, Tianjin.

This state-preserving rendering strategy increases the visual diversity associated with the collected human demonstrations. More importantly, because the same control preference is observed across different appearances of the same traffic state, the model is encouraged to rely on traffic-relevant visual information rather than appearance-specific cues. We therefore use these appearance-diverse renderings throughout policy training as a component for improving robustness to visual-domain changes and supporting transfer from simulated BEV observations to real UAV imagery.

## C.3 Human Control Preferences and Action Ranking

In addition to the selected action, we collect a complete ranking over all feasible signal phases at each decision step. The motivation is that a single selected-action label provides only the top choice, while all non-selected actions would otherwise be treated equivalently. In the TSC task, however, feasible actions that are not selected can still differ substantially in their relative priority. By recording the full ranking, we preserve this relative preference structure and provide richer supervision for learning the phase-scoring function in BEVLight.

Fig. 14 presents representative examples from four intersections: Beishahe, Beijing, Gaojiaoyuan, Beijing, Pinganli, Beijing, and Yau Ma Tei, Hong Kong. Each example shows the traffic observation at a decision step together with the human-assigned ranking over all feasible signal phases.

More specifically, the ranking supervision specifies not only the preferred phase, but also the relative ordering of the remaining feasible phases. In Figure 14(a), the action ranked third serves no vehicles under the current traffic state, whereas the second-ranked action serves active traffic, providing a meaningful distinction between two non-selected actions. Meanwhile, the first- and second-ranked actions correspond to traffic conditions of comparable congestion and are both placed ahead of the remaining alternatives. In Figure 14(c), the ranking further reflects the current signal context: the active phase is prioritized to allow vehicles already being served to clear the intersection before switching to another phase. In Figure 14(d), the highest-ranked action serves visibly heavier traffic than the other alternatives and is accordingly assigned the highest priority. These examples illustrate that the full ranking preserves distinctions among both selected and non-selected actions that would be lost if only the final action were recorded.

## D Additional Real-World Case Studies

Figure 15 presents six consecutive UAV frames from each of four real intersections in Songdo: Junctions O, P, T, and H, together with BEVLight’s predicted signal phases. The model is applied directly to real UAV imagery without realworld fine-tuning. Across these sequences, its phase predictions respond to changes in vehicle distribution, shifting priority as traffic clears and demand changes across movements.

Since the evaluation uses recorded videos, the predicted phases do not affect actual signals or subsequent traffic. These examples therefore provide qualitative evidence of zero-shot transfer rather than closed-loop control performance: BEVLight produces plausible phase decisions that adapt to evolving traffic conditions under real-world visual inputs.

## E Discussion

Practical considerations. Deployment hinges on the two properties evaluated in Sec. 4: a policy that is not tied to a particular junction or signal plan can be moved between sites without retraining, and a policy trained entirely in simulation can be applied to real UAV observations without collecting and labeling real control data, which is impractical at scale. Beyond the UAV video stream, BEVLight requires lane masks in the BEV frame and the lane-tomovement and movement-to-phase relations of the active signal plan. The relations are part of any signal timing plan, and the masks are specified once per junction and viewpoint. Decisions are made every 10 s, leaving ample margin for transmission and inference. The policy only selects among feasible phases, so it cannot release conflicting movements; phase transitions and minimum green times remain under the signal controller, and a human operator can override the policy at any step. UAV observation is unavailable under strong wind, heavy precipitation, or restricted airspace, so BEVLight is best viewed as an on-demand controller that complements installed infrastructure.

Limitations. Our simulation-to-real evaluation is offline: queue estimates and phase decisions transfer to recorded UAV videos, but we do not control real signals, and at the two sites with the sparsest queues the queue-count error remains above an always-zero predictor. BEVLight controls a single junction, and coordination across junctions is left for future work. The policy imitates demonstrations from graduate researchers filtered against MaxPressure, so it inherits their quality and is not optimized for a long-horizon objective. Lane masks are assumed static, so UAV drift would require image registration. Finally, the simulated UAV height of 100 m differs from the 140–150 m of the real recordings, and appearance rendering covers daytime and rain but not night-time.

Broader impact. BEVLight lowers the infrastructure barrier for adaptive signal control and can be deployed on demand where fixed sensing is absent or temporarily disrupted, with reduced waiting time translating into lower fuel consumption and emissions. UAV imagery raises privacy and regulatory concerns: BEVLight operates on lane-level features from images captured above 100 m, at which individuals and license plates are not identifiable, and does not track vehicles across frames, but deployments should follow local airspace regulations and data-retention policies. The demonstrations were collected in simulation from consenting participants and contain no personal data. Since a learned controller reflects the priorities of its demonstrations, operators should verify that its phase preferences do not systematically disadvantage particular approaches before deployment. The virtual teleoperation platform is not specific to BEVLight. It records complete simulator states alongside human decisions, so any recorded demonstration can be re-rendered under new appearances or used to supervise other visual controllers, and it can be extended to other observation modalities such as roadside cameras. We plan to release the platform and the collected demonstrations to support future work on human-supervised visual TSC.

![](images/40cdbb1bb57b3cfa3b5b0f7a95185005fc412e810a4d003b9c6ca43e6e204bbf.jpg)

![](images/ba791892b5b72f2d5a0b0f4420147f700f41628b5a7275038e87f5909eaa4b7d.jpg)

![](images/9268d2487c29d27633b3842c36aa844317db715555fdbbeaea4cbe545fa50867.jpg)  
(a) Beishahe, Beijing

![](images/e7302a538037624bde57a59d97a8d784549cebd5eaae130bea3526214d340650.jpg)

![](images/988c40dd8597658cbaf83b588814b646d5e0663076cfa22a3d396c02590b08ad.jpg)

![](images/6bec64a87738c2c1d2afaed61c80de1bdf60a8f8dbcda59103e2eaa19fba70ec.jpg)  
(b) Gaojiaoyuan, Beijing

![](images/192f8a93dde87a923b9e9f034ae07d6e0b63236263bd734ba0179a20c167f8c3.jpg)

![](images/52cdee9a3b5755551fac8ea32fa3815a6cbba1d8410e5d5cdfdf1cf5c9f25c07.jpg)

![](images/d76e8bdb037b49f6d8cad8f780eeaa642f4f673cc1acef4171d5001861b552e4.jpg)

![](images/a048f0d49ef2203eb975e8009a83f522ca75b7f34098da1a8e4bbfe0e56e65e3.jpg)  
(c) Pinganli, Beijing

![](images/38b6e22c17c12510b4fcc71496f860541c8f6419346eae745f338434dabcb8ef.jpg)

![](images/a993a015d2b79d05a76d712264553751e92058757463b52a12d89b255235631a.jpg)

![](images/2b663425564b0d807dd22cb4927de6e7cd15c659fff21b7b2f42a4542dbddf5e.jpg)

![](images/5e17c505a59c473a1e053c5e8c09597c87c3079ece09908ae1d5aeca0723d81b.jpg)

![](images/be6750d6be9470aae58a062df9d3153047f887335d31a9bf37374886e9f2e263.jpg)  
(d) Yau Ma Tei, Hong Kong  
Figure 14: Representative examples of human control preferences and action rankings. Each example shows a traffic observation together with the ranking assigned by the human operator over all feasible signal phases. The ranking preserves the relative ordering among feasible actions rather than retaining only the top-ranked phase, providing richer supervision for learning the phase-scoring function in BEVLight.

![](images/6b8b1fbf0ad20e793c57b41a1df3601dd668c0bf0dd60b61ddc1a2e594663b4c.jpg)  
Songdo, O Junction

![](images/380480874b0c6a90b4dad6ed9aaf29a339e7c2f070dda68f704d64250289c6b4.jpg)  
Songdo, P Junction

![](images/4bd40850fdce362e20dc048ca942c198f570c282db94e6ace8269273cd74185a.jpg)  
Songdo, T Junction

![](images/4518bb5006b3d8b9d953bebf818d28c9cb2f9db8a5c89fa2038115c39f8721e4.jpg)  
Songdo, H Junction  
Figure 15: Zero-shot phase predictions over six consecutive UAV frames at Junctions O, P, T, and H in Songdo. Predictions are evaluated offline without controlling actual signals.