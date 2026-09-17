# Structural Inference under Hidden Agents

Zhongben Gong

Xiaoqun Wu<sup>∗</sup> Hui Huang

Mingyang Zhou

College of Computer Science and Software Engineering, Shenzhen University, Shenzhen 518060, China

September 2026

## Abstract

Recovering latent interaction structures from multi-agent dynamics is important for understanding and predicting interacting systems. Trajectory-based structural inference has achieved promising performance, but conventional formulations assume that the trajectories of all modeled agents are available. In practice, agents may become unobserved at deployment because of limited sensing, occlusion, or communication failure. Existing studies have considered unseennode estimation, structural inference under partial observations, and missing-value imputation, yet the joint recovery of hidden-agent trajectories and their interactions remains underexplored. We formulate this problem as structural inference under hidden agents. Its key dificulty is a circular dependency: recovering interactions involving a hidden agent requires an estimate of its trajectory, while trajectory reconstruction can itself benefit from structural information. To address this challenge, we propose Structural Inference under Hidden Agents (SIHA), which combines structure-agnostic initialization with structure-guided iterative refinement. SIHA re constructs hidden trajectories from visible observations, infers interactions using Neural Relational Inference, and feeds the estimated structure back into hidden-state reconstruction through multi-strength structural attention and iterative state–structure updates. Experiments on three benchmark dynamical systems demonstrate consistent improvements in visible-to-visible structural inference, while also showing benefits in hidden-state reconstruction and future prediction. Motion-capture experiments with simulated whole-limb occlusion further demonstrate its efectiveness in realistic hidden-agent settings.

Keywords: structural inference, multi-agent systems, partial observability, complex networks, hidden agents

## 1 Introduction

Interaction and network structure shape observed dynamics across many scientific domains. In physical systems, collective dynamics contain information about the underlying interaction network [1]. In biology, inferring gene regulatory networks from single-cell measurements helps characterize regulatory organization [2]. Social-network experiments demonstrate that network structure afects behavioral difusion [3], while intersectoral production networks shape how shocks propagate into aggregate economic fluctuations [4]. These examples motivate methods for understanding or recovering latent interactions from observations of collective behavior.

![](images/cbf7d66cbafdd3d468ed91685a1dd87c2f3699795092ac0e48f533143b99aa72.jpg)  
Figure 1: Illustration of the problem setting considered in this paper. Given the observed trajectories of visible agents, the goal is to predict the hidden agents’ trajectories, all future trajectories, and recover the latent interaction structure.

Trajectory-based neural structural inference pursues this objective by learning relations among modeled entities from their state sequences. Neural Relational Inference (NRI) [5] provides a central formulation in which a latent interaction graph supports predictive dynamics; later extensions consider time-varying relations or iterative graph refinement [6, 7]. These conventional formulations typically operate on states or trajectories of the entities supplied as input. In practical systems, however, this assumption can be violated at the entity level. For example, in biological systems, the dynamics of some interacting species may remain unobserved because field surveys or sensors capture only a subset of the ecosystem; in engineered multi-agent systems, vehicles or robots may become unobservable because of occlusion, sensing limitations, or communication loss; and in motion-capture scenarios, some body parts may be entirely missing from a sequence because of persistent occlusion or tracking failure. In these cases, an interacting entity may be entirely absent from the observations over the considered time interval, rather than merely having sporadic missing values. Its state is therefore absent as a direct input, while interactions involving that entity must still be inferred through the dynamics of the observed agents. This information gap makes complete structural inference under-constrained, as illustrated in Figure 1.

Neighboring work studies unseen-node inference, node-level partial observation, and relational inference with missing values or temporal segments [8, 9, 10]; these regimes difer in their observation units, inference targets, and supervision, as detailed in Section 2.2. We consider a protocol in which complete multi-agent trajectories are available during training: selected agents are masked from the predictor input, and their ground-truth trajectories supervise hidden-state reconstruction, while ground-truth edge labels are not used for model training. At deployment, only visible-agent trajectories are provided, and the latent interaction structure is inferred separately for each sample rather than assumed to be shared across diferent observations. SIHA therefore reconstructs the hidden-agent trajectories and estimates the corresponding sample-specific interactions involving hidden agents. Compared with conventional structural inference under fully observed trajectories or fixed missing-value settings, this setting remains comparatively underexplored. This protocol matches the practical scenarios described above. In controlled training environments, complete trajectories can be collected with reliable communication or multi-view sensing, whereas deployment may involve occlusion, signal loss, or limited sensing coverage that leaves some entities entirely unobserved. Moreover, interaction patterns can vary across missions, scenes, or motions, while explicit edge annotations may be unavailable, making sample-specific structural inference from the remaining visible trajectories a realistic requirement.

Within this protocol, hidden-state reconstruction and complete structural inference depend on one another. Structural information can guide the reconstruction of unobserved agents, whereas complete structural inference in the current SIHA data flow relies on reconstructed hidden trajectories. This coupled inference problem motivates a structure-agnostic state initialization followed by structure-guided, iterative refinement of states and interactions. SIHA follows this computational flow. A structure-agnostic hidden-state predictor (HSP-sa) first initializes the hidden trajectories from visible observations. A standard NRI model, used in the current implementation as the structure-inference and future-prediction backbone, then estimates interactions from the visible and reconstructed trajectories. A structure-guided hidden-state predictor (HSP-sg) refines the hidden trajectories using the inferred structure. Its multi-strength structural guidance retains learnable attention paths alongside attention biased by the predicted structure, after which HSP-sg and NRI alternate to refine the reconstructed states and structural estimate.

We evaluate SIHA on the Springs, Charged Particles, and Kuramoto systems with one to five hidden agents. In the principal external comparison, SIHA and visible-only NRI are compared numerically only on visible-agent forecasting and visible-to-visible structural accuracy, which are defined for both methods. SIHA obtains higher visible-to-visible structural accuracy throughout the reported grid, while visible-agent forecasting results vary by system. Hidden histories, hidden future trajectories, and interactions involving hidden agents represent additional SIHA outputs that visible-only NRI does not define under this protocol. The internal comparison contrasts the HSP-sa baseline with the complete SIHA pipeline: both use matched hidden-state supervision and the same pretrained NRI module, while SIHA further introduces HSP-sg for structure-guided reconstruction and iterative state–structure refinement. Across the reported settings, SIHA matches or improves upon HSP-sa on the displayed metrics. The evaluation further examines hidden-agentcount trends, recorded CMU Motion Capture trajectories with simulated whole-limb occlusion [11], and mechanism and supervision analyses.

The contributions of this work are threefold: (1) we introduce structural inference under hidden agents, where complete trajectories can be available for training but only visible-agent trajectories are observed at deployment, and hidden trajectories and their interactions must be jointly recovered without edge-label supervision; (2) we propose SIHA to resolve the circular dependency between hidden-state reconstruction and structural inference through structure-agnostic bootstrapping, structure-guided reconstruction, and iterative state–structure refinement; and (3) we demonstrate the efectiveness of SIHA across three dynamical systems, varying numbers of hidden agents, and motion-capture sequences with whole-limb occlusion, showing consistent improvements in visible-to-visible structural inference and clear benefits from structure-guided refinement.

## 2 Related Work

## 2.1 Relational and Structural Inference

Recovering interactions from collective dynamics is a classical inverse problem [1]. Interaction Networks make objects and pairwise relations explicit when the graph is supplied [12], whereas Neural Relational Inference (NRI) learns discrete latent edge types from trajectories through a predictive decoder without ground-truth edge labels [5].

Subsequent work extends this paradigm through joint structure–dynamics learning and improved message passing [13, 14], time-varying or evolving relations [6, 15], and iterative graph refinement [7]. Other formulations infer relations through masked reconstruction, heterogeneous interaction modeling, or learned discrete graph structures [16, 17, 18]. Parallel time-series studies also estimate causal or predictive dependencies among observed variables or variable groups [19, 20]. Collectively, these methods broaden relational and structural inference while generally assuming that modeled entities are represented by observed states, trajectories, or node attributes. SIHA considers deployment in which entire agent trajectories are absent and must be reconstructed together

with their interactions.

## 2.2 Structural Inference under Incomplete Observations

Incomplete observation arises at multiple levels, including unseen nodes, partially observed topology, scarce state samples, and masked values or temporal segments. Several studies address closely related incomplete-observation settings. Alet et al. [8] estimate an unseen node at test time by optimizing its initial state under learned dynamics; their Section 5.2 demonstration uses a predictive model trained with ground-truth edges. SICSM [9] studies structural inference under node-level partial observation and represents efects of hidden intermediaries through indirect, multi-hop dependencies. Its primary target is structure under partial observation, without jointly treating explicit unobserved-node trajectory reconstruction and complete hidden-incident graph recovery as the inference objective. DifRI [10] combines self-supervised difusion imputation with relational inference when values or temporal segments are masked within a fixed set of known components. It imputes component states, while an entirely absent component trajectory lies outside its input formulation.

Neighboring regimes include recovering network-generating rules from partially observed topology [21] and learning continuous network dynamics from sparse, irregular, partial, or noisy state observations [22]. These regimes difer in the missing unit and inference target. Within this taxonomy, SIHA jointly reconstructs whole hidden-agent trajectories and estimates interactions involving hidden agents. Complete trajectories provide hidden-state supervision during training without edge-label supervision, whereas deployment uses visible-agent trajectories only.

## 2.3 State Reconstruction and Graph-Guided Imputation

Time-series imputation reconstructs missing observations over a fixed set of known variables using recurrent, probabilistic, attention-based, or difusion models, including BRITS, GP-VAE, SAITS, and CSDI [23, 24, 25, 26]. GRIN further uses graph message passing to incorporate relational information into multivariate imputation [27]. These methods show how temporal and relational context can support reconstruction of missing values or segments in known variables. Their primary objective remains value-level state reconstruction over fixed channels. SIHA addresses an agent absent as an entity from the deployment input and couples the two directions: estimated structure guides hidden-state reconstruction, while reconstructed hidden trajectories support inference of interactions involving hidden agents.

## 3 Problem Formulation

## 3.1 Interacting Systems with Hidden Agents

We consider an interacting dynamical system with N agents, of which the first $N _ { \mathrm { v i s } }$ agents are visible and the remaining $N _ { \mathrm { h i d } } = N - N _ { \mathrm { v i s } }$ agents are hidden. The state of agent i at time t is denoted by $\mathbf { x } _ { i } ^ { t } \in \mathbb { R } ^ { d }$ . Its trajectory over an observed history of $T$ time steps is

$$
\mathbf { x } _ { i } = [ \mathbf { x } _ { i } ^ { 1 } , \dots , \mathbf { x } _ { i } ^ { T } ] \in \mathbb { R } ^ { T \times d } .\tag{1}
$$

The visible- and hidden-agent trajectories are respectively collected as

$$
\begin{array} { r l } & { \mathbf { x } _ { \mathrm { v i s } } = [ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N _ { \mathrm { v i s } } } ] \in \mathbb { R } ^ { N _ { \mathrm { v i s } } \times T \times d } , } \\ & { \mathbf { x } _ { \mathrm { h i d } } = [ \mathbf { x } _ { N _ { \mathrm { v i s } } + 1 } , \ldots , \mathbf { x } _ { N } ] \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times T \times d } . } \end{array}\tag{2}
$$

The interaction structure among all agents is represented by a directed graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { N } \}$ and $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ . For K possible interaction types, the complete graph is encoded by an adjacency tensor $\mathbf { z } \in \mathbb { R } ^ { N \times N \times K }$ , where $z _ { i j k } = 1$ indicates an interaction of type k from agent i to agent j. Because entire hidden-agent trajectories are absent from the observed input, both their states and their interactions with the rest of the system must be estimated.

## 3.2 Training and Deployment Setting

During the training phase, complete multi-agent trajectories are available. To train the hiddenstate predictors, a designated subset of agents is masked from the predictor input to form ${ \mathbf { x } } _ { \mathrm { v i s } }$ , while the corresponding ground-truth trajectories $\mathbf { x } _ { \mathrm { h i d } }$ are used as reconstruction targets. The structureinference module is pretrained on complete trajectories using the standard NRI objective, without using the ground-truth interaction tensor z as a training label. When ground-truth structures are available in the experimental datasets, they are used only for ofline evaluation.

At deployment, only ${ \mathbf { x } } _ { \mathrm { v i s } }$ is observed. SIHA first reconstructs the hidden trajectories and then combines them with the visible trajectories for future-state prediction and complete-structure inference; ground-truth hidden states and interaction labels are not used during this process.

## 3.3 Inference Targets

Given the visible-agent histories ${ \mathbf { x } } _ { \mathrm { v i s } }$ at deployment, the task is to estimate three inference targets: (1) the historical trajectories of the hidden agents, $\hat { \mathbf { x } } _ { \mathrm { h i d } } \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times T \times d }$ ;

(2) the future trajectories of all agents over a horizon of $T ^ { \prime }$ steps, $\hat { \mathbf { x } } _ { \mathrm { a l l } } ^ { \mathrm { f u t u r e } } \in \mathbb { R } ^ { N \times T ^ { \prime } \times d } ;$ and

(3) the complete interaction structure, $\hat { \mathbf { z } } \in \mathbb { R } ^ { N \times N \times K }$ , including interactions involving visible and hidden agents.

The overall inference problem can thus be written as

$$
\begin{array} { r } { f : \mathbf { x } _ { \mathrm { v i s } } \mapsto \left( \hat { \mathbf { x } } _ { \mathrm { h i d } } , \hat { \mathbf { x } } _ { \mathrm { a l l } } ^ { \mathrm { f u t u r e } } , \hat { \mathbf { z } } \right) . } \end{array}\tag{3}
$$

## 4 Method

## 4.1 Framework Overview

The key challenge in structural inference with hidden agents is the circular dependency between hidden-state reconstruction and structural inference. Inferring interactions involving a hidden agent requires an estimate of its trajectory, while reconstructing that trajectory can benefit from knowing how the agent interacts with the observed system. At deployment, neither quantity is directly available, so the inference procedure must first establish an initial estimate before structural information can be exploited.

SIHA resolves this dependency through structure-free initialization followed by structure-guided refinement. As illustrated in Figure 2, the structure-agnostic hidden-state predictor (HSP-sa) first reconstructs the hidden trajectories from the visible observations alone. This initial estimate completes the multi-agent trajectory set and enables the pretrained NRI backbone [5] to produce a provisional complete interaction structure and future-state prediction.

Although this initialization enables the first structural estimate, the HSP-sa reconstruction itself does not exploit relational information. SIHA therefore feeds the predicted structure back into the structure-guided hidden-state predictor (HSP-sg), which refines the hidden trajectories under structural guidance. The refined trajectories are then passed to NRI again to update the interaction structure and future prediction. Repeating this process forms an iterative state–structure refinement loop, in which reconstructed states support structural inference and the inferred structure in turn provides additional information for hidden-state reconstruction.

The refinement loop is implemented diferently during training and deployment. During HSP-sg training, SIHA maintains a structure cache for each training sample so that structural guidance remains stable across optimization steps. The cache is initialized by the pretrained HSP-sa–NRI pipeline, kept fixed during an initial warm-up period, and then refreshed periodically using the current HSP-sg reconstruction and the pretrained NRI model. This delayed and periodic update reduces the influence of unreliable hidden state estimates at the early stage of training and stabilizes structure-guided reconstruction. At deployment, all model parameters are fixed. The structure is first initialized by HSP-sa and NRI, after which HSP-sg and NRI are directly alternated for a fixed number of refinement rounds to successively update the hidden trajectories and interaction structure.

In the current implementation, NRI serves as both the structure-inference and future-prediction backbone. We use two NRI edge types, whose semantics depend on the underlying system, such as the absence or presence of an interaction in Springs and repulsive or attractive interactions in Charged Particles. We denote the resulting interaction estimate by $\hat { \mathbf { A } } \in \mathbb { R } ^ { N \times N }$ in the following sections.

![](images/3de22343e3e3abb63bd717f82226a04a2dc76dfdb18b03c8b556c6189efc7b8c.jpg)  
Figure 2: Overview of the SIHA state–structure refinement framework. HSP-sa initializes the hidden trajectories, after which the pretrained NRI backbone estimates the interaction structure and future trajectories from the completed trajectories. HSP-sg then uses the cached structure to refine the hidden trajectories, and NRI recomputes the structure for subsequent refinement. Purple boxes denote pretrained modules, while the green box denotes the structure-guided refinement module.

## 4.2 Structure-Agnostic Hidden-State Reconstruction

HSP-sa initializes the coupled procedure by estimating the hidden histories from the visible-agent set alone:

$$
\mathrm { H S P - s a : } \quad \mathbf { x } _ { \mathrm { v i s } } \mapsto \hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( 0 ) } \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times T \times d } .\tag{4}
$$

It is implemented with a Set Transformer [28]. The visible trajectories are treated as an unordered input set, and $N _ { \mathrm { h i d } }$ learned seed vectors produce an unordered set of hidden-trajectory slots. For the HSP-sa mapping, each visible trajectory is flattened to form $\tilde { \mathbf { x } } _ { \mathrm { v i s } } \in \mathbb { R } ^ { N _ { \mathrm { v i s } } \times ( T \cdot d ) }$ , and the encoder produces

$$
\mathbf { Z } = \mathrm { S A B } ( \mathrm { S A B } ( \tilde { \mathbf { x } } _ { \mathrm { v i s } } ) ) \in \mathbb { R } ^ { N _ { \mathrm { v i s } } \times D } .\tag{5}
$$

Here D denotes the latent feature dimension. The HSP-sa decoder maps these visible-agent features to the hidden slots as

$$
\begin{array} { r } { \hat { \mathbf { X } } _ { \mathrm { h i d } } = \mathrm { r F F } ( \mathrm { S A B } ( \mathrm { P M A } _ { N _ { \mathrm { h i d } } } ( \mathbf { Z } ) ) ) \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times ( T \cdot d ) } , } \end{array}\tag{6}
$$

where unflattening $\hat { \mathbf { X } } _ { \mathrm { h i d } } \mathrm { ~ y ~ }$ ields $\hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( 0 ) } \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times T \times d }$ . For the synthetic systems, this set-to-set construction is compatible with the permutation ambiguity defined in Section 3, and the slots are aligned for the supervised objective and evaluation as described in Section 4.5. In the motioncapture experiments, each output slot is assigned to a predefined masked joint and retains that fixed ordering.

HSP-sa does not receive an estimated graph. Its output is combined with the visible trajectories to form $\hat { \mathbf { x } } ^ { ( 0 ) } = [ \mathbf { x } _ { \mathrm { v i s } } , \hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( 0 ) } ]$ , which the pretrained NRI module maps to an initial structural estimate $\hat { \bf A } _ { 0 }$ and future trajectories. The generic SAB, MAB, and PMA definitions are given in A.2.

## 4.3 Structure-Guided Hidden-State Reconstruction

HSP-sg has the same set-to-set input and output interface as HSP-sa, but additionally conditions its attention computations on the current predicted structure:

$$
\mathrm { H S P - s g : } \quad ( \mathbf { x } _ { \mathrm { v i s } } , \hat { \mathbf { A } } ) \mapsto \hat { \mathbf { x } } _ { \mathrm { h i d } } .\tag{7}
$$

![](images/8ac519b2fefc52322196e93c2daa3e00c89791ae535c430c195c88451c88d126.jpg)  
Figure 3: Illustration of the structure-guided hidden-state predictor (HSP-sg). The model takes visible trajectories ${ \mathbf { x } } _ { \mathrm { v i s } }$ and predicts hidden trajectories with multi-strength structural guidance. The encoder (red) captures visible-to-visible relations; the decoder applies PMA (green) between hidden-slot queries and visible features and SAB (blue) among hidden slots. The structural guidance uses $\hat { \mathbf { A } } _ { v v } , \hat { \mathbf { A } } _ { \mathrm { c r o s s } } ,$ , and $\hat { \mathbf { A } } _ { h h }$ in the corresponding attention modules.

As shown in Figure 3, the encoder self-attention operates among visible-agent features, PMA connects the hidden output slots with encoded visible features, and the decoder self-attention operates among hidden slots. SIHA uses the corresponding blocks of the predicted adjacency matrix to guide these three attention operations. Specifically,

$$
\hat { \mathbf { A } } = \left[ \begin{array} { l l } { \hat { \mathbf { A } } _ { \mathrm { v v } } } & { \hat { \mathbf { A } } _ { \mathrm { v h } } } \\ { \hat { \mathbf { A } } _ { \mathrm { h v } } } & { \hat { \mathbf { A } } _ { \mathrm { h h } } } \end{array} \right] ,
$$

where $\hat { \mathbf { A } } _ { \mathrm { v v } } \in \mathbb { R } ^ { N _ { \mathrm { v i s } } \times N _ { \mathrm { v i s } } } , \ \hat { \mathbf { A } } _ { \mathrm { v h } } \in \mathbb { R } ^ { N _ { \mathrm { v i s } } \times N _ { \mathrm { h i d } } }$ 2 $\hat { \mathbf { A } } _ { \mathrm { h v } } \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times N _ { \mathrm { v i s } } }$ , and $\hat { \mathbf { A } } _ { \mathrm { h h } } \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times N _ { \mathrm { h i d } } }$ . Under the convention in Section 3.1, $A _ { i j }$ continues to represent an interaction from agent i to agent $j .$ . For the PMA cross-attention, the two cross-agent blocks describe interactions between visible and hidden agents in opposite directions. To use them in the same hidden-query–visible-key attention matrix, we transpose $\hat { \mathbf { A } } _ { \mathrm { v h } }$ and average the two estimates:

$$
\hat { \bf A } _ { \mathrm { c r o s s } } = \frac { 1 } { 2 } \left( \hat { \bf A } _ { \mathrm { h v } } + \hat { \bf A } _ { \mathrm { v h } } ^ { \top } \right) \in \mathbb { R } ^ { N _ { \mathrm { h i d } } \times N _ { \mathrm { v i s } } } .
$$

This cross-set matrix is then used to guide the PMA attention between hidden slots and visibleagent features.

HSP-sg applies these structural blocks to the corresponding attention operations as follows:

$\hat { \mathbf { A } } _ { \mathrm { v v } }$ guides the encoder self-attention among visible agents;

$\hat { \mathbf { A } } _ { \mathrm { c r o s s } }$ guides the PMA cross-attention between hidden-slot queries and encoded visible features; and

$\hat { \bf A } _ { \mathrm { h h } }$ guides the decoder self-attention among hidden slots.

For the encoder self-attention, for example, head i receives a structural bias scaled by $\alpha _ { i } { : }$

$$
\mathrm { h e a d } _ { i } ^ { \mathrm { s g } } = \mathrm { s o f t m a x } \Bigg ( \frac { \mathbf { Q } \mathbf { W } _ { i } ^ { Q } ( \mathbf { K } \mathbf { W } _ { i } ^ { K } ) ^ { \top } } { \sqrt { d / h } } - \alpha _ { i } \big ( \mathbf { 1 } _ { N _ { \mathrm { v i s } } \times N _ { \mathrm { v i s } } } - \hat { \mathbf { A } } _ { \mathrm { v v } } \big ) \Bigg ) \mathbf { V } \mathbf { W } _ { i } ^ { V }\tag{8}
$$

The term ${ \mathbf { 1 } } _ { N _ { \mathrm { v i s } } \times N _ { \mathrm { v i s } } } - \hat { { \mathbf { A } } } _ { \mathrm { v v } }$ downweights attention between pairs not connected in the current estimate. The four attention heads use $\alpha _ { i } \in \{ 0 , 1 , 5 , 1 0 ^ { 9 } \}$ : the first remains unguided, the middle two use soft structural biases, and the last approximates a hard mask. This multi-strength design retains an unguided attention path while exposing other heads to diferent strengths of the predicted structure. The same construction is applied to PMA cross-attention with $\hat { \mathbf { A } } _ { \mathrm { c r o s s } }$ and hidden-slot self-attention with $\hat { \bf A } _ { \mathrm { h h } } ;$ generic Set Transformer equations are provided in A.2.

## 4.4 Iterative State–Structure Refinement

Let $f _ { \mathrm { p r e } } , f _ { \mathrm { s g } }$ , and $g _ { \mathrm { { N R I } } }$ denote HSP-sa, HSP-sg, and the structure estimator of the pretrained NRI module, respectively. With r indexing cache refreshes during training or refinement rounds during deployment, the state–structure updates are

$$
\begin{array} { r l } { \hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( 0 ) } = f _ { \mathrm { p r e } } ( \mathbf { x } _ { \mathrm { v i s } } ) , ~ } & { } \\ { \hat { \mathbf { A } } ^ { ( 0 ) } = \mathbf { A } _ { \mathrm { c a c h e } } ^ { ( 0 ) } = g _ { \mathrm { N R I } } \Big ( [ \mathbf { x } _ { \mathrm { v i s } } , \hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( 0 ) } ] \Big ) , ~ } & { } \\ { \hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( r + 1 ) } = f _ { \mathrm { s g } } \Big ( \mathbf { x } _ { \mathrm { v i s } } , \mathbf { A } _ { \mathrm { c a c h e } } ^ { ( r ) } \Big ) , ~ } & { } \\ { \hat { \mathbf { A } } ^ { ( r + 1 ) } = g _ { \mathrm { N R I } } \Big ( [ \mathbf { x } _ { \mathrm { v i s } } , \hat { \mathbf { x } } _ { \mathrm { h i d } } ^ { ( r + 1 ) } ] \Big ) , ~ } & { ~ \mathbf { A } _ { \mathrm { c a c h e } } ^ { ( r + 1 ) } \gets \hat { \mathbf { A } } ^ { ( r + 1 ) } . } \end{array}\tag{9}
$$

Thus each refinement first updates the hidden trajectories under the current cached structure and then recomputes the structure from the visible and reconstructed trajectories.

During HSP-sg training, a recurrence step is applied only at a scheduled cache refresh: the cache remains fixed during an initial warm-up period and between refreshes. Separate caches are maintained for the training, validation, and test samples. At deployment, the test cache is initialized by the HSP-sa–NRI pass and one recurrence step is applied in each of a fixed number of rounds. Thus the iterative path changes the reconstructed trajectories and cached structures while using the trained modules with fixed parameters at deployment. The cache schedule and number of deployment rounds are reported in B.

## 4.5 Learning and Inference Procedures

SIHA does not introduce a single joint loss, NRI and the two hidden-state predictors retain their respective objectives. The NRI module is pretrained on complete trajectories using the standard NRI evidence lower bound [5], whose trajectory-prediction term supports learning the latent graph without ground-truth edge labels. For HSP-sa and HSP-sg, the training target is the ground truth trajectory of the masked agents. In the synthetic systems, the predicted hidden slots are unordered, so Hungarian matching [29] aligns them with the target trajectories before mean squared error is evaluated. In the motion-capture experiments, the masked joints have predefined semantic identities, so direct joint-wise MSE is evaluated in their fixed order. The dataset-dependent loss is

$$
\mathcal { L } _ { \mathrm { H S P - s a } } = \mathcal { L } _ { \mathrm { H S P - s g } } = \left\{ \begin{array} { l l } { \mathrm { M S E } ( \mathrm { A l i g n } ( \hat { \mathbf { x } } _ { \mathrm { h i d } } , \mathbf { x } _ { \mathrm { h i d } } ) , \mathbf { x } _ { \mathrm { h i d } } ) , } & { \mathrm { s y n t h e t i c ~ s y s t e m s } , } \\ { \mathrm { M S E } ( \hat { \mathbf { x } } _ { \mathrm { h i d } } , \mathbf { x } _ { \mathrm { h i d } } ) , } & { \mathrm { m o t i o n ~ c a p t u r e } . } \end{array} \right.\tag{10}
$$

Training proceeds in two stages. First, HSP-sa and NRI are pretrained separately: HSP-sa receives visible trajectories as input and hidden trajectories as supervised targets, whereas NRI receives complete trajectories and is optimized with the standard NRI objective. Second, HSP-sa initializes the structure cache and HSP-sg is optimized with the dataset-appropriate hidden-state loss above. The pretrained NRI module processes $\left[ \mathbf { x } _ { \mathrm { v i s } } , \hat { \mathbf { x } } _ { \mathrm { h i d } } \right]$ to refresh the cache at the prescribed intervals. Detailed settings and pseudocode are given in B.

At deployment, HSP-sa first reconstructs the hidden trajectories from ${ \mathbf { x } } _ { \mathrm { v i s } }$ , and NRI uses the resulting completed trajectories to initialize the structure and future prediction. HSP-sg and NRI then alternate for the fixed refinement rounds described above. The last HSP-sg output supplies the hidden histories, and the final NRI pass supplies the complete interaction estimate and future trajectories.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. The synthetic evaluation uses the Springs, Charged Particles, and Kuramoto systems, following the trajectory-based structural-inference setting of NRI [5]. The number of visible agents is fixed at $N _ { \mathrm { v i s } } = 5$ , and the reported fixed-count settings use $N _ { \mathrm { h i d } } \in \{ 1 , 2 , 3 , 4 , 5 \}$ . Each trajectory contains 50 time steps for training and validation and 100 time steps for testing. Springs and Charged Particles use four-dimensional position–velocity states, whereas Kuramoto uses the threedimensional representation specified in B. Models are configured for the corresponding value of $N _ { \mathrm { h i d } }$ in these fixed-count evaluations. Section 5.4 additionally considers recorded human-motion trajectories from the CMU Motion Capture database [11], with one complete limb artificially masked. Dataset sizes, system-specific settings, and hyperparameters remain in B.

Table 1: Main results on three datasets with varying number of hidden agents $( N _ { \mathrm { h i d } } = 1 , \dots , 5 )$ Metrics: hidden-state prediction MSE, future-state prediction MSE, and structure prediction ACC. Each item reports NRI / HSP-sa / SIHA (HSP-sg); dashes mark outputs that are not defined for visible-only NRI. Best values among methods for which a metric is defined are bolded.
<table><tr><td rowspan=1 colspan=14> $\mathrm { M S E } _ { \mathrm { H S P } } \left( \downarrow \right)$                      $\mathrm { M S E } _ { \mathrm { F S P } } \left( \downarrow \right)$                                      ACC (↑)Dataset   $N _ { \mathrm { h i d } }$ Vis.                 Hid.              V-V            V-H          H-H</td></tr><tr><td rowspan=1 colspan=1>1   -</td><td rowspan=1 colspan=1>2.5e-3</td><td rowspan=1 colspan=2>2.1e-3  2.1e-5</td><td rowspan=1 colspan=3>1.5e-5 1.4e-5     1.7e-2</td><td rowspan=1 colspan=1>1.1e-2 89.7</td><td rowspan=1 colspan=1>99.7</td><td rowspan=1 colspan=1>99.7</td><td rowspan=1 colspan=1>98.2</td><td rowspan=1 colspan=3>98.4</td></tr><tr><td rowspan=1 colspan=1>2   1</td><td rowspan=1 colspan=1>5.6e-3</td><td rowspan=1 colspan=2>3.2e-3  3.0e-5</td><td rowspan=1 colspan=1>9.3e-6</td><td rowspan=1 colspan=1>7.6e-6</td><td rowspan=1 colspan=1>1.6e-2</td><td rowspan=1 colspan=1>9.4e-3 76.1</td><td rowspan=1 colspan=1>98.7</td><td rowspan=1 colspan=1>98.9 -</td><td rowspan=1 colspan=1>91.4</td><td rowspan=1 colspan=1>95.8</td><td rowspan=1 colspan=1>68.1</td><td rowspan=1 colspan=1>79.0</td></tr><tr><td rowspan=1 colspan=1>Springs   3</td><td rowspan=1 colspan=1>7.1e-3</td><td rowspan=1 colspan=2>4.9e-3 3.7e-5</td><td rowspan=1 colspan=1>6.4e-6</td><td rowspan=1 colspan=1>5.0e-6</td><td rowspan=1 colspan=1>1.3e-2</td><td rowspan=1 colspan=1>1.0e-2 72.9</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>97.2</td><td rowspan=1 colspan=1>83.5</td><td rowspan=1 colspan=1>88.6</td><td rowspan=1 colspan=1>61.6</td><td rowspan=1 colspan=1>67.4</td></tr><tr><td rowspan=1 colspan=1>4   1</td><td rowspan=1 colspan=1>8.1e-3</td><td rowspan=1 colspan=2>7.1e-3 3.9e-5</td><td rowspan=1 colspan=1>5.7e-6</td><td rowspan=1 colspan=2>5.0e-6     1.2e-2</td><td rowspan=1 colspan=1>1.1e-2 70.3</td><td rowspan=1 colspan=1>95.6</td><td rowspan=1 colspan=2>/96.0    78.6</td><td rowspan=1 colspan=1>81.7 1</td><td rowspan=1 colspan=1>57.9</td><td rowspan=1 colspan=1>60.5</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>9.1e-3</td><td rowspan=1 colspan=2>8.8e-3 4.3e-5</td><td rowspan=1 colspan=1>6.0e-6</td><td rowspan=1 colspan=2>6.0e-6  一 1.2e-2</td><td rowspan=1 colspan=1>1.1e-2 68.8</td><td rowspan=1 colspan=1>93.0</td><td rowspan=1 colspan=2>93.1 、75.2</td><td rowspan=1 colspan=2>75.9  一 56.3</td><td rowspan=1 colspan=1>57.0</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2.1e-2</td><td rowspan=1 colspan=2>1.9e-2 1.4e-3</td><td rowspan=1 colspan=1>1.4e-3</td><td rowspan=1 colspan=2>1.4e-3    2.0e-1</td><td rowspan=1 colspan=1>1.8e-1 71.4 /</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=2>/79.5 -70.6</td><td rowspan=1 colspan=3>/71.1    -1-1</td></tr><tr><td rowspan=1 colspan=1>2   1</td><td rowspan=1 colspan=1>3.8e-2</td><td rowspan=1 colspan=1>3.7e-2</td><td rowspan=1 colspan=1>2.0e-3</td><td rowspan=1 colspan=1>2.1e-3</td><td rowspan=1 colspan=1>2.1e-3</td><td rowspan=1 colspan=1>1.7e-1</td><td rowspan=1 colspan=1>1.6e-1 66.8</td><td rowspan=1 colspan=1>/74.5</td><td rowspan=1 colspan=1>/74.9 1</td><td rowspan=1 colspan=1>60.4</td><td rowspan=1 colspan=1>760.7</td><td rowspan=1 colspan=1>52.4</td><td rowspan=1 colspan=1>53.2</td></tr><tr><td rowspan=1 colspan=1>Charged   3</td><td rowspan=1 colspan=1>4.3e-2</td><td rowspan=1 colspan=1>4.1e-2</td><td rowspan=1 colspan=1>2.5e-3</td><td rowspan=1 colspan=1>2.6e-3</td><td rowspan=1 colspan=1>2.6e-3</td><td rowspan=1 colspan=1>1.4e-1</td><td rowspan=1 colspan=1>1.2e-1 62.1</td><td rowspan=1 colspan=1>71.2</td><td rowspan=1 colspan=1>72.3  一</td><td rowspan=1 colspan=1>57.8</td><td rowspan=1 colspan=1>58.5</td><td rowspan=1 colspan=1>51.5</td><td rowspan=1 colspan=1>51.8</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4.5e-2</td><td rowspan=1 colspan=1>4.4e-2</td><td rowspan=1 colspan=1>2.2e-3</td><td rowspan=1 colspan=1>2.4e-3</td><td rowspan=1 colspan=1>2.4e-3</td><td rowspan=1 colspan=1>1.3e-1</td><td rowspan=1 colspan=1>1.2e-1 61.4</td><td rowspan=1 colspan=1>68.7</td><td rowspan=1 colspan=1>69.6 -</td><td rowspan=1 colspan=1>56.2</td><td rowspan=1 colspan=1>57.4</td><td rowspan=1 colspan=1>51.6</td><td rowspan=1 colspan=1>52.6</td></tr><tr><td rowspan=1 colspan=1>5   -</td><td rowspan=1 colspan=1>4.6e-2</td><td rowspan=1 colspan=2>4.5e-2  3.3e-3</td><td rowspan=1 colspan=1>3.6e-3</td><td rowspan=1 colspan=1>3.6e-3</td><td rowspan=1 colspan=1>1.1e-1</td><td rowspan=1 colspan=1>1.1e-1 59.0</td><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=2>54.7    51.7</td><td rowspan=1 colspan=1>51.7</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3.6e-2</td><td rowspan=1 colspan=2>2.8e-2 4.3e-2</td><td rowspan=1 colspan=1>1.3e-2</td><td rowspan=1 colspan=2>1.3e-2    4.7e-2</td><td rowspan=1 colspan=1>4.0e-2 85.0</td><td rowspan=1 colspan=1>91.7</td><td rowspan=1 colspan=2>91.8    87.6</td><td rowspan=1 colspan=3>88.6          =</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1.0e-1</td><td rowspan=1 colspan=2>9.5e-2 5.0e-2</td><td rowspan=1 colspan=1>/1.5e-2</td><td rowspan=1 colspan=1>1.4e-2</td><td rowspan=1 colspan=1>1.1e-1</td><td rowspan=1 colspan=1>9.6e-2 72.3</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>86.7  1</td><td rowspan=1 colspan=1>73.7</td><td rowspan=1 colspan=1>74.7</td><td rowspan=1 colspan=1>54.6</td><td rowspan=1 colspan=1>55.3</td></tr><tr><td rowspan=1 colspan=1>Kuramoto  3</td><td rowspan=1 colspan=1>1.1e-1</td><td rowspan=1 colspan=2>1.0e-1  4.2e-2</td><td rowspan=1 colspan=1>1.6e-2</td><td rowspan=1 colspan=1>1.5e-2  1</td><td rowspan=1 colspan=1>1.3e-1</td><td rowspan=1 colspan=1>1.2e-1 65.2</td><td rowspan=1 colspan=1>83.8</td><td rowspan=1 colspan=1>84.0 、</td><td rowspan=1 colspan=1>68.8</td><td rowspan=1 colspan=1>70.0 1</td><td rowspan=1 colspan=1>53.5</td><td rowspan=1 colspan=1>54.5</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1.2e-1</td><td rowspan=1 colspan=2>1.0e-1 4.0e-2</td><td rowspan=1 colspan=1>1.5e-2</td><td rowspan=1 colspan=1>1.5e-2  -</td><td rowspan=1 colspan=1>1.3e-11</td><td rowspan=1 colspan=1>1.2e-1  58.4</td><td rowspan=1 colspan=1>75.4</td><td rowspan=1 colspan=1>75.7 一</td><td rowspan=1 colspan=1>62.4</td><td rowspan=1 colspan=1>62.4 -</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>52.0</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.2e-1</td><td rowspan=1 colspan=2>1.1e-1 3.7e-2</td><td rowspan=1 colspan=1>1.6e-2</td><td rowspan=1 colspan=2>1.6e-2    1.2e-1</td><td rowspan=1 colspan=1>1.2e-1 58.3</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>58.6</td><td rowspan=1 colspan=1>58.6 -</td><td rowspan=1 colspan=1>51.5</td><td rowspan=1 colspan=1>51.6</td></tr></table>

Methods and comparison protocol. The experiments instantiate the training and deployment setting defined in Section 3. HSP-sa and HSP-sg are trained with ground-truth trajectories of masked agents as hidden-state reconstruction targets, while interaction labels are not used for training. HSP-sg, together with the iterative state–structure refinement procedure, constitutes the complete method reported as SIHA (HSP-sg). HSP-sa is the internal controlled baseline: it uses the same hidden-state supervision but omits predicted-structure guidance and iterative refinement. For its structure and future-state outputs, the HSP-sa reconstruction is followed by the same pretrained NRI module.

Standard NRI [5] is the principal external reference. In this comparison it is trained and evaluated on the visible trajectories only. NRI and SIHA can therefore be compared numerically on visible-agent future prediction and visible-to-visible structure inference, which are defined for both methods. NRI does not produce hidden histories, hidden-agent future trajectories, or hiddenrelated edges under this protocol, so the corresponding entries are undefined. It therefore serves as an external reference, not a supervision-matched hidden-agent baseline.

Metrics and reporting. Hidden-state reconstruction is measured by ${ \mathrm { M S E } } _ { \mathrm { H S P } }$ . Future-state prediction is measured separately for visible and hidden agents by $\mathrm { M S E } _ { \mathrm { F S P , V i s . } }$ <sub>.</sub> and MSE<sub>FSP,Hid.</sub>. Structural accuracy is decomposed into visible-to-visible, visible-to-hidden, and hidden-to-hidden blocks, denoted by $\mathrm { A C C _ { V - V } , A C C _ { V - H } }$ , and $\mathrm { A C C } _ { \mathrm { H - H } }$ , respectively. MSE is lower-is-better, whereas structural accuracy is higher-is-better. Ground-truth edges are used only to compute these offline structural metrics. Table 1 reports point estimates over the full experimental grid, while Appendix C.3 provides five-run mean±standard-deviation results for three representative settings with $N _ { \mathrm { h i d } } = 3$

## 5.2 Overall Performance

Table 1 provides the broad synthetic comparison across all three systems and all five fixed hiddenagent counts. Each table entry lists NRI, ${ \mathrm { H S P - s a } } .$ and SIHA (HSP-sg) in that order. For the external NRI–SIHA comparison, performance on the common metrics varies across datasets. Both

HSP-based configurations achieve higher $\mathrm { \Delta A C C _ { V - V } }$ than visible-only NRI throughout the reported grid. Their visible-agent forecasting errors are lower on Springs and Kuramoto, while NRI is comparable or slightly better on Charged. The remaining hidden-related metrics highlight an additional capability of the HSP-based pipelines, since visible-only NRI does not produce these outputs.

The internal comparison between HSP-sa and SIHA (HSP-sg) holds hidden-state supervision and the NRI module fixed while changing structural guidance and refinement. Across the reported point estimates, SIHA (HSP-sg) is better than or equal to HSP-sa on the displayed metrics, with the largest diferences generally appearing on Springs and smaller diferences on Charged and Kuramoto. We conjecture that this diference may be related to the quality of the structures inferred by the underlying NRI model: its structural inference accuracy is lower on Charged and Kuramoto than on Springs, which may limit the benefit of the NRI-guided iterative refinement in HSP-sg. Since the SIHA refinement mechanism only requires a structural estimate together with future-state prediction, it could in principle be paired with other compatible structural-inference backbones. We therefore expect that a stronger backbone may further improve the efectiveness of HSP-sg, although this remains to be verified experimentally.

## 5.3 Efect of the Number of Hidden Agents

Table 1 shows a clear degradation in performance as the number of hidden agents increases while $N _ { \mathrm { v i s } } = 5$ remains fixed. As the latent portion of the system grows while the observed set remains unchanged, hidden-state reconstruction becomes progressively more dificult, and structural inference accuracy also generally declines. This trend is observed across Springs, Charged, and Kuramoto. Despite this increasing dificulty, the HSP-based pipelines consistently maintain higher visible-tovisible structural accuracy than the visible-only NRI reference across the reported hidden-agent counts. Additionally, the models in these experiments assume a fixed hidden-agent count; the setting in which this count is unknown is examined separately in D.2.

## 5.4 Motion-Capture Case Study

We evaluate SIHA on walking sequences from subjects #35 and #69 of the CMU Motion Capture database [11]. Each frame contains 31 body joints, each represented by 3D position and 3D velocity. We simulate two whole-limb occlusion settings. The left-arm setting masks seven joints— lclavicle, lhumerus, lradius, lwrist, lhand, lfingers, and lthumb—leaving 24 visible joints. The left-leg setting masks five joints—lhipjoint, lfemur, ltibia, lfoot, and ltoes—leaving 26 visible joints. These experiments apply artificial whole-limb masks to recorded human-motion trajectories and are not a benchmark of occlusion caused by a particular camera or sensor.

HSP-sa and HSP-sg are trained on fully observed motion-capture sequences, with the groundtruth trajectories of the designated masked joints used only as training-time hidden-state recon struction targets; deployment and evaluation receive only the visible joints. Training and evaluation use direct joint-wise MSE in the predefined masked-joint order. For each subject, SIHA uses an NRI backbone pretrained on that subject’s full 31-joint sequences for structure inference and future prediction. The same subject-specific full-observation NRI is reused for the left-arm and left-leg experiments. The external visible-only NRI baseline is trained and evaluated on 24 visible joints for the left-arm setting and 26 visible joints for the left-leg setting.

Across the four subject–occlusion settings, SIHA (HSP-sg) improves both hidden- and visibleagent future prediction over HSP-sa. It also reduces hidden-history reconstruction error in three settings; the exception is the subject #69 left-arm case. On the common visible-future metric, both HSP-based configurations have lower errors than the corresponding visible-only NRI baseline in all four settings. Overall, the expanded results show that structure-guided refinement provides consistent future-prediction benefits across the evaluated subjects and limb-occlusion patterns while generally improving hidden-history reconstruction.

Table 2: Results on two CMU motion-capture subjects under two whole-limb occlusion settings.
<table><tr><td>Subject</td><td>Occlusion</td><td>Method</td><td> ${ \mathrm { M S E } } _ { \mathrm { H S P } }$  ↓</td><td> $\mathrm { M S E } _ { \mathrm { F S P , H i d . } }$ </td><td>↓  $\mathrm { M S E } _ { \mathrm { F S P , V i s . } } \downarrow$ </td></tr><tr><td rowspan="5">CMU #35</td><td rowspan="5">Left arm</td><td>NRI</td><td></td><td></td><td>0.012463</td></tr><tr><td>HSP-sa</td><td>0.015547</td><td>0.039196</td><td>0.008176</td></tr><tr><td>SIHA (HSP-sg)</td><td>0.008056</td><td>0.021843</td><td>0.007788</td></tr><tr><td>NRI Left leg</td><td></td><td></td><td>0.039669</td></tr><tr><td>HSP-sa</td><td>0.025210</td><td>0.022922</td><td>0.006773</td></tr><tr><td rowspan="5">CMU #69</td><td rowspan="3">Left arm</td><td>SIHA (HSP-sg)</td><td>0.023735</td><td>0.014077</td><td>0.006594</td></tr><tr><td>NRI</td><td></td><td></td><td>0.001954</td></tr><tr><td>HSP-sa</td><td>0.003224</td><td>0.004434</td><td>0.001304</td></tr><tr><td>SIHA (HSP-sg)</td><td>0.003349</td><td>0.004263</td><td>0.001227</td></tr><tr><td rowspan="3">Left leg</td><td>NRI</td><td></td><td></td><td>0.001635</td></tr><tr><td>HSP-sa</td><td>0.001880</td><td>0.003977</td><td>0.000939</td></tr><tr><td>SIHA (HSP-sg)</td><td>0.001616</td><td>0.003633</td><td>0.000915</td></tr></table>

For qualitative visualization, we retain the subject #35 left-arm setting in Figure 4, with comparisons focused on the left and right hands. Human walking naturally involves coordinated motion between the two arms, so a strong relation between the left and right hands is expected. Consistent with this intuition, the NRI model trained on complete trajectories assigns the strongest connections of the right-hand node mainly to joints in the left-hand region. The visible-only NRI baseline has no hidden-arm nodes in its input and therefore cannot recover these cross-limb relations. After hidden-state reconstruction, both HSP-based pipelines can infer such connections, while HSP-sg produces a more concentrated cross-hand pattern than HSP-sa and is therefore closer to the fullobservation NRI reference. Together with the quantitative improvements in Table 2, this qualitative result suggests that HSP-sg achieves a stronger coupling between hidden-state prediction and structural inference. The focus-score definition, edge-selection thresholds, and panel-specific settings are given in D.1.

## 5.5 Model and Supervision Analysis

Multi-strength structural guidance. We analyze the attention guidance mechanism on Springs with $N _ { \mathrm { h i d } } = 3$ Three HSP-sg variants modify the attention-head coeficients to [1, 1, 5, 1e9] (no unguided head), [0, 1, 5, 5] (no hard-mask head), and [0, 0, 1e9, 1e9] (no intermediate-strength guidance).

Table 3 shows that removing the unguided head causes the largest degradation across the reported metrics, indicating that retaining a fully learnable attention path is important. Removing the intermediate-strength guidance also reduces performance, although to a smaller extent. In contrast, removing the hard-mask head leaves the reported point estimates unchanged in this setting, suggesting that the main benefit of the multi-strength design comes from combining unguided and softly structure-guided attention. Additionally, the strong masking head may also play a role when accurate structural information is available.

![](images/9ae5fa2f2b07729898b8b3928d9bc160739d654229a351e2e0769682b35161bf.jpg)  
(a)

![](images/abf810d618adbde4a18a61915a88d0ed8de064b1989d621f911eefe57bfdec82.jpg)  
(b)

![](images/e5ac76d414890a35ae8734780bec27d78fc678c11138ece6c7527dc23c0c5f2b.jpg)  
(c)

![](images/8351fec327c745d8d229fa876faa840c019765af9d0a9e010e19840c3ff9b64f.jpg)  
(d)

![](images/2339a3b5ae232d2f640018321e525041655ca1f3cd0cac7e4736de5ed861dd90.jpg)  
(e)

![](images/ed95efa4e03179212b214cde855119da8c9daeb49495467c358698a36db21c1f.jpg)  
(f)  
Figure 4: Visualizations of the left/right hand focus on the motion-capture limb-occlusion experiment. (a) Full-observation NRI reference; (b) visible-only NRI; (c) HSP-sa with left-hand focus; (d) HSP-sa with right-hand focus; (e) SIHA with left-hand focus; (f) SIHA with right-hand focus.

Table 3: Ablation study on the Springs dataset with $N _ { \mathrm { h i d } } = 3 .$
<table><tr><td>Model</td><td> ${ \mathrm { M S E } } _ { \mathrm { H S P } }$ </td><td> $\mathrm { M S E } _ { \mathrm { F S P , V i s . } }$ </td><td> $\mathrm { A C C } _ { \mathrm { V - V } }$ </td></tr><tr><td>SIHA  $\mathrm { ( H S P  – s g ) }$ </td><td>4.9e-3</td><td>5.0e-6</td><td>97.2</td></tr><tr><td>No Unguided</td><td>2.0e-2</td><td>1.3e-5</td><td>91.5</td></tr><tr><td>No Hard Mask</td><td>4.9e-3</td><td>5.0e-6</td><td>97.2</td></tr><tr><td>No Intermediate</td><td>8.2e-3</td><td>6.1e-6</td><td>96.6</td></tr></table>

Dependence on hidden-state supervision. We further examine three HSP-sa variants without direct supervision from ground-truth hidden trajectories. These variants difer in the initialization and optimization of NRI, while all of them train HSP-sa only through the visibleagent future-prediction objective. C.1 gives their complete definitions and results on Springs for $N _ { \mathrm { h i d } } \in \{ 1 , 2 , 3 , 4 , 5 \}$ . All three variants perform substantially worse than the supervised HSP-sa configuration in both hidden-state reconstruction and visible-agent prediction, with corresponding degradation in structural inference. These results indicate that, within the current SIHA framework, simply removing hidden-state supervision and relying on visible-future prediction is not suficient for efective hidden-agent reconstruction and structural inference. Extending SIHA toward a fully latent training setting without hidden-state supervision therefore represents an important direction for future work.

## 5.6 Additional Analyses

Additional analyses are provided in the appendices: C.2 reports the comparison of HSP-sa encoder. C.3 reports five-run mean±standard-deviation results for the three representative $N _ { \mathrm { h i d } } = 3$ settings. D.2 evaluates selection among models configured for diferent hidden-agent counts. D.3 reports the efect of additional unmodeled hidden agents on Springs. Detailed optimization settings and computational costs remain in B.

## 6 Conclusion

In this paper, we investigated structural inference in multi-agent systems when the trajectories of some agents are unavailable at deployment. We proposed Structural Inference under Hidden Agents (SIHA), which reconstructs hidden-agent trajectories and infers the complete interaction structure through coupled state–structure refinement. SIHA first initializes hidden trajectories without structural guidance, estimates interactions using NRI, and then incorporates the inferred structure into a structure-guided hidden-state predictor with multi-strength attention for iterative refinement. Extensive experiments on three benchmark dynamical systems and recorded motioncapture trajectories demonstrate the efectiveness of the proposed framework. SIHA achieves higher visible-to-visible structural accuracy than visible-only NRI across the evaluated synthetic settings while additionally recovering hidden-agent trajectories and interactions involving hidden agents. Compared with the structure-agnostic variant under the same supervision, structure-guided refinement improves or maintains the reported reconstruction, forecasting, and structural metrics. Experiments with diferent numbers of hidden agents, mechanism ablations, and simulated limb occlusion further demonstrate the efectiveness of the proposed design.

A limitation of the current formulation is its reliance on complete trajectories during training to supervise hidden-state reconstruction, although ground-truth edge labels are not required for model training or deployment. The tested objectives without direct hidden-state supervision do not recover the performance of the supervised formulation, leaving structural inference with fully latent hidden agents an open problem. Future work will investigate self-supervised hidden-state reconstruction and structural inference when hidden-agent trajectories are unavailable during both training and deployment.

## References

[1] Mor Nitzan, Jose Casadiego, and Marc Timme. Revealing physical interaction networks from statistics of collective dynamics. Science Advances, 3(2):e1600396, 2017.

[2] Aditya Pratapa, Amogh P Jalihal, Jefrey N Law, Aditya Bharadwaj, and TM Murali. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature Methods, 17(2):147–154, 2020.

[3] Damon Centola. The spread of behavior in an online social network experiment. Science, 329(5996):1194–1197, 2010.

[4] Daron Acemoglu, Vasco M. Carvalho, Asuman Ozdaglar, and Alireza Tahbaz-Salehi. The network origins of aggregate fluctuations. Econometrica, 80(5):1977–2016, 2012.

[5] Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting systems. In International Conference on Machine Learning, pages 2688–2697. PMLR, 2018.

[6] Colin Graber and Alexander G. Schwing. Dynamic neural relational inference. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8513–8522, 2020.

[7] Aoran Wang and Jun Pang. Iterative structural inference of directed graphs. Advances in Neural Information Processing Systems, 35:8717–8730, 2022.

[8] Ferran Alet, Erica Weng, Tom´as Lozano-P´erez, and Leslie Pack Kaelbling. Neural relational inference with fast modular meta-learning. In Advances in Neural Information Processing Systems, volume 32, pages 11804–11815, 2019.

[9] Aoran Wang and Jun Pang. Structural inference of dynamical systems with conjoined state space models. In Advances in Neural Information Processing Systems, volume 37, pages 75355– 75391, 2024.

[10] Shuhan Zheng, Ziqiang Li, Kantaro Fujiwara, and Gouhei Tanaka. Difusion model for relational inference in interacting systems. IEEE Transactions on Network Science and Engineering, 13:1990–2003, 2026.

[11] Carnegie Mellon University. Carnegie-Mellon Motion Capture Database, 2003.

[12] Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, and Koray Kavukcuoglu. Interaction networks for learning about objects, relations and physics. In Advances in Neural Information Processing Systems, volume 29, pages 4502–4510, 2016.

[13] Zhang Zhang, Yi Zhao, Jing Liu, Shuo Wang, Ruyi Tao, Ruyue Xin, and Jiang Zhang. A general deep learning framework for network reconstruction and dynamics learning. Applied Network Science, 4:110, 2019.

[14] Siyuan Chen, Jiahai Wang, and Guoqing Li. Neural relational inference with eficient message passing mechanisms. In AAAI Conference on Artificial Intelligence, volume 35, pages 7055– 7063, 2021.

[15] Jiachen Li, Fan Yang, Masayoshi Tomizuka, and Chiho Choi. EvolveGraph: Multi-agent trajectory prediction with dynamic relational reasoning. In Advances in Neural Information Processing Systems, volume 33, pages 19783–19794, 2020.

[16] Gerrit Großmann, Julian Zimmerlin, Michael Backenk¨ohler, and Verena Wolf. Unsupervised relational inference using masked reconstruction. Applied Network Science, 8:18, 2023.

[17] Zhichao Han, Olga Fink, and David S. Kammer. Collective relational inference for learning heterogeneous interactions. Nature Communications, 15:3191, 2024.

[18] Luca Franceschi, Mathias Niepert, Massimiliano Pontil, and Xiao He. Learning discrete structures for graph neural networks. In International Conference on Machine Learning, volume 97, pages 1972–1982, 2019.

[19] Ruichu Cai, Yunjin Wu, Xiaokai Huang, Wei Chen, Tom Z. J. Fu, and Zhifeng Hao. Granger causal representation learning for groups of time series. Science China Information Sciences, 67(5):152103, 2024.

[20] Dan Wang, Yingjie Liu, and Bin Song. A credible trafic prediction method based on selfsupervised causal discovery. Science China Information Sciences, 67(5):152303, 2024.

[21] Ruochen Yang, Frederic Sala, and Paul Bogdan. Hidden network generating rules from partially observed complex networks. Communications Physics, 4:199, 2021.

[22] Jiaxu Cui, Qipeng Wang, Bingyi Sun, Jiming Liu, and Bo Yang. Learning continuous network emerging dynamics from scarce observations via data-adaptive stochastic processes. Science China Information Sciences, 67(12):222206, 2024.

[23] Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li. BRITS: Bidirectional recurrent imputation for time series. In Advances in Neural Information Processing Systems, volume 31, pages 6776–6786, 2018.

[24] Vincent Fortuin, Dmitry Baranchuk, Gunnar R¨atsch, and Stephan Mandt. GP-VAE: Deep probabilistic time series imputation. In International Conference on Artificial Intelligence and Statistics, volume 108, pages 1651–1661, 2020.

[25] Wenjie Du, David Cˆot´e, and Yan Liu. SAITS: Self-attention-based imputation for time series. Expert Systems with Applications, 219:119619, 2023.

[26] Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. CSDI: Conditional score-based difusion models for probabilistic time series imputation. In Advances in Neural Information Processing Systems, volume 34, pages 24804–24816, 2021.

[27] Andrea Cini, Ivan Marisca, and Cesare Alippi. Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks. In International Conference on Learning Representations, 2022.

[28] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In International Conference on Machine Learning, pages 3744–3753. PMLR, 2019.

[29] Harold W. Kuhn. The hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1–2):83–97, 1955.

[30] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.

[31] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations, 2017.

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, 2017.

[33] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. Advances in Neural Information Processing Systems, 33:11525–11538, 2020.

## A Architectural Details

## A.1 Neural Relational Inference Backbone

NRI learns a latent interaction graph from observed trajectories without ground-truth edge labels [5]. In SIHA, it is the structure-inference and future-prediction backbone that processes visible trajectories together with reconstructed hidden trajectories.

NRI is built on the VAE framework [30]. It consists of two core components: a GNN-based encoder and a trajectory-prediction decoder. The encoder takes a sequence of features $\mathbf { x } \in \mathbb { R } ^ { N \times T \times d }$

and encodes it into a distribution over latent edge types, typically modeled as categorical variables with K classes and represented by $\mathbf { z } \in \mathbb { R } ^ { N \times N \times \bar { K } }$ :

$$
\mathbf { h } = f _ { \mathrm { e n c } } ( \mathbf { x } ) , \ q _ { \phi } ( \mathbf { z } | \mathbf { x } ) = \mathrm { s o f t m a x } ( \mathbf { h } ) .\tag{11}
$$

Directly sampling the discrete adjacency tensor z from $q _ { \phi } ( \mathbf { z } | \mathbf { x } )$ will lead to a non-diferentiable process. To allow back-propagation for training, NRI adopts the Gumbel-Softmax trick [31]. The inferred graph z is then passed to the GNN-based decoder, which simulates the next-step dynamics using message passing on the sampled interaction graph. The decoder is trained to minimize the reconstruction loss between predicted and true future states, and is expected to model the interactive dynamic patterns of the system:

$$
{ \bf z } _ { i j } = \mathrm { s o f t m a x } \left( ( { \bf h } _ { i j } + { \bf g } ) / \tau \right) ,\tag{12}
$$

$$
p _ { \theta } ( \mathbf { x } | \mathbf { z } ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( \mathbf { x } ^ { t + 1 } | \mathbf { x } ^ { 1 : t } , \mathbf { z } ) .\tag{13}
$$

The NRI model is trained by maximizing the evidence lower bound (ELBO):

$$
\begin{array} { r } { \mathrm { E L B O } = \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } [ \log p _ { \theta } ( \mathbf { x } | \mathbf { z } ) ] - D _ { \mathrm { K L } } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) | | p ( \mathbf { z } ) ) , } \end{array}\tag{14}
$$

where the former term corresponds to minimizing the state prediction error, while the latter term serves as the latent space regularization to constrain the discrepancy between the posterior distribution $q _ { \phi } ( { \bf z } | { \bf x } )$ in the latent space and the prior distribution $p ( \mathbf { z } )$ (typically assumed to be uniform).

## A.2 Set Transformer and Hidden-State Predictor Architecture

The Set Transformer [28] is an attention-based neural network architecture designed to model higher-order interactions among set elements via attention mechanisms.

Given an input set $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \}$ with $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ , the encoder of the Set Transformer applies stacked Set Attention Blocks (SABs) to produce latent representations $\mathbf { Z } \in \mathbb { R } ^ { n \times d }$

$$
\operatorname { E n c o d e r } ( \mathbf { X } ) = \operatorname { S A B } ( \operatorname { S A B } ( \mathbf { X } ) ) .\tag{15}
$$

Each SAB captures interactions within the set through a Multihead Attention Block (MAB), defined as:

$$
\operatorname { S A B } ( \mathbf { X } ) : = \mathrm { M A B } ( \mathbf { X } , \mathbf { X } ) ,\tag{16}
$$

where $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ is both the query and the key/value input.

The MAB computes multihead cross attention between a query set $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ and a key/value set $\mathbf { Y } \in \mathbb { R } ^ { m \times d }$

$$
\operatorname { M A B } ( \mathbf { X } , \mathbf { Y } ) = \operatorname { L N } \big ( \mathbf { H } + \mathrm { r F F } ( \mathbf { H } ) \big ) ,\tag{17}
$$

$$
{ \mathrm { w h e r e ~ } } \mathbf { H } = \operatorname { L N } \left( \mathbf { X } + \operatorname { M u l t i h e a d } ( \mathbf { X } , \mathbf { Y } , \mathbf { Y } ) \right) ,\tag{18}
$$

where $\mathrm { L N } ( \cdot )$ denotes layer normalization and rFF(·) is a row-wise feedforward network. The multihead attention mechanism Multihead(Q, K, V) [32] is defined as:

$$
\operatorname { M u l t i h e a d } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \operatorname { C o n c a t } \left( \operatorname { h e a d } _ { 1 } , \dots , \operatorname { h e a d } _ { h } \right) \mathbf { W } ^ { O } ,\tag{19}
$$

where each head is computed by scaled dot-product attention:

$$
\mathrm { h e a d } _ { i } = \mathrm { s o f t m a x } \left( \frac { { \mathbf { Q } } { \mathbf { W } } _ { i } ^ { Q } ( { \mathbf { K } } { \mathbf { W } } _ { i } ^ { K } ) ^ { \top } } { \sqrt { d / h } } \right) { \mathbf { V } } { \mathbf { W } } _ { i } ^ { V } .\tag{20}
$$

Here h denotes the number of attention heads, and $\mathbf { W } _ { i } ^ { Q } , \mathbf { W } _ { i } ^ { K } , \mathbf { W } _ { i } ^ { V } , \mathbf { W } ^ { O }$ are learnable projection matrices.

For aggregation and output, the decoder uses a Pooling by Multihead Attention (PMA) module, which maps the latent representations to a fixed-size output set using k learnable seed vectors $\mathbf { S } \in \mathbb { R } ^ { k \times d }$ :

$$
\mathrm { D e c o d e r } ( \mathbf { Z } ) = \mathrm { r F F } ( \mathrm { S A B } ( \mathrm { P M A } _ { k } ( \mathbf { Z } ) ) ) \in \mathbb { R } ^ { k \times d } ,\tag{21}
$$

$$
\mathrm { w h e r e ~ \ P M A } _ { k } ( \mathbf { Z } ) : = \mathrm { M A B } ( \mathbf { S } , \mathrm { r F F } ( \mathbf { Z } ) ) \in \mathbb { R } ^ { k \times d } .\tag{22}
$$

The SIHA-specific HSP-sa encoder–decoder mapping is given in Section 4.2. HSP-sg retains this topology and applies the structural biases described in Section 4 to the encoder self-attention, PMA, and decoder self-attention blocks.

## B Training and Reproducibility Details

Algorithm 1 specifies the cache-based HSP-sg training procedure. This section reports the configurations used for (1) separate pretraining of HSP-sa and NRI, (2) iterative training of HSP-sg, and (3) deployment-time refinement.

## B.1 Pretraining of HSP-sa and NRI on Fully Observed Data

HSP-sa and NRI are pretrained separately on complete trajectories. For HSP-sa, a subset of agents is masked from the input and its ground-truth trajectories are used as supervised reconstruction targets. The synthetic systems use Hungarian-aligned MSE, whereas the motion-capture experiments use direct joint-wise MSE in the predefined masked-joint order. NRI receives the complete trajectories and is optimized with the standard NRI objective, without ground-truth interaction labels.

For the synthetic systems, the HSP-sa module is implemented as a Set Transformer with a hidden dimension of 256 and head number of 4. It is optimized using the Adam optimizer with a learning rate of 5e-4, a weight decay of 1e-6 for Springs and 1e-5 for Charged Particles and Kuramoto, and batch size of 128. Training is performed for 500 epochs.

For the synthetic systems, the NRI module adopts the standard encoder-decoder architecture with Gumbel-Softmax edge sampling. The number of edge types is fixed to 2. It is optimized using the Adam optimizer with a learning rate of 5e-4, and batch size of 128. Training is performed for 200 epochs. The pretrained models are then used to generate initial structure estimates and hidden state predictions for the iterative training stage.

## B.2 Training of Structure-Guided Predictor (HSP-sg)

The structure-guided hidden-state predictor (HSP-sg) is trained using structure estimates initialized by the pretrained HSP-sa and NRI modules. The predicted structure is stored in a per-sample cache and periodically recomputed by the pretrained NRI module. During this stage, HSP-sg is optimized with the dataset-appropriate hidden-state reconstruction loss, while the pretrained modules provide the initialization and structure updates. For the synthetic systems, this is Hungarian-aligned MSE; for motion capture, it is direct joint-wise MSE in the predefined masked-joint order. Algorithm 1 gives the common cache-based training procedure.

During each synthetic training phase, HSP-sg receives the visible trajectories and current structure cache as input. The predicted hidden states are combined with the visible ones and passed to the NRI model to update the structure estimates, which are written back to the cache every 10 epochs after warm-up cache freezing for the first 80 epochs.

Algorithm 1 Iterative Training of HSP-sg   
1: Input: Training dataset $\{ x _ { \mathrm { v i s } } , x _ { \mathrm { h i d } } \}$ , pre-trained $f _ { \mathrm { p r e } }$ and g<sub>NRI</sub>   
2: Output: Trained $f _ { \mathrm { s g } }$ and updated structure cache   
3: Initialize structure cache $\mathbf { A } _ { \mathrm { c a c h e } }$ with g<sub>NRI</sub>([x<sub>vis</sub>, f<sub>pre</sub>(x<sub>vis</sub>)])   
4: for epoch = 1 to E do   
5: for each mini-batch in training data do   
6: Fetch cached structure $\mathbf { A } _ { \mathrm { c a c h e } }$   
7: Use $f _ { \mathrm { s g } }$ with structure guidance $\mathbf { A } _ { \mathrm { c a c h e } }$ to predict hidden states $\hat { x } _ { \mathrm { h i d } }$   
8: Compute the dataset-appropriate hidden-state reconstruction loss   
9: Update parameters of $f _ { \mathrm { s g } }$ via backpropagation   
10: end for   
11: if epoch $> T _ { 0 }$ and epoch mod $M = 0$ then   
12: Update $\mathbf { A } _ { \mathrm { c a c h e } }$ using g<sub>NRI</sub> $( [ x _ { \mathrm { v i s } } , f _ { \mathrm { s g } } ( x _ { \mathrm { v i s } } , \mathbf { A } _ { \mathrm { c a c h e } } ) ] )$   
13: end if   
14: end for

For the synthetic systems, we train HSP-sg using the Adam optimizer with a learning rate of 5e-4, a weight decay of 1e-6 for Springs and 1e-5 for Charged Particles and Kuramoto, batch size of 128. Each full training run consists of 500 epochs, and MSE is computed after aligning predicted hidden states to ground truth via Hungarian matching.

## B.3 Evaluation-time Refinement via Iterative Structure Cache

At test time, the structure cache is initialized using the pretrained models and refined for 5 update rounds. In each round, the HSP-sg model is used to predict hidden states, followed by a forward pass through the NRI module to update structure predictions.

## B.4 System-specific Settings and Embedding Modules

For the Kuramoto system, we use a three-dimensional state representation consisting of phase diference, amplitude, and intrinsic frequency. We also fix the null interaction type as always inactive to reflect the system’s continuous coupling nature.

All systems use Set Transformer-based architectures for the HSP modules. For trajectory embedding, Springs uses an MLP-based embedding, whereas Charged Particles and Kuramoto use one-dimensional convolutional embeddings. These system-specific choices follow the corresponding NRI configurations to maintain architectural consistency with the NRI baselines.

## B.5 Motion-Capture Training Configuration

For both motion-capture subjects, HSP-sa and HSP-sg use a hidden dimension of 256, 4 attention heads, dropout of 0, a history length of 49, and an input dimension of 6 corresponding to 3D position and 3D velocity. With random seed 1, the HSP modules are trained for 500 epochs using a batch size of 8, a learning rate of 1e-4, and weight decay of 1e-6. Their outputs follow the predefined masked-joint order, and training uses direct joint-wise MSE. For HSP-sg, the structure cache has a warm-up of 40 epochs and is updated every 20 epochs; interaction guidance uses edge type index 1.

The motion-capture backbone is a static-graph NRI model with 2 edge types, an encoder hidden dimension of 256, an encoder MLP hidden dimension of 256 with 3 layers, and a decoder hidden dimension of 256. It uses skip first=true, a Gumbel-Softmax temperature of 0.5, and 10 teacher-forcing steps. Each subject-specific NRI model is trained for 500 epochs using a batch size of 8 and a learning rate of 5e-4.

## B.6 Dataset Generation and Scale

We use the Springs, Charged Particles, and Kuramoto systems as synthetic benchmark environments, following the trajectory-based evaluation setting used in NRI [5]. Each trajectory contains $T = 5 0$ time steps for training and validation and $T = 1 0 0$ time steps for testing. Springs and Charged Particles use four-dimensional position–velocity states, whereas Kuramoto uses the threedimensional representation consisting of phase diference, amplitude, and intrinsic frequency.

We use 200,000 samples for training, 50,000 for validation, and 50,000 for testing. These dataset sizes exceed those used in the original NRI experiments and are reported here as part of our experimental configuration.

## B.7 Compute Resources

All experiments were conducted on a single NVIDIA RTX 4090 GPU with 24 GB memory using PyTorch 2.6.0 and CUDA 12.6. A complete training run, including pretraining and iterative refinement, takes approximately 5–15 hours for systems with 5–10 agents. Training scripts, datageneration tools, and configuration files will be released publicly.

## C Complete Quantitative Results

## C.1 Variants without Hidden-State Supervision

We further study whether the hidden-state predictor can be trained without direct supervision on hidden states. We consider three variants: (1) jointly training NRI and HSP-sa from scratch, with both modules randomly initialized; (2) jointly training NRI and HSP-sa while initializing NRI from a model pretrained on fully observed data; and (3) training only HSP-sa while keeping an NRI model pretrained on fully observed data frozen. In all three cases, supervision is provided only through the future states of the visible agents. That is, HSP-sa is not directly supervised by ground-truth hidden trajectories, but remains in the end-to-end backpropagation chain through the visible future prediction loss. Variant (1) uses neither fully observed NRI pretraining nor direct hidden-state supervision. For reference, we also report the main supervised setting, in which ground-truth trajectories of the masked agents supervise HSP-sa reconstruction during training.

Table 4 summarizes the results on the Springs task with $N _ { \mathrm { { v i s } } } = 5$ and $N _ { \mathrm { h i d } } \in \{ 1 , 2 , 3 , 4 , 5 \}$ . All three variants without hidden-state supervision have substantially higher hidden-state and visiblefuture errors than the supervised HSP-sa setting, and their visible-to-visible structural accuracy also deteriorates as $N _ { \mathrm { h i d } }$ increases. These results characterize the behavior of the three tested visiblefuture-only training objectives; they do not constitute an impossibility result for other objectives or models. The main experiments therefore retain direct hidden-state supervision during training, as specified in Section 3.

Table 4: Results of variants without hidden-state supervision on the Springs task with $N _ { \mathrm { v i s } } = 5$ and $N _ { \mathrm { h i d } } \in \{ 1 , 2 , 3 , 4 , 5 \}$ . Results are in the format of ${ \mathrm { M S E } } _ { \mathrm { H S P } }$ / MSE<sub>FSP,Vis.</sub> / $\mathrm { \Delta A C C _ { V - V } }$ . Lower MSE and higher ACC are better.
<table><tr><td>Method</td><td> $N _ { \mathrm { h i d } } = 1$ </td><td> $N _ { \mathrm { h i d } } = 2$ </td><td> $N _ { \mathrm { h i d } } = 3$ </td><td> $N _ { \mathrm { h i d } } = 4$ </td><td> $N _ { \mathrm { h i d } } = 5$ </td></tr><tr><td>Joint train NRI + HSP-sa (random init)</td><td>4.3e-1 2.0e-4 /98.2</td><td>3.6e-1 1.8e-4 72.1</td><td>2.8e-1 2.3e-4</td><td>3.3e-1 2.3e-4 50.0</td><td>4.0e-1 / 2.8e-4 / 50.0</td></tr><tr><td>Joint train NRI + HSP-sa (pretrained NRI)</td><td>4.1e-1 2.2e-4 98.4</td><td>3.3e-1 2.3e-4 73.5</td><td>2.6e-1 2.4e-4 66.2</td><td>3.0e-1 2.6e-4 50.0</td><td> $3 . 7 \mathrm { e } { - 1 } \textrm { / } 2 . 9 \mathrm { e } { - 4 } \textrm { / } 5 0 . 0$ </td></tr><tr><td>Train HSP-sa only + freeze pretrained NRI</td><td>3.8e-1 2.2e-4 98.2</td><td>3.1e-1 2.4e-4 73.8</td><td>2.5e-1 2.4e-4 66.1</td><td>2.7e-1 2.8e-4 50.0</td><td>3.5e-1 3.1e-4 / 50.0</td></tr><tr><td>Supervised HSP-sa on fully observed data</td><td>2.5e-3 / 1.5e-5 / 99.7</td><td>5.6e-3 9.3e-6 98.7</td><td>7.1e-3 6.4e-6 97.0</td><td>8.1e-3 5.7e-6 95.6</td><td>9.1e-3 6.0e-6 93.0</td></tr></table>

## C.2 Ablation on Encoder Modules for HSP-sa

We perform an ablation study to evaluate the efect of diferent encoder architectures used in the structure-agnostic hidden state predictor (HSP-sa). Specifically, we replace the Set Transformer encoder with alternative modules including a standard MLP, vanilla Transformer, GNN, and Slot Attention [33] encoder. All variants are trained on the Springs dataset with $N _ { \mathrm { v i s } } = 5$ visible agents and $N _ { \mathrm { h i d } } = 2$ hidden agents, under the same training configuration and data size.

Table 5 reports the hidden-state prediction MSE for each encoder choice. The Set Transformer has the lowest reported MSE among these variants; this table does not by itself isolate which architectural property accounts for the diference.

Table 5: Hidden state prediction MSE for HSP-sa with diferent encoder modules on the Springs dataset $( N _ { \mathrm { v i s } } = 5 , N _ { \mathrm { h i d } } = 2 )$
<table><tr><td>Encoder Module</td><td>MSEHSP</td></tr><tr><td>MLP Transformer</td><td> $1 . 2 \times 1 0 ^ { - 2 }$   $1 . 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>GNN</td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Slot Attention</td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Set Transformer</td><td> $\mathbf { 5 . 6 \times 1 0 ^ { - 3 } }$ </td></tr></table>

## C.3 Repeated-run statistics on representative settings

To assess statistical stability, we report repeated-run results for three representative settings: Springs, Charged Particles, and Kuramoto with $N _ { \mathrm { { v i s } } } = 5$ and $N _ { \mathrm { h i d } } = 3$ . All models, including SIHA (HSP-sg) and the pretrained NRI and HSP-sa modules, are trained independently over five random seeds.

## D Additional Analyses and Motion-Capture Details

## D.1 Motion-Capture Focus-Graph Construction

The focus plots in Figure 4 are constructed around the left or right hand. For a focus joint i, the score of joint j is $s _ { j } = ( A _ { i j } + A _ { j i } ) / 2$ with $s _ { i } = 0$ . The visualization retains the top-k edges with score greater than 0.75: panels (a) and (b) use $k = 8$ , whereas panels (c)–(f) use $k = 1 4$ The full-observation NRI structure is shown only as a reference estimate and is not treated as a ground-truth graph.

Table 6: Five-run statistics on representative settings. We report mean ± standard deviation over 5 independent runs with diferent random seeds.
<table><tr><td>Dataset</td><td>Method</td><td> $\mathrm { M S E } _ { \mathrm { H S P } } \downarrow$ </td><td> $\mathrm { M S E } _ { \mathrm { F S P , V i s . } } \downarrow$ </td><td> $\mathrm { M S E } _ { \mathrm { F S P , H i d . } } \downarrow$ </td><td> $\operatorname { A C C } _ { \operatorname { V - V } } \uparrow$ </td><td> $\operatorname { A C C } _ { \operatorname { V - H } } \uparrow$ </td><td> $\operatorname { A C C } _ { \mathrm { H - H } } \uparrow$ </td></tr><tr><td rowspan="3">Springs</td><td>NRI</td><td></td><td> $3 . 7 \mathrm { e } { - 5 } \pm 2 . 3 \mathrm { e } { - 6 }$ </td><td></td><td> $7 2 . 9 \pm 0 . 2$ </td><td></td><td></td></tr><tr><td> $_ \mathrm { H S P - s a }$ </td><td> $7 . 1 \mathrm { e } \mathrm { - } 3 \pm 4 . 0 \mathrm { e } \mathrm { - } 5$ </td><td> $6 . 3 \mathrm { e } { - 6 } \pm 3 . 4 \mathrm { e } { - 7 }$ </td><td> $1 . 3 \mathrm { e } { - 2 } \pm 1 . 7 \mathrm { e - 3 }$ </td><td> $9 7 . 0 \pm 0 . 4$ </td><td> $8 3 . 4 \pm 0 . 8$ </td><td> $6 1 . 3 \pm 1 . 1$ </td></tr><tr><td>SIHA (HSP-sg)</td><td> $\mathbf { 4 . 9 \mathrm { e } { - } 3 \pm 6 . 7 \mathrm { e } { - } 5 }$ </td><td> ${ \bf 5 . 0 e - 6 \pm 3 . 1 e - 7 }$ </td><td> $\mathbf { 1 . 0 \mathrm { e } { - 2 } \pm 1 . 6 \mathrm { e } { - 3 } }$ </td><td> ${ \bf 9 7 . 2 \pm 0 . 4 }$ </td><td> ${ \bf 8 8 . 7 \pm 0 . 6 }$ </td><td> ${ \bf 6 7 . 6 \pm 1 . 0 }$ </td></tr><tr><td rowspan="3">Charged</td><td>NRI</td><td></td><td> $\mathbf { 2 . 5 e { - 3 } \pm 1 . 7 e { - 4 } }$ </td><td></td><td> $6 2 . 1 \pm 0 . 4$ </td><td></td><td></td></tr><tr><td> $_ \mathrm { H S P - s a }$ </td><td> $4 . 3 \mathrm { { e } - 2 \pm 2 . 0 \mathrm { { e } - 4 } }$ </td><td></td><td> $1 . 3 \mathrm { e } { - } 1 \pm 1 . 8 \mathrm { e } { - } 2$ </td><td> $7 1 . 3 \pm 0 . 7$ </td><td> $5 7 . 5 \pm 2 . 2$ </td><td> ${ \bf 5 2 . 1 \pm 2 . 4 }$ </td></tr><tr><td>SIHA (HSP-sg)</td><td> $\mathbf { 4 . 1 \mathrm { e } { - 2 } \pm 3 . 8 \mathrm { e } { - 4 } }$ </td><td> $\begin{array} { c } { 2 . 6 \mathrm { e - 3 } \pm 2 . 2 \mathrm { e - 4 } } \\ { 2 . 6 \mathrm { e - 3 } \pm 2 . 3 \mathrm { e - 4 } } \end{array}$ </td><td> $\mathbf { 1 . 2 e { - 1 } \pm 1 . 7 e { - 2 } }$ </td><td> ${ \bf 7 2 . 4 \pm 0 . 8 }$ </td><td> ${ \bf 5 8 . 9 \pm 2 . 5 }$ </td><td> ${ \bf 5 2 . 1 \pm 2 . 7 }$ </td></tr><tr><td rowspan="3">Kuramoto</td><td>NRI</td><td></td><td> $4 . 2 \mathrm { { e } - 2 \pm 1 . 9 \mathrm { { e } - 3 } }$ </td><td></td><td> $6 5 . 2 \pm 0 . 3$ </td><td></td><td></td></tr><tr><td> $_ \mathrm { H S P - s a }$ </td><td> $1 . 1 \mathrm { e } { - } 1 \pm 6 . 0 \mathrm { e } { - } 4$ </td><td> $1 . 6 \mathrm { { e } - 2 \pm 1 . 2 \mathrm { { e } - 3 } }$ </td><td> $1 . 3 \mathrm { e } { - } 1 \pm 1 . 8 \mathrm { e } { - } 2$ </td><td> $8 3 . 6 \pm 0 . 5$ </td><td> $6 8 . 6 \pm 0 . 7$ </td><td> $5 3 . 3 \pm 1 . 8$ </td></tr><tr><td> $\mathrm { S I H A \ ( H S P { - } s g ) }$ </td><td> $\mathbf { 1 . 0 e { - 1 } \pm 6 . 3 e { - 4 } }$ </td><td> $\mathbf { 1 . 5 \mathrm { e - 2 } \pm 1 . 3 \mathrm { e - 3 } }$ </td><td>1.2e−1 ± 1.6e−2</td><td> ${ \bf 8 4 . 0 \pm 0 . 7 }$ </td><td> ${ \bf 7 0 . 2 \pm 0 . 9 }$ </td><td> ${ \bf 5 4 . 8 \pm 2 . 0 }$ </td></tr></table>

## D.2 Unknown and Variable Numbers of Hidden Agents

We evaluate an existing model-selection procedure when the number of hidden agents is unknown and takes a value in $N _ { \mathrm { h i d } } \in \{ 0 , \dots , 5 \}$ $_ \mathrm { H S P - S a }$ and NRI models configured for diferent $N _ { \mathrm { h i d } }$ values are applied separately, and the selected count is the one yielding the lowest visible-agent future-prediction MSE. Table 7 reports the resulting count-prediction accuracy.

The accuracy remains above 83% for all reported Springs settings, but decreases from 92.3 to 45.5 on Charged and from 87.5 to 24.7 on Kuramoto as the true count changes from 0 to 5. The result is therefore treated as a boundary analysis rather than evidence that the current fixed-count formulation resolves unknown cardinality.

Table 7: Accuracy (%) of hidden-agent count prediction across diferent numbers of hidden agents $N _ { \mathrm { h i d } }$
<table><tr><td></td><td colspan="6"> $N _ { \mathrm { h i d } }$ </td></tr><tr><td>Dataset</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Springs</td><td>99.9</td><td>99.1</td><td>90.5</td><td>88.8</td><td>87.2</td><td>83.7</td></tr><tr><td>Charged</td><td>92.3</td><td>88.2</td><td>76.5</td><td>69.1</td><td>56.2</td><td>45.5</td></tr><tr><td>Kuramoto</td><td>87.5</td><td>67.0</td><td>49.5</td><td>36.7</td><td>28.4</td><td>24.7</td></tr></table>

## D.3 Additional Unmodeled Hidden Agents

We additionally evaluate a Springs setting with $N _ { \mathrm { h i d } } = 2$ and $N _ { \mathrm { { v i s } } } = 5$ , in which $N _ { \mathrm { a d d } } \in \{ 1 , 2 , 3 \}$ additional agents are hidden from the model while it is trained to reconstruct only the first two hidden trajectories from the five visible trajectories.

Table 8 shows progressive degradation in the three displayed metrics as $N _ { \mathrm { a d d } }$ increases. This experiment documents sensitivity to additional unmodeled agents; it is not used to claim general robustness to arbitrary hidden-agent configurations.

Table 8: SIHA (HSP-sg) results on Springs when the data contain additional unmodeled hidden agents. For comparison, the NRI reference has an $\mathrm { M S E } _ { \mathrm { F S P , ~ V i s . } }$ of 3.0e-5 and an $\mathrm { A C C } _ { \mathrm { V - V } }$ of 76.1.
<table><tr><td rowspan="2"></td><td colspan="4"> $N _ { \mathrm { a d d } }$ </td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td> ${ \mathrm { M S E } } _ { \mathrm { H S P } }$ </td><td> $4 . 9 \mathrm { { e } - 3 }$ </td><td> $1 . 2 \mathrm { e } { - 2 }$ </td><td> $1 . 3 \mathrm { e - 2 }$ </td><td> $1 . 5 \mathrm { { e } - 2 }$ </td></tr><tr><td> $\mathrm { M S E } _ { \mathrm { F S P , ~ V i s . } }$ </td><td> $5 . 0 \mathrm { e - 6 }$ </td><td> $7 . 1 \mathrm { e } { - 6 }$ </td><td> $7 . 9 \mathrm { e - 6 }$ </td><td> $9 . 0 \mathrm { e } { - 6 }$ </td></tr><tr><td> $\mathrm { A C C } _ { \mathrm { V - V } }$ </td><td>97.2</td><td>92.3</td><td>85.8</td><td>77.1</td></tr></table>