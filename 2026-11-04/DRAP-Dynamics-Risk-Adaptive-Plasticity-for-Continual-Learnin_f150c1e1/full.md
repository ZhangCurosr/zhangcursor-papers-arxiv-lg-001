# DRAP: Dynamics-Risk Adaptive Plasticity for Continual Learning in Physical AI

Oswaldo Ludwig

Email: oswaldoludwig@gmail.com

Abstract—Continual learning in physical AI differs from conventional continual learning because small per-step policy errors can accumulate over a trajectory. Each deviation may move the agent outside the training distribution, increase subsequent prediction errors, and ultimately cause closed-loop task failure. This work introduces Dynamics-Risk Adaptive Plasticity (DRAP), a replay-free method that estimates the historical importance of adaptable model tensors from sample-level action gradients weighted by local trajectory-amplification risk. A ridgeregularized local linear model is fitted to expert state-action transitions, and the largest singular value of its state-transition matrix provides a policy-independent, Lyapunov-inspired estimate of one-step perturbation amplification. After each task, risk-weighted squared gradients are accumulated in a tensorlevel importance memory. During subsequent tasks, DRAP uses normalized cumulative importance to scale the effective AdamW learning rate of each tensor, immediately reducing the plasticity of components that supported previous behavior at dynamically sensitive trajectory points. We instantiate DRAP on a fixed KappaTune-selected subnetwork of OpenVLA and evaluate replay-free sequential adaptation across the LIBERO Spatial, Object, and Goal suites. Compared with standard AdamW fine-tuning and EWC, the results show substantially improved retention of previously learned behaviors while preserving strong new-suite acquisition.

Index Terms—physical AI, VLA, continual learning, catastrophic forgetting, adaptive plasticity, robotics.

## I. INTRODUCTION

Continual learning in physical AI differs from conventional continual learning because small changes in a policy can alter future observations and amplify into closed-loop task failure. In embodied control, an action error changes the next physical state, modifies the next observation, and may place the policy outside the training distribution. Subsequent errors can therefore compound over the remainder of the trajectory [1], [2].

Vision-language-action (VLA) models combine visual perception, language grounding, and low-level control within a single pretrained policy. OpenVLA provides an open 7B VLA with tokenized actions and a pretrained vision-language backbone [3]. OpenVLA-OFT later demonstrated that continuous actions, action chunking, and optimized decoding can substantially improve success and inference efficiency [4]. These advances strengthen task acquisition, but do not remove the continual-learning problem: sequential fine-tuning can overwrite behavior required by earlier manipulation skills.

Existing continual-learning strategies offer complementary trade-offs. Replay remains a strong baseline but requires storing or regenerating previous observations. Expansion and routing methods isolate tasks but add modules or routing mechanisms. Parameter-regularization methods preserve a previous solution but can restrict acquisition. Direct updatesuppression methods reduce future plasticity according to past update history, but can become too restrictive for large heterogeneous models.

This work develops a replay-free mechanism based on a physical observation: parameters supporting correct behavior at dynamically sensitive trajectory regions should become less plastic in later tasks.

Unlike reference-based consolidation methods, the proposed Dynamics-Risk Adaptive Plasticity (DRAP) does not penalize parameter displacement from a previously stored checkpoint. Instead, similar to CLASSP [5], it uses accumulated historical importance to directly regulate future plasticity. Protection is therefore active from the first update of each subsequent learning stage and does not require storing reference copies of the adaptable parameters.

This work contributes a local policy-independent dynamicsrisk estimator derived from expert state-action trajectories. The resulting risk scores weight the historical importance of adaptable tensors, which is then used to regulate their future plasticity through tensor-specific AdamW learning rates. This mechanism suppresses the actual optimizer step rather than merely scaling raw gradients.

The remainder of this paper is organized as follows. Section II reviews related work on continual learning for visionlanguage-action models and on importance-based plasticity control. Section III describes the sequential LIBERO evaluation protocol. Section IV motivates the dynamics-risk signal from the compounding-error perspective of imitation learning, and Section V formalizes the replay-free continual learning problem. Section VI presents the DRAP method, including the KappaTune-based subnetwork selection, the local dynamicsrisk estimator, the risk-weighted tensor importance, and the tensor-specific learning-rate control. Section VII reports the experimental results, including ablations of KappaTune selection and of dynamics-risk weighting. Finally, Section VIII concludes and outlines directions for future work.

## II. RELATED WORK

Recent continual-learning methods for vision-languageaction models primarily rely on architectural modularity, routing, replay, or parameter regularization. CLARE [6] autonomously expands and routes lightweight adapters, allowing new manipulation tasks to be learned without requiring task identifiers at deployment. Stellar VLA [7] instead models task and skill relationships in a learned knowledge space and uses knowledge-guided expert routing, with limited replay, to reduce interference between sequentially learned behaviors. These approaches obtain stability by separating or routing task-specific capacity, whereas DRAP retains a fixed model architecture and controls the plasticity of an existing trainable subnetwork.

Among replay-free regularization methods, EWC [8], Synaptic Intelligence [9], Memory Aware Synapses [10], and RWalk [11] estimate the historical importance of model parameters and restrict subsequent changes to important parameters. EWC and RWalk employ reference-based quadratic penalties, while Synaptic Intelligence accumulates optimizationpath contributions and Memory Aware Synapses measures the sensitivity of the learned output function. Despite their different motivations, these importance estimators have been shown to be closely related to Fisher-information-based quantities [12]. DRAP differs by weighting sample-level actiongradient responsibility according to the estimated physical consequence of errors along the demonstrated trajectory.

DRAP is most closely related to continual-learning methods that regulate future parameter plasticity according to historical importance. CLASSP [5] suppresses future updates using accumulated gradient magnitudes and optionally applies a gradient threshold to promote sparse adaptation. Bayesian approaches such as uncertainty-guided continual learning [13], [14] similarly use parameter uncertainty to determine which model components should remain stable and which should retain greater plasticity. DRAP follows this general principle of importance-dependent plasticity, but replaces generic update history or posterior uncertainty with dynamics-risk-weighted, sample-level behavioral responsibility. The accumulated importance is defined at tensor level and directly scales tensorspecific AdamW learning rates. Thus, DRAP is inspired by CLASSP’s adjustment-suppression principle, but uses a different importance signal and optimizer interface.

Reference-based methods adopt a different protection mechanism. Elastic Weight Consolidation (EWC) [8] estimates parameter importance using a diagonal Fisher approximation and penalizes displacement from a previous optimum. In contrast, DRAP does not store reference parameters or apply a displacement penalty. Historical importance regulates plasticity immediately from the first update of each subsequent learning stage.

Liu et al. [15] investigate continual learning with large pretrained VLAs by sequentially training across the ten tasks of each LIBERO suite. Their main finding is that pretrained models such as $\pi _ { 0 }$ and GR00T N1.5 can achieve little forgetting when combined with experience replay, and retain comparatively low forgetting even when the replay buffer contains only a small fraction of the previous-task data. However, their zero-replay Sequential and EWC baselines on LIBERO-Object and LIBERO-10 still exhibit substantial negative backward transfer, indicating that pretraining alone does not prevent forgetting when earlier demonstrations are unavailable.

Their protocol differs from ours in three respects. First, they train a separate continual model for each LIBERO suite and learn the ten tasks within that suite sequentially, whereas our protocol updates one checkpoint across complete suites in the order Spatial, Object, and Goal. Second, their strongest retention results store transitions from previous tasks, while DRAP is evaluated without storing any previous-suite demonstrations. Third, they use different VLA architectures with LoRA-based or partially frozen fine-tuning, whereas we adapt a KappaTune-selected subnetwork of OpenVLA. The two settings therefore address complementary questions: their results show that pretrained VLAs exploit small replay buffers particularly well, whereas DRAP targets retention under a strictly replay-free sequence of suite-level shifts. Since the replay-free setting is already difficult in both protocols, and our sequence additionally requires retention across suites that differ in layout, object identity, and goal semantics, we consider it a demanding test of forgetting.

KappaTune [16] uses the matrix condition number as a scale-invariant proxy for information encoding in neural tensors. Its central hypothesis is that tensors with larger condition numbers contain more strongly differentiated singular directions and are therefore more likely to encode specialized pretrained information. KappaTune preserves these tensors and concentrates adaptation on tensors with smaller condition numbers, which are treated as safer sources of plasticity.

## III. EVALUATION PROTOCOL

LIBERO provides a benchmark for knowledge transfer and lifelong robot learning across Spatial, Object, Goal, and Long-Horizon suites [17]. LIBERO results depend strongly on the training protocol. OpenVLA-OFT reports a 97.1% average success rate using four independently fine-tuned policies, one for each of the Spatial, Object, Goal, and Long-Horizon suites [4]. It also reports 96.8% for a single policy jointly trained on data from all four suites. These settings evaluate suitespecific or joint multi-suite acquisition. This work instead evaluates sequential continual learning without replay. A single checkpoint is trained successively on Spatial, Object and Goal, with only the current suite available at each stage. After every stage, the checkpoint is evaluated on all suites observed so far. The resulting lower-triangular success matrix therefore measures both new-suite acquisition and backward retention under closed-loop control.

As shown by the standard sequential-training baseline, this protocol presents a challenging catastrophic-forgetting setting, with performance on earlier suites degrading substantially after adaptation to subsequent suites.

## IV. COMPOUNDING ERRORS AND DYNAMICS-RISK MOTIVATION

Behavior cloning is affected by covariate shift because policy-generated actions modify future states and observations [1], [2]. A small action error can move the robot away from the demonstrated trajectory, expose the policy to unfamiliar observations, and increase the probability of subsequent errors.

In physical AI, these deviations can compound over a closed-loop rollout and cause task failure even when the per step prediction error remains small. DRAP addresses this problem by assigning greater importance to demonstration samples at which the local physical dynamics are more likely to amplify a state perturbation. The risk estimator is inspired by Lyapunov’s indirect method, which studies nonlinear systems locally through linearization [18], and by contraction analysis, which examines the local convergence or divergence of neighboring trajectories [19]. DRAP uses this local differential perspective to rank demonstration samples according to their estimated one-step perturbation amplification.

The resulting dynamics-risk score is a relative sensitivity indicator, its purpose is to identify trajectory regions where preserving accurate policy behavior may be particularly important because small deviations can have larger downstream physical consequences.

## V. PROBLEM FORMULATION

Tasks arrive sequentially as $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { K }$ . Each task dataset consists of expert transitions:

$$
( o _ { t } , s _ { t } , a _ { t } , s _ { t + 1 } , e _ { t } ) ,\tag{1}
$$

where $o _ { t }$ is the visual-language observation, $s _ { t } ~ \in ~ \mathbb { R } ^ { d _ { s } }$ is a logged robot state, $a _ { t } ~ \in ~ \mathbb { R } ^ { d _ { a } }$ is the expert action, and $e _ { t }$ identifies the episode. The VLA policy is $\pi _ { \theta } ( a \mid o )$ with parameters θ.

A fixed set of trainable tensors, selected using KappaTune, is denoted by $\tau$ . For tensor $T \in \mathcal { T } , I _ { T } ^ { ( k ) } \geq 0$ denotes cumulative importance after task k. The objective is to learn task $k + 1$ while regulating each tensor’s plasticity according to all previous task importance, without storing previous demonstrations.

## VI. METHOD

## A. KappaTune selection

For each eligible matrix W, KappaTune computes a regularized condition number

$$
\kappa ( W ) = \frac { \sigma _ { \mathrm { m a x } } ( W ) } { \sigma _ { \mathrm { m i n } } ( W ) + \varepsilon _ { \kappa } } .\tag{2}
$$

Within each architectural group (vision, language and action head), a fixed budget of low-κ tensors is selected from the pretrained checkpoint. Selection is computed once and reused throughout the task sequence. Parameters outside $\tau$ remain frozen. Therefore, KappaTune defines where adaptation may occur, while DRAP determines how much future plasticity each selected tensor retains.

## B. Local linear dynamics identification

State and action dimensions are robustly standardized using suite-level statistics. For a sliding window $\mathcal { W } _ { t }$ contained within one episode, ridge regression estimates

$$
\tilde { s } _ { j + 1 } = A _ { t } \tilde { s } _ { j } + B _ { t } \tilde { a } _ { j } + c _ { t } + \epsilon _ { j } , \qquad j \in \mathcal { W } _ { t } ,\tag{3}
$$

where $A _ { t } \in \mathbb { R } ^ { d _ { s } \times d _ { s } }$ is the local state-transition matrix, $B _ { t } \in$ $\mathbb { R } ^ { d _ { s } \times d _ { a } }$ is the local control matrix, $c _ { t }$ is an intercept, and $\epsilon _ { j }$ is the residual. The model is local and trajectory-conditioned; it is not a global linear model of manipulation dynamics. See Figure 1.

## C. Lyapunov-inspired dynamics risk

For a small state perturbation under the locally fixed logged action,

$$
\delta s _ { t + 1 } \approx A _ { t } \delta s _ { t } ,\tag{4}
$$

define the quadratic perturbation energy:

$$
V _ { s } ( \delta s ) = \delta s ^ { \top } \delta s .\tag{5}
$$

Then the following upper-bound holds:

$$
\begin{array} { r } { V _ { s } ( \delta s _ { t + 1 } ) = \delta s _ { t } ^ { \top } A _ { t } ^ { \top } A _ { t } \delta s _ { t } } \\ { \leq \sigma _ { \operatorname* { m a x } } ^ { 2 } ( A _ { t } ) V _ { s } ( \delta s _ { t } ) . } \end{array}\tag{6}
$$

(7)

Accordingly, we define the raw one-step dynamics sensitivity:

$$
\rho _ { t } = \sigma _ { \operatorname* { m a x } } ( A _ { t } ) .\tag{8}
$$

Let $\mathcal { D } _ { k }$ denote the set of demonstration samples in suite k. The corresponding training weight $r _ { t }$ is obtained by normalizing $\rho _ { t }$ with respect to the median raw dynamics sensitivity over $\mathcal { D } _ { k }$ and clipping the result:

$$
r _ { t } = \mathrm { c l i p } \left( \frac { \rho _ { t } } { \mathrm { m e d i a n } _ { j \in \mathcal { D } _ { k } } ( \rho _ { j } ) + \varepsilon _ { r } } , c _ { \mathrm { m i n } } , c _ { \mathrm { m a x } } \right) .\tag{9}
$$

where $\varepsilon _ { r } > 0$ ensures numerical stability, and $c _ { \mathrm { m i n } }$ and $c _ { \mathrm { m a x } }$ denote the lower and upper clipping bounds, respective $\mathrm { l y ^ { 1 } }$

This dynamics-risk score does not constitute a formal proof or certificate of closed-loop stability. Even if the policyfeedback contribution were included, the resulting closed-loop Jacobian would remain a first-order local approximation of the nonlinear robot–policy system around the expert trajectory. To clarify this limitation, consider the local dynamics model in (3), where $A _ { t }$ describes the sensitivity of the standardized next state $\tilde { s } _ { t + 1 }$ to the standardized current state $\tilde { s } _ { t } , B _ { t }$ describes the sensitivity of $\tilde { s } _ { t + 1 }$ to the standardized action $\tilde { a } _ { t }$ , and $c _ { t }$ is an affine offset. Let $o \big ( \tilde { s } \big )$ denote the observation generated from the standardized state, and let $\tilde { \pi } _ { \boldsymbol { \theta } }$ denote the policy expressed in standardized action coordinates. The local sensitivity of the policy action to the state is:

$$
J _ { \pi , t } = \left. \frac { \partial \tilde { \pi } _ { \theta } ( o ( \tilde { s } ) ) } { \partial \tilde { s } } \right| _ { \tilde { s } = \tilde { s } _ { t } } .\tag{10}
$$

The corresponding first-order closed-loop Jacobian is:

![](images/d49eb932d9cb3a672b16d583ddf9cdd9602bef50f4158902cf7f1fead00aa6c0.jpg)  
Fig. 1. Dynamics-risk computation from expert demonstrations. Local linear dynamics are estimated over sliding windows of logged state-action transitions. The largest singular value $\sigma _ { \operatorname* { m a x } } ( A _ { t } )$ defines a local one-step perturbation-amplification score, which is robustly normalized and stored as a sample-level risk weight aligned with the training dataset.

## D. Sample-Level Risk-Weighted Tensor Importance

$$
J _ { \mathrm { C L } , t } \approx A _ { t } + B _ { t } J _ { \pi , t } .\tag{11}
$$

The raw dynamics-risk score $\rho _ { t } ~ = ~ \sigma _ { \operatorname* { m a x } } ( A _ { t } )$ captures only the direct state-sensitivity term $A _ { t }$ and does not include the policy-feedback term $B _ { t } J _ { \pi , t }$ . However, even $J _ { \mathrm { C L } , t }$ would describe perturbation evolution only locally, near the linearization point and within the fitted operating regime. It would not account for higher-order nonlinearities, contactmode transitions, observation aliasing, local-model estimation error, or trajectories that leave the neighborhood of the expert demonstration. A formal stability analysis would require additional conditions, such as a valid Lyapunov function over a specified region and a corresponding decrease condition along the actual nonlinear closed-loop dynamics.

Accordingly, the normalized score $r _ { t }$ is interpreted only as a relative, policy-independent indicator of potential one-step physical perturbation amplification along the expert trajectory.

After completing task k, tensor importance is estimated from a deterministic subset of $N _ { k }$ demonstration samples. A demonstration sample i is a training example containing the policy observation $o _ { i } ,$ , the corresponding expert action ${ { a } _ { i } } ,$ and the aligned normalized dynamics-risk score $r _ { i } .$ The stateaction transition used to estimate $r _ { i }$ is associated with the same time step.

The importance-estimation loader uses a batch size of one. Consequently, the gradient produced by each sample is squared before being combined with the contributions of other samples. This prevents gradients associated with different samples, episodes, task phases, or action directions from canceling one another before the importance is calculated.

In the evaluated VLA, the continuous expert action $a _ { i }$ is represented by a short sequence of discrete action tokens. Let $\mathcal { Q } _ { i }$ denote the set of output-sequence positions associated with the expert action for sample i. An action-token position in the model output sequence represents one component of the discretized expert action. Positions corresponding to image patches, language instructions, prompt tokens, padding tokens, or other non-action content are excluded from $\mathcal { Q } _ { i }$

The sample-level imitation loss is defined as

$$
\ell _ { i } ( \theta ) = \frac { 1 } { | \mathscr { Q } _ { i } | } \sum _ { q \in \mathscr { Q } _ { i } } \ell _ { i , q } ( \theta ) ,\tag{12}
$$

where $\ell _ { i , q } ( \theta )$ is the cross-entropy loss between the target action token and the token distribution predicted at position $q ,$ and θ denotes the model parameters.

Let $T \in \mathcal { T }$ be an adaptable parameter tensor, where $\tau$ denotes the set of tensors selected via KappaTune for continual training. Let n<sub>T</sub> denote the number of scalar parameters contained in $T .$ The gradient $\nabla _ { T } \ell _ { i } ( \theta )$ has the same shape as $T$ and measures the sensitivity of the sample-level action loss to changes in that tensor.

Since $r _ { i }$ is precomputed from the demonstration trajectory and is independent of the model parameters,

$$
\begin{array} { r l } & { \left. \nabla _ { T } \left( r _ { i } \ell _ { i } ( \theta ) \right) \right. _ { F } ^ { 2 } = \left. r _ { i } \nabla _ { T } \ell _ { i } ( \theta ) \right. _ { F } ^ { 2 } } \\ & { \qquad = r _ { i } ^ { 2 } \left. \nabla _ { T } \ell _ { i } ( \theta ) \right. _ { F } ^ { 2 } , } \end{array}\tag{13}
$$

where $\Vert \cdot \Vert _ { F }$ denotes the Frobenius norm. Accordingly, the raw new-task importance of tensor T after task k is

$$
\widehat { I } _ { T , \mathrm { n e w } } ^ { ( k ) } = \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N _ { k } } { r _ { i } ^ { 2 } \frac { \left\| \nabla _ { T } \ell _ { i } ( \theta ) \right\| _ { F } ^ { 2 } } { n _ { T } } } .\tag{14}
$$

The quantity $\lVert \nabla _ { T } \ell _ { i } ( { \boldsymbol { \theta } } ) \rVert _ { F } ^ { 2 } / n _ { T }$ is the mean squared gradient over all scalar elements of tensor T. Division by $n _ { T }$ prevents larger tensors from receiving greater importance solely because they contain more parameters.

The estimator therefore contains three distinct reductions. First, the cross-entropy losses are averaged over the supervised action-token positions of each sample through (12). Second, the squared gradient is averaged over the $n _ { T }$ scalar elements of tensor T. Third, the resulting risk-weighted contributions are averaged over the $N _ { k }$ demonstration samples. The estimator produces one scalar raw importance coefficient for each adaptable tensor.

To express the normalization applied to both the new-task and cumulative importance, we define the normalization of a nonnegative tensor-level quantity $\{ x _ { T } \} _ { T \in T }$ as:

$$
\mathcal { N } ( x ) _ { T } = \frac { x _ { T } } { \mathrm { m e d i a n } _ { U \in \mathcal { T } : x _ { U } > 0 } ( x _ { U } ) + \varepsilon } .\tag{15}
$$

where $\varepsilon \ > \ 0$ is a small constant that ensures numerical stability.

Accordingly, the new-task importance is normalized before being added to the cumulative memory:

$$
\widetilde { I } _ { T , \mathrm { n e w } } ^ { ( k ) } = \mathcal { N } \Big ( \widehat { I } _ { \mathrm { n e w } } ^ { ( k ) } \Big ) _ { T } .\tag{16}
$$

This normalization removes the task-dependent global scale while preserving the relative importance distribution across tensors.

Algorithm 1 KappaTune + DRAP   
Require: Pretrained policy $\theta _ { 0 } ,$ task stream $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K }$   
1: Select fixed trainable tensor set $\tau$ with KappaTune   
2: Initialize $I _ { T } ^ { ( 0 ) }  0$ for all $T \in { \mathcal { T } }$   
3: for $k = 1 , \bar { \dots } , K$ do   
4: Compute $\bar { I } _ { T } ^ { ( k - 1 ) }$ by (18) and $s _ { T } ^ { ( k - 1 ) }$ by (19)   
5: Create AdamW groups with $\eta _ { T } ^ { \bf { \dot { \alpha } } } = \eta s _ { T } ^ { ( \mathrm { \dot { \alpha } } - 1 ) }$   
6: for each optimization step on $\mathcal { D } _ { k }$ do   
7: Compute imitation loss and accumulated gradient   
8: Apply AdamW update using tensor-specific $\eta _ { T }$   
9: end for   
10: Compute $\widehat { I } _ { T , \mathrm { n e w } } ^ { ( k ) }$ by (14) and $\widetilde { I } _ { T , \mathrm { n e w } } ^ { ( k ) }$ by (16)   
11: Merge cumulative importance by (17)   
12: Save checkpoint, fixed tensor set T   
13: Save cumulative importance $\{ I _ { T } ^ { ( k ) } \} _ { T \in \mathcal { T } }$   
14: end for

The normalized new-task contribution is then merged into the cumulative importance memory:

$$
I _ { T } ^ { ( k ) } = \gamma I _ { T } ^ { ( k - 1 ) } + \widetilde { I } _ { T , \mathrm { n e w } } ^ { ( k ) } ,\tag{17}
$$

where $I _ { T } ^ { ( k - 1 ) }$ contains the importance accumulated from previous tasks and $\gamma \in ( 0 , 1 ]$ controls the retention of this historical importance. The default setting $\gamma = 1$ preserves contributions from all earlier tasks.

Before training task $k + 1$ , the cumulative importance is normalized using the operator (15):

$$
\overline { { I } } _ { T } ^ { ( k ) } = \mathcal { N } \Big ( I ^ { ( k ) } \Big ) _ { T } .\tag{18}
$$

DRAP converts this normalized cumulative importance into the tensor-specific plasticity multiplier

$$
s _ { T } ^ { ( k ) } = \left( 1 + \beta \overline { { I } } _ { T } ^ { ( k ) } \right) ^ { - 1 / p } ,\tag{19}
$$

where $\beta \geq 0$ controls the suppression strength and $p > 0$ controls its curvature. The effective AdamW learning rate assigned to tensor T during task $k + 1$ is

$$
\eta _ { T } ^ { ( k + 1 ) } = \eta s _ { T } ^ { ( k ) } ,\tag{20}
$$

where $\eta$ is the base learning rate.

Thus, $I _ { T } ^ { ( k ) }$ directly regulates the future plasticity of tensor $T \colon$ tensors with larger historical importance receive smaller AdamW steps from the beginning of the next training stage, whereas tensors with lower importance remain comparatively plastic. Algorithm 1 summarizes the method, and Figure 2 illustrates its overall workflow.

## VII. EXPERIMENTAL RESULTS

We evaluate five sequential-training configurations on the task order Spatial → Object → Goal: standard sequential fine-tuning, KappaTune+EWC, KappaTune+DRAP, KappaTune+DRAP with uniform risk, and DRAP without KappaTune. The uniform-risk ablation sets $r _ { i } = 1$ for every sample while retaining sample-level gradient importance, cumulative importance memory, and tensor-specific AdamW learning rates<sup>2</sup>.

![](images/8fe1c852070e76a0f81f4142f615dbcc2f42ad73d077a97e086b0d6175266bc6.jpg)  
Fig. 2. Overview of DRAP. Expert trajectories provide a local dynamics-risk signal. Sample-level action gradients identify tensor responsibility at risky samples. Cumulative importance directly scales future AdamW learning rates; no previous tensor reference or displacement penalty is required.

TABLE I  
COMPARISON OF CLASSP, EWC, AND THE PROPOSED DRAP.
<table><tr><td>Dimension</td><td>CLASSP</td><td>EWC</td><td>Proposed DRAP</td></tr><tr><td>Historical signal</td><td>Powered gradient/update accumulation</td><td>Diagonal Fisher information estimated from previous-task gradients</td><td>Dynamics-risk-weighted sample-level gradients</td></tr><tr><td>Physical grounding</td><td>None</td><td>None</td><td>Local trajectory-amplification risk</td></tr><tr><td>Protection mechanism</td><td>Update denominator and gradient threshold</td><td>Fisher-weighted quadratic displacement penalty</td><td>Tensor-specific AdamW learning rate</td></tr><tr><td>Reference weights</td><td>Not required</td><td>Required,  $\theta ^ { \star }$ </td><td>Not required</td></tr><tr><td>Onset of protection</td><td>Immediate</td><td>Zero at the new-task initial point; in- creases with displacement from θ*</td><td>Immediate</td></tr><tr><td>Optimizer</td><td>Custom CLASSP optimizer</td><td>AdamW plus EWC regularization loss</td><td>AdamW with per-tensor learning-rate groups</td></tr><tr><td>Threshold</td><td>Core sparsity mechanism</td><td>None</td><td>None</td></tr><tr><td>Importance granularity</td><td>Usually parameter-level accumulated adjustment</td><td>Parameter-level diagonal Fisher</td><td>Tensor-level cumulative importance from sample-level gradients</td></tr><tr><td>Primary risk</td><td>Excessive suppression can block new- task acquisition</td><td>Inaccurate Fisher estimates or excessive regularization can restrict plasticity</td><td>Important tensors can still accumulate drift through repeated reduced updates</td></tr><tr><td>Replay</td><td>Not required</td><td>Not required after Fisher estimation</td><td>Not required after importance estima- tion</td></tr><tr><td>Stored continual-learning state</td><td>Accumulated adjustment statistics</td><td>Diagonal Fisher and reference parame- ters</td><td>Cumulative tensor importance only</td></tr></table>

Each checkpoint is evaluated after every training stage on all suites observed up to that point. To reduce finite-rollout variability, this experiment pools 6 Gym evaluation runs, each comprising 50 episodes. These repeated evaluations quantify rollout variability for fixed checkpoints.

## A. Evaluation Metrics

Let $S _ { k , j }$ denote the pooled success rate on suite $j$ after training stage k. The average success over all suites observed by stage k is

$$
\operatorname { A v g S u c c e s s } _ { k } = { \frac { 1 } { k + 1 } } \sum _ { j = 0 } ^ { k } S _ { k , j } .\tag{21}
$$

For a previous suite $j < k ,$ , forgetting is measured relative to its best earlier performance:

$$
F _ { k , j } = \operatorname* { m a x } _ { m \in \{ j , \ldots , k - 1 \} } S _ { m , j } - S _ { k , j } .\tag{22}
$$

A positive value indicates forgetting.

Average forgetting after stage k is

$$
\mathrm { A v g F o r g e t } _ { k } = \frac { 1 } { k } \sum _ { j = 0 } ^ { k - 1 } F _ { k , j } .\tag{23}
$$

## B. Sequential Success and Forgetting Rates

Table II reports the pooled lower-triangular success matrix for each method. Each row corresponds to a training stage, and each column reports closed-loop success on an observed LIBERO suite.

Table III summarizes average success and forgetting after Object and Goal training.

TABLE II  
SEQUENTIAL LIBERO SUCCESS RATES ACROSS REPEATED GYM EVALUATIONS.
<table><tr><td>Method</td><td>Checkpoint</td><td>Spatial</td><td>Object</td><td>Goal</td></tr><tr><td rowspan="3">Standard fine-tuning</td><td>After Spatial</td><td>0.77</td><td></td><td>一</td></tr><tr><td>After Object</td><td>0.11</td><td>0.83</td><td>一</td></tr><tr><td>After Goal</td><td>0.00</td><td>0.07</td><td>0.71</td></tr><tr><td rowspan="3">KappaTune+EWC</td><td>After Spatial</td><td>0.76</td><td>1</td><td>一</td></tr><tr><td>After Object</td><td>0.41</td><td>0.82</td><td></td></tr><tr><td>After Goal</td><td>0.03</td><td>0.30</td><td>0.64</td></tr><tr><td rowspan="3">KappaTune+DRAP</td><td>After Spatial</td><td>0.77</td><td></td><td>一</td></tr><tr><td>After Object</td><td>0.75</td><td>0.80</td><td></td></tr><tr><td>After Goal</td><td>0.34</td><td>0.61</td><td>0.71</td></tr><tr><td rowspan="3">KappaTune+Uniform-risk DRAP</td><td>After Spatial</td><td>0.78</td><td></td><td>一</td></tr><tr><td>After Object</td><td>0.64</td><td>0.80</td><td></td></tr><tr><td>After Goal</td><td>0.29</td><td>0.57</td><td>0.66</td></tr><tr><td rowspan="3">DRAP without KappaTune</td><td>After Spatial</td><td>0.78</td><td></td><td>一</td></tr><tr><td>After Object</td><td>0.71</td><td>0.82</td><td></td></tr><tr><td>After Goal</td><td>0.11</td><td>0.08</td><td>0.66</td></tr></table>

TABLE III

AVERAGE SUCCESS AND FORGETTING DERIVED FROM THE POOLED SEQUENTIAL EVALUATION MATRICES. FORGETTING IS MEASURED RELATIVE TO THE BEST EARLIER CHECKPOINT FOR EACH PREVIOUSLY LEARNED SUITE.
<table><tr><td rowspan="2">Method</td><td colspan="2">After Object</td><td colspan="4">After Goal</td></tr><tr><td>Avg. success</td><td>Spatial forget.</td><td>Avg. success</td><td>Spatial forget.</td><td>Object forget.</td><td>Avg. forget.</td></tr><tr><td>Standard fine-tuning</td><td>0.47</td><td>0.66</td><td>0.26</td><td>0.77</td><td>0.76</td><td>0.77</td></tr><tr><td>KappaTune+EWC</td><td>0.62</td><td>0.35</td><td>0.32</td><td>0.73</td><td>0.52</td><td>0.63</td></tr><tr><td>KappaTune+DRAP</td><td>0.77</td><td>0.02</td><td>0.55</td><td>0.43</td><td>0.19</td><td>0.31</td></tr><tr><td>KappaTune+Uniform-risk DRAP</td><td>0.72</td><td>0.14</td><td>0.51</td><td>0.49</td><td>0.23</td><td>0.36</td></tr><tr><td>DRAP without KappaTune</td><td>0.77</td><td>0.07</td><td>0.28</td><td>0.67</td><td>0.74</td><td>0.71</td></tr></table>

Standard sequential fine-tuning exhibits strongly recencydominated behavior. Each newly introduced suite is acquired successfully, while performance on earlier suites rapidly approaches zero.

KappaTune+EWC provides partial protection during the first task transition, but its retention deteriorates substantially after Goal adaptation. EWC preserves more of the immediately preceding Object suite than standard fine-tuning, but offers little protection for the earliest Spatial suite. This result indicates that Fisher-weighted anchoring improves short-term retention but remains insufficient over the complete evaluated sequence.

KappaTune+DRAP achieves the strongest overall stabilityplasticity balance. The resulting success matrix yields both the highest average success and the lowest average forgetting after Object and Goal training.

## C. Ablation of KappaTune Selection

DRAP without KappaTune initially exhibits strong plasticity and performs competitively through the Object stage. However, after Goal adaptation, performance on both previous suites approaches the standard fine-tuning baseline. The broader adaptable parameter space permits effective short-term acquisition but exposes more pathways for destructive interference. The experiment indicates that KappaTune and DRAP provide complementary forms of protection. KappaTune restricts adaptation to a structurally selected subnetwork, while DRAP allocates plasticity within that subnetwork according to accumulated historical responsibility.

## D. Ablation of Dynamics-Risk Weighting

The uniform-risk ablation sets $r _ { i } = 1$ for every sample while preserving all other elements of DRAP, including sample-level squared-gradient importance, cumulative importance memory, tensor-wise normalization, and tensor-specific AdamW learning rates. It therefore isolates the contribution of the demonstration-derived physical-risk signal from the broader adaptive-plasticity mechanism.

Uniform-risk DRAP retains much of the improvement over standard fine-tuning and EWC, showing that historical tensor importance and tensor-specific learning rates account for a substantial part of DRAP’s performance. Nevertheless, dynamics-risk DRAP provides better retention of both Spatial and Object. New-suite acquisition remains comparable, indicating that the additional retention does not result from simply preventing the model from learning the current suite.

These initial results suggest that physics-informed plasticity control is a promising direction and motivate extending the dynamics-risk estimator beyond the state-transition term

A to include control-path sensitivity, multi-step perturbation gains, and contact-aware or learned local dynamics, potentially yielding a stronger stability-plasticity balance.

## VIII. CONCLUSION

This work takes a first step toward physics-informed plasticity control by introducing DRAP, a replay-free method for continual learning in physical AI. DRAP estimates tensor importance from sample-level action gradients weighted by local trajectory-amplification risk and uses the accumulated importance to regulate tensor-specific AdamW learning rates during subsequent adaptation. Intuitively, DRAP suppresses future updates to tensors that were important for predicting expert actions at samples where errors are more strongly amplified by the physical dynamics.

Sequential LIBERO experiments showed that standard finetuning suffers severe catastrophic forgetting, while the physically ungrounded method, EWC, provides only partial retention. DRAP achieved the strongest stability-plasticity balance when combined with KappaTune. KappaTune restricts adaptation to a structurally selected subnetwork, while DRAP distributes plasticity within that subnetwork according to historical, dynamics-risk-weighted responsibility. The ablation without KappaTune confirmed that broader plasticity improves initial acquisition but increases later interference.

Sequential LIBERO experiments showed that standard finetuning suffers severe catastrophic forgetting, while EWC provides only partial retention. KappaTune+DRAP achieved the strongest stability-plasticity balance. KappaTune restricts adaptation to a selected subnetwork, while DRAP regulates plasticity within that subnetwork according to accumulated historical importance. The ablation without KappaTune showed that a broader adaptable parameter space supports strong initial acquisition but substantially increases later interference. The uniform-risk ablation showed that importancedependent plasticity accounts for most of DRAP’s benefit, while dynamics-risk weighting provides an additional improvement in retention without impairing new-suite acquisition.

These results support the hypothesis that continual learning in physical AI should consider both model sensitivity and the physical consequences of behavioral errors. Future work will improve the dynamics-risk estimator by incorporating the control pathway, online estimation, multi-step perturbation propagation, and contact-aware dynamics. It will also investigate the CLASSP-inspired update threshold and evaluate DRAP over longer continual-learning sequences.

## REFERENCES

[1] S. Ross and D. Bagnell, “Efficient reductions for imitation learning,” in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 2010.

[2] S. Ross, G. J. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, 2011.

[3] M. J. Kim et al., “OpenVLA: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[4] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” in Robotics: Science and Systems, 2025.

[5] O. Ludwig, “CLASSP: A biologically-inspired approach to continual learning through adjustment suppression and sparsity promotion,” arXiv preprint arXiv:2405.09637, 2024.

[6] R. Romer, Y. Zhang, Y. Li, and A. P. Schoellig, “CLARE: Continual¨ learning for vision-language-action models via autonomous adapter routing and expansion,” IEEE Robotics and Automation Letters, 2026.

[7] Y. Wu, G. Wang, Z. Yang, T. Deng, M. Yao, B. Sheil, and H. Wang, “Continually evolving skill knowledge in vision language action model,” arXiv preprint arXiv:2511.18085, 2025.

[8] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[9] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in Proceedings of the 34th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 70, 2017, pp. 3987–3995.

[10] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory aware synapses: Learning what (not) to forget,” in Proceedings of the European Conference on Computer Vision, 2018, pp. 139–154.

[11] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr, “Riemannian walk for incremental learning: Understanding forgetting and intransigence,” in Proceedings of the European Conference on Computer Vision, 2018, pp. 532–547.

[12] F. Benzing, “Unifying importance based regularisation methods for continual learning,” in Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 151, 2022, pp. 2372–2396.

[13] S. Ebrahimi, M. Elhoseiny, T. Darrell, and M. Rohrbach, “Uncertaintyguided continual learning with bayesian neural networks,” in International Conference on Learning Representations, 2020.

[14] H. Ahn, S. Cha, D. Lee, and T. Moon, “Uncertainty-based continual learning with adaptive regularization,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[15] H. Liu, C. Kim, B. Liu, M. Liu, and Y. Zhu, “Pretrained vision-languageaction models are surprisingly resistant to forgetting in continual learning,” arXiv preprint arXiv:2603.03818, 2026.

[16] O. Ludwig, “The condition number as a scale-invariant proxy for information encoding in neural units,” arXiv preprint arXiv:2506.16289, 2025.

[17] B. Liu et al., “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” in Advances in Neural Information Processing Systems, 2023.

[18] H. K. Khalil, Nonlinear Systems, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, 2002.

[19] W. Lohmiller and J.-J. E. Slotine, “On contraction analysis for non-linear systems,” Automatica, vol. 34, no. 6, pp. 683–696, 1998.