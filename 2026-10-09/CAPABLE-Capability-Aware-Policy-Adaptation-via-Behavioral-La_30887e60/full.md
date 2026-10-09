# CAPABLE: Capability-Aware Policy Adaptation via Behavioral Latent Encoding

Mohammad Khoshnazar<sup>1,∗,†</sup>, Mohammad Dehghani Tezerjani<sup>2,†</sup>, Deyuan Qu<sup>3</sup>, Zhiyuan Gao<sup>1</sup>, Yanxiang Zhan<sup>1</sup>, Jeroen Schafer ¨ <sup>1</sup>, Andrew Melnik<sup>1</sup>, Qing Yang<sup>2</sup>, Michael Beetz<sup>1</sup>

Abstract— Vision-language-action (VLA) policies assume the embodiment on which they were trained and can fail when a joint fault changes how commanded actions are physically executed. Existing fault-recovery methods often require taskspecific retraining, fault labels, explicit diagnosis, or privileged embodiment information. We introduce CAPABLE, a unified capability-aware adaptation framework for frozen VLAs that integrates self-supervised capability inference with residual reinforcement learning. CAPABLE infers capability, how much of the commanded motion each joint actually realizes and how that motion contributes to end-effector behavior, online from command–response history and kinematics using a temporal encoder shared across joints, Jacobian grounding, cross-joint attention, and self-supervised physical prediction. The resulting representation conditions a residual policy that adds bounded corrections to the VLA arm action without fault labels or faultyjoint identifiers. Across 28 LIBERO tasks, CAPABLE raises success on an actuator excluded from fault training from 24.8% to 59.3%, outperforming a parameter-matched global-history baseline by 17.4 points while preserving healthy performance. Leave-one-actuator-out experiments across six joints show that this transfer is not specific to one actuator, and additional evaluations characterize transfer to unseen fault families and demonstrate recovery on a physical Franka Panda. capablevla.github.io.

## I. INTRODUCTION

A manipulator is midway through placing a plate into a drawer when one of its motors stalls. The vision-languageaction policy driving it is unaffected. It continues to issue actions that are semantically correct for the task, and the arm continues to move, but it no longer moves the way the policy assumes. The plate never arrives. Incidents of this kind are a reliability and a safety concern, and are documented in space robotics [1], [2], [3], surgical robotics [4], and industrial manipulators [5], [6]. Prevailing practice is to halt the robot once a fault is detected. Stopping is safe but costly when repair is slow or the robot is hard to reach. We are interested in the harder alternative, which is to continue the task with a body that has changed.

Vision-language-action (VLA) policies are an attractive starting point, since they supply a strong semantic and visuomotor prior for language-conditioned manipulation [7], [8], [9], [10], [11], [12], [13]. Their training data, however, assume the robot retains the healthy embodiment. A joint failure breaks that assumption at the level of physics, and no amount of visual or linguistic competence repairs it. The fault itself is not directly observable. Moreover, the fault space—spanning the affected joint, failure type, and severity—is too large to cover during training, requiring recovery to generalize to previously unseen actuator failures.

![](images/44be4badd6ac744def8224597dbaee288ce365c274a689ff6e0201c4348ac152.jpg)  
Fig. 1. The illustrated sequence (1) opens the drawer, (2) lifts the plate, and (3) places it inside. A faulty joint (red highlight) prevents the frozen VLA motion from succeeding (red cross); CAPABLE supplies a corrective motion (green check). CAPABLE infers each joint’s remaining capability from its command–response history and the live Jacobian, and adds a bounded residual correction to the frozen VLA action. No fault label or explicit jointindex embedding is supplied, allowing the shared estimator to transfer to an actuator whose failure was not observed during training.

Existing methods address only parts of this problem. Classical fault-tolerant control detects the fault and redistributes motion across the remaining joints [14], [15], [16], but does not consider the VLA’s closed-loop behavior or task progress. Learning-based methods either train a task policy from scratch [17] or require explicit fault information or supervision [18], [19]. Prior work evaluates held-out joints, but does not establish systematic transfer for a frozen multitask VLA without fault labels, affected-joint supervision, or healthy reference trajectories.

Our central observation is that a fault need not be identified by its type or cause. Instead, it can be represented by the remaining capability of each actuator, measured by how much of a commanded motion the joint realizes and how that motion affects the end effector in the current configuration.

This capability can be inferred without privileged information from each joint’s command–response history. Because capability is represented consistently across all joints, an estimator learned from failures of one joint can transfer to another.

We propose CAPABLE (Capability-Aware Policy Adaptation via Behavioral Latent Encoding), a residual reinforcement-learning controller that keeps the VLA frozen and adds a correction to its arm action [20], using an off-policy actor-critic for sample-efficient residual policy learning [21]. CAPABLE encodes each joint’s command– response history using one temporal encoder shared across all seven joints. The shared map recognizes common command– response signatures across actuators rather than learning a separate encoder per joint (Fig. 1). Each joint’s live Jacobian column grounds this estimate in the current configuration, and cross-joint attention combines all joints into a representation of what motions the arm can still produce. A selfsupervised objective that predicts realized joint and endeffector motion ties this representation to physical behavior, and the representation conditions a Soft Actor-Critic pol icy [22] through FiLM. No fault label, faulty-joint identifier, or fault-specific demonstration enters any network during training or deployment; at deployment the networks are fixed and capability is re-estimated at every control step.

Our contributions are:

• A capability formulation of actuator faults for frozen VLAs, which represents a fault by what each actuator can still realize, infers it from command–response history and kinematics without fault labels, and uses it to correct a frozen generalist through bounded residual RL.

• A per-actuator factorized architecture for held-outactuator transfer that shares one temporal encoder across joints, grounds it in the live Jacobian, composes joints with attention, and trains it with self-supervised physical prediction; removing any of these components significantly reduces held-out-joint success.

• An evaluation that separates factorization from capacity, comparing against a parameter-matched unfactorized encoder across 28 tasks, six held-out joints, five unseen fault families that mark where transfer holds and where it fails, and a physical robot.

## II. RELATED WORK

## A. Adapting Pretrained Robot Policies

Large robot datasets [23], [24] have enabled VLA policies that map visual observations, language instructions, and robot states to actions [8], [9], [10], [11], [12], [13]. Because retraining these large models is expensive, recent work adapts them after pretraining. VLA-RL fine-tunes the VLA using reinforcement learning [25], while RobustVLA trains it against observation noise and action perturbations [26]. Other methods use test-time or recovery-oriented reinforcement learning to adapt the policy to new tasks and environments [27], [28], [29]. Residual learning freezes the base policy and learns correction on the base action [20], [30]. These methods generally assume that the robot’s physical capabilities remain unchanged. CAPABLE also keeps the base VLA frozen, but conditions its corrections on an online estimate of the robot’s remaining actuator capabilities.

## B. Fault-Tolerant Control and Learned Adaptation

Classical fault-tolerant control follows two main strategies. Passive methods tolerate a predefined set of faults, while active methods diagnose the fault and reconfigure the controller [14], [15]. For redundant manipulators, motion can be replanned around a locked joint [16]. Other methods estimate the remaining effectiveness of an actuator before applying compensation [31]. Both approaches require an explicit model of the fault. Learning-based methods instead infer changes in the robot’s dynamics from recent interaction history [32], [33], [34]. Fault-tolerant locomotion policies can recover from locked or unpowered joints, but their training often uses privileged fault information through teacher policies, privileged critics, or fault-specific experts [35], [36], [37], [38].

## C. Learning-Based Fault Recovery

Pham et al. [17] formulate Franka joint failure as a POMDP and train a recurrent PPO policy for permanent and intermittent failures on a drawer task, establishing that history-based [39], [40] can compensate for an unknown impaired actuator. Their policy is trained for the task directly rather than wrapped around a pretrained multi-task VLA. DEFT [18] uses an embodiment- and task-conditioned diffusion model to generate fail-active trajectories with broad simulated failure coverage, conditioning on an explicit embodiment vector. J-PARC [19] is the closest VLA-specific study. It trains an action calibrator on locked-joint faults using healthy reference rollouts and fault-presence and affectedjoint supervision, then evaluates transfer to restricted ranges, friction, and held-out joints. CAPABLE uses task reward and self-supervised physical prediction in place of all of these signals.

## D. Weight Sharing Across Actuators

Sharing one module across the parts of a body is a wellestablished route to morphology-agnostic control. NerveNet propagates messages over the kinematic graph [41], shared modular policies apply a single policy to every limb [42], and transformer variants attend over limb tokens [43], [44]. That literature shares modules across morphologies to generalize over bodies. We share a module across actuators of one body to generalize over which actuator has failed, grounding the shared tokens in the live Jacobian so the estimate becomes a configuration-specific correction. The components themselves are standard, combining recurrent state for partial observability [45], [46], attention for composition across entities [47], and FiLM for conditioning [48].

## III. PROBLEM FORMULATION

Let the pretrained VLA $\pi _ { \phi }$ receive observation o<sub>t</sub> and language instruction ℓ and produce a seven-dimensional

action

$$
a _ { t } ^ { \mathrm { V L A } } = \pi _ { \phi } ( o _ { t } , \ell ) = [ a _ { \mathrm { a r m } , t } ^ { \mathrm { V L A } } , a _ { \mathrm { g r i p } , t } ^ { \mathrm { V L A } } ] .\tag{1}
$$

The VLA parameters ϕ remain fixed during residual training. A persistent fault f changes the nominal transition dynamics $p _ { 0 } ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$ to fault-conditioned dynamics $p _ { f } ( s _ { t + 1 } \mid$ $s _ { t } , a _ { t } )$ . The policy does not observe f and must therefore infer its physical effects from execution history.

The actor produces

$$
u _ { t } = \pi _ { \theta } ( \widetilde { o } _ { t } ^ { \mathrm { R L } } , z _ { t } ^ { \mathrm { c a p } } ) , \qquad u _ { t } \in [ - 1 , 1 ] ^ { 6 } ,\tag{2}
$$

and the applied correction is

$$
\Delta a _ { t } = \eta u _ { t } , \qquad \eta = 0 . 1 .\tag{3}
$$

The executed action is

$$
a _ { t } ^ { \mathrm { e x e c } } = \left[ \mathrm { c l i p } \left( a _ { \mathrm { a r m } , t } ^ { \mathrm { V L A } } + \Delta a _ { t } , - 1 , 1 \right) , a _ { \mathrm { g r i p } , t } ^ { \mathrm { V L A } } \right] .\tag{4}
$$

The gripper passes through untouched; the sparse reward is the LIBERO task-success predicate. An unseen actuator is a joint whose failure is excluded from all stages of training and encountered only during evaluation.

## IV. CAPABLE: CAPABILITY-AWARE POLICY ADAPTATION VIA BEHAVIORAL LATENT ENCODING

CAPABLE combines shared per-joint history encoding, current kinematic grounding, and self-supervised physical prediction to infer actuator capability and generate residual corrections. Figure 2 shows the architecture, and Algorithm 1 gives deployment.

## A. Per-Joint Command–Response History

For Panda arm joint $i \in \{ 0 , \ldots , 6 \}$ , CAPABLE constructs the token

$$
\begin{array} { r } { x _ { t } ^ { i } = [ q _ { t } ^ { i } , \dot { q } _ { t } ^ { i } , \Delta q _ { t } ^ { i } , J _ { i } ( q _ { t } ) , a _ { \mathrm { a r m } , t } ^ { \mathrm { V L A } } , a _ { \mathrm { a r m } , t } ^ { \mathrm { e x e c } } , } \\ { \Delta x _ { t } ^ { \mathrm { e e f } } , \rho _ { t } ] , } \end{array}\tag{5}
$$

where $J _ { i } ( q _ { t } ) \in \mathbb { R } ^ { 6 }$ is the ith column of the $6 \times 7$ arm Jacobian, $\Delta x _ { t } ^ { \mathrm { e e f } } ~ \in ~ \mathbb { R } ^ { 6 }$ is realized end-effector change, and $\rho _ { t }$ is the VLA action-chunk phase. Each token is 28- dimensional vector with the same feature structure across all joints. Identity enters the token geometrically, through $J _ { i } ( q _ { t } )$ which describes how motion of joint i contributes to endeffector motion in the current configuration. The model uses no one-hot joint indicator or learned joint-index embedding, and no lock target, fault-family label, or severity value is provided to the encoder, actor, or critics during training or deployment.

## B. Shared Temporal Encoder

Each joint stores $K = 1 6$ recent tokens. One encoder with identical weights for all joints produces

$$
c _ { t } ^ { i } = E _ { \psi } ( x _ { t - K : t - 1 } ^ { i } ) , \qquad c _ { t } ^ { i } \in \mathbb { R } ^ { 3 2 } .\tag{6}
$$

The history contains past information while the currentstep information enters only through the physical query. $E _ { \psi }$ contains a linear projection, LayerNorm, SiLU, and a GRU with hidden size 128. Weight sharing is what makes capability transferable. Because the encoder weights are shared, a command–response pattern learned on one joint can be recognized on any other joint, including one whose failure was never observed during training.

## C. Kinematic Query and Cross-Joint Composition

The current query for joint i is

$$
\kappa _ { t } ^ { i } = [ q _ { t } ^ { i } , \dot { q } _ { t } ^ { i } , J _ { i } ( q _ { t } ) , a _ { \mathrm { a r m } , t } ^ { \mathrm { V L A } } ] .\tag{7}
$$

For each joint $i ,$ we concatenate its history embedding $c _ { t } ^ { i }$ with its current kinematic query $\kappa _ { t } ^ { i }$ and linearly project the result to form a joint token. A two-layer, four-head Transformer encoder with feed-forward width 256 then applies selfattention across the seven joint tokens, allowing information to be exchanged among joints. No joint-ID embedding is used. Finally, we average the seven output tokens and pass the result through an MLP and LayerNorm to obtain $z _ { t } ^ { \mathrm { c a p } } \in$ $\mathbb { R } ^ { 6 4 }$ . The command–response history shows how much each joint actually moves when commanded. The Jacobian maps this joint motion to its effect on the end effector. Combining the estimates from all seven joints describes what motions the impaired arm can still produce.

## D. Self-Supervised Physical Prediction

A shared local decoder predicts realized joint displacement and a global decoder predicts realized end-effector displacement:

$$
\widehat { \Delta q } _ { t } ^ { i } = D _ { \mathrm { j o i n t } } ( c _ { t } ^ { i } , \kappa _ { t } ^ { i } , \Delta a _ { t } ) ,\tag{8}
$$

$$
\begin{array} { r } { \widehat { \Delta x } _ { t } ^ { \mathrm { e e f } } = D _ { \mathrm { e e f } } ( z _ { t } ^ { \mathrm { c a p } } , o _ { t } ^ { \mathrm { R L } } , \Delta a _ { t } ) . } \end{array}\tag{9}
$$

Here $\widehat { \Delta q } _ { t } \ = \ [ \widehat { \Delta q } _ { t } ^ { 0 } , \ldots , \widehat { \Delta q } _ { t } ^ { 6 } ] ^ { \intercal } \ \in \ \mathbb { R } ^ { 7 }$ stacks the per-joint predictions from the shared local decoder. Using Huber loss $\ell _ { H }$ , the objectives are

$$
\mathcal { L } _ { \mathrm { j o i n t } } = \frac { 1 } { 7 } \sum _ { i } \ell _ { H } ( \widehat { \Delta q } _ { t } ^ { i } , \Delta q _ { t } ^ { i } ) ,\tag{10}
$$

$$
\mathcal { L } _ { \mathrm { e e f } } = \ell _ { H } ( \widehat { \Delta x } _ { t } ^ { \mathrm { e e f } } , \Delta x _ { t } ^ { \mathrm { e e f } } ) ,\tag{11}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { k i n } } = \ell _ { H } ( J ( q _ { t } ) \widehat { \Delta q } _ { t } , \Delta x _ { t } ^ { \mathrm { e e f } } ) , } \end{array}\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { c a p } } = \mathcal { L } _ { \mathrm { j o i n t } } + \mathcal { L } _ { \mathrm { e e f } } + 0 . 2 5 \mathcal { L } _ { \mathrm { k i n } } .\tag{13}
$$

$\mathcal { L } _ { \mathrm { k i n } }$ enforces kinematic consistency by requiring the predicted joint displacements, mapped through the Jacobian, to match the observed end-effector displacement.

## E. FiLM-Conditioned SAC and Gradient Routing

CAPABLE uses twin-critic SAC [22]. The 169- dimensional residual observation is

$$
\begin{array} { r l } & { o _ { t } ^ { \mathrm { R L } } = [ p _ { t } ^ { \mathrm { e e f } } , r _ { t } ^ { \mathrm { e e f } } , q _ { t } ^ { \mathrm { g r i p } } , q _ { t } , \dot { q } _ { t } , a _ { t } ^ { \mathrm { V L A } } , \rho _ { t } , \tau _ { t } , } \\ & { ~ h _ { t - 8 : t - 1 } , \mathrm { v e c } _ { \mathrm { j o i n t } } ( J ( q _ { t } ) ) ] , } \end{array}\tag{14}
$$

where $p _ { t } ^ { \mathrm { e e f } } , r _ { t } ^ { \mathrm { e e f } } \ \in \ \mathbb { R } ^ { 3 }$ are measured end-effector position and orientation, $q _ { t } ^ { \mathrm { g r i p } } \in \mathbb { R } ^ { 2 }$ the gripper joints, $q _ { t } , \dot { q } _ { t } \in \mathbb { R } ^ { 7 }$ the measured arm state, $\rho _ { t }$ the action-chunk phase, and $\tau _ { t } ~ = ~ t / T _ { \operatorname* { m a x } }$ normalized episode time. Each executionhistory element is

$$
h _ { k } = [ a _ { \mathrm { a r m } , k } ^ { \mathrm { e x e c } } , \Delta p _ { k } ^ { \mathrm { e e f } } , \Delta r _ { k } ^ { \mathrm { e e f } } ] \in \mathbb { R } ^ { 1 2 } ,\tag{15}
$$

All input components are obtained from robot sensing, controller outputs, internal counters, or the known kinematic

![](images/248ddcdee8ab5ecd155206c2b2177ebb45705cc0eff6318e20d7d58fdbce572e.jpg)  
Fig. 2. CAPABLE architecture. Shared encoders map per-joint command–response histories and live kinematics to $\boldsymbol { z } _ { t } ^ { \mathrm { c a p } }$ , which FiLM-conditions the residual SAC actor. The bounded arm correction is added to the frozen VLA action; the gripper bypasses RL. Critics, replay and rewards are training-only.

model. The actor and critics receive the normalized observation $\widetilde { o } _ { t } ^ { \mathrm { R L } } = ( o _ { t } ^ { \mathrm { R L } } - \mu _ { o } ) / ( \sigma _ { o } + \epsilon )$ , and FiLM modulates their intermediate features $v _ { t } \colon$

$$
\gamma _ { t } = 1 + 0 . 5 \operatorname { t a n h } ( \gamma _ { \mathrm { r a w } } ( z _ { t } ^ { \mathrm { c a p } } ) ) ,\tag{16}
$$

$$
\beta _ { t } = 0 . 5 \operatorname { t a n h } ( \beta _ { \mathrm { r a w } } ( z _ { t } ^ { \mathrm { c a p } } ) ) ,\tag{17}
$$

$$
v _ { t } ^ { \prime } = \gamma _ { t } \odot v _ { t } + \beta _ { t } .\tag{18}
$$

The actor’s deterministic output is initialized to zero, so training begins from exact reproduction of the base VLA arm action.

The capability encoder is optimized by the auxiliary physical-prediction loss and a critic loss weighted by 0.05:

$$
\nabla _ { \psi } \mathcal { L } = \nabla _ { \psi } \mathcal { L } _ { \mathrm { c a p } } + 0 . 0 5 \nabla _ { \psi } \mathcal { L } _ { Q } .\tag{19}
$$

Actor-loss gradients are blocked at $\boldsymbol { z } _ { t } ^ { \mathrm { c a p } }$ and therefore do not update $E _ { \psi }$ , keeping encoder learning dominated by the physical-prediction objective.

## V. EXPERIMENTAL SETUP

## A. Benchmark, Simulator, and Base Policy

We use LIBERO [49] with the Franka Panda implementation in robosuite [50] and MuJoCo [51]. The base checkpoint is moojink/openvla-7b-oftfinetuned-libero-spatial-object-goal-10, loaded once and shared across all tasks, with suite-specific action statistics used for unnormalization. OpenVLA-OFT executes action chunks of length eight. The residual policy changes only the six continuous arm dimensions.

A joint lock is a MuJoCo equality constraint on the physical Panda joint rather than an action mask, so the arm remains dynamically coupled to the locked link. Condition changes rebuild the relevant environment path, and lock drift is monitored. The 28-task and cross-fault studies are conducted in simulation; Sec. VI-E evaluates the trained residual policy on a physical Franka Panda under softwareenforced joint locks.

Algorithm 1 CAPABLE residual control (deployment)   
Require: frozen VLA $\pi _ { \phi } ,$ encoder $E _ { \psi } ,$ , actor $\pi _ { \theta } .$ chunk   
length $L = 8 ,$ scale $\eta = 0 . 1$   
1: initialize per-joint buffers $\mathcal { X } ^ { i }  \emptyset , i = 0 . . 6$   
2: for control step $t = 0 , 1 , \ldots$ . do   
3: if t mod $L = 0$ then   
4: $a _ { t : t + L - 1 } ^ { \mathrm { V L A } }  \pi _ { \phi } ( o _ { t } , \ell )$ {new chunk}   
5: end if   
6: $\rho _ { t }  ( t \mathrm { m o d } L ) / L$   
7: $c _ { t } ^ { i } \gets E _ { \psi } ( \mathcal { X } ^ { i } [ t - K : t - 1 ] ) \ \forall i$ {shared weights}   
8: $\kappa _ { t } ^ { i } \gets [ q _ { t } ^ { i } , \dot { q } _ { t } ^ { i } , J _ { i } ( q _ { t } ) , a _ { \mathrm { a r m } , t } ^ { \mathrm { V L A } } ] ^ { \mathrm { ~ } } \vee$ ∀i   
9: $z _ { t } ^ { \mathrm { c a p } } \gets \mathrm { A t t n } ( \{ c _ { t } ^ { i } , \kappa _ { t } ^ { i } \} _ { i = 0 } ^ { 6 } )$   
10: $u _ { t } \gets \pi _ { \theta } ( \widetilde { o } _ { t } ^ { \mathrm { R L } } , z _ { t } ^ { \mathrm { c a p } } ) ; \quad \Delta a _ { t } \gets \eta u _ { t }$   
11: execute $a _ { t } ^ { \mathrm { e x e c } }$ by Eq. (4)   
12: observe $\Delta q _ { t } , \Delta x _ { t } ^ { \mathrm { e e f } } ;$ ; append $\ v { x } _ { t } ^ { i }$ to $\mathcal { X } ^ { i }$   
13: end for

## B. Held-Out-Blind 28-Task Pool

We screen all 40 LIBERO tasks with the base VLA, using only healthy outcomes and candidate training conditions $\mathcal { I } _ { \mathrm { s e e n } } ~ = ~ \{ j _ { 0 } , j _ { 4 } , j _ { 5 } , j _ { 6 } \}$ . A task is retained when healthy success is at least 80% and at least one seen fault is usable, meaning that the corresponding fault condition has nonzero success and sufficient headroom for improvement. Twelve tasks fail this rule, leaving 28 tasks and 46 usable task– fault cells, comprising ten from LIBERO-10, seven from LIBERO-Spatial, six from LIBERO-Object, and five from LIBERO-Goal. The selected task indices are supplementary.

Failures on $j _ { 2 }$ are excluded from all training stages. Its screening results are used only for reporting, so all 16 tasks with 0% base-VLA success under $j _ { 2 }$ remain valid evaluation cases.

## C. Task Curriculum and Replay Distribution

Tasks are sampled uniformly, with $p ( m ) = 1 / 2 8$ . Within each task, healthy execution is sampled with probability 0.10, while usable seen faults share the remaining 0.90 in proportion to their screened headroom $s _ { m } ^ { \mathrm { H } } - s _ { m , j }$ , where $s _ { m } ^ { \mathrm { H } }$ is healthy base-VLA success for task m and $s _ { m , j }$ is success under locked joint j.

Replay uses the same task–condition distribution. Each minibatch weights tasks equally and samples conditions according to the rule above. The pool contains 28 healthy task– condition pairs among 74 total pairs, so uniform sampling over all pairs would allocate $2 8 / 7 4 \approx 3 7 . 8 \%$ of updates to healthy data instead of the target 10%. Section VI-D evaluates this alternative.

## D. Training and Deterministic Evaluation

For each task, a held-out set of initial states is reserved before any learning and is disjoint from the states available to training. Every method is trained for 1.4M environment steps per seed, and evaluated at the 1.4M-step checkpoint. We run three independently initialized seeds sharing the task pool, state partition, data budget, base VLA, and evaluation procedure; reported ± values are mean ± standard deviation across these seeds.

Evaluation compares zero correction with each learned policy on healthy execution, every usable seen fault, and globally unseen $j _ { 2 } .$ . Policy actions are deterministic, and within a seed each task–condition cell uses the same 300 held-out episodes for all methods. We additionally perform leave-one-actuator-out evaluation for six joints $( j _ { 0 } \mathrm { - } j _ { 5 } )$ on eight tasks, with two tasks selected from each LIBERO suite. In each split, the held-out joint is excluded from fault training and evaluated zero-shot.

We report task-balanced success, averaging seen-fault success within task before averaging across tasks, so tasks with more usable faults carry no extra weight.

All architecture, optimization, replay, and fault-severity hyperparameters were fixed using training data and validation or pilot episodes involving only seen joints. Held-out $j _ { 2 }$ outcomes were never used for task selection, checkpoint selection, hyperparameter tuning, or other design decisions.

## E. Cross-Fault Generalization Protocol

The headline CAPABLE and global-history policies are trained only on persistent locks affecting $j _ { 0 } , j _ { 4 } , j _ { 5 } ,$ and $j _ { 6 }$ . Without retraining, we evaluate the same three seed checkpoints on globally unseen $j _ { 2 }$ under six fault families, namely a persistent lock, increased viscous damping, increased Coulomb friction, partial actuator effectiveness, restricted joint range, and a late-onset lock.

The five unseen fault families are evaluated at three levels: viscous joint damping $d \in \{ 5 , 2 0 , 5 0 \} \mathrm { { N m s r a d } ^ { - 1 } }$ , Coulomb friction $f \in \{ 1 , 3 , 5 \}$ N m, retained actuator authority and joint-range fractions in {0.75, 0.50, 0.25}, and lock onset at control step $t _ { \mathrm { o n } } \in \{ 3 2 , 6 4 , 9 6 \}$ . Severity levels are fixed from seen-joint pilots without using $j _ { 2 }$ outcomes. All methods use the same 300 held-out episodes per task–condition cell. For multi-severity faults, success is averaged over severities within each task and seed, then equally across the 28 tasks.

## F. Baselines

The central baseline is Global-history SAC, which isolates the factorization prior. It shares the VLA, residual action space, 16-step window, 64-D latent, FiLM actor and twin critics, auxiliary targets, optimizer, replay distribution, budget, and evaluation states with CAPABLE, but concatenates the seven 28-D joint tokens at each step and encodes them with one global recurrent model, with no shared per-joint encoder and no cross-joint token structure. Hidden width is adjusted so trainable parameter count is within 1% of CAPA-BLE. Both models use SAC hidden width 256, encoder and SAC learning rates $1 0 ^ { - 4 }$ and $3 \times 1 0 ^ { - 4 }$ , batch size 256, UTD 4, and n-step 3. Classical RR is a non-learning redundancyresolution baseline. The same unnormalized six-dimensional VLA arm command is used as the desired Cartesian motion. Joint availability is estimated online from recent command– response mismatch, and a weighted damped pseudoinverse of the live Jacobian redistributes the command toward responsive joints. It receives no fault label or affected-joint identifier, and the gripper passes through unchanged.

POMDP-RL [17] and DEFT [18] are trained in their own formulations on the same 28-task pool, fault conditions, training budget, and held-out evaluation states. POMDP-RL learns a recurrent task policy under partial observability. DEFT generates joint-space trajectories between a start and a goal configuration, so it receives goal configurations derived from simulator object and target-region state through inverse kinematics, a scripted gripper at phase boundaries, and the current actuator-availability vector. It is therefore privileged in two respects, on task state and on fault identity, whereas CAPABLE uses only non-privileged runtime observations, controller signals, and known kinematics; its result measures the value of that privileged information rather than providing a matched comparison. The matched-information comparison for the factorization claim is CAPABLE versus Globalhistory SAC. Table I summarizes each method’s assumptions. We excluded J-PARC [19] as a similar baseline due to the absence of peer-reviewed validation and publicly released implementation.

## VI. RESULTS

We organize the evaluation around the three claims of Sec. I. H1: does joint factorization transfer capability to an actuator that was never faulty during training, beyond what an unfactorized encoder of equal capacity achieves? H2: does capability learned from persistent locks transfer to fault families never seen in training? H3: does held-outactuator transfer generalize systematically across actuator identity, and does it survive contact with hardware?

TABLE I  
COMPARISON ASSUMPTIONS. LEARNED METHODS USE THE SAME TASK POOL, FAULT CONDITIONS, BUDGET, AND HELD-OUT STATES. CLASSICAL RR IS NON-LEARNING; DEFT RECEIVES SIMULATOR-DERIVED GOALS AND ACTUATOR AVAILABILITY.
<table><tr><td>Method</td><td>Base VLA</td><td>Residual</td><td>Oracle fault</td><td>Oracle task state</td><td>Joint-fact.</td></tr><tr><td>Base VLA (no correction)</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>POMDP-RL</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Classical RR</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>DEFT</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>Global-history SAC</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>CAPABLE (ours)</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td></tr></table>

## A. H1: Factorization Transfers to a Held-Out Actuator

Table II is the primary result. Across 28 tasks, CAPABLE raises unseen-j success from 24.8% to 59.3%, versus 41.9% for Global-history SAC and 32.9% for classical redundancy resolution. The paired CAPABLE–global gap $\mathrm { i s + 1 7 . 4 }$ points under matched capacity, data, optimization, and evaluation. CAPABLE also exceeds classical redundancy resolution by 26.4 points on unseen $j _ { 2 }$ . Healthy success remains 90.8% versus 91.4% for zero correction. DEFT reaches 62.8% with oracle fault and task-state information, only 3.5 points above CAPABLE.

TABLE II  
MAIN 28-TASK SUCCESS (%). H/S/U: HEALTHY/SEEN/UNSEEN-j<sub>2</sub>; ∆ IS VERSUS BASE VLA ON UNSEEN $j _ { 2 } .$ . CLASSICAL RR IS NON-LEARNING; DEFT USES ORACLE FAULT AND TASK STATE.
<table><tr><td>Method</td><td>H</td><td>S</td><td>U</td><td>∆ vs. Base</td></tr><tr><td>Base VLA</td><td> ${ \bf 9 1 . 4 \pm 0 . 0 }$ </td><td> $4 1 . 6 { \pm } 0 . 0 \ $ </td><td> $2 4 . 8 { \pm } 0 . 0 $ </td><td></td></tr><tr><td>POMDP-RL</td><td> $1 9 . 6 \pm 2 . 4$ </td><td> $1 1 . 3 { \pm } 2 . 1 $ </td><td> $1 0 . 1 \pm 1 . 9$ </td><td> $- 1 4 . 7$ </td></tr><tr><td>Classical RR</td><td> $8 1 . 6 { \pm } 0 . 0 $ </td><td> $6 1 . 3 { \pm } 0 . 0 \ $ </td><td> $3 2 . 9 \pm 0 . 0$ </td><td> $+ 8 . 1$ </td></tr><tr><td>DEFT</td><td> $8 9 . 7 \pm 1 . 4$ </td><td> $7 0 . 1 \pm 2 . 3$ </td><td> ${ \bf 6 2 . 8 \pm 2 . 9 }$ </td><td> ${ \bf + 3 8 . 0 }$ </td></tr><tr><td>Global-history SAC</td><td> $8 8 . 1 \pm 1 . 3$ </td><td> $7 3 . 8 \pm 2 . 1$ </td><td> $4 1 . 9 { \pm } 3 . 2 $ </td><td> $+ 1 7 . 1$ </td></tr><tr><td>CAPABLE</td><td> $9 0 . 8 \pm 1 . 0$ </td><td> ${ \bf 8 6 . 7 \pm 1 . 8 }$ </td><td> $5 9 . 3 { \pm } 3 . 0 \ $ </td><td> $+ 3 4 . 5$ </td></tr></table>

Table III shows the advantage is not suite-specific: CA-PABLE leads Global-history SAC on all four suites and 24 of 28 task cells, with margins from +12.8 to +24.3 points. Zero-baseline cells are retained in all averages; full task-level results are supplementary.

TABLE III  
UNSEEN-j<sub>2</sub> SUCCESS BY SUITE (%).
<table><tr><td>Suite</td><td>Base VLA</td><td>Global-history SAC</td><td>CAPABLE</td><td>∆ vs. Global</td></tr><tr><td>LIBERO-10</td><td>8.0±0.0</td><td>26.5±4.6</td><td> ${ \bf 4 5 . 2 \pm 4 . 1 }$ </td><td>+18.7</td></tr><tr><td>LIBERO-Goal</td><td>40.0±0.0</td><td>49.0±3.8</td><td> ${ \bf 7 3 . 3 \pm 4 . 7 }$ </td><td>+24.3</td></tr><tr><td>LIBERO-Object</td><td>20.8±0.0</td><td>52.8±3.3</td><td> ${ \bf 6 5 . 6 \pm 4 . 4 }$ </td><td>+12.8</td></tr><tr><td>LIBERO-Spatial</td><td>41.4±0.0</td><td>49.5±1.2</td><td> ${ \bf 6 4 . 0 \pm 3 . 8 }$ </td><td>+14.5</td></tr><tr><td>Task-balanced mean</td><td>24.8±0.0</td><td>41.9±3.2</td><td> ${ \bf 5 9 . 3 \pm 3 . 0 }$ </td><td> $+ 1 7 . 4$ </td></tr></table>

## B. H2: Transfer Extends to Unseen Fault Families

Table IV tests zero-shot transfer on unseen $j _ { 2 }$ across the trained persistent lock and five unseen fault families. CAPA-BLE leads Global-history SAC on persistent lock, damping, friction, and late-onset lock, but not on partial effectiveness or range restriction. The successful cases share the trained signature of suppressed motion relative to command, whereas the latter two are graded or state-dependent.

## C. H3: Transfer Generalizes Across Held-Out Actuators

Table V tests actuator-independent transfer across six heldout joints on the same eight LIBERO tasks. The $j _ { 2 }$ entry uses this common eight-task subset, whereas Table II reports the primary 28-task $j _ { 2 }$ result. CAPABLE outperforms Globalhistory SAC on all six held-out actuators, showing that its transfer advantage is not specific to $j _ { 2 }$

## D. Mechanism Ablations

Table VI isolates which components produce the transfer. Removing history leaves a memoryless residual with current Jacobian information at 39.4% on unseen $j _ { 2 } ,$ , below Global-history SAC, so static kinematic redirection does not explain the result. Unsharing the temporal encoder costs 13.5 points on unseen $j _ { 2 }$ and 6.6 on seen faults; removing the Jacobian query costs 11.5 points, cross-joint attention 13.2, and the self-supervised capability loss 15.6, one of the largest drops. Removing $z ^ { \mathrm { c a p } }$ conditioning costs 9.7. Together these ablations support a coupled temporal–kinematic representation shaped by physical prediction rather than any single component. The unshared variant also performs worse with 50% more parameters. Equal-strata replay raises healthy success to 92.3% while reducing fault recovery, supporting the default headroom-weighted replay distribution.

## E. Hardware Validation

We evaluate CAPABLE and parameter-matched Globalhistory SAC on a physical Franka Panda under three task– joint conditions, opening a drawer with seen $j _ { 0 } ,$ , putting the object in the drawer with unseen $j _ { 2 } ,$ and object rearrangement with seen $j _ { 6 } .$ Each trial carries one persistent lock, software-enforced at the joint-reference layer, so that once end-effector commands are converted to joint references, the selected joint is clamped to its fault-onset angle while all others remain controllable. Commanding the held angle rather than mechanically opposing motion avoids triggering the Franka reflex path.

For hardware transfer, OpenVLA-OFT is initialized from the Sec. V-A checkpoint, fine-tuned in a digital twin matched to the physical workspace, and then frozen. CAPABLE and Global-history SAC are subsequently trained on top of this frozen VLA in the same simulated setup and transferred to the physical Panda without hardware training. The digital twin matches the robot/table geometry, camera pose, objects, controller, and high-level control rate, with randomized object poses and physical parameters. Each method receives 10 trials per condition.

CAPABLE reaches 85.0% across the two seen-joint conditions against 45.0% for Global-history SAC, and 70.0% against 40.0% on the unseen- $\cdot j _ { 2 }$ condition, for an overall 80.0% versus 43.3% across 30 trials per method (Table VII). Individual conditions are not significant at this sample size, whereas the pooled difference is significant $( p = 0 . 0 0 7$ , twosided Fisher exact test).

TABLE IV  
ZERO-SHOT CROSS-FAULT SUCCESS ON UNSEEN j<sub>2</sub> (%). TRAINING USES ONLY PERSISTENT LOCKS; NON-LOCK COLUMNS AVERAGE THREE SEVERITIES.
<table><tr><td>Method</td><td>Persistent lock</td><td>Damping</td><td>Friction</td><td>Partial effectiveness</td><td>Range restriction</td><td>Late-onset lock</td></tr><tr><td>Base VLA</td><td> $2 4 . 8 \pm 0 . 0$ </td><td> $4 5 . 8 \pm 0 . 0$ </td><td> $4 0 . 6 { \pm } 0 . 0 $ </td><td> $4 4 . 2 \pm 0 . 0$ </td><td> $3 2 . 7 \pm 0 . 0$ </td><td> $5 0 . 4 \pm 0 . 0$ </td></tr><tr><td>Global-history SAC</td><td> $4 1 . 9 \pm 3 . 2$ </td><td> $5 5 . 6 \pm 1 . 4$ </td><td> $5 1 . 8 { \pm } 1 . 6 $ </td><td> ${ \bf 6 4 . 1 \pm 3 . 3 }$ </td><td> ${ \bf 5 4 . 7 \pm 2 . 8 }$ </td><td> $6 1 . 5 { \pm } 3 . 1 \ $ </td></tr><tr><td>CAPABLE</td><td> ${ \bf 5 9 . 3 \pm 3 . 0 }$ </td><td> ${ \bf 6 7 . 8 \pm 2 . 9 }$ </td><td> ${ \bf 6 4 . 2 \pm 3 . 2 }$ </td><td> $6 2 . 3 { \pm } 3 . 0 \ $ </td><td> $4 8 . 2 \pm 3 . 5$ </td><td> ${ \bf 7 2 . 4 \pm 2 . 7 }$ </td></tr><tr><td>∆ vs. Global</td><td> $+ 1 7 . 4$ </td><td>+12.2</td><td> $+ 1 2 . 4$ </td><td>-1.8</td><td> $^ { - 6 . 5 }$ </td><td> $+ 1 0 . 9$ </td></tr></table>

TABLE V

LEAVE-ONE-ACTUATOR-OUT TRANSFER ACROSS SIX HELD-OUT JOINTS ON EIGHT LIBERO TASKS. EACH ROW USES A SEPARATE LEAVE-ONE-ACTUATOR-OUT TRAINING SPLIT IN WHICH THE INDICATED JOINT IS EXCLUDED FROM FAULT TRAINING AND EVALUATED ZERO-SHOT. VALUES ARE SUCCESS RATES (%).
<table><tr><td>Held-out joint</td><td>t Global-history SAC</td><td>CAPABLE</td><td>∆ vs. Global</td></tr><tr><td>jo</td><td> $5 5 . 0 { \pm } 4 . 4 $ </td><td> ${ \bf 7 6 . 7 \pm 3 . 3 }$ </td><td>+21.7</td></tr><tr><td>j1</td><td> $3 9 . 4 \pm 1 . 3$ </td><td> ${ \bf 5 6 . 2 \pm 2 . 3 }$ </td><td>+16.8</td></tr><tr><td>j2</td><td> $4 6 . 6 \pm 5 . 2$ </td><td> ${ \bf 6 9 . 3 \pm 3 . 0 }$ </td><td> $+ 2 2 . 7$ </td></tr><tr><td>j3</td><td> $3 9 . 6 \pm 4 . 6$ </td><td> ${ \bf 6 3 . 6 \pm 4 . 7 }$ </td><td> $+ 2 4 . 0$ </td></tr><tr><td>j4</td><td> $4 4 . 7 { \pm } 5 . 1 $ </td><td> ${ \bf 8 8 . 1 \pm 1 . 9 }$ </td><td> $+ 4 3 . 4$ </td></tr><tr><td>j5</td><td> $2 9 . 3 { \pm } 4 . 2 $ </td><td> ${ \bf 4 8 . 2 \pm 3 . 6 }$ </td><td>+18.9</td></tr><tr><td>Mean</td><td>42.4</td><td>67.0</td><td>+24.6</td></tr></table>

TABLE VI  
28-TASK ABLATIONS. P : PARAMETERS (M); H/S/U:

HEALTHY/SEEN/UNSEEN-j ; $\Delta$ IS VERSUS FULL CAPABLE ON UNSEEN j<sub>2</sub>.
<table><tr><td>Variant</td><td>P</td><td>H</td><td>S</td><td>U</td><td>∆ vs. Full</td></tr><tr><td>Full CAPABLE</td><td>0.82</td><td> $9 0 . 8 \pm 1 . 0$ </td><td> ${ \bf 8 6 . 7 \pm 1 . 8 }$ </td><td> ${ \bf 5 9 . 3 \pm 3 . 0 }$ </td><td>一</td></tr><tr><td>Global-history SAC</td><td>0.82</td><td> $8 8 . 1 \pm 1 . 3$ </td><td> $7 3 . 8 \pm 2 . 1$ </td><td> $4 1 . 9 \pm 3 . 2$ </td><td> $- 1 7 . 4$ </td></tr><tr><td>Equal-strata replay</td><td>0.82</td><td> ${ \bf 9 2 . 3 \pm 0 . 9 }$ </td><td> $7 1 . 5 { \pm 2 . 6 }$ </td><td>46.4±3.7</td><td>-12.9</td></tr><tr><td>No Jacobian query</td><td>0.81</td><td> $8 9 . 8 \pm 1 . 4$ </td><td> $7 5 . 9 \pm 2 . 4$ </td><td>47.8±3.8</td><td>-11.5</td></tr><tr><td>No cross-joint attention</td><td>0.71</td><td> $8 9 . 2 \pm 1 . 5$ </td><td> $8 1 . 2 \pm 2 . 7$ </td><td>46.1±3.6</td><td>-13.2</td></tr><tr><td>No capability loss</td><td>0.80</td><td> $8 8 . 7 \pm 1 . 7 $ </td><td> $7 2 . 0 { \pm } 3 . 1 $ </td><td>43.7±4.6</td><td>-15.6</td></tr><tr><td>Concatenation, not FiLM</td><td>0.84</td><td> $9 0 . 1 \pm 1 . 3 $ </td><td> $7 8 . 4 \pm 2 . 3$ </td><td>52.6±3.4</td><td>-6.7</td></tr><tr><td> $\mathbf { N o _ { \alpha } } z ^ { \mathbf { c a p } }$  conditioning</td><td>0.76</td><td> $8 9 . 7 \pm 1 . 6 $ </td><td> $7 6 . 6 \pm 2 . 7$ </td><td> $4 9 . 6 \pm 3 . 9$ </td><td>-9.7</td></tr><tr><td>Unshared per-joint encoders</td><td>1.23</td><td> $9 0 . 9 \pm 1 . 1$ </td><td> $8 0 . 1 \pm 2 . 3$ </td><td> $4 5 . 8 \pm 4 . 2$ </td><td>-13.5</td></tr><tr><td>No-history residual</td><td>0.60</td><td> $9 0 . 6 \pm 1 . 4$ </td><td> $6 8 . 4 \pm 3 . 1 \hphantom { 0 0 0 }$ </td><td>39.4±4.4</td><td>-19.9</td></tr></table>

A global recurrent encoder can represent fault history, but nothing requires ${ } ^ { 6 6 } j _ { 0 }$ locked” and ${ } ^ { 6 6 } j _ { 6 }$ locked” to occupy related latent regions, and nothing constrains what it does on a joint it has never seen fail. CAPABLE constrains that freedom by construction: one temporal function processes every actuator, so the same learned map interprets a stalled command–response pattern regardless of joint index. The

## VII. DISCUSSION

TABLE VII  
REAL-ROBOT SUCCESS UNDER PERSISTENT LOCKS, 10 TRIALS PER CONDITION. LOCKS ARE SOFTWARE-ENFORCED AT THE JOINT-REFERENCE LAYER.
<table><tr><td>Task</td><td>Locked joint</td><td>Global-history SAC</td><td>CAPABLE</td></tr><tr><td>Open drawer</td><td>jo (seen)</td><td>5/10 (50%)</td><td>9/10 (90%)</td></tr><tr><td>Put the object in the drawer</td><td>j2 (unseen)</td><td>4/10 (40%)</td><td>7/10 (70%)</td></tr><tr><td>Object rearrangement</td><td>j6 (seen)</td><td>4/10 (40%)</td><td>8/10 (80%)</td></tr><tr><td>Overall</td><td>1</td><td>13/30 (43.3%)</td><td>24/30 (80.0%)</td></tr></table>

Jacobian then grounds that shared estimate in configurationspecific geometry, allowing a transferable representation to drive a geometrically specific correction.

The ablations support this reading over a purely geometric or purely temporal alternative. Kinematic information without history reaches 39.4%, below the unfactorized baseline, while using separate per-joint history encoders reaches 45.8%. Full CAPABLE, which combines shared temporal encoding with live kinematic grounding, reaches 59.3%. The mechanism is the conjunction, and the strongest evidence for it is not the improvement over the frozen VLA but the reproducible advantage over a parameter-matched global encoder on identical data, across independently held-out actuators in the leave-one-actuator-out study.

The cross-fault results delimit where capability of this kind is portable. Damping, friction, and late-onset locks resemble the trained persistent locks in how they suppress motion relative to command. Partial effectiveness is graded rather than near-binary, and range restriction is state-dependent, so both differ from the training signatures in ways the current formulation does not capture; extending capability inference to graded and state-dependent impairments is the natural next step.

## VIII. LIMITATIONS

The broad study remains simulation-heavy, and the hardware evaluation covers only three task–joint conditions with software-enforced locks. The leave-one-actuator-out study uses eight rather than all 28 tasks. Training uses persistent locks only, and CAPABLE does not outperform globalhistory SAC under partial effectiveness or range restriction.

## IX. CONCLUSION

CAPABLE represents actuator faults through remaining physical capability rather than explicit fault identity. A shared temporal encoder, live kinematics, and bounded residual control enable transfer to actuators excluded from fault training while preserving healthy performance. Across 28 LIBERO tasks, all six leave-one-actuator-out splits, three of five unseen fault families, and hardware, CAPABLE outperforms a parameter-matched global-history baseline. These results show that factorized command–response representations provide a practical route to fault recovery for frozen VLA policies without fault labels or affected-joint supervision.

## X. ACKNOWLEDGMENT

AI tools (ChatGPT) was used to generate the line-art robotic arm schematic and to adjust image color balance in Figure 1.

## REFERENCES

[1] NASA, “STS-111 mission: Canadarm2 wrist roll joint replacement,” NASA Mission Archive, 2002.

[2] ——, “Space station canadarm2 joint replacement, U.S. spacewalk 95,” NASA ISS Blog, 2026.

[3] J. Lin, K. Klein, W. Green, and R. Kinnett, “Diagnosis and recovery of the curiosity rover drill feed mechanism stall anomaly,” NASA Jet Propulsion Laboratory, Tech. Rep. NTRS 20230005798, 2023.

[4] D. C. W. Friedman, T. S. Lendvay, and B. Hannaford, “Instrument failures for the da Vinci surgical system: A Food and Drug Administration MAUDE database study,” Surgical Endoscopy, vol. 27, no. 5, pp. 1503–1508, 2013.

[5] Q. Zheng and Y. Yang, “Failure analysis of a harmonic gear drive flexspline in a robot joint,” in IOP Conference Series: Materials Science and Engineering, vol. 452, 2018, p. 042148.

[6] A. Raviola, A. De Martin, R. Guida, M. Sorli, and G. Jacazio, “Fault detection of harmonic drives in collaborative robots: The case of the Universal Robots UR5,” in Proc. 6th European Conference ofthe PHM Society, 2021.

[7] D. Driess et al., “PaLM-E: An embodied multimodal language model,” in Proc. International Conference on Machine Learning (ICML), ser. PMLR, vol. 202, 2023, pp. 8469–8488.

[8] A. Brohan et al., “RT-1: Robotics Transformer for real-world control at scale,” arXiv preprint arXiv:2212.06817, 2022.

[9] B. Zitkovich et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in Proc. 7th Conference on Robot Learning (CoRL), ser. PMLR, vol. 229, 2023, pp. 2165–2183.

[10] D. Ghosh et al., “Octo: An open-source generalist robot policy,” in Proc. Robotics: Science and Systems (RSS), 2024.

[11] K. Black et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[12] M. J. Kim et al., “OpenVLA: An open-source vision-language-action model,” in Proc. 8th Conference on Robot Learning (CoRL), ser. PMLR, vol. 270, 2025, pp. 2679–2713.

[13] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-languageaction models: Optimizing speed and success,” arXiv preprint arXiv:2502.19645, 2025.

[14] Y. Zhang and J. Jiang, “Bibliographical review on reconfigurable faulttolerant control systems,” Annual Reviews in Control, vol. 32, no. 2, pp. 229–252, 2008.

[15] M. Blanke, M. Kinnaert, J. Lunze, and M. Staroswiecki, Diagnosis and Fault-Tolerant Control, 2nd ed. Berlin, Germany: Springer, 2006.

[16] B. Xie and A. A. Maciejewski, “Maximizing the probability of task completion for redundant robots experiencing locked joint failures,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 616–625, 2021.

[17] T.-H. Pham, G. Aikins, T. Truong, and K.-D. Nguyen, “Adaptive compensation for robotic joint failures using partially observable reinforcement learning,” Algorithms, vol. 17, no. 10, p. 436, 2024.

[18] G. G. Briscoe-Martinez, Y. Gautam, R. Shetty, A. Pasricha, M. M. Nicotra, and A. Roncone, “Moving on, even when you’re broken: Fail-active trajectory generation via diffusion policies conditioned on embodiment and task,” arXiv preprint arXiv:2602.02895, 2026.

[19] M. Jo, T. Kwon, J. Chun, Y. Jeong, and T. Kim, “Uncovering vulnerability of vision-language-action models under joint-level physical faults,” arXiv preprint arXiv:2606.10501, 2026.

[20] T. Johannink et al., “Residual reinforcement learning for robot control,” in Proc. IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 6023–6029.

[21] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, “Efficient online reinforcement learning with offline data,” in Proc. International Conference on Machine Learning (ICML), ser. PMLR, vol. 202, 2023, pp. 1577–1594.

[22] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in Proc. International Conference on Machine Learning (ICML), 2018, pp. 1861–1870.

[23] H. R. Walke et al., “BridgeData V2: A dataset for robot learning at scale,” in Proc. 7th Conference on Robot Learning (CoRL), ser. PMLR, vol. 229, 2023, pp. 1723–1736.

[24] A. Khazatsky et al., “DROID: A large-scale in-the-wild robot manipulation dataset,” in Proc. Robotics: Science and Systems (RSS), 2024.

[25] G. Lu et al., “VLA-RL: Towards masterful and general robotic manipulation with scalable reinforcement learning,” arXiv preprint arXiv:2505.18719, 2025.

[26] H. Zhang, S. Zhang, J. Jin, Q. Zeng, R. Li, and D. Wang, “RobustVLA: Robustness-aware reinforcement post-training for visionlanguage-action models,” arXiv preprint arXiv:2511.01331, 2025.

[27] C. Liu et al., “On-the-fly VLA adaptation via test-time reinforcement learning,” in Proc. 64th Annual Meeting of the Association for Computational Linguistics (ACL), 2026, pp. 40 107–40 125.

[28] W. Liufu et al., “RePO-VLA: Recovery-driven policy optimization for vision-language-action models,” arXiv preprint arXiv:2605.09410, 2026.

[29] X. Zhang, Y. Weng, Q. Liu, Y. Mu, and Y. Li, “Learning robust execution in robotic manipulation with agentic reinforcement learning,” arXiv preprint arXiv:2607.13818, 2026.

[30] T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, “Residual policy learning,” arXiv preprint arXiv:1812.06298, 2018.

[31] J. Huang, T. Cao, Z. Zhang, and S.-H. Yang, “Learning estimatorbased fault-tolerant control for robot manipulators with partial loss of actuator effectiveness,” Asian Journal of Control, vol. 27, pp. 2425– 2435, 2025.

[32] W. Yu, J. Tan, C. K. Liu, and G. Turk, “Preparing for the unknown— learning a universal policy with online system identification,” in Proc. Robotics Science and Systems, 2017.

[33] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “RMA—rapid motor adaptation for legged robots,” in Proc. Robotics Science and Systems, 2021.

[34] A. Nagabandi, I. Clavera, S. Liu, R. S. Fearing, P. Abbeel, S. Levine, and C. Finn, “Learning to adapt in dynamic, real-world environments through meta-reinforcement learning,” in Proc. International Conference on Learning Representations, 2019.

[35] D. Liu, T. Zhang, J. Yin, and S. See, “Saving the limping—faulttolerant quadruped locomotion via reinforcement learning,” arXiv preprint, 2022.

[36] T. Xu, Y. Cheng, P. Shen, and L. Zhao, “AcL—action learner for fault-tolerant quadruped locomotion control,” arXiv preprint, 2025.

[37] G. Gravina, L. Rossini, C. Rizzardo, A. Laurenzi, and N. Tsagarakis, “Learning fault-tolerant locomotion with adaptive gait timing,” arXiv preprint, 2026.

[38] G. Turrisi, O. Pali, L. Oneto, and C. Semini, “Mixture-of-experts rl for fault-tolerant legged locomotion,” arXiv preprint, 2026.

[39] M. D. Tezerjani, M. Khoshnazar, M. Tangestanizadeh, A. Kiani, and Q. Yang, “A survey on reinforcement learning applications in SLAM,” Journal of Machine Learning and Deep Learning, vol. 1, no. 1, pp. 20–31, 2024.

[40] M. Khoshnazar, A. Melnik, and M. Beetz, “LLM-guided future hypotheses for horizon-aware exploration in multi-step robot manipulation,” arXiv preprint arXiv:2605.29864, 2026.

[41] T. Wang, R. Liao, J. Ba, and S. Fidler, “NerveNet: Learning structured policy with graph neural networks,” in Proc. International Conference on Learning Representations (ICLR), 2018.

[42] W. Huang, I. Mordatch, and D. Pathak, “One policy to control them all: Shared modular policies for agent-agnostic control,” in Proc. International Conference on Machine Learning (ICML), ser. PMLR, vol. 119, 2020, pp. 4455–4464.

[43] V. Kurin, M. Igl, T. Rocktaschel, W. Boehmer, and S. Whiteson, “My¨ body is a cage: The role of morphology in graph-based incompatible control,” in Proc. International Conference on Learning Representations (ICLR), 2021.

[44] A. Gupta, L. Fan, S. Ganguli, and L. Fei-Fei, “MetaMorph: Learning universal controllers with transformers,” in Proc. International Conference on Learning Representations (ICLR), 2022.

[45] M. Hausknecht and P. Stone, “Deep recurrent Q-learning for partially observable MDPs,” in AAAI Fall Symposium on Sequential Decision Making for Intelligent Agents, 2015.

[46] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares,¨ H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proc. Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1724–1734.

[47] A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[48] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conference on Artificial Intelligence, vol. 32, 2018.

[49] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” in Advances in Neural Information Processing Systems, Datasets

[50] Y. Zhu et al., “robosuite: A modular simulation framework and benchmark for robot learning,” arXiv preprint arXiv:2009.12293, 2020.

and Benchmarks Track, 2023.

[51] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012, pp. 5026–5033.