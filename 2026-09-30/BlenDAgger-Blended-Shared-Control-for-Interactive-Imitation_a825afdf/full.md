# BlenDAgger: Blended Shared Control for Interactive Imitation Learning

Cailyn Smith   
The Robotics Institute,   
School of Computer Science   
Carnegie Mellon University   
Pittsburgh, USA   
cailyns@andrew.cmu.edu   
Geoffrey Sun   
School of Computer Science   
Carnegie Mellon University   
Pittsburgh, USA   
gsun2@andrew.cmu.edu   
Henny Admoni<sup>†</sup>   
The Robotics Institute,   
School of Computer Science   
Carnegie Mellon University   
Pittsburgh, USA   
hadmoni@andrew.cmu.edu   
Zackory Erickson<sup>†</sup>   
The Robotics Institute,   
School of Computer Science   
Carnegie Mellon University   
Pittsburgh, USA   
zackory@cmu.edu

Abstract—Robot policies are frequently trained from human corrections, yet teleoperating a robot to provide corrections is burdensome, and human demonstrators are not always optimal. We propose Blended DAgger (BlenDAgger), an approach for collecting data to train imitation learning policies by using shared control to blend the policy’s and demonstrator’s actions during interventions. By blending human and policy actions, we aim to improve the autonomous performance of manipulation policies. We validate our approach across five manipulation tasks, two in the real world and three in simulation. Our approach achieves higher autonomous performance by 30 or more percentage points on two real-world tasks compared to a typical humangated correction approach (HG-DAgger). We also investigate the advantages of BlenDAgger that allow for higher autonomous performance, finding that BlenDAgger results in 57% smoother transitions between policy control and human interventions, and 14% higher trajectory similarity to the training data. In a user study (n=14) on two real-world tasks, we find that BlenDAgger results in faster data collection (BF=13.32), and we do not find a difference in subjective perceptions. These results show that blended shared control leads to higher autonomous performance compared to typical methods for fine-tuning robot policies from fully teleoperated interventions.

Index Terms—Imitation learning, telerobotics and teleoperation, human factors and human-in-the-loop

## I. INTRODUCTION

Imitation learning has been successfully used to train policies to accomplish complex manipulation tasks using humancollected demonstration data [1]–[4]. To address the covariate shift introduced by the sequential nature of manipulation tasks, researchers often use DAgger [5] and its variants, in which an expert corrects policy errors during execution. Humangated and robot-gated forms of DAgger [6]–[8] have shown success in alleviating the covariate shift problem. However, these gated interventions put the human fully in control during interventions. Fully teleoperating a robot can result in suboptimal interventions and may be burdensome. Our insight is that by using blended shared control, where the human’s and robot’s actions are combined at each timestep based on their similarity and the robot’s uncertainty, we can improve the autonomous performance of manipulation policies (see Fig. 1).

![](images/dbdfb1f433a84e0ed5574a814ec63825d94082889147e9d633e23e08e7c81eda.jpg)  
Fig. 1: BlenDAgger shares control between the robot’s policy and the human demonstrator during corrections, resulting in higher autonomous success and faster data collection.

Blended shared control [9] has been studied extensively in assistive robotics [10]–[12]. Some existing work incorporates shared control to expand the robot’s skills as the user interacts with the system multiple times [13]–[15], but these works focus on shared control as a way to improve human-robot collaboration where the human remains in the loop. Our work aims to do the reverse: rather than using imitation learning to improve shared control systems, we leverage blended shared control to fine-tune manipulation policies. We ask the research question: Can blended shared control provide a valuable learning signal for imitation learning?

Blending the human’s and robot’s actions at each timestep offers several potential benefits over gated DAgger approaches:

1) Smoother corrections. Combining suboptimal human inputs with policy actions may smooth out corrections.

2) Reduced excursions to out-of-distribution (OOD) states. Unlike gated control, which alternates between the policy’s OOD rollout states and human-corrected states, blending may keep the robot more consistently near in-distribution states.

3) Better-timed interventions. Seeing the policy’s actions while correcting may help demonstrators judge when to start and stop intervening.

We evaluate BlenDAgger for fine-tuning policies on five manipulation tasks: two in the real world and three in simulation. We find that BlenDAgger achieves higher autonomous task performance than HG-DAgger on four tasks when trained on data collected by experienced demonstrators, with 30 or more percentage points higher performance on the real-world tasks. To evaluate whether the autonomous performance gains of our approach extend to non-expert demonstrators, we conduct a user study with 14 novice users. We find that BlenDAgger reduces task time and do not find an impact on subjective perceptions of providing corrections. We further find that BlenDAgger outperforms HG-DAgger on three of the four conditions when trained with participant data, but that gains from both BlenDAgger and HG-DAgger are smaller than with expert data.

We make the following contributions:

1) We propose an interactive imitation learning framework, Blended DAgger (BlenDAgger), that integrates adaptive, uncertainty-aware shared control into policy fine-tuning.

2) We demonstrate the efficacy of BlenDAgger across diverse simulation and real-world manipulation tasks, showing substantial gains in policy success rates over a traditional gated baseline.

3) We conduct a user study with novice operators to evaluate our proposed method against HG-DAgger, providing insights into blended shared control for data collection.

## II. RELATED WORK

## A. Human- and Robot-Gated DAgger

Existing approaches for learning from human interventions are typically either human-gated or robot-gated. In humangated approaches, such as HG-DAgger [6] and EIL [7], a human supervises the policy rollouts and intervenes as needed to correct the policy. HG-DAgger has been extended in many works, including by modifying its data sampling during training [16], [17], learning a residual policy [18], or using a compliant interface [19]. In robot-gated approaches [8], [20], [21], the robot actively queries an expert when unsure of the best action or in a risky state. In contrast, our approach is not gated but instead allows the human demonstrator to provide corrections that are blended with policy actions.

## B. Bilateral and Compliant Control

Existing work has incorporated ideas of continuous shared control with imitation learning, but not with explicit blending of policy and human actions. Work on bilateral control, including HACTS [22] and RoboCopilot [4], aims to synchronize autonomous policy rollouts with human interventions primarily through hardware advancements. CHG-DAgger [23] uses multilateral control to combine human interventions with policy rollouts rather than using a gated mechanism. Unlike our work, their use of multilateral control means that there is no explicit determination of shared autonomy arbitration at the action level. Furthermore, kinesthetic teaching and compliant interfaces have been used in prior work. By providing force corrections during policy rollouts rather than action labels, these interfaces, such as CR-DAgger [19] and that of Abi-Farraj et al. [24], effectively combine human and robot actions during deployment through the force applied to the robot rather than by explicitly blending those actions with shared control.

## C. Learning from Shared Autonomy

Prior works have used shared control for learning manipulation tasks, typically optimizing for the quality of shared control itself. Many of these works use shared control in a discrete, gated fashion rather than blending actions. ILSA [13] uses a gated mechanism between human and robot control depending on how significantly their actions differ. In [25], Cui et al. incorporate shared autonomy where the human demonstrator has full control of the arm movements while the robot policy autonomously commands a dexterous hand. SARI [14] and CASA [15] blend human and robot control while deferring to the human when the robot encounters an unfamiliar task or goal. These works aim to improve the robot’s behavior to stay in a shared control loop and therefore do not compare learning efficiency to gated imitation learning approaches. Li et al. [26] blend human corrections and policy actions for wrist motion in dexterous manipulation by adding human velocity commands to the policy velocity, which effectively results in a fixed arbitration value rather than an adaptive one. In contrast to prior work, our proposed algorithm leverages existing adaptive blended shared control formulations from assistive robotics, using action similarity and policy confidence, and combines this with imitation learning to fine-tune manipulation policies.

## III. FORMULATION

## A. Preliminaries

We train a robot policy $\pi _ { \theta }$ on a demonstration dataset $\boldsymbol { \mathcal { D } } = \{ ( \mathbf { O } _ { t } , \mathbf { A } _ { t } ) \}$ , where each $\mathbf { O } _ { t } = \left( o _ { t - T _ { o } + 1 } , \ldots , o _ { t } \right)$ is a history of $T _ { o }$ observations $o \in \mathcal { O }$ containing RGB images and proprioceptive state, and $\mathbf { A } _ { t } = ( a _ { t } , \ldots , a _ { t + 1 5 } )$ is an action chunk of length 16. Following Chi et al. [27], the policy models the conditional distribution $\pi _ { \theta } ( \mathbf { A } _ { t } \mid \mathbf { O } _ { t } )$ and is trained with the diffusion denoising objective. At inference, the policy samples an action chunk using Denoising Diffusion Implicit Models (DDIM) [28] and executes 8 actions before replanning.

In HG-DAgger [6], the executed policy is gated on whether the demonstrator intervenes:

$$
\pi _ { i } ( x _ { t } ) = g ( x _ { t } ) \pi _ { H } ( x _ { t } ) + \left( 1 - g ( x _ { t } ) \right) \pi _ { \theta _ { i } } ( \mathbf { O } _ { t } ) ,\tag{1}
$$

where i is the aggregation round, $\pi _ { H } ( x _ { t } )$ is the demonstrator’s policy, $x _ { t }$ is the full state, and $g ( x _ { t } ) \in \{ 0 , 1 \}$ is 1 when the demonstrator chooses to intervene and is 0 otherwise. Each round’s data $\mathcal { D } _ { i }$ is aggregated, $\mathcal { D }  \mathcal { D } \cup \mathcal { D } _ { i }$ , before retraining.

## B. BlenDAgger

In contrast to HG-DAgger, we leverage the policy blending formalism from [9], which is frequently used in goal-

![](images/3355d89ca21c8960473d644a285fff83764826844916709f53c9b3be6583d353.jpg)  
Fig. 2: BlenDAgger pipeline. When the demonstrator chooses to intervene, their action is blended with the policy’s predicted action. The resulting blended action is executed on the robot and used to fine-tune the policy.

conditioned assistive robotics domains. Rather than gating the policy based on human interventions, we blend control as:

$$
\pi _ { i } ( x _ { t } ) = ( 1 - \alpha _ { t } ) \pi _ { H } ( x _ { t } ) + \alpha _ { t } \pi _ { \theta _ { i } } ( \mathbf { O } _ { t } ) ,\tag{2}
$$

where $\alpha _ { t } \in [ 0 , 1 ]$ is the arbitration function. Following the convention from [9], $\alpha _ { t } = 0$ corresponds to pure teleoperation and $\alpha _ { t } = 1$ to full autonomy. This is a superset of HG-DAgger, becoming HG-DAgger if α goes to 0 during interventions. Unlike traditional policy blending, which predicts the user’s goal and assists toward it [9], we assume no goal set and blend only at the trajectory execution layer.

## C. Arbitration Function

When the human operator decides to intervene during policy execution, they use a device that allows 6DoF control. However, since operators rarely control all 6DoFs simultaneously, we group together DoFs and then compute α separately for each DoF group. Letting $x , y , z$ denote translation and ϕ, θ, ψ denote roll, pitch, and yaw, the DoF groups are,

$$
j \in \big \{ \{ x , y \} , \ \{ z \} , \ \{ \phi , \theta , \psi \} \ \big \} .\tag{3}
$$

We chose these groupings since manipulation tasks are often done on planar, human-made surfaces. We use $\mathbf { a } _ { j , t } ^ { r }$ and $\mathbf { a } _ { j , t } ^ { h }$ for the robot and human commands at timestep t of group j, with each axis normalized to [−1, 1].

Taking inspiration from shared control systems in assistive robotics that frequently blend control based on confidence [9]– [11] and similarity [29], [30], we utilize the following components as part of blending arbitration (see Fig. 2):

• Similarity. We use cosine similarity between the policypredicted action and the human action:

$$
S _ { j , t } = \frac { ( \mathbf { a } _ { j , t } ^ { h } ) ^ { T } \mathbf { a } _ { j , t } ^ { r } } { \Vert \mathbf { a } _ { j , t } ^ { h } \Vert \left. \mathbf { a } _ { j , t } ^ { r } \right. } .\tag{4}
$$

Since we use delta actions and represent orientation in axis-angle form, the action space is approximately linear, which satisfies the assumption behind cosine similarity.

• Policy uncertainty. We estimate policy uncertainty with k-nearest neighbors [31], using the Euclidean distance from the policy’s encoder embedding (see Section IV-G) of the current observation to the 10-th nearest embedding of the policy’s training data. This is normalized to $\hat { u } _ { t } ~ \in ~ [ 0 , 1 ]$ using bounds calibrated on that same data, which yields a confidence $C _ { t } = 1 - \hat { u } _ { t }$ . We recalibrate uncertainty with each data aggregation round.

• Magnitude of intervention. As a demonstrator pushes more in a specific direction, this may indicate that they want a quicker correction in that direction. We compute this as the magnitude of the normalized human action,

$$
M _ { j , t } = \operatorname* { m i n } \big ( \lVert \mathbf { a } _ { j , t } ^ { h } \rVert , \ 1 \big ) .\tag{5}
$$

Using these components, we compute α with a sigmoid, as sigmoids and other ramping functions have often been used in the shared control literature [9], [10], [32]. Action similarity and policy confidence are each passed through a sigmoid:

$$
f ( w , M ) = \frac { 1 } { 1 + \exp \big ( - \beta \cdot ( w - M - \delta ) \big ) } ,\tag{6}
$$

where $\beta = 1 0$ is a fixed slope and $\delta = 0 . 5$ centers the function on [0, 1]. When the demonstrator actively intervenes to control group j, the target arbitration for that group is the sigmoid resulting from the product of the two terms:

$$
\alpha _ { j , t } ^ { \mathrm { t g t } } = f \bigl ( S _ { j , t } , M _ { j , t } \bigr ) \cdot f \bigl ( C _ { t } , M _ { j , t } \bigr ) .\tag{7}
$$

This arbitration function allows for more human control when the demonstrator’s actions diverge from the policy’s predicted actions, the policy has high uncertainty, or the demonstrator’s input magnitude is large. This allows for blending while still providing the operator with sufficient ways to regain control (see Section V-D for analysis of responsiveness). Finally, the arbitration is smoothed across timesteps so that the demonstrator is able to react to the changing amount of control:

$$
\alpha _ { j , t } = \lambda \alpha _ { j , t - 1 } + \left( 1 - \lambda \right) \alpha _ { j , t } ^ { \mathrm { t g t } } \ ; \ \lambda = 0 . 5 .\tag{8}
$$

![](images/67ed18303e960024287849271d496bfb3acbde736eb352758c221edec929e793.jpg)  
(a) Cupboard Stowing

![](images/08ff6c0272add5d65232581814338dd78f6eda734d85eec9ff9232aa7fd28d21.jpg)  
(b) Almond Scooping

![](images/7cf25210f84cbfe3f3443570c8fbb9a45234e4084c1ce3914b37a21dc7e3b907.jpg)  
(c) Prepare Coffee  
Fig. 3: Experiment Tasks.

![](images/58234d06d7f2956672cb4d1040492a1e308e758a39c87d7e4d7b58f0bb5b238f.jpg)  
(d) Microwave Thawing

![](images/3e0ee93ee6fc2f1f3d209ea5be11cd39554c0e225b60b4566de7bc189733396e.jpg)  
(e) Pick and Place

We use EMA smoothing with the same value of $\lambda = 0 . 5$ for HG-DAgger at the start of interventions. This is to provide a fair comparison and isolate the effect of blending during interventions from the effect of smoothing. An example of α values over time from a segment of a participant’s collection episode is shown in Fig. 4.

## IV. METHODS

We evaluate the impact of BlenDAgger on policy performance with five manipulation tasks, three in simulation and two in the real world. We train base policies on human demonstrations and then fine-tune with interventions collected by two expert demonstrators, who are authors on this paper, with either HG-DAgger or BlenDAgger. For all tasks, we perform five iterative rounds of HG-DAgger and BlenDAgger. Per method and per round, 10 episodes are collected for the simulation tasks and 15 episodes for the real-world tasks. We use the diffusion policy architecture and training parameters described in Section IV-G. The following sections describe the tasks and evaluation, and Table II describes the task randomization. To evaluate user perceptions of BlenDAgger and whether non-expert data improves policy performance, we also conduct a user study $( \mathrm { n } { = } 1 4 ) ^ { 1 }$ , described in Section IV-C.

## A. Simulation Experiments

We first evaluate our approach in the RoboCasa simulation environment [33], using a 7DoF Franka Emika Panda robot arm. We use the PrepareCoffee, MicrowaveThawing, and PickPlaceCounterToCab tasks (see Fig. 3(c–e)), but use one fixed scene and object per task with randomized object and robot starting positions and orientations. From a base policy trained on 50 human demonstrations, we fine-tune with interventions collected using a SpaceMouse device. We train with Intervention-Weighted Regression (IWR) [16], drawing half of all training samples from $\mathcal { D } _ { I } ,$ , which contains only the intervention segments, and half from ${ \mathcal { D } } _ { R } ,$ , which contains the base demonstrations and the autonomous rollout segments. Policy performance is evaluated as the binary success rate on the task, averaged over 100 rollouts.

## B. Real-World Experiments

To validate our simulation results with real-world data collection, we perform real-world experiments with two longhorizon tasks, shown in Fig. 3(a–b), using a 6DoF UFAC-TORY xArm robot. Data was collected by trained demonstrators using an Oculus VR controller, and participants used a SpaceMouse since it can be easier to learn to use. The Cupboard Stowing task is motivated by household grocery stowing. This task involves opening a cupboard door, picking up a can, placing the can in the cupboard, and closing the cupboard door. The Almond Scooping task is motivated by grocery bulk-bin shopping. It requires picking up a scooper, scooping almonds, depositing almonds in a bag, and returning the scooper to a napkin. The task is considered a success if no almonds are spilled and >5 almonds are deposited.

For each task, we collect 45 demonstrations to train a base policy and then fine-tune with rounds of interventions. After finding that policy performance decreased if we trained with autonomous segments where the human did not intervene, likely due to the autonomous rollouts being lower quality than in simulation, we draw half of the training samples from $\mathcal { D } _ { I }$ and half from the original base policy demonstrations. Autonomous policy performance after each round of finetuning is averaged over 20 rollouts.

## C. User Study

To evaluate the experience of providing corrections with our approach, we recruited 15 participants. We restricted analysis to non-expert users, removing one participant who reported the maximum rating on both robotics-experience measures, leaving n=14. Intervention method (BlenDAgger vs. HG-DAgger) was varied within subjects and the task (the two real-world tasks) was varied between subjects. Within subject, we varied the policy the methods were applied to (base policy vs. round-1 fine-tuned policy). This yielded four trial sets per participant (2 intervention methods × 2 policies). The round-1 fine-tuned policy was fine-tuned on expert data to remain consistent across participants and due to time constraints of fine-tuning during the study. This allowed for measuring the collection experience, but did not evaluate the compounding effect of participant data across rounds.

![](images/5bf97d464c19518015c9b723b1c6b9b72a4d5eaaba501edc6e5e837094839c8a.jpg)  
(a) Cupboard Stowing

![](images/72748bc0a44cfd700908d167f62d9a196c76dce6018fd9accf30fc6aa85e0b34.jpg)  
(b) Almond Scooping

![](images/19ef3e3e11e13110341b281770b7480ab8a31434a060155dd150303ecc090548.jpg)  
(c) Prepare Coffee

![](images/4307533ffd6e63a586c0f3d6ff1da50b5ee6c5d341649bce790d30a1633e14c1.jpg)  
(d) Microwave Thawing

![](images/715678e66ad980db8972f852ee1f5ff6b360aef547ae447473db036408d7f4c3.jpg)  
(e) Pick and Place  
Fig. 5: Autonomous performance on real-world (a–b) and simulation (c–e) tasks. Bar colors indicate the highest subtask reached during an episode. Full task success is labeled.

After signing informed consent, each participant was shown how to use the SpaceMouse device and practiced using it. They were introduced to the task and practiced it twice with full teleoperation, twice with BlenDAgger, and twice with HG-DAgger. The order of intervention methods was counterbalanced between participants. Experimental trials began, in which they completed the four sets of five trials each. Participants answered a questionnaire after each of the four sets of trials. The questionnaire contained the following:

1) Weighted NASA TLX Workload Scale [34].

2) Four Likert scales, containing three 7-point items per scale. These were: I knew when the robot needed to be corrected, I was willing to intervene, The robot helped me provide corrections, and Perceived smoothness (Cronbach’s alpha = 0.83, 0.77, 0.48, 0.82).

Of the participants, 9 were female and 5 were male, with a mean age of 30.4 (SD=12.5). On a scale of 0 to 5 (0=no experience, 5=professional experience), the mean prior experience with robotics was 1.3 (SD=1.0). On a scale of 0 to 5 (0=no experience, 5=regularly control a robot arm), the mean prior experience controlling a robot arm was 0.5 (SD=0.7). The study lasted two hours and participants were compensated.

## D. Statistical Analysis

We conducted Bayesian analysis using [35], which allows for reporting effects both for and against a hypothesis. We report $B F _ { 1 0 }$ from the Bayesian equivalent of RM-ANOVAs, with method and round as repeated measures. The random effect is participant for the user study and task for the expert data. Results were interpreted using [36], with BF∈ [0.333, 3.0] as inconclusive, BFs above 3.0 as evidence for an effect, and BFs below 0.333 as evidence against an effect. For example, a Bayes factor of 3 means the data are 3 times more likely under the alternative hypothesis than under the null.

## E. Hypotheses

We hypothesize that, compared to HG-DAgger:

• H1: Policies trained with BlenDAgger data will achieve higher autonomous performance.

• H2: BlenDAgger will require less human effort to collect intervention data.

• H3: BlenDAgger will yield higher subjective perceptions (measured via workload and Likert scales).

We test H1 and H2 in our main data-collection experiments, and further evaluate all three in a user study with nonexpert operators to assess whether the benefits extend beyond trained demonstrators. We also analyze whether the performance improvements from BlenDAgger correspond with our motivations for using shared control, described in Section I.

## F. Analysis Metrics

We call a timestep t an intervention step if $M _ { t } > 0 ,$ , and a run of consecutive intervention steps an intervention. We define the following metrics for later analysis, averaged within an episode:

• Action discontinuity is $\lVert \pi _ { i } ( x _ { t } ) - \pi _ { i } ( x _ { t - 1 } ) \rVert _ { 2 }$ at the first and last step of each intervention. We use this as a proxy for the smoothness of a control change, where lower action discontinuity indicates greater smoothness.

• Intervention timing is the timestep of each control transition, measured from the start of the episode. Since the policies have different failure modes after fine-tuning, we only compare intervention timing on round 1 data.

• State distance is the mean distance, in joint-angle space, from a state to its 10 nearest neighbors [37] in the training data. The training data includes base policy demonstrations and interventions from previous rounds.

• Trajectory distance is the discrete Frechet distance [38],´ in joint-angle space, from an intervention to the closest equal-length trajectory segment in the training data.

## G. Policy Architecture and Training Parameters

Simulation and real-world experiments both use a diffusion policy [27] with a 1-D temporal convolutional U-Net and a per-camera ResNet-18 visual encoder. In simulation, we use the Robomimic implementation [39], and in the real world, we use the Diffusion Policy codebase [27]. Table I describes the architecture and training hyperparameters used. For each task, the base policy checkpoint was chosen by a short evaluation to determine the best-performing one. These were found to be epoch 4000 for Cupboard Stowing, epoch 3750 for Almond Scooping, and epoch 900 for Pick and Place. The other two simulation tasks had 0% task success, so we chose epoch 1000. Each fine-tuning round initializes from the base policy checkpoint and freezes the visual encoder.

TABLE I: Policy architecture and training hyperparameters.
<table><tr><td></td><td>Simulation</td><td>Real</td></tr><tr><td>Architecture</td><td></td><td></td></tr><tr><td>External / wrist cameras</td><td>2 / 1</td><td>1 /1</td></tr><tr><td>Image resolution</td><td colspan="2"> $1 2 8 \times 1 2 8$ </td></tr><tr><td>Random crop</td><td>116 × 116</td><td>112 × 112</td></tr><tr><td>Proprioception dim.</td><td>16</td><td>14</td></tr><tr><td>Action dim.</td><td>12</td><td>10</td></tr><tr><td>Obs. history  $T _ { o }$ </td><td>2</td><td>3</td></tr><tr><td>Pred. horizon  $T _ { p }$ </td><td colspan="2">16</td></tr><tr><td>Executed actions  $T _ { a }$ </td><td colspan="2">8</td></tr><tr><td>Training / DDIM inference steps</td><td colspan="2">100 /  16</td></tr><tr><td>Training</td><td></td><td></td></tr><tr><td>Optimizer (base)</td><td>Adam</td><td>AdamW</td></tr><tr><td>Optimizer (fine-tune)</td><td>AdamW</td><td></td></tr><tr><td>LR (base)</td><td> $1 \times 1 0 ^ { - }$  -4</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR (fine-tune)</td><td> $1 \times 1 0 ^ { - }$ </td><td></td></tr><tr><td>Schedule (base)</td><td>constant</td><td>cosine</td></tr><tr><td>Schedule (fine-tune)</td><td colspan="2">constant w/ warmup</td></tr><tr><td>Batch size (base)</td><td>128</td><td>512</td></tr><tr><td>Batch size (fine-tune)</td><td></td><td></td></tr><tr><td>Epochs (fine-tune)</td><td>150</td><td>500</td></tr></table>

TABLE II: Randomization for data collection and evaluation.
<table><tr><td></td><td colspan="3">Object</td><td colspan="2">Robot base</td><td colspan="2">RoboCasa</td></tr><tr><td>Task</td><td>Object</td><td>x/y Pos. (cm)</td><td>Yaw (rad)</td><td>x/y Pos. (cm)</td><td>Yaw (rad)</td><td>Layout Style</td><td></td></tr><tr><td>Prepare Coffee</td><td>Mug</td><td>±10 /  10</td><td>0</td><td>±2.5 / 4</td><td>±0.17</td><td>7</td><td>9</td></tr><tr><td>Microwave Thawing</td><td>Carrot</td><td>±5 / 5</td><td>0</td><td>±1.5 / 4</td><td>±0.17</td><td>4</td><td>0</td></tr><tr><td>Pick and Place</td><td>Lemon</td><td>±10 / 10</td><td>2π</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td rowspan="2">Cupboard Stowing</td><td>Can</td><td>±11/ 11</td><td>2π</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td>Cupboard</td><td>0 / 0</td><td>π/6</td><td>0</td><td>0</td><td></td><td></td></tr><tr><td rowspan="2">Almond Scooping</td><td>Bag / Scooper</td><td>±5 / 8</td><td>0</td><td>0</td><td>0</td><td></td><td>一</td></tr><tr><td>Almond bin</td><td>±3 / 5</td><td>0</td><td>0</td><td>0</td><td></td><td></td></tr></table>

## V. RESULTS

## A. Performance Results

As shown in Fig. 5(a–b), BlenDAgger results in higher autonomous performance than fine-tuning with HG-DAgger in every round on both real tasks, ending 35 percentage points higher on the Almond Scooping task (50% vs. 15%) and 30 percentage points higher on the Cupboard Stowing task (70% vs. 40%). As shown in Fig. 5(c–e), BlenDAgger also achieves higher success than HG-DAgger on two of the simulated tasks: on Prepare Coffee it is higher in every round (reaching 27% vs. 19%), and on Microwave Thawing it reaches 12% vs. 5%, which is low in absolute terms across both methods due to the task being long-horizon with 0% base policy performance. On Pick and Place, BlenDAgger achieves 85% task success with 33% fewer episodes of interventions than HG-DAgger, but both flatline at ∼90% success, demonstrating that BlenDAgger’s main performance gains are for complex, long-horizon tasks. The results on these five tasks support H1 for expert-collected data. BlenDAgger reduces data collection time by 5.9% (BF=70.8), while the number of intervention steps and total magnitude of interventions do not differ reliably between methods (BF=0.57, 0.28). This supports H2 for expert demonstrators. Since the number of intervention steps and magnitude of interventions do not differ significantly, the performance gap is attributable to blended control, rather than when or how the demonstrator chose to intervene.

![](images/71f95294631d373fbd448e3490aa7862523c1faebe85ef42ff89bb765b9eedfe.jpg)  
(a) Data Collection Time

![](images/1d835c191fd62e77afc330be3b0ce3111ea3a79fdf952893221c72e634859188.jpg)  
(b) Intervention Steps

Fig. 6: Participants complete the task faster and with fewer intervention steps with BlenDAgger. Error bars show 95% CIs.  
![](images/4369724d83aa2d0c173c1187d381dd98bf442b33bd1d19bae3d93c08a43d6dbb.jpg)  
(a) Cupboard Stowing

![](images/39256f51dd68a8831a62cd697b32e86e506cee2b749629a698030db1ba3b2cbb.jpg)  
(b) Almond Scooping  
Fig. 7: Autonomous performance (user study data).

## B. User Study Results

In a user study, BlenDAgger resulted in lower task completion time on successful episodes (BF=13.32), shown in Fig. 6. BlenDAgger also resulted in fewer intervention steps (BF=20.48), using a rank-based test because the paired differences were strongly skewed (Shapiro–Wilk p = .001). We did not find statistically significant differences in task success during data collection (BF=0.63), workload (BF=0.54), or three of the Likert scales (Knew when to correct BF=0.43, Willing to intervene BF=0.60, Helped during corrections BF=0.63). The Helped during corrections scale had low internal consistency (Cronbach’s alpha = 0.48), so its result should be interpreted with caution. There was evidence against an effect on the Perceived smoothness Likert scale (BF=0.26). These results support H2 for participant data collection but do not support H3, showing that BlenDAgger reduces task completion time and time spent intervening but does not result in an effect on subjective perceptions.

As shown in Fig. 7, participant-collected BlenDAgger data produced policy improvement on both rounds of both tasks and outperformed HG-DAgger in three of four conditions: both rounds of Almond Scooping and round 1 of Cupboard Stowing. On the Almond Scooping task, BlenDAgger increased performance in both rounds, whereas HG-DAgger caused performance to decrease to 0% full task success in both rounds. Consistent with prior findings that low-quality interventions limit policy performance [39], [40], there remains a performance gap between data collected by trained demonstrators and non-experts. This provides some support for H1 on user study data, but shows that limitations remain with training on robot manipulation data collected by novice users.

![](images/de3aa79166083ec0273cf02d61866cd6d6964af2421b916de1ab8577bccfe19d.jpg)

(a) Action Discontinuity during Control Transitions  
![](images/4abe129b87d14820e67237696e075e676c2e4c193695c9323005b7aff9264523.jpg)  
(b) Trajectory Distance to Training Data  
Fig. 8: BlenDAgger has smoother action transitions and lower trajectory distance to training data. Error bars show 95% CIs.

## C. Analysis of BlenDAgger Advantages

Based on the performance improvements supporting H1, we analyze the advantages of BlenDAgger and report the BFs in Table III. As shown in Fig. 8, we find that BlenDAgger results in 57% smoother control transitions when interventions start and end compared to HG-DAgger. Critically, these smoother transitions are caused by blending, not by smoothing at control changes, since HG-DAgger and BlenDAgger share an EMA smoothing value of 0.5. During interventions, the mean arbitration was $\alpha = 0 . 5 0 \ ( \mathrm { S D } \ 0 . 3 2 )$ for expert demonstrators and $\alpha ~ = ~ 0 . 5 7$ (SD 0.33) for participants, confirming that control was genuinely shared and that α was similar between trained demonstrators and participants. For novices, we find that BlenDAgger visits fewer highly OOD states $( { > } 9 9 ^ { \mathrm { t h } }$ percentile of training distances) than HG-DAgger. We further find that BlenDAgger interventions have 14% higher trajectory similarity to the training data. The timing of interventions is inconclusive. This analysis suggests that higher smoothness and higher trajectory similarity to training data may contribute to BlenDAgger’s policy performance improvements.

TABLE III: BFs comparing BlenDAgger to HG-DAgger on analysis metrics. Bold indicates evidence of an effect.
<table><tr><td>Metric</td><td>Experts</td><td>Participants</td></tr><tr><td>Action discontinuity, intervention start/end</td><td>6.2e7 / 1.5e6</td><td>15.34 / 3.8e36</td></tr><tr><td>Highly OOD states visited</td><td>0.34</td><td>17.07</td></tr><tr><td>Trajectory distance to training data</td><td>3.35</td><td>3.10</td></tr><tr><td>Intervention timing, intervention start/end</td><td>0.51 / 0.53</td><td>0.99 / 1.66</td></tr></table>

TABLE IV: Ablation performance and responsiveness. Bold marks the best overall and the best BlenDAgger variant.
<table><tr><td rowspan="2">Method</td><td colspan="3">Performance</td><td rowspan="2">Responsive- ness (s)</td></tr><tr><td>R1</td><td>R2</td><td>R3</td></tr><tr><td>HG-DAgger</td><td>0.01 ± 0.00</td><td>0.02±0.01</td><td>0.05±0.01</td><td>0.15</td></tr><tr><td>BlenDAgger</td><td>0.08 ± 0.03</td><td>0.15±0.02</td><td>0.27 ± 0.04</td><td>0.20</td></tr><tr><td>Without uncertainty</td><td>0.00</td><td>0.14</td><td>0.16</td><td>0.28</td></tr><tr><td>Without similarity</td><td>0.01</td><td>0.15</td><td>0.22</td><td>0.20</td></tr><tr><td>Without magnitude</td><td>0.03</td><td>0.13</td><td>0.17</td><td>0.20</td></tr><tr><td>Only uncertainty</td><td>0.08</td><td>0.09</td><td>0.23</td><td>0.33</td></tr><tr><td>Only similarity</td><td>0.00</td><td>0.15</td><td>0.18</td><td>0.85</td></tr><tr><td>Only magnitude</td><td>0.01</td><td>0.18</td><td>0.11</td><td>1.02</td></tr><tr><td>Fixed-α  $( \alpha = 0 . 5 )$ </td><td>0.03</td><td>0.13</td><td>0.15</td><td>1.15</td></tr><tr><td>Human action targets</td><td>0.00</td><td>0.05</td><td>0.23</td><td></td></tr></table>

## D. Ablations

We have two desired aspects for a blending formulation: 1) increased policy performance and 2) sufficient control over the robot when providing corrections. On the Prepare Coffee simulation task, we compare our BlenDAgger formulation against a fixed arbitration of $\alpha = 0 . 5$ and adaptive blending ablations with each of the three components described in Section III-C removed and each on its own. As shown in Table IV, we find that many of the ablations result in similar autonomous performance, demonstrating that blended shared control is the primary cause of higher performance, rather than a specific arbitration. These results also indicate that the BlenDAgger arbitration is not adapted specifically to our demonstrators as performance gains remain in a fixed-α approach. The components in BlenDAgger instead contribute towards the demonstrator’s ability to quickly regain some control of the robot’s movement, which we measure as the median time from the onset of an intervention to when the robot’s executed action is within 20% of the user’s commanded action direction and speed, shown in Table IV. We evaluate the effect of executing blended actions while training only on the portion corresponding to human actions rather than blended actions (Human action targets), which results in lower autonomous task success. We also show the mean and standard deviation in performance of HG-DAgger and BlenDAgger across three training seeds; the first seed is used for the following round of data collection.

## VI. LIMITATIONS AND FUTURE WORK

In this paper, we show simulation and real-world results for blended shared control as an alternative to HG-DAgger for learning manipulation tasks. While our arbitration formulation is inspired by prior shared-control literature and validated through ablations, we adopt fixed arbitration parameters and DoF groupings. Future work could learn or adapt these online, potentially personalizing to each operator. Although intervention data in interactive imitation learning is typically collected by a small number of skilled operators, frequently the researchers themselves [6], [16], [17], validating BlenDAgger with a larger pool of trained operators could help characterize how robustly the approach performs. Our ablations (Section V-D) suggest the performance gains do not depend on demonstrators adapting to the arbitration, as performance persists under different arbitrations, so we would expect to see similar performance trends with other demonstrators. Our user study was a single session, leaving open how users may adapt to the system over repeated use. Finally, both methods show limited gains when fine-tuned on non-expert data. Improving policy learning from imperfect demonstrations is an active area of research, and integrating blended shared control with existing approaches that filter or reweight lowquality corrections could lead to improved learning efficiency.

## VII. CONCLUSION

In this work, we formulate and validate a novel approach for interactive imitation learning using blended shared control, which we term BlenDAgger. BlenDAgger blends human corrections with the robot policy based on action similarity, policy uncertainty, and the amount of human intervention. Compared to HG-DAgger, our method achieves higher autonomous performance on four tasks, outperforming HG-DAgger by 30 or more percentage points on both real-world tasks. BlenDAgger also results in smoother intervention transitions and more similar trajectories to its training data. In a user study, we find statistically significant evidence that BlenDAgger speeds up data collection and do not find differences in subjective perceptions. Our results show the efficacy of blended shared control for fine-tuning manipulation policies.

## ACKNOWLEDGMENT

The authors thank David Yi for his support with data collection. This research is supported by ARPA-H through the University of Pittsburgh under the RAMMP (Robotic Assistive Mobility and Manipulation Platform Providing Independence for People with Disabilities) project (grant number: 75N99223S0001). This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No(s) DGE2140739 and DGE2631988. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## REFERENCES

[1] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” Robotics: Science and Systems, 2023.

[2] J. Wu, W. Chong, R. Holmberg, A. Prasad, Y. Gao, O. Khatib, S. Song, S. Rusinkiewicz, and J. Bohg, “Tidybot++: An open-source holonomic mobile manipulator for robot learning,” in Conference on Robot Learning, 2024.

[3] Z. Fu, T. Z. Zhao, and C. Finn, “Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation,” in Conference on Robot Learning (CoRL), 2024.

[4] P. Wu, Y. Shentu, Q. Liao, D. Jin, M. Guo, K. Sreenath, X. Lin, and P. Abbeel, “RoboCopilot: Human-in-the-loop interactive imitation learning for robot manipulation,” arXiv, no. arXiv:2503.07771, 2025.

[5] S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in International Conference on Artificial Intelligence and Statistics, 2011.

[6] M. Kelly, C. Sidrane, K. Driggs-Campbell, and M. J. Kochenderfer, “HG-DAgger: Interactive imitation learning with human experts,” in International Conference on Robotics and Automation (ICRA), 2019.

[7] J. Spencer, S. Choudhury, M. Barnes, M. Schmittle, M. Chiang, P. Ramadge, and S. Srinivasa, “Learning from interventions: Human-robot interaction as both explicit and implicit feedback,” in Robotics: Science and Systems, 2020.

[8] K. Menda, K. Driggs-Campbell, and M. J. Kochenderfer, “EnsembleDAgger: A bayesian approach to safe imitation learning,” in International Conference on Intelligent Robots and Systems, 2019.

[9] A. D. Dragan and S. S. Srinivasa, “A policy-blending formalism for shared control,” The International Journal of Robotics Research, 2013.

[10] D. Gopinath, S. Jain, and B. D. Argall, “Human-in-the-loop optimization of shared autonomy in assistive robotics,” IEEE Robotics and Automation Letters, 2016.

[11] S. Javdani, H. Admoni, S. Pellegrinelli, S. S. Srinivasa, and J. A. Bagnell, “Shared autonomy via hindsight optimization for teleoperation and teaming,” The International Journal of Robotics Research, 2018.

[12] D. P. Losey, C. G. McDonald, E. Battaglia, and M. K. O’Malley, “A review of intent detection, arbitration, and communication aspects of shared control for physical human–robot interaction,” Applied Mechanics Reviews, 2018.

[13] Y. Tao, G. Qiao, D. Ding, and Z. Erickson, “Incremental learning for robot shared autonomy,” arXiv, 2025.

[14] A. Jonnavittula, S. A. Mehta, and D. P. Losey, “SARI: Shared autonomy across repeated interaction,” ACM Transactions on Human-Robot Interaction, 2024.

[15] M. Zurek, A. Bobu, D. S. Brown, and A. D. Dragan, “Situational confidence assistance for lifelong shared autonomy,” International Conference on Robotics and Automation, 2021.

[16] A. Mandlekar, D. Xu, R. Mart´ın-Mart´ın, Y. Zhu, L. Fei-Fei, and S. Savarese, “Human-in-the-loop imitation learning using remote teleoperation,” arXiv, 2020.

[17] H. Liu, S. Nasiriany, L. Zhang, Z. Bao, and Y. Zhu, “Robot learning on the job: Human-in-the-loop autonomy and learning during deployment,” The International Journal of Robotics Research, 2025.

[18] Y. Jiang, C. Wang, R. Zhang, J. Wu, and L. Fei-Fei, “Transic: Sim-toreal policy transfer by learning from online correction,” in Conference on Robot Learning, 2024.

[19] X. Xu, Y. Hou, Z. Liu, and S. Song, “Compliant residual DAgger: Improving real-world contact-rich manipulation with human corrections,” in Conference on Neural Information Processing Systems, 2025.

[20] R. Hoque, A. Balakrishna, C. Putterman, M. Luo, D. S. Brown, D. Seita, B. Thananjeyan, E. Novoseller, and K. Goldberg, “LazyDAgger: Reducing context switching in interactive imitation learning,” in International Conference on Automation Science and Engineering. IEEE Press, 2021.

[21] R. Hoque, A. Balakrishna, E. Novoseller, A. Wilcox, D. S. Brown, and K. Goldberg, “ThriftyDAgger: Budget-aware novelty and risk gating for interactive imitation learning,” in Conference on Robot Learning, 2022.

[22] Z. Xu, Y. Zhao, K. Wu, N. Liu, J. Ji, Z. Che, C. H. Liu, and J. Tang, “HACTS: a human-as-copilot teleoperation system for robot learning,” International Conference on Intelligent Robots and Systems, 2025.

[23] T. Takahashi, Y. Ishida, T. Kanai, and N. Kuppuswamy, “CHG-DAgger: Interactive imitation learning with human-policy cooperative control,” in CoRL Workshop CoRoboLearn, 2024.

[24] F. Abi-Farraj, T. Osa, N. P. J. Peters, G. Neumann, and P. R. Giordano, “A learning-based shared control architecture for interactive task execution,” in International Conference on Robotics and Automation, 2017.

[25] Y. Cui, Y. Zhang, L. Tao, Y. Li, X. Yi, and Z. Li, “End-to-end dexterous arm-hand vla policies via shared autonomy,” arXiv, 2025.

[26] Z. Li, L. Huang, W. Xu, Z. Zhu, N. Lin, X. Ma, X. Sheng, and R. Wen, “Hand-in-the-loop: Improving vla policies for dexterous manipulation via seamless hand-arm intervention,” arXiv, 2026.

[27] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, 2024.

[28] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[29] A. Abou Allaban, V. Dimitrov, and T. Padır, “A blended human-robot shared control framework to handle drift and latency,” in International Symposium on Safety, Security, and Rescue Robotics, 2019.

[30] M. A. Collier, R. Narayan, and H. Admoni, “The sense of agency in assistive robotics using shared autonomy,” in ACM/IEEE International Conference on Human-Robot Interaction, 2025.

[31] Y. Sun, Y. Ming, X. Zhu, and Y. Li, “Out-of-distribution detection with deep nearest neighbors,” in International conference on machine learning. PMLR, 2022.

[32] K. Muelling, A. Venkatraman, J.-S. Valois, J. Downey, J. Weiss, S. Javdani, M. Hebert, A. Schwartz, J. Collinger, and A. Bagnell, “Autonomy infused teleoperation with application to bci manipulation,” in Robotics: Science and Systems XI, 2015.

[33] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu, “Robocasa: Large-scale simulation of everyday tasks for generalist robots,” in Robotics: Science and Systems, 2024.

[34] S. G. Hart and L. E. Staveland, “Development of nasa-tlx (task load index): Results of empirical and theoretical research,” in Advances in psychology. Elsevier, 1988.

[35] R. D. Morey, J. N. Rouder, and T. Jamil, “Package ‘bayesfactor’,” 2015.

[36] M. D. Lee and E.-J. Wagenmakers, Bayesian Cognitive Modeling: A Practical Course. Cambridge University Press, 2014.

[37] F. Angiulli and C. Pizzuti, “Fast outlier detection in high dimensional spaces,” in European Conference on Principles of Data Mining and Knowledge Discovery, 2002.

[38] T. Eiter and H. Mannila, “Computing discrete frechet distance,” 1994.´ [Online]. Available: https://api.semanticscholar.org/CorpusID:16010565

[39] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Conference on Robot Learning, 2021.

[40] M. Sakr, J. Zhang, H. F. M. V. d. Loos, D. Kulic, and E. Croft,´ “Consistency matters: Defining demonstration data quality metrics in robot learning from demonstration,” ACM Transactions on Human-Robot Interaction, 2025.