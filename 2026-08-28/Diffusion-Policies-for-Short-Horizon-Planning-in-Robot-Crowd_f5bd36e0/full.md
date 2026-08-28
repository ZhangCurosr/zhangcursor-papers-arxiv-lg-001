# Diffusion Policies for Short-Horizon Planning in Robot Crowd Navigation

Wendong Li Jochen Garcke

Institute for Numerical Simulation

University of Bonn

Bonn, Germany

Abstract: Robot crowd navigation requires safe and efficient decision-making under dense, dynamic, and multimodal human–robot interactions. Existing reinforcement-learning methods typically output a single reactive action at each timestep, which limits their ability to represent diverse short-term avoidance strategies. We propose Planning Diffusion Policy Optimization (PDPO), an offline-to-online reinforcement-learning framework that uses a diffusion policy to generate short-horizon action chunks for crowd navigation. PDPO is first pretrained on collision-avoidance demonstrations and then fine-tuned online with PPO by treating the denoising process as an internal decision process. During execution, the policy generates a five-step action chunk and applies it in a receding-horizon manner. Furthermore, we observe an evaluation artifact in common crowd-navigation benchmarks: without explicit boundary constraints, learned agents may leave the valid domain and bypass dense crowds. To address this, we introduce a setting in which boundary violations are treated as collisions. Experiments show that PDPO obtains an improved success rate over strong baselines, and ablations demonstrate that action chunks are especially important for the modified bounded benchmark.

Keywords: Crowd Navigation, Reinforcement Learning, Diffusion Policy

## 1 Introduction

Robot crowd navigation is a fundamental problem in robotics, where a robot must reach its goal safely and efficiently while moving among dense, dynamic, and partially observable crowds [1, 2]. The difficulty of this problem lies not only in collision avoidance, but also in the multi-modal nature of human–robot interactions. When approaching pedestrians, multiple maneuvers may be simultaneously feasible, such as passing on different sides, slowing down to yield, or moving through a temporary gap. Effective navigation therefore requires both representing diverse local decisions and maintaining temporal consistency over the next few steps.

Classical rule-based methods, such as ORCA and Social Force, compute collision-avoidance actions from hand-designed local interaction rules [3, 4]. Although these methods are efficient and widely used, they are often reactive and can become overly conservative in dense crowds, leading to freezing behaviors [5]. Learning-based methods, especially deep reinforcement learning, improve adaptability by optimizing long-term returns from interaction experience [2, 6, 7, 8, 9]. However, many reinforcement-learning-based navigation policies implicitly adopt a single-step action formulation, where the policy predicts one action per timestep and is often parameterized by a simple unimodal distribution such as a Gaussian. Such a representation is restrictive for crowd navigation, where different avoidance strategies can be valid under the same observation. Moreover, while future outcomes are considered implicitly through accumulated rewards or value estimation, the policy itself usually does not explicitly represent a short-horizon action plan.

To address these limitations, we introduce Planning Diffusion Policy Optimization (PDPO), an offline-to-online reinforcement learning framework that brings diffusion-policy-based reinforcement learning to crowd navigation. Although diffusion policies and action chunking have been studied in other control settings [10, 11], dense crowd navigation poses distinct challenges beyond structured manipulation-style control, due to dynamic multi-agent interactions and safety-critical collision avoidance. PDPO addresses these challenges by generating a short-horizon action chunk conditioned on the current crowd observation, which serves as a local motion plan rather than only a sequence of low-level motor commands. Following a receding-horizon scheme, only the first action is executed before replanning at the next timestep. This design enables the policy to capture multi-modal navigation choices while promoting temporally coherent motion in dynamic crowds.

Following the common two-stage training paradigm of diffusion policies [10, 11, 12], PDPO first pretrains the diffusion policy with behavioral cloning on demonstration trajectories with effective collision-avoidance behaviors, providing a reasonable initialization for navigation. It then fine-tunes the policy online with reinforcement learning to improve task performance beyond the demonstrations and adapt to the interaction distribution induced by its own decisions. Together, PDPO provides a practical and effective diffusion-based planning framework for dense crowd navigation.

In addition to policy design, we find that the evaluation protocol itself can substantially affect crowdnavigation results. In commonly used simulation benchmarks [7], the lack of explicit boundary constraints can allow a robot to leave the valid navigation region and bypass dense crowds, creating unintended shortcut behaviors. Such artifacts may lead to overly optimistic evaluations and obscure the true crowd-awareness of learned policies. To obtain a more faithful evaluation, we introduce a bounded benchmark setting in which leaving the workspace is treated as a collision, and we compare policies under both the original and corrected environments.

The main contributions of this work are threefold:

Diffusion policies for crowd navigation. We introduce a diffusion-policy formulation for robot crowd navigation, enabling the policy to represent non-Gaussian and multimodal avoidance behaviors that are difficult to capture with commonly used single-step unimodal Gaussian policies.

Action-chunk-based short-horizon planning. Instead of producing a single reactive action, PDPO generates a short sequence of velocity commands and executes the first action in a receding-horizon manner. This action-chunk representation provides a local motion plan while preserving responsiveness to dynamic crowds.

Bounded domain as a modified benchmark. We identify an evaluation artifact in commonly used crowd-navigation experiments: without explicit workspace boundary constraints, RL agents may leave the nominal navigation region and thereby bypass dense interactions. We introduce a bounded CrowdNav benchmark and observe that in this modified setting the performance substantially differs.

The code and benchmark setup will be made available on publication.

## 2 Related Work

Rule-based Crowd Navigation. Traditional crowd navigation largely relies on rule-based methods. Notable examples include ORCA [3], which enforces collision avoidance via geometric constraints, and the Social Force model [4], which simulates pedestrian motion using attraction–repulsion forces. These methods compute actions directly from observations based on predefined rules, without learning from data. As a result, they are reactive and local, often performing poorly in dense crowds where conservative behavior may stall the robot.

Learning-based Methods. Early learning-based crowd navigation relied on deep value-based RL, estimating state or state–action values to guide decisions [2, 9, 13, 8]. While effective in many benchmark settings, value-based methods are less direct for continuous action control and complex human–robot interactions. Policy optimization approaches, such as PPO-based crowd-navigation policies, have therefore become widely used due to their compatibility with continuous actions and neural network function approximation [6, 7, 14]. However, these methods typically optimize policies at the single-action level, so temporally coherent short-horizon motion plans are not explicitly represented but are instead encouraged indirectly through value estimation, rollout-based returns, or reward design.

More recent works such as CrowdNav++ [7] incorporate human trajectory prediction into reward design to improve safety and social compliance. These methods improve navigation performance, but they typically retain a single-step policy formulation and often implicitly rely on unimodal Gaussian action parameterizations. As a result, they do not explicitly model a distribution over short-horizon action sequences, which is the focus of our work. Zhang et al. [15] further improved safety under prediction errors through conformal uncertainty estimation and constrained policy optimization; their focus, however, remains on safe policy optimization rather than explicit short-horizon action sequence modeling.

Feature extraction architectures have also evolved: from MLP-based pooling in SARL [2], to spatiotemporal graphs with RNNs in DSRNN [6, 8], and further to self-attention with implicit temporal encoding in CrowdNav++ [7]. In contrast to these architecture-focused improvements, our work focuses on the policy representation itself, using diffusion-based action chunks to model multi modal short-horizon navigation behaviors.

Diffusion Policies. Recent advances in diffusion models, particularly denoising diffusion probabilistic models (DDPMs) [12], have motivated their use as expressive policy classes for decision making and robotic control. In robotic imitation learning, diffusion policies have been used to model complex action distributions and generate multi-step action sequences through iterative denoising [11]. Recent work further extends diffusion policies to reinforcement learning by treating the denoising process as an MDP and optimizing it with policy-gradient methods such as PPO [10, 14]. These methods commonly follow a two-stage pipeline, where the policy is first pretrained from demonstrations and then refined with goal-conditioned or reward-based fine-tuning [10, 11]. Building on this paradigm, PDPO adapts diffusion-based short-horizon action generation to dense crowd navigation, enabling temporally coherent local planning under dynamic multi-agent interactions.

## 3 Preliminaries

## 3.1 Problem Formulation

As in prior learning-based crowd-navigation work [2, 7], crowd navigation is formulated as a Markov Decision Process (MDP) $( S , { \mathcal { A } } , { \mathcal { P } } _ { 0 } , { \mathcal { P } } , { \mathcal { R } } )$ . At timestep t, the robot observes its own state $w ^ { t }$ , including its position, velocity, goal, maximum speed, orientation, and radius. For each observable human i, the observation includes the current position $u _ { i } ^ { t }$ and predicted future positions $\hat { u } _ { i } ^ { t + 1 : t + L }$ over a short horizon L. The state is represented as

$$
\boldsymbol s _ { t } = [ w ^ { t } , h _ { 1 } ^ { t } , \dots , h _ { n _ { t } } ^ { t } ] , \quad h _ { i } ^ { t } = [ u _ { i } ^ { t } , \hat { u } _ { i } ^ { t + 1 : t + L } ] ,\tag{1}
$$

where $n _ { t }$ is the number of humans observed at timestep t.

The robot action is a two-dimensional velocity command $a _ { t } \ \in \ \mathbb { R } ^ { 2 }$ . At each timestep, the robot receives reward $r _ { t } = \mathcal { R } ( s _ { t } , a _ { t } )$ and transitions to $s _ { t + 1 } \sim \mathcal { P } ( \cdot \mid s _ { t } , a _ { t } )$ . Humans follow a fixed policy that is unknown to the robot. An episode terminates when the robot reaches its goal, collides with a human or boundary, or exceeds the maximum episode length.

Following CrowdNav++ [7], we largely adopt its reward design and use a constant-velocity predictor for short-horizon human position prediction. Except for boundary handling in the bounded CrowdNav, these components are fixed across learning-based methods to isolate the effect of policy representation; details are provided in Appendix B.

## 3.2 Diffusion Policy Background

Most policy-gradient methods for continuous control parameterize the policy as a Gaussian distribution,

$$
\pi _ { \boldsymbol { \theta } } ( a \mid s ) = \mathcal { N } \big ( a ; \mu _ { \boldsymbol { \theta } } ( s ) , \Sigma _ { \boldsymbol { \theta } } ( s ) \big ) ,\tag{2}
$$

where $\mu _ { \boldsymbol { \theta } } ( s )$ and $\Sigma _ { \theta } ( s )$ are predicted by a neural network. This formulation provides an explicit likelihood and is convenient for PPO-style policy optimization [14], but restricts the conditional action distribution to a unimodal family, which can be limiting in tasks where multiple actions or action sequences may be feasible under the same observation.

A diffusion policy instead defines the conditional distribution implicitly through an iterative denoising process [12]. Let $x _ { 0 }$ denote the policy output, which may be either a single action or a sequence of actions. Starting from Gaussian noise $x _ { K } \sim \mathcal { N } ( 0 , I )$ , the policy applies K reverse denoising transitions,

$$
p _ { \theta } ( x _ { k - 1 } \mid x _ { k } , s ) = \mathcal { N } \big ( x _ { k - 1 } ; \mu _ { \theta } ( x _ { k } , s , k ) , \sigma _ { k } ^ { 2 } I \big ) , \quad k = K , \ldots , 1 .\tag{3}
$$

Equivalently, the mean $\mu _ { \theta }$ can be parameterized through a noise-prediction network $\varepsilon _ { \boldsymbol { \theta } } ( x _ { k } , s , k )$ Through this iterative denoising process, the induced policy $\pi _ { \theta } ( x _ { 0 } \mid s )$ can represent complex, non-Gaussian, and multi-modal action distributions [11]. In PDPO, we instantiate $x _ { 0 }$ as a short action chunk and optimize the diffusion policy with behavioral cloning following online RL [10].

## 4 Planning Diffusion Policy Optimization

PDPO is an offline-to-online reinforcement learning framework that uses a diffusion policy to generate short-horizon action chunks for crowd navigation. At each timestep, the robot encodes the current crowd observation $s _ { t }$ into a latent representation $z _ { t } ,$ , samples a five-step action chunk conditioned on $z _ { t } ,$ executes the first action, and replans at the next timestep. The diffusion policy is first pretrained from collision-avoidance demonstrations and then fine-tuned online with task rewards.

Crowd Observation Encoding. Our crowd observation encoder is adapted from prior crowdnavigation architectures [7, 8]. It maps the crowd observation $s _ { t }$ to a latent representation $z _ { t }$ that conditions the diffusion policy. The encoder uses MLP layers for human-feature embedding, masked self-attention over observable humans, and attention pooling to aggregate human–robot interaction features. A visibility mask excludes unobservable humans from the attention computation. Architecture details and hyperparameters are provided in Appendix B.

Diffusion Action Chunk Policy. Given the latent observation $z _ { t } ,$ , PDPO samples a five-step action chunk

$$
\boldsymbol { x } _ { 0 } ^ { t } = \left[ x _ { 0 , 1 } ^ { t } , x _ { 0 , 2 } ^ { t } , x _ { 0 , 3 } ^ { t } , x _ { 0 , 4 } ^ { t } , x _ { 0 , 5 } ^ { t } \right] ,\tag{4}
$$

where each $x _ { 0 , j } ^ { t } \in \mathbb { R } ^ { 2 }$ is a velocity command. The chunk is generated by the diffusion policy introduced in Sec. 3.2, conditioned on $z _ { t }$

The action chunk provides an explicit short-horizon representation of local motion. In crowd navigation, meaningful avoidance behaviors are often defined over several consecutive actions rather than a single velocity command, such as passing a pedestrian on one side, slowing down to yield, or moving through a temporary gap. Modeling a distribution over action chunks allows the policy to capture multi-modal motion patterns at the sequence level while encouraging consistency among consecutive actions within the same local plan.

During execution, only the first action is applied to the environment:

$$
a _ { t } = x _ { 0 , 1 } ^ { t } .\tag{5}
$$

At the next timestep, the robot receives a new observation and samples a new action chunk. This receding-horizon execution combines short-horizon planning with closed-loop feedback: the chunk provides a temporally coherent local motion pattern, while replanning keeps the robot responsive to changes in the crowd.

Offline Demonstration Pretraining. Training a diffusion policy directly from online reinforcement learning is challenging in crowd navigation, since the policy must learn a multi-step denoising process over action chunks while exploring safely in dense dynamic environments. We therefore first pretrain the diffusion policy with behavioral cloning on collision-avoidance demonstrations, following prior diffusion-policy methods that initialize action-generation models from demonstration data before downstream deployment or optimization [10, 11].

Demonstration trajectories are collected in the crowd simulation environment by rolling out ORCA. At each timestep, we record the observation and executed velocity command, and construct an expert action chunk from the velocity commands at the current timestep and the following $H - 1$ timesteps. We use $H = 5$ in our experiments.

The diffusion policy is pretrained with the standard denoising objective. Given an expert chunk $\boldsymbol { x } _ { 0 } ^ { t } ,$ we sample a diffusion step k and Gaussian noise ϵ, perturb the chunk into $\ v x _ { k } ^ { t }$ , and train the denoising network to predict the injected noise:

$$
\mathcal { L } _ { \mathrm { B C } } ( \theta ) = \mathbb { E } _ { t , k , \epsilon } \left[ \left| \left| \epsilon - \epsilon _ { \theta } ( x _ { k } ^ { t } , z _ { t } , k ) \right| \right| _ { 2 } ^ { 2 } \right] .\tag{6}
$$

This stage provides a reasonable initialization with feasible collision-avoidance behaviors, which is then refined through online reinforcement learning.

Online Reinforcement Learning Fine-tuning. After pretraining, we fine-tune the diffusion policy online with PPO under task rewards, following recent diffusion-policy optimization approaches [10]. The reverse denoising process is treated as an internal MDP: during rollout collection, we store denoising transitions and their log probabilities for the generated chunk. After the first action of the chunk is executed in the environment, the resulting reward and advantage are assigned to the denoising transitions that produced the chunk.

The policy is then updated with the PPO objective at the denoising-step level. The full augmented-MDP formulation, advantage assignment, and implementation details are provided in Appendix A.

## 5 Experiment

## 5.1 Simulation Environment

All experiments are conducted in the CrowdNav++ simulator [7], a recent and challenging benchmark for robot crowd navigation that evaluates policies in dense, dynamic human crowds. The environment is a $1 2 \times 1 2 \mathrm { m } ^ { 2 }$ two-dimensional workspace containing one robot and 20 human agents. The task requires the robot to reach its goal while avoiding collisions and maintaining socially acceptable navigation behavior in dense crowds.

At the beginning of each episode, the robot and all humans are initialized with random positions and goals. Human agents follow ORCA for collision avoidance among humans, but they do not observe or react to the robot. The ORCA policy used by humans is not accessible to the robot during training or execution. The robot is modeled as a circular agent with radius 0.2 m, maximum speed 1 ${ \mathrm { m / s } } ,$ and sensing range 6 m. Human radii are uniformly sampled from [0.3, 0.5] m, and their maximum speeds are sampled from [0.5, 1.5] m/s.

To generate dynamic crowd flows, human agents are assigned new goal locations upon reaching their current goals. Additionally, at each second, each human independently changes its goal with probability 0.5. Since humans do not react to the robot, the robot cannot exploit cooperative human responses through overly aggressive navigation.

Each episode terminates when the robot reaches its goal, collides with a human, or exceeds the maximum episode length of 50 seconds. Episodes that reach the time limit are treated as timeouts.

Bounded CrowdNav setting. We observe that the original CrowdNav++ benchmark [7] does not enforce explicit boundary constraints, allowing the robot to leave the workspace region and thereby bypass dense crowds. This artifact may lead to overly optimistic evaluation of learned policies. To address this, we introduce boundary constraints by adding walls around the environment. While using the same crowd-generation process, robot dynamics, and evaluation metrics, boundary violations are now treated as collisions. We refer to this modified environment as Bounded CrowdNav.

## 5.2 Experimental Setup and Training

Baselines and ablations. We compare Planning Diffusion Policy Optimization (PDPO) with two representative crowd-navigation baselines:

• ORCA: a rule-based collision-avoidance planner, also used to generate demonstration data for imitation pretraining [3].

• CrowdNav++ (Const. Vel.): a strong learning-based crowd-navigation baseline using a constant-velocity human trajectory predictor [7].

To isolate the effect of action chunking, we include one ablation:

• PDPO w/o action chunk: a diffusion-policy variant that generates a single action per timestep instead of a multi-step action chunk.

Unless noted otherwise, all learning-based methods use the same constant-velocity predictor to forecast human motion over the next five timesteps.

Training details. PDPO and its ablation variants are trained with the same two-stage offline-toonline procedure. First, the diffusion policy is pretrained via behavioral cloning using approximately 3,000 demonstration episodes collected in simulation. The pretraining stage runs for 200 optimization iterations. The policy is then fine-tuned online for around 12 million environment timesteps. We use the same training budget for PDPO and its ablations. PDPO uses chunk length H = 5, corresponding to a 1.25 s planning horizon under the 0.25 s control interval.

Crowd-navigation benchmarks are sensitive to random seeds. Following CrowdNav++ [7], we use separate random seed sets for training, validation, and testing. Each final policy is evaluated on 500 unseen test seeds, and results are averaged over these episodes.

Training under two different benchmark settings. The original benchmark follows the standard CrowdNav++ protocol without explicit boundary constraints, allowing comparison with previously reported results. We use Bounded CrowdNav as the main controlled evaluation setting because it removes shortcut behaviors caused by leaving the valid workspace. For the original benchmark, we report ORCA and CrowdNav++ results from the CrowdNav++ paper [7], since the environment and evaluation protocol are unchanged. For Bounded CrowdNav, all learning-based methods, including PDPO, its ablation, and CrowdNav++, are trained and evaluated under the same bounded environment.

Evaluation metrics. We evaluate navigation performance using standard crowd-navigation metrics that measure efficiency, safety, and social compliance: success rate (SR, %), navigation time (NT, s), path length (PL, m), intrusion rate (ITR, %), and social distance (SD, m). ITR measures how often the robot enters human personal-space regions, while SD measures the average closest social distance during navigation.

## 5.3 Results on the Original Benchmark

Table 1 reports performance on the original CrowdNav++ benchmark. PDPO achieves a success rate of 90.6%, outperforming the reported CrowdNav++ result by 3.6 percentage points. The singleaction diffusion-policy variant performs comparably to CrowdNav++, while action chunking improves the success rate from 86.2% to 90.6%. These results suggest that diffusion-based action chunks can improve navigation performance even under the original benchmark protocol.

Table 1: Navigation performance on the original CrowdNav++ benchmark without explicit boundary constraints. ORCA and CrowdNav++ results are reported from [7].
<table><tr><td>Method</td><td>SR↑</td><td>NT↓</td><td>PL↓</td><td>ITR↓</td><td>SD↑</td></tr><tr><td>ORCA CrowdNav++ (Const. Vel.)</td><td>69.0 87.0</td><td>14.77 14.03</td><td>17.67 20.14</td><td>19.61 7.00</td><td>0.38</td></tr><tr><td>PDPO w/o action chunk</td><td>86.2</td><td>14.59</td><td>19.20</td><td></td><td>0.42</td></tr><tr><td>PDPO (Ours)</td><td>90.6</td><td>14.50</td><td>20.25</td><td>8.04 7.22</td><td>0.40 0.41</td></tr></table>

To quantify the effect of missing boundary constraints, we further analyze successful trajectories under the original benchmark and measure the fraction in which the robot leaves the nominal workspace. Using the same evaluation scenes, we find that 16.1% of successful CrowdNav++ trajectories, 31.8% of successful ORCA trajectories, 18.7% of successful PDPO trajectories without action chunks, and 19.8% of successful PDPO trajectories go out of bounds while still being counted as successes. This shows that boundary handling can substantially affect evaluation outcomes, motivating Bounded CrowdNav as our main controlled setting.

## 5.4 Results on Bounded CrowdNav

Table 2: Navigation performance on Bounded CrowdNav. All learning-based methods are trained and evaluated under the same bounded environment, where learning-based methods are evaluated over three random seeds for training the model. ORCA is deterministic and reported without standard deviation. The standard deviation for SD is negligible for all approaches.
<table><tr><td>Method</td><td>SR↑</td><td>NT↓</td><td>PL↓</td><td>ITR↓</td><td>SD↑</td></tr><tr><td>ORCA</td><td>47.0</td><td>21.26</td><td>17.14</td><td>1.35</td><td>0.50</td></tr><tr><td>CrowdNav++ (Const. Vel.)</td><td> $7 4 . 5 \pm 2 . 1$ </td><td> $1 3 . 3 5 \pm 1 . 1 3$ </td><td> $1 8 . 4 2 \pm 0 . 6 5$ </td><td> $8 . 5 3 \pm 0 . 3 0$ </td><td>0.42</td></tr><tr><td>PDPO w/o action chunk</td><td> $7 3 . 7 \pm 0 . 7$ </td><td> $1 3 . 0 5 \pm 0 . 4 6$ </td><td> $2 0 . 9 3 \pm 0 . 5 0$ </td><td> $7 . 2 1 \pm 0 . 7 4$ </td><td>0.40</td></tr><tr><td>PDPO (Ours)</td><td> $8 4 . 7 \pm 1 . 4$ </td><td> $1 0 . 9 2 \pm 0 . 0 4$ </td><td> $2 0 . 5 8 \pm 0 . 1 9$ </td><td> $6 . 8 1 \pm 0 . 0 4$ </td><td>0.41</td></tr></table>

Table 2 reports results on Bounded CrowdNav. PDPO achieves a success rate of 84.7%, outperforming the retrained CrowdNav++ by 11.7 percentage points and the single-action diffusion-policy ablation by 11.0 percentage points on average. This indicates that PDPO’s improvement is not merely due to the diffusion-policy parameterization, but largely comes from generating short-horizon action chunks.

The bounded setting also exposes the impact of boundary handling: CrowdNav++ drops from its reported 87.0% success rate in the original benchmark to around 74.5% under boundary constraints. PDPO also achieves the lowest intrusion rate, reducing ITR from 8.53% for CrowdNav++ to 6.81%. Although its path length is longer than that of CrowdNav++, which is a reasonable trade-off in dense crowds, where safer trajectories may require detours, the navigation time is shorter. Together, these results show that action-chunk diffusion policies improve both task success and crowd-aware safety.

## 5.5 Visualization of Action Chunks

We provide qualitative visualizations to illustrate how PDPO uses action chunks for short-horizon planning in dense crowds. At each decision step, the policy generates a 5-step action chunk with a control interval of 0.25 s, corresponding to a 1.25 s local plan. Only the first action is executed before replanning. Figure 1 shows representative snapshots from a dense navigation episode. As nearby pedestrians enter the robot’s observation range, the generated chunks adjust toward available gaps while maintaining short-horizon motion consistency. Around step 64, the chunk exhibits a local turning behavior in response to surrounding pedestrians. By step 68, after the robot moves away from the densest interaction region, the planned motion becomes longer again, indicating faster progress toward the goal.

![](images/0e3b5db295d442c3cbc5e201eef0a5aa18bb2a388f7185e4458074d7202ed3d6.jpg)  
Step 62

![](images/fd327cf93ff639486b77ae1e6cf711132e4c09bee05407fed327bcee23ae5c93.jpg)  
Step 64

![](images/a82e9f6be34a63fb1db2c22ec1a23c7a6df7d0822d76cd571b34b1bc4763ca7f.jpg)  
Step 68  
Figure 1: Representative snapshots of action chunks generated by PDPO during navigation through a dense crowd. In each snapshot, the dashed circle denotes the robot’s observation range, blue dots denote humans with predicted linear trajectories, the green dot denotes the robot, and the red cross the end of the action chunk. The generated chunks provide coherent short-horizon local plans that adapt to nearby pedestrians. Complete storyboards and multimodal samples are in Appendix C.

## 6 Limitations and Future Work

PDPO uses an iterative denoising process to generate action chunks, which introduces additional inference overhead compared to single-step policies. More efficient diffusion sampling strategies could reduce this overhead and make action-chunk generation more suitable for real-time scenarios.

PDPO currently follows a receding-horizon execution scheme by applying only the first action of each generated chunk. While this preserves responsiveness to dynamic crowds, future extensions could introduce trajectory-level objectives or consistency regularization to better supervise the entire action sequence.

So far, PDPO evaluation is performed in simulated scenarios, where human agents follow fixed policies and do not react to the robot. This enables controlled comparisons with prior crowd-navigation methods, but it remains a simulation-level correction and does not fully capture real human–robot interactions. Future work should evaluate PDPO in more diverse crowd scenarios, including realworld or higher-fidelity simulated environments.

## 7 Conclusion

In this work, we introduce Planning Diffusion Policy Optimization (PDPO) for robot crowd navigation. PDPO uses a diffusion policy to generate short-horizon action chunks and executes them in a receding-horizon manner, enabling the robot to represent diverse local motion choices while remaining responsive to dynamic crowd interactions.

Our results highlight the role of diffusion-based action chunking in dense crowd navigation. Across quantitative ablations and qualitative visualizations, PDPO benefits from representing navigation decisions as short-horizon action sequences rather than isolated single-step actions. The generated chunks form locally coherent motion patterns and produce diverse short-horizon proposals under the same observation, reflecting the multi-modal structure of crowd navigation.

In addition, our benchmark analysis shows that missing boundary constraints can affect policy eval uation, motivating the use of bounded environments for more faithful assessment of crowd-aware navigation behaviors. Overall, PDPO provides an effective diffusion-based short-horizon planning framework for dense crowd navigation and highlights the importance of careful benchmark design for learned navigation policies.

## References

[1] M. Everett, Y. F. Chen, and J. P. How. Motion planning among dynamic, decision-making agents with deep reinforcement learning. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2018, Madrid, Spain, October 1-5, 2018, pages 3052–3059. IEEE, 2018. doi:10.1109/IROS.2018.8593871.

[2] C. Chen, Y. Liu, S. Kreiss, and A. Alahi. Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning. In International Conference on Robotics and Automation, ICRA 2019, Montreal, QC, Canada, May 20-24, 2019, pages 6015– 6022. IEEE, 2019. doi:10.1109/ICRA.2019.8794134.

[3] J. van den Berg, S. J. Guy, M. Lin, and D. Manocha. Reciprocal n-body collision avoidance. In C. Pradalier, R. Siegwart, and G. Hirzinger, editors, Robotics Research: The 14th International Symposium ISRR, pages 3–19. Springer Berlin Heidelberg, Berlin, Heidelberg, 2011. doi: 10.1007/978-3-642-19457-3\_1.

[4] D. Helbing and P. Molnár. Social force model for pedestrian dynamics. Physical review. E, Statistical physics, plasmas, fluids, and related interdisciplinary topics, 51(5):4282–4286, 1995.

[5] P. Trautman and A. Krause. Unfreezing the robot: Navigation in dense, interacting crowds. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems, October 18-22, 2010, Taipei, Taiwan, pages 797–803. IEEE, 2010. doi:10.1109/IROS.2010.5654369.

[6] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell. Decentralized structuralrnn for robot crowd navigation with deep reinforcement learning. In IEEE International Conference on Robotics and Automation, ICRA 2021, Xi’an, China, May 30 - June 5, 2021, pages 3517–3524. IEEE, 2021. doi:10.1109/ICRA48506.2021.9561595.

[7] S. Liu, P. Chang, Z. Huang, N. Chakraborty, K. Hong, W. Liang, D. L. McPherson, J. Geng, and K. R. Driggs-Campbell. Intention aware robot crowd navigation with attention-based interaction graph. In IEEE International Conference on Robotics and Automation, ICRA 2023, London, UK, May 29 - June 2, 2023, pages 12015–12021. IEEE, 2023. doi:10.1109/ICRA48891. 2023.10160660.

[8] Y. Zhou and J. Garcke. Learning crowd behaviors in navigation with attention-based spatialtemporal graphs. In IEEE International Conference on Robotics and Automation, ICRA 2024, Yokohama, Japan, May 13-17, 2024, pages 5485–5491. IEEE, 2024. doi:10.1109/ICRA57147. 2024.10610279.

[9] C. Chen, S. Hu, P. Nikdel, G. Mori, and M. Savva. Relational graph learning for crowd navigation. In IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2020, Las Vegas, NV, USA, October 24, 2020 - January 24, 2021, pages 10007–10013. IEEE, 2020. doi:10.1109/IROS45743.2020.9340705.

[10] A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz. Diffusion policy policy optimization. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[11] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. Int. J. Robotics Res., 44(10-11):1684– 1704, 2025. doi:10.1177/02783649241273668.

[12] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[13] W. Shi, Y. Zhou, X. Zeng, S. Li, and M. Bennewitz. Enhanced spatial attention graph for motion planning in crowded, partially observable environments. In 2022 International Conference on Robotics and Automation, ICRA 2022, Philadelphia, PA, USA, May 23-27, 2022, pages 4750–4756. IEEE, 2022. doi:10.1109/ICRA46639.2022.9812322.

[14] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.

[15] J. Yao, X. Zhang, Y. Xia, Z. Wang, A. Roy-Chowdhury, and J. Li. Towards generalizable safety in crowd navigation via conformal uncertainty handling. In J. Lim, S. Song, and H.-W. Park, editors, Proceedings ofThe 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 4206–4225. PMLR, 2025. URL https://proceedings. mlr.press/v305/yao25a.html.

## A Diffusion Policy and PPO Fine-tuning Details

This section provides additional details on how the diffusion policy used in PDPO is fine-tuned with Proximal Policy Optimization (PPO). We follow the general view of diffusion policy optimization in which the denoising process is treated as an internal Markov decision process and optimized jointly with the environment MDP.

## A.1 Diffusion Policy as an Augmented MDP

Following diffusion-policy optimization [10], we view the reverse denoising process as an internal decision process and construct an augmented MDP over denoising transitions. This formulation allows the diffusion policy to be optimized with policy-gradient methods while keeping the environment-level MDP unchanged.

We consider the environment MDP introduced in Section 3.1,

$$
\mathcal { M } _ { \mathrm { e n v } } = ( \mathcal { S } , \mathcal { A } , \mathcal { P } _ { 0 } , \mathcal { P } , \mathcal { R } ) ,\tag{7}
$$

where $s _ { t } \in S$ denotes the crowd-navigation state at environment time $t ,$ and $a _ { t } \in \mathcal A$ is the robot action executed in the environment.

In PDPO, the policy does not output a single Gaussian action directly. Instead, it generates an action chunk

$$
\mathbf { x } _ { 0 } ^ { t } = \left( x _ { 0 , 1 } ^ { t } , x _ { 0 , 2 } ^ { t } , \ldots , x _ { 0 , H } ^ { t } \right) ,\tag{8}
$$

where H is the action chunk length and each $x _ { 0 , j } ^ { t } \in \mathbb { R } ^ { 2 }$ denotes the j-th velocity command in the chunk. In our experiments, we use $H = 5 ,$ , corresponding to a short-horizon plan of 1.25 s. During execution, we follow a receding-horizon control scheme: only the first action $a _ { t } = x _ { 0 , 1 } ^ { t }$ is applied to the environment, and a new action chunk is generated at the next environment timestep.

The action chunk is generated through a $K \cdot$ -step reverse diffusion process. Starting from Gaussian noise,

$$
\mathbf { x } _ { K } ^ { t } \sim \mathcal { N } ( 0 , I ) ,\tag{9}
$$

the policy iteratively denoises the action chunk according to

$$
\begin{array} { r } { p _ { \theta } ( \mathbf { x } _ { k - 1 } ^ { t } \mid \mathbf { x } _ { k } ^ { t } , s _ { t } ) = \mathcal { N } \left( \mathbf { x } _ { k - 1 } ^ { t } ; \mu _ { \theta } ( \mathbf { x } _ { k } ^ { t } , s _ { t } , k ) , \sigma _ { k } ^ { 2 } I \right) , \qquad k = K , \ldots , 1 , } \end{array}\tag{10}
$$

where $\mu _ { \theta }$ is parameterized by the diffusion noise prediction network $\varepsilon _ { \theta }$ . The final denoised sample $\mathbf { x } _ { 0 } ^ { t }$ is used as the policy output.

To optimize this policy with policy gradient methods, we construct an augmented diffusion MDP

$$
\bar { \mathcal { M } } = ( \bar { \mathcal { S } } , \bar { \mathcal { A } } , \bar { \mathcal { P } } _ { 0 } , \bar { \mathcal { P } } , \bar { \mathcal { R } } ) ,\tag{11}
$$

where each augmented timestep corresponds to one denoising transition inside an environment timestep. For environment time t and denoising index $k \in \{ K , K - 1 , \ldots , 1 \}$ , we define the augmented state and action as

$$
\begin{array} { r } { \bar { s } _ { t , k } = ( s _ { t } , \mathbf { x } _ { k } ^ { t } , k ) , \qquad \bar { a } _ { t , k } = \mathbf { x } _ { k - 1 } ^ { t } . } \end{array}\tag{12}
$$

The corresponding augmented policy is

$$
\begin{array} { r } { \pi _ { \theta } ( \bar { a } _ { t , k } \mid \bar { s } _ { t , k } ) = p _ { \theta } ( \mathbf { x } _ { k - 1 } ^ { t } \mid \mathbf { x } _ { k } ^ { t } , s _ { t } ) . } \end{array}\tag{13}
$$

Within an environment timestep, transitions are given by the learned reverse diffusion process. After the final denoising step $k = 1$ , the first action in the generated chunk, $a _ { t } = x _ { 0 , 1 } ^ { t }$ , is executed in the environment. The environment then transitions according to

$$
s _ { t + 1 } \sim \mathcal { P } ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) , \qquad a _ { t } = x _ { 0 , 1 } ^ { t } ,\tag{14}
$$

and the next denoising trajectory is initialized by sampling ${ \bf x } _ { K } ^ { t + 1 } \sim \mathcal { N } ( 0 , I )$

The environment reward is assigned only after the final denoising step:

$$
\bar { R } ( \bar { s } _ { t , k } , \bar { a } _ { t , k } ) = \left\{ \begin{array} { l l } { \mathcal { R } ( s _ { t } , a _ { t } ) , } & { k = 1 , } \\ { 0 , } & { k > 1 , } \end{array} \right. \quad \quad a _ { t } = x _ { 0 , 1 } ^ { t } .\tag{15}
$$

This construction allows the multi-step denoising procedure to be interpreted as a sequence of internal policy decisions whose final output determines the action executed in the crowd-navigation environment.

## A.2 PPO Fine-tuning over Denoising Steps

Following PPO [14] and its use for diffusion-policy optimization [10], we optimize the augmented denoising policy at the level of individual denoising transitions.

Since the augmented process defines a valid stochastic policy over denoising transitions, we can apply PPO to fine-tune the diffusion policy after behavioral cloning pretraining. For each denoising transition, we store

$$
\begin{array} { r } { { ( \bar { s } _ { t , k } , \bar { a } _ { t , k } , \log \bar { \pi } _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( \bar { a } _ { t , k } \mid \bar { s } _ { t , k } ) , \hat { A } _ { t , k } ) } , } \end{array}\tag{16}
$$

where $\theta _ { \mathrm { o l d } }$ denotes the policy parameters used to collect the rollout data, and $\hat { A } _ { t , k }$ is the advantage assigned to the denoising step.

The PPO probability ratio is computed at the denoising-step level:

$$
r _ { t , k } ( \theta ) = \frac { \bar { \pi } _ { \theta } ( \bar { a } _ { t , k } \mid \bar { s } _ { t , k } ) } { \bar { \pi } _ { \theta _ { \mathrm { o l d } } } ( \bar { a } _ { t , k } \mid \bar { s } _ { t , k } ) } .\tag{17}
$$

The clipped PPO objective is then

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P P O } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t , k } \left[ \operatorname* { m i n } \left( r _ { t , k } ( \boldsymbol { \theta } ) \hat { A } _ { t , k } , \operatorname { c l i p } \left( r _ { t , k } ( \boldsymbol { \theta } ) , 1 - \epsilon , 1 + \epsilon \right) \hat { A } _ { t , k } \right) \right] . } \end{array}\tag{18}
$$

In practice, we maximize this surrogate objective together with the standard value-function and entropy terms:

$$
\mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { { P P O } } } ( \theta ) - c _ { v } \mathcal { L } _ { \mathrm { { v a l u e } } } + c _ { e } \mathcal { H } ( \bar { \pi } _ { \theta } ) ,\tag{19}
$$

where $c _ { v }$ and $c _ { e }$ are weighting coefficients.

## A.3 Advantage Assignment for Diffusion Policies

Following diffusion-policy optimization [10], we assign environment-level advantages to the denoising transitions that produce the executed action.

The environment reward is observed only after the denoised action is executed. Therefore, the same environment-level return must be assigned to the denoising transitions that produced the executed action. We first compute an environment-level return

$$
G _ { t } = \sum _ { t ^ { \prime } \geq t } \gamma _ { \mathrm { e n v } } ^ { t ^ { \prime } - t } \mathcal { R } \big ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } \big ) ,\tag{20}
$$

where $a _ { t ^ { \prime } } = x _ { 0 , 1 } ^ { t ^ { \prime } }$ is the first action of the generated chunk at environment time $t ^ { \prime } ,$ and $\gamma _ { \mathrm { e n v } }$ is the environment discount factor. The value function is defined over the environment state only:

$$
V _ { \phi } ( s _ { t } ) \approx \mathbb { E } [ G _ { t } \mid s _ { t } ] .\tag{21}
$$

This gives the environment-level advantage

$$
\hat { A } _ { t } ^ { \mathrm { e n v } } = G _ { t } - V _ { \phi } ( s _ { t } ) .\tag{22}
$$

To distribute this advantage over the denoising steps, we use a denoising discount factor γ<sub>denoise</sub>. Earlier denoising steps operate on noisier action chunks and are therefore assigned smaller weights, while later denoising steps receive larger credit. Specifically, for denoising step $k \in \{ K , \ldots , 1 \}$ , we define

$$
\hat { A } _ { t , k } = \gamma _ { \mathrm { d e n o i s e } } ^ { k - 1 } \hat { A } _ { t } ^ { \mathrm { e n v } } .\tag{23}
$$

Thus, the final denoising transition $k = 1$ receives the full environment-level advantage, while earlier transitions are downweighted according to their distance from the final denoised action.

This advantage assignment provides a practical way to couple environment-level reinforcement learning with the internal denoising process of the diffusion policy. It allows PPO to update all denoising transitions involved in generating an executed action, while giving higher weight to denoising steps that are closer to the final action chunk.

## A.4 Behavioral Cloning Pretraining

Following diffusion-policy imitation learning [11] and offline-to-online diffusion-policy optimization [10], we pretrain the diffusion policy using behavioral cloning on demonstration trajectories before online PPO fine-tuning.

Given an expert action chunk $\mathbf { x } _ { 0 } ^ { t } ,$ we sample a diffusion step k and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , and construct the noisy action chunk

$$
\mathbf { x } _ { k } ^ { t } = \sqrt { \bar { \alpha } _ { k } } \mathbf { x } _ { 0 } ^ { t } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon .\tag{24}
$$

The denoising network is trained to predict the injected noise:

$$
\mathcal { L } _ { \mathrm { B C } } ( \theta ) = \mathbb { E } _ { t , k , \epsilon } \left[ \left| \left| \epsilon - \epsilon _ { \theta } ( \mathbf { x } _ { k } ^ { t } , s _ { t } , k ) \right| \right| _ { 2 } ^ { 2 } \right] .\tag{25}
$$

This pretraining stage provides a strong initialization that captures common collision-avoidance behaviors before online reinforcement learning.

## B Network Architecture and Hyperparameters

## B.1 Observation Encoder Architecture

Our observation encoder follows prior crowd-navigation architectures, including CrowdNav++ [7] and ASTG [8], which use neural interaction modeling to encode human–robot and human–human relationships. The observation encoder maps the current robot state and the predicted human states into a compact latent representation used to condition the diffusion policy. At timestep t, the input observation is

$$
s _ { t } = [ w ^ { t } , h _ { 1 } ^ { t } , \dots , h _ { n _ { t } } ^ { t } ] ,
$$

where $w ^ { t }$ denotes the robot state and $h _ { i } ^ { t }$ denotes the feature of the i-th observable human. Each human feature contains the current human position and its constant-velocity predicted positions over the next L timesteps:

$$
h _ { i } ^ { t } = [ u _ { i } ^ { t } , \hat { u } _ { i } ^ { t + 1 : t + L } ] .
$$

In all experiments, we use $L = 5$ . For batching, the number of humans is padded to a fixed maximum number, and a visibility mask is used to ignore padded or unobservable humans.

Human embedding and masked self-attention. Each human feature is first projected into a latent embedding using a multilayer perceptron:

$$
h _ { i } ^ { \mathrm { e m b } } = \mathrm { M L P } _ { \mathrm { h u m a n } } ( h _ { i } ^ { t } ) .
$$

To model interactions among nearby humans, we apply masked multi-head self-attention over the set of human embeddings:

$$
\widetilde { h } _ { 1 } , \ldots , \widetilde { h } _ { n _ { t } } = \mathrm { S e l f A t t n } \left( h _ { 1 } ^ { \mathrm { e m b } } , \ldots , h _ { n _ { t } } ^ { \mathrm { e m b } } \right) .
$$

The attention mask prevents padded or unobservable humans from contributing to the attention computation. Residual connections, layer normalization, and dropout are used after the attention layer to stabilize optimization. The resulting embeddings $\tilde { h } _ { i }$ encode interaction-aware human features.

Robot-conditioned human–robot features. To capture how each human relates to the robot, we concatenate the robot state with each interaction-aware human embedding and process the result with an edge MLP:

$$
\begin{array} { r } { e _ { i } = \mathrm { M L P } _ { \mathrm { e d g e } } ( [ \tilde { h } _ { i } , w ^ { t } ] ) . } \end{array}
$$

This produces a set of robot-conditioned human–robot interaction features $\{ e _ { i } \} _ { i = 1 } ^ { n _ { t } }$

Attention pooling over humans. Since the number of observable humans varies over time, we aggregate the interaction features using attention pooling. For each human–robot feature $e _ { i } ,$ , an attention score is computed as

$$
q _ { i } = \mathrm { M L P _ { a t t } } ( e _ { i } ) ,
$$

and normalized across observable humans:

$$
\alpha _ { i } = \frac { \exp ( q _ { i } ) } { \sum _ { j \in \mathcal { V } _ { t } } \exp ( q _ { j } ) } ,
$$

where $\nu _ { t }$ denotes the set of visible humans at timestep t. The pooled crowd representation is then

$$
e ^ { \mathrm { a t t } } = \sum _ { i \in \nu _ { t } } \alpha _ { i } e _ { i } .
$$

This attention-pooling mechanism allows the encoder to emphasize humans that are most relevant to the robot’s near-future motion.

Latent projection. Finally, the pooled crowd representation is concatenated with the robot state and passed through a latent projection MLP:

$$
z _ { t } = \mathrm { M L P } _ { \mathrm { l a t e n t } } ( [ w ^ { t } , e ^ { \mathrm { a t t } } ] ) .
$$

The latent vector $z _ { t }$ is used as the conditioning input for both the diffusion denoising network and the value function during online fine-tuning.

## B.2 Diffusion Policy and Value Function

The diffusion policy operates on an action chunk

$$
\boldsymbol { x } _ { 0 } ^ { t } = \left[ \boldsymbol { x } _ { 0 , 1 } ^ { t } , \boldsymbol { x } _ { 0 , 2 } ^ { t } , \boldsymbol { x } _ { 0 , 3 } ^ { t } , \ldots , \boldsymbol { x } _ { 0 , H } ^ { t } \right] ,
$$

where each action $x _ { 0 , j } ^ { t } \in \mathbb { R } ^ { 2 }$ is a velocity command. The noisy action chunk $\ v x _ { k } ^ { t }$ is flattened and concatenated with the observation latent vector $z _ { t }$ and a diffusion timestep embedding ψ(k). The timestep embedding is produced using sinusoidal positional encoding followed by a small MLP. The denoising network predicts the injected Gaussian noise:

$$
\hat { \epsilon } = \epsilon _ { \theta } ( x _ { k } ^ { t } , z _ { t } , k ) .
$$

The output dimension is 2H, matching the flattened action-chunk dimension. For the single-action ablation, we use the same architecture with $H = 1$

The value function shares the same observation encoder and predicts a scalar value from the latent representation:

$$
V _ { \phi } ( s _ { t } ) = \mathrm { M L P } _ { \mathrm { v a l u e } } ( z _ { t } ) .
$$

This value estimate is used to compute the environment-level advantage during PPO fine-tuning, which is then assigned to the denoising transitions as described in Appendix A.3.

## B.3 Hyperparameters

Table 3 summarizes the main hyperparameters used in our experiments.

Table 3: Environment settings, network architecture, diffusion-model settings, and training hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Workspace size Number of humans</td><td> $1 2 \times 1 2 \mathrm { m } ^ { 2 }$ </td></tr><tr><td>Robot radius</td><td>20 0.2 m</td></tr><tr><td>Robot maximum speed Robot sensing range</td><td> $1 . 0 \mathrm { m / s }$  6.0 m</td></tr><tr><td>Control interval</td><td>0.25 s</td></tr><tr><td>Episode time limit Human prediction horizon L</td><td>50 s</td></tr><tr><td></td><td>5</td></tr><tr><td>Observation dimension</td><td>269</td></tr><tr><td>Action dimension</td><td>2</td></tr><tr><td>Action chunk length H</td><td>5</td></tr><tr><td>Policy output dimension</td><td></td></tr><tr><td>Planning horizon</td><td> $2 H = 1 0$ </td></tr><tr><td>Demonstrations</td><td>1.25 s</td></tr><tr><td>BC optimization iterations</td><td>3,000 episodes</td></tr><tr><td>BC batch size</td><td>200</td></tr><tr><td>BC learning rate</td><td>128</td></tr><tr><td>BC weight decay</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Online fine-tuning budget</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Evaluation episodes</td><td>12M environment steps</td></tr><tr><td>Denoising network MLP</td><td>500 unseen test seeds</td></tr><tr><td>Diffusion timestep embedding dimension Denoising activation</td><td>[512, 256, 256] 16</td></tr><tr><td>Value network MLP</td><td>ReLU [256, 256, 256]</td></tr><tr><td>Value activation</td><td>Mish</td></tr><tr><td>Diffusion objective</td><td>noise prediction</td></tr><tr><td>Noise schedule</td><td></td></tr><tr><td>Diffusion pretraining steps</td><td>cosine beta schedule</td></tr><tr><td>Diffusion fine-tuning steps</td><td>20</td></tr><tr><td></td><td>10</td></tr><tr><td>BC optimizer</td><td></td></tr><tr><td>PPO optimizer</td><td>AdamW</td></tr><tr><td>PPO actor learning rate</td><td>AdamW</td></tr><tr><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>PPO critic learning rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Discount factor γ</td><td>0.99</td></tr><tr><td>GAE parameter λ</td><td>0.95</td></tr><tr><td>Denoising discount factor  $\gamma _ { \mathrm { d e n o i s e } }$ </td><td>0.99</td></tr><tr><td>PPO batch size</td><td>500</td></tr><tr><td>PPO update epochs</td><td>5</td></tr><tr><td>Value loss coefficient</td><td>0.5</td></tr><tr><td>Denoising policy loss clipping coefficient</td><td>0.01</td></tr><tr><td>Base denoising policy loss clipping coefficient</td><td></td></tr><tr><td></td><td>0.001</td></tr><tr><td>Denoising policy loss clipping growth rate</td><td>3</td></tr><tr><td>Target KL</td><td>1.0</td></tr></table>

The denoising policy loss clipping coefficient is used in the clipped diffusion-policy loss during online fine-tuning and is distinct from the standard PPO probability-ratio clipping threshold.

## B.4 Model Size

We additionally report the number of trainable parameters for learning-based methods. During online fine-tuning, PDPO updates 1.049M trainable parameters, including a 0.848M-parameter finetuned diffusion actor and a 0.201M-parameter critic. CrowdNav++ contains 2.503M trainable parameters under our implementation. Thus, PDPO uses fewer trainable parameters than the learningbased baseline while achieving higher success rate in Bounded CrowdNav.

Table 4: Trainable parameter count of learning-based methods.
<table><tr><td>Method</td><td>Trainable Parameters</td></tr><tr><td>CrowdNav++ (Const. Vel.)</td><td>2.503M</td></tr><tr><td>PDPO (Ours)</td><td>1.049M</td></tr></table>

## B.5 Training Curves

Figure 2 shows the online fine-tuning curves on Bounded CrowdNav. Compared with the singleaction diffusion-policy ablation, PDPO achieves higher and more stable training success rates, supporting the benefit of action-chunk generation during online fine-tuning. For the final test evaluation, we select the checkpoint that achieves the highest success rate on the validation episodes during training.

![](images/dbd535807ab7f04d539c0090bcba82cedadaa225b1fb5f2e2c2b7384ddc1140e.jpg)

![](images/366e6cfc26863589e39b65cad2e512093d86e39214f27499739cb4b135e58017.jpg)  
Figure 2: Online fine-tuning curves of PDPO and its ablation on Bounded CrowdNav over the common training horizon. Left: training success rate. Right: average training reward.

## C Additional Visualization of Action Chunks

This section provides additional qualitative visualizations of action chunks generated by PDPO in dense crowd-navigation scenarios. Each snapshot is centered on the robot: the dashed circle denotes the observation range, blue dots denote humans with predicted linear trajectories, the green dot denotes the robot, and the red cross denotes the end of the action chunk. At each decision step, PDPO generates a 5-step action chunk with a control interval of 0.25 s, corresponding to a 1.25 s short-horizon plan. Only the first action is executed before replanning at the next timestep.

Figure 3 shows an episode where the robot moves through a dense crowd toward a northwest goal. As pedestrians enter the observation range, the generated chunks adjust toward available gaps while maintaining clearance. Around steps 62–64, the plan changes smoothly to turn around approaching pedestrians. Steps 65–67 show shorter segments in the dense region, indicating more cautious motion; after the robot leaves this constrained area, the planned motion becomes longer again.

Figure 4 shows another dense-crowd episode. At each timestep, multiple action chunks are sampled from the learned diffusion policy under the same observation. These samples correspond to different feasible short-horizon choices, such as passing through different gaps or slowing down near pedestrians. This variability shows that PDPO represents multiple plausible local plans and replans as the crowd evolves.

![](images/0ac2f99002e35360ce2d4f72ece3afc5609476be79cce4591be22d56b58d38e5.jpg)  
Step 60

![](images/fcac8c1bdb8a04d11c5e718ef4a00ad002b7d26d8e301607b99b9663d5f76303.jpg)  
Step 61

![](images/8b7b5e00025903d0ff67dbc1fb11b124805d353997a53d65b52a84f4b16b8700.jpg)  
Step 62

![](images/fc7e6bebda35148610c81e7f2d076abcc68e76f0f137165af833e77276c5490b.jpg)  
Step 63

![](images/633e1ba85f2f24fb921374b91ceb6a794518f3ae6b98eb60f72400a1009931cc.jpg)

![](images/0a44867dadf89d86aa3ccaf8d42311c327a13163af90c27db3a626ea190f70c9.jpg)  
Step 65

![](images/d66df7a5a5668d8f245210d6b7ab554c84b687431d53b7b8ba13488ea8fe1413.jpg)  
Step 66

![](images/976a03abc96d9018dc85c607b14aa9bfb5bffd1b69c6b08b3e11641815979d14.jpg)  
Step 67

![](images/6fcd7e42639cd78cdecea15942b60abe79ee74a1e3673ce156a25b1812edeb93.jpg)  
Step 68  
Figure 3: Complete storyboard of a PDPO trajectory from step 60 to step 68. The generated action chunks remain coherent while adapting to nearby pedestrians.

![](images/fd656bfb92fd852b2953c6c40f3f780a7e179cf2a91e13050a66a6ebffdf0aff.jpg)  
Step 265

![](images/e68e6607bab34ac057918921fdde00d41be8121570d996bf78cc3cd426816004.jpg)  
Step 267

![](images/026624037282dea344dba3583b6c30802a2c6b96145f61a3f7d826eaf2cb5c83.jpg)  
Step 270

![](images/e86cab2a2caa5a05a410dda1986c3320ac7f3cb48a4e811ff718ae760244effe.jpg)  
Step 272

![](images/44d11fb35f1a10ddfb1793e52b97a133fa5a53d0b628f26fe21c5e38dbfcef64.jpg)  
Step 274

![](images/432d9c666d111c0c9e8ba7c26b2641d6497ef03123bb7c1072ff5219d2e1727f.jpg)  
Step 276

![](images/bdd4047e725c499fc84c7fc6927df4456ee2cc2214b5a5462dff74de641705b1.jpg)  
Step 278

![](images/0333a176937ac82ca555c60424ff62e6272d74afc0787874937a6199d81d95de.jpg)  
Step 279

![](images/d2d90be70601e94e957613c28eaa6f4b8203aa7f069f7fdddb27a4ea6d0129ad.jpg)  
Step 280  
Figure 4: Complete storyboard of multimodal action chunks generated by PDPO. Multiple sampled short-horizon trajectories are visualized under the same observation, illustrating diverse feasible local plans.