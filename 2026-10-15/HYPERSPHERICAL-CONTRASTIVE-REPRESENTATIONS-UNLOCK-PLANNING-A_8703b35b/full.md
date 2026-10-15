# HYPERSPHERICAL CONTRASTIVE REPRESENTATIONS UNLOCK PLANNING AND EXPLORATION

Jonghae Park<sup>1∗</sup> Sungil Son<sup>1∗</sup> Jonghun Shin<sup>1</sup> Junseo Lee<sup>1</sup> H. Jin Kim<sup>1</sup> Benjamin Eysenbach<sup>2</sup>

<sup>1</sup>Seoul National University <sup>2</sup>Princeton University {bdfire1234,aldridge99}@snu.ac.kr

## ABSTRACT

We develop a self-supervised reinforcement learning algorithm that learns to build towers of up to eight cubes from scratch, without prior data, rewards, demonstrations, or pretrained models. Our method, Contrastive Planning via Spherical Linear Interpolation (C-Slerp), uses contrastive representations as reusable building blocks and repeatedly composes them to infer a sequence of intermediate waypoints. Under our hyperspherical parameterization, the learned representations allow intermediate waypoints to be inferred in closed form through spherical linear interpolation, without training a separate high-level policy or performing graph search. Importantly, the representation geometry also tells us how to adjust the number of waypoints based on the difficulty of each goal-reaching problem. Experiments show that our method succeeds on long-horizon navigation and robotic manipulation tasks, including settings where strong baselines fail to achieve a single success. Ablations further show that using a fixed number of waypoints substantially degrades performance, highlighting the importance of adapting the plan to task difficulty.

Project page: https://jonghaepark.github.io/c-slerp

![](images/43fc5803e6fe63a4f18f56459ed8d2c5e0beaa74b02ed9f5e61d359db13210da.jpg)  
Initial state

![](images/20cb23a93a24530cb5443b0353c33fd5690105cf089ce998d1456bff61af78a0.jpg)  
Goal state

![](images/201390c1b0216bb0fdda790206e233197b46a4dbd8bdd0d3af8e41c083dadb5d.jpg)  
Env step (×10<sup>9</sup>)  
Figure 1: Our paper studies long-horizon, sparse-reward tasks, including eight-cube stacking. Given only the current and goal states, the agent aims to manipulate the eight cubes into the goal state without receiving any external reward along the way. Strong exploration methods (Pathak et al., 2017; Burda et al., 2018) fail to discover a single successful state in one billion environment steps.

## 1 INTRODUCTION

Consider a robot asked either to place a single cube or to build an eight-cube tower. The former can be solved directly, whereas the latter may benefit from a plan with several intermediate stages. Too many stages needlessly complicate the single-cube task, while too few would leave the towerbuilding task too difficult to solve. The number of planning stages should therefore match the difficulty of the problem.

In goal-conditioned reinforcement learning (Kaelbling, 1993), these stages can be defined by subgoals, or waypoints, that divide long-horizon tasks into shorter subproblems. However, existing hierarchical approaches often use a fixed number of waypoints (Nasiriany et al., 2019; Zhang et al., 2021b) or generate waypoints that are a fixed number of steps into the future (Nachum et al., 2018; Park et al., 2023), rather than explicitly selecting how many stages to plan for each problem.

A natural way to plan over arbitrary distances is to define a reusable component and compose it as many times as needed. Contrastive reinforcement learning (CRL) (Eysenbach et al., 2022) offers a promising foundation for this approach. CRL learns contrastive representations that encode temporal relationships between states under a single geometric distribution, providing natural building blocks for multi-stage planning. While these representations are primarily used to train goal-conditioned policies, we ask whether they can also be leveraged to infer an adaptive number of waypoints.

To this end, we introduce Contrastive Planning via Spherical Linear Interpolation (C-Slerp), a hierarchical approach where a high-level planner provides waypoints for a low-level policy to follow. Our main idea is to make this hierarchy adaptive by using contrastive representations as building blocks for each stage of a plan. By repeatedly composing these building blocks, we can score how well the state and goal are connected through different numbers of stages. Using these scores, we select an appropriate number of stages and infer the corresponding intermediate waypoints.

A key advantage of our contrastive representations is that their spherical geometry enables intermediate waypoints to be inferred in closed form without training a separate high-level policy or performing graph search. Specifically, our high-level policy introduces no additional network parameters — it simply performs spherical linear interpolation in the learned representation space.

In this paper, we show how choosing the correct amount of planning can unlock new capabilities for reinforcement learning. We evaluate our approach on long-horizon navigation and robotic manipu lation tasks, including settings in which strong exploration methods never achieve a single success in one billion environment steps (Figure 1, Appendix C). Notably, our method successfully solves challenging benchmark tasks without a goal curriculum or explicit dense rewards, even where all baselines fail completely. Our ablations further show that hierarchy alone is insufficient: fixing the number of planning stages substantially degrades performance. Together, these results suggest that a key benefit of hierarchy for goal-reaching may lie in selecting an appropriate amount of planning for each problem rather than in making the policy more expressive.

## 2 RELATED WORK

Hierarchical goal-conditioned reinforcement learning. Goal-conditioned reinforcement learning (GCRL) trains a single policy to reach diverse goals (Schaul et al., 2015; Andrychowicz et al., 2017). Despite recent progress, reaching temporally distant goals remains a challenging problem. Hierarchical approaches (Vezhnevets et al., 2017; Levy et al., 2017) address this challenge by decomposing long-horizon tasks into shorter subproblems. Existing methods learn high-level policies that generate subgoals for low-level controllers (Nachum et al., 2018; Zhang et al., 2020; Park et al., 2023) or infer waypoints to guide policy learning or exploration (Chane-Sane et al., 2021; Zhang et al., 2021b). However, many of these approaches use a fixed number of waypoints or a fixed temporal horizon for subgoals, rather than explicitly selecting how many waypoints to plan for each stategoal pair. Graph-based methods offer an alternative by searching over observed states or learned landmarks to construct waypoint sequences (Eysenbach et al., 2019; Zhang et al., 2021a). Although these methods generate a variable number of waypoints, they require additional computation fo graph construction and depend on the coverage of states included in the graph.

Understanding the benefits of hierarchy. Prior work shows that hierarchy can improve exploration through temporally extended behavior (Nachum et al., 2019) and improve statistical and computational efficiency by reusing solutions to repeated subproblems (Wen et al., 2020). For policy learning, HIQL (Park et al., 2023) shows that hierarchy provides clearer policy-learning signals, reducing sensitivity to value estimation errors. Further, SAW (Zhou & Kao, 2026) brings the benefits of hierarchy to flat policies through bootstrapping, without a separate subgoal generator. These findings suggest that the benefits of hierarchy are closely tied to how it structures exploration and policy learning, rather than simply to the expressivity gained from a hierarchical policy structure.

Contrastive representations for goal-reaching. Contrastive reinforcement learning (CRL) (Ey senbach et al., 2022) learns representations for goal-reaching in a self-supervised manner, without externally specified rewards. It formulates goal-conditioned RL as a contrastive learning problem, jointly learning policies and representations that encode temporal relationships between states. Subsequent work shows that CRL exhibits strong exploration capabilities even when data collection is directed toward a single fixed goal (Liu et al., 2025; Bastankhah et al., 2026), and that increasing the depth of actor and critic networks further improves its goal-reaching performance (Wang et al., 2026). Despite these advances, it remains unclear whether exploration capability and network scaling alone are sufficient for CRL to solve substantially longer-horizon tasks. Beyond policy learning, Eysenbach et al. (2024) derive closed-form distributions over waypoint representations using contrastive learning. However, their method learns representations from offline data without policy learning and requires a predefined number of waypoints, rather than adapting it to task difficulty.

Our method, C-Slerp, jointly learns contrastive representations and a goal-conditioned policy online. Using the learned representations, it adapts the number of waypoints to each state-goal pair and infers their representations in closed form, without a separate high-level policy or graph search. This allows us to examine how adapting the number of planning stages contributes to the benefits of hierarchy in long-horizon tasks.

## 3 PRELIMINARIES

We focus on fully online RL without offline data, where the replay buffer is continually updated during policy training and the critic is trained with a self-supervised objective without external rewards.

Goal-conditioned reinforcement learning (GCRL). We consider a goal-conditioned Markov decision process $\mathcal { M } _ { g } = ( \mathcal { S } , \mathcal { A } , p , p _ { 0 } , \gamma , p _ { g } , \bar { r } _ { g } )$ , where $s$ and $\mathcal { A }$ denote the state and action spaces, $p : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ is the transition kernel, $\dot { p } _ { 0 } \in \Delta ( \mathcal { S } )$ is the initial-state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. A goal $g$ is sampled from the distribution $p _ { g } \in \Delta ( \mathcal { G } )$ over the goal space ${ \mathcal { G } } .$ and a goal-conditioned policy $\pi : { \mathcal { S } } \times { \mathcal { G } } \to \Delta ( { \mathcal { A } } )$ selects an action $a \sim \pi ( \cdot \mid s , g )$ conditioned on the current state s and desired goal $g .$ . For the analysis, we define the goal-reaching reward as $r _ { g } ( s _ { t } , a _ { t } ) \triangleq ( 1 - \gamma ) p ( s _ { t + 1 } = g \ | \ s _ { t } , a _ { t } )$ , and the corresponding Q-function is

$$
Q _ { g } ^ { \pi } ( s , a ) \triangleq \mathbb { E } _ { \pi ( \cdot | \cdot , g ) } \left[ \sum _ { t ^ { \prime } = t } ^ { \infty } \gamma ^ { t ^ { \prime } - t } r _ { g } ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } ) \ | \ s _ { t } = s , a _ { t } = a \right] .\tag{1}
$$

Prior work (Eysenbach et al., 2022) has shown that this Q-function can equivalently be expressed as the probability of reaching the goal under the discounted state occupancy measure: $Q _ { g } ^ { \pi } ( s , a ) =$ $p _ { \pi ( \cdot \vert \cdot , g ) } ^ { \gamma } ( s _ { + } = g \mid s , a )$ . Here, $p _ { \pi ( \cdot \vert \cdot , g ) } ^ { \gamma } ( s _ { + } = g \mid s , a )$ denotes the probability of reaching g from $( s , a )$ at a temporal offset drawn from GEOM(1 − γ) under the goal-conditioned policy $\pi ( \cdot | \cdot , g ) .$ 1

Contrastive reinforcement learning (CRL). Instead of relying on Bellman backups, CRL $\mathrm { ( E y - }$ senbach et al., 2022) learns a critic for the goal-conditioned Q-function $Q _ { g } ^ { \pi } ( s , a )$ in a self-supervised manner. Given a batch $\boldsymbol { B } = \{ ( s _ { i } , a _ { i } , g _ { i } ) \} _ { i = 1 } ^ { B }$ , where the goal $g _ { i }$ corresponds to the state $\Delta _ { i }$ steps after $s _ { i }$ and $\Delta _ { i } \sim \mathrm { G E O M } ( 1 - \stackrel { . } { \gamma } )$ , the critic $f _ { \phi , \psi } ( s , a , g )$ is trained using the InfoNCE objective

$$
\mathcal { L } ^ { b a c k w a r d } ( \phi , \psi ; \mathcal { B } ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( f _ { \phi , \psi } ( s _ { i } , a _ { i } , g _ { i } ) ) } { \sum _ { j = 1 } ^ { B } \exp ( f _ { \phi , \psi } ( s _ { j } , a _ { j } , g _ { i } ) ) } .\tag{2}
$$

Under the backward InfoNCE objective in $\operatorname { E q . } \left( 2 \right)$ , the Bayes-optimal critic can be expressed as

$$
f ^ { * } ( s , a , g ) = \log \frac { p _ { \pi } ^ { \gamma } ( s _ { + } = g \mid s , a ) } { p _ { \pi } ^ { \gamma } ( g ) C ( g ) } ,\tag{3}
$$

where $p _ { \pi } ^ { \gamma } ( g )$ is the future-state marginal and $C ( g ) > 0$ is a normalization term that depends only on the future state $g .$ Since the denominator is independent of the action, maximizing the critic $f ^ { * } ( s , a , g )$ over actions is equivalent to maximizing the discounted occupancy $p _ { \pi } ^ { \gamma } ( s _ { + } = g \mid s , a )$ The policy $\pi _ { \theta } ( a \mid s , g )$ is therefore trained by maximizing the critic:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { s \sim p _ { \mathcal { D } } , g \sim p _ { \mathcal { D } } ^ { \gamma } ( \cdot | s ) } \left[ f _ { \phi , \psi } ( s , a , g ) \right] ,\tag{4}
$$

where $p _ { \mathcal { D } }$ denotes the state distribution of the replay buffer and $p _ { \mathcal { D } } ^ { \gamma } ( \cdot \mid s )$ the distribution of future goals from s in the replay buffer, with temporal offsets drawn from $\operatorname { G E O M } ( 1 - \gamma )$ .

## 4 HYPERSPHERICAL CONTRASTIVE REPRESENTATIONS FOR PLANNING

In this section, we look at planning a sequence of intermediate waypoints between a current state and a goal. We approach this as an inference problem and show how the waypoints can be obtained in closed form by repeatedly composing contrastive representations. We then extend this approach to planning over a variable number of waypoints, deriving a principled criterion for selecting an appropriate number of waypoints for each state-goal pair using the same learned representations.

## 4.1 PROBLEM FORMULATION

Setup. We collect data online under a hierarchical goal-conditioned policy πˆ that combines a highlevel planner with a policy $\pi _ { \theta } .$ Every h environment steps, the planner provides a waypoint representation z as a subgoal to reach the final goal, and the policy $\pi _ { \boldsymbol { \theta } } ( a \mid s , z )$ pursues it for the next h steps. We alternate between collecting data under πˆ and updating the representations and policy.

We take a probabilistic approach to planning. Given a state s and goal g, we infer K intermediate waypoints $w _ { 1 : K } = ( w _ { 1 } , \dots , w _ { K } )$ along trajectories generated by $\hat { \pi }$ with goals drawn from $p _ { g } .$ . For notational consistency, we set $w _ { 0 } = s$ and $w _ { K + 1 } = g$ . We assume that consecutive waypoints $w _ { i }$ and $w _ { i + 1 }$ are separated by an interval $T _ { i } \sim \mathrm { G E O M } ( 1 - \gamma )$ , drawn independently for $i = 0 , \ldots , K$

Objective. Let $p _ { \hat { \pi } } ^ { \gamma , K } ( w _ { 1 : K } \mid s , g )$ denote the distribution over K intermediate waypoints given s and $g$ under the data-generating process described above. We approximate this posterior with a tractable distribution $q ( w _ { 1 : K } \mid s , g )$ by minimizing the following KL divergence:

$$
q ^ { \star } = \arg \operatorname* { m i n } _ { \boldsymbol { q } } D _ { \mathrm { K L } } ( \boldsymbol { q } ( \cdot  { | \begin{array} { l } { \boldsymbol { s } , \boldsymbol { g } } \end{array} | } | p _ { \hat { \pi } } ^ { \gamma , K } ( \cdot  { | \begin{array} { l } { \boldsymbol { s } , \boldsymbol { g } } \end{array} ) } ) .\tag{5}
$$

For now, we fix $K$ and derive the corresponding optimal waypoints. We then derive a criterion for adaptively selecting K for each state-goal pair.

## 4.2 HYPERSPHERICAL CONTRASTIVE REPRESENTATIONS

To make the planning objective in Eq. (5) tractable, we parameterize temporal relationships using contrastive representations that lie on the unit hypersphere $\mathbb { S } ^ { d - 1 }$ . Our approach learns a state-only representation $\psi ( s ) \in \mathbb { S } ^ { d - 1 }$ , a state-action representation $\phi ( s , a ) \in \mathbb { S } ^ { d - 1 }$ , a rotation matrix $R ,$ and a policy $\pi _ { \theta } ( a \mid s , z )$ , where $z \in \mathbb { S } ^ { d - 1 }$ is a representation of a goal. We use these representations to infer intermediate waypoints for the policy $\pi _ { \theta }$ to pursue.

Parameterization. We use $f _ { H } ( s , s ^ { + } )$ to score temporal relationships between states and $f _ { L } ( s , a , s ^ { + } )$ to score future states given a current state-action pair. To infer consecutive waypoints in a common latent space, $f _ { H }$ uses the same encoder ψ for both inputs. Following the inner-product form used in CRL, we could score state pairs using $f _ { H } ( u , v ) = \bar { \psi } ( u ) ^ { \top } \psi ( v )$ . However, since this inner product is symmetric in u and $v ,$ it cannot capture asymmetries in the dynamics. We thus introduce a learnable rotation matrix R to model these asymmetries while preserving the hyperspherical geometry. Specifically, we parameterize $f _ { H }$ and $f _ { L }$ as

$$
f _ { H } ( \boldsymbol { s } , \boldsymbol { s } ^ { + } ) = \psi ( \boldsymbol { s } ) ^ { \top } R \psi ( \boldsymbol { s } ^ { + } ) , \qquad f _ { L } ( \boldsymbol { s } , \boldsymbol { a } , \boldsymbol { s } ^ { + } ) = \phi ( \boldsymbol { s } , \boldsymbol { a } ) ^ { \top } \psi ( \boldsymbol { s } ^ { + } ) ,\tag{6}
$$

where $R \in \mathbb { R } ^ { d \times d }$ satisfies $R ^ { \top } R = I . ^ { 2 }$ This allows us to score an entire waypoint sequence by repeatedly applying a single $f _ { H }$ to consecutive waypoint pairs. For $f _ { L }$ , we follow the standard CRL parameterization and train the policy to maximize $\bar { f } _ { L } ( s , \bar { a } , s ^ { + } ) = \bar { \phi ( } s , a ) ^ { \top } \psi ( s ^ { + } )$ ) as in Eq. (4). The corresponding learning objectives are described in Section $4 . 5$

## 4.3 CLOSED-FORM WAYPOINT INFERENCE

Key assumptions. Our derivation relies on the following assumptions, with theoretical justification based on prior work (Wang & Isola, 2020; Eysenbach et al., 2024) provided in Appendix $\mathrm { A . 1 } \colon$

Assumption 1 (Bayes optimality). The learned $f _ { H }$ under the symmetric InfoNCE objective has the following Bayes-optimal form:

$$
f _ { H } ( u , v ) = \log \frac { p _ { \pi } ^ { \gamma } ( u ^ { + } = v \mid u ) } { p _ { \pi } ^ { \gamma } ( v ) C } ,\tag{7}
$$

where π is the data-collection policy and $C > 0$ is a constant independent ofu and v.

Assumption 2 (Uniformity). The marginal distribution over the representations learned for f<sub>H</sub> under the symmetric InfoNCE objective is uniform on the unit hypersphere $\mathbb { S } ^ { d - 1 }$

$$
p _ { \pi } ^ { \gamma } ( \psi ) \triangleq \int p _ { \pi } ^ { \gamma } ( v ) \delta ( \psi - \psi ( v ) ) \ \mathrm { d } v = \mathrm { U n i f } \left( \mathbb { S } ^ { d - 1 } \right) .\tag{8}
$$

Tractable reformulation. To obtain a tractable objective, we reformulate Eq. (5) in terms of the learned representations. We first factorize q as $\begin{array} { r } { q ( w _ { 1 : K } \mid s , g ) = \prod _ { i = 1 } ^ { K } q _ { i } ( w _ { i } \mid s , g ) } \end{array}$ , with one distribution $q _ { i }$ per waypoint $w _ { i }$ . This allows us to optimize q for different $\bar { K }$ within the same formulation below. Under Assumption 1 and the Markov assumption, the objective can be equivalently written as follows, with the derivation provided in Appendix A.2:

$$
\begin{array} { r l } & { q ^ { \star } = \underset { q } { \arg \operatorname* { m i n } } D _ { \mathrm { K L } } \left( q ( \cdot  { \lvert { s } , { g } ) } \right) \bigg \| p _ { \widehat { \pi } } ^ { \gamma , K } ( \cdot  { \lvert { s } , { g } ) } \bigg ) } \\ & { \quad = \underset { q } { \arg \operatorname* { m a x } } \mathbb { E } _ { w _ { 1 : K } \sim q ( \cdot  { \lvert { s } , { g } ) } } \left[ \displaystyle \sum _ { i = 0 } ^ { K } f _ { H } ( w _ { i } , w _ { i + 1 } ) \right] - \displaystyle \sum _ { i = 1 } ^ { K } D _ { \mathrm { K L } } \left( q _ { i } ( \cdot  { \lvert { s } , { g } ) } \parallel p _ { \widehat { \pi } } ^ { \gamma } ( \cdot ) \right) } \end{array}\tag{9}
$$

This reformulation provides a natural interpretation of our objective in Eq. (5): it maximizes the expected sum of $f _ { H }$ , which scores temporal relationships between waypoints, while encouraging each waypoint distribution $q _ { i }$ to remain close to the marginal distribution of states visited under πˆ.

However, inferring waypoints directly in the high-dimensional state space is computationally challenging. Since the policy $\pi _ { \boldsymbol { \theta } } ( a \mid s , z )$ depends on the waypoint w only through its representation $z = \psi ( w )$ , inferring its representation is sufficient to provide a subgoal to the policy. We therefore infer distributions $q _ { i } ( z _ { i } \mid s , g )$ over representations $z _ { i } = \psi ( w _ { i } )$ rather than the raw states $w _ { i }$ Assumption 2 and the parameterization of $f _ { H }$ then yield the tractable objective $\mathcal { I } ( q )$ to maximize:

$$
\mathcal { I } ( \boldsymbol { q } ) = \mathbb { E } _ { z _ { 1 : K } \sim q ( \cdot | s , g ) } \left[ \psi ( s ) ^ { \top } R z _ { 1 } + \sum _ { i = 2 } ^ { K } z _ { i - 1 } ^ { \top } R z _ { i } + z _ { K } ^ { \top } R \psi ( g ) \right] + \sum _ { i = 1 } ^ { K } \mathcal { H } ( q _ { i } ( \cdot | s , g ) ) ,\tag{10}
$$

where H denotes entropy. This objective links the current state s to the goal g by repeatedly aligning consecutive representations $z _ { i }$ through the rotation matrix R, while the entropy term encourages stochasticity in the sequence of representations. The full derivation is provided in Appendix $\mathrm { A } . 2$

Closed-form solution. To model the stochasticity encouraged by $\mathcal { I } ( q )$ , we parameterize each $q _ { i } ( \cdot \mid s , g )$ as the distribution of a Gaussian sample normalized onto the unit hypersphere:

$$
\widetilde { z } _ { i } \sim { \cal N } \big ( \mu _ { i } ( s , g ) , \sigma _ { i } ^ { 2 } ( s , g ) I \big ) , \qquad z _ { i } = \frac { \widetilde { z } _ { i } } { \| \widetilde { z } _ { i } \| _ { 2 } } \sim q _ { i } ( \cdot \mid s , g ) ,
$$

with $\| \mu _ { i } ( s , g ) \| _ { 2 } = 1$ . As in SAC (Haarnoja et al., 2018b), we weight the entropy term in $\operatorname { E q . } \left( 1 0 \right)$ by $\alpha > 0$ to control stochasticity, with $\alpha = 1$ recovering the original objective. Under a small-variance approximation, we obtain the following optimal solution for maximizing $\mathcal { I } ( q )$

$$
\left\{ \begin{array} { l l } { \displaystyle \mu _ { i } ^ { * } ( s , g ) = ( R ^ { T } ) ^ { i } \mathrm { s l e r p } \Big ( \psi ( s ) , R ^ { K + 1 } \psi ( g ) , \frac { i } { K + 1 } \Big ) , } \\ { \displaystyle \sigma _ { i } ^ { * 2 } ( s , g ) = \frac { \alpha } { 2 \cos ( \frac { \theta } { K + 1 } ) } } \end{array} \right. \mathrm { f o r } i = 1 , \dots , K ,\tag{11}
$$

where $\theta = \operatorname { a r c c o s } ( \psi ( s ) ^ { \top } R ^ { K + 1 } \psi ( g ) )$ and slerp denotes spherical linear interpolation. The derivation is provided in Appendix ${ \mathrm { A } } . 3$

When $R = I$ , the solution places all means $\mu _ { i }$ evenly along the geodesic between $\psi ( s )$ and $\psi ( g )$ on the unit hypersphere. In contrast, our method accounts for temporal transitions encoded in R by interpolating from $\psi ( s )$ toward the rotated goal $R ^ { K + 1 } \psi ( g )$ and mapping each interpolated point back using $\overline { { ( R ^ { \intercal } ) } } ^ { i }$ (see Figure 2 for a geometric interpretation). Importantly, these closed-form expressions allow us to infer distributions over waypoint representations directly from the learned contrastive representations without training an additional neural network.

![](images/598985ba0562f7e23bc18ee6320aa0b08595ca8d3dcf37e593f76c2454827762.jpg)  
Figure 2: Geometric interpretation of C-Slerp. The number of waypoints K is selected using the alignment between $( R ^ { \top } ) ^ { k + 1 } \psi ( s )$ and $\psi ( g )$ . Given the selected $K ,$ , waypoint mean directions are obtained by spherical interpolation from $\psi ( s )$ to $R ^ { K + 1 } \psi ( g )$ , followed by mapping back via $( R ^ { \top } ) ^ { i }$ The policy then pursues the furthest waypoint whose alignment with $\phi ( s , a )$ meets the threshold ϵ.

## 4.4 HOW MANY WAYPOINTS ARE NEEDED FOR EACH STATE-GOAL PAIR?

Since the closed-form solution in Eq. (11) applies to any number of waypoints K, we now consider how to select $K$ for each state-goal pair. Recall that $f _ { H } ( s , g )$ measures alignment between $R ^ { \top } \psi ( s )$ and $\psi ( g )$ to score a single geometric transition from s to g. Extending this to $K + 1$ geometric transitions, we define the following alignment score between $( R ^ { \top } ) ^ { K + 1 } \bar { \psi } ( s )$ and $\psi ( g )$

$$
\ell _ { K } ( s , g ) \triangleq \left( ( R ^ { \top } ) ^ { K + 1 } \psi ( s ) \right) ^ { \top } \psi ( g ) = \psi ( s ) ^ { \top } R ^ { K + 1 } \psi ( g ) .\tag{12}
$$

This score measures how well the goal representation aligns with the direction predicted from the current state after $K + 1$ geometric transitions under πˆ. A larger score indicates that the corresponding number of waypoints is more compatible with the state-goal pair. To infer $K$ probabilistically, we model $\psi ( g )$ with a von Mises–Fisher likelihood centered on the predicted direction $( \boldsymbol { R } ^ { \top } ) ^ { \boldsymbol { K } + 1 } \dot { \boldsymbol { \psi } } ( s ) ;$

$$
p ( \psi ( g ) \mid \psi ( s ) , K ) \triangleq C _ { d } ( \kappa ) \exp ( \kappa \ell _ { K } ( s , g ) ) ,\tag{13}
$$

where $\kappa > 0$ is the concentration parameter and $C _ { d } ( \kappa )$ is the normalizing constant. With a uniform prior over $K \in \{ 0 , \ldots , K _ { \operatorname* { m a x } } \}$ , Bayes’ rule yields

$$
p ( K \mid \psi ( s ) , \psi ( g ) ) = \mathrm { S o f t m a x } _ { K \in \{ 0 , \ldots , K _ { \operatorname* { m a x } } \} } \left( \frac { \ell _ { K } ( s , g ) } { \eta } \right) ,\tag{14}
$$

where $\eta = 1 / \kappa$ controls the sharpness of the distribution over $K .$ This allows us to infer the number of waypoints for each state-goal pair using only the learned contrastive representations. The theoret ical justification for the alignment score $\ell _ { K }$ and the full derivation are provided in Appendix $\mathrm { A } . 4$

Which waypoint should the policy pursue? After selecting $K ,$ , we can sample waypoint representations $z _ { 1 : K }$ from the closed-form distributions in $\operatorname { E q . } \left( 1 1 \right)$ . The remaining question is which waypoint representation $z _ { i }$ to provide to the policy, given its current goal-reaching capabilities.

Our learned $f _ { L } ( s , a , g ) = \phi ( s , a ) ^ { \top } \psi ( g )$ provides a natural proxy for the goal-reaching capability of the current policy, based on its probabilistic interpretation in Eq. (3). Since $z _ { i }$ lies in the same space as $\psi ( g )$ , we use its alignment with $\phi ( s , a )$ to assess how likely the policy is to reach it. For each candidate $z _ { i } ,$ including $z _ { K + 1 } = \psi ( g )$ , we sample $a _ { i } \sim \pi _ { \theta } ( \cdot \mid s , z _ { i } )$ and compute $\rho _ { i } = \phi ( s , a _ { i } ) ^ { \top } z _ { i } .$ We select the waypoint with the largest index whose score meets a threshold $\epsilon \in [ - 1 , 1 ]$

$$
i ^ { * } = \operatorname* { m a x } \left\{ i \in \left\{ 1 , \dots , K + 1 \right\} : \rho _ { i } \geq \epsilon \right\} .\tag{15}
$$

If no candidate meets the threshold, we set $i ^ { * } = 1$ . This allows the policy $\pi _ { \theta } ( \boldsymbol { a } ~ | ~ \boldsymbol { s } , z _ { i ^ { * } } )$ to make progress toward the goal by pursuing the furthest waypoint estimated to be reachable, avoiding overly conservative subgoals. We provide a geometric interpretation of selecting K and $i ^ { * }$ in Figure 2 and visualizations of the $\rho _ { i }$ values in Figure 17.

## 4.5 LEARNING OBJECTIVES AND ALGORITHM

Representation learning. To learn the representations used for planning, we sample a batch $\boldsymbol { B } =$ $\{ ( \bar { s _ { i } } , a _ { i } , s _ { i } ^ { + } ) \} _ { i = 1 } ^ { B }$ from the replay buffer, with the temporal interval between $s _ { i }$ and $s _ { i } ^ { + }$ drawn from $\mathbf { G } \mathbf { E } 0 \mathbf { M } { \left( 1 - \gamma \right) }$ . Using this batch, we train $f _ { L }$ with the backward InfoNCE objective in Eq. (2) and $f _ { H }$ with the symmetric InfoNCE objective defined in Eq. (19). The total representation loss is

$$
\mathcal { L } _ { \mathrm { r e p r } } = \mathbb { E } _ { \boldsymbol { B } } \left[ \mathcal { L } ^ { \mathrm { b a c k w a r d } } ( \phi , \psi ; \mathcal { B } ) + \lambda _ { \mathrm { r e p r } } \mathcal { L } ^ { \mathrm { s y m m e t r i c } } ( R , \psi ; \mathcal { B } ) \right] ,\tag{16}
$$

where the expectation is over batches $\boldsymbol { B }$ sampled as described above.

![](images/ea506db986595fcbd216dd69d32f0be58acbdb96a38518602d9da1e347fbb501.jpg)  
Figure 3: C-Slerp solves long-horizon tasks where baselines never achieve a single success. Learning curves show success rates on BuilderBench manipulation tasks (top) and JaxGCRL locomotion tasks (bottom). C-Slerp achieves the highest success rates on most tasks across both benchmarks and is the only method to succeed on all cube-stacking tasks with a gripper.

Policy learning. Following CRL, we train the policy $\pi _ { \theta }$ to maximize $f _ { L } ( s , a , g ) = \phi ( s , a ) ^ { \top } \psi ( g )$

$$
\begin{array} { r l } & { \underset { \theta } { \mathrm { m a x } } \ \mathbb { E } _ { s \sim p _ { \mathcal { D } } , g \sim p _ { \mathcal { D } } ^ { \gamma } ( \cdot | s ) } \left[ \phi ( s , a ) ^ { \top } \psi ( g ) \right] . } \\ & { \quad \quad \quad \quad a \sim \pi _ { \theta } ( \cdot | s , \psi ( g ) ) } \end{array}\tag{17}
$$

Specifically, the policy $\pi _ { \boldsymbol { \theta } } ( a \mid s , \psi ( g ) )$ is conditioned on the goal representation $\psi ( g )$ , enabling it to pursue waypoints z inferred in the representation space induced by ψ.

Algorithm. Every h environment steps, the high-level planner determines the number of waypoints for the current state-goal pair and selects a waypoint as a subgoal for the policy. Algorithm 1 provides the full pseudocode, including data collection and policy and representation learning.

## 5 EXPERIMENTS

Benchmarks. We evaluate C-Slerp on six locomotion tasks in JaxGCRL (Bortkiewicz et al., 2025) and six manipulation tasks in BuilderBench (Ghugare et al., 2025). The JaxGCRL tasks cover Ant and Humanoid locomotion, including navigation through mazes of varying sizes to test goalreaching capability over different horizons. In BuilderBench, the agent arranges cubes into target configurations using a gripper or applying forces directly to the cubes (Yan et al., 2026). The number of cubes determines the difficulty of these tasks, with more cubes requiring longer horizons. Using a gripper extends these horizons further, as each placement requires additional interactions to grasp and release the cube. Further details on both benchmarks are provided in Appendix C.1, with environment visualizations in Figure 18.

Baselines. We compare C-Slerp with five GCRL methods: As a baseline using contrastive representations, we include CRL (Eysenbach et al., 2022), which learns representations of state-action pairs and future states and trains a policy to maximize their alignment. To compare different uses of hierarchy, we include RIS (Chane-Sane et al., 2021) and Online HIQL (Park et al., 2023). RIS uses imagined subgoals to regularize a flat policy during training, whereas Online HIQL uses a high-level policy to produce latent subgoals that a low-level policy pursues. For this comparison, we adapt HIQL, originally designed for offline reinforcement learning, to the online setting. For graph-based planning, we include L3P (Zhang et al., 2021a), which constructs a graph of latent landmarks connected by reachability estimates and searches it to plan waypoint sequences. Finally, SAC + HER (Haarnoja et al., 2018a; Andrychowicz et al., 2017) is a standard GCRL baseline that relabels goals with achieved states to learn from unsuccessful trajectories. Implementation details and hyperparameter settings are provided in Appendix C.2.

## 5.1 EXPERIMENTAL RESULTS

Benchmark Performance. Figure 3 compares C-Slerp with five baselines across twelve benchmark tasks. C-Slerp achieves the highest success rates on most tasks across both benchmarks. In BuilderBench, the agent must learn to construct target structures without a goal curriculum or intermediate rewards. Building taller towers requires more placements, thus leading to longer horizons and making exploration more challenging. The results show that C-Slerp and CRL perform comparably on shorter-horizon tasks such as the 2-tower task, but the performance gap between C-Slerp and the baselines widens as the number of cubes increases or when a gripper is used. Notably, C-Slerp is the only method to succeed on all cube-stacking tasks with a gripper, while most baselines fail to achieve any success.

In navigation tasks, several baselines also perform comparably on shorter-horizon tasks such as Ant Big Maze and Humanoid. As the required horizon increases, however, C-Slerp achieves consistently higher success rates. These results suggest that C-Slerp not only enables long-horizon planning but also facilitates exploration, allowing the agent to discover successful behaviors more efficiently by decomposing challenging tasks into easier subproblems.

![](images/429d8164048095ea107af0453ec1222107b41de6de28ed9a42c423e8fdd05154.jpg)  
Figure 4: Ablation on K-selection

![](images/4e816bfaba4d5a845c8a4809e48bdeb07e43970d198cef05ac6d3820ddf20cf6.jpg)  
Figure 5: Ablation on i-selection

Is adaptive waypoint planning necessary for long-horizon tasks? We first examine whether adapting the number of waypoints K (as described in Section 4.4) improves performance over using a fixed number. Figure 4 compares adaptive and fixed choices of K on the 8-tower task and Ant U5 Maze. For each fixed-K variant, the number of waypoints remains constant throughout the episode. For $K = 0$ , no intermediate waypoints are used, and thus the policy is always conditioned directly on the goal representation ψ(g). The results show that adaptive selection achieves the highest success rates in both environments. The fixed-K variants make little progress on the 8-tower task and reach lower final performance on Ant U5 Maze. Furthermore, increasing the fixed number of waypoints does not consistently improve performance, suggesting that simply introducing more intermediate waypoints is insufficient.

These results suggest that the number of waypoints should match the difficulty of the goal-reaching problem. Too few waypoints can make each subproblem too difficult for the policy to solve, whereas too many may slow progress toward the goal by introducing unnecessary planning stages. Moreover, the appropriate number of waypoints can also change within an episode as the agent approaches the goal (see Figures 13 and 14). Consequently, a fixed number of waypoints that works well early in an episode may become unnecessarily large as the remaining task becomes easier. Adaptive selection accounts for this change by adjusting K to the current state-goal pair, highlighting that the benefit of C-Slerp comes not only from introducing intermediate waypoints but also from adapting their number to the problem.

We next examine how to select the waypoint for the policy to pursue. Figure 5 compares adaptive waypoint selection with fixed choices of i, while keeping the selection of K adaptive. Among the fixed selection rules, $i \ = \ 1$ and $i \ = \ 2$ perform better on the 8-tower task, whereas $i = \lceil K / 2 \rceil$ and i = 3 perform better on Ant U5 Maze. Thus, a choice that works well in one environment may not be the best choice in another. Earlier waypoints $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , i = 1 )$ provide more conservative subgoals but may slow progress, whereas later waypoints can enable greater progress but may be too difficult for the current policy to reach. Adaptive selection uses the learned $f _ { L }$ to make this choice and achieves the best performance in both environments (see Figure 17 for a visualization of the selected waypoints). Together, these ablations show that adaptive selection of both K and i is needed to achieve the full performance of C-Slerp on long-horizon tasks.

![](images/a853262b881d302d87464a9dc602c0025f0edd9771a53dad1f61827a56ec7727.jpg)

![](images/45a343d9bebc29296c9070947d4195e9cf05ae6e1cc38335bc4c96c5e211deff.jpg)  
(b) Learning curves  
Figure 6: Evaluation of C-Slerp in single-goal environments. To assess exploration capability, we evaluate C-Slerp with a single fixed goal throughout training. The environments and their fixed goals are shown in (a), and success rates for C-Slerp and the baselines are presented in (b).

Exploration in single-goal settings. We further evaluate C-Slerp in a single-goal setting, where a single fixed goal is provided throughout training and evaluation, following the setting considered in prior work (Liu et al., 2025; Bastankhah et al., 2026). As shown in Figure 6, we modify the goal distributions of Ant Big Maze and Ant U4 Maze by fixing the goal to the most distant location in each maze environment. C-Slerp achieves higher success rates than all baselines, with all baselines except CRL and L3P achieving zero success. These results suggest that C-Slerp can facilitate directed exploration by producing intermediate waypoints that guide the agent toward distant goals, even when only a single fixed goal is provided.

![](images/22fe9705075336955bdbca865133e4420c7a1645e5a929726f95bbe5b9e9e2f8.jpg)  
Figure 7: Adaptive planning in C-Slerp. The top row shows successive states in the 5-tower task with a gripper. The bottom row shows spherical PCA (Liu et al., 2019) projections of the latent trajectory, inferred waypoints, and selected subgoal. As the agent approaches the goal, it replans by adjusting the number of waypoints and selecting a waypoint to pursue. We provide complete rollout visualizations in Figures 15 and 16 and videos on our project page.

How do the inferred waypoints relate to task progress? Figure 7 shows successive states in the 5-tower task with a gripper, alongside projections of their representations and inferred waypoints. As the agent builds the tower, its state representation ψ(s) progresses along the latent trajectory toward the goal representation ψ(g). At each replanning step, C-Slerp updates the number of intermediate waypoints K, infers their representations, and selects one for the policy to pursue. Notably, the selected waypoint is not always the first in the inferred plan, which is consistent with our adaptive waypoint selection strategy described in Section 4.4. We can also observe that fewer waypoints are used as the tower nears completion, suggesting that the selected number of waypoints K reflects the remaining temporal distance to the goal. Together, these visualizations show how C-Slerp guides the policy toward the goal by adapting the number of waypoints and selecting a subgoal in the learned representation space. Additional rollout visualizations are provided in Appendix B.

## 6 CONCLUSION

We introduce C-Slerp, an online goal-reaching method that uses hyperspherical contrastive representations for adaptive planning. These representations enable closed-form waypoint inference through spherical linear interpolation, without a separate high-level network or graph search. Experiments show that C-Slerp solves long-horizon navigation and manipulation tasks, including settings where none of the baselines achieve any success. Ablations further show that adaptively selecting both the number of waypoints and the waypoint for the policy to pursue improves performance over fixed choices. These results suggest that adapting task decomposition to the goal-reaching problem contributes to the benefits of hierarchy in goal-conditioned reinforcement learning.

Our approach has limitations. Like other self-supervised RL methods, C-Slerp requires extensive environment interaction for training, which can be costly to collect on real robots. This motivates extending C-Slerp to an offline RL setting, where it could learn from offline data without additiona environment interaction. Also, our experiments use state observations in simulation, leaving learning from visual observations and deployment on real robots untested. Improving data efficiency and extending the method to pixel-based observations are important directions for future work

Final remarks. In this paper, we show how choosing an appropriate amount of planning can unlock new capabilities for reinforcement learning. C-Slerp learns to stack five cubes with a gripper without a goal curriculum or dense rewards, while all evaluated baselines fail to achieve a single success. These findings suggest that adapting how tasks are decomposed is a promising direction for extending online goal-conditioned reinforcement learning to more challenging problems. Addi tionally, although C-Slerp can be viewed as a hierarchical goal-conditioned RL method, it achieves these benefits without a separate high-level policy network or a planning graph. Instead, planning follows directly from spherical linear interpolation in the learned representation space, showing that adaptive planning need not require a complex architecture.

## ACKNOWLEDGMENTS

We thank Mahsa Bastankhah and Royina Jayanth for their helpful feedback on the manuscript. We also thank members of the Lab for Autonomous Robotics Research (LARR) and the Princeton RL Lab for insightful discussions.

## REFERENCES

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. Advances in neural information processing systems, 30, 2017.

Mahsa Bastankhah, Grace Liu, Dilip Arumugam, Thomas L Griffiths, and Benjamin Eysenbach. Demystifying the mechanisms behind emergent exploration in goal-conditioned rl. In International Conference on Learning Representations, volume 2026, pp. 125535–125575, 2026.

Michał Bortkiewicz, Władysław Pałucki, Vivek Myers, Tadeusz Dziarmaga, Tomasz Arczewski, Łukasz Kucinski, and Benjamin Eysenbach. Accelerating goal-conditioned reinforcement learn-´ ing algorithms and research. In International Conference on Learning Representations, volume 2025, pp. 15732–15754, 2025.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.

Elliot Chane-Sane, Cordelia Schmid, and Ivan Laptev. Goal-conditioned reinforcement learning with imagined subgoals. In International Conference on Machine Learning, 2021.

Ben Eysenbach, Russ R Salakhutdinov, and Sergey Levine. Search on the replay buffer: Bridging planning and reinforcement learning. Advances in neural information processing systems, 32, 2019.

Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, and Russ R Salakhutdinov. Contrastive learning as goal-conditioned reinforcement learning. Advances in Neural Information Processing Systems, 35:35603–35620, 2022.

Benjamin Eysenbach, Vivek Myers, Ruslan Salakhutdinov, and Sergey Levine. Inference via interpolation: Contrastive representations provably enable planning and inference. Advances in Neural Information Processing Systems, 37:58901–58928, 2024.

Raj Ghugare, Roger Creus Castanyer, Catherine Ji, Kathryn Wantlin, Jin Schofield, Karthik Narasimhan, and Benjamin Eysenbach. Builderbench: The building blocks of intelligent agents. arXiv preprint arXiv:2510.06288, 2025.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018a.

Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905, 2018b.

Leslie Pack Kaelbling. Learning to achieve goals. In International Joint Conference on Artificial Intelligence, 1993.

Andrew Levy, George Konidaris, Robert Platt, and Kate Saenko. Learning multi-level hierarchies with hindsight. arXiv preprint arXiv:1712.00948, 2017.

Grace Liu, Michael Tang, and Benjamin Eysenbach. A single goal is all you need: Skills and exploration emerge from contrastive rl without rewards, demonstrations, or subgoals. In International Conference on Learning Representations, volume 2025, pp. 78599–78621, 2025.

Kai Liu, Qiuwei Li, Hua Wang, and Gongguo Tang. Spherical principal component analysis. In Proceedings of the 2019 SIAM international conference on data mining, pp. 387–395. SIAM, 2019.

Vivek Myers, Catherine Ji, and Benjamin Eysenbach. Horizon generalization in reinforcement learning. In International Conference on Learning Representations, volume 2025, pp. 76302–76326, 2025.

Ofir Nachum, Shixiang Shane Gu, Honglak Lee, and Sergey Levine. Data-efficient hierarchical reinforcement learning. Advances in neural information processing systems, 31, 2018.

Ofir Nachum, Haoran Tang, Xingyu Lu, Shixiang Gu, Honglak Lee, and Sergey Levine. Why does hierarchy (sometimes) work so well in reinforcement learning? arXiv preprint arXiv:1909.10618, 2019.

Soroush Nasiriany, Vitchyr Pong, Steven Lin, and Sergey Levine. Planning with goal-conditioned policies. Advances in neural information processing systems, 32, 2019.

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, and Sergey Levine. HIQL: Offline goalconditioned RL with latent states as actions. In Advances in Neural Information Processing Systems, 2023.

Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pp. 2778–2787. PMLR, 2017.

Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of mutual information. In International conference on machine learning, pp. 5171–5180. PMLR, 2019.

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In International conference on machine learning, pp. 1312–1320. PMLR, 2015.

Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, and Koray Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. In International conference on machine learning, pp. 3540–3549. PMLR, 2017.

Kevin Wang, Ishaan Javali, Michał Bortkiewicz, Tomasz Trzcinski, and Benjamin Eysenbach. 1000 layer networks for self-supervised rl: Scaling depth can enable new goal-reaching capabilities. Advances in Neural Information Processing Systems, 38:157643–157670, 2026.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pp. 9929–9939. PMLR, 2020.

Zheng Wen, Doina Precup, Morteza Ibrahimi, Andre Barreto, Benjamin Van Roy, and Satinder Singh. On efficiency in hierarchical reinforcement learning. Advances in Neural Information Processing Systems, 33:6708–6718, 2020.

David Yan, Brandon Cho, Andrew Tu, Vedant Badoni, Zechang Yang, Raj Ghugare, and Benjamin Eysenbach. Scaling the horizon of contrastive reinforcement learning. In Third Reinforcement Learning Beyond Rewards Workshop at the Reinforcement Learning Conference, 2026.

Chun-Hsiao Yeh, Cheng-Yao Hong, Yen-Chi Hsu, Tyng-Luh Liu, Yubei Chen, and Yann LeCun. Decoupled contrastive learning. In European conference on computer vision, pp. 668–684. Springer, 2022.

Lunjun Zhang, Ge Yang, and Bradly C. Stadie. World model as a graph: Learning latent landmarks for planning. In International Conference on Machine Learning, 2021a.

Tianjun Zhang, Benjamin Eysenbach, Ruslan Salakhutdinov, Sergey Levine, and Joseph E Gonzalez. C-Planning: An automatic curriculum for learning goal-reaching tasks. arXiv preprint arXiv:2110.12080, 2021b.

Tianren Zhang, Shangqi Guo, Tian Tan, Xiaolin Hu, and Feng Chen. Generating adjacencyconstrained subgoals in hierarchical reinforcement learning. Advances in neural information processing systems, 33:21579–21590, 2020.

John Zhou and Jonathan Kao. Flattening hierarchies with policy bootstrapping. Advances in Neural Information Processing Systems, 38:82975–83007, 2026.

Alicja Ziarko, Michał Bortkiewicz, Michał Zawalski, Benjamin Eysenbach, and Piotr Miłos. Con-´ trastive representations for temporal reasoning. Advances in Neural Information Processing Systems, 38:109229–109259, 2026.

## APPENDIX

A Theoretical Results 14   
A.1 Justification of the key assumptions 14   
A.2 Derivation of the representation-space objective 15   
A.3 Derivation of the closed-form solution . 16   
A.4 Derivation of the distribution over the number of waypoints . 18   
B Additional Experimental Results 20   
B.1 Ablation Study 20   
B.2 CRL with larger discount factors . 22   
B.3 Visualizations 23   
C Experimental Details 33   
C.1 Environments 33   
C.2 Implementation details 34

## A THEORETICAL RESULTS

## A.1 JUSTIFICATION OF THE KEY ASSUMPTIONS

Assumption 1 (Bayes optimality). The learned $f _ { H }$ under the symmetric InfoNCE objective has the following Bayes-optimalform:

$$
f _ { H } ( u , v ) = \log \frac { p _ { \pi } ^ { \gamma } ( u ^ { + } = v \mid u ) } { p _ { \pi } ^ { \gamma } ( v ) C } ,\tag{18}
$$

where $\pi$ is the data-collection policy and $C > 0$ is a constant independent ofu and v.

The symmetric InfoNCE objective for a batch $\boldsymbol { B } = \{ ( s _ { i } , a _ { i } , s _ { i } ^ { + } ) \} _ { i = 1 } ^ { B }$ is defined as

$$
\mathcal { L } ^ { \mathrm { s y m m e t r i c } } ( R , \psi ; \mathcal { B } ) \triangleq - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ \log \frac { \exp \left( f _ { H } ( s _ { i } , s _ { i } ^ { + } ) \right) } { \sum _ { j = 1 } ^ { B } \exp \left( f _ { H } ( s _ { i } , s _ { j } ^ { + } ) \right) } + \log \frac { \exp \left( f _ { H } ( s _ { i } , s _ { i } ^ { + } ) \right) } { \sum _ { j = 1 } ^ { B } \exp \left( f _ { H } ( s _ { j } , s _ { i } ^ { + } ) \right) } \right] .\tag{19}
$$

The Bayes-optimal form in Eq. (18) follows directly from Lemma 5 of Eysenbach et al. (2024). Their formulation uses denominators without resubstitution (Poole et al., 2019; Yeh et al., 2022), excluding the positive pair from each denominator. Since $| f _ { H } | \le 1$ , including the positive pair changes the loss by at most $2 \log ( 1 + e ^ { 2 } / ( B - 1 ) )$ , which vanishes as $B  \infty$ . The same limiting result therefore applies to Eq. (19). We assume that $f _ { H }$ attains this optimum in our analysis.

Assumption 2 (Uniformity). The marginal distribution over the representations learned for f<sub>H</sub> under the symmetric InfoNCE objective is uniform on the unit hypersphere $\mathbb { S } ^ { d - 1 }$

$$
p _ { \pi } ^ { \gamma } ( \psi ) \triangleq \int p _ { \pi } ^ { \gamma } ( v ) \delta ( \psi - \psi ( v ) ) \ \mathrm { d } v = \mathrm { U n i f } \left( \mathbb { S } ^ { d - 1 } \right) .\tag{20}
$$

Following Wang & Isola (2020), we decompose the objective in Eq. (19) into alignment and uniformity terms corresponding to the numerators and denominators, respectively. We justify Assumption 2 using this decomposition and its entropy interpretation. Since $R$ is an orthogonal matrix and the representations ψ have unit norm, we have

$$
f _ { H } \big ( \boldsymbol { s } _ { i } , \boldsymbol { s } _ { i } ^ { + } \big ) = \psi ( \boldsymbol { s } _ { i } ) ^ { \top } R \psi ( \boldsymbol { s } _ { i } ^ { + } ) = 1 - \frac { 1 } { 2 } \| R ^ { \top } \psi ( \boldsymbol { s } _ { i } ) - \psi ( \boldsymbol { s } _ { i } ^ { + } ) \| _ { 2 } ^ { 2 } .
$$

The alignment term therefore pulls $R ^ { \top } \psi ( s _ { i } )$ toward $\psi ( s _ { i } ^ { + } )$ for each positive pair. For the uniformity term, we consider the log-denominator of the first term in Eq. (19). If positive pairs are sufficiently aligned, replacing $R ^ { \top } \psi ( s _ { i } )$ with $\psi ( s _ { i } ^ { + } )$ in the von Mises–Fisher (vMF) log-density gives the approximation below. After subtracting the constant log B, this log-denominator can be written as

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \Bigg ( \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \exp \big ( f _ { H } ( s _ { i } , s _ { j } ^ { + } ) \big ) \Bigg ) = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \Bigg ( \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \frac { \exp \big ( \psi ( s _ { i } ) ^ { \top } R \psi ( s _ { j } ^ { + } ) \big ) } { Z _ { \mathrm { v M F } } } \Bigg ) + \log Z _ { \mathrm { v M F } } } & { } \\ { = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \widehat { p } _ { \mathrm { v M F } } ( R ^ { \top } \psi ( s _ { i } ) ) + \log Z _ { \mathrm { v M F } } } & { } \\ { \approx \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \widehat { p } _ { \mathrm { v M F } } ( \psi ( s _ { i } ^ { + } ) ) + \log Z _ { \mathrm { v M F } } } & { } \\ { = - \widehat { \mathcal { H } } ( \psi ) + \log Z _ { \mathrm { v M F } } , } & { ( 2 1 ) } \end{array}
$$

where $\begin{array} { r } { \widehat { p } _ { \mathrm { v M F } } ( u ) = \left[ \sum _ { i } \exp \bigl ( u ^ { \intercal } \psi ( s _ { i } ^ { + } ) \bigr ) \right] / ( B Z _ { \mathrm { v M F } } ) } \end{array}$ is a vMF kernel density estimate, $Z _ { \mathrm { v M F } }$ is its normalizing constant, and $\widehat { \mathcal { H } } ( \psi ) = - \big | \textstyle \sum \ d u$ <sub>i</sub> log $\widehat { p } _ { \mathrm { v M F } } ( \psi ( s _ { i } ^ { + } ) ) \big ] / B$ is the corresponding resubstitution entropy estimate. The entropy interpretation for the denominator of the second term in Eq. (19) can be derived in the same way.

Minimizing the term $- { \widehat { \mathcal { H } } } ( \psi )$ in Eq. (21) maximizes the estimated marginal entropy of $\psi ,$ encouraging representations to spread across the sphere. Since maximizing entropy on the unit sphere yields the uniform distribution on $\mathbb { S } ^ { d - 1 }$ , we assume a uniform marginal distribution for the learned representations ψ in our analysis.

## A.2 DERIVATION OF THE REPRESENTATION-SPACE OBJECTIVE

In this section, we derive the objective in Eq. (10) from the posterior KL objective in Eq. (5). We fix the state-goal pair $( s , g )$ , the number of intermediate waypoints $K \geq 1$ , and the data-collection policy πˆ. Following the setup in Section 4.1, we set $w _ { 0 } = s$ and $\ d s ^ { 2 } K + 1 \ d s = \ d g$ and use the factorization $\begin{array} { r } { q ( w _ { 1 : K } \mid s , g ) = \prod _ { i = 1 } ^ { K } q _ { i } ( w _ { i } \mid s , g ) } \end{array}$

Assuming that the waypoint sequence is a Markov chain, the posterior can be written as

$$
p _ { \vec { \pi } } ^ { \gamma , K } ( w _ { 1 : K } \mid s , g ) = \frac { p _ { \vec { \pi } } ^ { \gamma , K } ( w _ { 1 : K , g } \mid s ) } { p _ { \vec { \pi } } ^ { \gamma , K } ( g \mid s ) } = \frac { \prod _ { i = 0 } ^ { K } p _ { \vec { \pi } } ^ { \gamma } ( w _ { i + 1 } \mid w _ { 0 : i } ) } { p _ { \vec { \pi } } ^ { \gamma , K } ( g \mid s ) } = \frac { \prod _ { i = 0 } ^ { K } p _ { \vec { \pi } } ^ { \gamma } ( w _ { i + 1 } \mid w _ { i } ) } { p _ { \vec { \pi } } ^ { \gamma , K } ( g \mid s ) } .
$$

Expanding the KL divergence in Eq. (5) gives

$$
\begin{array} { l }  - D _ { \mathrm { K L } } ( q ( \cdot  { | \begin{array} { l } { s , g } \end{array} ) \| \begin{array} { l } { p _ { \hat { \pi } } ^ { \gamma , K } ( \cdot  { | \begin{array} { l } { s , g } \end{array} ) } } \\ { i = 1 } \end{array}  } \\ { = \mathbb { E } _ { w _ { 1 : K } \sim q ( \cdot  { | s , g ) } } [ \displaystyle \sum _ { i = 0 } ^ { K } \log p _ { \hat { \pi } } ^ { \gamma } ( w _ { i + 1 }  { | \begin{array} { l } { w _ { i } } \end{array}  } ) - \displaystyle \sum _ { i = 1 } ^ { K } \log q _ { i } ( w _ { i }  { | \begin{array} { l } { s , g } \end{array} ) } ] - \log p _ { \hat { \pi } } ^ { \gamma , K } ( g  { | \begin{array} { l } { s } \end{array} ) } . } \end{array}\tag{22}
$$

Applying Assumption 1 with $\pi = { \hat { \pi } }$ , we have

$$
\log p _ { \hat { \pi } } ^ { \gamma } ( w _ { i + 1 } \mid w _ { i } ) = f _ { H } ( w _ { i } , w _ { i + 1 } ) + \log p _ { \hat { \pi } } ^ { \gamma } ( w _ { i + 1 } ) + \log C .\tag{23}
$$

Substituting Eq. (23) into Eq. (22) yields

$$
\begin{array} { r l } & { - D _ { \mathrm { K L } } ( q ( \cdot  { \lvert ~ s , g )  }  p _ { \hat { \pi } } ^ { \gamma , K } ( \cdot  { \lvert ~ s , g ) ) } } \\ & { = \mathbb { E } _ { w _ { 1 : K } \sim q ( \cdot  { \lvert ~ s , g ) } } [ \displaystyle \sum _ { i = 0 } ^ { K } f _ { H } ( w _ { i } , w _ { i + 1 } ) + \displaystyle \sum _ { i = 1 } ^ { K } \log \frac { p _ { \hat { \pi } } ^ { \gamma } ( w _ { i } ) } { q _ { i } ( w _ { i } \  { \lvert ~ s , g ) } } ] + \mathrm { c o n s t . } } \\ & { = \mathbb { E } _ { w _ { 1 : K } \sim q ( \cdot  { \lvert ~ s , g ) } } [ \displaystyle \sum _ { i = 0 } ^ { K } f _ { H } ( w _ { i } , w _ { i + 1 } ) ] - \displaystyle \sum _ { i = 1 } ^ { K } D _ { \mathrm { K L } } ( q _ { i } ( \cdot  { \lvert ~ s , g ) \rvert }  { \lVert ~ p _ { \hat { \pi } } ^ { \gamma } ( \cdot ) ) + \mathrm { c o n s t . } } } \end{array}\tag{24}
$$

Here, the constant is log $p _ { \hat { \pi } } ^ { \gamma } ( g ) + ( K + 1 ) \log C - \log p _ { \hat { \pi } } ^ { \gamma , K } ( g \mid s )$ . It is independent of q since $( s , g )$ and K are fixed. Removing it gives the objective in $\operatorname { E q . } \left( 9 \right)$

We now express the objective in terms of $z _ { i } = \psi ( w _ { i } )$ . Following the notation in Section $4 . 3 ,$ , let $q _ { i } ( z _ { i } \mid s , g )$ denote the representation density induced by $q _ { i } ( w _ { i } \mid s , g )$ through $\psi ,$ and let $p _ { \hat { \pi } } ^ { \gamma } ( z _ { i } )$ denote the representation marginal density induced by $p _ { \hat { \pi } } ^ { \gamma } ( w _ { i } )$ through the same mapping. Also, the joint distribution remains factorized as $\begin{array} { r } { q ( z _ { 1 : K } \mid s , g ) = \prod _ { i = 1 } ^ { K } q _ { i } ( z _ { i } \mid s , g ) } \end{array}$

By the chain rule, the KL divergence in Eq. (24) decomposes into a representation term and an expected conditional term over states:

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } \left( q _ { i } ( w _ { i } \mid s , g ) \parallel p _ { \hat { \pi } } ^ { \gamma } ( w _ { i } ) \right) } \\ & { \ = D _ { \mathrm { K L } } \left( q _ { i } ( z _ { i } \mid s , g ) \parallel p _ { \hat { \pi } } ^ { \gamma } ( z _ { i } ) \right) + \mathbb { E } _ { z _ { i } \sim q _ { i } ( \cdot \mid s , g ) } \left[ D _ { \mathrm { K L } } \left( q _ { i } ( w _ { i } \mid z _ { i } , s , g ) \parallel p _ { \hat { \pi } } ^ { \gamma } ( w _ { i } \mid z _ { i } ) \right) \right] , } \\ & { \ \geq D _ { \mathrm { K L } } \left( q _ { i } ( z _ { i } \mid s , g ) \parallel p _ { \hat { \pi } } ^ { \gamma } ( z _ { i } ) \right) . } \end{array}\tag{25}
$$

For any $q _ { i } ( z _ { i } \mid s , g )$ such that $D _ { \mathrm { K L } } ( q _ { i } ( z _ { i } \mid s , g ) \| p _ { \hat { \pi } } ^ { \gamma } ( z _ { i } ) ) < \infty$ , the equality in Eq. (25) is always achievable with $q _ { i } ( w _ { i } \mid z _ { i } , s , g ) = p _ { \hat { \pi } } ^ { \gamma } ( w _ { i } \mid z _ { i } )$ . Thus, with this choice, minimizing the KL divergence over representations can lead to minimizing the KL divergence in Eq. (24).

Under Assumption $2 , p _ { \hat { \pi } } ^ { \gamma } ( \psi )$ is uniform on $\mathbb { S } ^ { d - 1 }$ , so $\log p _ { \hat { \pi } } ^ { \gamma } ( z _ { i } )$ is constant. Therefore, the KL divergence simplifies as follows:

$$
\begin{array} { r l } & { - D _ { \mathrm { K L } } \left( q _ { i } ( z _ { i } \mid s , g ) \parallel p _ { \hat { \pi } } ^ { \gamma } ( z _ { i } ) \right) = - \mathbb { E } _ { z _ { i } \sim q _ { i } ( \cdot \mid s , g ) } \left[ \log q _ { i } ( z _ { i } \mid s , g ) \right] + \mathrm { c o n s t . } } \\ & { ~ = \mathcal { H } ( q _ { i } ( \cdot \mid s , g ) ) + \mathrm { c o n s t . } , } \end{array}\tag{26}
$$

where H denotes entropy with respect to the surface measure on $\mathbb { S } ^ { d - 1 }$ . Substituting $f _ { H } ( w _ { i } , w _ { i + 1 } ) =$ $z _ { i } ^ { \top } R z _ { i + 1 }$ and Eq. (26) into Eq. (24) and dropping constants gives the objective in Eq. (10):

$$
\mathcal { I } ( \boldsymbol { q } ) = \mathbb { E } _ { z _ { 1 : K } \sim q ( \cdot | s , g ) } \left[ \psi ( s ) ^ { \top } R z _ { 1 } + \sum _ { i = 2 } ^ { K } z _ { i - 1 } ^ { \top } R z _ { i } + z _ { K } ^ { \top } R \psi ( g ) \right] + \sum _ { i = 1 } ^ { K } \mathcal { H } ( q _ { i } ( \cdot | s , g ) ) .
$$

## A.3 DERIVATION OF THE CLOSED-FORM SOLUTION

In this section, we derive the approximate solution in Eq. (11) for a fixed state-goal pair $( s , g )$ and $K \geq 1$ intermediate waypoints. For simplicity, we suppress the dependence on $( s , g )$ and write $\begin{array} { r } { q ( z _ { 1 : K } ) = \prod _ { i = 1 } ^ { K } q _ { i } ( z _ { i } ) } \end{array}$ , with $z _ { 0 } = \psi ( s )$ and $z _ { K + 1 } = \psi ( g )$ . To control the stochasticity of the waypoint representations, we introduce an entropy weight $\alpha > 0$ into Eq. (10):

$$
\mathcal { I } ( \boldsymbol { q } ) = \mathbb { E } _ { z _ { 1 : K } \sim q } \left[ \sum _ { i = 0 } ^ { K } z _ { i } ^ { \top } R z _ { i + 1 } \right] + \alpha \sum _ { i = 1 } ^ { K } \mathcal { H } ( q _ { i } ) .\tag{27}
$$

As in the main text, each waypoint representation is obtained by normalizing a Gaussian sample:

$$
\tilde { z } _ { i } \sim \mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } I ) , \quad z _ { i } = \frac { \tilde { z } _ { i } } { \| \tilde { z } _ { i } \| _ { 2 } } , \quad \| \mu _ { i } \| _ { 2 } = 1 .
$$

We use a small-variance approximation to obtain both mean directions $\mu _ { i }$ and variances $\sigma _ { i } ^ { 2 }$ in closed form. We first determine the mean directions, then use them to optimize the variances.

Mean directions of waypoint representations. With isotropic Gaussian covariance, the entropy $\mathcal { H } ( q _ { i } )$ in Eq. (27) depends only on the variances and is therefore constant when optimizing the mean directions. Under $\begin{array} { r } { q ( z _ { 1 : K } ) = \prod _ { i = 1 } ^ { K } q _ { i } ( z _ { i } ) } \end{array}$ , each waypoint representation $z _ { i }$ is independent of the others, allowing us to express the objective in terms of their individual expectations:

$$
\mathcal { T } ( q ) = \sum _ { i = 0 } ^ { K } \mathbb { E } [ z _ { i } ] ^ { \top } R \mathbb { E } [ z _ { i + 1 } ] + \mathrm { c o n s t . } ,\tag{28}
$$

where the constant is independent of $\mu _ { 1 } , \ldots , \mu _ { K }$

Since $z _ { i }$ is normalized to unit length, we approximate small deviations from $\mu _ { i }$ by the perpendicular component of the Gaussian perturbation, denoted by $\xi _ { i }$ . Its $d - 1$ orthogonal coordinates each have variance $\sigma _ { i } ^ { 2 }$ , so

$$
\mathbb { E } \big [ \| \xi _ { i } \| _ { 2 } ^ { 2 } \big ] = ( d - 1 ) \sigma _ { i } ^ { 2 } .
$$

For a small perturbation, $\| \xi _ { i } \| _ { 2 }$ approximates the angular deviation from $\mu _ { i }$ . Thus, the small-variance assumption $( d - 1 ) \sigma _ { i } ^ { 2 } \ll 1$ ensures that the representation remains concentrated around its mean direction $\mu _ { i }$

In the tangent-plane approximation, $\xi _ { i } \perp \mu _ { i }$ and $\| \mu _ { i } \| _ { 2 } = 1$ imply $\| \mu _ { i } + \xi _ { i } \| _ { 2 } = \sqrt { 1 + \| \xi _ { i } \| _ { 2 } ^ { 2 } }$ . We can therefore write

$$
\mathbb { E } [ z _ { i } ] \approx \mathbb { E } \left[ \frac { \mu _ { i } } { \sqrt { 1 + \| \xi _ { i } \| _ { 2 } ^ { 2 } } } \right] + \mathbb { E } \left[ \frac { \xi _ { i } } { \sqrt { 1 + \| \xi _ { i } \| _ { 2 } ^ { 2 } } } \right] .
$$

The second term vanishes by symmetry since $\xi _ { i }$ and $- \xi _ { i }$ have the same distribution and the denominator is unchanged under this sign reversal. Using $( 1 + x ) ^ { - 1 / 2 } \approx 1 - x / 2$ , we obtain

$$
\mathbb { E } [ z _ { i } ] \approx \left( 1 - \frac { 1 } { 2 } \mathbb { E } \big [ \| \xi _ { i } \| _ { 2 } ^ { 2 } \big ] \right) \mu _ { i } = \left( 1 - \frac { d - 1 } { 2 } \sigma _ { i } ^ { 2 } \right) \mu _ { i } .
$$

Under the small-variance assumption, we use $\mathbb { E } [ z _ { i } ] \approx \mu _ { i }$ to optimize the mean directions independently of the variances. Setting $v _ { i } = R ^ { i } \mu _ { i } { \mathrm { ~ f o r ~ } } i = 1 , \ldots , K$ , maximizing Eq. (28) under this approximation reduces to

$$
\operatorname* { m a x } _ { v _ { 1 } , \ldots , v _ { K } \in \mathbb { S } ^ { d - 1 } } \sum _ { i = 0 } ^ { K } v _ { i } ^ { \top } v _ { i + 1 } ,
$$

where $v _ { 0 } = \psi ( s )$ and $v _ { K + 1 } = R ^ { K + 1 } \psi ( g )$ . Let $\theta \in ( 0 , \pi )$ denote the angle between $v _ { 0 }$ and $v _ { K + 1 }$ The sum is maximized by placing the directions at equal angular intervals $\Delta = \theta / ( K + 1 )$ along the shortest geodesic connecting them, yielding the spherical linear interpolation

$$
v _ { i } ^ { * } = \mathrm { s l e r p } \biggl ( v _ { 0 } , v _ { K + 1 } , \frac { i } { K + 1 } \biggr ) .
$$

Using $\mu _ { i } = ( R ^ { \top } ) ^ { i } v _ { i } .$ , we obtain

$$
\mu _ { i } ^ { * } ( s , g ) = ( R ^ { \top } ) ^ { i } \mathrm { s l e r p } \left( \psi ( s ) , R ^ { K + 1 } \psi ( g ) , \frac { i } { K + 1 } \right) .\tag{29}
$$

The case $\theta = 0$ follows by continuity, and we exclude $\theta = \pi ,$ , where the shortest arc is not unique.

Variances of waypoint representations. We now optimize the variances in Eq. (27), with the mean directions fixed at $\mu _ { i } ^ { * }$ from Eq. (29). For each i, we rewrite the objective to isolate its dependence on $\sigma _ { i } \mathbf { : }$

$$
\mathcal { I } ( \boldsymbol { q } ) = \left( R ^ { \top } \mathbb { E } [ z _ { i - 1 } ] + R \mathbb { E } [ z _ { i + 1 } ] \right) ^ { \top } \mathbb { E } [ z _ { i } ] + \alpha \mathcal { H } ( q _ { i } ) + \mathrm { c o n s t . } ,\tag{30}
$$

where the constant is independent of $\sigma _ { i } .$ To optimize the variances, we retain the reduction in mean length derived above. A larger variance allows sampled waypoint sequences to cover a broader region of representation space, increasing entropy while reducing the expected inner products between neighboring representations. Using only $\mathbb { E } [ z _ { i } ] \approx \mu _ { i } ^ { * }$ makes the first term independent of the variances and fails to capture this tradeoff.

Set $\mu _ { 0 } ^ { * } = \psi ( s ) , \mu _ { K + 1 } ^ { * } = \psi ( g )$ , and $\sigma _ { 0 } ^ { 2 } = \sigma _ { K + 1 } ^ { 2 } = 0$ for the fixed endpoints. Substituting $\mathbb { E } [ z _ { j } ] \approx$ $\begin{array} { r } { ( 1 - \frac { d - 1 } { 2 } \sigma _ { j } ^ { 2 } ) \mu _ { j } ^ { * } } \end{array}$ for $j = i , i + 1$ into each neighboring inner product gives

$$
\mathbb { E } [ z _ { i } ] ^ { \top } R \mathbb { E } [ z _ { i + 1 } ] \approx \left[ 1 - \frac { d - 1 } { 2 } \left( \sigma _ { i } ^ { 2 } + \sigma _ { i + 1 } ^ { 2 } \right) \right] ( \mu _ { i } ^ { * } ) ^ { \top } R \mu _ { i + 1 } ^ { * } ,
$$

where we omit the products $\sigma _ { i } ^ { 2 } \sigma _ { i + 1 } ^ { 2 }$ under the small-variance assumption. Applying this expansion to the two pairs $( i - 1 , i )$ and $( i , i + 1 )$ in Eq. (30), the terms proportional to $\sigma _ { i } ^ { 2 }$ are

$$
- \frac { d - 1 } { 2 } \sigma _ { i } ^ { 2 } \left[ ( \mu _ { i - 1 } ^ { * } ) ^ { \top } R \mu _ { i } ^ { * } + ( \mu _ { i } ^ { * } ) ^ { \top } R \mu _ { i + 1 } ^ { * } \right] = - ( d - 1 ) \sigma _ { i } ^ { 2 } \cos { \Delta } .
$$

Recall that consecutive transformed directions $v _ { j } ^ { * } = R ^ { j } \mu _ { j } ^ { * }$ are separated by the angle $\Delta = \theta / ( K +$ 1). Thus, each inner product satisfies

$$
( \mu _ { j } ^ { * } ) ^ { \top } R \mu _ { j + 1 } ^ { * } = ( v _ { j } ^ { * } ) ^ { \top } v _ { j + 1 } ^ { * } = \cos \Delta , \qquad j = i - 1 , i .
$$

Collecting the remaining terms independent of $\sigma _ { i }$ into a constant, we can therefore write

$$
\mathcal { I } ( q ) \approx - ( d - 1 ) \sigma _ { i } ^ { 2 } \cos \Delta + \alpha \mathcal { H } ( q _ { i } ) + \mathrm { c o n s t . }\tag{31}
$$

Under the same small-variance assumption, we approximate $\mathcal { H } ( q _ { i } )$ by the entropy of an isotropic Gaussian in the $( d - 1 )$ )-dimensional tangent plane. Each of its $d - 1$ independent coordinates follows $X \sim \mathcal { N } ( 0 , \sigma _ { i } ^ { 2 } )$ , with density

$$
p _ { X } ( x ) = \frac { 1 } { \sqrt { 2 \pi \sigma _ { i } ^ { 2 } } } \exp \biggl ( - \frac { x ^ { 2 } } { 2 \sigma _ { i } ^ { 2 } } \biggr ) .
$$

Substituting this density into $- \mathbb { E } [ \log p _ { X } ( X ) ]$ gives

$$
- \mathbb { E } [ \log p _ { X } ( X ) ] = \frac { 1 } { 2 } \log ( 2 \pi \sigma _ { i } ^ { 2 } ) + \frac { \mathbb { E } [ X ^ { 2 } ] } { 2 \sigma _ { i } ^ { 2 } } = \frac { 1 } { 2 } \log ( 2 \pi e \sigma _ { i } ^ { 2 } ) ,
$$

where $\mathbb { E } [ X ^ { 2 } ] = \sigma _ { i } ^ { 2 }$ . Since these coordinates are independent, their entropies add, giving the following expression under the tangent-plane approximation:

$$
\mathcal { H } ( q _ { i } ) \approx \frac { d - 1 } { 2 } \log ( 2 \pi e \sigma _ { i } ^ { 2 } ) = \frac { d - 1 } { 2 } \big ( 1 + \log ( 2 \pi ) \big ) + ( d - 1 ) \log \sigma _ { i } .
$$

Substituting the entropy expression above into Eq. (31), and collecting terms independent of $\sigma _ { i }$ into a constant, we obtain

$$
\mathcal { I } ( q ) \approx - ( d - 1 ) \sigma _ { i } ^ { 2 } \cos \Delta + \alpha ( d - 1 ) \log \sigma _ { i } + \mathrm { c o n s t } .\tag{32}
$$

Differentiating the right-hand side with respect to $\sigma _ { i }$ and setting the derivative to zero gives

$$
- 2 ( d - 1 ) \cos \Delta \sigma _ { i } + \frac { \alpha ( d - 1 ) } { \sigma _ { i } } = 0 .
$$

Using $\Delta = \theta / ( K + 1 )$ , we obtain

$$
\sigma _ { i } ^ { 2 * } ( s , g ) = \frac { \alpha } { 2 \cos \Delta } = \frac { \alpha } { 2 \cos ( \theta / ( K + 1 ) ) } , \qquad i = 1 , \dots , K .\tag{33}
$$

For $d \geq 2$ and $\alpha > 0$ , the approximate objective in Eq. (32) is strictly concave in $\sigma _ { i } > 0$ since cos $\Delta > 0 ,$ so the positive stationary point is its unique maximizer. Together, Eqs. $( 2 9 )$ and (33) yield the approximate solution in Eq. (11) under the small-variance condition $\alpha ( d - 1 ) / ( 2 \cos \Delta ) \stackrel { . . } { \ll } 1$

## A.4 DERIVATION OF THE DISTRIBUTION OVER THE NUMBER OF WAYPOINTS

In this section, we derive a distribution over the number of intermediate waypoints for a given state goal pair. Following the setup in Section 4.1, we use the notation

$$
w _ { 1 : K } = ( w _ { 1 } , \ldots , w _ { K } ) , \quad w _ { 0 } = s , \quad w _ { K + 1 } = g .
$$

The $K$ intermediate waypoints divide the trajectory into $K + 1$ segments, where the temporal intervals are sampled independently from a geometric distribution. We define $p _ { \hat { \pi } } ^ { \gamma , K } ( g \mid s )$ as the resulting distribution of $g$ under πˆ given the starting state s. For $K = 0$ , this distribution reduces to $p _ { \hat { \pi } } ^ { \gamma } ( g \mid s ) \bar $ with a time interval sampled from a single geometric distribution and no intermediate waypoints.

Recall that $f _ { H }$ is parameterized as

$$
f _ { H } ( s , s ^ { + } ) = \psi ( s ) ^ { \top } R \psi ( s ^ { + } ) = \left( R ^ { \top } \psi ( s ) \right) ^ { \top } \psi ( s ^ { + } ) ,
$$

where $\psi ( s ) \in \mathbb { S } ^ { d - 1 }$ and R is orthogonal. Next, we relate $R ^ { \top } \psi ( s )$ to the expected representation of a future state sampled from $p _ { \hat { \pi } } ^ { \gamma } ( \cdot \mid s )$

Lemma 1 (Expected positive alignment). Assume that, at convergence, the contrastive representations are uniformly distributed on $\mathbb { S } ^ { d - 1 }$ , with $d \gg 1$ . The expected representation of a future state after a time interval sampledfrom a geometric distribution satisfies

$$
\mathbb { E } _ { s ^ { + } \sim p _ { \hat { \pi } } ^ { \gamma } ( \cdot | s ) } \left[ \psi ( s ^ { + } ) \right] \approx \rho R ^ { \top } \psi ( s ) ,
$$

where $\rho \in ( 0 , 1 ]$ is a coefficient shared across states.

Proof. The result follows directly from the proof of Lemma 2 in Appendix B.2.2 of Bastankhah et al. (2026), with $\phi ( s , a )$ replaced by $R ^ { \top } \psi ( \bar { s } )$ □

Lemma 1 identifies $R ^ { \top } \psi ( s )$ as the predicted direction after one geometric transition, while ρ captures the contraction in the expected representation. The following result extends this relation to $\bar { K _ { \mathbf { + } } } 1$ geometric transitions.

Lemma 2 (Recursive composition of geometric transitions). Let $s _ { 0 } ~ = ~ s$ and recursively sample $s _ { i } \sim p _ { \pi } ^ { \gamma } ( \cdot \mid s _ { i - 1 } ) f o r i = 1 , \cdot \cdot . . , K + 1$ . Then the expected representation of $\cdot _ { s _ { K + 1 } }$ satisfies

$$
\begin{array} { r } { { \mathbb E } _ { s _ { K + 1 } \sim p _ { \hat { \pi } } ^ { \gamma , K } ( \cdot \vert s ) } \left[ \psi ( s _ { K + 1 } ) \right] \approx \rho ^ { K + 1 } ( R ^ { \top } ) ^ { K + 1 } \psi ( s ) , } \end{array}
$$

where $\rho$ is the coefficient in Lemma $I .$

Proof. By definition, $s _ { K + 1 } \sim p _ { \hat { \pi } } ^ { \gamma , K } ( \cdot \mid s )$ . For the expected representation, the case $K = 0$ follows directly from Lemma 1. Suppose that $\vec { \Sigma } [ \psi ( s _ { K } ) \mid s _ { 0 } = s ] \approx \rho ^ { K } ( R ^ { \top } ) ^ { K } \psi ( s )$ . Assuming that the process is Markov with transition kernel $p _ { \hat { \pi } } ^ { \gamma }$ , the law of iterated expectations gives

$$
\begin{array} { r l } & { \mathbb { E } [ \psi ( s _ { K + 1 } ) \mid s _ { 0 } = s ] = \mathbb { E } [ \mathbb { E } [ \psi ( s _ { K + 1 } ) \mid s _ { K } ] \mid s _ { 0 } = s ] } \\ & { \qquad \approx \mathbb { E } \left[ \rho R ^ { \top } \psi ( s _ { K } ) \big \mid s _ { 0 } = s \right] } \\ & { \qquad = \rho R ^ { \top } \mathbb { E } [ \psi ( s _ { K } ) \mid s _ { 0 } = s ] } \\ & { \qquad \approx \rho R ^ { \top } \rho ^ { K } ( R ^ { \top } ) ^ { K } \psi ( s ) } \\ & { \qquad = \rho ^ { K + 1 } ( R ^ { \top } ) ^ { K + 1 } \psi ( s ) . } \end{array}
$$

The stated approximation follows by induction.

By Lemma 2, the expected representation of $s _ { K + 1 }$ is approximated by $\rho ^ { K + 1 } ( R ^ { \top } ) ^ { K + 1 } \psi ( s )$ . Since $\psi ( s )$ has unit norm and R is orthogonal, this vector has norm $\rho ^ { K + 1 }$ . Normalizing it therefore gives the predicted direction

$$
\mu _ { K } ( s ) \triangleq ( R ^ { \top } ) ^ { K + 1 } \psi ( s ) .
$$

We define its alignment with the goal representation as

$$
\begin{array} { r l } & { \ell _ { K } ( s , g ) \triangleq \mu _ { K } ( s ) ^ { \top } \psi ( g ) } \\ & { \qquad = \left( ( R ^ { \top } ) ^ { K + 1 } \psi ( s ) \right) ^ { \top } \psi ( g ) } \\ & { \qquad = \psi ( s ) ^ { \top } R ^ { K + 1 } \psi ( g ) . } \end{array}
$$

This recovers the score in Eq. (12), which measures how well the predicted direction for K intermediate waypoints aligns with the goal representation.

We now convert this alignment score into a distribution over K. Since $\mu _ { K } ( s ) \in \mathbb { S } ^ { d - 1 }$ , we model the goal representation with a von Mises–Fisher likelihood centered at $\mu _ { K } ( s ) $

$$
\begin{array} { r l } & { p ( \psi ( g ) \mid \psi ( s ) , K ) \triangleq C _ { d } ( \kappa ) \exp \left( \kappa \mu _ { K } ( s ) ^ { \top } \psi ( g ) \right) } \\ & { \hphantom { p ( \psi ( g ) ) } = C _ { d } ( \kappa ) \exp \left( \kappa \ell _ { K } ( s , g ) \right) , } \end{array}
$$

as in Eq. (13). Here, $\kappa > 0$ is a fixed concentration parameter. The normalizing constant $C _ { d } ( \kappa )$ depends only on d and κ and is therefore independent of both s and K.

For a prior $p ( K )$ over $\{ 0 , \ldots , K _ { \operatorname* { m a x } } \}$ , Bayes’ rule gives

$$
\begin{array} { r } { p ( K \mid \psi ( s ) , \psi ( g ) ) = \frac { p ( \psi ( g ) \mid \psi ( s ) , K ) p ( K ) } { \sum _ { J = 0 } ^ { K _ { \operatorname* { m a x } } } p ( \psi ( g ) \mid \psi ( s ) , J ) p ( J ) } } \\ { = \frac { \exp \left( \kappa \ell _ { K } ( s , g ) \right) p ( K ) } { \sum _ { J = 0 } ^ { K _ { \operatorname* { m a x } } } \exp \left( \kappa \ell _ { J } ( s , g ) \right) p ( J ) } , } \end{array}
$$

where the shared normalizing constant $C _ { d } ( \kappa )$ cancels.

With the uniform prior $p ( K ) = 1 / ( K _ { \operatorname* { m a x } } + 1 )$ , this reduces to

$$
p ( K \mid \psi ( s ) , \psi ( g ) ) = \mathrm { S o f t m a x } _ { K \in \{ 0 , \ldots , K _ { \operatorname* { m a x } } \} } \left( \frac { \ell _ { K } ( s , g ) } { \eta } \right) ,
$$

where $\eta = 1 / \kappa ,$ yielding Eq. (14).

## B ADDITIONAL EXPERIMENTAL RESULTS

## B.1 ABLATION STUDY

![](images/2cfd30a282e70c6b5a2ba86988bdf8a68749ebe32f4a112e291f395ef4f3c1c5.jpg)

![](images/3ce36b82b4f7319407535e1f334da4d0a7deb8cfc2ac81bd3290674a838c7c96.jpg)

![](images/e9d32aaf2c2f4ffbfb95a076c0d4c5749ea1e1101db084410730fd113f008d76.jpg)  
Figure 8: Ablation on the threshold $\epsilon _ { \bullet }$

![](images/81f091f7dbfaa147e41f77281039a4ee3c18125ba609e80e9ed9cfaff3cf283f.jpg)

Ablation on the threshold ϵ. We investigate the effect of the threshold ϵ used to select the waypoint index $i ^ { * }$ . As described in Eq. (15), a larger ϵ favors earlier waypoints, leading to more conservative subgoal selection. Figure 8 shows that C-Slerp performs well across multiple threshold choices, although sensitivity varies across tasks. Performance is relatively robust across $\epsilon \in \{ 0 . 2 , 0 . 3 , 0 . 4 \}$ in all evaluated environments except the 8-tower task. The 8-tower task is more sensitive to the threshold, with $\epsilon = 0 . { }$ 4 achieving the highest success rate, whereas $\epsilon = 0 . 2$ and 0.3 perform better on Ant U5 Maze. We use $\epsilon = 0 . 4$ as the default configuration for our algorithm across all tasks.

![](images/b59e36508244d11918a5b00dd63be9718286a2ea16f932d0bc8c0b0723f1e86c.jpg)  
Figure 9: Ablation on learning the rotation matrix R.

Ablation on learning the rotation matrix R. We investigate the role of learning the rotation matrix R in $f _ { H } ( s , s ^ { \bar { + } } ) ~ = ~ \psi ( s ) ^ { \top } R \psi ( s ^ { + } )$ by comparing it with fixed alternatives. As described in Section $4 . 2 ,$ we parameterize R through the Cayley transform, where $\Omega \ = \ A - A ^ { \top }$ and $R = ( I + \Omega ) ^ { - 1 } ( I - \mathrm { \bar { \Omega } } )$ . Using this parameterization, we compare our default approach of learning A with three fixed variants. The first sets $A = 0 ,$ yielding the identity matrix $R = I$ . The other two independently sample $A _ { i j } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with $\sigma \in \{ 0 . 0 1 , 0 . 0 2 \}$ . In all three variants, we stop gradients to $A ,$ keeping R fixed throughout training. As shown in Figure $^ { 9 , }$ setting $R = I$ makes $f _ { H }$ symmetric and prevents successful learning in the evaluated tasks. In contrast, the two variants with randomly initialized, fixed R introduce asymmetry and enable learning, but yield unstable performance. These results suggest that asymmetry alone is insufficient for stable performance, highlighting the benefit of learning R to capture temporal relationships between states.

![](images/12431fa6187223032c3c860a5d95d8ea29dbe83ff7308c6a8d8fe16881b11315.jpg)  
Figure 10: Ablation on network depth.

Ablation on network depth. Wang et al. (2026) report that increasing network depth improves the goal-reaching performance of CRL. Since C-Slerp also uses contrastive representations, we examine whether it benefits similarly from increased network depth. The results show that deeper networks generally achieve higher success rates. On Ant U5 Maze, performance improves as depth increases, while on the 8-tower task, depths of 16 and 32 perform comparably and outperform shallower networks, suggesting that sufficient network depth is needed for this task.

![](images/c7a47fcccff46fc58b384770e01c84d4baaf4ad0671c2fc2577fb5c3aff107ee.jpg)  
Figure 11: Ablation on representation dimension d.

Ablation on representation dimension. We investigate the effect of the representation dimension d. Since the learned rotation matrix R in $f _ { H } ( s , \bar { s } ^ { + } ) = \psi ( s ) ^ { \top } R \psi ( s ^ { + } )$ is a linear operator, we examine whether increasing the representation dimension d provides greater expressive capacity and improves performance. The results show that performance is relatively robust once the representation dimension is sufficiently large, with further increases providing no consistent benefit. In particular, on the 8-tower task, only dimensions of $d \geq 2 5 6$ achieve nonzero success among the tested values. We hypothesize that this limited benefit from further increasing d arises because the encoder ψ already provides sufficient nonlinear expressivity for the linear operator R to model temporal relationships effectively.

## B.2 CRL WITH LARGER DISCOUNT FACTORS

![](images/ced806b5552a29ea676a3d2eed16724eda920e357737c9ef7390dd795977a560.jpg)  
Figure 12: Comparison with CRL with larger discount factors $\gamma .$

Comparison with CRL using larger discount factors γ. As discussed in Section 3, CRL samples future states for contrastive learning using a temporal offset drawn from $\boldsymbol { \mathrm { G E O M } } ( 1 - \gamma )$ . Thus, a natural approach for adapting CRL to long-horizon tasks is to increase the discount factor γ.

We investigate this approach by evaluating CRL across $\gamma \in \{ 0 . 9 9 , 0 . 9 9 5 , 0 . 9 9 9 \}$ on long-horizon tasks within our benchmark, while C-Slerp uses $\gamma = 0 . 9 9$ . As shown in Figure 12, CRL achieves low success rates across the evaluated tasks, and increasing γ does not close the performance gap with C-Slerp. One possible explanation is that a larger γ places more weight on temporally distant future states. This may make it harder for the network to distinguish temporally close and distant goals, limiting the effectiveness of the learned representations. These results suggest that simply extending the temporal horizon of contrastive learning does not offer the same benefits as planning with an appropriate number of waypoints.

## B.3 VISUALIZATIONS

Evolution of K and $i ^ { * }$ over time. Figures 13 and 14 show the mean and sampled values of $K + 1$ together with the selected waypoint index $i ^ { * }$ , over a single episode in each environment. Since K denotes the number of intermediate waypoints, $K + 1$ represents the total number of transition segments from the current state to the goal. We plot $K + 1$ to align with the waypoint indexing, where the goal has index $K + 1$ , allowing direct comparison with the selected index $i ^ { * }$

In general, the mean value of $K + 1$ decreases as the agent approaches the goal, suggesting that the learned representation captures the remaining temporal distance to the goal. Notably, in Humanoid U Maze, the selected $K + 1$ values increase again after the humanoid reaches the goal, as it subsequently falls and cannot recover, causing its torso height to deviate from the target.

Visualizations of environment rollouts with corresponding representations. Figures 15 and 16 visualize the environment rollouts together with spherical PCA (Liu et al., 2019) projections of the corresponding latent representations. Across the evaluated environments, these visualizations illustrate how C-Slerp adjusts the number of waypoints and selects a latent subgoal for the policy to pursue as it progresses toward the goal.

Visualization of waypoint representations in maze environments. Figure 17 visualizes the inferred waypoints $z _ { 1 : K }$ and the scores $f _ { L }$ over the maze. We first encode all $( x , y )$ positions on a grid with 0.2 m spacing using the state encoder $\psi .$ . For each inferred waypoint $z _ { i } ,$ we retrieve its nearest neighbor among these representations and plot the corresponding position in the maze. We also evaluate $f _ { L } ( s , a , \bar { s ^ { \prime } } ) = \phi ( s , a ) ^ { \top } \psi ( s ^ { \prime } )$ at every grid point and display the resulting scores as a heatmap, providing a proxy for how easily the current policy can reach each position. The visualizations indicate that, although the waypoints are generated entirely in the latent space, they correspond to intermediate $( x , y )$ positions along the path toward the goal.

![](images/21ee7baabc40475bbc36d3f2cdb434446c6e1d482265f734187689f7397565d2.jpg)

![](images/f127f480db5eaaf0de3e5f3642897e698c43170bb87d12620017778d25d84a42.jpg)  
(a) Ant Big Maze

![](images/e1dd5b28b8e0245dfc603d5ecdb3676b0a641c69d624833cf535a92965646aeb.jpg)  
(b) Ant U4 Maze

![](images/1eaf1c896b1657be3becbbd9b70bb6b68fe7734be857d2a77e74037278d3e171.jpg)  
(c) Ant U5 Maze

![](images/9a9eba4dae8d8a3f240f70cfd107ebdcd10839f78bb8ad869599eb7b9bc70216.jpg)  
(d) Ant Extreme Maze

![](images/2ab661f87824542dc209094438fb781531870611f685bb81cf66e4157b71b54a.jpg)  
(e) Humanoid

![](images/6b363b3b1561eec789f7f57187d00b62de9007a2b6c2a2e6ef7c6e9233f6dc3a.jpg)  
(f) Humanoid U Maze  
Figure 13: Evolution of $K$ and $i ^ { * }$ over time in JaxGCRL environments. We plot the mean and sampled values of $K + 1$ , together with the selected waypoint index $i ^ { * }$ , against environment steps. As the agent approaches the goal, the predicted number of waypoints decreases.

![](images/4c7ad298f03bb8930cd8c76e3acaee93d09142db0e01d712013145f398ff688d.jpg)  
(a) 2-tower

![](images/174be1ee05c7e518bac5891097d1d00f77853a62ec1f095723c09da00cc7b193.jpg)  
(b) 8-tower

![](images/b128cff9d5d278d4f79269d320e6ad577aa8da4fc0ac8c14da2ad998c4ba8595.jpg)  
(c) 3-tower (gripper)

![](images/0233f87369cd37543c5b6b20455923c1e518613bcc93655db83f40a41bd41ceb.jpg)  
(d) 4-pyramid (gripper)

![](images/674ef00609cbe854ca2c84828f175500b36abcd1174f54859d379063c0494e19.jpg)  
(e) 4-tower (gripper)

![](images/3dac27b6e9caa2d7be25fb1b0baeac43b66073e325e5277924eb2741c0f014b2.jpg)  
Figure 14: Evolution of K and i<sup>∗</sup> over time in BuilderBench environments. We plot the mean and sampled values of $K + 1$ , together with the selected waypoint index $i ^ { * }$ , against environment steps. As the agent approaches the goal, the predicted number of waypoints decreases.

�(�)

![](images/ff3eb14d4391e739d83f188b5d9f82650bb5d57a19f693538fddf8cd90c500bb.jpg)

![](images/d32788cee6a671ba45d0250f2609acac81144080b6f5820408e0af1d5ca82b9f.jpg)

![](images/c68cb0171ef1759a4e5846992042373fffefc4ebd6f7f3d229d5a5342eaf2cb5.jpg)  
waypoint representations �<sub>1:�</sub>

![](images/a4cbcfb574071d4a976b88397ce82c8bf5ebf1add73220f105da924de77163ee.jpg)

![](images/c0de7edd02ba2a746c52102a3914c1856ddeb05a303d1746883adf767786027d.jpg)

![](images/6fac50b130f2b5f4724160b150c2364a270a53f69084c1c462c4fdbf9353a70c.jpg)

![](images/7616751c9d86ccdaf427c48bf255256228eba16d17ad33e8485834455619b21f.jpg)  
selected waypoint �<sub>i</sub>∗

![](images/515255d354320c0a06a769541e45fc55c147bacf0d6cb365b2739b326919e74c.jpg)

![](images/0c83eaf5980e4f41fc9ebed7f645c18ead256a945f7afa4d57446b0b545fb229.jpg)

![](images/4cc9e6abcc554d65bd1f2ea6afd98a026c99da2abfd74a09a57373396cba1b9d.jpg)

![](images/208e890af98e7c50859577a621ee2f69e5a78d37dced1ec281e0ae26c676afcb.jpg)  
(a) Ant Big Maze

![](images/d4d6f59e72c2e72639c20821dac222e5807dcf6e3f9e3e037868fa16ded4a964.jpg)

![](images/c9bf005d8c2e5c39797d948fd2aaa18a61b3cb0bf7df775e886555a017e739da.jpg)

![](images/8d191ffde33fd2b155d79ab62e0b32f76969dc7311988d1ca37fc017e82b1aa4.jpg)

![](images/0b0cc5ab0faf37336eae3737a05082e942c9d22700d404553dbc2a1aa1a881dc.jpg)

![](images/62699f8471af7c6a6ae66cbddb1c4bce232569bb63cbd75df2038f492c2fa229.jpg)

![](images/6b742bd6f1cae3c53ff6a717f394494b55194f3107f017fa78fb4e84a1e77557.jpg)

![](images/76b8df2ed6ddcdc8edcd26fa4cb80a36f84f4533a1f853ab92b861628e2a9abe.jpg)

![](images/dd6e5113e89b9090a1a2afed63e5a14e0bd33a1cf9057a13aa629482403b5abc.jpg)

![](images/93ecddcfd20ea8f69c1e132081c14c3ac23a4ac43e5f4f5e3aa3c85cb535e0e5.jpg)

![](images/a90a7da81931e108728c9fa46488623804f36baa5fce6835479892d782f614b9.jpg)  
(b) Ant U4 Maze

![](images/1c8b42486f8019da40a11c514cbb2010f8f52d63e34ed15e9cf0baf5786ff92b.jpg)

![](images/b8ca2891b39df2403c6ecb4e4cd7a13f803f981e0208a8475e18c8249bcb5177.jpg)

![](images/15b369c0498b962c1873c7eedae5e8a7bba6db590f3b0e8a3b111273a9724ab4.jpg)

![](images/c0fa361e5a32797402e7c747ecd37b237ec7864e95b67300a08f4f37eddf036b.jpg)

![](images/dd455719989c2b97a55b9fb69724c4d2f3d154863aa25f7338ab50d9361c9f59.jpg)

![](images/04c8a1579f80a810f1b3858ff1f61c19682209a93071a577a3b73f5c3c6fc3ac.jpg)

![](images/cac1473627c9895dda56563e673b6aaee69dd646c76eb61a39ee3c809000b1d6.jpg)

![](images/304ff184e2a3a3e1212df3b9c26fd9cd50cdacaca82661e0888da99bdd94d2c6.jpg)

![](images/675de455cfcf71488d8af3416c87d26a1f95131485e43ebda2ee91a8062b41ce.jpg)

![](images/1e9a7790a537e4389276b6b7a63c4e3b400e121bf2380b1575c3f327520d9b83.jpg)  
(c) Ant U5 Maze

![](images/eab40ca032da562e8f76bb1011353fa5c29d59dad257af2b511807cd343efae7.jpg)

![](images/dedbd95248e8d2d7efb8c1c10d69885458ccd19e01a8c8dda3c44bcbe8cb1ed1.jpg)

#

(d) Ant Extreme Maze

![](images/28cecc6de43b8156a4e9401788458b0854299c68809abf5f379ba0de9144b03d.jpg)

![](images/8be820aa2bc60cb0a8ca67bd08c96abfd1b107e0ac2202b208648b32be6adbe2.jpg)

![](images/39e7b192b4494ecd6a0b02670d4f133cccc17711ce2d703aaece26b9c71d8c66.jpg)

![](images/ddd14973be23ad9026868aec649ce14dc0daeae1ef67b593da34a2045b7ddcb3.jpg)

![](images/246c82ffe1dfbacf024bae973571166605245bb0ddc392f6872b2ca14d5bf5e3.jpg)

![](images/9865e1dabd820cd387608214ea052fda21a70aee29d02481ffc916f02f16fd6b.jpg)

![](images/ca36ca672c8e517581c95baf80519c34c31c53eb0cb8efcc267766bcdf8e1ecc.jpg)  
(e) Humanoid

![](images/6e8cd22487f51eb9887510f6646b1d4fa162f209339f7ac004406f5ff4dc333e.jpg)

![](images/b86509f5f18c65183e0cda1f6eecec0cc0fd13f86bfb393e5e51ccfcc7a732ad.jpg)

![](images/971ef0cc4537996d82fc1e944969f1691907b4ecf27ed6d6c687116a0c1769e1.jpg)

![](images/753bd29daafcdb29ede3c221b1e5e32a2a02fd1e7bfffc3808a10e6f932a8750.jpg)

![](images/68419383baa49aa92ba5af0c450c9edc8328e1f6d30d98b773878bac025522a0.jpg)

![](images/670a4d1672cfd5ab694f43f5e854a284d5dd3265c7d61d8a45042d83285aba35.jpg)

![](images/1ae9e770b9b80e14aa24f2589774706cf36b43955c85991218fa01bf15051ee9.jpg)

![](images/f0a07f07af5f882d6a3e7066744a8589e4f5ed031b1f90b9aec433a2c3039739.jpg)

![](images/b006b0df9a9050b186575d50aaddb8e29f1d044497937a9a29c19301868e3740.jpg)

![](images/042595351bf727f913e0740fe02e92d1ffeb2d0399c2ab47a71c3aa28d17522a.jpg)

![](images/7dabfe7f7745f9baf9506c02cebf9daa21be14e63b396ae6753dd055277adaa3.jpg)  
(f) Humanoid U Maze

![](images/57bc74f974006fe67a464eeceefa90d9846d6c5851066037ddb6a2857b4d9412.jpg)

![](images/3742a69ffd8c519d219ac7c3f0ff500db8d8864b286bc38ad437605c976c0b53.jpg)  
Figure 15: Visualization of C-Slerp rollouts and corresponding latent representations in JaxGCRL. For each environment, the top row shows successive states, and the bottom row shows spherical PCA (Liu et al., 2019) projections of the corresponding state representations and inferred waypoints.

waypoint representations �<sub>1:�</sub>

selected waypoint �<sub>i</sub>∗

![](images/c2d35897e07503f7a091526ec836e15720d58637a6a0b13730b423dc87c13d10.jpg)  
(a) 2-tower

![](images/ae88067384d0d3bdca111a2fd5311dd4c23cc40a924086ce76aabd3c59c4a99c.jpg)  
(b) 8-tower

![](images/a52e980eb1622b237ee5a6a2d483d8b3337ec4841c728418bda801a77563c932.jpg)  
(c) 3-tower (gripper)

![](images/c22ec0f76a36a932327b7820de791d2c65dea90f9834b5562666401c084075d2.jpg)  
(d) 4-pyramid (gripper)

![](images/c3d8cf3e5b5f487f2a53bc066c2061fa3349f0eb5ff7584f917efc8166b9f344.jpg)  
(e) 4-tower (gripper)

![](images/5184190a34bfe9ad6f4719d4f8b60b7979ce09a600e5eae9e24c8c4328486808.jpg)  
(f) 5-tower (gripper)  
Figure 16: Visualization of C-Slerp rollouts and corresponding latent representations in BuilderBench. For each environment, the top row shows successive states, and the bottom row shows spherical PCA (Liu et al., 2019) projections of the corresponding state representations and inferred waypoints.

0.5

0.0

![](images/27bc2cf61667c9f770b91773fd9e73b9a8c3a2efe7d4f3fa7abbe6b46a5feaf3.jpg)

![](images/54224cd647b3e75286ad692255e1bb088f90b88014e8f3f72fc29b017abba919.jpg)

![](images/b0c8e2b64ee399915bc10cc4e71bb55ce878755a8c925006fbe762e9ce8e16ad.jpg)

![](images/049963b8accf830ecc538a65b98f3565a6818b143c8a3677b048aa7efabcbd89.jpg)

![](images/e9199e4faaafe9bf37ae082912a4297f84d8038123f925949bc2d5ec1a0cbb08.jpg)  
(a) Ant Big Maze

![](images/5dbe16ea55714bf7f451e2cd6d83d18a6484b4c4773d265262f05749ccf15e71.jpg)

![](images/aa70eb60a75d94bd4d77ed2479ce5bdfa446baedea33be73620c24a6ed1c70ee.jpg)

![](images/3fa8a5b918155ed81f0cc22b5a228876b2c6e3b8aa132fedb09c4111c6a4523c.jpg)

![](images/aea019701965ad7d9cdc2ad174a3a02724302a9f8a6bb863ab93f7dab1079efb.jpg)

![](images/6ffb2c2329c81121c27fe59dec172fe475f286bb50ea40726d8e83b1fef65265.jpg)

![](images/03ee41990f9a40aa101395d39f3a1badb48e4b527a74aa6407ca6fbc368e372e.jpg)  
(b) Ant U5 Maze

![](images/5eb588ba27c9c004ed70de51b904d7cd8cd99aca4f2d65295ecb9a334e004948.jpg)

#

(c) Ant Extreme Maze

Figure 17: Visualization of waypoint representations in maze environments. We visualize the inferred latent waypoints $z _ { 1 : K }$ and the corresponding score heatmaps. For each environment, the top row shows the $( x , y )$ positions corresponding to the inferred latent waypoints, where denotes a candidate waypoint $z _ { i } , \bigcirc$ highlights the selected waypoint $z _ { i ^ { * } }$ , and $\bigcirc$ represents the goal. The bottom row shows heatmaps of $f _ { L } ( s , a , s ^ { \prime } ) = \phi ( s , a ) ^ { \top } \bar { \psi } ( s ^ { \prime } )$ , with the current state marked by and the goal indicated by $\bigstar$

![](images/42d5df820595db517c682ce21ca37c5fe034f9871a6317a1d55c97ca005137e6.jpg)  
Figure 18: Visualization of benchmark environments. The top row shows the BuilderBench manipulation tasks, and the bottom row shows the JaxGCRL navigation tasks. The goal used for evaluation in each JaxGCRL environment is indicated by . We provide videos demonstrating C-Slerp on all benchmark tasks on our project page.

## C EXPERIMENTAL DETAILS

We implement C-Slerp on top of the publicly available JaxGCRL (Bortkiewicz et al., 2025) and BuilderBench (Ghugare et al., 2025) codebases. For baselines, we implement CRL (Eysenbach et al., 2022), RIS (Chane-Sane et al., 2021), an online version of HIQL (Park et al., 2023), L3P (Zhang et al., 2021a), SAC+HER (Haarnoja et al., 2018a; Andrychowicz et al., 2017) on the same codebase as C-Slerp. We run our experiments on a local machine consisting of A6000 GPUs, and each run in Section 5 takes 12-72 hours depending on the environment. All training metric curves are averaged over 5 seeds, and the shaded region denotes one standard error. We provide the environment and evaluation details in Appendix C.1, and the implementation details in Appendix C.2.

## C.1 ENVIRONMENTS

JaxGCRL. We evaluate our method on goal-conditioned navigation tasks in JaxGCRL, including Humanoid, Humanoid U Maze, Ant Big Maze, Ant U4 Maze, and Ant U5 Maze. To extend our evaluation to substantially longer-horizon problems, we additionally design Ant Extreme Maze, inspired by Myers et al. (2025).

The states consist of the torso’s global position and proprioceptive observations, including joint positions and velocities. The goals specify the desired torso position, and the actions correspond to joint-control commands. For Ant and Humanoid, respectively, the states have 29 and 268 dimensions, the goals indicate the desired global torso coordinates (x, y) and (x, y, z), and the actions have 8 and 17 dimensions. For the maze tasks, training goals are sampled uniformly from valid goal locations, whereas evaluation uses the goals shown as markers in Figure 18. The maximum episode length is 1,000 environment steps, except in Ant Extreme Maze, where the limit is 2,000 steps.

BuilderBench. We evaluate our method on gripper-based manipulation tasks with three to five cubes in BuilderBench. Each environment consists of N cubes and a PD-controlled gripper. The gripper state consists of its 3-dimensional position, 4-dimensional orientation, 3-dimensional linear velocity, and 2 finger positions, giving 12 dimensions. Each cube contributes 13 dimensions consisting of its 3-dimensional position, 4-dimensional orientation, 3-dimensional linear velocity, and 3-dimensional angular velocity, yielding a (12 + 13N)-dimensional full state. Goals consist of the desired (x, y, z) position of each cube and are 3N-dimensional. The action space has 5 dimensions, controlling gripper opening, Cartesian motion, and yaw. At each reset, the initial cube positions are randomized, and a randomly sampled (x, y) offset is added to the desired position of the goal structure. To accommodate the longer manipulation sequences required as the number of cubes N increases, we use 100 + 100N environment steps per episode during training and 200 + 200N environment steps per episode during evaluation.

Following Yan et al. (2026), we additionally evaluate a modified BuilderBench environment in which the policy selects a cube and applies force and torque directly to it without using a gripper. Each cube has the same 13 pose and velocity components as in the gripper environment, and an additional scalar indicates the selected cube, yielding a $( 1 + 1 3 N )$ )-dimensional state. Goals specify the desired $( x , y , z )$ position of each cube and are 3N-dimensional. Each 5-dimensional action is represented as $( f _ { x } , f _ { y } , f _ { z } , T _ { z } , i )$ , where $( f _ { x } , f _ { y } , f _ { z } )$ denotes the applied force, $T _ { z }$ denotes the yaw torque, and i identifies the cube receiving the actuation. As in the gripper environment, the initial cube positions are randomized, and the same randomly sampled $( x , y )$ offset is added to the desired position of the goal structure. To account for the longer manipulation sequences required as the number of cubes increases, we set the maximum episode length to $1 0 0 + 5 0 { \bar { N } }$ environment steps during both training and evaluation.

Metrics. For both JaxGCRL and BuilderBench, we report the fraction of evaluation episodes in which the agent meets the success criterion for each task. In JaxGCRL environments, success is achieved when the torso is within 0.5 m of the goal position. In BuilderBench environments, success requires all cubes to be within 0.01 m of their target heights and the structure to lie within 0.04 m of its target (x, y) position. Each cube has an edge length of 0.04 m.

For the four-cube pyramid task, we also require the horizontal arrangement of the cubes to match the target structure. We measure each cube’s horizontal position relative to the center of the bottom layer and compare it with the corresponding position in the target structure. The allowed error is approximately 0.01 m for bottom-layer cubes and 0.03 m for upper-layer cubes. The center of the bottom layer must also be within 0.04 m of its target position.

## C.2 IMPLEMENTATION DETAILS

During the learning process, we adjust the entropy coefficients α to match the target entropies using the automatic tuning procedure of SAC (Haarnoja et al., 2018a). The number of waypoints $K$ is sampled from the softmax distribution in Eq. (14) with a temperature of $\eta = 0 . 1$ during both data collection and evaluation. Separately, to control the sharpness of the softmax distributions in the InfoNCE loss, $f _ { H }$ and $f _ { L }$ are divided by a temperature of $\tau _ { \mathrm { N C E } } = 0 . 1$ during training. We also adopt the sampling strategy of Ziarko et al. (2026), which introduces in-trajectory negatives by sampling multiple positive pairs from each trajectory, with the number of pairs controlled by a repetition factor. For BuilderBench, we set the repetition factor to 12. For JaxGCRL, we use the default sampling strategy implemented in the codebase. We provide the hyperparameters of C-Slerp in Table 1 and the source code on our project page.

Algorithm 1 provides the full pseudocode for C-Slerp. Every h environment steps, the high-level planner determines the number of waypoints and selects a subgoal for the low-level policy to pursue for the next h steps. The resulting transitions are stored in the replay buffer and used to update the representations and policy.

Baseline methods. We implement all baselines within the same training and evaluation framework as our method. We closely follow each method’s original training objectives and algorithm-specific components while using shared infrastructure and matching the capacity of the main networks where appropriate.

For CRL (Eysenbach et al., 2022), we closely follow the implementation provided in each benchmark’s codebase, retaining its network architecture and L2-based energy function, while tuning the learning rates.

For RIS (Chane-Sane et al., 2021), we follow the original diagonal Laplace subgoal parameterization and goal-relabeling scheme. We use the rollout goal, an achieved goal sampled independently from the replay buffer, and a future achieved goal from the same trajectory with probabilities 0.2, 0.4, and 0.4, respectively. Future achieved goals are sampled uniformly from later states within the same trajectory. We draw 10 samples from the imagined-subgoal prior and use the resulting subgoals only during training. At evaluation time, the policy is conditioned directly on the final task goal.

For HIQL (Park et al., 2023), we adapt the original offline RL algorithm to our online setting by replacing the fixed offline dataset and pretraining stage with trajectories collected through online interaction and updates from the resulting replay buffer. We preserve the action-free expectile valuelearning objective, the high- and low-level advantage-weighted policy objectives, and the latent subgoal prediction used by the hierarchical policy. For value learning, we use the current state, a future achieved goal from the same trajectory, and an achieved goal sampled independently from the replay buffer with probabilities 0.2, 0.5, and 0.3, respectively. Future achieved goals are sampled from later states within the same trajectory according to a geometric distribution with parameter $1 - \gamma$ . For the high-level objective, we sample an achieved goal either from the future of the same trajectory or independently from the replay buffer with probabilities 0.7 and 0.3, respectively. The target horizon K determines how subgoals are selected during training. The low-level policy is trained to reach a state selected from the next K steps of the same trajectory, while the high-level policy is trained to generate subgoals using targets selected within the same horizon. We set $K = 2 5$ in JaxGCRL and $K = 1 0$ in BuilderBench.

For SAC+HER (Haarnoja et al., 2018a; Andrychowicz et al., 2017), goal relabeling uses a future achieved goal from the same trajectory with probability 0.8 and the original rollout goal with probability 0.2. Future achieved goals are sampled uniformly from later states within the same trajectory, and rewards are recomputed with respect to the resulting goal. Goal relabeling does not alter the original environment termination or discount signal.

For L3P (Zhang et al., 2021a), the training-time planning graph contains 50 learned landmarks and 150 additional replay-based landmarks, whereas evaluation uses only the 50 learned landmarks. Landmark pairs are scored using directed reachability estimates, so the retained graph topology depends on the learned values and the distance-clipping threshold rather than on a fixed edge set. L3P’s hindsight relabeling uses a uniformly sampled future achieved goal from the same trajectory with probability 0.85, with the future state restricted to at most 100 steps ahead. The graph is refreshed every $\mathrm { 5 \times 1 0 ^ { 5 } }$ environment interactions, and planning during data collection is enabled after $5 \times 1 0 ^ { 6 }$ interactions. To improve robustness in the online setting, each learned latent component is represented by a state selected from the replay buffer rather than directly by a decoded latent centroid. This ensures that planning landmarks correspond to states observed in the environment and avoids centroids that may fall in invalid or unreachable regions.

For model architecture, CRL, RIS, SAC+HER, and L3P use the same backbone as our method for their main policy and value-learning networks. For HIQL, we preserve its component-specific structure. The representation and value networks use three hidden layers, the high-level policy uses two hidden layers, and the low-level policy matches the number of hidden layers in the policy used by C-Slerp for each benchmark. To improve performance, both HIQL policies use half as many units per hidden layer as our method’s policy. HIQL uses a 10-dimensional latent subgoal representation, and L3P uses a 16-dimensional latent space. Unless otherwise specified, the baselines follow the same shared training settings as our method.

For baseline tuning, we use settings reported in the original papers and implementations as starting points and vary learning rates and method-specific parameters to identify the best-performing configuration for each baseline and benchmark. Table 2 summarizes the candidate values considered during tuning. Once selected, each configuration is fixed for all final evaluations. Tables 3–7 report the selected method-specific parameters used in our experiments.

Algorithm 1 Contrastive Planning via Spherical Linear Interpolation (C-Slerp)   
1: Initialize policy $\pi _ { \theta } ( a \mid s , z )$ , encoders $\phi ( s , a )$ and $\psi ( s )$ , rotation matrix R, replay buffer D   
2: for each training epoch do   
// Data collection   
3: for each episode do   
4: $t \gets 0$   
5: Sample an initial state $s _ { 0 } \sim p _ { 0 }$ and a goal $g \sim p _ { g }$   
6: while the episode is not finished do   
7: if t % h = 0 then   
// K-selection   
8: Sample K for $( s _ { t } , g )$ using Eq. (14)   
9: for $i = 1 , \ldots , K$ do   
10: Compute $( \mu _ { i } ^ { \star } , \sigma _ { i } ^ { \star 2 } )$ using Eq. (11)   
11: Sample $\tilde { z } _ { i } \sim \mathcal { N } ( \mu _ { i } ^ { \star } , \sigma _ { i } ^ { \star 2 } I )$   
12: Set $z _ { i } = \tilde { z } _ { i } / \| \tilde { z } _ { i } \| _ { 2 }$   
13: end for   
14: Set $z _ { K + 1 } = \psi ( g )$   
// i-selection   
15: for $i = 1 , \ldots , K + 1$ do   
16: Sample $\tilde { a } _ { i } \sim \pi _ { \theta } ( \cdot \mid s _ { t } , z _ { i } )$   
17: Compute $\rho _ { i } = \phi ( s _ { t } , \tilde { a } _ { i } ) ^ { \top } z _ { i }$   
18: end for   
19: if $\mathrm { m a x } _ { i \in \{ 1 , \dots , K + 1 \} } \rho _ { i } \geq \epsilon$ then   
20: Select $i ^ { \star }$ using Eq. (15)   
21: else   
22: Set $i ^ { \star } = 1$   
23: end if   
24: Set the subgoal $z = z _ { i ^ { \star } }$   
25: end if   
// Policy rollout   
26: Sample $a _ { t } \sim \pi _ { \theta } ( \cdot \mid s _ { t } , z )$   
27: Execute $a _ { t }$ and observe $s _ { t + 1 }$   
28: Store $\left( { { s _ { t } } , { a _ { t } } , { s _ { t + 1 } } } \right)$ in D   
29: $t \gets t + 1$   
30: end while   
31: end for   
// Representation and policy learning   
32: for each gradient step do   
33: Sample a batch $\dot { \boldsymbol { B } } = \{ ( s _ { j } , a _ { j } , s _ { j } ^ { + } ) \} _ { j = 1 } ^ { B }$ from D   
34: Update ϕ, ψ, R by minimizing ${ \mathcal { L } } _ { \mathrm { r e p r } }$ in Eq. (16)   
35: Update π<sub>θ</sub> using Eq. (17)   
36: end for   
37: end for

Table 1: Learning hyperparameters.
<table><tr><td>Parameter</td><td>JaxGCRL</td><td>BuilderBench</td></tr><tr><td>Actor learning rate</td><td> $\overline { { 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 1 0 ^ { - 4 } } }$ </td></tr><tr><td>Representation learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Actor entropy learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Waypoint entropy learning rate</td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Batch size</td><td>512</td><td>1024</td></tr><tr><td>Representation dimension d</td><td>128</td><td>512</td></tr><tr><td>Network depth</td><td>8</td><td>16</td></tr><tr><td>Network width</td><td>512</td><td>1024</td></tr><tr><td>Discount factor γ</td><td>0.99</td><td>0.99</td></tr><tr><td>InfoNCE temperature τNCE</td><td>0.1</td><td>0.1</td></tr><tr><td> $\lambda _ { \mathrm { { r e p r } } }$ </td><td>4.0</td><td>2.0</td></tr><tr><td>Control horizon h</td><td>10</td><td>10</td></tr><tr><td>Maximum planning stages  $K _ { \operatorname* { m a x } } + 1$ </td><td>5 (10 for Ant Extreme Maze)</td><td>4</td></tr><tr><td>K sampling temperature η</td><td>0.1</td><td>0.1</td></tr><tr><td>i-selection threshold €</td><td>0.4</td><td>0.4</td></tr><tr><td>In-trajectory repetition factor</td><td></td><td>12</td></tr><tr><td>Policy target entropy</td><td>-|A|</td><td>-2|A| (—|A| with gripper)</td></tr><tr><td>Waypoint target entropy</td><td>−2.5(d − 1)</td><td> $- 3 . 5 ( d - 1 )$   $( - 3 . 0 ( d - 1 )$  with gripper)</td></tr></table>

Table 2: Hyperparameter search spaces for baseline tuning.
<table><tr><td>Method</td><td>Parameter</td><td>Search space</td></tr><tr><td>CRL RIS</td><td>Actor learning rate Critic learning rate Actor learning rate</td><td> $\{ 3 \times 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } \}$   $\left\{ 3 \times 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } \right\}$   $\left\{ 3 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \right\}$ </td></tr><tr><td></td><td>Critic learning rate KL coefficient Prior samples</td><td> $\{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$   $\{ 0 . 1 , 0 . 3 \}$   $\{ 3 , 1 0 \}$ </td></tr><tr><td>SAC+HER Online HIQL</td><td>Actor/critic learning rate Target entropy Learning rate</td><td> $\overline { { \{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \} } }$   $\tilde { \{ } - 0 . 5 | \mathcal { A } |  , - 1 | \mathcal { A } | , - 2 | \mathcal { A } | \}$   $\overline { { \{ 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \} } }$ </td></tr><tr><td></td><td>High-level coefficient  $\beta _ { \mathrm { h i g h } }$  Low-level coefficient  $\beta _ { \mathrm { l o w } }$  Target horizon K Direct-goal bootstrap fraction Random-action warm-up</td><td> $\{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 5 , 1 . 0 \}$   $\{ 0 . 3 , 0 . 4 , 0 . 5 , 1 , 1 0 0 \}$  {3, 5, 10, 25} {0, 0.025} {0, 1000, 2000}  $\overline { { \{ 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \} } }$ </td></tr><tr><td>L3P</td><td>Actor/critic/value learning rate Discount factor γ Action regularization Learned landmarks Distance-clipping threshold Graph refresh interval Exploration noise Random-action probability</td><td>{0.98, 0.99} {0.05,0.1} {50, 100} {−20, −15}  $\{ 1 0 ^ { 5 } , 5 \times 1 0 ^ { 5 } \}$  {0.1,0.2} {0.05, 0.2}</td></tr></table>

Table 3: CRL-specific parameters.
<table><tr><td>Parameter</td><td>JaxGCRL</td><td>BuilderBench</td></tr><tr><td>Actor learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $\overline { { 3 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td>Critic learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Energy function</td><td> $\mathbf { L } 2 \mathrm { n o r m }$ </td><td> $\mathbf { L } 2 \mathbf { n o r m }$ </td></tr></table>

Table 4: RIS-specific parameters.
<table><tr><td>Parameter</td><td>JaxGCRL</td><td>BuilderBench</td></tr><tr><td>Actor/subgoal learning rate</td><td> $\overline { { { 1 0 } ^ { - 4 } } }$ </td><td> $\overline { { { 1 0 } ^ { - 4 } } }$ </td></tr><tr><td>Critic learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>KL coefficient</td><td>0.1</td><td>0.1</td></tr><tr><td>Bottleneck temperature</td><td>0.1</td><td>0.1</td></tr><tr><td>Prior samples</td><td>10</td><td>10</td></tr></table>

Table 5: SAC+HER-specific parameters.
<table><tr><td>Parameter</td><td>JaxGCRL</td><td>BuilderBench</td></tr><tr><td>Actor/critic learning rate</td><td> $\overline { { { 1 0 } ^ { - 4 } } }$ </td><td> $\overline { { { 1 0 } ^ { - 4 } } }$ </td></tr><tr><td>Entropy learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>HER probability</td><td>0.8</td><td>0.8</td></tr><tr><td>Initial entropy coefficient</td><td>1.0</td><td>0.01</td></tr><tr><td>Target entropy coefficient</td><td>-0.5</td><td>-1.0</td></tr></table>

Table 6: Online HIQL-specific parameters.
<table><tr><td>Parameter</td><td> $\overline { { \mathrm { J a x G C R L } } }$ </td><td>BuilderBench</td></tr><tr><td>Learning rate</td><td> $\overline { { 3 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 1 0 ^ { - 4 } } }$ </td></tr><tr><td>Expectile</td><td>0.7</td><td>0.7</td></tr><tr><td> $\beta _ { \mathrm { h i g h } }$ </td><td>0.2</td><td>0.5</td></tr><tr><td> $\beta _ { \mathrm { l o w } }$ </td><td>0.3</td><td>100</td></tr><tr><td>Target horizon K</td><td>25</td><td>10</td></tr><tr><td>Direct-goal bootstrap fraction</td><td>0</td><td>0</td></tr><tr><td>Random-action warm-up</td><td>0</td><td>0</td></tr><tr><td>Representation dimension</td><td>10</td><td>10</td></tr></table>

Table 7: L3P-specific parameters.
<table><tr><td>Parameter</td><td>JaxGCRL</td><td>BuilderBench</td></tr><tr><td>Actor/critic/value learning rate</td><td> $\overline { { 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 1 0 ^ { - 4 } } }$ </td></tr><tr><td>Autoencoder/cluster learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Target-network update period</td><td>10 learner steps</td><td>10 learner steps</td></tr><tr><td>Latent dimension</td><td>16</td><td>16</td></tr><tr><td>Action regularization</td><td>0.05</td><td>0.05</td></tr><tr><td>Learned landmarks</td><td>50</td><td>50</td></tr><tr><td>Additional training landmarks</td><td>150</td><td>150</td></tr><tr><td>Graph refresh interval</td><td> $5 \times 1 0 ^ { 5 }$ </td><td> $5 \times 1 0 ^ { 5 }$ </td></tr><tr><td>Planning activation</td><td> $5 \times 1 0 ^ { 6 }$ </td><td> $5 \times 1 0 ^ { 6 }$ </td></tr><tr><td>Distance-clipping threshold</td><td>-20</td><td>-20</td></tr><tr><td>Soft-planning temperature</td><td>1.1</td><td>1.1</td></tr><tr><td>Value-iteration steps</td><td>20</td><td>20</td></tr><tr><td>Local planning horizon</td><td>10</td><td>10</td></tr><tr><td>Exploration noise</td><td>0.2</td><td>0.2</td></tr><tr><td>Random-action probability</td><td>0.2</td><td>0.2</td></tr></table>