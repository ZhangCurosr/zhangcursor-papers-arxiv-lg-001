# HALight: Heterogeneous Multi-Agent Reinforcement Learning for Traffic Signal Control under Lane-Level Missing Observations

Chenkai Hu, Yingqian Zhang, Ming Xie, Yi Fang, Chengxiang Jin, Chao Li, Member, IEEE

Abstract—Multi-agent reinforcement learning has shown great promise for traffic signal control. However, most existing methods rely on two key assumptions: (i) agents are assumed to be homogeneous with unified observation/action spaces; (ii) full and reliable lane-level traffic observations are available to all agents. In practice, urban road networks exhibit substantial structural heterogeneity (e.g., lane configurations and phase designs), while sensor failures and incomplete detector coverage often result in missing incoming lane measurements. Such issues can severely impair coordination and destabilize learning. To address these challenges, we propose HALight, a communication-enhanced heterogeneous multi-agent reinforcement learning framework for traffic signal control under lane-level missing observations. HALight supports intersection-specific observation and action modeling to accommodate heterogeneous intersections, and incorporates a lightweight one-hop neighborhood communication module to enrich spatial context when lane-level observations are partially unavailable. Moreover, HALight adopts a sequential policy update scheme to improve coordination among heterogeneous agents. Experimental results on two real-world road networks demonstrate that HALight outperforms representative homogeneous and heterogeneous baselines in terms of traffic efficiency and convergence behavior. The code and datasets used in this work are available at https://github.com/Huchenkaii/HALight.

Index Terms—Traffic signal control, multi-agent reinforcement learning, heterogeneous-agent reinforcement learning, traffic networks.

## I. INTRODUCTION

RAFFIC signal control (TSC) plays a crucial role in reducing traffic congestion [1] and carbon emissions [2] by optimizing signal timing and phase switching. Traditional methods are generally model-driven, relying on parametric formulations to minimize vehicle delay and queue length [3], [4]. These approaches typically depend on explicitly specified traffic models and handcrafted objective functions. For instance, MaxBand [5] computes signal timings by optimizing progression bandwidth using predefined traffic parameters. While such methods have achieved success in controlled settings, their applicability becomes limited in urban road networks [6]. It is difficult to model the complex and highly dynamic traffic environment arising from interactions among multiple intersections. As a result, traditional approaches often struggle to adapt to time-varying multi-intersection conditions.

With the rapid progress of reinforcement learning (RL) and deep neural networks (DNNs), deep reinforcement learning (DRL) has emerged as a data-driven paradigm for TSC [7], [8], [9]. Existing multi-agent reinforcement learning (MARL) methods directly learn control policies through trial-and-error interactions [10], [11]. By assigning an individual learning agent to each intersection, decentralized approaches have become a popular solution for multi-intersection control. TeDA-GCRL [12] explicitly models lane-level interactions via graph convolution, where each lane is treated as a graph node to capture relations of intersections. CoordLight [13] introduces Queue Dynamic State Encoding (QDSE) to characterize queue dynamics. Despite their success, decentralized MARL methods for TSC are constrained by two key assumptions that limit their applicability in real-world road networks. (i) Most methods assume homogeneous agents with shared observation and action spaces across intersections [14], [15]. While this simplification aids learning and coordination, it fails to account for the structural differences in intersection geometry, lane configuration, and phase design. As illustrated in Fig. 1, we identify two common types of heterogeneity: the variation across different intersection types and the structural heterogeneity within the same intersection type. Consequently, the learned policies tend to favor an “average” behavior, which is suboptimal for intersections with distinct characteristics. (ii) Existing methods typically assume full lane-level observations [16], [17]. In practical deployments, sensor failures and incomplete or spatially varying sensing coverage can lead to missing incoming-lane measurements, which may persist over a control horizon or occur intermittently [18]. For example, the effective communication coverage of electronic toll collection (ETC) roadside units (RSUs) is determined by the spatial field distribution of the RSU antenna and can vary across roadway locations and lanes [19]. As illustrated in Fig. 2, such nonuniform coverage may result in unavailable traffic observations over particular lane segments. The lack of reliable lane-level information limits agents’ adaptation to local conditions. Combined with homogeneous agent modeling, it further degrades coordination performance.

Such limitations motivate the development of heterogeneous multi-agent TSC under lane-level missing observations. However, developing this model presents two primary challenges:

![](images/e051cabb93f8af05c74930021f9f09546ac498136a74646ee6032db51d34c12e.jpg)  
Heterogeneity across different intersection types

![](images/d6474986205af2c94a293baae80cf771857fe2ec62e0d3112d13c2be3d7309e8.jpg)  
(b)  
Fig. 1. Illustration of two prevalent forms of intersection heterogeneity in real-world road networks: (a) heterogeneity across different intersection types, such as crossing intersections and T intersections, (b) heterogeneity among intersections of the same type, characterized by variations in lane configurations, geometric layouts, and phase designs.

![](images/c5ed5ad9be2c70de3ef57cc8f0775736f200dc681bdacf0614b9bfe7543bbe51.jpg)  
Fig. 2. Illustration of lane-level observation unavailability under non-uniform ETC sensing coverage.

• How to model the observation and action spaces across heterogeneous intersections? In real-world road networks, there exist substantial variations in intersection types and configurations. The variations lead to significant differences in the observation and action spaces across agents. Most heterogeneous methods focus on the heterogeneity between crossing and T intersections [20], [21]. Therefore, these methods are restricted to specific types of road networks and fail to adapt to real-world conditions.

• How to enable heterogeneous agents to remain effective under different lane-level observation missing patterns? Each intersection may experience unique missing observation patterns. Such missing patterns can vary both across intersections and within the same intersection. The diversity of missing patterns, combined with the heterogeneity between intersections and agents, complicates the learning process and makes it more challenging for the multi-agent system to adapt to varying local observations.

To solve these issues, we propose HALight, a heterogeneous MARL framework for TSC under lane-level missing observations. To address the first challenge, HALight formulates TSC with heterogeneous agents and adopts intersection-specific observation and action representations. HALight explicitly tailors each agent’s representation to its local lane configuration and phase design, enabling agents to account for structural differences among intersections and overcome the limitations of homogeneous agent modeling. For the second challenge, HALight incorporates a lightweight heterogeneous communication mechanism. It allows agents to selectively aggregate information from one-hop neighboring intersections, thereby mitigating the impact of locally missing observations and improving coordination. The main contributions are summarized as follows:

1) We investigate a practically important problem: enabling effective decentralized MARL for TSC in heterogeneous road networks under lane-level missing observations. In real-world deployments, intersections differ in types and configurations, and lane-level traffic measurements are often incomplete. The coexistence of intersection heterogeneity and lane-level observation missingness poses significant challenges for decentralized MARL methods.

2) We propose HALight, a communication-enhanced heterogeneous MARL framework for TSC. HALight formulates TSC with heterogeneous agents by adopting intersection-specific observation and action representations tailored to local lane configurations and phase designs. In addition, HALight incorporates a lightweight one-hop communication mechanism to alleviate the impact of locally missing lane-level observations and improve coordination in heterogeneous road networks.

3) Extensive experiments are conducted on two real-world road networks in Hangzhou under varying demands and observation missing rates. The results demonstrate that HALight outperforms representative baselines in terms of efficiency and convergence behavior.

The rest of the paper is organized as follows. Section II reviews related work. Section III formulates the TSC problem under a MARL framework. Section IV presents the proposed method. Section V describes the experimental settings and discusses the results. Finally, Section VI concludes the paper.

## II. RELATED WORK

This section provides a summary of DRL-based multiintersection cooperative control methods, covering both homogeneous and heterogeneous control strategies, and discussing lane-level missing observations in TSC problems.

## A. Homogeneous MARL for Traffic Signal Control

Homogeneous MARL assumes that all agents share identical observation spaces, action spaces, and network architectures. Under this assumption, one commonly adopted early approach is the independent MARL paradigm. For example, [22] modeled the traffic network as a collection of independently controlled intersections, with coordination achieved through transfer planning and the max-plus algorithm. PressLight [16] further incorporated the Max-pressure [23] principle into an IDQN framework, allowing agents to optimize the pressure difference between upstream and downstream intersections. However, independent MARL methods treat other agents as part of the environment, often resulting in unstable training as neighboring policies change.

To mitigate this issue, the centralized training and decentralized execution (CTDE) paradigm [24], [25] is often coupled with parameter sharing [26] in MARL-based TSC [17], [27]. In such shared policy implementations, agents are trained with centralized information while sharing policy parameters, which allows them to benefit from collective experiences and improves sample efficiency. For instance, RegionLight [15] integrates CTDE with regional partitioning and an Adaptive Branching Dueling Q-Network (ABDQ) for dynamic action space decomposition, enabling coordinated decision making while maintaining decentralized execution. To better capture spatial dependencies among intersections, some MARL methods incorporate graph neural networks (GNNs) to aggregate information from neighboring intersections [14], [28], [29]. In addition, sequential decision-making has been considered in TSC [30], [31]. For example, GNSD-Light [32] introduces a priority-aware sequential decision MARL framework to coordinate intersections with different priorities.

While homogeneous MARL methods have shown promising performance, they typically rely on unified observation/action representations. This reliance limits their applicability to heterogeneous road networks where intersections differ in lane configuration and phase set. Relative indicators, such as pressure, density and occupancy ratio, can partially alleviate observation space differences across intersections [13], [16]. However, when intersections have different phase sets, constructing a unified action space through action masking can still lead agents to learn averaged behaviors, limiting their ability to learn intersection-specific control policies.

## B. Heterogeneous MARL for Traffic Signal Control

Real-world traffic networks contain intersections with diverse lane configurations and signal phase sets, leading to different control requirements. Heterogeneous MARL therefore enables intersections to learn intersection-specific policies while coordinating with structurally different agents.

Several studies have explored heterogeneous MARL for TSC. Early work assigns independent agents with intersectionspecific policies to heterogeneous intersections [33]. Other studies reduce structural differences through transformation or unified representations. HA2C [34] employs an intersection structure transformation scheme and learns both general and structure-specific policies, while HeteroLight [35] introduces general feature extraction to construct unified state and action representations across heterogeneous intersections. More recent methods further focus on generalization across diverse traffic scenarios. CROSS [36] employs a mixture-of-experts policy to enable adaptive specialization, whereas Unicorn [37] combines universal and intersection-specific representations for heterogeneous network control. These studies demonstrate different strategies for balancing shared representations with intersection-specific characteristics.

Beyond representation design, heterogeneous agent reinforcement learning has also studied policy optimization for agents with distinct policies. HAPPO and HATRPO [38] employ sequential policy updates based on the multi-agent advantage decomposition lemma [39], providing theoretically justified monotonic policy improvement. Heterogeneous-Agent Mirror Learning [40] further generalizes the theoretical framework for heterogeneous policy optimization. However, these theoretical analyses are established under the fully observable setting considered in their original formulations. When agents operate with incomplete observations and recurrent histories, the original monotonic-improvement argument does not directly apply. This motivates further study of heterogeneous policy optimization under partial observability in TSC.

## C. Lane-level Missing Observations in Traffic Signal Control

Lane-level observations can become incomplete in realworld TSC because of sensor failures, limited sensing coverage, or sparse detector deployment [18]. Existing traffic data imputation methods [41], [42], [43] mainly focus on reconstructing incomplete traffic measurements. However, applying a separate reconstruction process to online TSC introduces additional processing before decision making.

To address partial observability directly within the decisionmaking process, recurrent models are commonly used to encode historical observations and retain temporal context [44], [45]. Such temporal information can help distinguish similar current observations associated with different underlying traffic conditions. Meanwhile, spatiotemporal graph-based TSC methods like DynSTGAT [46] jointly model historical traffic dynamics and dependencies among neighboring intersections to improve coordinated control.

Communication provides another way to supplement incomplete local information. In general heterogeneous MARL, the Communication Complementary Graph Model (CCGM) [47] incorporates messages from other agents into local observations through dynamically constructed communication graphs. Such communication mechanisms demonstrate the potential of spatial information for mitigating limited observations. For TSC, where traffic interactions are largely driven by neighboring upstream and downstream intersections, efficiently combining historical context with localized inter-agent information is particularly important. This motivates a lightweight temporal-spatial modeling strategy for heterogeneous TSC under lane-level missing observations.

## III. PROBLEM FORMULATION

This section provides a formal definition of the TSC problem and a comprehensive overview of the partially observable reinforcement learning framework. Since intersections in practical road networks can exhibit diverse structural configurations, we formulate and discuss the problem separately under heterogeneous and homogeneous modeling settings, to clearly distinguish their observation, action, and reward designs.

## A. Traffic Signal Control Definition

For heterogeneous methods, we model the traffic network as a directed heterogeneous graph $G _ { \mathrm { h e t } } ~ = ~ ( V , E )$ , where each node $v _ { i } ~ \in ~ V$ corresponds to a signalized intersection, and each directed edge $e _ { i j } \in E$ represents a road segment that allows vehicles to flow from intersection $v _ { i }$ to $v _ { j }$ . The network is heterogeneous in the sense that each intersection has distinct structural properties, such as different numbers of lanes and signal phase configurations, leading to diverse observation and action spaces across agents. In this setting, each intersection is controlled by a heterogeneously modeled agent equipped with an individualized policy network that receives a local observation $o _ { t } ^ { i }$ and selects a control phase $a _ { t } ^ { i }$ from its predefined phase set $\mathcal { A } ^ { i }$ at every decision step.

For homogeneous methods, we model the traffic network as a directed homogeneous graph $G _ { \mathrm { h o m o } } = ( V ^ { ' } , E ^ { ' } )$ , which preserves the same road connectivity as $G _ { \mathrm { h e t } }$ but assumes identical structural and operation phases for all intersections. In this setting, the observation space O is obtained by aggregating the incoming lane features of each intersection into a fixed size feature vector following a unified aggregation rule, so that all agents share the same observation dimension. Similarly, the action space $\mathcal { A }$ is defined as the union of all intersection phase sets, representing a common set of feasible signal phases available to every agent. Each intersection is controlled by an agent parameterized through a shared policy network which receives an observation $o _ { t } \in \mathcal { O }$ and outputs a phase selection $a _ { t } \in \mathcal A$ at each decision step.

## B. Partially Observable MARL Formulation

MARL provides a framework in which multiple agents learn optimal policies through trial-and-error interactions with the environment. In practical TSC deployments, sensor failures and incomplete or spatially varying sensing coverage can lead to missing lane-level traffic observations, preventing agents from directly observing the complete environment state. Such settings can be naturally formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP), defined by a tuple $\mathcal { M } = < ~ \mathcal { S } , \mathcal { N } , \mathcal { O } , \mathcal { A } , P , r , \gamma ~ > . ~ s ~ \in ~ \mathcal { S }$ is the state of the environment, $i \in \mathcal { N } \equiv \{ 1 , 2 , \dotsc , n \}$ denotes the index of agents. The joint action space ${ \mathcal { A } } =$ $\prod ^ { n } A ^ { i }$ is the Cartesian product of all agents’ action space. $\stackrel { \iota = 1 } { \gamma \mathrm { ~ i s ~ } }$ the discount factor that determines the relative importance of future rewards compared to immediate ones. $\mathbf { A t }$ each timestep $t ,$ agent i receives the local observation $o ^ { i }$ from the environment and selects an action $a _ { t } ^ { i }$ from predefined action set $\mathcal { A } ^ { i }$ according to its policy $\overline { { \pi } } ^ { i } \left( a _ { t } ^ { i } \mid o _ { t } ^ { i } \right)$ The joint action $\mathbf { \delta } \mathbf { a } _ { t } = ( a _ { t } ^ { 1 } , a _ { t } ^ { 2 } , \ldots , a _ { t } ^ { n } )$ and the joint policy $\pi \left( \boldsymbol { a } _ { t } \mid \boldsymbol { o } _ { t } \right) = \prod _ { i = 1 } ^ { n } \pi ^ { i } \left( a _ { t } ^ { i } \mid \boldsymbol { o } _ { t } ^ { i } \right)$ are obtained by combining all individual agents’ actions and policies, respectively. After all agents take action, each agent i receives a local reward $r _ { t } ^ { i }$ and the environment transitions to the next state $s _ { t + 1 }$ according to the probability function $P : S \times A \times S  [ 0 , 1 ]$ . The systemlevel reward $r _ { t } = \sum _ { i = 1 } ^ { n } r _ { t } ^ { i }$ is obtained by aggregating all local rewards, which is consistent with the cooperative Dec-POMDP

formulation. The state value function in a state s is defined as $V _ { \pi } ( s ) = \mathbb { E } _ { \pi } \Big \lvert \sum _ { \iota = \Omega } ^ { \infty } \gamma ^ { t } r _ { t } \mid s _ { 0 } = s , \mathbf { a } _ { t } \sim \pi , s _ { t + 1 } \sim P \Big \rvert$ , and the t=0   
action value of a state-action pair is defined as $\bar { Q } _ { \pi } ( s , { \pmb a } ) =$ $\mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t } \mid s _ { 0 } = s , \pmb { a } _ { 0 } = \pmb { a } , \pmb { a } _ { t } \sim \pi , s _ { t + 1 } \sim P \right]$ . The advantage function is then given by ${ \cal A } _ { \pi } ( s , { \pmb a } ) = \mathsf { \bar { Q } } _ { \pi } ( s , { \pmb a } ) \ -$ $V _ { \pi } ( s )$ , which measures the relative benefit of taking joint action a compared with the expected performance of the policy at state s. In the traffic scenario, agents collaborate to maximize the sum of future rewards $R _ { t } = \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r _ { t + k }$ by optimizing the joint policy π.

## C. Observation

1) Observation Definition: For heterogeneous methods, the observation of an agent i corresponding to intersection u is defined as the concatenation of the states of all incoming lanes at that intersection. Following prior study [15], we adopt three types of features as the observation inputs: the number of waiting vehicles, the wave count representing the number of vehicles in the incoming lanes, and the current signal phase. Only incoming lanes are considered in the observation, since they provide direct information about approaching traffic flows that determine the signal control decisions. Formally, the $o ^ { i }$ is presented as:

$$
o ^ { i } = \left\{ \{ \mathrm { w a i t } [ l ] , \mathrm { w a v e } [ l ] \} _ { l \in \mathrm { I n L a n e s } [ u ] } , \mathrm { p h a s e } [ u ] \right\} ,\tag{1}
$$

where InLanes[u] denotes the set of incoming lanes at intersection u.

For homogeneous methods, the observation is designed to construct a unified feature representation across all intersections. To achieve this, we aggregate lane-level features according to the four directional approaches (North, South, West, East) at each intersection. For each approach, three functional lane types are considered, including left-turn, through, and right-turn. Each lane type contributes two numerical features: the wait count and wave count, resulting in 6 dimensions per approach and 24 dimensions in total. The current signal phase of the intersection is then appended, forming a fixedlength observation vector with 25 dimensions overall. For mixed lanes (e.g., left-through or through-right lanes), the corresponding lane features are evenly distributed across the related functional types. For instance, if a lane allows both left-turn and through movements, its wait and wave counts are divided by two and added equally to the left-turn and through dimensions. For T junctions, where certain approaches or lane types are absent, the corresponding feature entries are zero-padded to maintain consistent dimensionality across agents. Formally, the observation of agent i at intersection u is expressed as:

$$
o ^ { i } = \left\{ \{ \widehat { \mathrm { w a i t } } [ d , k ] , \widehat { \mathrm { w a v e } } [ d , k ] \} _ { d \in { \mathscr { D } } , k \in { \mathscr { F } } } , \mathrm { p h a s e } [ u ] \right\} ,\tag{2}
$$

where $\mathcal { D } \ = \ \{ \mathrm { N , E , S , W } \}$ denotes the set of directions, and $\mathcal { F } ~ = ~ \{ \mathrm { l e f t - t u r n } , \mathrm { t h r o u g h } , \mathrm { r i g h t - t u r n } \}$ represents the functional lane types.

![](images/ba0cb0cd61856ecc3cd415a24772d102e0d6d7b618c009375e1295e84163d52d.jpg)  
Fig. 3. Figure of phase setting scheme for different intersection types: $\mathrm { P _ { A } }$ and P correspond to two typical phase plans for crossing intersections, while P represents a phase plan for a T intersection.

2) Lane-level Missing Observation Modeling: Lane-level measurements can be partially unavailable in practice, resulting in missing observations for some incoming lanes. To account for this factor, we introduce an observation missing rate $p _ { \mathrm { m i s s } } ,$ representing the probability that the features of a given incoming lane are unavailable. Each episode corresponds to one simulated hour of traffic signal control. At the beginning of each episode, for each lane $l \in$ InLanes[u], its corresponding wait and wave features are randomly masked with probability $p _ { \mathrm { m i s s } }$ , and their values are set to zero for the entire episode. By resampling the missing lane set across episodes, this setting emulates diverse lane-level sensor availability patterns during training and enables robustness evaluation under varying missing configurations. The signal phase is assumed to be always observable and is therefore excluded from the masking process. Formally, for heterogeneous methods, the masked observation of agent i corresponding to intersection u is expressed as:

$$
\begin{array} { r l } & { ~ \tilde { o } ^ { i } = \left\{ \{ \mathsf { w a i t } [ l ] , ~ \mathrm { w a v e } [ l ] \} _ { l \in \mathrm { I n L a n e s } [ u ] } , ~ \mathsf { p h a s e } [ u ] \right\} , } \\ & { ~ \tilde { \mathsf { w a i t } } [ l ] = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } p _ { \mathrm { m i s s } } , } \\ { \mathrm { w a i t } [ l ] , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \\ & { ~ \mathsf { w a v e } [ l ] = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } p _ { \mathrm { m i s s } } , } \\ { \mathrm { w a v e } [ l ] , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{3}
$$

Similarly, for homogeneous methods, masking is applied at the lane level before aggregating features into directional and functional categories. At the beginning of each episode, each incoming lane’s wait and wave entries are independently set to zero with probability $p _ { \mathrm { m i s s } }$ . The remaining lane features are then aggregated into the [d, k] dimensions obeying the same rule as in the observation definition. Thus, the masked observation for agent i at intersection u is

$$
\tilde { o } ^ { i } = \left\{ \widehat { \{ \mathrm { w a i t } [ d , k ] , ~ }  \widehat { \mathrm { w a v e } } [ d , k ] \} _ { d \in { \mathscr { D } } , k \in { \mathscr { F } } } , ~ \mathsf { p h a s e } [ u ] \right\} .\tag{4}
$$

## D. Action

In real-world urban scenarios, intersections often exhibit different traffic volumes and geometric structures, which naturally lead to diverse phase setting strategies. To capture such heterogeneity, we introduce three phase setting schemes. Specifically, for crossing intersections, two alternative fourphase configurations are defined, denoted as $\mathrm { P _ { A } }$ and ${ \mathrm { P } } _ { \mathrm { B } } ;$ for T intersections, a three-phase configuration $\mathrm { P _ { C } }$ is adopted. It is assumed that right-turn movements are always allowed. The phase setting schemes of crossing intersections and T intersections $\mathrm { { P } _ { A } , \mathrm { { P } _ { B } , } }$ , and $\mathrm { P _ { C } }$ are illustrated in Fig. 3. Formally, the action space of agent i at intersection u is defined as:

$$
\begin{array} { r } { \mathcal { A } ^ { i } = \left\{ \begin{array} { l l } { \mathrm { P _ { A } ~ o r ~ \mathrm { P _ { B } } , } } & { \mathrm { i f ~  { u _ { \ell } } ~ i s ~ a ~ c r o s s i n g ~ i n t e r s e c t i o n , } } \\ { \mathrm { P _ { C } , } } & { \mathrm { i f ~  { u _ { \ell } } ~ i s ~ a ~ T ~ i n t e r s e c t i o n . } } \end{array} \right. } \end{array}\tag{5}
$$

## E. Reward

In TSC problem, the design of the reward function plays a crucial role in guiding agents toward efficient traffic coordination. Various performance indicators, including waiting time, the number of waiting vehicles, delay, and travel speed, have been widely adopted in prior studies to characterize traffic efficiency. In this work, we employ the negative summation of the number of waiting vehicles on all incoming lanes as the reward. The instantaneous reward for agent i at timestep t is defined as:

$$
r _ { t } ^ { i } = - \sum _ { l \in \mathrm { I n L a n e s } [ u ] } { \mathrm { w a i t } } _ { t } [ l ] ,\tag{6}
$$

where $\mathrm { w a i t } _ { t } [ l ]$ denotes the number of waiting vehicles on the incoming lane l of intersection at time t. The system-level reward at each timestep is obtained by aggregating the local rewards of all agents:

$$
r _ { t } = \sum _ { i = 1 } ^ { n } r _ { t } ^ { i } .\tag{7}
$$

## IV. HALIGHT ARCHITECTURE

This section presents a heterogeneous MARL algorithm for TSC with lane-level missing observations, referred to as HALight. The overall framework of the proposed model is illustrated in Fig. 4 and the pseudo-code of the HALight is presented in Algorithm 1. The proposed HALight framework mainly consists of two key components: Communication-Enhanced Heterogeneous Actors and a Centralized Critic, which will be described in the following subsections.

## A. Communication-Enhanced Heterogeneous Actors

Following the communication motivation of CCGM [47], HALight equips each heterogeneous actor with an explicit inter-agent communication module to enhance coordination under partial observability. Different from CCGM, we tailor the communication protocol to TSC by restricting message exchanges to one-hop physical neighbors and integrating the received messages via a lightweight dot-product attention mechanism, rather than relying on GAT-based aggregation and dynamic communication graph reconstruction. This design is motivated by the intrinsic locality of traffic interactions: adjacent intersections connected by road segments exhibit the strongest coupling in queue propagation and phase interference, and therefore provide the most informative spatial context for compensating missing observations while keeping the communication overhead scalable in large road networks.

![](images/8a5654898ca521f8d0a99e0582a5d07dd007211e299637abaa056817ffc55a2f.jpg)  
Fig. 4. An overview of HALight, including (a) the agent decision process, which consists of local observation encoding, one-hop neighborhood communication, attentive aggregation, information generation, and decision making, and (b) the sequential training pipeline with replay buffer sampling and sequential updates

1) Local Observation Encoding: At each timestep t, the encoder of agent i processes its local observation together with the communication information received from its neighboring agents at the previous timestep. Let $\operatorname { i n f o } _ { t - 1 } ^ { j }$ denote the message broadcast by neighbor $j .$ The encoded latent representation is given by:

$$
h _ { t } ^ { i } = f _ { \mathrm { e n c } } ^ { i } ( \mathrm { c o n c a t } ( \tilde { o } _ { t } ^ { i } , \{ \mathrm { i n f o } _ { t - 1 } ^ { j } \} _ { j \in \mathcal { N } ( i ) } ) ) ,\tag{8}
$$

where $f _ { \mathrm { e n c } } ^ { i } ( \cdot )$ denotes a single layer linear encoder, and $\mathcal { N } ( i )$ represents the set of one-hop neighboring intersections in $G _ { \mathrm { h e t } } .$ The generation process of $\operatorname { i n f o } _ { t - 1 } ^ { j }$ will be elaborated later in this section.

2) Attentive Neighborhood Representation Aggregation: The latent representation $h _ { t } ^ { i }$ reflects a preliminary understanding of the local environment for agent i at timestep t. Considering that neighboring intersections exhibit distinct traffic conditions and exert varying degrees of influence on the target intersection, each agent is expected to selectively aggregate information from its neighbors rather than treating them equally. To achieve this, we adopt a dot-product attention mechanism with learnable linear projections. Specifically, the current hidden state $h _ { t } ^ { i }$ is mapped to a query vector, while the neighbors’ hidden states $\{ h _ { t } ^ { j } \} _ { j \in \mathcal { N } ( i ) }$ are mapped to key and value vectors:

$$
q _ { t } ^ { i } = W _ { q } h _ { t } ^ { i } , \qquad k _ { t } ^ { j } = W _ { k } h _ { t } ^ { j } , \qquad v _ { t } ^ { j } = W _ { v } h _ { t } ^ { j } ,\tag{9}
$$

where $W _ { q } , W _ { k } , W _ { v } \in \mathbb R ^ { d _ { a } \times d _ { h } }$ are training projection matrices and $d _ { a }$ denotes the dimension of the attention vector. The normalized attention coefficient is then computed as

$$
\alpha _ { i j } = \frac { \exp \left( q _ { t } ^ { i } ( k _ { t } ^ { j } ) ^ { \mathrm { T } } / \sqrt { d _ { a } } \right) } { \sum _ { j \in \mathcal { N } ( i ) } \exp \left( q _ { t } ^ { i } ( k _ { t } ^ { j } ) ^ { \mathrm { T } } / \sqrt { d _ { a } } \right) } ,\tag{10}
$$

$\alpha _ { i j }$ measures the relative importance of neighbor $j$ to agent i. Thus, the enhanced neighborhood representation is obtained by the attention-weighted aggregation of values:

$$
h _ { t } ^ { \prime i } = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } v _ { t } ^ { j } .\tag{11}
$$

Since $f _ { \mathrm { e n c } } ^ { i } ( \cdot )$ projects all agents’ inputs into a shared latent space of dimension $d _ { h }$ , the linear projections $( W _ { q } , W _ { k } , W _ { v } )$ consistently map these representations into the common attention space $d _ { a }$ , enabling the dot-product operation to be applied in a principled manner.

3) Information Generation: After obtaining the enhanced neighborhood representation $h _ { t } ^ { \prime i }$ , each agent generates a piece of self-interested information through its communication decoder head, which is then transmitted to its neighboring agents and is simultaneously used for its own decision making. At this stage, the neighborhood information aggregated is condensed into a latent hypothesis $\operatorname { i n f } 0 _ { t } ^ { i }$ This hypothesis reflects the action tendency of the sending agent, providing implicit cues that allow neighboring agents to make more informed inferences and coordinate their subsequent actions accordingly. Since the generated information encapsulates the agent’s behavioral tendency, its values are constrained within the range [0, 1], using a sigmoid activation function. Formally, the self-interested information of agent i at timestep t is defined as:

```latex
Algorithm 1: The HALight Algorithm
Input: Learning rate α, batch size B, number of
agents n, episodes K, steps per episode T, traffic
network $G _ { h e t }$
1 Initialize actor networks $\{ \theta _ { 0 } ^ { i } \} _ { i = 1 } ^ { n }$ , centralized critic $\phi _ { 0 } .$
and replay buffer B;
2 Determine the first-order neighbor sets $\mathcal { N } ( i )$ for each
agent i from $G _ { \mathrm { h e t } } ;$
3 for $k = 0 , 1 , \ldots , K - 1$ do
4 Reset critic’s and all actors’ GRU hidden states;
5 Collect a trajectory using joint policy $\pi _ { \boldsymbol { \theta } _ { k } } ;$
6 Push $\{ ( s _ { t } , \tilde { o } _ { t } ^ { i } , a _ { t } ^ { i } , r _ { t } , \tilde { o } _ { t + 1 } ^ { i } , h _ { t } ^ { i } , \mathrm { i n f o } _ { t } ^ { i } ) , \forall t \in T \}$ in $\begin{array} { r } { B ; { } } \end{array}$
7 Sample B trajectory segments from B in temporal
order ; $/ /$ Required due to GRU
8 Compute advantages $\hat { A } ( s , \pmb { a } ) = \hat { R } - V _ { \phi } ( s )$ using
the centralized critic with GAE;
9 Generate a random permutation $( i _ { 1 } , i _ { 2 } , \ldots , i _ { n } )$ of
agent indices;
10 Set $M ^ { i _ { 1 } } = 1 ;$
11 for $m = 1 , 2 \dots , n$ do
12 Compute importance sampling ratio $\omega ^ { i _ { m } }$
according to Equation (21);
13 Update actor $i _ { m }$ to $\theta _ { k + 1 } ^ { i _ { m } }$ by maximizing:
$M ^ { i _ { 1 : m } } \hat { A } ( s , { \pmb a } ) \cdot \operatorname* { m i n } \left( \omega ^ { i _ { m } } , \mathrm { c l i p } ( \omega ^ { i _ { m } } , 1 \pm \epsilon ) \right)$
14 Update joint policy correction factor:
${ \cal M } ^ { i _ { 1 : m + 1 } } = \omega ^ { i _ { m } } { \cal M } ^ { i _ { 1 : m } } .$
15 end
16 Update critic parameters:
$\phi _ { k + 1 } \gets \arg \operatorname* { m i n } _ { \phi } \frac { 1 } { B T } \sum _ { b = 1 } ^ { B } \sum _ { t = 0 } ^ { T } \Big ( V _ { \phi } ( s _ { t } ) - \hat { R } _ { t } \Big ) ^ { 2 }$
17 end
```

$$
\begin{array} { r } { \operatorname { i n f o } _ { t } ^ { i } = \sigma \left( f _ { \mathrm { d e c } } ^ { i } \left( h _ { t } ^ { \prime i } \right) \right) , } \end{array}\tag{12}
$$

where $f _ { \mathrm { d e c } } ^ { i } ( \cdot )$ denotes the information decoder that projects $h _ { t } ^ { \prime i }$ into a $d _ { \mathrm { i n f c } }$ dimensional information vector and $\sigma ( \cdot )$ represents the sigmoid activation function. This design ensures that the transmitted information remains bounded and interpretable, facilitating stable communication and mutual coordination among heterogeneous agents.

4) Decision Making: Having generated the self-interested communication information info , agent i enters the action computation phase. A policy network is employed, consisting of a feedforward module, a gated recurrent unit (GRU) [48], and an action head. The GRU mitigates partial observability by maintaining a memory of past states, similar to recurrent reinforcement learning architectures.

At timestep t, the agent i concatenates its local observation $\tilde { o } _ { t } ^ { i }$ with the communication information info<sup>i</sup> to form a fused input vector, which is then processed by a two-layer feedforward module:

$$
x _ { t } ^ { i } = \mathrm { R e L U } \left( W _ { 2 } \mathrm { R e L U } \left( W _ { 1 } \mathrm { c o n c a t } ( \tilde { o } _ { t } ^ { i } , \mathrm { i n f o } _ { t } ^ { i } ) + b _ { 1 } \right) + b _ { 2 } \right) ,\tag{13}
$$

where $W _ { 1 , 2 }$ and $b _ { 1 , 2 }$ are the parameters of the feedforward module. The output $\boldsymbol { x } _ { t } ^ { i }$ is then passed through the GRU to obtain the action feature $z _ { t } ^ { i } .$ . Let $h _ { t } ^ { \mathrm { p o l } , i }$ denote the GRU hidden state of the policy network for agent i, the GRU processes the input as follows:

$$
z _ { t } ^ { i } , h _ { t } ^ { \mathrm { p o l } , i } = \mathrm { G R U } ( x _ { t } ^ { i } , h _ { t - 1 } ^ { \mathrm { p o l } , i } ) .\tag{14}
$$

The action feature $z _ { t } ^ { i }$ is then passed to the action head to produce the policy distribution over the available phases:

$$
\pi ^ { i } ( a _ { t } ^ { i } \mid \tilde { o } _ { t } ^ { i } , \{ \mathrm { i n f o } _ { t - 1 } ^ { j } \} _ { j \in \mathcal { N } ( i ) } ) = \mathrm { s o f t m a x } \big ( W _ { \mathrm { a c t } } z _ { t } ^ { i } + b _ { \mathrm { a c t } } \big ) ,
$$

from which the action $a _ { t } ^ { i }$ is sampled.

(15)

## B. Centralized Critic

Following the CTDE paradigm, HALight employs a centralized critic to estimate value functions using joint information during training. The critic receives a global shared state that integrates not only each agent’s local observation but also the communication information produced by the actors. This design is motivated by the fact that the generated information vectors encode each agent’s action tendency and implicitly reflect the influence of neighborhood interactions. Concretely, at timestep t, each agent i forms a compact local feature $g _ { t } ^ { i } = \mathrm { c o n c a t } ( \tilde { o } _ { t } ^ { i } , \mathrm { i n f o } _ { t } ^ { i } )$ . The global state $s _ { t }$ is obtained by concatenating all agents’ local features:

$$
s _ { t } = \mathrm { c o n c a t } ( g _ { t } ^ { 1 } , g _ { t } ^ { 2 } , . . . , g _ { t } ^ { n } ) .\tag{16}
$$

The critic network processes $s _ { t }$ through a two-layer feedforward module followed by a GRU, which helps improve the reliability of value estimation under partial observability. Formally, the critic update is given by:

$$
\boldsymbol { v } _ { t } , h _ { t } ^ { c } = \mathrm { G R U } \left( f _ { \mathrm { M L P } } \left( \boldsymbol { s } _ { t } \right) , h _ { t - 1 } ^ { \mathrm { c } } \right) ,\tag{17}
$$

where $f _ { \mathrm { M L P ( \cdot ) } }$ denotes the feedforward module, $v _ { t }$ denotes the output feature vector produced by the GRU and $h _ { t } ^ { \mathrm { c } }$ is the critic’s recurrent hidden state.

Finally, the state value $V ( s _ { t } )$ is obtained through a linear output layer:

$$
V ( s _ { t } ) = W _ { v } v _ { t } + b _ { v } ,\tag{18}
$$

where $W _ { v }$ and $b _ { v }$ denote the parameters of the value head.

## C. Training Pipeline

HALight adopts a HAPPO-style sequential policy update scheme for heterogeneous agents. Under full observability, the recurrent policy representation and communication modules can be regarded as components of individual policies. They enrich the policy inputs while preserving the factorized joint policy structure required by HAPPO. The convergence analysis of HALight under this setting is provided in Appendix A. Since the main task considers lane-level missing observations, the above theoretical guarantee is established only under full observability. The effectiveness of the sequential update and the proposed modules under partial observability is further validated through empirical evaluations.

![](images/1c10ccec055606d9fac4755bb3f9a07e8ee56dcb76f7899e6cbad4caba362300.jpg)  
Fig. 5. Figures of the evaluated road networks, including (a) Xiasha road network and (b) Gudang road network.

The actor and critic update procedures are described below.

1) Actor Optimization: At each training iteration, advantage estimates $\hat { A } ( s , { \pmb a } )$ for all agents are computed using generalized advantage estimation (GAE) and the centralized critic. Let agents be indexed as $i = 1 , 2 , \ldots , n .$ , and denote their policy network by $\theta ^ { 1 } , \theta ^ { 2 } , \ldots , \theta ^ { n }$ . A random permutation $( i _ { 1 } , i _ { 2 } , \ldots , i _ { n } )$ of these agent indices is sampled, and the actors are updated sequentially following this order.

Consider the update of the m-th agent in the permutation, i.e., agent $i _ { m }$ . To ensure that the sequential updates are consistent with the improvement of the joint policy, HALight adopts the joint policy correction factor M<sup>i1:m</sup>, which accounts for the influence of previously updated agents on the current policy update:

$$
M ^ { i _ { 1 : m } } = \prod _ { q = 1 } ^ { m - 1 } \frac { \pi _ { \theta _ { k + 1 } ^ { i _ { q } } } ^ { i _ { q } } \left( a ^ { i _ { q } } \mid \tilde { o } ^ { i _ { q } } , \left\{ \mathrm { i n f o } ^ { j } \right\} _ { j \in \mathcal { N } ( i _ { q } ) } \right) } { \pi _ { \theta _ { k } ^ { i _ { q } } } ^ { i _ { q } } \left( a ^ { i _ { q } } \mid \tilde { o } ^ { i _ { q } } , \left\{ \mathrm { i n f o } ^ { j } \right\} _ { j \in \mathcal { N } ( i _ { q } ) } \right) } ,\tag{19}
$$

where k denotes the policy optimization iteration, and $\theta _ { k } ^ { i _ { q } }$ and $\theta _ { k + 1 } ^ { i _ { q } }$ represent the policy parameters of agent $i _ { q }$ before and after its sequential update, respectively. For $m = 1$ , the correction factor is defined as $M ^ { i _ { 1 } } = 1$ . The policy of agent $i _ { m }$ is then updated by maximizing the clipping objective scaled by this correction factor:

$$
\begin{array} { r l } & { \theta _ { k + 1 } ^ { i _ { m } } \gets } \\ & { \arg \operatorname* { m a x } _ { \theta ^ { i _ { m } } } \mathbb { E } \Big [ M ^ { i _ { 1 : m } } \hat { A } ( s , a ) \operatorname* { m i n } \big ( \omega ^ { i _ { m } } , \mathrm { c l i p } ( \omega ^ { i _ { m } } , 1 \pm \epsilon ) \big ) \Big ] , } \end{array}\tag{20}
$$

where ϵ is the clipping parameter, the expectation is taken over $s \sim \rho _ { \pi _ { \theta _ { k } } }$ and $\mathbf { \Delta } a \sim \pi _ { \theta _ { k } }$ , and the importance sampling ratio $\omega ^ { i _ { m } }$ is defined as:

$$
\omega ^ { i _ { m } } = \frac { \pi _ { \theta ^ { i _ { m } } } ^ { i _ { m } } \left( a ^ { i _ { m } } \mid \tilde { o } ^ { i _ { m } } , \{ \mathrm { i n f o } ^ { j } \} _ { j \in \mathcal { N } ( i _ { m } ) } \right) } { \pi _ { \theta _ { k } ^ { i _ { m } } } ^ { i _ { m } } \left( a ^ { i _ { m } } \mid \tilde { o } ^ { i _ { m } } , \{ \mathrm { i n f o } ^ { j } \} _ { j \in \mathcal { N } ( i _ { m } ) } \right) } .\tag{21}
$$

This completes the sequential update for the m-th actor, and the process continues until all agents in the permutation have been updated once within the training iteration.

2) Critic Optimization: The centralized critic is trained to approximate the state value function $V ( s _ { t } )$ based on the global shared state. During training, the critic parameters $\phi$ are optimized by minimizing the mean squared error between the predicted value and a bootstrapped return $\hat { R } _ { t }$ . The critic’s loss function is given by:

$$
\mathcal { L } _ { \mathrm { c r i t i c } } ( \phi ) = \Big ( V _ { \phi } ( s _ { t } ) - \hat { R } _ { t } \Big ) ^ { 2 } .\tag{22}
$$

The target return $\hat { R } _ { t }$ is computed using GAE, and the critic update is performed after sequential actor updates.

## V. EXPERIMENTS AND RESULTS

In this section, we evaluate HALight on two real-world road networks, and compare the algorithm with representative homogeneous and heterogeneous baselines.

## A. Experiment Scenario

We select two real-world road networks in Hangzhou for evaluation: a $3 ~ \times ~ 3$ grid road network in Xiasha and a 4 $\times \ 4$ grid road network in Gudang. The Gudang network consists of 16 crossing intersections with varying structural configurations, while the Xiasha network contains 7 crossing intersections and 2 T intersections with different configurations. The road network topology and phase settings are illustrated in Fig. 5. The lane configurations of both networks are constructed according to real-world conditions, with intersection information obtained from AutoNavi<sup>1</sup>. For each road network, we consider two traffic demand scenarios: moderate and high demand. The origin and destination of each vehicle are randomly sampled from the boundary edges of the network, and the travel path is determined by the shortest route between them. The road network and flow statistics are summarized in Table I.

TABLE I  
ROAD NETWORKS AND FLOW STATISTICS
<table><tr><td rowspan=2 colspan=1>Road Networks</td><td rowspan=1 colspan=2>Arrival Rate (vehicles/s)</td><td rowspan=2 colspan=1>Road Length (m)</td></tr><tr><td rowspan=1 colspan=1>Moderate</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Xiasha</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>400</td></tr><tr><td rowspan=1 colspan=1>Gudang</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>600</td></tr></table>

TABLE II HYPERPARAMETERS
<table><tr><td rowspan=1 colspan=1>Algorithms</td><td rowspan=1 colspan=1>Hyperparameters</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=6 colspan=1>DQN-based</td><td rowspan=1 colspan=1>Learning rate α</td><td rowspan=1 colspan=1>0.0005</td></tr><tr><td rowspan=1 colspan=1>Replay buffer size</td><td rowspan=1 colspan=1>1,800,000</td></tr><tr><td rowspan=1 colspan=1>Hidden size</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>€ decay steps</td><td rowspan=1 colspan=1>1,800,000</td></tr><tr><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Training sequence length</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=4 colspan=1>PPO-based</td><td rowspan=1 colspan=1>Learning rate $\underline { { \alpha _ { \mathrm { c r i t i c } } } }$ </td><td rowspan=1 colspan=1>0.0005</td></tr><tr><td rowspan=1 colspan=1>Learning rate αactor</td><td rowspan=1 colspan=1>0.00005</td></tr><tr><td rowspan=1 colspan=1>Hidden size</td><td rowspan=1 colspan=1>64 for actors, 128 for critic</td></tr><tr><td rowspan=1 colspan=1>GAEλ</td><td rowspan=1 colspan=1>0.9</td></tr></table>

<sup>1</sup>https://ditu.amap.com

![](images/2c25651fcb917ead8722dc5463550dd467789460850a823706769bdd962839b0.jpg)  
(b)  
Fig. 6. Learning curves of RL-based TSC methods on (a) the Xiasha road network and (b) the Gudang road network.

## B. Experimental Setup

1) Comparison Methods: For comparison, we select the following baseline methods, categorized into homogeneous and heterogeneous methods, as well as a non-RL method. To mitigate the effects of partial observability, all baseline methods incorporate a recurrent neural network within the policy network. To adapt to heterogeneous road networks, the homogeneous methods utilize observation combination, action masking and parameter sharing. In particular, action masking sets the Q-values of illegal actions to −∞ to prevent them from being selected.

• Fixed Time: A rule-based, non-RL method that operates with fixed time signals for each intersection, regardless of traffic conditions. This method serves as a baseline to compare the performance of RL-based methods against

traditional TSC strategies.

• PS-RDQN [44]: A homogeneous approach based on Qlearning, where all agents share a common deep Qnetwork and are trained centrally while executing independently. To alleviate partial observability, RDQN integrates a recurrent neural network into the Q-network.

• PS-PressLight-RNN [16]: We adopt a parameter sharing, CTDE version of PressLight as a homogeneous method for comparison. PressLight optimizes agents by using the maximum pressure strategy, where the pressure at each intersection is defined as the difference between the incoming and outgoing traffic flow capacities.

• PS-DynSTGAT [46]: A parameter sharing homogeneous spatio-temporal graph attention baseline for TSC, which captures spatial dependencies among intersections through graph attention and temporal dependencies from historical traffic states. Action masks are used to handle different feasible phase sets under the shared action space.

TABLE III  
OVERALL PERFORMANCE OF HALIGHT AND BASELINE METHODS UNDER DIFFERENT $p _ { \mathrm { m i s s } }$ ON THE XIASHA ROAD NETWORK. THE BEST RESULTS ARE HIGHLIGHTED IN BOLD, AND THE SECOND-BEST RESULTS ARE UNDERLINED. RESULTS FOR METHODS THAT FAIL TO CONVERGE ARE OMITTED.
<table><tr><td rowspan="3">Algorithms</td><td rowspan="3">Metric</td><td colspan="9">Scenario</td></tr><tr><td colspan="5">Moderate Demand</td><td colspan="5">High Demand</td></tr><tr><td></td><td>0%</td><td>10%</td><td>20%</td><td></td><td></td><td>0%</td><td>10%</td><td>20% 30%</td><td>40%</td></tr><tr><td rowspan="4">Fixed Time</td><td>ATT</td><td> $1 9 6 . 6 6 _ { \pm 2 . 1 1 }$ </td><td> $1 9 6 . 6 6 _ { \pm 2 . 1 1 }$ </td><td> $1 9 6 . 6 6 _ { \pm 2 . 1 1 }$ </td><td> $1 9 6 . 6 6 _ { \pm 2 . 1 1 }$ </td><td> $1 9 6 . 6 6 _ { \pm 2 . 1 1 }$ </td><td> $2 1 2 . 2 5 _ { \pm 6 . 1 2 }$ </td><td> $2 1 2 . 2 5 _ { \pm 6 . 1 2 }$ </td><td> $2 1 2 . 2 5 _ { \pm 6 . 1 2 }$ </td><td> $2 1 2 . 2 5 _ { \pm 6 . 1 2 }$ </td><td> $2 1 2 . 2 5 _ { \pm 6 . 1 2 }$ </td></tr><tr><td>AQL</td><td> $1 . 2 6 _ { \pm 0 . 0 4 }$ </td><td> $1 . 2 6 _ { \pm 0 . 0 4 }$ </td><td> $1 . 2 6 _ { \pm 0 . 0 4 }$ </td><td> $1 . 2 6 _ { \pm 0 . 0 4 }$ </td><td> $1 . 2 6 _ { \pm 0 . 0 4 }$ </td><td> $2 . 7 4 _ { \pm 0 . 3 1 }$ </td><td> $\underline { { 2 . 7 4 _ { \pm 0 . 3 1 } } }$ </td><td> $\underline { { 2 . 7 4 _ { \pm 0 . 3 1 } } }$ </td><td> $\underline { { 2 . 7 4 _ { \pm 0 . 3 1 } } }$ </td><td> $2 . 7 4 _ { \pm 0 . 3 1 }$ </td></tr><tr><td>AWT</td><td> $2 8 . 8 3 _ { \pm 1 . 0 5 }$ </td><td> $2 8 . 8 3 _ { \pm 1 . 0 5 }$ </td><td> $2 8 . 8 3 _ { \pm 1 . 0 5 }$ </td><td> $2 8 . 8 3 _ { \pm 1 . 0 5 }$ </td><td> $2 8 . 8 3 _ { \pm 1 . 0 5 }$ </td><td> $4 3 . 5 0 _ { \pm 5 . 3 1 }$ </td><td> $4 3 . 5 0 _ { \pm 5 . 3 1 }$ </td><td> $4 3 . 5 0 _ { \pm 5 . 3 1 }$ </td><td> $4 3 . 5 0 _ { \pm 5 . 3 1 }$ </td><td> $4 3 . 5 0 _ { \pm 5 . 3 1 }$ </td></tr><tr><td>TP</td><td> $1 3 5 8 . 4 9 _ { \pm 4 . 1 1 }$   $1 8 3 . 6 3 _ { \pm 1 . 5 3 }$ </td><td> $1 3 5 8 . 4 9 _ { \pm 4 . 1 1 }$   $1 9 5 . 2 8 _ { \pm 1 2 . 1 0 }$ </td><td> $1 3 5 8 . 4 9 _ { \pm 4 . 1 1 }$ </td><td> $1 3 5 8 . 4 9 _ { \pm 4 . 1 1 }$ </td><td> $1 3 5 8 . 4 9 _ { \pm 4 . 1 1 }$ </td><td> $2 0 2 5 . 0 2 _ { \pm 8 . 7 1 }$   $1 8 9 . 9 2 _ { \pm 1 . 5 6 }$ </td><td> $2 0 2 5 . 0 2 _ { \pm 8 . 7 1 }$ </td><td> $2 0 2 5 . 0 2 _ { \pm 8 . 7 1 }$ </td><td> $2 0 2 5 . 0 2 _ { \pm 8 . 7 1 }$ </td><td> $\pm 0 2 5 . 0 2 _ { \pm 8 . 7 1 }$ </td></tr><tr><td rowspan="4">PS-RDQN</td><td>ATT</td><td></td><td></td><td> $2 0 6 . 5 7 { \scriptstyle \pm 2 8 . 2 4 }$ </td><td> $2 1 1 . 7 8 \pm 1 7 . 1 3$ </td><td></td><td></td><td> $2 2 8 . 3 0 _ { \pm 9 4 . 1 6 }$ </td><td> $2 8 8 . 3 1 _ { \pm 1 6 3 . 6 6 }$ </td><td></td><td></td></tr><tr><td>AQL</td><td> $0 . 7 1 _ { \pm 0 . 0 2 }$ </td><td> $\underline { { 1 . 2 2 _ { \pm 0 . 5 4 } } }$ </td><td> $1 . 7 3 _ { \pm 1 . 3 1 }$ </td><td> $1 . 9 7 _ { \pm 0 . 8 1 }$ </td><td></td><td> $1 . 3 8 _ { \pm 0 . 0 3 }$ </td><td> $3 . 9 8 _ { \pm 6 . 4 2 }$ </td><td> $8 . 0 5 _ { \pm 1 1 . 1 5 }$ </td><td></td><td></td></tr><tr><td>AWT</td><td> $1 4 . 4 6 _ { \pm 0 . 2 8 }$ </td><td> $2 6 . 3 0 { \scriptstyle \pm 8 . 6 6 }$ </td><td> $3 6 . 9 1 _ { \pm 1 9 . 0 5 }$ </td><td> $4 4 . 8 7 \pm 1 4 . 4 6$ </td><td></td><td> $2 0 . 0 1 _ { \pm 0 . 6 8 }$ </td><td> $4 2 . 6 2 _ { \pm 2 3 . 5 3 }$ </td><td> $6 5 . 9 2 _ { \pm 3 0 . 9 1 }$ </td><td></td><td></td></tr><tr><td>TP</td><td> $1 3 6 3 . 5 0 _ { \pm 2 . 8 7 }$ </td><td> $1 3 5 7 . 7 1 _ { \pm 1 0 . 7 4 }$ </td><td> $1 3 4 9 . 3 6 _ { \pm 3 4 . 7 4 }$ </td><td> $1 3 4 8 . 1 7 _ { \pm 2 5 . 7 2 }$ </td><td></td><td> $2 0 4 0 . 6 8 _ { \pm 3 . 9 1 }$ </td><td> $1 9 8 8 . 9 3 _ { \pm 1 6 8 . 0 2 }$ </td><td> $1 8 9 5 . 7 0 _ { \pm 2 9 6 . 3 5 }$ </td><td></td><td></td></tr><tr><td rowspan="4">PS-PressLight -RNN</td><td>ATT</td><td> $1 8 3 . 3 6 _ { \pm 1 . 7 9 }$ </td><td> $2 0 6 . 8 1 _ { \pm 2 0 . 3 4 }$ </td><td> $2 2 3 . 3 2 _ { \pm 9 . 8 8 }$ </td><td></td><td></td><td> $1 9 1 . 6 4 _ { \pm 1 . 7 7 }$ </td><td> $2 3 1 . 9 1 _ { \pm 4 3 . 8 0 }$ </td><td> $2 7 6 . 1 2 _ { \pm 7 6 . 3 4 }$ </td><td></td><td></td></tr><tr><td>AQL</td><td> $0 . 7 0 _ { \pm 0 . 0 2 }$ </td><td> $1 . 7 4 _ { \pm 0 . 9 0 }$ </td><td> $2 . 4 7 _ { \pm 0 . 4 5 }$ </td><td></td><td></td><td> $1 . 5 1 _ { \pm 0 . 0 5 }$ </td><td> $4 . 1 4 _ { \pm 2 . 9 0 }$ </td><td> $7 . 0 2 _ { \pm 5 . 1 0 }$ </td><td></td><td></td></tr><tr><td>AWT</td><td> $1 4 . 3 8 _ { \pm 0 . 4 4 }$ </td><td> $4 0 . 7 6 _ { \pm 2 0 . 1 1 }$ </td><td> $5 9 . 9 3 _ { \pm 1 0 . 1 1 }$ </td><td></td><td></td><td> $2 2 . 0 9 _ { \pm 0 . 7 4 }$ </td><td> $6 2 . 8 5 _ { \pm 2 8 . 2 5 }$ </td><td> $9 5 . 6 7 _ { \pm 2 5 . 6 9 }$ </td><td></td><td></td></tr><tr><td>TP</td><td> $1 3 6 3 . 5 6 _ { \pm 3 . 0 9 }$ </td><td> $1 3 5 3 . 2 7 _ { \pm 1 5 . 7 3 }$ </td><td> $1 3 4 5 . 1 8 _ { \pm 8 . 5 4 }$ </td><td></td><td></td><td> $2 0 3 9 . 8 8 _ { \pm 4 . 6 7 }$ </td><td> $2 0 0 5 . 6 0 _ { \pm 6 6 . 5 8 }$ </td><td> $1 9 5 5 . 3 7 _ { \pm 1 4 0 . 7 4 }$ </td><td></td><td></td></tr><tr><td rowspan="4">PS- DynSTGAT</td><td>ATT</td><td> $1 8 3 . 4 7 _ { \pm 1 . 2 0 }$ </td><td> $2 1 2 . 8 4 _ { \pm 6 4 . 7 8 }$ </td><td> $2 4 2 . 7 4 _ { \pm 1 5 4 . 2 0 }$ </td><td> $2 1 9 . 6 1 _ { \pm 4 9 . 0 5 }$ </td><td></td><td> $1 8 9 . 7 5 _ { \pm 1 . 7 2 }$ </td><td> $2 2 9 . 7 9 _ { \pm 1 1 7 . 6 0 }$ </td><td>371.22±272.65 259.14±141.07</td><td></td><td></td></tr><tr><td>AQL</td><td> $0 . 7 0 _ { \pm 0 . 0 2 }$ </td><td> $2 . 0 5 _ { \pm 3 . 0 2 }$ </td><td> $3 . 4 2 _ { \pm 7 . 1 2 }$ </td><td> $2 . 3 2 _ { \pm 2 . 2 3 }$ </td><td></td><td> $\underline { { 1 . 3 7 _ { \pm 0 . 0 4 } } }$ </td><td> $4 . 1 0 _ { \pm 8 . 0 1 }$ </td><td> $1 3 . 7 0 _ { \pm 1 8 . 3 9 }$ </td><td> $5 . 9 9 _ { \pm 9 . 3 7 }$ </td><td></td></tr><tr><td>AWT</td><td> $1 4 . 3 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $3 7 . 0 6 { \scriptstyle \pm 2 9 . 8 0 }$ </td><td> $3 3 . 7 9 { \scriptstyle \pm 2 8 . 8 0 }$ </td><td> $4 4 . 4 1 \pm 2 8 . 8 9$ </td><td></td><td> $1 9 . 5 8 _ { \pm 0 . 4 8 }$ </td><td> $3 7 . 3 9 _ { \pm 4 0 . 9 6 }$ </td><td> $6 2 . 7 4 { \scriptstyle \pm 3 4 . 0 4 }$ </td><td>47.12±37.34</td><td></td></tr><tr><td>TP</td><td> $1 3 6 3 . 9 7 _ { \pm 3 . 4 6 }$ </td><td> $1 3 3 1 . 4 5 _ { \pm 8 3 . 0 7 }$ </td><td>1301.15±172.08</td><td> $1 3 3 0 . 3 7 { \scriptstyle \pm 6 2 . 6 5 }$ </td><td></td><td> $2 0 4 0 . 8 3 _ { \pm 4 . 5 5 }$ </td><td> $1 9 7 4 . 7 8 { \scriptstyle \pm 2 1 3 . 6 1 }$ </td><td> $1 7 4 4 . 6 2 _ { \pm 4 6 7 . 1 2 }$ </td><td>1927.89±253.89</td><td></td></tr><tr><td rowspan="4">Het-MAPPO -RNN</td><td>ATT</td><td> $1 8 3 . 5 5 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $1 9 7 . 4 8 _ { \pm 4 2 . 3 1 }$ </td><td> $1 9 7 . 1 6 _ { \pm 2 3 . 0 2 }$ </td><td> $1 9 1 . 8 0 _ { \pm 2 . 7 4 }$ </td><td> $1 9 3 . 1 4 _ { \pm 2 . 7 5 }$ </td><td> $1 8 9 . 4 9 _ { \pm 1 . 7 1 }$ </td><td> $2 1 5 . 3 1 _ { \pm 4 5 . 1 1 }$ </td><td> $2 1 0 . 9 4 _ { \pm 4 5 . 5 2 }$ </td><td> $2 1 2 . 2 8 _ { \pm 5 6 . 5 8 }$ </td><td> $2 0 9 . 7 5 _ { \pm 2 0 . 2 2 }$ </td></tr><tr><td>AQL</td><td> $0 . 6 9 _ { \pm 0 . 0 2 }$ </td><td> $1 . 3 3 { \scriptstyle \pm 1 . 9 5 }$ </td><td> $1 . 3 1 _ { \pm 1 . 0 4 }$ </td><td> $\underline { { 1 . 0 6 _ { \pm 0 . 0 9 } } }$ </td><td> $\underline { { 1 . 1 2 _ { \pm 0 . 1 1 } } }$ </td><td> $\underline { { 1 . 3 7 _ { \pm 0 . 0 4 } } }$ </td><td> $3 . 1 0 _ { \pm 3 . 0 2 }$ </td><td> $2 . 7 8 _ { \pm 3 . 0 4 }$ </td><td> $2 . 8 5 _ { \pm 3 . 8 8 }$ </td><td> $\underline { { 2 . 6 7 _ { \pm 1 . 3 7 } } }$ </td></tr><tr><td>AWT</td><td> $1 4 . 4 0 _ { \pm 0 . 3 5 }$ </td><td> $2 3 . 2 0 _ { \pm 1 1 . 0 3 }$ </td><td> $2 6 . 2 3 _ { \pm 1 8 . 2 8 }$ </td><td> $2 3 . 4 1 _ { \pm 2 . 5 5 }$ </td><td> $2 4 . 6 3 { \scriptstyle \pm 2 . 3 8 }$ </td><td> $1 9 . 4 8 _ { \pm 0 . 6 7 }$ </td><td> $3 4 . 1 2 { \scriptstyle \pm 2 2 . 0 4 }$ </td><td> $3 5 . 3 6 \pm 2 2 . 3 7$ </td><td> $3 7 . 9 9 _ { \pm 1 8 . 2 4 }$ </td><td> $3 9 . 9 2 _ { \pm 1 4 . 5 1 }$ </td></tr><tr><td>TP</td><td> $1 3 6 4 . 0 0 _ { \pm 3 . 9 3 }$ </td><td> $1 3 5 2 . 1 1 _ { \pm 4 9 . 9 5 }$ </td><td> $1 3 5 4 . 8 4 _ { \pm 2 2 . 4 3 }$ </td><td> $1 3 6 0 . 6 4 _ { \pm 3 . 9 9 }$ </td><td> $1 3 6 0 . 0 3 _ { \pm 4 . 6 3 }$ </td><td> $2 0 4 1 . 0 4 _ { \pm 4 . 2 0 }$ </td><td>2009.59±72.74</td><td> $2 0 1 3 . 3 0 _ { \pm 8 4 . 3 2 }$ </td><td>2017.34±106.53</td><td> $2 0 2 2 . 3 3 _ { \pm 3 2 . 8 2 }$ </td></tr><tr><td rowspan="4">HAPPO- RNN</td><td>ATT</td><td> $1 8 3 . 6 6 _ { \pm 1 . 5 0 }$ </td><td> $2 4 9 . 5 2 _ { \pm 1 3 0 . 4 5 }$ </td><td> $1 9 6 . 6 6 _ { \pm 1 2 . 0 1 }$ </td><td> $1 9 5 . 3 3 _ { \pm 7 . 2 7 }$ </td><td> $1 9 3 . 7 3 _ { \pm 2 . 2 3 }$ </td><td> $1 8 9 . 7 0 _ { \pm 1 . 8 0 }$ </td><td> $2 6 1 . 3 8 \pm 1 2 4 . 9 9$ </td><td> $2 1 7 . 3 4 _ { \pm 4 9 . 8 7 }$ </td><td> $2 2 0 . 6 4 _ { \pm 7 0 . 0 0 }$ </td><td> $2 1 3 . 3 0 _ { \pm 5 5 . 1 5 }$ </td></tr><tr><td>AQL</td><td> $\underline { { 0 . 6 9 _ { \pm 0 . 0 2 } } }$ </td><td> $3 . 6 7 _ { \pm 5 . 9 1 }$ </td><td> $1 . 2 8 _ { \pm 0 . 5 5 }$ </td><td> $1 . 2 1 _ { \pm 0 . 3 1 }$ </td><td> $1 . 1 4 _ { \pm 0 . 0 7 }$ </td><td> $\underline { { 1 . 3 7 _ { \pm 0 . 0 5 } } }$ </td><td> $6 . 1 5 _ { \pm 8 . 3 2 }$ </td><td> $3 . 1 9 _ { \pm 3 . 2 7 }$ </td><td> $3 . 4 1 _ { \pm 4 . 6 3 }$ </td><td> $2 . 9 1 _ { \pm 3 . 7 7 }$ </td></tr><tr><td>AWT</td><td> $1 4 . 4 1 _ { \pm 0 . 3 2 }$ </td><td> $3 8 . 7 9 _ { \pm 2 0 . 9 5 }$ </td><td> $2 6 . 7 4 _ { \pm 9 . 4 8 }$ </td><td> $2 7 . 0 5 _ { \pm 6 . 7 6 }$ </td><td> $2 5 . 3 8 _ { \pm 1 . 9 5 }$ </td><td> $1 9 . 5 6 _ { \pm 0 . 7 4 }$ </td><td> $5 1 . 8 9 _ { \pm 3 6 . 4 1 }$ </td><td> $4 1 . 5 8 _ { \pm 2 1 . 1 1 }$ </td><td> $4 1 . 4 0 _ { \pm 1 7 . 9 1 }$ </td><td> $3 8 . 1 4 _ { \pm 1 5 . 5 5 }$ </td></tr><tr><td>TP</td><td> $1 3 6 3 . 9 8 _ { \pm 3 . 7 9 }$ </td><td> $1 2 9 9 . 8 5 _ { \pm 1 4 7 . 0 3 }$ </td><td> $1 3 5 6 . 7 0 { \scriptstyle \pm 1 0 . 3 5 }$ </td><td> $1 3 5 8 . 4 8 _ { \pm 6 . 8 9 }$ </td><td> $1 3 5 9 . 3 5 _ { \pm 4 . 3 0 }$ </td><td> $2 0 4 0 . 7 2 _ { \pm 4 . 1 9 }$ </td><td> $1 9 3 6 . 5 3 _ { \pm 2 0 5 . 4 8 } 2 0 1 0 . 2 6 _ { \pm 8 1 . 5 8 }$ </td><td></td><td></td><td>2003.56±123.912015.76±101.3</td></tr><tr><td rowspan="4">HALight</td><td>ATT</td><td> $\mathbf { 1 8 3 . 0 0 _ { \pm 1 . 7 8 } }$ </td><td> $\mathbf { 1 8 9 . 8 8 _ { \pm 6 . 7 3 } }$ </td><td> ${ \bf 1 8 9 . 8 7 } _ { \pm 2 . 7 9 }$ </td><td> $\mathbf { 1 9 0 . 7 4 _ { \pm 2 . 4 3 } }$ </td><td> $\mathbf { 1 9 1 . 9 8 _ { \pm 2 . 3 2 } }$ </td><td> ${ \bf 1 8 9 . 3 9 { \scriptstyle \pm 1 . 7 8 } }$ </td><td> ${ \bf 1 9 9 . 7 1 _ { \pm 1 1 . 8 2 } }$ </td><td> $\pm 9 9 . 2 5 _ { \pm 8 . 7 0 }$ </td><td> ${ \bf 2 0 8 . 0 2 _ { \pm 2 4 . 1 5 } }$ </td><td> $\mathbf { 2 0 8 . 0 2 _ { \pm 2 8 . 1 5 } }$ </td></tr><tr><td>AQL</td><td> $\mathbf { 0 . 6 8 _ { \pm 0 . 0 2 } }$ </td><td> $\mathbf { 0 . 9 8 _ { \pm 0 . 3 0 } }$ </td><td> ${ \bf 0 . 9 8 _ { \pm 0 . 0 9 } }$ </td><td> ${ \bf 1 . 0 1 _ { \pm 0 . 0 7 } }$ </td><td> ${ \bf 1 . 0 7 _ { \pm 0 . 0 7 } }$ </td><td> ${ \bf 1 . 3 6 _ { \pm 0 . 0 5 } }$ </td><td> $\mathbf { 2 . 0 4 _ { \pm 0 . 7 8 } }$ </td><td> ${ \bf 1 . 9 9 _ { \pm 0 . 5 9 } }$ </td><td> $2 . 5 7 _ { \pm 1 . 6 2 }$ </td><td> $2 . 5 7 _ { \pm 1 . 9 5 }$ </td></tr><tr><td>AWT</td><td> $\mathbf { 1 4 . 3 3 _ { \pm 0 . 4 8 } }$ </td><td> ${ \bf 2 0 . 8 0 _ { \pm 4 . 5 4 } }$ </td><td> $2 1 . 4 2 _ { \pm 2 . 5 7 }$ </td><td> $2 2 . 2 0 _ { \pm 1 . 9 5 }$ </td><td> ${ \pm 3 . 6 1 } _ { \pm 1 . 9 3 }$ </td><td> $\mathbf { 1 9 . 4 3 _ { \pm 0 . 7 3 } }$ </td><td> $2 8 . 6 4 _ { \pm 9 . 2 2 }$ </td><td> $\mathbf { 2 9 . 3 8 _ { \pm 6 . 8 3 } }$ </td><td> ${ 3 7 . 7 7 } _ { \pm 1 5 . 8 8 }$ </td><td> $3 6 . 6 3 _ { \pm 9 . 9 9 }$ </td></tr><tr><td>TP</td><td> $\mathbf { 1 3 6 4 . 0 1 } _ { \pm 3 . 6 7 }$ </td><td> $\mathbf { 1 3 6 0 . 4 4 _ { \pm 8 . 2 4 } }$ </td><td>1361.41±4.00</td><td> $\mathbf { 1 } 3 6 0 . 9 9 _ { \pm 3 . 9 4 }$ </td><td>1360.32±4.30</td><td> $\mathbf { 2 0 4 1 . 1 7 _ { \pm 4 . 7 4 } }$ </td><td> ${ \bf 2 0 3 3 . 0 2 _ { \pm 1 3 . 6 5 } }$ </td><td> ${ \bf 2 0 3 3 . 2 6 _ { \pm 2 1 . 3 5 } }$ </td><td>2025.08±38.37</td><td> $2 0 2 3 . 2 0 _ { \pm 6 4 . 0 4 }$ </td></tr></table>

• Het-MAPPO-RNN [49]: A heterogeneous variant of MAPPO, where each intersection is equipped with an independently modeled actor network to accommodate intersection-specific observation and action spaces, while all agents share a centralized critic for value estimation.

• HAPPO-RNN [38]: A PPO based method for heterogeneous agents. It uses the multi-agent advantage decomposition lemma and sequential updates to support monotonic joint policy improvement.

discount factor γ is set to 0.95. Each episode is set to simulate one hour of real time traffic. All agents execute actions every $\Delta t \ = \ 1 0$ seconds, and there is no yellow phase between consecutive signal phases. Thus, the length $T$ of each episode is 360. All algorithms are optimized using the Adam [50] optimizer. The hyperparameters used for training the models are listed in Table II. For HALight, the latent dimension $d _ { h }$ and the attention dimension $d _ { a }$ are both set to 16, while the information dimension $d _ { \mathrm { i n f o } }$ is set to 8. All experiments are conducted using the CityFlow [51] simulation platform on a Linux server equipped with an Intel(R) Xeon(R) Gold 6226R CPU and an NVIDIA GeForce RTX 3090 GPU.

The common experimental settings are as follows: the number of episodes is set to 10000, with the observation missing rate varying from 0% to 40% in increments of 10%. For each road network and demand level, we generate 100 traffic flow files. During training, each episode randomly samples one flow file, whereas evaluation is conducted over all 100 flow files. The missing observation mask is resampled at the beginning of each episode to cover diverse lane-level missing patterns. The

## C. Metric

We adopt four metrics to evaluate the agents’ performance. All time-related metrics are measured in seconds.

• Average Travel Time (ATT): ATT measures the average time it takes for vehicles to travel from their origin to the destination. It helps evaluate how efficiently the traffic control system allows vehicles to complete their journey.

TABLE IV  
OVERALL PERFORMANCE OF HALIGHT AND BASELINE METHODS UNDER DIFFERENT p<sub>miss</sub> ON THE GUDANG ROAD NETWORK. THE BEST RESULTS AREHIGHLIGHTED IN BOLD, AND THE SECOND-BEST RESULTS ARE UNDERLINED. RESULTS FOR METHODS THAT FAIL TO CONVERGE ARE OMITTED.
<table><tr><td rowspan="3">Algorithms</td><td rowspan="3">Metric</td><td colspan="9">Scenario</td></tr><tr><td colspan="4">Moderate Demand</td><td colspan="6"></td></tr><tr><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td rowspan="4">Fixed Time</td><td>ATT</td><td> $3 0 8 . 1 9 _ { \pm 1 . 7 3 }$ </td><td> $3 0 8 . 1 9 _ { \pm 1 . 7 3 }$ </td><td> $3 0 8 . 1 9 _ { \pm 1 . 7 3 }$ </td><td> $3 0 8 . 1 9 _ { \pm 1 . 7 3 }$ </td><td> $3 0 8 . 1 9 _ { \pm 1 . 7 3 }$ </td><td> $3 3 2 . 9 8 _ { \pm 5 . 2 9 }$ </td><td> $3 3 2 . 9 8 _ { \pm 5 . 2 9 }$ </td><td> $3 3 2 . 9 8 _ { \pm 5 . 2 9 }$ </td><td> $3 3 2 . 9 8 _ { \pm 5 . 2 9 }$ </td><td> $3 3 2 . 9 8 _ { \pm 5 . 2 9 }$ </td></tr><tr><td>AQL</td><td> $1 . 8 0 _ { \pm 0 . 0 3 }$ </td><td> $1 . 8 0 _ { \pm 0 . 0 3 }$ </td><td> $1 . 8 0 _ { \pm 0 . 0 3 }$ </td><td> $1 . 8 0 _ { \pm 0 . 0 3 }$ </td><td> $1 . 8 0 _ { \pm 0 . 0 3 }$ </td><td> $5 . 1 1 _ { \pm 0 . 3 9 }$ </td><td> $5 . 1 1 _ { \pm 0 . 3 9 }$ </td><td> $5 . 1 1 _ { \pm 0 . 3 9 }$ </td><td> $5 . 1 1 _ { \pm 0 . 3 9 }$ </td><td> $5 . 1 1 _ { \pm 0 . 3 9 }$ </td></tr><tr><td>AWT</td><td> $3 3 . 2 4 _ { \pm 0 . 5 9 }$ </td><td> $3 3 . 2 4 _ { \pm 0 . 5 9 }$ </td><td> $3 3 . 2 4 _ { \pm 0 . 5 9 }$ </td><td> $3 3 . 2 4 _ { \pm 0 . 5 9 }$ </td><td> $3 3 . 2 4 _ { \pm 0 . 5 9 }$ </td><td> $5 5 . 5 5 _ { \pm 4 . 2 9 }$ </td><td> $5 5 . 5 5 _ { \pm 4 . 2 9 }$ </td><td> $5 5 . 5 5 _ { \pm 4 . 2 9 }$ </td><td> $5 5 . 5 5 _ { \pm 4 . 2 9 }$ </td><td> $5 5 . 5 5 _ { \pm 4 . 2 9 }$ </td></tr><tr><td>TP</td><td> $2 7 8 4 . 8 6 _ { \pm 6 . 8 3 }$ </td><td> $2 7 8 4 . 8 6 _ { \pm 6 . 8 3 }$ </td><td> $2 7 8 4 . 8 6 _ { \pm 6 . 8 3 }$ </td><td> $2 7 8 4 . 8 6 _ { \pm 6 . 8 3 }$ </td><td> $2 7 8 4 . 8 6 _ { \pm 6 . 8 3 }$ </td><td>4856.63±19.56</td><td> $4 8 5 6 . 6 3 _ { \pm 1 9 . 5 6 }$ </td><td>4856.63±19.564856.63±19.56</td><td></td><td> $4 8 5 6 . 6 3 _ { \pm 1 9 . 5 6 }$ </td></tr><tr><td rowspan="4">PS-RDQN</td><td>ATT</td><td> $2 9 5 . 2 6 _ { \pm 1 . 5 0 }$ </td><td> $3 0 1 . 1 1 _ { \pm 4 . 6 4 }$ </td><td> $3 0 5 . 6 4 _ { \pm 4 . 7 4 }$ </td><td> $3 0 6 . 6 1 _ { \pm 3 . 3 8 }$ </td><td> $3 1 1 . 9 8 _ { \pm 5 . 2 3 }$ </td><td> $3 0 4 . 2 1 \pm 1 . 7 4$ </td><td> $3 1 7 . 4 5 _ { \pm 2 9 . 9 9 }$ </td><td> $3 5 9 . 6 0 _ { \pm 6 2 . 3 5 }$ </td><td> $3 8 3 . 9 8 _ { \pm 4 9 . 8 6 }$ </td><td></td></tr><tr><td>AQL</td><td> $\underline { { 1 . 1 3 _ { \pm 0 . 0 2 } } }$ </td><td> $1 . 4 6 _ { \pm 0 . 2 3 }$ </td><td> $1 . 7 0 _ { \pm 0 . 2 4 }$ </td><td> $1 . 7 5 _ { \pm 0 . 1 7 }$ </td><td> $2 . 0 4 _ { \pm 0 . 3 8 }$ </td><td> $2 . 6 7 _ { \pm 0 . 0 5 }$ </td><td> $3 . 9 4 _ { \pm 3 . 2 2 }$ </td><td> $8 . 1 4 _ { \pm 6 . 6 6 }$ </td><td> $1 0 . 5 7 _ { \pm 5 . 4 5 }$ </td><td></td></tr><tr><td>AWT</td><td> $1 9 . 1 3 _ { \pm 0 . 3 0 }$ </td><td> $2 5 . 7 3 _ { \pm 4 . 6 3 }$ </td><td> $3 0 . 5 7 _ { \pm 4 . 6 1 }$ </td><td> $3 2 . 1 2 _ { \pm 3 . 7 1 }$ </td><td> $3 7 . 3 9 _ { \pm 5 . 7 0 }$ </td><td> $2 6 . 5 4 _ { \pm 0 . 4 8 }$ </td><td> $3 6 . 8 5 _ { \pm 7 . 3 9 }$ </td><td> $6 4 . 2 2 _ { \pm 1 5 . 5 6 }$ </td><td> $8 5 . 8 6 _ { \pm 1 7 . 7 5 }$ </td><td></td></tr><tr><td>TP</td><td> $2 7 9 7 . 1 8 _ { \pm 5 . 1 4 }$ </td><td> $2 7 9 0 . 6 8 _ { \pm 8 . 7 8 }$ </td><td> $2 7 8 7 . 2 0 _ { \pm 9 . 9 0 }$ </td><td> $2 7 8 5 . 7 4 _ { \pm 7 . 0 0 }$ </td><td> $2 7 8 0 . 3 5 _ { \pm 9 . 9 0 }$ </td><td> $4 9 1 9 . 0 6 _ { \pm 7 . 6 1 }$ </td><td> $4 8 8 0 . 2 4 _ { \pm 1 4 8 . 6 2 }$ </td><td> $4 7 2 6 . 9 3 _ { \pm 3 2 7 . 4 1 }$ </td><td>4648.19±290.50</td><td></td></tr><tr><td rowspan="4">PS-PressLight -RNN</td><td>ATT</td><td> $2 9 5 . 4 6 _ { \pm 1 . 0 8 }$ </td><td> $3 0 1 . 3 7 { \scriptstyle \pm 2 . 9 9 }$ </td><td> $3 1 2 . 9 6 _ { \pm 2 5 . 9 3 }$ </td><td> $3 0 8 . 1 8 _ { \pm 2 . 7 5 }$ </td><td></td><td> $3 0 5 . 6 3 _ { \pm 1 . 6 4 }$ </td><td> $3 3 2 . 5 0 _ { \pm 9 . 6 6 }$ </td><td> $3 6 3 . 4 6 _ { \pm 2 1 . 5 1 }$ </td><td></td><td></td></tr><tr><td>AQL</td><td> $1 . 1 4 _ { \pm 0 . 0 2 }$ </td><td> $1 . 4 7 _ { \pm 0 . 1 5 }$ </td><td> $2 . 1 1 \pm 1 . 5 6$ </td><td> $1 . 8 3 _ { \pm 0 . 1 3 }$ </td><td></td><td> $2 . 8 0 _ { \pm 0 . 0 4 }$ </td><td> $5 . 2 6 _ { \pm 0 . 9 0 }$ </td><td> $8 . 2 0 _ { \pm 2 . 1 6 }$ </td><td></td><td></td></tr><tr><td>AWT</td><td> $1 9 . 3 1 _ { \pm 0 . 3 2 }$ </td><td> $2 5 . 7 9 _ { \pm 2 . 9 5 }$ </td><td> $3 4 . 5 5 _ { \pm 3 . 6 4 }$ </td><td> $3 3 . 3 6 _ { \pm 2 . 5 3 }$ </td><td></td><td> $2 8 . 4 3 _ { \pm 0 . 4 5 }$ </td><td> $5 7 . 3 4 _ { \pm 8 . 7 1 }$ </td><td> $8 5 . 5 2 _ { \pm 1 4 . 0 3 }$ </td><td></td><td></td></tr><tr><td>TP</td><td> $2 7 9 6 . 7 4 _ { \pm 6 . 6 8 }$ </td><td> $2 7 9 1 . 0 1 { \scriptstyle \pm 7 . 8 8 }$ </td><td> $2 7 7 3 . 8 8 _ { \pm 6 8 . 9 2 }$ </td><td> $2 7 8 4 . 7 4 _ { \pm 8 . 6 3 }$ </td><td></td><td> $4 9 1 8 . 1 9 _ { \pm 8 . 9 9 }$ </td><td> $4 8 6 6 . 5 6 _ { \pm 2 7 . 3 9 }$ </td><td> $4 7 8 6 . 0 5 _ { \pm 8 6 . 3 1 }$ </td><td></td><td></td></tr><tr><td rowspan="4">PS- DynSTGAT</td><td>ATT</td><td> ${ \bf 2 9 4 . 8 6 _ { \pm 1 . 5 4 } }$ </td><td> $3 0 3 . 4 9 _ { \pm 7 . 0 1 }$ </td><td> $3 1 5 . 7 3 _ { \pm 2 3 . 4 9 }$ </td><td> $3 0 9 . 0 5 _ { \pm 8 . 9 8 }$ </td><td> $3 0 7 . 6 1 _ { \pm 4 . 3 8 }$ </td><td> $\mathbf { 3 0 3 . 8 6 _ { \pm 1 . 6 4 } }$ </td><td> $3 5 1 . 5 4 _ { \pm 9 0 . 6 2 }$ </td><td> $3 6 5 . 0 3 _ { \pm 7 0 . 4 1 }$ </td><td></td><td></td></tr><tr><td>AQL</td><td> ${ \bf 1 . 0 9 _ { \pm 0 . 0 2 } }$ </td><td> $1 . 6 0 _ { \pm 0 . 4 0 }$ </td><td> $2 . 2 7 _ { \pm 1 . 3 5 }$ </td><td> $1 . 8 9 _ { \pm 0 . 5 2 }$ </td><td> $1 . 8 0 _ { \pm 0 . 2 3 }$ </td><td> $\pm . 6 2 _ { \pm 0 . 0 4 }$ </td><td> $7 . 5 0 { \scriptstyle \pm 9 . 4 1 }$ </td><td> $8 . 7 2 _ { \pm 7 . 2 8 }$ </td><td></td><td></td></tr><tr><td>AWT</td><td> $1 9 . 0 8 _ { \pm 0 . 3 3 }$ </td><td> $2 7 . 8 2 _ { \pm 6 . 0 3 }$ </td><td> $3 6 . 6 3 _ { \pm 1 2 . 1 4 }$ </td><td> $3 4 . 0 5 _ { \pm 6 . 8 0 }$ </td><td> $3 2 . 8 4 _ { \pm 4 . 0 8 }$ </td><td> $2 6 . 5 2 _ { \pm 0 . 4 3 }$ </td><td> $3 8 . 8 3 _ { \pm 1 4 . 0 9 }$ </td><td> $5 9 . 5 8 _ { \pm 2 1 . 3 6 }$ </td><td></td><td></td></tr><tr><td>TP</td><td> $2 7 9 7 . 3 2 _ { \pm 4 . 7 8 }$ </td><td> $2 7 8 6 . 5 7 { \scriptstyle \pm 1 4 . 2 0 }$ </td><td> $2 7 6 7 . 6 8 _ { \pm 5 6 . 4 9 }$ </td><td> $2 7 8 2 . 6 7 _ { \pm 1 7 . 8 3 }$ </td><td> $2 7 8 4 . 4 3 _ { \pm 9 . 3 0 }$ </td><td> $4 9 1 9 . 4 2 _ { \pm 8 . 0 2 }$ </td><td> $4 7 0 0 . 9 1 _ { \pm 4 4 9 . 3 7 } 4 6 7 8 . 0 4 _ { \pm 3 4 9 . 5 9 }$ </td><td></td><td></td><td></td></tr><tr><td rowspan="4">Het-MAPPO -RNN</td><td>ATT</td><td> $2 9 5 . 5 5 _ { \pm 1 . 3 2 }$ </td><td> $3 0 1 . 0 1 _ { \pm 2 . 9 0 }$ </td><td> $\underline { { 3 0 2 . 9 0 _ { \pm 5 . 1 8 } } }$ </td><td> $\underline { { 3 0 3 . 1 5 _ { \pm 2 . 0 1 } } }$ </td><td> $\underline { { 3 0 5 . 0 4 _ { \pm 2 . 5 2 } } }$ </td><td> $3 0 4 . 7 6 _ { \pm 1 . 3 7 }$ </td><td> $\underline { { 3 1 2 . 8 8 _ { \pm 1 1 . 3 5 } } }$ </td><td> $3 1 3 . 9 3 _ { \pm 8 . 8 5 }$ </td><td> $3 1 6 . 2 3 _ { \pm 7 . 9 7 }$ </td><td> $3 1 6 . 9 6 _ { \pm 5 . 4 2 }$ </td></tr><tr><td>AQL</td><td> $1 . 1 4 _ { \pm 0 . 0 2 }$ </td><td> $1 . 4 5 _ { \pm 0 . 1 3 }$ </td><td> $1 . 5 5 _ { \pm 0 . 2 7 }$ </td><td> $1 . 5 6 _ { \pm 0 . 0 8 }$ </td><td> $1 . 6 6 _ { \pm 0 . 1 1 }$ </td><td> $2 . 6 6 _ { \pm 0 . 0 3 }$ </td><td> $3 . 4 5 _ { \pm 1 . 1 5 }$ </td><td> $3 . 5 5 _ { \pm 0 . 8 7 }$ </td><td> $3 . 7 8 _ { \pm 0 . 7 8 }$ </td><td> $3 . 8 4 _ { \pm 0 . 5 0 }$ </td></tr><tr><td>AWT</td><td> $1 9 . 1 4 _ { \pm 0 . 1 3 }$ </td><td> $2 5 . 5 5 { \scriptstyle \pm 2 . 7 8 }$ </td><td> $\underline { { 2 7 . 3 1 \pm 3 . 9 1 } }$ </td><td> $2 7 . 9 4 \pm 1 . 7 2 $ </td><td> $\underline { { 3 0 . 1 7 \pm 2 . 3 6 } }$ </td><td> $2 6 . 5 8 _ { \pm 0 . 4 0 }$ </td><td> $3 4 . 6 9 _ { \pm 6 . 1 2 }$ </td><td> ${ \underline { { 3 6 . 2 3 } } } _ { \pm 5 . 0 9 }$ </td><td> $3 8 . 5 8 _ { \pm 5 . 0 4 }$ </td><td> $3 9 . 9 4 _ { \pm 4 . 5 4 }$ </td></tr><tr><td>TP</td><td> $2 7 9 7 . 2 9 _ { \pm 8 . 7 9 }$ </td><td> $2 7 9 1 . 1 4 { \scriptstyle \pm 7 . 3 3 }$ </td><td>2790.06±10.31</td><td> $2 7 9 0 . 1 8 { \scriptstyle \pm 6 . 9 4 }$ </td><td> $2 7 8 7 . 9 0 _ { \pm 7 . 6 0 }$ </td><td> $4 9 1 9 . 2 3 \pm 9 . 4 2$ </td><td>4898.08±54.14</td><td> $4 9 0 0 . 6 3 \pm 2 8 . 2 4$ </td><td> $4 8 9 3 . 6 3 _ { \pm 3 5 . 9 0 }$ </td><td> $4 8 9 5 . 0 1 _ { \pm 2 0 . 0 9 }$ </td></tr><tr><td rowspan="4">HAPPO- RNN</td><td>ATT</td><td> $2 9 5 . 4 8 _ { \pm 1 . 4 8 }$ </td><td> $3 1 3 . 0 9 _ { \pm 1 3 . 3 8 }$ </td><td> $3 2 0 . 7 9 _ { \pm 1 7 . 0 8 }$ </td><td> $3 2 9 . 6 4 _ { \pm 2 7 . 3 8 }$ </td><td> $3 1 1 . 3 2 _ { \pm 4 . 7 0 }$ </td><td> $3 0 4 . 9 5 _ { \pm 1 . 7 2 }$ </td><td> $3 6 3 . 1 7 _ { \pm 5 6 . 5 4 }$ </td><td> $3 6 7 . 2 6 { \scriptstyle \pm 4 8 . 8 8 }$ </td><td> $3 7 2 . 5 4 _ { \pm 4 7 . 8 1 }$ </td><td> $3 9 1 . 5 2 _ { \pm 4 3 . 0 6 }$ </td></tr><tr><td>AQL</td><td> $1 . 1 5 _ { \pm 0 . 0 2 }$ </td><td> $2 . 1 0 _ { \pm 0 . 7 4 }$ </td><td> $2 . 5 3 _ { \pm 0 . 9 6 }$ </td><td> $3 . 0 2 _ { \pm 1 . 5 5 }$ </td><td> $2 . 0 1 _ { \pm 0 . 2 5 }$ </td><td> $2 . 7 2 _ { \pm 0 . 0 5 }$ </td><td> $8 . 4 9 _ { \pm 5 . 7 8 }$ </td><td> $8 . 8 6 _ { \pm 5 . 0 8 }$ </td><td> $9 . 3 3 _ { \pm 4 . 9 7 }$ </td><td> $1 1 . 1 2 _ { \pm 4 . 3 9 }$ </td></tr><tr><td>AWT</td><td> $1 9 . 2 6 _ { \pm 0 . 3 2 }$ </td><td> $3 7 . 4 3 _ { \pm 1 0 . 9 1 }$ </td><td> $4 3 . 3 1 _ { \pm 9 . 9 0 }$ </td><td> $5 0 . 4 6 _ { \pm 1 3 . 4 8 }$ </td><td> $3 6 . 3 2 _ { \pm 4 . 1 7 }$ </td><td> $2 7 . 4 2 _ { \pm 0 . 4 9 }$ </td><td> $6 3 . 2 0 _ { \pm 2 3 . 5 1 }$ </td><td> $6 8 . 7 3 _ { \pm 1 7 . 4 4 }$ </td><td> $7 3 . 9 2 _ { \pm 2 0 . 8 7 }$ </td><td> $9 4 . 9 3 _ { \pm 2 3 . 6 4 }$ </td></tr><tr><td>TP</td><td> $2 7 9 7 . 1 7 _ { \pm 6 . 9 0 }$ </td><td> $2 7 7 5 . 1 3 _ { \pm 2 5 . 6 5 }$ </td><td> $2 7 6 4 . 2 0 _ { \pm 3 4 . 7 7 }$ </td><td> $2 7 5 4 . 0 4 _ { \pm 4 9 . 7 5 }$ </td><td> $2 7 8 0 . 9 4 _ { \pm 9 . 1 6 }$ </td><td> $4 9 1 8 . 9 0 _ { \pm 8 . 9 6 }$ </td><td> $4 6 7 6 . 1 5 _ { \pm 2 8 9 . 8 5 }$ </td><td>4684.21±256.17</td><td> $4 6 9 3 . 8 0 _ { \pm 2 3 2 . 5 2 }$ </td><td> $4 6 5 0 . 3 6 _ { \pm 1 9 7 . 9 1 }$ </td></tr><tr><td rowspan="4">HALight</td><td>ATT</td><td> $2 9 5 . 2 0 _ { \pm 1 . 5 5 }$ </td><td> ${ \bf 2 9 9 . 9 3 _ { \pm 2 . 5 0 } }$ </td><td> $\mathbf { 3 0 1 . 7 7 { \scriptstyle \pm 2 . 4 6 } }$ </td><td> ${ \bf 3 0 2 . 8 7 } _ { \pm 2 . 1 1 }$ </td><td> $\mathbf { 3 0 4 . 5 6 } _ { \pm 2 . 3 9 }$ </td><td> $\underline { { 3 0 4 . 1 7 _ { \pm 1 . 7 5 } } }$ </td><td> $\mathbf { 3 1 0 . 3 7 } _ { \pm 4 . 4 0 }$ </td><td> $\pm \mathbf { 1 2 . 8 7 } _ { \pm 4 . 0 0 }$ </td><td> $\mathbf { 3 1 3 . 0 6 } _ { \pm 3 . 2 0 }$ </td><td> $3 1 4 . 3 3 _ { \pm 3 . 3 2 }$ </td></tr><tr><td>AQL</td><td> $1 . 1 3 _ { \pm 0 . 0 2 }$ </td><td> ${ \bf 1 . 3 9 _ { \pm 0 . 1 2 } }$ </td><td> ${ \bf 1 . 4 8 _ { \pm 0 . 1 1 } }$ </td><td> ${ \bf 1 . 5 4 _ { \pm 0 . 0 8 } }$ </td><td> ${ \bf 1 . 6 4 _ { \pm 0 . 0 9 } }$ </td><td> $2 . 6 5 _ { \pm 0 . 0 5 }$ </td><td> $3 . 2 3 _ { \pm 0 . 3 9 }$ </td><td> ${ \bf 3 . 4 5 _ { \pm 0 . 3 3 } }$ </td><td> ${ \bf 3 . 4 8 _ { \pm 0 . 2 5 } }$ </td><td> ${ \phantom { - } 3 . 5 8 \pm 0 . 2 6 }$ </td></tr><tr><td>AWT</td><td> $\mathbf { 1 9 . 0 6 _ { \pm 0 . 3 3 } }$ </td><td> $\mathbf { 2 4 . 1 8 _ { \pm 2 . 0 3 } }$ </td><td> ${ \bf 2 6 . 4 1 } _ { \pm 2 . 2 7 }$ </td><td> $2 7 . 4 6 _ { \pm 1 . 4 4 }$ </td><td> $2 9 . 7 9 _ { \pm 1 . 8 9 }$ </td><td> $\pm 6 . 4 8 _ { \pm 0 . 5 1 }$ </td><td> $3 2 . 7 0 _ { \pm 2 . 9 9 }$ </td><td> $3 5 . 7 7 _ { \pm 3 . 3 3 }$ </td><td> ${ \bf 3 6 . 1 4 } _ { \pm 2 . 5 6 }$ </td><td> $\mathbf { 3 7 . 4 0 } _ { \pm 2 . 2 6 }$ </td></tr><tr><td>TP</td><td> $2 7 9 7 . 3 5 _ { \pm 6 . 7 7 }$ </td><td> $2 7 9 1 . 9 9 _ { \pm 7 . 4 6 }$ </td><td> ${ \bf 2 7 9 1 . 5 1 _ { \pm 6 . 6 0 } }$ </td><td> $2 7 9 0 . 2 7 _ { \pm 7 . 5 3 }$ </td><td> $\mathbf { 2 7 8 8 . 1 0 _ { \pm 6 . 9 8 } }$ </td><td> $\mathbf { 4 9 1 9 . 9 8 _ { \pm 8 . 9 7 } }$ </td><td>4907.28±16.11</td><td> $\mathbf { 4 9 0 3 . 3 2 _ { \pm 1 3 . 2 6 } }$ </td><td>4904.36±11.91</td><td> $\mathbf { 4 9 0 1 . 4 5 _ { \pm 1 3 . 5 6 } }$ </td></tr></table>

across both road networks, traffic demands, and observation missing rates, with its advantage becoming more evident when high traffic demand coincides with partial observability.

• Average Queue Length (AQL): AQL quantifies the average number of vehicles waiting at each intersection throughout the episode. This is an important indicator of congestion. Formally, AQL is defined as the average queue length across all intersections and timesteps.

• Average Waiting Time (AWT): AWT measures the average time that vehicles spend moving at speeds below 5 km/h during their journey. It provides insight into how much time vehicles spend in congested conditions.

• Throughput (TP): TP measures the total number of vehicles that successfully complete their trips and reach their destinations during an episode. It is used to evaluate the overall traffic handling capacity.

Under full observability $( p _ { \mathrm { m i s s } } = 0 \% )$ , both homogeneous and heterogeneous methods learn effective control policies. On the Gudang road network, PS-DynSTGAT achieves the best ATT and AQL among the RL-based methods under full observability, whereas heterogeneous methods generally perform better on the more structurally diverse Xiasha road network. As $p _ { \mathrm { m i s s } }$ increases, however, homogeneous baselines become substantially more sensitive to missing observations and even fail to converge in several settings. By comparison, HALight and Het-MAPPO-RNN remain considerably more stable. This suggests that intersection-specific observation and action modeling enhances adaptation to both observation missingness and structural heterogeneity, allowing each agent to learn a policy better suited to its local intersection characteristics.

## D. Performance Comparison

The quantitative results are summarized in Tables III and IV, while the learning curves of the RL-based methods are shown in Fig. 6. Overall, HALight achieves strong performance

Nevertheless, heterogeneous modeling alone is insufficient to ensure robustness under partial observability. HAPPO-RNN becomes unstable in several missing-observation scenarios;

![](images/7798b44e12661409d75c829b2fcd1824af6d35519bc0f1d0d8c4d970c96a7d64.jpg)

![](images/71d192d391e84570d2259f2b7f0e89cfc4fd4ce8446e197ddf3d68f1bf0eed52.jpg)  
Fig. 7. Ablation results of AWT on the Xiasha road network under different observation missing rates: (a) moderate demand and (b) high demand.

TABLE V  
CONFIGURATION OF ABLATION VARIANTS. HOMO AND HETERO DENOTE HOMOGENEOUS AND HETEROGENEOUS POLICY MODELING.
<table><tr><td>Variant</td><td>Policy Modeling</td><td>Comm</td><td>RNN</td><td>Sequential Update</td></tr><tr><td>PS-MAPPO-RNN</td><td>Homo</td><td>×</td><td>√</td><td>×</td></tr><tr><td>PS-MAPPO-RNN+Comm</td><td>Homo</td><td>√</td><td>√</td><td>×</td></tr><tr><td>Het-MAPPO-RNN</td><td>Hetero</td><td>×</td><td>√</td><td>×</td></tr><tr><td>Het-MAPPO-RNN+Comm</td><td>Hetero</td><td>√</td><td>√</td><td>×</td></tr><tr><td>HAPPO-RNN</td><td>Hetero</td><td>×</td><td>√</td><td>√</td></tr><tr><td>HALight-Linear</td><td>Hetero</td><td>√</td><td>×</td><td>√</td></tr><tr><td>HALight</td><td>Hetero</td><td>√</td><td>√</td><td>√</td></tr></table>

for example, on Gudang road network under high demand, its AQL increases from 2.72 at $p _ { \mathrm { m i s s } } ~ = ~ 0 \%$ to 11.12 at 40%. A plausible explanation is that incomplete observations increase policy uncertainty. This uncertainty may propagate through sequential policy updates as each agent accounts for previously updated policies. Het-MAPPO-RNN adopts independent policy updates and remains more stable in these settings. This indicates that the effectiveness of sequential updates can be significantly affected by partial observability and requires reliable complementary information.

HALight provides such complementary information through recurrent temporal modeling and one-hop neighborhood communication, and consequently achieves the most consistent performance under missing observations. Compared with Het-MAPPO-RNN, HALight generally obtains lower AQL together with improvements in ATT and AWT. On the Gudang road network under high demand, HALight reduces AQL by 6.38%, 2.82%, 7.94%, and 6.77% relative to Het-MAPPO-RNN at $p _ { \mathrm { m i s s } }$ equals to 10%, 20%, 30%, and 40%. Notably, even on the highly heterogeneous Xiasha road network under high demand and severe observation missingness, HALight maintains competitive average performance despite increased variance. At $ { p _ { \mathrm { m i s s } } } \ = \ 4 0 \%$ , Fixed Time achieves a slightly higher TP than HALight, whereas HALight still achieves lower ATT, AQL, and AWT. Thus, HALight maintains comparable throughput while retaining advantages in the other evaluated metrics under this challenging setting. The learning curves in Fig. 6 further show that HALight generally maintains stable learning and high converged rewards as $p _ { \mathrm { m i s s } }$ increases. Unlike parameter-sharing methods, HALight learns intersectionspecific policies without cross-agent experience sharing. This introduces a trade-off between experience utilization and policy specialization: the lack of cross-agent experience sharing reduces the benefit of collective experiences, especially in road networks where only a few intersections share similar structural characteristics. In contrast, independent policies allow agents to preserve intersection-specific behaviors and avoid policy averaging, leading to improved robustness under heterogeneous and partially observable environments.

## E. Ablation Study

To evaluate the contribution of different components in HALight, we conduct ablation experiments on the Xiasha road network under moderate and high traffic demands, with $p _ { \mathrm { m i s s } }$ ranging from 10% to 40%. As summarized in Table V, seven variants are constructed based on four design factors. “Policy Modeling” denotes whether homogeneous or heterogeneous policies are adopted, “Comm” indicates the use of one-hop neighborhood communication, “RNN” represents recurrent temporal modeling, and “Sequential Update” denotes whether sequential policy updates are employed. The corresponding AWT distributions are reported in Fig. 7.

• Heterogeneous policy modeling. Compared with PS-MAPPO-RNN and PS-MAPPO-RNN+Comm, their heterogeneous counterparts generally achieve better performance, particularly under more challenging conditions. This indicates that intersection-specific policies better accommodate structural differences among intersections.

• One-hop neighborhood communication. The effect of communication can be evaluated by comparing PS-MAPPO-RNN with PS-MAPPO-RNN+Comm, Het-MAPPO-RNN with Het-MAPPO-RNN+Comm, and HAPPO-RNN with HALight. For both homogeneous and heterogeneous MAPPO variants, introducing communication generally improves performance or reduces performance variation under missing observations. The improvement is more evident when comparing HAPPO-RNN with HALight, where HALight achieves lower AWT and a more concentrated distribution in most settings. These results show that one-hop neighboring

(a)

![](images/f765b0c15d628df31cbccb9ed6fe5dbae971535c2dd9142f856a28a85f52f76f.jpg)

![](images/57feb7b9425791c00e738ab01c56955033de4706913769a4708d633ba2ae0577.jpg)

Fig. 8. AQL performance under direction-level missing observations on the Xiasha road network: (a) moderate demand and (b) high demand.  
![](images/fb7a7c14d3d119aa2165975ece714c094012ed6dc8a8c0e49d1e0c2c815aea3b.jpg)

![](images/1f63c76046d498bddc28211d000e930f3629ee4d3ae5b98dc40bf4e372e13cad.jpg)

![](images/21ed54a4351efc833f9799c41fdb2c7aedfed8b195e68350ce71a06b5310f84b.jpg)  
Fig. 9. ATT of HALight under different communication conditions: (a) packet loss, (b) communication delay, and (c) communication interval.

communication can alleviate the negative effects of incomplete local observations and improve robustness.

• Recurrent temporal modeling. HALight-Linear replaces the recurrent layer in HALight with a linear layer while retaining the other components. Compared with HALight-Linear, HALight achieves lower median AWT and more concentrated distributions, especially under high demand. This indicates that recurrent modeling enables agents to aggregate historical observations and capture temporal dependencies. It helps compensate for missing information and learn richer latent representations beyond the spatial context provided by communication.

• Sequential policy update. The effect of sequential policy updates can be examined by comparing HAPPO-RNN with Het-MAPPO-RNN, as well as HALight with Het-MAPPO-RNN+Comm. Without communication, HAPPO-RNN exhibits larger performance variations than Het-MAPPO-RNN in several settings, indicating that sequential updates can be sensitive to incomplete observations. With one-hop neighborhood communication, HALight generally achieves better and more stable performance than Het-MAPPO-RNN+Comm. These results indicate that, although sequential updates are theoretically analyzed under full observability, they can still effectively facilitate heterogeneous-agent coordination in partially observable scenarios when complemented with sufficient spatial and temporal information.

Taken together, the ablation results demonstrate that these designs jointly improve the robustness and effectiveness of HALight under lane-level missing observations.

## F. Performance under Direction-Level Missing Observations

To further evaluate performance under correlated observation missingness, we consider a direction-level missing setting on the Xiasha road network. Such grouped observation missingness is common in practice, where sensing failures or coverage limitations may simultaneously affect multiple lanes belonging to the same traffic direction. We introduce a direction-level missing probability $p _ { \mathrm { d i f } }$ . At each intersection, with probability $p _ { \mathrm { d i r } }$ , one traffic direction is randomly selected and the observations of all lanes in that direction are masked simultaneously. We compare HALight with PS-DynSTGAT and Het-MAPPO-RNN, which are the best-performing homogeneous and heterogeneous baselines, respectively. Both moderate and high traffic demands are considered, with $p _ { \mathrm { d i r } }$ ranging from 0% to 30%. The corresponding AQL results are shown in Fig. 8.

Under moderate demand, HALight maintains consistently lower AQL across different direction-level missing probabilities compared with PS-DynSTGAT and Het-MAPPO-RNN. The advantage of HALight becomes more evident under high demand, where the two baselines experience substantial performance degradation as $p _ { \mathrm { d i r } }$ increases. In comparison, HALight achieves the best AQL performance across all direction-level observation missing scenarios.

These results show that HALight maintains strong performance under direction-level missing observations. By leveraging temporal information and one-hop neighborhood communication, HALight handles correlated missing observations across multiple lanes of the same direction, demonstrating adaptability to more structured missing patterns.

## G. Robustness to Communication Imperfections

Reliable agent communication cannot always be guaranteed in practical deployments, as message transmission can be affected by bandwidth limitations, packet loss, and communication delays. To evaluate HALight under such nonideal communication conditions, we introduce three types of communication perturbations during evaluation: packet loss, communication delay, and communication interval. For packet loss, the information received from the affected neighboring agent is replaced with an all-zero vector. HALight is trained under ideal communication conditions, whereas communication perturbations are introduced only during validation. We select the Gudang road network under high demand with $p _ { \mathrm { m i s s } } = 4 0 \%$ as the evaluation scenario. Specifically, the packet loss rate is varied from 0% to 50%, the communication delay from 0 to 5 steps, and the communication interval from 1 to 5 steps. The corresponding ATT results are shown in Fig. 9.

HALight maintains relatively stable performance across all three communication perturbations. With packet loss rates up to 50%, ATT remains close to the loss-free setting. Communication delays of up to 5 steps also cause only limited degradation. Increasing the communication interval has a modest effect in most cases, with a larger ATT increase observed at 5 steps. Across all settings, the ATT change remains below approximately 0.6s compared with ideal communication.

These results demonstrate that HALight remains robust to communication imperfections after training, with only minor performance degradation under packet loss, communication delay, and reduced communication frequency.

## H. HALight Communication Efficiency Analysis

In this subsection, we analyze the communication efficiency of HALight. The communication related computation at each timestep mainly consists of three components: Local Observation Encoding, Attentive Neighborhood Representation Aggregation, and Information Generation. At the system level, their per-timestep computational costs are $\mathcal { O } ( n d _ { o } d _ { h } + | E | d _ { \mathrm { i n f o } } d _ { h } )$ $O ( | E | d _ { h } d _ { a } )$ , and $O ( n d _ { a } d _ { \mathrm { i n f o } } )$ , respectively. Consequently, the overall communication related computation of HALight can be expressed as

$$
T = \mathcal { O } \big ( n ( d _ { o } d _ { h } + d _ { a } d _ { \mathrm { i n f o ~ } } ) + | E | ( d _ { \mathrm { i n f o ~ } } d _ { h } + d _ { h } d _ { a } ) \big ) ,\tag{23}
$$

where $d _ { o }$ denotes the observation dimension, n is the number of agents, and |E| is the number of edges in the road network.

As indicated by the above expression, T scales linearly with both n and |E|, suggesting that HALight can be extended to large-scale road networks. Moreover, HALight requires only a single round of one-hop neighborhood aggregation per decision, avoiding the latency and computation introduced by multi-round message passing. Empirically, we find that small dimensions $d _ { h } , d _ { a }$ , and $d _ { \mathrm { i n f o } }$ remain sufficient to achieve substantial performance gains even when they are much smaller than the hidden width of the policy network, indicating that lightweight communication can effectively enhance coordination under lane-level missing observations.

## VI. CONCLUSION

In this paper, we studied TSC in realistic settings where intersections are structurally heterogeneous and lanelevel observations can be partially missing. We proposed HALight, a communication-enhanced heterogeneous MARL framework. HALight supports intersection-specific observation/action modeling and integrates lightweight one-hop neighborhood communication with temporal modeling, while employing a sequential policy update scheme to stabilize heterogeneous agent learning. Extensive experiments on two realworld road networks, covering different network scales, traffic demands, and observation missing rates, demonstrate HA-Light’s effectiveness. Our communication efficiency analysis also shows that the proposed communication design scales favorably with network size.

Despite these promising results, several directions remain for future work. First, it is worthwhile to evaluate the proposed framework under more realistic traffic conditions, including richer phase plans and diverse sensing/communication failure patterns. Second, further validation on larger-scale road networks is needed, and exploring adaptive communication beyond one-hop neighborhoods may improve performance when traffic interactions extend over longer distances.

## APPENDIX A

## CONVERGENCE ANALYSIS OF HALIGHT UNDER FULLOBSERVABILITY

We analyze the convergence property of HALight under full observability. In this case, each agent conditions its policy on the Markovian state $s _ { t } ~ \in ~ S .$ , and the joint policy is factorized as $\begin{array} { r } { \pi _ { \theta } ( \mathbf { a } _ { t } | s _ { t } ) ~ = ~ \prod _ { i = 1 } ^ { n } \pi _ { \theta ^ { i } } ^ { i } ( a _ { t } ^ { i } | s _ { t } ) } \end{array}$ , where $\begin{array} { r l } { \mathbf { \delta } a _ { t } } & { { } = } \end{array}$ $( a _ { t } ^ { 1 } , a _ { t } ^ { 2 } , \ldots , a _ { t } ^ { n } ) \in \mathcal { A }$ . The communication encoder, attention aggregation, information decoder, and policy head are treated as differentiable components of the individual policies and do not alter this factorization. At iteration $k ,$ HALight updates the actors sequentially according to a random permutation $( i _ { 1 } , i _ { 2 } , \ldots , i _ { n } )$ , with each update weighted by the correction factor $M _ { i _ { 1 : m } } .$ . Therefore, under full observability, HALight follows the HAPPO-style optimization structure with factorized heterogeneous policies, sequential agent-wise updates, and correction for previously updated agents.

Proposition. Under the fully observable cooperative Markov game setting and the standard assumptions of HAPPO, HALight satisfies the monotonic improvement property, i.e.,

$$
J ( \pi _ { \theta _ { k + 1 } } ) \geq J ( \pi _ { \theta _ { k } } ) , \quad \forall k .
$$

Furthermore, if every agent permutation has a fixed non-zero probability of being sampled, the limit points of the generated joint-policy sequence are Nash equilibria.

Proof. Under full observability, the HALight actor of each agent defines a valid individual policy $\pi _ { \theta ^ { i } } ^ { i } ( a ^ { i } | s )$ . Since the communication and attention modules only affect the internal representation of $\pi _ { \theta ^ { i } } ^ { i }$ , the joint policy still factorizes as $\begin{array} { r } { \pi _ { \theta } ( \mathbf { a } | \bar { s } ) = \prod _ { i = 1 } ^ { n } \pi _ { \theta ^ { i } } ^ { i } ( a ^ { i } | s ) } \end{array}$

For any permutation $( i _ { 1 } , i _ { 2 } , \ldots , i _ { n } )$ , the multi-agent advantage decomposition lemma gives

$$
A _ { \pi } ( s , a ) = \sum _ { m = 1 } ^ { n } A _ { \pi } ^ { i _ { m } } \left( s , a ^ { i _ { 1 : m - 1 } } , a ^ { i _ { m } } \right) .
$$

Thus, the improvement of the joint policy can be evaluated through a sequence of agent-wise advantage terms. For agent $i _ { m }$ , HALight uses $M _ { i _ { 1 : m } }$ to account for the policies of agents $i _ { 1 } , \ldots , i _ { m - 1 }$ that have already been updated. Therefore, the update objective of $i _ { m }$ is equivalent to optimizing its contribution to the decomposed joint advantage under the updated previous agents.

Let $\pi _ { \theta _ { k } }$ and $\pi _ { \theta _ { k + } }$ denote the joint policies before and after one sequential update. Following the HAPPO policy improvement bound, the return difference is lower-bounded by the sum of the sequential improvement terms:

$$
J ( \pi _ { \theta _ { k + 1 } } ) - J ( \pi _ { \theta _ { k } } ) \geq \sum _ { m = 1 } ^ { n } \Delta _ { i _ { m } } ,
$$

where $\Delta _ { i _ { m } }$ denotes the improvement term obtained by updating agent $i _ { m }$ with the correction factor $M _ { i _ { 1 : m } }$ . Since agent $i _ { m }$ can always choose the old policy $\pi _ { \theta _ { r . } ^ { i _ { m } } } ^ { i _ { m } }$ , which yields zero improvement, maximizing the HAPPO-style objective gives $\Delta _ { i _ { m } } \geq 0$ for every m. Therefore,

$$
J ( \pi _ { \theta _ { k + 1 } } ) - J ( \pi _ { \theta _ { k } } ) \geq 0 ,
$$

which proves the monotonic improvement property.

Moreover, the random update order in HALight assigns nonzero probability to every permutation. This satisfies the updateorder condition in HAPPO convergence analysis. Hence, under the same full observation assumptions, the limit points of HALight’s joint policy sequence are Nash equilibria. □

## REFERENCES

[1] M. Keyvan-Ekbatani, M. Yildirimoglu, N. Geroliminis, and M. Papageorgiou, “Multiple concentric gating traffic control in large-scale urban networks,” IEEE Transactions on Intelligent Transportation Systems, vol. 16, no. 4, pp. 2141–2154, 2015.

[2] K. Wu, J. Ding, J. Lin, G. Zheng, Y. Sun, J. Fang, T. Xu, Y. Zhu, and B. Gu, “Big-data empowered traffic signal control could reduce urban carbon emission,” Nature Communications, vol. 16, no. 1, p. 2013, 2025.

[3] P. Koonce et al., “Traffic signal timing manual,” United States. Federal Highway Administration, Tech. Rep., 2008.

[4] Z. Yao, Y. Jiang, B. Zhao, X. Luo, and B. Peng, “A dynamic optimization method for adaptive signal control in a connected vehicle environment,” Journal of Intelligent Transportation Systems, vol. 24, no. 2, pp. 184– 200, 2020.

[5] J. D. Little, M. D. Kelson, and N. H. Gartner, “MAXBAND: A versatile program for setting signals on arteries and triangular networks,” Massachusetts Institute of Technology, Cambridge, MA, USA, Tech. Rep. 1185-81, 1981.

[6] C. Gershenson, “Self-organizing traffic lights,” arXiv preprint nlin/0411066, 2004.

[7] S. El-Tantawy, B. Abdulhai, and H. Abdelgawad, “Multiagent reinforcement learning for integrated network of adaptive traffic signal controllers (marlin-atsc): Methodology and large-scale application on downtown toronto,” IEEE transactions on Intelligent transportation systems, vol. 14, no. 3, pp. 1140–1150, 2013.

[8] H. Wei, G. Zheng, H. Yao, and Z. Li, “Intellilight: A reinforcement learning approach for intelligent traffic light control,” in Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 2018, pp. 2496–2505.

[9] W. Zhang, C. Yan, X. Li, L. Fang, Y.-J. Wu, and J. Li, “Distributed signal control of arterial corridors using multi-agent deep reinforcement learning,” IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 1, pp. 178–190, 2022.

[10] C. Chen, H. Wei, N. Xu, G. Zheng, M. Yang, Y. Xiong, K. Xu, and Z. Li, “Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 04, 2020, pp. 3414–3421.

[11] T. Wu, P. Zhou, K. Liu, Y. Yuan, X. Wang, H. Huang, and D. O. Wu, “Multi-agent deep reinforcement learning for urban traffic light control in vehicular networks,” IEEE Transactions on Vehicular Technology, vol. 69, no. 8, pp. 8243–8256, 2020.

[12] W.-Y. Lin, Y.-Z. Song, B.-K. Ruan, H.-H. Shuai, C.-Y. Shen, L.-C. Wang, and Y.-H. Li, “Temporal difference-aware graph convolutional reinforcement learning for multi-intersection traffic signal control,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 1, pp. 327–337, 2023.

[13] Y. Zhang, H. Goel, P. Li, M. Damani, S. Chinchali, and G. Sartoretti, “Coordlight: Learning decentralized coordination for network-wide traffic signal control,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[14] H. Wei, N. Xu, H. Zhang, G. Zheng, X. Zang, C. Chen, W. Zhang, Y. Zhu, K. Xu, and Z. Li, “Colight: Learning network-level cooperation for traffic signal control,” in Proceedings of the 28th ACM international conference on information and knowledge management, 2019, pp. 1913– 1922.

[15] H. Gu, S. Wang, X. Ma, D. Jia, G. Mao, E. G. Lim, and C. P. R. Wong, “Large-scale traffic signal control using constrained network partition and adaptive deep reinforcement learning,” IEEE transactions on intelligent transportation systems, vol. 25, no. 7, pp. 7619–7632, 2024.

[16] H. Wei, C. Chen, G. Zheng, K. Wu, V. Gayah, K. Xu, and Z. Li, “Presslight: Learning max pressure control to coordinate traffic signals in arterial network,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 1290–1298.

[17] H. Jiang, Z. Li, Z. Li, L. Bai, H. Mao, W. Ketter, and R. Zhao, “A general scenario-agnostic reinforcement learning for traffic signal control,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 9, pp. 11 330–11 344, 2024.

[18] Y. Liang, Z. Zhao, and L. Sun, “Memory-augmented dynamic graph convolution networks for traffic data imputation with diverse missing patterns,” Transportation Research Part C: Emerging Technologies, vol. 143, p. 103826, 2022.

[19] Y. Zhang, P. Sun, S.-w. Lu, and B. He, “Communication-zone field¨ analysis of electronic-toll-collection system by hybrid fdtd-po method,” in 2012 12th International Conference on ITS Telecommunications. IEEE, 2012, pp. 706–709.

[20] T. Wang, Z. Zhu, J. Zhang, J. Tian, and W. Zhang, “A large-scale traffic signal control algorithm based on multi-layer graph deep reinforcement learning,” Transportation Research Part C: Emerging Technologies, vol. 162, p. 104582, 2024.

[21] Q. Lu, H. Fang, Z. Yin, and G. Zhu, “Haps-ppo: A multi-agent reinforcement learning architecture for coordinated regional control of traffic signals in heterogeneous road networks,” Applied Sciences, vol. 15, no. 20, p. 10945, 2025.

[22] E. Van der Pol and F. A. Oliehoek, “Coordinated deep reinforcement learners for traffic light control,” Proceedings of learning, inference and control of multi-agent systems (at NIPS 2016), vol. 8, pp. 21–38, 2016.

[23] P. Varaiya, “Max pressure control of a network of signalized intersections,” Transportation Research Part C: Emerging Technologies, vol. 36, pp. 177–195, 2013.

[24] R. Lowe, Y. I. Wu, A. Tamar, J. Harb, O. Pieter Abbeel, and I. Mordatch, “Multi-agent actor-critic for mixed cooperative-competitive environments,” Advances in neural information processing systems, vol. 30, 2017.

[25] K. Son, D. Kim, W. J. Kang, D. E. Hostallero, and Y. Yi, “Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning,” in International conference on machine learning. PMLR, 2019, pp. 5887–5896.

[26] J. K. Gupta, M. Egorov, and M. Kochenderfer, “Cooperative multi-agent control using deep reinforcement learning,” in International conference on autonomous agents and multiagent systems. Springer, 2017, pp. 66–83.

[27] H. Huang, Z. Hu, Z. Lu, and X. Wen, “Network-scale traffic signal control via multiagent reinforcement learning with deep spatiotemporal attentive network,” IEEE transactions on cybernetics, vol. 53, no. 1, pp. 262–274, 2021.

[28] T. Nishi, K. Otaki, K. Hayakawa, and T. Yoshimura, “Traffic signal control based on reinforcement learning with graph convolutional neural nets,” in 2018 21st International conference on intelligent transportation systems (ITSC). IEEE, 2018, pp. 877–883.

[29] F.-X. Devailly, D. Larocque, and L. Charlin, “Ig-rl: Inductive graph reinforcement learning for massive-scale traffic signal control,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 7, pp. 7496–7507, 2021.

[30] C. Fan, F. Wang, B. Zhou, and D. Ma, “A centralized reinforcement learning-based method for traffic signal optimization using an adaptive sequential decision,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[31] H. Zhang, Z. Wang, J. Yu, R. Zhang, C. Jiang, G. Luo, W. Wu, and W. Wang, “Learning simultaneous and sequential decisions in multi-agent systems with application to traffic signal control,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[32] W. Liu, C. Zhang, K. Zhou, Y. Li, F. Zhan, W. Xue, and R. Chen, “Sequential decision marl for adaptive traffic signal control with different intersections priorities,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[33] J. A. Calvo and I. Dusparic, “Heterogeneous multi-agent deep reinforcement learning for traffic lights control.” in AICS, 2018, pp. 2–13.

[34] J. Shen, J. Hu, Q. Zhao, and W. Rao, “Heterogeneous traffic intersections control design based on reinforcement learning,” IET Intelligent Transport Systems, vol. 18, no. 10, pp. 1760–1776, 2024.

[35] Y. Zhang, P. Li, M. Fan, and G. Sartoretti, “Heterolight: A general and efficient learning approach for heterogeneous traffic signal control,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 1010–1017.

[36] X. Chen, Y. Zhang, Y. Xiao, M. Fan, M. Wang, and G. Sartoretti, “Cross: A mixture-of-experts reinforcement learning framework for generalizable large-scale traffic signal control,” arXiv preprint arXiv:2603.24930, 2026.

[37] Y. Zhang, Y. Liu, P. Gong, P. Li, M. Fan, and G. Sartoretti, “Unicorn: A universal and collaborative reinforcement learning approach toward generalizable network-wide traffic signal control,” IEEE Transactions on Intelligent Transportation Systems, 2026.

[38] J. Kuba, R. Chen, M. Wen, Y. Wen, F. Sun, J. Wang, and Y. Yang, “Trust region policy optimisation in multi-agent reinforcement learning,” in ICLR 2022-10th International Conference on Learning Representations. The International Conference on Learning Representations (ICLR), 2022, p. 1046.

[39] J. G. Kuba, M. Wen, L. Meng, H. Zhang, D. Mguni, J. Wang, Y. Yang et al., “Settling the variance of multi-agent policy gradients,” Advances in Neural Information Processing Systems, vol. 34, pp. 13 458–13 470, 2021.

[40] Y. Zhong, J. G. Kuba, X. Feng, S. Hu, J. Ji, and Y. Yang, “Heterogeneous-agent reinforcement learning,” Journal of Machine Learning Research, vol. 25, no. 32, pp. 1–67, 2024.

[41] L. Han, K. Zheng, L. Zhao, X. Wang, and H. Wen, “Content-aware traffic data completion in its based on generative adversarial nets,” IEEE Transactions on Vehicular Technology, vol. 69, no. 10, pp. 11 950– 11 962, 2020.

[42] Y. Zhang, X. Wei, X. Zhang, Y. Hu, and B. Yin, “Self-attention graph convolution residual network for traffic data completion,” IEEE Transactions on Big Data, vol. 9, no. 2, pp. 528–541, 2022.

[43] X. Chen, K. Wang, Z. Li, Y. Zhang, and Q. Ye, “A novel nonconvex low-rank tensor completion approach for traffic sensor data recovery from incomplete measurements,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–15, 2023.

[44] M. J. Hausknecht and P. Stone, “Deep recurrent q-learning for partially observable mdps.” in AAAI fall symposia, vol. 45, 2015, p. 141.

[45] X. Wang, A. Taitler, I. Smirnov, S. Sanner, and B. Abdulhai, “emarlin +: Addressing partial observability to promote traffic signal coordination by leveraging historical information,” IEEE Transactions on Intelligent Transportation Systems, 2024.

[46] L. Wu, M. Wang, D. Wu, and J. Wu, “Dynstgat: Dynamic spatialtemporal graph attention network for traffic signal control,” in Proceedings of the 30th ACM international conference on information & knowledge management, 2021, pp. 2150–2159.

[47] K. Peng, T. Ma, L. Jia, and H. Rong, “Enhancing collaboration in heterogeneous multiagent systems through communication complementary graph,” IEEE Transactions on Cybernetics, 2024.

[48] R. Dey and F. M. Salem, “Gate-variants of gated recurrent unit (gru) neural networks,” in 2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS). IEEE, 2017, pp. 1597–1600.

[49] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, “The surprising effectiveness of ppo in cooperative multi-agent games,” Advances in neural information processing systems, vol. 35, pp. 24 611– 24 624, 2022.

[50] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in The International Conference on Learning Representations (ICLR), 2015.

[51] H. Zhang, S. Feng, C. Liu, Y. Ding, Y. Zhu, Z. Zhou, W. Zhang, Y. Yu, H. Jin, and Z. Li, “Cityflow: A multi-agent reinforcement learning environment for large scale city traffic scenario,” in The world wide web conference, 2019, pp. 3620–3624.