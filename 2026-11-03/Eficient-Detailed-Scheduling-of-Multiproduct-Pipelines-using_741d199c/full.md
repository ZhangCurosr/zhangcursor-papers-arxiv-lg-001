# Eficient Detailed Scheduling of Multiproduct Pipelines using Action-Decoupled Deep Reinforcement Learning

Kai Shao<sup>a</sup>, Ling Jian<sup>a</sup>, Liming Liu<sup>a</sup>, Serguey V. Bogomolov<sup>b</sup> and Xinmin Wang<sup>a,∗</sup>

<sup>a</sup>School of Economics and Management, China University of Petroleum (East China), Qingdao, Shandong, China

<sup>b</sup>Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University, Leninskie Gory, Moscow, 119899, Russia

## A R T I C L E I N F O

Keywords:   
Multiproduct pipeline scheduling   
Deep reinforcement learning   
Action decoupling   
Nonlinear hydraulic constraints

## A BS T R AC T

Multiproduct pipelines transport refined products to delivery stations in sequential batches, and their detailed scheduling must meet market demand while respecting nonlinear hydraulic constraints. Retaining the full hydraulic nonlinearity yields a mixed-integer nonlinear program that is computationally intractable, while the linearization commonly adopted for solver-based optimization degrades hydraulic accuracy without resolving the heavy computational burden. This paper develops an action-decoupled proximal policy optimization (PPO) framework that eficiently produces detailed schedules while preserving hydraulic fidelity. The scheduling process is cast as a semi-Markov decision process (SMDP) with flexible action durations. To handle the discrete-continuous hybrid action space, a cascaded dual-actor architecture decouples the discrete operating duration from the continuous flow decisions, eliminating the gradient interference that destabilizes conventional coupled-action policies and thereby yielding highquality schedules far more eficiently. Validated on a real-world multiproduct pipeline in China and benchmarked against a single-actor baseline and a linearized MILP solved by Gurobi, the proposed framework achieves delivery accuracy close to the exact optimum while substantially improving flow rate stability and substantially reducing solution time.

## 1. Introduction

## 1.1. Background

The petroleum supply chain encompasses multiple critical stages, ranging from refining to the distribution of final products. Within this complex system, pipeline transportation serves as a fundamental pillar of modern energy logistics, providing a cost-efective and safe means of transporting large volumes of liquid fuel over long distances to downstream markets (Liao et al., 2022; Han et al., 2024). In contemporary operations, diferent grades of fuel—such as diesel and gasoline—are pumped sequentially into the pipeline without any physical barriers between them (Mostafaei et al.,. 2021a; Sidki et al., 2025). The continuous migration of these batches introduces highly dynamic hydraulic changes throughout the system (Zhou et al., 2019b). Therefore, under the strict premise of satisfying these complex hydraulic constraints, the core objective of pipeline scheduling is to safely and punctually meet downstream market demand while simultaneously minimizing operational costs (Zhang et al., 2017).

Over the past few decades, detailed scheduling optimization for multiproduct pipelines has been widely studied across diferent network structures and specific operating targets. The hydraulic environment is intrinsically complex, involving highly nonlinear relationships among batch interface positions, flow rates, and pressures (Lin et al., 2025). Consequently, the detailed scheduling of multiproduct pipelines is typically formulated as a mixed-integer nonlinear programming (MINLP) problem, where the primary computational challenge lies in dealing with severe model nonconvexity (Cafaro et al., 2015). To alleviate this, a widely adopted practice is to linearize the non-convex constraints to form a Mixed-Integer Linear Programming (MILP) formulation. However, While this improves tractability, it sacrifices hydraulic accuracy, and the resulting MILP models still necessitate extensive binary variables to encode spatial and temporal operational decisions. As key scale parameters increase, these variables grow exponentially, inevitably triggering a combinatorial explosion in the solution space (Liao et al., 2026). With the rapid development of data science and artificial intelligence, solving such complex scheduling optimization problems with significantly lower time costs has become possible (Tu et al., 2024). Therefore, this research aims to establish a rigorous mathematica modeling approach and investigate efective reinforcement learning algorithms for optimizing multiproduct pipeline scheduling.

## 1.2. Related work

To mathematically formulate this scheduling process, researchers must primarily decide how to represent time in their models. Discrete-time representations divide the entire scheduling horizon into uniform, fixed-length time intervals. Rejowski Jr and Pinto (2003) formulated a discrete-time MILP model as a representative approach for single pipeline scheduling problems, and Meira et al. (2021) proposed a new discrete-time MILP model to address the longterm scheduling of pumping and delivery operations in straight multiproduct pipeline systems connecting multiple sources to multiple destinations. While predefined discrete intervals simplify basic constraint formulation, they force a severe trade-of between coarse intervals that compromise hydraulic accuracy and fine intervals that trigger the curse of dimensionality (Liao et al., 2018a). Contrary to discrete models, continuous-time MILP representations have proven to be an eficient way to find the optimal transportation plan for single pipelines (Cafaro and Cerdá, 2004). By modeling operational event timings as continuous variables rather than fixed-duration boundaries, this representation allows the model to track batch movements while generally yielding formulations with smaller dimensions (Rejowski Jr and Pinto, 2008). Abdellaoui et al. (2021) developed a continuous-time MILP model to comprehensively optimize complex pipeline networks. Mostafaei et al. (2021b) proposed a holistic continuous-time approach to rigorously enforce strict operational constraints, such as forbidden product sequences and dynamic inventory. Hybrid-time representations merge discrete and continuous frameworks by establishing discrete time segments, which allows the model to evaluate macroscopic states over predefined intervals while continuously tracking microscopic hydraulic events. Utilizing this logic, Zhou et al. (2019b) proposed a hybrid-time MILP model that accurately captures dynamic variations in pipeline frictional pressure drops during multi-batch sequential transportation.

Even with these advanced formulations, solving the resulting models remains computationally daunting. To enhance solution eficiency, many researchers have linearized nonlinear equations associated with pressure, such as pipeline friction and pump characteristic equations (Liao et al., 2018a), or used flow constraints instead of pressure constraints to avoid the need for nonlinear terms (Liao et al., 2018b). These linearized models are subsequently solved using standard exact solvers such as CPLEX or Gurobi. However, this practice faces a twofold limitation. Mathematically, imposing nonlinear physical constraints as linear bounds inherently shrinks the feasible region and degrades hydraulic accuracy, failing to guarantee a true global optimum for the actual system (Lin et al., 2025). Computationally, even linearized models face severe practical limitations. The extensive binary variables required for rigorous batch tracking impose a massive computational burden, making it highly challenging for exact solvers to tackle practical-scale scheduling problems within an acceptable timeframe (Xu et al., 2021; MirHassani and Jahromi, 2011). To overcome these limitations and expedite the solution process, researchers have developed various decomposition frameworks. Two-stage decomposition algorithms have been widely used to systematically decouple the sequencing of batches and time events from the continuous calculation of detailed delivery volumes and flow rates (Yan et al., 2021; Zhang et al., 2017). Meira et al. (2021) and Liao et al. (2026) applied rolling-horizon frameworks and multi-stage decomposition to slice long-term tasks into manageable subproblems, which not only accommodate dynamic demand updates but also drastically accelerate the solution process. Concurrently, heuristic and meta-heuristic approaches have been extensively explored. Chen et al. (2017a) applied a simulated annealing algorithm to optimize the schedule of multiproduct pipelines, while Bamoumen et al. (2023) designed a greedy randomized adaptive search procedure (GRASP) to sequence batches and product volumes rapidly. Despite these improvements, heuristic rules are usually highly targeted to a specific scheduling issue, resulting in poor generality when applied to other scheduling systems with slightly diferent operational requirements (Liao et al., 2019). With the development of data science and machine learning, data-driven methods have been introduced to enrich the solution approaches (Liao et al., 2019; Zhang et al., 2018). These methods attempt to accelerate the initial search by mining historical schedules, assuming that analogous operational conditions will yield similar solutions. Consequently, they sufer from poor generalizability and fail to provide valid scheduling guidance in unprecedented scenarios lacking historical precedent.

In recent years, reinforcement learning (RL) has been recognized as a powerful approach for optimal sequential decision-making. Benefiting from deep neural networks, deep reinforcement learning (DRL) can handle decisionmaking problems with large state and action spaces, achieving tremendous success in complex domains such as robotics and games (Mnih et al., 2015; Kaufmann et al., 2023). More notably, DRL has proven highly successful in solving analogous network operation and scheduling tasks, such as real-time pump control in water distribution networks (Pei et al., 2025), network-constrained economic dispatch in large-scale power grids (Chen et al., 2025), continuous-action resource dispatch in isolated microgrids (Dominguez-Barbero et al., 2023), and compressor operation as well as supply reliability optimization in natural gas pipelines (Liu et al., 2026; Fan et al., 2022). These successes across diverse networked infrastructures demonstrate the broad potential of DRL for complex operation and scheduling problems. In multiproduct pipelines specifically, DRL has so far been limited to pump scheduling (Shao et al., 2025), while the detailed batch scheduling problem remains largely unexplored. However, directly applying standard DRL algorithms to this problem introduces a critical technical barrier regarding the discrete-continuous hybrid action space. Standard algorithms often struggle in complex engineering applications where an agent must simultaneously output discrete and continuous variables, a challenge recently highlighted in hybrid vehicle energy management (Liu et al., 2024). Multiproduct pipeline scheduling poses a comparable challenge because the agent must simultaneously make discrete operational decisions while adjusting continuous flow rates. Standard single-actor algorithms address this by treating both output types as a single vector. Inevitably, such rigid coupling neglects conditional dependencies among variables, leading the neural network to compute gradients for continuous parameters even when the corresponding discrete decisions render them operationally inactive. Consequently, forcing the agent to evaluate these invalid actions during exploration introduces severe noise into the training process, significantly hindering algorithmic convergence and reducing sample eficiency (Li et al., 2025). To address this structural limitation, recent studies demonstrate that spatially decoupling multidimensional actions efectively enhances policy flexibility and learning eficiency (Nie et al., 2025).

Inspired by these advancements, this study proposes a novel action-decoupled deep reinforcement Learning framework to optimize the detailed scheduling of multiproduct pipelines. To the best of our knowledge, this is the first work to apply deep reinforcement learning to the detailed batch scheduling of single-source, multiple-destination multiproduct pipelines. Experimental results show that the proposed framework yields high-quality detailed schedules for multiproduct pipelines with substantially reduced computation cost. The contributions of this paper are summarized as follows:

• We propose a hybrid-time MINLP model for the detailed scheduling of multiproduct pipelines that preserves the full hydraulic nonlinearity rather than relying on the linearization commonly adopted in prior work. By tracking exact batch arrival times, the model enforces rigorous batch tracking and accurate hydraulic loss calculation, and formulates scheduling as a multi-objective problem that jointly minimizes delivery deviation and flow rate fluctuation, thereby providing a high-fidelity formulation without sacrificing solution accuracy.

• We translate the detailed scheduling process into a semi-Markov decision process (SMDP) (Sutton and Barto, 1998). This framework natively accommodates flexible operational durations, enabling the agent to maintain specific scheduling schemes over variable time intervals. This SMDP reformulation thus enables a deep reinforcement learning approach to be applied directly to the detailed scheduling of multiproduct pipelines.

• To address the hybrid action space dilemma, we design a novel cascaded dual-actor architecture built on proximal policy optimization (PPO) (Schulman et al., 2017) that decouples operating duration from flow rates. This logical decoupling of discrete and continuous actions eliminates gradient interference from invalid actions, thereby enhancing sample eficiency and training stability.

The remainder of this paper is organized as follows. Section 2 describes the detailed scheduling problem of multiproduct pipelines and states the underlying assumptions. Section 3 formulates the hybrid-time MINLP model. Section 4 translates the scheduling process as an SMDP and presents the proposed action-decoupled SMDP-PPO framework. Section 5 reports the case study on a real-world pipeline, and Section 6 concludes the paper and discusses future work.

## 2. Problem description

A multiproduct pipeline system sequentially transports several refined products from a refinery to downstream delivery station. As illustrated in Fig. 1, the system consists of an initial injection station, a series of pipeline segments, intermediate pump stations, and delivery stations. The products are injected sequentially, without physical separators, and travel forward as distinct batches. Pump stations are strategically placed along the route to provide the necessary pressure to overcome frictional losses and elevation diferences, thereby maintaining continuous flow. As batches advance through the pipeline, the intermediate and terminal delivery stations withdraw the required volume from the corresponding batches to meet local market demand.

![](images/f5d0545f962cefc8f835c5830d42c5a1a7571a4ddde29a9fa0d452ccc9d96076.jpg)  
Figure 1: Schematic of sequential batch transportation in a multiproduct pipeline

Based on downstream market demands and injection plans, detailed pipeline scheduling determines the specific execution scheme for each operational period, determining the exact delivery volumes and delivery flow rates. In this study, the optimization model comprehensively addresses three key aspects: minimizing supply-demand deviations, maintaining flow rate stability, and accurately calculating dynamic hydraulic losses. The objective is to minimize delivery deviation while also pursuing flow rate stability. While precise delivery ensures supply reliability, maintaining stable flow rates efectively minimizes frictional pressure drop and thereby helps curb the associated pumping energy cost. It also improves scheduling practicability by reducing the operator workload from frequent valve adjustments and pump switching (Chen et al., 2017b). To ensure the feasibility of the schedule, the proposed model enforces exact batch tracking by incorporating the exact arrival times of batches at each station. This precise tracking enables accurate calculation of hydraulic loss, ensuring that all operational decisions comply with stringent hydraulic safety limits during the batch sequential transportation process (Zhou et al., 2019b).

To explicitly formulate the pipeline detailed scheduling problem, the given parameters and the decision variables to be determined are formally defined as follows.

Given:

• Pipeline data: Length, inner diameter, and flow rate limits of each pipeline segment.

• Station data: Elevation of each station, injection and delivery flow rate limits, max head of pump station.

• Production data: Physical properties of products (density, viscosity), initial batch states, predetermined batch injection volumes and sequence.

• Demand plan: Target delivery volumes for each product at respective delivery stations.

Determine:

• Injection schedule: Injection flow rate and execution durations.

• Delivery schedule: Delivery flow rate and execution durations.

To efectively solve this model, the following reasonable simplifications and assumptions are made:

• Refined products are treated as incompressible fluids with constant physical properties, completely disregarding any temperature- or pressure-induced variations.

• Within any time interval, the continuous variables, including station injection/delivery flow rate, remain strictly steady.

• Fluid flow in all segments is strictly assumed to be hydraulically smooth.

• Contamination between adjacent batches is modeled as a sharp zero-volume boundary, thereby neglecting the complex tracking of mixed-oil sections.

• Tank inventory constraints are excluded, assuming all stations have suficient capacity to store the scheduled products.

## 3. Mathematical model

In this section, we formulate the detailed scheduling problem for a multiproduct pipeline with a single injection source and several delivery stations as a hybrid-time MINLP model. The objective minimizes delivery deviations and flow rate fluctuations, subject to time node, batch tracking, delivery, flow rate, and pressure constraints. The Appendix A contains a complete nomenclature.

## 3.1. Objective function

The multiproduct pipeline detailed scheduling is developed as a multi-objective optimization problem that minimizes the delivery deviation $( f _ { 1 } )$ and the flow rate fluctuation $( f _ { 2 } )$ , as shown in Eq. (1).

$$
\operatorname* { m i n } { f } = f _ { 1 } + f _ { 2 }\tag{1}
$$

The delivery deviation $f _ { 1 }$ is defined as the absolute diference between the planned demand and the actual volumes delivered at each delivery station.

$$
f _ { 1 } = \sum _ { i } \sum _ { o } \left| V _ { i , o } ^ { x } - V _ { i , o } ^ { d } \right|\tag{2}
$$

The stability objective $f _ { 2 }$ suppresses operational fluctuations by penalizing the variation in flow rates between consecutive time intervals for each pipeline segment �.

$$
f _ { 2 } = \sum _ { k = 2 } ^ { k _ { \mathrm { m a x } } - 1 } \sum _ { i = 1 } ^ { i _ { \mathrm { m a x } } - 1 } \lambda _ { i } \cdot \left| Q _ { k , i } ^ { p } - Q _ { k - 1 , i } ^ { p } \right|\tag{3}
$$

A weight factor $\lambda _ { i }$ is applied to discourage fluctuations in flow rate in pipeline segments with larger volume capacities, as flow variations in such segments lead to higher frictional pressure, as expressed in Eq. (4).

$$
\lambda _ { i } = \frac { V _ { i + 1 } ^ { s } - V _ { i } ^ { s } } { V _ { i _ { \operatorname* { m a x } } } ^ { s } }\tag{4}
$$

## 3.2. Time node constraints

The scheduling horizon is represented by a sequence of time nodes $t _ { k }$ accumulated through Eq. (5), starting from $t _ { 1 } = 0$

$$
t _ { k + 1 } = t _ { k } + \Delta t _ { k } \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \}\tag{5}
$$

A binary variable $B _ { k + 1 }$ equals 1 if a batch head interface reaches a station at node $k + 1$ and 0 otherwise, as determined by Eqs. (6) and (7).

$$
B _ { k + 1 } \geq S _ { k + 1 , i , j } ^ { h e a d } - S _ { k , i , j } ^ { h e a d } \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \} , i \in I , j \in J\tag{6}
$$

$$
B _ { k + 1 } \leq \sum _ { i \in I } \sum _ { j \in J } \left( S _ { k + 1 , i , j } ^ { h e a d } - S _ { k , i , j } ^ { h e a d } \right) \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \}\tag{7}
$$

The step $\Delta t _ { k }$ is fixed to the base interval when no batch head interface reaches a station, and is released to a variable value within (0, 1] when a batch head arrives, as enforced by Eqs. (8) and (9), where � is a big-M constant and � is a small positive constant that prevents a zero-length step.

$$
1 - B _ { k + 1 } M \le \Delta t _ { k } \le 1 \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \}\tag{8}
$$

$$
\Delta t _ { k } \geq \varepsilon B _ { k + 1 } + ( 1 - B _ { k + 1 } ) \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \}\tag{9}
$$

## 3.3. Batch tracking constraints

The initial positions of all batches must be defined. For batches already inside the pipeline, the initial volumetric coordinate is determined by the known initial position $V _ { i } ^ { i n i t }$ , as expressed in Eq. (10). For new batches to be injected, the starting position is assigned a negative value calculated by Eq. (11), which represents the cumulative planned injection volumes $V _ { j ^ { \prime } } ^ { i n j }$ of all preceding new batches.

$$
V _ { 1 , j } ^ { b } = V _ { j } ^ { i n i t } \quad \forall j \in J ^ { o l d }\tag{10}
$$

$$
V _ { 1 , j } ^ { b } = - \sum _ { j ^ { \prime } = 1 } ^ { j - 1 } V _ { j ^ { \prime } } ^ { i n j } \quad \forall j \in J ^ { n e w }\tag{11}
$$

The binary indicator $S _ { k , i , j } ^ { h e a d }$ is introduced as a fundamental tracking variable. As defined by the big-M relaxation in Eqs. (12) and (13), $S _ { k , i , j } ^ { h e a d }$ equals 1 if the head interface of batch � has passed station � at time node �, and 0 otherwise. Eqs. (14) and (15) preserve the physical order by ensuring that each batch passes stations sequentially and that all batches maintain their relative positions.

$$
V _ { k , j } ^ { b } - V _ { i } ^ { s } \geq ( S _ { k , i , j } ^ { h e a d } - 1 ) M \quad \forall k \in K , i \in I , j \in J\tag{12}
$$

$$
V _ { k , j } ^ { b } - V _ { i } ^ { s } \le S _ { k , i , j } ^ { h e a d } M \quad \forall k \in K , i \in I , j \in J\tag{13}
$$

$$
S _ { k , i , j } ^ { h e a d } \geq S _ { k , i + 1 , j } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{14}
$$

$$
S _ { k , i , j } ^ { h e a d } \ge S _ { k , i , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I , j \in J \setminus \{ j _ { \operatorname* { m a x } } \}\tag{15}
$$

When a batch head interface reaches a station, i.e., $\left( S _ { k + 1 , i , j } ^ { \mathrm { h e a d } } - S _ { k , i , j } ^ { \mathrm { h e a d } } \right) = 1$ , the time node constraints in Eqs. (6)-(9) release the step length $\Delta t _ { k }$ to a variable value, while Eq. (16) coupled with Eq. (12) forces the coordinate $V _ { k + 1 , j } ^ { b }$ to coincide with the station coordinate $V _ { i } ^ { s }$ . Through the volume update in Eq. (19), this alignment determines the exact $\Delta t _ { k }$ and hence the precise arrival time $t _ { k + 1 }$ via Eq. (5).

$$
V _ { k + 1 , j } ^ { b } - V _ { i } ^ { s } \leq \left( 1 - S _ { k + 1 , i , j } ^ { h e a d } + S _ { k , i , j } ^ { h e a d } \right) M \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \} , i \in I , j \in J\tag{16}
$$

Product delivery from batch � at station � is permitted only when the batch covers the station, a condition captured by the binary variable $S _ { k , i , j } ^ { c o v e r }$ in Eqs. (17) and (18), which indicates that the head interface has passed the station while the tail interface remains upstream.

$$
S _ { k , i , j } ^ { c o v e r } = S _ { k , i , j } ^ { h e a d } - S _ { k , i , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I , j \in J \setminus \{ j _ { \operatorname* { m a x } } \}\tag{17}
$$

$$
S _ { k , i , j _ { \operatorname* { m a x } } } ^ { c o v e r } = S _ { k , i , j _ { \operatorname* { m a x } } } ^ { h e a d } \quad \forall k \in K , i \in I\tag{18}
$$

The volumetric coordinate $V _ { k , j } ^ { b }$ is updated in Eq. (19) by adding the volumetric displacement during the time interval, which is calculated as the diference between the source injection and all upstream deliveries.

$$
V _ { k + 1 , j } ^ { b } = V _ { k , j } ^ { b } + \left( Q _ { k } ^ { i n } - \sum _ { i \in I } S _ { k , i , j } ^ { h e a d } \cdot Q _ { k , i } ^ { d } \right) \Delta t _ { k } \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \} , j \in J\tag{19}
$$

As hydraulic loss varies with oil properties, the spatial relationship between batches and pipeline segments must be precisely determined. To facilitate this calculation, the occupancy of a batch within a segment is described by four mutually exclusive binary states. Since the last batch $j _ { m a x }$ has no succeeding batch and thus no tail interface within the horizon, the head indicator of the nonexistent succeeding batch is set to zero, i.e., $S _ { k , i , j _ { \mathrm { m a x } } + 1 } ^ { h e a d } = 0$ for all $k \in K , i \in I$ so that the following definitions apply uniformly to all batches.

As defined in Eqs. (20)–(22), the binary variable $Z _ { k , i , j } ^ { t h r u }$ equals 1 if batch � completely passes through segment �, which occurs when its head interface has passed station � + 1 while its tail interface has not yet reached station �.

$$
Z _ { k , i , j } ^ { t h r u } \leq S _ { k , i + 1 , j } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{20}
$$

$$
Z _ { k , i , j } ^ { t h r u } \le 1 - S _ { k , i , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{21}
$$

$$
Z _ { k , i , j } ^ { t h r u } \geq S _ { k , i + 1 , j } ^ { h e a d } - S _ { k , i , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{22}
$$

As defined in Eqs. (23)–(25), the binary variable $Z _ { k , i , j } ^ { H _ { - } i n }$ equals 1 if the head interface of batch � is inside segment � and its tail interface has not yet reached station �.

$$
Z _ { k , i , j } ^ { H _ { - } i n } \leq S _ { k , i , j } ^ { h e a d } - S _ { k , i + 1 , j } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{23}
$$

$$
Z _ { k , i , j } ^ { H _ { - } i n } \leq 1 - S _ { k , i , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{24}
$$

$$
Z _ { k , i , j } ^ { H _ { - } i n } \geq ( S _ { k , i , j } ^ { h e a d } - S _ { k , i + 1 , j } ^ { h e a d } ) - S _ { k , i , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{25}
$$

As defined in Eqs. (26)–(28), the binary variable $Z _ { k , i , j } ^ { T \_ i n }$ equals 1 if the tail interface of batch � is inside segment � and its head interface has passed station $i + 1$

$$
Z _ { k , i , j } ^ { T _ { - } i n } \leq S _ { k , i + 1 , j } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{26}
$$

$$
Z _ { k , i , j } ^ { T _ { - } i n } \leq S _ { k , i , j + 1 } ^ { h e a d } - S _ { k , i + 1 , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{27}
$$

$$
Z _ { k , i , j } ^ { T _ { - } i n } \geq S _ { k , i + 1 , j } ^ { h e a d } + ( S _ { k , i , j + 1 } ^ { h e a d } - S _ { k , i + 1 , j + 1 } ^ { h e a d } ) - 1 \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{28}
$$

As defined in Eqs. (29)–(31), the binary variable $Z _ { k , i , j } ^ { a l l \_ i n }$ equals 1 if both the head and tail interfaces of batch � are inside segment �.

$$
Z _ { k , i , j } ^ { a l l _ { - } i n } \leq S _ { k , i , j } ^ { h e a d } - S _ { k , i + 1 , j } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{29}
$$

$$
Z _ { k , i , j } ^ { a l l \_ i n } \leq S _ { k , i , j + 1 } ^ { h e a d } - S _ { k , i + 1 , j + 1 } ^ { h e a d } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{30}
$$

$$
Z _ { k , i , j } ^ { a l l } \underline { { i } } ^ { i n } \geq ( S _ { k , i , j } ^ { h e a d } - S _ { k , i + 1 , j } ^ { h e a d } ) + ( S _ { k , i , j + 1 } ^ { h e a d } - S _ { k , i + 1 , j + 1 } ^ { h e a d } ) - 1 \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} , j \in J\tag{31}
$$

## 3.4. Delivery constraints

Due to the technical limits of the measuring equipment and the limited receiving capacity of the station, the delivery flow rate $Q _ { k , i } ^ { d }$ is constrained by Eq. (32) to be zero or operate within its minimum threshold and maximum capacity.

$$
S _ { k , i } ^ { d } \cdot Q _ { i } ^ { d , \operatorname* { m i n } } \leq Q _ { k , i } ^ { d } \leq S _ { k , i } ^ { d } \cdot Q _ { i } ^ { d , \operatorname* { m a x } } \quad \forall k \in K , i \in I\tag{32}
$$

Summing the delivery volumes across all intervals where product � covers station � yields the total delivered volume, as expressed in Eq. (33).

$$
V _ { i , o } ^ { d } = \sum _ { k = 1 } ^ { k _ { \operatorname* { m a x } } - 1 } \sum _ { j \in J _ { o } } \left( S _ { k , i , j } ^ { c o v e r } \cdot Q _ { k , i } ^ { d } \right) \Delta t _ { k } \quad \forall i \in I , o \in O\tag{33}
$$

## 3.5. Flow rate constraints

The segment flow rate $Q _ { k , i } ^ { p }$ equals the adjacent upstream segment’s flow rate minus the delivery at the connecting station, as defined in Eq. (35). For the first segment, it equals the injection at the initial station in Eq. (34).

$$
Q _ { k , 1 } ^ { p } = Q _ { k } ^ { i n } \quad \forall k \in K\tag{34}
$$

$$
Q _ { k , i } ^ { p } = Q _ { k , i - 1 } ^ { p } - Q _ { k , i } ^ { d } \quad \forall k \in K , i \in I \setminus \{ 1 , i _ { \operatorname* { m a x } } \}\tag{35}
$$

Furthermore, to ensure transportation eficiency and equipment safety, the flow rate in each pipeline segment must be maintained within a specific allowable range, as specified by Eq. (36).

$$
Q _ { i } ^ { p , \operatorname* { m i n } } \leq Q _ { k , i } ^ { p } \leq Q _ { i } ^ { p , \operatorname* { m a x } } \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \}\tag{36}
$$

## 3.6. Pressure constraints

The nonlinear relationship between the flow rate and the pipeline frictional pressure is typically calculated using the empirical Leibenzon formula, which is widely used for friction loss calculations in multiproduct pipelines in China (Zhang et al., 2017; Lin et al., 2025). Assuming the oil flow regime stays within the hydrodynamic smooth zone $( \beta = 0 . 0 2 4 6 , m = 0 . 2 5 )$ ), the basic friction head loss $h _ { l }$ is defined by:

$$
h _ { l } = \beta \frac { Q ^ { 2 - m } \nu ^ { m } } { d ^ { 5 - m } } L .\tag{37}
$$

Fig. 2. illustrates the schematic for pressure drop calculation in a multiproduct pipeline. As oil products with diferent physical properties move sequentially, the frictional and elevation-related pressure drops change dynamically. At each time interval �, the total pressure required for segment � is determined by summing the efects of all batches within that segment; specifically, the frictional pressure drop $P _ { k , i } ^ { F }$ and elevation-related pressure drop $P _ { k , i } ^ { H }$ are defined by Eq. (38) and Eq. (39), respectively.

$$
P _ { k , i } ^ { F } = \frac { g \beta ( Q _ { k , i } ^ { p } ) ^ { 2 - m } } { d _ { i } ^ { 5 - m } } \left[ \rho _ { 1 } \nu _ { 1 } ^ { m } L _ { k , i , 1 } + \rho _ { 2 } \nu _ { 2 } ^ { m } L _ { k , i , 2 } + \cdots + \rho _ { j _ { \operatorname* { m a x } } } \nu _ { j _ { \operatorname* { m a x } } } ^ { m } L _ { k , i , j _ { \operatorname* { m a x } } } \right] \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \}\tag{38}
$$

$$
\begin{array} { r } { P _ { k , i } ^ { H } = g \left[ \rho _ { 1 } H _ { k , i , 1 } + \rho _ { 2 } H _ { k , i , 2 } + \cdots + \rho _ { j _ { \operatorname* { m a x } } } H _ { k , i , j _ { \operatorname* { m a x } } } \right] \quad \forall k \in K , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} } \end{array}\tag{39}
$$

Here, the exact physical length $L _ { k , i , j }$ and the elevation diference $H _ { k , i , j }$ spanned by batch � inside segment � are precisely determined by the four mutually exclusive binary topological variables $( Z ^ { t h r u } , Z ^ { H _ { - } i n } , Z ^ { T _ { - } i n } , Z ^ { a l l _ { - } i n } )$ formulated in the batch tracking section. By combining these with the volumetric coordinates, the detailed pressure drop formulas are derived:

$$
P _ { k , i } ^ { F } = \frac { 4 g \beta ( Q _ { k , i } ^ { P } ) ^ { 2 - m } \lambda _ { \mathrm { m a x } } } { \pi d _ { i } ^ { 7 - m } } \sum _ { j = 1 } ^ { m } \left\{ \rho _ { j } \nu _ { j } ^ { m } \left[ Z _ { k , i , j } ^ { t h n } ( V _ { i + 1 } ^ { s } - V _ { i } ^ { s } ) + Z _ { k , i , j } ^ { H } ( V _ { k , j } ^ { b } - V _ { i } ^ { s } ) + Z _ { k , i , j } ^ { T } ( V _ { i + 1 } ^ { s } - V _ { k , j + 1 } ^ { b } ) + Z _ { k , i , j } ^ { a l , i n } ( V _ { k , j } ^ { b } - V _ { k , j + 1 } ^ { b } ) \right] \right\} ,
$$

(40)

$$
P _ { k , i } ^ { H } = \frac { g ( E _ { i + 1 } - E _ { i } ) } { V _ { i + 1 } ^ { s } - V _ { i } ^ { s } } \sum _ { j = 1 } ^ { j _ { \operatorname* { m a x } } } \left\{ \rho _ { j } \left[ Z _ { k , i ; j } ^ { t h r u } ( V _ { i + 1 } ^ { s } - V _ { i } ^ { s } ) + Z _ { k , i ; j } ^ { H \mathcal { M } } ( V _ { k , j } ^ { b } - V _ { i } ^ { s } ) + Z _ { k , i ; j } ^ { T \mathcal { \bar { M } } } ( V _ { i + 1 } ^ { s } - V _ { k , j + 1 } ^ { b } ) + Z _ { k , i ; j } ^ { a l l , i n } ( V _ { k , j } ^ { b } - V _ { k , j + 1 } ^ { b } ) \right] \right\} .\tag{41}
$$

The sum $P _ { k , i } ^ { F } + P _ { k , i } ^ { H }$ gives the total pressure drop, i.e., the hydraulic loss, of segment � at node �. Under the hybrid

![](images/b113fba39029d1adc4177bcd8ff9a005e7f3d56118fc0193b4c57ee6ffed92fa.jpg)  
Figure 2: Schematic diagram of batch lengths and elevation diferences within pipeline segment �.

time framework, the transportation flow rate remains constant throughout any interval $( t _ { k } , t _ { k + 1 } )$ , during which no batch interfaces are permitted to enter or leave the pipeline segments. Consequently, the total pipeline pressure drop exhibits a strictly linear monotonic change over time during this interval (Zhou et al., 2019b). To ensure safety, the cumulative pressure demand up to node � must stay within the total upstream supply capacity. Due to pressure linearity, enforcing this constraint at both interval endpoints $( t _ { k }$ and $t _ { k + 1 } )$ guarantees compliance at any intermediate moment, as formulated in Eqs. (42) and (43):

$$
\sum _ { w = 1 } ^ { i } \left( P _ { k , w } ^ { F } + P _ { k , w } ^ { H } \right) \leq \sum _ { w = 1 } ^ { i } P _ { w } ^ { \operatorname* { m a x } } \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \} , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} ,\tag{42}
$$

$$
\sum _ { w = 1 } ^ { i } \left( P _ { k + 1 , w } ^ { F } + P _ { k + 1 , w } ^ { H } \right) \leq \sum _ { w = 1 } ^ { i } P _ { w } ^ { \operatorname* { m a x } } \quad \forall k \in K \setminus \{ k _ { \operatorname* { m a x } } \} , i \in I \setminus \{ i _ { \operatorname* { m a x } } \} .\tag{43}
$$

## 4. Action-decoupled deep reinforcement learning

In this section, we propose an action-decoupled deep reinforcement learning framework to optimize the detailed scheduling of a multiproduct pipeline. This framework is based on the semi-Markov decision process (SMDP) paradigm to enhance flow rate stability $( f _ { 2 } )$ through a sequence of "macro-actions". Each macro-action represents a stable operating state maintained until an asynchronous decision node is triggered, either by the expiration of a planned duration or by critical events. The following subsections present the formal SMDP formulation, the proposed action-decoupled architecture, and the PPO-based optimization procedure.

## 4.1. SMDP formulations for multiproduct pipeline scheduling

The scheduling problem is first mathematically structured as a semi-Markov decision process (SMDP) to recast the scheduling problem in a form amenable to reinforcement learning. The SMDP is formally defined by a tuple $\langle S , \mathcal { A } , \mathcal { R } , \mathcal { P } , \mathcal { T } , \gamma \rangle$ , where  is the state space,  is the action space,  is the reward function, $\mathcal { P } ( s ^ { \prime } , \tau \mid s , a )$ denotes the transition dynamics, and $\gamma \in ( 0 , 1 ]$ is the discount factor. The transition dynamics $\mathcal { P } ( s ^ { \prime } , \tau \mid s , a )$ represent the probability of transitioning to state $s ^ { \prime }$ after an actual execution duration $\tau \in \tau$ . The objective is to learn an optimal policy $\pi ^ { * }$ that maximizes the expected discounted rewards $\begin{array} { r } { G = \mathbb { E } \left[ \sum _ { n = 0 } ^ { L - 1 } \gamma ^ { t _ { n } } R _ { t _ { n } } \right] } \end{array}$ where � is the number of decision stages and $\begin{array} { r } { t _ { n } = \sum _ { l = 0 } ^ { n - 1 } \tau _ { l } } \end{array}$ is the cumulative number of steps at stage � (with $t _ { 0 } = 0 )$ . Here, $\tau _ { n } \in \mathcal { T }$ represents the actual execution duration of the �-th macro-action. To maintain temporal consistency, the stage reward $R _ { t _ { n } }$ is defined as the discounted accumulation of instantaneous rewards �(⋅) over the actual duration $\begin{array} { r } { \tau _ { n } \colon R _ { t _ { n } } = \sum _ { m = 0 } ^ { \tau _ { n } - 1 } \gamma ^ { m } R ( s _ { t _ { n } + m } , a _ { t _ { n } } ) } \end{array}$ . Here, � denotes the discrete step index within the �-th macro-action, and $s _ { t _ { n } + m }$ represents the environmental state at step $t _ { n } + m$

The components of the SMDP are specified as follows:

1) State space (): The state at decision stage � is represented by a feature vector $s _ { t _ { n } } = \{ V _ { t _ { n } } ^ { b } , \Delta V _ { t _ { n } } , \rho _ { t _ { n } } , v _ { t _ { n } } , a _ { t _ { n - 1 } } \}$ This vector encompasses the volumetric coordinates of all batch head interfaces $V _ { t _ { n } } ^ { b }$ , the densities � and kinematic $\rho _ { t _ { n } }$ viscosities $v _ { t _ { n } }$ of the products currently passing through each station, the remaining delivery demand $\Delta V _ { t _ { n } }$ for each station and product type, and the macro-action $a _ { t \_ }$ executed in the previous stage.

2) Action space (): Each macro-action $a _ { t _ { n } } = \{ d _ { n } , q _ { t _ { n } } \}$ in the hybrid action space  represents a stable operating state. Here, $d _ { n } \in \tau$ is the discrete planned duration, while the continuous vector � specifies both the injection flow $q _ { t _ { n } }$ rate $Q _ { t _ { n } } ^ { i n }$ at the initial station and the delivery flow rates $Q _ { t _ { n } , i } ^ { d }$ at all downstream delivery stations.

3) Reward function (): The reward function operationalizes the multi-objective optimization as RL feedback, providing the agent with progressive guidance by penalizing delivery deviations, operational instability, and safety violations. The total stage reward $R _ { t _ { n } }$ consists of an accumulated deviation component and several macro-level penalties:

$$
R _ { t _ { n } } = \alpha R _ { d e } + \beta \cdot R _ { s t a } + \eta \cdot R _ { p } ,\tag{44}
$$

where $\alpha , \beta$ and � are weight factors used to balance the operational objectives. Each term is defined as follows:

• Delivery deviation reward $( R _ { d e } ) \colon$ This component reflects the delivery deviation objective $f _ { 1 }$ defined in Eq. (2). It consists of an interval-based penalty $R _ { i n t }$ and a terminal penalty $R _ { t e r m }$ triggered at the end of the horizon:

$$
R _ { d e } = \sum _ { m = 0 } ^ { \tau _ { n } - 1 } \gamma ^ { m } R _ { i n t } ( s _ { t _ { n } + m } ) + \mathbb { I } ( d o n e ) \cdot R _ { t e r m } .\tag{45}
$$

The interval-based penalty $R _ { i n t }$ penalizes delivery errors at each discrete step $t _ { n } + m$ within the macro-action through a piecewise function specifically designed to enhance gradient sensitivity to minor errors:

$$
R _ { i n t } ( s _ { t _ { n } + m } ) = \left\{ \begin{array} { l l } { - \mathcal { E } _ { t _ { n } + m } , } & { \mathrm { i f } \mathcal { E } _ { t _ { n } + m } > \delta } \\ { - \ln ( \mathcal { E } _ { t _ { n } + m } + \varepsilon ) , } & { \mathrm { i f } \mathcal { E } _ { t _ { n } + m } \leq \delta } \end{array} \right. ,\tag{46}
$$

where $\mathcal { E } _ { t _ { n } + m }$ represents the normalized root mean square error (NRMSE) of the deviations between actual and planned delivery volumes in all stations and products at step $t _ { n } + m .$ , and � denotes an empirical threshold that balances coarse-scale and fine-scale optimization. The terminal penalty $R _ { t e r m }$ imposes a strict global constraint at the final step:

$$
R _ { t e r m } = - \Gamma \cdot \frac { \sum _ { i } \sum _ { o } \left| V _ { i , o } ^ { x } - V _ { i , o } ^ { d } \right| } { \sum _ { i } \sum _ { o } V _ { i , o } ^ { x } } ,\tag{47}
$$

where Γ is a penalty coeficient and �(����) is an indicator function that equals 1 only upon the end of the scheduling horizon.

• Flow rate stability reward $( R _ { s t a } )$ : This component encourages steady operations, corresponding to the stability objective $f _ { 2 } \operatorname { i n } \operatorname { E q . } \left( 3 \right)$ . It is composed of an operational smoothness reward $( R _ { s w } )$ and a switching penalty $( R _ { d u r } )$ i.e., $R _ { s t a } = R _ { s w } + R _ { d u r } ,$ where:

$$
R _ { s w } = - \sum _ { i = 1 } ^ { i _ { \operatorname* { m a x } } - 1 } \lambda _ { i } \left| \frac { Q _ { t _ { n } , i } ^ { p } - Q _ { t _ { n - 1 } , i } ^ { p } } { Q _ { \operatorname* { m a x } } } \right| ,\tag{48}
$$

$$
R _ { d u r } = - C _ { s w i t c h } .\tag{49}
$$

Here $R _ { s w }$ penalizes flow variations across pipeline segments between decision stages using volumetric weights $\lambda _ { i } ,$ , with $Q _ { \mathrm { m a x } }$ serving as a normalization constant representing the maximum allowable flow rate of the pipeline. $R _ { d u r }$ reduces intervention frequency by imposing a switching cost to encourage longer, more stable macroactions.

• Pressure penalty $( R _ { p } ) \colon$ : To ensure pressure safety according to Eqs. (42) and (43), a penalty $R _ { p } = - \Delta P _ { v i o l a t i o n }$ is applied. Here, $\Delta \bar { P _ { v i o l a t i o n } }$ represents the magnitude of the pressure violation when the hydraulic constraints are breached. Notably, such a violation is treated as a critical event, which immediately truncates the current macro-action and forces a transition to a new decision node.

4) Transition dynamics $( \mathcal { P } , \mathcal { T } )$ : The transition probability $\mathcal { P } ( s _ { t _ { n + 1 } } , \tau _ { n } \mid s _ { t _ { n } } , a _ { t _ { n } } )$ defines the environment’s evolution over an actual duration $\tau _ { n } .$ Although the agent proposes a planned duration $d _ { n } ,$ the actual duration $\tau _ { n }$ is determined by the simulator and may be truncated by critical events, such as a batch interface arriving at a delivery station or a pressure violation.

## 4.2. Action decoupling via cascaded dual-actor architecture

Traditional policy-based DRL methods typically use a single-actor to learn a joint policy $\pi _ { \theta } ( d , q \mid s )$ , where the planned scheduling duration � and flow rates � are generated simultaneously. In pipeline scheduling, however, these two sub-actions are strongly coupled: the appropriate flow rate � depends on the duration � over which it is held, so the same � may be either suitable or harmful depending on �. Following the policy gradient theorem (Sutton and Barto, 1998), the joint objective gradient in a single-actor framework is

$$
\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { \pi _ { \theta } } [ \nabla _ { \theta } \log \pi _ { \theta } ( d , q \mid s ) \cdot A ( s , d , q ) ] ,\tag{50}
$$

where $A ( s , d , q )$ is the advantage function evaluating the joint action. This leads to gradient interference: when a reasonable duration is paired with an incompatible flow rate during exploration, the low advantage of this combination drives the shared gradient to suppress the probability of the duration as well, even though the duration itself is sound. Such interference increases training instability and hampers the agent’s ability to identify compatible macro-actions eficiently.

To address this, we propose an action-decoupled architecture that employs a conditional factorization of the joint policy into two serial actors:

$$
\pi _ { \boldsymbol { \theta } } ( a \mid s ) = \pi _ { \boldsymbol { \theta } _ { 1 } } ( d \mid s ) \cdot \pi _ { \boldsymbol { \theta } _ { 2 } } ( q \mid s , d ) .\tag{51}
$$

In this serial structure, Actor 1 first determines the planned duration �, and Actor 2 then selects the corresponding flow rates � based on this duration. The decoupled optimization logic for these two actors is derived as follows:

• Duration network (Actor 1): To derive the update rule for Actor 1, we begin with the gradient of the joint objective with respect to the duration parameters $\theta _ { 1 }$ . Based on the policy factorization of Eq. (51), the gradient is initially expressed as:

$$
\begin{array} { r l } & { \nabla _ { \theta _ { 1 } } J = \mathbb { E } _ { s \sim \rho , a \sim \pi _ { \theta } } \left[ \nabla _ { \theta _ { 1 } } \log \pi _ { \theta } ( a \mid s ) \cdot A ( s , d , q ) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { s \sim \rho , ( d , q ) \sim \pi _ { \theta } } \left[ \nabla _ { \theta _ { 1 } } \log \pi _ { \theta _ { 1 } } ( d \mid s ) \cdot A ( s , d , q ) \right] . } \end{array}\tag{52}
$$

Here, the second factor log $\pi _ { \theta _ { 7 } } ( q \mid s , d )$ vanishes under $\nabla _ { \theta _ { 1 } }$ since it does not depend on $\theta _ { 1 }$ , leaving only the duration term. The joint expectation is decomposed into a nested form using the law of total expectation:

$$
\begin{array} { r } { \nabla _ { \theta _ { 1 } } J = \mathbb { E } _ { s \sim \rho , d \sim \pi _ { \theta _ { 1 } } } \left[ \mathbb { E } _ { q \sim \pi _ { \theta _ { 2 } } } \left[ \nabla _ { \theta _ { 1 } } \log \pi _ { \theta _ { 1 } } ( d \mid s ) \cdot A ( s , d , q ) \mid s , d \right] \right] } \\ { = \mathbb { E } _ { s \sim \rho , d \sim \pi _ { \theta _ { 1 } } } \left[ \nabla _ { \theta _ { 1 } } \log \pi _ { \theta _ { 1 } } ( d \mid s ) \cdot \mathbb { E } _ { q \sim \pi _ { \theta _ { 2 } } } [ A ( s , d , q ) \mid s , d ] \right] . } \end{array}\tag{53}
$$

Eq. (53) shows that the duration policy $\pi _ { \theta _ { 1 } }$ is updated based on the marginal advantage evaluated over the flow rate distribution. By evaluating the duration � based on the expected performance of subsequent flow rates, Actor 1 can determine whether a specific duration is suitable for the current multiproduct pipeline state. This approach prevents the duration policy from being misled by a single mismatched flow rate sample during exploration, ensuring more robust and stable credit assignment during the early training phase.

• Flow rate network (Actor 2): Conditioned on the duration � from the first stage, the flow rate policy � is $\pi _ { \theta _ { 2 } }$ optimized to find the most compatible flow rate control for that specific time window:

$$
\nabla _ { \theta _ { 2 } } J = \mathbb { E } _ { s , d \sim \pi _ { \theta _ { 1 } } , q \sim \pi _ { \theta _ { 2 } } } [ \nabla _ { \theta _ { 2 } } \log \pi _ { \theta _ { 2 } } ( q \mid s , d ) \cdot A ( s , d , q ) ] .\tag{54}
$$

For Actor 2, receiving the duration � as a prior condition significantly simplifies the exploration complexity, as the agent only needs to find the optimal flow rates within a given duration. Furthermore, the generated flow rates must also strictly adhere to safety boundaries. Therefore, to satisfy the delivery and pipeline segment flow rate constraints defined in Eqs. (32) and (36), a sequential action mapping mechanism is embedded in the output layer of Actor 2. Rather than directly outputting the delivery flow rate, the network generates a raw unconstrained vector $\boldsymbol { u } _ { t _ { n } } = [ u _ { t _ { n } } ^ { i n } , u _ { t _ { n } , 1 } ^ { d } , \ldots , u _ { t _ { n } , I } ^ { d } ] .$ , which is transformed into physical actions in two steps. First, the injection flow rate $Q _ { t _ { n } } ^ { i n }$ is mapped to strictly satisfy the capacity of the initial pipeline segment:

$$
Q _ { t _ { n } } ^ { i n } = Q _ { 1 } ^ { p , \operatorname* { m i n } } + \sigma ( u _ { t _ { n } } ^ { i n } ) \cdot ( Q _ { 1 } ^ { p , \operatorname* { m a x } } - Q _ { 1 } ^ { p , \operatorname* { m i n } } ) ,\tag{55}
$$

where $\sigma ( \cdot )$ is the sigmoid activation function. Then, the delivery flow rate $Q _ { t _ { n } , i } ^ { d }$ for each delivery station � is mapped sequentially to ensure that neither the delivery capacity nor the minimum flow rate limits are violated:

$$
\tilde { Q } _ { t _ { n } , i } ^ { d } = \operatorname* { m i n } \left( Q _ { i } ^ { d , \operatorname* { m a x } } , Q _ { t _ { n } , i - 1 } ^ { p } - Q _ { i } ^ { p , \operatorname* { m i n } } \right) ,\tag{56}
$$

$$
Q _ { t _ { n } , i } ^ { d } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } \tilde { Q } _ { t _ { n } , i } ^ { d } < Q _ { i } ^ { d , \operatorname* { m i n } } } \\ { Q _ { i } ^ { d , \operatorname* { m i n } } + \sigma ( u _ { t _ { n } , i } ^ { d } ) \cdot \left( \tilde { Q } _ { t _ { n } , i } ^ { d } - Q _ { i } ^ { d , \operatorname* { m i n } } \right) , } & { \mathrm { o t h e r w i s e ~ } } \end{array} \right. .\tag{57}
$$

This two-step mapping restricts the action space to a strictly feasible region, ensuring that all generated flow rates are legally feasible.

This hierarchical update mechanism ensures that the environment’s feedback is correctly attributed to duration selection and flow rate control, facilitating faster convergence in complex pipeline environments.

## 4.3. Action-Decoupled SMDP-PPO

In this subsection, the PPO algorithm is employed to obtain the optimized detailed scheduling policy within the SMDP framework. The training process adopts an actor-critic architecture, where the interaction between the dualactor agent and the multiproduct pipeline simulation environment is illustrated in Fig. 3. The simulation environmen inherently satisfies most mathematical constraints, such as batch tracking. For constraints not directly embedded in the simulator, the delivery flow rate bounds are strictly satisfied via an action mapping mechanism that ensures all outputs are inherently feasible, while hydraulic pressure constraints are enforced via penalty terms in the reward function.

![](images/7b431b681a3bbe4de85b39270e65727c6baf6fdb21b1044976b57019e1d8b248.jpg)  
Figure 3: Interaction between the cascaded dual-actor agent and the multiproduct pipeline simulation environment under the SMDP-PPO framework.

To consider the impact of execution duration in the SMDP-based scheduling model, the advantage function is estimated using extended generalized advantage estimation (EGAE) (Chen et al., 2023). This approach accounts for the actual execution duration $\tau _ { n }$ of each macro-action, which has been shown to balance the bias-variance trade-of in SMDP environments efectively. The temporal diference (TD) error $\delta _ { t _ { n } }$ for a scheduling transition starting at decision stage � is expressed as:

$$
\delta _ { t _ { n } } = R _ { t _ { n } } + \gamma ^ { \tau _ { n } } V _ { \phi } ( s _ { t _ { n } + \tau _ { n } } ) - V _ { \phi } ( s _ { t _ { n } } ) .\tag{58}
$$

Based on these TD errors, the EGAE for the �-th decision stage is computed as:

$$
\hat { A } _ { t _ { n } } ^ { E G A E } = \sum _ { j = 0 } ^ { \infty } \gamma ^ { \sum _ { l = 0 } ^ { j - 1 } \tau _ { n + l } } \lambda ^ { j } \delta _ { t _ { n + j } } ,\tag{59}
$$

where $\lambda \in [ 0 , 1 ]$ is a hyperparameter balancing the estimation bias and variance. The importance sampling ratio $r _ { t _ { n } } ( \theta _ { 1 } , \theta _ { 2 } )$ for the dual-actor network is defined as:

$$
r _ { t _ { n } } ( \theta _ { 1 } , \theta _ { 2 } ) = \frac { \pi _ { \theta _ { 1 } } ( d _ { n } \mid s _ { t _ { n } } ) \cdot \pi _ { \theta _ { 2 } } ( q _ { t _ { n } } \mid s _ { t _ { n } } , d _ { n } ) } { \pi _ { \theta _ { 1 } ^ { o l d } } ( d _ { n } \mid s _ { t _ { n } } ) \cdot \pi _ { \theta _ { 2 } ^ { o l d } } ( q _ { t _ { n } } \mid s _ { t _ { n } } , d _ { n } ) } .\tag{60}
$$

The parameters $\theta _ { 1 }$ and $\theta _ { 2 }$ are then optimized by maximizing the objective function $L ^ { C L I P } ;$

$$
\begin{array} { r } { L ^ { C L I P } ( \theta _ { 1 } , \theta _ { 2 } ) = \hat { \mathbb { E } } _ { n } \left[ \operatorname* { m i n } \left( r _ { t _ { n } } ( \theta _ { 1 } , \theta _ { 2 } ) \hat { A } _ { t _ { n } } ^ { E G A E } , \exp ( r _ { t _ { n } } ( \theta _ { 1 } , \theta _ { 2 } ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t _ { n } } ^ { E G A E } \right) \right] , } \end{array}\tag{61}
$$

where $\epsilon$ is the clipping coeficient. The critic network parameters $\phi$ are optimized by minimizing the value function loss:

$$
L ^ { V F } ( \phi ) = \hat { \mathbb { E } } _ { n } \left[ ( V _ { \phi } ( s _ { t _ { n } } ) - \hat { G } _ { t _ { n } } ) ^ { 2 } \right] ,\tag{62}
$$

where $\hat { G } _ { t _ { n } }$ is the target return for stage �.

The detailed optimization procedure is given in Algorithm 1. The agent iteratively samples a planned duration $d _ { n }$ and flow rates $q _ { t _ { n } }$ through the cascaded Actor 1 and Actor 2. During the step-by-step environment execution, an event-triggered termination mechanism is employed: the simulation is interrupted if a critical event occurs (e.g., batch head arrival or pressure violation), and the actual duration $\tau _ { n }$ is recorded. These transitions are stored in  for advantage estimation via EGAE followed by multi-epoch policy updates, enabling the agent to learn an optimal detailed scheduling policy under various operational constraints.

Algorithm 1 Action-Decoupled SMDP-PPO   
Initialize: Pipeline Environment $\varepsilon ,$ Actor 1 $\pi _ { \theta _ { 1 } } ( d | s )$ , Actor $2 \pi _ { \theta _ { 2 } } ( q | s , d )$ , Critic $V _ { \phi } ( s )$ , Experience Bufer    
Ensure: Optimized scheduling policy parameters $\theta _ { 1 } ^ { * } , \theta _ { 2 } ^ { * }$   
1: for iteration $= 1 , 2 , \ldots$ do   
2: while || < BATCH\_SIZE do   
3: if done then   
4: Reset environment $\varepsilon ,$ get initial state � , set $s _ { t _ { 0 } } ,$ $n  0 , t _ { n }  0$   
5: end if   
6: Actor 1: Sample planned duration $d _ { n } \sim \pi _ { \theta _ { 1 } } ( \cdot \mid s _ { t _ { n } } )$   
7: Actor 2: Sample flow rates $q _ { t _ { n } } \sim \pi _ { \theta _ { 2 } } ( \cdot \mid s _ { t _ { n } } \cdot \mid d _ { n } )$   
8: $R _ { t _ { n } } \gets 0 , \tau _ { n } \gets 0 , R _ { i n t } ^ { a c c u m } \gets \tilde { 0 }$   
9: <sup>�</sup> for $m = 0$ to $d _ { n } - 1$ do ⊳ Step-by-step simulation   
10: Step environment $\varepsilon$ with $q _ { t _ { n } }$ , receive interval-based deviation reward $R _ { i n t } ( \cdot )$   
11: $R _ { i n t } ^ { a c c u m } \gets R _ { i n t } ^ { a c c u m } + \gamma ^ { m } R _ { i n t } \ddot { ( \cdot ) }$   
12: $\tau _ { n } \gets m + 1$   
13: Update environment termination flag done   
14: if done or critical event triggered then $\triangleright \mathrm { e . g . }$ , batch head arrival or pressure violation   
15: break   
16: end if   
17: end for   
18: Compute total stage reward: $R _ { t _ { n } } \gets \alpha R _ { d e } + \beta R _ { s t a } + \eta R _ { p }$   
19: Store $( s _ { t _ { n } } , d _ { n } , q _ { t _ { n } } , \tau _ { n } , R _ { t _ { n } } , s _ { t _ { n } + \tau _ { n } } ) ^ { \cdot }$ in    
20: $t _ { n + 1 } \gets i _ { n } + \tau _ { n } , \ : n \gets n + 1$   
21: end while   
22: Compute advantages $\hat { A } _ { t _ { n } } ^ { E G A E }$ and returns $\hat { G } _ { t _ { n } }$ for all $n \in \mathcal { D }$   
23: for epoch = 1 to $E$ do   
24: for mini-batch $B \subset D$ do   
25: Update $\theta _ { 1 } , \theta _ { 2 }$ by maximizing $L ^ { C L I P }$   
26: Update $\phi$ by minimizing $L ^ { \cup _ { F } }$   
27: end for   
28: end for   
29: Clear Bufer    
30: end for

## 5. Results and discussion

In this section, we conduct a case study to validate the efectiveness and eficiency of the proposed Action-Decoupled SMDP-PPO framework on a multiproduct pipeline. Furthermore, we present a dedicated comparison demonstrating the training benefits of the cascaded dual-actor architecture over a single-actor baseline. The Python 3.9-based RL models, the C# pipeline simulator, and the Gurobi-based MILP baseline were all executed on an Intel Core i5-12450H CPU @ 2.00 GHz.

## 5.1. Basic pipeline data

A 375.5 km real-world multiproduct pipeline in China is considered for the case study, equipped with five stations: the initial station (IS), one delivery-pumping station (DPS1), two delivery stations (DS2 and DS3), and the terminal station (TS). The physical and operational parameters of these stations and the four connecting segments are listed in Tables 1 and 2, respectively. This pipeline sequentially transports 92# gasoline, 95# gasoline, and 0# diesel, whose physical properties are detailed in Table 3. During routine operations, the pipeline is typically filled with 0# diesel at both the beginning and the end of the scheduling horizon, and the batch injections generally follow a cyclic sequence of 92#G-95#G-92#G-0#D. The injection plan and the corresponding supply-demand requirements at each delivery station, adopted from the case study of Zhou et al. (2019a) and converted into volumetric units, are summarized in Table 4.

Table 1  
Physical and operational parameters of each station.
<table><tr><td>Station ID</td><td>Altitude (m)</td><td>Min delivery flow rate  $( \mathsf { m } ^ { 3 } / \mathsf { h } )$ </td><td>Max delivery flow rate  $( \mathsf { m } ^ { 3 } / \mathsf { h } )$ </td><td>Max head of pump station (m)</td></tr><tr><td>IS</td><td>0</td><td>一</td><td></td><td>1092</td></tr><tr><td>DPS1</td><td>7</td><td>100</td><td>900</td><td>1096</td></tr><tr><td>DS2</td><td>111</td><td>100</td><td>600</td><td>一</td></tr><tr><td>DS3</td><td>53</td><td>100</td><td>600</td><td>一</td></tr><tr><td>TS</td><td>51</td><td>100</td><td>600</td><td></td></tr></table>

Table 2  
Physical and operational parameters of pipeline segments.
<table><tr><td>Pipeline segment</td><td>Length (km)</td><td>Inner diameter (mm)</td><td>Min allowable flow rate  $( \mathsf { m } ^ { 3 } / \mathsf { h } )$ </td><td>Max allowable flow rate  $( \mathsf { m } ^ { 3 } / \mathsf { h } )$ </td></tr><tr><td>IS-DPS1</td><td>130.4</td><td>492.2</td><td>600</td><td>1100</td></tr><tr><td>DPS1-DS2</td><td>139.5</td><td>492.2</td><td>500</td><td>1100</td></tr><tr><td>DS2-DS3</td><td>66.9</td><td>441.2</td><td>100</td><td>1100</td></tr><tr><td>DS3-TS</td><td>38.7</td><td>441.2</td><td>100</td><td>1100</td></tr></table>

Table 3  
Physical properties of refined products.

<table><tr><td>Product Type</td><td>Density  $\left( \mathrm { k g } / \mathrm { m } ^ { 3 } \right)$ </td><td>Viscosity  $( \mathrm { m } ^ { 2 } / \mathrm { s } )$ </td></tr><tr><td>0# diesel</td><td>845</td><td> $6 . 1 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>92# gasoline</td><td>740</td><td> $2 . 0 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>95# gasoline</td><td>750</td><td> $1 . 0 7 \times 1 0 ^ { - 6 }$ </td></tr></table>

Table 4  
Planned supply and demand for the case study.
<table><tr><td>Station</td><td>0# Diesel  $\left( \mathsf { m } ^ { 3 } \right)$ </td><td>92# Gasoline  $\left( \mathsf { m } ^ { 3 } \right)$ </td><td>95# Gasoline  $\left( \mathsf { m } ^ { 3 } \right)$ </td><td>Total  $\left( \mathsf { m } ^ { 3 } \right)$ </td></tr><tr><td>DPS1</td><td>22,485</td><td>35,810</td><td>14,666</td><td>72,961</td></tr><tr><td>DS2</td><td>8,876</td><td>20,270</td><td>8,000</td><td>37,146</td></tr><tr><td>DS3</td><td>10,651</td><td>18,919</td><td>4,667</td><td>34,236</td></tr><tr><td>TS</td><td>31,361</td><td>23,648</td><td>6,000</td><td>61,009</td></tr><tr><td>Total demand</td><td>73,373</td><td>98,647</td><td>33,333</td><td>205,353</td></tr><tr><td>Total injection*</td><td>72,189</td><td>106,081</td><td>33,333</td><td>211,603</td></tr></table>

The injection sequence consists of four batches: 92# (52,027 m<sup>3</sup>), 95# (33,333 m<sup>3</sup>), 92# (54,054 m<sup>3</sup>), and 0# (72,189 m<sup>3</sup>), while the pipeline is initially filled with 0# diesel.

## 5.2. Case study

## 5.2.1. Experimental settings

The hyperparameters of the proposed Action-Decoupled SMDP-PPO agent are summarized in Table 5, with parameters for all evaluated methods carefully fine-tuned to be as optimal as possible. To benchmark the proposed framework, two baselines are employed: a standard Single-Actor SMDP-PPO model with the same hyperparameter configuration as the proposed action-decoupled framework, and a solver-based MILP baseline, obtained by reformulating the proposed MINLP model into an MILP via the piecewise linearization technique of Liao et al. (2018a) and solving it with Gurobi under the objective in Eq. (1). Furthermore, the choice of the discrete time window length primarily balances decision granularity against computational complexity. A large interval restricts the feasible solution space, whereas a small interval leads to a combinatorial explosion in the mathematical formulation. Consequently, the base discrete time interval for both the mathematical model and the RL environment is uniformly set to 1 hour.

Table 5  
Hyper-parameters for the Action-Decoupled SMDP-PPO Framework
<table><tr><td>Category</td><td>Parameters</td><td>Value</td></tr><tr><td rowspan="2">Network Architecture</td><td>Actor 1&amp;2 network (Layers × Neurons)</td><td> $2 \times 2 5 6$ </td></tr><tr><td>Critic network (Layers × Neurons)</td><td> $2 \times 2 5 6$ </td></tr><tr><td rowspan="4">PPO Algorithm</td><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Discount factor (γ)</td><td>0.99</td></tr><tr><td>PPO clip coefficient (e)</td><td>0.2</td></tr><tr><td>GAE parameter (λ)</td><td>0.95</td></tr><tr><td>SMDP Setting</td><td>Maximum decision duration  $\left( d ^ { \operatorname* { m a x } } \right)$ </td><td>30</td></tr><tr><td rowspan="6">Reward Weights</td><td>Delivery deviation reward weight (α)</td><td>1.0</td></tr><tr><td>Flow rate stability weight (β)</td><td>0.5</td></tr><tr><td>Pressure penalty weight (η)</td><td>5</td></tr><tr><td>NRMSE threshold (δ)</td><td>0.15</td></tr><tr><td>Terminal penalty coefficient (Γ)</td><td>500</td></tr><tr><td>Switching penalty  $( C _ { s w i t c h } )$ </td><td>2</td></tr></table>

## 5.2.2. Performance comparison and scheduling results

Fig. 4 presents the training curves of the proposed Action-Decoupled SMDP-PPO and the Single-Actor SMDP-PPO baseline with respect to the two optimization objectives, i.e., the delivery deviation $f _ { 1 } \ ( \mathrm { F i g . 4 } ( \mathbf { a } ) )$ and the flow rate stability $f _ { 2 } \left( \mathrm { F i g . 4 ( b ) } \right)$ . Specifically, the total planned injection exceeds the total demand by $6 { , } 2 5 0 \mathrm { m } ^ { 3 }$ (about 3.0% of the total demand) in this case, which makes a zero delivery deviation unattainable and sets the lower bound of $f _ { 1 }$ for this case. Despite this, Fig. 4(a) shows that both methods stabilize the total delivery deviation to below 5% of the total demand during the mid-to-late training stage. A schedule reaching this 5% threshold is regarded as feasible in this work, since the surplus injection makes a lower deviation hardly attainable. Both methods reach this level, confirming that the SMDP formulation efectively optimizes the detailed scheduling policy. Furthermore, the proposed method begins to stabilize around 5,000 episodes, markedly earlier than the single-actor baseline. A consistent advantage is observed for $f _ { 2 }$ in Fig. 4(b): the proposed method converges to a substantially lower flow rate fluctuation than the baseline, indicating smoother segment flow rates and thus a more practical schedule. This contrast verifies that decoupling the discrete duration and continuous flow rate via two cascaded actors eliminates the gradient interference inherent in the single-actor scheme, thereby improving both convergence speed and final accuracy. A more comprehensive comparison between the two architectures is provided in Section 5.2.3.

Fig. 5 depicts a representative detailed scheduling scheme found by the Action-Decoupled SMDP-PPO, presented as a batch migration chart. In this chart, the vertical and horizontal axes correspond to the length coordinate of each station and the operating time, respectively. Each black line traces the trajectory of a batch interface along the pipeline, with its slope indicating the prevailing pipeline flow rate. The injection and delivery operations are marked by colored blocks, where the color identifies the product type and the block thickness scales with the corresponding flow rate. The segment flow rates and hydraulic losses corresponding to this scheme are further reported in Fig. 6. Owing to the sequential action mapping mechanism embedded in the output layer of Actor 2, the flow rate of every pipeline segment is kept strictly within its allowable bounds over the entire horizon. Meanwhile, as the batch interfaces migrate, the hydraulic loss of each segment varies with the changing product distribution. Specifically, the hydraulic loss of Segment 3 (DS2–DS3) turns negative during certain low-flow intervals, where its negative elevation-related pressure drop $P ^ { H }$ (a gravitational head gain from the 58 m descent) outweighs the reduced frictional pressure drop $P ^ { F }$ . At the system level, the cumulative pressure demand along the pipeline stays below the supply capacity of the upstream pump stations, thereby satisfying the pressure constraints in Eqs. (42) and (43). These results confirm that the proposed method yields an operationally feasible schedule that satisfies both the flow rate and pressure constraints. To further evaluate the quality of this feasible schedule, its scheduling objectives and computational cost are compared against both baselines.

![](images/fcee3e2eaa816ea285116fd102b62cf908feb877462765b5cd9062bff003834e.jpg)

![](images/7e11ce8089c1339129639803d8b4cf930b629137fc45f3fc1777e401599f29d2.jpg)  
Figure 4: Comparison of training performance between the proposed Action-Decoupled and Single-Actor SMDP-PPO. (a) Convergence of the delivery deviation $f _ { 1 }$ ; (b) convergence of the flow rate stability $f _ { 2 }$ . The solid line and shaded band denote the mean and standard deviation over three independent runs.

![](images/e7228e7a4e333a8ceda0d5b9e9be74d6966c0bc844bdb618758b8634ecf7e617.jpg)  
Figure 5: Batch migration chart of a representative detailed scheduling scheme found by the Action-Decoupled SMDP-PPO during training.

Table 6 reports the actual delivered volumes at each station, and Table 7 summarizes the overall performance of the three methods, where the schedules of the two RL-based methods correspond to representative results that are attained by the trained policies. Since no cross-plan generalization is considered, each plan is solved by training a dedicated policy from scratch, and the reported solution time accordingly refers to the training time required to reach that schedule. In terms of delivery accuracy, the Gurobi-based MILP model attains the lowest deviation $f _ { 1 } = 6 , 2 5 0 \mathrm { m } ^ { 3 }$ exactly reaching the inherent lower bound set by the surplus injection, which amounts to about 3.0% of the total demand. In contrast, the proposed method reaches $^ { 7 , 1 5 3 \mathrm { m } ^ { 3 } }$ , about 3.5% of the total demand, closely approaching this bound and clearly outperforming the single-actor baseline at $8 { , } 4 4 2 \mathrm { m } ^ { 3 }$ . For the flow rate stability $f _ { 2 } ,$ however, the proposed method gives the smallest value of $5 6 1 ~ \mathrm { m } ^ { 3 } / \mathrm { h }$ , markedly better than the single-actor baseline at $1 { , } 4 4 9 \mathrm { m } ^ { 3 } / \mathrm { h }$ and the Gurobi-based model at 1,697 $\mathrm { m } ^ { 3 } / \mathrm { h }$ . Notably, although Gurobi drives $f _ { 1 }$ to its lower bound, it is terminated with a 13.8% optimality gap after 38,911 s, and since the deviation term is already optimal, this residual gap is essentially attributable to the unoptimized flow rate stability, which explains its poor $f _ { 2 }$ and would require substantially longer computation to be further reduced. The advantage of the proposed method lies in its computational eficiency: it produces a feasible schedule in about 900 s and a representative best schedule in about 1,503 s. The full training of 30,000 episodes takes about 3,000 s in total, beyond which no further improvement is observed. Overall, the proposed method achieves delivery accuracy close to the exact optimum, along with the stable flow rate, while reducing solution time by more than an order of magnitude, making it well suited for the eficient detailed scheduling of multiproduct pipelines.

Planned demand and actual delivered volumes at each station for the representative schedules found by the three methods during training/solving.
<table><tr><td rowspan="3">Station</td><td colspan="4">0# Diesel  $\left( \mathsf { m } ^ { 3 } \right)$ </td><td colspan="4">92# Gasoline  $\left( \mathsf { m } ^ { 3 } \right)$ </td><td colspan="4">95# Gasoline  $\left( \mathsf { m } ^ { 3 } \right)$ </td></tr><tr><td>Demand</td><td>Prop.</td><td>SA</td><td>Gurobi</td><td>Demand</td><td>Prop.</td><td>SA</td><td>Gurobi</td><td>Demand</td><td>Prop.</td><td>SA</td><td>Gurobi</td></tr><tr><td>DPS1</td><td>22,485</td><td>22,875</td><td>22,625</td><td>22,485</td><td>35,810</td><td>35,829</td><td>35,045</td><td>35,810</td><td>14,666</td><td>14,719</td><td>14,651</td><td>14,666</td></tr><tr><td>DS2</td><td>8,876</td><td>10,674</td><td>11,861</td><td>15,005</td><td>20,270</td><td>20,219</td><td>20,333</td><td>20,391</td><td>8,000</td><td>8,280</td><td>7,694</td><td>8,000</td></tr><tr><td>DS3</td><td>10,651</td><td>11,082</td><td>11,434</td><td>10,651</td><td>18,919</td><td>19,088</td><td>19,678</td><td>18,919</td><td>4,667</td><td>4,717</td><td>4,803</td><td>4,667</td></tr><tr><td>TS</td><td>31,361</td><td>35,031</td><td>33,407</td><td>31,361</td><td>23,648</td><td>23,681</td><td>23,998</td><td>23,648</td><td>6,000</td><td>5,792</td><td>6,093</td><td>6,000</td></tr><tr><td>Abs. dev.</td><td></td><td>6,289</td><td>5,954</td><td>6,129</td><td></td><td>272</td><td>1,937</td><td>121</td><td></td><td>592</td><td>551</td><td>0</td></tr></table>

Prop. = Action-Decoupled SMDP-PPO; SA = Single-Actor SMDP-PPO. “Abs. dev.” is the absolute delivery deviation summed over all stations for each product.

Performance comparison of the proposed method, the single-actor baseline, and the Gurobi-based MILP model.
<table><tr><td>Method</td><td> $f _ { 1 } \ \mathrm { ( m } ^ { 3 } \mathrm { ) }$ </td><td> $f _ { 2 } \ \mathrm { ( m ^ { 3 } / h ) }$ </td><td>Solution time (s)</td><td>Gap (%)</td></tr><tr><td>Action-Decoupled SMDP-PPO</td><td>7,153</td><td>561</td><td>1,503</td><td>一</td></tr><tr><td>Single-Actor SMDP-PPO</td><td>8,442</td><td>1,449</td><td>2,059</td><td></td></tr><tr><td>Gurobi</td><td>6,250</td><td>1,697</td><td>38,911</td><td>13.8</td></tr></table>

## 5.2.3. Efect of action decoupling

To further quantify the benefit of the action-decoupled architecture beyond the convergence trends in Fig. 4, two training-process metrics are compared between the proposed method and the single-actor baseline over three independent runs, as shown in Fig. 7. The first metric is the time to reach the first feasible solution, which measures training eficiency. As shown in Fig. 7(a), the proposed method obtains a feasible schedule in about 14.9 min on average, markedly faster than the single-actor baseline at about 21.0 min, and with a smaller variation across runs. The second metric is the number of feasible solutions found during the 30,000-episode training, which reflects the stability of the policy in maintaining feasibility once it is reached. As shown in Fig. 7(b), the proposed method produces substantially more feasible solutions in every run, with an average of 1,143 compared with 617 for the baseline, indicating that, once a feasible region is found, the decoupled policy remains within it far more consistently. These results confirm that decoupling the discrete duration and the continuous flow rate into two cascaded actors not only accelerates convergence but also improves the reliability of the learned policy.

![](images/c8ba593f31dc41ae750c0aee99cbead86ca2f7d37b3404c18774de0d374ab180.jpg)

Segment 2 (DPS1 DS2)  
![](images/96e7f72eb4af7b292d4d6753cb3651fa034650930b0c2d627dc67ee3cd018545.jpg)

Segment 3 (DS2 DS3)  
![](images/6001338d7c1ee75976959d8dff4c043fccb29b8e29754175c697148d76963e8d.jpg)

Segment 4 (DS3 TS)  
![](images/afc3468468ba08f2d2d8f0149aa0c61209028caa08bf13cbeea674923ee1b6ed.jpg)  
Figure 6: Flow rate and hydraulic loss of each pipeline segment under the obtained scheduling scheme. The red step lines and blue lines denote the segment flow rate and hydraulic loss, respectively, and the dashed lines indicate the minimum flow rate limits. (a) Segment 1 (IS–DPS1); (b) Segment 2 (DPS1–DS2); (c) Segment 3 (DS2–DS3); (d) Segment 4 (DS3–TS).

![](images/69a7008901852947aed0fb7289f165f3de42a9f38f7d290ed091e70062898dd4.jpg)

![](images/c0991f93b027dd46f9b2974818631fca354c9578a47bc97ab8ee1535444e99a4.jpg)  
Figure 7: Comparison between the Action-Decoupled and Single-Actor SMDP-PPO. (a) Time to the first feasible solution; (b) number of feasible solutions over 30,000 episodes. Bars and error bars denote mean and standard deviation over three runs; circles mark individual runs.

## 6. Conclusion

This paper proposed an Action-Decoupled SMDP-PPO framework that eficiently solves the detailed scheduling of multiproduct pipelines. The scheduling problem was first formulated as a hybrid-time MINLP model that jointly minimizes the delivery deviation and the flow rate fluctuation under rigorous batch tracking and hydraulic constraints, and was then translated to a semi-Markov decision process so that reinforcement learning could be applied directly. On a real-world pipeline case in China, the learned policy kept the delivery deviation close to the lower bound reached by the linearized MILP model while reducing the solution time by more than an order of magnitude. A cascaded dual-actor architecture was further designed to decouple the discrete operating duration from the continuous flow rates, removing the gradient interference that destabilizes conventional single-actor policies; this decoupling yielded a markedly lower flow rate fluctuation and, as confirmed by our experiments, accelerated convergence and improved the reliability of the learned policy.

Despite these results, the current study trains a dedicated policy for each scheduling plan without cross-plan generalization, on a single-source pipeline topology. Future work will develop generalizable policies that adapt to varying demand plans without retraining, and extend the framework to tree-structured and multi-source pipeline networks to further enhance its practical applicability.

## Acknowledgments

This work was supported by the National Key Research and Development Program of China under Grant Nos. 2021YFA1000104 and 2021YFA1000100, the Laboratory Project of Higher Education Institutions in Shandong Province—Energy System Intelligent Management and Policy Simulation Laboratory at China University of Petroleum, the Youth Innovation Team of Higher Education Institutions in Shandong Province—Data Intelligence Innovation Team at China University of Petroleum, and the Innovation Fund Project for Graduate Students of China University of Petroleum (East China) through the Fundamental Research Funds for the Central Universities under Grant No. 25CX04042A.

## References

Abdellaoui, W., Souier, M., Sahnoun, M., Abdelaziz, F.B., 2021. Multi-period optimal schedule of a multi-product pipeline: A case study in Algeria. Computers & Industrial Engineering 159, 107483.

Bamoumen, M., Elfirdoussi, S., Ren, L., Tchernev, N., 2023. An eficient GRASP-like algorithm for the multi-product straight pipeline scheduling problem. Computers & Operations Research 150, 106082.

Cafaro, D.C., Cerdá, J., 2004. Optimal scheduling of multiproduct pipeline systems using a non-discrete MILP formulation. Computers & Chemica Engineering 28, 2053–2068.

Cafaro, V.G., Cafaro, D.C., Méndez, C.A., Cerdá, J., 2015. Optimization model for the detailed scheduling of multi-source pipelines. Computers & Industrial Engineering 88, 395–409.

Chen, H., Wu, C., Zuo, L., Diao, F., Wang, L., Wang, D., Song, B., 2017a. Optimization of detailed schedule for a multiproduct pipeline using a simulated annealing algorithm and heuristic rules. Industrial & Engineering Chemistry Research 56, 5092–5106.

Chen, H., Zuo, L., Wu, C., Wang, L., Diao, F., Chen, J., Huang, Y., 2017b. Optimizing detailed schedules of a multiproduct pipeline by a monolithic MILP formulation. Journal of Petroleum Science and Engineering 159, 148–163.

Chen, Y., Ye, R., Tao, Z., Liu, H., Chen, G., Peng, J., Ma, J., Zhang, Y., Ji, J., Zhang, Y., 2023. Reinforcement learning for robot navigation with adaptive forward simulation time (AFST) in a semi-Markov model, in: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE. pp. 3597–3604.

Chen, Y., Zhu, J., Dong, H., Zeng, C., Pan, Z., 2025. Knowledge-augmented population-based deep reinforcement learning for real-time network constrained economic dispatch of large-scale power grid. IEEE Transactions on Power Systems .

Dominguez-Barbero, D., Garcia-Gonzalez, J., Sanz-Bobi, M.A., 2023. Twin-delayed deep deterministic policy gradient algorithm for the energy management of microgrids. Engineering Applications of Artificial Intelligence 125, 106693.

Fan, L., Su, H., Wang, W., Zio, E., Zhang, L., Yang, Z., Peng, S., Yu, W., Zuo, L., Zhang, J., 2022. A systematic method for the optimization of gas supply reliability in natural gas pipeline network based on bayesian networks and deep reinforcement learning. Reliability Engineering & System Safety 225, 108613.

Han, P., Zhang, Q., He, W., Chen, Y., Zhao, B., Li, Y., Zhou, G., 2024. A double inference engine belief rule base for oil pipeline leakage. Expert Systems with Applications 240, 122587.

Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., Scaramuzza, D., 2023. Champion-level drone racing using deep reinforcement learning. Nature 620, 982–987.

Li, Y., Ding, D., Xie, H., Zhao, L., Jin, Y., Xie, Y., Fang, Z., 2025. SPAD: Spatial perception and action decoupling empowered multi-agent ai task scheduling framework in cloud-edge computing. IEEE Transactions on Networking 34, 1896–1911.

Liao, Q., Liang, Y., Tu, R., Huang, L., Zheng, J., Wang, G., Zhang, H., 2022. Innovations of carbon-neutral petroleum pipeline: A review. Energy reports 8, 13114–13128.

Liao, Q., Liang, Y., Xu, N., Zhang, H., Wang, J., Zhou, X., 2018a. An MILP approach for detailed scheduling of multi-product pipeline in pressure control mode. Chemical Engineering Research and Design 136, 620–637.

Liao, Q., Tu, R., Xu, B., Huang, M., Li, J., Zheng, Z., Liang, Y., 2026. Optimizing large-scale multiproduct pipeline scheduling: a novel hierarchical rolling-horizon framework. Expert Systems with Applications 311, 131385.

Liao, Q., Zhang, H., Xia, T., Chen, Q., Li, Z., Liang, Y., 2019. A data-driven method for pipeline scheduling optimization. Chemical Engineering Research and Design 144, 79–94.

Liao, Q., Zhang, H., Xu, N., Liang, Y., Wang, J., 2018b. A MILP model based on flowrate database for detailed scheduling of a multi-product pipeline with multiple pump stations. Computers & Chemical Engineering 117, 63–81.

Lin, J., Liu, G., Chen, L., He, Y., You, Z., 2025. Optimizing batch scheduling of multiproduct pipelines using a smooth modeling approach. SPE Journal 30, 808–822.

Liu, M., Dong, N., Wang, X., Jian, L., 2026. Physics-based and data-driven adaptive compressor optimization in gas pipeline systems under dynamic and non-isothermal conditions. Engineering Applications of Artificial Intelligence 167, 113914.

Liu, Z.E., Li, Y., Zhou, Q., Li, Y., Shuai, B., Xu, H., Hua, M., Tan, G., Xu, L., 2024. Deep reinforcement learning-based energy management for heavy duty HEV considering discrete-continuous hybrid action space. IEEE Transactions on Transportation Electrification 10, 9864–9876.

Meira, W.H.T., Magatao, L., Neves Jr, F., Arruda, L.V., Vaqueiro, J.P., Relvas, S., Barbosa-Povoa, A.P., 2021. A solution framework for the long-term scheduling and inventory management of straight pipeline systems with multiple-sources. Computers & Operations Research 127, 105143.

MirHassani, S., Jahromi, H.F., 2011. Scheduling multi-product tree-structure pipelines. Computers & Chemical Engineering 35, 165–176.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., et al., 2015. Human-level control through deep reinforcement learning. Nature 518, 529–533.

Mostafaei, H., Castro, P.M., Oliveira, F., Harjunkoski, I., 2021a. Eficient formulation for transportation scheduling of single refinery multiproduct pipelines. European Journal of Operational Research 293, 731–747.

Mostafaei, H., Castro, P.M., Relvas, S., Harjunkoski, I., 2021b. A holistic MILP model for scheduling and inventory management of a multiproduct oil distribution system. Omega 98, 102110.

Nie, B., Fu, Y., Gao, Y., 2025. Select before act: Spatially decoupled action repetition for continuous control, in: Proceedings of the International Conference on Learning Representations.

Pei, S., Hoang, L., Fu, G., Butler, D., 2025. Real-time pump scheduling in water distribution networks using deep reinforcement learning. Journal of Water Resources Planning and Management 151, 04025012.

Rejowski Jr, R., Pinto, J.M., 2003. Scheduling of a multiproduct pipeline system. Computers & Chemical Engineering 27, 1229–1246.

Rejowski Jr, R., Pinto, J.M., 2008. A novel continuous time representation for the scheduling of pipeline systems with pumping yield rate constraints. Computers & Chemical Engineering 32, 1042–1066.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O., 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .

Shao, K., Wang, X., Liu, M., Xu, A., Jian, L., 2025. Eficient pump scheduling for large-scale multiproduct pipelines using deep reinforcement learning. International Journal of Adaptive Control and Signal Processing 39, 1567–1579.

Sidki, M., Tchernev, N., Féniès, P., Ren, L., Elfirdoussi, S., 2025. Multiproduct pipeline scheduling: A comprehensive bibliometric analysis and a systematic literature review. Computers & Chemical Engineering 192, 108911.

Sutton, R.S., Barto, A.G., 1998. Reinforcement learning: An introduction. volume 1. MIT press Cambridge.

Tu, R., Zhang, H., Xu, B., Huang, X., Che, Y., Du, J., Wang, C., Qiu, R., Liang, Y., 2024. Machine learning application in batch scheduling for multi-product pipelines: A review. Journal of Pipeline Science and Engineering 4, 100180.

Xu, N., Liao, Q., Li, Z., Liang, Y., Qiu, R., Zhang, H., 2021. An eficient decomposition approach for the low-energy scheduling of a straight multiproduct pipeline. Chemical Engineering Research and Design 165, 341–360.

Yan, Y., Castro, P.M., Liao, Q., Liang, Y., 2021. An efective decomposition algorithm for scheduling branched multiproduct pipelines. Computers & Chemical Engineering 154, 107494.

Zhang, H., Liang, Y., Liao, Q., Shen, Y., Yan, X., 2018. A self-learning approach for optimal detailed scheduling of multi-product pipeline. Journa of Computational and Applied Mathematics 327, 41–63.

Zhang, H., Liang, Y., Liao, Q., Wu, M., Yan, X., 2017. A hybrid computational approach for detailed scheduling of products in a pipeline with multiple pump stations. Energy 119, 612–628.

Zhou, X., Liang, Y., Zhang, X., Liao, Q., Gao, S., Zhang, W., Zhang, H., 2019a. A MILP model for the detailed scheduling of multiproduct pipelines with the hydraulic constraints rigorously considered. Computers & Chemical Engineering 130, 106543.

Zhou, X., Zhang, H., Qiu, R., Liang, Y., Wu, G., Xiang, C., Yan, X., 2019b. A hybrid time MILP model for the pump scheduling of multi-product pipelines based on the rigorous description of the pipeline hydraulic loss changes. Computers & Chemical Engineering 121, 174–199.

## A. Nomenclature

Nomenclature   
Sets and indices   
$i \in I$ Set of stations and pipeline segments. Pipeline segment � connects station � with station $i + 1$ $i = 1 , 2 , \cdots , i _ { \mathrm { m a x } }$   
$j \in J$ Set of batches, $j = 1 , 2 , \cdots , j _ { \mathrm { m a x } }$   
$J ^ { n e w }$ The set of batches to be injected into the pipeline during the scheduling horizon   
$J ^ { o l d }$ The set of batches in the pipeline at the start of the scheduling horizon   
$J _ { o }$ The set of batches that belong to product type �   
$k \in K$ Index set of time nodes, $k = 1 , 2 , \ldots , k _ { \mathrm { m a x } }$   
$o \in O$ Set of product types   
$t _ { k } \in T$ Time value of node � (ℎ)   
Parameters   
$\lambda _ { i }$ Weight factor of pipeline segment � used to prevent flow rate fluctuations from being concentrated in specific segments   
$\nu _ { j }$ Kinematic viscosity of batch $j \left( \mathrm { m } ^ { 2 } / \mathrm { s } \right)$   
$\rho _ { j }$ Density of batch � $\left( \mathrm { k g } / \mathrm { m } ^ { 3 } \right)$   
$d _ { i }$ Inner diameter of pipeline segment � (m)   
$E _ { i }$ Elevation of station � (m)   
$H _ { k , i , j }$ Elevation diference spanned by batch � in segment � at time node � (m)   
$L _ { k , i , j }$ Physical length of batch � in segment � at time node � (m)   
$P _ { k , i } ^ { F }$ Frictional pressure drop of pipeline segment � at time node � (Pa)   
$P _ { k , i } ^ { H }$ Elevation pressure drop of pipeline segment � at time node � (Pa)   
$P _ { i } ^ { \operatorname* { m a x } }$ Maximum allowable pressure provided by station � (Pa)

<table><tr><td> $Q _ { k , i } ^ { p }$ </td><td>Flow rate of pipeline segment i during time interval  $\left( t _ { k } , t _ { k + 1 } \right) \left( \mathrm { m } ^ { 3 } / \mathrm { h } \right)$ </td></tr><tr><td> $Q _ { i } ^ { d , \operatorname* { m i n } } , Q _ { i } ^ { d , \operatorname* { m a x } }$ </td><td>Minimum and maximum allowable delivery flow rates at station i  $\left( \mathrm { m } ^ { 3 } / \mathrm { h } \right)$ </td></tr><tr><td> $Q _ { i } ^ { p , \mathrm { { m i n } } } , Q _ { i } ^ { p , \mathrm { { m a x } } }$ </td><td>Minimum and maximum allowable flow rates for pipeline segment i  $\left( \mathrm { m } ^ { 3 } / \mathrm { h } \right)$ </td></tr><tr><td> $Q _ { k } ^ { i n }$ </td><td>The injection flow rate at the initial station during time interval  $\left( t _ { k } , t _ { k + 1 } \right) \left( \mathrm { m } ^ { 3 } / \mathrm { h } \right)$ </td></tr><tr><td> $Q _ { k , i } ^ { d }$ </td><td>The delivery flow rate at station i during time interval  $\left( t _ { k } , t _ { k + 1 } \right) \left( \mathrm { m } ^ { 3 } / \mathrm { h } \right)$ </td></tr><tr><td> $V _ { i , o } ^ { d }$ </td><td>Actual delivered volumes during the whole study horizon for product o at delivery station i</td></tr><tr><td> $V _ { i } ^ { s }$ </td><td>the volumetric coordinate of the delivery station  $i \left( \mathrm { m } ^ { 3 } \right)$ </td></tr><tr><td> $V _ { i , o } ^ { x }$ </td><td>Demand volumes for product o at delivery station  $i \left( \mathrm { m } ^ { 3 } \right)$ </td></tr><tr><td> $V _ { k , j } ^ { b }$ </td><td>The volumetric coordinate of the head interface of batch j at time node k  $\left( \mathbf { m } ^ { 3 } \right)$ </td></tr><tr><td colspan="2">Binary variables</td></tr><tr><td> $B _ { k }$ </td><td>1 if a head interface reaches a station at time node k; 0 otherwise</td></tr><tr><td> $S _ { k , i } ^ { d }$ </td><td>1 if station i delivers products during time interval  $\left( t _ { k } , t _ { k + 1 } \right)$  ; 0 otherwise</td></tr><tr><td> $S _ { k , i , j } ^ { c o v e r }$ </td><td>1 if batch j covers station i (available for delivery) at time node k; 0 otherwise</td></tr><tr><td> $S _ { k , i , j } ^ { h e a d }$ </td><td>1 if the head interface of batch j has passed station i at time node  $k ; 0$  otherwise</td></tr><tr><td> $Z _ { k , i , j } ^ { a l l \_ i n }$ </td><td>1 if the whole batch j is fully inside pipeline segment i at time node  $k ; 0$  otherwise</td></tr><tr><td> $Z _ { k , i , j } ^ { H \_ i n }$ </td><td>1 if only the head interface of batch j is in pipeline segment i at time node  $k ; 0$  otherwise</td></tr><tr><td> $Z _ { k , i , j } ^ { T \_ i n }$ </td><td>1 if only the tail interface of batch j is in pipeline segment i at time node  $k ; 0$  otherwise</td></tr><tr><td> $Z _ { k , i , j } ^ { t h r u }$ </td><td>1 if the head interface of batch j has passed through station i + 1 but the tail interface has not passed through station i at time node k; 0 otherwise</td></tr></table>