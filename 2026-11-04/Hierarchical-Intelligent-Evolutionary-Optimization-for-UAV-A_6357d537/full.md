# Hierarchical Intelligent Evolutionary Optimization for UAV-Assisted NOMA-MEC Systems

Yaxuan Jiao<sup>∗</sup>, Jiaxuan Lv<sup>∗</sup>, Jyoti Sahni<sup>†</sup>, Wuyungerile Li<sup>∗</sup>, Winston K.G. Seah<sup>†</sup>

<sup>∗</sup>School of Computer Science, Inner Mongolia University, Hohhot, China

Email: 32409126@mail.imu.edu.cn, 1518371393@qq.com, gerile@imu.edu.cn

<sup>†</sup>School of Engineering and Computer Science, Victoria University of Wellington, Wellington, New Zealand Email: jyoti.sahni@vuw.ac.nz, winston.seah@ecs.vuw.ac.nz

Abstract—The integration of unmanned aerial vehicles (UAVs), mobile edge computing (MEC) and non-orthogonal multiple access (NOMA) provides an effective solution for computationintensive and latency-sensitive services in infrastructure-deficient environments. However, the joint optimization of UAV trajectories, task offloading, communication and computing resources under heterogeneous task priorities results in a highly coupled mixed-integer non-convex problem. To address these challenges, this paper proposes a Hierarchical Intelligent Evolutionary Algorithm (HIEA) for priority-aware multi-UAV cooperative NOMA-MEC networks, which integrates multi-head deep Q-network (MH-DQN)-based intelligent decision generation, communication-aware initialization, adaptive evolutionary exploration and successive convex approximation (SCA)-based precision refinement. Specifically, a novel communication-aware initialization mechanism, termed Channel Quality Indicator (CQI)-guided Load-Aware Trajectory Initialization (CQI-LATI), is designed to incorporate CQI and task workload characteristics into evolutionary population construction, generating highquality and diversity-preserving initial populations. Theoretical analysis characterizes HIEA’s search-space reduction, the quality-diversity trade-off of CQI-guided initialization, and the accuracy-efficiency trade-off of SCA refinement, and extensive simulations demonstrate that HIEA outperforms benchmark schemes, with gains of up to 6.5% in throughput, 13.6% in task completion rate and 15.3% in energy efficiency.

Index Terms—Multi-UAV MEC, NOMA, Priority-Aware Task Offloading, Trajectory Optimization, Evolutionary Optimization.

## I. INTRODUCTION

The rapid development of Internet of Things (IoT), intelligent transportation and emergency communication services has generated increasing demands for computation-intensive and latency-sensitive applications. In such scenarios, tasks exhibit heterogeneous priorities, as emergency and missioncritical services demand stricter latency and reliability guarantees than routine IoT traffic. However, traditional cloud computing architectures suffer from long transmission latency and limited service availability, especially in remote and infrastructure-deficient environments. Mobile edge computing (MEC) enables low-latency services by deploying computing resources closer to end users [1]. Meanwhile, unmanned aerial vehicles (UAVs), with flexible deployment capability and three-dimensional mobility, have emerged as promising platforms for providing communication and computing services in dynamic scenarios [2].

To support massive connectivity and improve spectrum utilization, non-orthogonal multiple access (NOMA) has been widely adopted in UAV-assisted MEC networks [3, 4]. Consequently, the joint optimization of UAV trajectories, task offloading and computing resource management has become a fundamental problem in multi-UAV cooperative MEC systems [5, 6]. However, the strong coupling among heterogeneous decision variables makes this problem a large-scale mixed-integer non-convex optimization problem with a highly complex search space.

Existing studies have explored decomposition-based, heuristic, evolutionary and learning-based approaches for UAV-MEC optimization. Decomposition-based methods, such as block coordinate descent with convex relaxation, decouple the problem into tractable subproblems [7], but converge only to locally optimal solutions and are sensitive to initialization conditions. Heuristic methods, such as greedy and matching-based strategies, offer low complexity, yet provide no optimality guarantees and generalize poorly across network conditions. Evolutionary methods offer robust global search for nonlinear mixed-integer optimization problems [8–10] but suffer from high computational cost and poor efficiency. Meanwhile, learning-based methods enable adaptive decisionmaking under dynamic network conditions by exploiting historical interactions [11–13], although end-to-end learning requires extensive training and struggles to guarantee constraint satisfaction.

Evolutionary search thus best fits this problem class, and its limitations can be alleviated by incorporating prior knowledge [14] and learning techniques into the search framework. Realizing such an integrated framework, however, involves several challenges. First, evolutionary performance depends heavily on population initialization and diversity preservation, which govern initial solution quality, convergence efficiency and the exploration-exploitation balance [15–17]. Although knowledge-assisted initialization has demonstrated the value of prior information in population construction [14], such approaches mainly exploit mathematical properties or historical optimization information, while rarely considering domainspecific knowledge such as wireless channel conditions, user distribution and task heterogeneity.

Second, existing UAV-MEC optimization studies mainly focus on optimizing final decision variables, such as UAV trajectories, offloading strategies and resource allocation, leaving the impact of communication-aware initialization on search efficiency largely unexplored. Random or geometry-based initialization may generate low-quality candidates, resulting in inefficient exploration in large-scale search spaces. Third, although local refinement techniques can improve the accuracy of the solution, they usually introduce additional computational complexity and limit scalability.

To address these challenges, this paper proposes a Hierarchical Intelligent Evolutionary Algorithm (HIEA) for priorityaware multi-UAV cooperative NOMA-MEC networks. Unlike conventional frameworks that mainly improve final solutions, HIEA enhances the whole evolutionary process through four coordinated functions: multi-head deep Q-network (MH-DQN)-based intelligent decision generation, communicationaware initialization, adaptive evolutionary exploration and successive convex approximation (SCA)-based precision refinement.

The key innovation of HIEA lies in the proposed communication-aware initialization mechanism. By jointly exploiting wireless channel quality and task workload characteristics, the proposed mechanism constructs feasible and diverse initial populations, introducing communication-oriented guidance into population construction and adaptively balancing initialization quality and population diversity. Building on this initialization, the remaining functions jointly drive the search: MH-DQN injects learned decision candidates into the population, adaptive evolution performs global exploration, and SCA-based refinement sharpens elite solutions.

The main contributions of this paper are summarized as follows:

• We formulate a priority-aware joint optimization problem for multi-UAV cooperative NOMA-MEC networks, jointly optimizing UAV trajectories, task offloading and computing resource management.

• We propose HIEA, a hierarchical framework that coordinates MH-DQN-based intelligent decision generation, communication-aware initialization, adaptive evolutionary exploration and SCA-based precision refinement to solve the formulated mixed-integer non-convex problem.

• Within HIEA, we design a new communication-aware initialization mechanism, termed Channel Quality Indicator (CQI)-guided Load-Aware Trajectory Initialization (CQI-LATI), which incorporates CQI and task workload characteristics into evolutionary population construction.

Theoretical analysis evaluates MH-DQN search-space reduction, CQI-guided initialization, and SCA refinement tradeoffs. This is further supplemented by extensive simulations that demonstrate HIEA outperforms benchmark schemes in throughput, task completion rate and energy efficiency.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

We consider a NOMA-enabled multi-UAV mobile edge computing (UAV-MEC) system consisting of a set of ground users $\mathcal { K } = \{ 1 , \ldots , K \}$ and a set of UAVs $\mathcal { U } = \{ 1 , \dots , U \}$ Due to the lack of terrestrial infrastructure, users rely on UAVs for task offloading and computation services. The UAVs dynamically adjust their three-dimensional (3D) positions to provide communication and computing support.

Users are partitioned into U NOMA clusters according to UAV association. Let $\mathcal { C } _ { u }$ denote the user cluster served by UAV u. Each user is associated with only one UAV and each UAV serves one cluster. All UAVs operate on the same frequency band and thus inter-UAV interference is considered.

The system operates in discrete time slots $\mathcal { T } = \{ 1 , \ldots , T \}$ Users move randomly within the service area and generate computation tasks with different priorities. The task generated by user k at time slot t is represented as $q _ { k } ( t ) \ =$ $( c _ { k } ( t ) , n _ { k } ( t ) , m _ { k } ( t ) , o _ { k } ( t ) )$ where $c _ { k } ( t ) , n _ { k } ( t ) , m _ { k } ( t )$ and $o _ { k } ( t )$ denote the required computation (CPU cycles), input data size, latency requirement and task priority, respectively. Since mobile devices have limited computing capability, tasks can be offloaded to UAVs for execution.

The air-to-ground (A2G) channel between users and UAVs follows a probabilistic path-loss model incorporating both lineof-sight (LoS) and non-line-of-sight (NLoS) links [11].

At time slot t, let $( x _ { u } ( t ) , y _ { u } ( t ) , Z _ { u } ( t ) )$ denote the three-dimensional position of UAV u, where $x _ { u } ( t )$ and $y _ { u } ( t )$ represent the horizontal coordinates and $Z _ { u } ( t )$ represents the flight altitude. The horizontal distance between MU k and UAV u is denoted by $\begin{array} { r l } { d _ { k , u } ( t ) } & { { } = } \end{array}$ $\sqrt { ( x _ { u } ( t ) - x _ { k } ( t ) ) ^ { 2 } + ( y _ { u } ( t ) - y _ { k } ( t ) ) ^ { 2 } }$ . The elevation angle is defined as $\begin{array} { r } { \kappa _ { k , u } ( t ) = \frac { \dot { 1 } 8 0 } { \pi } \arctan ( \frac { Z _ { u } ( t ) } { d _ { k , u } ( t ) } ) } \end{array}$ . Then, the LoS probability between MU k and UAV u at time slot t is given by

$$
\begin{array} { r } { P _ { k , u } ^ { \mathrm { L o S } } ( t ) = \left\{ \begin{array} { l l } { 1 , } & { d _ { k , u } ( t ) \leq d _ { 0 } , } \\ { \frac { 1 } { 1 + a \exp \big ( - b ( \kappa _ { k , u } ( t ) - a ) \big ) } , } & { d _ { k , u } ( t ) > d _ { 0 } , } \end{array} \right. } \end{array}\tag{1}
$$

where a and b are environment-dependent constants and $d _ { 0 }$ is the horizontal distance threshold for guaranteed LoS transmission.

Accordingly, the NLoS probability is given by $P _ { k , u } ^ { \mathrm { N L o S } } ( t ) =$ $1 \mathrm { ~ - ~ } P _ { k , u } ^ { \mathrm { L o S } } ( t )$ . Following the air-to-ground channel model in [12], the path loss between MU k and UAV u under LoS and NLoS conditions (in dB) are respectively given by

$$
\begin{array} { r } { L ^ { \mathrm { L o S } } ( t ) = 3 0 . 9 + ( 2 2 . 2 5 - 0 . 5 \log _ { 1 0 } Z _ { u } ( t ) ) \log _ { 1 0 } d } \\ { + 2 0 \log _ { 1 0 } f _ { c } , \qquad } \end{array}\tag{2}
$$

$$
\begin{array} { r l } & { L ^ { \mathrm { N L o S } } ( t ) = \operatorname* { m a x } \Big \{ L ^ { \mathrm { L o S } } ( t ) , } \\ & { ~ 3 2 . 4 + ( 4 3 . 2 - 3 . 1 \log _ { 1 0 } Z _ { u } ( t ) ) \log _ { 1 0 } d } \\ & { ~ + ~ 2 0 \log _ { 1 0 } f _ { c } \Big \} , } \end{array}\tag{3}
$$

where d denotes the 3D distance between MU k and UAV u, given by $d = \sqrt { d _ { k , u } ^ { 2 } ( t ) + Z _ { u } ^ { 2 } ( t ) }$ , and $f _ { c }$ is the carrier frequency. Therefore, the average path loss is expressed as

$$
P L _ { k , u } ( t ) = P _ { k , u } ^ { \mathrm { L o S } } ( t ) L ^ { \mathrm { L o S } } ( t ) + P _ { k , u } ^ { \mathrm { N L o S } } ( t ) L ^ { \mathrm { N L o S } } ( t ) .\tag{4}
$$

Considering small-scale fading, the channel gain between MU k and UAV u at time slot t is given by

$$
g _ { k , u } ( t ) = H _ { k , u } ( t ) \cdot 1 0 ^ { - \frac { P L _ { k , u } ( t ) } { 1 0 } } ,\tag{5}
$$

where $H _ { k , u } ( t )$ denotes the small-scale fading power gain between UAV u and MU k [12].

## A. MU and UAV Mobility Models

UAV and MU GPS locations are used for cluster association. Users are grouped into UAV-associated clusters according to their spatial distribution. Both random-walk and directional mobility models are considered, following the mobility models in [3]. For the random mobility model, the movement direction and speed of MU k at each discrete time slot t are randomly generated, following uniform distributions $\theta _ { k } ( t ) \sim \mathcal { U } ( 0 , 2 \pi )$ and $V _ { k } ^ { \mathrm { M U } } ( t ) \sim \mathcal { U } ( \bar { 0 } , V _ { \operatorname* { m a x } } ^ { \mathrm { M U } } )$ , respectively. For the directional mobility model, the movement of each user at time slot t follows a vectorial mobility model, where the velocity is determined by the combination of a dominant direction component and a random perturbation component. Specifically, the velocity vector is expressed as $\mathbf { V } _ { k } ( t ) = \mathbf { D } _ { k } ( t ) + \boldsymbol { \epsilon } _ { k } ( t )$ where $\mathbf { D } _ { k } ( t )$ and $\epsilon _ { k } ( t )$ denote the dominant direction vector and random perturbation vector, respectively. The dominant direction vector $\mathbf { D } _ { k } ( t )$ is introduced to represent the main moving tendency of MU k. Notably, the direction and magnitude satisfy $\theta _ { k } ^ { D } ( t ) = \Theta$ and $\begin{array} { r } { \| \mathbf { D } _ { k } \mathbf { \bar { ( } } t { ) } \| = \frac { 4 } { 5 } V _ { \operatorname* { m a x } } ^ { \mathrm { M U } } } \end{array}$ , where Θ denotes the predefined dominant moving direction of MU k. The random perturbation vector $\epsilon _ { k } ( t )$ has a random direction $\theta _ { k } ^ { \epsilon } ( t ) \sim \mathcal { U } ( 0 , 2 \pi )$ and a magnitude $\begin{array} { r } { \| \epsilon _ { k } ( t ) \| \sim \mathcal { U } ( 0 , \frac { 1 } { 5 } V _ { \operatorname* { m a x } } ^ { \mathrm { M U } } ) } \end{array}$ The position of MU k at time slot t is denoted as $\mathbf { M } _ { k } ( t ) =$ $[ x _ { k } ( t ) , y _ { k } ( t ) , 0 ]$ . Let $d _ { k } ( t )$ denote the traveling distance of MU k during time slot t. Then, the position update at time slot $t { + } 1$ is given by

$$
x _ { k } ( t + 1 ) = x _ { k } ( t ) + d _ { k } ( t ) \cos \big ( \theta _ { k } ( t ) \big ) \quad \forall k , t\tag{6}
$$

$$
y _ { k } ( t + 1 ) = y _ { k } ( t ) + d _ { k } ( t ) \sin \big ( \theta _ { k } ( t ) \big ) \quad \forall k , t\tag{7}
$$

At time slot t, the three-dimensional position of UAV u is denoted as ${ \bf q } _ { u } ( t ) = ( x _ { u } ( t ) , y _ { u } ( t ) , z _ { u } ( t ) )$ . Different from mobile users with stochastic mobility patterns, UAVs act as controllable aerial MEC servers and their trajectories are optimized to improve communication and computation performance. The UAV trajectory evolution is expressed as

$$
\begin{array} { r } { \mathbf q _ { u } ( t + 1 ) = \mathbf q _ { u } ( t ) + \mathbf v _ { u } ( t ) \Delta t , \quad \forall u , t , } \end{array}\tag{8}
$$

where $\mathbf { v } _ { u } ( t )$ denotes the velocity vector of UAV u and $\Delta t$ is the duration of one time slot.

To ensure that UAVs remain within the rectangular service area, the horizontal coordinates must satisfy

$$
0 \leq x _ { u } ( t ) \leq X _ { \operatorname* { m a x } } , \quad 0 \leq y _ { u } ( t ) \leq Y _ { \operatorname* { m a x } }\tag{9}
$$

Moreover, the altitude of UAVs is restricted within the allowable flight range:

$$
z _ { \mathrm { m i n } } \le z _ { u } ( t ) \le z _ { \mathrm { m a x } } , \quad \forall u , t .\tag{10}
$$

To guarantee feasible UAV movement, the displacement between two consecutive time slots is constrained by the maximum flight speed:

$$
\| \mathbf { q } _ { u } ( t + 1 ) - \mathbf { q } _ { u } ( t ) \| \leq V _ { \operatorname* { m a x } } ^ { \mathrm { U A V } } \Delta t , \quad \forall u , t .\tag{11}
$$

where $V _ { \mathrm { m a x } } ^ { \mathrm { U A V } }$ denotes the maximum flight speed of UAVs.

Furthermore, to avoid collisions among UAVs, the distance between any two UAVs should be larger than a predefined safety distance:

$$
\| { \bf q } _ { m } ( t ) - { \bf q } _ { n } ( t ) \| \geq D _ { \mathrm { m i n } } , \quad \forall m \neq n , t .\tag{12}
$$

## B. NOMA Transmission Model

MUs access UAVs via uplink NOMA. Following the SIC principle, each UAV decodes users in descending channel-gain order and successively removes the signals of already decoded users.

Consider users associated with UAV u, indexed according to their channel gains as

$$
g _ { 1 , u } ( t ) \geq g _ { 2 , u } ( t ) \geq \cdot \cdot \cdot \geq g _ { K , u } ( t ) .\tag{13}
$$

The SINR of user k at time slot t is

$$
\gamma _ { k , u } ( t ) = \frac { g _ { k , u } ( t ) p _ { k } ( t ) } { I _ { k , u } ^ { \mathrm { i n t r a } } ( t ) + I _ { k , u } ^ { \mathrm { i n t e r } } ( t ) + \sigma ^ { 2 } } ,\tag{14}
$$

where $p _ { k } ( t )$ is the transmit power, $\sigma ^ { 2 }$ is the AWGN power, $\begin{array} { r } { I _ { k , u } ^ { \mathrm { i n t r a } } ( t ) \ = \ \sum _ { j = k + 1 } ^ { K } g _ { j , u } ( t ) p _ { j } ( t ) } \end{array}$ is intra-UAV interference from weaker users, and $\begin{array} { r } { I _ { k , u } ^ { \mathrm { i n t e r } } ( t ) = \sum _ { s \neq u } \sum _ { i \in \mathcal { C } _ { s } } g _ { i , u } ( t ) p _ { i } ( t ) } \end{array}$ is inter-UAV interference. The uplink rate is given by

$$
R _ { k , u } ( t ) = B \log _ { 2 } { \left( 1 + \gamma _ { k , u } ( t ) \right) } ,\tag{15}
$$

where B is the system bandwidth.

## C. User Association and Computation-Energy Model

A binary variable $A _ { k , u } ( t ) \in \{ 0 , 1 \}$ indicates whether user k is associated with UAV u at slot t, subject to

$$
\sum _ { u = 1 } ^ { U } A _ { k , u } ( t ) = 1 , \quad \forall k , t .\tag{16}
$$

The user set of UAV u is $\mathcal { C } _ { u } ( t ) = \{ k \ | \ A _ { k , u } ( t ) = 1 \}$ . Let $W _ { k , u } ( t ) \in \{ 0 , 1 \}$ denote offloading from user k to $\mathrm { U A V } \ u ,$ with feasibility constraint $W _ { k , u } ( t ) \leq A _ { k , u } ( t )$ . Define $W _ { k } ( t ) =$ $\textstyle \sum _ { u } W _ { k , u } ( t ) \in \{ 0 , 1 \}$ , then the delay of user k at slot t is given by

$$
\begin{array} { l } { \displaystyle T _ { k } ( t ) = \ ( 1 - W _ { k } ( t ) ) \frac { c _ { k } ( t ) } { f _ { k } ^ { \mathrm { l o c } } } } \\ { \displaystyle \qquad + \sum _ { u = 1 } ^ { U } W _ { k , u } ( t ) \left( \frac { n _ { k } ( t ) } { R _ { k , u } ( t ) } + \frac { c _ { k } ( t ) } { f _ { k , u } ^ { \mathrm { U A V } } ( t ) } \right) , } \end{array}\tag{17}
$$

where $c _ { k } ( t )$ and $n _ { k } ( t )$ are the required CPU cycles and input data size, respectively.

The user energy is

$$
\begin{array} { l } { { \displaystyle E _ { k } ( t ) = ( 1 - W _ { k } ( t ) ) \kappa _ { \mathrm { m } } \big ( f _ { k } ^ { \mathrm { l o c } } \big ) ^ { 2 } c _ { k } ( t ) } \ ~ } \\ { { \displaystyle ~ + \sum _ { u = 1 } ^ { U } W _ { k , u } ( t ) \big ( P _ { k } ^ { \mathrm { t x } } ( t ) + P _ { k } ^ { \mathrm { p r o c } } ( t ) \big ) T _ { k , u } ^ { \mathrm { t x } } ( t ) , } } \end{array}\tag{18}
$$

where $T _ { k , u } ^ { \mathrm { t x } } ( t ) = n _ { k } ( t ) / R _ { k , u } ( t )$

For UAV u, the flight power is

$$
P _ { u } ^ { \mathrm { f l y } } ( t ) = \left\{ \begin{array} { l l } { P _ { \mathrm { h o v e r } } + \alpha V _ { u } ^ { 2 } ( t ) + \beta V _ { u } ^ { 3 } ( t ) , } & { V _ { u } ( t ) > 0 , } \\ { P _ { \mathrm { h o v e r } } , } & { \mathrm { h o v e r i n g } , } \end{array} \right.\tag{19}
$$

and the slot energy is

$$
E _ { u } ( t ) = P _ { u } ^ { \mathrm { f l y } } ( t ) \Delta t + \sum _ { k } W _ { k , u } ( t ) \kappa _ { \mathrm { u } } { \left( f _ { k , u } ^ { \mathrm { U A V } } \right) } ^ { 2 } c _ { k } ( t ) .\tag{20}
$$

Therefore, the system throughput and total energy are

$$
R _ { \mathrm { s u m } } = \sum _ { t = 1 } ^ { T } \sum _ { k \in \mathcal { K } } R _ { k } ( t ) ,\tag{21}
$$

$$
E _ { \mathrm { t o t } } = \sum _ { t = 1 } ^ { T } \left( \sum _ { u } E _ { u } ( t ) + \sum _ { k } E _ { k } ( t ) \right) .\tag{22}
$$

## D. Priority-Aware Satisfaction Model

The satisfaction model is adapted from the delay-sensitive utility model in [5] and further extended to capture heterogeneous task priorities by introducing distinct satisfaction functions for low- and high-priority tasks.

To characterize heterogeneous task priorities, let $o _ { k } ( t ) \in$ {0, 1} denote the priority indicator, where $o _ { k } ( t ) = 1$ represents a high-priority task. Define the delay deviation as

$$
\eta _ { k } ( t ) = T _ { k } ( t ) - m _ { k } ( t ) ,
$$

where $m _ { k } ( t )$ is the maximum tolerable delay. The user satisfaction is modeled as

$$
\Phi _ { k } ( t ) = ( 1 - o _ { k } ( t ) ) \Phi _ { k } ^ { \mathrm { L } } ( t ) + o _ { k } ( t ) \Phi _ { k } ^ { \mathrm { H } } ( t ) ,\tag{23}
$$

where

$$
\Phi _ { k } ^ { \mathrm { L } } ( t ) = \left\{ { { P } _ { L } } , \ \qquad \eta _ { k } ( t ) \le 0 , \right.\tag{24}
$$

$$
\Phi _ { k } ^ { \mathrm { H } } ( t ) = \left\{ \begin{array} { l l } { P _ { L } + P _ { H } \log _ { 2 } ( 1 - \eta _ { k } ( t ) ) , } & { \eta _ { k } ( t ) \leq 0 , } \\ { - P _ { H } , } & { \eta _ { k } ( t ) > 0 . } \end{array} \right.\tag{25}
$$

The system-level satisfaction is given by

$$
\Phi _ { \mathrm { s u m } } = \sum _ { t = 1 } ^ { T } \sum _ { k = 1 } ^ { K } \Phi _ { k } ( t ) .\tag{26}
$$

## E. Problem Formulation

To maximize the overall system utility while reducing energy consumption, we jointly optimize the task offloading decisions, UAV three-dimensional trajectories, and computational resource allocation. The user–UAV association is generated by a K-means-based user clustering mechanism and dynamically updated via periodic re-clustering according to user mobility, as inspired by [12].

$$
\begin{array} { r l } { \displaystyle \underset { \mathbf { W } , \mathbf { Q } , \mathbf { F } } { \operatorname* { m a x } } } & { \displaystyle \omega _ { 1 } \frac { R _ { \mathrm { s u m } } } { R _ { \mathrm { n o r m } } } + \omega _ { 2 } \frac { \Phi _ { \mathrm { s u m } } } { \Phi _ { \mathrm { n o r m } } } - \omega _ { 3 } \frac { E _ { \mathrm { t o t } } } { E _ { \mathrm { n o r m } } } } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { k = 1 } ^ { K } f _ { k , u } ^ { \mathrm { U A V } } ( t ) \leq f _ { u } ^ { \mathrm { m a x } } , \quad \forall u , t , } \\ & { ( 9 ) - ( 1 3 ) , ( 1 6 ) . } \end{array}\tag{27a}
$$

(27b)

Constraint (27a) ensures that the computational resources allocated to users do not exceed the maximum CPU capacity of each UAV. Constraints (9), (11), and (12) guarantee the UAV mobility feasibility, including the horizontal boundary limitation, maximum flight speed constraint, and minimum safety distance between UAVs, respectively. Constraint (10) restricts the UAV flight altitude. Constraint (13) ensures the correctness of the SIC decoding order in NOMA transmission. Constraint (16) describes the user–UAV association and task offloading decisions.

## III. HIERARCHICAL INTELLIGENT EVOLUTIONARYALGORITHM

Fig. 1 illustrates the architecture of the proposed HIEA framework. HIEA decomposes the optimization process into four coordinated functions, including intelligent decision, communication-aware initialization, adaptive evolution and precision refinement, to efficiently solve the mixed-integer non-convex optimization problem.

## A. Intelligent Decision Function

The intelligent decision function aims to reduce the highdimensional search complexity caused by the joint optimization of task offloading and UAV mobility. A conventional DQN can learn effective policies for sequential decision-making; however, directly applying DQN to the coupled offloading and mobility optimization results in a large mixed action space and severe decision interference. To address this issue, we propose a Multi-Head Deep Q-Network (MH-DQN) which decomposes heterogeneous decisions into multiple coordinated heads. Instead of directly solving the entire optimization problem, MH-DQN serves as a candidate generator to learn promising offloading and trajectory decisions and provide high-quality initial solutions for subsequent evolutionary optimization.

![](images/ac13aa2bb111b228c5179dccd5df465c45a9eb129a686040c2b1ab0c3d409a34.jpg)  
Fig. 1: Functional architecture of HIEA

1) State and action design: The state consists of normalized UAV/user locations, user–UAV association, task features $( c _ { k } , n _ { k } , m _ { k } , o _ { k } )$ and user identifiers. Different from conventional DQN with a single action output, MH-DQN adopts two decoupled decision heads: an offloading head with binary actions and a mobility head with seven movement actions. This design reduces action coupling and enables efficient learning of heterogeneous decisions.

2) Network and reward design: MH-DQN employs a shared feature extractor followed by two Q-value heads to capture the correlations and differences between offloading and mobility decisions. It is trained using experience replay and a target network. The reward is aligned with the global optimization objective:

$$
J = \omega _ { 1 } \tilde { R } _ { \mathrm { s u m } } + \omega _ { 2 } \tilde { \Phi } _ { \mathrm { s u m } } - \omega _ { 3 } \tilde { E } _ { \mathrm { t o t } } - \Pi ,\tag{28}
$$

where $\tilde { R } _ { \mathrm { s u m } } , \tilde { \Phi } _ { \mathrm { s u m } } .$ and $\tilde { E } _ { \mathrm { t o t } }$ denote the normalized throughput, priority-aware satisfaction, and energy consumption, respectively. The penalty term Π is used to handle constraint violations.

3) Candidate generation: After training, MH-DQN generates trajectory–offloading candidates over multiple time slots. These candidates are projected into feasible solutions and injected into the initial population, providing effective search guidance and reducing the optimization burden of the evolutionary process.

## B. Communication-Aware Initialization Function

The communication-aware initialization function aims to improve the quality of evolutionary populations by incorporating wireless channel conditions into UAV trajectory initialization. Different from conventional random or geometrybased initialization, we propose a CQI-guided load-aware trajectory initialization (CQI-LATI) mechanism, which jointly considers communication quality, task heterogeneity and UAV mobility constraints. The overall procedure is summarized in Algorithm 1.

1) Communication-aware service point selection: Instead of selecting UAV locations merely according to user distribution, CQI-LATI identifies communication-efficient service points by evaluating the achievable NOMA transmission performance. For each UAV-associated cluster, a candidate service-point set is generated around the user distribution center:

$$
\mathcal { S } _ { u } = \{ \mathbf { s } _ { u , 1 } , \mathbf { s } _ { u , 2 } , . . . , \mathbf { s } _ { u , M } \} ,\tag{29}
$$

where each candidate point is generated as ${ \bf s } _ { u , m } = ( \bar { x } _ { u } +$ $\Delta x _ { m } , \bar { y } _ { u } + \Delta y _ { m } , H _ { u } )$ . The CQI-guided service point is selected by maximizing the aggregate achievable rate:

Algorithm 1: CQI-Guided Load-Aware Trajectory Ini  
tialization (CQI-LATI)   
Require: User association A, task features   
$\{ ( c _ { k } , n _ { k } , m _ { k } , o _ { k } ) \}$ , population size $N _ { p }$   
Ensure: Initial population $\{ \mathcal { X } _ { i } \} _ { i = 1 } ^ { N _ { p } }$   
1: Generate candidate service points and select   
CQI-optimal locations:   
$\mathbf { q } _ { u } ^ { \mathrm { C Q I } } = \arg \operatorname* { m a x } _ { \mathbf { s } \in S _ { u } } R _ { u } ( \mathbf { s } )$   
2: Compute load-adaptive guidance factor $\alpha _ { u }$   
3: for $i = 1$ to $N _ { p }$ do   
4: Generate UAV trajectories using fused random and   
CQI guidance:   
$\Delta { \bf q } _ { u } ( t ) = ( 1 - \alpha _ { u } ) \Delta { \bf q } _ { \mathrm { r a n d } } + \alpha _ { u } \Delta { \bf q } _ { \mathrm { C Q I } }$   
5: Project trajectories onto UAV mobility constraints   
6: Initialize $\mathbf { W } _ { i }$ and $\mathbf { f } _ { i } ,$ and construct $\dot { \mathcal { X } _ { i } } = ( \mathbf { Q } _ { i } , \mathbf { W } _ { i } , \mathbf { f } _ { i } )$   
7: end for   
8: return $\{ \mathcal { X } _ { i } \} _ { i = 1 } ^ { N _ { p } }$

$$
\mathbf { q } _ { u } ^ { C Q I } = \arg \operatorname* { m a x } _ { \mathbf { s } _ { u , m } \in S _ { u } } \sum _ { k \in \mathcal { C } _ { u } ( t ) } R _ { k , u } ( \mathbf { s } _ { u , m } ) ,\tag{30}
$$

where $R _ { k , u } ( { \bf s } _ { u , m } )$ denotes the achievable transmission rate of user k at candidate location $\mathbf { s } _ { u , m }$ . This enables the initialization process to exploit channel-aware spatial information rather than only geographical proximity.

2) Load-aware trajectory guidance: To adapt the trajectory exploration to heterogeneous task requirements, CQI-LATI introduces a priority-aware load coefficient:

$$
\alpha _ { u } = \frac { \sum _ { k \in \mathcal { C } _ { u } ( t ) } ( 1 + \eta o _ { k } ) c _ { k } } { \operatorname* { m a x } _ { j } \sum _ { k \in \mathcal { C } _ { j } ( t ) } ( 1 + \eta o _ { k } ) c _ { k } } ,\tag{31}
$$

where $o _ { k }$ represents the task priority indicator. A larger $\alpha _ { u }$ indicates heavier workload, leading to stronger CQI-oriented guidance, while lightly loaded UAVs preserve exploration diversity. Therefore, CQI-LATI achieves an adaptive balance between communication-aware exploitation and population diversity.

3) Velocity-constrained individual construction: To avoid infeasible trajectories caused by direct position sampling,CQI-LATI adopts trajectory-increment encoding satisfying the UAV mobility constraint in (11).

After trajectory generation, the remaining decision variables, including task offloading decisions and computational resource allocation,are initialized by the corresponding modules. The user–UAV association is inherited from the K-meansbased user clustering results [12]. Finally, a feasible individual is obtained as

$$
\begin{array} { r } { \mathscr { X } _ { i } = ( \mathbf { Q } _ { i } , \mathbf { W } _ { i } , \mathbf { f } _ { i } ) . } \end{array}\tag{32}
$$

By integrating communication-aware spatial guidance and workload-adaptive exploration, CQI-LATI provides informa-

tive trajectory priors for HIEA,thereby improving the quality, diversity, and convergence efficiency of the initial population.

## C. Adaptive Evolution Function

After initialization, HIEA employs an adaptive evolution function to enhance global search capability and overcome premature convergence in large-scale mixed-integer optimization. Conventional genetic algorithms usually rely on random evolutionary operators, which may lose high-quality solutions or suffer from insufficient exploration in highly coupled UAV-MEC systems.

To address this issue, we develop an enhanced genetic algorithm (EGA) with elite-guided evolution, heterogeneous variable-aware genetic operations, diversity recovery and adaptive termination strategies. The proposed EGA performs global exploration while maintaining population quality and diversity throughout the evolutionary process.

1) Elite-guided evolutionary search: Each individual is encoded as

$$
\begin{array} { r } { \mathcal { X } = ( \mathbf { Q } , \mathbf { W } , \mathbf { f } ) , } \end{array}
$$

where Q, W and f denote UAV trajectories,task offloading decisions and computational resource allocation,respectively.

At each generation, elite individuals are preserved to retain high-quality search directions, while offspring are generated through hybrid crossover and mutation operations. Continuous variables, including UAV trajectories and computational resource allocation, adopt convex-combination crossover, whereas discrete offloading decisions use user-level inheritance.

2) Diversity-aware evolution control: To avoid premature convergence caused by excessive exploitation, EGA monitors population diversity and evolutionary stagnation. When insufficient diversity is detected, diversity recovery operations are activated to expand the search region. Meanwhile, an intelligent early stopping strategy avoids unnecessary iterations when further improvement becomes limited.

By combining elite exploitation and diversity preservation, the adaptive evolution function achieves an effective balance between global exploration and local exploitation.

## D. Precision Refinement Function

Although EGA provides strong global exploration capability, the obtained solutions may still suffer from limited local accuracy. Therefore, HIEA introduces a precision refinement function to further enhance elite solutions through an SCAbased local optimization mechanism.

Different from directly applying conventional SCA, the proposed refinement function is designed for the heterogeneous decision variables of UAV-MEC systems and integrates project-and-convexify smoothing, augmented-objective neighborhood search and adaptive refinement depth control.

1) Elite solution refinement: After each evolutionary generation, the best individual obtained by EGA is selected for refinement. The UAV trajectories, task offloading decisions and computational resource allocation are jointly refined through successive convex approximation.

2) Project-and-convexify optimization: To improve local convergence stability, a smoothing operation is introduced:

$$
\tilde { \mathbf { Q } } = \mathbf { Q } ^ { ( r ) } + \beta ( \mathbf { Q } ^ { ( r ) } - \mathbf { Q } ^ { ( r - 1 ) } ) ,
$$

and similar refinement is applied to other decision variables.

Meanwhile, an augmented objective function is adopted:

$$
\tilde { F } ( \mathcal { X } ) = F ( \mathcal { X } ) - \mu V ( \mathcal { X } ) ,
$$

where $V ( \mathcal X )$ represents constraint violations.

3) Accuracy-efficiency adaptive control: The refinement depth is controlled by $N _ { \mathrm { S C A } }$ . Increasing the refinement iterations can improve solution accuracy, but introduces additional computational overhead. Therefore, HIEA selects a moderate refinement depth to achieve a favorable trade-off between optimization accuracy and computational efficiency.

## IV. THEORETICAL ANALYSIS OF HIEA

HIEA decomposes the mixed-integer non-convex optimization problem into four coordinated functions: intelligent decision generation, communication-aware initialization, adaptive evolution and precision refinement. This section analyzes the key mechanisms of HIEA from three perspectives: search space reduction, initialization quality–diversity balance and accuracy–efficiency trade-off. The theoretical observations are further validated through mechanism-oriented simulations in Sec. V-A.

## A. Search Space Reduction Analysis of MH-DQN

The joint optimization of UAV trajectories, task offloading decisions, and computational resources leads to a large mixedinteger search space. Let the original feasible solution space be

$$
\Omega = \Omega _ { Q } \times \Omega _ { W } \times \Omega _ { f }\tag{33}
$$

where $\Omega _ { Q } , \Omega _ { W }$ and $\Omega _ { f }$ denote the feasible sets of UAV trajectories, task offloading decisions and computing resource allocation, respectively. Instead of directly searching the entire space, MH-DQN learns priority-aware decision policies and generates promising trajectory–offloading candidates. The candidate region is therefore constrained as

$$
\Omega _ { \mathrm { M H - D Q N } } \subseteq \Omega .\tag{34}
$$

Consequently, the evolutionary optimizer explores a more concentrated solution region rather than performing blind search over the whole space, which reduces inefficient exploration and improves convergence efficiency.

## B. CQI Guidance Strength and Initialization Quality– Diversity Trade-off

Although MH-DQN provides high-quality candidate solutions, evolutionary optimization still requires sufficient population diversity to avoid premature convergence. Therefore, CQI-LATI introduces communication-aware trajectory guidance to balance initialization quality and population diversity.

The trajectory increment is formulated as

$$
\Delta { \bf q } _ { u } ( t ) = ( 1 - \alpha _ { u } ) \Delta { \bf q } _ { \mathrm { r a n d } } + \alpha _ { u } \Delta { \bf q } _ { \mathrm { C Q I } } ,\tag{35}
$$

where $\alpha _ { u }$ denotes the CQI guidance intensity. For ease of analysis, $\alpha _ { u }$ is assumed to be a constant $\alpha .$

The initialization quality is measured by

$$
\overline { { \mathrm { I Q } } } ( \alpha ) = \frac { 1 } { N _ { p } T U } \sum _ { i , t , u } \frac { 1 } { 1 + \Vert \mathbf { q } _ { i , u } ( t , \alpha ) - \mathbf { q } _ { u } ^ { \mathrm { C Q I } } \Vert _ { 2 } } ,\tag{36}
$$

where $\overline { { \mathrm { I Q } } } ( \alpha )$ characterizes the proximity between initialized trajectories and communication-efficient service points.

Meanwhile, the population diversity is quantified as

$$
\operatorname { D i v } _ { Q } ( \mathcal { P } _ { \alpha } ) = \frac { 1 } { N _ { p } } \sum _ { i = 1 } ^ { N _ { p } } \left. \mathbf { Q } _ { i } ( \alpha ) - \bar { \mathbf { Q } } ( \alpha ) \right. _ { 2 } ,\tag{37}
$$

where $\mathbf { Q } _ { i } ( \alpha )$ denotes the trajectory component of the i-th initialized individual and $\bar { \mathbf { Q } } ( \alpha )$ represents the average trajectory of the population.

According to the trajectory update mechanism, increasing α enhances the contribution of CQI-oriented guidance while reducing random exploration. Therefore, UAV trajectories become closer to communication-efficient service points, leading to higher initialization quality. However, the reduced random perturbation decreases the dispersion among individuals and may reduce population diversity. This indicates an inherent quality–diversity trade-off introduced by communicationaware guidance.

To balance exploitation and exploration, CQI-LATI adopts a load-aware guidance coefficient $\alpha _ { u }$ , where heavily loaded clusters receive stronger communication guidance, while lightly loaded clusters maintain larger exploration capability. This adaptive strategy improves the initialization quality without sacrificing population diversity.

## C. Accuracy–Efficiency Trade-off of SCA Refinement

After global evolutionary exploration, the obtained solutions may still suffer from limited local precision. Therefore, HIEA introduces an SCA-based refinement procedure to improve elite solutions.

Let $F ^ { \mathrm { E G A } }$ denote the best fitness obtained by EGA, where the fitness represents the objective function value of the optimization problem. The initial fitness before SCA refinement is given by

$$
F ( 0 ) = F ^ { \mathrm { E G A } } ,\tag{38}
$$

and $F ( n )$ denotes the fitness after n SCA refinement iterations. With the improvement-based acceptance strategy, only solutions satisfying

$$
F ( n ) \geq F ( n - 1 )\tag{39}
$$

are retained. Therefore, the refinement process can continuously improve or maintain the elite solution quality.

However, additional SCA iterations introduce extra optimization and evaluation overhead. The total runtime can be approximated as

$$
\mathcal { T } _ { \mathrm { r u n } } = \mathcal { T } _ { \mathrm { E G A } } + G _ { \mathrm { E G A } } N _ { \mathrm { S C A } } \tau _ { \mathrm { S C A } } + \mathcal { T } _ { \mathrm { o t h } }\tag{40}
$$

TABLE I: Main Simulation Parameters
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Number of UAVs/users (default)</td><td> $\overline { { U = 4 , K = 6 0 } }$ </td></tr><tr><td>Service area</td><td> $\mathrm { 5 0 0 \times 5 0 0 m ^ { 2 } }$ </td></tr><tr><td>UAV altitude</td><td> $z _ { u } ( t ) \in [ 2 0 , 1 0 0 ] \mathrm { m }$ </td></tr><tr><td>Maximum UAV velocity</td><td> $V _ { \mathrm { m a x } } = 8 \mathrm { m } / \mathrm { s }$ </td></tr><tr><td>Carrier frequency</td><td> $f _ { c } = 5 . 8 \mathrm { G H z }$ </td></tr><tr><td>Bandwidth</td><td> $B = 2 0 \mathrm { M H z }$ </td></tr><tr><td>Noise power spectral density</td><td> $N _ { 0 } = - 1 7 4 \mathrm { d B m / H z }$ </td></tr><tr><td>CPU frequency</td><td> $f _ { k } ^ { \mathrm { l o c } } = 2 \mathrm { G H z } , \ f _ { u } = 1 0 \mathrm { G H z }$ </td></tr><tr><td>Task data size</td><td> $\ddot { n _ { k } } \in [ 0 . 5 , 5 ] \mathrm { M B }$ </td></tr><tr><td>CPU cycles per bit</td><td> $c _ { k } \in [ 5 0 0 , 2 0 0 0 ]$  cycles/bit</td></tr><tr><td>High/low priority ratio</td><td> $4 0 \% / \dot { 6 } 0 \%$ </td></tr></table>

where $G _ { \mathrm { E G A } }$ denotes the number of EGA generations, τ<sub>SCA</sub> denotes the cost of one SCA iteration, and $\tau _ { \mathrm { o t h } }$ represents other computational overheads.

Hence, increasing $N _ { \mathrm { S C A } }$ improves local refinement accuracy but increases computational cost. When the solution improvement becomes saturated, further refinement provides limited benefits, resulting in an accuracy–efficiency trade-off.

## D. Unified Design Implications

The above analysis reveals three fundamental mechanisms of HIEA:

$$
| \Omega _ { \mathrm { M H - D Q N } } | \ll | \Omega | , ( \mathrm { S e a r c h ~ S p a c e ~ R e d u c t i o n } )\tag{41}
$$

$$
\overline { { \mathrm { I Q } } }  \mathrm { D i v } ( \mathcal P ) , \quad ( \mathrm { I n i t i a l i z a t i o n ~ B a l a n c e } )\tag{42}
$$

$$
F  { \mathcal { T } } _ { \mathrm { { r u n } } } , \qquad ( { \mathrm { R e f i n e m e n t ~ E f f i c i e n c y } } )\tag{43}
$$

Therefore, HIEA does not rely on a single optimization strategy. Instead, it achieves hierarchical optimization by integrating policy-guided decision generation, communicationaware initialization, evolutionary global exploration and precision refinement.

## V. SIMULATION RESULTS AND DISCUSSION

This section evaluates HIEA in a UAV-assisted MEC network. The main simulation parameters are summarized in Table I. Each set of simulations is repeated with multiple random seeds.

Unless otherwise specified, the default network configuration is adopted, where 4 UAVs serve 60 mobile users. Users are associated with UAV clusters via a K-means-based user clustering method with periodic re-clustering and uplink NOMA is employed within each cluster. The proposed optimizer jointly determines UAV trajectories, task offloading decisions and computing resources under the considered throughput-energy objective.

For mechanism validation, only the corresponding factor, including the CQI guidance coefficient $\alpha ,$ SCA refinement depth $N _ { \mathrm { S C A } }$ , or initialization strategy, is varied while other system settings remain unchanged.

## A. Mechanism Validation

The theoretical analysis in Sec. IV reveals two key tradeoffs in HIEA, i.e., initialization quality–diversity balance and refinement accuracy–efficiency trade-off. This subsection validates these mechanisms through controlled simulations.

![](images/51c398686383c2ed1e856a77d9064fe22554e457b3b2af2783a302875fc63b9a.jpg)  
Fig. 2: CQI guidance coefficient α and the IQ–Div trade-off

![](images/a7cba166f8dc44832b7dc7ca9dcdcf2aaa6b89ab3b9f0ac633bb2fdceffa8b72.jpg)  
Fig. 3: Initialization quality–fitness trade-off vs. CQI guidance intensities

![](images/87c252c32aceb68150feea487da818e23839f28fb6f51ac49ef8a1258be25c7c.jpg)  
Fig. 4: Quality–diversity comparison of different initialization strategies

![](images/47e40446f6c9eaf877e8f3c6c91f09f470f879c7fd4871f6d0eac2184368b753.jpg)  
Fig. 5: SCA Accuracy-efficiency tradeoff under different refinement depths

![](images/f056da3297880ed44b6a237814fdf14f82f258da662f0fe46badd17174faeb3f.jpg)  
Fig. 6: Module ablation

![](images/5b4ff461fa6f761474b3858783d6768f93911948be218c0d0df10cf49b0f5f2d.jpg)  
Fig. 7: Task completion rate under different high-priority ratios

1) Validation of CQI-LATI Initialization: The proposed CQI-LATI balances communication guidance and population diversity through the adaptive coefficient α. As shown in Fig. 2, increasing α improves the initialization quality IQ but reduces trajectory diversity, confirming the inherent quality– diversity trade-off.

Moreover, Fig. 3 shows that excessively strong CQI guidance does not always improve the final optimization performance. Therefore, the load-aware guidance strategy achieves a better balance between communication exploitation and evolutionary exploration.

Fig. 4 further compares different initialization strategies. CQI-LATI achieves higher initialization quality than random initialization while preserving more diversity than DQN-based initialization, demonstrating its effectiveness as an internal function of HIEA.

2) Validation of SCA Refinement: Fig. 5 evaluates the impact of refinement depth $N _ { \mathrm { S C A } }$ . Increasing $N _ { \mathrm { S C A } }$ improves solution quality but also increases computational cost, with diminishing performance gains in later iterations. This verifies the accuracy–efficiency trade-off of SCA refinement.

Overall, the results confirm that HIEA benefits from the coordinated design of communication-aware initialization and precision refinement.

## B. Ablation Study

Fig. 6 evaluates the contribution of the MH-DQN, EGA and SCA modules. The complete HIEA framework consistently achieves the highest objective value, demonstrating the complementary benefits of the three components.

Fig. 7 evaluates the priority-aware satisfaction model under different high-priority task ratios. As the ratio increases, the task completion rate decreases due to resource competition. The proposed model consistently outperforms the scheme without priority awareness, demonstrating its effectiveness in protecting high-priority task services.

## C. Convergence Analysis

Fig. 8 shows the convergence performance under different numbers of UAVs. All curves increase rapidly in the early episodes and then stabilize, indicating fast convergence of the proposed algorithm. The objective function becomes stable after about 100 episodes, with only slight fluctuations.

It can also be observed that the performance decreases slightly as the number of UAVs increases, due to higher system complexity and resource coupling. However, all cases achieve stable convergence, demonstrating the robustness of the proposed method.

Moreover, the SCA-based refinement introduces limited overhead by only optimizing elite solutions, achieving a favorable solution-quality versus complexity trade-off.

![](images/92770b50c80010b5e311341b0a10f559ddfb606e20cd189fe5817dfa90dba201.jpg)  
Fig. 8: Convergence Performance of the Proposed Algorithm under Different Numbers of UAVs

![](images/0524f5b3ebe8c97d40c8886c47ceb8b140f2db6c6c4dc05bb9169323f3f09297.jpg)  
Fig. 9: System throughput versus the number of UAVs

![](images/e41196620c9c460c17b6311a6bf7598534370ea2f38bb8ea9aadc95b08075373.jpg)  
Fig. 10: Task completion ratio versus the number of UAVs

![](images/752f564fd2a09f18c52472a0ea8b78064f3a90d564852b460351deb739014c9f.jpg)  
Fig. 11: Total system energy consumption versus the number of UAVs

![](images/a4b1c6cf7269aa10e642ac4466b262d8232800a5b89354e211eb3abb923fb2b2.jpg)  
Fig. 12: System throughput versus the number of MUs

![](images/e7b36d42260ad5526c37917a5f482286280a2e2178736d7d227103165556f94f.jpg)  
Fig. 13: Task completion ratio versus the number of MUs

![](images/cbb157951b26ca2b26ad51e3a5ca7e62727d09b53b8cb2693e997126d9ffb910.jpg)  
Fig. 14: Total system energy consumption versus the number of MUs

## D. Baseline Comparison

In this section, HIEA is evaluated against several benchmark schemes, including AGSP [18], Genetic Algorithm (GA), ISAC [3] and PPO-L [19]. AGSP combines GA, PSO and SA to enhance optimization performance, while GA uses standard evolutionary operations. ISAC is based on the Soft Actor-Critic framework with a perturbation mechanism for improved exploration. PPO-L integrates Louvain clustering for user grouping, a load penalty for balancing resources and Proximal Policy Optimization for task scheduling.

The performance is evaluated using three metrics: system throughput (Mbps), task completion ratio (%) and total energy consumption (J).

Fig. 9 and Fig. 12 show the system throughput under different numbers of UAVs and MUs, respectively. As the number of MUs increases, all algorithms improve and gradually saturate due to limited resources. Increasing the number of UAVs further improves throughput, while the gain gradually diminishes. HIEA achieves the highest throughput in both scenarios, improving the throughput over ISAC by up to 5.1% and 6.5%, respectively.

Fig. 10 and Fig. 13 compare the task completion ratio under scenarios of varying numbers of UAVs and MUs, respectively. Increasing the number of UAVs improves the completion ratio by providing additional communication and computing resources, while increasing the number of MUs causes performance degradation due to resource competition.

HIEA achieves the highest completion ratio in both scenarios, improving the task completion ratio over ISAC by up to 13.6% and 6.9% in the UAV and MU scenarios, respectively.

Lastly, we compare the total system energy consumption. Fig. 11 and Fig. 14 illustrate the total system energy consumption under different numbers of UAVs and MUs, respectively. The energy consumption increases with the numbers of UAVs and MUs due to additional communication, computation, and operating overhead. HIEA achieves the lowest energy consumption among all methods, reducing energy consumption over ISAC by up to 15.3% in the UAV scenario and up to 10.7% in the MU scenario.

## VI. CONCLUSION

This paper investigated the priority-aware multi-UAV cooperative NOMA-MEC optimization problem and proposed a Hierarchical Intelligent Evolutionary Algorithm (HIEA) to improve optimization efficiency through intelligent decision guidance, communication-aware initialization, evolutionary search and refinement. Through theoretical analysis, we characterized HIEA’s search-space reduction, the quality-diversity trade-off of CQI-guided initialization, and the accuracy-efficiency tradeoff of SCA refinement. Simulation results then verified that HIEA outperforms benchmark methods in terms of throughput, task completion rate and energy efficiency. Future work will focus on developing dedicated optimization algorithms for high-priority UAVs to further improve emergency task services in dynamic multi-UAV networks.

## REFERENCES

[1] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, “A survey on mobile edge computing: The communication perspective,” IEEE Communications Surveys & Tutorials, vol. 19, no. 4, pp. 2322–2358, 2017.

[2] Y. Zeng, R. Zhang, and T. J. Lim, “Wireless communications with unmanned aerial vehicles: Opportunities and challenges,” IEEE Communications magazine, vol. 54, no. 5, pp. 36–42, 2016.

[3] P. Chen, L. Luo, D. Guo, J. Wu, K. Chi, C. Yan, and X. Dong, “QoS-oriented task offloading in NOMA-based Multi-UAV cooperative MEC systems,” IEEE Transactions on Wireless Communications, vol. 25, pp. 1965–1980, 2026.

[4] S. R. Islam, M. Zeng, O. A. Dobre, and K.-S. Kwak, “Resource allocation for downlink NOMA systems: Key techniques and open issues,” IEEE Wireless Communications, vol. 25, no. 2, pp. 40–47, 2018.

[5] H. Hao, C. Xu, W. Zhang, S. Yang, and G.-M. Muntean, “Joint task offloading, resource allocation, and trajectory design for multi-UAV cooperative edge computing with task priority,” IEEE Transactions on Mobile Computing, vol. 23, no. 9, pp. 8649–8663, 2024.

[6] F. Pervez, A. Sultana, C. Yang, and L. Zhao, “Energy and latency efficient joint communication and computation optimization in a multi-UAV-assisted MEC network,” IEEE Transactions on Wireless Communications, vol. 23, no. 3, pp. 1728–1741, 2023.

[7] M. Zhao, R. Zhang, Z. He, and K. Li, “Joint optimization of trajectory, offloading, caching, and migration for uav-assisted mec,” IEEE Transactions on Mobile Computing, vol. 24, no. 3, pp. 1981–1998, 2025.

[8] J. Zhu, X. Wang, H. Huang, S. Cheng, and M. Wu, “A NSGA-II algorithm for task scheduling in UAV-enabled MEC system,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 7, pp. 9414–9429, 2021.

[9] Z. Kuang, H. Wang, J. Li, and F. Hou, “Utility-aware UAV deployment and task offloading in multi-UAV edge computing networks,” IEEE Internet of Things Journal, vol. 11, no. 8, pp. 14 755–14 770, 2023.

[10] Z. Ning, Y. Yang, X. Wang, L. Guo, X. Gao, S. Guo, and G. Wang, “Dynamic computation offloading and server deployment for uav-enabled multi-access edge computing,” IEEE Transactions on Mobile Computing, vol. 22, no. 5, pp. 2628– 2644, 2021.

[11] W. Zhang, Q. Wang, X. Liu, Y. Liu, and Y. Chen, “Threedimension trajectory design for multi-UAV wireless network with deep reinforcement learning,” IEEE Transactions on Vehicular Technology, vol. 70, no. 1, pp. 600–612, 2020.

[12] R. Zhong, X. Liu, Y. Liu, and Y. Chen, “Multi-agent reinforcement learning in NOMA-aided UAV networks for cellular offloading,” IEEE Transactions on Wireless Communications, vol. 21, no. 3, pp. 1498–1512, 2021.

[13] H. Huang, Z.-Y. Chai, B.-S. Sun, H.-S. Kang, and Y.-J. Zhao, “Multiobjective deep reinforcement learning for computation offloading and trajectory control in UAV-base-station-assisted MEC,” IEEE Internet of Things Journal, vol. 11, no. 19, pp. 31 805–31 821, 2024.

[14] G. Chen, Y. Guo, M. Jiang, S. Yang, X. Zhao, and D. Gong, “A subspace-knowledge transfer based dynamic constrained multiobjective evolutionary algorithm,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 8, no. 2, pp. 1500–1512, 2023.

[15] A. Tharwat and W. Schenck, “Population initialization techniques for evolutionary algorithms for single-objective con-

strained optimization problems: Deterministic vs. stochastic techniques,” Swarm and Evolutionary Computation, vol. 67, p. 100952, 2021.

[16] S. Elsayed, R. Sarker, and C. A. C. Coello, “Sequence-based deterministic initialization for evolutionary algorithms,” IEEE Transactions on Cybernetics, vol. 47, no. 9, pp. 2911–2923, 2016.

[17] J. Lengler, A. Opris, and D. Sudholt, “Analysing equilibrium states for population diversity,” in Proceedings of the Genetic and Evolutionary Computation Conference, Lisbon, Portugal, 15-19 July 2023, pp. 1628–1636.

[18] H. Yuan, M. Wang, J. Bi, S. Shi, J. Yang, J. Zhang, M. Zhou, and R. Buyya, “Cost-efficient task offloading in mobile edge computing with layered unmanned aerial vehicles,” IEEE Internet of Things Journal, vol. 11, no. 19, pp. 30 496–30 509, 2024.

[19] J. He, Z. Chang, and L. Wang, “Joint Optimization of UAV Deployment and Task Scheduling for Multi-UAV-assisted MEC System,” in Proceedings of the IEEE Global Communications Conference, Taipei, Taiwan, 8–12 December 2025, pp. 3152– 3157.

[20] J. Tian, D. Wang, H. Zhang, and D. Wu, “Service satisfactionoriented task offloading and UAV scheduling in UAV-enabled MEC networks,” IEEE Transactions on Wireless Communications, vol. 22, no. 12, pp. 8949–8964, 2023.

[21] Y. Zhang, Z. Kuang, Y. Feng, and F. Hou, “Task offloading and trajectory optimization for secure communications in dynamic user multi-UAV MEC systems,” IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 14 427–14 440, 2024.

[22] X. He, R. Jin, and H. Dai, “Joint service placement and resource allocation for multi-UAV collaborative edge computing,” in Proceedings ofthe IEEE Wireless Communications and Networking Conference (WCNC), Nanjing, China, 29 March - 1 April 2021, pp. 1–6.

[23] P. Qin, Y. Fu, Y. Xie, K. Wu, X. Zhang, and X. Zhao, “Multi-agent learning-based optimal task offloading and UAV trajectory planning for AGIN-power IoT,” IEEE Transactions on Communications, vol. 71, no. 7, pp. 4005–4017, 2023.

[24] Z. Ma, S. Zhang, Z. Chen, T. Han, Z. Qian, M. Xiao, N. Chen, J. Wu, and S. Lu, “Towards revenue-driven multi-user online task offloading in edge computing,” IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 5, pp. 1185–1198, 2021.

[25] W. Wu, C. Zhou, M. Li, H. Wu, H. Zhou, N. Zhang, X. S. Shen, and W. Zhuang, “AI-native network slicing for 6G networks,” IEEE Wireless Communications, vol. 29, no. 1, pp. 96–103, 2022.

[26] P. Qin, Y. Fu, J. Zhang, S. Geng, J. Liu, and X. Zhao, “DRLbased resource allocation and trajectory planning for NOMAenabled multi-UAV collaborative caching 6G network,” IEEE Transactions on Vehicular Technology, vol. 73, no. 6, pp. 8750– 8764, 2024.

[27] M. Zhang, Z. Su, Q. Xu, Y. Qi, and D. Fang, “Energy-efficient task offloading in UAV-RIS-assisted mobile edge computing with NOMA,” in Proceedings of the IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), Vancouver, Canada, 20–23 May 2024, pp. 01–06.

[28] Z. Ding and H. V. Poor, “Joint beam management and power allocation in THz-NOMA networks,” IEEE Transactions on Communications, vol. 71, no. 4, pp. 2059–2073, 2023.

[29] J. Gubbi, R. Buyya, S. Marusic, and M. Palaniswami, “Internet of Things (IoT): A vision, architectural elements, and future directions,” Future Generation Computer Systems, vol. 29, no. 7, pp. 1645–1660, 2013.