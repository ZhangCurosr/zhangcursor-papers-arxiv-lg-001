# Distributed Linear Bandits for Freshness-Aware Preprocessing in MEC-Enabled Digital Twin Networks

Mariam Yahya and Setareh Maghsudi

Faculty of Electrical Engineering and Information Technology, Ruhr-University Bochum, Germany

Abstract—Digital twins (DTs) require timely updates from Internet of Things (IoT) devices to accurately represent the state of the corresponding physical systems, but transmitting large data volumes can compromise freshness. Moreover, different task classes have distinct computational demands and freshness requirements, while stochastic task generation and sizes complicate resource allocation. In this paper, we propose a freshnessaware framework in which multi-access edge computing (MEC) servers preprocess IoT data, reducing transmission volume and DT-server processing load. The framework establishes mutual support between DT and MEC servers. The DTs determine feasible CPU-allocation strategies guaranteeing class-specific ageof-information (AoI) requirements under specified workload, delay, and update-generation bounds. MEC servers collaboratively identify the feasible strategy minimizing expected freshnessnormalized preprocessing delay. We formulate CPU-strategy selection as a fixed-confidence distributed best-arm identification problem in linear bandits, with MEC servers as agents and CPU allocations across task classes as arms. We employ DistLinGapE, a distributed collaborative learning method, to coordinate exploration and share observations. Numerical results show that the method identifies the optimal feasible strategy and reduces preprocessing delay when deployed under stationary task-class assignments and workload statistics. Collaboration yields an identification speedup nearly proportional to the number of participating MEC servers in the evaluated settings.

Index Terms—Multi-access edge computing (MEC), Age-of-Information (AoI), digital twins (DT), linear bandits, distributed learning.

## I. INTRODUCTION

The rapid growth of Internet of Things (IoT) networks has accelerated the use of digital twins (DTs), which are virtual representations of physical systems, in computation-intensive applications such as vehicular networks [1], industrial automation [2], and smart healthcare [3]. Through continuous synchronization with physical systems, DTs can support system monitoring, optimization, and autonomous operation [1]–[6]. Maintaining an accurate DT, however, requires sensing devices to provide timely data that reflects the current state of the physical system.

A common approach updates the virtual DT representations and uses them to guide operational decisions in the physical system, such as resource allocation in edge networks [2], [7]. However, updating the DT itself requires communication and computational resources. Transmitting large volumes of sensing data to the DT server can cause substantial communication overhead and processing delays [5], [8]. The resulting synchronization delays may cause the DT to rely on outdated information, reducing its usefulness in real-time applications. Moreover, depending on the application, updating the DT may require only a subset of the sensing data or information extracted from it [5], [8]–[10].

Preprocessing sensing data before updating the DT can help address this challenge. Techniques such as data compression [9], feature extraction [10], and semantic communication [5], [8] can reduce the volume of forwarded data and, depending on the application, the processing load at the DT server. These benefits motivate the use of multi-access edge computing (MEC) servers to preprocess data closer to IoT devices.

IoT devices or clusters can transmit sensing tasks from multiple classes to MEC servers for preprocessing. These classes may differ in their preprocessing needs, computational demands, and freshness requirements [11], [12]. Age of information (AoI) provides a measure of the freshness of the information represented by a DT and can be used to specify class-specific freshness requirements.

Allocating computational resources for preprocessing is challenging when task generation and task sizes are uncertain [7], [13], as the allocation must accommodate different computational demands while keeping DTs sufficiently up to date. Uniform or random CPU allocation across task classes may be inefficient because it does not account for differences in workloads and freshness requirements. This motivates learning approaches that use observed preprocessing delays to identify effective CPU allocations.

## A. Paper Contributions

We propose a framework that establishes mutual support between the DT and the MEC servers. The MEC servers preprocess sensing data before forwarding the results to the DT server, while the DT guides their CPU-allocation decisions through a two-stage process. First, the DT determines a feasible set of CPU-allocation strategies that guaranty class-specific AoI requirements under specified workload, delay, and updategeneration bounds. Second, the MEC servers collaboratively identify the strategy within this set that minimizes the expected freshness-normalized preprocessing delay.

We formulate CPU-allocation strategy selection as a fixedconfidence distributed best-arm identification (BAI) problem in linear bandits. MEC servers solving a shared allocation problem combine their observations to identify the best strategy with a predefined confidence using as few observations as possible. The linear structure allows observations from one strategy to inform the evaluation of others. We employ the distributed linear gap-based exploration (DistLinGapE) algorithm [14] to perform this collaborative identification.

The main contributions of this work are summarized as follows:

• We develop a freshness-aware MEC preprocessing framework in which the DTs determine feasible CPU-allocation strategies that satisfy class-specific AoI requirements.

• We formulate CPU-allocation strategy selection as a fixed-confidence distributed linear BAI problem, aiming to identify the feasible strategy that minimizes the expected freshness-normalized preprocessing delay under unknown mean workloads and task generation rates.

• We employ DistLinGapE to enable MEC servers to collaboratively identify the best feasible CPU allocation.

• The numerical results show that the DistLinGapE is effective in finding the best CPU allocation strategy with a predetermined confidence. Additionally, we show that collaboration among the MEC-servers speedup the learning process by reducing the number of learning rounds required per server.

## B. Paper Organization

This paper is organized as follows. Section II reviews the related work on DTs, AoI, and bandit learning. Section III introduces the system model, and Section IV presents the DT freshness requirements and strategy-update mechanism. Section V formulates the optimization problem, while Section VI develops its linear-bandit BAI representation. Section VII presents the solution approach using DistLinGapE. Numerical results are presented in Section VIII, and Section IX concludes the paper.

## II. BACKGROUND AND RELATED WORK

## A. Digital Twins and Age of Information

Research on DT-enabled networks considers two different but complementary directions: the first uses DTs to support physical-system monitoring, optimization, or operation, while the second focuses on maintaining timely DT representations through methods such as resource allocation, DT placement and migration, and update scheduling. These directions are closely connected, as the effectiveness of DT-assisted decisions depends on the freshness of the information available to the DT. Our work connects them through mutual support between the DT and MEC servers.

Our work considers fixed DT placement and focuses on allocating computational resources to reduce preprocessing delay while satisfying freshness requirements. Another research direction investigates DT placement and migration to determine where DTs should be hosted and when their locations should change to improve synchronization [6], [15], [16]. Dynamic placement adjusts DT locations to changing network conditions to improve synchronization efficiency [15]. For mobile physical systems, Noroozi et al. [6] formulate DT migration as a Markovian stopping problem and use dynamic programming to reduce the AoI observed when an application requests data. Considering both freshness and energy consumption, Guo et al. [16] jointly optimize edge association, transmission power, and DT deployment through a lightweight online algorithm.

Collaboration in DT-enabled networks has been studied both for providing shared services and for coordinating resource allocation. Although MEC servers and DTs may operate independently, applications involving multiple physical objects can benefit from exchanging information and coordination. For example, Li et al. [17] consider DT network slicing, where worker DTs process data from physical objects and a master DT aggregates their outputs to support a shared service. They develop approximation and online algorithms to maximize service utility based on AoI-dependent model quality, subject to delay and resource constraints. Collaboration can also support privacy preservation. Heydari et al. [18] propose a federated optimization framework in which a scheduler coordinates with DTs to minimize transmission time or maximize the amount of uploaded data under energy constraints, without revealing the DTs’ private information. In our framework, the DTs of the IoT clusters reside on a common DT server, which may be a MEC server or a separate server. This enables coordinated specification of class-specific freshness thresholds and workload and delay bounds. Collaboration takes place among the MEC servers, which share observations to accelerate the identification of an effective preprocessing CPU allocation.

A major role of the DT is to assist in resource allocation by using virtual representations to make decisions in the physical network. In our work, the DT does not directly select the resource allocation. Instead, it determines a set of feasible allocations based on task freshness requirements and specified workload and delay bounds. The MEC servers then collaboratively identify the best allocation within this set. Other studies use DTs to optimize scheduling and resource-allocation decisions directly. Bozkaya et al. [2] propose a genetic algorithmbased scheduler to minimize AoI and task completion time across industrial IoT devices and MEC servers. Huynh et al. [19] jointly optimize bandwidth, transmission power, taskoffloading fractions, and the processing rates of user devices and edge servers. They develop a centralized solution based on successive convex approximation, followed by a distributed solution.

Beyond resource allocation, some studies also optimize how information is processed for DT synchronization [5], [8]–[10]. For example, Li et al. [5] jointly optimize semantic extraction, transmission power, and computational resources using a soft actor-critic algorithm to minimize average synchronization latency under deadline, energy, and resource constraints. Other studies use data compression [9] or semantic feature extraction [10]. Our work considers MEC-side preprocessing for data-volume reduction without restricting the framework to a specific application or preprocessing technique. For given preprocessing operations, we allocate CPU resources across task classes with heterogeneous computational demands and freshness requirements.

Uncertainty in operating conditions motivates learningbased resource allocation when accurate system or workload information is unavailable [20]–[23]. Wirth et al. [20] use a risk-aware bandit algorithm for radio-access and DT-placement selection under changing conditions. Ai et al. [23] use a multiarmed bandit strategy to allocate computing resources between model retraining and inference services in DT-assisted MEC. In our setting, the preprocessing workloads are stochastic with unknown distributions. MEC servers therefore learn allocation performance from observed delays and share observations to accelerate identification within a freshness-feasible set. This motivates the bandit learning approach discussed next.

## B. Bandit Learning and Best-Arm Identification

Bandit learning is a sequential decision-making framework in which a learner, called an agent, selects actions, known as arms, with unknown reward distributions and updates its beliefs from observed rewards to guide subsequent selections. One objective is to minimize cumulative regret, which is the difference between the expected cumulative reward from consistently selecting an optimal arm and that from the selected arms [24], [25]. Alternatively, BAI focuses on recommending the best arm after an exploration phase [26], [27]. In the fixedconfidence setting, the objective is to identify an optimal arm with a predefined confidence, using as few observations as possible. The latter approach suits our problem, where CPU allocations are explored until the best feasible allocation is identified with the required confidence and then deployed until the network configuration changes.

Linear bandits have a shared structure across arms, it expresses each arm’s expected reward as the inner product of a known feature vector and an unknown parameter vector. This structure allows observations from one arm to inform estimates of other arms, potentially reducing the exploration required for identification. Xu et al. [27] propose the linear gap-based exploration (LinGapE) algorithm for fixed-confidence linear BAI. The algorithm is adaptive, meaning that it uses reward observations and confidence bounds to select the next best arm to eventually identify the best arm in the least number of rounds, a more efficient algorithm compared to static approaches [28]. The DistLinGapE [14] extends this approach to a distributed setting in which multiple agents collaborate by sharing learning updates. Each agent updates its model estimates using local data, and then exchanges the updates with other agents in each global round creating a tradeoff between communication and local computation.

In our problem, uncertainty in task generation and task sizes makes preprocessing delays unknown. We model the CPU-strategy selection as a fixed-confidence linear BAI problem, where each CPU allocation in the set of freshnessfeasible allocations determined by the DTs represents an arm. Each MEC server acts as an agent that selects and deploys an allocation and observes the resulting delay. We employ DistLinGapE to coordinate learning among MEC servers.

Servers observe local delays under the selected strategies and exchange learning updates. Under suitable conditions, sharing observations reduces learning rounds per agent compared with independent learning.

## III. SYSTEM MODEL

We first introduce the notation used in this work.

Notation: We use boldface lowercase and uppercase letters to represent vectors and matrices, respectively. For a positive integer K, [K] denotes the set $\{ 1 , \ldots , K \}$ . Furthermore, $\left\| \mathbf { x } \right\| _ { p }$ denotes the p-norm of a vector x. For a symmetric positivedefinite matrix A, the weighted 2-norm is defined as $\| \pmb { x } \| _ { A } =$ $\sqrt { \mathbfit { x } ^ { \top } }$ Ax. We use P[·] to denote the probability of the event inside the brackets.

We consider an IoT network consisting of a set $\mathcal { C }$ of C clusters, indexed by $c \in [ C ]$ , and a set M of M MEC servers, indexed by $m \in [ M ]$ . Each IoT cluster has a corresponding DT that virtually represents the physical system and is updated as new information becomes available. The DTs of all clusters are maintained at a common server, referred to as the DT server, which may be one of the MEC servers or a separate server with resources dedicated to DT operation. The system model is illustrated in Fig. 1.

![](images/0589e4b718bb566b16b0f471fc71406d6702663cebdbae58b47cc07ff1095f6d.jpg)  
Fig. 1: The system model.

## A. Cluster Association and Homogeneous MEC Servers

Each cluster is associated with one MEC server, and we assume that this association is based on proximity. Let ${ \mathcal { C } } _ { m }$ denote the set of clusters associated with server $m \in [ M ]$ . We consider applications where clusters generate large volumes of data, but only a subset of this data or information extracted from it is needed to update their DTs. Cluster data is therefore transmitted to the associated MEC server for preprocessing, and the results are forwarded to the DT server.

Tasks are grouped into classes according to their preprocessing operation, input-data type, and freshness requirement. Operations may include compression, feature extraction, or semantic extraction, among others. Their parameters depend on the application and are not optimized in this work. Different classes may have different input sizes and computational demands, and each MEC server may receive a random number of tasks from these classes in each time slot. Freshness is measured using the AoI, which is tracked separately for each cluster and associated task class, as detailed in Section IV.

The MEC servers allocate CPU resources to preprocessing tasks using a feasible set of strategies determined by the DT server. This set accounts for class-specific AoI requirements and the prescribed workload, delay, and update-generation bounds, as detailed in the next section.

A configuration, indexed by n, is defined by its task classes, AoI requirements, and the statistical characteristics of task generation and workloads. Let $J _ { n }$ and $K _ { n }$ denote the numbers of task classes and feasible CPU-allocation strategies, respectively, and let $\epsilon _ { j } ^ { ( n ) }$ denote the AoI requirement of class $j .$ These quantities and the cluster associations remain fixed within a configuration. Task generation and workload statistics also remain stationary, although realized workloads vary and their distributions are unknown. A change in the defining characteristics initiates a new CPU-allocation problem and learning process. Here, we describe the system under a fixed configuration and omit its index when clear from the context, using $j \in [ J ]$ and $k \in [ K ]$ to index task classes and feasible strategies, respectively.

Within each configuration, we assume that all MEC servers have the same computing capacity $F > 0$ , measured in CPU cycles/s, dedicated to task preprocessing. The average total workload of each task class is also assumed to be the same across MEC servers, although realized workloads may differ across servers and time slots. These homogeneity assumptions allow the servers to collaboratively identify a common optimal CPU-allocation strategy, as described in Section V. With heterogeneous computing capacities or mean workloads, each MEC server could instead apply the framework independently, using its own feasible allocation set and local observations to identify its optimal CPU allocation.

## B. Task Generation and Computational Workload

We consider time slots of duration $T > 0$ to coordinate task generation and CPU allocation. Slots are indexed by $r =$ $0 , 1 , \ldots \mathrm { A t }$ the beginning of slot $r ,$ at time $t _ { r } ^ { \mathrm { G } } = r T$ , each cluster may generate tasks from the predefined task classes and offloads them to its associated MEC server. Each task q generated in slot r contains data collected at its generation time, $t _ { q } ^ { \mathrm { G } } = t _ { r } ^ { \mathrm { G } }$ . This timestamp is used to evaluate the freshness of the information incorporated into the DT.

Let $\mathcal { Q } _ { m , j , r }$ be the set of tasks of class $j$ generated in slot $r$ and offloaded to MEC server m. Each task $q \in \mathcal { Q } _ { m , j , r }$ has an input size of $\gamma _ { q }$ bits and requires $W _ { q }$ CPU cycles for preprocessing, where $W _ { q }$ is a random variable with an unknown distribution. Its output size is $\nu _ { j } \gamma _ { q }$ bits, where $\nu _ { j } ~ \in ~ ( 0 , 1 ]$ is an application-defined reduction factor that remains fixed for each class within a configuration.

The total input size and computational load of class $j$ at server m in slot r are

$$
\Gamma _ { m , j , r } = \sum _ { q \in \mathcal { Q } _ { m , j , r } } \gamma _ { q } , \qquad W _ { m , j , r } = \sum _ { q \in \mathcal { Q } _ { m , j , r } } W _ { q } .\tag{1}
$$

Both quantities are zero when ${ \mathcal { Q } } _ { m , j , r } = \phi .$ The generation rate and size of the computational workloads are random, and their underlying distributions are unknown. However, we assume known bounds on the total workload and input size per class, server, and time slot:

$$
0 \leq W _ { m , j , r } \leq \overline { { W } } _ { j } , \qquad 0 \leq \Gamma _ { m , j , r } \leq \overline { { \Gamma } } _ { j } .\tag{2}
$$

These bounds may follow, for example, from limits on the number of generated tasks, their input sizes, and the computational requirements of the tasks.

The CPU-allocation strategy determines the processing rate assigned to each task class and thereby the tasks’ execution times.

## C. CPU-Allocation Strategies

The MEC servers process the tasks using one of a finite set of K feasible CPU-allocation strategies given by $\{ f ^ { ( 1 ) } , \ldots , f ^ { ( K ) } \}$ . Strategy $k \in [ K ]$ specifies

$$
\pmb { f } ^ { ( k ) } = [ f _ { 1 } ^ { ( k ) } , \dots , f _ { J } ^ { ( k ) } ] ^ { \top } , \quad f _ { j } ^ { ( k ) } > 0 , \quad \sum _ { j = 1 } ^ { J } f _ { j } ^ { ( k ) } \leq F .\tag{3}
$$

Here, $f _ { j } ^ { ( k ) }$ denotes the share of the server’s computing capacity allocated to class $j ,$ measured in Hz.

We assume that tasks within a class are processed sequentially, while different classes are processed concurrently. The allocated CPU rates remain unchanged regardless of which classes are present. Unused reservations are not redistributed to other classes during the slot. Each server selects its strategy at the start of a slot, before observing the new task collection or workloads. The allocation remains fixed throughout the slot and may change only at the next slot boundary.

Motivated by resource-management architectures that separate performance monitoring from allocation decisions [21], [22], we consider a monitoring component at each MEC server that reports the total processing delay resulting from the CPUstrategy selection, without access to each class’s execution time or computing capacity.

## D. Communication and Processing Delays

For a task $q \in \mathcal { Q } _ { m , j , r }$ generated by cluster $c ,$ the uplink transmission rate to MEC server m is given by

$$
R _ { q } ^ { \mathrm { U L } } ( t _ { q } ^ { \mathrm { G } } ) = \mathcal { B } _ { q , m } ^ { \mathrm { U L } } \log _ { 2 } \left( 1 + \frac { P _ { q } H _ { c , m } ( t _ { q } ^ { \mathrm { G } } ) } { I _ { c } ( t _ { q } ^ { \mathrm { G } } ) + I _ { \mathrm { N } } ^ { 2 } } \right) ,\tag{4}
$$

where $\mathcal { B } _ { q , m } ^ { \mathrm { U L } }$ is the allocated bandwidth, $P _ { q }$ is the transmission power, $H _ { c , m }$ is the channel power gain, and $I _ { c }$ is the interference power affecting the cluster $c \mathbf { \hat { s } }$ transmission to its associated MEC receiver. The quantity $I _ { \mathrm { N } } ^ { 2 }$ is the noise power over the allocated bandwidth.

The uplink delay is $d _ { q } ^ { \mathrm { U L } } = \mathrm { w } _ { q } ^ { \mathrm { U L } } + \gamma _ { q } / R _ { q } ^ { \mathrm { U L } }$ , where $\mathrm { w } _ { q } ^ { \mathrm { U L } } \geq 0$ is the waiting time for channel access and is zero when transmission can begin immediately. We assume that the uplink delay is bounded $d _ { q } ^ { \mathrm { U L } } \leq b _ { j } ^ { \mathrm { U L } }$ for every task $q$ of class $j .$ We assume a fixed upper bound $u < T$ on the class uplink delay bounds:

$$
\operatorname* { m a x } _ { j \in [ J ] } b _ { j } ^ { \mathrm { U L } } \leq u < T .\tag{5}
$$

Tasks of class $j$ at MEC server m are processed in their order of arrival. Let $q _ { \mathrm { o l d } }$ denote the task immediately preceding task $q .$ A task starts processing once it has arrived and the preceding task of the same class has completed processing. Assuming no unfinished workload carries over from the previous slot, the first task of each class starts processing upon arrival. The processing start and completion times of task $q$ are given by

$$
\begin{array} { l } { { \displaystyle t _ { q } ^ { \mathrm { S t a r t } , ( k ) } = \operatorname* { m a x } \left\{ t _ { q } ^ { \mathrm { A r r } } , t _ { q _ { \mathrm { o l d } } } ^ { \mathrm { O u t } , ( k ) } \right\} , } } \\ { { \displaystyle t _ { q } ^ { \mathrm { O u t } , ( k ) } = t _ { q } ^ { \mathrm { S t a r t } , ( k ) } + \frac { W _ { q } } { f _ { j } ^ { ( k ) } } . } } \end{array}\tag{6}
$$

Here, $t _ { q } ^ { \mathrm { A r r } }$ is the arrival time at the MEC server, $W _ { q }$ is the required computational workload in CPU cycles, and $f _ { j } ^ { ( k ) }$ is the processing rate allocated to class $j$ under strategy k.

Under strategy $k ,$ the execution time of task $q$ is

$$
d _ { q } ^ { \mathrm { p r o c } , ( k ) } = \frac { W _ { q } } { f _ { j } ^ { ( k ) } } .\tag{7}
$$

The total execution time of all class-j tasks at server m in slot $r$ is

$$
d _ { m , j , r } ^ { \mathrm { e x e c } , ( k ) } = \sum _ { q \in \mathcal { Q } _ { m , j , r } } d _ { q } ^ { \mathrm { p r o c } , ( k ) } = \frac { W _ { m , j , r } } { f _ { j } ^ { ( k ) } } .\tag{8}
$$

This execution time is zero when no class-j tasks arrive at server m in slot $^ { r } \cdot$ Individual processing start and completion times follow (6).

Let $d _ { q } ^ { \mathrm { D T } , ( k ) }$ be the delay from completion of preprocessing until the result updates the DT. We assume that

$$
d _ { q } ^ { \mathrm { D T } , ( k ) } \leq b _ { j } ^ { \mathrm { D T } } ,\tag{9}
$$

for all class-j tasks and candidate allocation strategies $k .$ This assumes sufficient forwarding and DT-server capacity to ensure the bound under the modeled traffic conditions. The bound includes the transmission delay $\nu _ { j } \gamma _ { q } / R _ { q } ^ { \mathrm { D T } }$ , where $R _ { q } ^ { \mathrm { D T } }$ is the forwarding rate, and any additional scheduling delays.

The task’s DT update time and end-to-end delay are

$$
\begin{array} { r l r } {  { t _ { q } ^ { \mathrm { D T } , ( k ) } = t _ { q } ^ { \mathrm { O u t } , ( k ) } + d _ { q } ^ { \mathrm { D T } , ( k ) } , } } \\ & { } & { D _ { q } ^ { ( k ) } = t _ { q } ^ { \mathrm { D T } , ( k ) } - t _ { r } ^ { \mathrm { G } } } \\ & { } & { = d _ { q } ^ { \mathrm { U L } } + ( t _ { q } ^ { \mathrm { S t a r t } , ( k ) } - t _ { q } ^ { \mathrm { A r r } } ) + \frac { W _ { q } } { f _ { j } ^ { ( k ) } } + d _ { q } ^ { \mathrm { D T } , ( k ) } . } \end{array}\tag{10}
$$

Here, $t _ { q } ^ { \mathrm { A r r } } = t _ { r } ^ { \mathrm { G } } + d _ { q } ^ { \mathrm { U L } }$ , and $t _ { q } ^ { \mathrm { S t a r t } , ( k ) } - t _ { q } ^ { \mathrm { A r r } }$ is the waiting time at the MEC server before execution begins.

All tasks generated in slot r arrive by $t _ { r } ^ { \mathrm { G } } + u$ . The quantity u is used only for delay analysis and does not impose a common processing start time. With no unfinished workload from the preceding slot, tasks of each class are processed sequentially at the reserved CPU rate as they become available. Since all tasks have arrived by $t _ { r } ^ { \mathrm { G } } + u$ , every class-j task satisfies

$$
t _ { q } ^ { \mathrm { O u t } , ( k ) } \leq t _ { r } ^ { \mathrm { G } } + u + \frac { W _ { m , j , r } } { f _ { j } ^ { ( k ) } } .\tag{11}
$$

Consequently,

$$
D _ { q } ^ { ( k ) } \leq \overline { { D } } _ { j } ( k ) : = u + \frac { \overline { { W } } _ { j } } { f _ { j } ^ { ( k ) } } + b _ { j } ^ { \mathrm { D T } } .\tag{12}
$$

The feasible strategy set defined in the next section ensures that all tasks are processed and their results update the DT by the end of their generation slot. Hence, no unfinished workload is carried between slots.

## IV. DT INFORMATION FRESHNESS AND STRATEGY UPDATE

A DT state is updated whenever a task is processed and its result is incorporated into the DT. We assume that each configuration change is accompanied by a fresh initialization of all monitored DT states. Thus, each configuration starts a new optimization problem with zero initial age.

## A. Digital-Twin Information Freshness

For configuration n, let $\mathcal { I } _ { c } ^ { ( n ) } \subseteq [ J ^ { ( n ) } ]$ denote the task classes associated with cluster c. Each pair $( c , j )$ , with $j \in$ $\mathcal { J } _ { c } ^ { ( n ) }$ , generates a sequence of updates to the corresponding DT state, with freshness requirement $\epsilon _ { j } ^ { ( n ) } > 0$ . Updates belonging to the same class share the same freshness requirement, while freshness is tracked separately for each cluster–class pair.

Let $t _ { n } ^ { \mathrm { c f g } }$ denote the beginning of configuration n. At this time, all monitored DT states are initialized with fresh measurements generated at $t _ { n } ^ { \mathrm { c f g } }$ , such that

$$
A _ { c , j } ( t _ { n } ^ { \mathrm { c f g } } ) = 0 , \qquad \forall c \in \mathcal { C } , j \in \mathcal { T } _ { c } ^ { ( n ) } .\tag{13}
$$

Let $\mathcal { Q } _ { c , j } ^ { ( n ) }$ denote the tasks of class $j$ generated by cluster c during configuration n. For $t \in [ t _ { n } ^ { \mathrm { c f g } } , t _ { n + 1 } ^ { \mathrm { c f g } } )$ , the generation time of the freshest information incorporated into the DT is

$$
U _ { c , j } ( t ) = \operatorname* { m a x } \left( \{ t _ { n } ^ { \mathrm { c f g } } \} \cup \left\{ t _ { q } ^ { \mathrm { G } } : q \in \mathcal { Q } _ { c , j } ^ { ( n ) } , t _ { q } ^ { \mathrm { D T } } \leq t \right\} \right)\tag{14}
$$

where $t _ { n } ^ { \mathrm { c f g } }$ represents the timestamp of the freshly initialized DT state. For a task generated in slot $r , t _ { q } ^ { \mathrm { G } } = r T$ . Here, $t _ { q } ^ { \mathrm { D T } }$ denotes the time at which the processed result updates the DT.

The age of information is

$$
\begin{array} { r } { A _ { c , j } ( t ) = t - U _ { c , j } ( t ) . } \end{array}\tag{15}
$$

The age increases linearly over time and decreases when a result with a newer generation timestamp updates the DT. Results with equal or older timestamps do not reduce the age.

The normalized age is

$$
\widetilde { A } _ { c , j } ( t ) = \frac { A _ { c , j } ( t ) } { \epsilon _ { j } ^ { ( n ) } } .\tag{16}
$$

## B. Freshness-Feasible CPU Allocations

A CPU-allocation strategy is freshness-feasible if it allows the DT to incorporate updates while keeping the AoI within the corresponding freshness thresholds. A simple approach would be to require the delay bound in (12) to satisfy $\overline { { D } } _ { j } ( k ) \le \epsilon _ { j }$ However, this condition only considers the delay of individual updates and does not account for the time between consecutive generations. In practice, an update may not be generated in every slot, and some updates may be lost. We therefore assume that each cluster generates at least one valid update for each associated class $j$ in every $h _ { j }$ consecutive slots that is successfully incorporated into the DT within the delay bound $\overline { { D } } _ { j } ( k )$ , where $h _ { j } \ \geq \ 1$ is a known integer. This assumption holds for every candidate allocation strategy. With updates generated at slot boundaries, the maximum generation time of successfully delivered updates is $h _ { j } T$ . This permits slots with no new updates and allows other updates to be lost, while bounding the time between generations of successfully delivered updates.

Consider consecutive distinct generation slots $r _ { \mathrm { p r e v } }$ and $r$ whose updates for cluster–class pair $( c , j )$ are successfully incorporated into the DT. Let $q$ be the first valid result from slot r incorporated into the DT. Provided that the freshest incorporated update before this event was generated in slot $r _ { \mathrm { p r e v } }$ , the age immediately before incorporation is

$$
A _ { c , j } \left( ( t _ { q } ^ { \mathrm { D T } } ) ^ { - } \right) = ( r - r _ { \mathrm { p r e v } } ) T + D _ { q } , \qquad r - r _ { \mathrm { p r e v } } \le h _ { j } ,\tag{17}
$$

where $D _ { q } = t _ { q } ^ { \mathrm { D T } } - t _ { r } ^ { \mathrm { G } }$ is the actual end-to-end delay. The generation-slot gap is bounded by the assumption that every $h _ { j }$ consecutive slots contain at least one generated update that is successfully delivered.

Using the conservative delay bound in (12), we construct a common set of K freshness-feasible CPU-allocation strategies, indexed by $k \in [ K ]$ . Every strategy satisfies

$$
\overline { { D } } _ { j } ( k ) \leq T , \quad h _ { j } T + \overline { { D } } _ { j } ( k ) \leq \epsilon _ { j } , \quad \forall j \in [ J ] , \forall k \in [ K ] .\tag{18}
$$

The first condition ensures that preprocessing finishes within the generation slot, preventing unfinished preprocessing workload from carrying into the next slot, and that successfully delivered updates are incorporated by that slot’s end. The second condition ensures that the peak AoI remains within the freshness threshold despite slots with no generated updates or update losses, provided the successful-update assumption holds. For $\begin{array} { r l r } { h _ { j } } & { { } = } & { 1 } \end{array}$ , each cluster–class pair generates at least one successfully delivered update in every slot, and the freshness condition reduces to $T + \overline { { D } } _ { j } ( k ) \leq \epsilon _ { j }$

The DT constructs the candidate strategies using workload upper bounds and freshness requirements, without knowledge of the workload means or distributions. Specifically, the feasibility conditions in (18) imply a minimum CPU allocation for each class:

$$
f _ { j } ^ { \mathrm { m i n } } = \frac { \overline { { W } } _ { j } } { \mathrm { m i n } \{ T , \epsilon _ { j } - h _ { j } T \} - u - b _ { j } ^ { \mathrm { D T } } } .\tag{19}
$$

We require positive denominators and $\begin{array} { r } { \sum _ { j = 1 } ^ { J } f _ { j } ^ { \operatorname* { m i n } } ~ \leq ~ F } \end{array}$ The remaining CPU capacity is distributed across classes to construct feasible candidates, let $v _ { j } ^ { ( k ) } \geq 0$ and $\begin{array} { r } { \sum _ { j = 1 } ^ { J } v _ { j } ^ { ( k ) } = 1 } \end{array}$ then

$$
f _ { j } ^ { ( k ) } = f _ { j } ^ { \operatorname* { m i n } } + \left( F - \sum _ { \ell = 1 } ^ { J } f _ { \ell } ^ { \operatorname* { m i n } } \right) v _ { j } ^ { ( k ) } .\tag{20}
$$

The weights ${ \pmb v } ^ { ( k ) }$ are selected to obtain diverse allocations across the feasible region without using expected losses. We assume that at least one feasible strategy exists, i.e., $K \geq 1 ;$ ${ \mathrm { i f ~ } } \sum _ { i } f _ { j } ^ { \operatorname* { m i n } } = F$ , the construction yields a single allocation. The feasible set is constructed before each configuration becomes effective and remains fixed throughout identification and deployment. Every strategy is available in every learning

round, including slots in which some task classes are absent.   
The set and its size may change between configurations.

Under the stated initialization, update-generation, and delay assumptions, any sequence of feasible CPU-allocation strategies maintains $A _ { c , j } ( t ) \leq \epsilon _ { j }$ for every monitored cluster–class pair throughout the configuration. This freshness guarantee is independent of the statistical guarantee of best-arm identification. Without a finite bound on $h _ { j } ,$ , random task generation alone does not ensure a deterministic upper bound on AoI.

## V. PROBLEM FORMULATION

Within configuration $n ,$ all $\begin{array} { r c l } { K } & { = } & { K _ { n } } \end{array}$ candidate CPU allocations satisfy the freshness requirements. However, these allocations distribute the computing capacity differently and may therefore produce different execution times for the same workload. Our objective is to identify a feasible allocation whose expected freshness-normalized total execution time is within a fixed tolerance of the minimum, with a predefined confidence and using as few observation slots as possible.

Each observation round corresponds to one time slot. Under allocation $k ,$ the total execution time of class-j tasks at MEC server m in slot $r$ is given by (6).

We define the freshness-normalized execution time loss as

$$
L _ { m , r } ( k ) = \sum _ { j = 1 } ^ { J } { \frac { d _ { m , j , r } ^ { \mathrm { e x e c } , ( k ) } } { \epsilon _ { j } } } = \sum _ { j = 1 } ^ { J } { \frac { W _ { m , j , r } } { \epsilon _ { j } f _ { j } ^ { ( k ) } } } .\tag{21}
$$

Normalization by $\epsilon _ { j }$ assigns a larger loss to a given execution time for classes with tighter freshness requirements.

Within configuration n, we assume that the total class workloads have fixed conditional means. For every slot r in this configuration,

$$
\mathbb { E } [ W _ { m , j , r } \mid { \mathcal F } _ { r - 1 } , k _ { m , r } ] = \overline { { w } } _ { m , j } ^ { ( n ) } , \qquad m \in [ M ] , \ j \in [ J ] ,\tag{22}
$$

where $\mathcal { F } _ { r - 1 }$ contains the information available before slot $r ,$ and $k _ { m , r } ~ \in ~ [ K ]$ is the index of the CPU-allocation strategy selected by MEC server m before observing its new workloads.

To apply the distributed learning approach introduced later and satisfy its theoretical guarantees [14], we assume that these conditional means are common across MEC servers:

$$
\begin{array} { r } { \overline { { w } } _ { m , j } ^ { ( n ) } = \overline { { w } } _ { j } ^ { ( n ) } , \qquad m \in [ M ] , \ j \in [ J ] . } \end{array}\tag{23}
$$

This homogeneity assumption concerns the total mean workload of each class. It does not require identical workload realizations or distributions across servers.

Under (22) and (23), the expected execution time loss is common to all MEC servers and is given by

$$
\ell _ { k } ^ { ( n ) } = \sum _ { j = 1 } ^ { J } \frac { \overline { { w } } _ { j } ^ { ( n ) } } { \epsilon _ { j } f _ { j } ^ { ( k ) } } .\tag{24}
$$

The optimal feasible allocation is

$$
k _ { n } ^ { * } \in \mathop { \mathrm { a r g } } _ { k \in [ K ] } \ell _ { k } ^ { ( n ) } .\tag{25}
$$

Minimizing (24) selects the allocation with the smallest expected normalized execution time among these feasible candidates. This objective is different from the constraints in (18), which ensure that the AoI of every monitored cluster–class pair remains within its freshness threshold under the stated bounds.

We aim to identify and recommend an allocation $\hat { k } _ { n }$ whose expected normalized execution time loss is within ε<sub>BAI</sub> of the minimum with confidence at least $1 - \delta .$ , using as few observation slots as possible. Specifically, the recommendation must satisfy

$$
\mathbb { P } \Big [ \ell _ { \widehat { k } _ { n } } ^ { ( n ) } - \ell _ { k _ { n } ^ { * } } ^ { ( n ) } > \varepsilon _ { \mathrm { B A I } } \Big ] \leq \delta ,\tag{26}
$$

where $\varepsilon _ { \mathrm { B A I } } = 0$ corresponds to exact best-arm identification. This guarantee applies separately to each configuration. The per-agent failure probabilities $\delta _ { m }$ are chosen such that $\delta =$ $M \delta _ { m }$

## VI. LINEAR MODEL FOR CPU ALLOCATION

In this section, we derive an exact linear representation of the expected execution time loss using features determined by the CPU allocations and freshness requirements, and describe the information observed at each MEC server. We note that alternative feature representations may also be considered, with linearity adopted as a modeling approximation rather than an exact relation, but these are not considered in this work.

## A. Linear Relation

For a configuration n, we define the feature vector of allocation k as

$$
\pmb { x } _ { k } = \left[ \frac { 1 } { \epsilon _ { 1 } f _ { 1 } ^ { ( k ) } } , \dots , \frac { 1 } { \epsilon _ { J } f _ { J } ^ { ( k ) } } \right] ^ { \top } ,\tag{27}
$$

where the configuration index is omitted from the features and allocations for notational simplicity. The unknown configuration-dependent parameter is

$$
\mathbf { \theta } _ { n } ^ { * } = - \left[ \overline { { w } } _ { 1 } ^ { ( n ) } , \ldots , \overline { { w } } _ { J } ^ { ( n ) } \right] ^ { \top } .\tag{28}
$$

Defining the expected reward as the negative expected execution time loss, we obtain

$$
\boldsymbol { \mu } _ { k } ^ { ( n ) } : = - \boldsymbol { \ell } _ { k } ^ { ( n ) } = \boldsymbol { x } _ { k } ^ { \top } \boldsymbol { \theta } _ { n } ^ { * } .\tag{29}
$$

This relation is exact under the stated execution and workload assumptions. The features depend only on the predefined CPU allocations and freshness thresholds and remain fixed throughout the configuration.

## B. Scalar Feedback

Following resource-management architectures that separate performance monitoring from allocation decisions [21], [22], we consider a monitoring component at each MEC server that normalizes task execution times by the class-specific freshness thresholds and sums them over each slot. At the end of the slot, this component reports the resulting scalar execution time loss to the allocation learner. We assume that the learner receives only this aggregate feedback, without access to execution times or realized workloads.

For configuration n, let $\tau _ { n } = r _ { n } T$ denote the start of its identification phase, where T is the slot duration and $r _ { n }$ is the first observation slot. We index consecutive observation rounds by $s = 1 , 2 , \ldots$ , with round s corresponding to slot $r = r _ { n } + s - 1$ and beginning at time $\tau _ { n } + ( s - 1 ) T$ . Each round therefore lasts one slot. Server m selects $k _ { m , n , s } = k _ { m , r } \in [ K ]$ before observing the slot’s workload and receives the scalar reward

$$
\begin{array} { c } { Y _ { m , n , s } = - L _ { m , r } ( k _ { m , n , s } ) } \\ { = \pmb { x } _ { k _ { m , n , s } } ^ { \top } \pmb { \theta } _ { n } ^ { * } + \eta _ { m , n , s } , } \end{array}\tag{30}
$$

where

$$
\eta _ { m , n , s } = - \sum _ { j = 1 } ^ { J } \frac { W _ { m , j , r } - \overline { { w } } _ { j } ^ { ( n ) } } { \epsilon _ { j } f _ { j } ^ { ( k _ { m , n , s } ) } }\tag{31}
$$

is the deviation of the realized workloads from their conditional means.

Let $\pmb { k } _ { r } = ( k _ { 1 , r } , \dots , k _ { M , r } )$ be the allocations selected by all MEC servers in slot r. We assume that the shared conditional workload means remain $\overline { { w } } _ { j } ^ { ( n ) }$ when conditioned on $\mathcal { F } _ { r - 1 }$ and $k _ { r }$ . Consequently,

$$
\mathbb { E } [ \eta _ { m , n , s } \mid \mathcal { F } _ { r - 1 } , k _ { r } ] = 0 .\tag{32}
$$

C. Observation and Parameter Bounds

Since $0 \leq W _ { m , j , r } \leq \overline { { W } } _ { j }$ , define

$$
L _ { \mathrm { m a x } } = \operatorname* { m a x } _ { k \in [ K ] } \sum _ { j = 1 } ^ { J } \frac { \overline { { W } } _ { j } } { \epsilon _ { j } f _ { j } ^ { ( k ) } } .\tag{33}
$$

Then $Y _ { m , n , s } \in [ - L _ { \operatorname* { m a x } } , 0 ]$ . Together with the conditional zeromean property in (32), Hoeffding’s lemma gives [24]

$$
\mathbb { E } [ e ^ { \alpha \eta _ { m , n , s } } \mid \mathcal { F } _ { r - 1 } , k _ { r } ] \leq \exp \left( { \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { 2 } } \right) , \sigma = { \frac { L _ { \operatorname* { m a x } } } { 2 } } , \alpha \in \mathbb { R } .\tag{34}
$$

Thus, the observation noise is conditionally σ-sub-Gaussian. The workload bounds also imply the parameter-norm bound

$$
\| \pmb { \theta } _ { n } ^ { * } \| _ { 2 } \leq S : = \left( \sum _ { j = 1 } ^ { J } \overline { { W } } _ { j } ^ { 2 } \right) ^ { 1 / 2 } .\tag{35}
$$

For the finite candidate set, a feature-norm bound is

$$
\| \pmb { x } _ { k } \| _ { 2 } \le \operatorname* { m a x } _ { k ^ { \prime } \in [ K ] } \left( \sum _ { j = 1 } ^ { J } \frac { 1 } { \epsilon _ { j } ^ { 2 } ( f _ { j } ^ { ( k ^ { \prime } ) } ) ^ { 2 } } \right) ^ { 1 / 2 } .\tag{36}
$$

These bounds are determined for each configuration and remain fixed throughout its identification phase. They are used to construct the confidence bounds in the distributed identification method introduced next.

## VII. DISTRIBUTED BANDIT LEARNING FOR CPU ALLOCATION

In this section, we formulate MEC CPU-allocation selection as a fixed-confidence best-arm identification problem in linear bandits. The objective is to identify an allocation with the predefined accuracy and confidence using as few learning rounds as possible. Each MEC server can solve this problem independently using LinGapE [27], without requiring workload homogeneity across servers. Under the homogeneity assumptions introduced above, the servers can instead share observations to identify a common allocation. We adopt DistLinGapE [14] to exploit this collaboration and accelerate identification.

Each MEC server acts as an agent, and each feasible CPUallocation strategy $k ~ \in ~ [ K ]$ corresponds to an arm. Each configuration defines a separate identification problem, with observation rounds corresponding to the learning rounds of that problem. If identification uses $N _ { n }$ consecutive observation slots, it ends at time $\hat { \tau } _ { n } = \tau _ { n } + N _ { n } T$

Before identification begins, the DT determines the feasible candidate allocations using the prescribed freshness requirements and workload, delay, and update-generation bounds. During each learning round, every server selects an allocation, applies it to the arriving tasks, and observes the scalar reward defined in (30). Meanwhile, the processed results are forwarded to the DT. The reward from each round is available before the allocation decision for the following round. We describe the procedure within a fixed configuration and omit the configuration index n for readability.

If server m were to learn independently, its local observations would give the following quantities at the beginning of learning round $s ,$ with the server index omitted for simplicity:

$$
\pmb { A } _ { s } = \lambda \pmb { I } + \sum _ { z = 1 } ^ { s - 1 } \pmb { x } _ { k _ { z } } \pmb { x } _ { k _ { z } } ^ { \top } , \quad \pmb { b } _ { s } = \sum _ { z = 1 } ^ { s - 1 } \pmb { x } _ { k _ { z } } Y _ { z } ,\tag{37}
$$

where $\lambda > 0$ is the regularization parameter, I is the $J \times J$ identity matrix, $k _ { z }$ is the arm selected in round $z ,$ and $Y _ { z }$ is the corresponding observed reward. The regularized least-squares estimate of the parameter vector would be $\hat { \pmb { \theta } } _ { s } = { \pmb { A } } _ { s } ^ { - 1 } { \pmb { b } } _ { s }$ [27].

However, When the servers collaborate, each server has access to its local observations in addition to information collected by the other servers. A coordinator, located at either an MEC server or the DT server, aggregates and distributes this information.

Algorithm 1 on the next page and the following description summarize the application of DistLinGapE to our CPUallocation problem. The detailed algorithm and theoretical analysis are provided in [14].

## A. Local Estimation and Information Sharing

Let $s _ { \ell }$ denote the first learning round of communication round $\ell ,$ with $s _ { 1 } = 1$ . A communication round comprises the learning rounds between two successive synchronizations. At the beginning of round $s ,$ server m stores the statistics of its observations collected since round $s _ { \ell } \colon$

$$
\begin{array} { c } { \displaystyle \Delta { \pmb A } _ { m , s } = \sum _ { z = s _ { \ell } } ^ { s - 1 } { \pmb x } _ { k _ { m , z } } { \pmb x } _ { k _ { m , z } } ^ { \top } , } \\ { \displaystyle \Delta { \pmb b } _ { m , s } = \sum _ { z = s _ { \ell } } ^ { s - 1 } { \pmb x } _ { k _ { m , z } } Y _ { m , z } . } \end{array}\tag{38}
$$

Here, $k _ { m , z }$ and $Y _ { m , z }$ are the allocation selected by server m and its reward in round $i ,$ respectively.

Let $A _ { \mathrm { { c o o r } , \ell } }$ and $b _ { \mathrm { c o o r } , \ell }$ denote the cumulative statistics shared at the beginning of communication round $\ell ,$ initially both zero.

Each server combines these statistics with its unshared local observations:

$$
\begin{array} { r } { \begin{array} { l } { A _ { m , s } = \lambda I + A _ { \mathrm { c o o r } , \ell } + \Delta A _ { m , s } , } \\ { b _ { m , s } = b _ { \mathrm { c o o r } , \ell } + \Delta b _ { m , s } , } \\ { \hat { \theta } _ { m , s } = A _ { m , s } ^ { - 1 } b _ { m , s } . } \end{array} } \end{array}\tag{39}
$$

Thus, each estimate is based on the total observations from all servers up to the latest synchronization, together with the server’s subsequent local observations.

Communication and information sharing follows the eventtriggered rule of [14], which measures the reduction in the volume of the confidence ellipsoid for the estimate of $\pmb { \theta } ^ { * }$ since the beginning of the current communication round. When any server meets this condition, all servers transmit their unshared statistics to the coordinator. If communication is triggered after learning round $s ,$ the coordinator updates

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { \mathrm { c o o r } , \ell + 1 } = { \cal A } _ { \mathrm { c o o r } , \ell } + \sum _ { m = 1 } ^ { M } \Delta { \cal A } _ { m , s + 1 } } , } \\ { { \displaystyle ~ b _ { \mathrm { c o o r } , \ell + 1 } = b _ { \mathrm { c o o r } , \ell } + \sum _ { m = 1 } ^ { M } \Delta b _ { m , s + 1 } } , } \end{array}\tag{40}
$$

where the increments include all observations collected since the previous synchronization until round s. The coordinator broadcasts its updates, and the servers clear their local increments. The next communication round starts at $s _ { \ell + 1 } = s + 1$

## B. Allocation Selection and Stopping Rule

Each server selects CPU allocations sequentially to identify the best feasible allocation. In each learning round, it selects an allocation that helps distinguish its currently preferred allocation from a competing allocation whose performance remains uncertain. Under the noise and parameter assumptions of [14], define the confidence radius for the parameter estimate [14], [25]

$$
C _ { m , s } = \sigma \sqrt { 2 \log \left( \frac { \operatorname * { d e t } ( A _ { m , s } ) ^ { 1 / 2 } } { \lambda ^ { J / 2 } ( \delta / M ) } \right) } + \sqrt { \lambda } S ,\tag{41}
$$

where $\sigma$ is the conditional sub-Gaussian noise parameter, $S$ bounds $\lVert { \pmb \theta } ^ { * } \rVert _ { 2 }$ , and δ is the fixed error probability.

For allocations i and $j ,$ the estimated reward difference and its confidence radius are [14]

$$
\begin{array} { r l } & { \hat { \Delta } _ { m , s } ( i , j ) = ( { \pmb x } _ { i } - { \pmb x } _ { j } ) ^ { \top } \hat { \pmb \theta } _ { m , s } , } \\ & { \beta _ { m , s } ( i , j ) = C _ { m , s } \| { \pmb x } _ { i } - { \pmb x } _ { j } \| _ { { \pmb A } _ { m , s } ^ { - 1 } } . } \end{array}\tag{42}
$$

Server m first identifies the allocation with the largest estimated reward:

$$
i _ { m , s } \in \mathop { \mathrm { a r g } } _ { i \in [ K ] } \ : \mathbf { \em x } _ { i } ^ { \top } \hat { \pmb { \theta } } _ { m , s } .\tag{43}
$$

For $K \geq 2 .$ , it then selects the most ambiguous allocation, $\mathrm { i . e . }$ , the one with the largest upper confidence bound on its improvement over $i _ { m , s } \colon$

$$
j _ { m , s } \in \operatorname * { a r g m a x } _ { j \in [ K ] \backslash \{ i _ { m , s } \} } \left\{ \hat { \Delta } _ { m , s } ( j , i _ { m , s } ) + \beta _ { m , s } ( j , i _ { m , s } ) \right\} .\tag{44}
$$

```latex
Algorithm 1 DistLinGapE for CPU-Allocation Identification:
Summary of [14]
1: Input: $\{ \pmb { x } _ { k } \} _ { k = 1 } ^ { K } , \ \varepsilon _ { \mathrm { B A I } } , \ \delta , \ \lambda , \ \sigma , \ S ,$ and communication
threshold $D _ { \mathrm { t h } }$
2: Initialize shared and local statistics to zero; set $\ell = 1$ and
$s _ { 1 } = 1$
3: for $s = 1 , 2 , \ldots$ . do
4: Each server forms its parameter estimate using (39).
5: Each server computes $i _ { m , s } , j _ { m , s } ,$ and $B _ { m , s }$ using (43)–
(45).
6: if some server m has $B _ { m , s } \leq \varepsilon _ { \mathrm { B A I } }$ then
7: Broadcast and return $\ddot { k } = i _ { m , s } .$
8: end if
9: Each server selects an allocation $k _ { m , s }$ using (47) or
allocation-ratio tracking [14].
10: Each server applies its allocation in the slot correspond
ing to round s, observes $Y _ { m , s } ,$ and updates its local
statistics.
11: if any server meets the communication condition with
threshold $D _ { \mathrm { t h } }$ [14] then
12: All servers transmit their unshared updates to the
coordinator, which aggregates them using (40) and
broadcasts the updated totals.
13: Servers clear local increments and reset the commu
nication counters; set $s _ { \ell + 1 } = s + 1$ and $\ell \gets \ell + 1$
14: end if
15: end for
```

The corresponding bound is

$$
B _ { m , s } = \hat { \Delta } _ { m , s } ( j _ { m , s } , i _ { m , s } ) + \beta _ { m , s } ( j _ { m , s } , i _ { m , s } ) .\tag{45}
$$

If $B _ { m , s } ~ \leq ~ \varepsilon _ { \mathrm { B A I } }$ , server m reports $i _ { m , s }$ to the coordinator and identification terminates. Here, $\varepsilon _ { \mathrm { B A I } } \geq 0$ is the permitted reward suboptimality. Under the assumptions of [14], the returned allocation <sup>ˆ</sup>k satisfies

$$
\mathbb { P } \left( \operatorname* { m a x } _ { k \in [ K ] } \pmb { x } _ { k } ^ { \top } \pmb { \theta } ^ { * } - \pmb { x } _ { k } ^ { \top } \pmb { \theta } ^ { * } \leq \varepsilon _ { \mathrm { B A I } } \right) \geq 1 - \delta .\tag{46}
$$

If the stopping condition is not met, the server samples an allocation that helps distinguish $i _ { m , s }$ from $j _ { m , s } .$ . DistLinGapE supports greedy sampling and allocation-ratio tracking [14], [27]. The greedy rule selects

$$
k _ { m , s } \in \mathop { \arg \operatorname* { m i n } } _ { k \in [ K ] } \| { \pmb x } _ { i _ { m , s } } - { \pmb x } _ { j _ { m , s } } \| _ { ( { \pmb A } _ { m , s } + { \pmb x } _ { k } { \pmb x } _ { k } ^ { \top } ) ^ { - 1 } } .\tag{47}
$$

This rule selects the allocation that minimizes the uncertainty for the estimated reward difference between $i _ { m , s }$ and $j _ { m , s }$ after observing its reward. Any candidate allocation may be selected, not only $i _ { m , s }$ and $j _ { m , s } ,$ , since the linear model allows observations from other allocations to inform this reward difference.

Alternatively, allocation-ratio tracking selects allocations according to target sampling proportions to distinguish between $i _ { m , s }$ and $j _ { m , s }$ . The target proportions and tracking rule are given in [14], [27].

After identification, the coordinator distributes the recommended CPU allocation to the MEC servers for deployment.

## VIII. NUMERICAL RESULTS

In this section, we show the performance of our proposed approach for identifying a CPU-allocation strategy with minimum expected execution delay normalized by freshness, while meeting the task classes’ freshness requirements.

## A. Simulation Setup

We consider a 600 m × 600 m region containing $M = 4$ MEC servers and $C ~ = ~ 1 6$ IoT clusters. The servers are located at (150, 150), (150, 450), (450, 150), and (450, 450) m. Four clusters are independently and uniformly placed within each server’s quadrant and associated with their nearest MEC server. Each server has a processing capacity of $F = 8$ GHz and processes tasks belonging to $J = 8$ classes. The slot duration is $T = 2 0 0 ~ \mathrm { { m s } }$

At the beginning of slot $r ,$ cluster c generates $N _ { c , j , r } ~ =$ $1 + \mathrm { B e r n o u l l i } ( p _ { j } )$ tasks of class $j .$ In the main experiment, we use $\pmb { p } ~ = ~ [ 0 . 9 5 , 0 . 9 0 , 0 . 6 0 , 0 . 5 0 , 0 . 2 0 , 0 . 2 0 , 0 . 1 0 , 0 . 1 0 ] ^ { \top }$ Each task has an independently and uniformly distributed size between 40 and 60 kbytes. The class-specific computational requirements are $\pmb { \kappa } = [ 4 4 , 4 0 , 2 0 , 1 6 , 6 , 5 , 4 , 3 ] ^ { \top }$ cycles per input bit.

Each cluster has an uplink rate of 30 Mbit/s, with the four clusters associated with a server served simultaneously. Uploading the maximum total input generated by one cluster requires 17.68 ms. We therefore use a common uplink delay bound of $u = b _ { i } ^ { \mathrm { U L } } = 2 0$ ms. Each MEC server has a dedicated 100 Mbit/s connection to the DT server, and preprocessing reduces the forwarded data volume to 10% of the input size. The combined forwarding and DT-incorporation delay is assumed to be bounded by $b _ { j } ^ { \mathrm { { \tiny { D T } } } } = 1$ ms.

The class-specific freshness requirements are $\begin{array} { r l } { \epsilon } & { { } = } \end{array}$ $[ 4 0 0 , 4 0 0 , 4 0 0 , \dot { 4 } 0 0 , 3 7 5 , 3 7 5 , 3 5 0 , 3 5 0 ] ^ { \top }$ ms. Since every cluster generates at least one task of each class in every slot, we use $h _ { j } = 1$

## B. Candidate Strategies and Evaluation

Each candidate set contains $K = 3 0$ distinct feasible CPUallocation strategies. We determine the minimum allocations from the workload upper bounds and freshness requirements using (19), and distribute the remaining capacity according to (20). Every set includes equal CPU allocation, $f _ { j } ^ { \mathrm { e q } } = 1$ GHz, and an allocation that distributes the residual capacity equally above the class-specific minima.

For the main experiment, the remaining candidates comprise 14 allocations generated using symmetric Dirichlet $( 1 , \ldots , 1 )$ weights and 14 using Dirichlet $( 3 , \ldots , 3 )$ weights. The results are obtained over 100 algorithm runs.

The suboptimality gap of candidate k is $\Delta _ { k } = \mu _ { k ^ { \star } } - \mu _ { k }$ . The smallest gap, $\begin{array} { r } { \Delta _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { k \neq k ^ { \star } } \Delta _ { k } , } \end{array}$ , is the expected reward difference between the best and second-best candidates. In the main example, $\Delta _ { \mathrm { m i n } } \approx 0 . 0 6 5 2$ . We also consider candidate sets with broader and smaller overall reward separation and minimum gaps in Figs 4 and 5.

We compare DistLinGapE with equal CPU allocation, random allocation, independent LinGapE, shared uniform exploration, and an oracle. Equal CPU allocation always assigns the same capacity to every class, whereas random allocation selects uniformly from the candidate set in each round. Independent LinGapE uses only local observations, and each server stops and deploys its recommendation independently. Shared uniform exploration cycles through all candidates equally, while using the same linear estimator, communication rule, and stopping criterion as DistLinGapE. The oracle knows the mean workloads and selects the candidate minimizing the expected freshness-normalized execution time loss.

Each server receives one scalar observation per round, giving M observations across the network. The learning methods begin with 30 initialization rounds in which each server samples every candidate once. We use an identification tolerance of $\varepsilon _ { \mathrm { B A I } } = 0 . 0 2$ , an overall error probability of $\delta = 0 . 2$ , and ridge regularization $\lambda = 0 . 1$ . Learners observe only the scalar freshness-normalized execution time loss. After identification, each method deploys its recommended allocation for the remaining rounds.

## C. Freshness-Normalized Loss and CPU Execution Delay

We first evaluate the freshness-normalized execution time loss averaged across MEC servers,

$$
\frac { 1 } { M } \sum _ { m = 1 } ^ { M } L _ { m , r } ( k _ { m , r } ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \sum _ { j = 1 } ^ { J } \frac { W _ { m , j , r } } { \epsilon _ { j } f _ { j } ^ { ( k _ { m , r } ) } } .
$$

This dimensionless metric directly corresponds to the optimization objective.

Fig. 2 shows that DistLinGapE reaches the oracle’s performance earlier than the other learning methods. It satisfies its stopping condition after an average of 1354 rounds, with stopping times ranging from 1260 to 1455 rounds across the 100 trials. Shared uniform exploration requires an average of 2932 rounds. Independent LinGapE requires 5127 rounds per server on average, while all four servers complete identification after an average of 5264 rounds.

All three learning methods identify the exact best candidate in this experiment. Over the final 500 rounds, their mean normalized losses coincide with the oracle at approximately 0.7774, compared with 0.9627 for equal CPU allocation and 0.9776 for random allocation. The benefit of DistLinGapE is therefore earlier identification and deployment, rather than a different final allocation.

To express the resulting performance in milliseconds, we also show the unnormalized mean class CPU execution time,

$$
\frac { 1 } { M J } \sum _ { m = 1 } ^ { M } \sum _ { j = 1 } ^ { J } \frac { W _ { m , j , r } } { f _ { j } ^ { ( k _ { m , r } ) } } .
$$

This metric averages class execution durations across servers and is distinct from per-task delay and end-to-end DT update delay. Minimizing the freshness-normalized loss does not generally minimize this unnormalized metric, since classes receive different weights through their freshness requirements. Here, the thresholds are relatively close, ranging from 350 to 400 ms, so the weights differ only modestly. We therefore report CPU execution times as an additional practical performance measure, while assessing optimality using the normalized objective.

![](images/cac9779b2e2260e4e5a79a8227df8604f3668ff9e95e40168d6de3c938c45a9b.jpg)  
Fig. 2: Mean freshness-normalized execution time loss for the main candidate set.

![](images/7b2393aa06cc25d6bb09705e714bbca729aa860af352ee8ce8b6bbaea69fc5db.jpg)  
Fig. 3: Mean class CPU execution time for the main candidate set.

Fig. 3 shows that the allocation selected by all three learning methods and the oracle achieves a mean execution time of approximately 38.34 ms over the final 500 rounds, compared with 47.74 ms for equal CPU allocation and 47.96 ms for random allocation. Thus, optimizing the normalized objective also reduces the unnormalized execution time in this experiment.

We also evaluate allocation sets with broader or smaller overall reward separation and minimum gaps in Figs. 4 and 5. For these experiments, we use $\begin{array} { r l } { p } & { { } = } \end{array}$ $[ 0 . 8 , 0 . 7 , 0 . 6 , 0 . 5 , 0 . 4 , 0 . 3 , 0 . 2 , 0 . 1 ] ^ { \top }$ and computation requirements $\pmb { \kappa } = [ 3 2 , 2 8 , 2 4 , 2 0 , 1 6 , 1 4 , 1 2 , 1 0 ] ^ { \top }$ cycles per input bit. Each experiment includes ten candidate sets and ten independent workload realizations per set. The freshness thresholds remain unchanged.

Fig. 4 considers candidates selected by geometric coverage from options generated using symmetric Dirichlet distributions with concentrations 0.5, 1, and 5. This produces larger reward gaps to many suboptimal candidates. DistLinGapE stops after an average of 68 rounds, compared with 133 rounds for shared uniform exploration and 221 rounds per server for independent

![](images/97935819284ca196eeb2d01aa798aba98a36c0f0c74c91c311de303f28193f4c.jpg)  
Fig. 4: Mean class CPU execution time for geometrically spread feasible candidates.

LinGapE. All three methods identify the exact best candidate under the normalized objective in all trials.

Fig. 5 considers candidates generated using symmetric Dirichlet distributions with concentrations 2 and 5, without separation filtering. Their best–second-best reward gaps range from approximately $6 . 4 3 \times 1 0 ^ { - 5 }$ to 0.0124. DistLinGapE requires an average of 1948 rounds. All 100 recommendations are $\varepsilon _ { \mathrm { B A I } ^ { - } } \mathrm { o p t i m a l }$ for $\varepsilon _ { \mathrm { B A I } } = 0 . 0 2$ , while 89 identify the exact best candidate.

Within the 7000-round horizon, shared uniform exploration completes identification in 72 of the 100 trials. For independent LinGapE, all four servers finish in 40 of the 100 trials. Their late-stage execution times therefore include continued exploration in unfinished runs.

These results show that identification complexity depends on the observed rewards, reward gaps, and the stopping conditions. DistLinGapE uses adaptive sampling and information sharing to reduce the exploration needed before deployment. Since its objective is identification rather than immediate delay minimization, it may select suboptimal allocations during exploration.

## D. Collaboration Speedup

We evaluate the benefit of collaboration using the smallgap candidate sets and $M \in \{ 1 , 2 , 4 , 8 \}$ agents. Each server serves four clusters, giving $C = 4 M$ . This preserves the peragent workload distribution while increasing the number of observations collected concurrently. For $M \ = \ 1$ , the agent learns independently without communication. Each value of M is evaluated over 100 trials using the same candidate sets and fresh workload realizations.

Fig. 6 shows that the mean number of identification rounds decreases from approximately 7323 for one agent to 3746, 1943, and 1052 for two, four, and eight agents, respectively. All runs satisfy the stopping condition and return an $\varepsilon _ { \mathrm { B A I } ^ { - } }$ optimal candidate.

We estimate the learning speedup, which is the ratio of mean number of samples per server in the independent case to the collaborative case [14]. Fig. 7 compares the measured speedup with the ideal speedup which is proportional to the number of collaborating servers. The observed speedups are 1.95, 3.77, and 6.96 for two, four, and eight CPUs, respectively.

![](images/fcaeab3468685f2e9ebe37fc6d5457bfa3cba9000c2b62f8d6c6a704b951dd35.jpg)  
Fig. 5: Mean class CPU execution time for candidate sets with small best–second-best reward gaps.

![](images/4ceb6de2e8822d9392704907400ece4601f59a311e5ced5de44da8d04ed8a9fd.jpg)  
Fig. 6: Mean identification rounds versus the number of collaborating MEC servers for the small-gap candidate sets.

The deviation from ideal linear speedup may result from initialization, server’s sampling decisions, and delayed information sharing. Changing the communication threshold $D _ { \mathrm { t h } }$ controls the local computation and communication tradeoff, which can affect the speedup.

## IX. CONCLUSION

In this paper, we proposed an approach to reduce the DTupdate delays in IoT applications with large computational workloads, where the DT can be updated using a subset of the IoT data or some information extracted from it. We considered sensing tasks belonging to multiple classes, with different preprocessing demands and freshness requirements. The proposed approach relies on mutual support between the DT and MEC servers: the DT constructs freshness-feasible CPU allocations, while the MEC servers collaboratively apply DistLinGapE to identify an allocation with minimum expected freshness-normalized execution delay. The numerical results show the effectiveness of this approach, while showing that learning complexity depends on the candidate CPU allocations. Collaboration significantly reduces the number of learning rounds, achieving near-linear speedup in the evaluated scenarios. Future work will investigate collaboration among heterogeneous MEC servers, and a better utilization of the CPU resources for further delay reduction.

![](images/60182dcf74ce0d10e66c4b46956b5613adeb68defb59c6dbd44a52f4c095ab60.jpg)  
Fig. 7: Learning speedup relative to a single independent agent, compared with ideal linear speedup.

## REFERENCES

[1] R. Chai, J. Pan, X. Wu, and T. Qiu, “Cost-efficient age-of-information minimization in digital twins for internet of vehicles,” in IEEE INFO-COM 2026-IEEE Conference on Computer Communications, pp. 1–6, IEEE, 2026.

[2] E. Bozkaya-Aras, “Digital twin-enabled age of information-aware scheduling for industrial IoT edge networks,” Pervasive and Mobile Computing, vol. 112, p. 102083, 2025.

[3] M. R. Kabir, F. S. Shishir, S. Shomaji, and S. Ray, “Digital twins in healthcare IoT: A systematic review,” High-Confidence Computing, vol. 5, no. 3, p. 100340, 2025.

[4] A. Alkhateeb, S. Jiang, and G. Charan, “Real-time digital twins: Vision and research directions for 6G and beyond,” IEEE communications magazine, vol. 61, no. 11, pp. 128–134, 2023.

[5] B. Li, H. Cai, L. Liu, and Z. Fei, “Delay-aware digital twin synchronization in mobile edge networks with semantic communications,” IEEE Transactions on Vehicular Technology, vol. 74, no. 7, pp. 10974–10983, 2025.

[6] K. Noroozi, T. D. Todd, D. Zhao, and G. Karakostas, “Age of information in digital twin migration,” IEEE Transactions on Vehicular Technology, 2025.

[7] X. Zhou, Y. Peng, T. Lan, Z. Zhang, B. Tang, and X. Guan, “Digital twin empowered task offloading for mobile-edge computing in 6G internet of vehicles,” IEEE Internet of Things Journal, vol. 12, no. 15, pp. 29189– 29202, 2025.

[8] Y. Sanjalawe, S. Fraihat, S. Al-E’mari, and S. N. Makhadmeh, “Bridging the gap: A comprehensive survey on AI-driven digital twin networks for future wireless systems,” Journal ofKing Saud University Computer and Information Sciences, vol. 38, no. 3, p. 113, 2026.

[9] M. Krantzik, A. Klein, and L. Xiang, “Optimizing sensor data compression and digital twin synchronization via a Stackelberg game,” in 2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), pp. 1–6, IEEE, 2025.

[10] J. Tang, J. Nie, J. Bai, J. Xu, S. Li, Y. Zhang, and Y. Yuan, “UAVassisted digital-twin synchronization with tiny-machine-learning-based semantic communications,” IEEE Internet of Things Journal, vol. 11, no. 17, pp. 28437–28451, 2024.

[11] S.-S. Tzeng, Y.-J. Lin, and S.-W. Wang, “Age of information in IoT devices with integrated heterogeneous sensors under slotted ALOHA,” IEEE Sensors Journal, vol. 25, no. 11, pp. 20842–20853, 2025.

[12] T. Park, W. Saad, and B. Zhou, “Centralized and distributed age of information minimization with nonlinear aging functions in the internet of things,” IEEE Internet of Things Journal, vol. 8, no. 10, pp. 8437– 8455, 2020.

[13] H. Choi, H. Yu, and E. Lee, “Latency-classification-based deadlineaware task offloading algorithm in mobile edge computing environments,” Applied Sciences, vol. 9, no. 21, p. 4696, 2019.

[14] M. Yahya, A. Sezgin, and S. Maghsudi, “Service placement in small cell networks using distributed best arm identification in linear bandits,” IEEE Transactions on Mobile Computing, 2026.

[15] J. Li, J. Wang, Q. Chen, Y. Li, and A. Y. Zomaya, “Digital twin-enabled service satisfaction enhancement in edge computing,” in IEEE INFO-COM 2023-IEEE Conference on Computer Communications, pp. 1–10, IEEE, 2023.

[16] Y. Guo, Y. Fu, Y. Zhang, and T. Q. S. Quek, “Age-of-information and energy optimization in digital twin edge networks,” in GLOBECOM 2024 - 2024 IEEE Global Communications Conference, p. 3194–3200, IEEE, Dec. 2024.

[17] J. Li, S. Guo, W. Liang, J. Wang, Q. Chen, Z. Hong, Z. Xu, W. Xu, and B. Xiao, “AoI-aware service provisioning in edge computing for digital twin network slicing requests,” IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 14607–14621, 2024.

[18] M. Heydari, T. D. Todd, D. Zhao, and G. Karakostas, “Channel sharing using digital twins and federated optimization,” IEEE Internet of Things Journal, 2026.

[19] D. Van Huynh, V.-D. Nguyen, S. R. Khosravirad, G. K. Karagiannidis, and T. Q. Duong, “Distributed communication and computation resource management for digital twin-aided edge computing with short-packet communications,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 10, pp. 3008–3021, 2023.

[20] M. Wirth, A. Ortiz, and A. Klein, “Risk-aware learning for digital twin placement in non-stationary hierarchical edge computing,” in 2026 IEEE Wireless Communications and Networking Conference (WCNC), pp. 1– 6, IEEE, 2026.

[21] R. Bhardwaj, K. Kandasamy, A. Biswal, W. Guo, B. Hindman, J. Gonzalez, M. Jordan, and I. Stoica, “Cilantro: Performance-aware resource allocation for general objectives via online feedback,” in 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23), pp. 623–643, 2023.

[22] Z. Wang, P. Li, C.-J. M. Liang, F. Wu, and F. Y. Yan, “Autothrottle: A practical bi-level approach to resource management for SLO-targeted microservices,” in 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24), pp. 149–165, 2024.

[23] X. Ai, W. Liang, and C. Liu, “Joint optimization of model retraining and inference services in DT-assisted edge computing,” IEEE Transactions on Networking, vol. 34, pp. 1804–1819, 2025.

[24] T. Lattimore and C. Szepesvari,´ Bandit algorithms. Cambridge University Press, 2020.

[25] Y. Abbasi-Yadkori, D. Pal, and C. Szepesv´ ari, “Improved algorithms for´ linear stochastic bandits,” Advances in Neural Inf. Proc. Sys., vol. 24, pp. 1–9, 2011.

[26] J.-Y. Audibert and S. Bubeck, “Best arm identification in multi-armed bandits,” in COLT-23th Conference on learning theory-2010, pp. 13–p, 2010.

[27] L. Xu, J. Honda, and M. Sugiyama, “A fully adaptive algorithm for pure exploration in linear bandits,” in International Conference on Artificial Intelligence and Statistics, pp. 843–851, PMLR, 2018.

[28] M. Soare, A. Lazaric, and R. Munos, “Best-arm identification in linear bandits,” Advances in Neural Inf. Proc. Sys., vol. 27, 2014.