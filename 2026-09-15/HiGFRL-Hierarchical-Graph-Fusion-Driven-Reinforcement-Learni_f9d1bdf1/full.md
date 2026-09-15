# HiGFRL: Hierarchical Graph Fusion-Driven Reinforcement Learning for Dependency-Aware Task Scheduling in Heterogeneous Cloud

Tiangang Li, Shi Ying, Xiangbo Tian

Abstract—Online scheduling of dependency-aware tasks in heterogeneous cloud clusters is a fundamental yet challenging problem due to the complex interplay between DAG topologies and multi-dimensional resource constraints. While Deep Reinforcement Learning (DRL) has shown promise, existing Graph Neural Network-based approaches often struggle to efficiently model high-order topological dependencies and suffer from loose coupling between task and resource states, leading to myopic scheduling decisions. To address these limitations, we propose HiGFRL, a Hierarchical Graph Fusion-Driven Reinforcement Learning framework. HiGFRL constructs a novel three-level state representation comprising a Static Hypergraph, a Dynamic Global Graph, and a Local Bipartite Graph to explicitly model the interplay between task dependencies and real-time cluster dynamics. Specifically, we design a fusion-driven dual-network architecture to optimize reinforcement learning decision-making, where a Context Fusion Allocator integrates local bipartite matching features with fused global context to execute precise task-to-node allocation, and a Global State Evaluator leverages the global dynamic graph representation to accurately estimate expected long-term cumulative reward. Furthermore, we incorporate a topology-prior-guided hybrid reward mechanism that distills static topological priors into the learning process to accelerate convergence. Extensive experiments using real-world Alibaba cluster traces demonstrate that HiGFRL significantly outperforms heuristics and DRL baselines. Specifically, in challenging large-scale high-load scenarios, HiGFRL reduces the Makespan by up to 32.55%, and optimizes the average task flow time and average task wait time by 13.58% and 13.79%, respectively. The experimental results confirm that HiGFRL not only significantly improves cluster job throughput but also ensures superior Quality of Service by substantially reducing queuing delays.

Index Terms—Cloud computing, task scheduling, quality of service, deep reinforcement learning, graph neural networks.

## 1. INTRODUCTION

geneous environments [1], [2]. These environments integrate diverse resource configurations, such as different specifications of CPU and memory combinations, and non-uniform processing capabilities, thoroughly revolutionizing the execution paradigm of cloud service applications [3], [4]. Concurrently, the complexity of cloud workloads has increased significantly [5], [6]. To unify the representation of diverse batch workloads in production environments, recent studies and large-scale traces, such as Alibaba cluster data, have widely adopted a standardized Job-Task-Instance hierarchical abstraction [7], [8], [9]. In this paradigm, a Job represents a complete workflow modeled as a Directed Acyclic Graph (DAG), a Task serves as a logical execution unit within the DAG, and an Instance is the physical execution entity carrying the actual computational load.

This hierarchical paradigm introduces a critical challenge: Instance-level Online Dependency-aware Scheduling. Unlike traditional DAG scheduling that treats tasks as atomic units, the scheduler must make real-time decisions to map dynamically arriving streams of fine-grained instances onto massive heterogeneous resources, while strictly satisfying precedence constraints derived from task dependencies, multi-dimensional resource capacity constraints, and assignment uniqueness constraints. The objective is to jointly optimize multiple conflicting performance metrics, specifically minimizing the average task flow time, task wait time, and overall makespan. This is a typical NP-hard problem that requires the scheduling algorithm to strike a delicate balance between long-term topological planning and immediate response for resource allocation in highly dynamic and uncertain environments [10], [11], [12].

Traditional solutions primarily rely on heuristic algorithms [13], [14], [15], [16], [17], [18]. Static list scheduling approaches prioritize tasks based on topological ranks. Although computationally efficient, they lack the adaptability to cope with the stochastic arrival patterns of runtime workloads. Dynamic heuristics, such as Tetris, adopt greedy strategies for multi-dimensional resource packing. However, they are often limited by myopia, tending to block future critical path tasks in pursuit of short-term resource utilization. To overcome these rigidities, Deep Reinforcement Learning (DRL) has emerged as a promising alternative. Nevertheless, early DRLbased approaches utilizing MLPs or CNNs, such as DeepRM, struggled to effectively capture the non-Euclidean topological dependencies inherent in DAGs.

Recognizing this, recent research has shifted towards Graph Neural Network (GNN)-based DRL paradigms [19], [20], [21], [22], [23]. Pioneering works like Decima and READYS have utilized Graph Convolutional Networks (GCNs) to encode DAG structures and combined them with DRL for decisionmaking, becoming the mainstream direction for solving complex scheduling problems [24], [25]. More advanced approaches have introduced Graph Attention Networks (GATs) to capture pairwise task correlations [26], [27]. Despite the progress, existing GNN-based approaches still suffer from two critical defects when applied to complex heterogeneous environments.

Firstly, conventional directed graph structures have intrinsic defects in capturing many-to-one synchronization dependencies. Real-world cloud workflows widely contain Join patterns, where the readiness of a task is strictly constrained by the simultaneous completion of multiple predecessor tasks. Existing approaches typically model these dependencies as pairwise edges, forcing GNNs to rely on multi-hop message passing to aggregate upstream information. This iterative process often leads to severe over-smoothing and signal dilution, making it difficult for the scheduler to precisely perceive the dependency aggregation constraints of tasks deep in the dependency chain. In contrast, introducing the Hypergraph is crucial because hyperedges can directly and losslessly aggregate features from all predecessors in a single convolution operation, thereby preserving high-order synchronization semantics and providing the scheduler with a clearer topological vision.

Secondly, existing state representation and fusion mechanisms suffer from a severe mismatch in aligning topological importance with resource availability. Existing approaches, such as MODRL, typically encode topological features and resource features independently and rely on simple concatenation to combine them [27]. Such approaches lack a decoupling and fusion strategy to explicitly model the highorder interactions between the topological importance of a task within the static view and the heterogeneous capabilities of specific nodes within the dynamic view. For example, the model inherently treats resource state merely as an additional node attribute and fails to simultaneously perceive the intrinsic connection between a task being on the critical path and a node possessing high available resource capacity and strong processing capability. This leads to decision myopia: the scheduler might maximize current resource packing efficiency but fail to reserve high-performance resources for future critical path tasks, ultimately damaging long-term performance.

To address these challenges, we propose HiGFRL, a hierarchical graph fusion-driven reinforcement learning framework. HiGFRL aims to synergistically optimize fine-grained topology-aware and resource-aware decisions through a multiview representation learning paradigm.

Contributions. The main contributions of this study are summarized as follows:

• Hierarchical Multi-View State Construction: We propose a novel multi-level state representation mechanism, comprising the static task dependency hypergraph for efficiently capturing high-order dependencies and synchronization constraints, the dynamic global cluster graph for real-time perception of cluster-wide resource posture, and the local bipartite matching graph for optimizing microscopic matching decisions. This hierarchical structure achieves explicit decoupling and semantic enhancement of the system state.

• Fusion-Driven Dual-Stream Architecture: A fusiondriven dual-network architecture is designed. The context-aware matching actor network employs a crosslevel fusion mechanism to integrate local bipartite matching features with the broadcasted global context, executing precise instance-to-node assignments. Meanwhile, the global state-value critic network utilizes the global dynamic graph representation to accurately estimate the expected long-term cumulative reward, effectively reducing the variance of policy gradients.

• Structural-Semantic Decoupling and Fusion: By hierarchically decoupling the logical task topology from physical resource states, HiGFRL effectively alleviates the semantic entanglement and over-smoothing issues inherent in flat graph representations. This mechanism successfully fuses topological importance with resource availability, ensuring precise identification of critical path bottlenecks by preventing their topological signals from being diluted by noisy resource fluctuations.

• Extensive Evaluation: HiGFRL is evaluated using largescale real-world Alibaba cluster traces. Experimental results demonstrate that HiGFRL significantly outperforms baselines across various metrics. Specifically, in the largescale high-load scenario, HiGFRL reduces the makespan by 32.55%. In the extreme load scenario, it reduces the average task flow time by 13.58% and the average task wait time by 13.79%. These results indicate that HiGFRL exhibits superior performance in both enhancing job throughput and improving user experience.

The rest of this paper is organized as follows. Section 2 introduces related work. Section 3 provides a detailed definition of the research problem and the system model. Section 4 presents the design of the HiGFRL framework. Section 5 presents the experimental design and results analysis. Section 6 concludes the research findings and future work.

## 2. RELATED WORK

In modern cloud and distributed systems, complex applications typically manifest as DAG workflows with strict task dependencies. Dependency-aware scheduling aims to optimize system metrics by rationally allocating computing resources across DAG nodes. To address this problem, existing research can be broadly categorized into heuristic-based approaches and DRL-based approaches.

Heuristic-Based Approaches. Regarding single-objective or constrained optimization, Wu et al. [13] propose the L-ACO and ProLiS algorithms, utilizing probabilistic upward rank for deadline-constrained subtask allocation to balance cost and time under strict constraints. For edge-cloud environments, Lin et al. [14] introduce a cost-driven strategy that reduces data transmission costs by merging cut edges and scheduling task sequences on partial critical paths. Zhang et al. [15] propose the BCWS algorithm to minimize overall completion time across heterogeneous VM and serverless resources under budget constraints. To address conflicting practical objectives, multi-objective optimization is widely studied. Ismayilov et al. [16] propose NN-DNSGA-II, a neural networkassisted evolutionary algorithm simultaneously optimizing six dimensions, including completion time, execution cost, energy consumption, load balancing, system reliability, and resource utilization. Qin et al. [17] develop RA-MOMA, enhancing scheduling robustness via critical path analysis and local resource optimization strategies. Additionally, Sun et al. [18] apply a multi-tree genetic programming approach to automatically evolve rules for task and resource selection in dynamic multi-cloud environments. Overall, while heuristic approaches achieve satisfactory performance in specific stable scenarios by leveraging domain knowledge, they generally struggle to adapt to highly dynamic workloads and heterogeneous resources. They often require tedious manual parameter tuning, thus lacking generalizability.

DRL-based Approaches. To overcome the poor adaptability of heuristics in dynamic environments, recent research actively utilizes DRL for adaptive dependency-aware scheduling, focusing on neural networks to process complex DAG topologies. Early DRL approaches rely on basic architectures like MLPs or CNNs. For instance, Dong et al. [19] formulate scheduling as a Markov Decision Process (MDP) and propose RLWS, an Actor-Critic approach with iterative local rescheduling. Cheng et al. [20] introduce H2O-Cloud, a hierarchical online framework enabling pre-training-free DRL scheduling, significantly easing practical deployment.

With the rise of GNNs, Graph Reinforcement Learning (GRL) approaches directly modeling non-Euclidean DAG structures have become a research hotspot [28]. Decima [24] pioneers this by embedding job graphs via GNNs and training with REINFORCE, significantly improving completion times compared to classical heuristics. Hu et al. [21] propose Spear, combining Monte Carlo Tree Search (MCTS) with DRL to optimize scheduling under complex dependencies and heterogeneous demands, yielding notable improvements. Improved GNN architectures further enhance DAG modeling accuracy. Drag-JDEC [22] employs GATs for edge computing scheduling, achieving a remarkable performance boost. GA-DRL [26] utilizes multi-head GATs with bidirectional aggregation and non-uniform sampling for strong generalization on unseen DAGs. Wang et al. [27] propose a spatio-temporal GNN with a Set Transformer for multi-objective scheduling. READYS [25] extracts topological features via GCNs and constructs graph-level states via global pooling for adaptive dynamic scheduling. Moreover, the introduction of attention mechanisms significantly improves scheduling decision quality. Wang et al. [29] combine GATs with self-attention MLPs, creatively mapping continuous actions to discrete ones via k-d trees. SPN-CWS [30] uses a self-attention policy network to capture global VM contexts, trained via evolutionary strategyassisted RL. SpotDAG [31] introduces self-attention with output masking to avoid invalid actions, ensuring deadlines while minimizing costs via spot instances.

Scenario-specific approaches continually enrich this field. Koslovski et al. [32] design an Actor-Critic scheduler to adaptively select heuristic policies based on real-time states. Co-ScheRRL [33] targets co-located scenarios using self-attention and DRL relational reasoning. Xue et al. [34] reduce Yarn cluster job latency by utilizing resource queue idle windows. Dong et al. [35] optimize Storm streaming workloads via weighted GNN embeddings. Shu et al. [36] propose a multipolicy MCTS approach to adaptively adjust search strategies.

In summary, while existing GRL-based scheduling approaches demonstrate immense advantages in task dependency modeling and structural feature extraction, an in-depth analysis reveals several limitations. First, most approaches assume static or single-workflow graphs, struggling to cope with complex real-world environments characterized by stochastic task arrivals and concurrent workflow execution. Furthermore, Actor and Critic networks in these frameworks frequently share the same representation layer, restricting their capability to differentially model local task features and global system states. A summary of the main related works on DRL-based approaches is presented in Table 1.

## 3. SYSTEM MODEL AND PROBLEM DEFINITION

## 3.1 System Model

This section formally defines the scheduling environment, process, constraints, and optimization objectives. Specifically, we formulate the problem of scheduling multi-instance, dependency-aware tasks onto heterogeneous virtual machine clusters. For readability, Table 2 summarizes the key notations used in the proposed system model.

(1) System Model. The system consists of a heterogeneous computing cluster and a dynamically arriving workload of jobs with precedence constraints.

Compute Cluster. The cluster comprises a set of heterogeneous Virtual Machines (VMs), denoted as $\begin{array} { r l } { \mathcal { M } } & { { } = } \end{array}$ $\left\{ M _ { 1 } , M _ { 2 } , \ldots , M _ { \left| \mathcal { M } \right| } \right\}$ . The system considers a set of |D| distinct resource dimensions, indexed by $\mathcal { D } = \{ 1 , 2 , \hdots , | \mathcal { D } | \}$ Each VM $M _ { k } \in \mathcal { M }$ is characterized by a resource capacity vector $\mathbf { C } _ { k } \ \in \ \mathbb { R } _ { + } ^ { | \mathcal { D } | }$ , where $C _ { k } ^ { d }$ represents the total capacity of VM $M _ { k }$ on resource dimension $d \in \mathcal { D }$ . At time step t, the state of VM $M _ { k }$ includes its available resource vector ${ \bf A } _ { k } ( t ) \in \mathbb { R } _ { + } ^ { | \mathcal { D } | }$ , where $A _ { k } ^ { d } ( t ) \leq C _ { k } ^ { d }$ is the amount of available resources. Additionally, each VM $M _ { k }$ possesses a processing speed coefficient $s _ { k } \in \mathbb { R } ^ { + }$ , which varies across different types of VMs.

Workload Model. The workload consists of dependencyaware tasks arriving over time. These tasks are aggregated into a set of streaming jobs, $\mathcal { I } = \{ J _ { 1 } , J _ { 2 } , \dotsc , J _ { | \mathcal { I } | } \}$ , where each job forms a DAG encapsulating their precedence constraints.

Job: Each job $J _ { i } \in \mathcal { I }$ is modeled as a DAG, $J _ { i } = ( T _ { i } , \mathcal { E } _ { i } )$ $\mathcal { T } _ { i } ~ = ~ \{ T _ { i , 1 } , T _ { i , 2 } , \ldots , T _ { i , | T _ { i } | } \}$ is the set of tasks within job $J _ { i \cdot } ~ { \mathcal { E } } _ { i } ~ \subseteq ~ { \mathcal { T } } _ { i } \times { \mathcal { T } } _ { i }$ is the set of precedence constraints. A directed edge $( T _ { i , j } , T _ { i , k } ) \in \mathcal { E } _ { i }$ indicates that task $T _ { i , j }$ must be completed before task $T _ { i , k }$ can start. The set of direct predecessors of $T _ { i , k }$ is denoted as $\mathrm { p r e d } ( T _ { i , k } )$

Task: Each task $T _ { i , j } ~ \in ~ \mathcal { T } _ { i }$ is defined by a tuple $T _ { i , j } ~ =$ $( a _ { i , j } , n _ { i , j } , d _ { i , j } , \mathbf { r } _ { i , j } )$ , where $a _ { i , j } \in \mathbb { R } _ { + }$ is the arrival time of the task, $n _ { i , j } \in \mathbb { Z } ^ { + }$ is the number of instances contained in the task, $d _ { i , j } \in \mathbb { R } _ { + }$ is the benchmark execution duration of a single instance, and $\mathbf { r } _ { i , j } \in \mathbb { R } _ { + } ^ { | \mathcal { D } | }$ is the resource demand vector of a single instance.

(2) Scheduling Process and Constraints. The scheduling process is modeled as a sequence of decisions made at a discrete event timeline $( t _ { 1 } , t _ { 2 } , \ldots , t _ { Z } )$ , where $Z$ represents the total number of decision steps. At any decision moment $t \in \{ t _ { 1 } , t _ { 2 } , \dots , t _ { Z } \}$ , the system needs to execute a scheduling action.

TABLE 1  
COMPARISON OF MAIN RELATED DRL-BASED DEPENDENCY-AWARE TASK SCHEDULING APPROACHES.
<table><tr><td>References</td><td>Topology Modeling</td><td>Representation Architecture</td><td>Resource-Task Fusion</td><td>RL Algorithm</td></tr><tr><td>Grinsztajn et al. [25]</td><td>Flat GCN</td><td>Coupled</td><td>Global Pooling &amp; Concat</td><td>Advantage Actor-Critic</td></tr><tr><td>Yu et al. [22] Liu et al. [26]</td><td>Flat GAT Bi-directional GAT</td><td>Coupled Coupled</td><td>Simple Attribute Concat Simple Attribute Concat</td><td>DQN Double DQN</td></tr><tr><td>Wang et al. [27]</td><td>Spatio-Temporal</td><td>Coupled</td><td>Set Transformer</td><td>Dueling Double DQN</td></tr><tr><td>Lin et al. [31] Our work</td><td>GNN Self-Attention</td><td>Coupled Decoupled Hierarchical</td><td>Simple Attribute Concat</td><td>Proximal Policy Optimization</td></tr></table>

TABLE 2  
SUMMARY OF KEY NOTATIONS IN SYSTEM MODEL
<table><tr><td>Notation</td><td>Definition</td></tr><tr><td> $\overline { { \mathcal { M } } }$ </td><td>Set of heterogeneous VMs in the cluster</td></tr><tr><td> $M _ { k }$ </td><td>The k-th VM in M, where  $k \in \{ 1 , \ldots , | \mathcal { M } | \}$ </td></tr><tr><td> $\mathcal { D }$ </td><td>Set of resource dimensions (e.g.,  $\{ C P U , M e m \} )$ </td></tr><tr><td> $\mathbf { C } _ { k }$ </td><td>Total resource capacity vector of VM Mk</td></tr><tr><td> ${ \bf A } _ { k } ( t )$ </td><td>Available resource vector of VM Mk at time step t</td></tr><tr><td> $s _ { k }$ </td><td>Processing speed coefficient of VM  $M _ { k }$ </td></tr><tr><td> $\mathcal { T }$ </td><td>Set of streaming jobs arriving over time</td></tr><tr><td> $J _ { i }$ </td><td>The i-th  $\mathrm { J o b , }$  modeled as a DAG  $J _ { i } = ( T _ { i } , \mathcal { E } _ { i } )$ </td></tr><tr><td> $\tau _ { i }$ </td><td>Set of tasks within Job  $J _ { i }$ </td></tr><tr><td> $\mathcal { E } _ { i }$ </td><td>Set of precedence constraints (directed edges) in Job  $J _ { i }$ </td></tr><tr><td> $T _ { i , j }$ </td><td>The j-th Task of Job Ji</td></tr><tr><td> $\mathrm { p r e d } ( \ddot { T } _ { i , j } )$ </td><td>Set of direct predecessors of Task  $T _ { i , j }$ </td></tr><tr><td> $a _ { i , j }$ </td><td>Arrival time of Task  $T _ { i , j }$ </td></tr><tr><td> $n _ { i , j }$ </td><td>Number of instances contained in Task  $T _ { i , j }$ </td></tr><tr><td> $d _ { i , j }$ </td><td>Benchmark execution duration of a single instance of</td></tr><tr><td> $\mathbf { r } _ { i , j }$ </td><td>Resource demand vector of a single instance of  $T _ { i , j }$ </td></tr><tr><td> $x _ { i , j , l , k } ( t )$ </td><td>Decision variable</td></tr><tr><td> $\underline { { \delta _ { i , j , k } } }$ </td><td>Actual execution duration of an instance on VM  $M _ { k }$ </td></tr></table>

Scheduling Decision Variables. The policy π makes a decision at time t by setting variables $x _ { i , j , l , k } ( t ) \in \{ 0 , 1 \}$ , which indicate whether the l-th instance of task $T _ { i , j }$ is assigned to VM $M _ { k }$

$x _ { i , j , l , k } ( t ) = \left\{ \begin{array} { l l } { 1 , } & { } \\ { 0 , } \end{array} \right.$ if instance l of $T _ { i , j }$ is assigned to $M _ { k }$ at t otherwise

(1)

where $l \in \{ 1 , \ldots , n _ { i , j } \}$ and $k \in \{ 1 , \ldots , | \mathcal { M } | \}$

Execution Model. The actual execution duration of an instance on VM $M _ { k } ,$ , denoted as $\delta _ { i , j , k }$ , depends on the speed coefficient of the VM.

$$
\delta _ { i , j , k } = \frac { d _ { i , j } } { s _ { k } }\tag{2}
$$

Scheduling Feasibility Constraints. To ensure the validity of scheduling decisions, the process must strictly adhere to the following task dependency and resource capacity constraints:

Precedence and Readiness Constraints. Task $T _ { i , j }$ is eligible for scheduling at time t only if it has arrived and all its predecessors have been completed. Let $F T _ { p }$ be the finish time of task $T _ { p } . \mathrm { ~ A ~ }$ decision for $T _ { i , j }$ is valid only if the following condition is met.

$$
t \geq \operatorname* { m a x } \left( \{ a _ { i , j } \} \cup \{ F T _ { p } \ | \ T _ { p } \in \mathrm { p r e d } ( T _ { i , j } ) \} \right)\tag{3}
$$

The earliest moment satisfying inequality Eq. (3) is defined as the ready time of the task, upon which the task enters the ready queue awaiting scheduling.

Resource Constraints. A scheduling decision $x _ { i , j , l , k } ( t ) =$ 1 is feasible only if the target VM has sufficient resources at the time of assignment. $\mathbf { r } _ { i , j } \leq \mathbf { A } _ { k } ( t )$ implies that for all dimensions $d \in \mathcal { D } , r _ { i , i } ^ { d } \leq A _ { k } ^ { d } ( \bar { t } )$

Resource Dynamics. Upon a feasible assignment $x _ { i , j , l , k } ( t ) ~ = ~ 1$ , the resources allocated on VM $M _ { k }$ are immediately reserved for the entire actual duration of the instance.

$$
\mathbf { A } _ { k } ( \tau ) \gets \mathbf { A } _ { k } ( \tau ) - \mathbf { r } _ { i , j } , \quad \forall \tau \in [ t , t + \delta _ { i , j , k } )\tag{4}
$$

Instance Assignment Constraints. Each instance must be successfully scheduled exactly once.

$$
\sum _ { t } \sum _ { k = 1 } ^ { | { \cal M } | } x _ { i , j , l , k } ( t ) = 1 , \quad \forall i , j , l\tag{5}
$$

The system model for dependency-aware task scheduling is illustrated in Fig. 1.

(3) Problem Definition and Optimization Objectives. The objective of the scheduling problem is to determine an optimal policy $\pi ^ { * }$ that generates a sequence of decisions to minimize a vector of performance objectives. Let $S T _ { i , j , l }$ be the start time and $F T _ { i , j , l }$ be the completion time of the l-th instance of task $T _ { i , j }$ . Note that $F T _ { i , j , l } = S T _ { i , j , l } + \delta _ { i , j , k }$ . The completion time of task $T _ { i , j }$ is $F T _ { i , j } = \mathrm { m a x } _ { l } \{ F T _ { i , j , l } \}$ . The start time of task $T _ { i , j }$ is $S T _ { i , j } = \operatorname* { m i n } _ { l } \{ S T _ { i , j , l } \}$ . The task ready time, defined as the moment the task enters the ready queue, is denoted as $a _ { i , j } ^ { r e a d y }$ . The optimization objective of the dependency-aware task scheduling problem is to find a policy π to minimize the objective function $\mathbf { F } ( \pi )$

$$
\operatorname* { m i n } _ { \pi } \mathbf { F } ( \pi ) = ( f _ { 1 } ( \pi ) , f _ { 2 } ( \pi ) , f _ { 3 } ( \pi ) )\tag{6}
$$

Objective Functions:

$f _ { 1 } ( \pi )$ (Average Task Flow Time): The average time spent by tasks in the system, from arrival to completion.

$$
f _ { 1 } ( \pi ) = \frac { 1 } { \sum _ { i } | \mathcal { T } _ { i } | } \sum _ { J _ { i } \in \mathcal { T } } \sum _ { T _ { i , j } \in \mathcal { T } _ { i } } ( F T _ { i , j } - a _ { i , j } )\tag{7}
$$

![](images/580b49ab3a40a18d367f5e3fd8fd0b014c4973796bad8f2a176c41c6ff07e404.jpg)  
Fig. 1. System Model for Dependency-Aware Task Scheduling.

$f _ { 2 } ( \pi )$ (Average Task Wait Time): The average time tasks wait in the ready queue for resources.

$$
f _ { 2 } ( \pi ) = \frac { 1 } { \sum _ { i } | T _ { i } | } \sum _ { J _ { i } \in \mathcal { I } } \sum _ { T _ { i , j } \in \mathcal { T } _ { i } } ( S T _ { i , j } - a _ { i , j } ^ { r e a d y } )\tag{8}
$$

$f _ { 3 } ( \pi )$ (Makespan): The completion time of the entire workload.

$$
f _ { 3 } ( \pi ) = \operatorname* { m a x } _ { J _ { i } \in \mathcal { T } , T _ { i , j } \in \mathcal { T } _ { i } } \{ F T _ { i , j } \}\tag{9}
$$

This problem is NP-hard. The objective is to find an optimal policy $\pi ^ { * }$ to minimize the objective function $\mathbf { F } ( \pi )$ under precedence constraints, resource constraints, and assignment constraints.

## 4. APPROACH

We propose HiGFRL, a hierarchical graph fusion-driven reinforcement learning framework designed to address the complexity of dependency-aware scheduling in heterogeneous clouds. Unlike existing approaches relying on flat state representations or loosely coupled graph embeddings, HiGFRL introduces a multi-view representation learning paradigm. It explicitly decouples the system state into static topological structures and dynamic resource contexts, which are subsequently reintegrated through a fusion-driven dual-stream network architecture.

In this section, the dependency-aware scheduling problem is first modeled as a MDP. Then, the hierarchical multi-view state abstraction and the neural network architecture designed to capture high-order interactions between task dependencies and heterogeneous resources are detailed. Finally, the end-toend joint optimization and training mechanism based on dualstream decision heads is described.

## 4.1 Graph Fusion-Driven Reinforcement Learning Model Construction

The online scheduling problem of dependency-aware tasks in heterogeneous cloud environments is formalized as a sequential decision-making process, aiming to optimize longterm system performance objectives through a series of discrete scheduling actions. The process is modeled as a MDP defined by the tuple $( S , { \mathcal { A } } , { \mathcal { R } } , { \mathcal { P } } , \gamma )$ . To handle the continuoustime dynamics of task arrivals and execution completions, this study adopts an event-driven mechanism to discretize the decision timeline. A decision step t is defined as the moment the scheduler must make an assignment decision, typically triggered by the following event: a new task instance $T _ { t a r g e t }$ satisfies all predecessor dependencies and enters the ready queue, and there are available virtual machines in the cluster meeting its resource requirements. In this framework, the reinforcement learning agent (scheduler) observes the current system state $s _ { t } \in S$ at each decision step t, selects an action $a _ { t }$ from the action space $\mathcal { A }$ according to the policy $\pi ( a _ { t } | s _ { t } )$ to assign the task instance to a specific virtual machine. Subsequently, the environment transitions to the next state $s _ { t + 1 }$ based on system dynamics and returns an immediate reward $\boldsymbol { r } _ { t } \in \mathcal { R }$ to the agent. Through this interaction process, the agent aims to learn an optimal policy $\pi ^ { * }$ to maximize the cumulative discounted return.

Fig. 2 illustrates a hierarchical graph fusion–driven RL model for dependency-aware task scheduling.

![](images/35f9d97bd152fd852c22f917732f65642d91ebe35d636f06a31da232c0c276a0.jpg)  
Fig. 2. Hierarchical Graph Fusion-Driven Reinforcement Learning Model for Dependency-Aware Task Scheduling.

Hierarchical State Space (S) State representation is crucial to the scheduling performance. For the Job-Task-Instance hierarchy, $s _ { t } ~ \in ~ S$ is defined as a hierarchical graph fusion state. This hierarchical composite representation integrates three distinct views:

(1) Static Topological View (H): Captures the long-term dependency structure. A hypergraph topology is constructed based on the DAG dependency list, using job ownership identifiers to distinguish subgraphs of different jobs. Node features consist of task static attributes (resource demands and duration) and task topological ranks, characterizing the inherent properties and critical path positions of tasks in the DAG.

(2) Dynamic Global View $( { \mathcal { G } } _ { g l o b a l } ) { : }$ : Encodes the real-time resource state across the entire cluster. This view aggregates the real-time available resources, machine resource capacities, and machine processing speeds of all computing nodes based on the current system time, forming a physical state snapshot of the cluster. Simultaneously, it combines resource load states with task state vectors and reflects global load pressure using the total number of task instances and the number of pending instances.

(3) Local Decision View $( \mathcal { G } _ { l o c a l } ) \mathrm { : }$ : Models the pairwise affinity between the target instance and candidate computing nodes. This view focuses on the current ready task, calculating dynamic waiting duration based on task ready time to measure scheduling urgency. Edge features quantify execution costs by estimating the expected completion time of the task on different nodes, while node features integrate the total number of instances and the number of pending instances for the task. Action Space (A) The action space is discrete and employs constraint-aware hard masking, $\mathcal { A } = \{ 1 , \dots , | \mathcal { M } | \}$ , where action $a _ { t } \ = \ k$ represents assigning $T _ { t a r g e t }$ to VM $M _ { k }$ . To strictly enforce resource constraints and accelerate exploration, the action space implements hard action masking, defining a binary validity mask vector $\mathbf { m } _ { t } \in \{ 0 , 1 \} ^ { | \mathcal { M } | }$ as shown in Eq. (10).

$$
\mathbf { m } _ { t } [ k ] = \mathbb { I } ( A _ { k } ^ { c p u } ( t ) \ge r _ { t a r g e t } ^ { c p u } ) \cdot \mathbb { I } ( A _ { k } ^ { m e m } ( t ) \ge r _ { t a r g e t } ^ { m e m } )\tag{10}
$$

This mask is applied to the unnormalized action scores output by the policy network, effectively constraining the action space to the feasible region $\mathcal { A } _ { v a l i d } = \{ k \ | \ \mathbf { m } _ { t } [ k ] = 1 \}$ The resource-constraint-based hard action masking mechanism imposes an inductive bias on the model to prevent the agent from learning invalid policies, improving sample efficiency in the early stages of training.

Topology-Prior-Guided Hybrid Reward (R) To address the problem of delayed feedback for long-term objectives in longhorizon scheduling, HiGFRL designs a dense hybrid reward function $R _ { t }$

$$
\begin{array} { r } { R _ { t } = \underbrace { - \alpha _ { t i m e } \cdot \delta _ { e x e c } } _ { \mathrm { M a k e s p a n ~ M i n i m i z a t i o n } } + \underbrace { \lambda _ { e f t } \cdot \frac { \mathrm { m i n } _ { k ^ { \prime } } \mathrm { E F T } _ { k ^ { \prime } } } { \mathrm { E F T } _ { a _ { t } } } } _ { \mathrm { C o m p l e t i o n ~ E f f i c i e n c y } } } \\ { + \underbrace { \lambda _ { r a n k } \cdot \mathrm { R a n k } ( T _ { t a r g e t } ) } _ { \mathrm { T o p o l o g i c a l ~ P r i o r i t y } } } \end{array}\tag{11}
$$

where $\delta _ { e x e c }$ is the task duration. The Completion Efficiency Incentive term encourages the agent to prefer nodes with earlier completion times in local decisions, thereby improving immediate execution efficiency. Here, $\mathrm { E F T } _ { a _ { t } }$ represents the Earliest Finish Time of the task under the selected action ${ { a } _ { t } } ,$ and mi $\boldsymbol { \mathrm { 1 } } _ { k ^ { \prime } } \mathrm { E F T } _ { k ^ { \prime } }$ represents the minimum possible finish time achievable for the task among all feasible VMs. This ratio guides the model towards selecting currently locally optimal resources. The Topological Priority term utilizes the task’s upward rank calculated from static graph analysis, i.e., the critical path length from the task to the DAG exit node, as prior knowledge. This enables the agent to identify and prioritize bottleneck tasks on the critical path, thereby optimizing the global job completion time.

## 4.2 Multi-View Representation Learning and Fusion Architecture

The core challenge of dependency-aware task scheduling lies in effectively modeling the complex interactions between task topological dependencies and machine resource heterogeneity. HiGFRL addresses this issue through a decoupling and fusion strategy, extracting features independently under different views and then performing multi-view fusion through a hierarchical architecture. The overall framework of the HiGFRL approach is illustrated in Fig. 3.

## (1) Hierarchical Graph-Fused State Construction

HiGFRL constructs a three-layer graph structure to capture system features at different semantic levels and granularities.

Level 1: Static Task Dependency Hypergraph (H). To effectively capture high-order dependencies, HiGFRL models job topology as a hypergraph. Its formal definition is as follows.

$$
\mathcal { H } = ( \mathcal { V } _ { t a s k } , \mathcal { E } _ { h y p e r } )\tag{12}
$$

where each node $v _ { i } \in \mathcal { V } _ { t a s k }$ is characterized by a static feature vector $\mathbf { x } _ { i } ^ { s t a t i c }$ containing resource requests, task duration, and topological rank. Explicitly introducing topological rank aims to provide the model with prior knowledge about the task’s global position in the DAG.

![](images/593c31d614fca18f19ebcb597dc75255097d9b29d166936b0e5a7bc3e7a2ea3d.jpg)  
Fig. 3. Overall Framework of HiGFRL.

Predecessor-Aggregation-Based Hyperedge Construction: HiGFRL adopts a Many-to-One dependency aggregation paradigm to construct hyperedges, where each hyperedge $e \in \mathcal { E } _ { h y p e r }$ connects a child task node and all its parent task nodes. This structure explicitly models the synchronization constraint in dependency-aware task scheduling, where a task becomes ready only after all predecessors are completed. Through hypergraph convolution, HiGFRL can directly aggregate features from all predecessors in a single message passing step, thereby efficiently inferring the task’s readiness state and avoiding the information dilution problem of ordinary graph convolution when processing long dependency chains.

Level 2: Dynamic Global Cluster Graph $( \mathcal { G } _ { g l o b a l } )$ . This view aims to capture the global real-time state of the heterogeneous cluster. To adapt to the dynamic characteristics of random task flow arrival and completion in dependency-aware task scheduling scenarios, this graph employs a construction strategy combining static topology with dynamic attributes. The formal definition of $\mathcal { G } _ { g l o b a l }$ is as follows.

$$
\mathcal { G } _ { g l o b a l } = ( \mathcal { V } _ { a l l } , \mathcal { E } _ { d e p } , \mathbf { M } _ { t } )\tag{13}
$$

where $\mathcal { V } _ { a l l } = \mathcal { V } _ { t a s k } \cup \mathcal { V } _ { v m }$ covers all task nodes and computing nodes, and $\mathcal { E } _ { d e p }$ reuses the inherent dependency structure of job DAGs. The dynamic evolution of this view is driven by the following mechanisms:

Real-time Feature Stream Evolution: Node attributes in the graph update in real-time with time step t. Task node features $\mathbf { x } ^ { d y n }$ encode current execution state, resource demands, and completion progress; VM node features $\mathbf { x } ^ { v m }$ reflect the realtime CPU and memory load rates of each node. This timevarying feature mapping ensures that graph data can characterize the physical state of the environment without latency.

Dynamic Validity Mask (M<sub>t</sub>): Addressing the dynamic arrival characteristic of jobs, HiGFRL introduces a binary mask vector $\mathbf { M } _ { t } ~ \in ~ \{ 0 , 1 \} ^ { | \mathcal { V } _ { a l l } | }$ . This mask is dynamically generated by the environment based on task arrival times and completion states. For task nodes that have not yet arrived or have terminated, their corresponding mask values are set to 0. This mechanism allows the model to maintain a fixed graph size while logically filtering out invalid nodes dynamically, providing a precise view of active nodes for subsequent feature aggregation.

Level 3: Local Bipartite Matching Graph $( \mathcal { G } _ { l o c a l } )$ . This view aims to optimize immediate scheduling decisions through relevance scoring from a local perspective. $\mathcal { G } _ { l o c a l }$ is defined as a bipartite graph connecting the target task node $u _ { t a r g e t }$ with all candidate VM nodes $\nu _ { v m }$

$$
\mathcal { G } _ { l o c a l } = ( \{ u _ { t a r g e t } \} \cup \mathcal { V } _ { v m } , \mathcal { E } _ { p a i r } )\tag{14}
$$

where the edge $( u _ { t a r g e t } , v _ { v m } ^ { k } ) \in \mathcal { E } _ { p a i r }$ represents a potential assignment path.

Candidate Set Focusing Mechanism: The core purpose of this view is to construct explicit pairwise correlation representations. By explicitly isolating and reinforcing the interaction features between target task demands and specific VM capabilities, this structure forces the model to directly evaluate the affinity of each (Task, VM) matching pair. This not only reduces the interference of global noise on local decisions but also effectively decouples the complexity of action evaluation, ensuring the policy network can capture fine-grained resource matching patterns.

## (2) Cross-Level Feature Fusion and Dual-Stream Decision

HiGFRL proposes a fusion-driven dual-network architecture that hierarchically injects information: from the Static Topological View H to the Dynamic Global View $\mathcal { G } _ { g l o b a l }$ , and finally converging at the Local Decision View $\mathcal { G } _ { l o c a l }$

Stage 1: Static Topological Feature Extraction

First, a hypergraph convolutional network is utilized to process the Static Task Dependency Hypergraph $\begin{array} { r l } { { \mathcal { H } } } & { { } = } \end{array}$ $( \mathcal { V } _ { t a s k } , \mathcal { E } _ { h y p e r } )$ to capture long-range dependencies and synchronization constraints between tasks. Let $\mathbf { X } ^ { \mathcal { H } }$ be the node feature matrix. HiGFRL represents hyperedge connections $\mathcal { E } _ { h y p e r }$ as an incidence matrix $\mathbf { H } \in \mathbb { R } ^ { | \hat { \mathcal { V } } _ { t a s k } | \times | \mathcal { E } _ { h y p e r } | }$ , where $H _ { i e } = 1$ indicates node $v _ { i }$ is contained in hyperedge e. The static topological embedding $\mathbf { Z } ^ { \mathcal { H } }$ is calculated following Eq. (15).

$$
\mathbf { Z } ^ { \mathcal { H } } = \sigma \left( \mathbf { D } _ { \boldsymbol { \nu } } ^ { - \frac { 1 } { 2 } } \mathbf { H } \mathbf { W } \mathbf { D } _ { \boldsymbol { \varepsilon } } ^ { - 1 } \mathbf { H } ^ { \top } \mathbf { D } _ { \boldsymbol { \nu } } ^ { - \frac { 1 } { 2 } } \mathbf { X } ^ { \mathcal { H } } \boldsymbol { \Theta } \right)\tag{15}
$$

where Θ is the learnable filter weight matrix. $\mathbf { D } _ { \mathcal { V } }$ and $\mathbf { D } _ { \mathcal { E } }$ are diagonal matrices of node degrees and hyperedge degrees, respectively, used for normalization. W is the hyperedge weight matrix. This formula implements a ”node-hyperedgenode” high-order information propagation path via $\bar { \mathbf { H D } _ { \mathcal { E } } ^ { - 1 } } \bar { \mathbf { H } ^ { \top } }$ The generated $\mathbf { Z } ^ { \mathcal { H } }$ integrates the task’s global topological priors and high-order dependency semantics.

## Stage 2: Global Dynamic Context Aggregation

Next, the Global State Encoder is utilized to process the Dynamic Global Cluster Graph $\mathcal { G } _ { g l o b a l }$ . In this stage, the model first performs cross-view feature fusion.

Specifically, for task nodes, their features are formed by concatenating the dynamic state $\mathbf { x } ^ { d y n }$ with the static topological embedding $\mathbf { Z } ^ { \mathcal { H } }$ extracted in Stage 1; for VM nodes, features are constituted by real-time load features $\mathbf { x } ^ { v m }$ after dimension alignment. Together, they form the node feature matrix $\mathbf { X } ^ { \mathcal { G } _ { g l o b a l } }$ of the global graph. This operation injects long-range dependency priors into the real-time state, enhancing the model’s ability to anticipate the subsequent impact of tasks.

To capture the complex asymmetric interactions between task urgency and heterogeneous resource loads, the encoder employs a Dynamic Graph Attention mechanism. Taking the fused feature matrix $\mathbf { X } ^ { \mathcal { G } _ { : } }$ <sup>global</sup> as input, it generates the node embedding matrix $\mathbf { H } ^ { g l o b a l }$ containing context information.

$$
{ \bf H } ^ { g l o b a l } = { \bf G } { \bf A } { \bf T } _ { g l o b a l } ( { \bf X } ^ { \mathcal { G } _ { g l o b a l } } , { \mathcal { E } } _ { d e p } )\tag{16}
$$

Subsequently, the encoder uses the mask $\mathbf { M } _ { t }$ to perform a masked global pooling operation on the node vectors in $\mathbf { H } ^ { g l o b a l }$ , generating the global context vector $\mathbf { h } _ { g l o b a l }$ as Eq. (17).

$$
\mathbf { h } _ { g l o b a l } = \frac { \sum _ { i = 1 } ^ { | \mathcal { V } _ { a l l } | } ( \mathbf { M } _ { t } [ i ] \cdot \mathbf { h } _ { i } ^ { g l o b a l } ) } { \sum _ { i = 1 } ^ { | \mathcal { V } _ { a l l } | } \mathbf { M } _ { t } [ i ] + \epsilon }\tag{17}
$$

where $\mathbf { h } _ { i } ^ { g l o b a l }$ is the row vector of the i-th node in matrix $\mathbf { H } ^ { g l o b a l }$ , representing the high-dimensional feature of node i after attention aggregation; ϵ is a smoothing term to prevent division by zero. By combining the dynamic graph attention mechanism with masked pooling, this module can dynamically aggregate effective resource load distribution information of the entire cluster while shielding invalid node noise, thereby precisely capturing the global state of the system.

## Stage 3: Dual-Stream Decision Heads

The architecture splits into two specialized streams sharing the global context $\mathbf { h } _ { g l o b a l }$

Global State Evaluator: Composed of the Global State-Value Critic network $V _ { \phi }$ (GSV Critic). This module estimates the value of the current state $V ( s _ { t } )$ . Taking the global vector $\mathbf { h } _ { g l o b a l }$ as input, it maps the global system state to the expected long-term return via a Multilayer Perceptron (MLP), serving as a baseline for calculating the advantage function. The estimation of $V ( s _ { t } )$ is calculated as shown in Eq. (18).

$$
V ( s _ { t } ) = \mathbf { M } \mathbf { L } \mathbf { P } _ { c r i t i c } ( \mathbf { h } _ { g l o b a l } )\tag{18}
$$

Context Fusion Allocator: Composed of the context-aware matching actor network $\pi _ { \theta }$ (CAM Actor). This module executes specific instance-to-node assignments. It operates on the local bipartite graph $\mathcal { G } _ { l o c a l }$ , where $\bar { \mathbf X } ^ { \mathcal G _ { l o c a l } }$ represents the initial node feature matrix of the graph, containing basic resource attributes and heuristic time estimation features of the target task and candidate VMs. The local graph attention module extracts pairwise local matching features as follows.

$$
\mathbf { Z } ^ { \mathcal { G } _ { l o c a l } } = \mathbf { G } \mathbf { A } \mathbf { T } _ { l o c a l } ( \mathbf { X } ^ { \mathcal { G } _ { l o c a l } } , \mathcal { E } _ { p a i r } )\tag{19}
$$

To break the limitation of the local view and integrate global context, a cross-level fusion mechanism is introduced. For each candidate VM $M _ { k } ,$ , a multi-view fused feature vector ${ \bf e } _ { k } ^ { f u s e d }$ is constructed as follows.

$$
{ \mathbf e } _ { k } ^ { f u s e d } = \mathrm { C o n c a t } \left( { \mathbf h } _ { k } ^ { l o c a l } , ~ \mathcal { B } ( { \mathbf h } _ { g l o b a l } ) , ~ { \mathbf x } _ { k } ^ { r a w } \right)\tag{20}
$$

where $B ( \cdot )$ denotes the broadcast operation, extending the global context vector $\mathbf { h } _ { g l o b a l }$ to match the number of candidate nodes, and Concat(·) denotes feature dimension concatenation. In this fused representation, $\mathbf { Z } ^ { \mathcal { G } _ { l o c a l } }$ represents the feature embedding space of the local bipartite graph, $\mathbf { h } _ { k } ^ { l o c a l } \in \mathbf { Z } ^ { \mathcal { G } }$ local is the local matching feature of the k-th candidate ${ \mathrm { V M } } ,$ aiming to quantify the pairwise correlation between this node and the target task; $\mathbf { h } _ { g l o b a l }$ aligns with the local representation of each candidate VM via the broadcast mechanism, providing unified global resource posture guidance; $\mathbf { x } _ { k } ^ { r a w } \in \mathbf { X } ^ { r a w }$ is the raw feature vector of the k-th VM directly extracted from the input matrix $\mathbf { X } ^ { \mathcal { G } _ { l o c a l } }$ via raw feature injection. This mechanism aims to alleviate the potential over-smoothing problem of deep GNNs, preventing node embeddings from losing precise numerical information after multi-layer aggregation. Since resource scheduling must strictly satisfy resource capacity constraints, explicitly introducing raw features provides the decision layer with a lossless view of node states, ensuring precise perception of remaining node capacity and compliant allocation.

Finally, the fused feature vector ${ \bf e } _ { k } ^ { f u s e d }$ is input into an MLP to generate a scalar score $l _ { k } .$ This score quantifies the potential value of assigning the current task to VM $M _ { k }$ . Subsequently, combining with the validity mask $\mathbf { m } _ { t }$ and applying the Softmax function, the final policy distribution $\pi _ { \boldsymbol { \theta } } \big ( a _ { t } | \boldsymbol { s } _ { t } \big )$ is calculated as Eq. (21).

$$
l _ { k } = \mathbf { M } \mathbf { L } \mathbf { P } _ { a c t o r } ( \mathbf { e } _ { k } ^ { f u s e d } )\tag{21}
$$

$$
\pi _ { \theta } ( a _ { t } = k | s _ { t } ) = { \frac { \mathbf { m } _ { t } [ k ] \cdot \exp ( l _ { k } ) } { \sum _ { j = 1 } ^ { | { \mathcal { M } } | } \mathbf { m } _ { t } [ j ] \cdot \exp ( l _ { j } ) } }\tag{22}
$$

## 4.3 Algorithm Training and Policy Optimization

HiGFRL adopts an end-to-end joint optimization framework based on the Global State Evaluator and Context Fusion Allocator. This framework integrates the aforementioned hybrid GNN architecture, realizing a direct mapping from high-dimensional multi-view graph structures to scheduling decisions. By jointly optimizing the context-aware matching actor network $\pi _ { \theta }$ and the Global State-Value Critic network $V _ { \phi }$ within a shared latent feature space, HiGFRL enables efficient policy iteration in complex heterogeneous environments.

## (1) Value Baseline Approximation

The GSV Critic aims to approximate the value function $V ( s _ { t } )$ of the current system state to reduce the variance of policy gradient estimation. Based on the global context vector $\mathbf { h } _ { g l o b a l }$ , the GSV Critic outputs a prediction of the cumulative discounted return. Generalized Advantage Estimation (GAE) is used to calculate the advantage function ${ \hat { A } } _ { t } ,$ which effectively balances bias and variance through a trade-off parameter λ. The TD error $\delta _ { t }$ is calculated as follows.

$$
\delta _ { t } = r _ { t } + \gamma V ( s _ { t + 1 } ) - V ( s _ { t } )\tag{23}
$$

The advantage estimate $\hat { A } _ { t }$ is the exponentially weighted accumulation of TD errors.

$$
\hat { A } _ { t } = \sum _ { l = 0 } ^ { \infty } ( \gamma \lambda ) ^ { l } \delta _ { t + l }\tag{24}
$$

The parameters $\phi$ of the GSV Critic network are updated by minimizing the mean squared error, ensuring the value baseline accurately reflects the long-term evolution of the global resource posture.

$$
L ^ { C r i t i c } ( \phi ) = \hat { \mathbb { E } } _ { t } \left[ ( V _ { \phi } ( s _ { t } ) - V _ { t } ^ { t a r g e t } ) ^ { 2 } \right]\tag{25}
$$

## (2) Trust Region Policy Iteration

The CAM Actor outputs the action probability distribution $\pi _ { \boldsymbol { \theta } } \big ( \boldsymbol { a } _ { t } | \boldsymbol { s } _ { t } \big )$ based on the multi-view fused feature ${ \bf e } _ { k } ^ { f u s e d }$ . To ensure monotonicity and stability of the training process and avoid performance collapse caused by excessively large steps, a clipping mechanism is employed to limit the magnitude of policy updates. Defining the probability ratio of the new policy to the old policy as $\begin{array} { r } { r _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { t } | s _ { t } \right) } { \pi _ { \theta _ { o l d } } \left( a _ { t } | s _ { t } \right) } } \end{array}$ , the policy objective function is designed as Eq. (26).

$$
L ^ { A c t o r } ( \theta ) = \hat { \mathbb { E } } _ { t } \left[ \operatorname* { m i n } \left( r _ { t } ( \theta ) \hat { A } _ { t } , \ \mathrm { c l i p } ( r _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } \right) \right]\tag{26}
$$

This objective function constrains policy updates within a trust region $[ 1 - \epsilon , 1 + \epsilon ]$ via the clip(·) operation, forcing the model to maintain the stationarity of the policy distribution while optimizing returns.

## (3) Dynamic Exploration-Exploitation Balance

To prevent the model from prematurely converging to suboptimal local optima, an entropy regularization term is introduced into the optimization objective. The entropy of the policy distribution $S [ \pi _ { \theta } ] ( s _ { t } )$ serves as part of the reward signal to encourage the agent to maintain action diversity. The total optimization objective combines policy improvement, value approximation, and entropy maximization as follows.

$$
L ( \theta , \phi ) = \hat { \mathbb { E } } _ { t } \left[ L ^ { A c t o r } ( \theta ) - c _ { 1 } L ^ { C r i t i c } ( \phi ) + \beta _ { t } S [ \pi _ { \theta } ] ( s _ { t } ) \right]\tag{27}
$$

where the entropy coefficient $\beta _ { t }$ follows a linear decay schedule.

$$
\beta _ { t } = \operatorname* { m a x } ( \beta _ { e n d } , \beta _ { s t a r t } - \frac { t } { T _ { m a x } } ( \beta _ { s t a r t } - \beta _ { e n d } ) )\tag{28}
$$

This dynamic adjustment mechanism enables the agent to possess higher entropy weight in the initial training stage for broad exploration of the solution space, and gradually reduce the weight in later stages to focus on refined exploitation of high-value regions, thereby achieving an adaptive balance between exploration and exploitation.

## 5. PERFORMANCE EVALUATION

HiGFRL is implemented based on Python 3.8.18 and Py-Torch 1.13.1 with CUDA support, and its GNN components are implemented via PyTorch Geometric 2.0.4. All experiments are conducted on a cloud server equipped with a 20- core CPU, 80 GB of system memory, and an NVIDIA GeForce RTX 4090 GPU. This section presents experimental results and analyses aiming to answer the following research questions. The source code and datasets of HiGFRL are available at https://github.com/igeng/HiGFRL.

• RQ1 (Training Convergence Performance: How does the convergence efficiency of HiGFRL compare to other DRL-based baselines during the training phase, and does it demonstrate superior stability?

• RQ2 (Scheduling Effectiveness): How effective is HiGFRL in optimizing overall scheduling performance, particularly in minimizing Makespan and Average Task Flow Time, compared to heuristics and DRL-based approaches under different workload scales?

• RQ3 (System Operational Efficiency): How does HiGFRL perform in optimizing system operational efficiency under different workload scales, specifically in terms of minimizing scheduling overheads including Total Retry Counts and Average Task Wait Times?

• RQ4 (Ablation Analysis): How does the hierarchical graph fusion mechanism contribute to the overall system performance, and what is its necessity?

## 5.1 Experimental Setup

Cluster Resource Configuration. In the training phase, the experiment sets up a cloud computing cluster consisting of 40 VMs. As summarized in Table 3, four VM types are considered, each with different resource configurations.

Dependency–aware Task Workload Patterns. Experiments are based on the Alibaba Cluster Trace v2018 dataset, which contains real production workload traces collected from approximately 4000 machines over 8 days. Tasks with explicit DAG structures are selected from batch job workloads to evaluate dependency-aware task scheduling and policy learning performance. Specifically, jobs are first sampled from the workload trace dataset, consisting of tasks of varying sizes, to form a training set containing a total of 1000 jobs and 4076 tasks. Subsequently, to further evaluate the adaptability of the approach, three scales of workload patterns are designed, with two different maximum instance configurations for each pattern, resulting in six experimental scenarios as follows:

TABLE 3  
CLUSTER RESOURCE CONFIGURATION DETAILS
<table><tr><td>Types</td><td>CPU cores</td><td>Memory capacity</td><td>Quantity</td><td>Processing speed</td></tr><tr><td>1</td><td>2</td><td>4</td><td>10</td><td>2</td></tr><tr><td>2</td><td>4</td><td>8</td><td>10</td><td>4</td></tr><tr><td>3</td><td>8</td><td>16</td><td>10</td><td>8</td></tr><tr><td>4</td><td>16</td><td>32</td><td>10</td><td>16</td></tr></table>

• Standard-scale pattern. 300 jobs, 2160 tasks, maximum instance counts of 50 and 100, referred to as Standard-50 and Standard-100.

• Medium-scale pattern. 600 jobs, 4417 tasks, maximum instance counts of 50 and 100, referred to as Medium-50 and Medium-100.

• Large-scale pattern. 900 jobs, 6704 tasks, maximum instance counts of 100 and 250, referred to as Large-100 and Large-250.

In the experiments, the six experimental configurations will be directly referenced using the above abbreviations for ease of description and comparison.

Baselines. We evaluate HiGFRL against six approaches, including three classic heuristics approaches and three advanced approaches combining GNNs and DRL:

• Random. Assigns tasks to a randomly selected VM from the set of currently feasible nodes.

• Round Robin. Allocates tasks to VMs in a fixed cyclic order, leaving tasks queued for future retries if the target VM lacks sufficient resources.

• Greedy-Tetris. A multi-dimensional bin-packing heuristic that selects the VM maximizing the dot product between task resource demands and available VM resources to minimize resource fragmentation [37].

• READYS. A representative Actor-Critic approach utilizing GCNs for DAG topological feature extraction, applying global pooling to aggregate node embeddings into a graph-level state [25].

• GA-DRL. A value-based DRL-based approach integrating Bi-directional GATs with Double DQN, aggregating predecessor and successor information to capture complex DAG dependencies for placement decisions [26].

• MODRL. A D3QN-based hybrid framework utilizing GAT, LSTM, and Set Transformer to respectively capture topological, sequential, and cluster features, enhanced by prioritized experience replay for stable multi-objective optimization [27].

Evaluation metrics. To comprehensively evaluate the performance of HiGFRL against baselines in complex scheduling scenarios, four metrics are selected: Makespan, Average Task Flow Time, Average Task Wait Time, and Total Retry Count.

These metrics comprehensively measure the system from the dimensions of efficiency, responsiveness, and scheduling stability.

## (1) Makespan

Makespan represents the total time required to complete the execution of the entire workload. It is defined as the completion time of the last finished task in the system minus the start time. A smaller Makespan indicates higher throughput and better overall efficiency of the cluster. Let $\tau$ be the set of all completed tasks, and $C _ { i }$ be the completion time of task i. The Makespan is mathematically formulated as follows.

$$
M K = \operatorname* { m a x } _ { i \in \mathcal { T } } ( C _ { i } ) - T _ { s t a r t }\tag{29}
$$

where $T _ { s t a r t }$ represents the task arrival start time.

(2) ATFT (Average Task Flow Time)

Task flow time measures the total duration a task spends in the system, covering waiting time in the ready queue and actual execution time on the virtual machine, reflecting system responsiveness from the user’s perspective. ATFT is calculated by averaging the flow times of all successfully completed tasks. Let $A _ { i }$ be the arrival time of task i. ATFT is defined as Eq. (30).

$$
A T F T = \frac { 1 } { | T | } \sum _ { i \in \mathcal { T } } ( C _ { i } - A _ { i } )\tag{30}
$$

where $| \tau |$ represents the total number of completed tasks. A lower ATFT implies tasks are processed and completed faster after submission.

(3) ATWT (Average Task Wait Time)

To further analyze the source of latency, ATWT is used to measure the scheduling delay. It is defined as the time interval between the task arrival moment and the moment the scheduler first attempts to assign it to a VM. Unlike flow time, this metric specifically quantifies queuing delays caused by dependency constraints or resource contention before task execution. Let $S _ { i } ^ { f i r s t }$ be the timestamp of the first scheduling attempt for task i. The ATWT is calculated as follows. Minimizing ATWT implies that the scheduler is efficient in handling pending tasks and reducing queue congestion.

$$
A T W T = \frac { 1 } { | T | } \sum _ { i \in \mathcal { T } } ( S _ { i } ^ { f i r s t } - A _ { i } )\tag{31}
$$

(4) TRC (Total Retry Count)

In a resource-constrained cluster, a scheduling decision may fail if the selected VM does not have sufficient available CPU or Memory at the specific moment of assignment. When this occurs, the task is rejected and must be re-queued for a later attempt. The TRC aggregates the number of such failed scheduling attempts across the entire scheduling process. Let $R _ { i }$ be the number of failed attempts for task i. The TRC is defined as follows.

$$
T R C = \sum _ { i \in { \mathcal { T } } _ { a l l } } R _ { i }\tag{32}
$$

where $\mathcal { T } _ { a l l }$ represents all tasks in the workload. A lower TRC indicates that the approach is making more accurate decisions

by effectively respecting resource constraints, thereby reducing scheduling overhead and wasted computational cycles.

## 5.2 Training Convergence Performance Analysis

Fig. 4 presents a comparative analysis of the cumulative reward per episode convergence curves for HiGFRL and three DRL-based baselines: GA-DRL, MODRL, and READYS. As illustrated, HiGFRL exhibits superior convergence efficiency and asymptotic stability throughout the training phase. Starting with a robust initial performance, HiGFRL demonstrates a rapid ascent within the first 20 episodes, quickly stabilizing at a high-level convergence value of approximately 93400.

In contrast, the baselines struggle to achieve comparable optimality, highlighting significant deficiencies in how they integrate resource heterogeneity into their state representations. GA-DRL shows a steady but slow improvement, eventually plateauing around 91400. While its Bi-GAT mechanism effectively captures task dependencies, the architecture structurally treats resource states as isolated feature vectors rather than integrating them into the graph structure. This limits the agent’s ability to learn complex Task-VM affinities within the graph context. Furthermore, its ϵ-greedy policy restricts exploration efficiency in high-dimensional action spaces, contributing to its slower convergence rate.

MODRL exhibits significant numerical oscillation and suboptimal convergence trends, fluctuating significantly between 70000 and 90000 during the early stages of training. This instability is partly attributed to the high variance characteristics of value-based approaches in highly stochastic environments. More critically, its graph embedding approach tends to over-compress resource information into a single global embedding, obscuring the fine-grained availability details of heterogeneous nodes and limiting the precision of scheduling decisions. In the later stages of training, MODRL converges to a level similar to that of GA-DRL. Meanwhile, READYS displays early performance saturation, hovering around 89000 with limited policy improvement in subsequent stages. This suggests that its basic GCN architecture focuses primarily on topological encoding, and its combination with the actor-critic mechanism often treats resources as static attributes rather than interactive graph nodes, failing to effectively capture the finegrained interactions between complex task dependencies and heterogeneous resources.

The superior performance of HiGFRL strongly validates the effectiveness of its hierarchical graph fusion architecture. Unlike baselines that tend to marginalize or over-compress resource information during graph construction, HiGFRL constructs a multi-view state representation that explicitly fuses task topology and resource heterogeneity at both global and local levels. This precise structured information and semantic context enable the agent to make more informed scheduling decisions. Additionally, the synergy between this architecture and the linear entropy decay strategy ensures a dynamic balance between broad exploration and stable exploitation, effectively preventing premature policy convergence and helping the agent explore better scheduling policies.

![](images/d1b673a0f2899bdc53d32d6ac6d9393db23375604516587612358c4c8b27bb4f.jpg)  
Fig. 4. Comparison of cumulative reward per episode convergence curves across DRL-based approaches.

## 5.3 Overall Scheduling Effectiveness

To rigorously evaluate the scheduling efficiency of HiGFRL relative to the six baseline approaches, the performance of different approaches on Makespan and ATFT is analyzed. Fig. 5 to Fig. 7 present the comparative results for Makespan, while Fig. 8 to Fig. 10 illustrate the results for ATFT under three different scales of workload patterns, respectively. Experimental results indicate that HiGFRL achieves optimal or near-optimal scheduling performance in all test scenarios, and its performance advantage significantly expands as workload scale and scheduling difficulty increase.

In the standard-scale scenarios of Standard-50 and Standard-100, HiGFRL exhibits extremely stable performance in the Makespan metric, with values of 550714 and 550822, respectively. These results are on par with the best-performing heuristic approach, Greedy-Tetris, and GA-DRL, while significantly outperforming READYS at 561816 and MODRL at 568460. However, Makespan alone does not fully reflect differences in scheduling quality. Observing the ATFT metric in Fig. 8, it is evident that even when the Makespan gap is narrow, HiGFRL maintains a lead in service quality. For instance, in Standard-50, HiGFRL’s average flow time is 519.50, distinctly lower than Greedy-Tetris’s 565.37 and GA-DRL’s 576.19. This indicates that while several strong baselines can complete the entire workload in a similar timeframe, HiGFRL prioritizes critical tasks more effectively through superior resource matching, thereby reducing the average residence time of tasks in the system.

As the workload scale expands to medium levels, the advantages of HiGFRL become more pronounced. As shown in Fig. 6, in the high-concurrency Medium-100 scenario, basic approaches like RoundRobin and Random suffer a performance collapse due to frequent resource conflicts, with Makespan soaring above 3000000. At this point, GA-DRL also exhibits instability, reaching a Makespan of 800645. In contrast, HiGFRL maintains the Makespan at a low level of 561999, significantly outperforming MODRL’s 637537. Regarding the ATFT metric in Fig. 9, HiGFRL achieves an outstanding result of 1635.77 in the Medium-100 scenario, which is 9.43% lower than the second-best Greedy-Tetris and

![](images/55fe2dfdef6dbc61d4b1fa7591cf5463a3092f36d52f4482c99e0e915aebd444.jpg)  
Fig. 5. Makespan comparison of different approaches under standard-scale workload patterns.

14.04% lower than GA-DRL. This widening gap is primarily attributed to the hierarchical graph fusion architecture of HiGFRL. Unlike GA-DRL and READYS, which are prone to feature over-smoothing on larger scale graphs, HiGFRL explicitly models high-order dependencies via hypergraph convolution and preserves feature differences of heterogeneous resources under a global view. This allows it to accurately identify critical paths affecting overall scheduling even as task volume surges.

In large-scale scenarios, HiGFRL demonstrates decisive performance dominance. As illustrated in Fig. 7, in the most complex Large-100 scenario, HiGFRL’s Makespan is only 1198246, whereas the closest competitors, GA-DRL and Greedy-Tetris, reach values as high as 1776491 and 1777151, respectively. HiGFRL achieves a 32.55% reduction in Makespan in this scenario. More notably, regarding the QoS metrics shown in Fig. 10, under the extreme load configuration of Large-250, HiGFRL controls the ATFT at 24158.44, while all other comparison approaches exceed 27000. In this scenario, HiGFRL achieves a 13.58% performance improvement compared to the second-best MODRL, 27953.26. This result fully proves the robustness of HiGFRL under extreme pressure. This is mainly attributed to HiGFRL’s unique fusiondriven dual-stream network design, where the global state evaluator, based on joint encoding of static topology and dynamic load, provides precise long-term value prediction, guiding the context fusion allocator to perform efficient policy iteration in a massive action space. Coupled with the adaptive entropy regularization mechanism, this architecture effectively avoids premature policy convergence in large-scale state spaces. Consequently, it maintains a globally optimal decision-making vision throughout long scheduling sequences, overcoming the limitations of other graph RL baselines that are prone to falling into local optima in complex scenarios.

## 5.4 Operational Efficiency and Stability Analysis

To thoroughly investigate the operational efficiency and stability of the system under varying workloads, we further analyzed the ATWT and TRC. These two metrics directly reflect queue congestion levels and the precision of resource matching. The comparative results for ATWT are shown in Fig. 11 to Fig. 13, and the statistics for TRC are presented in Table 4. Experimental data consistently indicate that HiGFRL has a significant advantage in optimizing operational efficiency.

![](images/9c30480b66199f6f57950c9692fb87ad9dff93b4fb3b5b17c46f3e887bf66dac.jpg)  
Fig. 6. Makespan comparison of different approaches under medium-scale workload patterns.

![](images/27efc11e4624772cf8b1d3d4d5371c00414a495839c051b073fd2a3cf855caa1.jpg)  
Fig. 7. Makespan comparison of different approaches under large-scale workload patterns.

![](images/90ad041b92e9a661cc3ac2b63602faf7f2f2ae97fd2456d3c852b3d018e70246.jpg)  
Fig. 8. Average task flow time comparison of different approaches under standard-scale workload patterns.

In terms of the ATWT metric, HiGFRL maintains the lowest record across all six experimental configurations, with the margin of superiority expanding as the load increases. Specifically, in the Standard-50 scenario, HiGFRL compresses the average wait time to 276.30, superior to 306.02 achieved by Tetris and 311.34 by GA-DRL. When the scenario switches to Large-100, HiGFRL records an ATWT of only 3069.95, representing a reduction in queuing latency of hundreds of seconds compared to GA-DRL at 3502.71 and READYS at 3856.87. This lowlatency characteristic benefits primarily from the combination of HiGFRL’s global dynamic state encoder and local bipartite graph. Unlike READYS which uses GCNs for indiscriminate average aggregation of neighbor node features, HiGFRL’s local graph explicitly encodes interaction features between tasks and specific VMs, combined with dynamic weighting via attention mechanisms. This allows the scheduler to capture the dynamic distribution of resource fragments and available time windows in the cluster in real-time, thereby rapidly dispatching tasks from the ready queue and greatly alleviating queue backlog. In the most challenging Large-250 scenario in Fig. 13, the gap is particularly evident. HiGFRL’s average wait time is 15234.66, while the best-performing baseline algorithm, Tetris, is 17672.55. HiGFRL reduces wait time by 13.79%, directly attributed to its local bipartite graph matching mechanism’s acute capture of immediate resource fragments.

![](images/20a2044185456e7b2cd1a785982c85c790ed7df10cbe88e12a8f0aa598ab7b22.jpg)  
Fig. 9. Average task flow time comparison of different approaches under medium-scale workload patterns.

![](images/3855d2f6fd8936605244786f2f56fb48dde067f02ffb3500fd77ab40eb7d69fb.jpg)  
Fig. 10. Average task flow time comparison of different approaches under large-scale workload patterns.

Regarding scheduling stability, the TRC reflects the approach’s adherence to resource constraints. As shown in Table 4, HiGFRL achieves the lowest retry counts in the majority of high-load scenarios, including Standard-100, Medium-50,

![](images/b20791dfd61e828439eee86d6215636127bf79ac35c671c0b8bae5e5976a018b.jpg)  
Fig. 11. Average task wait time comparison of different approaches under standard-scale workload patterns.

![](images/208d47c50f6e17b838fd5b2eae13eeac52f0515679dd6f00ab55e8750d6f86dc.jpg)  
Fig. 12. Average task wait time comparison of different approaches under medium-scale workload patterns.

Medium-100, and Large-250. Notably, in the most challenging Large-250 scenario, HiGFRL limits its retry count to 11192, a figure significantly better than the 11878 retries of GA-DRL and 11690 of MODRL, demonstrating superior resource adaptability. It is worth noting that in specific scenarios such as Standard-50 and Large-100, HiGFRL’s retry count is marginally higher than that of Tetris or GA-DRL. For instance, in the Large-100 scenario, HiGFRL records 8523 retries compared to 8147 for GA-DRL. However, considering the ATFT and ATWT data, although HiGFRL incurs a minimal number of additional retries, its corresponding flow and wait times are significantly lower. This indicates that HiGFRL has learned a more advanced policy: it does not mechanically pursue zero retries but rather engages in policy exploration for superior resource allocation. This mechanism allows the approach to skip current suboptimal solutions with minimal retry costs in exchange for substantial improvements in global scheduling efficiency and user service quality, thereby achieving an optimal balance among multiple key evaluation metrics.

## 5.5 Ablation Study on Hierarchical Graph Fusion Mechanism

To verify the effectiveness of HiGFRL’s hierarchical graph fusion mechanism, we conducted a comprehensive ablation study. As emphasized in the introduction, existing approaches based on GNNs and reinforcement learning often face the severe challenge of structural-semantic misalignment due to the loose coupling and simple concatenation of topological and resource features. To demonstrate how the hierarchical decoupled design of HiGFRL effectively overcomes this challenge, we designed an ablation baseline named HiGFRL w/o HGF (HiGFRL without Hierarchical Graph Fusion).

TABLE 4  
COMPARISON OF TOTAL RETRY COUNT UNDER DIFFERENT WORKLOAD PATTERNS
<table><tr><td>Workload Patterns</td><td>RoundRobin</td><td>Random</td><td>Tetris</td><td>READYS</td><td>MODRL</td><td>GA-DRL</td><td>HiGFRL</td></tr><tr><td>Standard-50</td><td>3301</td><td>21</td><td>0</td><td>32</td><td>7</td><td>5</td><td>13</td></tr><tr><td>Standard-100</td><td>4962</td><td>569</td><td>435</td><td>439</td><td>519</td><td>513</td><td>358</td></tr><tr><td>Medium-50</td><td>9890</td><td>967</td><td>881</td><td>842</td><td>996</td><td>584</td><td>578</td></tr><tr><td>Medium-100</td><td>13798</td><td>3388</td><td>2702</td><td>2760</td><td>3051</td><td>2870</td><td>2559</td></tr><tr><td>Large-100</td><td>26200</td><td>9143</td><td>8530</td><td>9053</td><td>9067</td><td>8147</td><td>8523</td></tr><tr><td>Large-250</td><td>30263</td><td>12155</td><td>11540</td><td>11450</td><td>11690</td><td>11878</td><td>11192</td></tr></table>

![](images/7aa436519bed52f8690eb6f51c16d955a6ea8c94b24b62ced6eddb11ca34ea57.jpg)  
Fig. 13. Average task wait time comparison of different approaches under large-scale workload patterns.

HiGFRL w/o HGF completely ablates the hierarchical graph fusion mechanism, degrading the model into a structureless, end-to-end flat GAT. Its core design philosophy is to construct a homogeneous global graph, degrading the hierarchical multiview design of HiGFRL entirely into a traditional flat structure. The hierarchical graph fusion mechanism is a deeply coupled organic whole, in which the static hypergraph provides global topological priors, the dynamic global graph captures the cluster heterogeneous resource posture, and the local bipartite graph executes precise microscopic matching. To achieve a complete degradation to a generic flat architecture, this baseline model underwent structured ablations and replacements at key stages of information flow.

Specifically, HiGFRL w/o HGF completely removes the explicit hypergraph convolutional network, abandons the crossview topological feature fusion strategy, and instead performs graph convolution directly based on raw node attributes on a single homogeneous large graph. This renders the baseline model unable to effectively perceive long-range dependencies such as critical paths and high-order synchronization semantics. Following the cancellation of the hierarchical processing mechanism, the global state encoder used to evaluate the global posture is replaced by a flat graph attention convolution. Due to the lack of cross-view feature decoupling, the topological state of tasks and the underlying physical resource features of VMs are implicitly mixed within a single graph structure. This leads to severe semantic entanglement and over-smoothing of node representations during information passing, which severely limits the evaluation accuracy of the global state evaluator regarding the real physical posture and global load pressure of the complex heterogeneous cluster. Meanwhile, to accommodate the setting of the global homogeneous graph, the original local bipartite graph construction mechanism is removed, and the context-aware matching actor network is downgraded to a global broadcasting and simple concatenation mechanism. This causes the model to lose its ability to focus on fine-grained microscopic matching relationships, failing to explicitly model the deep interactive correlations between tasks and candidate nodes during the feature extraction stage, thereby degrading the matching process into an implicit feature mapping lacking structural dependencies. In summary, this systematic structural degradation ablates the multi-view collaborative representation mechanism, aiming to provide a rigorous ablation baseline to verify the overall effectiveness and necessity of the hierarchical graph fusion mechanism.

Table 5 presents the performance comparison between the complete HiGFRL and the ablated HiGFRL w/o HGF across all six workload patterns. The results demonstrate that the complete hierarchical graph fusion mechanism of HiGFRL provides substantial performance gains, effectively resolving the structural-semantic misalignment problem.

In Standard-50 and Standard-100, both approaches achieve the same Makespan, primarily because the total execution time of small-scale workflows is often limited by the fixed absolute critical path length. However, HiGFRL significantly reduces ATFT and ATWT. In the Standard-100 workload pattern, HiGFRL reduces the ATWT from 432.56 to 401.86. This strongly validates that the local bipartite matching graph introduces a crucial inductive bias, endowing the agent with the ability to make more precise task-to-node pairing decisions, effectively avoiding suboptimal allocations caused by relying on over-smoothed, noisy node features within a flat global graph.

As the workload scale and task graph dependency complexity increase, such as in the Medium-100 and Large-100 scenarios, the performance gap between the two significantly widens. Particularly in the Large-100 scenario, having lost the deep perception of topological structures, HiGFRL w/o HGF suffers severe performance degradation, with its Makespan reaching 1779568. In contrast, HiGFRL maintains a Makespan of 1198246, achieving a substantial reduction of 32.67%.

TABLE 5  
PERFORMANCE COMPARISON BETWEEN HIGFRL AND HIGFRL W/O HGF
<table><tr><td>Workload Patterns</td><td>Approaches</td><td>Makespan</td><td>ATFT</td><td>ATWT</td><td>TRC</td></tr><tr><td>Standard-50</td><td>HiGFRL w/o HGF HiGFRL</td><td>550714 550714</td><td>558.39 519.50</td><td>300.53 276.30</td><td>3</td></tr><tr><td>Standard-100</td><td>HiGFRL w/o HGF</td><td>550822</td><td>772.99</td><td>432.56</td><td>13 454</td></tr><tr><td>Medium-50</td><td>HiGFRL HiGFRL w/o HGF</td><td>550822 552121</td><td>722.12 679.02</td><td>401.86 369.77</td><td>358 729</td></tr><tr><td>Medium-100</td><td>HiGFRL HiGFRL w/o HGF</td><td>551850 636664</td><td>597.53 1861.49</td><td>325.39 1106.39</td><td>578 2808</td></tr><tr><td>Large-100</td><td>HiGFRL HiGFRL w/o HGF</td><td>561999 1779568</td><td>1635.77 5694.29</td><td>970.58 3412.62</td><td>2559 8784</td></tr><tr><td>Large-250</td><td>HiGFRL HiGFRL w/o HGF</td><td>1198246 3950588</td><td>5097.41 27088.51</td><td>3069.95 17430.96</td><td>8523 11324</td></tr></table>

This proves that the static task dependency hypergraph is indispensable for preserving high-order topological priors. The absence of this module causes severe signal dilution in the flat GAT, rendering the agent unable to identify and prioritize the bottleneck tasks on the critical path hidden deep within the dependency chains.

In medium to large-scale workload scenarios, HiGFRL consistently maintains lower TRC and queueing delays. In the Large-250 scenario, HiGFRL not only effectively reduces delays but also decreases the TRC from 11324 to 11192. This indicates that the hierarchical decoupled design of the dynamic global state representation effectively avoids the mutual interference between task topological features and physical resource features in a flat large graph. Combined with the masked pooling mechanism, HiGFRL can independently and more accurately aggregate the true effective resource load posture across the entire cluster. This effectively prevents the agent from falling into the short-sighted resource over-packing trap under high loads due to distorted state signals, thereby substantially reducing scheduling retries triggered by resource contention.

## 6. CONCLUSION AND FUTURE WORK

In this study, we proposed HiGFRL, a hierarchical graph fusion-driven RL framework, to address instance-level online dependency-aware scheduling in heterogeneous clouds. To overcome the structural-semantic misalignment inherent in existing approaches, HiGFRL explicitly decouples the system state into a static task dependency hypergraph, a dynamic global cluster graph, and a local bipartite matching graph. By integrating these views through a fusion-driven dualstream architecture and a topology-prior-guided hybrid reward, our approach effectively bridges the gap between logical workflow topology and physical resource posture. Extensive evaluations on real-world Alibaba cluster traces confirm that HiGFRL significantly outperforms baselines, demonstrating exceptional performance in large-scale, high-load scenarios by substantially minimizing Makespan, average task flow time, and average task wait time.

For future work, we aim to expand our paradigm to multicluster federated scheduling, which involves scaling the multiview state construction to manage complex workload distributions across geographically distributed and privacy-preserving cloud environments.

## REFERENCES

[1] S. Tyagi and P. Sharma, “Omnilearn: A framework for distributed deep learning over heterogeneous clusters,” IEEE Transactions on Parallel and Distributed Systems, vol. 36, no. 6, pp. 1253–1267, 2025.

[2] Y. Sanjalawe, S. Al-E’mari, S. Fraihat, and S. Makhadmeh, “Ai-driven job scheduling in cloud computing: a comprehensive review,” Artificial Intelligence Review, vol. 58, no. 7, p. 197, 2025.

[3] Y. Cheng, L. Xu, T. Yang, W. Wu, Z. Lin, A. Yu, and W. Chen, “Beehive: Decentralised high-frequency small tasks scheduling in large clusters,” IEEE Transactions on Parallel and Distributed Systems, vol. 36, no. 6, pp. 1326–1337, 2025.

[4] M.-L. Chiang, H.-C. Hsieh, Y.-H. Cheng, W.-L. Lin, and B.-H. Zeng, “Improvement of tasks scheduling algorithm based on load balancing candidate method under cloud computing environment,” Expert Systems with Applications, vol. 212, p. 118714, 2023.

[5] A. Uhlig, I. Braun, and M. Wahlisch, “Challenges in vm scheduling and¨ placement: Insights from a real-world sap cloud dataset,” in Proceedings ofthe ACM SIGCOMM 2025 Posters and Demos, ser. ACM SIGCOMM Posters and Demos ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 124–126.

[6] F. Xu, X. Shen, S. Lin, L. Chen, Z. Zhou, F. Xiao, and F. Liu, “Tetris: Proactive container scheduling for long-term load balancing in shared clusters,” IEEE Transactions on Services Computing, vol. 17, no. 5, pp. 2918–2930, 2024.

[7] J. Guo, Z. Chang, S. Wang, H. Ding, Y. Feng, L. Mao, and Y. Bao, “Who limits the resource efficiency of my datacenter: an analysis of alibaba datacenter traces,” in Proceedings of the International Symposium on Quality of Service, ser. IWQoS ’19. New York, NY, USA: Association for Computing Machinery, 2019.

[8] C. Jiang, Y. Qiu, W. Shi, Z. Ge, J. Wang, S. Chen, C. Cerin, Z. Ren,´ G. Xu, and J. Lin, “Characterizing co-located workloads in alibaba cloud datacenters,” IEEE Transactions on Cloud Computing, vol. 10, no. 4, pp. 2381–2397, 2022.

[9] G. Liu, W. Lin, H. Zhang, J. Lin, S. Peng, and K. Li, “Public datasets for cloud computing: A comprehensive survey,” ACM Comput. Surv., vol. 57, no. 8, Mar. 2025.

[10] C. Zhang, F. Wu, Z. Wang, X. Wang, H. Ma, and Y. Liu, “A graph-based deep reinforcement learning model for task scheduling on heterogeneous resource-elastic management of resource pool,” IEEE Transactions on Cloud Computing, vol. 13, no. 4, pp. 1327–1342, 2025.

[11] Y. Zeng, R. Zhou, L. Jiao, and R. Zhang, “Online scheduling of edge multiple- model inference with dag structure and retraining,” in IEEE INFOCOM 2025 - IEEE Conference on Computer Communications, 2025, pp. 1–10.

[12] H. Liu, G. Zheng, Z. Liu, S. Tian, and Y. Li, “Dependency-aware dynamic priority scheduling for online multi-dag task offloading in mobile edge computing,” IEEE Internet of Things Journal, vol. 13, no. 3, pp. 5053–5068, 2026.

[13] Q. Wu, F. Ishikawa, Q. Zhu, Y. Xia, and J. Wen, “Deadline-constrained cost optimization approaches for workflow scheduling in clouds,” IEEE Transactions on Parallel and Distributed Systems, vol. 28, no. 12, pp. 3401–3412, 2017.

[14] B. Lin, C. Lin, X. Chen, M. Lin, G. Huang, and Z. Xu, “Costdriven scheduling for workflow decision making systems in fuzzy edgecloud environments,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 3756–3771, 2025.

[15] J. Zhang, X. Li, L. Chen, and R. Ruiz, “Scheduling workflows with limited budget to cloud server and serverless resources,” IEEE Transactions on Services Computing, vol. 17, no. 4, pp. 1766–1779, 2024.

[16] G. Ismayilov and H. R. Topcuoglu, “Neural network based multiobjective evolutionary algorithm for dynamic workflow scheduling in cloud computing,” Future Generation Computer Systems, vol. 102, pp. 307–322, 2020.

[17] S. Qin, D. Pi, Z. Shao, Y. Xu, and Y. Chen, “Reliability-aware multiobjective memetic algorithm for workflow scheduling problem in multicloud system,” IEEE Transactions on Parallel and Distributed Systems, vol. 34, no. 4, pp. 1343–1361, 2023.

[18] Z. Sun, Y. Mei, F. Zhang, H. Huang, C. Gu, and M. Zhang, “Multitree genetic programming hyper-heuristic for dynamic flexible workflow scheduling in multi-clouds,” IEEE Transactions on Services Computing, vol. 17, no. 5, pp. 2687–2703, 2024.

[19] T. Dong, F. Xue, C. Xiao, and J. Zhang, “Deep Reinforcement Learning for Dynamic Workflow Scheduling in Cloud Environment,” in 2021 IEEE International Conference on Services Computing (SCC), 2021, pp. 107–115.

[20] M. Cheng, J. Li, P. Bogdan, and S. Nazarian, “H<sub>2</sub>O-Cloud: A Resource and Quality of Service-Aware Task Scheduling Framework for Warehouse-Scale Data Centers,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 39, no. 10, pp. 2925– 2937, 2020.

[21] Z. Hu, J. Tu, and B. Li, “Spear: Optimized Dependency-Aware Task Scheduling with Deep Reinforcement Learning,” in 2019 IEEE 39th International Conference on Distributed Computing Systems (ICDCS), 2019, pp. 2037–2046.

[22] Z. Yu, W. Liu, X. Liu, and G. Wang, “Drag-JDEC: A Deep Reinforcement Learning and Graph Neural Network-based Job Dispatching Model in Edge Computing,” in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQOS), 2021, pp. 1–10.

[23] S. Munikoti, D. Agarwal, L. Das, M. Halappanavar, and B. Natarajan, “Challenges and opportunities in deep reinforcement learning with graph neural networks: A comprehensive review of algorithms and applications,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 11, pp. 15 051–15 071, 2024.

[24] H. Mao, M. Schwarzkopf, S. B. Venkatakrishnan, Z. Meng, and M. Alizadeh, “Learning Scheduling Algorithms for Data Processing Clusters,” in Proceedings of the ACM Special Interest Group on Data Communication, ser. SIGCOMM ’19. New York, NY, USA: Association for Computing Machinery, 2019, p. 270–288.

[25] N. Grinsztajn, O. Beaumont, E. Jeannot, and P. Preux, “READYS: A Reinforcement Learning Based Strategy for Heterogeneous Dynamic Scheduling,” in 2021 IEEE International Conference on Cluster Computing (CLUSTER), 2021, pp. 70–81.

[26] Z. Liu, L. Huang, Z. Gao, M. Luo, S. Hosseinalipour, and H. Dai, “GA-DRL: Graph Neural Network-Augmented Deep Reinforcement Learning for DAG Task Scheduling Over Dynamic Vehicular Clouds,” IEEE Transactions on Network and Service Management, vol. 21, no. 4, pp. 4226–4242, 2024.

[27] Z. Wang, W. Zhan, H. Duan, and H. Huang, “Multiobjective optimization deep reinforcement learning for dependent task scheduling based on spatio-temporal fusion graph neural network,” Engineering Applications of Artificial Intelligence, vol. 148, p. 110337, 2025.

[28] M. Nie, D. Chen, and D. Wang, “Reinforcement learning on graphs: A survey,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 7, no. 4, pp. 1065–1082, 2023.

[29] Z. Wang, W. Zhan, H. Duan, G. Min, and H. Huang, “Deep-Reinforcement-Learning-Based Continuous Workflows Scheduling in Heterogeneous Environments,” IEEE Internet ofThings Journal, vol. 12, no. 10, pp. 14 036–14 050, 2025.

[30] Y. Shen, G. Chen, H. Ma, and M. Zhang, “Cost-aware dynamic cloud workflow scheduling using self-attention and evolutionary reinforcement learning,” in Service-Oriented Computing, W. Gaaloul, M. Sheng, Q. Yu, and S. Yangui, Eds. Singapore: Springer Nature Singapore, 2025, pp. 3–18.

[31] L. Lin, L. Pan, and S. Liu, “SpotDAG: An RL-Based Algorithm for DAG Workflow Scheduling in Heterogeneous Cloud Environments,” IEEE Transactions on Services Computing, vol. 17, no. 5, pp. 2904–2917, 2024.

[32] G. P. Koslovski, K. Pereira, and P. R. Albuquerque, “DAG-based workflows scheduling using Actor–Critic Deep Reinforcement Learning,” Future Generation Computer Systems, vol. 150, pp. 354–363, 2024.

[33] J. Li, D. Xiao, J. Yao, Y. Long, and W. Wu, “Learning Scheduling Policies for Co-Located Workloads in Cloud Datacenters,” IEEE Transactions on Cloud Computing, vol. 11, no. 4, pp. 3725–3736, 2023.

[34] J. Xue, T. Wang, and P. Cai, “Towards Efficient Workflow Scheduling Over Yarn Cluster Using Deep Reinforcement Learning,” in GLOBE-COM 2023 - 2023 IEEE Global Communications Conference, 2023, pp. 473–478.

[35] G. Dong, J. Wang, M. Wang, and T. Su, “An improved scheduling with advantage actor-critic for Storm workloads,” Cluster Computing, vol. 27, no. 10, p. 13421–13433, Jun. 2024.

[36] L. Shu, G. Pan, B. Wang, W. Peng, M. Fang, Y. Chen, F. Huang, S. Li, and Y. Cheng, “Smart dag task scheduling based on mcts method of multi-strategy learning,” in Algorithms and Architectures for Parallel Processing, Z. Tari, K. Li, and H. Wu, Eds. Singapore: Springer Nature Singapore, 2024, pp. 224–242.

[37] R. Grandl, G. Ananthanarayanan, S. Kandula, S. Rao, and A. Akella, “Multi-resource packing for cluster schedulers,” in Proceedings of the 2014 ACM Conference on SIGCOMM, ser. SIGCOMM ’14. New York, NY, USA: Association for Computing Machinery, 2014, p. 455–466.