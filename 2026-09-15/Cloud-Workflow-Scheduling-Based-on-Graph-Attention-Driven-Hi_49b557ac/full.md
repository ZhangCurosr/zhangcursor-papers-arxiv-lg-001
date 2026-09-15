# Cloud Workflow Scheduling Based on Graph Attention-Driven Hierarchical Reinforcement Learning

Zongjin Li, Shaohan Feng Member, IEEE, Chunxi Yang Member, IEEE and Wenbo Wang Senior Member, IEEE,

Abstract—Dynamic cloud workflow scheduling must balance deadline satisfaction, container utilization, and energy consumption while dealing with stochastic task-execution speeds, placement-dependent communication, and coupled task and container decisions. Workflows are naturally modeled as directed acyclic graphs (DAGs), but conventional vector- or matrixbased states do not fully capture their dependency topology. To better represent task urgency and structural relationships, we assign predicted sub-deadlines to tasks and use a multihead graph attention network (GAT) to extract dependency information from the evolving DAGs. Based on these representations, we develop a Graph Attention-Driven Hierarchical Reinforcement Learning (GA-HRL) framework and model the scheduling process as an event-driven hierarchical semi-Markov decision process (SMDP). Workflow arrivals and task completions trigger scheduling events. At each scheduling event, the Task Scheduling (TS) agent first processes the currently ready tasks by assigning them to admissible existing containers or requesting new ones. The requested containers are then processed by the Container Scheduling (CS) agent for host placement before the environment advances. The two agents are trained alternately using separate Proximal Policy Optimization (PPO). Experiments on the 2018 Alibaba cluster trace show that GA-HRL maintains competitive workflow success rate and, in settings where success is comparable, generally achieves higher container utilization and lower energy consumption. Under the largest speed variation, it trades a small success-rate margin for substantially lower energy. Simulation code is available at: https://github.com/zongjin130/ GA-HRL.

Index Terms—Graph attention network, hierarchical reinforcement learning, dynamic workflow scheduling, multi-objective optimization, container placement.

## I. INTRODUCTION

N cloud computing environments, users submit their computational demands as workflows to the cloud platform, and the provider must allocate computing resources according to each workflow’s task structure and dependencies [1]. The scheduler therefore has to balance multiple objectives simultaneously [2]. From the user perspective, workflows should finish before their deadlines to maintain quality of service [3]. From the provider perspective, allocated resources should be utilized effectively, because unnecessary provisioning and poor utilization keep additional hosts and containers active and increase energy consumption. Consequently, dynamic workflow scheduling must jointly consider task timeliness, resource

utilization, and energy efficiency rather than optimizing them separately.

Cloud workflows are becoming larger and more complex, and cloud platforms usually operate online, so multiple workflows with different structures, arrival times, and deadlines may coexist [2]. Scheduling such workflows is difficult for two related reasons. First, a workflow is successful only after all its constituent tasks are coordinated, so task decisions are coupled. Second, precedence constraints restrict the order of execution, and treating tasks independently discards useful dependency information. Workflows are therefore commonly represented as Directed Acyclic Graphs (DAGs), where nodes denote tasks and directed edges represent both precedence constraints and data transfers [4].

From the resource-management perspective, containers are widely used because they are lightweight, start quickly, and support high resource density [5]. The platform may also need to create additional containers on suitable hosts when current resources are insufficient to meet workflow deadlines [6]. In practice, however, container performance is not fully predictable: the execution speed of a task can vary with host load and placement-dependent interference [2]. Communication is likewise placement-dependent, because intrahost and inter-host transfers use different bandwidths and transfer time cannot be known until containers are placed [3]. These uncertainties interact with workflow dependencies and can cause waiting time, reduce container reuse, and increase energy consumption. Dynamic workflow scheduling therefore requires a model that captures both runtime uncertainty and placement-dependent communication.

Cloud workflow scheduling requires trade-offs both among tasks of the same workflow and among workflows competing for shared resources, and the problem is NP-hard [7, 8]. Traditional heuristics depend on problem-specific rules, whereas optimization-based methods often rely on repeated search or expensive solving procedures. As the numbers of tasks and candidate resources grow, such methods become difficult to deploy in an online scheduler. Reinforcement Learning (RL) has therefore attracted attention as a way to learn scheduling policies from interaction with the environment [9]. However, many RL-based schedulers still represent tasks by ordinary vectors or matrices, which retain local attributes but do not explicitly capture the non-Euclidean dependency structure of a DAG [10]. How to preserve and exploit this structure under changing resource conditions remains an open problem.

This paper studies dynamic cloud workflow scheduling with stochastic task-execution speeds and placement-dependent communication. Each active workflow is kept as a DAG, and a predicted sub-deadline is assigned to every task to represent its temporal urgency. A multi-head GAT then aggregates dependency information so that tasks with similar local attributes can still be distinguished by their structural context. The resulting representation is used in an eventdriven hierarchical semi-Markov Decision Process (SMDP), where a Task Scheduling (TS) agent first assigns ready tasks to admissible existing containers or requests new ones. All new-container requests generated during the TS phase are then processed by a Container Scheduling (CS) agent for host placement before physical time advances. The two policies are trained alternately using separate PPO actor-critic networks while sharing the same scheduling environment. GA-HRL is designed to meet workflow deadlines while also improving container utilization and reducing total energy consumption. The main contributions are as follows:

1) We develop a dependency-aware task representation for dynamically arriving DAG workflows. Predicted subdeadlines express task urgency, while a multi-head GAT aggregates topological information that conventional vector or matrix states do not retain explicitly.

2) We formulate the coupled task-to-container and container-to-host decisions as an event-driven hierarchical SMDP. At each scheduling event, the TS agent first processes the ready-task assignments, after which the CS agent processes the host-placement decisions generated by new-container requests before physical time advances.

3) We train the two agent-specific policies with PPO over their event-driven decision sequences. Trace-driven experiments show that GA-HRL maintains competitive workflow success and, in settings where success is comparable, achieves higher container utilization and lower energy consumption. Under the largest speed variation, it trades a small success-rate margin for substantially lower energy.

The remainder of this paper is organized as follows. Section II reviews related work. Section III introduces the system model and problem statement. Section IV presents the proposed GA-HRL scheduling algorithm. Section V reports the experimental results, and Section VI concludes the paper.

## II. RELATED WORKS

This section reviews representative studies on cloud workflow scheduling from four perspectives: containerized workflow scheduling, mathematical optimization methods, heuristic optimization methods, and machine-learning-based methods. The discussion focuses on how these approaches represent workflow dependencies, handle resource allocation, and respond to dynamic or uncertain execution conditions.

## A. Containerized Workflow Scheduling

Workflows are commonly represented by Petri nets [11], UML diagrams, or DAGs [4]. DAGs are widely used because they naturally express task parallelism, precedence constraints, and data dependencies. Each workflow DAG has its own arrival time, structure, and deadline [2].

In containerized cloud environments, the scheduler must make two coupled decisions: task scheduling, which determines the execution order and start time of tasks, and container placement, which determines the host on which each container is deployed [6]. These two decisions jointly affect execution efficiency and resource consumption. In practice, container execution speed fluctuates with host load, and communication delay depends on whether containers are co-located. These uncertainties interact with workflow dependencies, which can waste resources and increase energy consumption.

## B. Mathematical Optimization based Methods

Optimization-based methods formulate cloud resource allocation or workflow scheduling as mathematical programs. Chen et al. [3] reconstructed the request sequence by priority and minimized matching distance and the number of active physical machines. Jiao et al. [12] formulated joint resource placement and allocation as an integer linear program and designed a dynamic-programming-based algorithm. Hahnel et al. [13] extended the cutting-stock model to consolidate heterogeneous service requests, reducing overload probability and energy consumption. Liu et al. [14] proposed an approximate function placement algorithm for edge-cloud job completion time minimization, and Das et al. [15] studied dynamic function placement under cost and deadline constraints. Deng et al. [16] later embedded dependent functions into distributed serverless edge computing to obtain the optimal function placement and start time.

These methods provide clear formulations and, in some cases, approximation guarantees. However, they generally assume deterministic inputs or require repeated optimization as the system state changes. This limits algorithm scalability because the computational burden grows quickly with the decision space. In our setting, workflows arrive online, execution speeds are stochastic, and each task assignment affects subsequent container placement, making direct online application of such methods difficult.

## C. Heuristic Optimization based Methods

Heuristic methods can be broadly divided into rule-based scheduling and iterative swarm-intelligence methods [2]. Rodriguez et al. [17] used particle swarm optimization to maximize the scheduling success rate and minimize cost in static cloud workflows. Arabnejad et al. [18] proposed deadlinebased heuristics for dynamic cloud workflows. Chen et al. [2] developed an uncertainty-aware online scheduler for realtime workflows with multiple objectives, while Fan et al. [6] proposed an energy-efficient heuristic for deadline-constrained workflows with container placement.

These heuristics are generally computationally efficient and often work well under fixed assumptions. However, their performance depends heavily on hand-crafted rules or tuned parameters. When the arrival process, execution-speed distribution, or resource configuration changes, the same rules may not adapt as effectively as a learned policy.

## D. Machine Learning based Methods

Machine-learning-based schedulers can be divided into methods that use learning as a predictor/evaluator and methods that use RL to generate policies [19, 20]. Yang et al. [19] combined machine-learning prediction with relaxed linear programming to schedule tasks with unknown execution times. Yu et al. [8] applied RL with custom reward functions to optimize dynamic workflow scheduling. Ding et al. [20] proposed a Transformer-enhanced Deep Q-Network for largescale workflow scheduling, but the approach does not fully exploit inter-task dependency structure. Xie et al. [21] used graph neural networks to extract features of workflows and resources, improving scheduling success and energy efficiency.

Despite these advances, many existing schedulers use dependency information mainly to determine eligible tasks, rather than embedding the DAG topology directly into the policy state. As a result, the structural information carried by the DAG is only partially exploited when resources are selected. Moreover, several methods assume that workflows are available at the beginning of scheduling or use pre-execution attributes, which does not fully capture the continual changes caused by online arrivals, fluctuating execution speeds, and dynamic container availability.

GA-HRL addresses these gaps by combining DAGpreserving GAT representations, predicted sub-deadlines, and a hierarchical TS/CS policy in an event-driven scheduling framework. It differs from prior cloud workflow schedulers at three connected levels:

• it explicitly models random workflow arrivals, taskspecific execution-speed realizations, and placementdependent communication;

• at the TS level, the task state combines a predicted urgency indicator with a GAT representation of the DAG topology; and

• for policy derivation, the state drives an event-triggered hierarchical policy that coordinates task assignment and container placement at separate decision epochs.

## III. MODEL AND PROBLEM STATEMENT

This section focuses on three key aspects: cloud resource modeling, workflow modeling, and the problem statement.

## A. Cloud Resource Model

We consider a cloud service center in which physical hosts are activated on demand. Let $\mathcal { H } = \{ H _ { 1 } , H _ { 2 } , \cdot \cdot \cdot , H _ { N } \}$ denote the set of host instances activated during a scheduling episode, where N is the total number of activated host instances. Each host $H _ { n }$ is characterized by a resource tuple $H _ { n } \ =$ $( C _ { n } , M _ { n } , \bar { Q } _ { n } , \bar { P } _ { n } )$ , where $C _ { n }$ represents the number of CPU cores, $M _ { n }$ is the memory size, ${ \bar { Q } } _ { n }$ is the mean computational capacity measured in Million Instructions Per Second (MIPS), and ${ \bar { P } } _ { n }$ is the mean power under full utilization. Let C<sup>all</sup> denote the set of all containers created during the scheduling horizon, and let $\mathcal { C } _ { n } ( t )$ denote the set of containers that are currently alive on host $H _ { n }$ at time t. We further define $C _ { n } ^ { \mathrm { a l l } } = \{ c _ { m } \in \mathcal { C } ^ { \mathrm { a l l } } : \eta _ { m } = n \}$ as the set of all containers deployed on $H _ { n }$ during the scheduling horizon. Then, at any time, a container $c _ { m } \in \mathcal { C } _ { n } ( t )$ is characterized by the resources allocated to it: $c _ { m } = ( C _ { m } , M _ { m } )$ , where $C _ { m }$ and $M _ { m }$ are respectively the number of CPU cores and memory required for container $c _ { m }$

The mean computational capacity and mean power of the container are determined by the number of CPU cores. The mean computational capacity of container $c _ { m }$ is expressed as

$$
\bar { Q } _ { m } = \frac { \bar { Q } _ { \eta _ { m } } C _ { m } } { C _ { \eta _ { m } } } ,\tag{1}
$$

where $\eta _ { m } = n$ means that container $c _ { m }$ is deployed on host $H _ { n }$ . The nominal capacity $\hat { Q } _ { m }$ is known to the scheduler, whereas the capacity realized during execution is task specific. Let $Q _ { k , j } ^ { ( m ) }$ denote the computational capacity experienced by task $t _ { k , j }$ when it executes on container $c _ { m }$ . At the start of every task execution, a new realization is sampled independently as

$$
Q _ { k , j } ^ { ( m ) } \sim { \mathcal { N } } \big ( \bar { Q } _ { m } , ( v \bar { Q } _ { m } ) ^ { 2 } \big ) , 0 < Q _ { k , j } ^ { ( m ) } < 2 \bar { Q } _ { m } ,\tag{2}
$$

where v is the variance coefficient. Thus, two tasks executed successively on the same container may experience different realized capacities, while scheduling decisions made before execution use the nominal value $\hat { Q } _ { m }$ . We assume that the data transmission speed between containers is different [6]: the communication speed within the same host shares the internal network bandwidth of the host, usually with lower latency and not limited by physical networks. The communication speed in this scenario is denoted as $B ^ { i n }$ $B ^ { i n }$ represents the effective intra-host bandwidth under the adopted sharing assumption. When the container is deployed on different hosts, all data transmission must pass through a shared physical network link, and its bus bandwidth, denoted as $B ^ { c r }$ , is strictly limited by the physical bandwidth of the cross-host link.

## B. Workflow Model

We consider a set of $\mathcal { W } = \{ W _ { 1 } , \ldots , W _ { K } \}$ of K workflows. Each workflow is characterized by the tuple $W _ { k } \ =$ $\left( A _ { k } , D _ { k } , G _ { k } \right)$ , where $A _ { k }$ is its arrival time, $D _ { k }$ is its deadline, and $G _ { k } = ( \mathcal { T } _ { k } , \mathcal { E } _ { k } )$ is the DAG representing its task structure. In particular, $\mathcal { T } _ { k } = \{ t _ { k , 1 } , \ldots , t _ { k , N _ { k } } \}$ is the task set of $W _ { k }$ , with $N _ { k } = | { \mathcal { T } } _ { k } | .$ , and $\mathcal { E } _ { k } \subseteq \{ e _ { k , i j } \mid i , j \in \{ 1 , . . . , N _ { k } \} , i \neq j \}$ is the edge set representing the data dependency between tasks. An edge $e _ { k , i j }$ indicates that $t _ { k , i }$ is an immediate predecessor of $t _ { k , j } .$ , equivalently, $t _ { k , j }$ is an immediate successor of $t _ { k , i } .$ We denote the immediate predecessor and successor sets of $t _ { k , j }$ by $\mathrm { p r e d } ( t _ { k , j } )$ and succ $: ( t _ { k , j } )$ , respectively. A task with no immediate predecessor is called an entry task, and a task with no immediate successor is called an exit task. Each task $t _ { k , j }$ has a required computation size $p _ { k , j }$ , measured in Million Instructions, and each edge $e _ { k , i j }$ carries a data volume $d _ { k , i j }$ that must be transmitted from $t _ { k , i }$ to $t _ { k , j }$ . We assume that the workflow structure, task computation sizes, and inter-task data volumes are known when $W _ { k }$ arrives (see also [1]).

An illustrative example is shown in Fig 1, where tasks $t _ { 1 , 1 }$ and $t _ { 1 , 1 0 }$ are respectively the entry task and the exit task. Tasks $t _ { 1 , 4 } , t _ { 1 , 5 }$ , and $t _ { 1 , 6 }$ are the immediate predecessors of task $t _ { 1 , 7 }$ Tasks $t _ { 1 , 8 }$ and $t _ { 1 , 9 }$ are the immediate successors of task $t _ { 1 , 7 }$

![](images/adf0c7cd5e2cc1bae525ee49c07c258865a99410072eff88325b2cf69182c544.jpg)  
Fig. 1: Workflow diagram.

TABLE I: Main notation used in the scheduling model.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { H } , H _ { n }$ </td><td>set of host instances and host instance n</td></tr><tr><td> $C _ { n } , M _ { n }$ </td><td>CPU-core capacity and memory capacity of host  $H _ { r }$ </td></tr><tr><td> ${ \bar { Q } } _ { n } , { \bar { P } } _ { n }$ </td><td>mean computational capacity and mean full-utilization power of host  $H _ { n }$ </td></tr><tr><td> $\boldsymbol { \nu } , \boldsymbol { W _ { k } }$ </td><td>set of workflows and workflow k</td></tr><tr><td> $A _ { k } , D _ { k }$ </td><td>arrival time and deadline of workflow  $W _ { k }$ </td></tr><tr><td> $G _ { k } = ( \mathcal { T } _ { k } , \mathcal { E } _ { k } )$ </td><td>DAG of workflow  $W _ { k } ,$  consisting of task and edge</td></tr><tr><td> $N _ { k }$ </td><td>sets number of tasks in workflow  $W _ { k }$ </td></tr><tr><td> $t _ { k , j } , e _ { k , i j }$ </td><td>task  $j$  of workflow  $W _ { k }$  and dependency edge from</td></tr><tr><td> $\mathrm { p r e d } ( t _ { k , j } ) , \mathrm { s u c c } ( t _ { k , j } )$ </td><td> $t _ { k , i } ~ \mathrm { t o } ~ t _ { k , j }$  immediate predecessor and successor sets of task  $\ d _ { t _ { k , j } }$ </td></tr><tr><td> $p _ { k , j } , d _ { k , i j }$ </td><td>computational workload of task  $t _ { k , j }$  and data volume transmitted from  $t _ { k , i } ~ \mathrm { t o } ~ t _ { k , j }$  set of all containers created during the scheduling</td></tr><tr><td> $\mathcal { C } ^ { \mathrm { a l l } } , c _ { m }$ </td><td>horizon and globally indexed container m</td></tr><tr><td> $\mathcal { C } _ { n } ( t )$  Call</td><td>set of containers currently alive on host  $H _ { n }$  at time t set of all containers deployed on host  $H _ { n }$  during the</td></tr><tr><td>n</td><td>scheduling horizon</td></tr><tr><td> $C _ { m } , M _ { m }$   $\bar { Q } _ { m } , Q _ { k , j } ^ { ( m ) }$ </td><td>CPU-core and memory requirements of container  $c _ { m }$  nominal capacity of container  $c _ { m }$  and capacity real-</td></tr><tr><td></td><td>ized by task  $t _ { k , j }$  on  $c _ { m }$ </td></tr><tr><td> $\bar { P } _ { m } , P _ { k , j } ^ { ( m ) }$ </td><td>nominal container power and task-specific power dur- ing execution</td></tr><tr><td> $\mu _ { k , j }$   $\eta _ { m }$ </td><td>index of the container assigned to task  $t _ { k , j }$  host index of container  $c _ { m } ; \eta _ { m } = n { \mathrm { ~ i f f ~ } } c _ { m } \in { \mathcal { C } } _ { n } ^ { \mathrm { a l l } }$ </td></tr><tr><td> $\dot { B } ^ { i n } , B ^ { c r }$ </td><td>intra-host and cross-host data transmission bandwidths</td></tr><tr><td> $n _ { \mathrm { a c t } } ^ { n } ( t )$ </td><td>number of containers on host  $H _ { n }$  simultaneously</td></tr><tr><td></td><td>receiving data at time t</td></tr><tr><td> $S T _ { c }$   $\tau _ { k , \textit { i } } ^ { \mathrm { e x } }$ </td><td>startup time required to deploy a new container execution time of task  $t _ { k , j }$ </td></tr><tr><td> $\tau _ { k , i j } ^ { \mathrm { t x } }$ </td><td>data transmission time from task  $t _ { k , i }$  to task  $t _ { k , j }$ </td></tr><tr><td> $R _ { m } , S _ { k , j } , F _ { k , j } , F _ { k }$ </td><td>container-ready, task-start, task-finish, and workflow-</td></tr><tr><td></td><td>finish times</td></tr><tr><td> $F _ { m } ^ { \mathrm { c u r } }$ </td><td>scheduled finish time of the task currently executing</td></tr><tr><td> $T _ { n } ^ { \mathrm { { o n } } }$ </td><td>on container  $c _ { m }$  total powered-on duration of host  $H _ { n }$ </td></tr><tr><td></td><td></td></tr><tr><td> $T _ { m } ^ { \mathrm { a c t } } , T _ { m } ^ { \mathrm { l i f e } }$ </td><td>active execution time and total lifetime of container</td></tr><tr><td></td><td> $c _ { m }$ </td></tr><tr><td> $r _ { h } , r _ { c }$ </td><td>host static-power ratio and container idle-power ratio</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td> $E _ { n } ^ { \mathrm { s t a } }$ </td><td></td></tr><tr><td></td><td> $H _ { n }$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>static energy consumption of host</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td> $E _ { m } ^ { \mathrm { a c t } } , E _ { m } ^ { \mathrm { i d l e } }$ </td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>active and idle energy consumption of container</td></tr><tr><td></td><td></td></tr><tr><td> $E _ { n }$ </td><td></td></tr></table>

We introduce two mappings for notational convenience: $\mu _ { k , j } = m$ (task $t _ { k , j }$ assigned to container $c _ { m } )$ and $\eta _ { m } = n$ (container $c _ { m }$ deployed on host $H _ { n } )$ . These are merely index simplifications and leave the scheduling model unchanged. The main notations are summarized in Table I.

## C. Execution Model

We first describe the event-driven execution lifecycle and defines the timing quantities used in the model. When workflow $W _ { k }$ arrives at time $A _ { k } ,$ , the scheduler first identifies the tasks that are ready according to the DAG. The TS agent then sequentially assigns these ready tasks to admissible existing containers or requests new containers. All new-container requests generated during the current TS phase are added to a deployment queue. After the TS phase is completed, the CS agent places the corresponding undeployed containers on hosts, and each newly created container becomes operational only after the startup delay $S T _ { c }$

Because task durations are stochastic, a long predicted queue can accumulate considerable timing error. We therefore restrict each container to at most two unfinished tasks: one task in execution and at most one waiting task. A container whose execution and waiting positions are both occupied is removed from the feasible TS action set. When the executing task finishes, the waiting task may start only after all required predecessor data have arrived. Otherwise, the container remains idle until the start-time condition in (6) is satisfied. Workflow $W _ { k }$ is successful only if all of its tasks finish no later than $D _ { k }$

The timing model distinguishes between quantities available when a scheduling action is chosen and quantities realized later during actual execution. Candidate actions are evaluated from nominal information, whereas the simulator advances according to sampled execution. Candidate actions are evaluated from nominal information, whereas the actual system evolution follows sampled execution capacities. The startup time $S T _ { c } ,$ execution time $\tau _ { k , j } ^ { \mathrm { e x } } ,$ transmission time $\tau _ { k , i j } ^ { \mathrm { t x } } ,$ container-ready time $R _ { \mu _ { k , j } }$ , task start time $S _ { k , j }$ , task finish time $F _ { k , j }$ , and workflow finish time $F _ { k }$ together define the event timeline.

1) Execution time of tasks: When task $t _ { k , j }$ starts on its assigned container ${ \mathit { c } } _ { \mu _ { k , j } } ,$ the task-specific capacity $Q _ { k , j } ^ { ( \mu _ { k , j } ) }$ is sampled according to (2). The realized execution time is then

$$
\tau _ { k , j } ^ { \mathrm { e x } } = \frac { p _ { k , j } } { Q _ { k , j } ^ { ( \mu _ { k , j } ) } } ,\tag{3}
$$

whereas the scheduler evaluates a candidate container $c _ { m }$ using the predicted execution time $\widehat { \tau } _ { k , j , m } ^ { \mathrm { e x } } = p _ { k , j } / \bar { Q } _ { m }$ . The prediction is available at the decision epoch; the realized capacity is sampled only when execution begins.

2) Data transmission time $o f$ tasks: After a predecessor task $t _ { k , i }$ finishes, its output data of volume $d _ { k , i j }$ must be transmitted to the container hosting its successor task $t _ { k , j }$ before $t _ { k , j }$ can start. The transmission time depends on the relative placement of the two tasks’ containers, and three cases are considered:

• Same container: If $t _ { k , i }$ and $t _ { k , j }$ are assigned to the same container, the output data already resides in local memory, so no data transfer is required and the transmission time is zero.

• Different containers on the same host: If $t _ { k , i }$ and $t _ { k , j }$ are assigned to different containers co-located on the same host, the data is transferred through the intra-host network (e.g., shared memory bus or virtual bridge) at bandwidth $B ^ { i n }$ . The transmission time is $d _ { k , i j } / B ^ { i n }$

• Containers on different hosts: If the containers hosting $t _ { k , i }$ and $t _ { k , j }$ are deployed on different physical hosts, the data must traverse the inter-host network link with total bandwidth $B ^ { c r }$ . Because this link is shared among all containers concurrently receiving data on the destination host $H _ { \eta _ { \mu _ { k , j } } }$ , each container receives an equal share of the bandwidth. Let $n _ { \mathrm { a c t } } ^ { { \eta } _ { \mu _ { k , j } } } \left( t \right)$ denote the number of such receiving containers at transmission start time t. The effective bandwidth per container is $B ^ { c r } / n _ { \mathrm { a c t } } ^ { \eta _ { \mu } _ { k , j } } ( t )$ yielding a transmission time of $d _ { k , i j } n _ { \mathrm { a c t } } ^ { \eta _ { \mu _ { k , j } } } ( t ) / B ^ { c r }$

Formally, the transmission time from $t _ { k , i }$ to $t _ { k , j }$ is

$$
\tau _ { k , i j } ^ { \mathrm { t x } } = \left\{ \begin{array} { l l } { 0 , } & { \mu _ { k , i } = \mu _ { k , j } , } \\ { \displaystyle \frac { d _ { k , i j } } { B ^ { i n } } , } & { \mu _ { k , i } \neq \mu _ { k , j } , \eta _ { \mu _ { k , i } } = \eta _ { \mu _ { k , j } } , } \\ { \displaystyle \frac { d _ { k , i j } n _ { \mathrm { a c t } } ^ { \eta _ { \mu _ { k , j } } } ( t ) } { B ^ { c r } } , } & { \eta _ { \mu _ { k , i } } \neq \eta _ { \mu _ { k , j } } . } \end{array} \right.\tag{4}
$$

3) Ready time of containers: Only a container with no waiting task can be selected for another assignment. For such a candidate, the ready time is the earliest instant at which the newly assigned task can occupy the execution position:

$$
R _ { \mu k , j } = \left\{ \begin{array} { l l } { T _ { 0 } + S T _ { c } , } & { \mathrm { i f ~ } c _ { \mu _ { k , j } } \mathrm { ~ i s ~ n e w l y ~ d e p l o y e d } , } \\ { F _ { \mu _ { k , j } } ^ { \mathrm { c u r } } , } & { \mathrm { i f ~ i t ~ i s ~ e x e c u t i n g ~ o n e ~ t a s k } , } \\ { t _ { \mathrm { n o w } } , } & { \mathrm { i f ~ i t ~ i s ~ i d l e } . } \end{array} \right.\tag{5}
$$

Here, $T _ { 0 }$ is the time at which the deployment of $c _ { \mu _ { k , } }$ j is requested, $F _ { \mu _ { k , j } } ^ { \mathrm { c u r } }$ is the already scheduled finish time of the task currently executing on $c _ { \mu _ { k , j } }$ , and $t _ { \mathrm { n o w } }$ is the current decision time. The newly assigned task may still start later than $R _ { \mu _ { k , j } }$ if its predecessor data have not yet arrived.

4) Start time of tasks: When task $t _ { k , j }$ is assigned to container $c _ { \mu k , j } ,$ , its start time is jointly determined by the container ready time $R _ { \mu _ { k , j } }$ and the data-ready times of all its predecessors:

$$
S _ { k , j } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ A _ { k } , R _ { \mu _ { k , j } } \} , } & { t _ { k , j } = t _ { k , \mathrm { e n t r y } } , } \\ { \operatorname* { m a x } \{ R _ { \mu _ { k , j } } , } \\ { \operatorname* { m a x } \quad } \\ { t _ { k , i } \in \mathrm { p r e d } ( t _ { k , j } ) } \end{array} \right.\tag{6}
$$

For an entry task, the workflow arrival time $A _ { k }$ acts as its dataready time, which avoids taking a maximum over an empty predecessor set.

5) Finish time of tasks: The finish time of task $t _ { k , j }$ on container $c _ { \mu _ { k , j } }$ is obtained by adding its realized execution time to its start time:

$$
F _ { k , j } = S _ { k , j } + \tau _ { k , j } ^ { \mathrm { e x } } .\tag{7}
$$

6) Finish time of workflows: A workflow is completed when all of its tasks have finished. Therefore, the completion time of workflow $W _ { k }$ is

$$
F _ { k } = \operatorname* { m a x } _ { t _ { k , j } \in \mathcal { T } _ { k } } F _ { k , j } .\tag{8}
$$

Workflow $W _ { k }$ is considered successfully completed if and only if $F _ { k } \le D _ { k }$

## D. Energy Model

This subsection formulates the energy consumption model for the cloud service center, where the total energy of a host is decomposed into host static energy, container active energy, and container idle energy. To that end, we first define the power parameters of hosts and containers, then compute each energy component accordingly.

1) Power parameters: The mean power of host $H _ { n }$ under full utilization is denoted by $\bar { P } _ { n }$ . Following the widely adopted linear server power model [22], a host that is powered on but not fully loaded still consumes a static power proportional to ${ \bar { P } } _ { n }$

$$
P _ { n } ^ { \mathrm { s t a } } = r _ { h } \bar { P } _ { n } ,\tag{9}
$$

where $r _ { h } \in ( 0 , 1 )$ is the host static-power ratio, representing the fraction of peak power consumed when the host is idle.

Similarly, the mean power of container $c _ { m }$ is proportional to its share of CPU cores on the hosting host:

$$
\bar { P } _ { m } = \frac { \bar { P } _ { \eta _ { m } } ( 1 - r _ { h } ) C _ { m } } { C _ { \eta _ { m } } } .\tag{10}
$$

Here, $\left( 1 - r _ { h } \right)$ represents the dynamic portion of the host power that is allocated to containers according to their CPUcore shares.

Under the linear power-performance model based on dynamic voltage and frequency scaling [1, 22], the dynamic power consumed while task $t _ { k , j }$ executes on container $c _ { m }$ scales with the task-specific capacity realization:

$$
P _ { k , j } ^ { ( m ) } = \bar { P } _ { m } \frac { Q _ { k , j } ^ { ( m ) } } { \bar { Q } _ { m } } .\tag{11}
$$

Consequently, the power consumption can differ between successive tasks on the same container because a new capacity realization is drawn for each execution.

2) Host static energy: A host is considered active while at least one container is deployed on it. During its active period, the host incurs static energy consumption proportional to its powered-on duration:

$$
E _ { n } ^ { \mathrm { s t a } } = P _ { n } ^ { \mathrm { s t a } } T _ { n } ^ { \mathrm { o n } } = r _ { h } \bar { P } _ { n } T _ { n } ^ { \mathrm { o n } } ,\tag{12}
$$

where $T _ { n } ^ { \mathrm { o n } }$ denotes the continuous powered-on duration of host $H _ { n }$ , from its activation until it is shut down after its last container is terminated.

3) Container active energy: The active energy of container $c _ { m }$ is the sum of the energy consumed by all tasks executed on it:

$$
E _ { m } ^ { \mathrm { a c t } } = \sum _ { ( k , j ) : \mu _ { k , j } = m } { P _ { k , j } ^ { ( m ) } } \tau _ { k , j } ^ { \mathrm { e x } } = \frac { \bar { P } _ { m } } { \bar { Q } _ { m } } \sum _ { ( k , j ) : \mu _ { k , j } = m } p _ { k , j } ,\tag{13}
$$

where the second equality follows from (3) and (11). Under the adopted linear model, the sampled speed changes power and duration in opposite directions, so their effects cancel for a fixed workload.

4) Container idle energy: Between consecutive task executions, a container remains alive but idle and still consumes a reduced level of power. The idle energy of container $c _ { m }$ is

$$
E _ { m } ^ { \mathrm { i d l e } } = r _ { c } \bar { P } _ { m } \left( T _ { m } ^ { \mathrm { l i f e } } - T _ { m } ^ { \mathrm { a c t } } \right) ,\tag{14}
$$

where $r _ { c } \in ( 0 , 1 )$ is the container idle-power ratio, $T _ { m } ^ { \mathrm { l i f e } }$ is the total lifetime of container $c _ { m }$ from creation to termination, and $T _ { m } ^ { \mathrm { a c t } }$ is its cumulative active execution time.

5) Total energy consumption: The total energy consumption of host $H _ { n }$ aggregates the host static energy and the energy consumed by all containers deployed on it:

$$
E _ { n } = E _ { n } ^ { \mathrm { s t a } } + \sum _ { c _ { m } \in \mathcal { C } _ { n } ^ { \mathrm { a l l } } } \left( E _ { m } ^ { \mathrm { a c t } } + E _ { m } ^ { \mathrm { i d l e } } \right) .\tag{15}
$$

Although $Q _ { k , j } ^ { ( m ) }$ does not alter the active energy of a fixed workload under the linear model, it changes the execution timeline through $\tau _ { k , j } ^ { \mathrm { e x } } = p _ { k , j } / Q _ { k , j } ^ { ( \mu _ { k , j } ) }$ . The resulting shifts in task completion, data readiness, and container reuse affect container idle durations and host powered-on durations. Therefore, energy differences mainly arise from these timeline effects, the selected container and host types, and the degree of resource reuse, rather than from a direct speed multiplier on active energy.

## E. Problem Formulation

Let π denote a scheduling policy that determines both taskto-container assignments and container-to-host placements. Given a workflow set ${ \mathcal W } ~ = ~ \{ W _ { 1 } , \ldots , W _ { K } \}$ arriving over a finite scheduling horizon and a cloud infrastructure $\mathcal { H } =$ $\{ H _ { 1 } , \ldots , H _ { N } \}$ , we aim to find a policy π that jointly optimizes three objectives subject to resource-capacity, task-dependency, and assignment constraints.

1) Optimization objectives: The goal is to maximize the workflow scheduling success rate and the container utilization while minimizing the total energy consumption. The three objectives are defined as follows:

$$
\left\{ \begin{array} { l l } { \displaystyle { J _ { \mathrm { s u c } } ( \pi ) = \frac { N _ { \mathrm { s u c c } } } { K } } } \\ { \displaystyle { J _ { \mathrm { u t i } } ( \pi ) = \frac { 1 } { | \mathcal { C } ^ { \mathrm { a l l } } | } \sum _ { c _ { m } \in \mathcal { C } ^ { \mathrm { a l l } } } \frac { T _ { m } ^ { \mathrm { a c t } } } { T _ { m } ^ { \mathrm { l i f e } } } } } \\ { \displaystyle { J _ { \mathrm { e n e } } ( \pi ) = \sum _ { n = 1 } ^ { | \mathcal { H } | } E _ { n } } } \end{array} \right.\tag{16}
$$

where $N _ { \mathrm { s u c c } } = | \{ k : F _ { k } \leq D _ { k } \} |$ is the number of workflows completed before their deadlines. The policy aims to maximize $J _ { \mathrm { s u c } } ( \pi )$ and $J _ { \mathrm { u t i } } ( \pi )$ while minimizing $J _ { \mathrm { e n e } } ( \pi )$

2) Problem constraints: The scheduling solution must satisfy the following constraints.

C1 (CPU capacity): for each host $H _ { n }$ and each time $t ,$

$$
\sum _ { c _ { m } \in \mathcal { C } _ { n } ( t ) } C _ { m } \leq C _ { n }\tag{17}
$$

ensures that the aggregate CPU-core demand of all currently alive containers does not exceed the host capacity.

C2 (memory capacity): for each host $H _ { n }$ and each time instance t,

$$
\sum _ { c _ { m } \in \mathcal { C } _ { n } ( t ) } M _ { m } \leq M _ { n }\tag{18}
$$

ensures that the aggregate memory demand of all currently alive containers does not exceed the host capacity.

C3 (dependency relationship):

$$
S _ { k , j } \geq F _ { k , i } + \tau _ { k , i j } ^ { \mathrm { t x } } , \forall t _ { k , i } \in \mathrm { p r e d } ( t _ { k , j } ) ,\tag{19}
$$

requires that the output of every immediate predecessor arrives before task $t _ { k , j }$ starts. For an entry task, the predecessor set is empty, so C3 is vacuously satisfied and its start time follows the first branch of (6).

C4 (unique assignment):

$$
\sum _ { m = 1 } ^ { | \mathcal { C } ^ { \mathrm { a l l } } | } x _ { k , j , m } \leq 1 ,\tag{20}
$$

where $x _ { k , j , m } \in \{ 0 , 1 \}$ is a binary assignment variable such that $x _ { k , j , m } = 1$ if task $t _ { k , j }$ is assigned to container $c _ { m } ,$ and $x _ { k , j , m } = 0$ otherwise. The inequality allows $\begin{array} { r } { \sum _ { m } x _ { k , j , m } = 0 } \end{array}$ when a task is discarded after its workflow deadline, while a scheduled task is assigned to at most one container.

The global objectives in (16) can only be evaluated after all workflows have been processed. In practice, however, the policy π makes decisions at the granularity of individual tasks and containers: for each ready task, a container is selected; for each new container, a host is selected. The global objectives therefore emerge from the accumulation of these per-step decisions over all scheduling intervals.

Since the scheduling decisions at each interval depend only on the currently observed system state and cannot anticipate future workflow arrivals or container-speed realizations, this problem is a dynamic stochastic optimization that is NPhard even in its static deterministic variant [8]. Moreover, because scheduling decisions are triggered by workflow-arrival and task-completion events rather than by fixed-duration time slots, the process is modeled as an SMDP. The TS and CS agents have separate decision sequences: TS decisions are generated for the currently ready tasks, whereas CS decisions are generated for undeployed containers produced by newcontainer requests. After the current TS phase is completed, the corresponding CS placement decisions are processed before physical time advances. Consequently, successive decisions of each agent may be separated by different amounts of physical time.

## IV. SCHEDULING ALGORITHMS

The GA-HRL scheduler follows a four-stage procedure that mirrors the scheduling model introduced above. First, the currently active workflows are kept in DAG form and preprocessed to obtain task-level timing indicators from their predicted execution time. Second, a multi-head GAT updates each task representation using information from its dependency neighborhood. Third, the TS agent combines the resulting task representation with the current container state and decides whether a ready task should reuse an admissible existing container or request a new container. Fourth, if a new container is requested, the CS agent selects a feasible host for it. This ordering connects the graph representation directly to the two coupled resource-allocation decisions, rather than treating GAT, hierarchical control, and policy generation as independent components.

The two decision layers operate in the same event-driven environment and can be described as two interacting SMDPs. Workflow arrivals and task completions trigger scheduling events. At each event, the TS agent first processes all currently ready tasks sequentially. New-container requests generated during the TS phase are added to a deployment queue. After the TS phase is completed, the CS agent processes all undeployed containers in the queue and places them on hosts. Physical epochs advances only after both the TS and CS decision phases associated with the current scheduling event have been completed. Because TS and CS are invoked at different frequencies, each agent has its own sequence of decision epochs and its own holding times between successive invocations.

Specifically, the TS Agent operates at its own decision epochs: it observes the dependency-aware representation of the selected ready task and the admissible containers, then either reuses an existing container with available capacity or requests a new one. Its reward combines predicted deadline margin, uncertainty-sensitive energy proxies, and container availability. The CS Agent, in contrast, is activated only after the TS phase is completed, and each undeployed container from a new-container request triggers a CS decision. The CS agent observes container requirements, feasible host capacities, and predecessor locality, then places the container on an existing or newly activated host. If the TS phase produces no new-container request, no CS decision occurs for that event.

For each agent $g ~ \in ~ \{ \mathrm { T S } , \mathrm { C S } \}$ , the induced process is represented as $\mathcal { M } ^ { g } = ( S ^ { g } , \mathcal { A } ^ { g } , P ^ { g } , R ^ { g } , \Delta ^ { g } )$ . Here, $S ^ { g } , A ^ { g }$ $P ^ { g }$ , and $R ^ { g }$ denote the state space, action space, transition map, and reward function of agent $^ { g , }$ respectively. Let $T _ { q } ^ { g }$ denote the starting time of the $q \cdot$ -th decision epoch of agent $^ { g , }$ and let $\Delta _ { q } ^ { g } = T _ { q + 1 } ^ { g } - T _ { q } ^ { g }$ denote the corresponding holding time between two consecutive decisions of the same agent. The holding time characterizes the irregular physical-time spacing between successive decisions and affects the subsequent state through task execution, data transmission, workflow arrivals, task completions, and resource-state evolution. For policy optimization, the successive invocations of each agent form its decision-epoch trajectory. Fig. 2 summarizes this complete path from workflow representation to hierarchical decision making and training.

## A. Workflow Embedding for State Construction

The state representation used by the TS policy is constructed in two complementary steps. The first step computes a reference timing profile for each workflow before task assignment, providing an estimate of task urgency. The second step applies a GAT to the active workflow DAGs, producing task embeddings that combine local task attributes with dependency information. The reference timing profile is not intended to guarantee feasibility under every stochastic execution realization, and the GAT itself does not make resource decisions. Instead, the two parts jointly define the task-level state consumed by the TS policy.

1) Sub-deadline calculation: Following the timing procedure adopted in the Stochastic Hybrid Workflows Scheduling (SHWS) system [1], we construct a reference timing profile using execution time that is available before task assignment. Let $\bar { Q } _ { \mathrm { r e f } }$ denote the nominal capacity of a high-capacity reference container. The reference execution time of task $t _ { k , j }$ is defined as

$$
\begin{array} { r } { \widehat { \tau } _ { k , j } ^ { \mathrm { e x , r e f } } = \frac { p _ { k , j } } { \bar { Q } _ { \mathrm { r e f } } } . } \end{array}\tag{21}
$$

Based on this reference profile, the forward pass computes the Earliest Start Time (EST) of each task:

$$
\begin{array} { r l } & { E S T _ { k , j } = } \\ & { \left\{ \begin{array} { l l } { \begin{array} { r l } { A _ { k } , } & { \mathrm { i f ~ } t _ { k , j } = t _ { k , \mathrm { e n t r y } } , } \\ & { \mathrm { m a x } } \end{array} } \right.} \\ & { \begin{array} { r l } & { t _ { k , i \mathrm { \ell } } \mathrm { e p r e d } ( t _ { k , j } ) } \end{array} } \end{array}  \{ E S T _ { k , i } + \widehat { \tau } _ { k , i } ^ { \mathrm { e x , r e f } } + d _ { k , i j } / B ^ { c r } \} , \mathrm { o t h e r w i s e . }  \end{array}\tag{22}
$$

The Earliest Completion Time (ECT) is then

$$
E C T _ { k , j } = E S T _ { k , j } + \widehat \tau _ { k , j } ^ { \mathrm { e x , r e f } } .\tag{23}
$$

Similarly, the backward pass computes the Latest Completion Time (LCT) of each task:

$$
\begin{array} { r l } & { L C T _ { k , j } = } \\ & { \left\{ \begin{array} { l l } { D _ { k } , \quad \mathrm { i f } \ t _ { k , j } = t _ { k , \mathrm { e x i t } } , } \\ { \displaystyle \operatorname* { m i n } _ { t _ { k , r } \in \mathrm { s u c c } ( t _ { k , j } ) } \{ L C T _ { k , r } - \widehat { \tau } _ { k , r } ^ { \mathrm { e x , r e f } } - d _ { k , j r } / B ^ { c r } \} , \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{24}
$$

Using the partial critical path-based deadlinedistribution [23] form adopted in SHWS, and taking the workflow entry and exit tasks as global anchors, the sub-deadline of task $t _ { k , j }$ is

$$
\begin{array} { l } { { \displaystyle D _ { k , j } ^ { \mathrm { s u b } } = E S T _ { k , \mathrm { e n t r y } } + \frac { E C T _ { k , j } - E S T _ { k , \mathrm { e n t r y } } } { E C T _ { k , \mathrm { e x i t } } - E S T _ { k , \mathrm { e n t r y } } } } } \\ { { \displaystyle \qquad \times \left( L C T _ { k , \mathrm { e x i t } } - E S T _ { k , \mathrm { e n t r y } } \right) . } } \end{array}\tag{25}
$$

These timing quantities are heuristic urgency indicators used in task features and reward shaping, rather than hard guarantees under every stochastic capacity realization. The actual success of a workflow is determined only by the realized completion condition $F _ { k } \le D _ { k }$

2) Graph embedding with GAT: The sub-deadline calculation provides each task with an urgency indicator, but it does not yet encode the dependency structure of the workflow. To capture this structural information, we apply a multi-head GAT to the active workflow DAGs following the attention mechanism in [24].

Because multiple workflows may coexist at a decision epoch, we merge all active workflows into a single DAG. Let $I _ { t }$ denote the current event-driven decision epoch. At epoch $I _ { t } ,$ a pseudo-entry task $t _ { \mathrm { p s d - e n t e r } }$ is prepended to all entry tasks, and a pseudo-exit task $t _ { \mathrm { p s d - e x i t } }$ is appended after all exit tasks. Both pseudo tasks have zero execution and transfer cost, and their initial feature vectors are set to zero because they do not correspond to real computational tasks. These auxiliary nodes are used only to connect multiple active workflows into one graph and are never scheduled as real tasks. Through the above processing, we can use GAT on multiple workflows, and the detailed information is shown in Fig. 2.

![](images/38b2a5bb38304988fd78667f3252e81fbb007563374e729957e8c2085bda5fbc.jpg)  
Fig. 2: Schematic diagram of the GA-HRL scheduling algorithm.

For each original task $t _ { k , j }$ in the merged graph, we denote its node by $t _ { k , j } ^ { I _ { t } }$ . To characterize the communication demand from its immediate predecessors, we define the maximum predecessor data volume as

$$
d _ { k , j } ^ { \mathrm { p r e d , m a x } } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m a x } _ { t _ { k , i } \in \mathrm { p r e d } ( t _ { k , j } ) } d _ { k , i j } , } & { \mathrm { p r e d } ( t _ { k , j } ) \neq \emptyset , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{26}
$$

The initial feature vector of task $t _ { k , j } ^ { I _ { t } }$ is then

$$
\begin{array} { r } { \pmb { h } _ { k , j } ^ { I _ { t } } ( 0 ) = \left[ D _ { k , j } ^ { \mathrm { s u b } } , E S T _ { k , j } , d _ { k , j } ^ { \mathrm { p r e d , m a x } } , p _ { k , j } , \left| \mathrm { s u c c } ( t _ { k , j } ^ { I _ { t } } ) \right| \right] _ { \alpha } , } \end{array}\tag{27}
$$

where $D _ { k , j } ^ { \mathrm { s u b } }$ is the sub-deadline obtained by assuming that $t _ { k , j } ^ { I _ { t } }$ is scheduled to a newly created container with the best computational capacity, and | succ $( t _ { k , j } ^ { I _ { t } } ) |$ is the number of immediate successors of $t _ { k , j } ^ { I _ { t } }$

The GAT encoder consists of several stacked graph attention layers. Let $N ^ { I _ { t } } = | T ^ { I _ { t } } |$ denote the number of nodes in the merged DAG at epoch $I _ { t }$ . The first-layer input feature is

$$
\pmb { H } ^ { \mathrm { g a t } } ( 0 ) = \left[ \pmb { h } _ { 1 } ^ { I _ { t } } ( 0 ) , \pmb { h } _ { 2 } ^ { I _ { t } } ( 0 ) , \dots , \pmb { h } _ { N ^ { I _ { t } } } ^ { I _ { t } } ( 0 ) \right] ^ { \top } ,\tag{28}
$$

where $d _ { 0 }$ is the initial feature dimension. At layer ℓ, the GAT layer updates each node representation by attending over its neighbors, producing the refined feature matrix

$$
\pmb { H } ^ { \mathrm { g a t } } ( \ell ) = \left[ \pmb { h } _ { 1 } ^ { I _ { t } } ( \ell ) , \dots , \pmb { h } _ { N ^ { I _ { t } } } ^ { I _ { t } } ( \ell ) \right] ^ { \top } \in \mathbb { R } ^ { N ^ { I _ { t } } \times d _ { h } } .\tag{29}
$$

where $d _ { h }$ denotes the embedding dimension at the output of layer ℓ.

To retain the node’s own information during aggregation and to avoid an empty neighborhood for exit nodes, we define the augmented successor neighborhood

$$
\mathcal { N } _ { k , j } ^ { + } = \operatorname { s u c c } ( t _ { k , j } ^ { I _ { t } } ) \cup \left\{ t _ { k , j } ^ { I _ { t } } \right\} .\tag{30}
$$

For head z of layer ℓ, the attention coefficient between $t _ { k , j } ^ { I _ { t } }$ and a node $t _ { k , r } ^ { I _ { t } } \in \mathcal { N } _ { k , j } ^ { + }$ is

$$
\alpha _ { k , j r } ^ { I _ { t } } ( \ell , z ) = \frac { \exp \Big ( e _ { k , j r } ^ { I _ { t } } ( \ell , z ) \Big ) } { \sum _ { t _ { k , r ^ { \prime } } ^ { I _ { t } } \in \mathcal { N } _ { k , j } ^ { + } } \exp \Big ( e _ { k , j r ^ { \prime } } ^ { I _ { t } } ( \ell , z ) \Big ) } .\tag{31}
$$

The scalar attention score is computed as

$$
\begin{array} { r } { e _ { k , j r } ^ { I _ { t } } ( \ell , z ) = \mathrm { L e a k y R e L U } \Big ( \pmb { a } ^ { \top } ( \ell , z ) \big [ W ( \ell , z ) { h } _ { k , j } ^ { I _ { t } } ( \ell - 1 ) } \\ { \big \| W ( \ell , z ) { h } _ { k , r } ^ { I _ { t } } ( \ell - 1 ) \big ] \Big ) , \qquad ( 3 ^ { \cdot } } \end{array}\tag{2}
$$

where $W ( \ell , z )$ is the learnable feature-transformation matrix and $\mathbf { \delta } _ { \mathbf { { \pmb { a } } } \left( \ell , z \right) }$ is the learnable attention vector of head z in layer ℓ. The node representation is then updated by multi-head aggregation over $\mathcal { \bar { N } } _ { k , j } ^ { + }$

$$
{ h } _ { k , j } ^ { I _ { t } } ( \ell ) = \sigma \Bigg ( \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } \sum _ { t _ { k , r } ^ { I _ { t } } \in \mathcal { N } _ { k , j } ^ { + } } { \alpha _ { k , j r } ^ { I _ { t } } ( \ell , z ) { W } ( \ell , z ) { h } _ { k , r } ^ { I _ { t } } ( \ell - 1 ) } \Bigg ) ,\tag{33}
$$

where $Z$ is the number of attention heads and $\sigma ( \cdot )$ is the activation function.

We denote the resulting stacked GAT encoder by $f _ { \psi } ^ { \mathrm { G A T } } ,$ where ψ collects all learnable parameters $\{ \mathbf { \bar { W } } ( \ell , z ) , \mathbf { a } ( \ell , z ) \} _ { \ell , z }$

## B. Task Scheduling (TS) Agent

The TS agent determines the container assignment for each ready task according to both the task characteristics and the current container state. This subsection defines its state representation, action space, and reward function.

1) State representation: At a TS decision epoch, the TS agent sequentially processes all currently ready tasks before physical time advances. For task $t _ { k , j }$ scheduled at step q, the state $s _ { q } ^ { \mathrm { T S } }$ contains the following information:

• the embedding vector of the selected task $t _ { k , j } ;$

• for each existing candidate container $c _ { m }$ , its hosting host $\eta _ { m }$ , CPU-core allocation $C _ { m }$ , memory allocation $M _ { m } .$ , mean computational capacity $\bar { Q } _ { m } .$ , execution-speed variation coefficient $v ,$ current ready time or workload, and the estimated predecessor-to-container data-transfer delays under the current placement;

• for each candidate new-container type, its CPU-core allocation, memory allocation, nominal computational capacity, and execution-speed variation coefficient v. The hostdependent transmission information of a new container is determined only after the CS agent selects its host.

2) Action space: At TS decision step $q ,$ the agent chooses either to assign the selected task to an admissible existing container or to request a new container of a specified type. Let $\mathcal { C } _ { q } ^ { \mathrm { f e a s } }$ denote the set of existing containers that are feasible under the current scheduling rules, $N _ { c } ^ { \mathrm { t y p e } }$ the number of available container types, and $\iota _ { i }$ the action of creating a new container of type i. The TS action space $( a _ { q } ^ { \mathrm { T S } } \in \mathcal { A } _ { q } ^ { \mathrm { T S } } )$ is

$$
\mathcal { A } _ { q } ^ { \mathrm { T S } } = \mathcal { C } _ { q } ^ { \mathrm { f e a s } } \cup \left\{ \iota _ { i } ~ | ~ i = 1 , \dots , N _ { c } ^ { \mathrm { t y p e } } \right\} .\tag{34}
$$

An existing container belongs to $\mathcal { C } _ { q } ^ { \mathrm { f e a s } }$ only if it has no waiting task; hence it contains either one executing task or is idle. A container that already has one executing task and one waiting task is masked out before policy sampling. If $a _ { q } ^ { \mathrm { T S } } =$ $c _ { m }$ , the selected task occupies the sole waiting position when $c _ { m }$ is busy, or the execution position when $c _ { m }$ is idle. If $a _ { q } ^ { \mathrm { T S } } =$ $\iota _ { i } ,$ a new container of type i is requested and added to the deployment queue. After the current TS phase is completed, the corresponding undeployed container is subsequently placed by the CS agent.

3) Reward function: For task $t _ { k , j }$ and candidate container $c _ { m }$ , the TS reward evaluates the expected scheduling quality of each feasible action:

$$
\begin{array} { r l } & { r _ { q } ^ { \mathrm { T S } } = w _ { 1 } ^ { \mathrm { T S } } \frac { \left( D _ { k , j } ^ { \mathrm { s u b } } - E S T _ { k , j } \right) ( 1 + v ) } { \widehat { T } _ { k , j , m } ^ { \mathrm { e x } } } } \\ & { ~ + w _ { 2 } ^ { \mathrm { T S } } \left( 1 - \frac { \bar { E } _ { m } ^ { \mathrm { a c t } } + \bar { E } _ { m } ^ { \mathrm { i d l e } } } { \bar { E } _ { \mathrm { a c t , s u b } } + \bar { E } _ { \mathrm { i d l e , s u b } } } \right) + w _ { 3 } ^ { \mathrm { T S } } \frac { E S T _ { k , j } - R _ { m } } { T _ { \mathrm { n o r m } } } , } \end{array}\tag{35}
$$

where $\widehat { \tau } _ { k , j , m } ^ { \mathrm { e x } } \ = \ p _ { k , j } / \bar { Q } _ { m }$ is the predicted execution time. The realized time $p _ { k , j } / Q _ { k , j } ^ { ( m ) }$ is sampled only when the task actually starts and is not used to rank actions. The containeravailability term is positive when $c _ { m }$ is expected to be ready before $E S T _ { k , j }$ and becomes negative when the task would wait beyond that reference time.

For an existing container, $\hat { Q } _ { m }$ and $\bar { P } _ { m }$ are evaluated using its actual hosting host. For a newly requested container whose placement has not yet been determined, the TS agent evaluates the reward by assuming that the container is deployed on the feasible host that yields the minimum estimated energy consumption. This assumption is used only to evaluate the new-container action; the actual host is subsequently selected by the CS agent.

The factor $( 1 + v )$ is introduced as an uncertainty-sensitive term for reward shaping. It should not be interpreted as a probabilistic upper bound or as a physical correction applied to the execution-time or energy model. Its purpose is to give more weight to preserving sub-deadline margin when the executionspeed variation becomes larger. Similarly, $\bar { E } _ { m } ^ { \mathrm { a c t } }$ and $\bar { E } _ { m } ^ { \mathrm { i d l e } }$ are energy proxies used to rank candidate actions rather than realized energy values. We define

$$
\bar { E } _ { m } ^ { \mathrm { a c t } } = \frac { \bar { P } _ { m } p _ { k , j } ( 1 + v ) } { \bar { Q } _ { m } } ,\tag{36}
$$

solely to introduce an uncertainty-sensitive penalty into the reward. This proxy does not contradict the system energy model, in which the realized active energy of a fixed workload is independent of the sampled speed under the linear powerperformance assumption.

Using the maximum predecessor data volume $d _ { k , j } ^ { \mathrm { p r e } }$ d,max defined in (26), we set $\bar { E } _ { m } ^ { \mathrm { i d l e } } ~ = ~ \bar { P } _ { m } r _ { c } d _ { k , j } ^ { \mathrm { p r e d , m a x } } / B ^ { c r }$ . The corresponding values for the fastest reference container are $\bar { E } _ { \mathrm { a c t , s u b } }$ and $\bar { E } _ { \mathrm { i d l e , s u b } }$ . The coefficients $w _ { 1 } ^ { \mathrm { T S } } , w _ { 2 } ^ { \mathrm { T S } }$ , and $w _ { 3 } ^ { \mathrm { T S } }$ control the three terms, and $T _ { \mathrm { n o r m } }$ is a constant used to normalize the container-availability term.

## C. Container Scheduling (CS) Agent

When the TS agent requests a new container, the request is added to the deployment queue. After the current TS phase is completed, the CS agent selects an appropriate host for each undeployed container according to the container requirements and the current host states. If no existing host satisfies the deployment requirements, the CS agent activates a new host. This subsection defines the CS state representation, action space, and reward function.

1) State representation: After the TS agent completes the current ready-task assignment phase, each undeployed container waiting for host placement triggers one CS decision. Thus, a TS phase may be followed by zero, one, or multiple CS decisions, depending on the number of new-container requests generated during task scheduling. The CS agent therefore maintains its own event-driven decision sequence.

For container $c _ { m }$ scheduled at step $q ,$ the state $s _ { q } ^ { \mathrm { C S } }$ includes the following information:

• the resource requirements of the selected container, including its CPU-core requirement, memory requirement, nominal computational capacity, and the data-transfer speeds achievable on different hosts;

• the current workload of each host, including the number of occupied CPU cores, occupied memory, available CPU percentage, available memory percentage, CPU utilization, and mean energy consumption;

• the host and container IDs of the critical predecessor task. For each predecessor $t _ { k , i } \in \mathrm { { p r e d } } ( t _ { k , j } )$ , its data-ready time for the current task is estimated from its start time, the execution time $p _ { k , i } / \bar { Q } _ { \mu _ { k , \cdot } }$ on its assigned container, and the corresponding data-transfer time to $t _ { k , j }$ . The predecessor with the largest estimated data-ready time is regarded as the critical predecessor, because the current task cannot start until the outputs of all its predecessors have arrived.

2) Action space: Let $C _ { n } ^ { \mathrm { u s e d } } ( q )$ and $M _ { n } ^ { \mathrm { u s e d } } ( q )$ denote the number of occupied CPU cores and the occupied memory on host $H _ { n }$ at CS decision step $q .$ The feasible existing-host set for container $c _ { m }$ is

$$
\begin{array} { r l r } {  { \mathcal { H } _ { q } ^ { \mathrm { f e a s } } = } } \\ & { \cdot \big \{ H _ { n } \in \mathcal { H } : C _ { n } - C _ { n } ^ { \mathrm { u s e d } } ( q ) \ge C _ { m } , M _ { n } - M _ { n } ^ { \mathrm { u s e d } } ( q ) \ge M _ { m } \big \} . } \end{array}
$$

Let $N _ { h } ^ { \mathrm { t y p e } }$ denote the number of available host types, and let $\xi _ { i }$ denote the action of activating a new host of type i. The CS action space is

$$
\mathcal { A } _ { q } ^ { \mathrm { C S } } = \mathcal { H } _ { q } ^ { \mathrm { f e a s } } \cup \left\{ \xi _ { i } \mid i = 1 , \dots , N _ { h } ^ { \mathrm { t y p e } } \right\} , a _ { q } ^ { \mathrm { C S } } \in \mathcal { A } _ { q } ^ { \mathrm { C S } } .\tag{38}
$$

Before policy sampling, an action mask removes all existing hosts that violate either the CPU-core or memory requirement of the selected container. Therefore, infeasible host-placement actions cannot be selected by the CS policy.

3) Reward function: Among feasible host-placement actions, the CS agent aims to improve data locality and resource consolidation. Let $H _ { k , j } ^ { \mathrm { c r i t } }$ denote the host on which the critical predecessor of task $t _ { k , j }$ is deployed, and define the locality indicator

$$
x _ { q } ^ { \mathrm { l o c } } = \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { i f ~ c o n t a i n e r ~ } c _ { m } \mathrm { ~ i s ~ d e p l o y e d ~ o n ~ } H _ { k , j } ^ { \mathrm { c r i t } } , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{39}
$$

For an entry task with no predecessor, no critical predecessor exists, and $x _ { q } ^ { \mathrm { l o c } }$ is set to zero.

The CS reward is then

$$
r _ { q } ^ { m a t h r m C S } = w _ { 1 } ^ { \mathrm { C S } } x _ { q } ^ { \mathrm { l o c } } + w _ { 2 } ^ { \mathrm { C S } } \frac { C _ { \eta _ { m } } ^ { \mathrm { u s e d } } ( q ) + C _ { m } } { C _ { \eta _ { m } } } ,\tag{40}
$$

where $w _ { 1 } ^ { \mathrm { C S } }$ , and $w _ { 2 } ^ { \mathrm { C S } }$ are the reward coefficients of the CS agent. $C _ { \eta _ { m } }$ and $C _ { \eta _ { m } } ^ { \mathrm { u s e d } } ( q )$ are the total and used CPU cores of the host selected by the action, respectively, and $C _ { m }$ is the CPU-core requirement of container $c _ { m }$ . The first term rewards data locality with the critical predecessor, while the second term promotes resource consolidation among feasible hosts.

The complete online scheduling procedure of GA-HRL is summarized in Algorithm 1. The algorithm combines the workflow representation, TS decisions, CS decisions, and the subsequent timing and energy updates into one event-driven procedure.

## D. Policy Derivation with PPO

Both the TS and CS agents are trained with PPO [25]. Although the underlying scheduling environment is eventdriven and successive decision epochs may be separated by nonuniform physical holding times, each invocation of an agent is treated as one step in that agent’s decision-epoch trajectory. Accordingly, we optimize the decision-indexed discounted return:

$$
J ^ { g } ( \pi _ { g } ) = \mathbb { E } _ { \pi _ { g } } \left[ \sum _ { q = 0 } ^ { Q _ { g } - 1 } \gamma ^ { q } r _ { q } ^ { g } \right] ,\tag{41}
$$

where $Q _ { g }$ is the number of decisions made by agent $g$ in an episode, and $\gamma \in ( 0 , 1 )$ discounts successive decisions rather than elapsed physical time. The holding time $\Delta _ { q } ^ { g }$ affects the transition to the next decision state through the physical system evolution, but is not used as an additional time-dependent discount.

Algorithm 1 GA-HRL Online Scheduling.   
Require: Trained TS policy $\pi _ { \theta _ { \mathrm { T S } } }$ , trained CS policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { C S } } }$ , and   
trained GAT encoder $f _ { \psi } ^ { \mathrm { { \bar { G } A T } } }$ ; workflow stream W; resource   
pool H.   
Ensure: Task-to-container assignments and container-to-host   
placements.   
1: Initialize the scheduling environment and resource states;   
2: while the scheduling episode is not terminated do   
3: Process workflow arrivals and task completions; for   
newly arrived workflows, compute timing indicators   
using (22)-(25) and update ready set $\mathcal { R } ;$   
4: Construct the merged DAG and update task embeddings   
using the trained GAT encoder $f _ { \psi } ^ { \dot { \mathrm { G A T } } }$ according to (27)-   
(33);   
5: while $\mathcal { R } \neq \emptyset$ do   
6: Select $t _ { k , j } ,$ construct $s _ { q } ^ { \mathrm { T S } }$ from (34), and sample   
$a _ { q } ^ { \mathrm { T S } } \sim \pi _ { \theta _ { \mathrm { T S } } } ( \cdot | s _ { q } ^ { \mathrm { T S } } )$   
7: Execute $a _ { q } ^ { \mathrm { T S } }$ by reusing a feasible container or re  
questing a new one; enqueue the new container if   
requested;   
8: end while   
9: while the deployment queue is not empty do   
10: Select $c _ { m } ;$ construct $ { \hat { s } } _ { q } ^ { \mathrm { C S } }$ using (37)-(38), and sample   
$a _ { q } ^ { \mathrm { C S } } \sim \pi _ { \theta _ { \mathrm { C S } } } ( \cdot | s _ { q } ^ { \mathrm { C S } } ) ;$   
11: Deploy $c _ { m }$ to the selected feasible host or activate a   
new host;   
12: end while   
13: Advance to the next scheduling event; update execution,   
transmission, and task timing using (2), (3), (4), and   
(5)-(8), and update energy using (12)-(15);   
14: end while   
15: return the final scheduling decisions;

1) TD residual and GAE: In the following, $g \in \{ \mathrm { T S } , \mathrm { C S } \}$ denotes the agent, and q indexes its consecutive decisions. The Temporal-Difference (TD) residual of agent $g$ is

$$
\delta _ { q } ^ { g } = r _ { q } ^ { g } + m _ { q } ^ { g } \gamma V _ { \phi _ { g } } ( s _ { q + 1 } ^ { g } ) - V _ { \phi _ { g } } ( s _ { q } ^ { g } ) ,\tag{42}
$$

where $r _ { q } ^ { g }$ is the reward after decision $q , V _ { \phi _ { g } } ( \cdot )$ is the critic, and $m _ { q } ^ { g }$ is a non-terminal mask with $m _ { q } ^ { g } = 0$ for terminal transitions and $m _ { q } ^ { g } ~ = ~ 1$ otherwise. $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ is the perdecision discount factor. The generalized advantage estimate (GAE) is then

$$
\hat { A } _ { q } ^ { g } = \delta _ { q } ^ { g } + m _ { q } ^ { g } \gamma \lambda _ { \mathrm { G A E } } \hat { A } _ { q + 1 } ^ { g } ,\tag{43}
$$

where $\lambda _ { \mathrm { G A E } }$ controls the bias-variance tradeoff.

2) PPO objective: For both agents, the actor maximizes the clipped surrogate objective

$$
L _ { \pi } ^ { g } ( \theta _ { g } ) = \hat { \mathbb { E } } \Big [ \operatorname* { m i n } \big ( u _ { q } ^ { g } ( \theta _ { g } ) \hat { A } _ { q } ^ { g } , \mathrm { c l i p } ( u _ { q } ^ { g } ( \theta _ { g } ) , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } _ { q } ^ { g } \big ) \Big ] ,\tag{44}
$$

where

$$
u _ { q } ^ { g } ( \theta _ { g } ) = \frac { \pi _ { \theta _ { g } } \big ( a _ { q } ^ { g } \big | s _ { q } ^ { g } \big ) } { \pi _ { \theta _ { g } ^ { \mathrm { o l d } } } \big ( a _ { q } ^ { g } \big | s _ { q } ^ { g } \big ) }\tag{45}
$$

is the probability ratio between the current and previous policies, and clip(·) constrains this ratio to $[ 1 - \varepsilon , 1 + \varepsilon ]$

3) Critic and combined objective: The target used to train the critic is

$$
\hat { G } _ { q } ^ { g } = \hat { A } _ { q } ^ { g } + V _ { \phi _ { g } } ( s _ { q } ^ { g } ) ,\tag{46}
$$

and the critic minimizes the mean-squared value loss

$$
\begin{array} { r } { L _ { V } ^ { g } ( \phi _ { g } ) = \hat { \mathbb { E } } \left[ \left( V _ { \phi _ { g } } ( s _ { q } ^ { g } ) - \hat { G } _ { q } ^ { g } \right) ^ { 2 } \right] . } \end{array}\tag{47}
$$

The actor-critic parameters are updated by maximizing the combined PPO objective:

$$
J _ { \mathrm { P P O } } ^ { g } ( \theta _ { g } , \phi _ { g } ) = L _ { \pi } ^ { g } ( \theta _ { g } ) - c _ { 1 } L _ { V } ^ { g } ( \phi _ { g } ) + c _ { 2 } \widehat { \mathbb { E } } \left[ \mathrm { E n t } \big ( \pi _ { \theta _ { g } } ( \cdot \mid s _ { q } ^ { g } ) \big ) \right]\tag{48}
$$

where $c _ { 1 }$ is the value-loss coefficient, Ent(·) is policy entropy, and $c _ { 2 }$ (set to 0.01) is the entropy coefficient used in the implementation.

4) Alternating training: The two agents are trained alternately using separate actor-critic networks and rollout buffers. The alternating PPO training procedure is summarized in Algorithm 2.

In implementation, scheduling one task or deploying one container constitutes one decision step in the corresponding agent-specific trajectory. During the TS training phase, the current CS policy is kept fixed and is used to process the intervening CS decisions required to advance the shared environment. After the TS rollout is collected, the TS agent computes its GAE advantages and updates its actor-critic network using the TS rollout buffer. The updated TS policy is then kept fixed during the subsequent CS training phase, in which it processes the intervening task-scheduling decisions. After the CS rollout is collected, the CS agent computes its GAE advantages and updates its actor-critic network using the CS rollout buffer.

For the finite-horizon scheduling episodes considered here, the number of workflows, tasks, and corresponding TS/CS decisions is finite. With bounded rewards and $0 ~ < ~ \gamma ~ < ~ 1$ the decision-indexed return in (41) is well defined. For a fixed policy, standard policy evaluation under the per-decision discount retains the usual contraction property, so the nonuniform physical holding times do not alter the policy-evaluation formulation adopted here. Since PPO uses nonlinear function approximation and the two policies are coupled through the scheduling environment, however, we do not claim global convergence to an optimal joint policy. Instead, the training curves in our experiments (e.g., Fig. 3) are used to evaluate empirical convergence.

## V. PERFORMANCE EVALUATION

We evaluate GA-HRL using three metrics: workflow scheduling success rate, average container resource utilization, and total system energy consumption.

Algorithm 2 Alternating PPO Training of TS and CS Agents.   
Require: GAT encoder $f _ { \psi } ^ { \mathrm { G A T } }$ TS policy-value pair   
$( \pi _ { \theta _ { \mathrm { T S } } } , V _ { \phi _ { \mathrm { T S } } } ) ;$ ; CS policy-value pair $( \pi _ { \theta _ { \mathrm { C S } } } , V _ { \phi _ { \mathrm { C S } } } ) ; ~ \gamma ,$   
$\lambda _ { \mathrm { G A E } } , \varepsilon ;$ number of alternating epochs $E ;$ rollout lengths   
$L _ { \mathrm { T S } }$ and $L _ { \mathrm { C S } } ;$ PPO optimization epochs S.   
Ensure: Trained GAT encoder $f _ { \psi } ^ { \mathrm { G A T } }$ , TS policy $\pi _ { \theta _ { \mathrm { T S } } }$ , and CS   
policy $\pi _ { \theta _ { \mathrm { { C S } } } }$   
1: Initialize the GAT encoder, the two actor-critic networks,   
and rollout buffers $\mathcal { D } _ { \mathrm { T S } }$ and $\mathcal { D } _ { \mathrm { { C S } } } ;$   
2: for $e = 1$ to E do   
3: Fix $\pi _ { \theta _ { \mathrm { { C S } } } }$ and reset the environment;   
4: for $q = 1$ to $L _ { \mathrm { T S } }$ do   
5: Compute the current task embedding with $f _ { \psi ^ { \mathrm { o l d } } } ^ { \mathrm { G A T } }$ and   
construct $s _ { q } ^ { \mathrm { T S } } ;$   
6: Sample $a _ { q } ^ { \mathrm { T S } }$ from $\pi _ { \theta _ { \mathrm { T S } } ^ { \mathrm { o l d } } } ( \cdot | s _ { q } ^ { \mathrm { T S } } )$ and execute the action;   
7: Use fixed $\pi _ { \theta _ { \mathrm { { C S } } } }$ for intervening CS decisions;   
8: Store the TS transition in D<sub>TS</sub>;   
9: end for   
10: Calculate $\hat { A } _ { q } ^ { \mathrm { T S } }$ and $\hat { G } _ { q } ^ { \mathrm { T S } } ;$   
11: for $i = 1$ to S do   
12: Jointly update ψ, $V _ { \phi _ { \mathrm { T S } } }$ , and $\pi _ { \theta _ { \mathrm { T S } } }$ using the TS PPO   
objective;   
13: end for   
14: Fix the updated $f _ { \psi } ^ { \mathrm { G A T } }$ and TS actor-critic network, and   
reset the environment;   
15: for $q = 1$ to $L _ { \mathrm { C S } }$ do   
16: Use the fixed TS policy and encoder until a CS   
decision is triggered;   
17: Construct $s _ { q } ^ { \mathrm { C S } } ;$   
18: Sample $a _ { q } ^ { \mathrm { C S } ^ { \bar { 1 } } }$ from $\pi _ { \theta _ { \mathrm { C S } } ^ { \mathrm { o l d } } } ( \cdot | s _ { q } ^ { \mathrm { C S } } )$ and execute the action;   
19: Store the CS transition in $\mathrm { \bar { \mathcal { D } } _ { C S } } ;$   
20: end for   
21: Calculate $\hat { A } _ { q } ^ { \mathrm { C S } }$ and $\hat { G } _ { q } ^ { \mathrm { C S } } ;$   
22: for $i = 1$ to S do   
23: Update $V _ { \phi _ { \mathrm { C S } } }$ and $\pi _ { \theta _ { \mathrm { C S } } }$ using the CS PPO objective;   
24: end for   
25: Update policy/encoder parameters, clear both buffers,   
and evaluate the joint policy;   
26: end for

## A. Experimental Setup

1) Environment Configuration: All experiments were implemented in Python 3.11.13 with PyTorch 2.6.1 and sb3-contrib 2.7. Each reported metric is averaged over 50 independent evaluation runs with different random seeds, covering random workflow arrivals, container execution-speed variations, and learning-based scheduling decisions.

At the beginning of each scheduling episode, no host is active. Hosts are activated on demand by the CS agent. A container is terminated when it has neither an executing task nor a waiting task, and a host is shut down when no container remains on it. If the same physical resource is activated again later, it is treated as a new host instance with a new host ID for scheduling and energy accounting. Accordingly, N denotes the total number of host instances activated during an episode.

TABLE II: Parameters of host types
<table><tr><td>Type</td><td>CPU cores</td><td>Mean capacity (MIPS)</td><td>Memory (GB)</td><td>Mean power (W)</td></tr><tr><td>1</td><td>96</td><td>264000</td><td>384</td><td>795</td></tr><tr><td>2</td><td>192</td><td>528000</td><td>768</td><td>1600</td></tr></table>

2) Parameter Settings: We construct a trace-driven environment from the 2018 Alibaba cluster trace [26]. Eight container types are considered, with CPU allocations {1, 2, 4, 6, 8, 16, 24, 32} cores and memory allocations {4, 8, 16, 24, 32, 64, 96, 128} GB. The mean capacity and power of each container are derived from its hosting host and CPU-core share. The host configurations are summarized in Table II. Following [27], we use 5,200 workflow DAGs from the trace, each containing at least 10 tasks. The output data volume of each task is uniformly distributed over [500, 5000] MB, the cross-host bandwidth is 200 MB/s, and the intra-host bandwidth is 500 MB/s.

For each task execution, the capacity $Q _ { k , j } ^ { ( m ) }$ is sampled from the rejection-sampled model in (2). The execution-speed variation coefficient is $v \in \{ 0 , \ldots , 0 . 4 5 \}$ with a step size of 0.05. The number of workflows is $K \in \{ 1 0 0 , \ldots , 1 0 0 0 \}$ with a step size of 100, and workflow arrivals follow a Poisson process with rate $\lambda = 0 . 5$

The TS and CS agents use the same PPO hyperparameters: learning rate $1 \times 1 0 ^ { - 5 }$ , discount factor $\gamma ~ = ~ 0 . 9 9$ , GAE parameter $\lambda _ { \mathrm { G A E } } ~ = ~ 0 . 9 5 .$ , clipping range $\varepsilon ~ = ~ 0 . 2 .$ , valueloss coefficient $c _ { 1 } ~ = ~ 0 . 5 .$ , entropy coefficient $c _ { 2 } ~ = ~ 0 . 0 1$ ， and maximum gradient norm 0.5. Each PPO rollout contains 2,048 decision steps, with a minibatch size of 32 and two optimization epochs per update. The reward coefficients are $\begin{array} { r } { \hat { w } _ { 1 } ^ { \mathrm { T S } } ~ = ~ 0 . 4 , ~ \hat { w } _ { 2 } ^ { \mathrm { T S } } ~ \overset { ^ { \mathrm { ~ } } } { = } ~ 0 . \hat { 3 } , ~ w _ { 3 } ^ { \mathrm { T S } } ~ = ~ 0 . 3 , ~ w _ { 1 } ^ { \mathrm { C S } } ~ = ~ 0 . 7 . } \end{array}$ and $w _ { 2 } ^ { \mathrm { \bar { C S } } } ~ = ~ 0 . 3$ . Training is performed for 1,000 alternating epochs, with 2,048 TS and 2,048 CS training steps per epoch. Following [1], the deadline of workflow $W _ { k }$ is

$$
D _ { k } = A _ { k } + \alpha ^ { d f } S _ { k } ^ { \mathrm { f a s t } } ,\tag{49}
$$

where $\alpha ^ { d f } \in \{ 2 . 0 , . . . , 2 . 9 \}$ (with a step size of 0.1) is the deadline factor, and $S _ { k } ^ { \mathrm { f a s t } }$ is the critical-path length under the fastest-resource assumption. Let $\mathcal { P } _ { k }$ denote the set of entryto-exit paths of $W _ { k } ,$ and let $\mathcal { T } ( \mathcal { P } )$ and $\mathcal { E } ( \mathcal { P } )$ denote the task and edge sets on path P. Then

$$
S _ { k } ^ { \mathrm { f a s t } } = \operatorname* { m a x } _ { \mathcal { P } \in \mathcal { P } _ { k } } \left( \sum _ { \substack { t _ { k , i } \in \mathcal { T } ( \mathcal { P } ) } } \tau _ { k , i } ^ { \mathrm { e x , m i n } } + \sum _ { \substack { e _ { k , i j } \in \mathcal { E } ( \mathcal { P } ) } } \frac { d _ { k , i j } } { B ^ { c r } } \right)\tag{50}
$$

where $\tau _ { k , i } ^ { \mathrm { e x , m i n } }$ is the execution time of $t _ { k , i }$ on the fastest container type. Thus, a larger $\alpha ^ { d f }$ relaxes the deadline.

## B. Ablation Study

We construct four ablated versions to examine the contribution of each main component. M/A2C and M/DDQN replace PPO with A2C and DDQN, respectively. M/f-GAT removes the GAT encoder and uses the original task features directly as policy input. M/f-CS retains the learned TS policy but replaces the CS policy with a random feasible host-placement rule.

(a)  
![](images/d130312aac4e10e520ff6b4b4ea79be50410b0d93795a746ceb70845aaa912ea.jpg)

(b)  
![](images/3280c6409a43f72e5850d18287426074b72364d819b72ee476fbc5a8943cc6f1.jpg)  
Fig. 3: Training convergence of different methods: (a) average reward of the TS agent and (b) average reward of the CS agent.

TABLE III: Ablation study results of GA-HRL
<table><tr><td>Method</td><td>Workflow success (%)</td><td>Container resource utilization (%)</td><td>Energy consumption (J)</td></tr><tr><td>GA-HRL</td><td>100</td><td>65</td><td> $1 . 8 4 \times 1 0 ^ { 7 }$ </td></tr><tr><td>M/A2C</td><td>97</td><td>64</td><td> $1 . 9 8 \times 1 0 ^ { 7 }$ </td></tr><tr><td>M/DDQN</td><td>95</td><td>66</td><td> $2 . 0 2 \times 1 0 ^ { 7 }$ </td></tr><tr><td>M/f-GAT</td><td>100</td><td>65</td><td> $2 . 0 3 \times 1 0 ^ { 7 }$ </td></tr><tr><td>M/f-CS</td><td>92</td><td>70</td><td> $2 . 1 9 \times 1 0 ^ { 7 }$ </td></tr></table>

Fig. 3 compares training rewards, and Table III reports the final scheduling metrics.

The training curves in Fig. 3 show that GA-HRL and M/A2C converge to relatively high TS rewards, whereas M/DDQN exhibits larger fluctuations. On the CS side, M/f-CS remains substantially lower and more volatile because its host decisions are random. Table III further shows that removing GAT preserves workflow success but increases energy consumption, while removing the learned CS policy reduces success to 92% and increases energy to $2 . 1 9 \times 1 0 ^ { 7 } \ ]$ . These results indicate that the dependency-aware representation and learned host placement contribute in complementary ways: the former improves task-decision quality, and the latter coordinates locality and resource consolidation.

![](images/a2c6882e41afbcc2cdda74fe6200730bdf574f44d71fa580d13b2c719b94ecf2.jpg)

(b)  
![](images/4b37c20305830cc3c165e602ac1724297c082942eff21e146ab478c7a0d93b1f.jpg)

![](images/de03e9825f2a17cce1c8cb86f41351332e1f104e221c0507c85497849da33a63.jpg)  
Fig. 4: Effect of the deadline factor $\alpha ^ { d f }$ with $K = 1 0 0$ workflows, $\lambda = 0 . 5 .$ and $v = 0 . 1 \colon ( \mathrm { a } )$ workflow scheduling success rate, (b) average container resource utilization, and (c) total system energy consumption.

## C. Performance Comparison

1) Benchmark Settings: We compare GA-HRL with five baselines representing the approaches of learning, heuristic, and optimization: DTODRL [10], OHDS [6], SMWDSA [1], HACPPO [28], and DS-CSP [13]. OHDS includes its own container-placement strategy, which is retained. For baselines that do not define host placement, we keep their original task policy and select the second-level placement randomly from feasible hosts. Within each run, all methods receive the same workflow instances, arrival process, and random seed.

2) Effect of the deadline factor: We first evaluate the effect of the deadline factor $\alpha ^ { d f }$ with K = 100 workflows, Poisson arrival rate $\lambda = 0 . 5$ , and execution-speed variation coefficient $v = 0 . 1$ . The results are shown in Fig. 4.

As shown in Fig. 4(a), relaxing the deadline generally enhances workflow scheduling success. GA-HRL and DTODRL maintain a 100% success rate across the entire tested range. HACPPO also exhibits steady improvement with increasing $\alpha ^ { d f }$ and consistently outperforms the heuristic and deterministic baselines, though it falls short of the two top-performing learning-based methods. In contrast, OHDS, SMWDSA, and especially DS-CSP yield lower success rates, as their policies are less responsive to the combined effects of dynamic arrivals, precedence constraints, and execution-speed variations.

The utilization and energy metrics in Fig. 4(b)–(c) offer a clearer distinction among the learning-based approaches. GA-HRL achieves the highest and most stable container utilization, ranging from 66% to 68%, at an energy cost of only $1 . 7 6 \times 1 0 ^ { 7 } .$ $1 . 8 8 \times 1 0 ^ { 7 } \mathrm { ~ J }$ . DTODRL attains the same workflow success rate but utilizes only 61%-64% of container capacity and consumes $2 . 1 4 \times 1 0 ^ { 7 } { - } 2 . 3 2 \times 1 0 ^ { 7 } \ \mathrm { J }$ . HACPPO shows utilization close to that of GA-HRL yet incurs consistently higher energy consumption, indicating that similar container occupancy does not necessarily imply equivalent placement efficiency. Among the major baselines, SMWDSA remains the least resourceefficient, with utilization below 50% and energy consumption around $3 . 0 9 \times 1 0 ^ { 7 }$ J. Overall, these results suggest that GA-HRL can satisfy both loose and stringent deadlines without resorting to indiscriminate resource over-provisioning.

3) The influence of execution-speed variation: We next evaluate robustness to execution-speed variation by changing v from 0 to 0.45 while fixing $K = 1 0 0 , \lambda = 0 . 5 .$ , and $\alpha ^ { d f } = 2 . 1$ The results are shown in Fig. 5.

Figure 5(a) shows that higher execution-speed variation lowers success for all methods, though at markedly different rates. $\mathrm { A t } \ v \ = 0 . 4 5$ , DTODRL leads with 86%, followed by GA-HRL at 82%. HACPPO exhibits intermediate robustness, degrading more slowly than heuristic baselines yet faster than the top two. In contrast, SMWDSA and DS-CSP drop to 47% and 5%, respectively, highlighting the fragility of reactive or deterministic policies under high variability.

The resource costs (Fig. 5(b)-(c)) reveal that GA-HRL maintains utilization at 63%-67% and energy at $2 . 3 2 \times 1 0 ^ { 7 }$ J at $v = 0 . 4 5$ . DTODRL achieves 4% higher success but consumes $3 . 2 7 \times 1 0 ^ { 7 } \ : \div$ J, while SMWDSA costs $4 . 9 8 \times 1 0 ^ { 7 } \mathrm { ~ J }$ . HACPPO’s utilization stays stable, yet its energy exceeds GA-HRL’s with increasing volatility. Thus, at maximum variation, GA-HRL sacrifices a modest success margin (4%) for roughly 29% energy savings over DTODRL, which is consistent with its uncertainty-aware design that favors preserving deadline slack over aggressive scale-out.

4) The influence of the number of workflows: Finally, we evaluate scalability by varying the number of workflows K from 100 to 1000 with $\lambda = 0 . 5 , v = 0 . 1$ , and $\alpha ^ { d f } = 2 . 1$ . The results are shown in Fig. 6.

Fig. 6(a) shows that GA-HRL and DTODRL maintain workflow scheduling success close to 100% as the workload increases. HACPPO remains comparatively stable in the highsuccess region but below GA-HRL and DTODRL, whereas OHDS and SMWDSA level off at lower success rates and DS-CSP degrades further. These results indicate that the learningbased schedulers are better able to absorb the denser arrival stream, but their resource efficiency differs.

As shown in Fig. 6(b), the average container utilization of GA-HRL rises gradually from about 66% to 70%, indicating that the scheduler increasingly reuses existing containers as more workflows overlap. HACPPO also maintains relatively high utilization, while DTODRL remains slightly lower than GA-HRL over most of the tested range. Fig. 6(c) shows that total energy consumption increases steadily with workflow volume for all methods. GA-HRL remains the most energyefficient across the tested range and exhibits a more gradual increase than the competing schedulers. At $K = 1 0 0 0 , \mathrm { G A } \cdot$ HRL consumes $1 5 . 3 5 \times 1 0 ^ { 7 } .$ J, compared with $1 7 . 0 7 \times 1 0 ^ { 7 }$ J for DTODRL and $2 2 . 5 3 \times 1 0 ^ { 7 } \ : ]$ J for SMWDSA; HACPPO also remains above GA-HRL in total energy. The results therefore support the conclusion that GA-HRL scales to denser workflow loads mainly through container reuse and coordinated placement rather than indiscriminate scale-out provisioning.

![](images/178a72465d9c7e411d3b5bf25b04f8440c01ad18a729c78ce5cb8757282f276f.jpg)

(b)  
![](images/499aeb3ffff4f6a0f35ad459cd17373738d88ca8e29f44c951bf14f0c1e8278b.jpg)

![](images/c4d34098e6650fa93e7264b76dafa21d486a35cc618a5920def116ebd94c5941.jpg)  
Fig. 5: Effect of the execution-speed variation coefficient v with $K = 1 0 0$ workflows, $\lambda \ = \ 0 . 5 ,$ and $\alpha ^ { d f ^ { \star } } = 2 . 1 \colon$ (a) workflow scheduling success rate, (b) average container resource utilization, and (c) total system energy consumption.

## VI. CONCLUSION AND FUTURE WORK

This paper presented GA-HRL, an event-driven hierarchical reinforcement learning scheduler for dynamic cloud workflows with stochastic execution speeds and placement-dependent communication. It represents workflows as DAGs, uses predicted sub-deadlines to capture task urgency, and applies a multi-head GAT to encode dependency information. A taskscheduling agent then assigns ready tasks to containers, and a container-scheduling agent places newly requested containers, with both agents trained alternately using separate PPO actorcritic networks. Trace-driven experiments on the 2018 Alibaba cluster trace showed that GA-HRL maintains competitive workflow success while generally achieving higher container utilization and lower energy consumption, and at the largest speed variation it trades a small success-rate gap relative to DTODRL for substantially lower energy. Ablation results further confirmed that the dependency-aware task representation and learned container placement improve scheduling efficiency in complementary ways. The current model represents runtime interference through execution-speed variation and does not explicitly consider resource failures or online estimation errors; future work will evaluate GA-HRL on a physical testbed and incorporate measured interference, failures, and online performance estimation.

![](images/1d6f200c7d9feec29d22c87863666091db17408ec48cdb9e57907b228af52b19.jpg)

(b)  
![](images/9345d66c5eee0ff5dbbe83d6669f6bdd6af497b36d129a9e40937ae62c2de7df.jpg)

![](images/cb1d4fd516e9e48a0f74b47ef11194ad15ee31a35a35a6e385a9b906359b7aa2.jpg)  
Fig. 6: Effect of the number of workflows K with $\lambda \ : = \ : 0 . 5 , \ : v \ : = \ : 0 . 1 .$ and $\alpha ^ { d f } = 2 . 1 \colon$ (a) workflow scheduling success rate, (b) average container resource utilization, and (c) total system energy consumption.

[1] L. Ye, Y. Xia, L. Yang, and C. Yan, “Shws: Stochastic hybrid workflows dynamic scheduling in cloud container services,” IEEE Transactions on Automation Science and Engineering, vol. 19, no. 3, pp. 2620–2636, 2022.

[2] H. Chen, X. Zhu, G. Liu, and W. Pedrycz, “Uncertainty-aware online scheduling for real-time workflows in cloud service environment,” IEEE Transactions on Services Computing, vol. 14, no. 4, pp. 1167–1178, 2021.

[3] C. Cheng, J. Li, and Y. Wang, “An energy-saving task scheduling strategy based on vacation queuing theory in cloud computing,” Tsinghua Science and Technology, vol. 20, no. 1, pp. 28–39, 2015.

[4] Z. Liu, L. Huang, Z. Gao, M. Luo, S. Hosseinalipour, and H. Dai, “Gadrl: Graph neural network-augmented deep reinforcement learning for dag task scheduling over dynamic vehicular clouds,” IEEE Transactions on Network and Service Management, vol. 21, no. 4, pp. 4226–4242, 2024.

[5] L. M. Al Qassem, T. Stouraitis, E. Damiani, and I. M. Elfadel, “Containerized microservices: A survey of resource management frameworks,” IEEE Transactions on Network and Service Management, vol. 21, no. 4, pp. 3775–3796, 2024.

[6] G. Fan, X. Chen, Z. Li, H. Yu, and Y. Zhang, “An energy-efficient dynamic scheduling method of deadline-constrained workflows in a cloud environment,” IEEE Transactions on Network and Service Management, vol. 20, no. 3, pp. 3089–3103, 2023.

[7] H. Topcuoglu, S. Hariri, and M.-Y. Wu, “Performance-effective and low-complexity task scheduling for heterogeneous computing,” IEEE transactions on parallel and distributed systems, vol. 13, no. 3, pp. 260– 274, 2002.

[8] X. Yu, W. Wu, and Y. Wang, “Integrating cognition cost with reliability qos for dynamic workflow scheduling using reinforcement learning,” IEEE Transactions on Services Computing, vol. 16, no. 4, pp. 2713– 2726, 2023.

[9] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” nature, vol. 518, no. 7540, pp. 529–533, 2015.

[10] Z. Cao, X. Deng, S. Yue, P. Jiang, J. Ren, and J. Gui, “Dependent task offloading in edge computing using gnn and deep reinforcement learning,” IEEE Internet of Things Journal, vol. 11, no. 12, pp. 21 632– 21 646, 2024.

[11] A. Kheldoun, K. Barkaoui, and M. Ioualalen, “Formal verification of complex business processes based on high-level petri nets,” Information Sciences, vol. 385, pp. 39–54, 2017.

[12] S. Jiao, X. Zhang, S. Yu, X. Song, and Z. Xu, “Joint virtual network function selection and traffic steering in telecom networks,” in GLOBE-COM 2017-2017 IEEE Global Communications Conference. IEEE, 2017, pp. 1–7.

[13] M. Hahnel, J. Martinovic, G. Scheithauer, A. Fischer, A. Schill, and¨ W. Dargie, “Extending the cutting stock problem for consolidating services with stochastic workloads,” IEEE Transactions on Parallel and Distributed Systems, vol. 29, no. 11, pp. 2478–2488, 2018.

[14] L. Liu, H. Tan, S. H.-C. Jiang, Z. Han, X.-Y. Li, and H. Huang, “Dependent task placement and scheduling with function configuration in edge computing,” in Proceedings of the International Symposium on Quality of Service, 2019, pp. 1–10.

[15] A. Das, S. Imai, S. Patterson, and M. P. Wittie, “Performance optimization for edge-cloud serverless platforms via dynamic task placement,” in 2020 20th IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGRID). IEEE, 2020, pp. 41–50.

[16] S. Deng, H. Zhao, Z. Xiang, C. Zhang, R. Jiang, Y. Li, J. Yin, S. Dustdar, and A. Y. Zomaya, “Dependent function embedding for distributed serverless edge computing,” IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 10, pp. 2346–2357, 2021.

[17] M. A. Rodriguez and R. Buyya, “Deadline based resource provisioningand scheduling algorithm for scientific workflows on clouds,” IEEE transactions on cloud computing, vol. 2, no. 2, pp. 222–235, 2014.

[18] V. Arabnejad, K. Bubendorfer, and B. Ng, “Scheduling deadline constrained scientific workflows on dynamically provisioned cloud resources,” Future Generation Computer Systems, vol. 75, pp. 348–364, 2017.

[19] Y. Yang, H. Shen, and H. Tian, “Scheduling workflow tasks with unknown task execution time by combining machine-learning and greedyoptimization,” IEEE Transactions on Services Computing, vol. 17, no. 3, pp. 1181–1195, 2024.

[20] F. Ding, Y. Yuan, L. Lv, R. Zhang, and W. Zhou, “Transformer-enhanced

dqn approach for energy and cost-efficient large-scale dynamic workflow scheduling in heterogeneous environment,” IEEE Internet of Things Journal, vol. 11, no. 22, pp. 37 351–37 367, 2024.

[21] Y. Xie, L. Huang, Y. Kong, S. Wang, S. Xu, X. Wang, and J. Ren, “Virtualized network function forwarding graph placing in sdn and nfvenabled iot networks: A graph neural network assisted deep reinforcement learning method,” IEEE Transactions on Network and Service Management, vol. 19, no. 1, pp. 524–537, 2022.

[22] C. Jin, X. Bai, C. Yang, W. Mao, and X. Xu, “A review of power consumption models of servers in data centers,” applied energy, vol. 265, p. 114806, 2020.

[23] Q. Wu, F. Ishikawa, Q. Zhu, Y. Xia, and J. Wen, “Deadline-constrained cost optimization approaches for workflow scheduling in clouds,” IEEE Transactions on Parallel and Distributed Systems, vol. 28, no. 12, pp. 3401–3412, 2017.

[24] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Ben-´ gio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.

[25] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[26] Website. Alibaba Inc. (2018), [Online]. Alibaba Production Cluster Data v2018. Available: https://github.com/alibaba/clusterdata/tree/v2018.

[27] Z. Sun, Y. Mei, F. Zhang, H. Huang, C. Gu, and M. Zhang, “Multitree genetic programming hyper-heuristic for dynamic flexible workflow scheduling in multi-clouds,” IEEE Transactions on Services Computing, vol. 17, no. 5, pp. 2687–2703, 2024.

[28] A. Jayanetti, S. Halgamuge, and R. Buyya, “Deep reinforcement learning for energy and time optimized scheduling of precedence-constrained tasks in edge–cloud computing environments,” Future Generation Computer Systems, vol. 137, pp. 14–30, 2022.