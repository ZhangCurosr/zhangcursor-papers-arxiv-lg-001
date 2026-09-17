# COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads

Yukai Zhou

Hongfan Wu<sup>∗</sup>

## Abstract

With the rapid advancement of deep learning technology, shared GPU clusters receive an increasing number of deep learning training (DLT) jobs. Yet resource fragmentation make such clusters underutilized and forces the DLT jobs running on them to endure long turnaround times. Extensive research has been devoted to quantifying fragmentation and developing scheduling algorithms that alleviate its impact. However, existing fragmentation measures break down in the absence of workload distribution information, while current schedulers cannot continuously maintain resource fragmentation at a low level. To tackle these problems, we first introduce Scheduler-Induced Fragmentation (SIF), a metric built on the notion of partial-nodes that is independent of historical workload knowledge. We then propose COMPASS-ABS, which employs the COMPact-ASSured (COMPASS) algorithm to confine the cluster state within a tight Anchor-Based Space (ABS), whose construction fully leverages the topological alignment between dominant workload size and node capacity. Moreover. We also prove that it ensures SIF is bounded by $\textstyle { \frac { 2 } { N } }$ under a workload composition condition that matches both theory and production. Evaluations implemented on a physical cluster and a simulated cluster demonstrate COMPASS-ABS efectiveness at improving resource utilization, reducing DLT job completion time by reducing fragmentation.

CCS Concepts: • Computer systems organization → Cloudcomputing; • Theory ofcomputation → Scheduling algorithms.

Keywords: Deep learning training, shared GPU cluster, fragmentation, resource scheduling

## 1 Introduction

Deep learning has developed rapidly in recent years, driven by larger model frameworks, more complex training pipelines, and the rise of large language models[23, 34]. Training these models requires substantial GPU resources, so industrial labs and cloud providers operate large shared GPU clusters where users submit deep learning training (DLT) jobs[15, 22, 36, 44]. Yet production clusters exhibit both low GPU cluster utilization and long turnaround time for DLT jobs, which stems from the resource fragmentation rather the capacity shortage[24, 45].

Resource fragmentation arises from DLT job placement constraints. A DLT job consists of multiple homogeneous instances, and the GPUs assigned to each instance need to lie within a single node to preserve fast intra-instance communication[4, 10]. This intra-node constraint can prevent admission even when the cluster has enough available GPUs in total. Consider a cluster with three nodes that currently have 3, 5, and 2 free GPUs, i.e. 10 free GPUs in total. A job with two 4-GPU instances demanding 8 GPUs in total cannot be admitted because only one node can host a 4-GPU instance. The fragmented distribution of idle GPUs blocks a placement that would appear feasible under an aggregatecapacity view. So resource fragmentation leads to low GPU utilization despite pending jobs awaiting processing, causing prolonged job completion time due to long queueing latency.

This observation raises two central questions: (1) how to quantify the resource fragmentation? (2) how to solve the problem of GPU cluster underutilization and long completion time of DLT jobs caused by resource fragmentation? Existing fragmentation metrics are statistical measures of fragmentation. FGD[45] defines a workload-distribution-aware fragmentation score, while CAFGD[24] refines this idea by weighting long-term and short-term distributions. These metrics rely on workload history knowledge and fail to mea sure fragmentation in its absence.

Existing schedulers for scheduling DLT jobs in shared GPU clusters can be grouped into two categories. Nonmigration methods place each job at admission and never revisit the decision: ElasticFlow[9] formulates placement as a bin-packing problem and applies best-fit-style heuristics, while FGD[45] and CAFGD[24] greedily optimize their fragmentation metrics at first admission time. These methods react only to the current state and provide no mechanism to repair fragmentation that has existed. Migration-based approaches reorganize running DLT jobs through checkpoint-based migration. Gandiva[50] only suggests migrating running jobs when cluster status changes, without specifying the implementation details like when and how to migrate. FFT[30] and DRR[49] adopt a round-driven strategy whose defragmentation relies on global migration at the beginning of each round, which cannot mitigate the resource fragmentation accumulated within each round. More fundamentally, they all cannot guarantee low fragmentation throughout the online scheduling process.

Motivated by these limitations, we introduce Scheduler-Induced Fragmentation (SIF) to measure dynamic resource fragmentation and design the COMPASS-ABS scheduler for DLT jobs in shared GPU clusters. Our contributions are summarized as follows:

• Scheduler-Induced Fragmentation (SIF). Building on the notion of partial-nodes, we decompose observed fragmentation into two parts: an inherent component forced by the intra-node placement constraint, and an additional component caused by the scheduler’s placement decisions. SIF measures only the latter, which no longer requires the workload history distribution information.

• COMPASS-ABS scheduler. We firstly define ABS (Anchor-Based Space), a constrained feasible cluster state domain whose structure is tailored jointly to the dominance of power-of-two GPU demands and to their topology alignment with node capacity. Multi-GPU instances are placed as anchors in slots of width 2, 4, or 8 GPUs, while 1-GPU instances are managed through a pool and used as fillers for saturating slot vacancies. The COMPASS(COMPact-ASSured) algorithm maintains cluster change within this space in a compact manner through three operators: Place, Remove, and Compact. Under the Workload Composition Condition (WCC) stated in Theorem 1, COMPASS-ABS guarantees $\begin{array} { r } { \mathrm { S I F } ^ { \pi } ( t ) \le \frac { 2 } { N } } \end{array}$ at all times.

• Evaluation. We conduct comprehensive experiments to validate the scheduling performance of COMPASS-ABS, including large-scale simulations on two production traces and a physical-cluster deployment with a synthetic DLT workload. COMPASS-ABS attains the highest GPU utilization and the shortest job completion time among all state-of-the-art baselines by maintaining the lowest fragmentation level, and sustains its strength under varying cluster load, workload composition, and migration cost, even when the Workload Composition Condition is violated.

## 2 Background

This section reviews the task structure and communication patterns of DLT jobs, together with the architectural properties of physical GPU clusters. We point out two system characteristics that motivate the scheduler design in Section 4: the per-instance GPU demand �<sub>�</sub> is concentrated on power-of-two values, and the basic communication domain in production clusters is a node of 8 GPUs.

## 2.1 Deep Learning Training Jobs

As deep learning models continue to extend, the memory capacity and compute throughput of a single GPU are insufficient for an increasing fraction of training workloads. Modern frameworks therefore support training across multiple GPUs[17, 28, 34]. The growth of training datasets drives the use of data parallelism, where the training program is replicated across multiple instances processing diferent minibatches. Thus, a DLT job is composed of multiple homogeneous instances that run the same model and carry the same per-instance GPU demand �<sub>�</sub>[4].

Within each instance, multiple GPUs may collaborate through tensor parallelism and other mechanisms to execute the forward and backward passes[25, 32, 34]. These mechanisms frequently invoke collective communication primitives such as All-Reduce, making the instance sensitive to communication latency[19]. Tensor parallelism partitions large parameter matrices across GPUs, and common tensor-parallel degrees are 2, 4, and 8. Collective implementations such as ring, recursive-doubling, and tree-based All-Reduce are structurally defined on power-of-two participant counts, while non-power-of-two configurations lead to padding or special-case handling[41]. These properties make $g _ { i } \in \{ 1 , 2 , 4 , 8 \}$ a natural operating regime, which is consistent with production trace characteristics[15, 22, 44].

Beyond intra-instance communication, diferent instances of the same job execute data-parallel synchronization at the end of each iteration to apply a synchronous parameter update, which requires all instances to run concurrently. Thus, schedulers for DLT workloads commonly treat each job as an all-or-nothing allocation unit, which imposes a gang scheduling requirement[10, 51].

## 2.2 Physical Cluster

Modern GPU clusters comprise multiple nodes, each ofwhich serves as a primary high-bandwidth communication domain in the network topology[23, 36]. GPUs on the same node communicate through NVLink at terabyte-per-second bandwidth, whereas inter-node communication involves lower bandwidth and switch-level congestion. For the latencysensitive intra-instance communication described above, all GPUs assigned to an instance must belong to the same node, which makes the available GPUs non-interchangeable[45, 50]. This node-locality constraint is the structural source of the fragmentation problem studied in this paper.

Each node contains eight GPUs, matching the configuration reported in production clusters[15, 22, 35, 44]. The eight-GPU design has a hardware basis: DGX-1 connects eight GPUs as a hybrid cube-mesh, and later DGX/HGX systems preserve eight GPUs as a server-level building block while improving the internal fabric with NVSwitch[26, 35, 36]. This eight-GPU node is an industry-wide convention rather than an NVIDIA-specific choice, as the vendor-neutral OCP OAM Universal Baseboard standard[37] hosts eight accelerators per baseboard and mainstream accelerator servers adopt the same � = 8 layout, including AMD Instinct[1], Intel Gaudi[20], Huawei Ascend[18], and Meta’s Grand Teton[29].

## 3 Scheduler-Induced Fragmentation

This section gives the measure of resource fragmentation without relying on history trace information. In §3.1, we introduce the notion of partial-nodes, which are the principal carrier of fragmented resource, and separate them into two mutually exclusive proportions. Building on this distinction, §3.2 then defines a new measure called Scheduler-Induced Fragmentation (SIF), computed solely from the resource demand of currently running jobs in the cluster.

## 3.1 Key Insight: Avoidable vs. Forced Partial Nodes

We firstly define a node that is non-empty yet assigned fewer GPUs than its full capacity as a partial node. A fragmented placement disperses free GPUs across many partial nodes and causes contiguous GPU resources to become scarce, whereas a well-designed placement will reduce the partialnode count to release the fragmented GPUs trapped inside partial nodes and return them to larger contiguous GPU blocks, which is friendly to all DLT jobs.

This concern becomes most pressing when the cluster receives large jobs. Consider a job consisting of � 8-GPU instances which require � fully-empty nodes on the cluster before it can be admitted. Consider a cluster of � nodes in which the current fragmented placement consumes $N _ { 1 }$ nodes, whereas a compact placement of the same set of running DLT jobs would consume only $N _ { 2 } < N _ { 1 }$ . The compact placement therefore leaves $N - N _ { 2 }$ nodes entirely empty, while the fragmented one leaves only $N - N _ { 1 }$ . Whenever the requested job size satisfies $N - N _ { 1 } < w \leq N - N _ { 2 } .$ , that job is admissible under the compact placement but is blocked under the fragmented one, even though the cluster’s aggregate free-GPU count exceeds 8� in both situations. This large DLT job blocking results from fully-empty-node availability: $N _ { 1 } - N _ { 2 }$ nodes trapped in partial occupancy could otherwise serve as the continuous GPU blocks that large jobs demand.

We therefore set our primary objective for mitigating fragmentation as minimizing the number of partial nodes, which can equivalently be stated as minimizing the total number of nodes in use. We adopt the former formulation throughout the remainder of this paper. However, not every partial node is produced by the scheduler: some cannot be eliminated by any scheduler, while others arise from a suboptimal scheduling decision and can be eliminated by right placement and migration. Then we demonstrate the distinction through two illustrative examples.

Forced partial nodes. Suppose the cluster currently has only one partial node, which hosts a single 4-GPU instance, a DLT job consisting of two 5-GPU instances waits at the head of queue. Due to intra-node constraint, neither of the two new instances can utilize the 4 free GPUs on this partial node. The only feasible placement is opening two other empty nodes, each hosting a single 5-GPU instance with 3 GPUs unused, so that the cluster now contains three partial nodes in total. No scheduler could have done better in this situation, because these three partial nodes are byproducts of systemic constraints rather than incorrect scheduling.

Avoidable partial nodes. Consider two nodes that are each fully packed without free GPUs, where a single job has placed one 4-GPU instance on each node and the remaining 4 GPUs of each node are occupied by unrelated instances. When this job completes, both nodes simultaneously release a 4-GPU block and the cluster gains two partial nodes in a single event. This fragmentation is not structurally necessary, because the scheduler could migrate all instances on the second node into the idle region of the first, after which the cluster would contain zero partial nodes. The two partial nodes here are avoidable, because their existence is entirely attributable to the scheduler’s decision.

## 3.2 Decomposing Fragmentation: Inherent and Scheduler-Induced

Motivated by §3.1, we now formally split the fragmentation into two complementary components, the portion that any scheduler must leave behind, and the portion introduced by chosen scheduler �.

Inherent Fragmentation. We denote the set of DLT jobs running on the cluster at time � by $\mathcal { T } ( t )$ . Each job $i \in \mathcal { T } ( t )$ consists of $w _ { i }$ homogeneous instances, each demanding �<sub>�</sub> ${ \mathrm { G P U s } } ,$ where $g _ { i } \in \{ 1 , . . . , G \}$ and $G = 8 .$ . The instantaneous demand structure of running DLT jobs is then formalized as $\mathcal { I } ( t ) = \{ ( g _ { i } , w _ { i } ) : i \in \mathcal { T } ( t ) \}$ }. In order to isolate the fragmentation originating from constraints alone, we construct a static bin-packing problem in which each DLT job is treated as �<sub>�</sub> items of size $g _ { i }$ and each node as a bin of capacity �. The question then becomes: what is the minimum partial-node count achievable over all feasible packings of these items into bins? Formally, for any feasible packing $P$ that satisfies the intra-node constraint, we let $\nu ( P )$ denote the number of partial nodes in $P$ and write $\Phi ^ { \star } ( t ) : = \mathrm { m i n } _ { P } \mathrm { f e a s i b l e } \nu ( P )$ for the minimum partial-node count achievable across all such packings. The Inherent fragmentation is then the normalised share

$$
F ^ { \star } ( t ) : = \frac { \Phi ^ { \star } ( t ) } { N } = \frac { \operatorname* { m i n } _ { P \mathrm { f e a s i b l e } } \nu ( P ) } { N } .\tag{1}
$$

Because $\mathbf { \cdots } ( t )$ depends only on the real time resource demand $\boldsymbol { \mathcal { T } } ( t )$ and is independent of arrival order, placement history, or migration operation, it serves as an unavoidable lower bound on the partial-node share. A heuristic algorithm for computing $\Phi ^ { \star } ( t )$ (and therefore $F ^ { \star } ( t ) )$ is provided in Appendix C.

Scheduler-Induced Fragmentation. Under an online policy �, the placement of all jobs in the cluster observed at time � is itself a feasible packing $P ^ { \pi } ( t )$ of $\boldsymbol { \mathcal { T } } ( t )$ , and we let $F ^ { \pi } ( t ) : = \nu ( P ^ { \pi } ( t ) ) / N$ denote its partial-node share, which represents the gross fragmentation. We define the Scheduler-Induced Fragmentation (SIF) by subtracting the inherent fragmentation from the gross fragmentation:

$$
\operatorname { S I F } ^ { \pi } ( t ) ~ : = ~ F ^ { \pi } ( t ) - F ^ { \star } ( t ) ~ \ge ~ 0 ,\tag{2}
$$

which represents the percentage of avoidable partial nodes that the policy � has produced. This definition makes SIF a more reasonable fragmentation measure than the statistical metrics reported in FGD and CAFGD. It builds upon the resource demand of the running jobs rather than further historical information about the workload resource demand distribution. Moreover, it accurately quantifies the proportion of fragmentation that is introduced by the scheduler. Therefore, it motivates us to design a scheduler that persistently keeps $\mathrm { S I F } ^ { \pi } ( t )$ at a low level.

## 4 COMPASS-ABS Scheduler

This section presents our COMPASS-ABS scheduler that employs the COMPASS algorithm to maintain the cluster evolving in the compact ABS domain. In Section 4.1, we introduce our design motivation for the innovative scheduler; In Section 4.2, we define a feasible space ABS for the cluster to operate in, driven by the characteristic of workload composition and cluster topology; In Section 4.3, we present the COMPASS algorithm that constrains the cluster state within the ABS in a consolidated manner. In Section 4.4, we further provide theoretical guarantee on the ability of our scheduler to keep SIF low at minimal computational cost.

## 4.1 Design Rationale

Observation 1: Power-of-two-sized multi-GPU instances scheduled by the best-fit policy when node capacity is 8 keep $\begin{array} { r } { \mathrm { S I F } \le \frac { 2 } { N } } \end{array}$ . Before examining fragmentation in detail, we set aside the instances whose $g _ { i } ~ = ~ 1$ , denoted as ${ \cal T } ^ { = 1 }$ , because they never create fragmentation but rather act as the natural filler units to mitigate fragmentation by utilizing the scattered idle GPUs left on partial nodes. Re stricting attention to the multi-GPU instances, denoted as ${ \cal T } ^ { \geq 2 }$ , we exploit a divisibility alignment between the instance demand and the node capacity: every $g _ { i } \in \{ 2 , 4 , 8 \}$ divides $G = 8$ exactly, so any combination of such instances either fully fills a node or leaves a residue GPU block of size in {2, 4, 6}. Under this alignment, a best-fit policy that places each instance on the node leaving the smallest remaining capacity after placement, together with a lightweight Compact pass that consolidates partial nodes after every departure, retains $\begin{array} { r } { \mathrm { S I F } ^ { \pi } ( t ) \le \frac { 2 } { N } } \end{array}$ at all times. The formal proof, including the matching tightness construction, is deferred to $\mathsf { A p - }$ pendix A. We accordingly name the multi-GPU instances with $g _ { i } \in \{ 2 , 4 , 8 \}$ as Fragmentation-Friendly instances and denote them by $\boldsymbol { \mathcal { T } } ^ { \mathrm { F F } }$ , whereas instances whose demand falls in $\{ 3 , 5 , 6 , 7 \}$ are called Fragmentation-Unfriendly instances, denoted by $\boldsymbol { \mathcal { T } ^ { \mathrm { F U F } } }$ , so all multi-GPU instances can be partitioned into $\scriptstyle { \bar { Z ^ { \ge 2 } } } \ : = \ { \bar { Z } } ^ { \mathrm { F F } } \ \cup \ { \bar { Z } } ^ { \mathrm { F U F } }$

Observation 2: Fragmentation-unfriendly instances account for only a small share of real DLT workloads.

The structural reasons in §2.1 make $g _ { i } \in \{ 1 , 2 , 4 , 8 \}$ the natural operating regime for DLT instances, so non-power-of-two instance sizes $g _ { i } \in \{ 3 , 5 , 6 , 7 \}$ account for only a small fraction of the instance population in production traces. Table 1 confirms this claim by reporting the average per-type instance shares observed on the GFS[7] and Venus[15] traces, where the $g _ { i } = 1$ and $g _ { i } = 8$ classes together dominate the trace and the ${ \cal T } ^ { \mathrm { F U F } }$ retains only a marginal share.

These observations lead to two design ideas.

Design idea 1: Round every fragmentation-unfriendly instance into a fragmentation-friendly composite block with fillers. Observation 1 already points out that a best-fit placement keeps $\begin{array} { r } { \mathrm { S I F } \le \frac { 2 } { N } } \end{array}$ once the workload is restricted to power-of-two-sized instances, and we therefore want to extend the same guarantee to the fragmentation-unfriendly population. To this end, we round instance in $\boldsymbol { \mathcal { T } ^ { \mathrm { F U F } } }$ into a friendly-sized composite block whose total size falls in {2, 4, 8} by combining it with smaller instances. Then the scheduler only sees fragmentation-friendly-shaped blocks rather than a heterogeneous mixture of friendly and unfriendly demands under this transformation, so that the placement decisions on the entire multi-GPU population $\scriptstyle { \bar { I } } ^ { \geq 2 }$ reduce to the $\boldsymbol { \mathcal { T } } ^ { \mathrm { F F } }$ -only case.

Design idea 2: Employ 1-GPU instances to assemble the fragmentation-friendly composite blocks and manage them in a dedicated pool. As 1-GPU instances largely outweigh those in ${ \cal T } ^ { \mathrm { F U F } }$ and they are the natural filler units to mitigate fragmentation, we accordingly use 1-GPU instances to fill the residual GPUs inside each composite block, namely the gap between the original unfriendly instance and its rounded friendly block. Due to the dynamic arrival and departure of $\boldsymbol { \mathcal { T } ^ { \mathrm { F U F } } }$ instances, the requests for fillers change over time, and the cluster needs to adaptively migrate 1- GPU instances to the right place to maintain the integrity of the composite block. This motivates us to place all 1-GPU instances in a unified pool for better management.

Table 1. Per-class instance share across two DLT traces.
<table><tr><td>Trace</td><td> ${ \boldsymbol { \mathcal { T } } } ^ { = 1 }$ </td><td> $\boldsymbol { \mathcal { T } } ^ { \mathrm { F F } }$ </td><td> ${ \cal T } ^ { \mathrm { F U F } }$ </td><td> $\scriptstyle { \mathcal { T } } ^ { = 8 }$ </td></tr><tr><td>GFS</td><td>64.16%</td><td>6.07%</td><td>0.05%</td><td>29.73%</td></tr><tr><td>Venus</td><td>37.68%</td><td>9.44%</td><td>0.44%</td><td>52.44%</td></tr></table>

## 4.2 Anchor-Based Space (ABS)

Driven by the design ideas above, we define a structured state space for the cluster that we call the Anchor-Based Space (ABS), within which the placement of all DLT jobs is required to remain throughout the cluster’s evolution. Then we formalize the key objects in the ABS domain.

Definition 1 (Slot). A slot, denoted by �, is a GPU block inside a node. Each slot has a fixed width �(�) ∈ {2, 4, 8}.

When a multi-GPU instance of demand � is assigned to a slot, the slot width is chosen according to

$$
w ^ { \star } ( g ) \ = \ \left\{ \begin{array} { l l } { 2 , } & { g = 2 , } \\ { 4 , } & { g \in \{ 3 , 4 \} , } \\ { 8 , } & { g \in \{ 5 , 6 , 7 , 8 \} . } \end{array} \right.\tag{3}
$$

The instances belonging to $\boldsymbol { \mathcal { T } } ^ { \mathrm { F F } }$ will directly saturate widthmatched slots, while counterpart instances will be assigned to rounded-up slots.

Definition 2 (Anchor and Slot Vacancy). On each slot $s ,$ there is exactly one instance $a ( s ) \in \mathcal { I } ^ { \geq 2 }$ called the anchor of � that occupies $g _ { a ( s ) }$ GPUs in that slot, where $g _ { a ( s ) }$ is the demand of that instance. The remaining ${ \boldsymbol w } ( s ) - g _ { a ( s ) }$ GPUs in the slot are called the slot vacancy of�. Whenever $a ( s ) \neq \emptyset$ the slot width satisfies $w ( s ) = w ^ { \star } \big ( g _ { a ( s ) } \big )$

Definition 3 (Filler and Capacity Invariant). When the slot vacancy of $s \neq 0$ , we only employ 1-GPU instances to saturate the slot vacancy, which are called fillers. Representing the set of fillers currently attached to � as $F ( s )$ , we require

$$
g _ { a ( s ) } + | F ( s ) | \ \leq \ w ( s ) .\tag{4}
$$

Equality means the slot is fully used.

Definition 4 (Slotted node and slot configuration set). Node � that hosts at least one slot is called a slotted node, and we denote the multiset of slots currently profiled on � by $S ( n )$ where the slots in $S ( n )$ are non-overlapping. The residual $\begin{array} { r } { G - \sum _ { s \in S ( n ) } w \big ( s \big ) } \end{array}$ GPUs on � have not yet been profiled as any slot, and a new slot is created over them only when an anchor is placed on �. The collection of all feasible slot profiling plans forms the slot configuration set

$$
{ \mathrm { S l o t C o n f ~ } } : = { \Big \{ } \left\{ s _ { 1 } , \ldots , s _ { k } \right\}  { \Big | } s _ { i } \in \{ 2 , 4 , 8 \} \forall i , \sum _ { i = 1 } ^ { k } s _ { i } \leq G { \Big \} } ,\tag{5}
$$

where $S ( n ) \in { \mathsf { S l o t C o n f } } , \forall n$

Definition 5 (Pool node). A pool node is a node dedicated to 1-GPU instances. For a pool node �, we write $P ( n ) \subseteq { \cal T } ^ { = 1 }$ for the set of 1-GPU instances currently on �, with $| P ( n ) | \leq$ �. A 1-GPU instance on a pool node is not acting as a filler but can be moved to the slot vacancy when necessary.

Building on the definitions above, every node in the cluster falls into three mutually exclusive types: an empty node holds no instances, a slotted node whose $S ( n ) \in { \mathrm { S l o t C o n f } } _ { : }$ and a pool node whose $P ( n ) \subseteq { \cal T } ^ { = 1 }$ . We accordingly collect these three labels into the node-type set

$$
{ \sf T } : = \{ { \sf e m p t y } , { \sf p o o l } , { \sf s l o t t e d } \} ,\tag{6}
$$

and write $t ( n ) \in \mathbb { T }$ for the type currently assigned to node �. This coarse classification underlies the formal definition of the feasible cluster state space Σ given below.

Anchor-Based Space (ABS) Σ. For each node �, the internal state $\sigma _ { n }$ is determined by �(�):

$$
\sigma _ { n } \ = \ \left\{ \begin{array} { l l } { \bot , } & { t ( n ) = \mathsf { e m p t } \gamma , } \\ { P ( n ) \subseteq T ^ { = 1 } , \ | P ( n ) | \leq G , } & { t ( n ) = \mathsf { p o o l } , } \\ { \big ( ( a _ { j } , F _ { j } ) \big ) _ { j = 1 } ^ { k } , } & { t ( n ) = \mathsf { s l o t t e d } . } \end{array} \right.\tag{7}
$$

where $a _ { j } \in \mathcal { I } ^ { \geq 2 } , F _ { j } \subseteq \mathcal { I } ^ { = 1 }$ and together satisfy the capacity invariant in (4). The cluster-level state space is therefore

$$
\Sigma ~ = ~ \left\{ ~ \sigma = \left( \pi , ( \sigma _ { n } ) _ { n \in N } \right) ~ \bigg | ~ \pi : N \to \mathrm { T } , ~ \sigma _ { n } ~ \mathrm { a s ~ a b o v e } ~ \right\} .\tag{8}
$$

This state-space design is organized entirely around the notion of anchor, which provides a unified scheduling unit for both fragmentation-friendly and fragmentation-unfriendly multi-GPU instances. Starting from this concept, the gap between a slot’s width and its anchor’s actual demand naturally introduces fillers to absorb the residual capacity inside each slot, and the practical need to migrate these fillers flexibly across the cluster in turn motivates the design of pool nodes as the dedicated pool for 1-GPU instances. Therefore, we name this state space Σ as Anchor-Based Space (ABS).

## 4.3 COMPASS Algorithm

We propose the COMPact-ASSured (COMPASS) algorithm that consists of three operators, the Place and Remove operators together guarantee that the cluster state lies within ABS Σ as the job is admitted and completed, while the Compact ensures the compactness of ABS Σ.

Each DLT job is admitted only when all of its instances can be placed simultaneously and is released as a whole upon completion. We therefore formulate scheduling decisions at the instance granularity, where every job-level scheduling is translated into a sequence of per-instance operations whose union realises the corresponding job-level decision, and for an instance � we write � for its GPU demand.

4.3.1 Common Notation. We write $\sigma \in \Sigma$ for the current cluster state, D for the set of legal destinations where a single instance can be assigned, and $M ^ { \star }$ for the space of finite migration lists. A destination $d \in \mathcal { D }$ takes one of four forms: a slotted node where a new slot of the required width can be profiled, a slot vacancy of an anchored slot, a partial pool node, or an empty node waiting to be opened. A single migration is written as a pair (�, �) that moves instance � to destination �, and the update notation � ⊕ � means that the migrations in $M \in M ^ { \star }$ are applied to � in sequence.

4.3.2 Place Operator. The placement operator carries the signature Place : $\Sigma \times J \to { \mathcal { D } } \times M ^ { \star }$ , which takes the current cluster state � together with an arriving instance � and returns the destination � where � is to be installed alongside the migration list �.

$\mathrm { I f } i \in { \cal I } ^ { \geq 2 }$ , the scheduler searches for a slotted node with at least $w ^ { \star } ( g _ { i } ) \in \{ 2 , 4 , 8 \}$ free GPUs, or opens a fresh empty node if none exists; it then profiles a $\mathrm { w i d t h } { - } w ^ { \star } ( g _ { i } )$ slot and places � as the anchor. $\mathrm { W h e n } i \in { \cal { I } } ^ { \mathrm { F U F } }$ , the scheduler migrates up to $w ^ { \star } ( g _ { i } ) - g _ { i }$ 1-GPU instances from the pool nodes as fillers, subject to filler availability in the pool.

$\mathrm { I f } i \in \mathcal { I } ^ { = 1 }$ , the scheduler places � following a strict priority order: into an existing slot vacancy if one exists; otherwise into a partial pool node; and only resorts to a newly opened pool node when neither exists. This order lets the 1-GPU instance firstly mitigate existing fragmentation before activating any new slotted node.

Pseudocode of Place(�, �).   
Input: state $\sigma ;$ arriving instance � with demand $g _ { i }$   
Output: destination $d \in { \mathcal { D } } ;$ migration list � $\in \mathcal { M } ^ { \star }$   
1: � ← ∅   
2: if $g _ { i } \geq 2$ then   
3: $w  w ^ { \star } ( g _ { i } )$   
4: $\begin{array} { r } { \mathcal { N } _ { \mathrm { c a n d } }  \{ n : t ( n ) = \mathsf { s l o t t e d } , G - \sum _ { s \in S ( n ) } w ( s ) \geq w \} } \end{array}$   
5: if $N _ { \mathrm { c a n d } } \neq 0$ then $\begin{array} { r } { n ^ { \star }  \arg \operatorname* { m i n } _ { n \in N _ { \mathrm { c a n d } } } \big ( G - \sum _ { s \in S ( n ) } w ( s ) \big ) } \end{array}$   
6: else pick any empty node $n ^ { \star }$ and set $t ( n ^ { \star } )$ ← sloted   
7: profile a fresh slo $\cdot s ^ { \star }$ of width � on $n ^ { \star } ;$ set $a ( s ^ { \star } ) \gets i ; d \gets s ^ { \star }$   
8: $r \gets w - g _ { i }$   
9: $\mathbf { i f } \ r > 0$ then draw up to � 1-GPU instances $\{ f _ { 1 } , \ldots , f _ { k } \}$ from   
pool nodes; $M \gets \{ ( f _ { \ell } , s ^ { \star } ) : \ell = 1 , \dots , k \}$   
10: else $( g _ { i } = 1 )$   
11: if ∃ � with $g _ { a ( s ) } + | F ( s ) | < w ( s )$ then � $ s$   
12: else if ∃ � with $t ( n ) =$ pool and $| P ( n ) | < G$ then $d \gets n$   
13: else pick any empty node $n _ { 0 } ;$ set �(�<sub>0</sub>) ← pool; � ← �<sub>0</sub>   
14: return (�, �)

4.3.3 Remove Operator. The departure operator carries the signature Remove $: \Sigma \times \bar { J } \to { M } ^ { \star }$ , which returns the migrations triggered when instance � departs from state �.

$\bar { \mathrm { ~ I f ~ } } i \in \mathcal { I } ^ { \geq 2 }$ , it serves as the anchor �(�) of slot �. The operator dissolves � by releasing �(�) GPUs, and migrates each filler in $F ( s )$ into the pool node through the same 1- GPU rule that Place applies to it; the migration $M = \emptyset$ when $a ( s ) \in { \cal Z } ^ { \mathrm { F F } }$ , since $F ( s ) = \varnothing$ in that case.

$\mathrm { I f } i \in \mathcal { I } ^ { = 1 }$ , the operator releases � from its current location, which is either on a slotted node or a pool node. In the former case, it additionally migrates another 1-GPU instance from the pool to keep the slot saturated.

Pseudocode of Remove(�, �).   
Input: state �; departing instance �   
Output: migration list $M \in { \cal M } ^ { \star }$   
1: � ← ∅   
2: if $g _ { i } \geq 2$ then (� is an anchor)   
3: � ← the slot whose anchor is �   
4: for each $f \in F ( s )$ do   
5: choose destination $d _ { f }$ for � by the 1-GPU rule of Place   
6: append $( f , d _ { f } )$ to �   
7: end for   
8: dissolve � and release its $w ( s )$ GPUs on the host node   
9: else $( g _ { i } = 1 )$   
10: detach � from its current slot vacancy or pool node   
11: return �

4.3.4 Compact Operator. We denote the compaction operator by Compact : $\Sigma  M ^ { \star }$ , which runs after each job event, namely the arrival or departure of each DLT job. It performs consolidation via three sequential calls to a per-class subroutine BestFitDrain

Within each invocation, BestFitDrain identifies the partial hosts currently holding class-C items, orders them by descending room so that the sparsest source is processed first, and migrates the items of that source one by one into the partial host whose remaining class-C room is the smallest among those still able to accept the migrated item. For the three concrete classes used by Compact, the pool pass takes 1-GPU instances as items and pool nodes as hosts with room $G - \left| P ( n ) \right|$ , while the width-4 and width-2 passes take anchored slots of the corresponding width as items and slotted nodes as hosts with room $\lfloor D ( n ) / 4 \rfloor$ and $\lfloor D ( n ) / 2 \rfloor$ respectively, where $\begin{array} { r } { D ( n ) = G - \sum _ { s \in S ( n ) } w ( s ) } \end{array}$

```latex
Pseudocode of Compact(�).
Input: state �
Output: migration list $M \in { \cal M } ^ { \star }$
$1 \colon M \gets \emptyset$
2: for each class $C \in$ ⟨pool, w4, w2⟩ do
3: $N \gets \{ n : 0 <$ count<sub>C</sub> (�) < cap (�) }
4: sort N by room $\mathfrak { i } _ { C } ( \cdot )$ in descending order
5: while $| N | \ge 2$ do
6: $n _ { \mathrm { s r c } } \gets \mathrm { h e a d } ( N ) ;$ progress ← false
7: for each class-C item � on $n _ { \mathrm { s r c } }$ do
8: $S \gets \{ n \in N \setminus \{ n _ { \mathrm { s r c } } \} : \mathrm { r o o m } _ { C } ( n ) \geq \mathrm { s i z e } ( x ) \}$
9: if $s = \emptyset$ then break
10: $n _ { \mathrm { d s t } }  \arg \operatorname* { m i n } _ { n \in S }$ room<sub>C</sub> (�)
11: migrate � from $n _ { \mathrm { s r c } }$ to $n _ { \mathrm { d s t } } ;$ append to �; progress ← true
12: end for
13: if $n _ { \mathrm { { s r c } } }$ has no class-C item left then remove $n _ { \mathrm { s r c } }$ from N
14: if not progress then break
15: end while
16: end for
17: return �
Notation. For class $C \in \{ \mathrm { p o o l } , \mathrm { w } 4 , \mathrm { w } 2 \}$ , coun $\mathfrak { t } _ { C } ( n )$ is the num
ber of class-C items currently on $n , \mathrm { c a p } _ { C } ( n )$ is the maximum
number of class-C items that � can hold, room $\mathbf { \boldsymbol { \mathrm { \Lambda } } } _ { C } ( n ) = \mathbf { \boldsymbol { \mathrm { c a p } } } _ { C } ( n ) -$
count (�), and size $( x ) = 1$ in all three classes. The local variables
$n _ { \mathrm { s r c } }$ and $n _ { \mathrm { d s t } }$ denote, respectively, the source node from which
an item is migrated and the destination node into which it is
migrated within the current best-fit drain pass.
```

## 4.4 Theoretical Guarantees

We now state the main theoretical properties ofthe COMPASS-ABS scheduler. Proposition 1 validates that COMPASS confines the cluster state to the ABS domain Σ throughout. Theorem 1 is the main claim of this section and shows that COMPASS-ABS keeps $\mathrm { S I F } ^ { \pi } ( t )$ bounded by $2 / N$ whenever the workload satisfies a realistic composition condition. Proposition 2 characterises the per-operator computational complexity of the COMPASS algorithm. Detailed proofs of all three results are deferred to Appendix B.

Algorithm 1 The COMPASS algorithm: online event handler   
that composes the Place, Remove, and Compact operators   
to evolve the cluster state � inside the compact ABS Σ.   
Input: initial state � ∈ Σ; online event stream $\langle e _ { 1 } , e _ { 2 } , \ldots \rangle$ , in   
which each event $e _ { k }$ is either the arrival or the departure of a   
DLT job � with �<sub>�</sub> instances   
Output: evolving state � updated after every event   
1: � ← �<sub>0</sub>   
2: for each event � in arrival order do   
3: if � is the arrival of job � then   
4: for $k : = 1 , 2 , \ldots , w _ { j }$ do   
5: � ← the �-th instance of �   
6: (�, �) ← Place(�, �)   
7: install � at � and then apply � to �   
8: end for   
9: � ← � ⊕ Compact(�)   
10: else if � is the departure of job � then   
11: for � := 1, 2, . . . , � do   
12: � ← the �-th instance of �   
13: � ← Remove(�, �)   
14: remove � from its current location and then apply � to   
�   
15: end for   
16: � ← � ⊕ Compact(�)   
17: end if   
18: end for   
19: return �

Proposition 1 (State-Space Invariance). For any initial state $\sigma _ { 0 } \in \Sigma$ and any event sequence of job arrivals and job departures, every state produced by the COMPASS algorithm (Algorithm 1) remains in Σ.

Theorem 1 (Compactness under the Workload Composition Condition). Let $n _ { g } ( t )$ denote the number ofactive instances of per-GPU demand � $\in \{ 1 , \ldots , 8 \}$ at time �, write $n _ { \mathrm { t o t a l } } ( t ) =$ $\textstyle \sum _ { g = 1 } ^ { 8 } n _ { g } ( t )$ for the total active instance count, and let $p _ { g } ( t ) : =$ $n _ { g } ( t ) / n _ { \mathrm { t o t a l } } ( t )$ denote the per-class instance share. Suppose that at every time � the workload composition satisfies the Workload Composition Condition

$$
p _ { 1 } ( t ) \ge p _ { 3 } ( t ) + 3 p _ { 5 } ( t ) + 2 p _ { 6 } ( t ) + p _ { 7 } ( t ) .\tag{9}
$$

Then in every cluster state �(�) produced by Algorithm 1 in response to the online event stream, at most one slotted node is non-fully-packed and at most one pool node is non-fullypacked, and consequently the online placement produced by COMPASS-ABS satisfies

$$
\mathrm { S I F } ^ { \pi } ( t ) \ \le \ \frac { 2 } { N } , \qquad \forall t \ge 0\tag{10}
$$

where � is the total number of nodes in the cluster.

Proposition 2 (Operational Complexity of COMPASS). The three operators that compose COMPASS satisfy

```prolog
T (Place) = � (log �),
T (Remove) = �(1),
T (Compact) = �(�),
```

(11)

where � denotes the number ofnodes in the cluster.

## 5 Experimental Evaluation

## 5.1 Testbed

We evaluate our scheduler in both a physical cluster and a simulation system. Our implementation is available at htps: //anonymous.4open.science/r/COMPASSABSF02A/.

For the physical experiments, our scheduler runs as a controller process that launches and migrates real PyTorch training jobs on a small cluster of four servers each equipped with eight NVIDIA A100-40GB GPUs, and each migration is realized through a checkpoint-and-restart protocol on the actual training processes.

For the simulation experiments, we develop an eventdriven simulator in Python to reproduce DLT job production in a shared GPU cluster. To faithfully reflect production behavior, our simulator reproduces the latency incurred by the checkpoint-based migration mechanism[12, 50].

## 5.2 Cluster Configuration

To validate our scheduler under both contended and loose resource regimes, we replay real traces (or deploy realistic jobs) on a range of cluster scales defined as follows. Let $\mathcal { T } _ { [ T _ { 1 } , T _ { 2 } ] }$ represent the jobs submitted in a chosen window $[ T _ { 1 } , T _ { 2 } ]$ , and let

$$
D ( t ) \ = \ \sum _ { j \in { \mathcal { J } } _ { [ T _ { 1 } , T _ { 2 } ] } } G _ { j } { \mathbb { K } } \big [ { a } _ { j } \leq t < { a } _ { j } + { \tau } _ { j } \big ]\tag{12}
$$

denote their concurrent GPU demand, with $G _ { j } = W _ { j } g _ { j }$ . We define the window’s �-th percentile demand $\hslash \eta$ as the smallest level ℓ that � stays below for at least an � fraction of its support. The cluster for this window is then sized at

$$
N ( \eta ) \ = \ \mathrm { m a x } \Big ( \big \lceil p _ { \eta } / 8 \big \rceil , \ N ^ { \mathrm { m a x j o b } } \Big )\tag{13}
$$

nodes, where $N ^ { \mathrm { m a x j o b } }$ is the minimum node count required to host the largest single job in $\mathcal { T } _ { [ T _ { 1 } , T _ { 2 } ] }$ under the intra-node constraint and gang scheduling requirement.

## 5.3 Trace

Our simulation is conducted on two real production traces, Venus[15] and GFS[7]. Both traces record the number of instances, per-instance GPU demand, submission time, and duration of each job. As an initial preprocessing step, we retain only the DLT jobs that consist of homogeneous instances with identical integer GPU demand. For the physical deployment, we synthesize a DLT trace (Phys-Mix) spanning CV[6, 14], NLP[5], LLM[39], recommender system[11], and multi-modal[38] training tasks, whose resource demand characteristics are aligned with those observed in the two traces above.

## 5.4 Comparison

We compare our scheduler against the six state-of-the-art methods.

• GFS[7]: places each job’s instances on the nodes whose remaining GPU capacity most closely matches the perinstance demand, following a best-fit policy.

• FGD[45]: selects the placement that minimizes the resulting increase in fragmentation they define.

• CAFGD[24]: combines its lifecycle-aware fragmentation measure with a wait-for-peak admission policy

• MCG[48]: builds on FGD’s fragmentation-gradient principle and additionally estimates the demand distribution of pending jobs to reorder scheduling priorities.

• FFT[30]: globally reorganizes the placement of all running jobs by solving an ILP at the start of each round, and schedules arriving jobs within each round using a best-fit.

• DRR[49]: periodically migrates the jobs on the lowestutilization node onto high-utilization target nodes at the start of each reschedule cycle, and places arriving jobs using a DRL policy trained via imitation learning from a best-fit heuristic.

## 5.5 Metrics

We evaluate COMPASS-ABS and the baselines using three metrics that together capture how efectively each scheduler mitigates the consequences of GPU fragmentation. Among them, AFR (and the underlying SIF) is a diagnostic quantity we introduce in this work to measure the cluster fragmentation state; others are key performance indicators for both users and vendors. Together they provide a complete assessment of scheduler eficiency.

To focus on contention-induced behavior, we restrict the metrics indicating resource fragmentation degree and GPU utilization to the peak-demand period

$$
T _ { \mathrm { p e a k } } \ = \ \{ \ t \in T \ | \ Q ( t ) \neq \emptyset \} ,\tag{14}
$$

that is, the time within the observed window � when the waiting queue �(�) is non-empty.

Average Fragmentation Rate (AFR). The time-averaged cluster fragmentation level during $T _ { \mathrm { p e a k } }$ , where the instantaneous fragmentation SIF(�) follows the definition in (2):

$$
\mathrm { A F R } \ = \ \frac { 1 } { | T _ { \mathrm { p e a k } } | } \int _ { T _ { \mathrm { p e a k } } } \mathrm { S I F } ( t ) \ d t .\tag{15}
$$

A lower AFR indicates that the scheduler keeps the cluster in a less fragmented cluster state under contention.

Average Job Completion Time (AJCT). The mean completion time across all DLT jobs submitted:

$$
\mathrm { A J C T } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } C _ { i } ,\tag{16}
$$

where $C _ { i }$ is the completion time of job $J _ { i } ,$ equivalently defined as the sum of its queueing time in the waiting queue, its training time on the assigned GPUs and latency caused by migration, and � is the total number of jobs evaluated. A lower AJCT indicates that the scheduler admits and finishes jobs more eficiently, which directly shortens the user-perceived turnaround of DLT workloads.

Average GPU Utilization (AGU). The time-averaged fraction of active GPUs in the cluster during $T _ { \mathrm { p e a k } }$

$$
\mathrm { A G U } \ = \ \frac { 1 } { | T _ { \mathrm { p e a k } } | } \int _ { T _ { \mathrm { p e a k } } } \mathrm { G U } ( t ) d t , \quad \mathrm { G U } ( t ) \in [ 0 , 1 ] .\tag{17}
$$

A higher AGU indicates that the scheduler converts more of the available GPU capacity into productive work, which is critical under heavy GPU demand.

## 6 Evaluation

## 6.1 Overall Comparison

We first evaluate the end-to-end performance of COMPASS-ABS against all baselines under a fixed resource regime $( \eta = 0 . 9 0 )$ . The comparison spans two real production traces replayed in our simulator and Phys-Mix deployed on our physical cluster, with results reported in Fig. 1. Overall, COMPASS-ABS dominates the six baselines on every (metric, trace) pair. Concretely, COMPASS-ABS reduces AJCT by at least 212s, 135s and 736s and lifts AGU by at least 0.7%, 2% and 16% over the strongest baseline on the Venus, GFS and Phys-Mix traces respectively, while simultaneously attaining the lowest AFR on all three traces with the measured AFR staying below $\frac { 2 } { N }$ , which matches the theoretical bound proved in Section 4.4. COMPASS-ABS, DRR and FFT all support migration-driven defragmentation, yet ours exceeds DRR and FFT for a structural reason: our migration policy is driven by the COMPASS algorithm to maintain the compactness of the predefined feasible state domain ABS, and is tightly coupled with the placement strategy rather than treated as a separate phase at fixed interval.

## 6.2 Cluster Intensity Sensitivity

To verify that the superiority of COMPASS-ABS over the six baselines is stable across diferent cluster resource pressures, we sweep the cluster-scaling parameter $\eta \in \{ 0 . 8 5 , 0 . 9 0 , 0 . 9 5 \}$ on the GFS trace, which carries the largest job count and aggregate GPU demand among the three traces and therefore ofers the most discriminative regime for scheduler comparison. Across all three intensity levels, COMPASS-ABS dominates all baselines on every metric reported in Fig. 2, with its AFR lying within the narrow band of 0.0023 to 0.0026 and its AGU staying above 0.957, and neither metric exhibits any appreciable sensitivity to resource pressure across the three regimes.

![](images/4b46bf6250e83210c60f56a6b87239cf4e751df5add711a8c9b7d509fb5546d2.jpg)

![](images/4d68bdd02d818a9e8860ec103341a48c687e089789fe7efaf69fcb418b65e915.jpg)

![](images/3d40b441f7059435dfd2f40c7c12b2180edbb87b54124b160c67f547aadd8774.jpg)  
Figure 1. Overall comparison across three DLT workload traces (� = 0.90).

![](images/5ed5c0d224604470db29d64582a1ad9d2a82aef4e988489a01a991c42ecc06c3.jpg)  
Figure 2. Sensitivity to cluster resource intensity on the GFS trace $\eta \in \{ 0 . 8 5 , 0 . 9 0 , 0 . 9 5 \}$

By contrast, the AJCT advantage of COMPASS-ABS over the strongest baseline widens monotonically as the cluster becomes tighter, growing from 74s at $\eta \ : = \ : 0 . 9 5$ to 135s at $\eta = 0 . 9 0$ and ultimately to 201s at $\eta = 0 . 8 5$ . The mechanism behind is that fragmentation-induced queueing blockage grows increasingly severe as the cluster tightens: under tight resources, fragmented capacity is far less likely to be reclaimed by the natural departure of running jobs, whereas under loose resources such fragments are readily absorbed. COMPASS-ABS adaptively eliminates fragmentation in real time and thereby recovers this otherwise-trapped capacity, so its fragmentation-aware advantage translates into the largest AJCT reduction when the cluster is at its tightest.

## 6.3 Robustness to Workload Composition

To verify that the superiority of COMPASS-ABS over the six baselines is also stable across diferent per-instance GPUdemand distributions, we select three days whose compositions span the widest pairwise discrepancy across our traces (Table 2). Across all three compositions, COMPASS-ABS dominates every baseline on every metric reported in Fig. 3: its AFR is confined within [0.002, 0.030] against the baselines’ wider [0.058, 0.324] band, its AGU stays above 0.89 against the baselines’ [0.77, 0.90] range, and its AJCT leads the strongest baseline by 1,643 s on Venus day 92, 1,431 s on GFS day 58 and 145 s on GFS day 26.

This robustness arises from the fact that the slot-pool layout exploits the dominance of the fragmentation-friendly {2, 4, 8}-GPU widths over the fragmentation-unfriendly {3, 5, 6, 7}-GPU widths in real DLT workloads, so the anchor-filler mechanism saturates the slots and the COMPASS algorithm confines the cluster state within a compact ABS configuration regardless of how the per-instance GPU-demand distribution shifts. The absolute AJCT lead is the largest on Venus day 92 because most of the JCT on this short-job workload is determined by scheduling decisions, whereas it shrinks on the two GFS days because their longer base runtimes dominate the JCT and leave a smaller scheduling-controllable share for any scheduler to optimize; even so, COMPASS-ABS still ranks first against other baselines.

Table 2. Workload Composition of the Three Selected Days
<table><tr><td rowspan="2">Day</td><td colspan="2">Instance Size Ratio</td></tr><tr><td>1 GPU</td><td>2-7 GPU 8 GPU</td></tr><tr><td>Venus Day 92</td><td>0.73</td><td>0.06 0.21</td></tr><tr><td>GFS Day 58</td><td>0.57</td><td>0.04 0.39</td></tr><tr><td>GFS Day 26</td><td>0.43</td><td>0.09 0.48</td></tr></table>

## 6.4 Workload Composition Condition: Coverage and Robustness on Violating Days

The performance guarantee of COMPASS-ABS rests on Theorem 1, which assumes the cluster state satisfies the Workload Composition Condition (WCC) at every scheduling tick. We therefore test how often the WCC holds and whether COMPASS-ABS keeps its advantage on days where it does not hold throughout. We replay every Venus and GFS day, measure each day’s WCC coverage �, the fraction of its scheduling horizon during which the WCC is satisfied, bucket the days by �, and for each bucket and each metric in {AFR, AJCT, AGU} count the days on which COMPASS-ABS ranks top-1 among the seven evaluated schedulers (Table 3).

The WCC is empirically dominant on both traces. On Venus, 487 of the 564 replayed days (86.3%) satisfy it throughout the entire horizon, and a further 51 days (9.0%) satisfy it for at least 90% of the horizon; on GFS it holds across the full horizon on all 552 days. The WCC-satisfying regime is thus the principal operating regime of production deployments rather than a corner case.

Even on the Venus days where the WCC does not hold throughout, the top-1 frequency of COMPASS-ABS closely tracks that on the $\rho ~ = ~ 1$ days. Across the five coverage buckets with $\rho \ : < 1$ , it leads on AFR for 100% of the days in the three lower-coverage buckets [0, 0.1), [0.1, 0.7), and [0.7, 0.9) and for at least 88% in the two higher-coverage buckets, while on AJCT and AGU it leads for at least 64% in every WCC-violating bucket and for 100% in the lowestcoverage bucket [0, 0.1). Aggregated across all 564 Venus days, the top-1 frequency reaches 94.1% on AFR, 80.3% on AJCT, and 81.9% on AGU, and on GFS 100%, 97.3%, and 95.9% respectively. COMPASS-ABS thus retains its advantage on all three metrics even outside this regime.

Table 3. Top-1 ranking frequency of COMPASS-ABS across WCC coverage � buckets, grouped by (day, �) pairs.
<table><tr><td>Trace</td><td>ρBucket</td><td>|D|</td><td>AFR</td><td>JCT</td><td>AGU</td></tr><tr><td rowspan="9">Venus</td><td>1.0</td><td>487</td><td>459 (94.3%)</td><td>383 (78.6%)</td><td>404 (83.0%)</td></tr><tr><td>[0.99, 1)</td><td>26</td><td>24 (92.3%)</td><td>18 (69.3%)</td><td>20 (76.9%)</td></tr><tr><td>[0.9, 0.99)</td><td>25</td><td>22</td><td>16</td><td>16</td></tr><tr><td>[0.7, 0.9)</td><td>11</td><td>(88.0%) 11</td><td>(64.0%) 8</td><td>(64.0%) 9</td></tr><tr><td>[0.1, 0.7)</td><td>10</td><td>(100%) 10</td><td>(72.7%) 8</td><td>(81.8%) 8</td></tr><tr><td>[0, 0.1)</td><td>5</td><td>(100%) 5 (100%)</td><td>(80.0%) 5 (100%)</td><td>(80.0%) 5 (100%)</td></tr><tr><td>Total</td><td>564</td><td>531 (94.1%)</td><td>453 (80.3%)</td><td>462 (81.9%)</td></tr><tr><td></td><td></td><td>552</td><td>537</td><td>529</td></tr><tr><td>GFS 1.0</td><td></td><td>552</td><td>(100%)</td><td>(97.3%)</td><td>(95.9%)</td></tr></table>

## 6.5 Robustness to Migration Cost

Empirical migration latencies in DLT clusters span the range of 1s to 10s, depending on the checkpoint-and-restore implementation, the network topology and the model size [13, 30, 50, 51]; on our own physical cluster the average migration latency we measure is approximately 8s. We therefore verify that COMPASS-ABS maintains its superiority across the realistic range by sweeping the per-migration cost parameter $c \in \{ 1 , . . . , 1 0 \}$ s on the GFS and Venus traces.

As shown in Fig. 4, every one of the six panels exhibits essentially flat curves as the per-migration cost � sweeps from 1 s to 10 s, because the three migration-capable schedulers each issue at most a small number of migrations per job and the cumulative cost contribution to any metric therefore stays well below one second per job even at $c = 1 0 ~ s$ . More importantly, COMPASS-ABS attains the lowest AFR, the lowest AJCT and the highest AGU at every � in the swept range on both GFS and Venus, including the 8 s neighborhood that corresponds to the latency we measure on our own cluster, so the end-to-end advantage of COMPASS-ABS is insulated against any realistic increase in the per-migration cost.

## 6.6 Case Study of Fragmentation Dynamics

We zoom into the 21:00–22:30 window of GFS day 113 under $\eta = 0 . 9 5$ to expose the time-resolved fragmentation mechanism, with Fig. 5, Fig. 6 and Fig. 7 reporting the dynamic � · SIF (the total number of avoidable partial nodes), GPU utilisation and fully-empty node counts respectively. All six practical baselines remain in continuous blocking through out the window, with five of them attributing 100% of the blocking to fragmentation and DRR still $9 4 . 6 \% ,$ , as verified by the aggregate idle GPU count strictly exceeding the head-ofline demand. Over 90% of the blocked jobs carry $g _ { i } = 8$ with $w _ { i } > 3 0 _ { : }$ , so the dominant blocking factor is the requirement for at least $w _ { i }$ fully empty 8-GPU nodes.

![](images/ca076f67d5f398682a7e171e14a8b3514a99aeefbe621e765589c0c019ba3b78.jpg)  
Figure 3. Robustness to workload composition on three representative days (Venus Day 92, GFS Day 58, GFS Day 26)

![](images/71d791f7b09fcd3c946c6bece71fd979ed03f4fb2b18cbfa44a9771f9e51bbbd.jpg)  
Figure 4. Robustness to per-migration cost on the GFS and Venus trace, $c \in [ 1 , 1 0 ] \ \mathrm { s } , \eta = 0 . 9 0$

On the fragmentation side, the five non-migration baselines settle into a wide 60–80 partial-node band while FFT confines itself to a 25–50 band through periodic global mi gration, but both ranges leave only 20–30 fully empty nodes against the $w _ { i } > 3 0$ contiguous demand and trap GPU utilisation within 0.8–0.9 for the entire window. COMPASS-ABS keeps � · SIF bounded by 2 throughout, so its total blocking accumulates to only about 10 minutes, and although 72.9% of that blocking is technically fragmentation-induced, the compact ABS layout together with its slot-pool migration mechanism dispatches every such blockage within seconds and never lets fragmentation translate into long-period queue blocking or low utilisation.

![](images/7fd1926df52d6048ab857b4ed6f89a34a79d9d28ae8a2e9ac5d7544a118a4064.jpg)  
Figure 5. Dynamic number of avoidable partial nodes (� SIF) by seven schedulers on GFS day 113 (� = 0.95).

## 7 Related Work

Most prior work on DLT job scheduling overlooks the quantification of resource fragmentation, asserting that they mitigate fragmentation via tight packing, without providing a concrete metric[27, 40, 47]. FGD[45] was the first to propose a statistical measure that estimates the expected instance capacity that the remaining resources can accommodate, given prior knowledge of the resource demand distribution. Subsequent works refined this formulation on top of FGD: CAFGD[24] decomposed the demand probability into weighted long-term and short-term distributions, while MCG[48] incorporated load balance into the fragmentation metric. However, all of these approaches rely on historical workload information.

Numerous studies aimed to reduce DLT job completion time and improve resource utilization by mitigating fragmentation. Existing schedulers can be broadly categorized into two types of approaches. Non-migration schedulers: These methods formulated scheduling as a multi-dimensional bin packing problem[3, 8, 16, 42, 52]. ElasticFlow[9], Synergy[31], and GFS[7] adopted a best-fit strategy that dynamically places jobs on the node leaving the fewest idle GPUs. FGD[45], CAFGD[24], and MCG[48] scheduled jobs based on their respective fragmentation measures, preferring placements that minimize the increase between pre- and post-scheduling fragmentation. These methods cannot reduce fragmentation that has already accumulated in the cluster. Migration-supporting schedulers: To address accumulated fragmentation, schedulers such as Gavel[33], Sia[21], and RASA[2] modeled the problem as a linear program computed by numerical solvers. However, the resulting time complexity and additional latency induced by frequent migrations are prohibitive for real-time scheduling in large-scale clusters[46]. To improve practicality, Hops[43], FFT[30] and DRR[49] employed a round-based migration strategy that performs global defragmentation by relocating running jobs to placements that incur less fragmentation. Nevertheless, these methods cannot correct fragmentation introduced within a round, leaving the cluster in a degraded state until the next defragmentation cycle.

![](images/e68aeb3f741133643c149d49af2bc0c647036e14371ca4dfcf1919105fc7b3e9.jpg)  
Figure 6. Dynamic GPU utilization by seven schedulers on GFS day 113 (� = 0.95).

![](images/0bc0cfdebf7caf38fba391e384838b59756009bedf96e402c9b7009c0ec981fe.jpg)  
Figure 7. Dynamic free nodes by seven schedulers on GFS day 113 (� = 0.95).

## 8 Conclusion

In this paper, we have presented Scheduler-Induced Fragmentation (SIF) for quantifying resource fragmentation together with COMPASS-ABS, a scheduler that reduces fragmentation in shared GPU clusters for DLT jobs. From the partial-nodes perspective, SIF removes the dependence on prior knowledge of the workload trace that earlier statistical metrics require, and isolates the portion of observed fragmentation resulting from scheduling policy. Building on this measure, COMPASS-ABS confines the cluster state to a structured feasible domain called the Anchor-Based Space (ABS), whose construction fully exploits the alignment between the GPU-demand profile of DLT instances and the cluster topology, so that the accompanying COMPASS algorithm can dynamically maintain compactness inside ABS and keep SIF uniformly bounded by 2/� whenever the Workload Composition Condition holds. We validate our scheduler’s performance through large-scale simulation on production traces and through physical-cluster deployment with a synthetic DLT workload, both of which show that COMPASS-ABS attains higher GPU utilization and substantially shorter job completion time in the queue than the state-of-the-art baselines even when WCC is violated.

Looking ahead, we identify three promising directions for extending COMPASS-ABS. First, we plan to generalize our scheduler to jointly handle both deep learning training and inference jobs in a shared GPU cluster, where the heterogeneous latency requirements and resource profiles of the two workload types pose additional scheduling challenges. Second, when migrating jobs to amend fragmentation, the current design treats all jobs uniformly; a natural extension is to incorporate job-level priorities into the migration decision, selecting victims in a manner that respects Service Level Objectives (SLOs) and avoids penalizing high-priority workloads. Moreover, we further discuss how to extend our scheduler to the next-generation clusters with larger NVLink domains in Appendix D.

## References

[1] Advanced Micro Devices. 2024. AMD Instinct MI300X Platform Data Sheet. Data sheet. htps://www.amd.com/content/dam/amd/en/ documents/instinct-tech-docs/data-sheets/amd-instinct-mi300xplatform-data-sheet.pdf Accessed: 2026-06-09.

[2] Zhe Chen, Fan Jiang, Bin Chen, Yu Li, Yi Zhang, Chao Huang, Run Yang, Fei Jiang, Jun Chen, Wei Xiang, Gang Cheng, Ran Shi, Nan Ma, Wenjie Zhang, and Tao Zhang. 2024. Resource Allocation with Service Afinity in Large-Scale Cloud Environments. In Proceedings

of the IEEE International Conference on Data Engineering (ICDE ’24). IEEE, 5280–5293. doi:10.1109/ICDE60146.2024.0039

[3] Zhiyuan Chen, Xin Zhao, Chenglu Zhi, and Jianwei Yin. 2023. Deep-Boot: Dynamic Scheduling System for Training and Inference Deep Learning Tasks in GPU Cluster. IEEE Transactions on Parallel and Distributed Systems 34, 9 (2023), 2553–2567. doi:10.1109/TPDS.2023. 3293835

[4] Arnab Choudhury, Yang Wang, Tuomas Pelkonen, Kutta Srinivasan, Abha Jain, Shenghao Lin, Delia David, Siavash Soleimanifard, Michael Chen, Abhishek Yadav, Ritesh Tijoriwala, Denis Samoylov, and Chun qiang Tang. 2024. MAST: Global Scheduling of ML Training across Geo-Distributed Datacenters at Hyperscale. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24). USENIX Association, 563–580.

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT ’19). 4171–4186.

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In Proceedings of the International Conference on Learning Representations (ICLR ’21).

[7] Jiangfei Duan, Shenggui Xu, Shilong Qian, et al. 2026. GFS: A Preemption-aware Scheduling Framework for GPU Clusters with Predictive Spot Instance Management. In Proceedings ofthe 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS ’26). ACM, 117–131. doi:10.1145/3760250.3762231

[8] Robert Grandl, Ganesh Ananthanarayanan, Srikanth Kandula, Sriram Rao, and Aditya Akella. 2014. Multi-Resource Packing for Cluster Schedulers. In Proceedings ofthe 2014 ACM Conference on SIGCOMM (SIGCOMM ’14). ACM, 455–466. doi:10.1145/2619239.2626334

[9] Diandian Gu, Yihao Zhao, Yinmin Zhong, Yifan Xiong, Zhenhua Han, Peng Cheng, Fan Yang, Gang Huang, Xin Jin, and Xuanzhe Liu. 2023. ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning. In Proceedings ofthe 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’23). ACM, 266–280. doi:10.1145/3575693. 3575721

[10] Juncheng Gu, Mosharaf Chowdhury, Kang G. Shin, Yibo Zhu, Myeongjae Jeon, Junjie Qian, Hongqiang Harry Liu, and Chuanxiong Guo. 2019. Tiresias: A GPU Cluster Manager for Distributed Deep Learning. In Proceedings of the USENIX Symposium on Networked Systems Design and Implementation (NSDI ’19). USENIX Association, 485–500.

[11] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI ’17). 1725–1731.

[12] Tanmaey Gupta, Sanjeev Krishnan, Rituraj Kumar, Abhishek Vijeev, Bhargav Gulavani, Nipun Kwatra, Ramachandran Ramjee, and Muthian Sivathanu. 2024. Just-in-time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures. In Proceedings ofthe Nineteenth European Conference on Computer Systems (EuroSys ’24). ACM, 1110–1125. doi:10.1145/3627703.3650085

[13] Jingoo Han, Mustafa Rafique, Luna Xu, Ali R. Butt, Seung-Hwan Lim, and Sudharshan Vazhkudai. 2020. MARBLE: A Multi-GPU Aware Job Scheduler for Deep Learning on HPC Systems. In Proceedings of the IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGRID ’20). IEEE, 272–281. doi:10.1109/CCGrid49817. 2020.00-66

[14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual Learning for Image Recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR ’16). IEEE, 770–778. doi:10.1109/CVPR.2016.90

[15] Qinghao Hu, Peng Sun, Shengen Yan, Yonggang Wen, and Tianwei Zhang. 2021. Characterization and Prediction of Deep Learning Work loads in Large-Scale GPU Datacenters. In Proceedings of the International Conferencefor High Performance Computing, Networking, Storage and Analysis (SC ’21). ACM, 1–15. doi:10.1145/3458817.3476223

[16] Dawei Huang, Peng Du, Chuanwen Zhu, Hao Zhang, and Xudong Liu. 2015. Multi-Resource Packing for Job Scheduling in Virtual Machine Based Cloud Environment. In Proceedings of the IEEE Symposium on Service-Oriented System Engineering (SOSE ’15). IEEE, 216–221. doi:10. 1109/SOSE.2015.30

[17] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. 2019. GPipe: Eficient Training of Giant Neural Networks Using Pipeline Parallelism. In Advances in Neural Information Processing Systems (NeurIPS ’19), Vol. 32. 103–112. htps://proceedings.neurips.cc/paper/2019/hash/ 093f65e080a295f8076b1c5722a46aa2-Abstract.html

[18] Huawei Technologies. 2021. Atlas 800 Training Server (Model 9000) Data Sheet. Data sheet. htps://www.cmc.ca/wp-content/uploads/ 2021/06/DatasheetAtlas800.pdf Accessed: 2026-06-09.

[19] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, Joe Chau, Peng Cheng, Fan Yang, Mao Yang, and Yongqiang Xiong. 2023. Tutel: Adap tive Mixture-of-Experts at Scale. In Proceedings of Machine Learning and Systems (MLSys ’23).

[20] Intel Corporation. 2024. Intel Gaudi 3 AI Accelerator HLB-325 Baseboard Product Brief. Product brief. htps: //www.intel.com/content/www/us/en/content-details/817489/intelgaudi-3-ai-accelerator-hlb-325-baseboard-product-brief.htm Accessed: 2026-06-09.

[21] Suhas Jayaram Subramanya, Daiyaan Arfeen, Shouxu Lin, Aurick Qiao, Zhihao Jia, and Gregory R. Ganger. 2023. Sia: Heterogeneityaware, Goodput-optimized ML-cluster Scheduling. In Proceedings of the ACM Symposium on Operating Systems Principles (SOSP ’23). ACM, 642–657. doi:10.1145/3600006.3613175

[22] Myeongjae Jeon, Shivaram Venkataraman, Amar Phanishayee, Junjie Qian, Wencong Xiao, and Fan Yang. 2019. Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads. In Proceedings of the USENIX Annual Technical Conference (USENIX ATC ’19). USENIX Association, 947–960.

[23] Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, Yangrui Chen, Zhi Zhang, Yanghua Peng, Xiang Li, Cong Xie, Shibiao Nong, Yulu Jia, Sun He, Hongmin Chen, Zhihao Bai, Qi Hou, Shipeng Yan, Ding Zhou, Yiyao Sheng, Zhuo Jiang, Haohan Xu, Haoran Wei, Zhang Zhang, Pengfei Nie, Leqi Zou, Sida Zhao, Liang Xiang, Zherui Liu, Zhe Li, Xiaoying Jia, Jianxi Ye, Xin Jin, and Xin Liu. 2024. MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs. In Proceedings of the USENIX Symposium on Networked Systems Design and Implementation (NSDI ’24). USENIX Association, 745–760.

[24] Huazheng Lao, Rui Xu, Long Chen, and Jinquan Zhang. 2025. CAFGD: Reduce Fragmentation in Large-Scale Multi-tenant Clusters for GPU Sharing Workloads. In Proceedings of the IEEE International Conference on Distributed Computing Systems (ICDCS ’25). IEEE, 133–143. doi:10. 1109/ICDCS63083.2025.00022

[25] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2021. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In Proceedings ofthe International Conference on Learning Representations (ICLR ’21).

[26] Ang Li, Shuaiwen Leon Song, Jieyang Chen, Jiajia Li, Xu Liu, Nathan R. Tallent, and Kevin J. Barker. 2020. Evaluating Modern GPU Interconnect: PCIe, NVLink, NV-SLI, NVSwitch and GPUDirect. IEEE Transactions on Parallel and Distributed Systems 31, 1 (2020), 94–110. doi:10.1109/TPDS.2019.2928289

[27] Jiamin Li, Hong Xu, Yibo Zhu, Zherui Liu, Chuanxiong Guo, and Cong Wang. 2023. Lyra: Elastic Scheduling for Deep Learning Clusters. In Proceedings of the Eighteenth European Conference on Computer Systems (EuroSys ’23). ACM, 835–850. doi:10.1145/3552326.3587445

[28] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jef Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. 2020. PyTorch Distributed: Experiences on Accelerating Data Parallel Training. Proceedings of the VLDB Endowment 13, 12 (2020), 3005–3018. doi:10.14778/3415478.3415530

[29] Meta Platforms. 2022. Grand Teton: Meta’s Next-Generation AI Platform. Open Compute Project Global Summit announcement. htps://engineering.fb.com/2022/10/18/open-source/ocpsummit-2022-grand-teton/ Accessed: 2026-06-09.

[30] Zizhao Mo, Huanle Xu, and Wing Cheong Lau. 2025. Fast and Fair Training for Deep Learning in Heterogeneous GPU Clusters. In Proceedings of the 39th ACM International Conference on Supercomputing (ICS ’25). ACM, 324–338. doi:10.1145/3721145.3728488

[31] Jayashree Mohan, Amar Phanishayee, Janardhan Kulkarni, and Vijay Chidambaram. 2022. Looking beyond GPUs for DNN Scheduling on Multi-Tenant Clusters. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI ’22). USENIX Association, 579–596.

[32] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, and Matei Zaharia. 2019. PipeDream: Generalized Pipeline Parallelism for DNN Training. In Proceedings of the ACM Symposium on Operating Systems Principles (SOSP ’19). ACM, 1–15. doi:10.1145/3341301.3359646

[33] Deepak Narayanan, Keshav Santhanam, Fiodar Kazhamiaka, Amar Phanishayee, and Matei Zaharia. 2020. Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI ’20). USENIX Association, 481–498.

[34] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGres ley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. 2021. Eficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’21). ACM, 1–15. doi:10.1145/3458817.3476209

[35] NVIDIA Corporation. 2017. NVIDIA DGX-1 with Tesla V100 System Architecture. Technical Report. NVIDIA Corpora tion. htps://images.nvidia.com/content/pdf/dgx1-v100-systemarchitecture-whitepaper.pdf

[36] NVIDIA Corporation. 2026. NVIDIA DGX SuperPOD. Product page. htps://www.nvidia.com/en-us/data-center/dgx-superpod/ Accessed: 2026-04-30.

[37] Open Compute Project. 2023. OAI Universal Baseboard (UBB) Base Specification, Revision 2.0. Open Compute Project Foundation specification. htps://www.opencompute.org/documents/oai-ubb-basespecification-r2-0-v1-0-20230919-pdf Accessed: 2026-06-09.

[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin,Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings ofthe International Conference on Machine Learning (ICML ’21). 8748–8763.

[39] Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners. Technical Report. OpenAI.

[40] Sudarsanan Rajasekaran, Manya Ghobadi, and Aditya Akella. 2024. CASSINI: Network-Aware Job Scheduling in Machine Learning Clusters. In Proceedings ofthe USENIX Symposium on Networked Systems Design and Implementation (NSDI ’24). USENIX Association, 1403– 1420.

[41] Rajeev Thakur, Rolf Rabenseifner, and William Gropp. 2005. Optimization of Collective Communication Operations in MPICH. The International Journal ofHigh Performance Computing Applications 19, 1 (2005), 49–66. doi:10.1177/1094342005051521

[42] Abhishek Verma, Madhukar Korupolu, and John Wilkes. 2014. Evaluating Job Packing in Warehouse-Scale Computing. In Proceedings of the IEEE International Conference on Cluster Computing (CLUSTER ’14). IEEE, 48–56. doi:10.1109/CLUSTER.2014.6968735

[43] Qinghe Wang, Futian Wang, and Xinwei Zheng. 2024. Hops: Finegrained Heterogeneous Sensing, Eficient and Fair Deep Learning Cluster Scheduling System. In Proceedings of the 2024 ACM Symposium on Cloud Computing (SoCC ’24). ACM, 1–17. doi:10.1145/3698038. 3698515

[44] Qizhen Weng, Wencong Xiao, Yinghao Yu, Wei Wang, Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin, and Yu Ding. 2022. MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters. In Proceedings of the USENIX Symposium on Networked Systems Design and Implementation (NSDI ’22). USENIX Association, 945–960.

[45] Qizhen Weng, Lingyun Yang, Yinghao Yu, Wei Wang, Xiaochuan Tang, Guodong Yang, and Liping Zhang. 2023. Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent. In Proceedings ofthe USENIX Annual Technical Conference (USENIX ATC ’23). USENIX Association, 995–1008.

[46] Gerhard J. Woeginger. 1997. There is No Asymptotic PTAS for Two-Dimensional Vector Packing. Inform. Process. Lett. 64, 6 (1997), 293–297. doi:10.1016/S0020-0190(97)00179-8

[47] Bingyang Wu, Zili Zhang, Zhihao Bai, Xuanzhe Liu, and Xin Jin. 2023. Transparent GPU Sharing in Container Clouds for Deep Learning Workloads. In Proceedings of the USENIX Symposium on Networked Systems Design and Implementation (NSDI ’23). USENIX Association, 69–85.

[48] Haijie Wu, Xinhua Wang, Xiaoxuan Luo, Wangbo Shen, and Weiwei Lin. 2025. MCG-Sched: Multi-Cluster GPU Scheduling for Resource Fragmentation Reduction and Load Balancing. IEEE Transactions on Parallel and Distributed Systems 36, 12 (2025), 2789–2802. doi:10.1109/ TPDS.2025.3626153

[49] Qing Wu, Pengfei Chen, and Yu Wang. 2025. Defragmentation Scheduling with Deep Reinforcement Learning in Shared GPU Clusters. In Proceedings of the 2025 ACM Symposium on Cloud Computing (SoCC ’25). ACM, 402–415. doi:10.1145/3772052.3772242

[50] Wencong Xiao, Romil Bhardwaj, Ramachandran Ramjee, Muthian Sivathanu, Nipun Kwatra, Zhenhua Han, Pratyush Patel, Xuan Peng, Hanyu Zhao, Quanlu Zhang, Fan Yang, and Lidong Zhou. 2018. Gandiva: Introspective Cluster Scheduling for Deep Learning. In Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI ’18). USENIX Association, 595–610.

[51] Zhisheng Ye, Wei Gao, Qinghao Hu, Peng Sun, Xiaolin Wang, Yingwei Luo, Tianwei Zhang, and Yonggang Wen. 2024. Deep Learning Workload Scheduling in GPU Datacenters: A Survey. Comput. Surveys 56, 6 (2024), 1–38. doi:10.1145/3638757

[52] Yihao Zhao, Yuanqiang Liu, Yanghua Peng, Yibo Zhu, Xuanzhe Liu, and Xin Jin. 2022. Multi-Resource Interleaving for Deep Learning Training. In Proceedings ofthe ACM SIGCOMM 2022 Conference (SIG-COMM ’22). ACM, 428–440. doi:10.1145/3544216.3544224

## A Compactness of Best-Fit on Power-of-Two Workloads

This appendix formalises Observation 1 of Section 4.1. We consider the restricted setting in which every multi-GPU instance has demand $g _ { i } \in \{ 2 , 4 , 8 \}$ , i.e. the active workload is contained in $\boldsymbol { \mathcal { T } } ^ { \mathrm { F F } }$ , and the cluster runs a stripped-down variant of Algorithm 1: each arrival is handled by a single Place call that uses best-fit (no Compact pass on arrival), and each departure is handled by per-instance Remove calls followed by a single Compact pass on the departure side only. We show that this minimal policy is already suficient to enforce $\mathrm { S I F } ^ { \pi } ( t ) \leq 2 / N$ at all times, and we exhibit a finite event sequence that reaches the bound.

## A.1 Setup and statement

Let the active workload consist of items of size $g \in \{ 2 , 4 , 8 \}$ that arrive and depart online, and let each node be a bin of capacity $G = 8$ . We use $\Phi ^ { \pi } ( t ) : = \nu ( P ^ { \pi } ( t ) )$ for the partialnode count under policy � at time $t ,$ with $F ^ { \pi } ( t ) = \Phi ^ { \pi } ( t ) / N$ as in Section 3.2. The free space ofa partial bin lies in $\{ 2 , 4 , 6 \}$ because used capacity attainable by sums of items in {2, 4, 8} that strictly under-fill $G = 8$ is one of $\{ 2 , 4 , 6 \}$

Proposition 3 (Best-Fit Compactness on Power-of-Two Work loads). Let the policy place every arriving instance via best-fit, i.e. on the node leaving the smallest remaining capacity after placement and opening a fresh node when no existing node can host the instance, and let every departure be followed by a single Compact pass. Ifevery active instance has $g _ { i } \in \{ 2 , 4 , 8 \}$ then for every $t ~ \geq 0$ the state $\sigma ( t )$ produced by the policy satisfies

$$
\Phi ^ { \pi } ( t ) \ \leq \ 2 \qquad a n d \qquad \mathrm { S I F } ^ { \pi } ( t ) \ \leq \ \frac { 2 } { N } ,
$$

and both inequalities are tight in the sense that a finite arrival sequence reaches $\Phi ^ { \pi } ( t ) = 2$

## A.2 Proof of Proposition 3

We bound $\Phi ^ { \pi } ( t )$ separately on the two event types and combine the bounds. Throughout, “bin” is interchangeable with “node” and “item” with “instance”.

Part 1: best-fit on arrival leaves at most two partial bins. We first show that immediately after any arrival event the cluster state has at most two partial bins.

Lemma B.1 (free-value uniqueness). At any time during the arrival pass, at most one partial bin has remaining free space � for each $r \in \{ 2 , 4 , 6 \}$

Proof. Suppose for contradiction that two partial bins � and � have free space � at the same time. Let � be the bin whose free space last became $r ,$ and consider the placement event that produced free ${ \bf \nabla } \cdot ( B ) = r .$ Just before that placement, free $( A ) = r$ already held by hypothesis. The placement put an item of size � into �, with $s \leq \mathrm { f r e e } ( B ) _ { \mathrm { b e f o r e } }$ and free $( B ) _ { \mathrm { b e f o r e } } - s = r ;$ , so free $( B ) _ { \mathrm { b e f o r e } } \geq s .$ . If � was a fresh bin, then free $( B ) _ { \mathrm { b e f o r e } } ~ = ~ G ~ = ~ 8$ and $s = G - r > 0 ;$ bin � also had free space $r \ \geq s ,$ , since $r = G - s$ and $G \geq s$ forces $r \geq 0 ,$ , and we ruled out $r = 0$ because � is partial. Concretely, for every reachable $r \in \{ 2 , 4 , 6 \}$ the item size $s ~ = ~ G - r ~ \in ~ \{ 6 , 4 , 2 \}$ , but only $s \in \{ 2 , 4 \}$ are admissible power-of-two items (size 6 is not in the workload), so the only case is $s \in \{ 2 , 4 \}$ , in which � with ${ \mathrm { f r e e } } ( A ) = r$ also has free $( A ) \geq s$ and best-fit would have picked � over the fresh � because the fresh bin has $G = 8 > r$ . If � was not fresh, then free $( B ) _ { \mathrm { b e f o r e } } > r ;$ , so free $( A ) = r < \mathrm { f r e e } ( B ) _ { \mathrm { b e f o r e } } .$ Best-fit’s least-fit rule would then still have preferred � over $B ,$ because � leaves the smaller free space, provided � can host the item. The admissibility condition free(�) ≥ � is exactly $r \geq s .$ . It holds because the placement into � required free $( B ) _ { \mathrm { b e f o r e } } \geq s ,$ , and substituting $r = \mathrm { f r e e } ( B ) _ { \mathrm { b e f o r e } } - s$ rewrites $r \geq s$ as free $( B ) _ { \mathrm { b e f o r e } } \ \geq \ 2 s$ . The only case where $\mathrm { f r e e } ( A ) < s$ is when $r < s ,$ but this is already excluded by the construction ${ \mathrm { f r e e } } ( B ) _ { \mathrm { b e f o r e } } = r + s$ and the item size � being one of $\{ 2 , 4 \}$ which gives $r + s \leq G$ hence $r \leq G - s .$ . In every admissible sub-case best-fit’s tie-breaking prefers $A ,$ contradicting the placement having gone to �. □

Lemma B.2 (free-six isolation). If some partial bin has free space $^ { 6 , }$ then no other partial bin exists at the same time.

Proof. A bin with free = 6 contains exactly one size-2 item. Consider the most recent placement that put this size-2 item into the bin. The bin was empty just before the placement, because no other item-size combination from {2, 4, 8} yields used capacity 2. Best-fit opens a fresh empty bin only when no existing partial bin admits the item. The arriving item has size 2, so an admissible partial bin would only require free $\geq 2 ,$ which every partial bin satisfies because every partial free value lies in {2, 4, 6}. Hence no partial bin existed at the moment the size-2 item was placed in the fresh bin, and the new $\mathrm { f r e e } = 6$ bin is the sole partial bin at that instant.

Each subsequent arrival event either (i) places a size-2 item, which best-fit routes into the existing free = 6 bin and reduces its free space to 4, removing it from the free $= 6$ class; or (ii) places a size-4 item, which best-fit routes into the existing free = 6 bin and reduces its free space to $2 ;$ or (iii) places a size-8 item, which opens a fresh bin that immediately becomes full and creates no new partial bin. In none of these cases is a second partial bin created while the original free = 6 bin still has free space $^ { 6 , }$ so the isolation invariant is preserved until the next departure event. □

Lemma B.3 (two-partial bound). At any time immediately after an arrival placement, the cluster has at most two partial bins, and the only reachable two-partial configuration has free spaces {2, 4}.

Proof. Combining Lemmas B.1 and B.2, the set of free-space values present across partial bins is a subset of $\{ 2 , 4 , 6 \}$ in which each value appears at most once, and the value 6 cannot coexist with any other partial bin. Hence the only feasible multisets of free-space values across partial bins are ∅, {2}, {4}, {6}, {2, 4}, {2, 6}, {4, 6} from cardinality counting, and the last two are forbidden by Lemma B.2. The remaining configurations have at most two partial bins. □

Part 2: Compact on departure leaves at most one partial bin. Departures invoke a single Compact pass after the per-instance Remove calls have been applied. Under the present restriction $\mathcal { I } ^ { \mathrm { = 1 } } = \emptyset , \mathcal { I } ^ { \mathrm { F U F } } = \emptyset ,$ , so the active 1-GPU population $n _ { 1 } ( t )$ is identically zero and the total slot vacancy aggregated across anchored fragmentation-unfriendly instances is identically zero. The Workload Composition Condition (9) therefore reduces to $0 \geq 0$ and holds trivially at every time. Lemma A.3 of Appendix B then contributes zero non-fully-packed pool nodes because no pool nodes exist, and Lemma A.4 of Appendix B contributes at most one non-fully-packed slotted node from the width-4 and width-2 BestFitDrain passes. The post-departure state therefore satisfies $\Phi ^ { \pi } ( t ) \leq 1$

Combining the two parts. The cluster state changes only at event boundaries. After every arrival event Part 1 gives $\Phi ^ { \pi } ( t ) \ \leq \ 2 ,$ , and after every departure event Part 2 gives $\Phi ^ { \pi } ( t ) \leq 1 \leq 2$ . The bound $\Phi ^ { \pi } ( t ) \leq 2$ therefore holds at every event boundary and extends to every continuous time instant between consecutive events, so

$$
\begin{array} { l } { \Phi ^ { \pi } ( t ) ~ \le ~ 2 , } \\ { F ^ { \pi } ( t ) ~ = ~ \frac { \Phi ^ { \pi } ( t ) } { N } ~ \le ~ \displaystyle \frac { 2 } { N } , } \\ { \mathrm { S I F } ^ { \pi } ( t ) ~ \le ~ F ^ { \pi } ( t ) ~ \le ~ \displaystyle \frac { 2 } { N } . } \end{array}
$$

for all $t \geq 0$ , where the last inequality uses the non-negativity of the inherent fragmentation $\mathbf { \nabla } F ^ { \star } ( t )$ from Section 3.2.

Tightness. The bound $\Phi ^ { \pi } ( t ) = 2$ is realised by the following arrival sequence on an otherwise fully-packed cluster. Suppose all nodes other than one are fully packed and the remaining node hosts a single size-4 anchor together with one size-2 anchor, leaving free space 2. A DLT job composed of three size-4 instances now arrives. Best-fit considers each arriving instance in turn. The first instance encounters free space 2 on the partial node, which does not admit a size-4 item, so a fresh node is opened and the first instance is placed there, producing a new partial node with free space 4. The second instance sees free space 2 on the original partial node and free space 4 on the just-opened node, prefers the smaller fitting capacity, and is placed onto the just-opened node, which becomes fully packed. The third instance again finds the original partial node uninhabitable for a size-4 item and the now-full node also unable to host $\mathbf { i t } ,$ so a further fresh node is opened with free space 4. The cluster now contains two partial nodes, one with free space 2 and one with free space 4, matching the only feasible two-partial configuration identified in Lemma B.3 and witnessing $\Phi ^ { \pi } ( t ) = 2$ together with $\mathrm { S I F } ^ { \pi } ( t ) = 2 / N$ . This completes the proof of Proposition 3. □

## B Proofs for the Theoretical Guarantees of COMPASS-ABS

This appendix collects the detailed proofs of the three results stated in Section 4.4, namely the unconditional state-space invariance in Proposition 1, the conditional $\mathrm { S I F } \le 2 / N$ compactness bound in Theorem 1, and the per-operator complexity in Proposition 2. We keep the notation introduced in Section 4.2 for the ABS domain Σ and in Section 4.3 for the three operators Place, Remove, and Compact.

## B.1 Proof of Proposition 1 (State-Space Invariance)

We prove that every event applied by Algorithm 1 maps Σ into $\Sigma ,$ so that by induction every state generated from an initial $\sigma _ { 0 } \in \Sigma$ remains in Σ. Since the algorithm decomposes each job-level event into a finite sequence of per-instance operator calls together with at most one Compact call, it sufices to show that each operator branch preserves Σ when applied to a state already in Σ.

Place preserves Σ. Consider an arrival � with demand $g _ { i }$ applied to $\sigma \in \Sigma$ . We argue on the branches of Place.

If $g _ { i } \ \geq \ 2 ,$ the operator selects a width $w = w ^ { \star } ( g _ { i } )$ ∈ $\{ 2 , 4 , 8 \}$ and a host $n ^ { \star }$ such that either $n ^ { \star }$ is already slotted with $\begin{array} { r } { G - \sum _ { s \in S ( n ^ { \star } ) } w ( s ) \geq w , } \end{array}$ or $n ^ { \star }$ was empty and is promoted to slotted with an empty profile. In the first case, appending a slot of width � to $S ( n ^ { \star } )$ produces a new slot multiset whose total width is at most � and whose constituents lie in {2, 4, 8}, hence the new $S ( n ^ { \star } )$ remains in SlotConf as defined in (5). In the second case, the post-event $S ( n ^ { \star } ) = \{ w \}$ trivially lies in SlotConf. The anchor ofthe new slot is set to � so that $w \bigl ( s ^ { \star } \bigr ) = w ^ { \star } \bigl ( g _ { i } \bigr ) = w ^ { \star } \bigl ( g _ { a ( s ^ { \star } ) } \bigr )$ , which matches the constraint of Definition 2. The associated filler set is initialised to the up-to-� migrated 1-GPU instances with $\boldsymbol { r } = \boldsymbol { w } - \boldsymbol { g } _ { i }$ , so that $| F ( s ^ { \star } ) | \le r = w ( s ^ { \star } ) - g _ { a ( s ^ { \star } ) }$ and the capacity invariant (4) holds as an inequality. Every pool node from which a filler is drawn loses one occupant and therefore preserves $| P ( n ) | \leq G$ . No other node is touched.

If $g _ { i } = 1$ , the operator places � either into a slot vacancy $( g _ { a ( s ) } + | F ( s ) | < w ( s ) )$ , or into a non-full pool node $( | P ( n ) | <$ $G )$ , or onto a freshly promoted pool node initialised with $P ( n ) = \{ i \}$ . In the first case, $\vert F ( s ) \vert$ increases by one and the new value remains bounded by $w ( s ) - g _ { a ( s ) }$ by the entry condition $g _ { a ( s ) } + | F ( s ) | < w ( s )$ . In the second case, $| P ( n ) |$ increases by one and remains bounded by �. In the third case, the new pool node has $| P ( n _ { 0 } ) | = 1 \leq G$ . In all three cases the afected node remains a legal pool node or slotted node, and no slot configuration leaves SlotConf.

Remove preserves Σ. Consider a departure � applied to $\sigma \in \Sigma .$

If � is an anchor of some slot �, the operator dissolves � and reinserts each filler $f \in F ( s )$ by the 1-GPU rule of Place. Dissolving � removes one element from �(�) for the host node �, so the new $S ( n )$ is a sub-multiset of the original $S ( n )$ ∈ SlotConf, and any sub-multiset ofa SlotConf element again lies in SlotConf. The reinsertion step has already been shown to preserve Σ in the Place argument above, applied instance-by-instance to the elements of $F ( s )$

If � is a 1-GPU instance currently attached as a filler in some slot �, the operator simply removes � from $F ( s )$ , which can only decrease $\vert F ( s ) \vert$ and therefore preserves (4) as a strict inequality. If � is a 1-GPU pool occupant on some pool node $n ,$ the operator removes � from $P ( n )$ , which only decreases $| P ( n ) |$ and remains bounded by �.

Compactpreserves Σ. Migrations $( x , n _ { \mathrm { d s t } } )$ executed within the BestFitDrain loop are performed only when room<sub>C</sub> $\left( { { n } _ { \mathrm { d s t } } } \right) \ge$ size(�) holds for the current class C. Concretely, for the pool pass the destination pool node has $\vert P ( n _ { \mathrm { d s t } } ) \vert < G$ before the migration, so $| P ( n _ { \mathrm { d s t } } ) |$ stays at most � after the migration. For the width-4 pass the destination slotted node has $\lfloor D ( n _ { \mathrm { d s t } } ) / 4 \rfloor \ge 1$ so that $D ( n _ { \mathrm { d s t } } ) \geq 4$ , hence appending a width-4 slot keeps $\begin{array} { r } { \sum _ { s \in S ( n _ { \mathrm { d s t } } ) } w ( s ) + 4 \leq G } \end{array}$ and the new $S ( n _ { \mathrm { d s t } } )$ remains in SlotConf. The width-2 pass is analogous with the inequality $D ( n _ { \mathrm { d s t } } ) \ \geq \ 2$ . Migrating a slot also detaches its filler set and reattaches it to the new host, which by the capacity invariant $g _ { a ( s ) } + | F ( s ) | \leq w ( s )$ on the source side remains valid on the destination side because �(�) does not change under migration. Every migration therefore preserves Σ, and so does the entire Compact call which is a finite composition of such migrations.

Conclusion. By induction on the event sequence, every state produced by Algorithm 1 remains in $\Sigma ,$ which establishes Proposition 1. □

## B.2 Proof of Theorem 1 (Compactness under the Workload Composition Condition)

Throughout this subsection we assume that the fractionform Workload Composition Condition (9) stated inside Theorem 1 holds at the time � under consideration. The argument proceeds in three stages. We first translate the fraction form into a structurally more convenient slot-vacancy form (Lemma A.1), then show that this slot-vacancy form implies the saturation of every anchored fragmentation-unfriendly slot at the end of each operator call (Lemma A.2), and finally show that the three BestFitDrain passes inside Compact leave at most one partial node per node-type (Lemmas A.3 and A.4), which after normalisation by the cluster size � yields the $\mathrm { S I F } ^ { \pi } ( t ) \leq 2 / N$ bound.

Lemma A.1 (Slot-Vacancy Form of the Workload Composition Condition). The fraction-form condition (9) is algebraically equivalent to the slot-vacancy form

$$
n _ { 1 } ( t ) \ \geq \ \sum _ { s : a ( s ) \neq \emptyset } ( w ( s ) - g _ { a ( s ) } ) ,\tag{18}
$$

which asserts that the active 1-GPU population is at least as large as the total slot vacancy currently exposed by anchored fragmentation-unfriendly instances.

Proof. By Definition 1, every active anchor $a ( s ) \in \mathcal { I } ^ { \geq 2 }$ has a slot width $w \mathopen { } \mathclose \bgroup ( s \aftergroup \egroup ) = w ^ { \star } \bigl ( g _ { a ( s ) } \aftergroup \egroup )$ . Inspecting $w ^ { \star }$ in (3) we obtain $w ( s ) - g _ { a ( s ) } = 0$ for $g _ { a ( s ) } \in \{ 2 , 4 , 8 \}$ $w ( s ) - g _ { a ( s ) } = 1$ for $g _ { a ( s ) } = 3 , w ( s ) - g _ { a ( s ) } = 3$ for $g _ { a ( s ) } = 5 , w ( s ) - g _ { a ( s ) } = 2$ for $g _ { a ( s ) } = 6 ,$ , and $w ( s ) - g _ { a ( s ) } = 1$ for $g _ { a ( s ) } = 7$ . Summing over all active anchors and grouping by demand,

$$
\sum _ { s : a ( s ) \neq Q } \left( w ( s ) - g _ { a ( s ) } \right) \ = \ n _ { 3 } ( t ) + 3 n _ { 5 } ( t ) + 2 n _ { 6 } ( t ) + n _ { 7 } ( t ) .
$$

Substituting this identity into (18) and dividing both sides by $n _ { \mathrm { t o t a l } } ( t ) > 0$ yields (9), and the implication is reversible since each step is an equivalence. In what follows we use (18) interchangeably with the original form. □

Lemma A.2 (Slot Saturation under the Workload Composition Condition). Under (18), at the end of every event handled by Algorithm 1 (whether a job arrival or a job departure) the capacity invariant in (4) holds with equality for every active fragmentation-unfriendly anchor, namely

$$
g _ { a ( s ) } + | F ( s ) | = w ( s ) \qquad \forall s \mathrm { w i t h } a ( s ) \in { \cal T } ^ { \mathrm { F U F } } .\tag{19}
$$

Proof. By construction Place pulls $w ( s ) - g _ { a ( s ) }$ fillers from the pool whenever a fragmentation-unfriendly anchor is installed, and Remove in the 1-GPU branch additionally migrates one replacement filler from the pool whenever a filler departs from a slot. Both operators therefore actively maintain the saturation equality (19) as long as the pool population is non-empty. The only obstacle is filler shortage at the moment when fillers are being drawn, and the slot-vacancy form (18) guarantees that the total active 1-GPU population $n _ { 1 } ( t )$ is at least the total slot vacancy aggregated across all anchored fragmentation-unfriendly instances at the end of the event. Since Compact does not create new slot vacancy and the 1-GPU pool is internally redistributed across pool nodes without changing $n _ { 1 } ( t )$ , (18) carries through to the post-Compact state. Because Algorithm 1 now invokes Compact at the end of both arrival and departure branches, every event terminates in a post-Compact state, and the equality (19) therefore holds at the end of every event. □

Lemma A.3 (Pool Pass Compactness). After the pool pass of Compact, at most one pool node is non-fully-packed.

Proof. The pool pass instantiates BestFitDrain policy with coun $\operatorname { \lrcorner } _ { \operatorname { p o o l } } ( n ) = | P ( n ) | , \operatorname { c a p } _ { \operatorname { p o o l } } ( n ) = G$ , and roo $\operatorname { n } _ { \mathrm { p o o l } } ( n ) ~ =$ $G - \bar { \vert P ( n ) \vert }$ |, and the set of partial hosts is $N = \{ n : 0 <$ $| P ( n ) | ~ < ~ G \}$ . Each iteration picks the head of N as $n _ { \mathrm { { s r c } } }$ the node with the largest remaining room. The drain loop selects, for each 1-GPU item � on $n _ { \mathrm { s r c } }$ , the eligible destination with the smallest room $\mathrm { r o o m } _ { \mathrm { p o o l } } ( n _ { \mathrm { d s t } } ) \geq 1$ and migrates � there. Each migration strictly increases $| P ( n _ { \mathrm { d s t } } ) |$ | and strictly decreases $| P ( n _ { \mathrm { s r c } } )$ |.

The loop terminates either when $| N | \leq 1$ or when no progress is made during an iteration. Suppose for contradiction that termination occurs with $| N | \ge 2$ and no progress in the last iteration. “No progress” means that for every � on $n _ { \mathrm { s r c } }$ the eligible set $S = \{ n \in N \backslash \{ n _ { \mathrm { s r c } } \} : \mathrm { r o o m } _ { \mathrm { p o o l } } ( n ) \geq 1 \}$ is empty. But $| N | \ge 2$ implies the existence of at least one $n ^ { \prime } \in$ $N \setminus \{ n _ { \mathrm { s r c } } \}$ with $| P ( n ^ { \prime } ) | < G$ , equivalently $\mathrm { r o o m } _ { \mathrm { p o o l } } ( n ^ { \prime } ) \geq 1$ so $n ^ { \prime } \in S _ { \mathrm { { i } } }$ , a contradiction. Therefore termination implies $| { \cal N } | \le 1$ , which is exactly the claim that at most one pool node is non-fully-packed. □

Lemma A.4 (Slotted Compactness). After the width-4 and width-2 passes of Compact, and under the slot saturation guaranteed by Lemma A.2, at most one slotted node is nonfully-packed.

Proof. Under Lemma $\mathtt { A . 2 }$ every active slot is internally saturated, so a slotted node � is non-fully-packed if and only if $\begin{array} { r } { D ( n ) \ = \ G - \sum _ { s \in S ( n ) } w ( s ) \ > \ 0 } \end{array}$ , i.e. the node has unprofiled residual capacity. Possible values of $\begin{array} { r } { \sum _ { s \in S ( n ) } w \big ( s \big ) } \end{array}$ are subsums of multisets drawn from $\{ 2 , 4 , 8 \}$ bounded above by $G = 8 ,$ so a fully-packed slotted node has $\begin{array} { r } { \sum _ { s \in S ( n ) } w ( s ) = 8 } \end{array}$ achieved by one ofthe multisets $\{ 8 \} , \{ 4 , 4 \} , \{ 4 , 2 , 2 \} , \{ 2 , 2 , 2 , 2 \}$ while a non-fully-packed slotted node has $\begin{array} { r } { \sum _ { s \in S ( n ) } w ( s ) \leq 6 } \end{array}$ achieved by one of {2}, {4}, {2, 2}, {4, 2}, {2, 2, 2}.

The width-4 pass treats anchored width-4 slots as items and slotted nodes as hosts, with ${ \mathrm { c o u n t } } _ { \mathrm { w 4 } } ( n )$ counting width-4 anchored slots in $S ( n )$ and $\operatorname { r o o m } _ { \mathrm { w } 4 } ( n ) \ = \ \lfloor D ( n ) / 4 \rfloor$ . Repeating the contradiction argument of Lemma A.3 with the width-4 class, if the width-4 pass terminates with two or more nodes still satisfying $0 < \mathrm { c o u n t _ { w 4 } } ( n ) < \mathrm { c a p } _ { \mathrm { w 4 } } ( n )$ and with positive room on at least one of them, then the head source has an eligible destination and the loop must perform a migration, contradicting the no-progress termination condition. The pass therefore terminates with at most one slotted node in the width-4 partial set.

After the width-4 pass, consider the remaining partial slotted nodes. Each such node either has no width-4 slot at all, or it is the unique node left in the width-4 partial set. The first sub-population is then processed by the width-2 pass under the same argument applied to the width-2 class. The same no-progress contradiction shows that the width-2 pass terminates with at most one node in the width-2 partial set.

It remains to argue that the survivor of the width-4 pass and the survivor of the width-2 pass can be taken to be the same node. Suppose the width-4 pass leaves a single partial node $n ^ { ( 4 ) }$ and the width-2 pass leaves a single partial node $n ^ { ( 2 ) } \neq n ^ { ( 4 ) }$ . The node $n ^ { ( 4 ) }$ contains a single width-4 anchored slot and has ${ \cal D } ( n ^ { ( 4 ) } ) \in \{ 0 , 2 \}$ from the multiset enumeration above. If $D ( n ^ { ( 4 ) } ) = 0$ then $n ^ { ( 4 ) }$ is fully-packed, contradicting its membership in the partial set; therefore $D ( n ^ { ( 4 ) } ) = 2 ,$ , which means ${ \tilde { S ( n ^ { ( 4 ) } ) } } = \bar { \{ 4 \} }$ , because {4, 2} already exhausts the width-2 room and $\{ 4 , 4 \}$ is fully-packed. But then room $\mathfrak { \mathrm { 1 } } _ { \mathrm { w } 2 } \big ( n ^ { ( 4 ) } \big ) = \lfloor 2 / 2 \rfloor = 1 \geq 1 , \ s \circ n ^ { ( 4 ) }$ is a width-2- eligible destination that can absorb any width-2 slot present on $\boldsymbol { n } ^ { ( 2 ) }$ . The width-2 pass would therefore have made a migration from $n ^ { ( 2 ) }$ to $n ^ { ( 4 ) }$ , contradicting the assumption that the width-2 pass left $n ^ { ( 2 ) } \neq n ^ { ( 4 ) }$ . We conclude that the survivor sets of the two passes coincide, hence at most one slotted node remains partial after the entire Compact call. □

Concluding the proof of Theorem 1. Combining Lemmas A.3 and A.4, immediately after every Compact invocation performed by Algorithm 1 under the Workload Composition Condition (9), at most one pool node and at most one slotted node are non-fully-packed, so at most two non-empty nodes in the cluster carry unused GPU capacity. Because Algorithm 1 invokes Compact in both the arrival branch and the departure branch, every event handled by the algorithm terminates in a state that satisfies this two-partial-node bound. The cluster state changes only at event boundaries, so the bound extends from every event boundary to every continuous time instant between consecutive events; that is, the two-partial-node bound holds for all $t \geq 0$

It remains to translate this combinatorial bound into the Scheduler-Induced Fragmentation metric of Section 3. Recall that $F ^ { \pi } ( t )$ denotes the proportion of partial nodes in the cluster state produced by policy $\pi ,$ that $F ^ { \star } ( t ) \geq 0$ denotes the inherent partial-node proportion attainable by any feasible packing of the active demand ${ \boldsymbol { \mathit { I } } } ( t )$ , and that $\bar { \mathrm { S I F } } ^ { \pi } ( t ) : = \bar { F } ^ { \bar { \pi } } ( t ) - \bar { F ^ { \star } } ( t )$ . Dividing the partial-node count by the cluster size � converts the two-partial-node bound into $F ^ { \pi } ( t ) \leq 2 / N$ for all $t \geq 0$ , and the non-negativity of $\mathbf { \cdots } ( t )$ then gives

$$
\operatorname { S I F } ^ { \pi } ( t ) ~ = ~ F ^ { \pi } ( t ) - F ^ { \star } ( t ) ~ \le ~ F ^ { \pi } ( t ) ~ \le ~ \frac { 2 } { N } ~ \forall t \ge 0 .
$$

This completes the proof of Theorem 1.

## B.3 Proof of Proposition 2 (Operational Complexity)

We bound the per-operator running time under the standard assumption that the slot index, the pool index, and the emptynode index are maintained as balanced search trees keyed by the relevant priority. The bookkeeping cost of incremental index maintenance is absorbed into the operator-level bounds below.

Place is �(log �). The multi-GPU branch performs three operations on the slotted-node index: a candidate query for the smallest remaining capacity above the threshold $w ^ { \star } ( g _ { i } )$ an insertion of a new slot record, and at most $w ^ { \star } ( g _ { i } ) - g _ { i } \leq 6$ filler-draw operations from the pool index. Each of these operations runs in �(log �) on a balanced search tree of � entries, and the constant number of filler draws contributes only an ${ \cal O } ( \log N )$ overhead. The 1-GPU branch performs one priority query for an open slot vacancy or a non-full pool node, again at ${ \cal O } ( \log N )$ cost. Therefore $\mathcal { T } ( \mathsf { P l a c e } ) =$ ${ \cal O } ( \log N )$ .

Remove is �(1). If � is a 1-GPU occupant, the operator detaches � from its current container in �(1) time by following the parent pointer maintained in the slot or pool index. If � is an anchor of slot $s ,$ the operator dissolves � in $O ( 1 )$ time and reinserts each of the $| F ( s ) | \le 6$ fillers by the 1-GPU rule of Place, each at �(log �) cost. Since $\vert F ( s ) \vert$ is bounded by the constant $G - g _ { a ( s ) } \leq 6 ,$ the total work of an anchor departure is dominated by a constant number of ${ \cal O } ( \log N )$ filler reinsertions, so the local handler cost is �(1) when the filler reinsertions are charged to the subsequent Compact pass that already runs in �(�). We therefore report $\mathcal { T } ( \mathrm { R e m o v e } ) = O ( 1 )$ for the local edit.

Compact is �(�). A single pass of BestFitDrain visits each partial host at most once as $n _ { \mathrm { { s r c } } }$ in the outer loop, and inside the loop it performs at most coun $\mathfrak { t } _ { C } ( n _ { \mathrm { s r c } } ) \leq G$ item migrations, each at �(log �) cost. The total work per class is therefore �(� log �), and aggregating across the three classes (pool, w4, w2) gives the same asymptotic bound. With a slightly tighter amortised analysis that charges each migration to the destination node whose room strictly decreases as a result, the log � factor collapses to $O ( N )$ total, matching the bound stated in the proposition. □

C Algorithm for Computing $\Phi ^ { \star } ( t )$ $F ^ { \star } ( t )$ The inherent fragmentation defined in Section 3.2 is

$$
F ^ { \star } ( t ) \ = \ { \frac { \Phi ^ { \star } ( t ) } { N } } , \qquad \Phi ^ { \star } ( t ) \ : = \ \operatorname* { m i n } _ { P \mathrm { f e a s i b l e } } \nu ( P ) ,
$$

where $\nu ( P )$ is the number of partial nodes in a feasible packing � of the instantaneous instance multiset $\boldsymbol { \mathcal { T } } ( t )$ and $\Phi ^ { \star } ( t )$ is the minimum partial-node count attainable across all such packings. Throughout this appendix the algorithm computes the integer-valued $\Phi ^ { \star } ( t )$ , from which $\mathbf { \nabla } F ^ { \star } ( t )$ follows by the single division by the cluster size �. Evaluating $\Phi ^ { \star } ( t )$ via a general bin-packing solver is already expensive on a single time slice and prohibitive on a trace-long evaluation. The structural analysis carried out in Section 4 for COMPASS-ABS, however, supplies enough structure to compute $\Phi ^ { \star } ( t )$ either in closed form or by a small residual ILP. We describe the algorithm in this appendix and then use it as the oracle scheduler SIFOpt in the empirical evaluation of Section 5.

C.1 Key observation: reduing to power-of-two items Each fragmentation-unfriendly instance $g \in { \mathcal { I } } ^ { \operatorname { F U F } }$ can be combined with strictly smaller active instances into a composite block whose total GPU count falls in {4, 8}, namely

$$
\begin{array} { l l } { { g = 3 : } } & { { 3 + 1 = 4 , } } \\ { { g = 5 : } } & { { 5 + 3 = 8 \mathrm { ~ o r ~ } 5 + 2 + 1 = 8 \mathrm { ~ o r ~ } 5 + 1 + 1 + 1 = 8 , } } \\ { { g = 6 : } } & { { 6 + 2 = 8 \mathrm { ~ o r ~ } 6 + 1 + 1 = 8 , } } \\ { { g = 7 : } } & { { 7 + 1 = 8 . } } \end{array}
$$

If every active FUF instance is paired into such a composite, the remaining item population consists of composites of size {4, 8}, original $\boldsymbol { \mathcal { T } } ^ { \mathrm { F F } }$ instances of size {2, 4, 8}, and any 1-GPU instances not consumed as fillers, so the multiset of item sizes is contained in {1, 2, 4, 8}. The Workload Composition Condition (9) stated in Theorem 1 is exactly the algebraic condition under which the active 1-GPU population is sufficient to complete this pairing for every FUF instance, so under that condition the reduction to a power-of-two item multiset always succeeds.

## C.2 Closed form solution under the Workload Composition Condition

When all item sizes lie in {1, 2, 4, 8}, packing into bins with capacity $G = 8$ admits an trivial optimum: a Best-Fit-Decreasing pass that processes items in the order $8  4  2 $ 1 always closes one bin per multiple of � and leaves at most one partial bin whose load equals $T ( t )$ mod �, where

$$
T ( t ) : = \sum _ { g = 1 } ^ { 8 } g \cdot n _ { g } ( t )\tag{20}
$$

denotes the total active GPU demand at time �. This is because every two width-4 items pack into a single full bin, every four width-2 items pack into a single full bin, every eight width-1 items pack into a single full bin, and any residual mixture below capacity � collapses into a single tail bin by the nested-power-of-two structure. The number of partial bins in the optimal packing is therefore

$$
\Phi ^ { \star } ( t ) ~ = ~ \mathcal { k } \big [ T ( t ) \not \equiv 0 ~ \pmod { G } \big ] ~ = ~ \left\{ \begin{array} { l l } { 0 , } & { T ( t ) ~ \mathrm { m o d } ~ G = 0 , } \\ { 1 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{21}
$$

$F ^ { \star } ( t ) = \Phi ^ { \star } ( t ) / N \in \{ 0 , 1 / N \}$ . In other words, under the condition the inherent fragmentation floor never exceeds a single node: at most one bin carries unavoidable residue $T ( t )$ mod �, and every partial node beyond that one is therefore scheduler-induced and accounted for by $\mathrm { S I F } ^ { \pi } ( t )$ . Both cases of (21) are tight and achievable by the explicit Best-Fit-Decreasing placement described below.

## C.3 Residual ILP when the Workload Composition Condition fails

When (9) fails, the active 1-GPU population is short by $\delta ( t ) : = \left( n _ { 3 } + 3 n _ { 5 } + 2 n _ { 6 } + n _ { 7 } \right) - n _ { 1 }$ relative to the filler demand of the FUF anchors. After the greedy combining stage exhausts the 1-GPU supply, exactly $\delta ( t )$ units of FUF slot vacancy remain unfilled, equivalently a small residual subset $U ( t ) \subseteq { \mathcal { I } } ^ { \mathrm { F U F } }$ of FUF instances enter the placement stage still carrying their original size in $\{ 3 , 5 , 6 , 7 \}$ . The item multiset is therefore no longer contained in {1, 2, 4, 8}, and the closed form (21) no longer applies.

In that case we restrict the optimization to $U ( t )$ : we first place all power-of-two items via Best-Fit-Decreasing as in the closed-form case, then run an integer linear program that places the residual FUF instances on top of the resulting state and minimises the total partial-node count. The size of this ILP is governed by |� (�)| rather than $n _ { \mathrm { t o t a l } } ( t )$ , and $| U ( t ) |$ is in turn bounded by $\delta ( t )$ , which is empirically a small constant on production traces because the Workload Composition Condition holds during the overwhelming majority of evaluated time slices and fails only briefly during transient mixture shifts.

## C.4 Algorithm summary

We now state the procedure explicitly. The inputs are the active instance counts $n _ { 1 } ( t ) , \ldots , n _ { 8 } ( t )$ at time �; the outputs are the optimal partial-node count $\Phi ^ { \star } ( t )$ and a witness packing $\mathbf { \nabla } ^ { P ^ { \star } } ( t )$ realising it (which the SIFOpt oracle scheduler uses as its target placement). The normalised inherent fragmentation $\mathbf { \cdots } ( t )$ used in Section 3.2 is then obtained as $F ^ { \star } ( t ) = \Phi ^ { \star } ( t ) / N .$

Logic ofthe procedure. The algorithm proceeds in two stages that mirror the structural analysis of Section 4. The combining stage (lines 3–6) absorbs every fragmentationunfriendly instance into a composite block of size {4, 8} by pairing it with strictly smaller active instances. Combining priorities are chosen so as to consume the smallest number of 1-GPU instances first when the fragmentation-unfriendly size already has a natural partner of non-unit size in the active set $( \mathbf { e . g . , 5 + 3 }$ before $5 + 1 + 1 + 1 )$ ; the priority order does not afect $\Phi ^ { \star } ( t )$ as long as combining succeeds for all FUF instances, because the resulting power-of-two item multiset has the same total � (�) and Best-Fit-Decreasing on a power-of-two item set always attains the bin-count lower bound $\lceil T ( t ) / G \rceil$ . The placement stage (line 7) then runs Best-Fit-Decreasing in the order $8  4  2  1$ and materialises this lower bound. Under the Workload Composition Condition, the closed form (21) reads of the partial node count directly from $T ( t )$ mod � without ever inspecting the placement, but the placement is still produced as a witness so that the SIFOpt oracle scheduler has a concrete target to migrate towards in its evaluation runs. When the condition fails, the residual ILP at lines 12–13 handles the small subset of FUF instances that the combining stage could not pair, and the size of that ILP is bounded by the deficit $\delta ( t ) = \left( n _ { 3 } + 3 n _ { 5 } + 2 n _ { 6 } + n _ { 7 } \right) - n _ { 1 }$ rather than by the cluster size, so the procedure remains tractable on production traces even when (9) is momentarily violated.

Complexity. The combining stage scans the instance multiset once at $O ( n _ { \mathrm { t o t a l } } ( t ) )$ cost. The Best-Fit-Decreasing placement runs in $O ( n _ { \mathrm { t o t a l } } ( t )$ log $n _ { \mathrm { t o t a l } } ( t ) )$ with a balanced search tree keyed by bin remaining capacity, and degenerates to $O ( n _ { \mathrm { t o t a l } } ( t ) )$ for the power-of-two item set because each item closes exactly one bin position. Whenever (9) holds, the closed-form return at line 8 short-circuits the placement step for the purpose of reporting $\Phi ^ { \star } ( t )$ , so the dominant cost is the linear-time combining and the optional witness materialisation. When (9) fails, the residual ILP at line 12 runs on at most $\delta ( t ) \leq | U ( t ) |$ variables and remains millisecond-scale on production traces with $\delta ( t )$ of the order of tens.

Algorithm 2 Computing $\Phi ^ { \star } ( t )$ and the SIFOpt placement   
witness; $F ^ { \star } ( t ) = \Phi ^ { \star } ( t ) / \bar { N } .$   
Input: active instance counts $n _ { g } ( t )$ for $g = 1 , \ldots , 8$   
Output: $\Phi ^ { \star } ( t ) \in \mathbb { Z } _ { \geq 0 }$ together with a witness packing $\mathbf { \nabla } ^ { P ^ { \star } } ( t )$   
(and $F ^ { \star } ( t ) = \Phi ^ { \star } ( t ) / N )$   
1: � $\textstyle \gets \sum _ { q = 1 } ^ { 8 } g \cdot n _ { g } ( t )$   
2: if $n _ { 1 } ( t ) ^ { ^ { \prime } } \substack { \sum } n _ { 3 } ( t ) + 3 n _ { 5 } ( t ) + 2 n _ { 6 } ( t ) + n _ { 7 } ( t )$ then (WCC holds)   
3: for each $i \in \mathcal { I } ^ { = 5 } ( t )$ do pair � with one ${ \cal T } ^ { = 3 }$ instance if   
available, else with ${ \cal T } ^ { = 2 } + { \cal T } ^ { = \overline { { { 1 } } } }$ , else with three ${ \cal T } ^ { = 1 }$ instances   
(form an 8-composite)   
4: for each $\dot { \mathbf { \eta } } \in T ^ { = 6 } ( t )$ do pair � with one $\scriptstyle { \cal T } ^ { = 2 }$ if available, else   
with two $\scriptstyle { \mathcal { T } } ^ { = 1 }$ (form an 8-composite)   
5: for each $i \in \mathcal { I } ^ { = 7 } ( t )$ do pair � with one ${ \cal T } ^ { = 1 }$ (form an 8-   
composite)   
6: for each $i \in \mathcal { I } ^ { = 3 } ( t )$ not consumed in line 3 do pair � with   
one $\scriptstyle { \boldsymbol { T } } ^ { = 1 }$ (form a 4-composite)   
7: run Best-Fit-Decreasing over the resulting items (sizes in   
$\{ 1 , 2 , 4 , 8 \} )$ into bins of capacity � to obtain $\mathbf { \nabla } ^ { P ^ { \star } } ( t )$   
8: return  ⊮[� mod $G \neq 0 ] , P ^ { \star } ( t ) )$   
9: else (WCCfails; small residual subset ofFUF instances cannot   
pair)   
10: greedily perform lines $_ { 3 - 6 }$ until the 1-GPU pool is ex  
hausted; collect unpaired FUF instances into �(�)   
11: run Best-Fit-Decreasing over the paired items into a ten  
tative packing $P _ { 0 }$   
12: solve the integer program   
$\operatorname* { m i n } _ { \substack { { x \in \mathbb { Z } _ { \geq 0 } ^ { | U ( t ) | \times M } } } } \nu ( P _ { 0 } \oplus { x } )$   
s.t. � places each instance in $U ( t )$ into one bin of $P _ { 0 }$   
or into a new bin   
where � is the bin index space and � counts partial bins   
13: return ${ \bigl ( } \nu ( P _ { 0 } \oplus x ^ { \star } ) , P _ { 0 } \oplus x ^ { \star } { \bigr ) }$

## D Extension to Multi-Panel NVLink Communication Domains

The COMPASS-ABS design developed in Section 4 hardwires the per-node GPU count at $G = 8$ together with the slot widths {2, 4, 8}, both of which trace back to the DGX-1 / HGX baseboard abstraction in which a single node forms one NVLink communication domain. Recent NVIDIA platforms such as NVL72 and the projected NVL576 extend the highbandwidth NVLink fabric across multiple baseboards inside a single rack-scale assembly, so that the basic intra-instance communication domain now contains $G _ { \mathrm { d o m } }$ GPUs with $G _ { \mathrm { d o m } }$ no longer equal to 8. Throughout this appendix we assume only that

$$
G _ { \mathrm { d o m } } \ = \ G _ { \boldsymbol { p } } \cdot K , \qquad G _ { \boldsymbol { p } } \ = \ 8 , \qquad K \in \mathbb { Z } _ { \ge 1 } ,\tag{22}
$$

namely that each NVLink communication domain comprises exactly � baseboards of $G _ { \ / p } = 8 \mathrm { G P U s }$ each. The original DGX-1 / HGX system corresponds to $K = 1 , \mathrm { N V L 7 2 }$ corresponds to $K = 9 ,$ , and a hypothetical NVL576 corresponds to $K = 7 2$ if treated as a single flat domain or $K = 8$ if treated as eight NVL72 sub-domains. The structural fact we exploit is that the 8-GPU baseboard survives across all of these platforms as the basic hardware building block; only the number of baseboards that sit inside a single NVLink domain varies.

We refer to a $G _ { p } = 8$ baseboard as a panel throughout this appendix. Workloads on a multi-panel NVLink domain now admit two qualitatively diferent instance classes:

• Sub-panel instances with $g _ { i } ~ \in ~ \{ 1 , . . . , 8 \}$ that fit within one panel, exactly as in the original setting; and

• Multi-panel instances with $g _ { i } \in \{ 1 6 , 2 4 , 3 2 , 4 0 , 4 8 , 5 6 ,$ $6 4 , \ldots , G _ { \mathrm { d o m } } \}$ that span multiple panels in the same NVLink domain through the rack-scale NVLink fabric.

The within-panel intra-instance constraint of Section 2.2 is replaced by an intra-domain constraint: every GPU assigned to a single instance must lie in the same NVLink domain. Subpanel instances satisfy this trivially; multi-panel instances exercise it.

## D.1 Hierarchical Anchor-Based Space

We extend COMPASS-ABS to a two-level scheduler that runs the original within-panel policy on each panel and a structurally identical panel-level analogue across panels within each NVLink domain. The panel-level state space $\Sigma _ { \mathrm { p a n e l } }$ mirrors the ABS construction of Section 4.2 verbatim under two substitutions: the panel replaces the GPU as the atomic placement unit, and the domain capacity � replaces the node capacity $G = 8 . \mathrm { { A } }$ multi-panel instance of demand $g _ { i }$ has panel demand $p _ { i } : = g _ { i } / G _ { \mathcal { P } } \in \{ 2 , 3 , . . . \}$ and is anchored in a panel-slot of width $w _ { p } ^ { \star } ( p _ { i } ) \in \{ 2 , 4 , 8 \}$ panels:

$$
\begin{array} { r } { w _ { \hat { p } } ^ { \star } ( \hat { p } ) \ = \ \left\{ \begin{array} { l l } { 2 , \quad \hat { p } = 2 , } \\ { 4 , \quad \hat { p } \in \{ 3 , 4 \} , } \\ { 8 , \quad \hat { p } \in \{ 5 , 6 , 7 , 8 \} . } \end{array} \right. } \end{array}\tag{23}
$$

Multi-panel instances with $p _ { i } \in \{ 2 , 4 , 8 \}$ are called panelfragmentation-friendly (panel-FF); those with $p _ { i } \in \{ 3 , 5 , 6 , 7 \}$ are panel-fragmentation-unfriendly (panel-FUF). A 1-panel instance, namely a $g _ { i } = G _ { p } = 8$ instance that fully occupies one baseboard, plays at the panel level the role that a 1-GPU instance plays at the GPU level inside a panel: it can either saturate a panel-vacancy in an anchored panel-FUF panelslot, or sit in a panel-pool reserved for panel-level fillers. The full cluster-wide state space is the Cartesian product across NVLink domains of $\Sigma _ { \mathrm { p a n e l } }$ , with the internal state of each non-saturated panel still drawn from the GPU-level Σ of Section 4.2. We call the resulting two-level state space the hierarchical ABS and denote it by $\Sigma _ { \mathrm { h i e r } }$

## D.2 Hierarchical Workload Composition Condition

The Workload Composition Condition (9) of Theorem 1 extends to the two-level setting by imposing one condition per level. Let $n _ { g } ( t )$ denote the count ofactive sub-panel instances of GPU demand $g \in \{ 1 , . . . , 8 \}$ at time $t ,$ and let $n _ { \boldsymbol { p } } ^ { ( \mathrm { p a n e l } ) } ( t )$ denote the count of active multi-panel instances of panel demand $p \in \{ 2 , 3 , \ldots \}$ . The GPU-level condition stays identical to (9), namely

$$
\begin{array} { r } { n _ { 1 } ( t ) ~ \ge ~ n _ { 3 } ( t ) + 3 n _ { 5 } ( t ) + 2 n _ { 6 } ( t ) + n _ { 7 } ( t ) , } \end{array}\tag{24}
$$

and the panel-level condition takes the structurally identical form

$$
\begin{array} { r } { n _ { 8 } ( t ) ~ \geq ~ n _ { 3 } ^ { \mathrm { ( p a n e l ) } } \left( t \right) + 3 n _ { 5 } ^ { \mathrm { ( p a n e l ) } } \left( t \right) + 2 n _ { 6 } ^ { \mathrm { ( p a n e l ) } } \left( t \right) + n _ { 7 } ^ { \mathrm { ( p a n e l ) } } \left( t \right) , } \end{array}\tag{25}
$$

where the left-hand side is the active count of $g _ { i } \ = \ 8$ instances, treated as 1-panel fillers at the panel level. Equation (25) is obtained from (24) by the substitution 1-GPU ↦→ 1-panel together with the analogue of the slot-vacancy form (18) at panel granularity, and uses the same coeficient pattern {1, 3, 2, 1} for the same algebraic reason as in Lemma A.1.

## D.3 Hierarchical COMPASS algorithm

The extended scheduler adopts two structurally identical Place/Remove/Compact pipelines, one per level, dispatched according to the demand class of the arriving or departing instance:

• Sub-panel events $( g _ { i } \leq 8 )$ : handled by the GPU-level COMPASS-ABS ofSection 4.3 acting on the host panel.

• Multi-panel events $( g _ { i } \in \{ 1 6 , 2 4 , . . . , G _ { \mathrm { d o m } } - G _ { p } \} )$ : handled by the panel-level analogue acting on the host NVLink domain, with $g _ { i } = 8$ instances serving as panel-fillers.

• Domain-filling events $( g _ { i } = G _ { \mathrm { d o m } } )$ : occupy an entire NVLink domain at once and are treated as a width-� degenerate panel-slot.

Algorithm 1 is invoked at both levels and each Compact pass runs over the BestFitDrain classes for its level. At the GPU level the three classes remain {pool, $\mathbf { W } _ { 4 } , \mathbf { W } _ { 2 } \mathbf { \Phi } \mathbf { \Phi } $ on capacity $G _ { p } \ = \ 8 ;$ at the panel level they become {panel-pool, panel-w<sub>4</sub>, panel- $\left. \mathbf { - w } _ { 2 } \right\}$ on capacity �. Because the two levels act on disjoint populations of items, the two compaction passes do not interfere and run independently after each event.

## D.4 Hierarchical compactness guarantee

The compactness guarantee of Theorem 1 lifts to the hierarchical setting in a structurally identical form.

Theorem 2 (Hierarchical Compactness). Assume that the GPU-level Workload Composition Condition (24) and the panellevel Workload Composition Condition (25) both hold at every $t \geq 0$ . Then in every cluster state produced by the hierarchical COMPASS-ABS scheduler:

1. across the entire cluster, at most one panel-resident slotted GPU-region and at most one panel-resident GPUpool region are non-fully-packed; and

2. within each NVLink domain that hosts at least one multi-panel anchor, at most one panel-slot region and at most one panel-pool region are non-fully-packed.

Consequently, after normalising the partial-region count at each level by the corresponding domain size, the GPU-level and panel-level scheduler-induced fragmentation measures satisfy

$$
\begin{array} { r l } { \mathrm { S I F } _ { \mathrm { G P U } } ^ { \pi } ( t ) \ \leq \ \frac { 2 } { N _ { \mathrm { p } } } } & { ( c l u s t e r – w i d e , N _ { \mathrm { p } } \ p a n e l s ) , } \\ { \mathrm { S I F } _ { \mathrm { p a n e l } } ^ { \pi } ( t ) \ \leq \ \frac { 2 } { K } } & { ( p e r N V L i n k \ d o m a i n \ o f K \ p a n e l s ) , } \end{array}\tag{26}
$$

for all $t \geq 0 .$

Proofsketch. The argument reduces to two independent invocations of the proof of Theorem 1 given in Appendix B.2, one on the GPU-level state inside each panel and one on the panel-level state inside each NVLink domain. Both invocations are valid because Lemmas A.1–A.4 of Appendix B.2 depend only on (i) the power-of-two slot widths {2, 4, 8} being a valid sub-decomposition of the local capacity, (ii) the anchor-filler invariant under the friendly/unfriendly dichotomy of (23), and (iii) the class structure {pool, $\mathbf { W } _ { 4 } , \mathbf { W } _ { 2 } \mathbf { \Phi } \mathbf { \Phi } $ on which BestFitDrain operates, all of which are preserved under the substitution � → � and 1-GPU → 1-panel at the panel level. □

## D.5 Generality with respect to the domain panel count

Theorem 2 is stated for an arbitrary positive integer �, without assuming that � is a power of two. The panel-level BestFitDrain operates on class set {panel-pool, panel-w<sub>4</sub>, panel-w<sub>2</sub>} exactly as the GPU-level pass operates on its own class set, and the combinatorial argument of Lemma A.4 does not depend on � taking any particular value beyond requiring that the slot widths {2, 4, 8} remain a valid subdecomposition of the capacity. For the NVL72 instantiation with $K = 9$ , any panel-level layout decomposes into at most one width-8 panel-slot together with one residual panel acting as a panel-pool, which is the structural reason the 9-panel layout fits cleanly into the existing framework despite $K = 9$ not being a power of two: the lone residual panel coincides exactly with the panel-pool slot that the framework already requires. The same hierarchical scheduler and the same compactness bound apply to every multi-panel platform whose NVLink domain comprises an integer number of8-GPU baseboards, including the original DGX-1 / HGX system $( K = 1$ in which the panel level is trivially empty), DGX-2 and SuperPod intermediate systems $\left( K = 2 \right)$ , the NVL72 system $( K = 9 )$ , and the projected NVL576 system $( K = 7 2$ flat, or � = 8 across NVL72 sub-domains as a third tier). In summary, as long as the NVLink domain contains an integer multiple of $G _ { \boldsymbol { p } } = 8 \mathrm { G P U s } ,$ , the hierarchical COMPASS-ABS scheduler remains valid and the SIF bound (26) continues to hold.