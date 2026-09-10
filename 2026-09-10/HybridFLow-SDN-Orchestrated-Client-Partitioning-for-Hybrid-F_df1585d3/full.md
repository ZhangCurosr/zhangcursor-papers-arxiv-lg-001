# HybridFLow: SDN-Orchestrated Client Partitioning for Hybrid Federated Learning

Osama Abu Hamdan<sup>∗</sup>, Rabin Pandey<sup>‡</sup>, Hao Che<sup>∗</sup>, Engin Arslan<sup>†</sup>, Md Arifuzzaman<sup>‡</sup>

<sup>∗</sup>University of Texas at Arlington

oma8085@mavs.uta.edu, hche@cse.uta.edu

<sup>†</sup>Meta Platforms, Inc.

enginarslan@meta.com

<sup>‡</sup>Missouri University of Science and Technology

rp5dm@mst.edu, marifuzzaman@mst.edu

Abstract—Cross-silo Federated Learning (FL) enables geographically distributed institutions to collaboratively train machine learning models without sharing raw data. In wide-area deployments, however, communication delays often dominate round completion time and exacerbate the straggler effect. Hybrid FL addresses this challenge by combining synchronous and asynchronous client participation, but effective partitioning requires visibility into network conditions such as shared bottlenecks, link utilization, and path contention that individual clients cannot observe. We present HybridFLow, a closed-loop SDN-driven orchestration framework that integrates network-layer intelligence directly into hybrid FL. Leveraging the SDN controller’s global topology view, HybridFLow generates calibrated per-client communication-time estimates before each training round and uses them to partition clients into synchronous and asynchronous groups while balancing round latency and update staleness. After each round, measured communication times are fed back to the controller to continuously refine future predictions. Experimental results across multiple network topologies show that HybridFLow reaches 80% target accuracy 33–40% faster than SmartFLow and reduces average round duration by 30–40 seconds, while FedAsync fails to reach the target accuracy under non-IID data distributions.

Index Terms—Hybrid Federated Learning, Software-Defined Networking, Network Orchestration, Client Partitioning

## I. INTRODUCTION AND MOTIVATION

Cross-silo Federated Learning (FL) is an emerging class of distributed AI workload in which geographically separated institutions collaboratively train a shared model without exchanging raw data [1], [2]. Unlike datacenter-local training, cross-silo FL runs over wide-area networks with heterogeneous bandwidth, latency, and congestion [3], [4], making communication management the central systems challenge. In synchronous FL, the server must wait for every participating client before aggregating model updates, so training throughput is gated by the slowest network path. As Fig. 1 illustrates, this straggler effect inflates per-round completion time by over 47% compared to optimized routing [5], with the penalty growing with topology scale and link heterogeneity.

Fully asynchronous FL [6] removes the synchronization barrier by integrating updates as they arrive, but under non-IID data distributions fast clients dominate aggregation while slow clients contribute stale gradients, degrading model convergence. As Fig. 2 shows, asynchronous FL fails to reach target accuracy under data heterogeneity [7], [8]. Hybrid FL addresses this trade-off by assigning a synchronous subset of clients per round while allowing the remainder to contribute asynchronously [8], preserving convergence quality without waiting on the slowest paths.

![](images/8601cf0378158a2cbf5d8606f679955876271558e27ec7fd23abcd55aaaaf39e.jpg)  
Fig. 1: Straggler effect in synchronous FL. Round completion time is capped by the slowest client, forcing all faster participants to idle.

Realizing the benefits of hybrid FL depends on correctly identifying which clients are likely to hold up the synchronous phase before each round begins. In cross-silo settings, this is fundamentally a network question: a client with capable hardware but a congested WAN path behaves as a straggler regardless of its local compute speed, and that congestion is invisible to the FL endpoints. End-host measurements expose only local symptoms; they cannot reveal whether candidate paths share bottleneck links with other clients, or how background traffic will evolve over the duration of a round. The partitioning decision therefore calls for a signal that no individual client can generate on its own.

Software-Defined Networking provides exactly that signal. An SDN controller maintains a topology-wide view of link utilization, latency, loss, and path contention across the entire WAN [9]. This global control-plane visibility is structurally distinct from anything observable at the compute layer: it captures shared bottlenecks, cross-client path interference, and network-wide congestion state that endpoint measurements cannot reach. Integrating this visibility into hybrid FL allows network-wide conditions to inform sync/async partitioning decisions.

![](images/b3e6e3435d4fdff696f5cb49fc50e5fa7c2135cc8d16fd0f0e3de21fca3e07e3.jpg)  
Fig. 2: Convergence under non-IID data. Asynchronous FL eliminates the straggler bottleneck but fails to reach target accuracy under data heterogeneity, motivating a hybrid approach.

HybridFLow is a closed-loop SDN orchestration framework for Hybrid Federated Learning in cross-silo deployments. It leverages real-time network telemetry from the SDN controller to generate calibrated per-client communication-time estimates and incorporates them into a cost-minimization algorithm that balances synchronous round latency against the staleness risk of asynchronous updates. By integrating topology-wide network intelligence directly into sync/async partitioning decisions, HybridFLow brings the network control plane into the Hybrid FL orchestration loop. The framework establishes a round-gated feedback mechanism in which communicationtime estimates guide path assignment and client partitioning before each round, while measured transfer times continuously refine future estimates, closing the control loop without reactive mid-round rerouting.

This paper makes three contributions: (1) a closed-loop SDN-FL orchestration architecture that translates network telemetry into per-round sync/async training-mode decisions; (2) a network-aware hybrid partitioning algorithm that balances synchronous round latency against asynchronous staleness risk using calibrated communication-time estimates from the SDN control plane; and (3) an emulated cross-silo evaluation demonstrating reduced round time and faster time-totarget accuracy under non-IID data distributions.

## II. RELATED WORK

Communication efficiency in FL has been extensively studied through gradient compression, quantization, and selective parameter updates [10]–[13]. These approaches reduce communication volume and improve scalability, but operate primarily at the model and update levels. As a result, they do not directly address network-level factors such as path contention, routing inefficiencies, or time-varying congestion, which frequently dominate completion time in wide-area cross-silo FL deployments.

The synchronization bottleneck has motivated asynchronous and hybrid training paradigms. FedAsync [6] demonstrated that a server can integrate updates without waiting for all clients, while AsyncFedED [7] introduced distance-based weighting to reduce the impact of stale updates. Hybrid FL further balances convergence quality and training throughput by maintaining a synchronous subset of clients while allowing slower participants to contribute asynchronously [8]. Existing hybrid approaches typically derive sync/async decisions from computelayer information such as local training time, dataset size, gradient characteristics, or historical round latency. While these signals capture important aspects of client heterogeneity, they provide limited visibility into network-wide conditions such as shared bottlenecks, path contention, and evolving WAN congestion. In cross-silo environments, where communication time often dominates round completion time, incorporating network-layer information into the partitioning process represents a natural extension of existing hybrid FL strategies.

The intersection of SDN and FL has recently attracted increasing attention. Ma et al. [14] surveyed the challenges and opportunities of SDN-assisted FL systems, while Mahmod et al. [15] demonstrated SDN-based bandwidth provisioning to mitigate link saturation during FL communication rounds. Mahmoud et al. [16] further explored SDN-guided client selection as a mechanism for improving model quality. These studies highlight the value of network visibility and control for improving FL performance, but focus primarily on resource allocation and participant selection. Our prior work, Smart-FLow [5], leveraged ONOS-based telemetry and topologyaware routing to reduce synchronization time in cross-silo FL by up to 47% relative to static routing. However, SmartFLow retained a fully synchronous training model and therefore remained susceptible to network-induced stragglers.

HybridFLow extends prior SDN-assisted FL approaches by using SDN telemetry not only to optimize communication paths, but also to inform application-level training orchestration. Specifically, network-derived communication-time estimates are incorporated directly into the hybrid sync/async partitioning process, enabling the network control plane to participate in per-round training-mode decisions. To the best of our knowledge, prior SDN-assisted FL studies have not used topology-wide network telemetry as a control signal for hybrid synchronous/asynchronous client partitioning in crosssilo federated learning.

## III. SYSTEM OVERVIEW

HybridFLow is organized across two cooperating but independent layers, as illustrated in Fig. 3: an SDN Controller layer implemented in Java on the ONOS platform [17], and a Federated Learning layer implemented in Python using the Flower framework [18]. A lightweight ZeroMQ messaging interface connects the two layers. The SDN layer handle network monitoring, path scoring, and per-client communication time estimation; the FL layer handles client partitioning, training orchestration, and model aggregation. Neither layer accesses the internal state of the other; the messaging protocol serves as the sole interface.

The remainder of this section presents the system in three parts. First, the SDN layer establishes the network-level infrastructure that produces per-client communication time estimates. Second, the FL layer consumes those estimates to partition clients into synchronous and asynchronous groups and to orchestrate hybrid training. Third, the data exchange protocol that connects the two layers closes the feedback loop.

![](images/0cc8f02bec5dddb6b50840175389e9231b20bb33dcbcdd1232465822de64116d.jpg)  
Fig. 3: HybridFLow architecture. The SDN controller produces calibrated per-client communication time estimates via the Flow Scheduler and Progress Tracker; the FL server consumes them to partition clients and orchestrate hybrid aggregation. One Estimate Request–Measurement Report exchange per round closes the feedback loop.

## A. SDN Layer

The SDN layer builds on the ONOS controller and retains the Stats Parser, Client Store, and Link Store components introduced in SmartFLow [5]. These components continuously collect OpenFlow statistics from network devices, maintaining per-link estimates of available capacity $\begin{array} { r } { F _ { l } ^ { \mathrm { f r e e } } = B _ { l } - U _ { l } } \end{array}$ , oneway latency $\lambda _ { l } .$ , and packet loss probability $p _ { l } .$ , all smoothed by an Exponentially Weighted Moving Average (EWMA) over the three most recent measurements [19]. The Flow Scheduler and Progress Tracker from SmartFLow are retained in Hybrid-FLow and extended to support the hybrid FL workflow.

1) The Path Assignment Problem: Assigning network paths to FL clients is a non-trivial decision because: (i) multiple candidate paths may connect each client to the server, with widely varying bandwidth, latency, and loss characteristics; (ii) paths share links, so assigning one client to a high-quality path reduces the available capacity for subsequent clients; and (iii) the quality of any assignment depends on real-time link conditions that change between rounds. The Flow Scheduler takes the current network state as input and produces, for every client in both the server-to-client (S2C) and client-toserver (C2S) directions, a single assigned path together with a calibrated transfer time estimate.

To quantify path quality, the Flow Scheduler evaluates each candidate path $P = ( l _ { 1 } , \ldots , l _ { m } )$ (a sequence of directed links excluding edge links) on three complementary metrics.

Projected Fair Share. The estimated per-client bandwidth on link l, assuming the incoming client joins the $n _ { l }$ flows already traversing it, is:

$$
\mathrm { F a i r S h a r e } _ { l } ^ { \mathrm { p r o j } } = \frac { F _ { l } ^ { \mathrm { f r e e } } } { n _ { l } + 1 } .\tag{1}
$$

The path-level fair share equals the bottleneck link’s share:

$$
\mathrm { F a i r S h a r e } _ { P } ^ { \mathrm { p r o j } } = \operatorname* { m i n } _ { l \in P } \mathrm { F a i r S h a r e } _ { l } ^ { \mathrm { p r o j } } .\tag{2}
$$

$n _ { l }$ is initialized from the per-link active-flow count reported by OpenFlow statistics and incremented by the Flow Scheduler each time a path using link l is assigned during the current scheduling sweep, so it reflects both existing background traffic load and the cumulative capacity consumed by earlier assignments within the same round.

Effective RTT. The round-trip time of path $P$ sums forward and reverse latencies across every link:

$$
\mathrm { R T T } _ { P } = \sum _ { l \in P } \bigl ( \lambda _ { l } + \lambda _ { \bar { l } } \bigr ) ,\tag{3}
$$

where <sup>¯</sup>l denotes the reverse direction of link l. This formulation accurately models TCP behavior: data packets traverse the forward path while acknowledgment segments return along <sup>¯</sup>l, so the per-link contribution to round-trip time is precisely $\lambda _ { l } + \lambda _ { \bar { l } } \ [ 2 0 ]$

End-to-End Packet Loss. Under the assumption of independent link failures:

$$
p _ { P } = 1 - \prod _ { l \in P } ( 1 - p _ { l } ) .\tag{4}
$$

Effective Score. Combining these metrics into a Mathisvariant throughput estimate [21]:

$$
\mathrm { E f f S c o r e } ( P ) = \frac { \mathrm { F a i r S h a r e } _ { P } ^ { \mathrm { p r o j } } } { \mathrm { R T T } _ { P } \cdot \sqrt { p _ { P } + \varepsilon } } ,\tag{5}
$$

where ε is a small positive value that prevents division by zero when the measured packet loss on a path is negligible. A higher score indicates a path that delivers more throughput per unit time. Dividing the model size D (in megabits) by EffScore(P) yields a raw transfer time estimate in seconds.

While the classical Mathis formulation models steady-state TCP behavior under random drop conditions, we employ EffScore(P) strictly as a comparative, network-layer utility proxy to rank candidate paths prior to transmission. Any real-world deviations from steady-state dynamics are captured and corrected downstream by the Progress Tracker’s EWMA feedback loop, which adjusts raw estimates toward observed per-client transfer times across successive rounds.

The Flow Scheduler interface is implementation-agnostic: any algorithm that maps the current network state to per-client path assignments and time estimates can serve as the scheduler. For instance, constraint programming formulations [5] and reinforcement learning-based solvers are both viable alternatives. In HybridFLow, the current implementation uses a greedy allocation that, for each direction $d \in \{ \mathrm { S } 2 \mathrm { C } , \mathrm { C } 2 \mathrm { S } \}$ sorts clients in ascending order of their best available adjusted score so that the most constrained clients receive first choice of paths, assigns each client to its highest-scoring path, and increments the active-flow count on every link of the assigned path to reflect the capacity consumed by that assignment.

Paths are assigned once per round, at the start of each training round; no dynamic reassignment occurs during the transfer phase. Because the Hybrid Partitioner divides clients into synchronous and asynchronous groups at the beginning of each round (Section III-B), fixing path assignments at the same point is a natural simplification that keeps the scheduling calculation tractable while ensuring consistency between routing and partitioning decisions.

2) Correction Factors and Adjusted Scoring: Raw effective scores are throughput-proxy values, not precise throughput predictions. To compensate for systematic bias, the Progress Tracker maintains a multiplicative correction factor $\mathrm { C F } ( c , d , P )$ for each (client, direction, path) triple, updated via EWMA after every round:

$$
r _ { c , d , P } ^ { ( t ) } = \frac { T _ { c , d } ^ { \mathrm { r e a l } , ( t ) } } { T _ { c , d } ^ { \mathrm { r a w } , ( t ) } } , \quad T _ { c , d } ^ { \mathrm { r a w } } = \frac { D } { \mathrm { E f f S c o r e } ( P ) } ,\tag{6}
$$

$$
\mathbf { C F } _ { c , d , P } ^ { ( t ) } = \boldsymbol { \beta } \cdot \boldsymbol { r } ^ { ( t ) } + \left( 1 - \boldsymbol { \beta } \right) \cdot \mathbf { C F } ^ { ( t - 1 ) } ,\tag{7}
$$

where $\beta$ is the EWMA smoothing parameter that controls how rapidly the correction factor adapts to new observations. When per-client history is unavailable, the system falls back to aggregate correction data; if no historical data exists at all, the correction factor defaults to 1.0. The adjusted score and final time estimate are:

$$
\operatorname { A d j S c o r e } ( P , c , d ) = { \frac { \operatorname { E f f S c o r e } ( P ) } { \operatorname { C F } ( c , d , P ) } } ,\tag{8}
$$

$$
\hat { T } _ { c , d } = \frac { D } { \mathrm { A d j S c o r e } ( P _ { c , d } ^ { * } , c , d ) } .\tag{9}
$$

Equation (9) yields a calibrated communication time estimate in seconds that improves with each round as the correction factor converges.

## B. FL Layer

The FL layer receives per-client communication time estimates from the SDN layer and uses them to partition clients, orchestrate training, and aggregate model updates.

1) Client State and Completion Time Estimation: The client state record tracks per-client state across rounds, including communication estimates and EWMA-smoothed computation times. The estimated total completion time for client c before round t is:

$$
T _ { c } = \hat { T } _ { c , \mathrm { S 2 C } } + \hat { T } _ { c , \mathrm { C 2 S } } + \bar { t } _ { c } ^ { \mathrm { c o m p } } ,\tag{10}
$$

where $\hat { T } _ { c , \mathrm { S 2 C } }$ and $\hat { T } _ { c , \mathrm { C 2 S } }$ are the SDN-provided communication time estimates, and $\bar { t } _ { c } ^ { \mathrm { c o m p } }$ is the EWMA of client c’s observed local computation times.

2) Gradient Importance Tracking: The gradient importance tracker maintains an EMA-smoothed estimate of each client’s gradient importance. After each round, it records the L2 norm of the parameter difference $\lVert g _ { c } ^ { \mathrm { l a s t } } \rVert = \lVert w _ { c } - w _ { \mathrm { g l o b a l } } \rVert _ { 2 }$ for every client c that contributed an update. The global average is then updated:

$$
\bar { g } _ { \mathrm { n e w } } = \delta \cdot \bar { g } _ { \mathrm { o l d } } + ( 1 - \delta ) \cdot \frac { 1 } { | R | } \sum _ { c \in R } \| g _ { c } ^ { \mathrm { l a s t } } \| ,\tag{11}
$$

where δ is a smoothing parameter that controls how quickly the global average adapts to recent gradient magnitudes, and R is the set of all clients that contributed updates in the current round (both synchronous and asynchronous). The per-client importance weight is:

$$
w _ { c } ^ { \mathrm { i m p } } = \frac { \lVert g _ { c } ^ { \mathrm { l a s t } } \rVert } { \bar { g } } .\tag{12}
$$

In the first round, $\| g _ { c } ^ { \mathrm { l a s t } } \| ~ = ~ \bar { g } ~ = ~ 1 . 0 .$ , so all clients begin with uniform importance $w _ { c } ^ { \mathrm { i m p } } ~ = ~ 1 . 0$ . Clients with aboveaverage gradient norms $( w _ { c } ^ { \mathrm { i m p } } > 1 )$ contribute more to model improvement; assigning them to the asynchronous group carries a higher staleness penalty, biasing the partitioner toward keeping them synchronous.

3) Network-Aware Client Partitioning: The Hybrid Partitioner solves the core optimization problem of Hybrid-FLow: given estimated completion times $\{ T _ { c } \}$ and importance weights $\{ w _ { c } ^ { \mathrm { i m p } } \}$ }, partition clients into a synchronous set S and an asynchronous set ${ \mathcal { A } } = C \backslash { \mathcal { S } } $ , minimizing a cost function that balances round latency against staleness risk. The cost function combines a sync latency term and an async penalty term, each normalized to [0, 1].

The sync latency term measures how much the synchronous round duration exceeds its minimum possible value:

$$
J _ { \mathrm { s y n c } } ( S ) = \frac { \operatorname* { m a x } _ { c \in S } T _ { c } - T _ { \mathrm { m i n } } } { T _ { \mathrm { m a x } } - T _ { \mathrm { m i n } } } ,\tag{13}
$$

where $T _ { \mathrm { m i n } } = T _ { c _ { k _ { \mathrm { m i n } } } }$ is the sync latency at the minimum sync set size $k _ { \mathrm { m i n } } ,$ , and $\mathbf { \mathop { T } _ { m a x } ^ { m u } } = { T } _ { c _ { n } }$ is the slowest client’s estimated time. This min-max normalization ensures the term spans the full [0, 1] range across all candidate split points.

The async penalty term captures the staleness risk of the asynchronous group. For each asynchronous client c, gradient importance $w _ { c } ^ { \mathrm { i m p } }$ is weighted by a quadratic staleness proxy, penalizing clients whose completion time far exceeds the synchronous boundary:

Algorithm 1 Hybrid Partitioner: Greedy Sweep   
Require: Clients C, times $\{ T _ { c } \}$ , weights $\{ w _ { c } ^ { \mathrm { i m p } } \} , \lambda , k _ { \mathrm { m i n } }$   
1: Sort C by $T _ { c } \colon T _ { c _ { 1 } } \leq \cdot \cdot \cdot \leq T _ { c _ { n } }$   
2: $T _ { \mathrm { m i n } } \quad \gets \quad T _ { c _ { k _ { \mathrm { m i n } } } } ; \quad T _ { \mathrm { m a x } } \quad \gets \quad T _ { c _ { n } } ; \quad P _ { \mathrm { m a x } } \quad \gets$   
AsyncPenalty $\left( k _ { \operatorname* { m i n } } \right)$   
3: $J ^ { * } \gets \infty ; \ k ^ { * } \gets k _ { \mathrm { m i n } }$   
4: for $k = k _ { \mathrm { m i n } }$ to n do   
5: $J _ { s } \gets \left( T _ { c _ { k } } - T _ { \operatorname* { m i n } } \right) / \left( T _ { \operatorname* { m a x } } - T _ { \operatorname* { m i n } } \right)$   
6: $J _ { a } \gets \mathbf { A } :$ syncPenalty(k) / P<sub>max</sub>   
7: $J \gets ( 1 - \lambda ) \cdot J _ { s } + \lambda \cdot J _ { a }$   
8: if $J < J ^ { * }$ then $J ^ { * } \gets J ; ~ k ^ { * } \gets k$   
9: end if   
10: end for   
11: return $S ^ { * } = \{ c _ { 1 } , \ldots , c _ { k ^ { * } } \} , \mathcal { A } ^ { * } = \{ c _ { k ^ { * } + 1 } , \ldots , c _ { n } \}$

$$
J _ { \mathrm { a s y n c } } ( S ) = \frac { \displaystyle \sum _ { c \in \mathcal { A } } w _ { c } ^ { \mathrm { i m p } } \cdot \left( \frac { T _ { c } } { \operatorname* { m a x } _ { s \in \mathcal { S } } T _ { s } } \right) ^ { 2 } } { P _ { \mathrm { m a x } } } ,\tag{14}
$$

where $P _ { \mathrm { m a x } }$ is the raw penalty at the split with the largest async set, normalizing the term to [0, 1]. The normalization constants $T _ { \mathrm { m i n } } , \ T _ { \mathrm { m a x } }$ , and $P _ { \mathrm { m a x } }$ are computed once at the beginning of each sweep and remain fixed for its duration.

The total cost combines the two terms as a convex combination:

$$
J ( S ) = ( 1 - \lambda ) J _ { \mathrm { s y n c } } ( S ) + \lambda J _ { \mathrm { a s y n c } } ( S ) ,\tag{15}
$$

subject to $| { \cal S } | \ge M _ { \mathrm { s y n c } } ^ { \mathrm { m i n } }$ . The trade-off parameter $\lambda \in [ 0 , 1 ]$ governs the balance: $\lambda \ : = \ : 0$ minimizes sync latency alone (fastest rounds), $\lambda = 1$ minimizes the async penalty alone (safest convergence), and the default $\lambda = 0 . 5$ weights both objectives equally. Because both constituent terms are normalized, λ has scale-independent semantics.

Because all clients are sorted by $T _ { c }$ and the optimal split is a contiguous prefix of this ordering, the minimizer can be found by a greedy sweep in $O ( n$ log n) time, as formalized in Algorithm 1.

4) Hybrid Aggregation: Once the Hybrid Partitioner determines $s$ and ${ \mathcal { A } } ,$ the hybrid server dispatches all asynchronous clients to background training immediately, then blocks on the completion of all synchronous clients. Completed asynchronous updates arrive at varying times and are deposited into a thread-safe buffer. Aggregation then proceeds in two stages.

Stage 1: Synchronous Aggregation. The server aggregates synchronous results using the configured aggregation strategy. HybridFLow is compatible with any aggregation algorithm (e.g., FedAvg, FedProx); the choice of strategy is orthogonal to the partitioning and scheduling mechanisms described here.

Stage 2: Asynchronous Integration. HybridFLow integrates buffered asynchronous updates following the stalenessaware aggregation framework introduced by FedAsync [6].

Each buffered update carries a staleness measure $\tau _ { c } = t _ { \mathrm { c u r r e n t } } -$ $t _ { \mathrm { s u b m i s s i o n } , c }$ (in rounds). The per-client effective mixing coefficient combines a configurable base mixing rate with a staleness decay:

$$
\alpha _ { \mathrm { e f f } , c } = \alpha _ { \mathrm { b a s e } } \cdot f _ { \mathrm { s t a l e n e s s } } ( \tau _ { c } ) ,\tag{16}
$$

where $\alpha _ { \mathrm { b a s e } } \in [ 0 , 1 ]$ controls the maximum influence any single asynchronous update can exert. The staleness decay function $f _ { \mathrm { s t a l e n e s s } }$ can be configured as polynomial $( \tau { + } 1 ) ^ { - a }$ , exponential $e ^ { - a \tau }$ , or constant 1.0; the default is polynomial with $a = 0 . 5 ,$ matching the FedAsync formulation [6].

The buffered updates are first combined into a stalenessweighted average:

$$
w _ { \mathrm { a v g } } = \frac { \displaystyle \sum _ { c \in \cal { A } } \alpha _ { \mathrm { e f f } , c } w _ { c } } { \displaystyle \sum _ { c \in \cal { A } } \alpha _ { \mathrm { e f f } , c } } .\tag{17}
$$

This average is then merged with the synchronous model $w _ { \mathrm { s y n c } }$ through a collective coefficient that separates data fraction from staleness quality:

$$
\alpha _ { \mathrm { c o l l } } = \frac { N _ { A } } { N _ { A } + N _ { S } } \ \times \ \frac { \overline { { \alpha _ { \mathrm { e f f } } } } } { \alpha _ { \mathrm { b a s e } } } ,\tag{18}
$$

$$
w _ { \mathrm { n e w } } = ( 1 - \alpha _ { \mathrm { c o l l } } ) \cdot w _ { \mathrm { s y n c } } + \alpha _ { \mathrm { c o l l } } \cdot w _ { \mathrm { a v g } } .\tag{19}
$$

The first factor reflects the asynchronous cohort’s share of the total training data; the second factor $( \in [ 0 , 1 ] )$ scales that share proportionally to the mean staleness quality of the batch. Their product always lies in [0, 1], requiring no clamping heuristics.

5) Async Dispatch and Re-Dispatch: Each asynchronous client is submitted to a thread-pool executor with a per-client queue depth of one: if a client already has an in-flight task, the dispatcher skips it. When an asynchronous client completes, the system deposits its update into the buffer and makes a re-dispatch decision based on round timing. If the client finished within the same round it was dispatched, it is not redispatched, because it may belong in the synchronous group in the next round’s partitioning. If the client finished after the server has already advanced to a new round, the system redispatches it with the latest global model to keep it productive until the next partitioning decision.

## C. FL-SDN Data Exchange Protocol

HybridFLow coordinates its two layers through a proactive, round-gated data exchange protocol. The protocol comprises two message types exchanged once per round via ZeroMQ, ensuring that all path assignments and time estimates are finalized before any data transfer occurs.

Estimate Request. At the start of round t, the FL server issues an estimate request to the SDN controller containing the current round number and the list of available client identifiers. The SDN controller invokes the Flow Scheduler to run path assignment and returns a map $\{ c _ { i } \ \mapsto \ [ \hat { T } _ { c _ { i } , \mathrm { S 2 C } } , \ \hat { T } _ { c _ { i } , \mathrm { C 2 S } } ] \}$ of adjusted communication time estimates for all clients. The FL server applies these estimates to compute per-client completion times via Equation (10).

Algorithm 2 HybridFLow: Per-Round Execution   
Require: Clients $C ,$ round $t , \lambda , \alpha _ { \mathrm { b a s e } } , M _ { \mathrm { s y n c } } ^ { \mathrm { m i n } }$   
Phase 1: Estimation and Partitioning   
1: $\hat { \mathbf { T } }  \mathrm { Q u e r y S D N } ( t , C )$ ▷ Estimate Request   
2: for all $c \in C$ do   
3: $T _ { c } \gets \hat { T } _ { c , \mathrm { S 2 C } } + \hat { T } _ { c , \mathrm { C 2 S } } + \bar { t } _ { c } ^ { \mathrm { c o m p } }$   
4: end for   
5: S, A ← HybridPartition $( C , \{ T _ { c } \} , \{ w _ { c } ^ { \mathrm { i m p } } \} , \lambda , M _ { \mathrm { s y n c } } ^ { \mathrm { m i n } } )$   
Phase 2: Training   
6: Dispatch $\mathcal { A }$ to background training   
7: $R s \gets \mathrm { T r a i n S y n c } ( S )$   
Phase 3: Two-Stage Aggregation   
8: $w _ { \mathrm { s y n c } }  \mathrm { A g g r e g a t e } ( R _ { S } )$ ▷ Eq. via configured strategy   
9: $w _ { \mathrm { g l o b a l } } ^ { ( \dot { t } ) } $ IntegrateAsync $( w _ { \mathrm { s y n c } } ,$ Buffer) ▷ Eqs. 16–19   
Phase 4: Update and Report   
10: Update $\{ w _ { c } ^ { \mathrm { i m p } } \}$ from $R$ and $w _ { \mathrm { g l o b a l } } ^ { ( t ) }$   
11: ReportMeasurements $( t , C )$ ▷ Measurement Report   
12: return $w _ { \mathrm { g l o b a l } } ^ { ( t ) }$

Measurement Report. After the round concludes, the FL server sends a map $\{ c _ { i } \ \mapsto \ [ T _ { c _ { i } , \mathrm { S 2 C } } ^ { \mathrm { r e a l } } , \ T _ { c _ { i } , \mathrm { C 2 S } } ^ { \mathrm { r e a l } } ] \}$ of measured communication times to the SDN controller. The Progress Tracker uses these measurements to update the EWMA correction factors per Equation (7), closing the feedback loop and improving estimation accuracy in subsequent rounds.

## D. End-to-End Round Execution

Algorithm 2 summarizes the complete per-round execution of HybridFLow, integrating the SDN and FL layers through the data exchange protocol described in Section III-C.

A key property of Algorithm 2 is that both path assignment and partitioning complete before training begins, eliminating the measurement instability associated with mid-round switching. The round execution unfolds in four phases.

Phase 1: Estimation and Partitioning. The FL server issues an estimate request to the SDN controller, which runs the Flow Scheduler and returns calibrated per-client communication time estimates. Using these estimates, each client’s total completion time $T _ { c }$ is computed per Equation (10). The Hybrid Partitioner then minimizes the cost function $J ( \boldsymbol { S } )$ (Equation 15) via a greedy sweep over clients sorted by $T _ { c } ,$ producing the optimal sync set S and async set A.

Phase 2: Training. The server dispatches all clients in A concurrently to background training and blocks on the completion of all clients in S. Asynchronous updates complete at varying times and are deposited into the thread-safe async buffer as they arrive.

Phase 3: Two-Stage Aggregation. Stage 1 aggregates the synchronous results $R _ { \cal { S } }$ to produce the intermediate model $w _ { \mathrm { s y n c } }$ . Stage 2 computes a staleness-weighted average of all buffered async updates per Equation (17) and merges it into $w _ { \mathrm { s y n c } }$ with the collective coefficient of Equation (18), yielding the updated global model $w _ { \mathrm { g l o b a l } } ^ { ( t ) }$ . Clients that do not complete within the current round remain in-flight; their expected residual time naturally biases them toward asynchronous classification in the next round’s partitioning.

Phase 4: Update and Report. The FL server updates gradient importance weights $\{ { \bar { w } } _ { c } ^ { \mathrm { i m p } } \}$ from the parameter differences observed across all contributing clients relative to $w _ { \mathrm { g l o b a l } } ^ { ( t ) } .$ . It then reports real per-client communication times to the SDN controller, which uses them to update the EWMA correction factors per Equation (7), closing the feedback loop and improving path-score calibration in subsequent rounds.

## IV. EXPERIMENTAL RESULTS

The preceding sections established how HybridFLow leverages SDN-provided communication time estimates to partition clients into synchronous and asynchronous groups before each round, balancing round latency against staleness risk. The experiments in this section evaluate the resulting orchestration system along four axes: (1) per-round efficiency and synchronization blocking reduction, as the primary management outcome; (2) adaptivity of the sync/async partition to perround network conditions; (3) accuracy of the SDN-layer closed-loop predictor that drives partitioning decisions; and (4) convergence speed, as evidence that the orchestration actions preserve ML utility. Together, these experiments test whether the SDN-FL design choices described in Section III-B translate into measurable management and training improvements under realistic network heterogeneity.

## A. Experimental Setup

All experiments use FLEET [22] as the emulation framework and draw network topologies from the Gabriel collection in TopologyHub [23]. Three topologies of increasing scale define the evaluation scenarios: E1 with 15 switches and 15 FL clients, E2 with 25 switches and 25 clients, and E3 with 35 switches and 35 clients. All links operate at 100 Mbps, and background traffic follows a Poisson arrival process that injects time-varying congestion across the network. Every client runs on identical compute hardware, so that observed performance differences arise exclusively from network heterogeneity. The hardware platform is a single CPU-only server equipped with 128 GB RAM and an AMD EPYC 9124 16-core, 32-thread processor.

All experiments use a fixed dataset and model to isolate the impact of network conditions on federated learning performance. CIFAR-10 is partitioned using PathologicalPartitioner, which assigns 7 of the 10 label classes to each client, creating a non-IID setting in which asynchronous FL is particularly susceptible to gradient staleness and contribution imbalance. MobileNetV3-Large [24] is selected to generate realistic communication workloads; its 14.2 MB model size makes wide-area communication a significant contributor to round completion time, allowing the evaluation to capture the effects of network heterogeneity. As topology size increases, the training dataset is distributed across a larger number of clients (approximately 3,000 samples per client in E1, 2,000 in E2, and 1,500 in E3), while a fixed 10,000-sample server-side evaluation set is used across all experiments.

![](images/501d9f49d5fdcb0e07f93407ab32d54f900c4f3b7c3ec48e8147c67979e3006e.jpg)  
(a) SmartFLow

![](images/5e0e0ad789fd6a0d82cee1f42320f8498b232b87a69e3a9b522a3212e74fe90b.jpg)  
(b) HybridFLow  
Fig. 4: Per-round stacked time breakdown for topology E1. Each bar decomposes one client’s round into S2C transfer, local training, and C2S transfer; (a) shows SmartFLow, (b) shows HybridFLow. Stars in (b) mark clients reclassified as asynchronous.

Emulation allows the same topology, traffic, and compute conditions to be replayed across baselines, isolating the effect of network-aware partitioning while preserving realistic control-plane interaction through ONOS and OpenFlow. All three baselines run the same implementation stack (ONOS, Flower, ZeroMQ) and share identical hardware and background traffic conditions, so observed differences arise exclusively from the partitioning strategy.

![](images/538ecfb7aac3b1447a658ecafc0e712e9b9578d1a40b7b1c9530688c5f3e4d07.jpg)  
Fig. 5: Average round time for SmartFLow and HybridFLow across E1, E2, and E3. HybridFLow reduces synchronization blocking by removing network-induced tail clients from the synchronous phase, cutting average round time by 30-40 seconds in every topology.

Three training configurations are compared. Smart-FLow uses SDN-optimized routing with fully synchronous training, while FedAsync follows the FedAsync protocol [6]. HybridFLow combines synchronous and asynchronous participation through network-aware client partitioning. For consistency, HybridFLow adopts the FedAvgM aggregation strategy used by SmartFLow (server\_momentum = 0.9) and the exponential staleness function used by FedAsync (staleness\_alpha=0.25, mixing\_alpha=0.75). Finally, HybridFLow uses $\lambda ~ = ~ 0 . 3 5$ in Equation (15), selected from preliminary experiments on E1. Across $\lambda \in$ {0.1, 0.3, 0.5, 0.8}, performance varied smoothly, with $\lambda \ =$

![](images/3a5caab68341dc4a9e51b35bb5b3b5e0c4fa87616ca40bbfaf0f86de192bdd18.jpg)  
Fig. 6: Distribution of per-round asynchronous-to-total client ratio for HybridFLow across E1, E2, and E3. Each violin encodes the full empirical distribution; a horizontal bar marks the median.

0.35 providing a balanced trade-off between round latency and update staleness.

Training terminates when server evaluation accuracy reaches 80%, making time-to-target-accuracy the primary metric. This criterion directly reflects how quickly each method produces a usable model under identical network conditions. Any method that fails to reach the target within the experiment budget is reported at its final achieved accuracy.

## B. Results

The central outcome of HybridFLow is the reduction of synchronization blocking during federated learning rounds. Fig. 4 shows the mechanism: network transfer time is substantial for straggler clients, whose bars are visibly elongated by S2C and C2S segments. SmartFLow must wait each round for these slowest clients, whereas HybridFLow reclassifies them as asynchronous (starred bars) and closes the round without them. Their updates are not discarded; they are incorporated asynchronously in subsequent rounds, subject to a staleness penalty, preserving their contribution while removing their latency from the synchronous phase. Per-client communication cost remains comparable between the two methods, indicating that the improvement arises from reduced synchronization delay rather than reduced communication volume.

![](images/51fd1b7cd71264ca5ee9abf4a1c9c129dd4c64435e0d8f14394e7f6f8512dc12.jpg)  
(a) Topology E1

![](images/c41dec8c107531991abeb5713233e7fcf3aa663bdf2990c1d2b56a7d55a6a368.jpg)  
(b) Topology E2

![](images/6cacf534b5e87e3d07b0dce24084b32de31fbca45e1f40e14e76749d38ebd216.jpg)  
(c) Topology E3

Fig. 7: Predicted vs. actual communication and computing time for HybridFLow across E1, E2, and E3. Close tracking validates the closed-loop control mechanism: measurements reported after each round calibrate the SDN controller’s future estimates, keeping prediction error low enough for reliable per-round orchestration.  
![](images/0d8fff3734597ff779d38f5b4273308b0d8a0e599aa354b6020b6d9b471e848a.jpg)  
(a) Topology E1

![](images/0d33d90e9b9e03b6592149f1f284c5d8e1121c29cb7687a8f517b407f10f7983.jpg)  
(b) Topology E2

![](images/c362a2d6d46f1197d6945f56f545eb0bb9df16f4877ee9001f5c1830cde7cecb.jpg)  
(c) Topology E3  
Fig. 8: Server evaluation accuracy versus wall-clock time for topologies E1, E2, and E3. Each subplot compares SmartFLow, FedAsync, and HybridFLow. HybridFLow reaches the 80% target first in all three topologies, while FedAsync fails to reach the target in any topology.

Fig. 5 confirms this at the aggregate level: average round time drops by 30–40 seconds for HybridFLow relative to SmartFLow in every topology, a reduction that is consistent across E1, E2, and E3. The box-plot spread for HybridFLow is narrower in E1 and E2, reflecting more predictable round durations when stragglers are excluded; in E3, richer path diversity at 35-client scale introduces greater round-to-round variability that offsets the variance reduction.

Fig. 6 shows that the SDN control plane does not apply a static policy: no violin collapses to a point across any topology. Distribution shape tracks network heterogeneity— skewed toward low async fractions in E1, shifted higher in E2 as greater path diversity increases reclassification frequency, and approximately symmetric in E3 where 35-client topology variability balances low- and high-reclassification rounds. The correlation between distribution shape and topology scale confirms that the partitioner continuously adjusts the synchronous boundary in response to per-round network conditions.

The closed-loop control mechanism is validated by Fig. 7, which overlays predicted and actual communication and computing times across all rounds and topologies. Communication MAE is 7.2 s (E1), 6.2 s (E2), and 9.2 s (E3); computing MAE stays below 1.5 s in all three. The close tracking validates the

EWMA correction mechanism (Equation 7), which calibrates the SDN controller’s estimates using real measurements reported after each round. Reliable prediction is a prerequisite for the round-time gains: misclassifying clients—retaining stragglers synchronously or ejecting fast ones prematurely— would negate the benefit.

The management actions preserve convergence quality. Fig. 8 validates this across all three topologies: HybridFLow reaches the 80% accuracy target 33–40% faster than Smart-FLow in every configuration, a consistent advantage from the 15-client E1 topology through the 35-client E3 topology. FedAsync fails to reach the 80% target in any topology under non-IID data, confirming that unconstrained asynchronous aggregation accumulates stale gradients that prevent convergence when data distributions are heterogeneous.

Taken together, the SDN-FL orchestration pipeline is coherent: accurate closed-loop time prediction (Fig. 7) enables reliable per-round partitioning (Fig. 6), which reduces synchronization blocking (Fig. 5) without sacrificing convergence quality (Fig. 8). The net effect is a 33–40% reduction in timeto-target-accuracy compared to SmartFLow, achieved without the convergence degradation that affects FedAsync under non-IID data distributions.

## C. Limitations of the Study

The evaluation is conducted in an emulated cross-silo environment rather than a production WAN deployment. While this enables analysis of network-aware partitioning with realistic SDN control-plane interactions, validation at larger scales and across operational WANs remains future work. HybridFLow assumes the availability of topology-wide SDN telemetry, which is appropriate for enterprise, campus, research or private networks but may not apply where network visibility is limited or routing decisions are controlled by external providers. Finally, the current design assumes trustworthy telemetry and well-behaved participants; the effects of telemetry inaccuracies, controller failures, and adversarial clients remain open directions for future work.

## V. CONCLUSION

HybridFLow is a closed-loop SDN orchestration framework for Hybrid Federated Learning in cross-silo environments. Unlike existing hybrid FL approaches that rely primarily on compute-layer signals, HybridFLow incorporates topologywide network telemetry into per-round sync/async partitioning decisions through an SDN control plane. By combining communication-time prediction, network-aware partitioning, and feedback-driven calibration within a unified orchestration loop, HybridFLow enables training decisions that account for both communication latency and update staleness. Experiments across emulated cross-silo topologies demonstrate that network-aware orchestration reduces synchronization blocking while preserving convergence quality under non-IID data distributions. Relative to SmartFLow, HybridFLow reaches the 80% accuracy target 33–40% faster and reduces average round duration by 30–40 seconds across all evaluated topologies. These results demonstrate that topology-aware network telemetry can serve as an effective control signal for hybrid federated learning orchestration, enabling communicationaware training decisions in heterogeneous wide-area environments.

## ACKNOWLEDGMENT

The work in this study was supported in part by NSF grant 2427408 and 2451376. The authors used LLM tools (ChatGPT and Claude) for text formatting. All scientific content including experimental design, analysis, results, figures, and citations is the authors’ own.

## REFERENCES

[1] B. McMahan, E. Moore, D. Ramage et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, A. Singh and J. Zhu, Eds., vol. 54. PMLR, 20–22 Apr 2017, pp. 1273–1282.

[2] K. A. Bonawitz, H. Eichner, W. Grieskamp et al., “Towards federated learning at scale: System design,” in Proceedings of the Second Conference on Machine Learning and Systems, SysML 2019, Stanford, CA, USA, March 31 - April 2, 2019. mlsys.org, 2019.

[3] P. Kairouz, H. B. McMahan, B. Avent et al., “Advances and open problems in federated learning,” Found. Trends Mach. Learn., vol. 14, no. 1–2, p. 1–210, Jun. 2021. [Online]. Available: https://doi.org/10.1561/2200000083

[4] L. Lyu, H. Yu, X. Ma et al., “Privacy and robustness in federated learning: Attacks and defenses,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 7, pp. 8726–8746, 2024.

[5] O. A. Hamdan, H. Che, E. Arslan, and M. Arifuzzaman, “Smartflow: A communication-efficient sdn framework for cross-silo federated learning,” in 2026 IEEE 23rd Consumer Communications & Networking Conference (CCNC), 2026, pp. 1–6.

[6] C. Xie, S. Koyejo, and I. Gupta, “Asynchronous federated optimization,” 2020. [Online]. Available: https://arxiv.org/abs/1903.03934

[7] Q. Wang, Q. Yang, S. He et al., “Asyncfeded: Asynchronous federated learning with euclidean distance based adaptive weight aggregation,” 2022. [Online]. Available: https://arxiv.org/abs/2205.13797

[8] E. T. Mart´ınez Beltran, M. Q. P ´ erez, P. M. S. S ´ anchez ´ et al., “Decentralized federated learning: Fundamentals, state of the art, frameworks, trends, and challenges,” IEEE Communications Surveys & Tutorials, vol. 25, no. 4, pp. 2983–3013, 2023.

[9] B. A. A. Nunes, M. Mendonca, X.-N. Nguyen et al., “A survey of software-defined networking: Past, present, and future of programmable networks,” IEEE Communications Surveys & Tutorials, vol. 16, no. 3, pp. 1617–1634, 2014.

[10] A. Malekijoo, M. J. Fadaeieslam, H. Malekijou et al., “Fedzip: A compression framework for communication-efficient federated learning,” 2021. [Online]. Available: https://arxiv.org/abs/2102.01593

[11] Y. He, H.-P. Wang, M. Zenk, and M. Fritz, “Cossgd: Communicationefficient federated learning with a simple cosine-based quantization,” 2022. [Online]. Available: https://arxiv.org/abs/2012.08241

[12] N. Bouacida, J. Hou, H. Zang, and X. Liu, “Adaptive federated dropout: Improving communication efficiency and generalization for federated learning,” in IEEE INFOCOM 2021 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), 2021, pp. 1–6.

[13] J. Konecnˇ y, H. B. McMahan, F. X. Yu ´ et al., “Federated learning: Strategies for improving communication efficiency,” 2017. [Online]. Available: https://arxiv.org/abs/1610.05492

[14] X. Ma, L. Liao, Z. Li, R. X. Lai, and M. Zhang, “Applying federated learning in software-defined networks: A survey,” Symmetry, vol. 14, no. 2, 2022. [Online]. Available: https://www.mdpi.com/2073- 8994/14/2/195

[15] A. Mahmod, G. Caliciuri, P. Pace, and A. Iera, “Improving the quality of federated learning processes via software defined networking,” in Proceedings of the 1st International Workshop on Networked AI Systems, ser. NetAISys ’23. Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3597062.3597281

[16] A. Mahmod, P. Pace, and A. Iera, “Sdn-assisted client selection to enhance the quality of federated learning processes,” in 2024 IEEE Wireless Communications and Networking Conference (WCNC), 2024, pp. 1–6.

[17] P. Berde, M. Gerola, J. Hart et al., “Onos: towards an open, distributed sdn os,” in Proceedings of the Third Workshop on Hot Topics in Software Defined Networking, ser. HotSDN ’14. New York, NY, USA: Association for Computing Machinery, 2014, p. 1–6. [Online]. Available: https://doi.org/10.1145/2620728.2620744

[18] D. J. Beutel, T. Topal, A. Mathur et al., “Flower: A friendly federated learning research framework,” 2022. [Online]. Available: https://arxiv.org/abs/2007.14390

[19] C. Metter, V. Burger, Z. Hu et al., “Towards an active probing extension for the onos sdn controller,” in 2018 28th International Telecommunication Networks and Applications Conference (ITNAC), 2018, pp. 1–8.

[20] M. Allman, V. Paxson, and E. Blanton, “Tcp congestion control,” IETF, RFC 5681, Sep. 2009. [Online]. Available: https://datatracker.ietf.org/doc/html/rfc5681

[21] J. Padhye, V. Firoiu, D. Towsley, and J. Kurose, “Modeling tcp reno performance: a simple model and its empirical validation,” IEEE/ACM Transactions on Networking, vol. 8, no. 2, pp. 133–145, 2000.

[22] O. A. Hamdan, H. Che, E. Arslan, and M. Arifuzzaman, “Fleet: A federated learning emulation and evaluation testbed for holistic research,” in 2026 IEEE 23rd Consumer Communications & Networking Conference (CCNC), 2026, pp. 1–6.

[23] E. K. C¸ etinkaya, M. J. Alenazi, Y. Cheng et al., “On the fitness of geographic graph generators for modelling physical level topologies,” in 2013 5th International Congress on Ultra Modern Telecommunications and Control Systems and Workshops (ICUMT), 2013, pp. 38–45.

[24] A. Howard, M. Sandler, B. Chen et al., “Searching for mobilenetv3,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1314–1324.