# Toward Composable Network Digital Twins: A Subgraph-Based Latency Prediction Study

Shenjia Ding   
School of Computing Science   
University of Glasgow   
Glasgow, UK   
2788155d@student.gla.ac.uk   
David Flynn   
School of Engineering   
University of Glasgow   
Glasgow, UK   
david.flynn@glasgow.ac.uk   
Paul Harvey   
School of Computing Science   
University of Glasgow   
Glasgow, UK   
paul.harvey@glasgow.ac.uk

Abstract—Modern networks must support changing topologies, configurations, and performance objectives, motivating fast and reliable performance estimation. Network digital twins (NDTs) enable what-if analysis for performance estimation in such network scenarios, however, existing machine learning-based NDT approaches often rely on entire topology representations, which are inherently monolithic and lack reusability under topological or traffic changes in the network.

This paper introduces a composable NDT approach that decomposes networks into subgraphs represented by reusable unit twins that capture subgraph structure, configuration and traffic behaviours. A lightweight composer aggregates unit twin combinations to create NDTs that predict per-route end-toend latency through an overall topology. Evaluation across controlled synthetic topologies and diverse traffic scenarios, realworld Topology Zoo topologies, and a public NDT challenge dataset demonstrates that the composable NDTs achieve high in-distribution accuracy while remaining stable under out-ofdistribution scenarios. Comparison with monolithic full topology NDTs demonstrates that our composable approach achieves reusability , while achieving comparable or superior accuracy.

Index Terms—Network Digital Twin, Network Testing, Graph Neural Networks

## I. INTRODUCTION

Modern communication networks are becoming increasingly complex and dynamic [1]. The growth of programmable networks, enabled by software-defined network, network function virtualization, and cloud-edge infrastructure, has enabled the flexibility to adapt network configurations to changing operational demands, while simultaneously increasing the complexity of network management, making testing more relevant and challenging [2]. Before deploying new network configurations, it is necessary to understand how the network will behave under different conditions. However, testing on real networks can be costly, risky and difficult to repeat [3]. Current simulation tools, such as ns-3 [4], can provide detailed network behaviour, but large-scale simulation requires significant time and computational effort [5].

Network Digital Twins (NDTs) [6] have emerged as a promising approach to support network testing and optimisation, allowing network behaviour to be analysed without directly disturbing the real system. In this context, Graph Neural Networks (GNNs) [7] have been used as a key enabler to construct data-driven NDTs for various tasks, such as latency prediction and traffic modelling. GNN-based NDTs can approximate expensive simulation and provide computationally efficient prediction for network testing [8].

Despite this potential, many GNN-based NDTs are still designed in a monolithic manner. In such designs, a single GNN model represents an entire network topology, requiring training data that spans all configurations and operational conditions for the full target network topology. Such models achieve strong performance on scenarios seen during training, however, their generalisation to unseen conditions is often limited, meaning they struggle to accommodate network modifications, which constrains flexibility, adaptability, and reusability in network testing, see Section II-A2.

To address this limitation, this paper explores a composable view of NDTs (Section III). Instead of treating an NDT as a monolithic network predictor, we consider an NDT to be a composition of reusable subgraph predictors known as unit twins (UTs), which can be trained, stored, and recomposed to create subsequent NDTs. Target network topologies are decomposed based on known subgraph shapes, translated into UTs, and then composed to create an NDT of the target network. Should the existing set of UTs be insufficient for the task, new UT families can be added without requiring full model retraining.

We evaluate the effectiveness of our approach across synthetic baseline topologies, topology modification, several traffic delay models, a GNNet NDT challenge dataset [9], and real-world topologies from Topology Zoo [10] (Section V). Results demonstrate that NDTs composed from UTs can effectively predict latency under both seen and unseen topologies and traffic conditions, whereas a centralised GNN-based NDT performs well for seen scenarios, but is less able to generalise to unseen scenarios without retraining.

To the best of our knowledge, this is the first work to study NDTs composed of GNN-based UTs for route-level latency prediction. Our focus is not composability as a general digital twin concept, but how subgraph predictors (UTs) can be reused for route-level NDT prediction under network modifications to enable more flexible NDTs.

Our contributions are:

• introducing a composable view of NDTs, where subgraph encoders (UTs) are treated as reusable components for

latency prediction.

• formulating subgraph encoder reuse as a graph composition problem via GNN-based latency predictors.

• demonstrating that a GNN-based subgraph composition approach can improve robustness under unseen topology composition and topology modifications compared with centralised GNN models.

## II. BACKGROUND AND MOTIVATION

## A. Network Digital Twin

Network digital twins (NDTs) [6] provide data-driven or simulation-based representations of communication networks for performance monitoring, testing, and control. A key role of an NDT is to support repeated what-if evaluation before changes are deployed to the network by estimating key performance metrics [11], [12], including quality of service (QoS), under varying topology, routing, and traffic conditions [13]. Latency prediction is especially important as high network latency directly affects user-perceived service quality and the operation of delay-sensitive applications, such as cloud services, video conferencing, industrial control, and edge computing [14].

1) NDT Representation: Simulation-based NDTs [15] can provide detailed performance measurements and remain important when high-fidelity behaviour is required. However, repeated simulation over many candidate configurations, topologies, or traffic behaviours can be computationally expensive, which limits their use for rapid network planning and configuration exploration [16].

Machine learning (ML)-based NDTs provide a complementary approach [6]. Wei et al. [17] use deep learning and reinforcement learning to build data-driven network predictions for routing optimisation, aiming to improve simulation speed and accuracy. For QoS prediction, Saravanan et al. [18] use learning models and ensemble methods to predict network delay and optimise service performance. Once trained, ML models can act as computationally efficient NDT predictors, estimating network performance without rerunning a full simulator for every candidate configuration [19]. This makes them attractive for rapid what-if analysis, configuration exploration, and performance-aware network planning.

2) Generalizability of Monolithic NDTs: Many existing ML-based NDTs adopt a monolithic design, training a single model to represent an entire network topology [20], [21]. While accurate under previously seen conditions, such models struggle when topology or traffic distributions change substantially, as the learned mapping from graph structure to performance no longer holds, requiring retraining on new data spanning the modified topology or context to recover accuracy [22]. This incurs significant data and computational costs, limiting flexibility, adaptability, and reusability.

## B. Digital Twin Composition

In digital twins (DTs) more broadly, composition has been proposed to enable flexibility, adaptability, and reuse. Instead of building a single monolithic twin for every target system, a composable DT treats the system as an assembly of smaller models, services, or tools that can be connected through defined interfaces [23]. This view is especially relevant for complex systems whose components may be added, removed, upgraded, or recombined over time. Existing works can be grouped into two categories:

1) Co-Simulation: approaches connect multiple simulators or behavioural models so that different parts of a networked system can exchange data and jointly evaluate system-level behaviour [24]. This enables DTs to combine physical models, controllers, and external data sources [25], often via standards, such as the Functional Mock-up Interface (FMI) [26]. Co-simulation-based DT architectures have been proposed to connect heterogeneous models and support coupled behaviour in complex systems [23]. In communication networks, CAVIAR [27] provides a modular co-simulation methodology for 6G scenarios by combining communication simulation, 3D scenes, and AI modules within a shared DT workflow.

A key limitation of this approach is computational efficiency. The overall runtime depends not only on the percomponent speed, but also on the cost of coordinating their interactions, where repeated data exchange, time synchronisation, and coordination with external simulators or processes are required [28]. Consequently, even if some components are efficient, the coupled system can be slowed by communication overhead or by the slowest component in the workflow. This limits the suitability of co-simulation for NDT scenarios that require repeated evaluation.

2) Modular Architectures: Several works propose architectural and design principles for composable DTs, noting that DT development should support reuse of effort, accelerated time to results, and generalisability [29]. Composite DT architectures describe how multiple DTs can be interconnected to form a collaborative digital ecosystem [30], while DT connectivity studies show that larger DTs may be organised as connected or aggregated structures built from multiple interacting twins [31]. However, most modular DT architectures remain at the level of system design, service composition, governance, or platform integration. While these concerns are essential considerations for DTs, they do not define how the internal units of a DT should be represented or reused, nor do they provide concrete implementations for reusable predictive units in NDTs.

3) Summary: Overall, existing DT-composition research shows a clear movement away from purely monolithic twins. However, most works treat composition as an integration, orchestration, privacy, or interoperability problem. Less attention has been given to composability as a modelling principle for NDTs.

## C. Graph Neural Networks

Graph neural networks (GNNs) are deep learning models that learn representations from graph-structured data by propagating and aggregating information across neighbouring nodes and edges.

1) GNN-based Network Representation: GNNs are well suited to communication network modelling because network topology, routing paths, link attributes, and traffic conditions can naturally be represented as graph features [32].

For NDTs, GNNs can represent network state and structural dependencies under changing routing, traffic, and configuration conditions. Prior work has applied GNNs to network performance modelling, routing-aware prediction, and NDT-supported network management [19], [20], [21]. For network performance prediction, GNNs have been used to estimate metrics, such as delay, loss, throughput, and flowlevel QoS [32]. Route-level latency prediction is concerned with a source–destination path whose delay depends on route structure, link conditions, queuing state, and neighbouring traffic interactions [33].

Many GNN-based network prediction models are trained as full-topology or scenario-level models. They map a complete graph, route context, and configuration features to a performance prediction, and are effective when deployment topologies are close to previously seen examples. When a network is modified, extended, or recomposed, the prediction accuracy often decreases, requiring retraining, as discussed in Section II-A2.

2) Distributed, Subgraph-Based, and Modular GNNs: Beyond full-topology prediction, research on GNNs has studied graph partitioning, subgraph processing, and compositional representations. As full-graph GNN training can become expensive on large graphs, distributed GNN systems can be used to address this by partitioning the graph across machines or devices. For example, DistGNN [34] uses graph partitioning and communication reduction for scalable full-batch GNN training. ByteGNN [35] introduces GNN-aware partitioning and scheduling to reduce communication overhead in distributed training. Compositional graph models also view graph representations as being built from parts or substructures [36]. Subgraph-level prediction methods, such as SLAPP [37], show that subgraphs can support larger-system prediction, although not for communication network latency. Similarly, graph pooling and subgraph representation methods suggest that subgraph-level prediction can benefit from structured intermediate representations [38].

Our work is inspired by these partitioned and compositional approaches, but uses them for a different purpose: graph decomposition is used for reusable unit twins in NDT construction, as opposed to distributed GNN training.

## III. METHODOLOGY: COMPOSABLE NDTS

To address the challenge of flexible NDTs, we now present a composable NDT approach for route-level latency prediction under network modifications.

## A. Concept in Nutshell

As shown in Figure 1, we use pre-defined network topology shapes to decompose a target network topology into subgraphs and then train GNN-based representations of these subgraphs to create reusable unit twins (UTs), see Section III-C. We then compose these UTs into NDTs via a composer, see Section III-F.

Given a route through the target network, as well as the configuration, the latency is predicted by decomposing the route into per-UT segments, predicting the latency of each segment, and combining to obtain the end-to-end latency.

## B. Problem Formulation

We represent a communication network as a directed graph $G = ( V , E )$ where V is the set of nodes and $E$ is the set of directed links. Each node $v \in V$ has a feature vector $x _ { v }$ that includes queue-related attributes, such as packet buffer queue size. Each edge $e = ( u , v ) \in E$ has a feature vector $a _ { e }$ for link information, indicating bandwidth, capacity, utilisation, traffic load, or route-membership indicators. A queried route $P$ is represented as an ordered sequence of nodes $( v _ { 0 } , v _ { 1 } , \ldots , v _ { H } )$ where we assume that each pair of consecutive network nodes is connected by a single link in this work.

End-to-end latency prediction is route-aware: the ordered path $P ,$ its directed edges, and route descriptors (see Section III-E2) are part of the input. The objective is to learn a predictor ${ \hat { y } } _ { P } ~ = ~ F ( G , P )$ for the end-to-end latency of a route through $G$ but with a composable structure. Instead of training a monolithic predictor over the full graph $G ,$ we train predictors based on subgraphs, predict the per-subgraph latency, and compose the values to get the end-to-end latency.

## C. Unit Twin and UT Library

A unit twin (UT) is the basic reusable component of the proposed composable NDT. It represents a subgraph network region and provides a per-route latency-prediction function through the region. In this work, each UT is realised by a pre-defined network topology family and a route-aware GNN encoder. For a subgraph $G _ { i } = ( V _ { i } , E _ { i } )$ and a route segment $P _ { i } \subseteq G _ { i }$ , the UT produces

$$
( z _ { i } , \hat { y } _ { i } ^ { s u b g r a p h } ) = f _ { \theta _ { i } } ( G _ { i } , P _ { i } ) ,\tag{1}
$$

where $z _ { i }$ is a learned representation of the subgraph-level route segment and $\hat { y } _ { i } ^ { s u b g r a p h }$ is the predicted latency contribution. The pair $( z _ { i } , \hat { y } _ { i } ^ { s u b g r a p h } )$ is the standardized UT output. This allows different UTs to be composed in a uniform way: the route-level composer only consumes the ordered sequence of UT outputs, rather than the internal details of each encoder.

A UT is associated with a network topology family, rather than a single fixed graph instance. Currently, the UT library contains five families: path-like, star-like, ring-like, diamondlike, and small-block regions. They represent structurally distinct local connectivity patterns, including linear forwarding, hub-based branching, cyclic connectivity, and multiple alternative paths. These families are selected as representative subgraphs for controlled composition experiments and for the decomposed regions used in our evaluation. They are not intended to cover all possible network subgraphs, but can be extended.

![](images/a748ecd2db6103730813fa3302b71d903d717e4f68918126bb1aad2975d08776.jpg)  
Fig. 1. Composable NDT workflow. Blue boxes show the process to create an NDT. Orange boxes show the process to use an NDT for latency prediction.

## D. GNN Realisation of Unit Twins

For a family $c ,$ the corresponding route-aware GNN encoder is trained on subgraphs whose node counts lie within a predefined range. This allows processing of variable-size subgraphs from the same family, rather than only one fixed node configuration. In this work, we use a range of three to eight; however, this is configurable.

Node and edge features encode subgraph state, link attributes, and route descriptors (see Section III-E2). Route descriptors are included to make the encoder route-aware: as the same subgraph may be reused for different route segments, the route must be represented as part of the input rather than being fixed by the topology. Rather than training a dedicated GNN per matched subgraph instance, we train one shared route-aware encoder for each UT family.

This design is inspired by distributed and subgraph-based GNN training, where large graphs are divided into smaller regions so that local message passing and representation learning can be performed more efficiently [34]. Our use of decomposition is different: distributed GNN methods mainly partition graphs to scale the training of one global model, while we use subgraph families to define reusable predictive units for NDT construction.

For each UT family $c ,$ the encoder $f _ { \theta _ { c } }$ maps a subgraph and route segment to a standardised output as in Eq. 1. The encoder consists of message passing, route-aware readout, and two output heads:

$$
h _ { i } = \mathrm { R e a d o u t } _ { P _ { i } } \big ( \mathrm { G N N } _ { \theta _ { c _ { i } } } ( G _ { i } , P _ { i } ) \big ) ,\tag{2}
$$

$$
z _ { i } = g _ { \theta _ { c _ { i } } } ^ { e m b } ( h _ { i } ) , \qquad \hat { y } _ { i } ^ { s u b g r a p h } = g _ { \theta _ { c _ { i } } } ^ { p r e d } ( h _ { i } ) .\tag{3}
$$

Message passing updates node and edge representations within $G _ { i } ,$ while the route-aware readout aggregates the representations associated with the route segment $P _ { i }$ . The embedding head produces $z _ { i }$ for composition, and the prediction head estimates the subgraph latency contribution. As all UT encoders expose the same output format, heterogeneous subgraphs can be reused and recomposed for end-to-end latency prediction. The architecture of the UT encoder that produces $z _ { k }$ and $\hat { y } _ { k } ^ { s u b g r a p h }$ is described in Section IV-A2a.

As message passing is performed on subgraphs and trained encoders can be reused across matched UT instances, our approach reduces the need to train a new monolithic GNN for every target topology.

## E. Topology Decomposition and Route Decomposition

We now describe how to take a target network topology and represent it as UTs, as well as how to predict the end-to-end latency for a route through the network. The approach has two connected pipelines, as shown in Figure 1.

1) Topology Decomposition: is the process of decomposing a target topology into subgraphs based on the pre-defined topology families described above, where each family has a pre-defined GNN encoder (Figure 1, blue boxes). Topology mapping produces edge-disjoint subgraphs (Section V-A discusses topology decomposition methods) $G _ { 1 } , \ldots , G _ { M }$ , where each physical link belongs to exactly one subgraph:

$$
E _ { i } \cap E _ { j } = \varnothing , \quad i \neq j , \quad \quad E = \bigcup _ { i = 1 } ^ { M } E _ { i } .\tag{4}
$$

The vertex sets may overlap, allowing adjacent UTs to share boundary nodes. If a subgraph cannot be matched to an existing UT family, direct reuse is not possible for that region and the library must be extended with a new family. Across the topologies evaluated in this paper in Section IV-B, the five listed UT families cover all partitioned subgraphs after topology decomposition.

2) Route Decomposition: Having generated the UTs and composed them to create an NDT, we can now estimate the latency for a given end-to-end route through the target topology, as shown by the orange boxes in Figure 1. A route is first decomposed into an ordered sequence of subgraph route segments, based on the UTs in the NDT. Given a queried route $P = ( v _ { 0 } , v _ { 1 } , \dots , v _ { H } )$ , the route is rewritten, using the decomposed subgraphs, as an ordered sequence of subgraph route segments:

$$
P = P _ { 1 } \oplus P _ { 2 } \oplus \cdots \oplus P _ { K } .\tag{5}
$$

Each segment $P _ { k }$ is the maximal consecutive part of the route whose edges belong to the same partitioned subgraph. It is processed by the UT assigned to that subgraph. The route is represented as an ordered UT sequence:

$$
( { \mathcal { T } } _ { s _ { 1 } } , P _ { 1 } ) , ( { \mathcal { T } } _ { s _ { 2 } } , P _ { 2 } ) , \ldots , ( { \mathcal { T } } _ { s _ { K } } , P _ { K } ) ,\tag{6}
$$

where $\mathcal { T } _ { s _ { k } }$ is the UT used for segment $P _ { k }$

## F. Route-Level Composer

After route decomposition, the route-level composer combines UT outputs to predict the end-to-end latency of the route.

As end-to-end latency $\hat { y } _ { P }$ may include non-additive effects caused by boundary interactions, bottlenecks, or congestion coupling between neighbouring route segments, we therefore add a learned route-level correction term $\Delta _ { \phi } .$ , which represents the residual adjustment added to the summed subgraph latency prediction:

$$
\hat { y } _ { P } = \sum _ { k = 1 } ^ { K } \hat { y } _ { k } ^ { s u b g r a p h } + \Delta _ { \phi } ( z _ { 1 } , z _ { 2 } , \ldots , z _ { K } ) ,\tag{7}
$$

where $z _ { k }$ is the embedding of the k-th subgraph segment and $\hat { y } _ { k } ^ { s u b g r a p \bar { h } }$ is its predicted subgraph latency contribution.

In the composed NDT, $\Delta _ { \phi }$ is implemented using a gated recurrent unit (GRU). A GRU is a recurrent neural network module that processes a sequence step by step while maintaining a hidden state. As UT outputs are naturally ordered along the queried route, a GRU is well suited to the problem setting. Given the ordered embeddings $z _ { 1 } , \dots , z _ { K } .$ , the GRU updates its hidden state as

$$
h _ { k } = \mathrm { G R U } _ { \phi } ( z _ { k } , h _ { k - 1 } ) ,\tag{8}
$$

and the final hidden state is mapped to the correction term using an MLP readout. We refer to the full NDT using UT encoders and a GRU-based composer as the Composable GRU. Its architecture is described in Section IV-A2d.

## IV. EXPERIMENTAL SETUP

We evaluate the proposed composable NDT approach through a staged experimental design, moving from controlled synthetic settings to congestion-aware traffic scenarios and simulator-generated network traces.

TABLE I  
EVALUATION SPLITS AND THEIR GENERALISATION PURPOSE.
<table><tr><td>Scenario</td><td>Difference from training set</td><td>Purpose</td></tr><tr><td>Seen</td><td>Held-out samples from the same design space</td><td>In-distribution prediction</td></tr><tr><td>Unseen composition</td><td>Arrangement of UTs in NDT</td><td>Compositional generali- sation</td></tr><tr><td>Unseen size</td><td>Number of nodes per UT</td><td>Size generalisation</td></tr><tr><td>Unseen composition + size</td><td>Number of nodes per UT and UT arrangement in NDT</td><td>Hardest systematic gen- eralisation setting</td></tr><tr><td>Unseen traffic regime</td><td>Arrangement of UTs in NDT and per-UT traffic delay regime</td><td>Traffic-behaviour gener- alisation</td></tr></table>

## A. Evaluation Protocol

1) Evaluation Scenarios and Size Definitions: We evaluate the accuracy of UT encoders and composer under seen scenarios, comprising held-out samples from the same design space as the training data, and unseen scenarios that differ from this design space, as described in Table I.

## 2) Training Protocol and Variants:

a) UT Encoder Training: Each UT family was trained using 6, 000 route-conditioned samples generated from multiple instances of the corresponding subgraph family, including subgraphs of different sizes where permitted by the family definition. For each subgraph instance, the network configuration was varied by changing parameters, including the source– destination pair, route, bandwidth, and queue size. The latency labels depend on the experiments, as shown in Section IV-B.

All family encoders used the same architecture and optimisation settings. The hidden dimension was set to 64, with three GINE message-passing layers and a dropout rate of 0.1. The encoders were trained for 200 epochs using Adam with a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 5 }$ , and batch size 128. A fixed random seed was used.

The encoder produces two outputs. First, it generates a compact 64-dimensional embedding that represents the subgraph in a way useful for composition. Second, it predicts a scalar value of the estimated subgraph segment latency.

b) centralised GNN (baseline): is a full-topology reference model. It receives the same node features, edge features, and route descriptors as the composable models, but predicts on the complete topology rather than on a sequence of UT outputs. The GNN uses three GINE message-passing layers with a hidden dimension of 64, residual connections, ReLU activations, global mean pooling, and a dropout rate of 0.1. The model was trained for 200 epochs using Adam with a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 5 }$ , a batch size of 64, and mean-squared error as the training loss.

For each experiment, it is trained using the same train/validation/test splits and full-route latency labels as the composable models. The same optimizer, learning-rate setting, early-stopping criterion, and evaluation metric are used unless otherwise stated.

This baseline represents the standard full-graph learning setting, where the model can observe the full topology and route context directly. It is not designed as a reusable compositional model, but to provide a comparison point for evaluating the cost and benefit of using composed subgraph predictors.

c) Encoder Summation (baseline): is the no-correction summation baseline. It removes the learned correction term in Eq. 7 and predicts route latency only by summing the UT latency estimates. This baseline isolates the contribution of the learned composer by testing how far subgraph latency predictions can explain end-to-end latency without correction.

d) Composable GRU: is the main sequence-aware variant described in Section III-F. For a route decomposed into K segments, the corresponding 64-dimensional UT embeddings are arranged according to their traversal order and processed by a single-layer GRU with a hidden dimension of 64. The composer consists of a single-layer GRU followed by a twolayer correction head, which produces the correction $\Delta _ { \phi }$ which is added to the summed UT predictions.

e) Composable MLP: is an order-invariant ablation uses the same UT encoders and latency estimates as the main approach, but computes the correction from mean-pooled UT embeddings. A multilayer perceptron (MLP) is a feed-forward neural network that maps an input vector to an output prediction through fully connected layers. The valid 64-dimensional UT embeddings are mean-pooled to form a fixed-dimensional route representation, while a mask excludes padded segments. This representation is processed by a three-layer MLP with 64- dimensional hidden layers, ReLU activations, and a dropout rate of 0.1. Because mean pooling removes traversal order, this variant tests whether UT predictions plus a route-level correction are sufficient.

3) Metrics: We evaluate prediction accuracy using the coefficient of determination, $R ^ { 2 }$ , as the main metric. As latency prediction is a regression task and our experiments span different topology sizes, traffic models, and latency ranges, $R ^ { 2 }$ provides a scale-normalized measure of comparison across heterogeneous settings. $R ^ { 2 }$ values close to 1 indicate accurate prediction. $R ^ { 2 } = 0$ means the performance is no better than predicting the mean latency of the test split. Negative $R ^ { 2 }$ values indicate worse-than-mean prediction and are useful for identifying failed transfer or poor generalisation under topology changes.

## B. Datasets

1) Controlled Synthetic Datasets: We first construct controlled synthetic datasets to study how topology decomposition affects UT composition. It is necessary to determine whether UTs can be composed reliably and which decomposition strategy provides a clear basis for reuse. We focus on identifying decomposition strategies that maximise opportunities for UT reuse rather than on finding graph partitions that are theoretically or computationally optimal. Consequently, decomposition quality is assessed in terms of reusability and composability, not partition optimality.

In the synthetic dataset, each dataset sample consists of a composed topology, a route, node and edge features, and a route-level latency label. The topologies are generated from compositions of topologies and sizes of the UT families described in Section III-C. Node features include route indicators, route position, and queue size; edge features include bandwidth and route-membership descriptors. These features are processed during message passing, while route descriptors indicate relevant route segment nodes and links.

2) Local Topology Change: To evaluate reuse when a subset of the full topology changes, we construct topology-change datasets. Each data sample contains a base topology and a changed topology. The changed topology randomly modifies a subgraph of the target topology. This setting captures the intended reuse scenario of a composable NDT: when a network changes locally, prediction should be maintained by reusing existing UTs, without rebuilding a full topology-level model. It specifically evaluates cases where the modified topology remains expressible within the existing UT library, avoiding the need for full remapping.

3) Congestion-Aware Traffic Dataset: Beyond topology modification, we also consider congestion-aware datasets to understand if UT composition remains effective when route latency depends on traffic load, rather than topology structure alone. We use three analytical delay-generation models to create utilization-driven queuing (M/M/1 [39]), arrival and service variability (Kingman G/G/1 [40]), and bound-style delay targets (network-calculus-inspired delay model [41]) for traffic-dependent latencies to evaluate NDT composition. For each link e, utilisation is defined as

$$
\rho _ { e } = { \frac { \lambda _ { e } } { \mu _ { e } } } ,\tag{9}
$$

where $\lambda _ { e }$ denotes the average rate at which packets arrive at link e, while $\mu _ { e }$ denotes the average rate at which packets can be served or transmitted by that link.

We use three congestion regimes: low $\rho _ { e } \sim U ( 0 . 1 0 , 0 . 3 0 )$ medium $\rho _ { e } \sim U ( 0 . 4 0 , 0 . 6 5 )$ , and high $\rho _ { e } \sim U ( 0 . 7 0 , 0 . 9 0 )$ The ranges are chosen to create distinguishable traffic regimes while avoiding degenerate cases. Very low utilisation would remove most queuing effects, while values close to one can make queuing-based delays numerically unstable and dominate the learning target. The high regime represents strong but stable congestion pressure rather than near-saturation behaviour.

4) GNNet DT Challenge Dataset & Topology Zoo: We use two external sources to evaluate whether the proposed approach can operate beyond manually constructed topology compositions.

a) GNNet NDT Challenge Dataset: The publicly available NDT challenge GNNet simulation traces [9] are used to evaluate whether the proposed approach can operate beyond controlled analytical latency generation. GNNet is a networkperformance dataset from an NDT challenge. It was generated from OMNeT++ packet-level network simulation and provides topology, routing, traffic, and route-level delay labels, allowing us to test the approach on data produced by a network simulator.

The evaluation on the GNNet challenge dataset should be interpreted as a bridge between the synthetic evaluation and broader real-world validation. It shows that the reusable UT encoder representations and the composer form a viable workflow on OMNeT++-generated network traces.

b) Topology Zoo: To evaluate our approach on real-world network topologies, we use Topology Zoo [10]. Topology Zoo is a public collection of network topologies collected from real operator, research, and backbone networks. Since Topology Zoo provides topology structure but not measured latency labels, we generate route-level latency targets using the approach described in Section IV-B1. The topology decomposition was manually specified in the current experiments. For each network, we inspected the graph structure and identified subgraphs in the UT libraries.

## V. RESULTS

## A. Controlled Validation of Unit Twin Composition

Table II compares two topology decomposition designs using the controlled synthetic data:

a) Edge-Disjoint: assigns each physical link to exactly one UT, while adjacent UTs may share boundary nodes. Both boundary-node variants achieve high accuracy under all scenarios, showing that pretrained UT encoders from known families can be recomposed for latency predictions, suggesting that preserving the ordered sequence of UT outputs is useful for route-level composition.

b) Edge-Shared: decomposition allows a physical link to appear in more than one UT. This diagnostic setting tests whether overlapping link ownership makes subgraph latency attribution less stable. Although the results show strong performance on seen configurations, their behaviour is less stable under unseen scenarios, suggesting that shared physical links make subgraph latency ownership ambiguous. The same linklevel contribution may be represented by multiple UTs, making composed prediction harder to generalise.

An interface-aware composer is included as a diagnostic variant for this setting. It augments the composer with additional information about the UT overlaps, testing whether explicit boundary information can compensate. The results improve stability over the failing GRU case, but do not recover the performance of the edge-disjoint node-overlap design. This indicates that the weakness is not simply caused by insufficient composer capacity. Rather, duplicating physical links across UTs weakens the interpretation of latency contributions, and makes systematic composition harder under unseen scenarios.

Based on these results, edge-disjoint is used in the following experiments.

## B. Reuse under Topology Modification

Table III evaluates whether the composable NDT can remain effective after a target topology is modified. We consider three scenarios: NDT performance on the original base topology, zero-shot reuse on the modified topology without retraining, and adaptation using 1000 samples from the modified topology to retrain the composer. We consider each scenario under three traffic delay models, where they are applied homogeneously. Because the objective of this experiment is to evaluate reuse under local topology modification rather than to evaluate different composer designs, we focus on comparing the main composable approach, Composable GRU, with the centralised GNN baseline.

The centralised GNN achieves strong performance on the base topology. However, its zero-shot performance drops substantially, with negative $R ^ { 2 }$ for M/M/1 and network-calculus traffic, and close to zero for Kingman G/G/1. This indicates that the full-topology representation is sensitive to local structural changes without retraining. In contrast, the composable NDT zero-shot performance remains close to its base-topology performance across all three traffic models. This suggests that the changed topology can still be represented through a new assembly of known UTs, allowing the NDT to remain useful without retraining.

After retraining, the centralised GNN recovers substantially, reaching positive $R ^ { 2 }$ across all three traffic models. However, it remains below the adapted composable NDT in these tests. The composable NDT also changes only slightly from zeroshot to adapted performance, suggesting that most of its benefit in this setting comes from UT reuse rather than from retraining. This result supports the intended reuse scenario of the composable NDT: when a topology changes locally but remains expressible through known UTs, route-level prediction can be maintained by recombining reusable UT outputs instead of rebuilding a monolithic full-topology predictor.

## C. Traffic-Focused Evaluation: Congestion-Aware Traffic

Table IV describes the ability of the composable NDT to generalise across network latency scenarios for a fixed topology using the congestion-aware traffic dataset.

For each traffic delay model, we consider two training settings for the composer: single-train setting, the composer is trained on traffic delay data from a single congestion regime and tested on both homogeneous and heterogeneous regimes; mixed-train, heterogeneous congestion delay data samples are also included during training. This allows us to distinguish the effect of compositional structure from the effect of exposure to mixed traffic conditions.

1) Homogeneous Congestion Regime: The Low, Medium and High columns in Table IV report on composer accuracy for the corresponding congestion regime. In this setting, all UTs within an NDT along a route use the same congestion regime and the same delay-generation model, see Section IV-B3. This tests whether UT prediction and composition remain accurate under different congestion separately.

Encoder Sum. consistently underperforms the composable approaches, particularly under the composition + size split, showing that UT predictions alone are not sufficient in congestion scenarios.

The centralised GNN performs well in seen and unseencomposition settings. However, its performance drops sharply under the composition-plus-size split across all traffic delay models. In contrast, the composable GRU and MLP NDTs remain positive in this split and remain comparable across all categories. This suggests that the hardest case is not congestion modelling alone, but congestion-aware prediction under a changed UT composition and route length.

TABLE II  
COMPARISON OF TOPOLOGY DECOMPOSITION & NDT COMPOSITION. VALUES ARE $R ^ { 2 }$
<table><tr><td>Decomposition setting</td><td>NDT</td><td>Seen</td><td>Unseen comp.</td><td>Unseen size</td></tr><tr><td rowspan="3">Edge-disjoint, single boundary node</td><td>Composable GRU</td><td>0.9995</td><td>0.9912</td><td>0.9633</td></tr><tr><td>Composable MLP</td><td>0.9988</td><td>0.9799</td><td>0.7310</td></tr><tr><td>Encoder Sum.</td><td>0.9694</td><td>0.9217</td><td>0.3486</td></tr><tr><td rowspan="3">Edge-disjoint, multiple boundary nodes</td><td>Composable GRU</td><td>0.9994</td><td>0.9969</td><td>0.9155</td></tr><tr><td>Composable MLP</td><td>0.9986</td><td>0.9885</td><td>0.9475</td></tr><tr><td>Encoder Sum.</td><td>0.9366</td><td>0.9230</td><td>0.6746</td></tr><tr><td rowspan="3">Edge-shared</td><td>Composable GRU</td><td>0.9912</td><td>-2.8882</td><td>-2.7486</td></tr><tr><td>Composable MLP</td><td>0.9383</td><td>0.8456</td><td>0.4963</td></tr><tr><td>Encoder Sum.</td><td>0.6184</td><td>0.5944</td><td>-1.0242</td></tr><tr><td>Edge-shared with interface information</td><td>Interface-aware composer</td><td>0.9758</td><td>0.8211</td><td>0.6975</td></tr></table>

TABLE III  
TOPOLOGY-CHANGE REUSE RESULTS WITH 1000 CHANGED-TOPOLOGY ADAPTATION SAMPLES. VALUES ARE $R ^ { 2 }$
<table><tr><td>Delay Model</td><td>NDT</td><td>Base</td><td>Zero-shot</td><td>Retrained</td></tr><tr><td>M/M/1</td><td>Composable</td><td>0.985</td><td>0.982</td><td>0.986</td></tr><tr><td rowspan="2">Kingman G/G/1</td><td>centralised</td><td>0.996</td><td>-1.064</td><td>0.952</td></tr><tr><td>Composable</td><td>0.967</td><td>0.973</td><td>0.975</td></tr><tr><td rowspan="2">Network calculus</td><td>centralised Composable</td><td>0.978</td><td>0.039</td><td>0.952</td></tr><tr><td>centralised</td><td>0.981 0.983</td><td>0.981 -0.626</td><td>0.982 0.963</td></tr></table>

2) Heterogeneous Congestion Regime: In this setting, we consider the mixed-regimes scenarios, where UTs along the same route use different congestion regimes from the same delay-generation model.

In Table IV, the centralised GNN produces negative $R ^ { 2 }$ for all scenarios under the composition + size setting, while the MLP and GRU composers remain positive. This suggests that traffic heterogeneity is not the main source of failure, the harder domain problem arises from topology composition.

We consider single- and mixed-train settings. Encoder Sum. performs substantially below the composable approach, confirming that a learned correction term is needed to capture route-level effects beyond independent subgraph latency predictions. For the MLP and GRU composers, mixed training does not consistently improve performance over single-regime training, suggesting that the composer can combine UT outputs from different congestion levels without observing every mixed-level order during training. For the centralised GNN, mixed training improves some mixed-seen and mixed-unseen cases, but does not resolve the failure under unseen composition + size. This supports the interpretation that robustness comes from the UT composition structure rather than from simply adding more mixed congestion samples.

Overall, the results show that the GRU and MLP composers maintain positive performance under different traffic scenarios.

## D. Simulator-Trace Evaluation: Challenge Dataset

Table V reports the route-delay prediction results on the GNNet NDT training dataset. Composable GRU achieves the best overall performance across all three splits, showing that the proposed approach can operate on simulator-generated network traces rather than only analytical data and improve prediction through learned correction. Centralised GNN also performs strongly, but is worse than Composable GRU under the unseen scenarios.

In contrast, MLP performs much worse, suggesting that mean-pooled UT embeddings are insufficient for this dataset. This indicates that route structure and link-level context are important for capturing topology effects on latency in the GNNet dataset.

## E. Topology Zoo

Table VI shows results for six Topology Zoo topologies. Due to space limitation, we only report the first six topologies in alphabetical order from the dataset. Composable GRU reuses the trained UT encoders and route-level composer trained on the controlled synthetic composition datasets described in Section IV-B1. Centralised transfer baseline applies a monolithic GNN trained on the same controlled synthetic composition datasets to each Topology Zoo graph. Centralised in-topology trains and evaluates a separate full-topology GNN for each target topology. Thus, centralised transfer tests direct cross-topology reuse of a full-graph predictor, while centralised in-topology training tests how well a target-specific full-graph model performs when data from the target topology are available.

Across the six evaluated topologies, the composable GRU achieves consistently high $R ^ { 2 }$ , with an average of 0.995. This indicates that, when a real topology structure can be decomposed into supported UT families, the trained UT encoders and route-level composer can be reused without training a new full-topology predictor for each graph. In this evaluation, the UT families defined in Section III-C are sufficient to cover all decomposed subgraphs in the Topology Zoo networks.

Compared with the composable GRU, the two centralised baselines highlight the benefit of structural reuse, as both consistently perform worse. The centralised transfer baseline performs poorly on all selected Topology Zoo graphs with negative $R ^ { 2 }$ values, showing that a monolithic GNN trained on the synthetic composed-topology distribution does not directly generalise to real topology structures. The centralised intopology model performs better because it is trained using data from each target topology, but its performance is still less stable and remains below the composable NDT in these experiments. One possible reason is that the centralised GNN depends on the route diversity available within a single graph, whereas the composable approach reuses UT encoders trained across repeated subgraphs.

TABLE IV  
RESULTS ACROSS HOMOGENEOUS AND HETEROGENEOUS CONGESTION SCENARIOS. VALUES ARE $R ^ { 2 }$
<table><tr><td>Delay Model</td><td>Composer</td><td>Low</td><td>Medium</td><td>High</td><td>Mixed-seen</td><td>Mixed-unseen</td><td>Unseen comp.</td><td>Comp. + size</td></tr><tr><td rowspan="7">M/M/1</td><td>Comp. GRU, single-train</td><td>0.983</td><td>0.988</td><td>0.987</td><td>0.983</td><td>0.983</td><td>0.980</td><td>0.917</td></tr><tr><td>Comp. GRU, mixed-train</td><td>0.984</td><td>0.987</td><td>0.986</td><td>0.986</td><td>0.984</td><td>0.956</td><td>0.866</td></tr><tr><td>Comp. MLP, single-train</td><td>0.983</td><td>0.987</td><td>0.987</td><td>0.983</td><td>0.981</td><td>0.985</td><td>0.921</td></tr><tr><td>Comp. MLP, mixed-train</td><td>0.981</td><td>0.987</td><td>0.986</td><td>0.985</td><td>0.983</td><td>0.987</td><td>0.895</td></tr><tr><td>Central, single-train</td><td>0.996</td><td>0.996</td><td>0.993</td><td>0.977</td><td>0.979</td><td>0.978</td><td>-2.533</td></tr><tr><td>Central, mixed-train</td><td>0.991</td><td>0.996</td><td>0.990</td><td>0.994</td><td>0.993</td><td>0.992</td><td>-2.175</td></tr><tr><td>Encoder Sum.</td><td>0.430</td><td>0.240</td><td>0.561</td><td>0.396</td><td>0.583</td><td>0.585</td><td>0.143</td></tr><tr><td rowspan="7">Kingman</td><td>Comp. GRU, single-train</td><td>0.981</td><td>0.983</td><td>0.966</td><td>0.957</td><td>0.966</td><td>0.974</td><td>0.934</td></tr><tr><td>Comp. GRU, mixed-train</td><td>0.978</td><td>0.981</td><td>0.965</td><td>0.970</td><td>0.974</td><td>0.976</td><td>0.934</td></tr><tr><td>Comp. MLP, single-train</td><td>0.980</td><td>0.982</td><td>0.966</td><td>0.968</td><td>0.973</td><td>0.976</td><td>0.925</td></tr><tr><td>Comp. MLP, mixed-train</td><td>0.979</td><td>0.977</td><td>0.966</td><td>0.969</td><td>0.974</td><td>0.977</td><td>0.923</td></tr><tr><td>Central, single-train</td><td>0.989</td><td>0.988</td><td>0.968</td><td>0.877</td><td>0.887</td><td>0.914</td><td>-1.151</td></tr><tr><td>Central, mixed-train</td><td>0.979</td><td>0.984</td><td>0.969</td><td>0.974</td><td>0.976</td><td>0.982</td><td>-0.671</td></tr><tr><td>Encoder Sum.</td><td>0.357</td><td>0.246</td><td>0.816</td><td>0.687</td><td>0.817</td><td>0.825</td><td>0.702</td></tr><tr><td rowspan="6">Network calculus</td><td>Comp. GRU, single-train</td><td>0.973</td><td>0.983</td><td>0.979</td><td>0.978</td><td>0.980</td><td>0.978</td><td>0.777</td></tr><tr><td>Comp. GRU, mixed-train</td><td>0.973</td><td>0.983</td><td>0.978</td><td>0.980</td><td>0.980</td><td>0.980</td><td>0.759</td></tr><tr><td>Comp. MLP, single-train</td><td>0.973</td><td>0.983</td><td>0.979</td><td>0.979</td><td>0.980</td><td>0.982</td><td>0.775</td></tr><tr><td>Comp. MLP, mixed-train</td><td>0.973</td><td>0.983</td><td>0.978</td><td>0.980</td><td>0.980</td><td>0.983</td><td>0.764</td></tr><tr><td>Central, single-train</td><td>0.982</td><td>0.985</td><td>0.979</td><td>0.947</td><td>0.962</td><td>0.959</td><td>-3.230</td></tr><tr><td>Central, mixed-train</td><td>0.979</td><td>0.986</td><td>0.977</td><td>0.982</td><td>0.981</td><td>0.984</td><td>-4.524</td></tr><tr><td></td><td>Encoder Sum.</td><td>0.555</td><td>0.648</td><td>0.733</td><td>0.694</td><td>0.776</td><td>0.792</td><td>0.082</td></tr></table>

TABLE V

ROUTE-DELAY PREDICTION RESULTS ON THE CONVERTED GNNET TRAINING DATASET. VALUES ARE $R ^ { 2 }$
<table><tr><td>NDT</td><td>Val. seen</td><td>Test seen</td><td>Test unseen size</td></tr><tr><td>Composable GRU</td><td>0.9893</td><td>0.9881</td><td>0.9772</td></tr><tr><td>Composable MLP</td><td>0.2953</td><td>0.3484</td><td>0.2407</td></tr><tr><td>centralised GNN</td><td>0.9852</td><td>0.9827</td><td>0.9511</td></tr><tr><td>Encoder Sum.</td><td>0.9870</td><td>0.9848</td><td>0.9758</td></tr></table>

Overall, the Topology Zoo results show that the proposed approach can decompose real network topologies into subgraphs represented by supported UT families and reuse learned encoders for latency prediction.

## VI. CONCLUSION

This paper presents a composable Network Digital Twin (NDT) approach for per-route end-to-end latency prediction based on reusable Unit Twins (UTs) and a route-level composer. Rather than training a single predictor for an entire network, the proposed approach decomposes network topologies into reusable subgraph components that can be retrieved and assembled into NDTs to model new topologies and routes. Across controlled composition experiments, congestion-aware traffic scenarios, topology-change studies, GNNet traces, and Topology Zoo networks, the composable approach achieved high predictive accuracy while maintaining robustness under unseen topology compositions, topology-size changes, heterogeneous traffic-delay conditions, and local network modifications.

A key finding is that, while centralised GNN-based DTs often achieved comparable or better accuracy in seen settings, their performance deteriorated substantially under the most challenging unseen scenarios, in several cases yielding negative $R ^ { 2 ^ { \circ } }$ values. In contrast, the composable NDT consistently maintained positive predictive performance through the reuse and recombination of previously trained UTs. These results suggest that reusable subgraph-level composition provides a practical and robust mechanism for NDT reuse when networks evolve.

## REFERENCES

[1] Z. Min et al., “Managing and optimizing 5g & beyond network resources for multi-task digital twin applications in industry 4.0,” in 2023 IEEE 26th International Symposium on Real-Time Distributed Computing (ISORC), IEEE, 2023, pp. 220–223.

[2] G. Nencioni, R. G. Garroppo, A. J. Gonzalez, B. E. Helvik, and G. Procissi, “Orchestration and control in software-defined 5g networks: Research challenges,” Wireless communications and mobile computing, vol. 2018, no. 1, p. 6 923 867, 2018.

[3] M. I. Syed, R. Teixeira, S. Ayoubi, and G. Grassi, “The challenges of trace-driven wi-fi emulation,” arXiv preprint arXiv:2002.03905, 2020.

[4] G. F. Riley and T. R. Henderson, “The ns-3 network simulator,” in Modeling and tools for network simulation, Springer, 2010, pp. 15–34.

[5] C. Guemes-Palau, M. Ferriol-Galm¨ es, J. Paillisse-´ Vilanova, A. Lopez-Bresc´ o, P. Barlet-Ros, and´ A. Cabellos-Aparicio, “Routenet-gauss: Hardwareenhanced network modeling with machine learning,” IEEE Transactions on Networking, 2026.

[6] P. Almasan et al., “Network digital twin: Context, enabling technologies and opportunities,” IEEE Communications Magazine, pp. 1–13, 2022. DOI: 10.1109/ MCOM.001.2200012

TABLE VI

TOPOLOGY ZOO END-TO-END LATENCY PREDICTION RESULTS. VALUES ARE

<table><tr><td>Topology</td><td>Nodes</td><td>Edges</td><td>Routes</td><td>Comp. GRU</td><td>Cen. transfer</td><td>Cen. in-topology</td><td>Encoder Sum.</td></tr><tr><td>AARNET</td><td>19</td><td>24</td><td>171</td><td>0.998</td><td>-5.927</td><td>0.887</td><td>0.980</td></tr><tr><td>Abilene</td><td>11</td><td>14</td><td>55</td><td>0.997</td><td>-6.589</td><td>0.061</td><td>0.982</td></tr><tr><td>AboveNet</td><td>23</td><td>31</td><td>253</td><td>0.996</td><td>-2.313</td><td>0.940</td><td>0.980</td></tr><tr><td>ACOnet</td><td>23</td><td>31</td><td>253</td><td>0.993</td><td>-21.570</td><td>0.804</td><td>0.959</td></tr><tr><td>AGIS</td><td>25</td><td>30</td><td>300</td><td>0.992</td><td>-1.678</td><td>0.570</td><td>0.975</td></tr><tr><td>AI3</td><td>10</td><td>9</td><td>45</td><td>0.995</td><td>-5.530</td><td>0.419</td><td>0.974</td></tr><tr><td>Average</td><td>一</td><td>一</td><td>一</td><td>0.995</td><td>-7.268</td><td>0.614</td><td>0.975</td></tr></table>

[7] H. Wang, Y. Wu, G. Min, and W. Miao, “A graph neural network-based digital twin for network slicing management,” IEEE Transactions on Industrial Informatics, vol. 18, no. 2, pp. 1367–1376, 2020.

[8] D.-T. Ngo, O. Aouedi, K. Piamrat, T. Hassan, and P. Raipin-Parvedy, “Empowering digital twin for fu-´ ture networks with graph neural networks: Overview, enabling technologies, challenges, and opportunities,” Future internet, vol. 15, no. 12, p. 377, 2023.

[9] J. Suarez-Varela et al., “The graph neural networking´ challenge: A worldwide competition for education in ai/ml for networks,” ACM SIGCOMM Computer Communication Review, vol. 51, no. 3, pp. 9–16, 2021.

[10] S. Knight, H. X. Nguyen, N. Falkner, R. Bowden, and M. Roughan, “The internet topology zoo,” IEEE Journal on Selected Areas in Communications, vol. 29, no. 9, pp. 1765–1775, 2011.

[11] D.-H. Tran et al., “Network digital twin for 6g and beyond: An end-to-end view across multi-domain network ecosystems,” IEEE Open Journal of the Communications Society, vol. 6, pp. 6866–6911, 2025. [Online]. Available: https ://api . semanticscholar. org/CorpusID : 279119307

[12] X. Lin, L. Kundu, C. Dick, E. Obiodu, and T. Mostak, “6g digital twin networks: From theory to practice,” IEEE Communications Magazine, vol. 61, pp. 72–78, 2022. [Online]. Available: https://api.semanticscholar. org/CorpusID:254246974

[13] B. Li et al., “Learnable digital twin for efficient wireless network evaluation,” in MILCOM 2023-2023 IEEE Military Communications Conference (MILCOM), IEEE, 2023, pp. 661–666.

[14] C. Modesto et al., “Towards a robust transport network with self-adaptive network digital twin,” Computer Networks, p. 111 967, 2025.

[15] T. Krishnamohan and P. Harvey, “Openrase: Service function chain emulation,” in 2025 International Conference on Software, Telecommunications and Computer Networks (SoftCOM), 2025, pp. 1–6.

[16] S. Ding, D. Flynn, and P. Harvey, “Automated digital twin generation for network testing: A multi-topology validation,” in ICC 2026 - IEEE International Conference on Communications, 2026.

[17] Z. Wei, S. Wang, D. Li, F. Gui, and S. Hong, “Data-Driven Routing: A Typical Application of Digital Twin Network,” in 2021 IEEE 1st International Conference on Digital Twins and Parallel Intelligence (DTPI), IEEE, Jul. 2021, pp. 1–4, ISBN: 978-1-6654-3337-2. DOI: 10 . 1109 / DTPI52967 . 2021 . 9540073 [Online]. Available: https : / / ieeexplore . ieee . org / document / 9540073/

[18] M. Saravanan, P. S. Kumar, and A. R. Kumar, “Enabling Network Digital Twin to improve QoS Performance in Communication Networks,” in 2022 IEEE Smartworld, Ubiquitous Intelligence & Computing, Scalable Computing & Communications, Digital Twin, Privacy Computing, Metaverse, Autonomous & Trusted Vehicles (Smart-World/UIC/ScalCom/DigitalTwin/PriComp/Meta), IEEE, Dec. 2022, pp. 2151–2160, ISBN: 979- 8-3503-4655-8. DOI: 10 . 1109 / SmartWorld - UIC - ATC - ScalCom - DigitalTwin - PriComp - Metaverse56740 . 2022 . 00309 [Online]. Available: https://ieeexplore.ieee.org/document/10189764/

[19] K. Rusek, J. Suarez-Varela, P. Almasan, P. Barlet-Ros,´ and A. Cabellos-Aparicio, “Routenet: Leveraging graph neural networks for network modeling and optimization in sdn,” IEEE Journal on Selected Areas in Communications, vol. 38, no. 10, pp. 2260–2270, 2020.

[20] A. Mozo, A. Karamchandani, S. Gomez-Canaval, M. Sanz, J. I. Moreno, and A. Pastor, “B5gemini: Ai-driven network digital twin,” Sensors, vol. 22, no. 11, p. 4106, 2022.

[21] L. Hui, M. Wang, L. Zhang, L. Lu, and Y. Cui, “Digital twin for networking: A data-driven performance modeling perspective,” IEEE Network, vol. 37, no. 3, pp. 202–209, 2022.

[22] M. Wang, Y. Cui, X. Wang, S. Xiao, and J. Jiang, “Machine learning for networking: Workflow, advances and opportunities,” IEEE Network, vol. 32, pp. 92–99, 2017. [Online]. Available: https://api.semanticscholar. org/CorpusID:3113937

[23] S. Gil, E. Kamburjan, P. Talasila, and P. G. Larsen, “An architecture for coupled digital twins with semantic lifting,” Software and Systems Modeling, vol. 24, no. 5, pp. 1379–1404, 2025.

[24] A. Zaki-Hindi et al., “A reference functional architecture for network digital twins in 6g systems,” IEEE Open Journal of the Communications Society, vol. 7, pp. 2068–2101, 2026. [Online]. Available: https://api. semanticscholar.org/CorpusID:286151928

[25] M. Frasheri, H. Ejersbo, C. Thule, and L. Esterle, “Rmqfmu: Bridging the real world with co-simulation technical report,” arXiv preprint arXiv:2107.01010, 2021.

[26] A. Junghanns et al., “The functional mock-up interface 3.0-new features enabling new applications,” in Modelica conferences, 2021, pp. 17–26.

[27] J. Borges, F. Bastos, I. Correa, P. Batista, and A. Klautau, “Caviar: Co-simulation of 6g communications, 3-d scenarios, and ai for digital twins,” IEEE Internet of Things Journal, vol. 11, no. 19, pp. 31 287–31 300, 2024.

[28] F. Oest, E. Frost, M. Radtke, and S. Lehnhoff, “Coupling omnet++ and mosaik for integrated co-simulation of ict-reliant smart grids,” ACM SIGENERGY Energy Informatics Review, vol. 3, no. 1, pp. 14–25, 2023.

[29] P. van Schalkwyk and D. Isaacs, “Achieving scale through composable and lean digital twins,” in The Digital Twin, Springer, 2023, pp. 153–180.

[30] P. Kuruppuarachchi, S. Rea, and A. McGibney, “An architecture for composite digital twin enabling collaborative digital ecosystems,” in 2022 IEEE 25th International Conference on Computer Supported Cooperative Work in Design (CSCWD), IEEE, 2022, pp. 980–985.

[31] G. N. Schroeder, C. Steinmetz, R. N. Rodrigues, A. Rettberg, and C. E. Pereira, “Digital twin connectivity topologies,” IFAC-PapersOnLine, vol. 54, no. 1, pp. 737–742, 2021, 17th IFAC Symposium on Information Control Problems in Manufacturing INCOM 2021, ISSN: 2405-8963. DOI: https : / / doi . org / 10 . 1016 / j . ifacol . 2021 . 08 . 086 [Online]. Available: https : / / www . sciencedirect . com / science / article / pii / S2405896321008302

[32] J. Suarez-Varela et al., “Graph neural networks for´ communication networks: Context, use cases and opportunities,” IEEE network, vol. 37, no. 3, pp. 146–153, 2022.

[33] Y. Jin, M. Daoutis, S. Girdzijauskas, and A. Gionis, “Open world learning graph convolution for latency estimation in routing networks,” in 2022 International Joint Conference on Neural Networks (IJCNN), IEEE, 2022, pp. 1–8.

[34] V. Md et al., “Distgnn: Scalable distributed training for large-scale graph neural networks,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2021, pp. 1–14.

[35] C. Zheng et al., “Bytegnn: Efficient graph neural network training at large scale,” Proceedings of the VLDB Endowment, vol. 15, no. 6, pp. 1228–1242, 2022.

[36] R. Kondor, H. T. Son, H. Pan, B. Anderson, and S. Trivedi, “Covariant compositional networks for learning graphs,” arXiv preprint arXiv:1801.02144, 2018.

[37] Z. Wang et al., “Slapp: Subgraph-level attention-based performance prediction for deep learning models,” Neural Networks, vol. 170, pp. 285–297, 2024.

[38] Z. Zhang et al., “Hierarchical graph pooling with structure learning,” ArXiv, vol. abs/1911.05954, 2019. [Online]. Available: https : / / api . semanticscholar. org / CorpusID:208006339

[39] D. P. Bertsekas and R. G. Gallager, Data Networks, 2nd. Prentice Hall, 1992, ISBN: 0132009161.

[40] J. F. C. Kingman, “The single server queue in heavy traffic,” Mathematical Proceedings of the Cambridge Philosophical Society, vol. 57, no. 4, pp. 902–904, 1961. DOI: 10.1017/S0305004100036094

[41] R. L. Cruz, “A calculus for network delay. i. network elements in isolation,” IEEE Trans. Inf. Theor., vol. 37, no. 1, pp. 114–131, Sep. 2006, ISSN: 0018-9448. DOI: 10.1109/18.61109 [Online]. Available: https://doi.org/ 10.1109/18.61109