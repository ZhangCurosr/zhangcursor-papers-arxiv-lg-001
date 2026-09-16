# Unified Heterogeneous Graph Neural Network solver for Power Flow, Optimal Power Flow and State Estimation

Ferran Bohigas-Daranas, Hamid Latif-Martínez, Eduardo Prieto-Araujo, Member IEEE, Oriol Gomis-Bellmunt, Member IEEE, Pere Barlet-Ros, Member IEEE

Abstract—Power Flow (PF), Optimal Power Flow (OPF), and State Estimation (SE) are fundamental problems in power system analysis, but solving them is computationally expensive. Graph Neural Networks (GNNs) have been proposed as fast surrogates, yet existing solvers are trained for a single problem at a time, producing narrow models that must be rebuilt for each new task.

We propose a more general approach: a single Heterogeneous Residual Gated Graph Convolutional Network that solves all three problems with one shared backbone. Rather than learning one mapping, the model learns a reusable representation of how the network behaves, from which PF, OPF, and SE can each be estimated. Trained jointly on the three problems across diverse topologies and loading conditions, and evaluated on the IEEE 14- bus and 118-bus systems, the shared model matches the accuracy of task-specific GNN solvers and stays robust on unseen loading levels and topologies.

These results show that a single model can capture the basic operation of a power network and serve several analysis tasks at once — a first step toward a foundation model for power systems.

Index Terms—Power Flow, Optimal Power Flow, State Estimation, Graph Neural Network, Residual Gated Graph Convolutional Network, ResGated GCN, Foundational Models

## I. INTRODUCTION

deployment of advanced metering infrastructure, has significantly increased the complexity of grid monitoring and control. Fundamental power system tasks [1], namely Power Flow (PF), Optimal Power Flow (OPF), and State Estimation (SE), remain the cornerstones of grid operations. While traditional iterative methods such as Newton–Raphson (NR) [2], Interior Point (IP) [3], and Weighted Least Squares (WLS) [4] are mathematically robust, their computational burden scales super-linearly with system size [5]. Furthermore, the increasing variability of distributed energy resources (DERs) demands real-time or near-real-time solutions that traditional solvers struggle to provide under strict latency constraints.

In recent years, Deep Learning (DL) has emerged as a promising alternative for approximating these complex power system mappings. Early applications utilized Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNN) [6], [7]. However, these architectures are fundamentally limited when applied to power systems: Multilayer Perceptrons (MLPs) fail to exploit the topological structure of the grid, while CNNs are restricted to regular, Euclidean data structures and cannot naturally represent the irregular connectivity of transmission networks.

Given that power systems are inherently non-Euclidean graphs, defined by buses (nodes) and lines/transformers (edges), Graph Neural Networks (GNNs) [8], [9], [10] have emerged as a natural candidate for power system applications. The message-passing mechanism of GNNs directly mirrors the way physical laws such as Kirchhoff Current Law (KCL) propagate information across the network, introducing a topological inductive bias that aligns the learning process with the relational structure of the electrical grid. GNNbased approaches have since demonstrated strong results for individual power system tasks: PF solvers based on GNNs [11], [12] have achieved near-NR accuracy at a fraction of the computational cost; GNN-based OPF methods [13], [14], [15], [16] have shown the ability to respect operational constraints while accelerating dispatch decisions; and GNN architectures for SE [17], [18] have demonstrated robustness to measurement noise and partial observability. Despite these advances, a key limitation persists: early and current GNN variants often lack the discriminative power required to handle the high-precision requirements of power system physics [19]. This work addresses this limitation by utilizing the Residual Gated Graph Neural Network (ResGated GCN) [20], as the basic computational cell of the proposed architecture (Fig.3), providing stronger graph-distinguishing capability than prior Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) based approaches.

Despite the progress in GNN-based power system solvers, existing literature consistently treats PF, OPF, and SE as isolated problems, with separate models trained and deployed for each task. This fragmentation ignores the shared underlying physics and topological constraints common to all three problems, results in redundant training pipelines, and prevents the development of a unified, grid-aware representation. The recent emergence of foundational model initiatives for power systems, notably the open-source initiatives GridFM [21] and AI.grids [22], and more recently by industry research groups [23] confirms that the community recognizes the need to move beyond task-specific architectures. However, to date no peer-reviewed architecture has demonstrated simultaneous solution of PF, OPF, and SE within a single trained model on standard benchmark systems. This paper directly addresses that gap.

The primary contributions of this work are:

• Unified multi-task architecture: A unified heterogeneous GNN architecture trained to solve PF, OPF, and SE simultaneously, including topological perturbations, so that the model learns a generalized representation of power system physics rather than a task-specific input-output mapping, achieving accuracy comparable to task-specific GNNs while eliminating the need for separate per-task training pipelines. To the best of the authors’ knowledge, this is the first peerreviewed architecture to demonstrate this on standard IEEE benchmark systems.

• Multi-problem loss: A composite loss function for the three problems (PF, OPF, SE) simultaneously, with a problemconditioned masking strategy that prevents cross-problem gradient interference.

• Architectural hyperparameter tuning and ablation study: A systematic hyperparameter tuning and ablation that quantifies the individual contribution of each element of the propose architecture, providing evidence-based architectural guidance for future unified power system models.

The remainder of this paper is organized as follows. Section II introduces the mathematical formulation of PF, OPF and SE. Section III reviews the GNN foundations underpinning the proposed architecture. Section IV describes the proposed architecture, training objective, and data generation pipeline. Section $\mathrm { v }$ presents the experimental setup, numerical results, and discussion on the IEEE 14-bus and 118-bus test systems [24], including an analysis of scalability and the path towards foundational power system models. Section VI concludes the paper.

## II. ELECTRICAL BACKGROUND

The power system is modeled as a network of buses interconnected by transmission lines and transformers. Each bus is characterized by four electrical quantities: voltage magnitude |V|, voltage phase angle $\delta ,$ net active power injection $P ,$ and net reactive power injection Q, where net denotes the algebraic difference between local generation and consumption.

Buses are classified into three types according to which variables are specified as inputs and which are computed as outputs. The slack bus (or reference bus) is typically the terminal of the largest generating unit; its voltage magnitude and angle are held fixed, and it absorbs any residual active and reactive power mismatch in the system. PV buses correspond to generator terminals at which active power output and voltage magnitude are prescribed, while reactive power output and voltage angle are unknowns. PQ buses represent load nodes, where both active and reactive power demands are specified and the voltage phasor is to be determined.

For each bus i in an n-bus system, the complex power injection is related to the nodal voltage phasor and the injected current by:

$$
S _ { i } = P _ { i } + j Q _ { i } = V _ { i } I _ { i } ^ { * } ,\tag{1}
$$

where $S _ { i }$ is the bus power, $P _ { i }$ is the bus active power, $Q _ { i }$ is the bus reactive power, $V _ { i }$ is the bus voltage and $I _ { i } ^ { * }$ is the complex conjugate of the current injection at bus $i .$

Applying Ohm’s law through the nodal admittance matrix $Y _ { \mathrm { b u s } } ,$ , whose $( \mathrm { i } , \mathrm { j } )$ -th element is expressed in polar form as $Y _ { i j } =$ $| Y _ { i j } | e ^ { j \theta _ { i j } }$ , with $\theta _ { i j } = \arg ( Y _ { i j } )$

$$
I _ { i } = \sum _ { j = 1 } ^ { n } Y _ { i j } \ : V _ { j } .\tag{2}
$$

Substituting (2) into (1) and separating real and imaginary parts yields the fundamental power flow equations:

$$
P _ { i } = \sum _ { j = 1 } ^ { n } \left| V _ { i } \right| \left| V _ { j } \right| \left| Y _ { i j } \right| \cos ( \theta _ { i j } - \delta _ { i } + \delta _ { j } ) ,\tag{3}
$$

$$
Q _ { i } = - \sum _ { j = 1 } ^ { n } | V _ { i } | | V _ { j } | | Y _ { i j } | \sin ( \theta _ { i j } - \delta _ { i } + \delta _ { j } ) ,\tag{4}
$$

where $Y _ { i j }$ is the $( i , j ) \ – \mathrm { t h }$ element of the admittance matrix and $\delta _ { i } , \delta _ { j }$ are the voltage angles at buses i and $j ,$ , respectively.

Nodal power balance, derived from KCL, at each bus $i \in \mathcal N \colon$

$$
P _ { G i } - P _ { D i } - P _ { i } ( V , \delta ) = 0 , \quad \forall i \in \mathcal { N } ,\tag{5}
$$

$$
Q _ { G i } - Q _ { D i } - Q _ { i } ( V , \delta ) = 0 , \quad \forall i \in \mathcal { N } ,\tag{6}
$$

where $P _ { G i } , Q _ { G i }$ are active and reactive generation, $P _ { D i } , Q _ { D i }$ are the corresponding demands, and $P _ { i } ( V , \delta ) , \ Q _ { i } ( V , \delta )$ are computed from (3)-(4).

This set of nonlinear equations form the shared mathematical foundation of all three problems addressed in this work.

## A. Power Flow

Power flow (PF) analysis determines the steady-state distribution of voltages and power throughout the network for a given set of generation and load conditions. Specifically, it computes the voltage magnitude and phase angle at every bus and the active and reactive power flows on all transmission branches.

PF analysis spans across a wide range of use cases, involving different time scales, like network planning and realtime operation, where it verifies that line loadings and bus voltages remain within security limits. PF is governed by a nonlinear system of equations, (3)-(4), which must be solved simultaneously for all buses.

The industry-standard algorithm is the NR method, which linearizes the system around an initial operating point, conventionally a flat start $( \vert V \vert = 1 \mathrm { p . u . , ~ } \delta = 0 )$ , and iteratively refines the state vector by solving a linear correction system governed by the Jacobian matrix J. Convergence is typically achieved in five iterations under normal operating conditions [2]. The principal computational bottleneck is the formation and sparse factorization of J at each iteration, an operation whose cost scales super-linearly with system size.

This computational burden provides the primary motivation for the GNN-based approach proposed in this paper: once trained, the network approximates the mapping defined by (3)- (4) in a single forward pass, retaining topological awareness without repeated matrix factorizations or the sensitivity to initial conditions that can cause NR to diverge under stressed network conditions [2].

## B. Optimal Power Flow

Optimal Power Flow is a mathematical optimization problem that determines the optimal operating state of an electric power system. While conventional PF analysis simply calculates the steady-state conditions for a given set of generation and load values, OPF finds the best generation dispatch, voltage settings, transformer tap positions, and other control variables to optimize a specified objective while enforcing the full set of physical and operational constraints.

First formulated by Carpentier in [25], OPF has become a fundamental tool in power system operation. The problem has evolved from simple economic dispatch considerations to complex multi-objective optimization problems that consider environmental factors, market operations, and the integration of renewable energy sources.

The general OPF problem is formulated as:

$$
\operatorname* { m i n } _ { \mathbf { x } , \mathbf { u } } \quad f ( \mathbf { x } , \mathbf { u } )\tag{7}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { { } \mathbf { g } ( \mathbf { x } , \mathbf { u } ) = \mathbf { 0 } } \end{array}\tag{8}
$$

$$
\mathbf { h } ( \mathbf { x } , \mathbf { u } ) \leq \mathbf { 0 } ,\tag{9}
$$

where $\mathbf { x } \in \mathbb { R } ^ { 2 n }$ is the state vector (voltage magnitudes and angles), u is the vector of control variables, $f ( \mathbf { x } , \mathbf { u } )$ is the scalar objective. The equality constraint $g ( \mathbf { x } , \mathbf { u } ) = 0$ enforces nodal power balance at each bus, as defined by the KCL (5) and (6), while the inequality constraints $h ( \mathbf { x } , \mathbf { u } ) \leq 0$ capture physical and operational bounds, such as generator active and reactive power limits, nodal voltage bounds, and branch thermal capacity constraints.

## C. State Estimation

SE reconstructs the complete operating state of the network, the full set of voltage magnitudes and phase angles, from a limited and inherently noisy set of real-time measurements. It addresses three practical challenges that arise in operational grids:

• Incomplete observability: Field instrumentation does not cover every bus and branch; the state at unmonitored locations must therefore be inferred from the available data.

• Measurement noise: Readings from current and voltage transformers are corrupted by instrument error. SE filters these disturbances to provide the best statistical estimate of the true state.

• Bad data detection: SE identifies and rejects gross measurement errors, caused by meter faults or data transmission failures, that would otherwise corrupt the state estimate.

The most widely used SE method is the WLS estimator, which minimizes the weighted sum of squared measurement residuals, alternative methods include the Least Absolute Value (LAV) estimator, and Kalman filtering, applicable to dynamic state tracking.

The operational SE workflow comprises four stages: (i) realtime measurement collection and validation; (ii) observability analysis to confirm that the available measurements uniquely determine the state; (iii) state computation via WLS or an equivalent method; and (iv) bad-data detection and identification through normalized residual testing.

The output of SE, a clean, complete state vector, feeds all higher-level grid-management functions: SCADA, Energy Management Systems (EMS), OPF solvers, security assessment, automatic generation control (AGC), economic dispatch, and contingency analysis. Any degradation in SE accuracy therefore propagates directly into all downstream applications [26].

## III. GRAPH NEURAL NETWORK BACKGROUND

Power systems are inherently non-Euclidean: their behavior is governed by the admittance matrix and the adjacency structure of buses and lines/transformers, not by the Euclidean distance between nodes. GNNs overcome these limitations by operating directly on the graph structure through an iterative message-passing mechanism [9], which updates node features by aggregating information from their immediate neighborhood. Each bus (node) updates its latent representation by aggregating information from its immediate electrical neighbors, closely mirroring the way KCL propagates information across the physical network. Because the same learned parameters are applied at every node and edge regardless of graph size or configuration, GNNs are permutation-invariant and can generalize across different topologies without retraining, properties that make them uniquely appropriate for the unified multi-task solver proposed in this work.

A standard GNN layer is shown at Fig. 1, where $h _ { v }$ is the latent embedding of node v and $\mathcal { N } ( i )$ denotes the set of adjacent nodes.

Early variants such as GCN and GAT demonstrated the viability of this paradigm for power system problems, but they suffer from limited discriminative power: it has been shown that these architectures cannot distinguish between certain non-isomorphic graph structures [19], meaning different electrical networks. In power grids, where even a single line trip can substantially alter the operating state, this limitation is particularly consequential.

## A. ResGated GCN

Introduced by Bresson and Laurent [20] (2018), the Res-Gated GCN addresses the limitations of early graph neural network paradigms. While traditional Graph Recurrent Networks (GRNNs) offered complex memory but suffered from computational overhead, and early GCNs [27] provided speed but lacked depth scalability, ResGated GCN combines dynamic gating and residual connections to enable deep, efficient graph learning.

The ResGated GCN layer updates a node’s features by filtering neighbor messages through a dynamic, learned gate. The layer transformation is defined as (Fig.1), where the equation is extended to incorporate edge attributes, which is expressed as:

$$
h _ { i } ^ { ( l + 1 ) } = h _ { i } ^ { ( l ) } + \mathrm { R e L U } \left( \mathrm { B N } \left( W _ { 1 } ^ { l } h _ { i } ^ { ( l ) } + \sum _ { j \in \mathcal { N } ( i ) } \eta _ { i j } ^ { l } \odot \left( W _ { 2 } ^ { l } h _ { j } ^ { ( l ) } \right) \right) \right)\tag{10}
$$

$$
\eta _ { i j } ^ { l } = \sigma \left( W _ { 3 } ^ { l } h _ { i } ^ { \left( l \right) } + W _ { 4 } ^ { l } h _ { j } ^ { \left( l \right) } + W _ { e } ^ { l } e _ { i j } \right)\tag{11}
$$

![](images/1acd715ca05c8da8bc63a6442f6d6aed8cdf6b4f1619ca3f8b3deffca9561a2a.jpg)  
Fig. 1. GNN and ResGated GCN structure comparison.

Let $h _ { i } \in \mathbb { R } ^ { d }$ and $h _ { j } \in \mathbb { R } ^ { d }$ denote the input features of the target node i and a neighbouring node $j ,$ respectively, where $h _ { i }$ also serves as a linear residual connection. The edge between them is characterised by $e _ { i j } = [ R _ { i j } , X _ { i j } , B _ { i j } ] ^ { T }$ , which encodes the physical transmission line parameters, projected by the learnable matrix $W _ { e } ^ { \ell } \in \mathbb { R } ^ { d ^ { \prime } \times d _ { e } }$ . Four learnable weight matrices govern the layer transformations: $W _ { 1 } ^ { \ell } \in \mathbb { R } ^ { d ^ { \prime } \times d }$ applies a self-loop transform to the centre node $h _ { i } ,$ while $W _ { 2 } ^ { \ell } \in \mathbf { \bar { R } } ^ { d ^ { \prime } \times d }$ projects the neighbouring features $h _ { j } .$ . The gating mechanism is controlled by $W _ { 3 } ^ { \ell } , W _ { 4 } ^ { \bar { \ell } } \in \mathbb { R } ^ { d ^ { \prime } \times d }$ , which project the local topological states of $h _ { i }$ and $h _ { j }$ , respectively, to compute the edge gate importance. The resulting gating coefficient $\eta _ { i j }$ where $\sigma$ is the sigmoid activation, where $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ constraining values smoothly to [0, 1], acts as an element-wise filter, applied via the Hadamard product $\odot ,$ that tracks the semantic relationship between connected nodes and controls information flow from each neighbour $j \in \mathcal { N } ( i )$ . Finally, batch normalisation $\mathrm { B N } ( \cdot )$ and ReLU function, where $R e L U ( x ) =$ max(0, x) are applied to the aggregated output before the residual addition.

The main innovations and advantages of ResGated GCN are:

• Vector Edge Gating: Unlike scalar attention (e.g., GAT), the gate $\eta _ { i j }$ is a vector of the same dimension as the hidden features, allowing independent modulation per dimension.

• Residual Connections: The inclusion of a hard identity shortcut (+h<sup>ℓ</sup>) prevents vanishing gradients, enabling stable training of deeper architectures.

• Dual Self-Connection: The model applies both an explicit linear projection $( \mathbf { W } _ { 1 } \mathbf { h } _ { i } )$ within the aggregation and a hard residual addition after the non-linearity.

• Direction-Dependent Message Weighting: By incorporating both endpoint features and explicit edge attributes $e _ { i j }$ , the network captures the full heterogeneity of transmission line parameters within the learned gate.

## B. Heterogeneous Graph Neural Networks for Power Systems

Standard GNNs are homogeneous: all nodes and edges share the same feature space and are updated by identical transformation functions. This assumption is incompatible with the structure of power systems, where buses represent physically distinct entities with different operational roles, boundary conditions, and sets of known and unknown variables.

![](images/ac7635139a86926697b1c8bd16d2b0cbb45fbc26352d7b56a59a379d2b5f879d.jpg)  
(a) IEEE 9-bus system

![](images/e779f16390695138a0379c8de180c23608a3f55716bd7cc7b458f35476a4f9e6.jpg)  
(b) IEEE 9-bus graph  
Fig. 2. IEEE 9-bus system [28] represented as an electrical network and as a heterogeneous graph.

In a homogeneous GNN, the distinct semantics of slack, PV, and PQ buses are flattened into a single feature representation, forcing the model to learn type-specific behavior purely from data rather than encoding it architecturally. This increases the effective learning difficulty and risks conflating physically distinct operating modes. Heterogeneous GNNs (HGNNs) resolve this limitation by associating each node and edge type with its own learnable parameters and message-passing logic, motivated by two principal considerations:

• Bus-Type Specificity: Electrical buses are defined by which variables are known and unknown. A slack bus has a fixed voltage magnitude and angle (V, θ); a PV bus has fixed active power and voltage magnitude $( P , V ) ;$ and a PQ bus has fixed active and reactive power $( P , Q )$ . HGNNs allow the architecture to apply specific transformation matrices to each bus type, ensuring that the GNN respects these distinct boundary conditions during the feature aggregation process. An example based on the IEEE 9-bus is shown at Fig. 2

• Edge Physics: Unlike social or citation networks, the edges in a power grid (transmission lines and transformers) carry critical physical parameters such as resistance (R), reactance (X), and shunt admittance (B), for transmission lines, or transformation relation and phase shift for transformers.

## IV. PROPOSED METHODOLOGY

The proposed architecture (Fig. 3) is a deep HGNNs designed to solve PF, OPF, and SE within a single unified architecture, based on the Encoder-Processor-Decoder architecture proposed by [29], extended to heterogeneous graphs. By treating buses of different types (slack, PV, PQ) as distinct node sets $\nu _ { k }$ with independent encoder and decoder parameters, the model enforces type-specific boundary conditions for each task while sharing a common physics-informed message-passing backbone across all three problems.

## A. Architectural Components

## 1) Node Encoding and Feature Transformation

To handle the heterogeneous and task-dependent input features across bus types, we employ a bank of type-specific linear encoders. For each node type $k \in \{ \mathrm { s l a c k , p v , p q } \}$ , the raw feature vector $\mathbf { x } _ { i } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ is projected into a shared latent space $\mathbb { R } ^ { d _ { h } }$ , where $d _ { h }$ is the hidden size, via a type-specific affine transformation followed by Layer Normalization:

$$
h _ { i } ^ { ( 0 ) } = \mathrm { L a y e r N o r m } ( \mathbf { W } _ { k } \mathbf { x } _ { i } + \mathbf { b } _ { k } ) .\tag{12}
$$

where $\mathbf { W } _ { k }$ are separate weight matrices for each bus type ensures that the model distinguishes the fixed-voltage boundary conditions of a slack bus from the power-injection inputs of a PQ bus from the first layer onward, and $b _ { k }$ is the bias term. Layer Normalization stabilizes training by preventing featurescale disparities from propagating through the deep stack.

## 2) Heterogeneous Message Passing

The core of the architecture consists of L stacked HGNN layers, each applying separate transformation matrices to each directed relation type (e.g., slack-to-PQ, PV-to-PQ). At each layer, the edge attribute vector ${ \bf e } _ { i j }$ , encoding the branch impedance parameters $( R , X )$ and shunt admittance (B), is incorporated into message passing according to (10)-(11). By applying the non-linear activation to the sum of the neighbor embedding and the projected edge attributes before aggregation, the model learns the nonlinear relationships that Ohm’s and KCL impose on nodal voltages and power injections.

![](images/28bffda8a5a7178c3962ee51eb5dc4dc38875d59761b561f03bac6bf5505d208.jpg)  
Fig. 3. Architecture of the proposed unified Heterogeneous ResGated GCN.

## 3) Global Context and Residual Integration

OPF requires awareness of global system constraints, such as total generation cost and network-wide reactive power balance, that local message passing alone cannot efficiently propagate in deep architectures. A global context vector, g, is therefore computed at each layer by mean-pooling the current node embeddings across all bus types, $k ,$ and passing them through a shared MLP:

$$
\mathbf { g } = \mathbf { M L P } _ { g l o b a l } \left( \frac { 1 } { \sum _ { k } | \mathcal { V } _ { k } | } \sum _ { k \in \mathrm { T y p e s } } \sum _ { i \in \mathcal { V } _ { k } } \mathbf { h } _ { i } ^ { ( l ) } \right)\tag{13}
$$

where $\begin{array} { r } { \left| \mathcal { V } \right| = \sum _ { k } \left| \mathcal { V } _ { k } \right| } \end{array}$ denotes the total number of nodes across all bus types, so that the mean-pooling normalizes over the full graph regardless of the type partition.

The global vector is broadcast back to all nodes and combined with the per-node update via a multi-path residual connection:

$$
\mathbf { h } _ { i } ^ { ( l ) } = \mathbf { h } _ { i } ^ { ( l - 1 ) } + \mathrm { S i L U } \Big ( \mathbf { h } _ { i , \mathrm { n e w } } ^ { ( l ) } \Big ) + \mathbf { g } + \mathbf { h } _ { i } ^ { ( 0 ) }\tag{14}
$$

where SiL $\mathrm { U } ( x ) = x / ( 1 + e ^ { - x } )$ is applied element-wise. Note that SiLU(·) is distinct from the sigmoid gate $\sigma ( \cdot )$ in (11), which is reserved exclusively for the ResGated GCN gating mechanism throughout this paper. The skip connection to the initial encoding $\overline { { h _ { i } ^ { ( 0 ) } } }$ is particularly important: it ensures that the task-specific and bus-type boundary conditions encoded in the input projection remain influential throughout the full network depth, preventing the model from losing track of the physical constraints of the problem as depth increases, due to oversmoothing, a well-known problem in GNNs.

## 4) Multi-Head Decoding

The final node embeddings $\mathbf { h } _ { i } ^ { ( L ) }$ are decoded by task using variable-specific output heads, providing greater expressivity for quantities of different physical nature — notably voltage magnitude and voltage angle, which span different numerical ranges. The output biases of voltage magnitude and angle are initialized to physically meaningful values: 1.0 p.u. for voltage magnitude and 0.0 rad for voltage angle, corresponding to the flat-start initialization used in classical iterative solvers. This physics-informed initialization places the initial output distribution in a physically plausible region, reducing the number of training epochs required to reach the target accuracy regime.

## B. Unified Multi-Task Training Objective

A central challenge in training a unified solver is the heterogeneous character of the three tasks: PF and SE are regression problems with smooth loss profile, while OPF is a constrained optimization problem in which constraint satisfaction is as operationally critical as prediction accuracy. To address this, we employ the mean squared error between predicted and ground-truth state variables, providing the primary gradient signal for all three tasks.

$$
M S E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 }\tag{15}
$$

where $n$ is the total number of observations, $y _ { i }$ represents the actual (observed) values and $\hat { y } _ { i }$ represents the predicted (simulated) values.

For each task, MSE is computed as a sum of per-variable terms, applied only to the state variables that are outputs for the corresponding bus type and task:

$$
\mathcal { L } _ { M S E } = \sum _ { k \in \mathcal { V } } ( \mathcal { L } _ { V m } + \mathcal { L } _ { V a } + \mathcal { L } _ { P g } + \mathcal { L } _ { Q g } )\tag{16}
$$

For example, during a PF forward pass, ${ \mathcal { L } } _ { V _ { m } }$ is applied only to PQ buses (for which |V| is unknown), while ${ \mathcal { L } } _ { V _ { a } }$ is applied to both PQ and PV buses. This task-conditioned masking ensures that the model is penalized only for quantities it is responsible for predicting, preventing spurious gradients from degrading the shared representation.

## C. Data Generation and Pre-processing

The reliability of a GNN-based solver depends critically on the diversity and physical consistency of its training data. A model trained on a narrow distribution of operating conditions will exhibit poor generalization to scenarios outside that distribution, a phenomenon known as out-of-distribution (OOD) failure, which can manifest at the feature level (input values outside the training range) or at the topological level (changes in network connectivity, as arise in N-1 contingency analysis). To mitigate this risk, a comprehensive data generation pipeline was developed using the PandaPower[30] and VeraGrid[31] Python packages.

The following randomization strategy was applied to four different dataset scenarios, to maximize the diversity of operating conditions (Table I):

• Load variability: Active and reactive power demands $( P _ { L }$ $Q _ { L } )$ were sampled uniformly over a defined range around the nominal values. This wide range, spanning both lightload off-peak and heavily stressed peak conditions, ensures that the model encounters the full extent of the feasible operating region, including near-voltage-collapse scenarios.

• Generator and line parameter perturbation: Generator voltage setpoints $( V _ { \mathrm { g e n } } )$ and transmission line impedance $( Z _ { i j } )$ were varied within ±5% of their nominal values, simulating set-point uncertainty and parametric drift in aging infrastructure.

• Topological augmentation: The network topology was modified by randomly removing existing branches or adding new ones. This structural diversity trains the GNN backbone to recognize that the governing physics is encoded in the admittance matrix and line parameters, not in a fixed node ordering, improving robustness to grid reconfigurations and N-1 contingencies.

• Measurement noise and masking for SE: State Estimation samples are further processed to replicate the conditions of a real-time operational environment. Each measurement is corrupted by additive zero-mean Gaussian noise, with a standard deviation drawn uniformly at random from the interval [1%, 2%] of the corresponding measurement magnitude, simulating the accuracy class of typical current and voltage instrument transformers. In addition, 5% of measurements are masked (set to missing) per sample, with the masked subset selected uniformly at random across all available measurements, simulating partial observability due to communication failures or instrumentation gaps. These perturbations are applied exclusively to SE samples, ensuring that the GNN learns to reconstruct the full system state from incomplete and noisy observations, consistent with the operational role of a state estimator.

To facilitate multi-task learning within a single shared model, we augmented the node feature matrix with a one-hot encoded task-indicator vector $\tau \in \{ 0 , 1 \} ^ { 3 }$

$$
\tau _ { s } = \left\{ \begin{array} { l l } { [ 1 , 0 , 0 ] ^ { T } } & { \mathrm { i f ~ } s \in \mathrm { P o w e r ~ F l o w } } \\ { [ 0 , 1 , 0 ] ^ { T } } & { \mathrm { i f ~ } s \in \mathrm { O P F } } \\ { [ 0 , 0 , 1 ] ^ { T } } & { \mathrm { i f ~ } s \in \mathrm { S t a t e ~ E s t i m a t i o n } } \end{array} \right.\tag{17}
$$

This allows the heterogeneous encoders to condition the initial embeddings on the specific requirements of the objective task.

TABLE I  
SCENARIO DEFINITIONS.
<table><tr><td>Dataset Name</td><td>Power Ranges</td></tr><tr><td>narrow</td><td>50% to 150% nominal powers</td></tr><tr><td>mid</td><td>20% to 180% nominal powers</td></tr><tr><td>wide</td><td>0% to 200% nominal powers</td></tr><tr><td>high-topo</td><td>180% to 250% nominal powers and 40% samples with topological changes</td></tr></table>

TABLE II

NUMBER OF REJECTED SAMPLES DUE TO INFEASIBILITY OR SOLVERNON-CONVERGENCE DURING DATA GENERATION FOR THE IEEE118-BUS NETWORK.
<table><tr><td>118-bus case</td><td>narrow</td><td>mid</td><td>wide</td><td>high-topo</td></tr><tr><td>PF</td><td>0</td><td>0</td><td>5</td><td>542</td></tr><tr><td>OPF</td><td>0</td><td>0</td><td>1</td><td>546</td></tr><tr><td>SE</td><td>0</td><td>0</td><td>9</td><td>3045</td></tr></table>

Table I defines the four load scenarios, from the most conservative (narrow) to the most heavily stressed (high-topo), the latter including 40% of samples with random topological modifications. It is worth noting that the high-topo scenario operates near the boundary of the feasibility region: in some cases, more than ten candidate samples had to be discarded due to solver non-convergence before a feasible operating point was found (Table II).

## D. Evaluation and Generalization Metrics

Mean Squared Error (MSE) serves as the primary training objective but is an insufficient standalone evaluation metric for power system applications, where worst-case performance and constraint satisfaction are operationally critical. The model is therefore evaluated using the following complementary metrics, as proposed in [32]:

• Normalized Root Mean Square Error (NRMSE): Indicator based on MSE, which is normalized by the range of the ground truth output, allowing a correct comparison between different magnitudes.

$$
\mathrm { N R M S E } = \frac { \mathrm { R M S E } } { y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } } } = \frac { \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } } { y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } } }\tag{18}
$$

where $y _ { \mathrm { m a x } }$ and $y _ { \mathrm { m i n } }$ are the maximum and minimum actual values in the dataset, respectively.

• Coefficient of determination $( R ^ { 2 } )$ :

$$
R ^ { 2 } = 1 - \frac { \sum _ { i } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \sum _ { i } ( y _ { i } - \bar { y } ) ^ { 2 } } ,\tag{19}
$$

measures the proportion of variance in the target variables explained by the model. Values close to unity confirm a high goodness-of-fit across the full operating range.

• Maximum Absolute Error (MaxAE):

$$
\mathrm { M a x A E } = \operatorname* { m a x } _ { i } \left| y _ { i } - \hat { y } _ { i } \right| ,\tag{20}
$$

identifies the single largest prediction error across all buses and test samples. In operational contexts, worst-case accuracy is often more consequential than average accuracy, since a single large voltage or power error can violate security limits.

## V. RESULTS AND DISCUSSION

## A. Simulation Setup

A systematic study was conducted along two axes. First, a hyperparameter search varied the number of GNN layers over 2, 4, 6 and the hidden dimension over 32, 64, 128, 256. Second, an ablation study individually removed the residual skip connection $h ^ { ( 0 ) }$ , global mean pooling, and Layer Normalization (NormLayer) to quantify the contribution of each architectural component.

The selected configuration was then evaluated on both the IEEE 14-bus and 118-bus test systems, using a combined dataset of 3000 samples, 1000 samples for each problem (PF/OPF/SE ), drawn from the mid scenario and partitioned into 80% training, 10% validation, and 10% test sets. To assess out-of-distribution generalization, the trained model on the mid scenario was further evaluated on the narrow, wide, and hightopo datasets, which include unseen loading conditions and network topologies, where existing lines can be removed or new lines added.

## B. Results

Table III and Figs. 5-8 present the quantitative results of the hyperparameter search, ablation study, and generalization evaluation.

Fig. 3 compares the average NRMSE, computed as the mean across all predicted quantities $( V _ { m a g } , V _ { a n g } , P _ { g e n } , Q _ { g e n } , P _ { d }$ and $Q _ { d } )$ , for seven GNN cell variants: GAT, Transformer, ResGated GCN, GraphSAGE, GPS, GiNE, and MPNN. All variants are evaluated within the proposed heterogeneous architecture, using a 300 samples dataset, for the 118-bus network under the mid scenario. ResGated GCN achieves the lowest average NRMSE and is therefore selected as the computational cell for all subsequent experiments.

![](images/3706afa33bea5b55a96492c45d34e70ab123add3f14521fd7fbe3f836e01dad5.jpg)  
Fig. 4. GNN cell comparison showing average NRMSE.

Table III summarizes the hyperparameter search and ablation results for the 118-bus network, trained on the mid scenario and tested on both the mid and high-topo scenarios. Inference times are reported relative to the fastest configuration to remove hardware-dependent bias. Fig. 7 complements the table by showing the average NRMSE graphically for each ablated configuration. Fig 5 shows two prediction cases, which exemplify two problems, OPF and SE, using two different networks, IEEE 14-bus and IEEE 118-bus network, both trained and tested on the mid scenario. Fig. 6 shows the training and validation loss evolution over 250 epochs for the IEEE 118-bus network, trained on the mid scenario, the loss is plotted on a logarithmic scale, encompassing PF, OPF, and SE simultaneously. Both curves converge smoothly without signs of overfitting. Fig. 8 shows PF results for the 118-bus network evaluated across all four scenarios (narrow, mid, wide, high-topo), confirming that prediction errors remain small and tightly distributed around zero even under conditions not seen during training.

## C. Discussion

The ablation study demonstrates that not each architectural component contributes meaningfully to overall accuracy. Removal of global mean pooling produces the largest accuracy drop, increasing average NRMSE by approximately 16% relative to the baseline (from 0.0031 to 0.0036), underscoring its importance for tasks such as OPF that require awareness of system-wide constraints. The local residual connection has negligible effect on in-distribution accuracy but slightly improves generalization to unseen scenarios, approximately 2%. In contrast, removing the root residual $h ^ { ( 0 ) }$ , the skip connection to the initial node encoding, reduces in-distribution accuracy, by 6.54%, while improving out-of-distribution performance, 4,9%. This confirms that residual connections are critical for mitigating the oversmoothing that degrades deep GNN representations, and that their benefit is most apparent when the model is evaluated on scenarios outside the training distribution. Contrary to expectation, removal of Layer Normalization marginally improves accuracy in both scenarios, despite its expected role in stabilizing training and preventing weight explosion. This suggests that, at the scale of the IEEE test systems used here, the regularization provided by Layer Normalization may be unnecessary, though it may become more important as the architecture is scaled to larger networks.

Increasing the hidden dimension consistently improves prediction accuracy up to a size of 128, beyond which gains diminish while inference time continues to grow. The number of GNN layers has a less consistent effect: additional layers improve in-distribution accuracy but degrade generalization to unseen scenarios, likely due to overfitting, while also increasing inference time. These trade-offs motivate the selection of 4 layers and a hidden size of 128 as the optimal configuration.

The relative inference time (test inference time/baseline inference time) reported in Table III scales moderately with model depth and hidden dimension, with the selected configuration (4 layers, hidden size 128) achieving a favorable accuracy–efficiency trade-off.

![](images/671495293e3d7b1beb56eddb41acd49ca0d7992b1453cb0ef6c0286f759e98e9.jpg)

![](images/7e5df00a6fb554975e1a8710a638f6a36ebd2ef5a8b30b540fbb83aa7a7cb298.jpg)

![](images/11c497ac24c6c10147363f54c899f7cd5c810cf77ff70f1a5f9ad820b01f7eab.jpg)

![](images/36c34e90445a9fdefe880f1cbf44d9e5e1a61020546d80cf7cdca9bc4aafefdb.jpg)

![](images/3c1d93252273fc319e7409c8492084046927c9c3248605dbed1e73ad2fbe3183.jpg)

![](images/519ce2677282ce1bc3ad47f69c899ce7f31ff0d44d65b95bba8514c4f8ded816.jpg)

![](images/0ffe5b840d1bbdd0338b1ad165ebf551b3171d009984eea3a365a7a368da2682.jpg)

![](images/384b64c18f65ecfcd75e3e31c9468cbdc392415aa6d48cfa09772e7074e1ed6f.jpg)  
Fig. 5. Prediction vs. ground truth scatter plots for the OPF task (IEEE 14-bus network, top row) and the SE task (IEEE 118-bus network, bottom row), both trained and evaluated on their respective mid scenario datasets.

TABLE III  
COMPARISON OF PREDICTION METRICS ACROSS CONFIGURATIONS. BASE LINE: 4 GNN LAYERS AND HIDDEN DIMENSION OF 64.
<table><tr><td rowspan="2">Case</td><td rowspan="2">inference time</td><td rowspan="2">avg NRMSE</td><td colspan="4">Vmag</td><td>Vang</td><td>Pgen</td><td>Qgen</td><td>Pd</td><td></td><td>Qd</td></tr><tr><td>NRMSE</td><td>MAE</td><td>R2</td><td>MSE</td><td></td><td>NRMSE</td><td>NRMSE</td><td>NRMSE</td><td>NRMSE</td><td>NRMSE</td></tr><tr><td>Mid Scenario</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base line</td><td>1</td><td>0.0031</td><td>0.0056 ± 0.000</td><td>0.001</td><td>0.998</td><td>2.0587e</td><td>06</td><td>0.0033 ± 0.000</td><td>0.0047 ± 0.000</td><td>0.0024 ± 0.000</td><td>0.0015 ± 0.000</td><td>0.0011 ± 0.000</td></tr><tr><td>2 GNN layers</td><td>0.8</td><td>0.0034</td><td>0.0058 ± 0.000</td><td>0.001</td><td>0.998</td><td>2.1972e</td><td>06</td><td>0.0038 ± 0.000</td><td>0.0052 ± 0.000</td><td>0.0028 ± 0.000</td><td>0.0016 ± 0.000</td><td>0.0013 ± 0.000</td></tr><tr><td>6 GNN layers</td><td>1.73</td><td>0.0030</td><td>0.0052 X 0.001</td><td>0.001</td><td>0.999</td><td>1.7727e</td><td>06</td><td>0.0034 ± 0.000</td><td>0.0046 ± 0.000</td><td>0.0023 ± 0.000</td><td>0.0017 ± 0.000</td><td>0.0011 ± 0.000</td></tr><tr><td>32 hidden dim.</td><td>0.86</td><td>0.0037</td><td>0.0065 ± 0.000</td><td>0.001</td><td>0.998</td><td>2.7967e</td><td>06</td><td>0.0041 ± 0.000</td><td>0.0053 ± 0.000</td><td>0.0028 ± 0.000</td><td>0.0020 ± 0.000</td><td>0.0015 ± 0.000</td></tr><tr><td>128 hidden dim.</td><td>1.2</td><td>0.0029</td><td>0.0050 ± 0.000</td><td>0.001</td><td>0.999</td><td>1.6324e</td><td>06</td><td>0.0029 ± 0.000</td><td>0.0048 ± 0.000</td><td>0.0023 ± 0.000</td><td>0.0015 ± 0.000</td><td>0.0010 ± 0.000</td></tr><tr><td>256 hidden dim.</td><td>1.66</td><td>0.0029</td><td>0.0051 ± 0.000</td><td>0.001</td><td>0.999</td><td>1.6871e</td><td>06</td><td>0.0029 ± 0.000</td><td>0.0048 ± 0.000</td><td>0.0023 ± 0.000</td><td>0.0015 ± 0.000</td><td>0.0010 ± 0.000</td></tr><tr><td>no Global Pool</td><td>0.93</td><td>0.0036</td><td>0.0076 ± 0.000</td><td>0.001</td><td>0.997</td><td>3.8088e</td><td>06</td><td>0.0033 ± 0.000</td><td>0.0053 ± 0.000</td><td>0.0025 ± 0.000</td><td>0.0017 ± 0.000</td><td>0.0012 ± 0.000</td></tr><tr><td>no Residual</td><td>0.97</td><td>0.0031</td><td>0.0051 ± 0.000</td><td>0.001</td><td>0.999</td><td>1.7124e</td><td>06</td><td>0.0031 ± 0.000</td><td>0.0050 ± 0.000</td><td>0.0026 ± 0.000</td><td>0.0016 ± 0.000</td><td>0.0009 ± 0.000</td></tr><tr><td>no Root</td><td>0.97</td><td>0.0029</td><td>0.0050 ± 0.000</td><td>0.001</td><td>0.999</td><td>1.6401e</td><td>06</td><td>0.0030 ± 0.000</td><td>0.0047 ± 0.000</td><td>0.0024 ± 0.000</td><td>0.0015 ± 0.000</td><td>0.0010 0.000</td></tr><tr><td>no NormLayer</td><td>0.98</td><td>0.0030</td><td>0.0053 ± 0.000</td><td>0.001</td><td>0.999</td><td>1.8427e</td><td>06</td><td>0.0029 ± 0.000</td><td>0.0050 ± 0.000</td><td>0.0024 ± 0.000</td><td>0.0015 ± 0.000</td><td>0.0010 ± 0.000</td></tr><tr><td>High-topo Scenario</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base line</td><td></td><td>0.0631</td><td>0.0523 ± 0.003</td><td>0.007</td><td>0.840</td><td>1.6655e</td><td>04</td><td>0.1088 ± 0.003</td><td>0.1026 ± 0.004</td><td>0.0719 ± 0.002</td><td>0.0218 ± 0.002</td><td>0.0209 ± 0.002</td></tr><tr><td>2 GNN layers</td><td></td><td>0.0613</td><td>0.0519 ± 0.004</td><td>0.007</td><td>0.842</td><td>1.6435e</td><td>04</td><td>0.1061 ± 0.002</td><td>0.1090 ± 0.002</td><td>0.0729 ± 0.004</td><td>0.0149 ± 0.002</td><td>0.0129 ± 0.001</td></tr><tr><td>6 GNN layers</td><td></td><td>0.0640</td><td>0.0572 ± 0.002</td><td>0.008</td><td>0.809</td><td>1.9879e</td><td>04</td><td>0.1036 ± 0.008</td><td>0.1075 ± 0.003</td><td>0.0696 ± 0.003</td><td>0.0233 ± 0.000</td><td>0.0228 ± 0.002</td></tr><tr><td>32 hidden dim.</td><td></td><td>0.0635</td><td>0.0548 ± 0.002</td><td>0.007</td><td>0.824</td><td>1.8273e</td><td>04</td><td>0.1039 ± 0.006</td><td>0.1065 ± 0.010</td><td>0.0742 ± 0.008</td><td>0.0224 ± 0.001</td><td>0.0191 ± 0.000</td></tr><tr><td>128 hidden dim.</td><td></td><td>0.0639</td><td>0.0539 ± 0.002</td><td>0.007</td><td>0.830</td><td>1.7683e</td><td>04</td><td>0.1153 ± 0.004</td><td>0.0987 ± 0.002</td><td>0.0709 ± 0.002</td><td>0.0214 ± 0.002</td><td>0.0229 ± 0.001</td></tr><tr><td>256 hidden dim.</td><td></td><td>0.0640</td><td>0.0525 ± 0.001</td><td>0.007</td><td>0.839</td><td>1.6731e</td><td>04</td><td>0.1210 ± 0.002</td><td>0.0966 ± 0.002</td><td>0.0696 ± 0.002</td><td>0.0204 ± 0.001</td><td>0.0239 ± 0.001</td></tr><tr><td>no Global Pool</td><td></td><td>0.0653</td><td>0.0504 ± 0.004</td><td>0.007</td><td>0.851</td><td>1.5503e 1.8895e</td><td>04 04</td><td>0.1217 ± 0.005</td><td>0.1091 ± 0.018</td><td>0.0712 ± 0.006</td><td>0.0201 ± 0.000</td><td>0.0195 ± 0.001</td></tr><tr><td>no Residual</td><td></td><td>0.0646</td><td>0.0557 ± 0.004</td><td>0.007</td><td>0.818 0.828</td><td>1.7866e</td><td>04</td><td>0.1036 ± 0.003</td><td>0.1075 ± 0.006</td><td>0.0732 ± 0.005</td><td>0.0240 ± 0.002</td><td>0.0238 ± 0.001</td></tr><tr><td>no Root</td><td></td><td>0.0662</td><td>0.0542 ± 0.002</td><td>0.007</td><td></td><td></td><td></td><td>0.1130 ± 0.002</td><td>0.1090 ± 0.004</td><td>0.0741 ± 0.003</td><td>0.0229 ± 0.002</td><td>0.0239 ± 0.001</td></tr><tr><td>no NormLayer</td><td></td><td>0.0612</td><td>0.0510 ± 0.003</td><td>0.007</td><td>0.848</td><td>1.5807e</td><td>04</td><td>0.1117 7± 0.004</td><td>0.0997 ± 0.016</td><td>0.0715 ± 0.005</td><td>0.0154 ± 0.001</td><td>0.0177 ± 0.002</td></tr></table>

![](images/6dcbd38642edd76cf14065b62e0c7fd5caf20e7a2e1b6b69c735a05f29a7711f.jpg)  
Fig. 6. Training and validation loss evolution over 250 epochs for the IEEE 118-bus network, trained on the mid scenario.

The generalization results in Fig. 8 show that a model trained exclusively on the mid scenario retains acceptable accuracy, with MSE ranging from 1.34e-7 to 3.55e-6, when evaluated on the narrow, wide, and high-topo datasets, including load levels up to 250% of nominal and topologies modified by branch additions and removals, none of which were seen during training. This out-of-distribution robustness is a key property for operational deployment, where realtime grid conditions may deviate significantly from historical operating points. Notably, the unified model achieves accuracy comparable to, and in some cases surpassing, task-specific GNN solvers [12], [16], [33], despite being trained across three simultaneous tasks; however, direct comparison is complicated by differences in evaluation metrics (MSE, R², NRMSE, TRMAE) and benchmark networks across studies.

![](images/0c86245c11243556bc30b686b663fa5d8a30fe6b300e654f7edc163dfc5a7f36.jpg)  
Fig. 7. Ablation study results showing average NRMSE.

![](images/82e8a021cbbb065c656f2f822f21046d7387ce6223b52a48400fab4e69a0c698.jpg)  
(a) narrow

![](images/48b594722de8f21962bc05fd49cc6a01248af5b724b84f730cdebf9ea8fe1e14.jpg)  
(b) mid

![](images/d3bc398f3808c9a2b5517503a4206f68247f47504ddd45e6035e323f5615b277.jpg)  
(c) wide

![](images/82823aeea8992a7b4c77751166080e8a45bc9411bdde31d559fa78871511d323.jpg)  
(d) high-topo  
Fig. 8. Voltage magnitude prediction vs. ground truth scatter plots for the IEEE 118-bus network, evaluated under four scenarios. The model was trained exclusively on the mid scenario.

## VI. CONCLUSION

This paper has demonstrated that a single Heterogeneous ResGated GCN architecture can simultaneously solve PF, OPF, and SE to accuracy comparable to task-specific models, while requiring only one training pipeline. By integrating line impedance parameters (R,X,B) directly into the messagepassing mechanism and augmenting node updates with a global context MLP, the architecture captures both local KCL physics and system-wide optimization constraints within a unified representation.

Our results demonstrate that this multi-task framework achieves high precision on PF and SE while producing feasible solutions within the constrained optimization setting of OPF. Specifically, the selected configuration achieves average NRMSE values below 0.004 (Table III) across all predicted quantities on the mid scenario, and generalizes to unseen loading conditions and topological configurations without retraining. The performance of the model is comparable to state-ofthe-art task-specific GNNs, yet it offers superior versatility by eliminating the need for redundant, task-isolated architectures. Furthermore, the inclusion of topological perturbations and extreme load scaling (up to 250%) supports robustness to grid reconfigurations and stressed operating conditions.

Future research is going to work to move the current implementation toward foundational models for power systems. This work establishes that a single GNN-based architecture can internalize shared physical principles, motivating future large-scale pre-training on global utility datasets. Such advancements will be critical for the autonomous operation and real-time control of the next generation of highly dynamic and decentralized power grids.

## A. Towards a Foundational Model for Power Systems

The proposed GNN architecture demonstrates the capability to solve PF, OPF, and SE within a single unified model (Table III). We characterize this, however, not as a fully realized foundational model, but as an initial step which can internalize the shared physics of these three fundamental problems. This work advances three key prerequisites toward that goal:

1) Multi-Task Consolidation: By merging descriptive tasks (PF, SE) with prescriptive optimization (OPF), we move away from specialized models toward a general-purpose grid solver.

2) Topological Invariance via ResGated GCN: The use of edge-conditioned message-passing layers, with strong graph-discriminative power, ensures that the model learns the underlying physics of KCL rather than the specific geometry of a single test case, a prerequisite for foundational transfer learning.

3) Physics-Informed Latent Space: The integration of global context and deep residual paths allows the model to handle the high-dimensional complexity inherent in the global energy landscape.

Consequently, this research serves as a proof of concept for a unified foundational power system model. Scaling this framework to encompass thousands of heterogeneous utilityscale systems would represent a concrete step toward a truly foundational model for autonomous power system operations.

## REFERENCES

[1] J. J. Grainger and W. D. Stevenson, Power System Analysis, ser. McGraw-Hill Series in Electrical and Computer Engineering. New York, NY, USA: McGraw-Hill, 1994.

[2] W. F. Tinney et al., “Power flow solution by Newton’s method,” IEEE Trans. Power App. Syst., vol. PAS-86, no. 11, pp. 1449–1460, Nov. 1967.

[3] F. Capitanescu, “Critical review of recent advances and further developments needed in AC optimal power flow,” Electr. Power Syst. Res., vol. 136, pp. 57–68, Jul. 2016.

[4] F. C. Schweppe and J. Wildes, “Power system static-state estimation, part I: Exact model,” IEEE Trans. Power App. Syst., vol. PAS-89, no. 1, pp. 120–125, Jan. 1970.

[5] M. L. Crow, Computational Methodsfor Electric Power Systems, 3rd ed. Boca Raton, FL, USA: CRC Press, 2015.

[6] A. Marot et al., “Learning to run a power network challenge for training topology controllers,” 2019, arXiv:1912.04211.

[7] T. Pham et al., “Neural network-based power flow model,” in Proc. IEEE Green Technol. Conf., Mar. 2022, pp. 105–109.

[8] F. Scarselli, M. Gori, Ah Chung Tsoi, M. Hagenbuchner, and G. Monfardini, “The graph neural network model,” IEEE Trans. Neural Netw., vol. 20, no. 1, pp. 61–80, Jan. 2009.

[9] J. Gilmer, O. Vinyals et al., “Neural message passing for quantum chemistry,” 2017, arXiv:1704.01212.

[10] P. W. Battaglia, O. Vinyals et al., “Relational inductive biases, deep learning, and graph networks,” 2018, arXiv:1806.01261.

[11] T. B. Lopez-Garcia et al., “Power flow analysis via typed graph neural networks,” Eng. Appl. Artif. Intell., vol. 117, p. 105567, Jan. 2023.

[12] N. Lin et al., “PowerFlowNet: Power flow approximation using message passing graph neural networks,” Int. J. Electr. Power Energy Syst., vol. 160, p. 110112, Sep. 2024.

[13] D. Owerko, F. Gama, and A. Ribeiro, “Optimal power flow using graph neural networks,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), May 2020, pp. 5930–5934.

[14] B. Donon et al., “Neural networks for power flow: Graph neural solver,” Electr. Power Syst. Res., vol. 189, p. 106547, Dec. 2020.

[15] T. B. Lopez-Garcia et al., “Optimal power flow with physics-informed typed graph neural networks,” IEEE Trans. Power Syst., pp. 1–14, 2024.

[16] A. Varbella et al., “Physics-informed GNN for non-linear constrained optimization: PINCO a solver for the AC-optimal power flow,” Oct. 2024, arXiv:2410.04818.

[17] A. Saur, “Graph neural networks for electrical grid state estimation,” Master’s thesis, University of Freiburg, Mar. 2024.

[18] O. Kundacina et al., “State estimation in electric power systems leveraging graph neural networks,” in Proc. 17th Int. Conf. Probab. Methods Appl. Power Syst. (PMAPS), Jun. 2022, pp. 1–6.

[19] K. Xu et al., “How powerful are graph neural networks?” 2019, arXiv:1810.00826.

[20] X. Bresson and T. Laurent, “Residual gated graph ConvNets,” Apr. 2018, arXiv:1711.07553.

[21] GridFM, “GridFM,” [Online], May 2025, available: https://gridfm.org/.

[22] Cresym, “AI.grids,” [Online], available: https://cresym.eu/ai-grids/.

[23] W. Yang and Andrea Britto Mattos Lima, “GridSFM: Small foundation model boosts power grid optimization,” Microsoft Research Blog, [Online], May 2026, available: https://www.microsoft.com/en-us/research/ blog/gridsfm-a-new-small-foundation-model-for-the-electric-grid/.

[24] Power Systems Test Case Archive, “IEEE power flow test case,” [Online], 1993, available: https://labs.ece.uw.edu/pstca/.

[25] J. Carpentier, “Contribution à l’étude du dispatching économique,” Bull. Soc. Fr. Électr., 1962.

[26] A. Monticelli, State Estimation in Electric Power Systems. Boston, MA, USA: Springer, 1999.

[27] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and locally connected networks on graphs,” May 2014, arXiv:1312.6203.

[28] P. M. Anderson and A.-A. A. Fouad, Power System Control and Stability, 1st ed. Ames, IA, USA: Iowa State University Press, 1977.

[29] A. Sanchez-Gonzalez et al., “Learning to simulate complex physics with graph networks,” in Proc. 37th Int. Conf. Mach. Learn. (ICML), Nov. 2020, pp. 8459–8468.

[30] pandapower, “pandapower,” [Online], available: https://www. pandapower.org/.

[31] VeraGrid, “VeraGrid,” [Online], available: https://github.com/SanPen/ VeraGrid.

[32] S. Jadhav et al., “Enhancing power flow estimation with topology-aware gated graph neural networks,” Jul. 2025, arXiv:2507.02078.

[33] O. Arowolo and J. L. Cremer, “Towards generalization of graph neural networks for AC optimal power flow,” 2025, arXiv:2510.06860.