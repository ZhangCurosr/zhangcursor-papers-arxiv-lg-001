# Beamforming Design Via GNN in mmWave Cell-Free Massive MIMO Using Sub-6 GHz CSI

Sina Tavakolian<sup>⋆</sup>, Abolfazl Zakeri<sup>⋆</sup>, Ahmed Alkhateeb<sup>†</sup>, Markku Juntti<sup>⋆</sup>, and Nhan Thanh Nguyen<sup>⋆</sup>,

⋆ Centre for Wireless Communications, University of Oulu, P.O.Box 4500, FI-90014, Finland

<sup>†</sup>School of Electrical, Computer, and Energy Engineering, Arizona State University, AZ, USA

{sina.tavakolian, abolfazl.zakeri, nhan.nguyen, markku.juntti}@oulu.fi, alkhateeb@asu.edu

Abstract—Beamforming methods in millimeter-wave (mmWave) cell-free massive multiple-input multiple-output (CFmMIMO) systems require accurate channel state information (CSI), whose acquisition entails significant training overhead. This paper shows that fully digital cell-free mmWave beamforming can be effectively learned from sub-6 GHz CSI using a graph neural network (GNN). Specifically, we represent a CFmMIMO system as a wireless graph, and The GNN is trained to approximate beamformers that maximize the downlink sum-rate, based on the available sub-6 GHz CSI. A message passing mechanism is proposed to capture inter-user interference and inter-basestation cooperation across different network topologies. Simulation results demonstrate that the proposed sub-6 GHz–assisted GNN-based beamformer achieves competitive and often superior sum-rate performance compared to classical baselines that rely on full mmWave CSI.

Index Terms—Beamforming, graph neural networks, out-ofband assisted communication, millimeter-wave, cell-free systems.

## I. INTRODUCTION

Cell-free massive multiple-input multiple-output (CFm-MIMO) has emerged as a promising architecture for future wireless networks, where multiple distributed base stations (BSs) jointly serve user equipments (UEs) and are all connected to a central processing unit (CPU) [1], [2]. When operating at millimeter-wave (mmWave) frequencies, however, beamforming becomes essential due to the severe propagation loss and the use of large antenna arrays [3]. Conventional mmWave beamforming designs typically rely on channel state information (CSI). However, acquiring mmWave CSI in CFm-MIMO systems entails significant training overhead.

Several works have explored the use of out-of-band (OOB) side information, particularly sub-6 GHz CSI [4]–[8]. Such sub-6 GHz channels can be estimated much more efficiently than mmWave channels using conventional pilot-based procedures, making them a useful source of side information for mmWave beamforming design. Early studies exploited crossband spatial correlation for beam selection [4], while more recent machine learning (ML)-based approaches use neural networks to predict mmWave beams directly from sub-6 GHz CSI [5], [6]. Particularly, graph neural network (GNN)-based methods have also been explored for OOB-assisted mmWave beamforming [7], [8]. However, existing OOB-assisted approaches, including [7], [8], primarily focus on single-BS or multi-cell systems. Specifically, [7] considers only a single BS, whereas in [8], each UE is assigned to one predetermined mmWave BS, allowing the graph edges to be a priori classified as desired or interfering links; moreover, they assume partial mmWave CSI is also required. These assumptions do not apply to CFmMIMO systems, where multiple distributed BSs jointly serve each UE without a fixed desired/interferinglink partition. Therefore, beamforming design for CFmMIMO systems using solely sub-6 GHz CSI remains unexplored.

To address the above gap, in this work, we study downlink beamforming design in a CFmMIMO system where only uplink sub-6 GHz CSI is available at the CPU. We formulate the beamforming problem as an ML task and develop a GNN framework that learns digital beamformers directly from sub-6 GHz CSI. The CFmMIMO system is modeled as a graph, and the proposed GNN performs customized message passing over this graph to capture both multi-user interference at each BS and cooperative transmission across BSs. Importantly, the proposed framework is trained end-toend to directly maximize the sum-rate, without relying on supervised labels (e.g., optimal beamforming vectors). Moreover, the graph-based formulation allows the learned model to naturally adapt to different and dynamic network topologies, e.g., varying number of users. Numerical results demonstrate that the learned beamformers achieve performance close to the baselines that rely on full mmWave CSI, including minimum mean-square error (MMSE) beamforming. Moreover, the proposed framework maintains good performance across varying network topologies and remains robust when only partial sub-6 GHz CSI is available.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

## A. System Model

We consider a downlink cell-free dual-band single-carrier communications system. The system consists of B distributed BSs connected to a CPU and jointly serving U UEs. Each BS is equipped with two co-located antenna arrays: a sub-6 GHz array with N<sup>s</sup> antennas, and a mmWave array with N <sup>mW</sup> antennas. Each UE employs a single antenna at both sub-6 GHz and mmWave frequency bands. Downlink data transmission is carried out in the mmWave band.

We adopt the geometric channel model in [5] for both the sub-6 GHz uplink and the mmWave downlink. Although the propagation environment is wideband and includes multiple paths with distinct delays, as described in [5, Eq. (3)], the following processing and analysis are performed over a single subcarrier for simplicity. Let $\mathbf { h } _ { b , u } ^ { \mathrm { s } } \in \mathbb { C } ^ { N ^ { \mathrm { s } } \times 1 }$ and $\mathbf { h } _ { b , u } ^ { \operatorname* { m W } } \in \mathbb { C } ^ { N ^ { \operatorname* { m W } } \times 1 }$ denote the corresponding sub-6 GHz and mmWave channel vectors, respectively. Their parameters are generated through 3D ray-tracing simulations, which account for realistic frequency-dependent propagation effects and preserve spatial consistency across the two frequency bands, as detailed in [5].

Let $\mathbf { s } _ { b } ~ = ~ [ s _ { b , 1 } , s _ { b , 2 } , \ldots , s _ { b , U } ] ^ { \mathsf { T } } ~ \in ~ \mathbb { C } ^ { U \times 1 }$ denote the data symbol vector transmitted by BS b, where $s _ { b , u }$ is the data symbol intended for UE $u \in \mathcal { U }$ and $\mathbb { E } [ { \mathbf { s } } _ { b } { \mathbf { s } } _ { b } ^ { \mathsf { H } } ] = { \mathbf { I } } _ { U }$ . Each BS employs a beamformer $\mathbf { F } _ { b } \in \mathbb { C } ^ { N ^ { \mathrm { { m W } } } \times U }$ to precode the transmitted signal, resulting in the transmit vector $\mathbf { F } _ { b } \mathbf { s } _ { b }$ . The received signal at UE $u \in \mathcal { U }$ is then given by

$$
y _ { u } = \sum _ { b = 1 } ^ { B } \mathbf { h } _ { b , u } ^ { \mathrm { m W } ^ { \sf H } } \mathbf { F } _ { b } \mathbf { s } _ { b } + n _ { u } ,\tag{1}
$$

where $n _ { u } \sim \mathcal { C N } ( 0 , \sigma ^ { 2 } )$ denotes the additive white Gaussian noise. Assuming that symbol $s _ { b , u }$ is intended for UE u from BS b, we can write:

$$
y _ { u } = \underbrace { \sum _ { b = 1 } ^ { B } { \bf h } _ { b , u } ^ { \mathrm { m W } ^ { \sf H } } { \bf f } _ { b , u } s _ { b , u } } _ { \mathrm { D e s i r e d ~ s i g n a l } } + \underbrace { \sum _ { b = 1 } ^ { B } \sum _ { j \in \mathcal { U } } { \bf h } _ { b , u } ^ { \mathrm { m W } ^ { \sf H } } { \bf f } _ { b , j } s _ { b , j } } _ { \mathrm { I n t e r - U E ~ i n t e r f e r e n c e } } + n _ { u } ,\tag{2}
$$

where $\mathbf { f } _ { b , u }$ denotes the u-th column of $\mathbf { F } _ { b }$ . The downlink SINR at UE u is therefore given as

$$
\gamma _ { u } = \frac { \left| \sum _ { b = 1 } ^ { B } { \bf h } _ { b , u } ^ { \mathrm { m W } } ^ { \sf H } { \bf f } _ { b , u } \right| ^ { 2 } } { \sum _ { \substack { j \in \mathcal { U } , j \neq u } } \left| \sum _ { b = 1 } ^ { B } { \bf h } _ { b , u } ^ { \mathrm { m W } } ^ { \sf H } { \bf f } _ { b , j } \right| ^ { 2 } + \sigma ^ { 2 } } .\tag{3}
$$

Accordingly, the achievable downlink rate of UE u is $R _ { u } = \log _ { 2 } ( 1 + \gamma _ { u } )$ . The sum-rate of the system is then given by

$$
R ^ { \mathrm { s u m } } = \sum _ { u = 1 } ^ { U } R _ { u } .\tag{4}
$$

## B. Problem Formulation

The goal is to design the fully digital beamformers $\{ \mathbf { F } _ { b } \} _ { b = 1 } ^ { B }$ 1 that maximize the downlink sum-rate in (4). The beamforming design problem can be written as

$$
\operatorname* { m a x i m i z e } _ { \{ \mathbf { F } _ { b } \} _ { b = 1 } ^ { B } } \quad R ^ { \mathrm { s u m } }\tag{5a}
$$

$$
\mathrm { s u b j e c t ~ t o ~ } \| \mathbf { F } _ { b } \| _ { F } ^ { 2 } = P _ { b } , \quad \forall b ,\tag{5b}
$$

where $P _ { b }$ denotes the power budget at BS b. The transmit power constraint is enforced as an equality in (5b) because, for sum-rate maximization, the optimal solution often utilizes the fully available transmit power [9].

Solving problem (5) directly would require the mmWave downlink CSI, which is supposed to be unavailable in our considered scenario. Instead, only the uplink sub-6 GHz CSI is available at the CPU, and we aim to solve the design problem centrally using this information. This assumption is motivated by the fact that the fully digital sub-6 GHz arrays provide direct baseband access to all antenna signals, enabling efficient pilot-based CSI estimation with low overhead, whereas estimating the mmWave channel typically involves extensive beam training [10].

The uplink sub-6 GHz CSI is collected as

$$
\mathbf { H } ^ { \mathrm { s } } = \left[ \mathbf { H } _ { 1 } ^ { \mathrm { s } } , \ldots , \mathbf { H } _ { B } ^ { \mathrm { s } } \right] \in \mathbb { C } ^ { B \times U \times N ^ { \mathrm { s } } } ,\tag{6}
$$

where

$$
\mathbf { H } _ { b } ^ { \mathrm { s } } = \left[ \begin{array} { c } { ( \mathbf { h } _ { b , 1 } ^ { \mathrm { s } } ) ^ { \top } } \\ { \vdots } \\ { ( \mathbf { h } _ { b , U } ^ { \mathrm { s } } ) ^ { \top } } \end{array} \right] \in \mathbb { C } ^ { U \times N ^ { \mathrm { s } } } .\tag{7}
$$

To perform beamforming design in (5) without the mmWave CSI, we aim to learn a mapping from the available sub-6 GHz CSI to the digital beamformers, i.e.,

$$
\Phi ^ { \mathrm { s } } ( { \bf H } ^ { \mathrm { s } } ) = \{ { \bf F } _ { b } \} _ { b = 1 } ^ { B } ,\tag{8}
$$

where $\Phi ^ { \mathrm { s } } ( \cdot )$ is learned such that the resulting beamformer maximizes the sum-rate in (4) while satisfying the per-BS power constraint in (5b). Finding the mapping $\Phi ^ { \mathrm { s } } ( \cdot )$ in closed form is analytically intractable [5]. This motivates the use of a data-driven ML framework that approximates $\Phi ^ { \mathrm { s } } ( \cdot )$ directly from sub-6 GHz channels. GNNs provide a suitable ML architecture for this problem, as they can effectively model interactions among multiple BSs and UEs whose wireless channels jointly determine the beamforming decisions. To this end, we develop a GNN-based framework that learns the mapping in (8) and generalizes to different network topologies, as described next.

## III. PROPOSED GNN-BASED FRAMEWORK

We model the considered CF-mMIMO system as a wireless graph, defined as $\mathcal { G } : = ( \nu , \mathcal { E } , \mathcal { T } _ { \nu } )$ , where the node set is given by $\begin{array} { r } {  { \mathcal { V } } \ = \ \mathcal { V } _ { \mathrm { B S } } \cup \mathcal { V } _ { \mathrm { U E } } . } \end{array}$ , consisting of the BS nodes $\mathcal { V } _ { \mathrm { B S } } = \{ 1 , \dots , B \}$ and UE nodes $\mathcal { V } _ { \mathrm { U E } } = \{ 1 , \dots , U \}$ . The edge set $\mathcal { E } ~ = ~ \{ ( b , u ) ~ \mid ~ b ~ \in ~ \mathcal { V } _ { \mathrm { B S } }$ $u \in \partial _ { \mathrm { U E } } \}$ captures all communication links between BSs and UEs. The nodetype mapping is specified by $\mathcal { T } _ { \mathcal { V } } ~ = ~ \{ \mathrm { B S } , \mathrm { U E } \}$ . For each edge $( b , u ) \in { \mathcal { E } } ,$ , we assign a feature vector extracted from the corresponding sub-6 GHz channel $\mathbf { h } _ { b , u } ^ { \mathrm { s } } .$ . Furthermore, each BS and UE node is associated with a learnable embedding vector, denoted by ${ \rho } _ { b } ~ \in ~ \mathbb { R } ^ { d }$ and $\rho _ { u } ~ \in ~ \mathbb { R } ^ { d }$ , respectively. These embeddings are later optimized jointly with the GNN parameters. The parameter d represents the dimensionality of all feature and embedding vectors.

## A. GNN Architecture

With the graph representation described above, we aim to design a GNN that generates the beamformers efficiently. GNNs operate by iteratively propagating information across the graph through message passing, enabling each element, i.e., each node or edge, to incorporate information from its neighbors [11], [12]. Following this principle, the proposed GNN method consists of three phases:

1) Initialization (Phase $I ) .$ Let $\mathbf { z } _ { b } ^ { ( t ) }$ and $\mathbf { z } _ { u } ^ { ( t ) }$ denote the hidden states of BS b and UE u at message-passing iteration t, respectively. Similarly, let $\mathbf { e } _ { b , u } ^ { ( t ) }$ represent the hidden state of edge (b, u). At the initialization step, i.e., $t \ = \ 0 ,$ the node hidden states are set to the learnable embeddings introduced in the previous subsection, while the edge hidden states are obtained from the sub-6 GHz channel features, i.e.,

$$
\begin{array} { r } { \mathbf { z } _ { b } ^ { ( 0 ) } = \pmb { \rho } _ { b } , \quad \mathbf { z } _ { u } ^ { ( 0 ) } = \pmb { \rho } _ { u } , \quad \mathbf { e } _ { b , u } ^ { ( 0 ) } = \varphi _ { \mathrm { e d g e } } ^ { \mathrm { i n i t } } \big ( \mathbf { h } _ { b , u } ^ { \mathrm { s } } \big ) , } \end{array}\tag{9}
$$

where $\varphi _ { \mathrm { e d g e } } ^ { \mathrm { i n i t } } ( \cdot )$ is a learnable affine mapping, i.e., a fully linear network without an activation layer, that projects the sub-6 GHz channel features into a d-dimensional embedding space. The initialized hidden states $\mathbf { z } _ { b } ^ { ( 0 ) } , \mathbf { z } _ { u } ^ { ( 0 ) }$ , and $\mathbf { e } _ { b , u } ^ { ( 0 ) }$ serve as the starting representations that will be progressively refined through the message passing process described next.

2) Message Passing (Phase 2): At each iteration t, the hidden states of the edges are updated by aggregating information from neighboring BS-UE edges. Our message passing mechanism is inspired by the graph attention layer with edge features (EdgeGATConv) in [13]. However, unlike [13], which aggregates edge information into node embeddings, we adapt the attention mechanism so that each edge directly aggregates information from its neighboring edges. For an edge $( b , u )$ , we consider two neighborhoods: (i) the edges connected to the same BS b, and (ii) the edges connected to the same UE u. We denote the BS-side neighborhood by $\mathcal { E } ( b ) = \{ ( b , u ^ { \prime } ) | u ^ { \prime } \in \mathcal { N } ( b ) \}$ and the UE-side neighborhood by $\mathcal { E } ( u ) = \{ ( b ^ { \prime } , u ) | b ^ { \prime } \in \mathcal { N } ( u ) \}$ . In the following equations, we use the superscripts $\mathrm { ^ { 6 6 } B S ^ { , } }$ and $^ { 6 6 } \mathrm { U E } ^ { , 9 }$ for parameters associated with the BS and UE, respectively.

We first compute attention coefficients that quantify the relevance of neighboring edges. For the BS-side neighborhood, we define the attention weights as

$$
\begin{array} { r } { \alpha _ { b , u  { v } } ^ { ( t ) } = \mathrm { s o f t m a x } \ \Big ( \mathbf { a } _ { \mathrm { B S } } ^ { \top } \sigma \big ( \mathrm { c o n c a t } ( \mathbf { W } _ { e } ^ { \mathrm { B S } } \mathbf { e } _ { b , u } ^ { ( t ) } , \mathbf { W } _ { e } ^ { \mathrm { B S } } \mathbf { e } _ { b , v } ^ { ( t ) } ,  } \\ {  \mathbf { W } _ { b } ^ { \mathrm { B S } } \mathbf { z } _ { b } ^ { ( t ) } ) \big ) \Big ) , \qquad ( 1 0 } \end{array}
$$

where $\mathbf { a } _ { \mathrm { B S } }$ is an attention vector used to score the importance of neighboring edges in the BS-side aggregation. Here, $\mathbf { W } _ { e } ^ { \mathrm { B S } } , \mathbf { W } _ { b } ^ { \mathrm { B S } } \in \mathbb { R } ^ { d \times d }$ are trainable projection matrices that transform the edge and BS-node features into a common latent space before attention is computed. The operator concat(·) denotes concatenation, and $\sigma ( \cdot )$ denotes the leaky rectified linear unit (LeakyReLU) activation function. Using these attention weights, we compute the BS-side aggregated message as

$$
\mathbf { m } _ { b , u } ^ { \mathrm { B S } } = \sum _ { ( b , v ) \in \mathcal { E } ( b ) } \alpha _ { b , u  v } ^ { ( t ) } \Big ( \mathbf { W } _ { e } ^ { \mathrm { B S } } \mathbf { e } _ { b , v } ^ { ( t ) } + \mathbf { W } _ { b } ^ { \mathrm { B S } } \mathbf { z } _ { b } ^ { ( t ) } + \mathbf { W } _ { u } ^ { \mathrm { B S } } \mathbf { z } _ { u } ^ { ( t ) } \Big ) ,\tag{11}
$$

where $\mathbf { W } _ { u } ^ { \mathrm { B S } } \in \mathbb { R } ^ { d \times d }$ transforms the UE-node features into the BS-side latent space before aggregation.

Analogously, for the UE-side neighborhood we compute UE-side attention coefficients for edge $( b , u )$ as

$$
\begin{array} { r } { \beta _ { b  w , u } ^ { ( t ) } = \mathrm { s o f t m a x } ( \mathbf { a } _ { \mathrm { U E } } ^ { \top } \sigma \big ( \mathrm { c o n c a t } ( \mathbf { W } _ { e } ^ { \mathrm { U E } } \mathbf { e } _ { b , u } ^ { ( t ) } , \mathbf { W } _ { e } ^ { \mathrm { U E } } \mathbf { e } _ { w , u } ^ { ( t ) } ,   } \\ {   \mathbf { W } _ { u } ^ { \mathrm { U E } } \mathbf { z } _ { u } ^ { ( t ) } ) \big ) ) , \qquad ( 1 2 ) } \end{array}
$$

and we compute the corresponding message as

$$
\begin{array} { r } { \mathbf { m } _ { b , u } ^ { \mathrm { U E } } = \displaystyle \sum _ { ( w , u ) \in \mathcal { E } ( u ) } \beta _ { b  w , u } ^ { ( t ) } \Big ( \mathbf { W } _ { e } ^ { \mathrm { U E } } \mathbf { e } _ { w , u } ^ { ( t ) } + \mathbf { W } _ { b } ^ { \mathrm { U E } } \mathbf { z } _ { b } ^ { ( t ) } + \mathbf { W } _ { u } ^ { \mathrm { U E } } \mathbf { z } _ { u } ^ { ( t ) } \Big ) , } \end{array}\tag{13}
$$

where $\mathbf { W } _ { b } ^ { \mathrm { U E } } , \mathbf { W } _ { e } ^ { \mathrm { U E } } , \mathbf { W } _ { u } ^ { \mathrm { U E } } \in \mathbb { R } ^ { d \times d }$ are trainable matrices. We then fuse the two aggregated messages in (11) and (13) as

$$
\mathbf { m } _ { b , u } ^ { ( t ) } = \frac { 1 } { 2 } \Big ( \mathbf { m } _ { b , u } ^ { \mathrm { B S } } + \mathbf { m } _ { b , u } ^ { \mathrm { U E } } \Big ) .\tag{14}
$$

Finally, we update the hidden state of edge $( b , u )$ using a residual rule such that

$$
\mathbf { e } _ { b , u } ^ { ( t + 1 ) } = \mathbf { W } ^ { \mathrm { u p d } } \mathbf { e } _ { b , u } ^ { ( t ) } + \mathbf { m } _ { b , u } ^ { ( t ) } .\tag{15}
$$

Here $\mathbf { W } ^ { \mathrm { u p d } }$ is a learnable matrix that linearly transforms the current edge state, forming the residual component of the update. Note that in equations (10)–(15), we omit the bias terms, following the design of EdgeGATConv in [13]. We stack T such message passing layers to progressively refine the edge representations.

3) Readout (Phase 3): After T message passing layers, for a given edge $( b , u )$ , we collect information from two groups of neighboring edges and summarize them through mean aggregation. Specifically, one group include the edges connected to the same BS b, namely $( b , u ^ { \prime } ) \ \in \ { \mathcal { E } } ( b )$ with $u ^ { \prime } \neq u ,$ and the other group is for the edges connected to the same UE $u ,$ namely $( b ^ { \prime } , u ) \in \mathcal { E } ( u )$ with $\boldsymbol { b } ^ { \prime } \neq \boldsymbol { b }$ . We construct the contextual feature vector as

$$
\begin{array} { r } { \mathbf { c } _ { b , u } = \mathrm { c o n c a t } \left( \mathbf { e } _ { b , u } ^ { ( T ) } , \mathbf { c } _ { b , u } ^ { \mathrm { i n t r a } } , \mathbf { c } _ { b , u } ^ { \mathrm { i n t e r } } \right) , } \end{array}\tag{16}
$$

where $\mathbf { c } _ { b , u } ^ { \mathrm { { i n t r a } } }$ and $\mathbf { c } _ { b , u } ^ { \mathrm { i n t e r } }$ are the mean aggregated edge features from the BS-side and UE-side neighborhoods, obtained as

$$
\mathbf { c } _ { b , u } ^ { \mathrm { i n t r a } } = \frac { 1 } { | \mathcal { E } ( b ) | - 1 } \sum _ { ( b , u ^ { \prime } ) \in \mathcal { E } ( b ) } \mathbf { e } _ { b , u ^ { \prime } } ^ { ( T ) } ,\tag{17}
$$

$$
\mathbf { c } _ { b , u } ^ { \mathrm { i n t e r } } = \frac { 1 } { | \mathcal { E } ( u ) | - 1 } \sum _ { ( b ^ { \prime } , u ) \in \mathcal { E } ( u ) } \mathbf { e } _ { b ^ { \prime } , u } ^ { ( T ) } .\tag{18}
$$

Thus, $\mathbf { c } _ { b , u }$ captures the direct BS-UE relation together with mean-aggregated contextual information from neighboring BS and UE connections.

We then transform these contextual features into beamforming vectors $\{ \mathbf { f } _ { b , u } \}$ using a mapping $\varphi ^ { \mathrm { o u t } } ( \cdot )$ implemented by a multilayer perceptron (MLP), i.e.,

$$
\mathbf { f } _ { b , u } = \varphi ^ { \mathrm { o u t } } ( \mathbf { c } _ { b , u } ) \in \mathbb { C } ^ { N ^ { \mathrm { m W } } } .\tag{19}
$$

We form the digital beamformer at BS b as $\mathbf { F } _ { b } = \left\lceil \mathbf { f } _ { b , 1 } , \ldots , \mathbf { f } _ { b , U } \right\rceil$ . Finally, we normalize $\mathbf { F } _ { b }$ to satisfy the per-BS transmit power constraint in (5b) as

$$
\mathbf { F } _ { b } \gets \sqrt { \frac { P _ { b } } { \lVert \mathbf { F } _ { b } \rVert _ { F } ^ { 2 } } } \mathbf { F } _ { b } .\tag{20}
$$

## B. Offline Training and Online Operation

Offline Training: The proposed GNN is trained in a selfsupervised manner by directly optimizing the communication performance metric, without relying on labelled beamforming targets. We train the network to maximize the downlink sum-rate defined in (4), and the loss function is defined as the negative sum-rate, i.e., $\mathcal { L } \ = \ -  { R ^ { \mathrm { s u m } } }$ . The overall training procedure of the proposed framework is summarized in Algorithm 1. After constructing the dataset from the sub-6 GHz CSI as shown in steps 1–5, the GNN performs a forward pass following the architecture in Section III-A, where node and edge states are initialized as in step 9 and refined through $T$ message-passing layers, following steps 10–14. The readout stage generates the beamforming vectors, which are normalized to satisfy the per-BS power constraints, according to steps 15–18. Given the beamformers obtained from the forward pass, the achievable rates and the loss function are computed. For this purpose, the mmWave CSI is required only during the training phase. Let the global mmWave CSI be

$$
\mathbf { H } ^ { \mathrm { m W } } = \left[ \mathbf { H } _ { 1 } ^ { \mathrm { m W } } , \hdots , \mathbf { H } _ { B } ^ { \mathrm { m W } } \right] \in \mathbb { C } ^ { B \times U \times N ^ { \mathrm { m W } } } ,\tag{21}
$$

where each block $\mathbf { H } _ { h } ^ { \mathrm { m W } } \in \mathbb { C } ^ { U \times N ^ { \mathrm { m W } } }$ contains the mmWave channels from BS b to all UEs. Using $\mathbf { H } ^ { \mathrm { m W } }$ together with the predicted beamformers $\{ \mathbf { F } _ { b } \} _ { b = 1 } ^ { B }$ , the loss is evaluated, following Steps 19–20. The model parameters are then updated iteratively over minibatches and training epochs to minimize the loss, as shown in Steps 21–22.

Online Operation: At inference time (i.e., during model deployment), only the sub-6 GHz CSI H<sup>s</sup> is required to feed the (pretrained) model. Given a new graph constructed from H<sup>s</sup>, the trained GNN generates the beamforming vectors $\{ \mathbf { F } _ { b } \} _ { b = 1 } ^ { B }$ through a single forward pass.

The complexity of the proposed method is dominated by the GNN forward pass, particularly the message passing and aggregation over BS–UE edges, which scales with the number of links and their neighborhoods. However, the main complexity reduction is at the system level, where the proposed approach avoids explicit mmWave CSI acquisition and instead infers beamformers from sub-6 GHz CSI, which requires significantly lower training and feedback overhead.

## IV. NUMERICAL RESULTS

Simulation Setup: We generate the sub-6 GHz and mmWave channels using the DeepMIMO dataset [14] under the O1 scenario, which is based on ray-tracing simulations obtained from Remcom Wireless InSite. Specifically, we use the $\ O 1 \_ 2 8$ and $\mathtt { O 1 \_ 3 p 5 }$ configurations corresponding to the 28 GHz mmWave band and the 3.5 GHz sub-6 GHz band, respectively. We consider $B = 3$ BSs that jointly serve the UEs. Each BS employs $N ^ { \mathrm { m W } } = 3 2$ antennas for the mmWave array and $N ^ { \mathrm { s } } ~ = ~ 8$ antennas for the sub-6 GHz array, both with halfwavelength spacing. The bandwidths are set to 0.1 GHz and 0.01 GHz for the mmWave and sub-6 GHz bands, respectively, and the number of propagation paths is fixed to $L \ = \ 1 5$ Using the available sub-6 GHz CSI, we construct the graphs described in Section III. For each graph, we randomly select the number of UEs from the range [3, 8] and connect them to all BSs, forming a complete BS-UE bipartite structure. We generate a dataset of 10,000 graphs and randomly split it into training (80%) and testing (20%) sets.

Algorithm 1: Training of the Proposed Framework   
Input: Global sub-6 GHz CSI H<sup>s</sup>; global mmWave CSI   
$\mathbf { H } ^ { \mathrm { m W } }$ ; per-BS powers $\{ P _ { b } \} _ { b = 1 } ^ { \mathbb { B } }$   
Output: Trained GNN parameters $\Theta$ and predicted digital   
beamformers $\left\{ \mathbf { F } _ { b } \right\} _ { b = 1 } ^ { B } .$   
Parameters: B; $U ; T ;$ total number of graph samples $N ;$   
batch size $M ;$ learning rate $\eta ;$ optimizer Opt;   
number of training epochs $E .$   
$/ /$ Dataset construction   
1 $\mathcal { D }  \emptyset ;$   
2 for $i \gets 1$ to N do   
3 Construct $\mathcal { G } ^ { ( i ) } = ( \mathcal { V } ^ { ( i ) } , \mathcal { E } ^ { ( i ) } , \mathcal { T } _ { \mathcal { V } } )$ from the sub-6 GHz   
CSI H<sup>s</sup> according to Section III;   
4 Store $\left( \mathcal { G } ^ { ( i ) } \right)$ in $\mathcal { D } ;$   
5 Split D into training/test sets and form batches of size $M ;$   
6 Initialize GNN parameters $\Theta ;$   
$/ /$ End-to-end training   
7 for $e \gets 1$ to $E$ do   
8 for each minibatch $\subset \mathcal { D } ^ { \mathrm { t r a i n } }$ do   
// Phase 1: Initialization   
9 Initialize node and edge hidden states using (9)   
$/ /$ Phase 2: Message passing   
10 for $t \gets 0$ to $T - 1$ do   
11 Compute BS-side attention coefficients $\alpha _ { b , u  v } ^ { ( t ) }$   
using (10), and UE-side attention coefficients   
$\beta _ { b  w , u } ^ { ( t ) }$ using (12);   
12 Compute BS-side messages m $^ { \mathrm { B S } } _ { ^ { b , u } }$ using (11),   
and UE-side messages $\mathbf { \bar { m } } _ { b , u } ^ { \mathrm { U E } }$ using (13);   
13 Fuse the two messages according to (14);   
14 Update the edge hidden states according to (15);   
$/ /$ Phase 3: Readout   
15 Construct the contextual features $\mathbf { c } _ { b , u }$ according to   
Section III-A;   
16 Generate beamforming vectors $\mathbf { f } _ { b , u } = \varphi ^ { \mathrm { { o u t } } } ( \mathbf { c } _ { b , u } )$   
17 Form the digital beamformer at each BS as   
$\mathbf { F } _ { b } = \left[ \mathbf { f } _ { b , 1 } , \ldots , \mathbf { f } _ { b , U } \right]$   
18 Normalize each beamformer to satisfy the per-BS   
power constraint using (20)   
$/ /$ Loss computation   
19 Compute the user rates $\{ R _ { u } \} _ { u = 1 } ^ { U }$ with the predicted   
beamformers $\{ \mathbf { F } _ { b } \} _ { b = 1 } ^ { B }$ and the corresponding   
mmWave CSI $\mathbf { H } ^ { \mathrm { m w } } ;$   
20 Compute the loss $\mathcal { L } = - \boldsymbol { R } ^ { \mathrm { s u m } }$   
$/ /$ Parameter update   
21 Compute gradients $\hat { \nabla } _ { \Theta } \mathcal { L }$ by backpropagation;   
22 Update parameters using the chosen optimizer and   
learning rate: $\Theta  0 \mathrm { { p t } } ( \Theta , \nabla \Theta \mathcal { L } , \dot { \eta } )$

We implement the proposed GNN with $T = 3$ messagepassing layers and hidden dimension $d = 6 4$ . The attention blocks use the LeakyReLU activation with slope 0.2. The readout module uses a 3-layer MLP with rectified linear unit activations to generate the beamforming vectors. The model is trained using the Adam optimizer with learning rate of $1 0 ^ { - 4 }$ and decay factor of $1 0 ^ { - 5 }$ . We use a batch size of $M = 1 0$ graphs and train the model for 15 epochs.

Convergence Behavior: We evaluate the convergence of the proposed GNN-based beamforming training described in Algorithm 1 in Fig. 1. The figure shows the training loss over epochs for $P _ { b } = 1 ~ \mathrm { W }$ and $P _ { b } = 3 \mathrm { ~ W } .$ In both cases, the training loss decreases rapidly during the first few epochs and gradually stabilizes afterward, indicating stable convergence.

![](images/e801ebe6f8ffb8b304bd8746b21259eb7a81d3a3f9e84d1a5522b6543893307f.jpg)  
Fig. 1. Training loss convergence of the proposed framework for two representative per-BS transmit powers, $P _ { b } \in \{ 1 , 3 \} \ \mathrm { \dot { W } } .$

![](images/13dbfe57eba2fbb02653d5767ef2c19278e2ab9e8b49349de0fb2cdd243b9504.jpg)  
Fig. 2. Sum-rate of the GNN-based framework versus the number of available sub-6 GHz antennas at inference for $P _ { b } = 3 ~ \mathrm { W } .$

![](images/e4b002ab3524e666f5f5422d98da3a1bce3c0939b3d7fe25f6de13f0c3e494b8.jpg)  
Fig. 3. Sum-rate versus per-BS transmit power $P _ { b }$ for the proposed GNN-based beamformer and benchmark beamformers.

Impact of the Number of Sub-6 GHz Antennas: Fig. 2 shows the downlink sum-rate for $P _ { b } ~ = ~ 3 ~ \mathrm { ~ W ~ }$ versus the number of sub-6 GHz antennas for which CSI is available during inference. It is seen that the performance improves consistently as the number of available antennas increases. For instance, increasing the number of antennas from 3 to 8 improves the sum-rate from 11.10 bps/Hz to 12.95 bps/Hz. Using only 3 antennas already achieves about 85.7% of the performance obtained with all sub-6 GHz antennas.

Sum-Rate Maximization Results: We compare the sum-rate achieved by the proposed GNN-based beamforming scheme with three baselines using full mmWave CSI: zero-forcing (ZF), MMSE, and maximum-ratio transmission (MRT) [15]. ZF and MMSE are implemented centrally at the CPU, while MRT is applied locally at each BS.

Fig. 3 shows the average downlink sum-rate versus the per-BS transmit power $P _ { b }$ . All methods use the same GNNinduced per-UE power allocation, where the power assigned by BS b to UE u is given by $p _ { b , u } ^ { \mathrm { G N N } } = \| \mathbf { f } _ { b , u } \| _ { 2 } ^ { 2 }$ . The proposed beamformer consistently outperforms ZF and MRT, while remaining close to MMSE. For example, at $P _ { b } = 5 \mathrm { ~ W ~ } _ { \ast }$ , the proposed method achieves 15.18 bps/Hz, compared to 15.15 bps/Hz achieved by MMSE, 13.95 bps/Hz by ZF, and 10.59 bps/Hz by MRT. To isolate the benefit of the learned power allocation, Fig. 3 also includes the MMSE beamformer with uniform power allocation, denoted by “MMSE (Uniform).” At $P _ { b } = 5 \mathrm { ~ W ~ } ,$ the GNN-induced allocation increases its sumrate from 13.66 to 15.15 bps/Hz, corresponding to a 10.9% performance improvement.

## V. CONCLUSION

This paper studied fully digital downlink beamforming in CFmMIMO systems using only uplink sub-6 GHz CSI. We proposed a GNN-based framework that represents the system as a graph and learns beamformers through customized message passing over BS-UE links. Numerical results showed that the proposed approach generalizes well across different network topologies, remains robust under partial sub-6 GHz CSI at inference, and achieves sum-rate performance close to fully digital benchmark schemes that rely on full mmWave CSI. As future work, this framework can be extended to hybrid beamforming architectures.

## ACKNOWLEDGMENT

This work was supported by the Research Council of Finland through 6G Flagship Program (grant 369116) and projects DIRECTION (grant 354901), DYNAMICS (grant 24305016), and CHIST-ERA PASSIONATE (grant 359817).

## REFERENCES

[1] H. Q. Ngo et al., “Cell-free massive MIMO versus small cells,” IEEE Trans. Wireless Commun., vol. 16, no. 3, pp. 1834–1850, 2017.

[2] E. Bjornson and L. Sanguinetti, “Scalable cell-free massive MIMO¨ systems,” IEEE Trans. Commun., vol. 68, no. 7, pp. 4247–4261, 2020.

[3] R. W. Heath et al., “An overview of signal processing techniques for millimeter wave MIMO systems,” IEEE J. Sel. Topics Signal Process., vol. 10, no. 3, pp. 436–453, 2016.

[4] M. Hashemi, C. E. Koksal, and N. B. Shroff, “Out-of-band millimeter wave beamforming and communications to achieve low latency and high energy efficiency in 5G systems,” IEEE Trans. Commun., vol. 66, no. 2, pp. 875–888, 2018.

[5] M. Alrabeiah and A. Alkhateeb, “Deep learning for mmwave beam and blockage prediction using sub-6 GHz channels,” IEEE Trans. Commun., vol. 68, no. 9, pp. 5504–5518, 2020.

[6] S. Tavakolian et al., “Knowledge distillation for mmwave beam prediction using sub-6 GHz channels,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Processing, pp. 22642–22646, 2026.

[7] W. Deng et al., “CSI transfer from sub-6G to mmwave: Reducedoverhead multi-user hybrid beamforming,” IEEE J. Sel. Areas Commun., vol. 43, no. 3, pp. 973–987, 2025.

[8] Z. Huang, Z. Wang, and S. Chen, “Sub-6GHz assisted mmwave hybrid beamforming with heterogeneous graph neural network,” IEEE Trans. Commun., vol. 72, no. 11, pp. 6917–6928, 2024.

[9] E. Bjornson, J. Hoydis, and L. Sanguinetti,¨ Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency. Now Foundations and Trends, 2017.

[10] H. Song et al., “Joint channel estimation and data detection in cell-free massive MU-MIMO systems,” IEEE Trans. Wireless Commun., vol. 21, no. 6, pp. 4068–4084, 2022.

[11] J. Zhou et al., “Graph neural networks: A review of methods and applications,” CoRR, vol. abs/1812.08434, 2018.

[12] J. Gilmer et al., “Neural message passing for quantum chemistry,” in Proc. ICML (D. Precup and Y. W. Teh, eds.), vol. 70 of PMLR, pp. 1263– 1272, 06–11 Aug 2017.

[13] T. Monninger et al., “Scene: Reasoning about traffic scenes using heterogeneous graph neural networks,” IEEE Robot. Autom. Lett., vol. 8, no. 3, pp. 1531–1538, 2023.

[14] A. Alkhateeb, “DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications,” in Proc. Inf. Theory Appli. Workshop (ITA), (San Diego, CA), pp. 1–8, Feb 2019.

[15] R. W. Heath Jr and A. Lozano, Foundations of MIMO Communication. Cambridge Univ. Press, 2018.