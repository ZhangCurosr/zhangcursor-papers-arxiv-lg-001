# Efficient Graph Neural Networks for Multicarrier Wideband Hybrid Beamforming Optimization

Beier Li, Student Member, IEEE, Mai Vu, Senior Member, IEEE

Abstract—6G wireless technology is poised to adopt higher and wider frequency bands, leveraging highly directional beamforming. However, the vast bandwidths amplify the impact of beam squinting. Traditional solutions, such as adding a truetime-delay filter to each antenna, are cost-prohibitive due to the required hardware scale. This paper proposes a signal processing alternative using Graph Neural Networks (GNNs) to optimize hybrid beamforming in multicarrier wideband systems. Using a bipartite graph to represent a shared analog beamformer among multiple subcarriers, we develop three GNN structures with distinct digital beamformer representations (i) at the subcarrier nodes, (ii) at the edges, or (iii) integrating traditional singularvalue decomposition solutions. By designing an efficient messagepassing mechanism, these structures offer insights into the impact of different GNN designs on communication system performance and computational complexity. Extensive analysis and ablation studies show that our proposed GNN structures outperform traditional optimization methods and existing ML-based solutions. Furthermore, the proposed GNNs exhibit strong resiliency to beam squinting and better robustness against imperfect CSI than even fully digital beamforming and all existing hybrid designs. These GNNs can also be extended to multi-user scenarios and demonstrate excellent generalization capabilities, allowing trained models to adapt to diverse multicarrier and multi-user settings without retraining.

Index Terms—6G communication, beam squinting, hybrid beamforming, graph neural networks, wideband systems.

## I. INTRODUCTION

YBRID beamforming, which combines analog and digital techniques, offers a cost-effective solution and robust performance for employing massive antenna arrays in modern communication systems. In wideband systems utilizing OFDM to enhance data rates and resistance to multipaths, the beam pattern increasingly varies with frequency across subcarriers [1]. This phenomenon, known as beam squinting, becomes particularly significant in 6G wireless networks operating in the sub-Terahertz (THz) spectrum from 100 GHz to 1 THz, where the bandwidth can be as wide as 18 GHz [2].

To manage beam squinting in hybrid beamforming systems, recent research efforts have focused on two main approaches: true-time-delay lines (TTD) and signal processing methods. TTD is a time-delay filter integrated into each antenna to provide precise control over signal timing and effectively eliminates beam squinting [3]. However, in the sub-THz frequency range, the ability to pack more antennas into the same device size not only enhances performance but also significantly increases the number of TTDs required, leading to substantially higher costs. Given these economic considerations, signal processing methods have become an attractive alternative for managing beam squinting in high-frequency domains because of their cost-efficiency.

Since our focus in this paper is on managing the beam squinting in MIMO-OFDM hybrid beamforming systems, we primarily consider a single-user scenario. This allows us to isolate beam squinting from inter-user interference, which enables the design and evaluation of more effective hybrid beamforming strategies. As a natural extension, we further apply the proposed methods to multi-user scenarios to demonstrate their applicability under more general system settings.

## A. Related Work

Despite recent progress, current literature indicates that beamforming designs for wideband channels lack efficient solutions to effectively address beam squinting [4]. Optimization-based algorithms such as Alternative Manifold Optimization (AMO) [5] and Iterative Coordinate Descent (ICD) [6] have shown promise, but suffer from the need for continuous optimization with every channel update and hence require significant computational resources, limiting their practical advantage. In contrast, machine learning (ML)- based approaches [7]–[17] offer a promising alternative that can significantly reduce this burden.

1) Traditional Optimization-based Methods: A comparative study has explicitly analyzed beam squint under wideband conditions [4], which compares six analog beamformer designs. While computationally efficient, these methods substantially degrade system performance. In the most severe cases, when the number of antennas increases to 160, the performance drops to 70% of its optimal level [4].

The benchmark for this comparison was the centralized optimization algorithm Alternative Manifold Optimization (AMO) [5], which uses manifold optimization to calculate the analog beamformer and then applies a pseudo-inverse for digital beamformers. AMO shows strong beam squinting resistance and achieves the best to date spectral efficiency performance among traditional hybrid beamforming methods.

Another optimization method is the Iterative Coordinate Descent (ICD) algorithm [6], which sequentially updates the elements in the analog beamformer one at a time while keeping others fixed. This greedy strategy sacrifices some performance compared to AMO but reduces memory requirements.

These traditional optimization-based methods have been widely adopted as benchmarks in existing machine learning–based hybrid beamforming studies [7]–[16].

2) Machine Learning-based Methods: Recent ML-based methods can approach the data rate obtained by traditional optimization algorithms while improving computational efficiency. For example, several studies have applied fullyconnected neural networks (FNNs) [7] and convolutional neural networks (CNNs) [8] to hybrid beamforming in singleuser MIMO systems, and have also developed several CNNbased methods for multiuser narrowband scenarios [9], [17]. However, the method in [9] relies on a pseudo-inverse to form the analog beamformer, making it suboptimal for extension to OFDM systems. A theoretical analysis in [17] shows that beamforming requires modeling global dependencies across the entire channel matrix, which conflicts with the local pattern extraction nature of convolutional kernels. Capturing such global interactions with CNNs requires large kernel sizes and leads to a significant increase in trainable parameters as the system size grows. In contrast, a comparison of GNNs with FNNs and CNNs in wireless communication tasks [14] highlights that GNNs are inherently permutation equivariant and naturally scalable to variable-sized graphs, making them more suitable for systems with dynamic configurations. These properties allow GNNs to outperform other ML architectures while maintaining high inference speed.

TABLE I: Comparison of GNN-based Hybrid Beamforming Methods
<table><tr><td rowspan=1 colspan=1>Refs.</td><td rowspan=1 colspan=1>Problem Descriptions</td><td rowspan=1 colspan=1>Graph Structure</td><td rowspan=1 colspan=1>GeneralizationAbility</td><td rowspan=1 colspan=1>OutperformAMO [5]?</td></tr><tr><td rowspan=1 colspan=1>[14]</td><td rowspan=1 colspan=1>A unified multi-dimensional graph model forbeamforming and other wireless tasks.</td><td rowspan=1 colspan=1>K-partite graph (nodes can be users,subcarriers, antennas, etc.)</td><td rowspan=1 colspan=1>Weak</td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=1>[15]</td><td rowspan=1 colspan=1>End-to-end hybrid beamforming</td><td rowspan=1 colspan=1>Near fully-connected graph (nodes arebase stations, users, and subcarriers)</td><td rowspan=1 colspan=1>Moderate</td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=1>[16]</td><td rowspan=1 colspan=1>Joint design of pilot, channel feedback, andhybrid beamforming</td><td rowspan=1 colspan=1>Bipartite graph (analog and digitalbeamformer nodes)</td><td rowspan=1 colspan=1>Weak</td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=1>NU-GNN (Ours)</td><td rowspan=3 colspan=1>Wideband hybrid beamforming, beamsquintingmitigation</td><td rowspan=3 colspan=1>Bipartite graph (analog and subcarriernodes)</td><td rowspan=1 colspan=1>Strong</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>EU-GNN (Ours)</td><td rowspan=1 colspan=1>Strong</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>AN-GNN (Ours)</td><td rowspan=1 colspan=1>Strong</td><td rowspan=1 colspan=1>x</td></tr></table>

In the context of hybrid beamforming design, most existing GNN-based methods have shown promising results in narrowband scenarios [10]–[13]. These methods, however, cannot be directly extended to multicarrier settings such as OFDM systems for various reasons, including the coupling of analog and digital components [10], relying on a fully connected graph structure where the number of edges grows quadratically with the number of nodes, leading to high computational complexity [11], being specifically designed for a system setting of coexisting sub-6GHz and mmWave [12], or being tailored to a partially connected hardware structure [13].

For GNN-based approaches in multicarrier settings, existing works typically use a naive message-passing mechanism in which a single MLP generates messages that are linearly combined to update node representations [14]–[16]. The hyper-edge-based method in [14] models antennas, users, and subcarriers as distinct node types, replicating information for scalability. However, the lack of problem-specific graph design and naive message-passing strategy limits existing GNNs’ generalization ability. Another end-to-end approach [15] utilizes a near fully-connected graph designed for TDD systems to infer downlink channel state information (CSI) from uplink measurements, but its learning update mechanism limits the overall performance. A related study [16] uses a GNN to jointly design pilot signals, channel feedback, and hybrid beamforming in an FDD system. When perfect CSI is available, the problem reduces to standard hybrid beamforming, making it a suitable baseline for our work.

## B. Contributions

In this work, we consider HBF for wideband multicarrier systems with beamsquinting effect, and aim to design GNN structures that not only outperforms traditional optimization methods but also effectively mitigate beam squinting. This is in contrast to existing ML solutions [10]–[16] which are designed for narrowband systems and typically only approach the spectral efficiency performance of traditional optimizationbased methods rather than outperforming them. In addition, beam squinting resiliency was not addressed in prior ML-based designs [14]–[16]. To fill this gap, we focus on a wideband, single-user MIMO-OFDM system. See Table I for comparison with existing works.

Specifically, we propose to solve the highly non-convex hybrid beamforming design problem by constructing three different GNN structures using a bipartite graph. The graph has two types of nodes, which represent the analog beamformer and the subcarriers, respectively. The GNNs employ an efficient yet effective message-passing mechanism to jointly learn the beamformers, with an unsupervised loss function that directly maximizes the system’s spectral efficiency. Our major contributions are summarized as follows:

• We design the GNNs to directly optimize the phases of the analog beamformer instead of its complex-value entries. This direct phase learning has not been explored in prior work [7]–[17], but can help to significantly reduce the computational complexity of the ML models. More importantly, learning the phase guarantees inherent satisfaction of the analog beamforming constraint during the learning process without the need for any constant modular projection (which is non-differentiable), thereby maintaining the optimality of what has been learned and leading to better performance, as confirmed in the numerical results.

• Our graph accurately captures the OFDM structure using a fully-connected bipartite topology, where a single analog node connects to multiple subcarrier nodes. We further design an efficient message-passing mechanism that outperforms existing GNN-based approaches [14]– [16] in both performance and generalization. Leveraging the permutation equivariance and scalability of GNNs, our model generalizes to a large number of subcarriers without

![](images/d29b9dd6c67d7d49deeb8e214c6035aa91ccdc82c5cb7851293914636087c8c2.jpg)  
Fig. 1: Block diagram of a single-user MIMO-OFDM system with hybrid beamforming at the BS.

retraining.

• We explore pure ML approaches by studying two different GNN update mechanisms: a node-update GNN and an edge-update GNN, achieved by placing the representation for the digital beamformer of each subcarrier at either a subcarrier node or an edge. These design choices provide insights into how representation placement affects the applicability of different GNN structures in hybrid beamforming. To the best of our knowledge, this is the first use of edge-update GNNs in hybrid beamforming. Building upon these designs, we directly extend the proposed pure ML models to multi-user scenarios through problem reformulation and graph structure expansion, without modifying the GNN updating rules.

• We further proposed a novel hybrid structure combining ML with traditional signal processing methods to develop a GNN structure in which the digital beamformers are updated using a closed-form singular-value decomposition result while learning the analog beamformer only. We also design an attention-based aggregation mechanism to overcome the reliance on a single learned analog node and improve the generalization of this GNN structure. Our hybrid design is unique and offers valuable insights into the performance of combining ML and traditional optimization methods.

• Finally, we perform extensive performance comparison and ablation study of the three proposed GNNs against multiple traditional optimization and existing ML-based solutions. The results show that the proposed messagepassing mechanism in our GNNs outperforms all existing methods in spectral efficiency, generalization ability with respect to the number of users and subcarriers, beam squinting resiliency, and computational efficiency, even under imperfect CSI. These advantages are particularly prominent when the number of antennas increases.

## II. SINGLE USER SYSTEM MODEL AND PROBLEMFORMULATION

## A. System and Signal Models

We consider a single-user MIMO-OFDM system as shown in Fig. 1, where the base station (BS) equipped with N<sub>RF</sub> RF chains and $N _ { t }$ antennas sends N data streams to the user equipment (UE). The receiver has $N _ { r }$ antennas, where $N _ { s } \le N _ { \mathrm { R F } } \le N _ { r } \ll N _ { t }$ because of hardware constraint. The system employs OFDM where the transmission bandwidth B is divided into K subcarriers with equal widths, and the BS employs hybrid beamforming to transmit data to the UE.

The hybrid beamformer consists of a digital baseband beamformer $\mathbf { F } [ k ] \in \mathbb { C } ^ { N _ { \mathrm { R F } } \times N _ { s } }$ for each subcarrier k, and an analog RF beamformer $\mathbf { W } \in \mathbb { C } ^ { N _ { t } \times N _ { \mathrm { R F } } }$ shared among all subcarriers. The transmitted signal vector on the k-th subcarrier is

$$
\begin{array} { r } { { \bf x } [ k ] = \sqrt { P _ { t } } { \bf W } { \bf F } [ k ] { \bf s } [ k ] , } \end{array}\tag{1}
$$

where $P _ { t }$ is the averaged transmit power per subcarrier, and $\mathbf { s } [ k ]$ is the normalized $N _ { s } \times 1$ symbol vector transmitted in each subcarrier $k = 1 , 2 , . . . , K$ , where $\mathbb { E } [ { \mathbf { s } } [ k ] { \mathbf { s } } ^ { * } [ k ] ] = \mathbf { I } _ { N _ { s } }$ . For the single-user setting, we assume equal power allocation across all subcarriers to focus on beam squinting mitigation, following [4]–[6]. Simulation results under relaxed per-subcarrier power constraint with upper-bounded power budget are also provided for comparison in Section VIII.

Accordingly, the transmit power constraint per subcarrier imposes a normalization on the beamformer matrices as

$$
| | \mathbf { W } \mathbf { F } [ k ] | | _ { F } ^ { 2 } = 1 .\tag{2}
$$

Furthermore, while the digital beamformer matrices $\mathbf { F } [ k ]$ can have complex-valued elements, the analog beamformer matrix W is restricted to having its elements with fixed magnitude because of phase array implementation. This condition leads to the constant modulo constraint as

$$
\left| [ \mathbf { W } ] _ { i , j } \right| = 1 , \quad \forall i , j ,\tag{3}
$$

which can also be written as

$$
\mathbf { W } = e ^ { j \Phi } ,\tag{4}
$$

where $\Phi \in \mathbb { R } ^ { N _ { t } \times N _ { \mathrm { R F } } }$ is the phase matrix, with unwrapped phase elements $- \infty \leq \Phi _ { i , j } \leq \infty$

The received signal on the k-th subcarrier is

$$
\mathbf { y } [ k ] = { \sqrt { P _ { t } } } \mathbf { H } [ k ] \mathbf { W } \mathbf { F } [ k ] \mathbf { s } [ k ] + \mathbf { n } [ k ] ,\tag{5}
$$

where $\mathbf { H } [ k ]$ is the channel matrix on the k-th subcarrier, and $\mathbf { n } [ k ] \sim \mathcal { C } \dot { \mathcal { N } } ( 0 , \sigma _ { n } ^ { 2 } \mathbf { I } _ { N _ { r } } )$ is the additive white Gaussian noise.

## B. Channel Model

We adopted a wideband clustered double-directional channel model [18]:

$$
\mathbf { H } [ k ] = \frac { 1 } { \sqrt { N _ { \mathrm { c l } } N _ { \mathrm { r a y } } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c l } } } \sum _ { l = 1 } ^ { N _ { \mathrm { r a y } } } \alpha _ { i l , k } \beta _ { i l , k } \mathbf { a } _ { k } ^ { r } ( \phi _ { i l } ^ { r } , \theta _ { i l } ^ { r } ) \mathbf { a } _ { k } ^ { t } ( \phi _ { i l } ^ { t } , \theta _ { i l } ^ { t } ) ^ { * } ,\tag{6}
$$

where $N _ { \mathrm { c l } }$ and $N _ { \mathrm { r a y } }$ represent the number of clusters and rays within each cluster. $\alpha _ { i l , k }$ denotes the complex path gain of the l-th ray in the i-th cluster, and $\beta _ { i l , k } = e ^ { - j 2 \pi \tau _ { i l } f _ { k } }$ represents the propagation path delay component. The angles $( \phi _ { i l } ^ { r } , \theta _ { i l } ^ { r } )$ and $\bar { ( \phi _ { i l } ^ { t } , \theta _ { i l } ^ { t } ) }$ represent the azimuth and elevation angles of arrival and departure, respectively. Considering uniform planar arrays (UPA), the array response vector corresponding to the l-th ray in the i-th cluster is

$$
\begin{array} { c } { { { \bf a } _ { k } ( \phi _ { i l } , \theta _ { i l } ) = \left[ 1 , . . . , e ^ { j \frac { 2 \pi } { \lambda _ { k } } d ( p \sin \phi _ { i l } \sin \theta _ { i l } + q \cos \theta _ { i l } ) } , . . . , \right. } } \\ { { \left. e ^ { j \frac { 2 \pi } { \lambda _ { k } } d ( ( M - 1 ) \sin \phi _ { i l } \sin \theta _ { i l } + ( N - 1 ) \cos \theta _ { i l } ) } \right] ^ { T } , } } \end{array}\tag{7}
$$

where $d$ and $\lambda _ { k }$ are the antenna spacing and the signal wavelength, $0 \leq p < N$ and $0 \leq q < M$ are the antenna indices in the 2D plane.

![](images/137f741a044323497840b9e6a870832e43e795c4f1d5414c796dfc7bd1da9972.jpg)

![](images/5ddcdcb299403fededfefccaf0650df1af40d9286c55576f3895f45116a48455.jpg)  
(a)  
(b)  
Fig. 2: Beam squinting effect in a wideband system with the central frequency $f _ { c } = 1 4 2 \mathrm { G H z }$ , bandwidth $B = 2 0 \mathrm { G H z } .$ . (a) The beam’s direction shifts across 4 subcarriers. (b) The normalized array gain versus the fractional bandwidth $b = B / f _ { c }$ for angle of arrivals (AoAs) $\theta = \{ \pi / 6 , \pi / 4 , \pi / 3 \}$ Typical narrowband settings [11], [12] and the considered system are also marked in (b) for comparison.

## C. Beam Squinting Effect

Beam squinting refers to the frequency-dependent array gain in wideband systems where the beam’s direction shifts across subcarriers, leading to array gain degradation [1]. This effect becomes more noticeable as the system bandwidth expands, as shown in Fig. 2. In hybrid beamforming systems, although the digital beamformer offers flexibility, the analog beamformer is constrained by a single set of phase shifters that is shared across all subcarriers, making it the cause for beam squinting. As a result, accurately characterizing beam squinting requires an explicit wideband modeling framework that accounts for the frequency-dependent behavior across subcarriers.

The shared analog beamformer is captured in the problem formulation that jointly optimizes the system across all subcarriers by enforcing a shared analog beamforming matrix W while allowing independent digital beamforming matrices F[k] for each subcarrier. The effectiveness of beam squinting mitigation depends on the algorithm to solve the problem and will serve as a metric to evaluate the algorithms’ performance.

## D. Problem Formulation

We focus on the design of the transmit beamformers at the BS, assuming the perfect channel state information (CSI). The achievable spectral efficiency can be expressed as

$$
\begin{array} { r l } & { R = \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log _ { 2 } \bigg [ \operatorname* { d e t } \bigg ( \mathbf { I } _ { N _ { r } } + \frac { P _ { t } } { \sigma _ { n } ^ { 2 } } \mathbf { H } [ k ] \mathbf { W } \mathbf { F } [ k ] } \\ & { ~ \times \mathbf { F } ^ { * } [ k ] \mathbf { W } ^ { * } \mathbf { H } ^ { * } [ k ] \bigg ) \bigg ] ( \mathrm { b p s / H z } ) . } \end{array}\tag{8}
$$

Then the beamforming design problem can be posed as

$$
\operatorname* { m a x } _ { \mathbf { W } , \mathbf { F } [ k ] } \ R \qquad \mathrm { s . t . ~ } ( 2 ) , ( 3 ) .\tag{9}
$$

This hybrid beamforming problem is non-convex due to the constant modulo constraint in (3). Traditional signal processing methods typically address this non-convexity through alternating optimization. In each algorithm step, the analog beamformer is reconstructed by extracting only the phase matrix of the resulting complex-valued matrix in each algorithm step to satisfy the constant modulo constraint [4]–[6], which can result in suboptimality. In our approach, we choose to treat the phases of the analog beamformer as the unknown variables directly. As such, the problem formulation becomes

![](images/d5ed68ae0f53179652f2094f66ed635dd4e5d3bca2c19ebbf79e8696cb2d724f.jpg)  
Subcarrier (1)  
Subcarrier (k)  
Subcarrier (K)  
Fig. 3: Bipartite graph model for a hybrid beamforming structure with two types of nodes: analog and subcarrier nodes. Information embedded at the analog node is represented as vector x, at each subcarrier node as $\mathbf { c } _ { k }$ , and on each edge as ${ \bf e } _ { k } .$ Each subcarrier node and edge corresponds to a subcarrier $k ,$ while the analog node is shared among all subcarriers. The proposed GNNs based on this graph model employ a message-passing mechanism where messages are denoted as $\mathbf { m } _ { k } ^ { a }$ and $\dot { \mathbf { m } } _ { k } ^ { d } .$

$$
\begin{array} { l } { { \displaystyle R ( \Phi , { \bf F } [ k ] ) \triangleq \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log _ { 2 } \left[ \operatorname* { d e t } \left( { \bf I } _ { N _ { r } } + \frac { P _ { t } } { \sigma _ { n } ^ { 2 } } { \bf H } [ k ] { \bf e } ^ { { \bf j } \Phi } { \bf F } [ k ] \right. \right. } } \\ { { \displaystyle \left. \left. \times { \bf F } ^ { * } [ k ] \left( e ^ { j \Phi } \right) ^ { * } { \bf H } ^ { * } [ k ] \right) \right] } } \end{array}\tag{maxΦ,F[k]}
$$

$$
\mathrm { s . t . } \ \left| \left| e ^ { j \Phi } \mathbf { F } [ k ] \right| \right| _ { F } ^ { 2 } = 1 .\tag{10}
$$

Formulation (10) is equivalent to formulation (9) but has fewer constraints because of the change of variables.

While traditional optimization methods exploit the problem’s mathematical structure, they require re-optimization for every new CSI, which is computationally intensive. In contrast, ML methods are not constrained by problem structure and only rely on dataset statistics. Once trained, they only require forward computations for any new, unseen CSI, which makes the computational process much simpler and faster. As such, we will examine the use of ML to tackle this problem.

## III. MESSAGE-PASSING GNN STRUCTURES

We propose graph-based learning structures for hybrid beamforming design in wideband systems. We begin by motivating our choice of GNNs over other ML architectures. We then construct a graph model that serves as the foundation for the GNN structures introduced in the following sections. Finally, we describe the training process applicable to all proposed GNN structures.

## A. Motivations of Selecting GNNs

Traditional fully-connected neural networks (FNNs) require fixed-length inputs whose dimensions scale with system size, such as the number of subcarriers, and must be retrained for different configurations, therefore allowing little to no generalization or scaling ability. Convolutional neural networks (CNNs) are designed for grid-structured data and have similar issues with varying input sizes. Techniques like padding or interpolation can be used to align input dimensions, but often degrade performance.

In contrast, GNNs utilize the system’s underlying structural graph, where nodes represent system elements such as subcarriers in our problem. Reordering nodes does not affect the underlying graph mapping, and each node type typically shares a common update function, so the number of trainable parameters does not depend on the graph size [14]. This scalability allows GNNs to generalize across systems with varying subcarrier configurations, which is a powerful and desirable property. Guided by these insights, we choose GNN as the architecture of choice for designing hybrid beamforming for an OFDM system. The underlying graph model in a GNN can capture the relationship between analog and digital beamformers in a multicarrier OFDM structure while preserving its inherent symmetry, allowing strong scalability.

![](images/46fbcbc3cafb7eebcc64cafa618a42e80f285c380b1962b60289891096083d2b.jpg)  
Fig. 4: Layer-wise GNN architecture overview. Each updating layer consists of a message generation and a representation update step. The final layer is a beamformer reconstruction to produce the output. The inputs $( \mathbf { \check { x } } ^ { ( 0 ) } , \mathbf { c } _ { k } ^ { ( 0 ) } , \mathbf { e } _ { k } ^ { ( 0 ) } )$ depend on the specific proposed GNN structures. The specific configurations in each updating layer for different proposed GNN structures are detailed in Figs. 5–7.

## B. Message-passing Graph Structure

To better understand the generalization property of a GNN, we examine the subcarrier-level behavior and establish the following permutation equivariant properties. We first show that the original problem exhibits this property, and when mapped to a GNN, the GNN also maintains this property.

Proposition 1: The formulated problem in (10) is permutation equivariant with respect to the subcarriers, RF chains, and data streams.

Proof: See Appendix A.

Proposition 1 implies that permuting the subcarrier, RF chain, and data stream indices does not affect the optimal solution of the problem. Since this work focuses on addressing beam squinting effects rather than exhaustively exploiting all possible equivariance properties, we mainly leverage the subcarrier-level equivariance to construct the proposed graph model.

We employ a bipartite, undirected graph, where a single “analog node” connects to K “subcarrier nodes”, each representing a subcarrier. As shown in Fig. 3, we embed information at the analog node as $\mathbf { x } \in \mathbb { R } ^ { N _ { t } N _ { \mathrm { R F } } }$ , at each subcarrier node k as $\mathbf { c } _ { k } \in \mathbb { R } ^ { 2 N _ { \mathrm { R F } } \mathbf { \bar { N } } _ { s } }$ , and on each edge as $\mathbf { e } _ { k } \in \mathbb { R } ^ { 2 N _ { t } N _ { \tau } }$ . Note that not all of these embeddings, also called representations, need to be present in a GNN structure, and we will propose three different GNN structures that use different subsets of these representations. The specific design of each proposed GNN structure is illustrated in Figs. 5–7 and described in Sections IV and V. Before introducing these structures in detail, we first describe the general form of message-passing used in our framework.

Figure 4 provides an overview of the layer-wise GNN architecture, which stacks L updating layers for progressively learning and refining representations. Each layer employs a message-passing mechanism, where nodes exchange messages with their neighbors to update their representations. The design of these mechanisms for message passing and representation updating will define the learning process in the GNN and is important for the final performance and generalization ability. Specifically, we generate messages m $\mathsf { l } _ { k } ^ { a ^ { - } } \in \mathbb { R } ^ { N _ { t } N _ { \mathrm { R F } } }$ to carry information from the analog node x to the k-th subcarrier node $\mathbf { c } _ { k } .$ , while $\mathbf { m } _ { k } ^ { d } \in \mathbb { R } ^ { 2 N _ { \mathrm { R F } } N _ { s } }$ transmit information in the reverse direction. At a high level, we can summarize the updating operations in our proposed GNNs as:

$$
\mathbf { b } _ { k } ^ { ( l ) } = f _ { \mathbf { b } } \left( \mathbf { a } _ { k } ^ { ( l - 1 ) } , \phi ( \cdot ) \right) , \quad \forall \ \mathbf { a } _ { k } , \mathbf { b } _ { k } \in \mathcal { A } , \ \forall k ,\tag{11}
$$

where $\mathcal { A } = \{ \mathbf { x } , \mathbf { e } , \mathbf { c } , \mathbf { m } ^ { a } , \mathbf { m } ^ { d } \}$ is a set of all representations and messages. Function $\phi ( \cdot )$ denotes a permutation invariant aggregation function that is consistently applied to the selected node’s neighbors, thus preserving the permutation equivariance of the overall updating operation. Each function $f _ { \mathbf { b } }$ is typically realized by a multi-layer perceptron (MLP) network and is shared across all subcarriers k. The specific design for the mapping in (11) varies with each GNN structure and will be discussed in Section IV and V.

Based on (11), all our designed GNN structures exhibit the following property:

Proposition 2: The output of each GNN updating layer is permutation equivariant with respect to the subcarrier order. As such, the final outputs of each GNN, including the analog and digital beamformers, are permutation equivariant.

Proof: See Appendix B.

Proposition 2 highlights the generalization ability of the proposed GNNs across subcarrier-level variations, enabling adaptation to dynamic system configurations.

## C. Considerations in Designing GNN Structures

The graph structure in Fig. 3 captures the underlying communication system by representing the analog beamformer as a single node x, shared across all subcarriers, and therefore is learned via the node representation at this analog node. The main consideration in designing a specific GNN model lies in how to represent the digital beamformers. We explore three strategies for representing the digital beamformers: (i) using the digital beamformer node representations (nodeupdate structure), (ii) using edge representations (edge-update structure), and (iii) a hybrid structure which only learns the analog beamformer and bypasses learning the digital beamformer entirely by applying the traditional SVD-based digital beamforming solutions derived from the learned analog beamformer.

Different GNNs are expected to show varying performance and complexities. The first two GNN structures will illustrate the differences in node or edge updates, and their impact on the overall performance. The third GNN structure will examine the difference between learning everything vs. using known traditional optimization results in some steps. Extensive analysis and ablation study of these structures will offer valuable insights for designing GNN structures in practical wireless systems, with impacts on achievable performance, computational complexity, and scalability.

## D. Loss Function and Unsupervised Training

For each GNN structure, we train the model such that the GNN can effectively update the representations to achieve a high average data rate, as formulated in (10). Let Ω denote all trainable parameters in a GNN for message generation and representation update. During offline training, we train the GNN in an unsupervised manner to optimize Ω by minimizing the loss function derived from (10) as:

$$
\mathrm { L o s s } ( \Omega ) = - \ R ( \Phi , { \bf F } _ { \mathrm { n o r m a l i z e d } } [ k ] ) .\tag{12}
$$

This loss function is computed using the GNN’s outputs, Φ and $\mathbf { F } _ { \mathrm { n o r m a l i z e d } } [ k ]$ ], along with the available CSI. To ensure that the outputs satisfy the constraints specified in (10), we apply normalization steps during the beamformer reconstruction process to meet the required power constraint.

We then minimize the loss in (12) using a mini-batch stochastic gradient descent (SGD) approach as:

$$
\pmb { \Omega } ^ { ( i + 1 ) }  \pmb { \Omega } ^ { ( i ) } - \eta ^ { ( i ) } \nabla _ { \pmb { \Omega } } \mathbb { E } _ { B } [ \mathrm { L o s s } ( \pmb { \Omega } ) ] ,\tag{13}
$$

where $\eta$ is the learning rate, and B denotes the mini-batch set. Because of the permutation equivariance and scalability of GNNs, it suffices to train on a small number of subcarriers, given sufficient CSI samples spanning a wide frequency range and channel conditions. Then during the online inference, the number of subcarrier nodes can vary as needed to suit different OFDM system configurations.

## IV. NODE AND EDGE UPDATE GNN STRUCTURES

In this section, we consider using pure machine learning to design hybrid beamformers. We propose two GNN structures: node-update GNN (NU-GNN) and edge-update GNN (EU-GNN). Both approaches represent the analog beamformer at the analog node but differ in handling the digital beamformers.

NU-GNN embeds the digital beamformers in the subcarrier nodes by using $\{ \mathbf { c } _ { 1 } , . . . , \mathbf { c } _ { K } \}$ as the digital beamformer representations, while keeping the edge features fixed. The edge features are the vectorized CSI, such that

$$
\mathbf { e } _ { k } = \mathbf { h } _ { k } = \left[ \mathrm { v e c } \left( \mathrm { R e } \left\{ \rho \mathbf { H } [ k ] \right\} \right) ^ { T } , \mathrm { v e c } \left( \mathrm { I m } \left\{ \rho \mathbf { H } [ k ] \right\} \right) ^ { T } \right] ^ { T } ,\tag{14}
$$

where $\rho ~ = ~ { \sqrt { \frac { P _ { t } } { \sigma _ { n } ^ { 2 } } } }$ scales the channels to mitigate the small magnitudes caused by pathloss. This ensures numerically suitable inputs for training without impacting the final sumrate, as the scaling is compensated during computation.

In contrast, EU-GNN embeds the digital beamformers in the edge representations $\{ \mathbf { e } _ { 1 } , . . . , \mathbf { e } _ { K } \}$ , without using any subcarrier node representations. We use the vectorized CSI in (14) as the initial input of the edge updating process:

$$
{ \bf e } _ { k } ^ { ( 0 ) } = { \bf h } _ { k } , \quad \forall k = 1 , . . . , K .\tag{15}
$$

By exploring both structures, we aim to understand how the update mechanism affects system performance and the tradeoff between learning flexibility and computational complexity.

## A. Node Update GNN (NU-GNN) Structure

Next, we design the message generation and representation update methods. These designs are important as they directly affect the GNN performance and computational complexity.

At the l-th updating layer of the GNN, the messages sent from the analog node to subcarrier node k, and from subcarrier

![](images/54ea90bb6fa75ca795f712607761e90db8d44e70bc565f8c088c56ced661bc7a.jpg)  
Fig. 5: NU-GNN Structure: During the message-passing process, two MLPs $f _ { 1 } ^ { a } ( \cdot )$ and $f _ { 1 } ^ { d } ( \cdot )$ generate messages $\mathbf { m } _ { k } ^ { a }$ and m<sup>d</sup><sub>k</sub>, which are marked in purple and green. Then, two other MLPs $f _ { 2 } ^ { a } ( \cdot )$ and $f _ { 2 } ^ { d } ( \cdot )$ update the node representations x and $\mathbf { c } _ { k } ,$ marked in blue and orange. The black colored $\mathbf { h } _ { k }$ represents the edge feature, which does not get updated in this structure.

![](images/dc9fa10dc355f2a35a9dcd00edbd93a461d53cf68b4185ca0644edbe946ea3e5.jpg)  
Fig. 6: EU-GNN Structure: During the message-passing process, two MLPs $f _ { 1 } ^ { a } ( \cdot )$ and f<sup>d</sup>(·) generate messages $\mathbf { m } _ { k } ^ { a }$ and $\mathbf { m } _ { k } ^ { d } ,$ marked in purple and green. We also used two other MLPs $f _ { 2 } ^ { a } ( \cdot )$ and $\ddot { f } _ { 2 } ^ { d } ( \cdot )$ to update the node representation x and the edge representation ${ \mathbf { e } } _ { k } .$ , which are marked in blue and orange. In this structure, the CSI h<sub>k</sub> is only used as the initial value of the edge representation, which will be updated during the learning process by MLP $\bar { f } _ { 2 } ^ { d } ( \cdot )$ marked in orange.

node $k$ to the analog node, can be generated as

$$
{ { \bf m } _ { k } ^ { a ( l ) } } = f _ { 1 } ^ { a } ( { \bf h } _ { k } , { \bf x ^ { ( } } l { - } 1 ) )\tag{16}
$$

$$
\begin{array} { r } { \mathbf { m } _ { k } ^ { d ( l ) } = f _ { 1 } ^ { d } ( \mathbf { h } _ { k } , \mathbf { c } _ { k } ^ { ( l - 1 ) } ) , } \end{array}\tag{17}
$$

where $f _ { 1 } ^ { a } ( \cdot )$ and $f _ { 1 } ^ { d } ( \cdot )$ are two MLPs, as shown in Fig. 5 in purple and green, to generate the messages at the analog node and subcarrier nodes, respectively.

After generating the messages, each node receives them via the connecting edges, aggregates these incoming messages from its neighbors, and then updates its representation vector accordingly as follows.

$$
\mathbf { x } ^ { ( l ) } = f _ { 2 } ^ { a } ( \mathbf { x } ^ { ( l - 1 ) } , \phi ( \mathbf { m } _ { k } ^ { d ( l ) } ) _ { k \in \mathcal { N } ( \mathbf { x } ) } )\tag{18}
$$

$$
\mathbf { c } _ { k } ^ { ( l ) } = f _ { 2 } ^ { d } ( \mathbf { c } _ { k } ^ { ( l - 1 ) } , \mathbf { m } _ { k } ^ { a ( l ) } )\tag{19}
$$

$f _ { 2 } ^ { a } ( \cdot )$ and $f _ { 2 } ^ { d } ( \cdot )$ are two other MLPs as shown in Fig. 5, marked in blue and orange, to update the analog and digital beamformer representations, respectively. $\mathcal { N } ( \mathbf { x } )$ represents the set of neighboring nodes of $\mathbf { x } ,$ and $\phi ( \cdot )$ is the elementwise mean function used to combine the information coming into the same node. While both mean and max aggregation functions are permutation invariant, mean aggregation captures the collective impact of beam squinting by incorporating all subcarriers rather than the most dominant one.

## B. Edge Update GNN (EU-GNN) Structure

In the EU-GNN, we design the message generation and representation update as follows.

At the l-th GNN layer, the messages are generated as

$$
{ \bf m } _ { k } ^ { a ( l ) } = f _ { 1 } ^ { a } ( { \bf e } _ { k } ^ { ( l - 1 ) } , { \bf x } ^ { ( l - 1 ) } )\tag{20}
$$

$$
\mathbf { m } _ { k } ^ { d ( l ) } = f _ { 1 } ^ { d } ( \mathbf { e } _ { k } ^ { ( l - 1 ) } )\tag{21}
$$

where $f _ { 1 } ^ { a } ( \cdot )$ and $f _ { 1 } ^ { d } ( \cdot )$ are two MLPs as shown in Fig. 6 marked in purple and green, respectively.

Similar to NU-GNN, we employ the generated messages as the inputs to the representation updates as follows.

$$
\mathbf { x } ^ { ( l ) } = f _ { 2 } ^ { a } ( \mathbf { x } ^ { ( l - 1 ) } , \phi ( \mathbf { m } _ { k } ^ { d ( l ) } ) _ { k \in \mathcal { N } ( \mathbf { x } ) } )\tag{22}
$$

$$
\mathbf { e } _ { k } ^ { ( l ) } = f _ { 2 } ^ { d } ( \mathbf { e } _ { k } ^ { ( l - 1 ) } , \mathbf { m } _ { k } ^ { a ( l ) } , \mathbf { m } _ { k } ^ { d ( l ) } )\tag{23}
$$

Here $f _ { 2 } ^ { a } ( \cdot )$ and $f _ { 2 } ^ { d } ( \cdot )$ are two MLPs marked in blue and orange, respectively, as shown in Fig. 6. The node representation x is updated by aggregating the messages from its neighbor nodes, and the edge representation $\mathbf { e } _ { k }$ is updated by aggregating the messages flowing into this edge.

## C. Beamformer Reconstruction

To obtain the analog beamforming matrix, the final layer’s analog node representation $\mathbf { x } ^ { ( L ) }$ is reshaped into the phase matrix Φ:

$$
\Phi = \mathrm { r e s h a p e } ( \mathbf { x } ^ { ( L ) } , ( N _ { t } , N _ { \mathrm { R F } } ) ) .\tag{24}
$$

By directly learning the phase of the analog beamformer, the GNN output in (24) inherently satisfies the constant modulus constraint (3) without requiring any further projection or approximation, which would harm the learning performance.

To satisfy the power constraint in (10), the updated node representations $\mathbf { c } _ { k }$ , or the edge representations $\mathbf { e } _ { k }$ , from the final layer L of the GNNs must be further processed. The digital representations $\mathbf { c } _ { k } ^ { ( L ) }$ (or ${ \bf e } _ { k } ^ { ( L ) } )$ are reassembled into a complex-valued matrix $\mathbf { F } _ { k }$ and normalized to fulfill the power constraint in (10):

$$
\mathbf { F } _ { \mathrm { n o r m a l i z e d } } [ k ] = \frac { \mathbf { F } _ { k } } { | | \mathbf { F } _ { k } e ^ { j \Phi } | | _ { F } ^ { 2 } } ,\tag{25}
$$

where, $\begin{array} { r l } { \mathbf { F } _ { k } } & { { } = } \end{array}$ reshape $\left( \mathbf { c } _ { k } ^ { \left( L \right) } \left[ 1 : N _ { \mathrm { R F } } \times N _ { s } \right] , \left( N _ { \mathrm { R F } } , N _ { s } \right) \right) \ +$ $j$ reshape $\left( \mathbf { c } _ { k } ^ { \left( L \right) } \left[ N _ { \mathrm { R F } } \times \dot { N _ { s } } : 2 N _ { \mathrm { R F } } \times N _ { s } \right] , \left( N _ { \mathrm { R F } } , N _ { s } \right) \right)$ in the NU-GNN. In the EU-GNN, $\mathbf { c } _ { k } ^ { ( L ) }$ can be replaced by the edge representation ${ \bf e } _ { k } ^ { ( L ) }$ in the same reconstruction process.

## V. ANALOG NODE GNN WITH ATTENTION STRUCTURE

Unlike the previous two GNN structures, which use pure ML to jointly learn both analog and digital beamformers, here we design a different structure termed analog-GNN (AN-GNN), which combines machine learning with traditional optimization. Specifically, we update analog node representation x via learning, derive digital beamformer representations $\mathbf { c } _ { k }$ via closed-form expressions, and fix edge features $\mathbf { e } _ { k } = \mathbf { h } _ { k }$ as in (14). Next, we reformulate problem (10) to solve for the digital beamformers and detail the AN-GNN’s learning mechanisms.

![](images/3d73d88e35bb65f987e2405071109c06b83b4a12ccb8312fd09a7c2712cee8a0.jpg)  
Fig. 7: AN-GNN Structure: Before updating the messages, we use the singular-value decomposition solution in (27) and (29) to generate the digital beamformer representations $\mathbf { c } _ { k } ^ { ( l - 1 ) }$ at the subcarrier nodes. Two MLPs $f _ { 1 } ( \cdot )$ and $f _ { 2 } ( \cdot )$ are deployed to generate the messages m<sub>k</sub> and to update the analog node representation $\mathbf { x } ,$ marked in green and blue, respectively. An attention aggregation $g ( \cdot )$ outlined in red is applied to aggregate the gathered information across all subcarriers.

## A. Problem Reformulation

1) Digital Beamformer Solution: For a given analog beamformer, the digital beamformer in (10) can be solved in closed-form by constructing an effective channel $\mathbf { H } _ { \mathrm { e f f } } [ k ] =$ ${ \bf H } [ k ] e ^ { j \Phi } \left( \left( e ^ { j \Phi } \right) ^ { * } e ^ { j \Phi } \right) ^ { - \frac { 1 } { 2 } }$ . Let $\tilde { \mathbf { F } } [ k ] = \left( \left( e ^ { j \Phi } \right) ^ { * } e ^ { j \Phi } \right) ^ { \frac { 1 } { 2 } } \mathbf { F } [ k ] ,$ the original problem in (10) is

$$
\operatorname* { m a x } _ { { \bf F } [ k ] } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log _ { 2 } [ \operatorname * { d e t } ( { \bf I } _ { N _ { r } } + \frac { P _ { t } } { \sigma _ { n } ^ { 2 } } ( { \bf H } _ { \mathrm { e f f } } [ k ] \tilde { \bf F } [ k ] \times \tilde { \bf F } ^ { * } [ k ] { \bf H } _ { \mathrm { e f f } } ^ { * } [ k ] ) ) ]
$$

s.t. $| | \tilde { \mathbf { F } } [ k ] | | _ { F } ^ { 2 } = 1$

(26)

Problem (26) has a well-known singular-value decomposition solution [4], [6]:

$$
\mathbf { F } [ k ] = \left( \left( e ^ { j \Phi } \right) ^ { * } e ^ { j \Phi } \right) ^ { - \frac { 1 } { 2 } } \frac { \mathbf { V } _ { \mathrm { e f f } } [ k ] } { | | \mathbf { V } _ { \mathrm { e f f } } [ k ] | | _ { F } } ,\tag{27}
$$

where $\mathbf { V } _ { \mathrm { e f f } } [ k ]$ is the truncated right singular vector matrix with columns corresponding to the largest $N _ { \varepsilon }$ nonzero singular values of $\mathbf { H } _ { \mathrm { e f f } } [ k ]$

2) Analog Beamformer Optimization Problem: Now we focus on designing the analog beamformer only by fixing the digital beamformers F[k], and the problem in (10) becomes:

$$
\begin{array} { c l } { { { \displaystyle \operatorname* { m a x } _ { \Phi } } } } & { { { \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log _ { 2 } [ \operatorname * { d e t } ( { \bf I } _ { N _ { r } } + \frac { P _ { t } } { \sigma _ { n } ^ { 2 } } ( { \bf H } [ k ] e ^ { j \Phi } { \bf F } [ k ] } } } \\ { { } } & { { \times { \bf F } ^ { * } [ k ] \left( e ^ { j \Phi } \right) ^ { * } { \bf H } ^ { * } [ k ] ) ) ] . } } \end{array}\tag{28}
$$

The power constraint can be satisfied after solving (28) by renormalizing the digital beamformers as in (25).

## B. Message Generation and Node Update in AN-GNN

To optimize the analog beamformer’s phase Φ in (28), we modify the NU-GNN structure introduced in Section IV-A by using the closed form solution in (27) to update the digital beamformer representations. This results in the AN-GNN structure in Fig. 7. We design the learning and updating process of the AN-GNN as follows.

Before updating the messages at each layer l, we use the previous $\mathbf { x } ^ { ( l - \bar { 1 } ) }$ to reconstruct the analog phase matrix $\Phi ^ { ( l - 1 ) }$ as in (24), then use that together with each channel matrix $\mathbf { H } [ k ]$ to calculate the digital beamformer $\mathbf { F } [ k ] ^ { ( l - 1 ) }$ as in (27). These digital beamformers are reshaped into digital beamformer representations $\mathbf { c } _ { k } ^ { ( l - 1 ) }$ and used as input into the l-th GNN updating layer, as shown in orange in Fig. 7. The feature at subcarrier node k can be expressed as

![](images/bb0e7fa4701f16b2eee4c7bfe8089684f9eec9283383c248b2c951e9e95973d8.jpg)  
Fig. 8: The attention aggregation approach: First a fully connected linear layer is used to calculate the attention score $a _ { k }$ for each subcarrier, then these scores are converted into attention weights $\xi _ { k }$ by applying a softmax operation. A weighted sum by the learned $\xi _ { k }$ values emphasizes the relative importance of different subcarriers.

$$
\begin{array} { r } { \mathbf { c } _ { k } ^ { ( l - 1 ) } = \left[ \mathrm { v e c } \left( \mathrm { R e } \left\{ \mathbf { F } [ k ] \right\} \right) ^ { T } , \mathrm { v e c } \left( \mathrm { I m } \left\{ \mathbf { F } [ k ] \right\} \right) ^ { T } \right] ^ { T } . } \end{array}\tag{29}
$$

In the l-th updating layer, the corresponding $\mathbf { c } _ { k }$ is combined with the edge feature $\mathbf { h } _ { k }$ to generate the message $\mathbf { m } _ { k } ^ { ( l ) }$

$$
\begin{array} { r } { \mathbf { m } _ { k } ^ { ( l ) } = f _ { 1 } ( \mathbf { h } _ { k } , \mathbf { c } _ { k } ^ { ( l - 1 ) } ) , } \end{array}\tag{30}
$$

where $f _ { 1 } ( \cdot )$ is an MLP as shown in green in Fig. 7. We update the analog representation as

$$
\mathbf { x } ^ { ( l ) } = f _ { 2 } ( \mathbf { x } ^ { ( l - 1 ) } , g ( \mathbf { m } _ { k } ^ { ( l ) } ) _ { k \in \mathcal { N } ( \mathbf { x } ) } ) ,\tag{31}
$$

where $f _ { 2 } ( \cdot )$ is an MLP marked in blue as shown in Fig. 7. $g ( \cdot )$   
is the attention aggregation as discussed in the next subsection.

## C. Aggregations Via an Attention Mechanism

An important change compared to previous GNN structures is that in the updating processes in (31), g(·) is no longer the element-wise mean aggregation function. Here, we draw inspiration from the attention mechanism to better differentiate the impact of individual subcarriers on the analog beamformer. Specifically, we use a fully-connected linear layer followed by a LeakyReLU activation to calculate the attention score for each subcarrier. Then, we apply a softmax operation to convert these scores into attention weights. By performing a weighted sum, the relative importance of information from different neighboring nodes representing different subcarriers is adaptively adjusted for aggregation. The attention weights for each subcarrier can be calculated as:

$$
\xi _ { k } = \frac { \exp \left( \mathrm { L e a k y R e L U } \left( f ( \mathbf { x } , \mathbf { c } _ { k } , \mathbf { h } _ { k } ) \right) \right) } { \sum _ { k = 1 } ^ { K } \exp \left( \mathrm { L e a k y R e L U } \left( f ( \mathbf { x } , \mathbf { c } _ { k } , \mathbf { h } _ { k } ) \right) \right) } ,\tag{32}
$$

where $f ( \cdot )$ is a linear layer. The aggregated information is

$$
g ( \mathbf { m } _ { k } ) _ { k \in \mathcal { N } ( \mathbf { x } ) } = \sum _ { k = 1 } ^ { K } \xi _ { k } \mathbf { m } _ { k } .\tag{33}
$$

The overall process is shown in Fig. 8. This attention-based aggregation is permutation invariant. Indeed, the attention scores in Eq. (32) are computed via a permutation equivariant activation, followed by a softmax function that preserves this property. Consequently, both $\xi _ { k }$ and $\mathbf { m } _ { k }$ in (33) are jointly permutation invariant under subcarrier reordering, leading to a permutation invariant outcome.

## D. Beamformer Reconstruction

We employ (24) to reconstruct the analog beamforming phase matrix $\Phi ^ { ( L ) }$ from the analog node representation $\mathbf { x } ^ { ( L ) }$ We then use $\Phi ^ { ( L ) }$ to update the digital beamforming matrices as in (27) and (25). The resulting $\mathbf { \bar { \Phi } } _ { \Phi } ( L )$ and $\mathbf { F } _ { \mathrm { n o r m a l i z e d } } [ k ]$ are used to compute the sum rate in (28) for both training and inference. The overall algorithm is summarized in Alg. 1.

Algorithm 1 AN-GNN with Attention Aggregation   
Input: H[k], where $k = 1 , \ldots , K$   
1: for each epoch do   
2: for each sample $\in { \mathfrak { B } }$ do   
3: Generate the edge feature $\mathbf { h } _ { k }$ as shown in (14).   
4: Initialize $\mathbf { x } ^ { ( 0 ) }$ and $\mathbf { m } ^ { ( 0 ) }$   
5: for $l = 1$ to L do   
6: Reconstruct $\Phi ^ { ( l - 1 ) }$ by using $\mathbf { x } ^ { ( l - 1 ) }$   
7: Calculate $\mathbf { F } ^ { ( l - 1 ) } [ k ]$ by (27).   
8: Generate $\mathbf { c } _ { k } ^ { ( l - 1 ) }$ as shown in (29).   
9: Perform forward propagation as in (30)–(33).   
10: end for   
11: Reconstruct $\Phi ^ { ( L ) }$ by using $\mathbf { x } ^ { ( L ) }$   
12: Calculate $\mathbf { F } _ { \mathrm { n o r m a l i z e d } } [ k ]$ by (27) and (25).   
13: Calculate the loss function according to (12).   
14: end for   
15: Update the GNN’s parameters according to (13) for   
the next epoch.   
16: end for

## VI. EXTENSION TO THE MULTI-USER SETTING

In this section, we directly extend the two pure machine learning approaches proposed in Section IV (NU-GNN and EU-GNN) to the multi-user case. This extension is achieved by reformulating the objective function to account for multiuser transmission and expanding the graph model to include multiple users while preserving the same embedded information and GNN updating rules. The AN-GNN cannot be directly extended to the multi-user case, however, because the digital beamformers have no closed-form solution for the multi-user setting as in the single-user scenario.

## A. Problem Reformulation

We consider the same wideband MIMO-OFDM hybrid beamforming system model as in Section II, and extend it to a multi-user scenario directly. In this case, the BS serves M UEs and transmits $N _ { d }$ data streams to each UE, resulting in a total of $N _ { s } = M N _ { d }$ data streams. The transmitted signal vector on the k-th subcarrier is

$$
\mathbf { x } [ k ] = \sqrt { P _ { t } } e ^ { j \Phi } \sum _ { m = 1 } ^ { M } \mathbf { F } _ { m } [ k ] \mathbf { s } _ { m } [ k ] ,\tag{34}
$$

![](images/8260bcca45a45299f1fa65c3ebdbdc0a0f6bf1f12ec0393d2b4c5bb4f23247b5.jpg)  
Fig. 9: Direct extension to multi-user case using the same bipartite graph model with two types of nodes: analog and subcarrier nodes. Information embedded at the analog node, each subcarrier node, and edge is represented as vector x, $\mathbf { c } _ { k , m } ,$ and $\mathbf { e } _ { k , m } ,$ respectively. Each dashed box corresponds to a user-specific subcarrier set, which groups all subcarrier nodes and their associated edges corresponding to a given user m.

where $e ^ { j \Phi } \in \mathbb { C } ^ { N _ { t } \times N _ { \mathrm { R F } } }$ is the analog beamformer with the constant modulus constraint as in Eq. (4). $\mathbf { F } _ { m } [ k ] \in \mathbb { C } ^ { N _ { \mathrm { R F } } \times N _ { d } }$ is the digital beamformer for UE m and subcarrier k.

The transmit power constraint per subcarrier remains the same as in (2), where $P _ { t }$ denotes the total transmit power allocated to each subcarrier across all UEs. Therefore, the digital beamformers implicitly handle the power allocation among different users.

The received signal for the m-th user on the k-th subcarrier is

$$
\begin{array} { r l } & { \mathbf { y } _ { m } [ k ] = \underbrace { \sqrt { P _ { t } } \mathbf { H } _ { m } [ k ] e ^ { j \Phi } \mathbf { F } _ { m } [ k ] \mathbf { s } _ { m } [ k ] } _ { \mathrm { d e s i r e d ~ s i g n a l } } } \\ & { \qquad + \underbrace { \sqrt { P _ { t } } \sum _ { n = 1 \atop n \ne m } ^ { M } \mathbf { H } _ { m } [ k ] e ^ { j \Phi } \mathbf { F } _ { n } [ k ] \mathbf { s } _ { n } [ k ] } _ { \mathrm { i n f e r a s e r ~ i n t e r f e r e n c e } } + \underbrace { \mathbf { n } _ { m } [ k ] } _ { \mathrm { n o i s e } } , } \end{array}\tag{35}
$$

where $\mathbf { n } _ { m } [ k ] \sim \mathcal { C } \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { N _ { r } } )$ is the additive white Gaussian noise for each UE m.

Based on the received signal model in (35), the achievable spectral efficiency can be expressed as

$$
\begin{array} { r } { \displaystyle R _ { \mathrm { M U } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \sum _ { m = 1 } ^ { M } \log _ { 2 } \Biggl [ \operatorname* { d e t } \Biggl ( \mathbf { I } _ { N _ { r } } + P _ { t } \mathbf { H } _ { m } [ k ] e ^ { j \Phi } \mathbf { F } _ { m } [ k ] } \\ { \times \mathbf { F } _ { m } ^ { * } [ k ] ( e ^ { j \Phi } ) ^ { * } \mathbf { H } _ { m } ^ { * } [ k ] \times \mathbf { I } _ { \mathrm { i n t e r } } ^ { - 1 } \Biggr ) \Biggr ] , ~ } \end{array}\tag{36}
$$

where $\mathbf { \Gamma } _ { \mathrm { i n t e r } }$ is the interference-plus-noise covariance matrix:

$$
\mathbf { \Gamma } _ { \mathrm { { i n t e r } } } = P _ { t } \sum _ { \stackrel { n = 1 } { n \neq m } } ^ { M } \mathbf { H } _ { m } [ k ] e ^ { j \Phi } \mathbf { F } _ { n } [ k ] \mathbf { F } _ { n } ^ { * } [ k ] ( e ^ { j \Phi } ) ^ { * } \mathbf { H } _ { m } ^ { * } [ k ] + \sigma ^ { 2 } \mathbf { I } _ { N _ { r } } .\tag{37}
$$

Subsequently, the reformulation in (10) still applies by replacing the objective function with (36). This multi-user objective will be used to construct the training loss for the proposed GNNs in (12).

## B. Extended Graph Model and Updating Rules

Next, we extend the graph model and updating rules of the proposed NU-GNN and EU-GNN to the multi-user scenario.

1) Graph Model: We extend the graph model in Fig. 3 to the multi-user case by introducing additional subcarrier nodes for new users, as illustrated in Fig. 9. Specifically, the subcarrier nodes corresponding to each user are grouped into a user-specific subcarrier set. This extension retains the permutation equivariance property of the proposed GNN, as stated in Proposition 3.

Proposition 3: The output of each GNN updating layer remains permutation equivariant with respect to both the subcarrier and user orders under the proposed multi-user graph extension.

## Proof: See Appendix C.

Proposition 3 implies generalization across varying subcarrier and user orders without retraining.

2) GNN Updating Rules: Since the underlying graph model is similar to the single-user case (except with a different loss function that dictates the interaction among nodes), the updating rules of the proposed NU-GNN and EU-GNN are directly applicable to the multi-user case. Accordingly, the message generation and representation update operations follow the same rules as in the single-user setting in Eqs. (16)– (23). The only difference lies in the node indexing, where the subcarrier index k is extended to $( k , m )$ to account for the m-th user. Accordingly, the neighboring set is extended to $( k , m ) \in \mathcal { N } ( \mathbf { x } )$ during aggregation, through which the interuser interference is naturally captured from neighboring nodes and directly accounted for in the loss function.

3) Beamformer Reconstruction: To satisfy the transmit power constraint in (10), we first reconstruct the digital beamformer for each user m on each subcarrier k. Specifically, $\mathbf { F } _ { k , m } \ = \ { \mathrm { r e s h a p e } } \Big ( \mathbf { c } _ { k , m } ^ { ( L ) } \left[ 1 : N _ { \mathrm { R F } } \times N _ { d } \right] , ( N _ { \mathrm { R F } } , N _ { d } ) \Big ) \ +$ j reshape $\left( \mathbf { c } _ { k , m } ^ { \left( L \right) } \left[ N _ { \mathrm { R F } } \times N _ { s } : 2 N _ { \mathrm { R F } } \times N _ { d } \right] , \left( N _ { \mathrm { R F } } , N _ { d } \right) \right)$ . The reconstructed digital beamformers are then concatenated to form $\mathbf { F } _ { k } = \left\lceil \mathbf { F } _ { k , 1 } \quad \cdot \cdot \cdot \quad \mathbf { F } _ { k , M } \right\rceil$ . The resulting digital beamformer is subsequently normalized according to (25), and the reconstruction of the analog beamformer follows the same procedure as in (24).

## VII. COMPUTATIONAL COMPLEXITY ANALYSIS

In this section, we provide an analysis of the forward computation complexity of our three GNN models during inference, and compare them with that of a traditional optimization method, the AMO algorithm in [5]. Since the training of the GNNs can be done offline before the models are deployed, we do not analyze training complexity. Instead, we focus on inference complexity since it captures the runtime computational requirements and is relevant for applying the trained GNNs in a practical communication system.

## A. Proposed GNN Structures

Our NU-GNN and EU-GNN structures both consist of 4 different MLPs of the same depth but different layer sizes. The computational complexity of the i-th MLP depends on the number of hidden layers d, and the size of each layer including input size $n _ { i } ,$ each hidden layer size $2 n _ { i } ,$ and output size $m _ { i }$ . The forward computational complexity can be calculated as $O \left( 2 n _ { i } ^ { 2 } + 4 d n _ { i } ^ { 2 } + 2 n _ { i } m _ { i } \right)$ . Typically, $n _ { i } \mathrm { ~ } \gg \mathrm { ~ } m _ { i }$ in each MLP, thus the complexity order per MLP can be simplified as $O \left( d n _ { i } ^ { 2 } \right)$ . During the forward propagation process, MLP $f _ { 2 } ^ { a } ( \cdot )$ is computed once, while the other three are computed K times. Therefore, we can express the overall complexity for one GNN updating layer as $O \left( d K n ^ { 2 } \right)$ , where $n = \operatorname* { m a x } \{ n _ { i } , i \in [ 1 , 4 ] \}$

AN-GNN shares a similar structure with NU-GNN but with an aggregation that includes a fully connected linear layer, contributing a computational complexity of $O \left( 2 N _ { t } N _ { r } + N _ { t } N _ { \mathrm { R F } } + 2 N _ { \mathrm { R F } } N _ { s } \right)$ . Unlike NU-GNN, the subcarrier node features in the AN-GNN are computed not through MLPs but via matrix multiplication and performing the SVD. The overall forward computation complexity is $O \left( d K n ^ { 2 } \right) + O \left( 2 N _ { t } N _ { r } + N _ { t } N _ { \mathrm { R F } } + 2 N _ { \mathrm { R F } } N _ { s } \right) + O ( N _ { t } N _ { \mathrm { R F } } +$ $K \dot { N } _ { r } N _ { t } \dot { N _ { \mathrm { R F } } } + K N _ { r } ^ { 2 } N _ { \mathrm { R F } } ^ { t } )$ , where the first term remains the dominant one.

Given the above complexity order for each GNN updating layer, the total complexity for all L layers of a GNN is $O \left( d L K n ^ { 2 } \right)$ . For the beamformer reconstruction operations in (24) and (25), the element-wise exponential operation on a matrix requires $O ( N _ { t } N _ { \mathrm { R F } } )$ complexity, while the Frobenius norm and matrix multiplication requires $O ( K N _ { t } N _ { \mathrm { R F } } N _ { s } )$ . Among all these complexities, the dominant term is still the calculation of L updating layers. As a result, the overall computational complexity for processing one sample through the GNN can be expressed as $O \left( d L K n ^ { 2 } \right)$ , where $n = 2 N _ { t } N _ { r } + N _ { t } N _ { \mathrm { R F } }$ for the NU-GNN and AN-GNN, and $n = 2 N _ { t } N _ { r } + + N _ { t } N _ { \mathrm { R F } } +$ $2 N _ { \mathrm { R F } } N _ { s }$ for the EU-GNN. Since $N _ { t } \ \gg \ N _ { \mathrm { R F } } \ \ge \ N _ { s }$ and $N _ { r } \ > \ N _ { s } ,$ , for all cases, the complexity of processing one GNN sample can be simplified as

$$
O ( d L K N _ { t } ^ { 2 } N _ { r } ^ { 2 } ) ,\tag{38}
$$

where d and $L$ are GNN settings, and K, $N _ { t }$ and $N _ { r }$ are wireless network settings.

## B. AMO Algorithm [5]

For the AMO algorithm in [5], the computational steps involve manifold optimization (MO) to update the value of W. According to Algorithm 1 in [5], the key computational costs are from the Armijo backtracking line search, retraction, cost function, and the computation of the Riemannian gradient. Assume that an average of J iterations is required to find an appropriate step size using the Armijo method. The complexity of computing the Riemannian gradient at each update step along with the cost function computation is $O \left( 3 J \bar { K } [ N _ { t } ^ { 3 } N _ { s } ^ { 2 } N _ { \mathrm { R F } } ] + 2 J K [ N _ { t } ^ { 3 } ( N _ { \mathrm { R F } } ) ^ { 2 } N _ { s } ] + 5 J \bar { N _ { t } ^ { 2 } } ( N _ { \mathrm { R F } } ) ^ { 2 } \right)$ Since $N _ { s } \le N _ { \mathrm { R F } }$ , the second term is dominant.

In each Riemannian update step, assume an average of I iterations for W to converge to the current optimal solution given F[k]. The complexity of retraction, which normalizes the amplitude of elements in W to 1, is $O ( I N _ { t } N _ { \mathrm { R F } } )$ , which is negligible compared to the line search. Furthermore, computing the K beamformers $\mathbf { F } [ k ]$ requires matrix inversion and multiplication. Combined with the power constraint, the resulting complexity is $\begin{array} { r } { O \left( K ( N _ { \mathrm { R F } } ) ^ { 3 } + 2 K N _ { t } N _ { \mathrm { R F } } N _ { s } \right) } \end{array}$ , which remains minor since $N _ { \mathrm { R F } } \ll N _ { t }$

Therefore, the overall complexity of the AMO algorithm, which requires an average of M iterations of Riemannian updates to converge, can be approximated as:

$$
O \left( M I J K N _ { t } ^ { 3 } ( N _ { \mathrm { R F } } ) ^ { 2 } N _ { s } \right) ,\tag{39}
$$

where J is the number of iterations required for each line search to find the step size, I for the current optimal value of W, and M for the convergence of the algorithm.

## C. Comparison Between GNNs and AMO

Since $N _ { t } \gg N _ { \mathrm { R F } } \ge N _ { s }$ and $N _ { r } \ \ge \ N _ { s }$ , when comparing the complexities of the two algorithms, the dominant term is the highest degree term related to $N _ { t }$ . From (38), we observe that the complexity of a GNN pass is of second order in $N _ { t } ,$ whereas for AMO in (39), it reaches cubic. This indicates that as we increase the number of antennas, especially in massive MIMO systems, the GNN model offers a significant advantage in terms of computation cost savings. In addition, as we increase the number of subcarriers K in an OFDM system, the complexity of both algorithms grows linearly. However, the multiplier factor associated with $K N _ { t } ^ { 3 }$ in (39) is much larger than the one associated with $K N _ { t } ^ { 2 }$ in (38). This implies that the complexity of the AMO algorithm will increase much faster than that of our proposed GNN with respect to the transmit antenna array size and number of subcarriers.

## VIII. NUMERICAL SIMULATIONS

## A. Simulation System Settings

Here we describe the settings for the simulation system and for training the GNN structures.

1) Communication System Settings: We use a carrier frequency of $f _ { c } = 1 4 2 \mathrm { G H }$ z and a bandwidth of $B = 2 0 \mathrm { G H z }$ , with $K = 4$ subcarriers selected for the offline training process. The number of subcarriers will be varied during testing and evaluation. The BS employs an $N _ { t } = 6 4$ UPA antenna system, equipped with $N _ { s } = N _ { \mathrm { R F } } = 4$ RF chains, while the UE uses an $N _ { r } = 8 ~ \mathrm { U P A }$ antenna system. In all simulations, the persubcarrier transmit power is set to be equal to $P _ { t }$ , except for one result in Fig. 11(b) (noted $\mathrm { w i t h } \leq P _ { t } )$

The channel generation follows the measurement-based mmWave double directional model in [18], which includes both large and small scale fading effects. The channel parameters are defined with $N _ { \mathrm { c l } } = 2$ clusters and $N _ { \mathrm { r a y } } = 3$ rays per cluster. The complex path gain $\alpha _ { i l } = a _ { i l } e ^ { j \psi _ { i l } }$ is a product of the real large-scale fading amplitude $a _ { i l }$ and the channel phase component $\psi _ { i l } \in [ 0 , 2 \pi )$ . Here $a _ { i l }$ follows the probabilisticbased pathloss model described in [18], [19], with an averaged transmit power $P _ { t } ~ = ~ 3 6 \mathrm { { d B m } }$ and a noise power spectral density of −174dBm/Hz. The UE distribution is followed by a uniform location distribution within a circular area, with a TX-RX distance ranging in [10,100]m. The propagation path delay follows $\tau _ { i l } \sim \mathcal { U } ( 0$ , 100ns) [18]. Both the azimuth and elevation angles are following a wrapped Gaussian distribution [18], [20]. The antenna elements are spaced at $\begin{array} { r } { d = \frac { \lambda _ { c } } { 2 } } \end{array}$

2) Machine Learning Settings: During offline training, we initialized the GNN as $\mathbf { x } \sim \mathcal { U } [ 0 , 2 \pi ) , \mathbf { c } _ { k } \sim \mathcal { N } ( 0 , 1 )$ , and $\mathbf { e } _ { k }$ as in (14). All MLPs had two hidden layers, each with the number of neurons twice the MLP input size, followed by ReLU activations. For EU-GNN, we applied dropout (30%) to hidden layers during message generation to improve convergence and generalization. We used the Adam optimizer with a $5 \times 1 0 ^ { - 4 }$ learning rate, halved every 200 epochs for NU-GNN and EU-GNN. This value was empirically selected as the best within the range of $[ 1 0 ^ { - 5 } , 1 0 ^ { - 3 } ]$ . AN-GNN used a warm restart scheduler [21], cyclically sweeping the learning rate in $[ 5 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 4 } ]$ to enhance performance. Training used mini-batches of 100 samples, with 100 batches per epoch. All three GNN models have $L = 2$ layers, as the bipartite graph can be fully traversed in two hops. This choice of L also helps mitigate the oversmoothing effect [22], which degrades the ability of different nodes to converge to distinct representations. We observed such effects in early experiments when using $L = 4$

![](images/fdecb6dcb396d3ed78eea3bd3974262b778d189015d25cfd20024aa16cd18180.jpg)

![](images/348a8ccb6a05fc3c2239e8b08caa9f8bfa09ac28d07139084a48d5f1470d2618.jpg)  
(a)  
(b)  
Fig. 10: Convergence of the proposed GNNs training for $P _ { t } = 3 6 \mathrm { d B m }$ , $N _ { t } =$ 64, $K = 4 , f _ { c } = 1 4 2 \mathrm { G H z } ,$ and $B = 2 0 \mathrm { G H z }$ . Training curves are shown as shaded regions, while solid lines indicate validation curves. (a) Comparison among the three proposed GNN structures, with traditional methods and FD. (b) Comparison of NU-GNN with three other ML benchmark methods.

3) Baseline Schemes: We compare our proposed GNNs with the following baseline schemes:

• FD: Fully digital beamforming method, which applies SVD to the channel matrix of each subcarrier to obtain the optimal fully digital beamformer.

• AMO [5]: Iteratively updating the analog beamformer via manifold optimization and alternating with digital beamforming update.

• ICD [6]: Iteratively updating the analog beamformer via coordinate descent and alternating with digital beamformers.

• AV-all [4]: Averaging the array response vectors across all subcarriers to construct the analog beamformer.

• MCM [4]: Using dominant eigenvectors of the mean channel matrix across all subcarriers for analog beamforming.

• FNN: A fully-connected neural network (FNN) is constructed to learn both the analog and digital beamformers.

• AN-FNN: A fully-connected neural network (FNN) is constructed to learn only the analog beamformer.

• LCMLP-GNN [16]: A GNN method using a linear combination of the learned representations without messagepassing as updating layers.

## B. ML Performance and Complexity Analysis

1) Data Rate Convergence: Fig. 10 shows the convergence of the proposed GNN structures. The NU-GNN outperforms the AMO benchmark and also outperforms all the other ML benchmark methods. Additionally, the EU-GNN converges to a spectral efficiency approaching that of the AMO, and the

AN-GNN achieves performance similar to that of the ICD.

2) Ablation Study: In our ablation study, we analyze the effect of the message and representation update methods, the choice of information embedding across all our proposed GNN structures, and the impact of attention specifically in AN-GNN. The comparison results across these GNN variants are summarized in Table II.

First, we compare our designed efficient structure with a more complex message-passing structure. NU-GNN (ExtMsg) incorporates extra messages by additionally concatenating $\mathbf { m } _ { k } ^ { d ( l - 1 ) }$ in the inputs of generating $\mathbf { m } _ { k } ^ { a ( l ) }$ in $\operatorname { E q . }$ . (16), and similarly includes $\mathbf { \dot { m } } _ { k } ^ { a ( l - 1 ) }$ in $\operatorname { E q . }$ . (17). Although this seems to provide more information during representation updates, the information carried by the additional messages is already contained in our simpler message-passing structure, and hence is double-counted, introducing unnecessary redundancy and complexity without increasing the convergence values.

For EU-GNN, we observe that EU-GNN (ExtMsg) improves the final converged value slightly, possibly because in EU-GNN, the CSI information is gradually overwritten during updates, causing the process to rely more on the graph structure than the input features. This motivated us to test EU-GNN $( \mathbf { h } _ { k } )$ by including the CSI inputs $\mathbf { h } _ { k }$ as additional edge features in Eq. (14), which similarly improves convergence values slightly. However, as the overall performance remains comparable while training time increases, we ultimately chose the simplest EU-GNN variant to balance performance and computational efficiency.

Finally, we explored the impact of incorporating attentionbased aggregation. The AN-GNN with attention achieves a modest improvement in converged value compared to the other at a small increase in training time. Since the training of the AN-GNN with attention is still smaller than NU-GNN and EU-GNN, we selected this attention-based AN-GNN since it offers superior generalization ability, as shown in Fig. 16.

3) Offline Training Time: Table II provides the total training time, per-epoch mean training time, standard deviation, and trainable parameters for the three proposed GNN models. Taking the largest model EU-GNN as the baseline, AN-GNN reduces parameters by 80% and speeds up training by 25%, while NU-GNN achieves a 59% parameter reduction and a 13% speed-up. AN-GNN’s shorter mean training time offers a meaningful advantage if the training is done online, where only fine-tuning is needed for new incoming data. EU-GNN exhibits the lowest standard deviation, indicating more consistent updates and greater training stability. On the other hand, AN-GNN shows the highest standard deviation, primarily due to the SVD computations in the digital beamformers at each algorithm step, which introduce runtime fluctuations.

4) Online Inference Running Time: The practicality of our GNNs is highlighted in the inference time comparison in Table II. Across $1 0 ^ { 3 }$ channel realizations, AMO is over 32 times or more than an order of magnitude slower than the proposed GNNs. This is because AMO needs to repeat the alternative optimization process for each subcarrier and channel realization, while the GNNs simply utilize the pretrained models to perform feed-forward computation, producing results quickly by directly scaling up the number of subcarrier nodes. Furthermore, the GNNs exhibit computation time standard deviations three orders of magnitude lower than that of AMO, ensuring highly stable computation time per CSI update. Notably, although AN-GNN has the fewest trainable parameters, this advantage is compromised by the SVD and matrix inversion in computing the digital beamformers, which ultimately leads to a slower inference time.

TABLE II: Ablation study of proposed GNN variants. See the text for the detailed description of different variants.
<table><tr><td rowspan=2 colspan=1>Methods</td><td rowspan=2 colspan=1>ConvergedValues(bps/Hz)</td><td rowspan=1 colspan=3>Training Time</td><td rowspan=1 colspan=2>Inference Time</td><td rowspan=1 colspan=2>Storage</td><td rowspan=2 colspan=1># Params</td></tr><tr><td rowspan=1 colspan=1>Mean/Epoch(sec)</td><td rowspan=1 colspan=1>Std(sec)</td><td rowspan=1 colspan=1>Total(min)</td><td rowspan=1 colspan=1>Mean(sec)</td><td rowspan=1 colspan=1>Std(sec)</td><td rowspan=1 colspan=1>Mean(Mb)</td><td rowspan=1 colspan=1>Std(Mb)</td></tr><tr><td rowspan=1 colspan=1>NU-GNN</td><td rowspan=1 colspan=1>6.7533</td><td rowspan=1 colspan=1>205.19</td><td rowspan=1 colspan=1>5.21</td><td rowspan=1 colspan=1>2386.43</td><td rowspan=1 colspan=1>0.0866</td><td rowspan=1 colspan=1>0.0006</td><td rowspan=1 colspan=1>567.97</td><td rowspan=1 colspan=1> $1 . 5 \times 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1>60.202M</td></tr><tr><td rowspan=1 colspan=1>NU-GNN (ExtMsg)</td><td rowspan=1 colspan=1>6.3426</td><td rowspan=1 colspan=1>230.09</td><td rowspan=1 colspan=1>11.87</td><td rowspan=1 colspan=1>2684.38</td><td rowspan=1 colspan=1>0.0974</td><td rowspan=1 colspan=1>0.0009</td><td rowspan=1 colspan=1>634.45</td><td rowspan=1 colspan=1> $1 . 5 \times 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1>74.054M</td></tr><tr><td rowspan=1 colspan=1>EU-GNN</td><td rowspan=1 colspan=1>6.2409</td><td rowspan=1 colspan=1>234.03</td><td rowspan=1 colspan=1>4.11</td><td rowspan=1 colspan=1>2745.45</td><td rowspan=1 colspan=1>0.0907</td><td rowspan=1 colspan=1>0.0005</td><td rowspan=1 colspan=1>773.64</td><td rowspan=1 colspan=1> $\overline { { 1 . 5 \times 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>133.369M</td></tr><tr><td rowspan=1 colspan=1>EU-GNN (add h)</td><td rowspan=1 colspan=1>6.3248</td><td rowspan=1 colspan=1>244.39</td><td rowspan=1 colspan=1>6.37</td><td rowspan=1 colspan=1>2882.98</td><td rowspan=1 colspan=1>0.0945</td><td rowspan=1 colspan=1>0.0008</td><td rowspan=1 colspan=1>1014.53</td><td rowspan=1 colspan=1> $\overline { { 1 . 5 \times 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>196.427M</td></tr><tr><td rowspan=1 colspan=1>EU-GNN (ExtMsg)</td><td rowspan=1 colspan=1>6.4022</td><td rowspan=1 colspan=1>269.96</td><td rowspan=1 colspan=1>8.09</td><td rowspan=1 colspan=1>3179.65</td><td rowspan=1 colspan=1>0.0986</td><td rowspan=1 colspan=1>0.0006</td><td rowspan=1 colspan=1>890.55</td><td rowspan=1 colspan=1> $\overline { { 1 . 5 \times 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>163.803M</td></tr><tr><td rowspan=1 colspan=1>AN-GNN</td><td rowspan=1 colspan=1>5.7004</td><td rowspan=1 colspan=1>177.09</td><td rowspan=1 colspan=1>6.7894</td><td rowspan=1 colspan=1>2078.56</td><td rowspan=1 colspan=1>0.1174</td><td rowspan=1 colspan=1>0.0008</td><td rowspan=1 colspan=1>356.28</td><td rowspan=1 colspan=1> $\overline { { 1 . 5 \times 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>24.411M</td></tr><tr><td rowspan=1 colspan=1>AN-GNN (No Attn)</td><td rowspan=1 colspan=1>5.6653</td><td rowspan=1 colspan=1>161.62</td><td rowspan=1 colspan=1>4.76</td><td rowspan=1 colspan=1>1905.48</td><td rowspan=1 colspan=1>0.0971</td><td rowspan=1 colspan=1>0.0008</td><td rowspan=1 colspan=1>356.27</td><td rowspan=1 colspan=1> $1 . 5 \times 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1>24.408M</td></tr><tr><td rowspan=1 colspan=1>AMO [5]</td><td rowspan=1 colspan=1>6.1084</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.8991</td><td rowspan=1 colspan=1>1.6632</td><td rowspan=1 colspan=1>4677.82</td><td rowspan=1 colspan=1>1846.4</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>ICD [6]</td><td rowspan=1 colspan=1>5.8397</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.3681</td><td rowspan=1 colspan=1>0.2798</td><td rowspan=1 colspan=1>53.7</td><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=1>-</td></tr></table>

![](images/337fabb5183f3fc117c32279fbc87c055ddd1e55b566fbe972681e8b0cb64d75.jpg)  
(a)

![](images/abff50d5f1b38a4a8cd285c843b2f6f8b5ad50ac2e17e1a5911fa10c6b5640f2.jpg)  
(b)  
Fig. 11: Spectral efficiency achieved by different beamforming design algorithms with $K = 6 4$ subcarriers, averaged over $1 0 ^ { 3 }$ channel realizations. (a) Comparison among the three proposed GNN structures, with FD and traditional methods. (b) Comparison of NU-GNN with three other benchmarks.

5) Dynamic Memory Allocation: Table II further emphasizes the GNNs’ computational efficiency in memory use. Since we use the pre-trained GNN models directly during the online inference phase, the amount of dynamic memory required for each channel realization remains constant, where EU-GNN requires the most dynamic memory allocation, and AN-GNN the least. In contrast, AMO’s repeated optimization leads to high variability and an average memory usage nearly 8 times higher than that of the GNN models. This highlights GNNs efficiency and stability in resource allocation, making them more suitable for practical deployment.

## C. Communication Performance

During the online inference phase for simulation, we increased the number of subcarriers to $K = 6 4$ , and averaged all presented simulation results over $1 0 ^ { 3 }$ channel realizations.

1) Spectral Efficiency vs P : Fig. 11 shows that our NU-GNN and EU-GNN outperform all traditional methods, including the two iterative optimization algorithms (AMO [5] and ICD [6]). Among the methods that all use singular-value decomposition closed form to calculate the digital beamformers (ICD [6], MCM [4], AV-all [4]), our AN-GNN achieves the best performance. In addition, as shown in Fig. 11(b), the performance difference between equal and upper-bounded subcarrier power allocation is relatively small, indicating that the observed performance gain mainly comes from the proposed GNN-based hybrid beamforming design rather than the specific power normalization strategy.

![](images/b27c49565251535762e47910325fcaa77f0749560d73866264e3577b4b01647f.jpg)  
Fig. 12: Spectral efficiency versus the number of transmitter antennas $N _ { t } .$ , with $K = 6 4$ subcarriers, $P _ { t } = 2 0 \mathrm { { d B m } } .$ , averaged over $1 0 ^ { 3 }$ channel realizations.

2) Spectral Efficiency vs Antenna Array Size: We evaluated our proposed GNN models across different antenna array sizes by training additional GNNs with different $N _ { t }$ . Shown in Fig. 12, as $N _ { t }$ increases, the advantage of our proposed models becomes more significant compared to traditional methods, and remains competitive with regard to fully digital beamforming performance. During training, we also observed that smaller learning rates benefit larger arrays, likely due to increased sensitivity in optimization.

Fig. 13 further shows that the inference time of our three GNNs grows only slightly as the number of transmit antennas $N _ { t }$ increases, while AMO and ICD increase exponentially (note the y-axis is in log scale). These trends align well with our theoretical analysis in (38), (39), and the $O ( N _ { t } ^ { 3 } )$ complexity described in [6] for ICD, where the computational complexities of AMO and ICD grow cubically with $N _ { t } ,$ while our GNNs grow quadratically. Although LCMLP-GNN benefits from a simpler structure as a linear-combination of MLPs output representation without a message passing structure, resulting in faster online inference, our GNNs achieve significantly better spectral efficiency, outperforming LCMLP-GNN by approximately 125%.

![](images/7524c2af7e4a04aa408403b354a149ce7108bfa752a145535a93410038fa5229.jpg)  
Fig. 13: Inference running time comparison per CSI update on NVIDIA A100 GPU, averaged over $\mathrm { 1 0 ^ { 3 } C S I }$ samples, versus the number of antennas $N _ { t }$

![](images/dab4b2664f45fe5334027d4875bd4191b31e8eebc8fd18fc54f9a1d319293e10.jpg)  
(a)

![](images/8a40f58e347990fd5ed7a8a418b89155bf2af4ba64b6c6ec6129cfe1321f2d6a.jpg)  
(b)  
Fig. 14: Spectral efficiency versus channel fractional bandwidth for different beamforming design algorithms, with $K = 6 4$ subcarriers, $P _ { t } = 2 0 \mathrm { { d B m } }$ averaged over $\mathrm { 1 0 ^ { 3 } }$ channel realizations. The central carrier frequency is $f _ { c } = $ 142GHz, and $\begin{array} { r } { b = \frac { B } { f _ { c } } . } \end{array}$ , where B is the communication channel bandwidth. The parameters in parentheses indicate the system settings used during training. (a) Comparison among the three proposed GNN structures, with an additional NU-GNN trained with B=30GHz, and $K = 8$ subcarriers. (b) Comparison of NU-GNN with four other benchmarks.

3) Beam Squinting Resiliency: Fig. 14 shows that the proposed GNN structures effectively mitigate beam squinting. Let $\begin{array} { r } { \boldsymbol { b } = \frac { \boldsymbol { B } } { f _ { c } } } \end{array}$ represent the fractional bandwidth. As b increases, the beam squinting effect becomes pronounced as seen in all the baselines. Both our NU-GNN and EU-GNN, on the other hand, achieve a starkly superior resiliency to beam squinting than all baseline methods. Interestingly, both these GNN structures exhibit an optimal fractional bandwidth value for mitigating beam squinting. We further examine the effect of training bandwidths (at B = 20GHz and B = 30GHz) on the generalization ability of our GNN structures to larger bandwidths, and show that a larger training bandwidth leads to better generalization and strong resiliency to beam squinting. For AN-GNN, although it exhibits weaker beam squinting resiliency compared to NU-GNN and EU-GNN, it is still more resilient to beam squinting than the ICD algorithm [6].

The resiliency of our GNN models against beam squinting is further illustrated in Fig. 15, showing the emitted power heatmap across subcarriers (left) and beam patterns at two representative subcarriers (right). In Fig. 15(a), the slope of the bright regions reflects the degree of beam squinting: a steep or vertical slope indicates stronger resiliency (the same beam direction across all subcarriers), while the more tilted or curved patterns suggest the stronger beam squint. As shown, AMO exhibits noticeable directional shifts across all three channel samples, whereas our NU-GNN and EU-GNN maintain strong resiliency. Although AN-GNN exhibits minor shifts, it remains within an acceptable range compared to AMO. MCM and AVall suffer from visible misalignment and display multiple side lobes across the subcarriers, leading to cluttered patterns that indicate unstable beam behavior and poor resiliency.

![](images/12939f61c0cff1f7ca1badb3cc61efa1e0f8c1d77fae8bb48e5af7767bcd0263.jpg)  
(a)

![](images/e5aad38ee167b8b233d25e0c417181c6637d30db68a4f1b249c03e0e15e67196.jpg)  
(b)  
Fig. 15: Visualization of beam squinting effects across three CSI samples. (a) Heatmap: Each row represents a beamforming method, with columns showing three distinct CSI samples. The x-axis is beam direction (degrees), and the y-axis is subcarrier index $( K = 6 4 )$ for each subplot. Brighter colors indicate higher power gain. The slope of the bright region across subcarriers reflects the degree of beam squinting: a steep or vertical slope indicates less beam squint. (b) Beam patterns: The plot shows the beam patterns of the first and last subcarriers for a CSI sample. In the absence of beam squint, the two beam patterns should align perfectly.

Fig. 15(b) compares the beam patterns of the first and last subcarriers for a CSI sample. In the absence of beam squint, the two beam patterns should align perfectly. The fully digital beamformer achieves this while AMO shows a $5 ^ { \circ }$ shift, reflecting a moderate beam squinting effect. Both NU-GNN and EU-GNN exhibit near-perfect alignment, and AN-GNN shows only a minor deviation within 2<sup>◦</sup>. In contrast, MCM and AV-all show distorted beam patterns with multiple side lobes and significant main lobe shifts. These irregularities suggest unfocused beam steering behavior and severe beam squinting.

4) Generalization over the Number of Subcarriers: We tested the generalization ability of the proposed GNNs by varying the number of subcarriers and applying the trained MLPs to new subcarriers without retraining. As shown in Fig. 16, even though we used K = 4 subcarriers during the offline training process for all GNNs to save training time, in the online inference process, all three proposed GNN structures demonstrated excellent generalization ability to much larger K values (up to $K = 6 4 )$ without the need for retraining. Both the NU-GNN and EU-GNN structures maintain consistently high performance, on par with AMO, across all subcarrier configurations. Notably, for the AN-GNN structure, when comparing the blue and the light blue lines, the use of the attention mechanism in the aggregation operation (see Fig. 8) significantly outperforms the element-wise mean aggregation function by improving the GNN’s generalization ability. This demonstrates the importance of including attention in the AN-GNN structure as it can dynamically assign a weight to signify the relative importance of each newly added subcarrier.

![](images/ddfd06e49b7dcc795b8ce574cce17baca2a13fe4048081d1e11e6ed51efa6340.jpg)

![](images/bc3090b2e87a5aa04cafc3e93721b13594b6b0114779a802c27f6563d13efcce.jpg)  
(a)  
(b)

Fig. 16: Spectral efficiency versus the number of subcarriers, when varying K from 4 to 64, $P _ { t } \ = \ \mathrm { \bar { 2 } 0 d B m }$ , averaged over $1 0 ^ { 3 }$ channel realizations. The GNN models were trained with $K \stackrel { = } { = } 4$ and then applied to all systems without retraining, with an additional AN-GNN trained without an attentionbased aggregation. (a) Comparison among the three proposed GNN structures. (b) Comparison of NU-GNN with three other benchmarks.  
![](images/89325b4b7c800b3c9813edfe33fd7cacbcfdbcfc7564ca1a4e90bde7ec34ada2.jpg)  
Fig. 17: Spectral efficiency versus the location error variance $\sigma ^ { 2 } .$ , with $K = 6 4$ subcarriers, $P _ { t } = 2 0 \mathrm { { d B m } }$ , averaged over $1 0 ^ { 3 }$ channel realizations.

![](images/0f942a48359853f88aa21208540ed70f097848fd8c286919cf3b62fcef149aab.jpg)  
Fig. 18: Spectral efficiency versus the number of users, with $K ~ = ~ 6 4$ subcarriers, $P _ { t } = 2 0 \mathrm { { d B m } . }$ , averaged over $1 0 ^ { 3 }$ channel realizations.

5) Generalization over the Number of Users:

We further evaluate the generalization capability of the proposed GNN models in multi-user scenarios. The models are trained with $N _ { d } = 2$ data streams and $N _ { r } = 2$ receive antennas per user. The BS is equipped with $N _ { \mathrm { R F } } ~ = ~ 1 2 ~ \mathrm { R F }$ chains, $N _ { t } = 6 4$ transmit antennas, serving $M = 3$ users over $K = 4$ subcarriers. Both NU-GNN and EU-GNN have $L = 3$ layers, which is chosen as it yields the best performance among the tested values $L = [ 2 , 3 , 4 ]$ (note that more GNN layers can lead to the over-smoothing effect and reduce learning). During training, weight decay regularization with a coefficient of $1 0 ^ { - 4 }$ is employed to improve the performance. We then assess the generalization ability of the trained GNN models under different user configurations.

The baseline methods considered in this comparison are

• WMMSE [23]: Fully digital beamforming method, which optimizes the beamformers by the weighted minimal meansquare error (WMMSE) algorithm, serving as a performance upper bound.

• AMO [5]: Iteratively updating the analog beamformer via manifold optimization and alternating with digital beamforming update.

• BD [24]: Fully digital beamforming method, which applies block diagonalization to suppress inter-user interference.

• MRT [25]: A fully digital beamforming method that maximizes the received signal power by aligning the precoder with the channel.

• MU-LCMLP-GNN [15]: An existing GNN method using a linear combination of the learned representation without message-passing for the multi-user scenario.

During the online inference process, all trained models are tested with $K = 6 4$ subcarriers, which is much higher than the number of trained subcarriers $K = 4$ . As shown in Fig. 18, our proposed GNN models demonstrate strong generalization across different numbers of users and outperform AMO, MU-LCMLP-GNN, and the two fully digital beamforming methods, BD and MRT, across all user configurations. However, a performance gap remains between our proposed models and the fully digital WMMSE benchmark. We conjecture that this gap can be further shortened by designing new GNN architectures to explicitly model the inter-user interference. The exploration of this direction is left for future work.

6) Robustness to Imperfect CSI: To further assess the robustness of our proposed methods under practical conditions, we evaluate their performance under imperfect CSI scenarios. Since we use the double-directional channel model in (6), we focus on the impact of UE location errors on channel CSI. Because this model relies on an accurate knowledge of the path angles, even small mismatches between the estimated and true UE locations can lead to significant CSI errors.

Denote the true UE location as $\mathbf { p } ~ = ~ [ x , y ] ^ { T }$ , and the estimated UE location as $\hat { \bf p } = [ \hat { x } , \hat { y } ] ^ { T }$ . The UE location error

$$
\Delta \mathbf { p } = [ \Delta x , \Delta y ] ^ { T } = [ \hat { x } - x , \hat { y } - y ] ^ { T } ,\tag{40}
$$

which can be modeled as a two-dimensional Gaussian distribution centered at the true location:

$$
\Delta \mathbf { p } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { 2 } ) .\tag{41}
$$

Here, the variance $\sigma ^ { 2 }$ controls the spread of the location error. In this evaluation, we use the imperfectly estimated channel $\hat { \bf { H } } ( \hat { \bf { p } } )$ to train the GNNs to learn the beamformers, but compute the spectral efficiency using the true channel $\mathbf { H } ( \mathbf { p } )$ to reflect realistic performance under imperfect CSI.

Fig. 17 shows the spectral efficiency versus the location error variance $\sigma ^ { 2 }$ . As the location error increases, all methods show performance degradation. Notably, the more optimal a design, such as the fully digital beamforming method, the more sensitive it becomes to location errors, as it relies heavily on precise CSI for each subcarrier. In contrast, our proposed NU-GNN and EU-GNN maintain better resiliency and achieve stronger spectral efficiency than fully digital beamforming across the entire error range. Due to the data-driven nature of these learning-based methods, the GNNs are trained to capture statistical patterns rather than depend solely on exact CSI, giving them better generalization ability and making them inherently more robust to such imperfections compared to traditional methods like AMO, leading to higher performance.

## IX. CONCLUSION

We proposed three novel GNN structures for efficient hybrid beamforming design in multicarrier wideband MIMO systems, while effectively mitigating the beam squinting effect. By capturing the unique structure of hybrid beamforming in an OFDM system via a bipartite graph, we designed an efficient yet highly effective message-passing mechanism to optimize the GNN performance. The proposed GNNs not only optimize both digital and analog beamforming matrices but also adjust them dynamically to any change in the number of subcarriers by scaling the subcarrier nodes without the need for retraining.

Among the proposed 3 GNN structures, NU-GNN achieves the highest spectral efficiency performance with the lowest average inference time, while EU-GNN is the most stable during both training and inference. Interestingly, the hybrid AN-GNN structure exhibits significant computational savings during training because of fewer parameters, but this advantage vanishes at inference time because of the required computation for digital beamformers.

Comparing against traditional signal processing algorithms and existing GNN designs, our GNNs offer significant advantages in spectral efficiency, computational complexity, running time, and memory requirements, while achieving superior performance in beam squinting mitigation and robustness to imperfect CSI. These advantages make the proposed GNNs a viable solution for real-time beamforming adaptation. Finally, we demonstrated that the proposed NU-GNN and EU-GNN can be directly extended to multi-user scenarios, which outperform all existing ML multi-user solutions and exhibit strong generalization across users.

## APPENDIX A: PROOF OF PROPOSITION 1

Permutation equivariance is defined as $f ( \mathbf { I I x } ) = \mathbf { I I } f ( \mathbf { x } )$ where Π $\mathbf { \Psi } \in \mathbf { \Psi } \mathbb { R } ^ { N \times N }$ is a permutation matrix, x is the input, $f ( \cdot )$ is a mapping function, and $f ( \mathbf { x } )$ denotes the corresponding output. Here we employ $\dot { \mathbf { I } } \mathbf { I } _ { 1 } \ \in \ \mathbb { R } ^ { N _ { \mathrm { R F } } \times N _ { \mathrm { R F } } } ,$ $\mathbf { I I } _ { 2 } ~ \in ~ \mathbb { R } ^ { N _ { s } \times N _ { s } }$ , and $\mathbf { I I } _ { 3 } ~ \in ~ \mathbb { R } ^ { K \times \bar { K } }$ to represent permutations of the RF chain, data stream, and subcarrier indices, respectively. Since the digital beamformer $\mathbf { F } \in \mathbb { C } ^ { N _ { \mathrm { R F } } \times N _ { s } \times K }$ involves permutations over multiple dimensions, we introduce a permutation operator as $[ \pi ( 1 ) , . . . , \pi ( N ) ] ~ = ~ [ 1 , . . . , N ] \mathbf { I } \mathbf { I }$ Then the permuted matrix multiplication is ${ \bf H } _ { k } ^ { \prime } e ^ { j \Phi _ { i } ^ { \prime } } { \bf F } _ { i , j , k } ^ { \prime } =$ ${ \bf H } _ { \pi _ { 3 } ( k ) } e ^ { j \Phi _ { \pi _ { 1 } ( i ) } } { \bf F } _ { \pi _ { 1 } ( i ) , \pi _ { 2 } ( j ) , \pi _ { 3 } ( k ) }$ , where $k ~ = ~ 1 , . . . , K , ~ i ~ =$ $1 , . . . , N _ { \mathrm { R F } }$ and $j = 1 , . . . , N _ { s }$ . Viewing the optimization problem (10) as a solution mapping: $( \Phi ^ { \prime } , \mathbf { F } ^ { \prime } ) = \operatorname { a r g m a x } { \mathcal { L } } ( \mathbf { H } ^ { \prime } )$ , we have $\left( \Phi _ { \pi _ { 1 } ( i ) } , { \bf F } _ { \pi _ { 1 } ( i ) , \pi _ { 2 } ( j ) , \pi _ { 3 } ( k ) } \right) \ =$ argmax $\mathcal { L } ( \mathbf { H } _ { \pi _ { 3 } ( k ) } )$ . And since $\Phi ^ { \prime }$ and $\mathbf { F ^ { \prime } }$ still satisfy the power constraint in (10), the optimization problem is permutation equivariant.

## APPENDIX B: PROOF OF PROPOSITION 2

Let $\pi$ be any permutation over the subcarrier index $k ,$ and define the permutation operator as $\mathbf { b } _ { \pi ( k ) } = \pi ( \mathbf { b } _ { k } )$ . By applying the same permutation to both sides of Eq. (11), we obtain $\begin{array} { r l r } { \mathbf { b } _ { \pi ( k ) } ^ { ( l ) } } & { = } & { \pi \left( f _ { \mathbf { b } } ( \mathbf { a } _ { k } ^ { ( l - 1 ) } , \phi ( \cdot ) \right) } \end{array}$ . Since $f _ { \mathbf { b } } ( \cdot )$ is shared across all subcarriers $k ,$ it is independent of the subcarrier index. Furthermore, as all aggregation functions $\phi ( \cdot )$ used in our design are permutation invariant, we have $\pi \left( f _ { \mathbf { b } } \left( \mathbf { a } _ { k } ^ { ( l - 1 ) } , \phi ( \cdot ) \right) \right) { \stackrel { - } { = } } f _ { \mathbf { b } } \left( \mathbf { \bar { a } } _ { \pi ( k ) } ^ { ( l - 1 ) } , \phi ( \cdot ) \right)$ , which implies that $\mathbf { b } _ { \pi ( k ) } ^ { ( l ) } ~ = ~ f _ { \mathbf { b } } \left( \mathbf { a } _ { \pi ( k ) } ^ { ( l - 1 ) } , \phi ( \cdot ) \right)$ ∀ ${ \mathbf { a } } _ { k } , { \mathbf { b } } _ { k } \in \mathcal { A } .$ ∀k. At the final beamformer reconstruction step in Fig. 4, the learned representations $\mathbf { b } _ { \pi ( k ) } ^ { ( L ) }$ are converted into beamforming matrices $\Phi$ and ${ \bf F } _ { \pi ( k ) }$ accordingly, which naturally inherit permutation equivariance established by GNN updating layers.

## APPENDIX C: PROOF OF PROPOSITION 3

The permutation equivariance with respect to subcarrier order has been established in Appendix B. In the multi-user extension, the graph is constructed by grouping subcarrier nodes into user-specific subsets without modifying the GNN updating rules. A permutation over user indices is therefore equivalent to a block-wise permutation over the corresponding subcarrier groups, which forms a subset of all possible subcarrier permutations. Hence, the permutation equivariance property is preserved, and the extended graph remains permutation equivariant with respect to both subcarrier and user indices.

## REFERENCES

[1] M. Cai et al., “Effect of Wideband Beam Squint on Codebook Design in Phased-Array Wireless Systems,” in Proc. IEEE GLOBECOM, Washington, DC, USA, 2016, pp. 1-6.

[2] H. Elayan, O. Amin, B. Shihada, R. M. Shubair et al., “Terahertz Band: The Last Piece of RF Spectrum Puzzle for Communication Systems,” in IEEE Open J. Commun. Soc., vol. 1, pp. 1-32, 2020.

[3] F. Gao, B. Wang, C. Xing, J. An and G. Y. Li, “Wideband Beamforming for Hybrid Massive MIMO Terahertz Communications,” in IEEE J. Sel. Areas Commun., vol. 39, no. 6, pp. 1725-1740, June 2021.

[4] Y. Chen, Y. Xiong, D. Chen et al., “Hybrid Precoding for WideBand Millimeter Wave MIMO Systems in the Face of Beam Squint,” in IEEE Trans. Wireless Commun., vol. 20, no. 3, pp. 1847-1860, March 2021.

[5] X. Yu, J. -C. Shen, J. Zhang et al., “Alternating Minimization Algorithms for Hybrid Precoding in Millimeter Wave MIMO Systems,” in IEEE J. Sel. Topics Signal Process., vol. 10, no. 3, pp. 485-500, April 2016.

[6] F. Sohrabi and W. Yu, “Hybrid Analog and Digital Beamforming for mmWave OFDM Large-Scale Antenna Arrays,” in IEEE J. Sel. Areas Commun., vol. 35, no. 7, pp. 1432-1443, July 2017

[7] H. Huang, Y. Song, J. Yang, G. Gui and F. Adachi, “Deep-Learning-Based Millimeter-Wave Massive MIMO for Hybrid Precoding,” in IEEE Trans. Veh. Technol., vol. 68, no. 3, pp. 3027-3032, March 2019.

[8] A. M. Elbir, “CNN-Based Precoder and Combiner Design in mmWave MIMO Systems,” in IEEE Commun. Lett., vol. 23, no. 7, pp. 1240-1243, July 2019.

[9] R. U. Murshed, Z. B. Ashraf, A. H. Hridhon, K. Munasinghe, A. Jamalipour and M. F. Hossain, “A CNN-LSTM-Based Fusion Separation Deep Neural Network for 6G Ultra-Massive MIMO Hybrid Beamform ing,” in IEEE Access, vol. 11, pp. 38614–38630, 2023.

[10] Y. Shen, J. Zhang, S. H. Song, et al., “Graph neural networks for wireless communications: From theory to practice,” in IEEE Trans. Wireless Commun., vol. 22, no. 5, pp. 3554–3569, May 2022.

[11] Y. Li et al., “Homogeneous and heterogeneous graph learning for hybrid beamforming in mmWave systems,” IEEE Trans. Wireless Commun., vol. 24, no. 10, pp. 8086–8100, Oct. 2025.

[12] Z. Huang et al., “Sub-6GHz assisted mmWave hybrid beamforming with heterogeneous graph neural network,” in IEEE Trans. Commun., 2024.

[13] S. Wan, Z. Wang et al., “Scalable Hybrid Beamforming for Multi-User MISO Systems: A Graph Neural Network Approach,” in IEEE Trans. Wireless Commun., vol. 23, no. 10, pp. 13694–13706, Oct. 2024.

[14] S. Liu, J. Guo and C. Yang, “Multidimensional Graph Neural Networks for Wireless Communications,” in IEEE Trans. Wireless Commun., vol. 23, no. 4, pp. 3057-3073, April 2024.

[15] R. Wang, C. Yang, S. Han, J. Wu, S. Han et al., “Learning End-to-End Hybrid Precoding for Multi-User mmWave Mobile System With GNNs,” in IEEE Trans. Mach. Learn. Commun. Netw., vol. 2, pp. 978–993, 2024.

[16] J. Yang, W. Zhu, S. Sun, X. Li, X. Lin, and M. Tao, “Deep Learning for Joint Design of Pilot, Channel Feedback, and Hybrid Beamforming in FDD Massive MIMO-OFDM Systems,” in IEEE Commun. Lett., vol. 28, no. 2, pp. 313–317, Feb. 2024.

[17] B. Zhao, J. Guo, and C. Yang, “Learning precoding policy: CNN or GNN?,” in Proc. IEEE WCNC, Austin, TX, USA, 2022, pp. 1027–1032.

[18] T. S. Rappaport, G. R. MacCartney, M. K. Samimi and S. Sun, “Wideband Millimeter-Wave Propagation Measurements and Channel Models for Future Wireless Communication System Design,” in IEEE Trans. Commun., vol. 63, no. 9, pp. 3029–3056, Sept. 2015.

[19] M. K. Samimi, T. S. Rappaport et al., “Probabilistic Omnidirectional Path Loss Models for Millimeter-Wave Outdoor Communications,” IEEE Wireless Commun. Lett., vol. 4, no. 4, pp. 357–360, Aug. 2015.

[20] H. Poddar, S. Ju, D. Shakya and T. S. Rappaport, “A tutorial on NYUSIM: Sub-terahertz and millimeter-wave channel simulator for 5G, 6G, and beyond,” IEEE Commun. Surveys Tuts., 2024.

[21] I. Loshchilov and F. Hutter, “SGDR: Stochastic Gradient Descent with Warm Restarts,” arXiv preprint arXiv:1608.03983, 2016.

[22] Q. Li, Z. Han, and X.-M. Wu, “Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning,” in Proc. AAAI Conf. Artif. Intell., vol. 32, no. 1, 2018.

[23] Q. Shi, M. Razaviyayn, Z.-Q. Luo, and C. He, “An iteratively weighted MMSE approach to distributed sum-utility maximization for a MIMO interfering broadcast channel,” IEEE Trans. Signal Process., vol. 59, no. 9, pp. 4331–4340, 2011.

[24] Q. Spencer, A. Swindlehurst, and M. Haardt, “Zero-forcing methods for downlink spatial multiplexing in multiuser MIMO channels,” IEEE Trans. Signal Process., vol. 52, no. 2, pp. 461–471, 2004.

[25] Y. Zhang, J. Gao, and Y. Liu, “MRT precoding in downlink multi-user MIMO systems,” EURASIP J. Wireless Commun. Netw., vol. 2016, no. 1, p. 241, 2016.