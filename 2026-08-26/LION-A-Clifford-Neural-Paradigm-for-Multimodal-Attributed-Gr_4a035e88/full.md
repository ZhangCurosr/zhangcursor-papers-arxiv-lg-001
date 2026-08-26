# LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning

Xunkai Li<sup>†,∗</sup>, Zekai Chen<sup>†,∗</sup>, Zhengyu Wu<sup>†</sup>, Henan Sun<sup>†</sup>, Daohan Su<sup>†</sup>, Guang Zeng<sup>#</sup>, Hongchao Qin<sup>†</sup>, Rong-Hua Li<sup>†</sup>, Guoren Wang<sup>†</sup>

<sup>†</sup> Beijing Institute of Technology, Beijing, China; <sup>∗</sup> Equal contribution. <sup>#</sup> Ant Group, Hangzhou, China

cs.xunkai.li@gmail.com, zackchen02@163.com, jeremywzy96@outlook.com, magneto0617@foxmail.com, dhsu@bit.edu.cn, zengguang 77@qq.com, qhc.neu@gmail.com, lironghuabit@126.com, wanggrbit@126.com

Abstract—Recently, the rapid advancement of multimodal domains has driven a data-centric paradigm shift in graph ML, transitioning from text-attributed to multimodal-attributed graphs. This advancement significantly enhances data representation and expands the scope of graph downstream tasks, such as modalityoriented tasks, thereby improving the practical utility of graph ML. Despite its promise, limitations exist in the current neural paradigms: (1) Neglect Context in Modality Alignment: Most existing methods adopt topology-constrained or modality-specific operators as tokenizers. These aligners inevitably neglect graph context and inhibit modality interaction, resulting in suboptimal alignment. (2) Lack of Adaptation in Modality Fusion: Most existing methods are simple adaptations for 2-modality graphs and fail to adequately exploit aligned tokens equipped with topology priors during fusion, leading to poor generalizability and performance degradation.

To address the above issues, we propose LION (cLIffOrd Neural paradigm) based on the Clifford algebra and decoupled graph neural paradigm (i.e., propagation-then-aggregation) to implement alignment-then-fusion in multimodal-attributed graphs. Specifically, we first construct a modality-aware geometric manifold grounded in Clifford algebra. This geometric-induced high order graph propagation efficiently achieves modality interaction, facilitating modality alignment. Then, based on the topologyaware Clifford components of aligned tokens, we propose adaptive holographic aggregation. This module integrates componentwise energy and propagation-scale information with learnable parameters to improve modality fusion. Extensive experiments on 9 text-image MAG datasets demonstrate that LION significantly outperforms SOTA baselines across 3 graph and 3 modality downstream tasks.

Index Terms—Multimodal-attributed graph learning, Clifford algebra, modality alignment, modality fusion.

## I. INTRODUCTION

With the proliferation of multimodal data (e.g., images and texts) and vision-language models [1]–[3], multimodalattributed graphs (MAGs) have recently emerged as a research frontier in the graph ML community. In MAG, nodes denote real-world entities described by multimodal data, which substantially expands the semantic dimensionality of node features and elevates the upper bound of representation learning capability [4]–[7]. This means that MAG not only offers immense potential for enhancing traditional graph-based tasks (e.g., node classification and link prediction) [8]–[10] but also broadens the task spectrum to encompass modality-oriented retrieval and generation [11]–[13], thereby enhancing practical utility. Consequently, developing effective MAG neural paradigms (i.e., MAGNNs) has become an urgent necessity [14]–[17]. Despite the remarkable progress made by recent studies, some inherent limitations still exist. Our in-depth analysis is as follows.

Limitation: Neglect MAG-Context in Modality Alignment. We argue that most existing methods inherit unnecessary entity isolation, at the node and modality levels, from conventional graph-agnostic modality alignment strategies. Specifically, (1) some methods employ topology-constrained aligners [11]– [13], [18]. While they aim to mitigate the receptive field limitations of the target nodes (i.e., single-entity) in MAG by incorporating broader node contexts, their scope remains restricted to 1-hop neighbors. Consequently, they still fail to capture the long-range dependencies, which have been proven to be highly beneficial in graphs [19]–[24]. (2) While other methods explicitly leverage graph topology to facilitate global alignment, they remain restricted to modality-specific scenarios and rely on the same topology to apply identical alignment across different modalities [25]–[28]. This coarsegrained alignment overlooks beneficial intra- and inter-modal interactions, thereby negatively impacting modality fusion.

Key Insight. It is imperative to leverage modality-aware, high-quality topology to dissolve entity semantic boundaries in MAGs, thereby fostering beneficial, profound modality interactions and understanding for better alignment.

Solution and Evaluation. To this end, we propose Clifford Geometric Propagation (CGP) to achieve topology-aware modality interaction for better alignment. Specifically, we first construct a MAG-specific geometric manifold where structural relationships are modeled as spatial rotations, and distinct modalities are represented as orthogonal geometric base vectors rather than modality-specific scale vectors. We then introduce a topology-aware geometric potential to characterize local semantic compatibility and regulate the contribution of neighboring signals during geometric transport. Together, the geometric potential and spatial rotor induce curvature-adaptive high-order graph propagation, enabling intra- and inter-modality interactions to facilitate modality alignment. To evaluate CGP, we illustrate the performance gains in Fig. 1(a) by employing CGP as a plug-and-play module to replace existing modality aligners. Despite several deeply coupled architectures that prevent broader plug-and-play evaluation, the consistent gains across these four representative cases support the effectiveness of CGP.

![](images/39a176070fa4269e2783fc447bd87d15162e3fdf536b10a65a4a0dc5269454e1.jpg)

![](images/8ab17424877af45f0c9c07447c97d667d4b5f5564d2bc60a9863eb042407495e.jpg)  
Fig. 1. Empirical evaluations of CGP (a) and AHA (b), where the x-axis for AHA (b) is training time in seconds.

Limitation: Lack of MAG-Adaptation in Modality Fusion. Given that enriched context, MAG significantly increases both the quantity and information density of aligned tokens, thereby presenting unique challenges. Specifically, (1) some methods directly employ Q-Former [29] to handle input scenarios involving topology-driven tokens (i.e., 1-hop neighbor tokens) by adaptive queries [11]–[13], [18]. Despite their simplicity and intuitiveness, these methods fail to explicitly leverage semantic aware topology priors within the aligned tokens. (2) Although other methods attempt to enhance fusion by partitioning tokens into multiple topology- and modality-specific views and achieving fine-grained integration, such approaches offer only limited exploration of suboptimal aligned tokens and fail to deeply capture their intricate dependencies. [25]–[28], [30].

Key Insight. It is imperative to reveal the intricate dependency among aligned tokens and design a comprehensive fusion mechanism to unleash their representation potential.

Solution and Evaluation. After obtaining the propagated features in the geometric manifold (i.e., aligned tokens) via the CGP, we clarify that they naturally possess geometric grade properties, which encode the topology and modality insights. Based on this, we propose Adaptive Holographic Aggregation (AHA), which employs learnable parameters to explicitly model the energy and scale inherent in these geometric grades. This strategy functions as a dynamic filter, capturing the most relevant topology and modality insights for task-adaptive fusion representation. To substantiate our claims, we present comprehensive results in Fig. 1(b), where LION demonstrates superior convergence and performance compared to the SOTA baselines. Furthermore, the complete LION (i.e., LION w/ AHA) significantly outperforms the LION w/o AHA variant, thereby validating the advantages of modality fusion in the holographic perspective.

Contributions. (1) New Perspective. We reveal the limitations of modality alignment and fusion in MAGs and introduce a mathematical framework grounded in Clifford algebra to address them. (2) Innovative Method. We propose LION (cLIffOrd Neural paradigm) based on this mathematical framework, which achieves alignment-then-fusion by propagation(CGP)-then-aggregation(AHA). Specifically, CGP first constructs a MAG-specific geometric manifold to enable curvature-adaptive high-order graph propagation, achieving efficient modality interaction for better alignment. Subsequently,

AHA captures the energy of topology-aware Clifford compo nents and their scale-dependent representations via learnable weights to facilitate modality fusion. (3) SOTA Performance. Evaluations from 6 domains demonstrate that LION achieves average improvements of 5.24% and 7.68% over SOTA baselines in graph and modality tasks.

## II. PRELIMINARIES

## A. Notations and Problem Formulation

MAG is defined as $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \{ \mathbf { X } ^ { ( m ) } \} _ { m \in \mathcal { M } } )$ where V is the set of nodes, E is the set of edges, and M is the set of available modalities (e.g., texts and images). For each node $v _ { i }$ and modality $m ,$ modality-specific attribute vector $x _ { i } ^ { ( m ) } \in \mathbb { R } ^ { d _ { m } }$ constitutes complete $\dot { \mathbf { X } } \in \mathbb { R } ^ { N \times d }$ . The modalityshared topology is described by the same adjacency matrix A, and D is the corresponding diagonal degree matrix. We further define the symmetric normalized graph Laplacian as $\tilde { \mathbf { L } } = \mathbf { I } - \tilde { \mathbf { D } } ^ { - 1 / 2 } \tilde { \mathbf { A } } \tilde { \mathbf { D } } ^ { - 1 / 2 }$ with self-loop $( \tilde { \mathbf { A } } = \mathbf { I } + \mathbf { A } )$ , which captures structural smoothness in graph signal processing [31], [32]. This formulation separates modality-specific attributes from the modality-shared topology, allowing us to analyze alignment and fusion as two coupled but distinct operations. It also clarifies the role of graph structure: topology is not merely an auxiliary input, but a shared geometric prior that regulates how multimodal signals interact across nodes. Accordingly, the learned representation should preserve both modality-specific semantics and topology-induced relational context.

Graph Tasks. (1) Node Classification: predict the specific label class of unlabeled nodes [33]–[35]; (2) Link Prediction: predict if $( u , v ) \ \in \ \mathcal { E }$ exists in the edge sets [36], [37]; (3) Node Clustering: partition all nodes into clusters using off-the-shelf clustering algorithms applied to the learned representations [38], [39]. These graph-centric tasks examine whether learned representations preserve structural proximity, class-level semantics, and community-level organization. They therefore provide a direct test of whether multimodal attributes can enhance classical graph learning beyond purely topologydriven modeling.

Multimodal Tasks. (1) Modality Retrieval: given a query in a specific modality (e.g., text), retrieve the corresponding cross-modal representation; (2) Text Generation: generate text responses conditioned on the target node, instructions, and graph contexts; (3) Image Generation: generate visual responses using diffusion models conditioned on the target node’s textual descriptions and the graph contexts. Compared with graphcentric tasks, these modality-oriented tasks require the model to preserve fine-grained cross-modal semantics while respecting graph-induced dependencies. This dual requirement makes MAG learning fundamentally different from applying a graph encoder after independent multimodal feature extraction.

## B. MAG Neural Paradigm

Conventional Methods. These approaches typically adapt previous multimodal architectures to MAGs by treating structural information as auxiliary tokens. The core idea is linearizing neighborhoods into sequences [11], [12] or projecting topology priors into soft tokens [13], [18]. While leveraging pre-trained knowledge, these methods rely on concatenation or rigid alignment, lacking a unified integration where topology and modality are unified in a continuous space. As a result, the graph often functions as an external context provider rather than a principle that governs modality interaction. Such a design is effective for injecting local context, but it may underuse topology when long-range or cross-modality dependencies are required.

Graph-enhanced Methods. These approaches explicitly design specialized model architectures for MAGs-such as refined message-passing [25], [26], [40], spectral graph filtering [27], [30]–[32], and structure-conditioned modules [28], [41]-to capture topology and modality insights. However, these approaches often treat multimodal attributes and graph topology as separate semantic spaces, relying on well-designed but suboptimal alignment and fusion components. This motivates a paradigm that represents modalities and topology within a shared geometric space, so that alignment and fusion can be derived from the same underlying operators.

## III. METHODOLOGY

## A. Clifford Algebra and Geometric Manifold

In this paper, we construct a mathematical framework specifically tailored for MAGs based on Clifford algebra [42]– [44], which extends the modality-specific graph operations to modality-aware high-dimensional geometric manifold spaces. Let $\mathcal { C } l _ { n }$ denote the Clifford algebra defined over an ndimensional space. By equipping each node with a local tangent space isomorphic to $\mathcal { C } l _ { n }$ and modeling edges as geodesic connections between them, we conceptualize the MAG as a discrete geometric manifold. The fundamental graph operation is the geometric product, which unifies spatial rotations (topology) with orthogonal geometric base vectors (modality).

Specifically, for any two connected nodes $u , v$ with attribute vectors $x _ { u } , x _ { v } ,$ their topology-aware modality interaction (i.e., transform modality-aware geometric base vectors via spatial rotations in the geometric manifolds) is formalized as $x _ { u } x _ { v } =$ $x _ { u } \cdot x _ { v } + x _ { u } \wedge x _ { v }$ . This geometric product explicitly decomposes the multimodal interaction: (1) The symmetric inner product $( x _ { u } \cdot x _ { v } )$ induces a modality-specific projection, facilitating intra-modality interaction. (2) The antisymmetric outer product $( x _ { u } \land x _ { v } )$ instantiates a bi-vector plane that encodes the topology curvature induced by cross-modality discrepancies, facilitating inter-modality interaction. This geometric manifold, spanned by Clifford algebra, provides a mathematical framework for topology-aware intra- and inter-modality interaction. Based on this, it enables modality alignment through topology-driven, curvature-induced spatial rotations (i.e., graph propagation) and facilitates modality fusion by capturing the geometric grade properties of aligned tokens (i.e., message aggregation).

## B. Clifford Geometric Propagation

Motivation. In order to achieve efficient modality alignment via the CGP (i.e., modality-aware, curvature-adaptive high-order graph propagation within the geometric manifold), it is essential to capture topology insights from the curved manifold to facilitate multimodal interactions. Consequently, we introduce parallel transport, which aligns the distinct semantic spaces of multiple modalities within connected nodes by rotating modality-aware orthogonal geometric basis vectors along the topology curvature. This high-dimensional transformation models modality interactions in a shared Clifford space and is mathematically formulated via the spatial rotor R and geometric potential Φ. Collectively, they modulate the principles of intra- and inter-modality interaction within a modality aware geometric manifold. The rotor provides norm-preserving local transport, while the geometric potential controls the contribution of transported signals, together enabling stable modality alignment for MAGs.

Modality-oriented Clifford Initialization. To begin with, we need to initialize X to support CGP. Since modality attributes may have unequal dimensions $d _ { k } ,$ a modality-specific encoder/projection $g _ { k } : \mathbb { R } ^ { d _ { k } }  \mathbb { R } ^ { d }$ first maps them into a shared feature dimension, $\mathrm { i . e . , ~ } \bar { x } _ { u } ^ { ( k ) } = g _ { k } ( x _ { u } ^ { ( k ) } ) ^ { \bullet }$ ; g<sub>k</sub> is the identity when the dimensions already match. We then use stabilized per-modality normalization $\hat { x } _ { u } ^ { ( k ) } = \bar { x } _ { u } ^ { ( k ) } / \operatorname* { m a x } ( \| \bar { x } _ { u } ^ { ( k ) } \| _ { 2 } , \epsilon _ { x } )$ and assign modality k to the orthogonal Grade-1 basis $e _ { k } \in \mathcal { C } l _ { K } ;$

$$
\mathbf { H } _ { u } ^ { ( 0 ) } \in \mathbb { R } ^ { d \times 2 ^ { K } } : = \frac { 1 } { \sqrt { K } } \sum _ { k = 1 } ^ { K } \hat { x } _ { u } ^ { ( k ) } e _ { k } , \| \mathbf { H } _ { u } ^ { ( 0 ) } \| _ { \mathcal { C } l } \leq 1 .\tag{1}
$$

This definition also makes the interaction semantics explicit. For feature coordinate $j ,$ , let $\begin{array} { r } { h _ { u , j } \ = \ K ^ { - 1 / 2 } \sum _ { k } \hat { x } _ { u , j } ^ { ( k ) } e _ { k } } \end{array}$ . Its geometric product satisfies

$$
\begin{array} { r l } & { \langle h _ { u , j } h _ { v , j } \rangle _ { 0 } = \displaystyle \frac { 1 } { K } \sum _ { k } \hat { x } _ { u , j } ^ { ( k ) } \hat { x } _ { v , j } ^ { ( k ) } , } \\ & { \langle h _ { u , j } h _ { v , j } \rangle _ { 2 } = \displaystyle \frac { 1 } { K } \sum _ { p < q } ( \hat { x } _ { u , j } ^ { ( p ) } \hat { x } _ { v , j } ^ { ( q ) } - \hat { x } _ { u , j } ^ { ( q ) } \hat { x } _ { v , j } ^ { ( p ) } ) e _ { p } e _ { q } . } \end{array}\tag{2}
$$

Thus, Grade-0 aggregates same-modality compatibility, whereas each Grade-2 coefficient represents an oriented pairwise crossmodality discrepancy; an ordered modality pair is not identified one-to-one with a Clifford basis component. The formulation is algebraically defined for K modalities, while all experiments in this paper use the prevalent $K \ : = \ : 2$ text-image setting; empirical generality beyond two modalities is not claimed.

Topology-oriented Geometric Potential. After Clifford initialization, we capture the semantic curvature between connected nodes through feature-wise geometric interactions. For each feature coordinate $j \in \{ 1 , \ldots , d \}$ and edge $( u , v ) \in \mathcal { E }$ define

$$
\begin{array} { r } { S _ { u v , j } = \langle h _ { u , j } h _ { v , j } \rangle _ { 0 } , } \\ { B _ { u v , j } = \langle h _ { u , j } h _ { v , j } \rangle _ { 2 } . } \end{array}\tag{3}
$$

Here $\mathcal { S } _ { u v , j }$ measures same-modality compatibility, whereas $B _ { u v , j }$ characterizes the oriented pairwise cross-modality discrepancy defined in Eq. (2).

Based on these quantities, the modality-adaptive geometric potential for coordinate $j$ is

$$
\Phi _ { u v , j } = \exp \left( - \frac { \| \mathcal { B } _ { u v , j } \| _ { \mathcal { C l } } ^ { 2 } } { | \mathcal { S } _ { u v , j } | + \epsilon } \right) , ( u , v ) \in \mathcal { E } .\tag{4}
$$

The scalar term measures semantic compatibility, while the bi-vector norm measures the magnitude of the corresponding cross-modality interaction plane. The exponential decay kernel therefore converts their relative discrepancy into a bounded feature-wise propagation weight. This construction provides the topology-aware interaction statistics used by the subsequent parallel transport.

Training-free Geometric Propagation. Based on the above feature-wise geometric statistics, we extend conventional graph propagation to Clifford-valued parallel transport. Specifically, $\mathcal { S } _ { u v , j }$ measures local semantic compatibility, while $B _ { u v , j }$ determines the oriented interaction plane for geometric transport. For each edge $( u , v )$ and feature coordinate $j ,$ , let

$$
\begin{array} { r l } & { \widehat { \mathcal { B } } _ { u v , j } = \frac { \mathcal { B } _ { u v , j } } { \sqrt { \| \mathcal { B } _ { u v , j } \| _ { \mathcal { C } l } ^ { 2 } + \epsilon _ { r } ^ { 2 } } } , } \\ & { \theta _ { u v , j } = \mathrm { a t a n 2 } \left( \| \mathcal { B } _ { u v , j } \| _ { \mathcal { C } l } , | \mathcal { S } _ { u v , j } | + \epsilon _ { r } \right) , } \\ & { \mathcal { R } _ { u v , j } = \mathrm { e x p } \left( - \frac { \theta _ { u v , j } \widehat { B } _ { u v , j } } { 2 } \right) . } \end{array}\tag{5}
$$

Here, atan2(·, ·) denotes the two-argument arctangent used to determine a stable rotation magnitude. The stabilized rotor continuously approaches the identity as $\| \boldsymbol { B _ { u v , j } } \| c _ { l } \to 0$

The rotor determines how a neighboring representation is geometrically transported, while the geometric potential controls its contribution. We further introduce a self-loop potential $\bar { \Phi } _ { u u , j } = 1$ and set $\bar { \Phi } _ { u v , j } = \Phi _ { u v , j }$ for $( u , v ) \in \mathcal { E }$ Feature-wise row normalization gives

$$
\widetilde { \Phi } _ { u v , j } = \frac { \bar { \Phi } _ { u v , j } } { \sum _ { w \in \mathcal { N } ( u ) \cup \{ u \} } \bar { \Phi } _ { u w , j } } .\tag{6}
$$

Define the feature-wise transport

$$
\begin{array} { r } { \mathcal { T } _ { u v , j } ( h ) = \mathcal { R } _ { u v , j } h \mathcal { R } _ { u v , j } ^ { - 1 } , } \end{array}\tag{7}
$$

with $\mathcal { T } _ { u u , j }$ being the identity. Accordingly, the CGP update is

$$
h _ { u , j } ^ { ( l ) } = \sum _ { v \in \mathcal { N } ( u ) \cup \{ u \} } \widetilde { \Phi } _ { u v , j } \mathcal { T } _ { u v , j } \left( h _ { v , j } ^ { ( l - 1 ) } \right) , \qquad j = 1 , \dotsc , d .\tag{8}
$$

Equivalently, Eq. (8) defines a block-diagonal connection operator over the direct-sum feature space

$$
\mathcal { H } = \bigoplus _ { j = 1 } ^ { d } \mathcal { C } l _ { K } ^ { ( 1 ) } .\tag{9}
$$

Thus, each feature coordinate is transported by its own curvature-dependent Clifford connection, while all coordinates share the same graph topology. The normalized connection average keeps the propagation bounded, and the rotor sandwich preserves the Grade-1 state space, providing topology-aware modality components for the subsequent AHA module.

Theoretical Analysis. We first establish the geometric stability bound of the Clifford construction (Theorem III.1; Supplementary Appendix A). We then analyze the normalized CGP operator in Eq. (8) under standard fixed-connection assumptions (Theorem III.2; Supplementary Appendix B).

Theorem III.1. (Stability Bound of Clifford Manifold). Let f obtain H and ${ \mathcal { A } } _ { \mathcal { G } } ~ = ~ \{ \Phi , { \mathcal { R } } \}$ . Assume the modality encoders/projections are bounded Lipschitz maps and $\epsilon _ { x } , \epsilon _ { r } > 0$ are fixed. For bounded perturbations of $( \mathbf { X } , \mathbf { A } )$ ,

$$
\begin{array} { r l } & { \| \mathbf { H } - \mathbf { H } ^ { \prime } \| _ { \mathcal { C } l } + \| \mathcal { A } _ { \mathcal { G } } - \mathcal { A } _ { \mathcal { G } } ^ { \prime } \| _ { \mathcal { C } l } } \\ & { \leq K _ { \operatorname* { m a p } } ( \| \mathbf { X } - \mathbf { X } ^ { \prime } \| _ { \mathcal { C } l } + \gamma \| \mathbf { A } - \mathbf { A } ^ { \prime } \| _ { \mathcal { C } l } ) . } \end{array}\tag{10}
$$

where $K _ { \mathrm { m a p } }$ and $\gamma$ depend on the encoder/projection bounds, graph degree, and stabilizers $\epsilon _ { x } , \epsilon _ { r }$

Proof Sketch. Stabilized normalization is Lipschitz for fixed $\epsilon _ { x } > 0 ;$ the orthogonal Grade-1 lift is linear and norm bounded. The geometric product is bilinear, its grade projections are non-expansive, and the stabilized atan2 rotor is continuous and Lipschitz on the resulting bounded domain, including the near-degenerate case. Combining these bounds with the edge perturbation term yields Eq. (10); details are in Supplementary Appendix A.

Theorem III.2. (Conditional Spectral Contraction of CGP). Assume an undirected graph, feature-wise symmetric potentials $\Phi _ { u v , j } = \Phi _ { v u , j } ,$ reciprocal isometric transports $\mathcal { T } _ { v u , j } = \mathcal { T } _ { u v , j } ^ { - 1 } ,$ and fixed edge operators during one cached propagation stage. Let $\mathcal { P } _ { \mathcal { G } }$ denote the block-diagonal normalized connection operator induced by Eq. (8) on $\begin{array} { r } { \mathcal { H } = \bigoplus _ { j = 1 } ^ { d } \mathcal { C } l _ { K } ^ { ( 1 ) } } \end{array}$ . Let Π be the orthogonal projection onto its unit-eigenspace, and let $\rho ~ < ~ 1$ denote the spectral radius of $\mathcal { P } _ { \mathcal { G } }$ restricted to the complementary invariant subspace. Then

$$
\left. \mathbf { H } ^ { ( l ) } - \Pi \mathbf { H } ^ { ( 0 ) } \right. _ { D , \mathcal { C l } } \leq \rho ^ { l } \left. \mathbf { H } ^ { ( 0 ) } - \Pi \mathbf { H } ^ { ( 0 ) } \right. _ { D , \mathcal { C l } } .\tag{11}
$$

Proof Sketch. For each feature coordinate, symmetry of the potential and reciprocity of the rotor transport induce a self-adjoint normalized connection averaging operator in the corresponding degree-weighted inner product. Their direct sum therefore defines a self-adjoint block-diagonal operator $\mathcal { P } _ { \mathcal { G } }$ on $\mathcal { H }$ . Decomposing $\mathbf { H } ^ { ( 0 ) }$ into the unit-eigenspace and its orthogonal complement yields Eq. (11). The statement is conditional on fixed reciprocal transports and on $\rho < 1$ on the complementary invariant subspace; it does not claim an unconditional contraction rate for arbitrary time-varying connections. Supplementary Appendix B provides the complete derivation.

## C. Adaptive Holographic Aggregation

Motivation. To unleash the representation potential of geometrically propagated features (i.e., aligned tokens), it is imperative to reveal their intricate dependencies. Unfortunately, prevalent message aggregation (e.g., simple concatenation or mean pooling) indiscriminately compresses modality-aware components and ignores receptive fields, leading to poor generalizability and performance degradation. We clarify that CGP maintains the propagated node states in the Grade-1 subspace because rotor sandwich transport preserves vector grade. The Grade-0 and Grade-2 quantities in Eq. (2) are therefore edgewise interaction statistics that parameterize the geometric potential and spatial rotor, rather than additional propagated state channels. Consequently, the propagated Grade-1 components encode modality-specific states that have already been modulated by topology-aware intra- and inter-modality interactions during CGP. Based on this observation, we propose

Adaptive Holographic Aggregation (AHA), which adaptively filters these propagated Clifford components according to their energy and subsequently reconciles their representations across propagation scales.

Energy-aware Component Filtering. Since $\mathbf { H } _ { u } ^ { ( 0 ) }$ lies in the Grade-1 subspace and rotor sandwich transport preserves grade, the normalized CGP update in Eq. (8) satisfies

$$
\mathbf { H } _ { u } ^ { ( l ) } = \sum _ { k = 1 } ^ { K } \mathbf { H } _ { u , k } ^ { ( l ) } e _ { k } \in \mathcal { C } l _ { K } ^ { ( 1 ) } , l = 0 , \ldots , L .\tag{12}
$$

Here $\mathbf { H } _ { u , k } ^ { ( l ) } \in \mathbb { R } ^ { d }$ denotes the propagated component associated with modality axis $e _ { k }$ . Although the propagated state remains Grade-1, each component has been transformed by the geometric potential and rotor constructed from the Grade-0 and Grade-2 interaction statistics in Eq. (2).

We quantify the information density of the active Clifford components by

$$
\mathbb { E } _ { u } ^ { ( l ) } = \left[ \| \mathbf { H } _ { u , 1 } ^ { ( l ) } \| _ { 2 } ^ { 2 } , \ldots , \| \mathbf { H } _ { u , K } ^ { ( l ) } \| _ { 2 } ^ { 2 } \right] ^ { \top } \in \mathbb { R } ^ { K } .\tag{13}
$$

A learnable gate then adaptively modulates these propagated modality components:

$$
\boldsymbol { \alpha } _ { u } ^ { ( l ) } = \sigma \Big ( \mathbf { W } _ { \mathcal { G } } \mathrm { N o r m } ( \mathbb { E } _ { u } ^ { ( l ) } ) + \mathbf { b } _ { \mathcal { G } } \Big ) , \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } = \sum _ { k = 1 } ^ { K } \boldsymbol { \alpha } _ { u , k } ^ { ( l ) } \mathbf { H } _ { u , k } ^ { ( l ) } \boldsymbol { e _ { k } } ,\tag{14}
$$

where ${ \pmb { \alpha } } _ { u } ^ { ( l ) } \in ( 0 , 1 ) ^ { K }$ . The soft gate attenuates less informative propagated components rather than removing them deterministically. Importantly, AHA does not identify the K propagated modality axes with the $K ^ { 2 }$ ordered modality interactions. Instead, cross-modality interactions affect these propagated states through the scalar and bi-vector statistics used to construct CGP, while AHA determines how the resulting topology-aware modality components contribute to downstream fusion.

Scale-aware Resonance Fusion. After obtaining the energyfiltered topology-aware modality components, their validity varies across topology scales corresponding to the propagation depth L within each channel. To reconcile these receptive fields and facilitate fusion, we first generate $\mathbf { H } _ { u } ^ { \mathrm { c t x } }$ by learnable $\mathbf { W } _ { \tau }$ to capture the consensus profile across all scales:

$$
\mathbf { H } _ { u } ^ { \mathrm { c t x } } \in \mathbb { R } ^ { d \times 2 ^ { K } } : = \mathrm { N o r m } \left( \sum _ { l = 0 } ^ { L } \left[ \mathbf { W } _ { \tau } ^ { ( l ) } \tilde { \mathbf { H } } _ { u } ^ { ( l ) } + \mathbf { b } _ { \tau } ^ { ( l ) } \right] \right) .\tag{15}
$$

Subsequently, we employ a resonance attention mechanism to compute the scale validity score $\beta _ { u , l } ,$ , where a learnable attention score $\mathbf { a } _ { S } ^ { \top }$ projects the interaction between the current representation and consensus profile into a scalar space. Based on this, AHA aggregates the grade-filtered and scale-weighted tokens, and the $d _ { f }$ -dimension fusion representation $\mathbf { Z } _ { u }$ is obtained by projecting it back into Euclidean space via a Clifford linear layer parameterized by $\mathbf { W _ { \mathrm { o u t } } } \mathbf { \dot { . } }$

$$
\mathbf { Z } _ { u } \in \mathbb { R } ^ { d _ { f } } : = \mathbf { W } _ { \mathrm { o u t } } \left( \sum _ { l = 0 } ^ { L } \beta _ { u , l } \cdot \tilde { \mathbf { H } } _ { u } ^ { ( l ) } \right) + \mathbf { b } _ { \mathrm { o u t } } ,
$$

$$
\beta _ { u , l } = \frac { \exp ( \mathbf { a } _ { S } ^ { \top } \cdot \operatorname { t a n h } ( \mathbf { W } _ { S } [ \mathbf { H } _ { u } ^ { \mathrm { c t x } } \mathbf {  { \lVert } } \tilde { \mathbf { H } } _ { u } ^ { ( l ) } ] ) ) } { \sum _ { k = 0 } ^ { L } \exp ( \mathbf { a } _ { S } ^ { \top } \cdot \operatorname { t a n h } ( \mathbf { W } _ { S } [ \mathbf { H } _ { u } ^ { \mathrm { c t x } } \mathbf {  { \lVert } } \tilde { \mathbf { H } } _ { u } ^ { ( k ) } ] ) ) } .\tag{16}
$$

Theoretical Analysis. We characterize the sensitivity introduced by AHA without assuming that the consensus profile is an unknown downstream optimum. Let $f _ { \mathrm { c l l } }$ denote the affine Clifford projection in Eq. (16), whose linear part has operator norm at most $\omega ,$ and let $\begin{array} { r } { { \bf Z } _ { u } ^ { \mathrm { r a w } } = f _ { \mathrm { c l l } } ( \sum _ { l } \beta _ { u , l } { \bf \bar { H } } _ { u } ^ { ( l ) } ) } \end{array}$

Theorem III.3. (AHA Stability and Consensus Bound). Since $\beta _ { u , l } \geq 0$ and $\begin{array} { r } { \sum _ { l } \beta _ { u , l } = 1 } \end{array}$

$$
\begin{array} { r l r } { \displaystyle \| \mathbf { Z } _ { u } - \mathbf { Z } _ { u } ^ { \mathrm { r a w } } \| _ { 2 } \leq \omega \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| ( \mathbf { 1 } - \boldsymbol { \alpha } _ { u } ^ { ( l ) } ) \odot \mathbf { H } _ { u } ^ { ( l ) } \| _ { \mathcal { C } ^ { l } } , } & { } & \\ { \displaystyle \| \mathbf { Z } _ { u } - f _ { \mathrm { c l l } } ( \mathbf { H } _ { u } ^ { \mathrm { c t x } } ) \| _ { 2 } \leq \omega \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| \tilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \| _ { \mathcal { C } ^ { l } } . } & { } & \\ { \displaystyle \| \mathbf { Z } _ { u } - f _ { \mathrm { c l l } } ( \mathbf { H } _ { u } ^ { \mathrm { c t x } } ) \| _ { 2 } \leq \omega \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| \tilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \| _ { \mathcal { C } ^ { l } } . } & { } & \\ { \displaystyle \| \mathbf { Z } _ { u } - f _ { \mathrm { c l l } } ( \mathbf { H } _ { u } ^ { \mathrm { c t x } } ) \| _ { 2 } } & { \leq \omega \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| \tilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \| _ { \mathcal { C } ^ { l } } . } & { } & \\ { \displaystyle \| \mathbf { Z } _ { u } - f _ { \mathrm { c l l } } ( \mathbf { H } _ { u } ^ { \mathrm { c t x } } ) \| _ { 2 } } &  \leq \omega \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| \tilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \| _ \end{array}\tag{17}
$$

These inequalities separately bound the representation change introduced by grade gating and the deviation from the learned multiscale consensus. They do not assume that suppressed components are noise or that $\mathbf { H } _ { u } ^ { \mathrm { c t x } }$ is the downstream-optimal representation. Both bounds follow from the linear part of the affine projection, the convex scale weights, and the triangle inequality; Supplementary Appendix C provides the complete proof.

Proof Sketch. The first inequality compares the gated and ungated convex scale aggregates and applies the operator-norm bound of $f _ { \mathrm { c l l } }$ . The second inserts the learned consensus profile inside the same convex aggregate and applies the triangle inequality. No assumption is made about whether an attenuated component is signal or noise. The complete proof is provided in Supplementary Appendix C.

## D. Algorithm and Complexity Analysis

Algorithm Details. For a more comprehensive presentation, we provide the complete LION in Algorithm 1. Specifically, our framework transforms raw multimodal inputs into a unified representation space via two decoupled phases: Clifford Geometric Propagation (CGP) for modality alignment and Adaptive Holographic Aggregation (AHA) for modality fusion. CGP first lifts raw modalities into the Clifford manifold, constructs the geometric potential and spatial rotor from topology-aware interactions, and performs training-free highorder propagation to generate aligned multiscale tokens. AHA then filters topology-aware modality components by energy and fuses propagation depths by resonance with a consensus profile. This design makes the expensive topology-aware geometric computation independent of iterative parameter learning, while retaining task-adaptive fusion in the trainable stage. More specifically, the Clifford lifting step assigns each modality to an orthogonal basis vector, which preserves modality identity while enabling explicit geometric products between modalities. The scalar and bi-vector components of the edgewise geometric products instantiate the intra- and inter-modality interaction statistics that parameterize CGP. After CGP, AHA operates on the resulting topology-aware Clifford token components and propagation depths to decide which components and receptive fields should dominate the final representation. In this sense, Algorithm 1 separates representation construction from taskspecific optimization while keeping both stages connected through the same Clifford token space. The propagated tokens can be shared by node-, edge-, and modality-level heads, which makes the framework naturally adaptable to heterogeneous MAG tasks. Detailed algorithmic interpretation and complexity derivations are provided in Supplementary Appendix E.

TABLE I  
THE STATISTICAL INFORMATION OF THE EXPERIMENTAL DATASETS. THE “TASKS” COLUMN SUMMARIZES APPLICABLE TASK TYPES; EVALUATED DATASET-TASK PAIRS ARE SPECIFIED BY THE RESULT-TABLE HEADERS.
<table><tr><td>Datasets</td><td># Modalities</td><td># Nodes</td><td> $\# \operatorname { E d g e s }$ </td><td># Classes</td><td>Tasks</td><td>Description</td></tr><tr><td>RedditS</td><td>Text, Image</td><td>15,894</td><td>566,160</td><td>20</td><td>Graph (Node, Link)</td><td>Social Network</td></tr><tr><td>Movies</td><td>Text, Image</td><td>16,672</td><td>218,390</td><td>20</td><td>Graph (Node, Link)</td><td>Movie Network</td></tr><tr><td>Grocery</td><td>Text, Image</td><td>17,074</td><td>171,340</td><td>20</td><td>Graph (Node, Link), Modality</td><td>Recommendation</td></tr><tr><td>SemArt</td><td>Text, Image</td><td>21,382</td><td>1,216,432</td><td></td><td>Graph (Link), Modality</td><td>Art Network</td></tr><tr><td>Flickr30k</td><td>Text, Image</td><td>31,783</td><td>181,151</td><td></td><td>Graph (Link), Modality</td><td>Image Network</td></tr><tr><td>Sports</td><td>Text, Image</td><td>50,250</td><td>356,202</td><td></td><td>Graph (Link), Modality</td><td>Recommendation</td></tr><tr><td>Ele-fashion</td><td>Text, Image</td><td>97,766</td><td>199,602</td><td>12</td><td>Graph (Node, Link), Modality</td><td>Recommendation</td></tr><tr><td>Cloth</td><td>Text, Image</td><td>125,839</td><td>951,271</td><td></td><td>Graph (Link), Modality</td><td>Recommendation</td></tr><tr><td>Goodreads</td><td>Text, Image</td><td>685,294</td><td>7,235,048</td><td>11</td><td>Graph (Node, Link)</td><td>Book Network</td></tr></table>

TABLE II  
PERFORMANCE COMPARISON ON GRAPH DOWNSTREAM TASKS. THE BEST/SECOND RESULT IS BOLD/UNDERLINE.
<table><tr><td>Tasks</td><td colspan="4">Node Classification</td><td colspan="4">Link Prediction</td><td colspan="4">Node Clustering</td></tr><tr><td> $\sim \mathbf { D a t a s e t s }$ </td><td>Movies</td><td></td><td>Goodreads</td><td></td><td>Sports</td><td></td><td>Cloth</td><td></td><td>RedditS</td><td></td><td></td><td>Grocery</td></tr><tr><td> $\bf { M e t h o d s } \overbrace { \phantom { . 5 \ m u _ { 0 } ^ { ( 1 ) } } } ^ { \substack { \bf { M e t h o d s } } }$  GCN</td><td> $\operatorname { A c c }$ </td><td>F1-Score</td><td>Acc</td><td>F1-Score</td><td>MRR  $4 6 . 2 3 _ { \pm 0 . 2 8 }$ </td><td>Hits@3</td><td>MRR  $\overline { { 4 0 . 5 2 _ { \pm 0 . 3 7 } } }$ </td><td>Hits@3</td><td>NMI 62.18±0.62</td><td>ARI</td><td>NMI 38.13±0.43</td><td>ARI  $2 9 . 5 4 _ { \pm 0 . 3 8 }$ </td></tr><tr><td></td><td> $\overline { { 3 9 . 1 2 _ { \pm 1 . 3 3 } } }$ </td><td> $3 6 . 1 7 _ { \pm 1 . 4 1 }$ </td><td> $5 6 . 0 3 _ { \pm 1 . 5 2 }$ </td><td> $\overline { { 5 0 . 1 4 _ { \pm 1 . 2 1 } } }$ </td><td></td><td> $5 8 . 0 9 _ { \pm 0 . 7 3 }$ </td><td></td><td> $4 8 . 1 5 _ { \pm 0 . 4 4 }$ </td><td></td><td> $6 4 . 0 7 _ { \pm 0 . 5 1 }$ </td><td> $4 5 . 8 4 _ { \pm 0 . 4 9 }$ </td><td> $3 2 . 2 6 { \scriptstyle \pm 0 . 4 1 }$ </td></tr><tr><td>MMGCN</td><td> $4 4 . 8 7 _ { \pm 1 . 4 2 }$ </td><td> $3 9 . 9 2 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $5 8 . 5 4 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $5 2 . 3 6 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $4 8 . 4 4 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $6 0 . 5 2 \pm 0 . 5 8$ </td><td> $4 2 . 8 3 _ { \pm 0 . 4 1 }$ </td><td> $5 0 . 9 4 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $7 2 . 1 0 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $7 0 . 0 8 { \scriptstyle \pm 0 . 6 2 }$ </td><td></td><td></td></tr><tr><td>GAT</td><td> $4 0 . 2 5 _ { \pm 1 . 4 8 }$ </td><td> $3 6 . 4 3 _ { \pm 1 . 5 2 }$ </td><td> $5 7 . 1 2 _ { \pm 1 . 0 8 }$ </td><td> $5 1 . 0 8 _ { \pm 1 . 2 4 }$ </td><td> $4 7 . 1 9 _ { \pm 0 . 4 3 }$ </td><td> $5 9 . 1 4 _ { \pm 0 . 7 7 }$ </td><td> $4 1 . 2 6 _ { \pm 0 . 4 9 }$ </td><td> $4 9 . 3 4 _ { \pm 0 . 5 8 }$ </td><td> $6 5 . 4 5 _ { \pm 0 . 6 8 }$ </td><td> $6 6 . 2 8 \scriptstyle \pm 0 . 6 3$ </td><td> $3 9 . 4 2 _ { \pm 0 . 4 5 }$ </td><td> $2 8 . 7 5 _ { \pm 0 . 4 7 }$ </td></tr><tr><td>MGAT</td><td> $4 3 . 1 4 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $3 8 . 0 9 { \scriptstyle \pm 1 . 4 4 }$ </td><td> $5 9 . 8 8 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $5 3 . 7 5 { \scriptstyle \pm 1 . 3 8 }$ </td><td> $4 9 . 8 2 _ { \pm 0 . 3 6 }$ </td><td> $6 2 . 0 3 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $4 3 . 9 1 _ { \pm 0 . 4 5 }$ </td><td> $5 2 . 5 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $7 3 . 3 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 9 . 1 4 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $4 3 . 0 8 _ { \pm 0 . 4 3 }$ </td><td> $3 1 . 1 5 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td>MLaGA</td><td> $4 8 . 3 7 _ { \pm 1 . 0 2 }$ </td><td> $4 2 . 0 8 _ { \pm 1 . 6 4 }$ </td><td> $6 6 . 4 2 _ { \pm 1 . 2 5 }$ </td><td> $6 0 . 2 9 _ { \pm 1 . 1 3 }$ </td><td> $5 4 . 3 3 _ { \pm 0 . 6 2 }$ </td><td> $6 8 . 4 1 _ { \pm 0 . 8 8 }$ </td><td> $4 9 . 1 2 _ { \pm 0 . 7 3 }$ </td><td> $5 8 . 3 7 _ { \pm 0 . 8 0 }$ </td><td> $8 2 . 5 8 _ { \pm 1 . 1 5 }$ </td><td> $8 2 . 1 4 _ { \pm 1 . 1 2 }$ </td><td> $\overline { { 5 1 . 9 2 _ { \pm 0 . 8 3 } } }$ </td><td> $3 7 . 5 8 _ { \pm 0 . 8 9 }$ </td></tr><tr><td>GraphGPT-O</td><td> $4 9 . 1 2 { \scriptstyle \pm 1 . 9 0 }$ </td><td> $4 2 . 9 4 { \scriptstyle \pm 1 . 1 3 }$ </td><td>64.25±0.92</td><td> $5 8 . 7 6 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $5 5 . 8 2 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $6 9 . 1 7 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $5 1 . 4 3 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $6 0 . 1 8 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $7 9 . 3 3 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $8 0 . 4 1 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $5 0 . 6 4 { \scriptstyle \pm 0 . 6 9 }$ </td><td>38.83±0.62</td></tr><tr><td>Graph4MM</td><td> $4 9 . 7 6 _ { \pm 1 . 8 5 }$ </td><td> $4 3 . 2 2 _ { \pm 1 . 9 5 }$ </td><td> $6 7 . 1 8 _ { \pm 0 . 7 8 }$ </td><td> $6 1 . 3 5 _ { \pm 0 . 8 3 }$ </td><td> $5 3 . 9 4 _ { \pm 0 . 7 1 }$ </td><td> $6 7 . 5 8 _ { \pm 0 . 8 3 }$ </td><td> $5 0 . 8 4 _ { \pm 0 . 6 8 }$ </td><td> $5 9 . 4 3 _ { \pm 0 . 6 3 }$ </td><td> $8 4 . 1 4 _ { \pm 0 . 9 2 }$ </td><td> $8 3 . 2 5 _ { \pm 0 . 9 0 }$ </td><td> $5 1 . 2 0 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $3 8 . 9 2 _ { \pm 0 . 7 5 }$ </td></tr><tr><td>InstructG2I</td><td> $5 0 . 5 7 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $4 4 . 0 8 { \scriptstyle \pm 1 . 3 8 }$ </td><td> $6 5 . 8 4 _ { \pm 0 . 9 0 }$ </td><td> $5 9 . 9 3 _ { \pm 1 . 0 2 }$ </td><td> $5 6 . 7 5 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $7 0 . 1 2 \pm 0 . 9 3$ </td><td> $5 2 . 2 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $6 1 . 5 4 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $8 2 . 8 3 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $8 1 . 5 8 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $4 9 . 8 7 _ { \pm 0 . 9 2 }$ </td><td> $3 5 . 2 8 { \scriptstyle \pm 0 . 8 2 }$ </td></tr><tr><td>DMGC</td><td> $5 1 . 7 3 { \scriptstyle \pm 1 . 9 5 }$ </td><td> $4 6 . 8 4 _ { \pm 1 . 5 9 }$ </td><td> $6 3 . 4 2 _ { \pm 1 . 2 7 }$ </td><td> $6 1 . 1 8 { \scriptstyle \pm 1 . 6 3 }$ </td><td> $5 6 . 4 7 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $6 9 . 8 3 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $5 1 . 8 7 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $6 0 . 8 4 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $\underline { { 8 9 . 6 2 } } \pm 0 . 7 3$ </td><td> $\underline { { 8 9 . 1 4 } } \pm \mathbf { 0 . 8 0 }$ </td><td> $5 6 . 1 3 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $4 1 . 8 2 { \scriptstyle \pm 0 . 6 0 }$ </td></tr><tr><td>DGF</td><td> $5 2 . 9 4 _ { \pm 1 . 6 4 }$ </td><td> $4 5 . 1 3 _ { \pm 1 . 7 0 }$ </td><td> $6 6 . 8 3 _ { \pm 1 . 6 1 }$ </td><td> $6 2 . 5 4 _ { \pm 1 . 1 2 }$ </td><td> $5 5 . 9 3 _ { \pm 0 . 5 1 }$ </td><td> $6 9 . 1 7 _ { \pm 0 . 5 4 }$ </td><td> $5 1 . 2 4 _ { \pm 0 . 6 3 }$ </td><td> $6 0 . 1 3 _ { \pm 0 . 5 9 }$ </td><td> $8 8 . 8 7 _ { \pm 0 . 7 0 }$ </td><td> $8 8 . 4 2 _ { \pm 0 . 7 6 }$ </td><td> $5 5 . 8 8 _ { \pm 0 . 6 3 }$ </td><td> $\underline { { 4 2 . 2 5 } } \underline { { : 0 . 6 8 } }$ </td></tr><tr><td>MIG-GT</td><td> $5 3 . 1 7 { \scriptstyle \pm 2 . 0 2 }$ </td><td> $4 7 . 6 0 { \scriptstyle \pm 1 . 8 9 }$ </td><td> $6 8 . 2 4 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $6 4 . 8 7 _ { \pm 1 . 2 4 }$ </td><td> $5 9 . 1 3 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $7 1 . 9 4 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $5 5 . 1 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $6 3 . 2 4 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $8 6 . 2 5 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $8 4 . 8 9 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $5 1 . 6 2 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $3 8 . 1 8 { \scriptstyle \pm 0 . 6 6 }$ </td></tr><tr><td>NTSFormer</td><td> $\underline { { 5 3 . 8 9 } } { \pm 2 . 1 6 }$ </td><td> $4 6 . 9 4 _ { \pm 1 . 9 3 }$ </td><td> $7 1 . 1 9 _ { \pm 0 . 9 3 }$ </td><td> $6 5 . 8 0 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $6 0 . 5 2 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $7 2 . 2 8 \pm 0 . 4 1$ </td><td> $5 4 . 8 3 _ { \pm 0 . 5 3 }$ </td><td> $6 2 . 9 4 _ { \pm 0 . 5 0 }$ </td><td> $8 5 . 8 1 _ { \pm 0 . 7 8 }$ </td><td> $8 4 . 1 4 _ { \pm 0 . 8 3 }$ </td><td> $5 2 . 3 3 _ { \pm 0 . 6 8 }$ </td><td> $3 7 . 8 1 _ { \pm 0 . 6 3 }$ </td></tr><tr><td> $\mathrm { U n i G r a p h } 2$   $\mathrm { L I O N \ ( O u r s ) }$ </td><td> $5 2 . 9 2 { \scriptstyle \pm 1 . 7 9 }$   ${ \bar { \bf 5 8 . 6 1 _ { \pm 1 . 2 8 } } }$ </td><td> $4 6 . 2 8 { \scriptstyle \pm 1 . 6 6 }$   ${ \bf 5 1 . 7 3 _ { \pm 1 . 7 5 } }$ </td><td> $7 2 . 8 2 { \scriptstyle \pm 0 . 8 2 }$   $7 8 . 5 4 _ { \pm 0 . 7 0 }$ </td><td> $6 5 . 3 2 { \scriptstyle \pm 0 . 9 6 }$   ${ \bf 6 8 . 9 1 _ { \pm 1 . 1 3 } }$ </td><td> $5 9 . 8 4 { \scriptstyle \pm 0 . 4 9 }$   ${ \bf 6 2 . 3 1 { _ { \pm 0 . 5 5 } } }$ </td><td> $7 1 . 2 6 { \scriptstyle \pm 0 . 4 7 }$   ${ \bf 7 3 . 8 7 _ { \pm 0 . 3 9 } }$ </td><td> $5 5 . 3 1 { \scriptstyle \pm 0 . 5 4 }$   ${ \bf 5 8 . 4 7 _ { \pm 0 . 8 5 } }$ </td><td> $6 3 . 1 8 { \scriptstyle \pm 0 . 5 2 }$   ${ \bf 6 6 . 5 8 _ { \pm 0 . 4 8 } }$ </td><td> $8 6 . 5 6 { \scriptstyle \pm 0 . 6 8 }$   $\mathbf { 9 0 . 5 3 _ { \pm 0 . 8 9 } }$ </td><td> $8 5 . 2 4 { \scriptstyle \pm 0 . 7 3 }$   $\mathbf { 9 0 . 1 7 _ { \pm 1 . 0 8 } }$ </td><td> $5 4 . 4 2 _ { \pm 0 . 6 4 }$   ${ \bar { \bf 5 8 . 5 4 } } _ { \pm 0 . 7 2 }$ </td><td> $4 0 . 8 6 _ { \pm 0 . 6 2 }$   ${ \bf 4 6 . 1 2 _ { \pm 0 . 8 9 } }$ </td></tr></table>

PERFORMANCE COMPARISON ON MODALITY-LEVEL TASKS.

TABLE III
<table><tr><td>Tasks</td><td colspan="2">Modal Retrieval</td><td colspan="2">G2Text</td><td colspan="2">G2Image</td></tr><tr><td> $ { \mathrm { ~ \textrm ~ { ~ ~ } ~ } }  { \mathrm { D a t a s e t s } }$   $\bf { M e t h o d s } \overbrace { \phantom { F _ { 0 } ^ { ( 1 ) } + \sum _ { p h o d s } ^ { ( 2 ) } } }$ </td><td colspan="2" rowspan="2">Ele-fashion</td><td colspan="2">Flickr30k</td><td colspan="2">SemArt</td></tr><tr><td>MRR Hits@3</td><td>BLEU-4</td><td>CIDEr</td><td>CLIP-S DINOv2-S</td></tr><tr><td>GCNII</td><td>77.24</td><td>70.15</td><td>5.42</td><td>39.54</td><td>50.52</td><td>35.81</td></tr><tr><td>MMGCN</td><td>81.85</td><td>74.58</td><td>6.15</td><td>44.82</td><td>54.88</td><td>39.87</td></tr><tr><td>GATv2</td><td>78.53</td><td>71.32</td><td>5.56</td><td>40.24</td><td>51.25</td><td>36.54</td></tr><tr><td>MGAT</td><td>82.41</td><td>75.26</td><td>6.28</td><td>45.56</td><td>55.43</td><td>40.52</td></tr><tr><td>MLaGA</td><td>87.65</td><td>79.85</td><td>9.54</td><td>71.58</td><td>68.52</td><td>53.15</td></tr><tr><td>GraphGPT-O</td><td>88.45</td><td>80.50</td><td>9.89</td><td>72.58</td><td>70.84</td><td>54.29</td></tr><tr><td>Graph4MM</td><td>86.25</td><td>78.42</td><td>10.42</td><td>74.82</td><td>67.21</td><td>52.56</td></tr><tr><td>InstructG2I</td><td>89.12</td><td>81.24</td><td>9.68</td><td>71.98</td><td>69.92</td><td>55.43</td></tr><tr><td>DMGC</td><td>91.20</td><td>83.12</td><td>7.83</td><td>61.82</td><td>61.54</td><td>47.26</td></tr><tr><td>DGF</td><td>91.55</td><td>83.47</td><td>7.92</td><td>62.20</td><td>61.98</td><td>47.68</td></tr><tr><td>MIG-GT</td><td>92.54</td><td>84.51</td><td>8.15</td><td>63.57</td><td>63.82</td><td>48.94</td></tr><tr><td>NTSFormer</td><td>92.88</td><td>84.93</td><td>8.24</td><td>64.12</td><td>63.26</td><td>48.56</td></tr><tr><td>Unigraph2</td><td>93.13</td><td>85.45</td><td>8.56</td><td>65.28</td><td>64.56</td><td>49.83</td></tr><tr><td>LION (Ours)</td><td>94.67</td><td>86.92</td><td>11.54</td><td>78.92</td><td>74.21</td><td>58.30</td></tr></table>

Theoretical Time-Space Complexity. Let $N = | \mathcal { V } | , M =$ |E|, L be the propagation depth, and $D = 2 ^ { K } d$ be the Clifford multi-vector dimension. The one-time CGP pre-processing costs $\mathcal { O } ( M \cdot 2 ^ { K } \cdot D + L \cdot M \cdot D )$ , while the trainable AHA phase costs $\mathcal { O } ( L \cdot N \cdot D )$ per epoch. The cached multiscale representations require $\mathcal { O } ( L { \cdot } N { \cdot } D )$ memory, and the learnable projection/gating parameters occupy $\mathcal { O } ( D ^ { 2 } )$ space. Since CGP is parameterfree and cached offline, LION avoids recursive neighborhood expansion during training and remains asymptotically linear in graph size for the practical MAG settings considered here. This decoupled computation is particularly useful for large MAGs because propagation results can be reused across epochs and downstream heads. The trainable AHA module therefore operates on fixed multiscale Clifford tokens, which substantially reduces repeated graph traversal while preserving the ability to learn task-specific fusion weights. The memory overhead is dominated by cached propagated features, but this cost is predictable and can be controlled by the propagation depth L and latent dimension d. This trade-off favors stable and efficient training when the same graph is evaluated across multiple downstream tasks.

## IV. EXPERIMENTS

In this section, we first introduce the experimental setup, and additional details for reproducibility are provided in the supplementary material. Then, we provide a comprehensive evaluation to address the following questions: Q1 (Effectiveness). How does LION perform as a new neural paradigm for MAG? Q2 (Ablation). If LION is effective, what contributes to its performance? Q3 (Interpretability). In depth, how do

Algorithm 1 LION: cLIffOrd Neural Paradigm for Multimodal Attributed Graphs Require: Multimodal-Attributed Graph G $\mathsf { \bar { \Gamma } } ( \mathcal { V } , \mathcal { E } , \{ \mathbf { X } ^ { ( m ) } \} _ { m \in \mathcal { M } } )$ , Model Layers L (Graph Propagation Depth). Ensure: Learned Node- or Modality-level Representations Z. 1: /\* Step 1: Clifford Geometric Propagation for Modality Alignment 2: // CGP obtains graph propagated features, which are equivalent to generating aligned tokens by modality alignment. 3: Lift Euclidean modality-attributed features X into the Clifford manifold to obtain $\mathbf { H } ^ { ( 0 ) }$ via Eq. (1). 4: Capture intricate semantic curvature to instantiate $\scriptstyle A _ { \mathcal { G } } : =$ {Φ, R} and then facilitate multimodal interactions via Eq. (4). 5: Perform modality-aware, curvature-adaptive high-order graph propagation to obtain $\{ \mathbf { H } ^ { ( 1 ) } , \ldots , \mathbf { \bar { H } } ^ { ( L ) } \}$ via Eq. (8). 6: /\* Step 2: Adaptive Holographic Aggregation for Modality Fusion 7: // AHA follows CGP to obtain node- or modality-level outputs for specific tasks, which is equivalent to modality fusion. 8: Quantify the information density of modality interaction channels within the propagated features via Eq. (13). 9: Dynamically modulate these channels based on the computed information density and a learnable gate via Eq. (14). 10: // This module facilitates adaptive integration of modality interaction channels by energy-based information density. 11: Capture the consensus profile across propagation depths via Eq. (15). 12: Dynamically modulate the contribution of each depth based on the consensus profile via Eq. (16). 13: // This module facilitates the adaptive integration of multi scale receptive fields by semantic consensus. 14: Return The task-adaptive fusion representation in Euclidean space.

these components exert their influence? Q4 (Robustness). How robust is LION when dealing with sparse scenarios? Q5 (Scalability). What are the time overhead, space overhead, and convergence efficiency of LION?

## A. Experimental Setup

Datasets. In our experiments, we evaluate LION across 9 publicly available MAG datasets from 6 domains, achieving a comprehensive validation. Specifically, these datasets include social network (RedditS) [45], movie network (Movies) [46], 4 recommendation networks (Grocery, Sports, Ele-fashion, Cloth) [46], [47], art network (SemArt) [48], image network (Flickr30k) [49], and book network (Goodreads) [50], [51]. The dataset statistics are summarized in Table I. These datasets cover social, movie, recommendation, art, image, and book networks, supporting both graph-centric and modality-oriented tasks. Their graph scales range from medium-sized image-text and product graphs to the large Goodreads network with more than 0.68M nodes and 7.23M edges, which allows us to evaluate both representation quality and scalability. Their task coverage also spans supervised, unsupervised, retrieval, and generation settings, preventing the evaluation from being biased toward a single form of multimodal supervision. Detailed descriptions of each dataset are provided in Supplementary Appendix F.

Baselines. To achieve a comprehensive comparison, we utilize (i) Single-modality GNN: GCN, GAT, GCNII, GATv2; (ii) Simple MAGNN: MMGCN [14], MGAT [15]; (iii) Conventional MAGNN: MLaGA [18], GraphGPT-O [12], Graph4MM [11], InstructG2I [13]; (iv) Graph-enhanced MAGNN: DMGC [27], DGF [41], MIG-GT [25], NTS-Former [26], UniGraph2 [28]. For fair comparison, all baselines are evaluated under the same dataset splits and task protocols whenever their original implementations support the corresponding downstream task. Details are shown in Supplementary Appendix G.

Evaluation Protocols. Fundamentally, the research motivation for MAG is to enhance data quality for traditional graph tasks and expand the graph downstream task spectrum to improve practical utility. Therefore, evaluation involves (i) Prevalent graph tasks: node classification, link prediction, node clustering; (ii) Novel graph-oriented modality tasks: modality retrieval, Graph-to-Text (G2Text), Graph-to-Image (G2Image). Given the complexity of the evaluation pipelines, hyperparameter settings, and quantitative metrics, please refer to Supplementary Appendix H for comprehensive details. All nine benchmarks provide text and image attributes. Graph-task and ablation entries shown with “±” are mean±std over five independent runs; modality retrieval/generation entries without “±” follow their benchmark single-run protocols.

Experiment Environment. Experiments are conducted on a workstation equipped with Intel Xeon Scalable processors and NVIDIA RTX 6000 Ada Generation GPUs with 96GB of VRAM, supported by 256GB of system RAM. The computational environment utilizes CUDA 12.9, while software implementations are developed using Python 3.10.18 and PyTorch 2.8.

## B. Performance Comparison

Graph Tasks. To answer Q1, we first report performance comparison on graph tasks. As shown in Table II, LION consistently achieves superior performance across all datasets, tasks, and metrics. Specifically, in node classification, LION outperforms the leading NTSFormer and Unigraph2 by a significant average margin of 5.84%. This success highlights the capacity of LION to capture topology and modality insights through the Clifford geometric manifold. The gains are consistent across both classification and link-level prediction, indicating that LION does not merely improve a single supervised objective but learns transferable graph representations. This is important for MAGs because downstream labels are often sparse, while relational and multimodal cues are available at different granularities. By aligning modality attributes through topology-aware geometric propagation, LION can exploit these cues before the task-specific training stage.

TABLE IV  
ABLATION STUDY.
<table><tr><td rowspan=1 colspan=1>Tasks</td><td rowspan=1 colspan=4>Node Classification               Link Prediction</td></tr><tr><td rowspan=2 colspan=1> $\sim$ Datasets $\bf { M e t h o d s } \overbrace { \phantom { F _ { 0 } ^ { ( 1 ) } + \left( \frac { d _ { 0 } ^ { 2 } } { d _ { s } ^ { 2 } } \right) } } ^ { \substack { \bf { M e t h o d s } } }$ </td><td rowspan=1 colspan=1>RedditS</td><td rowspan=1 colspan=1>Grocery</td><td rowspan=1 colspan=1>Flickr30k</td><td rowspan=1 colspan=1>SemArt</td></tr><tr><td rowspan=1 colspan=1>Acc   F1</td><td rowspan=1 colspan=1> $\operatorname { A c c }$    F1</td><td rowspan=1 colspan=1>MRR  $\overline { { \mathrm { H i t s } @ 3 } }$ </td><td rowspan=1 colspan=1>MRR  $\overline { { \mathrm { H i t s } @ 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>LION (Ours)</td><td rowspan=1 colspan=1> $\mathbf { 9 4 . 8 _ { \pm 0 . 4 } }$ 90.6±0.5</td><td rowspan=1 colspan=1> $\overline { { 8 8 . 3 _ { \pm 0 . 7 } } }$  $7 8 . 1 { \pm } 0 . 9$ </td><td rowspan=1 colspan=1> $\overline { { 6 8 . 4 _ { \pm 0 . 5 } } }$  $\overline { { 7 6 . 5 _ { \pm 0 . 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 5 . 2 _ { \pm 0 . 3 } } }$  ${ \bf 8 9 . 6 { \scriptstyle \pm 0 . 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Rotor</td><td rowspan=1 colspan=1> $9 3 . 9 { \pm } 0 . 3 $  $8 9 . 4 \pm 0 . 4$ </td><td rowspan=1 colspan=1> $8 7 . 4 \pm 0 . 6$  $7 7 . 6 { \scriptstyle \pm 0 . 8 }$ </td><td rowspan=1 colspan=1> $6 7 . 8 { \scriptstyle \pm 0 . 5 }$  $7 6 . 0 { \scriptstyle \pm 0 . 4 }$ </td><td rowspan=1 colspan=1> $8 4 . 7 _ { \pm 0 . 3 }$  $8 9 . 1 _ { \pm 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Potential</td><td rowspan=1 colspan=1> $9 3 . 4 _ { \pm 0 . 4 }$  $8 8 . 7 _ { \pm 0 . 5 }$ </td><td rowspan=1 colspan=1> $8 6 . 6 _ { \pm 0 . 7 }$  $7 6 . 8 _ { \pm 0 . 8 }$ </td><td rowspan=1 colspan=1> $6 7 . 2 _ { \pm 0 . 4 }$  $7 5 . 7 _ { \pm 0 . 5 }$ </td><td rowspan=1 colspan=1> $8 4 . 3 _ { \pm 0 . 3 }$  $8 8 . 8 _ { \pm 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Energy</td><td rowspan=1 colspan=1> $9 2 . 8 { \scriptstyle \pm 0 . 5 }$  $8 8 . 3 { \scriptstyle \pm 0 . 6 }$ </td><td rowspan=1 colspan=1> $8 6 . 2 \pm 0 . 8$  $7 5 . 9 { \pm } 1 . 0$ </td><td rowspan=1 colspan=1> $6 6 . 5 { \scriptstyle \pm 0 . 6 }$  $7 4 . 5 { \scriptstyle \pm 0 . 5 }$ </td><td rowspan=1 colspan=1> $8 3 . 5 { \scriptstyle \pm 0 . 4 }$  $8 7 . 5 { \scriptstyle \pm 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Consen.</td><td rowspan=1 colspan=1> $9 2 . 6 { \scriptstyle \pm 0 . 4 }$  $8 8 . 5 { \scriptstyle \pm 0 . 6 }$ </td><td rowspan=1 colspan=1> $8 5 . 9 { \scriptstyle \pm 0 . 7 }$  $7 6 . 6 { \scriptstyle \pm 0 . 9 }$ </td><td rowspan=1 colspan=1> $6 7 . 0 { \scriptstyle \pm 0 . 5 }$  $7 4 . 8 { \scriptstyle \pm 0 . 5 }$ </td><td rowspan=1 colspan=1> $8 2 . 9 { \scriptstyle \pm 0 . 4 }$  $8 7 . 9 { \scriptstyle \pm 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Scale</td><td rowspan=1 colspan=1> $9 1 . 7 _ { \pm 0 . 6 }$  $8 6 . 8 _ { \pm 0 . 7 }$ </td><td rowspan=1 colspan=1> $8 4 . 5 _ { \pm 0 . 9 }$  $7 5 . 5 { \scriptstyle \pm 1 . 1 }$ </td><td rowspan=1 colspan=1> $6 6 . 8 _ { \pm 0 . 7 }$  $7 3 . 5 \pm 0 . 6$ </td><td rowspan=1 colspan=1> $8 2 . 5 _ { \pm 0 . 5 }$  $8 6 . 7 _ { \pm 0 . 4 }$ </td></tr><tr><td rowspan=1 colspan=1>Tasks</td><td rowspan=1 colspan=1> $\mathbb { N } \mathbb { n } \mathbb { Q } \mathbb { Q }$ </td><td rowspan=1 colspan=1> $\mathbb { E } \mathbb { E } \mathbb { e } \mathbb { i } \mathbb { m } \mathbb { s }$ </td><td rowspan=1 colspan=1>Multimodal</td><td rowspan=1 colspan=1>Retrieval</td></tr><tr><td rowspan=2 colspan=1> $ { \mathrm { ~ \textrm ~ { ~ ~ } ~ } }  { \mathrm { D a t a s e t s } }$  $\bf { M e t h o d } \hat { s } \sim $ </td><td rowspan=1 colspan=1>Movies</td><td rowspan=1 colspan=1>Ele-Fashion</td><td rowspan=1 colspan=1>Sports</td><td rowspan=1 colspan=1>Cloth</td></tr><tr><td rowspan=1 colspan=1>NMI   ARI</td><td rowspan=1 colspan=1>NMI   ARI</td><td rowspan=1 colspan=1>MRRHits@3</td><td rowspan=1 colspan=1>MRR Hits@3</td></tr><tr><td rowspan=1 colspan=1>LION (Ours)</td><td rowspan=1 colspan=1> $2 4 . 6 _ { \pm 0 . 4 }$  $\mathbf { 1 0 . 6 { \scriptstyle \pm 0 . 4 } }$ </td><td rowspan=1 colspan=1> ${ \bf 5 6 . 8 _ { \pm 0 . 6 } }$  ${ \bf 4 8 . 9 2 0 . 8 }$ </td><td rowspan=1 colspan=1> $\overline { { 9 5 . 2 _ { \pm 0 . 2 } } }$  $\overline { { 8 8 . 7 _ { \pm 0 . 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 9 2 . 5 _ { \pm 0 . 1 } } }$  $\mathbf { 8 9 . 1 _ { \pm 0 . 2 } }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Rotor</td><td rowspan=1 colspan=1> $2 3 . 7 { \pm } 0 . 3$  $1 0 . 0 { \scriptstyle \pm 0 . 3 }$ </td><td rowspan=1 colspan=1> $5 6 . 0 { \scriptstyle \pm 0 . 5 }$  $4 7 . 9 { \scriptstyle \pm 0 . 7 }$ </td><td rowspan=1 colspan=1> $9 4 . 3 { \scriptstyle \pm 0 . 2 }$  $8 8 . 1 { \pm } 0 . 1$ </td><td rowspan=1 colspan=1> $9 1 . 8 { \scriptstyle \pm 0 . 1 }$  $8 8 . 6 { \scriptstyle \pm 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Potential</td><td rowspan=1 colspan=1> $2 3 . 5 { \scriptstyle \pm 0 . 4 }$   $9 . 4 { \pm } 0 . 3 $ </td><td rowspan=1 colspan=1> $5 5 . 9 { \scriptstyle \pm 0 . 5 }$   $4 8 . 1 \pm 0 . 7$ </td><td rowspan=1 colspan=1> $9 3 . 9 { \scriptstyle \pm 0 . 2 }$  $8 7 . 9 { \scriptstyle \pm 0 . 1 }$ </td><td rowspan=1 colspan=1> $9 0 . 5 { \scriptstyle \pm 0 . 1 }$  $8 7 . 8 { \scriptstyle \pm 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Energy</td><td rowspan=1 colspan=1> $2 2 . 8 { \scriptstyle \pm 0 . 5 }$   $8 . 9 { \pm } 0 . 4$ </td><td rowspan=1 colspan=1> $5 4 . 6 { \scriptstyle \pm 0 . 7 }$  $4 6 . 9 { \pm } 0 . 9$ </td><td rowspan=1 colspan=1> $9 3 . 8 { \scriptstyle \pm 0 . 3 }$  $8 7 . 0 { \scriptstyle \pm 0 . 2 }$ </td><td rowspan=1 colspan=1> $9 0 . 9 { \scriptstyle \pm 0 . 2 }$  $8 7 . 1 { \pm } 0 . 3 $ </td></tr><tr><td rowspan=1 colspan=1>w/o Consen.</td><td rowspan=1 colspan=1> $2 3 . 2 { \pm } 0 . 4$   $8 . 7 { \pm } 0 . 3 $ </td><td rowspan=1 colspan=1> $5 4 . 8 { \scriptstyle \pm 0 . 6 }$  $4 6 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td rowspan=1 colspan=1> $9 3 . 5 { \scriptstyle \pm 0 . 3 }$  $8 6 . 4 \pm 0 . 2$ </td><td rowspan=1 colspan=1> $9 0 . 4 { \scriptstyle \pm 0 . 2 }$  $8 7 . 3 { \scriptstyle \pm 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1>w/o Scale</td><td rowspan=1 colspan=1> $2 2 . 9 _ { \pm 0 . 5 }$  $8 . 2 _ { \pm 0 . 2 }$ </td><td rowspan=1 colspan=1> $5 4 . 2 _ { \pm 0 . 8 }$  $4 5 . 9 { \scriptstyle \pm 1 . 0 }$ </td><td rowspan=1 colspan=1> $9 3 . 0 _ { \pm 0 . 4 }$  $8 6 . 3 _ { \pm 0 . 3 }$ </td><td rowspan=1 colspan=1> $9 0 . 1 _ { \pm 0 . 3 }$  $8 6 . 8 _ { \pm 0 . 4 }$ </td></tr></table>

Regarding the baselines, graph-enhanced methods like NTS-Former generally outperform conventional methods such as $\mathbf { M L a G A } .$ , yet they remain inferior to LION primarily due to their reliance on suboptimal Euclidean spaces. Furthermore, most approaches fail to maintain consistent dominance across all three tasks and six metrics. For instance, while clusteringspecific DMGC and DGF exhibit competitiveness performance, they fail to generalize effectively to the node classification task. Meanwhile, there are currently no specialized models designed for the link-level task. These observations suggest that simply attaching graph modules to multimodal encoders is insufficient for a general MAG neural paradigm. A unified representation space is needed so that the same model can support node-, edge-, and cluster-level reasoning without redesigning task-specific interaction modules.

Modality Tasks. Table III further validates the superiority of LION in modality tasks, encompassing prevalent retrieval and generation, where it achieves SOTA performance across all evaluation metrics. A notable performance gain is observed on the SemArt dataset for the G2Image task, where LION outperforms the leading baselines, GraphGPT-O and InstructG2I, by 4.76% in CLIP-S and 5.18% in DINOv2- S. These results demonstrate that our alignment-then-fusion MAG neural paradigm effectively preserves the semantic integrity of multimodal attributes. This strategy facilitates more coherent and accurate content generation compared to existing methods that treat modality and topology as separate spaces. Notably, modality-agnostic prevalent GNNs and early MMGNNs fail to achieve competitive performance due to their outdated model architecture designs. The retrieval and generation results further show that topology is useful not only for graph prediction but also for modality-level semantic grounding. In retrieval tasks, aligned Clifford tokens improve cross-modal matching by reducing the discrepancy between visual and textual neighborhoods. In generation tasks, topologyaware fusion supplies structured context that guides the decoder or diffusion backbone toward outputs that are semantically compatible with the graph. Therefore, LION turns graph structure into a reusable multimodal prior rather than treating it as an additional feature appended to the input.

## C. Ablation Study

To answer Q2, we conduct an ablation study to investigate the contribution of each module within LION. As detailed in Sec. III, LION consists of two core modules, namely CGP and AHA. To evaluate their impact, we define five module variants. For CGP in Eq. (8), (1) w/o Rotor and (2) w/o Potential remove the spatial rotor R and geometric potential Φ . For AHA in Eq. (14), Eq. (15), and Eq. (16), (3) w/o Energy, (4) w/o Consen., and (5) w/o Scale omit the energy-aware gating, consensus profile, and scale-aware resonance fusion.

Based on this, Table IV confirms the necessity of module design. Regarding the CGP, removing the spatial rotor and geometric potential leads to a significant performance drop. This verifies that modeling topology as geometric rotations and differentiating interaction principles via semantic curvature is crucial for effective alignment. The two CGP variants also reveal complementary roles of the rotor and potential. The rotor determines how neighbor semantics are transported into the target tangent space, while the potential controls the strength of curvature-aware interaction. Removing either component weakens the geometric interpretation of propagation and reduces the quality of aligned tokens. Regarding the AHA, the w/o Energy variant exhibits poor performance. This demonstrates that indiscriminately aggregating all propagated modality components introduces redundancy and confirms that component-wise filtering based on information density is critical. Furthermore, the performance degradation in w/o Scale and w/o Consen. confirms that adaptive fusion guided by the consensus profile allows the model to reconcile receptive fields, which is beneficial for different propagated representations. These results support the view that modality fusion should consider both channel reliability and topology scale, rather than relying on a fixed pooling operation over all propagated features.

## D. In-depth Analysis

CGP Module. To answer Q3, we provide a visual analysis to intuitively interpret the underlying mechanisms of the CGP and AHA modules. First of all, we plot the intraand inter-modality interaction strength (y-axis) against connected node feature similarity (x-axis). For CGP, the plotted quantities are $I _ { \mathrm { i n t r a } } = \| \langle h _ { u } h _ { v } \rangle _ { 0 } \| / ( \| h _ { u } \| \| h _ { v } \| + \epsilon )$ and $I _ { \mathrm { i n t e r } } = \| \langle h _ { u } h _ { v } \rangle _ { 2 } \| / ( \| h _ { u } \| \| h _ { v } \| + \epsilon )$ ; for baselines, we use the corresponding normalized message/attention coefficient, with zero only when no cross-modal message is defined. As evident in Fig. 2 (a)-(b) for Flickr30k and Fig. 2 (e)-(f) for SemArt, the modality interaction baselines exhibit a sharp decay in interaction strength as node feature similarity decreases, which reveals a heavy reliance on the homophily assumption. Notably, the modality interactions in graph-enhanced methods occur exclusively within modality-specific views, resulting in an inter-modality interaction strength of zero.

In contrast, CGP breaks the node homophily assumption by modality-aware geometric manifolds and topology-adaptive high-order graph propagation. Specifically, by harmonizing modality interaction principles (Sec. III-B), CGP effectively facilitates both intra- and inter-modality interactions. Based

Graph-enhanced Potential (CGP)

Conventional (Q-Former)

![](images/2090fbab0d6ad3afb472b496c71c5eeacd0d8fa5344d941a4066542910c4c7e3.jpg)  
(a) Intra-modality

![](images/40dcb596912990ba2ecae6a678be2ff88f9efb90c906d22a31cf87c866f23107.jpg)  
(b) Inter-modality

![](images/a47acda73731b7642719b8348d7ecad2c5386f39dc0bcef980c21534fd271027.jpg)  
(e) Intra-modality

![](images/4cfb8c73d313fd19c68de563c630c5d6eb21194bfbde83baf4b5a208f4507a69.jpg)  
(f) Inter-modality

![](images/c41a697b3e4b558f9b606874f04e6e6c36b393b904b21282168ce10ab0247a8c.jpg)

(c) Before Alignment  
![](images/1e611c61d9c17f0ab41f381bacb6dd095544fc9b18e89b975e7e06cc851e9757.jpg)  
(g) Before Alignment

Image-only Image-neighbor  
![](images/512d4452323edf67e87e396d059b8f3cbf5696487c3cf85d904f7e320a004a92.jpg)

(d) After CGP Alignment  
![](images/f328e8715c20d3973d8ee2a009fcce848833ee7ba1dd70182809c6809e555ebb.jpg)  
(h) After CGP Alignment

Fig. 2. (a)-(d) The G2Text visualization of CGP modality interaction and alignment (i.e., geometric-induced adaptive graph propagation) on Flickr30k. (e)-(h) The G2Image visualization of CGP modality interaction and alignment on SemArt.  
![](images/829091b2235281a2e2c5e4ee7ab1861bf9ca5d68d677477ed24862c02ceef37c.jpg)  
(a) Flickr30k (Multimodal Retrieval)

![](images/340d5a164d915e6af61e66be9be5dd6570e28026ba46a2285fbd9258865e52af.jpg)  
Fig. 3. Hyperparameter analysis of CGP depth.  
(a) SemArt (Multimodal Retrieval)

![](images/1b856021c3cd67f9c290a6b3795fbf08fa6fcd7d9889583320f149cf47c05fc9.jpg)  
(a) Goodreads (Link Prediction)

![](images/f6d5373a096f8900edfdeadd030dcd9da4236c1794ed609f2c262a2d10717777.jpg)  
(b) Ele-fashion (Link Prediction)

on this, we visualize the token latent space before and after alignment as an interpretive diagnostic of how curvature-aware multimodality interaction reshapes the representation; the task results and ablations provide the quantitative performance evidence. The visualization also helps explain why the proposed propagation differs from conventional homophily smoothing. Instead of forcing all neighboring nodes to become similar in a scalar space, CGP transports modality-aware geometric bases according to the learned curvature. This allows semantically compatible signals to be aligned while preserving the distinction between different interaction channels.

Prior to alignment, the latent spaces of modalities appear disjointed. After CGP, a distinct trend emerges where entity representations transcend modality boundaries to capture high level semantic dependencies. This strategy effectively gathers semantically related modalities while distancing others. Meanwhile, G2Text and G2Image tasks exhibit enhanced text and image compactness, reflecting an adaptive alignment that conforms to the specific downstream requirements. This result intuitively demonstrates that our spatial rotor and geometric potential successfully guide the parallel transport of modality attribute vectors for efficient alignment. Furthermore, Fig. 3 illustrates the impact of varying CGP propagation depths on model performance. Experimental results suggest that a smaller propagation depth is preferable for dense graphs, while sparse graphs benefit from increased depth to capture a sufficient receptive field.

Fig. 4. AHA visualization. T and I are text and image; T-T, T-I, I-T, and I-I denote ordered modality-pair diagnostics rather than one-to-one Clifford basis blades.  
TABLE V  
EPOCH-BATCH EFFICIENCY ON GOODREADS (CLASSIFICATION).
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Pre-process (s)</td><td rowspan=1 colspan=1>E-Train. (s)</td><td rowspan=1 colspan=1>E-Infer. (s)</td><td rowspan=1 colspan=1>B-GPU Mem.</td><td rowspan=1 colspan=1>Param.</td></tr><tr><td rowspan=1 colspan=1>MLaGA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\overline { { 1 1 2 . 4 _ { \pm 1 . 0 5 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 4 5 . 2 _ { \pm 0 . 9 3 } } }$ </td><td rowspan=1 colspan=1>22.8G</td><td rowspan=1 colspan=1>125M</td></tr><tr><td rowspan=1 colspan=1>GraphGPT-O</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1> $1 2 5 . 6 _ { \pm 1 . 7 0 }$ </td><td rowspan=1 colspan=1> $4 8 . 9 _ { \pm 1 . 4 5 }$ </td><td rowspan=1 colspan=1>24.5G</td><td rowspan=1 colspan=1>140M</td></tr><tr><td rowspan=1 colspan=1>Graph4MM</td><td rowspan=1 colspan=1> $4 5 . 2 { \scriptstyle \pm 1 . 0 3 }$ </td><td rowspan=1 colspan=1> $2 8 . 5 { \scriptstyle \pm 0 . 8 5 }$ </td><td rowspan=1 colspan=1> $1 0 . 4 _ { \pm 0 . 4 2 }$ </td><td rowspan=1 colspan=1>17.2G</td><td rowspan=1 colspan=1>96M</td></tr><tr><td rowspan=1 colspan=1>InstructG2I</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1> $1 6 8 . 2 _ { \pm 2 . 5 0 }$ </td><td rowspan=1 colspan=1> $7 5 . 6 _ { \pm 1 . 1 0 }$ </td><td rowspan=1 colspan=1>32.1G</td><td rowspan=1 colspan=1>180M</td></tr><tr><td rowspan=1 colspan=1>DMGC</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $9 . 2 { \scriptstyle \pm 0 . 1 5 }$ </td><td rowspan=1 colspan=1> $4 . 1 { \pm } 0 . 0 8$ </td><td rowspan=1 colspan=1>3.8G</td><td rowspan=1 colspan=1>2.4M</td></tr><tr><td rowspan=1 colspan=1>DGF</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1> $1 0 . 8 _ { \pm 0 . 2 0 }$ </td><td rowspan=1 colspan=1> $3 . 6 _ { \pm 0 . 1 0 }$ </td><td rowspan=1 colspan=1>4.1G</td><td rowspan=1 colspan=1>2.6M</td></tr><tr><td rowspan=1 colspan=1>MIG-GT</td><td rowspan=1 colspan=1> $6 3 . 7 _ { \pm 1 . 1 6 }$ </td><td rowspan=1 colspan=1> $5 . 4 _ { \pm 0 . 1 2 }$ </td><td rowspan=1 colspan=1> $2 . 5 _ { \pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1>2.2G</td><td rowspan=1 colspan=1>1.8M</td></tr><tr><td rowspan=1 colspan=1>NTSFormer</td><td rowspan=1 colspan=1> $7 5 . 8 { \scriptstyle \pm 1 . 2 5 }$ </td><td rowspan=1 colspan=1> $4 . 1 { \pm } 0 . 1 0$ </td><td rowspan=1 colspan=1> $1 . 9 { \scriptstyle \pm 0 . 0 4 }$ </td><td rowspan=1 colspan=1>2.5G</td><td rowspan=1 colspan=1>2.1M</td></tr><tr><td rowspan=1 colspan=1>LION (Ours)</td><td rowspan=1 colspan=1> $9 2 . 4 _ { \pm 1 . 4 9 }$ </td><td rowspan=1 colspan=1> $3 . 2 _ { \pm 0 . 0 9 }$ </td><td rowspan=1 colspan=1> $1 . 2 _ { \pm 0 . 0 4 }$ </td><td rowspan=1 colspan=1>2.9G</td><td rowspan=1 colspan=1>1.5M</td></tr></table>

AHA Module. First of all, we clarify the modality fusion mechanism implemented from a holographic perspective involving both energy and scale (Sec. III-C). In Fig. 4, modality channels are primarily filtered via energy-initialized adaptive gating to select the informative and beneficial interaction subspaces. Then, propagation depth determines the manner in which multi-scale structural contexts are integrated within these interaction channels. Based on this, we reveal that LION

![](images/dc66bb01243ff2a2ac91fa03e281b5d208e22011df754f80361fd44699f4b206.jpg)  
(a) Node Clustering

![](images/652ca0a7443cfc6823996df5185fb880ff1f0a099926ca22b65230a275f9eb4e.jpg)  
(b) Graph-to-Text  
Fig. 5. Performance of LION under different sparsity levels on the Grocery dataset.

## E. Robustness Analysis

dynamically assigns varying importance to channels (energy) and depth (scale) according to the specific scenarios. For instance, a distinct modality-aware preference emerges where the model assigns significantly higher weights to text-related subspaces on Goodreads while shifting its focus toward image related subspaces on Ele-Fashion. Meanwhile, their optimal receptive fields also vary across these datasets to ensure optimal modality fusion. This adaptive behavior is consistent with the heterogeneous nature of MAGs. Some domains depend more heavily on textual semantics, such as book summaries and reviews, whereas fashion-related graphs often require visual compatibility cues. A fixed fusion rule would obscure such domain-specific preferences, but the energy-scale design allows LION to allocate capacity to the most informative channels and propagation ranges.

To answer Q4, we investigate the robustness of LION in sparse scenarios. This stems from the fact that MAGs often suffer from data incompleteness. As shown in Fig. 5, LION demonstrates remarkable robustness across all sparsity scenarios. Meanwhile, LION consistently outperforms other methods in all downstream tasks. This superiority is attributed to the Clifford geometric manifold, which can mitigate missing data through high-order graph propagation. This indicates that our alignment-then-fusion mechanism effectively exploits the complementary nature of topology and modality, allowing the model to learn high-quality representations even when raw information is severely corrupted. In sparse scenarios, local attributes alone become unreliable because either modality content or neighborhood evidence may be partially absent. CGP alleviates this issue by propagating curvature-aware context from broader graph neighborhoods, while AHA filters noisy channels before final prediction. As a result, the model remains stable even when the input graph provides incomplete or imbalanced multimodal evidence.

## F. Running Efficiency

To answer Q5, we provide the computational overhead and convergence curves of LION. As summarized in Table V, LION demonstrates superior efficiency compared to other baselines. This advantage is primarily attributed to our decoupled graph neural paradigm, which separates intensive geometric computations from the training loop. Specifically, the construction of the spatial rotor and geometric potential within the CGP is relegated to a one-time pre-processing. Consequently, the training phase focuses exclusively on the AHA and avoids the need for recursive neighborhood expansion. Fig. 6 illustrates that LION exhibits a rapid convergence rate, consistently reaching peak performance in significantly fewer epochs than other baselines. This empirical training curve is consistent with using stable cached CGP tokens, while Theorem III.2 concerns spectral contraction of the fixed CGP propagation operator rather than the convergence rate of the downstream optimizer. By reducing repeated structural computation in the trainable loop, LION facilitates efficient adaptation. The convergence behavior also reflects the benefit of decoupling propagation from trainable aggregation. Since CGP produces stable multiscale tokens before optimization, the trainable module does not need to repeatedly discover structural context from scratch. Instead, it focuses on selecting informative interaction channels and receptive fields, which explains the lower epoch-level training and inference cost in Table V.

(c) Graph-to-Image  
![](images/12218480aaeeafc01048b3d5173ebb1bcb778de15c870f1bbed719efe8bed886.jpg)

(d) Node Classification  
![](images/9173c1f955597c3b0ba1f9cb60a130243ce5a499cf7e32aa486e4ec5d042e02a.jpg)

![](images/1b82e03da0342cb3e84bafd0ce5ce3157e3b7be3824d43fe601a0253f44670ea.jpg)  
Fig. 6. The convergence efficiency curves.

## V. CONCLUSION

With the rapid advancement of multimodal domains, MAG has emerged as a critical frontier for enhancing graph representation and downstream utility. In this work, we propose LION, which unifies topology with modality by constructing the geometric manifold grounded in Clifford algebra. Our theoretical analysis establishes bounded geometric sensitivity, conditional spectral contraction of CGP, and stability bounds for AHA. Extensive experiments across a broad spectrum of supervised and unsupervised tasks demonstrate that LION significantly outperforms SOTA baselines, confirming its superiority in both graph and modality tasks. Although the manifold dimension scales exponentially with the number of modalities, this overhead remains marginal in practice, as the modality count is typically limited. Future work includes the integration of LION into broader applications such as Graph QA. Furthermore, enhancing the utility of Multimodal Large Models within LION to exploit cross-modal semantics under topology priors is also a compelling direction.

## APPENDIX A

PROOF OF GEOMETRIC STABILITY IN THEOREM 1

This section proves the stability statement in Theorem 1 using the same initialization, potential, and stabilized rotor as the main paper.

Proof. For modality k, let $\bar { x } _ { u } ^ { ( k ) } = g _ { k } ( x _ { u } ^ { ( k ) } ) \in \mathbb R ^ { d }$ and $\hat { x } _ { u } ^ { ( k ) } =$ $\bar { x } _ { u } ^ { ( k ) } \big / \operatorname* { m a x } ( \| \bar { x } _ { u } ^ { ( k ) } \| _ { 2 } , \epsilon _ { x } )$ with fixed $\epsilon _ { x } > 0$ . Assume each $g _ { k }$ is Lipschitz on the bounded input domain. The initialization is

$$
\mathbf { H } _ { u } ^ { ( 0 ) } = \frac { 1 } { \sqrt { K } } \sum _ { k = 1 } ^ { K } \hat { x } _ { u } ^ { ( k ) } e _ { k } , \qquad \| \mathbf { H } _ { u } ^ { ( 0 ) } \| _ { \mathcal { C } l } \leq 1 .\tag{18}
$$

Stabilized normalization is Lipschitz, and the orthogonal Clifford lift is linear; hence for some $L _ { H } < \infty$

$$
\| \mathbf { H } - \mathbf { H } ^ { \prime } \| _ { \mathcal { C } l } \leq L _ { H } \| \mathbf { X } - \mathbf { X } ^ { \prime } \| _ { \mathcal { C } l } \equiv \delta _ { H } .\tag{19}
$$

For an edge, the geometric product is bilinear. Because both lifted representations have norm at most one,

$$
\| \mathbf { H } _ { u } \mathbf { H } _ { v } - \mathbf { H } _ { u } ^ { \prime } \mathbf { H } _ { v } ^ { \prime } \| c \imath \leq 2 \delta _ { H } ,\tag{20}
$$

and the Grade-0 and Grade-2 projections are non-expansive. Let $\mathcal { S } _ { u v } = \langle \mathbf { H } _ { u } \mathbf { H } _ { v } \rangle _ { 0 }$ and $B _ { u v } = \langle \mathbf { H } _ { u } \mathbf { H } _ { v } \rangle _ { 2 }$ . The rotor used in the main paper is

$$
\begin{array} { r l } & { \widehat { \mathcal { B } } _ { u v } = \frac { \mathcal { B } _ { u v } } { \sqrt { \| \mathcal { B } _ { u v } \| ^ { 2 } + \epsilon _ { r } ^ { 2 } } } , } \\ & { \theta _ { u v } = \mathrm { a t a n 2 } ( \| \mathcal { B } _ { u v } \| , \| \mathcal { S } _ { u v } \| + \epsilon _ { r } ) , } \\ & { \mathcal { R } _ { u v } = \exp ( - \theta _ { u v } \widehat { \mathcal { B } } _ { u v } / 2 ) . } \end{array}\tag{21}
$$

With fixed $\epsilon _ { r } > 0 $ , both stabilized normalization and atan2 are Lipschitz on the bounded domain, including $\| B _ { u v } \|  0 ;$ the exponential is Lipschitz on the resulting compact set. Thus $\| \mathcal { R } - \mathcal { R } ^ { \prime } \| \leq L _ { R } \delta _ { H }$ . The potential in the main paper contains a positive stabilizer ϵ and is likewise Lipschitz, $\| \Phi - \Phi ^ { \prime } \| \leq$ $L _ { \Phi } \delta _ { H }$

Writing $\mathcal { A } _ { \mathcal { G } } = \mathbf { A } \circ ( \Phi , \mathcal { R } )$ , bounded degree and the product rule give

$$
\| \boldsymbol { \mathcal { A } } _ { \mathcal { G } } - \boldsymbol { \mathcal { A } } _ { \mathcal { G } } ^ { \prime } \| _ { \mathcal { C } l } \leq C _ { 1 } \delta _ { H } + C _ { 2 } \| \mathbf { A } - \mathbf { A } ^ { \prime } \| _ { \mathcal { C } l } .\tag{22}
$$

Combining the two inequalities yields

$$
\begin{array} { r l } & { \| \mathbf { H } - \mathbf { H } ^ { \prime } \| _ { \mathcal { C } l } + \| \mathcal { A } _ { \mathcal { G } } - \mathcal { A } _ { \mathcal { G } } ^ { \prime } \| _ { \mathcal { C } l } } \\ & { \quad \leq K _ { \operatorname* { m a p } } ( \| \mathbf { X } - \mathbf { X } ^ { \prime } \| _ { \mathcal { C } l } + \gamma \| \mathbf { A } - \mathbf { A } ^ { \prime } \| _ { \mathcal { C } l } ) . } \end{array}\tag{23}
$$

where $K _ { \mathrm { m a p } }$ and γ depend on the encoder/projection bounds, graph degree, and fixed stabilizers. This proves Theorem 1 without requiring the bi-vector norm to be bounded away from zero. □

## APPENDIX B

PROOF OF SPECTRAL EVIDENCE IN THEOREM 2

We analyze exactly the normalized CGP update defined in the main paper. During one cached propagation stage, assume an undirected graph, symmetric potentials $\Phi _ { u v } = \Phi _ { v u }$ , reciprocal isometric transports $\mathcal { T } _ { v u } = \mathcal { T } _ { u v } ^ { - 1 }$ , and fixed edge operators. Define $\begin{array} { r } { \bar { \Phi } _ { u u } = 1 , \bar { \Phi } _ { u v } = \Phi _ { u v } , d _ { u } = \sum _ { k \in \mathcal { N } ( u ) \cup \{ u \} } \bar { \Phi } _ { u k } } \end{array}$ , and $\tilde { \Phi } _ { u v } = \bar { \Phi } _ { u v } / d _ { u }$ . Then the main-paper propagation is

$$
\mathbf { H } _ { u } ^ { ( l ) } = \sum _ { v \in \mathcal { N } ( u ) \cup \{ u \} } \tilde { \Phi } _ { u v } \mathcal { T } _ { u v } \big ( \mathbf { H } _ { v } ^ { ( l - 1 ) } \big ) \equiv \big ( \mathcal { P } _ { \mathcal { G } } \mathbf { H } ^ { ( l - 1 ) } \big ) _ { u } .\tag{24}
$$

The corresponding unnormalized connection Dirichlet energy is

$$
\mathbb { E } _ { \mathrm { D i r } } ( \mathbf { H } ) = \frac { 1 } { 2 } \sum _ { ( u , v ) \in \mathcal { E } } \Phi _ { u v } \Vert \mathbf { H } _ { u } - \mathcal { T } _ { u v } ( \mathbf { H } _ { v } ) \Vert _ { \mathcal { C } l } ^ { 2 } .\tag{25}
$$

Symmetry, reciprocity, and isometry make the associated connection Laplacian positive semidefinite. After degree normalization, $\mathcal { P } _ { \mathcal { G } }$ is the connection averaging operator associated with this Laplacian (including the stated self loop) and is self-adjoint under the degree-weighted inner product $\begin{array} { r } { \langle { \bf X } , { \bf Y } \rangle _ { D } = \bar { \sum _ { u } } d _ { u } \langle { \bf X } _ { u } , { \bf Y } _ { u } \rangle _ { \mathcal { C } l } } \end{array}$ . Let Π be the corresponding orthogonal projection onto the unit-eigenspace of $\mathcal { P } _ { \mathcal { G } }$ . Decompose $\mathbf { \bar { H } } ^ { ( 0 ) } = \bar { \Pi } \mathbf { H } ^ { ( 0 ) } + ( \mathbf { I } - \Pi ) \mathbf { H } ^ { ( 0 ) }$ . Because $\mathcal { P } _ { \mathcal { G } } \Pi = \Pi$

$$
\mathbf { H } ^ { ( l ) } - \Pi \mathbf { H } ^ { ( 0 ) } = \mathcal { P } _ { \mathcal { G } } ^ { l } ( \mathbf { I } - \Pi ) \mathbf { H } ^ { ( 0 ) } .\tag{26}
$$

If the spectral radius of $\mathcal { P } _ { \mathcal { G } }$ on the complementary invariant subspace is $\rho < 1$ , operator submultiplicativity gives

$$
\| \mathbf { H } ^ { ( l ) } - \Pi \mathbf { H } ^ { ( 0 ) } \| _ { D , \mathcal { C } l } \leq \rho ^ { l } \| \mathbf { H } ^ { ( 0 ) } - \Pi \mathbf { H } ^ { ( 0 ) } \| _ { D , \mathcal { C } l } ,\tag{27}
$$

This is the contraction statement in Theorem 2. The assumptions are explicit: the result applies to the fixed reciprocal transport operator used for cached CGP and does not claim the same spectral rate for arbitrary time-varying edge operators. Because each $\mathcal { T } _ { u v }$ is a rotor sandwich map, it is isometric and does not collapse the orthogonal Clifford basis merely by transport.

## APPENDIX C PROOF OF AHA STABILITY AND CONSENSUS BOUND IN THEOREM 3

This section proves the AHA stability inequalities in Theorem 3. Importantly, the proof only uses the linear part of the affine Clifford projection and does not assume that the learned consensus profile is a downstream-optimal representation.

Proof. Let the affine Clifford projection used in the main paper be

$$
f _ { \mathrm { c l l } } ( \mathbf { X } ) = \mathbf { W } _ { \mathrm { o u t } } \mathbf { X } + \mathbf { b } _ { \mathrm { o u t } } ,\tag{28}
$$

and assume

$$
\| \mathbf { W _ { \mathrm { o u t } } } \| _ { 2  2 } \leq \omega .\tag{29}
$$

Define

$$
\begin{array} { r } { \mathbf { Z } _ { u } = f _ { \mathrm { c l l } } \left( \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } \right) , } \\ { \mathbf { Z } _ { u } ^ { \mathrm { r a w } } = f _ { \mathrm { c l l } } \left( \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } \mathbf { H } _ { u } ^ { ( l ) } \right) . } \end{array}\tag{30}
$$

Since the bias term cancels when two affine outputs are subtracted, we obtain

$$
\mathbf { Z } _ { u } - \mathbf { Z } _ { u } ^ { \mathrm { r a w } } = \mathbf { W } _ { \mathrm { o u t } } \sum _ { l = 0 } ^ { L } \beta _ { u , l } \left( \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { ( l ) } \right) .\tag{31}
$$

Using $\widetilde { \mathbf { H } } _ { u } ^ { ( l ) } = \boldsymbol { \alpha } _ { u } ^ { ( l ) } \odot \mathbf { H } _ { u } ^ { ( l ) }$ , the operator-norm inequality and triangle inequality yield

$$
\begin{array} { r l r } {  { \big \| \mathbf { Z } _ { u } - \mathbf { Z } _ { u } ^ { \mathrm { r a w } } \big \| _ { 2 } \leq \omega \| \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } ( \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { ( l ) } ) \| _ { \mathcal { C } l } } } \\ & { } & { \leq \omega \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| \big ( \mathbf { 1 } - \boldsymbol { \alpha } _ { u } ^ { ( l ) } \big ) \odot \mathbf { H } _ { u } ^ { ( l ) } \| _ { \mathcal { C } l } . } \end{array}\tag{32}
$$

This proves the first inequality and quantifies the representation change introduced by the soft component gate. No assumption is made that an attenuated component is noise.

For the consensus bound, because $\begin{array} { r } { \sum _ { l = 0 } ^ { L } \beta _ { u , l } = 1 } \end{array}$

$$
\begin{array} { r l } & { \mathbf { Z } _ { u } - f _ { \mathrm { c l l } } \left( \mathbf { H } _ { u } ^ { \mathrm { c t x } } \right) } \\ & { = \mathbf { W } _ { \mathrm { o u t } } \left[ \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \right] } \\ & { = \mathbf { W } _ { \mathrm { o u t } } \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } \left( \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \right) . } \end{array}\tag{33}
$$

Therefore,

$$
\begin{array} { r l r } {  { \| { \mathbf { Z } } _ { u } - f _ { \mathrm { c l l } } ( \mathbf { H } _ { u } ^ { \mathrm { c t x } } ) \| _ { 2 } \leq \omega \| \sum _ { l = 0 } ^ { L } \beta _ { u , l } ( \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } ) \| _ { \mathcal { C } l } } } \\ & { } & { \leq \omega \displaystyle \sum _ { l = 0 } ^ { L } \beta _ { u , l } \| \widetilde { \mathbf { H } } _ { u } ^ { ( l ) } - \mathbf { H } _ { u } ^ { \mathrm { c t x } } \| _ { \mathcal { C } l } . } \end{array}\tag{34}
$$

This proves the second inequality. The result is a consensus deviation bound and does not identify $\mathbf { H } _ { u } ^ { \mathrm { c t x } }$ with the unknown downstream-optimal representation. □

## APPENDIX D THEORETICAL FOUNDATIONS OF LION

The preceding analyses establish three scoped properties of LION. Theorem 1 bounds the sensitivity of the Clifford construction under bounded encoders/projections and fixed numerical stabilizers. Theorem 2 gives a conditional spectral contraction result for the same normalized, fixed reciprocal CGP operator used in the main paper. Theorem 3 bounds the representation change introduced by AHA gating and its deviation from the learned multiscale consensus. These results support stability and operator consistency; they do not assume that attenuated grades are necessarily noise, that the consensus is a downstream optimum, or that the fixed-operator spectral rate describes the downstream neural optimizer.

## APPENDIX E ALGORITHM AND COMPLEXITY DETAILS

We provide additional details for the algorithmic flow and computational complexity of LION. The data flow begins with the initialization of the Clifford geometric manifold. Given a multimodal-attributed graph ${ \mathcal { G } } ,$ the modality-oriented isomorphic lifting map projects raw scalar features X into the Grade-1 subspace of the Clifford algebra $\mathcal { C } l _ { K }$ . Assigning each modality to a unique orthogonal basis vector $e _ { k }$ preserves the independence of modality axes while unifying them in the multi-vector representation $\mathbf { H } ^ { ( 0 ) }$ . Topology modeling is then performed by computing the geometric product between connected nodes, whose Grade-0 and Grade-2 projections provide the same-modality compatibility and oriented crossmodality discrepancy statistics defined in the main paper. These edgewise statistics instantiate the topology-oriented geometric potential Φ and spatial rotor $\mathcal { R } ,$ , which jointly encode semantic curvature on the Clifford manifold. Guided by these operators, CGP conducts training-free high-order propagation through potential-gated parallel transport, producing aligned curvatureadaptive tokens $\{ \mathbf { H } ^ { ( 1 ) } , \ldots , \mathbf { \hat { H } } ^ { ( L ) } \}$

For the two-modality case, the text and visual attributes may have different raw dimensions $d _ { t }$ and $d _ { i } ;$ modality-specific projections first map both to the shared dimension $d$ before the Grade-1 lift. The $C l _ { 2 }$ storage then uses the basis slots $\{ 1 , e _ { 1 } , e _ { 2 } , e _ { 1 2 } \}$ , with the normalized text and image features assigned to $e _ { 1 }$ and $e _ { 2 }$ . The scalar and bi-vector quantities used by CGP are edgewise geometric-product statistics; they are not four ordered modality-pair channels stored one-to-one in these basis slots. Accordingly, the T-T, T-I, I-T, and I-I labels in the visualization denote diagnostic ordered-pair contributions rather than Clifford basis labels. AHA quantifies the information density of the resulting Clifford token components, applies a soft learnable gate, and fuses different propagation depths through resonance scores against the consensus profile $\mathbf { H } ^ { \mathrm { c t x } }$ The final representation is obtained by projecting the filtered and scale-weighted Clifford multi-vector back into Euclidean space.

We further detail the time-space complexity. Let $N = | \nu |$ and $M = | \mathcal { E } |$ denote the numbers of nodes and edges, L the propagation depth, and $D = 2 ^ { K } d$ the Clifford multi-vector dimension for $K$ modalities. Manifold initialization via the lifting map requires $\mathcal { O } ( N \cdot D )$ ). Constructing the geometric potential and spatial rotor requires evaluating the geometric product on each edge, leading to $\mathcal { O } ( M \cdot 2 ^ { K } \cdot D )$ . For $K > 2 ,$ the scalar component aggregates same-modality compatibility and the $\binom { K } { 2 }$ Grade-2 coefficients encode oriented pairwise cross-modality discrepancies as defined in the main paper; higher grades are available algebraically but are not empirically evaluated here. Curvature-adaptive high-order propagation costs $\mathcal { O } ( L \cdot M \cdot D )$ . Since CGP is training-free and parameterindependent, this phase is executed once and cached. During training, LION optimizes only AHA: energy-aware grade filtering and scale-aware resonance fusion both scale as $\mathcal { O } ( L \cdot N \cdot D )$ Therefore, the total computational cost can be summarized as $\mathcal { O } ( L ( M \cdot D + N \cdot 2 ^ { K } \cdot D ) )$ , while remaining linear in graph size for fixed K and L. The cached multiscale propagated features require $\mathcal { O } ( L \cdot N \cdot D )$ memory, and the learnable projection, gating, and attention parameters require $\mathcal { O } ( D ^ { 2 } )$ space. This decoupled design avoids recursive neighborhood expansion during back propagation and enables scalable minibatch training.

## APPENDIX F DATASET DETAILS

All nine benchmarks used in this paper provide text and image attributes; although the Clifford formulation is defined for K modality axes, empirical claims in this work are restricted to the K = 2 text-image setting. Dataset-task assignments follow Table I and the headers of the corresponding result tables in the main paper. Graph construction uses only benchmark relations or source metadata available independently of the downstream target split; class labels, held-out links, retrieval correspondences, and generation references are excluded from edge construction to avoid target leakage.

RedditS [45] is a social network from Reddit where nodes represent posts and edges denote threading relationships (e.g., comments and replies). Textual features are encoded from titles and body content, while visual features are extracted from embedded images. This dataset is used for node clustering in the reported graph-task comparison.

Movies [46] is sourced from Amazon’s Movies and TV category. Nodes correspond to DVD/Blu-ray products, and edges reflect consumer co-purchasing behavior. Node attributes include textual plot synopses and customer reviews, alongside visual features derived from official cover art. This dataset is used for node classification in the reported graph-task comparison.

Grocery [46] originates from Amazon’s Grocery and Gourmet Food segment. Edges indicate complementary purchasing habits derived from ”also-bought” metadata. Textual features are encoded from product titles and nutritional descriptions, while visual features are extracted from packaging images. This dataset is used for node clustering in the reported graph-task comparison.

SemArt [48] is a fine-art dataset where nodes represent paintings and edges are established based on shared metadata such as artist, period, or school. Node features include expert historical commentary and stylistic visual attributes from digital images. This dataset is used for Graph-to-Image (G2Image) in the reported modality-task comparison, with its paired textual and visual attributes retained as node modalities.

Flickr30k [49] is a canonical image-text reasoning dataset. In the OpenMAG setting, we follow the released benchmark graph relations, which are fixed before the downstream split; held-out target captions used for evaluation are not used to construct edges. This dataset is utilized for Graph-to-Text (G2Text) tasks, evaluating the model’s ability to generate descriptive captions from graph-structured visual-textual context.

Sports [46], [47] is an Amazon-based graph of athletic gear where edges denote functional complementarity. It incorporates technical specifications (text) and product images (visual) to capture design and utility. This dataset is primarily utilized for link prediction, aiming to forecast potential product associations by modeling the geometric compatibility between sports-related modalities.

Ele-fashion [46], [47] is a heterogeneous graph merging Amazon’s Electronics and Fashion categories. Nodes are connected via cross-category co-purchasing links, revealing latent consumer preferences across disparate domains. Features combine technical specs with style descriptions and product imagery. This dataset is used for modality retrieval in the reported modality-task comparison.

Cloth [46], [47] is a large-scale Amazon fashion graph linking items via visual compatibility or co-purchase history. It utilizes material compositions (text) and high-resolution model images (visual). The dataset is used for link prediction in the reported graph-task comparison, where visual and textual compatibility provides a test for modality fusion and alignment.

Goodreads [50], [51] is a book review graph where edges represent co-shelving relationships. Node features are derived from book summaries, editorial reviews, and cover art. The dataset is used for node classification in the reported graph-task comparison, requiring the synthesis of aesthetic cover cues with the semantic depth of textual summaries.

## APPENDIX G

## BASELINES DETAILS

GCN [8] utilizes a localized first-order approximation of spectral graph convolutions based on the renormalization trick. It scales linearly with the number of edges and learns hidden representations that jointly capture graph topology structure and node attribute features via a layer-wise propagation rule.

GCNII [52] extends the conventional GCN by incorporating two key techniques: initial residual connections and identity mapping. These mechanisms effectively simulate lazy random walks and are theoretically and empirically shown to alleviate the over-smoothing issue, allowing the model to stack deep architectures without performance degradation.

GAT [10] introduces masked self-attention layers to assign learnable importance scores to neighbors during aggregation. This strategy enables the model to handle anisotropic graphs by implicitly assigning varying weights to different nodes, thereby capturing complex local structural patterns without relying on a fixed, predefined graph Laplacian.

GATv2 [53] implements a dynamic graph attention mechanism that strictly improves upon the static attention of the original GAT. By modifying the order of operations in the attention scoring function, it achieves universal approximation capability, offering enhanced representation expressiveness and increased robustness against graph noise and irrelevant connections.

MMGCN [14] is a pioneering framework for micro-video recommendation that explicitly models user preferences across visual, acoustic, and textual modalities. It constructs separate modality-specific bipartite graphs and aggregates high-order connectivity within each graph before combining them through a structured fusion layer, effectively capturing the user-item interactions inherent in each sensory channel.

MGAT [15] employs a gated attention mechanism within parallel multimodal interaction graphs for recommendation. It adaptively identifies the importance of specific modalities to disentangle granular personal interests, effectively acting as a denoising filter to reduce the influence of noisy or conflicting multimodal signals during preference learning.

MLaGA [18] enables Large Language Models (LLMs) to reason over MAGs via a coherent two-stage alignment strategy. It first aligns visual and textual features with the graph structure through contrastive graph pre-training, and subsequently performs instruction tuning to seamlessly integrate graph connectivity priors into the LLM’s generative reasoning process.

GraphGPT-O [12] is a comprehensive multimodal LLM designed for joint comprehension and generation tasks on

MAGs. It addresses the challenge of encoding non-Euclidean dependencies and scalability by employing personalized PageR ank sampling and a hierarchical aligner equipped with both node-level and graph-level Q-Formers to bridge structural tokens with the LLM semantic space.

Graph4MM [11] integrates multi-hop structural information directly into the self-attention mechanism via a hop-diffused attention strategy. It utilizes a specialized MM-QFormer for principled cross-modal fusion, demonstrating that treating graph topology as a guided interaction modality significantly outperforms approaches that treat the graph merely as an auxiliary input feature.

InstructG2I [13] is a graph-conditioned diffusion model specifically designed for MAGs. It utilizes semantic neighbor sampling to construct contextual prompts and employs a Graph-QFormer to encode these prompts, offering fine-grained controllability over the generative process through a novel graph classifier-free guidance mechanism.

DMGC [27] addresses complex hybrid neighborhood patterns by explicitly disentangling the graph into cross-modality homophily-enhanced and modality-specific heterophily-aware components. It utilizes a dual-frequency fusion mechanism, which acts as coupled low-pass and high-pass filters to simultaneously capture both intra-class commonalities (smoothness) and inter-class distinctions (boundaries).

DGF [41] proposes a cross-contrastive clustering framework that utilizes a dual graph filtering scheme to systematically denoise features extracted from MAG. It employs a tri-cross contrastive objective spanning modalities, topology neighborhoods, and semantic communities to learn discriminative clustering representations robust to outliers.

MIG-GT [25] utilizes modality-independent GNNs with adaptive receptive fields to accommodate the unique propagation requirements and noise levels of distinct modalities. To complement local aggregation, it integrates a samplingbased global transformer, enabling the model to capture longrange semantic dependencies and global context that standard message passing often fails to preserve.

NTSFormer [26] introduces a self-teaching graph transformer tailored for isolated cold-start node classification. It employs a stochastic cold-start attention mask to supervise a student prediction (based solely on self-information) with a teacher prediction (which is neighbor-aware), thereby ensuring robust performance and generalization even when structural connections or modal attributes are partially missing.

UniGraph2 [28] is a cross-domain graph foundation model that integrates diverse modalities into a unified embedding space. It leverages frozen pre-trained encoders and a mixture of-experts (MoE) module for scalable alignment, followed by a universal GNN for structural aggregation, facilitating the learning of generalizable representations that transfer effectively across various downstream tasks and domains.

a) Baseline adaptation and tuning protocol.: We retain each baseline’s released backbone and modality preprocessing whenever available and evaluate it on the same dataset split and downstream metric. If a released implementation lacks the required task head, only the task head is adapted while the representation module is kept unchanged. Hyperparameters and early stopping are selected using validation data only, never the test set. Shared task-side dimensions and stopping rules are used when the original method permits them; otherwise, the authors recommended method-specific settings are retained. Graph edges follow the target-independent construction protocol described in Appendix F.

## APPENDIX H EVALUATION PROTOCOLS

To ensure a rigorous and standardized evaluation across diverse MAG learning models, we implement specific experimental configurations for three primary graph-centric downstream tasks: supervised node classification, supervised link prediction, and unsupervised node clustering. For node classification and node clustering, the learning rate is established at $5 \times 1 0 ^ { - 3 }$ with a batch size of 512 and a weight decay of $1 \times 1 0 ^ { - 5 }$ While 100 training epochs are sufficient for convergence in the node classification task, node clustering requires an extended duration of 500 epochs to stabilize the underlying self-supervised objectives. In contrast, the link prediction task utilizes an adjusted learning rate of $1 \times 1 0 ^ { - 3 }$ and a significantly larger batch size of 2048 to facilitate the efficient processing of extensive edge-pair samples. To maintain architectural consistency and ensure a fair comparison, we standardize core parameters by employing CLIP-ViT-Large-Patch14 as the default frozen feature encoder, with feature dimensions unified at 768 across all tasks. For graph-task comparisons and ablations, results are obtained from five independent runs and reported as mean±standard deviation, matching entries marked with $" \pm "$ in the main paper.

Node Classification is the fundamental supervised task in graph learning. Given a MAG, the model encodes each node into a low-dimensional embedding. These representations are subsequently processed by a projection head and a Softmax layer to generate class probabilities. Optimization is performed by minimizing the cross-entropy loss between the predicted distributions and ground-truth labels. Model performance is quantified using Accuracy (Acc) and the F1-score.

Specifically, Acc measures the proportion of correctly predicted samples over the evaluation set: $\begin{array} { r } { \mathrm { A c c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bar { \mathbb { I } ( \hat { y } _ { i } = } } \end{array}$ $y _ { i } )$ , where $N$ is the sample size, $y _ { i }$ is the ground-truth, $\hat { y } _ { i }$ is the prediction, and $\mathbb { I } ( \cdot )$ is the indicator function. F1-score is the harmonic mean of Precision and Recall, providing a robust evaluation under class imbalance common in real-world graphs: F1 = 2 (·Precision · Recall) / (Precision + Recall).

Link Prediction measures the model’s capacity to infer missing or potential edges. The model computes similarity scores between node pairs (e.g., via dot product) and assigns higher scores to true edges than to negative samples. In multimodal settings, this requires aligning structural proximity with cross-modal semantic similarity. Evaluation relies on ranking-based metrics: Mean Reciprocal Rank (MRR) and Hits@K.

Specifically, MRR assesses the model’s ability to prioritize the correct target at the top of a candidate list: MRR = $\begin{array} { r } { { \frac { 1 } { | Q | } } \sum _ { i = 1 } ^ { | Q | } { \frac { 1 } { \operatorname { r a n k } _ { i } } } } \end{array}$ , where rank denotes the rank of the first correct result for query i. Hits@K reflects the recall performance at a fixed cutoff $K ,$ indicating the practical utility of the retrieval results: Hits $\begin{array} { r } { \Theta \mathbf { K } = \frac { 1 } { | Q | } \sum _ { i = 1 } ^ { | Q | } \bar { \mathbb { I } } \big ( \mathbf { { r a n k } } _ { i } \leq K \big ) } \end{array}$

Node Clustering evaluates representation quality in an unsupervised setting, following the protocol in [27]. The model partitions nodes into semantic groups without label supervision. This involves disentangling the graph into homophilous and heterophilous views, followed by a dual-frequency fusion of filtered signals. The model is optimized via a joint objective comprising reconstruction, contrastive alignment, and clustering losses. Model performance is quantified using Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI).

Specifically, NMI quantifies the mutual dependence between predicted clusters C and ground-truth labels $Y$ independent of label permutations: $\begin{array} { l l l } { \operatorname { N M I } ( Y , C ) } & { = } & { 2 } \end{array}$ $I ( Y , C ) / \left( H ( Y ) + H ( C ) \right)$ , where $I ( \cdot )$ and $H ( \cdot )$ denote mutual information and entropy, respectively. ARI evaluates clustering similarity by considering all sample pairs and adjusting for chance grouping:

$$
\mathrm { A R I } = \frac { \sum _ { i j } { \binom { n _ { i j } } { 2 } } - \bigl [ \sum _ { i } { \binom { a _ { i } } { 2 } } \sum _ { j } { \binom { b _ { j } } { 2 } } \bigr ] / { \binom { n } { 2 } } } { \frac { 1 } { 2 } \bigl [ \sum _ { i } { \binom { a _ { i } } { 2 } } + \sum _ { j } { \binom { b _ { j } } { 2 } } \bigr ] - \bigl [ \sum _ { i } { \binom { a _ { i } } { 2 } } \sum _ { j } { \binom { b _ { j } } { 2 } } \bigr ] / { \binom { n } { 2 } } } ,\tag{35}
$$

where $n _ { i j }$ represents the overlap between ground-truth cluster i and predicted cluster $j .$

For the modality retrieval task, which evaluates the model’s capability to search for relevant instances across image and text modalities within a shared latent space, we employ contrastive learning objectives integrated with a temperature scaling factor of $\tau \ = \ 0 . 0 7$ . These retrieval models are trained for 500 epochs using a learning rate of $1 \times 1 0 ^ { - 3 }$ and a batch size of 256. To effectively mitigate over-fitting and ensure the generalizability of the learned representations, we implement an early stopping mechanism with a patience period ranging from 10 to 25 epochs. For modality generation tasks, the configurations are specifically tailored to accommodate highdimensional generative processes and complex many-to-many relationships within the graph. In the Graph-to-Text (G2Text) task, we set the learning rate to $1 \times 1 0 ^ { - 3 }$ with a weight decay of $1 \times 1 0 ^ { - 2 }$ and a batch size of 8, training the model for 15 epochs. We utilize the Self-Attention with Embeddings (SA-E) strategy to sample four multimodal neighbors and employ Graph Neural Networks (GNNs) for structural position encoding. The decoder backbone is powered by the pre-trained Facebook OPT-125M, and to ensure parameter efficiency during adaptation, we support both Prefix Tuning and Low-Rank Adaptation (LoRA) with a rank of $r = 6 4$ . In the Graph-to-Image (G2Image) task, we employ a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 16 for a duration of 20 epochs. Following the InstructG2I framework, we adopt Semantic Personalized PageRank (PPR)-based neighbor sampling, selecting between 0 and 6 informative neighbors to provide structural context. The image resolution is standardized at 256 to condition the Stable Diffusion v1.5 backbone through a Graph Classifier-Free Guidance mechanism.

To ensure experimental fairness and minimize performance variance, we standardize core architectural parameters across all task configurations. Unless otherwise specified, we employ CLIP-ViT-L/14 as the default feature encoder, with node embedding dimensions unified at 768 across all downstream tasks. All optimization processes are performed using the Adam or AdamW optimizers. For graph-task comparisons and ablations, we use five independent runs and report mean±standard deviation. Modality retrieval and generation follow the corresponding benchmark protocols; entries without $" \pm "$ in the main paper are single-run evaluations and are not presented as five-run averages.

Modality Retrieval evaluates the model’s capability to retrieve relevant instances across disparate modalities, specifically focusing on image-to-text and text-to-image retrieval within the LION. Given a query from a source modality, the model projects both the query and the candidate set from the target modality into a unified, high-dimensional latent space for direct comparison. Pairwise similarity scores are subsequently computed to rank candidates in descending order of predicted relevance. This task serves as a rigorous assessment of the robustness of cross-modal alignment and the discriminative power of the learned representations for information retrieval. To provide a comprehensive quantitative evaluation, model performance is measured using standard ranking metrics, including Mean Reciprocal Rank (MRR) and Hits@K.

Graph-to-Text (G2Text) is a generative task requiring the model to synthesize natural language descriptions conditioned on graph-structured multimodal inputs. Diverging from traditional one-to-one multimodal mappings, G2Text addresses complex many-to-many relationships where target nodes interact with diverse multimodal neighborhoods. Following the MMGL framework [54], the process involves three stages: (i) Neighbor Encoding, which maps diverse modalities into a compatible embedding space; (ii) Graph Structure Encoding, which captures topology context via GNNs or Laplacian position encodings; (iii) Integration, where structureaware signals are infused into a pre-trained LLM. This task evaluates the faithful translation of structured graph information into coherent text. Model performance is evaluated by using standard NLG metrics, including BLEU-4, ROUGE-L, and CIDEr.

Specifically, BLEU-4 evaluates lexical accuracy and fluency by calculating the geometric mean of modified n-gram precisions $( p _ { n } )$ up to length 4:

$$
{ \mathrm { B L E U - 4 } } = { \mathrm { B P } } \cdot \exp \left( \sum _ { n = 1 } ^ { 4 } w _ { n } \log p _ { n } \right) .\tag{36}
$$

ROUGE-L measures sentence-level recall based on the longest common subsequence, ensuring the output covers the comprehensive information content of the ground truth:

$$
\mathrm { R O U G E - L } = \frac { ( 1 + \beta ^ { 2 } ) R _ { l c s } P _ { l c s } } { R _ { l c s } + \beta ^ { 2 } P _ { l c s } } .\tag{37}
$$

CIDEr measures the consensus between generated captions and human references using TF-IDF weighting, emphasizing the semantic importance and distinctiveness of the generated terms:

$$
\mathrm { C I D E r } _ { n } ( c , r ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { g ^ { n } ( c ) \cdot g ^ { n } ( r _ { i } ) } { | g ^ { n } ( c ) | | g ^ { n } ( r _ { i } ) | } .\tag{38}
$$

Graph-to-Image (G2Image) focuses on synthesizing images conditioned on MAG, requiring visual content to reflect both textual prompts and complex graph-based associations. Following the InstructG2I framework [13], the workflow consists of: (i) Semantic PPR-based Sampling, which selects informative neighbors; (ii) Graph-QFormer Encoding, which transforms neighbors into graph conditioning tokens; (iii) Conditional Generation, where a latent diffusion model is guided by these tokens via a graph classifier-free guidance mechanism. This task evaluates the model’s ability to generate images consistent with the styles or categories defined by the graph structure. Model performance is quantified by using CLIP-Score and DINOv2-Score for instance-level consistency.

Specifically, CLIP-Score quantifies cross-modal semantic consistency. Building upon the textual encoder $( E _ { T } )$ and visual encoder $( E _ { I } )$ in pre-trained CLIP, this metric determines whether generated images faithfully preserve the semantic content of corresponding graph descriptions:

$$
\mathrm { C L I P - S c o r e } ( I , T ) = \operatorname* { m a x } \left( 1 0 0 \cdot \cos ( E _ { I } ( I ) , E _ { T } ( T ) ) , 0 \right) .\tag{39}
$$

DINOv2-Score assesses visual fidelity and structural consistency using feature embeddings from a pre-trained DINOv2 encoder, ensuring high perceptual quality and structural resemblance to reference samples:

$$
\mathrm { D I N O v 2 - S c o r e } ( I _ { \mathrm { g e n } } , I _ { \mathrm { r e f } } ) = \cos ( \mathrm { D I N O } ( I _ { \mathrm { g e n } } ) , \mathrm { D I N O } ( I _ { \mathrm { r e f } } ) ) .\tag{40}
$$

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, ICML, 2021.

[2] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al., “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022.

[3] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2022.

[4] H. Yan, C. Li, J. Yin, Z. Yu, W. Han, M. Li, Z. Zeng, H. Sun, and S. Wang, “When graph meets multimodal: benchmarking and meditating on multimodal attributed graph learning,” in Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD, 2025.

[5] J. Zhu, Y. Zhou, S. Qian, Z. He, T. Zhao, N. Shah, and D. Koutra, “Mosaic of modalities: A comprehensive benchmark for multimodal graph learning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, CVPR, 2025.

[6] J. Liu, D. Fan, J. Shen, C. Ji, D. Zha, and Q. Tan, “Graph-mllm: Harnessing multimodal large language models for multimodal graph learning,” arXiv preprint arXiv:2506.10282, 2025.

[7] X. Wang, Z. Zhang, L. Xiao, H. Chen, C. Ge, and W. Zhu, “Towards multi modal graph large language model,” arXiv preprint arXiv:2506.09738, 2025.

[8] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in International Conference on Learning Representations, ICLR, 2017.

[9] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” Advances in neural information processing systems, NeurIPS, 2017.

[10] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and´ Y. Bengio, “Graph attention networks,” in International Conference on Learning Representations, ICLR, 2018.

[11] X. Ning, D. Fu, T. Wei, W. Xu, and J. He, “Graph4mm: Weaving multimodal learning with structural information,” in Proceedings of the International Conference on Machine Learning, ICML, 2025.

[12] Y. Fang, B. Jin, J. Shen, S. Ding, Q. Tan, and J. Han, “Graphgpt-o: Synergistic multimodal comprehension and generation on graphs,” in Proceedings of the Computer Vision and Pattern Recognition Conference, CVPR, 2025, pp. 19 467–19 476.

[13] B. Jin, Z. Pang, B. Guo, Y.-X. Wang, J. You, and J. Han, “Instructg2i: Synthesizing images from multimodal attributed graphs,” Advances in Neural Information Processing Systems, NeurIPS, 2024.

[14] J. Hu, Y. Liu, J. Zhao, and Q. Jin, “Mmgcn: Multimodal fusion via deep graph convolution network for emotion recognition in conversation,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics, ACL, 2021.

[15] Z. Tao, Y. Wei, X. Wang, X. He, X. Huang, and T.-S. Chua, “Mgat: Multimodal graph attention network for recommendation,” Information Processing & Management, vol. 57, no. 5, p. 102277, 2020.

[16] Y. Li, Z. Tang, J. Zhuang, Z. Yang, F. Ameri, and J. Zhang, “C-mag: Cascade multimodal attributed graphs for supply chain link prediction,” in Proceedings of the 1st Workshop on ”AI for Supply Chain: Today and Future” in ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD, 2025.

[17] J. Cai, X. Wang, H. Li, Z. Zhang, and W. Zhu, “Multimodal graph neural architecture search under distribution shifts,” in Proceedings of the AAAI Conference on Artificial Intelligence, AAAI, 2024.

[18] D. Fan, Y. Fang, J. Liu, D. Difallah, and Q. Tan, “Mlaga: Multimodal large language and graph assistant,” arXiv preprint arXiv:2506.02568, 2025.

[19] J. Klicpera, A. Bojchevski, and S. Gunnemann, “Predict then propagate:¨ Graph neural networks meet personalized pagerank,” in International Conference on Learning Representations, ICLR, 2019.

[20] E. Chien, J. Peng, P. Li, and O. Milenkovic, “Adaptive universal generalized pagerank graph neural network,” in International Conference on Learning Representations, ICLR, 2021.

[21] F. Frasca, E. Rossi, D. Eynard, B. Chamberlain, M. Bronstein, and F. Monti, “Sign: Scalable inception graph neural networks,” arXiv preprint arXiv:2004.11198, 2020.

[22] W. Zhang, M. Yang, Z. Sheng, Y. Li, W. Ouyang, Y. Tao, Z. Yang, and B. Cui, “Node dependent local smoothing for scalable graph learning,” Advances in Neural Information Processing Systems, NeurIPS, 2021.

[23] X. Li, J. Ma, Z. Wu, D. Su, W. Zhang, R.-H. Li, and G. Wang, “Rethinking node-wise propagation for large-scale graph learning,” in Proceedings of the ACM Web Conference, WWW, 2024.

[24] H. Liu, N. Liao, and S. Luo, “Sigma: An efficient heterophilous graph neural network with fast global aggregation,” in IEEE International Conference on Data Engineering, ICDE, 2025.

[25] J. Hu, B. Hooi, B. He, and Y. Wei, “Modality-independent graph neural networks with global transformers for multimodal recommendation,” in Proceedings of the AAAI Conference on Artificial Intelligence, AAAI, 2025.

[26] J. Hu, Y. He, Y. Li, B. Hooi, and B. He, “Ntsformer: A self-teaching graph transformer for multimodal isolated cold-start node classification,” in Proceedings of the AAAI Conference on Artificial Intelligence, AAAI, 2025.

[27] Z. Guo, Z. Shen, X. Xie, L. Wen, and Z. Kang, “Disentangling homophily and heterophily in multimodal graph clustering,” in Proceedings of the ACM International Conference on Multimedia, MM, 2025.

[28] Y. He, Y. Sui, X. He, Y. Liu, Y. Sun, and B. Hooi, “Unigraph2: Learning a unified embedding space to bind multimodal graphs,” in Proceedings of the ACM on Web Conference, WWW, 2025.

[29] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International Conference on Machine Learning, ICML, 2023.

[30] L. Shen, Z. Chen, M. Song, Z. Su, J. Liu, X. Liu et al., “Decoupling and damping: Structurally-regularized gradient matching for multimodal graph condensation,” arXiv preprint arXiv:2511.20222, 2025.

[31] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and locally connected networks on graphs,” 2014. [Online]. Available: https://arxiv.org/abs/1312.6203

[32] M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” Advances in Neural Information Processing Systems, NeurIPS, 2016.

[33] Z. Yang, W. W. Cohen, and R. Salakhutdinov, “Revisiting semi-supervised learning with graph embeddings,” in Proceedings of the International Conference on Machine Learning, ICML, 2016, p. 40–48.

[34] O. Shchur, M. Mumme, A. Bojchevski, and S. Gunnemann, “Pitfalls¨ of graph neural network evaluation,” arXiv preprint arXiv:1811.05868, 2018.

[35] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec, “Open graph benchmark: Datasets for machine learning on graphs,” Advances in neural information processing systems, NeurIPS, vol. 33, pp. 22 118–22 133, 2020.

[36] M. Zhang and Y. Chen, “Link prediction based on graph neural networks,” Advances in neural information processing systems, NeurIPS, 2018.

[37] L. Cai, J. Li, J. Wang, and S. Ji, “Line graph neural networks for link prediction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

[38] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, “Fast unfolding of communities in large networks,” Journal of statistical mechanics: theory and experiment, vol. 2008, no. 10, p. P10008, 2008.

[39] G. Karypis and V. Kumar, “A fast and high quality multilevel scheme for partitioning irregular graphs,” SIAM Journal on scientific Computing, vol. 20, no. 1, pp. 359–392, 1998.

![](images/a12466156ef8e536fd1965acdd1a37920d46ab45fa05313c146d8310d33710e7.jpg)

[40] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum chemistry,” in International conference on machine learning. PMLR, 2017, pp. 1263–1272.

[41] H. Zheng, R. Yang, H. Wang, and J. Xu, “Cross-contrastive clustering for multimodal attributed graphs with dual graph filtering,” in Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD, 2025.

[42] D. Hestenes and G. Sobczyk, Clifford algebra to geometric calculus: a unified language for mathematics and physics. Springer Science & Business Media, 2012, vol. 5.

[43] P. Lounesto, “Clifford algebras and spinors,” in Clifford algebras and their applications in mathematical physics. Springer, 2001, pp. 25–37.

[44] L. Dorst, D. Fontijne, and S. Mann, Geometric algebra for computer science (revised edition): An object-oriented approach to geometry. Morgan Kaufmann, 2009.

[45] K. Desai, G. Kaul, Z. T. Aysola, and J. Johnson, “Redcaps: Web-curated image-text data created by the people, for the people,” in Advances in Neural Information Processing Systems, NeurIPS, Datasets and Benchmarks Track, NeurIPS DB Track, 2021.

[46] J. Ni, J. Li, and J. McAuley, “Justifying recommendations using distantly-labeled reviews and fine-grained aspects,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing, EMNLP-IJCNLP, 2019.

![](images/4e52fca324b526e528f0d80e90c893cce8db6905b8e5aae45e4d5a0cf1b6bfad.jpg)

[47] Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, “Bridging language and items for retrieval and recommendation,” arXiv preprint arXiv:2403.03952, 2024.

[48] N. Garcia and G. Vogiatzis, “How to read paintings: semantic art understanding with multi-modal retrieval,” in Proceedings ofthe European Conference on Computer Vision Workshops, ECCV, 2018.

[49] B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” in Proceedings of the IEEE International Conference on Computer Vision, ICCV, 2015.

![](images/cea1301f518db292ebe7ba25d510f9e74f0fe0ddad2ddd4302a2c27d82cd339f.jpg)

[50] M. Wan and J. McAuley, “Item recommendation on monotonic behavior chains,” in Proceedings ofthe ACM Conference on Recommender Systems, RecSys, 2018.

[51] M. Wan, R. Misra, N. Nakashole, and J. McAuley, “Fine-grained spoiler detection from large-scale review corpora,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics, ACL, 2019.

Zhengyu Wu is currently pursuing his PhD degree in Beijing Institute of Technology (BIT), Beijing, China. His research interests include Federated Graph Learning, directed graph learning, and social network analysis. He has co-authored papers published in top ML/DB/DM/AI conferences such as ICML, VLDB, WWW, AAAI.

[52] M. Chen, Z. Wei, Z. Huang, B. Ding, and Y. Li, “Simple and deep graph convolutional networks,” in International Conference on Machine Learning, ICML, 2020.

[53] S. Brody, U. Alon, and E. Yahav, “How attentive are graph attention networks?” International Conference on Learning Representations, ICLR, 2022.

Xunkai Li is currently working toward the PhD degree with the School of Computer Science, Beijing Institute of Technology. He received the BS degree in computer science from Shandong University in 2022. His research interests include data-centric graph intelligence, with a focus on graph databases and machine learning, multimodal understanding and generation, and AI for Science (AI4Science). He has published more than 20 papers in leading ML/DB/DM/AI conferences, including ICML, NeurIPS, VLDB, WWW, and AAAI.

[54] M. Yoon, J. Y. Koh, B. Hooi, and R. Salakhutdinov, “Multimodal graph learning for generative tasks,” arXiv preprint arXiv:2310.07478, 2023.

Zekai Chen is currently pursuing his Master’s degree in Computer Science at Beijing Institute of Technology under the supervision of Professor Rong-Hua Li. He obtained his Bachelor degree in Computer Science from the same institution in 2024. His research focuses on Graph Machine Learning and AI for Science (AI4Science).

![](images/feac271bf05cabbd73641bb20b23574318e7f12cdf81cbde1a06144521d32c5d.jpg)

Henan Sun received his Bachelor degree and Master degree from the College of Information Science, Northwest A&F University in 2022, and the School of Computer Science & Technology, Beijing Institute of Technology in 2024, respectively. His research interests include graph neural networks and graph out-of-distribution learning. He has published several academic papers in the ML community, such as ICDE, CEA and COMPSAC.

![](images/17a9646d0d3038f568271bea13cc950b8e4d811847d55a40e76f82ec61532478.jpg)

Daohan Su is currently working toward the master’s degree in the School of Computer Science & Technology, Beijing Institute of Technology. He received his bachelor’s degree from the same institution in 2024. His research interests lie in the areas of graph neural networks and large language models.

![](images/5bd89a056133211541724241b6774af3f311ab88f9560c9804cee01e759d61df.jpg)

Guang Zeng graduated from Northeastern University (Computer Science) in 2017 and currently works as an Algorithm Engineer at Ant Group. Specializing in time series forecasting, causal effect estimation, and graph computing, she has authored multiple academic papers and continues to explore innovative applications of cutting-edge technologies in realworld business scenarios.

![](images/667795807666677ee902b2e405b8e0f515894017e4fadff518e43a680b26e250.jpg)

![](images/b761f8634e7f640e85f1505afd8089b9ff56d4633836a76e7353fc9947d33eb3.jpg)  
WWW 2023.

Hongchao Qin is currently an Assistant Professor with Beijing Institute of Technology, China. He received the B.S. degree in mathematics, M.E. degree, and Ph.D. degree in computer science from Northeastern University, China in 2013, 2015, and 2020, respectively. His research interests include com munity detection, graph mining, knowledge graphs, financial fraud detection, vector databases, and RAG based intelligent question answering systems.

Rong-Hua Li received his PhD degree from the Department of Systems Engineering and Engineering Management, The Chinese University of Hong Kong, in 2013, and joined the School of Computer and Software, Shenzhen University this year. He joined the School of Computer Science & Technology, Beijing Institute of Technology in 2018 as a professor. His research interests include graph theory, graph computing systems, spectral graph theory, graph neural networks, and knowledge graphs. He is a committee member of VLDB 2024, KDD 2023, and

![](images/6972eb8a914e623fa04d56c188acefa141b42bf2a4162aba4792f63e0dab75ad.jpg)

Guoren Wang received his BS, MS, and PhD degrees from Northeastern University in 1988, 1991, and 1996. He joined the School of Computer Science & Technology, Beijing Institute of Technology in 2018 as a professor. He is a member of the Discipline Evaluation Group of the State Council and the Expert Review Group of the Information Science Department of the National Natural Science Foundation of China. His research interests include uncertain data management, data-intensive computing, visual media data analysis, and bioinformatics.