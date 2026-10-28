# Euclidean Crowding–Driven Adaptive Euclidean–Hyperbolic Graph Learning

Hao Chen, Jian Sun, Chuan Shi, and Hongying Zhang

Abstract—Hyperbolic graph representation learning has emerged as a powerful paradigm for modeling hierarchical data. Unlike Euclidean space, which exhibits polynomial volume growth, hyperbolic space grows exponentially with radius, naturally accommodating tree-like structures without the geometric distortion known as Euclidean crowding. Despite strong empirical performance, the theoretical mechanisms underlying hyperbolic embeddings remain poorly understood. In this work, we provide a geometric quantifcation research for Euclidean crowding, a generalization-theoretic explanation for the superiority of hyperbolic space on hierarchical data, and further develop an adaptive Euclidean-hyperbolic fusion learning framework toward bridging this gap. Specifically, we first establish a geometric quantification of Euclidean crowding by analyzing layer-wise minimum separation in node embeddings, proving that fixed-dimensional Euclidean embeddings of tree-like graphs inevitably exhibit layerwise crowding that leads to weaker margin-based generalization bounds. Second, we provide a generalization-bound for the superiority of hyperbolic geometry on hierarchical data. Third, we propose EH-Fusion, a geometry-adaptive framework that lifts Euclidean GNN representations into hyperbolic space via exponential mapping and adaptively fuses both geometries based on node-level neighborhood expansion patterns. Extensive experiments validate our theoretical insights and demonstrate the effectiveness of EH-Fusion on benchmark datasets.

Index Terms—Graph Learning, Hyperbolic Space, Hyperbolic Graph Neural Network, Generalization gap.

## I. INTRODUCTION

Euclidean geometry has long been the dominant representation space in machine learning [1]–[3], due to its strong correspondence with human spatial intuition and well-established mathematical foundations. In particular, graph representation learning has predominantly relied on Euclidean embeddings [4], [5]. However, an increasing body of evidence suggests that Euclidean geometry may not align well with the latent structures of many real-world datasets.

This mismatch is particularly evident in hierarchical or treelike data, which arise in a wide range of domains, including concept taxonomies [6], [7], fine-grained image [8], RNA-seq data [9], protein interaction data [10] and biological classification systems [11]. Such data typically exhibit exponential branching patterns. In contrast, the volume of Euclidean space grows only polynomially, leading to a fundamental geometric mismatch. As a result, Euclidean embeddings are prone to the so-called crowding phenomenon [12], where nodes at deeper layers become increasingly close to each other.

Hyperbolic geometry provides a natural alternative. Due to its exponential volume growth, hyperbolic space matches the expansion of hierarchical structures and admits lowdistortion embeddings of tree-like data [7]. In addition, prior studies [13]–[15] have suggested that hyperbolic space can also serve as a natural latent space for graphs with scalefree (power-law) structures and for heterogeneous information networks that exhibit similar power-law properties. This observation has motivated a large body of work on deep hyperbolic representation learning across computer vision [16]–[18], natural language processing [7], [19], and multi-modal learning [20]. Beyond representation learning, hyperbolic geometry has also been leveraged for graph generation. Most recently, Bu et al. [21] proposed GGBall, a hyperbolic generative framework operating in the Poincaré ball, which combines a hyperbolic autoencoder with Riemannian flow to generate hierarchical graphs in a one-shot manner. Their empirical results demonstrate that hyperbolic latent spaces significantly improve generation quality on tree-like and molecular graphs, further confirming the practical advantages of hyperbolic geometry. Despite these empirical successes, the mechanisms underlying the advantages of hyperbolic geometry and the intrinsic limitations of Euclidean geometry in modeling hierarchical data remain underexplored. Existing explanations are largely based on global distortion [22], [23] or qualitative geometric intuition [7], and do not provide a quantitative characterization of how Euclidean crowding arises or how it affects representation learning.

In this work, we provide a geometric analysis of hierarchical graph embeddings in Euclidean spaces. Rather than analyzing global metric distortion, we study the layer-wise minimum separation between node embeddings, providing a complementary perspective to the distortion-based theories of hyperbolic embeddings. Our analysis shows that, under fixed embedding dimensions, Euclidean representations of tree-like graphs inevitably exhibit layer-wise crowding, where the sep aration between nodes within the same layer shrinks as depth increases. Furthermore, we show that Euclidean hierarchical embeddings induce an intrinsic geometric tension between preserving parent–child distances and maintaining separation between non-adjacent nodes. Enforcing separation between non-adjacent nodes results in increased radial distances and stretching parent–child edges, whereas preserving tree-path distances inevitably compresses the separation between nonadjacent nodes. This phenomenon gives rise to a fundamental stretch–squeeze trade-off, revealing a structural limitation of Euclidean embeddings for tree-like graphs.

![](images/21899c51ba4c69c22c96fd61af9d629471b097bba89ab69a9a13799e3fbf5b2e.jpg)  
Fig. 1. Euclidean crowding and its connection to generalization. In hierarchical data, the exponential growth of nodes with depth compresses pairwise distances in Euclidean space, reducing the geometric separation ∆ and the induced functional margin γ. This margin reduction leads to a looser margin-based generalization bound, whereas hyperbolic space can better preserve hierarchical distances with lower distortion.

Beyond the geometric perspective, we establish a direct connection between representation geometry and learning theory. By linking the separation gap between positive and negative pairs to the functional margin of pairwise distancebased models, we show that Euclidean crowding reduces the achievable margin and consequently leads to weaker marginbased generalization guarantees. To our knowledge, this is the first work that quantitatively characterizes layer-wise crowding in Euclidean embeddings and connects it to generalization behavior.

Motivated by this observation, we first introduce a lightweight Euclidean-to-hyperbolic mapping module (GnnE2H), which lifts Euclidean graph representations into hyperbolic space, such as the Lorentz model or the Poincaré ball, via a simple exponential mapping layer. Experimental results on the strongly hyperbolic dataset Disease demonstrate that adding a single hyperbolic mapping layer is sufficient to unlock the advantages of hyperbolic geometry. Specifically, when applied to GCN, GCN2L achieves 94.9% AUC, outperforming the tangent-space approximation method HGCN (89.9%) by a substantial margin. When applied to GAT, GAT2L reaches 97.5% AUC, surpassing the fully hyperbolic network FHNN (96.0%) by 1.5 percentage points. These results not only support our theoretical analysis but also align with the principle of Occam’s razor, yielding a lightweight yet effective architecture that captures the benefits of hyperbolic geometry without requiring a fully hyperbolic network.

Moreover, recent studies have shown that real-world graphs often exhibit mixed geometric characteristics that cannot be fully described by a single manifold such as hyperbolic space [24]. This challenge reflects an underlying geometric mismatch: the advantages of hyperbolic space diminish for graphs with weaker hyperbolicity or less tree-like structures, where imposing a fixed hyperbolic space may lead to performance degradation. To address this issue, CurvGAD [25] proposes a mixed-curvature graph autoencoder that explicitly models geometric irregularities through curvature-aware encoding. In particular, it incorporates Ollivier–Ricci curvature, along with curvature reconstruction and Ricci-flow regularization to discern geometry-dependent patterns. While effective, such an explicit curvature modeling strategy relies on curvature estimation and curvature-driven reconstruction, which introduces additional computational overhead. To address the risk of geometric mismatch and implicitly capture structural properties for mixed-manifold modeling, building upon the GnnE2H mapping module, we further propose EH-Fusion, an adaptive Euclidean–hyperbolic fusion framework that leverages curvature via node-level neighborhood expansion patterns to dynamically balance Euclidean and hyperbolic geometries with low overhead and high portability.

Our method exploits the complementary roles of Euclidean and hyperbolic geometries in modeling graphs with varying degrees of hierarchy. Hyperbolic space provides a natural inductive bias for tree-like graphs through its exponential expansion property, whereas Euclidean space remains a reliable alternative when hierarchical structure is weak or less pronounced. In contrast, positively curved spherical geometry is more suitable for structures dominated by cycles or cliques, which fall outside the flat-to-hierarchical regime considered in this work. Therefore, we focus on Euclidean–hyperbolic complementarity for hierarchy-aware graph learning.

Overall, this work provides a unified geometric and learning-theoretic perspective on graph representation learning for hierarchical data. By quantifying layer-wise crowding in Euclidean embeddings for tree-like data and linking it to margin-based generalization, we offer a principled explanation for the advantages of hyperbolic representations and derive models from these theoretical insights.

Our main contributions are summarized as follows:

• Quantifying Euclidean crowding in hierarchical representations. We show that fixed-dimensional Euclidean representations inevitably suffer from layer-wise crowding when embedding tree-like graphs, by providing a quantitative geometric characterization of this phenomenon. Our analysis further reveals an intrinsic stretch–squeeze trade-off between preserving parent– child distances and maintaining separation among nonadjacent nodes.

• Linking Euclidean crowding to margin-based generalization. We establish a principled connection between Euclidean crowding and margin-based generalization by relating layer-wise separation to the functional margin of distance-based models. Our analysis shows that Euclidean crowding inherently limits the achievable margin, leading to weaker generalization guarantees.

• Geometry-Adaptive Graph Learning via Euclidean– Hyperbolic Fusion. Motivated by the above insights, we propose EH-Fusion, an adaptive Euclidean–hyperbolic framework that dynamically balances Euclidean and hyperbolic geometries according to node-level neighborhood expansion patterns, enabling robust learning across graphs with heterogeneous geometric structures.

## II. RELATED WORK

## A. Deep Hyperbolic Representation Learning

Hyperbolic geometry has recently attracted increasing attention in deep representation learning for modeling hierarchical structures. In visual representation learning, Yan et al. [16] introduced hyperbolic metrics and hierarchical clustering to capture multi-level semantic relations, addressing the difficulty of modeling latent hierarchical structures in tree-like data. In language representation learning, Nickel and Kiela [7] proposed Poincaré embeddings, demonstrating that hyperbolic geometry effectively encodes latent hierarchical structures with enhanced representational capacity and structural fidelity. In multimodal representation learning, Desai et al. [20] developed a hyperbolic image–text contrastive framework, leveraging the geometric properties of hyperbolic space to model multi-level semantic organization in vision–language representations.

A common design paradigm in these approaches is to introduce a hyperbolic realization layer on top of Euclidean backbone models to leverage the metric advantages of hyperbolic space for hierarchical representation, while the underlying architectures remain Euclidean. This paradigm suggests that introducing a simple exponential mapping layer enables deep Euclidean encoders to be effectively optimized under hyperbolic metrics, yielding high-quality hyperbolic embeddings. These observations indicate that the effectiveness of hyperbolic representations largely stems from the metric geometry itself.

## B. Hyperbolic Graph Representation Learning

Recent advances in hyperbolic graph representation learning have followed a progressively deepening geometric modeling paradigm. HGNN [26] introduced a tangent-space learning framework, where graph aggregation is performed in the Euclidean tangent space, followed by projection onto hyperbolic manifolds via exponential mappings. Building upon this idea, HGCN [27] instantiated this framework with GCNstyle architectures while preserving inductive capability. Subsequently, FHNN [28] proposed a fully hyperbolic framework, where feature transformations, neighborhood aggregation, and nonlinearities are defined entirely within hyperbolic manifolds. By performing all operations directly in hyperbolic space, FHNN achieves stronger geometric consistency and improved expressiveness for hierarchical graph structures. Overall, hyperbolic graph representation learning has evolved from tangent-space approximated GNNs to intrinsic hyperbolic graph operations, reflecting a transition from geometryassisted representations to fully geometry-consistent ones.

Taken together, existing studies reveal two distinct paradigms of hyperbolic modeling. In deep representation learning, hyperbolic geometry is introduced via a mapping layer that lifts Euclidean representations onto hyperbolic space and enables optimization under hyperbolic metrics. In graph neural networks, recent studies have explored deeper integration of hyperbolic geometry throughout the entire network architecture. This naturally raises the questions of what fundamentally drives the effectiveness of hyperbolic representations and what principles can guide our selection among different pathways. In this work, through marginbased generalization analysis and extensive experiments, we show that much of this advantage stems from the underlying metric geometry.

## C. Geometric Perspective on Euclidean Crowding

Hyperbolic geometry is widely recognized as a natural representation space for hierarchical or tree-like data. This advantage is often attributed to the exponential volume growth of hyperbolic space, which aligns with the exponential expansion of hierarchical structures. In contrast, the polynomial volume growth of Euclidean space creates a geometric mismatch, leading to the crowding phenomenon [7], [12], [29], [30] among nodes in deeper layers. Several studies have established the theoretical advantages of hyperbolic embeddings. In particular, the notion of δ-hyperbolicity introduced by Gromov characterizes the tree-likeness of graphs, where trees are 0-hyperbolic in the sense of Gromov and therefore naturally align with hyperbolic geometry [31]. Moreover, tree metrics admit embeddings into the hyperbolic plane with only O(1) distortion, whereas Euclidean embeddings typically incur O(log n) distortion [23]. Nevertheless, existing studies primarily focus on global distortion properties, while the mechanisms underlying this geometric mismatch and the resulting crowding phenomenon remain underexplored. In particular, how Euclidean crowding arises in hierarchical representations and how it affects learning performance has not been quantitatively characterized. In this work, we provide a complementary geometric perspective by analyzing the layerwise minimum separation between node embeddings. Inspired by sphere packing arguments [32], [33],which study the maximal number of points that can be placed within a sphere under a fixed minimum separation, we instead analyze the opposite scenario, where all nodes are embedded within the sphere and characterize how the minimum separation decays with tree depth. We show that fixed-dimensional Euclidean embeddings of tree-like graphs inevitably exhibit layer-wise crowding. Furthermore, by relating geometric separation to the functional margin of pairwise distance-based models, we establish that such crowding leads to weaker margin-based generalization guarantees, providing a theoretical perspective on the advantages of hyperbolic representations for hierarchical data.

## III. PRELIMINARY

## A. Riemannian Geometry

a) Manifold: An n-dimensional manifold M is a topological space that is locally homeomorphic to $\mathbb { R } ^ { n }$

b) Tangent Space: For any point $x \in \mathcal { M }$ , the tangent space $T _ { x } { \mathcal { M } }$ is the vector space of all tangent vectors at x,and is isomorphic to R<sup>n</sup>.

c) Riemannian Manifold: A Riemannian manifold $( \mathcal { M } , g )$ is a smooth manifold equipped with a Riemannian metric $g = ( g _ { x } ) _ { x \in \mathcal { M } }$ , where each $g _ { x } : T _ { x } { \mathcal { M } } \times T _ { x } { \mathcal { M } } \to { \mathbb { R } }$ is an inner product varying smoothly with x. This metric induces geometric notions such as length and curvature.

d) Geodesics and Distance: Let M be a Riemannian manifold, and let $\gamma : [ a , b ] \ \to \ { \mathcal { M } }$ be a piecewise smooth curve on the manifold. The length of the curve γ is defined as: $\begin{array} { r } { L ( \gamma ) = \int _ { a } ^ { b } \| \dot { \gamma } ( t ) \| _ { g } } \end{array}$ dt. For any two points $x , y \in { \mathcal { M } }$ the geodesic distance between them is defined as: $d _ { g } ( x , y ) =$ inf $\{ L ( \gamma ) \}$ , where $\gamma ( a ) \ = \ x \ \mathrm { a n d } \ \gamma ( b ) \ = \ y$ . The geodesic distance generalizes the notion of Euclidean distance to curved Riemannian manifolds, where a geodesic is defined as the shortest path between two points. In particular, for certain manifolds such as hyperbolic manifolds, geodesics can be expressed in a closed form.

e) Exponential and Logarithmic Maps: For each point $x \in \mathcal { M }$ and vector $v \in T _ { x } { \mathcal { M } }$ , there exists a unique geodesic $\gamma _ { v }$ such that $\gamma _ { v } ( 0 ) ~ = ~ x$ and $\dot { \gamma } _ { v } ( 0 ) ~ = ~ v$ . The Riemannian exponential map is defined as $\exp _ { x } ( v ) = \gamma _ { v } ( 1 )$ , which maps the tangent vector v to a point on the manifold M. The logarithmic map $\log _ { x } : \mathcal { M } \to T _ { x } \mathcal { M }$ is the inverse of $\exp _ { x } ,$ which pulls the points on the manifold back to the tangent space at x.

## B. Hyperbolic Manifold

A hyperbolic manifold is a Riemannian manifold with constant negative curvature, whose volume grows exponentially. This geometric property makes it an ideal embedding space for hierarchical data. Hyperbolic space is commonly represented by two isometric models, the Poincaré ball and the Lorentz model, both of which preserve its intrinsic geometry and are mutually transformable.

a) Poincaré Space: The d-dimensional Poincaré ball model with negative curvature $- c \ ( c > 0 )$ is denoted by $B _ { c } ^ { d }$ and its domain is given by: $B _ { c } ^ { d } = \big \{ x \in \overline { { \mathbb { R } } } ^ { d } : \| x \| ^ { 2 } < \overline { { 1 / c } } \big \}$ where ∥ · ∥ denotes the Euclidean norm. The origin in the Poincaré ball model corresponds to the Euclidean origin. The hyperbolic distance between two points $g _ { u } , g _ { v } \in B$ is defined as:

$$
d _ { c } ^ { \ B } ( g _ { u } , g _ { v } ) = \frac { 1 } { \sqrt { c } } \operatorname { a r c o s h } \left( 1 + \frac { 2 c \lVert g _ { u } - g _ { v } \rVert ^ { 2 } } { ( 1 - c \lVert g _ { u } \rVert ^ { 2 } ) ( 1 - c \lVert g _ { v } \rVert ^ { 2 } ) } \right)\tag{1}
$$

For $x \in \mathcal { T } _ { 0 } \mathcal { B } _ { c } ^ { d } = \mathbb { R } ^ { d }$ and $z \in B _ { c } ^ { d } .$ , the exponential and the logarithmic map between the hyperbolic space $B _ { c } ^ { d }$ and the tangent space $\mathcal { T } _ { 0 } B _ { c } ^ { d }$ is defined as :

$$
e x p _ { 0 } ^ { \mathcal { B } } ( x ) = \operatorname { t a n h } \left( \sqrt { c } \| x \| \right) \frac { x } { \sqrt { c } \| x \| }\tag{2}
$$

$$
l o g _ { 0 } ^ { \mathfrak { B } } ( z ) = \mathrm { a r t a n h } \left( \sqrt { c } \| z \| \right) \frac { z } { \sqrt { c } \| z \| }\tag{3}
$$

where the origin 0 is $( 0 , . . . , 0 ) \in B _ { c } ^ { d }$

b) Lorentz Space: The d-dimensional Lorentzian space with the negative curvature $- 1 / K \quad ( K \quad > \quad 0 )$ is denoted by $\mathcal { L } _ { K } ^ { \bar { d } }$ , and its domain is given by $\begin{array} { r l } { \mathcal { L } _ { K } ^ { d } } & { { } = } \end{array}$ $\{ x \in \mathbb { R } ^ { d + 1 } \mid \stackrel { \ast \cdot } { \langle x , x \rangle } _ { \mathcal { L } } = - K , x _ { 0 } > 0 \}$ , where the Lorentzian inner product is defined as $\begin{array} { r } { \langle x , y \rangle _ { \mathcal { L } } : = - x _ { 0 } y _ { 0 } + \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } = } \end{array}$ $- x _ { 0 } y _ { 0 } + x _ { s p a c e } \cdot y _ { s p a c e }$ . The Lorentzian distance between two points $x , y \in \mathcal { L } _ { K } ^ { d }$ is given by:

$$
d _ { K } ^ { \mathcal { L } } ( x , y ) = \sqrt { K } \mathrm { a r c o s h } ( - \langle x , y \rangle \varsigma / K ) .\tag{4}
$$

For $v \in \mathbb { R } ^ { d } , z \in \mathcal { L } _ { K } ^ { d }$ , the exponential and the logarithmic map is defined as:

$$
e x p _ { 0 } ^ { \mathcal { L } } ( v ) = \left( \sqrt { K } \cosh \left( \frac { \Vert v \Vert } { \sqrt { K } } \right) , \sqrt { K } \sinh \left( \frac { \Vert v \Vert } { \sqrt { K } } \right) \frac { v } { \Vert v \Vert } \right)
$$

$$
l o g _ { 0 } ^ { \mathcal { L } } ( z ) = \left( 0 , \sqrt { K } \mathrm { a r c o s h } \left( \frac { x _ { 0 } } { \sqrt { K } } \right) \frac { x _ { s p a c e } } { \| x _ { s p a c e } \| } \right)\tag{5}
$$

where the origin 0 is $( \sqrt { K } , 0 , . . . , 0 ) \in \mathcal { L } _ { K } ^ { d }$

(6)

## C. Metric Space

A metric space $( X , d )$ is a set X equipped with a distance function $\mathcal { I } : X \times X $ R that satisfies the following conditions for all $x , y , z \in X$

1) Non-negativity: $d ( x , y ) \geq 0 .$

2) Identity of indiscernibles: $d ( x , y ) = 0$ if and only if $x = y .$

3) Symmetry: $d ( x , y ) = d ( y , x )$

4) Triangle inequality: $d ( x , z ) \leq d ( x , y ) + d ( y , z ) .$

Euclidean hyperball space. The d-dimensional hyperball of radius r is defined as $B ^ { d } ( r ) : = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } : \| \mathbf { x } \| < r \right\}$ , which is equipped with the Euclidean $\ell _ { 2 }$ distance metric $d _ { 2 } ( { \bf x } , { \bf y } ) : =$ $\| \mathbf { x } - \mathbf { y } \| _ { 2 } , \forall \mathbf { x } , \mathbf { y } \in B ^ { d } ( r )$ . The volume of $B ^ { d } ( r )$ is given by Vol $\begin{array} { r } { \left( B ^ { d } ( r ) \right) \ = \ \frac { \pi ^ { d / 2 } r ^ { d } } { \Gamma \left( 1 + \frac { d } { 2 } \right) } } \end{array}$ , where $\Gamma ( \cdot )$ denotes the Gamma function, satisfying $\Gamma { \dot { ( n ) } } = ( n - 1 ) !$ for positive integers n.

Tree metric space. Let T be a rooted tree of depth $L \geq 1$ where each internal node has $k \geq 2$ children on average. We denote by $| \nu _ { l } |$ the number of nodes at depth l. Edges connecting nodes at depth $i - 1$ to depth i are assigned positive weights $d _ { i }$ with $d _ { 1 } \ \geq \ d _ { 2 } \ \geq \ \cdot \ \cdot \ \geq \ d _ { L } \ > \ 0$ , reflecting increasing similarity among nodes at deeper levels. We denote a node at depth k by $v _ { k , j _ { k } }$ , where $j _ { k }$ indexes its position at depth k. For any two nodes $v _ { k , j _ { k } }$ and $v _ { m , j _ { m } }$ , let $v _ { h , j _ { h } }$ denote their lowest common ancestor (LCA). Since T is a tree, there exists a unique shortest path between $v _ { k , j _ { k } }$ and $v _ { m , j _ { m } }$ that ascends from $v _ { k , j _ { k } } \ { \mathrm { t o } } \ v _ { h , j _ { h } }$ and then descends to $v _ { m , j _ { m } } \colon$ $v _ { k , j _ { k } } \to v _ { k - 1 , j _ { k - 1 } } \to \cdots \to v _ { l , j _ { l } } \to \cdots \to v _ { m - 1 , j _ { m - 1 } } \to$ $v _ { m , j _ { m } }$ . The induced tree metric $d _ { T } ( \cdot , \cdot )$ is defined as the sum of edge weights along this path: $d _ { T } \left( v _ { k , j _ { k } } , v _ { m , j _ { m } } \right) ~ =$ $\begin{array} { r } { \sum _ { i = h + 1 } ^ { k } d _ { i } ~ + ~ \sum _ { i = h + 1 } ^ { m } d _ { i } } \end{array}$ . Specifically, let u and v be two leaf nodes at depth l whose lowest common ancestor is the root. Then their tree distance is given by

$$
d _ { T } ( u , v ) = 2 \sum _ { i = 1 } ^ { l } d _ { i } .\tag{7}
$$

## D. Gromov δ-Hyperbolicity

To quantify the hierarchical structure of graph data, we adopt Gromov δ-hyperbolicity [31], [34] as a metric characterization of tree-likeness. Unlike curvature, which describes local bending behavior, δ-hyperbolicity captures how closely the global distance structure resembles a tree metric. This directly connects to our focus on hierarchical expansion and the corresponding benefits of hyperbolic representations.

a) Gromov δ-Hyperbolicity: Let $( X , d )$ be a metric space. For any four points $x , y , z , t \in X$ , define the three pairwise distance sums:

$$
\begin{array} { r } { S _ { 1 } = d ( x , y ) + d ( z , t ) , } \\ { S _ { 2 } = d ( x , z ) + d ( y , t ) , } \\ { S _ { 3 } = d ( x , t ) + d ( y , z ) . } \end{array}\tag{8}
$$

The metric space $( X , d )$ is said to be δ-hyperbolic if, for every quadruple of points, the two largest values among $\{ S _ { 1 } , S _ { 2 } , S _ { 3 } \}$ differ by at most 2δ, i.e.,

$$
\operatorname* { m a x } \{ S _ { 1 } , S _ { 2 } , S _ { 3 } \} - \operatorname* { m i d } \{ S _ { 1 } , S _ { 2 } , S _ { 3 } \} \leq 2 \delta ,\tag{9}
$$

where mid(·) denotes the second-largest value.

Intuitively, δ measures how close a metric space is to a tree metric. A smaller δ indicates stronger hierarchical organization and greater tree-likeness, favoring hyperbolic geometry as an inductive bias. Conversely, a relatively large δ suggests weaker tree-like structure, where Euclidean geometry becomes a natural complement. In particular, trees are 0-hyperbolic.

b) Why δ-Hyperbolicity Instead of Curvature: Although hyperbolic geometry is classically associated with negative curvature, curvature and Gromov δ-hyperbolicity offer different geometric perspectives. Curvature characterizes local geometric behavior, such as local bending, geodesic divergence, or neighborhood connectivity patterns. In comparison, δ-hyperbolicity provides a global geometric measure of treelikeness and hierarchical organization. Specifically, it characterizes how closely the global distance structure of a metric space resembles a tree metric, thereby reflecting the strength of hierarchical structure in graph data. The advantage of hyperbolic representations is closely related to the exponential expansion property of hyperbolic space, which enables hierarchical structures to be represented with improved separation and reduced crowding. Therefore, our analysis focuses on whether the overall graph geometry exhibits strong global hierarchy and tree-like expansion, rather than relying only on the curvature behavior of local regions. For this reason, we use

Gromov δ-hyperbolicity as a structural indicator for assessing whether graph data are likely to benefit from hyperbolic representation learning.

## E. Problem Setting and Learning Objective

We consider an undirected graph $G = ( \nu , \mathcal { E } )$ , where each node $i \in \mathcal V$ is associated with an initial Euclidean feature vector $x _ { i } ^ { 0 , E } \in \mathbb { R } ^ { d }$ , and each edge $( u , v ) \in \mathcal { E }$ represents to a direct parent–child (is-a) relation. The goal of hyperbolic graph representation learning is to learn an mapping $f : G $ $\mathbf { \bar { Z } } \in \mathbb { H } ^ { | \mathbf { \bar { \nu } } | \times d }$ that preserves the hierarchical structure of the graph.

To evaluate how well the learned embeddings preserve hierarchical structure, we employ a distance-based link prediction task, which directly examines whether parent–child pairs are embedded closer than non-adjacent pairs. The positive pair set $\mathcal { D } ^ { + } \subset \mathcal { V } \times \mathcal { V }$ with label $y _ { u v } = 1$ is given by all node pairs $( u , v ) \in \mathcal { E } ,$ , corresponding to direct parent–child relations. The negative pair set $\mathcal { D } ^ { - } \subset \mathcal { V } \times \mathcal { V }$ with label $y _ { u v } ~ = ~ - 1$ consists of non-adjacent node pairs, including sibling nodes sharing the same parent and nodes from different branches of the hierarchy. The negative pair separation is defined as $d ^ { - } = \operatorname* { i n f } _ { ( u , v ) \in \mathcal { D } ^ { - } } d ( u , v )$

Given hyperbolic embeddings $\mathbf { z } _ { u } , \mathbf { z } _ { v } \in \mathbb { H } .$ , prediction is performed based on the hyperbolic distance $d _ { \mathcal { H } } ( \mathbf { z } _ { u } , \mathbf { z } _ { v } )$ , where smaller distances correspond to higher probabilities of edge existence. To model this relationship, we employ the same Fermi–Dirac decoder as previously used to compute connection probabilities. Specifically, the scoring function is defined as

$$
s ( u , v ) = r - { \frac { d ( u , v ) } { t } } ,\tag{10}
$$

and the corresponding probability is given by

$$
\operatorname { p r o b } ( u , v ) = \sigma { \bigl ( } s ( u , v ) { \bigr ) } = { \frac { 1 } { 1 + \exp ( - s ( u , v ) ) } } .\tag{11}
$$

## IV. GEOMETRIC CROWDING AND GENERALIZATION

In this section, we develop a geometric framework that connects embedding geometry to generalization guarantees for hierarchical data. We first demonstrate that fixed-dimensional Euclidean embeddings inevitably suffer from layer-wise crowding when representing tree-structured data. This phenomenon induces a fundamental stretch–squeeze trade-off between preserving tree distances and maintaining within-layer separation. By relating geometric separation to the functional margin of pairwise distance-based models, we further reveal how Euclidean crowding weakens margin-based generalization guarantees. In contrast, hyperbolic geometry preserves separation more effectively and thus admit stronger generalization guarantees for near-tree structured data.

## A. Euclidean Crowding in Hierarchical Representations

We begin by analyzing the inherent limitations of Euclidean geometry when embedding tree-structured data. The key tension arises from the exponential growth of nodes in a tree versus the polynomial volume growth of Euclidean hyperballs. As the tree depth increases, this mismatch forces representations from different branches to become increasingly crowded.

The following theorem 1 formally characterizes this phenomenon by analyzing the layer-wise minimum separation among Euclidean embeddings of tree nodes.

Theorem 1 (Layer-wise crowding and stretch–squeeze trade-off). Let T be a rooted tree with branching factor k and let $f : T  \mathbb { R } ^ { d }$ be an embedding with $f ( \mathrm { r o o t } ) = 0 .$ . Let $\begin{array} { r } { \delta _ { l } : = \operatorname* { m i n } _ { u , v \in \mathcal { V } _ { l } } \| f ( u ) - f ( v ) \| _ { 2 } } \end{array}$ be the minimum Euclidean separation among node embeddings at depth l. For each level $i ,$ define the stretch factor $\begin{array} { r } { \alpha _ { i } : = \operatorname* { m a x } _ { ( u , v ) \in { \mathscr E } _ { i } } ^ { \bullet } \frac { \| f ( u ) - f ( v ) \| _ { 2 } } { d _ { i } } } \end{array}$ 2 and let $\begin{array} { r } { A _ { l } = \sum _ { i = 1 } ^ { l } \alpha _ { i } d _ { i } } \end{array}$ . Then there exists a constant C such that

$$
\delta _ { l } \leq C ( \sum _ { i = 1 } ^ { l } \alpha _ { i } d _ { i } ) k ^ { - l / d } .\tag{12}
$$

Consequently:

(i) If the embedding approximately preserves radial tree distances $( A _ { l } \approx T _ { l } )$ , then

$$
\delta _ { l } \leq C ( \sum _ { i = 1 } ^ { l } d _ { i } ) k ^ { - l / d } \leq C d _ { 1 } l k ^ { - l / d } .\tag{13}
$$

In particular, when ignoring the polynomial factor $O ( l )$ we obtain $\delta _ { l } = \tilde { O } ( k ^ { - l / d } )$ , indicating exponential crowding of nodes.

(ii) Conversely, maintaining a uniform layer-wise separation $\delta _ { l } \ge m > 0$ requires

$$
A _ { l } \geq \frac { m } { C } k ^ { l / d } ,\tag{14}
$$

which implies exponential radial stretching of tree edges.

Together, these results reveal a fundamental stretch–squeeze trade-off in Euclidean embeddings of hierarchical structures. 1Furthermore, the exponent 1/d indicates that higher embedding dimensions d can partially mitigate this crowding effect.

a) Proof sketch.: The proof uses a geometric packing argument.

At depth l, place a Euclidean ball of radius $\delta _ { l } / 2$ around each embedded node $f ( v )$ for $v \in \mathcal V _ { l }$ . These balls are pairwise disjoint by definition of $\delta _ { l }$

On the other hand, the cumulative stretch $\begin{array} { r } { A _ { l } = \sum _ { i = 1 } ^ { l } \alpha _ { i } d _ { i } } \end{array}$ bounds the maximal radial extent of layer-l embeddings. Therefore all balls lie inside a Euclidean ball of radius $A _ { l } + \delta _ { l } / 2$

Comparing the volumes of the disjoint small balls with that of the containing ball yields

$$
\delta _ { l } \leq C A _ { l } | \mathcal { V } _ { l } | ^ { - 1 / d } .
$$

Since $| \nu _ { l } | = k ^ { l }$ , this gives $\delta _ { l } \leq C A _ { l } k ^ { - l / d }$

If $A _ { l }$ remains proportional to the intrinsic tree distance $T _ { l }$ the separation $\delta _ { l }$ necessarily decays exponentially with depth, producing layer-wise crowding. Conversely, maintaining a constant separation requires $A _ { l }$ to grow exponentially, leading to radial stretch along tree paths. The full proof is provided in Appendix $\mathbf { A } .$

## B. From Geometric Separation to Functional Margin

Geometric separation is closely related to learning performance through its influence on the functional margin. To formalize this relationship, we consider the distance-based prediction setting introduced in Section 3.

Lemma 1 (Separation–margin relation). Consider a pairwise distance-based model with score $\begin{array} { r } { s ( u , v ) \ = \ \frac { r - d ( u , v ) } { t } , } \end{array}$ where $t \ > \ 0 .$ , and labels $y _ { u v } ~ \in ~ \{ + 1 , - 1 \}$ . Let $d ^ { + } : = \begin{array} { r l } \end{array}$ $\mathrm { s u p } _ { ( u , v ) \in { \mathcal { D } } ^ { + } } d ( u , v )$ denote the maximum positive-pair distance, and define the geometric separation gap $\Delta : = d ^ { - } - d ^ { + }$ Then the maximal achievable pairwise functional margin satisfies

$$
\operatorname* { s u p } _ { r \in \mathbb { R } } \hat { \gamma } = \frac { \Delta } { 2 t } .\tag{15}
$$

This lemma establishes a quantitative relationship between geometric separation and the maximal achievable pairwise functional margin. Reductions in the separation gap $\Delta$ are generally accompanied by a corresponding decrease in the maximal achievable margin, independent of model capacity or optimization details. Since margin-based generalization bounds typically scale inversely with the margin, geometric distortion directly leads to weaker generalization guarantees. Detailed proofs are provided in Appendix A.2.

C. Margin-Based Generalization Advantage of Hyperbolic Metrics

Tree metrics admit embeddings into hyperbolic space with arbitrarily small distortion [23], preserving geometric separation even at large depths. In contrast, Theorem 4.1 shows that fixed-dimensional Euclidean embeddings inevitably suffer from layer-wise crowding when representing exponentially expanding hierarchical structures, which compresses the separation between distant nodes. Since the functional margin is determined by the geometric separation between positive and negative pairs (Lemma 1), this crowding effect leads to a reduced attainable margin in Euclidean space. Combining margin-based Rademacher complexity bound [35] with the separation–margin relation in Lemma 1, we obtain the following result.

Theorem 2 (Hyperbolic margin-based generalization advantage). Consider representation learning on near-tree structured data with a pairwise score function $s ( u , v )$ . Let $\Delta =$ $d ^ { - } - d ^ { + }$ denote the geometric separation between negative and positive pairs, and let $\begin{array} { r } { \hat { \gamma } = \frac { \Delta } { 2 t } } \end{array}$ be the maximal achievable functional margin (Lemma 1).

Ifthe representations are learned in hyperbolic space H and Euclidean space $\mathbb { R } ^ { d }$ , respectively, then for tree-like structured data $\Delta _ { \mathbb { H } } > \Delta _ { \mathbb { R } }$ , which implies $\gamma _ { \mathbb { H } } > \gamma _ { \mathbb { R } }$ . in the margin-based generalization bound

$$
\mathcal { R } _ { 0 - 1 } ( s ) \leq \widehat { \mathcal { R } } _ { \gamma } ( s ) + \frac { 2 } { \gamma } \mathfrak { R } _ { n } ( \mathcal { F } ) + \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } }\tag{16}
$$

the hyperbolic model admits a smaller margin-dependent complexity term $\begin{array} { r } { \frac { 2 } { \gamma _ { \mathbb { H } } } \mathfrak { R } _ { n } ( \mathcal { F } ) < \frac { 2 } { \gamma _ { \mathbb { R } } } \mathfrak { R } _ { n } ( \mathcal { F } ) } \end{array}$ . Therefore, when empirical margin errors are comparable, models equipped with hyperbolic metrics admit tighter margin-based generalization guarantees than their Euclidean counterparts.

![](images/efd172d2d8f6e8e111fb4e8def050411230bf327311feeae10d6993e107ab693.jpg)  
Fig. 2. The overall framework of EH-Fusion: Upper part: A shared Euclidean graph encoder yields node representations, which are then mapped onto multiple geometric manifolds for metric learning. Lower part: High-order degree growth information is extracted to capture local geometric characteristics and guide the adaptive aggregation of manifold-specific prediction probabilities.

Theorem 2 provides a principled geometric explanation for the empirical success of hyperbolic representations: by preserving larger separation between positive and negative pairs, hyperbolic geometry enables larger functional margins and consequently stronger generalization guarantees on hierarchical data. Detailed derivations are provided in Appendix A.3.

## V. FROM THEORY TO MODEL: THE EH-FUSION FRAMEWORK

Motivated by the theoretical analysis in Section 1, we propose a lightweight framework that exploits the metric advantages of hyperbolic geometry while remaining the flexibility of Euclidean graph encoders. Unlike fully hyperbolic architectures that replace all Euclidean operations, our approach introduces a minimal geometric realization layer and an adaptive mechanism for balancing between Euclidean and hyperbolic geometries based on local structural patterns. The proposed EH-Fusion framework consists of two main components: (i) a multi-geometry embedding module that maps shared Euclidean representations onto different manifolds via lightweight exponential mapping layers, and (ii) a node-wise adaptive fusion module that dynamically balances predictions from multiple geometries according to local neighborhood expansion patterns.

Figure 2 provides an overview of the framework. Starting from a shared Euclidean graph encoder, node representations are projected into three geometric spaces: Euclidean space $\mathbb { R } ^ { d } .$ the Poincaré ball model $\mathbb { B } _ { c } ^ { d }$ , and the Lorentz model $\mathbb { L } _ { c } ^ { d }$ of hyperbolic geometry. To handle graphs with heterogeneous geometric structures, we further introduce a topology-aware selection mechanism hat uses high-order degree statistics to compute node-specific mixing weights, enabling adaptive aggregation of predictions from different manifolds.

## A. Multi-Geometry Embedding via Lightweight Exponential Mapping layers

Given a graph $G = ( \nu , \mathcal { E } )$ with node features X and adjacency matrix A, we first obtain Euclidean node representations through a standard graph neural network:

$$
\mathbf { H } = \operatorname { E n c o d e r } _ { \operatorname { E } } ( X , A ) , \qquad \mathbf { H } \in \mathbb { R } ^ { | \mathcal { V } | \times d } .\tag{17}
$$

where the Euclidean embedding H serves as a shared latent representation and Encoder<sub>E</sub>(·) can be any Euclidean GNN such as GCN or GAT. All message passing and feature aggregation are performed in Euclidean space, allowing the model to leverage mature architectural designs and stable optimization schemes. From this share representation H, we obtain embeddings in different geometric spaces via geometryconsistent mappings.

a) Euclidean Space Representation: The final Euclidean embeddings are obtained through a linear projection:

$$
\mathbf { Z } ^ { \mathrm { E } } = \mathbf { W } ^ { E } \mathbf { H } + \mathbf { b } ^ { E }\tag{18}
$$

where $\mathbf { W } ^ { E }$ and $\mathbf { b } ^ { E }$ are learnable parameters. This projection is kept separate from the hyperbolic mappings to reduce interference during optimization.

b) Poincaré Space Representation: To capture hierarchical relations, we map Euclidean representations to a Poincaré ball space via the exponential map at the origin:

$$
\mathbf { Z } ^ { \mathrm { P } } = \exp _ { 0 } ^ { \mathrm { P } } ( \mathbf { W } ^ { P } \mathbf { H } + \mathbf { b } ^ { P } ) ,\tag{19}
$$

where $\exp _ { 0 } ^ { \mathrm { P } } ( \cdot )$ denotes the exponential map of the Poincaré ball. The Poincaé ball model provides an intuitive geometric interpretation of hierarchical structures, where the radial coordinate roughly corresponds to the hierarchy level, while angular separation reflects similarity between sibling nodes.

c) Lorentz Space Representation: In parallel, we also consider the Lorentz space model of hyperbolic geometry, which offers numerically stable distance computations and is convenient for large-scale graphs:

$$
\mathbf { Z } ^ { \mathrm { L } } = \exp _ { 0 } ^ { \mathrm { L } } ( \mathbf { W } ^ { L } \mathbf { H } + \mathbf { b } ^ { L } ) ,\tag{20}
$$

where $\exp _ { 0 } ^ { \mathrm { L } } ( \cdot )$ is the exponential map defined on the Lorentz manifold.

The parameters $W ^ { E } , W ^ { P } , W ^ { L }$ and biases are learned independently for each geometry, enabling the model to learn geometry-specific representations.

## B. Node-wise Adaptive geometry Selection

Real-world graphs often exhibit mixed geometric structures: some nodes lie in regions with tree-like expansion patterns that are more naturally modeled by hyperbolic geometry, while others are better approximated by locally Euclidean geometry. Enforcing a single global geometry across the entire graph is therefore suboptimal. To address this, we propose a data-driven fusion mechanism that adaptively selects the most suitable geometry for each node based on its local neighborhood expansion pattern.

The underlying principle is rooted in the volume growth properties of the two geometries. Euclidean space exhibits polynomial volume growth, whereas hyperbolic space grows exponentially. This distinction manifests in graph-structured data as differences in how a node’s neighborhood expands with distance. Consequently, the pattern of local expansion serves as a structural signature that reveals which geometry better aligns with the node’s latent structure.

a) Extracting structural growth features: To capture this expansion behavior, we extract node-wise structural signals, including i-hop neighborhood degrees:

$$
\mathbf { S } = \left[ \operatorname { d i a g } ( A ) , \operatorname { d i a g } ( A ^ { 2 } ) , \ldots , \operatorname { d i a g } ( A ^ { k } ) \right] ,\tag{21}
$$

where $\operatorname { d i a g } ( A ^ { i } )$ denotes the number of length-i closed walks rooted at given nodes. Rapid growth in high-order degrees tend to exhibit tree-like or hierarchical structures, whereas slower growth is more consistent with locally Euclidean geometry.

b) Learning node-wise geometry mixing weights: Based on the structural growth features, we introduce a lightweight attention mechanism that assigns each node a set of geometryspecific weights. For node $v ,$ its feature vector $\mathbf { S } _ { v }$ is passed through a small multilayer perceptron $\phi ( \cdot )$ followed by a softmax normalization:

$$
\begin{array} { r } { \pmb { \alpha } _ { v } = \mathrm { S o f t m a x } \mathopen { } \mathclose \bgroup \left( \phi ( \mathbf { S } _ { v } ) \aftergroup \egroup \right) , \qquad \pmb { \alpha } _ { v } \in \mathbb { R } ^ { M } , } \end{array}\tag{22}
$$

where $M \ : = \ : 3$ corresponds to the three geometric models: Euclidean $( \mathbb { R } ^ { d } )$ , Poincaré ball $( \mathbb { B } _ { c } ^ { d } )$ , and Lorentz $( \mathbb { L } _ { c } ^ { d } ) . \ \phi ( \cdot )$ is implemented as a multilayer perceptron with two hidden layers.

c) Adaptive fusion of multi-geometry predictions: The final link probability is obtained as a convex combination of the geometry-specific probabilities:

$$
\begin{array} { r } { \mathbf { z } _ { v } = \alpha _ { v } ^ { \mathrm { E } } p _ { v } ^ { \mathrm { E } } + \alpha _ { v } ^ { \mathrm { P } } p _ { v } ^ { \mathrm { P } } + \alpha _ { v } ^ { \mathrm { L } } p _ { v } ^ { \mathrm { L } } . } \end{array}\tag{23}
$$

This formulation allows each node to emphasize the geometry that best matches its local growth pattern. Nodes in rapidly expanding, tree-like regions naturally assign higher weights to hyperbolic components, while nodes in flatter regions rely more on Euclidean representations.

## VI. EXPERIMENT

## A. Experimental setup

1) Datasets: We evaluate the proposed method on five graph learning benchmarks spanning citation networks, transportation networks, disease propagation graphs, and largescale product co-purchase graphs. These datasets exhibit varying degrees of tree-likeness, which we quantify using Gromov δ-hyperbolicity: smaller δ values indicate stronger hyperbolic (tree-like) characteristics.

a) Cora [36]: Cora is a standard citation network dataset, where nodes represent scientific publications and edges denote citation relationships. Each node is associated with a highdimensional bag-of-words feature vector.

b) Citeseer [37]: The Citeseer dataset is another classic citation network with similar characteristics to Cora. Nodes represent scientific publications belonging to a set of distinct classes, and edges correspond to citation relationships.

c) Airport [27]: The Airport dataset models an air transportation network, where nodes correspond to airports and edges represent flight routes between airports. This graph exhibits real-world topology and structural diversity.

d) Disease [27]: The Disease dataset represents a disease propagation network, where nodes denote individuals or entities and edges reflect potential transmission paths of a disease process.

e) Ele-Fashion [38]: Ele-Fashion is a node classification dataset constructed from Amazon-Fashion, where each node corresponds to a product image and edges indicate whether two products are co-purchased. The graph contains 97,766 nodes and 199,602 edges.

Together, these datasets provide a diverse evaluation spanning different scales, domains, and levels of hyperbolicity, enabling a comprehensive assessment of the proposed framework.

2) Baselines: To comprehensively evaluate the effectiveness of our method, we compare against representative graph learning approaches spanning both Euclidean and hyperbolic geometries.

a) Poincaré Embeddings (PoincaréEmb) [7]: Poincaré Embeddings learn node representations directly in hyperbolic space using the Poincaré ball model. It has been widely adopted as a foundational hyperbolic embedding baseline.

b) Graph Convolutional Network (GCN) [4]: GCN performs neighborhood aggregation through normalized adjacency-based propagation, followed by linear feature transformation. It serves as a Euclidean graph learning baseline.

c) Graph Attention Network (GAT) [5]: GAT incorporates attention mechanisms into neighborhood aggregation, allowing nodes to adaptively weight contributions from different neighbors.

d) Hyperbolic Neural Networks (HNN) [26]: HNN extends classical neural networks to hyperbolic space by reformulating basic neural operations, including linear transformations, bias addition, nonlinear activation, and optimization, under Riemannian geometry.

e) Hyperbolic Graph Convolutional Networks (HGCN) [27]: HGCN generalizes graph convolution to hyperbolic space by performing feature transformation and neighborhood aggregation through tangent-space approximation.

f) Fully Hyperbolic Neural Network(FHNN) [28]: FHNN is a fully hyperbolic graph neural network in which both feature transformations and aggregation operations are conducted entirely in lorentz hyperbolic space.

3) Implementation Details: We tune the learning rate in {0.01, 0.008, 0.005, 0.001}, the L2 regularization coefficient in $\{ 0 , 1 0 ^ { - 1 } , 5 \times 1 0 ^ { - 2 } , 1 0 ^ { - 2 } , 5 \times 1 0 ^ { - 3 } , \bar { 1 } 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 } \}$ and the dropout rate from 0.0 to 0.7 with a step size of 0.1. All hyperparameters are selected based on validation performance. For the image-attributed graph EF-Fashion, where each node is associated with visual features, we perform a grid search over 21 learning rates, ranging from 0.0001 to 0.0009, 0.001 to 0.009, and 0.01 to 0.03, and report the best-performing results. To fully exploit the visual modality, we evaluate both multimodal-aligned encoders (e.g., CLIP and ImageBind) and general-purpose visual encoders (e.g., DINO and ViT). The latent dimension dis set to {8, 16, 32, 64} for all methods. Poincaré embeddings are optimized using Riemannian stochastic gradient descent (RSGD) [39], while GCN and GAT-based models are optimized using Adam [40]. All GNN-based models adopt early stopping, where training is terminated if the validation performance does not improve for 100 consecutive epochs. All experiments are conducted on an Intel Xeon Silver 4410Y CPU (2.0–3.9 GHz) and an NVIDIA GeForce RTX 4090Ti GPU.

## B. Experimental Results

Table 1 summarizes the performance comparison across four benchmark datasets with varying δ-hyperbolicity levels, where smaller δ indicates stronger tree-likeness and more pronounced hyperbolic structure. The empirical results reveal a clear relationship between dataset hyperbolicity and geometric modeling effectiveness. On strongly hyperbolic datasets such as Disease (δ = 0) and Airport (δ = 1), introducing hyperbolic realization leads to substantial and consistent improvements over purely Euclidean baselines.

For example, On strongly hyperbolic datasets such as Disease $( \delta \ = \ 0 )$ , hyperbolic modeling brings substantial improvements over purely Euclidean baselines. For instance, Euclidean GCN achieves at most 87.1%, while Euclidean GAT reaches 89.9%. In contrast, hyperbolic variants such as FHNN achieve up to 96.0%, yielding nearly a 6 % points improvement over Euclidean GAT. Our lightweight hyperbolic extensions, GAT2P and GAT2L, further improve the performance to 97.4% and 97.5%, respectively. The proposed GAT-EH-Fusion achieves up to 98.1%, surpassing the strongest single-geometry hyperbolic baseline by approximately 1.3%. A similar trend is observed on the Airport dataset (δ = 1). Euclidean GCN achieves 95.1%, and Euclidean GAT reaches 94.1%. Hyperbolic baselines such as HGCN and FHNN improve performance to 96.2% and 96.7%, respectively. Our GAT2L achieves 96.1%, while GAT2P reaches 96.5%, again matching most hyperbolic baselines. The adaptive GAT-EH-Fusion further improves the result to 97.7%, achieving the best overall performance.

Notably, GAT2P and GAT2L introduce only a single hyperbolic exponential mapping layer on top of the original Euclidean GAT backbone, without redesigning the entire architecture into a fully hyperbolic network. Despite this minimal modification, they already achieve performance comparable to fully hyperbolic models such as HGCN and FHNN. This observation is consistent with the theoretical analysis in Section 4, which suggests that the primary advantages of hyperbolic modeling arise from the metric structure of the representation space. In particular, once node representations are realized in a geometry whose volume growth matches the exponential expansion of tree-like graphs, hierarchical separation can be better preserved under the hyperbolic distance. Therefore, even a lightweight hyperbolic realization is sufficient to effectively capture the structural characteristics of strongly hierarchical graphs. As the δ value increases (e.g., Cora with $\delta \ = \ 5$ and Citeseer with δ = 11), the relative advantage of hyperbolic modeling becomes less pronounced. On Cora, Euclidean GAT achieves 95.2%, which is already comparable to hyperbolic baselines around 95.0%. Similarly, on Citeseer, Euclidean GAT reaches 97.2%, remaining competitive with hyperbolic counterparts. In contrast to the strongly hyperbolic datasets, introducing only a single hyperbolic exponential mapping layer (e.g., GAT2P and GAT2L) brings limited improvement in these cases, as the underlying graph structure exhibits weaker hierarchical expansion and is more compatible with Euclidean geometry. This suggests that when the graph is moderately or weakly hyperbolic, Euclidean geometry already provides an effective inductive bias, and fixed hyperbolic realization may yield slight performance fluctuations.

We observe that the proposed EH-Fusion models maintain stable performance across all datasets , achieving 95.3% on Cora and 96.9% on Citeseer. Importantly, this robustness is not limited to high-dimensional settings. Even under low embedding dimensions (e.g., 8, 16, and 32), EH-Fusion consistently achieves competitive or superior performance, demonstrating strong representation efficiency. This behavior highlights the benefit of adaptive manifold selection in handling structural heterogeneity while maintaining stable performance even in low-dimensional regimes. By dynamically balancing Euclidean and hyperbolic representations according to dataset geometry, EH-Fusion achieves robust and consistently strong performance across varying levels of hyperbolicity and embedding dimensions.

TABLE I  
AUC(%) FOR LINK PREDICTION TASK.
<table><tr><td rowspan="2">Model</td><td colspan="4">Disease (δ = 0)</td><td colspan="4">Airport (δ = 1)</td><td colspan="4">Cora (δ = 11)</td></tr><tr><td colspan="4">8 16</td><td colspan="4">8 16 32</td><td colspan="2">Citeseer (δ = 5) 8 16 32 64</td><td colspan="4">8 16 32 64</td></tr><tr><td rowspan="2">poinEmb</td><td colspan="4">70.5 71.8</td><td colspan="2"></td><td colspan="2">64</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">87.8</td></tr><tr><td>HNN</td><td>94.2</td><td>75.2 94.0 94.2</td><td>78.2 94.1</td><td>96.2 93.8 94.1</td><td>96.5</td><td>96.7</td><td>96.7 94.3 94.3</td><td>81.4 80.7 95.5</td><td>80.3 96.2</td><td>80.9</td><td>86.1 92.6 92.8</td><td colspan="2">86.4 87.8 93.7</td></tr><tr><td>GCN</td><td colspan="4"></td><td></td><td colspan="2"></td><td></td><td colspan="2"></td><td colspan="2">96.1 96.3</td><td colspan="2">93.5</td></tr><tr><td>GAT</td><td>85.8 81.9</td><td>82.4 82.8</td><td>85.5 87.1</td><td>93.1 89.9 94.3</td><td>94.1 93.3</td><td>93.9 95.3</td><td>95.1</td><td>92.2 93.7 94.8 95.8</td><td>95.2 95.8</td><td colspan="2">90.6 92.6</td><td colspan="2">92.9 93.8 93.8 94.5 95.2</td></tr><tr><td>HGCN</td><td></td><td>89.3</td><td>82.2</td><td>94.4</td><td></td><td></td><td>94.1</td><td colspan="2"></td><td>96.6 97.2</td><td colspan="2">92.3</td></tr><tr><td>FHNN</td><td>89.3 96.7</td><td>96.5</td><td>89.9 89.8 96.8 96.0</td><td>97.3</td><td>95.0 96.3</td><td>95.3</td><td>96.2 96.7 96.7</td><td colspan="2">95.4 96.1 91.3 90.8</td><td>96.4 96.6 93.8 91.0 95.0</td><td colspan="2">94.6 95.0 95.0</td></tr><tr><td></td><td></td><td>95.1 95.0</td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td><td>92.5</td><td colspan="2">94.7 95.0 94.9</td></tr><tr><td>GCN2P(ours)</td><td>94.9</td><td>95.7</td><td>94.6</td><td>95.9 96.2</td><td>96.7 95.6</td><td>96.8 96.9</td><td>93.3</td><td colspan="2">93.4 93.8</td><td>92.0 92.4</td><td colspan="2">93.1 93.9</td></tr><tr><td>GCN2L(ours)</td><td>94.6</td><td>97.1</td><td>95.4 94.9</td><td></td><td></td><td>97.1</td><td>97.2</td><td>93.4 93.9 93.2</td><td colspan="2">92.0 91.9</td><td colspan="2">92.2 92.3 92.4</td></tr><tr><td>GAT2P(ours)</td><td>96.9</td><td>97.6</td><td>96.5 97.4</td><td>95.4</td><td>95.8</td><td>96.0 96.5</td><td></td><td>93.1 94.6 95.0 95.2</td><td colspan="2">92.9 94.0</td><td>94.3 94.3</td></tr><tr><td>GAT2L(ours)</td><td>97.7</td><td></td><td>97.3 97.5</td><td>95.9</td><td>95.8</td><td>96.3 96.1</td><td>93.1</td><td>93.9 94.5</td><td colspan="2">95.3 92.7 93.7</td><td>94.1 94.8</td></tr><tr><td>GCN-EH-Fusion(ours)</td><td>97.3</td><td>98.4 98.3</td><td>98.0 95.4</td><td>96.8</td><td>97.3</td><td>97.2</td><td>97.3 95.7</td><td colspan="2">95.6 95.7 95.9</td><td>93.4 93.8</td><td colspan="2">93.5 94.1</td></tr><tr><td>GAT-EH-Fusion(ours)</td><td>97.1</td><td>97.2</td><td>98.1</td><td>97.7 97.3</td><td></td><td>97.2 97.7</td><td colspan="2"></td><td>96.0 96.5 96.9 96.9</td><td colspan="2">94.3 95.0 95.4 95.3</td><td></td><td></td></tr></table>

TABLE II  
ACC(%) PERFORMANCE COMPARISON FOR IMAGE-ATTRIBUTED GRAPH.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Image Encoder</td><td colspan="4">Ele-Fashion</td></tr><tr><td>dim=8</td><td>dim=16</td><td>dim=32</td><td>dim=64</td></tr><tr><td rowspan="3">GCN</td><td>ImageBind</td><td>79.45</td><td>82.43</td><td>82.46</td><td>83.52</td></tr><tr><td>CLIP</td><td>78.73</td><td>79.53</td><td>80.91</td><td>80.57</td></tr><tr><td>DINOv2 ViT</td><td>79.50</td><td>80.13</td><td>80.07</td><td>81.84</td></tr><tr><td rowspan="3">GCN2L</td><td>ImageBind</td><td>78.35 80.34</td><td>80.46 83.08</td><td>79.80 83.30</td><td>80.81 84.29</td></tr><tr><td>CLIP</td><td>80.19</td><td>80.90</td><td>80.52</td><td>81.59</td></tr><tr><td>DINOv2</td><td>78.77</td><td>81.37</td><td>81.38</td><td>82.26</td></tr><tr><td></td><td>ViT</td><td>80.25</td><td>80.93</td><td>81.24</td><td>81.31</td></tr></table>

Table 2 reports the node classification results on the Ele-Fashion dataset under different visual encoders and embedding dimensions. We observe that the hyperbolic variant (GCN2L) generally outperforms the Euclidean GCN across different visual encoders and embedding dimensions, with consistent improvements observed in the majority of settings. These improvements are also relatively stable across different configurations. For example, with ImageBind features, GCN2L improves the performance from 79.45% to 80.34% at dim=8, from 82.43% to 83.08% at dim=16, and from 82.46% to 83.30% at dim=32. Similar trends can be observed for other encoders such as CLIP, DINOv2, and ViT. Overall, this pattern suggests that the performance gains brought by hyperbolic modeling are largely independent of the specific visual encoder. In addition, the relative advantage of GCN2L remains stable across different dimensionalities, further highlighting the robustness of hyperbolic modeling.

![](images/bac225bf348d175643abc78595b17124b09564d578a7eff0aa331f7d361f9ff9.jpg)

![](images/57b8872b4383ba1a81f8451a4423fcbc9ff3f535f8dccbf23c94368189c25afd.jpg)  
Fig. 3. Effect of the exponential mapping layer and EH-Fusion on strongly hyperbolic datasets. A single hyperbolic mapping significantly improves performance, and adaptive fusion provides further gains across embedding dimensions.

## C. analysis

TABLE III  
ABLATION STUDY OF THE HYPERBOLIC MAPPING LAYER AND EH-FUSION MODULE.
<table><tr><td></td><td>Disease</td><td>Airport</td><td>Citeseer</td><td>Cora</td></tr><tr><td>HGCN</td><td>89.9</td><td>96.2</td><td>96.6</td><td>95.0</td></tr><tr><td>GCN</td><td>87.1</td><td>95.1</td><td>95.8</td><td>93.8</td></tr><tr><td>+Exp2P</td><td>94.6</td><td>96.9</td><td>93.9</td><td>93.9</td></tr><tr><td>+Exp2L</td><td>94.9</td><td>97.2</td><td>92.0</td><td>92.4</td></tr><tr><td>+Exp+EH-Fusion</td><td>95.4</td><td>97.3</td><td>95.9</td><td>94.1</td></tr><tr><td>FHNN</td><td>96.0</td><td>96.7</td><td>95.0</td><td>94.9</td></tr><tr><td>GAT</td><td>89.9</td><td>94.1</td><td>97.2</td><td>95.2</td></tr><tr><td>+Exp2P</td><td>97.4</td><td>96.5</td><td>95.2</td><td>94.3</td></tr><tr><td>+Exp2L</td><td>97.5</td><td>96.3</td><td>95.2</td><td>94.8</td></tr><tr><td>+Exp+EH-Fusion</td><td>98.1</td><td>97.7</td><td>96.9</td><td>95.3</td></tr></table>

1) Ablation Study : To evaluate the contribution of each component, we conduct ablation experiments by progressively adding the hyperbolic exponential mapping layer and the adaptive EH-Fusion mechanism to standard Euclidean backbones (GCN and GAT). Table 2 summarizes the results.

a) Effect of Single Hyperbolic Mapping.: Adding a single exponential mapping layer to project Euclidean representations into either the Poincaré or Lorentz model consistently improves performance on strongly hyperbolic datasets. On Disease with $\delta \ : = \ : 0$ , GCN improves from 87.1% to 94.6% under the Poincaré mapping and to 94.9% under the Lorentz mapping, yielding gains of approximately 7.5 to 7.8 percentage points. Similarly, GAT increases from 79.7% to 97.4% with the Poincaré mapping and to 97.5% with the Lorentz mapping. On Airport with δ = 1, GCN rises from 95.1% to 97.2% under the Lorentz mapping, while GAT improves from 94.1% to 96.5% under the Poincaré mapping. These improvements indicate that even a lightweight hyperbolic realization can substantially enhance modeling capacity when the graph exhibits strong hierarchical structure.

As the δ value increases, the gains from fixed hyperbolic mapping become smaller and occasionally fluctuate. On Cora with $\delta \ = \ 5 .$ GCN increases marginally from 93.8% to 93.9% under the Poincaré mapping but decreases to 92.4% under the Lorentz mapping. On Citeseer with δ = 11, GCN drops from 95.8% to 92.0% under the Lorentz mapping. These observations suggest that when the dataset structure is less hierarchical, imposing a fixed hyperbolic geometry may not consistently improve performance. In addition,we show introducing the Fusion module enhances robustness across datasets. On Disease, GCN with EH-Fusion achieves 95.4%, and GAT with EH-Fusion reaches 98.1%, surpassing their single-geometry counterparts. On Airport, GAT with EH-Fusion achieves 97.7%. Importantly, on datasets with larger δ values, EH-Fusion mitigates the degradation observed under fixed hyperbolic mappings. For example, on Citeseer, GCN with EH-Fusion recovers performance to 95.9%, compared with 92.0% under the Lorentz mapping. Similarly, GAT with EH-Fusion maintains competitive performance at 96.9%.

Overall, the ablation results indicate that a single hyperbolic realization layer already provides substantial gains on strongly hierarchical graphs, and the adaptive manifold fusion further enhances stability and robustness across datasets with heterogeneous structural properties.

![](images/12be1abd047c1a56d54a734e030cbf431eb8019c16058995a79157b5f8ab7b4f.jpg)  
Fig. 4. AUC comparison of GAT in Lorentz hyperbolic space and Euclidean space on datasets with varying δ-hyperbolicity levels. Smaller δ indicates stronger hyperbolicity.

2) Effect of Gromov Hyperbolicity: Figure 4 compares the performance of GAT in hyperbolic and Euclidean spaces under different Gromov δ-hyperbolicity levels. From the perspective of increasing hyperbolicity (i.e., decreasing δ), the performance of GAT in the Lorentz hyperbolic space(GAT2L) improves consistently. As δ decreases, the AUC of the hyperbolic model steadily improves from 94.8% to 97.5%, demonstrating its increasing advantage on strongly hierarchical data. In contrast, Euclidean GAT exhibits an overall opposite trend: after a slight improvement at moderate δ, its performance deteriorates sharply from 97.2% to 89.9 % when the hierarchy becomes stronger. When viewed from the direction of decreasing hyperbolicity (i.e., increasing δ), Euclidean GAT shows steadily improved performance, while the advantage of hyperbolic embeddings correspondingly diminishes and may even vanish. This observation is consistent with our theoretical analysis. As the data becomes less hierarchical, neighborhood expansion departs from exponential growth, which alleviates the crowding effect and makes Euclidean geometry increasingly compatible with the underlying graph structure.

![](images/48936f2d5175a01c0df6992790cfff527a03852731279380eb1228be24d74899.jpg)  
Fig. 5. EH-Fusion analysis. Adaptive manifold selection mitigates geometric mismatch and stabilizes performance across datasets.

3) Impact of EH-Fusion for Avoiding Geometric Mismatch: Figure 4 compares GCN, GCN with a fixed hyperbolic exponential mapping, and GCN with the proposed EH-Fusion mechanism across datasets with different δ-hyperbolicity levels. The results reveal that fixed-geometry models may suffer from geometric mismatch when the assumed manifold does not align with the intrinsic structure of the dataset. On strongly hyperbolic data such as Disease with $\delta \ : = \ : 0 .$ , introducing a hyperbolic mapping significantly improves performance over the Euclidean baseline. However, as δ increases and the dataset becomes less tree-like, the limitation of fixed hyperbolic modeling becomes evident. On Cora and Citeseer, the performance gain from a fixed hyperbolic mapping diminishes and, in some cases, slightly decreases compared to the Euclidean baseline. This indicates that when the graph structure is closer to polynomial growth, imposing a hyperbolic metric may introduce unnecessary geometric distortion.

In contrast, EH-Fusion consistently maintains or improves performance across all datasets. By dynamically balancing Euclidean and hyperbolic representations according to structural signals, the model avoids committing to a single manifold assumption. When the data exhibits strong hierarchy, the hyperbolic component is emphasized. When the structure is flatter, Euclidean representations dominate. This adaptive mechanism effectively mitigates geometric mismatch and provides robust performance under varying levels of hyperbolicity.

4) Impact of Backbone Model: Figure 5 illustrates the performance comparison between the Euclidean backbones GCN and GAT. Overall, stronger backbone models tend to achieve better performance, both equipped with a single exponential mapping layer or the EH-Fusion module. In particular, when the dataset exhibits strong hierarchical structure (i.e., small δ-hyperbolicity), even a single hyperbolic realization is sufficient to capture the underlying hierarchical relationships. From an optimization perspective, the metric-based objective

![](images/6e36cd9cf8ae5823ab0e06910f01c2b11161048463ede8740c815d7415171a6c.jpg)  
Fig. 6. Effect of bacebone model on Exponential Mapping and EH-Fusion moudule.

encourages hierarchical separation in the representation space, and this structural inductive bias is propagated backward via standard gradient-based optimization. Consequently, when the backbone model possesses sufficient expressive capacity, introducing a single exponential mapping layer is often sufficient to exploit the hyperbolic metric and induce high-quality hierarchical representations. This provides an explanation for the different design paradigms in hyperbolic representation learning. In deep neural networks and large pretrained models, strong representational capacity often enables the backbone to produce high-quality latent features, making a single geometric realization layer sufficient to induce meaningful hierarchical representation under a hyperbolic metric. In contrast, graph neural networks are typically shallow to mitigate oversmoothing, which may limit their expressive power. Under these conditions, incorporating additional hyperbolic operations within the network can improve alignment with the target geometry and yield further performance gains.

Overall, the experimental results indicate that the effectiveness of Euclidean-to-hyperbolic (E2H) modeling is jointly influenced by the intrinsic latent geometry of the data and the expressive capacity of the baseline encoder. Stronger backbones amplify the benefits of metric-based realization, while adaptive fusion mitigates geometric mismatch and ensures consistent performance under heterogeneous hyperbolicity levels.

## D. Discussion and Future Work

This work provides a geometric and generalization-theoretic perspective on the advantage of hyperbolic representations for hierarchical data. By quantifying Euclidean crowding through layer-wise minimum separation and linking it to margin-based generalization, we explain why Euclidean embeddings struggle with tree-like structures. Furthermore, to handle graphs with mixed geometric structures, we propose EH-Fusion, a lightweight geometry-adaptive framework that balances Euclidean and hyperbolic metrics to leverage their complementary strengths, enabling improved geometric consistency.

Empirically, we observe that mixed-manifold models consistently outperform their single-manifold counterparts. Moreover, the relative advantage of each geometry is strongly governed by the intrinsic structure of the data: hyperbolic models are more effective for highly tree-like graphs with strong hyperbolicity, whereas Euclidean models become increasingly competitive as the degree of hyperbolicity decreases. These findings reinforce the connection between structural separability, geometric crowding, and representation effectiveness. Nevertheless, identifying a principled δ-hyperbolicity threshold that can reliably guide the pre-selection of embedding geometry remains an open and practically important direction.

Beyond δ-hyperbolicity, future work may explore more directly computable separability measures. To account for scale variations, one may consider a normalized variant of the separation margin, $\tilde { \Delta } = 1 - d ^ { + } / d ^ { - }$ , which provides a scale-invariant characterization of structural separability. For computational convenience, crowding in hierarchical graphs can be characterized by the ratio between the average sibling distance and the average parent–child distance, which captures intra-layer compression. This idea naturally extends to general graphs by replacing sibling nodes with k-hop intra-layer neighbors and parent nodes with predecessor nodes from the previous layer.

These geometry-aware measures not only provide analytical characterization of learned representations but also offer practical utility for downstream tasks. Among various applications, we identify graph generation as the most promising direction. Specifically, separation-based geometric measures such as $\Delta$ and its induced crowding metrics can serve multiple roles in graph generation: (1) as evaluation criteria for geometric fidelity, by comparing structural separability between real and generated graphs; (2) as regularization terms to suppress overly crowded structures and enable controllable generation of hierarchical patterns; and (3) as conditioning signals for the generation process, where a generator is trained to produce graphs whose $\Delta$ falls within a specified range, allowing finegrained control over branching patterns and hierarchical depth.

Overall, this perspective connects structural separability and local crowding quantification to geometry-aware graph generation. We hope these insights will inform the design of future graph learning methods with stronger geometric consistency.

## APPENDIX A PROOF OF THEOREM 1

Proof. We divide the proof into several steps.

a) Step 1: Construct disjoint balls around layer-l embeddings.: For each node $v _ { i } \in \mathcal { V } _ { l }$ , define the open Euclidean ball

$$
B _ { i } : = B \bigg ( f ( v _ { i } ) , \frac { \delta _ { l } } { 2 } \bigg ) = \left\{ h \in \mathbb { R } ^ { d } : \| h - f ( v _ { i } ) \| _ { 2 } < \frac { \delta _ { l } } { 2 } \right\} .\tag{24}
$$

We claim that the family $\{ B _ { i } \} _ { i = 1 } ^ { N _ { l } }$ is pairwise disjoint. Indeed, suppose for contradiction that there exists a point $h \in B _ { i } \cap B _ { j }$ for some $i \neq j .$ . Then, by the triangle inequality,

$$
\| f ( v _ { i } ) - f ( v _ { j } ) \| _ { 2 } \leq \| f ( v _ { i } ) - h \| _ { 2 } + \| h - f ( v _ { j } ) \| _ { 2 } < \delta _ { l } .\tag{25}
$$

This contradicts the definition of $\delta _ { l }$ . Hence,

$$
B _ { i } \cap B _ { j } = \emptyset \qquad { \mathrm { f o r ~ a l l ~ } } i \neq j .\tag{26}
$$

Therefore, since the balls are pairwise disjoint,

$$
\mathrm { V o l } \left( \bigcup _ { i = 1 } ^ { N _ { l } } B _ { i } \right) = \sum _ { i = 1 } ^ { N _ { l } } \mathrm { V o l } ( B _ { i } ) .\tag{27}
$$

Let

$$
\overline { { V } } _ { d } : = \mathrm { V o l } ( B ( 0 , 1 ) )
$$

denote the volume of the unit ball in $\mathbb { R } ^ { d }$ . Using the standard scaling law for Euclidean balls,

$$
\mathrm { V o l } ( B ( 0 , R ) ) = \overline { { V } } _ { d } R ^ { d } ,
$$

we have

$$
\mathrm { V o l } ( B _ { i } ) = \mathrm { V o l } \biggl ( B \biggl ( 0 , \frac { \delta _ { l } } { 2 } \biggr ) \biggr ) = \overline { { { V } } } _ { d } \biggl ( \frac { \delta _ { l } } { 2 } \biggr ) ^ { d } .\tag{28}
$$

Substituting (28) into (27) yields

$$
\mathrm { V o l } \left( \bigcup _ { i = 1 } ^ { N _ { l } } B _ { i } \right) = N _ { l } \overline { { V } } _ { d } \left( \frac { \delta _ { l } } { 2 } \right) ^ { d } .\tag{29}
$$

b) Step 2: Bound the radial extent of the l-th layer by $A _ { l } . .$ : Fix any node $v \in \mathcal { V } _ { l }$ . Let

$$
v _ { 0 } = \mathrm { r o o t } , \ v _ { 1 } , \ . . . , \ v _ { l } = v
$$

be a root-to-v path. By repeated application of the triangle inequality,

$$
\begin{array} { l } { \displaystyle \| f ( \boldsymbol { v } ) \| _ { 2 } \leq \| f ( \boldsymbol { v } _ { l } ) - f ( \boldsymbol { v } _ { 0 } ) \| _ { 2 } + \| f ( \boldsymbol { v } _ { 0 } ) \| _ { 2 } } \\ { \displaystyle \leq \sum _ { i = 1 } ^ { l } \| f ( \boldsymbol { v } _ { i } ) - f ( \boldsymbol { v } _ { i - 1 } ) \| _ { 2 } + \| f ( \boldsymbol { v } _ { 0 } ) \| _ { 2 } . } \end{array}\tag{30}
$$

Since translation does not change the internal structure, we may assume without loss of generality that the embedding of the root is the origin, $\begin{array} { r } { \mathrm { i . e . , ~ } f ( \mathrm { v } _ { 0 } ) = \mathbf { 0 } . } \end{array}$

Now, for each edge $( v _ { i - 1 } , v _ { i } ) \in \mathcal { E } _ { i }$ , the definition of $\alpha _ { i } =$ $\begin{array} { r } { \operatorname* { m a x } _ { ( u , v ) \in { \mathscr E } _ { i } } \frac { \| f ( u ) - \widecheck f ( v ) \| _ { 2 } } { d _ { i } } } \end{array}$ , implies

$$
\| f ( v _ { i } ) - f ( v _ { i - 1 } ) \| _ { 2 } \leq \alpha _ { i } d _ { i } .\tag{31}
$$

Substituting (31) into (30) gives

$$
\| f ( v ) \| _ { 2 } \leq \sum _ { i = 1 } ^ { l } \alpha _ { i } d _ { i } = A _ { l } .\tag{32}
$$

Since $v \in \mathcal V _ { l }$ was arbitrary, every embedded node in the l-th layer lies in the Euclidean ball

$$
f ( { \mathscr V } _ { l } ) \subset B ( 0 , A _ { l } ) .\tag{33}
$$

c) Step 3: Bound the union of small balls by one larger ball.: Take any $h \in B _ { i }$ . By definition of $B _ { i }$ and (32),

$$
\| h \| _ { 2 } \leq \| h - f ( v _ { i } ) \| _ { 2 } + \| f ( v _ { i } ) \| _ { 2 } < \frac { \delta _ { l } } { 2 } + A _ { l } .\tag{34}
$$

Hence,

$$
B _ { i } \subset B { \bigg ( } 0 , A _ { l } + { \frac { \delta _ { l } } { 2 } } { \bigg ) } \qquad { \mathrm { f o r ~ a l l ~ } } i .\tag{35}
$$

Therefore,

$$
\bigcup _ { i = 1 } ^ { N _ { l } } B _ { i } \subset B \left( 0 , A _ { l } + \frac { \delta _ { l } } { 2 } \right) .\tag{36}
$$

Taking volumes and using monotonicity of Lebesgue measure, we obtain

$$
\mathrm { V o l } \left( \bigcup _ { i = 1 } ^ { N _ { l } } B _ { i } \right) \le \mathrm { V o l } \left( B \left( 0 , A _ { l } + \frac { \delta _ { l } } { 2 } \right) \right) = \overline { { V } } _ { d } \left( A _ { l } + \frac { \delta _ { l } } { 2 } \right) ^ { d } .\tag{37}
$$

d) Step 4: Compare volumes.: Combining (29) and (37), we get

$$
N _ { l } \overline { { V } } _ { d } \left( \frac { \delta _ { l } } { 2 } \right) ^ { d } \leq \overline { { V } } _ { d } \left( A _ { l } + \frac { \delta _ { l } } { 2 } \right) ^ { d } .\tag{38}
$$

Canceling the positive constant $\overline { { V } } _ { d }$ gives

$$
N _ { l } \left( \frac { \delta _ { l } } { 2 } \right) ^ { d } \leq \left( A _ { l } + \frac { \delta _ { l } } { 2 } \right) ^ { d } .\tag{39}
$$

Next, we show that $\delta _ { l } \leq 2 A _ { l }$ . Indeed, since all points $f ( v _ { i } )$ lie in $B ( 0 , A _ { l } )$ , any two of them satisfy

$$
\| f ( v _ { i } ) - f ( v _ { j } ) \| _ { 2 } \leq 2 A _ { l } .\tag{40}
$$

Taking the minimum over $i \neq j$ gives

$$
\delta _ { l } \leq 2 A _ { l } .\tag{41}
$$

Substituting (41) into the right-hand side of (39), we obtain

$$
A _ { l } + \frac { \delta _ { l } } { 2 } \leq 2 A _ { l } .
$$

Therefore,

$$
N _ { l } \left( \frac { \delta _ { l } } { 2 } \right) ^ { d } \leq ( 2 A _ { l } ) ^ { d } .\tag{42}
$$

Taking the d-th root of both sides yields

$$
N _ { l } ^ { 1 / d } \frac { \delta _ { l } } { 2 } \leq 2 A _ { l } ,
$$

and hence

$$
\delta _ { l } \leq 4 A _ { l } N _ { l } ^ { - 1 / d } .\tag{43}
$$

e) Step 5: Specialize to exponential layer growth.: For the depth-l layer V , let $k _ { i }$ denote the average branching factor from depth i − 1 to depth i. Then the size of $\mathcal { V } _ { l }$ is exactly

$$
| \mathcal { V } _ { l } | = N _ { l } = \prod _ { i = 1 } ^ { l } k _ { i } .\tag{44}
$$

Defining $\begin{array} { r } { k = \left( \prod _ { i = 1 } ^ { l } k _ { i } \right) ^ { 1 / l } } \end{array}$ , we may equivalently write

$$
\left. \mathcal { V } _ { l } \right. = N _ { l } = k ^ { l } .\tag{45}
$$

Substituting this into (43) immediately gives

$$
\delta _ { l } \leq C A _ { l } k ^ { - l / d } = C ( \sum _ { i = 1 } ^ { l } \alpha _ { i } d _ { i } ) k ^ { - l / d } ,\tag{46}
$$

which proves (12).

Consequently, if the Euclidean embedding preserves the radial tree distance and the edge weights $d _ { i }$ along the tree remain unchanged, then

$$
\delta _ { l } \leq C A _ { l } k ^ { - l / d } = C \left( \sum _ { i = 1 } ^ { l } d _ { i } \right) k ^ { - l / d } = C d _ { 1 } ( l k ^ { - l / d } ) .
$$

For fixed and moderate embedding dimensions, the bound implies that $\delta _ { l }$ decreases with depth roughly at the rate $k ^ { - \bar { l } / d }$ , up to a polynomial factor in l. Moreover, the exponent $1 / d$ suggests that increasing the embedding dimension d can partially mitigate this crowding effect.

f) Step 6: Stretch–squeeze trade- $\cdot o f f . .$ : Conversely, suppose one requires that the same-layer separation remains bounded away from zero, i.e.,

$$
\delta _ { l } \geq m > 0 .
$$

Then (46) implies

$$
m \leq C A _ { l } k ^ { - l / d } ,
$$

which yields

$$
A _ { l } \geq \frac { m } { C } k ^ { l / d } .\tag{47}
$$

This reveals an inherent geometric tension in Euclidean embeddings of hierarchical graphs: maintaining sufficient separation between nodes within the same layer requires exponentially increasing radial distances, whereas constraining radial growth inevitably leads to crowding within layers. This tension constitutes the stated stretch–squeeze trade-off. □

## APPENDIX B PROOF OF LEMMA 1

Proof. Following standard margin-based learning theory, the functional margin is defined on the real-valued score $s ( u , v ) \in$ R rather than its sigmoid-transformed probability. The sigmoid transformation maps real-valued scores into the bounded interval (0, 1), which compresses large score values and distorts the relative magnitudes of margins.

Recall that the functional margin is defined as

$$
\hat { \gamma } : = \operatorname* { m i n } _ { ( u , v ) \in \mathcal { D } ^ { + } \cup \mathcal { D } ^ { - } } y _ { u v } s ( u , v ) , \qquad s ( u , v ) = \frac { r - d ( u , v ) } { t } ,\tag{48}
$$

where $y _ { u v } = + 1$ for $( u , v ) \in \mathcal { D } ^ { + }$ and $y _ { u v } = - 1$ for $( u , v ) \in$ $\mathcal { D } ^ { - }$

For any positive pair $( u , v ) \in \mathcal { D } ^ { + }$ , by definition of $d _ { + }$ we have $d ( u , v ) \leq d _ { + }$ , and therefore

$$
y _ { u v } s ( u , v ) = \frac { r - d ( u , v ) } { t } \geq \frac { r - d ^ { + } } { t } .\tag{49}
$$

Similarly, for any negative pair $( u , v ) \in \mathcal { D } ^ { - }$ , by definition of $d _ { - }$ we have $d ( u , v ) \geq d _ { - }$ , which implies

$$
y _ { u v } s ( u , v ) = - { \frac { r - d ( u , v ) } { t } } = { \frac { d ( u , v ) - r } { t } } \geq { \frac { d ^ { - } - r } { t } } .\tag{50}
$$

Combining the two cases yields

$$
\hat { \gamma } ( r ) = \operatorname* { m i n } \left\{ \frac { r - d ^ { + } } { t } , \frac { d ^ { - } - r } { t } \right\} .\tag{51}
$$

The right-hand side is a concave, piecewise-linear function of $r ,$ whose maximum is attained when the two terms are equal, namely at $\begin{array} { r } { r ^ { \star } = \frac { d _ { - } + d _ { + } } { 2 } } \end{array}$

Substituting $r ^ { \star }$ into the bound above gives

$$
\operatorname* { s u p } _ { r \in \mathbb { R } } \hat { \gamma } = \hat { \gamma } ( r ^ { \star } ) = \frac { d _ { - } - d _ { + } } { 2 t } = \frac { \Delta } { 2 t } ,\tag{52}
$$

where $\Delta : = d _ { - } - d _ { + }$ denotes the geometric separation between negative and positive pairs.

## APPENDIX C PROOF OF THEOREM 2

Proof. Following Bartlett and Mendelson’s Rademachercomplexity framework [35], we obtain a margin-based generalization bound by applying a margin surrogate $\phi _ { \gamma }$ (the ramp loss) to the pairwise signed scores $y _ { u v } s ( u , v )$ . The ramp loss is 1/γ-Lipschitz and dominates the 0–1 loss, i.e., $\phi ( z ) \geq \mathbf { 1 } ( z \leq 0 )$ , making it a valid surrogate for margin-based generalization analysis.

![](images/8e358a2414e0b0ef6485aa4fab733db767494e87c317775e62b51596d9d7cd7d.jpg)  
Fig. 7. Ramp Loss vs. 0–1 Loss. The ramp loss serves as a 1/γ-Lipschitz surrogate that upper-bounds the 0–1 loss. The parameter γ controls the slope of the loss, while $\gamma _ { 0 }$ denotes the functional margin.

We start from Theorem 7 of [35], which states the following. Let $\mathcal { F }$ be a class of real-valued functions defined on datalabel pairs, and let $\phi : \mathbb { R }  [ 0 , 1 ]$ satisfy (i) $\phi ( \alpha ) \geq { \bf 1 } [ \alpha \leq 0 ]$ for all α, and (ii) ϕ is Lipschitz with constant L. Then, with probability at least $1 - \delta ,$

$$
\begin{array} { r } { \mathbb { P } ( y _ { u v } f ( u , v ) \le 0 ) \le \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi \big ( y _ { u _ { i } v _ { i } } f ( u _ { i } , v _ { i } ) \big ) } \\ { + 2 L \Re _ { n } ( \mathcal { F } ) + \sqrt { \displaystyle \frac { \log ( 2 / \delta ) } { 2 n } } . } \end{array}\tag{53}
$$

Here, $\mathbb { P } ( y _ { u v } f ( u , v ) \le 0 )$ denotes the population pairwise misclassification probability, i.e., the true generalization error.

Fix any $\gamma > 0$ and define the ramp surrogate

$$
\phi _ { \gamma } ( \alpha ) = \left\{ \begin{array} { l l } { 1 , } & { \alpha \le 0 , } \\ { 1 - \alpha / \gamma , } & { 0 < \alpha \le \gamma , } \\ { 0 , } & { \alpha > \gamma . } \end{array} \right.
$$

By construction, $\phi _ { \gamma } ( \alpha ) \geq \mathbf { 1 } ( \alpha \leq 0 )$ for all α. Moreover, $\phi _ { \gamma }$ is $1 / \gamma { - } \mathrm { L }$ ipschitz and satisfies the assumptions of Theorem (53) with $L = 1 / \gamma$

Let $s ( u , v )$ be a pairwise score and define the function class

$$
{ \mathcal { F } } : = \{ ( u , v ) \mapsto y _ { u v } s ( u , v ) \} .
$$

Applying Theorem (53) to $f ( u , v ) ~ = ~ s ( u , v )$ and $\phi \ : = \ : \phi _ { \gamma }$ yields, with probability at least $1 - \delta ,$

$$
\mathcal { R } _ { 0 - 1 } ( s ) \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi _ { \gamma } \big ( y _ { u _ { i } v _ { i } } s ( u _ { i } , v _ { i } ) \big ) + \frac { 2 } { \gamma } \Re _ { n } ( \mathcal { F } ) + \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } ,\tag{54}
$$

where $\mathcal { R } _ { 0 - 1 } ( s ) = \mathbb { P } \big ( y _ { u v } s ( u , v ) \le 0 \big )$ denotes the generalization error (population 0–1 risk) of the pairwise classifier.

For each sample $( u _ { i } , v _ { i } )$ , we have $\begin{array} { r l } { \phi _ { \gamma } \left( y _ { u _ { i } v _ { i } } s ( u _ { i } , v _ { i } ) \right) } & { { } \geq } \end{array}$ $\mathbf { 1 } \big ( y _ { u _ { i } v _ { i } } s ( u _ { i } , v _ { i } ) \ \leq \ \gamma \big )$ , since $\phi _ { \gamma }$ is nonzero only when the functional margin does not exceed $\gamma .$ Averaging over samples yields

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi _ { \gamma } \big ( y _ { u _ { i } v _ { i } } s ( u _ { i } , v _ { i } ) \big ) \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \big ( y _ { u _ { i } v _ { i } } s ( u _ { i } , v _ { i } ) \leq \gamma \big ) = \widehat { \mathcal { R } } _ { \gamma } ( s ) .\tag{55}
$$

Substituting this inequality into (54) yields

$$
\mathcal { R } _ { 0 - 1 } ( s ) \ : \le \ : \widehat { \mathcal { R } } _ { \gamma } ( s ) \ : + \ : \frac { 2 } { \gamma } \Re _ { n } ( \mathcal { F } ) \ : + \ : \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{56}
$$

By Lemma 1, the maximal achievable pairwise functional margin satisfies

$$
\hat { \gamma } : = \operatorname* { s u p } _ { r \in \mathbb { R } } \operatorname* { m i n } _ { ( u , v ) } y _ { u v } s ( u , v ) = \frac { d ^ { - } - d ^ { + } } { 2 t } = \frac { \Delta } { 2 t } .\tag{57}
$$

The inner minimum gives the tightest margin guarantee for a fixed r, while the outer supremum optimizes over r to obtain the largest achievable functional margin.

For near-tree structured data, hyperbolic metrics admit lowdistortion representations of hierarchical distances with nearisometric guarantees [23], [41], whereas theorem 4.1 show that Euclidean representations suffer from exponential crowding with depth, which forces negative pairs to become progressively closer in the ambient space. Consequently, there exist

$$
\Delta _ { \mathbb { R } } \ < \ \Delta _ { \mathbb { H } } .\tag{58}
$$

Combining (57) and (58) yields a strict margin advantage:

$$
\hat { \gamma } _ { \mathbb { R } } = \frac { \Delta _ { \mathbb { R } } } { 2 t } \ < \ \frac { \Delta _ { \mathbb { H } } } { 2 t } = \hat { \gamma } _ { \mathbb { H } } .\tag{59}
$$

Since the complexity penalty in (56) scales as $O ( 1 / \gamma )$ the larger margin achieved in hyperbolic space can lead to tighter margin-based generalization guarantees than those in Euclidean space.

[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[3] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[4] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in International Conference on Learning Representations, 2017.

[5] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Liò, and´ Y. Bengio, “Graph attention networks,” in International Conference on Learning Representations, 2018.

[6] M. Le, S. Roller, L. Papaxanthos, D. Kiela, and M. Nickel, “Inferring concept hierarchies from text corpora via hyperbolic embeddings,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, A. Korhonen, D. Traum, and L. Màrquez, Eds. Florence, Italy: Association for Computational Linguistics, Jul. 2019, pp. 3231–3241.

[7] M. Nickel and D. Kiela, “Poincaré embeddings for learning hierarchical representations,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.

[8] Z. Yu, H. Lai, and Y. Liu, “Classify fine-grained images in hyperbolic space,” Pattern Recognition, p. 113348, 2026.

[9] A. Klimovskaia, D. Lopez-Paz, L. Bottou, and M. Nickel, “Poincaré maps for analyzing complex hierarchies in single-cell data,” Nature communications, vol. 11, no. 1, p. 2966, 2020.

[10] M. Zitnik, R. Sosic, M. W. Feldman, and J. Leskovec, “Evolution ofˇ resilience in protein interactomes across the tree of life,” Proceedings of the National Academy of Sciences, vol. 116, no. 10, pp. 4426–4433, 2019.

[11] D. Pogány and P. Antal, “Towards explainable interaction prediction: Embedding biological hierarchies into hyperbolic interaction space,” Plos one, vol. 19, no. 3, p. e0300906, 2024.

[12] S. Shahid, T. Anand, N. Srikanth, S. Bhatia, B. Krishnamurthy, and N. Puri, “HyHTM: Hyperbolic geometry-based hierarchical topic model,” in Findings of the Association for Computational Linguistics: ACL 2023, A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 11 672–11 688.

[13] Y. Zhang, X. Wang, C. Shi, N. Liu, and G. Song, “Lorentzian graph convolutional networks,” in Proceedings of the web conference 2021, 2021, pp. 1249–1261.

[14] Y. Zhang, X. Wang, C. Shi, X. Jiang, and Y. Ye, “Hyperbolic graph attention network,” IEEE Transactions on Big Data, vol. 8, no. 6, pp. 1690–1701, 2022.

[15] Y. Zhang, X. Wang, N. Liu, and C. Shi, “Embedding heterogeneous information network in hyperbolic spaces,” ACM Trans. Knowl. Discov. Data, vol. 16, no. 2, Sep. 2021.

[16] J. Yan, L. Luo, C. Deng, and H. Huang, “Unsupervised hyperbolic metric learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 12 465–12 474.

[17] S. Ge, S. Mishra, S. Kornblith, C.-L. Li, and D. Jacobs, “Hyperbolic contrastive learning for visual representations beyond objects,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 6840–6849.

[18] M. G. Atigh, J. Schoep, E. Acar, N. Van Noord, and P. Mettes, “Hyperbolic image segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4453– 4462.

[19] B. Chen, Y. Fu, G. Xu, P. Xie, C. Tan, M. Chen, and L. Jing, “Probing {bert} in hyperbolic spaces,” in International Conference on Learning Representations, 2021.

[20] K. Desai, M. Nickel, T. Rajpurohit, J. Johnson, and S. R. Vedantam, “Hyperbolic image-text representations,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of

Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 7694–7731.

[21] T. Bu, C. Wang, H. Ma, H. Zheng, X. Lu, and T. Wu, “GGBall: Graph generative model on poincaré ball,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=4zRRnDscqn

[22] A. Kratsios, R. Hong, and H. S. de Ocáriz Borde, “Capacity bounds for hyperbolic neural network representations of latent tree structures,” Neural Networks, vol. 178, p. 106420, 2024.

[23] R. Sarkar, “Low distortion delaunay embedding of trees in hyperbolic plane,” in International symposium on graph drawing. Springer, 2011, pp. 355–366.

[24] W. Peng, T. Varanka, A. Mostafa, H. Shi, and G. Zhao, “Hyperbolic deep neural networks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 12, pp. 10 023–10 044, 2022.

[25] K. Grover, G. J. Gordon, and C. Faloutsos, “CurvGAD: Leveraging curvature for enhanced graph anomaly detection,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu, Eds., vol. 267. PMLR, 13–19 Jul 2025, pp. 20 429–20 447.

[26] Q. Liu, M. Nickel, and D. Kiela, “Hyperbolic graph neural networks,” Advances in neural information processing systems, vol. 32, 2019.

[27] I. Chami, Z. Ying, C. Ré, and J. Leskovec, “Hyperbolic graph convolutional neural networks,” Advances in neural information processing systems, vol. 32, 2019.

[28] W. Chen, X. Han, Y. Lin, H. Zhao, Z. Liu, P. Li, M. Sun, and J. Zhou, “Fully hyperbolic neural networks,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 5672–5686.

[29] M. Yang, M. Zhou, L. Pan, and I. King, “κhgcn: Tree-likeness modeling via continuous and discrete curvature learning,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 2965–2977.

[30] S. Liu, Z. Yang, S. Liu, R. Liang, J. Sun, Q. Li, and X. Shen, “Hyperbolic embedding of discrete evolution graphs for intelligent tutoring systems,” Expert Systems with Applications, vol. 241, p. 122451, 2024.

[31] M. Gromov, Hyperbolic Groups. New York, NY: Springer New York, 1987, pp. 75–263.

[32] G. A. Kabatiansky and V. I. Levenshtein, “On bounds for packings on a sphere and in space,” Problemy peredachi informatsii, vol. 14, no. 1, pp. 3–25, 1978.

[33] H. Cohn and Y. Zhao, “Sphere packing bounds via spherical codes,” Duke Mathematical Journal, vol. 163, no. 10, pp. 1965 – 2002, 2014.

[34] A. B. Adcock, B. D. Sullivan, and M. W. Mahoney, “Tree-like structure in large social and information networks,” in 2013 IEEE 13th International Conference on Data Mining, 2013, pp. 1–10.

[35] P. L. Bartlett and S. Mendelson, “Rademacher and gaussian complexities: Risk bounds and structural results,” Journal of machine learning research, vol. 3, no. Nov, pp. 463–482, 2002.

[36] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad, “Collective classification in network data,” AI magazine, vol. 29, no. 3, pp. 93–93, 2008.

[37] C. L. Giles, K. D. Bollacker, and S. Lawrence, “Citeseer: An automatic citation indexing system,” in Proceedings of the third ACM conference on Digital libraries, 1998, pp. 89–98.

[38] J. Zhu, Y. Zhou, S. Qian, Z. He, T. Zhao, N. Shah, and D. Koutra, “Mosaic of modalities: A comprehensive benchmark for multimodal graph learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 14 215–14 224.

[39] S. Bonnabel, “Stochastic gradient descent on riemannian manifolds,” IEEE Transactions on Automatic Control, vol. 58, no. 9, pp. 2217–2229, 2013.

[40] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[41] F. Sala, C. De Sa, A. Gu, and C. Re, “Representation tradeoffs for hyperbolic embeddings,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, J. Dy and A. Krause, Eds., vol. 80. PMLR, 10–15 Jul 2018, pp. 4460–4469.