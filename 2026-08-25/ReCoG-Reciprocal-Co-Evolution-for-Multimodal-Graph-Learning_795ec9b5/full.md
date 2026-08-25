# ReCoG: Reciprocal Co-Evolution for Multimodal Graph Learning

Rui Xue & Tianfu Wu Rui Xue & Tianfu Wu

Department of Electrical and Computer Engineering (ECE) North Carolina State University, Raleigh, NC 27695, USA {rxue,twu19}@ncsu.edu

## Abstract

Multimodal graph learning requires jointly training over graph structure and heterogeneous node attributes, yet existing methods largely decouple these processes: prior multimodal graph neural networks (GNNs) focus on aligning modalities in a shared embedding space while operating on fixed or weakly adapted graph structures, and graph structure learning approaches infer topology from unimodal node representations without accounting for multimodal interactions. This separation fundamentally limits the ability of GNNs to capture semantically meaningful relationships in multimodal settings, where observed edges are often noisy, incomplete, or misaligned with underlying semantics. We propose ReCoG (Reciprocal Co-Evolution for Multimodal Graph Learning), a new learning paradigm that tightly couples graph structure learning and multimodal representation learning through end-to-end reciprocal interaction. Concretely, ReCoG integrates (i) a multimodal graph refiner that infers and corrects edges using cross-modal semantic evidence, and (ii) a coupled cross-modal message passing mechanism that performs joint intra- and inter-modality propagation over the refined graph. This unified design yields greater expressiveness than decoupled or two-stage formulations and allows dynamic interaction between topology and representation learning. Across diverse benchmarks for node classification and link prediction, ReCoG consistently outperforms strong multimodal graph structure learning baselines, including graph foundation models. Our results demonstrate that reciprocal co-evolution of structure and semantics is important for effective multimodal graph learning, challenging the prevailing separation between topology and representation learning.

## 1 Introduction

Graph neural networks (GNNs) have become a dominant paradigm for learning over relational data, achieving strong performance across applications such as recommendation systems, social networks, and scientific modeling [22, 21, 5, 27, 35, 13, 6, 23, 25, 29, 30]. While most existing GNNs assume unimodal node attributes, real-world graphs are inherently multimodal, where nodes are described by heterogeneous signals such as text, images, and metadata. These modalities provide complementary semantic views, making multimodal graph learning an important step toward more realistic and expressive representation learning.

Despite recent progress, existing approaches to multimodal graph learning remain fundamentally limited by a decoupled design. On one hand, multimodal GNNs typically focus on aligning representations across modalities—often through separate encoders or late fusion—while operating on fixed or weakly adapted graph structures. On the other hand, prior work on graph structure learning (e.g., IDGL [2], SLAPS [4], Pro-GNN [12], NodeFormer [26]) iteratively refines topology jointly with representations, but operates in the unimodal setting and does not model cross-modal evidence in either the refinement signal or the propagation rule. As a result, these methods fail to capture the tight interplay between structure and semantics that is essential in multimodal settings, where observed edges are often noisy, incomplete, or misaligned with multimodal semantic relationships.

![](images/8b702a8923e24ab8979a448cf742529ce6fd63562b2816dd6f569b599bf98fb4.jpg)  
Figure 1: Overview of the proposed ReCoG framework. Left: The input multimodal graph with text and vision features. Center-top: The Structure Refiner module with two variants: a similarity-based version, and a residual MLP-based version. Center-bottom: Cross-modal coupled message passing over L layers, where each layer performs intra-modal neighbor aggregation and cross-modal injection. Right: Modality fusion followed by a task-specific head. The entire pipeline is trained end-to-end. Best viewed in magnification. See text for detail.

We argue that graph structure and multimodal representations should not be treated as independent components, but rather as mutually dependent variables that must be learned jointly. In multimodal graphs, better representations provide stronger semantic signals for identifying meaningful connections, while improved graph structure enables more effective cross-modal information propagation.

To realize this idea, we propose ReCoG (Reciprocal Co-Evolution for Multimodal Graph Learning, Fig. 1), a unified framework that tightly couples graph structure learning and multimodal representation learning through end-to-end reciprocal interaction. ReCoG consists of two interdependent components:

• A multimodal graph refiner leverages cross-modal semantic signals to denoise and augment the observed topology, correcting spurious edges and recovering missing but semantically meaningful connections (Fig. 2 and Appendix F).

• A cross-modal message passing mechanism performs joint intra- and inter-modality propagation over the evolving graph, enabling fine-grained interaction between modalities at every layer.

Importantly, these components are optimized jointly: updated representations iteratively guide topology refinement, while the refined graph directly shapes subsequent cross-modal propagation.

We evaluate ReCoG on nine multimodal graph benchmarks spanning node classification and link prediction tasks. Across diverse encoder configurations, ReCoG consistently outperforms strong multimodal, and graph foundation model baselines, often by significant margins. These results demonstrate that jointly evolving graph structure and multimodal semantics is important for effective multimodal graph learning.

Our contributions. We provide details of related work positioning our proposed ReCoG in Appendix B. It makes three main contributions as follows:

• A unified multimodal graph learning framework. We propose ReCoG, which integrates a multimodal graph refiner with a coupled cross-modal message passing mechanism, enabling iterative interaction between structure and semantics throughout training, and challenging the conventional separation between topology and feature learning.

• Spectral and contraction analysis of coupled propagation. We give a supergraph reformulation of the coupled update that admits a closed-form spectral characterization of the modality coupling coefficients, and prove a depth-wise contraction bound that motivates our shared-weight design.

• Strong empirical performance. We demonstrate consistent improvements over state-of-theart multimodal GNNs and graph structure learning methods across nine benchmarks for node classification and link prediction.

## 2 Methodology

We now describe the three components of ReCoG (Fig. 1): (1) a multimodal graph refiner that refines the observed topology (Sec. 2.1), (2) a cross-modal message passing module that propagates information over the refined graph (Sec. 2.2), and (3) an adaptive modality fusion layer that integrates modality-specific representations (Sec. 2.3). These components are optimized end-to-end, forming a closed loop between topology refinement and representation learning.

## 2.1 Graph Refinement via Multimodal Semantics

We refine the observed topology using multimodal semantic evidence that adapts dynamically as representations evolve. We first state the refinement problem in general, then present two complementary instantiations: a cosine similarity refiner that is efficient and interpretable, and a residual refiner that provides greater expressive power.

Problem formulation. Given the multimodal graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with adjacency $\mathbf { A } _ { \mathrm { o r i g } }$ and multimodal node features $\{ \mathbf { x } _ { i } ^ { t } , \mathbf { x } _ { i } ^ { v } \} _ { i \in \mathcal { V } }$ , the refiner produces a refined adjacency $\mathbf { A } _ { \mathrm { r e f i n e } } \in [ 0 , \mathbf { \bar { 1 } } ] ^ { | \mathcal { V } | \times | \mathcal { V } | }$ by combining the original topology with a learned multimodal edge score.

Mini-batch target-centric candidates. Large-scale GNN training typically uses neighbor sampling. Let $\gamma _ { B }$ denote the sampled subgraph and $\lceil B \subseteq \gamma _ { B }$ the set of target nodes, with $| B | = { \bar { B } }$ and $| \nu _ { B } | = N _ { B }$ . Since only target nodes receive direct supervision [31], we restrict refinement to target-centric pairs:

$$
{ \mathcal { P } } = \left\{ ( i , j ) : i \in { \mathcal { B } } , \ j \in \mathcal { V } _ { \mathcal { B } } \setminus \{ i \} \right\} = { \mathcal { S } } _ { \mathcal { E } } \cup { \mathcal { S } } _ { \mathrm { c a n d } } ,\tag{1}
$$

where $\mathit { S } _ { \mathcal { E } } \subset \mathcal { E }$ contains original edges to be re-scored (denoising) and $S _ { \mathrm { c a n d } } \left( S _ { \mathrm { c a n d } } \cap { \mathcal { E } } = \emptyset \right)$ contains candidate non-edges to be discovered (recovery). This reduces candidate complexity from $O ( N _ { B } ^ { 2 } )$ to $O ( B N _ { B } )$ We provide a full analysis in Appendix G.

## 2.1.1 Cosine Similarity Refiner

This variant provides a simple and interpretable mechanism based on pretrained feature geometry.

Multimodal similarity scoring. We compute modality-specific cosine similarities and combine them via a learnable convex combination to obtain $s ( i , j )$

$$
s ( i , j ) = \lambda S _ { V } ( i , j ) + ( 1 - \lambda ) S _ { T } ( i , j )
$$

$$
S _ { T } ( i , j ) = \cos ( { \bf x } _ { i } ^ { t } , { \bf x } _ { j } ^ { t } ) , S _ { V } ( i , j ) = \cos ( { \bf x } _ { i } ^ { v } , { \bf x } _ { j } ^ { v } )\tag{2}
$$

where $\lambda = \sigma ( \lambda _ { \mathrm { l o g i t } } ) \in ( 0 , 1 )$ is learned. Textual similarity captures contextual alignment, visual similarity captures perceptual resemblance, and λ lets the model balance the two on a per-task basis. We mask self-loops and clamp negative similarities to zero to prevent anti-semantic edges.

Semantic neighbor selection. For each target node $i \in { \cal B }$ , we select the top-k neighbors by similarity:

$$
\mathcal { N } _ { \mathrm { s e m } } ( i ) = \mathrm { T o p K } _ { j \in \mathcal { V } _ { \mathcal { B } } } \{ s ( i , j ) \} , \qquad | \mathcal { N } _ { \mathrm { s e m } } ( i ) | = k .\tag{3}
$$

where k is a hyperparameter in training. This produces a set of candidate semantic edges, each associated with a similarity score. And we set $s ( \ r _ { i } , j ) = 0 , j \notin \mathcal { N } _ { \mathrm { s e m } } ( i )$

Topology integration. We combine original and semantic edges via a convex combination:

$$
\mathbf { A } _ { \mathrm { r e f i n e } } ( i , j ) = \mathrm { C l a m p } \Big ( \alpha \cdot \mathbf { A } _ { \mathrm { o r i g } } ( i , j ) + ( 1 - \alpha ) \cdot s ( i , j ) , 0 , 1 \Big ) , \forall ( i , j ) \in \mathcal { P } ,\tag{4}
$$

where $\alpha = \sigma ( \alpha _ { \mathrm { l o g i t } } ) \in ( 0 , 1 )$ is a learnable coefficient controlling the trade-off between observed topology and semantic edges. This design preserves the original graph as a structural prior while introducing new edges only when supported by strong multimodal similarity. However, because $s ( i , j ) \geq 0$ , this rule can only add or strengthen edges relative to $\mathbf { A } _ { \mathrm { o r i g } }$

## 2.1.2 Residual Refiner

The cosine similarity refiner relies on a fixed functional form and collapses feature interactions into a scalar. To capture richer, task-dependent relationships, we introduce a more expressive residual refinement mechanism.

Pairwise feature and residual scoring. For each pair $( i , j )$ , we construct a pairwise feature that preserves dimension-wise interaction patterns:

$$
\begin{array} { r } { \varphi ( i , j ) = [ \mathbf { x } _ { i } ^ { t } , \mathbf { x } _ { j } ^ { t } , \mathbf { x } _ { i } ^ { v } , \mathbf { x } _ { j } ^ { v } , \mathbf { x } _ { i } ^ { t } \odot \mathbf { x } _ { j } ^ { t } , \mathbf { x } _ { i } ^ { v } \odot \mathbf { x } _ { j } ^ { v } , | \mathbf { x } _ { i } ^ { t } - \mathbf { x } _ { j } ^ { t } | , | \mathbf { x } _ { i } ^ { v } - \mathbf { x } _ { j } ^ { v } | ] , } \end{array}\tag{5}
$$

combining individual node features, multiplicative interactions, and modality-specific differences rather than collapsing them into a scalar. We pass the pairwise feature through a learned MLP scorer $f _ { \theta }$ to produce a residual edge score $r ( i , j ) \in ( - 1 , 1 )$ , which indicates the strength and direction of the predicted topological correction:

$$
r ( i , j ) = \operatorname { t a n h } \big ( f _ { \theta } ( \varphi ( i , j ) ) \big ) .\tag{6}
$$

A positive $r ( i , j )$ encourages edge addition or strengthening, while a negative value signals edge removal or weakening.

Coarse-to-fine candidate selection. The complexity $O ( B N _ { B } )$ can be still expensive when $N _ { B }$ is large in conjuction with the high-dim of $\varphi ( i , \bar { j } )$ . We therefore adopt a coarse-to-fine strategy: (i) Coarse filtering. We project each node into a low-dimensional embedding: $\mathbf { h } _ { i } = \mathbf { W } _ { c } [ \mathbf { x } _ { i } ^ { \bar { t } } ; \mathbf { x } _ { i } ^ { v } ]$ where $\mathbf { W } _ { c } \in \mathbb { R } ^ { d _ { h } \times ( d _ { t } + d _ { v } ) }$ and $d _ { h } \ll d _ { t } + d _ { v }$ . For each target node $i \in { \cal B } _ { : }$ , we compute the cosine similarity $\tilde { S } ( i , j ) = \cos ( { \bf h } _ { i } , { \bf h } _ { j } )$ over all $j \in \mathcal { V } _ { B }$ and retain the top-c candidates, where $c = \eta k$ with overselection ratio $\eta > 1$

(ii) Fine scoring. We apply the MLP scorer to candidate pairs $( i , j )$ with $j \in \mathcal { C } ( i )$ and select the final top-k neighbors. Formally:

$$
\mathbf { C o a r s e : } \quad \mathscr { C } ( i ) = \mathrm { T o p K } _ { j \in \mathscr { V } _ { \ B { B } } } \tilde { S } ( i , j ) , \quad \mathbf { F i n e : } \quad \mathscr { N } _ { \mathrm { s e m } } ( i ) = \mathrm { T o p K } _ { j \in \mathscr { C } ( i ) } r ( i , j ) .\tag{7}
$$

Topology integration. We then integrate the MLP-based scores into the original topology as a residual signal:

$$
\mathbf { A } _ { \mathrm { r e f i n e } } ( i , j ) = \mathrm { C l a m p } \big ( \mathbf { A } _ { \mathrm { o r i g } } ( i , j ) + \delta \cdot r ( i , j ) , 0 , 1 \big ) ,\tag{8}
$$

where $\delta = \sigma ( \delta _ { \mathrm { l o g i t } } ) \in ( 0 , 1 )$ is a learnable scalar that controls the refinement magnitude. Because $r ( i , j )$ is signed, this rule supports both edge denoising and semantic edge recovery — a strict capability gain over Eq. (4). Proposition 1 shows that the family of $\mathbf { A } _ { \mathrm { r e f i n e } }$ functions induced by Eq. (8) strictly contains those induced by Eq. (4).

Proposition 1 (Strict Generalization of Residual Refinement). On afixed candidate set ${ \mathcal { C } } \subseteq V \times V ,$ let $\mathcal { F } _ { \mathrm { s i m } }$ denote the family of edge-scoring functions induced by the similarity refiner (Eqn. (2)), and $\mathcal { F } _ { \mathrm { r e s } }$ denote thefamily induced by the residual refiner (Eq. (6)). Assume the similarity refiner’s correction magnitude is bounded by δ on C. Then $\mathcal { F } _ { \mathrm { s i m } } \subsetneq \mathcal { F } _ { \mathrm { r e s } } .$

Remark 1. Proposition 1 characterizes expressiveness, not generalization. The inclusion follows from a universal-approximation argument on the residual MLP, and strictness from dimension-wise interactions $( e . g . , \mathbf { x } _ { i } ^ { t } \odot \mathbf { x } _ { j } ^ { t } )$ that cosine similarity collapses to a scalar. Whether this additional capacity translates to better task performance depends on the learning context: Sec. 4.3 show that the residual refiner’s advantage emerges only when coupled with cross-modal propagation.

## 2.1.3 Effects of Graph Refinement

To validate the effect of topology refinement, we compare the semantic similarity distribution of edges before and after refinement. As shown in Fig. 2, the original graph exhibits substantial overlap between connected and disconnected node pairs, indicating noisy and missing edges. In contrast, the edges introduced by the refiner are concentrated in the high-similarity region, with significantly reduced overlap. This demonstrates that the refiner selectively recovers semantically meaningful connec-

![](images/59f414f81430c064d825c382afd3bbb01daa67d581e020b0423e3a2a38aa010b.jpg)

![](images/3d50863517cc63f25ad5677ac73e27775f158bc942d4872887051fa728b99511.jpg)  
Figure 2: Topology–semantics alignment before and after refinement. Left: distribution of multimodal cosine similarity for connected and disconnected node pairs in the original graph (RedditM). Right: similarity distribution of edges added by the cosine refiner compared to disconnected pairs.

tions while avoiding spurious links, leading to a cleaner and more informative graph structure.

This shift in distribution provides direct evidence that ReCoG improves topology quality by increasing semantic consistency of graph edges. More detailed analysis is provided in Appendix F.

## 2.2 Semantic Alignment via Cross-Modal Message Passing

Given the refined graph $\mathbf { A } _ { \mathrm { r e f i n e } }$ , we propagate information jointly across modalities and graph structure. We first present the propagation rule directly through its supergraph formulation, which exposes its spectral and alignment properties (Sec. 3), and then show that it admits an efficient per-modality implementation.

We first project modality-specific features into a common latent space of dimension $d _ { h } \colon$

$$
\mathbf { h } _ { i } ^ { m , ( 0 ) } = \operatorname { P r o j } ( \mathbf { x } _ { i } ^ { m } ; \boldsymbol { \phi } ^ { m } ) , \quad m \in \{ t , v \} ,\tag{9}
$$

where $\phi ^ { m }$ parametrizes the projection (a linear layer followed by layer normalization (LN) and the GELU nonlinearity with dropout). This ensures dimensional alignment and stabilizes subsequent cross-modal interaction. We then adopt symmetric normalization of $\mathbf { A } _ { \mathrm { r e f i n e } }$ with self-loops.

Coupled propagation on a modality-augmented supergraph. We define the stacked representation $\mathbf { H } ^ { ( \ell ) } = [ \bar { \mathbf { H } } ^ { t , ( \bar { \ell } ) } ; \mathbf { \bar { H } } ^ { v , ( \ell ) } ]$ and the supergraph adjacency

$$
\mathbf { A } _ { \mathrm { s u p } } = \left[ \begin{array} { c c } { \mathbf { A } _ { \mathrm { r e f i n e } } } & { \gamma \mathbf { I } + \rho \mathbf { A } _ { \mathrm { r e f i n e } } } \\ { \gamma \mathbf { I } + \rho \mathbf { A } _ { \mathrm { r e f i n e } } } & { \mathbf { A } _ { \mathrm { r e f i n e } } } \end{array} \right] ,\tag{10}
$$

where $\gamma , \rho \in ( 0 , 1 )$ are learnable coupling scalars (parameterized via sigmoid). Propagation then takes the standard form

$$
\mathbf { H } ^ { ( \ell + 1 ) } = \sigma \big ( \mathbf { A } _ { \mathrm { s u p } } \mathbf { H } ^ { ( \ell ) } \mathbf { W } ^ { ( \ell ) } \big ) ,\tag{11}
$$

with a single shared weight matrix $\mathbf { W } ^ { ( \ell ) }$ acting on both modality blocks. Two design choices are central: (1) The symmetric block construction(identical diagonals and off-diagonals) places the two modalities on equal footing and enables decomposition into symmetric and anti-symmetric eigenmodes (Corollary 1). (2) Sharing $\mathbf { W } ^ { ( \ell ) }$ across modalities is what enables the cross-modal contraction guarantee in Theorem 2; a separate-weight variant would introduce a residual term proportional to $\lVert \mathbf { W } _ { T } ^ { ( \ell ) } - \mathbf { W } _ { V } ^ { ( \ell ) } \rVert$ that does not vanish even when the two modalities have aligned.

Per-modality view. Expanding the block product in Eq. (11) recovers an explicit per-modality update. For the textual stream:

$$
\mathbf { h } _ { i } ^ { t , ( \ell + 1 ) } = \sigma \Bigg ( \mathbf { W } ^ { ( \ell ) } \cdot \Big ( \underbrace { \sum _ { j \in N ( i ) } A _ { i j } \mathbf { h } _ { j } ^ { t , ( \ell ) } } _ { \mathrm { i n t r a - m o d a l a g e r e g a t i o n } } + \underbrace { \gamma \cdot \mathbf { h } _ { i } ^ { v , ( \ell ) } } _ { \mathrm { c r o s s - m o d a l s e l f i - i n j e c t i o n } } + \underbrace { \rho \cdot \sum _ { j \in N ( i ) } A _ { i j } \mathbf { h } _ { j } ^ { v , ( \ell ) } } _ { \mathrm { c r o s s - m o d a l n e i g h b o r a g e r e g a t i o n } } \Big ) \Bigg ) ,\tag{12}
$$

where $A _ { i j } = \mathbf { A } _ { \mathrm { r e f i n e } } ( i , j )$ (Eqn. (4) or (8)) and $\mathcal { N } ( i )$ is the top-k semantic neighbors (Eqn. (3) or Eqn. (7)). The visual update $\mathbf { h } _ { i } ^ { v , ( \ell + 1 ) }$ is defined symmetrically. Each term has a distinct role: intra-modal aggregation propagates within a stream, self-injection mixes the node’s own counterpart modality representation, and cross-modal neighbor aggregation propagates counterpart-modality signals over the refined graph. Importantly, this formulation reduces the number of nodes by half and decreases the number of edges from $4 E + 2 N$ in Eq. (11) to 2E, making it more efficient.

Complexity optimization. The per-modality view above avoids materializing $\mathbf { A } _ { \mathrm { s u p } }$ , but still requires four GNN calls per layer (two per modality: one for intra-modal and one for cross-modal neighbor aggregation). By exploiting linearity of aggregation,

$$
\sum _ { j } A _ { i j } ( \mathbf { h } _ { j } ^ { t } + \rho \mathbf { h } _ { j } ^ { v } ) = \sum _ { j } A _ { i j } \mathbf { h } _ { j } ^ { t } + \rho \sum _ { j } A _ { i j } \mathbf { h } _ { j } ^ { v } ,\tag{13}
$$

we merge intra-modal and cross-modal neighbor aggregation into a single GNN call per modality:

$$
\tilde { \mathbf { h } } _ { i } ^ { t , ( \ell ) } = \mathbf { G } \mathbf { N } \mathbf { N } ( \mathbf { h } _ { j } ^ { t , ( \ell ) } + \rho \mathbf { h } _ { j } ^ { v , ( \ell ) } , \mathbf { A } _ { \mathrm { r e f i n e } } ) , \qquad \tilde { \mathbf { h } } _ { i } ^ { v , ( \ell ) } = \mathbf { G } \mathbf { N } \mathbf { N } ( \mathbf { h } _ { j } ^ { v , ( \ell ) } + \rho \mathbf { h } _ { j } ^ { t , ( \ell ) } , \mathbf { A } _ { \mathrm { r e f i n e } } ) ,\tag{14}
$$

followed by the self-injection term:

$$
\mathbf { h } _ { i } ^ { t , ( \ell + 1 ) } = \sigma \Bigl ( \mathrm { L N } \bigl ( \tilde { \mathbf { h } } _ { i } ^ { t , ( \ell ) } + \gamma \mathbf { W } ^ { ( \ell ) } \mathbf { h } _ { i } ^ { v , ( \ell ) } \bigr ) \Bigr ) , \quad \mathbf { h } _ { i } ^ { v , ( \ell + 1 ) } = \sigma \Bigl ( \mathrm { L N } \bigl ( \tilde { \mathbf { h } } _ { i } ^ { v , ( \ell ) } + \gamma \mathbf { W } ^ { ( \ell ) } \mathbf { h } _ { i } ^ { t , ( \ell ) } \bigr ) \Bigr ) .\tag{15}
$$

This reduces the cost from four to two calls per layer and is exactly equivalent to Eq. (11) for any linear aggregation operator. A detailed complexity optimization analysis is provided in Appendix E.

Closed loop with the refiner. Combined with the graph refiner of Sec. 2.1, the propagation step closes a loop: at each training step, refined representations sharpen the topology produced by Eq. (4)/(8), and the updated topology in turn shapes the next round of cross-modal propagation. This is the reciprocal co-evolution formalized in Proposition 2: fixed refiners (i.e., not jointly optimized with the

GNN) are recovered as special cases, while ReCoG additionally allows task gradients to update the topology refiner, strictly enlarging the feasible optimization space.

Proposition 2 (Joint Optimization Contains Non-Reciprocal Alternatives). Let $\mathcal { I } ( \theta _ { \mathrm { r e f } } , \theta _ { \mathrm { g n n } } ) =$ $\ell ( \hat { f _ { \theta _ { \mathrm { g n n } } } } ( X , A _ { \mathrm { r e f i n e } } ( \theta _ { \mathrm { r e f } } ) \bar { ) } , Y )$ denote the end-to-end task objective. Then the joint optimization space contains anyfixed orfrozen refiner as a special case:

$$
\operatorname* { i n f } _ { \theta _ { \mathrm { r e f } } , \theta _ { \mathrm { g n n } } } \mathcal { I } ( \theta _ { \mathrm { r e f } } , \theta _ { \mathrm { g n n } } ) \leq \operatorname* { i n f } _ { \theta _ { \mathrm { g n n } } } \mathcal { I } ( \theta _ { \mathrm { r e f } } ^ { \mathrm { f i x } } , \theta _ { \mathrm { g n n } } ) ,\tag{16}
$$

for any fixed $\theta _ { \mathrm { r e f } } ^ { \mathrm { f i x } }$ , including the identity refiner $A _ { \mathrm { r e f i n e } } ~ = ~ A _ { \mathrm { o r i g } }$ when it belongs to the refiner parameter space, and anyfrozen two-stage refiner.

## 2.3 Adaptive Modality Fusion

After L layers of coupled message passing, we obtain modality-specific node representations ${ \bf z } _ { i } ^ { t } = { \bf h } _ { i } ^ { t , ( L ) }$ and ${ \bf z } _ { i } ^ { v } = { \bf h } _ { i } ^ { v , ( L ) }$ . While the two modalities have been progressively aligned through shared propagation and cross-modal interaction, they may still carry complementary or unevenly informative signals. We therefore fuse them into a unified representation through a node-adaptive gating mechanism. Specifically, we compute a node-specific scalar gate:

$$
g _ { i } = \sigma \Big ( \mathrm { M L P } \big ( [ \mathbf { z } _ { i } ^ { t } ; \mathbf { z } _ { i } ^ { v } ] \big ) \Big ) \in ( 0 , 1 ) ,\tag{17}
$$

where $\sigma ( \cdot )$ is the sigmoid function and $[ \cdot ; \cdot ]$ denotes concatenation. The final representation is:

$$
\mathbf { z } _ { i } = \mathrm { L N } \Big ( g _ { i } \cdot \mathbf { z } _ { i } ^ { t } + ( 1 - g _ { i } ) \cdot \mathbf { z } _ { i } ^ { v } \Big ) .\tag{18}
$$

Relation to coupled propagation. Unlike early or late fusion strategies, this design separates two roles: (i) Cross-modal message passing aligns and propagates multimodal information across the graph, and (ii) Adaptivefusion performs final integration based on node-specific relevance.

This separation allows ReCoG to first establish structurally grounded cross-modal representations and then perform flexible, task-adaptive combination. This final adaptive fusion ensures that the benefits of reciprocal co-evolution are translated into task-specific representations.

## 3 Theoretical Analysis

We provide formal justifications for the design choices in ReCoG from two complementary perspectives: the representational benefit of coupled cross-modal propagation, and the cross-modal alignment guarantee of shared-weight design. All proofs including propositions are deferred to Appendix H- L.

Theorem 1 (Layerwise Cross-Modal Information Flow). Define a multimodal GNN as decoupled if each modality stream’s update at layer ℓ depends only on that stream’s representations from layer ℓ − 1 and the graph A, with arbitrary per-modality weight matrices $W _ { T } ^ { ( \ell ) } , W _ { V } ^ { ( \ell ) }$ . Then the cross-modal Jacobian satisfies:

$$
\frac { \partial H _ { T } ^ { ( \ell ) } } { \partial H _ { V } ^ { ( 0 ) } } = \left\{ \begin{array} { l l } { 0 } \\ { \neq 0 ( g e n e r i c a l l y ) } \end{array} \right.
$$

$$
f o r a n y d e c o u p l e d m o d e l , \forall \ell \geq 0 ,\tag{19}
$$

$$
\rho > 0 o r \gamma > 0 , \forall \ell \geq 1
$$

Coupled propagation enables layerwise cross-modal information flow that is structurally impossible in any decoupled architecture, regardless of per-modality parameterization or model capacity. This distinguishes per-layer cross-modal couplingfrom latefusion (e.g., MMGCN), which cannot refine one modality using the graph neighbors ofthe other during propagation.

The supergraph interpretation (Eq. (10)) enables an explicit spectral characterization of how the coupling coefficients reshape the effective frequency response.

Corollary 1 (Spectral Characterization of Coupled Propagation). Let $\mu _ { 1 } \geq \mu _ { 2 } \geq \cdot \cdot \cdot \geq \mu _ { N }$ be the eigenvalues of $A _ { \mathrm { r e f i n e } }$ . Then the 2N eigenvalues of $A _ { \mathrm { s u p } }$ are

$$
\lambda _ { i } ^ { + } = \left( 1 + \rho \right) \mu _ { i } + \gamma , \qquad \lambda _ { i } ^ { - } = \left( 1 - \rho \right) \mu _ { i } - \gamma , \qquad i = 1 , \ldots , N ,\tag{20}
$$

where ${ \lambda } _ { i } ^ { + }$ correspond to eigenmodes in which text and vision components are aligned, and $\lambda _ { i } ^ { - }$ to eigenmodes in which they oppose. Intuitively, ρ and γ act as a learned spectral filter that amplify symmetric modes and attenuate anti-symmetric modes.

Theorem 2 (Shared-Weight Cross-Modal Consistency). Consider one coupled layer with shared weight $W ^ { ( \ell ) }$ and 1-Lipschitz activation σ(·). Define the cross-modal discrepancy $\Delta ^ { ( \ell ) } = H _ { T } ^ { ( \ell ) } - H _ { V } ^ { ( \ell ) }$

Table 1: Node classification accuracy (%) on the MM-Graph [39]. Best and runner-up are highlighted.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="4">Ele-fashion</td><td colspan="4">Goodreads-NC</td></tr><tr><td>CLIP</td><td> $\mathrm { T 5 + V i T }$ </td><td>IBind</td><td>T5+DINO</td><td>CLIP</td><td>T5+ViT</td><td>IBind</td><td>T5+DINO</td></tr><tr><td></td><td>MMGCN</td><td> $8 6 . 1 0 \pm 0 . 5 0$ </td><td> $8 2 . 3 9 \pm 0 . 3 0$ </td><td> $8 6 . 2 1 \pm 0 . 9 4$ </td><td> $8 5 . 5 3 \pm 0 . 3 3$ </td><td> $8 3 . 2 9 \pm 0 . 2 0$ </td><td> $8 1 . 8 5 \pm 0 . 2 2$ </td><td> $8 0 . 5 8 \pm 1 . 0 8$ </td><td> $8 2 . 4 4 \pm 0 . 1 1$ </td></tr><tr><td></td><td>MGAT</td><td> $8 4 . 6 6 \pm 0 . 2 9$ </td><td> $8 4 . 0 1 \pm 0 . 0 8$ </td><td> $8 6 . 1 2 \pm 0 . 0 8$ </td><td> $8 4 . 5 4 \pm 0 . 2 7$ </td><td> $7 6 . 4 8 \pm 0 . 5 9$ </td><td> $7 5 . 4 3 \pm 0 . 7 6$ </td><td> $6 9 . 4 5 \pm 6 . 2 5$ </td><td> $7 4 . 9 8 \pm 1 . 2 3$ </td></tr><tr><td></td><td>GCN</td><td> $7 9 . 8 3 \pm 0 . 0 3$ </td><td> $7 9 . 6 3 \pm 0 . 0 7$ </td><td> $8 0 . 3 5 \pm 0 . 0 2$ </td><td> $7 9 . 3 7 \pm 0 . 0 4$ </td><td> $8 1 . 6 1 \pm 0 . 0 1$ </td><td> $8 1 . 6 7 \pm 0 . 0 3$ </td><td> $7 8 . 9 1 \pm 0 . 0 4$ </td><td> $8 1 . 7 1 \pm 0 . 0 3$ </td></tr><tr><td></td><td>SAGE</td><td> $8 7 . 1 0 \pm 0 . 0 2$ </td><td> $8 4 . 4 1 \pm 0 . 0 9$ </td><td> $8 7 . 7 1 \pm 0 . 1 3$ </td><td> $8 5 . 3 1 \pm 0 . 0 9$ </td><td> $8 3 . 3 0 \pm 0 . 0 2$ </td><td> $8 3 . 0 3 \pm 0 . 0 4$ </td><td> $8 0 . 3 9 \pm 0 . 2 1$ </td><td> $8 2 . 9 9 \pm 0 . 0 8$ </td></tr><tr><td></td><td>MLP</td><td> $8 5 . 1 6 \pm 0 . 0 3$ </td><td> $8 4 . 9 8 \pm 0 . 0 5$ </td><td> $8 8 . 7 3 \pm 0 . 0 1$ </td><td> $8 4 . 8 7 \pm 0 . 0 1$ </td><td> $7 2 . 2 9 \pm 0 . 0 2$ </td><td> $6 7 . 8 2 \pm 0 . 0 7$ </td><td> $5 8 . 7 5 \pm 0 . 0 5$ </td><td> $6 8 . 8 3 \pm 0 . 0 3$ </td></tr><tr><td>m</td><td>ReCoG-GCN</td><td> $8 8 . 5 6 \pm 0 . 0 5$ </td><td> $8 7 . 2 5 \pm 0 . 0 8$ </td><td> $8 9 . 0 1 \pm 0 . 0 3$ </td><td> $8 7 . 7 0 \pm 0 . 0 6$ </td><td> $9 0 . 4 3 \pm 0 . 0 9$ </td><td> $8 8 . 1 8 \pm 0 . 1 0$ </td><td> $8 9 . 2 6 \pm 0 . 0 8$ </td><td> $8 9 . 2 1 \pm 0 . 1 2$ </td></tr><tr><td></td><td>ReCoG-SAGE</td><td> $8 9 . 0 1 \pm 0 . 0 5$  _</td><td> $8 8 . 0 2 \pm 0 . 0 7$ </td><td> $8 9 . 3 1 \pm 0 . 1 0$ </td><td> $8 8 . 2 1 \pm 0 . 1 0$ </td><td> $9 0 . 5 1 \pm 0 . 0 8$ </td><td> $8 8 . 2 4 \pm 0 . 1 1$ </td><td> $8 9 . 3 1 \pm 0 . 0 9$ </td><td> $8 9 . 3 0 \pm 0 . 1 0$ </td></tr><tr><td>res</td><td> $_ \mathrm { R e C o G - G C N }$ </td><td> $8 8 . 6 0 \pm 0 . 0 8$ </td><td> $8 7 . 9 9 \pm 0 . 0 9$ </td><td> $8 9 . 0 6 \pm 0 . 1 0$ </td><td> $8 8 . 2 8 \pm 0 . 0 8$ </td><td> $9 0 . 6 8 \pm 0 . 0 6$ </td><td> $9 0 . 9 8 \pm 0 . 0 3$  _</td><td> $9 0 . 0 5 \pm 0 . 0 3$ </td><td> $9 0 . 1 9 \pm 0 . 0 6$ </td></tr><tr><td></td><td>ReCoG-SAGE</td><td> $8 8 . 9 2 \pm 0 . 0 7$  _</td><td> $8 8 . 0 4 \pm 0 . 0 6$ </td><td> $8 9 . 5 1 \pm 0 . 0 3$  </td><td> $8 8 . 3 4 \pm 0 . 0 6$ </td><td> $9 0 . 6 6 \pm 0 . 1 1$ </td><td> $9 0 . 0 7 \pm 0 . 1 1$ </td><td> $9 0 . 2 5 \pm 0 . 0 3$ </td><td> $9 0 . 0 8 \pm 0 . 0 5$ </td></tr></table>

Table 2: Node classification accuracy (%) on the MAGB datasets [34].
<table><tr><td colspan="2"></td><td colspan="3">Movies</td><td colspan="3">Toys</td><td colspan="3">Grocery</td><td colspan="3">Reddit-S</td><td colspan="3">Reddit-M</td></tr><tr><td>Model</td><td></td><td>L+C</td><td>QWen</td><td>LLaVL</td><td>L+C</td><td>QWen</td><td>LLaVL</td><td>L+C</td><td>QWen</td><td> $\mathrm { L L a V L }$ </td><td>L+C</td><td>QWen</td><td>LLaVL</td><td>L+C</td><td>QWen</td><td>LLaVL</td></tr><tr><td></td><td>GCN</td><td>53.98</td><td>55.19</td><td>54.92</td><td>81.59</td><td>80.95</td><td>81.57</td><td>84.64</td><td>82.96</td><td>84.17</td><td>89.77</td><td>90.88</td><td>91.13</td><td>71.42</td><td>72.96</td><td>75.64</td></tr><tr><td></td><td>SAGE</td><td>56.08</td><td>58.01</td><td>57.39</td><td>82.35</td><td>81.45</td><td>82.35</td><td>86.65</td><td>85.55</td><td>86.81</td><td>91.08</td><td>92.57</td><td>92.94</td><td>80.84</td><td>81.23</td><td>85.61</td></tr><tr><td>sim</td><td>ReCoG-GCN</td><td>56.97</td><td>57.72</td><td>57.51</td><td>81.98</td><td>81.20</td><td>82.19</td><td>85.89</td><td>83.46</td><td>85.03</td><td>97.20</td><td>96.98</td><td>97.26</td><td>90.55</td><td>90.24</td><td>90.87</td></tr><tr><td></td><td>ReCoG-SAGE</td><td>56.94</td><td>58.92</td><td>60.00</td><td>82.58</td><td>82.27</td><td>82.75</td><td>86.86</td><td>86.24</td><td>89.05</td><td>96.76</td><td>96.38</td><td>96.73</td><td>88.75</td><td>89.18</td><td>90.11</td></tr><tr><td>res</td><td>ReCoG-GCN</td><td>57.45</td><td>60.12</td><td>58.62</td><td>82.07</td><td>82.27</td><td>82.27</td><td>86.18</td><td>86.39</td><td>86.56</td><td>96.89</td><td>96.82</td><td>96.73</td><td>87.28</td><td>87.31</td><td>87.42</td></tr><tr><td></td><td>ReCoG-SAGE</td><td>57.93</td><td>60.84</td><td>60.45</td><td>82.68</td><td>82.36</td><td>83.18</td><td>86.92</td><td>86.56</td><td>87.53</td><td>96.89</td><td>96.48</td><td>97.17</td><td>89.18</td><td>89.71</td><td>89.86</td></tr></table>

<sup>†</sup> Following the MAGB benchmark protocol [34], we report the average over 10 random seeds for node classification.

Then

$$
\begin{array} { r } { \left\| { \Delta ^ { ( \ell + 1 ) } } \right\| \leq \left\| { W ^ { ( \ell ) } } \right\| \left( \left( 1 - \rho \right) \left\| { A _ { \mathrm { r e f i n e } } } \right\| + \gamma \right) \left\| { \Delta ^ { ( \ell ) } } \right\| . } \end{array}\tag{21}
$$

Consequently, after L layers:

$$
\left\| \Delta ^ { ( L ) } \right\| \le \prod _ { \ell = 1 } ^ { L - 1 } \left\| W ^ { ( \ell ) } \right\| \left( ( 1 - \rho ) \left\| A _ { \mathrm { r e f i n e } } \right\| + \gamma \right) \cdot \left\| \Delta ^ { ( 0 ) } \right\| .\tag{22}
$$

When the per-layerfactor $\lVert W ^ { ( \ell ) } \rVert \big ( ( 1 - \rho ) \lVert A _ { \mathrm { r e f i n e } } \rVert + \gamma \big ) < 1 ,$ , the discrepancy contracts exponentially with depth, ensuring that text and vision embeddings converge to a shared representation space.

## 4 Experiments

Datasets. We evaluate ReCoG on two tasks: node classification (NC) and link prediction (LP) across nine multimodal graph datasets spanning diverse domains. For NC, we consider two product-review graphs, Ele-Fashion and Goodreads-NC [39], and five datasets from the MAGB benchmark [34], including three e-commerce graphs (Movies, Toys, Grocery) and two social-interaction graphs (Reddit-S, Reddit-M). For LP, we use two Amazon co-purchase graphs, Amazon-Sports and Amazon-Cloth [39]. Each node is associated with a text description and an image, from which pretrained encoders extract modality-specific feature vectors. Detailed dataset statistics are provided in Table 8 in the Appendix. All experiments were conducted on a single NVIDIA A100 GPU.

Encoders. To ensure fair and comprehensive evaluation, we follow benchmark protocols and consider multiple encoder configurations. For Ele-Fashion, Goodreads-NC, and LP datasets, we use four combinations: CLIP, T5+ViT, ImageBind, and $T 5 { + } D I N O \nu 2$ . CLIP and ImageBind produce aligned multimodal embeddings, while ViT/DINOv2 serve as vision encoders and T5 as the text encoder.

For MAGB datasets, we adopt three configurations: $L { + } C \mathrm { ( L L a M A { - } } 3 . 1 { - } 8 \mathrm { B } + \mathrm { C L I P ) }$ , QWen (QWen2- VL-7B), and LLaVL (LLaMA-3.2-11B-Vision). A key distinction is that QWen and LLaVL baselines directly use jointly pretrained multimodal embeddings, where text and image representations are inherently aligned. In contrast, ReCoG treats text and image features as separate streams and learns cross-modal alignment during training. As shown in our results, this learned alignment consistently matches or surpasses encoder-level alignment, highlighting the effectiveness of our architecture.

Baselines. Our baselines fall into two groups: (i) multimodal GNNs that operate on fixed graphs (MMGCN and MGAT [39] with late fusion, GCN [13]/SAGE [8]/MLP on concatenated features), and (ii) recent multimodal graph models, including NSG-MoE [36], LION [16], PLANET [18], and

Table 3: Comparison with recent multimodal graph methods. R-S = Reddit-S, R-M = Reddit-M.
<table><tr><td rowspan="2">Model</td><td colspan="7">Node Classification (Acc. %)</td><td colspan="2">Link Pred. (MRR %)</td></tr><tr><td>Fashion</td><td>GoodReads</td><td>Movies</td><td>Toys</td><td>Grocery</td><td>R-S</td><td>R-M</td><td>Sports</td><td>Cloth</td></tr><tr><td>NSG-MoE</td><td>88.42</td><td></td><td>53.98</td><td>80.92</td><td>86.28</td><td>95.93</td><td>85.03</td><td>34.39</td><td>24.10</td></tr><tr><td>LION</td><td></td><td>78.54</td><td>58.61</td><td></td><td>88.30</td><td>94.80</td><td></td><td></td><td></td></tr><tr><td>PLANET</td><td>87.37</td><td>84.16</td><td>57.06</td><td>81.22</td><td>85.16</td><td>96.62</td><td></td><td>27.51</td><td>20.25</td></tr><tr><td>UniGraph2</td><td>87.91</td><td>81.97</td><td>*52.92</td><td>*79.49</td><td>*85.24</td><td>*93.50</td><td>*70.19</td><td>31.61</td><td>25.01</td></tr><tr><td>ReCoG-sim</td><td>89.31</td><td>90.51</td><td>60.00</td><td>82.75</td><td>89.05</td><td>97.26</td><td>90.87</td><td>38.53</td><td>28.30</td></tr><tr><td>ReCoG-res</td><td>89.51</td><td>90.98</td><td>60.84</td><td>83.18</td><td>87.53</td><td>97.17</td><td>89.86</td><td>38.57</td><td>28.73</td></tr></table>

Results were not reported in the original paper, we use the numbers from [36, 16].  
“—” indicates the method was not evaluated on that dataset.

UniGraph2 [9]. Note that, following the settings of major baselines and benchmarks [39, 34], we focus on multimodal-aware baselines, as most structure learning methods are designed to enhance single-modality graphs without semantic consideration and would require nontrivial adaptation to handle cross-modal evidence fairly.

Evaluation. We evaluate two variants of our model: ReCoG-sim (similarity refiner) and ReCoG-res (residual refiner), both using the same cross-modal message passing backbone. Each variant is tested with both GCN-style and SAGE-style convolutions. We follow the same data splits and experimental settings as prior work, including multiple runs with different random seeds. For fair comparison, we reproduce results for MMGCN, MMGAT, and MLP, while reporting official results for recent state-of-the-art methods. For models with substantially different architectures (e.g., UniGraph2), we use the best reported results from their original papers.

## 4.1 Node Classification Results

Tables 1 and 2 report node classification accuracy across all datasets and encoder configurations.   
Table 3 compares the best performance between our ReCoG and SOTA multimodal graph models.

Overall performance. ReCoG consistently outperforms all baselines across datasets, encoders, and backbone architectures. The improvements are particularly significant on large and structurally complex graphs, indicating that jointly refining topology and representations is especially beneficial when the observed graph is noisy or incomplete.

Key observations. (i) Strong gains on large scale datasets. On Goodreads-NC, ReCoG-res achieves 90.98%, outperforming the strongest recent baseline (PLANET, 84.16%) by a large margin. On Reddit-M, ReCoG improves over GCN by more than 15%, demonstrating substantial robustness. We attribute this to the fact that large-scale interaction graphs tend to contain a higher proportion of semantically irrelevant edges, and the refiner mitigates this issue by down-weighting noisy edges and introducing semantically coherent ones. (ii) Consistent improvements across encoders. ReCoG improves performance under all encoder configurations, including CLIP, ImageBind, and VLM-based encoders (QWen, LLaVL), indicating that the gains are not tied to a specific feature extractor. (iii) Graph-level alignment surpasses encoder-level alignment. Even when using VLM embeddings that are already aligned (e.g., QWen, LLaVL), ReCoG still achieves significant gains (e.g., +6.10% on Reddit-S and +17.28% on Reddit-M).

Remark. These results highlight two key advantages of ReCoG. First, the graph refiner improves topology by introducing semantically meaningful connections and removing noisy edges. Second, cross-modal message passing enables information exchange during propagation rather than relying on static encoder alignment. Together, these components produce representations that are better aligned with both graph structure and multimodal semantics.

Effect of refiner variants. The residual refiner generally achieves the best performance, suggesting that its expressive edge scoring better captures complex multimodal relationships. However, the similarity refiner remains competitive and sometimes preferable when pretrained embedding geometry is already informative, due to its strong inductive bias and lower risk of overfitting.

## 4.2 Link Prediction Results

Table 5 (in the Appendix due to space limit) reports link prediction performance on Amazon-Sports and Amazon-Cloth, evaluated using MRR, Hits@1, and Hits@10. ReCoG consistently outperforms all baselines across both datasets and all encoder configurations, demonstrating robust improvements in LP quality.

Key observations. First, ReCoG achieves substantial gains in MRR. On Amazon-Sports with CLIP features, ReCoG-res (GCN) reaches 38.57% MRR, compared to 33.83% from the strongest baseline (SAGE). On Amazon-Cloth, ReCoG-res (GCN) achieves 28.73% MRR, outperforming SAGE (25.20%) by nearly 3 points. Second, improvements are particularly pronounced in Hits@10. For example, on Amazon-Sports with ImageBind features, ReCoG-res (GCN) achieves 75.60% H@10 versus 65.61% for GCN. This suggests that the refined topology is especially effective at expanding the set of high-quality candidate links.

Variant comparison. The two refiner variants show similar trends to node classification but with smaller differences. ReCoG-sim performs slightly better on Amazon-Sports across most encoders, while ReCoG-res shows marginal advantages on Amazon-Cloth. This reduced gap indicates that, in link prediction, performance is more sensitive to the quality of the learned representations than to the specific refinement mechanism, since the decoder directly operates on embedding similarities.

## 4.3 Ablation Studies

Synergistic Effect. We analyze the contribution of each component in ReCoG by comparing the full model with variants that remove (i) the graph refiner, (ii) cross-modal message passing, or (iii) both. The variant without both components reduces to independent modality-specific GNNs with late fusion. Experiments are conducted on three datasets using GCN as the backbone. We use CLIP features for Ele-Fashion, and CLIP+LLaMA for Movies and Reddit-S. Results are reported in Table 4.

Overall effect of components. Removing either component consistently degrades performance, while removing both leads to the largest drop. This confirms that both topology refinement and cross-modal interaction are essential, and their combination provides the strongest gains.

Synergistic interaction. A key observation is that the benefit of the refiner depends on whether cross-modal message passing is present. When crossmodal coupling is removed, the similarity refiner performs comparably to, or slightly better than, the residual refiner. However, once cross-modal interaction is enabled, the residual refiner consistently achieves the best performance.

Table 4: Ablation study on node classification accuracy (%). † MMGCN uses separate GCNs for each modality with late fusion.
<table><tr><td>Configuration</td><td>Fashion (Clip)</td><td>Movies (L+C)</td><td>Reddit-S (L+C)</td></tr><tr><td></td><td>MMGCN† (w/o both) 86.10</td><td>54.18</td><td>91.13</td></tr><tr><td rowspan="3">sm Full Model</td><td>w/o Cross-Modal</td><td>86.83</td><td>55.16 94.28</td></tr><tr><td>w/o Refiner</td><td>87.09 56.19</td><td>95.97</td></tr><tr><td>88.56</td><td>56.97</td><td>97.20</td></tr><tr><td rowspan="3">res</td><td>w/o Cross-Modal</td><td>86.41</td><td>55.74</td><td>92.74</td></tr><tr><td>w/o Refiner</td><td>87.09</td><td>56.19</td><td>95.97</td></tr><tr><td>Full Model</td><td>88.60</td><td>57.45</td><td>96.89</td></tr></table>

Analysis. This behavior reflects the interaction between topology learning and representation learning. The residual refiner constructs a joint multimodal topology based on pairwise features φ(i, j) that combine text and vision signals. When the two modality streams are processed independently, this unified topology becomes suboptimal, as the optimal neighborhood structure may differ across modalities. In addition, the absence of cross-modal interaction weakens the supervision signal for learning expressive edge scoring functions. In contrast, the similarity refiner relies on cosine similarity with a learnable modality weight, which provides a strong inductive bias. When one modality dominates, this bias allows it to produce a robust topology even without cross-modal interaction. When cross-modal coupling is introduced, the downstream GNN aggregates mixedmodality information at each layer. In this setting, a jointly learned topology becomes better aligned with the propagation mechanism, and the residual refiner can fully exploit its higher expressive power. The enriched cross-modal gradients further improve its learning dynamics, enabling it to surpass the similarity-based variant.

Takeaway. These results highlight that topology refinement and cross-modal propagation are not independent: the effectiveness of a more expressive refiner is unlocked only when coupled with cross-modal message passing, supporting the design of reciprocal co-evolution in ReCoG.

Top-K Selection (Eqns. (3) , (7)). Table 6 examines the sensitivity of both refiners to the number of candidate K. The results show that our model’s performance changes slightly within a reasonable range of K, demonstrating the stability of our design. A detailed analysis is provided in Appendix D.

## 5 Conclusion

We propose ReCoG, a co-evolution framework that jointly refines graph topology and learns crossmodal node representations end-to-end for multimodal attributed graphs. Unlike prior methods that treat topology as fixed or perform modality fusion only at the final stage, ReCoG enables the refined structure and learned embeddings to mutually reinforce each other throughout training, via a configurable Multimodal Graph Refiner and a Cross Modal Message Passing. Extensive experiments on nine datasets spanning node classification and link prediction demonstrate consistent improvements over both classical and recent state-of-the-art baselines. Moreover, our approach is parallelizable to standard GNN training pipelines and shows that learned cross-modal coupling can complement, and in some cases substitute for, the alignment quality of jointly pretrained VLMs.

## References

[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609 (2023).

[2] Yu Chen, Lingfei Wu, and Mohammed J. Zaki. 2020. Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings. In Advances in Neural Information Processing Systems (NeurIPS).

[3] Eli Chien, Wei-Cheng Chang, Cho-Jui Hsieh, Hsiang-Fu Yu, Jiong Zhang, Olgica Milenkovic, and Inderjit S Dhillon. 2022. Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction. In ICLR.

[4] Bahare Fatemi, Layla El Asri, and Seyed Mehran Kazemi. 2021. SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks. In Advances in Neural Information Processing Systems (NeurIPS).

[5] Alex Fout, Jonathon Byrd, Basir Shariat, and Asa Ben-Hur. 2017. Protein interface prediction using graph convolutional networks. Advances in neural information processing systems 30 (2017).

[6] Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. 2019. Combining Neural Networks with Personalized PageRank for Classification on Graphs. In International Conference on Learning Representations. https://openreview.net/forum?id=H1gL-2A9Ym

[7] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 15180–15190.

[8] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. Advances in neural information processing systems 30 (2017).

[9] Yufei He, Yuan Sui, Xiaoxin He, Yue Liu, Yifei Sun, and Bryan Hooi. 2025. Unigraph2: Learning a unified embedding space to bind multimodal graphs. In Proceedings ofthe ACM on Web Conference 2025. 1759–1770.

[10] Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. 2020. Heterogeneous graph transformer. In Proceedings ofthe web conference 2020. 2704–2710.

[11] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning. PMLR, 4904–4916.

[12] Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, and Jiliang Tang. 2020. Graph Structure Learning for Robust Graph Neural Networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD). 66–74.

[13] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016).

[14] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 19730–19742.

[15] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In International conference on machine learning. PMLR, 12888–12900.

[16] Xunkai Li, Zhengyu Wu, Zekai Chen, Henan Sun, Daohan Su, Guang Zeng, Hongchao Qin, Rong-Hua Li, and Guoren Wang. 2026. LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning. arXiv preprint arXiv:2601.21453 (2026).

[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.

[18] Sicheng Liu, Xunkai Li, Daohan Su, Ru Zhang, Hongchao Qin, Ronghua Li, and Guoren Wang. 2026. Toward Effective Multimodal Graph Foundation Model: A Divide-and-Conquer Based Approach. arXiv preprint arXiv:2602.04116 (2026).

[19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[20] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. 2022. Recipe for a general, powerful, scalable graph transformer. Advances in Neural Information Processing Systems 35 (2022), 14501–14515.

[21] Aravind Sankar, Yozen Liu, Jun Yu, and Neil Shah. 2021. Graph neural networks for friend ranking in large-scale social platforms. In Proceedings ofthe Web Conference 2021. 2535–2546.

[22] Xianfeng Tang, Yozen Liu, Neil Shah, Xiaolin Shi, Prasenjit Mitra, and Suhang Wang. 2020. Knowing your fate: Friendship, action and temporal explanations for user engagement prediction on social apps. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2269–2279.

[23] Petar Velickovi ˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua´ Bengio. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 (2017).

[24] Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S Yu. 2019. Heterogeneous graph attention network. In The world wide web conference. 2022–2032.

[25] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. 2019. Simplifying graph convolutional networks. In International conference on machine learning. PMLR, 6861–6871.

[26] Qitian Wu, Wentao Zhao, Zenan Li, David P. Wipf, and Junchi Yan. 2022. NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification. In Advances in Neural Information Processing Systems (NeurIPS).

[27] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022. Graph neural networks in recommender systems: a survey. Comput. Surveys 55, 5 (2022), 1–37.

[28] Rui Xue. 2025. VISAGNN: Versatile Staleness-Aware Efficient Training on Large-Scale Graphs. arXiv preprint arXiv:2511.12434 (2025).

[29] Rui Xue, Haoyu Han, MohamadAli Torkamani, Jian Pei, and Xiaorui Liu. 2023. Lazygnn: Large-scale graph neural networks via lazy propagation. In International Conference on Machine Learning. PMLR, 38926–38937.

[30] Rui Xue, Haoyu Han, Tong Zhao, Neil Shah, Jiliang Tang, and Xiaorui Liu. 2023. Large-Scale Graph Neural Networks: The Past and New Frontiers. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Long Beach, CA, USA) (KDD ’23). Association for Computing Machinery, New York, NY, USA, 5835–5836. doi:10. 1145/3580305.3599565

[31] Rui Xue, Xipeng Shen, Ruozhou Yu, and Xiaorui Liu. 2023. Efficient large language models fine-tuning on graphs. arXiv preprint arXiv:2312.04737 (2023).

[32] Rui Xue and Tianfu Wu. 2025. H<sup>3</sup> GNNs: Harmonizing Heterophily and Homophily in GNNs via Joint Structural Node Encoding and Self-Supervised Learning. arXiv preprint arXiv:2504.11699 (2025).

[33] Rui Xue, Tong Zhao, Neil Shah, and Xiaorui Liu. 2024. Haste Makes Waste: A Simple Approach for Scaling Graph Neural Networks. arXiv preprint arXiv:2410.05416 (2024).

[34] Hao Yan, Chaozhuo Li, Jun Yin, Zhigang Yu, Weihao Han, Mingzheng Li, Zhengxin Zeng, Hao Sun, and Senzhang Wang. 2025. When graph meets multimodal: benchmarking and meditating on multimodal attributed graph learning. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5842–5853.

[35] Jiahao Zhang, Rui Xue, Wenqi Fan, Xin Xu, Qing Li, Jian Pei, and Xiaorui Liu. 2024. Linear-Time Graph Neural Networks for Scalable Recommendations. In Proceedings of the ACM on Web Conference 2024. 3533–3544.

[36] Yihan Zhang and Ercan E Kuruoglu. 2026. Modality as Heterogeneity: Node Splitting and Graph Rewiring for Multimodal Graph Learning. arXiv preprint arXiv:2602.00067 (2026).

[37] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang. 2022. Learning on large-scale text-attributed graphs via variational inference. arXiv preprint arXiv:2210.14709 (2022).

[38] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. 2020. Beyond homophily in graph neural networks: Current limitations and effective designs. Advances in neural information processing systems 33 (2020), 7793–7804.

[39] Jing Zhu, Yuhang Zhou, Shengyi Qian, Zhongmou He, Tong Zhao, Neil Shah, and Danai Koutra. 2025. Mosaic of modalities: A comprehensive benchmark for multimodal graph learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14215– 14224.

## A Broader Impacts

Our proposed ReCoG advances multimodal graph learning by enabling joint refinement of graph structure and cross-modal representations, which can improve performance in applications such as recommendation systems, social networks, and scientific discovery. By leveraging multimodal semantic signals to infer graph connectivity, the proposed approach can better capture meaningful relationships in complex data.

However, these capabilities also introduce risks. Since ReCoG actively modifies graph topology, it may introduce spurious or biased connections when multimodal features are noisy or misaligned, potentially amplifying existing biases in pretrained encoders or datasets. In applications involving user data, such as social or e-commerce graphs, this may lead to unfair recommendations or unintended profiling.

Additionally, the ability to infer latent relationships from multimodal signals raises privacy concerns, as the model may uncover sensitive or hidden connections between entities. Misuse in surveillance or user behavior modeling contexts is also possible.

Future work should explore bias mitigation, robustness to noisy modalities, and privacy-preserving mechanisms to ensure responsible deployment.

## B Related Work

Graph Neural Networks. Graph Neural Networks (GNNs) have become a dominant paradigm for representation learning on graph-structured data. Early spectral and spatial methods, such as GCN [13], GraphSAGE [8], GAT [23], and SGC [25], learn node representations by recursively aggregating information from local neighborhoods. These models have achieved strong performance on node classification, link prediction, and graph classification tasks by exploiting graph topology as an inductive bias. Subsequent studies further extend GNNs to heterogeneous/heterophilic graphs, graph transformers, and scalable graph learning settings [10, 24, 20, 38, 33, 32, 28]. Despite their success, most conventional GNNs assume that the observed topology is reliable and that node attributes are uni-modal. This assumption is often violated in practical graphs, where edges may be noisy or incomplete with respect to multimodal semantics, and node attributes come from various modalities such as text and vision. Directly applying standard GNNs to concatenated multimodal features or fixed graph structures may therefore propagate modality-specific noise and fail to capture cross-modal dependencies during message passing.

Multimodal Representation Learning. Multimodal representation learning has been a central topic in recent years. It aims to encode signals from different modalities, such as text, images, audio, and video, into semantically aligned embedding spaces. Contrastive vision-language models, such as CLIP [19] and ALIGN [11], learn transferable representations by aligning paired images and texts through large-scale contrastive pre-training. Later models further improve multimodal understanding and generation through stronger vision encoders, language models, cross-attention modules, and instruction tuning, including BLIP [15, 14], ImageBind [7], LLaVA [17], and Qwen-VL [1]. These models provide powerful feature extractors and have been widely used to initialize multimodal node features. However, most multimodal representation learning methods focus on instance-level alignment and do not explicitly model graph topology. As a result, their pretrained embeddings are usually static, task-agnostic, and insensitive to downstream graph structures. In multimodal graphs, semantic relationships between nodes are jointly determined by modality content and graph connectivity. Therefore, simply using pretrained multimodal embeddings is insufficient: the model should further adapt modality alignment and information propagation according to graph topology and task supervision.

Multimodal Graph Learning. Most previous works primarily focus on a single modality, such as text-attributed graphs [3, 37, 31]. Recntly, multimodal graph learning extends graph representation learning to graphs whose nodes or edges are associated with multiple modalities. A straightforward solution is to concatenate multimodal features and feed them into conventional GNNs, or to encode each modality with separate GNN branches followed by late fusion. Representative multimodal graph methods, such as MMGCN and MGAT [39], exploit modality-specific representations and attention-based fusion to combine heterogeneous signals. More recent studies attempt to build more general multimodal graph learning frameworks. UniGraph2 [9] builds a graph foundation modal to learn a unified embedding space for multimodal graphs using modality-specific encoders, GNNs, and mixture-of-experts alignment across domains and modalities. NSG-MoE [36] treats modality as heterogeneity by splitting each node into modality-specific sub-nodes and routing heterogeneous message flows through structured experts. LION [16] introduces a Clifford-algebra-based paradigm to perform modality alignment and adaptive holographic aggregation on multimodal-attributed graphs. PLANET [18] further studies multimodal graph foundation models and proposes topology-aware modality interaction and alignment through embedding-wise domain gating and discretized semantic representation spaces. These methods demonstrate the importance of multimodal interaction and alignment in graph learning. However, many existing approaches either rely on fixed observed topology, perform modality interaction after independent propagation, or emphasize representation alignment without explicitly modeling the reciprocal relationship between graph structure and multimodal semantics, thereby limiting the effectiveness of graph-modality fusion and cross-moda alignment.

Positioning of ReCoG. Different from prior work, our approach treats multimodal graph learning as a coupled topology-semantics co-evolution problem. First, we introduce a multimodal graph refiner that adaptively reweights original edges and adds semantic neighbors using joint textual and visual evidence, addressing the topology–semantics mismatch in the original graph. Second, we perform iterative cross-modal message passing over the refined topology, allowing textual and visual representations to align and exchange information during propagation rather than only at the final fusion stage. This design jointly improves graph structure and multimodal representations in an end-to-end manner, providing a more task-adaptive alternative to fixed-topology GNNs, static multimodal encoders, and late-fusion multimodal graph models.

## C Results for Link Prediction

We provide the full link prediction results in Table 5. A comprehensive analysis is presented in Section 4.2.

## D Top-K Selection (Eqns. (3) and (7)).

Table 6 examines the sensitivity of both the Residual and Cosine refiners to the number of candidate neighbors K. We use CLIP as the encoder and GCN as the backbone for all experiments. Across all datasets and both refiners, we observe a consistent pattern: performance improves as K increases from 1 to around 3–5, and then gradually degrades when $K \stackrel { = } { = } 1 2$

When $K = 1$ , the refiner has too few candidates to meaningfully adjust the graph topology, leading to weaker performance. Conversely, an overly large K may introduce semantically distant nodes as candidates, injecting noise into the refined adjacency and diluting the quality of the added edges. Notably, performance remains relatively stable in the range of $K = 3 \tan K = 5 ,$ where the candidate pool is large enough to capture missing semantic neighbors while remaining small enough to avoid spurious connections. Based on these results, we set $\bar { K ^ { = } } 3$ or $K = 5$ as the default in all experiments for simplicity.

## E Complexity Optimization of Cross Modal Message Passing

We describe two successive optimizations that reduce computational cost without altering the output.

## E.1 Optimization I: Supergraph Reformulation

Directly constructing $\mathbf { A } _ { \mathrm { s u p } }$ requires 2N nodes and $4 E + 2 N$ edges. By computing each term in Eq. (12) independently, we avoid constructing the supergraph entirely. This decomposition requires four conv calls (two per modality), each operating on N nodes and E edges, plus two lin calls at $O ( N \cdot d _ { h } ^ { 2 } )$ each. The total per-layer cost is:

$$
4 \times \underbrace { O ( E \cdot d _ { h } + N \cdot d _ { h } ^ { 2 } ) } _ { \mathrm { c o n v } } + 2 \times \underbrace { O ( N \cdot d _ { h } ^ { 2 } ) } _ { \mathrm { l i n } }\tag{23}
$$

compared to the supergraph cost of $O ( ( 4 E + 2 N ) \cdot d _ { h } + 2 N \cdot d _ { h } ^ { 2 } )$ for a single conv over 2N nodes. The decomposed version halves the node count per call and eliminates the overhead of constructing

Table 5: Link prediction performance on Amazon-Sports and Amazon-Cloth. [39]. The Best and runner-up are highlighted.
<table><tr><td rowspan="2"></td><td rowspan="2">Encoder</td><td colspan="3">Amazon-Sports</td><td colspan="3">Amazon-Cloth</td></tr><tr><td>MRR</td><td>H@1</td><td>H@10</td><td>MRR</td><td>H@1</td><td>H@10</td></tr><tr><td rowspan="4">MMGCN</td><td>CLIP</td><td> $3 1 . 9 6 \pm 0 . 1 0$ </td><td> $1 6 . 3 5 \pm 0 . 1 1$ </td><td> $6 8 . 4 6 \pm 0 . 0 8$ </td><td> $2 2 . 2 0 \pm 0 . 0 5$ </td><td> $1 0 . 7 6 \pm 0 . 1 0$ </td><td> $4 6 . 6 2 \pm 0 . 1 2$ </td></tr><tr><td>ViT+T5</td><td> $3 0 . 3 3 \pm 0 . 0 8$ </td><td> $1 5 . 0 1 \pm 0 . 0 5$ </td><td> $6 6 . 4 1 \pm 0 . 1 1$ </td><td> $1 9 . 4 5 \pm 0 . 3 4$ </td><td> $9 . 2 2 \pm 0 . 2 0$ </td><td> $4 0 . 4 9 \pm 0 . 6 1$ </td></tr><tr><td>ImageBind</td><td> $3 1 . 7 4 \pm 0 . 2 1$ </td><td> $1 6 . 4 5 \pm 0 . 1 3$ </td><td> $6 7 . 3 9 \pm 0 . 7 4$ </td><td> $2 4 . 7 2 \pm 0 . 1 9$ </td><td> $1 2 . 4 7 \pm 0 . 0 9$ </td><td> $5 1 . 3 2 \pm 0 . 5 6$ </td></tr><tr><td>DINO+T5</td><td> $3 0 . 0 4 \pm 0 . 2 7$ </td><td> $1 4 . 9 8 \pm 0 . 0 7$ </td><td> $6 4 . 5 6 \pm 0 . 5 6$ </td><td> $2 1 . 7 7 \pm 0 . 2 3$ </td><td> $1 0 . 4 7 \pm 0 . 1 2$ </td><td> $4 5 . 8 1 \pm 0 . 5 2$ </td></tr><tr><td rowspan="4">MGAT</td><td>CLIP</td><td> $2 7 . 5 6 \pm 0 . 3 0$ </td><td> $1 3 . 5 5 \pm 0 . 2 9$ </td><td> $6 0 . 2 1 \pm 0 . 2 1$ </td><td> $2 1 . 3 8 \pm 0 . 2 3$ </td><td> $1 0 . 3 9 \pm 0 . 2 2$ </td><td> $4 4 . 6 0 \pm 0 . 3 6$ </td></tr><tr><td>ViT+T5</td><td> $3 0 . 1 5 \pm 0 . 3 4$ </td><td> $1 5 . 2 8 \pm 0 . 3 4$ </td><td> $6 4 . 8 4 \pm 0 . 4 1$ </td><td> $2 0 . 5 9 \pm 0 . 4 1$ </td><td> $9 . 7 9 \pm 0 . 3 0$ </td><td> $4 3 . 4 4 \pm 0 . 7 6$ </td></tr><tr><td>ImageBind</td><td> $3 0 . 1 5 \pm 0 . 1 2$ </td><td> $1 5 . 5 0 \pm 0 . 0 5$ </td><td> $6 4 . 2 0 \pm 0 . 4 3$ </td><td> $2 2 . 1 3 \pm 0 . 2 7$ </td><td> $1 0 . 9 6 \pm 0 . 1 5$ </td><td> $4 5 . 8 4 \pm 0 . 5 7$ </td></tr><tr><td>DINO+T5</td><td> $2 8 . 9 1 \pm 0 . 0 9$ </td><td> $1 4 . 4 7 \pm 0 . 1 8$ </td><td> $6 2 . 1 1 \pm 0 . 2 2$ </td><td> $2 1 . 4 2 \pm 0 . 1 3$ </td><td> $1 0 . 3 8 \pm 0 . 1 3$ </td><td> $4 4 . 1 1 \pm 0 . 5 0$ </td></tr><tr><td rowspan="4">GCN</td><td>CLIP</td><td> $3 1 . 3 8 \pm 0 . 0 8$ </td><td> $1 6 . 5 8 \pm 0 . 1 3$ </td><td> $6 6 . 1 4 \pm 0 . 0 8$ </td><td> $2 2 . 2 8 \pm 0 . 0 5$ </td><td> $1 1 . 8 3 \pm 0 . 0 4$ </td><td> $4 3 . 5 2 \pm 0 . 1 0$ </td></tr><tr><td>ViT+T5</td><td> $3 0 . 8 3 \pm 0 . 0 7$ </td><td> $1 6 . 3 1 \pm 0 . 0 8$ </td><td> $6 4 . 7 6 \pm 0 . 1 5$ </td><td> $2 1 . 6 0 \pm 0 . 0 5$ </td><td>11.37 ± 0.05</td><td>42.29±0.14</td></tr><tr><td>ImageBind</td><td> $3 1 . 6 7 \pm 0 . 0 9$ </td><td> $1 7 . 0 7 \pm 0 . 1 4$ </td><td> $6 5 . 6 1 \pm 0 . 1 0$ </td><td> $2 2 . 8 1 \pm 0 . 0 2$ </td><td> $1 2 . 2 7 \pm 0 . 0 5$ </td><td> $4 4 . 2 8 \pm 0 . 0 9$ </td></tr><tr><td>DINO+T5</td><td> $3 0 . 4 2 \pm 0 . 0 2$ </td><td> $1 6 . 0 2 \pm 0 . 0 3$ </td><td> $6 4 . 0 2 \pm 0 . 0 6$ </td><td> $2 1 . 1 9 \pm 0 . 0 4$ </td><td> $1 1 . 0 9 \pm 0 . 0 6$ </td><td> $4 1 . 4 6 \pm 0 . 1 6$ </td></tr><tr><td rowspan="4">SAGE</td><td>CLIP</td><td> $3 3 . 8 3 \pm 0 . 0 8$ </td><td> $1 7 . 5 7 \pm 0 . 1 4$ </td><td> $7 1 . 9 0 \pm 0 . 0 7$ </td><td> $2 4 . 5 8 \pm 0 . 1 3$ </td><td> $1 2 . 1 6 \pm 0 . 1 1$ </td><td> $5 1 . 1 2 \pm 0 . 0 9$ </td></tr><tr><td>ViT+T5</td><td> $3 2 . 0 1 \pm 0 . 1 0$ </td><td> $1 5 . 9 4 \pm 0 . 1 7$ </td><td>69.84±0.21</td><td> $2 3 . 1 1 \pm 0 . 0 5$ </td><td> $1 1 . 1 0 \pm 0 . 0 4$ </td><td>48.89±0.09</td></tr><tr><td>ImageBind</td><td> $3 4 . 3 2 \pm 0 . 1 1$ </td><td> $1 7 . 8 7 \pm 0 . 2 3$ </td><td> $7 3 . 0 4 \pm 0 . 1 5$ </td><td> $2 5 . 2 0 \pm 0 . 0 9$ </td><td> $1 2 . 6 3 \pm 0 . 0 5$ </td><td> $5 2 . 5 3 \pm 0 . 2 1$ </td></tr><tr><td>DINO+T5</td><td> $3 2 . 2 0 \pm 0 . 1 2$ </td><td> $1 6 . 1 9 \pm 0 . 2 0$ </td><td> $6 9 . 9 8 \pm 0 . 3 2$ </td><td> $2 2 . 9 8 \pm 0 . 0 1$ </td><td> $1 1 . 1 2 \pm 0 . 0 4$ </td><td> $4 8 . 2 8 \pm 0 . 1 1$ </td></tr><tr><td rowspan="4">MLP</td><td>CLIP</td><td> $2 8 . 2 2 \pm 0 . 0 9$ </td><td> $1 4 . 5 4 \pm 0 . 1 6$ </td><td> $5 9 . 4 0 \pm 0 . 0 8$ </td><td> $2 1 . 1 0 \pm 0 . 0 4$ </td><td> $1 0 . 7 0 \pm 0 . 0 5$ </td><td> $4 2 . 7 7 \pm 0 . 0 5$ </td></tr><tr><td>ViT+T5</td><td> $2 4 . 8 1 \pm 0 . 0 5$ </td><td> $1 1 . 6 3 \pm 0 . 0 5$ </td><td>54.78±0.04</td><td> $1 7 . 6 5 \pm 0 . 0 6$ </td><td>8.14±0.04</td><td> $3 6 . 7 7 \pm 0 . 0 6$ </td></tr><tr><td>ImageBind</td><td> $3 0 . 4 5 \pm 0 . 1 4$ </td><td> $1 5 . 9 1 \pm 0 . 1 0$ </td><td> $6 4 . 1 0 \pm 0 . 0 7$ </td><td> $2 2 . 1 8 \pm 0 . 0 2$ </td><td> $1 1 . 4 2 \pm 0 . 0 4$ </td><td> $4 4 . 8 6 \pm 0 . 0 6$ </td></tr><tr><td>DINO+T5</td><td> $2 4 . 8 1 \pm 0 . 1 6$ </td><td> $1 1 . 6 2 \pm 0 . 1 8$ </td><td> $5 4 . 9 7 \pm 0 . 2 2$ </td><td> $1 7 . 5 3 \pm 0 . 1 1$ </td><td> $8 . 0 7 \pm 0 . 0 9$ </td><td> $3 6 . 5 3 \pm 0 . 2 6$ </td></tr><tr><td rowspan="4">ReCoG-sim (GCN)</td><td>CLIP</td><td>38.53±0.03</td><td>21.90 ± 0.05</td><td> $7 5 . 1 1 \pm 0 . 0 2$ </td><td> $2 8 . 3 0 \pm 0 . 0 3$ </td><td> $1 5 . 6 4 \pm 0 . 0 6$ </td><td> $5 4 . 9 6 \pm 0 . 0 5$ </td></tr><tr><td>ViT+T5</td><td> $3 6 . 8 2 \pm 0 . 0 5$ </td><td> $2 0 . 4 9 \pm 0 . 0 3$ </td><td> $7 3 . 3 6 \pm 0 . 0 5$ </td><td> $2 4 . 6 6 \pm 0 . 0 5$ </td><td> $1 2 . 9 4 \pm 0 . 0 9$ </td><td> $4 9 . 1 9 \pm 0 . 0 5$ </td></tr><tr><td>ImageBind</td><td> $3 8 . 3 6 \pm 0 . 0 2$ </td><td> $2 1 . 9 5 \pm 0 . 0 3$ </td><td> $7 4 . 6 8 \pm 0 . 0 9$ </td><td> $2 8 . 6 0 \pm 0 . 0 6$  _</td><td> $1 5 . 8 9 \pm 0 . 0 8$ </td><td> $5 5 . 7 1 \pm 0 . 0 8$ </td></tr><tr><td>DINO+T5</td><td> $3 7 . 1 6 \pm 0 . 0 5$ </td><td>21.29±0.03</td><td> $7 3 . 3 1 \pm 0 . 0 2$ </td><td>27.10±0.06</td><td> $1 4 . 5 1 \pm 0 . 0 6$ </td><td> $5 4 . 0 2 \pm 0 . 1 6$ </td></tr><tr><td rowspan="4">ReCoG-sim (SAGE)</td><td>CLIP</td><td> $3 5 . 3 4 \pm 0 . 0 3$ </td><td> $1 8 . 6 6 \pm 0 . 1 2$ </td><td> $7 4 . 2 3 \pm 0 . 0 8$ </td><td> $2 7 . 0 2 \pm 0 . 0 9$ </td><td> $1 4 . 1 4 \pm 0 . 1 8$ </td><td> $5 4 . 3 8 \pm 0 . 0 8$ </td></tr><tr><td>ViT+T5</td><td> $3 4 . 5 5 \pm 0 . 1 8$ </td><td> $1 8 . 3 1 \pm 0 . 1 8$ </td><td> $7 1 . 1 1 \pm 0 . 1 0$ </td><td> $2 5 . 3 8 \pm 0 . 1 1$ </td><td> $1 2 . 9 9 \pm 0 . 0 9$ </td><td> $5 1 . 7 8 \pm 0 . 0 8$ </td></tr><tr><td>ImageBind</td><td> $3 6 . 9 0 \pm 0 . 0 8$ </td><td> $2 0 . 4 1 \pm 0 . 0 6$ </td><td> $7 5 . 0 3 \pm 0 . 0 8$ </td><td> $2 7 . 0 8 \pm 0 . 0 6$ </td><td> $1 4 . 4 8 \pm 0 . 1 1$ </td><td> $5 4 . 0 3 \pm 0 . 1 2$ </td></tr><tr><td> $\mathrm { D I N O + T } 5$ </td><td> $3 5 . 3 6 \pm 0 . 1 0$ </td><td> $1 9 . 1 3 \pm 0 . 1 1$ </td><td> $7 2 . 4 0 \pm 0 . 0 3$ </td><td> $2 6 . 3 6 \pm 0 . 0 3$ </td><td> $1 4 . 0 0 \pm 0 . 0 5$ </td><td> $5 3 . 0 5 \pm 0 . 0 3$ </td></tr><tr><td rowspan="4">ReCoG-res (GCN)</td><td>CLIP</td><td> $3 8 . 5 7 \pm 0 . 0 3$  </td><td> $2 2 . 0 4 \pm 0 . 0 5$ </td><td> $7 5 . 3 7 \pm 0 . 0 2$ </td><td> $2 8 . 3 5 \pm 0 . 0 6$ </td><td> $1 5 . 6 4 \pm 0 . 0 8$ </td><td> $5 4 . 6 0 \pm 0 . 0 5$ </td></tr><tr><td>ViT+T5</td><td> $3 7 . 2 8 \pm 0 . 0 3$ </td><td> $2 0 . 8 9 \pm 0 . 0 3$ </td><td> $7 4 . 0 6 \pm 0 . 0 6$ </td><td> $2 7 . 8 6 \pm 0 . 0 5$ </td><td> $1 5 . 3 2 \pm 0 . 0 9$ </td><td> $5 4 . 3 5 \pm 0 . 0 5$ </td></tr><tr><td>ImageBind</td><td> $3 8 . 4 2 \pm 0 . 0 9$ </td><td> $2 1 . 7 3 \pm 0 . 0 5$ </td><td> $7 5 . 6 0 \pm 0 . 0 3$  </td><td> $2 8 . 7 3 \pm 0 . 0 5$  _</td><td> $1 5 . 7 5 \pm 0 . 0 8$ </td><td>_  $5 5 . 8 5 \pm 0 . 0 8$ </td></tr><tr><td>DINO+T5</td><td> $3 7 . 3 9 \pm 0 . 0 6$ </td><td> $2 1 . 2 9 \pm 0 . 0 3$ </td><td> $7 4 . 8 9 \pm 0 . 0 1$ </td><td> $2 7 . 7 8 \pm 0 . 0 6$ </td><td> $1 5 . 1 1 \pm 0 . 0 5$ </td><td> $5 4 . 5 4 \pm 0 . 0 3$ </td></tr><tr><td rowspan="4">ReCoG-res (SAGE)</td><td>CLIP</td><td> $3 6 . 2 2 \pm 0 . 0 3$ </td><td> $1 9 . 9 2 \pm 0 . 1 2$ </td><td> $7 3 . 7 4 \pm 0 . 0 8$ </td><td> $2 7 . 3 7 \pm 0 . 0 9$ </td><td> $1 4 . 4 9 \pm 0 . 1 2$ </td><td> $5 5 . 3 7 \pm 0 . 0 2$ </td></tr><tr><td>ViT+T5</td><td> $3 5 . 0 7 \pm 0 . 1 0$ </td><td> $1 8 . 6 4 \pm 0 . 1 8$ </td><td> $7 3 . 4 0 \pm 0 . 1 0$ </td><td> $2 6 . 5 5 \pm 0 . 1 0$ </td><td> $1 3 . 7 2 \pm 0 . 0 9$ </td><td> $5 4 . 1 9 \pm 0 . 0 5$ </td></tr><tr><td>ImageBind</td><td> $3 6 . 1 3 \pm 0 . 0 8$ </td><td> $1 9 . 8 6 \pm 0 . 0 6$ </td><td> $7 3 . 9 7 \pm 0 . 1 2$ </td><td> $2 7 . 6 2 \pm 0 . 0 6$ </td><td> $1 4 . 7 6 \pm 0 . 1 1$ </td><td> $5 5 . 3 7 \pm 0 . 0 6$ </td></tr><tr><td>DINO+T5</td><td> $3 5 . 6 0 \pm 0 . 0 5$ </td><td> $1 8 . 8 1 \pm 0 . 1 0$ </td><td> $7 3 . 1 9 \pm 0 . 0 5$ </td><td> $2 5 . 9 5 \pm 0 . 0 8$ </td><td> $1 3 . 9 6 \pm 0 . 0 5$ </td><td> $5 4 . 0 8 \pm 0 . 0 3$ </td></tr></table>

the supergraph edge index (node ID offsets, feature concatenation, and additional 2N cross-modal identity edges).

## E.2 Optimization II: Linear Merging of Terms 1 and 3

For GNN layers whose aggregation operator is linear in the input features, Terms 1 and 3 can be merged into a single conv call:

$$
\mathsf { c o n v } ( \mathbf { h } _ { t } ^ { ( l ) } , \mathbf { A } ) + \rho \cdot \mathsf { c o n v } ( \mathbf { h } _ { v } ^ { ( l ) } , \mathbf { A } ) = \mathsf { c o n v } \big ( \mathbf { h } _ { t } ^ { ( l ) } + \rho \cdot \mathbf { h } _ { v } ^ { ( l ) } , \mathbf { A } \big )
$$

Without merging, computing Terms 1 and 3 for one modality requires two conv calls:

(24)

$$
\begin{array} { r } { \underbrace { \mathtt { c o n v } ( \mathbf { h } _ { t } ^ { ( l ) } , \mathbf { A } ) } _ { \mathtt { T e m 1 } } + \rho \cdot \underbrace { \mathtt { c o n v } ( \mathbf { h } _ { v } ^ { ( l ) } , \mathbf { A } ) } _ { \mathtt { T e r m 3 } } \quad \Rightarrow \quad 2 \times O ( E \cdot d _ { h } ) + 2 \times O ( N \cdot d _ { h } ^ { 2 } ) } \end{array}\tag{25}
$$

Table 6: Ablation study on the number of candidate neighbors K in the graph refiner.
<table><tr><td rowspan="2">Refiner K</td><td rowspan="2"></td><td colspan="2">Node Classification (%)</td><td colspan="2">Link Prediction (MRR %)</td></tr><tr><td>Ele-Fashion</td><td>Goodreads</td><td>Amazon-Sports</td><td>s Amazon-Clothing</td></tr><tr><td rowspan="5">sm</td><td>1</td><td> $8 7 . 6 0 \pm 0 . 0 5$ </td><td> $8 9 . 6 5 \pm 0 . 0 5$ </td><td> $3 6 . 7 2 \pm 0 . 1 1$ </td><td> $2 7 . 0 7 \pm 0 . 1 0$ </td></tr><tr><td>3</td><td> $8 8 . 1 7 \pm 0 . 0 3$ </td><td> $9 0 . 4 3 \pm 0 . 0 9$ </td><td> $3 7 . 8 1 \pm 0 . 0 6$ </td><td> $2 7 . 9 3 \pm 0 . 0 7$ </td></tr><tr><td>5</td><td> $8 8 . 5 6 \pm 0 . 0 5$ </td><td> $9 0 . 3 1 \pm 0 . 0 5$ </td><td> $3 8 . 5 3 \pm 0 . 0 3$ </td><td> $2 8 . 3 0 \pm 0 . 0 3$ </td></tr><tr><td>8</td><td> $8 8 . 5 3 \pm 0 . 0 5$ </td><td> $9 0 . 1 6 \pm 0 . 0 6$ </td><td> $3 8 . 3 2 \pm 0 . 0 9$ </td><td> $2 8 . 0 1 \pm 0 . 0 6$ </td></tr><tr><td>12</td><td> $8 7 . 9 5 \pm 0 . 0 6$ </td><td> $8 9 . 4 3 \pm 0 . 1 0$ </td><td> $3 7 . 2 3 \pm 0 . 1 0$ </td><td> $2 6 . 6 8 \pm 0 . 0 5$ </td></tr><tr><td rowspan="5">res</td><td>1</td><td> $8 7 . 8 8 \pm 0 . 0 8$ </td><td> $8 9 . 6 5 \pm 0 . 0 8$ </td><td> $3 7 . 3 6 \pm 0 . 0 9$ </td><td> $2 7 . 3 7 \pm 0 . 0 7$ </td></tr><tr><td>3</td><td> $8 8 . 3 0 \pm 0 . 0 6$ </td><td> $9 0 . 6 8 \pm 0 . 0 6$ </td><td> $3 8 . 5 1 \pm 0 . 0 5$ </td><td> $2 8 . 3 5 \pm 0 . 0 3$ </td></tr><tr><td>5</td><td> $8 8 . 6 0 \pm 0 . 0 8$ </td><td> $9 0 . 5 9 \pm 0 . 0 5$ </td><td> $3 8 . 5 7 \pm 0 . 0 3$ </td><td> $2 8 . 2 5 \pm 0 . 0 2$ </td></tr><tr><td>8</td><td> $8 8 . 3 8 \pm 0 . 0 3$ </td><td> $9 0 . 6 6 \pm 0 . 0 3$ </td><td> $3 8 . 5 1 \pm 0 . 0 8$ </td><td> $2 8 . 0 2 \pm 0 . 0 5$ </td></tr><tr><td>12</td><td> $8 7 . 8 5 \pm 0 . 0 8$ </td><td> $9 0 . 0 7 \pm 0 . 0 5$ </td><td> $3 7 . 7 0 \pm 0 . 0 8$ </td><td> $2 6 . 9 1 \pm 0 . 0 6$ </td></tr></table>

After merging, the cost reduces to a single element-wise addition followed by one conv call:

$$
\underbrace { \mathbf { h } _ { t } ^ { ( l ) } + \rho \cdot \mathbf { h } _ { v } ^ { ( l ) } } _ { O ( N \cdot d _ { h } ) } \longrightarrow \underbrace { \mathsf { c o n v } ( \mathbf { \cdot } , \mathbf { A } ) } _ { O ( E \cdot d _ { h } + N \cdot d _ { h } ^ { 2 } ) }\tag{26}
$$

The $O ( N \cdot d _ { h } )$ addition is negligible compared to the conv cost, so merging effectively halves both the aggregation and linear transformation costs per modality. The complete per-layer update becomes:

$$
\mathbf { h } _ { t } ^ { ( l + 1 ) } = \mathbf { L N } \Big ( \mathsf { c o n v } \big ( \mathbf { h } _ { t } ^ { ( l ) } + \rho \mathbf { h } _ { v } ^ { ( l ) } , ~ \mathbf { A } \big ) + \gamma \cdot \mathbf { \mathrm { 1 i n } } \big ( \mathbf { h } _ { v } ^ { ( l ) } \big ) \Big )\tag{27}
$$

$$
\mathbf { h } _ { v } ^ { ( l + 1 ) } = \mathbf { L N } \Big ( \mathsf { c o n v } \big ( \mathbf { h } _ { v } ^ { ( l ) } + \rho \mathbf { h } _ { t } ^ { ( l ) } , ~ \mathbf { A } \big ) + \gamma \cdot \mathbf { \mathrm { 1 i n } } \big ( \mathbf { h } _ { t } ^ { ( l ) } \big ) \Big )\tag{28}
$$

requiring two conv calls and two lin calls per layer, yielding the total per-layer cost of $O ( E \cdot d _ { h } +$ $N \cdot d _ { h } ^ { 2 } )$ reported in Table 7.

Applicability. The merging in Eq. (24) requires linearity of the aggregation operator and holds for GCN, GraphSAGE-mean, and other linear-aggregation GNNs. For attention-based architectures such as GAT, where the aggregation weights depend nonlinearly on the input features, the merging does not hold exactly. In this case, we fall back to the decomposed implementation (four conv calls), which still avoids constructing the supergraph.

## F Topology–semantics Misalignment

Let the observed multimodal graph be denoted as $\mathcal { G } = ( \nu , \mathcal { E } )$ , where V is the node set and $\mathcal { E }$ is the observed edge set. Each node $i \in \mathcal V$ is associated with multimodal features $\mathbf { x } _ { i } = [ \mathbf { x } _ { i } ^ { t } \parallel \mathbf { x } _ { i } ^ { v } ]$ , where $\mathbf { x } _ { i } ^ { t } \in \mathbb { R } ^ { d _ { t } }$ and $\mathbf { x } _ { i } ^ { v } \in \mathbb { R } ^ { d _ { \tau } }$ denote textual and visual features, respectively. Denote by $\mathbf { A } _ { \mathrm { o r i g } }$ the original adjacency matrix, $\mathbf { A } _ { \mathrm { o r i g } } ( i , j ) = 1 \mathrm { i f } ( i , j ) \in \mathcal { E } ,$ 0 otherwise.

A fundamental challenge in multimodal graphs is that the observed topology is only partially aligned with multimodal semantics. To examine this issue, we analyze the multimodal semantic similarity distributions on Reddit-M (see Sec. 2.1.3) and Movies [34]. For a node pair $( i , j )$ , we compute the textual and visual cosine similarities as

$$
S _ { T } ( i , j ) = \cos ( { \bf x } _ { i } ^ { t } , { \bf x } _ { j } ^ { t } ) , \qquad S _ { V } ( i , j ) = \cos ( { \bf x } _ { i } ^ { v } , { \bf x } _ { j } ^ { v } ) ,\tag{29}
$$

For simplicity, we plot the multimodal semantic similarity as the average

$$
S _ { M M } ( i , j ) = \frac { 1 } { 2 } \big ( S _ { T } ( i , j ) + S _ { V } ( i , j ) \big ) ,\tag{30}
$$

where the encoded features are obtained from CLIP.

As shown on the left side of both Fig. 2 and Fig. 3, we compare the $S _ { M M }$ distributions of connected and disconnected node pairs in the original graph. Connected pairs generally exhibit higher multimodal similarity, indicating that the original topology provides useful structural priors. However, the two distributions show considerable overlap, revealing two types of topology–semantics discrepancy: (i) some connected pairs have low multimodal similarity, suggesting the presence of noisy edges; (ii) some disconnected pairs have high multimodal similarity, indicating missing semantic neighbors.

![](images/01c1bf0f4f552e8e3775a09040ce2a3e7de5514ebec536f7cf2ee4a837efe219.jpg)

![](images/5c1e6c4b4956c88c78e5c88761e23a810a6fe854202131996383d43cce80983f.jpg)  
Figure 3: Topology–semantics alignment.

The right side of both Fig. 2 (Cosine Similarity Refiner) and Fig. 3 (Residual Refiner) illustrates the effect of the refiner. The newly added edges concentrate in a higher similarity range compared to the original connected pairs, and their overlap with the disconnected distribution is substantially reduced. This confirms that the refiner selectively recovers semantically meaningful missing edges without introducing topological noise.

These observations motivate a topology refinement mechanism that preserves the observed connectivity as a structural prior while adaptively suppressing noisy edges and recovering missing semantic neighbors based on joint textual and visual evidence.

## G Complexity Analysis.

We analyze the per-batch computational complexity of ReCoG (Table 7). Let N denote the number of nodes in the sampled subgraph, B the number of seed nodes, E the number of edges, k the number of new edges per seed, r the candidate ratio, $d = d _ { t } + d _ { \tau }$ the raw feature dimension, $d _ { h }$ the hidden dimension in the projection at coarse filtering, and L the number of GNN layers.

Table 7: Per-batch computational complexity of ReCoG major components.
<table><tr><td>Component</td><td>Complexity</td></tr><tr><td>Cosine similarity  $( S _ { T } , S _ { V } )$ </td><td> $O ( B N d )$ </td></tr><tr><td>Original edge scoring (MLP)</td><td> $O ( E \cdot d \cdot d _ { h } )$ </td></tr><tr><td>Coarse filtering</td><td> $O ( B N d _ { h } )$ </td></tr><tr><td>Fine scoring (MLP)</td><td> $O ( B r k \cdot d \cdot d _ { h } )$ </td></tr><tr><td>Cross-modal message passing  $( \times L )$ </td><td> $O ( L ( E \cdot d _ { h } + N \cdot d _ { h } ^ { 2 } ) )$ </td></tr></table>

## H Proof of Proposition 1 (Strict Generalization of Residual Refinement)

We prove $\mathcal { F } _ { \mathrm { s i m } } \subsetneq \mathcal { F } _ { \mathrm { r e s } }$ by establishing inclusion and then strictness. Recall that the similarity refiner induces the edge-scoring family

$$
g _ { \mathrm { s i m } } ( i , j ) = \alpha A _ { \mathrm { o r i g } } ( i , j ) + ( 1 - \alpha ) \big [ \lambda S _ { V } ( i , j ) + ( 1 - \lambda ) S _ { T } ( i , j ) \big ] , \quad ( i , j ) \in \mathcal { C } ,\tag{31}
$$

with global coefficients $\alpha , \lambda \in [ 0 , 1 ]$ , while the residual refiner induces

$$
g _ { \mathrm { r e s } } ( i , j ) = A _ { \mathrm { o r i g } } ( i , j ) + \delta \cdot \mathrm { t a n h } \big ( f _ { \theta } ( \phi ( i , j ) ) \big ) , \quad ( i , j ) \in \mathcal { C } ,\tag{32}
$$

where $\phi ( i , j )$ is the multimodal pair feature $\left( \operatorname { E q . } \left( 5 \right) \right)$ and $f _ { \theta }$ is an MLP with sufficient width.

Part 1: $\mathcal { F } _ { \mathrm { s i m } } \subseteq \mathcal { F } _ { \mathrm { r e s } } .$ . Take any $g _ { \mathrm { s i m } } \in \mathcal { F } _ { \mathrm { s i m } }$ . For each $( i , j ) \in \mathcal { C }$ , define the target value

$$
\begin{array} { r } { u ( i , j ) = \mathrm { a r c t a n h } \bigg ( \frac { g _ { \mathrm { s i m } } ( i , j ) - A _ { \mathrm { o r i g } } ( i , j ) } { \delta } \bigg ) . } \end{array}\tag{33}
$$

By the range assumption and $| g _ { \mathrm { s i m } } ( i , j ) - A _ { \mathrm { o r i g } } ( i , j ) | < \delta ,$ the argument of arctanh lies in (−1, 1), so $u ( i , j )$ is well-defined and finite for every $( i , j ) \in \mathcal { C }$

The similarity-based score depends continuously on $( S _ { T } ( i , j ) , S _ { V } ( i , j ) )$ , which are themselves continuous functions of the pair features $\phi ( i , j )$ on the fixed candidate set C. Since arctanh is continuous on $( - 1 , 1 )$ , the composite map $\phi ( i , j ) \mapsto u ( i , j )$ is continuous on C.

By the universal approximation theorem for MLPs with sufficient width, there exists an MLP $f _ { \theta }$ such that

$$
\left| f _ { \theta } ( \phi ( i , j ) ) - u ( i , j ) \right| < \epsilon , \quad \forall ( i , j ) \in \mathcal { C } ,\tag{34}
$$

for any $\epsilon > 0 .$ . Substituting into the residual scorer:

$$
g _ { \mathrm { r e s } } ( i , j ) = A _ { \mathrm { o r i g } } ( i , j ) + \delta \cdot \operatorname { t a n h } \bigl ( f _ { \theta } ( \phi ( i , j ) ) \bigr )\tag{35}
$$

$$
\approx A _ { \mathrm { o r i g } } ( i , j ) + \delta \cdot \operatorname { t a n h } ( u ( i , j ) )\tag{36}
$$

$$
= A _ { \mathrm { o r i g } } ( i , j ) + \delta \cdot \frac { g _ { \mathrm { s i m } } ( i , j ) - A _ { \mathrm { o r i g } } ( i , j ) } { \delta }\tag{37}
$$

$$
= g _ { \mathrm { s i m } } ( i , j ) .\tag{38}
$$

Since the approximation error can be made arbitrarily small, every function in $\mathcal { F } _ { \mathrm { s i m } }$ can be realized (or approximated to arbitrary precision) by a function in $\mathcal { F } _ { \mathrm { r e s } }$ , establishing $\mathcal { F } _ { \mathrm { s i m } } \subseteq \mathcal { F } _ { \mathrm { r e s } }$

Part 2: Strictness $( \mathcal { F } _ { \mathrm { s i m } } \neq \mathcal { F } _ { \mathrm { r e s } } ) .$ Every function in $\mathcal { F } _ { \mathrm { s i m } }$ depends on a node pair $( i , j )$ only through the two scalars $( S _ { T } ( i , j ) , S _ { V } ( i , j ) )$ , combined via a global linear rule with shared coefficients α and λ. Therefore, for any two pairs $( i , j )$ and $( u , v )$ satisfying

$$
S _ { T } ( i , j ) = S _ { T } ( u , v ) , \quad S _ { V } ( i , j ) = S _ { V } ( u , v ) , \quad A _ { \mathrm { o r i g } } ( i , j ) = A _ { \mathrm { o r i g } } ( u , v ) ,
$$

every $g _ { \mathrm { s i m } } \in \mathcal { F } _ { \mathrm { s i m } }$ must assign them the same score:

$$
g _ { \mathrm { s i m } } ( i , j ) = g _ { \mathrm { s i m } } ( u , v ) .\tag{39}
$$

Now consider the residual scorer. Its input is the full pair feature $\phi ( i , j ) \left( \operatorname { E q . } \left( 5 \right) \right)$ , which includes concatenated raw features, element-wise Hadamard products $\mathbf { x } _ { i } ^ { t } \odot \mathbf { x } _ { j } ^ { t } , \mathbf { x } _ { i } ^ { v } \odot \mathbf { \dot { x } } _ { j } ^ { v }$ , and absolute differences $| \mathbf { x } _ { i } ^ { t } - \mathbf { x } _ { j } ^ { t } | , | \mathbf { x } _ { i } ^ { v } - \mathbf { x } _ { j } ^ { v } |$

We construct a concrete counterexample. Let $d _ { t } = 2$ and consider two node pairs:

$$
\mathrm { P a i r \ 1 : } \quad \mathbf { x } _ { i } ^ { t } = ( 1 , 0 ) , \quad \mathbf { x } _ { j } ^ { t } = ( 0 , 1 ) ,\tag{40}
$$

$$
\begin{array} { r } { \mathrm { P a i r } \ 2 : \quad \mathbf { x } _ { u } ^ { t } = \frac { 1 } { \sqrt { 2 } } ( 1 , 1 ) , \quad \mathbf { x } _ { v } ^ { t } = \frac { 1 } { \sqrt { 2 } } ( 1 , - 1 ) . } \end{array}\tag{41}
$$

Both pairs have unit-norm vectors with ${ \cal { S } } _ { T } = { \bf { x } } _ { i } ^ { t } \cdot { \bf { x } } _ { i } ^ { t } = 0$ and ${ \cal S } _ { T } = { \bf x } _ { u } ^ { t } \cdot { \bf x } _ { v } ^ { t } = 0$ . Assigning identical vision features to both pairs ensures $S _ { V }$ is also equal.

However, the Hadamard products differ:

$$
\begin{array} { r } { \mathbf { x } _ { i } ^ { t } \odot \mathbf { x } _ { j } ^ { t } = ( 0 , 0 ) , } \end{array}\tag{42}
$$

$$
\begin{array} { r } { \mathbf { x } _ { u } ^ { t } \odot \mathbf { x } _ { v } ^ { t } = \frac { 1 } { 2 } ( 1 , - 1 ) , } \end{array}\tag{43}
$$

so $\phi ( i , j ) \neq \phi ( u , v )$ . Since an MLP can realize non-linear, dimension-weighted functions of ϕ, there exists $g _ { \mathrm { r e s } } \in \mathcal { F } _ { \mathrm { r e s } }$ such that

$$
g _ { \mathrm { r e s } } ( i , j ) \neq g _ { \mathrm { r e s } } ( u , v ) ,\tag{44}
$$

which is impossible for any $g _ { \mathrm { s i m } } \in \mathcal { F } _ { \mathrm { s i m } }$ by Eq. (39).

Therefore $\mathcal { F } _ { \mathrm { s i m } } \subsetneq \mathcal { F } _ { \mathrm { r e s } }$

Remark 2. Proposition 1 characterizes expressiveness, not generalization. The inclusion follows from a universal-approximation argument on the residual MLP, and strictnessfrom dimension-wise interactions $( e . g . , \ \bar { \mathbf { x } } _ { i } ^ { t } \odot \mathbf { x } _ { i } ^ { t } )$ that cosine similarity collapses to a scalar. Whether this additional capacity translates to better task performance depends on the learning context: our ablations (Sec. 4.3) show that the residual refiner’s advantage emerges only when coupled with cross-modal propagation.

## I Proof of Proposition 2 (Joint Optimization Contains Non-Reciprocal Alternatives)

Let $\Theta _ { \mathrm { r e f } }$ and $\Theta _ { \mathrm { g n n } }$ denote the parameter spaces of the refiner and the GNN, respectively. Any fixed refiner $\theta _ { \mathrm { r e f } } ^ { \mathrm { f i x } }$ restricts the optimization to the slice $\{ \theta _ { \mathrm { r e f } } ^ { \mathrm { f i x } } \} \times \Theta _ { \mathrm { g n n } } ,$ which is a subset of the full feasible set $\Theta _ { \mathrm { r e f } } \times \Theta _ { \mathrm { g n n } }$ . Therefore, the infimum over the full joint space cannot exceed the infimum over

this restricted slice:

$$
\operatorname* { i n f } _ { ( \theta _ { \mathrm { r e f } } , \theta _ { \mathrm { g n n } } ) \in \Theta _ { \mathrm { r e f } } \times \Theta _ { \mathrm { g n n } } } \mathcal { I } ( \theta _ { \mathrm { r e f } } , \theta _ { \mathrm { g n n } } ) \le \operatorname* { i n f } _ { \theta _ { \mathrm { g n n } } \in \Theta _ { \mathrm { g n n } } } \mathcal { I } ( \theta _ { \mathrm { r e f } } ^ { \mathrm { f i x } } , \theta _ { \mathrm { g n n } } ) .\tag{45}
$$

This includes the identity refiner $A _ { \mathrm { r e f i n e } } = A _ { \mathrm { o r i g } }$ when it belongs to the refiner parameter space, as well as any frozen two-stage refiner. □

## J Proof of Theorem 1 (Cross-Modal Information Accessibility)

We prove both parts for the text stream; the vision stream follows by symmetry.

Part (i): Decoupled model — zero cross-modal Jacobian. A decoupled multimodal GNN updates each modality stream independently, with arbitrary per-modality weights:

$$
H _ { T } ^ { ( \ell + 1 ) } = \sigma \big ( A H _ { T } ^ { ( \ell ) } W _ { T } ^ { ( \ell ) } \big ) , \qquad H _ { V } ^ { ( \ell + 1 ) } = \sigma \big ( A H _ { V } ^ { ( \ell ) } W _ { V } ^ { ( \ell ) } \big ) ,\tag{46}
$$

where $W _ { T } ^ { ( \ell ) }$ and $W _ { V } ^ { ( \ell ) }$ may differ.

We prove $\frac { \partial H _ { T } ^ { ( \ell ) } } { \partial H _ { V } ^ { ( 0 ) } } = 0$ for all $\ell \geq 0$ by induction.

Base case $( \ell = 0 ) . ~ H _ { T } ^ { ( 0 ) }$ is the projected text input, which is independent of $H _ { V } ^ { ( 0 ) }$ by construction.

Inductive step. Suppose $\frac { \partial H _ { T } ^ { ( \ell ) } } { \partial H _ { \scriptscriptstyle { V } } ^ { ( 0 ) } } = 0$ . Since A and $W _ { T } ^ { ( \ell ) }$ are independent of $H _ { V } ^ { ( 0 ) }$ , and $H _ { T } ^ { ( \ell + 1 ) } =$ $\sigma ( A H _ { T } ^ { ( \ell ) } W _ { T } ^ { ( \ell ) } )$ depends on $\dot { H } _ { V } ^ { ( 0 ) }$ only through $H _ { T } ^ { ( \ell ) }$ , by the chain rule:

$$
\frac { \partial H _ { T } ^ { ( \ell + 1 ) } } { \partial H _ { V } ^ { ( 0 ) } } = \frac { \partial H _ { T } ^ { ( \ell + 1 ) } } { \partial H _ { T } ^ { ( \ell ) } } \cdot \frac { \partial H _ { T } ^ { ( \ell ) } } { \partial H _ { V } ^ { ( 0 ) } } = \frac { \partial H _ { T } ^ { ( \ell + 1 ) } } { \partial H _ { T } ^ { ( \ell ) } } \cdot 0 = 0 .\tag{47}
$$

Note that this holds for any choice of per-modality weights $W _ { T } ^ { ( \ell ) } , W _ { V } ^ { ( \ell ) }$ , any activation σ, and any graph A. The result is a structural consequence of the decoupled update rule, not a property of specific parameters. □

Part (ii): Coupled model — non-zero cross-modal Jacobian. The coupled text-stream update is:

$$
H _ { T } ^ { ( \ell + 1 ) } = \sigma \Big ( W ^ { ( \ell ) } \big ( A H _ { T } ^ { ( \ell ) } + \gamma H _ { V } ^ { ( \ell ) } + \rho A H _ { V } ^ { ( \ell ) } \big ) \Big ) .\tag{48}
$$

We show that $\frac { \partial H _ { T } ^ { ( \ell ) } } { \partial H _ { V } ^ { ( 0 ) } } \neq 0$ for all $\ell \geq 1$ when $\rho > 0 \mathrm { o r } \gamma > 0$

Layer $\ell = 1$ . From Eq. (48) with $\ell = 0 :$

$$
H _ { T } ^ { ( 1 ) } = \sigma \Bigl ( { \cal W } ^ { ( 0 ) } \bigl ( A H _ { T } ^ { ( 0 ) } + \gamma H _ { V } ^ { ( 0 ) } + \rho A H _ { V } ^ { ( 0 ) } \bigr ) \Bigr ) .\tag{49}
$$

Let $Z = W ^ { ( 0 ) } \big ( A H _ { T } ^ { ( 0 ) } + \gamma H _ { V } ^ { ( 0 ) } + \rho A H _ { V } ^ { ( 0 ) } \big )$ denote the pre-activation. By the chain rule:

$$
\frac { \partial H _ { T } ^ { ( 1 ) } } { \partial H _ { V } ^ { ( 0 ) } } = \mathrm { d i a g } \big ( \sigma ^ { \prime } ( Z ) \big ) \cdot W ^ { ( 0 ) } \cdot ( \gamma I + \rho A ) .\tag{50}
$$

When $\gamma > 0$ or $\rho > 0$ , the factor $( \gamma I + \rho A ) \neq 0$ . The diagonal matrix $\mathrm { l i a g } ( \sigma ^ { \prime } ( Z ) ) ,$ is non-zero for generic inputs (i.e., except on a measure-zero set where all pre-activations are at non-differentiable points of σ). Hence $\frac { \partial H _ { T } ^ { ( 0 ) } } { \partial H _ { V } ^ { ( 0 ) } } \neq 0$ generically.

Deeper layers $\left( \ell \geq 2 \right)$ . By the chain rule and the multivariate dependency structure (where $H _ { T } ^ { ( \ell ) }$ depends on $H _ { V } ^ { ( 0 ) }$ both directly through the coupling terms and indirectly through $H _ { V } ^ { ( k ) }$ for $k < \ell )$ , the Jacobian $\frac { \partial H _ { T } ^ { ( \ell ) } } { \partial H _ { V } ^ { ( 0 ) } }$ accumulates contributions from all cross-modal paths through layers $1 , \ldots , \ell .$ Since the layer-1 contribution (50) is generically non-zero and subsequent layers compose via non-singular weight matrices, the Jacobian remains generically non-zero at all depths. □

## K Proof of Corollary 1 (Spectral Characterization)

Since $A _ { \mathrm { s u p } }$ is $\phantom { - } 1 2 \times 2$ block matrix with commuting blocks (all blocks are polynomials of $A _ { \mathrm { r e f i n e } } )$ we can diagonalize it via the block eigendecomposition.

Let $\mathbf { v } _ { i }$ be an eigenvector of $A _ { \mathrm { r e f i n e } }$ with eigenvalue $\mu _ { i } .$ Consider the two 2N-dimensional vectors:

$$
{ \bf q } _ { i } ^ { + } = \frac { 1 } { \sqrt { 2 } } \left( { \bf v } _ { i } \right) , \qquad { \bf q } _ { i } ^ { - } = \frac { 1 } { \sqrt { 2 } } \left( { \bf v } _ { i } \right) .\tag{51}
$$

Symmetric mode.

$$
\begin{array} { l } { A _ { \mathrm { s u p } } \mathbf { q } _ { i } ^ { + } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { l } { A _ { \mathrm { r e f i n e } } \mathbf { v } _ { i } + ( \gamma I + \rho A _ { \mathrm { r e f i n e } } ) \mathbf { v } _ { i } } \\ { ( \gamma I + \rho A _ { \mathrm { r e f i n e } } ) \mathbf { v } _ { i } + A _ { \mathrm { r e f i n e } } \mathbf { v } _ { i } } \end{array} \right) } \\ { = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { l } { \mu _ { i } \mathbf { v } _ { i } + \gamma \mathbf { v } _ { i } + \rho \mu _ { i } \mathbf { v } _ { i } } \\ { \gamma \mathbf { v } _ { i } + \rho \mu _ { i } \mathbf { v } _ { i } + \mu _ { i } \mathbf { v } _ { i } } \end{array} \right) = \left[ ( 1 + \rho ) \mu _ { i } + \gamma \right] \mathbf { q } _ { i } ^ { + } . } \end{array}\tag{52}
$$

Anti-symmetric mode.

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { \mathrm { s u p } } { \bf q } _ { i } ^ { - } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { l } { { \displaystyle A _ { \mathrm { r e f i n e } } { \bf v } _ { i } + ( \gamma I + \rho A _ { \mathrm { r e f i n e } } ) \left( - { \bf v } _ { i } \right) } } \\ { { \displaystyle \left( \gamma I + \rho A _ { \mathrm { r e f i n e } } \right) { \bf v } _ { i } + A _ { \mathrm { r e f i n e } } \left( - { \bf v } _ { i } \right) } } \end{array} \right) } } \\ { { \displaystyle ~ = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { l } { { \displaystyle \mu _ { i } { \bf v } _ { i } - \gamma { \bf v } _ { i } - \rho \mu _ { i } { \bf v } _ { i } } } \\ { { \displaystyle \gamma { \bf v } _ { i } + \rho \mu _ { i } { \bf v } _ { i } - \mu _ { i } { \bf v } _ { i } } } \end{array} \right) } } \\ { { \displaystyle ~ = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { l } { { \displaystyle \left[ ( 1 - \rho ) \mu _ { i } - \gamma \right] { \bf v } _ { i } } } \\ { { \displaystyle - \left[ ( 1 - \rho ) \mu _ { i } - \gamma \right] { \bf v } _ { i } } } \end{array} \right) = \left[ ( 1 - \rho ) \mu _ { i } - \gamma \right] { \bf q } _ { i } ^ { - } . } } \end{array}\tag{53}
$$

Since $\{ \mathbf { v } _ { 1 } , \dotsc , \mathbf { v } _ { N } \}$ form a complete eigenbasis for $A _ { \mathrm { r e f i n e } } ,$ the 2N vectors $\{ \mathbf { q } _ { 1 } ^ { + } , \dots , \mathbf { q } _ { N } ^ { + } , \mathbf { q } _ { 1 } ^ { - } , \dots , \mathbf { q } _ { N } ^ { - } \}$ form a complete eigenbasis for $A _ { \mathrm { s u p } } .$ , with eigenvalues $\{ ( 1 + \rho ) \mu _ { i } + \gamma \} _ { i = 1 } ^ { N } \cup \{ ( 1 - \rho ) \mu _ { i } - \gamma \} _ { i = 1 } ^ { N }$ □

Interpretation. The spectral decomposition reveals how $\rho$ and $\gamma$ jointly shape the coupled propagation:

• Symmetric amplification. Cross-modal coupling amplifies graph spectral components in the symmetric subspace by a factor $( 1 + \rho )$ relative to the decoupled case $( \rho = 0 )$ The self-injection $\gamma$ provides an additional frequency-independent boost to symmetric modes. For homophilic graphs, where low-frequency components (large $\mu _ { i } )$ carry the dominant class signal, this amplification strengthens the informative spectral components and accelerates convergence to useful representations.

• Anti-symmetric attenuation. Anti-symmetric modes are attenuated by factor $( 1 - \rho )$ and shifted by $- \gamma .$ . When $\rho$ and $\gamma$ are large, the model aggressively suppresses cross-modal disagreement, which is beneficial when the two modalities carry consistent semantic signals.

• Adaptivity to graph heterophily. For heterophilic graphs where high-frequency components (small or negative $\mu _ { i } )$ carry discriminative signal, the model can learn small $\rho$ and $\gamma ,$ , preserving highfrequency components in both symmetric and anti-symmetric subspaces. Since $\rho = \sigma ( \rho _ { \mathrm { l o g i t } } )$ and $\gamma = \sigma ( \gamma _ { \mathrm { l o g i t } } )$ are learnable, the spectral filter adapts to the homophily/heterophily characteristics of the data without manual tuning.

## L Proof of Theorem 2 (Shared-Weight Cross-Modal Consistency)

The coupled update rules are:

$$
H _ { T } ^ { ( \ell + 1 ) } = \sigma \Bigl ( W ^ { ( \ell ) } \bigl ( A _ { \mathrm { r e f i n e } } H _ { T } ^ { ( \ell ) } + \gamma H _ { V } ^ { ( \ell ) } + \rho A _ { \mathrm { r e f i n e } } H _ { V } ^ { ( \ell ) } \bigr ) \Bigr ) ,\tag{54}
$$

$$
H _ { V } ^ { ( \ell + 1 ) } = \sigma \Bigl ( W ^ { ( \ell ) } \bigl ( A _ { \mathrm { r e f i n e } } H _ { V } ^ { ( \ell ) } + \gamma H _ { T } ^ { ( \ell ) } + \rho A _ { \mathrm { r e f i n e } } H _ { T } ^ { ( \ell ) } \bigr ) \Bigr ) .\tag{55}
$$

Step 1: Express the pre-activation difference. Define the pre-activation inputs:

$$
u _ { T } = A _ { \mathrm { r e f i n e } } H _ { T } ^ { ( \ell ) } + \gamma H _ { V } ^ { ( \ell ) } + \rho A _ { \mathrm { r e f i n e } } H _ { V } ^ { ( \ell ) } ,\tag{56}
$$

$$
u _ { V } = A _ { \mathrm { r e f i n e } } H _ { V } ^ { ( \ell ) } + \gamma H _ { T } ^ { ( \ell ) } + \rho A _ { \mathrm { r e f i n e } } H _ { T } ^ { ( \ell ) } .\tag{57}
$$

□

Their difference is:

$$
\begin{array} { r l } & { \quad \cdots \quad \cdots } \\ & { \quad u _ { T } - u _ { V } = A _ { \mathrm { r e f i n e } } \big ( H _ { T } ^ { ( \ell ) } - H _ { V } ^ { ( \ell ) } \big ) + \gamma \big ( H _ { V } ^ { ( \ell ) } - H _ { T } ^ { ( \ell ) } \big ) + \rho A _ { \mathrm { r e f i n e } } \big ( H _ { V } ^ { ( \ell ) } - H _ { T } ^ { ( \ell ) } \big ) } \\ & { \quad \quad \quad = A _ { \mathrm { r e f i n e } } \Delta ^ { ( \ell ) } - \gamma \Delta ^ { ( \ell ) } - \rho A _ { \mathrm { r e f i n e } } \Delta ^ { ( \ell ) } } \\ & { \quad \quad \quad = ( 1 - \rho ) A _ { \mathrm { r e f i n e } } \Delta ^ { ( \ell ) } - \gamma \Delta ^ { ( \ell ) } . } \end{array}\tag{58}
$$

Step 2: Apply the Lipschitz bound. Taking the difference of Eqs. (54)–(55):

$$
\Delta ^ { ( \ell + 1 ) } = H _ { T } ^ { ( \ell + 1 ) } - H _ { V } ^ { ( \ell + 1 ) } = \sigma \big ( W ^ { ( \ell ) } u _ { T } \big ) - \sigma \big ( W ^ { ( \ell ) } u _ { V } \big ) .\tag{59}
$$

Since $\sigma ( \cdot )$ is 1-Lipschitz (e.g., ReLU, sigmoid):

$$
\begin{array} { r } { \left\| \Delta ^ { ( \ell + 1 ) } \right\| \leq \left\| W ^ { ( \ell ) } ( u _ { T } - u _ { V } ) \right\| \leq \left\| W ^ { ( \ell ) } \right\| \cdot \| u _ { T } - u _ { V } \| . } \end{array}\tag{60}
$$

Step 3: Bound the pre-activation difference. From Eq. (58) and the triangle inequality:

$$
\| u _ { T } - u _ { V } \| \leq \| ( 1 - \rho ) A _ { \mathrm { r e f i n e } } \Delta ^ { ( \ell ) } \| + \| \gamma \Delta ^ { ( \ell ) } \|
$$

$$
= ( 1 - \rho ) \| A _ { \mathrm { r e f i n e } } \| \| \Delta ^ { ( \ell ) } \| + \gamma \| \Delta ^ { ( \ell ) } \| ,\tag{61}
$$

where $| 1 - \rho | = 1 - \rho$ since $\rho = \sigma ( \rho _ { \mathrm { l o g i t } } ) \in ( 0 , 1 )$

Step 4: Combine and telescope. Substituting Eq. (61) into Eq. (60):

$$
\begin{array} { r } { \left\| \Delta ^ { ( \ell + 1 ) } \right\| \leq \left\| W ^ { ( \ell ) } \right\| \left( ( 1 - \rho ) \| A _ { \mathrm { r e f i n e } } \| + \gamma \right) \left\| \Delta ^ { ( \ell ) } \right\| . } \end{array}\tag{62}
$$

Telescoping over $\ell = 0 , 1 , \ldots , L - 1 ;$

$$
\Bigl \| \Delta ^ { ( L ) } \Bigr \| \le \dot { \prod _ { \ell = 0 } ^ { \cdot } } \bigl \| W ^ { ( \ell ) } \bigr \| \Bigl ( ( 1 - \rho ) \| A _ { \mathrm { r e f i n e } } \| + \gamma \Bigr ) \cdot \ \bigl \| \Delta ^ { ( 0 ) } \bigr \| .\tag{63}
$$

## M Dataset Statistics

Table 8: Dataset statistics. Datasets marked with † are from the MAGB benchmark [34]; those marked with ‡ are from MM-Graph [39]. NC = node classification, LP = link prediction.

<table><tr><td>Dataset</td><td>#Nodes</td><td>#Edges</td><td>#Classes</td><td>Task</td></tr><tr><td>Ele-fashion‡</td><td>97,766</td><td>199,602</td><td>12</td><td>NC</td></tr><tr><td> $\mathrm { G o o d r e a d s } { \mathrm { - N C } } ^ { \ddagger }$ </td><td>685,294</td><td>7,235,084</td><td>10</td><td>NC</td></tr><tr><td>Movies†</td><td>16,672</td><td>218,390</td><td>20</td><td>NC</td></tr><tr><td> $\mathrm { T o y s } ^ { \dagger }$ </td><td>20,695</td><td>126,886</td><td>18</td><td>NC</td></tr><tr><td> $\mathrm { G r o c e r y ^ { \dagger } }$ </td><td>17,074</td><td>171,340</td><td>20</td><td>NC</td></tr><tr><td> $\operatorname { R e d d i t } { - \mathbf { S } ^ { \dagger } }$ </td><td>15,894</td><td>566,160</td><td>20</td><td>NC</td></tr><tr><td> $\mathbf { R e d d i t { – } M } ^ { \dagger }$ </td><td>99,638</td><td>1,167,188</td><td>50</td><td>NC</td></tr><tr><td>Amazon-Sports‡</td><td>50,250</td><td>356,202</td><td>一</td><td>LP</td></tr><tr><td> $\mathrm { \ A m a z o n – C l o t h ^ { \ddagger } }$ </td><td>125,839</td><td>951,271</td><td>一</td><td>LP</td></tr></table>

## N Hyperparameters

We provide the major hyperparameter search space:

• K: {1, 3, 5, 8, 12}.

• Learning rate : {0.005, 0.003, 0.001}.

• Dropout: {0.1, 0.3, 0.5, 0.7, 0.8}.

• GNN layers: {1, 2, 3}.

• GNN hidden dimension of tokens: {256, 512, 1024}.

• Batch Size: {256, 512, 1024}.

• Weight Decay: $\{ 0 , \ 1 e - 0 5 , \ 5 e - 0 5 , \ 1 e - 0 4 \}$