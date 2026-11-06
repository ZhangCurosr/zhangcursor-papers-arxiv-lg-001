# FROM WHAT IS USED TO WHAT IS MISSING: RETHINKING DYNAMIC RANK IN LORA

Peng Xue<sup>1</sup> Guodong Fan<sup>2</sup> Guang-Yong Chen<sup>1</sup> Xing Chen<sup>1</sup>

<sup>1</sup> Fuzhou University

<sup>2</sup> Shandong Technology and Business University

## ABSTRACT

Parameter-efficient fine-tuning with LoRA is governed by rank, which controls parameter budget and locally accessible task updates. Existing dynamic-rank methods mainly focus on capacity allocation, leaving unresolved whether the current adaptation space is sufficient for the task. We characterize the first-order information boundary of standard factor gradients, showing that their unobservable subspace coincides with the normal space of the low-rank manifold. Thus, factor gradients alone cannot distinguish under-optimization within the current space from task-relevant directions beyond its first-order reach. This motivates a geometric view of rank evolution, where tangent information optimizes existing degrees of freedom and the normal spectrum quantifies the value of new ones. Under a local quadratic model, we show that leading normal singular directions yield the locally optimal rank expansion, with spectral energy determining its gain. Building on this characterization, we propose GEAR, which turns the tangent– normal decomposition into rank evolution through tangent-space optimization, matrix-free normal-spectrum estimation for rank expansion, and task-induced geometric initialization. Across language and vision benchmarks, GEAR achieves the best average performance in all evaluated settings with substantially fewer trainable parameters. On LLaMA3-8B, it raises the average commonsense score from 86.61 to 88.45 while reducing trainable parameters from 42.0M to 14.4M.

## 1 INTRODUCTION

Parameter-efficient fine-tuning (PEFT) has become a prevailing approach to adapting large pretrained models by updating only a small fraction of task-specific parameters (Ding et al., 2023b). Among PEFT methods, $\mathrm { L o R A }$ represents the weight increment as a low-rank matrix, $\Delta W = B A$ , where $A \in \mathbb { R } ^ { r \times m }$ and $\boldsymbol { B } \in \mathbb { R } ^ { n \times \hat { \boldsymbol { r } } }$ , reducing the number of trainable parameters from nm to $r ( n + m )$ (Hu et al., 2022). While r therefore directly controls the adaptation budget, it also determines the local first-order update space accessible under the low-rank constraint (Vandereycken, 2013). Rank thus governs both adaptation capacity and which task-induced directions can be expressed. Since these requirements vary across layers and throughout training, a uniform fixed rank is unlikely to match the evolving structure of task demand, motivating dynamic-rank adaptation.

Most dynamic-rank methods still treat rank primarily as a capacity-allocation variable. AdaLoRA, for example, reallocates a prescribed rank budget across weight matrices according to the importance of individual singular components (Zhang et al., 2023), while DyLoRA trains nested low-rank structures to support inference at multiple truncation ranks (Valipour et al., 2023). More recent approaches permit more flexible rank evolution: FlexLoRA uses the spectral entropy of the current low-rank update to determine when capacity should be pruned or expanded (Liu et al., 2026), whereas GeoLoRA combines gradient-driven basis augmentation with spectral truncation within a dynamical low-rank integration framework (Schotthöfer et al., 2025). Although these methods substantially improve the flexibility of rank adaptation, their rank decisions are primarily derived from the capacity or structure already represented by the current adapter. They therefore characterize what the current low-rank space uses, but provide no direct criterion for determining whether task-relevant descent directions remain missing outside that space. This leaves a more fundamental question unresolved:

Is the current adaptation space locally sufficient to express task-relevant descent directions, or must its rank expand to make such directionsfirst-order accessible?

Answering this question requires distinguishing insufficient optimization within the current low-rank space from insufficiency of the space itself. Limited progress may arise for two fundamentally different reasons: the current rank may be sufficient but under-optimized, or task-relevant descent directions may lie outside its first-order reachable space. Only the latter justifies increasing rank. Existing rank-allocation criteria do not explicitly separate these two cases, making it difficult to determine whether additional capacity is genuinely required. This distinction motivates our central view that dynamic rank should be driven not only by what the current adaptation space already represents, but also by what task-relevant descent information remains inaccessible to it.

![](images/e84206c7e1a5af9a20330be9d3b645698c46ccb2f150a6b8a6f57e5267d3bae4.jpg)  
Figure 1: Motivation and overview of GEAR. (a) standard factor-gradient observations are insensitive to normal components, so identical first-order observations can correspond to different rank requirements. (b) GEAR distinguishes optimization within the current manifold from expansion toward task-relevant directions beyond the current first-order reachable space. (c) this geometryguided rank evolution yields a stronger performance–parameter trade-off on commonsense reasoning.

To address this limitation, we propose GEAR (Geometry-guided Evolution of Adaptive Rank), which formulates dynamic-rank adaptation as the evolution of a task-dependent low-rank space. Rather than inferring rank adequacy from existing factor statistics, we characterize the first-order information available under the fixed-rank parameterization. For $X = B A$ , we show that the standard factorgradient map has an observability gap whose null space coincides with the normal space $N _ { X } { \mathcal { M } } _ { r }$ of the fixed-rank manifold ${ \mathcal { M } } _ { r } = \{ X \in \mathbb { R } ^ { n \times m } : { \mathrm { ~ r a n k } } ( X ) = r \}$ This establishes a structural information limitation in which standard factor gradients reveal the tangent component relevant to optimizing the current rank, while leaving the normal component unobservable. Consequently, they cannot distinguish under-optimization within the current adaptation space from task-relevant directions requiring rank expansion to become first-order accessible. For a prescribed rank increase, we further show that leading normal modes determine the locally optimal expansion, while their spectral energy quantifies its gain. Building on this characterization, we propose GEAR, which turns the tangent–normal decomposition into a practical rank-evolution mechanism through tangent space optimization, matrix-free normal-spectrum estimation for rank expansion, and task-induced geometric initialization. By expanding rank only when the estimated normal spectrum reveals valuable task-relevant directions, GEAR improves adaptation effectiveness and parameter efficiency. Across multiple benchmarks, it consistently outperforms competitive PEFT baselines while using fewer trainable parameters. Our contributions are as follows:

• First-order observability barrier for rank adaptation. To our knowledge, we are the first to characterize the information available from standard fixed-rank factor gradients and prove that their unobservable subspace is the normal space of the low-rank manifold, thereby separating current-space optimization from rank-expanding directions.

• Geometry-guided rank evolution. We introduce GEAR, which unifies optimization over the current fixed-rank manifold and rank expansion using tangent information to optimize the adaptation space and the normal spectrum to identify and value new task-relevant directions.

• Excellent performance–parameter trade-off. Across language and vision benchmarks, GEAR achieves the best average performance in every evaluated setting while using substantially fewer trainable parameters than strong PEFT baselines.

## 2 RELATED WORK

LoRA and Adaptive Rank Adaptation. LoRA parameterizes task-specific updates through lowrank factors, reducing trainable parameters and storage while preserving frozen pretrained weights (Hu et al., 2022). Subsequent work has improved low-rank adaptation through refined parameterizations and initialization strategies. DoRA decouples weight magnitude and direction to enhance adaptation flexibility (Liu et al., 2024), while PiSSA initializes adapters from principal singular subspaces of the pretrained weights (Meng et al., 2024) and LoRA-GA uses task gradients to identify more effective initial low-rank directions (Wang et al., 2024). Despite these advances, most methods still operate with a fixed rank, which may poorly match the heterogeneous adaptation demands across layers and training stages. This limitation has motivated adaptive-rank LoRA methods. AdaLoRA dynamically reallocates a prescribed rank budget across modules according to singular-component importance (Zhang et al., 2023); DyLoRA trains nested low-rank structures to support multiple effective ranks (Valipour et al., 2023); and SoRA learns effective rank through sparse gating (Ding et al., 2023a). More recent approaches allow rank to both contract and expand: FlexLoRA adjusts rank according to the spectral structure of the current adapter (Liu et al., 2026), whereas GeoLoRA combines gradient-driven basis augmentation with spectral truncation under a dynamical low-rank formulation (Schotthöfer et al., 2025). GeLoRA further estimates the intrinsic dimensionality of hidden representations to derive adaptive layer-wise ranks (Ed-dib et al., 2025). Collectively, these methods have advanced LoRA from fixed-rank adaptation toward increasingly flexible, taskdependent capacity allocation. However, existing adaptive-rank methods primarily decide how to allocate or reorganize represented low-rank capacity, whereas GEAR asks a different question: whether task-relevant descent information remains outside the current first-order reachable space and therefore warrants new rank dimensions.

Geometric Optimization for LoRA Fixed-rank matrices admit a natural Riemannian manifold structure, providing a principled way to optimize low-rank states independently of a particular factorization (Vandereycken, 2013). Early geometric approaches to LoRA exploit this structure through Riemannian metrics and preconditioning; for example, Riemannian Preconditioned LoRA derives low-dimensional preconditioners from the geometry of the low-rank matrix space to improve optimization stability and convergence (Zhang & Pilanci, 2024). GeoLoRA further introduces dynamical low-rank integration into PEFT, combining gradient-driven basis augmentation with spectral truncation for structure-preserving and rank-adaptive updates (Schotthöfer et al., 2025). Related work has also explored explicit manifold constraints on LoRA factors. Park et al. optimize the B factor on the Stiefel manifold to maintain orthogonality and effective rank (Park et al., 2025), while Manifold-LoRA develops a retraction-free Stiefel-manifold formulation for LoRA fine-tuning (Zhang et al., 2026). More recently, Riemannion directly optimizes the adaptation matrix on the fixed-rank manifold, removing factorization ambiguity and enabling parametrization-independent geometric updates (Bogachev et al., 2026).

These methods establish geometry as an effective tool for improving low-rank optimization, primarily by refining the metric, constraining the factor geometry, or evolving the current low-rank state. GEAR instead focuses on the information required to determine when the current adaptation space itself should expand. From a first-order observability perspective, it relates the normal component unobservable from standard factor gradients to task-relevant directions beyond the current first-order reachable space, thereby connecting manifold optimization with rank expansion.

## 3 METHOD

GEAR builds on a first-order observability gap in fixed-rank adaptation: standard factor gradients characterize optimization within the current space, but not the normal task information that governs its expansion. We first formalize this boundary and then develop a geometry-guided adaptation framework that optimizes the low-rank space while identifying when and how it should expand.

Algorithm 1: GEAR: Geometry-Guided Evolution of Adaptive Rank   
Input: Pretrained weights $W ;$ initial rank $r _ { 0 } ;$ rank upper bound $r _ { \mathrm { m a x } } ;$ audit set $\tau _ { \mathrm { a u d i t } } ;$ budget   
$B _ { \mathrm { t a r g e t } }$   
(U, C, V) ← TGI(W, r );   
X ← UC, W ← W − sX, r ← r<sub>0</sub>;   
for $t = 1 , \dots , T$ do   
$G _ { T } \gets U ( U ^ { \top } G ) + ( I - U U ^ { \top } ) G V V ^ { \top }$ ; // Eq. (1)   
$( X , U , C , V ) \longleftarrow \mathrm { R e t r } _ { \mathcal { M } _ { r } } ( X - \eta _ { t } G _ { T } )$ ; // Eq. (2)   
if $t \in { \mathcal { T } } _ { \mathrm { a u d i t } }$ and $r < r _ { \operatorname* { m a x } }$ then   
$( \widehat { U } , \widehat { \Sigma } , \widehat { V } ) \gets \mathrm { N }$ ormalSketch(X) ; // Eqs. (3)-(4)   
$\mathcal { S }  \mathrm { S e l e c t } ( \widehat { U } , \widehat { \Sigma } , \widehat { V } ; \mathcal { B } _ { \mathrm { t a r g e t } } )$ // Eq. (5)   
X<sub>cand</sub> ← Propose(X, S) ; // Eq. (6)   
if Validate $X _ { \mathrm { c a n d } } , X )$ then   
(U, C, r) ← Expand(U, C, S), X ← UC ; // Eq. (7)   
end   
end   
end   
Output: X

## 3.1 FIRST-ORDER INFORMATION BOUNDARY OF STANDARD FACTOR GRADIENTS

Consider a pretrained weight matrix $W \in \mathbb { R } ^ { n \times m }$ adapted as

$$
W ^ { \prime } = W + s X , \qquad X = B A ,
$$

where $s > 0$ is a fixed scaling factor. Let $f ( X ) = { \mathcal { L } } ( W + s X )$ and $G = \nabla _ { X } f ( X )$ . For a rank-r adaptation matrix $X = B A$ , we analyze $X$ on the fixed-rank manifold $\mathcal { M } _ { r } .$ , whose tangent space characterizes the first-order directions accessible without increasing rank (Vandereycken, 2013).

Let $X = U \Sigma V ^ { \top }$ be the compact SVD of X, and define $P _ { U } = U U ^ { \top }$ and $P _ { V } = V V ^ { \top }$ . The full gradient admits the orthogonal decomposition

$$
G = G _ { T } + G _ { N } ,
$$

where $G _ { T } = P _ { U } G + ( I - P _ { U } ) G P _ { V } \in T _ { X } { \cal M } _ { r }$ and $G _ { N } = ( I - P _ { U } ) G ( I - P _ { V } ) \in N _ { X } \mathcal { M } _ { \tau }$ . Under the factorized parameterization $X = B A$ , the corresponding gradients are

$$
\nabla _ { B } f = G \boldsymbol { A } ^ { \intercal } , \qquad \nabla _ { A } f = \boldsymbol { B } ^ { \intercal } \boldsymbol { G } .
$$

We therefore define the factor-gradient observation operator

$$
\mathcal { O } _ { A , B } ( G ) = \big ( G A ^ { \top } , B ^ { \top } G \big ) ,
$$

which captures the first-order information about $G$ available from standard factor backpropagation. Theorem 1 (Factor-Gradient Observability Barrier). Let $A \in \mathbb { R } ^ { r \times m }$ and $B \in \mathbb { R } ^ { n \times r }$ be full rank, with $X = B A \in { \mathcal { M } } _ { r } .$ . Then

$$
\ker { \mathcal { O } } _ { A , B } = N _ { X } { \mathcal { M } } _ { r } .
$$

Equivalently, for any $G _ { 1 }$ and $G _ { 2 }$ ,

$$
\mathcal { O } _ { A , B } ( G _ { 1 } ) = \mathcal { O } _ { A , B } ( G _ { 2 } )
$$

if and only $i f ( G _ { 1 } ) _ { T } = ( G _ { 2 } ) _ { T }$

The theorem establishes a precise first-order observability boundary, showing that standard factor gradients preserve the tangent component of the task gradient but cannot distinguish gradients that differ only in their normal components. A complete proof is provided in Appendix C.1. Consequently, the same factor-gradient observation may correspond either to a gradient fully represented by the current adaptation space or to one containing additional task-relevant directions beyond its firstorder reach. This ambiguity arises from the information available under the current fixed-rank parameterization, rather than from factorization itself.

This result distinguishes optimization within the current adaptation space from the need for rank expansion. The tangent component represents directions that are first-order accessible at rank $r ,$ whereas the normal component captures task-relevant directions beyond the current space. Accordingly, tangent information is used to optimize the existing degrees of freedom, while normal information provides the relevant signal for assessing the value of rank expansion. The subsequent sections develop these two mechanisms, respectively.

## 3.2 FIXED-RANK MANIFOLD OPTIMIZATION

For a fixed rank $r ,$ GEAR performs optimization on the manifold $\mathcal { M } _ { \tau }$ using the tangent component of the gradient. We represent the adaptation matrix as

$$
\begin{array} { r } { \boldsymbol { X } = \boldsymbol { U } \boldsymbol { C } , \qquad \boldsymbol { U } ^ { \top } \boldsymbol { U } = \boldsymbol { I } _ { r } , } \end{array}
$$

where $U$ is an orthonormal basis for the column space of $X$ , and $C$ contains the corresponding low-rank coefficients. Let $V$ be an orthonormal basis for the column space of $C ^ { \top }$ . The tangent gradient is then

$$
G _ { T } = U ( U ^ { \top } G ) + ( I - U U ^ { \top } ) G V V ^ { \top } .\tag{1}
$$

Equation (1) shows that $G _ { T }$ can be computed from the two gradient contractions $U ^ { \top } G$ and $G V$ avoiding explicit construction of the full gradient $G .$

To obtain these quantities directly through backpropagation, GEAR uses the forward-equivalent representation

$$
B _ { \mathrm { e x t } } = [ U , 0 ] , \qquad A _ { \mathrm { e x t } } = { \biggl [ } V ^ { \top } { \biggr ] } ,
$$

which satisfies $B _ { \mathrm { e x t } } A _ { \mathrm { e x t } } = X$ . Although this representation leaves the forward mapping unchanged, its backward derivatives expose the required gradient contractions:

$$
\nabla _ { C } f = U ^ { \top } G , \qquad \nabla _ { B _ { \mathrm { e x t } } ^ { ( 2 ) } } f = G V .
$$

Thus, the tangent gradient can be recovered without explicitly materializing G. GEAR then performs a tangent update followed by retraction onto the fixed-rank manifold,

$$
X ^ { - } = X - \eta G _ { T } , \qquad X _ { t + 1 } = \mathrm { R e t r } _ { { \mathcal { M } _ { r } } } ( X ^ { - } ) .\tag{2}
$$

Since $X ^ { - }$ has rank at most $2 r ,$ the retraction is performed on its compact factors rather than on the full matrix. Specifically, thin QR factorizations reduce the update to a small core matrix, whose truncated SVD yields the retracted rank-r state together with the updated left and right subspace bases. Full details are provided in Appendix B.1.

## 3.3 GEOMETRY-GUIDED RANK EVOLUTION

The preceding section optimizes the current adaptation state on $\mathcal { M } _ { r }$ . Rank expansion, by contrast, is governed by the normal component $G _ { N }$ , which captures task-relevant directions beyond the current first-order reachable space. We next quantify the local value of these directions for rank expansion. Let

$$
G _ { N } = \sum _ { i \geq 1 } \sigma _ { i } u _ { i } v _ { i } ^ { \top } , \qquad \sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdots ,
$$

and denote its best rank-k approximation by $( G _ { N } ) _ { k }$

Theorem 2 (Local Optimality of Rank Expansion). Let $X \in \mathcal { M } _ { r } ,$ , and let $\mathcal { C } _ { r , k } ( X )$ denote the tangent cone at X to the bounded-rank set $\mathcal { V } _ { r + k } = \bar  \{ Y \in \mathbb { R } ^ { n \times m }$ : rank $( Y ) \leq r + k \}$ . For the local quadratic model $\begin{array} { r } { q _ { \eta } ( H ) = \langle G , H \rangle _ { F } + \frac { 1 } { 2 \eta } \| H \| _ { F } ^ { 2 } , } \end{array}$ , an optimal direction over ${ \mathcal { C } } _ { r , k } ( X )$ is

$$
H _ { k } ^ { \star } = - \eta \left[ G _ { T } + ( G _ { N } ) _ { k } \right] ,
$$

where $( G _ { N } ) _ { k }$ is a best rank-k approximation of $G _ { N }$ . Moreover, the optimal model decrease enabled by the additional k rank dimensions, relative to optimization over $T _ { X } { \mathcal { M } } _ { r }$ , is

$$
\Delta _ { k } : = \operatorname* { m i n } _ { H \in T _ { X } \mathcal { M } _ { r } } q _ { \eta } ( H ) - \operatorname* { m i n } _ { H \in \mathcal { C } _ { r , k } ( X ) } q _ { \eta } ( H ) = \frac { \eta } { 2 } \sum _ { i = 1 } ^ { k } \sigma _ { i } ^ { 2 } ( G _ { N } ) .
$$

A complete proof is provided in Appendix C.2. The result shows that the leading singular directions of the normal gradient constitute the locally optimal directions for rank expansion, while their spectral energy exactly quantifies the additional decrease attainable under the local quadratic model.

However, constructing $G _ { N } = ( I - U U ^ { \top } ) G ( I - V V ^ { \top } )$ requires the full gradient G. GEAR instead estimates its dominant spectrum through randomized probing (Halko et al., 2011). Given $\Omega \in \mathbb { R } ^ { m \times h }$ , with $h \ll \operatorname* { m i n } \{ n , m \}$ , we project it onto the orthogonal complement of the right subspace, $\Omega _ { \perp } ~ = ~ ( I - V V ^ { \top } ) \Omega$ We introduce a zero-initialized auxiliary variable $P$ through $\Delta X _ { \mathrm { p r o b e } } = P \Omega _ { \textrm { l } } ^ { \top }$ . Since $P = 0$ , this leaves the forward mapping unchanged, while backpropagation yields $\nabla _ { P } f = \bar { G } \Omega _ { \perp }$ . Projecting onto the orthogonal complement of the left subspace gives

$$
\begin{array} { r } { Y = ( I - U U ^ { \top } ) \nabla _ { P } f = G _ { N } \Omega . } \end{array}\tag{3}
$$

Let $Q = \operatorname { o r t h } ( Y )$ . If $G _ { N } = U _ { N } \Sigma _ { N } V _ { N } ^ { \top }$ , then $Y = U _ { N } \Sigma _ { N } ( V _ { N } ^ { \top } \Omega )$ , so singular directions associated with larger singular values contribute more strongly to $Y .$ . Hence, span(Q) captures the dominant left singular subspace of $G _ { N }$ . Applying the same zero-output probe yields $Q ^ { \dagger } G$ , from which we compute

$$
\boldsymbol { Z } _ { N } = \boldsymbol { Q } ^ { \top } \boldsymbol { G } ( \boldsymbol { I } - \boldsymbol { V } \boldsymbol { V } ^ { \top } ) = \boldsymbol { Q } ^ { \top } \boldsymbol { G } _ { N } = \widetilde { \boldsymbol { U } } \widehat { \boldsymbol { \Sigma } } \widehat { \boldsymbol { V } } ^ { \top } .\tag{4}
$$

The leading normal components are therefore approximated by $\widehat { u } _ { i } = Q \widetilde { u } _ { i } , \widehat { v } _ { i } = \widehat { V } _ { : , i }$ , and $\widehat { \sigma } _ { i } = \widehat { \Sigma } _ { i i }$ without explicitly materializing $G _ { N }$ . The resulting approximation has a controlled effect on the local value of rank expansion.

Theorem 3 (Stability under Inexact Normal Estimation). Let $\widehat { G } _ { N } \in N _ { X } \mathcal { M } _ { r } s a t i s f y \| \widehat { G } _ { N } - G _ { N } \| _ { F } \leq$ $\varepsilon ,$ and let $\tau _ { k } = \| G _ { N } - ( G _ { N } ) _ { k } \| _ { F }$ . Then the rank-k expansion based on $( \widehat { G } _ { N } ) _ { k }$ satisfies

$$
0 \leq q _ { \eta } \Bigl ( - \eta ( \widehat { G } _ { N } ) _ { k } \Bigr ) - q _ { \eta } \bigl ( - \eta ( G _ { N } ) _ { k } \bigr ) \leq 2 \eta \varepsilon ( \tau _ { k } + \varepsilon ) .
$$

A complete proof is provided in Appendix C.3. GEAR then uses the estimated normal spectrum to allocate rank under a global parameter budget. Motivated by Theorem 2, we score each candidate normal direction by its estimated local gain per added parameter and impose a global rank budget:

$$
a _ { \ell , i } = \frac { \widehat { \sigma } _ { \ell , i } ^ { 2 } } { n _ { \ell } + m _ { \ell } } , \qquad \mathcal { B } _ { \mathrm { t a r g e t } } = \bar { r } _ { \mathrm { b u d g e t } } \sum _ { \ell = 1 } ^ { L } \left( n _ { \ell } + m _ { \ell } \right) .\tag{5}
$$

Here, $r _ { \ell }$ denotes the current rank of module $\ell , n _ { \ell } + m _ { \ell }$ is the cost of one additional rank dimension in module ℓ, and $\bar { r } _ { \mathrm { b u d g e t } }$ denotes the weighted-average rank budget. Candidates are selected according to $a \ell , i$ subject to the global budget and the per-module rank limit. For $k _ { \ell }$ selected directions, let $\widehat { U } _ { \ell , k _ { \ell } } = [ \widehat { u } _ { \ell , 1 } , \dots , \widehat { u } _ { \ell , k _ { \ell } } ] , \widehat { V } _ { \ell , k _ { \ell } } = [ \widehat { v } _ { \ell , 1 } , \dots , \widehat { v } _ { \ell , k _ { \ell } } ]$ , and $\widehat { \Sigma } _ { \ell , k _ { \ell } } = \mathrm { d i a g } ( \widehat { \sigma } _ { \ell , 1 } , \dots , \widehat { \sigma } _ { \ell , k _ { \ell } } )$ . They define the candidate expansion

$$
\begin{array} { r } { X _ { \ell , \mathrm { c a n d } } = X _ { \ell } - \eta _ { b } \widehat { U } _ { \ell , k _ { \ell } } \widehat { \Sigma } _ { \ell , k _ { \ell } } \widehat { V } _ { \ell , k _ { \ell } } ^ { \top } . } \end{array}\tag{6}
$$

We validate $X _ { \ell , \mathrm { c a n d } }$ on a small set of separate training microbatches shared across candidate variants and accept the expansion only if its mean loss improves by more than a relative margin $\rho ;$ full details are provided in Appendix B.2.

For an accepted expansion, $\widehat { U } _ { \ell , k _ { \ell } } \perp U _ { \ell }$ and $\widehat { V } _ { \ell , k _ { \ell } } \perp V _ { \ell } ,$ , so the low-rank state is updated directly as

$$
\begin{array} { r } { U _ { \ell } ^ { + } = [ U _ { \ell } , \widehat { U } _ { \ell , k _ { \ell } } ] , \qquad C _ { \ell } ^ { + } = \left[ \widehat { \mathbf { \Omega } } _ { - \eta _ { b } } ^ { C _ { \ell } } \widehat { \Sigma } _ { \ell , k _ { \ell } } ^ { \top } \right] , \qquad X _ { \ell } ^ { + } = U _ { \ell } ^ { + } C _ { \ell } ^ { + } . } \end{array}\tag{7}
$$

The accepted expansion moves the adaptation state from $\mathcal { M } _ { r _ { \ell } }$ to $\mathcal { M } _ { \boldsymbol { r } _ { \ell } + k _ { \ell } }$ . Optimizer states associated with existing components are inherited, while those of newly introduced directions are initialized to zero. Subsequent updates then follow the fixed-rank manifold optimization in Section 3.2.

## 3.4 TASK-INDUCED GEOMETRIC INITIALIZATION (TGI)

Rank inadequacy is defined relative to the current adaptation space. To avoid conflating random subspace mismatch with genuine missing directions, GEAR initializes this space from task-gradient structure estimated on a small calibration set. Let $G _ { 0 } = \nabla _ { X } f ( 0 )$ . In a left-probing stage, set $X = P \Omega ^ { \top }$ with $P = 0$ and fixed $\Omega \in \mathbb { R } ^ { m \times h }$ . This preserves the pretrained mapping, while backpropagation yields $Z _ { 0 } = \left. \nabla _ { P } f ( P \Omega ^ { \top } ) \right| _ { P = 0 } = G _ { 0 } \Omega \stackrel { \cdot } { \in } \mathbb { R } ^ { n \times h }$ without forming $G _ { 0 }$

To express the normalization for both stages uniformly, write $Z _ { 0 } \in \mathbb { R } ^ { d \times h }$ , with $d = n$ for the left probe and $d = m$ for the right probe. Set $\overset { \cdot } { M } = Z _ { 0 } ^ { \odot 2 }$ and $\mu = ( d h ) ^ { - 1 } \textstyle \sum _ { i , a } M _ { i a }$ . To prevent large gradient scales from dominating subspace estimation, we normalize the sketch along both dimensions:

$$
d _ { i } ^ { \mathrm { r o w } } = \frac { 1 } { h } \sum _ { a } M _ { i a } + \lambda \mu , \quad d _ { a } ^ { \mathrm { c o l } } = \frac { 1 } { d } \sum _ { i } M _ { i a } + \lambda \mu , \quad \widetilde { Z } _ { 0 } = D _ { \mathrm { r o w } } ^ { - 1 / 2 } Z _ { 0 } D _ { \mathrm { c o l } } ^ { - 1 / 2 } ,
$$

where $D _ { \mathrm { r o w } } = \mathrm { d i a g } ( d _ { i } ^ { \mathrm { r o w } } )$ and $D _ { \mathrm { c o l } } = \mathrm { d i a g } ( d _ { a } ^ { \mathrm { c o l } } )$ . A thin QR factorization of the left-stage $\widetilde { Z } _ { 0 }$ yields $Q _ { L } \in \mathbb { R } ^ { n \times h }$ . Holding $Q _ { L }$ fixed, a zero-initialized right probe $X = Q _ { L } R$ gives the transposed sketch $Z _ { 0 } = \left( \left. \nabla _ { R } f ( Q _ { L } R ) \right| _ { R = 0 } \right) ^ { \top } = G _ { 0 } ^ { \top } Q _ { L } \in \mathbb { R } ^ { m \times h }$ . Non-final right stages use QR to update the next left-probing basis. TGI performs $p + 1$ left–right rounds, totaling $2 ( p + 1 )$ stages. Additional rounds can refine the dominant-subspace estimate by probing with successively updated bases; we use $p = 1$ in our experiments.

At the final right stage, the normalized sketch $\widetilde { Z } _ { 0 } \in \mathbb { R } ^ { m \times h }$ gives $\bar { Z } _ { 0 } = \widetilde { Z } _ { 0 } ^ { \top } = \widehat { U } \Sigma V ^ { \top }$ . Retaining the leading $r _ { 0 } \leq h$ components, we initialize

$$
U _ { 0 } = Q _ { L } \widehat U _ { ; , 1 : r _ { 0 } } , \qquad C _ { 0 } = \mathrm { d i a g } ( \gamma _ { 1 } , \ldots , \gamma _ { r _ { 0 } } ) V _ { ; , 1 : r _ { 0 } } ^ { \top } , \qquad X _ { 0 } = U _ { 0 } C _ { 0 } .\tag{8}
$$

Here $\gamma _ { i }$ is the clipped ratio of a retained singular value to the mean of the retained values. Since $X _ { 0 }$ is generally nonzero, we compensate its initial contribution in the frozen weight:

$$
\widetilde { W } = W - s X _ { 0 } , \qquad \widetilde { W } + s X _ { 0 } = W .
$$

Thus, training starts from a task-aligned adaptation space while preserving the pretrained forward mapping at initialization.

## 4 EXPERIMENTAL RESULTS

We evaluate GEAR across commonsense reasoning, natural language understanding, and visual recognition benchmarks, considering both predictive performance and parameter efficiency. We compare against representative fixed-rank and adaptive-rank PEFT methods, including LoRA (Hu et al., 2022), AdaLoRA (Zhang et al., 2023), Riemannion (Bogachev et al., 2026), GeoLoRA (Schotthöfer et al., 2025), and FlexLoRA (Liu et al., 2026). All methods use consistent experimental settings for fair comparison. Ablation studies are provided in Appendix E.

## 4.1 RANK AUDIT SCHEDULE

To ensure stable rank adaptation, GEAR performs rank-growth audits periodically during early-tomiddle training and freezes the rank thereafter. Let $t _ { \mathrm { s t a r t } }$ denote the first audit step, ∆t the audit interval, and $t _ { \mathrm { f i n a l } }$ the step after which rank expansion stops. The audit steps are defined as

$$
\mathcal { T } _ { \mathrm { a u d i t } } = \left\{ t _ { \mathrm { s t a r t } } + k \Delta t \vert k \in \mathbb { Z } _ { \ge 0 } , t _ { \mathrm { s t a r t } } + k \Delta t \le t _ { \mathrm { f i n a l } } \right\} .
$$

At each audit, GEAR estimates task-relevant directions beyond the current adaptation space and evaluates them as candidates for rank expansion. After $t _ { \mathrm { f i n a l } }$ , the rank remains fixed and training proceeds solely with manifold optimization at the resulting rank.

## 4.2 COMMONSENSE REASONING

We evaluate GEAR on eight commonsense reasoning benchmarks, including BoolQ, PIQA, SIQA, and HellaSwag (Clark et al., 2019; Bisk et al., 2020; Sap et al., 2019; Zellers et al., 2019), together with WinoGrande, ARC-e/ARC-c, and OBQA (Sakaguchi et al., 2020; Clark et al., 2018; Mihaylov et al., 2018), using LLaMA3-8B (Grattafiori et al., 2024) and Qwen2.5-7B (Qwen et al., 2025) as backbone models. The reference rank is set to 16, while GEAR initializes at $r _ { 0 } = 4$ , with a weighted-average rank budget of 16 and a per-module upper bound $r _ { \operatorname* { m a x } } = 3 2$ ; the final ranks are determined by the accepted expansions rather than the upper bound. Performance is reported together with the number of trainable parameters to evaluate both adaptation quality and parameter efficiency.

As shown in Table 1, GEAR achieves the highest average performance on both backbones with substantially fewer trainable parameters. On LLaMA3-8B, it reaches 88.45, outperforming FlexLoRA by 1.84 points with 14.4M versus 42.0M parameters; on Qwen2.5-7B, it attains 89.40 with 14.0M parameters, versus 87.65 with 40.3M for FlexLoRA. Unlike adaptive-rank methods that rely mainly on statistics of the current low-rank representation, GEAR expands the adaptation space according to task-relevant normal directions while optimizing accessible directions on the current manifold, yielding a better performance–parameter trade-off across both backbones.

Table 1: Commonsense reasoning performance on LLaMA3-8B and Qwen2.5-7B. We report the mean over three runs with different random seeds. Best results are shown in bold.
<table><tr><td>Method (Params)</td><td>BoolQ</td><td></td><td>PIQA SIQA</td><td>HellaS.</td><td>WinoG.</td><td></td><td>ARC-e ARC-c</td><td>OBQA</td><td>Avg.</td></tr><tr><td>LLaMA3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA (41.9M)</td><td>72.37</td><td>87.83</td><td>81.34</td><td>94.81</td><td>85.08</td><td>91.46</td><td>81.32</td><td>86.00</td><td>85.03</td></tr><tr><td>AdaLoRA (31.6M)</td><td>72.23</td><td>87.21</td><td>81.27</td><td>94.60</td><td>85.16</td><td>93.18</td><td>82.68</td><td>85.80</td><td>85.27</td></tr><tr><td>Riemannion (41.9M)</td><td>72.45</td><td>87.32</td><td>81.17</td><td>94.95</td><td>87.64</td><td>93.14</td><td>82.51</td><td>87.40</td><td>85.82</td></tr><tr><td>GeoLoRA (41.3M)</td><td>72.84</td><td>87.54</td><td>79.99</td><td>94.62</td><td>85.56</td><td>93.52</td><td>83.96</td><td>87.00</td><td>85.63</td></tr><tr><td>FlexLoRA (42.0M)</td><td>72.91</td><td>88.30</td><td>82.96</td><td>95.68</td><td>87.14</td><td>93.56</td><td>83.70</td><td>88.60</td><td>86.61</td></tr><tr><td>GEAR (14.4M)</td><td>75.19</td><td>90.81</td><td>83.78</td><td>96.82</td><td>89.82</td><td>93.77</td><td>85.58</td><td>91.80</td><td>88.45</td></tr><tr><td>Qwen2.5-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA (40.4M)</td><td>71.54</td><td>88.52</td><td>78.73</td><td>94.05</td><td>82.56</td><td>96.13</td><td>90.00</td><td>90.12</td><td>86.46</td></tr><tr><td>AdaLoRA (35.2M)</td><td>71.59</td><td>89.12</td><td>78.92</td><td>94.06</td><td>82.79</td><td>96.00</td><td>90.27</td><td>89.80</td><td>86.57</td></tr><tr><td>Riemannion (40.4M)</td><td>71.68</td><td>89.17</td><td>79.32</td><td>94.72</td><td>84.53</td><td>95.96</td><td>90.53</td><td>91.00</td><td>87.11</td></tr><tr><td>GeoLoRA (40.1M)</td><td>71.90</td><td>89.12</td><td>79.02</td><td>94.88</td><td>85.00</td><td>96.21</td><td>89.93</td><td>91.00</td><td>87.13</td></tr><tr><td>FlexLoRA (40.3M)</td><td>72.29</td><td>89.17</td><td>79.68</td><td>95.19</td><td>86.03</td><td>96.21</td><td>90.19</td><td>92.40</td><td>87.65</td></tr><tr><td>GEAR (14.0M)</td><td>75.51</td><td>90.86</td><td>81.58</td><td>96.34</td><td>90.53</td><td>96.80</td><td>89.76</td><td>93.80</td><td>89.40</td></tr></table>

Table 2: GLUE performance with DeBERTa-v3-base. We report the mean over three runs with different random seeds. Best results are shown in bold.
<table><tr><td>Method (Params)</td><td>CoLA Mcc.</td><td>MNLI Acc.</td><td>MRPC Acc.</td><td>RTE Acc.</td><td>QNLI Acc.</td><td>SST-2 Acc.</td><td>STS-B Corr.</td><td>QQP Acc.</td><td>Avg.</td></tr><tr><td>LoRA (1.9M)</td><td>68.31</td><td>89.74</td><td>90.19</td><td>84.16</td><td>93.61</td><td>95.30</td><td>85.20</td><td>90.35</td><td>87.11</td></tr><tr><td>AdaLoRA (1.9M)</td><td>70.07</td><td>89.93</td><td>90.44</td><td>88.17</td><td>94.27</td><td>95.13</td><td>91.21</td><td>91.41</td><td>88.83</td></tr><tr><td>Riemannion (1.9M)</td><td>67.41</td><td>88.93</td><td>89.01</td><td>85.13</td><td>92.64</td><td>94.00</td><td>90.12</td><td>87.92</td><td>86.90</td></tr><tr><td>GeoLoRA (1.9M)</td><td>68.18</td><td>89.67</td><td>87.76</td><td>87.73</td><td>92.93</td><td>95.52</td><td>90.15</td><td>87.77</td><td>87.46</td></tr><tr><td>FlexLoRA (1.9M)</td><td>70.22</td><td>90.13</td><td>90.67</td><td>88.45</td><td>94.24</td><td>95.34</td><td>91.53</td><td>90.27</td><td>88.85</td></tr><tr><td>GEAR (1.5M)</td><td>70.85</td><td>90.97</td><td>92.46</td><td>89.53</td><td>94.27</td><td>96.22</td><td>91.83</td><td>90.88</td><td>89.63</td></tr></table>

## 4.3 NATURAL LANGUAGE UNDERSTANDING

We evaluate GEAR with DeBERTa-v3-base (He et al., 2023) on the GLUE benchmark (Wang et al., 2019), covering eight tasks with linguistic objectives, including acceptability, inference, paraphrase detection, semantic similarity, and sentiment classification. The reference rank is set to 8, while GEAR initializes at $r _ { 0 } = 4 \mathrm { w i t h } r _ { \mathrm { m a x } } = 1 6$ as an upper bound; the final rank is determined by the accepted expansions rather than the limit. We report the task-specific metrics and their average.

As shown in Table 2, GEAR achieves the best average score of 89.63 with only 1.5M trainable parameters, improving over FlexLoRA by 0.78 points while using about 21% fewer parameters. GEAR attains the strongest result on seven of the eight tasks, with particularly clear gains on MRPC, RTE, and STS-B. The improvement across heterogeneous objectives suggests that the benefit of GEAR is not tied to a specific task structure. By expanding the adaptation space according to task-relevant directions beyond its current first-order reach, GEAR can introduce additional capacity more selectively across diverse tasks.

Table 3: VTAB-1K performance with ViT-B/16. We report the mean over three runs with different random seeds. Best results are shown in bold.
<table><tr><td>Dataset</td><td>LoRA (0.3M)</td><td>AdaLoRA (0.3M)</td><td>Riemannion (0.3M)</td><td>GeoLoRA (0.3M)</td><td>FlexLoRA (0.3M)</td><td>GEAR (0.2M)</td></tr><tr><td>Natural</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CIFAR100</td><td>69.39</td><td>71.81</td><td>70.78</td><td>68.05</td><td>71.30</td><td>71.88</td></tr><tr><td>Caltech101</td><td>90.60</td><td>89.96</td><td>91.03</td><td>88.87</td><td>91.72</td><td>91.78</td></tr><tr><td>DTD</td><td>71.01</td><td>71.76</td><td>71.28</td><td>68.88</td><td>70.18</td><td>71.60</td></tr><tr><td>Flowers</td><td>98.80</td><td>99.04</td><td>99.09</td><td>99.06</td><td>98.98</td><td>99.12</td></tr><tr><td>Pets</td><td>89.89</td><td>91.58</td><td>91.85</td><td>91.71</td><td>91.14</td><td>91.93</td></tr><tr><td>SVHN</td><td>90.53</td><td>90.59</td><td>89.08</td><td>91.39</td><td>90.25</td><td>91.89</td></tr><tr><td>SUN397</td><td>53.78</td><td>55.61</td><td>55.12</td><td>54.63</td><td>54.14</td><td>55.03</td></tr><tr><td colspan="7">Specialized</td></tr><tr><td>Camelyon</td><td>83.26</td><td>81.92</td><td>81.60</td><td>82.16</td><td>83.98</td><td>83.93</td></tr><tr><td>EuroSAT</td><td>93.74</td><td>94.04</td><td>93.13</td><td>94.61</td><td>93.91</td><td>94.91</td></tr><tr><td>RESISC45</td><td>85.40</td><td>85.37</td><td>84.30</td><td>84.52</td><td>85.41</td><td>85.98</td></tr><tr><td>Retinopathy</td><td>73.60</td><td>75.40</td><td>75.16</td><td>75.41</td><td>74.13</td><td>74.91</td></tr><tr><td colspan="7">Structured</td></tr><tr><td>CLEVR-C</td><td>80.91</td><td>81.23</td><td>78.95</td><td>80.63</td><td>81.30</td><td>82.07</td></tr><tr><td>CLEVR-D</td><td>64.74</td><td>65.38</td><td>68.49</td><td>63.37</td><td>63.56</td><td>68.51</td></tr><tr><td>DMLab</td><td>50.04</td><td>50.23</td><td>49.95</td><td>50.08</td><td>52.02</td><td>52.18</td></tr><tr><td>KITTI-D</td><td>77.87</td><td>78.34</td><td>75.81</td><td>78.14</td><td>78.95</td><td>78.90</td></tr><tr><td>dSpr-Loc</td><td>75.76</td><td>73.38</td><td>72.40</td><td>73.62</td><td>75.49</td><td>76.78</td></tr><tr><td>dSpr-Ori</td><td>47.65</td><td>49.32</td><td>48.71</td><td>48.27</td><td>49.02</td><td>49.52</td></tr><tr><td>sNORB-Azi</td><td>29.49</td><td>28.00</td><td>27.76</td><td>30.25</td><td>29.84</td><td>31.32</td></tr><tr><td>sNORB-Ele</td><td>41.93</td><td>37.56</td><td>39.58</td><td>40.99</td><td>43.17</td><td>43.89</td></tr><tr><td>Average</td><td>72.02</td><td>72.13</td><td>71.79</td><td>71.82</td><td>72.55</td><td>73.48</td></tr></table>

## 4.4 VISUAL TASK

To assess the generality of GEAR beyond NLP, we evaluate it with ViT-B/16 (Dosovitskiy et al., 2021) on the Visual Task Adaptation Benchmark (VTAB-1K) (Zhai et al., 2020), which covers 19 recognition tasks grouped into Natural, Specialized, and Structured domains. We use the same rank configuration as in the NLU experiments.

As shown in Table 3, GEAR achieves the highest overall average of 73.48 with only 0.2M trainable parameters, compared with 72.55 for FlexLoRA using 0.3M. GEAR obtains the best result on 14 of the 19 tasks, including 7 of the 8 Structured tasks. The gains across visually and semantically diverse domains indicate that the proposed rank-evolution mechanism is not tied to a particular modality or task family. By adapting the low-rank space according to task-relevant directions beyond its current first-order reach, GEAR maintains strong transfer performance while introducing fewer additional degrees of freedom.

Across all evaluated settings, GEAR achieves the strongest average performance with fewer trainable parameters than the strongest adaptive-rank baseline. These gains span two language-model backbones, eight GLUE tasks, and 19 VTAB-1K tasks, indicating robustness across backbones, task families, and modalities. Importantly, they arise from adaptively selected final ranks rather than simply increasing the rank budget, supporting the hypothesis that identifying valuable directions outside the current adaptation space enables more efficient rank growth.

## 5 CONCLUSION

We presented GEAR, a geometry-guided framework for dynamic low-rank adaptation that shifts the focus from capacity allocation to the sufficiency of the current adaptation space. We established a first-order information boundary for standard factor gradients, showing that task-relevant normal information is unobservable from the current fixed-rank parameterization, and further related the normal spectrum to the locally optimal value of rank expansion. Building on this analysis, GEAR combines fixed-rank manifold optimization, efficient normal-spectrum estimation, and task-induced geometric initialization to adapt rank according to task-relevant directions beyond the current firstorder reachable space. Across commonsense reasoning, GLUE, and VTAB-1K, GEAR achieves the highest average performance in all evaluated settings while using substantially fewer trainable parameters than strong adaptive-rank baselines. Overall, our results suggest that dynamic rank is most effectively viewed not as a budget-allocation problem alone, but as the controlled evolution of the adaptation space in response to previously unrepresented task information.

## AI USE STATEMENT

Generative AI tools were used to assist with language editing, LAT<sub>E</sub>X formatting, and the examination of mathematical derivations. All AI-assisted suggestions were independently reviewed and verified by the authors, who take full responsibility for the final content of this work.

## REPRODUCIBILITY STATEMENT

We provide complete proofs of all theoretical results and detailed algorithm descriptions in the appendix. Experimental configurations, hyperparameters, dataset protocols, baseline settings, and additional ablations are also provided to facilitate reproducibility. All reported results are averaged over three random seeds.

## REFERENCES

Roy Bar-Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second PASCAL recognising textual entailment challenge. In Proceedings ofthe Second PASCAL Challenges Workshop on Recognising Textual Entailment, pp. 1–9, 2006.

Luisa Bentivogli, Bernardo Magnini, Ido Dagan, Hoa Trang Dang, and Danilo Giampiccolo. The fifth PASCAL recognizing textual entailment challenge. In Text Analysis Conference, 2009.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 7432–7439, 2020. doi: 10.1609/aaai.v34i05.6239.

Vladimir Bogachev, Vladimir Aletov, Alexander Molozhavenko, Denis Bobkov, Vera Soboleva, Aibek Alanov, and Maxim Rakhuba. LoRA meets Riemannion: Muon optimizer for parametrizationindependent low-rank adapters. In International Conference on Learning Representations, 2026.

Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings ofthe 11th International Workshop on Semantic Evaluation, pp. 1–14. Association for Computational Linguistics, 2017. doi: 10.18653/v1/S17-2001.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2924–2936. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1300.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge, 2018.

Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges: Evaluating Predictive Uncertainty, Visual Object Classification, and Recognising Textual Entailment, volume 3944 of Lecture Notes in Computer Science, pp. 177–190. Springer, 2006. doi: 10.1007/11736790\_9.

Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, and Maosong Sun. Sparse low-rank adaptation of pre-trained language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4133–4145. Association for Computational Linguistics, 2023a. doi: 10.18653/v1/2023.emnlp-main.252.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong Sun. Parameter-efficient finetuning of large-scale pre-trained language models. Nature Machine Intelligence, 5:220–235, 2023b. doi: 10.1038/s42256-023-00626-4.

William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Proceedings ofthe Third International Workshop on Paraphrasing, 2005.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=YicbFdNTTy.

Abdessalam Ed-dib, Zhanibek Datbayev, and Amine M. Aboussalah. GeLoRA: Geometric adaptive ranks for efficient LoRA fine-tuning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 25174–25196. Association for Computational Linguistics, 2025. doi: 10.18653/ v1/2025.findings-emnlp.1372.

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, pp. 1–9. Association for Computational Linguistics, 2007.

Aaron Grattafiori et al. The llama 3 herd of models, 2024.

Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53 (2):217–288, 2011. doi: 10.1137/090771806.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=sE7-XhLxHA.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, and Roy Lee. LLM-adapters: An adapter family for parameter-efficient fine-tuning of large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5254–5276. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.319.

Muqing Liu, Chongjie Si, and Yuheng Jia. FlexLoRA: Entropy-guided flexible low-rank adaptation. In International Conference on Learning Representations, 2026.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. DoRA: Weight-decomposed low-rank adaptation. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 32100–32121. PMLR, 2024.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. PiSSA: Principal singular values and singular vectors adaptation of large language models. In Advances in Neural Information Processing Systems, volume 37, 2024.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/v1/D18-1260.

JuneYoung Park, Minjae Kang, Seongbae Lee, Haegang Lee, Seongwan Kim, and Jaeho Lee. Riemannian optimization for LoRA on the stiefel manifold. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 20971–20985, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-emnlp.1143.

Qwen, An Yang, et al. Qwen2.5 technical report, 2025.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392. Association for Computational Linguistics, 2016. doi: 10.18653/v1/D16-1264.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 8732–8740, 2020. doi: 10.1609/aaai.v34i05.6399.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 4463–4473. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1454.

Steffen Schotthöfer, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco, and Jonas Kusch. GeoLoRA: Geometric integration for parameter efficient fine-tuning. In International Conference on Learning Representations, 2025.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631–1642. Association for Computational Linguistics, 2013.

Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, and Ali Ghodsi. DyLoRA: Parameterefficient tuning of pre-trained models using dynamic search-free low-rank adaptation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 3274–3287. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.eacl-main.239.

Bart Vandereycken. Low-rank matrix completion by riemannian optimization. SIAM Journal on Optimization, 23(2):1214–1236, 2013. doi: 10.1137/110845768.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019.

Shaowen Wang, Linxi Yu, and Jian Li. LoRA-GA: Low-rank adaptation with gradient approximation. In Advances in Neural Information Processing Systems, volume 37, 2024.

Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. Neural network acceptability judgments. Transactions ofthe Associationfor Computational Linguistics, 7:625–641, 2019. doi: 10.1162/ tacl\_a\_00290.

Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112–1122. Association for Computational Linguistics, 2018. doi: 10.18653/v1/N18-1101.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800. Association for Computational Linguistics, 2019. doi: 10.18653/v1/P19-1472.

Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, Lucas Beyer, Olivier Bachem, Michael Tschannen, Marcin Michalski, Olivier Bousquet, Sylvain Gelly, and Neil Houlsby. A large-scale study of representation learning with the visual task adaptation benchmark, 2020.

Fangzhao Zhang and Mert Pilanci. Riemannian preconditioned LoRA for fine-tuning foundation models. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 59641–59669. PMLR, 2024.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In International Conference on Learning Representations, 2023.

Yuan Zhang, Jiang Hu, Zhijian Lai, Lin Lin, and Zaiwen Wen. Retraction-free optimization over the stiefel manifold for the LoRA fine-tuning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026. doi: 10.1109/TPAMI.2026.3711595.

## A SUPPLEMENTARY OVERVIEW

This supplementary material provides additional methodological, theoretical, and experimental details for GEAR. Appendix B describes the implementation of fixed-rank retraction and the validation procedure for rank-expansion candidates. Appendix C provides complete proofs of the three theoretical results presented in the main paper. Appendix D reports the datasets, evaluation protocols, training configurations, and hyperparameter settings used in our experiments. Finally, ablation studies are provided in Appendix E.

## B ADDITIONAL METHOD DETAILS

## B.1 LOW-RANK RETRACTION

We detail the retraction used in Section 3.2. Recall that the current adaptation state is represented as $X = U C$ , with $U ^ { \top } U = I _ { r }$ , and let

$$
\begin{array} { r } { D = U ^ { \top } G , \qquad K = ( I - U U ^ { \top } ) G V . } \end{array}
$$

Using the tangent-gradient decomposition in Eq. (1), the update before retraction can be written as

$$
X ^ { - } = U ( C - \eta D ) - \eta K V ^ { \top } = L R ,
$$

where

$$
\begin{array} { r } { L = [ U , K ] , \qquad R = \left[ { C - \eta D } \right] . } \end{array}
$$

Hence, $X ^ { - }$ has rank at most 2r and can be manipulated without forming the full $n \times m$ matrix.

We compute thin QR factorizations of the two compact factors,

$$
L = Q _ { L } R _ { L } , \qquad R ^ { \top } = Q _ { R } R _ { R } ,
$$

which gives

$$
X ^ { - } = Q _ { L } \left( R _ { L } R _ { R } ^ { \top } \right) Q _ { R } ^ { \top } .
$$

The matrix $R _ { L } R _ { R } ^ { \top }$ is at most $2 r \times 2 r$ . We therefore compute only its compact SVD,

$$
R _ { L } R _ { R } ^ { \top } = P \Sigma Q ^ { \top } ,
$$

and retain the leading r singular components. The retracted state is

$$
X _ { t + 1 } = ( Q _ { L } P _ { r } ) \Sigma _ { r } ( Q _ { R } Q _ { r } ) ^ { \top } .
$$

Accordingly, the updated low-rank representation can be obtained directly as

$$
U _ { t + 1 } = Q _ { L } P _ { r } , \qquad V _ { t + 1 } = Q _ { R } Q _ { r } , \qquad C _ { t + 1 } = \Sigma _ { r } V _ { t + 1 } ^ { \top } .
$$

Thus, the retraction simultaneously restores the rank-r constraint and updates the left and right subspace bases required by the next iteration, while avoiding any SVD of the full adaptation matrix.

## B.2 RANK-EXPANSION CANDIDATE VALIDATION

At each rank audit, the selected normal directions are treated as candidate expansions rather than committed immediately. We order the proposed directions by the utility score in Eq. (5) and construct several prefixes with different birth-step scales. The current adapter and all candidate variants are evaluated on the same set of separate training microbatches under matched stochasticity, so that their losses are directly comparable.

Let ${ \bar { L } } _ { 0 }$ and $\bar { L } _ { \mathrm { c a n d } }$ denote the mean training losses of the current adapter and a candidate, respectively. We define the relative improvement as

$$
\mathrm { R I } = \frac { \bar { L } _ { 0 } - \bar { L } _ { \mathrm { c a n d } } } { \left| \bar { L } _ { 0 } \right| } .
$$

A candidate is accepted only if $\mathrm { R I } > \rho ,$ where $\rho$ is the prescribed acceptance threshold.

Among the accepted candidates, we favor the smallest rank increase whose loss is close to the best observed value, and then the smaller birth-step scale. The selected expansion is committed only if it also satisfies the per-module rank limit and the global budget. If no candidate passes the acceptance criterion, the audit leaves the adapter unchanged.

Table 4: Training time and peak CUDA memory on DTD with ViT-B/16.
<table><tr><td>Method</td><td>Training time (s)</td><td>Peak CUDA memory (MiB)</td></tr><tr><td>LoRA</td><td>789.83</td><td>3825.15</td></tr><tr><td>FlexLoRA</td><td>1006.75</td><td>6553.87</td></tr><tr><td>GEAR</td><td>1060.83</td><td>4458.21</td></tr></table>

## B.3 COMPUTATIONAL AND MEMORY COMPLEXITY

GEAR preserves the low-rank computational structure throughout training. For a rank-r adapter, a standard LoRA factor update requires $\mathcal { O } ( ( n + m ) r )$ element-wise operations, while the additional retraction in GEAR costs $\mathcal { O } ( ( n + m ) r ^ { 2 } + \stackrel { . . } { r ^ { 3 } } )$ due to thin QR factorizations and an SVD of a $2 r \times 2 r$ core matrix. Since $r \ll \operatorname* { m i n } \{ n , m \}$ , this overhead remains confined to the low-rank factors and small core matrices, without introducing any full $n \times m$ decomposition. The committed adapter state requires only $\mathcal { O } ( ( n + m ) r )$ memory.

TGI is performed once before fine-tuning. In the language experiments, its $2 ( p + 1 )$ probing stages require $2 ( p + 1 )$ calibration backward passes per initialization run and $\mathcal { O } ( h \operatorname* { m a x } \{ n , \dot { m } \} )$ peak sketch storage per adapted matrix. The QR and SVD operations cost $\mathcal { O } ( ( p + 1 ) ( n + m ) h ^ { 2 } )$ . We use $p = 1$ and $h \ll \operatorname* { m i n } \{ n , m \}$ . In VTAB, initialization instead aggregates dense gradient proxies for the selected $\mathrm { Q } / \mathrm { V }$ projections from three calibration batches, requiring $\mathcal { O } ( n m )$ temporary gradient storage per adapted matrix.

Rank audits introduce additional computation only at scheduled audit steps. With the same probe width h, the projections, orthogonalization, and decomposition of the compressed normal sketch incur $\mathcal { O } ( ( n + m ) \dot { r } h \dot { + } ( n + m ) h ^ { 2 } )$ low-rank linear-algebra cost and $\mathcal { O } ( ( n + \bar { m } ) h )$ temporary memory, in addition to the backward probes used to obtain gradient products. Both r and h are small relative to the ambient dimensions, and no full gradient or normal matrix is explicitly stored.

The main audit overhead therefore comes from the additional backward probes and candidate forward evaluations rather than large-scale matrix operations. These computations are performed only for $t \in { \mathcal { T } } _ { \mathrm { a u d i t } }$ , while ordinary training steps incur only the fixed-rank manifold update. Moreover, all probing variables are transient and do not contribute to the committed adapter rank or the trainableparameter counts reported in our experiments.

We further assess the measured resource costs on the DTD task with ViT-B/16 in Table 4.

## C PROOFS

## C.1 PROOF OF THEOREM 1: FACTOR-GRADIENT OBSERVABILITY BARRIER

Proof. Consider the factorization map

$$
\Phi : \mathbb { R } ^ { n \times r } \times \mathbb { R } ^ { r \times m }  \mathbb { R } ^ { n \times m } , \qquad \Phi ( B , A ) = B A .
$$

Its differential at $( B , A )$ is

$$
D \Phi _ { ( B , A ) } [ \Delta B , \Delta A ] = \Delta B A + B \Delta A .
$$

For arbitrary perturbations $( \Delta B , \Delta A )$ , define $B ( t ) = B + t \Delta B$ and $A ( t ) = A + t \Delta A$ . Since the set of full-rank matrices is open, both factors remain full rank for sufficiently small t. Hence $X ( t ) = B ( t ) A ( t ) \in \mathcal { M } _ { \sf \sf }$ <sub>r</sub> locally around $t = 0$ , and

$$
\dot { X } ( 0 ) = \Delta B A + B \Delta A = D \Phi _ { ( B , A ) } [ \Delta B , \Delta A ] .
$$

Therefore,

$$
\operatorname { I m } D \Phi _ { ( B , A ) } \subseteq T _ { X } { \mathcal { M } } _ { r } .
$$

To determine the dimension of this image, consider the kernel condition $\Delta B A + B \Delta A = 0$ . Since A has full row rank, it admits a right inverse $A _ { R }$ satisfying $A A _ { R } = I _ { r }$ . Right-multiplication by $A _ { R }$

gives $\Delta B = - B \Delta A A _ { R } ,$ so $\Delta B = B K$ for some $K \in \mathbb { R } ^ { r \times r }$ . Substitution into the kernel condition yields $B ( K A + \Delta A ) = 0$ . Since B has full column rank, $\Delta A = - K A$ . Conversely, every pair $\dot { ( B K , - K A ) }$ satisfies $D \Phi _ { ( B , A ) } [ B K , - K A ] = 0$ . Thus,

$$
\ker D \Phi _ { ( B , A ) } = \{ ( B K , - K A ) : K \in \mathbb { R } ^ { r \times r } \} { \mathrm { . } }
$$

The parametrization by K is injective because B has full column rank; hence dim ker $D \Phi _ { ( B , A ) } = r ^ { 2 }$ By the rank–nullity theorem,

$$
\dim \operatorname { I m } D \Phi _ { ( B , A ) } = n r + r m - r ^ { 2 } = r ( n + m - r ) .
$$

Since dim $T _ { X } \mathcal { M } _ { r } = \mathrm { d i m } \mathcal { M } _ { r } = r ( n + m - r )$ , the inclusion above is between spaces of equal dimension, and therefore

$$
\operatorname { I m } D \Phi _ { ( B , A ) } = T _ { X } \mathcal { M } _ { r } .
$$

We next identify the adjoint of $D \Phi _ { ( B , A ) }$ . Under the Frobenius inner product,

$$
\begin{array} { r l } & { \big \langle G , D \Phi _ { ( B , A ) } [ \Delta B , \Delta A ] \big \rangle _ { F } = \langle G , \Delta B A + B \Delta A \rangle _ { F } } \\ & { \qquad = \langle G A ^ { \top } , \Delta B \rangle _ { F } + \langle B ^ { \top } G , \Delta A \rangle _ { F } . } \end{array}
$$

Hence

$$
D \Phi _ { ( B , A ) } ^ { * } ( G ) = ( G A ^ { \top } , B ^ { \top } G ) = { \mathcal { O } } _ { A , B } ( G ) .
$$

For any linear operator L between finite-dimensional inner-product spaces, ker $L ^ { * } = ( \operatorname { I m } L ) ^ { \perp }$ Applying this identity to $D \Phi _ { ( B , A ) }$ and using the characterization above gives

$$
\ker { \mathcal { O } } _ { A , B } = \left( \operatorname { I m } D \Phi _ { ( B , A ) } \right) ^ { \perp } = ( T _ { X } \mathcal { M } _ { r } ) ^ { \perp } = N _ { X } \mathcal { M } _ { r } .
$$

Finally, for $D = G _ { 1 } - G _ { 2 } , \mathcal { O } _ { A , B } ( G _ { 1 } ) = \mathcal { O } _ { A , B } ( G _ { 2 } )$ if and only if $\mathcal { O } _ { A , B } ( D ) = 0$ , which is equivalent to $D \in N _ { X } { \mathcal { M } } _ { r }$ . Equivalently, the tangent projection of D vanishes, and hence

$$
( G _ { 1 } ) _ { T } = ( G _ { 2 } ) _ { T } .
$$

This proves both statements.

## C.2 PROOF OF THEOREM 2: LOCAL OPTIMALITY OF RANK EXPANSION

Proof. Let $\begin{array} { r } { X = U \Sigma V ^ { \top } \in \mathcal { M } } \end{array}$ <sub>r</sub> be a compact SVD, and let $U _ { \perp }$ and $V _ { \perp }$ be orthonormal complements of $U$ and V. For any perturbation H, write

$$
\left[ U _ { \perp } ^ { \top } \right] H \left[ V \quad V _ { \perp } \right] = \left[ { \cal H } _ { 1 1 } \quad { \cal H } _ { 1 2 } \right] .
$$

Its normal component is $H _ { N } = U _ { \perp } H _ { 2 2 } V _ { \parallel } ^ { \top }$ , while the remaining three blocks constitute the tangent component $H _ { T }$

We first characterize the tangent cone of $\mathcal { V } _ { r + k } = \{ Y : \operatorname { r a n k } ( Y ) \leq r + k \}$ at X. In the above coordinates,

$$
\begin{array} { r } { \mathcal { C } _ { r , k } ( X ) = \left\{ H _ { T } + H _ { N } : H _ { T } \in T _ { X } \mathcal { M } _ { r } ,  H _ { N } \in N _ { X } \mathcal { M } _ { r } , \mathrm { r a n k } ( H _ { N } ) \leq k \right\} . } \end{array}
$$

To verify this characterization, consider a feasible curve $X ( t ) \in \mathcal { V } _ { r + k }$ with $X ( 0 ) = X$ and $\dot { X } ( 0 ) =$ H. Its block form is

$$
X ( t ) = \left[ \begin{array} { c c } { { \Sigma + t H _ { 1 1 } + o ( t ) } } & { { t H _ { 1 2 } + o ( t ) } } \\ { { t H _ { 2 1 } + o ( t ) } } & { { t H _ { 2 2 } + o ( t ) } } \end{array} \right] .
$$

For sufficiently small $t ,$ the upper-left block is nonsingular. Since rank $( X ( t ) ) \leq r + k$ , its Schur complement has rank at most k. Dividing by t and letting $t \to 0$ therefore gives rank $( H _ { 2 2 } ) \le k$

Conversely, suppose ran $\ k ( H _ { 2 2 } ) \leq k$ and define $A _ { t } = \Sigma + t H _ { 1 1 }$ . Consider

$$
\begin{array} { r } { \widetilde { X } ( t ) = \left[ \begin{array} { c c } { A _ { t } } & { t H _ { 1 2 } } \\ { t H _ { 2 1 } } & { t H _ { 2 2 } + t ^ { 2 } H _ { 2 1 } A _ { t } ^ { - 1 } H _ { 1 2 } } \end{array} \right] . } \end{array}
$$

Its Schur complement with respect to $A _ { t }$ is $t H _ { 2 2 }$ , so ran $\tau ( \widetilde { X } ( t ) ) = r + \mathrm { r a n k } ( H _ { 2 2 } ) \le r + k$ Moreover, ${ \widetilde { X } } ( 0 ) = X$ and $\dot { \tilde { X } } ( 0 ) = H$ , establishing the stated tangent-cone characterization

We now minimize

$$
q _ { \eta } ( H ) = \langle G , H \rangle _ { F } + \frac { 1 } { 2 \eta } \| H \| _ { F } ^ { 2 }
$$

over $\mathcal { C } _ { r , k } ( X )$ . Since the tangent and normal spaces are orthogonal, write $G = G _ { T } + G _ { N }$ and $H = H _ { T } + H _ { N }$ . Then

$$
q _ { \eta } ( H ) = \left( \langle G _ { T } , H _ { T } \rangle _ { F } + \frac { 1 } { 2 \eta } \| H _ { T } \| _ { F } ^ { 2 } \right) + \left( \langle G _ { N } , H _ { N } \rangle _ { F } + \frac { 1 } { 2 \eta } \| H _ { N } \| _ { F } ^ { 2 } \right) ,
$$

subject only to rank $( H _ { N } ) \leq k .$

For the tangent component,

$$
\langle G _ { T } , H _ { T } \rangle _ { F } + \frac { 1 } { 2 \eta } \| H _ { T } \| _ { F } ^ { 2 } = \frac { 1 } { 2 \eta } \| H _ { T } + \eta G _ { T } \| _ { F } ^ { 2 } - \frac { \eta } { 2 } \| G _ { T } \| _ { F } ^ { 2 } ,
$$

so the unique minimizer is $H _ { T } ^ { \star } = - \eta G _ { T }$

Similarly, for the normal component,

$$
\langle G _ { N } , H _ { N } \rangle _ { F } + \frac { 1 } { 2 \eta } \| H _ { N } \| _ { F } ^ { 2 } = \frac { 1 } { 2 \eta } \| H _ { N } + \eta G _ { N } \| _ { F } ^ { 2 } - \frac { \eta } { 2 } \| G _ { N } \| _ { F } ^ { 2 } .
$$

Thus, under rank $( H _ { N } ) \leq k _ { \mathrm { { i } } }$ , minimizing the normal term is equivalent to finding the best rankk approximation of $- \eta G _ { N }$ in Frobenius norm. By the Eckart–Young–Mirsky theorem, $H _ { N } ^ { \star } =$ $- \eta \hat { ( G _ { N } ) _ { k } }$ . Hence

$$
H _ { k } ^ { \star } = - \eta \left[ G _ { T } + ( G _ { N } ) _ { k } \right] .
$$

It remains to evaluate the additional decrease relative to fixed-rank optimization. Over $T _ { X } . M _ { r }$ ， min $\begin{array} { r } { q _ { \eta } = - \frac { \eta } { 2 } \lVert G _ { T } \rVert _ { F } ^ { 2 } } \end{array}$ , whereas over $\mathcal { C } _ { r , k } ( X )$ ,

$$
\operatorname* { m i n } { q _ { \eta } } = - \frac { \eta } { 2 } \| G _ { T } \| _ { F } ^ { 2 } - \frac { \eta } { 2 } \| ( G _ { N } ) _ { k } \| _ { F } ^ { 2 } .
$$

Therefore,

$$
\Delta _ { k } = \frac { \eta } { 2 } \| ( G _ { N } ) _ { k } \| _ { F } ^ { 2 } = \frac { \eta } { 2 } \sum _ { i = 1 } ^ { k } \sigma _ { i } ^ { 2 } ( G _ { N } ) .
$$

Thus, the leading normal singular directions yield the optimal local rank-k expansion, and their squared singular values determine the additional decrease of the local quadratic model. □

## C.3 PROOF OF THEOREM 3: STABILITY UNDER INEXACT NORMAL ESTIMATION

Proof. For brevity, let $N = G _ { N } , \widehat { N } = \widehat { G } _ { N } , N _ { k } = ( G _ { N } ) _ { k }$ , and ${ \widehat { N } } _ { k } = ( { \widehat { G } } _ { N } ) _ { k }$ . By assumption, $\| \widehat { N } - N \| _ { F } \leq \varepsilon$ and $\tau _ { k } = \| \boldsymbol { N } - \boldsymbol { N } _ { k } \| _ { F }$

Since $N _ { k }$ and $\widehat { N } _ { k }$ lie in the normal space, they are orthogonal to the tangent component of $G .$ Hence, for any normal matrix $B _ { ; }$

$$
q _ { \eta } ( - \eta B ) = \frac { \eta } { 2 } \| N - B \| _ { F } ^ { 2 } - \frac { \eta } { 2 } \| N \| _ { F } ^ { 2 } .
$$

Applying this identity to $B = N _ { k }$ and $B = \widehat { N } _ { k }$ yields

$$
q _ { \eta } ( - \eta \widehat { N } _ { k } ) - q _ { \eta } ( - \eta N _ { k } ) = \frac { \eta } { 2 } \left( \| N - \widehat { N } _ { k } \| _ { F } ^ { 2 } - \| N - N _ { k } \| _ { F } ^ { 2 } \right) .
$$

Because $N _ { k }$ is a best rank-k approximation of $N$ and rank $\begin{array} { r } { \cdot ( \widehat { N } _ { k } ) \leq k , \| N - N _ { k } \| _ { F } \leq \| N - \widehat { N } _ { k } \| _ { F } , } \end{array}$ Therefore, $q _ { \eta } ( - \eta \widehat { N } _ { k } ) - q _ { \eta } ( - \eta N _ { k } ) \geq 0$

It remains to establish the upper bound. Since $\widehat { N } _ { k }$ is a best rank-k approximation of $\widehat { N } , \| \widehat { N } - \widehat { N } _ { k } \| _ { F } \leq$ $\| \widehat { N } - N _ { k } \| _ { F }$ . Using the triangle inequality,

$$
\begin{array} { r l } & { \| N - \widehat { N } _ { k } \| _ { F } \leq \| N - \widehat { N } \| _ { F } + \| \widehat { N } - \widehat { N } _ { k } \| _ { F } } \\ & { \qquad \leq \varepsilon + \| \widehat { N } - N _ { k } \| _ { F } } \\ & { \qquad \leq \varepsilon + \| \widehat { N } - N \| _ { F } + \| N - N _ { k } \| _ { F } } \\ & { \qquad \leq \tau _ { k } + 2 \varepsilon . } \end{array}
$$

Substituting this bound into the preceding identity gives

$$
\begin{array} { c } { { q _ { \eta } ( - \eta \widehat { N } _ { k } ) - q _ { \eta } ( - \eta N _ { k } ) \leq \displaystyle \frac { \eta } { 2 } \left[ ( \tau _ { k } + 2 \varepsilon ) ^ { 2 } - \tau _ { k } ^ { 2 } \right] } } \\ { { = 2 \eta \varepsilon ( \tau _ { k } + \varepsilon ) . } } \end{array}
$$

Consequently,

$$
0 \leq q _ { \eta } \Bigl ( - \eta ( \widehat { G } _ { N } ) _ { k } \Bigr ) - q _ { \eta } \bigl ( - \eta ( G _ { N } ) _ { k } \bigr ) \leq 2 \eta \varepsilon ( \tau _ { k } + \varepsilon ) .
$$

This completes the proof.

## D EXPERIMENTAL DETAILS

## D.1 DETAILS ON COMMONSENSE REASONING TASK

The commonsense reasoning suite comprises eight benchmarks: BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), ARC-e/ARC-c (Clark et al., 2018), and OBQA (Mihaylov et al., 2018). Following the standard multi-task protocol, we combine the training portions of these tasks into the Commonsense170K corpus (Hu et al., 2023) and evaluate each benchmark separately on its corresponding test set. We use LLaMA3-8B and Qwen2.5-7B as backbone models. The main hyperparameter settings are summarized in Table 5.

The probing configuration estimates task-induced gradient structure for geometric initialization. We use probe width $h = 2 4$ for both TGI and rank audits. For rank audits, each calibration batch contains 128 examples, with 16 from each task. The damping coefficient $\lambda = 0 . 1$ regularizes the gradient-scale normalization. Subsequent experiments follow the same initialization and rank-audit framework with task-specific adjustments; the VTAB implementation is detailed in Appendix D.3.

## D.2 DETAILS ON NATURAL LANGUAGE UNDERSTANDING TASK

For the natural language understanding experiments, we fine-tune DeBERTa-v3-base on the GLUE benchmark (Wang et al., 2019), comprising eight tasks across single-sentence classification, semantic similarity and paraphrase detection, and natural language inference. Specifically, we evaluate on CoLA (Warstadt et al., 2019), SST-2 (Socher et al., 2013), MRPC (Dolan & Brockett, 2005), QQP (Wang et al., 2019), STS-B (Cer et al., 2017), MNLI (Williams et al., 2018), QNLI (Rajpurkar et al., 2016), and RTE (Dagan et al., 2006; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009). Dataset statistics and evaluation metrics are summarized in Table 6, and the corresponding training and rank-adaptation settings are reported in Table 7. We use the same probing and subspace estimation procedure as in the commonsense reasoning experiments, with shared probe width $h = 2 4$ for TGI and rank audits and task-specific calibration data and optimization settings. Separate learning rates are used for the adapter and task head because they operate under different parameterizations and effective update scales: the head follows standard Euclidean optimization, whereas the adapter learning rate controls the manifold-constrained update before retraction.

## D.3 DETAILS ON VISUAL TASK

We evaluate GEAR with ViT-B/16 on VTAB-1K (Zhai et al., 2020), which contains 19 visual classification tasks across Natural, Specialized, and Structured categories. Dataset statistics and hyperparameter settings are reported in Tables 8 and 9, respectively. For TGI, we aggregate dense gradient proxies for the selected Q/V projections from three calibration batches. Rank audits and candidate validation use task-specific settings.

Table 5: Hyperparameter settings for LLaMA3-8B and Qwen2.5-7B in commonsense reasoning experiments.
<table><tr><td>Hyperparameter</td><td>Setting</td></tr><tr><td>Initial effective rank  $r _ { 0 }$ </td><td>4</td></tr><tr><td>Maximum per-module rank  $r _ { \mathrm { m a x } }$ </td><td>32</td></tr><tr><td>Weighted-average rank budget  $\bar { r } _ { \mathrm { b u d g e t } }$ </td><td>16</td></tr><tr><td>Target modules</td><td>Q, K, V, O, Up, Down, Gate</td></tr><tr><td>Probe Ω width h</td><td>24</td></tr><tr><td>Probe batch size</td><td>128 total (16 per task)</td></tr><tr><td>Damping λ</td><td>0.1</td></tr><tr><td>First audit step  $t _ { \mathrm { s t a r t } }$ </td><td>100</td></tr><tr><td>Audit interval  $\Delta t$ </td><td>50</td></tr><tr><td>Rank-audit cutoff  $t _ { \mathrm { f i n a l } }$ </td><td>500</td></tr><tr><td>Max. rank increment per module per audit</td><td>4</td></tr><tr><td>Acceptance threshold  $\rho$ </td><td>0.5%</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Dropout</td><td>0.05</td></tr><tr><td>Scheduler / warmup</td><td>Linear / 0.1</td></tr><tr><td>Batch configuration</td><td>16 microbatch × 4 accumulation</td></tr><tr><td>Epochs</td><td>2</td></tr></table>

Table 6: Details of the GLUE dataset.
<table><tr><td>Dataset</td><td>Task</td><td># Train # Dev # Test # Label</td><td></td><td></td><td></td><td>Metrics</td></tr><tr><td colspan="7">Single-Sentence Classification</td></tr><tr><td>CoLA</td><td>Acceptability</td><td>8.5k</td><td>1k</td><td>1k</td><td>2</td><td>Matthews corr.</td></tr><tr><td>SST-2</td><td>Sentiment</td><td>67k</td><td>872</td><td>1.8k</td><td>2</td><td>Accuracy</td></tr><tr><td colspan="7">Similarity and Paraphrase</td></tr><tr><td>MRPC</td><td>Paraphrase</td><td>3.7k</td><td>408</td><td>1.7k</td><td>2</td><td>Accuracy</td></tr><tr><td>QQP</td><td>Paraphrase</td><td>364k</td><td>40k</td><td>391k</td><td>2</td><td>Accuracy</td></tr><tr><td>STS-B</td><td>Similarity</td><td>7k</td><td>1.5k</td><td>1.4k</td><td>1</td><td>Pearson/Spearman corr.</td></tr><tr><td colspan="7">Natural Language Inference</td></tr><tr><td>MNLI</td><td>NLI</td><td>393k</td><td>20k</td><td>20k</td><td>3</td><td>Accuracy</td></tr><tr><td>QNLI</td><td>QA/NLI</td><td>108k</td><td>5.7k</td><td>5.7k</td><td>2</td><td>Accuracy</td></tr><tr><td>RTE</td><td>NLI</td><td>2.5k</td><td>276</td><td>3k</td><td>2</td><td>Accuracy</td></tr></table>

Table 7: Hyper-parameter settings of our method on the GLUE benchmark.
<table><tr><td>Dataset</td><td>Adapter LR</td><td>Head LR</td><td>Batch</td><td>Epochs</td><td> $r _ { 0 }$ </td><td> ${ \bar { r } } _ { \mathbf { b u d g e t } }$ </td><td> $r _ { \mathrm { m a x } }$ </td><td> $\pmb { t _ { \mathrm { s t a r t } } }$ </td><td>∆t</td><td> $\pmb { t } _ { \mathrm { f i n a l } }$ </td></tr><tr><td>MNLI</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>32</td><td>6</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>RTE</td><td> $3 . 6 \times 1 0 ^ { - 3 }$ </td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td>32</td><td>50</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>QNLI</td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>32</td><td>5</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>MRPC</td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>32</td><td>30</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>QQP</td><td> $1 . 6 \times 1 0 ^ { - 3 }$ </td><td> $8 \times 1 0 ^ { - 4 }$ </td><td>32</td><td>5</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>SST-2</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>32</td><td>24</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>CoLA</td><td> $1 . 6 \times 1 0 ^ { - 3 }$ </td><td> $8 \times 1 0 ^ { - 4 }$ </td><td>32</td><td>20</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr><tr><td>STS-B</td><td> $4 . 4 \times 1 0 ^ { - 3 }$ </td><td> $2 . 2 \times 1 0 ^ { - 3 }$ </td><td>32</td><td>25</td><td>4</td><td>8</td><td>16</td><td>0.1T</td><td>0.075T</td><td>0.7T</td></tr></table>

T denotes the total number of optimization steps. At each rank audit, each module may add at most one new rank-one direction, with an acceptance threshold of $\rho = 0 . 1 \%$ relative loss improvement.

Table 8: Details of the VTAB-1K benchmark.
<table><tr><td>Dataset</td><td># Classes</td><td># Train</td><td># Val</td><td># Test</td></tr><tr><td>Natural</td><td></td><td></td><td></td><td></td></tr><tr><td>CIFAR100</td><td>100</td><td>800/1000</td><td>200</td><td>10,000</td></tr><tr><td>Caltech101</td><td>102</td><td>800/1000</td><td>200</td><td>6,084</td></tr><tr><td>DTD</td><td>47</td><td>800/1000</td><td>200</td><td>1,880</td></tr><tr><td>Flowers102</td><td>102</td><td>800/1000</td><td>200</td><td>6,149</td></tr><tr><td>Pets</td><td>37</td><td>800/1000</td><td>200</td><td>3,669</td></tr><tr><td>SVHN</td><td>10</td><td>800/1000</td><td>200</td><td>26,032</td></tr><tr><td>SUN397</td><td>397</td><td>800/1000</td><td>200</td><td>21,750</td></tr><tr><td>Specialized</td><td></td><td></td><td></td><td></td></tr><tr><td>Camelyon</td><td>2</td><td>800/1000</td><td>200</td><td>32,768</td></tr><tr><td>EuroSAT</td><td>10</td><td>800/1000</td><td>200</td><td>5,400</td></tr><tr><td>RESISC45</td><td>45</td><td>800/1000</td><td>200</td><td>6,300</td></tr><tr><td>Retinopathy</td><td>5</td><td>800/1000</td><td>200</td><td>42,670</td></tr><tr><td>Structured</td><td></td><td></td><td></td><td></td></tr><tr><td>CLEVR-Count</td><td>8</td><td>800/1000</td><td>200</td><td>15,000</td></tr><tr><td>CLEVR-Dist</td><td>6</td><td>800/1000</td><td>200</td><td>15,000</td></tr><tr><td>DMLab</td><td>6</td><td>800/1000</td><td>200</td><td>22,735</td></tr><tr><td>KITTI-Dist</td><td>4</td><td>800/1000</td><td>200</td><td>711</td></tr><tr><td>dSpr-Loc</td><td>16</td><td>800/1000</td><td>200</td><td>73,728</td></tr><tr><td>dSpr-Ori</td><td>16</td><td>800/1000</td><td>200</td><td>73,728</td></tr><tr><td>sNORB-Azim</td><td>18</td><td>800/1000</td><td>200</td><td>12,150</td></tr><tr><td>sNORB-Ele</td><td>18</td><td>800/1000</td><td>200</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>12,150</td></tr></table>

Table 9: Hyperparameter settings for the VTAB-1K experiments.
<table><tr><td>Hyperparameter</td><td>Setting</td></tr><tr><td>Initial effective rank  $r _ { 0 }$ </td><td>4</td></tr><tr><td>Weighted-average rank budget  $\bar { r } _ { \mathrm { b u d g e t } }$ </td><td>8</td></tr><tr><td>Maximum per-module rank  $r _ { \mathrm { m a x } }$ </td><td>16</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>First audit step  $t _ { \mathrm { s t a r t } }$ </td><td>160</td></tr><tr><td>Audit interval  $\Delta t$ </td><td>50</td></tr><tr><td>Rank-audit cutoff  $t _ { \mathrm { f i n a l } }$ </td><td>1300</td></tr><tr><td>Acceptance threshold  $\rho$ </td><td>0.03%</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Epochs</td><td>100</td></tr></table>

## E ABLATION STUDIES

To isolate the contribution of each component, we conduct a component-wise ablation on LLaMA3- 8B over the commonsense reasoning suite. Starting from the standard LoRA baseline, we examine task-induced geometric initialization (TGI), fixed-rank manifold optimization, and geometry-guided rank expansion. The results are reported in Table 10.

TGI and manifold optimization independently improve the average score from 85.03 to 85.57 and 85.65, respectively, while combining them further raises it to 86.23. This suggests that task-informed initialization and geometry-preserving optimization provide complementary benefits in constructing and optimizing the current adaptation space.

Rank expansion brings a larger gain. Adding it to manifold optimization improves the average from 85.65 to 87.32, while combining all three components further increases the score to 88.45, with improvements across all eight tasks. TGI also remains beneficial when rank expansion is enabled, improving the manifold-plus-expansion configuration from 87.32 to 88.45. Taken together, these ablations align with the central view of GEAR: dynamic rank adaptation should distinguish between better exploiting the current adaptation space and expanding it when task-relevant directions remain unrepresented.

Table 10: Component ablation of GEAR on commonsense reasoning with LLaMA3-8B. TGI denotes task-induced geometric initialization.
<table><tr><td>TGI Manifold Rank Expansion</td><td></td><td colspan="7">BoolQ PIQA SIQA HellaS. WinoG. ARC-e ARC-c OBQA Avg.</td></tr><tr><td></td><td></td><td></td><td>72.37 87.83 81.34</td><td>94.81</td><td>85.08</td><td>91.46</td><td>81.32</td><td>86.00</td><td>85.03</td></tr><tr><td>V</td><td></td><td></td><td>72.51 87.57 81.27</td><td>94.88</td><td>86.94</td><td>92.16</td><td>83.08</td><td>86.16</td><td>85.57</td></tr><tr><td>一</td><td>V</td><td></td><td>72.40 87.03 81.12</td><td>94.46</td><td>87.38</td><td>92.51</td><td>83.12</td><td>87.21</td><td>85.65</td></tr><tr><td>V</td><td>√</td><td></td><td>73.15 89.13 82.01</td><td>94.90</td><td>87.27</td><td>92.88</td><td>83.30</td><td>87.17</td><td>86.23</td></tr><tr><td>1</td><td>√</td><td>√</td><td>73.95 89.73 83.13</td><td>95.84</td><td>88.66</td><td>93.04</td><td>84.71</td><td>89.51</td><td>87.32</td></tr><tr><td>√</td><td>√</td><td>√</td><td>75.19 90.81 83.78</td><td>96.82</td><td>89.82</td><td>93.77</td><td>85.58</td><td>91.80</td><td>88.45</td></tr></table>