# HiLNO: A Hierarchical Latent Neural Operator with Multi-Scale Supervision for PDEs on General Geometries

Zhicheng Hu<sup>a,b,∗</sup>, Jiacheng Li<sup>a,b,1</sup>, Min Yang<sup>c,2</sup>

<sup>a</sup>School of Mathematics, Nanjing University of Aeronautics and Astronautics, Nanjing, 211106, China

<sup>b</sup>Key Laboratory of Mathematical Modelling and High Performance Computing of Air Vehicles

(NUAA), MIIT, Nanjing, 211106, China

<sup>c</sup>School of Mathematics and Information Sciences, Yantai University, Yantai, China

## Abstract

Latent neural operators improve the eficiency of operator learning for partial diferential equations (PDEs) by performing the main computation on compact latent representations. However, directly compressing the input representation to obtain such compact representations may discard solution-relevant spatial information, especially for PDE solutions with multiscale structures. To address this problem, we propose HiLNO, a hierarchical latent neural operator that constructs a fine-to-coarse-to-fine latent space and further introduces multi-scale supervision (MSS) and anisotropic Gaussian attention. The hierarchy mitigates potential information loss during compression, while MSS aligns intermediate predictions with downsampled target fields, encouraging solution-relevant structures to be captured across multiple spatial scales. Anisotropic Gaussian attention enables feature transfer across the hierarchy, making HiLNO applicable to general geometries. Experiments on representative PDE benchmarks and a large-scale automotive aerodynamics task show that HiLNO achieves competitive predictive accuracy, while reducing the parameter count by an average of 84.4% and FLOPs by an average of 69.2% compared with LinearNO. Additional experiments demonstrate efective generalization to unseen spatial resolutions. Code is available at https://github.com/JcLimath/HiLNO.

Keywords: Neural operator; Latent representations; Multi-scale supervision; General geometries

## 1. Introduction

Partial diferential equations (PDEs) provide a fundamental mathematical framework for describing a wide range of scientific and engineering systems. Classical numerical methods, such as finite diference and finite volume methods [32], are widely used to solve PDEs and can attain high accuracy through mesh refinement, but repeated simulations are often computationally expensive. In recent years, neural operators have emerged as promising data-driven surrogates that learn mappings from input functions to solution functions and, once trained, rapidly predict outputs for unseen inputs [4, 16, 23]. However, practical PDE problems often involve large numbers of sampling points, making dense interactions among all points expensive in both computation and memory.

Latent neural operators provide an eficient approach by projecting features defined on the physical discretization into compact latent representations [15, 21, 30]. The main computation is then performed in latent space, reducing the computational dependence on discretization size. For example, PiT [3] projects input features onto a coarse grid, Transolver [45] projects them onto a set of physics-aware slices, and IPOT [17] projects them onto a set of inducing points. Although such projections substantially reduce the cost of global interactions, direct compression into compact latent representations may discard solution-relevant spatial information, making multiscale structures more dificult to represent [10, 44].

This observation suggests replacing direct compression with a progressive process across multiple spatial resolutions, inspired by the multilevel processing used in classical multigrid methods [12, 14]. Related hierarchical designs have been explored in neural operators. For example, LSM [44] progressively constructs latent representations at multiple resolutions through hierarchical downsampling, while CALM-PDE [10] progressively coarsens the spatial representation using continuous convolutions. However, although hierarchical designs naturally provide representations at multiple spatial resolutions, training is typically dominated by the final-resolution objective, leaving intermediate representations only indirectly guided. This may prevent the hierarchy from fully capturing solution structures across scales.

To address these limitations, we propose HiLNO, a hierarchical latent neural operator built on a supervised hierarchical latent space. HiLNO follows a fine-to-coarse-to-fine paradigm, with the encoder and decoder performing hierarchical compression and reconstruction, respectively, while the processor approximates the operator at the coarsest latent level to avoid costly dense interactions. To provide direct guidance for intermediate representations, we introduce multi-scale supervision (MSS) along the reconstruction path. MSS matches intermediate predictions with downsampled target fields at the corresponding resolutions, encouraging solution-relevant structures to be captured across multiple spatial scales. We further introduce lightweight anisotropic Gaussian attention for feature transfer throughout the hierarchy, making HiLNO applicable to general geometries.

We evaluate HiLNO on representative PDE benchmarks and a three-dimensional industrial application. The results show that HiLNO achieves competitive predictive accuracy while requiring substantially fewer parameters and lower computational costs than representative neural operator baselines. Additional experiments demonstrate efective generalization to unseen spatial resolutions. The remainder of this paper is organized as follows. Section 2 reviews related work on neural PDE solvers and latent space learning. Section 3 presents the proposed HiLNO framework in detail. Section 4 reports numerical experiments and model analysis. Section 5 concludes the paper.

## 2. Related Work

## 2.1. Neural PDE Solvers

Physics-informed neural networks [33] incorporate governing equations into neural network training and have been widely studied for solving PDEs. However, they are commonly trained to approximate the solution of a specific PDE instance under prescribed conditions. Neural operators instead learn mappings between function spaces and can therefore be reused across varying coeficients, source terms, or geometries [5, 11, 16]. This operator learning paradigm has become an efective data-driven approach for constructing PDE surrogate models.

A variety of neural operator architectures have been developed for learning PDE solution operators [7, 19, 25, 38]. DeepONet [29] approximates nonlinear operators through a branch–trunk architecture, motivated by the universal approximation theorem for operators. FNO [19] parameterizes the integral kernel in the Fourier domain and evaluates global interactions eficiently using the fast Fourier transform. Owing to its simple implementation and strong performance, FNO has become a widely used baseline in operator learning [37]. Subsequent studies have extended Fourier-based neural operators to improve scalability [43], data eficiency [9], and applicability to complex geometries [18].

Attention mechanisms have increasingly been adopted in PDE operator learning because they provide a flexible way to model pairwise interactions between spatial locations [24]. FactFormer [22] factorizes the attention kernel along spatial axes to improve eficiency and stability, while CViT [41] uses query-wise cross-attention to map encoded input features to arbitrary spatial locations. Linear attention mechanisms have also been explored in OFormer [20], GNOT [11], and ONO [46] to reduce the quadratic cost of standard attention. Despite these advances, many neural operators still operate on representations defined over the physical discretization, making large discretizations computationally and memory intensive. This motivates performing the main operator computation on compact latent representations.

## 2.2. Latent Space Learning

Latent space learning has been widely used in fields such as natural language processing and computer vision to reduce the cost of processing high-dimensional data [47]. The main idea is to transform raw inputs into compact representations and perform the dominant feature interactions in the resulting latent space. In computer vision, patchbased tokenization groups pixels into patch tokens, allowing subsequent computation to be performed over a shorter token sequence, as used in the Vision Transformer [6] and Swin Transformer [26]. However, these designs are primarily developed for regularly sampled images with fixed spatial organization and cannot be directly applied to PDE data represented on diverse discretizations.

Motivated by the eficiency of computation in latent space, latent-space approaches have increasingly been explored in operator learning for PDEs [1, 27, 36]. LNO [42] employs cross-attention to map geometric-space features into latent tokens. Transolver [45] constructs physics-aware slices to model correlations among physical states. LANO [36] introduces a gated physics-adaptive encoder to obtain discriminative latent physical representations. Geometry can also be incorporated into latent-space construction. PiT [3] defines latent features on a prescribed coarse mesh. AROMA [35] learns geometry-aware latent representations from the input geometry. These methods demonstrate the efectiveness of computation in latent space, but typically obtain compact latent representations through direct compression. Such compression may discard solution-relevant spatial information, particularly for PDE solutions with multiscale structures.

Hierarchical architectures therefore provide a promising direction by organizing representations and computation across multiple resolutions [12]. CNO [34] employs a convolutional encoder–decoder architecture with representations learned at diferent spatial resolutions. LSM [44] constructs latent representations through hierarchical projection. CALM-PDE [10] progressively coarsens spatial representations using continuous convolutions. These studies demonstrate the potential of hierarchical computation for PDE operator learning. However, intermediate representations are typically guided only indirectly by the final prediction objective, which may limit their ability to capture solution structures across scales. HiLNO extends latent-space learning to a supervised hierarchical latent space, where multi-scale supervision directly guides intermediate representations.

## 3. Methods

In this section, we introduce HiLNO, a hierarchical latent neural operator that constructs a fine-to-coarse-to-fine hierarchical latent space and incorporates multi-scale supervision and anisotropic Gaussian attention. Section 3.1 presents the spatial supports of the hierarchy, followed by the encoder, latent processor, decoder, and multi-scale supervision. Section 3.2 then introduces anisotropic Gaussian attention for feature transfer across diferent resolutions.

Problem Setup Consider a bounded domain $\Omega \subset \mathbb { R } ^ { D }$ . Let a denote the problemdependent input, which may include coeficient fields, geometric descriptors, or forcing terms, and let u denote the target function, typically the PDE solution or a solutionrelated physical quantity. We denote the underlying operator by

$$
\mathcal { G } ^ { \dagger } : a \mapsto u .\tag{1}
$$

The objective is to approximate $\mathcal { G } ^ { \dagger }$ using a parameterized neural operator $\mathcal { G } _ { \theta }$ from a given dataset

$$
\boldsymbol { \mathcal { D } } = \left\{ \left( a ^ { ( j ) } \vert _ { X } , u ^ { ( j ) } \vert _ { X } \right) \right\} _ { j = 1 } ^ { J } ,\tag{2}
$$

where $X ~ = ~ \{ \pmb { x } ^ { i } \} _ { i = 1 } ^ { N } ~ \subset ~ \Omega$ denotes the set of discretization points used to sample the input and output functions, and N is the number of discretization points. In practical applications, N is often large to accurately represent complex geometries and capture fine-scale solution structures.

Overview of HiLNO As illustrated in Figure 1, HiLNO comprises a fine-to-coarse encoder, a coarse latent processor, and a coarse-to-fine decoder. The encoder progressively compresses the input representation across multiple latent levels, the processor performs the main operator computation at the coarsest level, and the decoder reconstructs the representation on the original discretization. The forward process is summarized as

$$
( z ^ { 0 } , X ^ { 0 } )  \cdots  ( z ^ { L } , X ^ { L } )  ( \hat { z } ^ { L } , X ^ { L } )  \cdots  ( \hat { z } ^ { 0 } , X ^ { 0 } ) .\tag{3}
$$

Here, $X ^ { 0 } = X$ , and $\{ X ^ { \ell } \} _ { \ell = 0 } ^ { L }$ denotes the fine-to-coarse hierarchy of spatial supports for latent representations. The feature $z ^ { \ell }$ denotes the encoder-side representation on $X ^ { \ell }$ , while $\hat { z } ^ { \ell }$ denotes the corresponding decoder-side representation, with $\hat { z } ^ { L }$ produced by the latent processor. Compared with existing hierarchical designs such as LSM [44] and CALM-PDE [10], HiLNO maps decoder-side representations $\hat { z } ^ { \ell }$ to predictions $\hat { u } ^ { \ell }$ at the corresponding resolutions, where $\hat { u } ^ { 0 }$ is the final prediction and $\hat { u } ^ { \ell }$ serve as intermediate predictions for multi-scale supervision.

![](images/146481ed905356bb59f3231b29d57eecd496b29fc4538fa0693b260495b72dbe.jpg)  
Figure 1: Overall architecture of HiLNO. Latent representations follow a fine-to-coarse-to-fine hierarchy with multi-scale supervision at intermediate resolutions. Gaussian attention blocks (GauAttBlocks) transfer features between adjacent levels, and the shared projection Q produces multi-scale predictions.

## 3.1. Supervised Hierarchical Latent Space

Spatial Supports of the Hierarchy In HiLNO, diferent levels of the hierarchical latent space correspond to diferent spatial resolutions, facilitating the representation of solution structures across multiple spatial scales. We therefore construct an ordered sequence of spatial supports with progressively fewer points. Latent representations at deeper levels are defined on increasingly coarse discretizations, giving the hierarchical latent space a clear spatial interpretation.

Specifically, we define

$$
X ^ { 0 } = X , \qquad X ^ { \ell } \subset X , \qquad N _ { \ell } = | X ^ { \ell } | < N _ { \ell - 1 } = | X ^ { \ell - 1 } | , \quad \ell = 1 , \dots , L .\tag{4}
$$

Here, $X ^ { \ell }$ provides the spatial support for the latent representation $z ^ { \ell }$ at level ℓ. Importantly, we do not require nested supports, i.e., $X ^ { \ell } \subset X ^ { \ell - 1 }$ is not imposed. Instead, each $X ^ { \ell }$ is sampled directly from the original discretization X, allowing each level to independently cover the computational domain while ensuring that target values for multi-scale supervision can be obtained directly from the original solution field without interpolation. The specific sampling strategies used to construct these supports are described in the experimental section.

Fine-to-Coarse Encoder Let $a | _ { X ^ { 0 } }$ denote the discretized input function, and let $\{ X ^ { \ell } \} _ { \ell = 0 } ^ { L }$ denote the spatial supports defined above. A pointwise lifting layer first maps the input function to the initial feature representation

$$
z ^ { 0 } = \mathcal { P } \left( a | _ { X ^ { 0 } } \right) \in \mathbb { R } ^ { N _ { 0 } \times C } ,\tag{5}
$$

where $N _ { 0 } = | X ^ { 0 } |$ , C is the feature dimension, and $\mathcal { P }$ is implemented as a multilayer perceptron (MLP).

The encoder then transfers features along the fine-to-coarse path. For $\ell = 0 , \dots , L - 1$ the representation on the next coarser point set is obtained as

$$
z ^ { \ell + 1 } = \mathrm { G a u A t t B l o c k } \left( z ^ { \ell } , X ^ { \ell } , X ^ { \ell + 1 } \right) ,\tag{6}
$$

where GauAttBlock denotes the Gaussian attention block illustrated in Figure 2 and formally defined in Section 3.2. By transferring features between adjacent resolutions, the encoder avoids a direct projection from the full discretization to the coarsest latent point set.

Coarse Latent Processor The processor updates the representation on the coarsest point set $X ^ { L }$ . For a processor consisting of K Gaussian attention blocks, let $z ^ { L , 0 } = z ^ { L }$ The representation is updated successively as

$$
{ z } ^ { L , k + 1 } = \operatorname { G a u A t t B l o c k } \left(  { z } ^ { L , k } ,  { X } ^ { L } ,  { X } ^ { L } \right) , \qquad k = 0 , \ldots , K - 1 .\tag{7}
$$

The processor output is denoted by $\hat { z } ^ { L } = z ^ { L , K }$ . Since $N _ { L } \ll N _ { 0 }$ , performing the main operator updates on $X ^ { L }$ reduces the cost of repeated feature interactions compared with operating directly on the original discretization.

Coarse-to-Fine Decoder The decoder progressively reconstructs the latent representations along the coarse-to-fine path. For $\ell = L , \ldots , 1$ , the representation on $X ^ { \ell }$ is first transferred to the finer point set $X ^ { \ell - 1 }$

$$
\tilde { z } ^ { \ell - 1 } = \mathrm { G a u A t t B l o c k } \left( \hat { z } ^ { \ell } , X ^ { \ell } , X ^ { \ell - 1 } \right) .\tag{8}
$$

The transferred feature is then fused with the encoder feature at the corresponding resolution

$$
\hat { z } ^ { \ell - 1 } = \mathrm { M L P } \left( \tilde { z } ^ { \ell - 1 } + z ^ { \ell - 1 } \right) .\tag{9}
$$

The skip connection is introduced to improve training stability while reintroducing finescale information retained by the encoder at the corresponding resolution. The resulting decoder-side representations $\{ \hat { z } ^ { \ell } \} _ { \ell = 0 } ^ { L }$ are used to generate solution predictions at the corresponding resolutions.

Multi-Scale Supervision When training relies solely on the final-resolution loss, intermediate representations are optimized only indirectly through the final prediction. To provide direct guidance at multiple spatial scales, we introduce auxiliary supervision at every decoded level. Since each decoded representation $\hat { z } ^ { \ell }$ is associated with the point set $X ^ { \ell }$ , it is mapped to a solution prediction on the corresponding discretization using a shared MLP pointwise projection layer Q as

$$
\hat { u } ^ { \ell } = \mathcal { Q } \left( \hat { z } ^ { \ell } \right) , \qquad \ell = 0 , \dots , L .\tag{10}
$$

Here, $\hat { u } ^ { 0 }$ is the final prediction on $X ^ { 0 }$ , whereas $\{ \hat { u } ^ { \ell } \} _ { \ell = 1 } ^ { L }$ are intermediate predictions on the coarser point sets. Because $X ^ { \ell } \subset X$ , the corresponding target field can be obtained directly as

$$
u ^ { \ell } = u | _ { X ^ { \ell } } , \qquad \ell = 0 , \dots , L ,\tag{11}
$$

without interpolation.

The multi-scale supervision objective is defined as

$$
\mathcal { L } _ { \mathrm { M S } } ( \boldsymbol { \theta } ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \sum _ { \ell = 0 } ^ { L } \omega _ { \ell } \mathcal { L } \left( \hat { u } ^ { ( j ) , \ell } , u ^ { ( j ) , \ell } \right) ,\tag{12}
$$

where $J$ is the number of training samples, $\omega _ { \ell }$ is the loss weight assigned to level $\ell ,$ and $\mathcal { L }$ denotes the task-specific loss function. This multi-scale supervision directly guides intermediate representations, helping the hierarchy retain solution-relevant information across multiple scales.

## 3.2. Anisotropic Gaussian Attention for Cross-level Transfer

The hierarchical latent space introduced above requires repeated feature propagation between adjacent spatial supports. For this purpose, we introduce an anisotropic Gaussian attention mechanism that constructs transfer weights directly from the relative positions of the source and target points.

Anisotropic Gaussian Weights Let $X = \{ \pmb { x } ^ { i } \} _ { i = 1 } ^ { N _ { s } }$ and $Y = \{ \pmb { y } ^ { j } \} _ { j = 1 } ^ { N _ { t } }$ denote the source and target point sets in a D-dimensional domain, with source features $z ^ { s } \in \mathbb { R } ^ { N _ { s } \times C }$ on X and target features $z ^ { t } \in \mathbb { R } ^ { N _ { t } \times C }$ on $Y$

Let $\pmb { \sigma } = ( \sigma _ { 1 } , \dots , \sigma _ { D } ) \in \mathbb { R } _ { + } ^ { D }$ denote a learnable axis-aligned anisotropic Gaussian scale, where $\sigma _ { d }$ controls the spatial decay along the d-th coordinate direction. This minimal anisotropic parameterization uses only D learnable parameters to capture directiondependent decay, making it readily applicable to higher-dimensional problems. For $\mathbf { \boldsymbol { x } } ^ { i } = \mathbf { \boldsymbol { \bar { \mathbf { \mathit { x } } } } } $ $( x _ { 1 } ^ { i } , \dots , x _ { D } ^ { i } ) \in X$ and $\pmb { y } ^ { j } = ( y _ { 1 } ^ { j } , \dots , y _ { D } ^ { j } ) \in Y$ , define the anisotropic Gaussian kernel as

$$
g _ { \sigma } ( \pmb { y } ^ { j } , \pmb { x } ^ { i } ) = \exp \left[ - \sum _ { d = 1 } ^ { D } \left( \frac { y _ { d } ^ { j } - x _ { d } ^ { i } } { \sigma _ { d } } \right) ^ { 2 } \right] .\tag{13}
$$

The normalized Gaussian attention weights are then given by

$$
a _ { j i } = \frac { g _ { \sigma } ( { \pmb y } ^ { j } , { \pmb x } ^ { i } ) } { \sum _ { k = 1 } ^ { N _ { s } } g _ { \sigma } ( { \pmb y } ^ { j } , { \pmb x } ^ { k } ) } .\tag{14}
$$

Here, $a _ { j i }$ measures the contribution of the source point $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } ^ { i } }$ to the target point $\boldsymbol { y } ^ { j }$ . Following the locality strategy in PiT [3], the attention can optionally be restricted by a locality ratio $P _ { \mathrm { l o c } } \in ( 0 , 1 ]$ . For each target point $\boldsymbol { y } ^ { j }$ , only the $\lceil P _ { \mathrm { l o c } } N _ { s } \rceil$ source points with the smallest Euclidean distances participate in the normalization, while the remaining weights are set to zero. The value of $P _ { \mathrm { l o c } }$ is left as a hyperparameter; see Section 4.3.3.

Collecting the pairwise transfer weights gives the attention matrix as

$$
A = [ a _ { j i } ] \in \mathbb { R } ^ { N _ { t } \times N _ { s } } .\tag{15}
$$

Gaussian attention (GauAtt) transfers features from X to $Y$ , with the output given by

$$
\begin{array} { r } { z ^ { t } = \mathrm { G a u A t t } ( z ^ { s } , X , Y ) = A z ^ { s } W _ { v } , } \end{array}\tag{16}
$$

where $W _ { v } \in \mathbb { R } ^ { C \times C }$ is a learnable value projection.

Multi-Head Gaussian Attention To enhance representation capacity, following the standard multi-head attention setting [40], we introduce multi-head Gaussian attention, in which each head is assigned an independent learnable Gaussian scale. For the multihead extension, the source feature is first projected and then partitioned along the feature dimension into H subspaces:

$$
\begin{array} { r } { z ^ { s } W _ { v } = \operatorname { C o n c a t } \left( V ^ { ( 1 ) } , \ldots , V ^ { ( H ) } \right) , \qquad V ^ { ( h ) } \in \mathbb { R } ^ { N _ { s } \times C _ { H } } , \quad C _ { H } = C / H . } \end{array}\tag{17}
$$

Here, $V ^ { ( h ) }$ denotes the feature subspace associated with the h-th head, and Concat(·) denotes concatenation along the feature dimension.

For the h-th head, let $A ^ { ( h ) }$ denote the Gaussian attention matrix constructed according to Equation (15) using its own learnable Gaussian scale. The corresponding feature subspace is transferred from X to Y , yielding

$$
O ^ { ( h ) } = A ^ { ( h ) } V ^ { ( h ) } \in \mathbb { R } ^ { N _ { t } \times C _ { H } } , \qquad h = 1 , \dots , H .\tag{18}
$$

![](images/2614d37cddff485bd4e01632cd27545233d24c4141daeddd7b0ba0368a02862e.jpg)  
Figure 2: Structure of the Gaussian attention block, consisting of a position embedding, multi-head Gaussian attention (MHGauAtt), and a residual MLP connection.

The outputs of all heads are concatenated and linearly projected. The final output of multi-head Gaussian attention (MHGauAtt) is given by

$$
\mathrm { M H G a u A t t } ( z ^ { s } , X , Y ) = \mathrm { C o n c a t } \left( O ^ { ( 1 ) } , \ldots , O ^ { ( H ) } \right) W _ { o } ,\tag{19}
$$

where $W _ { o } \in \mathbb { R } ^ { C \times C }$ denotes the learnable output projection. Head-specific Gaussian scales enable diferent heads to capture distinct spatial interaction patterns, as illustrated in Section 4.3.3.

Gaussian Attention Block We construct a Gaussian attention block (GauAttBlock) as the basic feature-transfer unit in HiLNO, as illustrated in Figure 2. The block combines pointwise position embeddings with multi-head Gaussian attention, incorporating both the absolute positions of the source points and their spatial relations to the target points.

Specifically, for a source feature $z ^ { s } \in \mathbb { R } ^ { N _ { s } \times C }$ defined on X, the source coordinates are first mapped to a position embedding as

$$
p ^ { s } = \phi _ { x } ( X ) \in \mathbb { R } ^ { N _ { s } \times C } ,\tag{20}
$$

where $\phi _ { x }$ is implemented as a pointwise MLP. The position embedding is then added to the source feature before multi-head Gaussian attention

$$
h ^ { t } = \mathrm { M H G a u A t t } \left( z ^ { s } + p ^ { s } , X , Y \right) , \qquad z ^ { t } = h ^ { t } + \mathrm { M L P } \left( h ^ { t } \right) .\tag{21}
$$

Here, the multi-head Gaussian attention transfers the augmented source feature from X to $Y$ , while the residual MLP further refines the resulting target representation. The complete block is written as

$$
z ^ { t } = \mathrm { G a u A t t B l o c k } ( z ^ { s } , X , Y ) .\tag{22}
$$

Properties of Gaussian Attention Unlike standard attention [40], Gaussian attention constructs transfer weights solely from relative positions, avoiding query–key similarity computation and keeping the weight construction lightweight. This position-based formulation is closely related to PiT [3], but generalizes its scalar distance scaling by introducing independent Gaussian scales for diferent coordinate directions, allowing anisotropic spatial interactions to be modeled. Since the resulting transfer is defined directly from point coordinates, it does not rely on regular-grid structure and can be applied to diverse spatial discretizations.

Beyond its discrete formulation, Gaussian attention admits a continuous-operator interpretation under source-point refinement. The following theorem formalizes this connection and establishes the convergence of Gaussian attention.

Theorem 3.1. Let $\{ X ^ { n } \} _ { n = 1 } ^ { \infty }$ be a sequence of source point sets sampled from a fixed probability measure $\mu _ { \Omega }$ on Ω. For each n,

$$
X ^ { n } = \{ \pmb { x } ^ { n , i } \} _ { i = 1 } ^ { N _ { n } } \subset \Omega \subset \mathbb { R } ^ { D } , \qquad \pmb { x } ^ { n , i } \overset { \mathrm { i . i . d . } } { \sim } \mu _ { \Omega } ,
$$

with $N _ { n }  \infty$ . Let $Y = \{ \pmb { y } ^ { j } \} _ { j = 1 } ^ { N _ { t } } \subset \Omega$ be a finite target point set. Assume that $v ( { \pmb x } )$ $\Omega \to \mathbb { R } ^ { C }$ is bounded and measurable on Ω, and let $z _ { n } ^ { s }$ denote its values sampled on $X ^ { n }$ For a fixed ${ \pmb { \sigma } } \in \mathbb { R } _ { + } ^ { D }$ , as $n  + \infty$ , the Gaussian attention defined in Equations (13)–(16) converges to an integral operator. Specifically, for any $\varepsilon > 0$ 2

$$
\operatorname* { l i m } _ { n \to + \infty } \operatorname* { P r } \left\{ \| \mathrm { G a u A t t } ( z _ { n } ^ { s } , X ^ { n } , Y ) - \mathcal { F } | _ { Y } \| \leq \varepsilon \right\} = 1 ,\tag{23}
$$

where,

$$
\mathcal { F } ( \pmb { y } ) = \int _ { \Omega } \kappa _ { \pmb { \sigma } } ( \pmb { y } , \pmb { x } ) v ( \pmb { x } ) W _ { v } d \mu _ { \Omega } ( \pmb { x } ) ,\tag{24}
$$

and

$$
\kappa _ { \pmb { \sigma } } ( \pmb { y } , \pmb { x } ) = \frac { g _ { \pmb { \sigma } } ( \pmb { y } , \pmb { x } ) } { \int _ { \Omega } g _ { \pmb { \sigma } } ( \pmb { y } , \pmb { x } ^ { \prime } ) d \mu _ { \Omega } ( \pmb { x } ^ { \prime } ) }\tag{25}
$$

is the anisotropic integral kernel induced by Gaussian attention.

Proof. Following the proof strategy used in PiT [3], for a fixed target point $y \in Y$ , define

$$
G _ { n } ( { \pmb y } ) = \frac { 1 } { N _ { n } } \sum _ { i = 1 } ^ { N _ { n } } g _ { \sigma } ( { \pmb y } , { \pmb x } ^ { n , i } ) v ( { \pmb x } ^ { n , i } ) W _ { v } , \qquad H _ { n } ( { \pmb y } ) = \frac { 1 } { N _ { n } } \sum _ { i = 1 } ^ { N _ { n } } g _ { \sigma } ( { \pmb y } , { \pmb x } ^ { n , i } ) ,\tag{26}
$$

where $g _ { \sigma }$ is the anisotropic Gaussian kernel defined in Equation (13). By the normalization in Equation (14) and the feature transfer in Equation (16), the Gaussian attention output at y can be written as

$$
\mathrm { G a u A t t } ( z _ { n } ^ { s } , X ^ { n } , { \pmb y } ) = \frac { G _ { n } ( { \pmb y } ) } { H _ { n } ( { \pmb y } ) } .\tag{27}
$$

Since v is bounded and measurable, $W _ { v }$ is fixed, and $0 < g _ { \sigma } ( { \pmb y } , { \pmb x } ) \leq 1$ , both $g _ { \pmb { \sigma } } ( \pmb { y } , \cdot ) \pmb { v } ( \cdot ) W _ { v }$ and $g _ { \sigma } ( \pmb { y } , \cdot )$ are integrable. Therefore, by the law of large numbers,

$$
\begin{array} { l } { { G _ { n } ( { \pmb y } ) \stackrel { p } {  } G ( { \pmb y } ) : = \displaystyle \int _ { \Omega } g _ { \pmb \sigma } ( { \pmb y } , { \pmb x } ) \upsilon ( { \pmb x } ) W _ { v } d \mu _ { \Omega } ( { \pmb x } ) , } } \\ { { \displaystyle H _ { n } ( { \pmb y } ) \stackrel { p } {  } H ( { \pmb y } ) : = \displaystyle \int _ { \Omega } g _ { \pmb \sigma } ( { \pmb y } , { \pmb x } ) d \mu _ { \Omega } ( { \pmb x } ) . } } \end{array}\tag{28}
$$

Since $g _ { \pmb { \sigma } } ( \pmb { y } , \pmb { x } ) > 0$ , we have $H ( \pmb { y } ) > 0$ . Applying the continuous mapping theorem to Equation (28) yields

$$
{ \frac { G _ { n } ( { \pmb y } ) } { H _ { n } ( { \pmb y } ) } } \stackrel { p } {  } { \frac { G ( { \pmb y } ) } { H ( { \pmb y } ) } } = { \mathcal { F } } ( { \pmb y } ) .\tag{29}
$$

Since Y contains finitely many target points, the pointwise convergence in Equation (29) implies convergence of the corresponding finite-dimensional output on $Y$ . Therefore, for any $\varepsilon > 0$

$$
\operatorname* { l i m } _ { n \to + \infty } \operatorname* { P r } \left\{ \| \mathrm { G a u A t t } ( z _ { n } ^ { s } , X ^ { n } , Y ) - \mathcal { F } | _ { Y } \| \leq \varepsilon \right\} = 1 ,\tag{30}
$$

which proves the result.

The result shows that the Gaussian attention can be viewed as a Monte Carlo approximation of a continuous Gaussian kernel integral operator. When $P _ { \mathrm { l o c } } ~ < ~ 1$ , the corresponding integral operator is restricted to the receptive field $B _ { r _ { y } } ( \pmb { y } )$ , a ball centered at y with radius determined by the locality ratio:

$$
\mathcal { F } _ { \mathrm { l o c } } ( \pmb { y } ) = \int _ { B _ { r _ { y } } ( \pmb { y } ) } \kappa _ { \pmb { \sigma } } ^ { \mathrm { l o c } } ( \pmb { y } , \pmb { x } ) v ( \pmb { x } ) W _ { v } d \mu _ { \Omega } ( \pmb { x } ) ,\tag{31}
$$

where

$$
\kappa _ { \sigma } ^ { \mathrm { l o c } } ( y , x ) = \frac { g _ { \sigma } ( y , \pmb { x } ) } { \int _ { B _ { r _ { y } } ( y ) } g _ { \sigma } ( \pmb { y } , \pmb { x } ^ { \prime } ) d \mu _ { \Omega } ( \pmb { x } ^ { \prime } ) } .\tag{32}
$$

The multi-head formulation applies the same construction to each head with head-specific Gaussian scales. This continuous-operator interpretation provides theoretical motivation for applying Gaussian attention across diferent spatial resolutions, while the crossresolution behavior of the complete HiLNO model is evaluated empirically in Section 4.4.

## 4. Experiments

We conduct numerical experiments to evaluate HiLNO on PDE benchmarks with diferent discretizations, geometries, and problem scales.

Benchmarks As summarized in Table 1, the experiments include two two-dimensional PDE benchmarks and a large-scale three-dimensional automotive aerodynamics task. The Darcy and Airfoil benchmarks were introduced in FNO [19] and Geo-FNO [18], respectively, and have been widely adopted in subsequent studies. For the Shape-Net Car benchmark, derived from [39], we follow the data preprocessing and evaluation settings used in Transolver [45].

Table 1: Summary of the PDE benchmarks used in the experiments. #Points denotes the number of discretization points per sample, and #Dataset reports the numbers of training and test samples.
<table><tr><td>Benchmark</td><td>Data Structure</td><td>Dim.</td><td>#Points</td><td>#Dataset</td></tr><tr><td>Darcy</td><td>Regular grid</td><td>2D</td><td>7,225</td><td>(1000, 200)</td></tr><tr><td>Airfoil</td><td>Structured mesh</td><td>2D</td><td>11,271</td><td>(1000, 200)</td></tr><tr><td>Shape-Net Car</td><td>Unstructured mesh</td><td>3D</td><td>32,186</td><td>(789,100)</td></tr></table>

Baselines We compare HiLNO with representative neural operator baselines, including FNO [19], Geo-FNO [18], GNOT [11], PiT [3], LNO [42], Transolver++ [31], and SAOT [48]. Moreover, LSM [44] and CALM-PDE [10] are included as representative hierarchical neural operators, while Transolver [45] and LinearNO [13] serve as representative recent attention-based neural operator baselines. For the Shape-Net Car benchmark, PointNet [2] and GraphUNet [8] are additionally included as geometric deep learning baselines. Baseline results are taken from the corresponding experiments reported in the LinearNO, SAOT, and CALM-PDE papers, with HiLNO evaluated under the same problem settings.

Table 2: Model hyperparameters and training configurations of HiLNO. Here, K denotes the number of processor blocks and LR denotes the learning rate.
<table><tr><td>Benchmark</td><td>Epochs</td><td>Batch Size</td><td>Heads</td><td>Feature Dim. C</td><td>Processor K</td><td>LR</td></tr><tr><td>Darcy</td><td>500</td><td>4</td><td>8</td><td>64</td><td>2</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Airfoil</td><td>500</td><td>1</td><td>8</td><td>64</td><td>2</td><td> $4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Shape-Net Car</td><td>200</td><td>1</td><td>8</td><td>128</td><td>2</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr></table>

Implementations The HiLNO configurations used for each benchmark are summarized in Table 2. We use the AdamW optimizer [28] with a warm-up stage followed by cosine learning-rate decay. Following common practice [19], the relative L2 error is used as the evaluation metric

$$
\mathrm { R e l a t i v e \ L 2 ~ E r r o r } = \frac { \| u - \hat { u } \| _ { 2 } } { \| u \| _ { 2 } } ,\tag{33}
$$

where u denotes the target solution, uˆ denotes the model prediction, and $\| \cdot \| _ { 2 }$ denotes the L2 norm. During training, we use the relative L2 loss for the two-dimensional benchmarks [19] and the mean squared error for Shape-Net Car [45]. Unless otherwise specified, multi-scale supervision uses equal weights across all spatial levels, i.e., $\omega _ { \ell } = 1 . 0$ in Equation (12). For Gaussian attention, the locality ratio is set to $P _ { \mathrm { l o c } } = 0 . 1$ in the encoder and $P _ { \mathrm { l o c } } = 1 . 0$ in the latent processor and decoder. All experiments are conducted on a single NVIDIA RTX 4090 GPU with 24 GB of memory.

## 4.1. Main Results

Table 3: Performance comparison on the Darcy benchmark. Bold numbers indicate the best results, while underlined numbers denote the second-best results. “\*” indicates results reproduced by us using the hyperparameters reported in the original paper.
<table><tr><td>Model</td><td>Relative L2 Error ↓</td></tr><tr><td>FNO [2021] Geo-FNO [2023a]</td><td>0.0108 0.0108</td></tr><tr><td>LSM [2023]</td><td>0.0065</td></tr><tr><td>PiT* [2024]</td><td>0.0058</td></tr><tr><td>Transolver* [2024]</td><td>0.0053</td></tr><tr><td>LNO [2024]</td><td>0.0063</td></tr><tr><td>Transolver++ [2025]</td><td>0.0056</td></tr><tr><td>SAOT [2026]</td><td>0.0049</td></tr><tr><td>LinearNO [2026]</td><td>0.0050</td></tr><tr><td>HiLNO</td><td>0.0037</td></tr></table>

![](images/c467ca2bcfe76162f4e6be3c62e0bf48993de18a591b7647612ca67c96be3e52.jpg)  
Figure 3: Visualization of the hierarchical point sets used for the Darcy benchmark.

## 4.1.1. Darcy

The Darcy benchmark evaluates operator learning on a regular grid. It models steadystate flow through a porous medium and is governed by

$$
\begin{array} { c c } { - \nabla \cdot ( a ( { \pmb x } ) \nabla u ( { \pmb x } ) ) = f ( { \pmb x } ) , } & { { \pmb x } \in ( 0 , 1 ) ^ { 2 } , } \\ { u ( { \pmb x } ) = 0 , } & { { \pmb x } \in \partial ( 0 , 1 ) ^ { 2 } , } \end{array}\tag{34}
$$

where $a ( { \pmb x } )$ is the difusion coeficient and $u ( { \pmb x } )$ is the solution [19]. The model takes a as input and predicts u. Following [45], each sample is discretized on an $8 5 \times 8 5 ~ \mathrm { g r i d }$ , with 1000 samples used for training and 200 for testing.

For this benchmark, HiLNO constructs the hierarchical point sets as

$$
X ^ { 0 } \ ( 7 , 2 2 5 )  X ^ { 1 } \ ( 1 , 8 4 9 )  X ^ { 2 } \ ( 8 4 1 )  X ^ { 3 } \ ( 4 8 4 )  X ^ { 4 } \ ( 2 2 5 ) ,\tag{35}
$$

corresponding to regular grids of resolutions $8 5 ^ { 2 } , 4 3 ^ { 2 } , 2 9 ^ { 2 } , 2 2 ^ { 2 }$ , and $1 5 ^ { 2 }$ , respectively. The intermediate point sets are obtained by uniformly subsampling the original grid $X ^ { 0 }$ with stride r along each spatial axis, with the resulting number of points given by

$$
n _ { r } = { \frac { 8 5 - 1 } { r } } + 1 , \qquad r \in \{ 2 , 3 , 4 , 6 \} .\tag{36}
$$

The resulting hierarchy is visualized in Figure 3.

As shown in Table 3, HiLNO achieves a relative L2 error of 0.0037, corresponding to a 24.5% reduction in error compared with the second-best baseline, SAOT. This result demonstrates the strong predictive performance of HiLNO on regular-grid PDE problems. Figure 4 further compares HiLNO with Transolver, a strong recent neural operator baseline, in terms of predicted solution fields and pointwise errors. HiLNO exhibits visibly smaller errors over most of the domain, particularly in regions with pronounced spatial variations, indicating improved reconstruction of local solution structures.

![](images/730c8321b2620585223d547af3f0f8a7c32696f9bf629cb3346fb0ea5a3c2059.jpg)  
Figure 4: Visualized comparisons on the Darcy benchmark. Each row represents one test sample with a diferent coeficient field, showing the ground truth, model predictions, and corresponding errors.

## 4.1.2. Airfoil

The Airfoil benchmark considers operator learning on a structured mesh with complex geometry. The task is to predict the Mach number distribution around an airfoil from its geometric representation. The governing equations are given by the Euler equations as follows

$$
\frac { \partial \rho ^ { f } } { \partial t } + \nabla \cdot \left( \rho ^ { f } \boldsymbol { v } \right) = 0 , \quad \frac { \partial \rho ^ { f } \boldsymbol { v } } { \partial t } + \nabla \cdot \left( \rho ^ { f } \boldsymbol { v } \otimes \boldsymbol { v } + p \boldsymbol { I } \right) = 0 , \quad \frac { \partial E } { \partial t } + \nabla \cdot \left( ( E + p ) \boldsymbol { v } \right) = 0 ,\tag{37}
$$

Table 4: Performance comparison on the Airfoil benchmark.
<table><tr><td>Model Relative L2 Error ↓</td></tr><tr><td>Geo-FNO [2023a] 0.0138</td></tr><tr><td>LSM [2023] 0.0059</td></tr><tr><td>Transolver* [2024] 0.0050</td></tr><tr><td>LNO [2024] 0.0053</td></tr><tr><td>CALM-PDE [2025] 0.0058</td></tr><tr><td>Transolver++ [2025] 0.0051</td></tr><tr><td>SAOT [2026] 0.0048</td></tr><tr><td>LinearNO [2026] 0.0049</td></tr><tr><td>HiLNO 0.0046</td></tr></table>

![](images/4164eea116394a1e5c79de244d8bf040d73be3ddf588ab6df85ec994c5245d6e.jpg)  
Figure 5: Visualization of the hierarchical point sets used for the Airfoil benchmark.

![](images/1f971551d70948f6e3596030020f3f61bb64af5f795c1ed17638329a0e32a7b2.jpg)  
Figure 6: Visualized comparisons on the Airfoil benchmark. Each row represents one test sample with a diferent airfoil design, showing the ground truth, model predictions, and corresponding errors.

where $\rho ^ { f }$ is the fluid density, $\boldsymbol { v }$ is the velocity vector, $p$ is the pressure, and $E$ is the total energy [18]. Each sample is discretized on a structured mesh of size $2 2 1 \times 5 1$ , corresponding to 11, 271 mesh points. Following [45], 1000 samples with diferent airfoil designs are used for training, and the remaining 200 samples are used for testing.

For this benchmark, HiLNO constructs the hierarchical point sets as

$$
X ^ { 0 } \ ( 1 1 , 2 7 1 )  X ^ { 1 } \ ( 2 , 8 8 6 )  X ^ { 2 } \ ( 7 8 4 ) ,\tag{38}
$$

corresponding to structured meshes of sizes $2 2 1 \times 5 1 , 1 1 1 \times 2 6$ , and $5 6 \times 1 4$ , respectively. The spatial supports $X ^ { 1 }$ and $X ^ { 2 }$ are sampled directly from the original mesh $X ^ { 0 }$ using geometry-preserving uniform subsampling. This construction retains the boundary points while progressively reducing the spatial resolution. The resulting hierarchy is visualized in Figure 5.

As shown in Table 4, HiLNO achieves a relative L2 error of 0.0046, outperforming all baseline models on the Airfoil benchmark. Figure 6 compares the predicted Mach number fields and pointwise errors of HiLNO and Transolver. HiLNO produces smaller and more localized errors, especially in regions with sharp spatial variations, such as near the airfoil boundary and shock regions. This result indicates that HiLNO improves the reconstruction of local flow variations on structured meshes.

Table 5: Performance comparison on the Shape-Net Car benchmark. ↓ indicates lower is better, while ↑ indicates higher is better.
<table><tr><td rowspan="2">Model</td><td colspan="4">Shape-Net Car</td></tr><tr><td>Velocity↓</td><td>Pressure ↓</td><td>CD↓</td><td>ρD ↑</td></tr><tr><td>PointNet [2017]</td><td>0.0494</td><td>0.1104</td><td>0.0298</td><td>0.9583</td></tr><tr><td>GraphUNet [2019]</td><td>0.0471</td><td>0.1102</td><td>0.0226</td><td>0.9725</td></tr><tr><td>Geo-FNO [2023a]</td><td>0.1670</td><td>0.2378</td><td>0.0664</td><td>0.8280</td></tr><tr><td>GNOT [2023]</td><td>0.0329</td><td>0.0798</td><td>0.0178</td><td>0.9833</td></tr><tr><td>LNO [2024]</td><td>0.0269</td><td>0.0870</td><td>0.0174</td><td>0.9781</td></tr><tr><td>Transolver* [2024]</td><td>0.0221</td><td>0.0788</td><td>0.0127</td><td>0.9906</td></tr><tr><td>LinearNO* [2026]</td><td>0.0198</td><td>0.0766</td><td>0.0140</td><td>0.9881</td></tr><tr><td>HiLNO</td><td>0.0220</td><td>0.0753</td><td>0.0135</td><td>0.9893</td></tr></table>

## 4.1.3. Shape-Net Car

Finally, we consider the large-scale Shape-Net Car benchmark to evaluate HiLNO on complex three-dimensional vehicle geometries. The task is to predict the surface pressure and surrounding flow velocity from the vehicle geometry. The underlying physics is governed by the three-dimensional Reynolds-averaged Navier–Stokes equations

$$
\begin{array} { r } { \nabla \cdot \bar { \pmb { u } } = 0 , } \\ { \left( \bar { \pmb { u } } \cdot \nabla \right) \bar { \pmb { u } } + \nu \nabla ^ { 2 } \bar { \pmb { u } } + \frac { 1 } { \rho } \nabla \bar { p } + \nabla \cdot \pmb { R } = 0 , } \end{array}\tag{39}
$$

where $\bar { \mathbf { \Omega } } _ { \bar { \mathbf { \Omega } } } ^ { - }$ denotes the time-averaged velocity, $\bar { p }$ is the time-averaged pressure, $\rho$ and ν are the constant fluid density and kinematic viscosity, respectively, and R is the Reynolds stress tensor characterizing the additional momentum transport induced by turbulent fluctuations [39]. Each sample contains 32,186 unstructured points, including approximately

3,700 points on the vehicle surface. Following [45], 789 samples are used for training and   
100 samples for testing.

For this benchmark, HiLNO constructs the hierarchical point sets as

$$
X ^ { 0 } \ ( 3 2 , 1 8 6 )  X ^ { 1 } \ ( 6 , 1 4 4 )  X ^ { 2 } \ ( 3 , 0 7 2 ) .\tag{40}
$$

The latent point sets $X ^ { 1 }$ and $X ^ { 2 }$ are sampled directly from the original point set $X ^ { 0 }$ using stratified random sampling. Specifically, $X ^ { 1 }$ contains 2,048 surface points and 4,096 surrounding flow points, while $X ^ { 2 }$ contains 1,024 surface points and 2,048 surrounding flow points. This sampling strategy allocates a suficient number of latent points to the vehicle surface, where the pressure field is predicted. The weights of the multi-scale supervision in Equation (12) are set to $\omega _ { 0 } = 1 . 0 , \omega _ { 1 } = 0 . 5$ , and $\omega _ { 2 } = 0 . 5$ . We additionally report the relative L2 error of the drag coeficient $C _ { D }$ and the Spearman correlation coeficient $\rho _ { D }$ between the predicted and reference drag coeficients, following [45].

![](images/c8eaf0b949591dd14ee5636ac59a6040f3585ccd063c67141cf6a1dc76f9a6a7.jpg)  
Figure 7: Visualization of surface pressure and pointwise prediction errors on representative test samples, with each row corresponding to a diferent vehicle geometry.

As shown in Table 5, HiLNO achieves competitive performance across all four evaluation metrics. In particular, it obtains the lowest pressure error of 0.0753, while achieving the second-best results for the velocity error, drag-coeficient error $C _ { D }$ , and Spearman correlation coeficient $\rho _ { D }$ . These results indicate that HiLNO maintains accurate surfacepressure prediction while remaining competitive in flow-velocity and aerodynamic quantities. Figure 7 further visualizes the pointwise pressure-prediction errors of HiLNO and Transolver on representative test samples. HiLNO produces relatively small errors over most of the vehicle surface, demonstrating its ability to reconstruct the surface pressure distribution on complex three-dimensional geometries.

## 4.2. Computational Eficiency

We compare the parameter count and computational cost of HiLNO with Transolver and LinearNO on three benchmarks with relatively large spatial discretizations, as reported in Table 6. Both Transolver and LinearNO are competitive recent neural operator baselines. HiLNO consistently requires substantially fewer parameters and lower computational cost across all three benchmarks. Compared with LinearNO, HiLNO reduces the parameter count and FLOPs by 84.4% and 69.2% on average across the three benchmarks, respectively. In particular, HiLNO uses fewer than one million parameters for all three tasks.

The computational eficiency of HiLNO mainly benefits from its hierarchical latent computation and lightweight Gaussian attention. The hierarchical architecture progressively reduces the number of spatial points and performs the main latent processing on compact representations. Meanwhile, Gaussian attention constructs its attention weights directly from relative positions, avoiding feature-dependent query–key similarity computation and requiring only a small number of learnable Gaussian scale parameters. Together, these designs lead to a favorable trade-of between predictive accuracy, parameter count, and computational cost.

Table 6: Eficiency comparison of Transolver, LinearNO, and HiLNO. Params and FLOPs denote the number of model parameters and floating-point operations, respectively.
<table><tr><td>Metric</td><td>Model</td><td>Darcy</td><td>Airfoil</td><td>Shape-Net Car</td></tr><tr><td rowspan="3">Params (M)</td><td>Transolver [2024]</td><td>2.83</td><td>2.81</td><td>3.86</td></tr><tr><td>LinearNO [2026]</td><td>1.77</td><td>1.77</td><td>3.85</td></tr><tr><td>HiLNO</td><td>0.30</td><td>0.179</td><td>0.76</td></tr><tr><td rowspan="3">FLOPs (G)</td><td>Transolver [2024]</td><td>20.87</td><td>32.38</td><td>266.21</td></tr><tr><td>LinearNO [2026]</td><td>13.68</td><td>21.34</td><td>257.80</td></tr><tr><td>HiLNO</td><td>5.39</td><td>3.49</td><td>94.53</td></tr></table>

## 4.3. Understanding the HiLNO

## 4.3.1. Efect of the Hierarchical Latent Space

To examine the efect of the hierarchical latent space, we compare diferent hierarchical configurations on the Darcy benchmark with multi-scale supervision applied at the corresponding intermediate levels, while keeping the finest and coarsest point sets fixed at 85<sup>2</sup> and $1 5 ^ { 2 }$ , respectively. Hierarchy-2 denotes a direct mapping from the fine representation to the coarsest latent point set, whereas Hierarchy-3, Hierarchy-4, and Hierarchy-5 denote progressively deeper hierarchies with additional intermediate resolutions. To assess the effect of parameter count, we construct Hierarchy-2+, which retains the two-level structure but increases the feature dimension to match the parameter count of Hierarchy-5.

Table 7: Efect of the hierarchical latent space on the Darcy benchmark. All variants use the same finest and coarsest point sets, with multi-scale supervision applied at the corresponding intermediate levels. Only the encoder-side hierarchy is shown, while the decoder follows the corresponding symmetric structure.
<table><tr><td>Model</td><td>Hierarchy</td><td>Params (M)</td><td>FLOPs (G)</td><td>Relative L2 Error</td></tr><tr><td>Hierarchy-2</td><td> $8 5 ^ { 2 }  1 5 ^ { 2 }$ </td><td>0.12</td><td>1.21</td><td>0.00555</td></tr><tr><td>Hierarchy-2+</td><td> $8 5 ^ { 2 }  1 5 ^ { 2 }$ </td><td>0.34</td><td>2.91</td><td>0.00574</td></tr><tr><td>Hierarchy-3</td><td> $8 5 ^ { 2 }  4 3 ^ { 2 }  1 5 ^ { 2 }$ </td><td>0.18</td><td>4.76</td><td>0.00396</td></tr><tr><td>Hierarchy-4</td><td> $8 5 ^ { 2 }  4 3 ^ { 2 }  2 9 ^ { 2 }  1 5 ^ { 2 }$ </td><td>0.24</td><td>5.23</td><td>0.00377</td></tr><tr><td>Hierarchy-5</td><td> $8 5 ^ { 2 }  4 3 ^ { 2 }  2 9 ^ { 2 }  2 2 ^ { 2 }  1 5 ^ { 2 }$ </td><td>0.30</td><td>5.39</td><td>0.00372</td></tr></table>

As shown in Table 7, direct compression in Hierarchy-2 results in a relative L2 error of 0.00555, while introducing intermediate latent levels consistently reduces the error to 0.00396, 0.00377, and 0.00372 for Hierarchy-3, Hierarchy-4, and Hierarchy-5, corresponding to reductions of 28.6%, 32.1%, and 33.0%, respectively. In contrast, Hierarchy-2+ achieves an error of 0.00574 despite having more parameters than Hierarchy-5, indicating that the improvement is not simply due to increased parameter count. These results support the use of progressive hierarchical compression for improving latent representation learning. The smaller gains from Hierarchy-4 to Hierarchy-5 also suggest a diminishing return as additional latent levels are introduced, reflecting a trade-of between predictive accuracy and computational cost.

## 4.3.2. Efect of Multi-Scale Supervision

We next examine the contribution of multi-scale supervision by retaining only the loss at the finest discretization in Equation (12), while keeping the hierarchical latent space and all other training settings unchanged. The quantitative comparison is shown in Figure 8. Removing multi-scale supervision consistently degrades the prediction accuracy on Darcy, Airfoil, and Shape-Net Car. In particular, the relative L2 error on Darcy increases from 0.0037 to 0.0041 when multi-scale supervision is removed. These results indicate that explicitly supervising intermediate decoded levels improves the efectiveness of the hierarchical latent representations.

![](images/562d0d3bc8c066f5136d0127d969b33cf151144f415015146820233074279987.jpg)  
Figure 8: Ablation study of multi-scale supervision on Darcy, Airfoil, and Shape-Net Car. All bars report relative L2 errors; lower values indicate better performance.

![](images/93ce165709b4a5faad8e181c128b478425abb1cac50246d0910a5ce2ecdf4f80.jpg)  
Figure 9: Visualization of intermediate decoded predictions with and without (w/o) multi-scale supervision on the Darcy benchmark. Columns correspond to spatial resolutions of $8 5 ^ { 2 } , ~ 4 3 ^ { 2 } , ~ 2 9 ^ { 2 } , ~ 2 2 ^ { 2 }$ , and $1 5 ^ { 2 }$ , respectively. The rows show the ground-truth solutions, predictions of HiLNO, and predictions of HiLNO without multi-scale supervision.

More importantly, multi-scale supervision improves the ability of intermediate decoded representations to make accurate predictions at their corresponding resolutions. Figure 9 shows the predictions at diferent resolutions on the Darcy benchmark. Without multiscale supervision, coarse-level predictions are noticeably blurrier and fail to recover the main solution structures, whereas supervised predictions remain more consistent with the ground truth across resolutions. This suggests that multi-scale supervision guides intermediate representations to preserve solution-relevant information across multiple spatial scales, rather than relying solely on the final-resolution objective.

## 4.3.3. Analysis of Anisotropic Gaussian Attention

We first examine the efect of the locality ratio in anisotropic Gaussian attention on predictive performance. As shown in Figure 10, global attention gives the lowest error, while smaller encoder locality causes only a minor accuracy drop and reduces computation by involving fewer source points. Therefore, in the main experiments, we set $P _ { \mathrm { l o c } } ^ { \mathrm { e n } } = 0 . 1$ and $P _ { \mathrm { l o c } } ^ { \mathrm { d e } } = 1 . 0$ to balance predictive accuracy and computational eficiency.

![](images/eb380d08eba4edb4b4c75a3cf4e654580110021c4f64d8d0533c83cce904fce0.jpg)  
Figure 10: Efect of encoder and decoder locality ratios on the Darcy benchmark. Each cell reports the relative L2 error for the corresponding $( { P _ { \mathrm { l o c } } ^ { \mathrm { e n } } } , { P _ { \mathrm { l o c } } ^ { \mathrm { d e } } } )$ setting.

![](images/073d6deb3558b6b88bfd1bd57e83aae0cda7048bd6035716fa9d7846ef7331f2.jpg)  
Figure 11: Normalized Gaussian attention weights of eight heads in the first encoder block. The red marker denotes the target point, with $P _ { \mathrm { l o c } } ^ { \mathrm { e n } } = 0 . 1$ restricting attention to a local subset of source points.

We further qualitatively examine the spatial patterns learned by anisotropic Gaussian attention. Figure 11 visualizes the normalized attention weights of eight heads in the first encoder block on the Darcy benchmark, where a representative target point is marked in red and the learned Gaussian scales are reported above each head. The heads exhibit distinct spatial interaction patterns, ranging from approximately isotropic distributions to pronounced directional elongation, demonstrating the flexibility of multi-head Gaussian attention in capturing diverse spatial interactions.

## 4.4. Generalization Across Spatial Resolutions

Motivated by the continuous-operator interpretation of Gaussian attention established in Theorem 3.1, we further investigate the cross-resolution generalization of HiLNO. All models are trained exclusively on the $8 5 ^ { 2 }$ grid and directly evaluated on $1 0 6 ^ { 2 } , 1 4 1 ^ { 2 }$ , and $2 1 1 ^ { 2 }$ grids without retraining or fine-tuning. This setting examines whether a model trained on a fixed discretization can be directly applied to unseen spatial resolutions.

As shown in Figure 12, the prediction errors of all considered methods increase as the test grid becomes finer. Nevertheless, HiLNO consistently achieves the lowest relative L2 error at all evaluated resolutions, including those unseen during training. These results demonstrate that HiLNO retains its predictive advantage when generalized across spatial resolutions.

![](images/16fd810eff9d1b6b9a556c4295328daa93b9f3c66bd6aa2a414fbbf7ebb09c79.jpg)  
Figure 12: Generalization across spatial resolutions on the Darcy benchmark. All methods are trained on the $8 5 ^ { 2 }$ grid and directly evaluated at diferent resolutions without retraining or fine-tuning.

## 5. Conclusion

This paper presented HiLNO, a hierarchical latent neural operator for eficient PDE operator learning that constructs a fine-to-coarse-to-fine hierarchical latent space. Within this hierarchical latent space, multi-scale supervision directly guides intermediate predictions using target fields at the corresponding spatial resolutions, encouraging solutionrelevant structures to be captured across multiple spatial scales. Anisotropic Gaussian attention enables feature transfer across diferent levels through learnable directiondependent Gaussian kernels, making HiLNO applicable to general geometries.

Experiments on Darcy, Airfoil, and Shape-Net Car showed that HiLNO achieves competitive predictive accuracy while requiring substantially fewer parameters and lower computational cost than representative eficient neural operator baselines. Ablation studies showed that introducing intermediate hierarchical levels improves predictive accuracy, while multi-scale supervision benefits both final predictions and the predictive capability of intermediate representations. HiLNO also generalizes efectively to spatial resolutions unseen during training. Future work will extend HiLNO to time-dependent PDEs and investigate adaptive spatial supports and physical constraints to further improve physical consistency and generalization.

## Acknowledgements

This work was partially supported by the National Natural Science Foundation of China, No. 12171240. This work is partially supported by High Performance Computing Platform of Nanjing University of Aeronautics and Astronautics.

## References

[1] Alkin, B., Fürst, A., Schmid, S., Gruber, L., Holzleitner, M., Brandstetter, J., 2024. Universal physics transformers: A framework for eficiently scaling neural operators, in: Advances in Neural Information Processing Systems. URL: https://openreview.net/forum?id=oUXiNX5KRm.

[2] Charles, R.Q., Su, H., Kaichun, M., Guibas, L.J., 2017. Pointnet: Deep learning on point sets for 3D classification and segmentation, in: IEEE Conference on Computer Vision and Pattern Recognition. doi:10.1109/CVPR.2017.16.

[3] Chen, J., Wu, K., 2024. Positional knowledge is all you need: Position-induced transformer (PiT) for operator learning, in: International Conference on Machine Learning. URL: https://proceedings.mlr.press/v235/chen24au.html.

[4] Chen, Y., Lu, W., Xu, J., He, Y., Li, W., Zheng, J., 2026. Information-coupled neural operator for computational mechanics and parametric PDEs. Computer Methods in Applied Mechanics and Engineering 453, 118851.

[5] Deng, Z., Meng, Q., Li, Y., Liu, X., Chen, G., Chen, L., Liu, C., Hao, X., 2025. Bvnorm: A neural operator learning framework for parametric boundary value problems on complex geometric domains in engineering. Engineering Applications of Artificial Intelligence 144, 110109.

[6] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2021. An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations. URL: https://openreview. net/forum?id=YicbFdNTTy.

[7] Eshaghi, M.S., Anitescu, C., Valizadeh, N., Wang, Y., Zhuang, X., Rabczuk, T., 2026. NOWS: Neural operator warm starts for accelerating iterative solvers. Computer Methods in Applied Mechanics and Engineering 458, 118989.

[8] Gao, H., Ji, S., 2019. Graph U-Nets, in: International Conference on Machine Learning. URL: https://proceedings.mlr.press/v97/gao19a.html.

[9] George, R.J., Zhao, J., Kossaifi, J., Li, Z., Anandkumar, A., 2024. Incremental spatial and spectral learning of neural operators for solving large-scale PDEs, in: Transactions on Machine Learning Research. URL: https://openreview.net/forum?id= xI6cPQObp0.

[10] Hagnberger, J., Musekamp, D., Niepert, M., 2025. CALM-PDE: Continuous and adaptive convolutions for latent space modeling of time-dependent PDEs, in: Advances in Neural Information Processing Systems. URL: https://openreview.net/ forum?id=0r4yzkvt9j.

[11] Hao, Z., Wang, Z., Su, H., Ying, C., Dong, Y., Liu, S., Cheng, Z., Song, J., Zhu, J., 2023. GNOT: A general neural operator transformer for operator learning, in: International Conference on Machine Learning. URL: https://proceedings.mlr. press/v202/hao23c.html.

[12] He, J., Liu, X., Xu, J., 2024. MgNO: Eficient parameterization of linear operators via multigrid, in: International Conference on Learning Representations. URL: https://proceedings.iclr.cc/paper\_files/paper/2024/file/ eb3c8135137c8a60425a0320869ad87e-Paper-Conference.pdf.

[13] Hu, W., Liu, S., Qiao, P., Sun, Z., Dou, Y., 2026. Transolver is a linear transformer: Revisiting physics-attention through the lens of linear attention, in: Proceedings of the AAAI Conference on Artificial Intelligence. URL: https://doi.org/10.1609/ aaai.v40i1.37003.

[14] Hu, Z., Li, R., 2014. A nonlinear multigrid steady-state solver for 1D microflow. Computers & Fluids 103, 193–203.

[15] Karumuri, S., Graham-Brady, L., Goswami, S., 2026. Physics-informed latent neural operator for real-time predictions of time-dependent parametric PDEs. Computer Methods in Applied Mechanics and Engineering 450, 118599.

[16] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1–97.

[17] Lee, S., Oh, T., 2024. Inducing point operator transformer: A flexible and scalable architecture for solving PDEs, in: Proceedings of the AAAI Conference on Artificial Intelligence. URL: https://doi.org/10.1609/aaai.v38i1.27766.

[18] Li, Z., Huang, D.Z., Liu, B., Anandkumar, A., 2023a. Fourier neural operator with learned deformations for PDEs on general geometries. Journal of Machine Learning Research 24, 18593–18618.

[19] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial diferential equations, in: International Conference on Learning Representations. URL: https: //openreview.net/forum?id=c8P9NQVtmnO.

[20] Li, Z., Meidani, K., Farimani, A.B., 2023b. Transformer for partial diferential equations’ operator learning. Transactions on Machine Learning Research URL: https://openreview.net/forum?id=EPPqt3uERT.

[21] Li, Z., Patil, S., Ogoke, F., Shu, D., Zhen, W., Schneier, M., Buchanan, J.R., Barati Farimani, A., 2025. Latent neural PDE solver: A reduced-order modeling framework for partial diferential equations. Journal of Computational Physics 524, 113705.

[22] Li, Z., Shu, D., Barati Farimani, A., 2023c. Scalable transformer for PDE surrogate modeling, in: Advances in Neural Information Processing Systems. URL: https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 590daf74f99ee85df3d8c007df9c8187-Paper-Conference.pdf.

[23] Liu, Q., Zhong, W., Meidani, H., Abueidda, D., Koric, S., Geubelle, P., 2026a. Geometry-informed neural operator transformer for partial diferential equations on arbitrary geometries. Computer Methods in Applied Mechanics and Engineering 451, 118668.

[24] Liu, S., Yu, Y., Fan, C., Zhang, T., Liu, H., Liu, X., 2026b. Multi-particle neural operator transformer for solving partial diferential equations. Neural Networks 202, 109040.

[25] Liu, X., Xu, B., Cao, S., Zhang, L., 2024. Mitigating spectral bias for the multiscale operator learning. Journal of Computational Physics 506, 112944.

[26] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF International Conference on Computer Vision. doi:10.1109/ ICCV48922.2021.00986.

[27] Longhi, A., Lathouwers, D., Perkó, Z., 2026. Latent space modeling of parametric and time-dependent PDEs using neural odes. Computer Methods in Applied Mechanics and Engineering 448, 118394.

[28] Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations. URL: https://openreview.net/ forum?id=Bkg6RiCqY7.

[29] Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G.E., 2021. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 3, 218–229.

[30] Lu, X., Liu, Y., 2026. Physics-augmented latent fourier neural operator for eficient partial diferential equation solving. Neurocomputing 694, 133883.

[31] Luo, H., Wu, H., Zhou, H., Xing, L., Di, Y., Wang, J., Long, M., 2025. Transolver++: An accurate neural solver for PDEs on million-scale geometries, in: International Conference on Machine Learning. URL: https://openreview.net/forum? id=AM7iAh0krx.

[32] Mazumder, S., 2016. Numerical methods for partial diferential equations: Finite diference and finite volume methods. 1 ed., Academic Press.

[33] Raissi, M., Perdikaris, P., Karniadakis, G., 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics 378, 686–707.

[34] Raonic, B., Molinaro, R., Ryck, T.D., Rohner, T., Bartolucci, F., Alaifari, R., Mishra, S., de Bezenac, E., 2023. Convolutional neural operators for robust and accurate learning of PDEs, in: Advances in Neural Information Processing Systems. URL: https://openreview.net/forum?id=MtekhXRP4h.

[35] Serrano, L., Wang, T.X., Le Naour, E., Vittaut, J.N., Gallinari, P., 2024. AROMA: Preserving spatial structure for latent PDE modeling with local neural fields, in: Advances in Neural Information Processing Systems. URL: https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 185a120a3f709187e68bd092e6098851-Paper-Conference.pdf.

[36] Sun, L., Li, Z., Zhao, Q., 2026. Latent attention operator network with augmented representation for complex PDE systems in intricate geometries. Computer Methods in Applied Mechanics and Engineering 454, 118870.

[37] Tran, A., Mathews, A., Xie, L., Ong, C.S., 2023. Factorized fourier neural operators, in: International Conference on Learning Representations. URL: https: //openreview.net/forum?id=tmIiMPl4IPa.

[38] Tripura, T., Chakraborty, S., 2023. Wavelet neural operator for solving parametric partial diferential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering 404, 115783.

[39] Umetani, N., Bickel, B., 2018. Learning three-dimensional flow for interactive aerodynamic design. ACM Transactions on Graphics 37, 1–10.

[40] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need, in: Advances in Neural Information Processing Systems. URL: https://proceedings.neurips.cc/paper\_files/ paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

[41] Wang, S., Seidman, J., Sankaran, S., Wang, H., Pappas, G., Perdikaris, P., 2025. CViT: Continuous vision transformer for operator learning, in: International Conference on Learning Representations. URL: https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 3bf4b55960aaa23553cd2a6bdc6e1b57-Paper-Conference.pdf.

[42] Wang, T., Wang, C., 2024. Latent neural operator for solving forward and inverse PDE problems, in: Advances in Neural Information Processing Systems. URL: https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 39f6d5c2e310a5a629dcfc4d517aa0d1-Paper-Conference.pdf.

[43] Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., Benson, S.M., 2022. U-FNO– an enhanced fourier neural operator-based deep-learning model for multiphase flow. Advances in Water Resources 163, 104180.

[44] Wu, H., Hu, T., Luo, H., Wang, J., Long, M., 2023. Solving high-dimensional PDEs with latent spectral models, in: International Conference on Machine Learning. URL: https://proceedings.mlr.press/v202/wu23f.html.

[45] Wu, H., Luo, H., Wang, H., Wang, J., Long, M., 2024. Transolver: A fast transformer solver for PDEs on general geometries, in: International Conference on Machine Learning. URL: https://proceedings.mlr.press/v235/wu24r.html.

[46] Xiao, Z., Hao, Z., Lin, B., Deng, Z., Su, H., 2024. Improved operator learning by orthogonal attention, in: International Conference on Machine Learning. URL: https://openreview.net/forum?id=6w7zkf9FBR.

[47] Zhang, Z., Wu, R., Sun, L., Zhang, L., 2025. GPSToken: Gaussian parameterized spatially-adaptive tokenization for image representation and generation, in: Advances in Neural Information Processing Systems. URL: https://openreview.net/forum? id=BxoEDR2yQM.

[48] Zhou, C., Chen, J., Yang, Z., 2026. SAOT: An enhanced locality-aware spectral transformer for solving PDEs, in: Proceedings of the AAAI Conference on Artificial Intelligence. URL: https://doi.org/10.1609/aaai.v40i34.40128.