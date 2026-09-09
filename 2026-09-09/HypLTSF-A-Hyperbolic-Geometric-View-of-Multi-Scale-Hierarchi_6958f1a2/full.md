# HypLTSF: A Hyperbolic Geometric View of Multi-Scale Hierarchies for Long-Term Time Series Forecasting

Namwoo Kim<sup>∗</sup>, Hyungryul Baik<sup>†</sup>, and Yoonjin Yoon<sup>∗‡</sup>

<sup>∗</sup>Urban AI Institute, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, Republic of Korea

<sup>†</sup>Department of Mathematical Science, KAIST, Daejeon, Republic of Korea

<sup>‡</sup>Department of Civil and Environmental Engineering, KAIST, Daejeon, Republic of Korea

Abstract—Multi-scale modeling has become an effective approach for long-term time series forecasting, capturing temporal patterns that range from fine-grained local dynamics to coarse global trends. Representations across these temporal scales are inherently hierarchical, with coarser scales abstracting and aggregating information from finer ones. While existing approaches readily exchange information across these scales, the hierarchy itself is typically left as an emergent byproduct of such interactions rather than captured as a geometric structure in its own right. In this paper, we introduce HypLTSF, a framework that endows the multi-scale hierarchy with a concrete geometric form by embedding scale-wise representations into the Poincare ball,´ whose exponentially expanding volume naturally accommodates hierarchical structures. To align this geometry with the temporal hierarchy, HypLTSF imposes two constraints: (1) a radial constraint that orders embeddings by their level of abstraction, and (2) an angular constraint that groups fine-scale patterns sharing a common coarser-scale ancestor. Extensive experiments on longterm time series forecasting benchmarks show that HypLTSF achieves state-of-the-art performance, suggesting that explicitly modeling the multi-scale hierarchy as a geometric structure is effective for forecasting.

Index Terms—Time series modelling, Time series forecasting, multi-scale modelling, hyperbolic geometry

## I. INTRODUCTION

Time series forecasting has attracted significant attention across diverse domains such as economics [1], [2], energy systems [3], [4], and transportation [5], [6]. With the rapid advancement of deep learning, recent research has increasingly focused on learning expressive temporal representations capable of capturing complex and heterogeneous temporal dynamics. One particularly effective paradigm is multi-scale modeling, which represents a time series at multiple temporal resolutions and learns from them jointly [7]–[13]. Operating across resolutions enables models to capture both local and global dynamics. Coarse-scale representations highlight macroscopic behavior, whereas finer-scale representations encode localized variability and short-term dynamics. Beyond providing parallel views of the data, multi-scale representations inherently form a coarse-to-fine hierarchy over time. Existing forecasting methods instantiate this hierarchy by generating a spectrum of representations from raw observations, typically through temporal downsampling [7], [8], [11], [12] or convolution [9], [13]. As illustrated in Figure 1, repeated application of a downsampling operator or convolution maps fine-scale observations to progressively coarser representations, where each coarse time step aggregates multiple fine-grained observations, naturally inducing parent–child relationships across temporal scales.

![](images/e88e31d3f3602432f909085bfd8fad156f3fe37cdcc7e9c5dadb9407033bc74a.jpg)  
Fig. 1: Illustration of the hierarchical structure induced by multi-scale representations. Fine-scale observations are progressively aggregated into coarser representations, naturally forming parent–child relationships across scales.

Despite this inherent hierarchy, existing multi-scale methods generally do not explicitly encode hierarchical structure within the geometry of the representation space. They primarily facili tate information exchange across scales via parallel processing, residual aggregation, attention, or mixing operations, while not structurally enforcing the parent–child relationship within the representation space. This limitation raises a natural question: can we geometrically model the hierarchy underlying multiscale time series?. Our framework answers in the affirmative by embedding this hierarchy directly into the geometry of the representation space. We leverage hyperbolic space with a radial ordering constraint to encode abstraction depth. This design places coarse parent representations closer to the origin and fine-grained child representations at larger radii, transforming an implicit hierarchy into an explicit architectural prior. Unlike Euclidean space, where volume grows polynomially with radius, hyperbolic space exhibits exponential volume growth. This property closely mirrors the exponential branching structure of trees. As a result, hyperbolic embeddings can represent hierarchical data with significantly lower distortion than their Euclidean counterparts [14], [15]. This advantage has led to notable success in domains characterized by latent hierarchies, including knowledge graphs [16], natural language [17], and molecular structures [18]. However, despite the inherently hierarchical nature of multi-scale temporal representations, hyperbolic geometry has not been widely explored in the context of time series forecasting.

In this paper, we propose HypLTSF, a novel framework that models the hierarchical structure of multi-scale time series in hyperbolic space. We first construct a multi-scale pyramid from the input series via progressive downsampling, decompose each scale into trend and seasonal components, and fuse them into a unified representation. These scalewise representations are then mapped onto the Poincare ball,´ where we introduce two complementary hierarchy losses to explicitly shape the embedding geometry. The radial ordering loss enforces fine-scale embeddings to lie deeper than their coarse-scale counterparts, thereby reflecting the parent–child relationship along the radial axis. The angular coherence loss encourages sibling time steps that are aggregated by the same parent to cluster in similar directions, forming coherent branches in the embedding space. Together, these losses induce a tree-like structure where depth encodes temporal resolution and angular proximity encodes local temporal context. Predictions are generated from each scale and aggregated with learnable weights. Extensive experiments on eight real-world benchmark datasets demonstrate that HypLTSF achieves stateof-the-art forecasting performance. Ablation studies further validate the contributions of hyperbolic embeddings and hierarchy regularization. These results suggest that explicitly modeling the abstraction hierarchy serves as a useful inductive bias that improves multi-scale time series forecasting.

We make the following contributions in this work.

• We propose time series forecasting framework that applies hyperbolic geometry to multi-scale long-term time series forecasting, where the hierarchy is induced by progressive temporal downsampling.

• We introduce two complementary hierarchy losses in the Poincare ball: a radial ordering loss that enforces´ depth separation between parent and child scales, and an angular coherence loss that clusters sibling time steps together.

• Results from comprehensive experiments across eight benchmarks indicate that HypLTSF achieves state-of-theart performance, which verifies the benefits of explicit hierarchical modeling over implicit approaches.

## II. RELATED WORK

## A. Time Series Forecasting

Early deep learning approaches for long-term time series forecasting primarily employed RNNs [19], [20] and CNNs [21], [22] to capture temporal dependencies through recurrent hidden states or convolutional receptive fields. More recently, Transformer- and MLP-based architectures have become the predominant paradigms, demonstrating strong performance.

a) Transformer-based methods.: Transformers have been extensively adapted for long-term time series forecasting by redesigning the self-attention mechanism to handle long sequences efficiently. Informer [23] introduced ProbSparse attention to reduce quadratic complexity. Q-Informer [24] further extends sparse attention with trainable quantum-enhanced query-key scoring. Autoformer [25] replaced canonical attention with an auto-correlation mechanism coupled with trend– seasonal decomposition. FEDformer [26] further improved efficiency by performing attention in the frequency domain. Beyond efficiency, recent work has shifted focus toward the design of input representations and attention targets. MRformer [27] captures both long- and short-term temporal patterns by combining global attention with adaptive segmentwise attention. PatchTST [28] segments time series into subseries-level patches, enabling the model to attend over local semantic units rather than individual time steps. iTransformer [29] inverts the conventional paradigm by applying self-attention across variates instead of time steps, achieving strong multivariate forecasting performance.

b) MLP-based methods.: A parallel line of research has demonstrated that simple MLP architectures can match or even surpass Transformers on standard LTSF benchmarks, challenging the necessity of attention mechanisms. DLinear [30] showed that a single linear layer applied after trend–seasonal decomposition achieves surprisingly competitive performance, prompting a fundamental reconsideration of inductive biases in time series models. TiDE [31] extends this direction with an MLP-based encoder-decoder that incorporates covariates while remaining significantly faster than Transformer counterparts. TSMixer [32] alternates time-mixing and feature-mixing MLP layers inspired by MLP-Mixer [33], and FreTS [34] applies MLPs in the frequency domain to exploit the global view and energy compaction properties of spectral representations. More recently, SparseTSF [35] achieves competitive accuracy with fewer than 1k parameters by decoupling periodicity and trend through cross-period sparse forecasting.

c) Robust forecasting.: Beyond architectural advances, recent studies have explored improving forecasting robustness against nonstationarity and representation perturbations. JointPGM [36] explicitly models time-varying intraseries and inter-series transitional dynamics through a probabilistic graphical framework to address distribution shifts. TopoCL [37] incorporates topological information into contrastive learning to preserve structural properties against augmentation-induced distortions, yielding robust representations applicable to forecasting and other downstream tasks.

## B. Hyperbolic Embedding

Hyperbolic embeddings have been widely studied as an alternative to Euclidean representations for data with inherent hierarchical or tree-like structures. Due to the negative curvature of hyperbolic space, distances grow exponentially with radius, making such spaces particularly well suited for modeling hierarchies, taxonomies, and graphs with power-law degree distributions [15].

In natural language processing, hyperbolic embeddings have been applied to capture semantic hierarchies such as hypernymy relations and lexical taxonomies [15], [17], [38], [39]. More recent work has extended these representations to sentence- and document-level settings [40], [41], typically by projecting Euclidean text representations into hyperbolic space or by incorporating hyperbolic distance-based objectives. These methods have primarily been evaluated on hierarchical text classification and tasks that benefit from explicitly modeling label or semantic hierarchies.

In graph learning, several works have proposed hyperbolic extensions of graph neural networks to model graphs with latent hierarchical structures. HGCN [42] introduced hyperbolic graph convolutions in the Poincare ball, while other´ works explored Lorentzian geometry [18] and fully hyperbolic message passing [43]. HTGN [44] embeds discrete-time temporal graphs in hyperbolic space to jointly capture temporal evolution and latent hierarchy.

In this work, we target continuous multivariate time series, where no relational structure is given a priori and the hierarchy is instead induced by a temporal downsampling operator. To the best of our knowledge, hyperbolic geometry has not been widely studied in the context of long-term time series forecasting.

## C. Multi-Scale Modeling

Multi-scale modeling has emerged as an effective paradigm for time series forecasting, motivated by the observation that temporal dependencies manifest across multiple time resolutions. Real-world time series typically exhibit a mixture of long-term trends, seasonal patterns, and short-term fluctuations, which are difficult to capture within a single temporal scale.

Existing multi-scale approaches derive multi-resolution (often coarse-to-fine) representations from raw data through various mechanisms. Downsampling-based methods such as TimeMixer [7], TimeMixer++ [8], and N-HiTS [45] construct explicit temporal pyramids, where coarser scales summarize or parameterize finer observations. Pyraformer [10] builds multiresolution representations by constructing coarse-scale nodes using convolutions and enables information exchange across scales through pyramidal sparse attention. Ada-MSHyper [46] models multi-scale temporal dependencies via adaptive hypergraph learning and scale-aware hypergraph interactions. Convolution-based methods like MICN [9] employ varying kernel sizes to capture patterns at different temporal granularities, producing coarser representations through larger receptive fields. Similarly, ResMMoT-Informer [47] employs heterogeneous TCN experts with different receptive fields and sparse expert selection to adaptively capture temporal patterns across multiple scales.

These approaches collectively span a fine-to-coarse spectrum of temporal representations. However, while these methods exchange information across scales, they do not impose structural constraint that enforces a hierarchical ordering among scales. On the other hand, proposed framework models the hierarchical relationship directly as a geometric constraint in hyperbolic space. Specifically, we embed multi-scale representations in hyperbolic space, where a radial ordering constraint directly encodes abstraction depth, ensuring that fine-grained (child) representations lie deeper than their coarse (parent) counterparts.

## III. PRELIMINARIES

## A. Poincare ball´

We briefly review the Poincare ball model of hyperbolic space.´ Let $c > 0$ denote the curvature parameter, corresponding to constant sectional curvature −c. The d-dimensional Poincare´ ball is

$$
\mathbb { D } _ { c } ^ { d } = \big \{ \mathbf { x } \in \mathbb { R } ^ { d } : c \| \mathbf { x } \| ^ { 2 } < 1 \big \} ,\tag{1}
$$

equipped with the conformal Riemannian metric

$$
g _ { \bf x } ^ { c } = \lambda _ { \bf x } ^ { 2 } g ^ { E } , \qquad \lambda _ { \bf x } = \frac { 2 } { 1 - c \| { \bf x } \| ^ { 2 } } ,\tag{2}
$$

where $g ^ { E }$ is the Euclidean metric. The geodesic distance between $\mathbf { x } , \mathbf { y } \in \mathbb { D } _ { c } ^ { d }$ is

$$
d _ { c } ( \mathbf { x } , \mathbf { y } ) = \frac { 2 } { \sqrt { c } } \operatorname { a r t a n h } ( \sqrt { c } \parallel - \mathbf { x } \oplus _ { c } \mathbf { y } \parallel ) ,\tag{3}
$$

where $\oplus _ { c }$ is Mobius addition (defined below). A key property¨ is that the distance from the origin reduces to

$$
d _ { c } ( \mathbf { 0 } , \mathbf { x } ) = \frac { 2 } { \sqrt { c } } \mathrm { a r t a n h } ( \sqrt { c } \| \mathbf { x } \| ) ,\tag{4}
$$

which serves as a natural measure of hierarchical depth.

## B. Operations in hyperbolic space

a) Mobius addition and scalar multiplication.:¨ For $\mathbf { x } , \mathbf { y } \in \mathbb { D } _ { c } ^ { d }$ Mobius addition is¨

$$
\mathbf { x } \oplus _ { c } \mathbf { y } = { \frac { ( 1 + 2 c \langle \mathbf { x } , \mathbf { y } \rangle + c \| \mathbf { y } \| ^ { 2 } ) \mathbf { x } + ( 1 - c \| \mathbf { x } \| ^ { 2 } ) \mathbf { y } } { 1 + 2 c \langle \mathbf { x } , \mathbf { y } \rangle + c ^ { 2 } \| \mathbf { x } \| ^ { 2 } \| \mathbf { y } \| ^ { 2 } } } .\tag{5}
$$

Mobius scalar multiplication for¨ $r \in \mathbb { R }$ and $\mathbf { x } \neq \mathbf { 0 }$ is

$$
r \otimes _ { c } \mathbf { x } = { \frac { 1 } { \sqrt { c } } } \operatorname { t a n h } \bigl ( r \operatorname { a r t a n h } ( { \sqrt { c } } \| \mathbf { x } \| ) \bigr ) { \frac { \mathbf { x } } { \| \mathbf { x } \| } } , \qquad r \otimes _ { c } \mathbf { 0 } = \mathbf { 0 } .\tag{6}
$$

b) Exponential and logarithmic maps.: The exponential map at $\mathbf { x } \in \mathbb { D } _ { c } ^ { d }$ sends a tangent vector $\mathbf { v } \in T _ { \mathbf { x } } \mathbb { D } _ { c } ^ { d }$ to the manifold:

$$
\exp _ { \mathbf { x } } ^ { c } ( \mathbf { v } ) = \mathbf { x } \oplus _ { c } \left( \frac { 1 } { \sqrt { c } } \operatorname { t a n h } \left( \frac { \sqrt { c } \lambda _ { \mathbf { x } } \| \mathbf { v } \| } { 2 } \right) \frac { \mathbf { v } } { \| \mathbf { v } \| } \right) ,\tag{7}
$$

$$
\exp _ { \mathbf { x } } ^ { c } ( \mathbf { 0 } ) = \mathbf { x } .
$$

The logarithmic map $\log _ { \mathbf { x } } ^ { c } : \mathbb { D } _ { c } ^ { d } \to T _ { \mathbf { x } } \mathbb { D } _ { c } ^ { d }$ is

$$
\log _ { \mathbf { x } } ^ { c } ( \mathbf { y } ) = \frac { 2 } { \sqrt { c } \lambda _ { \mathbf { x } } } \operatorname { a r t a n h } \bigl ( \sqrt { c } \bigr \| - \mathbf { x } \oplus _ { c } \mathbf { y } \bigr \| \bigr ) \frac { - \mathbf { x } \oplus _ { c } \mathbf { y } } { \bigr \| - \mathbf { x } \oplus _ { c } \mathbf { y } \bigr \| } .\tag{8}
$$

c) Einstein midpoint.: The Einstein midpoint provides a hyperbolic analogue of the Euclidean centroid. Given points $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { n } \subset \mathbb { D } _ { c } ^ { d }$ , it is defined as

$$
\bar { \mathbf { x } } = \frac { \sum _ { i = 1 } ^ { n } \gamma _ { i } \mathbf { x } _ { i } } { \sum _ { i = 1 } ^ { n } \gamma _ { i } } , \qquad \gamma _ { i } = \frac { 1 } { \sqrt { 1 - c \| \mathbf { x } _ { i } \| ^ { 2 } } } ,\tag{9}
$$

where $\gamma _ { i }$ is the Lorentz factor, which assigns higher weight to points closer to the boundary to compensate for the rapid growth of hyperbolic distances in that region.

## C. Problem Statement

Multivariate Time Series Forecasting is the task of predicting future values of multiple variables based on their historical observations. Let $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } \}$ denote a multivariate time series, where $\mathbf { x } _ { t } \in \mathbb { R } ^ { C }$ represents the observations of $C$ variables at time step t. Given a look-back window of length L, the input at time t is defined as $\mathbf { X } _ { t } = [ \mathbf { x } _ { t - L + 1 } , \ldots , \mathbf { x } _ { t } ] \in$ $\mathbb { R } ^ { \boldsymbol { \dot { C } } \times \boldsymbol { L } }$

The objective is to learn a forecasting function $f : \mathbb { R } ^ { C \times L } $ $\mathbb { R } ^ { C \times H }$ that maps the historical observations to future predictions $\hat { \mathbf { Y } } _ { t } = [ \hat { \mathbf { x } } _ { t + 1 } , \hdots , \hat { \mathbf { x } } _ { t + H } ] \in \mathbb { R } ^ { C \times H }$ , where H denotes the prediction horizon. The function f is optimized to minimize the discrepancy between the predicted values $\hat { \mathbf { Y } } _ { t }$ and the ground truth $\mathbf Y _ { t } = [ \mathbf x _ { t + 1 } , \dots , \mathbf x _ { t + H } ]$

## IV. METHODS

We propose HypLTSF, a multi-scale forecasting framework that impose structural constraint that explicitly enforces a hierarchical ordering among scales. Our central hypothesis is that the multi-scale structure of time series—where finegrained fluctuations aggregate into coarser trends—forms a natural hierarchy that can be captured by the radial and angular geometry of hyperbolic space. For better understanding, overview of the HypLTSF can be found in Figure 2.

## A. Multi-scale Decomposition

a) Scale construction.: Given input $\mathbf { x } \in \mathbb { R } ^ { C \times L }$ , we construct S+1 scales by progressive average pooling:

$$
\begin{array} { r } { \mathbf { x } ^ { ( 1 ) } = \mathbf { x } , \qquad \mathbf { x } ^ { ( s + 1 ) } = \operatorname { A v g P o o l } _ { w } ( \mathbf { x } ^ { ( s ) } ) , \quad s = 1 , \dots , S , } \end{array}\tag{10}
$$

where w is the downsampling window. This produces a pyramid $\{ \mathbf { x } ^ { ( s ) } \} _ { s = 1 } ^ { S + 1 }$ with $\mathbf { x } ^ { ( s ) } \in \overline { { \mathbb { R } } } ^ { C \times L _ { s } }$ and $L _ { s } = L / w ^ { s - 1 }$ b) Normalization.: Each scale is independently normalized with learnable affine parameters to handle distribution shift across resolutions [48]:

$$
\tilde { \mathbf { x } } ^ { ( s ) } = \boldsymbol { \gamma } ^ { ( s ) } \odot \frac { \mathbf { x } ^ { ( s ) } - \boldsymbol { \mu } ^ { ( s ) } } { \sigma ^ { ( s ) } } + \boldsymbol { \beta } ^ { ( s ) } ,\tag{11}
$$

where $\mu ^ { ( s ) } , \sigma ^ { ( s ) }$ are per-instance statistics along the time dimension. Here, $\gamma ^ { ( s ) }$ and $\beta ^ { ( s ) }$ are learnable scale and shift parameters, respectively, which allow the model to adaptively rescale and shift the normalized representations at each scale. This normalization is inverted after prediction to recover the original scale.

c) Trend–seasonal decomposition.: Following classical time series analysis [49], the channel-independent representation at each scale is decomposed into trend and seasonal components:

$$
{ \bf x } _ { \mathrm { t r e n d } } ^ { ( s ) } = \mathrm { M o v i n g } \mathrm { A v g } _ { k } \big ( \tilde { \bf x } ^ { ( s ) } \big ) , \qquad { \bf x } _ { \mathrm { s e a s o n } } ^ { ( s ) } = \tilde { \bf x } ^ { ( s ) } - { \bf x } _ { \mathrm { t r e n d } } ^ { ( s ) } .\tag{12}
$$

## B. Euclidean Encoding

Each trend and seasonal component is processed by a dedicated encoder $\mathrm { E n c } ^ { ( s ) }$ that captures both temporal and crossfeature dependencies in two successive stages.

The first stage operates along the time axis with a scalespecific two-layer MLP. Because each scale s has a different temporal length $L _ { s } ~ = ~ L / w ^ { s - 1 }$ , we use separate weight matrices per scale to model resolution-appropriate temporal patterns:

$$
\tilde { \mathbf { x } } ^ { ( s ) } = W _ { 2 } ^ { ( s ) } \mathrm { G E L U } ( W _ { 1 } ^ { ( s ) } \mathbf { x } ^ { ( s ) ^ { \top } } ) ^ { \top } ,\tag{13}
$$

where $W _ { 1 } ^ { ( s ) } , W _ { 2 } ^ { ( s ) } \in \mathbb { R } ^ { L _ { s } \times L _ { s } }$

The second stage operates along the model dimension with a two-layer MLP that is shared across all scales, enabling crossfeature interaction and ensuring a consistent representation space regardless of temporal resolution:

$$
{ \bf h } ^ { ( s ) } = W _ { 4 } \ \mathrm { G E L U } \big ( W _ { 3 } \tilde { \bf x } ^ { ( s ) } \big ) ,\tag{14}
$$

where $W _ { 3 } \in \mathbb { R } ^ { d _ { \mathrm { f f } } \times a }$ <sup>dmodel</sup> and $W _ { 4 } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { \mathrm { f f } } }$

Finally, separate encoders with independent temporal weights but shared feature weights are applied to each season and trend component:

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { s e a s o n } } ^ { ( s ) } = \mathrm { E n c } _ { \mathrm { s e a s o n } } ^ { ( s ) } ( \mathbf { x } _ { \mathrm { s e a s o n } } ^ { ( s ) } ) , \qquad \mathbf { h } _ { \mathrm { t r e n d } } ^ { ( s ) } = \mathrm { E n c } _ { \mathrm { t r e n d } } ^ { ( s ) } ( \mathbf { x } _ { \mathrm { t r e n d } } ^ { ( s ) } ) . } \end{array}\tag{15}
$$

## C. Hyperbolic Encoding

The encoded representations live in Euclidean space and carry no notion of hierarchy. We map them into the Poincare ball´ $\mathbb { D } _ { c } ^ { d _ { h } }$ , whose radial geometry naturally encodes hierarchical depth $\left( \operatorname { E q } . 4 \right)$

a) Component fusion.: Trend and seasonal encodings are fused in Euclidean space with learnable per-scale weights before mapping onto the ball:

$$
\mathbf { f } ^ { ( s ) } = \mathrm { L a y e r N o r m } ( w _ { 1 } ^ { ( s ) } \mathbf { h } _ { \mathrm { s e a s o n } } ^ { ( s ) } + w _ { 2 } ^ { ( s ) } \mathbf { h } _ { \mathrm { t r e n d } } ^ { ( s ) } ) ,\tag{16}
$$

where $w _ { 1 } ^ { ( s ) } , w _ { 2 } ^ { ( s ) }$ are learnable scalars that allow the model to adapt the relative importance of each component at each scale. b) Hyperbolic projection.: The fused representation is first mapped onto $\mathbb { D } _ { c } ^ { d _ { h } }$ via the exponential map at the origin (Eq. 7 with ${ \bf x } = { \bf 0 } )$ , then transformed by a scale-specific hyperbolic linear layer:

$$
\begin{array} { r } { \mathbf { z } ^ { ( s ) } = \mathrm { H y p L i n e a r } ^ { ( s ) } \big ( \exp _ { \mathbf { 0 } } ^ { c } ( \mathbf { f } ^ { ( s ) } ) \big ) . } \end{array}\tag{17}
$$

The hyperbolic linear layer maps $d _ { \mathrm { m o d e l } }  d _ { h }$ via Mobius¨ matrix-vector multiplication and bias addition:

$$
\mathrm { H y p L i n e a r } ( \mathbf { x } ) = M \otimes _ { c } \mathbf { x } \oplus _ { c } \exp _ { \mathbf { 0 } } ^ { c } ( \mathbf { b } ) ,\tag{18}
$$

where $M \ \in \ \mathbb { R } ^ { d _ { h } \times d _ { \mathrm { m o d e l } } }$ is a learnable weight matrix and b $\in \mathbb { R } ^ { d _ { h } }$ is a learnable bias mapped onto the ball via the exponential map before Mobius addition. The M¨ obius matrix-¨ vector product is defined as

$$
M \otimes _ { c } \mathbf { x } = \frac { 1 } { \sqrt { c } } \operatorname { t a n h } \Bigl ( \frac { \| M \mathbf { x } \| } { \| \mathbf { x } \| } \operatorname { a r t a n h } ( \sqrt { c } \| \mathbf { x } \| ) \Bigr ) \frac { M \mathbf { x } } { \| M \mathbf { x } \| } .\tag{19}
$$

In practice, a projection that clips the norm to $( 1 - \epsilon ) / \sqrt { c }$ is applied after each Mobius operation to ensure numerical stability.¨ The result is a set of hyperbolic embeddings $\{ \mathbf { z } ^ { ( s ) } \} _ { s = 1 } ^ { S + 1 }$ . Then, our hierarchy losses (Section IV-D) shape these embeddings to impose a hierarchical ordering across scales, with finer scales embedded deeper than coarser scales.

![](images/39cf3875e2f269a88109c98bc737bc5521546507e6a9031f7f025e6dbc9b4306.jpg)  
Fig. 2: Overview of the HypLTSF. We build a multi-scale pyramid via progressive downsampling, decompose each scale into trend and seasonality, and fuse them to obtain scale-wise features. The scale-wise features are then mapped to the Poincare´ ball via hyperbolic encoding, where two complementary hierarchy losses shape the embedding geometry. Finally, predictions are generated in the tangent space.

## D. Hierarchy Losses

The downsampling procedure creates natural parent–child relationships across scales: each time step at scale s+1 (parent) corresponds to w consecutive time steps at scale s (children):

$$
\mathrm { c h i l d r e n } ( t , s + 1 ) = \{ \mathbf { z } _ { w t } ^ { ( s ) } , \ \mathbf { z } _ { w t + 1 } ^ { ( s ) } , \ \ldots , \ \mathbf { z } _ { w t + w - 1 } ^ { ( s ) } \} .\tag{20}
$$

Each scale embedding has shape $C \times L _ { s } \times d _ { h }$ . To compute the hierarchy losses, we first aggregate across channels via the Einstein midpoint (Eq. 9), yielding a channel-aggregated representation $\bar { \mathbf { z } } ^ { ( s ) } \in \mathbb { R } ^ { \bar { L _ { s } } \times d _ { h } }$ on which all subsequent loss computations are performed. We enforce the hierarchy along two complementary geometric axes.

a) Radial ordering.: In the Poincare ball, hierarchical depth´ is encoded by distance from the origin (Eq. 4). We require that every child individually lies deeper than its parent by a learnable margin m:

$$
\begin{array} { l }  { \displaystyle { \mathcal { L } } _ { \mathrm { r } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \frac { 1 } { L _ { s + 1 } \cdot w } \sum _ { t } \sum _ { j = 0 } ^ { w - 1 } \Bigl [ \mathrm { m a x } \bigl ( 0 , ~ d _ { c } ( { \bf 0 } , \bar { \bf z } _ { t } ^ { ( s + 1 ) } ) } \\ { { \displaystyle ~ - ~ d _ { c } ( { \bf 0 } , \bar { \bf z } _ { w t + j } ^ { ( s ) } ) + m \bigr ) \Bigr ] ^ { 2 } } . } \end{array}\tag{21}
$$

The squared hinge provides a strong gradient signal on large violations while contributing zero loss when the constraint is already satisfied. The learnable margin allows the model to discover the appropriate radial separation for each dataset.

b) Angular coherence.: Radial ordering establishes hierarchy levels but leaves the angular arrangement unconstrained— points at the correct depth could scatter in arbitrary directions, which does not constitute a tree. To induce branch structure, we measure angular coherence in the tangent space at the parent, which provides a geometrically principled local coordinate system.

For parent t at scale s+1, we compute the tangent vector from the parent to each child via the logarithmic map (Eq. 8):

$$
\mathbf { v } _ { t , j } = \log _ { \bar { \mathbf { z } } _ { t } ^ { ( s + 1 ) } } ^ { c } ( \bar { \mathbf { z } } _ { w t + j } ^ { ( s ) } ) , \qquad j = 0 , \ldots , w - 1 .\tag{22}
$$

The branch direction is defined as the normalised mean of these tangent vectors:

$$
\hat { \mathbf { b } } _ { t } = \frac { \sum _ { j = 0 } ^ { w - 1 } \mathbf { v } _ { t , j } } { \left\| \sum _ { j = 0 } ^ { w - 1 } \mathbf { v } _ { t , j } \right\| } .\tag{23}
$$

The angular coherence loss pulls each child’s tangent direction toward the branch direction:

$$
\mathcal { L } _ { \mathrm { a } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \frac { 1 } { L _ { s + 1 } \cdot w } \sum _ { t } \sum _ { j = 0 } ^ { w - 1 } \Bigl ( 1 - \bigl \langle \hat { \mathbf { v } } _ { t , j } , \hat { \mathbf { b } } _ { t } \bigr \rangle \Bigr ) ,\tag{24}
$$

where $\hat { \mathbf { v } } = \mathbf { v } / \| \mathbf { v } \|$ . By operating in the tangent space at the parent rather than using global directions from the origin, this formulation respects the local geometry of the Poincare ball´ and produces more meaningful angular comparison.

c) Joint effect.: The two losses shape complementary axes of the Poincare ball:´

${ \mathcal { L } } _ { \mathrm { r } }$ organizes the radial axis—children are positioned deeper than their parents, closer to the boundary.

$\mathcal { L } _ { \mathrm { a } }$ organizes the angular axis—siblings cluster in the tangent space at their parent, forming coherent subtrees.

Together, they induce a tree-like embedding where hierarchical depth corresponds to temporal resolution and angular proximity corresponds to local temporal context. For better understanding, graphical illustration of hierarchy losses is shown in Figure 3.

![](images/f5c3ece9a9611911aad6de2cc914e28849df3f07a850012a35fa4fb317737e3d.jpg)  
Fig. 3: Illustration of the two hierarchy-preserving losses in the Poincare disk. (a) The radial ordering loss enforces that coarse-´ scale (parent) embeddings lie relatively closer to the origin than fine-scale (child) embeddings, with their radii separated by a learnable margin m. (b) The angular coherence loss encourages children of the same parent to be angularly close in the tangent space at the parent, forming coherent branches.

## E. Prediction

Predictions are generated in the tangent space at the origin, which is locally Euclidean and amenable to standard linear operations.

a) Tangent-space predictor.: For each scale, we map hyperbolic embeddings back via the logarithmic map (Eq. 8 with x = 0) and apply scale-specific projection:

$$
\hat { \mathbf { y } } ^ { ( s ) } = W _ { \mathrm { d i m } } \Big ( W _ { \mathrm { t i m e } } ^ { ( s ) } \mathrm { G E L U } \big ( \log \mathbf { 0 } ^ { c } ( \mathbf { z } ^ { ( s ) } ) \big ) \Big ) ^ { \top } ,\tag{25}
$$

where $W _ { \mathrm { t i m e } } ^ { ( s ) } \in \mathbb { R } ^ { H \times L _ { s } }$ projects along the temporal dimension and $W _ { \mathrm { d i m } } ~ \in ~ \mathbb { R } ^ { 1 \times d _ { h } }$ reduces to univariate predictions per channel.

b) Multi-scale aggregation.: The final forecast is a learnable weighted combination of all scales:

$$
\hat { \mathbf { y } } = \sum _ { s = 1 } ^ { S + 1 } w ^ { ( s ) } \hat { \mathbf { y } } ^ { ( s ) } , \qquad w ^ { ( s ) } = \frac { \exp ( \delta ^ { ( s ) } ) } { \sum _ { s ^ { \prime } } \exp ( \delta ^ { ( s ^ { \prime } ) } ) } ,\tag{26}
$$

where $\{ \delta ^ { ( s ) } \}$ are learnable parameters.

## F. Training Objective

The complete objective combines the prediction loss with the hierarchy constraints:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r } } + \lambda _ { a } \mathcal { L } _ { \mathrm { a } } ,\tag{27}
$$

where $\lambda _ { r }$ and $\lambda _ { a }$ control the strength of each regularizer. For $\mathcal { L } _ { \mathrm { p r e d } }$ , the L1 loss is employed. The hierarchy losses serve as inductive biases: rather than discovering hierarchical structure from the prediction signal alone, the model is guided toward representations where temporal resolution maps to hyperbolic depth and local context maps to angular position.

## V. EXPERIMENTS

## A. Experimental Setup

We evaluate HypLTSF on eight widely used forecasting benchmarks, including ETT, Weather, Electricity, Traffic, and Solar. The baselines cover a broad range of forecasting methods, including Amplifier [50], iTransformer [29], PatchTST [28], Crossformer [51], TimesNet [21], DLinear [30], FEDformer [26], and Non-stationary Transformer (Stationary) [52]. Recent multi-scale approaches, including RAFT [53], TimeKAN [11], and TimeMixer [7], are also included for comparison. This diverse selection enables a thorough comparison across fundamentally different modeling strategies.

Following [54]–[56], we do not apply the drop-last strategy to ensure fair comparison. All methods are compared at four prediction horizons, namely 96, 192, 336, and 720, with MSE and MAE as evaluation metrics. To account for the sensitivity of different models to the input context length, we consider multiple look-back window sizes $L \in \{ 9 6 , 3 3 6 , 5 1 2 \}$ for all datasets and report each method’s best performance. This protocol ensures both consistent evaluation and fair model comparison across varying temporal dependencies. All experiments are conducted using PyTorch with Python 3.8 and the Adam optimizer on a single NVIDIA RTX 3090 GPU. Each experiment is repeated over three independent runs.

## B. Main Results

Table I summarizes the forecasting performance, with the top and runner-up results in each row marked in bold red and underlined blue, respectively. The Avg row reports the perdataset mean across the four prediction horizons, providing a compact summary of overall performance on each benchmark. From these results, we make the following observations.

First, HypLTSF achieves the strongest overall performance across the eight benchmarks, with a particularly pronounced advantage in MAE. At the dataset-average level, it attains the lowest MAE on all eight datasets and the lowest MSE on six. This advantage is even more evident across forecasting horizons: among the 32 settings, HypLTSF ranks first in 30 cases for MAE and 20 for MSE. The stronger gains in MAE, together with consistently competitive MSE results, indicate that the proposed multi-scale hyperbolic hierarchy provides robust improvements across different error criteria. Notably, the improvements hold across both the ETT benchmarks and larger-scale datasets, including Weather, Electricity, Solar, and Traffic, demonstrating consistent benefits across diverse forecasting regimes.

Second, compared with recent baselines, HypLTSF remains competitive with TimeKAN and RAFT on the ETT benchmarks, where performance differences among the top methods are relatively small. While TimeKAN achieves slightly lower average MSE on ETTh2 (0.350 vs. 0.352) and ETTm1 (0.344 vs. 0.347), HypLTSF consistently yields the lowest average MAE across all four ETT datasets. Its advantage becomes more pronounced on the larger datasets. On Solar at H=720,

HypLTSF reduces MSE by 4.8% (0.209→0.199) and MAE by 16.4% (0.269→0.225) relative to TimeKAN. Similarly, on Traffic, it achieves the best average MAE, improving upon the strongest competing baseline by 6.5% (0.275→0.257). These results indicate that the benefits of the proposed hyperbolic hierarchy remain consistent across datasets and become particularly pronounced in larger, high-dimensional forecasting settings.

Third, against TimeMixer, a representative multi-scale forecasting model that shares the same MLP-based backbone, HypLTSF exhibits consistent dataset-average improvements, with notable gains on Solar, where MAE is reduced by 14.7% (0.252→0.215), and on Traffic, where MAE is reduced by 7.9% (0.279→0.257). Given their similar architectures, these results suggest that explicitly modeling the multi-scale hierarchy in hyperbolic space provides an effective geometric prior for organizing representations across temporal scales.

## C. Ablation Studies

To examine the contribution of each hierarchy loss, we conduct ablation studies on four datasets that span both small-scale (ETTh1, ETTh2) and large-scale (Solar, Traffic) settings. For each dataset, we report MSE at two representative prediction horizons, $H { = } 9 6$ and $H { = } 7 2 0$ , corresponding to short-term and long-term forecasting respectively. This setup allows us to assess whether the proposed losses generalize across datasets of varying scale and across different forecasting ranges. Results are summarized in Table II.

a) Loss complementarity.: The radial ordering loss $\mathcal { L } _ { r }$ generally provides the larger individual contribution, particularly on ETTh2, where $\mathcal { L } _ { r }$ alone nearly recovers the full model’s improvement (e.g., 0.410→0.399 at H=720). This suggests that radial depth ordering is a key mechanism for encoding the coarse-to-fine hierarchy. However, the relative contribution varies across datasets, with $\mathcal { L } _ { a }$ providing greater standalone improvements in some settings such as Traffic.

More importantly, the two losses are most effective when combined. On ETTh1 at H=720, for example, neither loss alone yields a meaningful improvement, whereas their combination reduces MSE from 0.443 to 0.433. Similar patterns are observed on Solar. These results demonstrate the complementarity of the two constraints: $\mathcal { L } _ { r }$ organizes representations along the radial axis, while $\mathcal { L } _ { a }$ structures their angular relationships, jointly forming a hierarchical representation that neither constraint consistently achieves alone.

b) Effect grows with horizon and dataset scale.: The benefit of hierarchy losses is generally larger at the long-term horizon (H=720) than at the short-term horizon (H=96), and is most pronounced on the large-scale Solar dataset, where the full model reduces MSE by 9.1% at H=720. Even on Traffic, a 862-channel dataset where channel-wise variance often overshadows temporal structure, hierarchy losses still yield consistent improvements, reaching 1.4% MSE reduction at H=720 and reinforcing the pattern that longer horizons benefit more from explicit hierarchical structure.

## D. Emergence of Hierarchical Structure.

To examine how the hierarchy losses shape the embedding geometry during training, we visualize the per-scale hyperbolic depth across training iterations. Figure 4(a)–(b) plot the mean depth $d _ { c } ( \mathbf { 0 } , \mathbf { z } ^ { ( s ) } )$ for each scale s, measured on the test set at regular intervals on ETTh1 and ETTh2 over 2 training epochs.

At initialization, there is no hierarchical separation. As training progresses, a clear depth ordering emerges. The final depth ordering, Scale 1 > Scale 2 > Scale 3, confirms that the hierarchy losses successfully induce the intended structure. Finer temporal resolutions are embedded deeper in the Poincare ball,´ closer to the boundary, whereas coarser resolutions remain shallow near the origin. Figure 4(c)–(d) further illustrate this separation by showing the per-scale distribution of $d _ { c } ( \mathbf { 0 } , \mathbf { z } )$ on the test set after training, where the scales occupy clearly distinct radial bands with minimal overlap.

![](images/df567e855fd474058633e649192e339fd4de733eb7f1ffb3eb6b7072aa936531.jpg)

![](images/fc83d33a76bc7b081be2f80b42310f57322f9d654ca4f0885fe5963f9fb912aa.jpg)  
(b) ETTh2 depth emergence

(a) ETTh1 depth emergence  
![](images/3c52253a11ebdad73c2bca566abd0d99862dafae1d2a5704ae79e69de3555619.jpg)

![](images/4b2a83ca39d98d06dda369405cc7e8b6a7bc06fb3448e95a1a9a8e64a6fed9ec.jpg)  
(c) ETTh1 radial depth  
(d) ETTh2 radial depth  
Fig. 4: Emergence of hierarchical structure. (a)–(b) Per-scale mean hyperbolic depth over training iterations on ETTh1 and ETTh2. (c)–(d) Distribution of hyperbolic depths by scale on the test set after training.

## E. Advantage of Hyperbolic Space

To further investigate how effective our choice of modeling the multi-scale hierarchy in hyperbolic space is, we conduct a controlled comparison against an Euclidean counterpart. This experiment directly tests whether the geometric inductive bias of hyperbolic space contributes to forecasting performance beyond what an architecturally identical Euclidean variant can achieve.

a) Controlled comparison.: To isolate the effect of geometry from model capacity, we construct a Euclidean baseline by replacing HypLinear (Eq. 18) with a standard Linear layer, while keeping all other architectural components unchanged. Since the two layers share identical weight and bias dimensions, the total number of learnable parameters remains the same. For the hierarchy-preserving losses, we substitute their

TABLE I: Multivariate forecasting results with forecasting horizons $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ for the datasets. We consider multiple look-back window sizes $L \in \{ 9 6 , 3 3 6 , 5 1 2 \}$ for all datasets and report each method’s best performance. The Avg row reports the average MSE/MAE across the four horizons for each dataset.
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=19></td></tr><tr><td rowspan=1 colspan=1>Metrics</td><td rowspan=1 colspan=1>mse mae</td><td rowspan=1 colspan=3>mse mae | mse mae</td><td rowspan=1 colspan=3>mse mae</td><td rowspan=1 colspan=1>mse mae</td><td rowspan=1 colspan=4>mse mae</td><td rowspan=1 colspan=1>mse mae</td><td rowspan=1 colspan=2>mse mae</td><td rowspan=1 colspan=1>mse mae</td><td rowspan=1 colspan=1>mse mae</td><td rowspan=1 colspan=1>msemae</td><td rowspan=1 colspan=1>msemae</td></tr><tr><td rowspan=4 colspan=1>96ET1192336720</td><td rowspan=2 colspan=1>0.3610.3940.4010.418</td><td rowspan=2 colspan=1>0.3700.3960.4030.417</td><td rowspan=2 colspan=2>0.3730.3990.4140.420</td><td rowspan=2 colspan=3>0.3670.3970.4110.427</td><td rowspan=2 colspan=1>0.3860.4050.4240.440</td><td rowspan=2 colspan=4>0.3720.4010.4130.430</td><td rowspan=1 colspan=1>0.3770.397</td><td rowspan=1 colspan=2>0.4110.435</td><td rowspan=2 colspan=1>0.3890.4120.4400.443</td><td rowspan=2 colspan=1>0.3790.4030.4080.419</td><td rowspan=2 colspan=1>|0.5910.5240.6150.540</td><td rowspan=2 colspan=1>0.3790.4190.4200.444</td></tr><tr><td rowspan=1 colspan=1>0.4090.425</td><td rowspan=1 colspan=2>0.4090.438</td></tr><tr><td rowspan=1 colspan=1>0.4230.436</td><td rowspan=1 colspan=1>0.4200.432</td><td rowspan=1 colspan=2>0.4420.446</td><td rowspan=1 colspan=3>0.4360.442</td><td rowspan=1 colspan=1>0.4490.460</td><td rowspan=1 colspan=4>0.4380.450</td><td rowspan=1 colspan=1>0.4310.444</td><td rowspan=1 colspan=2>0.4330.457</td><td rowspan=1 colspan=1>0.5230.487</td><td rowspan=1 colspan=1>0.4400.440</td><td rowspan=1 colspan=1>0.6320.551</td><td rowspan=1 colspan=1>0.4580.466</td></tr><tr><td rowspan=1 colspan=1>0.4330.453</td><td rowspan=1 colspan=1>0.4420.463</td><td rowspan=1 colspan=2>0.4550.467</td><td rowspan=1 colspan=3>0.4670.478</td><td rowspan=1 colspan=1>0.4950.487</td><td rowspan=1 colspan=4>0.4860.484</td><td rowspan=1 colspan=1>0.4570.477</td><td rowspan=1 colspan=2>0.5010.514</td><td rowspan=1 colspan=1>0.5210.495</td><td rowspan=1 colspan=1>0.4710.493</td><td rowspan=1 colspan=1>0.8280.658</td><td rowspan=1 colspan=1>0.4740.488</td></tr><tr><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1>0.4050.425</td><td rowspan=1 colspan=1>|0.4090.427</td><td rowspan=1 colspan=2>|0.4210.433 |</td><td rowspan=1 colspan=3>0.4200.436 |</td><td rowspan=1 colspan=1>0.4390.448</td><td rowspan=1 colspan=4>|0.4270.441|</td><td rowspan=1 colspan=1>0.4190.436 |</td><td rowspan=1 colspan=2>0.4390.461 |</td><td rowspan=1 colspan=1>0.4680.459</td><td rowspan=1 colspan=1>||0.424 0.439 |</td><td rowspan=1 colspan=1>0.666 0.568 |</td><td rowspan=1 colspan=1>0.4330.454</td></tr><tr><td rowspan=6 colspan=1>96ET2192336720</td><td rowspan=1 colspan=1>0.2750.335</td><td rowspan=1 colspan=1>|0.2800.343</td><td rowspan=1 colspan=2>0.2870.349</td><td rowspan=1 colspan=3>|0.2760.344</td><td rowspan=1 colspan=1>0.2970.348</td><td rowspan=1 colspan=4>|0.2810.351</td><td rowspan=1 colspan=1>0.2740.337</td><td rowspan=1 colspan=2>0.7280.603</td><td rowspan=1 colspan=1>0.3340.370</td><td rowspan=1 colspan=1>|0.3000.364</td><td rowspan=1 colspan=1>|0.3470.387</td><td rowspan=1 colspan=1>|0.3370.380</td></tr><tr><td rowspan=5 colspan=1>0.3510.3810.3820.4090.3990.432</td><td rowspan=5 colspan=1>0.3290.3820.3700.4120.4200.450</td><td rowspan=1 colspan=2>0.3480.393</td><td rowspan=1 colspan=3>0.3470.393</td><td rowspan=4 colspan=1>0.3720.4030.3880.417</td><td rowspan=4 colspan=4>0.3490.3870.3660.413</td><td rowspan=4 colspan=1>0.3480.3840.3770.416</td><td rowspan=4 colspan=2>0.7230.6070.7400.628</td><td rowspan=4 colspan=1>0.4040.4130.3890.435</td><td rowspan=4 colspan=1>0.3870.4230.4900.487</td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=3 colspan=3>0.3760.425</td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=2 colspan=1>0.3790.4180.3580.413</td><td rowspan=3 colspan=1>0.4150.4280.3890.4570.4830.488</td></tr><tr><td rowspan=1 colspan=1>0.383</td><td></td></tr><tr><td rowspan=1 colspan=2>0.4070.444</td><td rowspan=1 colspan=3>0.4360.473</td><td rowspan=1 colspan=1>0.4240.444</td><td rowspan=1 colspan=4>0.4010.436</td><td rowspan=1 colspan=1>0.4060.441</td><td rowspan=1 colspan=2>1.3860.882</td><td rowspan=1 colspan=1>0.4340.448</td><td rowspan=1 colspan=1>0.7040.597</td><td rowspan=1 colspan=1>0.4220.457</td></tr><tr><td rowspan=1 colspan=1>Avg |</td><td rowspan=1 colspan=1>0.3520.389|</td><td rowspan=1 colspan=1>0.3500.397</td><td rowspan=1 colspan=2>|0.3560.402</td><td rowspan=1 colspan=3>|0.3590.409|</td><td rowspan=1 colspan=1>0.3700.403</td><td rowspan=1 colspan=4>0.3490.397</td><td rowspan=1 colspan=1>|0.3510.395 ||</td><td rowspan=1 colspan=2>0.8940.680 |</td><td rowspan=1 colspan=1>0.3900.416 |</td><td rowspan=1 colspan=1>0.4700.468 |</td><td rowspan=1 colspan=1>0.3770.419 |</td><td rowspan=1 colspan=1>0.4060.438</td></tr><tr><td rowspan=3 colspan=1>96ETm1192336720</td><td rowspan=2 colspan=1>0.2860.3320.3250.3590.3590.379</td><td rowspan=2 colspan=1>0.2900.3480.3320.3680.3540.386</td><td rowspan=1 colspan=2>|0.2920.3460.3270.365</td><td rowspan=1 colspan=3>|0.3020.3490.3290.367</td><td rowspan=1 colspan=1>0.3000.3530.3410.380</td><td rowspan=1 colspan=4>0.2930.3450.3350.372</td><td rowspan=1 colspan=1>0.2890.3430.3290.368</td><td rowspan=1 colspan=2>|0.3140.3670.3740.410</td><td rowspan=2 colspan=1>0.3400.3780.3920.4040.4230.426</td><td rowspan=2 colspan=1>|0.3000.3450.3360.3660.3670.386</td><td rowspan=2 colspan=1>|0.4150.4100.4940.4510.5770.490</td><td rowspan=2 colspan=1>|0.4630.4630.5750.5160.6180.544</td></tr><tr><td rowspan=1 colspan=2>0.3650.386</td><td rowspan=1 colspan=3>0.3550.383</td><td rowspan=1 colspan=1>0.3740.396</td><td rowspan=1 colspan=4>0.3680.386</td><td rowspan=1 colspan=1>0.3620.390</td><td rowspan=1 colspan=2>0.4130.432</td></tr><tr><td rowspan=1 colspan=1>0.4170.410</td><td rowspan=1 colspan=1>0.4010.417</td><td rowspan=1 colspan=2>0.4270.419</td><td rowspan=1 colspan=3>0.4060.413</td><td rowspan=1 colspan=1>0.4290.430</td><td rowspan=1 colspan=4>0.4260.417</td><td rowspan=1 colspan=1>0.4160.423</td><td rowspan=1 colspan=2>0.7530.613</td><td rowspan=1 colspan=1>0.4750.453</td><td rowspan=1 colspan=1>0.4190.416</td><td rowspan=1 colspan=1>0.6360.535</td><td rowspan=1 colspan=1>0.6120.551</td></tr><tr><td rowspan=1 colspan=1>Avg|</td><td rowspan=1 colspan=1>0.3470.370|</td><td rowspan=1 colspan=1>0.3440.380</td><td rowspan=1 colspan=2>|0.3530.379</td><td rowspan=1 colspan=3>|0.3480.378</td><td rowspan=1 colspan=1>0.3610.390 |</td><td rowspan=1 colspan=4>0.3550.380|</td><td rowspan=1 colspan=1>0.3490.381 |</td><td rowspan=1 colspan=2>0.4640.455 |</td><td rowspan=1 colspan=1>0.4070.415</td><td rowspan=1 colspan=1>|0.3560.378</td><td rowspan=1 colspan=1>|0.5300.472 |</td><td rowspan=1 colspan=1>0.5670.519</td></tr><tr><td rowspan=3 colspan=1>96ETm2192336720</td><td rowspan=1 colspan=1>0.1640.2480.2190.286</td><td rowspan=1 colspan=1>0.1640.2540.2380.300</td><td rowspan=1 colspan=2>0.1640.2540.2260.300</td><td rowspan=1 colspan=3>0.1640.2560.2190.296</td><td rowspan=1 colspan=1>0.1750.2660.2420.312</td><td rowspan=1 colspan=4>0.1650.2560.2250.298</td><td rowspan=1 colspan=1>0.1650.2550.2210.293</td><td rowspan=1 colspan=2>|0.2960.3910.3690.416</td><td rowspan=1 colspan=1>|0.1890.2650.2540.310</td><td rowspan=1 colspan=1>0.1640.2550.2240.304</td><td rowspan=1 colspan=1>|0.2100.2940.3380.373</td><td rowspan=1 colspan=1>|0.2160.3090.2970.360</td></tr><tr><td rowspan=2 colspan=1>0.2680.3210.3460.372</td><td rowspan=2 colspan=1>0.2780.3310.3590.387</td><td rowspan=1 colspan=2>0.2760.331</td><td rowspan=1 colspan=3>0.2750.336</td><td rowspan=2 colspan=1>0.2820.3370.3750.394</td><td rowspan=2 colspan=4>0.2770.3320.3600.387</td><td rowspan=2 colspan=1>0.2760.3270.3620.381</td><td rowspan=1 colspan=2>0.5880.600</td><td rowspan=1 colspan=1>0.3130.345</td><td rowspan=1 colspan=1>0.2770.337</td><td rowspan=2 colspan=1>0.4320.4160.5540.476</td><td rowspan=2 colspan=1>0.3660.4000.4590.450</td></tr><tr><td rowspan=1 colspan=2>0.3580.388</td><td rowspan=1 colspan=3>0.3590.392</td><td rowspan=1 colspan=2>0.7500.612</td><td rowspan=1 colspan=1>0.4130.402</td><td rowspan=1 colspan=1>0.3710.401</td></tr><tr><td rowspan=1 colspan=1>Avg </td><td rowspan=1 colspan=1>| 0.2490.307</td><td rowspan=1 colspan=1>|0.2600.318</td><td rowspan=1 colspan=2>|0.256|0.318</td><td rowspan=1 colspan=3>0.2540.320 |</td><td rowspan=1 colspan=1>0.2680.3270</td><td rowspan=1 colspan=4>.2570.318|</td><td rowspan=1 colspan=1>0.2560.314 |</td><td rowspan=1 colspan=2>0.5010.505 |</td><td rowspan=1 colspan=1>0.2920.331</td><td rowspan=1 colspan=1>|0.2590.324 </td><td rowspan=1 colspan=1>|0.3840.390</td><td rowspan=1 colspan=1>|0.3350.380</td></tr><tr><td rowspan=4 colspan=1>96Weaather192336720</td><td rowspan=4 colspan=1>0.1460.1890.1900.2300.2410.2710.3150.322</td><td rowspan=2 colspan=1>0.1510.2020.1950.244</td><td rowspan=1 colspan=2>0.1470.199</td><td rowspan=2 colspan=3>|0.1650.2220.2110.264</td><td rowspan=2 colspan=1>|0.1570.2070.2000.248</td><td rowspan=2 colspan=4>0.1470.1980.1920.243</td><td rowspan=2 colspan=1>0.1490.1960.1910.239</td><td rowspan=2 colspan=2>0.1430.2100.1980.260</td><td rowspan=3 colspan=1>0.1680.2140.2190.2620.2780.302</td><td rowspan=3 colspan=1>0.1700.2300.2160.2730.2580.307</td><td rowspan=3 colspan=1>|0.1880.2420.2410.2900.3410.341</td><td rowspan=3 colspan=1>| 0.2290.2980.2650.3340.3300.372</td></tr><tr><td rowspan=1 colspan=2>0.1940.245</td></tr><tr><td rowspan=1 colspan=1>0.2420.287</td><td rowspan=1 colspan=2>0.2430.282</td><td rowspan=1 colspan=3>0.2600.302</td><td rowspan=1 colspan=1>0.2520.287</td><td rowspan=1 colspan=4>0.2470.284</td><td rowspan=1 colspan=1>0.2420.279</td><td rowspan=1 colspan=2>0.2580.314</td></tr><tr><td rowspan=1 colspan=1>0.3170.340</td><td rowspan=1 colspan=2>0.3100.329</td><td rowspan=1 colspan=3>0.3270.355</td><td rowspan=1 colspan=1>0.3200.336</td><td rowspan=1 colspan=4>0.3180.330</td><td rowspan=1 colspan=1>0.3120.330</td><td rowspan=1 colspan=2>0.3350.385</td><td rowspan=1 colspan=1>0.3530.351</td><td rowspan=1 colspan=1>0.3230.362</td><td rowspan=1 colspan=1>0.4030.388</td><td rowspan=1 colspan=1>0.4230.418</td></tr><tr><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1>|0.2230.253</td><td rowspan=1 colspan=1>|0.2260.268</td><td rowspan=1 colspan=2>0.2230.264 |</td><td rowspan=1 colspan=3>0.2410.286 |</td><td rowspan=1 colspan=1>0.2320.270</td><td rowspan=1 colspan=4>|0.2260.264</td><td rowspan=1 colspan=1>0.2230.261|</td><td rowspan=1 colspan=2>0.2330.292 |</td><td rowspan=1 colspan=1>0.2550.282</td><td rowspan=1 colspan=1>|0.2420.293</td><td rowspan=1 colspan=1>|0.2930.315 |</td><td rowspan=1 colspan=1>0.3120.355</td></tr><tr><td rowspan=4 colspan=1>Eleicety96192336720</td><td rowspan=2 colspan=1>0.1270.2180.1470.236</td><td rowspan=2 colspan=1>0.1350.2310.1490.243</td><td rowspan=2 colspan=2>0.1320.2270.1490.243</td><td rowspan=2 colspan=3>|0.1330.2320.1490.247</td><td rowspan=2 colspan=1>0.1340.2300.1540.250</td><td rowspan=2 colspan=4>|0.1530.2560.1680.269</td><td rowspan=1 colspan=1>0.1430.247</td><td rowspan=1 colspan=2>|0.1340.231</td><td rowspan=2 colspan=1>|0.1690.2710.1800.280</td><td rowspan=2 colspan=1>0.1400.2370.1540.251</td><td rowspan=2 colspan=1>|0.1710.2740.1800.283</td><td rowspan=2 colspan=1>|0.1910.3050.2030.316</td></tr><tr><td rowspan=1 colspan=1>0.1580.260</td><td rowspan=1 colspan=2>0.1460.243</td></tr><tr><td rowspan=2 colspan=1>0.1620.2540.1990.286</td><td rowspan=2 colspan=1>0.1650.2600.2060.297</td><td rowspan=1 colspan=2>0.1670.261</td><td rowspan=1 colspan=3>0.1610.259</td><td rowspan=1 colspan=1>0.1690.265</td><td rowspan=2 colspan=4>0.1890.2910.2280.320</td><td rowspan=2 colspan=1>0.1680.2670.2140.307</td><td rowspan=2 colspan=2>0.1650.2640.2370.314</td><td rowspan=1 colspan=1>5.0.264</td><td rowspan=2 colspan=1>0.2040.3040.2050.304</td><td rowspan=2 colspan=1>0.1690.2680.2040.301</td><td rowspan=2 colspan=1>0.2040.3050.2210.319</td><td rowspan=2 colspan=1>0.2210.3330.2590.364</td></tr><tr><td rowspan=1 colspan=2>0.2030.292</td><td rowspan=1 colspan=3>0.1970.297</td><td rowspan=1 colspan=1>0.1940.288</td></tr><tr><td rowspan=1 colspan=1>Avg </td><td rowspan=1 colspan=1>0.1590.248|</td><td rowspan=1 colspan=1>0.1640.258 |</td><td rowspan=1 colspan=2>0.1630.256</td><td rowspan=1 colspan=3>|0.1600.259 |</td><td rowspan=1 colspan=1>0.1630.258 |</td><td rowspan=1 colspan=4>0.1840.284|</td><td rowspan=1 colspan=1>0.1710.270 |</td><td rowspan=1 colspan=2>0.1710.263 |</td><td rowspan=1 colspan=1>0.1890.290 |</td><td rowspan=1 colspan=1>0.1670.264 |</td><td rowspan=1 colspan=1>0.1940.295</td><td rowspan=1 colspan=1>|0.2180.330</td></tr><tr><td rowspan=4 colspan=1>96Solar192336720</td><td rowspan=4 colspan=1>0.1690.2020.1890.2140.1950.2200.1990.225</td><td rowspan=4 colspan=1>0.1870.2550.1940.2650.2030.2640.2090.269</td><td rowspan=1 colspan=2>|0.1750.237</td><td rowspan=1 colspan=3>|0.1920.251</td><td rowspan=1 colspan=1>0.1900.244</td><td rowspan=1 colspan=4>|0.1790.232</td><td rowspan=1 colspan=1>0.1700.234</td><td rowspan=4 colspan=2>|0.1830.2080.2080.2260.2120.2390.2150.256</td><td rowspan=4 colspan=1>0.1980.2700.2060.2760.2080.2840.2320.294</td><td rowspan=4 colspan=1>|0.1990.2650.2200.2820.2340.2950.2430.301</td><td rowspan=4 colspan=1>|0.3810.3980.3950.3860.4100.3940.3770.376</td><td rowspan=4 colspan=1>|0.4850.5700.4150.4771.0080.8390.6550.627</td></tr><tr><td rowspan=1 colspan=2>0.1980.259</td><td rowspan=1 colspan=2>0.247</td><td rowspan=1 colspan=2>0.323</td><td rowspan=1 colspan=1>0.1930.257</td><td rowspan=1 colspan=3>0.201</td><td rowspan=1 colspan=2>259</td><td rowspan=1 colspan=1>0.2040.302</td></tr><tr><td rowspan=1 colspan=2>0.2130.259</td><td rowspan=1 colspan=3>0.2400.300</td><td rowspan=1 colspan=1>0.2030.266</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.256</td><td rowspan=2 colspan=1>0.2120.2930.2150.307</td></tr><tr><td rowspan=1 colspan=2>0.2220.269</td><td rowspan=1 colspan=3>0.2460.311</td><td rowspan=1 colspan=1>0.2230.281</td><td rowspan=1 colspan=4>0.2030.261</td></tr><tr><td rowspan=1 colspan=1>Avg </td><td rowspan=1 colspan=1>0.1880.215</td><td rowspan=1 colspan=1>|0.1980.263</td><td rowspan=1 colspan=2>|0.2020.256 |</td><td rowspan=1 colspan=3>0.2310.296 |</td><td rowspan=1 colspan=1>0.2020.262</td><td rowspan=1 colspan=4>0.1930.252</td><td rowspan=1 colspan=1>|0.2000.284 |</td><td rowspan=1 colspan=2>0.2040.232</td><td rowspan=1 colspan=1>|0.2110.281</td><td rowspan=1 colspan=1>|0.2240.286|</td><td rowspan=1 colspan=1>0.3910.388 |</td><td rowspan=1 colspan=1>0.6410.628</td></tr><tr><td rowspan=3 colspan=1>96Tiaatic192336720</td><td rowspan=3 colspan=1>0.3630.2410.3830.2510.3940.2570.4310.278</td><td rowspan=3 colspan=1>0.3880.2690.4110.2860.4250.2840.4550.302</td><td rowspan=2 colspan=2>|0.3910.2770.4050.283</td><td rowspan=2 colspan=3>|0.3780.2730.3910.277</td><td rowspan=1 colspan=1>0.3630.265</td><td rowspan=1 colspan=4>0.3690.257</td><td rowspan=1 colspan=1>0.3700.262</td><td rowspan=1 colspan=2>|0.5260.288</td><td rowspan=2 colspan=1>0.5950.3120.6130.322</td><td rowspan=2 colspan=1>|0.3950.2750.4070.280</td><td rowspan=2 colspan=1>|0.6040.3300.6100.338</td><td rowspan=3 colspan=1>|0.5930.3650.6140.3810.6270.3890.6460.394</td></tr><tr><td rowspan=1 colspan=1>0.3840.273</td><td rowspan=1 colspan=4>0.4000.272</td><td rowspan=1 colspan=1>0.3860.269</td><td rowspan=1 colspan=2>0.5030.263</td></tr><tr><td rowspan=1 colspan=2>0.4160.2900.4540.312</td><td rowspan=1 colspan=3>0.4020.2820.4340.297</td><td rowspan=1 colspan=1>0.3960.2770.4450.308</td><td rowspan=1 colspan=4>0.4070.2720.4610.316</td><td rowspan=1 colspan=1>0.3960.2750.4350.295</td><td rowspan=1 colspan=2>0.5050.2760.5520.301</td><td rowspan=1 colspan=1>0.6260.3320.6350.340</td><td rowspan=1 colspan=1>0.4170.2860.4540.308</td><td rowspan=1 colspan=1>0.6260.3410.6430.347</td></tr><tr><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1>0.3930.257</td><td rowspan=1 colspan=1>|0.4200.285</td><td rowspan=1 colspan=2>|0.4160.291 |</td><td rowspan=1 colspan=3>0.4010.282</td><td rowspan=1 colspan=1>|0.3970.281|</td><td rowspan=1 colspan=4>0.4090.279</td><td rowspan=1 colspan=1>0.3970.275|</td><td rowspan=1 colspan=2>0.5210.282|</td><td rowspan=1 colspan=1>0.6170.327</td><td rowspan=1 colspan=1>|0.4180.287|</td><td rowspan=1 colspan=1>0.6210.339|</td><td rowspan=1 colspan=1>0.6200.382</td></tr></table>

Euclidean counterparts, using norm-based depth and cosinebased angular alignment. This controlled setting isolates the effect of the representation geometry from differences in model capacity.

b) Hyperbolic consistently outperforms Euclidean.: As shown in Table III, the hyperbolic model consistently outperforms its Euclidean counterpart, with larger differences at longer prediction horizons. On Solar, the MSE improvement increases from 1.0% at H=192 to 9.5% at H=720 (0.220→0.199), while on ETTh1 it increases from 1.0% to 4.0% (0.451→0.433). Since the two variants differ only in the representation geometry, these results support the use of hyperbolic space for modeling the proposed cross-scale hierarchy, particularly for longerhorizon forecasting.

## F. Plug-and-Play Experiment

In this analysis, we integrate the core components of HypLTSF—the hyperbolic embedding layer and the temporal hierarchy losses $( \mathcal { L } _ { r } , \mathcal { L } _ { a } ) { \mathrm { - i n t o } }$ two representative yet architecturally distinct multi-scale forecasting models: TimeMixer [7], which constructs its multi-scale representation through progressive downsampling in the time domain, and MICN [9], which derives multi-scale features via learned convolution maps at different kernel sizes. The two models cover complementary design philosophies, namely temporal pooling and convolutional filtering, making them a suitable test bed for evaluating the generalizability of our approach across different multi-scale paradigms.

As shown in Table IV, applying the proposed module improves performance in most settings. On ETTm1 at H = 96, for example, MSE decreases from 0.293 to 0.288 for TimeMixer and from 0.303 to 0.289 for MICN. A few exceptions occur at longer horizons, such as TimeMixer at H = 336, where MSE slightly increases while MAE decreases.

TABLE II: Ablation on small-scale (ETTh1, ETTh2) and large-scale (Solar, Traffic) datasets at short-term (H=96) and long-term (H=720) horizons. All values are MSE.
<table><tr><td>Data</td><td>H</td><td>None</td><td> $\mathcal { L } _ { r }$  only |</td><td> ${ \mathcal { L } } _ { a }$  only</td><td>Full Model</td></tr><tr><td rowspan="2">ETTh1</td><td>96</td><td>0.374</td><td>0.364</td><td>0.370</td><td>0.361</td></tr><tr><td>720</td><td>0.443</td><td>0.442</td><td>0.443</td><td>0.433</td></tr><tr><td rowspan="2">ETTh2</td><td>96</td><td>0.279</td><td>0.275</td><td>0.276</td><td>0.275</td></tr><tr><td>720</td><td>0.410</td><td>0.399</td><td>0.408</td><td>0.399</td></tr><tr><td rowspan="2">Solar</td><td>96</td><td>0.176</td><td>0.170</td><td>0.171</td><td>0.169</td></tr><tr><td>720</td><td>0.219</td><td>0.200</td><td>0.201</td><td>0.199</td></tr><tr><td rowspan="2">Traffic</td><td>96</td><td>0.364</td><td>0.364</td><td>0.363</td><td>0.363</td></tr><tr><td>720</td><td>0.437</td><td>0.432</td><td>0.434</td><td>0.431</td></tr></table>

TABLE III: Comparison between hyperbolic and Euclidean representations. All values are MSE.
<table><tr><td rowspan="2">H</td><td colspan="2">Solar</td><td colspan="2">ETTh1</td><td colspan="2">Weather</td></tr><tr><td>| Hyperbolic Euclidean</td><td></td><td></td><td>|Hyperbolic Euclidean</td><td>|Hyperbolic Euclidean</td><td></td></tr><tr><td>192</td><td>0.189</td><td>0.191</td><td>0.401</td><td>0.405</td><td>0.190</td><td>0.196</td></tr><tr><td>336</td><td>0.195</td><td>0.199</td><td>0.423</td><td>0.424</td><td>0.241</td><td>0.243</td></tr><tr><td>720</td><td>0.199</td><td>0.220</td><td>0.433</td><td>0.451</td><td>0.315</td><td>0.318</td></tr></table>

TABLE IV: Full results of the plug-and-play long-term forecasting experiments with HypLTSF, evaluated at prediction horizons $H \in$ {96, 192, 336, 720}.
<table><tr><td colspan="2">Models</td><td colspan="2">TimeMixer</td><td colspan="2">+HypLTSF</td><td colspan="2">MICN</td><td colspan="2">+HypLTSF</td></tr><tr><td>Dataset</td><td>H</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="4">ETm1</td><td>96</td><td>0.293</td><td>0.345</td><td>0.288</td><td>0.332</td><td>0.303</td><td>0.349</td><td>0.289</td><td>0.338</td></tr><tr><td>192</td><td>0.335</td><td>0.372</td><td>0.332</td><td>0.358</td><td>0.336</td><td>0.369</td><td>0.333</td><td>0.363</td></tr><tr><td>336</td><td>0.368</td><td>0.386</td><td>0.370</td><td>0.379</td><td>0.370</td><td>0.391</td><td>0.369</td><td>0.390</td></tr><tr><td>720</td><td>0.426</td><td>0.417</td><td>0.424</td><td>0.411</td><td>0.410</td><td>0.421</td><td>0.412</td><td>0.411</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.147</td><td>0.198</td><td>0.148</td><td>0.189</td><td>0.172</td><td>0.232</td><td>0.164</td><td>0.210</td></tr><tr><td>192</td><td>0.191</td><td>0.242</td><td>0.190</td><td>0.231</td><td>0.214</td><td>0.271</td><td>0.212</td><td>0.253</td></tr><tr><td>336</td><td>0.244</td><td>0.280</td><td>0.242</td><td>0.280</td><td>0.259</td><td>0.309</td><td>0.256</td><td>0.286</td></tr><tr><td>720</td><td>0.316</td><td>0.331</td><td>0.315</td><td>0.330</td><td>0.309</td><td>0.343</td><td>0.310</td><td>0.345</td></tr></table>

Overall, the results demonstrate that the hyperbolic embedding and hierarchy losses can serve as an architecture-agnostic plug-in that transfers to different multi-scale strategies. Performance improvements are observed in most cases, although the magnitude varies across base models and prediction horizons.

## G. Model Efficiency Analysis

TABLE V: Efficiency and performance comparison on Traffic and Solar datasets. Best results are in bold.
<table><tr><td>Dataset Model</td><td></td><td>Memory (MB) ↓</td><td>Infer (ms) ↓</td><td>Train (ms) ↓</td><td>MSE↓</td><td>MAE↓</td></tr><tr><td>Tratic</td><td>TimeKAN</td><td>2870.6</td><td>414.42</td><td>632.85</td><td>0.411</td><td>0.286</td></tr><tr><td></td><td>TimeMixer</td><td>3070.2</td><td>134.78</td><td>337.34</td><td>0.400</td><td>0.271</td></tr><tr><td></td><td>HypLTSF</td><td>3167.8</td><td>142.64</td><td>387.79</td><td>0.383</td><td>0.251</td></tr><tr><td></td><td>TimeKAN</td><td>469.2</td><td>37.65</td><td>95.29</td><td>0.203</td><td>0.264</td></tr><tr><td>Solar</td><td>TimeMixer</td><td>507.8</td><td>22.93</td><td>59.44</td><td>0.214</td><td>0.272</td></tr><tr><td></td><td>HypLTSF</td><td>497.2</td><td>20.67</td><td>56.85</td><td>0.195</td><td>0.219</td></tr></table>

To evaluate the computational efficiency of HypLTSF, we compare memory consumption, training time, inference time, and forecasting performance against state-of-the-art multiscale forecasting models on two large-scale datasets: Traffic with 862 variables and Solar with 137 variables. Table V presents the efficiency comparison across the two datasets. For Traffic, we use a look-back window of 512 and a forecasting horizon of 192, while for Solar, we use the same look-back window with a forecasting horizon of 336.

Overall, HypLTSF provides a favorable trade-off between forecasting accuracy and computational efficiency. On Traffic, it achieves the best forecasting accuracy with only moderate increases in training and inference time compared with TimeMixer. On Solar, HypLTSF achieves the best forecasting accuracy while also providing the fastest training and inference. Although TimeKAN maintains the lowest memory consumption, its computational latency is higher on both datasets. These results show that explicit hierarchical modeling improves forecasting accuracy while maintaining competitive computational efficiency.

## VI. CONCLUSION

In this work, we proposed HypLTSF, a geometry-aware framework that explicitly models the multi-scale hierarchy of time series by embedding scale-wise representations into the Poincare ball. To shape the embedding geometry, we´ introduced two hierarchy-preserving losses: a radial ordering loss that enforces fine-scale embeddings to lie deeper than their coarse-scale counterparts, encoding parent–child relationships along the radial axis, and an angular coherence loss that encourages sibling time steps to cluster directionally, forming coherent branches in the embedding space. Together, these losses induce a tree-like structure in which depth encodes temporal resolution and angular proximity encodes local temporal context. We evaluate HypLTSF on eight real-world benchmark datasets. The results demonstrate state-of-the-art performance across diverse forecasting horizons and the learned embeddings exhibit the intended hierarchical structure, empirically validating the effectiveness of geometry-aware hierarchical modeling as an inductive bias.

## REFERENCES

[1] H. Huang, M. Chen, and X. Qiao, “Generative learning for financial time series with irregular and scale-invariant patterns,” in Proc. Int. Conf. Learn. Represent., 2024.

[2] O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, “Financial time series forecasting with deep learning: A systematic literature review: 2005–2019,” Appl. Soft Comput., vol. 90, p. 106181, 2020.

[3] J.-S. Chou and D.-S. Tran, “Forecasting energy consumption time series using machine learning techniques based on usage patterns of residential householders,” Energy, vol. 165, pp. 709–726, 2018.

[4] H. Wang, Z. Wang, Y. Niu, Z. Liu, H. Li, Y. Liao, Y. Huang, and X. Liu, “An accurate and interpretable framework for trustworthy process monitoring,” IEEE Trans. Artif. Intell., vol. 5, no. 5, pp. 2241–2252, 2023.

[5] M. Lippi, M. Bertini, and P. Frasconi, “Short-term traffic flow forecasting: An experimental comparison of time-series analysis and supervised learning,” IEEE Trans. Intell. Transp. Syst., vol. 14, no. 2, pp. 871–882, 2013.

[6] S. B. Yang, C. Guo, J. Hu, J. Tang, and B. Yang, “Unsupervised path representation learning with curriculum negative sampling,” in Proc. Int. Joint Conf. Artif. Intell., 2021.

[7] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, and J. ZHOU, “Timemixer: Decomposable multiscale mixing for time series forecasting,” in Proc. Int. Conf. Learn. Represent., 2024.

[8] S. Wang, J. Li, X. Shi, Z. Ye, B. Mo, W. Lin, J. Shengtong, Z. Chu, and M. Jin, “Timemixer++: A general time series pattern machine for universal predictive analysis,” in Proc. Int. Conf. Learn. Represent., 2025.

[9] H. Wang, J. Peng, F. Huang, J. Wang, J. Chen, and Y. Xiao, “Micn: Multi-scale local and global context modeling for long-term series forecasting,” 2023.

[10] S. Liu, H. Yu, C. Liao, J. Li, W. Lin, A. X. Liu, and S. Dustdar, “Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting,” in Proc. Int. Conf. Learn. Represent., 2022.

[11] S. Huang, Z. Zhao, C. Li, and L. BAI, “Timekan: Kan-based frequency decomposition learning architecture for long-term time series forecasting,” in Proc. Int. Conf. Learn. Represent., 2025.

[12] Y. Hu, P. Liu, P. Zhu, D. Cheng, and T. Dai, “Adaptive multi-scale decomposition framework for time series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2025.

[13] F. Li, S. Guo, F. Han, J. Zhao, and F. Shen, “Multi-scale dilated convolution network for long-term time series forecasting,” arXiv preprint arXiv:2405.05499, 2024.

[14] R. Sarkar, “Low distortion delaunay embedding of trees in hyperbolic plane,” in Proc. Int. Symp. Graph Drawing. Springer, 2011, pp. 355– 366.

[15] M. Nickel and D. Kiela, “Poincare embeddings for learning hierarchical´ representations,” Adv. Neural Inf. Process. Syst., vol. 30, 2017.

[16] I. Chami, A. Wolf, D.-C. Juan, F. Sala, S. Ravi, and C. Re, “Low-´ dimensional hyperbolic knowledge graph embeddings,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2020, pp. 6901–6914.

[17] B. Dhingra, C. Shallue, M. Norouzi, A. Dai, and G. Dahl, “Embedding text in hyperbolic spaces,” in Proc. 12th Workshop Graph-Based Methods Nat. Lang. Process. (TextGraphs-12), 2018, pp. 59–69.

[18] Q. Liu, M. Nickel, and D. Kiela, “Hyperbolic graph neural networks,” Adv. Neural Inf. Process. Syst., vol. 32, 2019.

[19] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “Deepar: Probabilistic forecasting with autoregressive recurrent networks,” Int. J. Forecast., vol. 36, no. 3, pp. 1181–1191, 2020.

[20] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long-and shortterm temporal patterns with deep neural networks,” in Proc. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2018, pp. 95–104.

[21] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” in Proc. Int. Conf. Learn. Represent., 2023.

[22] M. Liu, A. Zeng, M. Chen, Z. Xu, Q. Lai, L. Ma, and Q. Xu, “Scinet: Time series modeling and forecasting with sample convolution and interaction,” in Adv. Neural Inf. Process. Syst., 2022.

[23] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2021.

[24] L. Li, J. Su, X. Zhen, S. Qin, Z. Jin, W. Li, and F. Gao, “Q-Informer: An informer-based framework with quantum-enhanced sparse attention for long-term time series forecasting,” Inf. Fusion, vol. 138, p. 104688, 2027.

[25] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” in Adv. Neural Inf. Process. Syst., 2021.

[26] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in Proc. Int. Conf. Mach. Learn., 2022.

[27] S. Zhu, J. Zheng, and Q. Ma, “Mr-transformer: Multiresolution transformer for multivariate time series prediction,” IEEE Trans. Neural Netw. Learn. Syst., vol. 36, no. 1, pp. 1171–1183, 2023.

[28] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Proc. Int. Conf. Learn. Represent., 2023.

[29] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” in Proc. Int. Conf. Learn. Represent., 2024.

[30] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proc. AAAI Conf. Artif. Intell., 2023.

[31] A. Das, W. Kong, A. Leach, S. K. Mathur, R. Sen, and R. Yu, “Longterm forecasting with tide: Time-series dense encoder,” Trans. Mach. Learn. Res., 2024.

[32] V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, and J. Kalagnanam, “Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2023.

[33] I. O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, et al., “Mlpmixer: An all-mlp architecture for vision,” in Adv. Neural Inf. Process. Syst., 2021.

[34] K. Yi, Q. Zhang, W. Fan, S. Wang, P. Wang, H. He, N. An, D. Lian, L. Cao, and Z. Niu, “Frequency-domain MLPs are more effective learners in time series forecasting,” in Adv. Neural Inf. Process. Syst., 2023.

[35] S. Lin, W. Lin, W. Wu, H. Chen, and J. Yang, “Sparsetsf: Modeling long-term time series forecasting with\* 1k\* parameters,” in Proc. Int. Conf. Mach. Learn., 2024.

[36] H. He, Q. Zhang, K. Yi, X. Xue, S. Wang, L. Hu, and L. Cao, “Robust multivariate time series forecasting against intraseries and interseries transitional shift,” IEEE Trans. Neural Netw. Learn. Syst., 2025.

[37] N. Kim, H. Baik, and Y. Yoon, “Topocl: Topological contrastive learning for time series,” IEEE Trans. Neural Netw. Learn. Syst., 2026.

[38] M. Nickel and D. Kiela, “Learning continuous hierarchies in the lorentz model of hyperbolic geometry,” in Proc. Int. Conf. Mach. Learn. PMLR, 2018, pp. 3779–3788.

[39] O. Ganea, G. Becigneul, and T. Hofmann, “Hyperbolic entailment cones´ for learning hierarchical embeddings,” in Proc. Int. Conf. Mach. Learn. PMLR, 2018, pp. 1646–1655.

[40] C. Zhang and J. Gao, “Hype-han: Hyperbolic hierarchical attention network for semantic embedding,” in Proc. Int. Joint Conf. Artif. Intell., 2021, pp. 3990–3996.

[41] C.-Y. Chen, T. M. Hung, Y.-L. Hsu, and L.-W. Ku, “Label-aware hyperbolic embeddings for fine-grained emotion classification,” in Proc. Annu. Meeting Assoc. Comput. Linguistics, 2023, pp. 10 947–10 958.

[42] I. Chami, Z. Ying, C. Re, and J. Leskovec, “Hyperbolic graph convolu-´ tional neural networks,” in Adv. Neural Inf. Process. Syst., 2019.

[43] J. Dai, Y. Wu, Z. Gao, and Y. Jia, “A hyperbolic-to-hyperbolic graph convolutional network,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021.

[44] M. Yang, M. Zhou, M. Kalander, Z. Huang, and I. King, “Discretetime temporal network embedding via implicit hierarchical learning in hyperbolic space,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2021.

[45] C. Challu, K. G. Olivares, B. N. Oreshkin, F. G. Ramirez, M. M. Canseco, and A. Dubrawski, “Nhits: Neural hierarchical interpolation for time series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2023.

[46] Z. Shang, L. Chen, B. Wu, and D. Cui, “Ada-mshyper: Adaptive multiscale hypergraph transformer for time series forecasting,” in Adv. Neural Inf. Process. Syst., 2024.

[47] W. Bao, Y. Cao, Y. Yang, and S. Wen, “Long short-term financial time series forecasting based on residual multiscale tcn sparse expert network and informer,” IEEE Trans. Neural Netw. Learn. Syst., vol. 36, no. 10, pp. 19 200–19 209, 2025.

[48] T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, “Reversible instance normalization for accurate time-series forecasting against distribution shift,” in Proc. Int. Conf. Learn. Represent., 2021.

[49] R. B. Cleveland, W. S. Cleveland, J. E. McRae, I. Terpenning, et al., “Stl: A seasonal-trend decomposition,” J. off. Stat, vol. 6, no. 1, pp. 3–73, 1990.

[50] J. Fei, K. Yi, W. Fan, Q. Zhang, and Z. Niu, “Amplifier: Bringing attention to neglected low-energy components in time series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2025.

[51] Y. Zhang and J. Yan, “Crossformer: Transformer utilizing crossdimension dependency for multivariate time series forecasting,” in Proc. Int. Conf. Learn. Represent., 2023.

[52] Y. Liu, H. Wu, J. Wang, and M. Long, “Non-stationary transformers: Exploring the stationarity in time series forecasting,” in Adv. Neural Inf. Process. Syst., 2022.

[53] S. Han, S. Lee, M. Cha, S. O. Arik, and J. Yoon, “Retrieval augmented time series forecasting,” in Proc. Int. Conf. Mach. Learn., 2025.

[54] X. Qiu, J. Hu, L. Zhou, X. Wu, J. Du, B. Zhang, C. Guo, A. Zhou, C. S. Jensen, Z. Sheng, and B. Yang, “Tfb: Towards comprehensive and fair benchmarking of time series forecasting methods,” Proc. VLDB Endow., 2024.

[55] X. Qiu, X. Wu, Y. Lin, C. Guo, J. Hu, and B. Yang, “Duet: Dual clustering enhanced multivariate time series forecasting,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2025.

[56] X. Wu, X. Qiu, H. Cheng, Z. Li, J. Hu, C. Guo, and B. Yang, “Enhancing time series forecasting through selective representation spaces: A patch perspective,” in Adv. Neural Inf. Process. Syst., 2025.