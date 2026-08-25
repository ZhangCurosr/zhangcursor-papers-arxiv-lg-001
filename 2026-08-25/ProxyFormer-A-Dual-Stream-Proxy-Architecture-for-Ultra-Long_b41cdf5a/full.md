# ProxyFormer: A Dual-Stream Proxy Architecture for Ultra-Long Context and High-Resolution Generation

Zhongpan Tang

Independent Researcher

tangzhongp@qq.com

## Abstract

The quadratic growth of attention computation and key-value (KV) cache with respect to sequence length is a central bottleneck for ultra-long-context language models and highresolution generative models. We propose ProxyFormer, a general dual-stream architecture built upon proxy tokens. In each layer, fine-grained local features are compressed bottomup into a small set of proxy states; expensive global interactions are performed only in the compressed proxy space; the globally contextualized proxies are then decompressed and injected top-down back into the local stream. Because the local stream persists across layers, fine-grained information that is not captured by one compression step remains accessible for later refinement, alleviating the irreversible information loss of conventional one-shot compression. We further introduce factorized multi-level compression/decompression, layerwise dynamic compression ratios, asymmetric dual embeddings, and a proxy-only KV-cache inference scheme. On a 16 GB GPU with batch size 1, a standard decoder-only model can train sequences of only about 20K tokens, whereas ProxyFormer with a compression ratio of 64 extends the trainable sequence length to about 0.7M. A model trained with a 64K window retains 92%–95% retrieval accuracy on a multi-needle retrieval task with 1,048,576 tokens, and a model trained with an 8K window exceeds 94% accuracy when extrapolated to 256K tokens. Preliminary image-generation experiments demonstrate the feasibility of ProxyFormer for both pixel-space and latent-space flow matching.

Keywords: long-sequence modeling; eficient attention; proxy token; KV-cache compression; length extrapolation; flow matching

## 1 Introduction

The attention mechanism [1] has established the Transformer as the foundational architecture for language modeling, long-context reasoning, and visual generation. For a sequence of length L and feature dimension d, the computational complexity of standard self-attention is $O ( L ^ { 2 } d )$ in autoregressive decoding, the key and value of each historical token must also be persistently cached, resulting in a single-layer KV cache size of O(Ld). As context lengths reach the hundreds of thousands or millions, attention computation and KV caching often far exceed the model weights themselves, becoming the primary factors limiting inference throughput and training sequence length [2, 3].

To address this problem, existing work has largely followed three routes: first, designing linear, sparse, or fixed-pattern attention approximations [4, 5, 6]; second, replacing attention with state space models or recurrent forms [7, 8]; third, pruning, quantizing, or evicting the KV cache [9, 10, 11]. While efective in their respective scenarios, these methods generally face a structural contradiction: if fine-grained information is discarded during compression or approximation, the capability for precise long-range retrieval degrades; if all fine-grained information is retained, the computational and memory advantages are limited. Single-shot, irreversible sequence compression is particularly prone to the ”lost in the middle” phenomenon and degradation in Needle-In-A-Haystack (NIAH) performance.

This paper introduces ProxyFormer, driven by the motivation to decouple information fidelity and global interaction cost. ProxyFormer does not require all tokens to participate directly in global attention, but instead organizes the input into two parallel feature streams:

1. Fine/local stream: maintains fine-grained features that correspond one-to-one with input elements, responsible for preserving local details;

2. Proxy stream: a small number of proxy vectors obtained by compressing local blocks, responsible for completing global interactions at an extremely low cost.

Each layer executes a closed loop of ”bottom-up compression  proxy space interaction  topdown decompression and injection”. Because the local stream maintains residual connections across layers, details not absorbed by a compression step in one layer can still be passed to deeper layers and re-utilized in subsequent compressions; meanwhile, the global information carried by the proxy stream is reversely injected into the local stream through decompression and mechanisms such as gating, modulation, or cross-attention. This mechanism allows the model, at a compression ratio of P, to reduce the sequence length for proxy space attention to $L / P$ while preserving the information integrity of the local stream.

The main contributions of this paper are as follows:

• We propose the general ProxyFormer architecture, uniformly supporting 1D sequences, 2D images, 3D point clouds/voxels/videos, and higher-dimensional tensors, covering tasks such as autoregressive language modeling, flow matching image generation, and difusion conditional generation;

• We provide an implementation-friendly architectural instance: compression and decompression are composed of standard operators such as reshape, linear projection, and convolution; proxy interactions can directly reuse existing attention or SSM modules without the need for sparse indexers, block-sparse attention kernels, or custom CUDA operators;

• We introduce a bidirectional communication mechanism between the local and proxy streams, as well as factorized multi-level compression/decompression and layer-wise dynamic compression ratios, which control the parameter size of the compression modules while maintaining high compression ratios;

• We propose a proxy-level KV cache and a streaming block-triggered update strategy, allowing autoregressive inference to only maintain historical proxy states of length $L / P$ , while original historical local features can be released or ofloaded;

• We systematically validate the architecture through preliminary experiments on language model perplexity, multi-needle NIAH, memory/speed benchmarks, and image generation tasks, providing quantitative comparisons with standard decoder-only baselines.

## 2 Related Work

## 2.1 Eficient Attention and Long-Context Modeling

Linear attention and kernel methods approximate attention in a low-rank or feature-mapped form, reducing complexity to O(L) [12]. Linformer demonstrates that the self-attention matrix can be low-rank approximated and uses a fixed number of learnable projections to reduce complexity to O(L) [4]. Longformer, BigBird, and others implement sparse attention by combining sliding windows, dilated windows, and global tokens [5, 6]. The FlashAttention series accelerates exact attention from an I/O perspective but does not alter its quadratic complexity [2, 3]. State space models formulate sequence modeling as a linear time-varying system, with Mamba and its extensions demonstrating competitive quality and eficiency on long sequences [7, 8]. Unlike the aforementioned methods, ProxyFormer does not require changing the mathematical form of the global interaction operator; its interaction operator can be attention, SSM, RNN, or an identity mapping. The compression benefits primarily stem from the structural property that interactions occur in the extremely short proxy space.

## 2.2 KV Cache Compression and Token Selection

To reduce KV cache usage in autoregressive inference, StreamingLLM retains ”attention sinks” and recent tokens [10]; $_ \mathrm { H _ { 2 } O }$ evicts low-scoring tokens based on attention scores [9]; Infiniattention compresses long-term memory into a fixed number of reusable states [11]. Landmark Attention and similar memory token methods compress historical blocks into summary tokens and cache them [13]. A commonality among these methods is that compression or eviction typically occurs after attention, and original tokens are unrecoverable once discarded. ProxyFormer, on the other hand, places compression before global interaction and treats historical proxy states as a trainable cross-layer stream; more importantly, the local stream remains present during training, enabling the compressor to receive fine-grained gradient signals from deeper layers.

## 2.3 Eficient Representations in Visual Generation

Difusion models [14] and flow matching [15] have become mainstream paradigms for high-quality image generation. To reduce the computational cost of high-resolution generation, common practices include generating in a VAE latent space first [16, 17] or applying local window attention to feature maps. ProxyFormer provides a unified patch-to-proxy compression pathway: for pixel-space (JIT) or latent-space (JLT) feature maps, proxy vectors are extracted based on 2D patches, and cross-modal conditions only perform cross-attention with the proxy vectors. This significantly reduces the cost of condition interaction and global attention while maintaining the local pixel/latent feature stream.

## 2.4 Position of this Work

Compared to fixed-length compressed attention works such as TLinformer and TConstFormer [18, 19], ProxyFormer’s compressed length can dynamically vary with the input length and network layers, and it features symmetric decompression and top-down injection pathways. Compared to the study of near-lossless compression on fixed-shape features in ”Compression is Routing” [20], this paper systemizes that concept into a trainable, stackable dual-stream architecture applicable to autoregressive decoding and visual generation.

## 3 ProxyFormer Methodology

## 3.1 Problem Definition and Overall Architecture

![](images/39888f89c0262e23ab0cd171fa59c752f4be848cab677c42e19f1a544dcbb8a0.jpg)  
Figure 1: Local stream and proxy stream dual-stream architecture. Fine-grained local features update proxy states through ”micro guiding macro”; after proxy states complete global interaction, they are injected into the local stream through ”macro guiding micro”.

Formally, let the input features be a general tensor $\mathcal { X } ^ { ( 0 ) } \in \mathbb { R } ^ { N _ { 1 } \times \cdots \times N _ { D } \times d _ { \mathrm { i n } } }$ , where D is the number of spatial/temporal/topological dimensions, $N _ { 1 } , \ldots , N _ { D }$ are the sizes of each dimension, and $d _ { \mathrm { i n } }$ is the number of channels. For convenience, we flatten it into a feature sequence $X ^ { ( 0 ) } \in \mathbb { R } ^ { L \times d _ { \mathrm { i n } } }$ of length $\begin{array} { r } { L = \prod _ { j = 1 } ^ { D } N _ { j } } \end{array}$ ; for text, $D = 1$ ; for images, $D = 2 ;$ ; for 3D point clouds/voxels or spatiotemporal video, $D = 3$ can be used; higher-dimensional data can be recursively grouped or flattened. A standard Transformer layer can be written as

$$
A ^ { ( l ) } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d _ { \mathrm { h e a d } } } } \right) V , \qquad X ^ { ( l ) } = X ^ { ( l - 1 ) } + \mathrm { F F N } ^ { ( l ) } ( A ^ { ( l ) } ) ,\tag{1}
$$

where $Q , K , V$ are obtained by projecting $X ^ { ( l - 1 ) }$ . The computation and activation cache of Equation (1) are both $O ( L ^ { 2 } )$ . ProxyFormer rewrites each layer into the following five steps:

$$
\{ X _ { 1 } ^ { ( l - 1 ) } , \ldots , X _ { m } ^ { ( l - 1 ) } \} = \mathrm { P a r t i t i o n } \Big ( X ^ { ( l - 1 ) } ; P ^ { ( l ) } \Big ) ,\tag{2}
$$

$$
\begin{array} { r } { c _ { i } ^ { ( l ) } = \mathcal { C } _ { i } ^ { ( l ) } \left( \boldsymbol { X } _ { i } ^ { ( l - 1 ) } \right) , \qquad \boldsymbol { H } _ { i } ^ { ( l ) } = \mathcal { U } _ { i } ^ { ( l ) } \left( \boldsymbol { H } _ { i } ^ { ( l - 1 ) } , \boldsymbol { c } _ { i } ^ { ( l ) } \right) , } \end{array}\tag{3}
$$

$$
\bar { H } ^ { ( l ) } = \mathcal { F } ^ { ( l ) } \Big ( H ^ { ( l ) } \Big ) ,\tag{4}
$$

$$
\Delta _ { i } ^ { ( l ) } = { \cal D } _ { i } ^ { ( l ) } \Big ( \bar { H } _ { i } ^ { ( l ) } \Big ) ,\tag{5}
$$

$$
\begin{array} { r } { X ^ { ( l ) } = \Phi ^ { ( l ) } \left( X ^ { ( l - 1 ) } , \Delta _ { 1 } ^ { ( l ) } , \dots , \Delta _ { m } ^ { ( l ) } \right) , } \end{array}\tag{6}
$$

where $P ^ { ( l ) }$ is the block size for the l-th layer, $m = \lceil L / P ^ { ( l ) } \rceil ; X _ { i } \in \mathbb { R } ^ { P ^ { ( l ) } \times d _ { \mathrm { f i n e } } }$ is the i-th local block; $H _ { i } \in \mathbb { R } ^ { d _ { \mathrm { p r o x y } } }$ is the corresponding proxy vector; ,  are the compression and decompression modules respectively;  is the proxy state update; $\mathcal { F }$ is the proxy space interaction; and Φ is the local feature injection and fusion. Algorithm 1 summarizes the forward process of a ProxyFormer layer, and Figure 1 depicts the cross-layer dual-stream information pathway.

Algorithm 1 Forward Process of a Single ProxyFormer Layer   
Require: Local stream $X ^ { ( l - 1 ) } \in \mathbb { R } ^ { L \times d _ { \mathrm { f i n e } } }$ , proxy stream $H ^ { ( l - 1 ) } \in \mathbb { R } ^ { m \times d _ { \mathrm { p r o x y } } }$ , block size $P ^ { ( l ) }$   
Ensure: Updated local stream $X ^ { ( l ) }$ and proxy stream $H ^ { ( l ) }$   
1: $\{ X _ { 1 } , \dots , X _ { m } \} \gets \mathrm { P a r t i t i o n } ( X ^ { ( l - 1 ) } ; P ^ { ( l ) } )$   
2: for $i = 1 , \ldots , m$ do   
3: $c _ { i }  { c _ { i } ( X _ { i } ) }$   
4: $H _ { i } \gets \mathcal { U } _ { i } ( H _ { i } , c _ { i } )$   
5: end for   
6: $\bar { H }  \mathcal { F } ( H )$   
7: for $i = 1 , \ldots , m$ do   
8: $\Delta _ { i }  \mathcal { D } _ { i } ( \bar { H } _ { i } )$   
9: end for   
10: $X ^ { ( l ) } \gets \Phi ( X ^ { ( l - 1 ) } , \Delta _ { 1 } , . . . , \Delta _ { m } )$   
11: return $\dot { X ^ { ( l ) } } , H ^ { ( l ) }$

## 3.2 Bottom-Up Compression

Partitioning. For 1D sequences, we adopt contiguous, non-overlapping block partitioning, though sliding overlapping windows are also supported; for 2D images, partitioning degenerates into spatial patches; for 3D voxels, point clouds, or spatiotemporal videos, one can use 3D voxel blocks, spatial neighborhood grouping, voxelization followed by partitioning, or dynamic clustering based on semantics/distance; for higher-dimensional tensors, they can be recursively flattened and partitioned. Let the i-th flattened block be $X _ { i } \in \mathbb { R } ^ { P \times d _ { \mathrm { f i n e } } }$ . In our experiments, we use fixed boundary partitioning; more general dynamic routing/clustering grouping is a generalization of Equation (2).

Proxy Extraction. The compression module $\mathcal { C } _ { i }$ maps $X _ { i }$ to a low-dimensional vector $c _ { i } .$ . The instance used in our experiments is ”reshape–linear projection”:

$$
c _ { i } = { \mathrm { P r o j } } ( { \mathrm { F l a t t e n } } ( X _ { i } ) ) \in \mathbb { R } ^ { d _ { \mathrm { p r o x y } } } ,\tag{7}
$$

For 2D patches, spatial flattening and 2D convolution/linear projection can be used; for 3D voxel/video blocks, 3D convolution or corresponding high-dimensional unfolding can be used. These operators are standard deep learning primitives and do not require custom sparse indexers. The proxy state update employs a residual form:

$$
H _ { i } ^ { ( l ) } = H _ { i } ^ { ( l - 1 ) } + c _ { i } ^ { ( l ) } ,\tag{8}
$$

which can also be replaced by gated updates, concatenated projection, or direct substitution. When a layer has no cross-layer proxy state input, $H _ { i } ^ { ( l - 1 ) } = \mathrm { 0 }$ , and Equation (8) degenerates to direct compression.

Asymmetric Dual Embeddings. In autoregressive language models, the historical region and the generation region bear diferent cost structures. ProxyFormer uses a low-dimensional embedding $E _ { h } : \mathcal { V }  \mathbb { R } ^ { d _ { h } }$ for the historical region and a full-dimensional embedding $E _ { x } : \mathcal { V } $ R<sup>d</sup>model for the generation region, with their parameters independent and dimensions decoupled. In our experiments, $d _ { h } = 6 4$ and $d _ { \mathrm { m o d e l } } = 5 1 2$ , meaning the historical sequence is compressed by a factor of 8 in channel memory at the very first step entering the network. This design is compatible with the dimensional decoupling of the local and proxy streams, and can also be replaced by shared embeddings with projection.

## 3.3 Global Interaction in Proxy Space

The proxy interaction function $\mathcal { F } ^ { ( l ) }$ is a pluggable operator in the architecture. Let $\bar { H } \in$ R<sup>m×dproxy</sup> , its optional forms include:

• Identity/Pass-through: $\bar { H } = H$ , suitable for completely independent block evolution;

• Single-Proxy MLP: applying a shared feed-forward network independently to each proxy vector;

• Causal/Bidirectional Attention:

$$
\bar { H } = \mathrm { s o f t m a x } \Bigg ( \frac { Q _ { p } K _ { p } ^ { \top } } { \sqrt { d _ { \mathrm { h e a d } } } } + M \Bigg ) V _ { p } ,\tag{9}
$$

where M is a causal mask (for autoregressive tasks) or a zero matrix (for bidirectional tasks);

• State Space or Recurrent Operators: feeding the proxy sequence into an SSM/RNN;

• Cross-Modal Cross-Interaction: using proxy vectors as queries to perform crossattention with text, time steps, or control conditions.

Since the sequence length in Equation (9) is $m \approx L / P$ , the cost of proxy space attention is approximately $P ^ { 2 }$ times lower than full attention. This property is independent of any specific low-rank or kernel approximation assumptions.

## 3.4 Top-Down Decompression and Injection

For each block, the interacted proxy vector is decompressed back to the local space:

$$
\Delta _ { i } = \mathrm { U n f l a t t e n } \Big ( \mathrm { P r o j } ^ { \uparrow } \big ( \bar { H } _ { i } \big ) \Big ) \in \mathbb { R } ^ { P \times d _ { \mathrm { f i n e } } } .\tag{10}
$$

The local fusion Φ can choose from the following mechanisms:

$$
\mathrm { R e s i d u a l \ A d d i t i o n } \colon X _ { i } \gets X _ { i } + \Delta _ { i } ,\tag{11}
$$

$$
\mathrm { G a t e d ~ F u s i o n : } \quad X _ { i } \gets X _ { i } + \sigma ( g ( X _ { i } , \Delta _ { i } ) ) \odot \Delta _ { i } ,\tag{12}
$$

$$
\mathrm { A f f n e ~ M o d u l a t i o n } \colon \ x _ { i } \gets ( 1 + s ( \Delta _ { i } ) ) \odot \mathrm { N o r m } ( X _ { i } ) + t ( \Delta _ { i } ) ,\tag{13}
$$

$$
\mathrm { C r o s s - A t t e n t i o n } \colonX _ { i } \gets X _ { i } + \mathrm { A t t n } \big ( Q = X _ { i } , K = \bar { H } _ { i } , V = \bar { H } _ { i } \big ) .\tag{14}
$$

Subsequently, the local stream can continue to be refined by a local Transformer or convolution. Our language model implementation uses gated cross-attention (a gated variant of Equation (14)) to inject proxy history into the generation region; details are in Section 3.7.

The key design is that compression is not the destruction of the local stream, but the summarization of it. Even if the compressor at layer l misses a detail, that detail is preserved in the residual stream of $X ^ { ( l ) }$ and can participate in compression again at layer $l + 1$ . Therefore, the compression bias of ProxyFormer can be corrected through cross-layer iteration, which is fundamentally diferent from methods that discard original tokens after a single compression.

![](images/f07e9418ce56fa40602a958ad247541e797e9625105a5d1086b58abd3eb21dd5.jpg)  
Figure 2: Multi-level cascaded factorized compression and symmetric decompression. $P =$ $\textstyle \prod _ { j = 1 } ^ { \bar { M } } k _ { j }$ , and intermediate features exist only on shorter sequences with narrower channels.

When the compression ratio P is large, the parameters and computation of a single linear projection are both $O ( P d _ { \mathrm { f i n e } } d _ { \mathrm { p r o x y } } )$ , which is prone to overfitting or becoming untrainable. Proxy-Former factorizes P into M integer factors:

$$
P = \prod _ { j = 1 } ^ { M } k _ { j } , \qquad k _ { j } \geq 1 ,\tag{15}
$$

and constructs a cascaded compression:

$$
\begin{array} { r l r l } & { \mathcal { C } = \mathcal { C } _ { M } \circ \mathcal { C } _ { M - 1 } \circ \cdots \circ \mathcal { C } _ { 1 } , } & & { \mathcal { C } _ { j } : \mathbb { R } ^ { K _ { j - 1 } \times d _ { j - 1 } } \to \mathbb { R } ^ { K _ { j } \times d _ { j } } , } \end{array}\tag{16}
$$

where $K _ { j } = K _ { j - 1 } / k _ { j } , K _ { 0 } = P$ , and $K _ { M } = 1$ . A typical implementation is ”reshape  project”: first group $K _ { j - 1 }$ elements into groups of $k _ { j }$ and concatenate along the channel dimension to obtain $\mathbb { R } ^ { K _ { j } \times ( \check { k _ { j } } d _ { j - 1 } ) }$ , then linearly project to $d _ { j }$ . The decompression side executes reverse reshape and up-projection in the opposite order:

$$
\mathcal { D } = \mathcal { D } _ { 1 } \circ \mathcal { D } _ { 2 } \circ \cdot \cdot \cdot \circ \mathcal { D } _ { M } , \qquad \mathcal { D } _ { j } : \mathbb { R } ^ { K _ { j } \times d _ { j } } \to \mathbb { R } ^ { K _ { j - 1 } \times d _ { j - 1 } } .\tag{17}
$$

Figure 2 illustrates a two-level factorization. If the parameter size of each sub-projection is approximately $k _ { j } d _ { j - 1 } d _ { j }$ , the total parameter size on the compression side is $\begin{array} { r l } { \sum _ { j } O ( k _ { j } d _ { j - 1 } d _ { j } ) } \end{array}$ whereas a single mapping is $O ( P d _ { \mathrm { f i n e } } d _ { \mathrm { p r o x y } } )$ . For example, when $P = 6 4 , M = \ddot { 3 } .$ , and $k _ { j } = 4$ the reshape-projection pathway significantly reduces the dimensionality of single-layer matrices.

Conv/ConvTranspose downsampling and upsampling in visual generation can be viewed as 2D special cases of Equations (16)–(17); 3D convolutions and their transposes can similarly be viewed as corresponding implementations for 3D data.

## 3.6 Layer-wise Dynamic Compression Ratios and Memory Hierarchy

ProxyFormer allows diferent layers to use diferent $P ^ { ( l ) }$ . When the compression ratio is inconsistent between adjacent layers, the proxy state H is re-initialized and re-extracted from the local stream of the current layer; this is equivalent to ”re-partitioning” across diferent scales of abstraction. Furthermore, diferent compression ratios can be configured for historical blocks at diferent temporal distances: distant history uses coarser-grained proxies, while recent history retains more fine-grained tokens. This horizontal/vertical dynamic configuration requires altering attention masks and cache layouts in standard Transformers, whereas in ProxyFormer it only involves hyperparameters for partitioning and compression modules.

## 3.7 Language Model Instantiation

![](images/52648b79947ac77ff4d69f16c87891de90e41729d871657a6011c6164f9d3c5a.jpg)  
Figure 3: Language model based on proxy features. Long history is compressed into an extremely small proxy KV cache; during decoding, the generation region only performs gated cross-attention with the proxy cache.

Figure 3 illustrates the autoregressive language model based on ProxyFormer. Given a history token sequence $t _ { h }$ and a generation region token sequence $t _ { x } ,$ the model first computes

$$
H _ { \mathrm { f i n e } } ^ { ( 0 ) } = E _ { h } ( t _ { h } ) \in \mathbb { R } ^ { L _ { h } \times d _ { h } } , \qquad X ^ { ( 0 ) } = E _ { x } ( t _ { x } ) \in \mathbb { R } ^ { L _ { x } \times d _ { \mathrm { m o d e l } } } .\tag{18}
$$

On the historical side, $L _ { \mathrm { d e p t h } }$ ProxyFormer blocks are stacked to compress $H _ { \mathrm { { f i n e } } }$ into proxy state H; under causal settings, the proxy vectors in H depend only on the current block and past blocks, and thus can be safely cached. The generation side uses standard Transformer blocks and adds cross-attention to the historical proxies alongside self-attention:

$$
A _ { x } = \mathrm { S D P A } ( q _ { x } , k _ { x } , v _ { x } ; \mathrm { c a u s a l } ) , \quad A _ { h } = \mathrm { S D P A } ( q _ { x } , k _ { h } , v _ { h } ) ,\tag{19}
$$

$$
\alpha = \mathrm { M L P } _ { \alpha } ( \mathrm { R M S N o r m } ( x ) ) , \qquad A = A _ { x } + \alpha \odot A _ { h } ,\tag{20}
$$

where α is computed per attention head, allowing the model to adaptively decide how much information to read from the historical proxies. The output layer is a shared-weight RMSNorm + linear language modeling head; the training loss is the standard next-token cross-entropy, appended with a loss term from intermediate layer auxiliary heads.

## 3.8 Proxy KV Cache and Streaming Decoding

In a standard autoregressive Transformer, the historical region caches $K _ { h } , V _ { h } \in \mathbb { R } ^ { L _ { h } \times d _ { \mathrm { m o d e l } } }$ at each layer. ProxyFormer executes compression and proxy interaction on historical blocks during the prefill stage, and thereafter only caches

$$
K _ { \mathrm { p r o x y } } , V _ { \mathrm { p r o x y } } \in \mathbb { R } ^ { ( L _ { h } / P ) \times d _ { \mathrm { m o d e l } } } ,\tag{21}
$$

The original historical local features can be released, quantized, or ofloaded. During the decoding stage, each step only computes self-attention for the current token and its cross-attention with the proxy cache; the historical proxy cache remains frozen. When the new tokens in the generation region reach the block size $P ,$ a local compression is triggered, appending the new block to the proxy cache and releasing the original local features of that block. Equation (21) shows that in the extreme case, the historical KV cache can be reduced to $1 / P$ of the standard implementation; moreover, since proxy attention during the prefill stage is only executed over $L _ { h } / { P }$ proxies, prefill latency is also substantially decreased.

## 3.9 Visual Generation Instantiation

![](images/6742c430b79a202378fdae9e4c26daa167bc5889749695f8a6883990f5b46adf.jpg)  
Figure 4: ProxyFormer in text-to-image and difusion conditional generation. Global conditions undergo cross-modal cross-attention only in the compressed image proxy space.

Figure 4 depicts the ProxyFormer structure for visual difusion/flow matching. Our imagegeneration experiments adopt both the JIT and JLT paradigms [21, 22]. Let the image or latent tensor be $u \in \mathbb { R } ^ { C \times H \times W }$ and the patch size be $p \times p .$ , then the spatial compression ratio is $P = p ^ { 2 }$ The local stream is obtained via convolutional embedding as $X _ { \mathrm { f i n e } } \in \mathbb { R } ^ { d _ { \mathrm { f i n e } } \times ( H / p ) \times ( W / p ) }$ ; the proxy stream $H \ \in \ \mathbb { R } ^ { ( H / p ) \times ( W / p ) \times d _ { \mathrm { p r o x y } } }$ is obtained by the 2D compression module. Text prompts, class labels, and time step conditions, once encoded, only perform cross-attention or condition injection with the proxy stream, before being symmetrically decompressed to reconstruct the local stream.

## 4 Computational and Memory Complexity Analysis

To simplify the analysis, let $d = d _ { \mathrm { p r o x y } } = d _ { \mathrm { m o d e l } }$ , and ignore the batch and attention head dimensions. Let the single-layer computation of full attention be $C _ { \mathrm { f u l l } } ( L ) \stackrel { } { = } \Theta ( L ^ { 2 } d )$ and the KV cache be $M _ { \mathrm { f u l l } } ( L ) = \Theta ( L _ { \mathrm { l a y e r } } L d )$ , where $L _ { \mathrm { l a y e r } }$ is the number of layers. In ProxyFormer, the proxy attention length is $m = L / P$ , and compression and decompression are block-wise linear operations. Thus, the single-layer computation is

$$
C _ { \mathrm { p r o x y } } ( L ) = \underbrace { \Theta \bigl ( ( L / P ) ^ { 2 } d \bigr ) } _ { \mathrm { P r o x y ~ A t t e n t i o n } } + \underbrace { \Theta ( L d _ { \mathrm { f i n e } } d ) } _ { \mathrm { C o m p . ~ + ~ D e c o m p . } } + \underbrace { \Theta ( L d _ { \mathrm { f i n e } } ) } _ { \mathrm { L o c a l ~ S t r e a m } } ,\tag{22}
$$

and the historical cache (proxy states only) is

$$
M _ { \mathrm { p r o x y } } ( L ) = \Theta \bigg ( L _ { \mathrm { l a y e r } } \frac { L } { P } d \bigg ) .\tag{23}
$$

Therefore, when $L \gg P d _ { \mathrm { f i n e } }$ , the proxy attention term dominates, and the computation is approximately $1 / P ^ { 2 }$ of full attention; when P is fixed, the compression/decompression terms grow only linearly. The NIAH and memory experiments in this paper use $P = 6 4$ , giving a theoretical proxy attention speedup of 4096 and a theoretical historical KV cache compression of 64 . Actual gains depend on the implementation, block size, and local stream maintenance cost.

Equation (22) also reveals a ”space-for-channel” reinvestment mechanism. Suppose the compression ratio is further increased from P to αP $( \alpha > 1 )$ , reducing the proxy sequence length to $1 / \alpha$ of its original length, and multiplying the proxy attention cost by $1 / \alpha ^ { 2 } ;$ if simultaneously the proxy channel is increased from $d _ { \mathrm { p r o x y } }$ to $\beta d _ { \mathrm { p r o x y } }$ , the proxy attention cost is multiplied by $\beta / \alpha ^ { 2 }$ . Therefore, as long as $\beta \leq \alpha ^ { 2 }$ , one can further compress the sequence without increasing the total proxy attention budget, while significantly enhancing the information capacity of each proxy vector. In other words, compute saved on the sequence axis through compression can be continuously reinvested into the channel axis, and channel capacity in turn provides room for further increasing the compression ratio.

Equation (22) also reveals the applicable boundaries of ProxyFormer: for short sequences or wide local streams where $d _ { \mathrm { f i n e } } \approx d ,$ , the linear compression term may exceed the quadratic attention term; thus, block compression is more suitable for long sequences and high-resolution inputs. By using asymmetric dual embeddings, $d _ { h } = 6 4 \ll d = 5 1 2$ , the constant of the linear term is further suppressed.

## 5 Experiments

## 5.1 Experimental Setup

Language Modeling and NIAH. Language model experiments use the GPT-2 tokenizer [23] (vocabulary size 50,257), model width $d _ { \mathrm { m o d e l } } = 5 1 2$ , attention head dimension $d _ { \mathrm { h e a d } } = 6 4$ totaling 10 layers. For ProxyFormer, the history embedding dimension is $d _ { h } = 6 4$ , generation region block size $C = 6 4$ , and compression ratio per layer $P = 6 4$ ; WikiText language modeling experiments use a block size of 128 and compression ratio of 32. Training employs a warmupcosine learning rate schedule, gradient accumulation, and bf16 mixed precision. The optimizer is Mano (performing manifold normalized updates on 2D parameters and AdamW on others) [24]. The final NIAH model for evaluation turns of positional encoding (NoPE); the implementation also supports Rotary Position Embedding (RoPE) [25]. The baseline decoder-only model uses the same width, depth, and attention structure, with the history region and generation region sharing full-dimensional embeddings.

Image Generation. Image experiments employ Flow Matching. CIFAR-10 and MNIST images are resized to $3 2 \times 3 2$ . The JIT model is trained in pixel space with 4 4 patches; the MNIST JLT first trains a ProxyFormer autoencoder to obtain a $1 6 \times 1 6 \times 4$ latent space, and then trains Flow Matching in the latent space. FID and Inception Score are calculated by torch-fidelity.

## 5.2 Language Model Perplexity

Table 1 presents the perplexity (PPL) results on WikiText-103-v1. All three sets of models turn of positional encoding. As shown: without history, ProxyFormer is close to the identically sized decoder-only baseline (22.10 vs 21.36); after introducing compressed history, the PPL drops to 21.01, outperforming the history-less baseline. This result indicates that proxy compression does not merely come at the expense of language modeling capability, but rather can bring gains through historical modeling.

Table 1: Perplexity on WikiText-103-v1 (NoPE).
<table><tr><td>Model</td><td>History</td><td>PPL ↓</td></tr><tr><td>Base (decoder-only)</td><td>None</td><td>21.36</td></tr><tr><td>ProxyFormer</td><td>None</td><td>22.10</td></tr><tr><td>ProxyFormer</td><td>Yes</td><td>21.01</td></tr></table>

## 5.3 Multi-Needle In A Haystack

Task Construction. Inspired by the NIAH evaluation [26, 27], we randomly sample a background text of length L tokens from WikiText-103; the background is evenly divided into N = 50 segments, and a needle is inserted at a random position within each segment. The format of the needle is

The passkey i is XXXXX . Remember it.

where i is the needle index and XXXXX is a random 5-digit password. Each document contains 50 needles covering a depth of roughly 0%–100%. We query each needle individually:

What is the passkey i ? The passkey i is

and perform greedy decoding (up to 10 new tokens). A hit is determined if the generated text contains the corresponding 5-digit password. We repeat 100 independent documents for each length, resulting in a total of 100  50 = 5, 000 retrievals per length. Background tokens are resampled when insuficient.

Curriculum Learning. Long-sequence NIAH training uses a four-stage curriculum: 256- token single-needle warmup, 256-token single-needle proxy training, 8K-token 50-needle training, and 8K/64K 50-needle training with disabled positional encoding. Each stage is manually stopped after the training loss approaches stabilizes, then proceeds to the next stage. Training a 64K-long context directly from scratch was found to be dificult to stably converge in our experiments.

Results. Figure 5 presents the depth-length heatmaps on 1K–1M (1,048,576) tokens for models with 8K and 64K training windows; color and value indicate retrieval accuracy. Within their respective efective extrapolation ranges, neither model exhibits obvious ”lost in the middle” phenomena, with accuracy varying smoothly across depth intervals. The 8K trained model maintains over 94% accuracy when extrapolated to 256K; the 64K trained model maintains 92%–95% accuracy across depth intervals when extrapolated to 1,048,576 tokens (approx. 16 the training length). It should be noted that these results come from smaller scale (d = 512, 10 layers) validation models intended to test the architecture’s extrapolability, rather than for direct comparison with larger-scale instruction models.

![](images/1c9337c0e246c7053d2d557ea05053e84f1a0577af1976d50be56b0ecf26a725.jpg)  
(a) 8K Training Window

![](images/99a332d4ad8188b084371e551b6e1efa91949fe850935389c9439c58eb8dd3bb.jpg)  
(b) 64K Training Window  
Figure 5: Multi-needle in a haystack heatmap. The horizontal axis is context length (tokens), the vertical axis is the depth interval of the needle, and the values are retrieval accuracy.

## 5.4 VRAM and Training Speed

Table 2 compares the standard decoder-only model, Base with full history attention, and ProxyFormer under 16 GB VRAM and batch size 1. Both standard decoder-only and Base require about 15.6 GB to train approx. 20K lengths; ProxyFormer requires only 2.9 GB for the same 20,992 token history, and training speed increases from 1.3 it/s to 16.10 it/s. When history length is increased to 716,800 (0.7M), ProxyFormer VRAM is 15.1 GB, still trainable on the same 16 GB GPU, whereas Base cannot run at that length. This comparison directly verifies the complexity analysis in Section 4.

Table 2: VRAM and training speed comparison under a 16 GB GPU with batch size 1.
<table><tr><td>Model</td><td>History len.</td><td>Gen. len.</td><td>VRAM (GB)</td><td>Thr.  $\mathrm { ( i t / s ) }$ </td></tr><tr><td>Standard decoder-only</td><td>0</td><td>20,496</td><td>15.6</td><td>2.6</td></tr><tr><td>Base (full-history attention)</td><td>20,992</td><td>128</td><td>15.6</td><td>1.3</td></tr><tr><td>ProxyFormer  $( P = 6 4 )$ </td><td>20,992</td><td>128</td><td>2.9</td><td>16.10</td></tr><tr><td>ProxyFormer (P = 64)</td><td>716,800</td><td>128</td><td>15.1</td><td>2.70</td></tr></table>

## 5.5 Image Generation

Table 3 gives the FID and Inception Score for image generation. ProxyFormer-JIT obtains FID 16.21 and IS 8.42 on CIFAR-10, and FID 2.06 and IS 1.95 on MNIST; JLT obtains FID 4.36 and IS 1.93 on MNIST. Compared to current large-scale difusion models, these scores are not SOTA, but they demonstrate that the same proxy compression and top-down reconstruction mechanism can be applied to both pixel-space and latent-space flow matching paradigms.

Table 3: Image Generation Evaluation (Lower FID is better, higher IS is better).
<table><tr><td>Dataset</td><td>Method</td><td>Space</td><td>FID ↓</td><td>IS↑</td></tr><tr><td>CIFAR-10</td><td>JIT</td><td>Pixel</td><td>16.21</td><td>8.42</td></tr><tr><td>MNIST</td><td>JIT</td><td>Pixel</td><td>2.06</td><td>1.95</td></tr><tr><td>MNIST</td><td>JLT</td><td>Latent</td><td>4.36</td><td>1.93</td></tr></table>

## 6 Discussion and Limitations

## 6.1 Source of Advantages

Our preliminary results support the following four mechanistic assessments. First, proxy space interaction reduces the cost of long-distance token communication from $L ^ { 2 }$ to $( L / P ) ^ { 2 }$ , which is the main source of VRAM and speed gains. Second, the local residual stream provides recoverability for compression: even if a layer’s compression is lossy, deeper layers can still access fine-grained information, which is likely a reason for the stable accuracy across depth in NIAH. Third, the proxy KV Cache allows history information to exist in an extremely short state during inference, obtaining linear caching without eviction-based compression. Fourth, implementation simplicity: compression, decompression, and proxy interaction are all composed of standard operators such as reshape, linear projection, convolution, and standard attention, facilitating replication, modification, and deployment in mainstream frameworks.

## 6.2 Limitations

• Limited benchmark coverage: Current public results focus on WikiText PPL and multineedle NIAH, and have not yet been evaluated at scale on benchmarks like LongBench, RULER, MMLU, or general dialogue;

• Small model scale: The experimental models are validation scale at $d _ { \mathrm { m o d e l } } = 5 1 2$ , 10 layers; scaling laws and training stability on larger models have yet to be verified;

• Compression module choices: The main results reported in this paper use fixed partitioning and linear/reshape compression; ablations on variants such as dynamic routing, overlapping patches, attention compression, and cross-layer heterogeneous dimensions are yet to be completed;

• Limited visual task scale: Image experiments are limited to MNIST and CIFAR-10, and have not been systematically compared with DiT/LDM-class models on high-resolution text-to-image generation;

• Theoretical analysis: The impact of the proxy stream on the expressive capacity of function classes and the upper bounds of compression error propagation with respect to depth and compression ratio require further formalization.

## 6.3 Future Work

Next steps include: verifying ProxyFormer’s scaling laws on 1B–7B scale models; studying dynamic routing partitioning based on saliency and semantic clustering; and further integrating proxy-vector interaction with eficient sequence modeling advances.

Specifically, we plan to migrate Mamba, SSM, and other linear-time sequence models [7, 8], as well as the fixed-length or constant-time state update strategies of TLinFormer and TConstFormer [18, 19], onto the proxy-vector sequence. The goal is to gradually replace the current proxy attention with linear- or near-constant-cost proxy state evolution, thereby further reducing proxy-level computation and KV cache. In particular, the O(1)-cache idea of TConstFormer is naturally complementary to ProxyFormer’s block compression; their combination may lead to a hierarchical state management scheme of “proxy compression + constant cache” for ultra-long streaming decoding.

Furthermore, we plan to extend the same architecture to higher-dimensional data such as 3D point clouds, voxels, and videos, evaluate JLT in high-resolution text-to-image and video generation, and conduct controlled comparisons with StreamingLLM, $_ \mathrm { H _ { 2 } O }$ , Infini-attention, and Landmark Attention on standard long-context benchmarks.

## 7 Conclusion

This paper proposed ProxyFormer, a dual-stream architecture centered on proxy vectors. By establishing a closed loop between the local stream and proxy stream—comprising bottomup compression, global interaction in proxy space, and top-down decompression and injection—ProxyFormer reduces the computation and caching costs of global interaction for ultralong sequences by approximately $P ^ { 2 }$ and $P$ times, respectively, without discarding the local residual stream. This architecture can uniformly describe 1D sequences, 2D images, 3D point clouds/voxels/videos, and higher-dimensional tensors, and can be simply implemented with common operators like reshape, linear projection, convolution, and standard attention. Multi-level factorization further controls the parameter size of compression modules at high compression ratios, a reinvestment mechanism on the channel dimension provides potential for further increasing compression ratios, and the proxy KV cache significantly reduces the memory and latency of long-history inference. Preliminary experiments have verified the architecture’s feasibility in language model perplexity, 1,048,576-token multi-needle in a haystack, trainable length under 16 GB VRAM, and pixel/latent-space flow matching image generation. The authors hope this architecture provides a new fundamental component for low-cost modeling of long contexts and high-resolution generation.

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need, August 2023. arXiv:1706.03762 [cs.CL].

[2] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness, June 2022. arXiv:2205.14135 [cs.LG].

[3] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, July 2023. arXiv:2307.08691 [cs.LG].

[4] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-Attention with Linear Complexity, June 2020. arXiv:2006.04768 [cs.LG].

[5] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The Long-Document Transformer, December 2020. arXiv:2004.05150 [cs.CL].

[6] Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big Bird: Transformers for Longer Sequences, January 2021. arXiv:2007.14062 [cs.LG].

[7] Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces, May 2024. arXiv:2312.00752 [cs.LG].

[8] Tri Dao and Albert Gu. Transformers are SSMs: Generalized Models and Eficient Algorithms Through Structured State Space Duality, May 2024. arXiv:2405.21060 [cs.LG].

[9] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R´e, Clark Barrett, Zhangyang Wang, and Beidi Chen. H\$ 2\$O: Heavy-Hitter Oracle for Eficient Generative Inference of Large Language Models, December 2023. arXiv:2306.14048 [cs.LG].

[10] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Eficient Streaming Language Models with Attention Sinks, April 2024. arXiv:2309.17453 [cs.CL].

[11] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave No Context Behind: Eficient Infinite Context Transformers with Infini-attention, August 2024. arXiv:2404.07143 [cs.CL].

[12] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, August 2020. arXiv:2006.16236 [cs.LG].

[13] Amirkeivan Mohtashami and Martin Jaggi. Landmark Attention: Random-Access Infinite Context Length for Transformers, November 2023. arXiv:2305.16300 [cs.CL].

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Difusion Probabilistic Models, December 2020. arXiv:2006.11239 [cs.LG].

[15] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling, February 2023. arXiv:2210.02747 [cs.LG].

[16] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-Resolution Image Synthesis with Latent Difusion Models, April 2022. arXiv:2112.10752 [cs.CV].

[17] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes, December 2022. arXiv:1312.6114 [stat.ML].

[18] Zhongpan Tang. Rethinking Transformer Connectivity: TLinFormer, A Path to Exact, Full Context-Aware Linear Attention, August 2025. arXiv:2508.20407 [cs.LG].

[19] Zhongpan Tang. From TLinFormer to TConstFormer: The Leap to Constant-Time Transformer Attention: Achieving O(1) Computation and O(1) KV Cache during Autoregressive Inference, August 2025. arXiv:2509.00202 [cs.LG].

[20] Zhongpan Tang. Compression is Routing: Reconstruction Error as an Intrinsic Signal for Modular Language Models, December 2025. arXiv:2512.16963 [cs.LG].

[21] Tianhong Li and Kaiming He. Back to Basics: Let Denoising Generative Models Denoise, January 2026. arXiv:2511.13720 [cs.CV].

[22] Funing Fu, Tenghui Wang, Guanyu Zhou, Junyong Cen, and Qichao Zhu. JLT: Clean-Latent Prediction in Latent Difusion Transformers, May 2026. arXiv:2605.27102 [cs.CV].

[23] Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Technical Report, 2019.

[24] Yufei Gu and Zeke Xie. Mano: Restriking Manifold Optimization for LLM Training, January 2026. arXiv:2601.23000 [cs.LG].

[25] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding, November 2023. arXiv:2104.09864 [cs.CL].

[26] Greg Kamradt. Needle in a haystack: A pressure test for llms. https://github.com/ gkamradt/LLMTest\_NeedleInAHaystack, 2023.

[27] Elliot Nelson, Georgios Kollias, Payel Das, Subhajit Chaudhury, and Soham Dan. Needle in the Haystack for Memory Based Large Language Models, July 2024. arXiv:2407.01437 [cs.CL].

## A Implementation and Reproduction Details

## A.1 Code and Reproducibility

The complete source code and training/evaluation configurations are publicly available at:

$$
\mathrm { h t t p s : / / g i t h u b . c o m / s i m o n F e l i x - A i / p r o x y ^ { - } f o r m e r . g i t }
$$

## A.2 Intellectual Property and Licensing

The core ProxyFormer architecture described in this paper has been submitted for patent protection and is currently patent pending. Making the source code public does not constitute an implied license for commercial implementation of the architecture. Academic and non-commercial research use follows the license in the repository; any commercial deployment, for-profit corporate research and development, or product integration must be separately negotiated with and authorized by the author.