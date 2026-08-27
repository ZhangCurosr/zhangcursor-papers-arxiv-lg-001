# Group-Shared Low-Rank Approximation for Mobile-Eficient Pointwise Convolutions in Large-Kernel CNNs

Hao Luo Xi’an University of Architecture and Technology Xi’an, China

Man Jiang Xi’an University of Architecture and Technology Xi’an, China

Ting Jiang   
China Mobile Chengdu Institute   
of Research and Development Chengdu, China   
Qingsen Yan   
Northwestern Polytechnical   
University   
Xi’an, China

Yiting Yang Xi’an University of Architecture and Technology Xi’an, China

Zhijun Lin   
Northwestern Polytechnical   
University   
Xi’an, China

Kunming Luo Hong Kong University of Science and Technology Hong Kong, China

Guoqing Wang University of Electronic Science and Technology of China Chengdu, China

Peng Wang University of Electronic Science and Technology of China Chengdu, China

Wenyi Zhao Xi’an University of Architecture and Technology Xi’an, China

Ghulam Mohiuddin Nanchang University Nanchang, China

Zihao Zhang   
Institute of AI for Industries,   
Chinese Academy of Sciences Beijing, China

Wei Dong Xi’an University of Architecture and Technology Xi’an, China

## Abstract

Large-kernel Convolutional Neural Networks (CNNs) de liver remarkable performance in vision tasks by significantly expanding receptive fields, yet their quadratic parameter growth critically impedes storage-eficient edge deployment. While existing eficient architectures adopt parameter-eficient depthwise separable convolution backbones that leverage techniques like low-rank approximation and weight sharing to compress depthwise convolutions, we

identify a critical oversight: pointwise convolutions dominate parameter volume (>87% in models like RepLKNet-31B) and constitute the primary deployment botleneck on resource-constrained edge devices. This results in prohibitive storage costs and severe memory-loading constraints on resource-limited devices (e.g., smartphones with 4-12 GB Random Access Memory (RAM)). To overcome this, we propose Channel Group-Shared (CGS) low-rank approximation, a novel Singular Value Decomposition (SVD)-based parameter-sharing strategy. CGS constructs a structured low-rank paradigm isomorphic to SVD decomposition, comprising shared (high-parameter-cost) down/up-projection matrices across channel groups within a layer and channelgroup-specific (low-parameter-cost) scalable diagonal matrices. This group-sharing design achieves significant parameter reduction. Extensive experiments demonstrate that large-kernel CNNs (RepLKNet, ConvNeXt, SLaK) enhanced with CGS strike an empirically favorable balance between competitive performance and substantially reduced storage costs. Crucially, by alleviating storage constraints, reducing memory bandwidth pressure during loading, and minimizing model loading latency, CGS enables the feasible deployment of pre-trained large-kernel CNN models on edge devices, thereby bridging the gap between high-performance vision models and practical edge deployment.

## CCS Concepts

• Computing methodologies → Neural networks.

## Keywords

Large-kernel CNNs, Pointwise convolutions, Channel group-shared low-rank approximation, Edge deployment, Model compression, Computational eficiency

## ACM Reference Format:

Hao Luo, Yiting Yang, Wenyi Zhao, ManJiang, Zhijun Lin, Ghulam Mohiuddin, Ting Jiang, Kunming Luo, Zihao Zhang, Qingsen Yan, Guoqing Wang, Wei Dong, and Peng Wang. 2026. Group-Shared Low-Rank Approximation for Mobile-Eficient Pointwise Convolutions in Large-Kernel CNNs. In The 32nd Annual International Conference on Mobile Computing and Networking (MobiCom ’26), October 26–30, 2026, Austin, TX, USA. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3795866.3844469

## 1 Introduction

Large-kernel Convolutional Neural Networks (CNNs) have recently emerged as a leading architecture for visual recognition tasks, owing to their ability to capture long-range dependencies through significantly expanded receptive fields [15, 35, 38]. The pursuit of expanding receptive fields has driven the development of large-kernel CNNs, which deliver remarkable performance in vision tasks such as image classification [17, 61], object detection [11, 37, 61], and semantic segmentation [7, 51, 52, 55]. By leveraging convolutional kernels significantly larger than traditional 3 × 3 filters (e.g., 7 × 7 to 51 × 51 [15, 35, 38]), these models efectively capture long-range dependencies and contextual information, achieving State-of-the-Art (SOTA) results across a variety of applications. However, this performance gain comes at a steep cost: the quadratic growth of param eters within large kernels severely compromises storage eficiency. For instance, the number of parameters of a 31 × 31 [15] kernel can reach approximately 100 times that of a 3×3 kernel.

To mitigate this issue, recent eficient large-kernel architectures (e.g., RepLKNet [15], ConvNeXt [38], SLaK [35]) predominantly adopt parameter-eficient depthwise separable convolution [8, 23] as their backbone. This backbone includes two components: depthwise convolution and pointwise convolution. Depthwise convolution applies a single filter per input channel to capture spatial features, while pointwise convolution (a 1 × 1 convolution) combines the output channels through linear projection. Together, they form the core of depthwise separable convolution, significantly reducing computation and parameters compared to standard convolution.

![](images/69a63ca5ba18fadad904da172d3d850e971d0e2102c9a439895238a9ea1f79c5.jpg)  
Figure 1: Parameter distribution in large-kernel CNN models: proportions from pointwise convolutions, depthwise convolutions, and other components across RepLKNet-31B, ConvNeXt-B, and SLaK-B pre-trained models (where � denotes the Base variant of each architecture).

Techniques like low-rank approximation [49, 64], weight sharing [4], and dilated convolution [57, 58] have been successfully applied to compress parameters in the depthwise convolution component, achieving manageable model sizes for resource-constrained deployment. Yet, a critical botleneck persists: pointwise convolutions now dominate the parameter volume of large-kernel CNN models, often accounting for over 87% of the total parameters, as shown in Fig. 1 for the Base variants. This impedes the deployment of highperformance vision models on edge devices [29, 44] such as smartphones and Internet of Things (IoT) terminals, a critical step toward scalable and privacy-preserving artificial intelligence. Furthermore, a layer-wise analysis of RepLKNet-31B (Fig. 2) reveals that this parameter imbalance persists across all network stages, with pointwise convolutions consistently dominating parameter allocation. Concurrently, Fig. 3 demonstrates that Stages 3 and 4 collectively account for approximately 97% of total parameters in ImageNet-1K [13] pre-trained RepLKNet-31B [15], ConvNeXt-B [38], and SLaK-B [35] architectures. This pronounced parameter concentration within later stages is characteristic ofmodern large-kernel CNNs. Surprisingly, despite its outsized impact, compressing pointwise convolutions in large-kernel CNNs remains largely unexplored; existing research focuses overwhelmingly on optimizing the depthwise convolution component while leaving this high-cost operator unaddressed. This greatly limits the process of deploying large-kernel CNN models on edge devices.

![](images/82f149a963e375574b9eec9de76677c0ec0c54bebe18035a82d8b5d841a3570e.jpg)  
Figure 2: Parameter distribution across stages in pre-trained RepLKNet-31B: pie charts quantify the proportion of parameters from pointwise convolutions, depthwise convolutions, and other components within each stage.

![](images/1046a692796c8a92b7d0e50573e89b63c2b360660b470357978ed89d0caaa2d0.jpg)  
Figure 3: Stage-wise parameter distribution across Stages 1–4 in pre-trained RepLKNet-31B, ConvNeXt-B, and SLaK-B models: each bar quantifies the proportion of parameters from individual stages relative to the total model parameters.

Therefore, deploying large-kernel CNNs [15] on edge devices [29, 44] poses significant challenges, particularly storage constraints arising from massive parameter counts dominated by pointwise convolutions [24, 43] that exceed device capacities and increase transmission overhead, and runtime memory limitations during model loading where redundant parameters cause ineficient memory bandwidth utilization while risking peak overload that may trigger system instability.

To bridge this gap, we propose Channel Group-Shared (CGS) low-rank approximation, a novel parameter compression strategy grounded in Singular Value Decomposition (SVD) theory. Our key innovation lies in constructing a structured low-rank approximation paradigm that is isomorphic to the SVD factorized form. This paradigm decomposes a pointwise convolution layer into two components: (1) High-parameter-cost down/up-projection matrices (analogous to left/right-unitary matrices in SVD) that are shared across multiple channel groups within the same layer; (2) Low-parameter-cost group-specific scalable diagonal matrices (analogous to singular value diagonal matrix in SVD) whose elements are uniquely learned per channel group. By sharing the bulky projection matrices across groups while retaining lightweight group-specific scaling elements, CGS achieves dramatic parameter reduction without structural degradation. This makes the compressed large-kernel CNN models feasible for deployment on resource-constrained edge devices such as smartphones and IoT terminals. Specifically, the significant reduction in parameters directly alleviates the storage burden of edge devices, allowing the model to fit into limited local storage. Meanwhile, the streamlined parameter structure reduces the memory footprint during model loading and lowers model loading latency. In this way, CGS paves the way for deploying high-performance large-kernel convolution pretrained models on edge devices, bridging the gap between SOTA model performance and real-world deployment constraints.

Our core contributions are fourfold:

• Through empirical investigation, we identify that pointwise convolutions consume over 87% of parameters in large-kernel CNNs, yet remain underoptimized. We thus propose an SVD-inspired CGS low-rank approximation that decomposes them into shared basis projections and group-adaptive scaling, significantly reducing parameters with lower capacity loss.

• We establish a mathematically-deduced compression framework derived from the isomorphism of SVD factorization, providing geometric interpretability and theoretical guarantees.

• Through extensive experiments on ImageNet-1K [13] classification, ADE20K [67] semantic segmentation, and COCO [33] object detection, we show CGS lowrank approximation reduces parameter by >81% with <4.2% accuracy drop (e.g., CGS-B versus RepLKNet-31B, as detailed in Tab. 2), achieving an empirically observed balance between storage and accuracy among the compared compression methods.

• In terms of deployment on edge devices, we achieve the successful deployment of large-kernel convolutional pre-trained backbones on mobile platforms, and furthermore, empirically validate its efectiveness.

## 2 Background and Motivation

## 2.1 Eficiency Challenges of Pointwise Convolutions in Large-Kernel CNNs

Modern large-kernel CNNs (such as RepLKNet [15], ConvNeXt [38], SLaK [35]) have atained remarkable breakthroughs in visual tasks like image classification [17, 61], object detection [11, 37, 61], and semantic segmentation [7, 51, 52, 55], achieved through employing large-sized convolutional kernels ranging from $7 \times 7 ~ [ 3 8 ]$ to $5 1 \times 5 1$ [35]. By expanding the receptive field, these architectures efectively capture long-range dependencies, pushing the performance boundaries of a series of visual tasks.

However, this architectural innovation also introduces significant storage eficiency challenges. Contrary to the common assumption that depthwise convolution are the primary driver of parameter growth, our analysis reveals that pointwise convolution constitute the dominant source of the overall parameter count. This highly skewed parameter distribution highlights a critical issue: while depthwise convolution are efective at capturing long-range spatial dependencies, the matrix multiplications of pointwise convolution, operating on high channel dimensions, have become the primary storage botleneck, severely hindering model deployment on resource-constrained edge devices.

## 2.2 Limitations of Existing Compression Methods

Contemporary eficient architectures predominantly leverage depthwise separable convolution backbones to mitigate parameter explosion. Current research focuses extensively on depthwise convolution compression through techniques including: (1) Kernel Factorization: SLaK [35] decomposes oversized kernels $( \mathrm { e . g . , } 5 1 \times 5 1  5 1 \times 5 + 5 \times 5 1 )$ with dynamic sparsity; (2) Parameter Sharing: PeLK [4] employs a “focus-blur” mechanism with exponential scaling grids for >90% parameter reduction; (3) Low-Rank Approximation: convolutional kernels are compressed through lowrank approximation using Truncated SVD [49, 64]; (4) Dilated Convolutions: multi-scale context aggregation without kernel expansion [57, 58].

Despite these advances, a critical limitation persists: existing methods universally neglect pointwise convolution, which dominates parameterization. This oversight creates a paradoxical eficiency gap, where compressing depthwise convolutions yields only marginal gains while pointwise operations remain prohibitively expensive for edge deployment. Specifically, (1) pointwise convolution parameters impose prohibitive storage demands; (2) memory loading requirements during initialization exceed mobile bandwidth capacities; and (3) in large-kernel CNNs, the compression of pointwise convolutions remains a botleneck.

## 3 Methodology

This section first provides a brief background on Singular Value Decomposition (SVD) [32] and the equivalence properties of orthogonal matrices [18, 47]. Based on this foundation, we then introduce the Channel Group-Shared (CGS) method for compressing pointwise convolutions by constructing kernels as a composition of shared down/upprojection matrices and channel-group-specific scalable diagonal matrices. Finally, we describe the deployment pipeline for transferring cloud pre-trained lightweight large-kernel convolutional foundation models to resourceconstrained mobile devices for eficient execution.

## 3.1 Preliminary Knowledge

3.1.1 Singular Value Decomposition Fundamentals. Singular Value Decomposition (SVD) [32] provides a complete factorization of any real-valued matrix $\mathbf { W } \in \mathbb { R } ^ { m \times n }$ into orthogonal matrices and a diagonal matrix of singular values:

$$
\mathbf { W } = \mathbf { U } \Sigma \mathbf { V } ^ { \top } ,\tag{1}
$$

where $\mathbf { U } \in \mathbb { R } ^ { m \times m }$ is the left singular matrix with orthogonal columns $( \mathbf { U } ^ { \top } \mathbf { U } = \mathbf { I } _ { m } ) ; \mathbf { V } \in \mathbb { R } ^ { n \times n }$ is the right singular matrix with orthogonal columns $( \mathbf { V } ^ { \top } \mathbf { V } = \mathbf { I } _ { n } ) ; \bar { \Sigma } \in \mathbb { R } ^ { m \times n }$ is the singular value matrix with diagonal elements $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdots \geq$ $\sigma _ { r } > 0 ( r = \mathrm { r a n k } ( \mathbf { W } ) )$ and $\Sigma _ { i j } = 0$ for $i \neq j .$

A key property of this decomposition is its dimensional consistency. Any two matrices $\mathbf { W } _ { i } , \mathbf { W } _ { j } \in \mathbb { R } ^ { m \times n }$ of the same dimensions decompose into factors of identical shapes:

$$
\begin{array} { r } { \mathbf { W } _ { i } = \mathbf { U } _ { i } \boldsymbol { \Sigma } _ { i } \mathbf { V } _ { i } ^ { \top } , } \\ { \mathbf { W } _ { j } = \mathbf { U } _ { j } \boldsymbol { \Sigma } _ { j } \mathbf { V } _ { j } ^ { \top } , } \end{array}\tag{2}
$$

where $\mathbf { U } _ { i } , \mathbf { U } _ { j } \in \mathbb { R } ^ { m \times m }$ and $\mathbf { V } _ { i } , \mathbf { V } _ { j } \in \mathbb { R } ^ { n \times n }$ . Because the SVD factors share the same dimensions, their parameters can be structured and shared eficiently. This dimensional matching is the foundational insight for our compression framework.

3.1.2 Orthogonal Matrix Equivalence. Building on the orthogonal properties of SVD factor matrices in the real domain, we leverage a fundamental equivalence: any two orthogonal matrices of identical dimensions can be expressed as linear transformations of one another through invertible mappings [18, 47]. Formally, for orthogonal matrices $\mathbf { M } _ { \mathrm { o r t h } _ { a } } , \mathbf { M } _ { \mathrm { o r t h } _ { b } } \in \mathbb { R } ^ { m \times m }$ , there exist invertible matrices $\mathbf { P } , \mathbf { Q } \in \mathbb { R } ^ { m \times m }$ such that:

![](images/aea459a61e8a3b7672a1c9a3a858f2ac15e4a77c71c1432c2bb8ae5a810ac72a.jpg)  
Figure 4: Schematic of the CGS method. Pointwise convolution weights are partitioned into $k _ { \mathbf { a u t o } }$ submatrices, each approximated by shared low-rank projections and a group-specific diagonal. Depthwise convolution weights are approximated per channel using shared projections and a channel-specific diagonal.

$$
\mathbf { M } _ { \mathrm { o r t h } _ { a } } = \mathbf { P } \mathbf { M } _ { \mathrm { o r t h } _ { b } } \mathbf { Q } .\tag{3}
$$

This equivalence enables our group-sharing paradigm. Consider � orthogonal matrices $\{ \mathbf { U } _ { g } \in \mathbb { R } ^ { m \times m } \} _ { g = 1 } ^ { k }$ with identical dimensions. We express each through a shared base matrix $\mathbf { U } _ { \mathrm { s h a r e d } } \in \mathbb { R } ^ { m \times m }$ and group-specific transformations:

$$
\left\{ \begin{array} { l l } { \mathbf { U } _ { 1 } = \mathbf { P } _ { 1 } \mathbf { U } _ { \mathrm { s h a r e d } } \mathbf { Q } _ { 1 } } \\ { \mathbf { U } _ { 2 } = \mathbf { P } _ { 2 } \mathbf { U } _ { \mathrm { s h a r e d } } \mathbf { Q } _ { 2 } } \\ { \quad \ : } \\ { \mathbf { U } _ { k } = \mathbf { P } _ { k } \mathbf { U } _ { \mathrm { s h a r e d } } \mathbf { Q } _ { k } } \end{array} \right. ,\tag{4}
$$

where $\mathbf { P } _ { g } , \mathbf { Q } _ { g } \in \mathbb { R } ^ { m \times m }$ are invertible matrices unique to group �.

![](images/398fc0e06c023af2ad46598749cf003b98d70ea677ca6ad76b5a95b9a3ae23df.jpg)  
Figure 5: Experimentally categorized parametersharing strategies across three complexity scales: (1) Base Configuration: assimilates outer invertible matrices into shared parameter space; (2) Small Configuration: employs complete matrix assimilation; (3) Large Configuration: directly implements the theoretically derived formulation with maximal parameter retention.

## 3.2 Channel Group-Shared Low-Rank Approximation

To implement this strategy in pointwise convolutions, we adopt a channel grouping scheme. An automatic group count $k _ { \mathrm { a u t o } }$ is designed based on the output-to-input channel ratio. This design removes the need for manual tuning while balancing model compression (reducing parameters) and representational capacity, establishing it as the baseline group number for subsequent experiments. Specifically, $k _ { \mathrm { a u t o } }$ is given by:

$$
k _ { \mathrm { a u t o } } = \{ C _ { \mathrm { i n } } / C _ { \mathrm { o u t } } , \mathrm { i f } \ : C _ { \mathrm { i n } } > C _ { \mathrm { o u t } } ,\tag{5}
$$

where $C _ { \mathrm { i n } }$ and $C _ { \mathrm { o u t } }$ represent the number of input/output channels.

The pointwise convolution weight matrix ${ \bf w } ^ { ( P ) } \in \mathbf { \Omega }$ $\mathbb { R } ^ { C _ { \mathrm { { o u t } } } \times C _ { \mathrm { { i n } } } ^ { * } }$ is partitioned into $k _ { \mathrm { a u t o } }$ equal-sized group matrices along the input channel dimension:

$$
\mathbf { W } ^ { ( P ) } = \left[ \mathbf { W } _ { 1 } ^ { ( P ) } \mid \mathbf { W } _ { 2 } ^ { ( P ) } \mid \cdots \mid \mathbf { W } _ { k _ { \mathrm { a u t o } } } ^ { ( P ) } \right] ,\tag{6}
$$

where each group matrix $\begin{array} { r l r } { { \bf W } _ { i } ^ { ( P ) } } & { { } \in } & { \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { o u t } } } } \end{array}$ for $i \in$ $\{ 1 , 2 , \dots , k _ { \mathrm { a u t o } } \}$ when $C _ { \mathrm { i n } } = k _ { \mathrm { a u t o } } \cdot C _ { \mathrm { o u t } } .$

Furthermore, our method is also applicable to depthwise convolution. The depthwise convolution kernel ${ \bf w } ^ { ( D ) } \in { \bf \Xi }$

$\mathbb { R } ^ { K _ { h } \times K _ { w } \times C }$ , where $K _ { h }$ and $K _ { w }$ represent the height and width of the convolutional kernel respectively, is partitioned along the channel dimension into � group kernels:

$$
\mathbf { W } ^ { ( D ) } = \left[ \mathbf { W } _ { 1 } ^ { ( D ) } \mid \mathbf { W } _ { 2 } ^ { ( D ) } \mid \cdots \mid \mathbf { W } _ { C } ^ { ( D ) } \right] ,\tag{7}
$$

where each group kernel $\begin{array} { r l r } { { \bf W } _ { i } ^ { ( D ) } } & { { } \in } & { \mathbb { R } ^ { K _ { h } \times K _ { w } } } \end{array}$ for $i \in$ $\{ 1 , 2 , \ldots , C \}$

This identitical group partitioning scheme maintains architectural consistency with our pointwise convolution implementation. Unified dimensional factorization enables parameter sharing strategy across all channel groups, as shown in Fig. 4.

Our derivation proceeds in two sequential steps:

(1) Application of Eq. (1) to $\mathbf { W } _ { i } ^ { ( P ) }$ and $\mathbf { w } _ { i } ^ { ( D ) }$ yields:

$$
\mathbf { W } _ { i } ^ { ( \cdot ) } = \mathbf { U } ^ { ( \cdot ) } \mathbf { D } ^ { ( \cdot ) } \mathbf { V } ^ { \top ( \cdot ) } ,\tag{8}
$$

where $\mathbf { w } _ { i } ^ { ( \cdot ) }$ generically represents either $\mathbf { W } _ { i } ^ { ( P ) }$ or $\mathbf { w } _ { i } ^ { ( D ) }$ throughout the derivation.

(2) Subsequent application of Eq. (3) to matrices $\mathbf { U } ^ { ( \cdot ) }$ and $\mathbf { V } ^ { \top ( \cdot ) }$ within Eq. (8) produces:

$$
\mathbf { W } _ { i } ^ { ( \cdot ) } = \mathbf { P } _ { u } ^ { ( \cdot ) } \mathbf { W } _ { \mathrm { s h a r e d \_ u } } ^ { ( \cdot ) } \mathbf { Q } _ { u } ^ { ( \cdot ) } \mathbf { D } ^ { ( \cdot ) } \mathbf { P } _ { \nu } ^ { ( \cdot ) } \mathbf { W } _ { \mathrm { s h a r e d \_ v } } ^ { ( \cdot ) } \mathbf { Q } _ { \nu } ^ { ( \cdot ) } .\tag{9}
$$

To reduce the number of parameters, we let P and Q be diagonal matrices $\mathbf { D } _ { P }$ and $\mathbf { D } _ { Q }$ , respectively. Eq. (9) can be rewriten as:

$$
\mathbf { W } _ { i } ^ { ( \cdot ) } = \mathbf { D } _ { P _ { u } } ^ { ( \cdot ) } \mathbf { W } _ { \mathrm { s h a r e d } _ { - } \mathbf { u } } ^ { ( \cdot ) } \mathbf { D } _ { Q _ { u } } ^ { ( \cdot ) } \mathbf { D } ^ { ( \cdot ) } \mathbf { D } _ { P _ { v } } ^ { ( \cdot ) } \mathbf { W } _ { \mathrm { s h a r e d } _ { - } \mathbf { v } } ^ { ( \cdot ) } \mathbf { D } _ { Q _ { v } } ^ { ( \cdot ) } .\tag{10}
$$

As illustrated in Fig. 5 (Large), the CGS-L method corresponds to the structure represented by Eq. (10). Since $\mathbf { W } _ { \mathrm { s h a r e d } } ^ { ( \cdot ) }$ are trainable parameters, to strike a balance between parameter count and model performance, we consider merging the outermost $\mathbf { D } _ { P _ { u } } ^ { ( \cdot ) }$ and $\mathbf { D } _ { Q _ { \nu } } ^ { ( \cdot ) }$ into $\mathbf { w } _ { \mathrm { s h a r e d } } ^ { ( \cdot ) } \mathrm { . }$

$$
\mathbf { W } _ { i } ^ { ( \cdot ) } = \mathbf { W } _ { \mathrm { s h a r e d } _ { - } \mathbf { u } } ^ { ( \cdot ) ^ { \prime } } \mathbf { D } _ { Q _ { u } } ^ { ( \cdot ) } \mathbf { D } ^ { ( \cdot ) } \mathbf { D } _ { P _ { v } } ^ { ( \cdot ) } \mathbf { W } _ { \mathrm { s h a r e d } _ { - } \mathbf { v } } ^ { ( \cdot ) ^ { \prime } } .\tag{11}
$$

As illustrated in Fig. 5 (Base), the CGS-B method corresponds to the structure represented by Eq. (11). To further explore the flexibility of the approach and achieve a beter trade-of between model parameter size and performance, we consider merging $\mathbf { D } _ { Q _ { u } } ^ { ( \cdot ) }$ and $\mathbf { D } _ { P _ { \nu } } ^ { ( \cdot ) }$ into $\mathbf { w } _ { \mathrm { s h a r e d } } ^ { ( \cdot ) ^ { \prime } } .$

$$
\mathbf { W } _ { i } ^ { ( \cdot ) } = \mathbf { W } _ { \mathrm { s h a r e d \_ u } } ^ { ( \cdot ) ^ { \prime \prime } } \mathbf { D } ^ { ( \cdot ) } \mathbf { W } _ { \mathrm { s h a r e d \_ v } } ^ { ( \cdot ) ^ { \prime \prime } } .\tag{12}
$$

As illustrated in Fig. 5 (Small), the CGS-S method corresponds to the structure represented by Eq. (12).

The CGS compression strategy is applied selectively to Stages 3-4 of the model, as shown in Tab. 6, while earlier stages (1-2) are responsible for extracting low-level features and exhibit high sensitivity to compression. As shown

![](images/07d1b78c067064ef164986e6f8202432615dc06d81e2ab5ebe18e014aeae563f.jpg)

![](images/8a4687166c9319f2f63942896b93647e0c91fa08b1602ed56e5cd6f470586d4a.jpg)

(a) Depthwise convolution (b) Pointwise convolution in Stage 3. in Stage 3.  
![](images/1a7f94e567469587be65ff47e9ced0b3140247f53ce7ac04f2154634f2d6182d.jpg)

![](images/94bb557b1bb4b5ce0ab74e3794ea07ac2994ab667bbc9e9e8118bd1b230f11c6.jpg)  
(c) Depthwise convolution (d) Pointwise convolution in Stage 4. in Stage 4.

Figure 6: Singular value distributions of pointwise and depthwise convolutional kernels in pre-trained RepLKNet-31B (Stages 3-4) after channel grouping. Each subfigure aggregates the kernel matrices grouped by stage and convolution type, with individual matrices color-coded. X-axis: Singular values; Yaxis: Count of singular values within defined bounds.

in Fig. 6, the analysis of the singular values from pointwise and depthwise convolutions in these stages of a pretrained RepLKNet-31B model reveals a pronounced lowrank property and a power-law distribution, indicating that the weight matrices are inherently sparse and can be efectively reconstructed using a limited set of basis vectors. This observation directly motivates the design of the down/upprojection matrices as low-rank factorizations, achieving an efective balance between model performance and parameter reduction.

## 3.3 Deployment on Mobile Devices

This section outlines the deployment pipeline for pretrained lightweight large-kernel convolutional foundation models on resource-constrained mobile devices. Models such as RepLKNet+CGS-B are first pre-trained on cloud servers using Graphics Processing Unit (GPU) acceleration. For deployment, the pre-trained weights are loaded with explicit Central Processing Unit (CPU) device mapping to ensure hardware compatibility, and the model architecture is programmatically reconstructed via parameter transplantation. The model is then converted into a static computation graph using TorchScript tracing with representative input tensors, and the hardware-agnostic binary is serialized for mobile execution. Finally, the compiled .pt file is integrated into an Android application via Android Studio [19]. As empirically validated in Fig. 7, the deployed model performs accurate image classification on mobile devices, with real-time measurements confirming the practical benefits of CGS for edge deployment.

![](images/0f11d8534d6a29f70a202ffb40dbc895a68fbead7cc531ac5eddfbdd7835c91b.jpg)  
Figure 7: This diagram schematically illustrates our edge deployment pipeline for pre-trained lightweight large-kernel convolutional foundation models: (1) pre-trained models optimized for mobile hardware are generated on cloud servers; (2) the compressed models are deployed to target mobile devices; (3) successful on-device inference execution is validated through real-world image inputs.

## 4 Experiments

We conduct extensive experiments to validate the efectiveness of the proposed lightweight sharing strategy. The evaluation encompasses multiple core vision tasks to assess generalization ability. In addition, we perform ablation studies to verify our design choices and investigate the impact of key hyperparameters. Finally, we conduct on-device experiments on mobile platforms to demonstrate practical feasibility.

## 4.1 Experiments Setting

4.1.1 Datasets and Tasks. We follow established evaluation protocols [4, 15, 35, 38] across diverse tasks: large-scale classification (ImageNet-1K [13]), fine-grained classification (CIFAR-100 [28]), scene parsing (ADE20K [67]), and objectlevel recognition (COCO [33]). This multi-task benchmark ensures a comprehensive assessment of model robustness and generalization.

4.1.2 Baselines. We compare against three state-of-the-art large-kernel CNN baselines: RepLKNet [15], SLaK [35], and ConvNeXt [38]. PeLK [4] is excluded due to the unavailability of its oficial implementation.

4.1.3 CGS Variants. To balance performance and eficiency, we instantiate three CGS variants of increasing capacity: CGS-S, CGS-B, and CGS-L. As illustrated in Fig. 5, they correspond to the formulations in Eq. 10, Eq. 11, and Eq. 12, respectively. Enhanced representational capacity in the CGS-B and CGS-L configurations is achieved by incorporating additional stretching vectors into the down/upprojection matrices.

Table 1: ImageNet-1K classification performance. Top-1 accuracy and parameter counts are compared across variants, demonstrating competitive results under similar model complexity.
<table><tr><td>Model</td><td>Params (M)</td><td>Top-1 Acc</td></tr><tr><td>ResNet18 [22]</td><td>11.7</td><td>69.8</td></tr><tr><td>PoolFormer-S12 [59] RepLKNet+CGS-S (Ours)</td><td>11.9 14.1</td><td>77.2 79.1</td></tr><tr><td>PVT-Tiny [52]</td><td>13.2</td><td>75.1</td></tr><tr><td>PVTv2-B1 [53]</td><td>13.1</td><td>78.7</td></tr><tr><td>RepLKNet+CGS-B (Ours)</td><td>14.7</td><td>80.0</td></tr><tr><td>RegNetY-4G [41]</td><td></td><td></td></tr><tr><td>DeiT-Small/16 [50]</td><td>21.0</td><td>80.0</td></tr><tr><td></td><td>22.1</td><td>79.8</td></tr><tr><td>T2T-ViT-14 [60]</td><td>21.5</td><td>81.7</td></tr><tr><td>gMLP-S [34]</td><td>20.0</td><td>79.6</td></tr><tr><td>PoolFormer-S24 [59]</td><td>21.4</td><td>80.3</td></tr><tr><td>RepLKNet+CGS-L (Ours)</td><td>17.1</td><td>82.1</td></tr></table>

4.1.4 Implementation Details. Fig. 3 shows the parameter distribution across stages for pre-trained RepLKNet-31B, ConvNeXt-B, and SLaK-B. A key observation is that parameters are heavily concentrated in the later stages (Stages 3- 4), accounting for approximately 97% of the total. As results in Tab. 6 confirm, the early stages (1-2) are crucial for lowlevel feature extraction and highly sensitive to compression. Therefore, we apply the CGS compression strategy selectively only to Stages 3 and 4. All models are implemented in PyTorch and trained on NVIDIA A800 GPUs; detailed configurations are provided in Appendix A.

## 4.2 Experimental Comparisons

4.2.1 ImageNet Classification. We evaluate CGS-S/B/L integrated into RepLKNet on ImageNet-1K [13]. As shown in Tab. 1, all variants achieve beter performance under comparable parameter budgets, with CGS-L ataining superior accuracy at lower cost. A broader comparison against SLaK [35] and ConvNeXt [38] (Tab. 2) confirms that CGS significantly reduces parameters while maintaining competitive accuracy.

Overall, CGS-B achieves an favorable eficiency-accuracy trade-of. Compared to PVT-Tiny [52] at a similar scale, it improves Top-1 accuracy by 6.5% with only a 1.5M parameters increase. Against RepLKNet-31B [15], it reduces parameters by 81% while limiting the accuracy drop to 4.2%. These results validate that our lightweight sharing strategy efectively compresses models without compromising performance.

Table 2: ImageNet-1K classification performance. Top-1 accuracy and parameter counts are compared across variants, with our method achieving an empirically favorable accuracy-parameter trade-of.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=2>Params (M) Top-1 Acc</td></tr><tr><td rowspan=7 colspan=1>T2T-ViTt-14 [60]PerViT-S [40]Swin-T [37]ConvNeXt-T [38]SLaK-T [35]PeLK-T [4]RepLkNet+CGS-S (Ours)</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>81.7</td></tr><tr><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>82.1</td></tr><tr><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>1.3</td></tr><tr><td rowspan=1 colspan=1>29</td><td rowspan=2 colspan=1>82.182.5</td></tr><tr><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>29</td><td rowspan=2 colspan=1>82.679.1</td></tr><tr><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=6 colspan=1>PVT-Large [52]T2T-ViTt-19 [60]PerViT-M [40]Swin-S [37]ConvNeXt-S [38]SLaK-S [35]</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>81.7</td></tr><tr><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>82.4</td></tr><tr><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>82.9</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>83.0</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>83.1</td></tr><tr><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>83.8</td></tr><tr><td rowspan=2 colspan=1>PeLK-S [4]RepLKNet+CGS-B (Ours)</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>83.9</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>80.0</td></tr><tr><td rowspan=7 colspan=1>DeiT-B/16 [50]RepLKNet-31B [15]Swin-B [37]ConvNeXt-B [38]SLaK-B [35]PeLK-B [4]RepLKNet+CGS-L (Ours)</td><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=1>81.8</td></tr><tr><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>83.5</td></tr><tr><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>83.5</td></tr><tr><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>83.8</td></tr><tr><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>84.0</td></tr><tr><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>84.2</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>82.1</td></tr></table>

4.2.2 Object Detection. We evaluate object detection performance on COCO [33] using Cascade Mask R-CNN [2, 21, 46] with a RepLKNet backbone. Following standard practice in MMDetection [5] and ConvNeXt [38], we use the AdamW optimizer with multi-scale training over 36 epochs.

Integrating the CGS-B strategy yields consistent gains under comparable parameter budgets (Tab. 3). In comparisons against SLaK [35] and ConvNeXt [38] (Tab. 4), CGS achieves competitive detection accuracy while reducing parameters substantially, with a reduction of over 47% for RepLKNet-31B. These results demonstrate that our lightweight sharing mechanism maintains strong detection capability while significantly improving model eficiency.

Table 3: MS COCO object detection performance. The CGS-B approach maintains competitive accuracy at comparable model complexity.
<table><tr><td>Method</td><td>Params (M)</td><td> $\mathrm { A P } ^ { \mathrm { b o x } }$ </td><td> $\mathrm { A P ^ { m a s k } }$ </td></tr><tr><td>ResNet101 [22]</td><td>63</td><td>40.4</td><td>36.4</td></tr><tr><td>ResNeXt101-32x4d [56]</td><td>63</td><td>41.9</td><td>37.5</td></tr><tr><td>PVT-Medium [52]</td><td>64</td><td>42.0</td><td>39.0</td></tr><tr><td>Swin-T [37]</td><td>86</td><td>50.5</td><td>43.7</td></tr><tr><td>ConvNeXt-T [38]</td><td>86</td><td>50.4</td><td>43.7</td></tr><tr><td>PeLK-T [4]</td><td>86</td><td>51.4</td><td>44.6</td></tr><tr><td>RepLKNet+CGS-B (Ours)</td><td>72</td><td>49.8</td><td>43.5</td></tr></table>

Table 4: MS COCO object detection performance. CGS-B strikes a competitive balance between eficiency and accuracy among the evaluated methods.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Params(M)</td><td rowspan=1 colspan=1> $\mathrm { A P } ^ { \mathrm { b o x } }$ </td><td rowspan=1 colspan=1> $\mathrm { A P ^ { m a s k } }$ </td></tr><tr><td rowspan=3 colspan=1>Swin-S [37]ConvNeXt-S [38]PeLK-S [4]</td><td rowspan=1 colspan=1>107</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>44.7</td></tr><tr><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>51.9</td><td rowspan=1 colspan=1>45.0</td></tr><tr><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>52.2</td><td rowspan=1 colspan=1>45.3</td></tr><tr><td rowspan=6 colspan=1>Swin-B [37]RepLKNet-31B [15]SLaK-B [35]ConvNeXt-B [38]PeLK-B [4]RepLKNet+CGS-B (Ours)</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>51.9</td><td rowspan=1 colspan=1>45.0</td></tr><tr><td rowspan=1 colspan=1>137</td><td rowspan=1 colspan=1>52.2</td><td rowspan=1 colspan=1>45.2</td></tr><tr><td rowspan=1 colspan=1>152</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>45.5</td></tr><tr><td rowspan=1 colspan=1>146</td><td rowspan=1 colspan=1>52.7</td><td rowspan=1 colspan=1>45.6</td></tr><tr><td rowspan=1 colspan=1>147</td><td rowspan=1 colspan=1>52.9</td><td rowspan=1 colspan=1>45.9</td></tr><tr><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>49.8</td><td rowspan=1 colspan=1>43.5</td></tr></table>

4.2.3 Semantic Segmentation. We evaluate semantic segmentation performance on the ADE20K dataset using UperNet [54] with a RepLKNet backbone pre-trained on ImageNet-1K and fine-tuned for 160K iterations. As shown in Tab. 5, our parameter-sharing strategy strikes an empirically favorable balance: it maintains competitive singlescale mean Intersection-over-Union (mIoU) while reducing parameters by 57% compared to the original RepLKNet-31B. This demonstrates that the approach preserves the large-kernel advantage in global receptive fields while significantly improving eficiency for dense prediction tasks.

## 4.3 Ablation Studies

This section conducts systematic ablation studies to validate our design choices and examine the impact of key hyperparameters.

4.3.1 Rationalefor Stage-Specific Compression. Restricting CGS compression to Stages 3-4 is a deliberate design choice. As shown in Fig. 3, these stages contain approximately 97% of the model parameters. In contrast, Stages 1-2 extract lowlevel features and are highly sensitive to compression. Ablation results in Tab. 6 confirm that, compared to compressing only Stages 3-4 which preserves accuracy, applying CGS-B to all stages of RepLKNet leads to a relative accuracy reduction of 1.6% while saving merely 1.3M parameters, indicating that concentrating compression on the parameterdense later stages yields an empirically favorable trade-of between eficiency and accuracy.

Table 5: ADE20K semantic segmentation performance. CGS-B achieves an empirically favorable accuracyparameter eficiency trade-of.
<table><tr><td>Method</td><td>Params (M) |</td><td>mIoU (%)</td></tr><tr><td>Swin-T [37]</td><td>60</td><td>44.5</td></tr><tr><td>ConvNeXt-T [38]</td><td>60</td><td>46.0</td></tr><tr><td>SLaK-T [35]</td><td>64</td><td>47.6</td></tr><tr><td>PeLK-T [4]</td><td>62</td><td>48.1</td></tr><tr><td>Swin-S [37]</td><td>81</td><td>47.6</td></tr><tr><td>ConvNeXt-S [38]</td><td>82</td><td>48.7</td></tr><tr><td>SLaK-S [35]</td><td>89</td><td>49.4</td></tr><tr><td>PeLK-S [4]</td><td>84</td><td>49.7</td></tr><tr><td>Swin-B [37]</td><td>121</td><td>48.1</td></tr><tr><td>ConvNeXt-B [38]</td><td>122</td><td>49.1</td></tr><tr><td>RepLKNet-31B [15]</td><td>112</td><td>49.9</td></tr><tr><td>SLaK-B [35]</td><td>131</td><td>50.2</td></tr><tr><td>PeLK-B [4]</td><td>126</td><td>50.4</td></tr><tr><td>RepLKNet+CGS-B (Ours)</td><td>48</td><td>45.3</td></tr></table>

Table 6: Comparative evaluation of classification performance on ImageNet-1K using CGS across diferent stage ranges.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Satge |</td><td rowspan=1 colspan=1>Params (M) |</td><td rowspan=1 colspan=1>Top-1 Acc</td></tr><tr><td rowspan=2 colspan=1>RepLKNet+CGS-B</td><td rowspan=2 colspan=1>3-41-4</td><td rowspan=1 colspan=1>14.7</td><td rowspan=1 colspan=1>80.0</td></tr><tr><td rowspan=1 colspan=1>13.4</td><td rowspan=1 colspan=1>78.7</td></tr></table>

4.3.2 Analysis ofBotleneck Dimension and Group Count �. We conduct a systematic study on CIFAR-100 to evaluate the impact of two key hyperparameters in CGS: the botleneck dimension � of the down/up-projection matrices and the group count �.

Using RepLKNet as the backbone, we test botleneck dimensions $R \in \{ 5 0 , 1 0 0 , 2 0 0 , 3 0 0 \}$ across the $\mathrm { C G S } { \cdot } { \cal S } , { \cdot } { \mathrm { B } } ,$ , and -L variants. Fig. 8 shows that � = 200 achieves an empirically favorable balance between model complexity and accuracy for all variants. Subsequently, with � fixed at 200, we ablate the group count � relative to the automatically determined $k _ { \mathrm { a u t o } }$ . As shown in Tab. 7, performance degrades when � deviates from $k _ { \mathrm { a u t o } } . \mathrm { A }$ smaller � limits the adaptation flexibility because each shared transformation must model a larger portion of the pointwise weight matrix. In contrast, a large � results in many fine-grained submatrices, reducing the structural information contained within each partition. Consequently, sharing the same transformation across these fragmented submatrices becomes less efective, limiting the quality of the learned weight updates and ultimately degrading accuracy. Therefore, $k _ { \mathrm { a u t o } }$ provides a robust empirical trade-of between adaptation flexibility and parameter sharing.

![](images/38a8825e12fc12fbaf115bdcf0881f37ed7e397a75f362d2fc772533ff546b3d.jpg)  
Figure 8: Systematic evaluation ofdimensionality scaling efects in down/up-projection matrices on CGS performance for CIFAR-100 datasets. Detailed results are presented in Appendix B.

Table 7: Systematic evaluation of Group Count (�) in down/up-projection matrices on CGS performance for CIFAR-100 datasets.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>k relative</td><td rowspan=1 colspan=1>Params(M)</td><td rowspan=1 colspan=1>Top-1Acc</td></tr><tr><td rowspan=2 colspan=1>RepLkNet+CGS-S</td><td rowspan=2 colspan=1> $0 . 5 k _ { \mathrm { a u t o } }$  $k _ { \mathrm { a u t o } }$  $2 k _ { \mathrm { a u t o } }$ </td><td rowspan=2 colspan=1>17.813.211.0</td><td rowspan=1 colspan=1>78.2</td></tr><tr><td rowspan=1 colspan=1>79.176.1</td></tr><tr><td rowspan=3 colspan=1>RepLkNet+CGS-B</td><td rowspan=3 colspan=1> $0 . 5 k _ { \mathrm { a u t o } }$  $k _ { \mathrm { a u t o } }$  $2 k _ { \mathrm { a u t o } }$ </td><td rowspan=1 colspan=1>18.2</td><td rowspan=1 colspan=1>78.2</td></tr><tr><td rowspan=2 colspan=1>13.711.6</td><td rowspan=1 colspan=1>80.3</td></tr><tr><td rowspan=1 colspan=1>76.4</td></tr><tr><td rowspan=3 colspan=1>RepLkNet+CGS-L</td><td rowspan=2 colspan=1> $0 . 5 k _ { \mathrm { a u t o } }$  $k _ { \mathrm { a u t o } }$ </td><td rowspan=1 colspan=1>20.6</td><td rowspan=1 colspan=1>78.9</td></tr><tr><td rowspan=1 colspan=1>16.2</td><td rowspan=1 colspan=1>81.6</td></tr><tr><td rowspan=1 colspan=1> $2 k _ { \mathrm { a u t o } }$ </td><td rowspan=1 colspan=1>14.2</td><td rowspan=1 colspan=1>76.3</td></tr></table>

4.3.3 Cross-Architecture Generalization. To evaluate the generality of CGS within its target application domain, we deploy the CGS-B variant on three representative largekernel CNN backbones: RepLKNet [15], ConvNeXt [38], and SLaK [35]. Tab. 8 shows that CGS-B achieves an empirically favorable accuracy-parameter trade-of, compressing parameters by >77% with accuracy drops below 6.5% across all evaluated architectures. These results demonstrate that CGS generalizes well across diverse large-kernel CNN architectures, supporting its efectiveness within the target deployment seting.

Table 8: ImageNet-1K classification performance. RepLKNet, ConvNeXt, and SLaK backbones integrated with the CGS-B compression.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Params (M) |</td><td rowspan=1 colspan=1>Top-1 Acc</td></tr><tr><td rowspan=1 colspan=1>RepLKNet-31B [15]RepLKNet+CGS-B (Ours)</td><td rowspan=1 colspan=1>7915</td><td rowspan=1 colspan=1>83.580.0</td></tr><tr><td rowspan=1 colspan=1>ConvNeXt-B [38]ConvNeXt+CGS-B (Ours)</td><td rowspan=1 colspan=1>8920</td><td rowspan=1 colspan=1>83.878.5</td></tr><tr><td rowspan=2 colspan=1>SLaK-B [35]SLaK+CGS-B (Ours)</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>84.0</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>79.8</td></tr></table>

![](images/d44646c9ebc406bc8c99ea1633b6c33a5a7c64c3a8d82ea17803d0eccc03c319.jpg)  
(a) Accuracy vs. Latency

![](images/3ea6751e49687a4694ade874c0939e06f7cb6b53b8fd90131bd3c8505ac222f3.jpg)  
(b) Accuracy vs. Energy  
Figure 9: Performance comparison on a Snapdragon 8 Gen 3 mobile platform.

4.3.4 Ablation on Structural and Quantization Synergy. Fig. 9 compares on-device eficiency. CGS-B cuts latency by 25% and energy by 43% over the INT8 baseline, with a moderate accuracy drop. Combined with INT8, CGS-B+INT8 achieves 135 ms latency and 7.8 mWh energy while incurring only a 0.5% relative accuracy drop, showing strong structural-quantization synergy. In contrast, ADMM pruning, despite fewer parameters, yields higher latency and energy due to irregular sparsity. Detailed results are in Appendix C.

## 4.4 Mobile Devices Performance

This section evaluates the deployment of pre-trained largekernel foundation models on resource-constrained mobile devices, assessing the practicality of our lightweight CGS approach. The specific devices and processors used are detailed in Appendix D.

4.4.1 Memory Eficiency Benchmark. We integrate CGS into several large-kernel convolutional models and deploy them on various mobile devices. Tab. 9 and Fig. 10 report key metrics including Random Access Memory (RAM) usage, initialization memory, model loading time, and inference FLOPs. Our method significantly reduces model loading time compared to alternatives, while maintaining Floating point Operations (FLOPs) only slightly higher than other methods. This demonstrates an efective trade-of that prioritizes storage and loading eficiency, which are often critical botlenecks in mobile deployment, with low computational overhead.

Table 9: Comparison of deployment performance of diferent large-kernel convolution models on various mobile devices, including RAM and loading time.
<table><tr><td rowspan=1 colspan=1>Device</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>RAM(M)</td><td rowspan=1 colspan=1>LoadingTime (ms)</td></tr><tr><td rowspan=1 colspan=1>GooglePixel 2(Emulator)</td><td rowspan=1 colspan=1>ConvNeXt-B [38]RepLKNet-31B [15]SLaK-B [35]RepLKNet+CGS-B</td><td rowspan=1 colspan=1>372.17344.75641.0896.58</td><td rowspan=1 colspan=1>154113133720591</td></tr><tr><td rowspan=1 colspan=1>GalaxyNexus(Emulator)</td><td rowspan=1 colspan=1>ConvNeXt-B [38]RepLKNet-31B [15]SLaK-B [35]RepLKNet+CGS-B</td><td rowspan=1 colspan=1>372.67345.63633.4296.95</td><td rowspan=1 colspan=1>251826394443609</td></tr><tr><td rowspan=2 colspan=1>HUAWEIP20 pro</td><td rowspan=2 colspan=1>ConvNeXt-B [38]RepLKNet-31B [15]SLaK-B [35]RepLKNet+CGS-B</td><td rowspan=2 colspan=1>385.84360.17666.80112.68</td><td rowspan=1 colspan=1>28543383</td></tr><tr><td rowspan=1 colspan=1>51671482</td></tr><tr><td rowspan=3 colspan=1>HUAWEInova 8</td><td rowspan=3 colspan=1>ConvNeXt-B [38]RepLKNet-31B [15]SLaK-B [35]RepLKNet+CGS-B</td><td rowspan=3 colspan=1>399.63370.16666.96125.14</td><td rowspan=1 colspan=1>1725</td></tr><tr><td rowspan=1 colspan=1>19123019</td></tr><tr><td rowspan=1 colspan=1>918</td></tr><tr><td rowspan=1 colspan=1>Honor200 Pro</td><td rowspan=1 colspan=1>ConvNeXt-B [38]RepLKNet-31B [15]SLaK-B [35]RepLKNet+CGS-B</td><td rowspan=1 colspan=1>399.67377.50673.13120.18</td><td rowspan=1 colspan=1>176116133012624</td></tr><tr><td rowspan=4 colspan=1>OPPOReno4</td><td rowspan=4 colspan=1>ConvNeXt-B [38]RepLKNet-31B [15]SLaK-B [35]RepLKNet+CGS-B</td><td rowspan=2 colspan=1>394.02372.26</td><td rowspan=1 colspan=1>1855</td></tr><tr><td rowspan=1 colspan=1>2080</td></tr><tr><td rowspan=2 colspan=1>668.22125.82</td><td rowspan=1 colspan=1>3197</td></tr><tr><td rowspan=1 colspan=1>1004</td></tr></table>

![](images/e39cbcb6a9948fc468e80bc46680366096a19ee17b37010c32aaa75fc8d15764.jpg)  
Figure 10: Comparison of memory footprint during model initialization and total FLOPs for inference across diferent models on mobile devices.

Table 10: Mobile deployment performance. Latency, energy consumption, memory footprint, and FPS are reported for diferent models.
<table><tr><td rowspan=1 colspan=1>Device</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Lat.(ms)</td><td rowspan=1 colspan=1>Energy(mWh)</td><td rowspan=1 colspan=1>|Mem.(MB)</td><td rowspan=1 colspan=1>FPS</td></tr><tr><td rowspan=2 colspan=1>OnePlus13</td><td rowspan=2 colspan=1>|ConvNeXt-T [38]|[RepLKNet-B [15]]RepLKNet-B+ CGS-B (Ours)</td><td rowspan=2 colspan=1>215 ± 5|242 ± 6|198 ± 51</td><td rowspan=1 colspan=1>13.8 ± 0.3|</td><td rowspan=2 colspan=1>141338115</td><td rowspan=2 colspan=1>4.74.15.0</td></tr><tr><td rowspan=1 colspan=1>16.3 ± 0.42.2 ± 0.3</td></tr><tr><td rowspan=2 colspan=1>Xiaomi14 Ultra</td><td rowspan=2 colspan=1>|ConvNeXt-T [38]|[RepLKNet-B [15][2RepLKNet-B+ CGS-B (Ours)</td><td rowspan=1 colspan=1>208 ± 4|</td><td rowspan=1 colspan=1>13.4 ± 0.3|</td><td rowspan=1 colspan=1>145</td><td rowspan=2 colspan=1>4.84.35.3</td></tr><tr><td rowspan=1 colspan=1>235 ± 5|189 ± 41</td><td rowspan=1 colspan=1>15.7 ± 0.41.8 ± 0.3</td><td rowspan=1 colspan=1>345118</td></tr><tr><td rowspan=2 colspan=1>vivoX300 Pro</td><td rowspan=2 colspan=1>[ConvNeXt-T [38]|[RepLKNet-B [15]]RepLKNet-B+ CGS-B (Ours)</td><td rowspan=2 colspan=1>205 ± 4|228 ± 5|1186 ± 4</td><td rowspan=1 colspan=1>13.0 ± 0.2|</td><td rowspan=2 colspan=1>150392121</td><td rowspan=2 colspan=1>5.14.45.4</td></tr><tr><td rowspan=1 colspan=1>5.1 ± 0.3| $1 1 . 3 \pm 0 . 2 |$ 1</td></tr></table>

4.4.2 On-Device Latency and Energy Profiling. We conduct device-level measurements on three representative Android devices with diferent mainstream processors. Results in Tab. 10 show that CGS-B achieves lower inference latency despite higher FLOPs. Wireless evaluation results are reported as the average of multiple independent trials. This counter-intuitive result stems from the memory-bound nature of modern mobile Neural Processing Units (NPUs), where data movement, not arithmetic, is the primary botleneck [6]. CGS addresses this via four key optimizations: (1) compressing pointwise convolutions to reduce model footprint and weight transfer; (2) sharing projection matrices to minimize random memory access and improve cache eficiency; (3) employing a low-rank representation to reduce activation bandwidth; and (4) restructuring operations into larger, more regular GEMM tasks for beter NPU acceleration. The per-run latency variation remained below 5%, confirming stable performance across hardware.

4.4.3 Wireless Ofloading Performance Analysis. To evaluate CGS in edge-computing scenarios, we test end-to-end performance under a wireless “device-to-cloud” ofloading setup, comparing 5G and Wi-Fi 6 networks, with ConvNeXt-T and RepLKNet-B as baselines. The total latency includes model load, deserialization, inference, and the associated communication overhead. As shown in Tab. 11, wireless communication consistently contributes 54%–62% of the overall latency across all evaluated models. This observation suggests that, in the considered deployment scenario, reducing model size has a greater impact on end-to-end latency than reducing FLOPs alone.

Table 11: The evaluation of end-to-end wireless execution was conducted under two distinct network condi tions: 5G connectivity was tested on a Xiaomi 14 Ultra device, while Wi-Fi 6 performance was measured on a OnePlus 13 device. The column “Ofload (ms)” denotes on-device computation latency.
<table><tr><td rowspan=1 colspan=1>Network</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Size(MB)</td><td rowspan=1 colspan=1>|Offload(ms)</td><td rowspan=1 colspan=1>| Comm.(MB)</td><td rowspan=1 colspan=1>|Total(ms)</td></tr><tr><td rowspan=2 colspan=1>5G</td><td rowspan=2 colspan=1>ConvNeXt-T [38]RepLKNet-B [15]RepLKNet-B+ CGS-B (Ours)</td><td rowspan=1 colspan=1>109.30</td><td rowspan=1 colspan=1>145</td><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>355</td></tr><tr><td rowspan=1 colspan=1>306.9458.41</td><td rowspan=1 colspan=1>178116</td><td rowspan=1 colspan=1>287172</td><td rowspan=1 colspan=1>420305</td></tr><tr><td rowspan=2 colspan=1>Wi-Fi 6</td><td rowspan=2 colspan=1>[ConvNeXt-T [38]RepLKNet-B [15]RepLKNet-B+ CGS-B (Ours)</td><td rowspan=1 colspan=1>109.30</td><td rowspan=1 colspan=1>170</td><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1>380</td></tr><tr><td rowspan=1 colspan=1>306.9458.41</td><td rowspan=1 colspan=1>203142</td><td rowspan=1 colspan=1>287172</td><td rowspan=1 colspan=1>438330</td></tr></table>

CGS compresses RepLKNet-B [15] from 306.94 MB to 58.41 MB, reducing transmited data from 287 MB to 172 MB, and cuts end-to-end latency by 27% on 5G and 25% on Wi-Fi 6. It also outperforms lightweight ConvNeXt-T [38], with 46.5% smaller model size and 13-14% lower latency. This confirms that CGS efectively mitigates the highcommunication-latency botleneck, making it suitable for eficient edge-to-cloud collaborative inference.

## 5 Related Works

## 5.1 Large-Kernel Convolutions

The evolution of Convolutional Neural Networks (CNNs) [1, 3, 25, 30, 31, 45, 53] has witnessed a paradigm shift from small kernels (e.g., 3 × 3 or 5 × 5) towards large-kernel convolutions with kernel sizes of 7 × 7 or larger [4, 15, 35, 38]. This shift is driven by the need to model long-range dependencies in high-dimensional vision data. Large-kernel convolutions expand the receptive field, capture broader contextual information, and reduce depth redundancy compared to stacks of small kernels. Pioneering architectures like ConvNeXt [38] and RepLKNet [15] demonstrate that kernels as large as 31 × 31 can substantially boost performance across diverse vision tasks. However, the quadratic parameter growth relative to kernel size presents significant deployment challenges on resource-constrained devices.

Parameter Compression in Large-Kernel CNNs. To mitigate the parameter explosion in large-kernel convolutions, SOTA eficient architectures predominantly adopt depthwise separable convolution as their backbone, comprising depthwise convolution and pointwise convolution. Within this framework, significant research efort has focused on compressing the depthwise convolution component. For in stance, SLaK [35] employs kernel decomposition (e.g., 51 × $5 + 5 \times 5 1 )$ and dynamic sparsity to achieve lightweight kernels up to 51 × 51, efectively reducing computational overhead. Similarly, PeLK [4] utilizes sophisticated parameter sharing strategies, specifically a “focus-blur” mechanism with exponentially scaled sharing grids and kernel positional embeddings, to support massive kernels up to 101 × 101 while achieving over 90% parameter reduction in the depthwise convolution. These innovations have suc cessfully made large-kernel models more practical for deployment by tackling the parameter burden of depthwise operations. However, these advances exclusively optimize the depthwise component. As shown in Fig. 1, they overlook the fact that pointwise convolution (1×1 convolution), responsible for crucial channel fusion, becomes the dominant parameter contributor in such models, accounting for over 87% of total parameters in typical large-kernel CNNs. This represents a major, yet largely unaddressed, barrier to storage-eficient deployment.

Parameter Sharing Strategies. Parameter sharing is a wellestablished paradigm for model compression across various architectures [16], with specialized implementations emerging for large-kernel CNNs. PeLK [4], as mentioned, represents a significant advancement in this domain for depthwise convolution. Its core contribution is a peripheral convolution framework combined with aggressive parameter sharing. By concentrating unique parameters in the kernel center (“focus”) and sharing parameters extensively in the periphery (“blur”) via an exponentially scaled grid, PeLK [4] dramatically reduces depthwise convolution parameters while atempting to balance computational eficiency and spatial detail retention, validated on benchmarks like ImageNet-1K classification [13] and semantic segmentation on ADE20K [67]. While highly efective for depthwise compression, PeLK [4] and similar strategies do not target the pointwise convolution layer. Consequently, strategies for compressing the parameter-dominant pointwise convolution within large-kernel CNNs remain an open and critical research gap, which our work directly addresses.

## 5.2 Model Compression and Deployment on Mobile Devices

The deployment of deep learning models on resourceconstrained mobile devices presents unique challenges that necessitate specialized compression techniques. Traditional model compression methods have evolved to address the stringent memory, computational, and energy constraints inherent to edge devices, with particular emphasis on maintaining inference accuracy while drastically reducing model footprint [44].

Quantization has emerged as a foundational technique for mobile deployment, where 8-bit integer quantization has been shown to achieve near-floating-point accuracy while reducing model size and accelerating inference through specialized hardware instructions [26]. Eficient architectures such as MobileNetV2 [42] further demonstrate the synergy between lightweight design and quantization.

Pruning methodologies have similarly undergone significant refinement for mobile scenarios. Structured pruning techniques [36] gained prominence by eliminating entire convolutional filters or channels, thereby producing hardware-friendly models that circumvent the computational ineficiencies associated with sparse matrix operations in unstructured pruning. The synergy between pruning and quantization was particularly impactful, as demonstrated by frameworks that jointly optimized both techniques to achieve 10-20x compression ratios without catastrophic accuracy drops [20].

The practical deployment of these compressed models has been facilitated by mobile-optimized inference engines such as TensorFlow Lite [12] and Open Neural Network Exchange (ONNX) Runtime [14]. These frameworks implement critical optimizations including operator fusion, memory-eficient tensor layouts, and hardware-specific kernel implementations, which collectively reduce inference latency by 3-5x compared to naive deployments. Moreover, they provide standardized pathways for deploying models compressed via diverse methodologies, thereby bridging the gap between algorithmic advances and practical implementation [27].

Our CGS method advances this field through a mathematically grounded compression strategy specifically designed for the parameter structure of large-kernel CNNs. Unlike general-purpose techniques, CGS directly targets the parameter-dominant pointwise convolution via an SVDinspired group-sharing mechanism. This focused approach achieves high compression ratios (e.g., >81% parameter reduction for RepLKNet-31B) while preserving the structural regularity required for eficient mobile inference. Consequently, CGS simultaneously alleviates the critical mobile deployment botlenecks: stringent storage limits and high memory bandwidth pressure during model loading.

## 6 Discussion

## 6.1 Core Design Principles

The CGS framework is built upon globally shared down/upprojection matrices across all channel groups, enhancing representational consistency and parameter eficiency. This design enforces geometric uniformity in feature maps and reduces the parameter footprint. To capture group-specific variations without sacrificing eficiency, a learnable diagonal scaling matrix is incorporated per channel group. This lightweight component fine-tunes the shared bases, introducing only linear parameter overhead and serving as an efficient alternative to full independent projections per group.

## 6.2 Training Stability and Deployment Eficiency

A notable advantage of CGS lies in its stable optimization dynamics. This property stems from three intrinsic characteristics: the low-rank botleneck limits basis redundancy, gradient averaging across channel groups mitigates directional noise, and the projection matrices gradually adapt to dominant feature subspaces during training.

From an engineering perspective, the shared architecture is naturally compatible with INT8 quantization, as the shared projections only need to be quantized once and can be reused across groups, drastically reducing overhead and enabling eficient weight reuse on NPUs. The lightweight per-group scaling matrices further preserve accuracy by mitigating quantization error.

Consequently, CGS unifies eficient architectural design, stable training behavior, and hardware-friendly deployment into a practical framework for edge computing.

## 6.3 Future Directions

Several promising directions remain for future research. First, although this work is designed and evaluated for pointwise convolutions in large-kernel CNNs, extending the proposed group-shared low-rank approximation strategy to other matrix-based components, such as Transformer feedforward networks and atention projections, warrants further investigation.

Second, CGS currently focuses on later network stages, where most parameters reside in the large-kernel CNNs considered in this work. Adapting the method to architectures with more parameter-intensive early layers is another promising direction.

Finally, this work focuses on the widely adopted INT8 quantization seting. Exploring more aggressive quantization schemes, such as INT4 or mixed-precision quantization, and their compatibility with CGS may further improve deployment eficiency.

## 7 Conclusions

This work presents Channel Group-Shared (CGS) low-rank approximation, a novel parameter compression strategy grounded in Singular Value Decomposition (SVD) theory, which efectively resolves critical deployment constraints in modern large-kernel CNNs. Our systematic analysis reveals that pointwise convolutions account for over 87% of parameters in leading architectures including RepLKNet-31B [15] and ConvNeXt-B [38]. This severe parametric imbalance constitutes the fundamental barrier to edge deployment, motivating our development of a geometrically interpretable compression framework. Our method decomposes convolutional kernels into two isomorphic components: (1) shared down/up-projection matrices capturing cross-channel correlations across layer-wide channel groups, analogous to unitary matrices in SVD factorization; (2) lightweight group-specific diagonal scaling matrices preserving channel-adaptive feature transformations, functionally equivalent to singular values.

This design achieves breakthrough parameter eficiency, significantly reducing storage overhead and peak memory load during model initialization, while simultaneously enhancing execution eficiency on mobile devices. Consequently, it enables the successful deployment of largekernel backbones on resource-constrained mobile devices, overcoming the dual constraints of storage and runtime memory that have hindered edge deployment, while establishing a new paradigm for lightweight network design.

## 8 Acknowledgements

W. Dong’s participation was in part supported by the National Natural Science Foundation of China (General Program) (Grant No. 62576267), the Major Science and Technology Innovation Project of Xianyang (Program No. L2025- ZDKJ-ZDGG-ZYRGZN-002), and the Major Science and Technology Innovation Project of Xianyang (Program No. L2024-ZDKJ-ZDGG-GY-0010).

## References

[1] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V Le. 2019. Atention augmented convolutional networks. In Proceedings of the IEEE/CVF international conference on computer vision. 3286–3295.

[2] Zhaowei Cai and Nuno Vasconcelos. 2018. Cascade r-cnn: Delving into high quality object detection. In Proceedings of the IEEE conference on computer vision and patern recognition. 6154–6162.

[3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. 2020. End-to-end object detection with transformers. In European conference on computer vision. Springer, 213–229.

[4] Honghao Chen, Xiangxiang Chu, Yongjian Ren, Xin Zhao, and Kaiqi Huang. 2024. Pelk: Parameter-eficient large kernel convnets with peripheral convolution. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 5557–5567.

[5] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. 2019. MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155 (2019).

[6] Tianshi Chen, Zidong Du, Ninghui Sun, Jia Wang, Chengyong Wu, Yunji Chen, and Olivier Temam. 2014. Diannao: A small-footprint high-throughput accelerator for ubiquitous machine-learning. ACM SIGARCH Computer Architecture News 42, 1 (2014), 269–284.

[7] Bowen Cheng, Alex Schwing, and Alexander Kirillov. 2021. Per-pixel classification is not all you need for semantic segmentation. Advances in neural information processing systems 34 (2021), 17864–17875.

[8] François Chollet. 2017. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and patern recognition. 1251–1258.

[9] MMSegmentation Contributors. 2020. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark.

[10] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. 2020. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings ofthe IEEE/CVF conference on computer vision and patern recognition workshops. 702–703.

[11] Xiyang Dai, Yinpeng Chen, Bin Xiao, Dongdong Chen, Mengchen Liu, Lu Yuan, and Lei Zhang. 2021. Dynamic head: Unifying object detection heads with atentions. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 7373–7382.

[12] Robert David, Jared Duke, Advait Jain, Vijay Janapa Reddi, Nat Jeffries, Jian Li, Nick Kreeger, Ian Nappier, Meghna Natraj, Tiezhen Wang, Pete Warden, and Rocky Rhodes. 2021. TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. In Proceedings of Machine Learning and Systems 3 (MLSys).

[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and patern recognition. Ieee, 248– 255.

[14] ONNX Runtime developers. 2021. ONNX Runtime. https:// onnxruntime.ai/. Version: x.y.z.

[15] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. 2022. Scaling up your kernels to 31x31: Revisiting large kernel de sign in cnns. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 11963–11975.

[16] Wei Dong, Dawei Yan, Zhijun Lin, and Peng Wang. 2023. Eficient adaptation of large vision transformer via adapter re-composing. Advances in Neural Information Processing Systems 36 (2023), 52548– 52567.

[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Mathias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An im age is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[18] Juan Carlos García-Ardila, Luis E Garza, and Francisco Marcellán. 2018. A canonical Geronimus transformation for matrix orthogonal polynomials. Linear and Multilinear Algebra 66, 2 (2018), 357–381.

[19] Ted Hagos. 2018. Android studio. In Learn Android Studio 3: Eficient Android App Development. Springer, 5–17.

[20] Song Han, Huizi Mao, and William J. Dally. 2016. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Hufman Coding. In International Conference on Learning Representations (ICLR). arXiv:1510.00149 [cs.CV] Published as a conference paper at ICLR 2016 (oral), arXiv:1510.00149 [cs.CV].

[21] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. 2017. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision. 2961–2969.

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and patern recognition. 770–778.

[23] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreeto, and Hartwig Adam. 2017. Mobilenets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017).

[24] Binh-Son Hua, Minh-Khoi Tran, and Sai-Kit Yeung. 2018. Pointwise convolutional neural networks. In Proceedings ofthe IEEE conference on computer vision and patern recognition. 984–993.

[25] Zilong Huang, Xinggang Wang, Lichao Huang, Chang Huang, Yunchao Wei, and Wenyu Liu. 2019. Ccnet: Criss-cross atention for semantic segmentation. In Proceedings ofthe IEEE/CVFinternational conference on computer vision. 603–612.

[26] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Mathew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and Training of Neural Networks for Eficient Integer-Arithmetic-Only Inference. In Proceedings ofthe IEEE Conference on Computer Vision and Patern Recognition (CVPR). 2704–2713.

[27] Xiaotang Jiang, Huan Wang, Yiliu Chen, Ziqi Wu, Lichuan Wang, Bin Zou, Yafeng Yang, Zongyang Cui, Yu Cai, Tianhang Yu, Chengfei Lyu, and Zhihua Wu. 2020. MNN: A Universal and Eficient Inference En gine. In Proceedings of Machine Learning and Systems 2 (MLSys).

[28] Alex Krizhevsky, Geofrey Hinton, et al. 2009. Learning multiple layers of features from tiny images. (2009).

[29] Liangzhen Lai and Naveen Suda. 2018. Rethinking machine learning development and deployment for edge devices. arXiv preprint arXiv:1806.07846 (2018).

[30] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. 1989. Backpropagation applied to handwriten zip code recognition. Neural computation 1, 4 (1989), 541–551.

[31] Yann LeCun, Léon Botou, Yoshua Bengio, and Patrick Hafner. 2002. Gradient-based learning applied to document recognition. Proc. IEEE 86, 11 (2002), 2278–2324.

[32] Sangho Lee, Youngjae Yu, Gunhee Kim, Thomas Breuel, Jan Kautz, and Yale Song. 2020. Parameter eficient multimodal transformers for video representation learning. arXiv preprint arXiv:2012.04124 (2020).

[33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European conference on computer vision. Springer, 740–755.

[34] Hanxiao Liu, Zihang Dai, David So, and Quoc V Le. 2021. Pay atention to mlps. Advances in neural information processing systems 34 (2021), 9204–9215.

[35] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Xuxi Chen, Qiao Xiao, Boqian Wu, Tommi Kärkkäinen, Mykola Pechenizkiy, Decebal Mocanu, and Zhangyang Wang. 2022. More convnets in the 2020s: Scaling up kernels beyond 51x51 using sparsity. arXiv preprint arXiv:2207.03620 (2022).

[36] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. 2017. Learning Eficient Convolutional Networks Through Network Slimming. In Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2736–2744.

[37] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. Swin transformer: Hierarchi cal vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision. 10012–10022.

[38] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. 2022. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 11976–11986.

[39] Ilya Loshchilov and Frank Huter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[40] Juhong Min, Yucheng Zhao, Chong Luo, and Minsu Cho. 2022. Periph eral vision transformer. Advances in Neural Information Processing Systems 35 (2022), 32097–32111.

[41] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. 2020. Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 10428–10436.

[42] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. 2018. MobileNetV2: Inverted Residuals and Linear Botlenecks. In Proceedings ofthe IEEE Conference on Computer Vision and Patern Recognition (CVPR). 4510–4520.

[43] Joao Paulo Schwarz Schuler, Santiago Romani, Mohamed Abdel-Nasser, Hatem Rashwan, and Domenec Puig. 2022. Grouped pointwise convolutions reduce parameters in convolutional neural net works. In Mendel, Vol. 28. 23–31.

[44] Divyasheel Sharma and Santonu Sarkar. 2022. Enabling inference and training of deep learning models for ai applications on iot edge devices. In Artificial Intelligence-based Internet of Things Systems. Springer, 267–283.

[45] Fumin Shen, Chunhua Shen, Wei Liu, and Heng Tao Shen. 2015. Supervised Discrete Hashing. In IEEE Conference on ComputerVision and Patern Recognition. 37–45.

[46] Fumin Shen, Yan Xu, Li Liu, Yang Yang, Zi Huang, and Heng Tao Shen. 2018. Unsupervised Deep Hashing with Similarity-Adaptive and Discrete Optimization. IEEE Transactions on Patern Analysis and Machine Intelligence 40, 12 (2018), 3034–3044.

[47] Gilbert Strang. 2012. Linear algebra and its applications.

[48] Christian Szegedy, Vincent Vanhoucke, Sergey Iofe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and patern recognition. 2818–2826.

[49] Cheng Tai, Tong Xiao, Yi Zhang, Xiaogang Wang, et al. 2015. Convolutional neural networks with low-rank regularization. arXiv preprint arXiv:1511.06067 (2015).

[50] Hugo Touvron, Mathieu Cord, Mathijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. 2021. Training dataeficient image transformers & distillation through atention. In International conference on machine learning. PMLR, 10347–10357.

[51] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. 2021. Max-deeplab: End-to-end panoptic segmentation with mask transformers. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 5463–5474.

[52] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. 2021. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision. 568–578.

[53] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. 2022. Pvt v2: Improved baselines with pyramid vision transformer. Computational visual media 8, 3 (2022), 415–424.

[54] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. 2018. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision (ECCV). 418–434.

[55] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. 2021. SegFormer: Simple and eficient design for semantic segmentation with transformers. Advances in neural information processing systems 34 (2021), 12077–12090.

[56] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. 2017. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and patern recognition. 1492–1500.

[57] Fisher Yu and Vladlen Koltun. 2015. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122 (2015).

[58] Fisher Yu, Vladlen Koltun, and Thomas Funkhouser. 2017. Dilated residual networks. In Proceedings of the IEEE conference on computer vision and patern recognition. 472–480.

[59] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. 2022. Metaformer is actually what you need for vision. In Proceedings of the IEEE/CVF conference on computer vision and patern recognition. 10819–10829.

[60] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. 2021. Tokensto-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF international conference on computer vision. 558–567.

[61] Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, and Shuicheng Yan. 2022. Volo: Vision outlooker for visual recognition. IEEE transactions on patern analysis and machine intelligence 45, 5 (2022), 6575–6586.

[62] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. 2019. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision. 6023–6032.

[63] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2017. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412 (2017).

[64] Meng Zhang, Fei Liu, and Dongpeng Weng. 2023. Speeding-up and compression convolutional neural networks by low-rank decomposition without fine-tuning. Journal of Real-Time Image Processing 20, 4 (2023), 64.

[65] Tianyun Zhang, Shaokai Ye, Kaiqi Zhang, Jian Tang, Wujie Wen, Makan Fardad, and Yanzhi Wang. 2018. A systematic dnn weight pruning framework using alternating direction method of multipliers. In Proceedings ofthe European conference on computer vision (ECCV). 184–199.

[66] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. 2020. Random erasing data augmentation. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 34. 13001–13008.

[67] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. 2019. Semantic understanding of scenes through the ade20k dataset. International Journal ofComputer Vision 127 (2019), 302–321.

## A Training Configurations

## A.1 ImageNet-1K

To pretrain a classification model on the ImageNet-1K dataset, we employed 8 NVIDIA A800 GPUs and optimized the model using AdamW [39] with a weight decay of 0.05. The learning rate was set to 8e-3, accompanied by a 10- epoch linear warmup and a cosine decay schedule. We conducted experiments with 120 and 300 training epochs, where the 120-epoch training was primarily used to observe phenomena in the main and ablation studies. For data augmentation, we used the parameter combination RandAugment [10] ‘rand-m9-mstd0.5-inc1’, along with label smoothing [48] (coeficient = 0.1), Mixup [63] (� = 0.8), Cut-Mix [62] (� = 1.0), and random erasing [66] (probability p = 0.25). The layer scale was initialized to 1e-6, and the Exponential Moving Average (EMA) decay factor was configured as 0.9999.

## A.2 MS-COCO

We followed the training configuration used in RepLKNet [15], and fine-tuned the pre-trained model CGS-B via Cascade Mask R-CNN[2, 21] in MMdetection [5] with 36 epochs. The hyperparameters used in the object detection process are consistent with those in the Github repository of RepLKNet-pytorch.

## A.3 ADE20K

We followed the training configuration used in RepLKNet [15], and fine-tuned the pre-trained model CGS-B via UperNet [54] in MMSegmentation [9] with 160K iterations. The hyperparameters used in the semantic segmentation process are consistent with those in the Github repository of RepLKNet-pytorch.

## A.4 CIFAR-100

We retain the identical training configuration from ImageNet-1K. For ablation analysis, we conduct 200-epoch experiments to observe critical paterns.

## B Detailed Results on Bottleneck Dimensions

This section presents a comprehensive empirical evaluation of the down/up-projection matrices in CGS-S/B/L across varying botleneck dimensions on the CIFAR-100 [28] dataset, encompassing both model performance and parameter count. Tab. 12 systematically quantifies the impact of the botleneck dimension within the down/upprojection matrices on model performance under the CGS-S/B/L configurations. Experiments were conducted using RepLKNet [15] as the backbone architecture, with controlled comparisons across botleneck dimensions of {50, 100, 200, 300}.

Notably, configurations employing dimensionalities below this critical threshold of 200 exhibit a consistent and significant degradation in model eficacy, failing to achieve the necessary representational capacity. Conversely, increasing the dimensionality beyond 200 incurs substantial and unnec essary costs. This escalation not only induces a substantial increase in parameter overhead, exemplified by a 33.6% parameter count increase observed for the CGS-B configuration when moving from dimension 200 to 300, but also paradoxically leads to a measurable detriment in model accuracy. For instance, the same dimensional increase for CGS-B resulted in a notable accuracy reduction of 1.4%. This counterintuitive performance decline at higher dimensions suggests potential over-parameterization or optimization challenges within the compressed structure.

Table 12: CIFAR-100 classification accuracy (%) comparison. For CGS-S/B/L variants, we systematically evaluate the impact of bottleneck dimensions in down/up-projection matrices by varying their sizes, employing RepLKNet as the backbone architecture.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>BottleneckDimension</td><td rowspan=1 colspan=1>Params(M)</td><td rowspan=1 colspan=1>Top-1Acc</td></tr><tr><td rowspan=3 colspan=1>RepLkNet+CGS-S</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>73.1</td></tr><tr><td rowspan=2 colspan=1>100200300</td><td rowspan=1 colspan=1>8.7</td><td rowspan=1 colspan=1>77.3</td></tr><tr><td rowspan=1 colspan=1>13.217.7</td><td rowspan=1 colspan=1>79.178.8</td></tr><tr><td rowspan=4 colspan=1>RepLkNet+CGS-B</td><td rowspan=4 colspan=1>50100200300</td><td rowspan=1 colspan=1>6.9</td><td rowspan=1 colspan=1>78.4</td></tr><tr><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=1>80.1</td></tr><tr><td rowspan=1 colspan=1>13.7</td><td rowspan=1 colspan=1>80.3</td></tr><tr><td rowspan=1 colspan=1>18.3</td><td rowspan=1 colspan=1>79.2</td></tr><tr><td rowspan=4 colspan=1>RepLkNet+CGS-L</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>78.5</td></tr><tr><td rowspan=2 colspan=1>100200</td><td rowspan=1 colspan=1>11.6</td><td rowspan=1 colspan=1>81.3</td></tr><tr><td rowspan=1 colspan=1>16.2</td><td rowspan=1 colspan=1>81.6</td></tr><tr><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>20.7</td><td rowspan=1 colspan=1>82.1</td></tr></table>

Table 13: The processor models corresponding to the devices.
<table><tr><td>Device</td><td>Processor</td></tr><tr><td>Google Pixel 2 (Emulator) Galaxy Nexus (Emulator)</td><td>Snapdragon 835 OMAP4460</td></tr><tr><td>HUAWEI P20 pro</td><td>HiSilicon Kirin 970</td></tr><tr><td>HUAWEI nova 8 Honor 200 Pro</td><td>Hisilicon Kirin 985</td></tr><tr><td>OPPO Reno 4</td><td>Snapdragon 8s Gen 3</td></tr><tr><td></td><td>Snapdragon 765G</td></tr><tr><td>OnePlus 13</td><td>Snapdragon 8</td></tr><tr><td>Xiaomi 14 Ultra</td><td>Snapdragon 8 Gen3</td></tr><tr><td>vivo X300 Pro</td><td>Dimensity 9500</td></tr></table>

Consequently, the dimension of 200 is identified as the singular configuration that simultaneously maximizes accuracy while minimizing unnecessary parameter growth across the evaluated CGS variants.

Table 14: Performance comparison on RepLKNet-31B with ImageNet-1K evaluated on a Snapdragon 8 Gen 3 mobile platform.
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs (G)</td><td>Top-1 Acc.</td><td>Lat. (ms)</td><td>Energy (mWh)</td></tr><tr><td>INT8 Quant. [26]</td><td>79.0</td><td>15.6</td><td>82.9</td><td>265</td><td>21.5</td></tr><tr><td>ADMM Prune [65]</td><td>42.5</td><td>9.1</td><td>82.3</td><td>290</td><td>24.0</td></tr><tr><td>CGS-B (ours)</td><td>14.7</td><td>28.7</td><td>80.0</td><td>198</td><td>12.2</td></tr><tr><td>CGS-B + INT8 [26]</td><td>14.7</td><td>28.7</td><td>79.6</td><td>135</td><td>7.8</td></tr></table>

Furthermore, the parameter increase associated with our proposed CGS-L configuration is marginal compared to CGS-B and CGS-S (e.g., merely +2.5M parameters versus CGS-B at botleneck dimension 200). This lower parameter cost alleviates concerns regarding scalability when deploying our larger-scale approach.

## C Detailed Results on Structural and Quantization Synergy

Tab. 14 compares the real-world performance of various compression methods applied to RepLKNet-31B. The results confirm that CGS efectively optimizes large-kernel CNNs for resource-constrained deployment. Energy consumption denotes the total energy measured over multiple runs.

CGS-B achieves a superior balance ofparameters, latency, and energy eficiency compared to standard INT8 quantization and ADMM-based pruning. Although CGS-B has higher FLOPs, its inference latency is significantly lower. This gain stems from improved memory access paterns and beter hardware utilization, alleviating the memorybandwidth botleneck common in mobile NPUs/GPUs.

CGS-B reduces model size by over 80% versus the original backbone while maintaining competitive accuracy. In contrast, pruning often creates irregular sparsity, limiting practical speedups. When combined with INT8 quantization, CGS-B further reduces latency and energy by over 30% with lower accuracy loss, demonstrating strong compatibility with precision reduction.

In summary, CGS enables joint structural and numerical optimization, making it suitable for memory- and powerconstrained mobile platforms. These results underscore the value of hardware-aware structural redesign beyond conventional compression.

## D Mobile Processor Model

The mobile device processors we used to deploy models and perform inference are shown in Tab. 13.

## E Code

Our code is available at https://github.com/LforikCyzzz/ CGS\_code.