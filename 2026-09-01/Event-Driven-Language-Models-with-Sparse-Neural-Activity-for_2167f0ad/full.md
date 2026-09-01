# Event-Driven Language Models with Sparse Neural Activity for Neuromorphic Hardware

Simon Richter<sup>1</sup>, Ruhai Lin<sup>2</sup>, Jason Yik<sup>3</sup>, Taylor Kergan<sup>2</sup>, Rui-Jie Zhu<sup>2</sup>, Farshad Moradi<sup>4</sup>, and Jason Eshraghian<sup>2</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, Aarhus University, Aarhus, Denmark

<sup>2</sup>Department of Computer Science and Engineering, University of California, Santa Cruz, Santa Cruz, USA

<sup>3</sup>School of Engineering and Applied Sciences, Harvard University, Cambridge, USA

<sup>4</sup>Department of Electrical and Computer Engineering, University of Southern Denmark, Odense, Denmark {siri, rlin50, tkergan, ridger, jsn}@ucsc.edu, jyik@g.harvard.edu, moradi@sdu.dk

Abstract—Inference with transformer-based large language models (LLMs) is often limited by the memory-bound KV cache and quadratic attention cost. State-space models (SSMs) mitigate this through linear attention and fixed-size recurrent states, but their large dense linear projections remain computationally expensive even after quantization. We introduce a method that induces sparse neural activity in heavily quantized linearattention models with minimal performance loss. Activations below a per-projection trainable threshold (±∆) are nullified while preserving crucial outliers, achieving comparable performance to dense models with up to 4× fewer effective arithmetic operations. Targeting a multi-core, multi-chip neuromorphic platform, where event-driven execution converts unstructured sparsity into throughput at both the compute and communication levels, a capability GPU architectures fundamentally lack, we project up to 37× higher throughput and 16× lower power versus edge GPU inference of a comparable transformer-based model, and up to 5.4× improvements over the non-sparsified baseline. These results position sparse, quantized linear-attention models as a natural fit for deploying LLMs on event-driven multi-core platforms.

Index Terms—activation sparsity, linear-attention models, event-driven inference, neuromorphic multi-core systems, edgecomputing, embedded AI

## I. INTRODUCTION

The growing scale of Large Language Models (LLMs) presents significant challenges, driven primarily by the selfattention mechanism whose cost scales quadratically with sequence length. This bottleneck makes long-context applications prohibitively expensive in resource-constrained settings. Linear-attention models and State space models (SSMs) have emerged as a powerful alternative, replacing quadratic attention with a linear-time recurrent mechanism and achieving competitive results across diverse domains [1]–[4]. Nevertheless, the billions of floating-point operations (FLOPs) required by SSMs still impede their deployment on edge devices, where low latency and energy efficiency are critical.

Previous model compression techniques have focused on reducing either the number of weights or activations, primarily through pruning. Weight pruning permanently removes parameters from the model, while activation pruning targets the intermediate outputs during inference. Pruning can be unstructured, removing individual weights or activations, or structured, which eliminates entire channels or blocks. While structured pruning is easier to exploit on modern specialized GPUs, the lack of fine-grain control often leads to lower model performance compared to unstructured pruning [5]. Theoretically, unstructured activation sparsity is promising because zero-valued activations can dynamically eliminate entire rows from memory access and subsequent Matrix-Vector Multiplication (MVM) operations without arbitrary (i.e. blockwise or channel wise) constraints. However, realizing benefits from unstructured sparsity is challenging for three primary reasons: 1) In long sequences, the self-attention mechanism becomes the primary memory bottleneck, diminishing any performance gains from sparsity in the dense linear layers; 2) Current methods for inducing unstructured sparsity typically achieve only modest levels model-wide or else significantly degrade model performance, and 3) GPU memory interfaces (HBM) are optimized for contiguous data access and are illsuited for the fine-grained, irregular memory patterns that result from unstructured sparsity.

SSMs inherently address the first challenge above by replacing the quadratic self-attention mechanism, thereby removing its associated memory bottleneck. In this work, we tackle the remaining two issues by introducing a method to induce high activation sparsity in quantized linear-attention models with minimal performance degradation. We achieve this by injecting a sparsity-inducing pre-activation gate before each layer that in the forward pass, pushes activations within a learnable range of ±∆ towards zero, while in the backward pass, it maintains a smooth gradient flow for these nearzero activations, ensuring stable training. Crucially, the gate preserves high-magnitude activations (outliers), both positive and negative, which are known to be vital for LLM performance [6], [7]. This allows the model to maintain expressiveness without greatly disrupting the original activation distribution. This approach yields model-wide MAC operation reduction from sparse activations of up to 76% with negligible impact on performance or additional training time. Multi-core, multi-chip systems with event-driven processing elements are uniquely positioned to exploit this unstructured sparsity at two levels of the system hierarchy simultaneously: within each core, zero-valued activations can be skipped locally, reducing intra-chip compute and memory traffic; across chips, event-driven interconnects transmit only non-zero activations, directly cutting inter-chip communication overhead. Deploying our sparse model on such a multi-core, multi-chip platform, the benefits become substantial. We project up to 37× reduction in latency and a 16× reduction in energy-per-token compared to a similarly sized transformer-based model on an edge GPU. Compared to a dense version of the same SSM on the same neuromorphic hardware, our method shows a 5.4× improvement in both metrics during the generate phase and 3.5× during prefill.

## II. RELATED WORK

Smoother non-saturating activations, such as GELU [8] and SiLU/Swish [9], have largely replaced ReLU due to improved optimization stability and downstream performance [10]. Unlike ReLU, these functions do not naturally produce zero activations. Recent studies in LLMs, however, show that switching back to ReLU can be done with minimal performance loss [11], with new variants like ReLU² [12] and dReLU [13] that aim to restore or enhance sparsity while retaining competitive performance.

TurboSparse [13] focuses on sparsity in the feed-forward network (FFN) by introducing the dReLU activation function. The Swish activation in the SwiGLU block is replaced with ReLU, and another ReLU is added to the Up projection. By continued pre-training of these modified models, they are able to recover most of the performance of the dense baseline on benchmark tasks. The sparsity they achieve, however, is localized. Only the inputs to the Down projection are zeroed, while most other projections remain dense. As a result, even though sparsities above 90% are reported in parts of the FFN, the overall proportion of active parameters across the model is much lower.

Other approaches have tried to achieve more model-level sparsity rather than only within the FFN. Q-Sparse [14] does this by applying top-K selection to activations and using a straight-through estimator to preserve gradients, combined with squared ReLU to promote sparsity. However, selecting the K largest magnitude activation requires sorting activations on a per-token basis, potentially introducing significant overheads, especially on constrained edge-hardware. Additionally, it requires synchronization across channels, complicating implementation on compute-memory integrated platforms [15]. This is particularly problematic in many-core systems where a single layer may be distributed across multiple cores: identifying the global top-K requires a synchronization barrier across all participating cores before any MAC computation can proceed, introducing inter-core communication overhead that can negate the latency savings from sparsity. TEAL [16] takes a different direction by introducing a layer-wise sparsification strategy. It computes token importance scores and selectively keeps only the most relevant tokens at each layer, allowing each layer to tolerate different levels of sparsity. However, this approach is only applied for the decode phase of inference, leaving the pre-fill phase fully dense and limiting potential end-to-end efficiency improvements.

## III. BACKGROUND

## A. Activation Sparsity in Neural Networks

In an MVM operation, a zero-valued activation implies that all multiply-accumulate (MAC) operations involving its corresponding column of the weight matrix contribute nothing to the output. As shown in Fig. 2, this enables structured skipping: entire columns of weights associated with zero activations can be bypassed, eliminating both the MAC operations and the need to fetch those weights from memory. Since MVMs are often memory-bound, where performance is constrained more by the cost of moving data than by raw compute throughput, reducing memory accesses can directly yield substantial energy and latency gains [17]. By avoiding both the computations and memory accesses for weights corresponding to zero-valued activations, we can save bandwidth, reduce cache pressure, improve latency, and lower overall energy consumption.

![](images/fed1e352609f2731a634ef55bff9a4b5f2c053f4d54e6f55f847cc8596bfd75d.jpg)  
Fig. 2: Illustration of activation sparsity in a matrix-vector multiplication, where zero-valued activations allow skipping associated weight accesses of entire rows in the weight matrix.

If $\rho$ denotes the fraction of zero activations, then in the ideal case, both the memory reads and the number of computations scale proportionally with $( 1 - \rho )$ . This means that the ideal throughput of hardware that can support sparsity is simply the dense throughput over the share of non-zero activations, i.e., $f _ { \mathrm { s p a r s e } } = f _ { \mathrm { d e n s e } } / ( 1 - \rho )$

GPUs tend to struggle to exploit activation sparsity in inference because their architecture is built for dense, highly parallel computation with regular memory access. Unstructured sparse activations break these patterns: nonzero elements are irregularly distributed, requiring indexing and indirection that hurt coalesced memory access and reduce arithmetic efficiency. Since GPU threads execute in lockstep, skipping zeros directly is difficult without wasting compute lanes. To take advantage of sparsity, weights would need to be stored in column-major order, so that all weights linked to a nonzero activation can be fetched contiguously; however, GPUs and their libraries are optimized for dense, row-major, or block layouts, and reordering weights adds overhead. While sparse kernels can exploit activation sparsity to some extent [18], real-world speedups on GPUs are often much lower due to overheads from irregular memory access of dynamic zeroactivation patterns and the limited ability of standard hardware and software to take full advantage of unstructured sparsity in a non-training setting, where the available hardware cannot be saturated with massive batch sizes.

![](images/a73a590f0bca418e6d654a3373de663dfc3994096086ee2d84b99cbf4fb42b87.jpg)  
(a) Projection-wise sensitivity to forced sparsity.

![](images/675cefe1cc5aa0584232f8cc452b432d62d7688f1468f391579f89fbe3ed7c61.jpg)  
(b) Layer-wise sensitivity to forced sparsity.  
Fig. 1: Sensitivity analysis to different degrees of forced top-k sparsity in the MMFreeLM model, showing the increase in loss over the dense base-model when varying levels of sparsity are enforced on a per projection type basis (left) and on per layer basis (right) .

## B. Leveraging activation Sparsity on Neuromorphic Hardware Accelerators

Activation sparsity has been extensively investigated in ASIC hardware due to the potential gains over GPUs, whose more regular and highly parallelized architecture cannot fully exploit it [19]. Accelerators such as Eyeriss v2 [20] and SCNN [21] primarily target convolutional networks, leveraging sparsity to reduce power consumption and increase inference throughput. More recently, neuromorphic computing has renewed interest in hardware optimized for sparse, eventdriven activity [22]–[24], although deployment of large models on the multi-million to billion-parameter scale required for language modeling remains limited in academic chips.

Loihi 2 [25] represents a state-of-the-art implementation of this approach, designed for sparse, event-based neural networks. By focusing on local event-driven computation, Loihi 2 efficiently leverages both sparse weight matrices and dynamically unstructured sparse activations using fixed-point arithmetic. In multi-chip setups, the system further leverages sparse activations through an event-driven inter-chip and intercore communication, minimizing communication overhead by transmitting only non-zero packages.

## IV. METHODS

## A. Model selection

To meet the demands of compact models under constrained power budgets and the need for low-latency, real-time inference at the edge, various linear-attention-based quantized LLMs have been proposed [26], [27]. The MatMul-Free Language Model (MMFreeLM) [28] pushes quantization to an extreme with ternary weights (−1, 0, +1), together with low-precision 8-bit fixed-point activations, transforming dense layers into BitLinear layers that reduce multiplications to simple additions and subtractions. Based on the Gated Recurrent Unit (GRU) proposed by [29], MMFreeLM uses a MMFree-Linear-GRU (MLGRU) with ternary weights, in place of the traditional self-attention mechanism in transformers. With these features combined, the MMFreeLM model has proven to be well-suited for energy-efficient inference across GPUs, edge-GPU, and neuromorphic hardware [26], [28]. To date, it is the only billion-parameter language model deployed on neuromorphic hardware, making it the natural target for sparsification studies aimed at improving efficiency on multicore, multi-chip neuromorphic platforms.

## B. Motivating study

When extending sparsity beyond dense FFN layers to the full model, care must be taken since different linear projections vary in their role and sensitivity to pruning/sparsification [30]. In SSMs, components closely tied to their linear attention, such as projections tied to the hidden state transition $h [ t ]  h [ t + 1 ]$ may be especially sensitive due to their stateful nature. To study projection-wise and layer-wise sensitivity, we injected a forced sparsity into a pre-trained MMFreeLM using top-k selection of the activations with the largest magnitude, applied either per projection type (uniform across layers) or per layer (uniform across projections). Results, shown in Fig. 1, normalize loss increases by each component’s share of FLOPs, highlighting where in the network inactive neurons provide the best tradeoff between active parameters and performance.

Projection-wise analysis: The projection-wise analysis shows that along with the very sensitive LM head, the projections directly involved in the next state transition $( h [ t ]  h [ t +$ 1]), I and $F ,$ are very sensitive to enforced sparsity, whereas projections involved in the output calculation based on the current state and input x[t] are less sensitive. The analysis further highlights the resilience of the large down-projection in the FFN to forced sparsity, which again aligns well with prior sparsification efforts that have targeted this projection with great success [12], [13]. Overall, these findings align well with prior work focusing on weight pruning: in transformers, pruning attention heads leads to larger performance drops than pruning feed-forward layers [31], [32], and in SSMs such as Mamba, stateful steps are much less tolerant of high sparsity than input projections or dense layers [33].

Layer-wise analysis: The layer-wise analysis, contrary to a previous sensitivity study on transformer-based models by [30], shows that there are peaks in the sensitivity of the very first and very last layers, as well as the middle layers. This observation aligns with some other recent studies on transformer architectures, which suggest that middle layers often possess greater redundancy and robustness compared to the more critical early and late layers. For example, [34] conducted a layer-wise importance analysis of feed-forward networks in transformer-based language models, showing that concentrating model capacity in the middle layers while reducing or removing components in the early and late layers improves downstream task performance.

## C. Proposed Sparsification Method

Building on the sensitivity analysis in Section IV-B and prior work on activation sparsity in LLMs, which showed that activations often follow Gaussian or Laplacian distributions with near-zero mean [16], we propose a sparsity-inducing pre-activation applied to the input of every linear projection. Combined with an $L _ { 0 }$ surrogate loss penalty, this mechanism encourages activations to collapse toward zero, while accounting for the varying sensitivities of different projections and layers to balance task performance with activation sparsity on a model-wide level.

Sparse per-projection pre-activation: The proposed preactivation is presented in equation 1 and illustrated in Fig. 3. It consists of a two-sided ReLU that zeros out activations within the range ±∆. This preserves the overall distribution of activations by retaining both positive and negative activations, introducing only a constant offset of $\pm \Delta$ outside the zero region. The threshold ∆ is treated as a learnable parameter and optimized separately for each projection during training.

$$
x _ { \mathrm { s p a r s e } } = \mathrm { s i g n } ( x ) \cdot \mathrm { R e L U } ( | x | - \Delta )\tag{1}
$$

$$
\frac { d x _ { \mathrm { s p a r s e } } } { d x } _ { \mathrm { s m o o t h } } = \sigma \big ( C ( | x | - \Delta ) \big )\tag{2}
$$

Similar to findings showing the smooth nature of the SiLU activation improves learning performance, especially at high levels of sparsity when a large share of gradients would be fully zeroed out by a ReLU activation (i.e. dead neurons) [10], [35], we found that a smooth surrogate for the backwards pass, described in equation 2, gave slightly better convergence with the base models’ training trajectory when compared to a hard magnitude thresholding, especially at higher sparsity levels. A slope parameter C controls the steepness of the smooth mask for the derivative, with larger C values pushing activations

![](images/90b00653eda66854ebe14a855d3ca5b50f4cd4f476b711ee17a493dc043a1a7e.jpg)  
Fig. 3: The sparse pre-activation function, with activations being zeroed out within the range $\pm \Delta ,$ , along with the smoothed out surrogate gradient during the backwards pass.

toward zero more aggressively. C is fixed during training;   
empirically we found that $C = 2 0$ gives good results.

Loss penalty and differentiable sparsity surrogate: To encourage the model to learn to push activations to the range within $\pm \Delta$ , as well as to learn an optimal value $\Delta$ on a perprojection basis, we use an $L _ { 0 }$ loss penalty added to the main task loss. Since directly counting zeros in the activation vector would obstruct gradient flow to this penalty, we instead employ a surrogate sparsity measure sˆ, resulting in the following learning objective:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda \left( 1 - \hat { s } \right) , \quad \hat { s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \exp \bigl ( - k \left| x _ { \mathrm { q - s p a r s e } , i } \right| \bigr ) ,\tag{3}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ is the primary task loss (e.g., cross-entropy), λ controls the strength of the sparsity penalty, N is the number of activations considered, $x _ { \mathrm { q - s p a r s e } , i }$ is the i-th sparse activation, k is the exponential steepness parameter, and sˆ serves as a differentiable proxy for the fraction of zero activations. Empirically, setting $k = 1 0$ was found to provide a good tradeoff between accurately estimating true sparsity and excessively large gradient norms. The sparsity penalty is weighed on a per-layer basis, depending on the resulting reduction in MAC operations due to a zero activation in that layer. Additionally, similar to previous works on neural network pruning through regularization [36], we ramp up the penalty weight term λ slowly at the start of training, to avoid excessive sparsity before important features of the dataset have been learned. We found that a linear warm-up of 5% of the total training steps performed well.

## D. Multi-Chip Deployment on Neuromorphic Hardware

1) Hardware Platform: Our hardware deployment results are derived from the real-world deployment of the dense MM-FreeLM model on the Loihi 2 platform. The platform supports two operating modes [28], illustrated in figure 4. In pipelined mode, new inputs are introduced at every fixed time per step (TPS) and passed through successive layers, maximizing throughput. In fall-through mode, inputs are introduced only after the previous ones have been fully processed, thereby minimizing latency and allowing for a dynamically varying TPS with the per-chip workload. LLM deployment aligns naturally with these modes: prefill processing of long input sequences leverages pipelined mode for throughput efficiency, while autoregressive token generation relies on fall-through mode, since producing token t must complete before token t + 1 can be processed.

![](images/21983ea83efd2fed646a2aca533aef54d8ff6b42f7808e0db28e75dbc00881c4.jpg)  
Fig. 4: Different execution modes on Loihi 2, with pipelined mode (top) optimizing throughput and Fall-through mode (bottom) optimizing latency.

Viewed as a many-core SoC pipeline, this deployment benefits from activation sparsity at two levels of the system hierarchy: intra-chip, where zero-skipping within each core reduces MAC operations and local memory traffic, and interchip, where event-driven links suppress zero-valued packets to cut communication overhead. Both effects are formalized in the performance model below.

## E. Performance Benchmarking and Modeling

We extend the performance modeling framework of [26] to account for the impact of activation sparsity on throughput, latency and power. Starting from the dense baseline, we first derive the effective MAC density r from the layerlevel structure of each block, and then use r to model how sparsity modifies multi-chip throughput under the two Loihi 2 execution modes.

1) Dense Baseline.: The MMFreeLM architecture consists of $N _ { \mathrm { b l o c k s } }$ sequential computational blocks, each mapped to a separate Loihi 2 core or chip. For the 370M parameter model, $N _ { \mathrm { b l o c k s } } = 2 4$ , where each block comprises an MLGRU tokenmixing unit and a ternary FFN channel-mixing block [28].

We adopt as baseline the measured dense multi-chip throughput $f _ { \mathrm { d e n s e } } ^ { \mathrm { g e n e r a t e } }$ and $f _ { \mathrm { d e n s e } } ^ { \mathrm { p r e f i l l } }$ reported by [26]. Results are provided for two inference modes: (i) prefill, where tokens are processed in a pipelined manner with all layers active concurrently, and (ii) generate, where tokens are produced autoregressively with one layer active at a time. Energy per token follows directly as $E _ { \mathrm { d e n s e } } \propto 1 / f _ { \mathrm { d e n s e } } .$ , since Loihi 2 operates under an approximately constant power envelope [26]. This throughput–energy pair serves as the reference for all sparse extensions.

2) Impact of sparse activations: We restrict the analysis to matrix-vector multiplication (MVM) operations, which dominate total FLOPs; nonlinearities and scalar operations (e.g., sigmoid, bias additions) are excluded, consistent with standard FLOP accounting [37]. The FFN in each block contains a fused Up/Gate projection of size $d _ { h } \ \times \ 2 d _ { i }$ and a Down projection of size $d _ { i } \times d _ { h }$ ; the MLGRU block contains four projections (i, f, g, o), each of size $d _ { h } \times d _ { h }$

For projection j, let $\rho _ { j } ~ \in ~ [ 0 , 1 ]$ denote the activation density (fraction of nonzeros) at its input. Since each nonzero activation results in access to one row of the weight matrix, MAC counts scale linearly with $\rho _ { j }$

$$
\begin{array} { r l } & { \mathbf { M A C } _ { \mathrm { d e n s e } } = \displaystyle \sum _ { j } d _ { \mathrm { i n } } ^ { ( j ) } \cdot d _ { \mathrm { o u t } } ^ { ( j ) } , } \\ & { \mathbf { M A C } _ { \mathrm { s p a r s e } } = \displaystyle \sum _ { j } \rho _ { j } \ d _ { \mathrm { i n } } ^ { ( j ) } \cdot d _ { \mathrm { o u t } } ^ { ( j ) } . } \end{array}\tag{4}
$$

(5)

The effective MAC density $r ,$ defined as the ratio of sparse to dense MACs,

$$
r \ = \ { \frac { \mathbf { M } \mathbf { A } \mathbf { C } _ { \mathrm { s p a r s e } } } { \mathbf { M } \mathbf { A } \mathbf { C } _ { \mathrm { d e n s e } } } } ,\tag{6}
$$

is therefore a size-weighted average of the per-projection densities (not a simple mean, since projection sizes differ across layers):

$$
r = \frac { \displaystyle \sum _ { j } \rho _ { j } \cdot d _ { \mathrm { i n } } ^ { ( j ) } \cdot d _ { \mathrm { o u t } } ^ { ( j ) } } { \displaystyle \sum _ { j } d _ { \mathrm { i n } } ^ { ( j ) } \cdot d _ { \mathrm { o u t } } ^ { ( j ) } } .\tag{7}
$$

Per-block latency scales approximately linearly with r, consistent with prior measurements on sparse accelerators [21], [23]. Activation sparsity therefore modifies the dense baseline through two multiplicative speedup factors:

(i) Inter-chip communication speedup. Prior measurements from deployment show that dense multi-chip inference suffers a ≈ 20% throughput reduction from inter-chip communication overhead [26]. Because Loihi 2 uses eventdriven communication, where zero-event packets are skipped [25], this communication overhead can be reduced by sparse packets at the block-boundaries. This principle generalizes to any multi-chip system with traffic-proportional interconnect cost: since the communication penalty scales directly with activation density, increasing model-level sparsity is equivalent to reducing the effective interconnect load. We therefore model the sparsity-dependent inter-chip penalty as

$$
S _ { \mathrm { c o m m } } ( \rho _ { \mathrm { c o m } } ) = \frac { 1 } { 0 . 8 + 0 . 2 \rho _ { \mathrm { c o m } } } ,\tag{8}
$$

which reduces to 1 (no gain) when $\rho _ { \mathrm { c o m } } = 1$ (fully dense), recovering the $f _ { \mathrm { d e n s e } }$ baseline of [26].

(ii) MAC density speedup. Within each block, latency scales in proportion to MAC density r. The intra-block factor, introduced by the zero-skipping of MAC operations, is:

$$
S _ { \mathrm { M A C } } ( r ) = { \frac { 1 } { r } } .\tag{9}
$$

Combined throughput. The sparse-mode throughput is obtained by multiplying the dense baseline with both factors:

$$
f _ { \mathrm { s p a r s e } } = f _ { \mathrm { d e n s e } } \cdot S _ { \mathrm { c o m m } } ( \rho _ { \mathrm { c o m } } ) \cdot S _ { \mathrm { M A C } } ( r ) .\tag{10}
$$

Lastly, due to the different execution modes on Loihi 2 used during prefill and generate, the latency reduction modeled with $S _ { \mathrm { { M A C } } }$ varies slightly between the two modes:

• Prefill: Execution is pipelined, so throughput is bottlenecked by the slowest block. Using the maximum MAC density $r _ { \mathrm { m a x } }$ across blocks:

$$
f _ { \mathrm { s p a r s e } } ^ { \mathrm { p r e f i l l } } = f _ { \mathrm { d e n s e } } ^ { \mathrm { p r e f i l l } } \cdot \frac { 1 } { 0 . 8 + 0 . 2 \rho _ { \mathrm { c o m } } } \cdot \frac { 1 } { r _ { \mathrm { m a x } } } .\tag{11}
$$

• Generate: Execution is sequential, so latency adds linearly across blocks. Using the mean MAC density $r _ { \mathrm { a v g } } \mathrm { : }$

$$
f _ { \mathrm { s p a r s e } } ^ { \mathrm { g e n e r a t e } } = f _ { \mathrm { d e n s e } } ^ { \mathrm { g e n e r a t e } } \cdot \frac { 1 } { 0 . 8 + 0 . 2 \rho _ { \mathrm { c o m } } } \cdot \frac { 1 } { r _ { \mathrm { a v g } } } .\tag{12}
$$

## V. RESULTS

## A. Training setup

We continue training pre-trained 370M and 2.7B MM-FreeLM models on 4B tokens of the FineWebEdu dataset [38] using a cosine learning rate schedule with a reduced initial rate to preserve patterns from the original training. For comparison, we include ReLU-fication [11] and the dReLU method [13] baselines, as well as a continued-training baseline to control for the effect of the additional training data. We train three models with our proposed method at varying sparsity penalty strengths.

## B. Sparsity of trained models

We present the effective MAC of the evaluated sparsification methods compared to our proposed method, along with a perprojection type sparsity, divided into the FFN and MLGRU blocks, in table II. Activation sparsities are captured over the entire benchmark set used in section V-C.

TABLE II: Activation sparsity (%) per projection for different sparsification methods on MMFreeLM.
<table><tr><td></td><td colspan="4">MLGRU</td><td colspan="2">FFN</td><td>Head</td></tr><tr><td>Method</td><td>I</td><td>F</td><td>G</td><td>0</td><td>Up Down</td><td></td><td>MAC sparsity (↑)†</td></tr><tr><td colspan="8">370M MMFreeLM</td></tr><tr><td>Baseline (SiLU) 1.1</td><td></td><td>3.2</td><td></td><td></td><td>1.5 37.3 1.1</td><td>30.1 2.2</td><td>10.0* (10.9)</td></tr><tr><td>ReLU</td><td>1.1 2.8</td><td></td><td></td><td>1.4 37.9 1.2</td><td>87.6</td><td>2.2</td><td>23.7 (21.6)</td></tr><tr><td>dReLU</td><td></td><td>1.1 2.8 1.4 25.0 1.3</td><td></td><td></td><td>93.2</td><td>2.4</td><td>25.1* (22.9)</td></tr><tr><td>Ours (λ = 1.0) 45.7 61.2 48.9 78.2 63.8</td><td></td><td></td><td></td><td></td><td>92.2</td><td>24.0</td><td>69.5* (64.3)</td></tr><tr><td>Ours (λ = 2.0) 61.9 75.6 66.0 85.2 79.1</td><td></td><td></td><td></td><td></td><td></td><td>93.4 40.2</td><td>80.1* (76.2)</td></tr><tr><td colspan="8">2.7B MMFreeLM</td></tr><tr><td>Baseline (SiLU) 1.7 4.3 1.4 41.8 1.3</td><td></td><td></td><td></td><td></td><td>30.4</td><td>2.9 9.2</td><td>12.5 (12.1) 63.2* (61.5)</td></tr><tr><td colspan="8">Ours  $( \lambda = 1 )$  45.7 61.1 40.6 66.3 56.6 91.0</td></tr></table>

Excluding LM Head, which is not included in the Loihi 2 implementation in [28]. Value in parentheses shows the effective MAC sparsity with the LM Head included.  
<sup>†</sup> Calculated as 1 - r as described in section IV-E2.

Due to some inherent sparsity from the fixed-point 8-bit quantization, even the base model exhibits an average baseline parameter sparsity of 10.0%. While both ReLU-based methods achieve significant levels of sparsity in the FFN at the input of the large down-projection, the impact on the overall share of active parameters is limited, as the FFN only makes up ≈ $2 / 3$ of total FLOPS, with the down-projection contributing to just $\approx 1 / 3$ of that. The result also shows that the second ReLU activation inserted with the dReLU method has a limited impact on model-wide sparsity, as the dot-product between the sparse post-activation output of the gate projection with the dense Up projection in and of itself already results in a sparse vector. Our proposed method is able to achieve significantly higher levels of model-wide sparsity by not only targeting linear projections where SiLU activations can be replaced by ReLU, but all linear projections in the model.

## C. Performance on Reasoning Tasks

TABLE III: Zero-shot accuracy of sparse MMFreeLM models compared to the dense baseline. All models use ternary weights and 8-bit activations.
<table><tr><td>Model</td><td>MACs (↓)</td><td>ARCe</td><td>HS</td><td>OQ</td><td>PQ</td><td>Avg</td></tr><tr><td colspan="7">370M MMFreeLM</td></tr><tr><td>Baseline (SiLU)</td><td>307M</td><td>41.54</td><td>32.69</td><td>30.40</td><td>62.89</td><td>41.88</td></tr><tr><td>ReLU</td><td>267M</td><td>39.90</td><td>32.99</td><td>31.40</td><td>61.70</td><td>41.50</td></tr><tr><td>dReLU</td><td>263M</td><td>38.68</td><td>32.08</td><td>29.20</td><td>61.04</td><td>40.25</td></tr><tr><td>Ours (λ = 1)</td><td>118M</td><td>41.29</td><td>31.58</td><td>31.00</td><td>61.10</td><td>41.24</td></tr><tr><td>Ours (λ = 2)</td><td>95M</td><td>38.38</td><td>30.60</td><td>29.80</td><td>60.17</td><td>39.74</td></tr><tr><td colspan="7">2.7B MMFreeLM</td></tr><tr><td>Baseline (SiLU)</td><td>2.32B</td><td>50.55</td><td>47.54</td><td>35.00</td><td>69.26</td><td>50.59</td></tr><tr><td>Ours (λ = 1)</td><td>1.01B</td><td>48.32</td><td>43.43</td><td>35.80</td><td>66.43</td><td>48.50</td></tr></table>

We evaluated the zero-shot performance of the sparsified models on the same set of language tasks as in the original MMFreeLM work, including ARC-Easy, ARC-Challenge [39], [40], HellaSwag [41], OpenBookQA [42], PIQA [43], and WinoGrande [44]. The results are presented in table III, showing a small degradation in the average reasoning task performance compared to the dense baseline model, with the 370M (λ = 1) model outperforming the dReLU model at just half the average active MAC operations.

TABLE I: Throughput and efficiency across of various dense and sparse language models, including our sparse MMFreeLM, for prefill and generation across various sequence lengths, running on a NVIDIA H100 GPU, Intel’s Loihi 2 and a Nvidia Jetson.
<table><tr><td colspan="3"></td><td colspan="4">Throughput (↑ tokens/sec)</td><td colspan="4">Efficiency (↓ mJ/token)</td></tr><tr><td colspan="3">Sequence length</td><td>500</td><td>1000</td><td>4000</td><td>8000</td><td>500</td><td>1000</td><td>4000</td><td>8000</td></tr><tr><td rowspan="6">Genate</td><td>MMF (sparse)</td><td> $\mathbf { L o i h i } \ 2 ^ { \dagger }$ </td><td>224.1</td><td>224.1</td><td>224.1</td><td>224.1</td><td>75.0</td><td>75.0</td><td>75.0</td><td>75.0</td></tr><tr><td>MMF (dense)</td><td> $\operatorname { L o i h i } 2 ^ { * }$ </td><td>41.5</td><td>41.5</td><td>41.5</td><td>41.5</td><td>405</td><td>405</td><td>405</td><td>405</td></tr><tr><td>MMF (dense)</td><td> $\mathrm { H } 1 0 0 ^ { \ddagger }$ </td><td>13.4</td><td>13.3</td><td>13.5</td><td>13.2</td><td>10.1k</td><td>10.1k</td><td>10.0k</td><td>9.9k</td></tr><tr><td>TF++</td><td> $\mathrm { H } 1 0 0 ^ { \ddagger }$ </td><td>22.4</td><td>22.9</td><td>21.7</td><td>21.3</td><td>5.5k</td><td>5.6k</td><td>6.2k</td><td>6.8k</td></tr><tr><td>Alireo (400M)</td><td> $\operatorname { J e t s o n } { \mathrm { ~ \sharp ~ } }$ </td><td>14.3</td><td>14.9</td><td>14.7</td><td>15.2</td><td>723</td><td>719</td><td>853</td><td>812</td></tr><tr><td>Qwen2 (500M)</td><td> $\operatorname { J e t s o n } { \mathrm { ~ \sharp ~ } }$ </td><td>13.4</td><td>14.0</td><td>14.1</td><td>15.4</td><td>791</td><td>785</td><td>912</td><td>839</td></tr><tr><td rowspan="6">Preil</td><td>MMF(sparse)</td><td> $\mathbf { L o i h i } \ 2 ^ { \dagger }$ </td><td>23.2k</td><td>23.2k</td><td>23.2k</td><td>23.2k</td><td>1.1</td><td>1.1</td><td>1.1</td><td>1.1</td></tr><tr><td>MMF (dense)</td><td> $\operatorname { L o i h i } 2 ^ { * }$ </td><td>6632</td><td>6632</td><td>6632</td><td>6632</td><td>3.7</td><td>3.7</td><td>3.7</td><td>3.7</td></tr><tr><td>MMF (dense)</td><td> $\mathrm { H } 1 0 0 ^ { \ddagger }$ </td><td>11.4k</td><td>13.1k</td><td>30.6k</td><td>51.6k</td><td>6.1</td><td>5.3</td><td>2.5</td><td>1.4</td></tr><tr><td>TF++</td><td> $\mathrm { H } 1 0 0 ^ { \ddagger }$ </td><td>21.6k</td><td>32.7k</td><td>44.3k</td><td>55.4k</td><td>11.3</td><td>7.3</td><td>5.4</td><td>4.3</td></tr><tr><td>Alireo (400M)</td><td> $\operatorname { J e t s o n } { \mathrm { ~ \sharp ~ } }$ </td><td>849</td><td>1620</td><td>3153</td><td>2258</td><td>11.7</td><td>7.8</td><td>6.8</td><td>7.6</td></tr><tr><td>Qwen2 (500M)</td><td> $\operatorname { J e t s o n } { \mathrm { ~ \sharp ~ } }$ </td><td>627</td><td>909</td><td>2639</td><td>3861</td><td>17.9</td><td>13.9</td><td>6.7</td><td>4.4</td></tr></table>

<sup>†</sup> Proposed sparse model $( \lambda = 2 . 0 )$ with metrics extrapolated from dense model using equations 11 and 12  
<sup>\*</sup> Baseline deployment results of 370M dense MMFreeLM in multi-chip setup from [26]. Includes inter-chip communication slowdown over single-chip measurements.  
<sup>‡</sup> Jetson and H100 metrics from reported deployment by [28].

## D. Energy efficiency of sparse model

We apply the methodology described in IV-B to the 24-chip Loihi measurements in [26] on our sparse 370M-parameter model (λ = 2) to estimate the performance gains of a sparse model. Our model shows a worst block MAC density of $r _ { m a x } = 0 . 3 1$ in layer 17. The input activation density of the same block is taken as the average activation sparsity to the MLGRU (i, f & g-proj) of the same layer (see [28] for mapping details) and calculated to $\rho _ { c o m } = 0 . 6 1$ . Using equation 11, we calculate a decrease in latency and energy-per-token of 3.5× against the dense deployment for prefill.

For generate, we use equation 12 with an average modelwide MAC density of $r _ { a v g } = 0 . 2 0$ and an average $\rho _ { c o m } =$ 0.67 to calculate an improvement in both metrics of 5.4× as compared to the dense baseline.

For further comparison, we also include deployment metrics by [28] of transformer models with comparable downstream task performance, including the 500M parameter Qwen2 model [45], and a 400M parameter Alireo model [46] on GPU and edge-GPU (Jetson).

## VI. CONCLUSION

We present a method for inducing high activation sparsity in heavily quantized linear-attention models through learnable sparsifying pre-activations, achieving up to 76% reduction in MAC operations with minimal performance loss. Sparsity reduces system cost at two levels in a multi-chip deployment: intra-chip, by skipping zero-activation MAC operations within each processing element, and inter-chip, by reducing activation volume transmitted across the chip network. Applying our performance model to the only billion-parameter model deployed on a neuromorphic multi-chip platform, we project up to 5.4× gains in throughput and energy efficiency over the dense baseline, and up to 37× over a comparable transformer on edge GPU, gains that GPU architectures, constrained by their regular memory hierarchies, cannot realize from unstructured sparsity alone. These results demonstrate that combining dynamic activation sparsity with aggressively quantized recurrent models is a natural fit for event-driven multi-core, multi-chip platforms.

## REFERENCES

[1] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” 2024. [Online]. Available: https://arxiv.org/abs/2312.00752

[2] M. Popov, A. Kallala, A. Ramesh, N. Hennouni, S. Khaitan, R. Gentry, and A.-S. Cohen, “Leveraging state space models in long range genomics,” 2025. [Online]. Available: https://arxiv.org/abs/2504.06304

[3] D. Wang, R.-J. Zhu, S. Abreu, Y. Shan, T. Kergan, Y. Pan, Y. Chou, Z. Li, G. Zhang, W. Huang et al., “A systematic analysis of hybrid linear attention,” arXiv preprint arXiv:2507.06457, 2025.

[4] A. Voelker, I. Kajic, and C. Eliasmith, “Legendre memory units:´ Continuous-time representation in recurrent neural networks,” Advances in neural information processing systems, vol. 32, 2019.

[5] H. Cheng, M. Zhang, and J. Q. Shi, “A survey on deep neural network pruning-taxonomy, comparison, analysis, and recommendations,” 2024. [Online]. Available: https://arxiv.org/abs/2308.06767

[6] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large language models,” in International conference on machine learning. PMLR, 2023, pp. 38 087–38 099.

[7] R. Raman, K. Sharma, and S. Q. Zhang, “Rethinking the outlier distribution in large language models: An in-depth study,” arXiv preprint arXiv:2505.21670, 2025.

[8] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” 2023. [Online]. Available: https://arxiv.org/abs/1606.08415

[9] P. Ramachandran, B. Zoph, and Q. V. Le, “Searching for activation functions,” 2017. [Online]. Available: https://arxiv.org/abs/1710.05941

[10] S. R. Dubey, S. K. Singh, and B. B. Chaudhuri, “Activation functions in deep learning: A comprehensive survey and benchmark,” 2022. [Online]. Available: https://arxiv.org/abs/2109.14545

[11] I. Mirzadeh, K. Alizadeh, S. Mehta, C. C. D. Mundo, O. Tuzel, G. Samei, M. Rastegari, and M. Farajtabar, “Relu strikes back: Exploiting activation sparsity in large language models,” 2023. [Online]. Available: https://arxiv.org/abs/2310.04564

[12] Z. Zhang, Y. Song, G. Yu, X. Han, Y. Lin, C. Xiao, C. Song, Z. Liu, Z. Mi, and M. Sun, “Relu<sup>2</sup> wins: Discovering efficient activation functions for sparse llms,” 2024. [Online]. Available: https://arxiv.org/abs/2402.03804

[13] Y. Song, H. Xie, Z. Zhang, B. Wen, L. Ma, Z. Mi, and H. Chen, “Turbo sparse: Achieving llm sota performance with minimal activated parameters,” 2024. [Online]. Available: https://arxiv.org/abs/2406.05955

[14] H. Wang, S. Ma, R. Wang, and F. Wei, “Q-sparse: All large language models can be fully sparsely-activated,” 2024. [Online]. Available: https://arxiv.org/abs/2407.10969

[15] A. Pierro, S. Abreu, J. Timcheck, P. Stratmann, A. Wild, and S. B. Shrestha, “Accelerating linear recurrent neural networks for the edge with unstructured sparsity,” 2025. [Online]. Available: https://arxiv.org/abs/2502.01330

[16] J. Liu, P. Ponnusamy, T. Cai, H. Guo, Y. Kim, and B. Athiwaratkun, “Training-free activation sparsity in large language models,” 2025. [Online]. Available: https://arxiv.org/abs/2408.14690

[17] Y.-H. Chen, T.-J. Yang, J. Emer, and V. Sze, “Eyeriss v2: A flexible accelerator for emerging deep neural networks on mobile devices,” IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 9, no. 2, pp. 292–308, 2019.

[18] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re, and B. Chen, “Deja vu: Contextual sparsity for efficient llms at inference time,” 2023. [Online]. Available: https://arxiv.org/abs/2310.17157

[19] M. Shi, A. Kneip, N. Chauvaux, J. Sun, C. Frenkel, and M. Verhelst, “Sparsity-aware hardware: From overheads to performance benefits,” IEEE Solid-State Circuits Magazine, vol. 17, no. 2, pp. 61–71, 2025.

[20] Y.-H. Chen, T.-J. Yang, J. Emer, and V. Sze, “Eyeriss v2: A flexible accelerator for emerging deep neural networks on mobile devices,” IEEE Journal of Emerging and Selected Topics in Circuits and Systems, vol. 8, no. 3, pp. 198–210, 2018.

[21] S. W. Keckler, D. Burger, H. Esmaeilzadeh et al., “Scnn: An accelerator for compressed-sparse convolutional neural networks,” Proceedings of the 44th Annual International Symposium on Computer Architecture, pp. 27–39, 2017.

[22] S. Kim, S. Kim, S. Um, S. Kim, J. Lee, and H.-J. Yoo, “Snpu: An energy-efficient spike domain deep-neural-network processor with twostep spike encoding and shift-and-accumulation unit,” IEEE Journal of Solid-State Circuits, vol. 58, no. 10, pp. 2812–2825, 2023.

[23] M. Sadeghi, Y. Rezaeiyan, D. F. Khatiboun, S. Eissa, F. Corradi, C. Augustine, and F. Moradi, “Nexus: A 28nm 3.3pj/sop 16-core spiking neural network with a diamond topology for real-time data processing,” IEEE Transactions on Biomedical Circuits and Systems, vol. 19, no. 3, pp. 523–535, 2025.

[24] Y. Liu, Z. Wang, W. He, L. Shen, Y. Zhang, P. Chen, M. Wu, H. Zhang, P. Zhou, J. Liu, G. Sun, J. Ru, L. Ye, and R. Huang, “An 82nw 0.53pj/sop clock-free spiking neural network with 40µs latency for alot wake-up functions using ultimate-event-driven bionic architecture and computingin-memory technique,” in 2022 IEEE International Solid-State Circuits Conference (ISSCC), vol. 65, 2022, pp. 372–374.

[25] Intel Corporation, “Taking neuromorphic computing to the next level with loihi 2,” 2021. [Online]. Available: https://download.intel.com/newsroom/2021/newtechnologies/neuromorphic-computing-loihi-2-brief.pdf

[26] S. Abreu, S. B. Shrestha, R.-J. Zhu, and J. Eshraghian, “Neuromorphic principles for efficient large language models on intel loihi 2,” arXiv preprint arXiv:2503.18002, 2025.

[27] H.-Y. Chiang, C.-C. Chang, N. Frumkin, K.-C. Wu, M. S. Abdelfattah, and D. Marculescu, “Quamba2: A robust and scalable post-training quantization framework for selective state space models,” 2025. [Online]. Available: https://arxiv.org/abs/2503.22879

[28] R.-J. Zhu, Y. Zhang, S. Abreu, E. Sifferman, T. Sheaves, Y. Wang, D. Richmond, S. B. Shrestha, P. Zhou, and J. K. Eshraghian,

“Scalable matmul-free language modeling,” 2025. [Online]. Available: https://arxiv.org/abs/2406.02528

[29] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using rnn encoder-decoder for statistical machine translation,” 2014. [Online]. Available: https://arxiv.org/abs/1406.1078

[30] H. Shao, B. Liu, B. Xiao, K. Zeng, G. Wan, and Y. Qian, “One-shot sensitivity-aware mixed sparsity pruning for large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2310.09499

[31] P. Michel, O. Levy, and G. Neubig, “Are sixteen heads really better than one?” arXiv preprint arXiv:1905.10650, 2019. [Online]. Available: https://arxiv.org/abs/1905.10650

[32] E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov, “Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned,” arXiv preprint arXiv:1905.09418, 2019. [Online]. Available: https://arxiv.org/abs/1905.09418

[33] T. Dao, H. Nguyen et al., “On pruning state-space llms,” arXiv preprint arXiv:2502.18886, 2025. [Online]. Available: https://arxiv.org/abs/2502.18886

[34] W. Ikeda, K. Yano, R. Takahashi, J. Lee, K. Shibata, and J. Suzuki, “Layerwise importance analysis of feed-forward networks in transformer-based language models,” arXiv preprint arXiv:2508.17734, 2025. [Online]. Available: https://arxiv.org/abs/2508.17734

[35] C. C. Horuz, G. Kasenbacher, S. Higuchi, S. Kairat, J. Stoltz, M. Pesl, B. A. Moser, C. Linse, T. Martinetz, and S. Otte, “The resurrection of the relu,” 2025. [Online]. Available: https://arxiv.org/abs/2505.22074

[36] H. Wang, C. Qin, Y. Zhang, and Y. Fu, “Neural pruning via growing regularization,” 2021. [Online]. Available: https://arxiv.org/abs/2012.09243

[37] U. Evci, T. Gale, J. Menick, P. S. Castro, and E. Elsen, “Rigging the lottery: Making all tickets winners,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, H. D. III and A. Singh, Eds., vol. 119. PMLR, 13–18 Jul 2020, pp. 2943–2952. [Online]. Available: https://proceedings.mlr.press/v119/evci20a.html

[38] A. Lozhkov, L. Ben Allal, L. von Werra, and T. Wolf, “Fineweb-edu: the finest collection of educational content,” 2024. [Online]. Available: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

[39] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try arc, the ai2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018.

[40] V. Yadav, S. Bethard, and M. Surdeanu, “Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2019, pp. 2578–2589.

[41] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, “Hellaswag: Can a machine really finish your sentence?” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2019, pp. 4791–4800.

[42] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor conduct electricity? a new dataset for open book question answering,” arXiv preprint arXiv:1809.02789, 2018.

[43] Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi, “Piqa: Reasoning about physical commonsense in natural language,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 7432–7439.

[44] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi, “Winogrande: An adversarial winograd schema challenge at scale,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 8732–8740.

[45] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, G. Dong, H. Wei, H. Lin, J. Tang, J. Wang, J. Yang, J. Tu, J. Zhang, J. Ma, J. Yang, J. Xu, J. Zhou, J. Bai, J. He, J. Lin, K. Dang, K. Lu, K. Chen, K. Yang, M. Li, M. Xue, N. Ni, P. Zhang, P. Wang, R. Peng, R. Men, R. Gao, R. Lin, S. Wang, S. Bai, S. Tan, T. Zhu, T. Li, T. Liu, W. Ge, X. Deng, X. Zhou, X. Ren, X. Zhang, X. Wei, X. Ren, X. Liu, Y. Fan, Y. Yao, Y. Zhang, Y. Wan, Y. Chu, Y. Liu, Z. Cui, Z. Zhang, Z. Guo, and Z. Fan, “Qwen2 technical report,” 2024. [Online]. Available: https://arxiv.org/abs/2407.10671

[46] M. Montebovi, “Alireo-400m: A lightweight italian language model,” 2024.