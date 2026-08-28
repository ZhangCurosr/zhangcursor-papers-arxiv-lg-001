# A Layer Importance Metric for Quantization Accounting for the Speed–Quality Trade-of in Autoregressive Models

A.K. Safronov

safro@sfedu.ru

## Abstract

Small large language models (sLLMs) are now deployed on devices with limited memory and compute budget. Within the autoregressive setting, inference is memory-bandwidth bound: typical homogeneous quantization is typically detrimental to such models due to their compactness, as their architectures do not retain much redundancy, and in particular a handful of components display extremely poor tolerance to lower precision.

This paper introduces a composite metric that combines two orthogonal assessment criteria for such components: the information retention after quantization, captured by a normalized coeficient, and the gains in throughput that can be expected from accelerating individual layers, analytically estimated using a roofline-based latency model. Using an architectural profiling benchmark on Gemma 3 1B, we identify the Feed-Forward Network (FFN) blocks and the embedding matrix as the most promising candidates for accelerated inference, and we estimate for each candidate a normalized quality score, computed via simulated quantization, and a normalized speed score, computed without any model execution. The metric uses a single prioritization coeficient to combine the two scores into a composite value, which can be tuned to prioritize speed vs quality according to a specific application’s needs. The metric is general and can be applied to blocks of layers (such as FFNs in the transformer), their internal projections (such as the feed forward network within a transformer block), or single transformer layers, with no modifications to the computation.

We evaluated our method on multiple model architectures, measuring around 4% error in predicted speedups on average, and demonstrated that it enables significantly better resource allocation decisions than existing approaches, which usually rely on extensive evolutionary search strategies, domain-specific inference accelerators, or require access to expensive Shapley-value approximations. In short, our analytical approach makes sLLM quantization both faster, easier, and more intuitive, turning a trial-and-error process into a mathematical exercise.

Keywords: quantization, small language models, sLLM, roofline model, SQNR, inference acceleration, model compression

## 1 Introduction

The current practice in large language model (LLM) design appears to be contradictory, in that while model architectures become increasingly complex, their physical realization is subject to rather strict limitations imposed by memory bus widths, battery capacity, or the available silicon area. AI assistants are now omnipresent in smartphones and other local machines where they perform a variety of non-critical tasks such as content creation and web browsing [23]. Yet even in this setting, when an LLM is used for generic-purpose assistance, the emphasis is placed on the model’s ability to operate in a memory-constrained environment rather than perform hundreds of billions of operations per second; put otherwise, a compactly represented LLM is significantly more valuable than a large one at the same level of functional complexity. Now, the critical question is: how to achieve a high degree of model reduction while preserving the response quality, as witnessed by the recent advances in knowledge distillation and parameter sharing.

It is not accidental that currently available small LLMs (Llama 3 1B/3B, Gemma 3 1B, Qwen 3 1.7B, etc.) are the result of aggressive model distillation: a large-scale LLM possesses ample resources to withstand the pruning of its parameter count, which has little efect on its accuracy, while an sLLM (small LLM) does not have these advantages. If post-training quantization (PTQ) is applied to an sLLM, it may not produce acceptable results: the internal structure of the model remains intact, but its behavioral patterns are altered, so that the response quality decreases substantially.

Another issue is that modern accelerators are often memory-bandwidth-constrained [3]: the majority of the time spent in autoregressive decoding is occupied by ofloading weights from video memory, not performing arithmetic operations, which also appears to be the case when naively implementing the attention mechanism [16].

Thus, the utilization of a GPU in this scenario is significantly lower than could be the case, and the user must wait for the next token to be generated.

A uniform reduction in the precision of weights and activations, applicable to each layer of the model, is not the only possibility; indeed, diferent network layers may respond diferently to such a reduction. Some layers are relatively inert and can safely undergo aggressive quantization without noticeable efects on the quality of generation, while others store crucial information that should be left untouched. Several methods have been developed to date that allow one to quantify different elements of the model with varying degrees of precision. However, each of them is limited by certain design decisions: CARVQ [4] and IMPQ [6] utilize multi-level vector quantization or require extensive computation to estimate each component’s contribution to the total model performance using Shapley-value estimation [21], while HAPM [5] requires specialized inference software to demonstrate its efectiveness. Thus, none of these approaches provides a practical solution for easily identifying the set of layers within an sLLM that could benefit from higher precision.

The paper suggests that in order to achieve the best possible trade-of between performance and quality, one must move away from uniform quantization and instead attempt to find groups of operations within the model that contribute the most to its quality. In other words, it is more useful to spend computational resources (measured in FLOPS) on operations that preserve information rather than discard it.

We propose a method by which one can assess the quality of an sLLM quantization configuration using a single metric that estimates the cost-quality trade-of. The paper reports the following results:

• profiling the Gemma 3 1B architecture to identify the blocks that contribute most to inference latency, and justifying the choice of the FFN and embedding matrix as priority quantization targets;

• obtaining a normalized quality score for the FFN and Embedding blocks via the SQNR coefficient using the SA-PTQ operator;

• obtaining a normalized speed score for each block based on an analytical roofline model, without re-running the model;

• combining both scores into an integrated metric with a priority parameter α, allowing the balance between speed and quality to be shifted depending on the requirements of a specific task.

The analysis presented confirms the applicability of the metric on standard hardware without quantizing the model, and substantiates the possibility of extending it to layer-wise heterogeneous quantization.

Goal of the present work. The goal is to develop an analytical metric for estimating layer importance that accounts for the trade-of between inference speed and generation quality in autoregressive models. This tool is intended to identify structural blocks and layers whose quantization will provide the maximum hardware gain at minimal information loss.

## 1.1 Domain analysis

To justify the necessity of the methodology for the choice of blocks presented in the paper, it is essential to explain the technological context of optimization. Namely, the common industry practice which allows to reduce VRAM usage and thus accelerate inference without retraining is model quantization - a process of reducing the bit width of weights to 4 or 8 bits. It goes without saying that quantization is never the sole optimization step - it rather represents the final stage when most of the optimizations are already performed. And the issue with optimization is not quantitative - the tooling for quantitative analysis is lacking. In other words, a significant amount of blocks can be subjected to quantization without afecting the performance of the model, while some blocks cannot. This is why quantization as a concept is not the ultimate goal but only the stage that requires analysis. And this is the analysis that the given paper provides with, which is why quantization is framed as a task to solve and not a goal to achieve.

It is vital to understand that ultimately, the objective of the proposed metric is to prioritize model blocks and layers in terms of information robustness. This way, all blocks that require high precision to maintain their functional capabilities are separated from those that allow for aggressive optimization with little loss in terms of accuracy. And this prioritization is ultimately the basis for heterogeneous models which can provide the most significant speed gains with the least losses in performance.

## 1.2 Related work

Small language models have enabled to run LLMs on the end-user device. However, due to the nature of the Transformer architecture, generating tokens is a slow process, and an unquantized model is often too slow to be useful [22].

Quantization, which reduces the bit-width of weights, is the standard method to tackle the problem. However, it is not trivial to apply existing methods to smaller models.

Compared to large models, small ones are more information-dense: they do not have the luxury of discarding some weights as not useful. As a result, applying the same compression to every layer harms the usefulness of the model more than its size.

Several papers addressed this issue from diferent angles. CARVQ aims to tackle the problem of embedding matrix size by compressing it to around 1.6 bits using a corrective adaptor and group residual vector quantization [4]. The authors note that, after PTQ of the transformer itself, the embedding matrix comprises the largest portion of the model (52% for LLaMA-3.2-1B) [4], so it should be compressed aggressively. CARVQ requires lookup tables and nonlinear transformations to be applied at inference time, which complicates the integration with other code. The metric proposed in this paper, in contrast, utilizes conventional scalar quantization (SA-PTQ) and measures the SQNR of the compressed weights, without requiring any additional structures. As a result, it can be used in existing tools such as llama.cpp.

HAPM paper [5] suggests another approach, prunning the weights of the language models on mobile phones with hardware-aware latency modeling. They introduce the concept of real latency sensitivity, which is linked directly to the accuracy drop of a pruned model. The structure of the resulting model becomes sparse, requiring specialized inference machinery to attain the reported speed gains. The $S _ { n }$ metric from the current paper does not depend on model structure and can be applied to any model. It measures the information contained in the weights, normalized by the time spent on processing them (∆L). It achieves the desired speed gains by eliminating lower-ranked layers outright rather than changing their structure, and can utilize existing accelerators.

IMPQ paper [6] analyzes interactions between layers using Shapley values from game theory to determine the importance of diferent weight tensors. While it provides theoretical justification for the layer ranking, the computation of the Shapley values is expensive to apply to models with tens of billions of weights, as in the LLaMA series. $S _ { n }$ , on the other hand, only needs to perform a relatively quick analytical probing of the model to approximate the importance of each layer. This allows to spend less time on model analysis and focus on finding the best compression option.

QRazor paper [7] combines two-step weight and activation compression with Significant Data Razoring to compress the LLM by removing the least-significant bits. They modify the Attention blocks, but the authors claim that their changes to the performance are minimal. Unlike QRazor, which modifies the Attention blocks directly, this work focuses primarily on FFN and Embedding: eficient attention variants such as GQA [20] already reduce its memory footprint architecturally to some extent, and our tests (Section 2) showed that Attention contributes comparatively little to per-token latency in the configurations we profiled, making it a lower priority target here. QRazor uses a heuristic based on amplitudes of the weights to rank the importance of bits, while $S _ { n }$ uses an energy-based SQNR metric to approximate the value of each bit more accurately. This allows to select the most valuable bits to keep, providing the flexibility to choose the desired generation quality/bit-width balance.

## 1.3 Problem statement

The goal of the present study is to develop a metric for evaluating neural network components that allows ranking them by significance for inference acceleration while preserving generation quality. Formally, the task can be represented as a search for a heterogeneous quantized model configuration that provides maximum performance under a given quality-loss threshold.

Formal description of parameters. Let the original model M be represented as an ordered set of components $\{ C _ { 1 } , C _ { 2 } , \dots , C _ { n } \}$ , where a component may be an entire structural block (FFN, Embedding), an individual projection within a block $( W _ { g a t e } , W _ { u p } , W _ { d o w n } )$ , or a separate transformer layer. The quantization process consists in mapping each layer to the space of admissible bit widths $B = \{ b _ { 4 } , b _ { 8 } , b _ { 1 6 } \}$ . The state of the system is characterized by the following functions:

$L ( L _ { i } , b )$ – the latency function, defining the inference execution time of block $L _ { i }$ at bit width $b ;$

$Q ( M , \{ b _ { 1 } , \dots , b _ { n } \} )$ – the integral quality indicator (functional robustness), characterizing the deviation of the quantized model’s output signal from the reference.

To find the optimal bit-width distribution $\left\{ b _ { 1 } , b _ { 2 } , \ldots , b _ { n } \right\}$ , we formulate an optimization problem. The objective function, aimed at minimizing total inference time (the eficiency criterion), is given by (1).

$$
\sum _ { i = 1 } ^ { n } \Big ( L _ { M L P } ^ { ( i ) } ( b _ { i } ) + L _ { E m b } ( b ) \Big ) \longrightarrow \operatorname* { m i n }\tag{1}
$$

At the same time, the system is subject to a boundary condition (2), defining the maximum admissible degradation of the information signal (the quality constraint).

$$
S Q N R _ { t o t a l } \geq S Q N R _ { b a s e l i n e } - \epsilon\tag{2}
$$

where expression (1) defines the eficiency criterion, and inequality (2) fixes the maximum admissible signal-energy loss $\epsilon ,$ expressed in decibels (dB).

The result of this formalization is the development of an integral selectivity criterion score(α) (Fig. 1).

![](images/0da019b8ff65f60891b964fc1f7865835e2addd2c2266da92030cee3194e0359.jpg)  
Figure 1: Algorithmic scheme for constructing the selectivity metric score(α). Source: compiled by the author.

This criterion combines two normalized components – the information preservation of a component $Q _ { k }$ and its speed potential $S _ { k }$ – through the priority parameter α, allowing a heterogeneous architecture to be formed that is balanced with respect to speed and generation quality.

## 2 Formalizing the Latency Parameters of FFN and Embedding

For an efective compression-evaluation metric, one has to move away from an abstract consideration of weight counts to the concrete impact each component’s size has on the resulting inference speed. And indeed, when speaking of sLLMs, the relevant metric is not the raw power, but the latency introduced by reading data from memory. A cost metric, discussed here, aims to provide a means of precise prediction by answering the question of what “gain” in performance, measured in seconds saved, is projected from reducing the bit-width of a particular layer. This metric turns the vague question of how many layers to quantize into an exacting calculation, the parameters of which are defined by the desired acceleration and the quality of the signal preserved.

Beforehand, however, a profiling operation was performed on the Gemma 1B model to identify the major contributors to token-generation latency. The metric of choice, for this particular study, was simply the token generation speed, as it defines the applicability of a quantized model in practice. The blocks under review were 8-bit quantized to identify the primary sources of overhead, which were then isolated for further analysis.

The operations of profiling and identifying computational dominants were performed in Google Colab with T4 GPU using the resources of the llama.cpp toolkit. In particular, the llama-quantize utility, which allows for changing the bit-width of individual blocks by means of typecasting particular tensors, was used. Gemma 3 1B was quantized to Q8\_0 for the relevant parameter groups: FFN, Attention mechanism layers (Q, K, V, Out) and Embedding matrix, with F16-weights model serving as the reference point. The metrics for each configuration were gathered automatically with the help of the llama-cpp-python library with at least three runs for each set of parameters to eliminate hardware variation noise.

The characteristics of architectural performance were defined with the consideration of three factors: token generation speed (tokens/s), context processing latency and dynamic VRAM consumption. The latter, in particular, was evaluated by means of direct reading of the VRAM utilization percentages from the nvidia-smi utility with 0.05s sampling frequency throughout the duration of calculations. Diversified input sequences were used for the calculations to observe the behavior of the model under both light (short queries) and heavy (8192 tokens, typical for gaming sessions) loads.

The results of the profiling runs (see Table 1) confirm the nonlinear nature of the relationship between the bit-width of model sections and their impact on the inference speed. In particular, the analysis demonstrates that the contribution of the FFN blocks and the Embedding matrix to the overall performance is disproportionately high compared to the input dimensions: thus, the speed gain from reducing their bit-depth from F16 to Q8 is substantially larger for both than for Attention layers.

On the other hand, the comparison of the configurations with reduced precision of FFN and Embedding (“Q8 Q8 Q8”) and the baseline (“Q8 fn\_all=F16”) show that the omission of these blocks from quantization results in significantly decreased performance of the 8-bit model. In particular, for the short context length, the token generation speed decreases to 72.3 tok/s for the latter configuration, compared to the fully quantized alternative. And while the contribution of the embedding layer is less obvious, the drop in throughput for completely unquantized Embedding is still substantial.

By comparison, the impact of the Attention layers is significantly lower: even with a similar reduction in precision (from F16 to Q8), the loss of performance from omitting these blocks is smaller for both context lengths. Altogether, the observations confirm the hypothesis that in the Gemma 3 1B architecture, the FFN blocks and the Embedding matrix are the critical performance factors. Their contribution to the overall latency dominates over other sources of overhead and, therefore, should be prioritized when selecting layers for quantization in a local deployment.

Table 1: Performance measurements for diferent quantization variants of Gemma 3 1B blocks (FFN/Attn/Emb) at context lengths of 10, 50, and 100 tokens.
<table><tr><td>Variant (FFN/Attn/Emb)</td><td>short</td><td>dnd_10</td><td>dnd_50</td><td>dnd_100 size, MB</td></tr><tr><td>Q8 only attn_v</td><td>62.3</td><td>63.7</td><td>61.3</td><td>58.0 1999</td></tr><tr><td> $\mathrm { Q 8 ~ o n l y ~ a t t n \_ ~ k v }$ </td><td>63.1</td><td>63.9</td><td>58.5</td><td>59.1 1992</td></tr><tr><td> $\mathrm { Q 8 ~ o n l y ~ a t t n \_ ~ k ~ }$ </td><td>64.4</td><td>60.8</td><td>61.4</td><td>58.0 1999</td></tr><tr><td> $\mathrm { F 1 6 ~ F 1 6 ~ F 1 6 }$ </td><td>67.2</td><td>64.6</td><td>64.4</td><td>61.7 2007</td></tr><tr><td>F16 Q8 F16</td><td>67.7</td><td>65.6</td><td>64.7</td><td>59.7 1935</td></tr><tr><td> ${ \mathrm { Q 8 ~ f f n \_ a l l { = } F 1 6 } }$ </td><td>72.3</td><td>73.7</td><td>70.7</td><td>65.5 1652</td></tr><tr><td> $\mathrm { F 1 6 ~ Q 8 ~ Q 8 }$ </td><td>74.0</td><td>73.9</td><td>72.6</td><td>67.5 1652</td></tr><tr><td>F16 F16 Q8</td><td>74.4</td><td>70.4</td><td>71.2</td><td>68.6 1723</td></tr><tr><td>Q8 Q8 F16</td><td>75.9</td><td>82.6</td><td>78.1</td><td>70.5 1352</td></tr><tr><td>Q8 ffn_gate+up=F16</td><td>78.5</td><td>78.3</td><td>70.7</td><td>71.8 1457</td></tr><tr><td> $\mathrm { Q 8 ~ F 1 6 ~ F 1 6 }$ </td><td>80.6</td><td>74.3</td><td>76.8</td><td>73.8 1424</td></tr><tr><td>Late blocks F16</td><td>81.9</td><td>78.8</td><td>77.7</td><td>74.5 1396</td></tr><tr><td>Early blocks F16</td><td>82.1</td><td>77.6</td><td>78.1</td><td>74.6 1396</td></tr><tr><td> $\mathrm { Q 8 \ f f n \_ d o w n { = } F 1 6 }$ </td><td>82.2</td><td>89.0</td><td>84.7</td><td>74.7 1263</td></tr><tr><td> $\mathrm { Q 8 ~ a t t n \_ q + o u t { = } F 1 6 }$ </td><td>84.3</td><td>92.3</td><td>87.4</td><td>78.7 1127</td></tr><tr><td> $\mathrm { Q 8 ~ F 1 6 ~ Q 8 }$ </td><td>92.2</td><td>93.1</td><td>80.9</td><td>84.0 1141</td></tr><tr><td> $\mathrm { Q 8 ~ a t t n \_ q { = } F 1 6 }$ </td><td>92.5</td><td>92.5</td><td>84.0 83.0</td><td>1127</td></tr><tr><td> $\mathrm { Q 8 ~ a t t n \_ k v + o u t = F 1 6 }$ </td><td>92.8</td><td>89.7</td><td>86.2</td><td>84.7 1112</td></tr><tr><td> $\mathrm { Q 8 ~ a t t n \_ o u t { = } F 1 6 }$ </td><td>93.0</td><td>93.7</td><td>81.9</td><td>84.2 1098</td></tr><tr><td> $\mathrm { Q 8 ~ a t t n \_ k v { = } F 1 6 }$ </td><td>93.2</td><td>90.8</td><td>90.5</td><td>85.7 1084</td></tr><tr><td> $\mathrm { Q 8 ~ Q 8 ~ Q 8 }$ </td><td>95.3</td><td>88.3</td><td>90.1</td><td>86.2 1069</td></tr></table>

The results obtained justify the selection of these blocks as priority optimization targets. Thus, the search for a balance between speed gain and accuracy preservation should be focused on these segments, which served as the direct premise for applying modification metrics to them.

Analytical latency metrics. This work uses a simulation of uniform quantization – an activation-quantization metric that also accounts for sensitivity, i.e., Sensitivity-Aware PTQ (SA-PTQ) – which moves the optimization process from the realm of guesswork into the domain of precise computation and energy-based signal analysis via the SQNR coeficient. This also forgoes labor-intensive latency profiling in favor of an analytical hardware-cost metric of the layers $( L _ { F F N }$ $L _ { E m b } )$ . The integrated selectivity metric $S _ { n }$ formed in the course of the study reveals the “informa tion endurance” of each block, providing maximum speed gain under strict quality control.

## 2.1 Theoretical Basis for Layer Sensitivity Estimation

The optimization problem for language models (sLLMs) is posed by their parameter set’s information density: unlike huge models, small ones have little or no redundancy, and even minor distortions produce significant accuracy drops. The traditional Quantization-Aware Training (QAT) is prohibitively expensive for sLLMs due to requiring substantial resources for quantization [8, 12], while the canonical Post-Training Quantization $\left( \mathrm { P T Q } \right)$ methods [1, 9, 10, 13, 17, 19], reviewed in detail in [14], apply “blind” uniform compression, destroying the connectivity structure. In this work, we propose the Sensitivity-Aware PTQ metric that evaluates a layer’s information “endurance” by quantizing its parameters and measuring the output vector’s norm deviation. By avoiding the calculation of the Hessian matrix, required for Hessian-aware mixed-precision methods [15] and too expensive for realistic settings, the sensitivity estimation step transforms the model into a heterogeneous structure in terms of block-wise bit-width, providing excellent speed gains while maintaining

the quality within the given constraints.

The choice of PTQ as the basis for our method is motivated by the need for quick adaptation of the model to the target hardware without additional training [11]. Unlike traditional $\mathrm { P T Q }$ , which assumes uniform weight compression and is optimized to bound the parameter error, our SA-PTQ focuses on the output signal, enabling the analytical treatment of the degradation, which is crucial for the sensitivity analysis of sLLMs.

The SA-PTQ algorithm is a mathematical function that converts a smoothly varying number into a stepped approximation and vice versa. This procedure involves two operations: Quantization and Dequantization.

The step width calculation: the distance between grid points is computed based on the number of bits used to represent the numbers. For b bits $( \mathrm { e . g . } , b = 4 )$ , the grid has $2 ^ { b } - 1$ intervals: For ∆-bit precision $( \mathrm { e . g . , ~ } b )$ , the value range of $b = 4$ is divided into $2 ^ { b } - 1$ equal intervals (i.e., for 4 bits, the range would be divided into 15 equal segments):

$$
\Delta = { \frac { \operatorname* { m a x } ( x ) - \operatorname* { m i n } ( x ) } { 2 ^ { b } - 1 } }\tag{3}
$$

The step (4) serves as the unit of measurement for the entire subsequent process. Knowing it, we can derive the complete SA-PTQ mapping, which describes the number-transformation cycle:

$$
\mathrm { S A - P T Q } ( x , b ) = \mathrm { r o u n d } \left( { \frac { x - \operatorname* { m i n } ( x ) } { \Delta } } \right) \cdot \Delta + \operatorname* { m i n } ( x )\tag{4}
$$

Here, in the expression, the round function discards insignificant signal components, leaving only the informational skeleton corresponding to the chosen bit width. Depending on the data type inside the transformer architecture, this base logic is adapted for specific tasks.

## 2.1.1 Formula variant for FFN (weight quantization)

The FFN block (also referred to as MLP in some architectures, hence the MLP subscript used in the formulas below) in the transformer architecture is traditionally viewed as the model’s “memory”: it is where the factual information learned by the model during training is stored. In the Gemma 3 architecture, the fully connected block is implemented as a SwiGLU variant [18]: instead of a standard two-layer perceptron, the output of the block is given by the element-wise product of two parallel projections, one of which is passed through a nonlinearity. While such a construction is more expressive than a standard perceptron, it becomes more challenging to analyze its sensitivity to individual matrix transformations: modifying any of the three matrices $( W _ { g a t e } , W _ { u p } , W _ { d o w n } )$ involved introduces non-linear distortions to the output of the FFN block.

From the point of view of quantization, the FFN block is also of particular interest: the weights of this block comprise the majority of the model’s weights, and their compression directly impacts the speed of model inference. At the same time, the hypothesis about the uniformity of sensitivity is by no means obvious: the behavior of early and late layers in the model can be significantly diferent.

The weights of the FFN layer are quantized using the SA-PTQ operator (4). For the per-channel scheme, the parameters $\Delta$ and min(W) are computed for each row of the matrix independently, thus minimizing the expected quantization error (5).

$$
\widehat { W } = \mathrm { r o u n d } \left( \frac { W _ { i } - \operatorname* { m i n } ( W _ { i } ) } { \Delta _ { i } } \right) \cdot \Delta _ { i } + \operatorname* { m i n } ( W _ { i } )\tag{5}
$$

where $\Delta _ { i } = \frac { \operatorname* { m a x } ( W _ { i } ) - \operatorname* { m i n } ( W _ { i } ) } { 2 ^ { b } - 1 }$ is the quantization step of row i. As a pessimistic baseline variant, a per-tensor scheme is considered, in which a single $\Delta$ is computed from the global maximum of the entire matrix.

To evaluate degradation at each n-th layer, the reference and distorted outputs of the FFN block are computed. The Gemma 3 architecture uses SiLU gating, so the output is formed as in (6).

$$
y _ { \mathrm { c l e a n } } = W _ { \mathrm { d o w n } } \cdot \left( \sigma ( W _ { \mathrm { g a t e } } h ) \cdot W _ { \mathrm { u p } } h \right) , \qquad y _ { \mathrm { d i r t y } } = \widehat { W } _ { \mathrm { d o w n } } \cdot \left( \sigma ( \widehat { W } _ { \mathrm { g a t e } } h ) \cdot \widehat { W } _ { \mathrm { u p } } h \right)\tag{6}
$$

where $h \in \mathbb { R } ^ { d }$ is the hidden vector of the last token, and $\sigma ( x ) = \frac { x } { 1 + e ^ { - x } }$ is the SiLU activation function. The resulting vectors are fed into the SQNR formula (7), forming a layer-wise sensitivity profile of the model with respect to FFN weight quantization. Because $y _ { c l e a n } ^ { ( n ) }$ and $y _ { d i r t y } ^ { ( n ) }$ are computed independently per layer, the result is a full layer-wise sensitivity profile, from which the quality drop for any subset of quantized layers can be predicted.

## 2.1.2 Formula variant for Embedding

The question of quantizing language models always involves identifying the blocks that allow for aggressive compression and, therefore, are not sensitive to precision and blocks that are extremely sensitive to precision loss and, therefore, should not be compressed. The Attention block traditionally falls into the second category since the attention mechanism demonstrates fragile behavior at low bit-width and insuficient compression eficiency. As a result, this block is typically kept at a higher precision. By contrast, the Embedding matrix is a good candidate for compression since its weights are fixed and independent from the input context. Moreover, given the large vocabulary size, the embedding matrix in Gemma 3 1B constitutes a considerable portion of the model’s weights. Thus, reducing its size would lead to significant memory savings and acceleration of the model. However, this hypothesis requires empirical verification: up to which bit-width can the embedding matrix be safely compressed without afecting the quality of generated tokens? This value can be identified by quantizing the embedding matrix and measuring the tokenization quality either by direct evaluation or some proxy metric that does not require full model quantization.

The output-layer (LM Head) weights are responsible for converting the hidden state of the last token into a logit distribution over the vocabulary, from which the next token is selected. Accordingly, LM Head weights directly impact the quality of generated text: their incorrect quantization would result in a significant degradation of the logit distribution. The current work evaluates the sensitivity of the LM Head weights to low-bitwidth quantization. Notably, this analysis is performed after model training, assuming that the LM Head weights are not fine-tuned during quantization. Accordingly, the impact of reduced precision on the output-layer performance can be estimated using two proxy metrics: SQNR, which characterizes the distortion of the logit vector, and the gap metric $\Delta _ { g a p }$

The output-layer weights W are quantized using the SA-PTQ operator (4), which is applied to the tensor directly, as in the case of fully connected layers (5). In the per-channel variant, the parameters $\Delta$ and min(W) are computed for each row of the weight matrix, thus allowing each row to have its own dynamic range and minimizing the quantization error (7).

$$
W _ { q } = \mathrm { r o u n d } \left( \frac { W _ { i } - \operatorname* { m i n } ( W _ { i } ) } { \Delta _ { i } } \right) \cdot \Delta _ { i } + \operatorname* { m i n } ( W _ { i } )\tag{7}
$$

where $\Delta _ { i } = \frac { \operatorname* { m a x } ( W _ { i } ) - \operatorname* { m i n } ( W _ { i } ) } { 2 ^ { b } - 1 }$ is the quantization step of row i. Since the weights of the

Embedding matrix are static, the parameters $\Delta _ { i }$ and min(W) are computed once and fixed for the entire inference session.

To evaluate the degradation of the output layer at each step of autoregressive generation (8), two logit vectors are computed – the reference and the distorted one:

$$
\begin{array} { r } { l _ { c l e a n } = W _ { F 1 6 } \cdot h _ { t } , \qquad l _ { d i r t y } = W _ { q } \cdot h _ { t } } \end{array}\tag{8}
$$

where $W _ { q }$ is the weight matrix after applying SA-PTQ, and $h _ { t } \in \mathbb { R } ^ { d }$ is the hidden vector of the last token at step t. Feeding $l _ { c l e a n }$ and $l _ { d i r t y }$ into the SQNR formula then gives a quantitative estimate of how much the entire logit distribution is distorted by quantizing the output-layer weights.

## 2.2 SQNR as a Universal Degradation Indicator

Within the proposed framework of activation-quantization metrics and $\mathrm { S A \mathrm { - } P T Q }$ , we decided to adopt the SQNR coeficient [2], which regards the compression process as the addition of noise to the information transmission channel. The main advantage of using SQNR is that the metric has a logarithmic scale and is based on energy considerations. Decibels allow to estimate distortions in both the large FFN weights and the small Embedding vectors with equal precision, while also providing a simple way to evaluate gradient of the compression ratio when reducing the bit-width (eg, from 16 to 8 bits). At the same time, SQNR is cheap to calculate for each layer, and its correlation with the final quality of generation is high, which makes it suitable for use within heterogeneous sLLMs, since these two properties are critical when selecting a metric.

The need for such a metric arises precisely from the fact that we are comparing two types of layers: FFNs and Embeddings. To do this, one needs a dimensionless coeficient that would indicate to what extent the information contained in these layers is distorted.

To do this, we use the SQNR metric (signal-to-quantization-noise ratio), which is conventionally measured in decibels. In this case, if the value of SQNR is high (eg, 40–50 dB), then this means that the layer is "not noticeable" after quantization, or, in other words, is not distorted. On the other hand, a value of SQNR below 20 dB indicates a significant loss of information during quantization. This property is very convenient for comparing diferent types of layers, since SQNR is a universal metric that quantifies the amount of information loss during compression, regardless of whether it is stored in vectors or tensors. The value of SQNR makes sense for any data, since it estimates the percentage of useful information in the compressed data, regardless of the initial size (whether it is millions of tokens or a hundredth of a token).

Thus, we use the SQNR (9) metric to estimate the informativeness of each layer. It calculates the ratio between the energy of information in a clean signal and the energy of information in a noisy signal. This metric helps to find the most informative layers of the neural network, which difer depending on the type of layer, within the limits of one common metric.

$$
S Q N R _ { M L P } ^ { ( n ) } = 1 0 \cdot \log _ { 1 0 } \left( \frac { \left\| y _ { c l e a n } ^ { ( n ) } \right\| _ { 2 } ^ { 2 } } { \left\| y _ { c l e a n } ^ { ( n ) } - y _ { d i r t y } ^ { ( n ) } \right\| _ { 2 } ^ { 2 } } \right)\tag{9}
$$

The SQNR calculation for FFN measures the impact of weight quantization on the transformation of the input vector, comparing the reference value $y _ { c l e a n } = x \cdot W$ with the distorted result $y _ { d i r t y } = x \cdot \mathrm { S A } \mathrm { - P T Q } ( W , b )$ to assess the layer’s robustness to compression, whereas the KV-cache analysis determines the direct degradation of vectors when written to memory, by comparing the original vector $y _ { c l e a n } = v$ with its quantized version $y _ { d i r t y } = \mathrm { S A } \ – \mathrm { P T Q } ( v , b )$ , in order to confirm context preservation and the uniformity of the data distribution.

For a more granular view, we compute SQNR<sup>(n)</sup> separately for each n-th transformer block, which pinpoints exactly which layers are most sensitive to precision loss.

## 3 Testing: Embedding Data

Two experiments were conducted (Figs. 2, 3). The first (top-5 tokens for a single prompt) allows a specific moment – a single token – to be observed, giving a detailed picture of which tokens compete and how their ranking changes after quantization. The second (diference + SQNR over 10 tokens) allows the dynamics of generation to be observed, averaging over several steps – a more statistically robust diference, showing how quantization afects the model’s confidence during generation.

As part of an initial analysis of the degradation of the output-layer weight matrix, we perform an ablation experiment comparing model predictions at diferent degrees of quantization against the reference precision F16 (Fig. 2). The “Top-1 match with F16” plot shows the ratio of test prompts where the winning token matched the reference model. The metric is the clearest indication of prediction degradation, as its value drops when further weight matrix reductions distort the model’s final prediction.

The weight-matrix MAE is the mean absolute error between the elements of the weight matrix of the reduced precision and the reference F16 matrix. This metric characterizes the deviation of the distribution of the weights at the matrix level before they are projected to logits, enabling a direct comparison between diferent schemes’ impact. The deviation of the value of the winning token for each prompt at each degree of reduction serves as another diagnostic metric. By plotting these values, we see “Top-1 token logit by prompt”, the logit of the winning token for each prompt, reduced to a certain precision, versus the F16 reference. The lower value of the logit is an indication of reduced potency of this particular token in the output distribution after reduction, which could afect its rank. Similarly, delta logit of the winning token is a useful metric for inspecting individual prompts. It is calculated as the value of the logit of the winning token for the reference minus the value of the logit of the winning token for the reduced precision model, for each prompt. A positive value signifies a reduction in the winning token’s potency after reduction compared to the reference, which can be a sign of prediction degradation. We analyze this value for each prompt individually, which allows us to rank prompts by their sensitivity to the reduction.

![](images/6e4d3f5e3634c2750f23a727477d44119da64bb867073d2965b7742aff31a589.jpg)  
Figure 2: Token ranking for a single prompt.

To visually assess the degradation of the output block at diferent quantization levels, three plots were constructed (Fig. 3), each reflecting a separate aspect of compression’s efect on prediction quality.

The first plot shows the average gap between the top-1 and top-2 token logits, averaged over several autoregressive generation steps. This metric characterizes the model’s confidence in choosing the next token: the larger the gap, the more robust the prediction is to external perturbations, including those introduced by quantization. The plot allows visual assessment of how well diferent bit widths preserve this confidence relative to the reference F16 model.

The second plot shows the error vector $\Delta _ { g a p }$ (10), the diference of the average gap between the reference and the quantized model for each test prompt. A positive value indicates a reduction in the gap between the winning token’s logits and its closest competitor, which indicates degradation of the model’s confidence in choosing the next token. A negative value means an increase in this diference due to the stochastic nature of quantization error and cannot be interpreted as an improvement in prediction quality. The plot makes it possible to identify prompts most vulnerable to compression and to assess the stability of degradation across diferent input contexts.

$$
\Delta _ { g a p } = g a p _ { F 1 6 } - g a p _ { Q }\tag{10}
$$

The third plot shows the SQNR of the logits for each prompt. Unlike the gap metric, which focuses exclusively on the two extreme values of the distribution, SQNR evaluates the distortion of the entire logit vector. Jointly examining SQNR and $\Delta _ { g a p }$ makes it possible to distinguish two

degradation scenarios: uniform distortion of the whole distribution versus a local shift in the region of candidate tokens that directly afects the final choice (Fig. 3).

![](images/57a66e30f9cc0059e1198ae8e3d814ee60554f7e19799b6988f156406fd12606.jpg)

![](images/c3ae081d84f387dfd371ed9400aadb557dfd5952a27799e4f32ff22ecc0d7bb0.jpg)

![](images/6660b20ea612b951b7395e8d59d55cde7a852726d6ba052f6c8abf63e98ca263.jpg)  
Figure 3: Diference and SQNR over 10 tokens.

For the subsequent use of SQNR in the generalized metric, in order to obtain a quantizationeficiency score, SQNR is normalized according to formula (11).

$$
S Q N R _ { n o r m } = 1 - \frac { \log _ { 1 0 } ( 2 ) } { \log _ { 1 0 } ( S Q N R _ { Q 8 } ) }\tag{11}
$$

The mean SQNR value for Q8-ch was 45.61 dB, giving a normalized score of 0.819. The value for Q16 (93.82 dB) serves as the reference and confirms that at full precision there is no distortion. The normalized value $Q _ { e m b } = \mathbf { 0 . 8 1 9 }$ reflects the information preservation of the output layer after quantization and is subsequently combined with the speed component into the unified metric.

The same normalization formula can also be applied not only to the SQNR value averaged over the entire block but also to each layer independently. Since SA-PTQ probing is performed for each layer separately, the natural output is not a single number but a vector $\{ Q ^ { ( 1 ) } , \bar { Q ^ { ( 2 ) } } , . . . , Q ^ { ( L ) } \}$ , where each element characterizes the information preservation of a specific layer. This opens up the possibility of layer-wise ranking, in which layers with high $Q ^ { ( n ) }$ can tolerate aggressive compression, while layers with low $Q ^ { ( n ) }$ require preserving higher precision. Thus, the proposed metric scales from a block-level estimate to a full layer-wise model profile without changing the formula itself, simply by adjusting the granularity of the input data.

## 4 Testing: FFN Data

Several tests were carried out, first, to obtain data for each FFN block (gate, up, and down) for analysis. Testing predicted the quality loss of the FFN at 8-4 bits, obtaining SQNR for each of its components and layers.

Two experiments were carried out (Figs. 4, 7-11): one is a static weight analysis, yielding SQNR by projection and layer, SQNR by prompt, and weight MAE, which characterizes the general degradation picture for the given quantization scheme; the others visualize the dynamics of the autoregressive generation process by means of a token × layer heatmap, which makes it possible to see how the quantization afects each step of the generation in all layers of the model.

As part of the analysis of the degradation of the FFN block, a comparison of SQNR values was made for diferent levels of quantization relative to the baseline F16 (Fig. 4). The “SQNR comparison by layer - all configs and projections” plot characterizes the dynamics of SQNR for all three projections $( W _ { g a t e } , W _ { u p } , W _ { d o w n } )$ for each layer of the FFN block for all four configurations. This allows one to estimate roughly the vulnerability of each projection to quantization. The drop in SQNR values in the middle layers with a tendency to rise at the output layer is characteristic of all configurations. For the Q8-ch configuration, the $W _ { d o w n }$ projection demonstrated significantly lower SQNR values compared to $W _ { g a t e }$ and $W _ { u p } ,$ while for Q4-ch, all three projections showed SQNR values below 20 dB, which indicates the destructive efect of quantization with 4 bits per channel on the FFN block.

![](images/99e2bdea9f0a4c7505ddd1eee148eaab4e9c75ef43e1bebd9a935da93dd4daa4.jpg)  
Figure 4: Predicted degradation across bit widths for each component of Gemma 3 1B-it, relative to the mean.

The plot in Fig. 5 shows the mean SQNR value of the block’s final output, averaged over all 26 layers of the model, for each test prompt. The results demonstrate high homogeneity of degradation: SQNR values are practically independent of the semantic content of the input context and are determined solely by the quantization scheme. Q8-ch shows a mean value of 29.0 dB – a zone of noticeable but non-destructive degradation. Q4-ch, Q8-tensor, and Q4-tensor drop below 20 dB, with Q4-tensor showing values around 0 dB, corresponding to complete signal destruction.

![](images/6c9f95e2a3b7efbc13ef7b9d138dccca3198fffc63f5ab2747c1a0afcb18667d.jpg)  
Figure 5: FFN output SQNR by prompt.

The plot in Fig. 6 characterizes the degree of weight distortion at the level of the matrices themselves, prior to projection into activation space. Q8-ch shows the smallest error, whereas Q4-tensor gives an order-of-magnitude larger deviation. Notably, the $W _ { d o w n }$ projection consistently shows lower MAE relative to $W _ { g a t e }$ and $W _ { u p }$ across all configurations, which is explained by diferences in matrix dimensionality and weight distribution.

MLP Weight MAE by Configs & Projections  
![](images/8c785d2f931ba5d743b61bec7844412c1f32536ed91b229ec993abd71605828f.jpg)  
Figure 6: FFN weight MAE by configuration and projection.

To visually assess the dynamics of degradation during autoregressive generation, SQNR heatmaps were constructed in the token × layer space (Figs. 7–11). Each row corresponds to one generation step, each column to a layer number; color reflects the mean SQNR value across the three FFN projections.

![](images/625990b6018a39a1a4f4a453b5d44aa07a63fbf34881287e51eda9d766e40fae.jpg)

![](images/8cc188c9022f2904e91b57a1ac3915fd8b0a08b5cd327e3c6afdd9ab25e7f566.jpg)

![](images/56b7b26487432a93350659b5aa824b20c4ed2a2372bec60bf913c2d97e34a913.jpg)  
Figure 7: Per-layer/per-token prediction heatmap, geography-knowledge prompt.

![](images/2bd0029fcab38198cf31b7ee08d69e47ba78959a4af7fe07fc353348f38589e2.jpg)  
SQNR Heatmap: token × layer "In Python, to open a file you use"

![](images/6017899c3097b8d450b7fd79efa11d9f63dd11c6ce489948d732087951407901.jpg)  
Figure 8: Per-layer/per-token prediction heatmap, function-knowledge prompt.  
Figure 9: Per-layer/per-token prediction heatmap, historical-knowledge prompt.

![](images/dcb7b8d6c6220237c4b2df10f6dcc2b3d5f6d484df7c18e3c2d4c6f4dda08d6c.jpg)

![](images/38124a7d5c707ba538d0b9fa73d930e685e35b6fac0e07f7eb29e5e0d7fa8c39.jpg)  
Figure 10: Per-layer/per-token prediction heatmap, composition prompt.  
SQNR Heatmap: token × layer "def fibonacci(n): if n <= 1: return n r..."

![](images/eb8644ccf8a5e7c04125841e2065c275f9410dd1006e74eb48fdf9cddfd6e09f.jpg)

![](images/be16b5feed2fbe133937d13fa7ada54c727941357bb06bf1a16d803c93fad6bf.jpg)  
Figure 11: Per-layer/per-token prediction heatmap, code-reading prompt.

Across all five prompts (Figs. 7–11), a consistent pattern is observed: Q8-ch forms a gradient transition from yellow-green tones in the early layers to warmer tones in the middle layers, with partial recovery toward the last layer, which correlates with the layer-wise dynamics observed in the SQNR-by-layer plot. Q4-ch, in contrast, colors nearly the entire space red, showing SQNR values below 10 dB as early as the second or third layer. The dominant degradation factor is the layer index, and this pattern is reproduced across all prompts. At the same time, context type introduces local deviations: in technical and structured prompts (“In Python, to open a file you use”, “The theory of relativity was developed by”), individual generation tokens show anomalous SQNR values not observed in narrative contexts. This indicates that FFN block degradation is predominantly structural in nature, although the semantics of the input context can locally amplify its manifestations.

To quantitatively summarize the observed degradation picture and for subsequent use, the normalized SQNR score of the FFN block is computed using the same scheme as for the output layer (12).

$$
S Q N R _ { n o r m } = 1 - \frac { \log _ { 1 0 } ( 2 ) } { \log _ { 1 0 } ( S Q N R _ { Q 8 } ) }\tag{12}
$$

The mean SQNR value at Q8-ch for the $W _ { d o w n }$ projection was 29.70 dB – noticeably lower than the corresponding value for the embedding matrix (45.61 dB), which is consistent with the visual picture from the heatmaps: the FFN block as a whole is more sensitive (since the gate, up, and down blocks are all quantized) to quantization than the output layer. The reference value Q16 (77.95 dB) confirms that at the original precision degradation is minimal. The normalized score $Q _ { f f n } = \mathbf { 0 . 7 9 6 }$ reflects the higher sensitivity of the FFN to compression losses, subsequently allowing a score to be obtained based on the ratio between information loss and inference speedup.

## 5 Formalizing the Latency Parameters of FFN and Embedding (Speed Component)

To construct a general metric of quality-vs-speed dependence, the speed component must be taken into account. Since SQNR measures the ratio of signal energy to quantization-noise energy – i.e., the degree of distortion of the output vector after weight quantization – it is a purely informational metric that contains no information about speed. Speed is determined diferently: by the number of bytes the GPU must read from memory per generation step.

To combine speed and quality in a single metric, both indicators need to be expressed on a logarithmic (decibel) scale. However, direct comparison is impossible because of the diference in scale: the speed score and SQNR lie in fundamentally diferent value ranges. Normalization via the logarithm – which represents the theoretical maximum speedup for Q8 and is derived from the ratio of F16 and Q8 byte sizes – brings both metrics to a common range. After normalization, both indicators reflect a fraction of their physical limit and can be combined into a single quantizationeficiency metric.

## 5.1 Embedding

To answer the question of how quickly the GPU can read the weight matrix from memory, only the roofline model can help – it describes the performance of an operation through the ratio of computation to data volume. For memory-bound operations, this is suficient, while other, more complex profiling methods or empirical measurements are redundant.

Roofline, as an analytical tool, predicts the execution time of an operation based on two physical constraints: memory bandwidth and GPU compute power. Since multiplying the embedding matrix by the hidden-state vector is memory-bound, execution time comes down entirely to memory bandwidth – which means token time for any weight bit width can be computed analytically, from architectural parameters and hardware specs alone, without running the model at all. Applying this principle to the transformer architecture, the execution time of any layer is expressed through the size of its weights and the GPU’s memory bandwidth (13).

$$
t _ { l a y e r } = \frac { W _ { b y t e s } } { B W }\tag{13}
$$

where $W _ { b y t e s }$ is the weight size in bytes and BW is the GPU’s memory bandwidth (320 GB/s for the T4). The formula holds for memory-bound operations: the arithmetic intensity AI = 1 FLOP/byte is significantly below the ridge point of 203 FLOP/byte, hence execution time is determined exclusively by the speed of reading weights from memory. Applying this dependency to the lm\_head matrix gives its execution time for F16 and Q8 separately (14).

$$
t _ { l m } ^ { F 1 6 } = \frac { V \cdot H \cdot 2 } { B W } , \qquad t _ { l m } ^ { Q 8 } = \frac { V \cdot H \cdot 1 } { B W }\tag{14}
$$

where V is the vocabulary size and H is the hidden-layer dimension. Under Q8 quantization, each matrix element occupies 1 byte instead of 2, which halves the volume of data read and proportionally

speeds up the operation. Knowing the lm\_head time for both bit widths, the total time to generate one token is the sum of the time for all transformer blocks and the time for the output layer (15).

$$
t _ { t o k } ^ { F 1 6 } = L \cdot t _ { l a y e r } + t _ { l m } ^ { F 1 6 } , \qquad t _ { t o k } ^ { Q 8 } = L \cdot t _ { l a y e r } + t _ { l m } ^ { Q 8 }\tag{15}
$$

where L is the number of transformer blocks and $t _ { l a y e r }$ is the total time of all operations of one block except lm\_head. When only the embedding matrix is quantized, block time remains unchanged and only the lm\_head time changes. The ratio of times for the two configurations gives the speedup coeficient (16).

$$
\eta _ { e m b } = \frac { t _ { t o k } ^ { F 1 6 } } { t _ { t o k } ^ { Q 8 } } = \frac { L \cdot t _ { l a y e r } + \frac { 2 V H } { B W } } { L \cdot t _ { l a y e r } + \frac { V H } { B W } }\tag{16}
$$

where η is the speedup coeficient; the memory bandwidth BW cancels out, and the speedup is expressed purely through the model’s architectural parameters, independent of specific hardware characteristics. When tested on the Qwen 0.5B, Qwen 1.5B, and Gemma 3 1B models, the mean error between predicted and actual results was approximately 4%.

To verify the proposed prediction method, a series of experiments was conducted on the Gemma 3 1B-it model using Google Colab (NVIDIA T4 GPU) as the runtime environment. Embeddingmatrix quantization was performed via LLM-Viewer and llama.cpp with the $\mathrm { Q 8 \_ 0 }$ format, while all other model components were kept in F16. The actual speedup was measured as the ratio of the decoding speed of the quantized configuration to that of the baseline F16 model.

The results are presented in Table 2. The analytically predicted token time for Q8 was 5.37 ms versus an actual 5.51 ms, corresponding to a prediction error of 2.6%. The predicted speedup of ×1.178 matched the actual ×1.148 within the acceptable margin of error. After normalization via $\log _ { 1 0 } ( 2 )$ , the speed score was $S _ { e m b } = \mathbf { 0 . 2 3 6 }$ , reflecting the fraction of the theoretical Q8 speedup potential that was actually realized.

Table 2: Prediction error relative to actual speedup for Gemma 3 1B-it (Embedding).
<table><tr><td>Layers 26</td></tr><tr><td>lm_head F16 1.900 ms</td></tr><tr><td>lm_head Q8 0.945 ms</td></tr><tr><td>Token F16 6.33 ms (158.0 tok/s)</td></tr><tr><td>Token Q8 (predicted) 5.37 ms (186.1 tok/s)</td></tr><tr><td>Actual speedup ×1.148, 5.51 ms (181.3 tok/s)</td></tr><tr><td>Error 2.6%</td></tr></table>

The normalized speed score corresponds to the fraction of the theoretical Q8 speedup potential that was realized. Together with the quality component, this value forms the basis for building the unified metric.

## 5.2 FFN

For the FFN, an anchor F16 measurement is used – a single real F16 measurement via llama-bench, which serves as the reference point for the prediction. The reasoning is that the roofline model theoretically predicts the diference between F16 and Q8 $\left( \Delta _ { F F N } \right)$ , but not the absolute token time – there is GPU load, the scheduler, CUDA graphs, and other factors that roofline does not account for. Since the FFN block is a heavy operation, its execution time is fully determined by the volume of weights read and by memory bandwidth (17, 18).

$$
t _ { F F N } ^ { F 1 6 } = \frac { W _ { F F N } } { B W } , \qquad t _ { F F N } ^ { Q 8 } = \frac { W _ { F F N } / 2 } { B W }\tag{17}
$$

where $W _ { F F N } = 3 \cdot H \cdot I \cdot 2$ bytes (gate, up, down projections). The coeficient 3 reflects the three matrices of the SwiGLU block, and the multiplier 2 is the size of one element in F16 format, in bytes. When moving to Q8, each element occupies 1 byte, so the volume of data read is halved and operation time drops proportionally. Knowing the times for both bit widths, we can compute the theoretical saving for a single block and extend it to all L transformer blocks of the model (19).

$$
\Delta _ { F F N } = \left( t _ { F F N } ^ { F 1 6 } - t _ { F F N } ^ { Q 8 } \right) \cdot L\tag{18}
$$

The remaining components of each block – the attention mechanism, normalizations, and residual connections – remain in F16 and are not included in the saving computation. To move from theoretical savings to an absolute prediction, a real anchor point is needed. It is obtained as the reciprocal of the measured generation speed of the F16 model (20).

$$
T _ { r e a l } ^ { F 1 6 } = \frac { 1 } { \mathrm { t o k } / \mathrm { s } _ { F 1 6 } }\tag{19}
$$

This measurement absorbs all the system overheads that roofline does not model. Given the real token time and the theoretically predicted saving, the predicted Q8 token time is obtained by subtraction (21):

$$
T _ { p r e d } ^ { Q 8 } = T _ { r e a l } ^ { F 1 6 } - \Delta _ { F F N }\tag{20}
$$

This approach preserves the accuracy of the absolute measurement while not requiring the model to be re-run for each configuration. The ratio of the original time to the predicted time gives a dimensionless speedup coeficient (22):

$$
\eta _ { f f n } = \frac { T _ { r e a l } ^ { F 1 6 } } { T _ { p r e d } ^ { Q 8 } }\tag{21}
$$

To evaluate the proposed prediction method, experiments were carried out on three models: Gemma 3 1B-it, LLaMA 3.2 1B, and Qwen 2.5 1.5B, using Google Colab (NVIDIA T4 GPU) as the runtime environment. For each model, the saving $\Delta _ { F F N }$ was calculated analytically from the LLM-Viewer data, and the predicted speedup was compared to the actual value obtained by benchmarking with llama-bench, with only the FFN blocks being quantized to int8 and the rest of the model left in float16.

The architectural parameters and roofline-computation results for each of the three models are given in Table 3. For Gemma 3 1B-it, the analytical saving is 1.94 ms, which would reduce the 12.53 ms per-token time to 10.59 ms. On the other hand, while LLaMA 3.2 1B has fewer blocks (16 vs. 26), its per-block saving is larger due to the higher FFN matrix width, 157.5 µs vs. 74.7 µs for Gemma 3 1B-it, for a total of 2.52 ms. Finally, with 28 blocks and the largest per-block weight volume, Qwen 2.5 1.5B has the absolute maximum saving of 3.61 ms, reducing the 17.53 ms per-token time to 13.92 ms.

Table 3: Architectural parameters and roofline results at test time for three models.
<table><tr><td></td><td>gemma-3-1b-it</td><td>llama3.2-1b</td><td>qwen2.5-1.5b</td></tr><tr><td>Layers</td><td>26</td><td>16</td><td>28</td></tr><tr><td>FFN F16 (1 block)</td><td> $1 4 9 . 4 \mu s \times 2 6 = 3 . 8 8$  ms</td><td> $3 1 4 . 7 \mu s \times 1 6 = 5 . 0 4$  ms</td><td> $2 5 8 . 3 \mu s \times 2 8 = 7 . 2 3$  ms</td></tr><tr><td>FFN Q8 (1 block)</td><td> $7 4 . 7 \mu s \times 2 6 = 1 . 9 4$  ms</td><td> $1 5 7 . 5 \mu s \times 1 6 = 2 . 5 2$  ms</td><td> $1 2 9 . 3 \mu s \times 2 8 = 3 . 6 2$  ms</td></tr><tr><td> $\Delta _ { F F N } ~ [ \mathrm { a n a l y t i c a l } )$ </td><td>1.94 ms</td><td>2.52 ms</td><td>3.61 ms</td></tr><tr><td> $T _ { r e a l } ^ { F 1 6 }$ </td><td>12.53 ms (79.8 tok/s)</td><td>11.57 ms (86.4 tok/s)</td><td>17.53 ms (57.0 tok/s)</td></tr><tr><td> $T _ { p r e d } ~ \mathrm { ( h y b r i d ) }$ </td><td>10.59 ms</td><td>9.05 ms</td><td>13.92 ms</td></tr></table>

The comparison between predicted and actual speedup coeficients is given in Table 4. The best match was achieved for Gemma 3 1B-it, with an error of 0.7%; for LLaMA 3.2 1B the error was 3.3%. The largest deviation was observed for Qwen 2.5 1.5B, at 6.3%, which may be explained by FFN architectural specifics and CUDA-scheduler behavior for this model. The mean error across the three models was 3.4%, confirming the applicability of the hybrid approach for predicting speedup without re-running the model. After normalization, the FFN speed score was 0.261.

Table 4: Prediction error relative to actual speedup (FFN).
<table><tr><td>Model</td><td>Predicted</td><td>Actual</td><td>Error</td></tr><tr><td>gemma-3-1b-it</td><td>1.183</td><td>1.175</td><td>0.7%</td></tr><tr><td>llama3.2-1b</td><td>1.278</td><td>1.236</td><td>3.3%</td></tr><tr><td>qwen2.5-1.5b</td><td>1.259</td><td>1.344</td><td>6.3%</td></tr></table>

For subsequent use, the speedup coeficient is converted into a normalized scale via $\log _ { 1 0 } ( 2 )$ ， where a value of one corresponds to the theoretical maximum speedup at a two-fold reduction in weight volume. For Gemma 3 1B-it, the predicted speedup of ×1.199 gives a normalized FFN speed score of $S _ { f f n } = \mathbf { 0 . 2 6 1 }$

The presented approach admits a natural generalization to the case of layer-wise heterogeneous quantization, where each of the L layers receives its own bit width $b _ { n } .$ . Since all Gemma 3 layers are architecturally identical, the execution time of one layer at bit width $b _ { n }$ is computed analytically via roofline (23).

$$
t _ { F F N , n } ^ { b _ { n } } = \frac { W _ { F F N } \cdot \frac { 1 6 } { b _ { n } } } { B W }\tag{22}
$$

where the factor $\frac { 1 6 } { b _ { n } }$ reflects the reduction in the volume of data read relative to F16. The total saving over all layers is defined as (24)

$$
\Delta _ { F F N } = \sum _ { n = 1 } ^ { L } \left( t _ { F F N , n } ^ { F 1 6 } - t _ { F F N , n } ^ { b _ { n } } \right)\tag{23}
$$

The predicted token time is computed by the same scheme as before. Since the architectural parameters of all 26 layers of Gemma 3 are known, the theoretical calculation for any bit-width configuration can likewise be performed analytically without additional measurements.

Considering a configuration in which the middle 10 layers – the most sensitive according to the SQNR heatmap data – are kept at Q8, while the first 8 and last 8 layers, being more robust, are quantized to Q4 (25, 26):

$$
\Delta _ { F F N } = 8 \cdot \left( t _ { F F N } ^ { F 1 6 } - t _ { F F N } ^ { Q 4 } \right) + 1 0 \cdot \left( t _ { F F N } ^ { F 1 6 } - t _ { F F N } ^ { Q 8 } \right) + 8 \cdot \left( t _ { F F N } ^ { F 1 6 } - t _ { F F N } ^ { Q 4 } \right)\tag{24}
$$

At a baseline token time of $t _ { F F N } ^ { F 1 6 } = 1 4 9 . 4 \mu s , t _ { F F N } ^ { Q 8 } = 7 4 . 7 \mu s , t _ { F F N } ^ { Q 4 } \approx 3 7 . 4 \mu s !$

$$
\Delta _ { F F N } = 1 6 \cdot 1 1 2 . 0 + 1 0 \cdot 7 4 . 7 = 1 7 9 2 . 0 + 7 4 7 . 0 = 2 5 3 9 . 0 \mu s \approx 2 . 5 4 \ \mathrm { m s }\tag{25}
$$

At a baseline token time of $T _ { r e a l } ^ { F 1 6 } = 1 2 . 5 3$ ms, the predicted time is $1 2 . 5 3 - 2 . 5 4 = 9 . 9 9$ ms, corresponding to a speedup of $\eta \approx 1 . 2 5 4$ versus $\eta \ : = \ : 1 . 1 8 3$ for uniform Q8. Layer-wise heterogeneity theoretically gives an additional speedup of about 6% relative to block-level Q8, with the sensitive middle layers retaining Q8 precision while the robust outer layers are more aggressively quantized. However, it is possible that under layer-wise heterogeneous quantization the overhead of dequantizing weights when loading them into compute cores may difer for Q4 and Q8, potentially introducing an additional prediction error. Practical verification of this result requires tools with fine-grained control over the computation graph, such as ExLlamaV3 or TensorRT-LLM, which were not available for the Gemma 3 architecture within the scope of this study.

## 6 The Unified Metric

The experiments conducted for measuring quality degradation and predicting inference speed yielded four normalized values: $Q _ { e m b } = 0 . 8 1 9 , Q _ { f f n } = 0 . 7 9 6 , S _ { e m b } = 0 . 2 3 6 , S _ { f f n } = 0 . 2 6 1$ . Each of them independently characterizes one aspect of quantizing a specific block and is brought to a common scale from zero to one. The task of the final metric is to combine these four numbers into a single estimate of a quantization configuration, while preserving the ability to control the balance between speed and quality depending on the requirements of a specific task. For this, a weighted average of the speed and quality components is introduced, with a priority parameter α (27).

$$
\mathrm { s c o r e } ( \alpha ) = ( 1 - \alpha ) \cdot \frac { 1 } { N } \sum _ { k } S _ { k } + \alpha \cdot \frac { 1 } { N } \sum _ { k } Q _ { k }\tag{26}
$$

where the summation runs over all evaluated blocks k. In the block-level variant, $k \in \{ e m b , f f n \}$ and $N = 2 ;$ in the layer-wise variant, $k \in \{ 1 , 2 , \ldots , L \}$ and $N = L$ . Here $S _ { k }$ is the normalized speed score of block k, reflecting the fraction of the theoretical speedup potential realized at a two-fold reduction in weight volume, and $Q _ { k }$ is the normalized SQNR score of block $k ,$ characterizing the information preservation of the output representations after quantization. The parameter $\alpha \in [ 0 , 1 ]$ sets the priority between speed and quality: at $\alpha = 0$ the metric reflects purely the speed potential, at $\alpha = 1$ purely the information preservation, and at $\alpha = 0 . 5$ both aspects contribute equally. All components are normalized on a unified scale, which allows quantization configurations to be directly compared both at the block level and at the level of individual layers, without reference to absolute latency values or decibels.

At $\alpha = 0 . 5$ , for the Q8 configuration on the Gemma 3 1B-it model, the final score was score = 0.497, where the speed component equals 0.187 and the quality component equals 0.808.

## 7 Conclusion

This paper aims to produce a metric that quantifies the eficiency of the sLLM quantization, combining two criteria usually studied in isolation, namely, the information preserved in a given block, captured by the normalized SQNR, and the gain in the quantized model’s performance, captured by an analytical roofline latency model. Unlike prior work, where a significant computational budget was spent on either quantization-aware training or lengthy evolutionary searches over design space, the proposed metric is purely analytical, requiring no training for any of the candidate configurations.

Profiling revealed that FFN blocks and embedding contribute the most to the overall latency, with attention being surprisingly less costly, informing the choice of target blocks to optimize over. These blocks were scored according to the metrics, achieving a value of $Q _ { e m b } = 0 . 8 1 9$

and $Q _ { f f n } = 0 . 7 9 6$ for their quality retention score, and a value of $S _ { e m b } = 0 . 2 3 6$ and $S _ { f f n } = 0 . 2 6 1$ for their performance gain score, respectively. A sanity check on the performance gain scores across three model architectures resulted in an average error of approximately 3.4%, demonstrating the metric’s practical applicability.

Together with the $\alpha = 0 . 5$ , this allows us to score a particular configuration, achieving a value of 0.497 for the Q8 configuration for Gemma 3 1B-it, balancing the information preservation and performance eficiency. As demonstrated by the α , this metric can be adjusted to reflect the priority of either quality or performance, depending on the specific use case and system constraints. Finally, due to the metric’s independence from the granularity of input blocks, it can be applied to blocks, layers, or projections interchangeably, enabling fine-grained optimization at various levels. This property is particularly useful, as it allows one to estimate the quality drop and performance gain for a particular configuration ahead of time.

## References

[1] Hasan J. Optimizing Large Language Models through Quantization: A Comparative Analysis of PTQ and QAT Techniques // arXiv preprint. 2411.06084 – 2024. – https://arxiv.org/ abs/2411.06084.

[2] Castagnetti A., Pegatoquet A., Miramond B. Trainable quantization for Speedy Spiking Neural Networks // Frontiers in Neuroscience. – 2023. – Vol. 17. – Art. 1154241. – https://doi.org/ 10.3389/fnins.2023.1154241.

[3] Davies M. Eficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need // arXiv preprint. 2507.14397 – 2025. – https://arxiv.org/html/2507.14397v1.

[4] Gou D., et al. CARVQ: Corrective Adaptor with Group Residual Vector Quantization for LLM Embedding Compression // arXiv preprint. 2510.12721 – 2025. – https://arxiv.org/abs/ 2510.12721.

[5] Murshed M.G.S., et al. HAPM – Hardware Aware Pruning Method for CNN hardware accelerators in resource constrained devices // arXiv preprint. 2408.14055 – 2024. – https: //arxiv.org/abs/2408.14055.

[6] Zhang X., et al. IMPQ: Interaction-Aware Layerwise Mixed Precision Quantization for LLMs // arXiv preprint. 2509.15455 – 2025. – https://arxiv.org/html/2509.15455v1.

[7] Liu J., et al. 4-bit Reliable Quantization // arXiv preprint. 2501.13331 – 2025. – https: //arxiv.org/html/2501.13331v1.

[8] Liu Z., et al. EficientQAT: Eficient Quantization-Aware Training for Large Language Models // Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL). – 2024. – https://arxiv.org/abs/2407.11062.

[9] Frantar E., Ashkboos S., Hoefler T., Alistarh D. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers // International Conference on Learning Representations (ICLR). – 2023. – https://arxiv.org/abs/2210.17323.

[10] Lin J., et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration // Proceedings of Machine Learning and Systems (MLSys). – 2023. – https: //arxiv.org/abs/2306.00978.

[11] Liu Z., et al. LLM-QAT: Data-Free Quantization Aware Training for Large Language Models // arXiv preprint. 2305.17888 – 2023. – https://arxiv.org/abs/2305.17888.

[12] Han S., Mao H., Dally W.J. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Hufman Coding // International Conference on Learning Representations (ICLR). – 2016. – https://arxiv.org/abs/1510.00149.

[13] Jacob B., et al. Quantization and Training of Neural Networks for Eficient Integer-Arithmetic-Only Inference // IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). – 2018. – P. 2704–2713. – https://arxiv.org/abs/1712.05877.

[14] Gholami A., et al. A White Paper on Neural Network Quantization // arXiv preprint. 2106.08295 – 2021 (updated 2024). – https://arxiv.org/abs/2106.08295.

[15] Yao Z., et al. HAWQV3: Dyadic Neural Network Quantization // Proceedings of the 38th International Conference on Machine Learning (ICML). – 2021. – P. 11875–11886. – https: //arxiv.org/abs/2011.10680.

[16] Dao T., Fu D.Y., Ermon S., Rudra A., Ré C. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness // Advances in Neural Information Processing Systems (NeurIPS). – 2022. – Vol. 35. – P. 16344–16359. – https://arxiv.org/abs/2205.14135.

[17] Xiao G., et al. SmoothQuant: Accurate and Eficient Post-Training Quantization for Large Language Models // Proceedings of the 40th International Conference on Machine Learning (ICML). – 2023. – https://arxiv.org/abs/2211.10438.

[18] Shazeer N. GLU Variants Improve Transformer // arXiv preprint. 2002.05202 – 2020. – https: //arxiv.org/abs/2002.05202.

[19] Dettmers T., Lewis M., Belkada Y., Zettlemoyer L. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale // Advances in Neural Information Processing Systems (NeurIPS). – 2022. – Vol. 35. – P. 22128–22141. – https://arxiv.org/abs/2208.07339.

[20] Ainslie J., Lee-Thorp J., de Jong M., Zemlyanskiy Y., Lebrón F., Sanghai S. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints // arXiv preprint. 2305.13245 – 2023. – https://arxiv.org/abs/2305.13245.

[21] Shapley L.S. A Value for n-person Games // Contributions to the Theory of Games. – 1953. – Vol. 2, No. 28. – P. 307–317.

[22] Ganesh P., Chen Y., Lou X. et al. Compressing Large-Scale Transformer-Based Models: A Case Study on BERT // Transactions of the Association for Computational Linguistics. – 2021. – Vol. 9. – P. 1061–1080. – https://doi.org/10.1162/tacl\_a\_00413.

[23] Chu J., Leng Y., Li M., Shen Y., Shen X., Zhang Y. GEO-Flag: Detecting and Measuring GEO-Optimized Web Content // arXiv preprint. 2608.16824 – 2026. – https://arxiv.org/ abs/2608.16824.