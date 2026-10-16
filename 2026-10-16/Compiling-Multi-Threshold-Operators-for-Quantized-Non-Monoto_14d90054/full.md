# Compiling Multi-Threshold Operators for Quantized Non-Monotonic and Fine-Granular Activations in Custom-Tailored DNN Accelerators

Christoph Berganski

Felix Jentzsch

Marco Platzner

Computer Engineering Group

Computer Engineering Group

Paderborn University

Computer Engineering Group

Paderborn, Germany

Paderborn University

Paderborn University

christoph.berganski@upb.de

Paderborn, Germany

felix.jentzsch@upb.de

Paderborn, Germany

platzner@upb.de

Abstract—Aggressive quantization of neural networks can expose activation functions as bottlenecks for efficient hardware acceleration. Implementing quantized activations as multithreshold operations has been shown to be a promising optimization, allowing for efficient hardware implementation of fewbit activations. However, current implementations are limited to monotonically increasing functions, quantization schemes with one-dimensional (per-channel) granularity and a small set of predefined activation-quantizer combinations. We formalize the multi-threshold representation to extend it to non-monotonic functions and fine-granular parameters, i.e., multi-dimensional thresholds. For automated conversion of quantized function compositions into threshold representation, we propose a systematic approach via algebraic rules integrated into the ONNXbased compiler infrastructure. We experimentally validate our solution on a selection of monotonic and non-monotonic functions by extending the binary search based multi-threshold operator that is integrated into the FINN framework for end-to-end synthesis of custom-tailored FPGA accelerators. The cost of hardware support for non-monotonicity and fine-granularity is characterized in comparison to an alternative approach utilizing existing operators via graph-level decomposition. We compare our method to prior work as well as the approach taken by the hls4ml framework. As an outlook, we demonstrate the prospects of using the multi-threshold representation during the training of a neural network as a form of quantization aware training.

## I. INTRODUCTION

Quantization has emerged as the key model compression technique for optimizing deep neural network inference [1]– [3]. Custom FPGA-based accelerators particularly benefit from aggressive low-bit quantization, typically using 8-bit or lower precisions [3]–[9]. Streaming dataflow architectures exploiting pipelined model parallelism are well suited for such deployments, enabling efficient implementations of flexible quantization schemes. FINN [5], [6], [10] provides an end-toend compilation flow for models trained and quantized with PyTorch [11] and Brevitas [12], with optimizations expressed through ONNX graph transformations [13], [14].

Quantization methods can be categorized by when quantization is applied – post-training quantization (PTQ) or quantization-aware training (QAT) – and what is quantized, i.e., weights, activations, or both [1]–[3]. Early work focused on quantizing weights and multiply-accumulate (MAC) operations in fully connected, convolutional, and attention layers. As these reach increasingly low bit widths, optimizing activations and layer tails, i.e., chains of element-wise operations following a layer, such as normalization, activation, and quantization, becomes critical [15]–[17]. Simultaneously, preserving accuracy requires expressive scale factors, such as floating-point scales, and fine-granular quantization schemes [15], [18]–[22].

At very low bit widths, implementing quantized functions through successive thresholding, also known as multithreshold functions, has emerged as an effective optimization [5], [6], [15], [23]. In FINN, multi-threshold functions exactly represent quantized layer tail operations, including nonlinearities and floating-point scales, without costly floatingpoint inference. Here, exact means that no approximation beyond the inherent quantization error is introduced, unlike fixed-point approximations of floating-point values. Enabled by streamlining [15], [23], scales, biases, and non-linearities are absorbed into threshold values through algebraic transformations, reducing layer tail computations to threshold comparisons. In streaming dataflow architectures, these functions are specialized for each layer tail.

The multi-threshold implementation in FINN is limited to monotonically increasing functions and, in practice, supports only predefined activation–quantizer combinations, i.e, combinations of quantized identity and Relu. However, modern architectures employ diverse activation functions [24], [25], including non-monotonic variants such as Silu [26]. Moreover, FINN restricts quantization granularity to the final axis of the activation tensor. While SIRA [15] relaxes some of these constraints, it relies on expensive compile-time simulation of the layer tail. Consequently, we identify the lack of a systematic and lightweight compilation methodology for translating quantized layer tails into multi-threshold representation and corresponding hardware as a key barrier to a broader adoption.

To extend the flexibility and applicability of the multithreshold representation, we make the following contributions:

1) We formalize the multi-threshold representation, including its assumptions, constraints, and principles for effi-

cient hardware acceleration.

2) We extend the representation and hardware architecture to non-monotonic and fine-granular functions.

3) We introduce an ONNX-based compiler integration converting quantized functions into multi-threshold representation and fusing operators through algebraic rules.

4) We present an initial study of directly learning multithreshold representations as a form of QAT, aligning training-time representation with the target hardware.

The remainder of this paper is structured as follows: Section II introduces background and formalism of the multithreshold representation, Sections III and IV extend its applicability by addressing monotonicity and granularity limitations through corresponding hardware implementations. Section V presents our compiler integration for converting functions into multi-threshold representation. Section VI discusses related work and contrasts our approach. Section VII explores the prospects of learning multi-threshold representations during training. Section VIII concludes the paper.

## II. BACKGROUND AND FORMALISM

We define the multi-threshold function $f _ { \Theta , \delta } ( x )$ with N thresholds $\Theta _ { i }$ and associated step direction weights $\delta _ { i }$ as

$$
f _ { \Theta , \delta } ( x ) = b _ { \Theta } + s _ { \Theta } \sum _ { i = 0 } ^ { N - 1 } \delta _ { i } \cdot ( x \ge \Theta _ { i } ) ,\tag{1}
$$

where $s _ { \Theta }$ and $b _ { \Theta }$ are output scale and bias parameters. The thresholds $\Theta _ { i }$ can always be sorted in increasing order:

$$
\forall i \in [ 1 , N - 2 ] : \Theta _ { i - 1 } \leq \Theta _ { i } \leq \Theta _ { i + 1 }\tag{2}
$$

The representation can be seen as piecewise constant approximation of a function. If $\delta _ { i }$ are integers, this relates to a uniform quantizer $\mathbf { Q }$ where $s _ { \Theta }$ and b<sub>Θ</sub> correspond to quantization scale and bias, respectively:

$$
\mathbf { Q } ( x ) = s _ { \Theta } \cdot \mathrm { c l i p } \left( \left\lfloor { \frac { x - b _ { \Theta } } { s _ { \Theta } } } \right\rceil ; ~ 0 , ~ 2 ^ { \beta } - 1 \right) + b _ { \Theta }\tag{3}
$$

For static quantization, scale and bias are calibrated to the input range and remain fixed thereafter, and the quantization bitwidth $\beta$ is usually controlled as a hyperparameter.

FINN’s multi-threshold implementation [5], [6] has two restrictions. First, as $\delta _ { i }$ can take only positive unit steps, the resulting functions are necessarily monotonically increasing:

$$
\forall i : \delta _ { i } = + 1\tag{4}
$$

The relation between quantization bitwidth $\beta$ and the number of thresholds N follows as:

$$
\beta = \log _ { 2 } \left( N + 1 \right) .\tag{5}
$$

With this constraint, evaluating the sum in Eq. (1) can be reduced to locating the input $x$ within the sorted list of thresholds, implicitly setting $\Theta _ { - 1 } = - \infty$ and $\Theta _ { N } = + \infty$ to account for inputs outside the threshold range. This can be solved efficiently by binary search [15].

$$
f _ { \Theta , \delta } ( x ) = b _ { \Theta } + s _ { \Theta } ( i + 1 ) \quad \mathrm { s . t . ~ } \Theta _ { i } \leq x < \Theta _ { i + 1 }\tag{6}
$$

The second restriction of FINN’s multi-threshold implementation concerns the granularity of thresholds: Only per-tensor or per-channel thresholds are permitted, meaning the same set of thresholds is shared by the entire tensor or along slices per axis. The shape of the parameter tensor can be expressed as

$$
\mathbf { S h a p e \{ \ominus \} } \subseteq 1 \times 1 \times \cdots \times 1 \times C \times N \to C \times N ,\tag{7}
$$

restricting all leading dimensions to singletons of size 1, which can be omitted according to tensor broadcasting rules.

## III. NON-MONOTONICITY

In this section we explore the representation of nonmonotonic functions by relaxing the constraint on the step weights $\delta _ { i }$ to allow for arbitrary integer-valued weights:

$$
\forall i : \delta _ { i } \in \mathbb { Z }\tag{8}
$$

Assuming sorted thresholds, the monotonically increasing output levels of Eq. (6) are replaced by levels $L _ { i } ,$ , given as the precomputed cumulative sum over the step weights up until including the largest threshold smaller than the input:

$$
\begin{array} { c l } { { f _ { \Theta , \delta } ( x ) = b _ { \Theta } + s _ { \Theta } { \cal L } _ { i } ~ \mathrm { s . t . } ~ \Theta _ { i } \le x < \Theta _ { i + 1 } } } \\ { { } } & { { } } \\ { { { \cal L } _ { i } = \displaystyle \sum _ { j = 0 } ^ { i } \delta _ { i } , ~ \mathrm { w i t h } ~ { \cal L } _ { - 1 } = 0 } } \end{array}\tag{9}
$$

Where $L _ { - 1 }$ accounts for inputs outside the threshold range. The quantization bitwidth now relates to the levels $L _ { i }$ instead of the number of thresholds as

$$
\beta ^ { * } = \log _ { 2 } \left( \operatorname* { m a x } L _ { i } - \operatorname* { m i n } L _ { i } + 1 \right) .\tag{10}
$$

With the monotonicity constraint, Eq. (4), this reduces exactly to Eq. (5), where max $L _ { i } = N$ and min $L _ { i } = 0$ , confirming our generalization to be consistent.

Figure 1 sketches our hardware implementation as a remapping of the output, using the output of the threshold search as the address into the precomputed level memory. We have implemented this in Vitis HLS and integrated it into FINN.

As an alternative to the hardware implementation, we explore a graph-level decomposition for positive and negative unit steps, pulling apart increasing $\delta _ { i } = + 1$ and decreasing $\delta _ { i } = - 1$ parts of the sum in Eq. (1). This implements nonmonotonic functions as the difference of two multi-threshold functions, each of which is monotonically increasing and operates on a corresponding subset $\Theta ^ { + } / \Theta ^ { - }$ of the thresholds:

$$
f _ { \Theta , \delta } ( x ) = f _ { \Theta ^ { + } , \delta = 1 } ( x ) - f _ { \Theta ^ { - } , \delta = 1 } ( x )\tag{11}
$$

This reduces the complexity of the thresholding hardware at the cost of additional operators, which besides a second multithreshold and a subtraction, incurs dataflow overheads.

Figure 2 shows results comparing the hardware implementation to the graph-level decomposition in terms of lookup table (LUT) resource utilization. Our setup is as follows: Thresholds are drawn from a uniform random distribution between −4 and +4 and non-monotonic step direction weights are generated from a random walk on ±1. Thresholds, as well as the random output scale and bias are single-precision floating-point numbers. In a complete accelerator design, scale and bias would be absorbed into subsequent operations via the streamlining optimization. For our isolated test functions, however, these are implemented and together account for about 5k LUTs. The shape of the test input is 64 $\times \ 4 \times 4 \times 4$ where the leading 64 is the batch dimension. A per-tensor set of thresholds and weights is generated. Test functions are exported to ONNX and compiled via FINN+ [10] with PE = 4 parallel processing elements along the final axis, representing a more realistic setting for the FINN framework. The design is synthesized using Vivado 2024.2 targeting the Alveo U55C at 100 MHz. We do not allow DSP or BRAM resources to ensure comparability of results in terms of LUT resources. The design is verified in RTL simulation. We report resource utilization averaged over seven repetitions using varying random seeds.

![](images/21947a2c4969e0d1e81865e2c52f137341737698647d8ef03a70c5604c997596.jpg)  
Fig. 1: Hardware implementation of binary search based thresholding adapted from [15]. A remapping stage is added to allow for non-monotonic output levels $L _ { i }$ according to Eq. (9). To locate the input within the list of seven sorted thresholds, three comparators are required, corresponding to the three levels of the search tree shown on the left.

![](images/119a1bd5567a7508fb6da520c9086a6982636e2f54f7617eaf577f02e4cca2c9.jpg)  
Fig. 2: Comparison of the hardware implementation to the graph-level decomposition in terms of lookup table resource utilization for increasing number of thresholds N when implementing monotonic or non-monotonic functions.

Non-monotonic functions are more costly to implement than monotonic functions, though the difference is rather small for the direct hardware implementation and on the order of at most a few hundred LUTs. The graph-level decomposition clearly shows the cost of inserting additional operators, where the total LUT utilization is up to 1.5× compared to the hardware implementation. The largest contributions to this increase are the multi-thresholds with 1.6×, and dataflow overheads with 3.4× compared to the hardware implementation. Multiplication and addition operations together contribute between 4.8k and 5.5k LUTs, with a worst-case increase of about 1.05× compared to the hardware implementation, which is almost negligible.

## IV. FINE GRANULARITY

In this section we explore the representation of arbitrarily fine-granular threshold parameters. Fine-granularity is realized via broadcasting of the thresholds to the input shape where the shape of the threshold tensor is a subset of the input shape extended by a trailing dimension of N thresholds:

$$
\begin{array} { c } { \begin{array} { c } { \mathbf { S h a p e \left\{ \Theta \right\} } \subseteq \mathbf { S h a p e } \left\{ X \right\} \times N } \\ { \mathbf { S h a p e } \left\{ \Theta \right\} = D _ { 0 } \times D _ { 1 } \times \cdots \times D _ { n } \times C \times N } \end{array} } \end{array}\tag{12}
$$

Address logic maps an index I ∈ Shape {X} to the corresponding index $\bar { \mathrm { I } } ^ { \prime } ~ \in$ Shape {Θ} to retrieve the set of thresholds from the parameter memory:

$$
\mathbb { I } _ { i } ^ { \prime } = \mathbb { I } _ { i } \mod D _ { i }\tag{13}
$$

![](images/4a3ad477978e792bfefcca20d4f79e5c5597d6b75cfdc8a02f8fbab8778e2ffa.jpg)  
Fig. 3: Address logic and layout of the threshold parameter memory. The address logic and the thresholding units are replicated as PE parallel processing elements resulting in a two-level loop nest (i, j) implemented in Vitis HLS.

Memory interfaces and address calculations are realized in terms of flat indices i and i<sup>′</sup>, which relate to the multidimensional index via lexicographic ordering. The inverse, turning a flat index back into a multidimensional index, is referred to as unraveling. Even though we keep referring to the final axis as the channel axis as this has a special role in FINN, the same also applies to this axis where $C = D _ { n + 1 }$ . Figure 3 sketches the address logic and memory layout of our hardware design.

As an alternative to the hardware implementation, we explore a graph-level decomposition exploiting the property that an arbitrary value can be added to both sides of an equation without changing the outcome. Following broadcasting semantics, we choose this value such that the resulting thresholds on the right have the desired per-channel granularity:

$x \geq \Theta \equiv x + b \geq \Theta ^ { \prime }$ , where $b = \Theta ^ { \prime } - \Theta$ , and

$$
\mathbf { S h a p e } \{ \Theta ^ { \prime } \}  C \times N\tag{14}
$$

$$
\mathbf { S h a p e } \{ b \}  D _ { 0 } \times D _ { 1 } \times \cdot \cdot \cdot \times D _ { n } \times C
$$

This decomposes thresholds as a fine-granular addition followed by an at most per-channel (last axis) multi-threshold function. However, an exact solution is not always possible as this generally results in reduced parameter volume

$$
\| \mathbf { S h a p e } \left\{ b \right\} \| + \| \mathbf { S h a p e } \left\{ \Theta ^ { \prime } \right\} \| \leq \| \mathbf { S h a p e } \left\{ \Theta \right\} \| ,\tag{15}
$$

and we solve the following optimization problem to find the decomposition, which might introduce an approximation error:

$$
\Theta ^ { \prime } , b = \operatorname * { a r g m i n } _ { \Theta ^ { \prime } , b } \| ( \Theta - ( \Theta ^ { \prime } - b ) \| _ { 2 } ^ { 2 }\tag{16}
$$

Figure 4 compares the hardware implementation to the graph-level decomposition in terms of LUT resource utilization. The setup is the same as described in Section III, except for the following: Only monotonically increasing functions are considered, i.e., all step directions are 1, and instead the granularity is varied by generating random thresholds for varying combinations of axes, which we group by the number of axes covered. Average LUT utilization over seven random seeds per configuration is reported.

Fine-granular parameter tensors are more costly to imple ment. For per-tensor and per-channel (last axis) granularity there is no difference between the direct hardware implementation and the graph-level decomposition. For fine-granular thresholds at high bitwidths, these results clearly favor the graph-level decomposition with a worst-case increase of about 1.9× from per-tensor to 3-channel granularity. In contrast, the direct hardware implementation, with an increase of about 58×, approaches the increase naively expected due to the 64× growth of the parameter tensor. At this point, the direct hardware implementation becomes impractical, though deviations from this at lower bitwidths (e.g., at $\beta = 4$ , there is only about a 2.2× difference) hint at automatic logic optimizations of the synthesis tool, which might be possible to exploit at higher bitwidths via more explicit memory partitioning. At very low bitwidths $( \mathrm { i . e . , ~ } \beta < 4 )$ and moderate granularities (i.e., 1-2 channels), the direct hardware implementation might be preferable, as decomposition overheads become relevant.

## V. COMPILING FUNCTIONS TO THRESHOLDS

While FINN converts functions into multi-threshold representation via rules for fixed and predefined activation-quantizer patterns and SIRA simulates entire layer tails to extract the steps, we propose absorption rules based on a generalized notion of the inverse $f ^ { - 1 }$ of a function:

$$
f ( x ) \geq \Theta _ { i } \iff x \geq f ^ { - 1 } ( \Theta _ { i } )\tag{17}
$$

As an example, the generalized inverse of the common Relu activation function is given as:

$$
\operatorname { R e l u } ^ { - 1 } ( x ) = { \left\{ \begin{array} { l l } { x , } & { { \mathrm { i f ~ } } x \geq 0 } \\ { - \infty , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{18}
$$

These inverses need to be predefined for each function of interest, but they are not tied to a particular quantizer nor do they assume a fixed pattern of functions leading up to a quantizer. No costly simulation of the layer tail is required, keeping conversion runtime and memory footprint low, while reducing the susceptibility to numerical imprecision due to insufficient resolution when sampling the input range. Range information is used for enumerating the initial thresholds for the rounding function and clipping range of the quantizer,

$$
\mathbf { Q } \left( f ( x ) \right) = b _ { \Theta } + s _ { \Theta } \sum _ { i = 0 } ^ { 2 ^ { \beta } - 1 } \left( \frac { f ( x ) - b _ { \Theta } } { s _ { \Theta } } \geq \underbrace { \left( i + 0 . 5 + \Delta _ { i } \right) } _ { \triangleq \Theta _ { i } } \right) ,
$$

where $\Delta _ { i }$ is a round to even correction, which is Ulp(i) for odd positive and even negative i and zero otherwise. Ulp(x) is the unit in the last place, i.e., the spacing between x and the next representable (floating-point) number.

Application of Eq. (17) to a non-monotonic function with multiple (increasing and decreasing) branches needs to consider all branches as valid solutions, i.e., the set of thresholds branches into corresponding segments. To give intuition, consider the following example (also depicted in Figure 6):

![](images/25ba2bbba762257b8bf0d7e2ac524e61ebef760e5bfc433a260b2fd206b2fb67.jpg)  
Fig. 4: Comparison of the hardware implementation to the graph-level decomposition in terms of lookup table resource utilization for increasing number of thresholds N and number of channels covered by parameters. $T$ and $C$ refer to per-tensor and per-channel (last axis) granularity, for which there is no difference between the two approaches.

$$
x ^ { 2 } \geq a \iff x \geq + { \sqrt { a } } \lor x \leq - { \sqrt { a } }
$$

Decreasing branches flip the direction of comparison or, equivalently, invert the step direction, again making use of an Ulp-based correction term:

$$
x \geq a \vee x \leq b \equiv x \geq a \vee \neg ( x \geq b + \mathbf { U } \mathbf { l p } ( b ) )\tag{19}
$$

With $b ^ { \prime } = b { + } \mathbf { U } \mathbf { l p } ( b )$ , this is rewritten in Xor-form and applied repeatedly to absorb consecutive functions, resulting in a tree of branching threshold segments, which is collapsed to sorted thresholds and step direction weights as shown in Figure 5:

$$
x \geq a \vee \neg ( x \geq b ^ { \prime } ) \equiv \neg ( x \geq b ^ { \prime } ) \oplus ( x \geq \operatorname* { m a x } \{ a , b ^ { \prime } \} )\tag{20}
$$

![](images/79ed47adf26c7efa2334a3c50568699a3ab52085c9d614c8d5d469f14eb1f94b.jpg)  
Fig. 5: Tree of branching threshold segments: Interpreting this as a parity function, this is equivalent to applying alternating weights ±1 before summing over the sorted segment contributions. Sorting is always allowed due to commutativity.

We exemplarily derive the generalized inverse of the Silu activation function [26], which shows a non-monotonic bump to the left of zero. The Silu function is defined as follows, where $\sigma ( x )$ is the logistic sigmoid function:

$$
\mathrm { S i l u } ( x ) = x \cdot \sigma ( x )\tag{21}
$$

The inverse is expressed in terms of the real-valued principal and secondary branch of the Lambert W function [27]

$$
\mathrm { S i l u } _ { k } ^ { - 1 } ( y ) = y + W _ { k } ( y e ^ { - y } ) \mathrm { , ~ w h e r e ~ } k \in \{ 0 , - 1 \} \mathrm { , }\tag{22}
$$

which is valid for all $y$ above the global minimum $y _ { \mathrm { m i n } }$ of the Silu function, which is

$$
\begin{array} { l } { x _ { \mathrm { m i n } } = - 1 - W ( e ^ { - 1 } ) \approx - 1 . 2 8 } \\ { y _ { \mathrm { m i n } } = \mathrm { S i l u } ( x _ { \mathrm { m i n } } ) \approx - 0 . 2 8 . } \end{array}\tag{23}
$$

Figure 6 sketches Silu with the branches of its inverse, and, similarly, $x ^ { 2 }$ with both branches $\pm \sqrt { x }$ of the square root.

![](images/dfd86a0c7d17be7c8b5fff1450ae5dfbdaaf972a5cefc3c84394ff1b0bc12a13.jpg)  
Fig. 6: Inverse of non-monotonic functions. Branch points are at the extrema, separating increasing from decreasing segments. Colored dashed lines represent the result of applying the corresponding branch to the output of the function.

We integrate this into the compiler infrastructure as an ONNX function approximating the Lambert W function according to [28] with initial values according to [29]. Similarly, absorption rules, further generalized inverses, the $\mathbf { U l p } .$ , as well as the graph-level decompositions proposed in the previous sections are integrated into the compiler infrastructure<sup>1</sup> as ONNX functions and graph transformations.

Figure 8 shows the resource utilization for various monotonic and non-monotonic functions approximated with increasing quantization bitwidth $\beta ^ { * }$ . We demonstrate operator fusion via absorption by implementing the following non-monotonic function composition as a single multi-threshold function:

![](images/b9c59a13d12db4dde38f24ada61958314ff3aa816432b87e082aea0b57e17b12.jpg)  
Fig. 7: Non-monotonic test function composition at quantization bitwidths $\beta ^ { * } = 3$ and $\beta ^ { * } = 6$ . The function has two increasing and one decreasing segment with branch points at approximately (−0.126, 0.078) and exactly (2.4, 0.0).

$$
\mathbf { Q } \left( f ( x ) \right) = \mathbf { Q } \left( \mathrm { S i l u } ( 0 . 5 x - 1 . 2 ) ^ { 2 } \right)\tag{24}
$$

This function at two different quantization bitwidths is shown in Figure 7. The experimental setup is similar to Section III, except for the following: As there is no randomness in these functions, a single result is reported without averaging. Thresholds result from calibrating and converting a uniform quantizer on the functions evaluated over the −4 to +4 range (log is evaluated over $+ 0 . 0 1 \ \mathrm { t o } \ + 8 . 0 1 )$ . The number of thresholds now is a dependent variable and instead the quantization bitwidth $\beta ^ { * }$ , i.e., number output levels, is controlled.

Non-monotonic functions are on average more costly to implement. This can be primarily attributed to requiring more thresholds at the same quantization bitwidth, i.e., usually $\beta > \beta ^ { * }$ . Besides the raw number of thresholds, the concrete values have an influence on the resource utilization as well: functions with more uniform threshold distribution, such as the identity or Relu, require fewer LUTs at the same quantization bitwidth. Ordering functions by their average LUT utilization per output level matches an intuitive notion of non-uniformity.

## VI. RELATED WORK

In this section, we first qualitatively compare our work to prior approaches for compiling multi-threshold operators and then provide a quantitative comparison to hls4ml.

## A. Qualitative Comparison

Table I presents the qualitative comparison. Originally [5], [6], FINN provides a small set of fixed activation-quantizer patterns which are converted to a direct implementation of the summation realized in HLS. SIRA [15] proposes an automated compilation which works out of the box for any operator supported by ONNX Runtime and thus does not require operatorspecific extensions. Alongside, a more efficient binary search based implementation of the multi-threshold operator is introduced, which is available as an RTL backend in FINN. To avoid the high compilation effort of subgraph evaluation and search, we propose a semi-automatic compilation, where primitives are implemented manually but are combined automatically by following function composition along the layer tail to support more complex activation-quantizer patterns. We extend frontend and backend to support non-monotonic and

arbitrarily fine-granular functions. Furthermore, it is possible to use our frontend to target the SIRA backend via graph-level decomposition.
<table><tr><td></td><td>Original [5], [6]</td><td>SIRA [15]</td><td>Ours</td></tr><tr><td>Compilation</td><td></td><td></td><td>Semi-Automatic</td></tr><tr><td>Approach</td><td>Manual Fixed Patterns</td><td>Automatic Search-Based</td><td>Fusing Primitives</td></tr><tr><td>Effort</td><td>Low</td><td>High1</td><td>Low</td></tr><tr><td>Support</td><td></td><td></td><td></td></tr><tr><td>Monotonicity</td><td>Monotonic</td><td>Monotonic</td><td>Monotonic Non-Monotonic</td></tr><tr><td>Granularity</td><td>Per-Channel</td><td>Per-Channel</td><td>Arbitrary</td></tr><tr><td>Backend</td><td></td><td></td><td></td></tr><tr><td>Algorithm</td><td>Summation</td><td>Binary Search</td><td>Binary Search</td></tr><tr><td>Implementation</td><td>HLS</td><td>RTL</td><td>HLS/RTL²</td></tr><tr><td>Extensibility</td><td>Add Converter per Pattern</td><td>_3</td><td>Add Inverse per Primitive</td></tr></table>

1. Search space resolution trades off effort vs. compilation error  
2. SIRA backend with decomposition of non-monotonic/fine-granular functions  
3. Subgraph evaluation works out of the box for standard operators

TABLE I: Layer tail compilation approaches in FINN

## B. Quantitative Comparison

The hls4ml framework [8], [30], [31] employs a lookup table approach to implement a variety of activation functions. The input range is uniformly split into sections and function output values are tabulated to be retrieved at runtime. hls4ml does not offer operator fusion, such as the streamlining optimization [15], [23] in FINN. Rather, hls4ml implements each operator in a layer tail as a standalone hardware operator, which we refer to as composite approach.

To compare our approach to hls4ml as well as FINN extended by the SIRA [15] RTL backend, we use a setup similar to Section V, configuring hls4ml to generate a pipelined design with $\mathbb { I } \mathbb { I } ~ = ~ 1$ and a reuse factor matching our $\mathrm { { P E } } = \ 4$ parallelism. The results are presented in Figure 9. For simpler activation functions which are supported out of the box, hls4ml can implement the function with comparatively fewer resources, but at the cost of higher quantization errors, especially for few-bit quantization. We attribute this to the uniform sampling of the input range, whereas the non-uniformity of the multi-thresholds allows to better allocate the intervals to steeper regions of the function, which can also be seen in Figure 7. For more complex function compositions, the resource costs of the multi-threshold approach grow only moderately due to operator fusion. By streamlining a model graph, which delays the de-quantization scale and bias to the very end of the graph (dark portion of the bars in Figure 9), this has the potential to implement the entire quantized activation data-path in just a few hundred LUTs (at, for example, 4- bit quantization). In contrast, the composite approach taken by hls4ml becomes largely unattractive due to the stark increase in resource utilization. Comparing the two FINN implementations, the SIRA RTL backend is generally more efficient in terms of thresholds per LUT. However, as SIRA does not directly support implementation of non-monotonic functions, decomposition overheads start to dominate for more complex non-monotonic function compositions, making our HLS backend more attractive.

![](images/38b5a14a5ad4653a87e32788c7e658e5b0786f99948c2097c1ca8a3d21b4f7e6.jpg)

![](images/9f2d104c2e6bdb01ccd3054dcb7dd6fea837ed65301f48ab6ba7337edf3dfa0c.jpg)

![](images/4ef35c9df48baf2db060c42de5336dfc98f8d7a16fac1e00731aa45ef2997373.jpg)  
Quantization Bitwidth $\beta ^ { * }$  
Quantization Bitwidth $\beta ^ { * }$

![](images/807b89138270551562bdd809cb7dde4abccf637c30bc6cc06c16913d82dee684.jpg)

![](images/d5938d1e106f5337b51cbac7f269ce7abe4c3432df6bfd5c1e6eb6e615987f79.jpg)

![](images/2f52c7cee14dc376a414999e22124fb2c420b7d8a2a749e89908f58e6da014fc.jpg)  
Quantization Bitwidth $\beta ^ { * }$

![](images/882719d61fbf6ce947843998c9aa1044dbb0311ce5e024e1d7d7fc6bc3e4e049.jpg)  
Quantization Bitwidth $\beta ^ { * }$  
Fig. 8: Comparison of functions implemented as multi-threshold functions in terms of lookup table resource utilization over increasing quantization bitwidth $\beta ^ { * }$ . The thresholding bitwidth $\beta$ results from calibrating the quantizer to the function at the bitwidth $\beta ^ { * }$ . For monotonic functions $\beta = \beta ^ { * }$ , as indicated by the black horizontal bars. Non-monotonic functions might revisit the same output level, resulting in $\beta > \beta ^ { * }$ . Functions are ordered from top left to bottom right by average LUTs per output level. The quantization error (mean absolute error) decreases with the quantization bitwidth as indicated below each bar.

## VII. LEARNING THRESHOLDS

As an outlook, we explore the prospects of learning quantized activation functions directly in multi-threshold representation. This offers an option for training aggressively quantized neural networks from scratch, which avoids having to find the ideal activation function at the desired bitwidth, and, by making the quantization thresholds flexible, brings back some of the representational capacity usually lost due to lowbitwidth quantization. Thus, we can interpret learning multithreshold representations as a form of quantization-aware training. Learning of activation functions has been proposed before, usually by expressing activation functions as learnable polynomials [32], [33], rational functions [34] or splines [35], whereas we propose a linear combination of step functions.

We relax the non-differentiable comparison as a smooth sigmoid function σ centered at each threshold. The sharpness of the step can be controlled by a factor $\alpha ,$ which for $\alpha \gg 1$ closely approximates the comparison. For $\lVert \boldsymbol { w } _ { i } \rVert \approx 1$ the step weights $\delta _ { i } \in \{ - 1 , + 1 \}$ can be made differentiable via the straight through estimator trick: $\delta _ { i } = \mathtt { s i g n } ( w _ { i } ) \approx w _ { i }$ . Thus, to compute gradients during the backward pass, we treat the multi-threshold function as follows:

$$
f _ { \Theta , \delta } ( x ) \approx b _ { \Theta } + s _ { \Theta } \sum _ { i = 0 } ^ { N - 1 } w _ { i } \cdot \sigma \left( \alpha \cdot ( x - \Theta _ { i } ) \right)\tag{25}
$$

This yields the following partial derivatives with respect to the function input x and the function parameters $\Theta _ { j }$ and $w _ { j }$ where $\sigma _ { i } = \sigma \left( \alpha \left( x - \Theta _ { i } \right) \right)$ ):

$$
\frac { \partial f _ { \Theta , \delta } } { \partial x } \approx s _ { \Theta } \sum _ { i = 0 } ^ { N - 1 } \alpha w _ { i } \sigma _ { i } \left( 1 - \sigma _ { i } \right)\tag{26}
$$

$$
\frac { \partial f _ { \boldsymbol { \Theta } , \delta } } { \partial \boldsymbol { \Theta } _ { j } } \approx - \alpha w _ { j } \sigma _ { j } \left( 1 - \sigma _ { j } \right)\tag{27}
$$

$$
\frac { \partial f _ { \Theta , \delta } } { \partial w _ { j } } \approx \sigma _ { j }\tag{28}
$$

We implement this alongside the sharp forward pass in PyTorch and demonstrate the effectiveness by QAT experiments over a range of quantization steps for a simple MNIST classification task with LeNet [36]. We train these models for 40 epochs minimizing the cross entropy loss using the Adam optimizer [37] with learning rate 0.001 and no weight decay. A non-quantized baseline is trained reaching about 99 % classification accuracy. We allow up to 2 percentage points accuracy drop with quantization. The results in comparison to hand-selected activation functions are shown in Figure 11, showing comparable results. In both settings, the model can be quantized down to 4 steps corresponding to approximately 2.32 bit, before the accuracy starts to break down completely. The accuracy increase observed at 1-bit quantization needs to be investigated in future work, but seems unrelated to the learned thresholds as it is observed with the uniform quantizer as well. While the results for the uniform quantizer heavily depend on the choice of activation function, the results suggest the learnable multi-threshold functions to be more robust in this low-bitwidth quantization regime. Figure 10 exemplarily shows functions learned for an instance of the non-monotonic per-tensor setting. The learned functions seem plausible in the sense, that they are centered at about zero and cover a range which is typical for neural network activation functions.

![](images/eb4cf16c8169e8ce3fe8c6e9f169adb24ff9668cf7acb7d86874c3d1bbf22639.jpg)

![](images/c1b0370382d30833d8d99eeb716a361991899ab4b642a0fe072937b11c8486e8.jpg)

![](images/0909c101bd990e9af7356b7121671b9737f96953ad6a93e6711a2b30725f8e8f.jpg)  
Fig. 9: Comparison of functions implemented as multi-threshold operators in FINN and a composite lookup-table based approach in hls4ml in terms of LUT count over quantization bitwidth $\beta ^ { * }$ . For FINN, HLS refers to our binary-search backend (Section III) and RTL refers to the backend introduced by SIRA, which requires graph-level decomposition for the non-monotonic functions. Shaded in dark grey is the portion of LUTs used by the multi-threshold operators, which is a more realistic estimate of the resource impact within a streamlined accelerator, where de-quantization occurs only once at the end of the model graph, not after each operation. For hls4ml we set the number of lookup table entries to the same number of thresholds required by our approach – this number is annotated on top of the first group of bars per function. The selected lookup table data type is either floating-point or $\beta ^ { * }$ bit fixed-point. The two functions on the right are not directly supported as a lookup table in hls4ml and are realized as a composition of a $\sigma ( x )$ lookup table and standalone additions and multiplications.

![](images/092c5a3f4da88e20c632873596379636be5d16339abfb960869b1a2e6a002b79.jpg)

![](images/c62a6fa57d536c55b2591a0160ab63aa64028e37b0ff80fcb9e33cbe7105812e.jpg)

![](images/5f4b4f2aeb40916f4baf901b2f6927d248961af34c116d35974027fdf55d4666.jpg)

![](images/575c5ce778ae62fa1205c862e5e703dc7d4395a7e440711c37418003aacd8c5d.jpg)  
Fig. 10: Non-monotonic per-tensor multi-threshold functions learned by the four activation layers of LeNet for N = 15 steps.

![](images/3b13e5fafdbd4a07b3db2f4af2932a5e65e3262072de28e7bfd3d0298f422e16.jpg)

![](images/ac434ec8542bb40f92665427734a4b519f735161b930cf4c7f8a786caaa29599.jpg)  
Fig. 11: Comparison of learned multi-threshold functions to hand-selected quantized activation functions for MNIST classification with LeNet. Standard deviation in shaded.

## VIII. CONCLUSION

We have extended the multi-threshold representation of quantized functions to non-monotonic functions with finegranular parameters by extending the binary search based multi-threshold operator integrated into the FINN framework and an alternative graph-level decomposition approach. The proposed solutions have been evaluated in terms of resource utilization on an FPGA to characterize the cost of nonmonotonicity and fine-granularity. A systematic and lightweight compilation approach of quantized layer tails to multithreshold representation has been proposed to facilitate end-toend synthesis of custom-tailored FPGA accelerators. We have successfully demonstrated this on a selection of monotonic and non-monotonic functions. Furthermore, we have proposed directly learning multi-threshold representations as a form of QAT, which we have successfully demonstrated on a simple MNIST classifier. Future work will extend function support by implementing the necessary inverse functions and further investigate quality and performance implications of the newly gained options by using them with more demanding neural network inference workloads.

## ACKNOWLEDGMENTS

This work has been funded by the Ministry of Culture and Science of the State of North Rhine-Westphalia under the grant no NW21-059D (SAIL) and by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection under the grant no 67KI32004A (eki).

We gratefully acknowledge the computing time provided to us on the high-performance computers Noctua 2 and Otus at the NHR Center PC2.

## REFERENCES

[1] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A survey of quantization methods for efficient neural network inference,” in Low-power computer vision. Chapman and Hall/CRC, 2022, pp. 291–326.

[2] B. Rokh, A. Azarpeyvand, and A. Khanteymoori, “A comprehensive survey on model quantization for deep neural networks in image classification,” ACM Transactions on Intelligent Systems and Technology, vol. 14, no. 6, pp. 1–50, 2023.

[3] G. Armeniakos, G. Zervakis, D. Soudris, and J. Henkel, “Hardware approximate techniques for deep neural network accelerators: A survey,” ACM Comput. Surv., vol. 55, no. 4, Nov. 2022. [Online]. Available: https://doi.org/10.1145/3527156

[4] I. Colbert, A. Pappalardo, and J. Petri-Koenig, “A2q: Accumulator-aware quantization with guaranteed overflow avoidance,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 16 943– 16 952.

[5] Y. Umuroglu, N. J. Fraser, G. Gambardella, M. Blott, P. Leong, M. Jahre, and K. Vissers, “Finn: A framework for fast, scalable binarized neural network inference,” in Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, ser. FPGA ’17. ACM, 2017, pp. 65–74.

[6] M. Blott, T. B. Preußer, N. J. Fraser, G. Gambardella, K. O’brien, Y. Umuroglu, M. Leeser, and K. Vissers, “Finn-r: An end-to-end deeplearning framework for fast exploration of quantized neural networks,” ACM Transactions on Reconfigurable Technology and Systems (TRETS), vol. 11, no. 3, pp. 1–23, 2018.

[7] S. I. Venieris and C.-S. Bouganis, “fpgaconvnet: A framework for mapping convolutional neural networks on fpgas,” in 2016 IEEE 24th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2016, pp. 40–47.

[8] J. Duarte, S. Han, P. Harris, S. Jindariani, E. Kreinar, B. Kreis, J. Ngadiuba, M. Pierini, R. Rivera, N. Tran et al., “Fast inference of deep neural networks in fpgas for particle physics,” Journal of instrumentation, vol. 13, no. 07, p. P07027, 2018.

[9] R. Bosio, F. Minnella, T. Urso, M. R. Casu, L. Lavagno, M. T. Lazarescu, and P. Pasini, “NN2FPGA: Optimizing cnn inference on fpgas with binary integer programming,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, pp. 1–1, 2024.

[10] F. Jentzsch, C. Berganski, L. Jungemann, and B. Wintermann, “FINN+,” 2026. [Online]. Available: https://github.com/eki-project/finn-plus

[11] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, M. Suo, P. Tillet, E. Wang, X. Wang, W. Wen, S. Zhang, X. Zhao, K. Zhou, R. Zou, A. Mathews, G. Chanan, P. Wu, and S. Chintala, “PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation,” in 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24). ACM, Apr. 2024. [Online]. Available: https://docs.pytorch.org/assets/pytorch2-2.pdf

[12] G. Franco, A. Pappalardo, and N. J. Fraser, “Xilinx/brevitas,” 2025. [Online]. Available: https://doi.org/10.5281/zenodo.3333552

[13] A. Pappalardo, Y. Umuroglu, M. Blott, J. Mitrevski, B. Hawks, N. Tran, V. Loncar, S. Summers, H. Borras, J. Muhizi, M. Trahms, S.-C. H. Hsu, S. Hauck, and J. Duarte, “QONNX: Representing Arbitrary-Precision Quantized Neural Networks,” in 4th Workshop on Accelerated Machine Learning (AccML) at HiPEAC 2022 Conference, 6 2022. [Online]. Available: https://accml.dcs.gla.ac.uk/papers/2022/ 4thAccML paper 1(12).pdf

[14] Y. Umuroglu, H. Borras, V. Loncar, S. Summers, and J. Duarte, “QONNX,” Jun. 2022. [Online]. Available: https: //github.com/fastmachinelearning/qonnx

[15] Y. Umuroglu, C. Berganski, F. Jentzsch, M. Danilowicz, T. Kryjak, C. Bezaitis, M. Sjalander, I. Colbert, T. Preusser, J. Petri-Koenig,¨ and M. Blott, “Sira: Scaled-integer range analysis for optimizing fpga dataflow neural network accelerators,” ACM Transactions on Reconfigurable Technology and Systems, vol. 19, no. 2, May 2026. [Online]. Available: https://doi.org/10.1145/3807510

[16] M. Neseem, C. McCullough, R. Hsin, C. Leichner, S. Li, I. S. Chong, A. Howard, L. Lew, S. Reda, V.-M. Rautio et al., “PikeLPN: Mitigating overlooked inefficiencies of low-precision neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 15 996–16 005.

[17] R. Karami, S.-C. Kao, and H. Kwon, “Understanding the performance horizon of the latest ml workloads with nongemm workloads,” in 2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 2025, pp. 1–14.

[18] R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper,” arXiv preprint arXiv:1806.08342, 2018.

[19] M. Nagel, M. v. Baalen, T. Blankevoort, and M. Welling, “Datafree quantization through weight equalization and bias correction,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 1325–1334.

[20] S. Dai, R. Venkatesan, M. Ren, B. Zimmer, W. Dally, and B. Khailany, “VS-Quant: Per-vector scaled quantization for accurate low-precision neural network inference,” Proceedings of Machine Learning and Systems, vol. 3, pp. 873–884, 2021.

[21] H. Yao, P. Li, J. Cao, X. Liu, C. Xie, and B. Wang, “RAPQ: Rescuing accuracy for power-of-two low-bit post-training quantization,” arXiv preprint arXiv:2204.12322, 2022.

[22] E. Frantar, S. Ashkboos, T. Hoefler, and D.-A. Alistarh, “OPTQ: Accurate post-training quantization for generative pre-trained transformers,” in 11th International Conference on Learning Representations, 2023.

[23] Y. Umuroglu and M. Jahre, “Streamlined deployment for quantized neural networks,” arXiv preprint arXiv:1709.04060, 2017.

[24] S. R. Dubey, S. K. Singh, and B. B. Chaudhuri, “Activation functions in deep learning: A comprehensive survey and benchmark,” Neurocomputing, vol. 503, pp. 92–108, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231222008426

[25] V. Kunc and J. Klema, “Three decades of activations: A comprehensive´ survey of 400 activation functions for neural networks,” arXiv preprint arXiv:2402.09092, 2024.

[26] D. Hendrycks, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016.

[27] R. M. Corless, G. H. Gonnet, D. E. Hare, D. J. Jeffrey, and D. E. Knuth, “On the lambert w function,” Advances in Computational mathematics, vol. 5, no. 1, pp. 329–359, 1996.

[28] R. Iacono and J. P. Boyd, “New approximations to the principal realvalued branch of the lambert w-function,” Advances in Computational Mathematics, vol. 43, no. 6, pp. 1403–1436, 2017.

[29] L. Loczi, “Guaranteed- and high-precision evaluation of the lambert´ w function,” Applied Mathematics and Computation, vol. 433, p. 127406, 2022. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0096300322004805

[30] FastML Team, “fastmachinelearning/hls4ml,” 2026. [Online]. Available: https://github.com/fastmachinelearning/hls4ml

[31] J.-F. Schulte, B. Ramhorst, C. Sun, J. Mitrevski, N. Ghielmetti, E. Lupi, D. Danopoulos, V. Loncar, J. Duarte, D. Burnette, L. Laatu, S. Tzelepis,ˇ K. Axiotis, Q. Berthet, H. Wang, P. White, S. Demirsoy, M. Colombo, T. K. Aarrestad, S. Summers, M. Pierini, G. Di Guglielmo, J. Ngadiuba, J. Campos, B. Hawks, A. Gandrakota, F. Fahim, N. Tran, G. A. Constantinides, Z. Que, W. Luk, A. Tapper, D. Hoang, N. Paladino, P. C. Harris, B.-C. Lai, M. Valentin, R. Forelli, S. Ogrenci, L. Gerlach, R. Brooks Flynn, M. Liu, D. Diaz, E. E. Koda, M. Quinnan, R. M. Solares, S. Parajuli, M. S. Neubauer, C. Herwig, H. F. Tsoi, D. Rankin, S.-C. Hsu, and S. Hauck, “hls4ml: A flexible, open source platform for

deep learning acceleration on reconfigurable hardware,” ACM Trans. Reconfigurable Technol. Syst., vol. 19, no. 2, May 2026. [Online]. Available: https://doi.org/10.1145/3801979

[32] M. Goyal, R. Goyal, and B. Lall, “Learning activation functions: A new paradigm for understanding neural networks,” arXiv preprint arXiv:1906.09529, 2019.

[33] ——, “Improved polynomial neural networks with normalised activations,” in 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020, pp. 1–8.

[34] H. Fang, J.-U. Lee, N. S. Moosavi, and I. Gurevych, “Transformers with learnable activation functions,” in Findings of the Association for Computational Linguistics: EACL 2023, 2023, pp. 2382–2398.

[35] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaciˇ c,´ T. Y. Hou, and M. Tegmark, “Kan: Kolmogorov-arnold networks,” arXiv preprint arXiv:2404.19756, 2024.

[36] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[37] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations (ICLR), 2015.