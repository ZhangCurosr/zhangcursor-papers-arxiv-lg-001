# EFQ-Softmax: Exp-Free Quantization for Softmax

Haohui Han<sup>1</sup>, Yuming Wan<sup>2</sup>, Hongni Wang<sup>3</sup>, Pengcheng Xie<sup>2</sup>

Xiaodong Yan<sup>1,\*</sup>, Runqi You<sup>1</sup>, Wencong Zhang<sup>1</sup>

<sup>1</sup>Xi’an Jiaotong University

<sup>2</sup>Huawei Technologies Co., Ltd

<sup>3</sup>Shandong University of Finance and Economics

hhhan200001@163.com, wanyuming3@huawei.com, wanghongnisd@126.com xiepengcheng2@huawei.com, yanxiaodong@xjtu.edu.cn 18174543996@163.com, 2161934877@qq.com

Abstract—Low-bit attention accelerates Transformer inference by moving the QK<sup>⊤</sup> and PV matrix multiplications to FP8 or FP4 matrix engines. However, the softmax probability path often still evaluates shifted-score exponentials in higher precision, forms a temporary probability block, and then quantizes it before the low-bit PV multiplication. This exp-then-quantize path creates a mismatch between a high-precision probability producer and a low-bit matrix consumer.

This paper proposes EFQ-Softmax (Exp-Free Quantization for Softmax), a low-bit probability-generation method that directly maps shifted attention scores to block-scaled E2M1 operands. For each microscaling block, EFQ-Softmax selects an exponentonly scale from the local maximum, maps the shifted scores to a normalized residual domain, and generates nonnegative E2M1 probability codes using a single affine rule. The resulting operand is used consistently in both the P Ve numerator update and the Pe1 denominator update. Meanwhile, the FlashAttentionstyle row-maximum update, historical rescaling, high-precision accumulation, and final normalization remain unchanged.

We evaluate the end-to-end quality of EFQ-Softmax across language, vision-language, and text-to-video workloads, and separately evaluate its kernel-level performance on the A5 vector unit. Specifically, we evaluate EFQ-Softmax on Qwen3-8B, Qwen3-VL-8B-Instruct, and WAN2.2-TI2V-5B. EFQ-Softmax improves the Qwen3-8B seven-task mean from 0.6749 with MXFP4 to 0.6773 and the Qwen3-VL nine-task mean from 0.7826 to 0.8000. On WAN2.2, it maintains temporal consistency and visual quality comparable to the FP16 and MXFP4 baselines under VBench. On the A5 vector unit, EFQ-Softmax reduces the vector-stage latency of the fused probability-generation kernel by 40.33% on average across sequence lengths from 16K to 128K. These results show that direct low-bit probability generation can replace the conventional exp-then-quantize path while preserving end-to-end model quality.

## I. INTRODUCTION

Self-attention is a central operator in modern Transformer models [20]. In a standard implementation, its arithmetic cost grows quadratically with the sequence length, and the score and probability matrices may also require quadratic storage. These costs make attention expensive in long-context language models, vision-language models, and video-generation models. FlashAttention reduces memory traffic by computing attention in tiles and maintaining the softmax statistics through an online recurrence, without materializing the full attention matrix [4].

FlashAttention-2 further improves parallelism and work partitioning [3], while FlashAttention-3 uses asynchronous execution and low-precision matrix multiplication on recent hardware [16]. In parallel, low-precision formats such as FP8 and microscaling FP4 provide compact operands for modern matrix engines [11], [14]. Quantized attention systems such as SageAttention and SageAttention3 show that the two matrix multiplications in attention can be executed at low precision while maintaining model quality across language, image, and video workloads [24], [25]. As these matrix multiplications become faster, the probability-generation operations between score computation and value multiplication account for a larger part of the attention execution time.

In a FlashAttention-style kernel, each score tile is first shifted by an updated row maximum for numerical stability. Elementwise exponentials are then evaluated to generate the current unnormalized probability weights. In a conventional FP4 attention path, these weights are first produced in FP16 or FP32 and are then quantized into a block-scaled E2M1 representation for the low-bit value multiplication. This expthen-quantize path therefore contains three separate steps: evaluating dense elementwise exponentials, forming a temporary high-precision probability tile, and quantizing the tile into a low-bit operand. The resulting pipeline uses a high-precision probability producer even though the following matrix multiplication consumes low-precision values. Existing FP4 attention methods improve the scaling and quantization of probability values, but still compute the exponential values before converting them into a low-bit representation [25]. This producerconsumer mismatch motivates bypassing the temporary highprecision probability representation and directly generating the low-bit operand from the shifted scores. However, doing so is not merely a scalar exponential-approximation problem. The generated values must be directly usable by the low-bit value multiplication, and the same approximation must be used in both the softmax numerator and denominator. At the same time, the online recurrence must preserve its row-maximum update, historical rescaling, high-precision accumulation, and final normalization.

To address this problem, we propose EFQ-Softmax, a lowbit probability-generation method that directly converts shifted attention scores into a block-scaled E2M1 operand. Here, “Exp-Free” refers to removing the dense elementwise exponential computation for the current score block, while retaining the row-level exponential required for historical rescaling. For each microscaling block, EFQ-Softmax selects an exponentonly scale from the local maximum, maps the shifted scores to a scale-normalized residual domain, and generates nonnegative E2M1 codes using a single affine index rule. Together with the block-level E8M0 scale, these codes form the final MXFP4 probability operand and are used consistently in both the numerator and denominator updates. This design eliminates the intermediate FP16 or FP32 probability tile and the separate post-exponential quantization step, while preserving the rowmaximum update, historical rescaling, high-precision accumulation, and final normalization of online attention.

![](images/2c7ddd23431e89d4fdda40d6f7328ee4525dfc4e6260b0a4475cdf1f6a41bdfe.jpg)  
Fig. 1: EFQ-Softmax directly generates a shared block-scaled E2M1 operand within online attention.

Fig. 1 illustrates how EFQ-Softmax is integrated into the online attention recurrence, while Fig. 2 compares its direct score-to-E2M1 generation path with the conventional expthen-quantize path.

We evaluate EFQ-Softmax on Qwen3-8B [18], Qwen3- VL-8B-Instruct [19], and WAN2.2-TI2V-5B [21]. On Qwen3- 8B, EFQ-Softmax improves the seven-task mean score from 0.6749 with MXFP4 to 0.6773. On Qwen3-VL-8B-Instruct, it improves the nine-task mean score from 0.7826 to 0.8000 and achieves better results on all six referring-expression grounding splits. On WAN2.2, EFQ-Softmax maintains temporal consistency and visual quality comparable to the FP16 and MXFP4 baselines under VBench [7]. We also evaluate the probability-generation path on the A5 vector unit. Across sequence lengths from 16K to 128K, EFQ-Softmax reduces the vector-stage latency of the fused probability-quantization kernel by 40.33% on average. These results show that EFQ-Softmax shortens the probability-generation path while preserving numerical accuracy and end-to-end model quality.

Our main contributions are summarized as follows:

• We propose EFQ-Softmax, a low-bit probabilitygeneration method that fuses exponential approximation and MXFP4 probability quantization. It maps shifted attention scores to E2M1 probability codes through a single affine transformation and rounding, together with a block-level E8M0 scale.

• Through integrating EFQ-Softmax into the FlashAttention-style online attention computation, the method preserves row-maximum and row-sum state updates, avoids storing full S or P matrices, and directly feeds the generated MXFP4 probability block into the low-bit PV multiplication.

• EFQ-Softmax is evaluated on both controlled simulations and real model workloads. Experiments on Qwen3-8B, Qwen3-VL, and Wan2.2 show that the proposed method improve attention throughput while preserving numerical accuracy and end-to-end output quality.

The rest of this paper is organized as follows. Section II presents the design of EFQ-Softmax, including the blockscaled E2M1 representation, exponent-only scale selection, residual-domain normalization, affine code generation, and the EFQ-Softmax-16 variant. Section III describes the integration of EFQ-Softmax into FlashAttention-style online attention, including the numerator and denominator updates, algorithm summary, parameter calibration, and implementation boundary. Section IV evaluates EFQ-Softmax on language, visionlanguage, text-to-video, and kernel-level workloads. Section V concludes the paper.

![](images/9980e0fe25d0e34cedb55b0118ff75794b2ddcbf8aba4c2b90dd1b62723c5ce0.jpg)  
Fig. 2: Conventional exp-then-quantize versus EFQ-Softmax. EFQ-Softmax directly generates block-scaled E2M1 probabilities from shifted scores, reducing A5 vector-stage latency by 40.33%.

## II. EFQ-SOFTMAX METHODOLOGY

## A. Design Scope and Overview

a) Targeted probability path.: For one attention head, let $Q \in \mathbb { R } ^ { N _ { q } \times d } , K \in \mathbb { R } ^ { N _ { k } \times d }$ , and $V ~ \in ~ \mathbb { R } ^ { N _ { k } \times d _ { v } }$ . The exact attention output is

$$
O = \operatorname { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) V .
$$

Let $S = Q K ^ { \top } / \sqrt { d }$ denote the score matrix. Stable softmax evaluates each row after subtracting the row maximum:

$$
P _ { i t } = \frac { \exp ( S _ { i t } - m _ { i } ) } { \sum _ { r } \exp ( S _ { i r } - m _ { i } ) } , \qquad m _ { i } = \operatorname* { m a x } _ { r } S _ { i r } .
$$

The elementwise term that must be generated before normalization is therefore

$$
p _ { i t } = \exp ( x _ { i t } ) , \qquad x _ { i t } = S _ { i t } - m _ { i } \leq 0 .\tag{1}
$$

We refer to $x _ { i t } = S _ { i t } - m _ { i }$ as a row-max-shifted attention score, or simply a shifted score.

EFQ-Softmax modifies only the probability-generation path. It replaces the elementwise computation of exp(x) for the current score block and the subsequent probability quantization with direct generation of a low-bit probability operand. The attention formulation, value-side representation, and precision of the online accumulators remain unchanged from the surrounding attention kernel. The value block is not modified by EFQ-Softmax; it only consumes the generated probability operand in the subsequent (PV) multiplication.

In a low-bit attention path, the PV consumer often expects a low-bit probability operand. A conventional construction first evaluates Eq. (1) in a higher-precision format and then quantizes the resulting probability block:

$$
x \to \exp ( x ) \to P _ { \mathrm { F P 1 6 / F P 3 2 } } \to \mathrm { Q u a n t } _ { \mathrm { E 2 M 1 } } ( P ) \to Q _ { P } ^ { \mathrm { E 2 M 1 } } .\tag{2}
$$

EFQ-Softmax replaces the exp-then-quantize sequence by direct probability-code generation:

$$
x \to c ( x ) \to Q _ { P } ^ { \mathrm { E 2 M 1 } } .\tag{3}
$$

The generated operand is then shared by the numerator branch ${ \widetilde { P } } V$ and the denominator branch $\widetilde { P } 1$ . The outputs of these two branches are accumulated in FP16 or FP32, as in standard lowbit matrix-multiply pipelines.

b) Online attention recurrence.: To describe how EFQ Softmax operates in a tiled online-attention kernel, we now move from the elementwise notation above to a blockwise formulation. FlashAttention processes the score matrix by query blocks and key/value blocks. For query block i and key/value block $j ,$ the score block is

$$
S _ { i } ^ { j } = \frac { Q _ { i } K _ { j } ^ { \top } } { \sqrt { d } } + M _ { i } ^ { j } ,
$$

where $M _ { i } ^ { j }$ denotes an optional mask term. The online state consists of a row maximum $m _ { i } ^ { j } .$ , a numerator accumulator $A _ { i } ^ { j } ,$ and a denominator accumulator $l _ { i } ^ { j }$ . The row maximum update is

$$
m _ { i } ^ { j } = \operatorname* { m a x } \left\{ m _ { i } ^ { j - 1 } , \mathrm { r o w m a x } ( S _ { i } ^ { j } ) \right\} .\tag{4}
$$

Using $x _ { i } ^ { j } = S _ { i } ^ { j } - m _ { i } ^ { j }$ , the exact current unnormalized probability block is

$$
P _ { i } ^ { j } = \exp ( x _ { i } ^ { j } ) .
$$

The historical state must be rescaled when the row maximum changes. With

$$
\alpha _ { i } ^ { j } = \exp ( m _ { i } ^ { j - 1 } - m _ { i } ^ { j } ) ,\tag{5}
$$

the exact online update is

$$
A _ { i } ^ { j } = \alpha _ { i } ^ { j } A _ { i } ^ { j - 1 } + { \cal P } _ { i } ^ { j } V _ { j } ,\tag{6}
$$

$$
l _ { i } ^ { j } = \alpha _ { i } ^ { j } l _ { i } ^ { j - 1 } + P _ { i } ^ { j } \mathbf { 1 } .\tag{7}
$$

EFQ-Softmax keeps Eqs. (4)-(7). Only the current-block probability block $P _ { i } ^ { j } = \exp ( x _ { i } ^ { j } )$ is replaced by a block-scaled lowbit approximation.

c) Required probability operand.: These constraints define the required EFQ-Softmax output. The generated operand must be directly consumed by the low-bit $P V$ path, reused by the denominator update, and accumulated into the existing high-precision online state. We therefore represent the current unnormalized probability block as

$$
\widetilde { P } _ { i } ^ { j } \approx s _ { P , i } ^ { j } Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } } .
$$

Here $Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } }$ is a nonnegative E2M1 code block and $s _ { P , i } ^ { j }$ is a block scale. The rest of this section describes how $Q _ { P , * } ^ { j , \mathrm { F } }$ E2M1 and $s _ { P , i } ^ { j }$ are generated directly from the shifted score block $x _ { i } ^ { j } = S _ { i } ^ { j } - m _ { i } ^ { j }$

## B. Block-Scaled E2M1 Probability Representation

a) Nonnegative E2M1 code set.: The E2M1 format contains a small number of representable values. Since attention probabilities are nonnegative, EFQ-Softmax only uses the nonnegative code set

$$
\mathcal { C } _ { 8 } = \left\{ 0 , \frac { 1 } { 2 } , 1 , \frac { 3 } { 2 } , 2 , 3 , 4 , 6 \right\} .
$$

The index set is $\{ 0 , \ldots , 7 \}$ , and $\mathcal { C } _ { 8 } [ c ]$ denotes the numerical value represented by code c. This notation is only used to state the mathematical value of each bit pattern. In the data path, the generated code is packed as a 4-bit E2M1 probability operand rather than used as an index for a runtime value lookup.

For each microscaling block b, EFQ-Softmax writes $\widetilde { p } _ { t } =$ $s _ { b } q _ { t } , q _ { t } \in \mathcal { C } _ { 8 } , t \in b$ . The scale $s _ { b }$ determines the exponent range of the block, while the code value $q _ { t }$ determines the local mantissa-like value. In the probability path, this is sufficient because all inputs satisfy $x _ { t } \le 0$ after the row-maximum shift and the largest exact value inside a row is at most one.

b) Exponent-Only Scale Rule: Let $M _ { b } = \operatorname* { m a x } _ { t \in b } x _ { t }$ be the largest shifted score inside a microscaling block. Since the largest value in $\mathcal { C } _ { 8 }$ is 6, the scale is chosen so that $6 s _ { b }$ is close to $\mathbf { \bar { \Pi } } _ { e } M _ { b }$ . EFQ-Softmax restricts the scale to an exponent-only value $s _ { b } = 2 ^ { k _ { b } }$ . The scale exponent is selected by

$$
k _ { b } = \left\lfloor \frac { M _ { b } + \log ( 2 / 9 ) } { \log 2 } \right\rfloor .\tag{8}
$$

Equivalently,

$$
2 ^ { k _ { b } } \leq \frac { 2 e ^ { M _ { b } } } { 9 } < 2 ^ { k _ { b } + 1 } .\tag{9}
$$

The constant $\log ( 2 / 9 )$ is part of the scale rule. It is not a fitted EFQ-Softmax parameter. Its role is to choose a powerof-two scale around the largest local probability value under the E2M1 maximum code 6. The only fitted EFQ-Softmax parameters are the affine code-generation parameters $( \tau , h )$ introduced below.

This exponent-only scale is compatible with the probabilityside data path. It avoids a general floating-point scale generation step and provides a compact representation that can be consumed together with the generated E2M1 code. The scale is shared by all elements in the microscaling block; therefore the elementwise decision that remains after scaling is only an index-generation problem.

c) Residual Log-Domain Normalization: After the scale has been selected, each shifted score is converted into a residual log-domain value

$$
z _ { t } = x _ { t } - \log ( 6 s _ { b } ) = x _ { t } - k _ { b } \log 2 - \log 6 .\tag{10}
$$

This residual satisfies

$$
e ^ { x _ { t } } = s _ { b } \cdot 6 e ^ { z _ { t } } .
$$

Thus the remaining operation is to approximate $6 e ^ { z _ { t } }$ by one of the values in $\mathcal { C } _ { 8 }$ . The residual $z _ { t }$ is the proper input to the code-generation rule because it has already absorbed the block scale. Using $x _ { t }$ directly without this normalization would mix two tasks: block-level dynamic-range selection and elementlevel code selection.

The largest residual in block b is

$$
z _ { \mathrm { m a x } , b } = M _ { b } - k _ { b } \log 2 - \log 6 ,
$$

From Eq. $( 9 ) , z _ { \operatorname* { m a x } , b }$ remains in a bounded interval determined by the power-of-two rounding of the scale. This bounded residual range is the reason that a single pair of EFQ-Softmax parameters can be reused across many blocks. The exact shifted scores may have a large dynamic range across rows and layers, but the residual code-generation input is locally normalized.

d) Microscaling blocks.: The microscaling block size is a design parameter of the probability path. For a flattened local score block, the partition can be written as

$$
B = \{ b _ { 1 } , b _ { 2 } , \ldots , b _ { R } \} , \qquad | b _ { r } | = B .
$$

For each $b _ { r }$ , the scale exponent $k _ { b _ { r } }$ is computed once and then reused by all elements in the block. A smaller B gives more local scaling and can reduce approximation error, while a larger B reduces scale metadata and scale-generation work. EFQ-Softmax does not rely on a particular value of B in the mathematical definition, and the implementation keeps B consistent with the low-bit matrix operand layout.

## C. EFQ-Softmax Probability Code Generation

Given the residual $z _ { t }$ from Eq. (10), EFQ-Softmax generates a 4-bit E2M1 probability code with a single affine rule:

$$
c _ { t } = \mathrm { c l i p } _ { [ 0 , 7 ] } \left( \lfloor ( z _ { t } - \tau ) h \rfloor + 1 \right) .\tag{11}
$$

The generated $c _ { t }$ is the probability code. Operationally, it is stored as a 4-bit pattern and interpreted as a nonnegative E2M1 operand:

$$
Q _ { P , t } ^ { \mathrm { E 2 M 1 } }  \mathrm { r e i n t e r p r e t } _ { \mathrm { E 2 M 1 } } ( c _ { t } ) .
$$

Mathematically, the represented probability value is $\begin{array} { r l } { \widetilde { p } _ { t } } & { { } = } \end{array}$ $s _ { b } \mathcal { C } _ { 8 } [ c _ { t } ]$ . Here $\mathcal { C } _ { 8 } [ c _ { t } ]$ denotes the numerical value of the E2M1 bit pattern and does not imply a runtime lookup of a highprecision exponential value.

![](images/86c3e5534f0ecd6ac293130b55bf5c6fa655805bb61a4d50bee0f192ef346a1f.jpg)  
Fig. 3: Overview of the five-stage EFQ-Softmax probability-generation procedure.

Combining the scale rule in Eq. (8), the residual definition in Eq. (10), and the index rule in Eq. (11), EFQ-Softmax implements the direct path

$$
x _ { t } \to z _ { t } \to c _ { t } \to Q _ { P , t } ^ { \mathrm { E 2 M 1 } } .
$$

Equivalently, for $t \in b ,$ , the represented probability value is

$$
\begin{array} { r l } & { \widetilde { p } _ { t } = 2 ^ { k _ { b } } \mathcal { C } _ { 8 } \left[ c _ { t } \right] , } \\ & { c _ { t } = \mathrm { c l i p } _ { [ 0 , 7 ] } \left( \left\lfloor \left( x _ { t } - k _ { b } \log 2 - \log 6 - \tau \right) h \right\rfloor + 1 \right) . } \end{array}
$$

The generated result is already the low-bit probability operand consumed by the following attention update, so EFQ-Softmax does not materialize an intermediate FP16/FP32 probability block required in the intended path.

The parameters τ and h define uniformly spaced thresholds in the residual log domain. Ignoring clipping, code c is selected when

$$
\tau + \frac { c - 1 } { h } \leq z _ { t } < \tau + \frac { c } { h } .
$$

The parameter τ shifts the thresholds, while h controls their spacing. Thus, EFQ-Softmax uses uniformly spaced residual thresholds together with a fixed nonuniform E2M1 output set. EFQ-Softmax does not use these thresholds to fetch a highprecision exponential value. The output of Eq. (11) is already the final 4-bit E2M1 code consumed by the attention data path.

The clipping in Eq. (11) bounds the generated code to the nonnegative E2M1 range. Residuals below the lowest threshold map to code 0, while residuals above the highest threshold map to code 7, which represents the largest nonnegative E2M1 value 6. The zero code is not a separate pruning rule; it is one representable outcome of the block-scaled E2M1 probability operand. This saturation is part of bounded index generation rather than an additional algorithmic branch.

If 6e<sup>z</sup> were first computed and then nearest-rounded to $\mathcal { C } _ { 8 } .$ , the resulting log-domain boundaries would be nonuniform because the E2M1 values are nonuniform. EFQ-Softmax deliberately gives up this scalar nearest-rounding rule and uses affine residual thresholds to simplify code generation.

This design trades scalar nearest-rounding optimality for a simpler probability-code generation path. Nearest-code rounding after an explicit exponential requires computing $e ^ { x }$ and then deciding the final E2M1 value. EFQ-Softmax removes the explicit exponential and learns a direct residual threshold rule. The correctness target is not exact scalar exponential reconstruction. The relevant target is output-level attention accuracy after the numerator and denominator updates.

The parameter τ controls the global threshold shift, and h controls the threshold spacing. Their values are selected offline by the calibration procedure described in Sec. III-C.

The complete EFQ-Softmax probability-generation procedure is summarized in Figure 3. The figure shows how shifted attention scores are transformed into packed block-scaled E2M1 probability operands through five consecutive stages.

## D. EFQ-Softmax-LUT Variant

EFQ-Softmax-LUT uses the same block scale and the same nonnegative E2M1 output set as EFQ-Softmax, but introduces a 16-level intermediate index before folding to the final 8-code E2M1 output. The intermediate index is

$$
j _ { 1 6 } ( z _ { t } ) = \mathrm { c l i p } _ { [ 0 , 1 5 ] } \left( \lfloor ( z _ { t } - \tau _ { 1 6 } ) h _ { 1 6 } \right\rfloor + 1 ) ,
$$

where $\tau _ { 1 6 } = \log ( 1 / 2 4 )$ $\begin{array} { r } { h _ { 1 6 } = \frac { 1 4 } { \log { 2 4 } } } \end{array}$ . The final E2M1 code is obtained by the monotone folding table

$$
C _ { 1 6 } = [ 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 3 , 3 , 4 , 5 , 5 , 6 , 7 , 7 ] .
$$

Then the final code is

$$
c _ { t } ^ { \mathrm { t a b } } = C _ { 1 6 } [ j _ { 1 6 } ( z _ { t } ) ] , \qquad Q _ { P , t } ^ { \mathrm { E 2 M 1 } } \gets \mathrm { p a c k 4 } _ { \mathrm { E 2 M 1 } } ( c _ { t } ^ { \mathrm { t a b } } ) .
$$

Mathematically, the represented probability value is $\widetilde { p } _ { t } ^ { \mathrm { t a b } } =$ $s _ { b } \mathcal { C } _ { 8 } [ c _ { t } ^ { \mathrm { t a b } } ]$ . In contrast to EFQ-Softmax, this variant contains an explicit 16-to-8 remapping stage.

Fig. 4 compares the affine code mapping of EFQ-Softmax with the two-stage mapping of EFQ-Softmax-LUT.

EFQ-Softmax-LUT is useful as an accuracy-oriented comparison point. It has finer intermediate boundary placement than the 8-code EFQ-Softmax rule, but it also introduces an additional remapping stage. For this reason, the main method remains EFQ-Softmax. EFQ-Softmax-LUT acts as an ablation that measures how much accuracy can be gained by increasing mapping flexibility at the probability-code generation point.

![](images/b4b3f9fc2e7b6d4a26a743a35a008d3dc691503d41f8674629468834d67d6f84.jpg)  
Fig. 4: Affine code mapping in EFQ-Softmax and two-stage mapping in EFQ-Softmax-LUT.

## III. ONLINE INTEGRATION AND IMPLEMENTATION

## A. Integration with Online Attention

a) Probability operand and consumers.: For a score block (i, j), EFQ-Softmax produces the probability operand

$$
\widetilde { P } _ { i } ^ { j } \approx s _ { P , i } ^ { j } Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } } .
$$

The numerator contribution is

$$
\Delta A _ { i } ^ { j } = \widetilde { P } _ { i } ^ { j } V _ { j } .\tag{12}
$$

If the value block is already represented by an existing low-bit path, $V _ { j } \approx s _ { V } ^ { j } Q _ { V } ^ { j }$ , the consumer can be written as

$$
\Delta A _ { i } ^ { j } \approx s _ { P , i } ^ { j } s _ { V } ^ { j } \mathrm { M M } \left( Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } } , Q _ { V } ^ { j } \right) .\tag{13}
$$

Equation (13) only specifies the consumer of the generated probability operand. It is not a new value quantization rule. $Q _ { P , i } ^ { j , \mathrm { E } }$ 2M1 The contribution of EFQ-Softmax is the generation of from the shifted score block.

The same probability operand is also used by the denominator branch:

$$
\Delta l _ { i } ^ { j } = \widetilde { P } _ { i } ^ { j } \mathbf { 1 } \approx s _ { P , i } ^ { j } \left( Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } } \mathbf { 1 } \right) .\tag{14}
$$

The denominator increment is not stored as an E2M1 value; the low-bit operand provides the input to the row-sum operation, and the row-sum output is accumulated in FP16 or FP32.

Using the same probability operand in Eqs. (12) and (14) is important. If the numerator used EFQ-Softmax but the denominator used a separately reconstructed probability block, the online update would combine two different probability approximations. EFQ-Softmax instead uses one block-scaled probability operand for both branches.

b) Online recurrence.: With the rescaling factor $\alpha _ { i } ^ { j }$ from Eq. (5), the EFQ-Softmax online update is

$$
\begin{array} { r l } & { \widetilde { A } _ { i } ^ { j } = \alpha _ { i } ^ { j } \widetilde { A } _ { i } ^ { j - 1 } + \Delta A _ { i } ^ { j } , } \\ & { \widetilde { l } _ { i } ^ { j } = \alpha _ { i } ^ { j } \widetilde { l } _ { i } ^ { j - 1 } + \Delta l _ { i } ^ { j } . } \end{array}
$$

The output after all key/value blocks are processed is

$$
\widetilde { O } _ { i } = \frac { \widetilde { A } _ { i } } { \widetilde { l } _ { i } } .
$$

Thus the online structure of FlashAttention is preserved. EFQ-Softmax replaces only the current-block generation of $P _ { i } ^ { j }$ The row maximum update, historical rescaling, high-precision accumulation, and final division remain the same.

c) Current-block exponential path.: The current probability block normally requires elementwise exponentials for all entries of $x _ { i } ^ { j }$ . EFQ-Softmax removes these current-block elementwise exponentials and replaces them with affine index generation. The historical rescaling factor $\alpha _ { i } ^ { j } = \exp ( m _ { i } ^ { j - 1 } -$ $m _ { i } ^ { j } )$ is a row-level term and remains part of the online recurrence. This distinction is important: the method does not remove every exponential operation in FlashAttention. It removes the dense elementwise exponential path used to materialize the current unnormalized probability block.

## B. Algorithm Summary

Algorithms 1 and 2 summarize the online execution of EFQ-Softmax. Algorithm 1 describes the probability-code generation rule for one microscaling block, while Algorithm 2 shows how the generated operand is consumed inside one online attention update. Algorithm 3 summarizes the offline calibration procedure used to select the global parameters (τ, h) by minimizing the attention-output error over representative calibration blocks.

## C. Parameter Calibration

The affine thresholds in Eq. (11) are controlled by two global parameters, τ and h. Calibration selects these parameters to align the threshold range with the residual distribution observed after block-scale normalization. Specifically, τ shifts the threshold range, while h controls the spacing between adjacent thresholds. Larger h produces denser thresholds and can assign more residuals to higher E2M1 codes, whereas smaller h produces a more conservative code distribution.

Fig. 5 illustrates how τ shifts the threshold range and how h controls the spacing between adjacent thresholds.

We select (τ, h) offline using representative attention activations. For each candidate pair, EFQ-Softmax is applied to calibration score blocks, and the resulting attention output is compared with the reference output:

Algorithm 1 EFQ-Softmax Probability-Code Generation for Algorithm 3 Offline Selection of EFQ-Softmax Parameters   
One Microscaling Block Require: Candidate sets $\tau$ and ${ \mathcal { H } } ,$ calibration score blocks,   
Require: Shifted scores $\{ x _ { t } : t \in b \}$ , EFQ-Softmax parame- reference outputs $O _ { \mathrm { r e f } }$   
ters $( \tau , h )$ Ensure: Selected parameters $( \tau ^ { \star } , h ^ { \star } )$   
Ensure: Block scale $s _ { b }$ and nonnegative E2M1 probability 1: Initialize best objective value $\rho ^ { \star }  + \infty$   
codes $\{ Q _ { P , t } ^ { \mathrm { E 2 M 1 } } : t \in b \}$ 2: for each $\tau \in \mathcal { T }$ do   
1: $M _ { b } \gets \operatorname* { m a x } _ { t \in b } x _ { t }$ 3: for each $h \in \mathcal H$ do   
2: $k _ { b } \gets \lfloor ( M _ { b } + \log ( 2 / 9 ) ) / \log 2 \rfloor$ 4: Run EFQ-Softmax attention on calibration blocks   
3: $s _ { b } \gets 2 ^ { k _ { b } }$ with $( \tau , h )$   
4: for each element $t \in b$ do 5: Compute $\rho ( \tau , h ) = \| \widetilde { O } ( \tau , h ) - O _ { \mathrm { r e f } } \| _ { F } / \| O _ { \mathrm { r e f } } \| _ { F }$   
5: $z _ { t } \gets x _ { t } - k _ { b } \log 2 - \log 6$ 6: if $\rho ( \tau , h ) < \rho ^ { \star }$ then   
6: $c _ { t } \gets \mathrm { c l i p } _ { [ 0 , 7 ] } \left( \lfloor ( z _ { t } - \tau ) h \rfloor + 1 \right)$ 7: $\rho ^ { \star } \gets \rho ( \tau , h )$   
7: $Q _ { P , t } ^ { \mathrm { E 2 M 1 } }  \mathrm { p a c k 4 } _ { \mathrm { E 2 M 1 } } ( c _ { t } )$ 8: $( \tau ^ { \star } , h ^ { \star } ) \gets ( \tau , h )$   
8: end for 9: end if   
9: return $s _ { b } , \ \{ Q _ { P , t } ^ { \mathrm { E 2 M 1 } } : t \in b \}$ 10: end for   
11: end for

Algorithm 2 EFQ-Softmax Online Update for One Key/Value   
Block   
Require: $Q _ { i } , K _ { j } , V _ { j }$ , previous state $( m _ { i } ^ { j - 1 } , \widetilde { A } _ { i } ^ { j - 1 } , \widetilde { l } _ { i } ^ { j - 1 } )$   
Ensure: Updated state $( m _ { i } ^ { j } , \widetilde { A } _ { i } ^ { j } , \widetilde { l } _ { i } ^ { j } )$   
1: $S _ { i } ^ { j }  \bar { Q } _ { i } K _ { j } ^ { \top } / \sqrt { d } .$   
2: $m _ { i } ^ { j } \gets \operatorname* { m a x } \{ m _ { i } ^ { j - 1 }$ , rowmax $\cdot ( S _ { i } ^ { j } ) \}$   
3: $x _ { i } ^ { j } \gets S _ { i } ^ { j } - { \dot { m } } _ { i } ^ { j }$   
4: Partition $x _ { i } ^ { j }$ into microscaling blocks   
5: for each microscaling block b do   
6: Generate $s _ { b }$ and $\bar { Q } _ { P , b } ^ { \mathrm { E 2 M 1 } }$ by Algorithm 1   
7: end for   
8: Assemble $\widetilde { P } _ { i } ^ { j } \approx s _ { P , i } ^ { j } Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } }$ from all blocks   
9: $\Delta A _ { i } ^ { j }  \widetilde { P } _ { i } ^ { j } V _ { j }$   
10: $\Delta l _ { i } ^ { j } \gets \widetilde { P } _ { i } ^ { j } \mathbf { 1 }$   
11: $\alpha _ { j . } ^ { j ^ { \prime } }  \mathrm { e x p } ( m _ { i . } ^ { j - 1 } - m _ { i } ^ { j } )$   
12: $\tilde { \mathcal { A } } _ { i } ^ { j }  \alpha _ { i } ^ { j } \tilde { \mathcal { A } } _ { i } ^ { j - 1 } + \Delta A _ { i } ^ { j }$   
13: $\widetilde { l _ { i } ^ { j } } \gets \alpha _ { i } ^ { j } \widetilde { l _ { i } ^ { j - 1 } } + \Delta l _ { j } ^ { j }$   
14: return $\dot { m } _ { i } ^ { j } , \widetilde { A } _ { i } ^ { j } , \widetilde { l } _ { i } ^ { j }$

$$
\operatorname* { m i n } _ { \tau , h } \frac { \| \widetilde { O } ( \tau , h ) - O _ { \mathrm { r e f } } \| _ { F } } { \| O _ { \mathrm { r e f } } \| _ { F } } .
$$

This output-level objective is used because the generated probability operand affects both the numerator branch $\widetilde P V$ and the denominator branch $\widetilde { P } \mathbf { 1 }$ . Scalar fitting to $e ^ { x }$ can be used as an initialization or diagnostic, but scalar exponential error is not the final selection criterion. This objective also captures the coupled effect of probability-code errors on both the accumulated numerator and denominator, which is not reflected by scalar exponential fitting alone.

After calibration, the selected operating point remains fixed during inference and is validated on downstream tasks. This separation prevents the parameter search from being interpreted as per-input adaptation or test-set tuning.

$$
( \tau ^ { \star } , h ^ { \star } )
$$

![](images/403a751b340393084c0682dcc519ac66969c7bd4aed473d25670d319ee7c9aa5.jpg)  
Fig. 5: Effect of τ and h on affine threshold placement: τ shifts the threshold range, whereas h controls the spacing between adjacent thresholds.

## D. Algorithmic Cost and Method Boundary

a) Operations removed.: The conventional path in Eq. (2) first evaluates dense elementwise exponentials for the current probability block and then quantizes the resulting high-precision probabilities into E2M1. EFQ-Softmax fuses these two stages into the residual-domain score-to-code rule in Eq. (11). For each element, the EFQ-Softmax path uses subtraction, multiplication by h, floor, clipping, and bit-pattern generation. For each microscaling block, it computes one local maximum and one exponent-only scale. The row-level exponential used for historical rescaling, $\alpha _ { i } ^ { j } = \exp ( m _ { i } ^ { j - 1 } - m _ { i } ^ { j } )$ , is outside the removed dense elementwise path and is retained.

b) Kernel interface and placement.: The probabilitygeneration unit receives the shifted score block $\bar { \boldsymbol { x } } _ { i } ^ { j }$ , the microscaling partition $B ,$ and the fixed parameters $( \tau , h )$ , and returns the probability operand

$$
( x _ { i } ^ { j } , \boldsymbol { B } , \tau , h ) \longmapsto \Big ( Q _ { P , i } ^ { j , \mathrm { E 2 M 1 } } , s _ { P , i } ^ { j } \Big ) .
$$

Masks are applied before this stage when the score block $S _ { i } ^ { j } =$ $Q _ { i } K _ { j } ^ { \top } / \sqrt { d } \bar { + } M _ { i } ^ { j }$ is formed and shifted by the updated row maximum. EFQ-Softmax is placed immediately after this shift, at the point where the exact implementation would evaluate $\exp ( x _ { i } ^ { j } )$ . The generated code and scale are then consumed by both the numerator branch and the denominator branch.

## IV. EXPERIMENTS

## A. Experimental Setup

a) Models.: We evaluate on three representative open models that span pure language modeling, vision-language understanding, and text-to-video generation: Qwen3-8B, Qwen3-VL-8B-Instruct [18], and WAN2.2-TI2V-5B [21]. For each model we replace the original self-attention and cross-attention over the token stream with our quantized attention, leaving all other components unchanged.

b) Quantization step.: Our attention follows the FlashAttention-2 [3] tiling loop and approximates the softmax numerator $\tilde { P } _ { i } ^ { j } = \exp ( S _ { i } ^ { j } - m _ { i } )$ with a block-wise quantizer. We adopt a high-precision FP16 implementation as the baseline and compare it against a standard MXFP4 quantization implementation [13], [24] and our proposed EFQ-Softmax method; see the method section for the detailed configuration of EFQ-Softmax.

c) Benchmarks and metrics.: Qwen3-8B: seven zeroshot classification tasks from lm-eval-harness [5] -BoolQ [1], OpenBookQA [12], RTE [22], WinoGrande [15], MMLU [6], ARC-Easy and ARC-Challenge [2]— reported as accuracy and averaged into a 7-task mean. Qwen3-VL-8B-Instruct: three VQA tasks (AI2D [8], OK-VQA [10], TextVQA [17], exact-match) and six referring-expression grounding splits (RefCOCO [23] and RefCOCO+ [9] × {testA, testB, val}, ACC@0.5), averaged into a 9-task mean. WAN2.2-TI2V-5B: 14 diverse text-to-video prompts at 1280×704, 121 frames, 50 UniPC [26] steps, with a fixed seed of 123469 across all prompts.

d) Parameter selection.: For the language-model evaluation, we report two calibrated EFQ-Softmax operating points. EFQ-Softmax-MMLU is configured with $\tau = - 2 . 9 0 , h =$ 2.00 and is selected for its highest MMLU accuracy in the parameter sweep, whereas EFQ-Softmax-Mean is configured with $\tau = - 3 . 0 6 , h = 2 . 3 0$ and is selected for its highest seven-task mean in the parameter sweep. For Qwen3-VL, we first screen candidate (τ, h) pairs on at most 16 samples per task and keep only points whose per-task accuracy is within one point of MXFP4 $( \Delta \ge - 0 . 0 1 )$ . The selected operating point, configured with $\tau \ = \ - 2 . 1 0 , h \ = \ 2 . 7 0$ , is denoted EFQ-Softmax-Balance and is used for the full vision-language evaluation.

For comparison, we also evaluate a 16-level lookup-table variant denoted EFQ-Softmax-LUT as an ablation. In contrast to our proposed 8-code configuration of EFQ-Softmax, EFQ-Softmax-LUT adopts a 16-level E2M1 code to isolate the effect of the proposed coding rule.

TABLE I: Qwen3-8B zero-shot accuracy (FP16, batch size 8), part 1 of 2. ∆ is relative to the MXFP4 reference (row 2).
<table><tr><td>Method</td><td>BoolQ</td><td>OBQA</td><td>RTE</td><td>WinoG.</td><td>MMLU</td></tr><tr><td>FP16</td><td>.8657</td><td>.3120</td><td>.7834</td><td>.6772</td><td>.7300</td></tr><tr><td>MXFP4</td><td>.8667</td><td>.3080</td><td>.7726</td><td>.6827</td><td>.7264</td></tr><tr><td>MXFP4-scale6</td><td>.8648</td><td>.3080</td><td>.7798</td><td>.6835</td><td>.7255</td></tr><tr><td>EFQ-LUT</td><td>.8664</td><td>.3180</td><td>.7762</td><td>.6851</td><td>.7248</td></tr><tr><td>EFQ-MMLU</td><td>.8703</td><td>.3160</td><td>.7690</td><td>.6748</td><td>.7223</td></tr><tr><td>EFQ-Mean</td><td>.8667</td><td>.3220</td><td>.7762</td><td>.6906</td><td>.7119</td></tr></table>

TABLE II: Qwen3-8B zero-shot accuracy, part 2 of 2 (ARC-Easy, ARC-Challenge, and the 7-task mean). $\Delta$ on the mean is relative to the MXFP4 reference.
<table><tr><td>Method</td><td>ARC-E</td><td>ARC-C</td><td>Mean</td><td> $\Delta$ </td></tr><tr><td>FP16</td><td>.8350</td><td>.5589</td><td>.6803</td><td>+.0055</td></tr><tr><td>MXFP4</td><td>.8207</td><td>.5469</td><td>.6749</td><td>0</td></tr><tr><td>MXFP4-scale6</td><td>.8194</td><td>.5435</td><td>.6749</td><td>+.0001</td></tr><tr><td>EFQ-LUT</td><td>.8325</td><td>.5461</td><td>.6784</td><td>+.0036</td></tr><tr><td>EFQ-MMLU</td><td>.8283</td><td>.5469</td><td>.6754</td><td>+.0005</td></tr><tr><td>EFQ-Mean</td><td>.8287</td><td>.5452</td><td>.6773</td><td>+.0024</td></tr></table>

## B. Quantitative Results on Qwen3-8B

Tables I–II report zero-shot accuracy on Qwen3-8B at batch size 8, with the FP16 in the first row and $\Delta$ relative to the MXFP4 in parentheses. The seven tasks are split across two tables. From these results, three observations emerge.

(i) EFQ-Softmax-LUT matches or beats MXFP4 on average. EFQ-Softmax-Mean reaches a 7-task mean of 0.6773, compared with 0.6749 for the MXFP4 (+0.0024). The 16- level EFQ-Softmax-LUT ablation reaches 0.6784 (+0.0036). EFQ-Softmax-MMLU is on par with MXFP4 while using the same 8-code E2M1 output set as EFQ-Softmax. Crucially, both 8-code operating points stay within 0.005 of the FP16 baseline (0.6803), so the accuracy lost to P-quant quantization is recovered by our simpler LUT.

(ii) Per-task behavior separates the two operating points. The two 8-code operating points exhibit different accuracy trade-offs. EFQ-Softmax-MMLU gives the best BoolQ result in the table (0.8703) and keeps MMLU within 0.0041 of MXFP4. By contrast, EFQ-Softmax-Mean favors WinoGrande (0.6906, +0.0079) and OpenBookQA (0.3220, +0.0140), yielding the highest seven-task mean among the 8-code EFQ-Softmax variants. The two (τ, h) configurations therefore deliver a tunable accuracy trade-off that MXFP4’s quantizeafter-exponential scheme cannot provide.

TABLE IV: Qwen3-VL-8B-Instruct VQA tasks (exact match). $\Delta$ is EFQ-Softmax-Balance minus MXFP4.
<table><tr><td>Task</td><td>MXFP4</td><td>EFQ-Softmax-bal.</td><td> $\Delta$ </td></tr><tr><td>AI2D</td><td>.83646</td><td>.83323</td><td>-.00324</td></tr><tr><td>OK-VQA</td><td>.49370</td><td>.49184</td><td> $- . 0 0 1 8 6$ </td></tr><tr><td>TextVQA</td><td>.79722</td><td>.79792</td><td> $+ . 0 0 0 7 0$ </td></tr></table>

TABLE III: Qwen3-8B 7-task mean across batch sizes and kernel backends. $\Delta$ is relative to the MXFP4 reference at the same batch size.
<table><tr><td rowspan="2">Method</td><td colspan="2">batch = 16</td><td colspan="2">batch = 24</td></tr><tr><td>Mean</td><td> $\Delta$ </td><td>Mean</td><td> $\Delta$ </td></tr><tr><td>FP16</td><td>.6805</td><td>+.0074</td><td>.6809</td><td> $+ . 0 0 7 4$ </td></tr><tr><td>MXFP4</td><td>.6731</td><td>0</td><td>.6735</td><td>0</td></tr><tr><td>MXFP4-scale6</td><td>.6736</td><td>+.0005</td><td>.6736</td><td> $+ . 0 0 0 1$ </td></tr><tr><td>EFQ-LUT</td><td>.6781</td><td>+.0050</td><td>.6757</td><td> $+ . 0 0 2 3$ </td></tr><tr><td>EFQ-MMLU</td><td>.6740</td><td>+.0009</td><td>.6764</td><td> $+ . 0 0 2 9$ </td></tr><tr><td>EFQ-Mean</td><td>.6733</td><td>+.0002</td><td>.6785</td><td>+.0050</td></tr></table>

(iii) Batch size does not change the conclusion. Table III replicates the comparison under batch sizes of 16 and 24. The seven-task average of each method deviates by no more than 0.006 from its batch-8 result, confirming that our comparison conclusions remain stable under batching. The FP16 baseline consistently yields the highest accuracy, and the MXFP4 reference the lowest. Meanwhile, the EFQ-Softmax variants — EFQ-Softmax-LUT, EFQ-Softmax-MMLU, and EFQ-Softmax-Mean — sit in the intermediate range, with their relative ordering interchangeable within a 0.005 accuracy band.

## C. Quantitative Results on Qwen3-VL-8B-Instruct

Tables IV–V compare EFQ-Softmax-Balance with the MXFP4 across the full Qwen3-VL evaluation suite. The results are split into VQA (Table IV) and referring-expression grounding (Table V) to fit within the column width. On the three VQA tasks, the two methods yield nearly indistinguishable performance. EFQ-Softmax-Balance incurs a maximum accuracy drop of only 0.0032 on AI2D and even achieves a slight gain of 0.0007 on TextVQA; the worst-case degradation on VQA is therefore well below 0.01. In contrast, EFQ-Softmax-Balance outperforms MXFP4 on all six grounding splits, with improvements ranging from +0.0028 (RefCOCO+ testB) to +0.0430 (RefCOCO testA). Aggregated over all nine tasks, the mean accuracy rises from 0.7826 to 0.8000(+0.0174). The IoU metrics underlying these ACC@0.5 scores (Table VI) follow the same trend, indicating that the quantized grounding boxes are not only correct more often but also spatially better aligned with target objects.

In short, for the vision-language model, EFQ-Softmax incurs no accuracy loss on VQA and strictly outperforms MXFP4 on grounding. Moreover, it achieves a clear advantage on referring-expression splits: on these tasks, the softmax distribution over candidate boxes is sharper, which enables the method to reap benefits from the arithmetic-midpoint scale.

TABLE V: Qwen3-VL-8B-Instruct referring-expression grounding (ACC@0.5). $\Delta$ is EFQ-Softmax-Balance minus MXFP4; every split improves. The 9-task mean $\mathrm { ( V Q A ~ + }$ grounding) is shown in the last row.
<table><tr><td>Split</td><td>MXFP4</td><td>EFQ-Softmax-Bal.</td><td> $\Delta$ </td></tr><tr><td>RefCOCO testA</td><td>.85418</td><td>.89722</td><td>+.04304</td></tr><tr><td>RefCOCO testB</td><td>.83094</td><td>.84862</td><td>+.01768</td></tr><tr><td>RefCOCO val</td><td>.85178</td><td>.88617</td><td>+.03439</td></tr><tr><td>RefCOCO+ testA</td><td>.84608</td><td>.88051</td><td>+.03443</td></tr><tr><td>RefCOCO+ testB</td><td>.74805</td><td>.75083</td><td>+.00278</td></tr><tr><td>RefCOCO+ val</td><td>.78502</td><td>.81367</td><td>+.02865</td></tr><tr><td>9-task mean</td><td>.78260</td><td>.80000</td><td>+.01740</td></tr></table>

TABLE VI: IoU of the EFQ-Softmax-Balance grounding boxes on the RefCOCO / RefCOCO+ splits (Qwen3-VL full evaluation).
<table><tr><td>Split</td><td>EFQ-Softmax-bal. IoU</td></tr><tr><td>RefCOCO testA</td><td>0.8257</td></tr><tr><td>RefCOCO testB</td><td>0.7778</td></tr><tr><td>RefCOCO val</td><td>0.8152</td></tr><tr><td> $\mathrm { R e f C O C O + t e s t A }$ </td><td>0.8058</td></tr><tr><td> $\mathrm { R e f C O C O + \ t e s t { B } }$ </td><td>0.7006</td></tr><tr><td> $\mathrm { R e f C O C O + \ v a l }$ </td><td>0.7527</td></tr></table>

## D. Text-to-Video Generation on WAN2.2-TI2V-5B

![](images/4b11b75f7d922ccf61422380516e38aaca0591aa4f65cb24d2aabd640ea31f05.jpg)  
Fig. 6: Middle-frame qualitative overview.

We generate 14 text-to-video clips with the WAN2.2-TI2V-5B model under three P-quant settings: the FP16 baseline, EFQ-Softmax, and MXFP4 SCALE4. All clips adopt identical generation parameters: 1280 × 704 resolution, 121 frames, 50 UniPC steps, and a fixed random seed of 123469. As such, the sole experimental variable is the attention kernel. The 14 prompts encompass the visual styles utilized in previous WAN2.2 [21]evaluations, including museum interiors, closeup object motion, archival footage, fantasy creatures, neon street scenes, drone landscapes, and papercraft/craft-material clips.

![](images/c46822bfa942c7f003d1330d7af17b44ba5147477a34ef478574c04fc12e8ff8.jpg)

(a) Television-wall scene.  
![](images/c134cf5c6a946c069dd5cc8c065a0b90e4991a2b825c3c2c2a0b4896f977ed25.jpg)  
(c) Outdoor scene.

![](images/8f156b1dbe26d2ac55d9e903bfdda81a1728c947360310ea6a3be0dff2123dc2.jpg)

(b) Coffee-cup scene.  
![](images/73f7f05ff6141232015b295e033f369ea1a8f25cec5f6dc915ebcaf9802dc34e.jpg)  
(d) Terrarium scene.  
Fig. 7: Temporal consistency comparison across four representative text-to-video examples. Columns within each panel show frames sampled at different timesteps, while rows correspond to different attention implementations.

a) VBench quantitative metrics.: To complement our qualitative comparison, we evaluate all video clips via VBench [7] with its default auxiliary-information configuration, which enables 10 valid dimensions covering the primary evaluation criteria. Table VII summarizes the results of the three variants. In terms of temporal-coherence metrics, EFQ Softmax achieves scores comparable to both the FP16 baseline and MXFP4 SCALE4. Subject consistency (0.9493), background consistency (0.9564), temporal flickering (0.9784), and motion smoothness (0.9891) reveal no observable degradation of temporal consistency. For visual quality dimensions, EFQ Softmax attains the highest imaging quality (0.7178), surpassing the baseline (0.7098) and MXFP4 SCALE4 (0.7004), whereas its aesthetic quality (0.6355) lies between the two reference variants. Overall, the VBench metrics verify that EFQ-Softmax retains the visual quality of the FP16 baseline while delivering competitive performance relative to MXFP4 SCALE4.

b) Visual quality.: Since a single scalar metric cannot comprehensively characterize text-to-video generation quality, we present side-by-side video comparisons below. Figure 6 shows middle-frame examples for four prompts, and Figure 7 compares the temporal consistency of different attention methods on these cases. Each of the 14 prompts yields three video clips (FP16 baseline, EFQ-Softmax, MXFP4 SCALE4) generated using identical prompts and random seeds. Qualitatively, videos produced by EFQ-Softmax retain the motion coherence, lighting, and fine texture details seen in the FP16 baseline, achieving performance comparable to MXFP4 SCALE4. Im-

TABLE VII: WAN2.2-TI2V-5B VBench results under the default auxiliary-information setting (15 valid dimensions, main evaluation criteria). Baseline is FP16, MXFP4 scale4 is the MXFP4 reference, and EFQ-Softmax.

<table><tr><td>VBench dimension</td><td>Baseline</td><td>MXFP4 scale4</td><td>EFQ-Softmax</td></tr><tr><td>subject consistency</td><td>0.9509</td><td>0.9536</td><td>0.9493</td></tr><tr><td>background consistency</td><td>0.9559</td><td>0.9624</td><td>0.9564</td></tr><tr><td>temporal flickering</td><td>0.9801</td><td>0.9775</td><td>0.9784</td></tr><tr><td>motion smoothness</td><td>0.9901</td><td>0.9887</td><td>0.9891</td></tr><tr><td>dynamic degree</td><td>0.5714</td><td>0.5000</td><td>0.5000</td></tr><tr><td>aesthetic quality</td><td>0.6488</td><td>0.6295</td><td>0.6355</td></tr><tr><td>imaging quality</td><td>0.7098</td><td>0.7004</td><td>0.7178</td></tr><tr><td>temporal style</td><td>0.2645</td><td>0.2475</td><td>0.2562</td></tr><tr><td>appearance style</td><td>0.2431</td><td>0.2416</td><td>0.2406</td></tr><tr><td>overall consistency</td><td>0.2645</td><td>0.2475</td><td>0.2562</td></tr></table>

portantly, EFQ-Softmax relies on a streamlined probabilitycode generation pipeline based on a single affine index rule, which differs from MXFP4’s scale-and-round procedure.

## E. Kernel-Level Speedup on the A5 Vector Unit

We benchmark the fused P-quant kernel on the A5 vector unit. We compare the MXFP4 with the fused EFQ-Softmax implementation across six sequence lengths spanning 16K to 128K tokens. The P-quant computation pipeline is split into three stages: matrix-multiply-accumulate (mmad), fixedpoint pipeline (fixpipe), and vector processing (vector). By construction, the mmad stage is identical for both variants. Both implementations dequantize operands to the identical E2M1 format for matrix multiplication. Accordingly, their measured latencies align down to the microsecond, and we exclude this stage from Table VIII.

TABLE VIII: A5 vector-unit per-stage timing (µs) and perstage speedup for fixpipe and vector: MXFP4 baseline vs. A5- EFQ-Softmax. Speedup is $1 - t _ { \mathrm { L U T } } / t _ { \mathrm { b a s e } } .$
<table><tr><td></td><td colspan="3">fixpipe</td><td colspan="3">vector</td></tr><tr><td>Seq</td><td>base</td><td>LUT</td><td>gain</td><td>base</td><td>LUT</td><td>gain</td></tr><tr><td>16K</td><td>106.1</td><td>105.4</td><td>0.69%</td><td>112.6</td><td>67.4</td><td>40.10%</td></tr><tr><td>24K</td><td>243.9</td><td>242.1</td><td>0.71%</td><td>250.1</td><td>149.7</td><td>40.12%</td></tr><tr><td>32K</td><td>431.2</td><td>429.8</td><td>0.32%</td><td>443.8</td><td>264.4</td><td>40.44%</td></tr><tr><td>48K</td><td>977.2</td><td>969.0</td><td>0.84%</td><td>994.4</td><td>592.7</td><td>40.39%</td></tr><tr><td>64K 128K</td><td>1738.0 6955.0</td><td>1720.5 6883.2</td><td>1.01% 1.03%</td><td>1764.1 7043.8</td><td>1050.5 4192.3</td><td>40.45% 40.48%</td></tr><tr><td>mean</td><td>1741.9</td><td>1725.0</td><td>0.77%</td><td>1768.1</td><td>1052.8</td><td>40.33%</td></tr></table>

The observed speedup is consistent with the underlying LUT design. The fixpipe stage executes per-element scaling, clipping and rounding logic, achieving a 0.3%–1.0% performance improvement. This modest yet steady gain arises from substituting the abs-max/log/ceil/bucketize pipeline with a single linear index lookup. The most significant improvement occurs within the vector stage: the fused LUT reduces latency by an average of 40.33%, and this reduction remains nearly invariant across all tested sequence lengths (40.10% at 16K up to 40.48% at 128K). As the end-to-end acceleration is constrained by the proportion of P-quant computation located in the vector stage, further speedups are anticipated for tensor shapes where the softmax-numerator tile accounts for a larger share compared with the matmul operation.

## V. CONCLUSION

This paper introduced EFQ-Softmax, a probability-side method for low-bit attention. Instead of computing highprecision shifted-score exponentials and then quantizing the resulting probabilities, EFQ-Softmax directly generates a block-scaled E2M1 probability operand. Its core generator selects an exponent-only scale per microscaling block, maps residual log-domain scores to 4-bit E2M1 probability codes, and feeds the same generated operand to both the numerator branch $\widetilde P V$ and the denominator branch $\widetilde { P } \mathbf { 1 }$ . EFQ-Softmax therefore changes only the probability-code generation path. It does not define a new value-side quantizer, a new attention rule, or a pruning method. The row-maximum update, historical rescaling, high-precision branch outputs, online accumulators, and final normalization remain part of the FlashAttentionstyle online recurrence.

Although EFQ-Softmax changes the generation of the current unnormalized weights, the final output remains a normalized weighted sum. The online recurrence accumulates $\widetilde P V$ and $\widetilde { P } \mathbf { 1 }$ , and applies the final division as in standard online softmax. Since all values in the E2M1 probability code set are nonnegative, the current-block denominator increments are nonnegative. Different microscaling blocks may use different scales, but their contributions are converted to high-precision branch outputs before being accumulated into the common online state. The only data-dependent metadata introduced by EFQ-Softmax is the scale exponent $k _ { b }$ per microscaling block; the parameters (τ, h) are fixed after calibration and are not stored per token, per row, or per block.

Across Qwen3-8B, Qwen3-VL-8B-Instruct, and WAN2.2- TI2V-5B, EFQ-Softmax preserves task accuracy and output quality relative to FP16 and MXFP4 baselines. These results show that direct low-bit probability-code generation can replace post-softmax probability quantization while maintaining the numerical behavior required by online attention. EFQ-Softmax therefore provides a practical path toward fully lowbit attention kernels in which probability generation, not only $Q K ^ { \top }$ and $P V$ , is aligned with low-bit execution.

## REFERENCES

[1] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova, “Boolq: Exploring the surprising difficulty of natural yes/no questions,” in Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers), 2019, pp. 2924–2936.

[2] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try arc, the ai2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018.

[3] T. Dao, “Flashattention-2: Faster attention with better parallelism and work partitioning,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 35 549–35 562.

[4] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “Flashattention: Fast and´ memory-efficient exact attention with io-awareness,” Advances in neural information processing systems, vol. 35, pp. 16 344–16 359, 2022.

[5] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “The language model evaluation harness,” 07 2024. [Online]. Available: https://zenodo.org/records/12608602

[6] D. Hendrycks, S. Basart, M. Belinkov, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” 2021. [Online]. Available: https://arxiv.org/abs/2009. 03300

[7] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, Y. Wang, X. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, “Vbench: Comprehensive benchmark suite for video generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 807–21 818.

[8] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi, “A diagram is worth a dozen images,” in European conference on computer vision. Springer, 2016, pp. 235–251.

[9] J. Mao, J. Huang, A. Toshev, O. Camburu, A. L. Yuille, and K. Murphy, “Generation and comprehension of unambiguous object descriptions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 11–20.

[10] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi, “Ok-vqa: A visual question answering benchmark requiring external knowledge,” in Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, 2019, pp. 3195–3204.

[11] P. Micikevicius, D. Stosic, P. Judd, J. Kamalu, S. Oberman, M. Shoeybi, M. Siu, H. Wu, N. Burgess, S. Ha, R. Grisenthwaite, N. Mellempudi, M. Cornea, A. Heinecke, and P. Dubey, “Fp8 formats for deep learning,” arXiv preprint arXiv:2209.05433, 2022.

[12] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor conduct electricity? a new dataset for open book question answering,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

[13] O. C. Project, “Mx floating-point (mxfp) format,” 2024. [Online]. Available: https://www.opencompute.org/documents/ocp-microscalingformats-mx-v1-0-0-spec-final-pdf

[14] B. D. Rouhani, R. Zhao, A. More, M. Hall, A. Khodamoradi, S. Deng, D. Choudhary, M. Cornea, E. Dellinger, K. Denolf, D. Stosic, V. Elango, M. Golub, A. Heinecke, P. James-Roxby, D. Jani, G. Kolhe, M. Langhammer, A. Li, L. Melnick, M. Mesmakhosroshahi, A. Rodriguez, M. Schulte, R. Shafipour, L. Shao, M. Siu, P. Dubey, P. Micikevicius, M. Naumov, C. Verrilli, R. Wittig, D. Burger, and E. Chung, “Microscaling data formats for deep learning,” arXiv preprint arXiv:2310.10537, 2023.

[15] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi, “Winogrande: An adversarial winograd schema challenge at scale,” pp. 8732–8740, 2020.

[16] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, “Flashattention-3: Fast and accurate attention with asynchrony and lowprecision,” Advances in Neural Information Processing Systems, vol. 37, pp. 68 658–68 685, 2024.

[17] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.

[18] Q. Team, “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[19] Q. Team, “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[21] A. G. Wan Team, “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[22] A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman, “Superglue: A stickier benchmark for generalpurpose language understanding systems,” Advances in neural information processing systems, vol. 32, 2019.

[23] L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in Proceedings of the European Conference on Computer Vision (ECCV), 2016.

[24] J. Zhang, J. Wei, H. Huang, P. Zhang, J. Zhu, and J. Chen, “Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration,” arXiv preprint arXiv:2410.02367, 2024.

[25] J. Zhang, J. Wei, H. Wang, P. Zhang, X. Xu, H. Huang, K. Jiang, J. Chen, and J. Zhu, “Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training,” Advances in Neural Information Processing Systems, vol. 38, pp. 53 901–53 931, 2026.

[26] W. Zhao, L. Bai, Y. Rao, J. Zhou, and J. Lu, “Unipc: A unified predictorcorrector framework for fast sampling of diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 49 842–49 869, 2023.