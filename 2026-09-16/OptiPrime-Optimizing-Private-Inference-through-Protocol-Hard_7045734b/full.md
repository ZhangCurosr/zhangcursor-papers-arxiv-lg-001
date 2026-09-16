# OptiPrime: Optimizing Private Inference through Protocol–Hardware Co-design

Jiangrui Yu<sup>1</sup>, Ye Yu<sup>1</sup>, Si Chen<sup>2</sup>, Chenqi Lin<sup>1</sup>, Wenxuan Zeng<sup>1</sup>, Junfeng Fan<sup>2</sup>, Mingyu Gao<sup>3</sup>, and Meng Li<sup>1,\*</sup> <sup>1</sup>Peking University, Beijing, China <sup>2</sup>Open Security Research, Shenzhen, China <sup>3</sup>Tsinghua University, Beijing, China Corresponding author

{jiangrui.yu, 2100012750, linchenqi}@stu.pku.edu.cn, {si.chen, fan}@osr-tech.com,

zwx.andy@outlook.com, gaomy@tsinghua.edu.cn, meng.li@pku.edu.cn

Abstract—Private deep neural network (DNN) inference based on hybrid homomorphic encryption (HE) and multi-party computation (MPC) can protect user data with a formal guarantee, but at the cost of significant latency overhead due to HE. Customized HE accelerators have been proposed and have achieved orders-of-magnitude speedup for individual HE operations. However, when directly applying a commercial HE accelerator to state-of-the-art HE-MPC frameworks, we observe only limited end-to-end performance gain. This is because HE-MPC frameworks often require wireless transmission of input and output ciphertexts for each HE operation, leading to a severe network communication bottleneck.

To overcome this challenge, we introduce OptiPrime, a protocol-hardware co-optimization framework for efficient private DNN inference. OptiPrime features a novel HE protocol for convolutions that substantially reduces the number of transmitted output ciphertexts and mitigates the network communication bottleneck. Meanwhile, as the new protocol introduces complex computation for fewer output ciphertext, we observe new memory access challenges due to a high volume of weight plaintexts and intermediate ciphertexts. Hence, we further propose a lightweight compression system for the weight plaintexts, reducing memory traffic by 10×, as well as a specialized dataflow to maximize on-chip data reuse of intermediate ciphertexts. Extensive experiments show that our framework outperforms the Cheetah baseline by at most 5.7× on CPUs and 4.2× with an accelerator.

Index Terms—Private inference, homomorphic encryption, multi-party computation, hardware acceleration, protocol– hardware co-design

## I. INTRODUCTION

The last decade has witnessed the rapid evolution of deep learning (DL) and its increasing adoption in privacy-sensitive applications, including medical diagnosis [1], face recognition [2], financial system [3], etc. Privacy has therefore emerged as a major concern, leading to a growing demand for privacypreserving DL (PPDL) [4]–[8].

PPDL frameworks based on hybrid Homomorphic Encryption (HE) and Multi-Party Computation (MPC) have recently been proposed and have attracted a lot of attention [7]–[25]. As shown in Figure 1 (a), an HE-MPC framework often involves two parties, namely the server and the client, which own private deep neural networks (DNNs) and input data, respectively. The two parties jointly execute a series of protocols, including HE for linear operations (e.g., convolutions) and MPC for nonlinear functions (e.g., ReLU), so that the final results can be computed while the privacy of both input data and DNN parameters can be preserved [9]–[11].

![](images/275481771fba8a4b9617d0b0e30c916885acd3ede12571fea11e875ce8e0aba4.jpg)  
Fig. 1. (a) Hybrid HE-MPC framework. (b) Latency reduction of HE operations with the hardware accelerator. (c) Latency breakdown of ImageNetscale ResNet50 under different network and computation conditions. ”Acc.” means the accelerator. (d) Latency breakdown of the Linear layer of ResNet50. ”Compute” represents the HE computation. ”Input NetIO” and ”Output NetIO” represent the server-client network transmission of the input ciphertexts and the output ciphertexts, respectively. ”Others” represents other CPU overhead, like ciphertext decryption.

An alternative approach for PPDL is to leverage end-toend fully HE (FHE) [6], [26]–[37]. It computes all DNN operations based on HE and avoids the interaction between the server and the client. However, it often requires extensive approximation for nonlinear activation functions and expensive bootstrapping operations, which may suffer from accuracy bottlenecks [38]. Therefore, in this paper, we focus on the HE-MPC framework.

The hybrid HE-MPC framework often incurs high latency primarily due to the costly HE operations [12], [18]. To speed up HE operations, numerous HE acceleration schemes, including ASICs [39]–[50], FPGAs [51]–[56], and GPU libraries [57]–[63], have been designed. However, when we apply a commercial FPGA-based accelerator [64] to the state-of-theart (SOTA) Cheetah protocol [18], only a modest 1.37× endto-end latency reduction is achieved as shown in Figure 1 (c), in stark contrast to its significant speedups of individual HE operations (Figure 1 (b)). Further performance breakdown in Figure 1 (d) reveals a critical insight: while the HE computation time is significantly reduced, the overall benefit is negated by the massive network communication overhead inherent to the hybrid HE-MPC framework.

![](images/de911d3b6f69267f2a77cd76810e0bcceb900056858dae1b4f4e7c6618e1d5ed.jpg)  
Fig. 2. Overview of OptiPrime, including the limitations of previous protocol and our protocol, the challenges to deploy this protocol, and our solutions.

As shown in Figure 1 (a), communication in the HE-MPC framework arises from two sources: (1) transmitting input and output ciphertexts for each HE-based linear layer, and (2) MPC protocols for nonlinear functions. Figure 1 (c) and (d) indicate that linear layers dominate the overall latency in Cheetah, with output ciphertext transmission as the primary bottleneck - particularly when HE accelerators are used. We observe that the inefficiency stems from Cheetah’s convolution protocol, which generates numerous ciphertexts because each ciphertext encodes only a small number of useful elements. Therefore, to address this bottleneck, we propose OptiEncode, a new convolution protocol that significantly reduces the number of output ciphertexts (as in Figure 2 (b)). While this lowers the communication overhead, it introduces more complex HE computation, leading to two additional memory access challenges.

Challenge 1: High Plaintext Memory Volume. As shown in Figure 2 (c), OptiEncode substantially increases memory demand for weight plaintexts. For HE operations, weights are encoded into polynomials, which are highly sparse with over 99% of coefficients padded with zero. Furthermore, existing protocols often convert these polynomials from the coefficient form to the evaluation form to reduce runtime computation, further inflating their sizes [65], [66]. Consequently, memory requirements for weight plaintext polynomials increase by over 1000×, creating a prohibitive bottleneck in weight plaintext fetching.

Solution 1: Lightweight Plaintext Compression (Opti-Comp). To reduce memory capacity and bandwidth demands, we introduce OptiComp, a lightweight compression system (Figure 2(e)). OptiComp exploits the sparsity of coefficientform polynomials by compressing them according to their sparse patterns. At runtime, compressed plaintexts are decompressed and converted to the evaluation form on the fly. Although this adds minor computation overhead, the substantial memory savings yield significant performance gains.

Challenge 2: Bad temporary ciphertext reuse. Unlike Cheetah, OptiEncode involves complex HE computations to reduce the number of output ciphertexts. As a result, many more intermediate ciphertexts are generated, whose working set size quickly exceeds the accelerator on-chip memory capacity (Figure 2(d)) and results in frequent off-chip memory accesses for temporary data.

Solution 2: Reuse-Centric Dataflow (OptiFlow). To resolve this, we propose OptiFlow, a specialized dataflow that reorders the computation to maximize data reuse (Figure 2(f)). By carefully scheduling the HE operations, OptiFlow ensures that subsets of the ciphertext working set are fully processed within the on-chip scratchpad before being evicted, thus minimizing costly off-chip memory traffic.

Extensive experiments show that our framework outperforms the Cheetah baseline by up to 5.7× on CPUs and 4.2× with an accelerator. On end-to-end tasks, it reduces the inference latency for ResNet-18 and ResNet-50 on ImageNet to just 2.9 seconds and 14.6 seconds, respectively.

## II. BACKGROUND

## A. Homomorphic Encryption and Encodings

HE allows one party to perform computation, e.g., addition and multiplication, on encrypted data without decryption. We mainly focus on the Leveled HE (LHE) scheme based on ring learning with error (RLWE), i.e., BFV [67], to compute linear layers. BFV computes on polynomials, and the main HE parameters include the polynomial degree N, the plaintext modulus t, and the ciphertext modulus q. As HE operates over 1-dimensional polynomials and DNN computes over highdimensional tensors, mapping from tensors to polynomials, denoted as encoding, is important and directly determines the computation efficiency.

![](images/20bc7778fa59b9396cc8d4d362c1f7a507e36d6f82a68941e0f9c4614f63b161.jpg)  
Fig. 3. (a) An example of a convolution operation. (b) An example of coefficient encoding. (c) The output of the coefficient encoding. Two correct results generated are colored in blue, while the rest are dummy ones. (d) Overview of the Hybrid HE-MPC framework.

There are two major encoding schemes: coefficient encoding [7], [15], [18], [20], [35] and SIMD encoding [12]– [14]. SIMD encoding enables element-wise addition and multiplication on encrypted vectors. However, it imposes a strict requirement that the plaintext modulus is of the form $2 k N { \ + 1 }$ where N is the polynomial degree and k is a positive integer. This restriction can degrade the performance of MPC [18] and is often not preferred in HE-MPC frameworks.

In contrast, coefficient encoding places elements directly in polynomial coefficients. For example, as shown in Figure 3 (b), the Cheetah protocol encodes input and weight tensor along their width dimension. After homomorphic computation, the correct results appear in specific coefficients of the output ciphertext, as illustrated in Figure 3 (c). It is inherently MPC-friendly and incurs lower communication cost [18]. Furthermore, it typically requires fewer HE operations for convolutions, as polynomial multiplication naturally implements convolution, as shown in Table I. Therefore, in this paper, we mainly focus on the coefficient encoding.

A major limitation of coefficient encoding is the prevalence of “dummy coefficients,” which reduces encoding density and increases the number of output ciphertexts. As illustrated in Figure 3(c), only two coefficients colored in blue carry useful data, while the rest are dummy. Since polynomial multiplication convolves all coefficients, these dummy values cannot be eliminated with simple masking as in SIMD encoding. Consequently, although each ciphertext can hold eight coefficients in the example, four outputs are inefficiently distributed across two ciphertexts, leading to substantial communication overhead. This highlights the need for a new protocol that produces fewer and more densely encoded ciphertexts.

## B. Hybrid HE-MPC Framework

As illustrated in Figure 3 (d), the linear layer begins with the client and server each holding an “additive share” $\langle \mathbf { X } \rangle _ { c }$ and $\langle \mathbf { X } \rangle _ { s }$ of the input activation tensor X, where $\mathbf { X } = \langle \mathbf { X } \rangle _ { c } + \langle \mathbf { X } \rangle _ { \cdot }$ mod $t ,$ and t is the plaintext modulus.

First, the client encodes and encrypts its shares into $\mathbb { I } ( \mathbf { X } ) _ { c ] }$ and sends it to the server. The server then locally adds to its share to recover the encrypted input: $\mathbb { I } \langle \mathbf { X } \rangle _ { c } \mathbb { I } + \langle \mathbf { X } \rangle _ { s } = \mathbb { I } \mathbf { X } \mathbb { I }$

![](images/a007faa0ac73d9d7ca9a1fa164aaafea8215eecb0c3b6daa8964edaa39de09e5.jpg)  
Fig. 4. Latency of HE computation operations on the accelerator and CPU for parameters N = 8192 and log q ≈ 64.

Next, the server homomorphically evaluates the linear layer by computing $\mathbf { W } \cdot \left[ \mathbf { X } \right] - \mathbf { R } = \left[ \left[ \mathbf { W } \mathbf { X } - \mathbf { R } \right] \right]$ , where R is randomly generated to mask Y = XW and keeps as the server’s share $\langle \mathbf { Y } \rangle _ { s }$ . This result is sent back to the client, who decrypts it to obtain its output share, $\langle \mathbf { Y } \rangle _ { c } .$ These shares of the output Y then serve as inputs for the subsequent non-linear layer, which is collaboratively evaluated using an MPC protocol like the accurate ReLU from CrypTFlow2 [9]. The primary bottleneck of this framework lies in the HE-based linear layer (detailed in section III), which suffers from server-side HE computation overhead and the back-and-forth network communication overhead of secret shares.

Figure 4(a) presents the hardware system for our HE-MPC framework. Both the client and server are equipped with a CPU and communicate over a wireless network. We augment the server with an FPGA-based HE accelerator [64], the configuration of which is detailed in Figure 4(b).

This accelerator has an architecture similar to previous works such as F1 [39] and ARK [41], consisting of High Bandwidth Memory (HBM), an on-chip scratchpad, and multiple compute clusters (CCs). Each cluster contains specialized units for key HE operations, including NTT (NTTU), automorphism (AutoU), modular multiplication (ModMultU), and modular addition (ModAddU). We omit further implementation details of the accelerator because our contributions are hardwareagnostic and applicable to most existing HE accelerators, such as F1 and ARK. The per-operation latency is shown in Figure 4(c). During inference, each ciphertext is first transmitted from the client to the server over the network, and then forwarded from the server CPU to the accelerator over PCIe. During the HE computation, the accelerator repeatedly loads ciphertexts and plaintexts from the HBM. Consequently, the network transmission and the HBM accesses constitute the two dominant sources of overhead.

## C. Threat Model

OptiPrime adopts the standard two-party honest-but-curious model of prior hybrid HE–MPC frameworks [7], [9], [14], [15], [18], in which each party follows the prescribed protocol honestly but may try to infer additional information about the other party’s private inputs. The client holds a private input, the server holds private model weights, and the network architecture and tensor dimensions are public. BFV semantic security protects the client’s encrypted share, the model weights never leave the server, and intermediate values are exchanged only in encrypted or secret-shared form. OptiEncode changes only the public coefficient layout, whereas OptiComp and OptiFlow are server-local. None introduces secret-dependent control flow, new messages, or additional interaction rounds. Thus, by standard composition, OptiPrime retains the privacy guarantees of the underlying HE–MPC framework: the client learns only the prescribed inference output, while the input and model remain private.

## III. OPTIENCODE: A COMMUNICATION-EFFICIENTCONVOLUTION PROTOCOL

## A. Limitations of previous protocol

We begin by profiling the current SOTA coefficient encoding-based convolution protocol, Cheetah [18]. As illustrated in Figure 5 (a) and (b), the HE-based linear layer is the bottleneck with and without a server-side hardware accelerator, which is consistent with previous work [12], [17], [18], [20]. This is because non-linear layers are less intensive and can be further optimized with accelerators [68]– [70]. In contrast, the linear layers suffer from HE’s high cost. Therefore, in this work, we focus on optimizing the HE part $( \mathbf { i . e . , }$ linear layer computation).

However, as shown in Figure 5 (c) and (d), further latency breakdown of linear layers shows that after applying hardware acceleration, the bottleneck shifts from the HE computation to wireless network communication, leading to limited overall speedup. This excessive communication is a direct consequence of Cheetah’s encoding, which generates numerous output ciphertexts filled with dummy coefficients. While SIMD-based protocols like Hyena [17], Orion [34], and Gazelle [12] also produce compact outputs, they incur more costly HE operations and MPC-incompatible prime moduli, as shown in Table I.

To address this, we propose OptiEncode, a communicationefficient protocol with two key components. First, a channel encoding encodes the input and weight tensor along the input channel dimension, which enables the use of automorphisms (i.e., rotations in SIMD encoding) to eliminate dummy coefficients [15], [71]. Second, to reduce the cost of these automorphisms, we introduce a computational reformulation that enables the baby-step-giant-step (BSGS) algorithm. Compared to Cheetah, our protocol reduces output ciphertexts by at most 128×. With further reformulation optimization, our protocol reduces automorphisms by at most 128×.

## B. Channel Encoding

We consider the convolution $Y ~ = ~ X * ~ W$ between a ciphertext tensor $X \in \mathbb { R } ^ { C _ { i } \times H \times W }$ and a plaintext weight $W \in$ $\mathbb { R } ^ { \mathbf { \bar { ( } } C _ { o } \times C _ { i } \times h \times w }$ , which results in an output $Y \in \mathbb { R } ^ { C _ { o } \times H \times W }$ Throughout this section, we will use the configuration shown in Figure 6 (a), where $C _ { o } = C _ { i } = 4 , H = h = 1 , W = 3$ and $w = 2 ,$ , as a running example for illustration. To leverage automorphisms to clean up dummy coefficients (as detailed subsequently), the useful terms in the output polynomial must occupy degrees that are multiples of a power of two (i.e., degrees $k \cdot 2 ^ { r } )$ . For instance, if the useful coefficients are at degrees $\{ x ^ { 0 } , x ^ { 4 } , x ^ { 8 } , \ldots \}$ , automorphisms can eliminate the intermediate dummy coefficients (at $\mathbf { \bar { \{ } }  x ^ { 1 } , x ^ { 2 } , x ^ { 3 } , \ldots \} $ ).

![](images/7ed1d6dad7a6ae498cac918fd64953b9f9f3d86f6e7554d7c8a09e1aa384c998.jpg)  
Fig. 5. (a) End-to-end latency breakdown of CPU baseline. (b) End-to-end latency breakdown of the accelerator. (c) Linear layer latency breakdown of the CPU baseline. (d) Linear layer latency breakdown of the accelerator.

To satisfy this constraint, we design a specific encoding method, depicted in Figure 6(b). The key insight is to encode the tensors X and $\bar { W }$ along the input channel dimension, $C _ { i }$ . This strategy ensures that the convolution directly yields the useful coefficients at the degrees: $\{ x ^ { 0 } , x ^ { C _ { i } } , x ^ { 2 C _ { i } } , \ldots \}$ . If $C _ { i }$ is a power of two (as in our example, where $C _ { i } = 4 )$ the useful coefficients are correctly positioned at the desired degrees $\{ x ^ { 0 } , x ^ { 4 } , \ldots \}$ . We now formally define our encoding schemes, $\pi _ { \mathrm { c o n v } } ^ { x }$ and $\pi _ { \mathrm { c o n v } } ^ { w } \mathrm { : }$

$$
\begin{array} { r l } & { \hat { x } = \pi _ { \mathrm { c o n v } } ^ { x } ( \mathbf { X } ) \mathrm { ~ a n d ~ } \hat { w } = \pi _ { \mathrm { c o n v } } ^ { w } ( \mathbf { W } ) \mathrm { ~ s u c h ~ t h a t ~ } } \\ & { \hat { x } [ i W C _ { i } + j C _ { i } + c ] = \mathbf { X } [ c , i , j ] } \\ & { \hat { w } [ 0 ] = \mathbf { W } [ C _ { o } - 1 , C _ { i } - 1 , H - 1 , W - 1 ] } \\ & { \hat { w } [ N - C _ { i } ( c ^ { \prime } H W + c W + j ) - c ] } \\ & { \qquad = - \mathbf { W } [ c ^ { \prime } , c , i , j ] } \end{array}\tag{otherwise}
$$

Figure 6 (b) illustrates this encoding scheme. For the input tensor $X$ , the first elements across all four channels $( \mathrm { i . e . , }$ $\{ 1 , 2 , 3 , 4 \} )$ are mapped to the first four coefficients of xˆ, followed by the second and the third elements of all input channels. For the weight tensor W, we adhere to the same channel-first encoding but arrange the elements in reverse order. Consequently, the polynomial multiplication $\hat { y } _ { 1 } = \hat { x } \cdot \hat { w } _ { 1 }$ yields valid convolution results at degrees that are multiples of 4 (specifically, coefficients of $x ^ { 0 }$ and $x ^ { 4 } )$ , while intermediate terms (e.g., coefficients of $x ^ { 1 } , x ^ { 2 } , \ldots )$ contain dummy values. To accommodate tensors exceeding the capacity of a single polynomial of degree $N ,$ we partition the tensor along the input channel dimension into smaller subtensors and apply this encoding process to each partition.

With this encoding, we now demonstrate how to utilize automorphism operations to eliminate dummy coefficients. Formally, given a polynomial $\begin{array} { r } { \hat { y } = \sum _ { i } c _ { i } X ^ { i } } \end{array}$ , the automorphism $\sigma ( \hat { y } , k )$ maps each term $c _ { i } X ^ { i } \stackrel { - } { \mathrm { t o } } c _ { i } X ^ { i k }$ (mod $X ^ { N } + 1 )$ . By selecting specific indices for $k ,$ we can preserve certain coefficients while inverting the signs of others.

For instance, applying the automorphism with $k = N + 1$ yields $\sum c _ { i } X ^ { i ( N + 1 ) }$ . Since $X ^ { N } \equiv - \bar { 1 }$ (mod $X ^ { N } + 1 )$ , this

![](images/da133dc7c12c578591b63701e41f25c09b0e9127642a91922432bbecaace3d60.jpg)  
Fig. 6. (a) A toy example of the convolution operation. (b) An example of channel encoding. (c) The automorphism σ(·, N + 1). (d) Eliminating dumm coefficients with automorphisms. (e) The key properties enabling the reformulation. (f) The key insights that guide the reformulation. (g) An example where all intermediate results $( \mathrm { e . g . } , \hat { y } _ { i } , \hat { y } _ { i } ^ { \prime } , \hat { y } _ { i } ^ { \prime \prime } )$ ) are replaced with xˆ and wˆ, and unifying automorphisms under the base $\sigma ( \cdot , N / 2 + 1 )$ . (h) The proposed computation optimized with the BSGS technique.

operation results in:

$$
\begin{array} { c l c r }  { \sigma ( \hat { y } , N + 1 ) = \displaystyle \sum _ { c _ { i } X ^ { i } ( X ^ { N } ) ^ { i } = \displaystyle \sum _ { c _ { i } X ^ { i } ( - 1 ) ^ { i } } \nonumber ^ { i } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { ~ } = c _ { 0 } - c _ { 1 } X + c _ { 2 } X ^ { 2 } - c _ { 3 } X ^ { 3 } + \ldots \mathrm { ~ } \mathrm { ~ m o d ~ } X ^ { N } + 1 } } \end{array}
$$

As illustrated in Figure $6 ~ ( \mathrm { c } ) .$ , this operation preserves the coefficients of even-degree terms $( \mathrm { e . g . , } x ^ { 0 } , x ^ { 4 } )$ while negating the coefficients of odd-degree terms $( \mathbf { e . g . } , x ^ { 1 } , x ^ { 3 } )$ . Therefore, computing the sum $\hat { y } + \sigma ( \hat { y } , N + 1 )$ cancels the odd-degree terms and doubles the even-degree terms:

$$
\hat { y } + \sigma ( \hat { y } , N + 1 ) = 2 c _ { 0 } + 0 X + 2 c _ { 2 } X ^ { 2 } + 0 X ^ { 3 } + \ldots \quad \mathrm { m o d } X ^ { N } + 1
$$

Subsequently, we apply the automorphism $k = N / 2 + 1$ . This operation targets terms with degrees that are odd multiples of two $( { \mathrm { i . e . , ~ } } \{ x ^ { 2 } , x ^ { 6 } , \ldots \} )$ :

$$
\sigma ( \hat { y } , N / 2 + 1 ) = 2 c _ { 0 } - 2 c _ { 2 } X ^ { 2 } + 2 c _ { 4 } X ^ { 4 } - . . . \mod X ^ { N } + 1
$$

By iteratively applying this logic using the recurrence:

$$
\hat { y } \gets \hat { y } + \sigma ( \hat { y } , N / 2 ^ { j } + 1 ) \quad \mathrm { f o r } \ j = 0 , 1 , \ldots , r - 1
$$

we effectively isolate the coefficients at positions that are multiples of $2 ^ { r }$ , and eliminate all remaining dummy coefficients.

Therefore, as shown in Figure 6 (d), starting with the polynomial $\hat { y } _ { 1 }$ , we first compute $\hat { y } _ { 1 } ^ { ' } = \hat { y } _ { 1 } + \sigma ( \hat { y } _ { 1 } , N + 1 )$ This operation effectively eliminates dummy coefficients corresponding to odd degrees $( \mathbf { e . g . , \ } x ^ { 1 } , x ^ { 3 } , \dots )$ . Subsequently, we compute the next stage of the accumulation using $\hat { y } _ { 1 } ^ { \prime \prime } \stackrel { } { = }$ $\hat { y } _ { 1 } ^ { ' } + \sigma ( \overset { \cdot } { \hat { y } _ { 1 } ^ { ' } } , N / 2 + 1 )$ , which removes coefficients at degrees such as $x ^ { 2 }$ and $x ^ { 6 }$ . Ultimately, only the valid coefficients at degrees $x ^ { 0 }$ and $x ^ { 4 }$ are preserved, while coefficients at all other degrees are zeroed out.

The outputs are then recombined by first multiplying a plaintext polynomial $x ^ { i }$ to rotate and align the coefficients, and then sum together (Figure 6 (d)). The preceding process shows that an automorphism’s ability to eliminate dummy coefficients relies on a specific ciphertext encoding. Our proposed encoding meets this requirement, unlike Cheetah’s output where useful coefficients are clustered together (Figure 3 (c)).

Table I provides a detailed comparison with other protocols, showing that our protocol produces fewer output ciphertexts but requires a large number of automorphisms on ciphertexts, potentially increasing computation time.

## C. Reducing Automorphisms: a BSGS Approach

We now present a more efficient method for reducing automorphisms. This approach achieves a greater than 128× reduction in automorphisms.

Our method leverages two properties illustrated in Figure 6 (e): 1) automorphisms on the result $\begin{array} { r l r } { \hat { y } } & { { } = } & { \hat { x } w } \end{array}$ can be applied beforehand on the inputs xˆ and wˆ, and then perform the multiplication, which is $\sigma ( \hat { y } ) = \sigma ( \hat { x } ) \sigma ( \hat { w } )$ . 2) applying $\sigma ( \cdot , N + 1 )$ to any xˆ is equivalent to applying $\sigma ( \overline { { \cdot , } } N / 2 + 1 )$ twice. This relationship, $\sigma ( \cdot , N + 1 ) = \sigma ^ { 2 } ( \cdot , N / 2 + 1 )$ , holds because:

$$
\sigma _ { N / 2 + 1 } ^ { 2 } ( a _ { i } x ^ { i } ) = a _ { i } x ^ { ( \frac { N } { 2 } + 1 ) ^ { 2 } i } = a _ { i } x ^ { \frac { N ^ { 2 } i } { 4 } + N i + i } = a _ { i } x ^ { ( N + 1 ) i }
$$

## TABLE I

COMPARISON WITH PRIOR WORKS. THE CONCRETE NUMBERS ARE BASEDON ONE LAYER OF THE IMAGENET-RESNET18 WITH  
$C _ { o } = 2 5 6 , C _ { i } = 2 5 6 , H = 1 4 , W = 1 4 , N = 8 1 9 2 . ^ { , \cdot } f = h \times w ^ { , }$ IS THENUMBER OF FILTER ELEMENTS.

<table><tr><td rowspan=1 colspan=1>Coeff.</td><td rowspan=1 colspan=1>Cheetah [18]</td><td rowspan=1 colspan=1>Ours w/o BSGS</td><td rowspan=1 colspan=1>Ours w/ BSGS</td></tr><tr><td rowspan=1 colspan=1>#CPMult</td><td rowspan=1 colspan=1>O(HWCiCo/N)2048</td><td rowspan=1 colspan=1>O(HWCiCo/N)2048</td><td rowspan=1 colspan=1>O(HWCiCo/N)2048</td></tr><tr><td rowspan=1 colspan=1>#Aut. (Rot.)</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O([HW/N]log(N/HW)Co)1280</td><td rowspan=1 colspan=1>O(√HWCiCo/N)10</td></tr><tr><td rowspan=1 colspan=1>#Input Ct.</td><td rowspan=1 colspan=1>O(HWCi/N)8</td><td rowspan=1 colspan=1>O(HWCi/N)8</td><td rowspan=1 colspan=1>O(HWCi/N)8</td></tr><tr><td rowspan=1 colspan=1>#Output Ct.</td><td rowspan=1 colspan=1>O([HW/N|Co)256</td><td rowspan=1 colspan=1>O(HWCo/N)8</td><td rowspan=1 colspan=1>O(HWCo/N)8</td></tr><tr><td rowspan=1 colspan=1>2PC moduli</td><td rowspan=1 colspan=1>PO2</td><td rowspan=1 colspan=1>PO2</td><td rowspan=1 colspan=1>PO2</td></tr><tr><td rowspan=1 colspan=1>SIMD</td><td rowspan=1 colspan=1>Orion [34]</td><td rowspan=1 colspan=1>Hyena [17]</td><td rowspan=1 colspan=1>Gazelle [12]</td></tr><tr><td rowspan=1 colspan=1>#CPMult</td><td rowspan=1 colspan=1>O(CoCiHWf/N)18432</td><td rowspan=1 colspan=1>O(HWCiCof/N)38016</td><td rowspan=1 colspan=1>O(CiCoHWf/N)36864</td></tr><tr><td rowspan=1 colspan=1>#Aut. (Rot.)</td><td rowspan=1 colspan=1> $O ( \sqrt { H W f / N } ( C _ { i } + C _ { o } ) )$ </td><td rowspan=1 colspan=1>O(f(CiHW/N+Cof/HW log(fN/HW)))920</td><td rowspan=1 colspan=1> $\boldsymbol { O } ( C _ { i } \cdot f )$ </td></tr><tr><td rowspan=1 colspan=1>#Input Ct.</td><td rowspan=1 colspan=1>O(CiHW/N)8</td><td rowspan=1 colspan=1>O(HWCi/N)16</td><td rowspan=1 colspan=1>O(CiHW/N)16</td></tr><tr><td rowspan=1 colspan=1>#Output Ct.</td><td rowspan=1 colspan=1>O(C0HW/N)8</td><td rowspan=1 colspan=1>O(HWCo/N)16</td><td rowspan=1 colspan=1>O(CoHW/N)16</td></tr><tr><td rowspan=1 colspan=1>2PC moduli</td><td rowspan=1 colspan=1>Prime</td><td rowspan=1 colspan=1>Prime</td><td rowspan=1 colspan=1>Prime</td></tr></table>

where $x ^ { \frac { N ^ { 2 } i } { 4 } } = ( x ^ { N } ) ^ { \frac { N i } { 4 } } \equiv ( - 1 ) ^ { \frac { N i } { 4 } } \equiv 1$ (mod $X ^ { N } + 1 )$ . In general, the set of automorphisms $\{ \sigma ( \cdot , N + 1 ) , \sigma ( \cdot , N / 2 +$ $1 ) , \ldots \}$ can all be expressed as powers of a single base automorphism as $\{ \sigma ^ { 2 ^ { m } } , \bar { \sigma ^ { 2 ^ { m - 1 } } } , \dots , \bar { \sigma } \}$ , where $\sigma \equiv \sigma ( \cdot , N / 2 ^ { m } +$ 1). In the context of our running example, the required set of automorphisms is $\{ \sigma ( \cdot , N + 1 ) , \sigma ( \cdot , N / 2 + 1 ) \}$ . By defining the base automorphism as $\sigma = \sigma ( \cdot , N / 2 + 1 )$ , this set can be represented as $\{ \bar { \sigma ^ { 2 } } , \sigma \}$

Based on these properties, we present two key insights, illustrated in Figure 6 (f): 1) We can apply the automorphism to the input ciphertext xˆ and plaintext wˆ beforehand rather than on the result $\hat { y } _ { i }$ as there are less xˆ. For instance, in our example, there are only one input ciphertext xˆ, but four intermediate output ciphertexts $\left\{ \hat { y } _ { 1 } , \hat { y } _ { 2 } , \hat { y } _ { 3 } , \hat { y } _ { 4 } \right\}$ . 2) BSGS algorithm can be applied if the automorphisms form a geometric progression.

Therefore, we reformulate the computation as shown in Figure 6 (g). First, we observe that the cleaned ciphertext $\hat { y } _ { i } ^ { \prime \prime }$ is obtained by computing $\hat { y } _ { i } ^ { \prime } = \hat { y } _ { i } + \sigma ( \hat { y } _ { i } , N + 1 )$ , followed by $\hat { y } _ { i } ^ { \prime \prime } = \hat { y } _ { i } ^ { \prime } + \sigma ( \hat { y } _ { i } ^ { \prime } , N / 2 + 1 )$ . By substituting the $\hat { y } _ { i } ^ { ' }$ with $\hat { y } _ { i }$ , this sequence is equivalent to:

$$
\hat { y } _ { i } ^ { \prime \prime } = \hat { y } _ { i } + \sigma ( \hat { y } _ { i } , N + 1 ) + \sigma ( \hat { y } _ { i } , N / 2 + 1 ) + \sigma ( \sigma ( \hat { y } _ { i } , N + 1 ) , N / 2 + 1 )
$$

Let $\sigma$ denote the automorphism $\sigma ( \cdot , N / 2 + 1 )$ . By substituting this notation, the equation simplifies to a summation over the powers of σ:

$$
\hat { y } _ { i } ^ { \prime \prime } = \hat { y } _ { i } + \sigma ( \hat { y } _ { i } ) + \sigma ^ { 2 } ( \hat { y } _ { i } ) + \sigma ^ { 3 } ( \hat { y } _ { i } ) = \sum _ { j = 0 } ^ { 3 } \sigma ^ { j } ( \hat { y } _ { i } )
$$

Replacing the intermediate components $\hat { y } _ { i } ~ = ~ \hat { x } ~ \cdot ~ \hat { w } _ { i }$ , the computation for a single ciphertext becomes:

$$
\hat { y } _ { i } ^ { \prime \prime } = \sum _ { j = 0 } ^ { 3 } \sigma ^ { j } ( \hat { x } ) \sigma ^ { j } ( \hat { w } _ { i } )
$$

We observe that the term $\sigma ^ { j } ( \hat { x } )$ remains the same across all $\hat { y } _ { i } ^ { \prime \prime }$ . Consequently, we can restructure the computation of the final output $\begin{array} { r } { \hat { y } = \sum _ { i } \hat { y } _ { i } ^ { \prime \prime } X ^ { i } } \end{array}$ by substituting $\hat { y } _ { i } ^ { \prime \prime }$ into the equation:

$$
\begin{array} { l } { { \hat { y } = \displaystyle \sum _ { i = 0 } ^ { 3 } \sum _ { j = 0 } ^ { 3 } \sigma ^ { j } ( \hat { x } ) \sigma ^ { j } ( \hat { w } _ { i } ) X ^ { i } } } \\ { { \qquad = \displaystyle \sum _ { j = 0 } ^ { 3 } \sigma ^ { j } ( \hat { x } ) \left( \sum _ { i = 0 } ^ { 3 } \sigma ^ { j } ( \hat { w } _ { i } ) X ^ { i } \right) = \sum _ { j } \sigma ^ { j } ( \hat { x } ) \hat { q } _ { j } } } \end{array}
$$

The term in parentheses consists entirely of constant weights; therefore, we can precompute it and define $\hat { q } _ { j } ~ =$ $\textstyle \sum _ { i = 0 } ^ { 3 ^ { - } } \sigma ^ { j } ( { \hat { w } } _ { i } ) X ^ { i }$ . This reformulated dataflow is illustrated in Figure 6 (g). Notably, this optimization reduces the number of required automorphisms in our example from eight to three.

The computation can be further optimized using the Baby-Step Giant-Step (BSGS) technique, as shown in Figure 6 (h). Instead of computing the full set $\{ \sigma ( \hat { x } ) , \sigma ^ { 2 } ( \hat { x } ) , \sigma ^ { 3 } ( \hat { x } ) \}$ , we compute only $\sigma ( \hat { x } )$ . We then derive the final result by grouping terms and applying $\sigma ^ { 2 }$ as the ”giant step”:

$$
\hat { y } = \hat { x } \hat { q } _ { 1 } + \sigma ( \hat { x } ) \hat { q } _ { 2 } + \sigma ^ { 2 } \left( \hat { x } \sigma ^ { - 2 } ( \hat { q } _ { 3 } ) + \sigma ( \hat { x } ) \sigma ^ { - 2 } ( \hat { q } _ { 4 } ) \right)
$$

This method reduces the required automorphisms from three to two. Generally, this process can be written as:

$$
\begin{array} { r l } & { \hat { \mathbf { y } } _ { k } = \displaystyle \sum _ { j = 0 } ^ { N _ { i } - 1 } \sum _ { i = 0 } ^ { 2 ^ { k } - 1 } \sigma ^ { i } ( \hat { x } _ { j } ) \hat { q } _ { i j k } } \\ & { \quad \quad = \displaystyle \sum _ { j = 0 } ^ { N _ { i } - 1 } \sum _ { i _ { 1 } = 0 } ^ { 2 ^ { k _ { 2 } } - 1 } \sigma ^ { i _ { 1 } \cdot 2 ^ { k _ { 1 } } } \left( \sum _ { i _ { 2 } = 0 } ^ { 2 ^ { k _ { 1 } } - 1 } \sigma ^ { i _ { 2 } } ( \hat { x } _ { j } ) \hat { q } _ { i j k } \right) } \end{array}\tag{1}
$$

where $N _ { i }$ is the number of input ciphertexts (1 in this case) and $2 ^ { k }$ is the number of channels contained in one ciphertexts (4 in this case). We provide a detailed comparison with other protocols regarding the number of HE operations in Table I. Our technique saves 128× rotations compared to an implementation without the BSGS optimization.

## IV. MEMORY AWARE OPTIMIZATIONS

## A. Memory Bottlenecks of our protocol

To identify the hardware performance bottlenecks of our proposed protocol, we profiled the data volume and memory access patterns of representative linear layers in ResNet-50, as shown in Figure 7. Our analysis reveals three key findings: 1) Evaluation keys are not the primary bottleneck. The memory footprint of evaluation keys is negligible for two reasons: the size of each key shrinks quadratically with its level L (e.g., only 384KB for $L \ = \ 1 )$ , and the BSGS method drastically reduces the total number of keys required. Consequently, the dominant memory overhead stems from plaintext and ciphertext data movement. 2) Plaintexts create a high-volume data bottleneck. As shown in Figure 7(b), the large volume of plaintext data creates significant memory traffic. This is a direct consequence of our protocol’s sparse encoding scheme (visualized in Figure 9(a)), which requires a large number of weight plaintext polynomials. 3) Ciphertexts create a memory thrashing bottleneck. The ”baby step” phase of the BSGS algorithm generates a large working set of temporary ciphertexts that exceeds the on-chip scratchpad capacity. Although these ciphertexts are reused intensively during the subsequent ”giant step” computations, their large collective size forces constant data swapping (thrashing) between the scratchpad and off-chip HBM. This results in the high volume of DRAM reads observed in Figure 7(b).

![](images/5e4ede69b06feec1ff5de3beb3edb861fda5e7f938052a3043f60848ae2462a9.jpg)  
Fig. 7. (a) Data size and arithmetic intensity of several Resnet50 layers with Cheetah. ”BS ciphertext” is the total ciphertext after baby step automorphisms. The baby step is chosen to minimize the total computation according to [72]. (b) Memory access of plaintext and ciphertexts.

As shown in Figure 7, the arithmetic intensity of our protocol is merely 0.6 Ops/Byte, classifying the workload as heavily memory-bound due to the large data movement required for both plaintexts and ciphertexts. To resolve these two challenges, we introduce two hardware optimizations. To tackle the high volume of plaintext data, we propose OptiComp, a lightweight compression system that exploits inherent data sparsity to reduce memory traffic. To resolve the ciphertext thrashing problem, we co-design OptiFlow, a reuse-centric dataflow that reorders computation to maximize on-chip data locality and reuse.

## B. OptiComp: Plaintext Compression System

We begin by illustrating why weight plaintexts in our application is sparse. As shown in Figure 9 (b), the plaintext encoding takes two steps. First, weight tensors are encoded into the plaintext with channel encoding, which is highly sparse (8 non-zero coefficients for N = 32 in this example). Then we compute $\begin{array} { r } { \hat { \bf q } _ { \mathrm { i } } = \sum _ { \mathbf { i } } \sigma ^ { \mathbf { i } } ( \hat { \mathbf { w } _ { \mathbf { j } } } ) \mathbf { x } ^ { \mathbf { j } } } \end{array}$ to get the plaintext we really use. Notably, the σ operation does not change the sparsity; it only permutes the coefficients, and only the summation might slightly reduce sparsity. We have also empirically verified this by profiling ResNet50, which shows high sparsity in most layers as in Figure 8.

Given this sparsity, two naive solutions could be considered as shown in Figure 9 (c). The first is to store plaintexts in the plaintext ring $R _ { t }$ in DRAM and perform the NTT onchip (⃝2 ). This approach is insufficient, as it only halves the memory traffic, which still remains a significant bottleneck. The second approach is to store the original weights and perform both encoding and NTT on-chip (⃝3 ). This is also infeasible because it requires performing automorphisms on a vast number of plaintexts; while cheaper than ciphertext automorphisms, the large volume makes this prohibitively expensive.

![](images/8debfe583ce06f75a6b9a6790c1e6b5b85331fdc11ee6906d89aed01a7623b3c.jpg)  
Fig. 8. Sparsity in each layer of ResNet50 (higher means sparser).

![](images/172574b9e73833cb1e98518828f0f0f4d6b61309ebd39d69262749814c0cec6e.jpg)  
Fig. 9. (a) The overall precomputation process. (b) Our application’s encoding process. A weight tensor is encoded into a sparse plaintext polynomial and then summed to derive $\begin{array} { r } { \hat { \bf q } _ { \mathrm { i } } = \sum _ { \mathbf { i } } \sigma ^ { \mathbf { i } } ( \hat { \mathbf { w } } _ { \mathbf { j } } ) \mathbf { x } ^ { \mathbf { j } } } \end{array}$ . (c) A comparison of three plaintext management strategies: Method ⃝1 precomputes both encoding and NTT, as used in current libraries. Method ⃝2 precomputes the encoding and performs NTT on-chip. Method ⃝3 performs both encoding and NTT on-chip. (d) Our proposed method: Leveraging the plaintext’s sparsity, we precompute the encoding and compress the result. At runtime, the data is decompressed before the on-chip NTT is performed.

Therefore, as shown in Figure 9 (d), we propose a third approach: directly storing the compressed plaintext in DRAM and decompressing it on-chip. This method avoids the high overhead of both large data transfers and on-chip plaintext automorphisms, offering a more effective solution to the memory bottleneck.

Figure 10 illustrates our proposed plaintext compression system. We profile the plaintexts of our protocols and find that all plaintexts exhibit a sparse pattern in Figure 10 (a), which consists of a few dense blocks of non-zero coefficients separated by long runs of zeros. We believe this pattern comes from channel encoding, which encodes weight elements in several clusters, and the automorphisms in our protocol also preserve this structure, as most elements are kept or negated in place. The compression strategy is to discard these zero-runs and store only the non-zero data blocks. To enable reconstruction, each non-zero block is paired with metadata specifying its length and original starting position in the polynomial. Thus, a complete sparse polynomial is compressed into a compact representation consisting of a series of these metadata-data pairs.

![](images/e53117f319fa4abc1b07e40714832f2121820cd11973dd03d7f3af5bcb9827d1.jpg)

![](images/b5633a70f687cacb8f9402f2954a22ef630c41288f9c6493ee0c7fd6fc28ae75.jpg)  
Fig. 10. (a) The compressed plaintext data structure. (b) The metadata structure. (c) Modification to the compiler. The compiler needs to first take in the plaintext and the user program, then output the compression info and the compressed plaintext. Then, the compiler will generate the output using the compression info. (d) The hardware architecture of the decompression unit.

Storing these variable-length non-zero sections sequentially in memory is problematic. Since existing memory systems manage data in fixed-size units of a full RNS polynomial, a direct layout would create memory alignment issues and complicate address calculation. To resolve this, we introduce a fixed-size container called a ”segment,” as shown in Figure 10 (a). A segment is constructed by packing multiple non-zero sections contiguously and zero-padding the remainder to match the full polynomial length, N. As detailed in Figure 10 (b), the metadata for each packed section has three components: its original starting position, its length, and an ”end bit” (‘e‘) to flag if it is the final section of the original sparse polynomial.

Our compiler is modified to support this new data format, as outlined in Figure 10 (c). Programmers annotate plaintexts for compression by using a new ‘ComPlain()‘ data type. The compiler then processes these annotations in a new, initial pass (⃝1 ). In this pass, it packs the compressed plaintexts into fixedsize segments, grouping them in the order they appear in the program. For each plaintext, the compiler generates ”compression information”—its location, specified by a segment index, a byte offset, and a length. This information is used by later compiler passes to generate the final machine code. To handle decompression at runtime, we introduce a new instruction, Decomp Len, Off. The compiler inserts this instruction before any operation (like NTT) that needs the full plaintext. The ‘Len‘ and ‘Off‘ operands tell the on-chip decompression unit exactly which portion of a segment to read and decompress.

![](images/982bfada5ff0d53e0d67e571c1616b2bcb6193147935c7971a91d06b2c5dc239.jpg)

Fig. 11. (a) The total computation and temporary ciphertext with varying baby steps. The orange dotted line represents the total on-chip SRAM size (25MB). The x-axis represents the baby step and giant step split. (b) The temporary ciphertexts’ memory access overhead. Those who suffer from the thrashing problem incur huge memory access.  
![](images/f3432898cbc3881bcf412205c8c51eb6fe4ae5c4b4c3613b86ec4d3849f7dcb0.jpg)  
Fig. 12. (a) An example of OptiPrime’s protocol. (b) The temporary ciphertext thrashing problem is due to the limited scratchpad size. (c) Output stationary dataflow to minimize the ciphertext data movement.

To support our compression scheme, we integrate a dedicated decompression unit into each Compute Cluster as shown in Figure 10 (d). The overall workflow is as follows: prior to a computation like NTT, a compressed plaintext segment is fetched into an on-chip scratchpad. The decompression unit then reads this segment and reconstructs the original, fulllength sparse polynomial. The unit itself consists of a simple controller and a set of multiplexers. During decompression, the controller parses the metadata for each non-zero section to determine its length and its target offset in the final polynomial. It then writes the non-zero data to the correct locations in the register bank while filling the rest with zeros.

## C. OptiFlow: A Reuse-Centric Dataflow

The computation of our protocol is formulated using BSGS in section III. A standard implementation, shown in Figure 12(a), proceeds in four phases: ⃝1 Baby step automorphisms are applied to all inputs, ⃝2 inner products are computed, ⃝3 giant step automorphisms are applied to the intermediate results, and ⃝4 the partial results are aggregated.

For all computations, we use the hoisting optimization for key switching, as the keys are small enough to reside entirely in the on-chip scratchpad (Figure 7).

The optimal split of k into $k _ { 1 }$ and $k _ { 2 }$ presents a critical trade-off. A standard split $( k _ { 1 } \approx k _ { 2 } )$ is suboptimal for our application because the hoisted baby-step computations are cheaper to compute and are heavily reused across all $N _ { o }$ outputs. Our analysis (Figure 11(a)) reveals that the most computationally efficient configuration favors a large number of baby steps $( k _ { 1 } > k _ { 2 } )$ .

Unfortunately, this computationally optimal strategy is memory-inefficient. A large $k _ { 1 }$ generates a massive working set of $N _ { i } \ \cdot \ 2 ^ { k _ { 1 } }$ temporary ciphertexts (e.g., $\{ \sigma ( \hat { x } ) , \ldots , \sigma ^ { 2 ^ { k _ { 1 } } - 1 } ( \hat { x } ) \} )$ which exceeds the on-chip scratchpad capacities. Consequently, these temporary results must be stored in off-chip HBM. Despite their high reuse potential, they must be repeatedly fetched, leading to severe memory thrashing between the scratchpad and HBM. The resulting memory traffic, shown in Figure 11(b), is immense and completely negates the computational benefits. Prior work such as SHARP [43] avoids this by using a small $k _ { 1 }$ , but this choice incurs a significant computational penalty (e.g., 4× more computation than optimal).

To resolve this dilemma, we propose OptiFlow, a reusecentric dataflow that achieves both the computation and memory access efficiency. The core idea is to process one input ciphertext at a time, calculating its contribution to all output ciphertexts before moving to the next input. This avoids materializing the entire enormous set of temporary ciphertexts at once.

We illustrate this output-stationary dataflow using the example in Figure 12(c). In step ⃝1 , A single input ciphertext, $\hat { \mathbf { x } } _ { \mathbf { j } }$ , is loaded from HBM into the on-chip scratchpad. The accelerator uses it to update the partial sums for all $N _ { o } \cdot 2 ^ { k _ { 2 } }$ outputs that depend on it. These partial sums remain resident on-chip. ⃝2 The next baby step automorphism $( \mathbf { e . g . } , \sigma ^ { 1 } ( \hat { x } _ { j } ) )$ is computed from the previous one. Its results are immediately used to again accumulate into the on-chip partial sums. This is repeated for all $2 ^ { k _ { 1 } }$ baby steps for the input $\hat { \mathbf { x } } _ { \mathbf { j } } . \textcircled { 3 }$ After all baby steps for $\hat { \mathbf { x } } _ { \mathbf { j } }$ are completed, its data is discarded. The process repeats by loading the next input ciphertext, ${ \hat { \mathbf { x } } } _ { \mathbf { j } + \mathbf { l } } . ~ \textcircled { 4 }$ Once all $N _ { i }$ input ciphertexts have been processed, the complete partial sums in the scratchpad undergo the final giant step automorphisms and aggregation to produce the $N _ { o }$ final outputs.

This dataflow relies on the scratchpad being large enough to hold all partial sums $( N _ { o } \cdot 2 ^ { k _ { 2 } }$ ciphertexts), which is feasible as the computationally optimal split uses a small $k _ { 2 }$ . If the partial sums still exceed capacity, we simply tile the computation along the output dimension $N _ { o }$ . Crucially, OptiFlow recomputes the baby-step ciphertexts on-the-fly rather than storing them. With hoisting, the cost of an automorphism recomputation is far less than the latency of a single HBM access, making this a highly effective trade-off.

## V. EVALUATION

In this section, we evaluate OptiPrime, quantify our improvements over prior work, and demonstrate the effectiveness of our approach. We begin by presenting our evaluations across different CNN networks on ImageNet [73] in Figure 13. We then provide a detailed analysis of the benefits of our protocol and our memory-efficient optimization.

## A. Methodology

a) Implementation: We implement OptiPrime on the LattiSense platform [64]. Using the provided toolchain, OptiPrime’s HE computation graph is defined in Python and compiled into accelerator instructions. The host program is implemented in C++ to perform encoding/decoding, encryption/decryption, network transmission of ciphertexts, and invocation of the accelerator. To support our memory-efficient contributions, we extend the compiler to generate instructions for the decompression unit and to optimize the Baby-Step Giant-Step (BSGS) dataflow. The decompression unit is implemented in Verilog. For the nonlinear layers, we integrate the implementations of CrypTFlow2 and Cheetah [9], [18] with the EMP toolkit [74] and the EzPC framework [75] in C++. We use the communication-efficient Vector-OLE-based Oblivious Transfer (VOLE-OT) protocol [76], [77] across all experiments for a fair comparison.

b) Experimental Setup: We evaluate OptiPrime on an Intel(R) Xeon(R) Gold 6226R CPU @ 2.90 GHz with 256 GB of RAM and use 16 threads. The FPGA accelerator is deployed on the AMD Alveo U55C Accelerator Card and interacts with the host CPU through PCIe. The network condition is simulated through Linux Traffic Control. The bandwidth is set to 384MBps for LAN and 44MBps for WAN, with a round-trip latency of 0.3ms for LAN and 40ms for WAN.

c) Baseline: The baseline for our protocol is two HE-MPC frameworks, a coefficient encoding protocol Cheetah [18] and SIMD encoding Hyena [17] for convolution operations. We further adapt the SIMD-based FHE protocol Orion [34] to the hybrid HE-MPC setting as an additional encoding baseline. Since the main contribution of our work is not the accelerator itself, we do not compare it with other accelerators and instead treat it as a general platform.

d) Parameters: We target 128-bit security for both HE and OT protocols. For HE, the multiplication depth is 1, we use the ciphertext modulus $q \approx 2 ^ { 6 4 }$ , special prime $p \approx 2 ^ { 3 2 }$ the plaintext modulus $t \ : = \ : 2 ^ { 2 1 }$ , and the polynomial degree $N = 8 1 9 2$ . For VOLE-OT, we select the security parameter $\lambda = 1 2 8$ . The same set of cryptographic parameters is used across all experiments, with $t \approx 2 ^ { 2 1 }$ chosen for the Hyena baseline, as it employs SIMD encoding.

e) Benchmarks: For benchmarks, we use the ImageNet dataset [73], whose image size is 20–50× larger than the previously used CIFAR-10/100 [78] and TinyImageNet [79] datasets. We evaluate end-to-end latency on ResNet [80], VGG [81], and MobileNetV2 [82].

![](images/9236594f47625d40707a93c86c9f95ff1d1ea3fc858085f1e06950fcd93e796c.jpg)

![](images/a8f12609f7dd737b9e9e4032d178b6631e63b11b1bd7216f4ddd06c351326b94.jpg)

Fig. 13. End-to-end inference evaluation under LAN and WAN network conditions. “+CPU” and “+Acc.” denote the CPU-only and accelerator-augmented platforms.  
![](images/2ba6e319f7f5608266a407ff9e080be8de8ae54632766190cd2c19aa7862a11b.jpg)

![](images/be236d491b0bfac145dfeef7a9970960066da07c96272bec2a6c04b00165081f.jpg)  
Fig. 14. Micro benchmark evaluation of linear layers.

## B. End-to-End Performance Evaluation

Figure 13 presents our end-to-end latency evaluation on both CPU and accelerator platforms with four encodings, with all results normalized to the Cheetah+CPU baseline. Our method consistently outperforms the baselines across all configurations. Compared to CPU-only implementations, OptiPrime reduces latency to just 0.02×–0.3× in a LAN setting and 0.03×–0.56× in a WAN setting. When compared against scenarios with accelerators, OptiPrime achieved normalized latencies of 0.4×–0.78× (LAN) and 0.29×–0.65× (WAN). These substantial performance gains come from two factors: our protocol’s communication efficiency and our accelerator’s codesigned compression unit and dataflow, which reduces computation time. The difference for Hyena from its original publication comes from our larger polynomial degree (N = 8192 vs. N = 1024), which adds the automorphisms its SIMD encoding needs to aggregate partial sums. Compared to Orion, OptiPrime is 1.4–1.6× faster under WAN and 1.3– 1.5× under LAN, since its coefficient encoding incurs far fewer HE operations and less communication than Orion’s SIMD encoding.

Figure 14 provides a detailed latency breakdown for representative ResNet50 layers to dissect the performance of OptiPrime, with all results normalized to the Cheetah+CPU baseline. Compared to Cheetah, our protocol design delivers a significant reduction in network communication latency in both network conditions with a small amount of extra computation. Furthermore, our method reduces CPU-side overhead for en/decryption and en/decoding (labeled ”Others”) by generating fewer output ciphertexts for the client to process. For Hyena, while its network latency is also small compared to Cheetah, its overall performance is bottlenecked by a computational cost that is over 10× higher, highlighting the effectiveness of our communication and computation efficient protocol. Orion also keeps the network latency low through its compact output packing, but its per-layer HE compute stays higher than OptiPrime’s, so its latency remains above ours across all the representative layers.

## C. Evaluation of OptiPrime’s Protocol

Figure 15 (a-c) presents an evaluation of our protocol’s communication efficiency against other baselines. Compared to Cheetah, our protocol reduces network communication to 0.11×-0.59× and the number of output ciphertexts to 0.07×- 0.46×. Compared to Hyena, the number of output ciphertexts is comparable, as the SIMD encoding with masking can also eliminate dummy slots [14]. Finally, the number of input ciphertexts remains similar across all three schemes because the input data is typically densely encoded. The difference mainly comes from their padding strategies. For example, Hyena pads each kernel to a power of two to accumulate the partial results of one convolution kernel. This difference has little effect on total communication because the output ciphertexts contribute the dominant overhead.

We also evaluate computational efficiency by comparing the required number of ciphertext-plaintext multiplications (CPMult) and automorphisms. As detailed in Figure 15 (d) and (e), our protocol requires only 0.05×-0.07× the CPMults and 0.04×-0.07× the automorphisms of the Hyena protocol. This twofold advantage is attributed to the Baby-Step Giant-Step (BSGS) technique and the use of a coefficient-based encoding scheme, which is inherently suited for convolution. Notably, though the BSGS changes the content of each weight plaintext, it does not increase communication overhead, the number of CPMults, or the number of weight plaintexts, and is purely a free lunch. Compared to an implementation of

Cheetah Hyena OptiPrime w/o BSGS OptiPrime w/BSGS  
![](images/de8a1cb38a5025ad8faf32c9ec49a77b94ba55de24d7c2ef54b66507c80c4541.jpg)

![](images/4b04bdf006ea65543a727c275c36f6652cb77eb3bbbf1ca5aa604462cd7cebac.jpg)

![](images/8af97306a0c229a11b12bd592372d8f6ff77dd8c362571e101bcc981de25d2b5.jpg)

![](images/b3e9c7b897c3709092ae4dc8aedc8435d66325bebe1608ca8e11acda6c12b5f7.jpg)

![](images/a6636ae3135402ad8c208ae2882a9054fd157a1259b7855072a3e9b72841e08f.jpg)

![](images/d10a712a338dccc0dfd411fbdddbd701fab5c26dfcc9430b151df120b9577fec.jpg)

Fig. 15. Operations Count and Communication Comparison against Baselines. ”Norm.” represents normalized. ”CPMult” represents ciphertext-plaintext multiplication.  
![](images/fcbc08025dd7f6b661cac11370fd24e1617077d875326a1dcdf7cd490ecd0f77.jpg)

![](images/0eac567b8f7d7d0fd35751793b254f957d47977056ba6e0e89bfd3c248d6f018.jpg)  
Fig. 16. Ablation from the Cheetah+CPU baseline, adding one optimization at a time, with latency normalized to the baseline. “+Acc.” adds the HE accelerator; “+Pro.” adds our channel-encoding protocol; “+BSGS” adds the BSGS automorphism reduction; “+Com.” adds plaintext compression; “+DF.” adds the ciphertext dataflow.

OptiPrime without BSGS, our protocol requires just 0.13×- 0.19× the automorphisms, further highlighting the efficacy of our approach. Although Cheetah requires no automorphisms, it generates a large volume of output ciphertexts, a tradeoff that is unfavorable in scenarios with an accelerator where communication is a primary bottleneck.

Figure 15 (f) also compares the number of weight plaintexts required by each protocol. Our approach uses a larger number of plaintexts due to a sparse encoding of weights, which increases memory traffic and are optimized through hardware codesign. In contrast, while the Hyena protocol reduces the plaintext count, it does so at the cost of more computationally intensive operations (e.g., automorphisms) and remains inefficient over wireless networks.

Consequently, although our encoding incurs a minor increase in automorphisms, it achieves a substantial reduction in communication overhead. This trade-off proves highly favorable, as with a hardware accelerator, this small computational cost has a negligible impact on the overall end-to-end latency but contributes to larger communication reduction, which greatly impacts the overall latency and is hard to accelerate.

## D. Evaluation of Memory Optimization

The decompression unit was implemented in RTL and integrated with the surrounding logic in collaboration with the LattiSense team. As detailed in Table II, the implementation utilizes 0.8k LUTs and 0.8K registers, constituting a mere

TABLE II  
RESOURCE UTILIZATION FOR THE DECOMPRESSION UNIT.
<table><tr><td rowspan=1 colspan=1>LUT</td><td rowspan=1 colspan=2>Reg  BRAM</td><td rowspan=1 colspan=1>DSP</td></tr><tr><td rowspan=1 colspan=1>0.8K</td><td rowspan=1 colspan=1>0.8K</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr></table>

0.7% of the total available on-chip resources the AMD U55C Accelerator Card (1304k LUTs and 2607k registers).

Figure 17 illustrates the reduction in memory access achieved by OptiComp and OptiFlow across several CNN architectures. The baseline configuration stores all plaintexts in the evaluation domain and employs the computation-efficient ’baby step’ selection of the BSGS algorithm. First, applying our plaintext compression technique (denoted “+Com.”) reduces memory access to 0.7×-0.85× that of the baseline, underscoring the effectiveness of plaintext compression. This also shows that for our encoding, the sparsity of the weight plaintext is prevalent, as all networks benefit from the exploitation of sparsity. Second, by incorporating OptiFlow(“+DF.”), we further reduce the memory access overhead to just 0.03×- 0.21× of the baseline. This large reduction is attributed to the high data reusability enabled by our dataflow, which effectively mitigates memory thrashing.

## E. Ablation study

Figure 16 reports an ablation that starts from the Cheetah+CPU baseline and adds one optimization at a time, with the latency normalized to the baseline. “+Acc.” offloads the HE computation to the accelerator and shrinks the compute part, bringing a 1.7–2.4× (LAN) / 1.2–1.7× (WAN) end-to-end speedup, after which the latency is dominated by the network communication. “+Pro.” applies our channel encoding, which reduces the network communication; this is the single largest step on WAN (up to 3.3× speedup); on LAN the gain is 1.0– 1.6×, as the extra compute partly offsets the communication saving. “+BSGS” then eliminates that extra compute by reducing the automorphisms of the channel encoding, bringing a $\mathbf { 1 . 0 6 { - 1 . 2 8 } \times ( L A N ) } / \mathbf { 1 . 0 4 { - 1 . 1 8 } \times ( W A N ) }$ speedup, largest on compute-heavy networks such as ResNet-50. Finally, “+Com.” reduces the plaintext memory access for a further end-to-end speedup of 1.06–1.17× (LAN) / 1.02–1.10× (WAN), and “+DF.” reduces the ciphertext memory access for a 1.05– 1.18× (LAN) / 1.01–1.11× (WAN) latency reduction.

![](images/cd70c923cb869470ac8333a458475123aa7c18f815a9078643495245b7b9e733.jpg)

Fig. 17. Memory access reduction through the plaintext compression technique and ciphertext dataflow. ”Com.” represents the plaintext compression technique, and ”DF.” represents the ciphertext dataflow technique.  
![](images/3783d6ad500951eae3766513e720c043ff8f0b6c38275083b9829d19ed7c345c.jpg)  
Fig. 18. Normalized end-to-end latency evaluation when integrated with ARK [41].

## F. Generality of OptiPrime

Generality to other accelerators. We evaluate OptiPrime on ARK [41] to demonstrate the generality of our work (Figure 18). Because neither ARK hardware nor its simulator and compiler are publicly available, we construct a cycle-accurate simulator from ARK’s published architectural parameters. We augment the modeled accelerator with our decompression unit and extend our compiler to emit ARKcompatible instructions, including the Decomp instruction, and to schedule the dataflow under ARK’s 512 MB scratchpad. Compared to the Cheetah+ARK baseline, applying our protocol (OptiEncode+ARK) reduces latency to 0.48×– 0.51× for ResNets and 0.91×–0.92× for VGGs under LAN, and $0 . 2 4 \times - 0 . 2 8 \times ~ / ~ 0 . 7 1 \times - 0 . 7 4 \times$ under WAN. With all optimizations (OptiPrime+ARK), latency further decreases to 0.35×–0.51× (LAN) and $0 . 2 1 \times - 0 . 2 8 \times$ (WAN) for ResNets, and $0 . 7 8 \times - 0 . 8 1 \times / \ 0 . 6 5 \times - 0 . 6 9 \times$ for VGGs.

![](images/f6dfb66672c9ad9e24be68bb582d285dea9ab2295a839113c013feec3b083504.jpg)  
Fig. 19. Whole-network linear-layer latency on BERT-base for Iron, BumbleBee (Bum.), and OptiEncode (Opti.).

![](images/549a62b35be687d66aa0c6bf6e1cedce55c54305c9901e4ea365f7177e9e35bf.jpg)  
Fig. 20. Network sensitivity for ResNet-18 (left) and ResNet-50 (right): bandwidth at 1 ms RTT (top), RTT at LAN bandwidth (middle), and RTT at WAN bandwidth (bottom). Bars show NetIO/non-NetIO latency for Cheetah and OptiPrime; lines show speedup.

Generality to Transformer models. We evaluate OptiPrime on BERT-base against the prior-art protocols Iron [7] and BumbleBee [15] to demonstrate its applicability to Transformer models. We implement each matrix multiplication as a 1 × 1 convolution. Under our LAN/WAN settings (Figure 19), OptiPrime is more efficient in both communication and computation, reducing the wholenetwork linear-layer latency to $0 . 6 7 \times / 0 . 2 3 \times$ (LAN/WAN) that of Iron and $0 . 5 7 { \times } / 0 . 6 9 { \times }$ that of BumbleBee.

## G. Inference Accuracy

We measure top-1 ImageNet accuracy under W8A8 quantization on pre-trained Torchvision models [83]. Since OptiPrime alters only the intermediate computation and keeps the result exact, its accuracy matches prior protocols, with only negligible quantization loss versus FP32 (Table III). The accumulator bit-width peaks at 19.3 bits, well below our plaintext modulus $t = 2 ^ { 2 1 }$ , confirming the parameter set is valid.

![](images/cc96bcf96773f1eb3a84daa9ccde89b2e1342aab4c9592b5c32deaa06de8ce50.jpg)

![](images/31288d806bf4c11a390d193b1421234dd8dbc74827de3ea85d84caba8c9f1dcb.jpg)

![](images/5028f7ba8c1318af4befa306896cc1e347f80a6873a3cf1e2833178edd6e4570.jpg)

Fig. 21. Performance of OptiPrime under nonideal (a) input-channel count, (b) spatial map size $H \times { \bar { W } }$ , and (c) filter size k, with the other dimensions fixed.  
![](images/a45a588e08737cf928f27cc9796b8a30694724fa1a2f78c72add676bf9a19950.jpg)  
Fig. 22. HE-parameter sensitivity on ResNet-18 (a–b) and ResNet-50 (c–d) under different polynomial degrees N and ciphertext moduli q.

## H. Sensitivity Study

Network. We evaluate ResNet-18 and ResNet-50 over bandwidths from 10 MB/s to 1 GB/s at a fixed 1 ms RTT, and over RTTs from 1 ms to 300 ms at both LAN and WAN bandwidths (Figure 20). Across this bandwidth range, OptiPrime consistently outperforms Cheetah, achieving 1.71–6.12× speedup on ResNet-18 and 2.04–6.69× on ResNet-50. Even at 1 GB/s, OptiPrime retains $1 . 7 1 \times / 2 . 0 4 \times$ speedup, as its acceleratorside optimizations continue to reduce the non-network latency. HE parameters On ResNet-18, we sweep the polynomial degree N (at a fixed log q =64) and the ciphertext modulus log q (at a fixed $N = 8 1 9 2 )$ under LAN and WAN in Figure 22. The speedup over Cheetah holds at every N and even grows with it, on WAN from 2.4× at $N { = } 4 0 9 6$ to 12.9× at $N = 3 2 7 6 8 .$ , since our communication is invariant to N while Cheetah’s grows. ResNet-50 exhibits the same trend: its WAN speedup rises from 2.7× to $1 3 . 6 \times$ over the same N sweep. The speedup also holds as log q grows from 64 to 128 bits, so the effectiveness of OptiPrime does not depend on the specific parameters $( N , q , t )$ . For ResNet-50 on WAN, it increases from $4 . 2 \times$ to 4.6× over this modulus range.

TABLE III  
W8A8 IMAGENET TOP-1 ACCURACY (%) AND MEASURED CONVOLUTION ACCUMULATOR BIT-WIDTH.
<table><tr><td>Model</td><td>FP32</td><td>Cheetah</td><td>Hyena</td><td>OptiPrime</td><td>Accum. (bits)</td></tr><tr><td>ResNet-18</td><td>69.76</td><td>69.51</td><td>69.51</td><td>69.51</td><td>18.5</td></tr><tr><td>ResNet-34</td><td>73.30</td><td>73.04</td><td>73.04</td><td>73.04</td><td>18.8</td></tr><tr><td>ResNet-50</td><td>76.14</td><td>75.93</td><td>75.93</td><td>75.93</td><td>19.1</td></tr><tr><td>VGG-16</td><td>71.58</td><td>71.44</td><td>71.44</td><td>71.44</td><td>19.1</td></tr><tr><td>VGG-19</td><td>72.39</td><td>72.33</td><td>72.33</td><td>72.33</td><td>19.3</td></tr><tr><td> $\mathbf { M o b i l e N e t V } 2$ </td><td>71.87</td><td>69.71</td><td>69.71</td><td>69.71</td><td>17.6</td></tr></table>

Layer shape. Starting from a representative convolution layer, we vary one shape dimension at a time and report the communication of the transmitted ciphertexts and its reduction over Cheetah in Figure 21, to confirm that OptiEncode performs well when handling shapes that do not fit N. Sweeping the channel count $C _ { i } = C _ { o }$ (at $H = W = 2 8 , \ k = 3 )$ , the reduction stays constant at 4.5×, as expected, since a non-ideal channel count only wastes space in the last ciphertext. Sweeping the spatial map $H \times W$ (at $C _ { i } = C _ { o } = 2 5 6 , k = 3 )$ , it falls from 32× for small maps to $\sim 1 . 5 \times$ for large maps; this does not mean our protocol degrades, but rather that a larger spatial map fills more useful slots in Cheetah’s ciphertexts and leaves Cheetah less to waste, so the ratio shrinks because Cheetah improves. Sweeping the filter size k (at $C _ { i } = C _ { o } = 2 5 6$ $H = W = 1 4 )$ , it drops from $1 6 . 7 \times ( k \leq 3 )$ to $4 . 6 \times ( k = 1 1 )$ for the same reason. In every case, OptiPrime requires far less communication than Cheetah, demonstrating its robustness to non-ideal layer shapes.

## VI. CONCLUSION

We propose OptiPrime, a protocol-hardware co-optimization framework for the hybrid HE-MPC setting. To address the wireless network communication bottleneck exposed by accelerators, we introduce a communication-efficient protocol based on channel encoding. We further deploy plaintext compression and a ciphertext-friendly dataflow to mitigate memory access overhead. Together, OptiPrime outperforms the Cheetah baseline by up to 5.7× on CPUs and 4.2× with an accelerator.

## ACKNOWLEDGMENT

This work was supported in part by NSFC under Grant 92464104, Grant 62495102, and Grant 62341407, in part by the National Key Research and Development Program under Grant 2024YFB4505004, in part by Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, in part by 111 Project under Grant B18001.

[1] I. Kononenko, “Machine learning for medical diagnosis: history, state of the art and perspective,” Artificial Intelligence in medicine, vol. 23, no. 1, pp. 89–109, 2001.

[2] W. Zhao, R. Chellappa, P. J. Phillips, and A. Rosenfeld, “Face recognition: A literature survey,” ACM computing surveys (CSUR), vol. 35, no. 4, pp. 399–458, 2003.

[3] N. Kumar, R. Chauhan, and G. Dubey, “Applicability of financial system using deep learning techniques,” in Ambient Communications and Computer Systems: RACCCS 2019. Springer, 2020, pp. 135–146.

[4] F. Mireshghallah, M. Taram, P. Vepakomma, A. Singh, R. Raskar, and H. Esmaeilzadeh, “Privacy in deep learning: A survey,” arXiv preprint arXiv:2004.12254, 2020.

[5] X. Liu, L. Xie, Y. Wang, J. Zou, J. Xiong, Z. Ying, and A. V. Vasilakos, “Privacy and security issues in deep learning: A survey,” IEEE Access, vol. 9, pp. 4566–4593, 2020.

[6] B. Reagen, W.-S. Choi, Y. Ko, V. T. Lee, H.-H. S. Lee, G.-Y. Wei, and D. Brooks, “Cheetah: Optimizing and accelerating homomorphic encryption for private inference,” in 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2021, pp. 26–39.

[7] M. Hao, H. Li, H. Chen, P. Xing, G. Xu, and T. Zhang, “Iron: Private inference on transformers,” Advances in Neural Information Processing Systems, vol. 35, pp. 15 718–15 731, 2022.

[8] T. Xu, M. Li, R. Wang, and R. Huang, “Falcon: Accelerating homomorphically encrypted convolutions for efficient private mobile network inference,” arXiv preprint arXiv:2308.13189, 2023.

[9] D. Rathee, M. Rathee, N. Kumar, N. Chandran, D. Gupta, A. Rastogi, and R. Sharma, “Cryptflow2: Practical 2-party secure inference,” in Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, 2020, pp. 325–342.

[10] D. Rathee, M. Rathee, R. K. K. Goli, D. Gupta, R. Sharma, N. Chandran, and A. Rastogi, “Sirnn: A math library for secure rnn inference,” in 2021 IEEE Symposium on Security and Privacy (SP). IEEE, 2021, pp. 1003–1020.

[11] P. Mohassel and Y. Zhang, “Secureml: A system for scalable privacypreserving machine learning,” in 2017 IEEE symposium on security and privacy (SP). IEEE, 2017, pp. 19–38.

[12] C. Juvekar, V. Vaikuntanathan, and A. Chandrakasan, “Gazelle: A Low Latency Framework for Secure Neural Network Inference,” arXiv:1801.05507 [cs], 2018. [Online]. Available: http://arxiv.org/abs/ 1801.05507

[13] P. Mishra, R. Lehmkuhl, A. Srinivasan, W. Zheng, and R. A. Popa, “Delphi: A cryptographic inference service for neural networks,” in 29th USENIX Security Symposium (USENIX Security 20). USENIX Association, Aug. 2020, pp. 2505–2522. [Online]. Available: https: //www.usenix.org/conference/usenixsecurity20/presentation/mishra

[14] Q. Pang, J. Zhu, H. Mollering, W. Zheng, and T. Schneider, “BOLT:¨ Privacy-preserving, accurate and efficient inference for transformers,” Cryptology ePrint Archive, Paper 2023/1893, 2023. [Online]. Available: https://eprint.iacr.org/2023/1893

[15] W. jie Lu, Z. Huang, Z. Gu, J. Li, J. Liu, C. Hong, K. Ren, T. Wei, and W. Chen, “BumbleBee: Secure two-party inference framework for large transformers,” Cryptology ePrint Archive, Paper 2023/1678, 2023. [Online]. Available: https://eprint.iacr.org/2023/1678

[16] T. Xu, L. Wu, R. Wang, and M. Li, “Privcirnet: Efficient private inference via block circulant transformation,” arXiv preprint arXiv:2405.14569, 2024.

[17] S. Singh, S. Singh, S. Gudaparthi, X. Fan, and R. Balasubramonian, “Hyena: Balancing packing, reuse, and rotations for encrypted inference,” in 2024 IEEE Symposium on Security and Privacy (SP). Los Alamitos, CA, USA: IEEE Computer Society, may 2024, pp. 3091–3108. [Online]. Available: https://doi.ieeecomputersociety.org/10. 1109/SP54263.2024.00107

[18] Z. Huang, W.-j. Lu, C. Hong, and J. Ding, “Cheetah: Lean and fast secure Two-Party deep neural network inference,” in 31st USENIX Security Symposium (USENIX Security 22), 2022, pp. 809–826.

[19] J. Yu, W. Zeng, T. Xu, R. Chen, Y. Liang, R. Wang, R. Huang, and M. Li, FlexHE: A flexible Kernel Generation Framework for Homomorphic Encryption-Based Private Inference. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3676536.3676739

[20] Z. Li, K. Yang, J. Tan, W.-j. Lu, H. Wu, X. Wang, Y. Yu, D. Zhao, Y. Zheng, M. Guo, and J. Leng, “Nimbus: secure and efficient two-party inference for transformers,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, ser. NIPS ’24. Red Hook, NY, USA: Curran Associates Inc., 2025.

[21] H. Cho, J. Jeon, J. Heo, and J.-Y. Kim, “Apint: A full-stack framework for acceleration of privacy-preserving inference of transformers based on garbled circuits,” in Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design, ser. ICCAD ’24. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3676536.3676786

[22] S. Balla and F. Koushanfar, “Heliks: He linear algebra kernels for secure inference,” in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 2306–2320. [Online]. Available: https://doi.org/10.1145/3576915.3623136

[23] K. Garimella, Z. Ghodsi, N. K. Jha, S. Garg, and B. Reagen, “Characterizing and optimizing end-to-end systems for private inference,” in Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ser. ASPLOS 2023. New York, NY, USA: Association for Computing Machinery, 2023, p. 89–104. [Online]. Available: https://doi.org/10.1145/3582016.3582065

[24] T. Xu, W.-j. Lu, J. Yu, Y. Chen, C. Lin, R. Wang, and M. Li, “Breaking the layer barrier: remodeling private transformer inference with hybrid ckks and mpc,” in Proceedings of the 34th USENIX Conference on Security Symposium, ser. SEC ’25. USA: USENIX Association, 2025.

[25] T. Zhang, C. Lin, J. Yu, Y. Chen, S. Deng, and M. Li, “(invited) fenix: Flexible and efficient hybrid he/mpc acceleration with near-memory processing,” in 2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD), 2025, pp. 1–9.

[26] J. H. Ju, J. Park, J. Kim, M. Kang, D. Kim, J. H. Cheon, and J. H. Ahn, “Neujeans: Private neural network inference with joint optimization of convolution and bootstrapping,” 2024. [Online]. Available: https://arxiv.org/abs/2312.04356

[27] E. Lee, J.-W. Lee, J. Lee, Y.-S. Kim, Y. Kim, J.-S. No, and W. Choi, “Low-complexity deep convolutional neural networks on fully homomorphic encryption using multiplexed parallel convolutions,” in International Conference on Machine Learning. PMLR, 2022, pp. 12 403– 12 422.

[28] A. Stoian, J. Frery, R. Bredehoft, L. Montero, C. Kherfallah, and B. Chevallier-Mames, “Deep neural networks for encrypted inference with tfhe,” in International Symposium on Cyber Security, Cryptology, and Machine Learning. Springer, 2023, pp. 493–500.

[29] J. Park, M. J. Kim, W. Jung, and J. H. Ahn, “Aespa: Accuracy preserving low-degree polynomial activation for fast private inference,” 2022. [Online]. Available: https://arxiv.org/abs/2201.06699

[30] R. Gilad-Bachrach, N. Dowlin, K. Laine, K. Lauter, M. Naehrig, and J. Wernsing, “Cryptonets: Applying neural networks to encrypted data with high throughput and accuracy,” in Proceedings of The 33rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. F. Balcan and K. Q. Weinberger, Eds., vol. 48. New York, New York, USA: PMLR, 20–22 Jun 2016, pp. 201–210. [Online]. Available: https: //proceedings.mlr.press/v48/gilad-bachrach16.html

[31] J. Liu, M. Juuti, Y. Lu, and N. Asokan, “Oblivious neural network predictions via minionn transformations,” Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:3617652

[32] R. Dathathri, O. Saarikivi, H. Chen, K. Laine, K. Lauter, S. Maleki, M. Musuvathi, and T. Mytkowicz, “Chet: an optimizing compiler for fully-homomorphic neural-network inferencing,” in Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation, ser. PLDI 2019. New York, NY, USA: Association for Computing Machinery, 2019, p. 142–156. [Online]. Available: https://doi.org/10.1145/3314221.3314628

[33] E. Aharoni, A. Adir, M. Baruch, N. Drucker, G. Ezov, A. Farkash, L. Greenberg, R. Masalha, G. Moshkowich, D. Murik, H. Shaul, and O. Soceanu, “Helayers: A tile tensors framework for large neural networks on encrypted data,” Proceedings on Privacy Enhancing Technologies, vol. 2023, no. 1, p. 325–342, Jan. 2023. [Online]. Available: http://dx.doi.org/10.56553/popets-2023-0020

[34] A. Ebel, K. Garimella, and B. Reagen, “Orion: A fully homomorphic

encryption framework for deep learning,” 2025. [Online]. Available: https://arxiv.org/abs/2311.03470

[35] D. Park, E. Lee, and J.-W. Lee, “Powerformer: Efficient and high-accuracy privacy-preserving language model with homomorphic encryption,” Cryptology ePrint Archive, Paper 2024/1429, 2024. [Online]. Available: https://eprint.iacr.org/2024/1429

[36] R. Ran, N. Xu, W. Wang, G. Quan, J. Yin, and W. Wen, “Cryptogcn: Fast and scalable homomorphically encrypted graph convolutional network inference,” 2022. [Online]. Available: https://arxiv.org/abs/2209.11904

[37] R. Ran, N. Xu, T. Liu, W. Wang, G. Quan, and W. Wen, “Penguin: Parallel-packed homomorphic encryption for fast graph convolutional network inference,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 19 104–19 116. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2023/file/3cc685788a311fa35d8d41df93e288ca-Paper-Conference.pdf

[38] Q. Lou, W.-j. Lu, C. Hong, and L. Jiang, “Falcon: Fast spectral inference on encrypted data,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 2364–2374. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2020/file/18fc72d8b8aba03a4d84f66efabce82e-Paper.pdf

[39] N. Samardzic, A. Feldmann, A. Krastev, S. Devadas, R. Dreslinski, C. Peikert, and D. Sanchez, “F1: A fast and programmable accelerator for fully homomorphic encryption,” in MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 238–252. [Online]. Available: https://doi.org/10.1145/3466752. 3480070

[40] S. Kim, J. Kim, M. J. Kim, W. Jung, J. Kim, M. Rhu, and J. H. Ahn, “Bts: an accelerator for bootstrappable fully homomorphic encryption,” in Proceedings of the 49th Annual International Symposium on Computer Architecture, ser. ISCA ’22. ACM, Jun. 2022, p. 711–725. [Online]. Available: http://dx.doi.org/10.1145/3470496.3527415

[41] J. Kim, G. Lee, S. Kim, G. Sohn, M. Rhu, J. Kim, and J. H. Ahn, “Ark: Fully homomorphic encryption accelerator with runtime data generation and inter-operation key reuse,” in Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO ’22. IEEE Press, 2023, p. 1237–1254. [Online]. Available: https://doi.org/10.1109/MICRO56248.2022.00086

[42] N. Samardzic, A. Feldmann, A. Krastev, N. Manohar, N. Genise, S. Devadas, K. Eldefrawy, C. Peikert, and D. Sanchez, “Craterlake: a hardware accelerator for efficient unbounded computation on encrypted data,” in Proceedings of the 49th Annual International Symposium on Computer Architecture, ser. ISCA ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 173–187. [Online]. Available: https://doi.org/10.1145/3470496.3527393

[43] J. Kim, S. Kim, J. Choi, J. Park, D. Kim, and J. H. Ahn, “Sharp: A short-word hierarchical accelerator for robust and practical fully homomorphic encryption,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, ser. ISCA ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3579371.3589053

[44] X. Deng, S. Fan, Z. Hu, Z. Tian, Z. Yang, J. Yu, D. Cao, D. Meng, R. Hou, M. Li, Q. Lou, and M. Zhang, “Trinity: A general purpose fhe accelerator,” 2024. [Online]. Available: https://arxiv.org/abs/2410.13405

[45] S. Fan, X. Deng, L. Kong, G. Shi, G. Fan, D. Meng, R. Hou, and M. Zhang, “Fast:an fhe accelerator for scalable-parallelism with tunable-bit,” in Proceedings of the 52nd Annual International Symposium on Computer Architecture, ser. ISCA ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 92–106. [Online]. Available: https://doi.org/10.1145/3695053.3731407

[46] T. Zhang, Y. Xue, L. Liang, Z. Gu, Y. Wang, R. Wang, R. Huang, and M. Li, “Flash: An efficient hardware accelerator leveraging approximate and sparse fft for homomorphic encryption,” 03 2025, pp. 1–7.

[47] A. Ebel and B. Reagen, “Osiris: A systolic approach to accelerating fully homomorphic encryption,” 2024. [Online]. Available: https: //arxiv.org/abs/2408.09593

[48] M. Zhou, Y. Nam, X. Wang, Y. Lee, C. Wilkerson, R. Kumar, S. Taneja, S. Mathew, R. Cammarota, and T. Rosing, “Ufc: A unified accelerator for fully homomorphic encryption,” in 2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO), 2024, pp. 352–365.

[49] Prasetiyo, A. Putra, and J.-Y. Kim, “Morphling: A throughputmaximized tfhe-based accelerator using transform-domain reuse,” in 2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2024, pp. 249–262.

[50] R. Agrawal, A. Chandrakasan, and A. Joshi, “Heap: A fully homomorphic encryption accelerator with parallelized bootstrapping,” in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), 2024, pp. 756–769.

[51] M. S. Riazi, K. Laine, B. Pelton, and W. Dai, “Heax: An architecture for computing on encrypted data,” in Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems, ser. ASPLOS ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 1295–1309. [Online]. Available: https://doi.org/10.1145/3373376.3378523

[52] R. Agrawal, L. de Castro, G. Yang, C. Juvekar, R. Yazicigil, A. Chandrakasan, V. Vaikuntanathan, and A. Joshi, “Fab: An fpgabased accelerator for bootstrappable fully homomorphic encryption,” 2022. [Online]. Available: https://arxiv.org/abs/2207.11872

[53] Y. Yang, H. Zhang, S. Fan, H. Lu, M. Zhang, and X. Li, “Poseidon: Practical Homomorphic Encryption Accelerator,” in 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA). Montreal, QC, Canada: IEEE, Feb. 2023, pp. 870–881. [Online]. Available: https://ieeexplore.ieee.org/document/10070984/

[54] S. S. Roy, F. Turan, K. Jarvinen, F. Vercauteren, and I. Verbauwhede, “Fpga-based high-performance parallel architecture for homomorphic computing on encrypted data,” in 2019 IEEE International symposium on high performance computer architecture (HPCA). IEEE, 2019, pp. 387–398.

[55] “Intel homomorphic encryption (he) acceleration library for fpgas,” 2021. [Online]. Available: https://github.com/intel/hexl-fpga

[56] Y. Yang, X. Xu, H. Zhang, J. Song, X. Tang, H. Lu, and X. Li, “Hydra: Scale-out fhe accelerator architecture for secure deep learning on fpga,” in 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), 2025, pp. 1174–1186.

[57] “Cuda-accelerated fully homomorphic encryption library (cufhe),” 2018. [Online]. Available: https://github.com/vernamlab/cuFHE

[58] W. Dai and B. Sunar, “cuHE: A homomorphic encryption accelerator library,” Cryptology ePrint Archive, Paper 2015/818, 2015. [Online]. Available: https://eprint.iacr.org/2015/818

[59] S. Fan, Z. Wang, W. Xu, R. Hou, D. Meng, and M. Zhang, “Tensorfhe: Achieving practical computation on encrypted data using gpgpu,” 2022. [Online]. Available: https://arxiv.org/abs/2212.14191

[60] W. Jung, S. Kim, J. H. Ahn, J. H. Cheon, and Y. Lee, “Over 100x faster bootstrapping in fully homomorphic encryption through memory-centric optimization with GPUs,” Cryptology ePrint Archive, Paper 2021/508, 2021. [Online]. Available: https://eprint.iacr.org/2021/508

[61] K. Shivdikar, Y. Bao, R. Agrawal, M. Shen, G. Jonatan, E. Mora, A. Ingare, N. Livesay, J. L. AbellAN, J. Kim, A. Joshi, and<sup>´</sup> D. Kaeli, “Gme: Gpu-based microarchitectural extensions to accelerate homomorphic encryption,” in Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, ser. MICRO ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 670–684. [Online]. Available: https://doi.org/10.1145/3613424. 3614279

[62] G. Fan, M. Zhang, F. Zheng, S. Fan, T. Zhou, X. Deng, W. Tang, L. Kong, Y. Song, and S. Yan, “Warpdrive: Gpu-based fully homomorphic encryption acceleration leveraging tensor and cuda cores,” in 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), 2025, pp. 1187–1200.

[63] D. Jiao, X. Deng, Z. Wang, S. Fan, Y. Chen, D. Meng, R. Hou, and Z. Mingzhe, “Neo: Towards efficient fully homomorphic encryption acceleration using tensor core,” 06 2025, pp. 107–121.

[64] LattiSense, “Lattisense he accelerator,” accessed: 2025-04-07. [Online]. Available: https://github.com/cipherflow-fhe/lattisense

[65] “Microsoft SEAL (release 4.1),” https://github.com/Microsoft/SEAL, Jan. 2023, microsoft Research, Redmond, WA.

[66] A. A. Badawi, A. Alexandru, J. Bates, F. Bergamaschi, D. B. Cousins, S. Erabelli, N. Genise, S. Halevi, H. Hunt, A. Kim, Y. Lee, Z. Liu, D. Micciancio, C. Pascoe, Y. Polyakov, I. Quah, S. R.V., K. Rohloff, J. Saylor, D. Suponitsky, M. Triplett, V. Vaikuntanathan, and V. Zucca, “OpenFHE: Open-source fully homomorphic encryption library,” Cryptology ePrint Archive, Paper 2022/915, 2022. [Online]. Available: https://eprint.iacr.org/2022/915

[67] J. Fan and F. Vercauteren, “Somewhat practical fully homomorphic encryption,” Cryptology ePrint Archive, Paper 2012/144, 2012, https://eprint.iacr.org/2012/144. [Online]. Available: https://eprint.iacr. org/2012/144

[68] S. Tan, B. Knott, Y. Tian, and D. J. Wu, “Cryptgpu: Fast privacypreserving machine learning on the gpu,” 2021. [Online]. Available: https://arxiv.org/abs/2104.10949

[69] X. Zhou, Z. Xu, C. Wang, and M. Gao, “Ppmlac: high performance chipset architecture for secure multi-party computation,” in Proceedings of the 49th Annual International Symposium on Computer Architecture, ser. ISCA ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 87–101. [Online]. Available: https://doi.org/10. 1145/3470496.3527392

[70] L. Xiaolin, Y. Wei, L. Hongwei, Z. Yong, H. Qinfen, L. Yong, and S. Ninghui, “Pota: A pipelined oblivious transfer acceleration architecture for secure multi-party computation,” IACR Transactions on Cryptographic Hardware and Embedded Systems, vol. 2025, no. 3, p. 262–292, Jun. 2025. [Online]. Available: https://tches.iacr.org/index. php/TCHES/article/view/12217

[71] H. Chen, W. Dai, M. Kim, and Y. Song, “Efficient homomorphic conversion between (ring) lwe ciphertexts,” in Applied Cryptography and Network Security: 19th International Conference, ACNS 2021, Kamakura, Japan, June 21–24, 2021, Proceedings, Part I. Berlin, Heidelberg: Springer-Verlag, 2021, p. 460–479. [Online]. Available: https://doi.org/10.1007/978-3-030-78372-3 18

[72] S. Halevi and V. Shoup, “Faster homomorphic linear transformations in helib,” in Advances in Cryptology – CRYPTO 2018: 38th Annual International Cryptology Conference, Santa Barbara, CA, USA, August 19–23, 2018, Proceedings, Part I. Berlin, Heidelberg: Springer-Verlag, 2018, p. 93–120. [Online]. Available: https://doi.org/10.1007/ 978-3-319-96884-1 4

[73] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[74] X. Wang, A. J. Malozemoff, and J. Katz, “EMP-toolkit: Efficient MultiParty computation toolkit,” https://github.com/emp-toolkit, 2016.

[75] N. Chandran, D. Gupta, A. Rastogi, R. Sharma, and S. Tripathi, “Ezpc: Programmable, efficient, and scalable secure two-party computation for machine learning,” in IEEE European Symposium on Security and Privacy. (IEEE EuroS&P 2019), February 2019.

[76] L. Roy, “SoftSpokenOT: Communication–computation tradeoffs in OT extension,” Cryptology ePrint Archive, Paper 2022/192, 2022. [Online]. Available: https://eprint.iacr.org/2022/192

[77] K. Yang, C. Weng, X. Lan, J. Zhang, and X. Wang, “Ferret: Fast extension for correlated ot with small communication,” in Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 1607–1626. [Online]. Available: https://doi.org/10.1145/3372297.3417276

[78] A. Krizhevsky, V. Nair, and G. Hinton, “Cifar-10 (canadian institute for advanced research).” [Online]. Available: http://www.cs.toronto.edu/ <sup>∼</sup>kriz/cifar.html

[79] A. Krizhevsky, “Learning multiple layers of features from tiny images,” pp. 32–33, 2009. [Online]. Available: https://www.cs.toronto.edu/<sup>∼</sup>kriz/ learning-features-2009-TR.pdf

[80] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” 2015.

[81] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 2015. [Online]. Available: https: //arxiv.org/abs/1409.1556

[82] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” 2019.

[83] T. maintainers and contributors, “Torchvision: Pytorch’s computer vision library,” https://github.com/pytorch/vision, 2016.