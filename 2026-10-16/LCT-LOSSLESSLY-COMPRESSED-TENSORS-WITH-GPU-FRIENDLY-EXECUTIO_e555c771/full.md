# LCT: LOSSLESSLY COMPRESSED TENSORS WITH GPU FRIENDLY EXECUTION FOR TRAINING

Max Zhu<sup>1†</sup> Bohang Sun<sup>1</sup>

<sup>1</sup> University of Cambridge

Pietro Liò<sup>1</sup> Jialin Yu<sup>2</sup>

<sup>2</sup> University of Oxford

## ABSTRACT

Lossless tensor compression could reduce memory usage during training, but training requires tensors to be repeatedly re-encoded. We introduce Losslessly Compressed Tensors (LCT), a GPU execution framework designed for this repeated encode–decode regime. LCT exploits the natural redundancy in floating point data to reduce storage for weights, activations and optimizer states. Highly optimized GPU kernels implement Huffman coding with precomputed codebooks for low compression and decompression overhead, essential for training. Our analysis characterizes LCT’s storage costs and adaptation to magnitude shifts. LCT achieves a compression ratio of up to 1.455×, and is demonstrated in training and finetuning settings. On a 162M-parameter NanoGPT, LCT reduces memory usage by 24.6% for a 7.2% step-time increase. In Qwen3-4B fine-tuning, LCT saves up to 23.8% of peak memory, enabling a 3.5X larger context length in training on the same GPU.

## 1 Introduction

Memory is a limiting factor in training large language models (LLMs), larger models and longer sequence lengths have increased memory requirements. Weights, optimizer buffers and transient activation caches all consume memory. Although a slower device can be compensated for with longer training, a memory limitation is much more difficult to overcome. Offloading storage to the CPU (Rhu et al., 2016; Ren et al., 2021), or even NVMe (Rajbhandari et al., 2021) allows for larger models, but can easily suffer from IO bottlenecks.

The dominant methods to tackle this trade off accuracy for storage. Low precision training datatypes such as FP8 (Micikevicius et al., 2022) and FP4 (Chmiel et al., 2025) use fewer bits per value. However, intermediate activations and caches still use higher precision to ensure numerical stability and hardware support is limited to the most modern hardware. Activation quantization has been explored down to 2 bits per value in ActNN (Chen et al., 2021), while other methods exploited the low rank nature of activations (Shamshoum et al., 2025). Optimizer states have been quantized to 8 bits (Dettmers et al., 2022). 3-4 bit weight quantization methods such as GPTQ (Frantar et al., 2023) and AWQ (Lin et al., 2024) are widely adopted for inference. These methods are designed to minimise accuracy loss, but are not entirely lossless and require careful tuning to avoid accuracy degradation.

Lossless compression is an alternative. Gist (Jain et al., 2018) applies lossless compression to ReLU activations, but is not universally applicable. General methods have been developed to take advantage of compression for inference (Zhang et al., 2025; Hershcovitch et al., 2025; Guo and Joshi, 2026; Fan et al., 2026). However, these methods have not been applied for training. Unlike inference, weights and optimizer states are updated every step, and activations must persist between the forward and backward pass. Table 1 summarizes these lifetimes and the corresponding operations for an SGD-momentum (SGDM) step. Existing methods for lossless compression offer fast decoding speeds, but are too slow to encode (Table 4). Training turns lossless compression into a repeated encode–decode problem rather than a decode-centric one. Figure 2 illustrates this distinction.

![](images/5d44de20971d84ee453f80be1bf3dc71efd61bef721663fbbeeeeaa13d457756.jpg)  
Figure 1: Overview of LCT. Sign-mantissa bytes remain unchanged while exponents are encoded with bounded Huffman codes with escapes for rare values. Values are packed into fixed sized streams with overflow buffers. Tensors are decoded losslessly, with fused kernels for optimizer updates.

We introduce Losslessly Compressed Tensors (LCT) (Figure 1). Designed for high performance on GPU hardware, LCT losslessly compresses the exponents of Bfloat16 (BF16) values to reduce memory usage during training. Fused pointwise operators are introduced for optimizers with compressed buffers, while a shared GPU buffer removes host synchronization during training. PyTorch autograd integration compresses cached activations. The tensors remain compressed until they are needed, at which point they are decompressed on-demand.

Our contributions are:

• A lossless BF16 codec and training integration, including GPU Triton kernels, fused pointwise updates, and a preallocated buffer.

• An analysis of coding and allocated storage costs under model mismatch, together with bounds explaining how exponent centering compensates for magnitude changes during training.

• An evaluation spanning codec throughput, fused operators, and training memory. LCT encodes 2.38× faster than the next-fastest evaluated codec on an RTX 3090 and reduces peak memory by over 20% in training and fine tuning applications.

![](images/1ddedef19cdeca030f793a083dd81ed23f68729cab86205d27de73870ae72c8c.jpg)  
Figure 2: Compression use cases ordered by increasing encode-bandwidth requirement. Static inference weights amortize encoding across repeated use; autoregressive KV caches add entries at each token; general tensor compression must accommodate frequent updates. The ordering is qualitative and reflects these access patterns, rather than measured throughput.

Table 1: Tensor lifecycle and LCT codec operations per SGD-with-momentum step, after momentum initialization. The weight row covers MLP and other layer weights, assuming one dense use in forward and one in backward. Encodes include output recompression. Dense decodes materialize a tensor; fused decodes consume it within an update. Counts describe logical operations, not kernel launches.
<table><tr><td>Tensor</td><td></td><td>Dense Encodes decodes</td><td>Fused decodes</td><td>Uses and lifetime</td></tr><tr><td>Weights W</td><td>1</td><td>2</td><td>1</td><td>Forward and backward computation; fused  $\overline { { W  W - \eta M } } .$  Persist across steps.</td></tr><tr><td>Biases</td><td>0</td><td>0</td><td>0</td><td>Kept dense because they are small; updated with dense operations.</td></tr><tr><td>Momentum M</td><td>1</td><td>0</td><td>2</td><td>Fused  $M  \beta M + G ,$  then fused  $W  W - \eta M$  . Persists across steps.</td></tr><tr><td>Activations</td><td>1</td><td>1</td><td>0</td><td>Activations are encoded in forward and decoded for gradient computation in backward.</td></tr></table>

## 2 Related Work

GPU-accelerated lossless compression. Several methods have investigated using lossless compression on the GPU. The nvCOMP library (NVIDIA Corporation, 2026) is a GPU-optimized library that implements several widely used compression methods, including LZ4 (Collet, 2011), as well as proprietary methods for floating-point data such as Bitcomp and Cascaded. Fast GPU-optimized ANS coding (Duda, 2013) was explored by Weißenberger and Schmidt (2019). Huffman encoding has emerged as an efficient GPU compression and decompression method. Yamamoto et al. (2020) explored using gap arrays for efficient compression and decoding. Tian et al. (2021) explored optimizing the full Huffman encoding and decoding process on GPUs.

Lossless compression for inference. Lossless weight compression techniques have been explored, such as DFloat11 (Zhang et al., 2025), which utilizes Huffman encoding with gap arrays for fast GPU decoding. ZipNN (Hershcovitch et al., 2025) uses similar principles. Both methods encode the exponent, with the encoding process done on the CPU. This allows for fast decoding, but encoding is comparatively slow. ZipServ (Fan et al., 2026) combines a tensor-core-aware bitmap format with ZipGEMM, which fuses decompression with matrix multiplication. Agrawal et al. (2026) and Lin et al. (2026) use GPU-native Huffman codes for network communication. Recently, SplitZip (Guo and Joshi, 2026) proposed lossless KV-cache compression by replacing exponents with 4-bit symbols and using escape values for rarer cases. While simple, its encoding achieves suboptimal compression. Appendix B.4 provides a comparison between LCT and SplitZip.

## 3 Compressibility in tensors

## 3.1 Exponent compressibility in floating point data

The structure of floating-point tensors provides an opportunity for lossless compression. The BF16 data type consists of 1 sign bit, 8 exponent bits, and 7 mantissa bits (Burgess et al., 2019). Typical data is well described as the quantization of a concentrated continuous distribution, the sign and mantissa have high entropy, while the exponent has lower entropy. Compressing the exponent only is sufficient, for example, a standard Gaussian rounded to BF16 has exponent entropy 2.545 bits/symbol, while the full distribution’s entropy is 10.464 bits/symbol. Encoding the full BF16 value saves only 0.081 bits/value compared with compressing only the exponent. Appendix D.5 gives the calculation. Appendix D.3 explains why this is the case.

## 3.2 Structure in weights and activations

The distributions encountered in neural networks are compressible. Figure 5 shows the exponent distributions of the six largest parameter groups in the Nemotron-H-8B LLM (Blakeman et al., 2025). All distributions are highly concentrated around their peaks. In each group, just 9–11 of the 256 exponent symbols account for at least 99% of the weights, with marginal entropy of 2.58–2.70 bits. The compressibility of the weights has been noted in several works (Hershcovitch et al., 2025; Zhang et al., 2025; Fan et al., 2026).

A similar trend emerges in the activations. Figure 6 shows the exponent distributions of five Nemotron-H-8B activation groups, aggregated across layers and four 1024-token sequences. Again, the distributions are highly concentrated, though there is more variation. In particular, the squared ReLU activation has high sparsity, with 93.23% being exactly zero. Exponent entropy ranges from 0.63 bits for the zero-rich FFN hidden state to 3.91 bits for Mamba hidden activations (Appendix E.2).

## 4 Methodology

## 4.1 Bounded exponent codes

Coding background. Huffman coding (Huffman, 1952) assigns variable-length prefix-free codewords, with shorter codes for more probable symbols. For a matched source model, its expected length is less than one bit per symbol above entropy, but individual codes can be as long as the alphabet size minus one. Length-limited Huffman coding (Larmore and Hirschberg, 1990) bounds this worst case, and Package-Merge efficiently constructs the optimal code under the length constraint. Escape coding (Vitter, 1987) groups rare symbols under a shared prefix followed by a literal, freeing shorter codes for common symbols. LCT uses bounded Huffman codes with escapes to combine compact storage with simple parallel decoding

Arithmetic coding (Witten et al., 1987) and ANS (Duda, 2013) can approach the entropy limit without the integer-length constraint on individual symbols, but are slower to implement on the GPU, as we show in the comparisons.

LCT code construction. LCT splits each BF16 value into an exponent byte and an unchanged sign–fraction byte. We write $\mathcal { P } = { \left( p _ { t } \right) }$ for the centered distribution of exponent symbols in the tensor, and $\overset { \vartriangle } { \boldsymbol { \mathcal { Q } } } = \left( \bar { q } _ { t } \right)$ for the fixed coding model.

The user selects $\mathcal { Q }$ based on prior knowledge of the tensor distribution. It is not estimated at runtime, avoiding codebook construction during training. At runtime, a sampled exponent center shifts the codebook to adapt to the tensor scale (Section 4.4). The codebooks are precomputed before running and a mismatch between $\mathcal { P }$ and $\mathcal { Q }$ can increase storage.

LCT adopts bounded Huffman coding with escape coding. Prefixes are capped at eight bits and rare escaped symbols are encoded with an escape prefix followed by an eight-bit literal. Without escapes, coding all 256 symbols under this cap would force every code to have length eight. Escapes free short prefixes for common exponents.

The codebook construction is based on Package-Merge (Larmore and Hirschberg, 1990). LCT further adapts this algorithm to handle escape values. We sort symbols by probability $q _ { t }$ and test every tail cutoff. Package-Merge optimizes prefix lengths for each candidate. The final code minimizes

$$
\mathcal { C } ( D ) = \sum _ { t \in D } q _ { t } \ell _ { \mathcal { Q } } ( t ) + q _ { \mathrm { e s c } } \left( \ell _ { \mathrm { e s c } } + 8 \right) ,\tag{1}
$$

where $D$ contains directly coded symbols, $\ell _ { \mathcal { Q } } ( t )$ is the length Q assigns to symbol $t , \ell _ { \mathrm { e s c } }$ is the escape-prefix length, and $q _ { \mathrm { e s c } }$ is the escaped tail mass. Algorithm 1 gives the full algorithm, with an optimality proof given in Appendix B.1.

## 4.2 Storage structure

Storage consists of two components. The sign and fraction bits are stored together as uncompressed sign–fraction bytes.   
The encoded exponents are stored in fixed-size blocks.

Each block holds B exponent symbols. The clean and sparse configurations use $B = 6 5 { , } 5 3 6$ and the medium and high configurations use 32,768 (Table 12). Blocks tile the tensor, with padding. Each block is divided into $L = 2 5 6$ streams, independent units that enable parallel encoding and decoding, containing $S = B / L$ symbols at a fixed budget of $F$ bits. The sizes are tuned based on the expected size of the encoded stream. The mapping from tensor positions to streams is cyclic. Consecutive elements land in different streams so streams do not hold a full row or column of a tensor, which may be strongly correlated (e.g. a neuron or batch may output all constant values).

A fixed budget cannot cover every stream, because code lengths vary. If a stream’s codes exceed $F ,$ LCT writes an overflow tail. The remaining exponents are stored uncompressed. A marker indicates where the tail starts, if any.

## 4.3 User-selected models and budgets

The users have several options to choose from when selecting Q. By specifying the distributions a priori, LCT avoids computing Huffman tables at runtime, improving compression performance, while still being applicable to different scenarios.

Distribution. LCT provides a set of default codebook distributions, including a power-law model with an exponential body and a power-law tail, based on Nemotron-H activations. In addition, several other parameters may be specified. A shift parameter allows for distributions shifted by an additive constant. Zero inflation specifies a percentage of the distribution that is set to zero. Together with exponent shifting, these handle the typical distributions encountered during training. The user may specify additional exponent distributions if required in terms of a probability density table.

Storage budget. The storage budget is specified as CLEAN (default), SPARSE, MEDIUM, or HIGH. Larger budgets increase the base storage used, but may decrease the amount of overflow for noisy or out-of-distribution tensors. The SPARSE setting provides the smallest budget and is intended for sparse ReLU activations. Table 12 lists the stream configurations and budgets.

## 4.4 Standard encoding and decoding

Encoding. The input is first flattened and padded if needed. A small GPU kernel samples up to 4096 pseudo-random elements to estimate an integer exponent center (to avoid batch correlations) and a second kernel shifts the selected encoding table to this center. The main encoding kernel assigns one program per block and one lane per stream. It reads BF16 values using the cyclic mapping, extracts and stores their unchanged sign-mantissa bytes, and packs the exponent codes into the fixed budget. Each stream records where its fixed budget is exceeded. Subsequent kernels count overflowing streams and tail bytes, compact the stream metadata, and copy raw exponent tails from the input into private or shared storage (Section 4.6), with early exit if there is no overflow.

Decoding. The decoding table is first shifted by the stored exponent center. The main decoding kernel traverses each stream independently, looking up a prefix, with a lookup table for decoding. Escaped symbols additionally consume their literal bits. It combines each recovered exponent with its sign–fraction byte and writes the reconstructed BF16 bits. A separate overflow kernel reads the raw tails and overwrites the corresponding output positions.

## 4.5 Fused pointwise operators

The optimizer requires several pointwise operations, such as adding and multiplying compressed tensors. For example, the SGD update is $W _ { t + 1 } = W _ { t } ^ { \mathbf { * } } - \eta M _ { t }$ , where $W _ { t }$ and $W _ { t + 1 }$ are compressed weights, $\bar { M _ { t } }$ is compressed momentum, and η is the learning rate. A naive implementation would require decompressing $W _ { t }$ and $M _ { t } ,$ , dense arithmetic, then recompressing the result. For an N-element tensor, each encode-decode cycle uses roughly 4N bytes of memory traffic. A naive SGD update would require approximately 16N bytes of memory traffic.

We introduce fused operations for pointwise operations required during training. Each operation consists of two stages. First, a kernel takes in the compressed tensors, computes the output and saves the exponent and sign–fraction byte to separate arrays. Next, the exponents are compressed. This substantially reduces memory traffic. Our fused implementation requires ∼ 8N bytes (the first kernel requires ∼ 6N bytes and exponent compression requires ∼ 2N)

A selection of kernels is provided covering all the use cases for typical training with compressed and dense outputs. GEMMs still reconstruct compressed operands as dense inputs, since we did not find significant speedup with fused encode/decode GEMMs. Table 13 lists all operators and training uses. Appendix C describes the codec and pointwise kernels, including stream parallelism, table lookups, register-based bit packing, and overflow handling.

## 4.6 Shared GPU buffer

Our compressed tensors have variable sizes. The sizes are not known at compile time, so allocating exact-sized buffers at runtime introduces host synchronization. This slows down execution, particularly for our fused pointwise kernels. To avoid this, we introduce the shared OverflowBuffer. The OverflowBuffer reserves a large block of device memory, and allocates variable-size tensors without host synchronization. All of our kernels work with the buffer by specifying storage location through the buffer’s address and tensor-specific offset. This is an optional feature.

The OverflowBuffer implements a minimal memory management framework entirely in Triton, including allocating tensors, releasing tensors, alignment, and merging adjacent free regions to reduce fragmentation. Operations are performed asynchronously. Appendix C.4 provides more details.

## 4.7 PyTorch training integration

Compressed tensors are represented as a CompressedTensor class, storing the compressed exponents, overflow tails, sign–fraction bytes, and metadata. Custom torch.autograd.Function classes integrate support for LCT into training. These functions compress saved tensors in the forward pass and reconstruct them in the backward pass, maintaining numerical equivalence. Custom compressed optimizers support compressed momentum buffers using the fused pointwise operators. Implementations of SGD-M, Adam (Kingma and Ba, 2015), and Muon (Jordan et al., 2024b) are provided. Listing 1 shows an example of a feed-forward layer implementation, including activation compression in forward and reconstruction for the backward matrix products.

Table 2: Ideal cross entropy (left) and allocated LCT exponent storage, (right), in bits per exponent. Source distributions are in rows and encoding models in columns. G: Gaussian; L: Laplace; PL: power-law; Γ: Gamma. Bold entries use matched models
<table><tr><td colspan="5">Ideal cross entropy (bits/exponent) Γ</td><td colspan="4">Allocated LCT storage (bits/exponent) Γ</td></tr><tr><td>P\Q</td><td>G</td><td>L</td><td>PL</td><td>P\Q</td><td>G</td><td></td><td>L</td><td>PL</td></tr><tr><td>Gaussian</td><td>2.545</td><td>2.630</td><td>2.637 3.007</td><td>Gaussian</td><td>3.000</td><td>3.125</td><td>3.125</td><td>3.692</td></tr><tr><td>Laplace</td><td>3.039</td><td>2.832</td><td>2.844 3.062</td><td>Laplace</td><td>3.636</td><td>3.126</td><td>3.126</td><td>3.725</td></tr><tr><td>Power-law</td><td> $\mathrm { 8 \times 1 0 ^ { 7 } }$ </td><td>2.976 </td><td>2.891 3.118</td><td>Power-law</td><td>3.942</td><td>3.176</td><td>3.133</td><td>3.828</td></tr><tr><td>Gamma</td><td>5.644</td><td>3.417</td><td>3.307 3.015</td><td>Gamma</td><td>6.554</td><td>4.385</td><td>3.972</td><td>3.235</td></tr></table>

Table 3: Average exponent storage (bits/exponent, left) and measured compression ratios (right) for Nemotron-H-8B parameters using four LCT codecs. Bold entries indicate the best results in each parameter group.
<table><tr><td rowspan="2"></td><td colspan="3">Average bits per exponent</td><td colspan="3">Compression ratio (×)</td></tr><tr><td></td><td></td><td>Mamba</td><td></td><td></td><td>Mamba</td></tr><tr><td>Codec</td><td>Attention</td><td>MLP</td><td>projection</td><td>Attention</td><td>MLP</td><td>projection</td></tr><tr><td>Power-Law</td><td>3.1250</td><td>3.4041</td><td>3.1304</td><td>1.4382</td><td>1.4030</td><td>1.4375</td></tr><tr><td>Gaussian</td><td>3.4984</td><td>4.1793</td><td>3.8003</td><td>1.3915</td><td>1.3137</td><td>1.3559</td></tr><tr><td>Laplace</td><td>3.1250</td><td>3.4025</td><td>3.1312</td><td>1.4382</td><td>1.4032</td><td>1.4374</td></tr><tr><td>Gamma</td><td>3.2668</td><td>3.4416</td><td>3.1312</td><td>1.4201</td><td>1.3984</td><td>1.4374</td></tr></table>

## 5 Analysis of LCT

For exponent symbols drawn from ${ \mathcal P } ,$ an ideal entropy coder using model $\mathcal { Q }$ has expected cost $\begin{array} { r } { H ( \mathcal { P } ) + D _ { \mathrm { K L } } ( \mathcal { P } | | \mathcal { Q } ) } \end{array}$ bits per exponent. Storing the sign–fraction byte unchanged adds eight bits per BF16 value. Excluding overflow, LCT uses finite prefix codes with mean length $\begin{array} { r } { \bar { \ell } \dot { = } \sum _ { t } p _ { t } \ell _ { \mathcal { Q } } \bar { ( t ) } } \end{array}$ , including escape literals. The expected code length per BF16 value is

$$
8 + \bar { \ell } = 8 + H ( \mathcal { P } ) + D _ { \mathrm { K L } } ( \mathcal { P } \| \widetilde { \mathcal { Q } } ) - \log _ { 2 } Z _ { \mathcal { Q } }\tag{2}
$$

where $\widetilde { \mathcal { Q } } = \widetilde { \left( q _ { t } \right) } _ { t }$ is the probability distribution represented by the resulting code lengths, rather than the design distribution Q, and $0 < \mathsf { \bar { Z } } _ { \mathcal { Q } } \leq 1$ is the codebook normalization constant. LCT’s cost consists of source entropy, mismatch to the code-induced distribution ${ \mathcal { \widetilde { Q } } } ,$ and unused prefix-code capacity $- \log _ { 2 } Z _ { \mathcal { Q } }$ . A full derivation and definitions of these quantities are given in Appendix D.

Theoretical compression performance. Table 2 compares the ideal cross entropy and average exponent storage for LCT for each source–model pair. Under mismatch LCT’s bounded codes can deliver better performance than entropy coding. This is particularly pronounced for a power-law source encoded with a Gaussian model, whose much lighter tail assigns extremely small probabilities to large exponents. Appendix D.4 explains the calculation and this tail effect.

Empirical compression performance. Table 3 reports measured compression ratios for Nemotron-H-8B parameters in attention, MLP, and Mamba projection layers using our power-law, Gaussian, Laplace, and Gamma codecs. The heavy-tailed weights mean the Laplace and Power-Law distributions give the best compression ratio, close to the theoretical optimum.

## 5.1 Adaptation to magnitude scale

Tensor magnitudes change during training which LCT handles by shifting the codebook to match the exponent center at runtime. The following propositions explain why this compensates for rescaling.

Proposition 1 (Rescaling mostly shifts exponents) Ignoring rounding and finite-range effects, multiplying finite, nonzero values by a finite, nonzero constant shifts their exponents by a common integer, with at most a one-bin difference remaining after correction. For power-of-two scaling, the shift correction is exact.

Proposition 2 (Rare boundary effects give a small correction) For the same rescaling, let A and B be the stored BF16 exponent bytes before and after scaling. Ifrounding orfinite-range effects alter either ideal exponent label with probability ε, then, with h as the binary entropyfunction,

$$
| H ( B ) - H ( A ) | \leq 1 + 7 \varepsilon + h _ { 2 } ( \varepsilon ) ,\tag{3}
$$

Table 4: Time taken to encode and decode a 1 GiB tensor. Standard errors in the times are shown. Compression ratio is original size divided by compressed size. The fastest method is marked in bold.
<table><tr><td>Method</td><td>Encode / ms</td><td>Decode / ms</td><td>Compression ratio</td></tr><tr><td>nvCOMP LZ4</td><td> $\overline { { 4 8 4 . 5 8 4 \pm 0 . 6 4 6 } }$ </td><td> $\overline { { 8 . 7 6 1 \pm 0 . 0 3 5 } }$ </td><td>0.998×</td></tr><tr><td>nvCOMP Bitcomp</td><td> $4 0 . 4 1 3 \pm 0 . 5 8 5$ </td><td> $4 . 1 9 3 \pm 0 . 0 3 5$ </td><td>0.998×</td></tr><tr><td>nvCOMP ANS</td><td> $5 3 . 0 2 8 \pm 0 . 4 8 9$ </td><td> $4 . 3 4 9 \pm 0 . 0 5 9$ </td><td>1.258×</td></tr><tr><td>DFloat11</td><td> $2 5 7 4 . 0 0 0 \pm 0 . 6 7 1$ </td><td> $4 . 4 7 5 \pm 0 . 0 1 4$ </td><td>1.483×</td></tr><tr><td>ZipNN</td><td> $8 4 5 . 4 0 0 \pm 0 . 8 4 1$ </td><td> $1 9 9 . 3 0 0 \pm 0 . 6 1 9$ </td><td>1.511×</td></tr><tr><td>SplitZip</td><td> $5 . 8 5 0 \pm 0 . 0 1 6$ </td><td> $\mathbf { 2 . 3 0 9 \pm 0 . 0 0 4 }$ </td><td>1.333×</td></tr><tr><td>LCT</td><td> $\mathbf { 2 . 4 5 4 \pm 0 . 0 0 5 }$ </td><td> $2 . 5 1 3 \pm 0 . 0 0 5$ </td><td>1.455×</td></tr></table>

Appendix D.2 proves both propositions. Proposition 1 explains how exponent centering corrects for changes in scale. Proposition 2 bounds the entropy change by one bit plus a small correction. Together, these propositions explain why shifting the codebook can adapt LCT to different input scales at runtime.

## 5.2 Allocated rate and scaling with tensor size

For a tensor of N elements comprising full blocks, let S be the symbols per stream, F the reserved bits per stream, T the number of raw tail exponents, and K the number of overflowing streams. Each tail exponent occupies one byte and each affected stream requires nine metadata bytes. The core rate is

$$
b _ { \mathrm { L C T } } = 8 + { \frac { F } { S } } + 8 { \frac { T } { N } } + 7 2 { \frac { K } { N } } , \qquad \rho _ { \mathrm { L C T } } = { \frac { 1 6 } { b _ { \mathrm { L C T } } } } .\tag{4}
$$

The first two terms are the regular payload; the last two account for raw overflow tails and overflow metadata. A smaller F reduces the fixed payload but can increase the number of overflowing streams K and raw tail length T, reducing the compression ratio. This assumes full blocks and excludes padding and allocation overhead. Appendix B.2 gives a more detailed analysis.

## 6 Experiments

## 6.1 Encoding and decoding bandwidth

First, we evaluate LCT’s encoding and decoding performance and compare it with existing methods. Table 4 compares the time to encode and decode a 1 GiB Gaussian tensor using LCT, SplitZip, DFloat11, ZipNN, and three nvCOMP codecs, LZ4, ANS, and Bitcomp. Appendix A.1 gives implementation details.

LCT encodes fastest, with 2.38× the throughput of SplitZip, the next-fastest encoder. SplitZip decodes slightly faster, but LCT has a lower combined encode–decode time (4.967 versus 8.159 ms) and a higher compression ratio (1.455× versus 1.333×). The nvCOMP LZ4 and Bitcomp configurations slightly expand the input (0.998×), while nvCOMP ANS achieves 1.258× compression. All three encode and decode more slowly than LCT. DFloat11 and ZipNN achieve higher compression ratios, but their encoding times are over two orders of magnitude longer than LCT’s.

Next, Figure 3 evaluates the throughput for varying input sizes. LCT scales well with input size, with peak encode throughput reaching 395.5 GiB/s.

Table 8 in the appendix extends the comparison to NVIDIA V100, A40, and H100 GPUs. LCT remains the fastest encoder, while SplitZip remains the fastest decoder.

## 6.2 Fused pointwise operators

Table 5 compares fused operators with two baselines using Gaussian BF16 input tensors of 1 GiB each. The naive path decompresses all operands before computation and returns the requested output format. The dense baseline is the standard PyTorch implementation, without any compression. Appendix A.2 gives the protocol. For dense output, LCT is 1.66–1.72× faster than the naive path. Operations with one compressed operand also slightly outperform the dense baseline since compression reduces the size of the tensor, and hence memory traffic. For compressed outputs, LCT i 1.64–2.56× faster than the naive path.

Table 5: Fused pointwise operator timings (mean ± standard error). C denotes a compressed operand, D a dense operand, and α a scalar. Speedup compares LCT with Naive.
<table><tr><td>Operation</td><td>Output</td><td>LCT /ms</td><td>Naive / ms</td><td>Dense / ms Speedup</td><td></td></tr><tr><td> $\overline { { C + D } }$ </td><td>Dense</td><td> $\overline { { 3 . 6 9 4 \pm 0 . 0 0 3 } }$ </td><td> $6 . 3 4 2 \pm 0 . 0 1 5$ </td><td> $\overline { { 3 . 8 5 5 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 1 . 7 1 7 \times } }$ </td></tr><tr><td> $C \odot D$ </td><td>Dense</td><td> $3 . 6 9 1 \pm 0 . 0 0 3$ </td><td> $6 . 3 4 7 \pm 0 . 0 2 1$ </td><td> $3 . 8 6 6 \pm 0 . 0 1 4$ </td><td> $1 . 7 2 0 \times$ </td></tr><tr><td> $\alpha C + D$ </td><td>Dense</td><td> $3 . 8 2 2 \pm 0 . 0 0 3$ </td><td> $6 . 3 3 1 \pm 0 . 0 0 8$ </td><td> $3 . 8 5 9 \pm 0 . 0 0 5$ </td><td> $1 . 6 5 6 \times$ </td></tr><tr><td> $\alpha C _ { 1 } + C _ { 2 }$ </td><td>Dense</td><td> $5 . 2 6 2 \pm 0 . 0 0 4$ </td><td> $8 . 7 0 9 \pm 0 . 0 0 3$ </td><td> $3 . 8 6 4 \pm 0 . 0 0 9$ </td><td> $1 . 6 5 5 \times$ </td></tr><tr><td> $\overline { { C + D } }$ </td><td>Compressed</td><td> $\overline { { 5 . 0 5 8 \pm 0 . 0 1 0 } }$ </td><td> $\overline { { 8 . 8 5 8 \pm 0 . 0 0 9 } }$ </td><td> $\overline { { 3 . 8 5 2 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 1 . 7 5 1 \times } }$ </td></tr><tr><td> $C \odot D$ </td><td>Compressed</td><td> $5 . 0 0 6 \pm 0 . 0 0 9$ </td><td> $1 2 . 8 1 4 \pm 0 . 0 1 9$ </td><td> $3 . 8 4 8 \pm 0 . 0 0 3$ </td><td> $2 . 5 6 0 \times$ </td></tr><tr><td> $\alpha C + D$ </td><td>Compressed</td><td> $5 . 1 4 0 \pm 0 . 0 1 0$ </td><td> $8 . 7 0 0 \pm 0 . 0 0 6$ </td><td> $3 . 8 6 8 \pm 0 . 0 1 2$ </td><td> $1 . 6 9 3 \times$ </td></tr><tr><td> $\alpha C _ { 1 } + C _ { 2 }$ </td><td>Compressed</td><td> $6 . 7 6 8 \pm 0 . 0 2 1$ </td><td> $1 1 . 0 9 1 \pm 0 . 0 0 9$ </td><td> $3 . 8 5 4 \pm 0 . 0 0 3$ </td><td> $1 . 6 3 9 \times$ </td></tr></table>

Table 6: NanoGPT training time per optimizer step and peak PyTorch-allocated GPU memory, including the shared buffer. The effective compression ratio is baseline peak memory divided by each configuration’s peak memory, including all live GPU allocations.
<table><tr><td>Compressed components</td><td>Peak memory / MiB</td><td>Effective compression ratio</td></tr><tr><td>None</td><td>Time / s 8.411</td><td>16058.0</td></tr><tr><td>Activations</td><td>9.013</td><td>1.000× 12103.0 1.327×</td></tr><tr><td>+ Weights</td><td>9.131</td><td>1.337×</td></tr><tr><td>+ Momentum Buffers</td><td>9.136</td><td>12013.8 11964.6 1.342×</td></tr></table>

## 6.3 Training NanoGPT

A 162M-parameter transformer is trained on the FineWeb-10B dataset (Penedo et al., 2024) for 3400 steps with and without LCT. The setup is based on Track 3 of the Modded NanoGPT project (Jordan et al., 2024a), with optimizations to reduce the baseline memory usage; Appendix A.3 gives the details. The LCT run uses a 64 MiB shared OverflowBuffer. Final validation losses are 3.27978 with LCT and 3.27981 without LCT, within nondeterministic rounding error.

Ablations and checkpoint analysis. Table 6 compares compression configurations in a short run, reporting time per training step, peak memory, and effective compression ratio. Activation compression reduces peak memory from 16058 to 12103 MiB (24.6%), with a 7.2% increase in time. Further compressing weights and momentum buffers saves another 138.4 MiB. Table 10 isolates the shared OverflowBuffer: it reduces step time by 2.3% at a small memory cost. Figure 4 tracks feed-forward weight and momentum exponent distributions across checkpoints. Their magnitudes shift by roughly $2 ^ { 6 } = 6 4$ . Table 11 reports the corresponding checkpoint compression ratios. The compression ratios are extremely consistent beyond the first few steps, reaching the 1.4382 limit of the CLEAN scheme. The first steps are slightly worse because many values are zero-initialized.

## 6.4 Fine-tuning larger models

A 4B parameter Qwen3-4B model (Qwen-Team, 2025) is fine-tuned to investigate the performance on a larger model, where parameters and momentum buffers make up a more significant portion of the memory usage. Full-parameter fine-tuning in BF16 is used for all weights, activations, and momentum. Without compression, momentum and parameters use around 8 GiB of memory each. Full setup details are given in Appendix A.4.

Table 7 compares standard tuning and LCT across sequence lengths from 500 to 3500 tokens. At 500 and 1000 tokens, LCT reduces peak memory by 22.1% and 23.8%, respectively, with step-time increases of 8.0% and 6.7%. The uncompressed baseline runs out of memory for the tested sequence lengths above 1000 tokens. This extends the largest supported sequence length by 3.5×, demonstrating the effectiveness of compression and how it can be used to train longer sequences on the same hardware.

## 7 Conclusion

Finding. We presented LCT, a lossless BF16 tensor codec designed for repeated encoding and decoding during training. LCT combines bounded Huffman coding, parallel GPU execution, fused pointwise updates, and shared GPU memory management. Across the evaluated GPUs, it provides the fastest encoding among the tested methods and reduces memory usage in both training and finetuning with little performance penalty. These results show that lossless compression can serve as a practical training-memory primitive when both encoding and decoding are designed for repeated GPU execution. Limitations and scope. LCT currently targets BF16, relies on user-specified tensor distributions, and requires explicit integration with training operations. Its effectiveness may vary for architectures or tensor types with different exponent statistics. Future work includes decoding within GEMM kernels, improved adaptation to distribution shift, and combining lossless with lossy compression.

Table 7: Qwen3-4B full-parameter fine-tuning at batch size 1. Memory is peak PyTorch-allocated GPU memory, including any shared buffer, and time is per training step. Dashes indicate runs that failed to complete.
<table><tr><td colspan="3">Baseline</td><td colspan="2">LCT</td></tr><tr><td>Sequence length</td><td>Memory / MiB</td><td>Time / s</td><td>Memory / MiB</td><td>Time / s</td></tr><tr><td>500</td><td>19439.2</td><td>3.759</td><td>15141.8</td><td>4.058</td></tr><tr><td>1000</td><td>21327.8</td><td>4.006</td><td>16262.4</td><td>4.273</td></tr><tr><td>1500</td><td></td><td>一</td><td>17370.1</td><td>4.523</td></tr><tr><td>2500</td><td></td><td></td><td>19617.6</td><td>5.055</td></tr><tr><td>3500</td><td></td><td>—</td><td>21842.1</td><td>5.607</td></tr></table>

## AI use statement

Agentic LLMs were used to optimize the Triton kernels. We first manually designed a series of test harnesses that contain correctness checks and benchmarks (given in the experiments folder of the Supplementary materials). An Autoresearch agent was targeted at these specifications. The prompt is reproduced in program.md. Manual supervision was used throughout the process, with suggestions and advice given to the agent. Ideas such as the bounded Huffman scheme, OverflowBuffer and fused optimizer operators are the Authors, and given to the LLM to implement.

Once the kernels were optimized, the rest of the code was written with less LLM assistance. This includes the PyTorch interface and autograd functions and main experimental setups.

Generative AI tools were used to assist with manuscript drafting and revision, mathematical exposition, and the preparation of figures and tables. AI assistance was also used to check for errors in the manuscript.

## Reproducibility statement

Appendix A describes the benchmark protocols and training configurations. Algorithm 1 specifies codebook construction, with its optimality argument in Appendix B.1. Appendix C details the GPU implementation, and Listing 1 illustrates integration with automatic differentiation. The assumptions and derivations underlying the compression analysis are provided in Appendix D, while Appendix E describes the weight and activation measurements.

The supplementary materials contain our codebase, including the LCT package, Autoresearch prompt and code for the experiments.

## Ethics statement

This work studies lossless compression of model tensors and evaluates it using synthetic data, the existing FineWeb dataset, and pretrained models. No human participants were recruited and no new personal data were collected. Use of web-derived training data remains subject to privacy, copyright, and bias considerations. The main impact of this work is lowering memory requirements, making larger training workloads more accessible. The energy impact of compression depends on the full training workload, including its runtime overhead.

## References

Aditya Agrawal, Albert Magyar, Hiteshwar Eswaraiah, Patrick Sheridan, Pradeep Janedula, Ravi Krishnan Venkatesan, Krishna Nair, and Ravi Iyer. Single-stage huffman encoder for ml compression, 2026.

Aaron Blakeman et al. Nemotron-H: A family of accurate and efficient hybrid mamba-transformer models, 2025.

Neil Burgess, Jelena Milanovic, Nigel Stephens, Konstantinos Monachopoulos, David Mansell, Jun Pei, and David Moreton. Bfloat16 processing for neural networks. In 2019 IEEE 26th Symposium on Computer Arithmetic (ARITH), 2019.

Jianfei Chen, Lianmin Zheng, Zhewei Yao, Dequan Wang, Ion Stoica, Michael W Mahoney, and Joseph E Gonzalez. Actnn: Reducing training memory footprint via 2-bit activation compressed training. In International Conference on Machine Learning, 2021.

Brian Chmiel, Maxim Fishman, Ron Banner, and Daniel Soudry. Fp4 all the way: Fully quantized training of large language models. In Advances in Neural Information Processing Systems, 2025.

Yann Collet. Lz4: Extremely fast compression algorithm, 2011.

Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization, 2022.

Jarek Duda. Asymmetric numeral systems: Entropy coding combining speed of huffman coding with compression rate of arithmetic coding, 2013.

Ruibo Fan, Xiangrui Yu, Xinglin Pan, Zeyu Li, Weile Luo, Qiang Wang, Wei Wang, and Xiaowen Chu. Zipserv: Fast and memory-efficient llm inference with hardware-aware lossless compression, 2026.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers, 2023.

Yipin Guo and Siddharth Joshi. Splitzip: Ultra fast lossless kv compression for disaggregated llm serving, 2026.

Moshik Hershcovitch, Andrew Wood, Leshem Choshen, Guy Girmonsky, Roy Leibovitz, Or Ozeri, Ilias Ennmouri, Michal Malka, Peter Chin, Swaminathan Sundararaman, and Danny Harnik. Zipnn: Lossless compression for ai models. In 2025 IEEE 18th International Conference on Cloud Computing (CLOUD), 2025.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

David A. Huffman. A method for the construction of minimum-redundancy codes. Proceedings of the IRE, 1952.

Animesh Jain, Amar Phanishayee, Jason Mars, Lingjia Tang, and Gennady Pekhimenko. Gist: Efficient data encoding for deep neural network training, 2018.

Zixuan Jiang, Jiaqi Gu, Mingjie Liu, Keren Zhu, and David Z. Pan. Optimizer fusion: Efficient training with better locality and parallelism, 2021.

Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, @fernbear.bsky.social, Boza Vlado, You Jiacheng, Franz Cesista, Braden Koszarsky, and @Grad62304977. modded-nanogpt: Speedrunning the nanogpt baseline, 2024a.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024b.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Lawrence L. Larmore and Daniel S. Hirschberg. A fast algorithm for optimal length-limited huffman codes. J. ACM, 1990.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration, 2024.

Wenxiang Lin, Xinglin Pan, Ruibo Fan, Shaohuai Shi, and Xiaowen Chu. Zipccl: Efficient lossless data compression of communication collectives for accelerating llm training, 2026.

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. Fp8 formats for deep learning, 2022.

NVIDIA. Floating-point computation. URL https://docs.nvidia.com/cuda/cuda-programming-guide/ 05-appendices/mathematical-functions.html.

NVIDIA Corporation. NVIDIA nvCOMP, 2026.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library, 2019.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Vonˇ Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

Qwen-Team. Qwen3 technical report, 2025.

Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. Zero-infinity: Breaking the gpu memory wall for extreme scale deep learning, 2021.

Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. Zero-offload: Democratizing billion-scale model training, 2021.

Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, and Stephen W. Keckler. vdnn: Virtualized deep neural networks for scalable, memory-efficient neural network design, 2016.

Yara Shamshoum, Nitzan Hodos, Yuval Sieradzki, and Assaf Schuster. CompAct: Compressed activations for memory efficient LLM training. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025.

Jiannan Tian, Cody Rivera, Sheng Di, Jieyang Chen, Xin Liang, Dingwen Tao, and Franck Cappello. Revisiting huffman coding: Toward extreme performance on modern gpu architectures. In 2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2021.

Philippe Tillet, Hsiang-Tsung Kung, and David D. Cox. Triton: an intermediate language and compiler for tiled neural network computations. Proceedings ofthe 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019.

Jeffrey Scott Vitter. Design and analysis of dynamic huffman codes. Journal of the ACM, 1987.

André Weißenberger and Bertil Schmidt. Massively parallel ans decoding on gpus, 2019.

Ian H. Witten, Radford M. Neal, and John G. Cleary. Arithmetic coding for data compression. Communications of the ACM, 1987.

Naoya Yamamoto, Koji Nakano, Yasuaki Ito, Daisuke Takafuji, Akihiko Kasagi, and Tsuguchika Tabaru. Huffman coding with gap arrays for gpu acceleration. In Proceedings of the 49th International Conference on Parallel Processing, 2020.

Tianyi Zhang, Mohsen Hariri, Shaochen Zhong, Vipin Chaudhary, Yang Sui, Xia Hu, and Anshumali Shrivastava. 70% size, 100% accuracy: Lossless LLM compression for efficient GPU inference via dynamic-length float (DFloat11). In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

## A Experimental Details

The main experiments are conducted on an NVIDIA RTX 3090 using PyTorch 2.14 (Paszke et al., 2019) with Triton 3.8 (Tillet et al., 2019). The evaluation across GPU models in Table 8 used the described GPUs, with an older PyTorch 2.11 setup for compatibility with the older GPUs.

Training measurements. For the NanoGPT and Qwen timing and memory comparisons, we discard three warm-up optimizer steps and measure the following five steps. We report the mean step time and the largest peak memory across those five steps. Wall-clock timing is synchronized with CUDA before and after each step and includes batch preparation, forward and backward computation, and optimizer updates. Model loading and setup are outside the timed region; initial compilation, where enabled, is performed during warm-up. Validation and checkpoint saving are excluded from these measurements.

Peak memory is measured with torch.cuda.max\_memory\_allocated(), resetting the peak counter before each step. This includes persistent tensors, temporary allocations, and the full shared overflow buffer when used. Values are reported in MiB and measure allocated tensor memory rather than the CUDA allocator’s reserved memory.

## A.1 Baseline codec evaluation

We compare LCT, SplitZip, DFloat11, ZipNN, and nvCOMP with LZ4, Bitcomp, and ANS on the encoding and decoding task in Section 6.1. The input contains 2<sup>29</sup> BF16 elements (1 GiB), sampled on the GPU from N(0, 2<sup>2</sup>) in FP32 and then converted to BF16, using random seed 0.

Codec configuration. SplitZip, DFloat11, and ZipNN use the official implementations. SplitZip’s codebook is calibrated on a separate tensor of the same size and distribution. The calibration time is excluded from timing since it is intended to be done once before use. DFloat11 uses a native C encoder on the CPU and the upstream CUDA decoder. Its encoding time includes histogram and codebook construction, transfer of the input to the CPU, and transfer of the compressed buffers back to the GPU. ZipNN uses Huffman coding with eight CPU threads. The encoding time include the input transfer to the CPU, and decoding includes transfer of the reconstructed tensor to the GPU.

LCT uses the Gaussian model, with a shared overflow arena allocated before timing. The nvCOMP methods all use the default configuration. Timings include any setup and codec configuration done as part of the library.

Timing and storage. Each codec performs three warm-up encode–decode cycles, followed by 50 measurements of each operation. Encoding and decoding are timed separately with a wall-clock timer and CUDA synchronization at their boundaries. Input generation, codec initialization, and output cleanup are outside the timed regions; allocations and transfers performed within each codec call are included. The reported means therefore measure complete codec calls, including host overhead. Timing uncertainties are standard errors of the mean, computed as the sample standard deviation divided by 50. Compression ratios divide the original tensor size by the compressed representation size. Baseline sizes include the encoded payload and decoding metadata; LCT uses the storage charged to the compressed tensor, including its overflow allocation, rather than the full capacity of the shared arena.

Additional GPU evaluations. Table 8 reports the codec comparison using a 1 GiB Gaussian tensor on NVIDIA V100 and A40 and a 4 GiB Gaussian tensor on NVIDIA H100. LCT has the lowest encoding time among the reported methods on all three GPUs, while SplitZip has the lowest decoding time. LCT’s storage geometry was changed on the V100 and H100, using 32,768 elements per block rather than 65,536, with stream configurations changed accordingly to match the compression ratio. This allows for better parallelism on the more powerful GPUs. Our V100 setup does not support nvCOMP methods.

Scaling with input size. Figure 3 compares encoding and decoding throughput for inputs from 16 to 2048 MiB.   
LCT’s encoding throughput exceeds SplitZip’s from 64 MiB onward, reaching approximately 396 GiB/s at 2048 MiB.   
Their decoding throughputs are similar for larger inputs, both reaching approximately 385 GiB/s at 2048 MiB.

## A.2 Fused pointwise operator evaluation

Each pointwise operation was benchmarked. Each input contains 2<sup>29</sup> BF16 elements (1 GiB). Addition operands are sampled on the GPU from N(0, 2<sup>2</sup>). For multiplication, the second operand instead follows N(1, 0.01<sup>2</sup>) with seed 2, to keep the multiplier concentrated near one. Input and output compression use the Gaussian codebook, CLEAN storage budget, and no added zero inflation or bias. The compressed paths use preallocated OverflowBuffers.

Table 8: Encoding and decoding Gaussian tensors of 1 GiB on NVIDIA V100 and A40, and 4 GiB on NVIDIA H100. Times are mean ± standard error of the mean; lower is better. Dashes indicate unreported results. Bold entries mark the lowest mean for each operation on each GPU.
<table><tr><td rowspan="2">Method</td><td colspan="2">NVIDIA V100 (1 GiB)</td><td colspan="2">NVIDIA A40 (1 GiB)</td><td colspan="2">NVIDIA H100 (4 GiB)</td></tr><tr><td>Encode / ms</td><td>Decode / ms</td><td>Encode / ms</td><td> $\overline { { \mathrm { D e c o d e } / \mathrm { m s } } }$ </td><td> $\overline { { \mathrm { E n c o d e } / \mathrm { m s } } }$ </td><td> $\overline { { \mathrm { D e c o d e } / \mathrm { m s } } }$ </td></tr><tr><td>nvCOMP LZ4</td><td></td><td></td><td> $\overline { { 6 2 9 . 2 \pm 0 . 2 4 2 } }$ </td><td> $\overline { { 1 3 . 6 4 \pm 0 . 1 3 8 } }$ </td><td> $\overline { { 5 0 0 . 1 \pm 1 2 . 3 3 } }$ </td><td> $\overline { { 1 6 . 6 3 \pm 1 1 . 7 9 } }$ </td></tr><tr><td>nvCOMP Bitcomp</td><td></td><td></td><td> $4 6 . 4 3 \pm 0 . 1 3 4$ </td><td> $5 . 3 5 4 \pm 0 . 0 0 2$ </td><td> $2 5 . 7 4 \pm 4 . 9 9 7$ </td><td> $4 . 7 7 1 \pm 4 . 7 7 1$ </td></tr><tr><td>nvCOMP ANS</td><td></td><td></td><td> $6 1 . 7 5 \pm 0 . 0 0 8$ </td><td> $5 . 3 4 1 \pm 0 . 0 0 2$ </td><td> $3 4 . 6 \pm 1 . 4 0 8$ </td><td> $6 . 3 4 9 \pm 0 . 0 8 1$ </td></tr><tr><td>DFloat11</td><td> $5 7 9 8 \pm 1 9 6 . 0$ </td><td> $6 . 2 7 1 \pm 0 . 1 0 3$ </td><td> $2 8 4 3 0 \pm 2 . 2 4 5$ </td><td> $4 . 7 5 0 \pm 0 . 0 0 7$ </td><td> $1 7 5 5 0 \pm 2 0 8 . 1$ </td><td> $1 1 . 1 8 \pm 0 . 0 2 2$ </td></tr><tr><td> $\mathrm { Z i p N N }$ </td><td> $2 5 0 4 \pm 1 7 9 . 6$ </td><td> $4 7 0 . 2 \pm 4 2 . 7 8$ </td><td> $1 2 5 0 \pm 1 . 5 9 2$ </td><td> $5 2 9 . 5 \pm 1 . 0 5 3$ </td><td> $7 0 0 1 \pm 3 2 9 . 5$ </td><td> $1 2 9 0 \pm 3 0 . 6 $ </td></tr><tr><td> $\mathrm { S p l i t Z i p }$ </td><td> $5 . 8 7 3 \pm 0 . 5 8 6$ </td><td> $\mathbf { 2 . 4 6 1 \pm 0 . 0 7 2 }$ </td><td> $7 . 7 5 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 3 . 5 8 0 \pm 0 . 0 0 2 }$ </td><td> $9 . 0 3 9 \pm 0 . 0 0 8$ </td><td> $\mathbf { 2 . 6 5 2 \pm 0 . 0 0 9 }$ </td></tr><tr><td>LCT</td><td> $\mathbf { 3 . 3 1 7 \pm 0 . 0 7 3 }$ </td><td> $3 . 4 7 9 \pm 0 . 0 7 8$ </td><td> $\mathbf { 3 . 5 9 3 \pm 0 . 0 0 1 }$ </td><td> $3 . 9 2 2 \pm 0 . 0 0 2$ </td><td> $\mathbf { 3 . 7 4 8 \pm 0 . 0 0 6 }$ </td><td> $3 . 9 0 4 \pm 0 . 0 0 7$ </td></tr></table>

![](images/e279d38374c140a09e53e3158d00ca4085246f5ad913a1b37848e910eb51713a.jpg)  
Figure 3: Encoding (left) and decoding (right) throughput versus uncompressed input size. Throughput is the uncompressed size in GiB divided by the mean time in seconds. LZ4 and Cascaded are omitted.

Before timing, each variant is executed once for compilation and bitwise comparison against a dense reference. Scaledadd references are computed in FP32 and rounded to BF16. Each variant then receives three warm-up calls and 50 timed calls. We use time.perf\_counter, with CUDA synchronization immediately before and after each call. The order of the three variants is rotated and reversed across iterations to reduce execution-order bias. Input generation, initial compression, buffer construction, compilation, validation, and output release are outside the timed region.

## A.3 Training Modded NanoGPT

The base model architecture, data, hyperparameters, and training code were taken from the baseline of Phase 3 of the Modded NanoGPT project. The code was then modified to reduce memory consumption without changing behavior. BF16 precision was used for all activations, momentum and weights. Activation checkpointing was applied to the RMSNorm layers, and the final output head was computed in chunks. Optimizer fusion was also used (Jiang et al., 2021). These do not affect the training process or model outputs, but significantly reduce the memory consumption. This is done to ensure a well-optimized baseline for a fair comparison. The Muon optimizer with 12 Newton–Schulz iterations is used, along with a batch size of 256 accumulated over four microbatches of 64 sequences, each of length 1024 tokens. The timing comparison measures a complete optimizer step, including all four microbatches.

LCT is applied to all matrix weights, activations, and momentum buffers. 1-D biases are excluded. Custom functions which compress cached activations are used for the feed-forward layers, attention block, and embedding block. Each type of compressed cached activation is given in Table 9. Activations make up the majority of the storage; model and optimizer states take up 767.88 MiB.

In order for torch.compile to work, the code is carefully structured so critical parts are compiled, while graph breaks are used around LCT operations, since compiling LCT does not lead to speedups.

Table 9: Cached activations compressed in the Modded NanoGPT experiment. Sizes are the dense BF16 storage before compression for a batch of 64.
<table><tr><td>Activation</td><td>Count</td><td>Shape</td><td>Each / MiB</td><td>Total / MiB</td><td>Budget</td></tr><tr><td>Embedding output before RMSNorm</td><td></td><td>[64, 1024, 768]</td><td>96</td><td>96</td><td>CLEAN</td></tr><tr><td>Input to RMSNorm and QKV projections</td><td>12</td><td>[64, 1024, 768]</td><td>96</td><td>1152</td><td>CLEAN</td></tr><tr><td>Q and K before per-head RMSNorm</td><td>24</td><td>[64, 1024, 6, 128]</td><td>96</td><td>2304</td><td>CLEAN</td></tr><tr><td>FlashAttention Q/K/V/output caches</td><td>48</td><td>[64, 1024, 6, 128]</td><td>96</td><td>4608</td><td>CLEAN</td></tr><tr><td>FFN input before RMSNorm and first linear</td><td>12</td><td>[64, 1024, 768]</td><td>96</td><td>1152</td><td>CLEAN</td></tr><tr><td>FFN ReLU output before squaring</td><td>12</td><td>[64, 1024, 3072]</td><td>384</td><td>4608</td><td>SPARSE</td></tr><tr><td>Total</td><td colspan="2">109</td><td colspan="2"></td><td></td></tr></table>

Table 10: Effect of the shared OverflowBuffer on NanoGPT training with compressed activations, weights, and momentum buffers. Time is per optimizer step; peak PyTorch-allocated GPU memory includes the shared buffer when enabled.
<table><tr><td>Setup</td><td>Time / s</td><td>Peak memory / MiB</td></tr><tr><td>With buffer</td><td>9.136</td><td>11964.6</td></tr><tr><td>Without buffer</td><td>9.353</td><td>11912.9</td></tr></table>

Effect of the shared buffer. Table 10 compares NanoGPT training with and without the shared OverflowBuffer. Using the buffer reduces optimizer-step time from 9.353 to 9.136 s (2.3%), while increasing peak allocated memory by 51.7 MiB.

Table 11: Weight and momentum compression ratios at NanoGPT training checkpoints. Ratios are uncompressed size divided by compressed size. The dash denotes an unreported value. \* The first momentum step uses a compression distribution with a higher zero fraction.

<table><tr><td colspan="2">Compression ratio (×)</td></tr><tr><td>Step</td><td>Weights Momentum</td></tr><tr><td>0</td><td>1.2240</td></tr><tr><td>1</td><td>1.2240 1.4382* 1.2426</td></tr><tr><td>2</td><td>1.4382 1.4116</td></tr><tr><td>5</td><td>1.4382</td></tr><tr><td>10</td><td>1.4382 1.4043</td></tr><tr><td>300</td><td>1.4382 1.4332</td></tr><tr><td>900</td><td>1.4382 1.4379</td></tr><tr><td>1500</td><td>1.4382 1.4382</td></tr><tr><td>2100</td><td>1.4382 1.4382</td></tr><tr><td>2700</td><td>1.4382 1.4382</td></tr><tr><td>3350</td><td>1.4382 1.4382</td></tr></table>

Feed-forward weight and momentum distributions during training. Figure 4 shows how exponent distributions of feed-forward momentum buffers (left) and weights (right) change during NanoGPT training. Both plots cover the two weight matrices in each feed-forward block. Each curve averages normalized histograms with equal weight per tensor, rather than per element. Probabilities use a logarithmic scale, and exponents have the BF16 bias removed. Exact zeros are shown separately, with their mean fractions in the legends; normalization includes all elements. The early momentum distributions lose their initial zero mass, while the weight distributions shift toward larger exponents at later checkpoints.

![](images/059a7251f685eb986d9ba88b4f943a5ea10a2161b552b8c1103ddd701b1e9861.jpg)  
Step 8 (0% zero) Step 3300 (0% zero)

![](images/ad5f3d61f9f9482bc03691da5d8c4ad1807ad680ae9d21912551d17a1a9786b4.jpg)  
Step 1 (50% zero) Step 300 (0% zero)  
Step 4 (0% zero) Step 1500 (0% zero)  
Step 8 (0% zero) Step 3300 (0% zero)

Figure 4: NanoGPT feed-forward exponent distributions during training: Muon momentum (left) and weights (right) at steps 1, 4, 8, 300, 1500, and 3300. Weight exponents below −20 are omitted from view.

## A.4 Fine-tuning Qwen3-4B

The Qwen3-4B-Base (Qwen-Team, 2025) model is used. The peak memory and time per step are investigated here. Since our method is lossless, these results are applicable to other fine tuning setups, and can be used on top of methods such as LoRA (Hu et al., 2021).

As in the Modded NanoGPT project, we fuse the RMSNorm layers, chunk the output head, and use optimizer fusion to reduce memory consumption without loss to ensure the baseline is well optimized. The timing test uses FineWeb training inputs with a batch size of 1 and varies the sequence length from 500 to 3500 tokens. Each optimizer step processes one sequence, without gradient accumulation. The Muon optimizer is used for hidden matrices with 5 Newton–Schulz iterations.

## B Representation and Implementation Details

## B.1 Distribution models and table construction

The implementation provides a set of default codebook distributions, one for each tensor role. Tables are cached by distribution family and parameters, independently of stream geometry.

Model probabilities come from integrating the magnitude density over exponent intervals: $[ 2 ^ { q } , 2 ^ { q + 1 } )$ for an interior exponent index $q = e - 1 2 7$ , with the endpoint bins extending to zero and infinity. These probabilities guide code assignment rather than certify a match to the tensor’s empirical statistics. At compression time, up to 4096 stratified, jittered samples estimate a tensor-specific center c from the mean of $e _ { i } - 1 2 7$ , rounded and clamped to [−128, 127]; the center is stored with the tensor and reused during decoding.

The centered symbol mapping reserves table index zero for exponent byte zero. With $d = ( e - 1 2 7 - c )$ mod 256 and $z = ( - 1 2 7 - \stackrel { \cdot } { c } )$ mod 256,

$$
t ( e ; c ) = { \left\{ \begin{array} { l l } { 0 , } & { e = 0 , } \\ { d + 1 , } & { e \neq 0 { \mathrm { ~ a n d ~ } } d < z , } \\ { d , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{5}
$$

This is a permutation of the 256 exponent bytes. The reserved symbol covers both exact zeros and subnormals, whose distinction is preserved by the side byte.

To select an escape cutoff, symbols are sorted by model probability and each of the 256 candidates, including no escape, is optimized with Package-Merge under the prefix cap $L _ { \operatorname* { m a x } } \dot { = } \operatorname* { m i n } ( 1 0 , 8 ) = 8$ (Larmore and Hirschberg, 1990), then scored by Equation 1. Because the literal cost is constant per candidate, minimizing weighted prefix length also minimizes expected cost. Escapes matter under this cap: eight-bit prefixes force all 256 direct codewords to length eight by Kraft’s inequality, so grouping rare symbols under one escape leaf frees prefix space for frequent ones. Since the no-escape code is included, the selected model-expected exponent length is at most eight bits. The search is optimal within the class of canonical codes that use a single escape leaf, an eight-bit literal, and the prefix cap $L _ { \mathrm { m a x } } ;$ we prove this below. Algorithm 1 states the procedure.

```latex
Algorithm 1 Codebook generation for a distribution Q
Require: model ${ \mathcal { Q } } = \left( q _ { t } \right)$ over $n = 2 5 6$ exponent symbols, prefix cap $L _ { \mathrm { m a x } } = 8 ,$ literal width $b = 8$
Ensure: encode table $E ,$ decode table $D ,$ escape length
1: integrate $\mathcal { Q }$ over exponent bins and apply zero mass to obtain $\mathbf { p } = \left( p _ { t } \right)$
2: apply the centered-symbol permutation of Equation $^ { 5 , }$ reserving index 0 for exponent byte zero
3: sort so $p _ { 1 } \geq \cdots \geq p _ { n } ;$ let $\begin{array} { r } { \dot { S } _ { k } = \sum _ { j = n - k + 1 } ^ { n } \dot { p } _ { j } } \end{array}$ be the mass of the k rarest symbols
4: L ← PACKAGE-MERGE(p, L<sub>max</sub>), $\begin{array} { r } { c ^ { \star } \gets \sum _ { t } p _ { t } L _ { t } , k ^ { \star } \gets 0 } \end{array}$ {no-escape candidate}
5: $\mathbf { L } ^ { \star }  \mathbf { L }$
6: for $k = 1$ to $n - 1$ do
7: $\mathbf { w }  ( p _ { 1 } , \hdots , p _ { n - k } , S _ { k } )$ {replace the tail by one escape leaf}
8: L ← PACKAGE-MERGE $\left( \mathbf { w } , L _ { \mathrm { m a x } } \right)$
9: $\begin{array} { r } { c \gets \sum _ { t } w _ { t } L _ { t } + b S _ { k } } \end{array}$ {add literal cost for escaped mass}
10: if $c < c ^ { \star }$ then
11: $c ^ { \star } \gets c , k ^ { \star } \gets k , \mathbf { L } ^ { \star } \gets \mathbf { L }$
12: end if
13: end for
14: assign canonical codes using $\mathbf { L } ^ { \star }$ , including an escape leaf only if $k ^ { \star } > 0$
15: for each escaped symbol t do
16: code<sub>t</sub> ← (esc_code $\ll b )$ | index<sub>b</sub>(t), $\ell _ { t } \gets L _ { \mathrm { e s c } } ^ { \star } + b$
17: end for
18: build $E [ 2 5 6 ]$ from $\left( \mathrm { c o d e } _ { t } , \ell _ { t } \right)$ and $D [ 2 ^ { { L _ { \mathrm { m a x } } } } ]$ by expanding each prefix
19: return $\dot { \boldsymbol { E } } , \dot { \boldsymbol { D } } ,$ and $\dot { L } _ { \mathrm { e s c } } ^ { \star }$ (or 0 if $k ^ { \star } = 0 )$
```

Table 12: Implemented stream configurations. The final column is the fixed exponent payload per padded element, before overflow and metadata.
<table><tr><td>Configuration</td><td>B</td><td>S</td><td>Bits/stream</td><td>Bits/element</td></tr><tr><td>Sparse (ReLU activations)</td><td>65536</td><td>256</td><td>640</td><td>2.500</td></tr><tr><td>Clean, non-Gaussian</td><td>65536</td><td>256</td><td>800</td><td>3.125</td></tr><tr><td>Clean, Gaussian</td><td>65536</td><td>256</td><td>768</td><td>3.000</td></tr><tr><td>Medium</td><td>32768</td><td>128</td><td>608</td><td>4.750</td></tr><tr><td>High</td><td>32768</td><td>128</td><td>704</td><td>5.500</td></tr></table>

Optimality. Sort the symbol probabilities as $p _ { 1 } \geq \cdots \geq p _ { n }$ , with $n = 2 5 6$ . Two reductions pin down the search. First, one escape leaf suffices: given several, reassigning every escaped symbol to the shortest leaf ℓ preserves Kraft’s inequality, since $\begin{array} { r } { 2 ^ { - \ell } \leq \sum _ { i } 2 ^ { - \ell _ { i } } } \end{array}$ , and cannot raise the escape cost $\begin{array} { r } { \sum _ { i } m _ { i } \dot { \ell } _ { i } . } \end{array}$ , since $\ell _ { i } \geq \ell$ for all i. Second, the escaped set is a probability suffix: exchanging an escaped symbol i with a directly coded symbol $j$ of probability $p _ { j } \leq p _ { i }$ reduces the cost by $( p _ { i } - p _ { j } ) ( \ell _ { \mathrm { e s c } } + 8 - \ell _ { j } ) \bar { \geq } 0$ , because $\ell _ { \mathrm { e s c } } + 8 \geq 9$ while $\ell _ { j } \leq L _ { \mathrm { m a x } } = 8 .$ . An optimal code therefore escapes exactly the k least probable symbols for some $k ,$ and it suffices to test these probability-ranked cutoffs. For fixed k the cost is $\begin{array} { r } { \sum _ { t < n - k } p _ { t } \ell _ { t } + \dot { S } _ { k } \ell _ { \mathrm { e s c } } + 8 S _ { k } } \end{array}$ , which equals $\begin{array} { r } { \sum _ { t } w _ { t } L _ { t } + 8 S _ { k } } \end{array}$ for the weight vector $\mathbf { w } = ( p _ { 1 } , \dots , p _ { n - k } , S _ { k } )$ that treats the escape leaf as one further symbol of weight $S _ { k }$ . Package-Merge minimizes $\sum _ { t } w _ { t } L _ { t }$ over all length assignments satisfying Kraft’s inequality and the cap (Larmore and Hirschberg, 1990), so the candidate cost $c _ { k }$ is the minimum over all codes whose escape set is $\{ n - k + 1 , \ldots , n \}$ . The loop evaluates every $k \in \{ 0 , \ldots , n - 1 \}$ , the omitted case $k = n$ being dominated: the all-eight code is Kraft-feasible and costs 8, whereas escaping everything costs at least 9. Keeping the smallest $c _ { k }$ thus returns the optimum, with ties resolved to the smaller escape set by the strict comparison.

Canonical codes are assigned in code-length order. Direct symbols use their assigned code; escaped symbols use the escape prefix followed by their eight-bit table index, which identifies the centered exponent symbol rather than the full BF16 value. Encoding and decoding use precomputed lookup tables (1 KiB and 4 KiB) rather than tree traversal, and kernels apply the sampled tensor center when loading them, adapting to scale without repeating the cutoff search.

The noise-level selector chooses the configuration in Table 12; it does not inject noise or change precision. If a symbol pair exceeds the stream budget, the overflow tail starts at that pair and all subsequent exponents are retained as raw bytes, with a start marker of 255 for streams without overflow.

## B.2 Overflow allocation and memory accounting

Kernels count overflowing streams and total tail length, compact their metadata with local prefix sums and atomic reservations, and gather raw exponent tails. Each affected stream records a 32-bit stream identifier, a 32-bit tail offset, and an eight-bit start step; raw tails store e − 127 modulo 256 in one byte per exponent. Let $N _ { p }$ be the padded element count, R the total stream count, and W the fixed number of 32-bit words per stream. If K streams overflow and T tail exponents are stored, the principal storage terms in bytes are

$$
M _ { \mathrm { c o r e } } = N _ { p } + 4 R W + T + 9 K\tag{6}
$$

Allocation additionally includes guard words, descriptors, alignment, cached codebooks, and transient workspaces. The nominal full-block rates follow from Equation 4; padding and overflow cause the departure from those rates and can produce expansion for small or poorly matched tensors.

For shared overflow storage, each allocation needs a sufficiently large contiguous free region, so capacity must account for concurrently live states and fragmentation. Per-tensor memory\_size() charges the tensor’s aligned region; processlevel reporting must instead count the entire shared arena once. The optional allow\_raw check returns dense storage when the fixed payload and padding alone exceed the input size; it does not guarantee against expansion after overflow. Appendix C.4 describes allocation and storage lifetimes. Keeping an unneeded dense tensor alongside its compressed representation retains both allocations and reduces the storage benefit.

## B.3 Feed-forward autograd implementation

Listing 1 shows a custom PyTorch autograd function for a feed-forward layer. The compressed-tensor wrapper stores the input and post-ReLU activation using separate distribution descriptors; the buffer provides an optional shared overflow arena. Enabling compression changes how activations are saved and reconstructed for backward. With compression disabled, the same function retains dense activations. The final two None entries in the backward return correspond to the buffer and compression flag.

## B.4 Comparison with SplitZip

SplitZip (Guo and Joshi, 2026) is a GPU-native framework for lossless KV-cache compression. Like LCT, it compresses BF16 exponents using a codebook prepared before use. SplitZip assigns 4-bit codes to the 16 most common exponent values. Less common exponents are stored verbatim with a 16-bit integer marking their position. Its average storage per value, excluding other metadata and padding, is therefore

$$
b _ { \mathrm { S p l i t Z i p } } = 8 + 4 + 2 4 \epsilon = 1 2 + 2 4 \epsilon \quad \mathrm { b i t s / v a l u e } ,\tag{7}
$$

where the terms account for the sign–fraction byte, the 4-bit code, and overflow with probability ϵ. With no overflow, the compression ratio is $1 6 / 1 2 = \overline { { 4 } } / 3 \approx 1 . 3 3$ . LCT achieves ratios above 1.4 by assigning variable-length codes and optimizing which symbols use escapes. SplitZip stores a 16-bit position for each overflow value; LCT instead uses an escape prefix of at most eight bits followed by an eight-bit literal.

LCT also adapts its codebook through runtime exponent centering. SplitZip does not perform this adjustment, so changes in tensor scale can increase overflow under a fixed codebook. Adaptation is particularly useful during training, when tensor magnitudes evolve. In the evaluated SplitZip implementation, encoding synchronizes with the host to determine the overflow-buffer size, adding to encoding latency.

## C Kernels

The Triton kernels implement the codec described in Section 4.4. This section details their memory layout, bit packing, and allocation optimizations.

## C.1 Stream parallelism and table preparation

Each lane retains its stream’s bit position and partial words in registers. Payload words are stored with the stream index varying fastest, placing neighboring lanes’ accesses to the same word position next to each other in memory.

Listing 1: Feed-forward autograd function with optional lossless storage of saved activations.

```python
class FFN(Function):
"""Dense baseline autograd FFN for comparison.
For ‘‘x[B, D]‘‘, ‘‘W1[H, D]‘‘, and ‘‘W2[D, H]‘‘ computes
‘‘z = relu(x @ W1.T)‘‘ and ‘‘output = z @ W2.T‘‘.

@staticmethod
def forward(ctx, x, W1, W2,
buffer: TensorBuffer | None = None,
compressed: bool = False):
"""Run the dense FFN forward pass and save tensors
for backward.

z = x @ W1.T
h = z.relu_()
output = h @ W2.T
if compressed:
x = MyCompressed(x, buffer=buffer, dist=act_dist)
h = MyCompressed(h, buffer=buffer, dist=act_relu_dist)
ctx.save_for_backward(x, W1, W2, h)
ctx.compressed = compressed
return output
@staticmethod
def backward(ctx, grad_output):
"""Compute dense FFN gradients from ‘‘grad_output[B, D]‘‘."""
needs_x = ctx.needs_input_grad[0]
x, W1, W2, h = ctx.saved_tensors
if ctx.compressed:
x = x.decompress_free()
h = h.decompress_free()
grad_z = grad_output @ W2
grad_W2 = grad_output.T @ h
grad_preact = torch.ops.aten.threshold_backward.grad_input(
grad_z, h, 0, grad_input=grad_z)
del h, grad_z
if not torch.compiler.is_compiling():
ctx.maybe_clear_saved_tensors()
if needs_x:
grad_x = grad_preact @ W1
else:
grad_x = None
grad_W1 = grad_preact.T @ x
return grad_x, grad_W1, grad_W2, None, None
```

Center estimation takes one deterministically jittered sample per input region. Integer hashing generates the offsets without a random-number buffer and reduces repeated sampling of the same matrix columns. Shifting the lookup tables once per tensor lets the encoder index raw exponent bytes and the decoder recover uncentered exponents directly, removing the centering transformation from the per-symbol loop.

## C.2 Encoding and decoding

Bit representation and invertibility. Let $u _ { i }$ be the unsigned 16-bit pattern of element i. Its exponent byte $e _ { i }$ and side byte $r _ { i }$ are

$$
e _ { i } = ( u _ { i } \gg 7 ) \& \ 2 5 5 ,\tag{8}
$$

$$
r _ { i } = ( u _ { i } \& 1 2 7 ) \mid ( ( u _ { i } \gg 8 ) \& 1 2 8 ) .\tag{9}
$$

The exponent is coded and $r _ { i }$ is stored verbatim. Reconstruction,

$$
u _ { i } = ( e _ { i } \ll 7 ) \mid ( r _ { i } \& 1 2 7 ) \mid ( ( r _ { i } \& 1 2 8 ) \ll 8 ) ,\tag{10}
$$

recovers the original 16-bit pattern, including signed zeros, subnormals, infinities, and NaN payloads, without floatingpoint conversion.

Packing and lookup. The encoder processes two symbols per iteration. Each table entry stores the code bits and length; since a complete code is at most 16 bits, a pair occupies at most 32 bits and crosses at most one output-word boundary. Partial words remain in registers until written.

The decoder uses a 64-bit window spanning two payload words to decode a pair of symbols. Direct codes return an exponent and consumed length; escapes extract an eight-bit literal from the same window. In the full-block path, the next payload word is loaded only when a lane crosses a word boundary. Decode-table loads retain cache entries, side bytes use global-cache loads, and dense outputs use streaming stores. Loop unrolling and autotuned warp counts, pipeline depths, and register limits control instruction scheduling and register pressure.

## C.3 Overflow compaction and allocation

Compaction uses local prefix sums to assign positions within each tile and atomic additions to reserve ranges across tiles. Reservations occur per tile rather than per symbol. Separate gather and scatter kernels access only the raw tails, keeping variable-length overflow handling outside the main packing and decoding loops.

## C.4 Overflow Buffer

The allocator tracks occupied address ranges, uses first-fit search to reserve aligned regions, and writes device descriptors for subsequent kernels. Freeing a region marks it available; coalescing adjacent free regions reduces fragmentation. During an update, old and replacement allocations may coexist. On the same CUDA stream, the old region can be released after its final consumer is launched, preserving execution order.

## C.5 Fused pointwise operations

Table 13 summarizes the supported operations. Operator and output-format choices are resolved at compile time, eliminating branches from the inner loop. For a compressed and a dense operand, a correction kernel handles raw overflow tails.

For two compressed operands, preparation maps stream identifiers to overflow offsets so each lane can select coded or raw exponents without searching compact metadata. Both operands must share stream geometry, escape length, and overflow-storage mode. For compressed output, sign–fraction bytes are written directly to their final locations, leaving only exponent bytes for subsequent centering and encoding.

## D Derivations for Compression Analysis

## D.1 Cross entropy of the implemented code

The implemented code lengths differ from ideal model lengths. Let $\ell _ { \mathcal { Q } } ( t )$ include escape prefixes and literal bits, and define

$$
Z = \sum _ { t } 2 ^ { - \ell _ { Q } ( t ) } \leq 1 , \qquad \widetilde { q } _ { t } = \frac { 2 ^ { - \ell _ { Q } ( t ) } } { Z } ,\tag{11}
$$

Table 13: LCT operators and training integration points. $C$ and D denote compressed and dense operands; α is a device scalar. Fused pointwise operators support dense or compressed output. Matrix products use dense GEMM after reconstruction.
<table><tr><td>Operator</td><td>Computation</td><td>Training role</td></tr><tr><td>compress</td><td> $\overline { { D  C } }$ </td><td>Retain BF16 weights or state between uses; store activations within an explicitly managed backward integration.</td></tr><tr><td>decompress</td><td> $C  D$ </td><td>Restore state or saved tensors for a dense consumer.</td></tr><tr><td>compA_add_B</td><td> $C + D$ </td><td>Accumulate microbatch gradients or apply a prepared parameter increment.</td></tr><tr><td>compA_mul_B</td><td> $C \odot D$ </td><td>Apply an existing elementwise mask, gate, or scale.</td></tr><tr><td>a_compA_add_B</td><td> $\alpha C + D$ </td><td>Update momentum or moment state; combine weight decay with a prepared dense step.</td></tr><tr><td>a_compA_add_compB</td><td> $\alpha C _ { 1 } + C _ { 2 }$ </td><td>Apply a compressed update to compressed state;  $W - \eta U$  uses U first and W second.</td></tr><tr><td>A_compBT</td><td> $D C ^ { \top }$ </td><td>Linear-layer forward product  $X W ^ { \top }$  with compressed weights  $W .$ </td></tr><tr><td> $\mathtt { A \_ c o m p B }$ </td><td> $_ { D C }$ </td><td>Input-gradient product  $G _ { Y } W$  with upstream gradient  $G _ { Y } .$ </td></tr></table>

where the inequality follows from the prefix-code property. Direct substitution yields the exact identity

$$
\begin{array} { r } { \bar { \ell } = H ( \mathcal { P } ) + D _ { \mathrm { K L } } ( \mathcal { P } \| \widetilde { \mathcal { Q } } ) - \log _ { 2 } Z . } \end{array}\tag{12}
$$

This separates source entropy, mismatch to the code-induced distribution, and unused code space, and it applies directly to $\mathrm { L C T ^ { \circ } s }$ length-limited escape construction. Every symbol retains a finite representation, including those assigned negligible probability by the analytic model.

## D.2 Exponent scaling and boundary effects

Proof of Proposition 1: shift-corrected exponents. Let X take finite, nonzero BF16 values and $a \neq 0$ be a finite constant. Define the ideal exponents $E = \mathsf { \bar { \rho } } _ { \lfloor \log _ { 2 } \vert X \vert \rfloor , E _ { a } } = \lfloor \log _ { 2 } \vert a X \vert \rfloor$ , and the common shift $\dot { k } = \lfloor \log _ { 2 } \vert a \vert \rfloor$ Excluding rounding and finite-range effects, the claim is

$$
( E _ { a } - k ) - E \in \{ 0 , 1 \} .\tag{13}
$$

Write $\log _ { 2 } { \left| X \right| } = E + U$ and $\log _ { 2 } | a | = k + \delta ,$ where $0 \leq U , \delta < 1$ . Then

$$
\left\lfloor \log _ { 2 } \left| a X \right| \right\rfloor = E + k + \left\lfloor U + \delta \right\rfloor = E + k + J , \qquad J = \mathbf { 1 } \{ U \geq 1 - \delta \} .\tag{14}
$$

The indicator is zero when $\delta \ = \ 0 ,$ , including values exactly on a power-of-two boundary. More generally, let $p _ { j } = \mathrm { P r } ( E = j )$ and $q _ { j } = \operatorname* { P r } ( E = j , J = 1 )$ ). The exact redistribution is

$$
\operatorname* { P r } ( E _ { a } = m ) = p _ { m - k } - q _ { m - k } + q _ { m - k - 1 } .\tag{15}
$$

No independence or uniformity of the fractional log magnitude $U$ is assumed: the mass crossing a boundary can depend on the original bin. Given $E ,$ there are at most two possible values of $E _ { a } ;$ given $E _ { a } ,$ , there are at most two possible values of E. Hence both conditional entropies are at most one bit, and the chain rule gives

$$
H ( E _ { a } ) - H ( E ) = H ( E _ { a } \mid E ) - H ( E \mid E _ { a } ) , \qquad | H ( E _ { a } ) - H ( E ) | \leq 1 .\tag{16}
$$

For $\delta = 0$ , integer translation is a bijection, proving entropy invariance.

Proof of the BF16 boundary rule. For a finite real $y ,$ let $R ( y )$ be the raw exponent byte after a single BF16 rounding to nearest, ties to even, with gradual underflow. The exact rule, including rounding, underflow, and overflow boundaries, is

$$
R ( y ) = \sum _ { r = 1 } ^ { 2 5 5 } \mathbf { 1 } \{ | y | \geq \tau _ { r } \} , \qquad \tau _ { r } = \left\{ 2 ^ { - 1 2 6 } ( 1 - 2 ^ { - 8 } ) , \quad r = 1 , \right. \qquad\tag{17}
$$

Scaling replaces each threshold by $\tau _ { r } / | a | .$

BF16 has seven fraction bits, minimum normal magnitude $2 ^ { - 1 2 6 }$ , and subnormal spacing $2 ^ { - 1 3 3 }$ (Burgess et al., 2019; NVIDIA). For an interior normal power $2 ^ { j } , - 1 2 5 \le j \le 1 2 7$ , the immediately preceding value is $\bar { 2 ^ { j } } - 2 ^ { j - 8 }$ . Their rounding midpoint is therefore $2 ^ { j } - 2 ^ { j - 9 }$ . The midpoint below the minimum normal value is instead $2 ^ { - 1 2 6 } - 2 ^ { - 1 3 4 }$ because the subnormal spacing is constant. At each midpoint the power of two has the even significand, so ties round into the higher exponent bin. At the upper end, the maximum finite value is $2 ^ { 1 2 8 } - 2 ^ { 1 2 0 }$ ; rounding overflows at $2 ^ { 1 2 8 } - 2 ^ { 1 1  5 }$ , including the tie. These are exactly the thresholds in Equation 17. Counting the thresholds crossed gives the raw exponent byte, proving the rule.

The interval below the first threshold contains both subnormals and zero; magnitudes at most $2 ^ { - 1 3 4 }$ round to zero, including the tie. For a continuous magnitude density $g$ and $a \neq 0$ , the normal-byte probabilities are consequently

$$
\operatorname* { P r } ( R ( a X ) = r ) = \int _ { \tau _ { r } / | a | } ^ { \tau _ { r + 1 } / | a | } g ( x ) d x , \qquad 1 \leq r \leq 2 5 4 .\tag{18}
$$

Byte zero collects the interval $[ 0 , \tau _ { 1 } / | a | )$ , and byte 255 collects $[ \tau _ { 2 5 5 } / | a | , \infty )$ . For discrete inputs or point masses, these are half-open probability intervals with the same endpoint convention, rather than density integrals. An exact zero remains zero under finite nonzero scaling. A zero multiplier sends all finite inputs to exponent byte zero; NaNs and infinite inputs are outside the propositions. Flush-to-zero arithmetic or an intermediate rounding step requires its own boundary rule. LCT itself preserves the resulting BF16 bits; the rounding here belongs to the arithmetic producing them.

Proof of Proposition 2: boundary entropy correction. For the proposition’s nonzero input, set $A = R ( X )$ and $B = R ( a X )$ . Let $D$ be the event that $A \bar { \neq } \bar { E } + 1 2 7$ or $B \neq E _ { a } + \bar { 1 2 7 }$ , and write $\varepsilon = \mathrm { P r } ( D )$ . This event captures departures from the ideal exponent labels due to rounding and finite range. On $D ^ { c } .$ , the same two-bin argument gives $| \hat { H ( B \mid D = 0 ) } - H ( A \mid D ^ { \cdot } = 0 ) | \leq 1$ . On $D ,$ each byte has only 256 possible values, so the corresponding difference is at most eight bits. Expanding entropy by the binary event D yields

$$
| H ( B ) - H ( A ) | \leq \operatorname* { m i n } \{ 8 , 1 + 7 \varepsilon + h _ { 2 } ( \varepsilon ) \} ,\tag{19}
$$

where $h _ { 2 }$ is binary entropy. Indeed, the weighted conditional differences contribute at most $( 1 - \varepsilon ) + 8 \varepsilon$ , and the remaining term $I ( \mathbf { \dot { B } } ; D ) \dot { - } I ( A ; D )$ has absolute value at most $H ( D ) = h _ { 2 } ( \varepsilon )$ . Dropping the trivial eight-bit cap gives the bound in Proposition 2; its correction vanishes as $\varepsilon \to 0$ . If scaling is by a power of two, the conditional difference on $D ^ { c }$ is zero and the bound improves to min $\{ 8 , 8 \varepsilon + h _ { 2 } ( \varepsilon ) \}$

For actual normal BF16 inputs, power-of-two scaling changes only the exponent when the output stays normal and finite, and is exactly representable. Reusing the same centered symbols also requires the estimated center to shift by that integer. Sampling, the half-integer rounding convention, center clamping, and reserved-zero remapping can affect this correspondence. Thus neither the proposition nor its entropy bound implies an unchanged code length for every implementation configuration.

## D.3 Exponent distributions from continuous densities

This section studies the distribution of exponents for data drawn from a particular data distribution.

Continuous change of parameterisation. We first derive the formula for the distribution of exponents. Let $X > 0$ have density $f _ { X }$ and first use natural logarithms, $e = \ln X$ . Since $X = \exp ( e )$ and $d x / d e = \exp ( e )$ , change of variables gives

$$
f _ { e } ( t ) = \exp ( t ) f _ { X } ( \exp ( t ) ) .\tag{20}
$$

For signed tensor values, take the magnitude $M = | X |$ |. For a continuous signed density, $g ( x ) = f _ { X } ( x ) + f _ { X } ( - x )$ for $x > 0$ is its magnitude density. Using base two, $V = \log _ { 2 } M$ , gives

$$
f _ { V } ( v ) = ( \ln 2 ) 2 ^ { v } g ( 2 ^ { v } ) , \qquad \operatorname* { P r } ( \lfloor V \rfloor = j ) = \int _ { j } ^ { j + 1 } f _ { V } ( v ) d v = \int _ { 2 ^ { j } } ^ { 2 ^ { j + 1 } } g ( x ) d x .\tag{21}
$$

The Jacobian factor $2 ^ { v }$ ln 2 accounts for the increasing width of value intervals. Equation 21 gives continuous exponentbin masses while Equation 18 gives the finite-format correction. The entropy of the discrete masses determines the entropy coding cost. Scaling translates the continuous density exactly, $f _ { V _ { a } } ( \stackrel { \cdot } { v } ) ^ { \cdot } = f _ { V } ( v - \log _ { 2 } | a | )$

Why observed exponent distributions are concentrated and asymmetric. Next, we discuss the shapes of exponent distributions.

Lemma 1 (Left-tail decay of log magnitudes) Let $M > 0$ have density g with a power-law upper bound near zero: $g ( x ) = O ( x ^ { \alpha - 1 } )$ as $x \downarrow 0$ for some $\alpha > 0$ . For $V = \log _ { 2 } M$

$$
f _ { V } ( v ) = O ( 2 ^ { \alpha v } ) \quad ( v  - \infty ) , \qquad \operatorname* { P r } ( \lfloor V \rfloor = j ) = O ( 2 ^ { \alpha j } ) \quad ( j  - \infty ) .\tag{22}
$$

Both the density and the exponent-bin probabilities therefore decay at least exponentially to the left.

Proof. There are constants $C , x _ { 0 } > 0$ such that $g ( x ) \leq C x ^ { \alpha - 1 }$ for $0 < x < x _ { 0 }$ . Equation 21 then gives $f _ { V } ( v ) \leq$ C ln $2 2 ^ { \alpha v }$ whenever $2 ^ { v } < x _ { 0 }$ . For $2 ^ { j + 1 } < x _ { 0 }$ , integrating the same bound gives

$$
\operatorname* { P r } ( \lfloor V \rfloor = j ) \leq C \int _ { 2 ^ { j } } ^ { 2 ^ { j + 1 } } x ^ { \alpha - 1 } d x = { \frac { C ( 2 ^ { \alpha } - 1 ) } { \alpha } } 2 ^ { \alpha j } .\tag{23}
$$

These inequalities prove both bounds.

Any density bounded near zero satisfies the lemma with $\alpha = 1$ , giving $f _ { V } ( v ) = { \cal O } ( 2 ^ { v } )$ . The lemma also covers densities that diverge at zero when they satisfy the stated bound for some $\alpha > 0$ . BF16 groups the smallest magnitudes into exponent byte zero as subnormal numbers or zeros.

Lemma 2 (Right-tail decay of log magnitudes) Let $M > 0$ have density g satisfying $g ( x ) = O ( x ^ { - 1 - \beta } )$ as $x \to \infty$ for some $\beta > 0$ . For $V = \dot { \log _ { 2 } { M } }$

$$
f _ { V } ( v ) = O ( 2 ^ { - \beta v } ) \quad ( v  \infty ) , \qquad \operatorname* { P r } ( \lfloor V \rfloor = j ) = O ( 2 ^ { - \beta j } ) \quad ( j  \infty ) .\tag{24}
$$

Both the density and the exponent-bin probabilities therefore decay at least exponentially to the right.

Proof. There are constants $C , x _ { 0 } > 0$ such that $g ( x ) \leq C x ^ { - 1 - \beta }$ for $x > x _ { 0 }$ . Equation 21 gives $f _ { V } ( v ) \leq C \ln { 2 } 2 ^ { - \beta v }$ whenever $2 ^ { v } > x _ { 0 }$ . For $2 ^ { j } > x _ { 0 }$

$$
\operatorname* { P r } ( \lfloor V \rfloor = j ) \le C \int _ { 2 ^ { j } } ^ { 2 ^ { j + 1 } } x ^ { - 1 - \beta } d x = { \frac { C ( 1 - 2 ^ { - \beta } ) } { \beta } } 2 ^ { - \beta j } .\tag{25}
$$

These inequalities prove both bounds.

A magnitude density satisfying the stated power-law bound therefore has an exponentially decaying right tail in its exponent distribution. In practice, BF16 values that overflow share exponent byte 255.

An example of these effects can be seen in the Gaussian $X \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ ,

$$
f _ { V } ( v ) = \frac { \sqrt { 2 / \pi } \ln 2 } { \sigma } 2 ^ { v } \exp \left( - \frac { 2 ^ { 2 v } } { 2 \sigma ^ { 2 } } \right) .\tag{26}
$$

The peak is at $\begin{array} { r } { \boldsymbol { v } \ = \ \log _ { 2 } \sigma } \end{array}$ with an exponentially decaying left tail and super-exponentially decaying right tail.   
Furthermore, changing $\sigma$ shifts the log-density without changing its shape.

## D.4 Calculation of theoretical coding and storage costs

We use the Gaussian, Laplace, power-law, and Gamma models compared in Table $^ { 2 , }$ with population exponent centering and no added zero mass. Exponent probabilities are analytically calculated.

The left panel of Table 2 gives the ideal cross entropy $\begin{array} { r } { H ( \mathcal { P } , \mathcal { Q } ) = - \sum _ { t } p _ { t } \log _ { 2 } q _ { t } } \end{array}$ . The right panel gives predicted allocated exponent storage under the CLEAN configuration. The code lengths $\ell _ { \mathcal { Q } } ( t )$ include escape literals, with mean $\begin{array} { r } { \bar { \ell } = \sum _ { t } p _ { t } \ell _ { \mathcal { Q } } ( t ) } \end{array}$ . Allocated storage also depends on the fixed stream budget and overflow: raw tails use one byte per exponent, with nine metadata bytes per overflowing stream, as in Equation 4. The calculation assumes independent symbols and full blocks, and excludes padding and allocation overhead. Neither panel includes the unchanged sign–fraction byte.

Effect of tail mismatch. The source and coding models can have similar central mass but very different tails. The power-law source places substantially more probability on large magnitudes than the Gaussian model, whose tail decays exponentially in the squared magnitude. Consequently, large exponent symbols can have very small model probabilities $q _ { t } ,$ making their ideal costs $- \log _ { 2 } q _ { t }$ extremely large. These terms dominate the power-law-to-Gaussian cross entropy, producing the large entry in Table 2; it is a mismatch penalty, not high source entropy. LCT limits each exponent code to at most 16 bits, including the escape literal, and stores overflowing tails as raw bytes. Its storage cost is therefore less sensitive to this mismatch. This does not violate the entropy bound: the comparison is with a mismatched ideal coder, not an optimal coder for the source.

## D.5 Entropy of exponent and sign–fraction byte

For typical floating-point data, almost all the compressibility is in the exponent. Let BF16 samples from a distribution be represented as $\bar { U } = ( E , R )$ , where E is the exponent byte and R contains the sign and seven fraction bits. Exponent only coding stores R verbatim and compresses $\dot { E }$ independently of $R ,$ at a cost of ${ \bar { H } } ( E ) + 8$ bits per value, while ideal

full-value coding costs $H ( E , R )$ . The additional saving is

$$
\Delta b = H ( E ) + 8 - H ( E , R ) = 8 - H ( R \mid E ) .\tag{27}
$$

For a symmetric continuous distribution rounded to BF16, the sign bit is uniform and independent of the magnitude. When the fraction bits also have high conditional entropy, $H ( R \bar { \mid } E )$ is close to eight bits and the additional saving is small. For example, for the Gaussian distribution $\ddot { X } \sim \dot { \mathcal { N } } ( \dot { 0 } , 1 \dot { ) }$ rounded to BF16, the conditional entropy is $H ( R \mid E ) \approx 7 . 9 1 9 5$ bits. Coding the full value therefore saves only $\Delta { \dot { b } } \approx 0 . 0 8 0 5$ bits per value beyond exponent-only coding.

The sign–fraction byte can have lower entropy in some cases. One example is nonnegative distributions whose sign bit is always zero. This is the case for certain activation functions such as ReLU and sigmoid. Another case is if the mean of the distribution is large and all values cluster around a single exponent, $\mathrm { e . g . \mathcal { N } ( 1 \bar { 2 } , 1 ) }$ where values are mostly distributed between $2 ^ { 3 }$ and $2 ^ { \overline { { 4 } } }$ so there is no pooling of mantissa values (and exponents have very low entropy). LCT does not take advantage of a low-entropy sign–fraction byte if this is the case.

## E Nemotron Weight and Activation Distributions

## E.1 Weight exponent distributions

Figure 5 shows BF16 exponent distributions for six Nemotron-H-8B parameter groups, each containing more than 500M values. Histograms include all weights and are pooled across layers. Most probability mass lies near exponents −7 and −8, supporting exponent-only compression.

## E.2 Activation exponent distributions

We collect activations in BF16 evaluation mode on four 1024-token sequences from FineWeb validation data, tokenized for Nemotron. Figure 6 pools counts across layers for the five groups with more than 500M observations. Exponents remain concentrated, although their spread varies by activation type. FFN hidden activations are particularly compressible, with 93.23% exact zeros.

(a) Embedding weights  
![](images/f1ca57548193090f4bac5dc3d5a004763341ab666e2e38fb2e8961322003dcc1.jpg)

(b) FFN input weights  
![](images/b9eb73f16529ca9c955d76fd74ec523e140b055c5395315660ae856ae4a5525e.jpg)

(c) FFN output weights  
![](images/f08e9d47e2226678e7320a899fa85f489af0469a3cc005367c91fcbad8b8105e.jpg)  
(e) Mamba output weights

(d) Mamba input weights  
![](images/ad57ec39bac3768dd62a1dab9fdbe4e63f9d779e3d8c0df80420bd7914967164.jpg)

![](images/1a7ac5f29474c287905fe8ade7977178ef0c22bb88e8e94353f19273be7937b5.jpg)  
Exponent (bias removed)

(f) LM head weights  
![](images/1ec4ad398e48c4f341fa8862579a8aaf39871630b7c502b5ac01358dacb30b81.jpg)  
Exponent (bias removed)  
Figure 5: Nemotron-H-8B weight exponent probabilities, with the BF16 bias removed. The hatched bar aggregates exponents below −20. Probabilities are normalized over all weights.

(a) FFN pre-activation  
![](images/8fa9701dfc027616b7b6b6e520d0d59db47a11c1a68b6c1ecfc9a9e7f5085c5c.jpg)

(b) FFN hidden  
![](images/7ae2111da6e9c9afe6f2db39a3bcc92d0b1bb1b21feb3ccaa950ad6956e07bfd.jpg)

(c) Mamba projected  
![](images/ca635cbe4e72651c52598dd07a5cb97f000c74f5f3b6788b5cb4b89f76c249df.jpg)

(d) Mamba hidden  
![](images/d0e91463542a4128fa896bb0d04f7ad2f493167c1a043e679d203f99be5eb881.jpg)

(e) Logits  
![](images/dc7bd05b67d421a38de75e487fb9a7f30d15382f62e824a81eee4a81afd1b1ed.jpg)  
Figure 6: Nemotron-H-8B activation exponent probabilities, with the BF16 bias removed. The hatched bar combines exponents below −20, exact zeros, and subnormals; the FFN hidden bar is dominated by exact zeros. Probabilities are normalized over all observations.