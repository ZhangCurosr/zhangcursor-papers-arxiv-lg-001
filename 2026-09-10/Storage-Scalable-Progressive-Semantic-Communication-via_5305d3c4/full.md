# Storage-Scalable Progressive Semantic Communication via

Knowledge-Base Reuse

Heng Zhu, Ye Liu, Kun Zhu, Member, IEEE, Feifei Song

Abstract—Existing knowledge-base-assisted semantic communication schemes commonly adopt either single knowledge-base quantization (SKBQ) or multi-knowledge-base residual quantization (MKBQ). SKBQ incurs limited storage overhead but has restricted quantization capacity, whereas MKBQ supports progressive refinement by assigning an independent knowledge base (KB) to each stage, causing the KB storage to grow linearly with the transmission depth. To address this problem, we propose storage-scalable knowledge-base reuse quantization (SSKBQ), which reuses a compact set of KBs across multiple residual refinement stages and thereby decouples the number of transmission stages from the number of maintained KBs. A stageaware residual supervision mechanism is further introduced to regularize intermediate quantized representations and encourage progressive refinement. Experimental results demonstrate that KB reuse provides an effective solution to the storage scalability problem while maintaining competitive progressive reconstruction performance.

Index Terms—Semantic communication, progressive image reconstruction, vector quantization, knowledge-base reuse, storage scalability.

## I. INTRODUCTION

EMANTIC communication has recently emerged as a S promising paradigm and has attracted extensive research interest [1]–[3]. Unlike conventional communication systems that pursue bit-level fidelity, semantic communication focuses on delivering task-relevant information. By removing taskirrelevant redundancy, it can significantly reduce transmission rates while preserving task performance.

Despite these advantages, early studies reveal a fundamental challenge: without carefully designed encoding and decoding mechanisms, semantic communication may even require higher transmission rates than conventional schemes. This issue largely stems from the use of deep neural networks for semantic feature extraction, where integer-valued inputs are transformed into high-dimensional floating-point representations. According to the IEEE 754 standard [4], a doubleprecision floating-point number typically occupies 64 bits, whereas an integer often requires only 8 bits. Consequently, directly transmitting semantic features requires substantial compression to maintain the same transmission cost, which is often impractical.

To alleviate this issue, quantization has been introduced into semantic encoding. A representative approach is SKBQ [5], where semantic features are mapped to the nearest codewords in a predefined KB. Instead of transmitting floatingpoint features, the corresponding integer indices are conveyed, thereby reducing transmission rates. However, the use of a single KB limits the representation capacity and may lead to considerable quantization distortion. Moreover, it typically supports only single-stage transmission and does not naturally support progressive refinement.

![](images/c30d10a5c7864b17135fdd00649165dab43e7dc379ead85b2042f4601647af95.jpg)  
Fig. 1: End-to-End semantic communication system for image reconstruction.

To enable progressive refinement, MKBQ schemes have been proposed [6]. These methods progressively quantize residual errors across multiple KBs, enabling progressive reconstruction through multi-stage transmission. However, the storage overhead grows linearly with the number of stages, since each stage requires an independent KB. When each KB is large, the cumulative storage cost becomes prohibitive.

To resolve this trade-off, we propose an SSKBQ scheme for progressive semantic communication. The key idea is to decouple transmission stages from dedicated KBs by reusing a compact set of KBs across multiple residual refinement stages, thereby enabling progressive reconstruction without linear KB storage growth. In addition, a stage-aware residual supervision mechanism is designed to regularize intermediate quantized representations and encourage progressive refinement across stages. Experimental results show that the proposed scheme provides a favorable trade-off between progressive reconstruction performance and KB storage overhead compared with SKBQ and MKBQ.

## II. SEMANTIC COMMUNICATION SYSTEM

In this work, we consider an end-to-end (E2E) semantic communication system for image reconstruction, as illustrated in Fig. 1. We focus on the semantic-layer design of storagescalable progressive transmission rather than physical-layer transmission or channel-robust modulation and coding. Let $\mathbf { I } \in \mathbb { R } ^ { 3 \times H \times W }$ denote an input image sampled from dataset $\mathcal { D } ,$ where the three dimensions represent the RGB channels, height, and width, respectively.

At the transmitter, the image is processed by a semantic encoder ϕ(·) to extract task-relevant semantic features:

$$
\mathbf { z } _ { e } = \phi ( \mathbf { I } ) , \quad \mathbf { z } _ { e } \in \mathbb { R } ^ { c \times h \times w } ,\tag{1}
$$

where $c , h ,$ , and w denote the number of channels, height, and width of the feature map, respectively. The extracted semantic feature $\mathbf { z } _ { e }$ is then transmitted to the receiver. However, direct transmission of $\mathbf { z } _ { e }$ in floating-point format incurs substantial transmission overhead. To reduce this overhead, a quantization module $\nu ( \cdot )$ is introduced to quantize $\mathbf { z } _ { e }$ as

$$
\begin{array} { r } { \mathbf { z } _ { q } = \nu ( \mathbf { z } _ { e } ) , \quad \mathbf { z } _ { q } \in \mathbb { R } ^ { c \times h \times w } , } \end{array}\tag{2}
$$

which is then transmitted to the receiver. We assume perfect physical-layer transmission. Accordingly, channel coding, modulation, equalization, and error-control mechanisms are abstracted as a reliable delivery interface. At the receiver, the received $\mathbf { z } _ { q }$ is fed into the semantic decoder $\phi ^ { - 1 } ( \cdot )$ to reconstruct the image:

$$
\hat { \mathbf { I } } = \boldsymbol { \phi } ^ { - 1 } ( \mathbf { z } _ { q } ) , \quad \hat { \mathbf { I } } \in \mathbb { R } ^ { 3 \times H \times W } .\tag{3}
$$

## III. STORAGE-SCALABLE KNOWLEDGE-BASE REUSE SCHEME

## A. Knowledge-Base Reuse Scheme

A straightforward implementation of the quantization module in Fig. 1 is based on vector quantization. Specifically, the quantization module $\nu ( \cdot )$ is parameterized by a learnable KB $\varphi \in \mathbb { R } ^ { N \times c }$ , which contains $N$ codewords of dimension c. The semantic feature $\mathbf { z } _ { e }$ is reshaped into hw vectors $\{ \mathbf { z } _ { e } ^ { i } \} _ { i = 1 } ^ { h w }$ , each of dimension $c .$ For each vector, the Euclidean distances to all codewords in the KB are computed, and the nearest codeword is selected as

$$
\mathbf { z } _ { q } ^ { i } = { \boldsymbol { \varphi } } ^ { k } , \quad k = \arg \operatorname* { m i n } _ { j } \left\| \mathbf { z } _ { e } ^ { i } - { \boldsymbol { \varphi } } ^ { j } \right\| _ { 2 } ^ { 2 } ,\tag{4}
$$

where $\varphi ^ { j }$ denotes the j-th codeword. By concatenating all $\mathbf { z } _ { q } ^ { i }$ and reshaping them into size $c \times h \times w$ , the quantized semantic feature $\mathbf { z } _ { q }$ is obtained. As illustrated in Fig. 2(a), SKBQ requires only one KB and thus incurs limited storage overhead. However, it provides limited reconstruction performance and does not support progressive transmission. Therefore, its reconstruction quality is often insufficient for high-resolution images.

To enable progressive transmission, MKBQ extends SKBQ by employing multiple homogeneous KBs. As illustrated in Fig. 2(c), the transmission process is divided into T stages. Accordingly, the quantization module $\nu ( \cdot )$ includes T KBs, denoted by $\{ \varphi _ { j } \} _ { j = 1 } ^ { \bar { T } }$ , where $\varphi _ { j } ~ \in ~ \mathbb { R } ^ { \tilde { N } \times c }$ . The first KB generates an initial approximation $\tilde { \mathbf { z } } _ { q } ~ = ~ \varphi _ { 1 } ( \mathbf { z } _ { e } )$ , while the remaining KBs iteratively quantize the residuals. Specifically, the first residual is ${ \bf r } _ { 1 } = { \bf z } _ { e } - \tilde { { \bf z } } _ { q } ,$ , and each subsequent KB produces $\tilde { \mathbf { r } } _ { i } = \varphi _ { i + 1 } ( \mathbf { r } _ { i } )$ , where $\begin{array} { r } { { \bf r } _ { i } = { \bf z } _ { e } - \tilde { \bf z } _ { q } - \sum _ { j = 1 } ^ { i - 1 } \tilde { \bf r } _ { j } } \end{array}$ After T stages, the final quantized semantic feature is

$$
\mathbf { z } _ { q } = \tilde { \mathbf { z } } _ { q } + \sum _ { i = 1 } ^ { T - 1 } \tilde { \mathbf { r } } _ { i } .\tag{5}
$$

By adaptively selecting the number of participating KBs, MKBQ supports progressive transmission. However, two limitations remain. First, MKBQ does not guarantee monotonic refinement across transmission stages. Consequently, a laterstage reconstruction may not outperform an earlier-stage reconstruction, resulting in unstable progressive refinement. Second, the KB storage grows linearly with the number of transmission stages because each stage requires an independent KB containing numerous codewords. Since high-resolution image reconstruction generally requires more transmission stages, the resulting storage overhead can become prohibitively high.

![](images/1538516c9d0f9e5fc767c782644a240cda5f7c8fa155b240ca0e197aa97458b1.jpg)

(a) Single Knowledge-Base Quantization  
![](images/8f492917e32d34cd7819fea04113df1ec0b8bbac86bc6598fbafacb41013ae53.jpg)

(b) Storage-Scalable Knowledge-Base Reuse Quantization  
![](images/55babc4d0ca927c048074eb0d79a1600a7f567451e413b24c7433c898cf2c905.jpg)  
(c) Multi-Knowledge-Base Residual Quantization  
Fig. 2: Knowledge-base quantization schemes.

To overcome these limitations, we propose the SSKBQ scheme, illustrated in Fig. 2(b). Unlike MKBQ, which assigns an independent KB to each transmission stage, SSKBQ reuses a compact set of KBs across multiple residual refinement steps, thereby decoupling the number of maintained KBs from the transmission depth.

Let $\tau _ { j }$ denote the number of residual refinement steps assigned to the j-th KB. The first KB quantizes the semantic feature $\mathbf { z } _ { e }$ to obtain the initial approximation $\tilde { \mathbf { z } } _ { q }$ and is then reused to quantize the following $\tau _ { 1 } - 1$ residuals. The second KB is subsequently reused to quantize the residuals from step $\tau _ { 1 } + 1 \ \mathrm { t o } \ \tau _ { 2 }$ , and the remaining KBs follow the same strategy. Consequently, using only $K$ reusable KBs $( K \ \leq \ T )$ , the proposed scheme supports $T$ progressive transmission stages through KB reuse. The resulting quantized semantic feature is expressed as

$$
{ \mathbf z } _ { q } = \tilde { \mathbf z } _ { q } + \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { \tau _ { j } } \tilde { \mathbf r } _ { j , i } ,\tag{6}
$$

where $\tilde { \mathbf { r } } _ { j , i }$ denotes the quantization result produced by the j-th KB at its i-th residual quantization step.

## B. Training Storage-Scalable Knowledge Base

The semantic codec and the proposed storage-scalable KB are jointly optimized in an E2E manner. The overall objective consists of three components: the reconstruction loss $\mathcal { L } _ { \mathrm { r e c } } .$ , the quantization loss $\mathcal { L } _ { \mathrm { v q } } ,$ and the residual supervision loss ${ \mathcal { L } } _ { \mathrm { r e s } } .$ The reconstruction loss measures the distortion between the reconstructed image <sup>ˆ</sup>I and the original image I using mean square error:

$$
\mathbb { E } _ { \mathbf { I } \sim \mathcal { D } } \left[ \left. \hat { \mathbf { I } } - \mathbf { I } \right. _ { 2 } ^ { 2 } \right] .\tag{7}
$$

The quantization loss enforces consistency between the semantic feature $\mathbf { z } _ { e }$ and quantized feature $\mathbf { z } _ { q } .$

$$
\begin{array} { r } { \mathrm { E } _ { \mathbf { I } \sim \mathcal { D } } \bigg [ \alpha \left. \mathbf { z } _ { e } - \mathrm { s g } [ \mathbf { z } _ { q } ] \right. _ { 2 } ^ { 2 } + \left. \mathrm { s g } [ \mathbf { z } _ { e } ] - \mathbf { z } _ { q } \right. _ { 2 } ^ { 2 } \bigg ] , } \end{array}\tag{8}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. The first term updates the semantic encoder by encouraging $\mathbf { z } _ { e }$ to approach $\mathbf { z } _ { q } ,$ whereas the second term updates the KBs by aligning $\mathbf { z } _ { q }$ with $\mathbf { z } _ { e }$ . The hyperparameter α balances these two effects.

Although the quantization loss explicitly aligns the final quantized feature $\mathbf { z } _ { q }$ with the semantic feature $\mathbf { z } _ { e } ,$ , it imposes no constraint on the intermediate quantized representations generated during progressive transmission. Therefore, we introduce a residual supervision loss to explicitly regularize the stage-wise refinement process. Let the intermediate quantized feature after t refinement stages be defined as:

$$
{ \bf z } _ { q } ( t ) = \tilde { \bf z } _ { q } + \sum _ { i = 1 } ^ { t } \tilde { \bf r } _ { i } .\tag{9}
$$

The residual supervision loss is formulated as

$$
\mathbb { E } _ { \mathbf { I } \sim \mathcal { D } } \left[ \sum _ { i = 1 } ^ { T } \frac { \Lambda } { i } \left( \beta \left. \mathbf { z } _ { e } - \mathrm { s g } \left[ \mathbf { z } _ { q } ( i ) \right] \right. _ { 2 } ^ { 2 } + \left. \mathrm { s g } \left[ \mathbf { z } _ { e } \right] - \mathbf { z } _ { q } ( i ) \right. _ { 2 } ^ { 2 } \right) \right]
$$

Similar to the quantization loss, the first term updates the semantic encoder, while the second term updates the KBs to progressively reduce the residual error. The constant Λ controls the overall strength of progressive supervision, and the weighting factor $1 / i$ assigns larger penalties to earlier refinement stages. By progressively minimizing the stage-wise residual error, the proposed supervision explicitly regularizes intermediate quantized representations under KB reuse, encouraging successive refinement stages to gradually reduce the residual error. The overall training objective is defined as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { v q } } + \mathcal { L } _ { \mathrm { r e s } } .\tag{10}
$$

## IV. SIMULATION RESULTS

## A. Experiment Settings

1) Datasets: The Cityscapes and COCO datasets are used to evaluate the proposed SSKBQ scheme. Cityscapes contains 2,975 training images and 500 validation images of urban street scenes, whereas COCO contains more diverse objects and visual scenes. For a consistent evaluation, images from both datasets are resized to 128 × 64.

2) Evaluation Metrics: Reconstruction quality is evaluated using PSNR [7], SSIM [8], FID [9], and KID [10].

3) Baselines: We compare SSKBQ with representative non-progressive and progressive schemes. JSCC [11] directly transmits continuous semantic features and serves as an unquantized reference. For SKBQ, the VQVAE [5] system is implemented using one-hot and Gumbel-softmax quantization.

MKBQ employs an independent KB at each transmission stage.

4) Parameter Settings: For both datasets, the input image and semantic feature dimensions are $3 \times 1 2 8 \times 6 4$ and $2 5 6 \times 3 2 \times 1 6$ , respectively. Each KB contains $N \ = \ 5 1 2$ codewords of dimension $c \ = \ 2 5 6$ , and hw = 512 spatial feature vectors are quantized at each stage. We consider 16- stage progressive transmission and set $\alpha = \beta = 0 . 2 5$ and $\Lambda = 8 .$ During testing, only the codeword indices associated with the quantized feature vectors are transmitted. The receiver retrieves the corresponding codewords from the shared KBs according to the received indices and reconstructs the quantized semantic feature.

## B. Complexity and Scalability

SKBQ, MKBQ, and SSKBQ require $O ( N c ) , O ( T N c )$ , and $O ( K N c )$ KB storage, respectively. By reusing K KBs across T stages, SSKBQ requires only $K / T$ of the KB storage of MKBQ.

At each stage, nearest-neighbor assignment compares hw feature vectors of dimension c with N codewords, resulting in O(chwN) transmitter-side complexity. The cumulative complexity after t stages is therefore O(tchwN) for both MKBQ and SSKBQ.

At the receiver, the transmitted indices directly identify the codewords. Index lookup, feature assembly, and residual accumulation require O(chw) operations per stage and $O ( t c h w )$ after t stages, independent of N.

Let $C _ { \mathrm { d e c } }$ denote the cost of one decoder execution. Reconstruction after all t stages requires $O ( t c h w + C _ { \mathrm { d e c } } )$ , whereas reconstruction after every stage requires $O ( t c h w + t C _ { \mathrm { d e c } } )$ Hence, progressive inference latency is mainly determined by repeated decoder executions, while KB processing grows only linearly with t. Actual wall-clock latency also depends on the hardware platform and implementation.

## C. Comparison with Existing Schemes

To compare the proposed SSKBQ with existing semantic communication schemes, experiments are conducted on the Cityscapes and COCO datasets. The progressive reconstruction results are presented in Figs. 3 and 4, where different numbers of reusable KBs are evaluated over 16 transmission stages.

Across both datasets, SSKBQ exhibits consistent progressive reconstruction trends. As more transmission stages are received, the reconstruction quality gradually improves in terms of PSNR, SSIM, FID, and KID, validating the effectiveness of KB reuse for progressive semantic refinement. Moreover, compared with MKBQ under comparable settings, SSKBQ achieves better reconstruction performance, demonstrating that the proposed stage-aware residual supervision facilitates the optimization of reused KBs across multiple refinement stages.

Compared with single-stage quantization schemes, including VQVAE(OH) and VQVAE(GS), SSKBQ achieves consistent improvements on both datasets, demonstrating the benefit of residual refinement. The comparison with JSCC shows dataset-dependent behavior. On Cityscapes, SSKBQ achieves higher PSNR after sufficient progressive stages, whereas JSCC maintains better performance on COCO. This difference results from the interaction among dataset complexity, semantic representation capability, and codec capacity. JSCC avoids quantization distortion by transmitting continuous semantic features, which is advantageous for complex image distributions. In contrast, SSKBQ provides a more efficient progressive transmission mechanism when semantic information can be effectively organized through reusable KBs.

PSNR vs Transmission Stage  
![](images/696de8dec156f67f01749d2ecaf619907c7ebb063de5a1d59b87937677506e7a.jpg)  
Transmission Stage  
(a) PSNR

![](images/d10bb21c481c880ef04ddb45510ad6a257d39a941b495eb5861192778bcb955d.jpg)  
(b) SSIM

![](images/1804493c6c170b3290569fd66945056ba8891060725df5cd29a3910ec030154b.jpg)  
(c) KID

![](images/b69ed9df13b473ecd51e6c8f66f716b3ed12f32bb793e390b9ca9087bbedab4b.jpg)  
(d) FID

Fig. 3: Reconstruction performance comparison on the Cityscapes dataset.  
![](images/4b275882edd4a85791200d5bd6dba704c26a232391ca901eca763f4d0272b0f4.jpg)  
(a) PSNR

![](images/8e85f9c4d807cbf9914237a42dc8abdd563fdd49abf2905a663b22cde20e7b72.jpg)  
(b) SSIM

![](images/ad6fbf855d4e02e10bff9cf238162a3de1dd926f5ca799825b58043daee85fe6.jpg)  
(c) KID

![](images/838fb7b6e9564c0121854f71b62a5129f97a97b4218b765cbb81589e619404ff.jpg)  
(d) FID

Fig. 4: Reconstruction performance comparison on the COCO dataset.  
![](images/ddefeadf679f240118b266867f4aca1986667d8da009ed5f5c1d7d4ef74e4908.jpg)  
(a) Ground Truth

![](images/02b66a8ea5997f9638ab7a872465bb3b99df7aa024d1d44d052f8c682c740b01.jpg)

![](images/14150c1041dc8a87b8db4419f9ca9bd8be58116a174248e716ab8ac9a403f0ed.jpg)

![](images/ae207629e9d677a8264b69b2562b6b0807ee549e605130f92052cecc6dc00c8f.jpg)  
(f) JSCC

(b) Stage1(SSKBQ)  
![](images/2d46d34d57161e3e95a009d38c831da60fb493577d4209622751c9eb16dca84b.jpg)  
(g) VQVAE(Gumbel-Softmax)

(c) Stage4(SSKBQ)  
![](images/3419fc78831978f57880f16e5ed4a1fdc4fe26ae369325a6a086f11691515e0e.jpg)  
(h) VQVAE(One-Hot)

(e) Stage16(SSKBQ)  
(d) Stage8(SSKBQ)  
![](images/73c2a1f0735a005435ce1e3aa878de942b47f3c7ae23f6bb5beb7e34ea13d0ab.jpg)  
(i) Stage8(MKBQ)

![](images/f12598225b45585581101dd0c0fce4fbf0de785b14cf0fa79eba094d27fcee90.jpg)  
(j) Stage16(MKBQ)  
Fig. 5: Image reconstruction performance under different baselines.

The influence of KB number is consistent with the observations in the previous subsection. In early transmission stages, smaller KB configurations provide better performance because semantic information is more concentrated within each reusable KB. With increasing transmission stages, larger KB configurations gradually benefit from their higher representation capacity and achieve better final reconstruction quality. This result highlights the necessity of balancing representation capacity and KB scalability, which motivates the proposed KB reuse strategy. A representative reconstruction example is

shown in Fig. 5.

For quantitative comparison, Tables I and II summarize the final reconstruction performance of different schemes on the Cityscapes and COCO datasets, respectively. The JSCC results are highlighted in bold as the continuous-feature transmission baseline. VQVAE(GS) and VQVAE(OH) denote the Gumbel– Softmax and one-hot quantization implementations, respectively. In MKBQ(t) and SSKBQ(t), t represents the number of received progressive transmission stages.

To evaluate communication efficiency, the number of transmitted bits for one image sample is adopted as the transmission rate. Specifically, JSCC directly transmits the continuous semantic feature $\mathbf { z } _ { e } ~ \in ~ \mathbb { R } ^ { 2 5 6 \times 3 2 \times \bar { 1 6 } }$ , resulting in 8,388,608 transmitted bits under 64-bit floating-point representation. For MKBQ and SSKBQ, only the indices of selected codewords are transmitted. Since each KB contains N = 512 codewords, each index requires 9 bits. Therefore, the transmission rates for 1, 8, and 16 progressive stages are 4,608, 36,864, and 73,728 bits, respectively. For VQVAE(GS), transmitting the

TABLE I: Image Reconstruction Performance Comparison (Cityscapes).
<table><tr><td>Method</td><td>PSNR (dB)</td><td>SSIM</td><td>FID</td><td>KID</td></tr><tr><td>JSCC</td><td>24.24</td><td>0.9123</td><td>71.38</td><td>0.0398</td></tr><tr><td>MKBQ(1)</td><td>15.69</td><td>0.3739</td><td>426.77</td><td>0.5728</td></tr><tr><td>MKBQ(8)</td><td>21.62</td><td>0.6912</td><td>299.22</td><td>0.3569</td></tr><tr><td>MKBQ(16)</td><td>24.58</td><td>0.7972</td><td>183.94</td><td>0.1831</td></tr><tr><td>SSKBQ(1)</td><td>25.28</td><td>0.8288</td><td>155.60</td><td>0.1418</td></tr><tr><td>SSKBQ(8)</td><td>25.15</td><td>0.8137</td><td>167.59</td><td>0.1602</td></tr><tr><td>SSKBQ(16)</td><td>27.36</td><td>0.8739</td><td>111.77</td><td>0.0860</td></tr><tr><td>VQVAE(GS)</td><td>23.16</td><td>0.6463</td><td>317.84</td><td>0.3793</td></tr><tr><td>VQVAE(OH)</td><td>21.22</td><td>0.6297</td><td>287.42</td><td>0.3336</td></tr></table>

512-dimensional soft assignment weights for each spatial feature vector requires 16,777,216 transmitted bits. These results demonstrate that SSKBQ enables discrete-index semantic transmission with substantially reduced communication overhead compared with continuous-feature and soft-assignmentbased schemes.

As shown in Table I, SSKBQ(16) achieves a PSNR of 27.36 dB on the Cityscapes dataset, improving by 11.31%, 28.93%, and 18.13% over MKBQ(16), VQVAE(OH), and VQVAE(GS), respectively. Meanwhile, SSKBQ(16) requires only 73,728 transmitted bits, corresponding to less than 1% of the transmission overhead of JSCC. Despite this substantial reduction in communication cost, SSKBQ achieves a 12.87% PSNR improvement over JSCC. However, JSCC obtains better SSIM, FID, and KID values, which can be attributed to its direct transmission of continuous semantic features and the resulting avoidance of quantization distortion.

The results on the COCO dataset exhibit a different trend. As shown in Table II, SSKBQ(16) achieves a PSNR of 23.53 dB, outperforming MKBQ(16), VQVAE(OH), and VQ-VAE(GS) by 10.16%, 24.17%, and 27.60%, respectively. However, JSCC still achieves higher reconstruction quality on COCO due to its stronger representation capability for diverse object categories and complex visual distributions. This observation indicates that SSKBQ is not intended to universally replace continuous-feature JSCC, but rather provides a storage-efficient and progressive semantic transmission framework that achieves a favorable trade-off between communication overhead and reconstruction performance.

## V. CONCLUSION

This letter investigates the storage-performance trade-off in progressive semantic communication with knowledge-base quantization. Conventional single knowledge-base schemes achieve low storage overhead but suffer from limited refinement capability, while multi-knowledge-base residual schemes support progressive transmission at the cost of storage complexity that scales with the number of refinement stages. To address this issue, we propose a storage-scalable knowledgebase reuse quantization scheme that decouples the number of maintained knowledge bases from the number of progressive transmission stages. By reusing a limited number of knowledge bases across multiple refinement stages, SSKBQ reduces storage requirements while preserving progressive reconstruction capability.

TABLE II: Image Reconstruction Performance Comparison (COCO)
<table><tr><td>Method</td><td>PSNR (dB)</td><td>SSIM</td><td>FID</td><td>KID</td></tr><tr><td>JSCC</td><td>29.20</td><td>0.9379</td><td>30.96</td><td>0.0027</td></tr><tr><td>MKBQ(1)</td><td>13.22</td><td>0.3238</td><td>400.74</td><td>0.3876</td></tr><tr><td>MKBQ(8)</td><td>18.77</td><td>0.5886</td><td>238.82</td><td>0.1763</td></tr><tr><td>MKBQ(16)</td><td>21.36</td><td>0.7116</td><td>168.82</td><td>0.0936</td></tr><tr><td>SSKBQ(1)</td><td>13.02</td><td>0.3721</td><td>344.97</td><td>0.3111</td></tr><tr><td>SSKBQ(8)</td><td>21.60</td><td>0.7240</td><td>171.49</td><td>0.0974</td></tr><tr><td>SSKBQ(16)</td><td>23.53</td><td>0.7939</td><td>122.80</td><td>0.0512</td></tr><tr><td>VQVAE(GS)</td><td>18.44</td><td>0.5341</td><td>221.64</td><td>0.1627</td></tr><tr><td>VQVAE(OH)</td><td>18.95</td><td>0.5734</td><td>215.05</td><td>0.1450</td></tr></table>

Experimental results on different image datasets demonstrate that SSKBQ achieves competitive reconstruction performance with substantially reduced storage overhead compared with existing knowledge-base quantization schemes. Moreover, the results reveal that increasing the number of knowledge bases does not always guarantee better performance, since semantic fragmentation and optimization difficulty may limit their effective utilization. Developing more effective training strategies for multi-KB architectures is therefore an important direction for future research. Furthermore, the current study focuses on semantic-layer KB reuse under reliable index delivery. Extending SSKBQ to practical communication environments, including noisy index transmission, fading, packet loss, and channel-aware KB adaptation, will be investigated in future work.

## REFERENCES

[1] Y. Shao, Q. Cao, and D. Gund¨ uz, “A theory of semantic communication,”¨ IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 12 211– 12 228, 2024.

[2] X. Luo, H.-H. Chen, and Q. Guo, “Semantic communications: Overview, open issues, and future research directions,” IEEE Wireless communications, vol. 29, no. 1, pp. 210–219, 2022.

[3] W. Yang, H. Du, Z. Q. Liew, W. Y. B. Lim, Z. Xiong, D. Niyato, X. Chi, X. Shen, and C. Miao, “Semantic communications for future internet: Fundamentals, applications, and challenges,” IEEE Communications Surveys & Tutorials, vol. 25, no. 1, pp. 213–250, 2022.

[4] P. Markstein, “The new ieee-754 standard for floating point arithmetic.” Schloss Dagstuhl–Leibniz-Zentrum fur Informatik, 2008.¨

[5] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.

[6] M. Adiban, K. Stefanov, S. M. Siniscalchi, and G. Salvi, “S-hr-vqvae: Sequential hierarchical residual learning vector quantized variational autoencoder for video prediction,” IEEE Transactions on Multimedia, 2025.

[7] B. Jahne,¨ Digital image processing. Springer Science & Business Media, 2005.

[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[9] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017.

[10] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demysti-´ fying mmd gans,” 2021.

[11] E. Bourtsoulatze, D. B. Kurka, and D. Gund¨ uz, “Deep joint source-¨ channel coding for wireless image transmission,” IEEE Transactions on Cognitive Communications and Networking, vol. 5, no. 3, pp. 567–579, 2019.