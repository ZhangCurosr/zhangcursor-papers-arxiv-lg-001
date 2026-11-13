# STeP-C: Real-Time Neural Compression for Distributed Fiber-Optic Sensing at Scale

Tingfeng Li Ming-Fang Huang Shaobo Han Yue Tian

NEC Laboratories America

{tli, mhuang, shaobo, ytian}@nec-labs.com

## Abstract

Distributed fiber-optic sensing (DFOS) technologies such as Distributed Acoustic Sensing (DAS) generate continuous, high-resolution spatio-temporal data at rates exceeding hundreds ofmegabytes per second, creating severe challenges for real-time storage and transmission. While neural compression methods can achieve strong rate-distortion performance on sensing data, prior approaches either require per-segment retraining incompatible with streaming, or have only been validated offline. None have been demonstrated end-to-end on a continuously streaming DAS acquisition system at scale. We present STeP-C (Spatio-Temporal Predictive Compression), a neural compression framework that decouples spatial encoding, temporal prediction, and entropy-optimized residual coding, trained end-to-end with a rate-distortion objective motivated by conditional entropy reduction. Critically, we identify entropy coding as the dominant bottleneckfor streaming deployment. We address it through GPU-side symbol computation, cached probability tables, and parallel GIL-free arithmetic coding, together with FP16 inference and batched entropy coding, enabling a full real-time pipeline built around a GPU ring buffer. We validate this pipeline on a live DAS testbed, achieving a 98.85% file size reduction at 31.04 dB full-band PSNR and an average throughput of535 MB/s, exceeding the 324 MB/s requirement for 1 kHz, 81 km DAS streams. We further show that STeP-C substantially outperforms conventional codecs at practical streaming bitrates, can be applied to unseen channel ranges within the same acquisition system without retraining, and largely maintains downstream detector consistency, with 96.62% high-confidence class agreement and 100% Jaccard similarity on detected event types.

## 1. Introduction

Distributed fiber-optic sensing (DFOS) has transformed large-scale environmental [16] and infrastructure [12] monitoring by enabling dense, continuous measurements along kilometer-scale cables. In particular, Distributed Acoustic

Sensing (DAS) converts an existing fiber-optic cable into an array of thousands of spatially distributed vibration-sensing channels [28], including cables already carrying live commercial traffic [27], sampled at 500–1000 Hz, producing data volumes from tens of gigabytes to over a terabyte per hour. The resulting stream is naturally a two-dimensiona “waterfall” image: one axis indexes position along the fiber, the other indexes time, and each pixel is a strain-rate measurement. Practitioners commonly inspect it through this plot, because coherent physical events appear as spatially and temporally extended visual structures, most visibly diagonal streaks whose slope encodes propagation velocity, see Figure 1(b). Viewed this way, a DAS interrogator acts like a camera producing a temporally unbounded video of the ground. For applications such as seismic monitoring [1], cable and pipeline intrusion detection [10, 11], traffic detection [14, 27], and subsea infrastructure monitoring [17], the ability to compress and transmit or store this data in real time at acquisition is a critical operational requirement.

The visual analogy is productive but not literal. Conventional image and video codecs are designed primarily around natural-image and video statistics and therefore do not directly match the structure of DAS data. JPEG 2000 primarily exploits two-dimensional spatial redundancy, while video codecs such as H.264 additionally use block-based temporal prediction. In contrast, DAS exhibits strongly anisotropic structure: neighboring sensing channels form a quasi-1D spatial sequence, while temporal variation reflects smooth signal evolution punctuated by localized propagating events rather than apparent object motion. Moreover, DAS compression must preserve highdynamic-range waveform fidelity for downstream physical inference rather than primarily perceptual visual quality. As we demonstrate experimentally, these mismatches limit reconstruction fidelity at the low bitrates required for practical streaming. What transfers from vision is therefore not the codec but the paradigm: learned nonlinear transform coding with an entropy model [2], and predictive residual coding across frames [6, 18], provided the transform, the predictor, and the entropy model are re-designed for the structure of

![](images/2777b114621565bb7a7be37df2a8cb09581963283fa0484967b9ca5f986d2b9a.jpg)  
Figure 1. STeP-C for real-time DAS compression. (a) A telecom fiber acts as a dense array of sensing channels. (b) The DAS readout forms a spatio-temporal “video” in which physical events appear as structured patterns. (c) STeP-C predicts each frame and entropy-codes only the residual. (d) Reconstructed signals yield consistent downstream detections.

the sensor data.

Existing DAS-specific compression methods address parts of this problem, but typically trade off compression efficiency, real-time operation, general-purpose reconstruction, or adaptability to continuous streams. These limitations leave a gap for a high-ratio neural compressor that can operate end-to-end on live DAS acquisition. In this work, we present STeP-C (Spatio-Temporal Predictive Compression), a neural compression framework for DFOS data whose overall workflow is illustrated in Figure 1. STeP-C achieves competitive rate-distortion performance through a modular three-stage architecture: (1) a 1D convolutional spatial encoder; (2) a recurrent temporal predictor that forecasts the next latent frame from past embeddings; and (3) an entropy-optimized residual encoder trained under a learned entropy model.

Each stage is the sensor-domain analogue of a component in learned video compression: the 1D convolutional spatial encoder plays the role of the intra-frame analysis transform, replacing 2D convolutions with convolutions along the fiber’s channel dimension; the recurrent temporal predictor replaces optical-flow motion compensation, forecasting the next latent directly rather than warping a reference frame; and the residual encoder performs residual coding under a learned entropy model. The temporal predictor is motivated by conditional entropy reduction: conditioning the current latent on available temporal context cannot increase its uncertainty, and reduces it when the current latent is statistically dependent on that context. DAS signals typically exhibit strong temporal correlation, motivating predictive coding before residual entropy coding (see Sec. 3.1).

To translate this into deployable real-time throughput, we identify entropy coding as the dominant bottleneck and redesign its implementation for streaming operation. Symbol computation is moved to the GPU, probability tables are cached across calls, and arithmetic coding is parallelized across GIL-free C++ threads, together yielding a

30× speedup over the unoptimized baseline. We validate the resulting pipeline not in simulation alone, but on a live DAS testbed with 81,000 sensing channels, corresponding to approximately 81 km of fiber at 1 m channel spacing, achieving 87.30× compression and 31.04 dB full-band PSNR at an average throughput of 535 MB/s. Beyond the live testbed, we evaluate reconstruction on fiber channels not seen during training and assess whether a downstream event detector produces consistent predictions on compressed reconstructions relative to the original data.

Our main contributions are:

• Vision-centric formulation for DFOS compression. We model the DAS waterfall as a spatio-temporal visual signal and adapt learned transform coding to its 1D spatial layout and high-dynamic-range statistics, with a modular architecture separating spatial encoding, temporal prediction, and residual coding.

• Real-time neural compression system. We identify entropy coding as the dominant throughput bottleneck and redesign the coding path with GPU-side symbol computation, cached probability tables, parallel GIL-free arithmetic coding, FP16 inference, and batched coding. The resulting system enables, to our knowledge, the first neural DAS compressor validated end-to-end in a live acquisition pipeline directly coupled to a continuously streaming DAS interrogator.

• Validation on real deployments. We outperform conventional and DAS-specific codecs, demonstrate reconstruction beyond the training channel range without retraining, and show that downstream detector predictions remain largely consistent after compression.

## 2. Related Work

STeP-C extends neural compression beyond its conventional image and video settings to a domain-specific but task-agnostic streaming problem. Rather than transferring an existing visual codec, we adapt the underlying principles of learned transform coding, temporal prediction, residual entropy coding, and rate-distortion optimization to the distinct spatial and temporal regularities of DFOS data. This formulation preserves a general-purpose reconstruction rather than optimizing the representation for a predefined downstream task, allowing the compressed signal to support storage, transmission, visualization, and future analyses. More broadly, STeP-C demonstrates that learned predictive and entropy coding can exploit structured redundancy beyond natural-image statistics, while its realtime deployment exposes practical bottlenecks, particularly entropy-coding throughput and host–device transfer overhead, that are often hidden in offline evaluation. These observations may also inform neural compression systems for other large-scale, continuously streamed sensing data. Table 1 summarizes how representative DAS data compression approaches compare along the deployment-oriented criteria discussed below.

Table 1. Comparison of representative DAS data compression approaches. ✓: demonstrated; ×: not demonstrated; △: partial or related demonstration.
<table><tr><td>Method</td><td>General-purpose No per-segment Entropy reconstruction</td><td>training</td><td>coding</td><td>Real-time processing</td><td>Downstream task validation</td></tr><tr><td>Dong et al. [7]</td><td>√</td><td>√</td><td>X</td><td>√</td><td>×</td></tr><tr><td>Wang et al. [25]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Ni et al. [21]</td><td>√</td><td>√</td><td>×</td><td>△</td><td>√</td></tr><tr><td>Chen et al. [4]</td><td>√</td><td>×</td><td>X</td><td>×</td><td>×</td></tr><tr><td>Shen et al. [23]</td><td>×</td><td>√</td><td>×</td><td>√</td><td>√</td></tr><tr><td>Wang et al. [26]</td><td>△</td><td>√</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Sun et al. [24]</td><td>√</td><td>V</td><td>√</td><td>√</td><td>×</td></tr><tr><td>DASPack [22]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>Ma et al. [19]</td><td>√</td><td>√</td><td>×</td><td>△</td><td>√</td></tr><tr><td>Gemeinhardt et al. [9]</td><td>√</td><td>√</td><td>×</td><td>√</td><td>△</td></tr><tr><td>STeP-C (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 2.1. Neural Image and Video Compression

End-to-end learned compression has achieved state-of-theart rate-distortion performance on natural images and video. [2] introduced the hyperprior framework using nonlinear transforms and learned entropy models. For video, [18] proposed DVC using optical flow for motion estimation, while [6] proposed inter-frame neural compression using residual coding in the feature space. [20] extended this with joint autoregressive and hierarchical priors. Multipleframe prediction was explored by [5, 15]. STeP-C adapts the residual latent coding paradigm from these works to the DFOS domain, replacing 2D image encoders with 1D convolutional encoders suited for fiber-optic sensor arrays. Unlike the methods above, which are typically evaluated on pre-recorded video, the proposed method explicitly addresses the throughput constraints of compressing a live sensor stream in real time.

## 2.2. Neural and Task-Oriented Compression Methods for DAS

Learned models have been applied directly to DAS compression, but under assumptions incompatible with streaming. Wang et al. [25] evaluate a recurrent autoencoder on pre-recorded DAS segments, without demonstrating continuous real-time compression. Chen et al. [4] compress 2D patches with a vision transformer, reaching ratios near 50× when combined with downsampling, but train selfsupervised per segment: each new 60-second window requires on the order of 30 minutes of GPU training, or roughly 10 fine-tuning iterations with transfer from a previous window. This suits archival compression of recorded campaigns but not a live stream, where a fixed model must generalize across incoming data without retraining. Shen et al. [23] instead map a frequency-band-energy extractor into the compressed-sensing domain, detecting and classifying events at a 70% reduction in transmitted data without reconstructing the signal; this is fast for the task it targets but produces no reusable data for storage, transmission, or analyses not anticipated at acquisition time.

## 2.3. Non-Neural Compression for DAS

Non-neural DAS compressors provide an important deployment baseline because several already support real-time operation without model training. Dong et al. [7] integrate lossless compression directly into the DAS storage workflow, demonstrating compression at acquisition rate. DAS-Pack [22] combines wavelet transforms and linear prediction with arithmetic entropy coding, providing lossless or bounded-error reconstruction while remaining suitable for real-time CPU deployment. Sun et al. [24] similarly employ asymmetric numeral systems (ANS) entropy coding for high-ratio DAS compression. These approaches demonstrate that explicit entropy-coded bitstreams and real-time processing are achievable with non-neural pipelines, while relying on hand-designed transforms and predictors rather than learned spatio-temporal representations. STeP-C targets a different operating point by combining learned predictive coding with entropy coding, general-purpose reconstruction, and real-time operation.

## 2.4. From Real-Time Inference to Live Streaming Deployment

Real-time capability has been explored at different stages of the DAS processing pipeline, but fast inference alone does not establish that a neural compressor can operate continuously at the acquisition rate. Ni et al. [21], for example, demonstrate that their lightweight reconstruction model can run on a Raspberry Pi, showing the feasibility of neural processing at the edge. Their evaluation, however, benchmarks reconstruction from subsampled DAS data rather than an end-to-end compression pipeline directly coupled to a live interrogator. Classical methods such as Dong et al. [7] and DASPack [22] do demonstrate real-time DAS compression, but learned approaches have largely been evaluated on prerecorded data or through component-level runtime measurements. For continuous neural compression, the complete acquisition-to-bitstream pipeline must keep pace with incoming data while handling buffering, preprocessing, encoding, and output without interrupting the stream. To our knowledge, STeP-C is the first neural DAS compressor validated under this end-to-end live-acquisition setting.

## 3. Method

## 3.1. Spatio-Temporal Predictive Architecture

Problem formulation. Let $X = ( x _ { 0 } , x _ { 1 } , \dots , x _ { T - 1 } )$ denote a sequence of T DAS frames, where each frame $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { C }$ represents measurements from C sensing channels at time step t. We train the compression model end-to-end by minimizing the following rate-distortion-regularization objective:

$$
\mathcal { L } = \lambda _ { r } \cdot R + \lambda _ { d } \cdot D ( x _ { t } , \hat { x } _ { t } ) + \lambda _ { \mathrm { t p } } \cdot \mathcal { L } _ { \mathrm { t p } }\tag{1}
$$

where R is the estimated bitrate, $D ( \cdot )$ measures reconstruction distortion (MSE), and $\mathcal { L } _ { \mathrm { t p } }$ is the temporal prediction loss described later in this section.

Spatial Encoder. Each frame $x _ { t }$ is mapped to a latent feature representation $k _ { t } = f _ { a } ( x _ { t } )$ using a 1D convolutional encoder. Stacked convolutions along the channel dimension capture spatial correlations among neighboring sensing channels and provide the representation used for temporal prediction and residual coding.

Temporal Predictor. An LSTM temporal predictor forecasts the current spatial latent from a shared temporal context available at both the encoder and decoder. At the beginning of each sequence, no temporal prediction is available, so we set $\hat { k } _ { 0 } = 0$ . The first spatial latent is therefore coded directly through the residual path, yielding the reconstructed latent $\tilde { k } _ { 0 }$ that initializes the temporal context. For subsequent frames, the predictor operates recursively using $\tilde { k } _ { 0 }$ followed by its own previous predictions:

$$
\hat { k } _ { t } = g ( \tilde { k } _ { 0 } , \hat { k } _ { 1 } , \dots , \hat { k } _ { t - 1 } ) , \qquad t \geq 1 .\tag{2}
$$

This open-loop design keeps the prediction history identical at the encoder and decoder, avoids feedback of quantization error from reconstructed latents, and allows all residuals of a sequence to be entropy-coded in a single batched call. Let $H ( \cdot )$ denote entropy. Because every $\ddot { k } _ { t }$ is a deterministic function of $\tilde { k } _ { 0 } .$ , the predictor conditions on $\tilde { k } _ { 0 }$ which cannot increase the uncertainty of the current latent: $H ( k _ { t } \ | \ \tilde { k } _ { 0 } ) \leq H ( k _ { t } )$ , with strict inequality when the two are statistically dependent. Since DAS signals are strongly correlated within a short sequence, the predictor can explain a substantial part of $k _ { t } ,$ , and the residual encoder represents the remaining prediction error.

![](images/41490574233831091b790101bead37850d3288209cbab1cbca720e702968e369.jpg)  
Figure 2. Spatio-temporal predictive compression architecture. The highlighted region corresponds to the components that are executed by the receiver to reconstruct the sensing data from a compressed bitstream.

Temporal prediction loss. To explicitly supervise the temporal predictor and encourage accurate latent forecasts, we introduce an auxiliary loss directly in latent space:

$$
\mathcal { L } _ { \mathrm { t p } } = \mathbb { E } \Big [ \| \hat { k } _ { t } - k _ { t } \| ^ { 2 } \Big ]\tag{3}
$$

This loss penalizes prediction errors independently of the rate-distortion objective, encouraging more accurate latent forecasts and smaller residuals.

Residual encoding and entropy coding. For each frame, the residual $r _ { t } = k _ { t } - \ddot { k } _ { t }$ captures the prediction error not explained by the temporal predictor. For $t = 0 , \hat { k } _ { 0 } = 0 .$ , and hence $r _ { 0 } = k _ { 0 } ;$ the first frame is therefore encoded without temporal prediction. For $t \geq 1$ , only the prediction residual is passed to the residual encoder. A nonlinear encoder $h _ { a }$ maps the residual to a compact latent: $z _ { t } ~ = ~ h _ { a } ( r _ { t } )$ The latent $z _ { t }$ is quantized to $\hat { z } _ { t } ~ = ~ Q ( z _ { t } )$ via uniform quantization, approximated during training by additive uniform noise $\mathcal { U } ( - 0 . 5 , 0 . 5 )$ as a differentiable proxy [2]. Entropy coding uses arithmetic encoding with a learned probability model $P ( \hat { z } _ { t } )$ ; the estimated bitrate contribution is $R = - \log _ { 2 } P ( \hat { z } _ { t } )$

Decoding and Reconstruction. At the receiver, arithmetic decoding restores $\hat { z } _ { t }$ . The residual decoder reconstructs $\tilde { r } _ { t } = h _ { s } ( \hat { z } _ { t } )$ , which is combined with the predicted latent: $\tilde { k } _ { t } ~ = ~ \hat { k } _ { t } + \tilde { r } _ { t }$ . Finally, the spatial decoder reconstructs $\hat { x } _ { t } = f _ { s } ( \tilde { k } _ { t } )$ . The entire pipeline is trained jointly end-to-end, as shown in Figure 2.

## 3.2. Real-Time Systems Engineering

Identifying the bottleneck. A trained STeP-C model is only deployable in real time if every stage of the pipeline sustains the target throughput. We find that the spatial encoder, temporal predictor, and residual network all run efficiently on the GPU and do not limit throughput. Instead, entropy coding becomes the primary bottleneck. Standard arithmetic coding implementations, including the one provided by CompressAI [3], exhibit several inefficiencies in a streaming setting. The coder holds Python’s Global Interpreter Lock during execution, so only one thread can run at a time even when calls are issued asynchronously. The probability table, expressed as a cumulative distribution function (CDF), is recomputed from float32 to int16 on every single call, even though the underlying model has not changed. Quantized symbols are computed by subtracting medians on the CPU, requiring a device-to-host transfer of the full latent tensor before the subtraction can occur. Coding also proceeds sequentially per element, with no parallelism across the batch.

An optimized entropy-coding path. We address these inefficiencies with a custom C++ extension that reorganizes the entropy-coding path around several complementary optimizations. Symbol computation moves to the GPU: medians are subtracted from quantized values directly on-device, before any data is transferred to the CPU, eliminating a redundant round trip across the PCIe bus. The CDF table is cached: the conversion from float32 to int16 happens once, at model load or update time, and every subsequent call reuses it. The extension releases Python’s GIL and performs arithmetic coding across multiple GIL-free C++ threads (8 in our deployment), enabling true parallelism across the batch. Together, these changes yield a 30× improvement in encoding throughput over the unoptimized pipeline. We apply two further accelerations on top of this fix: running the spatial encoder, temporal predictor, and residual MLP in float16 under mixed-precision autocast to reduce neural inference cost on supported hardware, and batching entropy coding across all time steps in a sequence, issuing a single coder call per batch rather than one call per time step. Section 4.5 reports the cumulative throughput gains from these optimizations.

Pipeline architecture. Sustaining real-time through put on a continuously arriving sensor stream requires an acquisition-aware streaming pipeline, as illustrated in Fig ure 3. We implement two concurrent processes that communicate through GPU shared memory. The acquisition process interfaces directly with the DAS interrogator, receiving raw phase frames at the full acquisition rate across all channels, and performs three tasks concurrently: writing raw frames to disk for archival (optional), writing each row into a GPU ring buffer, and incrementing a write counter in shared memory. The ring buffer holds 8,192 slots, providing approximately 8 seconds of buffering headroom, and is shared with the compression process via a CUDA interprocess communication (IPC) handle, avoiding CPU staging of the data. The compression process monitors the write counter with a lightweight polling thread, consumes new rows directly from the ring buffer via a GPUto-GPU copy, and applies, in sequence, a streaming bandpass filter, normalization, spatial patch extraction, neural compression, and parallel entropy coding, before writing compressed output. Two of these steps require modification from their offline counterparts to remain correct under streaming. The band-pass filter is stateful, so applying it independently to each incoming chunk introduces edge artifacts at chunk boundaries; we carry the filter state across chunks by prepending the final rows of each chunk to the next as warm-up context. Normalization uses a hyperbolic tangent transform with a fixed 99th-percentile scale established offline during training, so that scaling remains consistent across chunks rather than depending on the statistics of each chunk in isolation.

![](images/7fed549dea92822bc174594844fa0ee18868d693dcedacbca8fd9efe926fa2b2.jpg)  
Figure 3. Real-time Compression Pipeline.

## 4. Experiments

## 4.1. Datasets, Metrics, and Implementation

Datasets. We evaluate on three real-world DAS datasets collected using our system. The NJ Coastal Fiber Array (NJCFA) consists of 40 recordings from a fiber-optic cable deployment along the New Jersey coast, each containing 3,584 sensing channels. We segment the recordings into sequences of T consecutive frames, with each frame spanning 512 channels, yielding 22,281 training samples and 5,565 validation samples, together with a heldout test set of 27,916 samples. NJCFA is used for ratedistortion evaluation, comparison with conventional and DAS-specific codecs, and architecture ablations. The Dallas Ambient DAS (DADAS) dataset consists of continuous recordings collected from an urban fiber deployment in Dallas at 800 Hz across 81,000 channels. We use a 4,000- channel sub-band (channels 6,000–10,000), yielding 73,472 training samples, 18,368 validation samples, and 89,908 test samples. DADAS is used to evaluate rate-distortion performance, qualitative reconstruction, transfer to fiber channels outside the training range, and downstream detector consistency. For both NJCFA and DADAS, training, validation, and test sets are formed from separate recordings acquired at different times; no recording is shared across splits. The Princeton Fiber Testbed (PFT) consists of data streamed directly from a live DAS interrogator at 1,000 Hz across 81,000 channels. PFT is used to evaluate the complete real-time compression pipeline under live acquisition, including end-to-end throughput, compression ratio, and full-band reconstruction quality.

Metrics. We report two primary compression metrics. Bits per sample (BPS) measures the average number of compressed bits per raw data element, where each element is a 32-bit float. Lower BPS indicates more compression. Peak signal-to-noise ratio (PSNR, in dB) measures reconstruction fidelity as $1 0 \log _ { 1 0 } { ( M A X ^ { 2 } / M S E ) }$ , where MAX is the maximum amplitude of the normalized signal. Higher PSNR indicates better reconstruction.

Implementation. We train four model variants, STeP-C-tiny, small, medium, and large, with residual latent dimensions $z = 3 2 , 6 4 , 1 2 8$ , and 256, respectively. Unless otherwise noted, training and evaluation use sequences of $T = 1 0$ consecutive DAS frames, with each frame spanning 512 sensing channels. Training uses the Adam optimizer [13] with the rate-distortion objective described in Section 3.1, with $\lambda _ { r } = 0 . 0 0 1 , \lambda _ { d } = 1$ , and $\lambda _ { \mathrm { t p } } = 0 . 1$ . All experiments run on an NVIDIA RTX 6000 Ada GPU (48 GB) and AMD Threadripper PRO 7965WX CPU. Preprocessing follows Section 3.2. For data preprocessing, NJCFA data undergo phase unwrapping followed by a 5 Hz Chebyshev Type I high-pass filter to remove low-frequency background drift. DADAS and PFT data are band-pass filtered from 1 to 200 Hz and normalized using a tanh transform with a fixed 99th-percentile scale. For streaming deployment, the band-pass filter is applied statefully: the final rows of each chunk are prepended to the next as warm-up history, eliminating edge artifacts that arise when filtering each incoming window independently.

## 4.2. STeP-C Rate-Distortion Performance

Figure 5(a) reports the rate-distortion performance of STeP-C variants on NJCFA and DADAS. Across both datasets, increasing the residual latent dimension z improves reconstruction quality at the cost of higher BPS, providing a consistent capacity-rate-distortion trade-off across model variants. On NJCFA, STeP-C-small achieves 34.51 dB at 0.436 BPS, while on DADAS the same model variant reaches 41.13 dB at 0.424 BPS. The higher PSNR observed on DADAS at comparable bitrates reflects differences in the signal statistics of the two datasets, while the overall trend remains consistent across deployments.

Figure 4 shows qualitative reconstruction results on DADAS. On trained channels (band offset 6,000 to 6,512) (4a), the reconstruction closely matches the original, preserving dominant spatial features and temporal structure. On unseen channels (band offset 2,000 to 2,512) (4b), the model still recovers the primary signal structures despite not being trained on this channel range, with some smoothing of fine spatial detail. The full-band evaluation in Section 4.6 shows a similar trend, indicating that a model trained on a localized channel range can remain effective when applied to other portions of the same acquisition system, although reconstruction quality varies with channel location.

![](images/a603570904683d28d93204e0a87a8c030527c7abd8897f86267fcab0848781f8.jpg)

![](images/90d54766ad28978153cee2f762f6188888316c57b6bc23cd2e4f5bc07a15a19d.jpg)  
(a) On trained channels, band offset 6,000 to 6,512

![](images/2843e3cc1971477edd18ce70925d118c0ecd754bf5264e60a574c9b74b42c11e.jpg)

![](images/f221e9ab6f1e2bb9ab9a403654924d7b99bf920dc5db7a7c55e67cccb8eb5160.jpg)  
(b) On unseen channels, band offset 2,000 to 2,512  
Figure 4. Qualitative reconstruction results on DADAS. (a) Trained channel range. (b) Channel range excluded from training.

## 4.3. Comparison with Conventional Codecs

Figure 5(b) compares the rate-distortion performance of STeP-C with JPEG 2000, H.264, MP3, DASPack [22], and EnCodec [8] on NJCFA. STeP-C provides operating points below 0.5 BPS while maintaining comparatively high reconstruction quality. At 0.436 BPS, STeP-C achieves 34.51 dB PSNR. EnCodec is the only other method evaluated below 0.5 BPS, but its PSNR remains near 32 dB across the tested bitrate range. DASPack operates over a relatively narrow bitrate range of approximately 1.1–1.5 BPS, with reconstruction quality increasing sharply across its tested operating points, but it does not cover the sub-0.5 BPS regime. At its lowest tested bitrate, H.264 reaches 27.03 dB at 0.68 BPS, more than 7 dB below STeP-C despite using a higher bitrate. JPEG 2000 does not produce an operating point below 1.80 BPS in our evaluation, where it achieves 33.61 dB, while STeP-C reaches 37.48 dB at 1.01 BPS. MP3 achieves 28.62 dB at 1.10 BPS and remains below STeP-C over the evaluated range. In the practical low-bitrate regime below 0.5 BPS, STeP-C is the only evaluated method that operates in this regime while maintaining high reconstruction fidelity.

## 4.4. Architecture Ablations

Effect of the temporal predictor. Figure 6(a) evaluates the contribution of temporal prediction on NJCFA. At approximately 37.25 dB PSNR, the no-predictor baseline requires 1.53 BPS, while STeP-C achieves comparable quality at 1.13 BPS, corresponding to a 26% bitrate reduction. Across the evaluated operating points, the predictor consistently shifts the rate-distortion curve toward lower bitrates, showing that temporal prediction provides a substantial compression benefit without sacrificing reconstruction fidelity.

![](images/c79ea407737d4de9a1ad0277e5e175cda86b788ebe646590ddb8f55cbb89480f.jpg)

![](images/1b9b1b9601feba8b67d41c38f03157be745065f5c660378bc447014100031263.jpg)  
Figure 5. Rate-distortion performance across model variants and codec baselines. (a) Rate-distortion trade-off across STeP-C model variants on NJCFA and DADAS. (b) Comparison of STeP-C with conventional codecs (JPEG 2000, H.264, and MP3), a neural audio codec (EnCodec), and a DAS-specific codec (DASPack) on NJCFA.

Effect of temporal context length. Table 2 evaluates the effect of temporal context length T with all other settings fixed using the STeP-C-tiny model. Introducing temporal context with $T = 3$ reduces BPS from 0.281 to 0.207 while maintaining similar PSNR (33.15 versus 33.29 dB). Increasing T further to 5, 7, and 10 continues to reduce bitrate to 0.196, 0.189, and 0.184 BPS, respectively, while PSNR remains nearly unchanged. This indicates that most of the gain comes from introducing temporal conditioning, whereas longer context provides diminishing bitrate improvements.

Table 2. Effect of temporal context length T on NJCFA.
<table><tr><td> $\mathbf { T }$ </td><td>no predictor</td><td>3</td><td>5</td><td>7</td><td>10</td></tr><tr><td>PSNR</td><td>33.15</td><td>33.29</td><td>33.32</td><td>33.24</td><td>33.16</td></tr><tr><td>BPS</td><td>0.281</td><td>0.207</td><td>0.196</td><td>0.189</td><td>0.184</td></tr></table>

Effect of the temporal prediction loss weight $\lambda _ { \mathrm { t p } }$ . Figure 6(b) sweeps $\lambda _ { \mathrm { t p } } \in \{ 0 , 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ with $z = 2 5 6$ on NJCFA. The best tested value $\lambda _ { \mathrm { t p } } ~ = ~ 0 . 1$ achieves 37.48 dB at 1.01 BPS, outperforming the no-loss baseline $( \lambda _ { \mathrm { t p } } = 0$ , 37.39 dB at 1.03 BPS) by 0.09 dB while simultaneously reducing bitrate by 0.02 BPS. The improvement is modest in absolute PSNR but reflects a consistent shift of the operating point toward lower BPS. At $\lambda _ { \mathrm { t p } } = 1 0$ , BPS decreases to 0.93 at the cost of a 0.71 dB PSNR drop relative to $\lambda _ { \mathrm { t p } } ~ = ~ 0 .$ This effect is specific to larger latent dimensions $( z = 2 5 6 )$ ; at $z = 3 2$ the impact is negligible, suggesting the temporal prediction loss is most useful when the residual encoder has sufficient capacity to benefit from a well-trained predictor.

## 4.5. Real-Time Throughput

We first evaluate compression throughput in a simulated streaming setting, where pre-recorded DAS data are supplied to the compression pipeline at controlled rates. This isolates the computational throughput of the compressor from live acquisition and file handling overhead. Figure 7 summarizes the cumulative throughput gains achieved by each optimization applied to STeP-C-small model. The unoptimized baseline using CompressAI’s standard arithmetic coder achieves only 13 MB/s on encoding, far below the 324 MB/s real-time requirement. The custom entropy coder alone raises this to 392 MB/s, a 30× speedup. Adding FP16 mixed-precision inference raises encoding to 604 MB/s. Batching entropy coding across time steps reaches 647 MB/s. The 30× gain from the custom coder identifies standard entropy coding as the primary bottleneck in the unoptimized pipeline, while FP16 inference and batched coding provide additional throughput gains once this bottleneck is removed.

![](images/6f18a2928824bf23ff0a45ca0c58f44b445fe1e571a7246164c0034e18683738.jpg)

![](images/ec183b0f50fa0b3b987dc8b1377b1a24284d337ab39ba5bade70745cfe0b5935.jpg)  
Figure 6. Architecture ablations on NJCFA. (a) Rate-distortion curves of STeP-C and $\mathrm { S T e P \mathrm { - } C }$ (no predictor) across four operating points, showing that the temporal predictor shifts the curve left by up to 0.40 BPS at matched quality. (b) Effect of the temporal prediction loss weight $\lambda _ { \mathrm { t p } }$ on rate-distortion performance $( z \ =$ 256). The best tested value $\lambda _ { \mathrm { t p } } ~ = ~ 0 . 1$ (filled circle) achieves 37.48 dB at 1.01 BPS; excessive weighting $( \lambda _ { \mathrm { t p } } ~ = ~ 1 0 )$ reduces bitrate at the cost of a 0.71 dB PSNR drop.

![](images/d37489a541eecf426066ce522688ab4b728aadfc5aa4ba2c1346ff33dc18b62c.jpg)  
Figure 7. Cumulative throughput gains from successive optimizations of STeP-C-small model.

## 4.6. Live Testbed Evaluation

We next evaluate STeP-C on the PFT dataset under live acquisition. Unlike the simulated streaming evaluation in Section 4.5, this experiment measures the complete pipeline while data are continuously acquired from a DAS interrogator. The model is trained on a 256-channel sub-band and then applied without retraining to one hour of streamed data, comprising 1,111 consecutively acquired files of 3,240 time frames each, sampled at 1 kHz across 81,000 channels. Across the full acquisition, STeP-C compresses 1.166 TB of raw DAS data to 13.36 GB, corresponding to an

Table 3. Downstream detector consistency on DADAS.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Prediction consistency</td><td></td></tr><tr><td>Overall top-1 class agreement</td><td>83.01%</td></tr><tr><td>Overall top-5 class agreement</td><td>98.72%</td></tr><tr><td>High-confidence top-1 agreement</td><td>96.62%</td></tr><tr><td>Probability score correlation</td><td>0.851</td></tr><tr><td>Mean probability deviation</td><td>0.079</td></tr><tr><td>Detection consistency</td><td></td></tr><tr><td>Detection count ratio (recon / org)</td><td>92.13%</td></tr><tr><td>Detected class Jaccard similarity</td><td>100.00%</td></tr></table>

87.30× compression ratio and a 98.85% reduction in file size at 0.367 BPS. Reconstruction on the trained sub-band reaches 33.64 dB PSNR, while full-band reconstruction achieves 31.04 dB. The model therefore remains usable across the full acquisition footprint without channel-specific retraining, though reconstruction quality degrades measurably outside the trained channel range. Including streaming preprocessing, neural compression, parallel entropy coding, and file I/O, the end-to-end live pipeline achieves an average processing throughput of 535 MB/s over the live acquisition, exceeding the 324 MB/s throughput required for 1,000 Hz acquisition across 81,000 channels, with no dropped acquisition frames during the evaluated run.

## 4.7. Downstream Detector Consistency

We assess the impact of compression on the DADAS dataset using a consistency-based protocol in which the same pretrained detector [10] is applied independently to the original and reconstructed data. Before detector inference, both streams undergo the same detector-specific preprocessing. We invert the tanh normalization, apply the detector’s temporal filtering, and aggregate every 30 consecutive samples by averaging their absolute amplitudes. Quantitative results are summarized in Table 3. Figure 8 provides qualitative examples showing that representative event patterns remain visually consistent after reconstruction and lead to the same detector outputs. Across all event windows, the overall top-1 and top-5 class agreements are 83.01% and 98.72%, respectively. Restricting the comparison to high-confidence predictions $( p \geq 0 . 6 )$ increases top-1 agreement to 96.62%, showing substantially higher consistency among confident detector outputs. The corresponding prediction scores remain strongly correlated (r = 0.851), with a mean probability deviation of 0.079. Per-class top-1 and top-5 agreements for the six most frequent event classes are shown in Figure 9(a), while Figure 9(b) shows the corresponding top-1 confusion matrix. Using the same confidence threshold for event detection, the reconstructed data produce 92.13% as many detections as the original, while the set of detected event classes remains unchanged, yielding a Jaccard similarity of 100%. These results indicate that STeP-C largely preserves the pretrained detector’s output behavior, including high-confidence predictions and the set of event categories identified from the original data.

![](images/6371885019e00d16bceca6811fa4f99642d7a88896431d8d39419675d07af9cd.jpg)  
Figure 8. Qualitative comparison of downstream detections on DADAS. Each column shows an event class; rows show the labeled DAS sample and detector outputs from the original and reconstructed data.

![](images/ec6e6107b39e4cb8880e940437756f4afa0f4c460f4294fbb2102f1f6e059e81.jpg)

![](images/5afb971c2da93e5366aad24648a22bb702e7dfae5e002a03ecd60e2a2fd67561.jpg)  
Figure 9. Downstream event detection consistency between original and reconstructed data streams on DADAS. (a) Per-class top-1 (blue) and top-5 (green) agreement for the six most frequent event classes. (b) Class confusion matrix of top-1 predictions between original and reconstructed streams for the same classes.

## 5. Conclusion

We presented STeP-C, a neural compression framework for DAS data, and demonstrated its end-to-end deployment under live acquisition. The system achieves an average throughput of 535 MB/s with 87.30× compression on continuously streamed interrogator data and 31.04 dB full-band PSNR. Across offline evaluations, STeP-C provides favorable rate-distortion performance in the low-bitrate regime, while temporal prediction consistently reduces the bitrate required for a given reconstruction quality. Downstream evaluation further shows strong consistency between detector outputs on original and reconstructed data, reaching 96.62% high-confidence class agreement and 100% Jaccard similarity over detected event classes. These results demonstrate that learned predictive compression can support highratio, real-time DAS streaming while retaining both reconstruction quality and downstream utility. Future work will explore spatial representations that generalize more uniformly across fiber locations and extend downstream validation to additional tasks such as event localization and arrival-time picking.

## Acknowledgments

We acknowledge the use of LLMs to help write dataprocessing and visualization code and to improve the wording of the manuscript.

## References

[1] Jonathan B Ajo-Franklin, Shan Dou, Nathaniel J Lindsey, Inder Monga, Chris Tracy, Michelle Robertson, Veronica Rodriguez Tribaldos, Craig Ulrich, Barry Freifeld, Thomas Daley, et al. Distributed acoustic sensing using dark fiber for near-surface characterization and broadband seismic event detection. Scientific reports, 9(1):1328, 2019. 1

[2] Johannes Balle, David Minnen, Saurabh Singh, Sung Jin´ Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In International Conference on Learning Representations, 2018. 1, 3, 4

[3] Jean Begaint, Fabien Racap´ e, Simon Feltman, and Akshay´ Pushparaja. CompressAI: a PyTorch library and evaluation platform for end-to-end compression research. arXiv preprint arXiv:2011.03029, 2020. 4

[4] Yangkang Chen, Omar M. Saad, Yunfeng Chen, and Alexandros Savvaidis. Deep learning for seismic data compression in distributed acoustic sensing. IEEE Transactions on Geoscience and Remote Sensing, 63:1–14, 2025. 3

[5] Zhengxue Cheng, Heming Sun, Masaru Takeuchi, and Jiro Katto. Learning image and video compression through spatial-temporal energy compaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10071–10080, 2019. 3

[6] Abdelaziz Djelouah, Joaquim Campos, Simone Schaub-Meyer, and Christopher Schroers. Neural inter-frame compression for video coding. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6420–6428, 2019. 1, 3

[7] Bin Dong, Alex Popescu, Veronica Rodr ´ ´ıguez Tribaldos, Suren Byna, Jonathan Ajo-Franklin, and Kesheng Wu. Realtime and post-hoc compression for data from distributed acoustic sensing. Computers & Geosciences, 166:105181, 2022. 3, 4

[8] Alexandre Defossez, Jade Copet, Gabriel Synnaeve, and´ Yossi Adi. High fidelity neural audio compression. Transactions on Machine Learning Research, 2023. 6

[9] Hayden Gemeinhardt, Jyotsna Sharma, and Michael Kastoryano. Quantum-inspired workflow for processing distributed fiber-optic sensor data. Scientific Reports, 16(1): 14972, 2026. 3

[10] Shaobo Han, Ming-Fang Huang, Tingfeng Li, Jian Fang, Zhuocheng Jiang, and Ting Wang. Deep learning-based intrusion detection and impulsive event classification for distributed acoustic sensing across telecom networks. Journal ofLightwave Technology, 42(12):4167–4176, 2024. 1, 8

[11] Ming-Fang Huang, Shaobo Han, Glenn A Wellbrock, Tiejun J Xia, Chaitanya Narisetty, Milad Salemi, Yuheng Chen, James M Moore, Philip N Ji, Giovanni Milione, et al. Field trial of cable safety protection and road traffic monitoring over operational 5G transport network with fiber sensing and on-premise AI technologies. In OptoElectronics and

Communications Conference, pages T5A–8. Optica Publish ing Group, 2021. 1

[12] Ezra Ip, Jian Fang, Yaowen Li, Qiang Wang, Ming-Fang Huang, Milad Salemi, and Yue-Kai Huang. Distributed fiber sensor network using telecom cables as sensing media: technology advancements and applications. Journal of optical communications and networking, 14(1):A61–A68, 2022. 1

[13] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 6

[14] Tingfeng Li, Yuheng Chen, Ming-Fang Huang, Shaobo Han, and Ting Wang. Vehicle run-off-road event automatic detection by fiber sensing technology. In 2021 Optical Fiber Communications Conference and Exhibition (OFC), pages 1–3. IEEE, 2021. 1

[15] Jianping Lin, Dong Liu, Houqiang Li, and Feng Wu. M-lvc: Multiple frames prediction for learned video compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3

[16] Nathaniel J Lindsey and Eileen R Martin. Fiber-optic seis mology. Annual review of earth and planetary sciences, 49 (1):309–336, 2021. 1

[17] Nathaniel J Lindsey, T Craig Dawe, and Jonathan B Ajo-Franklin. Illuminating seafloor faults and ocean dynamics with dark fiber distributed acoustic sensing. Science, 366 (6469):1103–1107, 2019. 1

[18] Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, and Zhiyong Gao. DVC: An End-To-End Deep Video Compression Framework . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10998–11007, Los Alamitos, CA, USA, 2019. IEEE Com puter Society. 1, 3

[19] Yang Ma, Lingsen Meng, and Yen-Yu Lin. Distributed acoustic sensing data compression for seismological appli cations via compressive sensing. Geophysical Journal International, 246(3):ggag288, 2026. 3

[20] David Minnen, Johannes Balle, and George Toderici. Joint´ autoregressive and hierarchical priors for learned image compression. In Proceedings ofthe 32nd International Con ference on Neural Information Processing Systems, page 10794–10803, Red Hook, NY, USA, 2018. Curran Associates Inc. 3

[21] Yiyu Ni, Marine A. Denolle, Qibin Shi, Bradley P. Lipovsky, Shaowu Pan, and J. Nathan Kutz. Wavefield reconstruction of distributed acoustic sensing: Lossy compression, wavefield separation, and edge computing. Journal of Geophysical Research: Machine Learning and Computation, 1(3): e2024JH000247, 2024. 3

[22] Aleix Segu´ı, Arantza Ugalde, Andreas Fichtner, Sergi Ventosa, and Josep Ramon Morros. DASPack: controlled data compression for distributed acoustic sensing. Geophysical Journal International, 244(1):ggaf397, 2026. 3, 4, 6

[23] Xingliang Shen, Huan Wu, Kun Zhu, Huanhuan Liu, Yujia Li, Hua Zheng, Jialong Li, Liyang Shao, Perry Ping Shum, and Chao Lu. Fast and storage-optimized compressed domain vibration detection and classification for distributed acoustic sensing. Journal of Lightwave Technology, 42(1): 493–499, 2024. 3

[24] Jiayao Sun, Deyu Xu, Jingming Zhang, Shanfan Lu, Xiaobing Shi, Peide Zhang, Qi Mu, Chaoxian Qiu, and Liyang Shao. ANS-coded high-ratio data compression for a distributed acoustic sensing system. Optics Express, 33(22): 46969–46987, 2025. 3

[25] Honghui Wang, Xike Yang, Shangkun Zeng, Tong Liu, Xiang Wang, and Yiru Wang. Deep learning model for fiber optic distributed acoustic sensing seismic data compression based on autoencoder and recurrent neural networks. Appl. Opt., 63(20):5303–5315, 2024. 3

[26] Yixuan Wang, Junfeng Jiang, Kun Liu, Mingjiang Zhang, Shuang Wang, Tianhua Xu, Xuezhi Zhang, Zhenyang Ding, and Tiegen Liu. Ultra compression-ratio distributed acoustic sensing system via combination of envelope extraction and compressed sensing. Optics Express, 33(5):9854–9864, 2025. 3

[27] Glenn A Wellbrock, Tiejun J Xia, Ming-Fang Huang, Yuheng Chen, Milad Salemi, Yue-Kai Huang, Philip Ji, Ezra Ip, and Ting Wang. First field trial of sensing vehicle speed, density, and road conditions by using fiber carrying high speed data. In 2019 Optical Fiber Communications Conference and Exhibition (OFC), pages 1–3. IEEE, 2019. 1

[28] Zhongwen Zhan. Distributed acoustic sensing turns fiberoptic cables into sensitive seismic antennas. Seismological Research Letters, 91(1):1–15, 2020. 1