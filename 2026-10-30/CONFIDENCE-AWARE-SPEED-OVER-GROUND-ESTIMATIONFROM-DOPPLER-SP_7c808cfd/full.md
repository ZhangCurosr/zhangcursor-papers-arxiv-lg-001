# CONFIDENCE-AWARE SPEED-OVER-GROUND ESTIMATIONFROM DOPPLER SPECTRAVIA RECONSTRUCTION-BASED REPRESENTATION LEARNING

Malte Stelzer<sup>⋆</sup> Marvin Sach<sup>⋆</sup> Timo Bartels<sup>⋆</sup> Dinu Mihailescu-Stoica<sup>†</sup> S¨oren Kohnert<sup>†</sup> Florian Poprawa<sup>†</sup> Marc Zeller<sup>†</sup> Tim Fingscheidt<sup>⋆</sup>

<sup>⋆</sup>Technische Universit¨at Braunschweig, Germany {firstname.lastname, t.fingscheidt}@tu-bs.de

## ABSTRACT

Reliable speed-over-ground estimation from ground-facing Doppler spectra is essential for mobile platforms such as railway systems. Classical signal-processing approaches can provide accurate estimates under favorable conditions, but often degrade in challenging conditions. We propose confidenceaware speed-over-ground estimation trained on rail-domain data that estimates speed over ground and frame-level confidence from Doppler spectra with several novel contributions: First, we transfer the idea of exploiting a reconstruction task by an autoencoder for performance estimation from previous work to Doppler RaDAR. Second, we append a speed-overground estimation head operating on the autoencoder’s latent space features. Third, we omit the reconstruction decoder during inference but add a low-cost dedicated confidence head for frame-wise error-bound estimation. Fourth, we reveal how noise augmentation training can provide high robustness to the method. Experiments show that our approach improves speed estimation accuracy over classical baselines while providing informative confidence estimates that indicate the reliability of individual predictions, even under adverse signal conditions.

Index Terms— RaDAR, speed over ground, automated vehicles, spectrum reconstruction, confidence

## 1. INTRODUCTION

Ego-velocity estimation is essential in both automotive and railway applications, particularly for automated driving, where highly precise velocity information is required. While automotive systems predominantly rely on inertial measurement units and wheel-speed sensors for ego-velocity estimation [1], RaDAR-based approaches have also been investigated in the automotive domain [2]. In railway applications, sensor configurations include ground-facing Doppler RaDAR, which estimates the train velocity from the Doppler frequency shift in the reflected RaDAR waves from the track bed [3]; this task is commonly referred to as speed-overground estimation. Reflected RaDAR waves from the train bed can, however, be afected by surface conditions, weather, contamination, or other environmental influences [4].

Although related in terminology, ego-motion and position estimation in robotics constitute a diferent problem setting and should not be confused with the speed-over-ground estimation considered in this work. Such approaches typically <sup>†</sup>Siemens Mobility GmbH, Berlin, Germany {firstname.lastname}@siemens.com

rely on 3D RaDAR data and exploit spatial information such as range, azimuth, and elevation. In contrast, speed-overground estimation is based solely on the Doppler spectrum and therefore lacks the explicit 3D spatial information required by these methods [5, 6, 7, 8, 9], making them not directly applicable to our setting.

Since speed-over-ground estimates can degrade under disturbed measurement conditions, a frame-level confidence measure that reflects the current measurement conditions is needed to identify potentially unreliable predictions, particularly in safety-critical systems [5, 10, 11, 12, 13].

Auxiliary tasks have been shown to improve primary-task performance by encouraging the model to learn more informative shared representations [14, 15, 16]. Reconstruction-based auxiliary tasks are particularly attractive, as the original input provides a directly available reconstruction target and the reconstruction error can serve as an indicator of prediction confidence or uncertainty [17, 18, 19].

In this work, we adapt the idea of [17] to confidence-aware speed-over-ground estimation from RaDAR Doppler spectra. We transfer the autoencoder-based performance-estimation concept to Doppler RaDAR, but use reconstruction of clean spectra<sup>1</sup> only as an auxiliary training task and omit the decoder during inference. A dedicated confidence head shares encoder features with the new velocity head to provide framewise error-bound estimates. We further augment the input spectra with artificial noise to improve robustness under adverse signal conditions. Unlike [17], we jointly train the encoder, decoder, and velocity head first, followed by separate optimization of the confidence head.

## 2. METHOD

## 2.1. Architecture

Our proposed approach for confidence-aware speed-overground estimation is illustrated in Figure 1, were we transform an I/Q signal x(n) into a velocity signal ˆv<sub>ℓ</sub> and a corresponding error bound $\hat { \delta } _ { \ell }$ on frame level. We follow the concept introduced in [17], but adapt the reconstruction task into an auxiliary training objective and omit it during inference.

![](images/64231ef22d7c9aa56b4bd3a2e85d17f45b78374378939ceb12b850fa8b955e5a.jpg)  
Fig. 1: Our proposed speed-over-ground estimation with confidence estimation. Inference: Solid signal paths and blocks; training: also dashed ones.

I/Q Data and Preprocessing: Doppler RaDAR signals’ I/Q format consists of the in-phase (I) and quadrature (Q) components of the received signal and jointly encode its amplitude and phase information. The complex-valued $\mathrm { I / Q }$ signal $x ( n ) \in \mathbb { C } ,$ with integer sample index $n \in \mathbb { Z } ,$ , is segmented into frames $\mathbf { x } _ { \ell } \in \mathbb { C } ^ { N } , \ell \in \mathcal { L } = \{ 1 , 2 , \dots , L \}$ , with frame length N using a non-overlapping rectangular window. A discrete Fourier transform (DFT) is then applied to obtain the spectrum $\mathbf { X } _ { \ell } ~ = ~ ( X _ { \ell , k } ) ~ \in ~ \mathbb { C } ^ { K }$ of frame $\ell ,$ where $K = 2 0 4 8 \geq N$ denotes the number of frequency bins with index $k \in \mathcal { K } = \{ 0 , . . . , K - 1 \}$ . If $N < K ,$ , zeros are padded.

Spectrum Reconstruction: The encoder processes the complex DFT spectrum as two real-valued channels, i.e., the real and imaginary parts, with input size $\mathbb { R } ^ { 2 \times K }$ . The encoder consists of three 1D convolutional blocks that reduce the spectral dimension from $K = 2 0 4 8$ to 1024, 512, and finally 256, while increasing the channel dimension to 32, 64, and 128. This yields intermediate features $\mathbf { z } _ { \ell } ^ { \prime } \in \mathbb { R } ^ { 1 2 8 \times 2 5 6 }$ , which retain a position-preserving spectral representation of the input. These features are further compressed by global average pooling and a linear projection into a compact latent bottleneck vector of dimension $\mathbf { z } _ { \ell } \in \mathbb { R } ^ { 1 2 8 }$ . For reconstruction, this bottleneck representation is projected back to a feature map of size 128 × 256 and processed by three upsampling convolutional blocks with internal dimensions $1 2 8 \times 5 1 2 , 6 4 \times 1 0 2 4$ and 32 × 2048. A final $1 \times 1$ convolution reconstructs the two output channels corresponding to the real and imaginary parts of the reconstructed spectrum $\widehat { \mathbf { X } } _ { \ell } = ( \widehat { X } _ { \ell , k } ) \in \mathbb { C } ^ { \hat { K } }$

Velocity Estimation: The velocity head applies an additional 1D convolutional block to $\mathbf { z } _ { \ell } ^ { \prime } ,$ , followed by dropout and a 1 × 1 convolution that reduces the channel dimension to one. The resulting one-dimensional feature sequence is flattened and mapped by a final linear layer to the sequence of scalar velocity estimates $\hat { v } _ { \ell } , \ell \in \mathcal { L }$

Confidence Estimation: The confidence head is architecturally identical to the velocity head, with the only diference being a Softplus activation function [20] at the output to ensure non-negative predictions. It estimates an upper error bound $\hat { \delta } _ { \ell }$ corresponding to the 95th percentile of the absolute velocity error. Thus, for a well-calibrated model, approximately 95% of the absolute velocity errors are expected to lie below the predicted error bound $\hat { \delta } _ { \ell } .$ . Since smaller values of $\hat { \delta } _ { \ell }$ indicate a tighter expected error bound, they correspond to higher confidence in the velocity estimate.

While the overall processing pipeline is inspired by [17], we employ a more compact network architecture than imagebased approaches, since our inputs are one-dimensional Doppler spectra and our targets comprise a reconstructed spectrum and a scalar velocity, resulting in substantially lower input and output dimensionality. We further incorporate a dedicated module for confidence estimation and get rid of the comutationally demanding compression and reconstruction decoding in inference (∼ 10 times less FLOPs).

## 2.2. Training

Reference Velocity: The reference velocity v<sub>ℓ</sub> is created by fusing odometrical pulse generators (OPGs) mounted on two wheel axels of a wagon together with a tight coupling (global navigation satellite system (GNSS)/inertial measurement unit (IMU)) localization system and an optical speed correlation sensor. The wheel diameters are auto-calibrated via the GNSS velocity. To detect and exclude phases containing slip and slide efects<sup>2</sup> of the OPGs, their velocity estimates are compared to each other and to the velocities provided by GNSS and the optical sensor. The optical sensor improves the fused velocity estimate to reference-level accuracy; however, this processing is not available at runtime.

Losses: For training our network, we employ a loss function consisting of two components. To learn to reconstruct the spectrum, we use as first loss component [21]

$$
\begin{array} { r } { J _ { \ell } ^ { \mathrm { s p e c } } = \frac { 1 } { K } \sum _ { k \in { \mathcal K } } ( 1 - \beta ) \cdot ( | \widehat { X } _ { \ell , k } | ^ { c } - | \overline { { X } } _ { \ell , k } | ^ { c } ) ^ { 2 } } \\ { + \beta \cdot | | \widehat { X } _ { \ell , k } | ^ { c } e ^ { j \varphi _ { \ell , k } } - | \overline { { X } } _ { \ell , k } | ^ { c } e ^ { j \overline { { \varphi } } _ { \ell , k } } | | ^ { 2 } , } \end{array}\tag{1}
$$

where $\overline { { \varphi } } _ { \ell , k }$ and $\hat { \varphi } _ { \ell , k }$ denote the phases of $\overline { { \boldsymbol X } } _ { \ell , k }$ and $\widehat { X } _ { \ell , k }$ , respectively. Note that target spectra hold $\overline { { \boldsymbol X } } _ { \ell , k } = \boldsymbol X _ { \ell , k }$ . Furthermore, $\beta \in [ 0 , 1 ]$ is a weighting factor, and c denotes an amplitude compression factor. The second loss component is the mean absolute error (MAE)

$$
J _ { \ell } ^ { \mathrm { v e l } } = | \hat { v } _ { \ell } - v _ { \ell } |\tag{2}
$$

between the estimated and reference velocities. Both loss components are combined into the overall loss function

$$
J _ { \ell } = ( 1 - \alpha ) J _ { \ell } ^ { \mathrm { s p e c } } + \alpha J _ { \ell } ^ { \mathrm { v e l } } ,\tag{3}
$$

where $\alpha \in [ 0 , 1 ]$ balances the spectral reconstruction and velocity estimation losses. Unlike [17], we do not initialize a two-stage procedure with velocity estimation, as this simpler task could yield a less informative $\mathbf { z } _ { \boldsymbol { \ell } } ^ { \prime }$ and limit reconstruction performance. Instead, the encoder, reconstruction decoder, and velocity head are trained jointly first, then frozen while only the confidence head is optimized using the pinball loss [22]

$$
\begin{array}{c} \begin{array} { r } { J _ { \ell } ^ { \mathrm { c o n f } } = \left\{ \tau \left( \left| \hat { v } _ { \ell } - v _ { \ell } \right| - \delta _ { \ell } \right) , \quad \left| \hat { v } _ { \ell } - v _ { \ell } \right| \ge \delta _ { \ell } , \right.} \\ { \left( 1 - \tau \right) \left( \delta _ { \ell } - \left| \hat { v } _ { \ell } - v _ { \ell } \right| \right) , \quad \left| \hat { v } _ { \ell } - v _ { \ell } \right| < \delta _ { \ell } , } \end{array}   \end{array}\tag{4}
$$

with the quantile $\tau \in [ 0 , 1 ]$ . Learning rate, weight decay, and data are same as in the first stage.

Training Noise: To make spectrum reconstruction and velocity estimation more robust against degraded RaDAR spectra, we also add artificial noise to the spectra X<sub>ℓ</sub> during training. The signal power in frame ℓ is estimated as

$$
P _ { \ell } ^ { ( \mathrm { x } ) } = \frac { 1 } { K } \sum _ { k \in \mathcal { K } } \left| X _ { \ell , k } \right| ^ { 2 }\tag{5}
$$

For a chosen signal-to-noise ratio SNR (in dB), we then obtain the noise power

$$
\begin{array} { r } { P _ { \ell } ^ { ( \epsilon ) } = \frac { P _ { \ell } ^ { ( \mathrm { x } ) } } { 1 0 ^ { \mathrm { S N R } / 1 0 ~ \mathrm { d B } } } . } \end{array}\tag{6}
$$

The complex-valued noise $\boldsymbol { \epsilon } _ { \ell } = \left( \boldsymbol { \epsilon } _ { \ell , k } \right)$ with $\epsilon _ { \ell , k } \sim \mathcal { N } _ { \mathbb { C } } ( 0 , P _ { \ell } ^ { ( \epsilon ) } )$ is drawn from a complex normal distribution and then added independently in each time-frequency bin as

$$
\widetilde { X } _ { \ell , k } = X _ { \ell , k } + \epsilon _ { \ell , k } ,\tag{7}
$$

with $( { \tilde { X } } _ { \ell , k } ) = { \tilde { \mathbf { X } } } _ { \ell } .$ This frequency-domain noise model is equivalent to adding circularly symmetric complex white Gaussian noise before the $\mathrm { D F T }$ , up to a constant factor. The noise is not added to the targets $\overline { { \mathbf { X } } } _ { \ell } = \mathbf { X } _ { \ell }$ of the reconstruction, only to the input $\mathbf { X } _ { \ell }$ of the encoder. The SNRs and noise schedule used in training are described in Section 3.

## 3. EXPERIMENTAL SETUP

Dataset: The acquisition chain uses a continuous-wave patch antenna at $f _ { c } = 2 4 ~ \mathrm { G H z }$ . The ground-reflected RF signal is mixed with the transmitted signal and a $9 0 °$ phase-shifted copy to obtain the in-phase (I) and quadrature (Q) components. The $\mathrm { I / Q }$ imbalance is below 2% and neglected. Both signals are high-pass filtered to remove biases, low-pass filtered for anti-aliasing, and A/D-converted at $f _ { s } ~ = ~ 8 3 . 6 8 2$ kHz. The antenna is tilted by $\vartheta = 3 7 ^ { \circ }$ relative to the horizontal and has a half-power beamwidth of $5 ^ { \circ }$

We split the data into a training set $\mathcal { D } ^ { \mathrm { t r a i n } }$ with 200k frames, a validation set $\mathcal { D } ^ { \mathrm { v a l } }$ with 20k frames, and a test set $\mathcal { D } ^ { \mathrm { t e s t } }$ with 20k frames. Note that each set contains frames from disjoint routes, thereby covering diferent weather and ground conditions. The training set comprises 20 routes, while the validation and test sets each comprise 10 routes. Frames were assigned such that the velocity distributions are as similar as possible across the sets.

Validation/Test Noise: To evaluate the robustness of the velocity estimators under noisy conditions, additive noise is applied also for validation and test purposes to the complex spectra of the data (7). Signal power estimation follows (5). Metrics: As the primary evaluation metric, we use the MAE between ˆv<sub>ℓ</sub> and $v _ { \ell } ,$ consistent with (2). It is reported on the validation and test sets for clean and noisy data, both per noise level and averaged across all considered noise levels. MAE and standard deviation are given in $\mathrm { m } / \mathrm { s }$

Parameterization: For our experiments, we use a frame length of $N = 1 2 8 7$ , a compression factor of $c = 0 . 3$ following [21], and a reconstruction weight of $\beta = 0 . 3$ . Noise is introduced via a curriculum schedule with probabilities of $0 . 3 , 0 . 5 ,$ and 0.7 and SNR ranges of [15, 25], [5, 20], and [−5, 15] dB starting from epochs 11, 26, and 41, respectively. The network is trained for 80 epochs with an initial learning rate of $1 0 ^ { - 3 }$ , halved every 15 epochs, weight decay of $1 0 ^ { - 4 }$ , velocityhead dropout of $0 . 1 ,$ and quantile $\tau = 0 . 9 5$

Reference Methods: As classical baselines, we consider three frequency-domain estimators. Peak picking (PP) provides the most direct approach by selecting the frequency bin $k _ { \ell } ^ { \mathrm { m a x } }$ with maximum spectral magnitude [23, p. 654], while ignoring seven bins around DC to suppress residual low-frequency bias. Then, $\begin{array} { r } { v _ { \ell } = \frac { \left( k _ { \ell } ^ { \mathrm { m a x } } - \frac { K } { 2 } \right) \frac { f _ { s } } { K } c _ { 0 } } { 2 f _ { c } \cos \left( \vartheta \right) } } \end{array}$ , where $c _ { 0 }$ is the speed of light in vacuum. Center of mass (CMA) [2] estimates the Doppler frequency from the spectral mass within an adaptively selected signal band, reducing sensitivity to isolated spectral peaks. Cross-correlation (XCA) [2] smoothes the Doppler spectrum, constructs a Gaussian reference from the coarse peak estimate and antenna beamwidth, and selects the frequency shift that maximizes the correlation with this template.

## 4. EXPERIMENTAL VALIDATION ANDDISCUSSION

Table 1 summarizes performance on $\mathcal { D } ^ { \mathrm { v a l } }$ and $\mathcal { D } ^ { \mathrm { t e s t } }$ and compares our method with the reference methods. We evaluate diferent α in (3) and additive noise training $( 7 ) ; \alpha = 1 . 0$ disables reconstruction training. Mean and standard deviation of the MAE are reported. For $" \mathrm { + } \ \mathrm { n o i s e } ^ { \mathrm { 3 } }$ , both are computed across SNRs of −5, 0, 5, 10, 15, and 20 dB; for clean data and individual SNRs, they are computed across samples. The average rank is also reported on $\bar { \mathcal { D } } ^ { \mathrm { t e s t } }$

On $\mathcal { D } ^ { \mathrm { v a l } }$ , we observe that XCA [2] is the best among all baseline methods, with MAE only slightly depending on SNR. Our approaches are top in clean conditions, however, with a severe degradation in noisy conditions. Our method with noise augmentation training, however, very much improves in both noise conditions. Among our configurations, we choose $\alpha = 1 . 0 ,$ as it delivers two 1st ranks and one 2nd on $\mathcal { D } ^ { \mathrm { v a l } }$

On $\mathcal { D } ^ { \mathrm { t e s t } }$ , the overall ranks are very similar to $\mathcal { D } ^ { \mathrm { v a l } } ;$ : XCA [2] again is best among baselines and 2nd at very harsh noise (−10 dB). Our approaches with noise augmentation training show overall top performance in all four conditions. Our method with $\alpha = 1 . 0 ,$ i.e., without reconstruction loss (1) in $( 3 )$ , secures the first rank among all from clean data unto $- 1 0 \ d B$ test noise. The ranking shows another strong result for $\alpha \in \{ 0 . 5 , 0 . 6 \}$ , achieving the second-best rank. So far, we consider SNRs below −10 dB as out-of-operation range, errors increase above $0 . 4 ~ \mathrm { m / s }$ . Checking robustness towards −15 dB, however, the noise-augmented model with $\alpha = 0 . 6$ achieves an MAE of 2.23 $\mathrm { m } / \mathrm { s }$ versus 4.30 $\mathrm { m } / \mathrm { s }$ for $\alpha = 1 . 0 $

Figure 2 shows the velocity MAE of the three reference methods and our selected best-performing configuration across SNRs ranging from −10 to 20 dB, therefore illustrating the robustness and quality of the velocity prediction under diferent noise conditions. Our method shows almost constant MAE of about 0.15 m/s (0.54 km/h) over the entire SNR range unto −5 dB. At −10 dB, it degrades to the level of XCA [2](MAE of $0 . 3 4 ~ \mathrm { m / s ) }$ . In contrast, the performance of PP [23] deteriorates markedly below 5 dB, indicating a substantially higher sensitivity to noise. CMA [2] shows an even stronger dependence on the SNR, with its best performance around 10 dB and increasing errors toward both lower and higher SNRs. This may result from prominent spectral components, particularly around zero Doppler, shifting the center of mass from the velocity-related peak.

Figure 3 evaluates the robustness and quality of the predicted error bound $\hat { \delta } _ { \ell }$ of our best-performing model under different SNRs. Again, for each SNR, Gaussian noise is added to the entire $\mathcal { D } ^ { \mathrm { t e s t } }$ . The primary y-axis (green numbers) shows the percentage of frames for which the absolute velocity error holds $| \hat { v } _ { \ell } - v _ { \ell } | \le \hat { \delta } _ { \ell }$ , while the secondary y-axis (magenta numbers) reports the mean predicted error bound <sup>ˆ</sup>δ<sup>mean</sup> and half of its standard deviation using our best-performing model. Across all SNRs, the proportion of predictions within the estimated error bound, which was set to 95% in training, remains almost constant between 94% and 95%, demonstrating stable error-bound estimation even under noisy conditions. The only outlier is at −10 dB with 91%, an SNR not seen in training. The mean predicted error bound is approximately 0.38 m/s at 15 dB to 20 dB and increases to about 0.90 m/s at −10 dB. A comparatively large standard deviation, increasing with higher noise levels, is also observed. This is desirable in this context, as it indicates that the predicted error bound adapts to individual frames rather than remaining constant. Thus, error bound $\hat { \delta } _ { \ell }$ provides frame-dependent information about the confidence of the velocity estimate and may additionally reflect variations in signal quality.

Table 1: MAE results on $\mathcal { D } ^ { \mathrm { v a l } }$ and $\mathcal { D } ^ { \mathrm { t e s t } }$ : Average velocity MAE in $\mathrm { m } / \mathrm { s }$ across diferent noise conditions. For our models, we investigated noise augmentation training and vary α in (3). Best values are marked bold, second-best are underlined.
<table><tr><td>Method</td><td>Train. noise</td><td>α</td><td> $\mathcal { D } ^ { \mathrm { v a l } }$   $\mathrm { c l e a n }$ </td><td> $\mathcal { D } ^ { \mathrm { v a l } }$   $+ { \mathrm { ~ n o i s e } }$ </td><td> $\mathcal { D } ^ { \mathrm { v a l } }$   $+ - 1 0 ~ \mathrm { d B }$  noise|</td><td> $\mathcal { D } ^ { \mathrm { t e s t } }$  clean</td><td> $\mathcal { D } ^ { \mathrm { t e s t } }$   $+ { \mathrm { ~ n o i s e } }$ </td><td> $\mathcal { D } ^ { \mathrm { t e s t } }$  + -5 dB noise</td><td> $\mathcal { D } ^ { \mathrm { t e s t } }$   $+  { - } 1 0 ~  { \mathrm { d B } }$ </td><td> $\mathcal { D } ^ { \mathrm { t e s t } }$  noise |avg. rank ↓</td></tr><tr><td>PP [23]</td><td></td><td>-</td><td> $0 . 4 1 ^ { \pm 0 . 4 7 }$ </td><td> $0 . 4 5 ^ { \pm 0 . 0 6 }$ </td><td> $1 . 0 2 ^ { \pm 1 1 . 2 4 }$ </td><td> $0 . 4 2 ^ { \pm 0 . 4 3 }$ </td><td> $0 . 4 5 ^ { \pm 0 . 0 7 }$ </td><td> $0 . 6 0 ^ { \pm 6 . 3 4 }$ </td><td> $1 . 1 7 ^ { \pm 1 2 . 8 7 }$ </td><td>6.00</td></tr><tr><td>CMA [2]</td><td>1</td><td>-</td><td> $1 . 0 4 ^ { \pm 1 . 7 3 }$ </td><td> $1 . 3 0 ^ { \pm 1 . 5 4 }$ </td><td> $2 4 . 6 8 ^ { \pm 6 8 . 2 3 }$ </td><td> $0 . 9 9 ^ { \pm 1 . 5 1 }$ </td><td> $1 . 1 7 ^ { \pm 1 . 3 1 }$ </td><td> $4 . 0 9 ^ { \pm 2 6 . 5 6 }$ </td><td> $2 5 . 0 7 ^ { \pm 6 9 . 0 0 }$ </td><td>7.00</td></tr><tr><td>XCA [2]</td><td>-</td><td>-</td><td> $0 . 3 6 ^ { \pm 1 . 4 1 }$ </td><td> $0 . 3 6 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 3 8 ^ { \pm 1 . 5 2 }$ </td><td> $0 . 3 4 ^ { \pm 1 . 3 1 }$ </td><td> $0 . 3 4 ^ { \pm 0 . 0 0 }$ </td><td> $0 . 3 5 ^ { \pm 1 . 3 9 }$ </td><td> $\underline { { 0 . 3 7 } } ^ { \pm 1 . 5 9 }$ </td><td>3.50</td></tr><tr><td></td><td>no</td><td>1.0</td><td> $\mathbf { 0 . 1 4 ^ { \pm 0 . 2 0 } }$ </td><td> $9 . 6 2 ^ { \pm 7 . 9 1 }$ </td><td> $3 3 . 6 3 ^ { \pm 1 8 . 9 2 }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 2 0 } }$ </td><td> $9 . 6 3 ^ { \pm 7 . 9 1 }$ </td><td> $2 3 . 6 8 ^ { \pm 1 3 . 8 7 }$ </td><td> $3 3 . 9 3 ^ { \pm 1 9 . 2 8 }$ </td><td>6.75</td></tr><tr><td></td><td>no</td><td>0.9</td><td> $\mathbf { 0 . 1 4 ^ { \pm 0 . 2 0 } }$ </td><td> $1 3 . 0 2 ^ { \pm 1 2 . 4 2 }$ </td><td> $3 7 . 5 7 ^ { \pm 2 0 . 0 8 }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 1 9 } }$ </td><td> $1 3 . 1 8 ^ { \pm 1 2 . 5 4 }$ </td><td> $3 5 . 9 4 ^ { \pm 1 9 . 1 0 }$ </td><td> $3 8 . 1 7 ^ { \pm 2 0 . 9 3 }$ </td><td>8.25</td></tr><tr><td></td><td>no</td><td>0.8</td><td> $0 . 1 5 ^ { \pm 0 . 2 1 }$ </td><td> $1 1 . 0 1 ^ { \pm 1 1 . 9 9 }$ </td><td> $3 6 . 7 0 ^ { \pm 1 9 . 5 0 }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 2 2 } }$ </td><td> $1 1 . 1 6 ^ { \pm 1 2 . 1 4 }$ </td><td> $3 4 . 4 0 ^ { \pm 1 8 . 6 6 }$ </td><td> $3 7 . 2 0 ^ { \pm 2 0 . 1 7 }$ </td><td>7.50</td></tr><tr><td></td><td>yes</td><td>1.0</td><td> $0 . 1 5 ^ { \pm 0 . 1 9 }$ </td><td> $\mathbf { 0 . 1 8 ^ { \pm 0 . 0 2 } }$ </td><td> $\mathbf { 0 . 3 7 ^ { \pm 1 . 0 0 } }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 2 0 } }$ </td><td> $\mathbf { 0 . 1 8 ^ { \pm 0 . 0 2 } }$ </td><td> $\mathbf { 0 . 2 1 ^ { \pm 0 . 4 4 } }$ </td><td> $\mathbf { 0 . 3 6 ^ { \pm 0 . 9 9 } }$ </td><td>1.00</td></tr><tr><td>Ours</td><td>yes</td><td>0.9</td><td> $\mathbf { 0 . 1 4 ^ { \pm 0 . 1 8 } }$ </td><td> $\mathbf { 0 . 1 8 ^ { \pm 0 . 0 2 } }$ </td><td> $0 . 4 9 ^ { \pm 1 . 2 7 }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 2 1 } }$ </td><td> $\mathbf { 0 . 1 8 ^ { \pm 0 . 0 2 } }$ </td><td> $0 . 2 2 ^ { \pm 0 . 4 1 }$ </td><td> $0 . 4 6 ^ { \pm 1 . 1 0 }$ </td><td>2.50</td></tr><tr><td></td><td>yes</td><td>0.8</td><td> $0 . 1 5 ^ { \pm 0 . 2 0 }$ </td><td> $\mathbf { 0 . 1 8 ^ { \pm 0 . 0 3 } }$ </td><td> $0 . 5 0 ^ { \pm 1 . 3 6 }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 2 1 } }$ </td><td> $0 . 1 9 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 2 3 ^ { \pm 0 . 4 7 }$ </td><td> $0 . 4 8 ^ { \pm 1 . 2 2 }$ </td><td>3.50</td></tr><tr><td></td><td>yes</td><td>0.7</td><td> $\overline { { 0 . 1 5 } } ^ { \pm 0 . 1 8 }$ </td><td> $\mathbf { 0 . 1 8 ^ { \pm 0 . 0 2 } }$ </td><td> $0 . 4 4 ^ { \pm 1 . 0 4 }$ </td><td> $\mathbf { 0 . 1 5 ^ { \pm 0 . 2 0 } }$ </td><td> $\overline { { 0 . 1 9 } } ^ { \pm 0 . 0 2 }$ </td><td> $0 . 2 3 ^ { \pm 0 . 5 1 }$ </td><td> $0 . 4 3 ^ { \pm 1 . 0 9 }$ </td><td>2.75</td></tr><tr><td></td><td>yes</td><td>0.6</td><td> $0 . 1 5 ^ { \pm 0 . 1 8 }$ </td><td> $0 . 1 9 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 4 1 ^ { \pm 0 . 9 8 }$ </td><td> $0 . 1 6 ^ { \pm 0 . 2 6 }$ </td><td> $\overline { { 0 . 1 9 } } ^ { \pm 0 . 0 2 }$ </td><td> $0 . 2 3 ^ { \pm 0 . 4 3 }$ </td><td> $0 . 4 0 ^ { \pm 0 . 9 0 }$ </td><td>2.50</td></tr><tr><td></td><td>yes</td><td>0.5</td><td> $\overline { { 0 . 1 5 } } ^ { \pm 0 . 2 0 }$ </td><td> $\overline { { \bf 0 . 1 8 } } ^ { \pm 0 . 0 2 }$ </td><td> $0 . 4 3 ^ { \pm 1 . 2 7 }$ </td><td> $\overline { { \bf 0 . 1 5 ^ { \pm 0 . 2 0 } } }$ </td><td> $\overline { { 0 . 1 9 } } ^ { \pm 0 . 0 3 }$ </td><td> $0 . 2 2 ^ { \pm 0 . 4 6 }$ </td><td> $0 . 4 1 ^ { \pm 1 . 1 8 }$ </td><td>2.25</td></tr><tr><td></td><td>yes</td><td>0.4</td><td> $\underline { { 0 . 1 5 ^ { \pm 0 . 2 0 } } }$ </td><td> $\underline { { 0 . 1 9 } } ^ { \pm 0 . 0 3 }$ </td><td> $0 . 5 0 ^ { \pm 1 . 2 0 }$ </td><td> $0 . 1 6 ^ { \pm 0 . 2 0 }$ </td><td> $\overline { { 0 . 2 0 ^ { \pm 0 . 0 3 } } }$ </td><td> $\overline { { 0 . 2 4 } } ^ { \pm 0 . 4 6 }$ </td><td> $0 . 4 7 ^ { \pm 1 . 1 2 }$ </td><td>4.00</td></tr></table>

![](images/673c16d82f6fec1906a81c298aedc7a8090354c56fd23f3e5e0eaa599c76656e.jpg)  
Fig. 2: Velocity MAE results on noisy $\mathcal { D } ^ { \mathrm { t e s t } }$

![](images/ac60507a62dba4387c8d46e5cd2b10f67da32dbe4123afee6634bbdf2cdbe06a.jpg)  
Fig. 3: Our method: Percentage of velocity errors below error bound $\hat { \delta } _ { \ell }$ and mean error bound $\hat { \delta } ^ { \mathrm { m e a n } }$ on noisy $\mathcal { D } ^ { \mathrm { t e s t } } ; \hat { \delta } ^ { \mathrm { m e a n } } \pm \frac { 1 } { 2 } \sigma$ are shown for visualization purposes.

![](images/63453746841a718135365cece682e0c6bd03fdbac2676b7c040f8a6c1350429b.jpg)  
Fig. 4: Per-frame velocity error and predicted error bound: Sequence from $\mathcal { D } ^ { \mathrm { t e s \bar { t } } }$ at an SNR of 5 dB, including the velocity error smoothed using an IIR and FIR filter.

Figure 4 shows consecutive frames from the domain of $\mathcal { D } ^ { \mathrm { t e s t } }$ at an SNR of 5 dB. In addition to the per-frame velocity error and predicted error bound, the velocity error is smoothed using a first-order IIR filter with coeficient $\gamma = 0 . 9$ and an FIR filter with a non-causal rectangular impulse response of length $L ^ { \prime } = 9$ . The velocity error and the predicted error bound exhibit a closely matching temporal behavior.

The predicted bound is violated at only two frames $( \ell = 1 6$ and $\ell = 3 4 )$ , i.e., in 5% of the 40 displayed frames – adhering to our chosen $\tau = 0 . 9 5$ . Both smoothing approaches reduce the magnitude of the velocity error, but introduce some lag.

## 5. CONCLUSIONS

We presented a confidence-aware speed-over-ground estimation method for Doppler RaDAR. The proposed model consistently outperforms classical baselines across all investigated SNR levels, while the predicted frame-wise error bounds reliably reflect the actual estimation error. The results confirm robustness of our approach under adverse signal conditions.

## 6. ACKNOWLEDGEMENT

The research leading to these results is funded by the German Federal Ministry for Economic Afairs and Energy within the project “CONTROL – Controlling Risk for Highly Automated Transportation Systems Operating in Complex Open Environments”. The authors would like to thank the consortium for the successful cooperation.

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This study uses real RaDAR measurement data acquired by a sensor mounted underneath a train and directed toward the track bed. No human or animal subjects were involved in the study, and therefore no ethical approval was required.

## 8. REFERENCES

[1] H. Ragab, S. Givigi, and A. Noureldin, “Automotive Speed Estimation: Sensor Types and Error Characteristics from OBD-II to ADAS,” in Proc. of PLANS, Salt Lake City, UT, USA, Apr. 2025, pp. 124–130.

[2] K. K. M. Sharif, E. Hoare, L. Daniel, M. Antoniou, and M. Cherniakov, “Comparison of Adaptive Spectral Estimation for Vehicle Speed Measurement with Radar Sensors,” Sensors, vol. 17, no. 4, pp. 1–17, Apr. 2017.

[3] T. Reissland, B. Lenhart, J. Lichtblau, M. Sporer, R. Weigel, and A. Koelpin, “Evaluation of a Robust Correlation-Based True-Speed-Over-Ground Measurement System Employing a FMCW Radar,” International Journal of Microwave and Wireless Technologies, vol. 11, no. 7, pp. 686–693, Sep. 2019.

[4] T. Reissland, R. Weigel, A. Koelpin, and F. Lurz, “High-Accuracy True Speed-Over-Ground Measurement Approaches for Rail Vehicles: Comparison of Highly Accurate State of the Art True Speed-Over-Ground Measurement Approaches for Rail Vehicles,” IEEE Microwave Magazine, vol. 24, no. 12, pp. 60–73, Dec. 2023.

[5] J.-T. Huang, T. Huang, A. Rowe, and M. Kaess, “UN-RIO: Uncertainty-Aware Velocity Learning for Radar-Inertial Odometry,” arXiv 2604.13584, pp. 1–7, Apr. 2026.

[6] Y. Zhuang, B. Wang, J. Huai, and M. Li, “4D iRIOM: 4D Imaging Radar Inertial Odometry and Mapping,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3246–3253, June 2023.

[7] V.-J. Stironja, L. Petrovi´c, J. Perˇsi´c, I. Markovi´c, and<sup>ˇ</sup> I. Petrovi´c, “RAVE: A Framework for Radar Ego-Velocity Estimation,” in Proc. of MFI, Plzeˇn, Czech Republic, Sep. 2024, pp. 1–6.

[8] V. Kubelka, E. Fritz, and M. Magnusson, “Do We Need Scan-Matching in Radar Odometry?,” in Proc. of ICRA, Yokohama, Japan, May 2024, pp. 13710–13716.

[9] N. J. Abu-Alrub and N. A. Rawashdeh, “Radar Odometry for Autonomous Ground Vehicles: A Survey of Methods and Datasets,” IEEE Trans. on IV, vol. 9, no. 3, pp. 4275–4291, Mar. 2024.

[10] J. Sicking, M. Akila, J. D. Schneider, F. H¨uger, P. Schlicht, T. Wirtz, and S. Wrobel, “Tailored Uncertainty Estimation for Deep Learning Systems,” arXiv 2204.13963, pp. 1–37, Apr. 2022.

[11] Y. Geifman and R. El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject Option,” in Proc. of ICML, Long Beach, CA, USA, June 2019, pp. 2151–2159.

[12] T. Fingscheidt, H. Gottschalk, and S. Houben, Eds., Deep Neural Networks and Data for Automated Driving: Robustness, Uncertainty Quantification, and Insights Towards Safety, Springer International Publishing, Cham, 2022.

[13] M. Posner, M. Dazer, D. Lauer, R. Winkler-H¨ohn, M. Laporte, T. Herrmann, and M. K¨oppel, “A System for Train Condition Monitoring and Structural Health Assessment of Rail Vehicles,” in Proc. of IV, Detroit, MI, USA, June 2026, pp. 257–264.

[14] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “TransFuser: Imitation With Transformer-Based Sensor Fusion for Autonomous Driving,” IEEE Trans. on PAMI, vol. 45, no. 11, pp. 12878–12895, Nov. 2023.

[15] B. Jaeger, K. Chitta, and A. Geiger, “Hidden Biases of End-to-End Driving Models,” in Proc. of ICCV, Paris, France, Oct. 2023, pp. 8206–8215.

[16] M. Stelzer, T. Bartels, J. Bickerdt, V. P. Schomerus, J. Piewek, T. Bagdonat, and T. Fingscheidt, “A Generalized Waypoint Loss for End-to-End Autonomous Driving,” in Proc. of IV, Cluj-Napoca, Romania, June 2025, pp. 90–96.

[17] A. B¨ar, D. Kusuma, and T. Fingscheidt, “Improvements to Image Reconstruction-Based Performance Prediction for Semantic Segmentation in Highly Automated Driving,” in Proc. of CVPR-W, Vancouver, BC, Canada, June 2023, pp. 219–229.

[18] J. L¨ohdefink, J. Fehrling, M. Klingner, F. Huger, P. Schlicht, N. M. Schmidt, and T. Fingscheidt, “Self-Supervised Domain Mismatch Estimation for Autonomous Perception,” in Proc. of CVPR-W, virtual, June 2020, pp. 334–335.

[19] A. B¨ar, M. Klingner, J. L¨ohdefink, F. H¨uger, P. Schlicht, and T. Fingscheidt, “Performance Prediction for Semantic Segmentation by a Self-Supervised Image Reconstruction Decoder,” in Proc. of CVPR-W, New Orleans, LA, USA, June 2022, pp. 4399–4408.

[20] C. Dugas, Y. Bengio, F. B´elisle, C. Nadeau, and R. Garcia, “Incorporating Second-Order Functional Knowledge for Better Option Pricing,” in Advances in Neural Information Processing Systems, 2000, vol. 13, pp. 472–478.

[21] S. Braun and I. Tashev, “A Consolidated View of Loss Functions for Supervised Deep Learning-Based Speech Enhancement,” in Proc. of TSP, Brno, Czech Republic, July 2021, pp. 72–76.

[22] R. Koenker and J. Bassett, Gilbert, “Regression Quantiles,” Econometrica, vol. 46, no. 1, pp. 33–50, Jan. 1978.

[23] M. A. Richards, J. A. Scheer, and W. A. Holm, Eds., Principles of Modern Radar: Basic Principles, SciTech Publishing, 2010.