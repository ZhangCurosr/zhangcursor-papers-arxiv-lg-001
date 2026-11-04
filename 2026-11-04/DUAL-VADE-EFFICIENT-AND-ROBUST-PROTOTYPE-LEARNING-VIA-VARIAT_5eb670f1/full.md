# DUAL-VADE: EFFICIENT AND ROBUST PROTOTYPE LEARNING VIA VARIATIONAL DUAL-DOMAIN MIXTURE PRIORS

Wugeng Zheng<sup>1</sup>, Song Wang<sup>3</sup>, Jian Wang<sup>2,3</sup>

<sup>1</sup>College of Engineering, Northeastern University, Boston, MA, USA <sup>2</sup>Athinoula A. Martinos Center for Biomedical Imaging, Department of Radiology, Massachusetts General Hospital, Harvard Medical School, Boston, MA, USA <sup>3</sup>Computer Science, University of Central Florida, Orlando, FL, USA

## ABSTRACT

In this paper, we propose Dual-VaDE, an unsupervised generative framework for learning interpretable anatomical prototypes from unlabeled medical images. In contrast to previous variational methods that rely on a single spatial latent code, which often entangles global geometry with local textures and fails to capture their cross-domain dependencies, Dual-VaDE introduces a dual-domain Gaussian mixture prior that explicitly couples spatial anatomy with phase-preserving Fourier spectral representations. We derive an exact, closedform variational lower bound that fuses this joint prior into the decoder and optimizes the entire network end-to-end via an alternating EM formulation, avoiding heuristic two-stage clustering while ensuring superior training stability and robustness. Evaluated on OASIS-3 brain MRI, Dual-VaDE delivers substantially sharper and more plausible prototypes, achieves superior cluster separation with improvements of up to 94% over spatial-only baselines, exhibits markedly reduced variance across random seeds, and incurs negligible inference overhead (0.28 ms per image).<sup>1</sup>

Index Terms— Generative models, Prototype learning, Mixture prior, Fourier space

## 1. INTRODUCTION

High-dimensional medical imaging exhibits complex morphological variability that conventional unsupervised representation learning often fails to disentangle. Unlike supervised pipelines built on predefined categories or diagnostic labels, unsupervised distribution modeling can capture the continuous spectrum of anatomical heterogeneity directly from raw cohorts [1]. Deep generative models enable density estimation and unsupervised representation learning without manual annotations [2, 3, 4, 5]. Compared with discriminative clustering, generative methods can improve interpretability by decoding latent representations into anatomical prototypes—inspectable templates that support subsequent morphological analysis [6].

Variational autoencoders (VAEs) with Gaussian mixture priors [7, 8, 9] link each mixture component to a discrete morphological centroid, where decoding the mean yields a prototype. However, typical formulations use a single spatial latent vector, entangling global geometry (e.g., compartment volume) with local appearance (e.g., contrast and boundary texture). With unstructured diagonal or unimodal covariances, such priors fail to capture cross-scale geometry–texture dependencies, often producing blurred or degenerate prototypes. The Fourier domain offers a complementary view of the image manifold: low frequencies largely determine spectral energy, while phase preserves topology, boundaries, and edges [10]. Magnitude-only models drop this geometric phase information, whereas complex-valued representations [11, 12, 13] or real–imaginary decompositions [14, 15] retain full anatomical content. Although dual-domain fusion is popular [16, 17, 18], deterministic frequency features alone are insufficient and often redundant with spatial cues (Sec. 4). Learning coherent morphological prototypes requires a structured prior that explicitly couples spatial and spectral latents.

We present Dual-VaDE, an unsupervised prototypelearning framework that factorizes latents into coupled spatial and spectral subspaces. Our contributions are:

1. Dual-domain mixture prior: We propose a dual-domain Gaussian mixture prior, parameterized by a newly designed mean and covariance matrix whose off-diagonal cross-covariance terms explicitly capture within-prototype spatial–spectral co-variation.

2. End-to-End Variational EM Optimization: We derive a new closed-form variational lower bound, optimized endto-end via EM, avoiding multi-stage error accumulation and improving training stability and clustering robustness.

3. Empirical evaluation and open-source implementation: On structural brain MRI, Dual-VaDE yields sharper prototypes, better cluster separability, and more stable optimization than eight generative baselines with minimal overhead.

![](images/bd95293c371b81f6397e09d36e980c017925c3dcfddcf9229a29fc01fec31021.jpg)  
Fig. 1. Dual-VaDE overview. Left: a spatial encoder and a phase-preserving spectral encoder (real/imaginary Fourier channels) produce two latent codes. A dual-domain mixture prior couples them via the cross-domain block $\mathbf { C } _ { k }$ . The fused code feeds a shared decoder to generate reconstructions and prototypes. Right: prototype separation (mean Silhouette over K) vs. throughput, and robustness measured as seed-to-seed variability under the worst case over $K .$

## 2. BACKGROUND

Let x be a medical image, $\mathbf { z } \in \mathbb { R } ^ { d }$ a continuous latent code, and $c \in \{ 1 , \ldots , K \}$ a discrete prototype index. VaDE [7] assumes the generative process $c \sim \mathrm { C a t } ( \pi ) , \textbf { z } | \ c = k \sim$ $\mathcal { N } ( \mathbf { m } _ { k } , \pmb { \Sigma } _ { k } )$ , and $\textbf { x } | \textbf { z } \sim p _ { \theta } ( \textbf { x } | \textbf { z } )$ , with an amortized posterior $q _ { \phi } ( \mathbf { z } \mid \mathbf { x } ) = \mathcal { N } ( \pmb { \mu } , \mathbf { D } )$ and $\mathbf { D } = \mathrm { d i a g } ( \pmb { \sigma } _ { \phi } ^ { 2 } ( \mathbf { x } ) )$ . The evidence lower bound (ELBO) is

$$
\begin{array} { r l } & { \mathcal { L } = \mathbb { E } _ { q _ { \phi } } [ \log p _ { \theta } ( \mathbf { x } \mid \mathbf { z } ) ] - \mathrm { K L } ( { \boldsymbol \gamma } \parallel \pi ) } \\ & { \qquad - \sum _ { k = 1 } ^ { K } \gamma _ { k } \mathrm { K L } \bigl ( \mathcal { N } ( { \boldsymbol \mu } , \mathbf { D } ) \parallel \mathcal { N } ( \mathbf { m } _ { k } , \Sigma _ { k } ) \bigr ) , } \end{array}\tag{1}
$$

where $\gamma$ are the posterior prototype responsibilities [7, 8, 19], and related unsupervised clustering methods are explored in [19]. Because a single code z must capture both global anatomy and local texture, each $\Sigma _ { k }$ is an unstructured $d \times d$ matrix without factor blocks, and the model cannot represent cross-domain dependencies explicitly. Multi-view generative models address a related problem by assigning one latent code per view [20]; we follow this idea within the prior by giving it two codes whose covariance blocks follow this split.

## 3. METHODOLOGY: DUAL-VADE

## 3.1. Dual-domain mixture prior

We keep the generative process of Sec. 2 and change only the prior. Each image is embedded in a joint spatial and spectral latent space by concatenating a spatial code $\mathbf { z } _ { s }$ and a spectral code $\mathbf { z } _ { f } ,$ both in $\mathbb { R } ^ { d }$ , into $\pmb { \zeta } = [ \mathbf { z } _ { s } ; \mathbf { z } _ { f } ] \in \mathbb { R } ^ { 2 d }$ , on which we impose a Gaussian mixture with block-structured parameters:

$$
\begin{array} { r l r } {  { \zeta \mid c = k \sim \mathcal { N } ( \tilde { \mathbf { m } } _ { k } , \tilde { \Sigma } _ { k } ) , } } \\ & { } & { \tilde { \mathbf { m } } _ { k } = [ \mathbf { m } _ { k } ^ { s } ] , \quad \tilde { \Sigma } _ { k } = [ \mathbf { S } _ { k } ^ { s } \quad \mathbf { C } _ { k } ] . } \end{array}\tag{2}
$$

Here, $\mathbf { S } _ { k } ^ { s }$ and $\mathbf { S } _ { k } ^ { f }$ are the within-prototype spatial and spectral covariances, and the cross-domain block $\mathbf { C } _ { k } \in \mathbb { R } ^ { d \times d }$ captures how anatomy and spectral content co-vary within prototype k.

To obtain a single generative path, we define ${ { \bf z } } ^ { \star } = { \bf A } \boldsymbol { \zeta }$ with $\mathbf { A } = \left[ \beta \mathbf { W } _ { s } \alpha \mathbf { W } _ { f } \right]$ , learnable $\mathbf { W } _ { s } , \mathbf { W } _ { f } \in \mathbb { R } ^ { d \times d }$ , and scalars $\alpha , \beta , \mathrm { ~ A ~ }$ linear map of a Gaussian mixture is again a Gaussian mixture with unchanged weights [21], whose parameters are

$$
\begin{array} { r l } & { \mathbf { m } _ { k } ^ { \star } = \beta \mathbf { W } _ { s } \mathbf { m } _ { k } ^ { s } + \alpha \mathbf { W } _ { f } \mathbf { m } _ { k } ^ { f } , } \\ & { \pmb { \Sigma } _ { k } ^ { \star } = \beta ^ { 2 } \mathbf { W } _ { s } \pmb { \mathrm { S } } _ { k } ^ { s } \mathbf { W } _ { s } ^ { \top } + \alpha ^ { 2 } \mathbf { W } _ { f } \pmb { \mathrm { S } } _ { k } ^ { f } \mathbf { W } _ { f } ^ { \top } } \\ & { \quad \quad + \alpha \beta \big ( \mathbf { W } _ { s } \pmb { \mathrm { C } } _ { k } \mathbf { W } _ { f } ^ { \top } + \mathbf { W } _ { f } \pmb { \mathrm { C } } _ { k } ^ { \top } \mathbf { W } _ { s } ^ { \top } \big ) . } \end{array}\tag{3}
$$

The covariance thus decomposes exactly into a spatial, a spectral, and a cross-domain term, the last of which vanishes when $\mathbf { C } _ { k } = \mathbf { 0 }$ . The resulting dual-domain mixture prior on the decoder input is

$$
\begin{array} { r } { p ( \mathbf { z } ^ { \star } ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \mathcal { N } \big ( \mathbf { z } ^ { \star } \mid \mathbf { m } _ { k } ^ { \star } , \Sigma _ { k } ^ { \star } \big ) , } \end{array}\tag{4}
$$

which reduces to the VaDE prior for $\alpha = 0 , \beta = 1$ , and $\mathbf { W } _ { s } = \mathbf { I } ;$ we use this setting as initialization.

## 3.2. Phase-preserving spectral encoder

The spectral code $\mathbf { z } _ { f }$ is computed from the centered spectrum $\mathbf { X } = \mathbf { \bar { \mathrm { f f t s h i f t } } } ( \mathcal { F } ( \mathbf { x } ) \mathbf { \bar { ) } } \in \mathbb { C } ^ { H \times W }$ . Since magnitude-only inputs $g ( | \mathbf { X } | )$ are invariant to phase and thus discard spatial localization [10], we represent X by its real and imaginary parts as two channels, as in recent spectral networks [14, 15], on which the complex-valued layers [11] are implemented<sup>2</sup>. The final complex feature map $\mathbf { H } _ { L }$ is converted to real features by the real-part readout

$$
\begin{array} { r l } & { \mathcal { R } ( \mathbf { H } _ { L } ) = \Re \big ( \mathcal { F } ^ { - 1 } ( \mathbf { H } _ { L } ) \big ) = \mathcal { F } ^ { - 1 } ( \mathbf { H } _ { L } ^ { \mathrm { h } } ) , } \\ & { \mathbf { H } _ { L } ^ { \mathrm { h } } ( \omega ) = \frac { 1 } { 2 } \big ( \mathbf { H } _ { L } ( \omega ) + \overline { { \mathbf { H } _ { L } ( - \omega ) } } \big ) , } \end{array}\tag{5}
$$

which is linear and injective on Hermitian-symmetric spectra and therefore preserves both the magnitude and the relative phase of $\mathbf { H } _ { L } ^ { \mathrm { h } }$

## 3.3. Inference and objective

The two encoders share a single decoder and yield $q _ { \phi _ { s } } ( \mathbf { z } _ { s }$ $\mathbf { x } ) = \mathcal { N } ( \pmb { \mu _ { s } } , \mathrm { d i a g } \pmb { \sigma _ { s } ^ { 2 } } )$ and $q _ { \phi _ { f } } ( \mathbf { z } _ { f } \mid \mathbf { x } ) = \mathcal { N } ( \mu _ { f } , \operatorname { d i a g } \pmb { \sigma } _ { f } ^ { 2 } )$ The push-forward through A gives the fused posterior $q ( \mathbf { z } ^ { \star }$ $\mathbf { x } ) \ = \ \mathcal { N } ( \mu ^ { \star } , \mathbf { D } ^ { \star } )$ with $\mu ^ { \star } \ = \ \beta \mathbf { W } _ { s } \mu _ { s } + \alpha \mathbf { W } _ { f } \mu _ { f }$ and log $\pmb { \sigma } ^ { \star 2 } = \mathbf { G } _ { }$ <sub>s</sub> log σ<sup>2</sup><sub>s</sub> + G<sub>f</sub> log σ<sup>2</sup><sub>f</sub>, where the learnable weights ${ \bf G } _ { s } , { \bf G } _ { f }$ sum to one. Training minimizes the dualdomain negative ELBO

$$
\begin{array} { r l } & { \mathcal { L } ^ { \star } = \mathbb { E } _ { q ( \mathbf { z } ^ { \star } \mid \mathbf { x } ) } \big [ \| \mathbf { x } - p _ { \theta } ( \mathbf { z } ^ { \star } ) \| ^ { 2 } \big ] + \lambda _ { c } \mathrm { K L } ( \gamma \| \pi ) } \\ & { \qquad + \lambda _ { z } \sum _ { k = 1 } ^ { K } \gamma _ { k } \mathrm { K L } \big ( \mathcal { N } ( \pmb { \mu } ^ { \star } , \mathbf { D } ^ { \star } ) \| \mathcal { N } ( \mathbf { m } _ { k } ^ { \star } , \pmb { \Sigma } _ { k } ^ { \star } ) \big ) . } \end{array}\tag{6}
$$

In contrast to Eq. (1), the KL term is evaluated against the structured components of Eq. (3), so that the spatial and spectral codes are coupled through $\mathbf { C } _ { k }$ . The mixture parameters are updated by expectation-maximization [22] between gradient steps. The E-step computes responsibilities in the fused space,

$$
\begin{array} { r } { \gamma _ { i k } = \pi _ { k } \mathcal { N } ( { \pmb \mu } _ { i } ^ { \star } \mid \mathbf { m } _ { k } ^ { \star } , \pmb { \Sigma } _ { k } ^ { \star } ) \big / \sum _ { j } \pi _ { j } \mathcal { N } ( { \pmb \mu } _ { i } ^ { \star } \mid \mathbf { m } _ { j } ^ { \star } , \pmb { \Sigma } _ { j } ^ { \star } ) , } \end{array}\tag{7}
$$

whereas the M-step operates in the joint space, so that every block of $\tilde { \Sigma } _ { k }$ , including $\mathbf { C } _ { k }$ , is estimated. With $N _ { k } = \Sigma _ { i }$ γ<sub>ik</sub> and $\zeta _ { i } = [ \mu _ { s , i } ; \mu _ { f , i } ]$

$$
\begin{array} { r } { \pi _ { k } = N _ { k } / N , \qquad \tilde { \mathbf { m } } _ { k } = \frac { 1 } { N _ { k } } \sum _ { i } \gamma _ { i k } \zeta _ { i } , } \end{array}\tag{8}
$$

$$
\begin{array} { r } { \tilde { \Sigma } _ { k } = \frac { 1 } { N _ { k } } \sum _ { i } \gamma _ { i k } ( \zeta _ { i } - \tilde { \mathbf { m } } _ { k } ) ( \zeta _ { i } - \tilde { \mathbf { m } } _ { k } ) ^ { \top } , } \end{array}\tag{9}
$$

and Eq. (3) maps the updated parameters to the decoder input.

## 4. EXPERIMENTS

Data. We use 253 OASIS-3 brain MR images [23] from Alzheimer’s patients aged 63–75 to reduce age/diagnosis confounding and capture within-disease anatomical heterogeneity. Images are registered, resampled to 128 × 128, and split into 202 train and 51 test.

Implementation, Baselines & Metrics. For a controlled comparison, all models share the same encoder/decoder and training schedule; they differ only in which latent code uses a mixture prior. We train with seeds 42/92/123 and $K \in \{ 3 , 5 , 7 \}$ . After training, we fit a K-component GMM to posterior means on the training set and assign each test image to a prototype.

We compare VaDE [7] (mixture prior on the spatial code), Freq-VaDE (published spectral encoder), and Dual-VaDE (our spatial branch plus the dual-domain prior of Sec. 3 on the same spectral encoder). We test four Fourier inputs: full magnitude (FullFreq) [16], low-frequency magnitude (LowFreq) [24, 17], full complex spectrum with complex convolutions (PolarFreq) [11, 13], and real/imaginary channels with real convolutions (ReIm) [14, 12]. Unless noted, we use Dual-VaDE with ReIm. We report test Silhouette [25] and Davies-Bouldin (DB) [26] scores (mean ± SD over seeds) and compare to baselines with a one-sided Welch’s t-test [27]. Inference time is averaged over ten runs on an NVIDIA RTX PRO 6000 Blackwell GPU.

Prototype separation. In Tab. 1, Dual-VaDE achieves the best Silhouette for all K (0.648, 0.610, 0.447) and the lowest DB at K = 3 and 7 (mean rank 1.08 vs. 4.08 for the runnerup); at K = 5 its DB (0.449) ties Freq-VaDE+LowFreq. Versus VaDE, it raises Silhouette by 39.1%, 93.7%, and 13.7% and lowers DB by 46.4%, 61.9%, and 21.6% at $K = 3 , 5$ , and 7, with significant gains $( p < 0 . 0 5 )$ in 37/48 comparisons. Fusion effects vary by phase: moving from Freq-VaDE to Dual-VaDE, magnitude-only FullFreq/LowFreq reduce Silhouette at $K \ = \ 3$ and 5, while phase-preserving PolarFreq/ReIm improve for all K, suggesting magnitude overlaps with the spatial code but phase provides complementary information. LowFreq is strong for $K \leq 5$ but fails at $K = 7$ (Silhouette 0.058, DB 2.003).

Prototypes and latent structure. Fig. 2(a) compares learned prototypes and latent embeddings for our model, two singledomain variants, and the strongest baseline (Dual-VaDE with LowFreq), using one seed per K and Hungarian-matched labels [28]. The t-SNE [29] plots align with Tab. 1: VaDE prototypes overlap $( S = 0 . 1 5 6 )$ , while ours are more separated, especially at $K = 3 \left( S = 0 . 6 0 3 \right)$ and $K = 5 ( S = 0 . 5 4 5 )$ . At K = 5 (VaDE) and for both K in Dual-VaDE with LowFreq, some prototypes show large variance deviations near the lateral ventricles and periventricular white matter (red arrows), consistent with enlargement markers [30]. Our deviations are smaller and more uniform, and the prototypes remain sharp.

Efficiency, robustness, and complementarity. Fig. 1 (right) shows that our model is the most accurate and among the fastest spectral models: mean Silhouette 0.568 vs. 0.493 next best, at 0.277 ms/image (Freq-VaDE with ReIm: 0.276 ms), and 20%/22% faster than PolarFreq/LowFreq. It is also the most robust, with the lowest across-seed SD (0.015 averaged over K vs. 0.020–0.039) and the best worst-case over K (0.447).

Table 1. Prototype separation on OASIS-3 (mean±SD, 3 seeds). Silhouette (↑, [−1, 1]): closer to own than nearest other prototype. Davies-Bouldin (↓): within-prototype scatter vs between-prototype distance. Phase: Fourier phase preserved. Dual-VaDE augments the cited spectral encoder with a spatial branch and our dual-domain prior. Bold/underline: best/second best. $^ { \dagger } p < 0 . 0 5 , ^ { \ddagger } p < 0 . 0 1 $ : ours significantly better (one-sided Welch’s t-test).
<table><tr><td></td><td></td><td></td><td colspan="3">Silhouette ↑</td><td colspan="3">Davies-Bouldin ↓</td><td></td></tr><tr><td>Model</td><td>Spectral input Phase</td><td></td><td>K = 3</td><td>K = 5</td><td>K = 7</td><td>K = 3</td><td>K = 5</td><td>K = 7</td><td>Avg. rank</td></tr><tr><td>VaDE [7]</td><td>None</td><td></td><td> $0 . 4 6 6 \pm 0 . 0 2 2 ^ { \ddagger }$ </td><td> $0 . 3 1 5 \pm 0 . 0 0 9 ^ { \ddagger }$ </td><td> $0 . 3 9 3 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td> $0 . 8 4 1 \pm 0 . 0 8 3 ^ { \ddagger }$ </td><td> $1 . 1 7 9 \pm 0 . 2 5 2 ^ { \dag }$ </td><td> $0 . 7 1 7 \pm 0 . 1 9 6$ </td><td>6.67</td></tr><tr><td rowspan="4">Freq-VaDE</td><td>FullFreq [16]</td><td>x</td><td> $0 . 5 9 4 \pm 0 . 0 0 3 ^ { \dagger }$ </td><td> $0 . 4 7 0 \pm 0 . 0 2 3 ^ { \ddagger }$ </td><td> $0 . 2 9 3 \pm 0 . 0 3 9 ^ { \ddagger }$ </td><td> $0 . 4 9 3 \pm 0 . 0 0 6$ </td><td> $0 . 6 3 6 \pm 0 . 0 2 4 ^ { \ddagger }$ </td><td> $0 . 8 9 7 \pm 0 . 1 0 1 ^ { \dagger }$ </td><td>4.33</td></tr><tr><td>LowFreq [17]</td><td>x</td><td> $\underline { { 0 . 6 3 7 } } \pm 0 . 0 2 1$ </td><td> $0 . 6 0 0 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 0 5 8 \pm 0 . 0 4 6 ^ { \ddagger }$ </td><td> $0 . 4 8 0 \pm 0 . 0 2 2$ </td><td> $\mathbf { 0 . 4 4 9 \pm 0 . 0 1 1 }$ </td><td> $2 . 0 0 3 \pm 0 . 3 4 8 ^ { \ddagger }$ </td><td>4.25</td></tr><tr><td>PolarFreq [13]</td><td>√</td><td> $0 . 4 1 5 \pm 0 . 0 0 6 ^ { \ddagger }$ </td><td> $0 . 3 1 4 \pm 0 . 0 3 6 ^ { \ddagger }$ </td><td> $0 . 2 4 0 \pm 0 . 0 5 5 ^ { \ddagger }$ </td><td> $1 . 1 1 1 \pm 0 . 0 4 0 ^ { \ddagger }$ </td><td> $1 . 1 6 8 \pm 0 . 0 3 4 ^ { \ddagger }$ </td><td> $1 . 1 0 6 \pm 0 . 0 8 6 ^ { \ddagger }$ </td><td>8.33</td></tr><tr><td>ReIm [12]</td><td>√</td><td> $0 . 5 8 9 \pm 0 . 0 3 6 ^ { \dag }$ </td><td> $0 . 4 3 8 \pm 0 . 0 3 2 ^ { \ddagger }$ </td><td> $0 . 2 0 6 \pm 0 . 0 4 9 ^ { \ddagger }$ </td><td> $0 . 5 4 6 \pm 0 . 0 2 9 ^ { \ddagger }$ </td><td> $0 . 6 8 5 \pm 0 . 0 2 9 ^ { \ddagger }$ </td><td> $1 . 0 8 2 \pm 0 . 3 1 0 ^ { \dagger }$ </td><td>6.17</td></tr><tr><td rowspan="3">Dual-VaDE</td><td>FullFreq [16]</td><td>x</td><td> $0 . 5 5 0 \pm 0 . 0 0 9 ^ { \ddagger }$ </td><td> $0 . 4 5 3 \pm 0 . 0 4 4 ^ { \ddagger }$ </td><td> $\underline { { 0 . 4 2 7 } } \pm 0 . 0 0 6 ^ { \dagger }$ </td><td> $0 . 5 9 6 \pm 0 . 0 1 3 ^ { \ddagger }$ </td><td> $0 . 6 9 1 \pm 0 . 0 8 5 ^ { \dagger }$ </td><td> $\underline { { 0 . 7 0 3 } } \pm 0 . 0 7 0 ^ { \dag }$ </td><td>5.25</td></tr><tr><td>LowFreq [17]</td><td>x</td><td> $0 . 5 8 3 \pm 0 . 0 1 3 ^ { \ddagger }$ </td><td> $0 . 5 2 3 \pm 0 . 0 1 1 ^ { \ddagger }$ </td><td> $0 . 3 7 3 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td> $0 . 5 7 4 \pm 0 . 0 8 8$ </td><td> $0 . 5 9 9 \pm 0 . 0 3 2 ^ { \ddagger }$ </td><td> $0 . 7 0 3 \pm 0 . 0 9 8$ </td><td>4.08</td></tr><tr><td>PolarFreq [13]</td><td>√</td><td> $0 . 5 7 6 \pm 0 . 0 1 0 ^ { \ddagger }$ </td><td> $0 . 5 1 1 \pm 0 . 0 5 1 ^ { \dagger }$ </td><td> $0 . 2 8 5 \pm 0 . 0 3 5 ^ { \ddagger }$ </td><td> $0 . 5 3 8 \pm 0 . 0 1 2 ^ { \dag }$ </td><td> $0 . 5 6 3 \pm 0 . 0 8 0$ </td><td> $0 . 9 5 9 \pm 0 . 0 9 1 ^ { \ddagger }$ </td><td>4.83</td></tr><tr><td>Dual-VaDE (ours) ReIm</td><td></td><td>√</td><td> $\mathbf { 0 . 6 4 8 \pm 0 . 0 1 6 }$ </td><td> ${ \bf 0 . 6 1 0 \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 4 4 7 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 4 5 1 \pm 0 . 0 2 8 }$ </td><td> $0 . 4 4 9 \pm 0 . 0 2 6$ </td><td> $\mathbf { 0 . 5 6 2 \pm 0 . 0 2 4 }$ </td><td>1.08</td></tr><tr><td colspan="2">Relative change of ours vs. VaDE</td><td></td><td>+39.1%</td><td>+93.7%</td><td>+13.7%</td><td>-46.4%</td><td>-61.9%</td><td>-21.6%</td><td></td></tr></table>

![](images/d5f67d68c7d84e38b821f6d519e54a67c1313188d93e0cde45183ccb9429a4ff.jpg)  
Fig. 2. (a) Prototypes with relative-variance overlays (red: above mean, blue: below; red arrows: large ventricular deviations) and t-SNE for $K = 3$ and $K = 5$ . Labels match ours, so colors indicate the same prototype across rows. n: images per prototype; S: latent separation. (b) Ours vs. single-branch variants. (c) Principal angles between our spatial and spectral latent subspaces (dots) and their mean (bars).

Fig. 2(b) shows that the full model beats the better single branch by 0.038, 0.114, and 0.054 Silhouette at $K = 3 ,$ 5, and 7 (8/9 seeds). Fig. 2(c) links this to complementarity: principal angles [31] between the top-10-PC subspaces of spatial and spectral latent means average 46.5<sup>◦</sup>, 48.6<sup>◦</sup>, and $5 5 . 3 ^ { \circ }$ for $K = 3 , \ 5$ , and 7, with some near-orthogonal directions. Thus, the branches capture distinct factors, making the cross-domain block $\mathbf { C } _ { k }$ informative.

## 5. CONCLUSION & DISCUSSION

This work introduces Dual-VaDE, an unsupervised deep generative framework that jointly models latent mixture distributions across coupled spatial and Fourier domains. By explicitly encoding cross-domain covariance and phase-preserving spectral components, the formulation regularizes the latent manifold against high-frequency blur and prevents mode collapse. Model parameters are inferred through an analytically grounded expectation-maximization scheme tailored for the coupled dual-domain objective. Evaluated on real brain MRI cohorts, Dual-VaDE uncovers distinct neuroanatomical subtypes with sharp cortical and ventricular boundaries while substantially mitigating the random-seed sensitivity common to deep mixture clustering. These geometric and statistical gains are achieved without prohibitive parameter scaling or inference bottlenecks. Future work will extend this dual-domain mixture formulation to continuous spatiotemporal manifolds, establishing a unified generative prior for mapping longitudinal structural atrophy trajectories.

## 6. ACKNOWLEDGMENTS & ETHICS

This research was conducted using open-access data resources and received no external funding.

## 7. REFERENCES

[1] Mohamad Habes, Michel J. Grothe, Birkan Tunc, et al., “Disentangling heterogeneity in Alzheimer’s disease and related dementias using data-driven methods,” Biol. Psychiatry, vol. 88, no. 1, pp. 70–82, 2020.

[2] Diederik P. Kingma and Max Welling, “Auto-encoding variational Bayes,” in Proc. ICLR, 2014.

[3] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion probabilistic models,” in Proc. NeurIPS, 2020, vol. 33, pp. 6840–6851.

[4] Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Moein Heidari, et al., “Diffusion models in medical imaging: A comprehensive survey,” Med. Image Anal., vol. 88, pp. 102846, 2023.

[5] Evan M Yu and Mert R Sabuncu, “A convolutional autoencoder approach to learn volumetric shape representations for brain structures,” arXiv preprint arXiv:1810.07746, 2018.

[6] Zhijian Yang, Ilya M. Nasrallah, Haochang Shou, et al., “A deep learning framework identifies dimensional representations of Alzheimer’s disease from brain structure,” Nat. Commun., vol. 12, pp. 7065, 2021.

[7] Zhuxi Jiang, Yin Zheng, Huachun Tan, et al., “Variational deep embedding: An unsupervised and generative approach to clustering,” in Proc. IJCAI, 2017, pp. 1965–1972.

[8] Nat Dilokthanakul, Pedro A. M. Mediano, Marta Garnelo, et al., “Deep unsupervised clustering with Gaussian mixture variational autoencoders,” arXiv preprint arXiv:1611.02648, 2016.

[9] Fabian Falck, Haoting Zhang, Matthew Willetts, George Nicholson, Christopher Yau, and Chris C Holmes, “Multi-facet clustering variational autoencoders,” Advances in Neural Information Processing Systems, vol. 34, pp. 8676–8690, 2021.

[10] Alan V. Oppenheim and Jae S. Lim, “The importance of phase in signals,” Proc. IEEE, vol. 69, no. 5, pp. 529–541, 1981.

[11] Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, et al., “Deep complex networks,” in Proc. ICLR, 2018.

[12] Aleksandar Stanic, Anand Gopalakrishnan, Kazuki Irie, et al.,´ “Contrastive training of complex-valued autoencoders for object discovery,” in Proc. NeurIPS, 2023, vol. 36, pp. 11075– 11101.

[13] Anand Gopalakrishnan, Aleksandar Stanic, J´ urgen Schmidhu-¨ ber, et al., “Recurrent complex-weighted autoencoders for unsupervised object discovery,” in Proc. NeurIPS, 2024, vol. 37, pp. 140787–140811.

[14] Lu Chi, Borui Jiang, and Yadong Mu, “Fast Fourier convolution,” in Proc. NeurIPS, 2020, vol. 33.

[15] Yongming Rao, Wenliang Zhao, Zheng Zhu, et al., “Global filter networks for image classification,” in Proc. NeurIPS, 2021.

[16] Joel Frank, Thorsten Eisenhofer, Lea Schonherr, Asja Fischer,¨ Dorothea Kolossa, and Thorsten Holz, “Leveraging frequency analysis for deep fake image recognition,” in Proc. ICML, 2020, vol. 119 of PMLR, pp. 3247–3258.

[17] Oren Rippel, Jasper Snoek, and Ryan P. Adams, “Spectral representations for convolutional neural networks,” in Proc. NeurIPS, 2015, vol. 28.

[18] Kai Xu, Minghai Qin, Fei Sun, Yuhao Wang, Yen-Kuang Chen, and Fengbo Ren, “Learning in the frequency domain,” in Proc. CVPR, 2020, pp. 1740–1749.

[19] Junyuan Xie, Ross Girshick, and Ali Farhadi, “Unsupervised deep embedding for clustering analysis,” in Proc. ICML, 2016, pp. 478–487.

[20] Yuge Shi, N. Siddharth, Brooks Paige, et al., “Variational mixture-of-experts autoencoders for multi-modal deep generative models,” in Proc. NeurIPS, 2019, vol. 32.

[21] Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, New York, 2006.

[22] Arthur P. Dempster, Nan M. Laird, and Donald B. Rubin, “Maximum likelihood from incomplete data via the EM algorithm,” J. R. Stat. Soc. B, vol. 39, no. 1, pp. 1–38, 1977.

[23] Pamela J. LaMontagne, Tammie L. S. Benzinger, John C. Morris, et al., “OASIS-3: Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer disease,” medRxiv, 2019.

[24] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, et al., “Fourier neural operator for parametric partial differential equations,” in Proc. ICLR, 2021.

[25] Peter J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” J. Comput. Appl. Math., vol. 20, pp. 53–65, 1987.

[26] David L. Davies and Donald W. Bouldin, “A cluster separation measure,” IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-1, no. 2, pp. 224–227, 1979.

[27] Bernard L. Welch, “The generalization of “Student’s” problem when several different population variances are involved,” Biometrika, vol. 34, no. 1–2, pp. 28–35, 1947.

[28] Harold W. Kuhn, “The Hungarian method for the assignment problem,” Nav. Res. Logist. Q., vol. 2, no. 1–2, pp. 83–97, 1955.

[29] Laurens van der Maaten and Geoffrey Hinton, “Visualizing data using t-SNE,” J. Mach. Learn. Res., vol. 9, pp. 2579– 2605, 2008.

[30] Sean M. Nestor et al., “Ventricular enlargement as a possible measure of Alzheimer’s disease progression validated using the Alzheimer’s disease neuroimaging initiative database,” Brain, vol. 131, no. 9, pp. 2443–2454, 2008.

[31] Ake Bj<sup>˚</sup> orck and Gene H. Golub, “Numerical methods for com-¨ puting angles between linear subspaces,” Math. Comput., vol. 27, pp. 579–594, 1973.