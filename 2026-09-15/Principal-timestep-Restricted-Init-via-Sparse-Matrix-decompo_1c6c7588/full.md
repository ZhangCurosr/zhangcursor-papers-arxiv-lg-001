# Principal-timestep Restricted Init via Sparse Matrix-decomposition in Flow-matching

Jiayang Gu<sup>1∗</sup>, Zheng Fang<sup>1∗</sup>, Lichuan Xiang<sup>3</sup>, Xu Cai<sup>1</sup>, Fanghui Liu<sup>1</sup>, Hongkai Wen<sup>1</sup> <sup>†</sup>

<sup>1</sup>University of Warwick, <sup>2</sup>Apple Inc., <sup>3</sup>Bytedance {Jiayang-Gu.Gu, Zhengfang6, }@warwick.ac.uk.com

## Abstract

Flow-matching diffusion models have recently emerged as a strong paradigm for high-fidelity visual generation. However, their prohibitively high fine-tuning cost limits scalability to downstream tasks. While Low-Rank Adaptation (LoRA) combined with spectral initialization has demonstrated accelerated convergence and improved performance in autoregressive language models by better aligning gradient directions, we find that it fails to deliver similar gains in diffusion fine-tuning, often yielding marginal or even negative improvements over vanilla LoRA. We attribute this discrepancy to a fundamental mismatch between LoRA’s low-rank parameterization and the intrinsically high-rank gradients induced by the flow-matching objective. In particular, stochastic timestep sampling introduces directionally heterogeneous gradient signals across training steps, leading to misaligned updates under low-rank constraints. To address this issue, we propose PRISM-LoRA, a Principal-timestep Restricted Init via Sparse Matrix-decomposition framework that improves gradient alignment during fine-tuning. Our method consists of two key components: (i) principal timestep selection, which restricts initialization gradients to a subset of dominant timesteps to suppress effective gradient rank, and (ii) principal channel filtering, which removes task-irrelevant channels, enabling the one-step spectral initialization gradient to better align with the long-horizon optimization trajectory. Extensive experiments demonstrate that PRISM-LoRA consistently improves both convergence speed and final performance across multiple diffusion finetuning benchmarks, including subject-driven generation, controllable generation, and deblurring, achieving not only performance improvement but also earlier stages of convergence over baseline LoRA and other spectral-init methods. Our code is available at https://anonymous.4open.science/r/Prism-LoRA-28ED.

## 1 Introduction

Recently, the spectral initialization on LoRA has been receiving increasing research attention. Other LoRA variants [30, 23, 33], either require a dedicated design of hyperparameters or extra training parameters to improve the performance of fine-tuning. In contrast, spectral initialization [39, 47, 28] requires few hyperparameters and comes with tight theoretical guarantees on fine-tuning scaling. By initializing the LoRA adapters with the decomposition of either the pre-trained weights or a gradient obtained from a single-batch forward pass, it achieves both stronger performance and faster convergence.

The achievement of spectral initialization can be attributed to the successful transfer of the knowledge from the fine-tuning pattern of the full-parameter model to the low-rank parameters of LoRA.

![](images/f0bca504321978ceca0fd9a030d2a4774d49df3b214e60a519cbbc009e240813.jpg)  
(a) Gradient cosine similarity matrix.

![](images/dc2c331d9545e8930099df19f02af9029aef0777c967e235ac0e2d8fdc29f484.jpg)  
(b) Mean rank comparison.  
Figure 1: Left figure shows the cosine similarity matrix on different diffusion timesteps, right figures show when sampled from a wider range of timesteps and sampling numbers, the difference in gradient rank and compared with LLM gradient.

Existing one-step gradient methods typically justify their effectiveness along two axes: (i) they show that the LoRA parameterization obtained by decomposing the first-step gradient attains the minimal approximation error relative to full fine-tuning at initialization; and (ii) they argue that this initialization remains well-aligned with the dominant gradient directions throughout subsequent training. Both arguments implicitly rely on a critical premise—that the effective rank of the full-model gradient is comparable to the LoRA rank. As effective rank is related to the energy distribution towards singular value, it indicates that using how many of the singular vectors on the front would have a reconstructive effect towards the original matrix. If effective rank matches LoRA’s rank, a good LoRA initialization weight derived from the decomposition result of the matrix can set an optimization start point similar as a full fine-tuning model does.

This premise, however, breaks down in the diffusion setting. Figure 1 illustrates the issue from two angles. Figure 1a shows that gradients from different timesteps share little similarity, indicating that they span largely distinct subspaces. Consequently, the aggregated gradient $G ^ { \natural } = \mathbb { E } _ { t } [ G _ { t } ]$ exhibits a substantially higher effective rank than any single $G _ { t }$ . Figure 1b quantifies this effect. We define the empirical effective rank as the number of top singular vectors needed to reconstruct 99% of the gradient’s Frobenius norm, and progressively enlarge the aggregated timestep range $( ^ { 6 6 } 1 0 0 0 ^ { 5 }$ covers $\mathbf { \check { T } } _ { 1 } { - } T _ { 2 } , \mathbf { \cdots } 2 0 0 0 ^ { \cdots }$ covers $T _ { 1 } { - } T _ { 4 } ,$ , and so on). Compared to an LLM baseline, the diffusion gradient’s effective rank grows markedly faster as more timesteps are aggregated; the overlaid line further shows that this rank itself rises as timesteps move from noise to data. As a result, $G ^ { \natural }$ inherits an inflated rank structure, and the low-rank SVD on which one-step methods like LoRA-One rely becomes a lossy operation in the diffusion setting.

We attribute this failure to a fundamental mismatch between spectral-init’s low-rank assumption and the gradient geometry of diffusion training. In LLMs, the one-step gradient is wellapproximated as low-rank, justifying spectral initialization. In diffusion, the one-step gradient aggregates over a sequence of timesteps $\mathbb { E } _ { t } [ G _ { t } ] ;$ since different t correspond to denoising tasks at different signal-to-noise ratios, the $G _ { t }$ point in widely divergent directions. The aggregated gradient, therefore, has substantially higher effective rank, violating the core premise of spectral initialization.

On this basis, we raise 2 hypotheses corresponding to 2 aspects, shown as follows:

Hypothesis 1 (Principal Timesteps reduce effective rank) Restricting gradient estimation to small timestep t(closer to noise) t → 0 yields a gradient matrix $\dot { G } ^ { \natural }$ with substantially lower effective rank than uniform timestep sampling, improving the fidelity of ranking-r spectral initialization.

Hypothesis 2 (Sparse decomposition improves finite-sample subspace recovery) The irrelevant channel signal would hinder the reconstructed weight from being aligned with the fine-tuning orientation.

Corresponding to Hypothesis 1, we introduce a principal timestep selection strategy that confines gradient estimation to a small set of timesteps, suppressing the effective rank of the guidance gradient before decomposition. Corresponding to Hypothesis 2, we apply a sparse channel decomposition that isolates task-relevant channels from the gradient signal, so that the reconstructed low-rank update aligns with the long-horizon fine-tuning trajectory rather than being polluted by irrelevant directions. Our contributions are summarized as follows:

1. We identify and analyze the failure mode of spectral initialization in diffusion fine-tuning, attributing it to a mismatch between the low-rank assumption of spectral initialization and the high-rank gradient geometry induced by stochastic timestep sampling.

2. Motivated by two hypotheses on gradient rank and channel relevance, we propose PRISM-LoRA, which combines principal timestep selection and sparse channel decomposition to produce a low-rank, well-aligned initialization for diffusion LoRA.

3. We validate our method on four downstream tasks (Dreambooth, Canny, Depth, Deblur) across Stable Diffusion-3 and Flux-1, achieving not only performance improvement but also training acceleration at the initial training stage over strong spectral-init baselines.

## 2 Related Work

Diffusion-based Generative Models. The diffusion model was originally introduced by [34], which has replaced Generative Adversarial Network (GAN)-based methods [38, 43, 24], and has been widely applied in the field of image synthesis [14, 8, 46]. Compared to adversarial optimization approaches, which easily lead to training instability and even collapse, diffusion models utilize the probabilistic optimization scheme to ensure stable convergence and generation diversity. The Latent Diffusion Model (LDM) [31] reduces computational demands by transferring the diffusion process from pixel space to latent space. Based on it, Transformer-based architectures such as Diffusion Transformer (DiT) [29, 51, 4, 17, 1], Stable Diffusion 3.0/3.5 [7], FLUX [18] further extend diffusion’s capacity to model long-range dependencies and scale to massive multi-modal datasets. Meanwhile, Rectified Flow optimization [19, 21, 22] enables faster sampling with fewer denoising steps without degrading quality. Although the merits are obvious, pre-training a diffusion model is undoubtedly expensive.

Post-training of Diffusion Models. Compared to pre-training, fine-tuning the fully-trained diffusion model on downstream task with a small-scale dataset [42, 16, 5] can be more feasible and cost-friendly. DreamBooth [32] finetunes text-to-image diffusion models using a few subject-specific images provided by practitioners, binding the subject to a unique identifier token to enable highfidelity and controllable personalized generation. [44, 26, 27] freeze the denoising backbone and finetune an additional module to transfer conditional controls into the noise latent space, thereby achieving spatially-aligned generation. Meanwhile, due to the decoupled design with the denoising network—parameter updation occurring only on the trainable adapter, the training stability and convergence speed are both superior to the full-parameter fine-tuning. To further compress the train able parameters, [35, 36, 48, 37, 9] introduce efficient LoRA module [13] for diffusion fine-tuning, which not only reduces the demand for gpu memory, but also makes it convenient for the diffusion model to switch freely between different design tasks. However, the number of update steps required for fine-tuning is still quite large, and the performance is difficult to match that of full-parameter fine-tuning.

Improving Fine-tuning Efficiency. Low-Rank Adaptation (LoRA) [13, 6, 23, 33] has dominated efficient large vision/language models’ fine-tuning. Apart from the parameter efficiency, researchers have also extended to explore time efficiency. On the one hand, some research works focus on improving initialization strategies to enhance the effectiveness of low-rank updates. [28, 39, 47] propose an initialization method based on gradient update approximation, which enables the low-rank parameters to approach the optimal update direction at the early stage of training. [52, 25, 20] leverage singular value decomposition (SVD) [40] to confine the adaptation within the principal singular subspace, thereby enabling more stable and robust initialization and updates of model weights by better aligning with the intrinsic structure of the pretrained model. On the other hand, there are also efforts to improve training efficiency by focusing on the optimization process. [10, 15, 45, 49] allocate different learning rates to different matrices in low-rank decomposition, significantly accelerating convergence and improving performance without changing the initialization method. However, due to the multi-optimization training objectives in diffusion fine-tuning, these methods fail to capture a robust enough gradient by only relying on one-step gradient to guide the overall training process.

## 3 Method

In this section, we will introduce how we solve the new problem of the gradient-based method applied in the diffusion domain. We understand that fluctuating training loss is the inevitable nature of fine-tuning the diffusion model, and it can hardly ensure that the 1-step optimization direction points at the average optimization direction of the overall fine-tuning process. We notice this problem and simplify it in Sec. 3.1, defined as High-ranking $G ^ { \natural }$ problem. It can be solved in 2 aspects, the timestep and channel, shown in Sec. 3.2 and Sec. 3.3, respectively. Finally, we formulate the overall training process in Sec. 3.4, to have an overall look at our method. For preliminaries, see Appendix. A.

![](images/812563ef64ab9ade58ebf51baf6307b9107fd6935453910440bc73031a9d3dc0.jpg)  
(a)

![](images/34281eb4e25a5c517114bd6a0fcb7f78f9164ff8fa5e7021935b6001e9628070.jpg)  
(b)

![](images/84e65f774a652cae18e5919657467efd101a6361c5535b3babdfc7da45e39dfb.jpg)  
(c)

![](images/be388b157d984036db958736c6276417f6d9eae7eb16c6642a088d3136b0b461.jpg)  
(d)  
Figure 2: Empirical validation of our proposed hypotheses. Fig. 2a indicates that the principal timestep(PT) achieves a lower rank than stochastic timesteps sampling. Fig. 2b under principal timestep sampling, calculates the principal angle with G with 50K sample size, achieving a closer direction. Fig. 2c and 2d shows golden channel module has a better ability in filtering out irrelevant channel information compared with SVD. For further analysis, see Sec. 3.3.

## 3.1 Barriers in Flow-matching Diffusion Model

High-ranking $G ^ { \natural } .$ . Near-orthogonal per-timestep gradients do not lead to a low-rank average, as each $G _ { t }$ contributes an independent direction to the column or row space of $G ^ { \natural }$ . Formally, if $G _ { t _ { 1 } }$ and $G _ { t _ { 2 } }$ have near-orthogonal top-r subspaces, the rank of their sum approaches ran $k ( G _ { t _ { 1 } } ) + r a n k ( G _ { t _ { 2 } } )$ Averaging over a continuous number of timesteps therefore inflates the effective rank of $G ^ { \natural }$ relative to any single $G _ { t } .$ . From a mathematical perspective, average gradient from different directions would lead to a high-ranking $G ^ { \natural }$ . It has 2 drawbacks: The decomposed LoRA parameter from high-ranking $G ^ { \natural }$ has a conflict with the low-ranking training objective with LoRA; High-ranking $G ^ { \sharp }$ may have a deviated guidance on multiple training objects. However, directly measuring the true rank of the gradient matrix is challenging. To address this, we convert this problem into measuring the spectral of the gradient matrix, which would have a positive correlation with the actual rank $r ^ { * }$ of $G ^ { \natural }$

From gradient rank to posterior covariance spectrum. Directly analyzing the spectrum of $G ^ { \natural }$ is intractable, as it entangles the data-driven residual with the architecture-dependent Jacobian $J _ { \theta }$ Under a locally linearized view, the per-timestep gradient covariance decomposes as

$$
G _ { t } \approx \mathbb { E } _ { x _ { t } } \big [ J _ { \theta } ^ { \top } \Sigma _ { v } ( t ) J _ { \theta } \big ] ,\tag{1}
$$

where $\begin{array} { r } { \Sigma _ { v } ( t ) = \frac { 1 } { t ^ { 2 } } \mathrm { V a r } [ x _ { 0 } \mid x _ { t } ] } \end{array}$ is the velocity-regression residual covariance. Since the architecturedependent factor $J _ { \theta }$ admits no general treatment, and since an ill-conditioned output target necessarily induces an ill-conditioned $G _ { t }$ , we adopt $\Sigma _ { v } ( t )$ as a principled proxy for the spectral behavior of $G ^ { \natural }$ . Furthermore, as absolute rank is numerically unstable, we replace it with a scale-invariant surrogate—the concentration of the spectrum—which is positively correlated with the effective rank and amenable to rigorous comparison via majorization.

We measure spectral concentration via the $r _ { N }$ metric: the minimum number of top eigenvalues needed to capture N% of the total variance, $r _ { N } ( \Sigma ) = \operatorname* { m i n } \{ r : \textstyle \sum _ { i = 1 } ^ { r } \sigma _ { i } \ge N ^ { \circleddash } \}$ where $\sigma _ { i }$ are singular values. Small $r _ { N }$ indicates a sharp, concentrated spectrum (effectively low-rank); $r _ { N } $ d indicates a flat spectrum (high-rank).

## 3.2 Principal Timestep

The effectiveness of the principal timestep is based on the following justification: the spectrum of the one-step gradient covariance $\Sigma _ { v } ( t )$ becomes flatter when mixing with more close-to-target timestep. As a result, restricting spectral initialization to smaller t leads to top-r subspace capturing a substantially larger fraction of the total signal. We formulate this theorem as follows:

Theorem 1 (Monotone spectral flattening). Under the assumptions ofSec. 3.1, the rank-N index $r _ { N }$ ofthe one-step gradient covariance $G ^ { \natural }$ is monotonically non-decreasing in t on $[ 0 , 1 ] .$

$$
r _ { N } ( \Sigma _ { v } ( 0 ) ) \ \leq \ r _ { N } ( \Sigma _ { v } ( 0 . 5 ) ) \ \leq \ r _ { N } ( \Sigma _ { v } ( 1 ) ) \ = \ \lceil N / 1 0 0 , d \rceil .\tag{2}
$$

By the flow-matching identity $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ , we eliminate $x _ { 1 }$ via $x _ { 1 } = \left( x _ { t } - ( 1 - t ) x _ { 0 } \right) / t$ so that the posterior covariance Var[x | x ] depends on $x _ { 0 }$ alone. Diagonalizing in the eigenbasis of $\Sigma _ { 1 }$ , the i-th singular value of $\Sigma _ { v } ( t )$ admits the closed form

$$
\sigma _ { i } ( t ) = \frac { \lambda _ { i } } { ( 1 - t ) ^ { 2 } + t ^ { 2 } \lambda _ { i } } ,\tag{3}
$$

where $\{ \lambda _ { i } \} _ { i = 1 } ^ { d }$ are the singular values of the data covariance $\Sigma _ { 1 } = \mathrm { C o v } ( x _ { 1 } )$ , and by symmetric positive semi-definiteness coincide with its eigenvalues. A full derivation is deferred to Appendix B.1.

Two-Ends Comparison. We first examine the two endpoints $t  0$ and $t  1$ $\mathrm { ~ \bf ~ A t ~ } t \to 0 .$ $\sigma _ { i } ( 0 ) = \lambda _ { i } ,$ , recovering the spectrum of $\Sigma _ { 1 } ;$ at $t = 1 , \sigma _ { i } ( 1 ) = 1$ for all $i ,$ yielding an isotropic covariance $\Sigma _ { v } ( 1 ) = \bar { I _ { d } }$ . To compare these two spectra, we treat $\sigma _ { i } ( t ) = f _ { t } ( \lambda _ { i } )$ as a univariate map with t as a fixed parameter and $\lambda _ { i }$ as the input variable. A direct computation shows that $f _ { t }$ is strictly increasing, strictly concave, and satisfies $f _ { t } ( 0 ^ { + } ) = 0$ . By Lemma 1 (normalized-spectrum majorization), these three properties imply that the normalized spectrum of $\Sigma _ { v } ( t )$ is majorized by that of $\Sigma _ { 1 }$ —that is, the concave map $f _ { t }$ compresses large singular values and lifts small ones, producing a strictly flatter output sequence. Consequently, every Schur-concave flatness functional (including $r _ { N } )$ satisfies $r _ { N } ( \bar { \Sigma } _ { v } ( t ) ) \bar { \ } \geq r _ { N } ( \Sigma _ { 1 } )$ for any $t \in ( 0 , 1 )$ , with strict inequality whenever $\Sigma _ { 1 }$ has a non-degenerate spectrum. Detailed arguments for these two endpoints are given in Appendices B.2 and B.3.

Having established flattening relative to the endpoint $t = 0$ , we next strengthen the result tofull-range monotonicity: the normalized spectrum $\{ \sigma _ { i } ( t ) \} _ { i = 1 } ^ { d }$ becomes strictly flatter as t increases on the entire interval [0, 1], not merely at the endpoints. This is established in the following subsection by a second application of Lemma 1 to the transition map between arbitrary time pairs $t _ { 1 } < t _ { 2 }$

Full-Range Monotonicity. We now strengthen Theorem 1 from endpoint comparison to full-range monotonicity. Fix $0 \leq t _ { 1 } < t _ { 2 } \leq 1$ and define the transition map $h _ { t _ { 1 }  t _ { 2 } } : ( 0 , \infty )  ( 0 , \infty )$ that sends each singular value at time $t _ { 1 }$ to its counterpart at time $t _ { 2 } { \mathrm { : } }$

$$
h _ { t _ { 1 } \to t _ { 2 } } ( s ) \ \triangleq \ f _ { t _ { 2 } } \big ( f _ { t _ { 1 } } ^ { - 1 } ( s ) \big ) , \qquad s \in \big \{ \sigma _ { i } ( t _ { 1 } ) \big \} _ { i = 1 } ^ { d } ,\tag{4}
$$

so that $\sigma _ { i } ( t _ { 2 } ) = h _ { t _ { 1 }  t _ { 2 } } ( \sigma _ { i } ( t _ { 1 } ) )$ for every $i .$

A direct calculation (Appendix B.4) shows that $h _ { t _ { 1 }  t _ { 2 } }$ is strictly increasing, strictly concave, and satisfies $h _ { t _ { 1 } \to t _ { 2 } } ( 0 ^ { + } ) = 0$ . Applying Lemma 1 to $h _ { t _ { 1 }  t _ { 2 } }$ with input sequence $\{ \sigma _ { i } ( t _ { 1 } ) \} _ { i = 1 } ^ { d }$ yields

$$
{ \widehat { \sigma } } ( t _ { 1 } ) \ \succ \ { \widehat { \sigma } } ( t _ { 2 } ) ,\tag{5}
$$

where $\textstyle { \widehat { \sigma } } ( t ) = \sigma ( t ) / \sum _ { i } \sigma _ { j } ( t )$ denotes the normalized spectrum. Since every Schur-concave flatness functional respects the majorization order, and $r _ { N }$ is Schur-concave on the simplex of normalized spectra, we conclude

$$
r _ { N } \bigl ( \Sigma _ { v } ( t _ { 1 } ) \bigr ) \ \leq \ r _ { N } \bigl ( \Sigma _ { v } ( t _ { 2 } ) \bigr ) , \qquad \forall 0 \leq t _ { 1 } < t _ { 2 } \leq 1 ,\tag{6}
$$

with strict inequality whenever $\Sigma _ { 1 }$ has a non-degenerate spectrum. This completes the proof of Theorem 1.

In the Fig. 2, the $r _ { N }$ line using SVD method to decompose $G ^ { \natural }$ , and the sim score is calculated by similarity with the actual fine-tuning delta weights subtracted by the final fine-tuning weights and pre-training weights.

Sec. 3.2 characterizes the spectrum of the input-space conditional covariance $\Sigma _ { v } ( t )$ , which encodes the posterior uncertainty over $x _ { 0 }$ given $x _ { t }$ . The empirical gradient that drives spectral initialization, however, is the weight gradient $\begin{array} { r } { \hat { G } _ { t } = \frac { 1 } { N } \sum _ { i } e _ { t } ^ { ( i ) } ( h _ { t } ^ { ( i ) } ) ^ { \top } } \end{array}$ , where $e _ { t } ^ { ( i ) }$ is the per-sample prediction error and $h _ { t } ^ { ( i ) }$ is the backbone feature. The effective rank of $\hat { G } _ { t }$ is therefore governed by the diversity of $\{ h _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ across samples, rather than by $\Sigma _ { v } ( t )$ alone. The two quantities nonetheless align at the data endpoint: as $t  0$ , inputs $x _ { t } \ \to \ x _ { 0 }$ concentrate near the data manifold, so backbone features collapse to a low-dimensional response region and $\mathrm { r a n k } _ { \mathrm { e f f } } ( \hat { G } _ { t } )$ drops in tandem with the sharpening of $\textstyle { \dot { \sum _ { v } ( t ) } }$ . Sec. 3.2 thus provides the input-level justification for the data-side endpoint of our Principal Timesteps; the symmetric low-rank behavior at the noise-side endpoint $t $ 1 arises from a complementary mechanism (feature collapse under near-isotropic inputs) that we examine empirically in Section 1.

## 3.3 Golden Channel

Though $r _ { N } ^ { G ^ { \natural } }$ has decreased to a level that can be reconstructed well by SVD under the LoRA rank budget, two concerns remain. First, the irrelevant information carried by $x _ { t }$ may still contaminate $G ^ { \natural }$ so the leading singular directions of $G ^ { \natural }$ are not guaranteed to align with the true optimization direction of the fine-tuning trajectory. Second, since the gradients we use for reinitialization are estimated from a finite sample of $x _ { t }$ , SVD—which fits the dominant energy of this particular sample—is sensitive to sample-specific noise rather than to the underlying shared signal across samples.

From SVD to PMD. To address both issues, we replace SVD with Penalized Matrix Decomposition (PMD) [41] when extracting the low-rank subspace of $G ^ { \natural }$ . PMD reformulates the leading singular component as a constrained bilinear maximization with additional sparsity penalties on the factors:

$$
\operatorname* { m a x } _ { u , v } \ u ^ { \top } G v \quad \mathrm { s . t . } \quad \| u \| _ { 2 } \leq 1 , \ \| v \| _ { 2 } \leq 1 , \ \| u \| _ { 1 } \leq c _ { 1 } , \ \| v \| _ { 1 } \leq c _ { 2 } ,\tag{7}
$$

and subsequent components are obtained by deflation. When $c _ { 1 } , c _ { 2 }$ are large enough, Eq. 7 reduces to the variational form of SVD; when they are tightened, the recovered factors are encouraged to concentrate on a small set of coordinates rather than spreading energy uniformly across all dimensions. This sparsity prior is well aligned with our setting: the true fine-tuning direction is expected to live on a low-rank and structurally compact subspace of the parameter space, while the noise introduced by irrelevant $x _ { t }$ tends to be diffuse. By penalizing diffuse factors, PMD is biased away from samplespecific fluctuations and toward directions that are consistently expressed across gradient samples. We therefore expect PMD to yield a $G ^ { \natural }$ that is (i) better aligned with the large-sample limit and (ii) more robust to perturbations on individual samples.

Empirical verification. We validate the golden channel with the experiment summarized in Fig. 2. We treat the gradient estimated from a large number of $x _ { t }$ samples (50000 iters) as a proxy for the true optimization direction, and measure how well the rank-r subspace recovered from a small-sample gradient aligns with it, using the arc sine of the principal angle (smaller is better). We compare two decompositions, SVD and PMD, in two regimes: clean small-sample gradients, and under two kinds of noise (Gaussian noise sampled from a Gaussian distribution and random label noise sampled from another dataset).

Two observations support the use of PMD. (i) Better alignment. Across all sample budgets, the PMD subspace is markedly closer to the large-sample direction than the SVD subspace; for instance, with only 10 iters PMD already attains a closer angle compared with the $1 0 ^ { 4 }$ sampled SVD result. (ii) Better noise robustness. Under additive Gaussian and sample from another dataset perturbation, SVD degrades substantially, whereas PMD remains nearly unchanged, indicating that the PMD subspace is governed by the cross-sample shared signal rather than by sample-specific fluctuations.

Together, these results confirm that PMD recovers a $G ^ { \natural }$ that is both more faithful to the true fine-tuning direction and more stable under noise—exactly the properties we need for a reliable reinitialization of LoRA.

## 3.4 LoRA Initialization via Spectral Decomposition

The training pipeline of PRISM-LoRA consists of three stages: gradient estimation, parameter initialization, and fine-tuning. We will introduce the process of each stage in the following.

Stage 1: Gradient Estimation. In the first stage, we collect M gradient samples by performing forward-backward passes under the principal timestep distribution:

$$
G ^ { \sharp } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \nabla _ { \theta } { \mathcal L } _ { \mathrm { F M } } \big ( \theta ; x _ { 0 } ^ { ( i ) } , x _ { 1 } ^ { ( i ) } , t _ { i } \big ) , \quad t _ { i } \sim p ^ { \sharp } ( t ) , ~ x _ { 0 } ^ { ( i ) } \sim { \mathcal N } ( 0 , I ) , ~ x _ { 1 } ^ { ( i ) } \sim p _ { \mathrm { d a t a } } ,\tag{8}
$$

where θ denotes the (frozen) base-model parameters of the target weight matrix, M is the number of gradient samples, $\mathcal { L } _ { \mathrm { F M } }$ is the flow-matching loss defined in Eq. $( 1 0 ) , \bar { p ^ { \sharp } } ( t )$ is the principal timestep distribution supported on a low-t subinterval of $[ 0 , 1 ] , x _ { 0 } ^ { ( i ) }$ and $x _ { 1 } ^ { ( i ) }$ are the noise and data endpoints of the i-th sample, and $G ^ { \natural }$ is the resulting low-rank gradient estimate on which the subsequent decomposition operates.

Stage 2: Parameter Initialization. In the second stage, we apply the golden-channel sparse SVD to $\breve { G } ^ { \sharp }$ and initialize the LoRA factors from its leading spectral components:

$$
A _ { 0 } = \sqrt { \gamma } \left[ U _ { G ^ { \sharp } } \right] _ { [ : , 1 : r ] } \left[ S _ { G ^ { \sharp } } ^ { 1 / 2 } \right] _ { [ 1 : r ] } , \quad B _ { 0 } = \sqrt { \gamma } \left[ S _ { G ^ { \sharp } } ^ { 1 / 2 } \right] _ { [ 1 : r ] } \left[ V _ { G ^ { \sharp } } \right] _ { [ : , 1 : r ] } ^ { \top } ,\tag{9}
$$

where $U _ { G \natural } , S _ { G \natural }$ , and $V _ { G ^ { \natural } }$ are the left singular vectors, singular values, and right singular vectors produced by the golden-channel sparse decomposition of $G ^ { \natural } ;$ r is the target LoRA rank; $[ \cdot ] _ { [ : , 1 : r ] }$ and $[ \cdot ] _ { [ 1 : r ] }$ denote truncation to the top-r components; $\gamma$ is the LoRA scaling factor; and $A _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ and $\dot { B } _ { 0 } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ are the initialized LoRA down- and up-projection matrices, respectively.

Stage 3: Fine-tuning. In the final stage, we fine-tune the LoRA factors $( A , B )$ initialized from Eq. (9) using the standard flow-matching objective, without any further modification to the timestep distribution:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } ( 0 , 1 ) , ~ x _ { 0 } \sim \mathcal { N } ( 0 , I ) , ~ x _ { 1 } \sim p _ { \mathrm { d a t a } } } \Big [ \big \| v _ { \theta } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \big \| _ { 2 } ^ { 2 } \Big ] , \quad x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } , } \end{array}\tag{10}
$$

where $v _ { \theta } ( \cdot , \cdot )$ is the flow-matching velocity predictor parameterized by the base weights and the trainable LoRA factors, $t \sim \mathcal { U } ( 0 , 1 )$ is the training timestep sampled uniformly over the full interval, $x _ { t }$ is the linear interpolant between noise $x _ { 0 }$ and data $x _ { 1 }$ , and $x _ { 1 } - x _ { 0 }$ is the target velocity field. During this stage, the base parameters are kept frozen, and only $( A , B )$ are updated.

Table 1: Quantitative comparison with existing methods on different text-to-image generation tasks. The bold and underlined figures represent the optimal and sub-optimal results, respectively.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Methods / Setting</td><td colspan="2">Controllability</td><td colspan="2">Alignment</td><td rowspan="2">DINO↑</td><td colspan="3">Image Quality</td></tr><tr><td>F1↑</td><td>MSE↓</td><td>CLIP Text↑</td><td>CLIP Image↑</td><td>FID↓</td><td>SSIM↑</td><td>PSNR ↑</td></tr><tr><td rowspan="5">Dreambooth</td><td>SD3-medium</td><td></td><td></td><td>0.224</td><td>0.793</td><td>0.667</td><td>170.0</td><td></td><td></td></tr><tr><td>+ Pissa</td><td></td><td></td><td>0.104</td><td>0.475</td><td>0.024</td><td>418.0</td><td></td><td></td></tr><tr><td>+ LoRA-GA</td><td></td><td></td><td>0.168</td><td>0.699</td><td>0.533</td><td>260.0</td><td></td><td></td></tr><tr><td>+ LoRA-One</td><td></td><td></td><td>0.224</td><td>0.795</td><td>0.667</td><td>169.1</td><td></td><td></td></tr><tr><td>+ PRISM-LoRA</td><td></td><td></td><td>0.230</td><td>0.805</td><td>0.682</td><td>160.3</td><td></td><td></td></tr><tr><td rowspan="5">Canny</td><td>OminiControl</td><td>0.4763</td><td></td><td>0.2203</td><td>0.7085</td><td>0.5011</td><td>97.56</td><td>0.3702</td><td>8.57</td></tr><tr><td>+ Pissa</td><td>0.0588</td><td></td><td>0.1466</td><td>0.4964</td><td>-0.0166</td><td>456.62</td><td>0.2297</td><td>5.67</td></tr><tr><td>+ LoRA-GA</td><td>0.4305</td><td></td><td>0.2128</td><td>0.6802</td><td>0.4552</td><td>108.25</td><td>0.2397</td><td>7.34</td></tr><tr><td>+ LoRA-One</td><td>0.4893</td><td></td><td>0.2232</td><td>0.7214</td><td>0.5339</td><td>95.93</td><td>0.3618</td><td>8.73</td></tr><tr><td>+ PRISM-LoRA</td><td>0.4978</td><td></td><td>0.2227</td><td>0.7242</td><td>0.5471</td><td>95.38</td><td>0.3311</td><td>8.82</td></tr><tr><td rowspan="5">Depth</td><td>OminiControl</td><td></td><td>904.9</td><td>0.2082</td><td>0.6601</td><td>0.4470</td><td>101.88</td><td>0.3376</td><td>7.91</td></tr><tr><td>+ Pissa</td><td></td><td>9416</td><td>0.1650</td><td>0.5233</td><td>-0.0034</td><td>500.50</td><td>0.3140</td><td>8.46</td></tr><tr><td>+ LoRA-GA</td><td></td><td>12160</td><td>0.0766</td><td>0.4959</td><td>0.1319</td><td>283.29</td><td>0.2333</td><td>9.67</td></tr><tr><td>+ LoRA-One</td><td></td><td>1117</td><td>0.2022</td><td>0.6491</td><td>0.3999</td><td>119.16</td><td>0.3287</td><td>8.01</td></tr><tr><td>+ PRISM-LoRA</td><td></td><td>851.5</td><td>0.2074</td><td>0.7587</td><td>0.3939</td><td>127.66</td><td>0.3317</td><td>9.45</td></tr><tr><td rowspan="5">Deblur</td><td>OminiControl</td><td></td><td>89.39</td><td>0.2518</td><td>0.8083</td><td>0.6660</td><td>91.47</td><td>0.4835</td><td>15.99</td></tr><tr><td>+ Pissa</td><td></td><td>91.43</td><td>0.2497</td><td>0.7932</td><td>0.6465</td><td>102.0</td><td>0.4731</td><td>15.64</td></tr><tr><td>+ LoRA-GA</td><td></td><td>85.24</td><td>0.2517</td><td>0.8137</td><td>0.6575</td><td>91.70</td><td>0.4840</td><td>15.90</td></tr><tr><td>+ LoRA-One</td><td></td><td>69.07</td><td>0.2528</td><td>0.8203</td><td>0.6589</td><td>92.72</td><td>0.4913</td><td>16.53</td></tr><tr><td>+ PRISM-LoRA</td><td></td><td>55.87</td><td>0.2535</td><td>0.8328</td><td>0.7232</td><td>77.85</td><td>0.5224</td><td>17.32</td></tr></table>

## 4 Experiment

Experimental Setting. We evaluate our method against sota methods across different image generation tasks: subject-driven generation, spatially-aligned generation, image deblurring, etc. To prove PRISM-LoRA can be effective across models, we deploy PRISM-LoRA on two mainstream models, FLUX-1 [18] and Stable Diffusion-3 [7]. Regarding dataset setting, we use the last 100,000 images in the MultiGen-20M [50] dataset for training, and last 2,500 images in COCO for evaluation in Canny, Depth, and Deblur tasks. For Dreambooth, we use the original 30 categories of images in training and generate 4 images with 25 prompts, with a total of 3,000 images for evaluation. See more details about hyperparameter and other experiment settings in Appendix. C.

![](images/6424151b5b6910a7ba571d6c9eeb7bccdbb78f0a66b9aae2ca6bd6b73d4e7b9f.jpg)  
(a) Subject-driven generation.

![](images/961a8002253ccef75a75d590716e838c9adee9a430b03c185be544f1f623ed3b.jpg)  
(b) Canny-guided spatially-aligned generation.

![](images/763589aacac517c0ff7f1e5968f06db295e4cfae76653bac42b0dbe1fc12c5aa.jpg)  
(c) Depth-guided spatially-aligned generation.

![](images/89e85b9e4034bbe62c4a65e7d4c6615a3aebcc876d9818fd8d1f25e30bdde42d.jpg)  
(d) Image deblurring generation.  
Figure 3: Visual comparison with sota methods across various text-to-image generation tasks based on LoRA fine-tuning. Our method more rapidly aligns with the target distribution, and optimizes the final generation performance. Caption: A sks dog on the beach. A high resolution picture of building. A high resolution picture ofin-door decoration. A high resolution picture ofkitchen.

We compare our method with other spectral-init methods, Pissa [25], LoRA-GA [39], and LoRA-One [47]. In the spatial alignment task, we use OminiControl [35] as the baseline method, which uses the vanilla LoRA initialization method, and deploy the above spectral-init methods.

Evaluation Metrics. We evaluate generated images along three axes. Controllability measures fidelity to the input condition: F1 score between Canny edges extracted from the generated and input images for the Canny task, and MSE between predicted and input depth maps for the Depth task. Alignment assesses semantic consistency via CLIP Text (text-image similarity), CLIP Image (image-image similarity), and DINO [2] feature similarity against the reference. Image quality is reported in terms of FID [11], SSIM, and PSNR. Further details are provided in Appendix. C.

## 4.1 Main Results

Quantitative comparison. Table 1 summarizes the performance across the four evaluated tasks. Our proposed PRISM-LoRA demonstrates superior controllability across all spatially-aligned and restoration tasks, achieving peak performance with an F1 score of 0.4978 on Canny, and minimizing MSE to 851.5 on Depth and 55.87 on Deblur. Notably, on the Deblur task, PRISM-LoRA consistently outperforms the OmniControl baseline across nearly all metrics, significantly improving FID from 91.47 to 77.85 and PSNR from 15.99 to 17.32. For subject-driven generation (DreamBooth), our method establishes robust leadership among LoRA variants in both semantic alignment (CLIP Text: 0.230, CLIP Image: 0.805, DINO: 0.682) and image quality (FID: 160.3), and achieve an average of 557 steps to reach best score, compared with LoRA-Base with 650, LoRA-GA 957 and LoRA-One 757 steps. Furthermore, while improvements in alignment and fidelity on Canny and Depth are relatively modest, PRISM-LoRA remains highly competitive, consistently matching or surpassing existing spectral-init variants. It is also crucial to note that PiSSA experiences severe performance degradation on most tasks (e.g., yielding an FID of 456.62 on Canny and 500.50 on Depth). This stark contrast underscores the critical importance of gradient initialization quality and confirms that PRISM-LoRA provides a fundamentally more robust optimization starting point.

Qualitative comparison. Fig. 3 provides visual comparisons across the four downstream tasks at various training milestones. Across all settings, PRISM-LoRA exhibits a substantially accelerated convergence rate compared to both vanilla LoRA and LoRA-One. Specifically, coherent semantic structures emerge at remarkably early iterations (e.g., 10–500 steps), whereas the baselines continue to yield blurry, misaligned, or content-collapsed outputs at the equivalent stages. In the DreamBooth task, PRISM-LoRA faithfully preserves the unique identity of the reference subject (the specific dog) throughout the entire training trajectory. In contrast, vanilla LoRA suffers from concept drift toward generic dogs, and LoRA-One struggles to recover the subject identity until much later steps. For spatially-aligned generation (Canny and Depth), our model adheres more rigorously to the input conditioning signals, delivering sharper edge boundaries and highly accurate spatial layouts. Finally, in the image deblurring task, PRISM-LoRA not only restores high-frequency details earlier in the training process but also achieves perceptually cleaner reconstructions at convergence.

Table 2: Ablation experiments on Dreambooth and Canny-guided controllable generation.
<table><tr><td rowspan=1 colspan=1>Tasks</td><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>F1↑</td><td rowspan=1 colspan=1>FID↓  CLIP-I↑  DINO↑</td></tr><tr><td rowspan=2 colspan=1>Dreambooth</td><td rowspan=2 colspan=1>LoRA-Onew. principal timestepw. golden channelPRISM-LoRA</td><td rowspan=2 colspan=1>1一</td><td rowspan=1 colspan=1>169.1   0.795    0.667</td></tr><tr><td rowspan=1 colspan=1>161.7   0.798    0.677158.9   0.800    0.674160.3   0.805    0.682</td></tr><tr><td rowspan=4 colspan=1>Canny</td><td rowspan=4 colspan=1>LoRA-Onew. principal timestepw. golden channelPRISM-LoRA</td><td rowspan=4 colspan=1>0.48930.48340.49120.4978</td><td rowspan=1 colspan=1>95.93   0.7214   0.5339</td></tr><tr><td rowspan=1 colspan=1>95.82  0.7135   0.5351</td></tr><tr><td rowspan=1 colspan=1>97.14  0.7158   0.5387</td></tr><tr><td rowspan=1 colspan=1>95.38  0.7242   0.5471</td></tr></table>

## 4.2 Ablation Study

To isolate the individual contributions of our proposed modules—Principal Timestep selection and Golden Channel filtering—we evaluate their independent and combined effects on the Dreambooth (subject-driven generation) and Canny (spatially-aligned generation) tasks. As observed in Table. 2, integrating either the principal timestep or the golden channel independently into the strong LoRA-One baseline yields overall performance improvements, particularly in terms of identity preservation and image quality. On the Dreambooth task, applying the Golden Channel alone improves the DINO score from 0.667 to 0.674 and substantially reduces FID, suggesting that filtering out task-irrelevant channels is crucial for maintaining optimization stability even under stochastic timestep sampling.

While individual components exhibit distinct merits, their combination in the full PRISM-LoRA framework achieves the optimal performance balance. Interestingly, on the Canny task, while applying independent modules causes slight fluctuations in specific metrics, the complete PRISM-LoRA clearly dominates, achieving the highest F1 score (0.4978) and DINO score (0.5471). This demonstrates a strong synergistic effect: suppressing the effective gradient rank (via Principal Timestep) creates a cleaner spectral foundation, which enables the sparse decomposition (via Golden Channel) to isolate the optimal fine-tuning directions more accurately. Together, they successfully mitigate the high-rank gradient dilemma inherent in flow-matching diffusion models.

## 5 Conclusion

In this paper, we identify and resolve a fundamental mismatch between the low-rank assumption of standard spectral initialization and the intrinsically high-rank gradient geometry found in flowmatching diffusion models. We demonstrate that stochastic timestep sampling induces directionally heterogeneous gradients, which renders conventional low-rank approximations lossy and misaligned. To overcome this barrier, we propose PRISM-LoRA, a principled spectral initialization framework specifically tailored for diffusion fine-tuning. By synergizing Principal Timestep selection to inherently suppress the effective gradient rank, and Golden Channel sparse decomposition to filter out task-irrelevant noise, PRISM-LoRA ensures that the initialized parameters robustly align with the long-horizon optimization trajectory.

Limitation and future works While our approach achieves both faster convergence and strong performance on this task, a more comprehensive validation across diverse subject-driven scenarios remains an interesting direction for future work. On the theoretical side, our analysis primarily focuses on the observation that oversampling intermediate timesteps can be harmful to gradient quality. However, these timesteps also exhibit the largest variance, suggesting that they carry informative gradient signals that should be properly trained and accelerated rather than downweighted. Extending our theory to characterize the spectral behavior across the full range of diffusion timesteps is a promising avenue for future investigation.

## Acknowledgments and Disclosure of Funding

Use unnumbered first level headings for the acknowledgments. All acknowledgments go at the end of the paper before the list of references. Moreover, you are required to declare funding (financial activities supporting the submitted work) and competing interests (related financial activities outside the submitted work). More information about this disclosure can be found at: https: //neurips.cc/Conferences/2026/PaperInformation/FundingDisclosure.

Do not include this section in the anonymized submission, only in the final paper. You can use the ack environment provided in the style file to automatically hide this section in the anonymized submission.

## References

[1] X. Cai, Y. Wu, Q. Chen, H. Wu, L. Xiang, and H. Wen. Shortcutting pre-trained flow matching diffusion models is almost free lunch. arXiv preprint arXiv:2510.17858, 2025.

[2] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021.

[3] C. Chen and J. Mo. IQA-PyTorch: Pytorch toolbox for image quality assessment. [Online]. Available: https://github.com/chaofengc/IQA-PyTorch, 2022.

[4] X. Chen, Z. Zhang, H. Zhang, Y. Zhou, S. Y. Kim, Q. Liu, Y. Li, J. Zhang, N. Zhao, Y. Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12501–12511, 2025.

[5] Z. Chen, Y. Wang, X. Cai, Z. You, Z. Lu, F. Zhang, S. Guo, and T. Xue. Ultrafusion: Ultra high dynamic imaging using exposure fusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16111–16121, 2025.

[6] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36:10088–10115, 2023.

[7] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis, march 2024. URL http://arxiv. org/abs/2403.03206, 2024.

[8] Z. Fang, L. Xiang, X. Cai, K. Zhou, and H. Wen. Flexcontrol: Computation-aware conditional control with differentiable router for text-to-image generation. In Forty-second International Conference on Machine Learning, 2025.

[9] Z. Fang, L. Xiang, X. Cai, B. Wang, B. Yang, and H. Wen. Dynfusion: Rethinking condition fusion for adaptive multi-conditional text-to-image generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[10] S. Hayou, N. Ghosh, and B. Yu. Lora+: Efficient low rank adaptation of large models. arXiv preprint arXiv:2402.12354, 2024.

[11] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[12] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[13] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

[14] M. Hua, J. Liu, F. Ding, W. Liu, J. Wu, and Q. He. Dreamtuner: Single image is enough for subject-driven generation. arXiv preprint arXiv:2312.13691, 2023.

[15] H. Huang and R. Balestriero. Allora: Adaptive learning rate mitigates lora fatal flaws. arXiv preprint arXiv:2410.09692, 2024.

[16] Y. Ji, K. Ma, H. Cai, A. Zhang, L. Ma, and X. Tan. Lidarpainter: One-step away from any lidar view to novel guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5332–5340, 2026.

[17] W. Jia, M. Huang, N. Chen, L. Zhang, and Z. Mao. Dˆ 2it: Dynamic diffusion transformer for accurate image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12860–12870, 2025.

[18] B. F. Labs. Flux: Official inference repository for flux.1 models, 2024. URL https://github. com/black-forest-labs/flux. Accessed: 2024-11-12.

[19] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[20] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng, and M.-H. Chen. Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning, 2024.

[21] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[22] X. Liu, X. Zhang, J. Ma, J. Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.

[23] S. Luo, Y. Tan, S. Patil, D. Gu, P. Von Platen, A. Passos, L. Huang, J. Li, and H. Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.

[24] Y. Mei, Y. Zeng, H. Zhang, Z. Shu, X. Zhang, S. Bi, J. Zhang, H. Jung, and V. M. Patel. Holo-relighting: Controllable volumetric portrait relighting from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4263–4273, 2024.

[25] F. Meng, Z. Wang, and M. Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems, 37: 121038–121072, 2024.

[26] C. Mou, X. Wang, L. Xie, Y. Wu, J. Zhang, Z. Qi, and Y. Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024.

[27] B. Peng, J. Wang, Y. Zhang, W. Li, M.-C. Yang, and J. Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024.

[28] K. Ponkshe, R. Singhal, E. Gorbunov, A. Tumanov, S. Horvath, and P. Vepakomma. Initialization using update approximation is a silver bullet for extremely efficient low-rank fine-tuning. arXiv preprint arXiv:2411.19557, 2024.

[29] Y. Pu, Z. Xia, J. Guo, D. Han, Q. Li, D. Li, Y. Yuan, J. Li, Y. Han, S. Song, et al. Efficient diffusion transformer with step-wise dynamic attention mediators. In European Conference on Computer Vision, pages 424–441. Springer, 2024.

[30] Z. Qiu, W. Liu, H. Feng, Y. Xue, Y. Feng, Z. Liu, D. Zhang, A. Weller, and B. Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. Advances in Neural Information Processing Systems, 36:79320–79362, 2023.

[31] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[32] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.

[33] V. Soboleva, A. Alanov, A. Kuznetsov, and K. Sobolev. T-lora: Single image diffusion model customization without overfitting. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 9051–9059, 2026.

[34] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.

[35] Z. Tan, S. Liu, X. Yang, Q. Xue, and X. Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.

[36] Z. Tan, Q. Xue, X. Yang, S. Liu, and X. Wang. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280, 2025.

[37] H. Wang, J. Peng, Q. He, H. Yang, Y. Jin, J. Wu, X. Hu, Y. Pan, Z. Gan, M. Chi, et al. Unicombine: Unified multi-conditional combination with diffusion transformer. arXiv preprint arXiv:2503.09277, 2025.

[38] J. Wang, L. Bhagat, C. Yang, Y. Xu, Y. Shen, H. Li, and B. Zhou. Spatial steerability of gans via self-supervision from discriminator. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9493–9507, 2024.

[39] S. Wang, L. Yu, and J. Li. Lora-ga: Low-rank adaptation with gradient approximation. In A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

[40] S. Weiland and F. Van Belzen. Singular value decompositions and low rank approximations of tensors. IEEE transactions on signal processing, 58(3):1171–1182, 2009.

[41] D. M. Witten, R. Tibshirani, and T. Hastie. A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis. Biostatistics, 10(3):515–534, 2009.

[42] J. Z. Wu, Y. Zhang, H. Turki, X. Ren, J. Gao, M. Z. Shou, S. Fidler, Z. Gojcic, and H. Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26024–26035, 2025.

[43] Z. Yan, J. Wang, P. Jin, K.-Y. Zhang, C. Liu, S. Chen, T. Yao, S. Ding, B. Wu, and L. Yuan. Orthogonal subspace decomposition for generalizable ai-generated image detection. arXiv preprint arXiv:2411.15633, 2024.

[44] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

[45] L. Zhang, L. Zhang, S. Shi, X. Chu, and B. Li. Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning. arXiv preprint arXiv:2308.03303, 2023.

[46] L. Zhang, A. Rao, and M. Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In The Thirteenth International Conference on Learning Representations, 2025.

[47] Y. Zhang, F. Liu, and Y. Chen. Lora-one: One-step full gradient could suffice for fine-tuning large language models, provably and efficiently. arXiv preprint arXiv:2502.01235, 2025.

[48] Y. Zhang, Y. Yuan, Y. Song, H. Wang, and J. Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19513–19524, 2025.

[49] J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian. Galore: Memory-efficient llm training by gradient low-rank projection. arXiv preprint arXiv:2403.03507, 2024.

[50] S. Zhao, D. Chen, Y.-C. Chen, J. Bao, S. Hao, L. Yuan, and K.-Y. K. Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in neural information processing systems, 36:11127–11150, 2023.

[51] W. Zhao, Y. Han, J. Tang, K. Wang, H. Luo, Y. Song, G. Huang, F. Wang, and Y. You. Dydit++: Dynamic diffusion transformers for efficient visual generation. arXiv preprint arXiv:2504.06803, 2025.

[52] B. Zi, X. Qi, L. Wang, J. Wang, K.-F. Wong, and L. Zhang. Delta-lora: Fine-tuning high-rank parameters with the delta of low-rank matrices. arXiv preprint arXiv:2309.02411, 2023.

## Appendix

## A Preliminaries

## A.1 Flow Matching Models

Prior diffusion models based on denoising diffusion probabilistic models (DDPMs) [12] learn generative processes by gradually adding Gaussian noise to the data and training a model to reverse this stochastic diffusion process. In contrast, Flow Matching (FM) [21] formulates generative modeling as learning a continuous-time velocity field that deterministically transports samples from a simple prior distribution to the data distribution. Let $x _ { 0 } \sim p _ { 0 } ( x )$ and $x _ { 1 } \sim q ( x )$ denote noise and data samples. The interpolation path is as follows:

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } .\tag{11}
$$

As above, a vector field $v _ { \theta } ( x , t ) -$ —parameterized by a neural network, is constructed to approximate a target vector field $v _ { t } ( x )$ , which transports a simple prior distribution $p _ { 0 } ( x )$ to a complex data distribution $p _ { 1 } ( x ) \approx q ( x )$ via an ordinary differential equation (ODE): $\begin{array} { r } { \frac { d } { d t } \phi _ { t } ( x ) = v _ { t } ( \phi _ { t } ( x ) ) } \end{array}$ ) with $\phi _ { 0 } ( x ) = x$ . The ideal marginal FM objective is defined as:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , x \sim p _ { t } ( x ) } \left[ \big \| v _ { \theta } ( x , t ) - v _ { t } ( x ) \big \| _ { 2 } ^ { 2 } \right] .\tag{12}
$$

However, the marginal probability path $p _ { t } ( x )$ and the vector field $v _ { t } ( x )$ are generally unknown. To bypass this intractability, CFM [19] constructs the target vector field by conditioning on individual data samples $x _ { 1 } \sim q ( x _ { 1 } )$ . Given a conditional probability path $p _ { t } ( x | x _ { 1 } )$ ) and its generating vector field $v _ { t } ( x | x _ { 1 } )$ , the tractable CFM objective is:

$$
\begin{array} { r l } & { \mathcal { L } ^ { * } ( \theta ) = \mathbb { E } _ { t , x _ { 1 } \sim q ( x _ { 1 } ) , x \sim p _ { t } ( x \mid x _ { 1 } ) } \left[ \| v _ { \theta } ( x , t ) - v _ { t } ( x \mid x _ { 1 } ) \| _ { 2 } ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { t , x 1 \sim q ( x _ { 1 } ) , x \sim p _ { t } ( x \mid x _ { 1 } ) } \left[ \| v _ { \theta } ( ( 1 - t ) x _ { 0 } + t x _ { 1 } , t ) - ( x _ { 1 } - x _ { 0 } ) \| _ { 2 } ^ { 2 } \right] } \end{array}\tag{13}
$$

Minimizing $\mathcal { L } _ { \theta } ^ { \ast }$ is mathematically equivalent to minimizing the intractable $\mathcal { L } _ { \theta }$ up to a constant. The inference process solves the ODE backward from $t = 0 \mathrm { t o } t = 1$ through n iterative updates:

$$
x _ { t _ { i + 1 } } = x _ { t _ { i } } - ( t _ { i } - t _ { i + 1 } ) v _ { \theta } ( x _ { t _ { i } } , t _ { i } ) , \quad x _ { t _ { 0 } } \sim \mathcal { N } ( 0 , I ) ,\tag{14}
$$

with final output $x _ { t _ { n } }$ as the generated sample.

## A.2 Gradient-guided Parameter Efficient Fine-tuning

Low-Rank Adaptation [13] is one of the parameter efficient fine-tuning methods that adapts large pre-trained models by injecting trainable low-rank matrices into existing weight layers. Given a pre-trained weight matrix $W \in \overline { { \mathbb { R } } } ^ { d \times k }$ , LoRA models its update as a low-rank decomposition:

$$
\begin{array} { r } { W ^ { \prime } = W + \Delta W , \quad \Delta W = \eta B A , } \end{array}\tag{15}
$$

where $B \in \mathbb { R } ^ { d \times r }$ and $A \in \mathbb { R } ^ { r \times k } , r \ll \operatorname* { m i n } ( d , k )$ denotes the rank, and $\eta$ is a scaling factor. This significantly reduces the number of trainable parameters and memory footprint.

Theoretically, the update direction of the LoRA module can be aligned with the full fine-tuning at the early training stage via the first gradient descent step, which can be mathematically expressed as:

$$
\eta \left( \Delta B A _ { \mathrm { i n i t } } + B _ { \mathrm { i n i t } } \Delta A \right) = \eta \lambda \left[ \nabla _ { B } \mathcal { L } ( B _ { \mathrm { i n i t } } ) A _ { \mathrm { i n i t } } + B _ { \mathrm { i n i t } } \nabla _ { A } \mathcal { L } ( A _ { \mathrm { i n i t } } ) \right] ,\tag{16}
$$

To measure its approximation quality of scaled the update of the weights in full fine-tuning $\zeta \Delta W =$ $\zeta \lambda \nabla _ { W } \mathcal { L } ( W _ { 0 } )$ , the Frobenius norm of the difference between these two updates [39, 47] is commonly used as a criterion:

$$
\begin{array} { r l } & { \quad \| \eta \left( \Delta B A _ { \mathrm { i n i t } } + B _ { \mathrm { i n i t } } \Delta A \right) - \zeta \lambda \nabla _ { W } \mathcal { L } ( W _ { 0 } ) \| _ { F } } \\ & { = \lambda \| \eta \nabla _ { B } \mathcal { L } ( B _ { \mathrm { i n i t } } ) A _ { \mathrm { i n i t } } + \eta B _ { \mathrm { i n i t } } \nabla _ { A } \mathcal { L } ( A _ { \mathrm { i n i t } } ) - \zeta \nabla _ { W } \mathcal { L } ( W _ { 0 } ) \| _ { F } } \\ & { = \lambda \| \eta ^ { 2 } \nabla _ { W ^ { \prime } } \mathcal { L } ( W _ { 0 } ) \cdot A _ { \mathrm { i n i t } } ^ { T } A _ { \mathrm { i n i t } } + \eta ^ { 2 } B _ { \mathrm { i n i t } } B _ { \mathrm { i n i t } } ^ { T } \cdot \nabla _ { W } \mathcal { L } ( W _ { 0 } ) - \zeta \nabla _ { W } \mathcal { L } ( W _ { 0 } ) \| _ { F } . } \end{array}\tag{17}
$$

The svd decomposition can be formulated as $G ^ { \natural } = U _ { G ^ { \natural } } S _ { G ^ { \natural } } V _ { G ^ { \natural } }$ . Matrix U and V are orthogonal matrix, and S is a strictly increasing sequence of singular value formulated as $S _ { G ^ { \natural } } = \{ \sigma _ { 1 } , \sigma _ { 2 } , . . . \sigma _ { d } \}$

## B Detailed proof of principal timesteps

## B.1 Problem Formulation

Consider the linear interpolation $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ , where $x _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$ is independent of $x _ { 1 }$ Throughout this section, we adopt the Gaussian surrogate assumption $x _ { 1 } \sim \dot { \mathcal { N } } ( 0 , \Sigma _ { 1 } )$ , which yields closed-form conditional covariances; for non-Gaussian $p _ { \mathrm { d a t a } }$ the expressions below serve as the best linear-Gaussian approximation of $\mathrm { V a r } [ x _ { 0 } \mid x _ { t } ]$ . Let the eigenvalues of $\Sigma _ { 1 }$ be

$$
\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { d } > 0 ,
$$

and assume $\lambda _ { 1 } > \lambda _ { d }$ (non-degenerate spectrum). The (rescaled) gradient covariance matrix is defined as

$$
\Sigma _ { v } ( t ) = { \frac { 1 } { t ^ { 2 } } } \ \operatorname { V a r } [ x _ { 0 } \mid x _ { t } ] .\tag{18}
$$

To obtain Eq. 18, we need to first make a simple conversion for $x _ { 1 }$ . According to the flow-matching identity, we can express $x _ { 1 }$ as $\begin{array} { r } { x _ { 1 } = \frac { x _ { t } - ( 1 - \bar { t } ) x _ { 0 } } { t } } \end{array}$ . As a result, the model prediction $v ( x _ { t } , t )$ can be express as

$$
v ( x _ { t } , t ) = \mathbb { E } [ x _ { 1 } - x _ { 0 } \mid x _ { t } ] = \mathbb { E } \left[ { \frac { x _ { t } - x _ { 0 } } { t } } \mid x _ { t } \right] = { \frac { x _ { t } - \mathbb { E } [ x _ { 0 } \mid x _ { t } ] } { t } } .\tag{19}
$$

From this expression, we observe that the velocity field is determined by posterior expectation. Finally, with the irreducible noise $\begin{array} { r } { x _ { 1 } - x _ { 0 } - v ( x _ { t } , t ) = \frac { \mathbb { E } [ x _ { 0 } | x _ { t } ] - x _ { 0 } } { t } } \end{array}$ , the covariance of the irreducible noise is Cov $\begin{array} { r } { \left( \frac { \mathbb { E } \left[ x _ { 0 } \vert x _ { t } \right] - x _ { 0 } } { t } \mid x _ { t } \right) = \frac { 1 } { t ^ { 2 } } \operatorname { V a r } [ x _ { 0 } \mid x _ { t } ] } \end{array}$

Derivation of the spectrum. Because $x _ { 0 } ~ \perp ~ x _ { 1 }$ , the marginal covariance of $x _ { t }$ is $\mathrm { C o v } ( \boldsymbol { x } _ { t } ) \ = $ $( 1 - t ) ^ { 2 } I _ { d } + t ^ { 2 } \Sigma _ { 1 }$ , and the cross-covariance is $\operatorname { C o v } ( x _ { 0 } , x _ { t } ) = ( 1 - t ) I _ { d }$ . Under the joint-Gaussian assumption, the conditional covariance is

$$
\operatorname { V a r } [ x _ { 0 } \mid x _ { t } ] = I _ { d } - ( 1 - t ) ^ { 2 } \big [ ( 1 - t ) ^ { 2 } I _ { d } + t ^ { 2 } \Sigma _ { 1 } \big ] ^ { - 1 } .\tag{20}
$$

Diagonalizing in the eigenbasis of $\Sigma _ { 1 }$ , the i-th eigenvalue of $\mathrm { V a r } [ x _ { 0 } \mid x _ { t } ]$ equals $\frac { t ^ { 2 } \lambda _ { i } } { ( 1 - t ) ^ { 2 } + t ^ { 2 } \lambda _ { i } }$ , so the eigenvalues $\sigma _ { i } ( t )$ of $\Sigma _ { v } ( t )$ satisfy

$$
\sigma _ { i } ( t ) = \frac { \lambda _ { i } } { ( 1 - t ) ^ { 2 } + t ^ { 2 } \lambda _ { i } } .\tag{21}
$$

## B.2 Spectral Comparison at t in {0, 0.5, 1}

Evaluating (21) at characteristic timesteps:

$\mathbf { A } \mathbf { t } t  0 \colon \sigma _ { i } ( 0 ) = \lambda _ { i }$ . The spectrum coincides with the data covariance $\Sigma _ { 1 }$

$$
\bullet \ \mathbf { A t } \ t = 0 . 5 \colon \sigma _ { i } ( 0 . 5 ) = { \frac { 4 \lambda _ { i } } { 1 + \lambda _ { i } } } .
$$

• At $t = 1 \colon \sigma _ { i } ( 1 ) = 1$ for all $i ,$ so $\Sigma _ { v } ( 1 ) = I _ { d }$ . Intuitively, when $x _ { t } ~ = ~ x _ { 1 }$ carries no information about $x _ { 0 } .$ , the conditional variance reduces to the prior $I _ { d }$ and the gradient direction is dominated entirely by noise.

## B.3 Majorization Lemma

We first establish a general majorization result from which the three-point comparison (and the full-range monotonicity) will follow.

Lemma 1 (Normalized-spectrum majorization). Let $\lambda = ( \lambda _ { 1 } , \ldots , \lambda _ { d } )$ with $\lambda _ { 1 } \geq \cdot \cdot \cdot \geq \lambda _ { d } > 0$ and $\lambda _ { 1 } > \lambda _ { d } .$ Let $f : ( 0 , \infty ) \to ( 0 , \infty )$ be strictly increasing, strictly concave, and $s a t i s f y \ f ( 0 ^ { + } ) = 0$ Define the normalized spectra

$$
\hat { a } _ { i } = \frac { \lambda _ { i } } { \sum _ { j } \lambda _ { j } } , \qquad \hat { b } _ { i } = \frac { f ( \lambda _ { i } ) } { \sum _ { j } f ( \lambda _ { j } ) } .
$$

Then $\hat { a } \succ \hat { b } ( s t r i c t l y ) ,$ , i.e.

$$
\sum _ { i = 1 } ^ { k } \widehat { a } _ { i } \ \geq \ \sum _ { i = 1 } ^ { k } \widehat { b } _ { i } \quad \forall k \in \{ 1 , \ldots , d \} ,
$$

with strict inequality for at least one $k < d .$

Proof. Both $\hat { a }$ and $\hat { b }$ sum to 1, and since $f$ is strictly increasing, both are non-increasing in i (same ordering). Consider the ratio

$$
\frac { \hat { a } _ { i } } { \hat { b } _ { i } } \ : = \ : \frac { \lambda _ { i } } { f ( \lambda _ { i } ) } \cdot \frac { \sum _ { j } f ( \lambda _ { j } ) } { \sum _ { j } \lambda _ { j } } .\tag{22}
$$

Strict concavity of $f$ together with $f ( 0 ^ { + } ) = 0$ implies that $\lambda \mapsto \lambda / f ( \lambda )$ is strictly increasing on $( 0 , \infty ) . \mathrm { \dot { \qquad } }$ <sup>3</sup> Since $\lambda _ { i }$ is non-increasing in i, the ratio $\hat { a } _ { i } / \hat { b } _ { i }$ is non-increasing in i; because $\lambda _ { 1 } > \lambda _ { d }$ , it is not constant.

Cut-property argument. Since $\hat { a } _ { i } / \hat { b } _ { i }$ is non-increasing and non-constant, and $\textstyle \sum _ { i } { \hat { a } } _ { i } = \sum _ { i } { \hat { b } } _ { i } = 1$ there exists a unique index $i ^ { \star } \in \{ 1 , \ldots , d - 1 \}$ such that

$$
\hat { a } _ { i } \geq \hat { b } _ { i } \mathrm { f o r } i \leq i ^ { \star } , \qquad \hat { a } _ { i } \leq \hat { b } _ { i } \mathrm { f o r } i > i ^ { \star } .
$$

For $k \leq i ^ { \star }$ , each summand in $\textstyle \sum _ { i = 1 } ^ { k } ( { \hat { a } } _ { i } - { \hat { b } } _ { i } )$ is non-negative, so the partial sum is $\geq 0$ . For $k > i ^ { \star }$ using $\begin{array} { r } { \sum _ { i = 1 } ^ { d } ( \hat { a } _ { i } - \hat { b } _ { i } ) = 0 } \end{array}$

$$
\sum _ { i = 1 } ^ { k } ( \hat { a } _ { i } - \hat { b } _ { i } ) \ = \ - \sum _ { i = k + 1 } ^ { d } ( \hat { a } _ { i } - \hat { b } _ { i } ) \ \geq \ 0 ,\tag{23}
$$

since each term in the tail sum is $\leq 0$ . Hence $\textstyle \sum _ { i = 1 } ^ { k } { \hat { a } } _ { i } \geq \sum _ { i = 1 } ^ { k } { \hat { b } } _ { i }$ for all k, with strict inequality at $k = i ^ { \star }$ □

## B.4 Full-Range Monotonicity of Spectral Flatness

We now promote the three-point comparison to full-range monotonicity in t.

Lemma 2 (Monotone majorization along t). For any $0 \leq s < t \leq 1$ , let $\hat { \sigma } ( s )$ and $\hat { \sigma } ( t )$ denote the normalized spectra of $\textstyle \cdot \sum _ { v } { \bar { ( } } s )$ and $\Sigma _ { v } ( t )$ respectively. Then $\hat { \sigma } ( s ) \succ \hat { \sigma } ( t )$

Proof. Define, for fixed $t \in [ 0 , 1 ]$ , the map $\begin{array} { r } { g _ { t } : ( 0 , \infty ) \to ( 0 , \infty ) \mathrm { ~ b y ~ } g _ { t } ( \lambda ) = \frac { \lambda } { ( 1 - t ) ^ { 2 } + t ^ { 2 } \lambda } } \end{array}$ , so that $\sigma _ { i } ( t ) = g _ { t } ( \lambda _ { i } )$ . A direct computation gives

$$
g _ { t } ^ { \prime } ( \lambda ) = \frac { ( 1 - t ) ^ { 2 } } { [ ( 1 - t ) ^ { 2 } + t ^ { 2 } \lambda ] ^ { 2 } } > 0 , \qquad g _ { t } ^ { \prime \prime } ( \lambda ) = - \frac { 2 t ^ { 2 } ( 1 - t ) ^ { 2 } } { [ ( 1 - t ) ^ { 2 } + t ^ { 2 } \lambda ] ^ { 3 } } \leq 0 ,
$$

so $g _ { t }$ is strictly increasing and (for $t \in ( 0 , 1 ) ,$ ) strictly concave; moreover $g _ { t } ( 0 ^ { + } ) = 0$

For $s = 0$ and any $t \in ( 0 , 1 ]$ , the claim $\hat { \sigma } ( 0 ) \succ \hat { \sigma } ( t )$ follows immediately from Lemma 1 applied with $f = g _ { t }$

For general $0 < s < t \leq 1$ , write $\sigma _ { i } ( t ) = ( g _ { t } \circ g _ { s } ^ { - 1 } ) ( \sigma _ { i } ( s ) )$ . Let $h = g _ { t } \circ g _ { s } ^ { - 1 }$ . Both $g _ { t }$ and $g _ { s } ^ { - 1 }$ are strictly increasing, so h is strictly increasing; and the composition of a concave increasing function with an increasing concave inverse (i.e. with a convex $g _ { s } ^ { - 1 } - \mathrm { n o t e }$ that the inverse of a concave increasing function is convex increasing) requires care. A direct calculation yields

$$
\begin{array} { l } { { h ( \mu ) = g _ { t } \bigg ( \frac { ( 1 - s ) ^ { 2 } \mu } { 1 - s ^ { 2 } \mu } \bigg ) } } \\ { { \ } } \\ { { \displaystyle = \frac { ( 1 - s ) ^ { 2 } \mu } { ( 1 - t ) ^ { 2 } ( 1 - s ^ { 2 } \mu ) + t ^ { 2 } ( 1 - s ) ^ { 2 } \mu } \ } } \\ { { \ \displaystyle = \frac { ( 1 - s ) ^ { 2 } \mu } { ( 1 - t ) ^ { 2 } + \big [ t ^ { 2 } ( 1 - s ) ^ { 2 } - s ^ { 2 } ( 1 - t ) ^ { 2 } \big ] \mu } . } } \end{array}\tag{24}
$$

<sup>3</sup>Indeed, $\begin{array} { r } { \frac { d } { d \lambda } \left( \lambda / f ( \lambda ) \right) = \left( f ( \lambda ) - \lambda f ^ { \prime } ( \lambda ) \right) / f ( \lambda ) ^ { 2 } } \end{array}$ ; strict concavity and $f ( 0 ^ { + } ) = 0 { \mathrm { ~ g i v e ~ } } f ( \lambda ) > \lambda f ^ { \prime } ( \lambda )$ for all $\lambda > 0$

Setting $\alpha = ( 1 - s ) ^ { 2 } > 0$ and $\beta = t ^ { 2 } ( 1 - s ) ^ { 2 } - s ^ { 2 } ( 1 - t ) ^ { 2 }$ , we have $\begin{array} { r } { h ( \mu ) = \frac { \alpha \mu } { ( 1 - t ) ^ { 2 } + \beta \mu } } \end{array}$ . For $s < t \leq 1$ one checks that $\beta > 0 ! ^ { 4 }$ hence h is of the same Möbius form $\displaystyle \frac { \alpha \mu } { c + \beta \mu }$ with $c , \alpha , \beta > 0 .$ , so h is strictly increasing, strictly concave, and $h ( 0 ^ { + } ) = 0$

Applying Lemma 1 to the spectrum $\sigma ( s )$ with transformation $f = h \mathrm { g i v e s } \hat { \sigma } ( s ) \succ \hat { \sigma } ( t )$

## B.5 r<sub>N</sub> Flatness

Define the $r _ { N }$ metric $r _ { N } ( \hat { x } ) = \operatorname* { m i n } r : \textstyle \sum _ { i = 1 } ^ { r } \hat { x } _ { ( i ) } \geq N / 1 0 0 ,$

where ${ \hat { x } } _ { ( 1 ) } \geq { \hat { x } } _ { ( 2 ) } \geq \cdot \cdot \cdot$ is the sorted spectrum. A standard consequence of majorization is that if $\hat { a } \succ \hat { b }$ then $r _ { N } ( \hat { a } ) \leq r _ { N } ( \hat { b } )$ , since larger partial sums reach the N/100 threshold no later.

Combining this with Lemma 2 yields:

Theorem 2 (Monotone spectral flattening). Under the assumptions ofSection B (Gaussian surrogate, $\lambda _ { 1 } > \lambda _ { d } ) ,$ , the $r _ { N }$ metric is non-decreasing in t on [0, 1]: for all $0 \leq s < t \leq 1 , r _ { N } ( \Sigma _ { v } ( s ) ) \leq$ $\begin{array} { r } { r _ { N } ( \Sigma _ { v } ( t ) , } \end{array}$ ). In particular,

$$
r _ { N } \bigl ( \Sigma _ { v } ( 0 ) \bigr ) \leq r _ { N } \bigl ( \Sigma _ { v } ( 0 . 5 ) \bigr ) < r _ { N } \bigl ( \Sigma _ { v } ( 1 ) \bigr ) = \lceil N / 1 0 0 d \rceil ,\tag{25}
$$

where the strict inequality $r _ { N } ( \Sigma _ { v } ( 0 . 5 ) ) < r _ { N } ( \Sigma _ { v } ( 1 ) )$ holds because $\hat { \sigma } ( 1 ) = ( 1 / d , \dots , 1 / d )$ is the uniform distribution, which is strictly majorized by any non-uniform σˆ(0.5) whenever $\lambda _ { 1 } > \lambda _ { d }$

This proves that the gradient covariance becomes progressively flatter as t moves from the data distribution toward the noise: small t (the “head” of the sampling trajectory) admits a sharper lowrank structure in its gradient spectrum, whereas large t (the “tail”) approaches an isotropic spectrum and is inherently high-rank. This monotonicity motivates treating small-t timesteps as the golden timesteps for low-rank gradient-based initialization.

## C Experimental Settings and Additional Results

Evaluation Metrics We evaluate our method from three complementary perspectives: controllability, alignment, and image quality. For controllability, we measure how faithfully the generated images respect the input control signal. On the Canny task, we extract Canny edges from the generated images and compute the F1 score against the input edge map, where higher values indicate better edge preservation. On the Depth task, we estimate the depth map of the generated images using a pretrained depth predictor and report the Mean Squared Error (MSE) against the input depth condition, where lower values indicate more accurate geometric alignment.

For alignment, we assess semantic consistency from both textual and visual perspectives. CLIP Text computes the cosine similarity between CLIP embeddings of the generated image and the text prompt, reflecting text-image semantic alignment. CLIP Image measures the cosine similarity between CLIP embeddings of the generated and reference images, capturing high-level visual consistency. DINO reports the cosine similarity in the DINO [2] feature space, which is more sensitive to fine-grained structural and identity-level correspondence than CLIP.

For image quality, we adopt three widely-used metrics, and calculated by pyiqa [3]. FID measures the distributional distance between generated and real images in the Inception feature space, with lower values indicating more realistic generations. SSIM evaluates structural similarity between the generated image and the reference at the pixel level, accounting for luminance, contrast, and structural information. PSNR measures pixel-wise reconstruction fidelity in decibels, where higher values denote less distortion. Together, these metrics provide a comprehensive assessment of perceptual realism, structural consistency, and reconstruction accuracy.

## C.1 Implementation Details

In this section, we briefly introduce the datasets, training details, and evaluation metrics used in experiments.

Hyperparameters setting We implement our method on top of two baselines. For the OminiControl [35] baseline (Canny-to-image, depth-to-image, and deblurring), we follow the original setup: FLUX.1 as the base DiT, LoRA rank 4, effective batch size 8 (batch size 1 with gradient accumulation of 8), prodigy optimizer with weight decay 0.01, and $5 1 2 \times 5 1 2$ resolution. Training uses the last 300,000 images of text-to-image-2M, with Canny/depth maps and Gaussian blur generated on the fly. For the DreamBooth [32] baseline we use Stable Diffusion 3-medium with the default DreamBooth setting: 30 subjects with 3–5 reference images each, prompts of the form “a [sks] [class noun]”, class-specific prior-preservation loss with $\lambda = 1 ( { \sim } 1 0 0 0$ class samples), and learning rate $5 \times 1 0 ^ { - 6 }$ Training iterations for each LoRA variant are reported in Table 1.

For our method hyperparameters, we use 1000 to 800 and 100 to 0 as our timestep selection to better fit the low-rank hypothesis. For golden channel, we set the $L _ { 1 }$ normalized bound on 1, indicating the most sparse sampling situation. For rank selection after sparse iteration, we select the top-2 rank of singular values to remain instead of being set to 0.

We deploy our models on NVIDIA L40S GPU, with 48 gigabytes of VRAM. For dreambooth, the total training time is 25 minutes per class; for the subject-alignment task, we use 2 L40S and train for almost 20 hours.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We clearly include our paper’s contributions and scope in the abstract, and detail them in the introduction.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We discuss the limitations of our work from several aspects in the appendix. Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: This paper presents the hypothesis and partial proof in the main text and the full hypothesis and proof in the appendix.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We fully disclose the conditions to reproduce our results in the paper.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The repository link mentioned in the abstract can refer to our code. We also provide the scripts to download or generate the data we used in the paper.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [TODO]

Justification: [TODO]

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For small dataset, the results can be reproduced through a different server we used. For large dataset and task, we use visualization results of different steps to show the quality of generated images.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We disclose the details of our training server in the appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The whole paper follows the NeurIPS code of ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We fully discuss both positive and negative societal impacts of our work.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: Our released models do not have the risk for misuse and some safeguard risks. Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We mention and respect the license and terms of use of the creators of code, data, and models related to this paper.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Our code and new form of data all have detailed documentation provided with the new assets.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This paper does not involve anything related to crowdsourcing nor research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: This paper does not involve anything related to crowdsourcing nor research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [No]

Justification: The core method and analysis are done by carefully researchers themselves.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.