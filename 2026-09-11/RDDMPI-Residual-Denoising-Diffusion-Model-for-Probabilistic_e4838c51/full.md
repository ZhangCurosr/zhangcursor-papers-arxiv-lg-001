# RDDMPI: Residual Denoising Diffusion Model for Probabilistic Multivariate Time Series Imputation

Ramiro Valdes Jara, David Chapman, Adam Meyers

Abstract—Multivariate time series imputation (MTSI) aims to recover missing values in temporal data composed of multiple interdependent variables. This problem is central to real-world applications such as healthcare monitoring, traffic networks, and energy systems. Recent diffusion-based approaches have shown strong potential for probabilistic imputation by learning to generate missing values through iterative denoising. However, most existing approaches perform diffusion directly in the original data space, requiring the denoising network to simultaneously capture global structure, temporal dynamics, and stochastic variability. This makes the generative task unnecessarily complex, especially when modern deterministic imputers can already provide accurate initial reconstructions. To address this limitation, we propose RDDMPI, a conditional residual diffusion framework that operates directly in residual space. Instead of modeling the full missing signal directly, we reformulate probabilistic imputation as a baseline-residual decomposition, where a pretrained model captures the dominant signal and a diffusion process models the residual uncertainty. To better exploit deterministic guidance, RDDMPI conditions the reverse denoising process on both the baseline-completed signal and its latent representation, while a reliability-aware conditioning mechanism adaptively controls the influence of baseline information during residual generation. This formulation simplifies the diffusion learning objective, enabling it to focus on structured correction terms rather than reconstructing the full signal. Experiments on multiple benchmark datasets demonstrate that RDDMPI consistently improves both reconstruction accuracy and uncertainty quantification.

Index Terms—Multivariate time series imputation, residual diffusion, diffusion model, uncertainty quantification.

## I. INTRODUCTION

M <sup>ULTIVARIATE</sup> <sup>time</sup> <sup>series</sup> <sup>data</sup> <sup>are</sup> <sup>widely</sup> <sup>collected</sup> in real-world domains such as healthcare monitoring, traffic systems, finance, energy, and climate science [1], [2], [3], [4], [5]. In these settings, each variable evolves over time while interacting with other variables. However, missing observations are common due to sensor failures, irregular sampling, communication failures, or data acquisition limitations. Since missing values can degrade downstream tasks such as forecasting, anomaly detection, decision support, and system monitoring, multivariate time series imputation (MTSI) has become a fundamental problem in time series analysis.

The goal of MTSI is to recover missing entries by exploit ing both temporal continuity and cross-variable correlations.

Early approaches relied primarily on statistical and traditional machine learning techniques, including interpolation methods, matrix and tensor factorization, and Gaussian process models. Although these methods provide interpretable solutions, their ability to capture complex nonlinear temporal dynamics and cross-variable dependencies present in modern multivariate datasets is limited. To overcome these limitations, deep learningbased approaches learn rich temporal and cross-variable representations directly from data, leading to substantial improvements in reconstruction accuracy and transforming the MTSI landscape.

Existing deep learning approaches can be broadly categorized into deterministic and probabilistic methods. Deterministic methods estimate a single value for each missing entry and have achieved strong performance by capturing temporal dynamics and cross-variable interactions. However, they do not explicitly model uncertainty, which is problematic when multiple plausible imputations are consistent with the observed data. Probabilistic methods address this limitation by modeling a conditional distribution over missing values rather than a single point estimate. Among them, diffusion models have recently emerged as a powerful framework for multivariate time series imputation. By learning an iterative reverse denoising process conditioned on observed entries, diffusion models can generate multiple plausible imputations, thereby enabling uncertainty estimation.

Despite their success, existing diffusion-based methods typically reconstruct the full missing signal directly from noise. Consequently, the denoising network must jointly learn global temporal structure, cross-variable dependencies, local dynamics, and stochastic variability, resulting in a highly complex learning problem. This observation is particularly relevant because modern deterministic imputers are already capable of recovering much of the underlying signal structure. In many cases, the remaining prediction error contains both systematic, potentially reducible, errors in the reconstruction and irreducible conditional variability arising when multiple missing-value realizations are consistent with the observed data.

This raises a central question: can applying diffusion to residual correction, rather than full-signal reconstruction, simplify the generative learning problem and improve reconstruction accuracy and uncertainty estimation? We argue that residual modeling leads to a simpler and more targeted generative problem. By reformulating probabilistic imputation in residual space, diffusion can focus on correcting systematic baseline errors and modeling the remaining conditional variability rather than reconstructing the full signal from scratch. However, leveraging deterministic predictions as conditioning information introduces an additional challenge. The quality of baseline imputations can vary across datasets, missingness patterns, and regions of the time series. While accurate baseline predictions provide valuable guidance, unreliable estimates may propagate errors through the diffusion process. Therefore, conditioning information should be incorporated adaptively according to its estimated reliability.

Motivated by these observations, we propose RDDMPI, a Residual Denoising Diffusion Model for Probabilistic Multivariate Time Series Imputation. RDDMPI decomposes the imputation task into two stages. First, a pretrained deterministic imputation model produces a baseline-completed signal and a structured latent representation. Second, a conditional diffusion model is trained in residual space, where it learns to generate correction terms only for the missing regions. The diffusion process is conditioned on both the baseline reconstruction and its latent representation, allowing the model to focus on probabilistic residual refinement rather than reconstructing the full signal from scratch. In addition, RDDMPI incorporates a reliability-aware conditioning mechanism to adaptively control the influence of baseline information and reduce the propagation of unreliable deterministic estimates.

The main contributions of this work are summarized as follows:

• We reformulate probabilistic multivariate time series imputation as a baseline-residual decomposition, separating deterministic signal reconstruction from probabilistic residual uncertainty modeling.

• We propose RDDMPI, a conditional residual diffusion framework that performs diffusion directly in residual space and leverages both baseline-completed signals and latent baseline representations to guide denoising.

• We introduce a reliability-aware conditioning mechanism that adaptively modulates the contribution of deterministic baseline information during the reverse diffusion process.

• We provide extensive empirical evidence across five benchmark datasets showing that RDDMPI achieves state-of-the-art performance in most evaluated settings, consistently outperforming deterministic and probabilistic baselines in both reconstruction accuracy and uncertainty quantification

The remainder of this paper is organized as follows. Section II reviews existing deterministic and probabilistic approaches for multivariate time series imputation. Section III formulates the imputation problem. Section IV presents RD-DMPI, including the residual diffusion formulation, the conditional denoising network, the reliability-aware conditioning mechanism, and the theoretical motivation. Section V reports experimental results and ablation studies. Finally, Section VI concludes the paper.

## II. RELATED WORK

Research on multivariate time series imputation (MTSI) has evolved considerably, progressing from deterministic reconstruction methods toward probabilistic generative approaches capable of modeling uncertainty. Existing methods can broadly be categorized into deterministic (Section II-A) and probabilistic approaches (Section II-B). In this section, we review these two research directions, with particular emphasis on recent diffusion-based methods that motivate the proposed residual diffusion framework.

## A. Deterministic Time Series Imputation

Deterministic imputation methods aim to recover missing values by exploiting temporal dependencies and cross-variable correlations present in the observed data. These methods produce a single imputed estimate for each missing entry and have demonstrated strong performance across a wide range of applications.

Traditional approaches include interpolation and smoothing methods [6], as well as low-rank matrix and tensor completion methods [7], [2]. These methods are efficient and interpretable, exploiting local smoothness or shared low-dimensional structure across variables and time. However, their smoothness and low-rank assumptions limit their ability to capture complex nonlinear dependencies.

Early deep learning approaches, such as GRU-D [8] and BRITS [9], primarily relied on recurrent neural networks (RNNs) to model temporal dynamics while explicitly handling missing observations. However, sequential processing can accumulate reconstruction errors and limit parallelization across timesteps during training and inference. More recently, attention-based architectures have become increasingly popular due to their ability to capture long-range temporal dependencies and complex interactions among variables. SAITS [10] employs diagonally masked self attention and combines two reconstruction stages, whereas NRTSI [11] treats observations as permutation-equivariant sets and progressively imputes missing values without recurrent processing. Other methods jointly model temporal and feature-level dependencies through multidimensional, global-local, or structured attention mechanisms [12], [13].

Recent methods can be divided into two broad categories. One adapts general-purpose time series architectures, and the other develops specialized architectures for time series imputation. General-purpose architectures based on temporal convolutions [14], two-dimensional temporal variation modeling [15], inverted attention [16], and multiscale mixing [17] have been adapted to imputation through masked reconstruction objectives. These approaches benefit from advances in general time series representation learning but are not designed exclusively for missing-value reconstruction.

In contrast, specialized imputation architectures introduce mechanisms tailored to partially observed data. ImputeFormer [18] employs low-rankness-induced attention to exploit the latent low-dimensional structure of spatiotemporal data and improve generalization across sensors and missingness patterns. T1 [19] introduces one-to-one channel-head binding, assigning each variable to a dedicated attention head to strengthen the modeling of variable-specific dynamics while retaining crossvariable interactions. Beyond architectural design, another line of research incorporates explicit inductive biases about signal structure or distributional similarity. Decompositionbased methods separate trend, periodic, and local components to facilitate the reconstruction of long-range and recurring patterns [20]. Optimal-transport approaches instead recover missing values by aligning distributions of time series patches. In particular, PSW-I [21] incorporates spectral information into a Wasserstein-based discrepancy, enabling nonparametric imputation under temporal and distributional shifts without training a separate predictive model.

For spatiotemporal data, relationships among variables can also be encoded by a graph. Graph-recurrent architectures propagate information across both time and connected variables, while sparse spatiotemporal attention improves scalability to larger sensor networks [22], [23]. These methods are effective when a meaningful spatial or relational graph is available. However, general multivariate time series may not provide such a graph, requiring relationships among variables to be learned directly from the observed data.

Despite their strong reconstruction performance, deterministic methods inherently produce only a single imputed trajectory. Consequently, they do not explicitly represent the conditional variability of missing values, which can be problematic when multiple plausible completions exist for a partially observed sequence. This limitation has motivated the development of probabilistic approaches that model distributions over missing values rather than single-point estimates.

## B. Probabilistic Time Series Imputation

Probabilistic imputation methods seek to estimate the conditional distribution of missing values given the observed data. Unlike deterministic approaches, these methods provide uncertainty estimates alongside imputations, allowing them to represent multiple plausible reconstructions and better characterize ambiguity in the missing regions.

Early probabilistic approaches primarily relied on latentvariable models and Bayesian formulations. Methods such as V-RIN [24] combine recurrent architectures with variational inference to learn uncertainty-aware latent representations, while Gaussian process-based approaches, including Multi-task GP [25] and GP-VAE [26], provide principled uncertainty quantification through probabilistic priors. However, these methods struggle to capture highly complex temporal dependencies.

More recently, diffusion models have emerged as a powerful probabilistic framework for time series imputation. By learning to reverse a gradual noising process, diffusion models can approximate complex conditional distributions without strong distributional assumptions. CSDI [27] established the foundation for diffusion-based time series imputation by formulating the task as a conditional score-based diffusion process, demonstrating that diffusion models can generate multiple plausible imputations while providing uncertainty estimates.

Subsequent work has improved diffusion-based imputation through more informative structural conditioning, consistency constraints, and specialized denoising architectures. Spatiotemporal diffusion models incorporate graph and geographic relationships into the conditioning process [28], while consistency-aware formulations encourage coherent estimates across variables, timesteps, or complementary views [29], [30].

FGTI [31] introduces spectral information to better reconstruct periodic and high-frequency components. SSSD [32] replaces conventional denoising backbones with structured state-space layers designed to capture long-range temporal dependencies.

Beyond diffusion, flow matching has recently emerged as an alternative generative formulation for probabilistic imputation. GiFlow [33] constructs a graph-informed source distribution from the observed signal and learns a continuous transport process toward the target distribution, enabling efficient generation while explicitly modeling spatial and temporal dependencies. However, it is specifically designed for settings in which meaningful graph structure is available.

Despite their differences, existing diffusion-based imputation methods share a common characteristic: diffusion is performed directly in the original data space. Consequently, the denoising network must learn both the dominant signal structure and the residual uncertainty simultaneously while reconstructing the entire missing signal from noise. This motivates the exploration of alternative formulations that simplify the generative task while retaining the uncertainty modeling benefits of diffusionbased imputation.

## C. Residual Modeling and Position of This Work

Residual modeling has been widely adopted in machine learning as a strategy for simplifying complex prediction tasks. For example, gradient boosting incrementally improves an initial predictor by fitting successive models to its remaining errors [34]. More recently, residual formulations have been incorporated into diffusion models. RDDM [35] separates image restoration into a directional residual diffusion process and a stochastic noise diffusion process, distinguishing restoration from sample diversity. This idea has recently gained attention in time series forecasting, where RDIT [36] combines a point forecaster with conditional diffusion over its residuals, allowing the deterministic model to capture the central forecast while diffusion models the remaining predictive distribution.

RDDMPI proposes a residual diffusion formulation for probabilistic multivariate time series imputation. Unlike RDDM, which addresses image restoration, and RDIT, which predicts future values, RDDMPI models residuals only at missing positions. A pretrained deterministic imputer first reconstructs the dominant signal structure, after which diffusion models the conditional distribution of its remaining errors. The denoising process is conditioned on the baseline-completed signal, its latent representation, and the observation mask, while reliability-aware conditioning adaptively controls the influence of potentially inaccurate baseline information. Thus, RDDMPI applies residual diffusion specifically to missingvalue reconstruction without requiring the diffusion model to generate the complete missing signal from noise.

## III. PROBLEM DEFINITION

In this section, we present the mathematical setup and problem definition. Let $X _ { 0 } ~ \in ~ \mathbb { R } ^ { V \times L }$ denote a multivariate time series with V variables and L timesteps, and let $M \in$ $\{ 0 , 1 \} ^ { V \times L }$ be the corresponding observation mask, where

$$
M _ { v , \ell } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } X _ { 0 , v , \ell } \mathrm { ~ i s ~ o b s e r v e d , } } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

Using the mask, the observed and missing components of the time series can be written as

$$
X _ { 0 } ^ { \mathrm { o b } } = M \odot X _ { 0 } ,\tag{1}
$$

$$
X _ { 0 } ^ { \mathrm { m i } } = \left( 1 - M \right) \odot X _ { 0 } ,\tag{2}
$$

where $\odot$ denotes element-wise multiplication. The objective of multivariate time series imputation is to recover the missing component $X _ { 0 } ^ { \mathrm { m i } }$ from the available observations $X _ { 0 } ^ { \mathrm { o b } }$ . Formally, an imputation model seeks to learn a mapping

$$
\hat { X } _ { 0 } = f ( X _ { 0 } ^ { \mathrm { o b } } , M ) ,\tag{3}
$$

where $\hat { X } _ { 0 }$ denotes the reconstructed complete time series. From a probabilistic perspective, the goal is to estimate the conditional distribution

$$
p ( X _ { 0 } ^ { \mathrm { m i } } \mid X _ { 0 } ^ { \mathrm { o b } } , M ) ,\tag{4}
$$

which characterizes all plausible values of the missing entries given the observed context. Deterministic methods approximate this distribution through a single point estimate, whereas probabilistic approaches seek to model the full conditional distribution and provide uncertainty-aware imputations. RD-DMPI belongs to the latter category and models this conditional distribution through a residual diffusion process.

## IV. METHODOLOGY

In this section, we present RDDMPI, a residual diffusion framework for probabilistic multivariate time series imputation. We begin in Section IV-A by introducing a baselineresidual decomposition that reformulates imputation as residual modeling around a deterministic reconstruction. Section IV-B then presents a conditional diffusion process that learns the distribution of residual corrections over missing entries. Next, Section IV-C describes the proposed conditional denoising network, including the reliability-aware conditioning mechanism used to incorporate baseline information during denoising. Finally, Section IV-D provides a theoretical motivation for performing diffusion in residual space rather than directly in the original data space.

## A. Baseline-Residual Decomposition

Given the observed component $X _ { 0 } ^ { \mathrm { o b } }$ and mask M, a pretrained deterministic imputation model first produces a baselinecompleted signal $\tilde { X } ^ { \mathrm { b a s e } }$ , obtained by filling the missing entries with the baseline model’s deterministic predictions, and its latent representation $H ^ { \mathrm { b a s e } }$ as

$$
\begin{array} { r } { \tilde { X } ^ { \mathrm { b a s e } } , H ^ { \mathrm { b a s e } } = f _ { \mathrm { b a s e } } ( X _ { 0 } ^ { \mathrm { o b } } , M ) . } \end{array}\tag{5}
$$

Instead of modeling the missing signal directly, we define the missing-region residual target as

$$
R _ { 0 } ^ { \mathrm { m i } } = \left( 1 - M \right) \odot \left( X _ { 0 } - \tilde { X } ^ { \mathrm { b a s e } } \right) .\tag{6}
$$

The residual target is defined only at missing positions, ensuring that the diffusion model learns corrections to the deterministic baseline exclusively within the unobserved region. These corrections capture both systematic baseline errors and the remaining conditional variability. The proposed residual diffusion model is conditioned on the baseline-completed signal, the baseline latent representation, and the observation mask. We collectively denote this conditioning information as

$$
c = \left( \tilde { X } ^ { \mathrm { b a s e } } , H ^ { \mathrm { b a s e } } , M \right) .\tag{7}
$$

After sampling a residual correction $\hat { R } _ { 0 } ^ { \mathrm { m i } }$ , the final imputed series is obtained by preserving observed entries and correcting the baseline only on missing positions as

$$
\begin{array} { r } { \hat { X } _ { 0 } = M \odot X _ { 0 } + ( 1 - M ) \odot \left( \tilde { X } ^ { \mathrm { b a s e } } + \hat { R } _ { 0 } ^ { \mathrm { m i } } \right) . } \end{array}\tag{8}
$$

Figure 1 illustrates the overall RDDMPI training process.

## B. Conditional Residual Diffusion

To model uncertainty over missing values, RDDMPI employs a conditional Denoising Diffusion Probabilistic Model (DDPM) [37] operating in residual space. DDPMs are latent-variable generative models that learn a data distribution $p ( X _ { 0 } )$ through a sequence of latent variables $\{ X _ { t } \} _ { t = 1 } ^ { T }$ . These latent variables correspond to progressively noisier versions of the original data, where $X _ { 0 }$ denotes a clean sample and $X _ { T }$ approaches an isotropic Gaussian distribution after repeated perturbations. The diffusion framework consists of two Markov processes: a forward diffusion process that gradually corrupts data by adding Gaussian noise and a reverse diffusion process that learns to recover clean samples from noisy observations.

In the proposed framework, we adapt this formulation to the residual imputation setting by performing diffusion on the missing-region residual $R _ { 0 } ^ { \mathrm { m i } }$ . Let $\{ R _ { t } ^ { \operatorname* { m i } } \} _ { t = 1 } ^ { T }$ denote the noisy residual states generated by the forward diffusion process. Given the conditioning information c defined in Eq. (7), the objective is to learn the conditional residual distribution $p ( R _ { 0 } ^ { \mathrm { m i } }$ c), which characterizes plausible residual corrections around the deterministic baseline reconstruction.

1) Forward Diffusion Process: The forward diffusion process progressively perturbs the clean residual target with Gaussian noise over T diffusion steps. Formally, the forward Markov chain is defined as

$$
q ( R _ { 1 : T } ^ { \mathrm { m i } } \mid R _ { 0 } ^ { \mathrm { m i } } ) = \prod _ { t = 1 } ^ { T } q ( R _ { t } ^ { \mathrm { m i } } \mid R _ { t - 1 } ^ { \mathrm { m i } } ) ,\tag{9}
$$

$$
q ( R _ { t } ^ { \mathrm { m i } } \mid R _ { t - 1 } ^ { \mathrm { m i } } ) = \mathcal { N } \left( R _ { t } ^ { \mathrm { m i } } ; \sqrt { \alpha _ { t } } R _ { t - 1 } ^ { \mathrm { m i } } , ( 1 - \alpha _ { t } ) I \right) ,\tag{10}
$$

At each diffusion step, a small amount of Gaussian noise is added while preserving part of the original residual signal. As the process progresses, the residual state becomes increasingly corrupted and gradually loses its structure. The distribution at an arbitrary diffusion step t can be written in closed form as

$$
q ( R _ { t } ^ { \mathrm { m i } } \mid R _ { 0 } ^ { \mathrm { m i } } ) = \mathcal { N } \left( R _ { t } ^ { \mathrm { m i } } ; \sqrt { \bar { \alpha } _ { t } } R _ { 0 } ^ { \mathrm { m i } } , ( 1 - \bar { \alpha } _ { t } ) I \right) ,\tag{11}
$$

$$
\alpha _ { t } : = 1 - \beta _ { t } , \qquad \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } .\tag{12}
$$

where $\beta _ { t }$ denotes the variance schedule and $\bar { \alpha } _ { t }$ represents the cumulative signal retention coefficient up to diffusion step

![](images/f5af7fa738113335970ad8d57010581d84f661d87d8ff3bf4451b4880e73da11.jpg)  
Fig. 1: Overview of the residual diffusion training process. Missingness is injected into the input time series to obtain a partially observed series, which is processed by a pretrained baseline to produce a baseline-completed signal, a latent representation and a residual target. Noise is added only to the missing-region residual, and the diffusion model is conditioned on the baseline-completed signal, the baseline latent representation, the noise timestep, and the mask to predict the masked noise. Training minimizes the discrepancy between the true masked noise and the predicted noise.

t. Using the reparameterization property of DDPMs, a noisy residual at any diffusion step can be sampled directly as

$$
R _ { t } ^ { \mathrm { m i } } = \sqrt { \bar { \alpha } _ { t } } R _ { 0 } ^ { \mathrm { m i } } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{13}
$$

As t increases, the residual state converges toward a Gaussian noise distribution. A complete derivation of the residual forward process is provided in Appendix B.

2) Reverse Diffusion Process: The objective of the reverse process is to progressively remove noise from $R _ { t } ^ { m i }$ while conditioning on c. Starting from $R _ { T } ^ { \mathrm { m i } } \sim \mathcal { N } ( 0 , I )$ , the model learns a conditional reverse Markov chain

$$
\begin{array} { r l } & { \displaystyle p _ { \theta } \big ( R _ { 0 : T } ^ { \mathrm { m i } } \mid c \big ) = p ( R _ { T } ^ { \mathrm { m i } } ) \prod _ { t = 1 } ^ { T } p _ { \theta } \left( R _ { t - 1 } ^ { \mathrm { m i } } \mid R _ { t } ^ { \mathrm { m i } } , t , c \right) , } \\ & { \displaystyle p _ { \theta } \left( R _ { t - 1 } ^ { \mathrm { m i } } \mid R _ { t } ^ { \mathrm { m i } } , t , c \right) = \mathcal { N } \left( R _ { t - 1 } ^ { \mathrm { m i } } ; \mu _ { \theta } \big ( R _ { t } ^ { \mathrm { m i } } , t , c \big ) , \sigma _ { t } ^ { 2 } I \right) . } \end{array}\tag{14}
$$

where $\mu _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c )$ denotes the learned conditional mean of $R _ { t - 1 } ^ { \mathrm { m i } }$ under the reverse transition, and $\sigma _ { t } ^ { 2 }$ is the corresponding reverse-process variance, fixed according to the diffusion noise schedule. We set $\sigma _ { t } ^ { 2 } = \tilde { \beta } _ { t }$ , where $\begin{array} { r } { \tilde { \beta } _ { t } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } } \end{array}$ . Following the standard DDPM parameterization, the reverse mean is expressed through the noise prediction network $\epsilon _ { \theta }$ , whose output $\hat { \epsilon } _ { \theta } = \epsilon _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c )$ denotes the predicted noise, as

$$
\mu _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( R _ { t } ^ { \mathrm { m i } } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \hat { \epsilon } _ { \theta } \right) .\tag{15}
$$

3) Training Objective: The reverse process introduced above depends on the noise prediction network $\epsilon _ { \theta }$ , which is trained to estimate the injected Gaussian noise ϵ used to construct $R _ { t } ^ { m i }$ at an arbitrary diffusion step t in Eq. (13) by minimizing

$$
\mathcal { L } _ { \theta } = \mathbb { E } _ { R _ { t } ^ { \mathrm { m i } } , \epsilon , t } \left[ \left. \left( \epsilon - \hat { \epsilon } _ { \theta } \left( R _ { t } ^ { \mathrm { m i } } , t , c \right) \right) \odot \left( 1 - M \right) \right. _ { 2 } ^ { 2 } \right] ,\tag{16}
$$

Importantly, the sampled noise ϵ is not provided directly as an input to the denoising network. Instead, it serves as the supervision target in the training objective, while $\epsilon _ { \theta }$ receives the noisy residual $R _ { t } ^ { \operatorname* { m i } }$ , the diffusion step $t ,$ and the conditioning information c. The mask restricts optimization to missing entries, preventing the model from being penalized on observed values already fixed by the conditioning signal. The complete self-supervised training procedure is provided in Appendix C.

4) Inference: After training, the denoising network $\epsilon _ { \theta }$ parameterizes the reverse diffusion process. The overall inference procedure of RDDMPI is summarized in Algorithm 1. The baseline model first provides a deterministic baseline reconstruction and latent representation. Then, the residual diffusion model performs N independent reverse sampling trajectories from $\textit { t } = \textit { T } \mathrm { t o } \textit { t } = \textit { 1 }$ , producing a set of residual samples $\{ R _ { 0 } ^ { \mathrm { m i } , n } \} _ { n = 1 } ^ { N }$ . These samples approximate the conditional residual distribution around the deterministic baseline. Each sample produces a plausible imputation by adding the sampled residual to the baseline at missing positions. The complete sample set is retained for probabilistic evaluation and uncertainty quantification, whereas its element-wise median is used as the point estimate for reconstruction-accuracy evaluation. The architecture used to parameterize $\epsilon _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c )$ is described next in Section IV-C.

## C. Conditional Denoising Network

The diffusion process is parameterized by a conditional denoising network $\epsilon _ { \theta } ( R _ { t } ^ { m i } , t , c )$ . As illustrated in Figure 2, the network consists of three main components: a reliabilityaware input fusion mechanism, a FiLM-based conditioning module using the baseline latent representation, and a stack of residual denoising blocks. Together, these components allow the denoiser to adaptively incorporate information from the deterministic baseline while estimating the noise contained in the current residual state.

![](images/1a4879eb723368867dddd05480b66fe44a92630dc25a1f8a2825349f9d234f26.jpg)  
Fig. 2: Architecture of the conditional denoising network $\epsilon _ { \theta } ,$ which receives the noisy residual $R _ { t } ^ { \operatorname* { m i } }$ , diffusion step t, and conditioning information $c = ( \widetilde { X } ^ { \mathrm { b a s e } } , H ^ { \mathrm { b a s e } } , M )$ , and outputs the noise prediction $\widehat { \epsilon } _ { \theta }$ . The first residual denoising block receives the FiLM-conditioned representation, $U _ { t } ^ { ( 0 ) } = \tilde { Z } _ { t } .$ , and each subsequent block receives the output of the preceding block, $U _ { t } ^ { ( b - 1 ) }$ A complete mathematical description of the residual denoising blocks is provided in Appendix E.

Algorithm 1: Imputation (Sampling) with RDDMPI.   
Input : Partially observed sample $\overline { { \boldsymbol { X } _ { 0 } ; } }$ mask M; pretrained   
deterministic baseline $f _ { \mathrm { b a s e } } ;$ trained denoising   
network $\epsilon _ { \theta } ;$ number of stochastic samples $N .$   
Output : Imputed sequence $\hat { X } _ { 0 } .$   
Compute observed context: $X _ { 0 } ^ { \mathrm { o b } }  M \odot X _ { 0 } ;$   
Obtain deterministic baseline-completed signal and latent   
representation: $\tilde { X } ^ { \mathrm { b a s e } } , H ^ { \mathrm { b a s e } }  \biggl { } f _ { \mathrm { b a s e } } ( X _ { 0 } ^ { \mathrm { o b } } , M ) ;$   
Define conditioning information: $c \gets ( \tilde { X } ^ { b a s e } , H ^ { b a s e } , M ) ;$   
for $n = 1$ to N do   
Initialize missing-region residual with Gaussian noise:   
$R _ { T } ^ { \mathrm { { m i } } , ( n ) } \sim \mathcal { N } ( \bar { 0 } , I ) ;$   
for $\mathbf { \bar { \chi } } _ { t } = T \ \mathbf { t o } \ 1$ do   
Predict noise using the conditional residual diffusion   
model: ${ \hat { \epsilon } } _ { \theta } \gets \epsilon _ { \theta } ( R _ { t } ^ { \mathrm { m i } , n } , t , c ) ;$   
Compute DDPM reverse mean:   
$\begin{array} { r } { \mu _ { \theta }  \frac { 1 } { \sqrt { \alpha _ { t } } } ( R _ { t } ^ { \mathrm { m i } , n } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \hat { \epsilon } _ { \theta } ) } \end{array}$   
Sample reverse step:   
$R _ { t - 1 } ^ { \operatorname { m i } , n } \gets \mu _ { \theta } + \sigma _ { t } ^ { \cdot } z \ , \quad z \sim \mathcal { N } ( 0 , I ) ;$   
end   
end   
Estimate final residual correction using the element-wise   
median: $\hat { R } _ { 0 } ^ { \mathrm { m i } } \gets$ median $\left( \left\{ R _ { 0 } ^ { \mathrm { m i } , n } \right\} _ { n = 1 } ^ { \overline { { N } } } \right)$ ;   
Recover the missing signal by adding the residual correction   
to the baseline: $\bar { \dot { X } _ { 0 } ^ { \mathrm { m i } } } \dot {  } ( 1 \dot { - } M ) \dot { \odot } ( \tilde { X } ^ { \mathrm { b a s e } } + \hat { R } _ { 0 } ^ { \mathrm { m i } } )$   
Return the full imputed sequence: $\hat { X } _ { 0 } \stackrel { \cdot } {  } X _ { 0 } ^ { \mathrm { o b } } + \hat { X } _ { 0 } ^ { \mathrm { m i } } ;$

1) Reliability-Aware Conditioning: The initial input to the denoising network is the noisy missing-region residual $R _ { t } ^ { \operatorname* { m i } } \in \mathbb { R } ^ { V \times \mathbf { \breve { L } } }$ , where V is the number of variables and $L$ is the sequence length. The baseline-completed signal $\widetilde { X } ^ { \vert }$ base is incorporated separately through a reliability-aware fusion mechanism.

Since the quality of deterministic baseline predictions can vary across timesteps and variables, conditioning information should not be incorporated uniformly. To address this issue, we introduce a learned reliability gate that adaptively controls the influence of the baseline reconstruction on the denoising representation. The reliability map is computed as

$$
A = \sigma \Big ( t _ { \mathrm { c o n v } } \left( \Big [ \tilde { X } ^ { \mathrm { b a s e } } , M \Big ] \right) \Big ) .\tag{17}
$$

Here, $t _ { \mathrm { c o n v } } ( \cdot )$ is a lightweight temporal convolutional module, $\sigma ( \cdot )$ denotes the sigmoid function, and $A \in [ 0 , 1 ] ^ { V \times L }$ acts as a learned reliability gate over variables and timesteps. The reliability module applies short- and long-range depthwise temporal convolutions independently to each variable, followed by pointwise projections and a sigmoid activation. The resulting map acts as a learned gate rather than an explicitly supervised estimate of baseline error.

The noisy residual and baseline-completed signal are projected independently to $d _ { h }$ hidden channels. The initial hidden representation is constructed as

$$
Z _ { t } = \phi _ { \mathrm { r e s } } ( R _ { t } ^ { \mathrm { m i } } ) + A \odot \phi _ { \mathrm { c t x } } ( \widetilde { X } ^ { \mathrm { b a s e } } ) ,\tag{18}
$$

where $\phi _ { \mathrm { r e s } }$ and $\phi _ { \mathrm { c t x } }$ are the independent input projections. The reliability map is broadcast across the hidden channels $d _ { h }$ . Thus, the noisy residual defines the primary denoising state, while the deterministic baseline is incorporated through reliabilityweighted fusion. The resulting representation $\bar { Z _ { t } } \in \mathbb { R } ^ { V \times L \times \bar { d } _ { h } }$ serves as the initial hidden representation, which is subsequently modulated by latent conditioning (Section IV-C2) before being processed by the residual denoising blocks. The same reliability map A is passed to later layers as additional conditioning information.

2) FiLM-Based Latent Conditioning: The latent representation extracted from the deterministic baseline is additionally used to condition the denoising process. Because its original shape depends on the selected baseline architecture, it is first aligned with the variable and temporal dimensions of the denoising representation. A learned projection then produces multiplicative and additive FiLM [38] tensors as

$$
\left( \gamma _ { H } , \delta _ { H } \right) = g _ { \mathrm { F i L M } } \left( g _ { \mathrm { a l i g n } } \left( H ^ { \mathrm { b a s e } } \right) \right) ,\tag{19}
$$

where $g _ { \mathrm { a l i g n } } ( \cdot )$ denotes the baseline-specific reshaping and temporal alignment operation, and $g _ { \mathrm { F i L M } } ( \cdot )$ is a learned linear projection. After dimension permutation, $\dot { \gamma } _ { H } , \delta _ { H } \in \mathbb { R } ^ { V \times L \times d _ { h } }$ FiLM conditioning is applied as

$$
\tilde { Z } _ { t } = \left( 1 + \gamma _ { H } \right) \odot Z _ { t } + \delta _ { H } ,\tag{20}
$$

where $Z _ { t }$ is the initial hidden representation. This operation injects structured information from the baseline latent representation into the denoising process by adaptively scaling and shifting the hidden features independently across channels, variables, and timesteps. The result is the FiLM-conditioned hidden representation $\tilde { Z } _ { t } ~ \in ~ \mathbb { R } ^ { V \times L \times d _ { h } }$ , which is provided as input to the residual denoising blocks. Through this conditioning, the denoiser can exploit the temporal and crossvariable dependencies encoded by the deterministic baseline while retaining the noisy residual as its primary denoising state.

3) Residual Denoising Blocks: As illustrated in Figure 2, the FiLM-conditioned hidden representation $\widetilde { Z } _ { t }$ is processed by a stack of residual denoising blocks inspired by DiffWave [39]. We denote the representation entering the b-th block by $U _ { t } ^ { ( b - 1 ) }$ , with ${ U _ { t } ^ { ( 0 ) } } ^ { * } = \widetilde { Z } _ { t }$ . Each block produces an updated representation that is passed to the next block, together with a skip representation used by the final output path.

Within each block, a projected diffusion-step embedding is first added to the incoming representation, allowing the denoiser to adapt to the current noise level. Temporal and variable transformer layers then process the representation sequentially. The temporal transformer models dependencies across timesteps separately for each variable, while the variable transformer models interactions among variables separately at each timestep. The resulting features are combined with projected side information consisting of temporal-position embeddings, variable embeddings, the observation mask, and the learned reliability map A.

The conditioned representation is passed through a gated activation and projected into a residual update and a skip representation. The residual update is combined with the block input and forwarded to the next block, while the skip representations from all blocks are aggregated by the output path. Two final pointwise projections, with a ReLU activation between them, produce $\widehat { \epsilon _ { \theta } } = \epsilon _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c ) \in \mathbb { R } ^ { V \times L }$ which has the same variable and temporal dimensions as $R _ { t } ^ { \operatorname* { m i } }$ and represents the Gaussian noise predicted for the reverse diffusion process. A complete mathematical description of the block operations, residual and skip connections, and output aggregation is provided in Appendix E.

## D. Theoretical Motivation for Residual Diffusion

A natural question is whether performing diffusion in residual space provides a principled advantage over directly modeling the missing signal. Let $X _ { 0 } ^ { \mathrm { m i } }$ denote the missing target and let $f ( c )$ be a deterministic baseline prediction computed from conditioning information c. We define the residual variable on the missing entries as

$$
R _ { 0 } ^ { \mathrm { m i } } = ( 1 - M ) \odot \left( X _ { 0 } ^ { \mathrm { m i } } - f ( c ) \right) .\tag{21}
$$

This transformation corresponds to a conditional reparameterization rather than a fundamentally different inference problem. In particular, the conditional distributions satisfy

$$
\begin{array} { r } { p _ { R } ( r \mid c ) = p _ { X } \big ( r + f ( c ) \mid c \big ) , } \end{array}\tag{22}
$$

which implies that residual diffusion is equivalent to standard conditional diffusion under a change of variables.

The practical advantage of this formulation arises when the deterministic baseline removes a substantial fraction of the conditional mean of the missing signal. Let $m ( c ) = \mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } \mid c ]$ denote the conditional mean. By adding and subtracting m(c), the residual can be decomposed as

$$
R _ { 0 } ^ { \mathrm { m i } } = \bigl ( X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \bigr ) + \bigl ( m ( c ) - f ( c ) \bigr ) .\tag{23}
$$

Taking squared norms and expectations yields

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \big \| R _ { 0 } ^ { \mathrm { m i } } \big \| ^ { 2 } \Big ] = \mathbb { E } \Big [ \big \| X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \big \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \big \| m ( c ) - f ( c ) \big \| ^ { 2 } \Big ] } \\ & { \qquad + \underbrace { 2 \mathbb { E } \big [ \big \langle X _ { 0 } ^ { \mathrm { m i } } - m ( c ) , m ( c ) - f ( c ) \big \rangle \big ] } _ { = 0 } , \quad ( 2 4 , } \end{array}
$$

where the cross term vanishes due to conditional centering, since $\mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \mid c ] = 0 . \ A$ complete derivation is provided in Appendix A-A. This decomposition shows that the residual consists of two components: the intrinsic conditional irreducible uncertainty and the squared bias of the deterministic baseline model. Consequently, when $f ( c )$ closely approximates $m ( c )$ the diffusion model only needs to learn the remaining residual variation rather than the conditional mean structure underlying the missing signal.

This interpretation is particularly relevant in score-based diffusion. Let $s _ { X } ( x _ { t } , c , t ) \ = \ \nabla _ { x _ { t } } \log p ( x _ { t } \ | \ c )$ denote the conditional score in data space. When diffusion is performed in residual space, the practical score induced by the baseline, denoted $s _ { f }$ , is given by

$$
s _ { f } ( r _ { t } , c , t ) = s _ { X } \big ( r _ { t } + \sqrt { \bar { \alpha } _ { t } } f ( c ) , c , t \big ) ,\tag{25}
$$

whereas the ideal score centered at the conditional mean, denoted $s _ { m }$ , is

$$
s _ { m } ( r _ { t } , c , t ) = s _ { X } \left( r _ { t } + \sqrt { \bar { \alpha } _ { t } } m ( c ) , c , t \right) .\tag{26}
$$

Under a standard regularity assumption that $s _ { X } ( \cdot , c , t )$ is $L _ { t ^ { - } }$ Lipschitz in its first argument, we obtain

$$
\| s _ { f } ( r _ { t } , c , t ) - s _ { m } ( r _ { t } , c , t ) \| \leq L _ { t } \sqrt { \bar { \alpha } _ { t } } \| f ( c ) - m ( c ) \| .\tag{27}
$$

Therefore, if the deterministic baseline accurately approximates the conditional mean, the induced residual score remains close to the ideal centered score. This yields a lower-energy and more correction-oriented target distribution, resulting in a more favorable learning problem for finite-capacity denoising networks in practice. Complete derivations of the score-based analysis are provided in Appendix A-B.

## V. EVALUATION

In this section, we evaluate RDDMPI on multivariate time series imputation under different missingness scenarios. We first describe the datasets, evaluation protocol, baseline methods, metrics, and implementation details. We then report quantitative results for reconstruction accuracy and uncertainty quantification, followed by ablation studies.

## A. Evaluation Setup

1) Datasets: We evaluate RDDMPI on five multivariate time series benchmark datasets: ETTh1 [40], ETTh2 [40], Weather [41], Exchange [42], and Illness [43]. These datasets cover diverse domains, including energy, climate, finance, and public health. For all datasets, we use an input window length of 96.

ETTh1 and ETTh2 contain hourly electricity transformer measurements, each with 7 correlated variables related to oil temperature and load covariates. Weather contains 21 meteorological variables collected at 10-minute intervals from the Max Planck Institute weather station. Exchange contains 8 daily international exchange rate series, representing a nonstationary financial time series setting. Illness contains 7 weekly influenza-related variables from the U.S. Centers for Disease Control and Prevention (CDC) surveillance system. During testing, non-overlapping windows are used to avoid repeated evaluation over the same temporal regions.

2) Experimental Design: All experiments are conducted using five random seeds: 2, 102, 202, 302, and 402. Training is performed on an NVIDIA RTX 6000 GPU. To evaluate generalization under different observability regimes, we consider both point-wise and structured block missingness.

In the point missing setting, we test the model under missing ratios of 0.2, 0.4, 0.6, and 0.8, corresponding to 20%, 40%, 60%, and 80% missing entries, with missing positions sampled independently and uniformly at random. In the block missing setting, we simulate realistic sensor failure patterns by combining two types of corruption: (i) a point missing component with rate 0.05, where 5% of entries are randomly removed, and (ii) a sequence missing component with rate 0.0015, meaning that approximately 0.15% of positions initiate a missing block. For every initiated block, its length is sampled uniformly from the integers between 24 and 96, and the corresponding consecutive entries of that variable are masked. Blocks extending beyond the end of a window are truncated, and overlapping blocks are combined. This mixed protocol produces localized measurement dropouts together with extended contiguous gaps, yielding a more challenging and practically relevant evaluation scenario.

3) Baselines: We compare the proposed method against ten representative baselines spanning three model categories: general-purpose time series models, specialized deterministic imputation methods, and specialized probabilistic imputation methods. This distinction is important because some baselines were originally developed for general time series analysis and are adapted here using masked reconstruction, whereas others were designed explicitly for imputation. We briefly summarize the role of each baseline below.

The general-purpose time series models include:

• DLinear [44]: A linear baseline that decomposes temporal dynamics through linear projections, providing a strong low-complexity benchmark.

• ModernTCN [14]: A modern convolutional architecture based on large-kernel temporal convolutions, designed to capture multi-scale temporal patterns efficiently.

• iTransformer [16]: A Transformer variant that models multivariate time series through variable-wise tokenization, allowing for capturing cross-variable interactions.

• TimesNet [15]: A temporal convolutional model that captures multi-period temporal variation by transforming time series into structured 2D representations.

The specialized deterministic imputation methods include:

• SAITS [10]: A self-attention-based imputation model specifically designed for multivariate time series, with masked training objectives for missing-value reconstruc tion.

• ImputeFormer [18]: A transformer-based imputation architecture tailored for generalizable spatiotemporal imputation through low-rank structured attention.

• T1 [19]: A CNN-transformer hybrid imputation model that combines temporal convolutional feature extraction with selective cross-variable information transfer through channel-head binding.

The probabilistic imputation methods include:

• GP-VAE [26]: A latent-variable model that combines a variational autoencoder with a Gaussian-process prior to capture temporal correlations.

• CSDI [27]: A conditional score-based diffusion model for probabilistic time series imputation, representing a strong diffusion-based baseline with uncertainty-aware generation.

• FGTI [31]: A frequency-aware diffusion model that extracts spectral information from the observed values to guide the denoising process.

All baseline implementations are based on established frameworks including Time-Series Library<sup>1</sup>, PyPOTS [45], and Awesome-Imputation [46] repositories to ensure reproducibility and fair comparison.

4) Evaluation Metrics: Let M denote the set of artificially masked positions used for evaluation, consisting of elements (v, ℓ), where v indexes the variable and ℓ indexes the timestep. Its cardinality is denoted by |M|. At position $( v , \ell )$ , the ground-truth and imputed values are denoted by $y _ { v , \ell }$ and $\begin{array} { r } { \hat { x } _ { v , \ell } , } \end{array}$ respectively. For reconstruction accuracy, we use mean absolute error (MAE) and mean squared error (MSE), computed over the positions in M as

$$
\mathrm { M A E } = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { ( v , \ell ) \in \mathcal { M } } \vert \hat { x } _ { v , \ell } - y _ { v , \ell } \vert ,\tag{28}
$$

TABLE I: Imputation performance on five benchmark datasets under point and block missing scenarios. Results are averaged across four point missing ratios (0.2, 0.4, 0.6, 0.8). Dataset abbreviations: ETTh1/2 = Eh1/2, Exchange = Exch, Illness = Illn, and Weather = Wthr. Best results are marked in bold and second-best results in underlined.
<table><tr><td colspan="3">Models Metric</td><td rowspan="2">RDDMPI (Ours) MAE MSE</td><td rowspan="2">DLinear MAE</td><td rowspan="2">ModernTCN MSE</td><td rowspan="2">iTransformer MAE MSE</td><td rowspan="2">MAE MSE</td><td rowspan="2">SAITS MAE</td><td rowspan="2">ImputeFormer MSE MAE</td><td rowspan="2">TimesNet MSE MAE</td><td rowspan="2">T1 MSE MAE</td><td rowspan="2">GP-VAE MSE MAE</td><td rowspan="2">CSDI MSE</td><td rowspan="2">MAE</td><td rowspan="2">FGTI MSE MAE</td></tr><tr><td>Point</td><td>MSE</td></tr><tr><td>B41 Block</td><td>0.0548 0.0168</td><td>0.1365 0.0851</td><td></td><td>0.2184 0.2906 0.1728 0.2844</td><td>0.1145 0.2111 0.0592 0.1735</td><td>0.1646 0.2569 0.1016 0.2097</td><td>0.1351 0.0257</td><td>0.2113 0.2856 0.1075 0.0594</td><td>0.3044 0.1634 0.1543 0.0920</td><td>0.2567 0.0807 0.2117 0.0258</td><td>0.1679 0.6246 0.1077 0.2842</td><td>0.5860 0.4231</td><td>0.0641 0.0179</td><td>0.1470 0.0718 0.0896 0.0170</td><td>0.1520 0.0877</td></tr><tr><td>E42 Point Block</td><td>0.0442 0.0203</td><td>0.1123 0.0744</td><td>|0.0801 0.0770</td><td>0.1851 0.1890</td><td>0.0576 0.1524 0.0477 0.1414</td><td>0.0712 0.1745 0.0546 0.1552</td><td>0.4370 0.4100 0.1471 0.2711</td><td>|0.6667 0.4457 0.2654 0.2672</td><td>0.0749 0.1799 0.0533 0.1592</td><td>0.0442 0.1264 0.0292 0.1014</td><td>1.4139 0.6001</td><td>0.8712 0.5663</td><td>|0.0735 0.1444 0.0561 0.1070</td><td>0.0982 0.0948</td><td>0.1696 0.1321</td></tr><tr><td>Exch | Point Block</td><td>0.0022 0.0043</td><td>0.0203 0.0181</td><td>|0.0053 0.0063</td><td>0.0453 0.0557</td><td>|0.0095 0.0664 0.0054 0.0498</td><td>0.0033 0.0339 0.0034 0.0336</td><td>|0.2311 0.3815 0.1864 0.3330</td><td>|0.0693 0.1098 0.1217 0.1233</td><td>|0.0034 0.0335 0.0036 0.0362</td><td>0.0019 0.0218 0.0115 0.0312</td><td>0.6415 0.5100</td><td>0.6944 0.6290</td><td>|0.0485 0.2237 0.1458</td><td>0.1135 0.0056 0.0177</td><td>0.0409 0.0457</td></tr><tr><td>uIII Point</td><td>0.0236</td><td>0.0714</td><td>|0.2175 0.2603</td><td>0.3046</td><td>|0.0835 0.1736 0.2647</td><td>|0.1211 0.2201 0.3345 0.3598</td><td>|0.2873 0.3044 0.1475 0.2373</td><td>|0.3394 0.3423 0.2697 0.3100</td><td>|0.0969 0.2054 0.1368 0.2535</td><td>0.0252 0.0849</td><td>|0.6355</td><td>0.5177</td><td>0.0806 0.1479</td><td>0.0860</td><td>0.1464</td></tr><tr><td>Block Wthr | Point</td><td>0.0390 0.0310 Block 0.0233</td><td>0.1282 0.0315 0.0254</td><td>0.0472 0.0495</td><td>0.3691 0.0882 0.1053</td><td>0.1521 0.0425 0.0799 0.0371 0.0827</td><td>0.0920 0.1452 0.1003 0.1480</td><td>0.0453 0.0647 0.0251 0.0328</td><td>|0.0475 0.0594 0.0401 0.0461</td><td>0.0471 0.0903 0.0390 0.0855</td><td>0.0874 0.0363 0.0248 0.0400</td><td>0.1792 0.3270 0.0582 0.1768 0.0627</td><td>0.4122 0.2578 0.1349</td><td>0.0977 0.0321 0.0234 0.0254</td><td>0.1895 0.0516 0.0316 0.0335</td><td>0.1283 0.0320</td></tr></table>

$$
\mathrm { M S E } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( v , \ell ) \in \mathcal { M } } \left( \hat { x } _ { v , \ell } - y _ { v , \ell } \right) ^ { 2 } .\tag{29}
$$

MAE measures the average reconstruction error, while MSE penalizes large deviations more strongly. For probabilistic imputation, we use the continuous ranked probability score (CRPS), which jointly evaluates the accuracy and distributional quality of the predictive distribution. Given a predictive cumulative distribution function $F _ { v , \ell } ,$ CRPS is defined as

$$
\mathrm { C R P S } = \frac { 1 } { | \mathcal { M } | } \sum _ { ( \boldsymbol { v } , \boldsymbol { \ell } ) \in \mathcal { M } } \int _ { - \infty } ^ { \infty } \left( F _ { \boldsymbol { v } , \boldsymbol { \ell } } ( \boldsymbol { z } ) - \mathbb { I } ( \boldsymbol { z } \geq \boldsymbol { y } _ { \boldsymbol { v } , \boldsymbol { \ell } } ) \right) ^ { 2 } d \boldsymbol { z } .\tag{30}
$$

In this paper, the predictive distribution is approximated using $N ~ = ~ 1 0 0$ generated samples $\{ \tilde { x } _ { v , \ell } ^ { ( n ) } \} _ { n = 1 } ^ { N }$ at each masked position, with empirical cumulative distribution

$$
\hat { F } _ { v , \ell } ( z ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbb { I } \left( \tilde { x } _ { v , \ell } ^ { ( n ) } \leq z \right) .\tag{31}
$$

Following the common evaluation protocol for diffusion-based time series imputation, we compute a normalized quantilebased approximation. Let $\mathcal { Q } = \{ 0 . 0 5 , 0 . 1 0 , \hdots , 0 . 9 5 \}$ , and let $\hat { x } _ { v , \ell } ^ { ( q ) }$ be the empirical q-quantile of the generated samples. Then,

$$
\mathrm { C R P S } = \frac { 1 } { \left| \mathcal { Q } \right| } \sum _ { q \in \mathcal { Q } } \frac { 2 \sum _ { \left( v , \ell \right) \in \mathcal { M } } \rho _ { q } \left( y _ { v , \ell } - \hat { x } _ { v , \ell } ^ { \left( q \right) } \right) } { \sum _ { \left( v , \ell \right) \in \mathcal { M } } \left| y _ { v , \ell } \right| } ,\tag{32}
$$

where the quantile loss is $\rho _ { q } ( u ) = u ( q - \mathbb { I } ( u < 0 ) )$

5) Implementation Details: All methods are evaluated using identical train, validation, and test splits under the experimental protocol described above. For general-purpose time series models DLinear [44], ModernTCN [14], iTransformer [16], and TimesNet [15], we employ a unified training setup using 0.4 point-wise masking during training. Optimization is performed using Adam with learning rate $\bar { 1 } 0 ^ { - 3 }$ , batch size 16, and a maximum of 300 epochs with early stopping.

For the specialized imputation models SAITS [10], ImputeFormer [18], T1 [19], GP-VAE [26], CSDI [27], and FGTI [31] we preserve the original training procedures and hyperparameters reported in their official implementations whenever available. An analysis of parameter count, training cost, and inference efficiency is provided in Appendix D.

RDDMPI adopts T1 [19] as the pretrained deterministic backbone, providing both the baseline-completed signal and the latent representation used to condition the residual diffusion model. The deterministic backbone is pretrained and subsequently frozen during diffusion training.

For the diffusion component, we employ a DDPM formulation with T = 50 diffusion steps. The variance schedule increases quadratically between the initial and final noise levels, allocating smaller noise increments to the earlier diffusion steps. Its architectural hyperparameters are selected separately for each dataset. These include the learning rate, number of residual denoising blocks, and number of hidden channels, among other hyperparameters. The complete dataset-specific configurations are reported in Appendix F.

During training, a self-supervised masking strategy is adopted, where a masking ratio is randomly sampled and applied to the observed entries to construct reconstruction targets. During inference, N = 100 residual samples are generated through the reverse diffusion process. The median prediction is used for deterministic evaluation, while the complete sample set is used to compute probabilistic metrics such as CRPS and to construct predictive intervals.

## B. Experimental Results

In this section, we evaluate the proposed method from two complementary perspectives: deterministic reconstruction accuracy and probabilistic uncertainty quantification. Since the goal of probabilistic imputation is not only to recover missing values accurately but also to provide well-calibrated predictive distributions, we report both types of metrics separately for clarity. The tables in the main paper report results averaged across the four point missing ratios for conciseness. Full results for each individual missing ratio, together with the corresponding standard deviations and block missing results, are provided in Appendix G.

1) Reconstruction Accuracy: We first assess deterministic imputation quality using MAE and MSE. The corresponding quantitative results across all benchmark datasets are reported in Table I. RDDMPI achieves the best result in 18 of the 20 comparisons, indicating that residual diffusion provides a robust improvement over both deterministic and probabilistic baselines in reconstruction accuracy. Relative to its deterministic backbone, T1, our proposed model improves in 19 of the 20 comparisons. The improvement is particularly consistent under block missingness, where RDDMPI outperforms T1 in both MSE and MAE on all five datasets. These results suggest that the benefits of residual diffusion are especially consistent when the deterministic baseline is less accurate.

TABLE II: Probabilistic imputation performance in terms of CRPS on five benchmark datasets under point and block missing scenarios. Results for the point missing setting are averaged across four missing ratios (0.2, 0.4, 0.6, 0.8). Dataset abbreviations: ETTh1/2 = Eh1/2, Exchange = Exch, Illness = Illn, and Weather = Wthr. Best results are marked in bold, and second-best results are marked in underlined.
<table><tr><td rowspan=1 colspan=2>ModelsMetric</td><td rowspan=1 colspan=1>RDDMPICRPS</td><td rowspan=1 colspan=1>GP-VAECRPS</td><td rowspan=1 colspan=1>CSDICRPS</td><td rowspan=1 colspan=1>FGTICRPS</td></tr><tr><td rowspan=1 colspan=2> PointBlock</td><td rowspan=1 colspan=1>0.13130.0811</td><td rowspan=1 colspan=1>0.73760.5298</td><td rowspan=1 colspan=1>0.14060.0867</td><td rowspan=1 colspan=1>0.14540.0832</td></tr><tr><td rowspan=1 colspan=1>E2</td><td rowspan=1 colspan=1>PointBlock</td><td rowspan=1 colspan=1>0.06410.0412</td><td rowspan=1 colspan=1>0.63950.4076</td><td rowspan=1 colspan=1>0.08260.0596</td><td rowspan=1 colspan=1>0.09790.0751</td></tr><tr><td rowspan=1 colspan=1>Exch</td><td rowspan=1 colspan=1>PointBlock</td><td rowspan=1 colspan=1>0.01790.0153</td><td rowspan=1 colspan=1>0.78480.6903</td><td rowspan=1 colspan=1>0.09900.1238</td><td rowspan=1 colspan=1>0.03540.0391</td></tr><tr><td rowspan=1 colspan=1>uIII</td><td rowspan=1 colspan=1>PointBlock</td><td rowspan=1 colspan=1>0.07280.1238</td><td rowspan=1 colspan=1>0.66680.4793</td><td rowspan=1 colspan=1>0.15580.1873</td><td rowspan=1 colspan=1>0.14450.1173</td></tr><tr><td rowspan=1 colspan=1>Wthr</td><td rowspan=1 colspan=1>PointBlock</td><td rowspan=1 colspan=1>0.04200.0338</td><td rowspan=1 colspan=1>0.45010.2383</td><td rowspan=1 colspan=1>0.04210.0336</td><td rowspan=1 colspan=1>0.04650.0348</td></tr></table>

2) Uncertainty Quantification: We next evaluate probabilistic imputation quality using the continuous ranked probability score (CRPS), which measures the compatibility between the predicted distribution and the observed target values. Since deterministic methods do not model predictive distributions, CRPS comparisons are reported against GP-VAE, CSDI, and FGTI baselines in Table II.

RDDMPI obtains the lowest CRPS in eight of the ten dataset and missingness settings. In particular, it obtains the best result on all five datasets under point missingness and on ETTh1, ETTh2, and Exchange under block missingness. FGTI performs best on Illness under block missingness, while CSDI achieves a marginally lower CRPS on Weather under block missingness.

Taken together, these results show that the advantages of RDDMPI are not limited to the median imputation used to compute MSE and MAE. These results are consistent with the hypothesis that modeling corrections around a deterministic reconstruction provides a more focused generative objective than reconstructing the complete missing signal directly from noise.

![](images/d4605d1bd7165e9b08e35906bb36bae5f36a89bb444449f8910fb3e1903d45c6.jpg)

Figure 3 provides a qualitative comparison with CSDI. Both methods recover the local signal morphology, but the median imputations produced by RDDMPI more closely follow the ground-truth targets, particularly along the descending segment in ETTh2 and the short-term fluctuations in Exchange. RDDMPI also produces narrower predictive intervals that remain centered around the reconstructed trajectory and contain most of the target values. In these examples, its generated imputations are therefore less dispersed and maintain close agreement with the ground truth. Additional qualitative results under different missing ratios are provided in Appendix H.

## C. Ablation Studies

As a framework, RDDMPI is designed to extend a strong deterministic imputation baseline with probabilistic residual refinement. We first examine whether its benefits depend on the selected deterministic backbone. We then evaluate the contributions of baseline conditioning and reliability-aware fusion.

1) Effect of the Deterministic Backbone: To determine whether the proposed framework depends on the use of T1, we instantiate RDDMPI with either T1 or ImputeFormer as the pretrained deterministic backbone. Table IV compares each complete model with its corresponding deterministic baseline. RDDMPI reduces both MSE and MAE for both backbones across every evaluated point and block missingness setting. For T1, the gains generally increase with the point missing ratio, indicating that residual correction becomes increasingly useful as the baseline reconstruction problem becomes more difficult. The improvements obtained with ImputeFormer are larger because its baseline errors leave more substantial correction terms for the residual diffusion model to recover. These results demonstrate that the proposed formulation can generalize to different deterministic imputation architectures, although the magnitude of the improvement depends on the quality and error structure of the selected deterministic backbone.

2) Component Ablation: To isolate the contributions of the proposed components, we compare three residual diffusion variants on the Exchange dataset that progressively incorporate the proposed design choices.

RDDMPI without baseline conditioning. In this variant, the model performs diffusion in residual space but does not receive any conditioning derived from the deterministic backbone. In particular, neither the baseline-completed signal nor the associated latent features are provided to the denoiser. As a result, conditioning mechanisms such as FiLM modulation and reliability-aware fusion are entirely removed. The model therefore relies solely on the noisy residual and diffusion step. This setting isolates the effect of residual diffusion alone.

![](images/eda5a5cd9a5ddec5499d1f3601599dc79094c454ee466d624c9815d7be8feedc.jpg)  
Fig. 3: Select examples of probabilistic time series imputation on the ETTh2 and Exchange datasets (zoomed-in views). The red crosses denote observed values, and the blue circles denote the ground-truth imputation targets. For each method, the median imputation is shown as a solid line, while the shaded region represents the 5% and 95% predictive quantiles.

TABLE III: Component ablation of RDDMPI on Exchange dataset under point missing and block missing settings. Lower is better.
<table><tr><td rowspan="2">Method</td><td colspan="10">0.2 0.4</td><td colspan="3">Block Missing</td></tr><tr><td>MSE</td><td>MAE</td><td>CRPS</td><td>MSE</td><td>MAE</td><td>CRPS</td><td>0.6 MSE MAE</td><td>CRPS</td><td>MSE</td><td>0.8 MAE</td><td>CRPS</td><td>MSE MAE</td><td>CRPS</td></tr><tr><td>RDDMPI w/o baseline conditioning</td><td>0.00184</td><td>0.01436</td><td>0.01322</td><td>0.00129</td><td>0.01536 0.01511</td><td>0.00193</td><td>0.02045</td><td>0.01824</td><td>0.00309</td><td>0.02995</td><td>0.02622</td><td>0.00948</td><td>0.02665</td><td>0.02151</td></tr><tr><td>RDDMPI w/o reliability- aware fusion</td><td>0.00184</td><td>0.01408</td><td>0.01283</td><td>0.00129</td><td>0.01646</td><td>0.01462</td><td>0.00183</td><td>0.02006</td><td>0.01765</td><td>0.00309</td><td>0.02920</td><td>0.02585 0.01007</td><td>0.03077</td><td>0.02495</td></tr><tr><td>RDDMPI (full)</td><td>0.00210</td><td>0.01465</td><td>0.01312</td><td>0.00148</td><td>0.01705</td><td>0.01511</td><td>0.00193</td><td>0.02055 0.01814</td><td>0.00300</td><td>0.02883</td><td>0.02521</td><td>0.00426</td><td>0.01805</td><td>0.01530</td></tr></table>

TABLE IV: Comparison between deterministic baselines and RDDMPI on the ETTh1 dataset under point missing and block missing settings. For each deterministic model, RDDMPI uses that model as its pretrained baseline. IF denotes ImputeFormer. $\Delta$ denotes the percentage improvement of RDDMPI over its corresponding baseline, where higher $\Delta$ indicates greater improvement. Lower is better for MSE and MAE. Best results are marked in bold.
<table><tr><td rowspan="3"></td><td rowspan="3">Method</td><td colspan="6">Point Missing 0.6</td><td colspan="2">Block Missing</td></tr><tr><td colspan="2">0.2</td><td colspan="2">0.4</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">T</td><td>Base</td><td>0.02960.1112</td><td>0.0397 0.1276|</td><td></td><td>|0.0657 0.1632</td><td></td><td>|0.18780.2694</td><td>|0.0258</td><td>0.1077</td></tr><tr><td>RDDMPI</td><td>0.0228 0.0955</td><td>0.0336</td><td>0.1122</td><td>0.0480</td><td>0.1362</td><td>0.1144 0.2020</td><td>0.0168</td><td>0.0850</td></tr><tr><td>∆ (%)</td><td>22.97</td><td>14.12 | 15.37</td><td>12.07</td><td>26.94</td><td>16.54 | 39.08</td><td>25.02</td><td>34.88</td><td>21.08</td></tr><tr><td rowspan="3">F</td><td>Base</td><td>|0.07970.1697</td><td>0.15460.2232</td><td></td><td></td><td>|0.2931 0.3137|</td><td>0.6148 0.5108|</td><td>|0.0594</td><td>0.1542</td></tr><tr><td>RDDMPI</td><td>0.0294 0.1091</td><td>0.0402</td><td>0.1263</td><td>0.0625</td><td>0.1566</td><td>0.1441 0.2292</td><td>0.0208</td><td>0.0973</td></tr><tr><td>∆ (%)</td><td>63.11 35.71</td><td>74.00</td><td>43.41</td><td>78.68</td><td>50.08</td><td>76.56</td><td>55.13 64.98</td><td>36.90</td></tr></table>

RDDMPI without reliability-aware fusion. Here, baseline conditioning is available, including both the completed signal and latent features from the deterministic backbone. However, the proposed reliability-aware fusion mechanism is removed, meaning that all conditioning information is treated uniformly. This variant corresponds to a conditional residual diffusion formulation and allows us to evaluate whether naive conditioning is sufficient.

Full RDDMPI. The full model combines residual diffusion with baseline conditioning and the proposed reliability-aware fusion mechanism. In this setting, the learned reliability gate adaptively modulates the contribution of baseline-derived information, rather than incorporating the deterministic context uniformly during denoising.

Results analysis. Table III shows that the contributions of baseline conditioning and reliability-aware fusion depend on the missingness regime. At point missing ratios 0.2 and 0.4, the variants without baseline conditioning or without reliabilityaware fusion obtain the best results. When observations remain sufficiently distributed throughout the sequence, residual diffusion can recover the correction term with limited reliance on baseline-derived guidance.

At point-0.6, baseline conditioning without reliability-aware fusion achieves the best MSE, MAE, and CRPS. This suggests that the deterministic reconstruction provides useful temporal and cross-variable guidance at moderate missingness, while its errors are not yet sufficiently severe to require adaptive gating. However, this behavior changes in the more difficult regimes.

The complete RDDMPI obtains the best MSE, MAE, and CRPS at point-0.8 and under block missingness. At point-0.8, the limited number of observed values makes the deterministic reconstruction less dependable. Under block missingness, contiguous gaps additionally remove local temporal context. Incorporating baseline information uniformly can therefore propagate inaccurate estimates into the denoiser. The reliability-aware mechanism mitigates this effect by adapting the contribution of baseline-derived information across variables and timesteps.

Overall, the ablation results do not indicate that additional conditioning is uniformly beneficial. Instead, they show that residual diffusion alone can be sufficient in simpler regimes, baseline conditioning becomes useful as reconstruction difficulty increases, and reliability-aware fusion provides its clearest benefit when baseline errors have a greater effect on the final imputation.

## VI. CONCLUSIONS

In this paper, we presented RDDMPI, a conditional residual diffusion framework for probabilistic multivariate time series imputation. By decomposing the imputation problem into a deterministic initial reconstruction and a residual refinement stage, the proposed approach allows the diffusion model to focus on modeling structured correction terms rather than the full missing signal directly. In addition, the framework leverages both latent representations extracted from the pretrained baseline and reliability-aware conditioning mechanisms to guide denoising under diverse missingness patterns. Across five benchmark datasets, RDDMPI achieves the best reconstruction result in 18 of the 20 aggregated MSE and MAE comparisons and the lowest CRPS in eight of the ten probabilistic comparisons.

The main limitation of RDDMPI is its computational cost. The framework requires a separately pretrained deterministic backbone, and generating N stochastic imputation samples through T reverse steps entails NT denoiser evaluations. Future work will therefore investigate accelerated or reduced-step sampling, and joint training of the deterministic and diffusion components. Extending the framework to other downstream tasks also represents an important direction for future research.

[1] X. Yi, Y. Zheng, J. Zhang, and T. Li, “St-mvl: filling missing values in geo sensory time series data,” in Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, 2016, p. 2704–2710.

[2] H. Tan, G. Feng, J. Feng, W. Wang, Y.-J. Zhang, and F. Li, “A tensor-based method for missing traffic data completion,” Transportation Research Part C: Emerging Technologies, vol. 28, pp. 15–27, 2013.

[3] F. V. Nelwamondo, S. Mohamed, and T. Marwala, “Missing data: A comparison of neural network and expectation maximization techniques,” Current Science, pp. 1514–1521, 2007.

[4] A. T. Hudak, N. L. Crookston, J. S. Evans, D. E. Hall, and M. J. Falkowski, “Nearest neighbor imputation of species-level, plot-scale forest structure attributes from lidar data,” Remote Sensing of Environment, vol. 112, no. 5, pp. 2232–2245, 2008.

[5] S. Van Buuren and K. Groothuis-Oudshoorn, “mice: Multivariate imputation by chained equations in r,” Journal of Statistical Software, vol. 45, no. 3, p. 1–67, 2011.

[6] X. Yi, Y. Zheng, J. Zhang, and T. Li, “St-mvl: filling missing values in geo-sensory time series data,” in Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, ser. IJCAI’16, 2016, p. 2704–2710.

[7] H.-F. Yu, N. Rao, and I. S. Dhillon, “Temporal regularized matrix factorization for high-dimensional time series prediction,” in Proceedings of the 30th International Conference on Neural Information Processing Systems. Red Hook, NY, USA: Curran Associates Inc., 2016, p. 847–855.

[8] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, “Recurrent neural networks for multivariate time series with missing values,” 2016.

[9] W. Cao, D. Wang, J. Li, H. Zhou, Y. Li, and L. Li, “Brits: bidirectional recurrent imputation for time series,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, 2018, p. 6776–6786.

[10] W. Du, D. Cotˆ e, and Y. Liu, “Saits: Self-attention-based imputation for´ time series,” Expert Systems with Applications, vol. 219, p. 119619, 2023.

[11] S. Shan, Y. Li, and J. B. Oliva, “Nrtsi: Non-recurrent time series imputation,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[12] P. Bansal, P. Deshpande, and S. Sarawagi, “Missing value imputation on multidimensional time series,” Proc. VLDB Endow., vol. 14, no. 11, p. 2533–2545, 2021.

[13] Q. Suo, W. Zhong, G. Xun, J. Sun, C. Chen, and A. Zhang, “Glima: Global and local time series imputation with multi-directional attention learning,” in 2020 IEEE International Conference on Big Data (Big Data), 2020, pp. 798–807.

[14] L. Donghao and W. Xue, “ModernTCN: A modern pure convolution structure for general time series analysis,” in The Twelfth International Conference on Learning Representations, 2024.

[15] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” in International Conference on Learning Representations, 2023.

[16] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” in The Twelfth International Conference on Learning Representations, 2024.

[17] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, and J. ZHOU, “Timemixer: Decomposable multiscale mixing for time series forecasting,” in International Conference on Learning Representations (ICLR), 2024.

[18] T. Nie, G. Qin, W. Ma, Y. Mei, and J. Sun, “Imputeformer: Low ranknessinduced transformers for generalizable spatiotemporal imputation,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 2260–2271.

[19] D. Park, H. Ryu, S. Bae, K. Park, and H.-S. Kim, “T1: One-to-one channel-head binding for multivariate time-series imputation,” in The Fourteenth International Conference on Learning Representations, 2026.

[20] S. LIU, X. Li, G. Cong, Y. Chen, and Y. JIANG, “Multivariate time-series imputation with disentangled temporal representations,” in The Eleventh International Conference on Learning Representations, 2023.

[21] H. Wang, zhengnan li, H. Li, X. Chen, M. Gong, BinChen, and Z. Chen, “Optimal transport for time series imputation,” in The Thirteenth International Conference on Learning Representations, 2025.

[22] A. Cini, I. Marisca, and C. Alippi, “Filling the g ap s: Multivariate time series imputation by graph neural networks,” in International Conference on Learning Representations, 2022.

[23] I. Marisca, A. Cini, and C. Alippi, “Learning to reconstruct missing data from spatiotemporal graphs with sparse observations,” arXiv preprint arXiv:2205.13479, 2022.

[24] A. W. Mulyadi, E. Jun, and H.-I. Suk, “Uncertainty-aware variationalrecurrent imputation network for clinical time series,” IEEE Transactions on Cybernetics, vol. 52, no. 9, pp. 9684–9694, 2022.

[25] E. V. Bonilla, K. Chai, and C. Williams, “Multi-task gaussian process prediction,” in Advances in Neural Information Processing Systems, vol. 20. Curran Associates, Inc., 2007.

[26] V. Fortuin, D. Baranchuk, G. Raetsch, and S. Mandt, “Gp-vae: Deep probabilistic time series imputation,” in Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 108. PMLR, 2020, pp. 1651–1661.

[27] Y. Tashiro, J. Song, Y. Song, and S. Ermon, “Csdi: conditional score-based diffusion models for probabilistic time series imputation,” in Proceedings of the 35th International Conference on Neural Information Processing Systems, 2021.

[28] M. Liu, H. Huang, H. Feng, L. Sun, B. Du, and Y. Fu, “Pristi: A conditional diffusion framework for spatiotemporal imputation,” in 2023 IEEE 39th International Conference on Data Engineering (ICDE), 2023, pp. 1927–1939.

[29] X. Wang, H. Zhang, P. Wang, Y. Zhang, B. Wang, Z. Zhou, and Y. Wang, “An observed value consistent diffusion model for imputing missing values in multivariate time series,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 2409–2418.

[30] J. Zhou, J. Li, G. Zheng, X. Wang, and C. Zhou, “Mtsci: A conditional diffusion model for multivariate time series consistent imputation,” in Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, ser. CIKM ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 3474–3483.

[31] X. Yang, Y. Sun, X. Yuan, and X. Chen, “Frequency-aware generative models for multivariate time series imputation,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[32] J. L. Alcaraz and N. Strodthoff, “Diffusion-based time series imputation and forecasting with structured state space models,” Transactions on Machine Learning Research, 2023.

[33] Z. Zhang, A. Einizade, J. H. Giraldo, and O. Fink, “Spatiotemporal imputation with graph-informed flow matching,” 2026.

[34] J. H. Friedman, “Greedy function approximation: A gradient boosting machine.” The Annals of Statistics, vol. 29, no. 5, pp. 1189 – 1232, 2001.

[35] J. Liu, Q. Wang, H. Fan, Y. Wang, Y. Tang, and L. Qu, “Residual denoising diffusion models,” 2024.

[36] C.-Y. Lai, Y.-C. Ning, and D. S. Boning, “Rdit: Residual-based diffusion implicit models for probabilistic time series forecasting,” 2025.

[37] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 6840–6851.

[38] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “Film: visual reasoning with a general conditioning layer,” in Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, 2018.

[39] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” in International Conference on Learning Representations, 2021.

[40] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 11 106–11 115, 2021.

[41] Wetterstation, “Weather.” [Online]. Available: https://www.bgc-jena.mpg. de/wetter/

[42] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long- and short-term temporal patterns with deep neural networks,” in The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. Association for Computing Machinery, 2018, p. 95–104.

[43] CDC, “Illness.” [Online]. Available: https://gis.cdc.gov/grasp/fluview/ fluportaldashboard.html

[44] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” 2023.

[45] W. Du, “PyPOTS: A Python Toolkit for Data Mining on Partially-Observed Time Series,” KDD 2023 MiLeTS, 2023.

[46] W. Du, J. Wang, L. Qian, Y. Yang, F. Liu, Z. Wang, Z. Ibrahim, H. Liu, Z. Zhao, Y. Zhou, W. Wang, K. Ding, Y. Liang, B. A. Prakash, and Q. Wen, “Tsi-bench: Benchmarking time series imputation,” arXiv preprint arXiv:2406.12747, 2024.

[47] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” 2022.

# APPENDIX A THEORETICAL ANALYSIS OF RESIDUAL DIFFUSION

## A. Residual Energy Decomposition

Let $m ( c ) = \mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } \mid c ]$ denote the conditional mean. We introduce and subtract $m ( c )$ to obtain

$$
R _ { 0 } ^ { \mathrm { m i } } = \bigl ( X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \bigr ) + \bigl ( m ( c ) - f ( c ) \bigr ) .
$$

We now analyze the second moment of the residual. Expanding the squared norm yields

$$
\| R _ { 0 } ^ { \mathrm { m i } } \| ^ { 2 } = \| X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \| ^ { 2 } + \| m ( c ) - f ( c ) \| ^ { 2 } + 2 \langle X _ { 0 } ^ { \mathrm { m i } } - m ( c ) , m ( c ) - f ( c ) \rangle .
$$

Taking the conditional expectation with respect to c, we obtain

$$
\begin{array} { r } { \mathbb { E } \big [ \| R _ { 0 } ^ { \mathrm { m i } } \| ^ { 2 } \mid c \big ] = \mathbb { E } \big [ \| X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \| ^ { 2 } \mid c \big ] + \| m ( c ) - f ( c ) \| ^ { 2 } + 2 \mathbb { E } \big [ \langle X _ { 0 } ^ { \mathrm { m i } } - m ( c ) , m ( c ) - f ( c ) \rangle \mid c \big ] . } \end{array}
$$

Vanishing of the cross term. We now show that the cross term vanishes. Since $m ( c ) = \mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } \mid c ]$ , we have

$$
\mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \mid c ] = 0 .
$$

Moreover, $m ( c ) - f ( c )$ is deterministic given $c .$ Therefore, using linearity of expectation,

$$
\begin{array} { r l } & { \mathbb { E } \big [ \langle X _ { 0 } ^ { \mathrm { m i } } - m ( c ) , m ( c ) - f ( c ) \rangle \mid c \big ] = \big \langle \mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \mid c ] , m ( c ) - f ( c ) \big \rangle } \\ & { \qquad = \langle 0 , m ( c ) - f ( c ) \rangle } \\ & { \qquad = 0 . } \end{array}
$$

Thus,

$$
\mathbb { E } \big [ \| R _ { 0 } ^ { \mathrm { m i } } \| ^ { 2 } ~ | ~ c \big ] = \mathbb { E } \big [ \| X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \| ^ { 2 } ~ | ~ c \big ] + \| m ( c ) - f ( c ) \| ^ { 2 } .
$$

Taking the expectation over c and applying the law of total expectation, we obtain

$$
\begin{array} { r } { \mathbb { E } \big [ \| R _ { 0 } ^ { \mathrm { m i } } \| ^ { 2 } \big ] = \mathbb { E } \big [ \| X _ { 0 } ^ { \mathrm { m i } } - m ( c ) \| ^ { 2 } \big ] + \mathbb { E } \big [ \| m ( c ) - f ( c ) \| ^ { 2 } \big ] . } \end{array}
$$

## B. Score-Based Diffusion Perspective

Let $x _ { t }$ denote the noisy variable at diffusion step $t ,$ and let $p _ { t } ( x _ { t } \mid c )$ denote its conditional marginal distribution induced by the forward diffusion process. The corresponding conditional score function is defined as

$$
\begin{array} { r } { s _ { X } ( x _ { t } , c , t ) = \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } \mid c ) . } \end{array}
$$

When performing diffusion in residual space, the noisy residual is given by

$$
r _ { t } = \sqrt { \bar { \alpha } _ { t } } R _ { 0 } ^ { \mathrm { m i } } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .
$$

This induces a corresponding noisy variable in data space through the transformation

$$
x _ { t } = r _ { t } + \sqrt { \bar { \alpha } _ { t } } f ( c ) .
$$

The score function associated with this shifted representation is therefore

$$
s _ { f } ( r _ { t } , c , t ) = s _ { X } \big ( r _ { t } + \sqrt { \bar { \alpha } _ { t } } f ( c ) , c , t \big ) .
$$

In contrast, the ideal centered residual representation would shift by the conditional mean $m ( c ) = \mathbb { E } [ X _ { 0 } ^ { \mathrm { m i } } \mid c ]$ , leading to the score

$$
s _ { m } ( r _ { t } , c , t ) = s _ { X } \big ( r _ { t } + \sqrt { \bar { \alpha } _ { t } } m ( c ) , c , t \big ) .
$$

Assume that for each diffusion step t, the score function $s _ { X } ( \cdot , c , t )$ is $L _ { t ^ { - } } \mathrm { I }$ ipschitz in its first argument, i.e.,

$$
\| s _ { X } ( u , c , t ) - s _ { X } ( v , c , t ) \| \leq L _ { t } \| u - v \| , \quad \forall u , v .
$$

Applying the Lipschitz property with

$$
u = r _ { t } + \sqrt { \bar { \alpha } _ { t } } f ( c ) , \quad v = r _ { t } + \sqrt { \bar { \alpha } _ { t } } m ( c ) ,
$$

we obtain

$$
\begin{array} { r } { \| s _ { f } ( r _ { t } , c , t ) - s _ { m } ( r _ { t } , c , t ) \| = \left\| s _ { X } ( r _ { t } + \sqrt { \bar { \alpha } _ { t } } f ( c ) , c , t ) - s _ { X } ( r _ { t } + \sqrt { \bar { \alpha } _ { t } } m ( c ) , c , t ) \right\| \leq L _ { t } \sqrt { \bar { \alpha } _ { t } } \| f ( c ) - m ( c ) \| . } \end{array}
$$

This inequality shows that, under regularity conditions, the discrepancy between the practical residual score $s _ { f }$ and the ideal centered score $s _ { m }$ is controlled by the baseline approximation error $\| f ( c ) - m ( c ) \|$ . In particular, if $f ( c )$ provides an accurate approximation of the conditional mean $m ( c )$ , then the induced residual score remains close to the optimal centered score. Moreover, the discrepancy is modulated by the diffusion coefficient $\sqrt { \bar { \alpha } _ { t } }$ , implying that the influence of baseline error diminishes as the diffusion process progresses toward higher noise levels. Consequently, residual diffusion yields a score field that is both stable and closer to the ideal centered representation, leading to a simpler and more favorable learning problem for a finite capacity denoising network.

## APPENDIX B

MATHEMATICAL DETAILS OF FORWARD PROCESS DERIVATION OF RESIDUAL DIFFUSION

This appendix derives the closed-form expression for the noisy residual $R _ { t } ^ { \operatorname* { m i } }$ given in Eq. (13), starting from the one-step forward diffusion transition in Eq. (10).

The one-step forward transition is

$$
R _ { t } ^ { m i } = \sqrt { \alpha _ { t } } R _ { t - 1 } ^ { m i } + \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( 0 , \mathrm { I } ) .
$$

Similarly,

$$
R _ { t - 1 } ^ { m i } = \sqrt { \alpha _ { t - 1 } } R _ { t - 2 } ^ { m i } + \sqrt { 1 - \alpha _ { t - 1 } } \epsilon _ { t - 1 } , \quad \epsilon _ { t - 1 } \sim \mathcal { N } ( 0 , \mathrm { I } ) .\tag{33}
$$

Substituting Eq. (33) into the one-step forward transition, we obtain

$$
\begin{array} { r } { R _ { t } ^ { m i } = \sqrt { \alpha _ { t } } \left( \sqrt { \alpha _ { t - 1 } } R _ { t - 2 } ^ { m i } + \sqrt { 1 - \alpha _ { t - 1 } } \epsilon _ { t - 1 } \right) + \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } } \\ { = \sqrt { \alpha _ { t } \alpha _ { t - 1 } } R _ { t - 2 } ^ { m i } + \sqrt { \alpha _ { t } ( 1 - \alpha _ { t - 1 } ) } \epsilon _ { t - 1 } + \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } . } \end{array}
$$

Continuing this expansion recursively, we obtain

$$
\begin{array} { r l } & { R _ { t } ^ { m i } = \sqrt { \alpha _ { t } \alpha _ { t - 1 } \cdot \cdot \cdot \alpha _ { 1 } } R _ { 0 } ^ { m i } } \\ & { \phantom { a a a a a } + \sqrt { \alpha _ { t } \alpha _ { t - 1 } \cdot \cdot \cdot \alpha _ { 2 } ( 1 - \alpha _ { 1 } ) } \epsilon _ { 1 } } \\ & { \phantom { a a a a a } + \sqrt { \alpha _ { t } \alpha _ { t - 1 } \cdot \cdot \cdot \alpha _ { 3 } ( 1 - \alpha _ { 2 } ) } \epsilon _ { 2 } } \\ & { \phantom { a a a a a a } + \cdots } \\ & { \phantom { a a a a a a a } + \sqrt { \alpha _ { t } ( 1 - \alpha _ { t - 1 } ) } \epsilon _ { t - 1 } + \sqrt { 1 - \alpha _ { t } } \epsilon _ { t } . } \end{array}
$$

Now define $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ . Thus, the first term becomes $\sqrt { \bar { \alpha } _ { t } } R _ { 0 } ^ { m i }$ . Now consider the noise terms. Since $\epsilon _ { i } \sim \mathcal { N } ( 0 , \mathrm { I } )$ are i.i.d., each scaled term is

$$
\sqrt { \alpha _ { t } \cdot \cdot \cdot \alpha _ { i + 1 } ( 1 - \alpha _ { i } ) } \epsilon _ { i } \sim { \mathcal N } \left( 0 , \alpha _ { t } \cdot \cdot \cdot \alpha _ { i + 1 } ( 1 - \alpha _ { i } ) \mathrm { I } \right) .
$$

Since linear combinations of independent Gaussian variables remain Gaussian, the sum of these terms is also Gaussian with variance given by the sum of the individual variances; that is,

$$
\sum _ { i = 1 } ^ { t } \alpha _ { t } \cdot \cdot \cdot \alpha _ { i + 1 } ( 1 - \alpha _ { i } ) = 1 - \bar { \alpha } _ { t } .
$$

Therefore, the total noise term becomes

$$
\begin{array} { r } { \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \mathrm { I } ) . } \end{array}
$$

Finally, we obtain

$$
R _ { t } ^ { m i } = \sqrt { \bar { \alpha } _ { t } } R _ { 0 } ^ { m i } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \mathrm { I } ) .
$$

This yields the closed-form forward noising expression used in Eq. (13).

## APPENDIX C RDDMPI TRAINING PROCEDURE

Following the conditional residual diffusion formulation described in Section IV, let $X _ { 0 } \in \mathbb { R } ^ { V \times L }$ denote a training window sampled from the data distribution, with V variables and sequence length L. We use $v \in \{ 1 , \ldots , V \}$ to index variables and $\ell \in \{ 1 , \ldots , L \}$ to index timesteps. Let $M \in \{ 0 , 1 \} ^ { V \times L }$ be the binary observation mask, where $M _ { v , \ell } = 1$ indicates an observed value, whereas $M _ { v , \ell } = 0$ indicates a missing value. During training, a subset of originally observed entries is randomly masked and treated as missing targets. The residual diffusion process is then applied only to the missing-region residual $R _ { 0 } ^ { \mathrm { m i } }$ , while the observed mask, the baseline-completed signal, and the latent baseline representation remain fixed and serve as conditioning information. At diffusion step t, Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ is injected into the missing-region residua

$$
R _ { t } ^ { \mathrm { m i } } = \sqrt { \bar { \alpha } _ { t } } R _ { 0 } ^ { \mathrm { m i } } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ,
$$

where $\{ \alpha _ { t } \} _ { t = 1 } ^ { T }$ is a predefined variance schedule and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ . The model is trained to predict the injected noise using a conditional noise predicto

$$
\epsilon _ { \theta } \big ( R _ { t } ^ { \mathrm { m i } } , t , c \big ) \ : .
$$

Since diffusion is applied only to the missing-region residual, the denoising loss is computed exclusively on missing positions:

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } \left[ \frac { 1 } { \sum _ { \boldsymbol { v } , \boldsymbol { \ell } } ( 1 - M _ { \boldsymbol { v } , \boldsymbol { \ell } } ) } \sum _ { \boldsymbol { v } = 1 } ^ { V } \sum _ { \boldsymbol { \ell } = 1 } ^ { L } ( 1 - M _ { \boldsymbol { v } , \boldsymbol { \ell } } ) \left( \epsilon _ { \boldsymbol { v } , \boldsymbol { \ell } } - \epsilon _ { \boldsymbol { \theta } } ( \cdot ) _ { \boldsymbol { v } , \boldsymbol { \ell } } \right) ^ { 2 } \right] .
$$

```latex
The self-supervised training procedure of the proposed residual diffusion model is summarized in Algorithm 2.
Algorithm 2: Training of RDDMPI
Input: training data distribution $q ( X _ { 0 } )$ , pretrained deterministic baseline $f _ { \mathrm { b a s e } } ,$ number of optimization iterations $N _ { \mathrm { i t e r } } .$
noise schedule $\{ \alpha _ { t } \} _ { t = 1 } ^ { T }$ with $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$
2: Output: trained conditional noise predictor $\epsilon _ { \theta }$
3: for $i = 1$ to $N _ { \mathrm { i t e r } }$ do
4: Sample a training window $X _ { 0 } \sim q ( X _ { 0 } )$ and a diffusion timestep $t \sim$ Uniform $( \{ 1 , \ldots , T \} )$
5: Sample a mask by randomly hiding a subset of currently observed entries, producing M
6: Compute observed and missing components: $X _ { 0 } ^ { \mathrm { o b } } \gets \dot { M } \odot X _ { 0 } , ~ X _ { 0 } ^ { \mathrm { m i } } \gets ( 1 - \overline { { \dot { } M } } ) \odot \bar { X } _ { 0 }$
7: Obtain baseline-completed signal and latent representation: $\tilde { X } ^ { \mathrm { b a s e } } , H ^ { \mathrm { b a s e } }  f _ { \mathrm { b a s e } } ( X _ { 0 } ^ { \mathrm { o b } } , M )$
8: Construct the missing-region residual target: $R _ { 0 } ^ { \mathrm { { m i } } }  ( 1 - M ) \odot ( X _ { 0 } - { \tilde { X } } ^ { \mathrm { { b a s e } } } )$
9: Sample Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ with the same shape as $R _ { 0 } ^ { \mathrm { m i } }$
10: Apply forward noising to the residual target: ${ R _ { t } ^ { \mathrm { m i } } } \gets \sqrt { \bar { \alpha } _ { t } } { R _ { 0 } ^ { \mathrm { m i } } } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ ϵ
11: Predict noise using the residual diffusion model: $\hat { \epsilon } \gets \epsilon _ { \theta } \big ( R _ { t } ^ { \mathrm { m i } } , t , c \big )$
12: Take a gradient step minimizing $\begin{array} { r } { \nabla _ { \theta } \left( \frac { 1 } { \sum _ { v , \ell } \left( 1 - M _ { v , \ell } \right) } \sum _ { v = 1 } ^ { V } \sum _ { \ell = 1 } ^ { L } ( 1 - M _ { v , \ell } ) \left( \epsilon _ { v , \ell } - \hat { \epsilon } _ { v , \ell } \right) ^ { 2 } \right) } \end{array}$
13: end for
```

## APPENDIX D EFFICIENCY ANALYSIS

Table V presents computational efficiency and performance metrics across RDDMPI, DLinear [44], ModernTCN [14], iTransformer [16], TimesNet [15], SAITS [10], ImputeFormer [18], T1 [19], GP-VAE [26], CSDI [27] and FGTI [31].

TABLE V: Computational efficiency and performance comparison on the ETTh1 and Illness datasets. Params (M): parameters in millions; Train Speed: ms per iteration; Inference Speed: ms per sample; MAE: Mean Absolute Error averaged over point missing ratios 0.2, 0.4, 0.6, and 0.8 (lower is better).
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=2>Model             Parameters (M)</td><td rowspan=1 colspan=3>Train Speed (ms/iter) | Inference Speed (ms/sample) MAE</td></tr><tr><td rowspan=11 colspan=1>ETTh1</td><td rowspan=1 colspan=1>RDDMPI (Ours)</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>213.96</td><td rowspan=1 colspan=1>2220.24</td><td rowspan=1 colspan=1>0.1365</td></tr><tr><td rowspan=1 colspan=1>DLinear</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>4.02</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.2906</td></tr><tr><td rowspan=1 colspan=1>ModernTCN</td><td rowspan=1 colspan=1>1.72</td><td rowspan=1 colspan=1>7.42</td><td rowspan=1 colspan=1>0.50</td><td rowspan=2 colspan=1>0.21110.2569</td></tr><tr><td rowspan=1 colspan=1>iTransformer</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>7.27</td><td rowspan=1 colspan=1>0.08</td></tr><tr><td rowspan=2 colspan=1>TimesNetSAITS</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>22.06</td><td rowspan=1 colspan=1>1.28</td><td rowspan=1 colspan=1>0.2567</td></tr><tr><td rowspan=1 colspan=1>5.27</td><td rowspan=1 colspan=1>51.68</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.2113</td></tr><tr><td rowspan=5 colspan=1>ImputeFormerT1GP-VAECSDIFGTI</td><td rowspan=1 colspan=1>1.37</td><td rowspan=1 colspan=1>21.83</td><td rowspan=1 colspan=1>0.47</td><td rowspan=3 colspan=1>0.30440.16790.5860</td></tr><tr><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>12.80</td><td rowspan=1 colspan=1>0.33</td></tr><tr><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>23.04</td><td rowspan=1 colspan=1>19.60</td></tr><tr><td rowspan=1 colspan=1>1.19</td><td rowspan=1 colspan=1>188.98</td><td rowspan=1 colspan=1>2315.40</td><td rowspan=2 colspan=1>0.14700.1520</td></tr><tr><td rowspan=1 colspan=1>1.96</td><td rowspan=1 colspan=1>45.72</td><td rowspan=1 colspan=1>3696.13</td></tr><tr><td rowspan=11 colspan=1>Illness</td><td rowspan=2 colspan=1>RDDMPI (Ours)DLinear</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>313.97</td><td rowspan=1 colspan=1>12614.79</td><td rowspan=1 colspan=1>0.0714</td></tr><tr><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>5.48</td><td rowspan=1 colspan=1>1.07</td><td rowspan=1 colspan=1>0.3046</td></tr><tr><td rowspan=1 colspan=1>ModernTCN</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>22.10</td><td rowspan=1 colspan=1>3.46</td><td rowspan=1 colspan=1>0.1736</td></tr><tr><td rowspan=5 colspan=1>iTransformerTimesNetSAITSImputeFormerT1</td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>18.56</td><td rowspan=1 colspan=1>1.70</td><td rowspan=2 colspan=1>0.22010.2054</td></tr><tr><td rowspan=1 colspan=1>4.69</td><td rowspan=1 colspan=1>52.15</td><td rowspan=1 colspan=1>15.81</td></tr><tr><td rowspan=1 colspan=1>5.27</td><td rowspan=1 colspan=1>83.61</td><td rowspan=1 colspan=1>6.32</td><td rowspan=1 colspan=1>0.3044</td></tr><tr><td rowspan=1 colspan=1>1.37</td><td rowspan=1 colspan=1>39.84</td><td rowspan=1 colspan=1>5.79</td><td rowspan=1 colspan=1>0.3423</td></tr><tr><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>15.24</td><td rowspan=1 colspan=1>5.47</td><td rowspan=1 colspan=1>0.0849</td></tr><tr><td rowspan=1 colspan=1>GP-VAE</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>23.19</td><td rowspan=1 colspan=1>19.24</td><td rowspan=3 colspan=1>0.51770.14790.1464</td></tr><tr><td rowspan=1 colspan=1>CSDI</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>268.52</td><td rowspan=1 colspan=1>13997.47</td></tr><tr><td rowspan=1 colspan=1>FGTI</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>87.67</td><td rowspan=1 colspan=1>18958.83</td></tr></table>

Table V shows that RDDMPI has a substantially higher inference cost than deterministic baselines, although it achieves better imputation accuracy on both datasets. This difference is primarily caused by iterative probabilistic sampling rather than the model size. Deterministic methods produce a point estimate through a single forward pass, whereas generating one stochastic imputation sample with a diffusion model requires a complete reverse trajectory of $T$ sequential denoising steps. At each step, the current noisy state is updated according to

$$
x _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t , \mathrm { c o n d } ) \right) + \sigma _ { t } z , \qquad z \sim \mathcal { N } ( 0 , I ) .
$$

Also, a single reverse trajectory produces one imputed sample, i.e., one possible estimate of the missing values. To approximate the predictive distribution, RDDMPI generates N independent samples, each requiring its own T-step reverse trajectory. Inference therefore requires NT denoising-network evaluations in total. The resulting samples are aggregated to obtain the reported point imputation and are also used to quantify predictive uncertainty. Although the $N$ trajectories may be evaluated in paralle through batching, the $T$ denoising steps within each trajectory remain sequential. Consequently, inference cost depends on both the number of diffusion steps $T$ and the number of generated samples $N ,$ , explaining the higher inference times of RDDMPI, CSDI, and FGTI relative to single-pass deterministic methods.

a) Accelerated Diffusion Sampling.: To mitigate this limitation, we explore faster sampling strategies such as Denoising Diffusion Implicit Models (DDIM) [47], which reduce the number of required steps while maintaining performance. DDIM introduces a non-Markovian deterministic sampling process:

$$
\begin{array} { r } { \mathbf { x } _ { t - 1 } = \sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathbf { x } } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) , } \end{array}
$$

where $\hat { \mathbf { x } } _ { 0 }$ is the predicted clean signal. By selecting a subset of timesteps $\{ t _ { 1 } , \ldots , t _ { K } \}$ with $K \ll T .$ , inference can be significantly accelerated. We evaluate DDIM-based sampling with reduced step counts and observe that RDDMPI maintains competitive performance even with substantially fewer inference steps, while achieving notable speedups. This highlights the potential of accelerated diffusion methods to bridge the efficiency gap with deterministic models.

TABLE VI: Results of RDDMPI under point missing ratios (0.2, 0.4, 0.6, 0.8). This table compares the default setting $( T = 5 0 )$ with a reduced number of steps (T = 10), along with the relative degradation $\Delta$ in %.
<table><tr><td colspan="2">Models</td><td colspan="3">RDDMPI (T = 50)</td><td colspan="3">RDDMPI (T = 10)</td><td colspan="3">∆ (in %)</td></tr><tr><td colspan="2">Metric</td><td>MSE</td><td>MAE</td><td>CRPS</td><td>MSE</td><td>MAE</td><td>CRPS</td><td>MSE</td><td>MAE</td><td>CRPS</td></tr><tr><td rowspan="4">ET1</td><td>0.2 0.4</td><td>0.0229</td><td>0.0956</td><td>0.0924</td><td>0.0236</td><td>0.0970</td><td>0.0957</td><td>+3.06</td><td>+1.46</td><td>+3.57</td></tr><tr><td></td><td>0.0337</td><td>0.1123</td><td>0.1080</td><td>0.0345</td><td>0.1139</td><td>0.1117</td><td>+2.37</td><td>+1.42</td><td>+3.43</td></tr><tr><td>0.6 0.8</td><td>0.0480</td><td>0.1362</td><td>0.1310</td><td>0.0495</td><td>0.1390</td><td>0.1358</td><td>+3.13</td><td>+2.06</td><td>+3.66</td></tr><tr><td></td><td>0.1145</td><td>0.2020</td><td>0.1940</td><td>0.1216</td><td>0.2102</td><td>0.2031</td><td>+6.20</td><td>+4.06</td><td>+4.69</td></tr><tr><td rowspan="4">ETTT2</td><td>0.2</td><td>0.0228</td><td>0.0771</td><td>0.0438</td><td>0.0223</td><td>0.0786</td><td>0.0455</td><td>-2.19</td><td>+1.95</td><td>+3.88</td></tr><tr><td>0.4</td><td>0.0305</td><td>0.0924</td><td>0.0528</td><td>0.0295</td><td>0.0935</td><td>0.0540</td><td>-3.28</td><td>+1.19</td><td>+2.27</td></tr><tr><td>0.6</td><td>0.0437</td><td>0.1150</td><td>0.0657</td><td>0.0422</td><td>0.1157</td><td>0.0666</td><td>-3.43</td><td>+0.61</td><td>+1.37</td></tr><tr><td>0.8</td><td>0.0797</td><td>0.1649</td><td>0.0941</td><td>0.0760</td><td>0.1646</td><td>0.0941</td><td>-4.64</td><td>-0.18</td><td>0.00</td></tr><tr><td rowspan="4">Excange</td><td>0.2</td><td>0.0022</td><td>0.0147</td><td>0.0131</td><td>0.0022</td><td>0.0150</td><td>0.0138</td><td>0.00</td><td>+2.04</td><td>+5.34</td></tr><tr><td>0.4</td><td>0.0015</td><td>0.0171</td><td>0.0151</td><td>0.0042</td><td>0.0185</td><td>0.0169</td><td>+180.00</td><td>+8.19</td><td>+11.92</td></tr><tr><td>0.6</td><td>0.0019</td><td>0.0206</td><td>0.0181</td><td>0.0035</td><td>0.0219</td><td>0.0198</td><td>+84.21</td><td>+6.31</td><td>+9.39</td></tr><tr><td>0.8</td><td>0.0030</td><td>0.0288</td><td>0.0252</td><td>0.0030</td><td>0.0290</td><td>0.0257</td><td>0.00</td><td>+0.69</td><td>+1.98</td></tr><tr><td rowspan="4">Ins</td><td>0.2</td><td>0.0057</td><td>0.0449</td><td>0.0460</td><td>0.0059</td><td></td><td>0.0494</td><td>+3.51</td><td>+4.23</td><td>+7.39</td></tr><tr><td>0.4</td><td>0.0081</td><td>0.0518</td><td>0.0526</td><td>0.0092</td><td>0.0468 0.0547</td><td>0.0563</td><td>+13.58</td><td>+5.60</td><td>+7.03</td></tr><tr><td>0.6</td><td>0.0132</td><td>0.0664</td><td>0.0680</td><td>0.0151</td><td>0.0702</td><td>0.0727</td><td>+14.39</td><td>+5.72</td><td>+6.91</td></tr><tr><td>0.8</td><td>0.0673</td><td>0.1225</td><td>0.1248</td><td>0.0720</td><td>0.1355</td><td>0.1370</td><td>+6.98</td><td>+10.61</td><td>+9.78</td></tr><tr><td rowspan="4">Weather</td><td>0.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.4</td><td>0.0246 0.0271</td><td>0.0246</td><td>0.0327</td><td>0.0233</td><td>0.0258</td><td>0.0361</td><td>-5.28</td><td>+4.88</td><td>+10.40</td></tr><tr><td>0.6</td><td>0.0313</td><td>0.0271 0.0317</td><td>0.0362 0.0423</td><td>0.0270 0.0321</td><td>0.0289 0.0336</td><td>0.0401 0.0462</td><td>-0.37 +2.56</td><td>+6.64 +5.99</td><td>+10.77</td></tr><tr><td>0.8</td><td>0.0409</td><td>0.0425</td><td>0.0573</td><td>0.0412</td><td>0.0441</td><td>0.0599</td><td>+0.73</td><td>+3.76</td><td>+9.22 +4.54</td></tr></table>

Future work should focus on improving inference efficiency through advanced sampling techniques, such as adaptive timestep selection or one-step diffusion models. These directions could further reduce computational cost while preserving the strong performance benefits of diffusion-based imputation.

## APPENDIX E

## RESIDUAL DENOISING BLOCK ARCHITECTURE

This appendix provides a detailed description of the residual denoising blocks used in the conditional noise prediction network $\epsilon _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c )$ . Let V denote the number of variables, L the sequence length, $d _ { h }$ the number of hidden channels, and $N _ { \mathrm { b l k } }$ the number of residual denoising blocks.

The reliability-aware fusion and FiLM-based conditioning mechanisms described in Section IV-C produce the FiLM-conditioned hidden representation $\tilde { Z } _ { t }$ , which is the input to the residual denoising stack and initialized as $\dot { \boldsymbol U } _ { t } ^ { ( 0 ) } = \tilde { \boldsymbol Z } _ { t } \in \mathbb { R } ^ { V \times L \times d _ { h } }$ . This representation is passed sequentially through $N _ { \mathrm { b l k } }$ residual denoising blocks. For block $b \in \{ 1 , \dots , N _ { \mathrm { b l k } } \}$ , we define

$$
U _ { t } ^ { ( b , 0 ) } = U _ { t } ^ { ( b - 1 ) } ,
$$

where $U _ { t } ^ { ( b - 1 ) }$ is the output of the preceding block. Each block then applies diffusion-step conditioning, temporal attention, variable attention, side-information conditioning, a gated activation, and residual and skip projections. The complete sequence of operations is summarized in Algorithm 3 and described below.

## A. Diffusion-Step Conditioning

The discrete diffusion step t is represented by a fixed sinusoidal embedding, followed by two learned linear projections with SiLU activations as

$$
e _ { t } = \mathrm { S i L U } \left( W _ { e , 2 } \mathrm { S i L U } \left( W _ { e , 1 } e _ { t } ^ { ( 0 ) } + b _ { e , 1 } \right) + b _ { e , 2 } \right) ,
$$

where $e _ { t } ^ { ( 0 ) }$ is the fixed sinusoidal representation, d<sub>dif</sub> denotes the dimension of the diffusion-step embedding, and $e _ { t } \in \mathbb { R } ^ { d _ { \mathrm { d i f f } } }$ is the resulting diffusion-step embedding. Within block b, this embedding is projected to the hidden dimension as

$$
\boldsymbol { d } _ { t } ^ { ( b ) } = \boldsymbol { W } _ { \mathrm { d i f f } } ^ { ( b ) } \boldsymbol { e } _ { t } + \boldsymbol { b } _ { \mathrm { d i f f } } ^ { ( b ) } \in \mathbb { R } ^ { d _ { h } } .
$$

The projected embedding is broadcast across variables and timesteps and added to the incoming representation:

$$
U _ { t } ^ { ( b , 1 ) } = U _ { t } ^ { ( b , 0 ) } + \mathrm { B r o a d c a s t } \left( d _ { t } ^ { ( b ) } \right) .
$$

Thus, $U _ { t } ^ { ( b , 1 ) } \in \mathbb { R } ^ { V \times L \times d _ { h } }$ contains both the hidden residual features and information about the current diffusion noise level.

## B. Temporal and Cross-Variable Modeling

The representation is first processed by a temporal transformer independently for each variable as

$$
U _ { t } ^ { ( b , 2 ) } [ v , : , : ] = \mathcal { T } _ { \mathrm { t i m e } } ^ { ( b ) } \left( U _ { t } ^ { ( b , 1 ) } [ v , : , : ] \right) , \qquad v = 1 , \ldots , V .
$$

For each variable, the corresponding $L \times d _ { h }$ slice is treated as a temporal sequence. Therefore, the temporal transformer models dependencies across timesteps without mixing the variable dimension. The resulting representation is subsequently processed by a variable transformer independently at each timestep as

$$
U _ { t } ^ { ( b , 3 ) } [ : , \ell , : ] = \mathcal { T } _ { \mathrm { v a r } } ^ { ( b ) } \left( U _ { t } ^ { ( b , 2 ) } [ : , \ell , : ] \right) , \qquad \ell = 1 , \ldots , L .
$$

At each timestep, the corresponding $V \times d _ { h }$ slice is treated as a sequence over variables. The variable transformer therefore models cross-variable interactions while preserving the temporal positions.

Both transformer modules use multi-head self-attention, residual connections, layer normalization, and a position-wise feed-forward network with a GELU activation. For an input sequence X, an individual attention head is computed as

$$
\mathrm { h e a d } _ { j } ( X ) = \mathrm { s o f t m a x } \left( \frac { ( X W _ { j } ^ { Q } ) ( X W _ { j } ^ { K } ) ^ { \top } } { \sqrt { d _ { k } } } \right) X W _ { j } ^ { V } ,
$$

and the multi-head output is

$$
\mathrm { M H A } ( X ) = \mathrm { C o n c a t } \left( \mathrm { h e a d } _ { 1 } ( X ) , \dots , \mathrm { h e a d } _ { H } ( X ) \right) W ^ { O } ,
$$

where H is the number of attention heads and $d _ { k }$ is the dimension of each query and key head.

## C. Side-Information Conditioning

Let $S \in \mathbb { R } ^ { V \times L \times d _ { \mathrm { s i d e } } }$ denote the side-information tensor provided to every residual denoising block. The side information consists of a temporal-position embedding, a learned variable embedding, the observation mask, and the learned reliability map. For each temporal coordinate $\tau _ { \ell } .$ , we construct a $d _ { \mathrm { t i m e } }$ -dimensional sinusoidal embedding. Its components are given by

$$
\begin{array} { r } { E _ { \mathrm { t i m e } } ( \ell , 2 j ) = \sin \left( \frac { \tau \ell } { 1 0 0 0 0 ^ { 2 j / d _ { \mathrm { t i m e } } } } \right) , } \\ { E _ { \mathrm { t i m e } } ( \ell , 2 j + 1 ) = \cos \left( \frac { \tau \ell } { 1 0 0 0 0 ^ { 2 j / d _ { \mathrm { t i m e } } } } \right) . } \end{array}
$$

The same temporal embedding is replicated across all variables, resulting in $E _ { \mathrm { t i m e } } \in \mathbb { R } ^ { V \times L \times d _ { \mathrm { t i m e } } }$ , where $d _ { \mathrm { t i m e } }$ denotes the dimension of the temporal-position embedding. Each variable v is assigned a learned embedding vector $e _ { v } \in \mathbb { R } ^ { d _ { \operatorname { v a r } } }$ , where $d _ { \mathrm { v a r } }$ denotes the dimension of the variable embedding, through an embedding lookup table: $e _ { v } = \operatorname { E m b } _ { \operatorname { v a r } } ( v )$ . This embedding is replicated across all timesteps to obtain $E _ { \mathrm { v a r } } \in \mathbf { \tilde { \mathbb { R } } } ^ { \mathbf { \tilde { V } } \times L \times d _ { \mathrm { v a r } } }$ . The observation mask $M \in \{ 0 , 1 \} ^ { V \times L }$ indicates which entries are available as conditioning information. The learned reliability map $A \in [ 0 , 1 ] ^ { V \times L }$ is computed by the reliability-aware mechanism described in Section IV-C. After treating M and A as one-dimensional feature channels, the complete side-information tensor is

$$
S = { \mathrm { C o n c a t } } \left( E _ { \mathrm { t i m e } } , E _ { \mathrm { v a r } } , M , A \right) \in \mathbb { R } ^ { V \times L \times d _ { \mathrm { s i d e } } } ,
$$

where $d _ { \mathrm { s i d e } }$ denotes the total dimension of the side-information features and is given by $d _ { \mathrm { s i d e } } = d _ { \mathrm { t i m e } } + d _ { \mathrm { v a r } } + 2$ , with the additional two dimensions corresponding to the observation mask M and reliability map A. The reliability map therefore affects

the network twice: first during the reliability-weighted fusion of the baseline-completed signal and again as side information within every residual denoising block.

The output of the variable transformer and the side-information tensor are projected independently to $2 d _ { h }$ hidden dimensions. Let $\psi _ { \mathrm { m i d } } ^ { ( b ) }$ and $\psi _ { \mathrm { c o n d } } ^ { ( b ) }$ denote the learned pointwise projections in block $b ,$ implemented as $1 \times 1$ convolutions. Specifically, $\psi _ { \mathrm { m i d } } ^ { ( b ) }$ maps $d _ { h }$ to $2 d _ { h }$ dimensions, whereas $\bar { \psi } _ { \mathrm { c o n d } } ^ { ( b ) }$ maps $d _ { \mathrm { s i d e } }$ to $2 d _ { h }$ dimensions. Their outputs are combined as

$$
U _ { t } ^ { ( b , 4 ) } = \psi _ { \mathrm { m i d } } ^ { ( b ) } \left( U _ { t } ^ { ( b , 3 ) } \right) + \psi _ { \mathrm { c o n d } } ^ { ( b ) } \left( S \right) ,
$$

where $U _ { t } ^ { ( b , 4 ) } \in \mathbb { R } ^ { V \times L \times 2 d _ { h } }$ . Because both operators are pointwise, the same learned transformation is applied independently at every variable-timestep position, without mixing information across variables or timesteps.

## D. Gated Activation

The conditioned representation is split equally along its hidden dimension as

$$
\left( U _ { t , \mathrm { g a t e } } ^ { \left( b , 4 \right) } , U _ { t , \mathrm { f i l t e r } } ^ { \left( b , 4 \right) } \right) = \mathrm { S p l i t } \left( U _ { t } ^ { \left( b , 4 \right) } \right) ,
$$

where both components belong to $\mathbb { R } ^ { V \times L \times d _ { h } }$ . A gated activation is then applied as

$$
U _ { t } ^ { ( b , 5 ) } = \sigma \left( U _ { t , \mathrm { g a t e } } ^ { ( b , 4 ) } \right) \odot \operatorname { t a n h } \left( U _ { t , \mathrm { f i l t e r } } ^ { ( b , 4 ) } \right) .
$$

The sigmoid component controls how much information is propagated, while the hyperbolic-tangent component provides the nonlinear candidate features. Their element-wise product gives $\mathbf { \bar { \chi } } _ { t } ^ { ( b , 5 ) } \in \mathbb { R } ^ { V \times L \times d _ { h } }$ .

## E. Residual and Skip Outputs

The gated representation is projected pointwise from $d _ { h }$ to $2 d _ { h }$ hidden dimensions. Let $\psi _ { \mathrm { o u t } } ^ { ( b ) }$ denote the learned block-output projection, implemented as $\textbf { a } 1 \times 1$ convolution: $\psi _ { \mathrm { o u t } } ^ { ( b ) } : \bar { \mathbb { R } } ^ { V \times L \times \ddot { d } _ { h } } \to \mathbb { R } ^ { V \times L \times 2 d _ { h } }$ . The projected representation is therefore

$$
U _ { t } ^ { ( b , 6 ) } = \psi _ { \mathrm { o u t } } ^ { ( b ) } \left( U _ { t } ^ { ( b , 5 ) } \right) \in \mathbb { R } ^ { V \times L \times 2 d _ { h } } .
$$

Because the projection is pointwise, it transforms the hidden features independently at every variable-timestep position, without mixing information across variables or timesteps.

The resulting representation is divided equally along the hidden dimension into a residual update and a skip connection as

$$
\left( U _ { t , \mathrm { r e s } } ^ { ( b ) } , U _ { t , \mathrm { s k i p } } ^ { ( b ) } \right) = \mathrm { S p l i t } \left( U _ { t } ^ { ( b , 6 ) } \right) ,
$$

where $U _ { t , \mathrm { r e s } } ^ { ( b ) } , U _ { t , \mathrm { s k i p } } ^ { ( b ) } \in \mathbb { R } ^ { V \times L \times d _ { h } }$ . The residual output is added to the block input and scaled by $1 / \sqrt { 2 }$

$$
U _ { t } ^ { ( b ) } = \frac { U _ { t } ^ { ( b , 0 ) } + U _ { t , \mathrm { r e s } } ^ { ( b ) } } { \sqrt { 2 } } .
$$

The resulting tensor ${ U } _ { t } ^ { ( b ) }$ becomes the input to the next residual denoising block. The factor $1 / \sqrt { 2 }$ helps control the magnitude of the hidden activations across the sequence of blocks. The skip output $\dot { U } _ { t , \mathrm { s k i p } } ^ { ( b ) }$ is retained for the final noise prediction path.

## F. Skip Aggregation and Noise Prediction

After all residual denoising blocks have been evaluated, their skip outputs are summed and normalized as

$$
U _ { t , \mathrm { s k i p } } = \frac { 1 } { \sqrt { N _ { \mathrm { b l k } } } } \sum _ { b = 1 } ^ { N _ { \mathrm { b l k } } } U _ { t , \mathrm { s k i p } } ^ { ( b ) } .
$$

The aggregated skip representation is processed by two learned pointwise projections. The first projection, $\psi _ { \mathrm { o u t , 1 } }$ , is a $1 \times 1$ convolution that preserves the hidden dimension: $\dot { \psi _ { \mathrm { o u t , 1 } } } : \mathbb { R } ^ { V \times L \times \hat { d } _ { h } } \to \mathbb { R } ^ { V \times L \times \mathbf { \tilde { d } } _ { h } }$ . It is followed by a ReLU activation as

$$
\begin{array} { r } { U _ { t } ^ { \mathrm { o u t } } = \mathrm { R e L U } \left( \psi _ { \mathrm { o u t , 1 } } \left( U _ { t , \mathrm { s k i p } } \right) \right) . } \end{array}
$$

The second projection, $\psi _ { \mathrm { o u t , 2 } } ,$ is a $1 \times 1$ convolution that maps the $d _ { h }$ hidden features to one predicted noise value at every variable-timestep position: $\psi _ { \mathrm { o u t } , 2 } : \mathbb { R } ^ { V \times L \times d _ { h } } \to \mathbb { R } ^ { V \times L }$ . The final noise prediction is

$$
\begin{array} { r } { \widehat { \epsilon } _ { \theta } = \psi _ { \mathrm { o u t , 2 } } \left( U _ { t } ^ { \mathrm { o u t } } \right) \in \mathbb { R } ^ { V \times L } . } \end{array}
$$

Both output projections operate independently at each variable-timestep position. The temporal and cross-variable interactions have already been incorporated by the transformer layers within the residual denoising blocks.

Therefore, $\widehat { \epsilon } _ { \theta } = \epsilon _ { \theta } \left( R _ { t } ^ { \mathrm { m i } } , t , c \right)$ has the same variable and temporal dimensions as the noisy residual and represents the Gaussian noise predicted for the reverse diffusion process.

Algorithm 3: Sequential Operations of the Residual Denoising Network   
Input: FiLM-conditioned representation $U _ { t } ^ { ( 0 ) } = \widetilde { Z } _ { t } ;$ diffusion step $t ;$ side-information tensor $S ;$ number of residual   
blocks $N _ { \mathrm { b l k } }$   
Output: Predicted noise $\widehat { \epsilon } _ { \theta } = \epsilon _ { \theta } ( R _ { t } ^ { \mathrm { m i } } , t , c )$   
Compute the diffusion-step embedding $e _ { t }$ (Subsection $\mathrm { E } { \mathrm { - } } \mathrm { A } )$   
Initialize an empty collection of skip representations;   
for $b = 1$ to $N _ { \mathrm { b l k } }$ do   
Set the block input: $U _ { t } ^ { ( b , 0 ) } \gets U _ { t } ^ { ( b - 1 ) }$ ;   
Project and inject the diffusion-step embedding: $U _ { t } ^ { ( b , 1 ) } \gets U _ { t } ^ { ( b , 0 ) }$ + Broadcast $( W _ { \mathrm { d i f f } } ^ { ( b ) } e _ { t } + b _ { \mathrm { d i f f } } ^ { ( b ) } ) ;$   
Apply temporal self-attention independently to each variable: $U _ { t } ^ { ( b , 2 ) } \gets \mathcal { T } _ { \mathrm { t i m e } } ^ { ( b ) } ( \dot { U } _ { t } ^ { ( b , 1 ) } ) ;$   
Apply cross-variable self-attention independently at each timestep: $U _ { t } ^ { ( b , 3 ) }  \mathcal { T } _ { \mathrm { v a r } } ^ { ( \bar { b } ) } ( U _ { t } ^ { ( b , 2 ) } ) ;$   
Apply the pointwise hidden and side-information projections: $\bar { U _ { t } ^ { ( b , 4 ) } }  \psi _ { \mathrm { m i d } } ^ { ( b ) } ( U _ { t } ^ { ( b , 3 ) } ) + \bar { \psi } _ { \mathrm { c o n d } } ^ { ( b ) } ( S ) :$   
Split $\bar { U } _ { t } ^ { ( b , 4 ) }$ into $U _ { t , \mathrm { g a t e } } ^ { ( b , 4 ) }$ and $U _ { t , \mathrm { f i l t e r } } ^ { ( b , 4 ) } ;$   
Apply the gated activation: $U _ { t } ^ { ( \bar { b } , 5 ) }  \sigma ( U _ { t , \mathrm { g a t e } } ^ { ( b , 4 ) } ) \odot$ tanh $( U _ { t , \mathrm { f i l t e r } } ^ { ( b , 4 ) } ) ;$   
Apply the pointwise output projection: $U _ { t } ^ { ( \breve { b , 6 } ) } \gets \psi _ { \mathrm { o u t } } ^ { ( b ) } ( U _ { t } ^ { ( \breve { b , 5 } ) } ) ;$   
Split $U _ { t } ^ { ( b , 6 ) }$ into the residual update $U _ { t , \mathrm { r e s } } ^ { ( b ) }$ and skip representation $U _ { t , \mathrm { s k i p } } ^ { ( b ) } ;$   
Update the residual path: $U _ { t } ^ { ( b ) } \gets ( U _ { t } ^ { ( b , 0 ) } + U _ { t , \mathrm { r e s } } ^ { ( b ) } ) / \sqrt { 2 } ;$   
Store $U _ { t , \mathrm { s k i p } } ^ { ( b ) } ;$   
Aggregate and normalize the skip representations: $\begin{array} { r } { U _ { t , \mathrm { s k i p } }  \sum _ { b = 1 } ^ { N _ { \mathrm { b l k } } } U _ { t , \mathrm { s k i p } } ^ { ( b ) } / \sqrt { N _ { \mathrm { b l k } } } ; } \end{array}$   
Apply the first pointwise output projection: $U _ { t } ^ { \mathrm { o u t } } \gets \mathrm { R e L U } ( \psi _ { \mathrm { o u t , 1 } } ( U _ { t , \mathrm { s k i p } } ) ) ;$   
Apply the final pointwise output projection: $\widehat { \epsilon } _ { \theta } \gets \psi _ { \mathrm { o u t , 2 } } ( U _ { t } ^ { \mathrm { o u t } } ) ;$   
return ${ \widehat { \epsilon } } _ { \theta } ;$

## APPENDIX F

## DATASET-SPECIFIC RDDMPI HYPERPARAMETERS

Table VII reports the selected dataset-specific optimization and architectural hyperparameters used to train RDDMPI. These include the batch size, learning rate, number of residual denoising blocks, number of hidden channels, and number of attention heads. The deterministic T1 backbone is pretrained separately for each dataset and remains frozen during residual diffusion training.

TABLE VII: Dataset-specific hyperparameters for RDDMPI. $N _ { \mathrm { b l k } }$ denotes the number of residual denoising blocks, and $d _ { h }$ denotes the number of hidden channels.
<table><tr><td>Dataset</td><td>Batch Size</td><td>Learning rate</td><td> $N _ { \mathrm { b l k } }$ </td><td> $d _ { h }$ </td><td>Number of heads</td></tr><tr><td>ETTh1</td><td>16</td><td>10-3</td><td>4</td><td>128</td><td>8</td></tr><tr><td>ETTh2</td><td>16</td><td>10-3</td><td>4</td><td>128</td><td>8</td></tr><tr><td>Exchange</td><td>16</td><td> $1 0 ^ { - 3 }$ </td><td>4</td><td>128</td><td>8</td></tr><tr><td>Illness</td><td>64</td><td> $1 0 ^ { - 3 }$ </td><td>4</td><td>64</td><td>8</td></tr><tr><td>Weather</td><td>16</td><td> $1 0 ^ { - 4 }$ </td><td>4</td><td>64</td><td>8</td></tr></table>

## APPENDIX G

## FULL RESULTS

This section reports the complete quantitative results underlying the aggregated comparisons presented in the main paper. For point missingness, results are shown separately for missing ratios of 0.2, 0.4, 0.6, and 0.8, rather than averaging across these ratios as in the main tables. We also report the results under block missingness.

All experiments are conducted using five random seeds: 2, 102, 202, 302, and 402. For each seed, the models are trained independently and evaluated on the corresponding artificially masked test data. MAE, MSE, and CRPS are first computed for each seed, after which we report their mean and standard deviation across the five runs. The mean tables summarize average performance, while the standard deviation tables quantify sensitivity to model initialization, training stochasticity, and the generated missingness patterns. We provide both the disaggregated results and their variability for transparency, reproducibility, and a more complete assessment of model robustness.

TABLE VIII: Full results under point missing ratios (0.2, 0.4, 0.6, 0.8) across datasets. Best results are marked in bold, and second-best results are marked in underlined.
<table><tr><td rowspan="2">Models Metric</td><td colspan="2">|RDDMPI (Ours)</td><td rowspan="2">MSE</td><td rowspan="2">DLinear MAE MSE</td><td rowspan="2">ModernTCN MAE MSE</td><td rowspan="2">iTransformer MAE</td><td rowspan="2">SAITS MSE MAE</td><td rowspan="2">ImputeFormer MSE MAE</td><td rowspan="2">TimesNet MSE MAE</td><td rowspan="2">T1 MSE</td><td rowspan="2">MAE MSE</td><td rowspan="2">GP-VAE MAE</td><td rowspan="2">CSDI MSE MAE</td><td rowspan="2">MSE</td><td rowspan="2">FGTI MAE</td></tr><tr><td>MSE</td><td>MAE</td></tr><tr><td>0.2</td><td>|0.0229</td><td></td><td></td><td>0.2320</td><td>0.1594</td><td>0.0900 0.2033</td><td>|0.0335 0.1180</td><td>0.0797 0.1697</td><td>0.0916</td><td>0.2075 0.0296</td><td>0.1113</td><td></td><td>0.0266</td><td>0.1016</td><td></td></tr><tr><td>0.4</td><td>0.0337</td><td>0.0956 0.1123</td><td>0.1158 0.1166</td><td>0.0533 0.2232 0.0584</td><td>0.1632 0.1054</td><td>0.2172 0.0568</td><td>0.1482 0.1547</td><td>0.2232</td><td>0.0998 0.2123</td><td>0.0398</td><td>|0.3531 0.5075</td><td>0.4602 0.5365</td><td>0.0408 0.1225</td><td>0.0253 0.0391</td><td>0.1009 0.1220</td></tr><tr><td>0.6</td><td>0.0480</td><td>0.1362</td><td>0.2179</td><td>0.2927</td><td>0.0943 0.2020</td><td>0.1526 0.2533</td><td>0.1162 0.2097</td><td>0.2932 0.3138</td><td>0.1401 0.2434</td><td>0.0658</td><td>0.1277 0.1632 0.6978</td><td>0.6235</td><td>0.0610</td><td>0.1512 0.0634</td><td>0.1537</td></tr><tr><td>ETh1</td><td>0.1145</td><td>0.2020</td><td>0.4234</td><td>0.4143 0.2520</td><td>0.3198 0.3105</td><td>0.3537 0.3337</td><td>0.3691 0.6149</td><td>0.5109</td><td>0.3220 0.3637</td><td>0.1878</td><td>0.2694 0.9399</td><td>0.7237</td><td>0.1281 0.2128</td><td>0.1594</td><td>0.2315</td></tr><tr><td></td><td>0.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Avg |0.0548</td><td>0.1365</td><td>|0.2184</td><td>0.2906 |0.1145</td><td>0.2111</td><td>|0.1646 0.2569 |0.1351</td><td>0.2113 | 0.2856</td><td>0.3044</td><td>|0.1634 0.2567</td><td>0.0807</td><td>0.1679 |0.6246</td><td>0.5860</td><td>|0.0641 0.1470</td><td>|0.0718</td><td>0.1520</td></tr><tr><td></td><td>0.2 |0.0228</td><td>0.0771</td><td>0.0562</td><td>0.1578 0.0390</td><td>0.1269</td><td>0.0495 0.1463 0.1442</td><td>0.2685 0.1490</td><td>0.2279</td><td>0.0524 0.1542</td><td>0.0255</td><td>0.0954 0.6106</td><td>0.5741</td><td>0.0415 0.1008</td><td>0.0402</td><td>0.1149</td></tr><tr><td></td><td>0.4 0.0305</td><td>0.0924</td><td>0.0557</td><td>0.1560</td><td>0.0403 0.1286</td><td>0.0560 0.1553 0.1788</td><td>0.2950</td><td>0.2451 0.2845</td><td>0.0559 0.1577</td><td>0.0310</td><td>0.1057 0.8039</td><td>0.6763</td><td>0.0532</td><td>0.1206 0.0502</td><td>0.1313</td></tr><tr><td>ET2</td><td>0.6 0.0437 0.0797</td><td>0.1150 0.1649</td><td>0.0799</td><td>0.1871</td><td>0.0540 0.1501</td><td>0.0693 0.1744</td><td>0.3431 0.3897</td><td>0.6173 0.4458</td><td>0.0729</td><td>0.1787 0.0437</td><td>0.1283 1.4455</td><td>0.9409</td><td>0.0712</td><td>0.1475 0.0726</td><td>0.1594</td></tr><tr><td></td><td>0.8</td><td></td><td>0.1287</td><td>0.2396</td><td>0.0973 0.2043</td><td>0.1099 0.2219</td><td>1.0818 0.6869</td><td>1.6554 0.8247</td><td>0.1182</td><td>0.2291 0.0764</td><td>0.1764</td><td>2.7956 1.2937</td><td>0.1280</td><td>0.2089 0.2299</td><td>0.2728</td></tr><tr><td></td><td>Avg |0.0442</td><td>0.1123</td><td>|0.0801</td><td>0.1851</td><td>|0.0576 0.1524</td><td>|0.0712 0.1745</td><td>0.4370 0.4100 | 0.6667</td><td>0.4457</td><td>0.0749</td><td>0.1799 0.0442</td><td>0.1264</td><td>1.4139 0.8712</td><td>|0.0735</td><td>0.1444 | 0.0982</td><td>0.1696</td></tr><tr><td></td><td>0.2 |0.0022</td><td>0.0147</td><td>|0.0029</td><td>0.0364</td><td>|0.0053 0.0512</td><td>0.0016 0.0262</td><td>|0.1106 0.2863</td><td>|0.0115 0.0401</td><td>|0.0017</td><td>0.0258 0.0007</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.4 0.0015</td><td>0.0171</td><td>0.0025</td><td>0.0317</td><td>0.0075 0.0602</td><td>0.0021 0.0278</td><td>0.1480 0.3225</td><td>0.0134 0.0493</td><td>0.0021</td><td>0.0266 0.0012</td><td>0.0151 0.0174</td><td>|0.5095 0.6226 0.5551 0.6523</td><td>|0.0315 0.0382</td><td>0.0889 0.0029 0.1008 0.0021</td><td>0.0250</td></tr><tr><td>Exchane</td><td>0.6 0.0019</td><td>0.0206</td><td>0.0053</td><td>0.0453</td><td>0.0109 0.0717</td><td>0.0032 0.0328</td><td>0.2328 0.3882</td><td>0.0451 0.0951</td><td>0.0033</td><td>0.0321 0.0020</td><td>0.0217</td><td>0.6376 0.6981</td><td>0.0472</td><td>0.1147 0.0034</td><td>0.0279 0.0355</td></tr><tr><td></td><td>0.8 0.0030</td><td>0.0288</td><td>0.0105</td><td>0.0680</td><td>0.0144 0.0823</td><td>0.0064 0.0488</td><td>0.4330 0.5289</td><td>0.2070 0.2549</td><td>0.0067</td><td>0.0496 0.0038</td><td>0.0332</td><td>0.8640 0.8046</td><td>0.0772</td><td>0.1496 0.0142</td><td>0.0753</td></tr><tr><td></td><td>Avg | |0.0022</td><td>0.0203</td><td>|0.0053</td><td>0.0453 |</td><td>|0.0095 0.0664</td><td>|0.0033 0.0339</td><td>|0.2311 0.3815</td><td>|0.0693 0.1098</td><td>|0.0034</td><td>0.0335 |0.0019</td><td>0.0218</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.2 |0.0057</td><td>0.0449</td><td>|0.1392</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.6415 0.6944</td><td>|0.0485</td><td>0.1135| |0.0056</td><td>0.0409</td></tr><tr><td>IIn</td><td>0.4 0.0081</td><td>0.0518</td><td>0.0790</td><td>0.2691 0.1842</td><td>|0.0236 0.1078 0.0261 0.1079</td><td>|0.0722 0.1724 0.0860 0.1985</td><td>|0.1344 0.2193 0.1772 0.2448</td><td>|0.2115 0.2625</td><td>0.0434</td><td>0.1468 0.0079</td><td>0.0561</td><td>0.2975 0.3760</td><td>0.0394</td><td>0.1055 0.0288</td><td>0.0918</td></tr><tr><td></td><td>0.6 0.0132</td><td>0.0664</td><td>0.1781</td><td>0.2829</td><td>0.0451 0.1447</td><td>0.0965 0.2093</td><td>0.2600 0.3040</td><td>0.2690 0.3014 0.3376 0.3512</td><td>0.0394 0.0670</td><td>0.1390 0.0095 0.1818 0.0156</td><td>0.0597 0.0759</td><td>0.5426 0.4767 0.7072 0.5588</td><td>0.0506 0.0772</td><td>0.1160 0.0362 0.1525</td><td>0.1045</td></tr><tr><td></td><td>0.8 0.0673</td><td>0.1225</td><td>0.4736</td><td>0.4823</td><td>0.2394 0.3339</td><td>0.2298 0.3001</td><td>0.5777 0.4493</td><td>0.5394 0.4542</td><td>0.2377</td><td>0.3541 0.0676</td><td>0.1479</td><td>0.9947 0.6593</td><td>0.1550</td><td>0.0660 0.2178 0.2128</td><td>0.1452 0.2441</td></tr><tr><td></td><td>Avg |0.0236</td><td>0.0714</td><td>|0.2175</td><td>0.3046|</td><td>|0.0835 0.1736 | 0.1211</td><td>0.2201</td><td>|0.2873 0.3044</td><td>|0.3394 0.3423</td><td>|0.0969</td><td>0.2054 0.0252</td><td>0.0849</td><td>|0.6355 0.5177</td><td>|0.0806</td><td>0.1479| |0.0860</td><td></td></tr><tr><td></td><td>0.2 |0.0246</td><td>0.0246</td><td>0.0354</td><td>0.0745</td><td>0.0305 0.0632</td><td>0.0911 0.1436</td><td>0.0252 0.0308</td><td>0.0324 0.0355</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.1464</td></tr><tr><td>Weather</td><td>0.4 0.0271</td><td></td><td>0.0271 0.0316</td><td>0.0585</td><td>0.0302 0.0563</td><td>0.0873 0.1431</td><td>0.0272 0.0359</td><td>0.0329 0.0392</td><td>0.0351 0.0348</td><td>0.0729 0.0249 0.0685 0.0268</td><td>0.0375 0.0407</td><td>0.0806 0.1582 0.1201 0.2048</td><td>0.0269 0.0289</td><td>0.0253 0.0255 0.0277 0.0285</td><td>0.0252</td></tr><tr><td></td><td>0.6 0.0313 0.0409</td><td></td><td>0.0317 0.0449 0.0425</td><td>0.0863 0.0769 0.1334</td><td>0.0401 0.0773 0.0692 0.1228</td><td>0.0900 0.1443 0.0998 0.1497</td><td>0.0387 0.0557 0.0902 0.1363</td><td>0.0409 0.0514 0.0839 0.1114</td><td>0.0444 0.0743 0.1317</td><td>0.0880 0.0330 0.0606</td><td>0.0529</td><td>0.1859 0.2753</td><td>0.0319 0.0408</td><td>0.0317 0.0346</td><td>0.0278 0.0323</td></tr><tr><td>0.8</td></table>

TABLE IX: The standard deviation of Table VIII.
<table><tr><td rowspan="2">Models Metric</td><td colspan="2">RDDMPI (Ours)</td><td rowspan="2">MAE</td><td rowspan="2">DLinear MSE MAE</td><td rowspan="2">ModernTCN MSE MAE</td><td rowspan="2">iTransformer MSE MAE</td><td rowspan="2">SAITS MSE MAE</td><td rowspan="2">ImputeFormer MSE MAE</td><td rowspan="2">TimesNet MSE</td><td rowspan="2">T1 MAE MSE</td><td rowspan="2">GP-VAE MAE MSE</td><td rowspan="2">MAE MSE</td><td rowspan="2">CSDI MAE</td><td rowspan="2">FGTI MSE MAE</td></tr><tr><td></td><td>MSE</td></tr><tr><td></td><td>0.2 |0.0025</td><td></td><td>0.0008</td><td>0.0071 0.0047</td><td>0.0026 0.0026</td><td>0.0038 0.0023</td><td>0.0046 0.0068</td><td>0.0087 0.0080</td><td>0.0064</td><td>0.0081 0.0014</td><td>0.0017 0.0086</td><td>0.0057 0.0022</td><td>0.0031</td><td>0.0017 0.0017</td></tr><tr><td>0.4</td><td>0.0028</td><td>0.0015</td><td>0.0068</td><td>0.0039 0.0035</td><td>0.0028 0.0033</td><td>0.0032 0.0137</td><td>0.0125 0.0184</td><td>0.0116 0.0028</td><td>0.0036</td><td>0.0024 0.0017</td><td>0.0130 0.0065</td><td>0.0058 0.0044</td><td>0.0045</td><td>0.0035</td></tr><tr><td>ETh1</td><td>0.6 0.0023</td><td>0.0015</td><td>0.0167</td><td>0.0065 0.0070</td><td>0.0045 0.0112</td><td>0.0046 0.0318</td><td>0.0241 0.0421</td><td>0.0281</td><td>0.0133 0.0051</td><td>0.0036 0.0023</td><td>0.0119 0.0057</td><td>0.0034</td><td>0.0040 0.0058</td><td>0.0048</td></tr><tr><td></td><td>0.8 0.0081</td><td>0.0051</td><td>0.0141</td><td>0.0071 0.0156</td><td>0.0089 0.0195</td><td>0.0064 0.0501</td><td>0.0367 0.0613</td><td>0.0457 0.0206</td><td>0.0120</td><td>0.0074 0.0066</td><td>0.0136 0.0093</td><td>0.0218 0.0106</td><td>0.0171</td><td>0.0096</td></tr><tr><td></td><td>Avg |0.0039</td><td>0.0022</td><td>|0.0111</td><td>0.0056 |0.0072</td><td>0.0047 0.0095</td><td>0.0041 |0.0251</td><td>0.0200 |0.0326</td><td>0.0233</td><td>0.0108 0.0072</td><td>0.0037 0.0031</td><td>|0.0118 0.0068</td><td>|0.0083</td><td>0.0055 |0.0073</td><td>0.0049</td></tr><tr><td></td><td></td><td>0.0034</td><td>0.0030</td><td>|0.0022</td><td>|0.0027</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.2 |0.0012 0.4 0.0022</td><td>0.0030</td><td>0.0014</td><td>0.0024 0.0013 0.0003</td><td>0.0027 0.0006</td><td>0.0023 |0.0071 0.0005 0.0163</td><td>0.0078 |0.0126 0.0152 0.0429</td><td>0.0056 0.0221</td><td>|0.0029 0.0026 0.0011</td><td>|0.0012 0.0015</td><td>0.0730 0.0339</td><td>|0.0052 0.0063 0.0085 0.0084</td><td>|0.0037</td><td>0.0054</td></tr><tr><td>ET2</td><td>0.6 0.0044</td><td>0.0049</td><td>0.0010</td><td>0.0016</td><td>0.0008 0.0018</td><td>0.0003 0.0016 0.0019 0.0937</td><td>0.0546 0.1761</td><td>0.0650</td><td>0.0011 0.0014 0.0028</td><td>0.0012 0.0011 0.0015 0.0018</td><td>0.0784 0.0389 0.1417 0.0374</td><td>0.0112 0.0093</td><td>0.0016 0.0035</td><td>0.0029 0.0027</td></tr><tr><td></td><td>0.8 0.0081</td><td>0.0079</td><td>0.0035</td><td>0.0024</td><td>0.0041 0.0034</td><td>0.0035 0.0028 0.2958</td><td>0.1241 0.3109</td><td>0.0934</td><td>0.0021 0.0020</td><td>0.0041 0.0037</td><td>0.0974 0.0177</td><td>0.0285</td><td>0.0192 0.0792</td><td>0.0425</td></tr><tr><td></td><td>| Avg |0.0040</td><td></td><td>0.0048 |0.0022</td><td>0.0019</td><td>|0.0019 0.0021</td><td>|0.0020 0.0019 |0.1032</td><td>0.0504 0.1356</td><td></td><td></td><td></td><td>0.0976</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0465</td><td>|0.0019 0.0021</td><td>|0.0020 0.0020</td><td>0.0320</td><td>0.0133</td><td>0.0108 | 0.0220</td><td>0.0134</td></tr><tr><td></td><td>0.2 |0.0019 0.4</td><td>0.0003</td><td>0.0002</td><td>0.0005</td><td>0.0004 0.0010</td><td>0.0002 0.0002 0.0154</td><td>0.0221 0.0084</td><td>0.0035</td><td>0.0002 0.0009</td><td>0.0002 0.0004</td><td>0.0400 0.0259</td><td>0.0383</td><td>0.0789 0.0019</td><td>0.0009</td></tr><tr><td></td><td>0.0008 0.6 0.0005</td><td>0.0005 0.0003</td><td>0.0008 0.0006</td><td>0.0003 0.0009</td><td>0.0008 0.0005 0.0006 0.0005</td><td>0.0009 0.0003 0.0056</td><td>0.0082 0.0054</td><td>0.0117</td><td>0.0007 0.0007</td><td>0.0008 0.0003</td><td>0.0300 0.0197</td><td>0.0548</td><td>0.0970 0.0011</td><td>0.0030</td></tr><tr><td>Exchange</td><td>0.8 0.0001</td><td>0.0002</td><td>0.0004</td><td>0.0007</td><td>0.0003 0.0005</td><td>0.0006 0.0006 0.0106 0.0004 0.0011 0.0229</td><td>0.0130 0.0417 0.0210 0.2059</td><td>0.0532 0.1700</td><td>0.0004 0.0007 0.0005</td><td>0.0006 0.0004</td><td>0.0373 0.0208 0.0497</td><td>0.0698</td><td>0.1167 0.0006</td><td>0.0052</td></tr><tr><td></td><td>|Avg |0.0008</td><td>0.0003</td><td>|0.0005</td><td></td><td></td><td></td><td></td><td></td><td>0.0010</td><td>0.0006 0.0012</td><td>0.0251</td><td>0.1148</td><td>0.1518 0.0059</td><td>0.0203</td></tr><tr><td rowspan="5">Il</td><td></td><td></td><td></td><td>0.0006</td><td>|0.0005 0.0007</td><td>|0.0005 0.0006 |0.0136</td><td>0.0161 |0.0653</td><td>0.0596 0.0005</td><td>0.0008</td><td>0.0005 0.0006</td><td>0.0392 0.0229</td><td>|0.0694 0.1111</td><td>0.0024</td><td>0.0073</td></tr><tr><td>0.2 |0.0007</td><td></td><td>0.0029</td><td>|0.0159 0.0107</td><td>|0.0043 0.0120</td><td>|0.0200 0.0249</td><td>|0.0327 0.0314</td><td>|0.0447 0.0276</td><td>|0.0052</td><td>0.0132 |0.0018</td><td>0.0061 |0.0642</td><td>0.0333 |0.0075</td><td>0.0088</td><td>|0.0045 0.0104</td></tr><tr><td>0.4 0.0014</td><td>0.0028</td><td>0.0098</td><td>0.0091</td><td>0.0031 0.0047</td><td>0.0077 0.0049 0.0385</td><td>0.0121 0.0768</td><td>0.0367</td><td>0.0050 0.0073</td><td>0.0013 0.0016</td><td>0.0805 0.0268</td><td>0.0090</td><td>0.0083 0.0088</td><td>0.0117</td></tr><tr><td>0.6 0.0020 0.8</td><td>0.0389</td><td>0.0036 0.0138</td><td>0.0514 0.0329</td><td>0.0089 0.0134</td><td>0.0228 0.0181 0.0445</td><td>0.0182 0.0384</td><td>0.0161</td><td>0.0135 0.0200</td><td>0.0022 0.0028</td><td>0.0734 0.0267</td><td>0.0153</td><td>0.0127 0.0145</td><td>0.0169</td></tr><tr><td>| Avg | 0.0108</td><td></td><td>0.0858</td><td>0.0268</td><td>0.0644 0.0334</td><td>0.0728 0.0316 0.1379</td><td>0.0425 0.1117</td><td>0.0419</td><td>0.0426 0.0230</td><td>0.0387 0.0271</td><td>0.0599 0.0184</td><td>0.0290 0.0131</td><td>0.0891</td><td>0.0312</td></tr><tr><td rowspan="4"></td><td></td><td>0.0058</td><td>|0.0407</td><td>0.0199</td><td>0.0202 0.0159|</td><td>|0.0308 0.0199 |0.0634</td><td>0.0260 0.0679</td><td>0.0306</td><td>0.0166 0.0159</td><td>|0.0110 0.0094 </td><td>|0.0695</td><td>0.0263 | 0.0152</td><td>0.0107 |0.0292</td><td>0.0175</td></tr><tr><td>0.2</td><td>|0.0018</td><td>0.0003</td><td>0.0026 0.0013</td><td>|0.0010 0.0055</td><td>0.0040 0.0013</td><td>|0.0022 0.0012 0.0039</td><td>0.0034</td><td>|0.0014 0.0037</td><td>|0.0017 0.0025</td><td>0.0079 0.0096</td><td>|0.0023</td><td>0.0006 0.0014</td><td>0.0022</td></tr><tr><td>0.4 0.0020</td><td></td><td>0.0006</td><td>0.0007 0.0004</td><td>0.0006 0.0012</td><td>0.0018 0.0005 0.0016</td><td>0.0018 0.0039</td><td>0.0036</td><td>0.0011 0.0026</td><td>0.0022 0.0032</td><td>0.0081 0.0087</td><td>0.0022 0.0010</td><td>0.0027</td><td>0.0028</td></tr><tr><td>0.6</td><td>0.0020</td><td>0.0014</td><td>0.0015 0.0003</td><td>0.0025 0.0040 0.0020</td><td>0.0019 0.0002</td><td>0.0022 0.0033 0.0004</td><td>0.0047</td><td>0.0005 0.0028</td><td>0.0027 0.0058</td><td>0.0101 0.0109</td><td>0.0024</td><td>0.0012 0.0027</td><td>0.0032</td></tr><tr><td>Weather</td><td>0.8 0.0014</td><td>0.0022</td><td>0.0004</td><td>0.0005</td><td>0.0046</td><td>0.0042 0.0013 0.0079</td><td>0.0116 0.0161</td><td>0.0219</td><td>0.0011 0.0015</td><td>0.0043 0.0106</td><td>0.0258 0.0244</td><td>0.0007 0.0014</td><td>0.0056</td><td>0.0048</td></tr><tr><td></td><td>Avg 0.0018</td><td></td><td>0.0011</td><td>0.00130.0006</td><td>0.0015 0.0038</td><td>|0.0030 0.0008 |0.0035</td><td>0.0045</td><td>|0.0061 0.0084</td><td>0.0010 0.0026</td><td>0.0027</td><td>0.0055 |0.01300.0134|</td><td>|0.0019 0.0010|</td></table>

TABLE X: Full results under the block missing scenario across datasets. Best results are marked in bold, and second-best results are marked in underlined.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">RDDMPI (Ours) MAE</td><td rowspan="2">DLinear MAE</td><td rowspan="2">ModernTCN MSE MAE</td><td rowspan="2">iTransformer MSE MAE</td><td rowspan="2">SAITS MSE</td><td rowspan="2">MAE MSE</td><td rowspan="2">ImputeFormer MAE</td><td rowspan="2">TimesNet MSE MAE</td><td rowspan="2">MSE</td><td rowspan="2">T1 MAE</td><td rowspan="2">GP-VAE MSE MAE</td><td rowspan="2">MSE</td><td rowspan="2">CSDI MAE</td><td rowspan="2">FGTI MSE MAE</td></tr><tr><td>MSE</td><td>MSE</td></tr><tr><td>ETTh1</td><td>0.0168</td><td>0.0851</td><td></td><td>|0.0592</td><td></td><td></td><td>0.1075</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.1728 0.2844</td><td>0.1735</td><td>0.1016 0.2097</td><td>0.0257</td><td></td><td>0.0594 0.1543</td><td>|0.0920 0.2117</td><td>0.0258</td><td>0.1077</td><td>0.2842</td><td>0.4231 0.0179</td><td>0.0896</td><td>0.0170 0.0877</td></tr><tr><td>ETTh2 Exchange</td><td>0.0203 0.0043</td><td>0.0744 0.0181</td><td>0.0770 0.1890 0.0063 0.0557</td><td>0.0477 0.1414 0.0054 0.0498</td><td>0.0546 0.1552 0.0034 0.0336</td><td>0.1471 0.1864</td><td>0.2711 0.3330</td><td>0.2654 0.2672 0.1217 0.1233</td><td>0.0533 0.1592 0.0036 0.0362</td><td>0.0292 0.0115</td><td>0.1014 0.0312</td><td>0.6001 0.5100</td><td>0.5663 0.0561 0.6290</td><td>0.1070 0.1458</td><td>0.0948 0.1321</td></tr><tr><td>Illness</td><td>0.0390</td><td>0.1282</td><td>0.2603 0.3691</td><td>0.1521 0.2647</td><td>0.3345 0.3598</td><td>0.1475</td><td>0.2373</td><td>0.2697 0.3100</td><td>0.1368 0.2535</td><td>0.0874</td><td>0.1792</td><td>0.3270</td><td>0.2237 0.4122 0.0977</td><td>0.1895</td><td>0.0177 0.0457</td></tr><tr><td>Weather</td><td>0.0233</td><td>0.0254</td><td>0.0495 0.1053</td><td>0.0371 0.0827</td><td>0.1003 0.1480</td><td>0.0251</td><td>0.0328</td><td>0.0401 0.0461</td><td>0.0390 0.0855</td><td>0.0248</td><td>0.0400</td><td>0.0627</td><td>0.1349 0.0234</td><td>0.0254</td><td>0.0516 0.1283 0.0236 0.0256</td></tr></table>

TABLE XI: Standard deviations corresponding to the block missing results reported in Table X.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">RDDMPI (Ours) MSE</td><td rowspan="2">DLinear MSE MAE</td><td rowspan="2">ModernTCN MSE</td><td rowspan="2">MAE MSE</td><td rowspan="2">iTransformer MAE</td><td rowspan="2">SAITS MSE</td><td rowspan="2">MAE MSE</td><td rowspan="2">ImputeFormer MAE</td><td rowspan="2">TimesNet MSE MAE</td><td rowspan="2">MSE</td><td rowspan="2">T1 MAE</td><td rowspan="2">MSE</td><td rowspan="2">GP-VAE MAE</td><td rowspan="2">CSDI MSE</td><td rowspan="2">MAE</td><td rowspan="2">FGTI MSE MAE</td></tr><tr><td>MAE</td></tr><tr><td>ETTh1</td><td>0.0020</td><td>0.0025</td><td>0.0050 0.0024</td><td>0.0066</td><td></td><td>0.0129</td><td>0.0061</td><td></td><td></td><td>0.0085</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0020</td></tr><tr><td>ETTh2</td><td>0.0024</td><td>0.0043</td><td>0.0099 0.0143</td><td>0.0060 0.0117 0.0163</td><td>0.0345 0.0102</td><td></td><td>0.0100 0.0148</td><td>0.0155 0.1646</td><td>0.0130 0.0641</td><td>0.0077</td><td>0.0051</td><td>0.0049</td><td>|0.0114</td><td>0.0067</td><td>0.0022</td><td>0.0046</td><td>0.0041</td></tr><tr><td>Exchange</td><td>0.0074</td><td>0.0091</td><td>0.0027</td><td></td><td>0.0018</td><td>0.0132 0.0168 0.0049 0.0852</td><td></td><td>0.1218</td><td></td><td>0.0070 0.0109</td><td>0.0078 0.0192</td><td>0.0129 0.0204</td><td>0.0693</td><td>0.0356</td><td>0.0324 0.3921</td><td>0.0202</td><td>0.1091 0.0341</td></tr><tr><td>Illness</td><td>0.0384</td><td>0.0675</td><td>0.0012 0.1112 0.0515</td><td>0.0020 0.0060 0.0985 0.0867</td><td>0.2518</td><td></td><td>0.0602 0.0893</td><td>0.1174</td><td>0.0673</td><td>0.0014 0.0042 0.1194</td><td></td><td></td><td>0.0608</td><td>0.0364</td><td></td><td>0.1583</td><td>0.0322 0.0363</td></tr><tr><td>Weather</td><td>0.0041</td><td>0.0024</td><td>0.0062 0.0041</td><td>0.0050 0.0040</td><td>0.0140</td><td>0.1288 0.0663 0.0059 0.0061</td><td>0.0040</td><td>0.0023</td><td>0.0934 0.0025</td><td>0.1058 0.0047 0.0020</td><td>0.0885 0.0052</td><td>0.1076 0.0031</td><td>0.1543 0.0084</td><td>0.1103 0.0102</td><td>0.0984 0.0051</td><td>0.1226 0.0026</td><td>0.0719 0.1089 0.0032 0.0039</td></tr></table>

TABLE XII: Full CRPS comparison of RDDMPI, GP-VAE, CSDI, and FGTI under point missing and block missing scenarios across datasets. For point missingness, results are reported at missing ratios 0.2, 0.4, 0.6, and 0.8. Best results are marked in bold, and second-best results are marked in underlined. Lower is better.  
(a) Point missing
<table><tr><td colspan="2">Models Metric</td><td>RDDMPI CRPS</td><td>GP-VAE CRPS</td><td>CSDI CRPS</td><td>FGTI CRPS</td></tr><tr><td rowspan="4">ET1</td><td>0.2 0.4</td><td>0.0924 0.1080</td><td>0.5829 0.6739</td><td>0.0970 0.1165</td><td>0.0966 0.1161</td></tr><tr><td>0.6</td><td>0.1310</td><td>0.7856</td><td>0.1446</td><td>0.1469</td></tr><tr><td>0.8</td><td>0.1940</td><td>0.9080</td><td>0.2044</td><td>0.2222</td></tr><tr><td>Avg</td><td>0.1313</td><td>0.7376</td><td>0.1406</td><td>0.1454</td></tr><tr><td rowspan="4">ET2</td><td>0.2</td><td>0.0438</td><td>0.4202</td><td>0.0570</td><td>0.0657</td></tr><tr><td>0.4</td><td>0.0528</td><td>0.4967</td><td>0.0682</td><td>0.0756</td></tr><tr><td>0.6</td><td>0.0657</td><td>0.6921</td><td>0.0848</td><td>0.0928</td></tr><tr><td>0.8</td><td>0.0941</td><td>0.9488</td><td>0.1203</td><td>0.1576</td></tr><tr><td rowspan="4">Exchange</td><td>Avg</td><td>0.0641</td><td>0.6395</td><td>0.0826</td><td>0.0979</td></tr><tr><td rowspan="3">0.2 0.4 0.6</td><td>0.0131</td><td>0.7039</td><td>0.0766</td><td>0.0216</td></tr><tr><td>0.0151</td><td>0.7360</td><td>0.0869</td><td>0.0239</td></tr><tr><td>0.0181</td><td>0.7887</td><td>0.0998</td><td>0.0304</td></tr><tr><td rowspan="4"></td><td>0.8 Avg</td><td>0.0252</td><td>0.9106</td><td>0.1328</td><td>0.0658</td></tr><tr><td></td><td>0.0179</td><td>0.7848</td><td>0.0990</td><td>0.0354</td></tr><tr><td rowspan="2">0.2 0.4 0.6</td><td>0.0460</td><td>0.4830</td><td>0.1111</td><td>0.0917</td></tr><tr><td>0.0526</td><td>0.6118</td><td>0.1209 0.1617</td><td>0.1026</td></tr><tr><td rowspan="4">Ilnns</td><td rowspan="2">0.8</td><td>0.0680 0.1248</td><td>0.7244 0.8479</td><td>0.2296</td><td>0.1412 0.2427</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Avg 0.2</td><td>0.0728</td><td>0.6668</td><td>0.1558</td><td>0.1445</td></tr><tr><td rowspan="4">Weather</td><td rowspan="2">0.4 0.6</td><td>0.0327 0.0362</td><td>0.2764 0.3575</td><td>0.0335 0.0368</td><td>0.0365 0.0400</td></tr><tr><td>0.0423</td><td>0.4809</td><td>0.0421</td><td>0.0468</td></tr><tr><td rowspan="2">0.8 Avg</td><td>0.0573</td><td>0.6855</td><td>0.0555</td><td>0.0626</td></tr><tr><td>0.0420</td><td>0.4501</td><td>0.0421</td><td>0.0465</td></tr></table>

(b) Block missing
<table><tr><td>Dataset</td><td>RDDMPI</td><td>GP-VAE</td><td>CSDI</td><td>FGTI</td></tr><tr><td>ETTh1</td><td>0.0811</td><td>0.5298</td><td>0.0867</td><td>0.0832</td></tr><tr><td>ETTh2</td><td>0.0412</td><td>0.4076</td><td>0.0596</td><td>0.0751</td></tr><tr><td>Exchange</td><td>0.0153</td><td>0.6903</td><td>0.1238</td><td>0.0391</td></tr><tr><td>Illness</td><td>0.1238</td><td>0.4793</td><td>0.1873</td><td>0.1173</td></tr><tr><td>Weather</td><td>0.0338</td><td>0.2383</td><td>0.0336</td><td>0.0348</td></tr></table>

TABLE XIII: Standard deviations of the CRPS results reported in Table XII.
<table><tr><td colspan="2">Models Metric</td><td>RDDMPI CRPS</td><td>GP-VAE CRPS</td><td>CSDI CRPS</td><td>FGTI CRPS</td></tr><tr><td rowspan="3">ET1 ET2</td><td>0.2 0.4</td><td>0.0012 0.0017 0.0017</td><td>0.0073 0.0050 0.0056</td><td>0.0023 0.0046 0.0045</td><td>0.0021 0.0031 0.0044</td></tr><tr><td>0.6 0.8 Avg</td><td>0.0064 0.0027</td><td>0.0098</td><td>0.0103</td><td>0.0091</td></tr><tr><td>0.2 0.4 0.6</td><td>0.0022 0.0019 0.0027</td><td>0.0069 0.0261 0.0269</td><td>0.0054 0.0037 0.0044</td><td>0.0047 0.0030 0.0019</td></tr><tr><td>Excne</td><td>0.8 Avg 0.2 0.4</td><td>0.0050 0.0030 0.0004</td><td>0.0264 0.0134 0.0232 0.0253</td><td>0.0061 0.0116 0.0065 0.0652</td><td>0.0010 0.0229 0.0072 0.0011</td></tr><tr><td></td><td>0.6 0.8 Avg 0.2</td><td>0.0005 0.0003 0.0002 0.0004 0.0038</td><td>0.0220 0.0220 0.0298 0.0248</td><td>0.0798 0.0944 0.1247 0.0910</td><td>0.0025 0.0043 0.0182 0.0065</td></tr><tr><td>Iness</td><td>0.4 0.6 0.8 Avg</td><td>0.0028 0.0030 0.0149 0.0061</td><td>0.0058 0.0250 0.0206 0.0195 0.0177</td><td>0.0099 0.0077 0.0094 0.0155 0.0106</td><td>0.0079 0.0128 0.0171 0.0380 0.0190</td></tr><tr><td>Weather</td><td>0.2 0.4 0.6 0.8 Avg</td><td>0.0005 0.0008 0.0020 0.0034 0.0017</td><td>0.0165 0.0152 0.0186 0.0421 0.0231</td><td>0.0008 0.0012 0.0017 0.0020 0.0014</td><td>0.0032 0.0039 0.0043 0.0066 0.0045</td></tr></table>

(a) Point missing

(b) Block missing
<table><tr><td>Dataset</td><td>RDDMPI</td><td>GP-VAE</td><td>CSDI</td><td>FGTI</td></tr><tr><td>ETTh1</td><td>0.0015</td><td>0.0080</td><td>0.0038</td><td>0.0027</td></tr><tr><td>ETTh2</td><td>0.0021</td><td>0.0326</td><td>0.0107</td><td>0.0185</td></tr><tr><td>Exchange</td><td>0.0070</td><td>0.0188</td><td>0.1331</td><td>0.0320</td></tr><tr><td>Illness</td><td>0.0546</td><td>0.0783</td><td>0.1111</td><td>0.0969</td></tr><tr><td>Weather</td><td>0.0032</td><td>0.0194</td><td>0.0029</td><td>0.0058</td></tr></table>

## APPENDIX H

## ADDITIONAL QUALITATIVE RESULTS

We provide additional qualitative imputation visualizations under varying point wise missing ratios (20%, 40%, 60%, and 80%) and block missingness for ETTh2. These figures illustrate median predictions and 90% predictive intervals, highlighting reconstruction accuracy and uncertainty calibration.

![](images/e2d6f19d8acbd01068b5501dddd59bd4cb774a10e13ecf10b8c96a3ad5d64f28.jpg)

![](images/f91566d7d1dff99fab6067c1446229745c1b89a85bf9241963e640fdc6191380.jpg)

![](images/b91f54faa741b0c52793690294b735b79629990e0d701b5dd04a5ec8a4f1ab37.jpg)

![](images/70e0eb4d63e08bddf5c5fe7e9f86090204396374d84fe758b9c5917d5ba6221f.jpg)

![](images/b7b45a81b6983e95a8cad3a8ce5dc5e885799ad2f49c684481eaa1f35add03a1.jpg)

![](images/480367e9baa75bb5a4d232aeb62113c0575e764350884a88284c9ede598cca61.jpg)

![](images/b1d7665c2392047041dc9ee7245a8533f2aefa82529cf9e593a818ccabb3435c.jpg)  
Fig. 4: Visualization of probabilistic imputation results on ETTh2 (20% missingness). The results are for a time series sample with all 7 features. The median and 90% predictive intervals are shown in green, observed points in red and targets in blue.

![](images/ecc86f218c603a95c7d21e68680304a102fe4888550d39bee16619792a1747c4.jpg)

![](images/382b8ddfc2896b9a085103c15c8913e5eb176ffd98ac03ad708e8e1dabcb7a47.jpg)

![](images/061b5cbf752dc7610f6f108b3f59833b276a3563ff95ca458cc54961d77def7a.jpg)

![](images/cb73ea9b0b3460366b682c32020cb13d568943ba3a2067220e352c2c1917e363.jpg)

![](images/d855d8f95045063beb9be2df9daf06fab187bb1e16cae58193602f01d8e7a8e1.jpg)

![](images/df09a6ffc49f747d4e2ef5368dd6a71957adff3da0a9293887df2ecb24950c7d.jpg)

![](images/9efce899d6e2e7f94474c42b812a4c73f0a60f1cf0b17d4d83feb65d856ac305.jpg)  
Fig. 5: Visualization of probabilistic imputation results on ETTh2 (40% missingness). The results are for a time series sample with all 7 features. The median and 90% predictive intervals are shown in green, observed points in red and targets in blue.

![](images/d1ec7cacf73bf49a98b4f8cb280a8f5779bdc519d669128703f6a71e1f76c702.jpg)

![](images/499bb94448b5e20de969baad63124349d7840be1eba0e825e5638e6a642b9e93.jpg)

![](images/71398c67bfe7c6ade0d1437257907c1b93709ee82818658638262e2a22abe4bf.jpg)

![](images/f1f61f3342bd3d019879b19ac8b4e647c7b6df1b8dceda77d1324f7f3851f27e.jpg)

![](images/ad065753142d80eb8ef2d45873e0535ac491d24243651fb91fa22e5b130fe58e.jpg)

![](images/2085dced1022a5c5910f3745c3837dc262b74ff533de4eb8ea3c967446d601e8.jpg)

![](images/c72b88ce6905cca195df481c10eab03f791b0f56da8151b2e1bbf7eb825785a4.jpg)  
Fig. 6: Visualization of probabilistic imputation results on ETTh2 (60% missingness). The results are for a time series sample with all 7 features. The median and 90% predictive intervals are shown in green, observed points in red and targets in blue.

![](images/554c7f0e8b13f2651abe2650235e51d716d06e01f356df944b531488a94acf22.jpg)

![](images/145f5d3f87c89cb5050e784daec7073ec3a5859abaa346c98fb08d60a9eb4986.jpg)

![](images/3136e413cd97ad847a1a66130771fa29914b4771c170f59f675efb8b4124b15e.jpg)

![](images/0e9f0964d9caf43e98bc3c4ca986a0cd1c2499f4f49d90dc2fdd6758359548cb.jpg)

![](images/429269adfc5cd1e96cd4b325a18b2ba450e27285f5e71bf5c029f1ed14143b6e.jpg)

![](images/d33e3a58851e85a02832f3271740c85178799260079e5950d24be175763c5f13.jpg)

![](images/1dcbc414ee42cd37e2cfd8129df29741c023c991b4a74df2e8407bb5d92b969b.jpg)  
Fig. 7: Visualization of probabilistic imputation results on ETTh2 (80% missingness). The results are for a time series sample with all 7 features. The median and 90% predictive intervals are shown in green, observed points in red and targets in blue.

![](images/6af74e8f229272c525b7c2f64c3c2990140b1d48caa567b03d7f9631103890db.jpg)

![](images/742d60b100ccaeaecca6c0992134e2dd5c4a97ef2d2293ed5d896e7aa6de1eb1.jpg)

![](images/d402e57cd7a98009f7fa3115e41093641e5c69618309240c3b4f2f9c2b30962b.jpg)

![](images/bd6b7c1f9f8f4933ab9d7634b601605279b654e6dd02ce57bca1f401a9ab748f.jpg)

![](images/a4947d01572ead58720d69e97a22c08a5cf80ed27d444c126b7fcc1e7877a97e.jpg)

![](images/2bbed6a6fe03acfa77e7e5cf7e60818d82298bed9c1930a3bc0c45ae97da1998.jpg)

![](images/63f91c1884def7e5df776e21a67d07dee8d49fbfb21fae59742ecd2a73516197.jpg)  
Fig. 8: Visualization of probabilistic imputation results on ETTh2 (block missingness). The results are for a time series sample with all 7 features. The median and 90% predictive intervals are shown in green, observed points in red and targets in blue.