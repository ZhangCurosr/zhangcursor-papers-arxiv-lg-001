# DynG-Dif: A State-Aware Dynamic Guidance Difusion Framework for Probabilistic Time Series Forecasting

Zhente Zhang<sup>1</sup>, Zhengwei Ni<sup>1,\*</sup>, and Wei Fan<sup>2</sup>

<sup>1</sup>School of Information and Electronic Engineering (Sussex Artificial Intelligence Institute),

Zhejiang Gongshang University, Hangzhou, Zhejiang, China

<sup>2</sup>School of Computer Science, University of Auckland, Auckland, New Zealand

Corresponding author: zhengwei.ni@zjgsu.edu.cn

zhangzhente@163.com; wei.fan@auckland.ac.nz

## Abstract

Probabilistic multivariate time series (MTS) forecasting is crucial for modeling complex dynamical systems. However, existing difusion-based methods rely on task-specific conditional paradigms that lack flexibility and struggle with inherent "information heterogeneity"—the significantly varying noise levels and evolutionary patterns across variables. To address this, we propose DynG-Dif, a variable-sensitive dynamic guidance difusion framework for probabilistic multivariate time-series forecasting: (1) DynG-Dif adopts a two-stage separated training strategy and uses an unconditional difusion backbone to model the joint distribution of multivariate time series. (2) DynG-Dif introduces a lightweight state-aware policy network that adaptively infers variable reliability from real-time noisy states and one-step denoising estimates, outputting a dynamic guidance strength matrix. (3) DynG-Dif mathematically formulates this dynamic weight as the local precision of the observation distribution, enabling precise guidance for high-confidence variables during inference while filtering out interference from anomalous noise. Extensive experiments on real-world benchmarks demonstrate competitive probabilistic forecasting performance against state-of-the-art conditional difusion models and improved robustness under severe observation corruption. The implementation code is available at: https://github.com/TT-20011031/DynG-Diff

Keywords: Multivariate time series; probabilistic forecasting; difusion; dynamic guidance; state-aware network; information heterogeneity

## 1 Introduction

Probabilistic forecasting of multivariate time series (MTS) is the foundation for modeling complex dynamical systems such as energy grids [1], transportation [2], finance [3], and healthcare [4]. A core challenge in this domain lies in capturing the intricate heterogeneity [5] and coupling [6, 7] within high-dimensional data: diferent variables often exhibit distinctly diferent physical characteristics, noise levels, and distribution shifts, yet their temporal evolutions are highly interdependent. Although traditional deterministic methods are efective at capturing trend components, they often fall short in characterizing the inherent stochasticity and multimodal distributions of such systems [8]. This limitation has driven a paradigm shift toward generative probabilistic models [9–11], which aim to fundamentally learn the underlying joint data distribution. However, efectively modeling these distributions requires not only capturing global dependencies but also respecting the varying information densities of individual variables—a requirement that poses a significant challenge to existing generative frameworks.

Given the limitations of traditional methods, difusion models [12, 13] have rapidly become the framework of choice for generative modeling, owing to their progressive noise removal mechanisms and exceptional distribution-fitting capabilities. Consequently, conditional difusion models have emerged as a prominent approach. Existing research primarily translates forecasting into a conditional generation problem through two strategies: the first is an end-to-end conditioning strategy, which directly embeds historical observations as conditions into the denoising network to guide generation [14–16]; the second is a divide-and-conquer strategy, which utilizes deterministic models to capture trends and then relies on difusion models to model the residual distribution [17, 18]. Although these methods have achieved notable results on specific benchmarks, they are generally constrained by a "task-specific" training paradigm—meaning the model must be specifically trained for a particular prediction horizon or task objective. Once the scenario changes, the model must be retrained, lacking the flexibility expected of generative models. Meanwhile, the few existing studies on unconditional generation primarily focus on the image [19, 20] and audio [21] domains, or are limited to univariate time series [22], leading to two critical scientific gaps that urgently need addressing. First, the unconditional generative potential of difusion models in the MTS domain is severely underestimated. Existing unconditional frameworks have failed to fully demonstrate their capability as "universal generative priors," where a single task-agnostic foundation model can be flexibly adapted to diverse downstream tasks through posterior guidance during inference, without requiring parameter fine-tuning for each task. Second, existing guidance mechanisms struggle to address the "information heterogeneity" challenge in multivariate systems. Unlike univariate sequences, diferent channels in real-world multivariate systems often exhibit significantly diferent statistical properties, noise levels, and physical correlation strengths. When using observational data to guide the generation process, the model must possess the ability to distinguish between "primary and secondary" variables and their "signal-to-noise ratios." That is, it must intelligently apply precise guidance to high-confidence key variables while tolerating the randomness of high-noise variables. However, designing a "variable-sensitive" adaptive mechanism to dynamically quantify and utilize this heterogeneous information in the absence of explicit labels remains the key to improving the forecasting accuracy of difusion models in complex MTS systems.

To overcome these challenges, this paper proposes DynG-Dif (Dynamic Guidance Difusion), a variable-aware probabilistic forecasting framework for multivariate time series. DynG-Dif maintains the difusion backbone network in an unconditional generation mode and designs a lightweight State-Aware Policy Network as a plug-in. Rather than relying on manually set hyperparameters, this network adaptively infers the "reliability" of each variable at the current timestep based on the real-time noisy state and one-step estimation during the difusion process, outputting a Dynamic Guidance Strength Matrix. Mathematically, we interpret this dynamic weight as a plug-in local precision of the observation distribution, yielding a weighted observation-likelihood guidance objective. This design enables the model to intelligently apply precise guidance to high-confidence key variables during inference while automatically reducing the weight interference from highly noisy or anomalous variables. Notably, the policy network is optimized independently while the unconditional backbone remains frozen. This separation allows the guidance mechanism to be adapted without jointly retraining the difusion backbone.

To address the limitations of existing time series forecasting models in handling complex data distributions and quantifying uncertainty, we propose DynG-Dif, a Dynamic-Guidance Difusion Model designed specifically for probabilistic time series forecasting. The main contributions of this paper are summarized as follows:

• We propose DynG-Dif, a decoupled probabilistic forecasting framework that explicitly addresses information heterogeneity by combining an unconditional difusion backbone with state-dependent, variable-specific guidance during inference. This separation enables the framework to selectively exploit reliable observations without jointly retraining the difusion backbone.

• We develop a self-supervised State-Aware Policy Network that estimates variable reliability from the current noisy state and its one-step denoising estimate, producing a dynamic guidance matrix for each variable and difusion timestep. We further interpret this matrix as a plug-in local precision of an Asymmetric Laplace Distribution and derive a weighted observation-likelihood objective, providing a probabilistic motivation for adaptive guidance.

• We conduct extensive experiments on real-world benchmark datasets across various domains. The results demonstrate that DynG-Dif achieves highly competitive performance in both forecasting accuracy and probabilistic metrics. Furthermore, comprehensive ablation studies and visualization analyses validate the robustness of the proposed architecture and its substantial potential for practical applications.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 covers the problem formulation and preliminaries. Section 4 details our proposed DynG-Dif framework. Section 5 evaluates the model through extensive experiments, and Section 6 concludes the paper.

## 2 Related Work

Time Series Generation. In recent years, deep generative models have shown immense potential in handling data generation tasks across various domains, and time series generation, being one of the most challenging tasks in the generative field, has also received widespread attention. In early explorations, Generative Adversarial Networks (GANs) played a dominant role. Mogren et al. [23] pioneered C-RNN-GAN, innovatively integrating Long Short-Term Memory (LSTM) networks into the generator and discriminator of GANs to process continuous sequential data. Esteban et al. [24] introduced a conditioning mechanism and developed the Recurrent Conditional Generative Adversarial Network (RCGAN), successfully achieving the synthesis of multidimensional real-valued time series using auxiliary label information. Yoon et al. [25] proposed TimeGAN, a framework that explicitly constrains the temporal dynamics of data within a jointly optimized latent space, thereby generating higher-fidelity samples. Paul et al. [26] proposed PSA-GAN, which significantly improves the synthesis quality of long multivariate time series by incorporating a progressive growing strategy and self-attention mechanisms.

Due to the inherent instability of adversarial training, researchers began to explore other types of deep generative paradigms. For instance, TimeVAE, proposed by Desai et al. [27], implemented an interpretable temporal structure and achieved initial success in using VAEs for multivariate time series synthesis. HyVAE, proposed by Cai et al. [28], integrated the joint learning of local patterns (e.g., seasonality and trend) and temporal dynamics of time series into a unified framework via variational inference. Wu et al. [29] combined Koopman theory and Kalman filtering, utilizing KoopmanNet to transform nonlinear time series dynamics into a biased linear dynamical system, and employing KalmanNet to refine predictions and model uncertainties within this linear system. Additionally, energy-based models have been used to mimic the sequential behavior of time series through progressively decomposable structures. Alaa et al. [30] proposed Fourier Flows based on normalizing flows and a series of spectral filters to achieve exact likelihood optimization.

Time Series Difusion Models. Denoising Difusion Probabilistic Models (DDPMs), as a novel class of generative models, have been widely applied to time series forecasting and imputation tasks. The pioneering work in this area was introduced by Rasul et al. [31], whose designed TimeGrad combines autoregressive models with the denoising difusion process to achieve multivariate probabilistic forecasting. Subsequently, CSDI, proposed by Tashiro et al. [14], constructed a score-based conditional difusion framework that unified time series imputation and forecasting tasks through self-attention and a specialized masking strategy. As research deepened, to overcome the computational bottleneck of long sequence modeling, SSSD [15] replaced the attention mechanism in traditional difusion architectures with structured state space models (S4). TimeDif, proposed by Shen et al. [32], introduced two novel conditioning mechanisms: future mixup and autoregressive initialization, which significantly improved the forecasting quality of long sequences while efectively capturing complex temporal dynamics. For example, Yuan and Qiao [33] achieved highly interpretable time series generation by decoupling trend and seasonal priors. However, these methods generally couple joint-distribution learning with task-specific conditioning. In contrast, DynG-Dif combines unconditional pre-training with variable-aware inference guidance to address information heterogeneity in probabilistic multivariate time-series forecasting.

Difusion Guidance. Classifier Guidance adds the gradient of an auxiliary classifier to the unconditional score during sampling [19]. Classifier-Free Guidance (CFG) removes the auxiliary classifier by jointly learning conditional and unconditional score estimates and blending them with a user-defined guidance scale [20]. Guidance has also been used for text-driven image generation and editing [34, 35], while constraint-based guidance expresses desired time-series properties as diferentiable energy functions [36]. For time series, TSDif conditions an unconditionally trained difusion model through observation self-guidance during inference, without an auxiliary guidance network or changes to backbone training [22]. Its observation-likelihood gradient is regulated by a globally shared scale. Feedback Guidance makes the guidance coeficient state- and time-dependent by feeding back the model’s estimate of conditional-signal informativeness [37]. However, it adapts guidance at the sample-trajectory level rather than estimating variable-wise observation reliability.

DynG-Dif is rooted in TSDif’s observation-likelihood formulation: it retains the unconditional backbone, the one-step estimate ${ \hat { x } } ^ { 0 }$ , and the inference-time likelihood gradient. It extends this formulation with a separately trained policy network, a self-supervised reliability target, and a variable- and timestep-specific matrix $A ^ { t }$ . Unlike CFG and Feedback Guidance, DynG-Dif does not interpolate conditional and unconditional score estimates; unlike TSDif, it does not apply one shared guidance strength to all observed variables.

## 3 Preliminaries

## 3.1 Problem statement

This paper investigates the problem of probabilistic multivariate time series forecasting. Given a multivariate time series $y \in \mathbb { R } ^ { L \times D }$ of length L containing D variables. For the forecasting task, we introduce a binary observation mask $M \in \{ 0 , 1 \} ^ { L \times D }$ with the same dimensions as y. The mask M indicates the observation status of the data points: $M _ { l , d } = 1$ denotes that the data at this position is a known historical observation, denoted as $y _ { o b s } ;$ whereas $M _ { l , d } = 0$ indicates that the data is an unknown future value to be predicted, denoted as $y _ { t a r g e t }$ . The known observations yobs and the unknown targets ytarget can be expressed as:

$$
y _ { o b s } = M \odot y , \quad y _ { t a r g e t } = ( 1 - M ) \odot y\tag{1}
$$

where ⊙ denotes the element-wise product. Unlike deterministic forecasting, which solely aims to find a mapping $f : y _ { o b s }  \hat { y } _ { t a r g e t }$ to minimize the point error, the goal of probabilistic forecasting is to learn the conditional probability distribution of the target values given the observations, $p _ { \theta } ( y _ { t a r g e t } | y _ { o b s } )$ . During practical inference, the model approximates this distribution by generating a set of samples $\{ \hat { y } ^ { s } \} _ { s = 1 } ^ { S } \sim p _ { \theta } ( \cdot | y _ { o b s } )$ . Consequently, it not only provides the predictive mean but also quantifies the predictive uncertainty through the statistical properties of the samples, efectively capturing the stochasticity and coupling relationships of the multivariate system as it evolves over time.

## 3.2 Denoising Difusion Probabilistic Models

Difusion models [12, 13] are a class of generative models based on non-equilibrium thermodynamics; their core idea encompasses two processes: forward noise addition and reverse denoising. The normalized multivariate time series y is treated as the noise-free initial state of the diffusion process, explicitly denoted as $\boldsymbol { x } ^ { 0 } \in \mathbb { R } ^ { L \times D }$ . During the difusion process, $\boldsymbol { x } ^ { t } \in \mathbb { R } ^ { L \times D }$ is defined as the latent state variable at an arbitrary difusion time step $t \in [ 1 , T ]$ . The forward process is a fixed Markov chain, where the single-step transition probability is parameterized as $\begin{array} { r } { q ( x ^ { t } | x ^ { t - 1 } ) = \mathcal N ( x ^ { t } ; \sqrt { 1 - \beta _ { t } } x ^ { t - 1 } , \beta _ { t } I ) } \end{array}$ , and $\beta _ { t } \in ( 0 , 1 )$ is the variance schedule parameter. Utiliz ing the reparameterization trick, the latent variable at any arbitrary timestep t can be sampled directly from the initial data $x ^ { 0 } \colon x ^ { t } = \sqrt { \bar { \alpha } _ { t } } x ^ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ , where $\begin{array} { r } { \alpha _ { t } = 1 - \beta _ { t } , \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ , and $\epsilon \sim \mathcal { N } ( 0 , I )$ is standard Gaussian noise.

The reverse process aims to learn the conditional probability distribution to reverse the noise addition process. Since the true reverse transition distribution is intractable, Ho et al. [13] approximate the noise that needs to be removed at each step by training a neural network $\epsilon _ { \theta } ( x ^ { t } , t )$ , whose training objective employs a simplified mean squared error (MSE) loss function:

$$
\mathcal { L } _ { s i m p l e } = \mathbb { E } _ { x ^ { 0 } , \epsilon , t } [ | | \epsilon - \epsilon _ { \theta } ( x ^ { t } , t ) | | ^ { 2 } ]\tag{2}
$$

Notably, based on the noise $\epsilon _ { \theta } ( x ^ { t } , t )$ predicted by the model and the reparameterized $x ^ { t }$ , we can derive the estimation of the original data at the current step:

$$
\hat { x } ^ { 0 } = \frac { x ^ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( x ^ { t } , t ) } { \sqrt { \bar { \alpha } _ { t } } }\tag{3}
$$

This capability of real-time estimation of the original data $x ^ { 0 }$ during the generation process serves as the critical foundation for realizing observation-based dynamic guidance mechanisms. For more details, please refer to Appendix A.

## 3.3 Observation Guidance

Unconditional difusion models learn the joint distribution of the data $p ( x )$ . In time series forecasting tasks, it is necessary to sample from the conditional distribution $p ( x | y _ { o b s } )$ [22]. According to Bayes’ theorem, the conditional score function can be decomposed into the sum of an unconditional score term and an observation likelihood guidance term [19, 22]:

$$
\nabla _ { \boldsymbol { x } ^ { t } } \log p ( \boldsymbol { x } ^ { t } | \boldsymbol { y } _ { o b s } ) = \underbrace { \nabla _ { \boldsymbol { x } ^ { t } } \log p ( \boldsymbol { x } ^ { t } ) } _ { \mathrm { U n c o n d . ~ S c o r e } } + \underbrace { \nabla _ { \boldsymbol { x } ^ { t } } \log p ( \boldsymbol { y } _ { o b s } | \boldsymbol { x } ^ { t } ) } _ { \mathrm { O b s e r v a t i o n ~ G u i d a n c e } }\tag{4}
$$

Here, the first term is provided by the pre-trained denoising network $\epsilon _ { \theta } ( x ^ { t } , t )$ , while the second term measures the consistency between the currently generated latent variable $x ^ { t }$ and the historical observations $y _ { o b s }$ . Since directly computing $p ( y _ { o b s } | x ^ { t } )$ is intractable, we can utilize the original data estimation ${ \hat { x } } ^ { 0 }$ obtained from Eq. (3) to approximate the true $x ^ { 0 }$ , thereby calculating the observation guidance loss $\mathcal { L } _ { g u i d e }$ . We then compute its gradient and inject it into the predicted noise. The modified noise prediction can be expressed as:

$$
\hat { \epsilon } = \epsilon _ { \theta } ( x ^ { t } , t ) - s \cdot \sqrt { 1 - \bar { \alpha } _ { t } } \cdot \nabla _ { x ^ { t } } \mathcal { L } _ { g u i d e } \big ( \hat { x } ^ { 0 } , y _ { o b s } \big )\tag{5}
$$

Here, $\mathcal { L } _ { g u i d e }$ depends on the assumed observation noise distribution, and s is a global guidance scale shared by all observed variables. Equation (5) therefore represents homogeneous observation self-guidance and serves as the starting point of DynG-Dif. Our method retains this posteriorscore formulation but replaces homogeneous loss weighting with state-aware local precision.

## 4 DynG-Dif: Variable-Sensitive Dynamic Guidance Difusion

In this section, we present DynG-Dif, a novel probabilistic forecasting framework for multivariate time series, which aims to address the inherent information heterogeneity challenge in complex dynamical systems by combining an unconditional difusion backbone with variable-sensitive adaptive guidance. As illustrated in Figure 1, the overall architecture of DynG-Dif is designed into three stages: the unconditional backbone network pre-training stage, the state-aware policy learning stage, and the dynamic guidance inference stage. This decoupled paradigm preserves the backbone’s capability to fit joint distributions while enabling fine-grained interventions for diferent variables during inference.

Relationship to Existing Guidance. DynG-Dif is not a variant of CFG because it neither trains a conditional denoising branch nor blends conditional and unconditional score estimates. Its direct methodological starting point is the homogeneous observation self-guidance in Eq. (5).

Beyond this starting point, DynG-Dif adds four components: the State-Aware Policy Network $g _ { \phi }$ , a self-supervised reliability target derived from denoising error, the variable- and timestepspecific guidance matrix $A ^ { t }$ , and an ALD-based weighted observation-likelihood objective. These additions transform a shared scalar into fine-grained local precision while preserving the unconditional backbone.

In the subsequent sections, we detail the core components and theoretical derivations of the DynG-Dif framework. First, serving as the generative foundation of the entire framework, the unconditional backbone network is independently optimized during the pre-training stage using a standard denoising difusion objective to fit the underlying joint probability distribution of the multivariate time series. Once pre-training is complete, the frozen backbone network not only provides a powerful unconditional generative prior for subsequent inference sampling, but its real-time one-step denoising estimation output during intermediate difusion steps also serves as the core data source for the subsequent policy network to quantify the system’s recovery state and variable reliability. For more details on the pre-training of the unconditional backbone network, please refer to Appendix B.1. Subsequently, in Section 4.1, we elaborate on the design motivation and specific architecture of the state-aware policy network, explaining how it utilizes the real-time noisy state and one-step denoising estimation during the difusion process to quantify the reliability of diferent variables and generate the dynamic weight matrix. Finally, in Section 4.2, we introduce the variable-sensitive guidance mechanism, proving from a probabilistic perspective how dynamic weights influence the model’s tolerance to observation errors, and we derive the guidance loss based on the weighted observation likelihood in detail.

## 4.1 State-Aware Dynamic Weight Generation

Real-world complex multivariate systems often exhibit significant information heterogeneity, with diferent variables frequently displaying vastly diferent dynamic evolution patterns. Based on this, we propose a state-aware dynamic weight generation mechanism aimed at adaptively allocating fine-grained guidance weights for diferent variables.

Policy Network Design. To endow the model with this dynamic perception capability, we introduce an auxiliary lightweight policy network alongside the unconditional difusion backbone. Its overall architecture and learning process are illustrated in Figure 2. The core objective of this network is to accurately represent the recovery state of the system at the current difusion timestep and subsequently infer the relative reliability of each variable. Since the true noise-free historical data $x ^ { 0 }$ is unobservable during the inference stage, the model must quantify the current generation quality based solely on intermediate states. Given the progressive denoising nature of difusion models, the current noisy state $x ^ { t }$ and the one-step denoising estimation $\hat { x } ^ { 0 } ( x ^ { t } , t )$ output by the backbone network (derived from Eq. (3)) contain rich local evolution information. Therefore, we transpose both tensors and concatenate them along the channel dimension to construct the composite state representation of the current system, $\mathcal { S } ^ { t } = [ ( \boldsymbol { x } ^ { t } ) ^ { T } \mathrm { ~ } | | \mathbf { \eta } ( \hat { \boldsymbol { x } } ^ { 0 } ) ^ { T } ] \in \mathbb { R } ^ { 2 D \times L }$ This state representation strategy not only preserves the current true noise distribution of the observation sequence but also fully integrates the unconditional prior knowledge already learned by the backbone model. Meanwhile, to equip the policy network with time-awareness, we map the discrete difusion timestep t into a continuous high-dimensional vector via sinusoidal positional encoding, and extract the time embedding feature $e ^ { t } \in \mathbb { R } ^ { d _ { t } }$ through a Multi-Layer Perceptron (MLP).

![](images/ff661316732672e9d94a14e3b4f50c4769483b19f986496d33d3ce9dd8cb36eb.jpg)  
Figure 1: Overall architecture of DynG-Dif. The framework comprises three core stages: (1) Unconditional Backbone Pre-training Stage: the backbone denoising network is trained using the standard difusion loss $\mathcal { L } _ { s i m p l e }$ to capture the underlying joint probability distribution of the multivariate time series $\boldsymbol { x } ^ { 0 } \in \mathbb { R } ^ { L \times D }$ (2) State-Aware Policy Learning Stage: the pre-trained denoising network $\epsilon _ { \theta }$ is frozen. Its one-step estimate ${ \hat { x } } ^ { 0 }$ is concatenated with the current noisy state $x ^ { t }$ and fed into the policy network $g _ { \phi }$ . Using the reconstructed inverse error as a proxy target, the policy network learns and outputs a dynamic guidance strength matrix $A ^ { t } \in \mathbb { R } ^ { L \times D }$ tailored to each difusion timestep and variable. (3) Inference with Dynamic Guidance Stage: at each reverse difusion step, the frozen policy network $g _ { \phi }$ recomputes $A ^ { t }$ . Its detached copy $\bar { A } ^ { t } = \mathrm { s g } ( A ^ { t } )$ modulates the observation-guidance gradient in a variable-wise manner.

Regarding the network architecture design, considering the stringent requirements of time series forecasting on inference speed and computational overhead, the policy network employs a series of lightweight one-dimensional convolutional (1D-CNN) layers as the core feature extractor. The input state $S ^ { t }$ first passes through non-linear convolutional layers to extract cross-variable local correlations and temporal features, generating a high-dimensional hidden state representation. Subsequently, the time embedding $e ^ { t }$ is deeply fused with the hidden state along the feature dimension and fed into a Weight Prediction Head composed of activation functions and 1D convolutions. Assuming the network’s hidden variable output before the final layer mapping is $H ^ { t } \in \mathbb { R } ^ { D \times L }$ (where D is the variable dimension and L is the sequence length), to ensure that the generated weights reasonably represent the relative "state confidence" among variables and to prevent the introduction of dynamic weights from shifting the global gradient scale, we adopt a weight generation strategy based on exponential mapping and mean normalization. Specifically, the hidden variable is first mapped to a relative score matrix via a

1D convolutional layer:

$$
Z ^ { t } = W _ { h e a d } * H ^ { t } + b _ { h e a d }\tag{6}
$$

where $W _ { h e a d }$ and $b _ { h e a d }$ are the weights and biases of the prediction head’s convolutional layer, respectively. Subsequently, an exponential operation is applied to ensure the non-negativity of the weights, followed by normalization by dividing by the global mean:

$$
A ^ { t } = \frac { \exp ( Z ^ { t } ) } { \frac { 1 } { D \cdot L } \sum _ { i = 1 } ^ { D } \sum _ { j = 1 } ^ { L } \exp ( ( Z ^ { t } ) _ { i , j } ) + \epsilon }\tag{7}
$$

where ϵ is a tiny constant to prevent division by zero, and $A ^ { t }$ is the final generated non-negative guidance strength matrix, with a global mean approximately equal to 1. For its probabilistic interpretation, we introduce the following modeling definition:

Definition 1 (Dynamic Local Precision). Any element $a _ { i , j } ^ { t }$ in the dynamic guidance strength matrix $A ^ { t }$ is interpreted as a plug-in estimate of the reciprocal scale of an Asymmetric Laplace Distribution (ALD), i.e., the local precision. It represents the model’s inverse error tolerance for variable i at time point $j$ and difusion timestep t.

Based on Definition 1, the generated weight matrix $A ^ { t }$ serves not merely as an attention mask for feature selection, but possesses explicit physical and probabilistic interpretability.

Proxy Target Construction. To enable the policy network to accurately learn the latent local precision described in Definition 1, we design an uncertainty-aware self-supervised training objective. Since the absolute "true reliability" labels for each variable at every intermediate timestep are inaccessible during the difusion process, we transform the residual between the backbone model’s one-step denoising estimation $\hat { x } ^ { 0 } ( x ^ { t } , t )$ and the true noise-free signal $x ^ { 0 }$ into local precision, constructing a proxy supervision signal based on this.

Specifically, we first compute the squared error between the denoising estimation and the true signal, mapping it to log-precision:

$$
P _ { r a w } ^ { t } = - \log ( ( \hat { x } ^ { 0 } ( x ^ { t } , t ) - x ^ { 0 } ) ^ { \odot 2 } + \epsilon )\tag{8}
$$

where ϵ is a tiny constant to prevent numerical underflow. Because the absolute magnitude of errors spans vastly across diferent difusion stages, directly fitting the absolute precision would cause severe gradient instability in the network. Therefore, we apply Z-Score normalization to $P _ { r a w } ^ { t }$ across spatial and variable dimensions, followed by threshold clipping, to construct a smooth relative confidence target $Y _ { t a r g e t } ^ { t } \mathrm { : }$

$$
Y _ { t a r g e t } ^ { t } = C l i p \left( \frac { P _ { r a w } ^ { t } - \mu _ { P } } { \sigma _ { P } } , - 3 , 3 \right)\tag{9}
$$

where $\mu { } _ { P }$ and $\sigma _ { P }$ are the mean and standard deviation of the log-precision within the current batch, respectively. Finally, the policy network $g _ { \phi }$ fits this standardized relative confidence target using MSE as the loss function:

$$
\mathcal { L } _ { p o l i c y } = \mathbb { E } _ { t , x ^ { 0 } , \epsilon } \left[ \frac { 1 } { D \cdot L } \sum _ { i = 1 } ^ { D } \sum _ { j = 1 } ^ { L } \| ( Z ^ { t } ) _ { i , j } - ( Y _ { t a r g e t } ^ { t } ) _ { i , j } \| _ { 2 } ^ { 2 } \right]\tag{10}
$$

Under this optimization objective, the policy network’s output $Z ^ { t }$ represents the relative log-precision of the recovery state for each variable at the current timestep, which is then processed through Eq. (7) to generate the non-negative dynamic guidance strength matrix $A ^ { t }$ The generated guidance strength matrix $A ^ { t }$ possesses clear physical significance and probabilistic interpretability. Higher weights in the matrix indicate that the current generation state of the corresponding variable is highly reliable and exhibits clear trend features; thus, strong gradient guidance should be applied in subsequent sampling to force it into strict alignment with the conditional distribution. Conversely, lower weight values imply that the current variable is still in a noise-dominated state with high uncertainty. The network will grant it greater tolerance, allowing the difusion backbone model to freely explore relying on the unconditional generative prior, thereby avoiding adversarial gradients caused by enforcing strong guidance on noisy estimations. After obtaining this crucial dynamic weight matrix $A ^ { t } { \mathrm { . } }$ subsequent sections will detail how to seamlessly integrate it into the Bayes’ theorem-based difusion sampling to drive diferentiated gradient updates.

![](images/07ddd9392bb01f583362787f3770107373b517669043ac8aa7cac29200e36554.jpg)  
Figure 2: Schematic of the State-Aware Policy Network architecture. The network takes the concatenated features of the current noisy state $x ^ { t }$ and the one-step estimate $\hat { x } ^ { 0 }$ as input, combines them with the timestep embeddi $^ { 1 9 } \mathrm { , }$ and outputs the dynamic weight matrix $A ^ { t }$ . During training, the difusion backbone remains frozen. The model computes the log-precision from the squared error between ${ \hat { x } } ^ { 0 }$ and the ground truth $x ^ { 0 }$ and, after $\mathrm { Z }$ -score normalization, constructs a smooth proxy target $Y _ { t a r g e t } ^ { t }$ to update the policy network parameters through the loss $\mathcal { L } _ { p o l i c y }$

## 4.2 Variable-Sensitive Guidance Mechanism

Recent advances in the field of time series generation [22] indicate that by introducing observation likelihood guidance during inference, unconditional difusion models can flexibly adapt to various downstream forecasting and imputation tasks without explicit conditional training. However, existing self-guiding difusion frameworks are either limited to modeling univariate time series or, when directly generalized to multivariate dynamical systems, ignore the distinctly diferent dynamic evolution patterns and noise levels inherent in diferent variables at the same timestep. If gradient guidance of equal strength is applied to all channels, the model is highly prone to producing adversarial gradients on variables containing extreme anomalies or high-noise disturbances, thereby severely degrading the generation quality of the overall multivariate joint distribution. To this end, we propose a variable-sensitive self-guidance mechanism. This mechanism integrates the dynamic guidance strength matrix $A ^ { t }$ output by the policy network into the observation-likelihood gradient, providing a local probabilistic interpretation for fine-grained

sampling in multivariate systems.

Conditional Probability Modeling. In probabilistic time series forecasting tasks, the model is required not only to provide point estimates but also to accurately delineate predictive uncertainty. Considering that widely used evaluation metrics in this domain (such as the Continuous Ranked Probability Score, CRPS) heavily rely on the quantile loss, we adopt the ALD to model the conditional observation distribution $p ( y _ { o b s } | x ^ { t } )$ given the latent variable $x ^ { t }$ For theoretical details of ALD modeling, please refer to Appendix B.2.

At each reverse difusion step, the policy network recomputes $A ^ { t }$ from the current state. For likelihood-gradient evaluation, we use $\bar { A } ^ { t } = \mathrm { s g } ( A ^ { t } )$ , where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. Thus, $\nabla _ { x ^ { t } } { \bar { A } } ^ { t } = 0$ . The matrix remains dynamic across reverse steps but is treated as fixed only during diferentiation at the current step.

For a single data point $( l , d )$ in a multivariate sequence, the one-step estimation of the original data provided by the backbone network at the current timestep t can be obtained from Eq. (3). Under this estimation, the conditional probability density function of the true observation $y _ { l , d }$ can be defined as:

$$
p ( \boldsymbol { y } _ { l , d } | \boldsymbol { x } ^ { t } ; \kappa , \bar { a } _ { l , d } ^ { t } ) \propto \exp ( - \bar { a } _ { l , d } ^ { t } \cdot \rho _ { \kappa } ( \boldsymbol { y } _ { l , d } - \hat { \boldsymbol { x } } _ { l , d } ^ { 0 } ) )\tag{11}
$$

where $\kappa \in ( 0 , 1 )$ is the specified quantile level, $\rho _ { \kappa } ( e ) = \operatorname* { m a x } ( \kappa \cdot e , ( \kappa - 1 ) \cdot e )$ is the asymmetric quantile loss function, and $\bar { a } _ { l , d } ^ { t }$ is the corresponding stop-gradient element of $\bar { A } ^ { t }$ . We interpret $\bar { a } _ { l , d } ^ { t }$ as a plug-in local precision, i.e., the reciprocal of the scale parameter of the Laplace distribution. A large weight produces a sharper local likelihood and stronger alignment with the observation. A small weight produces a flatter likelihood, allowing uncertain variables to rely more on the unconditional prior.

During inference, trajectory s uses $\kappa _ { s } = s / ( S + 1 ) , s = 1 , \dots , S$ . The $S$ trajectories form an equally weighted κ-conditioned ensemble rather than exact conditional quantiles.

Joint Conditional Probability and Guidance Loss. To achieve global guidance in multivariate systems, we assume that given the local estimations, the individual observed variables are conditionally independent. Thus, the joint conditional probability density of the multivariate time series can be expressed as the product of the individual univariate densities:

$$
p ( y _ { o b s } | x ^ { t } ; \bar { A } ^ { t } ) = \prod _ { ( l , d ) : M _ { l , d } = 1 } p ( y _ { l , d } | x ^ { t } ; \kappa , \bar { a } _ { l , d } ^ { t } )\tag{12}
$$

where $M \in \{ 0 , 1 \} ^ { L \times D }$ is the binary observation mask defined in Section 3.1. The conditional independence assumption does not remove cross-variable information from the guidance weights. Before detachment, $A ^ { t }$ is inferred through cross-channel feature fusion with a global receptive field, so each plug-in precision can encode system-wide coupling and relative reliability.

Taking the negative logarithm of the joint conditional probability density converts the product into a summation. Discarding constant terms irrelevant to $x ^ { t }$ , we derive the core energy function of the guided difusion process—the Weighted Quantile Guidance Loss (WQ-Loss):

$$
\begin{array} { l l } { \displaystyle - \log p ( y _ { o b s } | x ^ { t } ; \bar { A } ^ { t } ) = \mathcal { L } _ { g u i d e } ( \hat { x } ^ { 0 } , y _ { o b s } ; \bar { A } ^ { t } ) + C ( \bar { A } ^ { t } , \kappa ) , } \\ { \displaystyle \mathcal { L } _ { g u i d e } ( \hat { x } ^ { 0 } , y _ { o b s } ; \bar { A } ^ { t } ) = \sum _ { l = 1 } ^ { L } \sum _ { d = 1 } ^ { D } M _ { l , d } \cdot \bar { a } _ { l , d } ^ { t } \cdot \rho _ { \kappa } ( y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } ) } \end{array}\tag{13}
$$

Here, $C ( { \bar { A } } ^ { t } , \kappa )$ collects terms constant with respect to $x ^ { t }$ during the current update. The matrix $\bar { A } ^ { t }$ acts as an adaptive weighting mask, and Eq. (5) uses the resulting local likelihood gradient. The following theorem gives this gradient under the stated stop-gradient convention.

Theorem 1 (Variable-Sensitive Guidance Gradient). Given $y _ { o b s }$ and $M \in \{ 0 , 1 \} ^ { L \times D }$ , assume the local observation probability follows an ALD with quantile level $\kappa \in ( 0 , 1 )$ . For the fixed

current-step matrix $\bar { A } ^ { t }$ , satisfying $\nabla _ { x ^ { t } } { \bar { A } } ^ { t } = 0$ , the gradient of the Weighted Quantile Guidance Loss is given by:

$$
\begin{array} { l } { \displaystyle \nabla _ { x ^ { t } } \mathcal { L } _ { g u i d e } ( \hat { x } ^ { 0 } , y _ { o b s } ; \bar { A } ^ { t } ) = \sum _ { l = 1 } ^ { L } \sum _ { d = 1 } ^ { D } M _ { l , d } \cdot \bar { a } _ { l , d } ^ { t } } \\ { \displaystyle \cdot ( I ( y _ { l , d } < \hat { x } _ { l , d } ^ { 0 } ) - \kappa ) \cdot \nabla _ { x ^ { t } } \hat { x } _ { l , d } ^ { 0 } } \end{array}\tag{14}
$$

where I(·) denotes the indicator function.

Proof. Based on the definition of the weighted quantile guidance loss in Eq. (13), the objective function is essentially the weighted sum of asymmetric quantile losses $\rho _ { \kappa } ( e )$ for each data point. For a single data point $( l , d )$ , let the residual error be $e = y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } .$

First, the subgradient of the asymmetric quantile loss function $\rho _ { \kappa } ( e )$ with respect to the residual e can be equivalently expressed using the indicator function I(·) as:

$$
\frac { \partial \rho _ { \kappa } ( e ) } { \partial e } = \kappa - I ( e < 0 )\tag{15}
$$

Second, according to the backpropagation mechanism, we apply the chain rule to compute the gradient of this local loss with respect to the latent state $x ^ { t } { } _ { ; }$

$$
\nabla _ { x ^ { t } } \rho _ { \kappa } ( y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } ) = \frac { \partial \rho _ { \kappa } ( e ) } { \partial e } \cdot \nabla _ { x ^ { t } } ( y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } )\tag{16}
$$

Since the true observation $y _ { l , d }$ is a given constant, its gradient with respect to $x ^ { t }$ is $\nabla _ { x ^ { t } } y _ { l , d } = 0$ Substituting this yields:

$$
\begin{array} { r } { \nabla _ { x ^ { t } } \rho _ { \kappa } ( y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } ) = ( \kappa - I ( y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } < 0 ) ) \cdot ( - \nabla _ { x ^ { t } } \hat { x } _ { l , d } ^ { 0 } ) } \\ { = ( I ( y _ { l , d } < \hat { x } _ { l , d } ^ { 0 } ) - \kappa ) \cdot \nabla _ { x ^ { t } } \hat { x } _ { l , d } ^ { 0 } } \end{array}\tag{17}
$$

Finally, substituting this expression into Eq. (13) and summing over positions and variables, with the fixed current-step weights $\bar { a } _ { l , d } ^ { t } .$ , completes the proof. □

As derived in Theorem 1, the quantile error signal $( I ( \cdot ) - \kappa )$ determines the direction and quantile bias of the gradient. The fixed current-step weight $\bar { a } _ { l , d } ^ { t }$ scales this signal, while $\nabla _ { x ^ { t } } \hat { x } _ { l , d } ^ { 0 }$ maps it back to the latent sampling direction. Substitution into Eq. (5) yields variable-sensitive guidance without jointly retraining the unconditional backbone.

Relation to Homogeneous Guidance. When $\bar { A } ^ { t } = \mathbf { 1 }$ , Eq. (13) reduces to a homogeneous quantile observation loss. Substitution into Eq. (5) recovers observation self-guidance with one globally shared scale s. Thus, homogeneous observation guidance is a special case of DynG-Dif, whereas the recomputed $A ^ { t }$ generalizes it to variable-, position-, and difusion-timestep-specific regulation.

## 5 Experiments

## 5.1 Experimental setup

Datasets. We selected six widely used, publicly available real-world multivariate time series benchmark datasets covering complex dynamical systems across diverse domains, including energy, economics, meteorology, and transportation. Specifically, these include: ETTh1<sup>1</sup>,

Table 1: Detailed information of the datasets. The split sizes denote the total number of time steps in the training, validation, and test sets, respectively.
<table><tr><td>Dataset</td><td>Dimension</td><td>Frequency</td><td>Split size</td><td>Field</td></tr><tr><td>ETTh1</td><td>7</td><td>Hourly</td><td>(8361, 2091, 6968)</td><td>Energy</td></tr><tr><td>Exchange</td><td>8</td><td>Daily</td><td>(4248, 849, 2276)</td><td>Finance</td></tr><tr><td>Weather</td><td>21</td><td>10min</td><td>(29509, 7377, 15809)</td><td>Weather</td></tr><tr><td>Appliance</td><td>28</td><td>10min</td><td>(11051, 2763, 5920)</td><td>Energy</td></tr><tr><td>Solar-Energy</td><td>137</td><td>10min</td><td>(33638, 8410, 10512)</td><td>Energy</td></tr><tr><td>Traffic</td><td>862</td><td>Hourly</td><td>(9824, 2456, 5263)</td><td>Traffic</td></tr></table>

Exchange<sup>2</sup>, Weather<sup>3</sup>, Appliance<sup>4</sup>, Solar<sup>5</sup>, and Trafic<sup>6</sup>. The detailed attributes of these datasets are summarized in Table 1. To thoroughly evaluate the model’s forecasting capability and stability across diferent time spans, we set both the historical context length H and the prediction horizon length L within the range of {96, 168, 336, 720} for all datasets.

Baselines. To evaluate the proposed framework and the efectiveness of the dynamic guidance mechanism, we compared DynG-Dif against six representative difusion-based probabilistic time series forecasting methods, including TimeGrad [31], CSDI [14], SSSD [15], TimeDif [32], TMDM [38], and D3U [17].

Evaluation Metrics. We use the Continuous Ranked Probability Score (CRPS) and Mean Squared Error (MSE). The S trajectories $x ^ { ( s ) }$ define the empirical predictive distribution $\begin{array} { r } { \hat { F } _ { S } = S ^ { - 1 } \sum _ { s = 1 } ^ { S } \delta _ { x ^ { ( s ) } } } \end{array}$ and point forecast $\begin{array} { r } { \hat { y } = S ^ { - 1 } \sum _ { s = 1 } ^ { S } x ^ { ( s ) } } \end{array}$

$$
M S E = \frac { 1 } { N \times D } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { D } ( y _ { i , j } - \hat { y } _ { i , j } ) ^ { 2 }\tag{18}
$$

$$
C R P S ( \hat { F } _ { S } , y ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } | x ^ { ( s ) } - y | - \frac { 1 } { 2 S ^ { 2 } } \sum _ { s = 1 } ^ { S } \sum _ { r = 1 } ^ { S } | x ^ { ( s ) } - x ^ { ( r ) } |\tag{19}
$$

where $y _ { i , j }$ and $\hat { y } _ { i , j }$ are the observation and point forecast at time step i and variable $j .$ CRPS is evaluated from the equally weighted empirical ensemble and averaged across all forecast locations and variables.

Implementation Details. DynG-Dif is optimized via a two-stage separated training strategy. The unconditional difusion backbone and the state-aware policy network are trained for 300 and 30 epochs, respectively, with a learning rate of 0.001. The forward difusion employs 100 timesteps, a time embedding dimension of 128, and a linear noise schedule $( \beta _ { 1 } = 1 0 ^ { - 4 }$ to $\beta _ { 1 0 0 } = 1 0 ^ { - 1 } )$ . During inference, we generate S = 100 trajectories per prediction horizon and assign $\kappa _ { s } = s / ( S + 1 )$ to trajectory s. The resulting ensemble defines $\hat { F } _ { S }$ . All experiments are conducted on a single NVIDIA V100 32GB GPU. The core hyperparameters are summarized in Table 2.

Due to varying variable dimensions and GPU memory constraints across datasets, we adaptively adjust specific configurations (Table 3). Specifically, the global guidance scale is fine-tuned according to the dataset’s signal-to-noise ratio to optimally balance the unconditional prior and observation guidance. Low-dimensional datasets (e.g., ETTh1, Exchange) utilize a lightweight backbone (hidden dimension 64, 3 S4 blocks), whereas high-dimensional datasets (e.g., Solar, Trafic) require higher capacity.

Table 2: Hyperparameters of DynG-Dif.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone epochs</td><td>300</td></tr><tr><td>Policy epochs</td><td>30</td></tr><tr><td>Learning rate Time embedding dim</td><td>0.001</td></tr><tr><td></td><td>128</td></tr><tr><td>Diffusion steps</td><td>100</td></tr><tr><td> $\beta$  scheduler</td><td>Linear 0.0001</td></tr><tr><td> $\beta _ { 1 }$ </td><td></td></tr><tr><td> $\beta _ { 1 0 0 }$ </td><td>0.1</td></tr></table>

Table 3: Specific configurations for diferent datasets. L denotes the prediction horizon length.
<table><tr><td>Dataset</td><td>Horizon (L)</td><td>Guidance scale</td><td>Samples</td><td>Hidden dim</td><td>S4 blocks</td><td>Batch size</td></tr><tr><td>ETTh1</td><td>All</td><td>3</td><td>100</td><td>64</td><td>3</td><td>128</td></tr><tr><td>Exchange</td><td>All</td><td>4</td><td>100</td><td>64</td><td>3</td><td>256</td></tr><tr><td>Weather</td><td>All</td><td>2</td><td>100</td><td>128</td><td>6</td><td>32</td></tr><tr><td>Appliance</td><td>All</td><td>2</td><td>100</td><td>128</td><td>6</td><td>64</td></tr><tr><td>Solar</td><td>96  $\geq 1 6 8$ </td><td>5</td><td>50</td><td>128</td><td>6</td><td>32 16</td></tr><tr><td>Traffic</td><td> $\leq 3 3 6$  720</td><td>1</td><td>100 70</td><td>512</td><td>3</td><td>8 4</td></tr></table>

## 5.2 Main results

In this section, we conduct a comprehensive quantitative comparison between the proposed DynG-Dif framework and representative state-of-the-art (SOTA) multivariate time series probabilistic forecasting methods in terms of point forecasting (MSE) and probabilistic forecasting (CRPS) metrics. Table 4 demonstrates the highly competitive performance of DynG-Dif across six real-world benchmark datasets. Particularly on the ETTh1 and Appliance datasets (as visualized in Figures 3 and 4), which feature complex heterogeneity, DynG-Dif achieves average CRPS values of 0.321 and 0.395, respectively. These results are significantly superior to both the strong difusion-based baseline TMDM (0.454 and 0.574, respectively) and the D3U model employing the latest decoupled architecture (0.425 and 0.563, respectively).

On Weather, DynG-Dif achieves the lowest average CRPS (0.190), whereas its average MSE (0.502) is higher than those of D3U and TMDM. This divergence indicates a stronger advantage in distributional forecasting than in mean-point estimation. Figure 6 shows localized precision drops across Weather variables and timesteps. These heterogeneous recovery patterns may afect the sample mean more strongly than the predictive distribution, consistent with the MSE–CRPS gap. On Solar, DynG-Dif ranks second across all horizons. The shared diurnal and illumination-driven dynamics produce the vertical block patterns in Figure C.2d, favoring methods that explicitly extract regular deterministic trends. This regularity narrows the advantage of variable-specific guidance, while DynG-Dif remains consistently competitive across all horizons.

Conditional difusion forecasting methods like TMDM tend to entrust the overall data distribution—encompassing trends, seasonality, and extreme noise—to the difusion model for indiscriminate global fitting. This homogeneous constraint strategy overlooks the significant information heterogeneity across diferent channels in multivariate sequences. Consequently, the model is highly susceptible to interference from high-noise or anomalous variables during inference generation, which in turn generates adversarial gradients and disrupts the multivariate joint distribution. D3U utilizes a front-end deterministic point forecasting model to extract highcertainty components, relying solely on the difusion model to fit the high-uncertainty residual distribution. This physical decoupling paradigm enables it to achieve excellent performance in most evaluation scenarios. However, it is noteworthy that when the prediction horizon is extended to 720 steps, D3U’s CRPS on the ETTh1 and Appliance datasets deteriorates to 0.471 and 0.875, respectively. In long-term forecasting scenarios, the prediction errors of the upstream deterministic model amplify sharply over time. D3U rigidly attributes all these accumulated cognitive biases to "high-uncertainty residuals" and forces the downstream difusion model to fit them, ultimately resulting in the unconditional amplification of errors and the breakdown of probabilistic forecasting boundaries.

![](images/6e8d6ed7e35ef2df2e5523bba830ff5c2d04155723bca8aa70a0a5f798be4e06.jpg)

![](images/03dbfc4f08fc39fd59e2228be8e9ab3d078f55bd649a0db1bf9f79ac780abc6a.jpg)

Figure 3: Forecasting results on the ETTh1 dataset: MSE (left) and CRPS (right).  
![](images/49391ee4c06fe872663e005887a1ecffbff475e839ab154d27c6dd0374e9bf1f.jpg)

![](images/521a30687f96029e418a69637ea828f794ad1f21062e637fbf8075a85cbd7205.jpg)  
Figure 4: Forecasting results on the Appliance dataset: MSE (left) and CRPS (right).

Overall, the variable-sensitive mechanism benefits most settings by adapting the observation guidance to local precision. However, DynG-Dif does not dominate every Trafic setting. At L = 720, its MSE is close to that of D3U (0.626 versus 0.610), whereas its CRPS is higher (0.362 versus 0.289). This suggests that the main limitation lies in long-horizon probabilistic calibration rather than point estimation. Trafic contains 862 variables, so the long horizon may make dense reliability estimation more dificult. The absence of explicit sensor-topology modeling may also limit cross-variable calibration in this setting.

Table 4: Comparison of MSE and CRPS on six real-world datasets. The prediction horizon length L ∈ {96, 168, 336, 720}. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td colspan="2" rowspan="2">Methods</td><td colspan="2" rowspan="2">DynG-Diff</td><td colspan="2" rowspan="2">D3U</td><td colspan="2">TMDM</td><td colspan="2">TimeDiff</td><td colspan="2">SSSD</td><td colspan="2">CSDI</td><td colspan="2">TimeGrad</td></tr><tr><td>CRPS MSE</td><td>CRPS</td><td>MSE</td><td>CRPS</td><td>MSE</td><td>CRPS</td><td>MSE CRPS</td><td>MSE</td><td></td><td>CRPS</td><td>MSE</td><td>CRPS</td></tr><tr><td colspan="2">Metrics 96</td><td>MSE 0.281</td><td>0.290</td><td>0.421</td><td>0.347</td><td>0.475</td><td>0.392</td><td>0.387</td><td>0.365</td><td>0.985</td><td>0.665</td><td>1.195</td><td>0.602</td><td>1.279</td><td>0.703</td></tr><tr><td rowspan="4">ETT1</td><td>168</td><td>0.285</td><td>0.300</td><td>0.577</td><td>0.418</td><td>0.541</td><td>0.418</td><td>0.408</td><td>0.381</td><td>0.872</td><td>0.623</td><td>1.172</td><td>0.589</td><td>1.196</td><td>0.627</td></tr><tr><td>336</td><td>0.292</td><td>0.317</td><td>0.632</td><td>0.466</td><td>0.601</td><td>0.482</td><td>0.454</td><td>0.409</td><td>0.935</td><td>0.626</td><td>1.201</td><td>0.637</td><td>1.175</td><td>0.589</td></tr><tr><td>720</td><td>0.389</td><td>0.378</td><td>0.644</td><td>0.471</td><td>0.678</td><td>0.524</td><td>0.512</td><td>0.457</td><td>1.106</td><td>0.711</td><td>1.198</td><td>0.634</td><td>1.305</td><td>0.721</td></tr><tr><td>Avg</td><td>0.311</td><td>0.321</td><td>0.568</td><td>0.425</td><td>0.573</td><td>0.454</td><td>0.440</td><td>0.403</td><td>0.974</td><td>0.656</td><td>1.191</td><td>0.615</td><td>1.238</td><td>0.660</td></tr><tr><td rowspan="5">Excange</td><td>96</td><td>0.120</td><td>0.178</td><td>0.111</td><td>0.180</td><td>0.177</td><td>0.261</td><td>0.115</td><td>0.207</td><td>0.467</td><td>0.398</td><td>0.225</td><td>0.273</td><td>1.315</td><td>0.872</td></tr><tr><td>168</td><td>0.193</td><td>0.231</td><td>0.198</td><td>0.257</td><td>0.254</td><td>0.315</td><td>0.211</td><td>0.292</td><td>0.465</td><td>0.397</td><td>0.484</td><td>0.447</td><td>1.304</td><td>0.876</td></tr><tr><td>336</td><td>0.352</td><td>0.317</td><td>0.456</td><td>0.359</td><td>0.396</td><td>0.425</td><td>0.479</td><td>0.443</td><td>0.498</td><td>0.421</td><td>0.410</td><td>0.395</td><td>1.557</td><td>0.982</td></tr><tr><td>720</td><td>0.404</td><td>0.369</td><td>0.687</td><td>0.479</td><td>0.726</td><td>0.685</td><td>0.678</td><td>0.579</td><td>0.703</td><td>0.615</td><td>0.921</td><td>0.709</td><td>1.624</td><td>1.006</td></tr><tr><td>Avg</td><td>0.267</td><td>0.273</td><td>0.363</td><td>0.318</td><td>0.388</td><td>0.421</td><td>0.370</td><td>0.380</td><td>0.533</td><td>0.457</td><td>0.510</td><td>0.456</td><td>1.450</td><td>0.934</td></tr><tr><td rowspan="5">Weathher</td><td></td><td>0.484</td><td>0.178</td><td>0.186</td><td>0.197</td><td>0.255</td><td>0.206</td><td>0.351</td><td>0.326</td><td>0.581</td><td>0.381</td><td>0.501</td><td>0.335</td><td>0.628</td><td>0.413</td></tr><tr><td>168</td><td>0.522</td><td>0.191</td><td>0.217</td><td>0.208</td><td>0.276</td><td>0.242</td><td>0.352</td><td>0.338</td><td>0.488</td><td>0.326</td><td>0.510</td><td>0.347</td><td>0.645</td><td>0.438</td></tr><tr><td>336</td><td>0.521</td><td>0.199</td><td>0.259</td><td>0.227</td><td>0.324</td><td>0.280</td><td>0.359</td><td>0.323</td><td>0.461</td><td>0.313</td><td>0.493</td><td>0.318</td><td>0.497</td><td>0.368</td></tr><tr><td>720</td><td>0.483</td><td>0.195</td><td>0.438</td><td>0.359</td><td>0.347</td><td>0.298</td><td>0.371</td><td>0.384</td><td>0.625</td><td>0.433</td><td>0.343</td><td>0.275</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.519</td><td>0.375</td></tr><tr><td rowspan="5">Appinnce</td><td>Avg 96</td><td>0.502</td><td>0.190</td><td>0.275</td><td>0.247</td><td>0.300</td><td>0.256</td><td>0.358</td><td>0.342</td><td>0.538</td><td>0.363</td><td>0.461</td><td>0.318</td><td>0.572</td><td>0.398</td></tr><tr><td>168</td><td>0.823 0.763</td><td>0.380 0.378</td><td>0.501</td><td>0.347 0.439</td><td>0.587 0.602</td><td>0.417 0.485</td><td>0.688 0.649</td><td>0.496 0.503</td><td>0.964 0.795</td><td>0.587</td><td>0.496 0.583</td><td>0.358 0.446</td><td>1.003</td><td></td><td>0.758</td></tr><tr><td>336</td><td>0.808</td><td>0.423</td><td>0.601 0.880</td><td>0.593</td><td>0.963</td><td>0.627</td><td>0.908</td><td></td><td></td><td>0.508</td><td></td><td></td><td></td><td>1.084</td><td>0.788</td></tr><tr><td>720</td><td></td><td>0.401</td><td>1.276</td><td>0.875</td><td>1.105</td><td>0.768</td><td>1.193</td><td>0.641</td><td>1.286</td><td>0.792</td><td>0.854</td><td>0.569</td><td></td><td>1.207</td><td>0.849</td></tr><tr><td></td><td>0.691</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.823</td><td>1.794</td><td>0.957</td><td>0.899</td><td>0.606</td><td>1.826</td><td></td><td>0.982</td></tr><tr><td rowspan="5">Solar</td><td>Avg</td><td>0.771</td><td>0.395</td><td>0.814</td><td>0.563</td><td>0.814</td><td>0.574</td><td>0.859</td><td>0.615</td><td></td><td>1.209 0.711</td><td></td><td>0.708</td><td>0.494</td><td>1.280</td><td>0.844</td></tr><tr><td>96</td><td>0.297</td><td>0.261</td><td>0.227</td><td>0.235</td><td>0.268</td><td></td><td>0.296</td><td>0.798</td><td>0.587</td><td>0.397</td><td>0.368</td><td>0.358</td><td>0.329</td><td>0.589</td><td>0.408</td></tr><tr><td>168</td><td>0.328</td><td>0.271</td><td>0.273</td><td>0.268</td><td>0.268</td><td>0.294</td><td>0.673</td><td>0.462</td><td></td><td>0.878 0.625</td><td></td><td>0.387</td><td>0.346</td><td>0.711</td><td>0.493</td></tr><tr><td>336</td><td>0.457</td><td>0.324</td><td>0.218</td><td>0.231</td><td>0.410</td><td>0.341</td><td>0.980</td><td>0.698</td><td></td><td>0.794 0.578</td><td></td><td>0.402</td><td>0.358</td><td>0.795</td><td>0.568</td></tr><tr><td>720</td><td>0.273</td><td>0.314</td><td>0.221</td><td>0.234</td><td>0.462</td><td>0.358</td><td>1.195</td><td>0.802</td></table>

![](images/4dc26007a5daac2ab1c0db5396e91221c9402837201786d55493ec2eb028aaf2.jpg)  
(a) Dynamic Guidance

![](images/e15f7ea5c045f935d17c223ba9d497b588e828dabc68185a53036d55b95a295c.jpg)  
(b) Scalar Guidance

![](images/52f6b0c9b648a4fdbf060d1aa1556dadfb53c52a0ffa284e7c8adc43abbcb0d8.jpg)  
(c) None Guidance  
Figure 5: Comparison of probabilistic forecasting intervals generated by the model on the ETTh1 dataset under diferent weight modes.

## 5.3 Ablation studies

To investigate the specific contributions of the core components in the DynG-Dif framework, we quantitatively analyzed the model’s forecasting performance under diferent weight modes to verify the superiority of dynamic weight guidance (Dynamic) over homogeneous scalar guidance (Scalar) and no guidance (None). Specifically, by setting the detached importance weight $\bar { a } _ { l , d } ^ { t }$ in Eq. (14) to a constant 1, the dynamic weight guidance degenerates into homogeneous scalar guidance; by setting the global guidance scale s to a constant 0, the dynamic weight guidance degenerates into no guidance. Table 5 presents the metrics of these three modes across six benchmark datasets under diferent prediction time scales.

When the guidance mechanism is entirely omitted (NONE), the model’s forecasting performance undergoes severe degradation, with its average MSE and CRPS worsening by 125.6% and 81.7%, respectively. This significant performance decay indicates that for an unconditionally trained difusion backbone network, introducing observation guidance during the inference stage is a prerequisite for achieving accurate time series forecasting. Furthermore, when homogeneous guidance is applied, the model’s average MSE and CRPS deteriorate by 26.1% and 14.8%, respectively. This further demonstrates that diferent variables in multivariate time series possess distinctly diferent degrees of importance and information contributions. The "variable-sensitive" dynamic guidance efectively resolves the conflict of variable heterogeneity by intelligently allocating local precision weights to diferent variables, thereby avoiding the disruption of the overall multivariate joint distribution caused by applying indiscriminate homogeneous guidance.

Figure 5 illustrates the probabilistic forecasting intervals generated by the model on a single channel of the ETTh1 dataset under diferent guidance modes. It can be observed that the forecasting intervals generated without guidance (NONE) are extremely difuse, and their median predictions almost completely fail to capture future fluctuation trends. Although homogeneous guidance (SCALAR) can correct the prediction trajectory to some extent, the generated intervals are overly conservative, appearing as a result of "compromise." In contrast, dynamic guidance can apply fine-grained interventions based on the real-time evolutionary state of the system, and its generated probabilistic intervals appear much tighter while ensuring the coverage rate of the ground truth. More probabilistic forecasting results can be found in Appendix C.1.

## 5.4 Model analysis

## 5.4.1 Spatio-Temporal Consistency and Heterogeneity Perception of Dynamic Weights

As illustrated in Figure 6, we extracted the true single-step observation precision of the Weather dataset at key timesteps during the difusion denoising process and compared it with the real-time dynamic guidance weights generated by the network through heatmap visualization. It can be observed that the weight matrix generated by the network (bottom half) and the true prediction precision (top half) exhibit a high degree of semantic consistency in their spatio-temporal distributions. High-precision regions (bright yellow) in the heatmaps often precisely correspond to strong guidance weights, whereas low-precision or high-uncertainty regions (dark black) are assigned extremely weak guidance intensities. The dynamic weight network is also able to acutely perceive and adaptively respond to local anomalous features, such as the sudden drop in precision in certain channels present in the Weather dataset. This result demonstrates that the policy network can not only distinguish the importance diferences among various variables in the spatial dimension but also discern the reliability evolution of the denoising state as the timesteps progress. This aligns well with our initial design objective: applying strong gradient guidance to high-confidence variables to stabilize the recovery path of the overall system, while simultaneously applying weak guidance to low-confidence (unreliable) variables to efectively suppress their negative interference with the denoising process. More visualization results can be found in Appendix C.2.

Table 5: Comparison of MSE and CRPS results under diferent weight modes.
<table><tr><td colspan="2">Weight Mode</td><td colspan="2">Dynamic</td><td colspan="2">Scalar</td><td colspan="2">None</td></tr><tr><td colspan="2">Metrics</td><td>MSE</td><td>CRPS</td><td>MSE</td><td>CRPS</td><td>MSE</td><td>CRPS</td></tr><tr><td rowspan="4">ETT1</td><td>96</td><td>0.281</td><td>0.290</td><td>0.315</td><td>0.307</td><td>0.647</td><td>0.468</td></tr><tr><td>168</td><td>0.285</td><td>0.300</td><td>0.311</td><td>0.325</td><td>0.641</td><td>0.474</td></tr><tr><td>336</td><td>0.292</td><td>0.317</td><td>0.294</td><td>0.324</td><td>0.684</td><td>0.486</td></tr><tr><td>720</td><td>0.389</td><td>0.378</td><td>0.407</td><td>0.400</td><td>0.772</td><td>0.532</td></tr><tr><td rowspan="4">Excaange</td><td>96</td><td>0.121</td><td>0.179</td><td>0.120</td><td>0.178</td><td>0.162</td><td>0.208</td></tr><tr><td>168</td><td>0.193</td><td>0.231</td><td>0.200</td><td>0.234</td><td>0.249</td><td>0.269</td></tr><tr><td>336</td><td>0.351</td><td>0.319</td><td>0.352</td><td>0.317</td><td>0.497</td><td>0.414</td></tr><tr><td>720</td><td>0.404</td><td>0.369</td><td>0.403</td><td>0.371</td><td>1.278</td><td>0.846</td></tr><tr><td rowspan="4">Weather</td><td>96</td><td>0.484</td><td>0.178</td><td>0.818</td><td>0.275</td><td>1.021</td><td>0.394</td></tr><tr><td>168</td><td>0.522</td><td>0.191</td><td>1.683</td><td>0.388</td><td>0.989</td><td>0.385</td></tr><tr><td>336</td><td>0.521</td><td>0.199</td><td>0.704</td><td>0.246</td><td>0.988</td><td>0.394</td></tr><tr><td>720</td><td>0.483</td><td>0.195</td><td>0.587</td><td>0.248</td><td>1.170</td><td>0.470</td></tr><tr><td rowspan="4">Apnce</td><td>96</td><td>0.823</td><td>0.380</td><td>0.804</td><td>0.403</td><td>0.982</td><td>0.722</td></tr><tr><td>168</td><td>0.803</td><td>0.387</td><td>0.763</td><td>0.378</td><td>1.001</td><td>0.487</td></tr><tr><td>336</td><td>0.808</td><td>0.423</td><td>0.824</td><td>0.477</td><td>1.025</td><td>0.534</td></tr><tr><td>720</td><td>0.742</td><td>0.406</td><td>0.691</td><td>0.406</td><td>1.108</td><td>0.632</td></tr><tr><td rowspan="5">Soar</td><td>96</td><td>0.297</td><td>0.261</td><td>0.378</td><td>0.303</td><td>1.497</td><td>0.690</td></tr><tr><td>168</td><td>0.328</td><td>0.271</td><td>0.660</td><td>0.376</td><td>1.496</td><td>0.691</td></tr><tr><td>336</td><td>0.457</td><td>0.324</td><td>0.456</td><td>0.355</td><td>1.497</td><td>0.693</td></tr><tr><td>720</td><td>0.273</td><td>0.314</td><td>0.389</td><td>0.393</td><td>1.471</td><td>0.689</td></tr><tr><td>96</td><td>0.460</td><td>0.310</td><td>0.596</td><td>0.389</td><td>1.281</td><td>0.639</td></tr><tr><td rowspan="4">Trahc</td><td>168</td><td>0.295</td><td>0.247</td><td>0.310</td><td>0.274</td><td>1.148</td><td>0.590</td></tr><tr><td>336</td><td>0.412</td><td>0.282</td><td>0.444</td><td>0.298</td><td>1.138</td><td>0.581</td></tr><tr><td>720</td><td>0.651</td><td>0.371</td><td>0.646</td><td>0.382</td><td>1.323</td><td>0.647</td></tr><tr><td>Overall Avg</td><td>0.444</td><td>0.296</td><td>0.560</td><td>0.340</td><td>1.002</td><td>0.538</td></tr></table>

The localized precision drops in Figure 6 show that reliability varies across both channels and time, providing a qualitative basis for analyzing the response of dynamic guidance to channel corruption.

![](images/2c84e419922fb3d96f47a56b0c6a33a788c74bef9a7afe550b4159b17f09a343.jpg)  
Figure 6: Heatmap comparison between the true observation precision and the dynamic guidance weights generated by the network at key timesteps of the difusion process in the Weather dataset.

Table 6: Comparison of MSE and CRPS degradation between dynamic weight guidance and homogeneous weight guidance models after injecting Gaussian noise into the ETTh1 and Weather datasets.
<table><tr><td colspan="2">Destruction Mode</td><td colspan="4">Dynamic</td><td colspan="4">Scalar</td></tr><tr><td>Metrics</td><td></td><td>MSE</td><td>Degradation</td><td>CRPS</td><td>Degradation</td><td>MSE</td><td>Degradation</td><td>CRPS</td><td>Degradation</td></tr><tr><td>ETTh1</td><td>96</td><td>0.302</td><td>7.47%</td><td>0.315</td><td>8.62%</td><td>0.345</td><td>7.81%</td><td>0.340</td><td>9.67%</td></tr><tr><td>(Noise Ch5)</td><td>168</td><td>0.319</td><td>11.92%</td><td>0.341</td><td>13.66%</td><td>0.384</td><td>24.67%</td><td>0.389</td><td>20.06%</td></tr><tr><td>Weather</td><td>96</td><td>0.401</td><td>-17.14%</td><td>0.207</td><td>16.29%</td><td>0.888</td><td>8.55%</td><td>0.277</td><td>23.11%</td></tr><tr><td>(Noise Ch10, 20)</td><td>168</td><td>0.439</td><td>-15.90%</td><td>0.229</td><td>19.89%</td><td>1.927</td><td>14.49%</td><td>0.450</td><td>25.69%</td></tr></table>

## 5.4.2 Robustness Evaluation under Extreme Noise Scenarios

Based on the conclusion that "the dynamic network can efectively suppress interference from unreliable variables," we further conducted stress tests to evaluate the model’s stability when facing sensor damage or extreme high-frequency noise. We injected independent standard Gaussian noise into partially specified channels of the ETTh1 and Weather datasets, and comprehensively compared the performance degradation rates of dynamic weight guidance and homogeneous scalar guidance on the MSE and CRPS metrics. The results are shown in Table 6, Figure 7.

Under injected noise, homogeneous guidance degrades sharply because it constrains clean and corrupted variables equally. Dynamic guidance instead reduces the influence of unreliable channels and limits the propagation of misleading gradients. On Weather, the negative MSE degradation should not be interpreted as a benefit of noise. Reweighting corrupted channels can incidentally shift the finite-sample mean closer to the ground truth, whereas the higher CRPS shows that distributional quality still deteriorates. This behavior is consistent with the reliability-suppression pattern in Figure 6, although the current evidence does not establish a channel-level causal relationship.

## 5.4.3 Computation eficiency analysis

We quantitatively analyzed the computational overhead of DynG-Dif on ETTh1, Weather, and Trafic. Table 7 and Figure 8 compare parameter counts, training time, and inference time over the full datasets at $L = 9 6$ . The lightweight policy network accounts for approximately 3% to

Table 7: Comparison of parameters and computational eficiency between the unconditional backbone network and the state-aware policy network on three datasets.
<table><tr><td>Dataset</td><td colspan="3">ETTh1</td><td colspan="3">Weather</td><td colspan="3">Traffic</td></tr><tr><td>Model Component</td><td>Params (MB)</td><td>Train (min)</td><td>Infer (min)</td><td>Params (MB)</td><td>Train (min)</td><td>Infer (min)</td><td>Params (MB)</td><td>Train (min)</td><td>Infer (min)</td></tr><tr><td>Backbone</td><td>0.19</td><td>13.62</td><td>3.06</td><td>0.95</td><td>21.62</td><td>30.72</td><td>8.60</td><td>14.65</td><td>28.62</td></tr><tr><td>Policy Network</td><td>0.02</td><td>1.08</td><td>/</td><td>0.03</td><td>1.22</td><td>/</td><td>0.41</td><td>1.50</td><td>1</td></tr><tr><td>DynG-Diff</td><td>0.21</td><td>14.70</td><td>4.18</td><td>0.98</td><td>22.84</td><td>39.74</td><td>9.01</td><td>16.15</td><td>37.31</td></tr></table>

![](images/7f97d657479d6e8e41be7fdc4c2e2e78e24e4b7ea52ec476ebed00a633b4a6be.jpg)

![](images/bc9034ce48bc5b36fc2cca0286c3e7323ec64fe0b09e51b7601f3803d019ceae.jpg)  
Figure 7: Comparison of degradation rates under extreme noise for diferent guidance mechanisms: MSE (left) and CRPS (right).

![](images/f90b8d25b88cc82fc2f63e85aff384506b90d0ce04a250ca9cbdf0ef1efea367.jpg)

![](images/a11cb6b82821dab0588a124d4e028ceee623295c08adbf7057310c043fd0d7a4.jpg)  
Figure 8: Comparison of computational overhead (in minutes) across diferent datasets: Train (left) and Inference (right).

10% of the backbone parameters.

The policy network adds 1.5 minutes of training on Trafic. During inference, however, dynamic guidance increases the total time from 28.62 to 37.31 minutes, an overhead of approximately 30.4%. This cost is non-negligible for latency-sensitive applications.

The overhead mainly arises from computing the weighted observation-likelihood gradient at each reverse step. The current implementation is therefore better suited to ofline or batch forecasting. Future work will explore selective guidance steps and faster difusion solvers.

## 6 Conclusion

In this paper, we proposed DynG-Dif, a variable-sensitive dynamic guidance difusion framework designed to address the challenge of information heterogeneity in probabilistic multivariate time series forecasting. The framework separates unconditional joint-distribution learning from variable-sensitive observation guidance during inference. Its lightweight policy network adaptively infers the reliability of each variable from real-time noisy states and one-step denoising estimates. $\mathrm { B y }$ formulating the dynamic guidance matrix as the local precision of an Asymmetric Laplace Distribution, DynG-Dif applies stronger guidance to high-confidence variables while suppressing interference from anomalous noise. Experiments on multiple real-world datasets demonstrate competitive performance in most scenarios, while stress tests validate its robustness under severe noise.

Building on the probabilistic forecasting capability demonstrated in this study, future work will investigate extensions to missing-value imputation and anomaly detection through taskappropriate guidance mechanisms. Additional directions include accelerating reverse difusion sampling and integrating topological structures to model spatial dependencies among heterogeneous variables. Overall, this work provides a basis for studying reusable unconditional generative backbones in multivariate time-series analysis.

## A Supplementary Derivations of Denoising Difusion Probabilistic Models

In Section 3.2, we briefly introduced the core objective function of Denoising Difusion Probabilistic Models (DDPMs). In this section, we provide supplementary explanations of the core theoretical formulations in both the forward and reverse processes, which serve as the foundation for constructing the dynamic guidance mechanism.

Forward Process and Arbitrary-Step Sampling. During the forward noise-addition process, the model progressively injects Gaussian noise into the initial data $x ^ { 0 }$ through a fixed Markov chain. By leveraging the reparameterization trick and the additivity of independent Gaussian distributions, the latent state $x ^ { t }$ at an arbitrary time step t can be directly sampled from the initial state $x ^ { 0 }$ via a single-step computation:

$$
x ^ { t } = \sqrt { \bar { \alpha } _ { t } } x ^ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon\tag{A.1}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ and the merged noise is $\epsilon \sim \mathcal { N } ( 0 , I )$ . From this formulation, we can inversely derive the model’s real-time, single-step estimation of the original data $\hat { x } ^ { 0 }$ , which constitutes the core premise for DynG-Dif to construct the state-aware policy network and calculate the observation guidance loss.

Reverse Process and Optimization Objective. The reverse denoising process aims to progressively reconstruct the true data distribution starting from standard Gaussian noise $x ^ { T } \sim \mathcal { N } ( 0 , I )$ by learning the transition distributions. Ho et al. [13] demonstrated that by parameterizing and simplifying the Evidence Lower Bound (ELBO), minimizing the KL divergence between the reverse transition distribution and the true posterior distribution is ultimately equivalent to optimizing the simplified mean squared error loss $\mathcal { L } _ { s i m p l e }$ as shown in Eq. (2). During the inference stage, based on the noise $\epsilon _ { \theta } ( x ^ { t } , t )$ predicted by the unconditional difusion backbone, the single-step reverse denoising sampling process can be formulated as:

$$
x ^ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x ^ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x ^ { t } , t ) \right) + \sigma _ { t } z\tag{A.2}
$$

where $z \sim \mathcal { N } ( 0 , I )$ denotes the introduced random noise term. During the guided inference phase of DynG-Dif, we utilize the dynamic weights outputted by the policy network to compute the

observation likelihood-based guidance gradient $\nabla _ { \boldsymbol { x } ^ { t } } \mathcal { L } _ { g u i d e }$ , which is then strictly superimposed onto the aforementioned sampled noise prediction term $\epsilon _ { \theta } ( x ^ { t } , t )$ according to Bayes’ theorem, thereby achieving variable-sensitive conditional generation without the need for retraining.

## B Technical Details

## B.1 Unconditional Backbone Network Architecture and Training Details

In the DynG-Dif framework, the core role of the unconditional backbone network $\epsilon _ { \theta }$ is to learn the underlying joint distribution of multivariate time series, providing a solid generative prior for dynamic guidance during the inference stage. To balance computational eficiency for long sequence modeling and the ability to capture complex dynamics, the network departs from the traditional Transformer architecture and adopts a residual network design based on Structured State Space models (S4) [39].

The input to the network at any arbitrary difusion timestep consists of two parts: one is the current noisy intermediate state $\boldsymbol { x } ^ { t } \in \mathbb { R } ^ { L \times D }$ , where L is the sequence length and D is the variable dimension; the other is the current discrete difusion timestep t. The output of the network is the prediction of the Gaussian noise added at that timestep, $\hat { \epsilon } = \epsilon _ { \theta } ( x ^ { t } , t )$ , whose dimension is consistent with the input, i.e., R<sup>L×D</sup>.

Because DynG-Dif adopts a decoupled training paradigm, the backbone and policy networks are optimized independently. During pre-training, the backbone remains unconditional and does not use historical observations $y _ { o b s }$ or forecasting labels. Its objective is the simplified mean squared error loss in Eq. (2), through which it approximates the joint distribution represented by the training data. The trained backbone then provides one-step estimates at arbitrary difusion timesteps, which are combined with observation likelihoods during guided inference.

The internal architecture of the unconditional backbone network primarily consists of three core modules: feature and time mapping, deep temporal feature extraction, and output aggregation, as illustrated in Figure B.1. After the input $x ^ { t }$ and t undergo feature and time mapping, N S4 residual blocks extract long-term dependencies. The skip features from each block are globally aggregated and fused with residuals to output the predicted noise. Among them, the S4 layer, acting as the core operator, can eficiently capture long-range physical dependencies in multivariate sequences with near-linear complexity, and is combined with a gating mechanism to filter out redundant noise.

## B.2 Asymmetric Laplace Distribution Modeling for Conditional Observation Distribution

In Section 4.2, to accurately delineate the predictive uncertainty in probabilistic forecasting tasks, we adopted the Asymmetric Laplace Distribution (ALD) to model the conditional observation distribution $p ( y _ { o b s } | x ^ { t } )$ given the latent variable $x ^ { t }$ . This section provides the detailed theoretical derivations for this modeling.

For a random variable Y following an Asymmetric Laplace Distribution with a location parameter $\mu ,$ a scale parameter $b > 0$ , and an asymmetry parameter $\kappa \in ( 0 , 1 )$ , its standard probability density function is defined as:

$$
f ( y ; \mu , b , \kappa ) = \frac { \kappa ( 1 - \kappa ) } { b } \exp \left( - \frac { \rho _ { \kappa } ( y - \mu ) } { b } \right)\tag{B.1}
$$

where $\rho _ { \kappa } ( e ) = \operatorname* { m a x } ( \kappa \cdot e , ( \kappa - 1 ) \cdot e )$ is the asymmetric quantile loss. For $( l , d )$ , y corresponds to $y _ { l , d } , \mu$ to $\hat { x } _ { l , d } ^ { 0 } ,$ and $1 / b$ to the policy output $a _ { l , d } ^ { t }$ . During guidance, we use its detached value $\bar { a } _ { l , d } ^ { t } = \mathrm { s g } ( a _ { l , d } ^ { t } )$ as the plug-in local precision. Substitution gives:

$$
p ( y _ { l , d } | x ^ { t } ; \kappa , \bar { a } _ { l , d } ^ { t } ) = \kappa ( 1 - \kappa ) \bar { a } _ { l , d } ^ { t } \exp \left( - \bar { a } _ { l , d } ^ { t } \cdot \rho _ { \kappa } ( y _ { l , d } - \hat { x } _ { l , d } ^ { 0 } ) \right)\tag{B.2}
$$

![](images/aff9c93ecdb41fccf02992d5d73f6c2b55c552ec857a6909a58633ec32300812.jpg)  
Figure B.1: Schematic of the unconditional backbone network architecture and data flow. The network receives the noisy state $x ^ { t }$ and timestep t as inputs. The timestep t is encoded into global temporal features $e ^ { t }$ via a Time MLP and injected layer by layer into N S4 residual blocks; the noisy state $x ^ { t }$ enters the S4 core layer after spatial mapping to extract long-term sequential dependencies. The skip features generated by each residual block are aggregated at a global summation node (Global Skip Sum). After being mapped by the Output Prediction Head, they undergo global residual fusion (Global Residual Add) with the initial state from the input end, ultimately outputting the predicted noise $\hat { \boldsymbol { \epsilon } } = \epsilon _ { \boldsymbol { \theta } } ( \boldsymbol { x } ^ { t } , t )$

Because $\bar { a } _ { l , d } ^ { t }$ is a stop-gradient quantity, $\kappa ( 1 - \kappa ) \bar { a } _ { l , d } ^ { t }$ is constant with respect to $x ^ { t }$ during the current local guidance update. It can therefore be omitted when taking this gradient, yielding Eq. (11).

## C Experiments

## C.1 Additional Comparisons of Generated Probabilistic Forecasting Intervals

In Section 5.3, we illustrated the probabilistic forecasting intervals generated by the model on the ETTh1 dataset under diferent guidance modes. In Figure C.1, we additionally present comparisons of the probabilistic forecasting intervals generated by the model on five datasets: Exchange, Weather, Appliance, Solar, and Trafic. We make targeted adjustments to the historical observation window and the prediction window based on the sampling frequencies of the datasets. For datasets with a "10min" sampling interval (Weather, Appliance, Solar), we set $H = L = 1 6 8$ ; for datasets with an hourly or daily sampling interval (Trafic and Exchange, respectively), we set $H = L = 9 6$ . Furthermore, across all datasets, we select variable channels exhibiting distinct fluctuation amplitudes and evolutionary patterns for visualization.

From the comparisons across the subfigures, it is clearly observable that the dynamic guidance mechanism consistently generates more compact probabilistic intervals that efectively cover the true observations. In contrast, the prediction intervals under the no-guidance mode exhibit significant dispersion, while homogeneous scalar guidance is often constrained by global trade-ofs, yielding overly conservative interval boundaries. This once again corroborates the exceptional performance of the variable-sensitive dynamic intervention mechanism in enhancing the prediction quality of multivariate joint distributions.

Dynamic Guidance  
![](images/f2570faad327b8f881a4765b3f58977051833542d394188867cadd9d76de013b.jpg)

![](images/0fd10b26e02f2ce267acb4773478c969bcde4ba1bdde5c5360f117ac7289c215.jpg)  
(a) Exchange 7th dimension

![](images/fa9f7595333c40d6e6a3ae9fcccb48fcfa6b325b80f2ba9be1fe61ba0120e1a0.jpg)

![](images/27a367a502b989dd67abe55bff2759b7ba3a8fd42e4baecb89a5a82745347243.jpg)

![](images/d72dc3b0689de7b5a6d759dee4ded597d4659a3d67875ce060d91f25d6ab245c.jpg)  
(b) Weather 17th dimension

![](images/0a7adfdb2cd8be4da61faf7108071af5151a5d07b4b24fef4096aa502f4cc60e.jpg)

![](images/4e0afd99db164379f637a3d5bb04fc52456df4b47fbb648046f6d6ff5a2c7488.jpg)

![](images/6c373b4f4e61a4ed7cb3a5081991b9398507abda0b586a5f65efde98380f19be.jpg)  
(c) Appliance 12th dimension

![](images/914469140abe1cf7da11c902c9e6ab93694e2e0a8a5e195a24b6a73fb69e4b1a.jpg)

![](images/311bf01a8253c4e83c26d4d9d5433bef06ca10c8a6c57554466d3c224567886b.jpg)

![](images/9fdb0258da6d54dd511e76636ab9caa6a36e1b73f3fa60e2c5e1f12c6cbaefa5.jpg)  
(d) Solar 6th dimension

![](images/b10c24a5efbace2ba85b1b5e9c79b0e3358c0d3a07dca054a003c2010a1fa32e.jpg)

![](images/0bd827008f02d4849e9fc6ecd8342fe873b5a61f15d12539840c9bac4ba78814.jpg)

![](images/e830456b82e1e8d3a8691487cf125694bc6ca3f5b46d575f1c6d42064a640bc7.jpg)  
(e) Trafic 3rd dimension

![](images/98a6dec558201d059df2fc82a2a86d961db3de9ac89bc59bc37059a0db1a6414.jpg)  
Figure C.1: Comparison of probabilistic forecasting intervals generated by the model under diferent weight modes across various forecasting scenarios.

## C.2 Panoramic Views of Spatiotemporal Consistency in the Dynamic Weight Network

In Section 5.4.1, using the Weather dataset as an example, we focused on analyzing the acute perception capabilities of the dynamic guidance network towards local abnormal features. Figure C.2 further displays heat map comparisons for five datasets—ETTh1, Exchange, Appliance, Solar, and Trafic—at key time steps during the difusion denoising process. In this experiment, we set $H = L = 1 6 8$

As observed from the experimental results, regardless of variations in variable dimensions and sampling frequencies across datasets, the dynamic guidance weights generated by the network consistently maintain a high degree of spatiotemporal semantic consistency with the true observational precision. For instance, in the Trafic heat map shown in Figure C.2e, the true precision exhibits prominent "horizontal band-like" features, indicating the existence of long-term and inherent signal-to-noise ratio diferences among diferent trafic nodes (variable channels). The dynamic weight network successfully captures this cross-channel spatial heterogeneity, continuously allocating strong guidance to high-confidence nodes while precisely isolating the interference from high-noise nodes. In contrast to Trafic, the true precision of the Solar dataset in Figure C.2d presents regular "vertical block-like" features, which highly align with the common physical laws shared by all nodes in solar power generation scenarios (e.g., abrupt illumination changes caused by diurnal cycles). The network demonstrates robust temporal perception and adaptive capabilities towards this, adaptively and synchronously reducing the guidance strength across all channels during time periods when the overall system uncertainty surges.

![](images/eefb006179283d7a19ed0684c0c9a505b6af76eb42aaecf6ca5e7076b3e117c5.jpg)

![](images/aa7af9b6a4d0e2fe9ac03e31e8e1710d46876df0af9b55172217cea736adeb57.jpg)

![](images/a01c88b942a7f7cd44ec918a99770c94068a9061cc255bb2adfb9df2d2300ac6.jpg)

![](images/8c7c059b80ac8c7c2c134144e0d1161cc2a77dffa2b986bba322f0e2f5baac60.jpg)

![](images/57d3f540bcfec742947daf8703d7b442ebaac4174865e3a7b688e93ee7c1ae8f.jpg)

![](images/0ec8c24497e62b00992836e881cd637c52e7977510b38aad17644218fbb76850.jpg)

![](images/ed0cfad9191fa82f0b5083d46071c9909993135598bebfec666d10d6d19db29a.jpg)  
(c) Appliance

![](images/98f9d934d6bc0a709d99ecee7fb916b8defa8231b4637d906cb85b3760d4824e.jpg)  
(e) Trafic  
Figure C.2: Heatmap comparisons between true observational precision and network-generated dynamic guidance weights at key time steps of the difusion process across diferent datasets.

## References

[1] Jakub Nowotarski and Rafał Weron. Recent advances in electricity price forecasting: A review of probabilistic forecasting. Renewable and Sustainable Energy Reviews, 81:1548–1568, 2018.

[2] Xingshuai Huang, Di Wu, and Benoit Boulet. Metaprobformer for charging load probabilistic forecasting of electric vehicle charging stations. IEEE Transactions on Intelligent Transportation Systems, 24(10):10445–10455, 2023.

[3] Yuan Gao, Haokun Chen, Xiang Wang, Zhicai Wang, Xue Wang, Jinyang Gao, and Bolin Ding. DifsFormer: A Difusion Transformer on Stock Factor Augmentation. arXiv preprint arXiv:2402.06656, 2024.

[4] Xian Teng, Sen Pei, and Yu-Ru Lin. Stocast: Stochastic disease forecasting with progression uncertainty. IEEE Journal of Biomedical and Health Informatics, 25(3):850–861, 2020.

[5] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[6] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted Transformers Are Efective for Time Series Forecasting. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

[7] Yunhao Zhang and Junchi Yan. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[8] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International journal of forecasting, 36(3):1181–1191, 2020.

[9] David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico, and Jan Gasthaus. High-dimensional multivariate forecasting with low-rank gaussian copula processes. Advances in neural information processing systems, 32, 2019.

[10] Yan Li, Xinjiang Lu, Yaqing Wang, and Dejing Dou. Generative time series forecasting with difusion, denoise, and disentanglement. Advances in Neural Information Processing Systems, 35:23009–23022, 2022.

[11] Kashif Rasul, Abdul-Saboor Sheikh, Ingmar Schuster, Urs M. Bergmann, and Roland Vollgraf. Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[12] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[14] Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: Conditional scorebased difusion models for probabilistic time series imputation. Advances in neural information processing systems, 34:24804–24816, 2021.

[15] Juan Miguel Lopez Alcaraz and Nils Strodthof. Difusion-based Time Series Imputation and Forecasting with Structured State Space Models. Trans. Mach. Learn. Res., 2023.

[16] Siyang Li, Yize Chen, and Hui Xiong. Channel-aware Contrastive Conditional Difusion for Multivariate Probabilistic Time Series Forecasting. arXiv preprint arXiv:2410.02168, 2024.

[17] Qi Li, Zhenyu Zhang, Lei Yao, Zhaoxia Li, Tianyi Zhong, and Yong Zhang. Difusion-based Decoupled Deterministic and Uncertain Framework for Probabilistic Multivariate Time Series Forecasting. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[18] Chih-Yu Lai, Yu-Chien Ning, and Duane S. Boning. RDIT: Residual-based Difusion Implicit Models for Probabilistic Time Series Forecasting. arXiv preprint arXiv:2509.02341, 2025.

[19] Prafulla Dhariwal and Alexander Nichol. Difusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[20] Jonathan Ho and Tim Salimans. Classifier-Free Difusion Guidance. arXiv preprint arXiv:2207.12598, 2022.

[21] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DifWave: A Versatile Difusion Model for Audio Synthesis. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[22] Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, and Yuyang Bernie Wang. Predict, refine, synthesize: Self-guiding difusion models for probabilistic time series forecasting. Advances in Neural Information Processing Systems, 36:28341–28364, 2023.

[23] Olof Mogren. C-RNN-GAN: Continuous recurrent neural networks with adversarial training. arXiv preprint arXiv:1611.09904, 2016.

[24] Cristóbal Esteban, Stephanie L. Hyland, and Gunnar Rätsch. Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs. arXiv preprint arXiv:1706.02633, 2017.

[25] Jinsung Yoon, Daniel Jarrett, and Mihaela Van der Schaar. Time-series generative adversarial networks. Advances in neural information processing systems, 32, 2019.

[26] Jeha Paul, Bohlke-Schneider Michael, Mercado Pedro, Kapoor Shubham, Singh Nirwan Rajbir, Flunkert Valentin, Gasthaus Jan, and Januschowski Tim. PSA-GAN: Progressive self attention GANs for synthetic time series. arXiv preprint arXiv:2108.00981, 2021.

[27] Abhyuday Desai, Cynthia Freeman, Zuhui Wang, and Ian Beaver. Timevae: A variational auto-encoder for multivariate time series generation. arXiv preprint arXiv:2111.08095, 2021.

[28] Borui Cai, Shuiqiao Yang, Longxiang Gao, and Yong Xiang. Hybrid variational autoencoder for time series forecasting. Knowledge-Based Systems, 281:111079, 2023.

[29] Xingjian Wu, Xiangfei Qiu, Hongfan Gao, Jilin Hu, Bin Yang, and Chenjuan Guo. K<sup>2</sup>VAE: A Koopman-Kalman Enhanced Variational AutoEncoder for Probabilistic Time Series Forecasting. arXiv preprint arXiv:2505.23017, 2025.

[30] Ahmed Alaa, Alex James Chan, and Mihaela van der Schaar. Generative time-series modeling with fourier flows. In International Conference on Learning Representations, 2021.

[31] Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising difusion models for multivariate probabilistic time series forecasting. In International conference on machine learning, pages 8857–8868. PMLR, 2021.

[32] Lifeng Shen, Weiyu Chen, and James Kwok. Multi-resolution difusion models for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024.

[33] Xinyu Yuan and Yan Qiao. Difusion-TS: Interpretable Difusion for General Time Series Generation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

[34] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended difusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022.

[35] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided difusion models. arXiv preprint arXiv:2112.10741, 2021.

[36] Andrea Coletta, Sriram Gopalakrishnan, Daniel Borrajo, and Svitlana Vyetrenko. On the constrained time-series generation problem. Advances in Neural Information Processing Systems, 36:61048–61059, 2023.

[37] Felix Koulischer, Florian Handke, Johannes Deleu, Thomas Demeester, and Luca Ambrogioni. Feedback Guidance of Difusion Models. In Advances in Neural Information Processing Systems, 2025.

[38] Yuxin Li, Wenchao Chen, Xinyue Hu, Bo Chen, Baolin Sun, and Mingyuan Zhou. Transformer-modulated difusion models for probabilistic multivariate time series forecasting. In The Twelfth International Conference on Learning Representations, 2024.

[39] Albert Gu, Karan Goel, and Christopher Ré. Eficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.