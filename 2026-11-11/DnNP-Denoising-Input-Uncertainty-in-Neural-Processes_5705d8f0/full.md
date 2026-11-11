# DnNP: Denoising Input Uncertainty in Neural Processes

Fatemeh Tohidian<sup>1</sup> Chengzhi Shi<sup>1</sup> Soomi Lee<sup>1</sup> Matthew Elia<sup>2</sup> Amy Mueller<sup>2,3</sup> Stratis Ioannidis<sup>1</sup> Jennifer Dy<sup>1</sup>

<sup>1</sup>Electrical & Computer Engineering <sup>2</sup>Civil & Environmental Engineering <sup>3</sup>Marine & Environmental Sciences Northeastern University, Boston, MA, USA

## Abstract

Neural Processes (NPs) are flexible meta-learning models capable of uncertainty quantification across diverse applications. However, the standard NP framework assumes access to noise-free inputs. This is unrealistic in many practical settings, where sensor noise or measurement errors are unavoidable. This discrepancy between model assumptions and real-world data can significantly degrade prediction quality and uncertainty estimates. We introduce Denoising Neural Processes (DnNPs), a principled extension of the NP framework that explicitly models input uncertainty. We train DnNPs via variational inference. DnNPs jointly learn to denoise corrupted inputs and perform meta-learning, making them robust to noisy training data. We demonstrate DnNPs’ effectiveness on both synthetic and realworld datasets, showing consistent improvements in prediction and uncertainty quantification compared to standard NPs. On PM spatial interpolation from real-world air quality sensors, DnNP reduces RMSE by up to 30% and NLL by up to 73% relative to standard NPs. We also provide theoretical grounding: we show that the Bayes-optimal predictor under noisy inputs belongs to the DnNP model class, and that plug-in models with fixed observation variance incur an irreducible KL divergence gap that DnNP provably avoids. Code is available at https://github.com/iSUPER-ML/DnNP.

## 1 Introduction

Meta-learning enables models to quickly adapt to new tasks by learning from many related tasks [8, 4]. Neural Processes (NPs) are flexible meta-learning models that provide uncertainty quantification and have shown strong performance across diverse applications [10]. However, NPs assume that inputs are observed without noise. In practice, inputs are often collected from sensors with measurement uncertainty [25]. This mismatch between model assumptions and real-world data can degrade prediction quality.

Classic NPs [5, 6, 7] do not model input noise. Inputs are treated as deterministic observations, and all uncertainty is captured through the latent variable and the output likelihood [6]. Although the problem of handling label noise for NPs has been studied [21], training NPs with inputs contaminated by noise remains unaddressed. This problem has been studied for Gaussian Processes [14], another method that outputs distributions over functions in a principled manner [24]. Moreover, handling noisy inputs in NPs is challenging for several reasons. Noisy inputs corrupt the learned relationships between features and outputs. This leads to both degraded predictions and miscalibrated uncertainty estimates that fail to account for measurement noise.

We propose Denoising Neural Processes (DnNP), which jointly learn to denoise inputs and make predictions with uncertainty quantification. We derive a principled training objective using variational inference that explicitly accounts for input uncertainty, which is applicable to other NP variants that follow the standard NP framework. We evaluate DnNP across multiple scenarios including synthetic datasets and real-world air quality data, demonstrating consistent improvements. To summarize, our contributions are as follows:

• We propose DnNP, a Neural Process model that explicitly handles noisy input coordinates through a variational denoising mechanism, extending the NP family to a practically important but previously unaddressed setting.

• We provide theoretical guarantees showing that the Bayes-optimal predictor under noisy coordinates belongs to the DnNP model class, and that plug-in models with fixed observation variance incur an irreducible KL divergence gap that DnNP provably avoids.

• We demonstrate the effectiveness of DnNP on real-world $\mathrm { P M } _ { 2 . 5 }$ spatial interpolation on an Air Quality dataset, achieving up to 30% reduction in RMSE and 73% reduction in NLL over standard NPs across multiple feature configurations (Table 2).

## 2 Related Work

Neural Processes (NPs) have shown strong performance for function estimation across various applications, including weather forecasting [9, 1], spatiotemporal prediction [23, 19, 16], and medical image analysis [2]. NPs leverage the fast and scalable nature of neural networks, while providing a distribution over functions like Gaussian Processes (GPs). The first model of the Neural Processes family, Conditional Neural Processes (CNPs) [5], was inspired by the flexibility of GPs while remaining scalable to large and complex datasets. Latent Neural Processes (LNPs) [6] introduce a global latent variable that enables sampling functions which better match the underlying distribution by capturing global structure and uncertainty, addressing the independent prediction issue of CNPs, whereby predictions at different target points are made independently and thus lack coherence across the output space.

Attentive Neural Processes (ANPs) [11] leverage attention to allow each target prediction to selectively attend to the most relevant context points, in contrast to prior work that treats all context points equally through mean pooling. By explicitly modeling translation equivariance, a suitable structural assumption when the underlying process is stationary, Convolutional Conditional Neural Processes (ConvCNPs) [7] broaden the applicability of the Neural Processes family. Transformer Neural Processes (TNPs) [17] adopt a transformer architecture, recasting function approximation as a sequence modeling problem. Distance-Informed Neural Processes (DNPs) [22] enrich the latent space by adding a local latent variable. Functional Neural Processes (FNPs) [13] learn graph-based dependencies between data points rather than relying solely on global latent variables. Despite these advances, all the mentioned variants assume that the model inputs are noise-free, which causes overconfident and inaccurate predictions at test time in real-world settings where inputs are corrupted by noise from various sources.

Robust Neural Processes [21] introduce a training strategy to overcome label noise through two modifications to the standard NP loss function: an additional variance penalty term that discourages the model from inflating pointwise variance, instead encouraging uncertainty to be expressed at the global function level, and restricting the reconstruction loss to target points only to prevent memorization of noisy context values. In contrast, DnNP differs not only in the type of noise addressed, inputs x rather than labels y, but also in the nature of the contribution itself: while Robust NPs propose a training strategy applied to existing architectures, DnNP introduces changes to both the generative model and the architecture.

Gaussian Processes (GPs) have motivated most design choices in the NP family. Gaussian Process Training with Input Noise [14] addresses the problem of noisy inputs by applying a first-order Taylor expansion around the noisy observations as a proxy for the latent noise-free inputs. While an exact treatment would require maintaining a distribution over Taylor expansions, whose first and second moments are analytically tractable but computationally prohibitive, the authors instead differentiate only the GP mean function. This yields a linear model for the input noise, adding a term to the likelihood proportional to the squared posterior mean gradient. DnNP instead treats noise-free inputs as latent variables and learns a variational posterior over them within the NP meta-learning framework, enabling amortized inference across tasks.

## 3 Background

Let $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ and $\mathcal { V } = \mathbb { R }$ . A regression episode (task) $\boldsymbol { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ consists of input– output pairs, where $x _ { i } \in \mathcal X$ and $\bar { y _ { i } } \in \mathcal { V } , i \in [ n ] \equiv \{ 1 , \dots , n \}$ , are generated from an episodespecific latent function $g : \mathcal { X } \overset { } {  } \mathcal { Y }$ . We partition indices [n] into a context set C and target set $T ,$ , where $C \cup T = [ n ]$ . For $S \subseteq [ n ]$ , denote by $\mathcal { D } _ { S } = \bar { \{ ( x _ { i } , y _ { i } ) \} } _ { i \in S } , x _ { S } = \{ x _ { i } \} _ { i \in S }$ and $y _ { S } = \{ y _ { i } \} _ { i \in S }$ , the corresponding restricted datasets, features, and labels, respectively. Given a meta-dataset of episodes $\mathcal { M } = \{ \mathcal { D } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , the meta-learning objective is to learn the conditional predictive distribution $p ( y _ { T } \mid x _ { T } , \mathcal { D } _ { C } )$ . Intuitively, at test time, given a context dataset $\mathcal { D } _ { C }$ , and target inputs x , we should be able to predict the corresponding target labels $y _ { T }$ . For brevity, we omit superscript <sup>(k)</sup> from all per-episode quantities (e.g. $\mathcal { D } _ { T } , x _ { C } , \mathrm { e t c . } )$ where it is unambiguous and write it explicitly when distinguishing across episodes matters (e.g., in the meta-objective).

Neural Processes (NPs) learn conditional $p ( y _ { T } \mid x _ { T } , \mathcal { D } _ { C } )$ across episodes in M while providing uncertainty quantification, assuming that inputs $x _ { [ n ] }$ are observed accurately. In particular, NPs introduce a global latent variable z per episode to represent uncertainty over functions. The standard NP generative model is:

$$
z \sim \mathcal { N } ( 0 , I ) \ \mathrm { ( p r i o r ) } , \qquad y _ { i } \mid x _ { i } , z \sim \mathcal { N } \big ( f _ { \theta } ( x _ { i } , z ) , \sigma _ { y } ^ { 2 } \big ) , \quad i \in [ n ] , \ \mathrm { ( l a b e l \ l i k e l i h o o d ) }\tag{1}
$$

where $f _ { \theta }$ is a learnable decoder parameterized by θ and $\sigma _ { y } ^ { 2 }$ is the observation noise variance. We write $f ^ { \star } ( \cdot , z )$ for the ground-truth counterpart of $f _ { \theta } ( \cdot , z ) , \mathrm { i . e . }$ , the realizable optimum used in our theoretical analysis (Section 5). NPs utilize amortized inference, i.e., approximate the posterior on z via an encoder $q _ { \phi } ( z | \cdot )$ , parameterized by $\phi ;$ from this, NPs construct two approximate posteriors over z given the context, $\dot { q } _ { \phi } ( z \mid \mathcal { D } _ { C } )$ , and given context plus targets, $q _ { \phi } ( z \mid \mathcal { D } _ { C } \cup \mathcal { D } _ { T } )$ , used in training. The NP training objective is the evidence lower bound (ELBO) on the conditional log-likelihood:

$$
\log p _ { \theta } ( y _ { T } | x _ { T } , \mathcal { D } _ { C } ) \geq \mathbb { E } _ { q _ { \phi } ( z | \mathcal { D } _ { C } \cup \mathcal { D } _ { T } ) } \Big [ \sum _ { i \in T } \log p _ { \theta } ( y _ { i } | x _ { i } , z ) \Big ] - \mathrm { K L } \Big ( q _ { \phi } ( z | \mathcal { D } _ { C } \cup \mathcal { D } _ { T } ) \| q _ { \phi } ( z | \mathcal { D } _ { C } ) \Big )\tag{2}
$$

We denote the right-hand side of (2) as $\mathcal { L } _ { \mathrm { E L B O } } ^ { ( k ) } ( \Theta )$ when evaluated on episode $\mathcal { D } ^ { ( k ) }$ with split $( C ^ { ( k ) } , T ^ { ( k ) } )$ , where $\Theta = ( \theta , \phi )$ collects all learnable parameters. The parameters are learned by maximizing the average per-episode ELBO over the meta-dataset:

$$
\begin{array} { r } { \Theta ^ { \star } = \underset { \Theta } { \arg \operatorname* { m a x } } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { L } _ { \mathrm { E L B O } } ^ { ( k ) } ( \Theta ) . } \end{array}\tag{3}
$$

This objective is optimized via stochastic gradient ascent. At each step, a mini-batch of episodes is sampled from $\mathcal { M } ,$ a context/target split is sampled for each, and the per-episode ELBO is estimated with a single Monte Carlo sample from $q _ { \phi }$ using the reparameterization trick. At test time, given a new episode with context $\mathcal { D } _ { C }$ and target inputs $x _ { T }$ , predictions can be obtained by: (i) drawing $z \sim q _ { \phi } ( \cdot \mid \mathcal { D } _ { C } )$ , then (ii) for each $i \in T$ , predicting $\hat { y } _ { i } = f _ { \theta } ( x _ { i } , z )$

## 4 Denoising Neural Processes (DnNP)

Neural Processes capture two sources of uncertainty: function-level uncertainty through the global latent variable $z ,$ and output noise through the variance $\sigma _ { y } ^ { 2 } .$ . However, they assume that inputs are observed precisely, and cannot directly model input noise. In many practical scenarios, such as sensor networks with known measurement error, the observed input $x _ { i } \in \mathcal X$ is a noisy version of an underlying clean input ${ x ^ { \ast } } _ { i } \in \mathcal { X }$ . A standard NP applied directly to $x _ { i }$ implicitly assumes the inputs are noise-free; the resulting model is misspecified whenever measurement noise is non-negligible.

We address this by establishing a data-generative process that explicitly accounts for noisy inputs and propose a methodology for handling them within the Neural Process framework. We adopt the same episodic setup as in Section 3: a meta-dataset $\mathcal { M } = \{ \mathcal { D } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ of episodes, each split into context and target sets $C , T$ . The key difference is that we assume that episodes $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ comprise noisy input–output pairs: each $x _ { i }$ is a corrupted observation of an unobserved noise-free input $\boldsymbol { x ^ { * } } _ { i }$ We extend the NP generative process (1) by treating noise-free inputs as latent variables as follows:

$$
\begin{array} { r l r l r l } & { x ^ { * } _ { i } \sim p ( x ^ { * } _ { i } ) , } & & { x _ { i } \mid x ^ { * } _ { i } \sim \mathcal { N } ( x ^ { * } _ { i } , \Gamma ) , } & & { z \sim \mathcal { N } ( 0 , I ) , } & & { y _ { i } \mid x ^ { * } _ { i } , z \sim \mathcal { N } \big ( f _ { \theta } ( x ^ { * } _ { i } , z ) , \sigma _ { y } ^ { 2 } \big ) , } \end{array}\tag{4}
$$

where $\Gamma \succ 0$ is the known measurement-noise covariance. The joint model is

$$
\begin{array} { r } { p ( \mathcal { D } , x ^ { * } { } _ { [ n ] } , z ) = p ( z ) \prod _ { i = 1 } ^ { n } p ( x ^ { * } i ) p ( x _ { i } \mid x ^ { * } i ) p _ { \theta } ( y _ { i } \mid x ^ { * } i , z ) . } \end{array}\tag{5}
$$

The graphical model contains a global latent z (function uncertainty) and local latents $x ^ { * }$ <sub>i</sub> (measurement uncertainty); see Figure 1. The prior $p ( x ^ { * } , i )$ can be chosen to reflect domain knowledge, e.g., uniform on a bounded region or a broad Gaussian.

The posterior $p ( x ^ { * } { } _ { [ n ] } , z \mid { \mathcal { D } } )$ is intractable in general. We introduce a structured variational family that mirrors the conditional dependence in the generative model: a denoiser $q _ { \psi } .$ , parameterized by ψ, that infers noise-free inputs from noisy ones, and an NP-style latent encoder $q _ { \phi }$ that infers the function-level latent given noise-free inputs and observations:

$$
\begin{array} { r } { q _ { \psi , \phi } ( x ^ { * } { } _ { [ n ] } , z | \mathcal { D } ) = q _ { \psi } ( x ^ { * } { } _ { [ n ] } | x _ { [ n ] } ) q _ { \phi } ( z | x ^ { * } { } _ { [ n ] } , y _ { [ n ] } ) , \quad q _ { \psi } ( x ^ { * } { } _ { [ n ] } | x _ { [ n ] } ) = \prod _ { i = 1 } ^ { n } q _ { \psi } ( x ^ { * } { } _ { i } | x _ { i } ) . } \end{array}\tag{6}
$$

The factorization $q _ { \psi } ( x ^ { * } { } _ { i } \mid x _ { i } )$ reflects an architectural choice: each noise-free input is inferred from its corresponding noisy measurement (and optionally a known noise level Γ), independently across observations. The latent encoder $q _ { \phi }$ shares the form of the standard NP variational posterior (2) but now takes denoised inputs $x ^ { * } { } _ { [ n ] }$ rather than the noisy $x _ { [ n ] }$

Combining the variational family (6) with the joint model (5) and applying Jensen’s inequality yields the evidence lower bound (ELBO) on the conditional log-likelihood of the labels given the noisy inputs (full derivation in Appendix D):

$$
\begin{array} { r l } & { \log p ( y _ { [ n ] } \mid x _ { [ n ] } ) \ge \mathbb { E } _ { q _ { \psi , \phi } } \Big [ \underbrace { \sum _ { i = 1 } ^ { n } \log p _ { \theta } ( y _ { i } \mid x ^ { * } _ { i } , z ) } _ { \mathrm { p r e d i c i v e } } + \underbrace { \sum _ { i = 1 } ^ { n } \log p ( x _ { i } \mid x ^ { * } _ { i } ) } _ { \mathrm { c o n s u r e m e n t } } \Big ] } \\ & { - \underbrace { \mathrm { K L } \Big ( q _ { \psi } ( x ^ { * } _ { [ n ] } \mid x _ { [ n ] } ) \ \| \ p ( x ^ { * } _ { [ n ] } ) \Big ) } _ { \mathrm { d e n o i s e r } } - \underbrace { \mathbb { E } _ { q _ { \psi } } \mathrm { K L } \Big ( q _ { \phi } ( z \mid x ^ { * } _ { [ n ] } , y _ { [ n ] } ) \ \| \ p ( z ) \Big ) } _ { \mathrm { r e g u l a r i z a t i o n } } . } \end{array}\tag{7}
$$

This bound has three interpretable components: (i) a predictive reconstruction term that scores y under denoised inputs, (ii) a measurement-consistency term log $p ( x _ { i } \mid x ^ { * } { } _ { i } )$ that ties the denoiser to the noise model, and (iii) two KL regularizers that prevent the denoiser and the latent encoder from drifting too far from their priors.

DnNP extends the NP parameter set with a denoiser network: $\Theta = ( \theta , \phi , \psi )$ , where θ parameterizes the decoder, $\phi$ the latent encoder, and $\psi$ the denoiser variational posterior $q _ { \psi }$ . For meta-learning, the quantity of interest is the conditional likelihood of the targets given the context, log $p ( y _ { T } \mid x _ { [ n ] } , y _ { C } )$ Applying the same bound to this conditional replaces the prior $p ( z )$ in (7) with the conditional prior $p ( z \mid x ^ { * } c , y _ { C } )$ , and restricts the predictive reconstruction term to the target set. Since this conditional prior is intractable, we follow [6] and approximate it with the variational posterior $q _ { \phi } ( z \mid x ^ { * } c , y _ { C } )$ which yields the training objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { E L B O } } ^ { ( k ) } ( \Theta ) = \mathbb { E } _ { q _ { \psi } ( x ^ { * } \mid \pi | ) } \Bigg [ \displaystyle \sum _ { i = 1 } ^ { n } \log p ( x _ { i } \mid x ^ { * } _ { i } ) \Bigg ] - \mathrm { K L } \Big ( q _ { \psi } ( x ^ { * } \lceil n \rceil | x _ { \lceil n \rceil } ) \| p ( x ^ { * } \lfloor n \rfloor ) \Big ) } \\ & { + \mathbb { E } _ { q _ { \psi } ( x ^ { * } \lceil n \rceil | x _ { \lceil n \rceil } ) } \Bigg [ \mathbb { E } _ { q _ { \phi } ( z | x ^ { * } \lceil n \rceil , y _ { \lceil n \rceil } ) } \displaystyle \sum _ { i \in T } \log p _ { \theta } ( y _ { i } \mid x ^ { * } _ { i } , z ) - \mathrm { K L } \Big ( q _ { \phi } ( z | x ^ { * } \lceil n \rceil , y _ { \lfloor n \rfloor } ) \| q _ { \phi } ( z | x ^ { * } _ { C } , y _ { C } ) \Big ) \Bigg ] . } \end{array}\tag{8}
$$

We learn Θ by maximizing the average per-episode ELBO over the meta-dataset, mirroring the NP meta-objective (3). We optimize this objective by stochastic gradient ascent with Adam: at each step, we sample a mini-batch of episodes from $\mathcal { M } .$ , sample a context/target split for each, and estimate the per-episode ELBO with a single Monte Carlo sample from each variational distribution $( q _ { \psi }$ and $q _ { \phi } )$ using the reparameterization trick. Gradients flow through both sampling steps to update Θ. At test time, given a new episode with noisy context $( x _ { C } , y _ { C } )$ and noisy target inputs $x _ { T }$ , we predict $y _ { T }$ by ancestral sampling: (i) draw $x ^ { * } { } _ { [ n ] } \sim q _ { \psi } ( \cdot \mid x _ { [ n ] } )$ , (ii) draw $z \stackrel { \cdot } { \sim } q _ { \phi } \bar { ( } { \cdot } \mid x ^ { \bar { * } } c , y _ { C } )$ , and (iii) for each $i \in T$ , predict $\hat { y } _ { i } = f _ { \theta } ( x ^ { * } { } _ { i } , z )$ . Note that the denoising step (i) departs from standard NP inference. If some inputs, or individual input dimensions, are known to be noise-free, the denoiser is skipped for them and the clean values are passed directly to the encoder and decoder.

![](images/970b541f63dbef3825b91dc128fbc2b5692903f6a48a4fce2ed029aa43b1c930.jpg)  
Figure 1: Comparison of Neural Processes (NP) and DnNP in terms of graphical model (top) and architecture (bottom). In the graphical model, shaded nodes represent observed variables. Standard NPs assume that inputs are observed noise-free, an assumption violated in many real-world settings where measurements are inherently corrupted by noise. DnNP addresses this by treating noise-free inputs as latent variables $x ^ { * }$ that must be inferred from noisy observations $x ,$ while preserving the standard NP structure of context and target sets with a global latent z that captures functionlevel uncertainty. In the architecture, DnNP extends the standard NP encoder-decoder pipeline by prepending a denoising component (shown in green) that recovers noise-free input estimates from noisy inputs before passing them to the encoder.

## 5 Theoretical Guarantees

In this section, we analyze DnNP theoretically and show that it provably outperforms noise-unaware NP baselines in the presence of input noise. We proceed in four steps. First, under a Gaussian prior, the posterior for the unknown noise-free input $x ^ { * }$ is Gaussian (Proposition 5.1). Second, we derive the Bayes-optimal predictive distribution $p ^ { \star } ( y \mid x , z )$ under noisy inputs and characterize how input uncertainty inflates predictive variance (Equation 10). Third, we compare two predictive families: the DnNP family, which marginalizes over a learned posterior on $x ^ { * }$ , and the plug-in family used by NP, which treats $x$ as if it were noise-free. We prove that any plug-in model with fixed observation variance $\sigma _ { y } ^ { 2 }$ incurs an irreducible KL gap to $p ^ { \star }$ whenever input uncertainty propagates through the ground-truth function (Theorem 5.2), while DnNP can close this gap entirely under realizability (Theorem 5.3). Fourth, we quantify how the plug-in gap scales with posterior input uncertainty: a moment expansion (Lemma 5.4) yields explicit KL lower bounds in the small-noise regime (Theorem 5.5), which simplifies to a clean $\tau ^ { 4 } \lVert \nabla f ^ { \star } \rVert ^ { 4 } / ( 4 \sigma _ { y } ^ { 4 } )$ scaling under isotropic posterior covariance (Corollary 5.6). All proofs are deferred to Appendix F. We use a Gaussian prior on $x ^ { * }$ as a concrete example; the proof and an extension to a uniform prior are in Appendix C.

Proposition 5.1 (Gaussian posterior under Gaussian corruption). Under (4) with a Gaussian prior $x ^ { * } \overset { \cdot } { \sim } \mathcal { N } ( \mu , \Sigma ) , \Sigma \succ 0$ , the posterior over the noise-free input is Gaussian, $p ( x ^ { * } \mid x ) = \Lambda { \hat { ( } } x ^ { * } \mid$ $\mu _ { \mathrm { p o s t } } ( x ) , \Sigma _ { \mathrm { p o s t } } ) , w i t h \mu _ { \mathrm { p o s t } } ( x ) = ( \Sigma ^ { - 1 } + \Gamma ^ { - 1 } ) ^ { - 1 } ( \Sigma ^ { - 1 } \mu + \Gamma ^ { - 1 } x ) , \Sigma _ { \mathrm { p o s t } } = ( \Sigma ^ { - 1 } + \Gamma ^ { - 1 } ) ^ { - 1 } \mu ,$

A bounded-domain version, obtained by replacing the Gaussian prior with a uniform prior on a hyper-rectangle, yields a truncated multivariate normal posterior; see Appendix C.

Bayes-Optimal Predictive Conditional. Given Proposition 5.1, the Bayes-optimal predictive conditional under noisy inputs is $\begin{array} { r } { p ^ { \star } ( y \mid x , z ) = \int \hat { \mathcal { N } } \hat { ( } y \mid f ^ { \star } ( x ^ { \ast } , z ) , \sigma _ { y } ^ { 2 } ) \bar { p } ( x ^ { \ast } \hat { \mid } x ) d x ^ { \ast } } \end{array}$ . If $X \sim$ $p ( x ^ { * } \mid x )$ and $\varepsilon \sim \mathcal N ( 0 , \sigma _ { y } ^ { 2 } )$ is independent of X, then $Y = f ^ { \star } ( X , z ) + \varepsilon$ has conditional mean and variance

$$
\mu _ { \star } ( x , z ) : = \mathbb { E } [ Y \mid x , z ] = \mathbb { E } [ f ^ { \star } ( X , z ) \mid x ] ,\tag{9}
$$

$$
v _ { \star } ( x , z ) : = \mathrm { V a r } ( Y \mid x , z ) = \sigma _ { y } ^ { 2 } + \mathrm { V a r } ( f ^ { \star } ( X , z ) \mid x ) .\tag{10}
$$

Equation (10) shows explicitly how input uncertainty inflates predictive variance. Since $p ^ { \star } ( y \mid x , z )$ requires integrating over the intractable posterior $p ( x ^ { * } \mid x , z )$ , we instead utilize the variational posterior $q _ { \psi } ( \bar { x } ^ { * } \mid x )$ . The difference between DnNP and noise-unaware NP is most transparent at the level of conditional predictive distributions given the episode representation. Accordingly, throughout this section we condition on the episode latent z (random for NP). We assume the scalar-output generative model as in Section 4.

The DnNP predictive family (conditional on z) is

$$
q _ { \theta , \psi } ^ { \mathrm { D n N P } } ( y \mid x , z ) = \int { \mathcal N } \big ( y \mid f _ { \theta } ( x ^ { * } , z ) , \sigma _ { y } ^ { 2 } \big ) q _ { \psi } ( x ^ { * } \mid x ) d x ^ { * } .\tag{11}
$$

In contrast, a noise-unaware NP baseline that plugs in x as if it were noise-free has conditional form

$$
q _ { \theta } ^ { \mathrm { p l u g } } ( y \mid x , z ) = \mathcal { N } \big ( y \mid m _ { \theta } ( x , z ) , \sigma _ { y } ^ { 2 } \big ) ,\tag{12}
$$

where $m _ { \theta }$ is a measurable mean function. For NPs, (12) is the conditional decoder after fixing the global latent z.

Theorem 5.2 (Irreducible error in NPs). Fix $( x , z )$ and let $P _ { x , z } : = p ^ { \star } ( \cdot \mid x , z )$ . For any measurable meanfunction m(x, z),

$$
\mathrm { K L } \big ( P _ { x , z } \parallel \mathcal { N } \big ( m ( x , z ) , \sigma _ { y } ^ { 2 } \big ) \big ) \geq \frac { \big ( \mu _ { \star } ( x , z ) - m ( x , z ) \big ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \frac { 1 } { 2 } \left( \frac { v _ { \star } ( x , z ) } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } ( x , z ) } { \sigma _ { y } ^ { 2 } } \right) .
$$

In particular,

$$
\operatorname* { i n f } _ { m } \operatorname { K L } \bigl ( P _ { x , z } \parallel \mathcal { N } ( m , \sigma _ { y } ^ { 2 } ) \bigr ) \geq \frac { 1 } { 2 } \left( \frac { v _ { \star } ( x , z ) } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } ( x , z ) } { \sigma _ { y } ^ { 2 } } \right) ,\tag{13}
$$

and the right-hand side is strictly positive whenever Va $\operatorname { r } \bigl ( f ^ { \star } ( X , z ) \mid x \bigr ) > 0$

Equation (13) shows that NP-based models incur an irreducible KL gap whenever input uncertainty propagates through $f ^ { \star }$ . We now show that DnNP, in contrast, can close this gap entirely because of the noise awareness.

Theorem 5.3 (Noise-aware realizability of DnNP). Assume there exist parameters $( \theta ^ { \star } , \psi ^ { \star } )$ such that

$$
f _ { \theta ^ { \star } } ( x ^ { * } , z ) = f ^ { \star } ( x ^ { * } , z ) \quad f o r a l l ( x ^ { * } , z ) ,
$$

$$
q _ { \psi ^ { \star } } ( x ^ { * } \mid x ) = p ( x ^ { * } \mid x ) \quad f o r a l l \ : x .\tag{14}
$$

Then the noise-aware predictor coincides with the Bayes predictor:

$$
q _ { \theta ^ { \star } , \psi ^ { \star } } ^ { \mathrm { D n N P } } ( y \mid x , z ) = p ^ { \star } ( y \mid x , z ) ~ f o r ~ a l l ~ ( x , z ) .\tag{15}
$$

Consequently,

$$
\operatorname* { i n f } _ { \theta , \psi } \mathbb { E } _ { x , Z } \Big [ \mathrm { K L } \big ( p ^ { \star } ( \cdot \mid x , Z ) \mid \mid q _ { \theta , \psi } ^ { \mathrm { D n N P } } ( \cdot \mid x , Z ) \big ) \Big ] = 0 .\tag{16}
$$

Theorem 5.3 shows that the Bayes predictor under noisy inputs belongs to the DnNP class. We now quantify how the plug-in KL gap from Theorem 5.2 scales with posterior input uncertainty in the small-noise regime. We first establish a moment expansion for the Bayes predictive moments, then combine it with Theorem 5.2 to obtain explicit KL lower bounds.

Lemma 5.4 (Small-noise moment expansion). Assume $f ^ { \star } ( \cdot , z )$ is twice continuously differentiable with globally Lipschitz Hessian. As $\| \dot { \Sigma } _ { \mathrm { p o s t } } \| \stackrel { \cdot } {  } 0$

$$
\mu _ { \star } ( x , z ) = f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) + \frac { 1 } { 2 } \mathrm { t r } \big ( \nabla _ { x ^ { * } } ^ { 2 } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) \Sigma _ { \mathrm { p o s t } } \big ) + O ( \| \Sigma _ { \mathrm { p o s t } } \| ^ { 3 / 2 } ) ,\tag{17}
$$

$$
\begin{array} { r } { v _ { \star } ( x , z ) - \sigma _ { y } ^ { 2 } = ( \nabla _ { x ^ { * } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) ) ^ { \top } \Sigma _ { \mathrm { p o s t } } \nabla _ { x ^ { * } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) + O ( \| \Sigma _ { \mathrm { p o s t } } \| ^ { 2 } ) . } \end{array}\tag{18}
$$

This lemma shows the posterior mean and variance of the ground truth function $f ^ { \star }$

Theorem 5.5 (NP KL gap in the small-noise regime). Under the assumptions of Lemma 5.4, as $\| \Sigma _ { \mathrm { p o s t } } \|  0 ,$

$$
\begin{array} { r l } & { \mathrm { K L } \bigl ( P _ { x , z } \parallel \mathcal { N } \bigl ( f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) , \sigma _ { y } ^ { 2 } \bigr ) \bigr ) \geq \ \frac { 1 } { 8 \sigma _ { y } ^ { 2 } } \bigl ( \mathrm { t r } ( \nabla _ { x ^ { \star } } ^ { 2 } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) \Sigma _ { \mathrm { p o s t } } ) \bigr ) ^ { 2 } } \\ & { + \frac { 1 } { 4 \sigma _ { y } ^ { 4 } } \bigl ( \nabla _ { x ^ { \star } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) ^ { \top } \Sigma _ { \mathrm { p o s t } } \nabla _ { x ^ { \star } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) \bigr ) ^ { 2 } + o ( \Vert \Sigma _ { \mathrm { p o s t } } \Vert ^ { 2 } ) . } \end{array}\tag{19}
$$

Moreover, after optimizing over the predictive mean,

$$
\operatorname* { i n f } _ { m } \mathrm { K L } \bigl ( P _ { x , z } \| \mathcal { N } ( m , \sigma _ { y } ^ { 2 } ) \bigr ) \geq \frac { 1 } { 4 \sigma _ { y } ^ { 4 } } \bigl ( \nabla _ { x ^ { \star } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) ^ { \top } \Sigma _ { \mathrm { p o s t } } \nabla _ { x ^ { \star } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) \bigr ) ^ { 2 } + o ( \| \Sigma _ { \mathrm { p o s t } } \| ^ { 2 } ) .\tag{20}
$$

Then, in small noise regime, we have the concrete KL lower bound for NP and the optimal Bayes predictor. Moreover, if we consider an isotropic posterior covariance, we have the following corollary. Corollary 5.6 (Isotropic posterior covariance). Under the assumptions ofTheorem 5.5, $i f \Sigma _ { \mathrm { p o s t } } =$ $\tau ^ { 2 } I _ { d } ,$ , then the irreducible term in (20) scales as $\begin{array} { r l } {  { \frac { \tau ^ { 4 } } { 4 \sigma _ { y } ^ { 4 } } \bigl \| \nabla _ { x ^ { * } } f ^ { \star } ( \mu _ { \mathrm { p o s t } } ( x ) , z ) \bigr \| _ { 2 } ^ { 4 } + o ( \tau ^ { 4 } ) } \qquad } & { { } } \end{array}$

Thus, in the small noise regime, the irreducible error is related to the gradient of $f ^ { \star }$ at the ground truth $x ^ { * }$ , the input noise variance, and the output noise variance.

## 6 Experiments

To evaluate the effectiveness of DnNP in handling noisy inputs, we conduct experiments comparing it against multiple Neural Processes variants on four datasets. We begin by describing the datasets and baseline methods, then introduce the experimental configuration, and conclude with a discussion of the results.

Data. Table 3 in the appendix summarizes the datasets used in our experiments. The synthetic dataset comprises 1D regression tasks with functions sampled from a $\bar { \mathrm { G P } } ^ { \bar { } }$ with an RBF kernel. We also include MNIST [12] image completion, a widely-used benchmark in Neural Processes evaluations. Additionally, we apply our method to two real-world spatial interpolation datasets: an Air Quality dataset of $\mathrm { P M } _ { 2 . 5 }$ measurements that we collected from a network of urban sensors in Brookline, MA, which we make publicly available at https://hdl.handle.net/2047/D20881761, and a publicly available dataset of daily sea-level pressure observations from NOAA weather stations across California [18], obtained via the Meteostat library [15]. Both datasets contain real-world sensor measurements with inherent noise, demonstrating the importance of accounting for input uncertainty in practical applications.

Methods. We compared DnNP with standard Neural Processes (NP) [6], Conditional Neural Processes (CNP) [5], Attentive Neural Processes (ANP) [11], and Convolutional Conditional Neural Processes (ConvCNP) [7]. To assess the impact of input noise, we train an additional baseline on noise-free data for synthetically corrupted datasets (1D GP and MNIST), representing the best achievable performance without input noise. To simulate input uncertainty, we add zero-mean Gaussian noise to the model inputs with varying noise levels. Noise is applied independently across input dimensions and data points, modeling realistic measurement scenarios in which each sensor reading is affected by independent noise.

Metrics. We evaluate model performance on the target set using the posterior predictive, with latent variables marginalized by Monte Carlo: we draw K samples from the variational posteriors, obtaining predictive means and standard deviations $( \mu _ { k } , \sigma _ { k } )$ , and compute $\mathrm { \Delta N L L } \ =$ $\begin{array} { r } { - \frac { 1 } { | T | } \sum _ { i \in T } \log \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { N } ( y _ { i } \mid \mu _ { k , i } , \sigma _ { k , i } ^ { 2 } ) } \end{array}$ and $\begin{array} { r } { \mathrm { R M S E } \ = \ \sqrt { \frac { 1 } { | T | } \sum _ { i \in T } ( y _ { i } - \bar { \mu } _ { i } ) ^ { 2 } } } \end{array}$ , where $\bar { \mu } _ { i } ~ =$ $\textstyle { \frac { 1 } { K } } \sum _ { k } \mu _ { k , i }$ . We compare our method against two neural process (NP) baselines. The first baseline is an NP trained on noise-free data (prior to noise injection). The second baseline is an NP trained directly on noisy inputs, representing the setting in which input uncertainty is ignored during modeling. We adopt the same backbone architecture for DnNP as for NP, with the addition of a denoising component. The model is trained using the loss defined in Equation 8. Hyper-parameters, including the weighting coefficient β that controls the strength of the denoiser KL regularizer relative to the predictive reconstruction term, are selected via grid search. The model achieving the lowest validation NLL is selected and evaluated on the test set, where it is compared against all baselines.

## 6.1 1D Synthetic Functions

We evaluated on synthetic 1D regression tasks where functions are sampled from a Gaussian Process with an RBF kernel (length-scale 0.5). During both training and evaluation, context points are corrupted by additive Gaussian noise $x _ { C } = \bar { x } ^ { * } c + \varepsilon , \varepsilon \sim \mathcal { N } ( 0 , \sigma _ { x } ^ { 2 } )$ , across five noise levels $\sigma _ { x } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ , with a separate model trained for each noise level. All models share the same encoder-decoder architecture (hidden dim 128, latent dim 64) and are trained on 10,000 GP-sampled tasks with random context size drawn uniformly from [5, 30]. Clean NP is trained and evaluated on noise-free inputs $x ^ { * }$ and serves as a lower bound on both NLL and RMSE, while Noisy NP is trained and evaluated on corrupted inputs x. Figure 2 shows that as noise increases, Noisy NP progressively drifts from the true function. The NLL sweep confirms this quantitatively, showing that DnNP achieves consistently lower NLL across all noise levels.

![](images/9a7c437f753d6579310e874f016284f16a8a47651ac3c74f28b2c3282192ffe5.jpg)  
Figure 2: Comparison of Clean NP, Noisy NP, and DnNP on 1D GP regression across noise levels $\sigma _ { x } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ . Shaded regions show the predictive mean ±σ and ±2σ uncertainty bands. Black circles indicate true context inputs and orange triangles show the corrupted observed input. As input noise increases, Noisy NP produces increasingly biased mean predictions, whereas DnNP maintains closer agreement with the true function. The NLL sweep (right) confirms this quantitatively, showing that DnNP achieves consistently lower NLL than Noisy NP across all noise levels.

DnNP is also robust to misspecification of the input noise level: on the 1D benchmark with true $\sigma _ { x } = 0 . 3$ , training with an assumed σ ranging from 0.5 to 2 times the true value changes RMSE only from 0.363 to 0.406 (vs. 0.373 at the true σ), with NLL nearly flat $( \approx 0 . 4 7 – 0 . 5 4 )$ , within seed variation. This is consistent with our real-world experiments, where σ is set from manufacturer specifications rather than known exactly. When no estimate of σ is available, σ can be learned jointly with the model; the learned value tracks the true noise level monotonically, with a slight upward bias, and gives predictive performance close to, though not exceeding, the fixed-σ model, where the known σ is provided. We also experimented with more complex function classes such as periodic kernels, where all models, including Clean NP, underfit; this is consistent with prior observations that latent NPs with a global representation require more expressive architectures, such as attentive or convolutional NPs, to capture such structure [3].

## 6.2 MNIST Image Completion

We use the MNIST dataset of $2 8 \times 2 8$ grayscale handwritten digits with the standard 60K/10K train/test split. We further split the test set into 5K validation and 5K test subsets (fixed random seed for reproducibility). The task is image completion: given sparse pixel observations, predict the full 784-pixel image. During training, we randomly sample 3-200 context points and use all 784 pixels as targets. At test time, we use 100 context points. Inputs are 2D pixel coordinates $( x _ { 1 } , x _ { 2 } )$ normalized $\mathrm { t o } \ [ - 1 , 1 ] ^ { 2 }$ . To simulate input uncertainty, we corrupt context inputs by adding independent Gaussian noise $\mathcal { N } ( \mathbf { 0 } , \sigma _ { x } ^ { 2 } \mathbf { I } _ { 2 } )$ where $\bar { \sigma _ { x } } \in \{ 0 . 0 0 1 , \bar { 0 . } 0 0 5 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . \bar { 2 } , 0 . 5 \}$ in the normalized input space. These noise levels correspond to average pixel displacements of 0.02–7.4 pixels. We train separate NP and DnNP models for each noise level.

Table 1 and Figure 3 summarize the results on MNIST image completion under input noise. DnNP outperforms Noisy NP in NLL and RMSE at every noise level, and in accuracy for $\sigma _ { x } ~ \le ~ 0 . 1$ demonstrating that explicitly modeling input noise leads to more accurate and better-calibrated predictions. Notably, at low to moderate noise levels, DnNP recovers performance close to the noise-free baseline. As noise increases further, accuracy degrades for all methods as expected, since highly corrupted coordinates make the spatial structure of the image increasingly difficult to recover; at the largest noise levels $( \sigma _ { x } \ge 0 . 2 )$ , accuracy drops close to chance for both models, and the two are comparable. Note that RMSE and NLL remain nearly constant across noise levels because they are dominated by pixel-intensity reconstruction, which changes little as coordinate noise grows, whereas classification accuracy depends directly on whether the spatial structure of the digit is preserved; accuracy is therefore the more discriminative metric for this task. Qualitatively, Figure 3 shows that

Table 1: MNIST results. Left: clean-input baselines $( \sigma _ { x } = 0 )$ . Right: performance under increasing input noise $\sigma _ { x }$ for Noisy NP vs. DnNP; bold marks the better of the two at each noise level.
<table><tr><td>Method</td><td>NLL↓</td><td>RMSE↓</td><td> $\operatorname { A c c } \uparrow$ </td></tr><tr><td>NP</td><td>-0.0631</td><td>0.1413</td><td>0.706</td></tr><tr><td>CNP</td><td>-0.00571</td><td>0.232</td><td>0.319</td></tr><tr><td>ANP</td><td>-0.00702</td><td>0.190</td><td>0.733</td></tr><tr><td>ConvCNP</td><td>-0.00454</td><td>0.213</td><td>0.576</td></tr></table>

<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="7"> $\sigma _ { x }$ </td></tr><tr><td>0.001</td><td>0.005</td><td>0.01</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.5</td></tr><tr><td rowspan="2">NLL↓</td><td>Noisy NP</td><td>-0.0349</td><td>-0.0398</td><td>-0.0403</td><td>-0.0400</td><td>-0.0401</td><td>-0.0393</td><td>-0.0380</td></tr><tr><td>DnNP</td><td>-0.0616</td><td>-0.0630</td><td>-0.0624</td><td>-0.0637</td><td>-0.0646</td><td>-0.0637</td><td>-0.0628</td></tr><tr><td rowspan="2">RMSE↓</td><td>Noisy NP</td><td>0.243</td><td>0.236</td><td>0.232</td><td>0.233</td><td>0.233</td><td>0.235</td><td>0.239</td></tr><tr><td>DnNP</td><td>0.148</td><td>0.142</td><td>0.145</td><td>0.138</td><td>0.134</td><td>0.139</td><td>0.143</td></tr><tr><td rowspan="2">Acc ↑</td><td>Noisy NP</td><td>0.205</td><td>0.207</td><td>0.221</td><td>0.210</td><td>0.215</td><td>0.193</td><td>0.157</td></tr><tr><td>DnNP</td><td>0.660</td><td>0.726</td><td>0.710</td><td>0.592</td><td>0.370</td><td>0.184</td><td>0.123</td></tr></table>

![](images/39740612c730703449cb20b8de76ea7733ce863dd6bdf4a3bf4057a8289f7874.jpg)  
Figure 3: MNIST image completion with noisy pixel coordinates $( \sigma _ { x } = 0 . 1 )$ at varying context sizes. Each row shows context, prediction, and per-pixel uncertainty for Clean NP $( \sigma _ { x } = 0$ , reference), Noisy NP, and DnNP. Noisy NP fails to recover the correct digit at all context sizes, while DnNP recovers digits comparable to Clean NP, with uncertainty concentrated along edges.

Noisy NP produces a persistent shape distortion regardless of context size, whereas DnNP recovers sharper and more coherent digit structures.

## 6.3 Spatial Interpolation

We collect hourly $\mathrm { P M } _ { 2 . 5 }$ data from 34 QuantAQ [20] air quality sensors located in Brookline, Massachusetts, USA, a town of approximately 17.6 km<sup>2</sup>. The dataset is publicly available at https://hdl.handle.net/2047/D20881761. The analysis period spans May 24 to September 19, 2024, and all sensors used had more than 90% data completeness over the study period. Missing values were filled using linear imputation. We apply a log transformation to the $\mathrm { P M } _ { 2 . 5 }$ data and min-max normalization to other predictor variables such as latitude, longitude, temperature, and humidity. The dataset is divided into three splits along both the temporal and spatial axes. Temporally, the data are sequentially partitioned into three consecutive segments for training, validation, and testing: the first 80% for training, the next 10% for validation, and the remaining 10% for testing. Spatially, sensors are randomly split into three non-overlapping groups—context, target, and heldout—at approximately 55%, 15%, and 30%, respectively. For the training and validation splits, the context sensors are used to construct the feature vector and the model is trained/evaluated on the target sensors. To assess generalization to unobserved locations, the test split uses context-sensor values to construct the feature vector and evaluates predictions on the held-out sensors.

Input noise for the temperature and relative humidity dimensions is derived from the MODULAIR™ sensor specifications [20], which report measurement accuracies of $\pm 0 . 2 ^ { \circ } \mathrm { C }$ for temperature and ±1.8% for relative humidity. Table 2 presents results on $\mathrm { P M } _ { 2 . 5 }$ spatial interpolation on the Air Quality dataset across three feature configurations. DnNP consistently outperforms standard NPs across all feature sets in both NLL and RMSE, and both improve over the mean baseline. These results demonstrate that explicitly accounting for noisy input is particularly beneficial in spatial interpolation settings, where sensor data uncertainty is a natural and unavoidable source of noise. We further verified that the denoiser does not collapse to the identity map: its corrections to the noisy inputs are non-zero across all feature configurations and seeds, and small relative to the assumed noise level, as expected when sensor noise is small compared to the spread of each feature.

Table 2: Spatial interpolation results on two real-world datasets, reported as mean $\pm$ standard deviation over 5 seeds. Mean RMSE is the constant-prediction baseline (predicting the dataset mean). Bold marks the better mean of NP vs. DnNP per metric within each row.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Feature Set</td><td rowspan="2">Mean RMSE↓</td><td colspan="2">NP</td><td colspan="2">DnNP (ours)</td></tr><tr><td>NLL↓</td><td>RMSE↓</td><td>NLL↓</td><td>RMSE↓</td></tr><tr><td rowspan="3">PM2.5 (Air Quality)</td><td>Lat, Lon, RH</td><td>0.776</td><td> $1 . 0 9 8 \pm 0 . 2 7 3$ </td><td> $0 . 4 5 7 \pm 0 . 0 3 6$ </td><td> $\mathbf { 0 . 2 9 6 \pm 0 . 0 7 8 }$ </td><td> $\mathbf { 0 . 3 2 0 \pm 0 . 0 2 3 }$ </td></tr><tr><td>Lat, Lon, Temp</td><td>0.776</td><td> $1 . 0 3 8 \pm 0 . 1 8 7$ </td><td> $0 . 4 3 9 \pm 0 . 0 3 0$ </td><td> $\mathbf { 0 . 3 6 3 \pm 0 . 0 8 9 }$ </td><td> $\mathbf { 0 . 3 3 7 \pm 0 . 0 5 5 }$ </td></tr><tr><td>Lat, Lon, RH, Temp</td><td>0.776</td><td> $0 . 9 2 5 \pm 0 . 1 6 1$ </td><td> $0 . 4 3 0 \pm 0 . 0 3 2$ </td><td> $\mathbf { 0 . 3 5 1 \pm 0 . 1 3 0 }$ </td><td> $\mathbf { 0 . 3 4 8 \pm 0 . 0 6 2 }$ </td></tr><tr><td rowspan="3">Pressure (NOAA CA 2026)</td><td>Lat, Lon, Temp</td><td>0.108</td><td> $\mathbf { - 1 . 2 4 0 \pm 0 . 0 3 2 }$ </td><td> $0 . 0 5 2 \pm 0 . 0 0 8$ </td><td> $- 1 . 2 3 9 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 0 3 6 \pm 0 . 0 0 6 }$ </td></tr><tr><td>Lat, Lon, RH</td><td>0.108</td><td> $- 1 . 0 9 1 \pm 0 . 0 6 4$ </td><td> $0 . 0 7 6 \pm 0 . 0 0 8$ </td><td> $\mathbf { - 1 . 2 2 9 \pm 0 . 0 3 6 }$ </td><td> $\mathbf { 0 . 0 3 9 \pm 0 . 0 1 0 }$ </td></tr><tr><td>Lat, Lon, Temp, RH</td><td>0.108</td><td> $- 1 . 1 7 1 \pm 0 . 0 3 1$ </td><td> $0 . 0 6 4 \pm 0 . 0 0 5$ </td><td> $\mathbf { - 1 . 2 0 5 \pm 0 . 0 6 7 }$ </td><td> $\mathbf { 0 . 0 4 8 \pm 0 . 0 1 5 }$ </td></tr></table>

We additionally evaluate DnNP on sea-level pressure interpolation across NOAA weather stations in California. We retrieve daily observations of sea-level pressure, temperature, and relative humidity via the Meteostat library [15] for Q1 2026, retaining 29 stations with $\geq 9 5 \%$ data completeness over the 90-day period. Missing values are linearly imputed, sea-level pressure is the prediction target, and predictor variables (latitude, longitude, temperature, humidity) are min-max normalized. We split the 90 days temporally into 80% training, 10% validation, and 10% test. Within each daily episode, sensors are randomly partitioned into context and target sets following the same protocol as the air quality experiment. Input noise standard deviations for temperature and relative humidity are derived from ASOS sensor specifications, which report typical accuracies of $\pm 0 . 5 ^ { \circ } \mathrm { C }$ for the platinum resistance thermometer and ±2% for the Vaisala capacitive humidity sensor. The corresponding rows of Table 2 report results across three input feature configurations. DnNP achieves lower RMSE than the NP in all three settings and lower NLL in two of three, with the largest gains under (Lat, Lon, RH) and (Lat, Lon, Temp, RH), where humidity enters as a noisy input. In the (Lat, Lon, Temp) setting, the two models are statistically tied in NLL $( - 1 . 2 4 0 \pm 0 . 0 3 2 \mathrm { v s . - 1 . 2 3 9 \pm 0 . 0 2 1 } )$ , while DnNP still reduces RMSE.

## 7 Conclusion

In this paper, we introduced DnNP, a new member of the Neural Process family that explicitly handles noisy input coordinates through a variational denoising mechanism. We evaluated DnNP on 1D GP regression, MNIST image completion, and two real-world $\mathrm { P M } _ { 2 . 5 }$ and sea-level pressure spatial interpolation datasets, achieving lower or comparable NLL and lower RMSE across different settings, with up to 30% RMSE reduction on the air quality dataset. We further provided theoretical guarantees showing that the Bayes-optimal predictor under noisy inputs belongs to the DnNP model class, while plug-in models that treat noisy inputs as clean incur an irreducible KL divergence gap. Natural future directions include combining DnNP with methods that handle noisy labels [21], particularly relevant for air quality datasets where both input covariates and target measurements are corrupted, and extending the framework to noise that is correlated across observations.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Institutes of Health (NIH) under grant 5R01HL167072, the National Science Foundation (NSF) under grant 2414652, and the iSUPER Impact Engine at Northeastern University.

## References

[1] Anna Allen, Stratis Markou, Will Tebbutt, James Requeima, Wessel P Bruinsma, Tom R Andersson, Michael Herzog, Nicholas D Lane, Matthew Chantry, J Scott Hosking, and Richard E Turner. End-to-end data-driven weather prediction. Nature, 641(8065):1172–1179, May 2025.

[2] R. James Cotton, Fabian H. Sinz, and Andreas S. Tolias. Factorized neural processes for neural processes: k-shot prediction of neural responses, 2020.

[3] Yann Dubois, Jonathan Gordon, and Andrew YK Foong. Neural process family. http: //yanndubs.github.io/Neural-Process-Family/, September 2020.

[4] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pages 1126–1135. PMLR, 06–11 Aug 2017.

[5] Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and S. M. Ali Eslami. Conditional neural processes. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 1704–1713. PMLR, 10–15 Jul 2018.

[6] Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S. M. Ali Eslami, and Yee Whye Teh. Neural processes. CoRR, abs/1807.01622, 2018.

[7] Jonathan Gordon, Wessel P. Bruinsma, Andrew Y. K. Foong, James Requeima, Yann Dubois, and Richard E. Turner. Convolutional conditional neural processes. In International Conference on Learning Representations, 2020.

[8] Timothy M. Hospedales, Antreas Antoniou, Paul Micaelli, and Amos J. Storkey. Meta-learning in neural networks: A survey. CoRR, abs/2004.05439, 2020.

[9] Daniel Jenson, Jhonathan Navott, Piotr Grynfelder, Mengyan Zhang, Makkunda Sharma, Elizaveta Semenova, and Seth Flaxman. Scalable spatiotemporal inference with biased scan attention transformer neural processes. In The 29th International Conference on Artificial Intelligence and Statistics, 2026.

[10] Saurav Jha, Dong Gong, Xuesong Wang, Richard E. Turner, and Lina Yao. The neural process family: Survey, applications and perspectives, 2023.

[11] Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, S. M. Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. CoRR, abs/1901.05761, 2019.

[12] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

[13] Christos Louizos, Xiahan Shi, Klamer Schutte, and Max Welling. The functional neural process, 2019.

[14] Andrew Mchutchon and Carl Rasmussen. Gaussian process training with input noise. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc., 2011.

[15] Meteostat. Meteostat Python library, version 2.1.4. https://dev.meteostat.net.

[16] Sumantrak Mukherjee, Mouad Elhamdi, George Mohler, David Antony Selby, Yao Xie, Sebastian Josef Vollmer, and Gerrit Großmann. Neural spatiotemporal point processes: Trends and challenges. Transactions on Machine Learning Research, 2025. Survey Certification.

[17] Tung Nguyen and Aditya Grover. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling, 2023.

[18] NOAA National Centers for Environmental Information. Surface weather station observations. https://www.ncei.noaa.gov.

[19] Shenghao Qin, Jiacheng Zhu, Jimmy Qin, Wenshuo Wang, and Ding Zhao. Recurrent attentive neural process for sequential data. CoRR, abs/1910.09323, 2019.

[20] QuantAQ, Inc. MODULAIR™ Product Specification Sheet. https://assets.quant-aq. com/downloads/spec-sheets/modulair.latest.pdf, 2025. Accessed: 2026.

[21] Chen Shapira and Dan Rosenbaum. Robust neural processes for noisy data, 2024.

[22] Aishwarya Venkataramanan and Joachim Denzler. Distance-informed neural processes. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[23] Timon Willi, Jonathan Masci, Jürgen Schmidhuber, and Christian Osendorfer. Recurrent neural processes. CoRR, abs/1906.05915, 2019.

[24] Christopher Williams and Carl Rasmussen. Gaussian processes for regression. In D. Touretzky, M.C. Mozer, and M. Hasselmo, editors, Advances in Neural Information Processing Systems, volume 8. MIT Press, 1995.

[25] World Meteorological Organization. Guide to Meteorological Instruments and Methods of Observation. Secretariat of the World Meteorological Organization, 1983.

## A Limitations

DnNP assumes measurement noise that is independent across observations, encoded by the pointwise denoiser $q _ { \psi } ( x _ { i } ^ { * } \mid x _ { i } ) $ ; spatially correlated sensor bias, documented in both of our application domains, is not captured, and a set encoder approximating the joint posterior over clean inputs is a natural extension. The method also relies on a user-specified noise model and prior over clean inputs, which reduces the purely data-driven character of NPs and opens the door to misspecification, although DnNP is robust to moderate misspecification of σ. Finally, DnNP inherits the capacity of its backbone; since the denoiser is architecture-agnostic, combining it with more expressive models such as TNPs is a promising direction.

## B Broader Impact

DnNP is designed for settings where input measurements are inherently noisy, a common condition in real-world sensor networks. Our spatial interpolation experiments on urban air quality data demonstrate that explicitly modeling input noise improves prediction accuracy in environmental monitoring applications. Improved interpolation in this setting can support exposure assessment, environmental justice analyses, and public health decision-making, particularly in cities that rely on sparse, low-cost sensor deployments where measurement uncertainty is unavoidable. More broadly, principled handling of input uncertainty is relevant to many scientific and engineering domains that depend on noisy measurements, including geosciences and climate modeling. We recommend that practitioners deploying DnNP validate the assumed input-noise model against sensor specifications or held-out calibration data.

## C Posterior Over Noise-Free Inputs

Here we provide the proofs for the posterior propositions including the Gaussian prior and a truncated uniform prior.

## C.1 Proof of Proposition 5.1

Proof. By Bayes’ rule,

$$
p ( x ^ { * } \mid x ) \propto p ( x \mid x ^ { * } ) p ( x ^ { * } ) .
$$

Using the generative model from Section 4,

$$
\begin{array} { r l r } { p ( x \mid x ^ { * } ) \propto \exp \Big ( - \frac { 1 } { 2 } ( x - x ^ { * } ) ^ { \top } \Gamma ^ { - 1 } ( x - x ^ { * } ) \Big ) , } & { } & { p ( x ^ { * } ) \propto \exp \Big ( - \frac { 1 } { 2 } ( x ^ { * } - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x ^ { * } - \mu ) \Big ) . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { p ( { \boldsymbol x } ^ { * } \mid { \boldsymbol x } ) \propto \exp \Big ( - \frac { 1 } { 2 } \Big [ ( { \boldsymbol x } ^ { * } - { \boldsymbol \mu } ) ^ { \top } { \boldsymbol \Sigma } ^ { - 1 } ( { \boldsymbol x } ^ { * } - { \boldsymbol \mu } ) + ( { \boldsymbol x } ^ { * } - { \boldsymbol x } ) ^ { \top } { \boldsymbol \Gamma } ^ { - 1 } ( { \boldsymbol x } ^ { * } - { \boldsymbol x } ) \Big ] \Big ) . } \end{array}
$$

Expanding the quadratic form in $x ^ { * }$ yields

$$
( x ^ { * } - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x ^ { * } - \mu ) + ( x ^ { * } - x ) ^ { \top } \Gamma ^ { - 1 } ( x ^ { * } - x ) = { x ^ { * } } ^ { \top } ( \Sigma ^ { - 1 } + \Gamma ^ { - 1 } ) x ^ { * } - 2 ( \Sigma ^ { - 1 } \mu + \Gamma ^ { - 1 } x ) ^ { \top } x ^ { * } + \mathrm { c o n s t . }
$$

Set

$$
\Lambda _ { \mathrm { p o s t } } : = \Sigma ^ { - 1 } + \Gamma ^ { - 1 } , \qquad \Sigma _ { \mathrm { p o s t } } : = \Lambda _ { \mathrm { p o s t } } ^ { - 1 } , \qquad h : = \Sigma ^ { - 1 } \mu + \Gamma ^ { - 1 } x .
$$

Then

$$
{ x ^ { * } } ^ { \top } \Lambda _ { \mathrm { p o s t } } { x ^ { * } } - 2 { h ^ { \top } } { x ^ { * } } = ( { x ^ { * } } - \mu _ { \mathrm { p o s t } } ( x ) ) ^ { \top } \Lambda _ { \mathrm { p o s t } } ( { x ^ { * } } - \mu _ { \mathrm { p o s t } } ( x ) ) - \mu _ { \mathrm { p o s t } } ( x ) ^ { \top } \Lambda _ { \mathrm { p o s t } } \mu _ { \mathrm { p o s t } } ( x ) ,
$$

where

$$
\mu _ { \mathrm { p o s t } } ( x ) : = \Sigma _ { \mathrm { p o s t } } h .
$$

Hence

$$
\begin{array} { r } { p ( x ^ { * } \mid x ) \propto \exp \Big ( - \frac { 1 } { 2 } ( x ^ { * } - \mu _ { \mathrm { p o s t } } ( x ) ) ^ { \top } \Sigma _ { \mathrm { p o s t } } ^ { - 1 } ( x ^ { * } - \mu _ { \mathrm { p o s t } } ( x ) ) \Big ) , } \end{array}
$$

which is the Gaussian density stated in Proposition 5.1.

## C.2 Bounded-Domain Extension

Proposition C.1 (Uniform box prior + Gaussian corruption). Let $x ^ { * } \in \mathbb { R } ^ { d }$ have a uniform prior on the box

$$
\begin{array} { r } { B = [ a _ { 1 } , b _ { 1 } ] \times \cdot \cdot \cdot \times [ a _ { d } , b _ { d } ] . } \end{array}
$$

Under the Gaussian corruption model $p ( x \mid x ^ { * } ) = \mathcal { N } ( x \mid x ^ { * } , \Gamma )$ , the posterior is the truncated multivariate normal

$$
p ( x ^ { * } \mid x ) = { \frac { \mathbf { 1 } _ { B } ( x ^ { * } ) } { Z ( B ; x , \Gamma ) } } \phi _ { d } ( x ^ { * } ; x , \Gamma ) ,
$$

where

$$
\phi _ { d } ( x ^ { * } ; x , \Gamma ) = \frac { 1 } { ( 2 \pi ) ^ { d / 2 } \operatorname * { d e t } ( \Gamma ) ^ { 1 / 2 } } \exp \Big ( - { \textstyle \frac { 1 } { 2 } } \big ( x ^ { * } - x \big ) ^ { \top } \Gamma ^ { - 1 } \big ( x ^ { * } - x \big ) \Big )
$$

and

$$
Z ( \mathcal B ; x , \Gamma ) = \int _ { \mathcal B } \phi _ { d } ( u ; x , \Gamma ) d u .
$$

Proof. The prior density is constant on $\boldsymbol { B }$ and zero outside, so Bayes’ rule gives

$$
p ( x ^ { * } \mid x ) \propto \mathbf { 1 } _ { B } ( x ^ { * } ) p ( x \mid x ^ { * } ) .
$$

Substituting the Gaussian corruption density yields

$$
p ( x ^ { * } \mid x ) \propto \mathbf { 1 } _ { \mathcal { B } } ( x ^ { * } ) \phi _ { d } ( x ^ { * } ; x , \Gamma ) .
$$

Normalizing over B gives the stated form.

## D ELBO Derivation

We derive the evidence lower bound (ELBO) for DnNP in four steps: first the full-data ELBO, then the context–target training objective.

Step 1: Marginal Likelihood. We aim to maximize the log-likelihood of the observed labels conditioned on the noisy inputs. Starting from the joint model (5), we marginalize over the latent variables z and $x ^ { * } { } _ { [ n ] }$

$$
p ( \boldsymbol { y } _ { [ n ] } \mid \boldsymbol { x } _ { [ n ] } ) = \iint p ( z ) p ( x ^ { * } { } _ { [ n ] } \mid \boldsymbol { x } _ { [ n ] } ) \prod _ { i = 1 } ^ { n } p ( \boldsymbol { y } _ { i } \mid x ^ { * } { } _ { i } , z ) d z d x ^ { * } ,\tag{21}
$$

where $p ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } ) = p ( x _ { [ n ] } \mid x ^ { * } { } _ { [ n ] } ) p ( x ^ { * } { } _ { [ n ] } ) / p ( x _ { [ n ] } )$ follows from Bayes’ rule.

Step 2: Introducing the Variational Family. Since the integral in (21) is intractable, we introduce the structured variational posterior from (6) by multiplying and dividing inside the integral:

$$
\begin{array} { r } { \log p ( y _ { [ n ] } \mid x _ { [ n ] } ) = \log \iint \frac { q _ { \psi } ( x ^ { * } [ n ] \mid x _ { [ n ] } ) q _ { \phi } ( z \mid x ^ { * } [ n ] , y _ { [ n ] } ) } { q _ { \psi } ( x ^ { * } [ n ] \mid x _ { [ n ] } ) q _ { \phi } ( z \mid x ^ { * } [ n ] , y _ { [ n ] } ) } } \\ { \times \ p ( z ) p ( x ^ { * } [ n ] \mid x _ { [ n ] } ) \prod _ { i = 1 } ^ { n } p ( y _ { i } \mid x ^ { * } { } _ { i } , z ) d z d x ^ { * } . } \end{array}\tag{22}
$$

Recognizing the double integral as an expectation under $q _ { \psi } q _ { \phi }$ and applying Jensen’s inequality (log $\breve { \mathbb { E } } [ \cdot ] \geq \breve { \mathbb { E } } [ \log \cdot ] )$ :

$$
\log p ( y _ { [ n ] } \mid x _ { [ n ] } ) \ge \mathbb { E } _ { q _ { \psi } q _ { \phi } } \left[ \log { \frac { p ( x ^ { * } { [ n ] } \mid x _ { [ n ] } ) } { q _ { \psi } ( x ^ { * } { [ n ] } \mid x _ { [ n ] } ) } } + \log { \frac { p ( z ) } { q _ { \phi } ( z \mid x ^ { * } { [ n ] } , y _ { [ n ] } ) } } + \sum _ { i = 1 } ^ { n } \log p ( y _ { i } \mid x ^ { * } { _ { i } } , z ) \right] .\tag{23}
$$

Step 3: Rewriting as KL Divergences. The first two terms in (23) are negative KL divergences by definition:

$$
\mathbb { E } _ { q _ { \psi } } \left[ \log \frac { p ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } ) } { q _ { \psi } ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } ) } \right] = - \mathrm { K L } \Big ( q _ { \psi } ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } ) \parallel p ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } ) \Big ) ,\tag{24}
$$

$$
\mathbb { E } _ { q _ { \phi } } \left[ \log \frac { p ( z ) } { q _ { \phi } ( z \mid x ^ { * } { } _ { [ n ] } , y _ { [ n ] } ) } \right] = - \mathrm { K L } \Big ( q _ { \phi } ( z \mid x ^ { * } { } _ { [ n ] } , y _ { [ n ] } ) \mid \mid p ( z ) \Big ) .\tag{25}
$$

Note that $p ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } )$ factors as $\textstyle \prod _ { i } p ( x ^ { * } { } _ { i } \mid x _ { i } )$ under the assumed independence structure, so the KL decomposes point-wise. Furthermore, expanding $p ( x ^ { * } { } _ { [ n ] } \mid x _ { [ n ] } ) \propto p ( x _ { [ n ] } \mid x ^ { * } { } _ { [ n ] } ) p ( x ^ { * } { } _ { [ n ] } )$ and absorbing the log $p ( x _ { [ n ] } \mid x ^ { * } { } _ { [ n ] } )$ term into the reconstruction objective yields the form stated in (7).

Step 4: Context–Target Split. For meta-learning we condition on the context set and predict targets. Splitting indices into context $C = \{ 1 , \dots , m \}$ and target $T = \{ m + 1 , \ldots , n \}$ , and replacing the prior $p ( z )$ with the context posterior $q _ { \phi } ( z \mid x ^ { * } c , y _ { C } )$ as the baseline (following the NP training objective (2)):

$$
\begin{array} { r l } & { \log p ( \boldsymbol { y } _ { T } \mid \boldsymbol { x } _ { [ n ] } , \boldsymbol { y } _ { C } ) \geq \mathbb { E } _ { \boldsymbol { q } _ { \psi } ( x ^ { * } \mid n ] } | \boldsymbol { x } _ { [ n ] } ) \left[ \mathbb { E } _ { \boldsymbol { q } _ { \phi } ( \boldsymbol { z } \mid x ^ { * } \mid n ] } \boldsymbol { y } _ { [ n ] } ) \displaystyle \sum _ { i \in T } \log p ( \boldsymbol { y } _ { i } \mid x ^ { * } , \boldsymbol { z } ) \right. } \\ & { \qquad \left. - \mathrm { K L } \Big ( \boldsymbol { q } _ { \phi } ( \boldsymbol { z } \mid x ^ { * } \mid n ] , \boldsymbol { y } _ { [ n ] } ) \parallel \boldsymbol { q } _ { \phi } ( \boldsymbol { z } \mid x ^ { * } , \boldsymbol { y } _ { C } ) \Big ) \right] } \\ & { \qquad - \mathrm { K L } \Big ( \boldsymbol { q } _ { \psi } ( x ^ { * } \mid n ] \mid x _ { [ n ] } ) \parallel p ( x ^ { * } \mid n ) \Big ) + \mathbb { E } _ { \boldsymbol { q } _ { \psi } } \displaystyle \sum _ { i = 1 } ^ { n } \log p ( x _ { i } \mid x ^ { * } , \boldsymbol { z } ) , } \end{array}\tag{26}
$$

which matches (8) in the main text.

## E Auxiliary Lemmas for Section 5

Here, we provide the lemmas that are used to prove the theorems and corollaries in Section 5.

Lemma E.1 (Gaussian cross-entropy). Let P be an absolutely continuous distribution on R with finite mean $\mu _ { P }$ and variance $v _ { P }$ . For $Q = \mathcal N ( m , \sigma ^ { 2 } )$ with density q, one has

$$
- \mathbb { E } _ { Y \sim P } [ \log q ( Y ) ] = \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) + \frac { v _ { P } + ( \mu _ { P } - m ) ^ { 2 } } { 2 \sigma ^ { 2 } } .\tag{27}
$$

Proof. Since

$$
- \log q ( Y ) = \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) + \frac { ( Y - m ) ^ { 2 } } { 2 \sigma ^ { 2 } } ,
$$

taking expectation under P gives

$$
- \mathbb { E } _ { P } [ \log q ( Y ) ] = \frac { 1 } { 2 } \log ( 2 \pi \sigma ^ { 2 } ) + \frac { \mathbb { E } _ { P } [ ( Y - m ) ^ { 2 } ] } { 2 \sigma ^ { 2 } } .
$$

Using the bias–variance decomposition

$$
\mathbb { E } _ { P } [ ( Y - m ) ^ { 2 } ] = \operatorname { V a r } _ { P } ( Y ) + ( \mathbb { E } _ { P } [ Y ] - m ) ^ { 2 } = v _ { P } + ( \mu _ { P } - m ) ^ { 2 }
$$

proves (27).

Lemma E.2 (Entropy upper bound at fixed variance). Let P be an absolutely continuous distribution on R with finite mean $\mu _ { P }$ and finite variance $v _ { P }$ . Then

$$
H ( P ) \leq { \frac { 1 } { 2 } } \log ( 2 \pi e v _ { P } ) ,\tag{28}
$$

with equality ifand only ifP is Gaussian with variance $v _ { P }$

Proof. Let $G = \mathcal { N } ( \mu _ { P } , v _ { P } )$ with density g. Since $\mathrm { K L } ( P \| G ) \geq 0$

$$
- H ( P ) - \mathbb { E } _ { P } [ \log g ( Y ) ] \geq 0 .
$$

Applying Lemma E.1 with $m = \mu _ { P }$ and $\sigma ^ { 2 } = v _ { P }$ gives

$$
- \mathbb { E } _ { P } [ \log g ( Y ) ] = \frac { 1 } { 2 } \log ( 2 \pi v _ { P } ) + \frac { v _ { P } } { 2 v _ { P } } = \frac { 1 } { 2 } \log ( 2 \pi e v _ { P } ) .
$$

Hence

$$
H ( P ) \leq { \frac { 1 } { 2 } } \log ( 2 \pi e v _ { P } ) .
$$

Equality holds if and only if $\mathrm { K L } ( P \| G ) = 0 , { \mathrm { i . e . } } P = G$ almost everywhere.

Lemma E.3 (Third-order Taylor expansion with Lipschitz Hessian). Let $f : \mathbb { R } ^ { d }  \mathbb { I }$ R be twice continuously differentiable and assume its Hessian is globally L-Lipschitz:

$$
\Vert \nabla ^ { 2 } f ( u ) - \nabla ^ { 2 } f ( v ) \Vert \leq L \Vert u - v \Vert \qquad f o r \ : a l l \ : u , v \in \mathbb { R } ^ { d } .
$$

Fix $u _ { 0 } \in \mathbb { R } ^ { d }$ , and set $g = \nabla f ( u _ { 0 } )$ and $H = \nabla ^ { 2 } f ( u _ { 0 } )$ . Then for every $h \in \mathbb { R } ^ { d }$

$$
f ( u _ { 0 } + h ) = f ( u _ { 0 } ) + g ^ { \top } h + \frac { 1 } { 2 } h ^ { \top } H h + r ( h ) ,\tag{29}
$$

where

$$
\lvert r ( h ) \rvert \leq \frac { L } { 6 } \lvert \lvert h \rvert \rvert ^ { 3 } .\tag{30}
$$

Proof. By the integral form of Taylor’s theorem,

$$
f ( u _ { 0 } + h ) = f ( u _ { 0 } ) + g ^ { \top } h + \int _ { 0 } ^ { 1 } ( 1 - t ) h ^ { \top } \nabla ^ { 2 } f ( u _ { 0 } + t h ) h d t .
$$

Subtracting $\begin{array} { r } { \frac { 1 } { 2 } h ^ { \top } H h = \int _ { 0 } ^ { 1 } ( 1 - t ) h ^ { \top } H h \ d a } \end{array}$ t gives

$$
r ( h ) = \int _ { 0 } ^ { 1 } ( 1 - t ) h ^ { \top } \big ( \nabla ^ { 2 } f ( u _ { 0 } + t h ) - H \big ) h d t .
$$

Hence

$$
\lvert r ( h ) \rvert \leq \int _ { 0 } ^ { 1 } ( 1 - t ) \| \nabla ^ { 2 } f ( u _ { 0 } + t h ) - H \| \| h \| ^ { 2 } d t \leq \int _ { 0 } ^ { 1 } ( 1 - t ) L t d t \| h \| ^ { 3 } = \frac { L } { 6 } \| h \| ^ { 3 } .
$$

Lemma E.4 (Gaussian moment bounds). Let $\Delta \sim \mathcal { N } ( 0 , S )$ in $\mathbb { R } ^ { d }$ with $S \succeq 0$ . For every integer $k \geq 1 ,$

$$
\mathbb { E } \| \Delta \| _ { 2 } ^ { k } \leq C _ { d , k } \| S \| ^ { k / 2 } ,\tag{31}
$$

where $C _ { d , k } : = \mathbb { E } \| G \| _ { 2 } ^ { k } < \infty$ for $G \sim \mathcal { N } ( 0 , I _ { d } )$

Proof. Write $\Delta = S ^ { 1 / 2 } G$ with $G \sim \mathcal { N } ( 0 , I _ { d } )$ . Then

$$
\| \Delta \| _ { 2 } = \| S ^ { 1 / 2 } G \| _ { 2 } \leq \| S ^ { 1 / 2 } \| \| G \| _ { 2 } = \| S \| ^ { 1 / 2 } \| G \| _ { 2 } .
$$

Raising both sides to the kth power and taking expectation yields

$$
\begin{array} { r } { { \mathbb { E } } \| \Delta \| _ { 2 } ^ { k } \leq \| S \| ^ { k / 2 } { \mathbb { E } } \| G \| _ { 2 } ^ { k } = C _ { d , k } \| S \| ^ { k / 2 } . } \end{array}
$$

Lemma E.5 (Odd symmetry for centered Gaussians). Let $\Delta \sim \mathcal { N } ( 0 , S )$ in $\mathbb { R } ^ { d }$ $I f h : \mathbb { R } ^ { d } $ R is measurable, integrable, and odd, i.e. $h ( - u ) = - h ( u )$ for all $u ,$ then

$$
\mathbb { E } [ h ( \Delta ) ] = 0 .\tag{32}
$$

In particular, for every $g \in \mathbb { R } ^ { d }$ and every symmetric $H \in \mathbb { R } ^ { d \times d }$

$$
\mathbb { E } \big [ ( g ^ { \top } \Delta ) ( \Delta ^ { \top } H \Delta ) \big ] = 0 .\tag{33}
$$

Proof. Since $\Delta$ is centered Gaussian, $\Delta \stackrel { d } { = } - \Delta$ . Therefore

$$
\begin{array} { r } { \mathbb { E } [ h ( \Delta ) ] = \mathbb { E } [ h ( - \Delta ) ] = - \mathbb { E } [ h ( \Delta ) ] , } \end{array}
$$

which implies (32). For (33), note that

$$
u \mapsto ( g ^ { \top } u ) ( u ^ { \top } H u )
$$

is odd because the linear factor changes sign and the quadratic factor does not.

## F Proofs for Section 5

In this section, we prove our theorems and corollaries.

## F.1 Proof of Theorem 5.3

By the definition of the DnNP predictive class,

$$
q _ { \theta ^ { \star } , \psi ^ { \star } } ^ { \mathrm { D n N P } } ( y \mid x , z ) = \int \mathcal { N } \big ( y \mid f _ { \theta ^ { \star } } ( x ^ { * } , z ) , \sigma _ { y } ^ { 2 } \big ) q _ { \psi ^ { \star } } ( x ^ { * } \mid x ) d x ^ { * } .
$$

Using the realizability assumptions (14),

$$
q _ { \theta ^ { \star } , \psi ^ { \star } } ^ { \mathrm { D n N P } } ( y \mid x , z ) = \int { \mathcal N } \big ( y \mid f ^ { \star } ( x ^ { \star } , z ) , \sigma _ { y } ^ { 2 } \big ) p ( x ^ { \star } \mid x ) d x ^ { \star } .
$$

The right-hand side is exactly the Bayes-optimal predictive conditional from Section $5 ,$ proving (15).   
The KL divergence therefore vanishes pointwise, and taking expectation over $( x , Z )$ gives (16).

## F.2 Proof of Theorem 5.2

Fix $( x , z )$ and abbreviate

$$
P : = P _ { x , z } , \qquad Q : = \mathcal { N } ( \boldsymbol { m } ( x , z ) , \sigma _ { y } ^ { 2 } ) , \qquad \mu _ { \star } : = \mu _ { \star } ( x , z ) , \qquad v _ { \star } : = v _ { \star } ( x , z ) , \qquad \boldsymbol { m } : = \boldsymbol { m } ( x , z ) .
$$

By the Bayes-optimal predictive conditional, P is a Gaussian mixture and is therefore absolutely continuous. Let q denote the density of $Q$

By definition,

$$
\begin{array} { r } { \mathrm { K L } ( P \| Q ) = - H ( P ) - \mathbb { E } _ { Y \sim P } [ \log q ( Y ) ] . } \end{array}
$$

Applying Lemma E.1 gives

$$
- \mathbb { E } _ { Y \sim P } [ \log q ( Y ) ] = \frac { 1 } { 2 } \log ( 2 \pi \sigma _ { y } ^ { 2 } ) + \frac { v _ { \star } + ( \mu _ { \star } - m ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } .
$$

Applying Lemma E.2 to $P$ yields

$$
H ( P ) \leq { \frac { 1 } { 2 } } \log ( 2 \pi e v _ { \star } ) .
$$

Combining the last two displays,

$$
\begin{array} { l } { { \displaystyle \mathrm { K L } ( P \| Q ) \geq - \frac { 1 } { 2 } \log ( 2 \pi e v _ { \star } ) + \frac { 1 } { 2 } \log ( 2 \pi \sigma _ { y } ^ { 2 } ) + \frac { v _ { \star } + ( \mu _ { \star } - m ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } } } \\ { { \displaystyle \quad = \frac { ( \mu _ { \star } - m ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \frac { 1 } { 2 } \left( \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } \right) . } } \end{array}
$$

The right-hand side is minimized over m by choosing $m = \mu _ { \star }$ , proving (13).

It remains to verify the strict-positivity claim. By (10),

$$
v _ { \star } = \sigma _ { y } ^ { 2 } + \mathrm { V a r } \big ( f ^ { \star } ( X , z ) \mid x \big ) .
$$

If Var $\left( f ^ { \star } ( X , z ) \mid x \right) > 0 .$ , then $v _ { \star } > \sigma _ { y } ^ { 2 }$ . Define

$$
\phi ( u ) : = u - 1 - \log u , \qquad u > 0 .
$$

Then

$$
\phi ^ { \prime } ( u ) = 1 - { \frac { 1 } { u } } , \quad \quad \phi ^ { \prime \prime } ( u ) = { \frac { 1 } { u ^ { 2 } } } > 0 .
$$

Hence $\phi$ is strictly convex and attains its unique minimum 0 at $u = 1$ . Therefore $\phi ( u ) > 0$ for every $u \ne 1$ , and in particular

$$
\frac { 1 } { 2 } \left( \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } \right) > 0 \qquad \mathrm { w h e n e v e r } v _ { \star } > \sigma _ { y } ^ { 2 } .
$$

This completes the proof.

## F.3 Proof of Lemma 5.4, Theorem 5.5, and Corollary 5.6

For brevity, write $\bar { x ^ { * } } : = \mu _ { \mathrm { p o s t } } ( x ) , S : = \Sigma _ { \mathrm { p o s t } }$ , and $f ( \cdot ) : = f ^ { \star } ( \cdot , z )$ . By Proposition 5.1,

$$
X \mid x \sim { \mathcal { N } } ( { \bar { x ^ { \ast } } } , S ) .
$$

Define

$$
\Delta : = X - \bar { x ^ { * } } , \qquad g : = \nabla f ( \bar { x ^ { * } } ) , \qquad H : = \nabla ^ { 2 } f ( \bar { x ^ { * } } ) .
$$

Then $\Delta \sim { \mathcal { N } } ( 0 , S )$

Since $\nabla ^ { 2 } f$ is globally Lipschitz, Lemma E.3 gives

$$
f ( \bar { x ^ { * } } + \Delta ) = f ( \bar { x ^ { * } } ) + g ^ { \top } \Delta + \frac { 1 } { 2 } \Delta ^ { \top } H \Delta + r ( \Delta ) ,\tag{34}
$$

with

$$
| r ( \Delta ) | \leq C \| \Delta \| ^ { 3 }\tag{35}
$$

for a constant C independent of $\Delta$

Proof of (17). Taking expectation in (34) yields

$$
\mathbb { E } [ f ( X ) \mid x ] = f ( \bar { x ^ { * } } ) + \mathbb { E } [ g ^ { \top } \Delta ] + \frac { 1 } { 2 } \mathbb { E } [ \Delta ^ { \top } H \Delta ] + \mathbb { E } [ r ( \Delta ) ] .
$$

Because $\Delta$ is centered Gaussian,

$$
\begin{array} { r } { \mathbb { E } [ g ^ { \top } \Delta ] = 0 , \qquad \mathbb { E } [ \Delta ^ { \top } H \Delta ] = \mathrm { t r } ( H S ) . } \end{array}
$$

Moreover, by (35) and Lemma E.4,

$$
| \mathbb { E } [ r ( \Delta ) ] | \leq C \mathbb { E } \| \Delta \| ^ { 3 } = O ( \| S \| ^ { 3 / 2 } ) .
$$

Therefore

$$
\mathbb { E } [ f ( X ) \mid x ] = f ( { \bar { x ^ { * } } } ) + { \frac { 1 } { 2 } } \mathrm { t r } ( H S ) + O ( \| S \| ^ { 3 / 2 } ) ,
$$

which proves (17).

Proof of (18). Define the centered quadratic and remainder terms

$$
\begin{array} { r } { Q ( \Delta ) : = \displaystyle \frac { 1 } { 2 } \Big ( \Delta ^ { \top } H \Delta - \operatorname { t r } ( H S ) \Big ) , \qquad R ( \Delta ) : = r ( \Delta ) - \mathbb { E } [ r ( \Delta ) ] . } \end{array}
$$

Then

$$
f ( X ) - \mathbb { E } [ f ( X ) \mid x ] = g ^ { \top } \Delta + Q ( \Delta ) + R ( \Delta ) .
$$

Hence

$$
\begin{array} { r l } & { { \mathrm { V a r } } ( f ( X ) \mid x ) = \operatorname { \mathbb { E } } \bigl [ ( g ^ { \top } \Delta ) ^ { 2 } \bigr ] + \operatorname { \mathbb { E } } \bigl [ Q ( \Delta ) ^ { 2 } \bigr ] + \operatorname { \mathbb { E } } \bigl [ R ( \Delta ) ^ { 2 } \bigr ] } \\ & { \qquad + 2 \operatorname { \mathbb { E } } \bigl [ g ^ { \top } \Delta Q ( \Delta ) \bigr ] + 2 \operatorname { \mathbb { E } } \bigl [ g ^ { \top } \Delta R ( \Delta ) \bigr ] + 2 \operatorname { \mathbb { E } } \bigl [ Q ( \Delta ) R ( \Delta ) \bigr ] . } \end{array}\tag{36}
$$

We now bound each term.

First,

$$
\mathbb { E } [ ( g ^ { \top } \Delta ) ^ { 2 } ] = g ^ { \top } S g .
$$

Second, since

$$
| Q ( \Delta ) | \leq { \frac { 1 } { 2 } } | \Delta ^ { \top } H \Delta | + { \frac { 1 } { 2 } } | \mathrm { t r } ( H S ) | \leq { \frac { 1 } { 2 } } \| H \| \| \Delta \| ^ { 2 } + { \frac { 1 } { 2 } } | \mathrm { t r } ( H S ) | ,
$$

we have

$$
\begin{array} { r } { \mathbb { E } [ Q ( \Delta ) ^ { 2 } ] = O \big ( \mathbb { E } \| \Delta \| ^ { 4 } \big ) + O ( \| S \| ^ { 2 } ) = O ( \| S \| ^ { 2 } ) , } \end{array}
$$

where Lemma E.4 was used in the last step.

Third, by (35),

$$
| R ( \Delta ) | \leq | r ( \Delta ) | + | \mathbb { E } [ r ( \Delta ) ] | \leq C \| \Delta \| ^ { 3 } + C \mathbb { E } \| \Delta \| ^ { 3 } .
$$

Therefore

$$
\begin{array} { r } { \mathbb { E } [ R ( \Delta ) ^ { 2 } ] = O \big ( \mathbb { E } \| \Delta \| ^ { 6 } \big ) = O ( \| S \| ^ { 3 } ) } \end{array}
$$

by Lemma E.4.

Fourth, by Lemma E.5,

$$
\mathbb { E } [ g ^ { \top } \Delta Q ( \Delta ) ] = \frac { 1 } { 2 } \mathbb { E } \big [ ( g ^ { \top } \Delta ) ( \Delta ^ { \top } H \Delta ) \big ] - \frac { 1 } { 2 } \mathrm { t r } ( H S ) \mathbb { E } [ g ^ { \top } \Delta ] = 0 .
$$

Fifth, by Cauchy–Schwarz,

$$
\left| \mathbb E [ g ^ { \top } \Delta R ( \Delta ) ] \right| \leq \left( \mathbb E [ ( g ^ { \top } \Delta ) ^ { 2 } ] \right) ^ { 1 / 2 } \left( \mathbb E [ R ( \Delta ) ^ { 2 } ] \right) ^ { 1 / 2 } = O ( \| S \| ^ { 1 / 2 } ) O ( \| S \| ^ { 3 / 2 } ) = O ( \| S \| ^ { 2 } ) .
$$

Sixth, again by Cauchy–Schwarz,

$$
\begin{array} { r } { \left| \mathbb { E } [ Q ( \Delta ) R ( \Delta ) ] \right| \leq \left( \mathbb { E } [ Q ( \Delta ) ^ { 2 } ] \right) ^ { 1 / 2 } \left( \mathbb { E } [ R ( \Delta ) ^ { 2 } ] \right) ^ { 1 / 2 } = O ( \| S \| ) O ( \| S \| ^ { 3 / 2 } ) = O ( \| S \| ^ { 5 / 2 } ) . } \end{array}
$$

Substituting these bounds into (36) gives

$$
\operatorname { V a r } ( f ( X ) \mid x ) = g ^ { \top } S g + O ( \| S \| ^ { 2 } ) ,
$$

which proves (18).

Proof of (19). Let $m _ { 0 } : = f ( \bar { x ^ { * } } )$ . By Theorem 5.2,

$$
\mathrm { K L } \big ( P _ { x , z } \parallel \mathcal { N } ( m _ { 0 } , \sigma _ { y } ^ { 2 } ) \big ) \geq \frac { ( \mu _ { \star } - m _ { 0 } ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } + \frac { 1 } { 2 } \left( \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } \right) .\tag{37}
$$

By (17),

$$
\mu _ { \star } - m _ { 0 } = \frac { 1 } { 2 } \mathrm { t r } ( H S ) + O ( \vert \vert S \vert \vert ^ { 3 / 2 } ) ,
$$

so

$$
\frac { ( \mu _ { \star } - m _ { 0 } ) ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } = \frac { 1 } { 8 \sigma _ { y } ^ { 2 } } \big ( \mathrm { t r } ( H S ) \big ) ^ { 2 } + o ( \| S \| ^ { 2 } ) .\tag{38}
$$

Next, let $a : = v _ { \star } - \sigma _ { y } ^ { 2 } . \mathbf { B } \mathbf { y }$ (18),

$$
a = g ^ { \top } S g + O ( \| S \| ^ { 2 } ) , \qquad a = O ( \| S \| ) .
$$

Write $u : = a / \sigma _ { y } ^ { 2 }$ . Then $u = O ( \left. S \right. )$ , and

$$
\frac { 1 } { 2 } \left( \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } \right) = \frac { 1 } { 2 } \big ( u - \log ( 1 + u ) \big ) .
$$

Since $u \to 0$ , Taylor expansion of $\log ( 1 + u )$ at 0 gives

$$
u - \log ( 1 + u ) = \frac { u ^ { 2 } } { 2 } + O ( u ^ { 3 } ) ,
$$

hence

$$
\frac { 1 } { 2 } \left( \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } - 1 - \log \frac { v _ { \star } } { \sigma _ { y } ^ { 2 } } \right) = \frac { 1 } { 4 \sigma _ { y } ^ { 4 } } ( g ^ { \top } S g ) ^ { 2 } + o ( \| S \| ^ { 2 } ) .\tag{39}
$$

Combining (37), (38), and (39) proves (19).

Proof of (20) and Corollary 5.6. (20) follows directly from (13) and the expansion (39). If $S = \tau ^ { 2 } I _ { d } .$ , then

$$
g ^ { \top } S g = \tau ^ { 2 } \| g \| _ { 2 } ^ { 2 } ,
$$

so the isotropic scaling stated in Corollary 5.6 follows immediately.

## G Experimental Detail

All experiments were run on a single NVIDIA H200 GPU on an internal computing cluster. The training and inference time of DnNP is approximately the same as that of standard NPs, since the additional denoiser is a small MLP applied per input and contributes negligible overhead relative to the encoder and decoder networks.

Table 3: Overview of datasets used in experiments. d denotes the input dimensionality, K the number of training tasks (episodes), Task specifies the type of problem (regression, image completion, or spatial interpolation), and Noise indicates whether inputs are naturally corrupted. MNIST and 1D GP datasets do not inherently contain input noise, so we add noise synthetically to simulate noisy measurements.
<table><tr><td>DATASET</td><td>K</td><td>d</td><td>TASK</td><td>NOISE</td></tr><tr><td>1D GP</td><td>10,000</td><td>1</td><td>REGRESSION</td><td>No</td></tr><tr><td>MNIST</td><td>70,000</td><td>2</td><td>IMAGE COMPL.</td><td>No</td></tr><tr><td>AIR QUALITY</td><td>2,856</td><td>4</td><td>SPATIAL INTERP.</td><td>YES</td></tr><tr><td>NOAA CA 2026</td><td>90</td><td>4</td><td>SPATIAL INTERP.</td><td>YES</td></tr></table>