# Generalization, memorization, and overfitting for difusion models trained in the lazy high-dimensional regime

Hugo Latourelle-Vigeant<sup>1</sup>, Sinho Chewi<sup>1</sup>, Aram-Alexandre Pooladian<sup>2</sup>, John Sous<sup>3</sup>, and Theodor Misiakiewicz<sup>1</sup>

<sup>1</sup>Department of Statistics and Data Science, Yale University <sup>2</sup>Institute for Foundations of Data Science, Yale University <sup>3</sup>Department of Applied Physics, Yale University

August 26, 2026

## Abstract

Modern score-based generative models have achieved remarkable empirical success in highdimensional tasks such as image, audio, and video synthesis. These models reduce distribution learning to a sequence of regression problems that, if solved exactly on finite data, would ultimately reproduce the training samples. Their ability to generalize must therefore arise from the implicit or explicit regularization during training. In this work, we develop a generative counterpart to the theory of benign overfitting and algorithmic regularization for overparameterized neural networks in the supervised lazy-training regime. We study denoising score matching in a vector-valued reproducing kernel Hilbert space with an inner-product kernel. In the proportional high-dimensional regime $n \asymp d ,$ we derive exact risk trajectories under gradient flow training. These trajectories exhibit three phases governed by qualitatively distinct estimators: a spectral estimator that generalizes, a pure-noise score with localized peaks that interpolate the training objective, and an empirical Bayes estimator that memorizes the data. We then analyze how these estimators combine along the reverse-time SDE and characterize the distribution of the resulting samples. The analysis reveals familiar mechanisms from supervised learning, including kernel linearization and self-induced regularization from the nonlinear part of the kernel, but also reveals a distinct phenomenology specific to generative modeling.

## Contents

Introduction 3   
1.1 The lessons from supervised learning . 4   
1.2 Summary of main results: lazy training for generative modeling 6   
1.3 Related work . 9   
1.4 Notation and organization . 10   
2 Kernel denoising score matching 11   
2.1 Denoising score matching 11   
2.2 The empirical objective and its memorizing optimum 12   
2.3 Inner-product kernels and the model class 13   
2.4 Kernel operators, gradient flow, and ridge regression 13   
3 Learning the score along the gradient flow trajectory 15   
3.1 Linearization of the kernel and localized bumps 16   
3.2 Phase I: linearized dynamics and generalization 19   
3.3 Phase II: vanishing error and overfitting 21   
3.4 Phase III: the memorization endpoint 22   
4 Generation: the implicit bias of the reverse process 24   
4.1 The learned reverse process and its linearized approximation 24   
4.2 Delocalization and pathwise linearization 25   
4.3 The Gaussian generated distribution 26   
5 Conclusion and future directions 27   
A Ridge-regularized denoising score matching 37   
A.1 Moderate regularization . 37   
A.2 Vanishing regularization: overfitting and memorization 38   
B Technical preliminaries 39   
B.1 Consequences of the Logarithmic Sobolev Inequality 39   
B.2 Deterministic equivalents for the resolvent matrix 43   
B.3 Power series decomposition of the RKHS 47   
C Linearization of the kernel and localized bumps 51   
C.1 Linearization of the kernels and Bayes denoiser 51   
C.2 Construction of the localized bumps 55   
D Proofs of the kernel ridge regression results 59   
D.1 Proof of Theorem 6 59   
D.2 Proof of Theorem 7 63   
E Proofs of the gradient flow results 65   
E.1 Proof of Theorem 1 65   
E.2 Proof of Theorem 2 71   
F The interpolation limit 73   
F.1 Convergence of the test error under truncation 73   
F.2 Instability of the test error at the ridgeless limit 75   
F.3 Test error of the empirical Bayes denoiser 82   
G Reverse process dynamics 86   
G.1 Properties of the linearized reverse process . 86   
G.2 Uniform linearization of the empirical kernel operator 89   
G.3 Proof of Theorem 4 . 94   
G.4 Proof of Theorem 5 . 97

## 1 Introduction

The goal of generative modeling is to learn, from n independent samples drawn from an unknown distribution $\pi _ { 0 }$ , a procedure that generates new samples distributed approximately as $\pi _ { 0 }$ . Modern generative models, most notably difusion models, have achieved striking success across a wide range of domains, including image, audio, video, and scientific data generation [HJA20, RBL<sup>+</sup>22, KPH<sup>+</sup>21, WJB<sup>+</sup>23, $\mathrm { A A D ^ { + } 2 4 } ]$ Remarkably, these models routinely operate in high-dimensional regimes, where the ambient dimension d ranges from thousands to millions and the number of training samples n is comparable to, or even smaller than, d. In such regimes, classical nonparametric density estimation is hopeless, as minimax rates deteriorate exponentially with dimension. Their success must therefore rely on implicit or explicit regularization induced by the model architecture and training algorithm. A central challenge for a statistical theory of generative modeling is therefore to characterize what these models learn under practical training algorithms, through which mechanisms, and over what training and data scales.

Difusion models [SDWMG15, SE19, HJA20, SSDK<sup>+</sup>21] are particularly attractive for such a theory, because they reduce generative modeling to a family of familiar regression tasks. We briefly review this reduction below<sup>1</sup>. Starting from the data distribution $\pi _ { 0 }$ , consider the Ornstein– Uhlenbeck process

$$
\mathrm { d } \pmb { x } _ { t } = - \pmb { x } _ { t } \mathrm { d } t + \sqrt { 2 } \mathrm { d } \pmb { B } _ { t } , \qquad \pmb { x } _ { 0 } \sim \pi _ { 0 } ,\tag{1}
$$

where $( B _ { t } ) _ { t \geq 0 }$ is a standard Brownian motion in $\mathbb { R } ^ { d }$ independent of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . This process progressively transforms the data distribution into Gaussian noise. Its marginal distribution $\mathbf { \mathcal { x } } _ { t } \sim \pi _ { t }$ admits the explicit representation

$$
\begin{array} { r } { \pmb { x } _ { t } \mid \pmb { x } _ { 0 } \stackrel { \mathrm { d } } { = } \rho _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { z } , \qquad \rho _ { t } : = e ^ { - t } , \quad \sigma _ { t } : = \sqrt { 1 - e ^ { - 2 t } } , \quad \pmb { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) , } \end{array}\tag{2}
$$

where $_ { z }$ is independent of $\begin{array} { r } { \mathbf { \delta } x _ { 0 } . \mathrm { ~ A s ~ } t  \infty , } \end{array}$ π<sub>t</sub> converges exponentially fast to the standard Gaussian distribution. Consequently, for a suficiently large terminal time $T$ such that $\pi _ { T } \approx \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ ， approximate samples from $\pi _ { 0 }$ can be generated by initializing $\pmb { x } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and simulating from $T$ to 0 the reverse-time SDE [And82, HP86]

$$
\mathrm { d } \pmb { x } _ { t } = \left( - \pmb { x } _ { t } - 2 \nabla \log \pi _ { t } ( \pmb { x } _ { t } ) \right) \mathrm { d } t + \sqrt { 2 } \mathrm { d } \bar { B } _ { t } , \qquad \pmb { x } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) ,\tag{3}
$$

where $\bar { B } _ { t }$ is a standard reverse-time Brownian motion.

Simulating (3) requires estimating the score ∇ log $\pi _ { t }$ at every noise level t from the training samples $\left( \pmb { x } _ { 0 , i } \right) _ { i \leq n }$ . By Tweedie’s formula, this can be accomplished by writing the score estimator as $\widehat S _ { t } : = \widehat f _ { t } / { \sigma _ { t } } ^ { - }$ and fitting the denoiser $\widehat { \pmb f } _ { t }$ over a model class $\mathcal { F } _ { d }$ by minimizing the empirical denoising score matching (DSM) objective [HD05, Vin11]:

$$
\widehat { \pmb f } _ { t } = \underset { \pmb f \in \mathscr F _ { d } } { \arg \operatorname* { m i n } } \frac { 1 } { d n } \sum _ { i = 1 } ^ { n } \mathbb E _ { \pmb z \sim \mathscr N ( \ b 0 , \mathbf I _ { d } ) } \left[ \left\| \pmb f ( \rho _ { t } \pmb x _ { \ b 0 , i } + \sigma _ { t } \pmb z ) + \pmb z \right\| _ { 2 } ^ { 2 } \right] .\tag{4}
$$

Difusion models thus reduce generative modeling to a sequence of supervised learning problems indexed by the noise level t, whose solutions are subsequently composed through the stochastic dynamics (3) to produce the generated distribution $\widehat { \pi } _ { \mathrm { g e n } }$

Despite its algorithmic simplicity, the statistical properties of denoising score matching are far from straightforward and, at first sight, appear incompatible with generalization. Indeed, when the model class is suficiently rich—as is typically the case for overparameterized neural networks—and

the empirical objective (4) is minimized exactly, the resulting estimator coincides with the empirical Bayes denoiser for the Gaussian-smoothed empirical distribution

$$
\widehat { \pmb f _ { t } ^ { \star } } = \sigma _ { t } \nabla \log \widehat { \mu } _ { d , n } , \qquad \widehat { \mu } _ { d , n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \cal N } ( \rho _ { t } { \pmb x } _ { 0 , i } , \sigma _ { t } ^ { 2 } { \bf I } _ { d } ) .\tag{5}
$$

As $t \to 0 ^ { + }$ , and hence $\sigma _ { t }  0 , \widehat { \mu } _ { d , n }$ concentrates around the training samples, and the reverse-time dynamics reduce to nearest-neighbor retrieval. Exact minimization of (4) thus leads to memorization and not generalization [Pid22, BBBM24, BDK<sup>+</sup>25].

In practice, however, trained difusion models often generalize: they produce samples that are genuinely new while sharing global and local statistics of the data. Empirically, memorization recedes as the training set grows [KGSM24, ZZL<sup>+</sup>24, GDP<sup>+</sup>25], sets in only after training for long times [BUBM26, FSW25], and, most strikingly, models can overfit their training objective while continuing to generate novel samples [KK26]. What prevents the learned score from fitting the empirical Bayes denoiser? Which components of the model class and of the training algorithm act as implicit regularizers? And what distribution does the sampler generate? A growing body of work has investigated these questions from complementary perspectives, reviewed in Section 1.3. Our aim in this paper is to develop a high-dimensional theory that makes the underlying mechanisms explicit and delineates several concepts that are often conflated in the literature: overfitting the objective, memorizing the samples, generalizing to the population objective, and generalizing to the target distribution (generated distribution $\widehat { \pi } _ { \mathrm { g e n } }$ close to $\pi _ { 0 } )$

In this paper, we focus on the canonical setting of kernel methods with inner-product kernels, which describes wide neural networks trained in the lazy regime [JGH18, COB19] and for which the supervised learning phenomenology is understood in fine detail. The comparison to the supervised setting is instructive in both directions: several mechanisms carry over to generative modeling, while others lead to vastly diferent conclusions. We first review the main insights from supervised learning and then summarize our results.

## 1.1 The lessons from supervised learning

A central puzzle in deep learning is why highly expressive models trained by empirical risk minimization (ERM), often without explicit regularization, nevertheless generalize. Consider a regression problem with n samples $( { \pmb x } _ { i } , y _ { i } ) _ { i \leq n }$ where $\pmb { x } _ { i } \in \mathbb { R } ^ { d }$ and $y _ { i } = f _ { \star } ( \pmb { x } _ { i } ) + \varepsilon _ { i }$ with mean-zero, independent label noise $\varepsilon _ { i } .$ . Given a class $\mathcal { F }$ of functions $f : \mathbb { R } ^ { d }  \mathbb { R }$ , the empirical risk minimizer is

$$
\widehat { f } = \underset { f \in \mathcal { F } } { \arg \operatorname* { m i n } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( f ( \pmb { x } _ { i } ) - y _ { i } ) ^ { 2 } .\tag{6}
$$

Without capacity control, minimizing the empirical risk fits not only the signal $f _ { \star }$ but also the noise in the observed labels, and classical theory predicts poor out-of-sample behavior. Yet overparameterized models trained in practice often generalize well. A decade of work has isolated two key mechanisms to resolve this apparent paradox: benign overfitting [BLLT20, LR20, GMMM21, TB23, HMRT22] and algorithmic regularization $[ \mathrm { S H N ^ { + } 1 8 } , \mathrm { A K T 1 9 } ]$

These mechanisms are understood particularly sharply for neural networks trained in the socalled lazy regime [JGH18, COB19], in which the dynamics of a wide, fully connected network are efectively those of a kernel method with an inner-product kernel<sup>2</sup>

$$
K ( \pmb { x } , \pmb { x } ^ { \prime } ) = h ( \langle \pmb { x } , \pmb { x } ^ { \prime } \rangle / d ) .\tag{7}
$$

The corresponding kernel ridge estimator admits the explicit representation

$$
\widehat { f } = \underset { f \in \mathcal { H } } { \arg \operatorname* { m i n } } \left\{ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( f ( \pmb { x } _ { i } ) - y _ { i } ) ^ { 2 } + \frac { \kappa } { n } \| f \| _ { \mathcal { H } } ^ { 2 } \right\} = k _ { n } ^ { \top } ( \pmb { K } + \kappa \mathbf { I } _ { n } ) ^ { - 1 } \pmb { y } ,
$$

where H is the reproducing kernel Hilbert space (RKHS) associated with K, $\pmb { K } = ( K ( \pmb { x } _ { i } , \pmb { x } _ { j } ) ) _ { i , j \in [ n ] }$ is the kernel matrix, and $\pmb { k } _ { n } = ( K ( \cdot , \pmb { x } _ { i } ) ) _ { i \in [ n ] }$ is the vector of kernel evaluations at a test point. The empirical risk minimizer (6) over H corresponds to the interpolation limit $\kappa  0 ^ { + }$ , in which the estimator fits the training labels exactly whenever the kernel matrix is invertible.

Consider now the high-dimensional proportional regime $n \asymp d .$ Suppose that the covariates have mean zero, covariance $\mathbb { E } [ { \pmb { x } } { \pmb { x } } ^ { \top } ] = { \pmb { \Sigma } }$ , and satisfy appropriate concentration assumptions, such as Assumption 2. For simplicity of presentation, suppose also that $h ( 0 ) = h ^ { \prime \prime } ( 0 ) = 0$ and $h ^ { \prime } ( 0 ) = 1$ Several important phenomena emerge in this setting.

Kernel linearization. The kernel matrix is approximated in operator norm by a linear function of the sample covariance matrix, together with an additive diagonal correction [Kar10]:

$$
{ \pmb K } = { \pmb X } ^ { \mathsf { T } } { \pmb X } / d + \gamma _ { d } { \mathbf I } _ { n } + o _ { \parallel \cdot \parallel _ { \infty } , \mathbb P } ( 1 ) , \qquad \gamma _ { d } = h ( \mathrm { T r } ( \pmb \Sigma ) / d ) - \frac { \mathrm { T r } ( \pmb \Sigma ) } { d } ,
$$

where $\pmb { X } = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ] \in \mathbb { R } ^ { d \times n }$ . At a fresh test point x, the kernel vector similarly satisfies $\pmb { k } _ { n } ( \pmb { x } ) \approx \pmb { X } ^ { \top } \pmb { x } / d .$ The nonlinear part of the kernel survives on the training set only through the diagonal correction $\gamma _ { d } \mathbf { I } _ { n } .$ , whereas the kernel behaves linearly at a typical test point. This follows from high-dimensional near-orthogonality, $\langle { \pmb x } , { \pmb x } ^ { \prime } \rangle / d = O _ { \mathbb { P } } ( d ^ { - 1 / 2 } )$ , for independent samples.

Benign overfitting. At a test point, the fitted model is asymptotically equivalent to the linear ridge predictor

$$
\widehat { f } ( \pmb { x } ) \approx \pmb { x } ^ { \top } \pmb { X } ^ { \top } \big ( \pmb { X } ^ { \top } \pmb { X } + d ( \kappa + \gamma _ { d } ) \mathbf { I } _ { n } \big ) ^ { - 1 } \pmb { y } ,\tag{8}
$$

where the nonlinear part of the kernel acts as an additive self-induced ridge $\gamma _ { d }$ [LR20, BMR21, MM24]. Thus, even in the interpolation limit $\kappa = 0 ^ { + }$ , the predictor retains a nonvanishing efective regularization and can overfit benignly. The interpolating model decomposes into a regular, linear component that captures the signal and generalizes, and a localized, spiky component that fits the training residuals while remaining invisible at test time.

Limitations of the kernel regime. Although the RKHS of K may be universal at any fixed $d ,$ its efective expressive power in high dimension is sharply limited. In the proportional regime $n \asymp d ,$ an isotropic inner-product kernel can learn only the linear component of the target; more generally, with $n \asymp d ^ { \ell }$ samples it recovers at most a degree-ℓ polynomial approximation [GMMM21, MMM22]. Escaping this polynomial barrier requires feature learning or architectural structure such as locality and weight sharing [GMMM20, MM22].

Given the explanatory success of these results, and the reduction of difusion models to regression through the objective (4), it is natural to ask for an analogous theory in generative modeling. Three diferences, however, make denoising score matching a genuinely distinct problem.

First, the regression problem (6) is replaced by the denoising problem (4) whose unrestricted empirical minimizer is the empirical Bayes denoiser $\widehat { f } _ { t } ^ { \star }$ of (5), not the population score of $\pi _ { t }$ Around this empirical target, the regression problem is efectively noiseless: we show that the residual denoising noise is exponentially small in d (Proposition 2), and the classical rationale for early stopping—avoiding overfitting the noise—does not apply. Second, interpolation—driving the training objective to zero—is not the benign endpoint it can be in supervised learning: the exact minimizer is the memorizing score. Whatever generalization occurs must arise from the gap between the trained model $\widehat { \pmb f _ { t } }$ and the empirical minimizer $\widehat { f } _ { t } ^ { \star }$ . Third, a difusion model is not one regression problem but a sequence of them, indexed by the noise level t, and the fitted denoisers are composed through the nonlinear, random dynamics (3). Small score risk at each level transfers to the generated distribution by now-standard sampling theory $[ \mathrm { C C L ^ { + } 2 3 }$ , BBDD24]; but in the regimes of interest here the score risk is not small, and the generated distribution must be understood directly from the estimator geometry along the reverse trajectories.

These diferences make the theory of denoising score matching both conceptually and technically distinct from its supervised counterpart, and, as we shall see, they overturn several of its conclusions.

## 1.2 Summary of main results: lazy training for generative modeling

We study denoising score matching over the vector-valued RKHS induced by the inner-product kernel (7), with an independent model at each noise level. We work in the proportional highdimensional regime $n \asymp d$ and assume that the data distribution is centered with covariance Σ and satisfies a log-Sobolev inequality. Our primary estimator, denoted by ${ \widehat { f } } _ { \mathrm { t } } ,$ , is obtained by running gradient flow on the empirical objective from zero initialization for training time t (reserving t for difusion time). This gradient-flow formulation idealizes the practical training of difusion models by stochastic gradient descent with fresh noise resampled at each step. A parallel theory for the ridge estimator, which exhibits the same phenomenology under the correspondence $\kappa = d / \mathrm { t }$ , is developed in Appendix A.

We establish sharp characterizations for the score-estimator trajectories, their training and test DSM errors, and the sampling distribution obtained by combining these estimators through the reverse difusion process. In particular, our theory provides the following analogues of the phenomena described above in supervised learning.

Kernel linearization. With $\mathcal { F } _ { d }$ the vector-valued RKHS of an inner-product kernel, the objective (4) is a kernel regression problem in which the $n \times n$ kernel matrix is replaced by an infinite-dimensional kernel operator $\widehat { \kappa }$ on the noisy empirical space, with the n data points replaced by $n$ Gaussian tubes $\{ \rho _ { t } \pmb { x } _ { 0 , i } + \sigma _ { t } \pmb { z } \}$ We prove an operator-level analogue of El Karoui’s linearization (Proposition 1): $\widehat { \kappa }$ is approximated by an efective operator $\widehat { \mathcal { K } } _ { \mathrm { e f f } }$ that is linear except for a per-sample correction of strength

$$
\delta _ { n } : = \frac { h ( \rho _ { t } ^ { 2 } \tau _ { \Sigma } ) - \rho _ { t } ^ { 2 } \tau _ { \Sigma } } { n } , \qquad \tau _ { \Sigma } : = \mathrm { T r } ( \Sigma ) / d ,
$$

a self-induced regularization induced by the nonlinear part of the kernel. Unlike in supervised learning, this correction is not a ridge penalty: it acts by averaging the estimator over each tube. Thus, the learned score efectively decomposes into a linear map and localized nonlinear bumps around each training sample. These bumps allow the estimator to interpolate the residuals in the DSM objective while being invisible at test time and generation.

Benign overfitting? We delineate three related but distinct notions: (O) overfitting, in which the empirical denoising objective is driven to zero; (M) memorization, in which the generated distribution collapses onto the training samples; and $\left( G ^ { \mathsf { c } } \right)$ failing to generalize to the population distribution. By propagating the score estimators through the reverse dynamics (3), we show that $( O ) \not \Rightarrow ( M )$ : even when the training loss vanishes, the generative model need not reproduce training samples. At the same time, the onsets of (O) and $\left( G ^ { \mathsf { c } } \right)$ nearly coincide:

the training loss vanishes by forgetting the covariance of the data and the test-time score degrades to the score of pure noise. This supports the assertion that overfitting in difusion models is generally not benign [FDDS26].

Limitations of the kernel regime. For lazily trained denoisers, the generated distribution is asymptotically Gaussian, with a covariance we characterize exactly as a function of the covariance Σ, the sample ratio $n / d ,$ the self-induced regularization $\delta _ { n } ,$ , and the training and sampling schedules. This is the generative analogue of the polynomial barrier: at proportional scales, lazy difusion models generate at most a Gaussian approximation to the data. Escaping it requires feature learning, architectural structure, or genuinely low-dimensional data, which suggests natural next steps for future research [KGSM24, KG24, SFW25, CPL26, BMG26].

Although prior work has also studied denoising score matching asymptotics in the proportional regime [CZ23, GVM25, BUBM26, WP26, WZVP26, MG26a], we emphasize that the conclusions of the second and third bullet points above require going beyond studying the DSM objectives in isolation and ask about the reverse dynamics (3) directly. This is a key novelty of our analysis. In particular, our results allow us to study the interplay between generalization to the population objective and generalization to the target distribution. In Figure 3, we compare several early-stopping schedules for gradient-flow-trained score estimators and evaluate $\mathrm { K L } ( \pi _ { 0 } \| \widehat { \pi } _ { \mathrm { g e n } } )$ at diferent stages of the reverse SDE. We find that, when the reverse SDE is stopped early, a training schedule with suboptimal test error can yield a slightly smaller KL divergence than the optimal early-stopping schedule, which minimizes the population DSM objective along the gradient-flow trajectory at every noise level. We believe that clarifying the relationship between score-matching performance and generation quality is an important direction for future work, and that future theory should study the generated distribution directly rather than focus exclusively on the DSM loss.

## 1.2.1 Algorithmic regularization via early stopping

At the center of our investigation is a precise characterization of the gradient-flow trajectory and the implicit regularization induced by early stopping. This analysis identifies three distinct optimization regimes:

Linear regime $\mathfrak { t } = O ( d )$ . On this timescale, the kernel operator linearizes and the learned denoiser decomposes into a global linear component and localized bumps around the training samples, corresponding to two spectral channels. The first arises from the linear part of the kernel and relaxes on the timescale d; through this channel, the global component learns the spectrum of the sample covariance, top directions first. The second arises from the nonlinear part of the kernel and relaxes on the timescale $1 / \delta _ { n } \asymp n ;$ through this channel, the model begins to fit the per-sample bumps. We derive exact deterministic equivalents for the training and test errors along the entire trajectory (Theorem 1). The test risk is minimized at ${ \sf t } ^ { \star } \asymp d ,$ where the estimator implements a linear shrinkage of the noisy sample covariance. Past t<sup>⋆</sup>, the fit overshoots: interpolation forces the linear component toward the pure-noise score, and the model progressively forgets the covariance it had learned.

Polynomial regime $\mathbf { t } = d ^ { 1 + \Theta ( 1 ) }$ . On polynomially longer timescales, the kernel’s high-degree components fit increasingly sharp localized bumps around the training samples, and the empirical denoising loss vanishes. Note that interpolation here is far more demanding than in supervised learning, since the model must interpolate an entire function on n Gaussian tubes, not n scalar labels. At the same time, at a typical test point, the estimator is simply the pure-noise linear score and the population risk plateaus at $\frac { \rho _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \tau _ { \Sigma }$ (Theorem 2). The model has overfitted its objective; it has not memorized its samples: the reverse process, as we discuss below, still generates a Gaussian law.

![](images/aa048d0e3acfd4cd0da9956407210d543a797cfadbdacca85d993b27ef5f5d7e.jpg)  
Figure 1: Stylized evolution of the empirical denoising loss $\mathcal { L } ( \widehat { f } _ { \mathrm { t } } ; 0 )$ and population risk $\mathcal { R } ( \widehat { f } _ { \mathrm { t } } )$ along gradient flow at fixed noise level $( \rho , \sigma )$ , with $\tau _ { \Sigma } = \mathrm { T r } ( \Sigma ) / d .$ As training time increases, the score estimator transitions from a spectral shrinkage estimator at $\mathfrak { t } = O ( d )$ (Theorem 1), to fitting the pure-noise score at polynomial times $\mathbf { t } = d ^ { 1 + \Theta ( 1 ) }$ <sup>)</sup>, with localized bumps that drive the training loss to zero while remaining invisible on typical test inputs (Theorem 2), and finally to the empirical Bayes denoiser which memorizes the training samples at super-polynomial times (Theorem 3). In this figure, generalization means that the train and test objectives are close.

Super-polynomial regime ${ \sf t } = d ^ { \omega ( 1 ) }$ . Only on super-polynomial timescales does gradient flow converge to the empirical Bayes denoiser (Theorem 3). The population risk doubles to $2 \frac { \rho _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \tau _ { \Sigma }$ and the reverse process collapses onto the training set. This agrees with the empirical observation that memorization in difusion models emerges only at very long training times [BUBM26, FSW25, KK26].

The qualitative behavior of the train and test error trajectories at fixed difusion time t is summarized in the stylized diagram of Figure 1.

This hierarchy mirrors the observation that difusion models learn coarse, global statistics before fitting increasingly fine, and eventually sample-specific, details [WP26, BMG26, BUBM26, FSW25]. Its distinctive feature is the separation of the onset of interpolation (polynomial time) from the onset of memorization (super-polynomial time): between the two lies an entire regime in which the training loss is vanishing, the score risk is large, and the model generates neither the data distribution nor the training set.

## 1.2.2 Generation: delocalization and the Gaussian barrier

We next analyze how the learned denoisers compose along the reverse difusion process (3).

Linearization of the reverse SDE. Our main result (Theorem 4) is a pathwise linearization: for any training schedule $\mathrm { t } _ { t } \leq d ^ { 1 + c }$ with $c < 1 / 6$ (covering the generalization phase and the onset of overfitting), the Kullback–Leibler divergence between the path measures induced by the learned scores and their linearized surrogates vanishes asymptotically. The mechanism is high-dimensional delocalization: with high probability, a reverse trajectory has normalized overlap $O ( d ^ { - 1 / 2 } )$ with every training sample, uniformly over the trajectory, while the fitted nonlinear bumps activate only within narrow neighborhoods of those samples. Thus, along a typical generative trajectory, the learned kernel scores are indistinguishable from their linearized counterparts.

Gaussian inductive bias. As a consequence of this pathwise linearization, the generated distribution is asymptotically Gaussian, with covariance given by an explicit deterministic equivalent (Theorem 5): a Marchenko–Pastur-type map applied to Σ, composed with an ODE that tracks the training schedule and sampler early stopping. The generated law depends on $\pi _ { 0 }$ only through its first two moments, a universality principle for lazy difusion models. Sample-specific nonlinear corrections can interpolate the training objective, but they are invisible to typical reverse trajectories and therefore do not afect the limiting generated law. This provides a mechanism for the efectiveness of linear and Gaussian approximations to trained difusion models reported in [WV23, WV24, LDQ24]. All information in $\pi _ { 0 }$ beyond second order is lost in the lazy-training model, even though the underlying function class is universal and the trained model may interpolate the empirical denoising objective.

Our results delimit what isotropic lazy architectures can achieve in proportional high-dimensional regimes. Non-Gaussian generation requires leaving this regime through feature learning, architectural anisotropy—such as convolution, attention, or locality—or genuinely low-dimensional data structure. This is the generative analogue of the polynomial barrier for kernel methods in supervised learning: when $n \asymp d$ , lazy supervised models learn at most linear functions of their inputs, while the lazy difusion models studied here generate at most Gaussian distributions.

## 1.3 Related work

Score matching was introduced by Hyv¨arinen [HD05] as a way to fit unnormalized models by matching gradients of log-densities rather than normalizing constants, and Vincent [Vin11] established its denoising formulation as the regression (4). Difusion-based generative modeling originates with Sohl-Dickstein et al. [SDWMG15]; its modern score-based formulations were developed by Song and Ermon [SE19], Ho, Jain, and Abbeel [HJA20], and Song et al. $[ \mathrm { S S D K ^ { + } 2 1 } ]$ , with design refinements surveyed in [KAAL22].

A growing body of research develops sampling and statistical guarantees for difusion models. A first line takes a score estimate of prescribed $L ^ { 2 }$ accuracy as given and controls the resulting sampler: the generated law is close to $\pi _ { 0 }$ whenever the score risk is small, with polynomial dependence on the dimension, with recent improvements to linear dependence on the intrinsic dimensionality of the data [CCL<sup>+</sup>23, LLT23, BBDD24, CDS25, Bor22, CCDR26]. A second line of work studies the task of score estimation itself: neural network estimators achieve minimax rates for densities with smoothness or low-dimensional structure [OAS23, CHZW23, ZYLL24], and optimal score estimation is closely tied to empirical Bayes smoothing [WWY24, DKXZ24]. These frameworks presuppose that training succeeds at controlling the population score risk, typically as $n \to \infty$ at fixed (or efectively low) dimension, with statistical rates subject to the curse of dimensionality. By contrast, in the proportional regime $n \asymp d ,$ consistent score estimation is impossible and our paper instead characterizes what the training algorithm actually learns in this regime. This requires analyzing the reverse process pathwise rather than a reduction to risk bounds.

That difusion models can replicate training data was established by $\mathrm { [ C H N ^ { + } 2 3 , ~ S S G ^ { + } 2 3 a }$ $\mathrm { S S G ^ { + } 2 3 b } ]$ ]. The transition between memorization and generalization has since been mapped along several axes, such as dataset size [YCKR23, GDP<sup>+</sup>25, ZZL<sup>+</sup>24, KGSM24], training time [BUBM26, FSW25, KK26], model capacity [BPMB26, YZTC26], and local properties of the underlying data density [MG26b, RKW<sup>+</sup>25]. On the theoretical side, a line of work in statistical physics analyzes the reverse process driven by the exact empirical score: it identifies speciation and collapse transitions along the reverse dynamics and shows that avoiding collapse onto training points requires n to be exponential in the dimensions d [BM23, BBBM24, RA23, AAL<sup>+</sup>25, AVS<sup>+</sup>24, LC24].

Closest to this paper is a rapidly developing line of work on exactly solvable models for difusion. For linear denoisers, prior work has characterized spectral bias laws in the training dynamics [WP26], derived random matrix deterministic equivalents for the learned denoiser [WZVP26], and analyzed generalization dynamics [MG26a, Hal25, GAH26, PG25]. For random features and shallow architectures, prior work obtained exact DSM learning curves with memorization phase diagrams [GVM26, GM26], and sharp analyses of trained shallow generative models on structured targets [CKVEZ24, CPL25, CZ23]. Most relevant to our three-phase picture, [BUBM26] identified two training timescales—a generalization time and a memorization time growing linearly in n—analytically in a random feature model and empirically in diferent models; see also [Vas25, YZTC26, BPMB26, WMBB25] for complementary mechanisms such as approximationcapacity separations and the efects of learning rates.

Relative to this literature, our advances are: (i) a universal kernel class—an infinite-dimensional model that genuinely can interpolate and memorize—handled through a new operator-level linearization with a non-ridge self-induced regularization, for general anisotropic data under a log-Sobolev condition rather than Gaussian data; (ii) exact deterministic equivalents along the gradient flow trajectory that includes the overshoot by which the trained model unlearns the covariance; (iii) the resulting separation between interpolation (polynomial time) and memorization (superpolynomial time), a distinction that cannot arise in models not rich enough to interpolate; and (iv) a pathwise analysis of the reverse SDE driven by the trained nonlinear score, where prior generated-law characterizations concern models that are linear by construction. Furthermore, our results sharpen and provide a more transparent interpretation of the two-timescale phenomenology in [BUBM26, FSW25]. Finally, [FDDS26] proves in a general framework that small train and population DSM losses are incompatible unless n is exponential in d—an impossibility statement consistent with our exact plateau; our results sharpen the picture in the kernel regime by showing that this score-level harm coexists with unharmed generation.

Finally, the supervised-learning phenomena that we take as a starting point are developed in [ZBH<sup>+</sup>17, BHMM19, BLLT20, TB23, HMRT22, GMMM21, MVSS20, BMR21] and references therein, such as interpolation with benign test error, its spectral characterizations, and the decomposition of interpolants into regular and spiky parts. For kernels specifically, the high-dimensional linearization of inner-product kernel random matrices goes back to [Kar10, CS13] and the selfinduced regularization of kernel interpolation and the polynomial approximation barrier to [LR20, GMMM21, MMM22]. On the technical side we rely on deterministic equivalents for sample covariance resolvents [CD11, HLN07, LVP26, MS24, FMPW26]. Our operator-level linearization extends the kernel matrix theory to the infinite-dimensional designs generated by denoising objectives.

## 1.4 Notation and organization

We write $\| \cdot \| _ { 2 }$ for the Euclidean norm, $\| \cdot \| _ { \mathrm { o p } } , \| \cdot \| _ { \mathrm { F } } , \| \cdot \|$ for the operator, Frobenius, and nuclear norms, and $\langle \cdot , \cdot \rangle$ for the Euclidean inner product. For a positive semidefinite matrix Σ we set $\tau _ { \Sigma } : = \mathrm { T r } ( \Sigma ) / d$ . We use $f ( d ) \lesssim g ( d )$ to denote that there exists a constant $C > 0$ , depending only on the problem parameters and the constants in the assumptions, such that $f ( d ) \leq C g ( d )$ for all suficiently large d; $f \asymp g$ means $f \lesssim g \lesssim f$ . An event E holds with very high probability (w.v.h.p.)

if its failure probability is at most $\mathbb { P } ( \mathcal { E } ^ { c } ) \le e ^ { - d ^ { c } }$ for some constant $c > 0$ and all large d. We denote $o _ { d , \mathbb { P } } ( \cdot ) , O _ { d , \mathbb { P } } ( \cdot )$ for the standard small-o and big-O in probability. For notational convenience, we introduce a notion of stochastic domination with very high probability. For sequences of nonnegative random variables $X = \{ X _ { d } \}$ and $Y = \{ Y _ { d } \}$ , we say that X is stochastically dominated by $Y$ , denoted by $X \prec Y$ , if for every $\epsilon > 0$ there exist constants $c > 0$ and $d _ { 0 }$ such that

$$
\mathbb { P } ( X _ { d } > d ^ { \epsilon } Y _ { d } ) \leq e ^ { - d ^ { c } } , \qquad { \mathrm { f o r ~ a l l ~ } } d \geq d _ { 0 } .
$$

Throughout, $C , c , c ^ { \prime } , \ldots$ . denote positive constants that may change from line to line. When the difusion time t is fixed, we allow constants to depend on t, notably through $\rho _ { t } , \sigma _ { t }$ . For a probability measure $\mu , L ^ { 2 } ( \mu )$ is the space of square-integrable functions with norm $\| \cdot \| _ { L ^ { 2 } ( \mu ) } ; \mathrm { K L } ( \cdot \| \cdot )$ denotes the Kullback–Leibler divergence. We reserve t for the difusion time and t, s for the training (optimization) time.

Organization. Section 2 sets up kernel denoising score matching: the empirical objective and its memorizing optimum, the RKHS model class, and the gradient flow and ridge estimators. Section 3 analyzes the training dynamics at a fixed noise level: the linearization and localized-bump structure (Section 3.1), and the three phases (Sections 3.2–3.4). Section 4 analyzes generation: the delocalization of reverse trajectories, the pathwise linearization of the trained score, and the Gaussian characterization of the generated distribution. Finally, we conclude with future research directions in Section 5. Appendix A develops the parallel ridge theory; the remaining appendices contain the proofs.

## 2 Kernel denoising score matching

This section introduces the setting that we consider throughout this paper. Section 2.1 describes the denoising score matching objective, Section 2.2 introduces the empirical objective and its memorizing optimum, Section 2.3 describes the RKHS model class with inner-product kernels, and Section 2.4 introduces the kernel operators and the gradient flow and ridge estimators.

## 2.1 Denoising score matching

Let $\pi _ { 0 } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ be the data distribution we wish to sample from. We progressively corrupt samples from $\pi _ { 0 }$ with Gaussian noise using the Ornstein–Uhlenbeck forward process (1). Let $\pi _ { t } : = \operatorname { L a w } ( \pmb { x } _ { t } )$ denote the resulting distribution at time t, whose explicit form is given in (2). To generate new samples, we simulate the reverse-time SDE (3) from a terminal time $T > 0$ , chosen suficiently large so that $\pi _ { T } \approx \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ , back to time zero. This requires an estimate of the marginal score ∇ log $\pi _ { t }$ at every noise level $t \in [ 0 , T ]$ . One therefore trains a time-dependent score estimator $S ( \cdot ; t ) : \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ Exploiting the identity

$$
\nabla _ { \pmb { x } _ { t } } \log \pi _ { t | 0 } ( \pmb { x } _ { t } \mid \pmb { x } _ { 0 } ) = - ( \pmb { x } _ { t } - \rho _ { t } \pmb { x } _ { 0 } ) / \sigma _ { t } ^ { 2 } = - z / \sigma _ { t } ,
$$

the score can be learned by regression on the injected noise. After rescaling the score estimator $\pmb { f } _ { t } = \sigma _ { t } \pmb { S } ( \cdot ; t )$ and with the standard weighting $\omega ( t ) = \sigma _ { t } ^ { 2 }$ , the population DSM objective is

$$
\mathcal { R } ( f ) : = \frac { 1 } { d } \int _ { 0 } ^ { T } \mathbb { E } _ { \pmb { x } _ { 0 } , \pmb { z } } \left[ \left\| f _ { t } ( \rho _ { t } \pmb { x } _ { 0 } + \sigma _ { t } \pmb { z } ) + \pmb { z } \right\| _ { 2 } ^ { 2 } \right] \mathrm { d } t ,\tag{9}
$$

where the $1 / d$ normalization keeps the risk of order one as $d \to \infty$ . By Tweedie’s formula [Rob92, Efr11], its minimizer is the (rescaled) Bayes denoiser

$$
\pmb { f } _ { t } ^ { \star } ( \pmb { x } ) = - \mathbb { E } [ \pmb { z } \ | \ \pmb { x } _ { t } = \pmb { x } ] = \sigma _ { t } \nabla \log \pi _ { t } ( \pmb { x } ) .
$$

Decoupling across difusion times. In this paper, we model the family of scores $\{ f _ { t } \} _ { t \in [ 0 , T ] }$ by a separate RKHS function at each noise level, so that the objective (9) decouples across time and each regression problem can be trained and analyzed separately. Abusing notation slightly, we fix $t \in ( 0 , T ]$ , write $\rho = \rho _ { t }$ and $\sigma = \sigma _ { t }$ with $\rho ^ { 2 } + \sigma ^ { 2 } = 1$ , and define the fixed-time population risk of a denoiser $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ as

$$
\mathcal { R } ( \pmb { f } ) = \frac { 1 } { d } \mathbb { E } _ { ( \pmb { x } _ { 0 } , \pmb { z } ) \sim \pi _ { 0 } \otimes \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ \big \| \pmb { f } ( \rho \pmb { x } _ { 0 } + \sigma \pmb { z } ) + \pmb { z } \big \| _ { 2 } ^ { 2 } \right] .
$$

Section 3 studies this fixed-time problem, treating t (and hence $( \rho , \sigma ) )$ as a constant as $n , d \to \infty ;$ Section 4 reassembles the estimators trained at the diferent noise levels into the reverse SDE. Note that, in practice, the score is often modeled by a single neural network with time as an additional input, which couples the regression problems across t. We comment further on this simplifying assumption in Section 5.

## 2.2 The empirical objective and its memorizing optimum

In practice, we observe n i.i.d. samples $\pmb { x } _ { 0 , 1 } , \ldots , \pmb { x } _ { 0 , n } \sim \pi _ { 0 }$ from the target distribution. The empirical DSM objective replaces $\pi _ { 0 }$ by the empirical measure and the score estimator f is obtained by minimizing

$$
\mathcal { L } ( \pmb { f } ) : = \frac { 1 } { d n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { \pmb { z } } \left[ \left\| \pmb { f } ( \rho \pmb { x } _ { 0 , i } + \sigma \pmb { z } ) + \pmb { z } \right\| _ { 2 } ^ { 2 } \right] ,\tag{10}
$$

over a function class $\mathcal { F } \subseteq \{ f : \mathbb { R } ^ { d } \to \mathbb { R } ^ { d } \}$ . Define the noisy empirical and population measures

$$
\widehat { \pi } _ { d , n } : = \mathrm { U n i f } \left( \{ \pmb { x } _ { 0 , i } \} _ { i = 1 } ^ { n } \right) \otimes \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) , \qquad \pi _ { d } : = \pi _ { 0 } \otimes \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d } ) ,
$$

and let $\widehat { \mu } _ { d , n }$ and $\mu _ { d }$ be their pushforwards under $( { \pmb x } , z ) \mapsto \rho { \pmb x } + \sigma { \pmb z } ;$ thus $\widehat { \mu } _ { d , n }$ is the Gaussian mixture

$$
{ \widehat { \mu } } _ { d , n } : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \mathcal { N } } ( \rho \mathbf { { x } } _ { 0 , i } , \sigma ^ { 2 } \mathbf { { I } } _ { d } ) ,
$$

a random measure, while $\mu _ { d }$ is the deterministic noisy marginal. Over an unrestricted function class, the minimizer of $\mathcal { L } ( \cdot )$ is the Bayes denoiser of $\widehat { \mu } _ { d , n }$

$$
{ \widehat { f } } ^ { \star } ( u ) : = - \mathbb { E } _ { ( x , z ) \sim { \widehat { \pi } } _ { d , n } } \left[ z \mid \rho x + \sigma z = u \right] = { \frac { 1 } { \sigma } } \sum _ { k = 1 } ^ { n } \omega _ { k } ( u ) ( \rho x _ { 0 , k } - u ) ,\tag{11}
$$

with softmax weights

$$
\omega _ { k } ( \pmb { u } ) : = \frac { \exp \left( - \frac { 1 } { 2 \sigma ^ { 2 } } \| \rho \pmb { x } _ { 0 , k } - \pmb { u } \| ^ { 2 } \right) } { \sum _ { j = 1 } ^ { n } \exp \left( - \frac { 1 } { 2 \sigma ^ { 2 } } \| \rho \pmb { x } _ { 0 , j } - \pmb { u } \| ^ { 2 } \right) } .\tag{12}
$$

This is the empirical Bayes denoiser: the exact optimum of the training objective (10) over any suficiently rich function class, at every noise level. Once inserted into the reverse SDE (3), this score drives the sampler back to the training set as $t \to 0 ^ { + }$ (that is, $\sigma  0 ^ { + } )$ , essentially performing nearest-neighbor retrieval from the training data. This paper will be interested in understanding the extent to which the RKHS model class and the training algorithm can avoid this memorization and generalize to the population Bayes denoiser $f ^ { \star }$

## 2.3 Inner-product kernels and the model class

We estimate each coordinate of the denoiser in the RKHS H of a positive definite inner-product kernel

$$
K ( \pmb { x } , \pmb { x } ^ { \prime } ) = h ( \langle \pmb { x } , \pmb { x } ^ { \prime } \rangle / d ) , \qquad h : \mathbb { R }  \mathbb { R } .
$$

More precisely, H is the completion of the space span $\{ K ( \cdot , \pmb { x } ) : \pmb { x } \in \mathbb { R } ^ { d } \}$ under the inner product $\langle K ( \cdot , \pmb { x } ) , K ( \cdot , \pmb { y } ) \rangle _ { \mathscr { H } } = K ( \pmb { x } , \pmb { y } )$ , and satisfies the reproducing property $f ( \pmb { x } ) = \langle f , K ( \cdot , \pmb { x } ) \rangle _ { \mathscr { H } } \left[ \mathrm { S C 0 8 } \right]$ The vector-valued class is the product space $\mathcal { H } ^ { d }$ with norm

$$
\| \pmb { f } \| _ { \mathcal { H } ^ { d } } ^ { 2 } = \sum _ { i = 1 } ^ { d } \| f _ { i } \| _ { \mathcal { H } } ^ { 2 } .
$$

This class is especially interesting in the context of difusion models for two reasons. First, when h has strictly positive Taylor coeficients, the RKHS is universal: it is dense in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ (and $L ^ { 2 } ( \mu _ { d } )$ under mild assumptions on $\pi _ { 0 } ;$ see Appendix F) at any fixed $d ,$ hence expressive enough to represent the empirical Bayes denoiser (11) itself; nothing in the model class forbids memorization. Second, inner-product kernels are naturally connected to wide neural networks trained by gradient descent in the lazy regime [JGH18, COB19], and their high-dimensional behavior in supervised regression is understood in fine detail [Kar10, GMMM21, MMM22, MM24]; the comparison between that theory and ours is one of the main goals of the present paper.

We assume throughout that K is continuous and diagonally integrable with respect to $\mu _ { d }$ and $\widehat { \mu } _ { d , n }$ (almost surely), meaning that $\mathbb { E } _ { \mathbf { u } \sim \nu } [ K ( \mathbf { u } , \mathbf { u } ) ] < \infty$ for $\nu \in \{ \mu _ { d } , \widehat { \mu } _ { d , n } \}$ , so that all operators and risks below are well-defined. It will be useful to introduce a ridge-regularized version of the empirical risk (10):

$$
\mathcal { L } ( \pmb { f } ; \kappa ) : = \frac { 1 } { d n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z } \left[ \left\| \pmb { f } ( \rho x _ { 0 , i } + \sigma z ) + z \right\| _ { 2 } ^ { 2 } \right] + \frac { \kappa } { d ^ { 2 } } \| \pmb { f } \| _ { \mathcal { H } ^ { d } } ^ { 2 } ,\tag{13}
$$

where $\kappa \geq 0$ is the ridge parameter. Here, the $1 / d ^ { 2 }$ scaling is added to match the ridge term to the empirical risk in the high-dimensional regime $n \asymp d .$ When H is universal, the minimizer of (13) converges as $\kappa  0 ^ { + }$ to the empirical Bayes denoiser (11).

Decoupling across coordinates. Fix an orthonormal basis $\{ { v } _ { i } \} _ { i = 1 } ^ { d }$ of $\mathbb { R } ^ { d }$ and decompose ${ \pmb f } ( { \pmb x } ) =$ $\textstyle \sum _ { i = 1 } ^ { d } f _ { { v _ { i } } } ( { \pmb x } ) { \pmb v } _ { i }$ with $f _ { { \pmb v } _ { i } } : = \langle { \pmb v } _ { i } , { \pmb f } \rangle$ . A direct computation shows $\textstyle \sum _ { i = 1 } ^ { d } \| f _ { \pmb { v } _ { i } } \| _ { \mathcal { H } } ^ { 2 } = \| \pmb { f } \| _ { \mathcal { H } ^ { d } } ^ { 2 }$ for any orthonormal basis, and since the losses are separable across coordinates,

$$
\mathcal { L } ( \boldsymbol { f } ; \kappa ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \mathcal { L } _ { v _ { i } } ( f _ { v _ { i } } ; \kappa ) , \qquad \mathcal { L } _ { v } ( \boldsymbol { f } ; \kappa ) : = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { z } \left[ \left| \boldsymbol { f } ( \rho \mathbf { x } _ { 0 , j } + \sigma z ) + \langle { \boldsymbol { v } } , \boldsymbol { z } \rangle \right| ^ { 2 } \right] + \frac { \kappa } { d } \| \boldsymbol { f } \| _ { \mathcal { H } } ^ { 2 } ,\tag{14}
$$

and similarly $\begin{array} { r } { \mathcal { R } ( \pmb { f } ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \mathcal { R } _ { \pmb { v } _ { i } } ( f _ { \pmb { v } _ { i } } ) } \end{array}$ with $\mathcal { R } _ { \pmb { v } } ( f ) : = \mathbb { E } _ { ( \pmb { x } _ { 0 } , \pmb { z } ) } [ | f ( \rho \pmb { x } _ { 0 } + \sigma \pmb { z } ) + \langle \pmb { v } , \pmb { z } \rangle | ^ { 2 } ]$ . In the study of the gradient flow and ridge estimators, it will be convenient to fix an arbitrary unit vector v and analyze the scalar regression problem for the coordinate function $f _ { v } ;$ overall risks are recovered by averaging over an orthonormal basis and the union bound.

## 2.4 Kernel operators, gradient flow, and ridge regression

To define our estimators, we introduce the following evaluation operators between H and the noisy $L ^ { 2 }$ spaces. Define $\mathcal { S } : \mathcal { H }  L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ and $\mathcal { T } : \mathcal { H } \to L ^ { 2 } ( \pi _ { d } )$ by

$$
( S f ) ( { \pmb x } , { \pmb z } ) : = f ( \rho { \pmb x } + \sigma { \pmb z } ) , \qquad ( T f ) ( { \pmb x } , { \pmb z } ) : = f ( \rho { \pmb x } + \sigma { \pmb z } ) ,\tag{15}
$$

acting on the empirical and population spaces respectively. By the reproducing property, their adjoints are the integral operators

$$
\begin{array} { r } { \begin{array} { r l } { S ^ { * } g = \mathbb { E } _ { ( \pmb { x } , \pmb { z } ) \sim \widehat { \pi } _ { d , n } } \left[ K ( \rho \pmb { x } + \sigma \pmb { z } , \cdot ) g ( \pmb { x } , \pmb { z } ) \right] , \quad } & { \mathcal { T } ^ { * } g = \mathbb { E } _ { ( \pmb { x } , \pmb { z } ) \sim \pi _ { d } } \left[ K ( \rho \pmb { x } + \sigma \pmb { z } , \cdot ) g ( \pmb { x } , \pmb { z } ) \right] . } \end{array} } \end{array}
$$

It is occasionally useful to factor these operators through the noisy marginal spaces: $\mathcal { S } = \widehat { \mathcal { P } } \circ \mathcal { S } _ { 0 }$ where $S _ { 0 } : \mathcal { H } \hookrightarrow L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ is the natural inclusion and ${ \widehat { \mathcal { P } } } : L ^ { 2 } ( { \widehat { \mu } } _ { d , n } ) \to L ^ { 2 } ( { \widehat { \pi } } _ { d , n } ) , ( { \widehat { \mathcal { P } } } g ) ( { \pmb x } , z ) : =$ $g ( \rho \pmb { x } + \sigma z )$ , is the (isometric) pullback; similarly $\mathcal { T } = \mathcal { P } \circ \mathcal { T } _ { 0 }$ with $\mathcal { T } _ { 0 } : \mathcal { H } \hookrightarrow L ^ { 2 } ( \mu _ { d } )$ and the population pullback P. We write

$$
\widehat { \mathcal { K } } : = S S ^ { * } : L ^ { 2 } ( \widehat { \pi } _ { d , n } )  L ^ { 2 } ( \widehat { \pi } _ { d , n } ) , \qquad \mathcal { K } : = \mathcal { T S } ^ { * } : L ^ { 2 } ( \widehat { \pi } _ { d , n } )  L ^ { 2 } ( \pi _ { d } ) ,
$$

for the empirical kernel operator and the train-to-population transfer operator respectively. The operator $\widehat { \kappa }$ is self-adjoint and positive semidefinite on $L ^ { 2 } ( \widehat \pi _ { d , n } )$ ; it plays the role of the kernel matrix $\kappa / n$ of supervised learning, with the crucial diference that it acts on the infinite-dimensional space $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ . The operator $\kappa$ maps functions on the empirical space to their extensions on the population space and is the operator through which train-time quantities are evaluated at test time.

Decomposing the coordinate-wise empirical risk around its unrestricted minimizer (the empirical Bayes denoiser (11)) gives $\begin{array} { r } { \mathcal { L } _ { v } ( f ; \kappa ) = \mathcal { L } _ { v } ^ { \mathrm { r e d } } ( f ; \kappa ) + \mathcal { L } _ { v } ^ { \mathrm { i r r } } } \end{array}$ , where

$$
\begin{array} { r } { \mathcal { L } _ { v } ^ { \mathrm { r e d } } ( f ; \kappa ) : = \| \mathcal { S } f - \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } + \frac { \kappa } { d } \| f \| _ { \mathcal { H } } ^ { 2 } , \qquad \mathcal { L } _ { v } ^ { \mathrm { i r r } } : = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \widehat { \pi } _ { d , n } } \left[ | \langle \boldsymbol { v } , \boldsymbol { z } \rangle + \widehat { f } _ { v } ^ { \star } ( \rho \boldsymbol { x } + \sigma \boldsymbol { z } ) | ^ { 2 } \right] , } \end{array}\tag{16}
$$

and we identify $\widehat { f } _ { v } ^ { \star }$ with its pullback to $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ . Let us emphasize that the decomposition (16) is diferent from the standard bias-variance decomposition of supervised learning, which is taken around the population Bayes predictor. Here the decomposition is around the empirical Bayes denoiser $\widehat { f } _ { v } ^ { \star } ,$ the unrestricted minimizer of the empirical risk. The reducible part $\mathcal { L } _ { v } ^ { \mathrm { r e d } }$ is the error that optimization over the function class can remove. The irreducible part $\mathcal { L } _ { v } ^ { \mathrm { i r r } }$ is the intrinsic noise of denoising under the empirical mixture; as we will see (Proposition 2), in high dimensions it is exponentially small—the empirical DSM problem is efectively noiseless, in contrast with noisy supervised learning.

Gradient flow. Our primary estimator is the gradient flow of the (unregularized) empirical risk in the RKHS geometry, started from zero: with $\mathsf { t } \geq 0$ as the training time,

$$
\frac { \partial \widehat { f } _ { \mathrm { t } , v } } { \partial \mathrm { t } } = - S ^ { * } ( S \widehat { f } _ { \mathrm { t } , v } - \widehat { f } _ { v } ^ { \star } ) , \qquad \widehat { f } _ { 0 , v } = 0 ,\tag{17}
$$

which is the functional counterpart of training a lazily parametrized network by gradient descent on (13) with $\kappa = 0$ . Since $\| S \| _ { \mathrm { o p } } ^ { 2 } = \| \widehat { K } \| _ { \mathrm { o p } } \leq \mathbb { E } _ { \widehat { \pi } _ { d , n } } [ K ( \rho x + \sigma z , \rho x + \sigma z ) ] < \infty$ , the dynamics (17) are globally well-posed, and applying S yields the closed-form training-space solution

$$
\widehat { f } _ { \mathrm { t } , v } = \mathcal { S } ^ { * } \int _ { 0 } ^ { \mathrm { t } } \exp ( - ( \mathrm { t } - \mathrm { s } ) \widehat { K } ) \widehat { f } _ { v } ^ { \star } \mathrm { d } \mathrm { s } , \qquad \mathcal { S } \widehat { f } _ { \mathrm { t } , v } = \big ( \mathrm { i d } _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } - \exp ( - \mathrm { t } \widehat { K } ) \big ) \widehat { f } _ { v } ^ { \star } .\tag{18}
$$

Thus, gradient flow applies to the empirical Bayes target the spectral filter $\lambda \mapsto 1 - e ^ { - \mathtt { t } \lambda }$ of the empirical kernel operator: components of $\widehat { f } _ { v } ^ { \star }$ on eigenfunctions of $\widehat { \kappa }$ with eigenvalue λ are fitted on the timescale $1 / \lambda$ . The entire phenomenology of Section 3 unfolds from the spectrum of $\widehat { \kappa }$ and the correlation of $\widehat { f _ { v } ^ { \star } }$ with its eigenvectors. The reducible train error and the test error along the trajectory are

$$
\begin{array} { r } { \mathcal { L } _ { \boldsymbol { v } } ^ { \mathrm { r e d } } ( \widehat { f } _ { \mathrm { t } , \boldsymbol { v } } ; 0 ) = \big \| \exp ( - \mathrm { t } \widehat { \mathcal { K } } ) \widehat { f } _ { \boldsymbol { v } } ^ { \star } \big \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } , } \end{array}\tag{19}
$$

$$
\begin{array} { r } { \mathcal { R } _ { v } ( \widehat { f } _ { \mathfrak { t } , v } ) = \big \| K \widehat { K } ^ { \dagger } \big ( \mathrm { i d } _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } - \exp ( - \mathfrak { t } \widehat { K } ) \big ) \widehat { f } _ { v } ^ { \star } - z _ { v } \big \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } , \qquad z _ { v } ( x , z ) : = - \langle v , z \rangle , } \end{array}
$$

where $\widehat { \kappa } ^ { \dagger }$ is the Moore–Penrose pseudoinverse. Throughout, coordinate flows are assembled into the vector-valued estimator $\begin{array} { r } { \widehat { \pmb f _ { \mathrm { t } } } : = \sum _ { i = 1 } ^ { d } \widehat { f _ { \mathrm { t } , { \pmb v } _ { i } } } { \pmb v } _ { i } } \end{array}$ (independent of the chosen basis, by linearity of the flow), and analogously $\widehat { \pmb f _ { \kappa } }$ for the ridge estimator (20) and $\widehat { f } ^ { \star }$ for the vector-valued empirical Bayes denoiser (11).

Ridge regression. In addition to gradient flow, we also study the kernel ridge regression estimator. Minimizing $\mathcal { L } _ { v } ( \cdot ; \kappa )$ over H for $\kappa > 0$ gives the kernel ridge estimator, characterized by the normal equation $\begin{array} { r } { ( S ^ { * } S + \frac { \kappa } { d } \mathrm { i d } _ { \mathcal { H } } ) \widehat { f } _ { \kappa , v } = S ^ { * } \widehat { f } _ { v } ^ { \star } , } \end{array}$ i.e.,

$$
\widehat { f } _ { \kappa , v } = S ^ { * } \left( \widehat { \mathcal { K } } + \frac { \kappa } { d } \mathrm { i d } _ { L ^ { 2 } \left( \widehat { \pi } _ { d , n } \right) } \right) ^ { - 1 } \widehat { f } _ { v } ^ { \star } ,\tag{20}
$$

with train error

$$
\mathcal { L } _ { v } ^ { \mathrm { r e d } } ( \widehat { f } _ { \kappa , v } ; 0 ) = \frac { \kappa ^ { 2 } } { d ^ { 2 } } \big \| \big ( \widehat { K } + \frac { \kappa } { d } \mathrm { i d } _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \big ) ^ { - 1 } \widehat { f } _ { v } ^ { \star } \big \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 }\tag{21}
$$

and test error

$$
\mathcal { R } _ { \pmb { v } } ( \widehat { f } _ { \kappa , \pmb { v } } ) = \big \| \mathcal { K } \big ( \widehat { \mathcal { K } } + \frac { \kappa } { d } \mathrm { i d } _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \big ) ^ { - 1 } \widehat { f } _ { \pmb { v } } ^ { \star } - z _ { \pmb { v } } \big \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } .
$$

Comparing the spectral filters $1 - e ^ { - \mathsf { t } \lambda }$ and $\lambda / ( \lambda + \kappa / d )$ shows that gradient flow at time t and ridge at penalty $\kappa _ { \mathrm { e f f } } = d / \mathrm { t }$ apply comparable shrinkage to eigencomponents. This is the classical early-stopping–ridge correspondence [YRC07, RWY14, AKT19], and it can be made quantitative along the whole trajectory (Lemma 21). We use ridge in two ways: as a source of intuition and direct comparison with the supervised learning literature—each phase of gradient flow has a ridge analogue at the corresponding $\kappa _ { \mathrm { e f f } }$ —and as a technical device in the proofs. The full ridge theory (exact risk asymptotics for constant, vanishing, and zero penalty) is developed in Appendix A and mirrors, statement by statement, the gradient flow results presented next.

## 3 Learning the score along the gradient flow trajectory

Fix a difusion time $t \in ( 0 , T ]$ and write $( \rho , \sigma ) = ( \rho _ { t } , \sigma _ { t } )$ . This section characterizes the gradient flow estimator $\widehat { f } _ { \mathrm { t } }$ across all training times t, and identifies three phases with distinct statistical behavior: a generalization phase at times $\mathrm { \ t } \lesssim d ^ { 1 + c }$ , in which the estimator behaves as a regularized linear model (Section 3.2); an overfitting phase at polynomial times, in which the training loss vanishes while the test risk is pinned at the risk of the pure-noise score (Section 3.3); and a memorization phase at super-polynomial times, at which the estimator finally reaches the empirical Bayes denoiser (Section 3.4). The mechanism behind the first two phases is a linearization of the kernel operator, developed in Section 3.1, which splits the estimator into a global linear component and localized nonlinear corrections around the training data.

Throughout, we work in the proportional high-dimensional regime and under a standard concentration condition on the data distribution.

Assumption 1 (High-dimensional proportional limit). The sample size n and the data dimension d grow to infinity, $n , d \to \infty ,$ , such that their ratio converges to a strictly positive constant $n / d $ $r \in ( 0 , \infty )$

Assumption 2 (Regularity of the data distribution). The data distribution $\pi _ { 0 } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ satisfies the following conditions:

(a) Mean and covariance: The first and second moments of $\pi _ { 0 }$ satisfy

$$
\mathbb { E } _ { { \pmb x } \sim \pi _ { 0 } } [ { \pmb x } ] = 0 , \qquad \mathbb { E } _ { { \pmb x } \sim \pi _ { 0 } } [ { \pmb x } { \pmb x } ^ { \top } ] = { \pmb \Sigma } ,
$$

and lim $\begin{array} { r } { \operatorname* { i n f } _ { d  \infty } \frac { \operatorname { T r } ( \Sigma ) } { d } \geq c > 0 } \end{array}$ for some constant $c > 0$

(b) Logarithmic Sobolev inequality: There exists a constant $C _ { \mathrm { L S I } } \geq 1$ such that $\pi _ { 0 }$ satisfies the log-Sobolev inequality

$$
\mathrm { E n t } _ { \pi _ { 0 } } ( f ^ { 2 } ) \leq 2 C _ { \mathrm { L S I } } \mathbb { E } _ { \pi _ { 0 } } [ \| \nabla f \| _ { 2 } ^ { 2 } ]
$$

for every $f \in W ^ { 1 , 2 } ( \pi _ { 0 } ) , ^ { 3 }$ where $\operatorname { E n t } _ { \pi _ { 0 } } ( f ^ { 2 } ) : = \mathbb { E } _ { \pi _ { 0 } } [ f ^ { 2 } \log f ^ { 2 } ] - \mathbb { E } _ { \pi _ { 0 } } [ f ^ { 2 } ] \log \mathbb { E } _ { \pi _ { 0 } } [ f ^ { 2 } ] .$

Assumption 2.(b) should be viewed as a high-dimensional concentration assumption and it is imposed mainly for convenience: we expect the results to extend to other standard high-dimensional models, for instance $\pmb { x } = \pmb { \Sigma } ^ { 1 / 2 } \pmb { u }$ with u having independent sub-Gaussian entries. Apart from this concentration requirement, Assumption 2 constrains only the first two moments of $\pi _ { 0 }$ and the data distribution may be multimodal and have intricate higher-order dependencies. Nevertheless, we will see that the RKHS estimators studied here depend only on the covariance Σ, and therefore do not learn higher-order structure in the data distribution. Throughout, we write

$$
\tau _ { \Sigma } : = \mathrm { T r } ( \Sigma ) / d \geq c > 0 ,
$$

for the average eigenvalue of Σ. In particular, under Assumption 2.(b), $\tau _ { \Sigma } \leq \| \Sigma \| _ { \mathrm { o p } } \leq C _ { \mathrm { L S I } }$ . We arrange the n training inputs as the columns of the data matrix $\pmb { X } : = [ \pmb { x } _ { 0 , 1 } , \dotsc , \pmb { x } _ { 0 , n } ] \in \mathbb { R } ^ { d \times n }$ , and denote their empirical covariance matrix by ${ \widehat { \pmb { \Sigma } } } : = { \pmb { X } } { \pmb { X } } ^ { \mathsf { T } } / n$

## 3.1 Linearization of the kernel and localized bumps

In the moderate-time regime $\mathrm { t } \lesssim d ^ { 1 + c }$ with $c > 0$ small enough, we assume the following<sup>4</sup>.

Assumption 3 (Kernel regularity). The kernel is a positive definite inner-product kernel of the form $K ( { \pmb x } , { \pmb x } ^ { \prime } ) = h ( \langle { \pmb x } , { \pmb x } ^ { \prime } \rangle / d )$ . The kernel function $h : \mathbb { R }  \mathbb { R }$ is three times continuously diferentiable, with $h ( 0 ) = h ^ { \prime \prime } ( 0 ) = 0$ and $h ^ { \prime } ( 0 ) = 1$ . Furthermore, there exists a constant $C \geq 1$ such that $| h ^ { \prime \prime \prime } ( x ) | \leq C e ^ { C | x | }$ for all $x \in \mathbb { R }$

The condition $h ^ { \prime } ( 0 ) = 1$ is a normalization convention and is without loss of generality. For simplicity, we also assume that $h ( 0 ) = h ^ { \prime \prime } ( 0 ) = 0$ , which eliminates the constant function spike from the kernel linearization. Remark C.1 explains how this assumption can be removed.

The key phenomenon we use in the first two phases is the linearization of the kernel on delocalized pairs of inputs. On pairs of inputs whose overlap $\langle { { \pmb u } , { \pmb u } ^ { \prime } } \rangle / d$ is $o _ { d } ( 1 )$ , as is typical for independent noisy samples, the nonlinear kernel is indistinguishable from its linearization $h ( x ) \approx x ;$ in particular, this is always the case for new independent data at test time. The nonlinearity of h survives only on coincident inputs ${ \pmb u } = \rho { \pmb x } + \sigma { \pmb z }$ and $\mathbf { \boldsymbol { u } } ^ { \prime } = \rho \mathbf { \boldsymbol { x } } + \sigma z ^ { \prime }$ that share the same data point x: there the overlap concentrates around $\rho ^ { 2 } \tau _ { \Sigma }$ and $h ( \langle { \pmb u } , { \pmb u } ^ { \prime } \rangle / d ) = h ( \rho ^ { 2 } \tau _ { \Sigma } ) + o _ { d } ( 1 )$ retains the full nonlinearity of the kernel. Thus, in the high-dimensional linear regime $n \asymp d .$ the kernel operators behave as the efective operators $\widehat { \mathcal { K } } _ { \mathrm { e f f } } : L ^ { 2 } ( \widehat { \pi } _ { d , n } ) \to L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ at train time, and ${ \mathcal { K } } _ { \mathrm { e f f } } : L ^ { 2 } ( { \widehat { \pi } } _ { d , n } ) \to L ^ { 2 } ( \pi _ { d } )$ at test time, defined by

$$
\widehat K _ { \mathrm { e f f } } f ( \boldsymbol x , \boldsymbol z ) : = \mathbb { E } _ { ( \boldsymbol x ^ { \prime } , \boldsymbol z ^ { \prime } ) \sim \widehat \pi _ { d , n } } \left[ \frac { \langle \rho \boldsymbol x + \sigma \boldsymbol z , \rho \boldsymbol x ^ { \prime } + \sigma \boldsymbol z ^ { \prime } \rangle } { d } f ( \boldsymbol x ^ { \prime } , \boldsymbol z ^ { \prime } ) \right] + \delta _ { n } \mathbb { E } _ { \boldsymbol z ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf I _ { d } ) } [ f ( \boldsymbol x , \boldsymbol z ^ { \prime } ) ] ,\tag{22}
$$

and

$$
K _ { \mathrm { e f f } } f ( \pmb { x } , z ) : = \mathbb { E } _ { ( \pmb { x } ^ { \prime } , z ^ { \prime } ) \sim \widehat { \pi } _ { d , n } } \left[ \frac { \langle \rho \pmb { x } + \sigma z , \rho \pmb { x } ^ { \prime } + \sigma z ^ { \prime } \rangle } { d } f ( \pmb { x } ^ { \prime } , z ^ { \prime } ) \right] ,
$$

where we denoted

$$
\delta _ { n } : = \frac { h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma } } { n } .
$$

The first term in (22) is the operator of the linear kernel $\langle { \pmb x } , { \pmb x } ^ { \prime } \rangle / d ;$ the second, present only on the empirical space, averages over the noise at a fixed training sample and is weighted by the kernel excess curvature $h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma }$ at that scale. Following [GMMM21], we refer to $\delta _ { n }$ as the denoising self-induced regularization from the nonlinear part of the kernel. Positive definiteness of K, together with $h ( 0 ) = 0$ and $h ^ { \prime } ( 0 ) = 1$ , ensures that $\delta _ { n } \geq 0$ (see Remark C.2).

Proposition 1 (Kernel operator linearization). Suppose that Assumptions 1, 2, and 3 hold. Then, $\| \widehat { \mathcal { K } } - \widehat { \mathcal { K } } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \vee \| \mathcal { K } - \mathcal { K } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \prec d ^ { - \frac { 3 } { 2 } }$

Proposition 1 is the analogue, for denoising score matching, of the classical linearization of $n \times n$ inner-product kernel random matrices in the regime n ≍ d [Kar10]. The key distinction is that our result holds at the operator level on the noisy, infinite-dimensional empirical space $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ , where $\widehat { \pi } _ { d , n }$ is a mixture of n Gaussians, rather than n points. In particular, the linearization must hold uniformly over functions of the Gaussian noise directions, requiring careful control of the kernel’s power-series expansion; see the proof in Appendix C.1.

Thus, in the proportional regime and at moderate training times, the sole efect of the kernel’s nonlinearity is to add a per-sample regularization $\delta _ { n } \mathbb { E } _ { z ^ { \prime } } [ f ( { \pmb x } , z ^ { \prime } ) ]$ to the efective linear operator. This is the DSM analogue of the additive self-induced ridge regularization identified for kernel interpolation in supervised learning [LR20, GMMM21, MM24]. However, the self-induced regularization here is not a ridge penalty (a multiple of the identity) on $L ^ { 2 } ( \widehat \pi _ { d , n } )$ : it acts through the conditional noise-averaging operator $f ( \pmb { x } _ { 0 , i } , z ) \mapsto \mathbb { E } _ { z ^ { \prime } } [ f ( \pmb { x } _ { 0 , i } , z ^ { \prime } ) ]$

To understand what the efective operator implies for the estimator, it is instructive to write down the RKHS and empirical risk minimization problem associated with $\widehat { \kappa } _ { \mathrm { e f f } }$ : the efective kernel function is

$$
K _ { \mathrm { e f f } } ( { \pmb x } _ { 0 , i } , { z } ; { \pmb x } _ { 0 , j } , { z } ^ { \prime } ) = \frac 1 d \langle \rho { \pmb x } _ { 0 , i } + \sigma { z } , \rho { \pmb x } _ { 0 , j } + \sigma { z } ^ { \prime } \rangle + n \delta _ { n } \mathbb { 1 } _ { i = j } ,
$$

with associated RKHS (with b = 0 when $\delta _ { n } = 0 )$

$$
\begin{array} { r } { \mathcal { H } _ { \mathrm { e f f } } : = \Big \{ f _ { w , b } \in L ^ { 2 } ( \widehat { \pi } _ { d , n } ) \ : \ f _ { w , b } ( x _ { 0 , i } , z ) = \langle w , \rho x _ { 0 , i } + \sigma z \rangle + b _ { i } , \ w \in \mathbb { R } ^ { d } , b \in \mathbb { R } ^ { n } \Big \} , } \end{array}
$$

$$
\Vert f _ { \pmb { w } , \pmb { b } } \Vert _ { \mathcal { H } _ { \mathrm { e f f } } } ^ { 2 } : = d \Vert \pmb { w } \Vert _ { 2 } ^ { 2 } + \frac { 1 } { n \delta _ { n } } \Vert \pmb { b } \Vert _ { 2 } ^ { 2 } .
$$

For simplicity, consider the ridge regression estimator $\widehat { f } _ { \kappa , v }$ , which is related to the gradient flow estimator at $\mathfrak { t } = d / \kappa$ . For moderate regularization $\kappa \gtrsim d ^ { - c }$ (that is, $\displaystyle \mathbf { t } \lesssim d ^ { 1 + c } )$ , the ridge empirical

minimization problem (14) is well approximated by

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \boldsymbol { v } } ( \widehat { f } _ { \star , \boldsymbol { v } } ; \kappa ) = \operatorname* { m i n } _ { f \in \mathcal { H } } \mathcal { L } _ { \boldsymbol { v } } ( f ; \kappa ) \approx \operatorname* { m i n } _ { f \in \mathcal { H } _ { \mathrm { e f f } } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { \boldsymbol { z } } \left[ \big | f ( \rho x _ { 0 , i } + \sigma z ) + \langle { \boldsymbol { v } } , \boldsymbol { z } \rangle \big | ^ { 2 } \right] + \frac { \kappa } { d } \| f \| _ { \mathcal { H } _ { \mathrm { e f f } } } ^ { 2 } } \\ { = \displaystyle \operatorname* { m i n } _ { w \in \mathbb { R } ^ { d } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } _ { b _ { i } \in \mathbb { R } } \Big \{ \mathbb { E } _ { \boldsymbol { z } } \left[ \big \langle w , \rho x _ { 0 , i } + \sigma z \big \rangle + b _ { i } + \langle { \boldsymbol { v } } , \boldsymbol { z } \rangle \big | ^ { 2 } \right] + \frac { \kappa } { d \delta _ { n } } b _ { i } ^ { 2 } \Big \} + \kappa \| w \| _ { 2 } ^ { 2 } , } \end{array}\tag{23}
$$

and the test error in this regime (with $( \widehat { \pmb w } , \widehat { \pmb b } )$ the minimizer of the ERM over $\mathcal { H } _ { \mathrm { e f f } } )$ is

$$
\begin{array} { r } { \mathcal { R } _ { \pmb { v } } ( \widehat { f } _ { \kappa , \pmb { v } } ) \approx \mathbb { E } _ { ( \pmb { x } , \pmb { z } ) \sim \pi _ { d } } \left[ \big | \langle \widehat { \pmb { w } } , \rho \pmb { x } + \sigma \pmb { z } \rangle + \langle \pmb { v } , \pmb { z } \rangle \big | ^ { 2 } \right] , } \end{array}\tag{24}
$$

where

$$
\widehat { \pmb { b } } = - \frac { d \delta _ { n } } { \kappa + d \delta _ { n } } \rho \pmb { X } ^ { \top } \widehat { \pmb { w } } , \qquad \widehat { \pmb { w } } = - \sigma \Big [ \kappa \mathbf { I } _ { d } + \sigma ^ { 2 } \mathbf { I } _ { d } + \frac { \rho ^ { 2 } \kappa } { \kappa + d \delta _ { n } } \widehat { \pmb { \Sigma } } \Big ] ^ { - 1 } \pmb { v } .\tag{25}
$$

Three features of the efective problem (23)–(24) and its minimizer (25) drive the phenomenology in the next two subsections:

(a) The kernel nonlinearity efectively behaves as a per-sample ofset $b _ { i }$ . In the original RKHS, $b _ { i }$ corresponds to a ‘localized bump’ on the tube around ${ \pmb x } _ { 0 , i }$ that allows the model to fit the conditional denoising target $- \mathbb { E } [ \langle { \pmb v } , z \rangle | { \pmb u } = \rho { \pmb x } _ { 0 , i } + \sigma z ]$ . The fitted model $\widehat { f } _ { \kappa , v }$ is thus a linear combination of a global linear predictor $\langle \widehat { \pmb { w } } , \rho \pmb { x } + \sigma \pmb { z } \rangle$ and n localized nonlinear corrections $b _ { i }$ The inverse curvature $( h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma } ) ^ { - 1 }$ plays the role of a ridge penalty on these ofsets.

(b) For $\kappa = o _ { d } ( 1 )$ (but still large enough for the linearization approximation to hold) and $\delta _ { n } > 0$ the minimizer (25) becomes $\widehat { \pmb { w } } = - \pmb { v } / \sigma + o _ { d , \mathbb { P } } ( 1 )$ and $b _ { i } = - \rho \langle \widehat { \pmb { w } } , \pmb { x } _ { 0 , i } \rangle + o _ { d , \mathbb { P } } ( 1 )$ , and achieves vanishing train error $\mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ) = o _ { d , \mathbb { P } } ( 1 )$ and test error $\begin{array} { r } { \mathcal { R } _ { v } ( \widehat { f } _ { \kappa , v } ) = \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \tau _ { \Sigma } + o _ { d , \mathbb { P } } ( 1 ) } \end{array}$ . The linear part corresponds to the score of the pure noise component $- \langle { \pmb v } , z \rangle$ and the bumps cancel the resulting bias $- \rho \langle { \pmb v } , { \pmb x } _ { 0 , i } \rangle / \sigma$ on each tube. In fact, this solution (and test error) persists for all polynomially vanishing $\kappa = d ^ { - C }$ (equivalently, polynomial time ${ \bf { t } } = d ^ { 1 + C } )$ , and memorization happens only for superpolynomially vanishing κ (superpolynomial t).

(c) At test time, the RKHS score model reduces efectively to a linear score, even though $\widehat { f } _ { \kappa , v }$ is nonlinear. Moreover, unlike in the supervised setting (8), the kernel nonlinear contribution $\delta _ { n }$ does not act as an additive ridge regularizer. Instead, it shrinks the empirical covariance matrix $\widehat { \pmb { \Sigma } }$ by the factor $\kappa / ( \kappa + d \delta _ { n } )$ . Thus, whenever $\delta _ { n } > 0$ , the solution $\widehat { \pmb { w } }$ in (25) loses its dependence on $\widehat { \pmb { \Sigma } }$ as $\kappa  0$ . This shrinkage also explains the observation following Theorem 1 that larger $\delta _ { n }$ lead to an earlier onset of overfitting and covariance forgetting; see Figure 2 for a numerical illustration.

We further discuss the consequences of the kernel linearization for estimation in Section 3.2.

Proposition 2 (Vanishing irreducible error). Suppose that Assumptions 1 and 2 hold. Let $\pmb { v } \in \mathbb { R } ^ { d }$ with $\| \pmb { v } \| _ { 2 } = 1$ . Then there exists a constant $c > 0$ such that

$$
\| \widehat { f } _ { v } ^ { \star } - z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq e ^ { - d ^ { c } }
$$

with very high probability, where $z _ { v } ( \pmb { x } , z ) = - \langle \pmb { v } , z \rangle$

Two consequences follow immediately from this proposition (proved in Appendix C.1). First, the irreducible part of the training risk $\mathcal { L } _ { v } ^ { \mathrm { i r r } }$ in (16) is exponentially small: the empirical regression problem is efectively noiseless around the empirical Bayes denoiser. Second, the informative part of $\widehat { f } _ { v } ^ { \star }$ (where it deviates from $- \langle { \pmb v } , z \rangle$ , encoding the locations $\pmb { x } _ { 0 , i } )$ lives at exponentially small $L ^ { 2 } ( \widehat \pi _ { d , n } )$ -scale. The entire training signal about the data distribution is carried not by the target $\widehat { f } _ { v } ^ { \star }$ but by the design geometry $\{ \rho { \pmb x } _ { 0 , i } + \sigma z \}$ on which the regression is performed. This is in sharp contrast with supervised learning and underlies the diferent phenomenology of DSM.

## 3.2 Phase I: linearized dynamics and generalization

We now describe the gradient flow trajectory on timescales $\mathrm { t } \lesssim d ^ { 1 + c }$ for a small constant $c > 0$ . On these timescales the dynamics are governed by the linearized operator $\widehat { \mathcal { K } } _ { \mathrm { e f f } }$ , and the risks concentrate on deterministic equivalents expressed through the spectrum of the sample covariance.

The deterministic equivalents are built from standard self-consistent resolvent approximations. Let $\mathbb { C } _ { + } : = \{ z \in \mathbb { C } : \operatorname { I m } ( z ) > 0 \}$ , and define $M : \mathbb { C } _ { + } \to \{ A \in \mathbb { C } ^ { d \times d } : \operatorname { I m } [ A ] \succ 0 \}$ and $\mu : \mathbb { C } _ { + } \to \mathbb { C } _ { + }$ as the unique analytic solutions of

$$
M ( z ) = \left( { \frac { \Sigma } { 1 + \mu ( z ) } } - z { \bf I } _ { d } \right) ^ { - 1 } , \qquad \mu ( z ) = { \frac { 1 } { n } } \mathrm { T r } \left( \Sigma M ( z ) \right) .\tag{26}
$$

Since M is a matrix-valued Herglotz function [GT00], there exists a compactly supported matrixvalued probability measure Ω on Borel subsets of R such that

$$
M ( z ) = \int _ { \mathbb { R } } \frac { 1 } { \lambda - z } \pmb { \Omega } ( \mathrm { d } \lambda ) ;\tag{27}
$$

Ω is the deterministic equivalent of the spectral measure of the sample covariance $\widehat { \Sigma }$ , resolved along the eigenspaces of Σ [CD11, LVP26].

In the linearized regime, the gradient flow dynamics decouple along this spectral decomposition, and in each spectral direction λ, they reduce to a two-dimensional linear system coupling the signal component of the estimator (the coeficient of $\langle \pmb { x } _ { 0 } , \cdot \rangle )$ with a ‘noise component’ (associated to $\langle z , \cdot \rangle )$ ; the nonlinearity of the kernel enters only through the self-induced regularization $\delta _ { n }$ . For every $x \geq 0$ , define

$$
\psi ( x , { \mathfrak { t } } ) : = - { \frac { 1 } { \sigma } } e _ { 2 } ^ { \mathsf { T } } \left( { \mathbf { I } } _ { 2 } - e ^ { - { \mathsf { t } } Q ( x ) } \right) e _ { 2 } , \qquad Q ( x ) : = \left[ { \frac { \rho ^ { 2 } x } { d } } + \delta _ { n } \begin{array} { c c } { \frac { \rho \sigma { \sqrt { x } } } { d } } \\ { \frac { \rho \sigma { \sqrt { x } } } { d } } & { \frac { \sigma ^ { 2 } } { d } } \end{array} \right] ,\tag{28}
$$

and the resulting deterministic equivalents for the train and test errors

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } : = \displaystyle \int _ { \mathbb { R } } \ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( \lambda ) \frac { 1 } { d } \mathrm { T r } ( \Omega ( \mathrm { d } \lambda ) ) , \qquad \ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( x ) : = \big \| e ^ { - \mathrm { t } Q ( x ) } e _ { 2 } \big \| _ { 2 } ^ { 2 } , } \\ & { } & { \mathcal { R } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } : = \displaystyle \int _ { \mathbb { R } } \psi ( \lambda , \mathrm { t } ) ^ { 2 } \frac { 1 } { d } \mathrm { T r } ( \Sigma _ { \sigma } \Omega ( \mathrm { d } \lambda ) ) + 2 \sigma \displaystyle \int _ { \mathbb { R } } \psi ( \lambda , \mathrm { t } ) \frac { 1 } { d } \mathrm { T r } ( \Omega ( \mathrm { d } \lambda ) ) + 1 , } \end{array}\tag{29}
$$

where $\Sigma _ { \sigma } : = \rho ^ { 2 } \Sigma + \sigma ^ { 2 } \mathbf { I } _ { d }$ is the covariance of the noisy data and $e _ { 2 } = ( 0 , 1 ) ^ { \mathsf { T } }$ . We denote by $\gamma _ { - } ( x ) \leq \gamma _ { + } ( x )$ the eigenvalues of $Q ( x )$ , which are the relaxation rates of the two-dimensional linear system in spectral direction $x ,$ and by $\ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( x )$ and $\psi ( x , \mathfrak { t } )$ the corresponding profiles for the training and test errors, respectively. Remark E.1 gives explicit expressions for these quantities.

![](images/7d893599db27db01846bba451db3b079513a007d2bd68fd7126ed15384620f82.jpg)  
Figure 2: Training and test errors as functions of the normalized gradient-flow time $\mathrm { t } / d$ for $n = 2 4 0$ difusion time $t = 0 . 2$ , and Gaussian data $\pi _ { 0 } = \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ . Dashed and solid lines correspond, respectively, to the training and test errors of the finite-dimensional linearized estimator obtained by solving (23) with gradient flow. Open and filled circles show the corresponding deterministic equivalents from Theorem 1. The dotted horizontal line indicates the asymptotic test-error plateau $\displaystyle \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \frac { \mathrm { T r } ( \Sigma ) } { d }$ . Left: We use $h ( x ) = x + x ^ { 4 } / 1 2$ and vary $d / n$ while holding $n ,$ and hence $\delta _ { n } .$ , fixed. This changes the relaxation rates $\gamma _ { - } .$ , shifting the onset of overfitting. Right: We fix $d = 1 2 0$ and vary the efective linearization parameter $\delta _ { n }$ . When $\delta _ { n } = 0$ , the estimator is linear and cannot interpolate the training loss. By contrast, whenever $\delta _ { n } > 0$ , the training error converges to zero and the test error approaches the dotted interpolation plateau. Because $\gamma _ { - }$ increases with $\delta _ { n }$ , larger $\delta _ { n }$ leads to an earlier transition to overfitting.

On a new test sample, the estimator at time t acts as the linear shrinkage $\widehat { f } _ { \mathrm { t } } ( u ) \approx \psi ( \widehat { \Sigma } , \mathrm { t } )$ u where $\psi ( \cdot , \mathfrak { t } )$ acts spectrally. The expression $\mathcal { R } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) }$ is then obtained by replacing $\widehat { \pmb { \Sigma } }$ by its deterministic equivalent.

Theorem 1 (Phase I: deterministic equivalents along gradient flow). Suppose that Assumptions $^ { 1 , }$ ${ \mathcal { Q } } ,$ and 3 hold. Then, for every $\epsilon , D > 0$ , there exists a constant $C > 0$ such that for every $\mathrm { { t } \geq 0 }$

$$
| \mathcal { L } ( \widehat { f } _ { \sf t } ; 0 ) - \mathcal { L } _ { \sf d e t } ^ { ( \sf t ) } | \lesssim \left( 1 + \frac { \sf t } { d } \right) d ^ { { \epsilon } - \frac { 1 } { 2 } } , \qquad | \mathcal { R } ( \widehat { f } _ { \sf t } ) - \mathcal { R } _ { \sf d e t } ^ { ( \sf t ) } | \lesssim \left( 1 + \frac { \sf t } { d } \right) ^ { 3 } d ^ { { \epsilon } - \frac { 1 } { 2 } }
$$

with probability at least $1 - C d ^ { - D }$ . In particular, $i f \ t \leq d ^ { 1 + \alpha }$ for a constant $\alpha < 1 / 6$ and $\varepsilon < 1 / 2 - 3 \alpha$ then the right hand sides are $o _ { d } ( 1 )$

The proof is given in Appendix E.1. We next highlight the main phenomenology of gradient flow at moderate times:

(a) The relaxation rates satisfy $\begin{array} { r } { \gamma _ { + } ( x ) \approx \frac { \rho ^ { 2 } x + \sigma ^ { 2 } } { d } + \delta _ { n } } \end{array}$ and $\begin{array} { r } { \gamma _ { - } ( x ) \approx \frac { \sigma ^ { 2 } \delta _ { n } } { d \gamma _ { + } ( x ) } } \end{array}$ . Each spectral direction of the sample covariance is learned on the timescale $1 / \gamma _ { + } ( \lambda ) \asymp d / ( \rho ^ { 2 } \lambda + \sigma ^ { 2 } + d \delta _ { n } )$ : top principal directions first, bulk directions later, and the timescales are modulated by the noise level—at high noise $( \rho \to 0 )$ all directions relax together, while at low noise the training schedule follows the spectral hierarchy of $\widehat { \Sigma } .$ This was already described for purely linear models in [WP26, WZVP26]. Here we show that, to leading order, the kernel nonlinearity simply shifts the timescales by the self-induced regularization $\delta _ { n }$

(b) For the linear kernel $( \delta _ { n } = 0$ and $\gamma _ { - } \equiv 0 )$ , gradient flow converges to the linear ridgeless fit without ever interpolating. On the other hand, any kernel curvature $h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma } > 0$ opens a second channel with relaxation rate $\gamma _ { - } .$ proportional to the self-induced regularization $\delta _ { n }$ through which the training error continues to decrease toward zero. The data eigendirections are learned at timescale $1 / \gamma _ { + } \approx d ,$ while the localized bumps are fitted at timescale $1 / \gamma _ { - } \approx$ $\delta _ { n } ^ { - 1 } \approx n$ . The separation between the two families of timescales is the mechanism behind the two-stage training curves that was observed and analytically described for random-feature models in [BUBM26]. Here, we sharpen this picture by characterizing how this timescale separation depends on the data covariance Σ and the kernel curvature $\delta _ { n }$

(c) At any fixed λ in the bulk, the shrinkage symbol $\psi ( \lambda , \mathtt { t } )$ is monotonically decreasing in t, from $\psi ( \lambda , 0 ) = 0$ to

$$
\operatorname * { l i m } _ { { \bf t }  { \bf \infty } } \psi ( \lambda , { \bf t } ) = - \frac { 1 } { \sigma } \qquad ( \delta _ { n } > 0 ) .
$$

Along the way, it crosses the pointwise-optimal shrinkage $\begin{array} { r } { \psi ^ { \mathrm { o p t } } ( \lambda ) = - \frac { \sigma \mathrm { T r } \left( \Omega ( \mathrm { d } \lambda ) \right) } { \mathrm { T r } \left( \Sigma _ { \sigma } \Omega ( \mathrm { d } \lambda ) \right) } } \end{array}$ , which minimizes the integrand of $\mathcal { R } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) }$ in (29). The contribution to the test risk of each eigendirection therefore descends toward the risk of the best linear denoiser and then climbs as the fit overshoots toward the constant $- 1 / \sigma$ . The limit $- \pmb { u } / \sigma ^ { 2 }$ of $\widehat { f _ { \mathrm { t } } } / \sigma$ is the rescaled score of the pure-noise distribution ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { d } )$ . Trained to convergence within the linearized description, the model forgets the covariance it had learned and treats all of its input as noise. Substituting $\psi \equiv - 1 / \sigma$ into (29) gives $\begin{array} { r } { \mathcal { R } = \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \tau _ { \Sigma } ( 1 + o ( 1 ) ) } \end{array}$ , which is the interpolation plateau of Phase II described below. The optimal early-stopping window is the interval between the covariance-learning descent and this overshoot; its optimum is located at ${ \sf t } ^ { \star } \asymp d ,$ with a constant determined by $\Sigma , \sigma _ { ; }$ , and $\delta _ { n }$ through (29). In particular, since $\gamma _ { - }$ is nondecreasing in $\delta _ { n } .$ larger curvature shortens the good generalization window: more strongly nonlinear kernels overfit earlier.

We illustrate the train and test error dynamics during Phase I in Figure 2.

## 3.3 Phase II: vanishing error and overfitting

Beyond the timescale $d ^ { 1 + c }$ , the operator-level kernel linearization used in Theorem 1 no longer applies. Nevertheless, its approximation remains valid for the explicit gradient flow solution (18). We establish this directly by showing that the kernel nonlinearity fits localized bumps around the individual training samples. Our analysis of this regime requires the following additional condition on the kernel.

Assumption 4 (Kernel high-degree coeficients). The kernel is an inner-product kernel of the form $K ( { \pmb x } , { \pmb x } ^ { \prime } ) = h ( \langle { \pmb x } , { \pmb x } ^ { \prime } \rangle / d )$ The base function $h : \mathbb { R }  \mathbb { R }$ admits a power series expansion $\begin{array} { r } { h ( x ) = \sum _ { k = 0 } ^ { \infty } \alpha _ { k } x ^ { k } } \end{array}$ with non-negative coeficients $\alpha _ { k } \geq 0 ~ f o r$ all $k \geq 0$ , with $\alpha _ { 0 } = \alpha _ { 2 } = 0$ and $\alpha _ { 1 } = 1$ . Furthermore, there exists an even integer $k _ { 0 } \geq 6$ and a constant $c > 0$ such that $\alpha _ { k } > c$ for all $k \in \{ k _ { 0 } , \ldots , 2 k _ { 0 } \}$

Finally, there exists a constant $\zeta \in \mathsf { \Gamma } ( 0 , 1 )$ such that the higher-order coeficients satisfy the summability condition

$$
\sum _ { k = 3 } ^ { \infty } k ^ { 2 } \alpha _ { k } C _ { \zeta , k } ^ { 2 } < \infty , \qquad C _ { \zeta , k } : = ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( ( 4 k - \ln \zeta + 2 ) k ) ^ { 2 } ) ^ { \frac { k } { 2 } } ( 4 k - \ln \zeta ) .
$$

Under this assumption, we show that using the kernel’s high-degree components, one can fit a family of localized bump functions (Appendix C.2): elements $f _ { m , v } ^ { \mathrm { l o c } } \in \mathcal { H }$ which approximate the target $- \langle { \pmb v } , z \rangle$ on the union of the training tubes while remaining bounded in RKHS norm.

Proposition 3 (Localized interpolants). Suppose that Assumptions 1, 2, and 4 hold. Let $\pmb { v } \in \mathbb { R } ^ { d }$ be such that $\| \pmb { v } \| _ { 2 } = 1$ . Let $k _ { 0 } \geq 6$ be the even integer guaranteed by Assumption $\it 4$ and denote $k _ { 0 } = 2 m + 6$ . Then, there exists $f _ { m , v } ^ { \mathrm { l o c } } : \mathbb { R } ^ { d } $ R such that

$$
\begin{array} { r } { \mathcal { L } _ { \boldsymbol { v } } ( f _ { m , v } ^ { \mathrm { l o c } } ; 0 ) \prec { d ^ { - { k _ { 0 } + 2 } } } , \qquad \| f _ { m , v } ^ { \mathrm { l o c } } \| _ { \mathcal { H } } ^ { 2 } \prec d . } \end{array}
$$

It is worth emphasizing that this result is much stronger than the interpolation of n labels in supervised learning. The empirical DSM objective integrates over the noise: driving it to zero requires matching the target as a function of a mixture of n d-dimensional Gaussian vectors, i.e., interpolating in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ rather than at finitely many points.

Combining Proposition 3 with a comparison between gradient flow and ridge along the trajectory (Lemma 21) yields the description of the second phase.

Theorem 2 (Phase II: overfitting). Suppose that Assumptions 1, 2, and 4 hold. Suppose further that there exists an $\epsilon > 0$ such that $d ^ { 1 + \epsilon } \lesssim \mathfrak { t } \lesssim d ^ { k _ { 0 } - 1 }$ , where $k _ { 0 }$ is given in Assumption 4. Then, for every constant $c \in ( 0 , ( \epsilon \land 1 ) / 2 )$ , there exist constants $C , c ^ { \prime } > 0$ such that

$$
\mathcal { L } ( \widehat { f } _ { \mathrm { t } } ; 0 ) \prec \frac { d } { \mathrm { t } } = o _ { d } ( 1 ) ,
$$

and

$$
\left| { \mathcal R } ( \widehat f _ { \mathrm { t } } ) - \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \frac { \mathrm { T r } ( \Sigma ) } { d } \right| \lesssim ( C _ { \zeta } \vee 1 ) d ^ { c - \frac { \epsilon \wedge 1 } { 2 } } + \sqrt { 1 + ( C _ { \zeta } \vee 1 ) d ^ { c - \frac { \epsilon \wedge 1 } { 2 } } } d ^ { \frac { c } { 2 } - \frac { \epsilon \wedge 1 } { 4 } }
$$

with probability at least $1 - \zeta - e ^ { - d ^ { c ^ { \prime } } }$ for all $d \ge C$ , where $C _ { \zeta }$ is defined in (52).

The proof is given in Appendix E.2. In the polynomial-time regime of Phase II, the training error continues to vanish at rate $d / \mathrm { t }$ , while the population risk plateaus at ${ \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } } \tau _ { \Sigma }$ , which is the risk of the rescaled pure-noise score $- \pmb { u } / \sigma ^ { 2 }$ . The gap between the two is carried by the localized bumps which are invisible to the population risk. The model has overfitted to the training data, but the overfitting is confined to an exponentially thin neighborhood of the training tubes, and the population risk is determined by the pure-noise score, not by the covariance fit. Whether the overfitting is benign is set by the noise level $\rho _ { t } ^ { 2 } / \sigma _ { t } ^ { 2 } = 1 / \sigma _ { t } ^ { 2 } - 1$ : at high noise the plateau is close to the best predictor and overfitting is benign, while at low noise the plateau exceeds the risk of the zero estimator and overfitting is detrimental. Note that in either case, the plateau is flat: throughout the polynomial window, further training changes the estimator only inside the tubes and leaves the population risk unchanged.

## 3.4 Phase III: the memorization endpoint

The plateau of Phase II describes a model that has interpolated the training objective while behaving, at test points, as the pure-noise score: overfitted, but far from the empirical Bayes denoiser. The final phase of training, which occurs at super-polynomial times, is the memorization phase. We characterize the gradient flow endpoint in fixed dimension, and we give a closed-form expression for its population risk in the high-dimensional limit. In this phase, we assume that the kernel has all higher-degree coeficients positive.

Assumption 5 (Kernel high-degree tail). There exists an even integer $k _ { 0 } \geq 0$ such that $K ( { \pmb x } , { \pmb y } ) =$ $h ( \langle { \pmb x } , { \pmb y } \rangle / d )$ with $\begin{array} { r } { h ( x ) = \sum _ { k = 0 } ^ { \infty } \alpha _ { k } x ^ { k } , \alpha _ { k } \geq 0 } \end{array}$ for every $k \geq 0$ , and $\alpha _ { k } > 0$ for all $k \geq k _ { 0 }$

Under this assumption, the gradient flow estimator $\widehat { f } _ { \mathrm { t } }$ converges to the empirical Bayes denoiser $\widehat { f } ^ { \star }$ in $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ as $\mathrm { \Omega t } \to \infty$ . This convergence, however, does not automatically imply convergence of the test error (in $L ^ { 2 } ( \pi _ { d } ) )$ : we show in Appendix F.2 that, without further assumptions, the gradient flow estimator can develop a tail instability as $\mathrm { ~ t ~ } \to \infty$ , with $\mathcal { R } ( \widehat { f } _ { \mathrm { t } } )$ diverging along the trajectory.

Instead, we add a mild post-processing step that prevents such instabilities for any kernel satisfying Assumption 5 and any $\pi _ { 0 }$ with bounded second moment. Let A be a compact convex set containing all scaled samples $\{ \rho \pmb { x } _ { 0 , i } \} _ { i \in [ n ] }$ , and let $\mathsf { P } _ { A }$ denote the Euclidean projection onto A. We define the post-processed estimator by

$$
\widetilde { f } _ { \mathrm { t } } ( \pmb { u } ) : = \frac { \mathsf { P } _ { A } D _ { \widehat { f } _ { \mathrm { t } } } ( \pmb { u } ) - \pmb { u } } { \sigma } ,\tag{30}
$$

where $D _ { \widehat { f } _ { \mathrm { t } } } ( \mathbf { u } ) : = \mathbf { u } + \sigma \widehat { f } _ { \mathrm { t } } ( \mathbf { u } )$ . This is a mild post-processing step; its form is chosen to preserve the empirical Bayes denoiser interpolation limit, that is, $\widehat { \pmb f ^ { \star } } ( \pmb u ) = ( \mathsf { P } _ { A } { D } _ { \widehat { \pmb f ^ { \star } } } ( \pmb u ) - \pmb u ) / \sigma$

Proposition 4 (Gradient flow converges to the empirical Bayes denoiser). Fix $d , n ,$ the training samples $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ , and $\sigma > 0$ . Let $\widehat { f } _ { \mathrm { t } }$ solve the unregularized gradient flow (17) from initialization 0. Under Assumptions $\mathcal { Q }$ and 5, H is dense in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ and $L ^ { 2 } ( \mu _ { d } )$ , and the gradient flow converges to the empirical Bayes denoiser:

$$
S _ { 0 } \widehat { f } _ { \mathrm { t } } \longrightarrow \widehat { f } ^ { \star } \qquad i n { \cal L } ^ { 2 } ( \widehat { \mu } _ { d , n } ; \mathbb { R } ^ { d } ) \qquad \ \& s \mathrm { t }  \infty .
$$

Furthermore,

$$
\operatorname* { l i m } _ { t  \infty } \operatorname* { i n f } _ { \mathbf { \mathcal { R } } } ( \widehat { f } _ { t } ) \geq \mathcal { R } ( \widehat { f } ^ { \star } ) , \qquad \operatorname* { l i m } _ { t  \infty } \mathcal { R } ( \widetilde { f } _ { t } ) = \mathcal { R } ( \widehat { f } ^ { \star } ) .
$$

This proposition holds for any d, n. Next, we characterize the risk of the empirical Bayes denoiser in the linear high-dimensional scaling regime.

Theorem 3 (The memorization infinite-time limit). Assume Assumptions 1 and ${ \mathcal { Q } } ,$ and $\sigma > 0$ hold. Then, the risk of the empirical Bayes denoiser $\widehat { f } ^ { \star }$ is given by

$$
\left| \mathcal { R } ( \widehat { f } ^ { \star } ) - 2 \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \frac { \mathrm { T r } ( \Sigma ) } { d } \right| \prec \frac { 1 } { \sqrt { d } } .
$$

If we further assume Assumption 5, the post-processed gradient flow estimator (30) satisfies

$$
\left| \operatorname* { l i m } _ { \mathrm { t } \to \infty } \mathcal { R } ( \widetilde { f _ { \mathrm { t } } } ) - 2 \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \frac { \mathrm { T r } ( \Sigma ) } { d } \right| \prec \frac { 1 } { \sqrt { d } } .
$$

The proofs of the above results are given in Appendix F. The test risk of the empirical Bayes denoiser is twice the interpolation plateau of Phase II, and it is worth explaining where this doubling comes from. Of-sample, at a fresh noisy point ${ \pmb u } = \rho { \pmb x } + \sigma { \pmb z }$ , the softmax weights of the empirica score undergo a winner-take-all collapse: $\begin{array} { r } { \sum _ { i } \omega _ { i } ( { \pmb u } ) ^ { 2 }  ~ 1 } \end{array}$ (Lemma 29), so the empirical Bayes denoiser output is $( \rho \pmb { x } _ { 0 , i ^ { \star } ( \pmb { u } ) } - \pmb { u } ) / \sigma$ with $\pmb { x } _ { 0 , i ^ { \star } ( \pmb { u } ) }$ being the single training point nearest to $\mathbf { \delta } \mathbf { \boldsymbol { u } } / \rho$ . Its error thus has two components of identical size, $\| \pmb { x } \| ^ { 2 } / d \approx \tau _ { \Sigma }$ and $\| \pmb { x } _ { 0 , i ^ { \star } } \| ^ { 2 } / d \approx \tau _ { \Sigma }$ , which are near orthogonal in high dimensions, while Phase II’s plateau only has the first component.

## 4 Generation: the implicit bias of the reverse process

Section 3 characterizes the score model learned by the DSM regression problem at each noise level, with sharp asymptotic train and test errors. In generative modeling, however, we are interested in the distribution produced by composing the learned denoisers $\{ \widehat { f } _ { t } \} _ { t \in ( 0 , T ] }$ through the reverse dynamics (3). The standard reductions from score accuracy to sampling accuracy $[ \mathrm { C C L ^ { + } 2 3 } ,$ , BBDD24] require the $L ^ { 2 }$ score error to be small, whereas in our setting, the score error is of constant order and these bounds are not very informative. This section instead analyzes the reverse SDE driven by the trained score directly, focusing on the early-stopped (non-memorized) regime of gradient flow. We show that, with high probability along the reverse process, only the linearized part of the learned score is ever evaluated. The generated distribution is efectively Gaussian, with covariance only depending on the data through the empirical covariance $\widehat { \pmb { \Sigma } }$ . The nonlinear part of the score—the bumps that overfit the samples—is invisible to the sampler.

## 4.1 The learned reverse process and its linearized approximation

Reintroducing the difusion time, suppose that for each $t \in ( 0 , T ]$ we train by gradient flow at the noise level $( \rho _ { t } , \sigma _ { t } )$ for time $\mathrm { t } _ { t } ,$ and write $\widehat { f } _ { t } : = \widehat { f } _ { { \mathrm { t } } , t }$ and $\widehat { S } _ { t } : = \widehat { f } _ { t } / \bar { \sigma } _ { t }$ . The learned reverse process is

$$
\mathrm { d } \widehat { \mathbf { x } } _ { t } = - \big ( \widehat { \pmb { x } } _ { t } + 2 \widehat { \pmb { S } } _ { t } ( \widehat { \pmb { x } } _ { t } ) \big ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } \bar { \pmb { B } } _ { t } , \qquad \widehat { \pmb { x } } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) ,\tag{31}
$$

run backward on $[ t _ { 0 } , T ]$ , where $t _ { 0 } \in ( 0 , T )$ is the usual early-stopping time of the sampler and $\bar { B } _ { t }$ is a Brownian motion independent of the training data. We stress that the drift of (31) is a nonlinear random field: it is built from the trained gradient flow estimator, including the bumps. We assume the training-time schedule $t \mapsto \mathtt { t } _ { t }$ is deterministic and piecewise continuous on $[ t _ { 0 } , T ]$

For the pathwise comparison, we replace the trained gradient-flow score by a linearized gradientflow solution. For every $t > 0$ , let $\tilde { \mathcal { K } } _ { t } : L ^ { 2 } ( \widehat { \pi } _ { d , n } ) \to \bar { L } ^ { 2 } ( \widehat { \pi } _ { d , n } )$ be a modification of the efective empirical kernel operator $\widehat { \mathcal { K } } _ { \mathrm { e f f } }$ at noise level t that replaces the normalized trace $\tau _ { \Sigma }$ in the selfinduced regularization $\delta _ { n }$ by the sample norm of the input:

$$
\widetilde { K } _ { t } f ( x _ { 0 , i } , z ) : = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { z ^ { \prime } } \left[ \frac { \langle \rho _ { t } x _ { 0 , i } + \sigma _ { t } z , \rho _ { t } x _ { 0 , j } + \sigma _ { t } z ^ { \prime } \rangle } { d } f ( x _ { 0 , j } , z ^ { \prime } ) \right] + \widetilde { \delta } _ { i , t } \mathbb { E } _ { z ^ { \prime } } [ f ( x _ { 0 , i } , z ^ { \prime } ) ] ,
$$

where

$$
\widetilde { \delta } _ { i , t } : = \frac { h ( \rho _ { t } ^ { 2 } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d ) - \rho _ { t } ^ { 2 } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d } { n }
$$

for $i \in [ n ]$ . By Remark C.2, applied at $\rho _ { t } ^ { 2 } \lVert \boldsymbol { x } \rVert ^ { 2 } / d \geq 0$ , one has $\widetilde { \delta } _ { i , t } \geq 0$ for all $i \in [ n ]$ and $t > 0$ Then, for a unit vector $v ,$ define the linearized score estimator coordinate-wise as

$$
\widetilde { S } _ { t , v } ( \pmb { x } ) : = \frac { 1 } { \sigma _ { t } } \left[ S _ { \mathrm { e f f } , t } ^ { * } \int _ { 0 } ^ { \mathrm { t } _ { t } } \exp ( - \mathsf { s } \widetilde { \mathcal { K } } _ { t } ) z _ { v } \mathrm { d } \mathsf { s } \right] ( \pmb { x } ) ,\tag{32}
$$

where $S _ { \mathrm { e f f } , t } ^ { * } : L ^ { 2 } ( \widehat { \pi } _ { d , n } )  \mathcal { H }$ is the efective adjoint,

$$
S _ { \mathrm { e f f } , t } ^ { * } ( g ) : = \mathbb { E } _ { ( \pmb { x } , z ) \sim \widehat { \pi } _ { d , n } } \left[ \frac { \langle \cdot , \rho _ { t } \pmb { x } + \sigma _ { t } z \rangle } { d } g ( \pmb { x } , z ) \right] .
$$

Unwinding the definitions, we show in Lemma 30 that $\widetilde { S } _ { t } ( \pmb { x } ) = \widetilde { \Psi } _ { t } \pmb { x } / \sigma _ { t }$ for some bounded negative semidefinite matrix $ { \widetilde { \Psi } } _ { t } \in  { \mathbb { R } } ^ { d \times d }$ depending on t and the training data $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$

The linearized reverse process is defined as

$$
\mathrm { d } \widetilde { \pmb { x } } _ { t } = - \left( \widetilde { \pmb { x } } _ { t } + 2 \widetilde { \pmb { S } } _ { t } ( \widetilde { \pmb { x } } _ { t } ) \right) \mathrm { d } t + \sqrt { 2 } \mathrm { d } \bar { \pmb { B } } _ { t } , \qquad \widetilde { \pmb { x } } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) ,\tag{33}
$$

coupled to (31) through the same Brownian motion. Since its drift is linear and its initialization is Gaussian, $\widetilde { \pmb { x } } _ { t }$ is, conditionally on the training data, a Gaussian process.

## 4.2 Delocalization and pathwise linearization

The fitted score $\widehat { S } _ { t }$ and its linearization $\widetilde { S } _ { t }$ difer substantially only where the kernel’s nonlinearity is active: within the thin tubes around the training points, where the overlaps $\langle \pmb { x } , \pmb { x } _ { 0 , i } \rangle / d$ are of order one. The linearized process, however, is a Gaussian process in dimension d with bounded covariance; against any fixed direction, its overlaps are of order $d ^ { - 1 / 2 }$ . We prove (Lemma 31) that this delocalization holds uniformly along the trajectory and over all n training directions:

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { m a x } _ { i \in [ n ] } | \langle \widetilde { \pmb x } _ { t } , \pmb x _ { 0 , i } \rangle | ^ { k } \ \Big | \ X \right] ^ { 1 / k } \prec \sqrt { d } \qquad \mathrm { f o r ~ e v e r y ~ } k \in \mathbb { N } .
$$

Thus, with high probability, the sampling trajectory remains nearly orthogonal to every training sample, with overlap of order $d ^ { - 1 / 2 }$ . The sampler never visits the ‘bumps’ where the model memorized, so the process driven by the nonlinear score is indistinguishable from the linear surrogate. The following theorem makes this heuristic precise on path measures. To simplify the proofs, we assume a stronger condition on the kernel than Assumption 3.

Assumption 6 (Kernel regularity for reverse process). The kernel is a positive definite innerproduct kernel of the form $K ( { \pmb x } , { \pmb x } ^ { \prime } ) = h ( \langle { \pmb x } , { \pmb x } ^ { \prime } \rangle / d )$ . The kernel function $h : \mathbb { R }  \mathbb { R }$ is five times continuously diferentiable, with $h ( 0 ) = 0 , h ^ { \prime } ( 0 ) = 1$ , and $h ^ { ( k ) } ( 0 ) = 0 f o r k \in \{ 2 , 3 , 4 \}$ . Furthermore, there exists a constant $C \geq 1$ such that $| h ^ { ( 5 ) } ( x ) | \le C e ^ { C | x | }$ for all $x \in \mathbb { R }$

Let $\mathbb { R } _ { \infty } ^ { d } : = \mathbb { R } ^ { d } \cup \{ \infty \}$ be the one-point compactification of $\mathbb { R } ^ { d }$ . Conditionally on X, let xb denote the unique maximal solution of the empirical reverse SDE (31), extended beyond its explosion time by making ∞ absorbing. Let $\widetilde { \pmb x }$ be the unique global solution of the linearized reverse SDE (33), viewed as an $\mathbb { R } _ { \infty } ^ { d } .$ -valued process. We denote their conditional path laws on $[ t _ { 0 } , T ]$ by $\mathbb { P } _ { \widehat { \pmb { x } } } ( \cdot \mid \pmb { X } )$ and $\mathbb { P } _ { \widetilde { \pmb { x } } } ( \cdot \mid \pmb { X } )$ , respectively. Both are probability measures on $C ( [ t _ { 0 } , T ] ; \mathbb { R } _ { \infty } ^ { d } )$ , as the linearized process (33) is nonexplosive and any explosive empirical trajectory of (31) is continuous in the topology of the one-point compactification after being absorbed at $\infty$

Theorem 4 (Pathwise linearization of the reverse process). Let $T > 0$ and $t _ { 0 } \in ( 0 , T )$ be constants, and let $t \mapsto \mathtt { t } _ { t }$ be a deterministic piecewise-continuous schedule. Suppose that Assumptions 1, 2, and 6 hold. Then

$$
\operatorname { K L } \left( \mathbb { P } _ { \widetilde { \pmb { x } } } ( \cdot \mid \pmb { X } ) \parallel \mathbb { P } _ { \widehat { \pmb { x } } } ( \cdot \mid \pmb { X } ) \right) \prec \operatorname* { s u p } _ { t \in \left[ t _ { 0 } , T \right] } \mathsf { t } _ { t } ^ { 2 } d ^ { - 5 } \left( 1 + \mathsf { t } _ { t } ^ { 2 } + \mathsf { t } _ { t } ^ { 4 } d ^ { - 2 } \right) .
$$

The proof is given in Appendix G. In particular, if $\begin{array} { r } { \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathfrak { t } _ { t } \lesssim d ^ { 1 + c } } \end{array}$ for some $0 \leq c < 1 / 6$ ， then<sup>5</sup>

$$
\operatorname { K L } \left( \mathbb { P } _ { \widetilde { \pmb { x } } } ( \cdot \mid \pmb { X } ) \parallel \mathbb { P } _ { \widehat { \pmb { x } } } ( \cdot \mid \pmb { X } ) \right) \prec d ^ { - 1 + 6 c } = o _ { d , \mathbb { P } } ( 1 ) .
$$

Thus, the conditional KL divergence converges to zero in probability. Nonlinear overfitting is invisible to the sampler, and the generated distribution is efectively Gaussian with covariance determined by the linearized score.

![](images/36884b866694d004e88393cab37a3a284c0f8140e9b14c33ff6367b1cbc1019f.jpg)

![](images/6993377873b0c3d1c3264c4bc27460248797d09d65450c1894ce8c9445ef7b70.jpg)

![](images/c7ca356d1c7a5c3334eeaf6a793164babf48f04d902d3a6e990f71516a07c494.jpg)  
Figure 3: Recovery of two covariance scales along the linearized reverse process for $d = 7 2 0$ $n = 8 0 0 , T = 3 , t _ { 0 } = 1 0 ^ { - 3 } , h ( x ) = x + 0 . 0 0 5 x ^ { 5 } { \mathrm { ~ a n d ~ } } \pi _ { 0 } = \mathcal { N } ( 0 , \Sigma )$ where $\begin{array} { r } { \pmb { \Sigma } = \mathrm { d i a g } ( \frac { 1 7 } { 1 2 } \mathbf { I } _ { 5 0 4 } , \frac { 1 } { 3 6 } \mathbf { I } _ { 2 1 6 } ) } \end{array}$ has condition number 51 and $\tau _ { \Sigma } ~ = ~ 1$ . Left: A schedule $t \mapsto \ t _ { t }$ specifies the gradient-flow training time in (17) at every difusion time t. We consider the pointwise DSM-optimal schedule $\begin{array} { r } { \mathsf { t } _ { t } ^ { \star } \in \arg \operatorname* { m i n } _ { \mathsf { t } / d \in [ 1 0 ^ { - 1 } , 1 0 ^ { 2 } ] } \mathcal { R } _ { \mathrm { d e t } , t } ^ { ( \mathrm { t } ) } } \end{array}$ . We compare it with the constant schedule $\mathfrak { t } _ { t } ^ { \mathrm { c o n s t } } = \mathfrak { t } _ { t _ { 0 } } ^ { \star }$ and an exponential schedule $\mathfrak { t } _ { t } ^ { \mathrm { e x p } }$ interpolating between $\mathfrak { t } _ { T } ^ { \star }$ and $\mathfrak { t } _ { t _ { 0 } } ^ { \star }$ on a logarithmic difusion-time scale. Middle: Solid and dashed lines show, respectively, the average projected covariances $\mathrm { T r } ( \mathsf { P } _ { \pm } \widetilde { \pm } _ { t } ) / \mathsf { \Gamma }$ rank $( \mathsf { P } _ { \pm } )$ along the dynamics in (34), where $\mathsf { P } _ { + }$ and $\mathsf { P } _ { - }$ denote the projections onto the large- and small-eigenvalue eigenspaces of $\Sigma ,$ respectively. Filled and open circles show the corresponding deterministic predictions from Theorem $5 ,$ while gray dashed lines mark the population eigenvalues. Upper right: Solid curves show the conditional normalized Gaussian divergence $\mathrm { K L } ( \mathcal { N } ( 0 , \Sigma ) \| \mathcal { N } ( 0 , \widetilde { \Sigma } _ { t } ) ) / d .$ Markers show a natural spectral plug-in prediction using (35). Lower right: Solid curves show the cumulative population score MSE $\begin{array} { r l } { \int _ { t } ^ { T } ( \mathcal { R } _ { s } ( \widehat { \mathbf f } _ { \mathrm { t } _ { s } } ^ { \mathrm { l i n } } ) - \mathcal { R } _ { s } ^ { \star } ) / \sigma _ { s } ^ { 2 } \mathrm { d } s } & { { } } \end{array}$ , where $\widehat { f } _ { \mathrm { t } _ { s } } ^ { \mathrm { l i n } }$ is the linearized estimator from Lemma 30 and $\mathcal { R } _ { s } ^ { \star }$ is the Bayes population DSM risk at difusion time s. Markers show the deterministic prediction from Theorem 1.

## 4.3 The Gaussian generated distribution

It remains to identify the linear process (33), which is Gaussian with zero mean; its conditional covariance $\widetilde { \Sigma } _ { t } : = \mathbb { E } [ \widetilde { \pmb { x } } _ { t } \widetilde { \pmb { x } } _ { t } ^ { \mathsf { T } } \mid \pmb { X } ]$ satisfies, by Itˆo’s formula, the matrix ODE

$$
\frac { \mathrm { d } } { \mathrm { d } t } \widetilde { \Sigma } _ { t } = - 2 \widetilde { \Sigma } _ { t } - 2 \mathbb { E } [ \widetilde { S } _ { t } ( \widetilde { x } _ { t } ) \widetilde { x } _ { t } ^ { \top } \mid X ] - 2 \mathbb { E } [ \widetilde { x } _ { t } \widetilde { S } _ { t } ( \widetilde { x } _ { t } ) ^ { \top } \mid X ] - 2 { \mathbf { I } _ { d } } , \qquad \widetilde { \Sigma } _ { T } = { \mathbf { I } _ { d } } .\tag{34}
$$

The covariance $\widetilde { \pmb { \Sigma } } _ { t }$ is well-approximated by an auxiliary covariance flow with coeficient matrices that are spectral functions of the sample covariance ${ \widehat { \pmb { \Sigma } } } = \pmb { X } \pmb { X } ^ { \top } / n$ (see Lemma 36). Consequently, the covariance ODE of this auxiliary process diagonalizes in an eigenbasis of $\widehat { \pmb { \Sigma } }$ and reduces to a scalar ODE along its eigenvalues. Lemma 37 then identifies its deterministic equivalent in terms of the matrix-valued measure Ω from (27). Let $\psi _ { t } ( x , \mathfrak { t } _ { t } )$ be the time-dependent analogue of the spectral function $\psi ( x , \mathfrak { t } )$ defined in (28). We define the efective covariance

$$
\bar { \Sigma } _ { t } = \int _ { \mathbb { R } } \bar { \Lambda } _ { t } ( \lambda ) \ d \Omega ( \mathrm { d } \lambda ) ,
$$

where the scalar profile $\bar { \Lambda } _ { t } ( x )$ solves the backward ODE

$$
\frac { \mathrm { d } } { \mathrm { d } t } \bar { \Lambda } _ { t } ( x ) = - 2 \left( 1 + \frac { 2 } { \sigma _ { t } } \psi _ { t } ( x , \mathfrak { t } _ { t } ) \right) \bar { \Lambda } _ { t } ( x ) - 2 , \qquad \bar { \Lambda } _ { T } ( x ) = 1 .\tag{35}
$$

Theorem 5. Let $T > 0$ and $t _ { 0 } \in ( 0 , T )$ be constants, and let $t \mapsto \mathtt { t } _ { t }$ be a deterministic piecewisecontinuous training-time schedule. Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 6 hold. Then, for every deterministic $0 \preceq A \in \mathbb { R } ^ { d \times d }$ such that $\| A \| _ { * } \leq 1$ and constants $\epsilon , D > 0$ , there exists a constant $C > 0$ such that

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left| \operatorname { T r } \bigl ( \pmb { A } ( \widetilde { \pmb { \Sigma } } _ { t } - \bar { \pmb { \Sigma } } _ { t } ) \bigr ) \right| \lesssim d ^ { \epsilon - \frac { 1 } { 2 } } \left( 1 + d ^ { - 1 } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathbf { t } _ { t } \right)
$$

with probability at least $1 - C d ^ { - D }$

The proof of this theorem can be found in Appendix G.4.

Theorems 4 and 5 together describe the output distribution for early-stopped gradient flow score estimators. Conditionally on the training data, the generated samples are approximately Gaussian, whose limiting covariance has the deterministic equivalent given by $\bar { \Sigma } _ { t _ { 0 } }$ . Equivalently, every fixed deterministic projection of the generated samples is asymptotically Gaussian with covariance $\bar { \Sigma } _ { t _ { 0 } } .$ where $\bar { \Sigma } _ { t _ { 0 } }$ is the composition of the Marchenko–Pastur-type map $\Sigma \mapsto \Omega$ with the ODE flow (35).

As discussed in the introduction, combining the results of Sections 3 and 4 allows us to examine the relationship between generalization to the population DSM objective and generalization to the target distribution. In Figure 3, we consider a Gaussian target distribution $\boldsymbol { \pi } _ { 0 } = \mathcal { N } ( 0 , \boldsymbol { \Sigma } )$ , where Σ has two eigenspaces and condition number 51, under three early-stopping schedules (left panel). The middle panel shows the evolution, throughout the reverse process, of the learned covariance $\widetilde { \pmb { \Sigma } } _ { t }$ projected onto each eigenspace for each schedule. The right panel shows the cumulative test score MSE and the KL divergence $\mathrm { K L } \big ( \pi _ { 0 } \| \mathcal { N } ( 0 , \widetilde { \pmb { \Sigma } } _ { t } ) \big )$ along the reverse difusion process. Notably, an early-stopping schedule that is suboptimal in terms of the DSM objective can nevertheless yield a (slightly) smaller KL divergence when the reverse process is stopped early. We consider exploring further the relationship between score-matching performance and generation quality to be an important future research direction.

## 5 Conclusion and future directions

Difusion models present a statistical puzzle: their training objective is minimized by a memorizing solution, and available sample sizes are far too small for consistent estimation of an unrestricted score function. Yet models trained in practice generalize, producing genuinely new samples that reproduce both global and local statistics of the target distribution. Resolving this puzzle requires characterizing what practical training algorithms learn: which score functions they fit and which distributions the resulting reverse difusion dynamics generate.

Motivated by analogous successes in the theory of supervised learning, we studied lazy denoising score matching in the proportional high-dimensional regime. At each noise level, we fit a kernel model corresponding to a neural network trained in the lazy regime. We characterize the estimator learned by gradient flow and derive precise asymptotics for its training and test errors throughout the optimization trajectory. The estimator passes through three distinct phases (Figure 1):

$$
\begin{array} { r } { \underbrace { \mathrm { t = } \Theta ( d ) \mathrm { : ~ g e n e r a l i z a t i o n } } _ { \mathrm { P h a s e l : ~ f i e } \nu ^ { \mathrm { o p e t } } ( X X ^ { \top } / n ) u } \quad \longrightarrow \quad \underbrace { \mathrm { t = } d ^ { 1 + \Theta ( 1 ) } \mathrm { : ~ o v e r f i t t i n g } } _ { \mathrm { P h a s e ~ I I : ~ f i e - } u / \sigma } \quad \longrightarrow \quad \underbrace { \mathrm { t = } d ^ { \omega ( 1 ) } \mathrm { : ~ m e m o r i z a t i o n } } _ { \mathrm { F h a s e l I : ~ f i e } \langle \rho x \mathrm { o } , i \mathrm { : } u \mathrm { : } u \mathrm { ) / \sigma } } . } \\ { \mathrm { c o v a r i a n c e ~ l e a r n i n g } \quad \qquad } \end{array}
$$

The population risk initially decreases and attains an optimum at stopping time $\mathfrak { t } ^ { \star } = \Theta ( d )$ , corresponding to a spectral estimator applied to the empirical data covariance. It then rises to ${ \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } } \tau _ { \Sigma }$ as the estimator loses its dependence on the data and approaches the pure-noise score, before ultimately reaching $2 \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \tau _ { \Sigma }$ upon memorizing the training set.

Our analysis separates three phenomena that are often conflated: generalization in sampling, overfitting (vanishing training objective), and memorization (reproducing training samples). In particular, overfitting does not imply memorization: in the lazy high-dimensional regime, these phenomena are separated by an exponential gap in training time and a model can interpolate its training objective while continuing to generate genuinely new samples. At the level of score estimation, however, interpolation requires the lazy model to discard some of the statistical structure learned from the data. Overfitting therefore degrades population-level generalization, and the separation between the generalization and overfitting timescales is governed by the kernel nonlinearity and the data covariance. At the level of generation, by contrast, a form of benign overfitting remains possible. The learned score decomposes into a global linear component and localized corrections around the training samples. Because the reverse difusion trajectory is delocalized, it typically avoids these corrections, so the SDE stays efectively linear and produces a Gaussian distribution.

Our results suggest several natural directions for future research:

(1) Beyond the proportional regime. Our analysis focuses on $n \asymp d ,$ where the learned score is asymptotically at most linear. In supervised learning, the polynomial scaling $n \asymp d ^ { \ell }$ gives rise to a precise hierarchy: inner-product kernel methods learn exactly the degree-≤ ℓ polynomial component of the target [GMMM21, MMM22, MM24]. We expect an analogous hierarchy for lazy generative models, in which the generated distribution progressively acquires non-Gaussian corrections and captures higher-order moments of $\pi _ { 0 }$ as $\ell$ increases. Establishing this generative hierarchy—and, in particular, extending the delocalization argument to scores with higher-degree components—is a natural next step. Although these polynomial samplesize regimes may be less directly relevant to practice, they would help delineate the capabilities of lazy training from those of feature learning.

(2) Feature learning. At proportional sample sizes, any non-Gaussian structure captured by the generated distribution must arise from feature learning or architectural inductive bias. Solvable analyses of feature learning in shallow autoencoders [CZ23, CKVEZ24, CPL25] and the analysis of convolutional score models [KG24] provide important first steps. A theory coupling feature learning to the reverse process analysis developed here could determine which learned features are actually encountered by the sampler and, consequently, which non-Gaussian statistics are acquired first and at what cost in samples and optimization time.

(3) Low-dimensional and structured data. Our assumptions exclude data supported on, or concentrated near, low-dimensional manifolds, where memorization and generalization interact with the geometry of the support [Pid22, AAL<sup>+</sup>25, GM26, WZZ<sup>+</sup>25, RKW<sup>+</sup>25]. A natural next step is to extend our analysis to these structured data settings.

(4) Coupled noise levels and practical samplers. We train an independent RKHS model at each noise level, whereas practical difusion models use a single time-conditioned network. Parameter sharing couples the learning problems across noise levels and understanding what changes in this case is an important direction.

## Acknowledgments and use of AI

J.S. was supported in part by the Yale Ofice of the Provost. We used AI tools to assist in constructing the example of an unstable ridgeless limit in Appendix F.2 and in writing code to generate the figures. Their use was otherwise limited to light editing and proofreading. We take full responsibility for the content of the paper.

## References

[AAD<sup>+</sup>24] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, et al., Accurate structure prediction of biomolecular interactions with AlphaFold 3, Nature 630 (2024), no. 8016, 493–500.

[AAL<sup>+</sup>25] Beatrice Achilli, Luca Ambrogioni, Carlo Lucibello, Marc M´ezard, and Enrico Ventura, Memorization and generalization in generative difusion under the manifold hypothesis, Journal of Statistical Mechanics: Theory and Experiment 2025 (2025), no. 7, 073401.

[Ada15] Radoslaw Adamczak, A note on the Hanson–Wright inequality for random vectors with dependencies.

[AKT19] Alnur Ali, Zico Kolter, and Ryan J. Tibshirani, A continuous-time view of early stopping for least squares regression, The 22nd International Conference on Artificial Intelligence and Statistics, PMLR, 2019, pp. 1370–1378.

[And82] Brian D. O. Anderson, Reverse-time difusion equation models, Stochastic Processes and their Applications 12 (1982), no. 3, 313–326.

[AVS<sup>+</sup>24] Beatrice Achilli, Enrico Ventura, Gianluigi Silvestri, Bao Pham, Gabriel Raya, et al., Losing dimensions: geometric memorization in generative difusion, arXiv preprint arXiv:2410.08727 (2024).

[BBBM24] Giulio Biroli, Tony Bonnaire, Valentin De Bortoli, and Marc M´ezard, Dynamical regimes of difusion models, Nature Communications 15 (2024), no. 1, 9957.

[BBDD24] Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis, Nearly d-linear convergence bounds for difusion models via stochastic localization, International Conference on Learning Representations, vol. 2024, 2024, pp. 36916–36936.

[BDK<sup>+</sup>25] Ricardo Baptista, Agnimitra Dasgupta, Nikola B. Kovachki, Assad Oberai, and Andrew M. Stuart, Memorization and regularization in generative difusion models, arXiv preprint arXiv:2501.15785 (2025).

[BFS24] Alexandre Belloni, Ethan X. Fang, and Shuting Shen, Anti-concentration inequalities for the diference of maxima of Gaussian random vectors, arXiv preprint arXiv:2408.13348 (2024).

[BGL14] Dominique Bakry, Ivan Gentil, and Michel Ledoux, Analysis and geometry of Markov difusion operators, vol. 103, Springer, 2014.

[BHMM19] Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal, Reconciling modern machine-learning practice and the classical bias–variance trade-of, Proceedings of the National Academy of Sciences 116 (2019), no. 32, 15849–15854.

[BLLT20] Peter L. Bartlett, Philip M. Long, G´abor Lugosi, and Alexander Tsigler, Benign overfitting in linear regression, Proceedings of the National Academy of Sciences 117 (2020), no. 48, 30063–30070.

[BM23] Giulio Biroli and Marc M´ezard, Generative difusion in very large dimensions, Journal of Statistical Mechanics: Theory and Experiment 2023 (2023), no. 9, 093402.

[BMG26] Lorenzo Bardone, Claudia Merger, and Sebastian Goldt, A theory of learning data statistics in difusion models, from easy to hard, Proceedings of the 43rd International Conference on Machine Learning, PMLR, 2026.

[BMR21] Peter L. Bartlett, Andrea Montanari, and Alexander Rakhlin, Deep learning: a statistical viewpoint, Acta Numerica 30 (2021), 87–201.

[Bor22] Valentin De Bortoli, Convergence of denoising difusion models under the manifold hypothesis, Transactions on Machine Learning Research (2022).

[BPMB26] Sam Buchanan, Druv Pai, Yi Ma, and Valentin De Bortoli, On the edge of memorization in difusion models, Advances in Neural Information Processing Systems 38 (2026), 96113–96157.

[BUBM26] Tony Bonnaire, Rapha¨el Urfin, Giulio Biroli, and Marc M´ezard, Why difusion models don’t memorize: the role of implicit dynamical regularization in training, Advances in Neural Information Processing Systems 38 (2026), 141266–141286.

[CCDR26] Fan Chen, Sinho Chewi, Constantinos Daskalakis, and Alexander Rakhlin, Highaccuracy sampling for difusion models and log-concave distributions, arXiv preprint 2602.01338 (2026).

[CCL<sup>+</sup>23] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, et al., Sampling is as easy as learning the score: theory for difusion models with minimal data assumptions, The Eleventh International Conference on Learning Representations, 2023.

[CD11] Romain Couillet and Merouane Debbah, Random matrix methods for wireless communications, Cambridge University Press, 2011.

[CDS25] Giovanni Conforti, Alain Durmus, and Marta Gentiloni Silveri, KL convergence guarantees for score difusion models under minimal data assumptions, SIAM Journal on Mathematics of Data Science 7 (2025), no. 1, 86–109.

[CHN<sup>+</sup>23] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, et al., Extracting training data from difusion models, 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 5253–5270.

[Cho22] Cl´ement Chouard, Quantitative deterministic equivalent of sample covariance matrices with a general dependence structure, arXiv preprint arXiv:2211.13044 (2022).

[CHZW23] Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang, Score approximation, estimation and distribution recovery of difusion models on low-dimensional data, International Conference on Machine Learning, PMLR, 2023, pp. 4672–4712.

[CKVEZ24] Hugo Cui, Florent Krzakala, Eric Vanden-Eijnden, and Lenka Zdeborov´a, Analysis of learning a flow-based generative model from limited sample complexity, The Twelfth International Conference on Learning Representations, 2024.

[CM24] Chen Cheng and Andrea Montanari, Dimension free ridge regression, The Annals of Statistics 52 (2024), no. 6, 2879–2912.

[COB19] L´ena¨ıc Chizat, Edouard Oyallon, and Francis Bach, On lazy training in diferentiable programming, Advances in Neural Information Processing Systems 32 (2019).

[CPL25] Hugo Cui, Cengiz Pehlevan, and Yue M. Lu, A solvable model of learning generative difusion: theory and insights, The Thirty-Ninth Annual Conference on Neural Information Processing Systems, 2025.

[CPL26] , A solvable model of learning generative difusion: theory and insights, Advances in Neural Information Processing Systems 38 (2026), 5253–5296.

[CS13] Xiuyuan Cheng and Amit Singer, The spectrum of random inner-product kernel matrices, Random Matrices: Theory and Applications 2 (2013), no. 04, 1350010.

[CZ23] Hugo Cui and Lenka Zdeborov´a, High-dimensional asymptotics of denoising autoencoders, Advances in Neural Information Processing Systems, vol. 36, Curran Associates, Inc., 2023, pp. 11850–11890.

[DKXZ24] Zehao Dou, Subhodh Kotekal, Zhehao Xu, and Harrison H. Zhou, From optimal score matching to optimal sampling, arXiv:2409.07032 (2024).

[Efr11] Bradley Efron, Tweedie’s formula and selection bias, Journal of the American Statistical Association 106 (2011), no. 496, 1602–1614.

[EN00] Klaus-Jochen Engel and Rainer Nagel, One-parameter semigroups for linear evolution equations, Springer, 2000.

[FDDS26] Tyler Farghly, Benjamin Dupuis, Alain Durmus, and Umut Simsekli, Benign overfitting does not occur in difusion models, arXiv:2607.02671 (2026).

[FMPW26] Zhou Fan, Renyuan Ma, Elliot Paquette, and Zhichao Wang, Anisotropic local law for non-separable sample covariance matrices, arXiv preprint arXiv:2602.17960 (2026).

[FSW25] Alessandro Favero, Antonio Sclocchi, and Matthieu Wyart, Bigger isn’t always memorizing: early stopping overparameterized difusion models, arXiv preprint arXiv:2505.16959 (2025).

[GAH26] Reza Ghane, Danil Akhtiamov, and Babak Hassibi, Precise performance of linear denoisers in the proportional regime, arXiv preprint 2603.18483 (2026).

[GDP<sup>+</sup>25] Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, et al., On memorization in difusion models, Transactions on Machine Learning Research (2025).

[GM26] Anand Jerry George and Nicolas Macris, Asymptotic learning curves for difusion models with random features score and manifold data, arXiv preprint 2603.22962 (2026).

[GMMM20] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari, When do neural networks outperform kernel methods?, Advances in Neural Information Processing Systems 33 (2020), 14820–14830.

[GMMM21] , Linearized two-layers neural networks in high dimension, The Annals of Statistics 49 (2021), no. 2, 1029–1054.

[Gro75] Leonard Gross, Logarithmic Sobolev inequalities, American Journal of Mathematics 97 (1975), no. 4, 1061–1083.

[GT00] Fritz Gesztesy and Eduard Tsekanovskii, On matrix–valued Herglotz functions, Mathematische Nachrichten 218 (2000), no. 1, 61–138.

[GVM25] Anand Jerry George, Rodrigo Veiga, and Nicolas Macris, Analysis of difusion models for manifold data, 2025 IEEE International Symposium on Information Theory (ISIT), 2025, pp. 1–6.

[GVM26] , Denoising score matching with random features: insights on difusion models from precise learning curves, The 29th International Conference on Artificial Intelligence and Statistics, 2026.

[Hal25] Indranil Halder, A solvable generative model with a linear, one-step denoiser, ICML 2025 Workshop on High-Dimensional Learning Dynamics, 2025.

[HD05] Aapo Hyv¨arinen and Peter Dayan, Estimation of non-normalized statistical models by score matching, Journal of Machine Learning Research 6 (2005), no. 4.

[HJA20] Jonathan Ho, Ajay Jain, and Pieter Abbeel, Denoising difusion probabilistic models, Advances in Neural Information Processing Systems 33 (2020), 6840–6851.

[HLN07] Walid Hachem, Philippe Loubaton, and Jamal Najim, Deterministic equivalents for certain functionals of large random matrices, The Annals of Applied Probability 17 (2007), no. 1, 875–930.

[HMRT22] Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J. Tibshirani, Surprises in high-dimensional ridgeless least squares interpolation, The Annals of Statistics 50 (2022), no. 2, 949–986.

[HP86] Ulrich G. Haussmann and Etienne Pardoux, <sup>´</sup> Time reversal of difusions, The Annals of Probability (1986), 1188–1205.

[JGH18] Arthur Jacot, Franck Gabriel, and Cl´ement Hongler, Neural tangent kernel: convergence and generalization in neural networks, Advances in Neural Information Processing Systems 31 (2018).

[KAAL22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine, Elucidating the design space of difusion-based generative models, Advances in Neural Information Processing Systems 35 (2022), 26565–26577.

[Kar10] Noureddine El Karoui, The spectrum ofkernel random matrices, The Annals of Statistics 38 (2010), no. 1, 1–50.

[KG24] Mason Kamb and Surya Ganguli, An analytic theory of creativity in convolutional difusion models, arXiv:2412.20292 (2024).

[KGSM24] Zahra Kadkhodaie, Florentin Guth, Eero Simoncelli, and St´ephane Mallat, Generalization in difusion models arises from geometry-adaptive harmonic representations, International Conference on Learning Representations, vol. 2024, 2024, pp. 46543– 46567.

[KK26] Tim Kaiser and Markus Kollmann, Difusion models memorize in training—and generalize in inference, arXiv:2603.13419 (2026).

[KPH<sup>+</sup>21] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro, DifWave: a versatile difusion model for audio synthesis, International Conference on Learning Representations, 2021.

[LC24] Marvin Li and Sitan Chen, Critical windows: non-asymptotic theory for feature emergence in difusion models, Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 235, PMLR, 2024, pp. 27474–27498.

[LDQ24] Xiang Li, Yixiang Dai, and Qing Qu, Understanding generalizability of difusion models requires rethinking the hidden Gaussian structure, Advances in Neural Information Processing Systems, vol. 37, Curran Associates, Inc., 2024, pp. 57499–57538.

[LLT23] Holden Lee, Jianfeng Lu, and Yixin Tan, Convergence of score-based generative modeling for general data distributions, International Conference on Algorithmic Learning Theory, PMLR, 2023, pp. 946–985.

[LR20] Tengyuan Liang and Alexander Rakhlin, Just interpolate: kernel “ridgeless” regression can generalize, The Annals of Statistics 48 (2020), no. 3.

[LVP26] Hugo Latourelle-Vigeant and Elliot Paquette, Dyson equation for correlated linearizations and test error of random features regression, Random Matrices: Theory and Applications 15 (2026), no. 01, 2550026.

[MG26a] Claudia Merger and Sebastian Goldt, Generalization dynamics of linear difusion models, arXiv preprint 2505.24769 (2026).

[MG26b] , Local coverage governs memorization in difusion models, arXiv:2606.14390 (2026).

[MM22] Theodor Misiakiewicz and Song Mei, Learning with convolution and pooling operations in kernel methods, Advances in Neural Information Processing Systems 35 (2022), 29014–29025.

[MM24] Theodor Misiakiewicz and Andrea Montanari, Six lectures on linearized neural networks, Journal of Statistical Mechanics: Theory and Experiment 2024 (2024), no. 10, 104006.

[MMM22] Song Mei, Theodor Misiakiewicz, and Andrea Montanari, Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration, Applied and Computational Harmonic Analysis 59 (2022), 3–84.

[MS24] Theodor Misiakiewicz and Basil Saeed, A non-asymptotic theory of kernel ridge regression: deterministic equivalents, test error, and GCV estimator, arXiv preprint arXiv:2403.08938 (2024).

[MVSS20] Vidya Muthukumar, Kailas Vodrahalli, Vignesh Subramanian, and Anant Sahai, Harmless interpolation of noisy data in regression, IEEE Journal on Selected Areas in Information Theory 1 (2020), no. 1, 67–83.

[OAS23] Kazusato Oko, Shunta Akiyama, and Taiji Suzuki, Difusion models are minimax optimal distribution estimators, International Conference on Machine Learning, PMLR, 2023, pp. 26517–26582.

[PG25] Emile Pierret and Bruno Galerne, <sup>´</sup> Difusion models for Gaussian distributions: exact solutions and Wasserstein errors, Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 267, PMLR, 7 2025, pp. 49355–49381.

[Pid22] Jakiw Pidstrigach, Score-based generative models detect manifolds, Advances in Neural Information Processing Systems 35 (2022), 35852–35865.

[RA23] Gabriel Raya and Luca Ambrogioni, Spontaneous symmetry breaking in generative difusion models, Thirty-Seventh Conference on Neural Information Processing Systems, 2023.

[RBL<sup>+</sup>22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer, High-resolution image synthesis with latent difusion models, 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2022, pp. 10674–10685.

[RKW<sup>+</sup>25] Brendan Ross, Hamidreza Kamkari, Tongzi Wu, Rasa Hosseinzadeh, Zhaoyan Liu, et al., A geometric framework for understanding memorization in generative models, International Conference on Learning Representations, vol. 2025, 2025, pp. 34773– 34805.

[Rob92] Herbert E. Robbins, An empirical Bayes approach to statistics, Breakthroughs in Statistics: Foundations and Basic Theory, Springer, 1992, pp. 388–394.

[RWY14] Garvesh Raskutti, Martin J. Wainwright, and Bin Yu, Early stopping and nonparametric regression: an optimal data-dependent stopping rule, The Journal of Machine Learning Research 15 (2014), no. 1, 335–366.

[SC08] Ingo Steinwart and Andreas Christmann, Support vector machines, Springer, 2008.

[SDWMG15] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli, Deep unsupervised learning using nonequilibrium thermodynamics, International Conference on Machine Learning, PMLR, 2015, pp. 2256–2265.

[SE19] Yang Song and Stefano Ermon, Generative modeling by estimating gradients of the data distribution, Advances in Neural Information Processing Systems 32 (2019).

[SFW25] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart, A phase transition in difusion models reveals the hierarchical nature of data, Proceedings of the National Academy of Sciences 122 (2025), no. 1, e2408799121.

[SHN<sup>+</sup>18] Daniel Soudry, Elad Hofer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro, The implicit bias of gradient descent on separable data, Journal of Machine Learning Research 19 (2018), Paper No. 70, 57.

[SSDK<sup>+</sup>21] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, et al., Score-based generative modeling through stochastic diferential equations, International Conference on Learning Representations, 2021.

[SSG<sup>+</sup>23a] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein, Difusion art or digital forgery? investigating data replication in difusion models, 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2023, pp. 6048–6058.

[SSG<sup>+</sup>23b] , Understanding and mitigating copying in difusion models, Advances in Neural Information Processing Systems 36 (2023), 47783–47803.

[TB23] Alexander Tsigler and Peter L. Bartlett, Benign overfitting in ridge regression, Journal of Machine Learning Research 24 (2023), Paper No. [123], 76.

[Tro26] Joel A. Tropp, Universality laws for random matrices via exchangeable counterparts, arXiv preprint arXiv:2603.05803 (2026).

[Vas25] John J. Vastola, Generalization through variance: how noise shapes inductive biases in difusion models, The Thirteenth International Conference on Learning Representations, 2025.

[vEM90] Stephanus J. L. van Eijndhoven and J. L. H. Meyers, New orthogonality relations for the Hermite polynomials and related Hilbert spaces, Journal of Mathematical Analysis and Applications 146 (1990), no. 1, 89–98.

[Vin11] Pascal Vincent, A connection between score matching and denoising autoencoders, Neural Computation 23 (2011), no. 7, 1661–1674.

[Wai19] Martin J. Wainwright, High-dimensional statistics: a non-asymptotic viewpoint, vol. 48, Cambridge University Press, 2019.

[WJB<sup>+</sup>23] Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, et al., De novo design of protein structure and function with RFdifusion, Nature 620 (2023), no. 7976, 1089–1100.

[WMBB25] Yu-Han Wu, Pierre Marion, G´erard Biau, and Claire Boyer, Taking a big step: large learning rates in denoising score matching prevent memorization, Proceedings of the 38th Conference on Learning Theory, Proceedings of Machine Learning Research, vol. 291, PMLR, 2025, pp. 5718–5756.

[WP26] Binxu Wang and Cengiz Pehlevan, An analytical theory of spectral bias in the learning dynamics of difusion models, Advances in Neural Information Processing Systems 38 (2026), 95865–95963.

[WV23] Binxu Wang and John J. Vastola, The hidden linear structure in score-based models and its application, arXiv:2311.10892 (2023).

[WV24] , The unreasonable efectiveness of Gaussian score approximation for difusion models and its applications, Transactions on Machine Learning Research (2024).

[WWY24] Andr´e Wibisono, Yihong Wu, and Kaylee Yingxi Yang, Optimal score estimation via empirical Bayes smoothing, The Thirty-Seventh Annual Conference on Learning Theory, PMLR, 2024, pp. 4958–4991.

[WZVP26] Binxu Wang, Jacob Zavatone-Veth, and Cengiz Pehlevan, A random matrix theory perspective on the consistency of difusion models, arXiv:2602.02908 (2026).

[WZZ<sup>+</sup>25] Peng Wang, Huijie Zhang, Zekai Zhang, Siyi Chen, Yi Ma, et al., Difusion models learn low-dimensional distributions via subspace clustering, 2025 IEEE 10th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP), IEEE, 2025, pp. 211–215.

[Yat95] Roy D. Yates, A framework for uplink power control in cellular radio systems, IEEE Journal on Selected Areas in Communications 13 (1995), no. 7, 1341–1347.

[YCKR23] TaeHo Yoon, Joo Young Choi, Sehyun Kwon, and Ernest K. Ryu, Difusion probabilistic models generalize when they fail to memorize, ICML 2023 Workshop on Structured Probabilistic Inference and Generative Modeling, 2023.

[YRC07] Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto, On early stopping in gradient descent learning, Constructive Approximation 26 (2007), no. 2, 289–315.

[YZTC26] Zeqi Ye, Qijie Zhu, Molei Tao, and Minshuo Chen, Provable separations between memorization and generalization in difusion models, International Conference on Learning Representations, vol. 2026, 2026, pp. 143375–143425.

[ZBH<sup>+</sup>17] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals, Understanding deep learning requires rethinking generalization, International Conference on Learning Representations, 2017.

[Zho08] Ding-Xuan Zhou, Derivative reproducing properties for kernel methods in learning theory, Journal of computational and Applied Mathematics 220 (2008), no. 1-2, 456– 463.

[ZYLL24] Kaihong Zhang, Heqi Yin, Feng Liang, and Jingbo Liu, Minimax optimality of scorebased difusion models: beyond the density lower bound assumptions, Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 235, PMLR, 2024, pp. 60134–60178.

[ZZL<sup>+</sup>24] Huijie Zhang, Jinfan Zhou, Yifu Lu, Minzhe Guo, Peng Wang, et al., The emergence of reproducibility and consistency in difusion models, Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 235, PMLR, 2024, pp. 60558–60590.

## A Ridge-regularized denoising score matching

This appendix presents the exact asymptotics for the train and test risks of the kernel ridge estimator $\widehat { f } _ { \kappa , v }$ of (20) at a fixed noise level $\rho , \sigma > 0$ , across the three regimes of the (deterministic) penalty κ. Under the calibration $\kappa = d / \mathrm { t }$ , the three regimes correspond to the three phases of Section 3: moderate κ (equivalently $\displaystyle \mathbf { t } \lesssim d ^ { 1 + c } )$ , polynomially vanishing κ (polynomial t; training loss interpolation), and the ridgeless limit $\kappa  0 ^ { + }$ (the memorization endpoint in Theorem 3, proved in Appendix F).

## A.1 Moderate regularization

For $\kappa = \Omega _ { d } ( d ^ { - c } )$ with a small constant $c > 0$ , the ridge estimator is stable and the kernel linearization (Proposition 1) of Section 3.1 applies directly. We work under Assumptions 1, 2, and $^ { 3 , }$ and recall $\tau _ { \Sigma } = \mathrm { T r } ( \Sigma ) / d$ and the efective operators (22). By Proposition 2, the irreducible train error is exponentially small.

We first define the deterministic equivalents. Let $M _ { \kappa } \in \mathbb { R } ^ { d \times d }$ and $\mu _ { \kappa } > 0$ solve the system

$$
M _ { \kappa } : = \left( \frac { \Sigma } { 1 + \mu _ { \kappa } } + \kappa _ { \kappa } ^ { \star } { \mathbf I } _ { d } \right) ^ { - 1 } , \qquad \mu _ { \kappa } = \frac { 1 } { n } \mathrm { T r } ( \Sigma M _ { \kappa } ) ,\tag{36}
$$

where the auxiliary parameters $\kappa _ { \kappa } ^ { \star }$ and $\chi _ { \kappa }$ are

$$
\chi _ { \kappa } : = \frac { \kappa } { d } + \frac { h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma } } { n } = \frac { \kappa } { d } + \delta _ { n } , \qquad \kappa _ { \kappa } ^ { \star } : = \frac { d ( \sigma ^ { 2 } + \kappa ) \chi _ { \kappa } } { \rho ^ { 2 } \kappa } = \frac { \sigma ^ { 2 } + \kappa } { \rho ^ { 2 } } \left( 1 + \frac { d } { n } \frac { h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma } } { \kappa } \right)\tag{37}
$$

Note that $\kappa _ { \kappa } ^ { \star } \ge \sigma ^ { 2 } / \rho ^ { 2 } > 0$ is bounded away from 0 whenever $\sigma$ is. Existence and uniqueness of the solution to (36) follow from standard interference-function arguments [Yat95, CD11]. Define further the second-order equivalent

$$
M _ { \kappa } ^ { ( 2 ) } : = M _ { \kappa } ^ { 2 } + \frac { \frac 1 n \mathrm { T r } ( \Sigma M _ { \kappa } ^ { 2 } ) } { ( 1 + \mu _ { \kappa } ) ^ { 2 } - \frac 1 n \mathrm { T r } ( \Sigma ^ { 2 } M _ { \kappa } ^ { 2 } ) } M _ { \kappa } \Sigma M _ { \kappa } .
$$

The deterministic equivalent train and test errors are given by

$$
\mathcal { L } _ { \mathrm { d e t } } ^ { ( \kappa ) } : = 1 - \frac { \sigma ^ { 2 } } { \rho ^ { 2 } \kappa } \left( 2 \chi _ { \kappa } - \frac { \kappa } { d } \right) \mathrm { T r } ( M _ { \kappa } ) + \frac { \sigma ^ { 2 } \chi _ { \kappa } \left( \sigma ^ { 2 } \chi _ { \kappa } d - \kappa ( \sigma ^ { 2 } + \kappa ) \right) } { \rho ^ { 4 } \kappa ^ { 2 } } \mathrm { T r } ( M _ { \kappa } ^ { ( 2 ) } ) ,\tag{38}
$$

$$
\mathcal { R } _ { \mathrm { d e t } } ^ { ( \kappa ) } : = 1 - \frac { 2 \sigma ^ { 2 } \chi _ { \kappa } } { \rho ^ { 2 } \kappa } \mathrm { T r } ( M _ { \kappa } ) + \frac { \sigma ^ { 2 } \chi _ { \kappa } ^ { 2 } d } { \rho ^ { 4 } \kappa ^ { 2 } } \mathrm { T r } ( \Sigma _ { \sigma } M _ { \kappa } ^ { ( 2 ) } ) ,\tag{39}
$$

where $\begin{array} { r } { \Sigma _ { \sigma } = \rho ^ { 2 } \Sigma + \sigma ^ { 2 } \mathbf { I } _ { d } } \end{array}$ . Both quantities are explicit functions of the spectrum of Σ, computable by solving the scalar fixed point for $\mu _ { \kappa }$ . These asymptotics are equivalent to the asymptotics in the equivalent linearized risk (23) and (24) with linearized kernel $\hat { \mathcal { K } } _ { \mathrm { e f f } }$ and linearized RKHS $\mathcal { H } _ { \mathrm { e f f } } ;$ the nonlinear part of the kernel only enters through the shift $\delta _ { n }$ in $\chi _ { \kappa }$

Theorem 6 (Ridge DSM with moderate regularization). Suppose that Assumptions 1, 2 and $\mathcal { B }$ hold. Then,

$$
| \mathcal { L } ( \widehat { f } _ { \kappa } ; 0 ) - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { \kappa + 1 } { \kappa \sqrt { d } } , \qquad | \mathcal { R } ( \widehat { f } _ { \kappa } ) - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { \kappa ^ { 3 } + 1 } { \kappa ^ { 3 } \sqrt { d } } .
$$

Furthermore, $i f h ( x ) = x$ for all $x \in \mathbb { R }$ , there exists a constant $c > 0$ such that

$$
| \mathcal { L } ( \widehat { f } _ { \kappa } ; 0 ) - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { 1 } { \sqrt { d } } , \qquad | \mathcal { R } ( \widehat { f } _ { \kappa } ) - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { e ^ { - d ^ { c } } } { \kappa ^ { 2 } } + \frac { 1 } { \sqrt { d } } .
$$

The proof is given in Appendix D; it proceeds by linearizing the empirical risk via propositions 1 and 2 and applying deterministic-equivalent estimates to the associated resolvents. The bounds are non-vacuous provided $\kappa \geq d ^ { c }$ for some $c > - 1 / 6 ;$ below this scale, the empirical kernel resolvent is ill-conditioned and one cannot substitute $\widehat { \kappa }$ by $\widehat { \mathcal { K } } _ { \mathrm { e f f } }$

## A.2 Vanishing regularization: overfitting and memorization

For $\kappa = O _ { d } ( d ^ { - 1 / 6 } )$ and non-linear kernel function h, the direct comparison to the linearized kernel breaks down. Instead we use Proposition 3 which shows that, under Assumption 4, we can construct localized bumps that approximately interpolate the empirical Bayes denoiser, with training error $O _ { d } ( d ^ { - k _ { 0 } + 2 } )$ , while having RKHS norm bounded by $O _ { d } ( d )$ . The optimality of the ridge estimator directly implies $\mathcal { L } ( \widehat { f } _ { \kappa } ; \kappa ) \prec d ^ { - k _ { 0 } + 2 } + \kappa$ and the training error vanishes $\mathcal { L } ( \widehat { f } _ { \kappa } ; 0 ) = o _ { d } ( 1 )$ . We show that this forces $\hat { f } _ { \kappa } ( \pmb { u } ) \approx - \pmb { u } / \sigma$ on test inputs, and the population risk stabilizes at the purenoise plateau uniformly over the polynomial range of $\kappa ,$ similarly to Phase II of the gradient flow estimator (Theorem 2).

Theorem 7 (Ridge DSM with polynomially vanishing regularization). Suppose that Assumptions $\begin{array} { r } { \boldsymbol { { 1 } } , \boldsymbol { { 2 } } , } \end{array}$ and 4 hold. Suppose further that there exists $a n \epsilon \in ( 0 , k _ { 0 } { - } 2 )$ such that $d ^ { - k _ { 0 } + 2 } \lesssim \kappa \lesssim d ^ { - \epsilon }$ where $k _ { 0 }$ is given in Assumption 4. Then, $| \mathcal { L } ( \widehat { f } _ { \kappa } ; 0 ) | ~ \prec ~ \widehat { \kappa }$ . Furthermore, for every constant $c \in ( 0 , ( \epsilon \land 1 ) / 2 )$ and $\zeta > 0$ given in Assumption $^ { 4 , }$

$$
\left| \mathcal { R } ( \widehat { f } _ { \kappa } ) - \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \frac { \mathrm { T r } ( \Sigma ) } { d } \right| \lesssim ( C _ { \zeta } \vee 1 ) d ^ { c - \frac { \epsilon \wedge 1 } { 2 } } + \sqrt { 1 + ( C _ { \zeta } \vee 1 ) d ^ { c - \frac { \epsilon \wedge 1 } { 2 } } } d ^ { \frac { c } { 2 } - \frac { \epsilon \wedge 1 } { 4 } }
$$

with probability at least $1 - \zeta - e ^ { - d ^ { c ^ { \prime } } }$ for all $d \ge C$ , where $C _ { \zeta }$ is defined in (52).

The proof is given in Appendix D.

Finally, the ridgeless limit estimator $\kappa  0 ^ { + }$ with $n , d$ fixed coincides with infinite-time limit of the Gradient Flow estimator $\mathrm { \Omega t } \to \infty$ , and share the same limiting test error with doubled risk of Theorem 3 (after the mild post-processing step). This is studied in Appendix F.

## B Technical preliminaries

In this appendix, we collect some technical preliminaries that will be used in the proofs of the main results.

## B.1 Consequences of the Logarithmic Sobolev Inequality

We review some standard consequences of the Logarithmic Sobolev inequality (LSI), which is satisfied by the target data distribution $\pi _ { 0 }$ (Assumption 2.(b)). The standard Gaussian measure $\mathcal { N } ( 0 , \mathbf { I } _ { d } )$ satisfies an LSI with constant 1 [Gro75]. By the tensorization property of LSI [BGL14, Proposition 5.2.7], the product measure $\pi _ { d } = \pi _ { 0 } \otimes \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ satisfies an LSI with constant given by the maximum of the marginal constants, which is 1 ∨ $C _ { \mathrm { L S I } } = C _ { \mathrm { L S I } }$ Since the pushforward $( { \pmb x } , z )  \rho { \pmb x } + \sigma z$ , viewed as a function from $\mathbb { R } ^ { 2 d } \mathrm { ~ t o ~ } \mathbb { R } ^ { d }$ , has operator norm bounded by 1, the Lipschitz contraction property of LSI implies that $\mu _ { d }$ also satisfies an LSI with constant $C _ { \mathrm { L S I } }$ . The LSI inequality in Assumption 2.(b) implies a Poincar´e inequality

$$
\operatorname { V a r } _ { \pi _ { 0 } } ( f ) \leq C _ { \mathrm { L S I } } \mathbb { E } _ { \pi _ { 0 } } [ \| \nabla f \| _ { 2 } ^ { 2 } ] ,
$$

for every $f \in W ^ { 1 , 2 } ( \pi _ { 0 } )$ . Applying this to the linear function ${ \pmb x } \mapsto \langle { \pmb x } , { \pmb v } \rangle$

$$
\| \Sigma \| _ { \mathrm { o p } } \leq C _ { \mathrm { L S I } } .\tag{40}
$$

Suppose that $\pi$ on $\mathbb { R } ^ { d }$ satisfies LSI with constant $C > 0$ . By [BGL14, (5.4.2)],

$$
\mathbb { P } _ { \pmb { x } \sim \pi } ( | f ( \pmb { x } ) - \mathbb { E } _ { \pi } [ f ] | > t ) \le 2 \exp \left( - \frac { t ^ { 2 } } { 2 C \| f \| _ { \mathrm { L i p } } ^ { 2 } } \right)\tag{41}
$$

for any Lipschitz function $f : \mathbb { R } ^ { d }  \mathbb { R }$ and any $t > 0$ . By [Ada15, Theorem 2.3], for any $\pmb { A } \in \mathbb { R } ^ { d \times d }$ and $t \geq 0$

$$
\mathbb { P } _ { x \sim \pi _ { 0 } } \left( \vert x ^ { \mathsf { T } } A x - \mathrm { T r } ( A \Sigma ) \vert \ge t \right) \le 2 \exp \left( - \frac { 1 } { C } \operatorname* { m i n } \left\{ \frac { t ^ { 2 } } { 2 \| A \| _ { \mathrm { F } } ^ { 2 } } , \frac { t } { \| A \| _ { \mathrm { o p } } } \right\} \right)
$$

where $C > 0$ is some constant depending only on the LSI constant of $\pi _ { 0 }$ and thus

$$
\operatorname* { m a x } _ { i \in [ n ] } | \| x _ { 0 , i } \| ^ { 2 } - \operatorname { T r } ( \Sigma ) | \prec \sqrt { d } , \qquad \operatorname* { m a x } _ { i \neq j \in [ n ] } | \langle x _ { 0 , i } , x _ { 0 , j } \rangle | \prec \sqrt { d } .\tag{42}
$$

The analogous $L ^ { p }$ bounds also hold for any $p \geq 1$ 1:

$$
\begin{array} { r l } & { \mathbb { E } _ { { \pmb x } \sim \pi _ { 0 } } [ | \| { \pmb x } \| ^ { 2 } - \operatorname { T r } ( { \pmb \Sigma } ) | ^ { p } ] \leq C _ { p } d ^ { p / 2 } , \qquad \mathbb { E } _ { { \pmb x } ^ { \prime } \sim \pi _ { 0 } } [ | \langle { \pmb x } , { \pmb x } ^ { \prime } \rangle | ^ { p } ] ^ { 1 / p } \prec \sqrt { d } , } \\ & { \qquad \mathbb { E } _ { { \pmb x } , { \pmb x } ^ { \prime } \sim \pi _ { 0 } } [ | \langle { \pmb x } , { \pmb x } ^ { \prime } \rangle | ^ { p } ] ^ { 1 / p } \leq C _ { p } ^ { \prime } \sqrt { d } , } \end{array}\tag{43}
$$

for some constants $C _ { p } , C _ { p } ^ { \prime } > 0$ depending only on $p$ and $C _ { \mathrm { L S I } }$ . Since $\mu _ { d }$ satisfies the same LSI inequality as $\pi _ { 0 } ,$ similar bounds also hold for $\mu _ { d }$ . The mixed moments bounds also hold for the overlap between $\mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and $\pi _ { 0 }$ . We will use the following lemma to control the exponential moments of the overlaps between independent random vectors from $\pi _ { 0 }$ and $\mathcal { N } ( 0 , \mathbf { I } _ { d } )$

Lemma 1. Let $\pi$ be a distribution on $\mathbb { R } ^ { d }$ , and suppose that π satisfies LSI with constant $L$ and that $\| \mathbb { E } _ { { \pmb x } \sim \pi } { \pmb x } \| _ { 2 } ^ { 2 } \leq M d$ for some constant $M > 0$ . Then, for every $s > 0$

$$
\mathbb { E } _ { \pmb { x } \sim \pi } \exp \left( \frac { s } { 2 d } \| \pmb { x } \| _ { 2 } ^ { 2 } \right) \leq 3 e ^ { s ( L + M ) }
$$

whenever $d \geq 4 L s$

Proof. The LSI concentration inequality (41) applied to ${ \pmb x } \mapsto \| { \pmb x } \| _ { 2 }$ gives

$$
\mathbb { P } _ { { \pmb x } \sim \pi } ( | \| { \pmb x } \| _ { 2 } - \mathbb { E } \| { \pmb x } \| _ { 2 } | > t ) \le 2 \exp \left( - \frac { t ^ { 2 } } { 2 L } \right) ,
$$

while Poincar´e inequality gives $0 \preceq \mathsf { C o v } _ { \pi } ( \pmb { x } ) \preceq L \mathbf { I } _ { d }$ and thus $\mathbb { E } _ { \pi } [ \| \pmb { x } \| _ { 2 } ^ { 2 } ] \ \le \ ( L + M ) d .$ . Because $\| { \pmb x } \| _ { 2 } \le \mathbb { E } [ \| { \pmb x } \| _ { 2 } ] + R$ with $R = ( \| \pmb { x } \| _ { 2 } - \mathbb { E } [ \| \pmb { x } \| _ { 2 } ] ) _ { + }$ , we have $\| { \pmb x } \| _ { 2 } ^ { 2 } \leq 2 ( L + M ) d + 2 R ^ { 2 }$ by Young’s inequality. Integrating the Gaussian tail of R concludes the proof. □

Finally, a standard covering-net and Bernstein argument for independent sub-Gaussian vectors, in the form of the usual sample-covariance bound in [Wai19, Theorem 6.5], gives

$$
\left\| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \pmb x } _ { 0 , i } { \pmb x } _ { 0 , i } ^ { \mathsf T } - { \pmb \Sigma } \right\| _ { \mathrm { o p } } \lesssim \sqrt { \frac { d } { n } } + { \frac { d } { n } }\tag{44}
$$

with very high probability.

We apply the Poincar´e inequality to bound the second moment of the order-k tensor polynomial. For every integer $k \geq 0 .$ , denote by $( \mathbb { R } ^ { d } ) ^ { \otimes k }$ the set of tensors of order k on $\mathbb { R } ^ { d }$ , and by $\mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ the subset of (totally) symmetric tensors. That is, $\pmb { A } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ if $A \in ( \mathbb { R } ^ { d } ) ^ { \otimes k }$ and $A _ { i _ { 1 } , . . . , i _ { k } } =$ $A _ { i _ { s ( 1 ) } , \dots , i _ { s ( k ) } }$ for every permutation s of $\{ 1 , \ldots , k \}$

Lemma 2. Suppose that Assumption 2 holds. For every integer $k \geq 1$ and symmetric tensor $\pmb { A } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ R

$$
\begin{array} { r } { \mathbb { E } [ \langle \pmb { A } , \pmb { x } ^ { \otimes k } \rangle ^ { 2 } ] \leq ( 2 C _ { \mathrm { L S I } } ^ { 2 } k ^ { 2 } ) ^ { k } d ^ { \frac { k } { 2 } } \| \pmb { A } \| _ { \mathrm { F } } ^ { 2 } , } \end{array}
$$

where the expectation is taken over $\pmb { x } \sim \pi _ { 0 } ~ o r ~ \pmb { x } \sim \mu _ { d } .$

Proof. We prove the result for $\pi _ { 0 }$ . The same proof applies to $\mu _ { d }$ since both measures satisfy the same Poincar´e inequality. The proof proceeds by induction on k. For $k = 1 , A \in \mathbb { R } ^ { d }$ is a vector and

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { x } \sim \pi _ { 0 } } \left[ \langle \pmb { A } , \pmb { x } \rangle ^ { 2 } \right] = \pmb { A } ^ { \top } \pmb { \Sigma } \pmb { A } \leq \| \pmb { \Sigma } \| _ { \mathrm { o p } } \| \pmb { A } \| _ { 2 } ^ { 2 } . } \end{array}
$$

The bound holds with $C _ { 1 } = C _ { \mathrm { L S I } } \geq \Vert \pmb { \Sigma } \Vert _ { \mathrm { o p } }$

We now turn to the inductive step. Let $k \geq 2$ . We decompose the second moment of our order-k tensor polynomial into its variance and squared mean:

$$
\begin{array} { r } { \mathbb { E } _ { { \pmb x } \sim \pi _ { 0 } } [ \langle { \pmb A } , { \pmb x } ^ { \otimes k } \rangle ^ { 2 } ] = \mathrm { V a r } ( \langle { \pmb A } , { \pmb x } ^ { \otimes k } \rangle ) + \mathbb { E } _ { { \pmb x } \sim \pi _ { 0 } } [ \langle { \pmb A } , { \pmb x } ^ { \otimes k } \rangle ] ^ { 2 } . } \end{array}\tag{45}
$$

To bound the variance, we apply the Poincar´e inequality (see Assumption 2) to the function $f ( { \pmb x } ) =$ $\langle \boldsymbol { A } , \boldsymbol { x } ^ { \otimes k } \rangle$ to obtain

$$
\operatorname { V a r } _ { \substack { \pi \sim \pi _ { 0 } } } ( \langle A , x ^ { \otimes k } \rangle ) \leq C _ { \mathrm { L S I } } \mathbb { E } _ { { \pi \sim \pi _ { 0 } } } [ \| \nabla \langle A , x ^ { \otimes k } \rangle \| _ { 2 } ^ { 2 } ] = C _ { \mathrm { L S I } } k ^ { 2 } \sum _ { i = 1 } ^ { d } \mathbb { E } _ { { \pi \sim \pi _ { 0 } } } [ \langle B _ { i } , x ^ { \otimes ( k - 1 ) } \rangle ^ { 2 } ] ,
$$

where $B _ { i } \in \mathrm { S y m } ^ { k - 1 } ( \mathbb { R } ^ { d } )$ is the $( k - 1 )$ -order slice defined by $( \pmb { { \cal B } } _ { i } ) _ { j _ { 1 } , \dots , j _ { k - 1 } } : = { \cal A } _ { i , j _ { 1 } , \dots , j _ { k - 1 } }$ . Applying the inductive hypothesis to each slice $B _ { i }$

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { x } \sim \pi _ { 0 } } \left[ \langle \pmb { B } _ { i } , \pmb { x } ^ { \otimes ( k - 1 ) } \rangle ^ { 2 } \right] \leq C _ { k - 1 } d ^ { \frac { k - 1 } { 2 } } \| \pmb { B } _ { i } \| _ { \mathrm { F } } ^ { 2 } } \end{array}
$$

where $C _ { k - 1 } = ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( k - 1 ) ^ { 2 } ) ^ { k - 1 }$ . Summing over all coordinate slices using $\textstyle \sum _ { i = 1 } ^ { d } \| B _ { i } \| _ { \mathrm { F } } ^ { 2 } = \| A \| _ { \mathrm { F } } ^ { 2 }$ we obtain

$$
\operatorname { V a r } ( \langle A , x ^ { \otimes k } \rangle ) \leq C _ { \mathrm { L S I } } k ^ { 2 } C _ { k - 1 } d ^ { \frac { k - 1 } { 2 } } \| A \| _ { \mathrm { F } } ^ { 2 } .\tag{46}
$$

We now turn to the squared mean term in (45). We decompose the order-k tensor A into its coordinate slices as $\begin{array} { r } { \pmb { A } = \sum _ { i = 1 } ^ { d } e _ { i } \otimes \pmb { B } _ { i } } \end{array}$ , where $e _ { i }$ is the i-th standard basis vector in $\mathbb { R } ^ { d }$ . Since $\mathbb { E } _ { \pmb { x } \sim \pi _ { 0 } } [ \pmb { x } ] = 0$ by Assumption 2, we apply Cauchy-Schwarz inequality to obtain

$$
\mathbb { E } _ { { \mathbf { x } } \sim { \pi } _ { 0 } } [ \langle A , { \boldsymbol { \mathbf { x } } } ^ { \otimes k } \rangle ] ^ { 2 } = \left( \sum _ { i = 1 } ^ { d } \mathbb { E } _ { { \mathbf { x } } \sim { \pi } _ { 0 } } [ x _ { i } \langle B _ { i } , { \boldsymbol { \mathbf { x } } } ^ { \otimes ( k - 1 ) } \rangle ] \right) ^ { 2 } \leq \mathrm { T r } ( { \Sigma } ) \sum _ { i = 1 } ^ { d } \mathrm { V a r } ( \langle B _ { i } , { \boldsymbol { \mathbf { x } } } ^ { \otimes ( k - 1 ) } \rangle ) .
$$

On one hand, $\mathrm { T r } ( \pmb { \Sigma } ) \leq d \| \pmb { \Sigma } \| _ { \mathrm { o p } } \leq d C _ { \mathrm { L S I } }$ by (40). On the other hand, we use (46) to get

$$
\sum _ { i = 1 } ^ { d } \operatorname { V a r } ( \langle B _ { i } , \boldsymbol { x } ^ { \otimes ( k - 1 ) } \rangle ) \leq C _ { \mathrm { L S I } } ( k - 1 ) ^ { 2 } C _ { k - 2 } d ^ { \frac { k - 2 } { 2 } } \sum _ { i = 1 } ^ { d } \| B _ { i } \| _ { \mathrm { F } } ^ { 2 } = C _ { \mathrm { L S I } } ( k - 1 ) ^ { 2 } C _ { k - 2 } d ^ { \frac { k - 2 } { 2 } } \| A \| _ { \mathrm { F } } ^ { 2 } .
$$

For $k = 2$ , we have $C _ { 0 } = 1$ and the bound above is valid. Combining the bounds above,

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { x } \sim \pi _ { 0 } } [ \langle \pmb { A } , \pmb { x } ^ { \otimes k } \rangle ] ^ { 2 } \leq C _ { \mathrm { L S I } } ^ { 2 } ( k - 1 ) ^ { 2 } C _ { k - 2 } d ^ { \frac { k } { 2 } } \| \pmb { A } \| _ { \mathrm { F } } ^ { 2 } } \end{array}
$$

and

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { x } \sim \pi _ { 0 } } [ \langle \pmb { A } , \pmb { x } ^ { \otimes k } \rangle ^ { 2 } ] \leq 2 C _ { \mathrm { L S I } } ^ { 2 } k ^ { 2 } C _ { k - 1 } d ^ { \frac { k } { 2 } } \| \pmb { A } \| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

This completes the inductive step and the proof.

We now apply the moment bound in Lemma 2 and the Rosenthal bound in Lemma 4 to obtain a high-probability bound on the Frobenius norm of the empirical average of the order-k tensor polynomials $\mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ ( \rho \pmb { x } _ { 0 , i } + \sigma z ) ^ { \otimes k } ]$

Lemma 3. Suppose that Assumptions 1 and 2 hold. Then, for every $\zeta \in ( 0 , 1 )$ ,

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ ( \rho x _ { 0 , i } + \sigma z ) ^ { \otimes k } ] \right\| _ { \mathrm { F } } \lesssim C _ { \zeta , k } d ^ { \frac { k - 1 } { 2 } }
$$

for every $k \in \mathbb N$ with probability at least $1 - \zeta$ , where

$$
\begin{array} { r } { C _ { \zeta , k } : = ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( ( 4 k - \ln \zeta + 2 ) k ) ^ { 2 } ) ^ { k / 2 } ( 4 k - \ln \zeta ) . } \end{array}
$$

Proof. Let $\begin{array} { r } { \pmb { T } _ { i } = \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ ( \rho \pmb { x } _ { 0 , i } + \sigma z ) ^ { \otimes k } ] } \end{array}$ . By the triangle inequality,

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { T } _ { i } \right\| _ { \mathrm { F } } \leq \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \mathbf { T } _ { i } - \mathbb { E } _ { \boldsymbol { u } \sim \boldsymbol { \mu } _ { d } } [ \boldsymbol { u } ^ { \otimes k } ] \right) \right\| _ { \mathrm { F } } + \left\| \mathbb { E } _ { \boldsymbol { u } \sim \boldsymbol { \mu } _ { d } } [ \boldsymbol { u } ^ { \otimes k } ] \right\| _ { \mathrm { F } } .
$$

Using the variational representation of the Frobenius norm, the deterministic bias term is bounded by

$$
\left\| \mathbb { E } _ { \pmb { u } \sim \mu _ { d } } [ \pmb { u } ^ { \otimes k } ] \right\| _ { \mathrm { F } } = \operatorname* { s u p } _ { \| \pmb { A } \| _ { \mathrm { F } } \leq 1 } \mathbb { E } [ \langle \pmb { A } , \pmb { u } ^ { \otimes k } \rangle _ { \mathrm { F } } ] \leq ( 2 C _ { \mathrm { L S I } } ^ { 2 } k ^ { 2 } ) ^ { \frac { k } { 2 } } d ^ { \frac { k } { 4 } }
$$

by taking the square root of the bound established in Lemma 2. Next, we apply the Rosenthal bound in Lemma 4 to obtain

$$
\left\| \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( T _ { i } - \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \right) \right\| _ { \mathrm { F } } \right\| _ { L ^ { p } } \leq \sqrt { \frac { p - 1 } { n } } \| \| T _ { i } - \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \| _ { \mathrm { F } } \| _ { L ^ { 2 } }
$$

$$
+ \frac { \sqrt { ( p - 1 ) ( p - 2 ) } } { n } \left[ \frac { ( d ^ { k } + 1 ) n } { 2 } \right] ^ { \frac { 1 } { p } } \| \| T _ { i } - \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \| _ { \mathrm { F } } \| _ { L ^ { p } }
$$

for every even $p \geq 2$ . By Jensen’s inequality and the triangle inequality,

$$
\begin{array} { r } { \| \| T _ { i } - \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \| _ { \mathrm { F } } \| _ { L ^ { p } } \leq \| \| T _ { i } \| _ { \mathrm { F } } \| _ { L ^ { p } } + \| \| \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \| _ { \mathrm { F } } \| _ { L ^ { p } } \leq 2 \| \| T _ { i } \| _ { \mathrm { F } } \| _ { L ^ { p } } \leq 2 \| \| u \| _ { 2 } ^ { k } \| _ { L ^ { p } ( \mu _ { d } ) } . } \end{array}
$$

By the monotonicity of $L ^ { p }$ with respect to $p ,$

$$
\left\| \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( T _ { i } - \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \Big ) \right\| _ { \mathrm { F } } \right\| _ { L ^ { p } } \leq 2 \sqrt { \frac { p - 1 } { n } } \left( 1 + \sqrt { \frac { ( p - 2 ) } { n } } \left[ \frac { ( d ^ { k } + 1 ) n } { 2 } \right] ^ { \frac { 1 } { p } } \right) \| \| u \| _ { 2 } ^ { k } \| _ { L ^ { 2 p } ( \mu _ { d } ) } .
$$

As

$$
\| \| \boldsymbol { u } \| _ { 2 } ^ { k } \| _ { L ^ { 2 p } ( \mu _ { d } ) } = \mathbb { E } _ { \boldsymbol { u } \sim \mu _ { d } } [ \| \boldsymbol { u } \| _ { 2 } ^ { 2 p k } ] ^ { \frac { 1 } { 2 p } } = \mathbb { E } _ { \boldsymbol { u } \sim \mu _ { d } } [ \langle \boldsymbol { \mathrm { I } } _ { d } ^ { \otimes p k / 2 } , \boldsymbol { u } ^ { \otimes p k } \rangle ^ { 2 } ] ^ { \frac { 1 } { 2 p } } ,
$$

where $\mathbf { I } _ { d } ^ { \otimes p k / 2 }$ is the order-pk identity tensor defined such that $\langle \mathbf { I } _ { d } ^ { \otimes p k / 2 } , \pmb { A } \otimes \pmb { B } \rangle _ { \mathrm { F } } = \langle \pmb { A } , \pmb { B } \rangle _ { \mathrm { F } }$ for every $A , B \in ( \mathbb { R } ^ { d } ) ^ { \otimes p k / 2 }$ , we can apply Lemma 2 with $A = \mathsf { P } _ { \mathrm { S y m } ^ { p k } ( \mathbb { R } ^ { d } ) } ( \mathbf { I } _ { d } ^ { \otimes p k / 2 } )$ to obtain

$$
\| \| \boldsymbol { u } \| _ { 2 } ^ { k } \| _ { L ^ { 2 p } ( \mu _ { d } ) } \leq ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( p k ) ^ { 2 } ) ^ { \frac { k } { 2 } } d ^ { \frac { k } { 4 } } \| \boldsymbol { A } \| _ { \mathrm F } ^ { 1 / p } .
$$

Here, $\mathsf { P } _ { \mathrm { S y m } ^ { p k } ( \mathbb { R } ^ { d } ) }$ denotes the orthogonal projection $( \mathbb { R } ^ { d } ) ^ { \otimes p k } \ \mapsto \ \operatorname { S y m } ^ { p k } ( \mathbb { R } ^ { d } )$ with respect to the Frobenius inner product. Since $\| \mathbf { I } _ { d } ^ { \otimes p k / 2 } \| _ { \mathrm { F } } = d ^ { p k / 4 }$ ,

$$
\left. \left. \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( T _ { i } - \mathbb { E } _ { u \sim \mu _ { d } } [ u ^ { \otimes k } ] \right) \right. \right. _ { \mathbb { F } } \right. _ { L ^ { p } } \leq 2 ( 2 C _ { \mathrm { I S I } } ^ { 2 } ( p k ) ^ { 2 } ) ^ { \frac { k } { 2 } } \sqrt { \frac { p - 1 } { n } } \left( 1 + \sqrt { \frac { ( p - 2 ) } { n } } \left[ \frac { ( d ^ { k } + 1 ) n } { 2 } \right] ^ { \frac { 1 } { p } } \right) d ^ { \frac { k } { 2 } }
$$

and

$$
\left\| \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \pmb { T } _ { i } \right\| _ { \mathrm { F } } \right\| _ { L ^ { p } } \leq 3 ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( p k ) ^ { 2 } ) ^ { \frac { k } { 2 } } \sqrt { \frac { p - 1 } { n } } \left( 1 + \sqrt { \frac { ( p - 2 ) } { n } } \left[ \frac { ( d ^ { k } + 1 ) n } { 2 } \right] ^ { \frac { 1 } { p } } \right) d ^ { \frac { k } { 2 } }
$$

for every even $p \geq 2$

To obtain a tail bound, let $\zeta \in ( 0 , 1 )$ and, for every $k \in \mathbb N$ let $p _ { k }$ be the smallest even integer such that $p _ { k } \ge 4 k -$ ln ζ. By Markov’s inequality, it follows that

$$
\mathbb { P } \left( \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \pmb { T } _ { i } \right\| _ { \mathrm { F } } \geq t _ { k } \right) \leq \frac { \| \big \| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \pmb { T } _ { i } \big \| _ { \mathrm { F } } \big \| _ { L ^ { p _ { k } } } ^ { p _ { k } } } { t _ { k } ^ { p _ { k } } } \leq e ^ { - 4 k + \ln \zeta } \leq \zeta 2 ^ { - k }
$$

with

$$
t _ { k } = 3 e ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( p _ { k } k ) ^ { 2 } ) ^ { \frac { k } { 2 } } { \sqrt { \frac { p _ { k } - 1 } { n } } } \left( 1 + { \sqrt { \frac { ( p _ { k } - 2 ) } { n } } } \left[ { \frac { ( d ^ { k } + 1 ) n } { 2 } } \right] ^ { \frac { 1 } { p _ { k } } } \right) d ^ { \frac { k } { 2 } } .
$$

In particular, with probability at least $1 - \zeta$

$$
\left. \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \pmb { T } _ { i } \right. \right. _ { \mathrm { F } } \leq 3 e ( 2 C _ { \mathrm { L S I } } ^ { 2 } ( p _ { k } k ) ^ { 2 } ) ^ { \frac { k } { 2 } } \sqrt { \frac { p _ { k } - 1 } { n } } \left( 1 + \sqrt { \frac { ( p _ { k } - 2 ) } { n } } \left[ \frac { ( d ^ { k } + 1 ) n } { 2 } \right] ^ { \frac { 1 } { p _ { k } } } \right) d ^ { \frac { k } { 2 } }
$$

for every $k \in \mathbb N$ . Observe that $p _ { k } \geq 4 k$ and thus

$$
{ \sqrt { \frac { ( p _ { k } - 2 ) } { n } } } \left[ { \frac { ( d ^ { k } + 1 ) n } { 2 } } \right] ^ { \frac { 1 } { p _ { k } } } \lesssim { \sqrt { p _ { k } - 2 } } \left( { \frac { n } { d } } \right) ^ { \frac { 1 } { 4 k } }
$$

for every $k \in \mathbb N$ . This completes the proof.

We establish the Rosenthal-type inequality used above for sums of independent random tensors. The proof relies on the matrix Rosenthal inequality established by [Tro26, Fact B.3].

Lemma 4 (Rosenthal inequality for sums of random tensors). Suppose that $\{ \pmb { T } _ { i } \} _ { i = 1 } ^ { n }$ are independent random order-k tensors satisfying $\mathbb { E } [ \pmb { T } _ { i } ] = 0$ for all $i \in [ n ]$ , and let $\begin{array} { r } { \pmb { T } = \sum _ { i = 1 } ^ { n } \pmb { T } _ { i } } \end{array}$ . Then, for every $p = 1 / 2 , p = 1$ and $p \geq 3 / 2$

$$
\| \| T \| _ { \mathrm { F } } \| _ { L ^ { 4 p } } \leq \sqrt { 4 p - 1 } \left( \sum _ { i = 1 } ^ { n } \mathbb { E } \| T _ { i } \| _ { \mathrm { F } } ^ { 2 } \right) ^ { 1 / 2 } + \sqrt { ( 4 p - 1 ) ( 4 p - 2 ) } \left( \frac { d ^ { k } + 1 } { 2 } \right) ^ { \frac { 1 } { 4 p } } n ^ { \frac { 1 } { 4 p } } \operatorname* { m a x } _ { i \in [ n ] } \| \| T _ { i } \| _ { \mathrm { F } } \| _ { L ^ { 4 p } } .
$$

Proof. The proof relies on a simple application of the matrix Rosenthal inequality of [Tro26, Fact $\mathrm { B . 3 } ]$ to the symmetric dilation of the vectorization of the random tensors $\mathbf { \delta T } _ { i }$ . Define $M _ { i } \in$ $\mathbb { R } ^ { ( d ^ { \bar { k } } + 1 ) \times ( d ^ { k } + 1 ) }$ as the symmetric dilation of $\mathrm { V e c } ( T _ { i } ) \in \mathbb { R } ^ { d ^ { k } }$ . In particular, $M _ { i }$ is a block matrix of the form

$$
M _ { i } = \left[ \begin{array} { c c } { 0 } & { \mathrm { V e c } ( \mathbf { T } _ { i } ) ^ { \mathsf { T } } } \\ { \mathrm { V e c } ( \mathbf { T } _ { i } ) } & { 0 } \end{array} \right]
$$

where $\mathrm { V e c } ( \pmb { T } _ { i } )$ is the vectorization of the order-k tensor $\mathbf { \delta } \mathbf { T } _ { i }$ . Let $\begin{array} { r } { M = \sum _ { i = 1 } ^ { n } M _ { i } } \end{array}$ and notice that

$$
\mathbb { E } [ \mathrm { T r } ( | M | ^ { p } ) ] ^ { \frac { 1 } { p } } = ( 2 \mathbb { E } \| \mathrm { V e c } ( \pmb { T } ) \| _ { 2 } ^ { p } ) ^ { 1 / p } = 2 ^ { 1 / p } \| \| \pmb { T } \| _ { \mathrm { F } } \| _ { L ^ { p } }
$$

for every $p \in \mathbb N$ . Therefore, applying the matrix Rosenthal inequality of [Tro26, Fact B.3] to the sum of symmetric dilations M yields the following bound on the $L ^ { p }$ moment of the Frobenius norm of $\mathbf { T }$ :

$$
\begin{array} { r l r } {  { \| \| T \| _ { \mathrm { F } } \| _ { L ^ { q _ { p } } } = 2 ^ { - \frac { 1 } { 4 p } } \mathbb { E } [ \mathrm { T r } ( | M | ^ { 4 p } ) ] ^ { \frac { 1 } { 4 p } } } } \\ & { } & { \leq 2 ^ { - \frac { 1 } { 4 p } } \bigg ( \sqrt { 4 p - 1 } \mathrm { T r } \big ( ( \mathbb { E } [ M ^ { 2 } ] ) ^ { 2 p } \big ) ^ { \frac { 1 } { 4 p } } + \sqrt { ( 4 p - 1 ) ( 4 p - 2 ) } ( d ^ { k } + 1 ) ^ { \frac { 1 } { 4 p } } \| \underset { i \in [ n ] } { \operatorname* { m a x } } \| M _ { i } \| _ { \infty } \| _ { L ^ { 4 p } } \bigg ) } \end{array}
$$

for $p \ = \ 1 / 2 , \ p \ = \ 1$ and $p \ \geq \ 3 / 2$ . Since the summands $M _ { i }$ are independent and centered, $\begin{array} { r } { \mathbb { E } [ M ^ { 2 } ] = \sum _ { i = 1 } ^ { n } \mathbb { E } [ M _ { i } ^ { 2 } ] } \end{array}$ and $\begin{array} { r } { { \mathrm { T r } } \left( ( \mathbb { E } [ M ^ { 2 } ] ) ^ { 2 p } \right) ^ { \frac { 1 } { 4 p } } \leq 2 ^ { \frac { 1 } { 4 p } } \left( \sum _ { i = 1 } ^ { n } \mathbb { E } \| \pmb { T } _ { i } \| _ { \mathrm { F } } ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ . Furthermore, $\| M _ { i } \| _ { \mathrm { o p } } =$ $\lVert \mathrm { V e c } ( T _ { i } ) \rVert _ { 2 } = \lVert \mathbf { T } _ { i } \rVert _ { \mathrm { F } }$ for every $i \in [ n ]$ . Therefore, $\begin{array} { r } { \left\| \operatorname* { m a x } _ { i \in [ n ] } \left\| M _ { i } \right\| _ { \mathrm { o p } } \right\| _ { L ^ { 4 p } } \leq n ^ { \frac { 1 } { 4 p } } \operatorname* { m a x } _ { i \in [ n ] } \| \| \pmb { T } _ { i } \| _ { \mathrm { F } } \| _ { L ^ { 4 p } } } \end{array}$ Combining the bounds above concludes the proof. □

## B.2 Deterministic equivalents for the resolvent matrix

Define the empirical resolvent matrix as

$$
\widehat { R } _ { \kappa } : = \left( \frac { X X ^ { \mathsf { T } } } { n } + \kappa ^ { \star } \mathbf { I } _ { d } \right) ^ { - 1 } \in \mathbb { R } ^ { d \times d } ,\tag{47}
$$

where we recall that $\kappa ^ { \star } \equiv \kappa _ { \kappa } ^ { \star }$ is defined explicitly in (37). In the regime of Theorem $6 , \kappa ^ { \star }$ is bounded away from zero by assumption. However, we do not have an upper bound on $\kappa _ { \kappa } ^ { \star } .$ , and thus we will need to allow for the possibility that $\kappa ^ { \star }$ diverges with $d .$

We momentarily write

$$
\begin{array} { c c c } { { \widetilde { R } ( \vartheta ) : = \displaystyle \frac { 1 } { \kappa ^ { \star } } \left( \frac { \boldsymbol { X } \boldsymbol { X } ^ { \top } } { \kappa ^ { \star } n } + ( 1 + \vartheta ) \mathbf { I } _ { d } \right) ^ { - 1 } , } } & { { \widetilde { M } ( \vartheta ) = \displaystyle \frac { 1 } { \kappa ^ { \star } } \left( \frac { \boldsymbol { \Sigma } } { \kappa ^ { \star } \left( 1 + \widetilde { \mu } ( \vartheta ) \right) } + ( 1 + \vartheta ) \mathbf { I } _ { d } \right) ^ { - 1 } , } } \\ { { \widetilde { \mu } ( \vartheta ) = \displaystyle \frac { 1 } { n } \mathrm { T r } ( { \boldsymbol \Sigma } \widetilde { M } ( \vartheta ) ) } } & { { } } & { { } } \end{array}
$$

such that $\widehat { \pmb R } _ { \kappa } = \widetilde { \pmb R } ( 0 )$ and $M _ { \kappa } = \widetilde { M } ( 0 )$ . While we are ultimately interested in the case $\vartheta = 0$ , we will show the following more general result, which will be used to obtain a second-order deterministic equivalent for $\widehat { R } _ { \kappa } ^ { 2 }$ in Lemma 8.

Lemma 5. Suppose that Assumptions 1 and 2 hold, and that $\kappa ^ { \star } \geq c$ for some constant $c > 0$ Uniformly over $\vartheta \in [ 0 , 1 ]$ 2

$$
\| \mathbb { E } [ \widetilde { \pmb { R } } ( \vartheta ) ] - \widetilde { \pmb { M } } ( \vartheta ) \| _ { \mathrm { F } } \lesssim \frac { 1 } { \kappa ^ { \star } \sqrt { d } }
$$

Furthermore, for every deterministic $\pmb { A } \in \mathbb { R } ^ { d \times d }$ with $\| A \| _ { \mathrm { F } } \le 1$

$$
\left| \mathrm { T r } ( A ( \widetilde { \pmb { R } } ( \vartheta ) - \widetilde { \pmb { M } } ( \vartheta ) ) ) \right| \prec \frac { 1 } { \kappa ^ { \star } \sqrt { d } } .
$$

Proof. The first part follows directly from [Cho22, Theorem 2.3] and the fact that $\kappa ^ { \star } \geq C$ for some constant $C > 0$ . For the concentration, let $X _ { 1 } , X _ { 2 } \in \mathbb { R } ^ { d \times n }$ and denote $\widetilde { R } _ { i } = ( X _ { i } X _ { i } ^ { \mathsf { T } } / n + \kappa ^ { \star } ( 1 +$ $\vartheta ) \mathbf { I } _ { d } ) ^ { - 1 }$ . Using the resolvent identity $\begin{array} { r } { \widetilde { R } _ { 1 } - \widetilde { R } _ { 2 } = \frac { 1 } { n } \widetilde { R } _ { 1 } ( X _ { 2 } X _ { 2 } ^ { \top } - X _ { 1 } X _ { 1 } ^ { \top } ) \widetilde { R } _ { 2 } } \end{array}$ , we have

$$
\| \tilde { R } _ { 1 } - \tilde { R } _ { 2 } \| _ { \mathrm { F } } \leq \frac { 1 } { n } \| \tilde { R } _ { 1 } \| _ { \mathrm { o p } } \| X _ { 2 } ^ { \top } \widetilde { R } _ { 2 } \| _ { \mathrm { o p } } \| X _ { 1 } - X _ { 2 } \| _ { \mathrm { F } } + \frac { 1 } { n } \| \tilde { R } _ { 1 } X _ { 1 } \| _ { \mathrm { o p } } \| \tilde { R } _ { 2 } \| _ { \mathrm { o p } } \| X _ { 1 } - X _ { 2 } \| _ { \mathrm { F } } .
$$

Note that $\| \widetilde { R } _ { i } \| _ { \mathrm { o p } } \leq 1 / \kappa ^ { \star }$ and, by applying the scalar bound

$$
\frac { s } { s ^ { 2 } / n + \kappa ^ { \star } ( 1 + \vartheta ) } \leq \frac { \sqrt { n } } { 2 \sqrt { \kappa ^ { \star } ( 1 + \vartheta ) } }
$$

to the singular values of $X _ { i } ,$ , we have $\| \widetilde { \pmb { R } } _ { i } \pmb { X } _ { i } \| _ { \mathrm { o p } } \lesssim \sqrt { n }$ , using that $\kappa ^ { \star }$ is bounded below. Substituting this back yields

$$
\Vert \widetilde { R } _ { 1 } - \widetilde { R } _ { 2 } \Vert _ { \mathrm { F } } \lesssim \frac { 1 } { \kappa ^ { \star } \sqrt { n } } \Vert X _ { 1 } - X _ { 2 } \Vert _ { \mathrm { F } } .
$$

In particular, for any $\pmb { { A } } \in \mathbb { R } ^ { d \times d } ;$ , the function $\begin{array} { r } { \pmb { X } \mapsto \operatorname { T r } ( \pmb { A } \widetilde { \pmb { R } } ( \vartheta ) ) } \end{array}$ is Lipschitz with respect to the Frobenius norm with Lipschitz constant $\| A \| _ { \mathrm { F } } / ( \kappa ^ { \star } \sqrt { n } )$ . By the tensorization property of the logarithmic Sobolev inequality, $\pi _ { 0 } ^ { \otimes n }$ admits a logarithmic Sobolev inequality with the same constant $C _ { \mathrm { L S I } }$ as $\pi _ { 0 }$ . Therefore, by the Gaussian concentration of Lipschitz observables (41), for every $t > 0$

$$
\mathbb { P } \left( \left| \mathrm { T r } ( A \widetilde { \pmb { R } } ( \vartheta ) ) - \mathbb { E } [ \mathrm { T r } ( A \widetilde { \pmb { R } } ( \vartheta ) ) ] \right| \geq t \right) \leq 2 \exp \left( - \frac { t ^ { 2 } n ( \kappa _ { \kappa } ^ { \star } ) ^ { 2 } } { 2 C _ { \mathrm { L S I } } \| \pmb { A } \| _ { \mathrm F } ^ { 2 } } \right) .
$$

We conclude the proof using the bound on the bias established in the first part of the lemma, along with the definition of stochastic domination. □

Our analysis also requires a second-order deterministic equivalent for the squared resolvent, $\widehat { R } _ { \kappa } ^ { 2 }$ . Observe that these squared matrices emerge naturally as the negative derivatives of their firstorder counterparts evaluated at zero. Consequently, we can establish this second-order equivalent by applying a finite-diference approximation to the first-order bounds established in Lemma 5.

## Lemma 6. We have

$$
\partial _ { \boldsymbol { \theta } } \widetilde { \mathbf { R } } ( \boldsymbol { \vartheta } ) = - \kappa ^ { \star } \widetilde { \mathbf { R } } ( \boldsymbol { \vartheta } ) ^ { 2 } , \qquad \partial _ { \boldsymbol { \vartheta } } \widetilde { M } ( \boldsymbol { \vartheta } ) = - \kappa ^ { \star } \widetilde { M } ( \boldsymbol { \vartheta } ) \left( \frac { \frac { 1 } { n } \mathrm { T r } ( \boldsymbol { \Sigma } \widetilde { M } ( \boldsymbol { \vartheta } ) ^ { 2 } ) } { ( 1 + \widetilde { \mu } ( \boldsymbol { \vartheta } ) ) ^ { 2 } - \frac { 1 } { n } \mathrm { T r } ( \boldsymbol { \Sigma } ^ { 2 } \widetilde { M } ( \boldsymbol { \vartheta } ) ^ { 2 } ) } \boldsymbol { \Sigma } + \mathbf { I } _ { d } \right) \widetilde { M } ( \boldsymbol { \vartheta } )
$$

for every $\vartheta > 0$ . In particular, $\widehat { \pmb { R } } _ { \kappa } ^ { 2 } = - ( \kappa ^ { \star } ) ^ { - 1 } \partial _ { \vartheta } \widetilde { \pmb { R } } ( 0 )$ and $M _ { \kappa } ^ { ( 2 ) } = - ( \kappa ^ { \star } ) ^ { - 1 } \partial _ { \vartheta } \widetilde { M } ( 0 )$

Proof. The expression for $\partial _ { \vartheta } \widetilde { R } ( \vartheta )$ follows directly from the standard identity for the derivative of a matrix inverse. For $\partial _ { \vartheta } \widetilde { M } ( \vartheta )$ , we first compute $\partial _ { \vartheta } \widetilde { \mu } ( \vartheta )$ by diferentiating the fixed-point equation defining $\widetilde { \mu } ( \vartheta )$

$$
\begin{array} { l } { { \displaystyle \partial _ { \vartheta } \widetilde { \mu } ( \vartheta ) = - \frac { \kappa ^ { \star } } { n } \mathrm { T r } \left( \Sigma \widetilde { M } ( \vartheta ) \left( - \frac { \Sigma } { \kappa ^ { \star } ( 1 + \widetilde { \mu } ( \vartheta ) ) ^ { 2 } } \partial _ { \vartheta } \widetilde { \mu } ( \vartheta ) + { \bf I } _ { d } \right) \widetilde { M } ( \vartheta ) \right) } } \\ { { \displaystyle \qquad = \frac { 1 } { n } \mathrm { T r } ( \Sigma \widetilde { M } ( \vartheta ) \Sigma \widetilde { M } ( \vartheta ) ) \frac { \partial _ { \vartheta } \widetilde { \mu } ( \vartheta ) } { ( 1 + \widetilde { \mu } ( \vartheta ) ) ^ { 2 } } - \frac { \kappa ^ { \star } } { n } \mathrm { T r } ( \Sigma \widetilde { M } ( \vartheta ) ^ { 2 } ) . } } \end{array}
$$

By the definition of $\widetilde { M } ( \vartheta )$ , we have the matrix inequality $\begin{array} { r } { 0 \prec \frac { \Sigma } { 1 + \widetilde { \mu } ( \vartheta ) } \widetilde { M } ( \vartheta ) \prec { \mathbf { I } } _ { d } } \end{array}$ . Because Σ and $\widetilde { M } ( \vartheta )$ commute, it follows that

$$
0 \leq \frac { \mathrm { T r } ( \Sigma ^ { 2 } \widetilde { M } ( \vartheta ) ^ { 2 } ) } { n ( 1 + \widetilde { \mu } ( \vartheta ) ) ^ { 2 } } \leq \frac { 1 } { n ( 1 + \widetilde { \mu } ( \vartheta ) ) } \mathrm { T r } ( \Sigma \widetilde { M } ( \vartheta ) ) = \frac { \widetilde { \mu } ( \vartheta ) } { 1 + \widetilde { \mu } ( \vartheta ) } < 1 .
$$

Thus, the coeficient of $\partial _ { \vartheta } \widetilde { \mu } ( \vartheta )$ is strictly positive, and we can invert the expression to obtain

$$
\partial _ { \vartheta } \widetilde \mu ( \vartheta ) = - \kappa ^ { \star } \frac { \frac { 1 } { n } \mathrm { T r } ( \Sigma \widetilde { M } ( \vartheta ) ^ { 2 } ) } { 1 - \frac { 1 } { n } \mathrm { T r } ( \Sigma ^ { 2 } \widetilde { M } ( \vartheta ) ^ { 2 } ) / ( 1 + \widetilde \mu ( \vartheta ) ) ^ { 2 } } .
$$

Substituting this back into the chain rule expansion for $\partial _ { \vartheta } \widetilde { M } ( \vartheta )$ yields the desired expression.

In order to establish the second-order deterministic equivalent, we need as a preliminary step to bound the higher-order derivatives of $\mathbb { E } [ \widetilde { R } ( \vartheta ) ]$ and $\widetilde { M } ( \vartheta )$ with respect to ϑ.

Lemma 7. For every fixed $\vartheta \in [ 0 , C )$ and every $k \in \mathbb N$

$$
\| \partial _ { \vartheta } ^ { k } \mathbb { E } [ \widetilde { R } ( \vartheta ) ] \| _ { \mathrm { o p } } \le \frac { k ! } { \kappa _ { \kappa } ^ { \star } } , \qquad \| \partial _ { \vartheta } ^ { k } \widetilde { M } ( \vartheta ) \| _ { \mathrm { o p } } \le \frac { k ! 2 ^ { k + 2 } } { \kappa _ { \kappa } ^ { \star } } .
$$

Proof. We first establish a bound on the higher-order derivatives of $\mathbb { E } [ \widetilde { R } ( \vartheta ) ]$ . For every $k \in \mathbb N$ , we use the dominated convergence theorem to exchange the order of diferentiation and expectation, and then apply the standard formula for the derivative of a matrix inverse to obtain

$$
\partial _ { \vartheta } ^ { k } \mathbb { E } [ \widetilde { R } ( \vartheta ) ] = \frac { ( - 1 ) ^ { k } k ! } { \kappa ^ { \star } } \mathbb { E } [ ( \kappa ^ { \star } \widetilde { R } ( \vartheta ) ) ^ { k + 1 } ] .
$$

The operator norm of the resolvent is bounded by the scalar shift $\| \kappa ^ { \star } \widetilde { R } ( \vartheta ) \| _ { \mathrm { o p } } \leq 1$ . By Jensen’s inequality and the sub-multiplicativity of the operator norm, it immediately follows that

$$
\| \partial _ { \vartheta } ^ { k } \mathbb { E } [ \widetilde { R } ( \vartheta ) ] \| _ { \mathrm { o p } } \leq \frac { k ! } { \kappa ^ { \star } } \mathbb { E } [ \| \kappa ^ { \star } \widetilde { R } ( \vartheta ) \| _ { \mathrm { o p } } ^ { k + 1 } ] \leq \frac { k ! } { \kappa ^ { \star } } .
$$

For the higher-order derivatives of $\widetilde { M } ( \vartheta )$ , let $M ( z )$ denote the solution of the fixed-point equation (26). By (27),

$$
M ( z ) = \int _ { \mathbb { R } } \frac { 1 } { \lambda - z } \Omega ( \mathrm { d } \lambda ) , \qquad z \in \mathbb { C } \setminus \operatorname { s u p p } ( \Omega ) ,
$$

where Ω is a positive matrix-valued measure satisfying $\pmb { \Omega } ( \mathbb { R } ) = \mathbf { I } _ { d }$ and $\operatorname { s u p p } ( \Omega ) \subseteq [ 0 , \infty )$ . This representation extends M analytically across the negative real axis and satisfies the Schwarz symmetry $M ( \overline { { { z } } } ) = M ( z ) ^ { * }$

For every real $\vartheta \geq 0$ , the matrix $\widetilde { M } ( \vartheta )$ defined above is the restriction of this resolvent to the negative real axis:

$$
\widetilde { M } ( \vartheta ) = M \bigl ( - \kappa ^ { \star } ( 1 + \vartheta ) \bigr ) = \int _ { \mathbb { R } } \frac { 1 } { \lambda + \kappa ^ { \star } ( 1 + \vartheta ) } \Omega ( \mathrm { d } \lambda ) .
$$

Indeed, both sides solve the same fixed-point equation, whose admissible solution is unique. Differentiating under the integral therefore gives

$$
\partial _ { \vartheta } ^ { k } \widetilde { M } ( \vartheta ) = ( - 1 ) ^ { k } k ! ( \kappa ^ { \star } ) ^ { k } \int _ { \mathbb { R } } \frac { \Omega ( \mathrm { d } \lambda ) } { ( \lambda + \kappa ^ { \star } ( 1 + \vartheta ) ) ^ { k + 1 } } .
$$

Since $\lambda \geq 0$ on the support of Ω and $\pmb { \Omega } ( \mathbb { R } ) = \mathbf { I } _ { d } .$ , it follows that

$$
\lVert \partial _ { \vartheta } ^ { k } \widetilde { M } ( \vartheta ) \rVert _ { \mathrm { o p } } \leq \frac { k ! } { \kappa ^ { \star } ( 1 + \vartheta ) ^ { k + 1 } } \leq \frac { k ! } { \kappa ^ { \star } } ,
$$

which proves the desired bound.

Lemma 8. Suppose that Assumptions 1 and 2 hold, and that $\kappa ^ { \star } \geq c$ for some constant $c > 0$ Then,

$$
\| \mathbb { E } [ \widehat { R } _ { \kappa } ^ { 2 } ] - M _ { \kappa } ^ { ( 2 ) } \| _ { \mathrm { F } } \prec \frac { 1 } { ( \kappa _ { \kappa } ^ { \star } ) ^ { 2 } \sqrt { d } }
$$

Furthermore, for any deterministic $\pmb { A } \in \mathbb { R } ^ { d \times d }$ with $\| A \| _ { \mathrm { F } } \le 1$ 2

$$
\left| \mathrm { T r } ( A ( \widehat { \pmb { R } } _ { \kappa } ^ { 2 } - { \cal M } _ { \kappa } ^ { ( 2 ) } ) ) \right| \prec \frac { 1 } { ( \kappa _ { \kappa } ^ { \star } ) ^ { 2 } \sqrt { d } } .
$$

Proof. Let $\pmb { A } \in \mathbb { R } ^ { d \times d }$ with $\| A \| _ { \mathrm { F } } \le 1$ and define $f _ { A } ( \vartheta ) = \mathrm { T r } ( A ( \mathbb { E } [ \widetilde { \pmb { R } } ( \vartheta ) ] - \widetilde { \pmb { M } } ( \vartheta ) ) )$ . Let $k \in \mathbb N$ Using the standard $( k + 1 )$ -point forward diference formula for the derivative at $\vartheta = 0$ with nodes $\vartheta _ { j } = j \Delta / k$ for $j = 0 , \ldots , k$ , we obtain

$$
| f _ { A } ^ { \prime } ( 0 ) | \leq \frac { k 2 ^ { k + 1 } } { \Delta } \operatorname* { m a x } _ { j \in \{ 0 , \ldots , k \} } | f _ { A } ( \vartheta _ { j } ) | + \frac { \Delta ^ { k } } { ( k + 1 ) k ^ { k } } \operatorname* { m a x } _ { \vartheta \in [ 0 , \Delta ] } | f _ { A } ^ { ( k + 1 ) } ( \vartheta ) | ,\tag{48}
$$

where we bounded the sum of the absolute values of the Lagrange basis derivatives as $\begin{array} { r } { \sum _ { j = 0 } ^ { k } | L _ { j } ^ { \prime } ( 0 ) | \leq } \end{array}$ $k 2 ^ { k + 1 } / \Delta ~ ( \mathrm { e . g . }$ , see [CM24, Lemma 6.2]).

By Lemma 5, the function values are bounded by

$$
\frac { k 2 ^ { k + 1 } } { \Delta } \operatorname* { m a x } _ { j \in \{ 0 , \ldots , k \} } | f _ { A } ( \vartheta _ { j } ) | \lesssim \frac { k 2 ^ { k + 1 } } { \Delta \kappa ^ { \star } \sqrt { d } } .
$$

For the remainder term, we bridge the trace to the operator norm using the inequality $| \mathrm { T r } ( A B ) | \leq$ $\| A \| _ { \mathrm { F } } \| B \| _ { \mathrm { F } } \leq \sqrt { d } \| A \| _ { \mathrm { F } } \| B \| _ { \mathrm { o p } }$ . Applying Lemma 7 to the $( k + 1 )$ -th derivative yields

$$
| f _ { A } ^ { ( k + 1 ) } ( \vartheta ) | \leq \sqrt { d } \left( \| \partial _ { \vartheta } ^ { k + 1 } \mathbb { E } [ \widetilde { R } ( \vartheta ) ] \| _ { \mathrm { { o p } } } + \| \partial _ { \vartheta } ^ { k + 1 } \widetilde { M } ( \vartheta ) \| _ { \mathrm { { o p } } } \right) \leq \frac { \sqrt { d } ( k + 1 ) ! 2 ^ { k + 4 } } { \kappa ^ { \star } } .
$$

Plugging this into the second term of (48), the (k + 1) factor in the denominator cancels, giving

$$
\frac { \Delta ^ { k } } { ( k + 1 ) k ^ { k } } \operatorname* { m a x } _ { \vartheta \in [ 0 , \Delta ] } | f _ { A } ^ { ( k + 1 ) } ( \vartheta ) | \leq \frac { \Delta ^ { k } k ! 2 ^ { k + 4 } \sqrt { d } } { k ^ { k } \kappa ^ { \star } } .
$$

Bringing both bounds together, we have shown that

$$
\vert f _ { A } ^ { \prime } ( 0 ) \vert \lesssim \frac { C _ { k } } { \kappa ^ { \star } } \left( \frac { 1 } { \Delta \sqrt { d } } + \Delta ^ { k } \sqrt { d } \right)
$$

for some constant $C _ { k } > 0$ depending on $k .$ . This proves the first part of the lemma by choosing $\Delta = d ^ { - \epsilon }$ for some suficiently small $\epsilon > 0$ and $k = \lceil \epsilon ^ { - 1 } \rceil$ , and then using Lemma $6$ to divide $f _ { A } ^ { \prime } ( 0 )$ $\mathrm { b y } - \kappa ^ { \star }$ and identify $\mathrm { T r } ( A ( \mathbb { E } [ \widehat { \pmb { R } } _ { \kappa } ^ { 2 } ] - M _ { \kappa } ^ { ( 2 ) } ) )$

We apply a similar argument as in the first-order case to establish the concentration of $\mathrm { T r } ( A \widehat { R } _ { \kappa } ^ { 2 } )$ around its expectation. Let $X _ { 1 } , X _ { 2 } \in \mathbb { R } ^ { d \times n }$ and denote $\widehat { \pmb { R } } _ { i } = ( X _ { i } \pmb { X } _ { i } ^ { \top } / n + \kappa ^ { \star } \mathbf { I } _ { d } ) ^ { - 1 }$ . Then,

$$
\| \widehat { R } _ { 1 } ^ { 2 } - \widehat { R } _ { 2 } ^ { 2 } \| _ { \mathrm { F } } \leq ( \| \widehat { R } _ { 1 } \| _ { \mathrm { o p } } + \| \widehat { R } _ { 2 } \| _ { \mathrm { o p } } ) \| \widehat { R } _ { 1 } - \widehat { R } _ { 2 } \| _ { \mathrm { F } } \leq \frac { 2 } { \kappa _ { \kappa } ^ { \star } } \| \widehat { R } _ { 1 } - \widehat { R } _ { 2 } \| _ { \mathrm { F } } .
$$

We established in the proof of Lemma 5 that $\begin{array} { r } { \| \widehat { R } _ { 1 } - \widehat { R } _ { 2 } \| _ { \mathrm { F } } \leq \frac { 1 } { \sqrt { n } \kappa _ { \kappa } ^ { \star } } \| X _ { 1 } - X _ { 2 } \| _ { \mathrm { F } } } \end{array}$ . Therefore, the function $X \mapsto \operatorname { T r } ( A \widehat { R } _ { \kappa } ^ { 2 } )$ is Lipschitz with respect to the Frobenius norm with Lipschitz constant $2 \| A \| _ { \mathrm { F } } / ( \sqrt { n } ( \kappa _ { \kappa } ^ { \star } ) ^ { 2 } )$ . We conclude by following the exact same concentration argument as in the first-order case. □

Finally, we need a quantitative local law that characterizes the distance between the empirical resolvent and its deterministic equivalent outside the spectral support. Recall the empirical resolvent

$$
\widehat { R } ( z ) : = \left( \frac { X X ^ { \mathsf { T } } } { n } - z \mathbf { I } _ { d } \right) ^ { - 1 } \in \mathbb { C } ^ { d \times d } ,
$$

which is well-defined for all $z \in \mathbb { C }$ outside the spectrum of $X X ^ { \mathsf { T } } / n$ . The corresponding deterministic equivalent is given by $\begin{array} { r } { M ( z ) = ( \frac { \Sigma } { 1 + \mu ( z ) } - { \overset { \textstyle > } { z } } \mathbf { I } _ { d } ) ^ { - 1 } } \end{array}$ , where $\mu ( z )$ is the unique solution to the fixed-point equation $\begin{array} { r } { \mu ( z ) = \frac { 1 } { n } \mathrm { T r } ( \Sigma M ( z ) ) } \end{array}$ as defined in (26).

The following proposition, which follows directly from the local law for sample covariance matrices established in [FMPW26, Theorem 2.10], provides a uniform bound on the distance between the empirical resolvent and its deterministic equivalent outside the spectral support. Note that by Assumption 2.(b), $c \leq \tau _ { \Sigma } \leq C _ { \mathrm { L S I } }$ , and therefore there exists $c > 0$ such that at most $( 1 - c ) d$ eigenvalues of Σ belong to the interval [0, c].

Proposition 5 (Outside-support anisotropic local law [FMPW26, Theorem 2.10]). Suppose that Assumptions 1 and $2 \ h o l d$ For every fixed $\delta > 0$ , every deterministic $0 \preceq A \in \mathbb { R } ^ { d \times \bar { d } }$ with $\| A \| _ { * } \leq 1$ and every $\epsilon , D > 0 $ , there exists $C > 0$ such that, with probability at least $1 - C d ^ { - D }$ , uniformly over $a l l z \in \mathbb { C }$ at distance at least δ from the support of the limiting spectral distribution of $X X ^ { \mathsf { T } } / n$ and satisfying $| z | \geq \delta , | \Re z | \leq \delta ^ { - 1 }$ , and $| \mathrm { I m } z | \leq 1$ , we have

$$
\left| \operatorname { T r } \left( A ( \widehat { R } ( z ) - M ( z ) ) \right) \right| \leq d ^ { \epsilon - \frac { 1 } { 2 } } .
$$

## B.3 Power series decomposition of the RKHS

Under the regularity conditions of Assumption 4, the base function admits a power series expansion $\begin{array} { r } { h ( x ) = \sum _ { k = 0 } ^ { \infty } \alpha _ { k } x ^ { k } } \end{array}$ with non-negative coeficients. This analytic structure naturally lifts to the inner-product kernel $K ( \pmb { x } , \pmb { x } ^ { \prime } )$ , inducing a direct sum decomposition of the associated RKHS into orthogonal subspaces of active homogeneous-polynomial degrees

$$
{ \mathcal { A } } _ { h } : = \{ k \geq 3 : \alpha _ { k } > 0 \} .
$$

Lemma 9 (RKHS Functional Decomposition). Suppose that Assumption 4 holds. Then, every function $f \in \mathcal H$ can be uniquely represented as

$$
f ( \pmb { x } ) = \langle \pmb { \theta } , \pmb { x } \rangle + \sum _ { k \in \mathcal { A } _ { h } } \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \langle \pmb { W } _ { k } , \pmb { x } ^ { \otimes k } \rangle ,
$$

where $\pmb \theta \in \mathbb { R } ^ { d }$ and $W _ { k } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ is a symmetric tensor for all $k \geq 3$ . Furthermore, the associated RKHS norm of f is given by

$$
\| f \| _ { \mathcal { H } } ^ { 2 } = d \| \pmb { \theta } \| _ { 2 } ^ { 2 } + \sum _ { k \in \mathcal { A } _ { h } } \| \pmb { W } _ { k } \| _ { \mathrm { F } } ^ { 2 } .
$$

Proof. By Assumption 4, the base function h admits a power series expansion $\begin{array} { r } { h ( x ) = \sum _ { k = 0 } ^ { \infty } \alpha _ { k } x ^ { k } } \end{array}$ with $\alpha _ { 0 } = \alpha _ { 2 } = 0$ and $\alpha _ { 1 } = 1$ . Evaluating the kernel $K ( { \pmb x } , { \pmb y } ) = h ( \langle { \pmb x } , { \pmb y } \rangle / d )$ and using the standard tensor product identity $\langle \pmb { x } , \pmb { y } \rangle ^ { k } = \langle \pmb { x } ^ { \otimes k } , \pmb { y } ^ { \otimes k } \rangle _ { \mathrm { F } }$ , we can expand it as

$$
K ( \pmb { x } , \pmb { y } ) = \frac { 1 } { d } \langle \pmb { x } , \pmb { y } \rangle + \sum _ { k \in \mathcal { A } _ { h } } \left. \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \pmb { x } ^ { \otimes k } , \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \pmb { y } ^ { \otimes k } \right. _ { \mathrm { F } } .
$$

This defines a natural feature map $\phi : \mathbb { R } ^ { d }  \mathcal { F } .$ , where the feature space is the direct sum of active symmetric tensor spaces $\mathcal { F } = \mathbb { R } ^ { d } \oplus \bigoplus _ { k \in \mathcal { A } _ { h } } \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ , explicitly given by

$$
\phi ( \pmb { x } ) = \left( \frac { 1 } { \sqrt { d } } \pmb { x } , \left( \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \pmb { x } ^ { \otimes k } \right) _ { k \in \mathcal { A } _ { h } } \right) .
$$

By construction, the kernel is exactly the inner product in this feature space, $K ( \pmb { x } , \pmb { y } ) = \langle \phi ( \pmb { x } ) , \phi ( \pmb { y } ) \rangle _ { \mathcal { F } }$ By the standard isometry between the RKHS and the feature space image (see, for instance, [SC08, Theorem 4.21]), for any function $f \in \mathcal H$ we can write

$$
f ( \pmb { x } ) = \langle \phi ( \pmb { x } ) , \pmb { w } \rangle _ { \mathcal { F } } = \frac { 1 } { \sqrt { d } } \langle \pmb { x } , \pmb { w } _ { 1 } \rangle + \sum _ { k \in \mathcal { A } _ { h } } \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \langle \pmb { x } ^ { \odot k } , \pmb { W } _ { k } \rangle _ { \mathrm { F } }
$$

for some ${ \pmb w } = ( { \pmb w } _ { 1 } , { \pmb W } _ { 3 } , { \pmb W } _ { 4 } , \ldots ) \in \mathcal { F }$ , where $\pmb { w } _ { 1 } \in \mathbb { R } ^ { d }$ and $W _ { k } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ . We match the linear term to the form in the lemma by defining $\pmb { \theta } = \pmb { w } _ { 1 } / \sqrt { d }$ , which gives the desired functional form. Furthermore, we can write the RKHS norm of $f \in \mathcal H$ as

$$
\| f \| _ { \mathcal H } ^ { 2 } : = \operatorname* { i n f } \{ \| w \| _ { \mathcal F } ^ { 2 } : w \in \mathcal F , f = \langle \phi ( \cdot ) , w \rangle _ { \mathcal F } \} = d \| \theta \| _ { 2 } ^ { 2 } + \sum _ { k \in \mathcal A _ { h } } \| W _ { k } \| _ { \mathrm F } ^ { 2 } .
$$

Here, we have used the fact that the RKHS norm of $f$ is defined as the minimum F-norm of any weight sequence w that represents $f$ through the feature map. This infimum is uniquely attained by the weight sequence $\pmb { w }$ that lies in the closure of the linear span of the feature map $\phi ( \mathbb { R } ^ { d } )$ . □

Next, we bound the contribution of the higher-order terms in the power series decomposition of the RKHS. We will define

$$
{ \mathcal { H } } _ { \geq 3 } : = \{ f \in { \mathcal { H } } : f ( x ) = \sum _ { k \in { \mathcal { A } } _ { h } } \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \langle W _ { k } , x ^ { \otimes k } \rangle \}\tag{49}
$$

as the subspace of the RKHS consisting of functions with no linear component. Before stating the bound, we briefly detour to introduce the necessary machinery regarding multivariate Hermite

tensors and the Wiener chaos expansion, which will provide the required isometries. Let $\gamma _ { d } ( { \pmb x } ) =$ $( 2 \pi ) ^ { - \frac { d } { 2 } } e ^ { - \| \pmb { x } \| _ { 2 } ^ { 2 } / 2 }$ denote the density of a standard Gaussian vector in $\mathbb { R } ^ { d }$ . For any $d \geq 1$ and $k \geq 0$ define $\mathrm { H e } _ { k } : \mathbb { R } ^ { d }  \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ to be the normalized degree-k Hermite tensor in $\mathbb { R } ^ { d }$ given by

$$
\mathrm { H e } _ { k } ( \boldsymbol { x } ) : = \frac { ( - 1 ) ^ { k } } { \sqrt { k ! } } \frac { \nabla ^ { k } \gamma _ { d } ( \boldsymbol { x } ) } { \gamma _ { d } ( \boldsymbol { x } ) } ,\tag{50}
$$

where $\nabla ^ { k } \gamma _ { d } ( \pmb { x } )$ denotes the k-th derivative of $\gamma _ { d }$ , viewed as a symmetric k-tensor. In particular, $\mathrm { H e } _ { 0 } ( { \pmb x } ) = 1$ and ${ \mathrm { H e } } _ { 1 } ( { \pmb x } ) = { \pmb x }$

The Hermite tensors realize the classical isometry between $\mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ and the k-th Wiener chaos: for all $\pmb { A } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ and $B \in \mathrm { S y m } ^ { j } ( \mathbb { R } ^ { d } )$ ),

$$
\mathbb { E } _ { \gamma _ { d } } [ \langle \pmb { A } , \mathrm { H e } _ { k } ( \pmb { x } ) \rangle _ { \mathrm { F } } \langle \pmb { B } , \mathrm { H e } _ { j } ( \pmb { x } ) \rangle _ { \mathrm { F } } ] = \mathbb { 1 } _ { j = k } \langle \pmb { A } , \pmb { B } \rangle _ { \mathrm { F } } .
$$

Hence any $f \in L ^ { 2 } ( \gamma _ { d } )$ admits the Wiener chaos expansion

$$
f ( \pmb { x } ) = \sum _ { k = 0 } ^ { \infty } \langle \pmb { A } _ { k } , \mathrm { H e } _ { k } ( \pmb { x } ) \rangle _ { \mathrm { F } } , \qquad \pmb { A } _ { k } : = \mathbb { E } _ { \gamma _ { d } } [ f ( \pmb { x } ) \mathrm { H e } _ { k } ( \pmb { x } ) ] \in \mathrm { S y m } _ { k } ( \mathbb { R } ^ { d } ) ,
$$

with convergence in $L ^ { 2 } ( \gamma _ { d } )$ . Of particular interest to us will be the Hermite coeficients

$$
\Gamma _ { k } f ( \pmb { x } ) : = \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ f ( \rho \pmb { x } + \sigma z ) \mathrm { H e } _ { k } ( z ) ] .\tag{51}
$$

In light of the general Wiener chaos expansion, the tensor $\Gamma _ { k } f ( { \pmb x } )$ is exactly the k-th Hermite coeficient of the shifted and scaled function $z \mapsto f ( \rho \mathbf { x } + \sigma z )$ for a fixed x. Consequently, we can decompose this function over the Gaussian noise z as

$$
f ( \rho x + \sigma z ) = \sum _ { k = 0 } ^ { \infty } \langle \Gamma _ { k } f ( \pmb { x } ) , \mathrm { H e } _ { k } ( z ) \rangle _ { \mathrm { F } } .
$$

Lemma 10. Suppose that Assumptions 1, 2, and $\it 4$ hold. Then,

$$
\| g \| _ { L ^ { 2 } ( \mu _ { d } ) } \lesssim d ^ { - 3 / 4 } \| g \| _ { \mathcal { H } } , \qquad \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Gamma _ { 1 } g ( \pmb { x } _ { 0 , i } ) \right\| _ { 2 } \lesssim \frac { C _ { \zeta } } { d } \| g \| _ { \mathcal { H } } , \qquad C _ { \zeta } ^ { 2 } : = \sum _ { k = 3 } ^ { \infty } k ^ { 2 } \alpha _ { k } C _ { \zeta , k } ^ { 2 } ,\tag{52}
$$

for every $g \in \mathcal { H } _ { \geq 3 }$ with power series representation (49), with probability at least $1 - \zeta$ , where $\zeta$ and $C _ { \zeta , k }$ are given in Assumption $\it 4 .$

Proof. We first establish the result for finite truncations of the series. Let $g _ { L }$ be the polynomial of maximum degree $L \geq 3$ , defined as

$$
g _ { L } ( \pmb { x } ) = \sum _ { k = 3 } ^ { L } \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \langle \pmb { W } _ { k } , \pmb { x } ^ { \otimes k } \rangle .
$$

By Lemma 9,

$$
\| g _ { L } \| _ { \mathcal { H } } ^ { 2 } = \sum _ { k = 3 } ^ { L } \| \mathbf { W } _ { k } \| _ { \mathrm { F } } ^ { 2 } .
$$

Let $m ,$ k be arbitrary positive integers. By Cauchy-Schwarz inequality and Lemma $2 ,$

$$
\begin{array} { r } { { \mathbb E } _ { { \boldsymbol x } \sim \mu _ { d } } [ \langle { \boldsymbol W } _ { m } , { \boldsymbol x } ^ { \otimes m } \rangle \langle { \boldsymbol W } _ { k } , { \boldsymbol x } ^ { \otimes k } \rangle ] \leq \sqrt { { \mathbb E } _ { { \boldsymbol x } \sim \mu _ { d } } [ \langle { \boldsymbol W } _ { m } , { \boldsymbol x } ^ { \otimes m } \rangle ^ { 2 } ] } { \mathbb E } _ { { \boldsymbol x } \sim \mu _ { d } } [ \langle { \boldsymbol W } _ { k } , { \boldsymbol x } ^ { \otimes k } \rangle ^ { 2 } ] } \end{array}
$$

$$
\leq B _ { m } B _ { k } d ^ { \frac { m + k } { 4 } } \| \mathbf { W } _ { m } \| _ { \mathrm { F } } \| \mathbf { W } _ { k } \| _ { \mathrm { F } }
$$

where we defined $B _ { k } ^ { 2 } : = ( 2 C _ { \mathrm { L S I } } ^ { 2 } k ^ { 2 } ) ^ { k }$ . Thus,

$$
\begin{array} { r l } & { \displaystyle \| g _ { L } \| _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } \leq \sum _ { m = 3 } ^ { L } \sum _ { k = 3 } ^ { L } \sqrt { \alpha _ { m } \alpha _ { k } } B _ { m } B _ { k } d ^ { - \frac { m + k } { 4 } } \| \pmb { W } _ { m } \| _ { \mathrm { F } } \| \pmb { W } _ { k } \| _ { \mathrm { F } } } \\ & { \qquad \leq d ^ { - \frac { 3 } { 2 } } \left( \displaystyle \sum _ { k = 3 } ^ { L } \sqrt { \alpha _ { k } } B _ { k } \| \pmb { W } _ { k } \| _ { \mathrm { F } } \right) ^ { 2 } \leq d ^ { - \frac { 3 } { 2 } } \| g _ { L } \| _ { \mathcal { H } } ^ { 2 } \displaystyle \sum _ { k = 3 } ^ { L } \alpha _ { k } B _ { k } ^ { 2 } , } \end{array}
$$

where the last inequality follows from Cauchy-Schwarz. Since $B _ { k } \le C _ { \zeta , k }$ for any $\zeta \in ( 0 , 1 )$ , it follows from Assumption 4 that $\begin{array} { r } { \sum _ { k = 3 } ^ { L } \alpha _ { k } B _ { k } ^ { 2 } \le \sum _ { k = 3 } ^ { \infty } \alpha _ { k } B _ { k } ^ { 2 } < \infty } \end{array}$ and hence $\lVert g _ { L } \rVert _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } \lesssim d ^ { - 3 / 2 } \lVert g _ { L } \rVert _ { \mathcal { H } } ^ { 2 }$ Since $\| g _ { L } - g \| _ { \mathcal { H } } \to 0$ , the reproducing property gives $g _ { L } ( { \pmb x } )  g ( { \pmb x } )$ pointwise. Fatou’s lemma and $\| g _ { L } \| _ { \mathcal { H } } \to \| g \| _ { \mathcal { H } }$ therefore yield

$$
\| g \| _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } \leq \operatorname* { l i m i n f } _ { L \to \infty } \| g _ { L } \| _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } \lesssim d ^ { - 3 / 2 } \| g \| _ { \mathcal { H } } ^ { 2 } ,
$$

proving the first inequality.

We now turn to the second inequality. Let $k \in [ L ]$ and define $f _ { k } ( \pmb { x } ) = \langle \pmb { W } _ { k } , \pmb { x } ^ { \otimes k } \rangle _ { \mathrm { F } }$ . Then, by Gaussian integration by parts,

$$
\Gamma _ { 1 } f _ { k } ( x ) = \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \nabla _ { z } \langle W _ { k } , ( \rho x + \sigma z ) ^ { \otimes k } \rangle _ { \mathrm { F } } ] = \sigma k W _ { k } \otimes _ { k - 1 } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ ( \rho x + \sigma z ) ^ { \otimes ( k - 1 ) } ]
$$

where, for any $A \in \mathrm { S y m } _ { k } ( \mathbb { R } ^ { d } ) , B \in \mathrm { S y m } _ { k - 1 } ( \mathbb { R } ^ { d } ) , A \otimes _ { k - 1 } B \in \mathbb { R } ^ { d }$ denotes the partial contraction along any $k - 1$ indices. By triangle inequality, sub-multiplicativity of the Frobenius norm, Lemma 3 and Cauchy-Schwarz inequality,

$$
\begin{array} { r l } { \displaystyle \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Gamma _ { 1 } g _ { L } ( \pmb { x } _ { 0 , i } ) \right\| _ { 2 } \leq \sum _ { k = 3 } ^ { L } \sigma k \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \| \pmb { W } _ { k } \| _ { \mathrm { F } } \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ ( \rho \pmb { x } _ { 0 , i } + \sigma z ) ^ { \otimes ( k - 1 ) } ] \right\| _ { \mathrm { F } } } & { } \\ { \displaystyle \lesssim \sigma \sum _ { k = 3 } ^ { L } k \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \| \pmb { W } _ { k } \| _ { \mathrm { F } } C _ { \zeta , k - 1 } d ^ { \frac { k - 2 } { 2 } } \leq \frac { \sigma } { d } \| g _ { L } \| _ { \mathcal { H } } \left( \sum _ { k = 3 } ^ { L } k ^ { 2 } \alpha _ { k } C _ { \zeta , k - 1 } ^ { 2 } \right) ^ { 1 / 2 } } & { } \end{array}
$$

with probability at least $1 { - } \zeta .$ . Since $C _ { \zeta , k - 1 } \leq C _ { \zeta , k }$ , it follows from Assumption 4 that $\begin{array} { r } { \sum _ { k = 3 } ^ { L } k ^ { 2 } \alpha _ { k } C _ { \zeta , k } ^ { 2 } \le } \end{array}$ $\begin{array} { r } { \sum _ { k = 3 } ^ { \infty } k ^ { 2 } \alpha _ { k } C _ { \zeta , k } ^ { 2 } = C _ { \zeta } ^ { 2 } < \infty } \end{array}$ . The event in Lemma 3 is simultaneous over all $k ,$ so the preceding estimate holds for every L on the same event. Moreover, for each fixed training sample, the reproducing property and Cauchy–Schwarz give

$$
\| \Gamma _ { 1 } ( g _ { L } - g ) ( { \pmb x } _ { 0 , i } ) \| _ { 2 } \le \| g _ { L } - g \| _ { \mathcal { H } } \left( \mathbb { E } _ { z } \big [ K ( \rho { \pmb x } _ { 0 , i } + \sigma z , \rho { \pmb x } _ { 0 , i } + \sigma z ) \| z \| _ { 2 } ^ { 2 } \big ] \right) ^ { 1 / 2 } \longrightarrow 0 ,
$$

where the expectation is finite under Assumption 4. We may therefore take $L \to \infty$ in the preceding estimate, which proves the second inequality. □

Remark B.1 (Degree-damped power-series kernels are admissible). Assumption 4 is satisfied by a broad family of degree-damped power-series kernels. This includes, for instance, kernel functions of the form

$$
h ( x ) = x + \sum _ { k = 3 } ^ { \infty } \frac { b _ { k } } { ( k ! ) ^ { q } } x ^ { k } , \qquad q > 4 , \qquad 0 \le b _ { k } \le e ^ { O ( k ) } \mathrm { ~ f o r ~ } k \ge 3 , \qquad b _ { k } > 0 \mathrm { ~ f o r ~ } k _ { 0 } \le k \le 2 k _ { 0 } .
$$

## C Linearization of the kernel and localized bumps

This appendix proves the kernel linearization results and constructs localized RKHS bumps. Throughout, the difusion time $t > 0$ is fixed, so we suppress the dependence on $t ,$ and treat $\sigma , \rho$ as positive constants.

## C.1 Linearization of the kernels and Bayes denoiser

In this section, we establish the linearization results stated in propositions 1 and 2.

Proof of Proposition 1. We provide a complete proof of the first bound in Proposition 1. The second bound follows by a similar argument and we only sketch the main diferences at the end of the proof.

Decomposition. For every $i , j \in [ n ]$ and $z , z ^ { \prime } \in \mathbb { R } ^ { d }$ , define $U _ { i , j } ( z , z ^ { \prime } ) : = \langle \rho \pmb { x } _ { 0 , i } + \sigma z , \rho \pmb { x } _ { 0 , j } +$ $\sigma z ^ { \prime } \rangle / d .$

$$
D _ { i , j } ( z , z ^ { \prime } ) : = h ( U _ { i , j } ( z , z ^ { \prime } ) ) - U _ { i , j } ( z , z ^ { \prime } ) - \mathbb { 1 } _ { i = j } \big ( h ( \rho ^ { 2 } \tau _ { \Sigma } ) - \rho ^ { 2 } \tau _ { \Sigma } \big ) ,
$$

and recall that $\tau _ { \Sigma } = \mathrm { T r } ( \Sigma ) / d$ is the average eigenvalue of Σ. Let $f \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ be an arbitrary function with $\| f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = 1$ . We decompose the operator diference into its of-diagonal and diagonal components: $( \widehat { \mathcal { K } } - \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) f = \Delta _ { \mathrm { o d } } f + \Delta _ { \mathrm { d } } f$ , where

$$
\Delta _ { \mathrm { o d } } f ( { \boldsymbol x } _ { 0 , i } , \boldsymbol z ) = \frac { 1 } { n } \sum _ { j \neq i } \mathbb { E } _ { \boldsymbol z ^ { \prime } } \left[ D _ { i , j } ( \boldsymbol { z } , \boldsymbol { z } ^ { \prime } ) f ( { \boldsymbol x } _ { 0 , j } , \boldsymbol { z } ^ { \prime } ) \right] , \quad \Delta _ { \mathrm { d } } f ( { \boldsymbol x } _ { 0 , i } , \boldsymbol { z } ) = \frac { 1 } { n } \mathbb { E } _ { \boldsymbol z ^ { \prime } } \left[ D _ { i , i } ( \boldsymbol { z } , \boldsymbol { z } ^ { \prime } ) f ( { \boldsymbol x } _ { 0 , i } , \boldsymbol { z } ^ { \prime } ) \right]
$$

for every $i \in [ n ]$ and $z \in \mathbb { R } ^ { d }$ . By the triangle inequality, $\| ( \widehat { K } - \widehat { K } _ { \mathrm { e f f } } ) f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq \| \Delta _ { \mathrm { o d } } f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } +$ $\| \Delta _ { \mathrm { d } } f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) }$ . We bound each term separately.

Of-diagonal term. For the of-diagonal operator, we apply Cauchy-Schwarz over the joint measure for $j \neq i \colon$

$$
\begin{array} { l } { \displaystyle \| \Delta _ { \mathrm { o d } } f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z } \Bigg [ \Bigg ( \frac { 1 } { n } \sum _ { j \neq i } \mathbb { E } _ { z ^ { \prime } } \left[ D _ { i , j } ( z , z ^ { \prime } ) f ( x _ { 0 , j } , z ^ { \prime } ) \right] \Bigg ) ^ { 2 } \Bigg ] } \\ { \displaystyle \leq \frac { 1 } { n ^ { 2 } } \sum _ { i \neq j } \mathbb { E } _ { z , z ^ { \prime } } \left[ D _ { i , j } ( z , z ^ { \prime } ) ^ { 2 } \right] = \frac { 1 } { n ^ { 2 } } \sum _ { i \neq j } \mathbb { E } _ { z , z ^ { \prime } } \left[ \left( h \left( U _ { i , j } ( z , z ^ { \prime } ) \right) - U _ { i , j } ( z , z ^ { \prime } ) \right) ^ { 2 } \right] . } \end{array}
$$

Taylor expanding h around 0,

$$
h ( U _ { i , j } ) - U _ { i , j } = \frac { 1 } { 6 } h ^ { \prime \prime \prime } ( \xi _ { i , j } ) U _ { i , j } ^ { 3 }
$$

for some $\xi _ { i , j }$ between 0 and $U _ { i , j }$ . The exponential-growth bound in Assumption 3 along with Young’s inequality and Lemma 1 on the event max $_ i \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d \lesssim 1$ gives

$$
\operatorname* { m a x } _ { i , j } \mathbb { E } _ { z , z ^ { \prime } } \left[ | h ^ { \prime \prime \prime } ( \xi _ { i , j } ) | ^ { 4 } \right] \lesssim 1
$$

with very high probability. Hence,

$$
\begin{array} { r } { \mathbb { E } _ { z , z ^ { \prime } } \left[ ( h ( U _ { i , j } ) - U _ { i , j } ) ^ { 2 } \right] \lesssim \mathbb { E } _ { z , z ^ { \prime } } \left[ | U _ { i , j } | ^ { 1 2 } \right] ^ { 1 / 2 } . } \end{array}
$$

By (42) and (43) (which hold under Assumption 2) and standard properties of Gaussian vectors,

$$
\begin{array} { r l } & { \underset { i \neq j } { \operatorname* { m a x } } \left| \langle \pmb { x } _ { 0 , i } , \pmb { x } _ { 0 , j } \rangle \right| \prec \sqrt { d } , \qquad \underset { i \in [ n ] } { \operatorname* { m a x } } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ | \langle \pmb { x } _ { 0 , i } , z \rangle | ^ { k } ] ^ { 1 / k } \prec \sqrt { d } , } \\ & { \qquad \mathbb { E } _ { ( z , z ^ { \prime } ) \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) ^ { \otimes 2 } } [ | \langle z , z ^ { \prime } \rangle | ^ { k } ] ^ { 1 / k } \lesssim \sqrt { d } , } \end{array}\tag{53}
$$

for every positive integer k. It follows by expanding the definition of $U _ { i , j }$ and Assumption 1 that $\| \Delta _ { \mathrm { o d } } f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \prec d ^ { - 3 / 2 }$ . Since $f \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ is arbitrary, we conclude that $\| \Delta _ { \mathrm { o d } } \| _ { \mathrm { o p } } \prec d ^ { - 3 / 2 }$ by the variational characterization of the operator norm.

Diagonal term. For the diagonal operator, we apply Cauchy-Schwarz solely to the inner expectation over $z ^ { \prime } { : }$

$$
\Vert \Delta _ { \mathbf { d } } f \Vert _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z } \left[ \left( \frac { 1 } { n } \mathbb { E } _ { z ^ { \prime } } \left[ D _ { i , i } ( z , z ^ { \prime } ) f ( x _ { 0 , i } , z ^ { \prime } ) \right] \right) ^ { 2 } \right] \leq \frac { 1 } { n ^ { 2 } } \operatorname* { m a x } _ { i \in [ n ] } \mathbb { E } _ { z , z ^ { \prime } } [ D _ { i , i } ( z , z ^ { \prime } ) ^ { 2 } ] .
$$

Taylor expanding $h ( x ) - x$ around $\rho ^ { 2 } \tau _ { \Sigma }$ , we have $D _ { i , i } ( z , z ^ { \prime } ) = ( h ^ { \prime } ( \xi _ { i } ) - 1 ) ( U _ { i , i } ( z , z ^ { \prime } ) - \rho ^ { 2 } \tau _ { \Sigma } )$ for some $\xi _ { i }$ between $U _ { i , i } ( z , z ^ { \prime } )$ and $\rho ^ { 2 } \tau _ { \Sigma }$ . By Cauchy-Schwarz,

$$
\begin{array} { r } { \mathbb { E } _ { z , z ^ { \prime } } [ D _ { i , i } ( z , z ^ { \prime } ) ^ { 2 } ] \leq \mathbb { E } _ { z , z ^ { \prime } } [ ( h ^ { \prime } ( \xi _ { i } ) - 1 ) ^ { 4 } ] ^ { 1 / 2 } \mathbb { E } _ { z , z ^ { \prime } } [ ( U _ { i , i } ( z , z ^ { \prime } ) - \rho ^ { 2 } \tau _ { \Sigma } ) ^ { 4 } ] ^ { 1 / 2 } . } \end{array}
$$

Recalling the standard identities in (53) and $\begin{array} { r } { \operatorname* { m a x } _ { i \in [ n ] } | \| { \pmb x } _ { 0 , i } \| ^ { 2 } - \mathrm { T r } ( { \pmb \Sigma } ) | \prec \sqrt { d } } \end{array}$ by (42), it follows that $\begin{array} { r } { \mathbb { E } _ { z , z ^ { \prime } } [ | U _ { i , i } ( z , z ^ { \prime } ) - \rho ^ { 2 } \tau _ { \Sigma } | ^ { k } ] ^ { 1 / k } \prec d ^ { - 1 / 2 } } \end{array}$ for every positive integer k. Combining this with (40) and the triangle inequality, $\mathbb { E } _ { z , z ^ { \prime } } [ | U _ { i , i } ( z , z ^ { \prime } ) | ^ { k } ] ^ { 1 / k } \lesssim 1$ with very high probability for every positive integer k. Integrating the exponential-growth bound on $h ^ { \prime \prime \prime }$ and using $h ^ { \prime } ( 0 ) = 1$ gives $| h ^ { \prime } ( u ) - 1 | \leq C \exp ( C | u | )$ Since $\xi _ { i }$ lies on the segment between $U _ { i , i } ( z , z ^ { \prime } )$ and $\rho ^ { 2 } \tau _ { \Sigma } , ( 4 0 )$ , Lemma 1 and the convexity of $x \mapsto e ^ { x }$ yields $\mathbb { E } _ { z , z ^ { \prime } } [ ( h ^ { \prime } ( \xi _ { i } ) - 1 ) ^ { 4 } ] ^ { 1 / 2 } \lesssim 1$ , uniformly in i, with very high probability. The proportional regime (see Assumption 1) yields $\| \Delta _ { \mathrm { d } } f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \prec d ^ { - 3 / 2 }$ uniformly over $f \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ . This concludes the proof of the first part of Proposition 1.

Population kernel. In order to adapt the argument to control $\| \mathcal { K } - \mathcal { K } _ { \mathrm { e f f } } \| _ { \mathrm { o p } }$ , for every $i \in [ n ]$ ， $\pmb { x } , \pmb { z } , \pmb { z } ^ { \prime } \in \mathbb { R } ^ { d }$ define $U _ { i } ( \pmb { x } , z , z ^ { \prime } ) : = \langle \rho \pmb { x } _ { 0 , i } + \sigma z ^ { \prime } , \rho \pmb { x } + \sigma z \rangle / d$ along with

$$
D _ { i } ( { \pmb x } , z , z ^ { \prime } ) : = h ( U _ { i } ( { \pmb x } , z , z ^ { \prime } ) ) - U _ { i } ( { \pmb x } , z , z ^ { \prime } ) .
$$

Then, for every $f \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ with $\| f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq 1$ , we apply Cauchy-Schwarz inequality to obtain

$$
\| ( \boldsymbol { K } - \boldsymbol { K } _ { \mathrm { e f f } } ) f \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } \le \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \pi _ { d } , \boldsymbol { z } ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ D _ { i } ( \boldsymbol { x } , \boldsymbol { z } , \boldsymbol { z } ^ { \prime } ) ^ { 2 } ] .
$$

Taylor expanding $h ( x ) = x + h ^ { \prime \prime \prime } ( \xi _ { i } ) x ^ { 3 } / 6$ around 0, and applying the same H¨older argument, with Assumption 3 and Lemma 1 controlling $h ^ { \prime \prime \prime } ( \xi _ { i } )$ , gives $\mathbb { E } [ D _ { i } ( { \pmb x } , z , z ^ { \prime } ) ^ { 2 } ] \prec d ^ { - 3 }$ uniformly in i. Hence $\| ( \mathcal { K } - \mathcal { K } _ { \mathrm { e f f } } ) f \| _ { L ^ { 2 } ( \pi _ { d } ) } \prec d ^ { - 3 / 2 }$ □

Next, we prove Proposition 2. We recall that $\widehat { f } _ { v } ^ { \star } \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ is defined via the pullback of (11). Proof of Proposition 2. By definition (see (11)), we have

$$
\Vert \widehat { f } _ { v } ^ { \star } - z _ { v } \Vert _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } = \frac { \rho ^ { 2 } } { \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \Bigg [ \Bigg ( \sum _ { k \neq i } \omega _ { k } ( \rho x _ { 0 , i } + \sigma z ) \langle x _ { 0 , i } - x _ { 0 , k } , v \rangle \Bigg ) ^ { 2 } \Bigg ]
$$

where the weights $\omega _ { k }$ are defined in (12). Because $\textstyle \sum _ { k \neq i } \omega _ { k } \leq 1$ , it follows from Jensen’s inequality that

$$
\left( \sum _ { k \neq i } \omega _ { k } ( \rho \pmb { x } _ { 0 , i } + \sigma z ) \langle \pmb { x } _ { 0 , i } - \pmb { x } _ { 0 , k } , \pmb { v } \rangle \right) ^ { 2 } \leq \sum _ { k \neq i } \omega _ { k } ( \rho \pmb { x } _ { 0 , i } + \sigma z ) \langle \pmb { x } _ { 0 , i } - \pmb { x } _ { 0 , k } , \pmb { v } \rangle ^ { 2 } ,
$$

and thus

$$
\| \widehat { f } _ { v } ^ { \star } - z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \leq \frac { \rho ^ { 2 } } { \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \sum _ { k \neq i } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ \omega _ { k } ( \rho x _ { 0 , i } + \sigma z ) \right] \langle x _ { 0 , i } - x _ { 0 , k } , v \rangle ^ { 2 } .
$$

By Assumption 2 and (42), for every $c \in ( 0 , 1 / 2 )$

$$
\| { \pmb x } _ { 0 , k } - { \pmb x } _ { 0 , i } \| _ { 2 } ^ { 2 } \geq d ^ { 1 - c } , \qquad \langle { \pmb x } _ { 0 , i } - { \pmb x } _ { 0 , k } , { \pmb v } \rangle ^ { 2 } \leq d ^ { 1 + c }
$$

for every $i \ne k \in [ n ]$ with very high probability. We denote this high-probability event for the training data by $\mathcal { E } _ { x }$ and condition on it for the rest of the proof, supposing that $d \ge C$ is large enough. We also define a “typical noise” event $\mathcal { Z } _ { i , k }$ for each pair $i \neq$ k as

$$
\mathcal { Z } _ { i , k } : = \left. z \in \mathbb { R } ^ { d } : \frac { \rho } { \sigma } | \langle { \pmb x } _ { 0 , k } - { \pmb x } _ { 0 , i } , z \rangle | \leq \frac { \rho ^ { 2 } } { 4 \sigma ^ { 2 } } \| { \pmb x } _ { 0 , k } - { \pmb x } _ { 0 , i } \| _ { 2 } ^ { 2 } \right. .
$$

By standard Gaussian tail bounds, $\mathcal { Z } _ { i , k }$ holds with very high probability over the randomness of the noise z. We use the event $\mathcal { Z } _ { i , k }$ to safely integrate over the noise z. Indeed, for every $i \neq$ k and $\boldsymbol { z } \in \mathcal { Z } _ { i , k }$ , we isolate the $j = i$ term in the denominator of $\omega _ { k }$ to obtain the deterministic bound

$$
\begin{array} { r } { \omega _ { k } ( \rho x _ { 0 , i } + \sigma z ) \leq \exp \left( - \displaystyle \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \| x _ { 0 , k } - x _ { 0 , i } \| ^ { 2 } + \displaystyle \frac { \rho } { \sigma } \langle x _ { 0 , k } - x _ { 0 , i } , z \rangle \right) } \\ { \leq \exp \left( - \displaystyle \frac { \rho ^ { 2 } } { 4 \sigma ^ { 2 } } \| x _ { 0 , k } - x _ { 0 , i } \| ^ { 2 } \right) \leq \exp \left( - \displaystyle \frac { \rho ^ { 2 } d ^ { 1 - c } } { 4 \sigma ^ { 2 } } \right) . } \end{array}
$$

On the complement $\mathcal { Z } _ { i , k } ^ { c } ;$ , we simply use the trivial bound $\omega _ { k } \leq 1$ . Combining these two bounds yields

$$
\mathbb { E } _ { z } \left[ \omega _ { k } ( \rho x _ { 0 , i } + \sigma z ) \right] = \mathbb { E } _ { z } \left[ \omega _ { k } \mathbb { I } _ { { \mathcal { Z } } _ { i , k } } \right] + \mathbb { E } _ { z } \left[ \omega _ { k } \mathbb { I } _ { { \mathcal { Z } } _ { i , k } ^ { c } } \right] \leq \exp \left( - \frac { \rho ^ { 2 } d ^ { 1 - c } } { 4 \sigma ^ { 2 } } \right) + \mathbb { P } _ { z } ( { \mathcal { Z } } _ { i , k } ^ { c } ) \leq 2 \exp ( - d ^ { c ^ { \prime } } )
$$

for some $c ^ { \prime } > 0$ and suficiently large d. Substituting this back into the expression for the squared norm, we obtain

$$
\| \widehat { f } _ { v } ^ { \star } - z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \le \frac { 2 \rho ^ { 2 } } { \sigma ^ { 2 } n } \sum _ { i = 1 } ^ { n } \sum _ { k \neq i } \exp ( - d ^ { c ^ { \prime } } ) \langle x _ { 0 , i } - x _ { 0 , k } , v \rangle ^ { 2 } \le \frac { 2 \rho ^ { 2 } n d ^ { 1 + c } } { \sigma ^ { 2 } } \exp ( - d ^ { c ^ { \prime } } ) ,
$$

where we have used the bound on the overlaps $\langle \pmb { x } _ { 0 , i } - \pmb { x } _ { 0 , k } , \pmb { v } \rangle ^ { 2 } \leq d ^ { 1 + c }$ for every $i \neq k \in [ n ]$ on the event $\mathcal { E } _ { x }$ . Because $n \propto d ,$ the polynomial prefactor is exponentially dominated for suficiently large d. We conclude that $\| \widehat { f } _ { v } ^ { \star } - z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \leq \exp ( - d ^ { c ^ { \prime \prime } } )$ for some $c ^ { \prime \prime } > 0$ □

Remark C.1 (Kernel linearization with $h ( 0 ) , h ^ { \prime \prime } ( 0 ) \neq 0 )$ . The conditions $h ( 0 ) = h ^ { \prime \prime } ( 0 ) = 0$ are added for convenience to simplify the analysis. If these conditions do not hold, the kernel linearization will include additional constant-function contributions. Specifically, let $\widehat { \mathcal { I } }$ be the orthogonal

projection onto the constant functions in $L ^ { 2 } ( \widehat \pi _ { d , n } )$ , let $\mathcal { I } : L ^ { 2 } ( \widehat { \pi } _ { d , n } )  L ^ { 2 } ( \pi _ { d } )$ map a function to its constant empirical mean, and let $\mathcal { E } f ( \pmb { x } _ { 0 , i } , z ) : = \mathbb { E } _ { z ^ { \prime } } [ f ( \pmb { x } _ { 0 , i } , z ^ { \prime } ) ]$ . Then

$$
\| \widehat { \mathcal { K } } - \widehat { \mathcal { K } } _ { \{ 0 , 2 \} } \| _ { \mathrm { o p } } \vee \| \mathcal { K } - \mathcal { K } _ { \{ 0 , 2 \} } \| _ { \mathrm { o p } } \prec d ^ { - \frac { 3 } { 2 } } ,
$$

where

$$
\widehat { K } _ { \{ 0 , 2 \} } : = \widehat { K } _ { \mathrm { e f f } } + h ( 0 ) \widehat { \mathcal { I } } + \frac { h ^ { \prime \prime } ( 0 ) } { 2 } \nu _ { d } \left( \widehat { \mathcal { I } } - \frac { 1 } { n } \varepsilon \right) , \qquad K _ { \{ 0 , 2 \} } : = K _ { \mathrm { e f f } } + \left( h ( 0 ) + \frac { h ^ { \prime \prime } ( 0 ) } { 2 } \nu _ { d } \right) \mathcal { I } ,
$$

where $\begin{array} { r } { \nu _ { d } : = \frac { \mathrm { T r } ( \Sigma _ { \sigma } ^ { 2 } ) } { d ^ { 2 } } = O _ { d } ( d ^ { - 1 } ) } \end{array}$ . Now, $\delta _ { n } = ( h ( \rho ^ { 2 } \tau _ { \Sigma } ) - h ( 0 ) - \rho ^ { 2 } \tau _ { \Sigma } ) / n$ . These contributions can be handled separately without afecting the main results.

Remark C.2 (Non-negativity of $\delta _ { n } )$ . The self-induced regularization term $\delta _ { n } ~ = ~ ( h ( \rho ^ { 2 } \tau _ { \Sigma } ) ~ -$ $\rho ^ { 2 } \tau _ { \Sigma } ) / n$ , which appears in the definition of the efective kernel operator $\widehat { \kappa } _ { \mathrm { e f f } }$ in (22) as a consequence of the kernel linearization, is non-negative under Assumption 3.

Indeed, fix $\pmb { u } \in \mathbb { R } ^ { d }$ , let $u = \| \pmb { u } \| ^ { 2 } / d .$ , and take $\epsilon > 0$ . Since the kernel is positive definite, the Gram matrix associated with the points u and $\sqrt { \epsilon } \pmb { u }$ satisfies

$$
\begin{array} { r } { \left[ \begin{array} { l l } { h ( u ) } & { h ( \sqrt { \epsilon } u ) } \\ { h ( \sqrt { \epsilon } u ) } & { h ( \epsilon u ) } \end{array} \right] \succeq 0 . } \end{array}
$$

In particular, its determinant is non-negative, and hence $h ( u ) h ( \epsilon u ) \geq h ( \sqrt { \epsilon } u ) ^ { 2 }$ . Dividing by ϵ and taking $\epsilon \to 0$ using $h ( 0 ) = 0$ and $h ^ { \prime } ( 0 ) = 1$ , we obtain $h ( u ) u \geq u ^ { 2 }$ . Thus, $h ( u ) \geq u$ for every $u > 0$ Taking $u = \rho ^ { 2 } \tau _ { \Sigma }$ proves that $\delta _ { n } \geq 0$

Moreover, equality at one positive point forces equality everywhere. Suppose that $h ( u _ { 0 } ) = u _ { 0 }$ for some $u _ { 0 } > 0$ , and choose $\pmb { u } \in \mathbb { R } ^ { d }$ such that $\lVert \mathbf { u } \rVert _ { 2 } ^ { 2 } / d = u _ { 0 }$ . Fix $t \in \mathbb { R }$ and set $\pmb { y } = ( t / u _ { 0 } ) \pmb { u }$ . For $\epsilon > 0$ , positive definiteness applied to u, y, ϵu gives

$$
\left[ \begin{array} { c c c } { h ( u _ { 0 } ) } & { h ( t ) } & { h ( \epsilon u _ { 0 } ) } \\ { h ( t ) } & { h ( t ^ { 2 } / u _ { 0 } ) } & { h ( \epsilon t ) } \\ { h ( \epsilon u _ { 0 } ) } & { h ( \epsilon t ) } & { h ( \epsilon ^ { 2 } u _ { 0 } ) } \end{array} \right] \succeq 0 .
$$

Since $h ( \epsilon ^ { 2 } u _ { 0 } ) > 0$ for suficiently small ϵ, its Schur complement gives

$$
\left[ h ( u _ { 0 } ) \qquad h ( t ) \atop h ( t ^ { 2 } / u _ { 0 } ) \right] - \frac { 1 } { h ( \epsilon ^ { 2 } u _ { 0 } ) } \left[ h ( \epsilon u _ { 0 } ) \right] \left[ h ( \epsilon u _ { 0 } ) \quad h ( \epsilon t ) \right] \succeq 0 .
$$

Letting $\epsilon \to 0$ and using $h ( s ) = s + o ( s )$ at the origin yields

$$
\begin{array} { r l r } & { } & { \left[ h ( u _ { 0 } ) - u _ { 0 } \right. \qquad h ( t ) - t } \\ & { } & { \left. h ( t ) - t \quad h ( t ^ { 2 } / u _ { 0 } ) - t ^ { 2 } / u _ { 0 } \right] \succeq 0 . } \end{array}
$$

Since $h ( u _ { 0 } ) = u _ { 0 }$ , the upper-left entry vanishes. Non-negativity of the determinant therefore gives

$$
- ( h ( t ) - t ) ^ { 2 } \geq 0 ,
$$

and hence $h ( t ) = t$ . Since $t \in \mathbb { R }$ was arbitrary, $h ( t ) = t$ for every $t \in \mathbb { R }$

## C.2 Construction of the localized bumps

In this section, we construct the localized bump functions used in the proof of Theorem 7. Fix $m \in \{ 0 , 1 , 2 , . . . \}$ such that $2 m + 6 = k _ { 0 }$ . Define

$$
p _ { m } ( x ) : = x ^ { k _ { 0 } } \sum _ { j = 0 } ^ { k _ { 0 } } ( - 1 ) ^ { j } { \binom { k _ { 0 } + j - 1 } { j } } ( x - 1 ) ^ { j }\tag{54}
$$

where the convention is that $( x - 1 ) ^ { 0 } = 1$ even when $x = 1$ . We have the following simple properties of the polynomial $p _ { m }$

Lemma 11. Let $m \in \{ 0 , 1 , 2 , . . . \} , \ k _ { 0 } = 2 m + 6$ and $p _ { m }$ be defined as in (54). Then, $p _ { m }$ is a polynomial of degree $2 k _ { 0 }$ with no monomial of degree less than $k _ { 0 }$ . Moreover, $p _ { m } ( 1 ) = 1 , p _ { m } ^ { ( k - 1 ) } ( 0 ) =$ 0 and $p _ { m } ^ { ( k ) } ( 1 ) = 0$ for every $k \in \{ 1 , \ldots , k _ { 0 } \}$

Proof. Momentarily define $f ( x ) \ = \ x ^ { k _ { 0 } }$ and $\begin{array} { r } { g ( x ) = \sum _ { j = 0 } ^ { k _ { 0 } } ( - 1 ) ^ { j } \binom { k _ { 0 } + j - 1 } { j } ( x - 1 ) ^ { j } } \end{array}$ . Note that g is precisely the k<sub>0</sub>-th order Taylor expansion of the function $x \mapsto x ^ { - k _ { 0 } }$ around $x = 1$ , so $g ^ { ( k ) } ( 1 ) =$ $( 1 / f ) ^ { ( k ) } ( 1 )$ for every $k \in \{ 0 , \ldots , k _ { 0 } \}$ . In particular, $g ( 1 ) = 1 / f ( 1 ) = 1$ . Furthermore, note that $p _ { m } ^ { ( k ) } ( 0 ) = 0$ for every $k \in \{ 0 , \ldots , k _ { 0 } - 1 \}$ since the lowest degree monomial in the expansion of $p _ { m }$ is of degree $k _ { 0 }$ . Finally, by the general Leibniz rule, for every $k \in \{ 1 , \ldots , k _ { 0 } \}$ 2

$$
p _ { m } ^ { ( k ) } ( 1 ) = \sum _ { i = 0 } ^ { k } { \binom { k } { i } } f ^ { ( i ) } ( 1 ) g ^ { ( k - i ) } ( 1 ) = \sum _ { i = 0 } ^ { k } { \binom { k } { i } } f ^ { ( i ) } ( 1 ) ( 1 / f ) ^ { ( k - i ) } ( 1 ) = ( f / f ) ^ { ( k ) } ( 1 ) = 0 .
$$

For every $i \in [ n ]$ , define

$$
\phi _ { m , i } ( \pmb { x } ) : = p _ { m } \left( \frac { \langle \pmb { x } , \pmb { x } _ { 0 , i } \rangle } { \rho \tau _ { \pmb { \Sigma } } d } \right) .
$$

Since $p _ { m }$ is a polynomial with terms of degree at least $k _ { 0 } \geq 6 , \phi _ { m , i } \in \mathcal { H } _ { > 3 }$ for every $i \in [ n ]$ ， with $\mathcal { H } _ { \geq 3 }$ defined in (49). Furthermore, for every $z \in \mathbb { R } ^ { d }$ , let $\pmb { \Phi } _ { m } ( \pmb { z } ) \in \mathbb { R } ^ { n \times n }$ be the matrix with entries $\Phi _ { m , i j } ( z ) = \phi _ { m , j } ( \rho x _ { 0 , i } + \sigma z )$ for every $i , j \in [ n ]$ . The following lemma shows that the mean interpolation matrix $\mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \Phi _ { m } ( z ) ]$ is close to the identity matrix in operator norm.

Lemma 12. Under Assumptions 1 and 2,

$$
\begin{array} { r } { \vert \vert \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \Phi _ { m } ( z ) ] - \mathbf { I } _ { n } \vert \vert _ { \mathrm { o p } } \prec d ^ { - 2 } . } \end{array}
$$

Proof. By (42) and (43), for every $k \in \mathbb N$ we have

$$
\operatorname* { m a x } _ { i , j \in [ n ] } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ \left| \frac { \langle \rho x _ { 0 , i } + \sigma z , \pmb { x } _ { 0 , j } \rangle } { \rho \tau _ { \Sigma } d } - \mathbb { 1 } _ { i = j } \right| ^ { k } \right] ^ { 1 / k } \prec \frac { 1 } { \sqrt { d } } .
$$

Here, Assumption 2 ensures that $\tau _ { \Sigma } > c$ for some constant $c > 0$ . By flatness of $p _ { m }$ at 0 and 1 (see Lemma 11), there exists a constant $C _ { m } > 0$ such that

$$
| p _ { m } ( x ) - 1 | \leq C _ { m } | x - 1 | ^ { k _ { 0 } + 1 } ( 1 + | x - 1 | ^ { k _ { 0 } - 1 } ) , \qquad | p _ { m } ( x ) | \leq C _ { m } | x | ^ { k _ { 0 } } ( 1 + | x | ^ { k _ { 0 } } )
$$

for all $x \in \mathbb { R }$ . Hence, by Jensen’s inequality and Cauchy-Schwarz inequality,

$$
\operatorname* { m a x } _ { i \in [ n ] } | \mathbb { E } [ \Phi _ { m , i i } ( z ) ] - 1 | \prec d ^ { - \frac { k _ { 0 } + 1 } { 2 } } , \qquad \operatorname* { m a x } _ { i \neq j \in [ n ] } | \mathbb { E } [ \Phi _ { m , i j } ( z ) ] | \prec d ^ { - \frac { k _ { 0 } } { 2 } } .\tag{55}
$$

By Schur’s test, $\Vert \mathbb { E } [ \pmb { \Phi } _ { m } ( z ) ] - \mathbf { I } _ { n } \Vert _ { \mathrm { o p } } ^ { 2 } \leq \Vert \mathbb { E } [ \pmb { \Phi } _ { m } ( z ) ] - \mathbf { I } _ { n } \Vert _ { 1 } \Vert \mathbb { E } [ \pmb { \Phi } _ { m } ( z ) ] - \mathbf { I } _ { n } \Vert _ { \infty }$ where

$$
\| A \| _ { 1 } : = \operatorname* { m a x } _ { j } \sum _ { i = 1 } ^ { n } | A _ { i , j } | , \qquad \| A \| _ { \infty } : = \operatorname* { m a x } _ { i } \sum _ { j = 1 } ^ { n } | A _ { i , j } |
$$

are the maximum row/column sum, respectively. By (55),

$$
\big \| \mathbb { E } [ \Phi _ { m } ( z ) ] - \mathbf { I } _ { n } \big \| _ { 1 } \leq \operatorname* { m a x } _ { j } \left( \big | \mathbb { E } [ \Phi _ { m , j j } ( z ) ] - 1 \big | + \sum _ { i \neq j } \big | \mathbb { E } [ \Phi _ { m , i j } ( z ) ] \big | \right) \prec \frac { 1 } { d ^ { \frac { k _ { 0 } } { 2 } - 1 } } ,
$$

and similarly $\| \mathbb { E } [ \pmb { \Phi } _ { m } ( z ) ] - \mathbf { I } _ { n } \| _ { \infty } \prec d ^ { - \frac { k _ { 0 } } { 2 } + 1 }$ . Since $k _ { 0 } \geq 6$ , we conclude that $\| \mathbb { E } [ \pmb { \Phi } _ { m } ( z ) ] - \mathbf { I } _ { n } \| _ { \mathrm { o p } } \prec$ $d ^ { - 2 }$ □

Next, define the fluctuation matrix $\pmb { R } _ { m } ( z ) : = \pmb { \Phi } _ { m } ( z ) - \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \pmb { \Phi } _ { m } ( z ) ]$

Lemma 13. Under Assumptions 1 and 2, for any vector $\beta \in \mathbb { R } ^ { n }$

$$
\frac 1 n \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \| R _ { m } ( z ) \beta \| _ { 2 } ^ { 2 } ] \prec d ^ { - k _ { 0 } + 1 } \| \beta \| _ { 2 } ^ { 2 } .
$$

Proof. Using the high-probability bounds on the overlaps in (42), it follows from a similar argument as in the proof of Lemma 12 that

$$
\operatorname* { m a x } _ { i \in [ n ] } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ R _ { m , i i } ( z ) ^ { 2 } \right] ^ { 1 / 2 } \prec d ^ { - \frac { k _ { 0 } + 1 } { 2 } } , \qquad \operatorname* { m a x } _ { i \neq j \in [ n ] } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ R _ { m , i j } ( z ) ^ { 2 } \right] ^ { 1 / 2 } \prec d ^ { - \frac { k _ { 0 } } { 2 } } .
$$

The details are similar so we omit them. For any $\beta \in \mathbb { R } ^ { n }$ , we first apply Minkowski’s inequality to pull the sum out of the expectation, and then apply the Cauchy-Schwarz inequality to separate the fluctuations from the vector components:

$$
\begin{array} { r l } & { \displaystyle \frac 1 n \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \| R _ { m } \beta \| _ { 2 } ^ { 2 } ] \leq \frac 1 n \sum _ { i = 1 } ^ { n } \left( \sum _ { j = 1 } ^ { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ R _ { m , i j } ( z ) ^ { 2 } \right] ^ { 1 / 2 } | \beta _ { j } | \right) ^ { 2 } } \\ & { \quad \quad \quad \leq \frac 1 n \sum _ { i = 1 } ^ { n } \left( \sum _ { j = 1 } ^ { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ R _ { m , i j } ( z ) ^ { 2 } \right] \right) \left( \sum _ { j = 1 } ^ { n } \beta _ { j } ^ { 2 } \right) . } \end{array}
$$

Using the entry-wise bounds and $n = O _ { d } ( d )$ by Assumption 1, the sum of variances for any fixed row i is bounded by

$$
\sum _ { j = 1 } ^ { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ R _ { m , i j } ( z ) ^ { 2 } \right] = \mathbb { E } _ { z } \left[ R _ { m , i i } ( z ) ^ { 2 } \right] + \sum _ { j \neq i } \mathbb { E } _ { z } \left[ R _ { m , i j } ( z ) ^ { 2 } \right] \prec d ^ { - k _ { 0 } + 1 } .
$$

Substituting this back into the sequence of inequalities and noting that $\begin{array} { r } { \sum _ { j = 1 } ^ { n } \beta _ { j } ^ { 2 } = \| \beta \| _ { 2 } ^ { 2 } } \end{array}$ , we obtain

$$
\frac { 1 } { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \| R _ { m } \beta \| _ { 2 } ^ { 2 } ] \prec \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( d ^ { - k _ { 0 } + 1 } \right) \| \beta \| _ { 2 } ^ { 2 } = d ^ { - k _ { 0 } + 1 } \| \beta \| _ { 2 } ^ { 2 } .
$$

Define the localized bump functions as

$$
f _ { m , v } ^ { \mathrm { l o c } } ( \pmb { x } ) : = - \frac { \langle \pmb { x } , \pmb { v } \rangle } { \sigma } + \sum _ { i = 1 } ^ { n } \beta _ { m , i } \phi _ { m , i } ( \pmb { x } ) , \qquad \beta _ { m , v } : = \frac { \rho } { \sigma } \mathbb { E } _ { \pmb { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } [ \Phi _ { m } ( \pmb { z } ) ] ^ { - 1 } \pmb { X } ^ { \top } \pmb { v } .\tag{56}
$$

Proof of Proposition 3. By Lemma 12, $\| \mathbb { E } _ { z } [ \Phi _ { m } ( z ) ] - \mathbf { I } _ { n } \| _ { \mathrm { o p } } \prec d ^ { - 2 }$ . On the resulting very high probability event, $\mathbb { E } _ { z } [ \pmb { \Phi } _ { m } ( z ) ]$ is invertible and $\| \mathbb { E } _ { z } [ \Phi _ { m } ( z ) ] ^ { - 1 } \| _ { \mathrm { o p } } \leq 2$ . Hence, by (40) and (44),

$$
\| \beta _ { m , v } \| _ { 2 } ^ { 2 } \leq \| \mathbb { E } _ { z } [ \Phi _ { m } ( z ) ] ^ { - 1 } \| _ { \mathrm { o p } } ^ { 2 } \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } v ^ { \top } \boldsymbol { X } \boldsymbol { X } ^ { \top } \boldsymbol { v } \lesssim v ^ { \top } \boldsymbol { X } \boldsymbol { X } ^ { \top } \boldsymbol { v } \prec d \| \boldsymbol { v } \| _ { 2 } ^ { 2 } = d .
$$

Next, we bound the loss. By (14),

$$
\mathcal { L } _ { \pmb { v } } ( f _ { m , \pmb { v } } ^ { \mathrm { l o c } } ; 0 ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { \pmb { z } } \left[ \big | e _ { j } ^ { \top } \pmb { \Phi } _ { m } ( \pmb { z } ) \beta _ { m , \pmb { v } } - \frac { \rho } { \sigma } \langle \pmb { x } _ { 0 , j } , \pmb { v } \rangle \big | ^ { 2 } \right] .
$$

Note that by the definition of $\beta _ { m , v }$

$$
\mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ e _ { j } ^ { \top } \Phi _ { m } ( z ) \beta _ { m , v } \right] = \frac { \rho } { \sigma } e _ { j } ^ { \top } \pmb { X } ^ { \top } \pmb { v } = \frac { \rho } { \sigma } \langle \pmb { x } _ { 0 , j } , \pmb { v } \rangle ,
$$

so

$$
\mathcal { L } _ { v } ( f _ { m , v } ^ { \mathrm { l o c } } ; 0 ) = \frac { 1 } { n } \mathbb { E } _ { z \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) } \left[ \| \mathbf { R } _ { m } ( z ) \beta _ { m , v } \| _ { 2 } ^ { 2 } \right]
$$

where we recall that $\pmb { R } _ { m } ( z ) = \pmb { \Phi } _ { m } ( z ) \ – _ { z } [ \pmb { \Phi } _ { m } ( z ) ]$ is the matrix of centered residuals. By Lemma 13,

$$
\begin{array} { r } { \mathcal { L } _ { v } ( f _ { m , v } ^ { \mathrm { l o c } } ; 0 ) \prec d ^ { - k _ { 0 } + 1 } \| \beta _ { m , v } \| _ { 2 } ^ { 2 } \prec d ^ { - k _ { 0 } + 2 } . } \end{array}
$$

Finally, we control the RKHS norm. By Young’s inequality,

$$
\lVert f _ { m , v } ^ { \mathrm { l o c } } \rVert _ { \mathcal { H } } ^ { 2 } \leq \frac { 2 d } { \sigma ^ { 2 } } + 2 \left. \sum _ { i = 1 } ^ { n } \beta _ { m , i } \phi _ { m , i } \right. _ { \mathcal { H } } ^ { 2 } .
$$

We can expand the polynomial function as

$$
\phi _ { m , i } ( \pmb { x } ) = p _ { m } \left( \frac { \langle \pmb { x } , \pmb { x } _ { 0 , i } \rangle } { \rho \tau _ { \pmb { \Sigma } } d } \right) = \sum _ { k = k _ { 0 } } ^ { 2 k _ { 0 } } c _ { m , k } \left( \frac { \langle \pmb { x } , \pmb { x } _ { 0 , i } \rangle } { \rho \tau _ { \pmb { \Sigma } } d } \right) ^ { k } = \sum _ { k = k _ { 0 } } ^ { 2 k _ { 0 } } \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \langle \pmb { W } _ { i , k } , \pmb { x } ^ { \otimes k } \rangle ,
$$

where we define $\begin{array} { r } { W _ { i , k } : = \sqrt { \frac { d ^ { k } } { \alpha _ { k } } } \frac { c _ { m , k } } { ( \rho \tau _ { \Sigma } d ) ^ { k } } x _ { 0 , i } ^ { \otimes k } } \end{array}$

By Lemma 9, the RKHS norm of this sum is exactly the sum of the squared Frobenius norms of these scaled tensors. To bound this, let $G ^ { ( k ) } \in \mathbb { R } ^ { n \times n }$ be the Gram matrix with entries $G _ { i j } ^ { ( k ) } =$ $\langle W _ { i , k } , W _ { j , k } \rangle$ . We can rewrite the squared norm for each degree k as a quadratic form:

$$
\left\| \sum _ { i = 1 } ^ { n } \beta _ { m , i } W _ { i , k } \right\| _ { \mathrm { F } } ^ { 2 } = \beta _ { m , v } ^ { \top } G ^ { ( k ) } \beta _ { m , v } \leq \| G ^ { ( k ) } \| _ { \mathrm { o p } } \| \beta _ { m , v } \| _ { 2 } ^ { 2 } .
$$

By Schur’s test, $\begin{array} { r } { \| \boldsymbol { G } ^ { ( k ) } \| _ { \mathrm { o p } } \leq \operatorname* { m a x } _ { i } \sum _ { j = 1 } ^ { n } | \boldsymbol { G } _ { i j } ^ { ( k ) } | } \end{array}$ . The entries of the Gram matrix evaluate directly to

$$
G _ { i j } ^ { ( k ) } = \frac { d ^ { k } } { \alpha _ { k } } \frac { c _ { m , k } ^ { 2 } } { ( \rho \tau _ { \Sigma } d ) ^ { 2 k } } \langle x _ { 0 , i } , x _ { 0 , j } \rangle ^ { k } = \frac { c _ { m , k } ^ { 2 } } { \alpha _ { k } ( \rho \tau _ { \Sigma } ) ^ { 2 k } } \left( \frac { \langle x _ { 0 , i } , x _ { 0 , j } \rangle } { d } \right) ^ { k } .
$$

By (42), for the diagonal entries we have $( \langle { \pmb x } _ { 0 , i } , { \pmb x } _ { 0 , i } \rangle / d ) ^ { k } \ \prec \ 1$ . Since $\alpha _ { k } \ > \ 0$ is a constant by Assumption 4, the diagonal elements satisfy $G _ { i i } ^ { ( k ) } \prec 1$

For the of-diagonal entries $( i \neq j )$ , the cross-terms decay rapidly since $| \langle x _ { 0 , i } , x _ { 0 , j } \rangle / d | ^ { k } \prec d ^ { - k / 2 }$ Because $n \asymp d$ and the active degrees in the bump construction satisfy $k \geq k _ { 0 } \geq 6$ , the sum over the $n - 1$ of-diagonal elements vanishes asymptotically:

$$
\sum _ { j \neq i } | { \cal G } _ { i j } ^ { ( k ) } | \prec n d ^ { - k / 2 } \asymp d ^ { 1 - k / 2 } \prec 1 .
$$

This yields $\| G ^ { ( k ) } \| _ { \mathrm { o p } } \prec 1$ . Therefore,

$$
\bigg \| \sum _ { i = 1 } ^ { n } \beta _ { m , i } \phi _ { m , i } \bigg \| _ { \mathcal { H } } ^ { 2 } = \sum _ { k = k _ { 0 } } ^ { 2 k _ { 0 } } \bigg \| \sum _ { i = 1 } ^ { n } \beta _ { m , i } W _ { i , k } \bigg \| _ { \mathrm { F } } ^ { 2 } \leq \sum _ { k = k _ { 0 } } ^ { 2 k _ { 0 } } \| \pmb { G } ^ { ( k ) } \| _ { \mathrm { l o p } } \| \beta _ { m , v } \| _ { 2 } ^ { 2 } \prec \| \beta _ { m , v } \| _ { 2 } ^ { 2 } \prec d .
$$

Substituting this into the initial inequality gives $\| f _ { m , \pmb { v } } ^ { \mathrm { l o c } } \| _ { \mathcal { H } } ^ { 2 } \prec d ,$ concluding the proof.

## D Proofs of the kernel ridge regression results

This section is devoted to proving the asymptotic risk equivalents stated in Theorems 6 and $7 .$ It is implicitly assumed that the difusion time $t > 0$ is fixed, implying that $\sigma , \rho$ are positive constants.

## D.1 Proof of Theorem 6

We proceed through a sequence of intermediate results, beginning with the linearization of the training and test errors via Proposition 1 and Proposition 2. For notational convenience, let

$$
\widehat { \mathcal { G } } ^ { ( \kappa ) } : = \left( \widehat { \mathcal { K } } + \frac { \kappa } { d } \mathrm { i d } \right) ^ { - 1 } , \qquad \widehat { \mathcal { G } } _ { \mathrm { e f f } } ^ { ( \kappa ) } : = \left( \widehat { \mathcal { K } } _ { \mathrm { e f f } } + \frac { \kappa } { d } \mathrm { i d } \right) ^ { - 1 } .
$$

Since $\widehat { \kappa }$ and $\widehat { \mathcal { K } } _ { \mathrm { e f f } }$ are positive semi-definite, $\| \widehat { \mathcal { G } } ^ { ( \kappa ) } \| _ { \mathrm { o p } } \vee \| \widehat { \mathcal { G } } _ { \mathrm { e f f } } ^ { ( \kappa ) } \| _ { \mathrm { o p } } \leq d / \kappa$ . Define the linearized training and test errors coordinate-wise as

$$
\mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } : = \frac { \kappa ^ { 2 } } { d ^ { 2 } } \left\| \widehat { \mathcal { G } } _ { \mathrm { e f f } } ^ { ( \kappa ) } z _ { v } \right\| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } , \qquad \mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } : = \left\| \mathcal { K } _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } ^ { ( \kappa ) } z _ { v } - z _ { v } \right\| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } .
$$

The full linearized training and test errors are then defined by averaging over any orthonormal basis of $\mathbb { R } ^ { d }$ . For simplicity, we drop the dependence of the various quantities on κ in the notation. The following lemma controls the operator norm of the kernel operators $\kappa , \kappa _ { \mathrm { e f f } }$

Lemma 14. Suppose that Assumptions 1, 2 and 3 hold. Then, $\| \mathcal { K } \| _ { \mathrm { o p } } \vee \| \mathcal { K } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \prec d ^ { - 1 }$

Proof. For every $f \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$

$$
\begin{array} { r l } & { \| \displaystyle \mathbb { K } _ { \mathrm { e f f } } f \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \pi _ { d } } \left[ \mathbb { E } _ { ( \boldsymbol { x } ^ { \prime } , \boldsymbol { z } ^ { \prime } ) \sim \widehat { \pi } _ { d , n } } \left[ \frac { \left. \rho \boldsymbol { x } ^ { \prime } + \sigma \boldsymbol { z } ^ { \prime } , \rho \boldsymbol { x } + \sigma \boldsymbol { z } \right. } { d } f ( \boldsymbol { x } ^ { \prime } , \boldsymbol { z } ^ { \prime } ) \right] ^ { 2 } \right] } \\ & { \quad \quad \quad \quad \quad \quad = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) , ( \boldsymbol { x } ^ { \prime } , \boldsymbol { z } ^ { \prime } ) \sim \widehat { \pi } _ { d , n } ^ { \otimes 2 } } \left[ \frac { \left. \rho \boldsymbol { x } ^ { \prime } + \sigma \boldsymbol { z } ^ { \prime } , \sum _ { \sigma } ( \rho \boldsymbol { x } + \sigma \boldsymbol { z } ) \right. } { d ^ { 2 } } f ( \boldsymbol { x } , \boldsymbol { z } ) f ( \boldsymbol { x } ^ { \prime } , \boldsymbol { z } ^ { \prime } ) \right] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}
$$

where we recall that $\Sigma _ { \sigma } = \rho ^ { 2 } \Sigma + \sigma ^ { 2 } \mathbf { I } _ { d } . \mathrm { ~ B y ~ } ( 4 0 ) , \| \Sigma _ { \sigma } \| _ { \mathrm { o p } } \leq \rho ^ { 2 } C _ { \mathrm { L S I } } + \sigma ^ { 2 } \leq C _ { \mathrm { L S I } }$ for $C _ { \mathrm { L S I } }$ the Poincar´e constant of the input distribution (see Assumption 2). On the other hand, by Cauchy-Schwarz,

$$
\begin{array} { r l } { \Big | \Big | \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \widehat { \pi } _ { d , n } } [ f ( \boldsymbol { x } , \boldsymbol { z } ) ( \rho \boldsymbol { x } + \sigma \boldsymbol { z } ) ] \Big | \Big | _ { 2 } ^ { 2 } = \underset { \| \boldsymbol { u } \| _ { 2 } = 1 } { \operatorname* { s u p } } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \widehat { \pi } _ { d , n } } [ f ( \boldsymbol { x } , \boldsymbol { z } ) \langle \rho \boldsymbol { x } + \sigma \boldsymbol { z } , \boldsymbol { u } \rangle ] ^ { 2 } } & { } \\ { \leq \| f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \underset { \| \boldsymbol { u } \| _ { 2 } = 1 } { \operatorname* { s u p } } \ \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \widehat { \pi } _ { d , n } } [ \langle \rho \boldsymbol { x } + \sigma \boldsymbol { z } , \boldsymbol { u } \rangle ^ { 2 } ] } & { } \\ { \leq \| f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \left\| \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { z } ) \sim \widehat { \pi } _ { d , n } } [ ( \rho \boldsymbol { x } + \sigma \boldsymbol { z } ) ( \rho \boldsymbol { x } + \sigma \boldsymbol { z } ) ^ { \mathsf { T } } ] \right\| _ { \mathrm { o p } } . } & { } \end{array}
$$

By (40) and (44),

$$
\bigg \vert \bigg \vert \mathbb { E } _ { ( \pmb { x } , z ) \sim \widehat { \pi } _ { d , n } } \big [ ( \rho \pmb { x } + \sigma z ) ( \rho \pmb { x } + \sigma z ) ^ { \mathsf { T } } \big ] \bigg \vert _ { \mathsf { o p } } = \bigg \vert \bigg \vert \frac { \rho ^ { 2 } \pmb { X } \pmb { X } ^ { \mathsf { T } } } { n } + \sigma ^ { 2 } \mathbf { I } _ { d } \bigg \vert \bigg \vert _ { \mathsf { o p } } \prec 1
$$

and thus we may combine the bounds to get $\| { \cal K } _ { \mathrm { e f f } } f \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } \prec d ^ { - 2 } \| f \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 }$ . By the variational characterization of the operator norm, $\| { \mathcal { K } } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \prec d ^ { - 1 }$ . We conclude the proof using Proposition 1.

We have the following linearization result for the training and test errors.

Lemma 15. Suppose that Assumptions $\textstyle 1 , \ 2$ and 3 hold. Then, there exists $c > 0$ such that for every $\pmb { v } \in \mathbb { R } ^ { d }$ with $\| \pmb { v } \| _ { 2 } = 1$ 2

$$
| \mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; 0 ) - \mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } | \prec \frac { 1 } { \kappa \sqrt { d } } + e ^ { - d ^ { c } } , \qquad | \mathcal { R } _ { v } ( \widehat { f } _ { \kappa , v } ) - \mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } | \prec \frac { ( \kappa + 1 ) ^ { 2 } } { \kappa ^ { 3 } \sqrt { d } } .
$$

Proof. We first consider the training error. By Proposition 2 there exists a constant $c > 0$ such that

$$
\| \widehat { f } _ { v } ^ { \star } - z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq e ^ { - d ^ { c } }
$$

with very high probability, which implies that

$$
\left| \mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; 0 ) - \frac { \kappa ^ { 2 } } { d ^ { 2 } } \| \widehat { \mathcal { G } } z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \right| \lesssim e ^ { - d ^ { c } } .
$$

Next, using the scalar inequality $| \| x \| ^ { 2 } - | | y | | ^ { 2 } | \leq | | x - y | | ( | | x | | + | | y | | )$ for every $x , y$ in a normed space, we have

$$
\begin{array} { r } { \left| \| \widehat { \mathcal { G } } { \mathbf 2 } _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } - \| \widehat { \mathcal { G } } _ { \mathrm { e f f } } { \mathbf 2 } _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \right| \leq \| \widehat { \mathcal { G } } { \mathbf 2 } _ { v } - \widehat { \mathcal { G } } _ { \mathrm { e f f } } { \mathbf 2 } _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \left( \| \widehat { \mathcal { G } } { \mathbf 2 } _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } + \| \widehat { \mathcal { G } } _ { \mathrm { e f f } } { \mathbf 2 } _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \right) . } \end{array}
$$

By the resolvent identity, the norm bounds $\| \widehat { \mathcal { G } } ^ { ( \kappa ) } \| _ { \mathrm { o p } } \vee \| \widehat { \mathcal { G } } _ { \mathrm { e f f } } ^ { ( \kappa ) } \| _ { \mathrm { o p } } \leq d / \kappa$ and Proposition 1,

$$
\| \widehat { \mathcal { G } } - \widehat { \mathcal { G } } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \leq \| \widehat { \mathcal { G } } \| _ { \mathrm { o p } } \| \widehat { K } - \widehat { K } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \| \widehat { \mathcal { G } } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \prec \frac { \sqrt { d } } { \kappa ^ { 2 } } .\tag{57}
$$

Furthermore, $\| z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = 1$ and so

$$
\left| \frac { \kappa ^ { 2 } } { d ^ { 2 } } \| \widehat { \mathcal { G } } \boldsymbol { z } _ { \boldsymbol { v } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } - \frac { \kappa ^ { 2 } } { d ^ { 2 } } \| \widehat { \mathcal { G } } _ { \mathrm { e f f } } \boldsymbol { z } _ { \boldsymbol { v } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \right| \prec \frac { 1 } { \kappa \sqrt { d } } .
$$

This proves the first part of the lemma.

We use a similar argument to control the test error. We swap the empirical Bayes estimator $\widehat { f } _ { v } ^ { \star }$ with $z _ { v }$ using Proposition 2:

$$
\begin{array} { r } { \Big | \mathcal { R } _ { v } ( \widehat { f } _ { \kappa , v } ) - \Big \| K \widehat { \mathcal { G } } z _ { v } - z _ { v } \Big \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } \Big | \leq \| K \widehat { \mathcal { G } } \| _ { \mathrm { l o p } } \| \widehat { f } _ { v } ^ { * } - z _ { v } \Big \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \Big ( 2 + \| K \widehat { \mathcal { G } } \| _ { \mathrm { l o p } } \| \widehat { f } _ { v } ^ { * } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } + \| K \widehat { \mathcal { G } } \| _ { \mathrm { l o p } } \Big ) . } \end{array}
$$

Here, we have used the triangle inequality and the fact that $\| z _ { v } \| _ { L ^ { 2 } ( \pi _ { d } ) } = 1$ . By Lemma 14 and the norm bounds on the resolvents, $\| \mathcal { K } \widehat { \mathcal { G } } \| _ { \mathrm { o p } } \leq \| \mathcal { K } \| _ { \mathrm { o p } } \| \widehat { \mathcal { G } } \| _ { \mathrm { o p } } \prec 1 / \kappa$ . Hence, there exists a constant $c > 0$ such that

$$
\left| \mathcal { R } _ { v } ( \widehat { f } _ { \kappa , v } ) - \left\| \mathcal { K } \widehat { \mathcal { G } } z _ { v } - z _ { v } \right\| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } \right| \prec \frac { ( \kappa + 1 ) e ^ { - d ^ { c } } } { \kappa ^ { 2 } } .
$$

We can now compare $\| \mathcal { K } \widehat { \mathcal { G } } z _ { v } - z _ { v } \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 }$ with $\| \mathcal { K } _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } \mathsf { z } _ { v } - \mathsf { z } _ { v } \| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 }$ using a similar argument as for the training error. We have

$$
\begin{array} { r l } & { \bigg \vert \bigg \vert \big \vert K \widehat { \mathcal { G } } z _ { v } - z _ { v } \bigg \vert \bigg \vert _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } - \mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } \bigg \vert \leq \bigg \| K \widehat { \mathcal { G } } z _ { v } - K _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v } \bigg \| _ { L ^ { 2 } ( \pi _ { d } ) } \left( 2 + \| K \widehat { \mathcal { G } } z _ { v } \| _ { L ^ { 2 } ( \pi _ { d } ) } + \| K _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v } \| _ { L ^ { 2 } ( \pi _ { d } ) } \right) } \\ & { \qquad \prec \frac { \kappa + 1 } { \kappa } \left( \frac { 1 } { d } \| \widehat { \mathcal { G } } - \widehat { \mathcal { G } } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } + \frac { d } { \kappa } \| K - K _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \right) . } \end{array}
$$

By Proposition 1 and (57), we conclude that

$$
\left| \left\| \mathcal { K } \widehat { \mathcal { G } } \mathsf { z } _ { v } - \mathsf { z } _ { v } \right\| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } - \mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } \right| \prec \frac { \kappa + 1 } { \kappa ^ { 3 } \sqrt { d } } + \frac { \kappa + 1 } { \kappa ^ { 2 } \sqrt { d } } .
$$

We absorb the exponentially small term into the polynomially small term and conclude the proof.

Having approximated the training and test errors by their linearized counterparts, we now derive more explicit representations of the linearized estimators $\widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v }$ and $\kappa _ { \mathrm { e f f } } \hat { \mathcal G } _ { \mathrm { e f f } } z _ { v }$ . This will allow us to compute the asymptotic limits of the linearized training and test errors. Recall the definition of the resolvent matrix $\widehat { R }$ in (47). Given fixed ${ \boldsymbol w } _ { \boldsymbol x } , { \boldsymbol w } _ { z } \in \mathbb { R } ^ { d }$ , we define the linear function $L _ { w _ { x } , w _ { z } } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \to \mathbb { R }$ as $L _ { w _ { x } , w _ { z } } ( \mathbf { x } , z ) = \langle \pmb { w } _ { x } , \pmb { x } \rangle + \langle \pmb { w } _ { z } , z \rangle$ for every $\pmb { x } , z \in \mathbb { R } ^ { d }$ . We will also write $L _ { w } ( { \pmb x } , z )$ for $\pmb { w } \in \mathbb { R } ^ { 2 d }$ , in which case ${ \pmb w } _ { \pmb x }$ and $\pmb { w } _ { z }$ are the first and last d coordinates of ${ \pmb w } _ { \mathrm { : } }$ respectively. We have the following representation.

Lemma 16. For every $\pmb { v } \in \mathbb { R } ^ { d }$ with $\| \pmb { v } \| _ { 2 } = 1$ , we have $\widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v } = L _ { \widehat { w } }$ and $K _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v } = L _ { w }$ where

$$
\small \widehat { \pmb { w } } : = \left[ - \left( \frac { \sigma d } { \sigma ^ { 2 } + \kappa } \mathbf { I } _ { d } + \frac { \sigma ^ { 2 } d } { \kappa ( \sigma ^ { 2 } + \kappa ) n } { \pmb X } { \pmb X } ^ { \top } \widehat { \pmb R } \right) \pmb v \right] , \qquad \pmb { w } = - \frac { \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } \left[ \rho \sigma \widehat { \pmb { R } } \pmb v \right] .
$$

Proof. Let $\boldsymbol { w } _ { \boldsymbol { x } } , \boldsymbol { w } _ { z } \in \mathbb { R } ^ { d }$ be arbitrary and denote $\pmb { w } = [ \pmb { w } _ { \pmb { x } } ^ { \top } ; \pmb { w } _ { z } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { 2 d }$ . By definition of the efective kernel operator $\widehat { \kappa } _ { \mathrm { e f f } }$ (22), we readily verify that ${ \dot { K } } _ { \mathrm { e f f } } L _ { w } = L _ { { \widehat { K } } w }$ , where the empirical kernel matrix $\widehat { \kappa }$ is defined as

$$
\widehat { \pmb { K } } : = \left[ \begin{array} { c c } { \frac { \rho ^ { 2 } } { d n } \pmb { X } \pmb { X } ^ { \mathsf { T } } + \delta _ { n } \mathbf { I } _ { d } } & { \frac { \rho \sigma } { d } \mathbf { I } _ { d } } \\ { \frac { \rho \sigma } { d n } \pmb { X } \pmb { X } ^ { \mathsf { T } } } & { \frac { \sigma ^ { 2 } } { d } \mathbf { I } _ { d } } \end{array} \right] .
$$

Therefore, $\widehat { \mathcal { K } } _ { \mathrm { e f f } } + \frac { \kappa } { d }$ id acts as the linear operator on the space of linear functions given by the matrix $\widehat { \pmb { K } } + \frac { \kappa } { d } \mathbf { I } _ { 2 d }$ , and thus $\begin{array} { r } { \widehat { \mathcal { G } } _ { \mathrm { e f f } } L _ { w } = L _ { ( \widehat { K } + \frac { \kappa } { d } \mathbf { I } _ { 2 d } ) ^ { - 1 } w } . } \end{array}$ . Using the block structure of $\widehat { \pmb { K } }$ and the definition of ${ \widehat { R } } ,$ , we verify that

$$
\begin{array} { r } { \left( \widehat { \pmb { K } } + \frac { \kappa } { d } \mathbf { I } _ { 2 d } \right) ^ { - 1 } = \left[ \frac { \frac { d ( \sigma ^ { 2 } + \kappa ) } { \rho ^ { 2 } \kappa } \widehat { \pmb { R } } } { - \frac { \sigma d } { \rho \kappa n } \pmb { X } \pmb { X } ^ { \mathsf { T } } \widehat { \pmb { R } } } \frac { d } { \sigma ^ { 2 } + \kappa } \mathbf { I } _ { d } + \frac { \sigma ^ { 2 } d } { \kappa ( \sigma ^ { 2 } + \kappa ) n } { \pmb { X } \pmb { X } } ^ { \mathsf { T } } \widehat { \pmb { R } } \right] . } \end{array}
$$

Plugging $w _ { x } = 0$ and ${ \pmb w } _ { z } = - { \pmb v }$ into the expression above, we obtain

$$
\begin{array} { r } { \widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v } = L _ { \widehat { w } } ( \boldsymbol { x } , z ) , \qquad \widehat { w } : = \left[ - \left( \frac { \frac { \sigma d } { \rho \kappa } \boldsymbol { \mathbf { I } } _ { d } } { \sigma ^ { 2 } + \kappa } \boldsymbol { \mathbf { I } } _ { d } + \frac { \sigma ^ { 2 } d } { \kappa ( \sigma ^ { 2 } + \kappa ) n } \boldsymbol { X } \boldsymbol { X } ^ { \top } \widehat { \boldsymbol { R } } \right) \boldsymbol { v } \right] . } \end{array}
$$

By a similar argument, we can also verify that $K _ { \mathrm { e f f } } L _ { w } ( { \pmb x } , z ) = L _ { K w } ( { \pmb x } , z )$ where $K \in \mathbb { R } ^ { 2 d \times 2 d }$ is the kernel matrix defined as

$$
\begin{array} { r } { \pmb { K } : = \left[ \frac { \rho ^ { 2 } } { d n } \pmb { X } \pmb { X } ^ { \top } \quad \frac { \rho \sigma } { d } \mathbf { I } _ { d } \right] . } \\ { \pmb { \mathcal { C } } } \end{array}
$$

Therefore, $K _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v } = L _ { K \widehat { w } }$ which simplifies to

$$
K \widehat { \pmb { w } } = - \frac { \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } \left[ \rho \sigma \widehat { \pmb { R } } \pmb { v } \right] ,
$$

which is what we wanted to show.

We can now compute the asymptotic limits of the linearized training and test errors using the explicit representations of the linearized estimators and the first- and second-order deterministic equivalents of the resolvent matrix $\widehat { R }$ established in Lemmas 5 and 8.

Lemma 17. Suppose that Assumptions 1 and 2 hold. Then,

$$
| \mathcal { L } _ { \mathrm { l i n } } ^ { ( \kappa ) } - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \vee | \mathcal { R } _ { \mathrm { l i n } } ^ { ( \kappa ) } - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { 1 } { \sqrt { d } } .
$$

Proof. We first consider the training error. Let $\pmb { v } \in \mathbb { R } ^ { d }$ with $\| \pmb { v } \| _ { 2 } = 1$ . By Lemma 16 and the identity $n ^ { - 1 } X X ^ { \top } \widehat { R } = \mathbf { I } _ { d } - \kappa ^ { \star } \widehat { R }$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } = \displaystyle \frac { \kappa ^ { 2 } } { d ^ { 2 } } \left( \frac { \sigma ^ { 2 } d ^ { 2 } } { \rho ^ { 2 } \kappa ^ { 2 } } v ^ { \top } \widehat { R } ( \mathbf { I } _ { d } - \kappa ^ { \star } \widehat { R } ) v + \left\| \left( \frac { d } { \sigma ^ { 2 } + \kappa } \mathbf { I } _ { d } + \frac { \sigma ^ { 2 } d } { \kappa ( \sigma ^ { 2 } + \kappa ) } ( \mathbf { I } _ { d } - \kappa ^ { \star } \widehat { R } ) \right) v \right\| _ { 2 } ^ { 2 } \right) } \\ & { \quad \quad = 1 + \left( \frac { \sigma ^ { 2 } } { \rho ^ { 2 } } - \frac { 2 \sigma ^ { 2 } \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } \right) v ^ { \top } \widehat { R } v + \left( \frac { \sigma ^ { 4 } ( \kappa ^ { \star } ) ^ { 2 } } { ( \sigma ^ { 2 } + \kappa ) ^ { 2 } } - \frac { \sigma ^ { 2 } \kappa ^ { \star } } { \rho ^ { 2 } } \right) v ^ { \top } \widehat { R } ^ { 2 } v . } \end{array}
$$

Summing over an orthonormal basis $\{ \pmb { v } _ { 1 } , \ldots , \pmb { v } _ { d } \}$ of $\mathbb { R } ^ { d }$ , we obtain

$$
{ \mathcal { L } } _ { \mathrm { l i n } } ^ { ( \kappa ) } = 1 + \left( { \frac { \sigma ^ { 2 } } { \rho ^ { 2 } } } - { \frac { 2 \sigma ^ { 2 } \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } } \right) { \frac { \mathrm { T r } ( { \widehat { \pmb { R } } } ) } { d } } + \left( { \frac { ( \sigma ^ { 2 } ) ^ { 2 } ( \kappa ^ { \star } ) ^ { 2 } } { ( \sigma ^ { 2 } + \kappa ) ^ { 2 } } } - { \frac { \sigma ^ { 2 } \kappa ^ { \star } } { \rho ^ { 2 } } } \right) { \frac { \mathrm { T r } ( { \widehat { \pmb { R } } } ^ { 2 } ) } { d } } .
$$

By Lemmas 5 and 8,

$$
\frac { 1 } { d } \left| \mathrm { T r } ( \widehat { R } - M ) \right| \prec \frac { 1 } { \kappa ^ { \star } \sqrt { d } } , \qquad \frac { 1 } { d } \left| \mathrm { T r } ( \widehat { R } ^ { 2 } - M ^ { ( 2 ) } ) \right| \prec \frac { 1 } { ( \kappa ^ { \star } ) ^ { 2 } \sqrt { d } } .\tag{58}
$$

Hence,

$$
\mathcal { L } _ { \mathrm { l i n } } ^ { ( \kappa ) } = 1 + \left( \frac { \sigma ^ { 2 } } { \rho ^ { 2 } } - \frac { 2 \sigma ^ { 2 } \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } \right) \frac { \mathrm { T r } ( M ) } { d } + \left( \frac { ( \sigma ^ { 2 } ) ^ { 2 } ( \kappa ^ { \star } ) ^ { 2 } } { ( \sigma ^ { 2 } + \kappa ) ^ { 2 } } - \frac { \sigma ^ { 2 } \kappa ^ { \star } } { \rho ^ { 2 } } \right) \frac { \mathrm { T r } ( M ^ { ( 2 ) } ) } { d } + O _ { \prec } ( d ^ { - 1 / 2 } ) .
$$

Substituting $\begin{array} { r } { \kappa _ { \kappa } ^ { \star } = \frac { d ( \sigma ^ { 2 } + \kappa ) \chi _ { \kappa } } { \rho ^ { 2 } \kappa } } \end{array}$ , we recover the expression for ${ \mathcal { L } } _ { \mathrm { d e t } } ^ { ( \kappa ) }$ in (38).

For the test error, we use the representation of the linearized estimator $\kappa _ { \mathrm { e f f } } \widehat { \mathcal { G } } _ { \mathrm { e f f } } z _ { v }$ in Lemma 16 to verify that

$$
\mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \kappa ) } = 1 - \frac { 2 \kappa ^ { \star } \sigma ^ { 2 } } { \sigma ^ { 2 } + \kappa } v ^ { \top } \widehat { R } v + \left( \frac { \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } \right) ^ { 2 } \sigma ^ { 2 } v ^ { \top } \widehat { R } \Sigma _ { \sigma } \widehat { R } v .
$$

Summing over an orthonormal basis $\{ \pmb { v } _ { 1 } , \ldots , \pmb { v } _ { d } \}$ of $\mathbb { R } ^ { d }$ and using (58), we obtain

$$
\mathcal { R } _ { \mathrm { l i n } } ^ { ( \kappa ) } = 1 - \frac { 2 \kappa ^ { \kappa } \sigma ^ { 2 } } { d ( \sigma ^ { 2 } + \kappa ) } \mathrm { T r } ( M ) + \left( \frac { \kappa ^ { \star } } { \sigma ^ { 2 } + \kappa } \right) ^ { 2 } \frac { \sigma ^ { 2 } } { d } \mathrm { T r } ( M ^ { ( 2 ) } \Sigma _ { \sigma } ) + O _ { \prec } ( d ^ { - 1 / 2 } ) = \mathcal { R } _ { \mathrm { d e t } } ^ { ( \kappa ) } + O _ { \prec } ( d ^ { - 1 / 2 } ) ,
$$

where we have substituted the expression for $\kappa ^ { \star }$ to recover the expression for ${ \mathcal { R } } _ { \mathrm { d e t } } ^ { ( \kappa ) }$ in (39). □

We are now ready to prove Theorem 6.

Proof of Theorem 6. Let $\{ \pmb { v } _ { j } \} _ { j = 1 } ^ { d }$ be an orthonormal basis of $\mathbb { R } ^ { d }$ . Combining Lemma 15 and Lemma 17, we conclude that

$$
| \mathcal { L } ( \widehat { f } _ { \kappa } ; 0 ) - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \leq \frac { 1 } { d } \sum _ { j = 1 } ^ { d } | \mathcal { L } _ { v _ { j } } ( \widehat { f } _ { \kappa , v _ { j } } ; 0 ) - \mathcal { L } _ { \mathrm { l i n } , v _ { j } } ^ { ( \kappa ) } | + | \mathcal { L } _ { \mathrm { l i n } } ^ { ( \kappa ) } - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { \kappa + 1 } { \kappa \sqrt { d } } ,
$$

where we have absorbed the exponentially small term into the polynomially small term. Similarly,

$$
| \mathcal { R } ( \widehat { \mathbf { f } _ { \kappa } } ) - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \kappa ) } | \prec \frac { \kappa ^ { 3 } + 1 } { \kappa ^ { 3 } \sqrt { d } } ,
$$

where we absorbed the term $1 / ( \kappa ^ { 2 } \sqrt { d } )$ into the term $( \kappa ^ { 3 } + 1 ) / ( \kappa ^ { 3 } \sqrt { d } )$ . This concludes the proof of the first part of the theorem.

For the second part, when $h ( x ) = x$ for all $x \in \mathbb { R } , \kappa = \mathcal { K } _ { \mathrm { e f f } }$ and $\widehat { \kappa } = \widehat { \kappa } _ { \mathrm { e f f } }$ . Adapting Lemma 15, we find that there exists a constant $c > 0$ such that for any unit vector $\pmb { v } \in \mathbb { R } ^ { d }$ ，

$$
\vert \mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; 0 ) - \mathcal { L } _ { \operatorname* { l i n } , v } ^ { ( \kappa ) } \vert \prec e ^ { - d ^ { c } } , \qquad \mathrm { a n d } \qquad \vert \mathcal { R } _ { v } ( \widehat { f } _ { \kappa , v } ) - \mathcal { R } _ { \operatorname* { l i n } , v } ^ { ( \kappa ) } \vert \prec \frac { ( \kappa + 1 ) e ^ { - d ^ { c } } } { \kappa ^ { 2 } } .
$$

The rest of the proof is then identical to the previous case.

## D.2 Proof of Theorem 7

The proof of Theorem 7 relies on the construction of a localized bump function, with n bumps that interpolate the empirical Bayes denoiser $\widehat { f } _ { v } ^ { \star }$ around each sample ${ \pmb x } _ { 0 , i }$ , with small training error and RKHS norm. This construction is done in Appendix C.2. Theorem 7 is then proved by comparing the ridge estimator with the localized bump function, and then using the structure of the RKHS to show that the linear coeficients of the ridge estimator are close to the target vector v.

Proof of Theorem 7. Let $m \in \{ 0 , 1 , 2 , . . . \}$ and $k _ { 0 } = 2 m + 6$ be guaranteed by Assumption 4, and let $v \in \mathbb { S } ^ { d - 1 }$ . By Proposition 3, the localized bump function $f _ { m , v } ^ { \mathrm { l o c } }$ defined in (56) satisfies

$$
\begin{array} { r } { \mathcal { L } _ { \boldsymbol { v } } ( f _ { m , \boldsymbol { v } } ^ { \mathrm { l o c } } ; 0 ) \prec { d ^ { - { k _ { 0 } + 2 } } } , \qquad \| f _ { m , \boldsymbol { v } } ^ { \mathrm { l o c } } \| _ { \mathcal { H } } ^ { 2 } \prec d . } \end{array}
$$

The regularized empirical risk is defined as $\begin{array} { r } { \mathcal { L } _ { v } ( f ; \kappa ) = \mathcal { L } _ { v } ( f ; 0 ) + \frac { \kappa } { d } \| f \| _ { \mathcal H } ^ { 2 } } \end{array}$ . By the optimality of the ridge estimator ${ \widehat f _ { \kappa , v } } ,$ its risk must be upper-bounded by that of the localized bump function. Since we assume $d ^ { - k _ { 0 } + 2 } \lesssim \kappa .$ , this yields

$$
\begin{array} { r } { \mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; \kappa ) \leq \mathcal { L } _ { v } ( f _ { m , v } ^ { \mathrm { l o c } } ; \kappa ) \prec d ^ { - k _ { 0 } + 2 } + \kappa \lesssim \kappa . } \end{array}
$$

Consequently, we obtain both the training loss bound $\mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; 0 ) \prec \kappa$ and the RKHS norm bound $\| \widehat { f } _ { \kappa , v } \| _ { \mathcal { H } } ^ { 2 } \prec d .$

We decompose the estimator into its linear and higher-order components as $\widehat { f } _ { \kappa , v } ( \pmb { x } ) = \langle \widehat { \pmb { \theta } } _ { \kappa , v } , \pmb { x } \rangle +$ $\widehat { g } _ { \kappa , \pmb { v } } ( \pmb { x } )$ , where $\widehat { g } _ { \kappa , v } \in \mathcal { H } _ { \geq 3 }$ . By the norm decomposition in Lemma $9 , \| \widehat { f } _ { \kappa , v } \| _ { \mathcal { H } } ^ { 2 } = d \| \widehat { \pmb { \theta } } _ { \kappa , v } \| _ { 2 } ^ { 2 } + \| \widehat { g } _ { \kappa , v } \| _ { \mathcal { H } } ^ { 2 }$ Thus, the RKHS bound implies $\lVert \widehat { \pmb { \theta } } _ { \kappa , v } \rVert _ { 2 } \prec 1$ and $\| \widehat { g } _ { \kappa , v } \| _ { \mathcal { H } } \prec \sqrt { d }$

Recall the definition of the Hermite coeficients in (51). By convexity and Parseval’s identity,

$$
\left\| \sigma \widehat { \theta } _ { \kappa , v } + v + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \Gamma _ { 1 } \widehat { g } _ { \kappa , v } ( \boldsymbol { x } _ { 0 , j } ) \right\| _ { 2 } ^ { 2 } \leq \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \left\| \sigma \widehat { \theta } _ { \kappa , v } + v + \Gamma _ { 1 } \widehat { g } _ { \kappa , v } ( \boldsymbol { x } _ { 0 , j } ) \right\| _ { 2 } ^ { 2 } \leq \mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; 0 ) \prec \kappa .
$$

By the triangle inequality, Lemma 10, and Proposition 3, we bound the distance of the linear coeficients to the target vector. Since $\kappa \lesssim d ^ { - \epsilon }$ for some $\epsilon > 0$ , for every $c \in ( 0 , ( \epsilon \wedge 1 ) / 2 )$ there exist constants $c ^ { \prime } , C > 0$ such that

$$
\| \sigma \widehat { \pmb { \theta } } _ { \kappa , \pmb { v } } + \pmb { v } \| _ { 2 } \lesssim d ^ { c } \sqrt { \kappa } + d ^ { c } \frac { C _ { \zeta } } { \sqrt { d } } \lesssim d ^ { c - \epsilon / 2 } + C _ { \zeta } d ^ { c - 1 / 2 }\tag{59}
$$

with probability at least $1 - \zeta - e ^ { - d ^ { c ^ { \prime } } }$ for all $d \geq C ,$ , where $C _ { \zeta }$ is defined in (52). We place ourselves on this high-probability event for the remainder of the proof. Furthermore, still by Lemma 10, the non-linear residual vanishes in $L ^ { 2 } ( \pi _ { d } )$ , yielding $\lVert \widehat { g } _ { \kappa , v } \rVert _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } \prec d ^ { - 1 / 2 }$

Finally, we expand the population risk:

$$
\begin{array} { r l } & { \mathcal { R } _ { v } ( \widehat { f } _ { \kappa , v } ) = \mathbb { E } _ { ( \alpha _ { 0 } , z ) } \left[ \big | \langle \widehat { \theta } _ { \kappa , v } , \rho x _ { 0 } + \sigma z \rangle + \widehat { g } _ { \kappa , v } ( \rho x _ { 0 } + \sigma z ) + \langle v , z \rangle \big | ^ { 2 } \right] } \\ & { \quad \quad \quad = \widehat { \theta } _ { \kappa , v } ^ { \mathrm { T } } \Sigma _ { \sigma } \widehat { \theta } _ { \kappa , v } + 2 \sigma \widehat { \theta } _ { \kappa , v } ^ { \mathrm { T } } v + 1 + \big \| \widehat { g } _ { \kappa , v } \big \| _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } } \\ & { \quad \quad \quad \quad \quad + 2 \mathbb { E } _ { ( \alpha _ { 0 } , z ) } \left[ \langle \widehat { \theta } _ { \kappa , v } , \rho x _ { 0 } + \sigma z \rangle \widehat { g } _ { \kappa , v } ( \rho x _ { 0 } + \sigma z ) \right] + 2 \mathbb { E } _ { ( \alpha _ { 0 } , z ) } \left[ \langle v , z \rangle \widehat { g } _ { \kappa , v } ( \rho x _ { 0 } + \sigma z ) \right] . } \end{array}
$$

We treat the terms separately. By (59) and (40), the deviation of the linear component from its target is bounded by

$$
\mathopen { } \mathclose \bgroup \left| \widehat { \theta } _ { \kappa , v } ^ { \mathsf { T } } \Sigma _ { \sigma } \widehat { \theta } _ { \kappa , v } - \frac { 1 } { \sigma ^ { 2 } } v ^ { \mathsf { T } } \Sigma _ { \sigma } v \aftergroup \egroup \right| \vee \left| \widehat { \theta } _ { \kappa , v } ^ { \mathsf { T } } v + \frac { 1 } { \sigma } \right| \lesssim d ^ { c - \epsilon / 2 } + C _ { \zeta } d ^ { c - 1 / 2 } .
$$

An application of the Cauchy-Schwarz inequality allows us to bound the cross-terms using the linear coeficients and the non-linear residual directly:

$$
\begin{array} { r l } & { \Big | \mathbb { E } _ { ( \alpha _ { 0 } , \varepsilon ) } \left[ \langle \widehat { \theta } _ { \kappa , v } , \rho x _ { 0 } + \sigma z \rangle \widehat { g } _ { \kappa , v } ( \rho x _ { 0 } + \sigma z ) \right] \Big | \leq \sqrt { \mathbb { E } _ { ( x _ { 0 } , z ) } \left[ \langle \widehat { \theta } _ { \kappa , v } , \rho x _ { 0 } + \sigma z \rangle ^ { 2 } \right] } \sqrt { \mathbb { E } _ { ( x _ { 0 } , z ) } \left[ \widehat { g } _ { \kappa , v } ( \rho x _ { 0 } + \sigma z ) ^ { 2 } \right] } } \\ & { \qquad \lesssim \sqrt { 1 + d ^ { c - \epsilon / 2 } + C _ { \zeta } d ^ { c - 1 / 2 } } d ^ { c / 2 - 1 / 4 } . } \end{array}
$$

A similar bound holds for the second cross-term involving $\langle { \pmb v } , z \rangle$ . Combining all the components, and using $\begin{array} { r } { \Sigma _ { \sigma } = \rho ^ { 2 } \Sigma + \sigma ^ { 2 } \mathbf { I } _ { d } } \end{array}$ alongside $\| \pmb { v } \| _ { 2 } ^ { 2 } = 1$ , we conclude that

$$
\left| { \mathcal { R } } _ { v } ( \widehat { f } _ { k , v } ) - \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } v ^ { \mathsf { T } } \Sigma v \right| \lesssim d ^ { c - \epsilon / 2 } + C _ { \zeta } d ^ { c - 1 / 2 } + \sqrt { 1 + d ^ { c - \epsilon / 2 } + C _ { \zeta } d ^ { c - 1 / 2 } } d ^ { c / 2 - 1 / 4 } .
$$

We recover the final statement of the theorem by averaging the expected risk over an orthonormal basis $\{ \pmb { v } _ { i } \} _ { i = 1 } ^ { d } \ \mathrm { o f } \ \mathbb { R } ^ { d }$ □

## E Proofs of the gradient flow results

This section is devoted to proving the asymptotic risk equivalents stated in Theorems 1 and 2. The proofs are similar to those in Appendix D, and we similarly suppose that the difusion time $t > 0$ is fixed throughout this appendix.

## E.1 Proof of Theorem 1

The argument mirrors the proof of Theorem 6 for ridge regression. Analogously, we define the linearized training and test errors as

$$
\mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } : = \left\| \exp ( - \mathrm { t } \widehat { K } _ { \mathrm { e f f } } ) z _ { v } \right\| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } , \qquad \mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } : = \left\| \mathcal { K } _ { \mathrm { e f f } } \int _ { 0 } ^ { \mathrm { t } } \exp ( - \mathfrak { s } \widehat { K } _ { \mathrm { e f f } } ) z _ { v } \mathrm { d } \mathsf { s } - z _ { v } \right\| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } .
$$

We then have the following linearization result, which is similar to Lemma 15.

Lemma 18. Suppose that Assumptions 1, 2 and 3 hold. Then, there exists $c > 0$ such that for every $\pmb { v } \in \mathbb { R } ^ { d }$ with $\| \pmb { v } \| _ { 2 } = 1$ 2

$$
| \mathcal { L } _ { v } ( \widehat { f } _ { \mathrm { t } , v } ; 0 ) - \mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } | \prec \frac { \mathrm { t } } { d ^ { \frac { 3 } { 2 } } } + e ^ { - d ^ { c } } , \qquad | \mathcal { R } _ { v } ( \widehat { f } _ { \mathrm { t } , v } ) - \mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } | \prec \frac { \mathrm { t } } { d ^ { \frac { 3 } { 2 } } } \left( 1 + \frac { \mathrm { t } } { d } \right) ^ { 2 }
$$

uniformly over $\mathsf { t } \geq 0$

Proof. Note that $\| \exp ( - \mathrm { t } \widehat { \mathcal { K } } ) \| _ { \mathrm { o p } } \vee \| \exp ( - \mathrm { t } \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) \| _ { \mathrm { o p } } \le 1$ since $\widehat { \kappa }$ and $\widehat { \kappa } _ { \mathrm { e f f } }$ are positive semidefinite operators. By Proposition 2, there exists a constant $c > 0$ such that

$$
\| \widehat { f } _ { v } ^ { \star } - z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq e ^ { - d ^ { c } }
$$

with very high probability, which implies that replacing the target noise with the linear signal introduces an exponentially small error, i.e.,

$$
\begin{array} { r } { \left| \mathcal { L } _ { v } ( \widehat { f } _ { \mathsf { t } , v } ; 0 ) - \| \exp ( - \mathsf { t } \widehat { \mathcal { K } } ) z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \right| \lesssim e ^ { - d ^ { c } } . } \end{array}
$$

Next, using the scalar inequality $| x ^ { 2 } - y ^ { 2 } | \leq | x - y | ( | x | + | y | )$ alongside the reverse triangle inequality yields

$$
\begin{array} { r } { \Big \lvert \| \exp ( - \mathrm { t } \widehat { K } ) z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } - \mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } \Big \rvert \leq 2 \| \exp ( - \mathrm { t } \widehat { K } ) - \exp ( - \mathrm { t } \widehat { K } _ { \mathrm { e f f } } ) \| _ { \mathrm { o p } } , } \end{array}
$$

where we used the fact that $\| \exp ( - \mathrm { t } \widehat { \mathcal { K } } ) \| _ { \mathrm { o p } } \vee \| \exp ( - \mathrm { t } \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) \| _ { \mathrm { o p } } \vee \| z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq 1$ . To bound the operator diference, define the operator-valued mapping ${ \mathsf { \pmb { \mathsf { s } } } } \mapsto \Phi ( { \mathsf { \pmb { \mathsf { s } } } } )$ on the interval [0, t] as

$$
\begin{array} { r } { \Phi ( \mathsf { s } ) : = \exp ( - ( \mathsf { t } - \mathsf { s } ) \widehat { \mathcal { K } } ) \exp ( - \mathsf { s } \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) . } \end{array}
$$

Diferentiating with respect to s via the operator product rule yields

$$
\frac { \mathrm { d } } { \mathrm { d } s } \Phi \big ( { \mathsf { s } } \big ) = \mathrm { e x p } \big ( - ( { \mathsf { t } } - { \mathsf { s } } ) \widehat { K } \big ) \widehat { K } \mathrm { e x p } \big ( - { \mathsf { s } } \widehat { K } _ { \mathrm { e f f } } \big ) - \mathrm { e x p } \big ( - ( { \mathsf { t } } - { \mathsf { s } } ) \widehat { K } \big ) \widehat { K } _ { \mathrm { e f f } } \mathrm { e x p } \big ( - { \mathsf { s } } \widehat { K } _ { \mathrm { e f f } } \big ) .
$$

The space of bounded linear operators on $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ , equipped with the uniform operator norm, is a Banach space. By the Fundamental Theorem of Calculus for Bochner integrals, integrating over s gives $\Phi ( { \sf t } ) - \Phi ( 0 )$ , which evaluates to

$$
\mathrm { e x p } ( - \mathfrak { t } \widehat { K } _ { \mathrm { e f f } } ) - \mathrm { e x p } ( - \mathfrak { t } \widehat { K } ) = \int _ { 0 } ^ { \mathfrak { t } } \mathrm { e x p } ( - ( \mathfrak { t } - \mathfrak { s } ) \widehat { K } ) ( \widehat { K } - \widehat { K } _ { \mathrm { e f f } } ) \mathrm { e x p } ( - \mathfrak { s } \widehat { K } _ { \mathrm { e f f } } ) \mathrm { d } \mathfrak { s } .
$$

Taking the operator norm of both sides and using submultiplicativity, we bound the right-hand side by $\mathrm { t } \| \widehat { \mathcal { K } } - \widehat { \mathcal { K } } _ { \mathrm { e f f } } \| _ { \mathrm { o p } }$ . Combined with Proposition 1, it follows that

$$
\Big | \| \exp ( - \mathrm { t } \widehat { K } ) z _ { v } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } - \mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } \Big | \prec \mathrm { t } \| \widehat { K } - \widehat { K } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \prec \frac { \mathrm { t } } { d ^ { \frac { 3 } { 2 } } } .
$$

For the test error, note that

$$
\mathcal { K } \widehat { K } ^ { \dagger } ( \mathrm { i d } _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } - \exp ( - \mathrm { t } \widehat { K } ) ) = \mathcal { K } \int _ { 0 } ^ { \mathsf { t } } \exp ( - \mathsf { s } \widehat { K } ) \mathrm { d } \mathsf { s }
$$

since $\kappa$ annihilates the null space of $\widehat { \kappa }$ . We apply Lemma 14 along with the integration over time to bound the operator norm:

$$
\left\| \mathcal { K } \int _ { 0 } ^ { \mathsf { t } } \exp ( - \mathsf { s } \widehat { \mathcal { K } } ) \mathrm { d } \mathsf { s } \right\| _ { \mathrm { o p } } \leq \mathsf { t } \| \mathcal { K } \| _ { \mathrm { o p } } \prec \frac { \mathsf { t } } { d } ,
$$

and $\mathrm { a }$ similar bound holds for the efective kernel operators. Consequently, swapping $\widehat { f } _ { v } ^ { \star }$ for $z _ { v }$ yields

$$
\left| \mathcal { R } _ { v } ( \widehat { f } _ { \mathsf { t } , v } ) - \left\| K \int _ { 0 } ^ { \mathsf { t } } \exp ( - \mathsf { s } \widehat { \mathcal { K } } ) z _ { v } \mathrm { d } \mathsf { s } - z _ { v } \right\| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } \right| \prec \frac { \mathsf { t } } { d } e ^ { - d ^ { c } } \left( 1 + \frac { \mathsf { t } } { d } \right) .
$$

By the triangle inequality,

$$
\begin{array} { r l } & { \left\| \mathcal { K } \displaystyle \int _ { 0 } ^ { \mathsf { t } } \exp ( - \mathsf { s } \widehat { \mathcal { K } } ) \mathrm { d } \mathsf { s } - \mathcal { K } _ { \mathrm { e f f } } \displaystyle \int _ { 0 } ^ { \mathsf { t } } \exp ( - \mathsf { s } \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) \mathrm { d } \mathsf { s } \right\| _ { \mathrm { o p } } } \\ & { \qquad \leq \| \mathcal { K } \| _ { \mathrm { o p } } \left\| \displaystyle \int _ { 0 } ^ { \mathsf { t } } \left( \exp ( - \mathsf { s } \widehat { \mathcal { K } } ) - \exp ( - \mathsf { s } \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) \right) \mathrm { d } \mathsf { s } \right\| _ { \mathrm { o p } } + \| \mathcal { K } - \mathcal { K } _ { \mathrm { e f f } } \| _ { \mathrm { o p } } \left\| \displaystyle \int _ { 0 } ^ { \mathsf { t } } \exp ( - \mathsf { s } \widehat { \mathcal { K } } _ { \mathrm { e f f } } ) \mathrm { d } \mathsf { s } \right\| _ { \mathrm { o p } } } \\ & { \qquad \prec \displaystyle \frac { \mathsf { t } ^ { 2 } } { d ^ { \frac { 5 } { 2 } } } + \frac { \mathsf { t } } { d ^ { \frac { 3 } { 2 } } } . } \end{array}
$$

The last line follows from Proposition 1, Lemma 14 and the exponential diference bound derived above for the training error. We obtain the final approximation error

$$
\left| \left\| \kappa \int _ { 0 } ^ { \ t } \exp ( - \mathfrak { s } \widehat { \mathcal { K } } ) z _ { v } \mathrm { d } \mathsf { s } - z _ { v } \right\| _ { L ^ { 2 } ( \pi _ { d } ) } ^ { 2 } - \mathcal { R } _ { \mathrm { i n } , v } ^ { ( \mathfrak { t } ) } \right| \prec \left( \frac { \mathfrak { t } } { d ^ { \frac { 3 } { 2 } } } + \frac { \mathfrak { t } ^ { 2 } } { d ^ { \frac { 5 } { 2 } } } \right) \left( 1 + \frac { \mathfrak { t } } { d } \right) .
$$

This concludes the proof.

Having approximated the training and test errors by their linearized counterparts, we now derive explicit finite-dimensional representations of the linearized training and test errors. This reduction will allow us to compute the asymptotic limits via random matrix theory. Recall the definition of the linear functionals $L _ { w _ { x } , w _ { z } }$ for $\boldsymbol { w } _ { x } , \boldsymbol { w } _ { z } \in \mathbb { R } ^ { d }$ introduced in the paragraph preceding Lemma 16.

Lemma 19 (Exact spectral representation of the linearized errors). Let $\mathfrak { t } \geq 0$ . Then,

$$
{ \mathcal L } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } = \frac { 1 } { d } \mathrm { T r } \left( \ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( \widehat { \Sigma } ) \right) , \qquad { \mathcal R } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } = \frac { 1 } { d } \mathrm { T r } \left( \pmb { \Sigma } _ { \sigma } \psi ( \widehat { \Sigma } , \mathrm { t } ) ^ { 2 } \right) + \frac { 2 \sigma } { d } \mathrm { T r } \left( \psi ( \widehat { \Sigma } , \mathrm { t } ) \right) + 1 ,
$$

where we recall the definitions of $\psi$ in (28) and $\ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) }$ in (29).

Proof. Let $\pmb { v } \in \mathbb { S } ^ { d - 1 }$ . By Lemma 16, the efective operators preserve the subspace of linear functions and act on their coeficient vectors through the matrices

$$
\widehat { \pmb { K } } : = \left[ \frac { \rho ^ { 2 } } { d } \widehat { \pmb { \Sigma } } + \delta _ { n } \mathbf { I } _ { d } \quad \frac { \rho \sigma } { d } \mathbf { I } _ { d } \right] , \qquad \pmb { K } : = \left[ \frac { \rho ^ { 2 } } { d } \widehat { \pmb { \Sigma } } \quad \frac { \rho \sigma } { d } \mathbf { I } _ { d } \right] .
$$

More precisely, the power-series representation of the exponential gives

$$
e ^ { - \mathrm { t } \widehat { K } _ { \mathrm { e f f } } } L _ { w } = L _ { e ^ { - \mathrm { t } \widehat { K } _ { w } } } , \qquad K _ { \mathrm { e f f } } \int _ { 0 } ^ { \mathrm { t } } e ^ { - \mathrm { s } \widehat { K } _ { \mathrm { e f f } } } L _ { w } \mathrm { d } \mathbf { s } = L _ { K \int _ { 0 } ^ { \mathrm { t } } e ^ { - \mathrm { s } \widehat { K } _ { w } } \mathrm { d } \mathbf { s } } .\tag{60}
$$

Note that the matrix $\widehat { \pmb { K } }$ is not symmetric in the raw coeficient coordinates. Define

$$
\widehat { \pmb { J } } : = \left[ \begin{array} { c c } { \widehat { \pmb { \Sigma } } ^ { 1 / 2 } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { I } _ { d } } \end{array} \right] , \qquad \widehat { \pmb { Q } } : = \left[ \begin{array} { c c } { \displaystyle \frac { \rho ^ { 2 } } { d } \widehat { \pmb { \Sigma } } + \delta _ { n } \mathbf { I } _ { d } } &  \displaystyle \frac { \rho \sigma } { d } \widehat { \pmb { \Sigma } } ^ { 1 / 2 } \right] , \end{array}
$$

and note via direct multiplication that ${ \widehat { J K } } = { \widehat { Q } } { \widehat { J } }$ Applying this identity term by term in the exponential series yields

$$
\widehat { J } e ^ { - \mathrm { t } \widehat { K } } = e ^ { - \mathrm { t } \widehat { Q } } \widehat { J } .\tag{61}
$$

In an eigenbasis of $\widehat { \Sigma }$ , the matrix $\widehat { Q }$ decomposes into the two-dimensional blocks $Q ( x )$ defined in (28).

Define the block embedding

$$
\mathbf { { \cal E } } _ { 2 } : = \left[ \mathbf { { \boldsymbol 0 } } _ { } \right] .
$$

Since $z _ { v } = L _ { - E _ { 2 } v }$ , (60) and (61) give

$$
e ^ { - \mathrm { t } \widehat { K } _ { \mathrm { e f f } } } \mathsf { z } _ { v } = L _ { \widehat { w } _ { \mathrm { t } } } , \qquad \widehat { J } \widehat { w } _ { \mathrm { t } } = - e ^ { - \mathrm { t } \widehat { Q } } E _ { 2 } v .
$$

For every $\pmb { w } = ( \pmb { w } _ { \pmb { x } } , \pmb { w } _ { z } ) \in \mathbb { R } ^ { 2 d }$ , independence and centering of the Gaussian noise imply

$$
\| L _ { \pmb { w } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } = \pmb { w } _ { x } ^ { \top } \widehat { \Sigma } \pmb { w } _ { x } + \| \pmb { w } _ { z } \| _ { 2 } ^ { 2 } = \| \widehat { \pmb { J } } \pmb { w } \| _ { 2 } ^ { 2 } .
$$

Therefore, using the symmetry of $\widehat { Q }$ ,

$$
\mathcal { L } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } = \left. e ^ { - \mathrm { t } \widehat { Q } } E _ { 2 } v \right. _ { 2 } ^ { 2 } = v ^ { \mathsf { T } } E _ { 2 } ^ { \mathsf { T } } e ^ { - 2 \mathrm { t } \widehat { Q } } E _ { 2 } v = v ^ { \mathsf { T } } \ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( \widehat { \Sigma } ) v .
$$

For the test error, we may factor the matrix K as

$$
\boldsymbol { K } = \left[ \rho \mathbf { I } _ { d } \right] \left[ \frac { \rho } { d } \widehat { \mathbf { \Sigma } } \boldsymbol { \widehat { \mathbf { Z } } } ^ { \prime } \right. \left. \frac { \sigma } { d } \mathbf { I } _ { d } \right] = \frac { 1 } { \sigma } \left[ \rho \mathbf { I } _ { d } \right] \boldsymbol { E } _ { 2 } ^ { \intercal } \boldsymbol { \widehat { Q } } \boldsymbol { \widehat { J } }
$$

Hence, using (61) and $\widehat { J } E _ { 2 } = E _ { 2 }$

$$
\begin{array} { r } { K \displaystyle \int _ { 0 } ^ { \mathsf { t } } e ^ { - { \widehat { s } \widehat { K } } } ( - E _ { 2 } v ) \mathrm { d } s = - \frac { 1 } { \sigma } \left[ \rho \mathbf { I } _ { d } \right] E _ { 2 } ^ { \mathsf { T } } \widehat { Q } \displaystyle \int _ { 0 } ^ { \mathsf { t } } e ^ { - s \widehat { Q } } E _ { 2 } v \mathrm { d } s = - \frac { 1 } { \sigma } \left[ \rho \mathbf { I } _ { d } \right] E _ { 2 } ^ { \mathsf { T } } \left( \mathbf { I } _ { 2 d } - e ^ { - { \mathsf { t } \widehat { Q } } } \right) E _ { 2 } v } \\ { = \frac { \left[ \rho \mathbf { I } _ { d } \right] } { \left[ \sigma \mathbf { I } _ { d } \right] } \psi ( \widehat { \Sigma } , \mathsf { t } ) v . \qquad } \end{array}
$$

Evaluating this linear function on an independent test pair $( \pmb { x } , \pmb { z } ) \sim \pi _ { d }$ gives

$$
\mathcal { R } _ { \mathrm { l i n } , v } ^ { ( \mathrm { t } ) } = v ^ { \mathsf { T } } \psi ( \widehat { \Sigma } , \mathrm { t } ) \Sigma _ { \sigma } \psi ( \widehat { \Sigma } , \mathrm { t } ) v + 2 \sigma v ^ { \mathsf { T } } \psi ( \widehat { \Sigma } , \mathrm { t } ) v + 1 .
$$

Averaging over an orthonormal basis of $\mathbb { R } ^ { d }$ and using cyclicity of the trace yields the result.

Remark E.1 (Explicit form of the spectral profiles). We can express the spectral profiles $\ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) }$ and ψ in a more explicit closed form. Let

$$
m ( x ) : = \frac { \rho ^ { 2 } x } { d } + \delta _ { n } + \frac { \sigma ^ { 2 } } { d } , \qquad \gamma _ { \pm } ( x ) : = \frac 1 2 \left( m ( x ) \pm \sqrt { m ( x ) ^ { 2 } - \frac { 4 \sigma ^ { 2 } \delta _ { n } } { d } } \right) , \qquad I _ { \pm } ( x , \mathbf { t } ) : = \int _ { 0 } ^ { \mathbf { t } } e ^ { - s \gamma _ { \pm } ( x ) } \mathrm { d } s ,
$$

Since $\mathrm { T r } Q ( x ) = m ( x )$ and det $Q ( x ) = \sigma ^ { 2 } \delta _ { n } / d .$ , the numbers $\gamma _ { + } ( x ) \geq \gamma _ { - } ( x )$ are the two eigenvalues of $Q ( x )$ . Whenever $\gamma _ { + } ( x ) > \gamma _ { - } ( x )$ , Sylvester’s formula gives

$$
e ^ { - { \ t Q ( x ) } } = \frac { e ^ { - { \ t r } \gamma _ { + } ( x ) } \big ( Q ( x ) - \gamma _ { - } ( x ) \mathbf { I } _ { 2 } \big ) - e ^ { - { \ t } \gamma _ { - } ( x ) } \big ( Q ( x ) - \gamma _ { + } ( x ) \mathbf { I } _ { 2 } \big ) } { \gamma _ { + } ( x ) - \gamma _ { - } ( x ) } .
$$

Applying this identity to $e _ { 2 }$ and taking the squared Euclidean norm yields

$$
\ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( x ) = \frac { \rho ^ { 2 } \sigma ^ { 2 } x } { d ^ { 2 } } \left( \frac { e ^ { - \mathrm { t } \gamma _ { - } ( x ) } - e ^ { - \mathrm { t } \gamma _ { + } ( x ) } } { \gamma _ { + } ( x ) - \gamma _ { - } ( x ) } \right) ^ { 2 } + \left( \frac { \left( \frac { \sigma ^ { 2 } } { d } - \gamma _ { + } ( x ) \right) e ^ { - \mathrm { t } \gamma _ { - } ( x ) } - \left( \frac { \sigma ^ { 2 } } { d } - \gamma _ { - } ( x ) \right) e ^ { - \mathrm { t } \gamma _ { + } ( x ) } } { \gamma _ { + } ( x ) - \gamma _ { - } ( x ) } \right) ^ { 2 } ,
$$

Similarly, using $\gamma _ { + } ( x ) + \gamma _ { - } ( x ) = m ( x )$ and $\gamma _ { + } ( x ) \gamma _ { - } ( x ) = \sigma ^ { 2 } \delta _ { n } / d .$ , the $( 2 , 2 )$ entry of the same identity gives

$$
\psi ( x , \mathbf { t } ) = \frac { \sigma } { d } \frac { \left( \gamma _ { - } ( x ) - \delta _ { n } \right) I _ { - } ( x , \mathbf { t } ) - \left( \gamma _ { + } ( x ) - \delta _ { n } \right) I _ { + } ( x , \mathbf { t } ) } { \gamma _ { + } ( x ) - \gamma _ { - } ( x ) } .
$$

The apparent singularities in these expressions are removable. If $\gamma _ { + } ( x ) = \gamma _ { - } ( x ) = \gamma ( x )$ , symmetry forces $Q ( x ) = \gamma ( x ) \mathbf { I } _ { 2 }$ , and the continuous extensions are

$$
\ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( x ) = e ^ { - 2 { \mathrm { t } } \gamma ( x ) } , \qquad \psi ( x , \mathrm { t } ) = - \frac { 1 - e ^ { - { \mathrm { t } } \gamma ( x ) } } { \sigma } .
$$

Remark E.2 (Spectral form of the linearized denoiser). The proof of Lemma 19 also shows that the linearized denoiser admits the explicit spectral representation

$$
\widehat { \pmb { f } } _ { \mathrm { t } } ^ { \mathrm { l i n } } ( \pmb { u } ) = \psi ( \widehat { \pmb { \Sigma } } , \mathrm { t } ) \pmb { u } .
$$

Lemma 20. Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 3 hold. Then, for every $\epsilon , D > 0$ , there exists a constant $C > 0$ such that

$$
\operatorname* { s u p } _ { \ t \geq 0 } \Big ( | \mathcal { L } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } | \vee | \mathcal { R } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } | \Big ) \lesssim d ^ { \epsilon - \frac { 1 } { 2 } }
$$

with probability at least $1 - C d ^ { - D }$

Proof. Set $q : = d \delta _ { n }$ , which satisfies $0 \leq q \lesssim 1$ by Remark $\mathrm { C } . 2 , n \asymp d ,$ and the bounds on $\tau _ { \Sigma }$ . The equality characterization in the same remark, together with the standing bounds on $\rho ^ { 2 } \tau _ { \Sigma }$ and $d / n$ gives the following dichotomy: either $q = 0$ , or there exists a deterministic constant $c _ { q } > 0$ such that $q \geq c _ { q }$ uniformly in d.

The symmetric matrix $Q ( x )$ is convenient on the nonnegative real spectrum, but its square-root dependence on $x$ is not convenient for holomorphic functional calculus. For $z \in \mathbb { C }$ , and $x > 0$ , let

$$
\pmb { { \cal B } } ( z ) : = \left[ \begin{array} { l l } { \rho ^ { 2 } z + q } & { \rho \sigma } \\ { \rho \sigma z } & { \sigma ^ { 2 } } \end{array} \right] , \qquad \pmb { { \cal D } } ( x ) : = \left[ \begin{array} { l l } { \sqrt { x } } & { 0 } \\ { 0 } & { 1 } \end{array} \right] ,
$$

such that $D ( x ) B ( x ) = d Q ( x ) D ( x )$ . Hence, term by term in the exponential series,

$$
D ( x ) e ^ { - u B ( x ) } = e ^ { - d u { \cal Q } ( x ) } D ( x ) , \qquad u \geq 0 .
$$

Since $D ( x ) e _ { 2 } = e _ { 2 }$ , the definitions in Lemma 19 imply

$$
\ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( x ) = \left[ 0 \quad 1 \right] e ^ { - \frac { \mathrm { t } } { d } B ( x ) ^ { \top } } \left[ \begin{array} { l l } { x } & { 0 } \\ { 0 } & { 1 } \end{array} \right] e ^ { - \frac { \mathrm { t } } { d } B ( x ) } \left[ \begin{array} { l } { 0 } \\ { 1 } \end{array} \right] ,\tag{62}
$$

$$
\psi ( x , { \bf t } ) = - \left[ \rho x \quad \sigma \right] \int _ { 0 } ^ { { \bf t } / d } e ^ { - u { \cal B } ( x ) } \mathrm { d } u \left[ \begin{array} { l } { { 0 } } \\ { { 1 } } \end{array} \right] .\tag{63}
$$

For the second identity, we additionally used

$$
\mathbf { I } _ { 2 } - e ^ { - \mathsf { t } Q ( x ) } = \pmb { Q } ( x ) \int _ { 0 } ^ { \mathsf { t } } e ^ { - \mathsf { s } \pmb { Q } ( x ) } \mathrm { d } \mathsf { s } , \qquad e _ { 2 } ^ { \mathsf { T } } \pmb { Q } ( x ) \pmb { D } ( x ) = \frac { \sigma } { d } \left[ \rho x \sigma \right] ,
$$

followed by the change of variables $u = \mathsf { s } / d$ . Because $B ( z )$ is afine in $z ,$ its exponential power series converges locally uniformly in z, uniformly for u in compact intervals. Thus the right-hand sides of (62) and (63) are entire in z and provide the entire analytic continuations of the two profiles defined by $Q ( x )$ for $x \geq 0$

By (44), there exists a deterministic $\Lambda < \infty$ such that the limiting spectral support is contained in $[ 0 , \Lambda ]$ and, with very high probability, so is $\mathrm { s p e c } ( \widehat { \Sigma } )$ . Fix a rectangular contour Γ enclosing this interval, with horizontal edges Imz = ±1 and vertical edges $\Re z = - \eta$ and $\Re z = \Lambda + \eta$ , where $\eta > 0$ is small enough that

$$
\operatorname* { i n f } _ { z \in \Gamma } \Re ( \rho ^ { 2 } z + \sigma ^ { 2 } ) \ge c > 0
$$

for some constant $c > 0$ . The characteristic roots of $B ( z )$ satisfy

$$
\lambda _ { + } ( z ) + \lambda _ { - } ( z ) = \rho ^ { 2 } z + \sigma ^ { 2 } + q , \qquad \lambda _ { + } ( z ) \lambda _ { - } ( z ) = \sigma ^ { 2 } q .
$$

Suppose that $q > 0$ . Assume, by contradiction, that $\Re \lambda _ { + } ( z ) \leq 0$ . Then

$$
\Re \lambda _ { - } ( z ) = \sigma ^ { 2 } q { \frac { \Re \lambda _ { + } ( z ) } { | \lambda _ { + } ( z ) | ^ { 2 } } } \leq 0 ,
$$

contradicting $\Re ( \lambda _ { + } ( z ) + \lambda _ { - } ( z ) ) > 0$ . Hence both roots have positive real parts. In the present case, the dichotomy given by Remark C.2 gives $q \geq c _ { q } > 0 ;$ ; since also $q \lesssim 1$ and Γ is compact, there exists $c _ { 0 } > 0$ such that $\Re \lambda _ { + } ( z ) \wedge \Re \lambda _ { - } ( z ) \geq c _ { 0 }$ for every $z \in \Gamma$ . Let $u \geq 0$ and write

$$
e ^ { - u B ( z ) } = e ^ { - u \lambda _ { - } ( z ) } { \bf I } _ { 2 } + \frac { e ^ { - u \lambda _ { + } ( z ) } - e ^ { - u \lambda _ { - } ( z ) } } { \lambda _ { + } ( z ) - \lambda _ { - } ( z ) } \big ( { \pmb B } ( z ) - \lambda _ { - } ( z ) { \bf I } _ { 2 } \big ) ,
$$

where, when $\lambda _ { + } = \lambda _ { - } = \lambda$ , the quotient is defined by continuity as − $- u e ^ { - u \lambda }$ . By the fundamental theorem of calculus,

$$
\left| \frac { e ^ { - u \lambda _ { + } ( z ) } - e ^ { - u \lambda _ { - } ( z ) } } { \lambda _ { + } ( z ) - \lambda _ { - } ( z ) } \right| \le u \int _ { 0 } ^ { 1 } e ^ { - u \Re \left\{ ( 1 - \theta ) \lambda _ { - } ( z ) + \theta \lambda _ { + } ( z ) \right\} } \mathrm { d } \theta \le u e ^ { - c _ { 0 } u } \lesssim 1 .
$$

Since $\boldsymbol { B } ( z )$ and its characteristic roots are uniformly bounded, it follows that

$$
\operatorname* { s u p } _ { u \geq 0 } \operatorname* { s u p } _ { z \in \Gamma } \| e ^ { - u B ( z ) } \| _ { \mathrm { o p } } \lesssim 1 .
$$

Now suppose that $q = 0$ . Then, the roots are 0 and $\rho ^ { 2 } z + \sigma ^ { 2 }$ , which has positive real part for $z \in \Gamma$ . Moreover,

$$
B ( z ) = \left[ \rho \right] \left[ \rho z \quad \sigma \right] , \qquad B ( z ) ^ { 2 } = ( \rho ^ { 2 } z + \sigma ^ { 2 } ) B ( z ) .
$$

Consequently,

$$
e ^ { - u B ( z ) } = { \bf I } _ { 2 } + \frac { e ^ { - u ( \rho ^ { 2 } z + \sigma ^ { 2 } ) } - 1 } { \rho ^ { 2 } z + \sigma ^ { 2 } } { \cal B } ( z ) .
$$

Since $B ( z )$ is bounded on Γ, $| \rho ^ { 2 } z + \sigma ^ { 2 } | \geq c .$ and $| e ^ { - u ( \rho ^ { 2 } z + \sigma ^ { 2 } ) } | \le 1$ 2

$$
\operatorname* { s u p } _ { u \geq 0 } \operatorname* { s u p } _ { z \in \Gamma } \| e ^ { - u B ( z ) } \| _ { \mathrm { o p } } \lesssim 1 .
$$

Combining the two cases, we conclude that su $\begin{array} { r } { { \mathrm { p } } _ { u \geq 0 } { \mathrm { s u p } } _ { z \in \Gamma } \| e ^ { - u B ( z ) } \| _ { \mathrm { o p } } \lesssim 1 } \end{array}$ , and the same bound holds with $B ( z )$ replaced by $B ( z ) ^ { \top }$ . Since $\mathrm { d i a g } ( z , 1 )$ is uniformly bounded on Γ, (62) gives

$$
\operatorname* { s u p } _ { \ t \geq 0 } \operatorname* { s u p } _ { z \in \Gamma } | \ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( z ) | \lesssim 1 .
$$

It remains to bound the time integral in (63). Let $\pmb { E } _ { 1 1 } : = \mathrm { d i a g } ( 1 , 0 )$ . For $q > 0$

$$
( { \pmb B } ( z ) - q { \pmb E } _ { 1 1 } ) \int _ { 0 } ^ { { \pmb \imath } / d } e ^ { - u { \pmb B } ( z ) } { \mathrm d } u = \left( { \bf I } _ { 2 } - q { \pmb E } _ { 1 1 } { \pmb B } ( z ) ^ { - 1 } \right) \left( { \bf I } _ { 2 } - e ^ { - { \frac { { \bf t } } { d } } { \pmb B } ( z ) } \right) ,
$$

where

$$
q { \cal E } _ { 1 1 } B ( z ) ^ { - 1 } = \left[ { 1 - { \rho / \sigma } } \atop { 0 } \right] .
$$

For $q = 0$

$$
B ( z ) \int _ { 0 } ^ {  t / d } e ^ { - u B ( z ) } \mathrm { d } u = \mathbf { I } _ { 2 } - e ^ { - \frac { \mathrm { t } } { d } B ( z ) } .
$$

In both cases,

$$
{ \pmb B } ( z ) - q { \pmb E } _ { 1 1 } = \left[ \rho \right] \left[ \rho z \quad \sigma \right] , \qquad \left\| \left[ \rho \right] \right\| _ { 2 } = 1 .
$$

It follows from (63) and $\rho ^ { 2 } + \sigma ^ { 2 } = 1$ that

$$
| \psi ( z , \mathbf { t } ) | = \left\| \left( \boldsymbol { B } ( z ) - q E _ { 1 1 } \right) \int _ { 0 } ^ { \mathbf { t } / d } e ^ { - u \boldsymbol { B } ( z ) } \mathrm { d } u \left[ 1 \right] \right\| _ { 2 } .
$$

The preceding identities and the semigroup bound therefore imply

$$
\operatorname* { s u p } _ { \ t \geq 0 } \operatorname* { s u p } _ { z \in \Gamma } | \psi ( z , \ t ) | \lesssim 1 .\tag{64}
$$

Lemma 19 gives directly

$$
\mathcal { L } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } = \frac { 1 } { d } \mathrm { T r } \left( \ell _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } ( \widehat { \Sigma } ) \right) ,
$$

and

$$
\mathcal { R } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } = \frac { 1 } { d } \mathrm { T r } \Big ( \Sigma _ { \sigma } \psi ( \widehat { \Sigma } , \mathrm { t } ) ^ { 2 } \Big ) + \frac { 2 \sigma } { d } \mathrm { T r } \Big ( \psi ( \widehat { \Sigma } , \mathrm { t } ) \Big ) + 1 .
$$

Holomorphic functional calculus on Γ and (27) therefore give

$$
\begin{array} { r l } & { \left| \mathcal { L } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } \right| \lesssim \underset { z \in \Gamma } { \operatorname* { s u p } } \left| \frac { 1 } { d } \mathrm { T r } \Big ( \widehat { R } ( z ) - M ( z ) \Big ) \right| , } \\ & { \left| \mathcal { R } _ { \mathrm { l i n } } ^ { ( \mathrm { t } ) } - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } \right| \lesssim \underset { z \in \Gamma } { \operatorname* { s u p } } \left| \frac { 1 } { d } \mathrm { T r } \Big ( \Sigma _ { \sigma } ( \widehat { R } ( z ) - M ( z ) ) \Big ) \right| + \underset { z \in \Gamma } { \operatorname* { s u p } } \left| \frac { 1 } { d } \mathrm { T r } \Big ( \widehat { R } ( z ) - M ( z ) \Big ) \right| . } \end{array}
$$

The contour Γ lies a fixed positive distance from the limiting spectral support, satisfies inf $z \in \Gamma \left| z \right| >$ 0, has bounded real part, and obeys $| \mathrm { I m } [ z ] | \le 1$ . Proposition $5 ,$ applied with $A = \mathbf { I } _ { d } / d$ and $\pmb { A } = \pmb { \Sigma } _ { \sigma } / \mathrm { T r } ( \pmb { \Sigma } _ { \sigma } )$ , now proves the claim, since $\mathrm { T r } ( \Sigma _ { \sigma } ) / d \asymp 1$ . Both the symbol bounds and the local-law event are independent of t, so the estimates hold simultaneously for every $\mathfrak { t } \geq 0$ □

We can now complete the proof of Theorem 1 by combining the above results.

Proof of Theorem 1. Let $\{ \pmb { v } _ { j } \} _ { j = 1 } ^ { d }$ be an orthonormal basis of $\mathbb { R } ^ { d }$ . Combining Lemma 18 and Lemma 20, we conclude that for every $\epsilon , D > 0$ there exists a constant $C > 0$ such that

$$
| \mathcal { L } ( \widehat { f } _ { \dagger } ; 0 ) - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \mathsf { t } ) } | \leq \frac { 1 } { d } \sum _ { j = 1 } ^ { d } | \mathcal { L } _ { v _ { j } } ( \widehat { f } _ { \mathsf { t } , v _ { j } } ; 0 ) - \mathcal { L } _ { \mathrm { l i n } , v _ { j } } ^ { ( \mathsf { t } ) } | + | \mathcal { L } _ { \mathrm { l i n } } ^ { ( \mathsf { t } ) } - \mathcal { L } _ { \mathrm { d e t } } ^ { ( \mathsf { t } ) } | \lesssim \left( 1 + \frac { \mathsf { t } } { d } \right) d ^ { \epsilon - \frac { 1 } { 2 } }
$$

with probability at least $1 - C d ^ { - D }$ . A similar argument gives $\vert \mathcal { R } ( \widehat { f } _ { \mathrm { t } } ) - \mathcal { R } _ { \mathrm { d e t } } ^ { ( \mathrm { t } ) } \vert \lesssim ( 1 + \frac { \mathrm { t } } { d } ) ^ { 3 } d ^ { \epsilon - \frac { 1 } { 2 } }$ with probability at least $1 - C d ^ { - D }$ □

## E.2 Proof of Theorem 2

Lemma 21 (Gradient flow versus ridge). Let $\pmb { v } \in \mathbb { R } ^ { d }$ satisfy $\| \pmb { v } \| _ { 2 } = 1$ . For $\mathrm { { t } > 0 , \ \kappa _ { t } = \it { d } / \mathrm { { t } } }$ , we have

$$
\begin{array} { r } { \mathcal { L } _ { v } ^ { \mathrm { r e d } } ( \widehat { f } _ { \mathrm { t } , v } ; 0 ) \leq \mathcal { L } _ { v } ^ { \mathrm { r e d } } ( \widehat { f } _ { \kappa _ { \mathrm { t } } , v } ; 0 ) , \qquad \Vert \widehat { f } _ { \mathrm { t } , v } \Vert _ { \mathcal { H } } ^ { 2 } \leq 4 \Vert \widehat { f } _ { \kappa _ { \mathrm { t } } , v } \Vert _ { \mathcal { H } } ^ { 2 } . } \end{array}
$$

Proof. Consider the (complete) eigendecomposition of the empirical kernel

$$
\widehat { \mathcal { K } } = \sum _ { j \ge 1 } \lambda _ { j } \phi _ { j } \phi _ { j } ^ { * } , \qquad \lambda _ { j } \ge 0 , \quad \langle \phi _ { j } , \phi _ { j ^ { \prime } } \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = \mathbb { 1 } _ { j = j ^ { \prime } } ,
$$

and define the weighted spectral measure

$$
\nu _ { v } = \sum _ { j \geq 1 } \delta _ { \lambda _ { j } } \langle \phi _ { j } , \widehat { f } _ { v } ^ { \star } \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } ,
$$

where $\delta _ { \lambda _ { j } }$ is the Dirac measure at $\lambda _ { j }$ . By (19),

$$
\mathcal { L } _ { v } ^ { \mathrm { r e d } } ( \widehat { f } _ { \mathrm { t } , v } ; 0 ) = \Big \| \mathrm { e x p } ( - \mathfrak { t } \widehat { K } ) \widehat { f } _ { v } ^ { \star } \Big \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } = \int e ^ { - 2 \mathrm { t } x } \nu _ { v } ( \mathrm { d } x ) .
$$

On the other hand, (21) gives

$$
\mathcal { L } _ { v } ^ { \mathrm { r e d } } ( \widehat { f } _ { \kappa _ { \mathfrak t } , v } ; 0 ) = \int \left( \frac { x } { \kappa _ { \mathfrak t } / d } + 1 \right) ^ { - 2 } \nu _ { v } ( \mathrm { d } x ) = \int \left( \mathfrak { t } x + 1 \right) ^ { - 2 } \nu _ { v } ( \mathrm { d } x ) .
$$

Since $e ^ { - 2 u } \leq ( 1 + u ) ^ { - 2 }$ for $u \geq 0$ and $\nu _ { v }$ is supported on $[ 0 , \infty )$ , the first claim follows.

For the norm comparison, it follows by (18) and the definition of the empirical kernel $\widehat { \kappa }$ that

$$
\lVert \widehat { f } _ { \mathfrak { t } , v } \rVert _ { \mathcal { H } } ^ { 2 } = \left. \int _ { 0 } ^ { \mathfrak { t } } \exp ( - ( \mathfrak { t } - \mathfrak { s } ) \widehat { K } ) \widehat { f } _ { v } ^ { \star } \mathrm { d } \mathfrak { s } , \widehat { K } \int _ { 0 } ^ { \mathfrak { t } } \exp ( - ( \mathfrak { t } - \mathfrak { s } ) \widehat { K } ) \widehat { f } _ { v } ^ { \star } \mathrm { d } \mathfrak { s } \right. _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } .
$$

Define $q _ { \mathrm { t } } : \mathbb { R } _ { \geq 0 }  \mathbb { R }$ so that

$$
\int _ { 0 } ^ { \mathsf { t } } \exp ( - ( \mathsf { t } - \mathsf { s } ) \widehat { K } ) \mathrm { d } \mathsf { s } = q _ { \mathsf { t } } ( \widehat { K } ) , \qquad q _ { \mathsf { t } } ( x ) = \left\{ \begin{array} { l l } { \frac { 1 - e ^ { - \mathsf { t } x } } { x } , } & { x > 0 , } \\ { \mathsf { t } , } & { x = 0 . } \end{array} \right.\tag{65}
$$

Then,

$$
\lVert \widehat { f } _ { \mathrm { t } , v } \rVert _ { \mathcal { H } } ^ { 2 } = \left. q _ { \mathrm { t } } ( \widehat { K } ) \widehat { f } _ { v } ^ { \star } , \widehat { K } q _ { \mathrm { t } } ( \widehat { K } ) \widehat { f } _ { v } ^ { \star } \right. _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = \int x q _ { \mathrm { t } } ( x ) ^ { 2 } \nu _ { v } ( \mathrm { d } x ) = \int \frac { ( 1 - e ^ { - \mathrm { t } x } ) ^ { 2 } } { x } \nu _ { v } ( \mathrm { d } x )
$$

where the last integrand is set to zero at $x = 0$ . On the other hand, by (20),

$$
\| \widehat { f } _ { \kappa _ { \mathrm { t } } , v } \| _ { \mathcal H } ^ { 2 } = \int \frac { x } { ( x + \kappa _ { \mathrm { t } } / d ) ^ { 2 } } \nu _ { v } ( \mathrm { d } x ) = \int \frac { { \mathrm { t } } ^ { 2 } x } { ( { \mathrm { t } } x + 1 ) ^ { 2 } } \nu _ { v } ( \mathrm { d } x ) .
$$

Since $( 1 - e ^ { - u } ) ^ { 2 } / u \leq 4 u / ( 1 + u ) ^ { 2 }$ for $u \geq 0 .$ , the second claim follows by plugging in $u = \mathrm { t } x$ □

Proof of Theorem 2. The only properties of the ridge estimator used in the proof of Theorem $7$ are the training-loss bound $\mathcal { L } _ { v } ( \widehat { f } _ { \kappa , v } ; 0 ) \prec \kappa$ and the RKHS norm bound $\| \widehat { f } _ { \kappa , v } \| _ { \mathcal { H } } ^ { 2 } \prec d .$ By Lemma 21, these bounds also hold for the gradient flow estimator $\widehat { f } _ { \mathrm { t } , v }$ with $\kappa = d / \mathrm { t }$ . The rest of the proof then carries through identically as Theorem 7. □

## F The interpolation limit

This appendix is dedicated to the study of the gradient flow estimator (18) when the dimension d and the number of samples n are fixed while the gradient flow time t tends to infinity. Throughout, the difusion time $t > 0$ , and so $0 < \rho , \sigma < 1$ , are fixed. We focus on the gradient flow estimator, but the same arguments can be used to show analogous results for the ridge estimator (20).

## F.1 Convergence of the test error under truncation

We establish Proposition 4 via three sublemmas. We first show that the reproducing kernel Hilbert space H is dense in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ and $L ^ { 2 } ( \mu _ { d } )$ . This is a consequence of the universality of the kernel K and the fact that both measures have finite exponential moments.

Lemma 22 (Density of the RKHS). Assume Assumption 5 holds. Fix d, n and a realization $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ . Then H is dense in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ . Moreover, under Assumption 2, H is also dense in $L ^ { 2 } ( \mu _ { d } )$

Proof. A straightforward adaptation of Lemma 9 implies that every polynomial of the form ${ \pmb x } \mapsto$ $\| \boldsymbol { x } \| _ { 2 } ^ { k _ { 0 } } p ( \pmb { x } )$ , where $k _ { 0 } \geq 0$ is given in Assumption 5 and $p : \mathbb { R } ^ { d }  \mathbb { R }$ is a polynomial, belongs to H. It remains to recall why these polynomials are dense for the two measures at hand.

The measure $\widehat { \mu } _ { d , n }$ is a finite mixture of Gaussian measures with common covariance $\sigma ^ { 2 } \mathbf { I } _ { d } .$ , while $\mu _ { d }$ is the convolution of the law of $\rho { \pmb x } _ { 0 }$ with ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { d } )$ . Thus, for either $\nu \in \{ \widehat { \mu } _ { d , n } , \mu _ { d } \}$ , it follows from Lemma 1 that there exists $a _ { \nu } > 0$ such that $\begin{array} { r } { \int e ^ { a _ { \nu } \| \pmb { x } \| _ { 2 } ^ { 2 } } \nu ( \mathrm { d } \pmb { x } ) < \infty } \end{array}$

Define the finite measure $\bar { \nu } ( \mathrm { d } \pmb { x } ) = \| \pmb { x } \| ^ { 2 k _ { 0 } } \nu ( \mathrm { d } \pmb { x } )$ . Indeed, it follows from the inequality

$$
\| x \| ^ { 2 k _ { 0 } } \exp ( a _ { \nu } \| x \| ^ { 2 } / 2 ) \leq C e ^ { a _ { \nu } \| x \| ^ { 2 } }
$$

for some constant $C > 0$ that ¯ν inherits finite exponential moments from ν. If $g \in L ^ { 2 } ( \bar { \nu } )$ and g is orthogonal to all polynomials, then the signed measure $g { \bar { \nu } }$ has a (finite) Laplace transform $\begin{array} { r } { \int \exp ( \langle t , \pmb { x } \rangle ) g ( \pmb { x } ) \bar { \nu } ( \mathrm { d } \pmb { x } ) } \end{array}$ whose Taylor coeficients all vanish. The Laplace transform is therefore identically zero, and hence the signed measure $g { \bar { \nu } }$ is zero. Thus, $g = 0$ in $L ^ { 2 } ( \bar { \nu } )$ , proving that polynomials are dense in $L ^ { 2 } ( \bar { \nu } )$

Finally, since $\sigma > 0 , \| \pmb { x } \| ^ { k _ { 0 } }$ is almost surely nonzero under ν and $f \mapsto \| \pmb { x } \| ^ { k _ { 0 } } f$ is an isometry from $L ^ { 2 } ( \bar { \nu } )$ to $L ^ { 2 } ( \nu )$ . In addition, this isometry is onto. Therefore, the density of polynomials in $L ^ { 2 } ( \bar { \nu } )$ implies the density of H in $L ^ { 2 } ( \nu )$ □

Next, we show that the gradient flow estimator converges to the empirical Bayes denoiser $\widehat { f } ^ { \star }$ in $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ as $\mathbf { t }  \infty ,$ while the test error is asymptotically lower bounded by the Bayes risk $\mathcal { R } ( \widehat { \pmb f } ^ { \star } )$

Lemma 23. Suppose that Assumption 2 and Assumption 5 hold. Fix $d , n _ { i }$ , the training samples $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ and $\pmb { v } \in \mathbb { S } ^ { d - 1 }$ . Then,

$$
\operatorname* { l i m } _ { \mathrm { t \to \infty } } \| \mathcal { S } \widehat { f } _ { \mathrm { t } , v } - \widehat { \mathcal { P } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = \operatorname* { l i m } _ { \mathrm { t \to \infty } } \| \mathcal { S } _ { 0 } \widehat { f } _ { \mathrm { t } , v } - \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } = 0 ,\tag{66}
$$

and

$$
\operatorname* { l i m } _ { \ t  \infty } \operatorname* { i n f } _ { \mathbf { \ell } } \mathcal { R } _ { v } ( \widehat { f } _ { \mathrm { t } , v } ) \geq \mathcal { R } _ { v } ( \widehat { f } _ { v } ^ { \star } ) .\tag{67}
$$

Proof. Denote $g _ { v } : = \widehat { \mathcal { P } } \widehat { f } _ { v } ^ { \star } \in L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ . By (18), recall that

$$
\begin{array} { r } { S \widehat { f } _ { \mathsf { t } , v } = \big ( \mathrm { i d } _ { L ^ { 2 } ( \widehat \pi _ { d , n } ) } - \exp ( - \mathrm { t } \widehat { \mathcal { K } } ) \big ) g _ { v } . } \end{array}
$$

The operator $\widehat { \cal K } = { \cal S } { \cal S } ^ { * }$ is bounded, self-adjoint and nonnegative. Since $1 - e ^ { - \mathsf { t } \lambda }$ is a bounded, continuous function of λ for every $\mathrm { \Delta t > 0 }$ , and $\mathbb { 1 } - e ^ { - \mathsf { t } \lambda } \to \mathbb { 1 } _ { ( 0 , \infty ) } ( \lambda )$ as $\mathrm { \sf t } \to \infty$ , the spectral theorem and dominated convergence ensure that

$$
\operatorname* { l i m } _ { \ t \to \infty } ( \mathrm { i d } _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } - \exp ( - \mathrm { t } \widehat { K } ) ) = \mathsf { P } _ { \widehat { \mathrm { R a n } ( \widehat { K } ) } } ,
$$

where the convergence is in the strong operator topology and $\mathsf { P } _ { \overline { { \mathrm { R a n } ( \widehat { \mathcal { K } } ) } } }$ is the orthogonal projection onto the closed subspace $\overline { { \mathrm { R a n } ( \widehat { \mathcal { K } } ) } } \subseteq L ^ { 2 } ( \widehat { \pi } _ { d , n } )$

Since $\mathrm { R a n } ( \widehat { \mathcal { K } } ) = \overline { { \mathrm { R a n } ( \mathcal { S } ) } }$ , it remains to identify the closure of Ran(S). By Lemma 22, $ { \boldsymbol { S } } _ { 0 }  { \mathcal { H } }$ is dense in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ . Furthermore, $\widehat { \mathcal P }$ is an isometry. Hence $\begin{array} { r } { S \mathcal { H } = \widehat { \mathcal { P } } S _ { 0 } \mathcal { H } } \end{array}$ is dense in $\widehat { \mathcal { P } } L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ , and $g _ { v } = \widehat { \mathcal { P } } \widehat { f } _ { v } ^ { \star }$ belongs to this closed subspace. This proves (66).

Since $\sigma > 0$ , both $\widehat { \mu } _ { d , n }$ and $\mu _ { d }$ have strictly positive continuous densities, say $\widehat { p } _ { d , n }$ and $p _ { d } .$ with respect to Lebesgue measure. Therefore, for every fixed $R < \infty$ 2

$$
C _ { R } : = \operatorname* { s u p } _ { \| \pmb { u } \| _ { 2 } \leq R } \frac { p _ { d } ( \pmb { u } ) } { \widehat { p _ { d , n } } ( \pmb { u } ) } < \infty .
$$

For $B _ { R } = \left\{ \| { \pmb u } \| _ { 2 } \leq R \right\}$

$$
\int _ { B _ { R } } \left. \mathcal { T } _ { 0 } \widehat { f } _ { \mathrm { t } , v } - \widehat { f } _ { v } ^ { \star } \right. ^ { 2 } \mathrm { d } \mu _ { d } \leq C _ { R } \left. \mathcal { S } _ { 0 } \widehat { f } _ { \mathrm { t } , v } - \widehat { f } _ { v } ^ { \star } \right. _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } ^ { 2 } \longrightarrow 0
$$

as $\mathbf { t } \to \infty$ by (66). Thus, $\mathcal { T } _ { 0 } \widehat { f } _ { \mathrm { t } , v }  \widehat { f } _ { v } ^ { \star }$ in probability under $\mu _ { d } \mathrm { a s } \mathrm { t }  \infty$ . Since $\mu _ { d }$ is the pushforward of $\pi _ { d }$ under $( { \pmb x } , z ) \mapsto \rho { \pmb x } + \sigma z$ , it follows that

$$
| T \widehat { f } _ { { \sf t } , v } - z _ { v } | ^ { 2 } \longrightarrow | { \mathcal { P } } \widehat { f } _ { v } ^ { \star } - z _ { v } | ^ { 2 }
$$

in probability under $\pi _ { d }$ . Now choose a sequence $\mathfrak { t } _ { k } \to \infty$ along which the risks converge to their liminf. Passing to a further subsequence, the preceding convergence holds almost surely. Fatou’s lemma then gives (67). □

Finally, we show that the test error of the truncated gradient flow estimator converges to the test error of the empirical Bayes denoiser. We denote

$$
\Pi _ { A } f ( { \pmb u } ) = \frac { \mathsf { P } _ { A } D _ { f } ( { \pmb u } ) - { \pmb u } } { \sigma } ,
$$

where $D _ { f } ( \pmb { u } ) = \pmb { u } + \sigma \pmb { f } ( \pmb { u } )$ and $\mathsf { P } _ { A }$ is the Euclidean projection onto the compact convex set $A _ { i }$ which is assumed to contain all the sample vectors $\{ \rho \pmb { x } _ { 0 , i } \} _ { i \in [ n ] }$

Lemma 24. Suppose that Assumptions 2 and 5 hold. Fix d, n and a training realization $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ $T h e n ,$

$$
\operatorname* { l i m } _ { \mathrm { t }  \infty } \mathcal { R } ( \Pi _ { A } \widehat { \pmb f _ { \mathrm { t } } } ) = \mathcal { R } ( \widehat { \pmb f } ^ { \star } ) .
$$

Proof. Note that

$$
D _ { \widehat { f } ^ { \star } } ( \pmb { u } ) = \rho \sum _ { i = 1 } ^ { n } \omega _ { i } ( \pmb { u } ) \pmb { x } _ { 0 , i } \in \rho \mathrm { ~ c o n v } \{ \pmb { x } _ { 0 , 1 } , \dots , \pmb { x } _ { 0 , n } \} \subseteq A ,
$$

so $\Pi _ { A } \widehat { f } ^ { \star } = \widehat { f } ^ { \star }$ . Thus,

$$
\| \Pi _ { A } \widehat { \pmb f _ { \mathrm { t } } } ( \pmb u ) - \widehat { \pmb f ^ { \star } } ( \pmb u ) \| _ { 2 } = \frac { 1 } { \sigma } \| \mathsf { P } _ { A } D _ { \widehat { \pmb f _ { \mathrm { t } } } } ( \pmb u ) - \mathsf { P } _ { A } D _ { \widehat { \pmb f ^ { \star } } } ( \pmb u ) \| _ { 2 } .
$$

Since the Euclidean projection onto a convex set is non-expansive,

$$
\| \Pi _ { A } \widehat { f } _ { \mathrm { t } } ( \boldsymbol { u } ) - \widehat { f } ^ { \star } ( \boldsymbol { u } ) \| _ { 2 } \le \| \widehat { f } _ { \mathrm { t } } ( \boldsymbol { u } ) - \widehat { f } ^ { \star } ( \boldsymbol { u } ) \| _ { 2 } , \qquad \| \Pi _ { A } \widehat { f } _ { \mathrm { t } } ( \boldsymbol { u } ) - \widehat { f } ^ { \star } ( \boldsymbol { u } ) \| _ { 2 } \le \frac { \mathrm { d i a m } ( A ) } { \sigma } .\tag{68}
$$

On a fixed ball $B _ { R }$ , using a similar density comparison as in the proof of Lemma 23,

$$
\int _ { B _ { R } } \Vert \Pi _ { A } \widehat { f } _ { \mathrm { t } } ( \boldsymbol { u } ) - \widehat { f } ^ { \star } ( \boldsymbol { u } ) \Vert _ { 2 } ^ { 2 } \mathrm { d } \mu _ { d } \leq C _ { R } \int \Vert \widehat { f } _ { \mathrm { t } } ( \boldsymbol { u } ) - \widehat { f } ^ { \star } ( \boldsymbol { u } ) \Vert _ { 2 } ^ { 2 } \mathrm { d } \widehat { \mu } _ { d , n } \longrightarrow 0 .
$$

The second bound in (68) gives

$$
\int _ { B _ { R } ^ { c } } \| \Pi _ { A } \widehat { \pmb f _ { \mathrm { t } } } ( \pmb u ) - \widehat { \pmb f ^ { \star } } ( \pmb u ) \| _ { 2 } ^ { 2 } \mathrm { d } \mu _ { d } \leq \frac { \mathrm { d i a m } ( A ) ^ { 2 } } { \sigma ^ { 2 } } \mu _ { d } ( B _ { R } ^ { c } ) .
$$

First take the limit $\mathrm { \Omega t } \to \infty$ and then let $R \to \infty$ to obtain $\Pi _ { A } { \widehat { f _ { \mathrm { t } } } } \to { \widehat { f } } ^ { \star }$ in $L ^ { 2 } ( \mu _ { d } )$ . This implies the convergence of the test error. □

Proof of Proposition $\it 4 .$ The fact that the RKHS is dense in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ and $L ^ { 2 } ( \mu _ { d } )$ follows from Lemma 22. The vector-valued convergence of the gradient flow estimator to the empirical Bayes denoiser in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ and the lower bound on the test error follow from Lemma 23 by an averaging argument over an orthonormal basis of $\mathbb { R } ^ { d }$ to decompose the vector-valued $L ^ { 2 }$ norm into a sum of scalar $L ^ { 2 }$ norms. The convergence of the test error follows from Lemma 24. □

## F.2 Instability of the test error at the ridgeless limit

Before establishing the instability of the ridgeless limit, we record some estimates for polynomials under Gaussian measures. For $s > 0 , N \in \mathbb { N }$ , let $\gamma _ { s } : = \mathcal { N } ( 0 , s ^ { 2 } \mathbf { I } _ { d } )$ , and let $\mathcal { P } _ { N }$ denote the space of polynomials on $\mathbb { R } ^ { d }$ of total degree at most N. Every $P \in \mathcal { P } _ { N }$ admits a unique decomposition

$$
P ( \pmb { u } ) = \sum _ { k = 0 } ^ { N } \langle \pmb { A } _ { k } , \pmb { u } ^ { \otimes k } \rangle _ { \mathrm { F } } , \qquad \pmb { A } _ { k } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) .
$$

Recall the normalized Hermite tensors $\mathrm { H e } _ { k }$ from (50).

For the counterexample in this subsection, we specialize the population distribution to $\pi _ { 0 } = \gamma _ { 1 }$ Since $\rho ^ { 2 } + \sigma ^ { 2 } = 1$ , this gives $\mu _ { d } = \gamma _ { 1 }$ , with $0 < \sigma < 1$ fixed throughout.

Lemma 25 (Gaussian polynomial estimates). Fix $d \geq 1$ and $s > 0$ . Suppose that

$$
P ( \pmb { x } ) = \sum _ { k = 0 } ^ { N } \langle \pmb { A } _ { k } , \pmb { x } ^ { \otimes k } \rangle _ { \mathrm { F } } , \qquad \pmb { A } _ { k } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) .
$$

Then there are unique tensors $B _ { k } \in \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ such that

$$
P ( \pmb { x } ) = \sum _ { k = 0 } ^ { N } \langle \pmb { B } _ { k } , \mathrm { H e } _ { k } ( \pmb { x } / s ) \rangle _ { \mathrm { F } } .
$$

Moreover, there exists $C = C ( d , s ) < \infty$ such that

$$
e ^ { - C N \log ( N + 2 ) } \sum _ { k = 0 } ^ { N } \| A _ { k } \| _ { \mathrm { F } } ^ { 2 } \leq \sum _ { k = 0 } ^ { N } \| B _ { k } \| _ { \mathrm { F } } ^ { 2 } = \| P \| _ { L ^ { 2 } ( \gamma _ { s } ) } ^ { 2 } \leq e ^ { C N \log ( N + 2 ) } \sum _ { k = 0 } ^ { N } \| A _ { k } \| _ { \mathrm { F } } ^ { 2 } .
$$

Proof. We use the convention

$$
\mathbf { I } _ { d } ^ { \otimes j } : = \sum _ { i _ { 1 } , \ldots , i _ { j } = 1 } ^ { d } \left( e _ { i _ { 1 } } \otimes \cdot \cdot \cdot \otimes e _ { i _ { j } } \right) \otimes \left( e _ { i _ { 1 } } \otimes \cdot \cdot \cdot \otimes e _ { i _ { j } } \right)
$$

such that $\langle { \pmb A } \otimes { \pmb B } , { \bf I } _ { d } ^ { \otimes j } \rangle _ { \mathrm { F } } = \langle { \pmb A } , { \pmb B } \rangle _ { \mathrm { F } }$ for A, $B \in ( \mathbb { R } ^ { d } ) ^ { \otimes j }$ . The standard Hermite identities read

$$
\mathrm { H e } _ { k } ( \pmb { x } ) = \frac { 1 } { \sqrt { k ! } } \sum _ { j = 0 } ^ { \lfloor k / 2 \rfloor } \frac { ( - 1 ) ^ { j } k ! } { 2 ^ { j } j ! ( k - 2 j ) ! } \mathsf { P } _ { \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) } \left( \pmb { x } ^ { \otimes ( k - 2 j ) } \otimes \mathbf { I } _ { d } ^ { \otimes j } \right) ,\tag{69}
$$

and, conversely,

$$
\pmb { x } ^ { \otimes k } = \sum _ { j = 0 } ^ { \lfloor k / 2 \rfloor } \frac { k ! } { 2 ^ { j } j ! \sqrt { ( k - 2 j ) ! } } \mathsf { P } _ { \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) } \left( \mathrm { H e } _ { k - 2 j } ( \pmb { x } ) \otimes \mathbf { I } _ { d } ^ { \otimes j } \right) .\tag{70}
$$

For $\pmb { A } \in \operatorname { S y m } ^ { k } ( \mathbb { R } ^ { d } )$ , let $\mathrm { T r } ^ { j } ( A ) \ \in \ \mathrm { S y m } ^ { k - 2 j } ( \mathbb { R } ^ { d } )$ denote its j-fold trace. Let $\mathbf { \sigma } _ { \mathbf { \mathcal { I } } } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ Applying (70) to $P ( s \mathbf { \mathbf { \mathbf { g } } } )$ gives

$$
P ( s \pmb { g } ) = \sum _ { k = 0 } ^ { N } \langle B _ { k } , \mathrm { H e } _ { k } ( \pmb { g } ) \rangle _ { \mathrm { F } } ,
$$

where

$$
B _ { k } = \sum _ { j = 0 } ^ { \lfloor ( N - k ) / 2 \rfloor } \frac { s ^ { k + 2 j } ( k + 2 j ) ! } { 2 ^ { j } j ! \sqrt { k ! } } \mathrm { T r } ^ { j } ( { \cal A } _ { k + 2 j } ) .
$$

The Hermite-tensor isometry gives

$$
\| P \| _ { L ^ { 2 } ( \gamma _ { s } ) } ^ { 2 } = \sum _ { k = 0 } ^ { N } \| B _ { k } \| _ { \mathrm { F } } ^ { 2 } .
$$

Conversely, expanding the Hermite tensors by (69) gives

$$
A _ { k } = \sum _ { j = 0 } ^ { \lfloor ( N - k ) / 2 \rfloor } \frac { ( - 1 ) ^ { j } s ^ { - k } \sqrt { ( k + 2 j ) ! } } { 2 ^ { j } j ! k ! } \mathrm { T r } ^ { j } ( B _ { k + 2 j } ) .
$$

Since $\Vert \mathrm { T r } ^ { j } ( A ) \Vert _ { \mathrm { F } } \leq d ^ { j / 2 } \Vert A \Vert _ { \mathrm { F } }$ ,

$$
\| A _ { k } \| _ { \mathrm { F } } \leq \sum _ { j = 0 } ^ { \lfloor ( N - k ) / 2 \rfloor } \frac { s ^ { - k } \sqrt { ( k + 2 j ) ! } } { 2 ^ { j } j ! k ! } d ^ { j / 2 } \| B _ { k + 2 j } \| _ { \mathrm { F } } \leq ( 1 \vee s ^ { - 1 } ) ^ { N } N ^ { N / 2 } d ^ { N } \sum _ { k = 0 } ^ { N } \| B _ { k } \| _ { \mathrm { F } } .
$$

Squaring and summing over k using Cauchy–Schwarz, and using a similar argument for $\| B _ { k } \| _ { \mathrm { F } }$ in terms of $\| A _ { k } \| _ { \mathrm { F } } ,$ gives the desired estimate. □

We also require an estimate for the complex evaluation of polynomials. Recall the elementary bound

$$
| H _ { m } ( z ) | \leq 2 ^ { m / 2 } \sqrt { m ! } \exp \left( \sqrt { 2 m } | z | \right) , \qquad z \in \mathbb { C } ,
$$

for the physicists’ Hermite polynomials (see, for instance, [vEM90, Eq. (1.2)]). Since

$$
\mathrm { H e } _ { m } ( w ) = \frac { 2 ^ { - m / 2 } H _ { m } ( w / \sqrt { 2 } ) } { \sqrt { m ! } } ,
$$

it follows that $| \mathrm { H e } _ { m } ( w ) | \leq \exp ( \sqrt { m } | w | )$ for every $w \in \mathbb { C } .$ . Then, for every $P \in \mathcal { P } _ { N }$ and $z \in \mathbb { C } ^ { d }$ we apply Cauchy–Schwarz and expand the entries of $\mathrm { H e } _ { k } ( z / s )$ in the normalized product Hermite basis to obtain

$$
| P ( z ) | ^ { 2 } \leq \binom { N + d } { d } \exp \left( \frac { 2 \sqrt { N } } { s } \| z \| _ { 2 } \right) \| P \| _ { L ^ { 2 } ( \gamma _ { s } ) } ^ { 2 } .
$$

Consequently, for every compact $\mathcal { D } \subset \mathbb { C } ^ { d }$ , there exists $C = C ( \mathcal { D } , d , s ) < \infty$ such that

$$
\operatorname* { s u p } _ { z \in { \cal D } } | P ( z ) | \leq C ( N + 1 ) ^ { d / 2 } \exp \Bigl ( C \sqrt { N + 1 } \Bigr ) \| P \| _ { L ^ { 2 } ( \gamma _ { s } ) } .\tag{71}
$$

Lemma 26. Fix $d \geq 1 , n \geq 2$ , and a training realization $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ . Let $v \in \mathbb { S } ^ { d - 1 }$ be such that the numbers $\{ \langle { \pmb v } , { \pmb x } _ { 0 , i } \rangle \} _ { i = 1 } ^ { n }$ are pairwise distinct. For every $N \geq 0$ , let $\mathsf { P } _ { \mathcal { P } _ { N } }$ denote the $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ orthogonal projection onto $\mathcal { P } _ { N }$ . Then, for every $r > \sigma$

$$
\operatorname* { s u p } _ { N \geq 0 } \| \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \gamma _ { r } ) } = \infty .
$$

Proof. Write

$$
p _ { N } : = \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } , \qquad h _ { N } : = p _ { N } - p _ { N - 1 } , \qquad N \geq 1 .
$$

Fix $r > \sigma$ and suppose, toward a contradiction, that

$$
\operatorname* { s u p } _ { N } \| p _ { N } \| _ { L ^ { 2 } ( \gamma _ { r } ) } = : M < \infty ,
$$

so that $\| h _ { N } \| _ { L ^ { 2 } ( \gamma _ { r } ) } \ \leq \ 2 M$ Choose $\ : 0 < s _ { - } < \sigma < s _ { + } < r$ . Since $\widehat { \mu } _ { d , n }$ is a finite mixture of translates of $\gamma _ { \sigma } .$ , there exist $c _ { - } , c _ { + } > 0$ such that $c _ { - } \gamma _ { s _ { - } } \leq \widehat { \mu } _ { d , n } \leq c _ { + } \gamma _ { s _ { + } }$ in the sense of pointwise comparison of densities. Write the top homogeneous part of $h _ { N }$ as ${ \pmb u } \stackrel { \cdot } { \mapsto } \langle { \pmb A } _ { N } , { \pmb u } ^ { \otimes N } \rangle _ { \mathrm { \pmb F } }$ . By (69), the top homogeneous part of the polynomial $s _ { + } ^ { N } \sqrt { N ! } \langle A _ { N } , \mathrm { H e } _ { N } ( { \pmb u } / s _ { + } ) \rangle _ { \mathrm { F } }$ is precisely $\langle { \pmb A } _ { N } , { \pmb u } ^ { \otimes N } \rangle _ { \mathrm { \scriptscriptstyle F } }$ Consequently,

$$
s _ { + } ^ { N } \sqrt { N ! } \langle A _ { N } , \mathrm { H e } _ { N } ( \cdot / s _ { + } ) \rangle _ { \mathrm { F } } - h _ { N } \in \mathcal { P } _ { N - 1 } .
$$

Since $h _ { N } \perp _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \mathcal { P } _ { N - 1 }$

$$
\begin{array} { r } { \| h _ { N } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \leq \left\| s _ { + } ^ { N } \sqrt { N ! } \langle A _ { N } , \mathrm { H e } _ { N } ( \cdot / s _ { + } ) \rangle _ { \mathrm { F } } \right\| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \leq \sqrt { c _ { + } } s _ { + } ^ { N } \sqrt { N ! } \| A _ { N } \| _ { \mathrm { F } } , } \end{array}
$$

where the second inequality follows from $\widehat { \mu } _ { d , n } \leq c _ { + } \gamma _ { s _ { + } }$ <sub>+</sub> and the Hermite-tensor isometry.

On the other hand, for every $r > 0$ , the degree-N Wiener-chaos component of $h _ { N }$ under $\gamma _ { r }$ is $r ^ { N } \sqrt { N ! } \langle A _ { N } , \mathrm { H e } _ { N } ( \cdot / r ) \rangle _ { \mathrm { F } }$ , so

$$
r ^ { N } \sqrt { N ! } \| { \pmb { A } } _ { N } \| _ { \mathrm { F } } \leq \| { \pmb { h } } _ { N } \| _ { L ^ { 2 } ( \gamma _ { r } ) } .
$$

Combining the preceding two estimates yields

$$
\| h _ { N } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \leq \sqrt { c _ { + } } \left( \frac { s _ { + } } { r } \right) ^ { N } \| h _ { N } \| _ { L ^ { 2 } ( \gamma _ { r } ) } .
$$

Finally, using $c _ { - } \gamma _ { s _ { - } } \leq \widehat \mu _ { d , n }$ , we obtain

$$
\| h _ { N } \| _ { L ^ { 2 } ( \gamma _ { s _ { - } } ) } \leq \sqrt { \frac { c _ { + } } { c _ { - } } } \left( \frac { s _ { + } } { r } \right) ^ { N } \| h _ { N } \| _ { L ^ { 2 } ( \gamma _ { r } ) } \leq 2 \sqrt { \frac { c _ { + } } { c _ { - } } } \left( \frac { s _ { + } } { r } \right) ^ { N } M .
$$

Let D be a compact subset of $\mathbb { C } ^ { d }$ . By (71), there exists a constant $C = C ( \mathcal { D } , d , s _ { - } )$ such that

$$
\operatorname* { s u p } _ { z \in { \mathcal { D } } } | h _ { N } ( z ) | \leq C ( N + 1 ) ^ { d / 2 } \exp \left( C \sqrt { N + 1 } \right) 2 \sqrt { \frac { c _ { + } } { c _ { - } } } \left( \frac { s _ { + } } { r } \right) ^ { N } M .
$$

The right-hand side is summable in $N ,$ so the Weierstrass convergence theorem shows that $p _ { N } =$ $\textstyle p _ { 0 } + \sum _ { k = 1 } ^ { N } h _ { k }$ converges locally uniformly on $\mathbb { C } ^ { d }$ to an entire function G. On the other hand, polynomial density (see Lemma 22) gives $p _ { N }  \widehat { f } _ { v } ^ { \star }$ in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ . Passing to an almost-everywhere convergent subsequence therefore shows that $G = \widehat { f } _ { v } ^ { \star } , \widehat { \mu } _ { d , n }$ -almost everywhere on $\mathbb { R } ^ { d }$ . Since both functions are continuous and $\widehat { \mu } _ { d , n }$ has a strictly positive density, they agree everywhere on $\mathbb { R } ^ { d }$

Define

$$
F _ { v } ( z ) : = \sum _ { i = 1 } ^ { n } \exp \left( - \frac { \rho ^ { 2 } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } + \frac { \rho \langle \pmb { v } , \pmb { x } _ { 0 , i } \rangle } { \sigma ^ { 2 } } z \right) , \qquad z \in { \mathbb C } .
$$

For every $t \in \mathbb { R }$ , the empirical-score formula gives

$$
\widehat { f } _ { v } ^ { \star } ( t v ) = - \frac { t } { \sigma } + \sigma \frac { F _ { v } ^ { \prime } ( t ) } { F _ { v } ( t ) } .
$$

Since $G ( t v ) = \widehat { f } _ { v } ^ { \star } ( t v )$ , multiplying by $\sigma F _ { v } ( t )$ yields

$$
F _ { v } ( t ) ( \sigma G ( t v ) + t ) = \sigma ^ { 2 } F _ { v } ^ { \prime } ( t )
$$

for every $t \in \mathbb { R }$ . Both sides are restrictions to R of entire functions of the complex variable z, so the identity theorem gives

$$
F _ { v } ( z ) ( \sigma G ( z v ) + z ) = \sigma ^ { 2 } F _ { v } ^ { \prime } ( z ) , \qquad z \in \mathbb { C } .
$$

By assumption, $F _ { v }$ is an exponential sum with at least two distinct frequencies, and therefore has a complex zero. Indeed, $F _ { v }$ is entire of exponential type, so if it had no zeros, Hadamard factorization would give $F _ { v } ( z ) = e ^ { a z + b }$ , whereas $F _ { v } ^ { \prime } ( t ) / F _ { v } ( t )$ converges as $t \to \pm \infty$ to the largest and smallest frequencies, respectively, which are distinct. Suppose that $F _ { v }$ has a zero of order $m \geq 1$ at $z _ { 0 } \in \mathbb { C }$ Since $\sigma G ( z { \pmb v } ) + z$ is entire, the left-hand side has a zero of order at least m at $z _ { 0 }$ , while the righthand side has a zero of exactly $m - 1$ . This is a contradiction. Since $r > \sigma$ was arbitrary, the claim follows. □

Fix $\eta > 4$ and set, for every $N \geq 2$

$$
h ( s ) : = \sum _ { k = 0 } ^ { \infty } \alpha _ { k } s ^ { k } , \qquad h _ { \leq N } ( s ) : = \sum _ { k = 0 } ^ { N } \alpha _ { k } s ^ { k } , \qquad \alpha _ { 0 } = \alpha _ { 1 } = 1 , \qquad \alpha _ { k } : = e ^ { - \eta ^ { k } } , \quad k \geq 2 .\tag{72}
$$

Let $K , K _ { < N } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R }$ be the inner-product kernels associated with h and $h { \_ } N$ , respectively, and let H and $\mathcal { H } _ { \leq N } \subset \mathcal { H }$ be the corresponding RKHSs. For $\nu \in \{ \mu _ { d } , \widehat { \mu } _ { d , n } \}$ , Gaussian-mixture moment bounds give

$$
\int K ( \boldsymbol { u } , \boldsymbol { u } ) \nu ( \mathrm { d } \boldsymbol { u } ) \leq 1 + C + \sum _ { k \geq 2 } e ^ { - \eta ^ { k } } ( C k ) ^ { k } < \infty .\tag{73}
$$

Hence, K is an admissible kernel, and the inclusion operators $S _ { 0 } : \mathcal { H } \to L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ and $\mathcal { T } _ { 0 } : \mathcal { H } $ $L ^ { 2 } ( \mu _ { d } )$ are bounded. Let $\mathcal { S } _ { 0 , \leq N }$ and $\mathcal { T } _ { 0 , \leq N }$ be their restrictions to $\mathcal { H } _ { < N }$ , and define

$$
\widehat { K } _ { 0 } : = { S } _ { 0 } { S } _ { 0 } ^ { * } , \qquad K _ { 0 } : = { \mathcal { T } } _ { 0 } { S } _ { 0 } ^ { * } , \qquad \widehat { K } _ { 0 , \le N } : = { S } _ { 0 , \le N } { S } _ { 0 , \le N } ^ { * } , \qquad K _ { 0 , \le N } : = { \mathcal { T } } _ { 0 , \le N } { S } _ { 0 , \le N } ^ { * } .
$$

These are respectively the full and truncated empirical and population-transfer kernel operators.

Lemma 27. Suppose that $\pi _ { 0 } = \gamma _ { 1 }$ . Fix $d \geq 1 , n \geq 2$ , and a training realization $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ . For every $N \geq 2$ , let $\lambda _ { N }$ denote the smallest positive eigenvalue of $\widehat { \mathcal { K } } _ { 0 , \leq N }$ , and

$$
I _ { N } : = \operatorname* { s u p } _ { 0 \neq P \in \mathcal { P } _ { N } } \frac { \Vert P \Vert _ { L ^ { 2 } ( \mu _ { d } ) } } { \Vert P \Vert _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } } .
$$

Then $\mathrm { R a n } ( \widehat { \mathcal { K } } _ { 0 , \leq N } ) = \mathcal { P } _ { N }$ , and there exists a constant $C = C ( d , \sigma , { \pmb X } ) > 0$ independent of N such that

$$
e ^ { - \eta ^ { N } - C N \log ( N + 2 ) } \leq \lambda _ { N } \leq e ^ { - \eta ^ { N } + C N \log ( N + 2 ) } , \qquad I _ { N } \leq e ^ { C N \log ( N + 2 ) } .
$$

Furthermore,

$$
\begin{array} { r } { \| \widehat { \mathcal { K } } _ { 0 } - \widehat { \mathcal { K } } _ { 0 , \leq N } \| _ { \mathrm { o p } } + \| \mathcal { K } _ { 0 } - \mathcal { K } _ { 0 , \leq N } \| _ { \mathrm { o p } } \leq e ^ { - \eta ^ { N + 1 } + C N \log ( N + 2 ) } . } \end{array}
$$

Proof. Using the tensor feature representation as in Lemma 9, set

$$
\mathcal { F } _ { \leq N } : = \bigoplus _ { k = 0 } ^ { N } \mathrm { S y m } ^ { k } ( \mathbb { R } ^ { d } ) , \qquad \phi _ { \leq N } ( \boldsymbol { u } ) : = \left( \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } \boldsymbol { u } ^ { \otimes k } \right) _ { k = 0 } ^ { N } ,
$$

such that

$$
K _ { \leq N } ( { \pmb u } , { \pmb u } ^ { \prime } ) = \langle \phi _ { \leq N } ( { \pmb u } ) , \phi _ { \leq N } ( { \pmb u } ^ { \prime } ) \rangle _ { \mathcal { F } _ { \leq N } } .
$$

Let $\Phi _ { N } : { \mathcal { F } } _ { \leq N } \to L ^ { 2 } ( { \widehat { \mu } } _ { d , n } )$ be the feature operator $( \Phi _ { N } ( { \pmb W } ) ) ( { \pmb u } ) : = \langle { \pmb W } , \phi _ { \leq N } ( { \pmb u } ) \rangle _ { \mathcal { F } _ { \leq N } }$ . Then, $\Phi _ { N } ^ { * } f = \mathbb { E } _ { \pmb { u } \sim \widehat { \mu } _ { d , n } } [ f ( \pmb { u } ) \phi _ { \leq N } ( \pmb { u } ) ]$ and $\widehat { \cal K } _ { 0 , \le N } = \Phi _ { N } \Phi _ { N } ^ { * }$ . Thus,

$$
C _ { N } : = \Phi _ { N } ^ { * } \Phi _ { N } = \mathbb { E } _ { \pmb { u } \sim \hat { \mu } _ { d , n } } [ \phi _ { \le N } ( \pmb { u } ) \otimes \phi _ { \le N } ( \pmb { u } ) ]
$$

has the same nonzero spectrum as $\widehat { \mathcal { K } } _ { 0 , \leq N }$ . Note that, for every $\pmb { W } = ( \pmb { W } _ { k } ) _ { k = 0 } ^ { N }$

$$
\langle \pmb { W } , \pmb { C } _ { N } \pmb { W } \rangle = \mathbb { E } _ { \pmb { u } \sim \hat { \mu } _ { d , n } } \left[ | \langle \pmb { W } , \phi _ { \leq N } ( \pmb { u } ) \rangle | ^ { 2 } \right] = \| P _ { \pmb { W } } \| _ { L ^ { 2 } ( \hat { \mu } _ { d , n } ) } ^ { 2 } ,
$$

where

$$
P _ { W } ( u ) : = \langle W , \phi _ { \leq N } ( u ) \rangle _ { \mathcal { F } _ { \leq N } } = \sum _ { k = 0 } ^ { N } \langle A _ { k } , u ^ { \otimes k } \rangle _ { \mathrm { F } } , \qquad A _ { k } : = \sqrt { \frac { \alpha _ { k } } { d ^ { k } } } W _ { k } .\tag{74}
$$

If the quadratic form is $0 ,$ the fact that $\widehat { \mu } _ { d , n }$ has a strictly positive density implies that the polynomial $P _ { W }$ vanishes identically and hence that $W = 0$ . Therefore, $C _ { N }$ is positive definite. Since $\Phi _ { N }$ has finite-dimensional domain, it follows that $\mathrm { R a n } ( \widehat { \mathcal { K } } _ { 0 , \leq N } ) = \mathrm { R a n } ( \Phi _ { N } ) = \mathcal { P } _ { N }$ . Moreover,

$$
\lambda _ { N } = \operatorname* { m i n } _ { \| \pmb { W } \| _ { \mathcal { F } _ { \leq N } = 1 } } \langle \pmb { W } , \pmb { C } _ { N } \pmb { W } \rangle .
$$

Let $\pmb { W } = ( \pmb { W } _ { k } ) _ { k = 0 } ^ { N }$ be a unit vector in $\mathcal { F } _ { < N }$ and $P _ { W }$ the corresponding polynomial from (74). Since $N \geq 2$ and $\alpha _ { k } d ^ { - k } \geq e ^ { - \eta ^ { N } } d ^ { - N }$ for every $0 \leq k \leq N$ ，

$$
\sum _ { k = 0 } ^ { N } \| \pmb { A } _ { k } \| _ { \mathrm { F } } ^ { 2 } = \sum _ { k = 0 } ^ { N } \frac { \alpha _ { k } } { d ^ { k } } \| \pmb { W } _ { k } \| _ { \mathrm { F } } ^ { 2 } \geq e ^ { - \eta ^ { N } } d ^ { - N } .
$$

Choose $0 < s _ { - } < \sigma < s _ { + } < 1$ . As in the proof of Lemma 26, there exist $c _ { - } , c _ { + } > 0$ such that $c _ { - } \gamma _ { s _ { - } } \leq \widehat { \mu } _ { d , n } \leq c _ { + } \gamma _ { s _ { + } }$ in the sense of pointwise comparison of densities. By Lemma 25, there exists a constant $C = C ( d , s _ { - } )$ such that

$$
\begin{array} { r } { \langle \pmb { W } , \pmb { C } _ { N } \pmb { W } \rangle _ { \mathcal { F } _ { \leq N } } = \| P _ { \pmb { W } } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } ^ { 2 } \geq c _ { - } \| P _ { \pmb { W } } \| _ { L ^ { 2 } ( \gamma _ { s _ { - } } ) } ^ { 2 } \geq e ^ { - \eta ^ { N } - C N \log ( N + 2 ) } , } \end{array}
$$

where we absorbed $c _ { - }$ and $d ^ { - N }$ into the last exponential. Taking the infimum over unit vectors W proves the lower bound for $\lambda _ { N }$

For the reverse bound, take ${ \pmb W } _ { N } = { \pmb e } _ { 1 } ^ { \otimes N }$ and $W _ { k } = 0$ for $k < N$ . This is a unit vector in $\mathcal { F } _ { \leq N }$ Using $\widehat { \mu } _ { d , n } \leq c _ { + } \gamma _ { s _ { + } }$ and the upper bound in Lemma 25, we obtain

$$
\lambda _ { N } \leq \| P _ { W } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } ^ { 2 } \leq e ^ { - \eta ^ { N } + C N \log ( N + 2 ) }
$$

where we potentially increase the constant C. This proves the upper bound for $\lambda _ { N }$

Next, let $P \in { \mathcal { P } } _ { N } \setminus \{ 0 \}$ . We combine the estimates obtained from Lemma $2 5$ to upper bound $\| P \| _ { L ^ { 2 } ( \mu _ { d } ) } = \| P \| _ { L ^ { 2 } ( \gamma _ { 1 } ) }$ and lower bound $\| P \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \gtrsim \| P \| _ { L ^ { 2 } ( \gamma _ { s } _ { - } ) }$ to obtain

$$
\| P \| _ { L ^ { 2 } ( \mu _ { d } ) } ^ { 2 } \leq e ^ { C N \log ( N + 2 ) } \| P \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } ^ { 2 } .
$$

This proves the bound for $I _ { N }$

It remains to control the operator diferences. Define

$$
R _ { N } ( { \pmb u } , { \pmb u } ^ { \prime } ) : = \sum _ { k > N } e ^ { - \eta ^ { k } } \frac { \langle { \pmb u } , { \pmb u } ^ { \prime } \rangle ^ { k } } { d ^ { k } } , \qquad K ( { \pmb u } , { \pmb u } ^ { \prime } ) - K _ { \leq N } ( { \pmb u } , { \pmb u } ^ { \prime } ) = R _ { N } ( { \pmb u } , { \pmb u } ^ { \prime } ) .
$$

For $\widetilde { \mu } \in \{ \mu _ { d } , \widehat { \mu } _ { d , n } \}$ , let $\textstyle { \boldsymbol { \mathscr { u } } } \sim { \widetilde { \mu } }$ and $\pmb { u } ^ { \prime } \sim \widehat { \mu } _ { d , n }$ be independent. Sub-Gaussian moment bounds give

$$
d ^ { - 1 } \| \langle { \pmb u } , { \pmb u } ^ { \prime } \rangle \| _ { L ^ { 2 k } ( \widetilde { \mu } \otimes \widehat { \mu } _ { d , n } ) } \leq d ^ { - 1 } \| \| { \pmb u } \| _ { 2 } \| _ { L ^ { 2 k } ( \widetilde { \mu } ) } \| \| { \pmb u } ^ { \prime } \| _ { 2 } \| _ { L ^ { 2 k } ( \widehat { \mu } _ { d , n } ) } \leq C k
$$

for some $C = C ( d , \sigma , \boldsymbol { X } )$ . Then, by Cauchy-Schwarz inequality and Minkowski’s inequality,

$$
\| \widehat K _ { 0 } - \widehat K _ { 0 , \le N } \| _ { \mathrm { o p } } \le \| R _ { N } \| _ { L ^ { 2 } ( \widehat \mu _ { d , n } \otimes \widehat \mu _ { d , n } ) } \le \sum _ { k > N } e ^ { - \eta ^ { k } } ( C k ) ^ { k } \le e ^ { - \eta ^ { N + 1 } + C N \log ( N + 2 ) } .
$$

A similar argument gives the analogous bound for $\| \mathcal { K } _ { 0 } - \mathcal { K } _ { 0 , \leq N } \| _ { \mathrm { o p } } .$

Recall that $\boldsymbol { \mathcal { S } } = \widehat { \mathcal { P } } \boldsymbol { \mathcal { S } } _ { 0 }$ and $\mathcal { T } = \mathcal { P T } _ { 0 }$ , where $\widehat { \mathcal P }$ and $\mathcal { P }$ are pullback operators from $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ to $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ and from $L ^ { 2 } ( \mu _ { d } )$ to $L ^ { 2 } ( \pi _ { d } )$ , respectively. The next lemma shows that, at training time $\mathrm { t } _ { N } = \lambda _ { N } ^ { - 2 }$ , gradient flow shadows the degree-N polynomial projection of its target.

Lemma 28. Under the specialization $\pi _ { 0 } = \gamma _ { 1 }$ , fix d $\geq 1 , n \geq 2$ , and a training realization $\{ \pmb { x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ For every $N \geq 2 _ { : }$ , let $\lambda _ { N }$ denote the smallest positive eigenvalue of $\widehat { \mathcal { K } } _ { 0 , \leq N . }$ , and let $\mathsf { P } _ { \mathcal { P } _ { N } }$ denote the $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ -orthogonal projection onto $\mathcal { P } _ { N }$ . Then, for every $v \in \mathbb { S } ^ { d - 1 }$ ,

$$
\operatorname* { l i m } _ { N \to \infty } \| \mathcal { T } _ { 0 } \widehat { f } _ { \mathrm { t } _ { N } , v } - \mathrm { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \mu _ { d } ) } = \operatorname* { l i m } _ { N \to \infty } \| \mathcal { T } \widehat { f } _ { \mathrm { t } _ { N } , v } - \mathcal { P } \mathrm { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \pi _ { d } ) } = 0 , \qquad \mathrm { t } _ { N } : = \lambda _ { N } ^ { - 2 } .
$$

Proof. The diagonal-integrability estimate (73) ensures that $ { \boldsymbol { S } } _ { 0 }$ and $\mathcal { T } _ { 0 }$ are bounded. Moreover, the tail bound in Lemma 27 gives

$$
\operatorname* { s u p } _ { N \geq 2 } \| \mathcal { K } _ { 0 , \leq N } \| _ { \mathrm { o p } } \leq \| \mathcal { K } _ { 0 } \| _ { \mathrm { o p } } + \operatorname* { s u p } _ { N \geq 2 } \| \mathcal { K } _ { 0 } - \mathcal { K } _ { 0 , \leq N } \| _ { \mathrm { o p } } < \infty .
$$

Since $\widehat { \mathcal P }$ is an isometry, we may rewrite (17) as

$$
\begin{array} { r } { \partial _ { \mathrm { t } } \widehat { f _ { \mathrm { t } , v } } = - S _ { 0 } ^ { * } ( S _ { 0 } \widehat { f _ { \mathrm { t } , v } } - \widehat { f _ { v } ^ { \star } } ) . } \end{array}
$$

Recall the spectral filter $q _ { \mathrm { t } }$ from (65). The solution formula therefore gives $\mathcal { T } _ { 0 } \widehat { f } _ { \mathrm { t } _ { N } , v } = \mathcal { K } _ { 0 } q _ { \mathrm { t } _ { N } } ( \widehat { \mathcal { K } } _ { 0 } ) \widehat { f } _ { v } ^ { \star }$ and we may decompose

$$
K _ { 0 } q _ { \mathrm { t } _ { N } } ( { \widehat K } _ { 0 } ) - K _ { 0 , \le N } q _ { \mathrm { t } _ { N } } ( { \widehat K } _ { 0 , \le N } ) = ( K _ { 0 } - K _ { 0 , \le N } ) q _ { \mathrm { t } _ { N } } ( { \widehat K } _ { 0 } ) + K _ { 0 , \le N } ( q _ { \mathrm { t } _ { N } } ( { \widehat K } _ { 0 } ) - q _ { \mathrm { t } _ { N } } ( { \widehat K } _ { 0 , \le N } ) ) .
$$

By (89),

$$
\| e ^ { - \mathsf { s } \widehat { \mathcal { K } } _ { 0 } } - e ^ { - \mathsf { s } \widehat { \mathcal { K } } _ { 0 , \leq N } } \| _ { \mathrm { o p } } \leq \mathsf { s } \| \widehat { \mathcal { K } } _ { 0 } - \widehat { \mathcal { K } } _ { 0 , \leq N } \| _ { \mathrm { o p } }
$$

for every ${ \mathsf s } \geq 0$ . Since $\begin{array} { r } { q _ { \mathrm { t } _ { N } } ( \widehat { \mathcal { K } } _ { 0 } ) = \int _ { 0 } ^ { \mathrm { t } _ { N } } e ^ { - s \widehat { \mathcal { K } } _ { 0 } } d s . } \end{array}$ , it follows that

$$
\begin{array} { r } { \| q _ { \mathrm { t } _ { N } } ( \widehat K _ { 0 } ) - q _ { \mathrm { t } _ { N } } ( \widehat K _ { 0 , \le N } ) \| _ { \mathrm { o p } } \le \mathtt t _ { N } ^ { 2 } \| \widehat K _ { 0 } - \widehat K _ { 0 , \le N } \| _ { \mathrm { o p } } , \qquad \| q _ { \mathrm { t } _ { N } } ( \widehat K _ { 0 } ) \| _ { \mathrm { o p } } \le \mathtt t _ { N } . } \end{array}
$$

Then, by Lemma 27 and the fact that $\| \mathcal { K } _ { 0 , \le N } \| _ { \mathrm { o p } }$ is bounded uniformly in $N$

$$
\begin{array} { r } { \| \mathcal { K } _ { 0 } q _ { \mathrm { t v } } ( \widehat K _ { 0 } ) - \mathcal { K } _ { 0 , \le N } q _ { \mathrm { t v } } ( \widehat K _ { 0 , \le N } ) \| _ { \mathrm { o p } } \le \mathrm { t } _ { N } \| \mathcal { K } _ { 0 } - \mathcal { K } _ { 0 , \le N } \| _ { \mathrm { o p } } + \| \mathcal { K } _ { 0 , \le N } \| _ { \mathrm { o p } } \mathrm { t } _ { N } ^ { 2 } \| \widehat K _ { 0 } - \widehat K _ { 0 , \le N } \| _ { \mathrm { o p } } } \\ { \lesssim e ^ { ( 4 - \eta ) \eta ^ { N } + C N \log ( N + 2 ) } \to 0 } \end{array}
$$

as $N \to \infty$ . Since $\| \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \leq 1$ by conditional Jensen’s inequality,

$$
\| \mathcal { K } _ { 0 } q _ { \mathrm { t } _ { N } } ( \widehat { \mathcal { K } } _ { 0 } ) \widehat { f } _ { v } ^ { \star } - \mathcal { K } _ { 0 , \le N } q _ { \mathrm { t } _ { N } } ( \widehat { \mathcal { K } } _ { 0 , \le N } ) \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \mu _ { d } ) } \to 0\tag{75}
$$

as $N \to \infty$

It remains to compare the truncated flow with the polynomial projection. Note that

$$
\widehat { \mathcal { K } } _ { 0 , \le N } q _ { \mathrm { t } _ { N } } ( \widehat { \mathcal { K } } _ { 0 , \le N } ) = \mathrm { i d } - e ^ { - \mathsf { t } _ { N } \widehat { \mathcal { K } } _ { 0 , \le N } } .
$$

Since $\mathrm { R a n } ( \widehat { \mathcal { K } } _ { 0 , \leq N } ) = \mathcal { P } _ { N }$ and $\widehat { \mathcal { K } } _ { 0 , \leq N }$ is self-adjoint, ke $\begin{array} { r } { \cdot ( \widehat { \mathcal { K } } _ { 0 , \leq N } ) = \mathcal { P } _ { N } ^ { \perp } } \end{array}$ , and therefore

$$
\widehat { \mathcal { K } } _ { 0 , \le N } q _ { \mathrm { t } _ { N } } ( \widehat { \mathcal { K } } _ { 0 , \le N } ) \widehat { f } _ { v } ^ { \star } - \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } = - e ^ { - \mathsf { t } _ { N } \widehat { \mathcal { K } } _ { 0 , \le N } } \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } .
$$

By definition of $\lambda _ { N }$ , it follows that

$$
\| \widehat { K } _ { 0 , \le N } q _ { \mathrm { t } _ { N } } ( \widehat { K } _ { 0 , \le N } ) \widehat { f } _ { v } ^ { \star } - \mathsf { P } _ { \mathcal P _ { N } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \le e ^ { - \mathsf { t } _ { N } \lambda _ { N } } \| \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \widehat { \mu } _ { d , n } ) } \le e ^ { - \mathsf { t } _ { N } \lambda _ { N } } .
$$

Recall the definition of $I _ { N }$ from Lemma 27. More precisely, if

$$
g _ { N } : = q _ { \mathrm { t } _ { N } } ( \widehat { K } _ { 0 , \le N } ) \widehat { f } _ { v } ^ { \star } , \qquad P _ { N } ( \boldsymbol { u } ) : = \int K _ { \le N } ( \boldsymbol { u } , \boldsymbol { u } ^ { \prime } ) g _ { N } ( \boldsymbol { u } ^ { \prime } ) \widehat { \mu } _ { d , n } ( \mathrm { d } \boldsymbol { u } ^ { \prime } ) ,
$$

then $P _ { N } \in \mathcal { P } _ { N }$ , while $\widehat { \mathcal { K } } _ { 0 , \le N } g _ { N }$ and ${ \cal K } _ { 0 , \le N } g _ { N }$ are the realizations of $P _ { N }$ in $L ^ { 2 } ( \widehat { \mu } _ { d , n } )$ and $L ^ { 2 } ( \mu _ { d } )$ respectively. Identifying $\mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star }$ with its (unique) polynomial representative, the definition of $I _ { N }$ applied to $P _ { N } - \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star }$ , gives

$$
\begin{array} { r l } & { \| K _ { 0 , \le N } q _ { \mathrm { t } _ { N } } ( \widehat K _ { 0 , \le N } ) \widehat f _ { v } ^ { \star } - \mathsf { P } _ { \mathcal P _ { N } } \widehat f _ { v } ^ { \star } \| _ { L ^ { 2 } ( \mu _ { d } ) } \le I _ { N } e ^ { - \mathsf { t } _ { N } \lambda _ { N } } } \\ & { \qquad \le \exp \Big ( C N \log ( N + 2 ) - e ^ { \eta ^ { N } - C N \log ( N + 2 ) } \Big ) \to 0 } \end{array}
$$

as $N \to \infty$ , where the last inequality follows from Lemma 27. Combining this with (75) yields

$$
\| \mathcal { T } _ { 0 } \widehat { f } _ { \mathrm { t } _ { N } , \pmb { v } } - \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { \pmb { v } } ^ { \star } \| _ { L ^ { 2 } ( \mu _ { d } ) } \to 0 .
$$

The upper bound on $\lambda _ { N }$ in Lemma 27 also gives $\lambda _ { N } \  \ 0$ , and hence $\mathrm { t } _ { N } = \lambda _ { N } ^ { - 2 } \to \infty$ . Since $\mathcal { T } = \mathcal { P T } _ { 0 }$ and $\mathcal { P }$ is an isometry, the second convergence follows immediately. □

We are finally ready to prove the main result of this subsection, which shows that the gradientflow risk can be unbounded for a universal kernel.

Proposition 6. Suppose that $\pi _ { 0 } = \gamma _ { 1 }$ . Fix $d \geq 1$ and $n \geq 2$ . Let K be the inner-product kernel associated with (72). Then K satisfies Assumption 5, and, for every fixed $v \in \mathbb { S } ^ { d - 1 }$ , almost surely over $\{ \pmb { x } _ { 0 , i } \} _ { i = 1 } ^ { n } \sim \pi _ { 0 } ^ { \otimes n } .$ ，

$$
\operatorname* { l i m } _ { \mathrm { t }  \infty } \mathcal { R } _ { v } ( \widehat { f } _ { \mathrm { t } , v } ) = \infty .
$$

Proof. All Taylor coeficients of h are strictly positive, so Assumption 5 holds. Let $\pmb { v } \in \mathbb { S } ^ { d - 1 }$ Since $\pi _ { 0 } = \gamma _ { 1 }$ , almost surely $\{ \langle \pmb { x } _ { 0 , i } , \pmb { v } \rangle \} _ { i = 1 } ^ { n }$ are pairwise distinct. Conditional on such a training realization, Lemma 26, applied with $r = 1$ , gives

$$
\operatorname* { s u p } _ { N } \| \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \mu _ { d } ) } = \infty .
$$

On the other hand, Lemma 28 gives a sequence $\mathrm { t } _ { N } \to \infty$ such that

$$
\| T \widehat { f } _ { \mathrm { t } _ { N } , v } - \mathcal { P } \mathsf { P } _ { \mathcal { P } _ { N } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \pi _ { d } ) } \to 0 .
$$

Choose a subsequence $N _ { \ell }$ along which $\| \mathsf { P } _ { \mathcal { P } _ { N _ { \ell } } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \mu _ { d } ) } \to \infty$ . Since $\mathcal { P } : L ^ { 2 } ( \mu _ { d } ) \to L ^ { 2 } ( \pi _ { d } )$ is an isometry,

$$
\begin{array} { r } { \| \mathcal { T } \widehat { f } _ { \mathrm { t } _ { N _ { \ell } } , v } \| _ { L ^ { 2 } ( \pi _ { d } ) } \geq \| \mathcal { P } \mathsf { P } _ { \mathcal { P } _ { N _ { \ell } } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \pi _ { d } ) } - \| \mathcal { T } \widehat { f } _ { \mathrm { t } _ { N _ { \ell } } , v } - \mathcal { P } \mathsf { P } _ { \mathcal { P } _ { N _ { \ell } } } \widehat { f } _ { v } ^ { \star } \| _ { L ^ { 2 } ( \pi _ { d } ) } \to \infty } \end{array}
$$

as $\ell \to \infty$ . Finally, $\| z _ { v } \| _ { L ^ { 2 } ( \pi _ { d } ) } = 1$ , and hence

$$
\mathcal { R } _ { v } ( \widehat { f } _ { \mathfrak { t } _ { N _ { \ell } } , v } ) ^ { 1 / 2 } = \Vert \mathcal { T } \widehat { f } _ { \mathfrak { t } _ { N _ { \ell } } , v } - z _ { v } \Vert _ { L ^ { 2 } ( \pi _ { d } ) } \geq \Vert \mathcal { T } \widehat { f } _ { \mathfrak { t } _ { N _ { \ell } } , v } \Vert _ { L ^ { 2 } ( \pi _ { d } ) } - 1 \to \infty
$$

as $\ell \to \infty$

## F.3 Test error of the empirical Bayes denoiser

We compute the population risk of the empirical Bayes denoiser (11) itself. To do so, we use the following “winner-take-all” lemma, which shows that the empirical Bayes weights are asymptotically concentrated on a single training point.

Lemma 29. Suppose that Assumptions 1 and 2 hold. Then,

$$
\mathbb { E } _ { ( { \pmb x } , z ) \sim \pi _ { d } } \left[ 1 - \sum _ { i = 1 } ^ { n } \omega _ { i } ( \rho { \pmb x } + \sigma z ) ^ { 2 } \ | \ { \pmb X } \right] \prec \frac { 1 } { \sqrt { d } } .
$$

Proof. Fix $\epsilon \in ( 0 , 1 / 2 )$ , and let ${ \mathcal { E } } _ { d }$ be the training-sample event on which

$$
\operatorname* { m a x } _ { i \in [ n ] } \left| \| x _ { 0 , i } \| _ { 2 } ^ { 2 } - \operatorname { T r } ( \Sigma ) \right| \leq d ^ { 1 / 2 + \epsilon } , \qquad \operatorname* { m a x } _ { i \neq j } | \langle x _ { 0 , i } , x _ { 0 , j } \rangle | \leq d ^ { 1 / 2 + \epsilon } .
$$

By the concentration bounds in (42), $\mathcal { E } _ { d }$ holds with very high probability. On $\mathcal { E } _ { d } ,$ Assumption 2 gives $\operatorname { T r } ( \pmb { \Sigma } ) \geq c d$ for all large $d ,$ and therefore

$$
\operatorname* { m i n } _ { i \neq j } \| \pmb { x } _ { 0 , i } - \pmb { x } _ { 0 , j } \| _ { 2 } ^ { 2 } \geq c ^ { \prime } d\tag{76}
$$

for some constant $c ^ { \prime } > 0$ . Write

$$
\omega _ { k } ( \rho x + \sigma z ) = \frac { \exp { \left( - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \| \mathbf { x } _ { 0 , k } \| ^ { 2 } + \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \langle \mathbf { x } _ { 0 , k } , \mathbf { x } \rangle + \frac { \rho } { \sigma } \langle \mathbf { x } _ { 0 , k } , z \rangle \right) } } { \sum _ { j = 1 } ^ { n } \exp { \left( - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \| \mathbf { x } _ { 0 , j } \| ^ { 2 } + \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \langle \mathbf { x } _ { 0 , j } , \mathbf { x } \rangle + \frac { \rho } { \sigma } \langle \mathbf { x } _ { 0 , j } , z \rangle \right) } }
$$

and set, for every $k \in [ n ]$

$$
S _ { k } \equiv S _ { k } ( z ; x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) : = - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \| { x } _ { 0 , k } \| ^ { 2 } + \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \langle { x } _ { 0 , k } , { x } \rangle + \frac { \rho } { \sigma } \langle { x } _ { 0 , k } , { z } \rangle .
$$

Let $S _ { ( 1 ) } \geq S _ { ( 2 ) }$ be the two largest values of $\{ S _ { k } \} _ { k = 1 } ^ { n }$ and set $\Delta : = S _ { ( 1 ) } - S _ { ( 2 ) }$ . By (76), the two largest values are attained at unique indices almost surely. Randomly partition $[ n ]$ , independently of $z _ { i }$ , into two balanced sets $A , B$ . Note that the two indices corresponding to $S _ { ( 1 ) }$ and $S _ { ( 2 ) }$ belong to diferent sets with probability at least $1 / 2$ . Therefore, for every $r \geq 0$

$$
\mathbb { P } ( \Delta \leq r \mid \boldsymbol { x } , \{ \boldsymbol { x } _ { 0 , i } \} _ { i = 1 } ^ { n } ) \leq 2 \mathbb { E } _ { \boldsymbol { A } , \boldsymbol { B } } \mathbb { P } \left( \left| \operatorname* { m a x } _ { k \in \boldsymbol { A } } S _ { k } - \operatorname* { m a x } _ { j \in \boldsymbol { B } } S _ { j } \right| \leq r \mid \boldsymbol { x } , \{ \boldsymbol { x } _ { 0 , i } \} _ { i = 1 } ^ { n } \right) .
$$

Conditionally on x and $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ , the random variables $\{ S _ { k } \} _ { k \in [ n ] }$ are jointly Gaussian with

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ S _ { k } \mid \displaystyle x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ] = - \frac { \rho ^ { 2 } } { 2 \sigma ^ { 2 } } \| x _ { 0 , k } \| ^ { 2 } + \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \langle x _ { 0 , k } , x \rangle , \qquad \mathrm { V a r } ( S _ { k } \mid x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) = \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \| x _ { 0 , k } \| ^ { 2 } , } \\ { \displaystyle \qquad \mathbb { C } \mathsf { o v } ( S _ { k } , S _ { j } \mid x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) = \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \langle x _ { 0 , k } , x _ { 0 , j } \rangle . } \end{array}
$$

In particular, on the event $\mathcal { E } _ { d } .$

$$
\left| \frac { \mathsf { C o v } ( S _ { k } , S _ { j } \mid x , \{ \pmb { x _ { 0 , i } } \} _ { i = 1 } ^ { n } ) } { \sqrt { \operatorname { V a r } ( S _ { k } \mid x , \{ \pmb { x _ { 0 , i } } \} _ { i = 1 } ^ { n } ) \operatorname { V a r } ( S _ { j } \mid x , \{ \pmb { x _ { 0 , i } } \} _ { i = 1 } ^ { n } ) } } \right| = \frac { | \langle \pmb { x _ { 0 , k } } , \pmb { x _ { 0 , j } } \rangle | } { \| \pmb { x _ { 0 , k } } \| _ { 2 } \| \pmb { x _ { 0 , j } } \| _ { 2 } } \lesssim d ^ { \epsilon - \frac 1 2 } < 1
$$

for all $k \neq j$ . Furthermore, for $k \neq j$ , the conditional correlation is bounded away from 1, and

$$
\sqrt { \operatorname { V a r } ( S _ { k } \mid x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) } - \frac { \operatorname { C o v } ( S _ { k } , S _ { j } \mid x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) } { \sqrt { \operatorname { V a r } ( S _ { k } \mid x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) } } = \frac { \rho } { \sigma } \left( \| x _ { 0 , k } \| _ { 2 } - \frac { \langle x _ { 0 , k } , x _ { 0 , j } \rangle } { \| x _ { 0 , k } \| _ { 2 } } \right) \gtrsim \frac { \rho } { \sigma } \sqrt { d } .
$$

The ratio in the first line equals one when $k = j$ . Therefore, using the anticoncentration inequality of [BFS24, Theorem $2 . 4 ]$ , it follows that, for every $r \geq 0$

$$
\mathbb { P } \left( \left| \operatorname* { m a x } _ { k \in \mathcal { A } } S _ { k } - \operatorname* { m a x } _ { j \in \mathcal { B } } S _ { j } \right| \leq r \mid \mathbf { x } , \{ \mathbf { x } _ { 0 , \downarrow } \} _ { i = 1 } ^ { n } \right) \lesssim \frac { \sigma } { \rho } \frac { \mathbb { E } [ \operatorname* { m a x } _ { k \in [ n ] } | \langle \mathbf { x } _ { 0 , k } , z \rangle | \mid x , \{ \mathbf { x } _ { 0 , i } \} _ { i = 1 } ^ { n } ] } { d } r \lesssim \frac { \sigma } { \rho } \sqrt { \frac { \log ( 2 n ) } { d } } r .
$$

The constants are independent of ${ \pmb x } , \ \{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ , and the balanced partition, and we applied the Gaussian maximal inequality to bound the expected maximum. Taking $r = L \log ( d )$ for a fixed constant $L > 2$ , and using $n \asymp d$ and $\rho / \sigma \geq c > 0$ for some constant $c > 0$ , we obtain

$$
\mathbb { P } ( \Delta \leq r \mid x , \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } ) \lesssim \frac { ( \log d ) ^ { 3 / 2 } } { \sqrt { d } } \prec \frac { 1 } { \sqrt { d } } .
$$

On the event $\{ \Delta ( z ; \pmb { x } ) > r \}$ , we have

$$
1 - \sum _ { i = 1 } ^ { n } \omega _ { i } ( \rho { \pmb x } + \sigma z ) ^ { 2 } \leq 1 - \omega _ { i _ { \star } } ( \rho { \pmb x } + \sigma z ) ^ { 2 } \leq 2 n e ^ { - r } ,
$$

where $i _ { \star }$ is an index attaining the maximum (which depends on $z , \ x ,$ and the training sample). Since $\begin{array} { r } { 0 \leq 1 - \sum _ { i = 1 } ^ { n } \omega _ { i } ( \rho { \pmb x } + \sigma z ) ^ { 2 } \leq 1 } \end{array}$ , it follows that

$$
\mathbb { E } \left[ 1 - \sum _ { i = 1 } ^ { n } \omega _ { i } ( \rho { \pmb x } + \sigma z ) ^ { 2 } \mid { \pmb x } , \{ \pmb x _ { 0 , i } \} _ { i = 1 } ^ { n } \right] \lesssim \frac { ( \log d ) ^ { 3 / 2 } } { \sqrt { d } } + d ^ { 1 - L } \prec \frac { 1 } { \sqrt { d } } .
$$

The bound is uniform in ${ \pmb x } .$ . Averaging over $\mathbf { \boldsymbol { x } } \ \sim \ \pi _ { 0 }$ and using that $\mathcal { E } _ { d }$ holds with very high probability concludes the proof. □

We are now ready to prove Theorem 3.

Proof of Theorem 3. Let $( \pmb { x } , z ) \sim \pi _ { d } ,$ set ${ \pmb u } = \rho { \pmb x } + \sigma { \pmb z }$ , and write

$$
M ( \pmb { u } ) : = \sum _ { i = 1 } ^ { n } \omega _ { i } ( \pmb { u } ) \pmb { x } _ { 0 , i } .
$$

By definition,

$$
\widehat { \pmb { f } } ^ { \star } ( \rho \pmb { x } + \sigma z ) + z = \frac { \rho } { \sigma } \big ( \pmb { M } ( \rho \pmb { x } + \sigma z ) - \pmb { x } \big ) .
$$

Therefore

$$
\mathcal { R } ( \widehat { \mathbf { f } } ^ { \star } ) = \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \left\{ \frac { 1 } { d } \mathbb { E } \| M ( \mathbf { \boldsymbol { u } } ) \| _ { 2 } ^ { 2 } - \frac { 2 } { d } \mathbb { E } \langle M ( \mathbf { \boldsymbol { u } } ) , \mathbf { \boldsymbol { x } } \rangle + \frac { 1 } { d } \mathbb { E } \| \mathbf { \boldsymbol { x } } \| _ { 2 } ^ { 2 } \right\} ,\tag{77}
$$

where the expectation is over the fresh pair $( x , z )$ , conditionally on the training data. The last term equals $\tau _ { \Sigma }$ . We show that the first term is $\tau _ { \Sigma } + o _ { d , \mathbb { P } } ( 1 )$ and the middle term is $o _ { d , \mathbb { P } } ( 1 )$ . Define

$$
\varepsilon _ { d } : = \operatorname* { m a x } _ { i \in [ n ] } \left| \frac { \| x _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } - \tau _ { \Sigma } \right| + \operatorname* { m a x } _ { i \neq j } \frac { \left| \langle x _ { 0 , i } , x _ { 0 , j } \rangle \right| } { d } , \qquad \varepsilon _ { d } ^ { \prime } : = {  { \mathbb E } } _ { ( x , z ) } \left[ 1 - \sum _ { i = 1 } ^ { n } \omega _ { i } ( u ) ^ { 2 } \mid \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } \right] .
$$

By (42) and Lemma 29, $\varepsilon _ { d } , \varepsilon _ { d } ^ { \prime } \prec d ^ { - 1 / 2 }$ . Since the weights are non-negative and sum to one,

$$
\frac { \left| \| M ( u ) \| _ { 2 } ^ { 2 } \right|} { d } - \tau _ { \Sigma } \sum _ { i = 1 } ^ { n } \omega _ { i } ( u ) ^ { 2 }  \leq \operatorname* { m a x } _ { i \in [ n ] } \left| \frac { \| x _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } - \tau _ { \Sigma } \right| \sum _ { i = 1 } ^ { n } \omega _ { i } ( u ) ^ { 2 } + \sum _ { i \neq j } \frac { \left| \langle x _ { 0 , i } , x _ { 0 , j } \rangle \right| } { d } \omega _ { i } ( u ) \omega _ { j } ( u )
$$

and therefore

$$
\left| \frac { 1 } { d } \mathbb { E } \big [ \| M ( u ) \| _ { 2 } ^ { 2 } \big | \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } \big ] - \tau _ { \Sigma } \right| \leq \varepsilon _ { d } + \tau _ { \Sigma } \mathbb { E } \Bigg [ 1 - \sum _ { i = 1 } ^ { n } \omega _ { i } ( u ) ^ { 2 } \Bigg | \{ x _ { 0 , i } \} _ { i = 1 } ^ { n } \Bigg ] = \varepsilon _ { d } + \tau _ { \Sigma } \varepsilon _ { d } ^ { \prime } \prec d ^ { - 1 / 2 } .
$$

It remains to control the cross term. Conditionally on the training data,

$$
\left| \frac { 1 } { d } \mathbb { E } \big [ \langle M ( \boldsymbol { u } ) , \boldsymbol { x } \rangle \mid \{ \boldsymbol { x } _ { 0 , i } \} _ { i = 1 } ^ { n } \big ] \right| \leq \frac { 1 } { d } \mathbb { E } _ { \boldsymbol { x } } \left[ \operatorname* { m a x } _ { i \in [ n ] } \left| \langle \pmb { x } _ { 0 , i } , \pmb { x } \rangle \right| \right] .
$$

Since $\mathbb { E } x = 0$ , the logarithmic Sobolev inequality and the Herbst argument give, conditionally on the training data,

$$
\mathbb { E } _ { \pmb { x } } \exp \left( \lambda \langle \pmb { x } _ { 0 , i } , \pmb { x } \rangle \right) \le \exp \left( \frac { C _ { \mathrm { L S I } } \lambda ^ { 2 } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { 2 } \right) .
$$

Applying the standard maximal inequality yields

$$
\mathbb { E } _ { \pmb { x } } \left[ \operatorname* { m a x } _ { i \in [ n ] } | \langle \pmb { x } _ { 0 , i } , \pmb { x } \rangle | \right] \lesssim ( \operatorname* { m a x } _ { i } \| \pmb { x } _ { 0 , i } \| _ { 2 } ) \sqrt { \log ( n ) } \prec \sqrt { d } .
$$

Here, we have used (42), $\operatorname { T r } ( \Sigma ) \lesssim d$ and the proportional limiting regime $n \asymp d$ to obtain the last bound. Substituting these estimates into (77) gives

$$
\left| \mathcal { R } ( \widehat { f } ^ { \star } ) - 2 \frac { \rho ^ { 2 } } { \sigma ^ { 2 } } \tau _ { \Sigma } \right| \prec \frac { 1 } { \sqrt { d } } .
$$

Under Assumption 5, Proposition 4 gives, for every fixed $d , n$ and training realization,

$$
\operatorname* { l i m } _ { t  \infty } \mathcal { R } ( \widetilde { f } _ { \mathrm { t } } ) = \mathcal { R } ( \widehat { f } ^ { \star } ) .
$$

Combining this identity with the preceding high-dimensional asymptotic proves the second assertion. □

## G Reverse process dynamics

In this section, we study the dynamics of the reverse difusion process and its linearization. We suppose throughout that $0 < t _ { 0 } < T$ are fixed, that $t \in [ t _ { 0 } , T ] \ \mapsto \ t _ { t } \in \mathbb { R } _ { \geq 0 }$ is a deterministic piecewise-continuous schedule, and that $\widetilde { \pmb { x } } _ { t }$ solves (33).

## G.1 Properties of the linearized reverse process

Unlike $\widehat { \mathcal { K } } _ { \mathrm { e f f } , t }$ , whose self-induced regularization is identical across training samples, the samplecorrected operator $\widetilde { \kappa } _ { t }$ contains the sample-dependent diagonal corrections $\widetilde { \delta } _ { i , t }$ . Consequently, it cannot be reduced to a spectral function of the empirical covariance $\widehat { \Sigma } .$ , and the 2d-dimensional representation of Lemma 19 no longer applies. Nevertheless, $\widetilde { \kappa } _ { t }$ preserves a larger $( n + d )$ -dimensional subspace, on which it admits an explicit matrix representation.

Define the subspace

$$
\widetilde { \gamma } : = \{ F _ { a , b } : a \in \mathbb { R } ^ { n } , \ b \in \mathbb { R } ^ { d } \} \subset L ^ { 2 } ( \widehat { \pi } _ { d , n } ) , \qquad F _ { a , b } ( x _ { 0 , i } , z ) : = \sqrt { n } a _ { i } + \langle b , z \rangle .\tag{78}
$$

Lemma 30. For every $t \in [ t _ { 0 } , T ] , \widetilde { \mathcal { K } } _ { t }$ maps $\widetilde { \nu }$ to itself, and the linearized score function $\widetilde { S } _ { t }$ admits the representation

$$
\widetilde { S } _ { t } ( { \pmb x } ) = \frac { 1 } { \sigma _ { t } } \widetilde { \Psi } _ { t } { \pmb x }
$$

where

$$
\tilde { \Psi } _ { t } : = - \frac { 1 } { \sigma _ { t } } \left[ 0 \quad \mathbf { I } _ { d } \right] \left( \mathbf { I } _ { n + d } - e ^ { - \mathrm { t } _ { t } \widetilde { K } _ { t } } \right) \left[ \mathbf { I } _ { d } \right] , \qquad \widetilde { K } _ { t } : = \left[ \begin{array} { c c } { \widetilde { D } _ { t } + \frac { \rho _ { t } ^ { 2 } } { d n } X ^ { \top } X } & { \frac { \rho _ { t } \sigma _ { t } } { d \sqrt { n } } X ^ { \top } } \\ { \frac { \rho _ { t } \sigma _ { t } } { d \sqrt { n } } X } & { \frac { \sigma _ { t } ^ { 2 } } { d } \mathbf { I } _ { d } } \end{array} \right] ,
$$

and $\widetilde { D } _ { t } : = \mathrm { d i a g } ( \widetilde { \delta } _ { 1 , t } , \ldots , \widetilde { \delta } _ { n , t } )$ . Furthermore, $- \sigma _ { t } ^ { - 1 } \mathbf { I } _ { d } \preceq \tilde { \Psi } _ { t } \preceq 0$

Proof. For every $F _ { a , b } \in \widetilde { \mathcal { V } }$

$$
\widetilde { K } _ { t } F _ { a , b } ( x _ { 0 , i } , z ) = \sqrt { n } \left( \widetilde { \delta } _ { i , t } a _ { i } + \frac { \rho _ { t } ^ { 2 } } { d n } \langle x _ { 0 , i } , X a \rangle + \frac { \rho _ { t } \sigma _ { t } } { d \sqrt { n } } \langle x _ { 0 , i } , b \rangle \right) + \left. \frac { \rho _ { t } \sigma _ { t } } { d \sqrt { n } } X a + \frac { \sigma _ { t } ^ { 2 } } { d } b , z \right. .
$$

In other words, $\widetilde { \kappa } _ { t }$ maps $\widetilde { \mathcal { V } }$ to itself: $\widetilde { \mathcal { K } } _ { t } F _ { a , b } = F _ { a ^ { \prime } , b ^ { \prime } }$ with $( ( \boldsymbol { a } ^ { \prime } ) ^ { \top } , ( \boldsymbol { b } ^ { \prime } ) ^ { \top } ) ^ { \top } = \widetilde { K } _ { t } ( ( \boldsymbol { a } ) ^ { \top } , ( \boldsymbol { b } ) ^ { \top } ) ^ { \top }$

Let $\pmb { v } \in \mathbb { R } ^ { d }$ and ${ \mathsf s } \geq 0$ . Note that $z _ { v } = F _ { 0 , - v } \in \bar { \mathcal { V } }$ , and therefore

$$
e ^ { - { \bf s } \widetilde { \cal K } _ { t } } { \bf z } _ { v } = F _ { a _ { t , { \bf s } } ( v ) , b _ { t , { \bf s } } ( v ) } , \qquad \left[ { \bf \overline { { { a _ { t , { \bf s } } } } } } ( v ) \right] = e ^ { - { \bf s } \widetilde { \cal K } _ { t } } \left[ { \bf \nabla } _ { - } ^ { 0 } \right] .
$$

Since $F _ { a , b }$ depends linearly on $( a , b )$ , we can push the integral inside the finite-dimensional representation:

$$
\int _ { 0 } ^ { \mathrm { t } _ { t } } e ^ { - \mathsf { s } \widetilde { K } _ { t } } z _ { v } \mathrm { d } \mathsf { s } = F _ { \overline { { a } } _ { t } ( v ) , \overline { { b } } _ { t } ( v ) } , \qquad \left[ \overline { { \overline { { b } } } } _ { t } ( v ) \right] = \int _ { 0 } ^ { \mathrm { t } _ { t } } e ^ { - \mathsf { s } \widetilde { K } _ { t } } \mathrm { d } \mathsf { s } \left[ { 0 \atop - v } \right] .
$$

For every $F _ { a , b } \in \widetilde { \mathcal { V } }$

$$
S _ { \mathrm { e f f } , t } ^ { * } F _ { a , b } ( \pmb { x } ) = \frac { 1 } { d } \langle \pmb { x } , \frac { \rho _ { t } } { \sqrt { n } } \pmb { X } \pmb { a } + \sigma _ { t } \pmb { b } \rangle .
$$

Applying this to the integrated finite-dimensional representation gives

$$
\widetilde { S } _ { t , v } ( \boldsymbol { x } ) = \frac { 1 } { \sigma _ { t } d } \left. \boldsymbol { x } , \left[ \frac { \rho _ { t } } { \sqrt { n } } \boldsymbol { X } \quad \sigma _ { t } \mathbf { I } _ { d } \right] \int _ { 0 } ^ { \mathbf { t } _ { t } } e ^ { - s \widetilde { K } _ { t } } \mathrm { d } s \left[ - v \right] \right. .
$$

Summing over an orthonormal basis $\{ \pmb { v } _ { \ell } \} _ { \ell = 1 } ^ { d } \ \mathrm { o f } \ \mathbb { R } ^ { d }$ gives the matrix representation $\widetilde { S } _ { t } ( \pmb { x } ) = \sigma _ { t } ^ { - 1 } \widetilde { \Psi } _ { t } \pmb { x }$ with

$$
\widetilde { \Psi } _ { t } : = \frac { 1 } { d } \left[ \frac { \rho _ { t } } { \sqrt { n } } \pmb { X } _ { \sigma _ { t } } \mathbf { I } _ { d } \right] \int _ { 0 } ^ { \mathbf { t } _ { t } } e ^ { - \pmb { \mathrm { s } } \widetilde { \pmb { K } } _ { t } } \mathrm { d } \mathbf { s } \left[ - \mathbf { I } _ { d } \right] .
$$

Observe that

$$
\left[ \frac { \rho _ { t } } { \sqrt { n } } { \cal X } \quad \sigma _ { t } { \bf I } _ { d } \right] = \frac { d } { \sigma _ { t } } \left[ 0 \quad { \bf I } _ { d } \right] \widetilde { \cal K } _ { t } ,
$$

and therefore we may integrate the matrix exponential to obtain the alternative representation.

It is immediate from the representation that $\widetilde { \Psi } _ { t }$ is symmetric. In order to establish the eigenvalue bounds, we decompose $\widetilde { \pmb { K } _ { t } }$ as

$$
\widetilde { \mathbf { K } } _ { t } = \frac { 1 } { d } \left[ \frac { \rho _ { t } } { \sqrt { n } } \mathbf { X } ^ { \mathsf { T } } \right] \left[ \frac { \rho _ { t } } { \sqrt { n } } \mathbf { X } \quad \sigma _ { t } \mathbf { I } _ { d } \right] + \left[ \overset { \widetilde { D } _ { t } } { 0 } \mathbf { \Lambda } 0 \right] .
$$

Since $\widetilde { \delta } _ { i , t } \geq 0$ for all $i \in [ n ]$ by Remark C.2, it follows that $\widetilde { \pmb { K } } _ { t } \succeq 0$ . Consequently, for every ${ \mathsf s } \geq 0$ 2 $\begin{array} { r } { 0 \preceq e ^ { - \mathsf { s } K _ { t } } \preceq \mathbf { I } _ { n + d } . } \end{array}$ , and the result follows from the representation of $\widetilde { \Psi } _ { t }$ □

As a consequence of the representation in Lemma 30, it follows that, conditionally on the training data X, the linearized reverse process $\widetilde { \pmb { x } } _ { t }$ is a centered Gaussian process.

Next, we establish uniform bounds on the linearized reverse process $\widetilde { \pmb { x } } _ { t }$ and its inner products with the training data $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$

Lemma 31 (Boundedness and delocalization of the linearized process). Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 6 hold. Then, conditionally on the training data, for every fixed $k \in \mathbb N$

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widetilde { \pmb x } _ { t } \| _ { 2 } ^ { k } \ | \textbf { \textup { X } } \right] ^ { 1 / k } \lesssim \sqrt { d } , \qquad \mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { m a x } _ { i \leq n } | \langle \widetilde { \pmb x } _ { t } , \pmb x _ { 0 , i } \rangle | ^ { k } \ | \textbf { \textup { X } } \right] ^ { 1 / k } \prec \sqrt { d } .
$$

Furthermore, there exists a constant $C > 0$ such that for every $s > 0$ 2

$$
\mathbb { E } \left[ \exp \left( \frac { s } { d } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widetilde { \pmb x } _ { t } \| _ { 2 } ^ { 2 } \right) | \mathbf { \cal X } \right] \le 3 e ^ { 2 s C }
$$

whenever $d \ge 8 s C$

Proof. Reverse time by setting $s = T - t \in [ 0 , T - t _ { 0 } ]$ and $y _ { s } : = \widetilde { x } _ { T - s }$ . Then

$$
\mathrm { d } y _ { s } = D _ { T - s } y _ { s } \mathrm { d } s + \sqrt { 2 } \mathrm { d } W _ { s } , \qquad D _ { t } : = \mathbf { I } _ { d } + \frac { 2 } { \sigma _ { t } } \widetilde { \Psi } _ { t } , \qquad y _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) .
$$

Let $\Phi ( s , u )$ denote the state-transition matrix for this forward-time linear drift:

$$
\partial _ { s } \Phi ( s , u ) = D _ { T - s } \Phi ( s , u ) , \qquad \Phi ( u , u ) = { \bf I } _ { d } .
$$

By the spectral bound on $\widetilde { \Psi } _ { t }$ from Lemma 30, $\begin{array} { r } { \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| D _ { t } \| _ { \mathrm { o p } } \leq 1 + 2 \sigma _ { t _ { 0 } } ^ { - 2 } = : C _ { D } } \end{array}$ . By Gronwall’s inequality,

$$
\| \Phi ( s , u ) \| _ { \mathrm { o p } } \leq e ^ { C _ { D } ( s - u ) } , \qquad 0 \leq u \leq s \leq T - t _ { 0 } .\tag{79}
$$

The process admits the variation-of-constants representation

$$
\pmb { y } _ { s } = \pmb { \Phi } ( s , 0 ) \pmb { y } _ { 0 } + \sqrt { 2 } \int _ { 0 } ^ { s } \pmb { \Phi } ( s , u ) \mathrm { d } \pmb { W } _ { u } .\tag{80}
$$

Applying the Itˆo isometry to (80),

$$
\mathbb { E } [ \pmb { y } _ { s } \pmb { y } _ { s } ^ { \top } \mid \pmb { X } ] = \Phi ( s , 0 ) \Phi ( s , 0 ) ^ { \top } + 2 \int _ { 0 } ^ { s } \Phi ( s , u ) \Phi ( s , u ) ^ { \top } \mathrm { d } u
$$

and hence

$$
\operatorname* { s u p } _ { s \in [ 0 , T - t _ { 0 } ] } \| \mathbb { E } [ \pmb { y } _ { s } \pmb { y } _ { s } ^ { \mathsf { T } } \mid \pmb { X } ] \| _ { \mathrm { o p } } \leq e ^ { 2 C _ { D } T } + 2 T e ^ { 2 C _ { D } T } = : C _ { \mathrm { c o v } } .\tag{81}
$$

In particular, the covariance of $\widetilde { \pmb { x } } _ { t }$ conditionally on the input data $\{ { \pmb x } _ { 0 , i } \} _ { i = 1 } ^ { n }$ , is bounded uniformly in t and $d .$

We next prove delocalization. Consider the centered Gaussian process

$$
G ( i , s ) : = \langle y _ { s } , x _ { 0 , i } \rangle , \qquad ( i , s ) \in [ n ] \times [ 0 , T - t _ { 0 } ] ,
$$

conditionally on X, and let dist<sub>G</sub> be its canonical metric. We first control temporal increments. For $0 \leq u \leq s \leq T - t _ { 0 }$ 2

$$
{ \pmb y } _ { s } = \pmb { \Phi } ( s , u ) { \pmb y } _ { u } + \sqrt { 2 } \int _ { u } ^ { s } { \pmb { \Phi } } ( s , r ) \mathrm { d } { \pmb W } _ { r } ,
$$

and the martingale increment is independent of $\mathbf { \nabla } _ { \pmb { y } } _ { u }$ . Thus

$$
\begin{array} { r l } & { \mathrm { d i s t } _ { G } ( ( i , s ) , ( i , u ) ) ^ { 2 } = \mathbb { E } [ |  ( \Phi ( s , u ) - \mathbf { I } _ { d } ) y _ { u } , \pmb { x } _ { 0 , i }  + \sqrt { 2 } \int _ { u } ^ { s } \langle \Phi ( s , r ) ^ { \top } \pmb { x } _ { 0 , i } , \mathrm { d } W _ { r } \rangle | ^ { 2 } | X ] } \\ & { \qquad \leq \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } ( \| \Phi ( s , u ) - \mathbf { I } _ { d } \| _ { \mathrm { o p } } ^ { 2 } C _ { \mathrm { c o v } } + 2 \int _ { u } ^ { s } \| \Phi ( s , r ) \| _ { \mathrm { o p } } ^ { 2 } \mathrm { d } r ) . } \end{array}
$$

By the fundamental theorem of calculus and (79),

$$
\| \Phi ( s , u ) - \mathbf { I } _ { d } \| _ { \mathrm { o p } } \leq \int _ { u } ^ { s } \| D _ { T - r } \| _ { \mathrm { o p } } \| \Phi ( r , u ) \| _ { \mathrm { o p } } \mathrm { d } r \leq C _ { D } e ^ { C _ { D } T } ( s - u ) .
$$

By (40) and (42), there exists a constant $C > 0$ such that the event ma $\mathfrak { c } _ { i \leq n } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } \leq C d$ holds with very high probability. We condition on this event. Since $( s - u ) ^ { 2 } \leq T ( s - u )$ , the preceding bounds imply

$$
\mathrm { d i s t } _ { G } ( ( i , s ) , ( i , u ) ) \leq C _ { \mathrm { t i m e } } \sqrt { d } | s - u | ^ { 1 / 2 }\tag{82}
$$

for some constant $0 < C _ { \mathrm { t i m e } } < \infty$ . For spatial increments at fixed time,

$$
\operatorname { d i s t } _ { G } ( ( i , u ) , ( j , u ) ) ^ { 2 } = \mathbb { E } [ \langle y _ { u } , x _ { 0 , i } - x _ { 0 , j } \rangle ^ { 2 } \mid X ] \leq C _ { \mathrm { c o v } } \| \boldsymbol { x } _ { 0 , i } - \boldsymbol { x } _ { 0 , j } \| _ { 2 } ^ { 2 } .\tag{83}
$$

Combining (82) and (83), dist<sub>G</sub> is dominated by the metric

$$
\begin{array} { r } { \rho ( ( i , s ) , ( j , u ) ) : = C \left( \sqrt { d } | s - u | ^ { 1 / 2 } + \| \pmb { x } _ { 0 , i } - \pmb { x } _ { 0 , j } \| _ { 2 } \right) , } \end{array}
$$

where C is a constant depending on $C _ { \mathrm { c o v } }$ and $C _ { \mathrm { t i m e } }$ . Its diameter is $O _ { d } ( \sqrt { d } )$ , and its covering numbers obey

$$
N ( [ n ] \times [ 0 , T - t _ { 0 } ] , \rho , r ) \leq n \left( 1 + \frac { C d } { r ^ { 2 } } \right) .
$$

Dudley’s entropy integral therefore gives

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { i , s } \left| G ( i , s ) \right| \mid X \right] \lesssim { \sqrt { d } } { \sqrt { \log d } } .
$$

Moreover, su $\operatorname { p } _ { i , s } \mathbb { E } [ G ( i , s ) ^ { 2 } \mid { \cal X } ] \le C d$ . Borell–TIS upgrades the expectation bound to every fixed moment:

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { s \in [ 0 , T - t _ { 0 } ] } \operatorname* { m a x } _ { i \leq n } | G ( i , s ) | ^ { k } \mid X \right] ^ { 1 / k } \leq C _ { k } { \sqrt { d } } { \sqrt { \log d } } .
$$

This implies the stated stochastic domination for the overlaps $\langle \widetilde { \pmb { x } } _ { t } , \pmb { x } _ { 0 , i } \rangle$

The norm bound follows from the same chaining argument applied to $G ^ { \prime } ( { \pmb v } , s ) : = \langle { \pmb v } , { \pmb y } _ { s } \rangle$ , indexed by $\mathbb { S } ^ { d - 1 } \times [ 0 , T - t _ { 0 } ]$ . The canonical metric is bounded by

$$
\mathrm { d i s t } _ { G ^ { \prime } } ( ( \pmb { v } , s ) , ( \pmb { u } , \pmb { u } ) ) \leq C \left( | s - \pmb { u } | ^ { 1 / 2 } + \| \pmb { v } - \pmb { u } \| _ { 2 } \right) ,
$$

and the standard volumetric covering estimate for $\mathbb { S } ^ { d - 1 }$ gives

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { \substack { { \boldsymbol v } \in \mathbb { S } ^ { d - 1 } , { \boldsymbol s } \in [ 0 , T - t _ { 0 } ] } } G ^ { \prime } ( \pmb { v } , { \boldsymbol s } ) \mid \boldsymbol { X } \right] \lesssim \sqrt { d } .
$$

Together with the variance bound $\begin{array} { r } { \operatorname* { s u p } _ { \pmb { v } , s } \mathbb { E } [ G ^ { \prime } ( \pmb { v } , s ) ^ { 2 } ~ | ~ \pmb { X } ] \leq C _ { \mathrm { c o v } } } \end{array}$ , Borell–TIS gives the claimed fixed-moment bound for $\operatorname* { s u p } _ { s } \| y _ { s } \| _ { 2 }$ , equivalently for $\operatorname { s u p } _ { t } \| { \widetilde { \pmb x } } _ { t } \| _ { 2 }$

Borell–TIS also gives the probability bound

$$
\mathbb { P } \left( \operatorname* { s u p } _ { s \in [ 0 , T - t _ { 0 } ] } \| y _ { s } \| _ { 2 } \geq \mathbb { E } \left[ \operatorname* { s u p } _ { s \in [ 0 , T - t _ { 0 } ] } \| y _ { s } \| _ { 2 } \mid X \right] + u \mid X \right) \leq 2 \exp \left( - \frac { u ^ { 2 } } { 2 C _ { \mathrm { c o v } } } \right)\tag{84}
$$

for all $u > 0$ . Applying the same proof as in Lemma 1, replacing the covariance bound (40) by (81), the Gaussian tails by (84) and the expectation bound by the chaining bound, we obtain the exponential moment bound. □

## G.2 Uniform linearization of the empirical kernel operator

The fixed-time linearization lemmas control the training and test risks. For the reverse process we need the corresponding statement along the random, but delocalized, Gaussian path generated by (33).

It will be convenient to introduce the following notation. For any $t > 0 , \pmb { x } , \pmb { x } ^ { \prime } , z \in \mathbb { R } ^ { d }$ , define

$$
\phi _ { t , \pmb { x } } ( \pmb { x } ^ { \prime } , z ) : = K ( \pmb { x } , \rho _ { t } \pmb { x } ^ { \prime } + \sigma _ { t } z ) , \qquad \phi _ { \mathrm { e f f } , t , \pmb { x } } ( \pmb { x } ^ { \prime } , z ) : = \frac { \langle \pmb { x } , \rho _ { t } \pmb { x } ^ { \prime } + \sigma _ { t } z \rangle } { d } .\tag{85}
$$

The next lemma shows that, along the delocalized path $\widetilde { \pmb { x } } _ { t }$ , the feature map $\phi _ { t , \widetilde { \pmb { x } } _ { t } }$ is well-approximated by the linearized feature map $\phi _ { \mathrm { e f f } , t , \widetilde { \pmb x } _ { t } }$

Lemma 32. Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 6 hold. Then, for every fixed $k \in \mathbb N$

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in \left[ t _ { 0 } , T \right] } \| \phi _ { t , \widetilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \widetilde { x } _ { t } } \| _ { L ^ { 2 } \left( \widehat { \pi } _ { d , n } \right) } ^ { 2 k } \bigm | \ X \right] ^ { 1 / k } \prec { d ^ { - 5 } } .
$$

Proof. Let $U _ { i , t } ( z ) : = \langle \widetilde { \pmb { x } } _ { t } , \rho _ { t } \pmb { x } _ { 0 , i } + \sigma _ { t } \pmb { z } \rangle / d .$ . Taylor’s theorem and Assumption 6 give

$$
| h ( u ) - u | ^ { 2 } \leq C e ^ { C | u | } | u | ^ { 1 0 }
$$

for some constant $C > 0$ , and hence Cauchy-Schwarz gives

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \phi _ { t , \widetilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \widetilde { x } _ { t } } \| _ { L ^ { 2 } ( \widetilde { \pi } _ { d , n } ) } ^ { 2 } \leq C \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \mathbb { E } _ { z } [ e ^ { 2 C | U _ { i , t } ( z ) | } ] ^ { 1 / 2 } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \mathbb { E } _ { z } [ | U _ { i , t } ( z ) | ^ { 2 0 } ] ^ { 1 / 2 } .\tag{86}
$$

For the exponential term, conditionally on $\widetilde { \pmb { x } } _ { t } , \ \langle \widetilde { \pmb { x } } _ { t } , z \rangle$ is Gaussian with variance $\| \widetilde { \pmb { x } } _ { t } \| _ { 2 } ^ { 2 }$ so

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \mathbb { E } _ { z } [ e ^ { 2 C | U _ { i , t } ( z ) | } ] \lesssim \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \exp \left( \frac { C \rho _ { t } | \langle x _ { 0 , i } , \widetilde { x } _ { t } \rangle | } { d } + \frac { 2 C ^ { 2 } \sigma _ { t } ^ { 2 } \| \widetilde { x } _ { t } \| _ { 2 } ^ { 2 } } { d ^ { 2 } } \right) .
$$

By Young’s inequality and, conditionally on the very high probability event $\{ \operatorname* { m a x } _ { i \leq n } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } \lesssim d \}$ ,

$$
\exp \left( \frac { C \rho _ { t } | \langle \pmb { x } _ { 0 , i } , \widetilde { \pmb { x } } _ { t } \rangle | } { d } \right) \lesssim \exp \left( \frac { C ^ { \prime } \| \widetilde { \pmb { x } } _ { t } \| _ { 2 } ^ { 2 } } { 2 d } \right)
$$

for some constant $C ^ { \prime } > 0$ . By the exponential bound in Lemma 31, $\mathbb { E } [ \exp ( \operatorname* { s u p } _ { t } u \| \widetilde { \mathbf { x } } _ { t } \| _ { 2 } ^ { 2 } / d ) \mid X ]$ is bounded for every fixed $u > 0$ and d large enough depending on u. Hence, for every fixed $m \in \mathbb { N }$ we get

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \mathbb { E } _ { z } [ e ^ { 2 C | U _ { i , t } ( z ) | } ] ^ { m / 2 } \mid X \right] \lesssim 1 .
$$

For the polynomial term, conditionally on $\widetilde { \pmb { x } } _ { t } , \ \langle \widetilde { \pmb { x } } _ { t } , z \rangle$ is Gaussian with variance $\| \widetilde { \pmb { x } } _ { t } \| _ { 2 } ^ { 2 }$ so

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \mathbb { E } _ { z } [ | U _ { i , t } ( z ) | ^ { 2 0 } ] ^ { 1 / 2 0 } \lesssim \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \left( \frac { \rho _ { t } | \langle x _ { 0 , i } , \widetilde { x } _ { t } \rangle | } { d } + \frac { \sigma _ { t } \| \widetilde { x } _ { t } \| _ { 2 } } { d } \right) .
$$

For every fixed $m \in \mathbb { N } .$ , by Lemma 31, it follows that

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , i \in [ n ] } \mathbb { E } _ { z } [ | U _ { i , t } ( z ) | ^ { 2 0 } ] ^ { m / 2 } \mid X \right] \prec d ^ { - 5 m } .
$$

Taking $m \ : = \ : 2 k$ in the preceding two estimates, raising (86) to the k-th power, and applying conditional Cauchy–Schwarz yields the desired bound. □

In addition to Lemma 32, we will also need a uniform-in-time linearization of the empirical kernel operator $\widehat { \boldsymbol { K } } _ { t } = \boldsymbol { S } _ { t } \boldsymbol { S } _ { t } ^ { * }$ , with $S _ { t }$ defined in (15) for difusion time $t > 0$ . For every $t > 0$ decompose

$$
\widehat { \mathcal { K } } _ { t } - \widetilde { \mathcal { K } } _ { t } = \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ,
$$

where the operators $\Delta _ { \mathrm { o d } , t } , \Delta _ { \mathrm { d } , t } : L ^ { 2 } ( \widehat { \pi } _ { d , n } ) \to L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ are defined by

$$
\Delta _ { \mathrm { o d } , t } f ( \boldsymbol { x } _ { 0 , i } , \boldsymbol { z } ) = \frac { 1 } { n } \sum _ { j \neq i } \mathbb { E } _ { \boldsymbol { z } ^ { \prime } } \left[ D _ { i j , t } ( \boldsymbol { z } , \boldsymbol { z } ^ { \prime } ) f ( \boldsymbol { x } _ { 0 , j } , \boldsymbol { z } ^ { \prime } ) \right] , \quad \Delta _ { \mathrm { d } , t } f ( \boldsymbol { x } _ { 0 , i } , \boldsymbol { z } ) = \frac { 1 } { n } \mathbb { E } _ { \boldsymbol { z } ^ { \prime } } \left[ D _ { i i , t } ( \boldsymbol { z } , \boldsymbol { z } ^ { \prime } ) f ( \boldsymbol { x } _ { 0 , i } , \boldsymbol { z } ^ { \prime } ) \right]
$$

with

$$
D _ { i , j , t } ( z , z ^ { \prime } ) : = h ( U _ { i j , t } ( z , z ^ { \prime } ) ) - U _ { i j , t } ( z , z ^ { \prime } ) - \mathbb { 1 } _ { i = j } n \widetilde { \delta } _ { i , t } , \qquad U _ { i j , t } ( z , z ^ { \prime } ) : = \frac { \langle \rho _ { t } x _ { 0 , i } + \sigma _ { t } z , \rho _ { t } x _ { 0 , j } + \sigma _ { t } z ^ { \prime } \rangle } { d } .
$$

We have the following uniform-in-time linearization lemma for the diagonal and of-diagonal kernel operator diferences.

Lemma 33. Suppose that Assumptions 1, 2, and 6 hold. Then

$$
\operatorname* { s u p } _ { t > 0 } \| \Delta _ { \mathrm { o d } , t } \| _ { \mathrm { o p } } \prec d ^ { - 5 / 2 } , \qquad \operatorname* { s u p } _ { t > 0 } \| \Delta _ { \mathrm { d } , t } \| _ { \mathrm { o p } } \prec d ^ { - 3 / 2 } .
$$

Proof. We repeat the proof of Proposition 1, keeping the estimates uniform in t. This uniformity follows from $0 \leq \rho _ { t } , \sigma _ { t } \leq 1$ . Indeed, for $i \neq j$

$$
\operatorname* { s u p } _ { t > 0 } | U _ { i j , t } ( z , z ^ { \prime } ) | \leq \frac { | \langle { \pmb x } _ { 0 , i } , { \pmb x } _ { 0 , j } \rangle | + | \langle { \pmb x } _ { 0 , i } , { z ^ { \prime } } \rangle | + | \langle { \pmb z } , { \pmb x } _ { 0 , j } \rangle | + | \langle { \pmb z } , { z ^ { \prime } } \rangle | } { d } .
$$

Thus the standard overlap and exponential-moment bounds used in the proof of Proposition 1 hold with a common, time-independent majorant.

For $i \neq j$ , Assumption 6 gives

$$
h ( U _ { i j , t } ) - U _ { i j , t } = \frac { h ^ { ( 5 ) } ( \xi _ { i j , t } ) } { 5 ! } U _ { i j , t } ^ { 5 } .
$$

Consequently,

$$
\operatorname* { s u p } _ { t > 0 } \operatorname* { m a x } _ { i \neq j } \mathbb { E } _ { z , z ^ { \prime } } | h ( U _ { i j , t } ) - U _ { i j , t } | ^ { 2 } \prec d ^ { - 5 } .
$$

Applying the of-diagonal Cauchy–Schwarz estimate from the proof of Proposition 1 yields

$$
\operatorname* { s u p } _ { t > 0 } \| \Delta _ { \mathrm { o d } , t } \| _ { \mathrm { o p } } \prec d ^ { - 5 / 2 } .
$$

For the diagonal blocks, we apply the mean-value theorem to obtain

$$
D _ { i i , t } ( z , z ^ { \prime } ) = ( h ^ { \prime } ( \xi _ { i , t } ( z , z ^ { \prime } ) ) - 1 ) \left( \frac { \rho _ { t } \sigma _ { t } } { d } \langle { \bf { x } } _ { 0 , i } , z \rangle + \frac { \rho _ { t } \sigma _ { t } } { d } \langle { \bf { x } } _ { 0 , i } , z ^ { \prime } \rangle + \frac { \sigma _ { t } ^ { 2 } } { d } \langle z , z ^ { \prime } \rangle \right) ,
$$

where $\xi _ { i , t }$ lies between $U _ { i i , t }$ and $\rho _ { t } ^ { 2 } \lVert \mathbf { x } _ { 0 , i } \rVert _ { 2 } ^ { 2 } / d$ . For every fixed $m \geq 2$

$$
\operatorname* { s u p } _ { t > 0 } \operatorname* { m a x } _ { i \leq n } \mathbb { E } _ { z , z ^ { \prime } } \left[ \left| \frac { \rho _ { t } \sigma _ { t } } { d } \langle x _ { 0 , i } , z \rangle + \frac { \rho _ { t } \sigma _ { t } } { d } \langle x _ { 0 , i } , z ^ { \prime } \rangle + \frac { \sigma _ { t } ^ { 2 } } { d } \langle z , z ^ { \prime } \rangle \right| ^ { m } \right] ^ { 1 / m } \prec d ^ { - 1 / 2 }
$$

by the same argument as in the proof of Proposition 1. Moreover, since $\xi _ { i , t }$ lies between $U _ { i i , t }$ and $\rho _ { t } ^ { 2 } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d ,$

$$
| \xi _ { i , t } | \leq \rho _ { t } ^ { 2 } \frac { \| { \pmb x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } + \left| U _ { i i , t } - \rho _ { t } ^ { 2 } \frac { \| { \pmb x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right| .
$$

Therefore, the exponential-moment bound from Lemma 1 and max<sub>i</sub> $\| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d \lesssim 1$ with very high probability imply that, for every fixed $s > 0$

$$
\operatorname* { s u p } _ { t > 0 } \operatorname* { m a x } _ { i \leq n } \mathbb { E } _ { z , z ^ { \prime } } e ^ { s | \xi _ { i , t } ( z , z ^ { \prime } ) | } \lesssim 1
$$

with very high probability. Using $| h ^ { \prime } ( u ) - 1 | \leq C e ^ { C | u | }$ by Assumption 6, H¨older’s inequality, and the preceding moment bound, we get

$$
\operatorname* { s u p } _ { t > 0 } \operatorname* { m a x } _ { i \leq n } \mathbb { E } _ { z , z ^ { \prime } } | D _ { i i , t } ( z , z ^ { \prime } ) | ^ { 2 } \prec d ^ { - 1 } .
$$

Finally, $\Delta _ { \mathrm { d } , t }$ is block diagonal in $i ,$ so

$$
\operatorname* { s u p } _ { t > 0 } \| \Delta _ { \mathrm { d } , t } \| _ { \mathrm { o p } } \leq \frac { 1 } { n } \operatorname* { s u p } _ { t > 0 } \operatorname* { m a x } _ { i \leq n } \big ( \mathbb { E } _ { z , z ^ { \prime } } | D _ { i i , t } ( z , z ^ { \prime } ) | ^ { 2 } \big ) ^ { 1 / 2 } \prec d ^ { - 3 / 2 } ,
$$

where we have used $n \asymp d$ from Assumption 1.

Lemma 33 provides uniform-in-time operator-norm bounds for the of-diagonal and diagonal remainders. The of-diagonal bound is suficiently strong to control its contribution in the subsequent analysis. The diagonal operator-norm bound is too weak for the gradient-flow times considered here. Nevertheless, the diagonal remainder has additional structure, which we exploit to obtain a stronger estimate on the invariant subspace $\widetilde { \nu }$

Lemma 34. Suppose that Assumptions 1, 2, and 6 hold. Let $\mathsf { P } _ { \widetilde { \nu } }$ denote the orthogonal projection onto the subspace $\widetilde { \mathcal { V } } \subseteq L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ defined in (78). Then,

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \mathsf { P } _ { \widetilde { \nu } } \Delta _ { \mathrm { d } , t } \mathsf { P } _ { \widetilde { \nu } } \| _ { \mathrm { o p } } \prec d ^ { - 2 } .
$$

Proof. Consider $F _ { a , b } , F _ { a ^ { \prime } , b ^ { \prime } } \in \widetilde { \mathcal { V } }$ with $\| F _ { \pmb { a } , \pmb { b } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \vee \| F _ { \pmb { a } ^ { \prime } , \pmb { b } ^ { \prime } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \leq 1$ , and write

$$
\langle F _ { a , b } , \Delta _ { \mathrm { d } , t } F _ { a ^ { \prime } , b ^ { \prime } } \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z , z ^ { \prime } } \left[ D _ { i i , t } ( z , z ^ { \prime } ) \left( \sqrt { n } a _ { i } + \langle b , z \rangle \right) \left( \sqrt { n } a _ { i } ^ { \prime } + \langle b ^ { \prime } , z ^ { \prime } \rangle \right) \right] .
$$

Taylor expansion of $h ( u ) - u$ at $\rho _ { t } ^ { 2 } \lVert \mathbf { x } _ { 0 , i } \rVert _ { 2 } ^ { 2 } / d$ gives

$$
D _ { i i , t } = \left( h ^ { \prime } \left( \frac { \rho _ { t } ^ { 2 } \| \mathbf { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right) - 1 \right) \left( U _ { i i , t } - \frac { \rho _ { t } ^ { 2 } \| \mathbf { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right) + \frac { 1 } { 2 } h ^ { \prime \prime } ( \xi _ { i , t } ) \left( U _ { i i , t } - \frac { \rho _ { t } ^ { 2 } \| \mathbf { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right) ^ { 2 } ,
$$

where $\xi _ { i , t }$ lies between $U _ { i i , t }$ and $\rho _ { t } ^ { 2 } \lVert \mathbf { x } _ { 0 , i } \rVert _ { 2 } ^ { 2 } / d$ . We decompose the bilinear form into the linear and quadratic Taylor terms:

$$
\langle F _ { a , b } , \Delta _ { \mathrm { d } , t } F _ { a ^ { \prime } , b ^ { \prime } } \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } = E _ { 1 } + E _ { 2 } ,
$$

where, with $\eta _ { i , t } = \rho _ { t } ^ { 2 } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d$

$$
E _ { 1 } : = \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z , z ^ { \prime } } \left[ \left( h ^ { \prime } ( \eta _ { i , t } ) - 1 \right) \left( U _ { i i , t } - \eta _ { i , t } \right) \left( \sqrt { n } a _ { i } + \langle b , z \rangle \right) \left( \sqrt { n } a _ { i } ^ { \prime } + \langle b ^ { \prime } , z ^ { \prime } \rangle \right) \right] ,
$$

$$
E _ { 2 } : = \frac { 1 } { 2 n ^ { 2 } } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { z , z ^ { \prime } } \left[ h ^ { \prime \prime } ( \xi _ { i , t } ( z , z ^ { \prime } ) ) \left( U _ { i i , t } - \eta _ { i , t } \right) ^ { 2 } \left( \sqrt { n } a _ { i } + \left. b , z \right. \right) \left( \sqrt { n } a _ { i } ^ { \prime } + \left. b ^ { \prime } , z ^ { \prime } \right. \right) \right] .
$$

The uniform overlap and exponential-moment bounds used in the proof of Lemma 33 imply

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathbb { E } _ { z , z ^ { \prime } } \left[ | h ^ { \prime \prime } ( \xi _ { i , t } ) | ^ { 2 } \left| U _ { i i , t } - \rho _ { t } ^ { 2 } \frac { \| { \pmb x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right| ^ { 4 } \right] ^ { 1 / 2 } \prec d ^ { - 1 } ,
$$

and therefore

$$
| E _ { 2 } | \le \frac { 1 } { 2 n } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathbb { E } _ { z , z ^ { \prime } } \left[ | h ^ { \prime \prime } ( \xi _ { i , t } ) | ^ { 2 } \left| U _ { i i , t } - \rho _ { t } ^ { 2 } \frac { \| \mathbf { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right| ^ { 4 } \right] ^ { 1 / 2 } \prec d ^ { - 2 } ,
$$

where we have used $n \asymp d$ from Assumption 1.

It remains to control the linear Taylor term. Using

$$
U _ { i i , t } - \rho _ { t } ^ { 2 } \frac { | | x _ { 0 , i } | | _ { 2 } ^ { 2 } } { d } = \frac { \rho _ { t } \sigma _ { t } } { d } \langle x _ { 0 , i } , z \rangle + \frac { \rho _ { t } \sigma _ { t } } { d } \langle x _ { 0 , i } , z ^ { \prime } \rangle + \frac { \sigma _ { t } ^ { 2 } } { d } \langle z , z ^ { \prime } \rangle
$$

and the independence and standard Gaussian moments of $( z , z ^ { \prime } )$ , its contribution to the bilinear form is

$$
E _ { 1 } = \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \left( h ^ { \prime } \left( \eta _ { i , t } \right) - 1 \right) \left( \frac { \rho _ { t } \sigma _ { t } \sqrt { n } } { d } a _ { i } ^ { \prime } \langle b , x _ { 0 , i } \rangle + \frac { \rho _ { t } \sigma _ { t } \sqrt { n } } { d } a _ { i } \langle x _ { 0 , i } , b ^ { \prime } \rangle + \frac { \sigma _ { t } ^ { 2 } } { d } \langle b , b ^ { \prime } \rangle \right) .\tag{87}
$$

On an event of very high probability,

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { m a x } _ { i \leq n } \left| h ^ { \prime } \left( \rho _ { t } ^ { 2 } \frac { \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } \right) - 1 \right| \lesssim 1 , \qquad \| \pmb { X } \| _ { \mathrm { o p } } \lesssim \sqrt { n } .
$$

Consequently,

$$
\begin{array} { r l } & { | E _ { 1 } | \lesssim \displaystyle \frac { 1 } { n ^ { 2 } } \left( \frac { \rho _ { t } \sigma _ { t } \sqrt { n } } { d } b ^ { \top } X H a ^ { \prime } + \frac { \rho _ { t } \sigma _ { t } \sqrt { n } } { d } b ^ { \prime \top } X H a + \frac { \sigma _ { t } ^ { 2 } n } { d } \langle b , b ^ { \prime } \rangle \right) } \\ & { \quad \lesssim \displaystyle \frac { 1 } { d ^ { 2 } } ( \| b \| _ { 2 } \| a ^ { \prime } \| _ { 2 } + \| b ^ { \prime } \| _ { 2 } \| a \| _ { 2 } + \| b \| _ { 2 } \| b ^ { \prime } \| _ { 2 } ) } \end{array}
$$

where $H : = \mathrm { d i a g } ( h ^ { \prime } ( \rho _ { t } ^ { 2 } \| { \pmb x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d ) - 1 ) _ { i \leq n }$ . By Cauchy–Schwarz inequality applied to $\mathbb { R } ^ { 2 ( n + d ) }$

$$
\| b \| _ { 2 } \| a ^ { \prime } \| _ { 2 } + \| a \| _ { 2 } \| b ^ { \prime } \| _ { 2 } \leq \sqrt { \| a \| _ { 2 } ^ { 2 } + \| b \| _ { 2 } ^ { 2 } } \sqrt { \| a ^ { \prime } \| _ { 2 } ^ { 2 } + \| b ^ { \prime } \| _ { 2 } ^ { 2 } } .
$$

Bounding $\| \pmb { b } \| _ { 2 } \| \pmb { b } ^ { \prime } \| _ { 2 } \leq \sqrt { \| \pmb { a } \| _ { 2 } ^ { 2 } + \| \pmb { b } \| _ { 2 } ^ { 2 } } \sqrt { \| \pmb { a } ^ { \prime } \| _ { 2 } ^ { 2 } + \| \pmb { b } ^ { \prime } \| _ { 2 } ^ { 2 } }$ , we get that

$$
| E _ { 1 } | \lesssim d ^ { - 2 } \sqrt { \| \pmb { a } \| _ { 2 } ^ { 2 } + \| \pmb { b } \| _ { 2 } ^ { 2 } } \sqrt { \| \pmb { a } ^ { \prime } \| _ { 2 } ^ { 2 } + \| \pmb { b } ^ { \prime } \| _ { 2 } ^ { 2 } } .
$$

Finally, $( a , b ) \mapsto F _ { a , b }$ is an isometry from $\mathbb { R } ^ { n + d }$ onto $\widetilde { \nu } .$ . Taking the supremum over unit $F _ { a , b }$ and $F _ { { \pmb a } ^ { \prime } , { \pmb b } ^ { \prime } }$ , and combining the linear and quadratic contributions concludes the proof. □

Remark G.1 (Role of the sample correction). The sample-dependent correction $\widetilde { \delta } _ { i , t }$ is essential to obtain the $d ^ { - 2 }$ bound in Lemma 34. Using the common correction $\delta _ { n , t }$ instead would yield an extra term:

$$
U _ { i i , t } - \rho _ { t } ^ { 2 } \tau _ { \Sigma } = \rho _ { t } ^ { 2 } \left( \frac { \lVert \boldsymbol { x } _ { 0 , i } \rVert ^ { 2 } } { d } - \tau _ { \Sigma } \right) + \frac { \rho _ { t } \sigma _ { t } } { d } \langle \boldsymbol { x } _ { 0 , i } , \boldsymbol { z } \rangle + \frac { \rho _ { t } \sigma _ { t } } { d } \langle \boldsymbol { x } _ { 0 , i } , \boldsymbol { z } ^ { \prime } \rangle + \frac { \sigma _ { t } ^ { 2 } } { d } \langle \boldsymbol { z } , \boldsymbol { z } ^ { \prime } \rangle .
$$

Repeating the linear-term computation (87) after Taylor expanding around $\rho _ { t } ^ { 2 } \tau _ { \Sigma }$ produces the additional contribution

$$
\frac { \rho _ { t } ^ { 2 } } { n } ( h ^ { \prime } ( \rho _ { t } ^ { 2 } \tau _ { \Sigma } ) - 1 ) \sum _ { i = 1 } ^ { n } \left( \frac { \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } - \tau _ { \Sigma } \right) a _ { i } a _ { i } ^ { \prime } .
$$

Taking the supremum over unit $F _ { a , b }$ and $F _ { { a } ^ { \prime } , { b } ^ { \prime } }$ , this term is at most

$$
\frac { \rho _ { t } ^ { 2 } } { n } ( h ^ { \prime } ( \rho _ { t } ^ { 2 } \tau _ { \Sigma } ) - 1 ) \operatorname* { m a x } _ { i \leq n } \left. \frac { \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } - \tau _ { \Sigma } \right. \prec d ^ { - 3 / 2 } ,
$$

uniformly over $t \in [ t _ { 0 } , T ]$ . Thus, the common correction would only yield $\textrm { a d } ^ { - 3 / 2 }$ bound, whereas the sample-dependent correction cancels this term and yields the stronger $d ^ { - 2 }$ estimate.

## G.3 Proof of Theorem 4

We next establish a uniform-in-time bound on the discrepancy between the empirical score and its linearized approximation, evaluated along the linearized reverse process. This estimate will control the KL divergence between the path laws of the empirical reverse SDE and its linearized counterpart.

Lemma 35 (Reverse-time score linearization). Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 6 hold. Then,

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widehat { S } _ { t } ( \widetilde { x } _ { t } ) - \widetilde { S } _ { t } ( \widetilde { x } _ { t } ) \| _ { 2 } ^ { 2 } \mid X \right] \prec \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathbf { t } _ { t } ^ { 2 } d ^ { - 5 } \left( 1 + \mathbf { t } _ { t } ^ { 2 } + \mathbf { t } _ { t } ^ { 4 } d ^ { - 2 } \right) .
$$

Proof. Write

$$
\widehat { \mathcal { A } } _ { t } : = \int _ { 0 } ^ { \mathfrak { t } _ { t } } e ^ { - s \widehat { \mathcal { K } } _ { t } } \mathrm { d } s , \qquad \widetilde { \mathcal { A } } _ { t } : = \int _ { 0 } ^ { \mathfrak { t } _ { t } } e ^ { - s \widetilde { \mathcal { K } } _ { t } } \mathrm { d } s ,
$$

and fix an orthonormal basis $\{ { \pmb v } _ { \ell } \} _ { \ell = 1 } ^ { d }$ of $\mathbb { R } ^ { d } .$ . The gradient-flow representations give, for every unit vector v,

$$
\begin{array} { r l } & { \sigma _ { t } \big ( \widehat { S } _ { t , v } ( \boldsymbol { x } ) - \widetilde { S } _ { t , v } ( \boldsymbol { x } ) \big ) = \Big \langle \phi _ { t , \boldsymbol { x } } - \phi _ { \mathrm { e f f } , t , \boldsymbol { x } } , \widehat { \mathcal { A } } _ { t } \widehat { f } _ { t , v } ^ { \star } \Big \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } + \Big \langle \phi _ { \mathrm { e f f } , t , \boldsymbol { x } } , ( \widehat { \mathcal { A } } _ { t } - \widetilde { \mathcal { A } } _ { t } ) \boldsymbol { z } _ { v } \Big \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } } \\ & { \qquad + \Big \langle \phi _ { \mathrm { e f f } , t , \boldsymbol { x } } , \widehat { \mathcal { A } } _ { t } ( \widehat { f } _ { t , v } ^ { \star } - \boldsymbol { z } _ { v } ) \Big \rangle _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } , } \end{array}
$$

where $\phi _ { t , { \pmb x } }$ and $\phi _ { \mathrm { e f f } , t , \boldsymbol { x } }$ are defined in (85).

Both $\widehat { \mathcal { K } } _ { t }$ and $\widetilde { \kappa } _ { t }$ are positive definite, and hence $\| \widehat { \mathcal { A } } _ { t } \| _ { \mathrm { o p } } \vee \| \widetilde { \mathcal { A } } _ { t } \| _ { \mathrm { o p } } \le \mathrm { t } _ { t }$ . By definition,

$$
\| \phi _ { \mathrm { e f f } , t , \widetilde { \mathbf { x } } _ { t } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } = \frac { 1 } { d ^ { 2 } } \widetilde { \pmb { x } } _ { t } ^ { \mathsf { T } } ( \rho _ { t } ^ { 2 } \widehat { \pmb { \Sigma } } + \sigma _ { t } ^ { 2 } \mathbf { I } _ { d } ) \widetilde { \pmb { x } } _ { t } .
$$

Lemma 32, with the bounds for $\widetilde { \pmb { x } } _ { t }$ from Lemma 31, along with the operator-norm bounds from (40), (44) gives

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \lVert \phi _ { t , \tilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } \rVert _ { L ^ { 2 } ( \tilde { \pi } _ { d , n } ) } ^ { 2 } \mid X \right] \prec d ^ { - 5 } , \qquad \mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \lVert \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } \rVert _ { L ^ { 2 } ( \tilde { \pi } _ { d , n } ) } ^ { 2 } \mid X \right] \lesssim d ^ { - 1 }\tag{88}
$$

with very high probability. Moreover, Proposition 2 holds uniformly over $t \in [ t _ { 0 } , T ]$ and unit v.

We upper bound the square of the Euclidean norm of the score diference by the sum of three terms:

$$
\| \widehat { S } _ { t } ( \widetilde { \pmb { x } } _ { t } ) - \widetilde { S } _ { t } ( \widetilde { \pmb { x } } _ { t } ) \| _ { 2 } ^ { 2 } \leq \frac { 3 } { \sigma _ { t } ^ { 2 } } ( E _ { 1 , t } + E _ { 2 , t } + E _ { 3 , t } ) ,
$$

where

$$
\begin{array} { c } { { E _ { 1 , t } : = \displaystyle \sum _ { \ell = 1 } ^ { d } \left| \left. \phi _ { \ell , \tilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } , \widehat { \mathcal { A } } _ { t } \widehat { f } _ { t , v _ { \ell } } ^ { \star } \right. _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \right| ^ { 2 } , \qquad E _ { 2 , t } : = \displaystyle \sum _ { \ell = 1 } ^ { d } \left| \left. \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } , ( \widehat { \mathcal { A } } _ { t } - \widehat { \mathcal { A } } _ { t } ) z _ { v _ { \ell } } \right. _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \right| ^ { 2 } , } } \\ { { E _ { 3 , t } : = \displaystyle \sum _ { \ell = 1 } ^ { d } \left| \left. \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } , \widehat { \mathcal { A } } _ { t } ( \widehat { f } _ { t , v _ { \ell } } ^ { \star } - z _ { v _ { \ell } } ) \right. _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \right| ^ { 2 } . } } \end{array}
$$

Let $\mathcal { F } _ { t } : \mathbb { R } ^ { d } \to L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ denote the linear map $v \mapsto \widehat { f } _ { t , v } ^ { \star }$ . Jensen’s inequality gives $\| \mathcal { F } _ { t } \| _ { \mathrm { o p } } \leq 1$ , and

$$
E _ { 1 , t } = \sum _ { \ell = 1 } ^ { d } \left| \left. \mathcal { F } _ { t } ^ { * } \widehat { \mathcal { A } } _ { t } ( \phi _ { t , \tilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } ) , v _ { \ell } \right. \right| ^ { 2 } = \| \mathcal { F } _ { t } ^ { * } \widehat { \mathcal { A } } _ { t } ( \phi _ { t , \tilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } ) \| _ { 2 } ^ { 2 } \lesssim \ t _ { t } ^ { 2 } \| \phi _ { t , \tilde { x } _ { t } } - \phi _ { \mathrm { e f f } , t , \tilde { x } _ { t } } \| _ { L ^ { 2 } ( \widehat { \mathbb { T } } _ { d , n } ) } ^ { 2 } ,
$$

where the last inequality follows from the sub-multiplicativity of the norm and $\| \widehat { A } _ { t } \| _ { \mathrm { o p } } \leq \mathsf { t } _ { t }$ . Therefore,

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } E _ { 1 , t } \mid \boldsymbol { X } \right] \prec \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathsf { t } _ { t } ^ { 2 } d ^ { - 5 } .
$$

A similar argument gives

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } E _ { 3 , t } \mid \boldsymbol { X } \right] \prec \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathbf { t } _ { t } ^ { 2 } d ^ { - 1 } e ^ { - 2 d ^ { c } } .
$$

It remains to control $E _ { 2 , t }$ . Note that $\{ z _ { v _ { \ell } } \} _ { \ell = 1 } ^ { d }$ is an orthonormal family in $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ , and that ${ \mathbf { \Psi } } _ { { \mathbf { } } z _ { v _ { \ell } } , \phi _ { \mathrm { e f f } , t , \widetilde { { \mathbf { x } } } _ { t } } } \in \widetilde { \mathcal { V } }$ for every ℓ and t. Hence, by Bessel’s inequality,

$$
E _ { 2 , t } = \sum _ { \ell = 1 } ^ { d } \left| \left. \phi _ { \mathrm { e f f } , t , \bar { x } _ { t } } , \mathsf { P } _ { \bar { \nu } } ( \widehat { A } _ { t } - \widetilde { A } _ { t } ) \mathsf { P } _ { \bar { \nu } } z _ { v _ { \ell } } \right. _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } \right| ^ { 2 } \leq \| \phi _ { \mathrm { e f f } , t , \bar { x } _ { t } } \| _ { L ^ { 2 } ( \widehat { \pi } _ { d , n } ) } ^ { 2 } \| \mathsf { P } _ { \bar { \nu } } ( \widehat { A } _ { t } - \widetilde { A } _ { t } ) \mathsf { P } _ { \bar { \nu } } \| _ { \mathrm { o p } } ^ { 2 } .
$$

The variation-of-parameters identity gives

$$
e ^ { - s \widehat { \mathcal { K } } _ { t } } - e ^ { - s \widetilde { \mathcal { K } } _ { t } } = - \int _ { 0 } ^ { s } e ^ { - ( s - r ) \widetilde { \mathcal { K } } _ { t } } ( \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ) e ^ { - r \widehat { \mathcal { K } } _ { t } } \mathrm { d } r .\tag{89}
$$

The integrals are Bochner integrals in the Banach space of bounded linear operators on $L ^ { 2 } ( \widehat { \pi } _ { d , n } )$ ; see [EN00, Corollary 1.7]. Applying the same identity to the last $e ^ { - r \widehat { \mathcal { K } } _ { t } }$ gives the exact expansion

$$
\begin{array} { r l r } {  { \widehat { \mathcal { A } } _ { t } - \widetilde { \mathcal { A } } _ { t } = - \int _ { 0 } ^ { \mathsf { t } _ { t } } \int _ { 0 } ^ { s } e ^ { - ( s - r ) \widetilde { \mathcal { K } } _ { t } } ( \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ) e ^ { - r \widetilde { \mathcal { K } } _ { t } } \mathrm { d } r \mathrm { d } s } } \\ & { } & { \quad + \int _ { 0 } ^ { \mathsf { t } _ { t } } \int _ { 0 } ^ { s } \int _ { 0 } ^ { r } e ^ { - ( s - r ) \widetilde { \mathcal { K } } _ { t } } ( \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ) e ^ { - ( r - u ) \widetilde { \mathcal { K } } _ { t } } ( \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ) e ^ { - u \widehat { \mathcal { K } } _ { t } } \mathrm { d } u \mathrm { d } r \mathrm { d } s . } \end{array}
$$

By the contraction property and the operator-norm bounds from Lemma 33,

$$
\begin{array} { r l } & { \left\| \displaystyle \int _ { 0 } ^ { \mathrm { t } _ { t } } \int _ { 0 } ^ { s } \int _ { 0 } ^ { r } e ^ { - ( s - r ) \widetilde { \mathcal { K } } _ { t } } ( \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ) e ^ { - ( r - u ) \widetilde { \mathcal { K } } _ { t } } ( \Delta _ { \mathrm { o d } , t } + \Delta _ { \mathrm { d } , t } ) e ^ { - u \widehat { \mathcal { K } } _ { t } } \mathrm { d } u \mathrm { d } r \mathrm { d } s \right\| _ { \mathrm { o p } } \prec \mathfrak { t } _ { t } ^ { 3 } d ^ { - 3 } , } \\ & { \qquad \quad \left\| \displaystyle \int _ { 0 } ^ { \mathrm { t } _ { t } } \int _ { 0 } ^ { s } e ^ { - ( s - r ) \widetilde { \mathcal { K } } _ { t } } \Delta _ { \mathrm { o d } , t } e ^ { - r \widetilde { \mathcal { K } } _ { t } } \mathrm { d } r \mathrm { d } s \right\| _ { \mathrm { o p } } \prec \mathfrak { t } _ { t } ^ { 2 } d ^ { - 5 / 2 } . } \end{array}
$$

Since $\widetilde { \kappa } _ { t }$ is self-adjoint and $\widetilde { \mathcal { V } } .$ invariant, $\widetilde { \nu }$ is reducing and hence

$$
\mathsf { P } _ { \widetilde { \mathcal { V } } } e ^ { - u \widetilde { K } _ { t } } = e ^ { - u \widetilde { K } _ { t } } \mathsf { P } _ { \widetilde { \mathcal { V } } } .
$$

for every $u \geq 0$ . Lemma 34 thus gives

$$
\left. \mathsf { P } _ { \widetilde { \nu } } \int _ { 0 } ^ { \mathsf { t } _ { t } } \int _ { 0 } ^ { s } e ^ { - ( s - r ) \widetilde { K } _ { t } } \Delta _ { \mathrm { d } , t } e ^ { - r \widetilde { K } _ { t } } \mathrm { d } r \mathrm { d } s \mathsf { P } _ { \widetilde { \nu } } \right. _ { \mathrm { o p } } \prec t _ { t } ^ { 2 } d ^ { - 2 } .
$$

Therefore, $\| \mathsf { P } _ { \widetilde { \nu } } ( \widehat { A } _ { t } - \widetilde { A } _ { t } ) \mathsf { P } _ { \widetilde { \nu } } \| _ { \mathrm { o p } } \prec \mathsf { t } _ { t } ^ { 2 } d ^ { - 2 } + \mathsf { t } _ { t } ^ { 3 } d ^ { - 3 }$ . Using (88),

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } E _ { 2 , t } \mid X \right] \prec \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } ( \mathfrak { t } _ { t } ^ { 4 } d ^ { - 5 } + \mathfrak { t } _ { t } ^ { 6 } d ^ { - 7 } ) .
$$

Since $\mathrm { i n f } _ { t \in [ t _ { 0 } , T ] } \sigma _ { t } > 0$ and $e ^ { - 2 d ^ { c } } \leq d ^ { - 4 }$ for all suficiently large $d ,$ combining the preceding three estimates with the score decomposition proves the claim. □

Proof of Theorem $\it 4 .$ Fix the training sample; the following arguments are conditional on $\boldsymbol { X }$

Note that the drift of the linearized reverse SDE (33) has a uniformly bounded time-dependent linear coeficient and is therefore globally Lipschitz with linear growth. Hence, the linearized reverse SDE has a unique global strong solution.

For the empirical reverse SDE (31), note that by (18),

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , v \in \mathbb { S } ^ { d - 1 } } \| \widetilde { \cal S } _ { t , v } \| _ { \mathcal { H } } \lesssim \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] , v \in \mathbb { S } ^ { d - 1 } } \mathrm { t } _ { t } \| { \cal S } _ { t } \| _ { \mathrm { o p } } < \infty .
$$

Since $h \in C ^ { 5 } ( \mathbb { R } )$ by Assumption $6 ,$ it follows from the derivative reproducing property $( \mathrm { e . g . }$ , [Zho08, Theorem 1]) applied to the restriction of $K$ to compact balls that for every $R > 0$ , there exists $C _ { R } > 0$ such that

$$
\operatorname* { s u p } _ { \| \pmb { x } \| _ { 2 } \le R } \| \nabla f ( \pmb { x } ) \| _ { 2 } \le C _ { R } \| f \| _ { \mathcal H }
$$

for all $f \in \mathcal { H } .$ . Hence, there exists a constant $C > 0$ such that for all x, $\pmb { y } \in \mathbb { R } ^ { d }$ with $\| { \pmb x } \| _ { 2 } \vee \| { \pmb y } \| _ { 2 } \le R$

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widehat { S } _ { t } ( \pmb { x } ) - \widehat { S } _ { t } ( \pmb { y } ) \| _ { 2 } \leq C \| \pmb { x } - \pmb { y } \| _ { 2 } .
$$

Standard local SDE theory gives a unique maximal strong solution up to an explosion time. Setting the process equal to ∞ after explosion yields a continuous $\mathbb { R } _ { \infty } ^ { d }$ -valued process on the full interval.

Let Pe and Pb denote the forward-time path laws of $\{ \widetilde { \pmb { x } } _ { T - s } \} _ { s \in [ 0 , T - t _ { 0 } ] }$ and $\{ \widehat { \pmb { x } } _ { T - s } \} _ { s \in [ 0 , T - t _ { 0 } ] }$ , respectively, on $C ( [ 0 , T - t _ { 0 } ] ; \mathbb { R } _ { \infty } ^ { d } )$ , and let y denote the canonical coordinate process. Since time reversal is a bimeasurable bijection,

$$
\mathrm { K L } ( \mathbb { P } _ { \widetilde { \pmb { x } } } \| \mathbb { P } _ { \widehat { \pmb { x } } } ) = \mathrm { K L } ( \widetilde { \mathbb { P } } \| \widehat { \mathbb { P } } ) .
$$

For $R > 0$ , define the canonical stopping time

$$
\tau _ { R } : = \operatorname* { i n f } \bigg \{ s \in [ 0 , T - t _ { 0 } ] : \int _ { 0 } ^ { s } \Big \| \widehat { S } _ { T - u } ( y _ { u } ) - \widetilde { S } _ { T - u } ( y _ { u } ) \Big \| _ { 2 } ^ { 2 } \mathrm { d } u \geq R \bigg \} \wedge ( T - t _ { 0 } ) ,
$$

with the convention that the integrand equals $\infty$ when $y _ { u } = \infty$

Since the linearized drift has the form $D _ { T - s } { \pmb y } _ { s }$ , with $\operatorname* { s u p } _ { t } \| D _ { t } \| _ { \mathrm { o p } } < \infty$ , the empirical process satisfies, for s smaller than the explosion time ζ,

$$
y _ { s } = y _ { 0 } + \int _ { 0 } ^ { s } D _ { T - u } y _ { u } \mathrm { d } u + 2 \int _ { 0 } ^ { s } ( \widehat { S } _ { T - u } ( y _ { u } ) - \widetilde { S } _ { T - u } ( y _ { u } ) ) \mathrm { d } u + \sqrt { 2 } W _ { s } .
$$

If $\begin{array} { r } { \int _ { 0 } ^ { \zeta } \| \Delta _ { u } \| _ { 2 } ^ { 2 } \mathrm { d } u < \infty . } \end{array}$ then it follows from Cauchy-Schwarz and Gronwall’s inequality that $\mathrm { s u p } _ { s < \zeta } \| \pmb { y } _ { s } \| _ { 2 } <$ $\infty ,$ , contradicting the explosion alternative. Consequently, $\tau _ { R } < \zeta$ on every explosive path, for every $R < \infty$

Let $\widetilde { \mathbb { P } } ^ { R }$ and ${ \widehat { \mathbb { P } } } ^ { R }$ denote the laws of the stopped canonical process $\{ y _ { s \wedge \tau _ { R } } \} _ { s \in [ 0 , T - t _ { 0 } ] }$ under $\widetilde { \mathbb { P } }$ and ${ \widehat { \mathbb { P } } } ,$ respectively. Girsanov’s theorem gives

$$
\begin{array} { r l } & { \mathrm { K L } ( \widetilde { \mathbb { P } } ^ { R } \| \widehat { \mathbb { P } } ^ { R } ) \leq \mathbb { E } _ { \widetilde { \mathbb { P } } } \left[ \displaystyle \int _ { 0 } ^ { \tau _ { R } } \Big \| \widehat { \pmb { S } } _ { T - s } ( \pmb { y } _ { s } ) - \widetilde { \pmb { S } } _ { T - s } ( \pmb { y } _ { s } ) \Big \| _ { 2 } ^ { 2 } \mathrm { d } s \ | \ \pmb { X } \right] } \\ & { \qquad \leq \mathbb { E } _ { \widetilde { \mathbb { P } } } \left[ \displaystyle \int _ { 0 } ^ { T - t _ { 0 } } \Big \| \widehat { \pmb { S } } _ { T - s } ( \pmb { y } _ { s } ) - \widetilde { \pmb { S } } _ { T - s } ( \pmb { y } _ { s } ) \Big \| _ { 2 } ^ { 2 } \mathrm { d } s \ | \ \pmb { X } \right] . } \end{array}
$$

Since the linearized process is nonexplosive and the scores are locally bounded, $\tau _ { R } \uparrow T - t _ { 0 }$ almost surely under $\widetilde { \mathbb { P } } .$ . Under $\widehat { \mathbb { P } } , { \boldsymbol \tau } _ { R }$ increases to $T - t _ { 0 }$ on nonexplosive paths and to the explosion time on explosive paths. In the latter case, the stopped paths converge to the absorbed path in the topology of $C ( [ 0 , T - t _ { 0 } ] ; \mathbb { R } _ { \infty } ^ { d } )$ . Hence,

$$
\widetilde { \mathbb { P } } ^ { R } \Longrightarrow \widetilde { \mathbb { P } } , \qquad \widehat { \mathbb { P } } ^ { R } \Longrightarrow \widehat { \mathbb { P } } .
$$

By the joint lower semicontinuity of relative entropy, the preceding estimate and Lemma 35,

$$
\begin{array} { r l } & { \displaystyle \mathrm { K L } \big ( \mathbb { P } _ { \widetilde { \pmb { x } } } \big \| \mathbb { P } _ { \widehat { \pmb { x } } } \big ) = \mathrm { K L } \big ( \widetilde { \mathbb { P } } \big \| \widehat { \mathbb { P } } \big ) \leq \int _ { t _ { 0 } } ^ { T } \mathbb { E } \left[ \left\| \widehat { S } _ { t } \big ( \widetilde { \pmb { x } } _ { t } \big ) - \widetilde { S } _ { t } ( \widetilde { \pmb { x } } _ { t } ) \right\| _ { 2 } ^ { 2 } \bigm | \frac { X } { X } \right] \mathrm { d } t } \\ & { \qquad \prec \left( T - t _ { 0 } \right) \displaystyle \operatorname* { s u p } _ { t \in \left[ t _ { 0 } , T \right] } \mathsf { t } _ { t } ^ { 2 } d ^ { - 5 } \left( 1 + \mathsf { t } _ { t } ^ { 2 } + \mathsf { t } _ { t } ^ { 4 } d ^ { - 2 } \right) . } \end{array}
$$

## G.4 Proof of Theorem 5

We introduce an auxiliary spectral linearized process, which is analogous to the linearized reverse process $\widetilde { \pmb { x } } _ { t }$ but is obtained by replacing the kernel $\widetilde { \kappa } _ { t }$ by the time-dependent efective empirical operator $\widehat { \mathcal { K } } _ { \mathrm { e f f } , t }$ . More particularly, for every $t \in [ t _ { 0 } , T ]$ , let ${ \check { S } } _ { t }$ denote the linear score estimator obtained by replacing $\widetilde { \kappa } _ { t }$ with $\widehat { \mathcal { K } } _ { \mathrm { e f f } , t }$ in (32), where $\widehat { \mathcal { K } } _ { \mathrm { e f f } , t }$ and $\delta _ { n , t }$ are defined in (22) for time t. Equivalently, unwinding the definitions using the proof of Lemma 19 shows that $\check { S } _ { t }$ is an explicit linear shrinkage field:

$$
\check { S } _ { t } ( { \pmb x } ) = \frac { 1 } { \sigma _ { t } } \psi _ { t } ( \widehat { { \Sigma } } , { \ t } _ { t } ) { \pmb x } .
$$

Define the auxiliary spectral linearized process by

$$
\mathrm { d } \check { \pmb { x } } _ { t } = - ( \check { \pmb { x } } _ { t } + 2 \check { \pmb { S } } _ { t } ( \check { \pmb { x } } _ { t } ) ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } \bar { \pmb { B } } _ { t } , \qquad \check { \pmb { x } } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } ) ,
$$

and write

$$
\breve { \Sigma } _ { t } : = \mathbb { E } [ \breve { \pmb { x } } _ { t } \breve { \pmb { x } } _ { t } ^ { \mathsf { T } } \mid X ] .
$$

By Itˆo’s lemma, $\check { \Sigma } _ { t }$ satisfies the ODE

$$
\frac { \mathrm { d } } { \mathrm { d } t } \check { \Sigma } _ { t } = - 2 \check { \Sigma } _ { t } - \frac { 2 } { \sigma _ { t } } \psi _ { t } ( \widehat { \Sigma } , \mathbf { t } _ { t } ) \check { \Sigma } _ { t } - \frac { 2 } { \sigma _ { t } } \check { \Sigma } _ { t } \psi _ { t } ( \widehat { \Sigma } , \mathbf { t } _ { t } ) - 2 \mathbf { I } _ { d } , \qquad \check { \Sigma } _ { T } = \mathbf { I } _ { d } .\tag{90}
$$

The next lemma shows that the covariance of the linearized reverse process $\widetilde { \pmb { \Sigma } } _ { t }$ is close to the covariance of the spectral linearized process $\check { \Sigma } _ { t }$

Lemma 36. Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 6 hold. Then

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \Vert \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \Vert _ { \mathrm { o p } } \prec \left( d ^ { - 3 / 2 } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathfrak { t } _ { t } \right) \wedge 1 .
$$

Proof. Let

$$
\check { \pmb { K } } _ { t } : = \widetilde { \pmb { K } } _ { t } + \left[ \begin{array} { c c } { \delta _ { n , t } \mathbf { I } _ { n } - \widetilde { \pmb { D } } _ { t } } & { 0 } \\ { 0 } & { 0 } \end{array} \right] .
$$

Thus, $\check { \pmb { K } } _ { t }$ is the finite-dimensional representation of $\widehat { \mathcal { K } } _ { \mathrm { e f f } , t }$ on the invariant subspace $\widetilde { \nu } .$ Both $\widetilde { K _ { t } }$ and $\check { \pmb { K } } _ { t }$ are positive semidefinite, since their diagonal corrections are nonnegative. Repeating the calculation in Lemma 30 with $\widetilde { D } _ { t }$ replaced by $\delta _ { n , t } \mathbf { I } _ { n }$ gives

$$
\psi _ { t } ( \widehat { \Sigma } , \mathfrak { t } _ { t } ) = - \frac { 1 } { \sigma _ { t } } \left[ 0 \quad \mathbf { I } _ { d } \right] \left( \mathbf { I } _ { n + d } - e ^ { - \mathfrak { t } _ { t } \widehat { K } _ { t } } \right) \bigg [ \mathbf { I } _ { d } \bigg ] , \qquad - \sigma _ { t } ^ { - 1 } \mathbf { I } _ { d } \preceq \psi _ { t } ( \widehat { \Sigma } , \mathfrak { t } _ { t } ) \preceq 0 .\tag{91}
$$

An argument analogous to the one used to obtain (81) shows that

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left( \lVert \widetilde { \boldsymbol { \Sigma } } _ { t } \rVert _ { \mathrm { o p } } \vee \lVert \check { \boldsymbol { \Sigma } } _ { t } \rVert _ { \mathrm { o p } } \right) \lesssim 1 .\tag{92}
$$

On the very-high-probability event on which ma $\mathrm { x } _ { i \leq n } \| \pmb { x } _ { 0 , i } \| _ { 2 } ^ { 2 } / d$ is bounded, the mean-value the orem and the definition of the two diagonal corrections yield

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widetilde { K } _ { t } - \check { K } _ { t } \| _ { \mathrm { o p } } = \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { m a x } _ { i \leq n } | \widetilde { \delta } _ { i , t } - \delta _ { n , t } | \lesssim \frac { 1 } { n } \operatorname* { m a x } _ { i \leq n } \left| \frac { \| x _ { 0 , i } \| _ { 2 } ^ { 2 } } { d } - \tau _ { \Sigma } \right| \prec d ^ { - 3 / 2 } ,\tag{93}
$$

where the last estimate follows from (42) and $n \asymp d$ . By (89),

$$
e ^ { - \mathrm { t } _ { t } \widetilde { K } _ { t } } - e ^ { - \mathrm { t } _ { t } \check { K } _ { t } } = - \int _ { 0 } ^ { \mathrm { t } _ { t } } e ^ { - ( \mathrm { t } _ { t } - s ) \widetilde { K } _ { t } } ( \widetilde { K } _ { t } - \check { K } _ { t } ) e ^ { - s \check { K } _ { t } } \mathrm { d } s .
$$

Hence, using (91) and (93), together with the sub-multiplicativity of the operator norm and the fact that $\| \exp ( - s \mathbf { \widetilde { K } } _ { t } ) \| _ { \mathrm { o p } } \vee \| \exp ( - s \mathbf { \check { K } } _ { t } ) \| _ { \mathrm { o p } } \leq 1$ for every $s \geq 0$

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widetilde { \Psi } _ { t } - \psi _ { t } ( \widehat { \Sigma } , \mathfrak { t } _ { t } ) \| _ { \mathrm { o p } } \leq \frac { 1 } { \sigma _ { t _ { 0 } } } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathfrak { t } _ { t } \| \widetilde { K } _ { t } - \check { K } _ { t } \| _ { \mathrm { o p } } \prec d ^ { - 3 / 2 } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathfrak { t } _ { t } .\tag{94}
$$

Taking the diference between the covariance ODEs (34) and (90), using the linearity of the corresponding score functions, we have

$$
\begin{array} { r } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \big ( \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \big ) = - 2 \big ( \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \big ) - \frac { 2 } { \sigma _ { t } } \big ( \widetilde { \Psi } _ { t } \big ( \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \big ) + \big ( \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \big ) \widetilde { \Psi } _ { t } \big ) } \\ { - \frac { 2 } { \sigma _ { t } } \big ( \big ( \widetilde { \Psi } _ { t } - \psi _ { t } \big ( \widehat { \Sigma } , \mathfrak { t } _ { t } \big ) \big ) \check { \Sigma } _ { t } + \check { \Sigma } _ { t } \big ( \widetilde { \Psi } _ { t } - \psi _ { t } \big ( \widehat { \Sigma } , \mathfrak { t } _ { t } \big ) \big ) \big ) } \end{array}\tag{95}
$$

Integrating (95) from t to $T _ { \ast }$ , using the common terminal condition, the bound on $\| \widetilde { \Psi } \| _ { \mathrm { o p } }$ from Lemma 30, (92), and taking the operator norm gives

$$
\| \widetilde { \boldsymbol { \Sigma } } _ { t } - \breve { \boldsymbol { \Sigma } } _ { t } \| _ { \mathrm { o p } } \lesssim \int _ { t } ^ { T } \Big ( \| \widetilde { \boldsymbol { \Sigma } } _ { s } - \breve { \boldsymbol { \Sigma } } _ { s } \| _ { \mathrm { o p } } + \| \widetilde { \boldsymbol { \Psi } } _ { s } - \boldsymbol { \psi } _ { s } ( \widehat { \boldsymbol { \Sigma } } , \mathbf { t } _ { s } ) \| _ { \mathrm { o p } } \Big ) \mathrm { d } s .
$$

Applying backward Gronwall’s inequality and (94), yields

$$
\| \widetilde { \boldsymbol { \Sigma } } _ { t } - \check { \boldsymbol { \Sigma } } _ { t } \| _ { \mathrm { o p } } \lesssim \int _ { t } ^ { T } \| \widetilde { \boldsymbol { \Psi } } _ { s } - \psi _ { s } ( \widehat { \boldsymbol { \Sigma } } , \mathbf { t } _ { s } ) \| _ { \mathrm { o p } } \mathrm { d } s \prec d ^ { - 3 / 2 } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \mathbf { t } _ { t } .
$$

Finally, (92) also gives $\operatorname* { s u p } _ { t } \| \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \| _ { \mathrm { o p } } \lesssim 1$ , which yields the minimum with 1.

The covariance ODE associated with the spectral linearized process is diagonalized by the eigenvectors of $\widehat { \pmb { \Sigma } }$ . Hence, we may apply tools from random matrix theory to analyze the spectrum of $\check { \Sigma } _ { t }$

Lemma 37 (Deterministic equivalent of the spectral linearized covariance). Suppose that Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 6 hold. Then, for every deterministic $0 \preceq A \in \mathbb { R } ^ { d \times d }$ with $\| A \| _ { * } \leq 1$ and every constant $\epsilon , D > 0$ , there exists a constant $C > 0$ such that

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left. \mathrm { T r } \big \{ A ( \check { \Sigma } _ { t } - \bar { \Sigma } _ { t } ) \big \} \right. \leq d ^ { \epsilon - 1 / 2 }
$$

with probability at least $1 - C d ^ { - D }$

Proof. Since all the coeficient matrices in (90) are spectral functions of $\widehat { \pmb { \Sigma } }$ and $\check { \Sigma } _ { T } = \mathbf { I } _ { d } .$

$$
\check { \Sigma } _ { t } = \bar { \Lambda } _ { t } ( \widehat { \Sigma } ) ,
$$

where $\bar { \Lambda } _ { t }$ is defined by (35).

By (44) and the compact support of $\Omega _ { i }$ there is a deterministic $\Lambda < \infty$ such that, with very high probability, both $\mathrm { s p e c } ( \widetilde { \Sigma } )$ and supp(Ω) are contained in $[ 0 , \Lambda ]$ . Fix a positively oriented rectangular contour Γ enclosing this interval, with vertical edges $\Re z = - \eta$ and $\Re z = \Lambda + \eta$ and horizontal edges $\mathrm { I m } z = \pm 1$ , where $0 < \eta < \operatorname* { m i n } \{ 1 , \sigma _ { t _ { 0 } } ^ { 2 } / 4 \}$ . In particular,

$$
\operatorname* { i n f } _ { t \in [ t _ { 0 } , T ] } \operatorname* { i n f } _ { z \in \Gamma } \Re ( \rho _ { t } ^ { 2 } z + \sigma _ { t } ^ { 2 } ) > 0 .\tag{96}
$$

We show that the contour estimate (64) holds uniformly along the reverse-time schedule. The matrix-exponential representation (63), applied with $( \rho , \sigma , \delta _ { n } , \mathrm { t } ) = ( \rho _ { t } , \sigma _ { t } , \delta _ { n , t } , \mathrm { t } _ { t } )$ shows that $z \mapsto$ $\psi _ { t } ( z , \mathfrak { t } _ { t } )$ is entire for every t. Also, since $t \mapsto \mathsf { t } _ { t }$ is piecewise continuous and $t \mapsto ( \rho _ { t } , \sigma _ { t } , \delta _ { n , t } )$ is continuous, $t \mapsto \psi _ { t } ( z , \mathfrak { t } _ { t } )$ is piecewise continuous for every z.

It remains to verify that the estimates used in the proof of Lemma 20 are uniform over $t \in [ t _ { 0 } , T ]$ Assumption 6 implies Assumption $^ { 3 , }$ while $\begin{array} { r } { 0 < \operatorname* { i n f } _ { t \in [ t _ { 0 } , T ] } \sigma _ { t } \wedge \rho _ { t } \leq \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \sigma _ { t } \vee \rho _ { t } < 1 } \end{array}$ . Moreover, by Remark C.2, either $h ( u ) = u$ for every $u ,$ in which case $d \delta _ { n , t } = 0$ for every t, or $d \delta _ { n , t } \asymp 1$ uniformly over $t \in [ t _ { 0 } , T ]$ . Indeed, in the latter case $h ( u ) - u > 0$ for every $u > 0$ , while $\rho _ { t } ^ { 2 } \tau _ { \Sigma }$ remains in a compact subset of $( 0 , \infty )$ . Together with the contour bound (96), these are precisely the uniform conditions used to derive (64). Consequently,

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { s u p } _ { z \in \Gamma } | \psi _ { t } ( z , \mathbf { t } _ { t } ) | \lesssim 1 .\tag{97}
$$

Since $t \mapsto \psi _ { t } ( z , \mathsf { t } _ { t } )$ is piecewise continuous and $z \mapsto \psi _ { t } ( z , \mathfrak { t } _ { t } )$ is entire, the integral form of (35) and Picard iteration show that $z \mapsto \bar { \Lambda } _ { t } ( z )$ is entire for every t. Moreover, $\sigma _ { t } \geq \sigma _ { t _ { 0 } } > 0$ and (97) imply that the ODE coeficient is uniformly bounded on $[ t _ { 0 } , T ] \times \Gamma$ . Backward Gronwall’s inequality therefore gives

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { s u p } _ { z \in \Gamma } | \bar { \Lambda } _ { t } ( z ) | \lesssim 1 .\tag{98}
$$

On the spectral-containment event above, Cauchy’s formula and (27) give

$$
\breve { \Sigma } _ { t } = - \frac { 1 } { 2 \pi i } \oint _ { \Gamma } \bar { \Lambda } _ { t } ( z ) \widehat { \pmb { R } } ( z ) { \mathrm { d } } z , \qquad \bar { \Sigma } _ { t } = - \frac { 1 } { 2 \pi i } \oint _ { \Gamma } \bar { \Lambda } _ { t } ( z ) \pmb { M } ( z ) { \mathrm { d } } z .
$$

The contour Γ lies a fixed positive distance from supp(Ω), satisfies in $\mathrm { f } _ { z \in \Gamma } | z | > 0$ , has bounded real part, and obeys $| \mathrm { I m } z | \leq 1$ , with horizontal edges Imz = ±1. Hence Proposition 5 applies uniformly on Γ. Together with (98), this gives

$$
\operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left. \mathrm { T r } \big \{ A ( \check { \Sigma } _ { t } - \bar { \Sigma } _ { t } ) \big \} \right. \leq \frac { \vert \mathrm { T } \vert } { 2 \pi } \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \operatorname* { s u p } _ { z \in \Gamma } \vert \bar { \Lambda } _ { t } ( z ) \vert \operatorname* { s u p } _ { z \in \Gamma } \left. \mathrm { T r } \Big \{ A ( \hat { R } ( z ) - M ( z ) ) \Big \} \right. \leq d ^ { \epsilon - 1 / 2 }
$$

with probability at least $1 - C d ^ { - D }$ for some constant $C > 0$ , where $\epsilon , D > 0$ are arbitrary. □

The proof of Theorem 5 follows by combining Lemmas 36 and 37.

Proof of Theorem 5. For every deterministic $0 \preceq A \in \mathbb { R } ^ { d \times d }$ with $\| A \| _ { * } \leq 1$ ，

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left. \mathrm { T r } \Big \{ A ( \widetilde { \Sigma } _ { t } - \bar { \Sigma } _ { t } ) \Big \} \right. \leq \displaystyle \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left. \mathrm { T r } \Big \{ A ( \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } ) \Big \} \right. + \displaystyle \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left. \mathrm { T r } \big \{ A ( \check { \Sigma } _ { t } - \bar { \Sigma } _ { t } ) \big \} \right. } \\ & { \qquad \leq \displaystyle \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \| \widetilde { \Sigma } _ { t } - \check { \Sigma } _ { t } \| _ { \mathrm { o p } } + \displaystyle \operatorname* { s u p } _ { t \in [ t _ { 0 } , T ] } \left. \mathrm { T r } \big \{ A ( \check { \Sigma } _ { t } - \bar { \Sigma } _ { t } ) \big \} \right. . } \end{array}
$$

The first term is bounded by Lemma 36 and the second term is bounded by Lemma 37. □