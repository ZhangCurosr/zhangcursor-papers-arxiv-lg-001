# Spatially Adaptive Noise Injection

Frantzeska Lavda<sup>1</sup> , Maciej Falkiewicz<sup>1,2</sup> , Van Khoa Nguyen<sup>1,2</sup> , and Alexandros Kalousis<sup>1</sup>

<sup>1</sup> University of Applied Sciences Western Switzerland, HES-SO Geneva <sup>2</sup> Computer Science Department, University of Geneva frantzeska.lavda@hesge.ch

Abstract. Difusion samplers reverse a learned noising process using either stochastic (DDPM) or deterministic (DDIM) updates, which represent endpoints of a single family controlled by a scalar noise-injection variance that is applied identically at every spatial location. This uniform approach neglects the geometry of natural images: high-curvature regions such as edges and textures, where the denoiser is uncertain, benefit from stochastic correction, whereas smooth regions, where the score is precise, are degraded by injected noise. This work investigates whether each pixel requires stochastic correction at a given timestep and introduces Spatially Adaptive Noise Injection (SANI), a novel sampling framework that dynamically adjusts noise application on a per-pixel basis. SANI integrates a probabilistic gating mechanism with a derived spatially adaptive variance, ensuring that noise is injected precisely where needed to refine complex features while preserving well-formed structures. Experimental results and decoupling ablations demonstrate that SANI consistently improves Fréchet Inception Distance (FID) over the vanilla DDPM and DDIM endpoint samplers across diverse sampling timesteps, while remaining competitive with variance-learning baselines, highlighting the importance of spatial adaptivity in difusion sampling.

Keywords: Difusion models · Sampling · Spatial Adaptivity

## 1 Introduction

Difusion Probabilistic Models (DPMs) [8, 24] have become a leading approach in generative modeling, achieving state-of-the-art sample quality across images, audio, and molecular design. These models learn to reverse a gradual noising process by training a denoising network, resulting in a tractable variational bound and stable training. During inference, two primary families of samplers are typically used. Stochastic samplers [8, 11] inject noise at each step, emulating Langevin dynamics to correct approximation errors. In contrast, deterministic samplers [13, 14, 22] follow a noise-free probability flow, ordinary diferential equation (ODE) trajectory, which accelerates generation but limits error correction. [22] demonstrated that these two regimes present endpoints of an unified family, interpolated by a scalar parameter η that governs the noise injection variance at each step. However, a key limitation of both approaches is that the noise injection variance $\sigma _ { t } ^ { 2 }$ is applied uniformly across all spatial location, discarding the geometric heterogeneity inherent in high-dimensional data manifolds.

High-frequency regions, such as edges, textures, and fine details, correspond to areas of high curvature on the data manifold, where the denoiser’s gradient changes rapidly and estimation error tends to be higher. These regions benefit from stochastic correction [23]. In contrast, smooth regions such as uniform backgrounds are situated on locally flat portions of the manifold, where the score estimate is accurate and additional noise injection reduces fidelity. Ideally, a sampler would adapt its behavior to the local geometry: employing deterministic dynamics in regions where the model is confident and stochastic dynamics where uncertainty is greater, within the same image and timestep.

This consideration leads to a central question: how can one determine, at each pixel and timestep, whether stochastic correction is necessary? We address this by deriving the Fisher Information of the denoising distribution, which provides a closed-form per-pixel measure of uncertainty without requiring additional supervision. The analysis begins with the observation that the denoising distribution $p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } )$ constitutes an exponential family parametrized by the noisy observation $\mathbf { x } _ { t }$ . This structure enables the derivation of the Fisher Information Matrix (FIM) of the posterior with respect to $\mathbf { x } _ { t }$ in closed form. We show that this FIM is proportional to both the Jacobian of the optimal denoiser and the posterior covariance. The trace of this relationship establishes a direct connection between the model’s total sensitivity and reconstruction uncertainty. Regions exhibiting large Fisher Information correspond to areas where the model is uncertain about the underlying signal, indicating that stochastic correction is most beneficial in these locations.

Building on this insight, we introduce Spatially Adaptive Noise Injection (SANI), a sampling framework that modulates stochasticity on a per-pixel basis. We derive a closed-form gating function $g _ { t , i } \in [ 0 , 1 ]$ with a clear probabilistic interpretation, representing the probability that the local denoising error at pixel i exceeds a specified tolerance threshold (Prop. 2). This gating mechanism adjusts noise injection at each pixel, interpolating per pixel between between ODE dynamics $( g _ { t , i } = 0$ , deterministic) and stochastic DDPM-style update $( g _ { t , i } = 1 )$ within a single image and timestep. As a result, stochasticity increases in uncertain pixels while deterministic per-pixel updates are enforced in confident regions. The gating function is computed from the per-pixel posterior variance, which is directly related to the model’s sensitivity. SANI operates at inference time and is compatible with any pre-trained difusion model.

In summary, the main contributions are as follows: (i) Theoretical Link. We show that the Fisher Information Matrix of the denoising distribution is proportional to the denoiser Jacobian and posterior covariance (Prop. 1), establishing a computable relationship between local sensitivity and reconstruction uncertainty. (ii) Probabilistic gating. We derive a closed-form, per-pixel gating function $g _ { t , i }$ representing the probability that the local denoising error exceeds a specific tolerance $\left( \mathrm { E q . ~ 7 } \right)$ , and show that its spatial structure is robust to the choice of the residual-distribution model. (iii) Spatially adaptive sampler. We present SANI (Alg. 1), which applies the deterministic (DDIM) and stochastic (DDPM-style) update rules on a per-pixel basis. (iv) Empirical Validation. We show that SANI consistently improves over vanilla DDPM and DDIM across all datasets and step counts, and is competitive with variance-learning DDPM-family samplers, with the largest gains in the low-step regime, while its gating maps track image geometry without requiring edge supervision.

## 2 Background

## 2.1 Denoising Difusion Probabilistic Models

Difusion models define a forward noising process and a learned reverse denoising process. Given data $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } _ { 0 } )$ , the forward process adds Gaussian noise according to a schedule $\beta _ { 1 } , \ldots , \beta _ { T } \in ( 0 , 1 )$ :

$$
q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ) = { \mathcal { N } } \big ( \mathbf { x } _ { t } ; { \sqrt { 1 - \beta _ { t } } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } \big ) .\tag{1}
$$

Defining $\alpha _ { t } ~ = ~ 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } \ = \ \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ , the marginal at any timestep is $q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )$ , so that $\mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ with $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . The reverse process learns to invert this chain by parameterizing $p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } ) = \mathcal { N } ( \mathbf { x } _ { t - 1 } ; \mu _ { \theta } ( \mathbf { x } _ { t } , t ) , \Sigma _ { \theta } ( \mathbf { x } _ { t } , t ) )$ . Training maximizes the ELBO on log p<sub>θ</sub>(x<sub>0</sub>) via a noise prediction network $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t )$ with the simplified objective $\mathcal { L } _ { \mathrm { s i m p l e } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \epsilon } [ \| \epsilon - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) \| ^ { 2 } ]$ . This objective provides a learning signal only for the mean $\scriptstyle \mu _ { \theta } .$ , the covariance $\scriptstyle \pmb { \Sigma } _ { \theta }$ must be specified separately. Ho et al. [8] set it to one of two scalar schedules, $\sigma _ { t } ^ { 2 } = \beta _ { t }$ or $\begin{array} { r } { \bar { \sigma _ { t } ^ { 2 } } = \tilde { \beta } _ { t } : = \bar { \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } } \bar { \beta } _ { t } } \end{array}$ , both spatially uniform.

## 2.2 Generalized Sampling: Unifying DDPM and DDIM

The noise prediction network implicitly defines a clean data estimate via Tweedie’s formula [7]: $\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } , t ) = \frac { 1 } { \sqrt { \bar { \alpha } _ { \star } } } ( \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) ) } \end{array}$ . Song et al. [22] showed that a family of non-Markovian reverse processes, parameterized by the noise injection variance $\sigma _ { t } ^ { 2 }$ , shares the same marginals as the forward process. The generalized update rule is:

$$
\begin{array} { r } { \mathbf { x } _ { t - 1 } = \underbrace { \sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathbf { x } } _ { 0 } } _ { \mathrm { d e n o i s e d ~ m e a n } } + \underbrace { \sqrt { 1 - \bar { \alpha } _ { t - 1 } - \sigma _ { t } ^ { 2 } } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) } _ { \mathrm { d i r e c t i o n ~ c o r r e c t i o n } } + \underbrace { \sigma _ { t } \mathbf { z } } _ { \mathrm { n o i s e } } , } \end{array}\tag{2}
$$

where $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ for $t > 1$ and ${ \bf z } = { \bf 0 }$ for $t = 1$ . Setting $\sigma _ { t } ^ { 2 } = 0$ yields DDIM (deterministic), setting $\sigma _ { t } ^ { 2 } = \tilde { \beta } _ { t }$ recovers DDPM (stochastic).

The spatial uniformity bottleneck. In all of these methods, $\sigma _ { t } ^ { 2 }$ is a scalar applied identically to every spatial location. This ignores the heterogeneous geometry of natural images: edges and textures lie on curved manifold regions where stochastic correction is beneficial, while smooth regions lie on flat portions where noise degrades fidelity. We argue that this trade-of should not be global but spatially adaptive.

## 3 Spatially Adaptive Noise Injection

This section presents the proposed method and the theoretical framework. We first show that the local geometry of the difusion process, as quantified by the Fisher Information, is directly related to posterior reconstruction uncertainty. This relationship is then leveraged to develop a sampling algorithm. The general concept involves deriving a gating function $g _ { t , i } \in [ 0 , 1 ]$ with clear probabilistic interpretation, which is used to modulate the per-pixel noise injection variance. This approach enables each pixel to follow its geometrically optimal trajectory between deterministic and stochastic dynamics.

## 3.1 Fisher Information, Local Sensitivity and Model’s Uncertainty

The denoising distribution $p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } )$ for any fixed timestep t, indexed by the noisy observation $\mathbf { x } _ { t } ,$ belongs to an exponential family. The exponential-family structure guarantees that the posterior mean and covariance are related to the log-partition function by standard identities, and it enables the application of classical results from information geometry, in particular, the Fisher Information Matrix.

Proposition 1 (FIM-Jacobian-Posterior Covariance Correspondence). Let $\hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } ) = \mathbb { E } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ]$ denote the optimal (MMSE) denoiser. The Fisher Information Matrix (FIM) of the denoising distribution with respect to $\mathbf { x } _ { t }$ satisfies

$$
\boxed { \mathcal { T } ( \mathbf { x } _ { t } ) = \frac { \sqrt { \bar { \alpha } _ { t } } } { 1 - \bar { \alpha } _ { t } } \mathbf { J } _ { \hat { \mathbf { x } } _ { 0 } } ( \mathbf { x } _ { t } ) = \frac { \bar { \alpha } _ { t } } { ( 1 - \bar { \alpha } _ { t } ) ^ { 2 } } \operatorname { C o v } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ] , }\tag{3}
$$

where $\begin{array} { r } { \mathbf { J } _ { \hat { \mathbf { x } } _ { 0 } } = \frac { \partial \hat { \mathbf { x } } _ { 0 } \left( \mathbf { x } _ { t } \right) } { \partial \mathbf { x } _ { t } } \in \mathbb { R } ^ { D \times D } } \end{array}$ is the Jacobian of the optimal denoiser.

The underlying Jacobian-posterior-covariance relation in $\operatorname { E q . } \ ( 3 )$ is closely related to classical results on conditional mean estimation in Gaussian noise [6], and per-pixel posterior moments computed from denoiser derivatives [15]. Proposition 1 is the information-geometric reading of these identities, identifying the shared quantity as the Fisher Information of the denoising distribution, which underlies the gating construction developed next.

The FIM quantifies the sensitivity of a distribution to perturbations in its parameters. In the context of the denoising distribution, the primary interest lies in sensitivity with respect to the observation $\mathbf { x } _ { t }$ , specifically, the extent to which the posterior belief $\mathbf { x } _ { \mathrm { 0 } }$ about changes when $\mathbf { x } _ { t }$ is perturbed. Proposition 1 characterizes model sensitivity and uncertainty via the posterior covariance and the denoiser Jacobian. Regions exhibiting high sensitivity correspond to areas where the model is uncertain about the underlying clean signal. Furthermore, local sensitivity, as measured by the total Fisher Information, is proportional to reconstruction uncertainty (mean squared error, MSE), $\mathrm { T r } ( \mathcal { T } ) = \bar { \alpha } _ { t } \big ( 1 - \bar { \alpha } _ { t } \big ) ^ { - 2 } , \mathrm { M S E } ( \mathbf { x } _ { t } )$ . Areas with large Tr(I) indicate high model uncertainty, which are precisely the regions where stochastic correction is most beneficial.

Via Tweedie’s formula, the denoiser Jacobian decomposes as $\begin{array} { r } { \mathbf { J } _ { \hat { \mathbf { x } } _ { 0 } } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( \mathbf { I } _ { D } + } \end{array}$ $\big ( 1 - \bar { \alpha } _ { t } \big ) \mathbf { H } _ { t } \big ( \mathbf { x } _ { t } \big ) \big )$ , where $\mathbf { H } _ { t } ( \mathbf { x } _ { t } ) : = \nabla _ { \mathbf { x } _ { t } } \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t )$ approximates the Hessian of the log-marginal density. Substituting into $\operatorname { E q . } ( 3 )$ yields the posterior in terms of the log-marginal density Hessian:

$$
\mathrm { C o v } [ { \bf x } _ { 0 } \mid { \bf x } _ { t } ] = \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } \big ( { \bf I } _ { D } + ( 1 - \bar { \alpha } _ { t } ) { \bf H } _ { t } ( { \bf x } _ { t } ) \big ) .\tag{4}
$$

Extracting the diagonal gives the per-pixel posterior variance:

$$
v _ { i } ( \mathbf { x } _ { t } , t ) : = \big [ \mathrm { C o v } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ] \big ] _ { i i } = \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } \big ( 1 + ( 1 - \bar { \alpha } _ { t } ) [ \mathbf { H } _ { t } ] _ { i i } \big ) .\tag{5}
$$

The sign of $[ \mathbf { H } _ { t } ] _ { i i }$ determines the local uncertainty regime. Strong concavity $( [ \mathbf { H } _ { t } ] _ { i i } \ll 0 )$ indicates that the marginal distribution is sharply peaked along coordinate i, resulting in a small $v _ { i }$ and thus high confidence. In contrast, positive curvature is indicative of high uncertainty.

Validity and clamping. The right-hand side of $\operatorname { E q . }$ 4 defines a valid covariance matrix if and only if all eigenvalues satisfy $\lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { t } ^ { \star } ) \geq - ( 1 - \bar { \alpha } _ { t } ) ^ { - 1 }$ , a condition met by the true log-marginal Hessian $\mathbf { H } _ { t } ^ { \star }$ by construction. Since the true Hessian $\mathbf { H } _ { t } ^ { \star }$ is not accessible, it is approximated using either the Hutchinson estimator or a learned Hessian network, i.e., $\mathbf { H } _ { t } ( \mathbf { x } _ { t } ) \approx \mathbf { H } _ { t } ^ { \star } ( \mathbf { x } _ { t } )$ . The approximate Hessian may not satisfy the eigenvalue condition, which can result in the right-hand side of Eq. 4 failing to be positive semi-definite (PSD). Such violations, however, only occur in regions where the true variance $v _ { i } ^ { * }$ is already close to zero (see App. Appendix E). In these cases, $v _ { i }$ is clamped as $v _ { i } \gets \operatorname* { m a x } ( v _ { i } , \epsilon )$ with $\epsilon = 1 0 ^ { - 6 }$ ， causing the afected pixels to default to deterministic dynamics.

## 3.2 Probabilistic Gating Function

The per-pixel variance $v _ { i } ( \mathbf { x } _ { t } , t )$ is mapped to a gating function $g _ { t , i } \in [ 0 , 1 ]$ that regulates local noise injection.

Definition 1 (Gating Function). The gate at pixel i and timestep t is defined as the probability that the squared denoising error exceeds a tolerance τ :

$$
g _ { t , i } : = \mathbb { P } \big ( ( X _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 } > \tau \big | \mathbf { X } _ { t } = \mathbf { x } _ { t } \big ) .\tag{6}
$$

Assuming (A1) that the denoiser approximates the minimum mean squared error (MMSE) estimator $, ~ \hat { x } _ { 0 , i } \approx \mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ]$ , as justified by the squared-error training objective, and $\left( { \mathrm { A 2 } } \right)$ that the per-pixel posterior is approximately Gaussian (the maximum-entropy distribution given the first two moments from the score network), the residual $R _ { i } = X _ { 0 , i } - { \hat { x } } _ { 0 , i } \mid \mathbf { x } _ { t }$ is a zero-mean Gaussian with variance $v _ { i }$ . Standardization under these assumptions yields a closed-form expression for the gating function.

Proposition 2 (Closed-Form Gating). Under Assumptions $( A 1 ) – ( A 2 )$ , the gating function yields the closed form

$$
g _ { t , i } ~ = ~ 2 \Bigl ( 1 - \varPhi \bigl ( \sqrt { \tau / v _ { i } ( \mathbf { x } _ { t } , t ) } \bigr ) \Bigr ) ,\tag{7}
$$

where $\varPhi ( \cdot )$ is the standard normal CDF.

The closed-form expression depends on $v _ { i }$ solely through the ratio $\tau / v _ { i }$ meaning that τ determines the threshold at which a pixel transitions from the stochastic regime $( v _ { i } \gg \tau , g _ { t , i }  1 )$ to the deterministic regime $( v _ { i } \ll \tau$ $g _ { t , i } \to 0 )$ . The clamping operation $v _ { i } \gets$ max $( v _ { i } , \epsilon )$ ensures the argument remains real-valued. Assumption (A2) holds exactly in the high signal-to-noise ratio (SNR) limit (see App. Appendix F.3), and empirical results in Sec. 4.3 indicate that the spatial gating pattern is robust to the choice of residual distribution. Re-derivation of the gate under Laplace residuals (App. Appendix G) afects only the sharpness of the stochastic-to-deterministic transition.

## 3.3 KL-Optimal Per-Pixel Noise Injection

The gating variable $g _ { t , i }$ determines whether a pixel receives stochastic correction. Conventional samplers employ a spatially uniform scale $\sigma _ { t } ^ { 2 } \in \{ \tilde { \beta } _ { t } , \beta _ { t } \}$ , which does not account for local reconstruction quality or the model’s prediction error. Instead, the proposed approach derives the scale that minimizes the expected Kullback-Leibler (KL) divergence between the true reverse transition $q ( x _ { t - 1 , i } \mid \mathbf { x } _ { t } , x _ { 0 , i } )$ and the model transition $p _ { \theta } ( x _ { t - 1 , i } \mid \mathbf { x } _ { t } )$ , with the model mean fixed at the Tweedie estimate $\hat { \mu } _ { t , \cdot }$ <sub>i</sub> (see App. Appendix H).

Proposition 3 (KL-Optimal Per-Pixel Variance). With the model mean fixed at $\hat { \mu } _ { t , i }$ , the per-pixel variance that minimizes the expected KL to $q ( x _ { t - 1 , i } \ |$ $\mathbf { x } _ { t } , x _ { 0 , i } )$ is

$$
\gamma _ { t , i } ^ { * } = \tilde { \beta } _ { t } + c _ { t } ^ { 2 } v _ { i } ( \mathbf { x } _ { t } , t ) , \qquad c _ { t } : = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } .\tag{8}
$$

The optimal variance has two terms with structurally diferent roles. This distinction drives how we allocate stochasticity in space. The term $\tilde { \beta } _ { t }$ is the variance of the true reverse posterior, the irreducible stochasticity of the reverse SDE, present even when the denoiser is perfect $( v _ { i } = 0 )$ . The term $c _ { t } ^ { 2 } v _ { i }$ , by contrast, equals the expected squared error of the Tweedie mean (App. Appendix I),

$$
\mathbb { E } \big [ \big ( \tilde { \mu } _ { t , i } - \hat { \mu } _ { t , i } \big ) ^ { 2 } \big | \mathrm { \mathbf { x } } _ { t } \big ] \ = \ c _ { t } ^ { 2 } v _ { i } .\tag{9}
$$

Since SANI substitutes $\hat { \mu } _ { t , i }$ for the unknown true posterior mean, this variance is required for the transition to remain calibrated to the true posterior. It is determined by the mean estimate and is not a tunable parameter. This asymmetry motivates our spatial allocation strategy, the calibration term $c _ { t } ^ { 2 } v _ { i }$ is applied at every pixel, while only the exploratory term $\tilde { \beta } _ { t }$ is gated.

$$
{ \cal { \Sigma } } _ { t , i } ^ { 2 } = \underbrace { c _ { t } ^ { 2 } v _ { i } } _ { \mathrm { m e a n - e r r o r ~ c o r r e c t i o n } } + \underbrace { g _ { t , i } { \tilde { \beta } } _ { t } } _ { \mathrm { g a t e d ~ e x p l o r a t i o n } } .\tag{10}
$$

Gating only $\tilde { \beta } _ { t }$ ensures that mean-error correction is preserved at pixels where the substituted Tweedie mean is least reliable. For confident pixels, both $g _ { t , i }$ and $v _ { i }$ are small, resulting in $\Sigma _ { t , i } ^ { 2 } \to 0$ and the update reducing to the deterministic DDIM step. In contrast, for uncertain pixels, $g _ { t , i } \to 1$ and $\Sigma _ { t , i } ^ { 2 } \to \gamma _ { t , i } .$ , corresponding to the KL-optimal variance.

Scope of the optimality. We emphasize that $\gamma _ { t , i } ^ { * }$ is KL-optimal for the fully stochastic per-pixel transition with the mean fixed at the Tweedie estimate. The gated variance $\Sigma _ { t , i } ^ { 2 }$ coincides with $\gamma _ { t , i } ^ { * }$ only in the limit $g _ { t , i } = 1$ . For $g _ { t , i } < 1$ ， and after the admissibility clip (App. Appendix J), it departs from the KLoptimal value in exchange for deterministic updates at confident pixels. $\gamma _ { t , i } ^ { * }$ therefore serves as the stochastic endpoint of the interpolation rather than a global optimality guarantee for SANI.

## 3.4 Spatially Adaptive Noise Injection (SANI) Update

By substituting Eq. 10 into the generalized reverse step from [22], the SANI update is obtained.

Proposition 4 (SANI Update). Let $\hat { \mathbf { x } } _ { 0 } = \bar { \alpha } _ { t } ^ { - 1 / 2 } \big ( \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) \big )$ denote the Tweedie estimate and $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . With $\Sigma _ { t , i } ^ { 2 }$ as in Eq. 10, the SANI update at pixel i is

$$
x _ { t - 1 , i } = \sqrt { \bar { \alpha } _ { t - 1 } } \hat { x } _ { 0 , i } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } - \itSigma _ { t , i } ^ { 2 } } \epsilon _ { \theta , i } + \itSigma _ { t , i } z _ { i } ,\tag{11}
$$

$f o r t > 1$ , with $\mathbf { z } = \mathbf { 0 } ~ f o r ~ t = 1$

The full procedure is summarized in Algorithm 1. g

Coupled dynamics. The coupling between stochastic and deterministic components is explicit in Eq. 11. As $\textstyle \sum _ { t , i }$ increases, the noise term $\textstyle \sum _ { t , i } z _ { i }$ becomes larger, while the direction coeficient $\sqrt { 1 - \bar { \alpha } _ { t - 1 } - \Sigma _ { t , i } ^ { 2 } }$ decreases. Because $\textstyle \sum _ { t , i } ^ { 2 } =$ $c _ { t } ^ { 2 } v _ { i } + g _ { t , i } { \tilde { \beta } } _ { t }$ increases with both $v _ { i }$ and $g _ { t , i } .$ , the predicted direction is automatically down-weighted and replaced by stochastic exploration in regions where the Tweedie mean is unreliable. The same quantity that increases the variance in $\operatorname { E q . 9 }$ also indicates the unreliability of the direction. Therefore, a decoupled rule that adjusts only the variance would continue to follow an inaccurate direction with full weight.

Computational cost. Evaluating the gate requires the Hessian diagonal $[ \mathbf { H } _ { t } ] _ { i i }$ Using the lightweight pretrained network of [18], this adds a single forward pass of a model substantially smaller than the score network, i.e., a small constant overhead per step. Alternatively, a single Hutchinson probe (one Jacobian-vector product per step) requires no auxiliary network at roughly $2 \times$ the cost of a DDIM step. All remaining SANI operations are elementwise in the pixels. Details are provided in App. Appendix K.

Algorithm 1 SANI Sampling   
Require: x<sub>T</sub> $\sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , model ϵ<sub>θ</sub>, schedule $\{ \bar { \alpha } _ { t } , \tilde { \beta } _ { t } , c _ { t } \}$ , tolerance τ , floor ϵ   
for $t = T , \dots , 1$ do   
$\mathbf { z } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) { \mathrm { ~ i f ~ } } t > 1 .$ , else $\mathbf { z } = \mathbf { 0 }$   
$\hat { \epsilon } \gets \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) ; \quad \hat { \mathbf { x } } _ { 0 } \gets \bar { \alpha } _ { t } ^ { - 1 / 2 } ( \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \hat { \epsilon } )$   
v ← diag Cov[x<sub>0</sub> | x<sub>t</sub>]; v<sub>i</sub> ← max(v<sub>i</sub>, ϵ) {per-pixel variance, Eq 5}   
$\mathbf { g } _ { t }  2 \big ( \mathbf { 1 } - \varPhi ( \sqrt { \tau / \mathbf { v } } ) \big )$ {per-pixel gate, Eq. 7}   
$\Sigma _ { t } ^ { 2 }  c _ { t } ^ { 2 } \mathbf v + \mathbf g _ { t } \odot \tilde { \beta } _ { t } ; ~ \Sigma _ { t } ^ { 2 }  \operatorname* { m i n } ( \Sigma _ { t } ^ { 2 } , 1 - \bar { \alpha } _ { t - 1 } )$ {noise injection, Eq. 10}   
$\mathbf { x } _ { t - 1 }  \sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathbf { x } } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } - \Sigma _ { t } ^ { 2 } \odot \hat { \pmb { \epsilon } } + \Sigma _ { t } \odot \mathbf { z }$   
end for   
return $\mathbf { x } _ { 0 }$

## 4 Experiments

The experimental evaluation follows the logical structure of the proposed method. First, we validate empirically the theoretical identities from Sec. 3.1. Next, we show the alignment of the gating function with image structure without any edge supervision (Sec. 4.2). After confirming the intended mechanism, we test the sole distributional approximation on which the method depends (Sec. 4.3) and characterize the gating function along its two axes, the tolerance τ and the reverse-process time (Sec. 4.4). Finally, we evaluate generation quality (Sec. 4.5), showing that spatial adaptivity improves FID over stochastic samplers and we conclude with a decoupling ablation (Sec. 4.6) to isolate the contributions of per-pixel gating, the asymmetric allocation of the calibration term, and the coupling between injected variance and the direction term.

## 4.1 The FIM-Jacobian-Covariance Correspondence

Proposition 1 establishes that the Fisher Information of the denoising distribution, the denoiser Jacobian, and the posterior covariance are proportional (Eq. 3), and that the total Fisher Information equals the reconstruction MSE up to a schedule-dependent factor. These identities provide the theoretical foundation for treating the posterior covariance as a proxy for local reconstruction uncertainty. We verify them on held-out validation images. At three noise levels we compute and visualize the one-step Tweedie prediction $\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } . } \end{array}$ , the denoiser Jacobian diagonal $[ \mathbf { J } _ { \hat { \mathbf { x } } _ { 0 } } ] _ { i i }$ , the posterior covariance diagonal $[ \mathrm { C o v } [ { \bf x } _ { 0 } ~ \vert ~ { \bf x } _ { t } ] ] _ { i i } ,$ and the per-pixel reconstruction accuracy, Fig. 1. The Jacobian and posterior covariance maps coincide up to a global scale at every noise level, as $\operatorname { E q . }$ 3 requires. Since this proportionality is exact under Tweedie’s identity, the agreement also confirms that the Hessian estimators used downstream introduce no systematic bias. Reconstruction accuracy is lowest where these maps are largest, which is the empirical content of the trace identity. As the noise decreases from 80% to $2 0 \% ,$ all three maps sharpen and concentrate on edges and contours, tracking the denoiser’s increasing precision.

![](images/3ddeba2354be1b24da67bea64e83155fb67ec8ef724000dcef9345f8eb6ce7fb.jpg)

![](images/b8d2c62a560636e87b26f5f8af0f200b17819e99c27f4749db061c804479d4e8.jpg)  
Fig. 1: Empirical validation of Prop. 1 on LSUN Bedroom. Columns: decreasing noise (80%, 50%, 20%) and the clean image $\mathbf { x } _ { 0 }$ Rows: (1) one-step prediction xˆ<sub>0</sub>; (2) denoiser Jacobian diagonal $[ \mathbf { J } _ { \hat { \mathbf { x } } _ { 0 } } ] _ { i i } ; ( \mathbf { 3 } )$ posterior covariance diagonal $[ \mathrm { C o v } [ { \bf x } _ { 0 } \ | \ { \bf x } _ { t } ] ] _ { i i } ;$ (4) per-pixel reconstruction accuracy. Rows 2-3 are proportional $\left( \operatorname { E q . 3 } \right)$ and both are spatially anticorrelated with reconstruction accuracy.  
Fig. 2: One-step predictions at three noise levels (80%, 50%, 20%) and the corresponding gating map $( { \mathrm { d a r k } } = 0 ,$ bright = 1) on LSUN Bedroom. The gate assigns high stochasticity to edges and textures, near-zero to flat regions, with the separation sharpening as noise decreases.

## 4.2 Gating Maps Align with Image Geometry

We now test whether the gate $g _ { t , i }$ derived from it allocates stochasticity to the geometrically complex regions of an image, through controlled-noise maps, correlation with per-pixel error, and alignment with edges during sampling.

Spatial allocation of stochasticity. Fig. 2 presents gating maps for LSUN Bedroom validation images at three noise levels. The gate assigns values near 1 to complex regions (furniture contours, painting edges) and near 0 to smooth regions (walls, uniform bed sheets). As noise decreases, the separation sharpens, outlining the image’s geometric structure and concentrating stochasticity at the most challenging features. This behavior results solely from the per-pixel gating function estimated by the pretrained denoising network.

Correlation with reconstruction error. We compute the per-pixel gate $g _ { t , i }$ and the validation loss $( x _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 }$ on held-out images, measuring their log-space Pearson correlation at six noise levels $\left( \mathrm { F i g . \ 3 } \right)$ . The correlation increases monotonically as noise decreases indicating that pixels with $\mathrm { h i g h } { - g _ { t , i } }$ are reconstructed less accurately. At high noise, the lower correlation reflects that most gate values are compressed near 1, reducing variation across pixels.

Alignment with edge structure during sampling. The previous tests use controlled noise levels. Fig. 4 compares the gate to the Sobel gradient magnitude of $\hat { \mathbf { x } } _ { 0 }$ along the reverse trajectory. At $t = 8 0 0$ , the model is globally uncertain and $g _ { t , i } \approx 1$ everywhere. As sampling progresses $( t = 4 0 0 \mathrm { t o } 1 0 0 )$ , flat regions with $| \nabla \hat { \mathbf { x } } _ { 0 } | \approx 0$ shift to $g _ { t , i } \approx 0$ , while edges and fine details retain high $g _ { t , i }$ . At $t = 2 5 , g _ { t , i } \approx 0$ globally, with residual stochasticity only at the highest-gradient pixels. This map identifies structural dificulty without requiring explicit training, as other methods do [21].

![](images/9f8837f1badffa03251034577f6df99de4112ba1cf582899c95b833794b0eaed.jpg)

![](images/124cda2b316aea41f4cd451a59630a22d0b44c64159168d5ec84f35bdfdb04e4.jpg)

![](images/d14d1bd8125ba192a9c6c64f772c493aa23a71d5ed9a1c3d4cc4d7e4b88029f2.jpg)

![](images/6cc102a3312f454e7153d2568a4e5d4ba0594797a3ad6fbc044956affab641a7.jpg)

![](images/f0aa063c65b2f021ebee06d77881d8eb4c1ab9ced994859c84630def6661881a.jpg)

![](images/e5ac21a791442888bd9a992d1be60d626441157d4572f4ac37c9d1ec252723b8.jpg)  
Fig. 3: Per-pixel gating value $g _ { t , i }$ vs. validation loss $( x _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 }$ on LSUN Bedroom at six noise levels (left to right: decreasing noise). The log-space Pearson correlation rises monotonically from 0.33 to 0.78

![](images/8136976425de062c3fafa63ea1af8a38dd681debc0bb4ca3de06299144cc457a.jpg)  
(a) t = 800

![](images/bb99917ae587a621dc19a8566fea3acc8197969acdf145ce1ca7733aa44303be.jpg)  
(b) t = 400

![](images/464fda38c3da8507c516290d326144d01a3793c60c47a8ef0396ff2721b6bb0e.jpg)  
(c) t = 200

![](images/6b0f0c1195102a56e384c578565e4154537c84535f29853dc592130e8d3f1dc0.jpg)  
(d) t = 100

![](images/7c6cec6eed6af9507937c97aae0bd8689b0318331bf9b4f4880d3b3845a9f64b.jpg)  
(e) t = 25  
Fig. 4: Per-pixel gating $_ { g _ { t , i } }$ vs. Sobel edge map of $\hat { \mathbf { x } } _ { 0 }$ along a single reverse trajectory on LSUN Bedroom. Each subplot shows, for the same generated sample at the indicated timestep: the predicted image (left), the Sobel gradient magnitude |∇xˆ<sub>0</sub>| (center), and the gate ${ \mathit { g } } _  t , $ <sub>i</sub> (right; blue = 0, red = 1). The gate is near 1 in high noise, progressively aligns with edges and textures during denoising, and collapses to ≈ 0 everywhere except for the highest-gradient pixels at the end of the trajectory.

## 4.3 Validating the Residual-Distribution Assumption

To validate our assumption that the per-pixel posterior residual $R _ { i } = X _ { 0 , i } - { \hat { x } } _ { 0 , i } \ |$ $\mathbf { x } _ { t }$ is Gaussian with variance $v _ { i }$ , we compare the empirical distribution of the standardized residuals, $Z _ { i } = R _ { i } / \sqrt { v _ { i } }$ to Gaussian and Laplace unit-variance references at four representative timesteps, Fig. 5. The empirical standardized residuals are sharply peaked at high noise, resembling the unit-variance Laplace distribution, and converge toward $\mathcal { N } ( 0 , 1 )$ as t decreases. The Kolmogorov-Smirnov (KS) distance to the Gaussian drops from 0.23 at t=800 to 0.03 at t=25. This matches the high-SNR limit of Lemma 2 in Appendices, where the Gaussian forward kernel dominates the posterior. Re-deriving the gate under Laplace residuals (App. Ap pendix G) preserves the spatial gating structure and only afects the sharpness of the stochastic-to-deterministic transition. Sampling results confirm that the Gaussian and Laplace gates $( { \mathrm { S A N I } } ^ { G }$ and $\mathrm { S A N I } ^ { L }$ , Tab. 1) yield comparable FID.

![](images/0d39fcccffc418f680ad46d46e2a4031beab591789b4c854bd002368c65bcf5f.jpg)

![](images/d41f954daeea5dd3ef9f10cdb9004e204d047a370f4e845c073be7940de8db0a.jpg)

![](images/fff45585e9348a360bb694500b092870b1a2028e0e862acaca50918fba39246a.jpg)

![](images/841de07fff301ef47e97f812c73674e0fdf1534daa5ded0adff99e818a582dc6.jpg)  
Fig. 5: Density histograms of standardized residuals $Z _ { i } = R _ { i } / \sqrt { v _ { i } }$ on CIFAR10 (CS) against Gaussian and Laplace unit-variance reference distributions at four reverse-process timesteps. The empirical density is sharply peaked at high noise and concentrates toward the Gaussian as t decreases.

![](images/a93bcbeb08e14f77388ce5a8d0b7d90d5b0816799bf0640fff9813bf98ac38cc.jpg)  
(a) $\tau = 1 0 ^ { - 4 }$

![](images/2ed5b56f8b1faea8fdcae0897744ef3b41dbbd2d7785d1f30d22cb617fabeead.jpg)  
(b) $\tau = 1 0 ^ { - 3 }$

![](images/9cb1ce910224e0b146bf235b14b1c8426634a95f948e7e69126fb964112a1e72.jpg)  
(c) $\tau = 1 0 ^ { - 2 }$

![](images/6ff17d61458b25e0155a5c114b688507272b5afb09aebd46924daae4fd128fad.jpg)  
(d) $\tau = 1 0 ^ { - 1 }$  
Fig. 6: Spatial distribution of gating values $g _ { t , i }$ along the full reverse trajectory on CIFAR-10 (CS), for four tolerances (subplot). Each subplot overlays five timesteps t ∈ {1000, 500, 250, 125, 20}, with density on a logarithmic scale. The distribution is a near-point mass at $t = 1 0 0 0$ (spatially uniform uncertainty under pure noise), spreads over [0, 1] at intermediate timesteps (spatial discrimination between confident and uncertain regions), and collapses toward $g = 0$ at the final steps. Increasing τ by a decade translates this stochastic-to-deterministic transition earlier along the trajectory without changing its shape, as predicted by the dependence of the gate on $\tau / v _ { i }$ alone.

## 4.4 Behavior of the Gating Function

We analyze the gate along two axes of variation. The tolerance τ , which determines its operating point, and reverse-process time t, which governs its evolution. Fig. 6 presents the spatial distribution of gating values across the full reverse trajectory for four tolerances, with density shown a logarithmic scale. The general pattern is consistent. A near-point mass at t=1000, broad support over [0, 1] mid-trajectory as the gate distinguishes between resolved and unresolved regions, and a collapse toward 0 at the end. Increasing τ shifts the stochastic→deterministic transition earlier, as expected, since the gate depends only on $\tau / v _ { i }$ . Fig. 7 summarizes the dynamics per-step mean and variance across datasets. The band is widest in the middle, reflecting peak spatial heterogeneity.

## 4.5 Quantitative Evaluation

Table 1 reports FID scores across diferent sampling steps, likelihoods are reported in Appendices, Appendix O. We evaluate two gate variants: (i) $\operatorname { S A N I } ^ { G }$ , which uses the Gaussian residual assumption, and (ii) $\mathrm { S A N I } ^ { L }$ , which follows the Laplace assumption (App. Appendix G.1). Since SANI interpolates between stochastic and deterministic regimes on a per pixel basis, it does not fit into either family, so we compare it to both.

![](images/566a724f7caa0394596f4056424580f91a96fb87e0c7cf705c891dc61682127a.jpg)  
Fig. 7: Per-step gating statistics over the reverse process at $\tau = 1 0 ^ { - 3 }$ , averaged over 1000 generated images per dataset. Solid line: spatial mean $\bar { g } _ { t } ;$ dark band: $\bar { g } _ { t } \pm \sigma _ { g , t }$ Light band: per-step [min, max]. The mean decreases monotonically from 1 (stochastic) to 0 (deterministic) on every dataset, and the band is widest at intermediate timesteps where spatial heterogeneity peaks.

A primary test for per-pixel interpolation is whether it outperforms the two samplers it interpolates. $\operatorname { S A N I } ^ { G }$ consistently achieves lower FID than both vanilla DDPM and vanilla DDIM across all dataset and step-count configurations, often by a significant margin. So adjusting noise levels spatially is consistently as good as, and typically outperforms, to applying a uniform global regime.

The Gaussian and Laplace gates perform comparably across all three settings, with neither showing clear superiority. This confirms that sampling is insensitive to the residual model. SANI also remains competitive in NLL compared to variance-learning baselines (App. Appendix O), indicating that spatial gating does not sacrifice likelihood for sample quality. Additionally, the per-pixel gating maps (Fig. 4) ofer interpretable uncertainty diagnostics.

## 4.6 Ablation: Spatial Allocation and Coupling

To isolate the contributions of the probabilistic gating mechanism and the spatially adaptive variance formulation, we perform a decoupling ablation on CelebA-64 (Tab. 2). We compare standard endpoint samplers, ungated spatial variances (with a fixed gate), and various decoupled gated combinations. The results show that combining optimal per-pixel variance with the probabilistic gating mechanism (SANI) consistently achieves the best FID across all evaluated step counts, confirming that both components are essential for optimal performance.

## 5 Related Work

Difusion models and sampling. [8, 24] established the foundational framework of difusion as a hierarchical latent variable model. [22] revealed the probability flow ODE connection, enabling deterministic sampling. [10] refined sampling via schedule design and higher-order solvers. Our work extends this line of research by introducing a spatial degree of freedom that interpolates between deterministic and stochastic dynamics at the pixel level.

Table 1: FID (↓) across datasets and sampling steps. Bold: best among stochastic samplers (DDPM family and SANI). Underline: best in the DDIM family. SANI values are additionally underlined when they outperform the best DDIM-family sampler.
<table><tr><td rowspan="2">Method</td><td colspan="5">DDPM family</td><td colspan="5">DDIM family</td></tr><tr><td>10</td><td>25</td><td>50</td><td>100</td><td>200</td><td>10</td><td>25</td><td>50</td><td>100</td><td>200</td></tr><tr><td>CelebA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base, β</td><td>36.69</td><td>24.46</td><td>18.96</td><td>14.31</td><td>10.48</td><td>20.54</td><td>13.54</td><td>9.33</td><td>6.60</td><td>4.96</td></tr><tr><td>A-</td><td>28.99</td><td>16.01</td><td>11.23</td><td>8.08</td><td>6.51</td><td>15.62</td><td>9.22</td><td>6.13</td><td>4.29</td><td>3.46</td></tr><tr><td>NPR-</td><td>28.37</td><td>15.74</td><td>10.89</td><td>8.23</td><td>7.03</td><td>14.98</td><td>8.93</td><td>6.04</td><td>4.27</td><td>3.59</td></tr><tr><td>SN-</td><td>20.60</td><td>12.00</td><td>7.88</td><td>5.89</td><td>5.02</td><td>10.20</td><td>5.48</td><td>3.83</td><td>3.04</td><td>2.85</td></tr><tr><td>OCM-</td><td>21.55</td><td>12.71</td><td>9.24</td><td>6.97</td><td>5.92</td><td>10.28</td><td>5.72</td><td>4.42</td><td>3.54</td><td>3.17</td></tr><tr><td> $\operatorname { S A N I } ^ { G }$ </td><td>K=10: 17.09</td><td></td><td>K=25: 8.87</td><td></td><td>K=50: 5.56</td><td></td><td>K=100: 4.64</td><td></td><td></td><td>K=200: 3.04</td></tr><tr><td>SANIL</td><td>K=10: 16.07</td><td></td><td></td><td>K=25: 9.84</td><td>K=50: 6.25</td><td></td><td>K=100: 3.93</td><td></td><td></td><td>K=200: 4.52</td></tr><tr><td colspan="9">CIFAR-10 (LS)</td><td></td><td></td></tr><tr><td>Base, β A-</td><td>44.45</td><td>21.83</td><td>15.21</td><td>10.94</td><td>8.23</td><td>21.31</td><td>10.70</td><td>7.74</td><td>6.08</td><td>5.07</td></tr><tr><td>NPR-</td><td>34.26</td><td>11.60</td><td>7.25 6.18</td><td>5.40 4.52</td><td>4.01</td><td>14.00 13.34</td><td>5.81 5.38</td><td>4.04 3.95</td><td>3.55</td><td>3.39</td></tr><tr><td>SN-</td><td>32.35 24.06</td><td>10.55 6.91</td><td>4.63</td><td>3.67</td><td>3.57 3.11</td><td>12.19</td><td>4.28</td><td>3.39</td><td>3.53 3.23</td><td>3.42</td></tr><tr><td>OCM-</td><td>24.94</td><td>9.19</td><td>5.95</td><td>4.36</td><td>3.48</td><td>10.66</td><td>4.35</td><td>3.48</td><td>3.27</td><td>3.22 3.29</td></tr><tr><td>SANIG</td><td></td><td>K=10: 15.27</td><td></td><td>K=25: 6.64</td><td></td><td>K=50: 6.50</td><td></td><td>K=100: 3.82</td><td></td><td></td></tr><tr><td>SANIL</td><td></td><td>K=10: 17.96</td><td></td><td>K=25: 8.13</td><td></td><td>K=50: 5.65</td><td>K=100: 4.85</td><td></td><td></td><td>K=200: 3.99</td></tr><tr><td>CIFAR-10 (CS)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>K=200: 5.85</td><td></td></tr><tr><td colspan="9"></td><td></td><td></td><td></td></tr><tr><td>Base, β A-</td><td>34.76 22.94</td><td>16.18</td><td>11.11</td><td>8.38</td><td>6.66</td><td>34.34</td><td>16.68</td><td>10.48</td><td>7.94</td><td>6.69</td></tr><tr><td></td><td></td><td>8.50</td><td>5.50</td><td>4.45</td><td>4.04</td><td>26.43</td><td>9.96</td><td>6.02</td><td>4.88</td><td>4.92</td></tr><tr><td>NPR-</td><td>19.94</td><td>7.99</td><td>5.31</td><td>4.52</td><td>4.10</td><td>22.81</td><td>9.47</td><td>6.04</td><td>5.02</td><td>5.06</td></tr><tr><td>SN-</td><td>16.33</td><td>6.05</td><td>4.17</td><td>3.83</td><td>3.72</td><td>17.90</td><td>7.36</td><td>5.16</td><td>4.63</td><td>4.63</td></tr><tr><td>OCM-</td><td>14.32</td><td>5.54</td><td>4.10</td><td>3.84</td><td>3.75</td><td>16.70</td><td>6.71</td><td>4.72</td><td>4.30</td><td>4.54</td></tr><tr><td>SANIG</td><td>K=10: 17.01</td><td></td><td></td><td>K=25: 7.35</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SANIL</td><td>K=10: 16.37</td><td></td><td></td><td></td><td>K=50: 4.95</td><td></td><td>K=100: 3.83</td><td></td><td></td><td>K=200: 4.17</td></tr><tr><td></td><td></td><td></td><td></td><td>K=25: 6.55</td><td>K=50: 4.88</td><td></td><td>K=100: 4.50</td><td></td><td></td><td>K=200: 3.52</td></tr></table>

Variance learning. Improved DDPM [17] first proposed learning Σ via ${ \mathcal { L } } _ { \mathrm { v l b } }$ Analytic-DPM [3] derived the optimal scalar variance. [2] explored diagonal covariances. OCM [18] estimates diagonal covariance by matching the score Hessian. However, OCM applies learned variances uniformly, all pixels follow the same regime at each step. SANI employs Hessian-diagonal estimates for a distinct purpose: deriving per-pixel gating that enables spatially heterogeneous interpolation within a single image.

Adaptive sampling. Dpm-Solver [13] and Dpm-Solver++ [14] adapt step spacing temporally but not spatially. [12] skip computations in easy regions for eficiency, but do not modulate the stochastic properties of the process. MU-LAN [20] learns a multivariate noise schedule during training. In contrast, SANI derives noise modulation from the pre-trained score network at inference time.

Table 2: Decoupling ablation on CelebA-64. FID (↓). Gated variants report the best FID across a τ sweep at each step count, endpoint and ungated variants are τ -independent.
<table><tr><td>Variant/#Timesteps</td><td>10</td><td>25</td><td>50</td><td>100 200</td></tr><tr><td>Endpoint samplers</td><td></td><td></td><td></td><td></td></tr><tr><td>DDIM DDPM  $( { \tilde { \beta } } _ { t } )$ </td><td>20.5413.549.336.604.96 36.69 24.4618.9614.31 10.48</td><td></td><td></td><td></td></tr><tr><td>Ungated variance choices OCM-DDPM  $( \gamma ^ { * }$  on noise only) Uniform  $\gamma ^ { * }$  (both terms)</td><td>21.55 12.71 9.24 6.97 34.13 18.97 13.9810.66</td><td></td><td></td><td>5.92</td></tr><tr><td>Gated variants</td><td></td><td></td><td></td><td>9.70</td></tr><tr><td> $g _ { t , i } , \gamma _ { t , i } ^ { * } \mathrm { - c o u p l e d }$ </td><td>24.74 8.81 7.88 4.74 3.20</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { S A N I - n a i v e } \ ( g _ { t , i } , \tilde { \beta } _ { t } )$ </td><td>22.53 7.32 8.21 4.12</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { S A N I - d e c o u p l e d }$ </td><td>22.09 13.57 10.40 8.70 8.88</td><td></td><td></td><td>3.34</td></tr><tr><td>SANI (proposed)</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>17.09 8.87 5.56 4.64 3.04</td><td></td><td></td><td></td></tr></table>

Uncertainty in difusion models. [5] estimate pixel-wise uncertainty via ensembles, incurring multiple forward passes. [23] show that high-frequency regions benefit from stochastic exploration in wavelet-based compression. [9] use posterior variance for OOD detection. Closest to our use of posterior moments, [15] compute per-pixel posterior moments from derivatives of a pretrained denoiser for uncertainty quantification and posterior exploration in image restoration. The proposed approach derives the same per-pixel uncertainty from the FIM diagonal, but uses it to control the sampler itself, gating the noise injection at inference time.

Information geometry. [1] established links between Fisher Information and model sensitivity. [25] connected denoising autoencoders to score estimation. Tweedie’s formula [7] relates scores to posterior means, and [16] derived uncertainty bounds via the score Hessian. [4] showed denoiser Jacobian norms correlate with sample diversity. This work builds on these identities by interpreting them through the Fisher Information of the denoising distribution and by deriving a practical sampling algorithm from this relationship.

Spatially Adaptive Sampling. A recognized limitation of standard difusion models is the uniform allocation of computational resources and stochasticity across spatially heterogeneous images. Recent work, such as Patch Forcing [21], addresses this limitation by dynamically varying the difusion timestep per patch, advancing “easy” patches faster than “hard” patches. However, such methods typically require architectural modifications, such as training a dedicated dificulty head, and operate at the coarser patch level. In contrast, SANI addresses spatial heterogeneity at inference time without requiring model fine-tuning or structural modifications.

## 6 Conclusion

In this work, we introduced Spatially Adaptive Noise Injection (SANI), a trainingfree inference framework designed to address the spatial uniformity bottleneck in standard difusion samplers. Our approach, grounded in information geometry, establishes a formal theoretical link between the Fisher Information Matrix of the denoising distribution, the denoiser Jacobian, and the posterior covariance. We translated this geometric insight into a tractable, closed-form probabilistic gating function that quantifies the probability the local denoising error exceeds a specific tolerance, enabling SANI to modulate noise injection per-pixel. By coupling this gating mechanism with a spatially adaptive variance, SANI allocates stochastic corrections to uncertain, high-curvature regions while preserving deterministic DDIM-style updates in confident, smooth areas. Empirically, SANI improves generation quality over established DDIM and DDPM samplers across various step counts and is competitive with variance-learning DDPM-family baselines, with the largest gains in the low-step regime. Our decoupling analysis further confirms that the synergy between probabilistic gating and adaptive variance is essential for the method’s performance, especially in low-step regime where standard samplers are less efective.

Limitations and Future Work. While SANI establishes a foundation for spatially adaptive sampling, our results reveal an asymmetric impact of spatial gating across sampling steps. At low step counts, gating overcomes the imprecision of the underlying model. However, as the number of sampling steps increase, the coupled update rule becomes more sensitive to the diagonal Hessian approximation. In addition, our quantitative evaluation is restricted to small, low-resolution benchmarks (CIFAR-10 at 32 × 32 and CelebA at 64 × 64, with qualitative results on LSUN Bedroom at 256 × 256), and the spatially adaptive transition carries no marginal-preservation guarantee. Future work should explore dynamic tolerance scheduling, adjusting the threshold τ as a function of the sampling steps to improve the transition from high-stochasticity exploration to high-fidelity deterministic refinement, as well as validation on high-resolution and latent-space difusion models.

## Acknowledgements

We acknowledge financial support from the Swiss National Science Foundation through the LegoMol project (grant no. 207428). The computations were performed on the Baobab and Yggdrasil cluster at the University of Geneva.

[1] Amari, S.i.: Natural gradient works eficiently in learning. Neural Computation 10(2), 251–276 (1998). https : / / doi . org / 10 . 1162 / 089976698300017746

[2] Bao, F., Li, C., Sun, J., Zhu, J., Zhang, B.: Estimating the optimal covariance with imperfect mean in difusion probabilistic models. In: Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., Sabato, S. (eds.) Proceedings of the 39th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 162, pp. 1555–1584. PMLR (17–23 Jul 2022), https://proceedings.mlr.press/v162/bao22d.html

[3] Bao, F., Li, C., Zhu, J., Zhang, B.: Analytic-dpm: an analytic estimate of the optimal reverse variance in difusion probabilistic models. In: International Conference on Learning Representations (ICLR) (2022), https: //openreview.net/forum?id=ly5M8mE4v49

[4] Daras, G., Chung, H., Lai, C., Mitsufuji, Y., Ye, J.C., Milanfar, P., Dimakis, A.G., Delbracio, M.: A survey on difusion models for inverse problems. CoRR abs/2410.00083 (2024). https://doi.org/10.48550/ARXIV.2410.00083, https://doi.org/10.48550/arXiv.2410.00083

[5] De Vita, M., Belagiannis, V.: Difusion model guided sampling with pixel-wise aleatoric uncertainty estimation. In: Proceedings of the Winter Conference on Applications of Computer Vision (WACV). pp. 3844–3854 (February 2025)

[6] Dytso, A., Poor, H.V., Shitz, S.S.: A general derivative identity for the conditional mean estimator in gaussian noise and some applications. In: 2020 IEEE International Symposium on Information Theory (ISIT). pp. 1183–1188. IEEE (2020)

[7] Efron, B.: Tweedie’s formula and selection bias. Journal of the American Statistical Association 106(496), 1602–1614 (2011)

[8] Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 6840–6851. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/paper\_files/ paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

[9] Jazbec, M., Wong-Toi, E., Xia, G., Zhang, D., Nalisnick, E., Mandt, S.: Generative uncertainty in difusion models. In: Chiappa, S., Magliacane, S. (eds.) Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence. Proceedings of Machine Learning Research, vol. 286, pp. 1837– 1858. PMLR (21–25 Jul 2025), https://proceedings.mlr.press/v286/ jazbec25a.html

[10] Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of difusion-based generative models. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A. (eds.) Advances in Neural Information Processing Systems. vol. 35, pp. 26565–26577. Curran Associates, Inc. (2022)

[11] Kingma, D., Salimans, T., Poole, B., Ho, J.: Variational difusion models. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems. vol. 34, pp. 21696– 21707. Curran Associates, Inc. (2021), https://proceedings.neurips.cc/ paper\_files/paper/2021/file/b578f2a52a0229873fefc2a4b06377fa-Paper.pdf

[12] Liu, Z., Yang, Y., Zhang, C., Zhang, Y., Qiu, L., You, Y., Yang, Y.: Regionadaptive sampling for difusion transformers (2025), https://openreview. net/forum?id=gCZYei3PLL

[13] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for difusion probabilistic model sampling in around 10 steps. In: Advances in Neural Information Processing Systems (2022)

[14] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver++: Fast solver for guided sampling of difusion probabilistic models. Machine Intelligence Research 22(4), 730–751 (Jun 2025). https://doi.org/10.1007/s11633- 025-1562-4, http://dx.doi.org/10.1007/s11633-025-1562-4

[15] Manor, H., Michaeli, T.: On the posterior distribution in denoising: Application to uncertainty quantification. In: International Conference on Learning Representations. vol. 2024, pp. 49233–49263 (2024)

[16] Meng, C., Song, Y., Li, W., Ermon, S.: Estimating high order gradients of the data distribution by denoising. In: Advances in Neural Information Processing Systems. vol. 34, pp. 25364–25377 (2021)

[17] Nichol, A.Q., Dhariwal, P.: Improved denoising difusion probabilistic models (2021), https://openreview.net/forum?id=-NEXDKk8gZ

[18] Ou, Z., Zhang, M., Zhang, A., Xiao, T.Z., Li, Y., Barber, D.: Improving probabilistic difusion models with optimal diagonal covariance matching. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=fV0t65OBUu

[19] Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4195–4205 (2023)

[20] Sahoo, S.S., Gokaslan, A., De, C., Kuleshov, V.: Difusion models with learned adaptive noise. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 105730–105779. Curran Associates, Inc. (2024). https://doi.org/10.52202/079017-3354

[21] Schusterbauer, J., Gui, M., Li, Y., Ma, P., Krause, F., Ommer, B.: Denoising, fast and slow: Dificulty-aware adaptive sampling for image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026)

[22] Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. In: International Conference on Learning Representations (2021), https:// openreview.net/forum?id=St1giarCHLP

[23] Song, J., He, J., Yang, L., Feng, M., Wang, K.: High frequency matters: Uncertainty guided image compression with wavelet difusion. arXiv preprint arXiv:2407.12538 (2024)

[24] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic diferential equations. In: International Conference on Learning Representations (2021), https: //openreview.net/forum?id=PxTIG12RRHS

[25] Vincent, P.: A connection between score matching and denoising autoencoders. Neural Computation 23(7), 1661–1674 (2011)

## Appendix for “Spatially Adaptive Noise Injection”

## A Notation

Let $\mathbf { x } = ( x ^ { 1 } , \ldots , x ^ { D } ) ^ { \top } \in \mathbb { R } ^ { D }$ denote a point in D-dimensional Euclidean space (represented as a column vector), and let $\textstyle { \mathrm { T r } } ( \mathbf { A } ) = \sum _ { i } A _ { i i }$ denote the trace of $\mathbf { A } \in \mathbb { R } ^ { k \times k }$ . For a scalar function $f : \mathbb { R } ^ { D } \to \mathbf { \overline { { \mathbb { R } } } } , \nabla _ { \mathbf { x } } \overline { { f ( \mathbf { x } ) } } \in \mathbb { R } ^ { D }$ is the gradient and $\nabla _ { \mathbf { x } } ^ { 2 } f ( \mathbf { x } ) \in \mathbb { R } ^ { D \times D }$ is the Hessian. For a vector-valued function $f : \mathbb { R } ^ { k }  \mathbb { R } ^ { m }$ $\partial f / \partial \mathbf { x } \in \mathbb { R } ^ { m \times k }$ is the Jacobian. Throughout we write $\begin{array} { r } { \bar { \alpha } _ { t } \ = \ \prod _ { s = 1 } ^ { t } \alpha _ { s } , \ \tilde { \beta } _ { t } \ = } \end{array}$ $( 1 - \bar { \alpha } _ { t - 1 } ) \beta _ { t } / ( 1 - \bar { \alpha } _ { t } )$ , and $c _ { t } = \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } / ( 1 - \bar { \alpha } _ { t } )$

## B Recovering DDPM from the Generalized Update

Proposition A1. Setting $\begin{array} { r } { \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \big ( 1 - \frac { \bar { \alpha } _ { t } } { \bar { \alpha } _ { t - 1 } } \big ) } \end{array}$ in the generalized update $E q .$ 2 yields $\sigma _ { t } ^ { 2 } = \tilde { \beta } _ { t }$

Proof. From $\bar { \alpha } _ { t } = \bar { \alpha } _ { t - 1 } \alpha _ { t }$ , we have $1 - \bar { \alpha } _ { t } / \bar { \alpha } _ { t - 1 } = 1 - \alpha _ { t } = \beta _ { t }$ , and substitution gives $\begin{array} { r } { \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } = \tilde { \beta } _ { t } } \end{array}$ □

Remark. The DDIM paper [22] uses $\alpha _ { t }$ to denote the cumulative product, which corresponds to $\bar { \alpha } _ { t }$ in this work, with the convention $\alpha _ { 0 } : = 1$ . Eq. 2 is presented to ensure consistency with the main text.

## C Exponential-Family Structure of the Denoising Distribution

Proposition A2 (Exponential Family). The denoising distribution $p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } )$ belongs to an exponential family with canonical form

$$
p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ) = h ( \mathbf { x } _ { 0 } ) \exp \bigl ( \eta ( \mathbf { x } _ { t } , t ) ^ { \top } \mathbf { T } ( \mathbf { x } _ { 0 } ) - \psi ( \mathbf { x } _ { t } , t ) \bigr ) ,
$$

where

$$
\begin{array} { r l } & { h ( \mathbf { x } _ { 0 } ) = q ( \mathbf { x } _ { 0 } ) , \quad \eta ( \mathbf { x } _ { t } , t ) = \left( \frac { \sqrt { \bar { \alpha } _ { t } } } { 1 - \bar { \alpha } _ { t } } \mathbf { x } _ { t } , \mathbf { \xi } - \frac { \bar { \alpha } _ { t } } { 2 ( 1 - \bar { \alpha } _ { t } ) } \right) , } \\ & { \mathbf { T } ( \mathbf { x } _ { 0 } ) = ( \mathbf { x } _ { 0 } , \| \mathbf { x } _ { 0 } \| ^ { 2 } ) , \quad \psi ( \mathbf { x } _ { t } , t ) = \log p _ { t } ( \mathbf { x } _ { t } ) + \frac { D } { 2 } \log ( 2 \pi ( 1 - \bar { \alpha } _ { t } ) ) + \frac { \| \mathbf { x } _ { t } \| ^ { 2 } } { 2 ( 1 - \bar { \alpha } _ { t } ) } . } \end{array}
$$

Proof. By Bayes’ rule $p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ) = p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) q ( \mathbf { x } _ { 0 } ) / p _ { t } ( \mathbf { x } _ { t } )$ , with the Gaussian forward kernel $p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = \mathcal { N } ( \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } ) ,$ q the data distribution, and and $\begin{array} { r } { p _ { t } ( \mathbf { x } _ { t } ) = \int p ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) q ( \mathbf { x } _ { 0 } ) d \mathbf { x } _ { 0 } } \end{array}$ is the marginal distribution at time t. Expanding and collecting terms in $\mathbf { x } _ { 0 } \mathbf { : }$

$$
\begin{array} { r } { p ( { \bf x } _ { t } \mid { \bf x } _ { 0 } ) = \exp \Bigl ( - \frac { D } { 2 } \log \bigl ( 2 \pi ( 1 - \bar { \alpha } _ { t } ) \bigr ) - \frac { \| { \bf x } _ { t } \| ^ { 2 } } { 2 ( 1 - \bar { \alpha } _ { t } ) } \Bigr ) \exp \Bigl ( - \frac { \bar { \alpha } _ { t } } { 2 ( 1 - \bar { \alpha } _ { t } ) } \| { \bf x } _ { 0 } \| ^ { 2 } + \frac { \sqrt { \bar { \alpha } _ { t } } } { 1 - \bar { \alpha } _ { t } } { \bf x } _ { t } ^ { \top } { \bf x } _ { 0 } \Bigr ) . } \end{array}
$$

Substituting and identifying with the canonical form gives the stated parameters.

## D Proof of Proposition 1: FIM-Jacobian-posterior Covariance Correspondence

Proof. For an exponential family with natural parameters η, the FIM with respect to η equals the covariance of the suficient statistics: $\mathcal { T } ( \pmb { \eta } ) = \mathrm { C o v } [ \mathbf { T } ( \mathbf { x } _ { 0 } ) \ | \ \mathbf { x } _ { t } ]$ Since $\mathbf { x } _ { t }$ parameterizes the distribution via $\pmb { \eta } ( \mathbf { x } _ { t } )$ , the change-of-variables formula gives

$$
\begin{array} { r } { \boldsymbol { \mathcal { T } } ( \mathbf { x } _ { t } ) = \left( \nabla _ { \mathbf { x } _ { t } } \pmb { \eta } \right) ^ { \top } \boldsymbol { \mathcal { T } } ( \pmb { \eta } ) \big ( \nabla _ { \mathbf { x } _ { t } } \pmb { \eta } \big ) . } \end{array}\tag{A1}
$$

From Proposition $\mathrm { A 2 } ,$ the Jacobian of the first component of η with respect to $\mathbf { x } _ { t }$ is $\frac { \sqrt { { \bar { \alpha } } _ { t } } } { 1 - { \bar { \alpha } } _ { t } } \mathbf { I } _ { D }$ and the second component is constant in $\mathbf { x } _ { t }$ . The relevant block of $\mathcal { T } ( \eta )$ <sup>t</sup>is $\operatorname { C o v } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ]$ . Substituting into (A1):

$$
\mathcal { T } ( \mathbf { x } _ { t } ) = \frac { \bar { \alpha } _ { t } } { ( 1 - \bar { \alpha } _ { t } ) ^ { 2 } } \operatorname { C o v } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ] .\tag{A2}
$$

By Tweedie’s identity [7], $\begin{array} { r } { \operatorname { C o v } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ] = \frac { 1 - \bar { \alpha } _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \mathbf { J } _ { \hat { \mathbf { x } } _ { 0 } } ( \mathbf { x } _ { t } ) } \end{array}$ . Substituting into (A2) yields the result. □

Trace identity. Taking the trace of (A2) yields

$$
\mathrm { T r } \left( \boldsymbol { \mathcal { T } } ( \mathbf { x } _ { t } ) \right) = \frac { \bar { \alpha } _ { t } } { ( 1 - \bar { \alpha } _ { t } ) ^ { 2 } } \mathrm { M S E } ( \mathbf { x } _ { t } ) ,\tag{A3}
$$

where $\mathrm { M S E } ( \mathbf { x } _ { t } ) : = \mathbb { E } [ \| \mathbf { x } _ { 0 } - \hat { \mathbf { x } } _ { 0 } \| ^ { 2 } \ | \ \mathbf { x } _ { t } ] = \mathrm { T r } ( \mathrm { C o v } [ \mathbf { x } _ { 0 } \ | \ \mathbf { x } _ { t } ] )$ is the optimal reconstruction error. Eq. A3 certifies that the total Fisher information is proportional to the MMSE, so regions of large $\operatorname { T r } ( \mathcal { T } )$ are exactly the regions where the model is uncertain about the underlying signal.

Relation to known identities. Combining Eq. (A2) with Tweedie’s identity recovers, in the difusion parameterization, the general derivative identity for the conditional mean estimator in Gaussian noise of Dytsoet al. [6], which relates derivatives of the posterior mean to posterior moments and Manoret al. [15] exploit the same relation to compute per-pixel posterior moments from a pretrained denoiser for uncertainty quantification. Proposition 1 adds the information-geometric interpretation of this quantity as the Fisher Information of the denoising distribution.

## E PSD Condition and Self-Limiting Violations

Assumption 1 (PSD Condition). For the true log-marginal Hessian ${ \bf H } _ { t } ^ { \star } ( { \bf x } _ { t } ) =$ $\nabla _ { \mathbf { x } _ { t } } ^ { 2 } \log p _ { t } ( \mathbf { x } _ { t } )$ , all eigenvalues satisfy $\lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { t } ^ { \star } ) \geq - ( 1 - \bar { \alpha } _ { t } ) ^ { - 1 }$

Since $\operatorname { C o v } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ]$ is PSD by definition and the prefactor $( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t }$ in Eq. 4 is positive, ${ \bf \cal I } _ { D } + ( 1 - \bar { \alpha } _ { t } ) { \bf H } _ { t } ^ { \star }$ must be PSD, which is equivalent to $\lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { t } ^ { \star } ) \geq$ $- ( 1 - \bar { \alpha } _ { t } ) ^ { - 1 }$ . Assumption 1 is therefore not an additional restriction on the data distribution; it is a structural consequence of the forward process. Its role is to make explicit the spectral condition that the true Hessian automatically satisfies, so that we can diagnose when a learned approximation $\mathbf { H } _ { t } \approx \mathbf { H } _ { t } ^ { \star }$ may violate it.

Proposition A3 (Self-Limiting PSD Violations). Let $\delta _ { i } : = [ \mathbf { H } _ { t } ] _ { i i } - [ \mathbf { H } _ { t } ^ { \star } ] _ { i i }$ denote the per-pixel approximation error. A PSD violation at pixel $i \ ( i . e . , v _ { i } < 0 $ in $E q . \ 5 )$ requires $[ \mathbf { H } _ { t } ^ { \star } ] _ { i i } + \delta _ { i } < - ( 1 - \bar { \alpha } _ { t } ) ^ { - 1 }$ . For bounded error $| \delta _ { i } | \leq \varDelta$ , the true per-pixel variance at any violating pixel satisfies

$$
v _ { i } ^ { \star } = \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } \bigl ( 1 + ( 1 - \bar { \alpha } _ { t } ) [ \mathbf { H } _ { t } ^ { \star } ] _ { i i } \bigr ) \ \leq \ \frac { ( 1 - \bar { \alpha } _ { t } ) ^ { 2 } } { \bar { \alpha } _ { t } } \varDelta .\tag{A4}
$$

Proof. A PSD violation means $1 + ( 1 - \bar { \alpha } _ { t } ) [ \mathbf { H } _ { t } ] _ { i i } < 0$ . Writing $[ \mathbf { H } _ { t } ] _ { i i } = [ \mathbf { H } _ { t } ^ { \star } ] _ { i i } + \delta _ { i } ;$

$$
1 + ( 1 - { \bar { \alpha } } _ { t } ) [ \mathbf { H } _ { t } ^ { \star } ] _ { i i } < - ( 1 - { \bar { \alpha } } _ { t } ) \delta _ { i } \leq ( 1 - { \bar { \alpha } } _ { t } ) | \delta _ { i } | \leq ( 1 - { \bar { \alpha } } _ { t } ) \Delta ,
$$

since a violation requires $\delta _ { i }$ negative with $| \delta _ { i } |$ large. Multiplying by the positive prefactor yields (A4). □

Interpretation. At late timesteps $( 1 - \bar { \alpha } _ { t } \ll 1 )$ , the bound (A4) is extremely small: PSD violations can only arise where the true variance is already negligible. At early timesteps, the bound is loose, but the gate saturates at $g _ { t , i } \approx 1$ for all pixels regardless, so the precise value of $v _ { i }$ is irrelevant. In both regimes, clamping $v _ { i } \gets \operatorname* { m a x } ( v _ { i } , \epsilon )$ correctly defaults violating pixels to near-deterministic dynamics. Empirically, the clamp is active for $< 0 . 1 \%$ of pixels at any timestep when using the Hessian network of Ou et al. [18].

## F Gating Function: Derivation and Justification of Assumptions

The following section restates the assumptions, provides justification, and proves the closed-form equation for the gating function $\left( \operatorname { E q . 2 } \right)$

## F.1 Assumptions

(A1) MMSE Optimality. $\hat { x } _ { 0 , i } \approx \mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ]$ , justified by the squared-error training objective $\mathcal { L } _ { \mathrm { s i m p l e } }$ , whose Bayes-optimal solution is the MMSE estimator, $\hat { x } _ { 0 , i } \approx \mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ]$

(A2) Gaussian Marginal Residual. $R _ { i } \mid \mathbf { x } _ { t } \approx { \mathcal { N } } ( 0 , v _ { i } )$ with $R _ { i } : = X _ { 0 , i } - { \hat { x } } _ { 0 , i } .$

The denoiser network determines only the first two moments of $R _ { i } .$ , so (A2) is a modeling choice (specifically, the maximum-entropy distribution given those moments) rather than a consequence of the model. The justification proceeds in two steps.

## F.2 Residual Moments

Lemma 1 (Residual Moments). Under (A1), $\mathbb { E } [ R _ { i } \ | \ \mathbf { x } _ { t } ] = 0$ and $\mathrm { V a r } ( R _ { i } \ )$ $\mathbf { x } _ { t } ) = v _ { i } ( \mathbf { x } _ { t } , t )$

Proof. Since $\hat { x } _ { 0 , i } = \mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ]$ is a deterministic function of $\mathbf { x } _ { t } , \mathbb { E } [ R _ { i } \ | \ \mathbf { x } _ { t } ] =$ $\mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ] - \hat { x } _ { 0 , i } = 0$ , and $\operatorname { V a r } ( R _ { i } \mid \mathbf { x } _ { t } ) = \operatorname { V a r } ( X _ { 0 , i } \mid \mathbf { x } _ { t } ) = v _ { i }$ □

Lemma 1 fixes only the first two moments of $R _ { i }$ but infinitely many distributions are consistent with them. Step 2 justifies the Gaussian shape as a limit.

## F.3 Posterior Concentration as a Gaussian

Lemma 2 (Posterior as Prior × Gaussian Kernel). The denoising posterior factorizes as

$$
p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ) = \frac { 1 } { Z ( \mathbf { x } _ { t } ) } q ( \mathbf { x } _ { 0 } ) \exp \biggl ( - \frac { \lambda _ { t } } { 2 } \| \mathbf { x } _ { 0 } - \pmb { \mu } _ { t } \| ^ { 2 } \biggr ) ,\tag{A5}
$$

where $\lambda _ { t } : = \bar { \alpha } _ { t } / ( 1 - \bar { \alpha } _ { t } )$ is the precision, $\mu _ { t } : = \mathbf { x } _ { t } / \sqrt { \bar { \alpha } _ { t } }$ , and $Z ( \mathbf { x } _ { t } )$ normalizes. Proof. $\mathrm { B y }$ Bayes’ rule $p ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ) \propto q ( \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } )$ , with $p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } )$ ∝ $\exp ( - \| \mathbf { x } _ { t } - \| \mathbf { x } _ { t } - { }$ $\sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } \| ^ { 2 } / \big ( 2 ( 1 - \bar { \alpha } _ { t } ) \big ) \big )$ . Completing the square in $\mathbf { x } _ { \mathrm { 0 } } \mathbf { : }$

$$
\frac { \| \mathbf { x } _ { t } - \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } \| ^ { 2 } } { 2 ( 1 - \bar { \alpha } _ { t } ) } = \frac { \bar { \alpha } _ { t } } { 2 ( 1 - \bar { \alpha } _ { t } ) } \| \mathbf { x } _ { 0 } - \mathbf { x } _ { t } / \sqrt { \bar { \alpha } _ { t } } \| ^ { 2 } + C ( \mathbf { x } _ { t } ) ,
$$

where $C ( \mathbf { x } _ { t } )$ is absorbed into $Z ( \mathbf { x } _ { t } )$

Why this justifies (A2). The decomposition (A5) holds exactly for any data distribution q. The Gaussian kernel arises from the forward process and is not an approximation. Assumption (A2) states that this kernel dominates the prior in shaping the per-pixel marginal, a claim controlled by $\lambda _ { t }$ . For large $\lambda _ { t }$ (late reverse timesteps), the kernel becomes high concentrated near $\pmb { \mu } _ { t }$ , allowing the smooth prior to be locally well-approximated by a constant and the per-pixel marginal then inherits the Gaussian shape of the kernel. For small $\lambda _ { t }$ (early timesteps), the prior dominates and the marginal may be strongly non-Gaussian. However, in this regime $v _ { i } \gg \tau$ for all pixels and $g _ { t , i } \approx 1$ universally (Figure 7), making the gate insensitive to the distributional shape of $R _ { i }$ . Thus, the gate relies on (A2) only in the intermediate regime where it actively discriminates. Even in this case, Sec. 4.3 shows that the spatial gating pattern remains consistent across diferent residual distributions.

## F.4 Proof of Proposition 2(Closed-Form Gating)

Proof. Under (A1), $\hat { x } _ { 0 , i } = \mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ]$ , so the squared error equals $R _ { i } ^ { 2 }$ and

$$
\mathbb { P } \big ( ( X _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 } > \tau \mid \mathbf { x } _ { t } \big ) = \mathbb { P } \big ( | R _ { i } | > \sqrt { \tau } \mid \mathbf { x } _ { t } \big ) .\tag{A6}
$$

Under (A2) and Lemma 1, $R _ { i } \ | \ \mathbf { x } _ { t } \sim \mathcal N ( 0 , v _ { i } )$ . The clamp $v _ { i } \gets \operatorname* { m a x } ( v _ { i } , \epsilon )$ guarantees $\sqrt { v _ { i } } > 0$ , so we may define $Z : = R _ { i } / \sqrt { v _ { i } } \sim \mathcal { N } ( 0 , 1 )$ . Then $| R _ { i } | >$ $\sqrt { \tau } \iff | Z | > \sqrt { \tau / v _ { i } }$ , and by symmetry of the standard normal,

$$
\mathbb { P } ( | Z | > a ) = 2 ( 1 - \phi ( a ) ) \quad \mathrm { f o r ~ a n y ~ } a \geq 0 .\tag{A7}
$$

Combining (A6) and (A7) with $a = \sqrt { \tau / v _ { i } }$ yields Eq. 7.

## G Gating Under Alternative Residual Distributions

Using the same gating function definition $g _ { t , i } ~ = ~ \mathbb { P } \big ( R _ { i } ^ { 2 } > \tau ~ | ~ \mathbf { x } _ { t } \big ) ~ = ~ \mathbb { P } \big ( | R _ { i } | >$ $\sqrt { \tau } \mid \mathbf { x } _ { t } )$ , we re-derive the gate for an alternative residual model, the Laplace distribution parameterized to have zero mean and variance $v _ { i } ,$ matching Lemma 1. The Laplace distribution has sharper peak and heavier tails than the Gaussian.

## G.1 Laplace Residual

Proposition A4 (Laplace Gating). If $R _ { i } \ | \ \mathbf { x } _ { t }$ ∼ Laplace $( 0 , b _ { i } )$ with $b _ { i } =$ $\sqrt { { v _ { i } } / { 2 } }$ (matching $\begin{array} { r } { \mathrm { V a r } ( R _ { i } ) = v _ { i } ) ; } \end{array}$ then

$$
g _ { t , i } ^ { \mathrm { L a p } } = \exp \bigl ( - \sqrt { 2 \tau / v _ { i } } \bigr ) .\tag{A8}
$$

Proof. The Laplace survival function is $\mathbb { P } ( | R _ { i } | > a ) = \exp ( - a / b _ { i } )$ for $a \geq 0$ Setting $a = { \sqrt { \tau } }$ and substituting $b _ { i } = \sqrt { v _ { i } / 2 }$ yields Eq. A8. □

## G.2 Comparison

Both gates exhibit the same boundary behavior $( g \to 0$ as $v _ { i } / \tau  0 , g  1$ as $v _ { i } / \tau  \infty )$ and difer only in their tail decay, as shown in Table A1. Heavier-tailed distributions assign more probability to large denoising errors, so for a given $\tau / v _ { i }$ the ordering at large $\rho$ is $g ^ { \mathrm { G a u s s } } < g ^ { \mathrm { L a p } }$ (for small $\nu )$ . At moderate $\rho ,$ the ordering is non-monotone. The Laplace distribution’s sharper peak concentrates mass near zero, reducing the probability in the tails. For $\rho \lesssim 3$ , the Gaussian gate is higher than the Laplace, with a crossover point where the heavier tails of the Laplace dominate.

Table A1: Gating functions under alternative residual models. $\rho : = \tau / v _ { i }$ , Φ is the standard normal CDF, $I _ { x }$ is the regularized incomplete beta function.
<table><tr><td colspan="2">Distribution Gating function  $g _ { t , i }$  Tail decay  $( \rho \to \infty )$ </td></tr><tr><td>Gaussian</td><td> $2 ( 1 - \varPhi ( \sqrt { \rho } ) )$ </td></tr><tr><td>Laplace  $\exp ( - \sqrt { 2 \rho } )$ </td><td> $\begin{array} { l } { { \sim e ^ { - \rho / 2 } / \sqrt { \rho } } } \\ { { \sim e ^ { - \sqrt { 2 \rho } } } } \end{array}$ </td></tr></table>

Empirical results (Sec. 4.3, Tab. 1) show that $\operatorname { S A N I } ^ { G }$ and $\mathrm { S A N I } ^ { L }$ achieve similar FID scores, confirming that the spatial gating structure is consistent across these choices.

## H Proof of Proposition 3 (KL-Optimal Per-Pixel Variance)

Setup. The true reverse posterior at pixel $i ,$ conditioned on $\mathbf { x } _ { t }$ and $x _ { 0 , i } ,$ is given by [8]

$$
q ( x _ { t - 1 , i } \mid \mathbf { x } _ { t } , x _ { 0 , i } ) = \mathcal { N } \big ( \tilde { \mu } _ { t , i } ( \mathbf { x } _ { t } , x _ { 0 , i } ) , \tilde { \beta } _ { t } \big ) ,\tag{A9}
$$

with $\begin{array} { r } { \tilde { \mu } _ { t , i } ( \mathbf { x } _ { t } , x _ { 0 } ) = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } x _ { 0 , i } + \frac { \sqrt { \alpha _ { t } } ( 1 - \bar { \alpha } _ { t - 1 } ) } { 1 - \bar { \alpha } _ { t } } x _ { t , i } } \end{array}$ . The model transition is $p _ { \theta } ( x _ { t - 1 , i }$ | $\mathbf x _ { t } ) = \mathcal N ( \hat { \mu } _ { t , i } , ( \sigma _ { t , i } ^ { \theta } ) ^ { 2 } )$ , where $\hat { \mu } _ { t , i }$ is obtained by substituting $\hat { x } _ { 0 , i }$ for $x _ { 0 , i }$ in $\tilde { \mu } _ { t , i }$

Step 1: Mean-error decomposition. Writing $x _ { 0 , i } = \hat { x } _ { 0 , i } + e _ { i }$ with $e _ { i } : = x _ { 0 , i } - { \hat { x } } _ { 0 , i } \colon$

$$
\begin{array} { r } { \tilde { \mu } _ { t , i } = \hat { \mu } _ { t , i } + c _ { t } e _ { i } , \qquad c _ { t } = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } . } \end{array}\tag{A10}
$$

Under (A1), $\mathbb { E } [ e _ { i } \mid \mathbf { x } _ { t } ] = 0$ and $\mathbb { E } [ e _ { i } ^ { 2 } \mid { \bf x } _ { t } ] = v _ { i }$ . Therefore,

$$
\mathbb { E } \big [ ( \tilde { \mu } _ { t , i } - \hat { \mu } _ { t , i } ) ^ { 2 } \mid \mathbf { x } _ { t } \big ] = c _ { t } ^ { 2 } v _ { i } .\tag{A11}
$$

Step 2: Expected KL. For two univariate Gaussian distributions $\mathcal { N } ( \mu _ { q } , \sigma _ { q } ^ { 2 } )$ and $\mathcal { N } ( \mu _ { p } , \sigma _ { p } ^ { 2 } )$ , the Kullback-Leibler divergence is given by $\mathrm { K L } = \log ( \sigma _ { p } / \sigma _ { q } ) \stackrel { \mathbf { \sigma } } { + } ( \sigma _ { q } ^ { 2 } +$ $( \mu _ { q } - \mu _ { p } ) ^ { 2 } ) / ( 2 \sigma _ { p } ^ { 2 } ) - 1 / 2$ . by setting $\mu _ { q } = \tilde { \mu } _ { t , i } , \sigma _ { q } ^ { 2 } = \tilde { \beta } _ { t } , \mu _ { p } = \hat { \mu } _ { t , i } , \sigma _ { p } ^ { 2 } = ( \sigma _ { t , i } ^ { \bar { \theta _ { i } } } ) ^ { 2 }$ and taking the expectation over $x _ { 0 , i } \mid \mathbf { x } _ { t }$ , using Eq. A11:

$$
\mathbb { E } [ \mathrm { K L } ] = \log \frac { \sigma _ { t , i } ^ { \theta } } { \sqrt { \tilde { \beta } _ { t } } } + \frac { \tilde { \beta } _ { t } + c _ { t } ^ { 2 } v _ { i } } { 2 ( \sigma _ { t , i } ^ { \theta } ) ^ { 2 } } - \frac { 1 } { 2 } .\tag{A12}
$$

Step 3: Optimization. Diferentiating Eq. A12 with respect to $( \sigma _ { t , i } ^ { \theta } ) ^ { 2 }$

$$
\frac { \partial } { \partial ( \sigma _ { t , i } ^ { \theta } ) ^ { 2 } } \mathbb { E } [ \mathrm { K L } ] = \frac { 1 } { 2 ( \sigma _ { t , i } ^ { \theta } ) ^ { 2 } } - \frac { \tilde { \beta } _ { t } + c _ { t } ^ { 2 } v _ { i } } { 2 ( \sigma _ { t , i } ^ { \theta } ) ^ { 4 } } .
$$

Setting the derivative to zero yields $( \sigma _ { t , i } ^ { \theta } ) _ { \mathrm { o p t } } ^ { 2 } = \tilde { \beta } _ { t } + c _ { t } ^ { 2 } v _ { i } = \gamma _ { t , i } ^ { * }$ , which is Eq. 8. The second derivative at the optimum is $1 / ( 2 \gamma _ { t , i } ^ { * 4 } ) > 0$ , confirming that this point is a minimum.

Consistency with prior work. Averaging Eq. 8 over pixels and applying the score-Hessian relation recovers the scalar optimal variance of Bao et al. [3], up to the diagonal approximation. This derivation extends their result to a per-pixel quantity, which is the form required for spatially adaptive gating.

Recovery of standard samplers. This update recovers DDIM in the joint limit $g _ { t , i } ~ = ~ 0 , ~ v _ { i } ~ = ~ 0$ (confident pixels), and the KL-optimal stochastic sampler when $g _ { t , i } = 1 \ : ( \Sigma _ { t , i } ^ { 2 } = \gamma _ { t , i } ^ { * } )$ . It also recovers DDPM when $g _ { t , i } = 1$ and $v _ { i } = 0$ (perfect denoiser, $\Sigma _ { t , i } ^ { 2 } = \tilde { \beta } _ { t } )$ . A spatially uniform gate $g _ { t , i } = \eta$ implements the η-interpolation of [22], but with a per-pixel, data-aware variance instead of the fixed $\tilde { \beta } _ { t }$

Scope of optimality. As discussed in Sec. 3.3, the optimality of $\gamma _ { t , i } ^ { * }$ concerns the fully stochastic per-pixel transition with the mean fixed at the Tweedie estimate. The deployed SANI variance $\begin{array} { r } { \Sigma _ { t , i } ^ { 2 } = c _ { t } ^ { 2 } v _ { i } + g _ { t , i } \tilde { \beta } _ { t } } \end{array}$ equals $\gamma _ { t , i } ^ { * }$ only when $g _ { t , i } = 1$ and the admissibility clip of Appendix J may reduce it further. SANI as a whole therefore carries no global KL-optimality guarantee.

Relation to the DDIM family. For a scalar, state-independent noise scale, the non-Markovian family of [22] preserves the marginals of the forward process. The SANI variance $\Sigma _ { t , i } ^ { 2 }$ is per-pixel and depends on $\mathbf { x } _ { t }$ through $v _ { i }$ and $g _ { t , i } ,$ so this marginal-preservation argument does not carry over, and we do not claim that the spatially adaptive transition matches the forward marginals. SANI should therefore be understood as a per-pixel update rule whose endpoints coincide with the DDIM and DDPM updates, supported empirically in Sec. 4

## I Mean-Error Identity and the Asymmetric Allocation

The mean-error identity $\left( \operatorname { E q . 9 } \right)$ stated in the main text is proved here. This result underpins the rationale for applying the calibration term $c _ { t } ^ { 2 } v _ { i }$ in $\Sigma _ { t , : } ^ { 2 }$ <sub>i</sub> uniformly, while only $\tilde { \beta } _ { t }$ is gated.

Proposition A5 (Mean-Error Identity). Let $\hat { \mu } _ { t , i }$ denote the model’s posterior mean obtained by substituting the Tweedie estimate $\hat { x } _ { 0 , i }$ for $x _ { 0 , i }$ in the true posterior mean, and let $\tilde { \mu } _ { t , i }$ denote the true posterior mean. Under (A1),

$$
\mathbb { E } \big [ ( \tilde { \mu } _ { t , i } - \hat { \mu } _ { t , i } ) ^ { 2 } \mid \mathbf { x } _ { t } \big ] = c _ { t } ^ { 2 } v _ { i } ( \mathbf { x } _ { t } , t ) .\tag{A13}
$$

Proof. As in Step 1 of Appendix H: substituting $x _ { 0 , i } = \hat { x } _ { 0 , i } + e _ { i }$ into $\tilde { \mu } _ { t , i }$ yields $\tilde { \mu } _ { t , i } = \hat { \mu } _ { t , i } + c _ { t } e _ { i } ,$ so $( \tilde { \mu } _ { t , i } - \hat { \mu } _ { t , i } ) ^ { 2 } = c _ { t } ^ { 2 } e _ { i } ^ { 2 }$ . Under (A1), $\hat { x } _ { 0 , i } = \mathbb { E } [ X _ { 0 , i } \mid \mathbf { x } _ { t } ]$ is a deterministic function of $\mathbf { x } _ { t } ,$ so $\mathbb { E } [ e _ { i } ^ { 2 } \mid \mathbf { x } _ { t } ] = \operatorname { V a r } ( X _ { 0 , i } \mid \mathbf { x } _ { t } ) = v _ { i }$ □

Implication for the asymmetric allocation. Proposition A5 demonstrates that when the posterior variance $v _ { i }$ is large, the model’s mean $\hat { \mu } _ { t , i }$ becomes an unreliable estimate of $\tilde { \mu } _ { t , i }$ , with error variance precisely $c _ { t } ^ { 2 } v _ { i }$ . Two key consequences arise from this observation.

First, the KL-optimal variance $\gamma _ { t , i } ^ { * } = \tilde { \beta } _ { t } + c _ { t } ^ { 2 } v _ { i }$ in Eq. 3 includes this term as a necessary additive component. This term ensures that the transition remains calibrated to the true posterior when $\hat { \mu } _ { t , i }$ is used in place of $\tilde { \mu } _ { t , i }$ . Gating this term $( { \mathrm { i . e . } }$ , scaling it by some $g \in [ 0 , 1 ) ,$ ) would suppress the correction that compensates for the mean error, particularly at pixels where the mean error is largest.

Second, the exploratory term $\beta _ { t }$ represents the irreducible stochasticity of the true reverse SDE. This term persists even with a perfect denoiser $( v _ { i } = 0 )$ The decision to include or suppress this term distinguishes DDIM (where $\tilde { \beta } _ { t }$ is suppressed, resulting in a deterministic process) from DDPM (where $\tilde { \beta } _ { t }$ is retained, resulting in a stochastic process). This is the term that is gated on a per-pixel basis:

$$
\begin{array} { r } { \Sigma _ { t , i } ^ { 2 } = c _ { t } ^ { 2 } v _ { i } + g _ { t , i } \tilde { \beta } _ { t } . } \end{array}
$$

The coupling in the SANI update (Eq. 11) reinforces this point. Since $\Sigma _ { t , i }$ appears in both the noise scale and through $\sqrt { 1 - \bar { \alpha } _ { t - 1 } - \Sigma _ { t , i } ^ { 2 } } .$ , the direction coeficient, increasing $v _ { i }$ simultaneously inflates the noise and reduces the weight of the unreliable direction. The same quantity that increases the variance also indicates the unreliability of the direction, so the update addresses both aspects simultaneously. In contrast, a decoupled rule that adjusted only the variance would continue to follow an inaccurate direction at full weight.

## J Admissibility of the SANI Update

The SANI update (Eq. 11) is well-defined when

$$
0 \leq \Sigma _ { t , i } ^ { 2 } \leq 1 - \bar { \alpha } _ { t - 1 }\tag{A14}
$$

at every pixel, ensuring that both $\textstyle \sum _ { t , i }$ and the direction coeficient $\sqrt { 1 - \bar { \alpha } _ { t - 1 } - \Sigma _ { t , i } ^ { 2 } }$ are real. The lower bound is satisfied due to the clamp $v _ { i } \gets \operatorname* { m a x } ( v _ { i } , \epsilon )$ and the condition $g _ { t , i } \geq 0$ . The following establishes the upper bound.

## J.1 Per-Pixel Bounds

Since $g _ { t , i } \in [ 0 , 1 ] , \Sigma _ { t , i } ^ { 2 } \leq c _ { t } ^ { 2 } v _ { i } + \tilde { \beta } _ { t } = \gamma _ { t , i } ^ { * }$ . It therefore sufices to bound $\gamma _ { t , i } ^ { * }$

Proposition A6 (Per-pixel bounds on $\gamma _ { t , i } ^ { * } ) . ~ \gamma _ { t , i } ^ { * } \geq \tilde { \beta } _ { t }$ , with equality if $v _ { i } ~ = ~ 0$ . Moreover, $\gamma _ { t , i } ^ { * } ~ \leq ~ \beta _ { t } / \alpha _ { t }$ if $[ { \bf { H } } _ { t } ] _ { i i } \ \leq \ 0$ , with equality at $[ { \bf { H } } _ { t } ] _ { i i } = 0$ (equivalently $v _ { i } = ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t } )$

Proof. The lower bound is immediate. For the upper bound, recall from $\mathrm { E q . ~ 5 }$ that $v _ { i } \leq ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t } \iff [ \mathbf { H } _ { t } ] _ { i i } \leq 0$ . Evaluating $\gamma _ { t , i } ^ { * }$ at $v _ { i } = ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t }$ with $c _ { t } = \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } / ( 1 - \bar { \alpha } _ { t } )$ :

$$
\begin{array} { l } { \gamma _ { t , i } ^ { * } \big | _ { v _ { i } = ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t } } = \tilde { \beta } _ { t } + c _ { t } ^ { 2 } \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } = \frac { \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \Big [ ( 1 - \bar { \alpha } _ { t - 1 } ) + \frac { \bar { \alpha } _ { t - 1 } \beta _ { t } } { \bar { \alpha } _ { t } } \Big ] } \\ { = \frac { \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \cdot \frac { 1 - \bar { \alpha } _ { t } } { \alpha _ { t } } = \frac { \beta _ { t } } { \alpha _ { t } } , } \end{array}
$$

where we used $\bar { \alpha } _ { t - 1 } / \bar { \alpha } _ { t } = 1 / \alpha _ { t }$ and $( 1 - \bar { \alpha } _ { t - 1 } ) + \beta _ { t } / \alpha _ { t } = ( 1 - \bar { \alpha } _ { t } ) / \alpha _ { t }$ . Since $\gamma _ { t , i } ^ { * }$ is increasing in $v _ { i } , \gamma _ { t , i } ^ { * } \le \beta _ { t } / \alpha _ { t }$ holds exactly when $v _ { i } \le ( 1 - \bar { \alpha } _ { t } ) / \bar { \alpha } _ { t }$ □

Proposition A6 demonstrates that $\beta _ { t } / \alpha _ { t }$ bounds $\gamma _ { t , i } ^ { * }$ only at pixels of negative local curvature. At pixels of positive curvature, $\gamma _ { t , i } ^ { * }$ may exceed $\beta _ { t } / \alpha _ { t }$ , indicating that the bound is not uniform. However, this value serves as the average-case bound. Averaging $\gamma _ { t , i } ^ { * }$ over all pixels and applying the score-Hessian relation yiels the scalar Analytic-DPM variance $\sigma _ { t } ^ { * 2 }$ , which satisfies $\tilde { \beta } _ { t } \leq \sigma _ { t } ^ { * 2 } \leq \beta _ { t } / \alpha _ { t } \ \mathrm { [ 3 ] }$ ].

## J.2 Schedule Condition

Lemma 3. $\beta _ { t } / \alpha _ { t } \leq 1 - \bar { \alpha } _ { t - 1 } \iff \bar { \alpha } _ { t } \leq 1 - 2 \beta _ { t }$

Proof. $\beta _ { t } / \alpha _ { t } \leq 1 - \bar { \alpha } _ { t - 1 } \iff \beta _ { t } \leq \alpha _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) = \alpha _ { t } - \bar { \alpha } _ { t }$ . With $\beta _ { t } = 1 - \alpha _ { t }$ this is $1 - \alpha _ { t } \le \alpha _ { t } - \bar { \alpha } _ { t } , \mathrm { i . e . , } \bar { \alpha } _ { t } \le 2 \alpha _ { t } - 1 = 1 - 2 \beta _ { t }$ □

For standard schedules where $\beta _ { t } \ll 1$ , the right-hand side approaches 1, so the condition is violated only when $\bar { \alpha } _ { t }$ is close to 1, corresponding to the earliest forward steps (or latest reverse steps). $\mathrm { A t } ~ t = 1 , \bar { \alpha } _ { 0 } = 1$ and the condition fails, however, the algorithm sets $\mathbf { z } = \mathbf { 0 } .$ , so admissibility vacuous. At $t = 2 , 1 - \bar { \alpha } _ { 1 } = \beta _ { 1 }$ and the condition requires $\beta _ { 2 } / \alpha _ { 2 } \leq \beta _ { 1 }$ , which does not hold for monotonically increasing schedules. Numerically, the condition is satisfied for all $t \geq t ^ { * }$ , with $t ^ { * } = 3$ for the linear schedule $( \beta _ { 1 } = 1 0 ^ { - 4 } , \beta _ { T } = 0 . 0 2 , T = 1 0 0 0 )$ and $t ^ { * } = 6$ for the cosine schedule [3].

## J.3 Admissibility and the Per-Pixel Clip

Combining Proposition A6 and Lemma 3: for $t \geq t ^ { * }$ , every pixel with $[ \mathbf { H } _ { t } ] _ { i i } \leq 0$ satisfies

$$
\varSigma _ { t , i } ^ { 2 } \leq \gamma _ { t , i } ^ { * } \leq \frac { \beta _ { t } } { \alpha _ { t } } \leq 1 - \bar { \alpha } _ { t - 1 } ,\tag{A15}
$$

and is admissible by construction. Only pixels with positive local curvature, where $v _ { i }$ exceeds the forward-noise level, can violate Eq. A14. For these cases, the per-pixel clip is applied

$$
\Sigma _ { t , i } ^ { 2 }  \operatorname* { m i n } ( \Sigma _ { t , i } ^ { 2 } , 1 - \bar { \alpha } _ { t - 1 } ) .\tag{A16}
$$

This procedure guarantees a real direction coeficient and preserves $\Sigma _ { t , i } ^ { 2 } \geq 0$

Empirical negligibility. The clip (Eq. A16) is activated for only a negligible fraction of pixels. During sampling, the iterate $\mathbf { x } _ { t }$ remains within high-probability regions, where the log-marginal is locally concave and $[ \mathbf { H } _ { t } ] _ { i i } \leq 0$ at the majority of coordinates. By Proposition $_ \mathrm { A 6 }$ , these are automatically admissible. The remaining cases are further suppressed at timesteps where Eq. A14 is most restrictive. $\mathrm { A t } ~ t = 1$ no noise is injected $( \mathbf { z } = \mathbf { 0 } )$ , and for $t \in \{ 2 , \ldots , t ^ { * } - 1 \}$ the gate satisfies $g _ { t , i } \approx 0$ and $v _ { i }$ is small (the signal-to-noise ratio $\bar { \alpha } _ { t } / ( 1 - \bar { \alpha } _ { t } ) \gg 1 )$ , so $\varSigma _ { t , i } ^ { 2 } \approx c _ { t } ^ { 2 } v _ { i } \approx 0$ regardless of the clip $\mathrm { ( F i g . ~ 7 ) }$

## K Experimental Details

This appendix provides reproducibility details, pretrained-model sources, perdataset hyperparameters, and information about the Hessian-diagonal source and its training setup to support Section 4. The experiments include comparisons with DDPM using $\sigma _ { t } ^ { 2 } = \tilde { \beta } _ { t }$ and $\sigma _ { t } ^ { 2 } = \beta _ { t } \ [ 8 ]$ , DDIM [22], Analytic-DPM (A-DDPM/DDIM) [3], NPR DDPM/DDIM [2] (which models the noise prediction residual), SN-DDPM/DDIM [2] (which models the second moment of the noise), and OCM-DDPM/DDIM [18]. Consistent with [2], the approach relies on a pretrained score-based neural network in fixed parameters throughout our procedures.

Details of Pretrained Models. Table A2 summarizes the pretrained score and Hessian-diagonal network used to estimate $v _ { i } .$ . Score models parameterize the noise prediction, $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t )$ , from which the score is recovered as $\mathbf { s } _ { \theta } ( \mathbf { x } _ { t } ) =$ $\nabla _ { \mathbf x _ { t } }$ <sub>t</sub> log $p _ { \theta } ( \mathbf { x } _ { t } ) = - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) / \sqrt { 1 - \bar { \alpha } _ { t } }$

For the per-pixel posterior variance $v _ { i } ( \mathbf { x } _ { t } , t ) = \left[ \mathrm { C o v } [ \mathbf { x } _ { 0 } \ | \ \mathbf { x } _ { t } ] \right] _ { i i }$ of Eq. 5, the pre-trained Hessian-diagonal networks released by [18] are employed, which predict $[ \mathbf { H } _ { t } ] _ { i i }$ in a single forward pass. These networks are lightweight, being significantly smaller than the score network, and are trained by minimizing $\mathbb { E } _ { t , \mathbf { x } _ { t } , \mathbf { r } } \Vert h _ { \phi } ( \mathbf { x } _ { t } , t ) - \mathbf { r } \odot ( \mathbf { H } _ { t } \mathbf { r } ) \Vert ^ { 2 }$ with Rademacher probes $\mathbf { r } \in \{ - 1 , + 1 \} ^ { D }$ . The same EMA weightsare used at inference as in [18].

The alternative Hutchinson estimator $[ \mathbf { H } _ { t } ] _ { i i } \approx r _ { i } \cdot [ \mathbf { H } _ { t } \mathbf { r } ] _ { i }$ does not require an auxiliary network but introduces one additional Jacobian-vector product (JVP) per step, resulting approximately twice the computational cost of DDIM.

Table A2: Source of pretrained score prediction networks used in our experiments.
<table><tr><td>PROVIDED BY</td></tr><tr><td>CIFAR10 (LS) [3]</td></tr><tr><td>CIFAR10 (CS) [3]</td></tr><tr><td>CELEBA 64X64 [22]</td></tr><tr><td>LSUN BEDROOM [8]</td></tr><tr><td>HESSIAN NET [18]</td></tr></table>

## K.1 Per-Dataset Hyperparameters

The only hyperparameter introduced by SANI is the tolerance τ , which determines the operating point of the gating function via the ratio $\tau / v _ { i }$ . The value of τ is calibrated for each dataset, selected from the range $[ 1 0 ^ { - 4 } - 1 0 ^ { - 2 } ]$ in which the gate exhibits the intended stochastic-to-deterministic transition within the trajectory. Table A3 lists the values used for the main FID and qualitative results.

Table A3: Per-dataset hyperparameters. T is the training-time number of timesteps; K ∈ {10, 25, 50, 100, 200, 1000} at inference is reported in the relevant figures.
<table><tr><td colspan="5">CIFAR-10 (LS) CIFAR-10 (CS) CelebA 64 LSUN Bedroom</td></tr><tr><td>Resolution</td><td> $3 2 \times 3 2$ </td><td> $3 2 \times 3 2$ </td><td> $6 4 \times 6 4$ </td><td> $2 5 6 \times 2 5 6$ </td></tr><tr><td>Schedule</td><td>linear</td><td>cosine</td><td>linear</td><td>linear</td></tr><tr><td>T (training)</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Tolerance τ</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Variance floor €</td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 6 }$ </td></tr></table>

Evaluation. All evaluations employ exponential-moving-average (EMA) weights with decay 0.9999, which is the standard setting in the difusion literature [2, 17, 19]. FID is computed from 50,000 generated samples. For CIFAR-10, the reference statistics are extracted from the entire training set (50,000 images), while for CelebA, the standard test split is used. Likelihoods are reported as variational upper bounds in bits per dimension along the same sub-sequence used at sampling time, ensuring that FID and NLL are reported under matched schedules.

## L Extended Theory Validation

This appendix provides extended visualizations of the theoretical identities across the remaining datasets and presents the complete gate-error correlation scatter to support Section 4.1.

## L.1 FIM-Jacobian-Posterior Covariance Correspondence on Additional Datasets

Figure A1 extends Figure 1 from LSUN Bedroom to CelebA and CIFAR-10 (LS and CS). Despite the lower resolution $( 3 2 \times 3 2$ to $6 4 \times 6 4 )$ , the spatial correspondence between the denoiser Jacobian diagonal (row 2), the posterior covariance diagonal (row 3), and the reconstruction accuracy (row 4) remains clearly visible along object boundaries and persists across diferent noise schedules.

![](images/014d57336b33c83f1d56f8727cafb6bcbc94ff60ab624d4a6a4d448c151dac15.jpg)  
(a) CelebA 64 × 64

![](images/8809b93fbb1453a0446db2f16b7a3bd94c03873115aba79bdad043813ffeb92f.jpg)  
(b) CIFAR-10 (LS)

![](images/c7d77869b1fec3ffad92e1d77309bfcf9f8febc553c0fe87fc9665487df83b6c.jpg)  
(c) CIFAR-10 (CS)  
Fig. A1: FIM-Jacobian-Posterior Covariance correspondence on additional datasets, same layout as Fig. 1. The spatial correspondence between the Jacobian diagonal, the posterior covariance diagonal, and reconstruction accuracy is preserved at all resolutions and across both noise schedules.

## L.2 Gate-Error Correlation Across Noise Levels

Figure A2 presents the per-pixel gating value $g _ { t , i }$ versus the validation loss $( x _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 }$ on held-out images at six noise levels for LSUN Bedroom, CelebA, and CIFAR-10 (LS and CS) datasets. The log-space Pearson correlation rises monotonically from high (80%) noise levels to lower noise levels across all datasets. The lower correlation at high noise due to the compression of $g _ { t , i }$ near 1 across all pixels, resulting in limited variance for correlation. The correlation reported here uses the gating value $g _ { t , i }$ directly, but the same monotonic trend is observed if $v _ { i }$ is substituted for $g _ { t , i } ,$ as the two are related by the monotone CDF transform of Proposition 2.

## M Extended Gating Visualizations

This appendix provides additional results supporting Sec. 4.2 on the remaining datasets. First, the gate is evaluated at controlled levels of forward noise on validation images, which isolates the gate’s behavior from trajectory dynamics by

![](images/1e1981110591cdd8237e7732517302145eeb802ca363b1278f00dc0833dd82b7.jpg)  
(a) CIFAR-10 (LS)

![](images/913a85aae2359edf0c01a535906dfa6ccd67460bea91a9cf4d00118d47066818.jpg)  
(b) CIFAR-10 (CS)

![](images/7ae82626da8684572f216a2cba0e46b1af44d4f80c6123983729ad97a88b047f.jpg)  
(c) CelebA 64

![](images/4cf09139165362d31f065652d00f7a748191584f1a6afb525a1377c444b97d76.jpg)  
(d) LSUN Bedroom  
Fig. A2: Per-pixel gating value $g _ { t , i }$ versus validation loss $( x _ { 0 , i } - \hat { x } _ { 0 , i } ) ^ { 2 }$ at six noise levels (left to right: decreasing noise). The log-space Pearson correlation R rises monotonically, confirming that high-g<sub>t,i</sub> pixels are those the model reconstructs poorly.

removing dependence on the unfolding of the reverse process (see Appendix M.1). Subsequently, the same gate-versus-Sobel analysis is conducted along the reverse trajectory (see Appendix M.2).

## M.1 Gate at Controlled Noise Levels

To isolate the gate’s response from trajectory dynamics, we evaluate it on held-out validation images at controlled levels of forward noise. Given a clean image $\mathbf { x } _ { 0 } ,$ we sample $\mathbf { x } _ { t } \sim q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } )$ at noise levels 80%, 50%, and 20%, and compute the gate from the Tweedie prediction. This setting tests the gate’s behavior as a function of the pair $\left( \mathbf { x } _ { 0 } , t \right)$ alone.

Figure A3 presents the resulting gating maps on three datasets. The spatial selectivity observed in this setting is consistent with the behavior during sampling. The gate saturates near 1 across the image at high noise, sharpens onto geometrically complex regions at moderate noise, and collapses to near-zero on flat regions at low noise. These results confirm that alignment with image structure is a property of the score field at $\left( \mathbf { x } _ { t } , t \right)$ , rather than an artifact of the specific trajectory followed by the sampler.

## M.2 Gate vs. Sobel Along the Reverse Trajectory: Additional Datasets

The gate is next evaluated along an actual reverse trajectory. Figures A4-A7 further supports that SANI efectively isolates “dificult” spatial regions bases solely on inherent geometry, without requiring explicit edge-detection supervision or the architectural modifications proposed in recent work (e.g., [21]).

## N Extended Residual Assumption & Gating Dynamics

This appendix provides additional analysis for Sections 4.3-4.4 extending the discussion to all four datasets.

![](images/d58ec52ea535861873506cab99564655bec5521a6ca9f34cdf0eed5642720ecf.jpg)  
(a) CelebA

![](images/46910194e4a0e1b3d498dcf7bcfb3d61a9283c47d35324b70c8ecbd5ce70a6d7.jpg)  
(b) Cifar10 (LS)

![](images/0d9d34d93d5bfc357cbd88723a1927547ac4c16b65aa0bb872a66db0c5385fe1.jpg)  
(c) Cifar10 (CS)

Fig. A3: Visualization of one-step predictions at varying timesteps (80%, 50%, 20% noise) and the corresponding gating map (dark = 0, bright = 1). The gating function assigns high stochasticity to geometrically complex regions such as edges and textures, and near-zero stochasticity to flat regions such as walls and bed sheets. The spatial separation becomes more pronounced as noise decreases.  
![](images/27ec4cd6f4741c8c4a74434a6bd74f0f15ac938e6cbb461c273250eb63a5fd00.jpg)  
(a) t = 800

![](images/d4abc569f9a1ccabbe9455483edfec5f9588ff53b03dccb0017f072d5def62ad.jpg)  
(b) t = 200

![](images/1c6ad8144aa7a05d87e83773489125d5dac30ed094e8a3cb11b03323f1f756d8.jpg)  
(c) t = 100

![](images/b51f4fa0115926ce3cbf77df3fd6e740a2464e6d4b210909a7585a642335d31b.jpg)  
(d) t = 25  
Fig. A4: Per-pixel gating $_ { g _ { t , i } }$ vs. Sobel edge map of $\hat { \mathbf { x } } _ { 0 }$ on LSUN Bedroom. Each subplot: predicted image (left), Sobel magnitude (center), gating map (right, blue = 0, red = 1). The gating map progressively aligns with image structure without any edge-detection supervision.

## N.1 Gating Dynamics

Figure A8 presents gating histograms for $\tau \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } 1 0 ^ { - 1 } \}$ . Despite substantial variation in image content and resolution (CIFAR10 at 32 × 32, LSUN Bedroom at $2 5 6 \times 2 5 6 )$ ), the qualitative behavior of $g _ { t , i }$ remains consistent, exhibiting a three-stage pattern across all datasets. The distribution is unimodal near 1 st early timesteps, flattens and spreads broadly at intermediate steps(indicating spatial discrimination between confident and uncertain regions), and collapses near 0 at late timesteps. This invariance across dataset indicates that the spatial adaptivity of SANI is governed by the intrinsic information geometry of the denoising distribution, rather than by dataset-specific characteristics.

Sections 4.4 analyzes the efecs of tolerance τ and temporal gating dynamics independently. The first examines how τ influences the distribution of $g _ { t , i }$ at fixed timesteps, while the second tracks the mean and variance of $g _ { t , i }$ over time for a single τ . Figure A9 integrates both perspectives by plotting the mean gating value g¯ (solid line) with ±1 standard deviation (shaded band) throughout the reverse process for four tolerance values $\tau \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ , across all four datasets with $K = 1 0 0 0$

![](images/3640ecba87a18f19d7b63496785a53fd54903e172da7456cbaa3e26628bca145.jpg)  
(a) t = 800

![](images/c42a6b4c08ec1282d8eced20556aeb92be9207aefe4586d0ab7e905a15223991.jpg)  
(b) t = 200

![](images/197a8d2f71ef5142604754663742aef45a85cf8a5606facf1a190e04483215f7.jpg)  
(c) t = 100

![](images/5a0d2c95228d3adb3946aa2d750c7569a61430507f73929c06a35027804d682c.jpg)  
(d) t = 25

Fig. A5: Per-pixel gating $_ { g _ { t , i } }$ vs. Sobel edge map of xˆ<sub>0</sub> on CelebA. Each subplot: predicted image (left), Sobel magnitude (center), gating map (right, blue = 0, red = 1). The gating map progressively aligns with image structure without any edge-detection supervision.  
![](images/49465a97888bed9c1da79e8e8786b0bec8e8b54fda58a489006e2779656440e6.jpg)  
(a) t = 800

![](images/59e30e6c7c3c51babe3ed38d65a0d1ef574989738c4faf387e10070bb6cd8de3.jpg)  
(b) t = 200

![](images/dce849cfb844e1808a2fc5e98f9bcce4062126516e5159bd7058429d4edbb808.jpg)  
(c) t = 100

![](images/68a2a7ff0e02bccb02c304d087ba0bff1e567aaa37b27e329d94e310824f8a2f.jpg)  
(d) t = 25  
Fig. A6: Per-pixel gating $_ { g _ { t , i } }$ vs. Sobel edge map of $\hat { \mathbf { x } } _ { 0 }$ on Cifar10 (LS). Each subplot: predicted image (left), Sobel magnitude (center), gating map (right, blue = 0, red = 1). The gating map progressively aligns with image structure without any edge-detection supervision.

All four τ curves begin near $\bar { g } = 1$ at $t = T$ , indicating global uncertainty, and decay monotonically toward $\bar { g } \approx 0$ as t approaches 1, reflecting increasing model confidence. Larger τ values result in an earlier and steeper decay. For $\tau = 1 0 ^ { - 2 }$ (yellow), g¯ fails below 0.75 by $t \approx 6 0 0$ on CIFAR10, areas for $\tau = 1 0 ^ { - 5 }$ (dark blue), it remains above 0.95 until $t \approx 1 0 0$ . These results support the theoretical prediction from Eq. 7 that the gating function transitions from stochastic to deterministic when the per-pixel variance $v _ { i }$ drops below τ , with larger τ inducing this transition earlier.

![](images/3769d10ed631c6e5850eb352b2de288df531e60b4ae4d91532e757c0e6adb8f3.jpg)  
(a) t = 800

![](images/ba66fe1225f7d46d95ddec79369405cff7a3c0f1e6b5cc0fc410dfc678de190f.jpg)  
(b) t = 200

![](images/750925510e3852e14deca691fae6aec43feac187d46e0e11362dd6adf62cec82.jpg)  
(c) t = 100

![](images/8bcd88df3d95716d76696ce0eee3fdd41d84740b2517d6d2ad6380a8285e8625.jpg)  
(d) t = 25  
Fig. A7: Per-pixel gating $_ { g _ { t , i } }$ vs. Sobel edge map of xˆ<sub>0</sub> on Cifar10 (CS). Each subplot: predicted image (left), Sobel magnitude (center), gating map (right, blue = 0, red = 1). The gating map progressively aligns with image structure without any edge-detection supervision.

The width of the ±std band quantifies the spatial heterogeneity of the gating map at each timestep. At the beggining and end of the process (t near $T$ or near 1), the bands are narrow indicating that all pixels are either uniformly uncertain or uniformly confident. The bands reach their maximum width at intermediate timesteps, corresponding to the phase where some image regions have been resolved while others remain ambiguous. This peak in heterogeneity occurs earlier for larger τ values, consistent with the observation that a more permissive tolerance initiates pixel discrimination sooner.

The ordering and monotonicity of the τ curves are maintained across all datasets.

## N.2 Residual-Distribution Assumption

Protocol. At each timestep $t \in \{ 8 0 0 , 2 0 0 , 1 0 0 , 2 5 \}$ we draw 500 test images $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } _ { 0 } )$ , and sample $\mathbf { x } _ { t } \sim q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } )$ . The Tweedie estimate $\hat { \mathbf { x } } _ { 0 }$ and per-pixel variance $v _ { i }$ are computed, and standardized residuals $Z _ { i } = ( x _ { 0 , i } - { \hat { x } } _ { 0 , i } ) / { \sqrt { v _ { i } } }$ are formed over all pixels and samples. Under Assumption $2 \ Z _ { i } \sim \mathcal { N } ( 0 , 1 )$ . We compare the empirical distribution of $Z _ { i }$ to two unit-variance references: Gaussian and Laplace $( b = 1 / \sqrt { 2 } )$ , each moment-matched to zero mean and unit variance.

Figure A10 compares the empirical density of $Z _ { i }$ to the parametric references across multiple datasets. No single parametric family provides a consistent fit throughout the entire trajectory. At intermediate noise levels, the Laplace reference aligns most closely with the central mode. At high noise, the empirical peak surpasses the Laplace, indicating super-Laplacian concentration, while at low noise, the distribution bulk is closest to the Gaussian. Consequently, the Kolmogorov-Smirnov distance between the standardized residuals and the standard normal decreases monotonically as t decreases.

![](images/d0085169d84aea143c8eed989edd604f4c8c00dd2dd003d7ccb47bcd8bab56c8.jpg)

![](images/56681b7e7d271fa860368a2c1b052666b201bc64dfe2331020eb54673e2fd6e0.jpg)

![](images/e6fa95552fe7c016de6fd35512366b10920053752907c221a0a14e90a535bf05.jpg)

![](images/e8347f89924d83bbaeb8ec5e6b295e113f34e48481ff86d9a265c5bd8f216224.jpg)

![](images/33bb5895bfef92067ceb8d587c97543202a6fc1325e707deac3fd54001b305d1.jpg)

![](images/3a31b04512983414a00feb8484506e27a26b4ee7e664f664db52a6c2c97304e3.jpg)

![](images/d121f1740f9e0f70f1d906b53985cfe62933adf094eb892d83931daeb37ac822.jpg)

![](images/47848451c8783bf8e8797fd4d2be75eacdbf9dfb94d03f5ee5042a843fa8ac05.jpg)

![](images/b243a02e7f72bf8cbd58ca97f70ad4eeb4e4adfed3ff7ea5c476f807ad881a85.jpg)

![](images/798e25a4daaa15241c970f9286c262c610f65a45f81955b20b9699f0b25f2a98.jpg)

![](images/26e87882cde869bcbf534dcb35f55ec7ea449f7a07f249294e693e7461393b99.jpg)

![](images/1be35c062b012d47781847e95d2fde2a98ac8da6099451bebab9cace1f6d410c.jpg)

![](images/d9bd91127c1425f2182d191d1e0fe790e2914f8aa48ae10ab025465f67fd5327.jpg)  
(a) $\tau = 1 0 ^ { - 4 }$

![](images/853d2c74267c1552c85b9ba7aa00111871993e1465ff7700f9fb7f930beefffd.jpg)  
(b) $\tau = 1 0 ^ { - 3 }$

![](images/1178debffc5fc36a57bc6db5531c872fed1aec0ae6a21a2ba77c2b91ed34c793.jpg)  
(c) $\tau = 1 0 ^ { - 2 }$

![](images/716f9f44e00cf112cdb129363fa67db2b54939bee98112d461ba8b9110af849f.jpg)  
(d) $\tau = 1 0 ^ { - 1 }$

Fig. A8: Spatial distribution of gating values $_ { g _ { t , i } }$ along the full reverse trajectory across datasets (rows), for four tolerances (columns). Each subplot overlays five timesteps t ∈ {1000, 500, 250, 125, 20}, with density on a logarithmic scale. Despite varying resolution and content, the temporal evolution of $_ { g _ { t , i } }$ is qualitatively consistent.  
![](images/4fff6bcbf61010c00fda5ec2a088752f64b08265b966df9fb81e3331c158aa7d.jpg)  
(a) CIFAR10 (CS)

![](images/2ae2afeae6f928392ff691b8ad26c4a82b181c9669c2989b763b4d8524edc2e7.jpg)  
(b) CIFAR10 (LS)

![](images/da57b16fe659d5e0e0246f13fc3dddc4c9397113b027e920433e14f97a451b79.jpg)  
(c) LSUN Bedroom

![](images/7ce31a64dbf8e005acef9c8290fd476486acbb76a94b0e48caeaecab972b96e6.jpg)  
(d) CelebA-64

Fig. A9: Mean gating value $\bar { g } \pm \mathrm { s t d } ( g )$ over the reverse process $( K = 1 0 0 0 )$ for four tolerance values. All datasets exhibit the same qualitative pattern: monotonic decay from 1 to 0, with larger τ producing earlier transition. The standard deviation bands peak at intermediate timesteps where spatial heterogeneity is maximal. LSUN Bedroom shows the widest bands, reflecting the greater spatial complexity at $2 5 6 \times 2 5 6$ resolution.

## O Likelihood Evaluation

This appendix provides a detailed description of the likelihood computation referenced in Section 4.5. We report variational upper bounds on − log $p _ { \theta } ( \mathbf { x } _ { 0 } )$ in bits per dimension (BPD) and evaluate along the same timestep sub-sequence

![](images/5453ba4cb4451982f370084e271fc92b90be59057346ff16e5ac11dfd88353e5.jpg)  
(a) CIFAR10 (CS)

![](images/ecfd3892e98f1fc7c3a9ff6581c6cafc5cea3bb33f8dfa928d82a365ca0a1b72.jpg)

![](images/96a0a072350e91d06b8d6d003c1f6d75595668ed3326bdd4b26fadcee487d38e.jpg)  
(c) CelebA-64

(b) CIFAR10 (LS)  
![](images/ca5290f7ecbc131262a432aadb5f3f64ea409d6a4f560cd17d03decfa86146d1.jpg)  
(d) LSUN Bedroom  
Fig. A10: Empirical standardized residuals $Z _ { i } = R _ { i } / \sqrt { v _ { i } }$ against Gaussian and Laplace residual distributions (all standardised to unit variance) on various datasets. Each subplot visualize density histograms at four reverse-process timesteps.

used during sampling. This ensures that both likelihood and sample quality assessed under consistent schedules.

Following the discrete-time decomposition of [8, 17]:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } = \underbrace { \mathrm { K L } \big ( q ( \mathbf { x } _ { T } \mid \mathbf { x } _ { 0 } ) \| { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) \big ) } _ { \mathrm { p r i o r ~ t e r m } } + \sum _ { ( s , t ) } \mathrm { K L } \big ( q ( \mathbf { x } _ { s } \mid \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) \| p _ { \theta } ^ { \mathrm { S A N I } } ( \mathbf { x } _ { s } \mid \mathbf { x } _ { t } ) \big ) } } \\ { ~ - ~ { \mathbb { E } } _ { q } \big [ \log p _ { \theta } ^ { \mathrm { S A N I } } ( \mathbf { x } _ { 0 } \mid \mathbf { x } _ { 1 } ) \big ] . } \end{array}\tag{A17}
$$

Here, the summation is taken over adjacent pairs $( s , t )$ in the sampling sub sequence. The true reverse posterior $q ( \mathbf { x } _ { s } \mid \mathbf { x } _ { t } , \mathbf { x } _ { 0 } )$ depends solely on the forward process and remains unchanged from the standard DDPM [8]. It is a Gaussian distribution with mean $\tilde { \mu } _ { t , i } ( \mathbf { x } _ { t } , x _ { 0 , i } )$ and variance $\tilde { \beta } _ { t }$

The model transition $p _ { \theta } ^ { \operatorname { S A N I } } ( \mathbf { x } _ { s } \mid \mathbf { x } _ { t } )$ utilizes the SANI mean and per-pixel variance as defined in Proposition 4:

$$
\mu _ { p , i } ^ { \mathrm { S A N I } } = \sqrt { \bar { \alpha } _ { t - 1 } } \hat { x } _ { 0 , i } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } - \Sigma _ { t , i } ^ { 2 } } \epsilon _ { \theta , i } , \qquad ( \sigma _ { p , i } ^ { \mathrm { S A N I } } ) ^ { 2 } = \Sigma _ { t , i } ^ { 2 } = c _ { t } ^ { 2 } v _ { i } + g _ { t , i } \tilde { \beta } _ { t } .
$$

Each KL term in Eq. A17 thus decomposes coordinate-wise between two diagonal Gaussians. The terminal term at t = 1 corresponds to the log-likelihood of $\mathbf { x } _ { \mathrm { 0 } }$ under a discretized Gaussian with the standard fixed variance $\tilde { \beta } _ { 1 }$ , consistent with [8].

Clamping for Near-Deterministic Pixels For pixels routed to deterministic dynamics by the gate $( g _ { t , i } \approx 0 )$ and exhibiting low per-pixel posterior variance $( v _ { i } \approx 0 )$ ， the model variance $\Sigma _ { t , i } ^ { 2 }$ collapses, whereas the true posterior maintains strictly positive variance $\tilde { \beta } _ { t }$ . This results in an unbounded KL divergence, a degeneracy also observed in DDIM at $\eta = 0 .$ , where the ELBO is similarly undefined.

To obtain a finite bound, we clamp the model variance per pixel and per step:

$$
( \sigma _ { p , i } ^ { \mathrm { S A N I } } ) ^ { 2 } \  \ \operatorname* { m a x } \big ( \Sigma _ { t , i } ^ { 2 } , \ \tilde { \beta } _ { t } \big ) .\tag{A18}
$$

This approach leaves stochastic pixels unafected $( \Sigma _ { t , i } ^ { 2 } \geq \tilde { \beta } _ { t }$ whenever $g _ { t , i }$ or $\tilde { \beta } _ { t }$ alone reaches $\tilde { \beta } _ { t }$ , since $c _ { t } ^ { 2 } v _ { i } \geq 0$ is always non-negative) and reduces neardeterministic pixels to the standard DDPM small-variance reverse process. In this case, the contribution to Eq. A17 matches exactly with the DDPM ELBO term at that pixel. The clamping afects only the likelihood computation, while sampling continuous to use the unclamped $\dot { \Sigma } _ { t , i } ^ { 2 }$

The bound is averaged over the test set and converted to bits per dimension by dividing by D log 2.

Results. Table A4 presents NLL results on CIFAR-10 (LS), CIFAR-10 (CS), and CelebA-64 for both Gaussian and laplace residual assumption. Together with the FID results (Table 1), the NLL evaluation demonstrates that spatial gating does not compromise likelihood in favor of sample quality. Across all steps counts and dataset, SANI remains competitive with, or close to, the variance-learning baselines, while also providing an interpretable per-pixel uncertainty diagnostic as shown in Figure 4.

Table A4: Negative log-likelihood (↓, bits per dimension) across datasets and sampling steps. Bold: best overall. Underline: second-best. SANI is competitive with the variancelearning baselines at every step count.
<table><tr><td></td><td colspan="5">CIFAR-10 (LS)</td><td colspan="5">CIFAR-10 (CS)</td><td colspan="5">CelebA 64 × 64</td></tr><tr><td>Method</td><td>10</td><td>25</td><td>50</td><td>100</td><td>200 |</td><td>10</td><td>25</td><td>50</td><td>100</td><td>200 |</td><td>10</td><td>25</td><td>50</td><td>100</td><td>200</td></tr><tr><td>DDPM, β</td><td>74.95</td><td>24.98</td><td>12.01</td><td>7.08</td><td>5.03</td><td>75.96</td><td>24.94</td><td>11.96</td><td>7.04</td><td>4.95</td><td>33.42</td><td>13.09</td><td>7.14</td><td>4.60</td><td>3.45</td></tr><tr><td>DDPM, β</td><td>6.99</td><td>6.11</td><td>5.44</td><td>4.86</td><td>4.39</td><td>6.51</td><td>5.55</td><td>4.92</td><td>4.41</td><td>4.03</td><td>6.67</td><td>5.72</td><td>4.98</td><td>4.31</td><td>3.74</td></tr><tr><td>A-DDPM</td><td>5.47</td><td>4.79</td><td>4.38</td><td>4.07</td><td>3.84</td><td>5.08</td><td>4.45</td><td>4.09</td><td>3.83</td><td>3.64</td><td>4.54</td><td>3.89</td><td>3.48</td><td>3.16</td><td>2.92</td></tr><tr><td>NPR-DDPM</td><td>5.40</td><td>4.64</td><td>4.25</td><td>3.98</td><td>3.79</td><td>5.03</td><td>4.33</td><td>3.99</td><td>3.76</td><td>3.59</td><td>4.46</td><td>3.78</td><td>3.40</td><td>3.11</td><td>2.89</td></tr><tr><td>SN-DDPM</td><td>30.79</td><td>11.83</td><td>7.13</td><td>5.24</td><td>4.39</td><td>90.85</td><td>19.81</td><td>9.72</td><td>6.72</td><td>5.58</td><td>18.09</td><td>8.05</td><td>5.29</td><td>4.05</td><td>3.40</td></tr><tr><td>OCM-DDPM</td><td>5.32</td><td>4.63</td><td>4.25</td><td>3.97</td><td>3.78</td><td>4.99</td><td>4.34</td><td>3.99</td><td>3.76</td><td>3.59</td><td>4.69</td><td>3.86</td><td>3.43</td><td>3.13</td><td>2.90</td></tr><tr><td> $\operatorname { S A N I } _ { - } ^ { G }$ </td><td>5.31</td><td>4.67</td><td>4.31</td><td>4.04</td><td>3.85</td><td>4.97</td><td>4.39</td><td>4.07</td><td>3.83</td><td>3.66</td><td>4.67</td><td>3.88</td><td>3.45</td><td>3.15</td><td>2.95</td></tr><tr><td> $\mathrm { S A N I } ^ { L }$ </td><td>5.34</td><td>4.68</td><td>4.34</td><td>4.06</td><td>3.87</td><td>5.04</td><td>4.40</td><td>4.08</td><td>3.82</td><td>3.68</td><td>4.71</td><td>3.87</td><td>3.48</td><td>3.15</td><td>2.97</td></tr></table>

## P Generated Samples

Figures A11-A14 display generated samples produced by SANI with varying numbers of sampling steps K.

![](images/6a273763da512d5ff526275a81a70170a2a5ecb625a9f58ac07f3b25b19f88b0.jpg)  
(a) K = 10

![](images/da2c06436202d91992b6396873dccd5dd5d55c77db0d1cf28a8f77ca89d3e99e.jpg)  
(b) K = 25

![](images/0eb949a91adf1b9209c5840df3f549e3b16a88faf03c2962d7b158a54d5150d2.jpg)  
(c) K = 50

![](images/bb847acab3135a4541df9b79ec484a714cc08074a7ba464bbd5168aded1fe5ab.jpg)  
(d) $K = 1 0 0$

![](images/044b1b169445e325ee382237865b45b23269be94edb62d2209155257d657c0ff.jpg)  
(e) $K = 2 0 0$

![](images/a36ec85da06e683ed7ec9904c1fb4e566d0ce5e6e6b81c9451c4f15f38adc5cd.jpg)  
(f ) $K = 1 0 0 0$  
Fig. A11: CelebA (64 × 64). Generated samples using SANI using varying number of timesteps.

![](images/678281e9c387dfbe5f3db92d48c6fee4222b6996477ed3517f64fc668f44537c.jpg)  
(a) $K = 2 5$

![](images/8c009472ff5e74558153e3f6ec9bcad0696c0e62e6fb7688db77de3263f934fe.jpg)  
(b) $K = 5 0$

![](images/bd00251f890a3efd4c76dba508982214fb22ce501193d147d22d2c4b3c5d4c66.jpg)  
(c) $K = 1 0 0$

![](images/ffebf5c53aba219a443b111edfbae79ac86898fd179d1ef5a667f0118a526d43.jpg)  
(d) $K = 2 0 0$

![](images/0418009a25530128688fde43f766fbf5850c0eb4317f356cb192b3360f641fab.jpg)  
(e) $K = 1 0 0 0$  
Fig. A12: LSUN Bedroom. Generated samples using SANI using varying number of timesteps

![](images/77a20ad4e83b80caa3e3a787a73247e73ab66698702a01c04b3037a1f2560d8f.jpg)  
(a) K = 10

![](images/324cfe650eceabb2bdf28cc6b97c9b61f8ca2678414531b31e7c297e96b01964.jpg)  
(b) K = 25

![](images/f34bbedf17203a5c0efda0b7a59623fcdb13675d28e5fb7089e41d6cc7e13648.jpg)  
(c) K = 50

![](images/5fe00c0b36cbeef6733cb14fa4ae39613e05188c876116ccbcd7930c81609e3b.jpg)  
(d) K = 100

![](images/e29c256112e43a936743534697f5f9259f3ad189d13fc96c17ac829b4412476b.jpg)  
(e) K = 200

![](images/6e32176133e0f0f1402b1ea8dc6219015727b91d3ef3e539cca7f80d0649e276.jpg)  
(f ) $K = 1 0 0 0$  
Fig. A13: Cifar10 (LS). Generated samples using SANI using varying number of timesteps.

![](images/5d5ce0ba0cdc3469cbe40676bd73d17d30a59cc0df9de95f4e8776b613c5a07e.jpg)  
(a) K = 10

![](images/02fe713bc4c38539fed8d78e43cdf962fb7177474a2e6e44bedfb4f322e67c94.jpg)  
(b) K = 25

![](images/34076b51df5d3aa0c38ff6614eff87840311fbb7925342bcaa784199f731c3b8.jpg)  
(c) K = 50

![](images/81a3ce6cb4817d11d568de4c0393b4a6105d459afe6ea587bac8b4428193cf3b.jpg)  
(d) K = 100

![](images/e86ccd914aeaf5f7d89d0f0d24b975aea322b1f35dbe9dd0d66daeddf3b87648.jpg)  
(e) K = 200

![](images/6060361a95879428544904215bc000b8216cdd5da5a57f4c8a1b74e3b6e6c31b.jpg)  
(f ) K = 1000  
Fig. A14: Cifar10 (CS). Generated samples using SANI using varying number of timesteps.