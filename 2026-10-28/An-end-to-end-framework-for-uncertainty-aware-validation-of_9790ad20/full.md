# An end-to-end framework for uncertainty-aware validation of generative models

Giada Badaracco,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> Sean Benevedes,<sup>3,</sup> <sup>4,</sup> <sup>2</sup> Christina Reissel,<sup>2,</sup> <sup>5</sup> Philip Harris,<sup>2,</sup> <sup>5</sup> and Gaia Grosso<sup>2,</sup> <sup>5</sup>

<sup>1</sup>ETH Z¨urich, Z¨urich, CH

<sup>2</sup>NSF AI Institute for Artificial Intelligence and Fundamental Interactions, Cambridge, MA, USA

<sup>3</sup>School of Physics, Georgia Institute of Technology, Atlanta, GA, USA

<sup>4</sup>MIT Center for Theoretical Physics – a Leinweber Institute, Cambridge, MA, USA

<sup>5</sup>MIT Laboratory for Nuclear Science, Cambridge, MA, USA

Density estimation with generative models is a common task in the physical sciences, for example in background estimation in particle physics. Existing machine learning techniques, however, lack methods to quantify uncertainties that can be propagated into a statistical test. That limits the validation of generative models to standard goodness-of-fit tests, in which the generative model is treated as exact (point-null hypothesis). But a model fitted to a finite training set is never exact, so if its uncertainties are neglected, any suficiently powerful goodness-of-fit test will eventually reject it once the test set is large enough to resolve this estimation error. We propose SIGMA, an end-to-end framework combining density estimation with principled uncertainty quantification. We use Gaussian mixture models, for which uncertainties are tractable, and further construct a new goodness-of-fit test in which these uncertainties are profiled as nuisance parameters to account for the statistical error of the fit, promoting the conventional point-null test to a composite one. A rejection then signals genuine mismodeling rather than a test set large enough to resolve the error of the fit, provided the propagated uncertainties faithfully cover that error. We demonstrate the framework on a controlled two-dimensional toy model and on a collider physics benchmark modeling the QCD background process. With uncertainties included, the same model can be trusted on test sets up to an order of magnitude larger than the standard test allows. Our work thus provides a statistically grounded basis for a robust use of generative models in the physical sciences.

## CONTENTS

I. Introduction 1   
II. Powerful tests reject every generative model 3   
III. Related work 3   
IV. Method 5   
A. Generative model 5   
B. Estimation uncertainty 6   
C. Uncertainty-aware goodness-of-fit test 7   
D. Calibration 8   
V. Numerical experiments 9   
A. Toy model 10   
B. Embedded particle jets 11   
VI. Conclusion 14   
Acknowledgments 15   
A. Additional discussion on the limits of generative   
models 15   
B. Normalizing flows and the limits of ensemble   
uncertainties 16   
1. Density estimation with normalizing flows 16   
2. Uncertainty quantification via $w _ { i } f _ { i }$   
ensembling 17   
3. Uncertainty-aware goodness-of-fit test 17   
4. Results 18   
C. Implementation details 19   
D. Null distribution of the test statistic 19   
E. Efects of kernel width 23   
References 26

## I. INTRODUCTION

Many scientific analyses compare observed data with a reference model in search of significant deviations, which may signal new phenomena. The reliability of the reference model is therefore critical: any mismodeling can be mistaken for a genuine signal and lead to a false discovery. Such reference models are often built from highfidelity simulations that require substantial computing resources. In particle physics, for example, the background from known processes is modeled by simulating the full chain of hard process, parton showering, and detector response. The demands on the reference model are particularly high in model-agnostic searches, for example anomaly detection [1], where any deviation from it across the full feature space may be reported as a signal.

Generative models have emerged as a way to signifi cantly reduce the cost of such expensive simulations, acting as fast surrogates (or even replacements) [2–7] or as learned likelihoods in simulation-based inference [8, 9]. Like any fitted model, however, generative models carry an inherent error due to the finite size of the training set. Generated events are moreover not independent draws from the true density: they all come from a single model (or ensemble of models) fitted to a finite sample, whose fluctuations are imprinted in the model. No amount of generated data can therefore add information beyond the training sample [10]. In practice, any trained model will eventually fail a statistical test if it is checked against enough data. This happens even when the model is, in principle, capable of describing the data perfectly (Sec. II). Whether this failure matters depends on the use case in the analysis workflow [11]. A few representative examples are listed in Fig. 1. Errors can be tol erable when the result does not impact the downstream task; this includes examples when the model’s output is further validated by exact downstream checks, or when only qualitative fidelity is required, as is the case in exploratory studies or pretraining. But when the fitted density serves as the reference model of a statistical test, as a background estimate, a null hypothesis, or a calibration input, its error is indistinguishable from a signal. To be used reliably in this role, a generative model must therefore come with uncertainties, and these must enter the statistical test itself.

![](images/bc684929d5fc3c5488b8347d5580599aa4999ee95b6af4c10c731bd5102924b1.jpg)  
FIG. 1. When uncertainty quantification matters. Representative uses of generative models in the physical sciences, classified by whether the error of the fit should be propagated into a quantitative claim. For the uses on the left (blue), a wrong model costs eficiency or statistical power but not validity of the claim, and qualitative validation sufices. For the uses on the right (red), the fitted density is the reference model of a statistical inference: its error is indistinguishable from a signal, so it must be propagated into the test (Sec. IV).

For models with few parameters, estimating uncertainties follows standard procedures. However, for modern density estimators, particularly machine learning driven generative models, with high-dimensional inputs and large numbers of non-identifiable parameters, the problem becomes technically challenging, and attributing robust uncertainties to them remains an active area of research.

To address this challenge, we propose SIGMA (Score-Induced Gaussian Mixture Approximation), a generative model whose uncertainties are tractable and can be propagated into a statistical test. To validate it, we further introduce an uncertainty-aware goodness-of-fit test that incorporates the derived uncertainties.

SIGMA models the density with a Gaussian mixture model and describes its uncertainties in the tangent space spanned by its score functions around the best fit of the density. Unlike for neural networks, this simple Ansatz makes the asymptotic maximum-likelihood covariance of the parameters tractable. The test then propagates these uncertainties by promoting the conventional point-null Neyman–Pearson likelihood-ratio test [12–14] (also referred to as simple hypothesis) to a composite one [15], in which the nuisance parameters encoding the uncertainties are profiled, as is common practice in LHC searches [16]. Unlike in searches, the nuisance parameters are not systematic uncertainties of the experiment, but the statistical uncertainties of the fitted generative model itself. Because they enter the test in the same standard form, conventional systematic uncertainties can be profiled alongside them, so that a single composite test accounts for both the error of the fit and the systematics of the analysis. The test outcome is therefore determined by the compatibility of the data with the model within its uncertainties, rather than driven by the fit error.

Calibrated with pseudo-experiments drawn from the composite-null, the test retains its nominal false-positive rate even when the number of test events exceeds the training sample size, provided that the uncertainties faithfully cover the error of the fit.

The paper is organized as follows. In Sec. II, we motivate the problem with a controlled one-dimensional example. Section III reviews related work. In Sec. IV, we describe the components of our algorithm, from the density model to the uncertainty quantification and the design of the goodness-of-fit test. Section V applies the method to a controlled two-dimensional toy study, where the true density is known, and to the QCD background in a learned embedding of particle jets, a setting relevant for anomaly detection searches. We conclude in Sec. VI and outline future research directions.

In App. A we further discuss the intrinsic limits of generative models and standard goodness-of-fit tests. Appendix B explores a related approach to quantifying and propagating model uncertainties with ensembles of neural-network density estimators, and finds that the resulting uncertainties are insuficient for the composite-null goodness-of-fit test to close on large observed datasets. Appendix C contains technical details of the experiments and studies the efects of varying a hyperparameter, App. D derives the asymptotic distribution of the composite-null test statistic under the null hypothesis, and App. E studies the choice of σ, another hyperparameter in the goodness-of-fit test.

## II. POWERFUL TESTS REJECT EVERY GENERATIVE MODEL

A generative model with fitted parameters $\hat { \theta }$ only ever provides an approximation $p _ { \hat { \theta } } ( x )$ of the true data distribution $p _ { 0 } ( x )$ . Following Refs. [17, 18], the discrepancy has two classes of sources, both epistemic, i.e. uncertainties about the model rather than about the intrinsic randomness of the data:

i) The model bias, a systematic uncertainty: the model may be unable to represent the true distribution exactly, however well it is trained. This bias depends on the choice of architecture and does not shrink with the size of the training sample. It is distinct from the systematic uncertainties usually considered in physics analyses, such as imperfect knowledge of the detector or of the theory, which concern how well the training data, typically a simulation, describe reality.

ii) The statistical uncertainty of the fit: the finite training sample limits how precisely the parameters can be determined (data variability), and training may converge to diferent solutions depending on its initialization (optimization variability). Both vanish in the limit of infinite data and ideal optimization.

Our work addresses the statistical uncertainty ii). Even in the best-case scenario in which the model family contains the ground truth, the finite size of the training dataset places a fundamental limit on the accuracy of the fitted density. This limit has a direct and often underappreciated consequence for the validation of generative models via goodness-of-fit tests.

To understand the efect, we consider the following one-dimensional experiment: samples are drawn from a standard normal distribution, $p _ { 0 } = \mathcal { N } ( 0 , 1 )$ , and the generative model is chosen to be perfectly well specified, $p _ { \theta } = \mathcal { N } ( \mu , \sigma )$ with trainable parameters $\theta = ( \mu , \sigma )$

The parameters are estimated by maximum likelihood optimization on $N _ { \mathrm { t r a i n } }$ training points, and the fitted model is then compared with independent test samples of increasing size $N _ { \mathrm { t e s t } }$ by means of a goodness-of-fit test.

The fitted model, indistinguishable from the truth at small test sample sizes, is eventually rejected with high significance once $N _ { \mathrm { t e s t } }$ grows past $N _ { \mathrm { t r a i n } }$ . This degradation is not specific to the test choice: any consistent goodness-of-fit test eventually detects the finite-statistics error between $p _ { \hat { \theta } } ( x )$ and the true density $p _ { 0 } ( x )$ , so the pvalues decrease with $N _ { \mathrm { t e s t } }$ regardless of the test used. The gray markers in Figure 2 show this trend for the Gaussian example and the point-null likelihood-ratio test that will be defined in Sec. IV C (see Eq. (17)) calibrated with pseudo-experiments drawn from the fitted model.

This behavior is not a pathology of the example but a generic and quantifiable consequence of information limits [10] (see App. A for a derivation). How fast the pvalue curve degrades depends both on the sensitivity of the goodness-of-fit test employed and on the quality of the model design, i.e. how well the model design matches the properties of the true distribution. Crucially, how ever, the degradation itself cannot be avoided: no amount of architectural or training improvement makes a density estimated from $N _ { \mathrm { t r a i n } }$ samples exact.

How, then, can generative models be used reliably for science? It is worth being precise about what this “intrinsic” failure means. Standard goodness-of-fit methods test a point-null hypothesis: the fitted model is treated as an exact description of the data. Since the fitted model difers from the truth with probability one, rejection at large $N _ { \mathrm { t e s t } }$ is guaranteed by construction. However, this is not the question of scientific interest. The relevant question is whether the data are compatible with the generative model within its statistical uncertainty, the spread of fits that would result from diferent training samples and diferent runs of the optimization. This is a composite-null hypothesis. A test that ignores this uncertainty conflates genuine mismodeling with the unavoidable statistical fluctuations of the model fit, and its outcome is driven by the relation between $N _ { \mathrm { t e s t } }$ and $N _ { \mathrm { t r a i n } }$ rather than by the quality of the model. On the other hand, the bias introduced by an incorrect model is one that uncertainties cannot solve, and needs to be addressed at the level of model design, e.g. by replacing parametric background functions that fail as the dataset grows with more flexible Gaussian processes [19]. The statistical uncertainty of the fit, however, can be propagated into the goodness-of-fit test itself, promoting the point-null to a composite one. With the uncertainties included, the median p-value remains close to its nominal value of 0.5 over the full range of test sizes we probe (blue markers in Fig. 2).

## III. RELATED WORK

a. Generative models and uncertainty quantification. Uncertainties on generative models have been estimated with Bayesian approaches [20, 21] and Monte Carlo dropout [22, 23]. While both return a spread of predictions, they do not come in a form that can be written down as a parametrized density function. This makes them incompatible with composite likelihood-ratio tests, which require a parametrized family of null densities.

![](images/0ee32469ee9ca5e7eecfe27040e50cc8c982d70fcc39b4912917bcd3d76a7171.jpg)  
FIG. 2. Impact of fit uncertainties on goodness-of-fit tests of generative models. A Gaussian $\mathcal { N } ( \mu , \sigma )$ is fitted by maximum likelihood to $N _ { \mathrm { t r a i n } }$ samples drawn from $\mathcal { N } ( 0 , 1 )$ , with $N _ { \mathrm { t r a i n } } = 1 0 ^ { 3 } , 1 0 ^ { 4 }$ and $\mathrm { 1 0 ^ { 5 } }$ (left to right). The fitted model is then tested on $N _ { \mathrm { t e s t } }$ fresh samples from the ground truth, using either the point-null $\mathrm { ( g r a y ) }$ , which takes the fit as exact, or the Gaussian composite-null (blue), which includes the fit uncertainties. For each $N _ { \mathrm { t r a i n } } ,$ , the fit is repeated with 10 independent training samples and each fit is tested on 100 test samples. Markers show the median p-value across trainings, and error bars show the 68% spread of the per-training medians. If the tested null hypothesis holds, p-values are uniformly distributed with median 0.5 (horizontal dotted line). Although the model family contains the ground truth, the fitted model never coincides with it exactly. The point-null, which ignores this estimation error, is therefore eventually rejected once $N _ { \mathrm { t e s t } } \gtrsim N _ { \mathrm { t r a i n } }$ (vertical dashed line). The composite-null accounts for it and remains satisfied up to the largest test size probed, $N _ { \mathrm { t e s t } } = 1 0 ^ { 6 }$ , i.e. between 10 and $1 0 ^ { 3 }$ times the training-set size.

The frequentist ensembling in Ref. [24] provides one such parametrization, describing the uncertainty through the weights of a fixed ensemble of trained models. Ensembling can be performed for any architecture, but the resultant uncertainties span only the coeficients of the trained models. As a result, the uncertainties cover the true density only under the strong assumption that it lies within the span of the trained ensemble members. Ensembling is also used in experimental searches, where the spread of independently trained models provides the uncertainty of a generative background model in the final fit [25]. Our SIGMA approach follows a similar logic but spans the full parameter space of the generative model, weakening the well-specification assumption, at the price of requiring a tractable model likelihood.

b. Data amplification and the statistical value of generated samples. A closely related line of work asks whether samples drawn from a generative model carry statistical precision beyond that of the training sample, quantified through an amplification factor [26–29]. Refs. [28, 29] quantify the uncertainty of the learned densities, but do not propagate any uncertainties to the point-null hypothesis. Ref. [10] establishes that the generated data cannot contain more information than the data on which the model was trained, and Ref. [30] derives an information-theoretic bound on the achievable gain, posing a theoretical limit to amplification.

We address a prerequisite for trustworthy amplification: whether the fitted density, together with its estimated uncertainty, is compatible with independent data from the true distribution.

c. Validating generative models. Previous work typically validates the learned density through a lowdimensional summary test statistic: quantile occupancies [26], projected Jensen–Shannon divergences of high-level observables [27], the per-bin coverage of Bayesian uncertainties [28], the Kolmogorov asymptotics of a one-dimensional statistic [29], or classifier-based scores [31, 32], including the full distribution of classifier weights [33]. Since all of these tests treat the fitted density as exact, they eventually reject even a model that is as good as its training statistics allow, once $N _ { \mathrm { t e s t } }$ is large enough to resolve the error of the fit (Sec. II).

Two-sample tests for validating generative models have been compared in Refs. [34, 35]. The latter finds the Neyman–Pearson test to be overall the most robust, outperforming marginal-based comparisons, at a higher computational cost. The Neyman–Pearson test with a learned alternative hypothesis [12–14, 36] compares the hypotheses in the full feature space, assumes little about the position or shape of a deviation, and returns a $p \textmd { - }$ value that is calibrated under the assumption that the reference model is exact. While Refs. [37, 38] applied the Neyman–Pearson test to the validation of generative models in high-dimensional settings, these studies did not account for their uncertainties. Ref. [15] extends the Neyman–Pearson test to include nuisance parameters in the standard LHC form [16]. We adopt this structure, but in our case the nuisance parameters are not handchosen systematic uncertainties, but the estimation uncertainty of the generative model.

d. Gaussian mixture models. Gaussian mixture models are a classical tool of statistics [39]. Their maximum-likelihood fitting via expectation– maximization (EM) [40, 41], the computation of the observed information matrix [42], and the asymptotic theory of mixture likelihoods [43, 44] are all well established. So are the ways mixtures depart from textbook regularity: components can be degenerate or weakly identified, and the model is identifiable only up to relabeling [45]. These issues motivate the regularization choices in Sec. IV. We claim no novelty on this machinery. Our contribution is to use the tractable score and Fisher information of a fitted mixture to extend the point-null goodness-of-fit test to a composite one, so that the uncertainty in the fit enters the test itself.

e. Goodness-of-fit with estimated parameters. Standard goodness-of-fit tests such as Kolmogorov–Smirnov [46] assume a fully specified null. Testing a null hypothesis that was itself fitted to data is a classical problem in statistics [47, 48], and the standard remedy is the parametric bootstrap [49–51]. Particle physics knows the same process as calibration with pseudo-experiments, or toy Monte Carlo [52, 53]. Our bootstrap prescription of Sec. IV D mirrors this classical procedure, combined with a composite-null test in which the fit uncertainty enters as an explicit nuisance parametrization.

## IV. METHOD

The method, summarized in Fig. 3, has three parts: a generative model with an unconstrained parametrization (Sec. IV A), a standardized description of its estimation uncertainty (Sec. IV B), and a goodness-of-fit test in which the model may move within that uncertainty (Sec. IV C). The calibration of the goodness-of-fit test is discussed in Sec. IV D.

## A. Generative model

The model family should be chosen to satisfy two conditions. First, we find that the uncertainty must be propagated on all of the model’s parameters, since all of them fluctuate with the training sample. Appendix B shows that approaches propagating it on only a subset, such as the mixture coeficients of an ensemble of trained normalizing flows, fail. Second, every point of parameter space must define a valid density, so that Gaussian fluctuations of the parameters never leave the model family. Gaussian mixture models (GMMs) [44] with a suitable parametrization satisfy both. For K components in D dimensions,

$$
p _ { \theta } ( x ) = \sum _ { k = 1 } ^ { K } w _ { k } { \mathcal { N } } ( x \mid \mu _ { k } , \Sigma _ { k } ) ,\tag{1}
$$

where the weights $w _ { k }$ are given by softmax logits and the covariances, $\Sigma _ { k }$ , by Cholesky factors with a logparametrized diagonal,

$$
w _ { k } = \frac { e ^ { \ell _ { k } } } { \sum _ { j = 1 } ^ { K } e ^ { \ell _ { j } } } , \quad \Sigma _ { k } = L _ { k } L _ { k } ^ { \top } , \quad [ L _ { k } ] _ { i i } = e ^ { s _ { k , i } } .\tag{2}
$$

The logits $\ell _ { k } ,$ means $\mu _ { k } .$ , log-diagonals $s _ { k , i } ,$ and ofdiagonal entries $[ L _ { k } ] _ { i j } \ ( i > j )$ are all unconstrained real numbers, and fixing one of the $\ell _ { k }$ using the $\textstyle \sum _ { k } w _ { k } = 1$ constraint removes the invariance of the softmax under a common shift of the logits. The parameter vector θ therefore ranges over all of $\mathbb { R } ^ { P }$ , with

$$
\begin{array} { r } { P = K \left( 1 + D + \frac { D \left( D + 1 \right) } { 2 } \right) - 1 . } \end{array}\tag{3}
$$

The parameter count grows as $P \sim K D ^ { 2 } / 2$ , and the number of components needed to resolve a complex density also tends to grow with dimension. In high dimensions, Gaussian mixtures thus become heavily overparametrized [54], and neural density estimators such as normalizing flows have largely replaced them [2, 55]. In low dimensions, however, they remain expressive [56] and ofer what neural estimators lack: a moderate number of parameters, identifiable up to a relabeling of the components [45], with analytic scores from which the uncertainty of the fit can be computed. This makes them well suited to the low-dimensional learned embeddings considered in Sec. V. An alternative, strongly related approach with an ensemble of independently trained flows, inspired by Ref. [24], is presented in App. B.

The GMM is fitted by maximum likelihood to a training sample $\{ x _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } }$ drawn from the unknown true density $p _ { 0 }$

$$
\hat { \theta } = \arg \operatorname* { m a x } _ { \theta } \sum _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } \log p _ { \theta } ( x _ { i } ) ,\tag{4}
$$

using the expectation–maximization (EM) algorithm [40], the standard tool for maximum-likelihood estimation of mixtures [44]. Starting from a k-means initialization, EM alternates between computing the posterior probability that each training point belongs to each component (E step) and re-estimating the weights, means, and covariances in closed form with the points weighted by those probabilities (M step); each iteration increases the likelihood, and the algorithm converges to a stationary point. The GMM likelihood is highly multimodal, so the resulting $\hat { \theta }$ is in general a local maximum [43].

![](images/ff7c17af0dd96ed7f014e37db2b90b5673fce0cbd219630493e3f21ebac93830.jpg)  
FIG. 3. Schematic overview of the method. (A) The generative model $p _ { \hat { \theta } }$ is fitted by maximum likelihood and linearized around the best fit, so that parameter shifts act through the fixed score ψ(x). (B) The Fisher information sets the uncertainty of the $\operatorname { f i t } ,$ and standardized coordinates measure parameter shifts in units of that uncertainty. (C) The test compares a null family $g _ { v }$ (green band), in which the model moves within its uncertainty, with an alternative $h _ { v , \alpha }$ (orange dashed) that also switches on localized Gaussian kernels. (D) The null distribution of the test statistic is obtained from pseudo-experiments following the Gaussian or bootstrap prescription.

For small displacements $\delta \theta$ around the fit,

$$
\log p _ { \hat { \theta } + \delta \theta } ( x ) = \log p _ { \hat { \theta } } ( x ) + \delta \theta ^ { \top } \psi ( x ) + O ( \| \delta \theta \| ^ { 2 } ) ,\tag{5}
$$

where

$$
\psi _ { a } ( x ) \equiv \left. \frac { \partial \log p _ { \theta } ( x ) } { \partial \theta _ { a } } \right| _ { \theta = \hat { \theta } } , \quad \quad a = 1 , \ldots , P ,\tag{6}
$$

is the score function. Once the fit is done, $\psi$ is a fixed function of x (see App. C for its computation). The test of Sec. IV C is built in this tangent space.

## B. Estimation uncertainty

We test the fitted model on an independent test sample $\mathcal { D } = \{ x _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \sim p _ { 0 }$ . The standard approach treats the fit as a point-null hypothesis, $H _ { 0 } ^ { \mathrm { p o i n t } } : \mathcal { D } \sim p _ { \hat { \theta } }$ . But $\hat { \theta }$ is itself a random variable, which scatters around the best parameter value by $O ( N _ { \mathrm { t r a i n } } ^ { - 1 / 2 } )$ . The modified hypothesis, a composite-null hypothesis, $H _ { 0 } ^ { \mathrm { c o m p } }$ , asks whether the test data are consistent with $p _ { \theta }$ for some $\theta$ close to $\hat { \theta }$ within the range of values θ could take due to random variation in the training sample. The test statistic of Sec. IV C encodes these fluctuations through a Gaussian constraint in standardized coordinates, constructed below, and Sec. IV D describes two ways of sampling H<sup>comp</sup><sub>0</sub> to calibrate it. Both hypotheses are tested against an alternative hypothesis $H _ { 1 }$ which extends the null family by adding additional components and thus provides the flexibility needed to detect possible deviations.

All expectations under the fitted model are estimated with a reference set $R _ { 0 } = \{ x _ { s } \} _ { s = 1 } ^ { S } \sim p _ { \hat { \theta } }$ , with $S \gg N _ { \mathrm { t e s t } }$ It is used for the Fisher information, for the normalization of the densities in the test, and to choose the addi tional components of the alternative hypothesis.

If the model is correctly specified, <sup>ˆ</sup>θ is asymptotically Gaussian with covariance $\bar { F } ^ { - 1 } / N _ { \mathrm { t r a i n } }$ [43], where $F$ is the per-event Fisher information. We estimate F on the reference set from the scores of Eq. (6),

$$
\widehat { F } = \frac { 1 } { S } \sum _ { x _ { s } \in R _ { 0 } } \psi ( x _ { s } ) \psi ( x _ { s } ) ^ { \top } ,\tag{7}
$$

and take $\widehat { \mathrm { C o v } } ( \widehat { \theta } ) \ : = \ : \widehat { F } ^ { + } / N _ { \mathrm { t r a i n } } .$ , where $( \cdot ) ^ { + }$ is a pseudoinverse with a small ridge.<sup>1</sup> For simplicity of presentation, we write estimates from $R _ { 0 }$ as averages over the whole dataset, but in practice each estimate uses a subset of $R _ { 0 }$ for computational eficiency (see App. C).

We diagonalize $\widehat { \mathrm { C o v } } ( \hat { \theta } ) \ = \ V \Lambda V ^ { \top }$ , with eigenvalues $\lambda _ { 1 } \ \geq \ \cdot \ \cdot \ \geq \ \lambda _ { P } \ \geq \ 0 .$ , and discard the directions with $\lambda _ { m } > \Lambda _ { \mathrm { m a x } }$ . These are directions the training data barely constrain [44], along which the fit can move far from <sup>ˆ</sup>θ and the expansion of Eq. (5) breaks down. Collecting the $M \leq P$ retained eigenvectors and eigenvalues in $V _ { M }$ and $\Lambda _ { M }$ , we define standardized coordinates $v \in \mathbb { R } ^ { M }$ through

$$
v = W ^ { + } \delta \theta , \qquad W \equiv V _ { M } \Lambda _ { M } ^ { 1 / 2 } \in \mathbb { R } ^ { P \times M } ,\tag{8}
$$

which measure each parameter displacement in units of its own uncertainty. The resulting coordinates are uncorrelated and have unit variance asymptotically, $v \sim$ $\mathcal { N } ( 0 , I _ { M } )$ , so that a displacement of $\lVert \boldsymbol { v } \rVert = 1$ corresponds to a one-sigma shift of the corresponding parameter. The corresponding standardized score features,

$$
\Phi ( x ) \equiv W ^ { \top } \psi ( x ) \in \mathbb { R } ^ { M } ,\tag{9}
$$

satisfy

$$
\mathbb { E } _ { { p _ { \hat { \theta } } } } [ \Phi ] = 0 , \qquad \mathrm { C o v } _ { p _ { \hat { \theta } } } [ \Phi ] = W ^ { \top } { \widehat F } W = \frac { \mathbb { I } _ { M } } { N _ { \mathrm { t r a i n } } } ,\tag{10}
$$

where the first identity is the vanishing mean of the score and the second follows from $\widehat { F } = V _ { M } \Lambda _ { M } ^ { - 1 } V _ { M } ^ { \top } / N _ { \mathrm { t r a i n } }$ on the retained subspace.

## C. Uncertainty-aware goodness-of-fit test

We compare data and model with a profile<sup>2</sup> likelihood ratio between two exponential families built on $p _ { \hat { \theta } }$

a. Null hypothesis. To propagate the fit uncertainty into the test, the null hypothesis must be promoted from a single fixed density $p _ { \hat { \theta } }$ to a family of densities that accounts for the uncertainties; we construct this family by tilting $p _ { \hat { \theta } }$ along the uncertainty-normalized score directions. An exponential tilt is used instead of $p _ { \hat { \theta } + W v }$ because its log-likelihood is linear in $v ,$ which makes the test a convex problem. The null family reads:

$$
g _ { v } ( x ) = \frac { p _ { \hat { \theta } } ( x ) e ^ { v ^ { \top } \Phi ( x ) } } { Z ( v , 0 ) } ,\tag{11}
$$

with the normalization Z defined in Eq. (13) below. By Eq. (5), we have that $g _ { v } ~ = ~ p _ { \hat { \theta } + W v }$ to first order in v, so the tilt parameters are the standardized parameter displacements of Eq. (8), and $H _ { 0 } ^ { \mathrm { c o m p } }$ corresponds to $v \sim$ $\mathcal { N } ( 0 , \mathbb { I } _ { M } )$

b. Alternative hypothesis. The alternative adds additional degrees of freedom via tilts along a fixed dictionary of J localized kernels, following the New Physics Learning Machine (NPLM) approach of Refs. [12, 59],

$$
h _ { v , \alpha } ( x ) = \frac { p _ { \hat { \theta } } ( x ) e ^ { v ^ { \top } \Phi ( x ) + \alpha ^ { \top } \tilde { k } ( x ) } } { Z ( v , \alpha ) } ,\tag{12}
$$

$$
Z ( v , \alpha ) = \int p _ { \hat { \theta } } ( x ) e ^ { v ^ { \top } \Phi ( x ) + \alpha ^ { \top } \tilde { k } ( x ) } d x ,\tag{13}
$$

so that $g _ { v } = h _ { v , 0 }$ . The exponential form keeps $h _ { v , \alpha }$ positive for all $( v , \alpha )$ . The features $\tilde { k } _ { j }$ are Gaussian kernels of a common width $\sigma , \ k _ { j } ( x ) = \ \overset { \ J } { \exp } ( - \| x - c _ { j } \| ^ { 2 } / 2 \sigma ^ { 2 } )$ standardized to zero mean and unit variance on $R _ { 0 } .$ , with centers placed by k-means clustering of $R _ { 0 }$ so that they tile the support of the model (see App. C). The normalization in Eq. (13) is estimated on the reference set,

$$
\widehat { Z } ( v , \alpha ) = \frac { 1 } { S } \sum _ { x _ { s } \in R _ { 0 } } e ^ { v ^ { \top } \Phi ( x _ { s } ) + \alpha ^ { \top } \widetilde { k } ( x _ { s } ) } .\tag{14}
$$

The dictionary should tile the support of the model densely enough that no localized departure falls between kernels; we use $J = 5 0 0$ Beyond this point additional kernels add little: neighboring kernels of a common width overlap strongly, so the dictionary is highly collinear, and the ridge penalty in Eq. (15) suppresses the redundant combinations. The efective number of degrees of freedom of the test is therefore set jointly by the kernel width σ and the ridge rather than by J itself (see App. D), and our results are stable under moderate changes of J.

The width σ sets the length scale on which the augmented model Eq. (12) is sensitive to departures from the null family, and its choice is bounded on both sides. From above, to ensure power, σ should not exceed the scale of the discrepancies the test is meant to detect, which for the smooth densities considered here is set by the length scale of the data. From below, to ensure calibration, σ is bounded by the validity of the calibration procedures to be described in Sec. IV D (this point is discussed further in App. E). To strike a balance between these considerations, we take σ to be set by the characteristic scale of the data (alternatively, tests at several widths can be combined [60]); see App. C for details.

c. Test statistic. On the test sample, the loglikelihood ratio of $h _ { v , \alpha }$ to $p _ { \hat { \theta } }$ , together with the Gaussian constraint on v and a ridge penalty on $\alpha ,$ is

$$
\begin{array} { r l } & { \ell ( v , \alpha ) = \displaystyle \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \big [ v ^ { \top } \Phi ( { \boldsymbol x } _ { i } ) + \alpha ^ { \top } \tilde { { \boldsymbol k } } ( { \boldsymbol x } _ { i } ) \big ] } \\ & { ~ - N _ { \mathrm { t e s t } } \log \widehat { Z } ( v , \alpha ) - \frac { 1 } { 2 } \| v \| ^ { 2 } - \lambda _ { \alpha } \| \alpha \| ^ { 2 } . } \end{array}\tag{15}
$$

Treating the v as nuisance parameters with a Gaussian constraint [16], the test statistic is the profile likelihood ratio

$$
t _ { \mathrm { c o m p } } = 2 \Big [ \operatorname* { m a x } _ { \boldsymbol { v } , \alpha } \ell ( \boldsymbol { v } , \alpha ) - \operatorname* { m a x } _ { \boldsymbol { v } } \ell ( \boldsymbol { v } , 0 ) \Big ] .\tag{16}
$$

The denominator lets the model move within its uncertainty to accommodate the data, and the numerator additionally switches on the localized distortions. The statistic $t _ { \mathrm { c o m p } }$ therefore measures only the incompatibility that the model’s uncertainty cannot explain. As an uncertainty-blind baseline, the point-null statistic fixes $v = 0$

$$
t _ { \mathrm { p o i n t } } = 2 \operatorname* { m a x } _ { \alpha } \ell ( 0 , \alpha ) ,\tag{17}
$$

with no subtraction needed since $\ell ( 0 , 0 ) = 0$ Because $\ell ( v , 0 )$ is the restriction of $\ell ( v , \alpha )$ to $\alpha = 0$ , both statistics are nonnegative. For numerical safety, the maximizations over α are also restricted to a box that bounds the logdensity distortion of each kernel (see App. C for details).

Every maximization above is a convex problem. In Eq. (15), the data term is linear in $( v , \alpha )$ , the penalties are concave quadratics, and

$$
\log \widehat { Z } ( \boldsymbol { v } , \alpha ) = \log \frac { 1 } { S } \sum _ { x _ { s } \in R _ { 0 } } \boldsymbol { e } ^ { \boldsymbol { v } ^ { \top } \Phi ( x _ { s } ) + \alpha ^ { \top } \widetilde { \boldsymbol { k } } ( x _ { s } ) }\tag{18}
$$

is a log-sum-exp of linear functions, and hence convex; the box constraint defines a convex set. The objective ℓ is therefore strictly concave, and each maximization has a unique optimum, which a quasi-Newton method reaches promptly.

d. Additional nuisance parameters. The construction above is the standard profile-likelihood treatment of constrained nuisance parameters [16], with v encoding the statistical uncertainty of the fit. Nothing restricts the nuisance vector to this role: experimental and theoretical systematic uncertainties, parametrized as additional constrained directions in Eq. (15), can be profiled jointly with $v ,$ so that a single composite test accounts for all sources of uncertainty on the reference model. This holds provided the systematic variations admit a density parametrization, as in standard template-morphing approaches. We leave this combination to future work and focus here on the fit uncertainty alone, which is the component that existing validation practice omits.

## D. Calibration

The null distribution of $t _ { \mathrm { c o m p } }$ is obtained from $N _ { B }$ pseudo-experiments that sample $H _ { 0 } ^ { \mathrm { { c o m p } } }$ . For each, the statistic $t _ { b }$ is computed with the full procedure above, including the same penalties and constraints, and the $p \mathrm { - }$ value is the rank of the observed statistic,

$$
p = \frac { \# \{ t _ { b } \geq t _ { \mathrm { o b s } } \} + 1 } { N _ { B } + 1 } .\tag{19}
$$

We consider two ways of generating the pseudoexperiments. The statistic is the same in both; only its calibration difers.

a. Gaussian prescription. If $\hat { \theta }$ is Gaussian with the covariance of Sec. IV B, $H _ { 0 } ^ { \mathrm { { c o m p } } }$ can be sampled directly,

$$
z ^ { ( b ) } \sim \mathcal { N } ( 0 , \mathbb { I } _ { M } ) , \quad \theta _ { b } = \hat { \theta } + W z ^ { ( b ) } , \quad \mathcal { D } _ { b } \sim p _ { \theta _ { b } } .\tag{20}
$$

For each pseudo-experiment, a value of $\theta _ { b }$ is chosen, and pseudo-data are then drawn from the GMM at $\theta _ { b } .$ The test statistic $t _ { b }$ is then computed on this pseudo-data in the same way as it is for the observed data. This procedure sufices to calibrate the observed statistic if the asymptotic approximation is reliable and the Gaussian distribution of $\theta _ { b }$ reproduces the sampling distribution of ${ \hat { \theta } } . ^ { 3 }$

b. Bootstrap prescription. For a GMM with many parameters and a multimodal likelihood, the Gaussian description of $\hat { \theta }$ can fail: expectation maximization (EM) may converge to diferent local maxima for diferent training samples [43], and the scatter of the fit need not be Gaussian, nor small enough for $\mathrm { E q . ~ ( 5 ) }$ to hold.

The parametric bootstrap relaxes this assumption by directly simulating the fitting procedure. Treating $p _ { \hat { \theta } }$ as if it were the true density, we repeatedly draw training samples from it and refit; the spread of the refits $\hat { \theta } _ { b }$ around $\hat { \theta }$ then serves as a proxy for the (unobservable) spread of $\hat { \theta }$ around the truth. The necessary assumption is that the spread from the fitting procedure is similar whether the data come from $p _ { 0 }$ or from the nearby $p _ { \hat { \theta } } \cdot$ , which is weaker than the asymptotic conditions required for the Gaussian prescription. However, this assumption can fail even for a well-specified model, and in particular we find in App. E that the bootstrap calibration breaks down if the kernel width of Sec. IV C is chosen too narrow compared to the characteristic scale of the data.

In practice, for each pseudo-experiment, we draw a training sample of size $N _ { \mathrm { t r a i n } }$ from $p _ { \hat { \theta } }$ and refit the full GMM with EM from a fresh k-means initialization, obtaining $\hat { \theta } _ { b }$ . We rebuild the entire construction at $\hat { \theta } _ { b }$ exactly as for the nominal fit (i.e. the reference set, the Fisher information, the standardization, and the kernel dictionary), draw a test sample of size $N _ { \mathrm { t e s t } }$ from $p _ { \hat { \theta } }$ , and compute $t _ { b }$

These two prescriptions for generating the calibration distribution are, in a sense, mirror images of each other. The Gaussian prescription holds the fit fixed at $\hat { \theta }$ and moves the data-generating model to $\theta _ { b }$ , while the bootstrap prescription moves the fit to $\hat { \theta } _ { b }$ and holds the datagenerating model fixed at ${ \hat { \theta } } .$ Both, then, should reproduce the displacement between fit and truth, but the bootstrap does not assume that this displacement is small or Gaussian, and it captures the optimization variability. We will find that it remains calibrated to larger $N _ { \mathrm { t e s t } }$ (see Sec. V) and is our nominal prescription. The Gaussian prescription requires no refits and provides an explicit parametrization of the model uncertainty that can be easily propagated to the density and to downstream applications, and it remains useful where its assumptions hold. The two prescriptions also difer in their exposure to the truncation choice: $\Lambda _ { \mathrm { m a x } }$ enters the Gaussian prescription twice, through the profiled directions in the test and through the fluctuations injected in the pseudoexperiments, whereas the bootstrap refits are constructed independently of it and dependence appears only in the test. As such, scanning $\Lambda _ { \mathrm { m a x } }$ , we find in App. C that the bootstrap calibration is robust against the choice of $\Lambda _ { \mathrm { m a x } }$ , while the Gaussian prescription shows a stronger dependence.

![](images/ef7d5e518faf8cd3b2a2134324d854aed7c1b6815a0d1069fd2daa742e63eae0.jpg)

![](images/7335bd9d7d8fb89273900302b8550e58b4b4154c50f785f9dd18aa53a74be2e7.jpg)  
FIG. 4. The marginal distribution of the two features in the 2D toy problem. The black crosses show the binned density evaluated with $1 0 ^ { 5 }$ held-out events. The solid line shows the marginals of a GMM with $K = 6 4$ components fit to $N _ { \mathrm { t r a i n } } = 1 0 0 , 0 0 0$ events, and the colorful envelopes are uncertainties corresponding to the calibration prescriptions of Sec. IV D. Specifically, the blue envelope corresponds to the propagated Gaussian uncertainties on the best-fit parameters, and the red envelope is obtained as the 0.16 and 0.84 quantiles of the refit models obtained using a parametric bootstrap.

In terms of the sources of Sec. II, the Gaussian prescription captures the data variability around a single optimum, while the bootstrap, which refits the model from a fresh initialization, also captures the optimization variability. This is why, in practice, we find that the bootstrap prescription yields uncertainty bands that are comparable to or larger than those of the Gaussian prescription. Neither, however, accounts for the model bias.

For the point-null baseline, the pseudo-experiments are simply drawn from $p _ { \hat { \theta } }$ with the construction held fixed.

c. Asymptotic $\chi ^ { 2 }$ limit. In the quadratic approximation to ℓ and without the ridge penalty $\lambda _ { \alpha }$ on the α, $t _ { \mathrm { c o m p } }$ follows a $\chi _ { J } ^ { 2 }$ distribution under the Gaussian composite-null, as shown in App. D. The same holds for $t _ { \mathrm { p o i n t } }$ under the point-null, and the ridge lowers the efective number of degrees of freedom below J. We will see in Sec. V that this approximate $\chi _ { J } ^ { 2 }$ prediction holds (up to a decrease in the efective degrees of freedom due to $\lambda _ { \alpha } )$ in the regime where the asymptotics are reliable and the Gaussian and bootstrap prescriptions are in good agreement, but that it breaks down outside of this regime (as it should, since the derivation of this $\chi _ { J } ^ { 2 }$ distribution assumes Gaussianity). As such, our results for test calibration are always evaluated using pseudo-experiments.

It is worth noting that the test described in Sections IV C and IV D is independent of how the density is estimated and of how its uncertainty is obtained, and it works provided this uncertainty is reliably estimated and can be propagated into the composite test.

## V. NUMERICAL EXPERIMENTS

In this section we apply the method presented in Sec. IV to two problems: a two-dimensional toy model with known analytic ground truth and a four-dimensional problem from particle physics.

![](images/8faad81adf965521b997d26e6feb438d176c744748a4792f7c78793ca6e4d61f.jpg)  
FIG. 5. Closure test for the 2D toy problem. Median null p-value for the three goodness-of-fit tests of Sec. IV C versus test-set size $N _ { \mathrm { t e s t } }$ , for $K = 1 6 , 3 2 , 6 4$ , 128 (left to right) at fixed $\bar { N _ { \mathrm { t r a i n } } } = 1 0 ^ { 5 }$ . A calibrated test gives median $p = 0 . 5$ (horizontal dotted line); the vertical dashed line mark $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$ . Error bars are 68% confidence intervals on the median.

## A. Toy model

We first validate the pipeline on a controlled twodimensional toy problem, where the ground truth is known analytically and can be resampled at will. The target factorizes into two independent features, each chosen to stress a diferent aspect of the fit:

• Feature 1: a bimodal Gaussian mixture, $p ( x _ { 1 } ) =$ $\begin{array} { r l r } { \frac { 1 } { 2 } \mathcal { N } ( x _ { 1 } ; - 0 . 7 0 , 0 . 1 2 ^ { 2 } ) } & { { } + } & { \frac { 1 } { 2 } \mathcal { N } ( x _ { 1 } ; - 0 . 3 0 , 0 . 1 2 ^ { 2 } ) } \end{array}$ whose modes are separated by more than three times their common width, with analytic mean $\mu _ { 1 } ^ { \star } = - 0 . 5 0$

• Feature 2: a skew-normal distribution with location $\xi = 1 . 0$ , scale $\omega = 0 . 7 5$ and shape $\alpha = 8 . 0 $ i.e. strong positive skew, with analytic mean $\mu _ { 2 } ^ { \star } =$ $\xi + \omega \delta \sqrt { 2 / \pi }$ and $\delta = { \alpha } / { \sqrt { 1 + { \alpha } ^ { 2 } } }$

The first feature probes the ability of the model to resolve well-separated modes, the second its ability to reproduce a long, asymmetric tail.

We draw $N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ training points and fit the Gaussian mixture model of Eqs. (1) and (2) for four choices of the number of components, $K = 1 6 , 3 2 , 6 4 , 1 2 8$ . We then estimate the model uncertainties following Sec. IV B and assess the quality of the fit with the point-null test of Eq. (17), which treats the fitted density as exact, and the composite test of $\operatorname { E q . }$ (16), which accounts for the uncertainty of the fit. For the composite test we report both our default bootstrap calibration as well as the Gaussian prescription of Sec. IV D. Because the ground truth is available in closed form, a fresh test sample is drawn for every replica, so the closure test is not subject to reuse of validation data across toys.

Figure 4 shows the marginal distributions of the two features together with a $K = 6 4$ fit. The fitted model (black solid line) is compared with the empirical distribution of a held-out sample the same size as the training set (× markers), with the data-to-fit ratio shown in the bottom panels. The 68% bands are built under the Gaussian prescription (light blue) and under the bootstrap prescription (light red). The Gaussian and bootstrap bands are overall comparable, with the bootstrap band slightly larger in regions where the Gaussian approximation underestimates the spread of the fit around the truth.

Figure 5 shows the median p-value for the point null as well as the Gaussian and bootstrap composite nulls as a function of the test-set size $N _ { \mathrm { t e s t } }$ for the four values of K. For each curve, a median p-value of 0.5 would indicate correct calibration, since under the null hypothesis the p-value is uniformly distributed by construction. We perform 20 independent model fits for each value of K, each evaluated on 100 test datasets per value of $N _ { \mathrm { t e s t } }$ The point-null and Gaussian composite statistics are cal ibrated with 250 pseudo-experiments per fit, while the bootstrap calibration uses 100 refits in total per fit, with the same refits reused across all values of $N _ { \mathrm { t e s t } }$

The point-null test is calibrated only in the small sample regime. Its median p-value is compatible with 0.5 for $N _ { \mathrm { t e s t } } < N _ { \mathrm { t r a i n } } / 1 0$ , starts degrading beyond that point, and collapses to zero by $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$ for every value of K we consider. The behavior is essentially independent of the number of mixture components $K$ , as the test fails because it neglects the uncertainty of the fit, regardless of the flexibility of the model.

Promoting the test to the composite-null, with the model uncertainties estimated under the Gaussian prescription, largely restores calibration at intermediate values of $N _ { \mathrm { t e s t } }$ . The median p-value stays compatible with 0.5 up to about $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } } / 5$ , but calibration still fails dramatically at $N _ { \mathrm { t e s t } } = 2 N _ { \mathrm { t r a i n } }$ , with a median p-value less than 0.2 for all K.

The bootstrap calibration prescription fully restores closure, remaining calibrated up to $N _ { \mathrm { t e s t } } = 2 N _ { \mathrm { t r a i n } }$ for all K. This is consistent with the bootstrap capturing the optimization variability and the non-Gaussian scatter of the fit, which the Gaussian description misses.

These results demonstrate that the point-null hypothesis is insuficient. We find that closure is dramatically improved through the use of a composite-null hypothesis, and that maintaining closure for $N _ { \mathrm { t e s t } } \gtrsim N _ { \mathrm { t r a i n } }$ requires bootstrap (rather than Gaussian) calibration. $\mathrm { F i g \mathrm { - } }$ ure 6 illustrates this directly for $K \ = \ 6 4$ at $N _ { \mathrm { t e s t } } ~ =$ $N _ { \mathrm { t r a i n } } { : }$ the observed point-null test statistic lies almost entirely above its null distribution, while the compositenull statistic is well reproduced by its bootstrap null.

![](images/487f1784eef12eafc3c9154f8fe3302e007f21ccebf65aec5f450e09580d97c6.jpg)  
FIG. 6. Calibrated uncertainties make the goodnessof-fit test cover. Distribution of the test statistic t for a GMM fit with $K = 6 4$ components at $N _ { \mathrm { t r a i n } } = N _ { \mathrm { t e s t } } = 1 0 ^ { 5 }$ to the 2D toy model. The observed point statistic of Eq. (17) (dashed gray step histogram) lies almost entirely above its point-null (gray filled histogram), which treats the fitted density as exact, so the point-null test rejects the ground truth essentially always. The observed composite statistic of Eq. (16) (dashed red step histogram) is instead reproduced by its bootstrap null (red filled histogram), which propagates the uncertainty of the fit, so the composite test closes. Each distribution is estimated using 20 model fits, each of which is used to evaluate the point and composite test statistic on 100 test datasets for each fit. The point-null is then estimated with 250 point-null toys and the bootstrap null is estimated with 100 bootstrap toys for each fit.

## B. Embedded particle jets

Recent advances in representation learning have made it possible to construct powerful neural embeddings of highly complex physics data. Among other applications, these embeddings have enabled anomaly detection in a latent space [61–66]. Anomaly detection in a latent space, however, requires a robust understanding of the density of the reference model. The method proposed in this paper ofers a path to estimate a background data distribution and its uncertainties with statistical guarantees.

As a demonstration, we emulate a realistic physics analysis aiming to identify anomalous jet decays. We use the JetClass dataset [67], a large-scale public bench mark for jet tagging. The JetClass dataset provides 100M simulated jets for training, 5M for validation, and 20M for testing. Each jet is represented as a particle cloud of charged and neutral hadrons, electrons, muons, and photons and labeled by the initiating parton and decay process, with ten equally represented classes. We restrict the analysis to purely hadronic final states and consider five classes: decay of top quarks, W and Z bosons, Higgs bosons and QCD multijet production.

![](images/43c92d94eefd002b4d65e72abf70741d76e3760810cb0164aa4096dbae9940df.jpg)  
FIG. 7. Particle jets embedded in a 4D space. Corner plot of the four-dimensional jet embedding learned with a supervised contrastive (SupCon) objective [68]. Diagonal panels show the one-dimensional marginal distributions of each embedding coordinate $( \alpha , \beta , \gamma , \delta )$ ; of-diagonal panels show the pairwise joint densities as contours. The QCD multijet density (red) is highlighted because it is the density we apply our framework to; other classes are used to obtain the embedding and are shown for reference only.

Following Ref. [65], the embedding is trained with a SupCon (contrastive) objective [68], which extends selfsupervised contrastive learning [69] to labeled data: all jets sharing a class label are treated as positives, and jets from diferent classes as negatives. For training the embedding, we restrict the JetClass training and validation set to 10% of the available statistics, corresponding to 5M and 250k jets, respectively, equally distributed among the five considered classes. The resulting representation is shown in Fig. 7.

![](images/58898d9c8ee54bc3a942ba45332c5acf951b638ef6c241b6ac02c2e27edef254.jpg)  
FIG. 8. The marginal distribution of the four features in the 4D particle jets problem. The black crosses show the binned density evaluated with $1 0 ^ { 5 }$ held-out events. The solid line shows the marginals of a GMM with $K = 2 5 6$ components fit to $N _ { \mathrm { t r a i n } } = 1 0 0 , 0 0 0$ events, and the colorful envelopes are uncertainties corresponding to the calibration prescriptions of Sec. IV D. Specifically, the blue envelope corresponds to the propagated Gaussian uncertainties on the best-fit parameters, and the red envelope is obtained as the 0.16 and 0.84 quantiles of the refit models obtained using a parametric bootstrap.

Anomaly detection via two-sample testing [12–14] requires either a large reference (simulated) sample or an equivalent of the underlying probability density function in the embedding space. Since large simulated samples can be prohibitively expensive to generate, we apply the SIGMA method as described in Sec. IV to model the QCD background in the embedding space.<sup>4</sup> We choose

QCD multijet production because it is a dominant background whose simulated statistics are hard to match to the large number of events expected in LHC data. We apply our pipeline to the full JetClass QCD multijet test partition with $N _ { \mathrm { t e s t } } ^ { \mathrm { T O T } } = 2 \mathrm { M }$ jets, which we split into a training pool and a testing pool each consisting of 1M jets. The dataset can be found on Zenodo [70].

![](images/9a19a754017fca06fd764ffb0869a7b0a7884645c3d372747d620e6b260f8d37.jpg)  
FIG. 9. Closure test for the 4D particle jets problem. Median null p-value for the three goodness-of-fit tests of Sec. IV C versus test-set size $N _ { \mathrm { t e s t } }$ , for $K = 1 2 8 , 2 5 6 , 5 1 2$ (left to right) at fixed $N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ . A calibrated test gives median $p = 0 . 5$ (horizontal dotted line); the vertical dashed line marks $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$ . Error bars are 68% confidence intervals on the median.

Each SIGMA training and p-value evaluation is performed by sampling from these finite pools, so we report closure results only up to $N _ { \mathrm { t e s t } } = \bar { N _ { \mathrm { t r a i n } } } = 1 0 ^ { 5 }$ events to mitigate correlations between diferent trainings and draws of pseudo-observed data.

Figure 8 shows the marginal distributions of the four latent features of our embedding for the QCD sample, together with a $K = 2 5 6 ~ \mathrm { f i t }$ . As for the two-dimensional problem, we compare the fitted model (black solid line) with the empirical distribution of a same-size held-out sample $( N = 1 0 ^ { 5 } )$ marked by the $^ { 6 6 } \times \vec { } ^ { , 9 }$ symbol. The datato-fit ratio is shown in the bottom panels. The 68% bands are built under the Gaussian prescription (light blue) and under the bootstrap prescription (light red).

Figure 9 shows the median p-value for the goodnessof-fit test using the point-null, the Gaussian composite and the bootstrap composite as a function of $N _ { \mathrm { t e s t } }$ for $K = 1 2 8 , 2 5 6 , 5 \bar { 1 } 2$ For each value of $K$ , we perform 10 independent model fits, each evaluated on 100 test datasets per value of $N _ { \mathrm { t e s t } }$ . The point-null and Gaussian composite statistics are calibrated with 250 pseudoexperiments per fit; since the bootstrap calibration is considerably more expensive, it uses 100 refits in total per fit, with the same refits reused across all values of $N _ { \mathrm { t e s t } }$

The results are qualitatively similar to those in 2D. At all values of $K ,$ the point-null test starts degrading before $N _ { \mathrm { t e s t } } \sim N _ { \mathrm { t r a i n } } / 1 0$ . The composite test with the Gaussian prescription improves the calibration, enabling closure up to $N _ { \mathrm { t e s t } } \sim N _ { \mathrm { t r a i n } } / 1 0$ for $K \geq 2 5 6$ . The bootstrap prescription further improves the results allowing closure up to $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } } / 5$ for $K \geq 2 5 6$ , with calibration degrading noticeably (but not completely) for larger $N _ { \mathrm { t e s t } }$

The K = 128 SIGMA model performs visibly worse than either of the two larger models, and the diference between the Gaussian and bootstrap null prescriptions is substantially smaller than for larger K. We interpret this as underparametrization-induced model bias in the sense of Sec. II, where a mixture model with only 128 components is insuficiently expressive to fit the data. This model misspecification is not quantified in our statistical uncertainties, so suficiently large $N _ { \mathrm { t e s t } }$ necessarily rejects this model.

By contrast, the calibration is evidently saturating with K for large $N _ { \mathrm { t e s t } }$ for the $K = 2 5 6 , 5 1 2$ models, with the larger diference between the bootstrap and Gaussian prescriptions indicating a breakdown in the asymptotics. This suggests that the finite $N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ , rather than model capacity, is likely the primary barrier to achieving a more performant SIGMA in the context of this case study, as could be expected from fitting a model with several thousand parameters. However, even with these limitations, the bootstrap composite null is calibrated to values of $N _ { \mathrm { t e s t } }$ an order of magnitude larger than the point-null.

To characterize the extent to which these results are specific to SIGMA models, we also test parametrized ensembles of normalizing flows,<sup>5</sup> and the results can be found in $\mathrm { A p p }$ . B 4. In short, we find that the SIGMA models achieve substantially better calibration properties than even ensembles of $\mathcal { O } ( 1 0 ^ { 2 } )$ flows, while being dramatically cheaper to train and evaluate.

## VI. CONCLUSION

All generative models trained on finite samples are inexact approximations of their training distribution, and a suficiently powerful goodness-of-fit test will eventually detect deviations. This is a consequence of finite information, not a failure of design. A density estimated from $N _ { \mathrm { t r a i n } }$ samples carries an inherent estimation error, so even a perfectly specified model is rejected by a powerful goodness-of-fit test. Validating a generative model against a point-null therefore measures the size of the test sample rather than the quality of the model, and no architectural or training improvement can change this.

An approximate model can nonetheless be useful, provided its estimation error is known. In this work, we make this statement quantitative. By linearizing a Gaussian mixture model around its best fit, we obtain the covariance of its parameters from its analytic score functions and Fisher information, and the goodness-of-fit test becomes a convex problem. We find that promoting the null from a point hypothesis to a composite hypothesis, together with a parametric bootstrap calibration procedure, is able to calibrate the goodness-of-fit test on datasets an order of magnitude larger than the point-null equivalent.

The success of SIGMA is in contrast with neural generative models trained through stochastic gradient descent, which typically do not converge to a well-defined optimum and which have non-invertible information matrices due to parameter degeneracies. We find in App. B that even a parametrized ensemble of normalizing flows is only able to capture part of the uncertainty, and cannot match the performance of SIGMA in our low-dimensional case studies.

Uncertainty quantification for generative models bears directly on how generated samples can be used. A generative model cannot create information that was not in its training data [10]. What it can do is produce arbitrarily many samples whose deviation from the truth is bounded by stated uncertainties. Generated samples can therefore be used beyond the training statistics only if these uncertainties are carried along into the downstream analysis, e.g. by using a composite-null hypothesis in the context of a goodness-of-fit or hypothesis test. For statistical inference, the true figure of merit of a generative surrogate is therefore the precision to which it can be trusted.

Within this framework, SIGMA still has clear limitations. Because the number of parameters of a GMM grows quickly with dimension, the method is suited to low-dimensional spaces rather than directly to highdimensional ones. This further motivates the compression of high-dimensional data into a small number of summary statistics, such as through learned embeddings [61– 66].

Furthermore, SIGMA accounts for the statistical variability of the fit due to the finite training sample and to the stochasticity of optimization, but it is still subject to model bias: a model family that cannot represent the data will still be rejected by a suficiently powerful test. The need to reduce model bias makes the choice of $K$ a trade-of. Increasing the number of components reduces mismodeling but it also adds parameters, leading to larger uncertainties and potentially undermining the calibration procedures. The Gaussian calibration mani festly requires the asymptotic regime where $N _ { \mathrm { t r a i n } } \gg P ,$ and the bootstrap calibration relies on this indirectly through the assumption that the estimator spread is similar for fits to draws from $p _ { 0 }$ and from $p _ { \hat { \theta } }$

We currently choose $K .$ , as well as the truncation hyperparameter $\Lambda _ { \operatorname* { m a x } } ,$ empirically. These model selection choices, as well as verifying that the model can be trusted beyond its training statistics, require closure tests on held-out samples at least as large as the desired inference dataset. This is costly in exactly the simulation-scarce settings where generative models are most useful, and the necessity of the bootstrap for the most trustworthy calibration adds the cost of refitting the model for every pseudo-experiment.

Each of these limitations ofers promising avenues for progress. Both the restriction to low-dimensionality and the model bias considerations motivate exploring distinct model classes. In particular, the test of Sec. IV C is agnostic to the model class: it requires only tractable score vectors, which automatic diferentiation supplies for any neural density estimator, and a faithful estimate of the uncertainty of the fit. Our tangent-space construction is then the exact, low-dimensional counterpart of the lin earized Laplace approximations developed for deep networks [71, 72], and that literature’s low-rank and subspace curvature estimates ofer a natural route to extending the composite-null test to larger models; the open problem is promoting such uncertainties from predictive heuristics to calibrated ingredients of a goodness-of-fit test.

On the other hand, the need for empirical tuning of $K$ and $\Lambda _ { \mathrm { m a x } }$ could be eliminated with a data-driven procedure for hyperparameter selection. Such a procedure could involve the use of cross-validation to check closure on subsets of held-out simulation data.

Moreover, it would be interesting to consider appli cations with systematic uncertainties. Because the fit uncertainty enters as a standard constrained nuisance vector, the Gaussian calibration prescription composes naturally with the existing treatment of systematic uncertainties [15, 73]: additional nuisance directions are profiled jointly, and a single composite test accounts for all sources of uncertainty on the reference model. In contrast, composing our nominal bootstrap calibration prescription (which does not yield straightforwardly parametrized uncertainties) with additional nuisance parameters may be more challenging.

The immediate application of our method is background estimation for anomaly detection in lowdimensional learned embeddings. There, a validated model could enter the final test in closed form, together with its uncertainties, as the composite-null hypothesis. More broadly, as generative models increasingly substitute for simulation across various scientific fields, uncertainty-aware validation is necessary for these surrogates to serve as instruments of trustworthy statistical inference.

The code required to reproduce our experiments and implement SIGMA can be found here: https://github. com/gbadarac/UQ-GoF-GenModels/tree/paper.

## ACKNOWLEDGMENTS

We are especially grateful to Thea <sup>˚</sup>Arrestad for her supervision, guidance and support in the development of this work. We also thank Davide Valsecchi for discussions on uncertainty quantification for generative models and for his detailed feedback on the manuscript. We further thank Prasanth Shyamsundar for conversations on the information content of generated samples and the limits of data amplification. The authors acknowledge the financial support of the National Science Founda tion under Cooperative Agreement PHY-2019786 (NSF AI Institute for Artificial Intelligence and Fundamental Interactions, http://iaifi.org/), and the National Science Foundation AI Expand program (OAC-2334265). C.R. further acknowledges support under a Postdoc.Mobility fellowship from the Swiss National Science Foundation (SNSF) (Project No. P500PT 222340). Claude $( \mathrm { A n } \cdot$ thropic) assisted with manuscript editing and code development. All scientific ideas and experiments are the original work of the authors.

## Appendix A: Additional discussion on the limits of generative models

In this appendix, we quantify the statistical uncertainty of the fit introduced in Sec. II in terms of the Kullback–Leibler (KL) divergence between the true and the fitted densities, and derive the test sample size at which it becomes detectable by a goodness-of-fit test. We consider a parametric family p<sub>θ</sub> with $\theta \in \mathbb { R } ^ { P }$ and assume the best-case scenario of Sec. II: the model is well specified, $p _ { 0 } = p _ { \theta _ { 0 } }$ for some interior point $\theta _ { 0 }$ with finite, non-singular Fisher information $F ( \theta _ { 0 } )$ , and the optimizer returns the global maximum-likelihood estimate $\hat { \theta }$ computed on $N _ { \mathrm { t r a i n } }$ independent samples of $p _ { 0 } \ ( \mathrm { i . e }$ . we isolate the data variability; any optimization variability would only add to the result below).

Expanding $\theta \ \mapsto \ \operatorname { K L } ( p _ { 0 } \| p _ { \theta } )$ around $\theta _ { 0 } .$ , the zerothorder term vanishes because $p _ { 0 } = p _ { \theta _ { 0 } }$ , and the gradient vanishes because $\theta _ { 0 }$ minimizes the divergence over the family. The curvature at the minimum is the Fisher information: since the θ-dependence of the divergence sits entirely in the cross-entropy $- \mathbb { E } _ { p _ { 0 } } [ \log p _ { \theta } ]$ , its Hessian is

$$
F ( \theta _ { 0 } ) = - \mathbb { E } _ { p _ { 0 } } \left[ \nabla _ { \theta } \nabla _ { \theta } ^ { \top } \log p _ { \theta } \right] _ { \theta _ { 0 } } .\tag{A1}
$$

To leading order, therefore,

$$
\begin{array} { r } { \mathrm { K L } ( p _ { 0 } \| p _ { \hat { \theta } } ) \simeq \frac { 1 } { 2 } ( \hat { \theta } - \theta _ { 0 } ) ^ { \top } F ( \theta _ { 0 } ) ( \hat { \theta } - \theta _ { 0 } ) : } \end{array}\tag{A2}
$$

the divergence of the fitted density is a quadratic form in the estimation error, with the information matrix as its metric. The maximum-likelihood estimate asymptotically fluctuates around the true value as $\hat { \theta } - \theta _ { 0 } \sim$ $\mathcal { N } ( 0 , \bar { F } ( \theta _ { 0 } ) ^ { - 1 } / { N _ { \mathrm { t r a i n } } } )$ , and for a Gaussian vector u ∼ $\mathcal { N } ( 0 , \Sigma )$ one has $\mathbb { E } [ u ^ { \top } A u ] = \operatorname { t r } ( A \Sigma )$ . Averaging Eq. (A2) over the training sample then gives, for large enough $N _ { \mathrm { t r a i n } } .$

$$
\mathbb { E } \big [ \mathrm { K L } ( p _ { 0 } \| p _ { \hat { \theta } } ) \big ] \simeq \frac { \mathrm { t r } \big ( F F ^ { - 1 } \big ) } { 2 N _ { \mathrm { t r a i n } } } = \frac { P } { 2 N _ { \mathrm { t r a i n } } } .\tag{A3}
$$

This is the data variability, expressed in terms of the KL divergence. Note that the Fisher information cancels against its own inverse: every identifiable direction in parameter space contributes $1 / ( 2 N _ { \mathrm { t r a i n } } )$ nats to the expected divergence, independently of how tightly the data constrain it. A poorly measured direction fluctuates more, but the divergence is correspondingly flat along it, and the two efects compensate. Only the number of directions, $P ,$ survives. Equivalently, $2 N _ { \mathrm { t r a i n } } \mathrm { K L } ( p _ { 0 } \| p _ { \hat { \theta } } )$ is asymptotically the Wald statistic $N _ { \mathrm { t r a i n } } ( \hat { \theta } - \theta _ { 0 } ) ^ { \top } F ( \hat { \theta } - \theta _ { 0 } )$ , which, like the likelihood-ratio statistic of Wilks’ theorem, is $\chi _ { P } ^ { 2 }$ -distributed with mean $P .$ For the Gaussian example of Fig. $2 , P = 2$ , and indeed $\mathrm { K L } \simeq \hat { \mu } ^ { 2 } / 2 + ( \hat { \sigma } - 1 ) ^ { \hat { 2 } }$ with $\mathrm { V a r } ( \hat { \mu } ) = 1 / N _ { \mathrm { t r a i n } }$ and $\mathrm { V a r } ( \hat { \sigma } ) = 1 / ( 2 N _ { \mathrm { t r a i n } } )$ , so that $\mathbb { E } [ \mathrm { K L } ] \simeq 1 / N _ { \mathrm { t r a i n } }$

A goodness-of-fit test performed with $N _ { \mathrm { t e s t } }$ samples becomes sensitive to a discrepancy of size $D$ (in KL) once $N _ { \mathrm { t e s t } } D \gtrsim \mathcal { O } ( 1 )$ This is the Neyman–Pearson benchmark: for the most powerful test of $p _ { \hat { \theta } }$ against the specific alternative $p _ { 0 } .$ , the expected log-likelihood ratio per test sample is $\bar { \mathbb { E } _ { p _ { 0 } } } [ \log ( p _ { 0 } / \bar { p _ { \hat { \theta } } } ) ] = \bar { \mathrm { K L } } ( p _ { 0 } \| p _ { \hat { \theta } } ) = D$ , so the evidence accumulated over the test set, $N _ { \mathrm { t e s t } } D .$ , must exceed its $\mathcal { O } ( 1 )$ null fluctuations. Substituting Eq. (A3), which represents the discrepancy that the most powerful test would detect, this threshold is crossed once

$$
N _ { \mathrm { t e s t } } \ \gtrsim \ \frac { N _ { \mathrm { t r a i n } } } { P } .\tag{A4}
$$

This argument should be understood as establishing a sensitivity ceiling for goodness-of-fit tests, describing when the training-set discrepancy becomes detectable in principle, not necessarily when a specific test statistic used in practice (such as the test statistics described in Sec. IV C) rejects. The actual rejection point of a test depends on its power against this particular alternative and on its empirical calibration, and can in general occur later, and with a diferent O(1) prefactor, than this bound suggests: a statistic built on J degrees of freedom, for instance, resolves the same D only once $N _ { \mathrm { t e s t } } D \gtrsim \sqrt { J / 2 }$ , which rescales Eq. (A4) by a constant without afecting the scaling in $N _ { \mathrm { t r a i n } }$ and $P .$

Two qualifications delimit the validity of Eq. (A3) itself. First, it is the leading term of an asymptotic expansion: it requires $N _ { \mathrm { t r a i n } }$ large enough for the quadratic approximation (A2) to hold across the sampling distribution of ${ \hat { \theta } } ,$ which fails when P is not small compared to $N _ { \mathrm { t r a i n } }$ or when some directions become non-identifiable (e.g. degenerate mixture components); in that regime the efective number of parameters is smaller than $P .$ . Second, under misspecification the same expansion around the KL projection $\theta ^ { * }$ of $p _ { 0 }$ onto the family yields a variance term tr $( A ^ { - 1 } B ) / ( 2 \dot { N } _ { \mathrm { t r a i n } } )$ , with A the expected curvature and B the score covariance $( A \ : = \ : B \ : = \ : F$ for a correct model), on top of the bias floor $\mathrm { K L } ( p _ { 0 } \| p _ { \theta ^ { * } } )$ which does not shrink with $N _ { \mathrm { t r a i n } }$ . The floor eventually dominates the discrepancy, and the rejection point saturates at $N _ { \mathrm { t e s t } } \sim 1 / \mathrm { K } \bar { \mathrm { L } } ( p _ { 0 } \bar { \| } p _ { \theta ^ { * } } )$ independently of further growth in $N _ { \mathrm { t r a i n } } { : }$ : this is the bias that uncertainties cannot absorb and that must be addressed at the level of model design.

These considerations sharpen the usage criteria of Fig. 1 [11]. When the fitted density enters an inference as the reference hypothesis, its estimation error may be neglected only while it lies below the sensitivity of that inference, and the validation test used to establish this must be at least as powerful as the downstream analysis (see the discussion of Sec. III); otherwise, mismodeling that validation cannot see may still surface downstream as a false positive. Equation (A4) bounds how long such safety can last for any analysis: beyond $N _ { \mathrm { t e s t } } \gtrsim N _ { \mathrm { t r a i n } } / P$ even a perfectly specified model is rejected when tested, and the uncertainties of the fit must be incorporated as a composite null, as described in Sec. IV C.

## Appendix B: Normalizing flows and the limits of ensemble uncertainties

Normalizing flows [55] are among the most widely used and flexible neural density estimators in highenergy physics, both as surrogates for expensive simulation [2, 31] and as background models [4–7, 25], including density estimation in learned embedding spaces for anomaly detection searches [66]. Unlike the Gaussian mixtures of the main text, a flow has no tractable parameter covariance, so the score-based, full-parameter-space construction of Sec. IV is unavailable.

In this appendix, we consider the same two numerical experiments from Sec. V, a two-dimensional toy with a bimodal Gaussian and a skew-normal feature and a four-dimensional latent space for jet representations, and we ask the following two questions: (1) does the epistemic uncertainty of a normalizing flow ensemble admit a parametrization that a likelihood-ratio test can profile over? and (2) with this uncertainty propagated in the null, does a normalizing flow ensemble pass a composite goodness-of-fit test such as the one presented in the main text?

Ref. [74] provides such a parametrization through the maximum-likelihood $w _ { i } f _ { i }$ ensembling of Ref. [24], in which independently trained flows are frozen and combined into a weighted mixture whose weights carry the uncertainty. This appendix extends that work in two directions: to more complex target distributions, and to a goodness-of-fit test built on the same construction as the main text.

We describe how we train the normalizing flow models in Sec. B 1, how we estimate uncertainties with the $w _ { i } f _ { i }$ ensembling in Sec. B 2, and how we incorporate the uncertainties in the goodness-of-fit test in Sec. B 3. The results of our numerical experiments, presented in Sec. B 4, show that a normalizing-flow ensemble admits such a weight parametrization, and that propagating it restores closure up to and beyond $N _ { \mathrm { t e s t } } \simeq N _ { \mathrm { t r a i n } }$ for the largest ensemble (M = 128) in the two-dimensional toy, but not in the four-dimensional embedding, where the propagated weight uncertainty is negligible compared to the flow’s misspecification: the composite test collapses onto the point-null and the model is rejected well before $N _ { \mathrm { t e s t } } = \hat { N } _ { \mathrm { t r a i n } }$

## 1. Density estimation with normalizing flows

A normalizing flow learns an invertible, diferentiable map $T _ { \phi }$ from a simple base distribution to the target,

$$
f _ { \phi } ( x ) = \ d p _ { z } \big ( T _ { \phi } ^ { - 1 } ( x ) \big ) \left| \operatorname* { d e t } \frac { \partial T _ { \phi } ^ { - 1 } ( x ) } { \partial x } \right| , \qquad \ d p _ { z } = \mathcal { N } ( 0 , \mathbb { I } _ { D } ) ,\tag{B1}
$$

and is trained with a maximum likelihood objective. We use nflows [75] with masked rational-quadratic autoregressive layers, reverse permutations between layers, dropout 0.2, and a standard normal base distribution. The architecture is chosen separately for each dataset by a grid search on held-out likelihood: four layers (8 bins, 4 blocks, hidden width 64) for the two-dimensional toy, and a larger flow with six layers (10 bins, 8 blocks, hidden width 128) for the four-dimensional jet embedding. The training settings are the same for both datasets and were kept fixed in both grid searches: Adam [76] with batch size 512 and an initial learning rate of $1 0 ^ { - 5 }$ , cosineannealed to $1 0 ^ { - 6 }$ over 1000 epochs, with early stopping (patience 10 epochs) on a 20% validation split of each member’s training sample. In both cases training ends by early stopping, typically after ∼100 epochs, well before the 1000-epoch limit.

For each of the two problems (the two-dimensional toy and the four-dimensional jet embedding), we train a pool of 512 flows, each an independent run on its own bootstrap replica of the $\ r _ { N _ { \mathrm { t r a i n } } } \dot { \ = \ 1 0 ^ { 5 } }$ training set, so the spread across members reflects both data and optimization variability. An ensemble of size M is built by drawing M members uniformly at random from this pool, and we show $M = 1 , 3 2 , 1 2 8$ . Both the fitted density and its propagated weight uncertainty of Eq. (B5) depend on which members happen to be drawn, so a single draw gives a noisy estimate of the uncertainty. To remove this dependence, we repeat the draw ten times, obtaining ten independent ensemble choices of size M, each fit independently (Sec. B 2). In the marginal figures below the fitted model is the median over the ten choices and the band is the per-choice weight uncertainty averaged over them. This same per-choice uncertainty is what the goodness-of-fit test profiles over: it enters the null as the Gaussian prior $\pi ( { \boldsymbol { \mathbf { u } } } ) = \mathcal { N } ( \hat { \boldsymbol { \mathbf { u } } } , { \boldsymbol { \Sigma } } _ { u } )$ of Sec. B 3, so the band shown here is exactly the uncertainty propagated into the test. Each of the ten choices is then tested and calibrated independently, and the ten results are pooled into the reported p-value and its band (Sec. B 4).

## 2. Uncertainty quantification via $w _ { i } f _ { i }$ ensembling

Following [24], we freeze the trained flows $f _ { i } ( x )$ as fixed basis functions and place the uncertainty on a small set of mixture weights $w _ { i }$ . The ensemble density is

$$
g ( x ) = \sum _ { i = 1 } ^ { M } w _ { i } f _ { i } ( x ) , \qquad w _ { M } = 1 - \sum _ { i = 1 } ^ { M - 1 } w _ { i } ,\tag{B2}
$$

where the sum-to-one constraint preserves normalization and leaves $M - 1$ free weights $\pmb { u } = \left( w _ { 1 } , \dots , w _ { M - 1 } \right)$ . The underlying assumption is that the true density lies in the span of $\{ f _ { i } \}$ . The weights are signed $( w _ { i } \ \geq \ 0$ is not imposed because allowing cancellations between the networks improves expressivity) and are fit by minimizing the negative log-likelihood on the training sample,

$$
\hat { \pmb { u } } = \arg \operatorname* { m i n } _ { \pmb { u } } \left[ - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log g ( x _ { n } ) \right] ,\tag{B3}
$$

which is convex in u on the feasible set $\{ { \pmb u } : { \pmb g } ( { \pmb x } _ { n } ) >$ $0 \forall n \}$ . We find uˆ by damped Newton steps with a feasibility line search. The resulting estimator $\hat { g } ( x ) = g ( x ; \hat { { \boldsymbol u } } )$ is the best estimate of the ensemble density.

Treating uˆ as an M-estimator [77], its covariance is given by the sandwich formula [57]

$$
\begin{array} { r l r } { \mathrm { C o v } ( \hat { \pmb u } ) } & { = } & { \displaystyle \frac { 1 } { N } { \cal A } ^ { - 1 } { \cal B } { \cal A } ^ { - 1 } , } \\ { { \cal A } } & { = } & { \displaystyle \left. \frac { \partial ^ { 2 } { \cal L } } { \partial { \pmb u } \partial { \pmb u } ^ { \top } } \right| _ { \hat { \pmb u } } , \qquad { \cal B } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } s _ { n } ^ { \top } . } \end{array}\tag{B4}
$$

Here $\mathcal { L }$ is the mean negative log-likelihood of Eq. (B3), A its Hessian at the optimum, and B the covariance of the per-event scores $s _ { n } ,$ whose components are the weightdiference sensitivities $s _ { n , j } = - \bigl ( f _ { j } ( x _ { n } ) - f _ { M } ( x _ { n } ) \bigr ) / g ( x _ { n } )$ We denote this $( M { - } 1 ) \ \overset { ^ { \cdot } } { \times } \ ( M \overset { ^ { \cdot } } { - } \overset { _ { 1 } } { 1 } )$ covariance by $\Sigma _ { u } \equiv$ $\mathrm { C o v } ( \hat { \pmb u } )$ . Propagating it to the density through the same sensitivities gives the pointwise predictive variance

$$
\sigma _ { \hat { g } } ^ { 2 } ( x ) \ = \ \sum _ { j , k = 1 } ^ { M - 1 } \left( f _ { j } ( x ) - f _ { M } ( x ) \right) \Sigma _ { u , j k } \left( f _ { k } ( x ) - f _ { M } ( x ) \right) ,\tag{B5}
$$

i.e. an uncertainty band $\hat { g } ( x ) \pm \sigma _ { \hat { g } } ( x )$ that supplies the pointwise epistemic uncertainty later propagated into the goodness-of-fit test of Sec. $\mathrm { { B 3 . ^ { 6 } } }$

Equation (B4) is the exact analogue of the sandwich covariance discussed in the main text,<sup>7</sup> but restricted to the $M - 1$ mixture weights: it captures only re-weightings $\textstyle \sum _ { i } \delta w _ { i } f _ { i } ( x )$ of the fixed members, not deformations of any individual flow. As shown in Sec. B 4, this is the reason why the composite test built on it approaches the uniform null $p \textmd { - }$ value only in the 2D toy, and fails to improve on the point-null in the 4D embedding.

## 3. Uncertainty-aware goodness-of-fit test

We test the ensemble density with the composite-null likelihood-ratio construction of the main text $( \bar { \mathrm { S e c . } } \mathrm { I V C } )$ now applied with the mixture weights as the profiled parameters.

a. Null. The null is the family of re-weighted mixtures $g$ of $\operatorname { E q }$ . (B2), with u constrained by the Gaussian prior centered on the fit,

$$
\pi ( { \pmb u } ) = \mathcal { N } \big ( { \pmb u } ; \hat { { \pmb u } } , \Sigma _ { u } \big ) .\tag{B6}
$$

The trainable parameters u are profiled under this prior when the statistic is evaluated.

b. Alternative. The alternative enriches the null by a multiplicative exponential tilt along a fixed dictionary of localized kernels, as in the main text [59],

$$
h _ { { \pmb u } , { \pmb \alpha } } ( { \pmb x } ) = \frac { g _ { { \pmb u } } ( { \pmb x } ) \exp \big ( \sum _ { j = 1 } ^ { J } \alpha _ { j } k _ { j } ( { \pmb x } ) \big ) } { Z ( { \pmb u } , { \pmb \alpha } ) } ,\tag{B7}
$$

$$
Z ( \pmb { u } , \pmb { \alpha } ) = \int g _ { \pmb { u } } ( x ) \exp \big ( \sum _ { j = 1 } ^ { J } \alpha _ { j } k _ { j } ( x ) \big ) \ : d x ,\tag{B8}
$$

with Gaussian kernels $k _ { j } ( x ) = ( 2 \pi \sigma ^ { 2 } ) ^ { - D / 2 } \exp \bigl ( - \mathrm { \Omega } \| x -$ $c _ { j } \| ^ { 2 } / 2 \sigma ^ { 2 } )$ The kernel dictionary is similar to that of the main text (Sec. IV C), using J Gaussian kernels of a single width σ (values in Sec. B 4), but in this case centered on the first J points of the sample under test. Since the $J = 5 0 0$ kernels are each about as wide as the spread of the data and overlap heavily, so every region populated by the data is covered whichever way the centers are placed, and we expect the choice of placement to have no significant efect on the test. Setting $\alpha = 0$ recovers the null $( h = g )$ , so the null is nested in the alternative. The exponential tilt preserves the sign of $g _ { u } ,$ and h is normalized up to the Monte Carlo estimate $\widehat { Z }$ of Eq. (B9).

As with the test in the main text, two regularizations   
act on the tilt: an $L _ { 2 }$ ridge $\scriptstyle { \frac { 1 } { 2 } } \lambda \| \alpha \| ^ { 2 }$ and a box $| \alpha _ { j } | \le A$ on   
the tilt coeficients. The normalization (B8) is estimated   
by importance sampling from the equal-weight member   
mixture $\begin{array} { r } { q ( x ) = \frac { 1 } { M } \dot { \sum _ { m } { f _ { m } ( x ) } } } \end{array}$ 2

$$
\widehat { Z } ( { \pmb u } , { \pmb \alpha } ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \frac { g _ { { \pmb u } } ( y _ { r } ) } { q ( y _ { r } ) } \exp \big ( \sum _ { j } \alpha _ { j } k _ { j } ( y _ { r } ) \big ) ,\tag{B9}
$$

with $y _ { r } \sim q$ and $R = N _ { \mathrm { r e f } } = N _ { \mathrm { t e s t } }$ reference points. The reference weights are self-normalized so that $\widehat Z ( u , \mathbf { 0 } ) = 1$

c. Test statistic. The test statistic follows the construction of the main text (Sec. IV C), with the mixture weights u in place of the whitened coordinates v and the prior $\pi ( { \boldsymbol u } )$ in place of the Gaussian constraint on v. Following the standard LHC nuisance-parameter prescription [16], we define the penalized log-likelihood

$$
\begin{array} { r } { \ell ( { \pmb u } , { \pmb \alpha } ) = \log \mathcal { L } ( { \mathcal { D } } \mid h _ { { \pmb u } , { \pmb \alpha } } ) + \log \pi ( { \pmb u } ) - \frac { \lambda } { 2 } \| { \pmb \alpha } \| ^ { 2 } . } \end{array}\tag{B10}
$$

The composite statistic is the profile likelihood ratio

$$
t _ { \mathrm { c o m p } } ( \mathcal { D } ) = 2 \Big [ \operatorname* { m a x } _ { \boldsymbol { u } , \boldsymbol { \alpha } } \ell ( \boldsymbol { u } , \boldsymbol { \alpha } ) - \operatorname* { m a x } _ { \boldsymbol { u } } \ell ( \boldsymbol { u } , 0 ) \Big ] ,\tag{B11}
$$

where the second term lets the ensemble move within its weight uncertainty to best fit the data, and the first additionally switches on the localized kernels, so that only the discrepancy the weights cannot absorb contributes. As in the main text, the point-null statistic fixes the weights at their best fit,

$$
t _ { \mathrm { p o i n t } } ( \mathcal D ) = 2 \Big [ \operatorname* { m a x } _ { \pmb { \alpha } } \ell ( \hat { \pmb u } , \pmb { \alpha } ) - \ell ( \hat { \pmb u } , 0 ) \Big ] .\tag{B12}
$$

d. Calibration. The training data (used to fit the flows and the weights uˆ) are kept strictly separate from the sample D on which t is evaluated. The null distribution of t is built from B pseudo-experiments. We draw posterior-predictive toys: each toy draws a weight vector $\bar { \mathbf { \boldsymbol { u } } } ^ { ( b ) } \sim \pi$ and then a dataset $\mathcal { D } ^ { ( b ) } \sim g _ { u ^ { ( b ) } }$ , so that the null itself carries the weight uncertainty. Because the weight fit of Eq. (B3) is convex, unlike the non-convex mixture fit of the main text, whose toys must be re-fit from a cold start, re-fitting each toy would recover a $\mathbf { \pmb { u } } ^ { ( b ) }$ in the vicinity of uˆ regardless of initialization. We then expect drawing ${ \pmb u } ^ { ( b ) } \sim \bar { \boldsymbol \pi }$ to perform similarly, so we do not re-fit the data-generating weights per toy and thus only show the Gaussian calibration. A full bootstrap calibration which refits the whole ensemble of flows in every pseudoexperiment could outperform this procedure, but would be computationally prohibitive.

Because the weights are signed, $g _ { \pmb { u } }$ is not a probability mixture and cannot be sampled member-by-member. We instead draw toys by sampling-importance-resampling (SIR) from the positive proposal q. The statistic is then evaluated on each toy exactly as on the data, remaximizing over u and α as in Eq. (B11), giving $\{ t ^ { ( b ) } \}$ and the p-value of the observed statistic is the empirical tail probability $p = ( \# \{ t ^ { ( b ) } \geq t ( \mathcal { D } ) \} + 1 ) / ( B + 1 )$ . For the point-null statistic, toys are drawn from $g _ { \hat { \pmb u } }$ without sampling the weights. Because the tilt is regularized (ridge and box) and its kernel dictionary is strongly collinear, neither statistic follows the asymptotic Wilks $\chi ^ { 2 }$ , whose number of degrees of freedom would equal the number of tilt coeficients; the efective number of degrees of freedom is much smaller. We therefore do not rely on Wilks’ theorem, but calibrate the statistics empirically from the pseudo-experiments, against which all p-values in Sec. B 4 are evaluated.

## 4. Results

We apply the construction of Secs. B 1–B 3 to the twodimensional toy of Sec. $\mathrm { \Delta V } \mathrm { A } \left( N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 } \right)$ , for ensembles of M flows up to $M = 1 2 8$ The tests use the kernel dictionary of the main text $( J = 5 0 0 , \sigma = 0 . 4 )$ , ridge $\lambda = 1$ and box $A = 3$ . Each null is calibrated on $B = 1 0 0$ pseudo-experiments and the p-value is estimated from 100 draws of $N _ { \mathrm { t e s t } }$ events from the truth. We then repeat the construction on the four-dimensional jet embedding of Sec. V B, with the same $N _ { \mathrm { t r a i n } } ,$ ensemble sizes and test settings, and the four-dimensional kernel width of the main text, $\sigma = 0 . 5 3$

a. Density quality. Figure 10 shows the fitted twodimensional marginals for $M = 1 , 3 2 , 1 2 8$ . A single flow $( M = 1 )$ , which carries no parametrized uncertainty, reproduces the bulk of the target but visibly mismodels the peaks and the valley of the bimodal feature 1. As M increases the ensemble median tightens onto the target (most visibly the bimodal valley of feature 1) and a weight-uncertainty band develops, narrow in the bulk and widening in the low-statistics tails. In the fourdimensional embedding (Fig. 11) the picture is diferent: the fitted marginals show structured deviations from the held-out data, most visibly around the sharp peaks, while the weight-uncertainty band remains negligibly small at every M. The propagated uncertainty therefore does not cover the discrepancy between the ensemble and the data, which anticipates the goodness-of-fit results below.

b. Goodness-of-fit. Fig. 12 shows the median test pvalue versus $N _ { \mathrm { t e s t } }$ for $M = 3 2 , 1 2 8$ . As expected, the test sensitivity increases with $N _ { \mathrm { t e s t } } { : }$ the p-value stays near nominal while the test sample is too small to resolve the finite- $N _ { \mathrm { t r a i n } }$ error of the fit, and falls once it does. The single flow degrades earliest, by $N _ { \mathrm { t e s t } } \sim 1 0 ^ { 4 }$ as anticipated in Sec. II: a flow’s large parameter count and imperfect inductive bias push the rejection onset to $N _ { \mathrm { t e s t } } \ll N _ { \mathrm { t r a i n } }$ . A single flow is therefore not a valid reference model at these test sizes. Ensembling and propagating the weight uncertainty push this onset outward, and the benefit grows with M. A moderate ensemble $( M = 3 2 )$ delays the degradation to $N _ { \mathrm { t e s t } } \sim 2 \mathrm { - } 5 { \times } 1 0 ^ { 4 }$ but still reaches the floor $1 / ( B + 1 )$ by $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ with little diference between the composite and pointnull (median $p \approx 0 . 0 9$ vs 0.07 at $N _ { \mathrm { t e s t } } = 5 \times 1 0 ^ { 4 } )$ . This happens although the $M = 3 2$ ensemble closely reproduces the two-dimensional marginals (Fig. 10): the test probes the joint density, where the residual error of the fit is resolved. Only at $M = 1 2 8$ does propagating the weight uncertainty make a clear diference: the composite test stays near nominal, $p \approx 0 . 4 3$ , whereas the point-null falls to $p \approx 0 . 1 6$ at $N _ { \mathrm { t e s t } } = 5 \times 1 0 ^ { 4 }$ and to the floor by $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$ With $M = 1 2 8$ the composite test gives $p \approx 0 . 4 3$ out to $N _ { \mathrm { t e s t } } = 2 \times 1 0 ^ { 5 }$ , i.e. it does not reject the model even in the genuine oversampling regime $N _ { \mathrm { t e s t } } > N _ { \mathrm { t r a i n } }$

In four dimensions the picture changes $\left( { \mathrm { F i g . ~ 1 3 } } \right)$ . The single flow degrades by $\bar { N _ { \mathrm { t e s t } } } \sim 2 \times 1 0 ^ { 3 }$ and the ensembles by $\bar { N } _ { \mathrm { t e s t } } \sim 1 \bar { 0 } ^ { 4 }$ There is no improvement from propagating the weight uncertainty: the composite p-value is inseparable from the point-null within the pseudoexperiment variations at every M and every $N _ { \mathrm { t e s t } }$ , with no separation even at $M = 1 2 8$ . All constructions reach the floor $1 / ( B + 1 )$ by $N _ { \mathrm { t e s t } } ~ \sim ~ 2 ~ \times ~ 1 0 ^ { 4 }$ , well below $N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ , and even $M = 1 2 8$ is fully rejected.

c. Discussion. Since every null is calibrated with pseudo-experiments drawn from the composite-null itself, the loss of p-value at large $N _ { \mathrm { t e s t } }$ reflects a failure of the ensemble uncertainty to cover the error of the $\operatorname { f i t } ,$ not an artifact of the statistic. Ensembling the flows and propagating the weight uncertainty push this rejection outward in the 2D toy study, and with $M = 1 2 8$ the model survives up to $N _ { \mathrm { t e s t } } \simeq N _ { \mathrm { t r a i n } }$ and into the oversampling regime $N _ { \mathrm { t e s t } } > N _ { \mathrm { t r a i n } }$ probed here, up to $N _ { \mathrm { t e s t } } = 2 N _ { \mathrm { t r a i n } } .$ In the more complex 4D problem, the construction fails to close: ensembling still improves, but the weight covariance $\Sigma _ { u }$ spans only re-weightings $\begin{array} { r } { \sum _ { i } \delta w _ { i } f _ { i } ( x ) } \end{array}$ of the frozen members, and in the embedding the resulting uncertainty is negligible compared to the discrepancy between the ensemble and the data. With an uncertainty this small, the profiled weights can barely move, the composite null efectively reduces to the point-null, and the two statistics coincide: even an ensemble of $M = 1 2 8$ flows is rejected well before $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$ This is the limitation anticipated in Sec. B 2: re-weighting frozen flows can make up for the error of a simple target, but not for that of a more complex one.

## Appendix C: Implementation details

The nominal fit and every bootstrap refit use EM with a k-means initialization (specifically, the k-means++ initialization implemented in the GaussianMixture class in scikit-learn [78]). The scores of Eq. (6) can be computed analytically or by central finite diferences,

$$
\psi _ { a } ( x ) \approx \frac { \log p _ { \hat { \theta } + \epsilon e _ { a } } ( x ) - \log p _ { \hat { \theta } - \epsilon e _ { a } } ( x ) } { 2 \epsilon } ,\tag{C1}
$$

which costs $2 P$ evaluations of the log-density on each sample. The normalization of the tilted model for the alternative hypothesis, as well as the standardization of the features, are calculated using reference samples drawn from a pool of size $2 \times 1 0 ^ { 6 }$ from the best-fit SIGMA model. $1 0 N _ { \mathrm { t e s t } }$ samples are used to compute the normalization to ensure that its uncertainty is subdominant, and $1 0 ^ { 6 } = 1 0 N _ { \mathrm { t r a i n } }$ samples are used to compute the Fisher matrix. The feature standardization is performed with the whole pool.

The $J = 5 0 0$ kernel centers $c _ { j }$ are obtained by k-means clustering of a subsample of $R _ { 0 }$ . The width σ of the kernels is taken to be uniform and isotropic, with $\sigma = 0 . 4$ in the two-dimensional case study and $\sigma = 0 . 5 3$ in the four-dimensional case study.

The kernel amplitudes carry the ridge penalty $\lambda _ { \alpha } \| \alpha \| ^ { 2 }$ of Eq. (15) and are restricted to

$$
\left| \alpha _ { j } \right| \le b _ { j } = \frac { A } { \operatorname* { m a x } _ { R _ { 0 } } \left| \tilde { k } _ { j } \right| } ,\tag{C2}
$$

which bounds the log-density distortion of each kernel by A. We use $A = 1 0$ , i.e., local distortions within a factor $e ^ { \pm 1 0 }$ . Both regularizations are applied identically to the data and to every pseudo-experiment. $\lambda _ { \alpha }$ is taken to be 1 throughout.

The $\Lambda _ { \mathrm { m a x } }$ hyperparameter is taken to be 0.008 in all of the studies in the main body, and Fig. 14 shows the efects of changing $\Lambda _ { \mathrm { m a x } }$ in the 2D example with $K = 6 4$ As anticipated in the main body, the performance of the Gaussian-calibrated null is strongly dependent on the choice of $\Lambda _ { \mathrm { m a x } } .$ since its sampling from the null hypothesis is directly dependent on $\Lambda _ { \mathrm { m a x } } .$ , while the bootstrapcalibrated null is more robust, as $\Lambda _ { \mathrm { m a x } }$ only afects the definition of the test statistic and not the sampling of the null.

## Appendix D: Null distribution of the test statistic

We show that, to second order in $( v , \alpha )$ and for $\lambda _ { \alpha } =$ $0 , \ t _ { \mathrm { c o m p } }$ follows a $\chi _ { J } ^ { 2 }$ distribution under the Gaussian composite null. Let $\dot { f } = ( \Phi , \tilde { k } ) \in \mathbb { R } ^ { M + J }$ denote the full feature vector, which has zero mean under $p _ { \hat { \theta } }$ and, by Eq. (10), covariance

$$
\Sigma = \left( \begin{array} { c r c } { { \mathbb { I } _ { M } / N _ { \mathrm { t r a i n } } } } & { { C _ { v \alpha } } } \\ { { C _ { \alpha v } } } & { { C _ { \alpha \alpha } } } \end{array} \right) ,\tag{D1}
$$

![](images/278e1da5199ebb7b584a3cccb83bee1b8780ca85c8670343c329835c0c4c0baa.jpg)  
FIG. 10. Two-dimensional toy: fitted marginals of features $x _ { 1 }$ (top) and $x _ { 2 }$ (bottom) for ensembles of $M = 1 , 3 2$ , 128 flows (columns). Points are held-out data $( N = 1 0 ^ { 5 } )$ ; the $\mathrm { g r a y }$ line is the fitted ensemble density $\begin{array} { r } { \hat { g } = \sum _ { i } \hat { w } _ { i } f _ { i } } \end{array}$ , taken as the median over the ten ensemble choices (Sec. B 1); the blue band is the 68% weight-uncertainty $\hat { \boldsymbol g } \pm \sigma _ { \hat { \boldsymbol g } }$ of Eq. (B5), averaged over the ten choices and shown in the ratio-to-fit sub-panels. The single flow $( M = 1 )$ carries no fitted weights and hence no band. As M grows the fit tightens onto the target (most visibly the bimodal valley of $x _ { 1 } )$ and the band, narrow in the bulk, widens in the low-statistics tails.

with $\begin{array} { r } { C _ { \alpha \alpha } ~ = ~ \mathrm { C o v } [ \tilde { k } ] } \end{array}$ and $C _ { \alpha v } \ = \ C _ { v \alpha } ^ { \top } \ = \ \mathrm { C o v } [ \tilde { k } , \Phi ]$ Write $N ~ \equiv ~ N _ { \mathrm { t e s t } } , ~ a ~ \equiv ~ N _ { \mathrm { t e s t } } / N _ { \mathrm { t r a i n } } , ~ u ~ = ~ ( v , \alpha )$ , and $\begin{array} { r } { s = \sum _ { i } f ( x _ { i } ) = ( s _ { v } , s _ { \alpha } ) } \end{array}$ for the feature sums over the test sample. Expanding log $\begin{array} { r } { \widehat { Z } ( u ) = \frac { 1 } { 2 } u ^ { \top } \Sigma u + O ( \| u \| ^ { 3 } ) } \end{array}$ gives

$$
{ \ell } ( u ) \simeq u ^ { \top } s - \frac { 1 } { 2 } u ^ { \top } H u , H = \left( \begin{array} { c c } { ( 1 + a ) \mathbb { I } _ { M } } & { N C _ { v \alpha } } \\ { N C _ { \alpha v } } & { N C _ { \alpha \alpha } } \end{array} \right) ,\tag{D2}
$$

where the identity in the upper block comes from the Gaussian constraint. The numerator and denominator maximizations give $s ^ { \top } H ^ { - 1 } s / 2$ and $\| s _ { v } \| ^ { 2 } / 2 ( 1 + a )$ , respectively, and the block-inverse identity yields

$$
t _ { \mathrm { c o m p } } = s ^ { \top } H ^ { - 1 } s - \frac { \| s _ { v } \| ^ { 2 } } { 1 + a } = r ^ { \top } S ^ { - 1 } r ,\tag{D3}
$$

with

$$
r = s _ { \alpha } - \frac { N } { 1 + a } C _ { \alpha v } s _ { v } ,\tag{D4}
$$

$$
S = N C _ { \alpha \alpha } - \frac { N ^ { 2 } } { 1 + a } C _ { \alpha v } C _ { v \alpha } .\tag{D5}
$$

Here $r$ is the vector of kernel sums with their projection on the score sums removed, and S is the Schur complement of the v-block of H.

Under the Gaussian composite null, the test sample is drawn from $p _ { \hat { \theta } + W z } \simeq g _ { z }$ with $z \sim \mathcal { N } ( 0 , \mathbb { I } _ { M } )$ The tilt shifts the mean of f by $\Sigma . _ { v } z = ( z / N _ { \mathrm { t r a i n } } , C _ { \alpha v } z )$ , so that, including the spread of z,

$$
\mathrm { C o v } ( s _ { v } ) = a ( 1 + a ) \mathbb { I } _ { M } ,\tag{D6}
$$

$$
\mathrm { C o v } ( s _ { \alpha } , s _ { v } ) = { \cal N } ( 1 + a ) { \cal C } _ { \alpha v } ,\tag{D7}
$$

$$
\mathrm { C o v } ( s _ { \alpha } ) = N C _ { \alpha \alpha } + N ^ { 2 } C _ { \alpha v } C _ { v \alpha } .\tag{D8}
$$

Substituting gives $\mathrm { C o v } ( r ) = S .$ Since $r$ is asymptotically Gaussian, t<sub>comp</sub> $\sim \chi _ { J } ^ { 2 }$ , or $\chi ^ { 2 }$ with rank(S) degrees of freedom if the kernels are linearly dependent on $R _ { 0 }$ The result is independent of a and of the correlations $C _ { \alpha v }$ . The M standardized directions drop out because the constraint on v matches the distribution of z.

![](images/948580c1b0f9d75187a957233c3ad5d0ef68db83e85f6e137b90cb4f06e40718.jpg)  
FIG. 11. Four-dimensional jet embedding: fitted marginals of $x _ { 1 } { - } x _ { 4 } \ \mathrm { ( r o w s ) }$ for $M = 1 , 3 2 ,$ , 128 flows (columns), constructed as in Fig. 10; held-out data $N = 1 . 9 \times \bar { 1 0 } ^ { 6 }$ . The fitted marginals deviate from the data around the sharp peaks, while the weight-uncertainty band remains negligible at every M, so the propagated uncertainty does not cover the discrepancy.

![](images/18fd42a57d09fbfb1c0149b0b3e96c5fdbb12fc97a8475b3bf1d4801a4e8c19f.jpg)  
FIG. 12. Closure test for the 2D toy problem with NF ensemble. Median test p-value versus $N _ { \mathrm { t e s t } }$ for ensembles of M = 32 (left) and $M = 1 2 8$ (right) flows, $N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ . Each panel shows the single-flow baseline (black), the point-null (gray), and the composite “Gaussian” null that propagates the weight uncertainty (blue). Error bars are the 95% confidence interval of the pooled median, from a hierarchical bootstrap over the ten ensemble choices and their null/test toys; the dotted line marks the nominal $p = 0 . 5$ and the dashed vertical $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$ . Only at $M = 1 2 8$ does the composite test stay near nominal up to and beyond $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } }$

![](images/104924e035cc7143bf36209bff0862e41b804eb69edc745bb4270bdb2b2749cb.jpg)  
FIG. 13. Closure test for the 4D particle jets problem with NF ensemble. Median test p-value versus $N _ { \mathrm { t e s t } }$ for $M = 3 2$ (left) and $M = 1 2 8$ (right) flows, constructed as in Fig. 12. At every M the composite (Gaussian) null tracks the point-null to within the pseudo-experiment scatter: propagating the weight uncertainty does not help, and all constructions reach the floor 1/(B + 1) by $N _ { \mathrm { t e s t } } = N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$

![](images/e66a835df5108643fb9b2f8e5dc6716ff9b66602a8e29b0285cfe3e5f6a9183f.jpg)

![](images/b6b619442f720980642f92f4f87337cdcba10b5a9954cde6d65d4ed39024bfbc.jpg)  
FIG. 14. $\Lambda _ { \mathbf { m a x } }$ scan for the 2D case study. Median test p-value versus $N _ { \mathrm { t e s t } }$ for $K = 6 4 ~ \mathrm { S I G M A }$ with the bootstrap (left) and Gaussian (right) calibration prescription. The bootstrap calibration is largely robust to $\Lambda _ { \mathrm { m a x } }$ , while the results for the Gaussian calibration vary strongly with $\Lambda _ { \mathrm { m a x } }$ . Each shade corresponds to a diferent value of $\Lambda _ { \mathrm { m a x } } .$ , and the point null is independent of this hyperparameter.

A ridge $\lambda _ { \alpha } > 0$ adds $2 \lambda _ { \alpha } \mathbb { I } _ { J }$ to the α-block of H, which leaves r unchanged and replaces S by $S + 2 \lambda _ { \alpha } \mathbb { I } _ { J }$ in the quadratic form. The statistic is then a weighted sum of J independent $\chi _ { 1 } ^ { 2 }$ variables with weights $\sigma _ { j } / ( \sigma _ { j } + 2 \lambda _ { \alpha } ) <$ 1, where $\sigma _ { j }$ are the eigenvalues of S. For $t _ { \mathrm { p o i n t } }$ , with $v \equiv 0$ and data drawn from $p _ { \hat { \theta } } \cdot$ the same argument gives $t _ { \mathrm { p o i n t } } = s _ { \alpha } ^ { \top } ( N C _ { \alpha \alpha } ) ^ { - 1 } s _ { \alpha } \sim \chi _ { J } ^ { 2 }$

Figure 15 examines the empirical null distributions for the four-dimensional particle-jets problem of Sec. V B with $K = 2 5 6$ and $N _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ , for the three null prescriptions of Sec. IV D and increasing test-set size. At $N _ { \mathrm { t e s t } } = 1 \mathrm { k \Omega }$ , all three empirical nulls agree with each other and with the observed statistic, and a $\chi _ { \hat { \nu } } ^ { 2 }$ fitted to each null returns $\hat { \nu } \approx 9 3$ , consistent with the predicted effective $\nu \approx 8 2$ and the presence of additional noise due to the finite sample used to estimate $Z .$ In the smallsample regime, the asymptotic approximation holds and the test cannot resolve the finite-sample uncertainty, and the three calibrations are interchangeable.

The tests separate as $N _ { \mathrm { t e s t } }$ grows. At $N _ { \mathrm { t e s t } } ~ = ~ 2 0 \mathrm { k \Omega }$ the two composite nulls still nearly reproduce the observed distribution (with bootstrap somewhat outperforming Gaussian), while the point-null distribution has already shifted below that of the observed test statistic.

At $N _ { \mathrm { t e s t } } = 1 0 ^ { 5 } = N _ { \mathrm { t r a i n } } ,$ all three null distributions fall below the observed test statistic distribution (consistent with the p-values in Fig. 9), but the point-null and Gaussian prescription fail dramatically while the bootstrap calibration remains fairly close to the observed distribution.

## Appendix E: Efects of kernel width

In this appendix, we study the efects of the choice of kernel width σ on the behavior of the goodness-of-fit test. As discussed in the main body, we find that choosing σ too small relative to the length scale of the data results in spurious calibration failure, establishing a tradeof where σ must be chosen large enough for the test to be reliable, and small enough for it to be sensitive.

In Fig. 16, we show results of the closure test for the two-dimensional example for $K = 1 6 , 3 2$ with all parameters chosen as in the main body, except $\sigma = 0 . 1$ instead of 0.4. As with the larger kernels, the point null is insufficient and is rejected for all but the smallest values of $N _ { \mathrm { t e s t } }$ ; the Gaussian null outperforms the point null, but is also miscalibrated for $N _ { \mathrm { t e s t } } \geq 1 0 \mathrm { k }$

On the other hand, the bootstrap-calibrated null deviates from closure in the opposite way: the median pvalue is greater than 0.5 at large $N _ { \mathrm { t e s t } }$ , indicating that the distribution of the test statistic is shifted rightward in calibration pseudo-experiments relative to the distribution in observed data. In other words, the test finds more discrepancy, on average, with the best-fit trained model in pseudo-experiments (where that model is guaranteed to be well-specified) than in the observed data (where it is not), so this deviation from calibration is clearly spurious.

![](images/bba4c7826fe05b1744d2b138022a9ae07bcd7b49b9a8f1192d7b3035f5d01a95.jpg)  
FIG. 15. Empirical distributions of the test statistics for the 4D particle jets problem, for a GMM with $K = 2 5 6$ fitted to $\bar { N } _ { \mathrm { t r a i n } } = 1 0 ^ { 5 }$ points. Columns: test-set size $N _ { \mathrm { t e s t } } = 1 0 0 0$ , 20 000, 100 000. Rows: composite statistic with bootstrap calibration (red), composite statistic with Gaussian calibration (blue), point-null statistic (gray). Filled histograms: empirical null from 2 500 pseudo-experiments for Gaussian and point-null and 1 000 pseudo-experiments for bootstrap; black outline: observed statistic on 1 000 test sets, 100 drawn from the held-out pool for each of 10 independently trained models; gray dotted: best-fit $\chi _ { \hat { \nu } } ^ { 2 }$

We ascribe this to an interplay between the structure of the SIGMA model, which is a GMM, and the test. When trained on a finite training set without strong regularization, GMMs generally overfit by placing narrow components in regions of feature space with disproportionately many events. This overfitting is not pathological in our context, as it is part of the variance we aim to capture with uncertainties (as opposed to model bias, which we cannot capture; strong regularization would induce such bias).

The fine-grained structure induced by overfitting is then generically present in our trained models but not in the data-generating process, so an alternative hypothesis comprised of small-length-scale Gaussian functions will systematically fit a dataset drawn from a trained SIGMA model better than it would observed data. As such, we do not recommend using SIGMA models in conjunction with a goodness-of-fit test sensitive to length scales sub stantially smaller than the natural length scales of the data, hence our choice of σ in the main body to be comparable to the data standard deviation.

![](images/ae420239e893e399ae1c9013ae8408d6903965bb01962cc619a9900656a25f76.jpg)  
FIG. 16. Closure test for the 2D toy problem with $\sigma = 0 . 1$ . As with Fig. $5 ,$ median null $p \textmd { - }$ value for the three goodnessof-fit tests of Sec. IV C versus test-set size $N _ { \mathrm { t e s t } }$ for $K = 1 6 , 3 2$ . Unlike Fig. $5 ,$ however, the width of the test kernels is taken to be $\sigma = 0 . 1$ , which is substantially finer than the natural scale of the data.

[1] V. Belis, P. Odagiu, and T. K. <sup>˚</sup>Arrestad, Machine learning for anomaly detection in particle physics, Reviews in Physics 12, 10.1016/j.revip.2024.100091 (2024).

[2] A. Butter, T. Plehn, S. Schumann, S. Badger, S. Caron, K. Cranmer, et al., Machine learning and LHC event generation, SciPost Phys. 14, 079 (2023), arXiv:2203.07460 [hep-ph].

[3] R. Di Sipio, M. Faucci Giannelli, S. Ketabchi Haghighat, and S. Palazzo, DijetGAN: A Generative-Adversarial Network Approach for the Simulation of QCD Dijet Events at the LHC, JHEP 2019 (08), 110, arXiv:1903.02433 [hep-ex].

[4] A. Hallin, J. Isaacson, G. Kasieczka, C. Krause, B. Nachman, T. Quadfasel, M. Schlafer, D. Shih, and M. Sommerhalder, Classifying anomalies through outer density estimation, Phys. Rev. D 106, 055006 (2022), arXiv:2109.00546 [hep-ph].

[5] J. A. Raine, S. Klein, D. Sengupta, and T. Golling, CUR-TAINs for your sliding window: Constructing unobserved regions by transforming adjacent intervals, Front. Big Data 6, 899345 (2023), arXiv:2203.09470 [hep-ph].

[6] R. Das, T. Finke, M. Hein, G. Kasieczka, M. Kr¨amer, A. M¨uck, and D. Shih, Accurate and robust methods for direct background estimation in resonant anomaly detection, Phys. Rev. D 111, 094041 (2025), arXiv:2411.00085 [hep-ph].

[7] C. Krause, B. Nachman, I. Pang, D. Shih, and Y. Zhu, Anomaly detection with flow-based fast calorimeter simulators, Phys. Rev. D 110, 035036 (2024), arXiv:2312.11618 [hep-ph].

[8] K. Cranmer, J. Brehmer, and G. Louppe, The frontier of simulation-based inference, Proceedings of the National Academy of Sciences 117, 10.1073/pnas.1912789117 (2020).

[9] J. Brehmer, G. Louppe, J. Pavez, and K. Cranmer, Mining gold from implicit models to improve likelihood-free inference, Proceedings of the National Academy of Sciences 117, 10.1073/pnas.1915980117 (2020).

[10] K. T. Matchev, A. Roman, and P. Shyamsundar, Uncertainties associated with GAN-generated datasets in high energy physics, SciPost Phys. 12, 104 (2022), arXiv:2002.06307 [hep-ph].

[11] G. Grosso, V. Mikuni, and L. Heinrich, Are We Ready for AI-Driven Discovery? AI Verification Before the Next Fundamental Physics Breakthrough, (2026), arXiv:2607.10039 [physics.data-an].

[12] R. T. D’Agnolo and A. Wulzer, Learning New Physics from a Machine, Phys. Rev. D 99, 015014 (2019), arXiv:1806.02350 [hep-ph].

[13] R. T. D’Agnolo, G. Grosso, M. Pierini, A. Wulzer, and M. Zanetti, Learning multivariate new physics, Eur. Phys. J. C 81, 89 (2021), arXiv:1912.12155 [hep-ph].

[14] G. Grosso, M. Letizia, M. Pierini, and A. Wulzer, Goodness of fit by Neyman–Pearson testing, SciPost Phys. 16, 123 (2024), arXiv:2305.14137 [hep-ph].

[15] R. T. D’Agnolo, G. Grosso, M. Pierini, A. Wulzer, and M. Zanetti, Learning new physics from an imperfect machine, Eur. Phys. J. C 82, 275 (2022), arXiv:2111.13633 [hep-ph].

[16] G. Cowan, K. Cranmer, E. Gross, and O. Vitells, Asymptotic formulae for likelihood-based tests of new physics,

Eur. Phys. J. C 71, 1554 (2011), [Erratum: Eur.Phys.J.C 73, 2501 (2013)], arXiv:1007.1727 [physics.data-an].

[17] M. Haußmann, R. Winterhalder, and M. Ubiali, Uncertainty in Physics and AI: Taxonomy, Quantification, and Validation, (2026), arXiv:2605.10378 [stat.ML].

[18] S. Jim´enez, M. J¨urgens, and W. Waegeman, Position: Epistemic uncertainty estimation methods are fundamentally incomplete, in Proceedings of the 43rd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 306 (PMLR, 2026) pp. 170691–170717, arXiv:2505.23506 [cs.LG].

[19] M. Frate, K. Cranmer, S. Kalia, A. Vandenberg-Rodes, and D. Whiteson, Modeling Smooth Backgrounds and Generic Localized Signals with Gaussian Processes, (2017), arXiv:1709.05681 [physics.data-an].

[20] M. Bellagente, M. Haußmann, M. Luchmann, and T. Plehn, Understanding event-generation networks via uncertainties, SciPost Physics 13, 003 (2022), arXiv:2104.04543 [hep-ph].

[21] A. Butter, S. Diefenbacher, T. Plehn, and L. Vogel, Know What You Don’t Flow, (2026), arXiv:2608.21509 [hepph].

[22] Y. Gal and Z. Ghahramani, Dropout as a Bayesian approximation: Representing model uncertainty in deep learning, in International Conference on Machine Learning (PMLR, 2016) pp. 1050–1059.

[23] M. Hasan, I. Hossain, A. Rahman, and S. Nahavandi, Controlled Dropout for Uncertainty Estimation, in 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (2023) pp. 973–980.

[24] S. Benevedes and J. Thaler, Frequentist uncertainties on neural density ratios with w f ensembles, Phys. Rev. D 112, 056024 (2025), arXiv:2506.00113 [hep-ph].

[25] G. Aad et al. (ATLAS), Search for new scalars via <sub>X → SH → b</sub>¯<sub>bb</sub>¯<sub>b in proton-proton collisions at</sub> √<sub>s = 13</sub> TeV with the ATLAS detector, (2026), arXiv:2607.18484 [hep-ex].

[26] A. Butter, S. Diefenbacher, G. Kasieczka, B. Nachman, and T. Plehn, GANplifying event samples, SciPost Phys. 10, 139 (2021), arXiv:2008.06545 [hep-ph].

[27] S. Bieringer, A. Butter, S. Diefenbacher, E. Eren, F. Gaede, D. Hundhausen, G. Kasieczka, B. Nachman, T. Plehn, and M. Trabs, Calomplification — the power of generative calorimeter models, JINST 17 (09), P09028, arXiv:2202.07352 [hep-ph].

[28] S. Bieringer, S. Diefenbacher, G. Kasieczka, and M. Trabs, Calibrating Bayesian generative machine learning for Bayesiamplification, Mach. Learn. Sci. Tech. 5, 045044 (2024), arXiv:2408.00838 [cs.LG].

[29] H. Bahl, S. Diefenbacher, N. Elmer, T. Plehn, and J. Spinner, Forecasting generative amplification, SciPost Phys. 20, 150 (2026), arXiv:2509.08048 [hep-ph].

[30] S. J. Watts and L. Crow, An information theoretic limit to data amplification, Machine Learning: Science and Technology 6, 025046 (2025), arXiv:2412.18041 [stat.ML].

[31] C. Krause and D. Shih, Fast and accurate simulations of calorimeter showers with normalizing flows, Phys. Rev. D 107, 113003 (2023), arXiv:2106.05285 [physics.ins-det].

[32] R. Kansal, A. Li, J. Duarte, N. Chernyavskaya, M. Pierini, B. Orzari, and T. Tomei, Evaluating generative models in high energy physics, Phys. Rev. D 107, 076017 (2023), arXiv:2211.10295 [hep-ex].

[33] R. Das, L. Favaro, T. Heimel, C. Krause, T. Plehn, and

D. Shih, How to understand limitations of generative networks, SciPost Phys. 16, 031 (2024), arXiv:2305.16774 [hep-ph].

[34] S. Grossi, M. Letizia, and R. Torre, Refereeing the referees: evaluating two-sample tests for validating generators in precision sciences, Mach. Learn. Sci. Tech. 6, 015052 (2025), arXiv:2409.16336 [stat.ML].

[35] S. Grossi, M. Letizia, and R. Torre, Comparing generative models with the new physics learning machine, Nucl. Phys. B 1024, 117349 (2026), arXiv:2508.02275 [stat.ML].

[36] S. Bressler, I. Savoray, and Y. Zurgil, Learning new physics from data: A symmetrized approach, Phys. Rev. D 110, 095004 (2024), arXiv:2401.09530 [hep-ex].

[37] P. Cappelli, G. Grosso, M. Letizia, H. Reyes-Gonz´alez, and M. Zanetti, Learning to validate generative models: a goodness-of-fit approach, Mach. Learn. Sci. Tech. 7, 045011 (2026), arXiv:2511.09118 [stat.ML].

[38] G. Grosso, S. S. R. Hindupur, T. Fel, S. Bright-Thonney, P. Harris, and D. Ba, Sparse, self-organizing ensembles of local kernels detect rare statistical anomalies, (2025), arXiv:2511.03095 [cs.LG].

[39] G. J. McLachlan, S. X. Lee, and S. I. Rathnayake, Finite mixture models, Ann. Rev. Stat. Appl. 6, 355 (2019).

[40] A. P. Dempster, N. M. Laird, and D. B. Rubin, Maximum likelihood from incomplete data via the EM algorithm, J. Roy. Statist. Soc. Ser. B 39, 1 (1977).

[41] G. J. McLachlan and T. Krishnan, The EM algorithm and extensions, 2nd ed., Wiley Series in Probability and Statistics (Wiley, Hoboken, NJ, 2008).

[42] T. A. Louis, Finding the observed information matrix when using the EM algorithm, J. Roy. Statist. Soc. Ser. B 44, 226 (1982).

[43] R. A. Redner and H. F. Walker, Mixture densities, maximum likelihood and the EM algorithm, SIAM Rev. 26, 195 (1984).

[44] G. McLachlan and D. Peel, Finite Mixture Models, Wiley Series in Probability and Statistics (John Wiley & Sons, New York, 2000).

[45] S. J. Yakowitz and J. D. Spragins, On the identifiability of finite mixtures, Ann. Math. Statist. 39, 209 (1968).

[46] F. J. Massey, Jr., The Kolmogorov–Smirnov test for goodness of fit, J. Am. Stat. Assoc. 46, 68 (1951).

[47] H. W. Lilliefors, On the Kolmogorov–Smirnov test for normality with mean and variance unknown, J. Am. Stat. Assoc. 62, 399 (1967).

[48] J. Durbin, Weak convergence of the sample distribution function when parameters are estimated, Ann. Statist. 1, 279 (1973).

[49] W. Stute, W. Gonz´alez Manteiga, and M. Presedo Quindimil, Bootstrap based goodnessof-fit-tests, Metrika 40, 243 (1993).

[50] G. J. Babu and C. R. Rao, Goodness-of-fit tests when parameters are estimated, Sankhy¯a 66, 63 (2004).

[51] C. Genest and B. R´emillard, Validity of the parametric bootstrap for goodness-of-fit testing in semiparametric models, Ann. Inst. H. Poincar´e Probab. Statist. 44, 1096 (2008).

[52] K. G. Hayes, M. L. Perl, and B. Efron, Application of the bootstrap statistical method to the tau-decay-mode problem, Phys. Rev. D 39, 274 (1989).

[53] G. Cowan, Statistical data analysis (Clarendon Press, Oxford, 1998).

[54] C. Bouveyron and C. Brunet-Saumard, Model-based

clustering of high-dimensional data: A review, Comput. Stat. Data Anal. 71, 52 (2014).

[55] I. Kobyzev, S. J. D. Prince, and M. A. Brubaker, Normalizing flows: an introduction and review of current methods, IEEE Trans. Pattern Anal. Mach. Intell. 43, 3964 (2021), arXiv:1908.09257 [stat.ML].

[56] S. B. Menary and D. D. Price, Learning to discover: expressive Gaussian mixture models for multi-dimensional simulation and parameter inference in the physical sciences, Mach. Learn. Sci. Tech. 3, 015021 (2022), arXiv:2108.11481 [physics.data-an].

[57] H. White, Maximum likelihood estimation of misspecified models, Econometrica 50, 1 (1982).

[58] P. Hall, The bootstrap and Edgeworth expansion, Springer Series in Statistics (Springer, New York, 1992).

[59] M. Letizia, G. Losapio, M. Rando, G. Grosso, A. Wulzer, M. Pierini, M. Zanetti, and L. Rosasco, Learning new physics eficiently with nonparametric methods, Eur. Phys. J. C 82, 879 (2022), arXiv:2204.02317 [hep-ph].

[60] G. Grosso and M. Letizia, Multiple testing for signalagnostic searches for new physics with machine learning, Eur. Phys. J. C 85, 4 (2025), arXiv:2408.12296 [hep-ph].

[61] B. M. Dillon, R. Mastandrea, and B. Nachman, Selfsupervised anomaly detection for new physics, Phys. Rev. D 106, 056005 (2022), arXiv:2205.10380 [hep-ph].

[62] P. Harris, J. Krupa, M. Kagan, B. Maier, and N. Woodward, Resimulation-based self-supervised learning for pretraining physics foundation models, Phys. Rev. D 111, 032010 (2025), arXiv:2403.07066 [hep-ph].

[63] G. Matos, E. Busch, K. R. Park, and J. Gonski, Semisupervised permutation invariant particle-level anomaly detection, JHEP 2025 (05), 116, arXiv:2408.17409 [hepph].

[64] K. Metzger, L. Xu, M. Sodini, T. K. <sup>˚</sup>Arrestad, K. Govorkova, G. Grosso, and P. Harris, Anomaly-preserving contrastive neural embeddings for end-to-end modelindependent searches at the LHC, Phys. Rev. D 112, 072011 (2025), arXiv:2502.15926 [hep-ex].

[65] S. Bright-Thonney, C. Reissel, G. Grosso, N. S. Woodward, K. Govorkova, A. Novak, S. E. Park, E. A. Moreno, and P. Harris, AutoSciDACT: Automated scientific discovery through contrastive embedding and hypothesis testing, in The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025) arXiv:2510.21935 [cs.LG].

[66] R. Li, B. Nachman, and D. Noll, Signal-Aware Contrastive Latent Spaces for Anomaly Detection, (2026), arXiv:2603.25794 [hep-ph].

[67] H. Qu, C. Li, and S. Qian, Particle Transformer for Jet Tagging, in Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 162 (PMLR, 2022) pp. 18281– 18292, arXiv:2202.03772 [hep-ph].

[68] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, Supervised contrastive learning, in Advances in Neural Information Processing Systems, Vol. 33 (2020) pp. 18661– 18673, arXiv:2004.11362 [cs.LG].

[69] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, A simple framework for contrastive learning of visual representations, in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119 (PMLR, 2020) pp. 1597– 1607, arXiv:2002.05709 [cs.LG].

[70] C. Reissel, G. Badaracco, S. Benevedes, G. Grosso, and P. Harris, 4d supcon jetclass embedding (qcd class), 10.5281/zenodo.23693193 (2026).

[71] A. Immer, M. Korzepa, and M. Bauer, Improving predictions of Bayesian neural nets via local linearization, in Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, Vol. 130 (PMLR, 2021) pp. 703–711, arXiv:2008.08400 [stat.ML].

[72] E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer, and P. Hennig, Laplace Redux – Efortless Bayesian Deep Learning, in Advances in Neural Information Processing Systems, Vol. 34 (2021) pp. 20089–20103, arXiv:2106.14806 [cs.LG].

[73] S. Tong, P. Harris, and G. Grosso, Covariant Contrastive Learning for Uncertainty-Aware Anomaly Detection, (2026), arXiv:2609.35755 [hep-ph].

[74] G. Badaracco, S. Benevedes, G. Grosso, C. Reissel, T. K. <sup>˚</sup>Arrestad, and P. Harris, An end-to-end pipeline

for uncertainty-aware validation of generative AI, in Machine Learning and the Physical Sciences Workshop @ NeurIPS 2025 (2025).

[75] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, nflows: normalizing flows in PyTorch, Zenodo, version v0.14 (2020).

[76] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, in Proceedings of the 3rd International Conference on Learning Representations (ICLR) (2015) arXiv:1412.6980 [cs.LG].

[77] P. J. Huber, Robust estimation of a location parameter, Annals of Mathematical Statistics 35, 10.1214/aoms/1177703732 (1964).

[78] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, Scikit-learn: Machine learning in Python, Journal of Machine Learning Research 12, 2825 (2011).