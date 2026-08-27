# Fast rates in Bayesian online learning with approximate posteriors

Ilsang Ohn

Department of Statistics, Inha University

## Abstract

Exact Bayes prediction enjoys fast predictive regret guarantees, but exact posterior updating or representation may be too costly for online use. We study when these statistical guarantees are preserved by computational approximations. We show that the cumulative price of posterior approximation can be governed by the interaction between the contraction radius of the exact Gibbs posterior and the Wasserstein distance between the approximate and exact posteriors. Our general theorem shows that whenever exact Bayes prediction achieves a fast regret bound, any approximate posterior method that tracks the exact posterior with suficient accuracy inherits the same fast regret, up to an additive term determined by the approximation error. Three online learning examples are developed. For linear models with strongly convex regularized losses, a projected Langevin algorithm yields an approximate posterior that achieves logarithmic regret. For an infinite-dimensional canonical exponential family sequence model over a Sobolev ellipsoid, a prior-preserving truncation method attains the minimax predictive regret rate with sublinear memory and constant update cost per observation. For random-design Gaussian process (GP) regression, a sparse variational posterior with inducing variables achieves the same predictive regret rate as the exact GP, but at substantially lower computational cost.

Keywords: Bayesian online learning; Gibbs posterior; Langevin algorithm; sparse variational Gaussian processes; exponential families; Sobolev ellipsoids

## 1 Introduction

Bayesian online prediction maintains a distribution over parameters and uses it as the mixing law for the next prediction: after t observations, the posterior Π induces the predictive mixture for round t + 1. Under logarithmic loss such as negative log-likelihood, this construction has an exceptional cumulative structure: the one-step predictive normalizers telescope into a marginal likelihood, so that cumulative predictive regret can be analyzed through a single integrated likelihood ratio rather than by controlling the prediction error separately at every round. In regular parametric models, this connection leads to the classical logarithmic redundancy and regret behavior of Bayes mixtures. The information-theoretic analysis of Clarke and Barron (1990) establishes the asymptotic redundancy of Bayesian mixtures in regular parametric families, while (Xie and Barron, 2000) develop closely related asymptotic minimax regret results for universal coding, gambling, and sequential prediction.

The same phenomenon has a natural interpretation from online learning. Under log loss, Bayesian updating is an instance of exponential weighting, and logarithmic loss is mixable, so a mixture predictor can compete sharply with a fixed expert or parameter chosen in hindsight (Vovk, 1990; Cesa-Bianchi and Lugosi, 2006). Kakade and Ng (2004) derive online bounds for Bayesian algorithms and show how Bayesian predictors can be analyzed as online learning procedures without stochastic assumptions on the outcome sequence. More generally, exponential-weights methods admit regret guarantees under a variety of mixable and exp-concave losses, with fast rates under suitable curvature or stochastic conditions (Hazan et al., 2007; van Erven et al., 2015; van der Hoeven et al., 2018; Jun and Ohn, 2026). The resulting viewpoint is broader than ordinary Bayes: posterior-like exponential weights can be formed from generic losses, and the same variational structure underlying Bayesian marginal likelihoods can be used to compare a mixture predictor with a fixed parameter or comparator distribution.

The above literature, however, primarily analyzes the statistical regret of the exact updating or weighting rule. The distribution used for prediction is assumed to be available at every round, and the computational error incurred in representing, sampling from, or optimizing that distribution is not priced separately. This distinction is immaterial when the posterior or exponential weights can be computed exactly, but becomes important in modern Bayesian models where each online update is itself approximate. Posterior approximation, which is statistically adequate at a fixed terminal sample size, need not be adequate for online prediction, because the approximation is queried repeatedly along the entire posterior path. Consequently, terminal contraction of an approximate posterior toward the truth does not by itself control the cumulative predictive loss. What is needed is a way to quantify how perturbations of the posterior at each round are translated into perturbations of the predictive mixture, and how this sensitivity changes as the exact posterior concentrates. Because of this challenge, to our knowledge, there is no general result that converts posterior approximation error into a cumulative predictive regret penalty while retaining the fast rates of exact Bayesian prediction. Aside from the specific setting of variational Bayes with a Gaussian variational family (Ch´erief-Abdellatif et al., 2019), the known results only provide slow regret bounds of the order $O ( \sqrt { T } )$ for horizon $T$ (e.g., Ch´erief-Abdellatif et al., 2019; Alquier, 2021).

This raises the question addressed in this paper: do the fast rates of Bayesian online prediction survive computation? Concretely, is there an implementable online learner, with explicit per-round operations, whose expected cumulative predictive regret is logarithmic in regular parametric models and minimax optimal in nonparametric ones?

We answer this question afirmatively. We establish a theoretical framework to derive fast predictive regret guarantees for general computational Bayesian online learning methods. Let $\Pi _ { t }$ be the exact Gibbs or standard posterior after t observations and let $Q _ { t }$ be an approximate posterior distribution attained by a computationally feasible algorithm. We derive an upper bound on the expected diference between the predictive regrets of Π<sub>t</sub> and $Q _ { t }$ , in terms of $\epsilon _ { t } ,$ the contraction rate of $\Pi _ { t }$ toward a common stochastic target, and $\alpha _ { t }$ , the Wasserstein distance between $Q _ { t }$ and $\Pi _ { t }$ The novelty of our result lies in showing that posterior approximation need not be controlled at a fixed first-order accuracy throughout the online path. We devise a second-order predictive stability condition, under which the efect of approximation error is automatically attenuated by posterior contraction, so the cumulative approximation cost scales with the interaction $\epsilon _ { t } \alpha _ { t } .$ , rather than with $\alpha _ { t }$ alone.

We instantiate this principle in three regimes. The first is a finite-dimensional generalized Bayesian model with predictable bounded covariates and a globally strongly convex regularized loss. Strong convexity gives $O ( t ^ { - 1 / 2 } )$ contraction of the exact Gibbs posterior, while local prior mass gives O(log T) expected regret for exact prediction. We approximate each constrained posterior with a projected Moreau–Yosida Langevin construction. We derive a squared Wasserstein tracking error of order $t ^ { - 1 }$ on the compact parameter space. This results in an ${ \cal O } ( \log T )$ cumulative approximation penalty, preserving the logarithmic rate of exact Gibbs prediction.

The second regime is an infinite-dimensional canonical exponential family sequence model. Here the computational constraint is representation rather than finite-dimensional sampling. The exact posterior is a countable product of one-dimensional conjugate posteriors. A streaming implementation may retain the suficient statistics of every coordinate observed so far, but its memory can grow linearly with the horizon. We instead update only the first m coordinates and preserve the original prior on the unresolved tail. This construction approximates one fixed infinitedimensional Bayesian model rather than replacing it with a sequence of truncated-prior models. Over a Sobolev ellipsoid, the contraction radius, exact Gibbs regret, and truncation error are all explicit. Choosing m at the efective-dimension scale yields the minimax cumulative predictive regret rate with sublinear memory and constant state-update work per observation.

The third regime is random-design Gaussian process (GP) regression. Here the statistical question is whether a sparse variational representation preserves the predictive performance of the full GP posterior. We fix the first J population covariance operator eigenfunctions as interdomain inducing variables. Under Gaussian noise, the resulting sparse variational posterior is represented by finite-dimensional suficient statistics and can be updated recursively. When J is greater than a certain sublinear threshold in T, its Wasserstein distance from the exact posterior is of the same order as the exact posterior contraction radius, and its cumulative predictive regret has the same nonparametric order as exact Bayes. This example also shows that vanishing variational KL divergence is not necessary for preserving statistical performance: the relevant discrepancy is its displacement relative to the shrinking posterior geometry.

The three applications make diferent computational choices. In the finite-dimensional example, approximation arises from incomplete sampling; in the sequence model, it arises from truncating an infinite representation; and in the GP example, it arises from variational sparse compression. Despite these diferent mechanisms, all three are analyzed through the same pair of quantities: the statistical radius of exact Bayes and the distance of the computational approximation from exact Bayes. The resulting framework makes the required computational accuracy explicit and exposes where model-specific work is unavoidable.

## 1.1 Related work

Streaming and approximate Bayesian inference A large algorithmic literature develops tractable approximations to sequential Bayesian updating. Examples include streaming variational Bayes (Broderick et al., 2013), sequential Monte Carlo (Del Moral et al., 2006), particle mirror descent (Dai et al., 2016), online variational inference with generalization guarantees (Ch´erief-Abdellatif et al., 2019), recursive optimization-based Bayesian rules (Khan and Rue, 2023; Jones et al., 2024), online sampling from log-concave sequences (Lee et al., 2019), and particle-based approximations of moving posteriors (Yang et al., 2023). Ch´erief-Abdellatif et al. (2019); Alquier (2021) establish generalization guarantees for several online variational procedures, providing one of the closest theoretical connections between approximate Bayesian computation and online learning. By contrast, (Yang et al., 2023) study Wasserstein tracking of a dynamically changing posterior through a distributional dynamic-regret analysis.

Recent work has also begun to characterize the frequentist accuracy of sequentially approximated posteriors themselves. Lee et al. (2026a) stablish an online Bernstein–von Mises theorem for sequential variational updating with mini-batches, showing that accumulated approximation error can become asymptotically negligible and that the resulting terminal posterior can be asymptotically equivalent to the full posterior, while Lee et al. (2026b) treat the more stringent one-pass regime and construct a warm-started Bayesian online procedure that attains an optimal convergence rate and a Bernstein–von Mises limit without diverging mini-batch sizes. These results concern the inferential fidelity of the sequentially constructed posterior, especially at the end of the data stream. In contrast, our criterion is prequential: approximation error is charged whenever the approximate posterior is used for prediction, and hence must be controlled along the entire posterior path rather than only at the terminal sample size.

Statistical theory for approximate posteriors A complementary literature studies whether computational approximations retain the frequentist concentration properties of exact Bayes. Variational and tempered posteriors have been shown to attain optimal or near-optimal contraction rates in a range of parametric, high-dimensional, and nonparametric models (Alquier et al., 2016; Alquier and Ridgway, 2020; Zhang and Gao, 2020; Ohn and Lin, 2024). Such results typically compare an approximate posterior directly with the true parameter or data-generating distribution. This comparison is not suficient for cumulative prediction: two posterior sequences can contract toward the same target at the same statistical rate while remaining separated enough to generate diferent predictive mixtures at every round. Our analysis therefore keeps two distinct quantities of the statistical radius of the exact posterior and the computational distance between the approximate and exact posteriors. This separation is what allows a posterior approximation guarantee to be converted into a fast predictive regret guarantee.

Constrained log-concave sampling Nonasymptotic sampling theory provides explicit distributional approximation guarantees for log-concave targets. For smooth log-concave distributions on Euclidean space, unadjusted Langevin algorithms have quantitative convergence guarantees Dalalyan (2017); Durmus and Moulines (2017). For compactly supported targets, projected Langevin Monte Carlo (Bubeck et al., 2018) and Moreau–Yosida regularized Langevin methods (Brosse et al., 2017) provide tractable schemes with total-variation or Wasserstein guarantees with explicit schedules, while proximal samplers achieve sharper complexity under alternative oracle assumptions (Lee et al., 2021). Online sampling from sequence of log-concave distributions has also been studied directly (Lee et al., 2019).

Computational-statistical tradeofs and low-rank posteriors. The principle that computation may be deliberately stopped or compressed once its error falls below statistical resolution appears in several batch problems. Examples include convex-relaxation hierarchies (Chandrasekaran and Jordan, 2013), Nystr¨om subsampling for kernel regression (Rudi et al., 2015), and early stopping for nonparametric regression (Raskutti et al., 2014), Low-rank approximation of Gaussian inverse problems similarly retains directions that are most informative relative to the prior (Spantini et al., 2015), while finite-dimensional computation in infinite-dimensional Bayesian models is often imposed through sieve or truncated priors (Zhao, 2000). Our results give a sequential Bayesian counterpart to this computational–statistical principle. In particular, in the sequence model approximation in Section 4, the Bayesian model remains infinite-dimensional, whereas the online algorithm updates only a finite number of coordinates, leaving the unresolved tail at its original prior.

Sparse variational Gaussian processes. Sparse variational GP regression represents a process posterior through finitely many inducing variables (Titsias, 2009; Matthews et al., 2016). Their approximation accuracy and the required growth of the inducing budget have been studied extensively. Burt et al. (2019, 2020) derive quantitative KL approximation guarantees and suficient inducing dimensions, while Nieman et al. (2022, 2023) establish posterior contraction and uncertainty quantification results for several inducing schemes, including population-spectral constructions. More recent work develops adaptive sparse variational procedures that retain minimax contraction behavior under hyperparameter selection (Nieman and Szab´o, 2025). On the algorithmic side, variational Fourier features and streaming sparse GP methods provide scalable spectral or sequentia representations (Hensman et al., 2018; Bui et al., 2017). Our concern is diferent from contraction of the sparse posterior around the truth. We compare the sparse and exact predictive mixtures over the entire online sequence and ask how large the inducing representation must be to preserve the cumulative predictive regret order of full GP Bayes.

## 1.2 Contributions and organization

The paper makes four contributions. First, Section 2 establishes a general predictive comparison theorem. It decomposes the expected regret of an approximate posterior into the regret of the exact Gibbs posterior and a posterior approximation penalty. Second, Section 3 proves a logarithmic expected regret for an implementable projected Moreau–Yosida Langevin online learner over globally log-concave linear-predictor Gibbs models. Third, Section 4 proves that a prior-preserving truncated Bayesian algorithm attains the minimax predictive regret rate for an exponential family sequence model with sublinear memory and constant-time updates. Fourth, Section 5 develops an online population-spectral sparse variational GP and proves that an efective-dimension representation preserves the nonparametric predictive regret order of exact Bayes. The analysis identifies posterior uncertainty, rather than a vanishing variational objective gap, as the statistical scale that determines the required inducing dimension. Section 6 concludes the paper and discusses limitations. Detailed proofs are collected in the appendices.

## 1.3 Notation

For a set $A ,$ let $\mathbf { 1 } _ { A } ( \cdot )$ denote the indicator function of A such that $\mathbf { 1 } _ { A } ( x ) = 1$ if $x \in A$ and $\mathbf { 1 } _ { A } ( x ) = 0$ otherwise. For a vector x, let $\lVert x \rVert$ denote its $L ^ { 2 }$ norm. We write diam $( A ) : = \operatorname* { s u p } _ { x , y \in A } \| x - y \|$ for a subset A of $\mathbb { R } ^ { d }$ . We write $a \lesssim b$ or $b \gtrsim a$ if there is a constant $C > 0$ , not relevant to the main argument, such that $a \leq C b$ . We write $a \asymp b$ if both $a \gtrsim b$ and $a \lesssim b$ hold. For two probability measures P and $Q , \mathrm { K L } ( P \| Q )$ denotes the Kullback–Leibler (KL) divergence defined as KL $\begin{array} { r } { ( P \| Q ) : = \int \log ( \mathrm { d } P / \mathrm { d } Q ) \mathrm { d } P } \end{array}$ if $P \ll Q$ , and $\mathrm { K L } ( P \| Q ) : = \infty$ otherwise.

## 2 Main results

## 2.1 Statistical setup

We work on a filtered probability space. At round $t \in \mathbb { N } ,$ , we observe a pair of the input or context variable $X _ { t }$ and the response $Y _ { t }$ . The sigma-field $\mathcal { G } _ { t - 1 }$ contains history before the round-t. In particular it may contain predictable side information such as the current covariate or context, as well as all computational randomness generated before round t. We allow the round t loss map

$$
( \theta , ( x , y ) ) \mapsto \ell _ { t } ( \theta , ( x , y ) )
$$

to be $\mathcal { G } _ { t - 1 } { \mathrm { - m e a s u r a b l e } }$ as a random function of $( \theta , ( x , y ) )$ . Thus fixed-design, predictable-design, independent non-identically distributed, and i.i.d. models are all special cases. We do not consider adversarial outcomes; the results below require a common stochastic target through conditional moment and curvature conditions that are verified in the applications.

Let $( \Theta , { \mathsf { d } } )$ be a Polish metric space. We fix a prior $\Pi _ { 0 }$ and inverse temperature $\eta > 0$ . The exact Gibbs posterior is

$$
\Pi _ { t } ( \mathrm { d } \theta ) = \frac { \mathrm { e } ^ { - \eta L _ { t } ( \theta ) } \Pi _ { 0 } ( \mathrm { d } \theta ) } { \int \mathrm { e } ^ { - \eta L _ { t } ( u ) } \Pi _ { 0 } ( \mathrm { d } u ) } ,\tag{1}
$$

where we write

$$
L _ { t } ( \theta ) : = \sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ; ( X _ { s } , Y _ { s } ) ) .
$$

For a probability measure $Q$ on Θ, define the round-t predictive loss by

$$
\mathsf { m } _ { t } ( Q ; ( x , y ) ) : = - \frac { 1 } { \eta } \log \int \mathrm { e } ^ { - \eta \ell _ { t } ( \theta ; ( x , y ) ) } Q ( \mathrm { d } \theta ) .\tag{2}
$$

When $\eta = 1$ and the loss function is the negative log density of $Y _ { t }$ conditional on $X _ { t }$ and $\mathcal { G } _ { t - 1 }$ , this is ordinary Bayesian predictive log loss. Let $\theta ^ { \star } \in \Theta$ be a fixed stochastic target. A computational procedure maintains a random probability measure $Q _ { t }$ that is measurable with respect to the information available after round t and before $t + 1$ . We assume throughout that every Gibbs normalizing constant and predictive loss integral displayed above is almost surely finite and strictly positive.

We set $Q _ { 0 } = \Pi _ { 0 }$ and define the predictive regret of a sequence of online distributions $( Q _ { t } ) _ { t = 0 } ^ { T - 1 } : =$ $( Q _ { 0 } , Q _ { 1 } , \dots , Q _ { T - 1 } )$ as

$$
\mathsf { R e g r e t } ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } ) : = \sum _ { t = 1 } ^ { T } [ \mathsf { m } _ { t } ( Q _ { t - 1 } ; ( X _ { t } , Y _ { t } ) ) - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) ]\tag{3}
$$

and denote its expectation as

$$
\Re ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } ) : = \mathbb { E } \left[ \mathsf { R e g r e t } ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } ) \right]
$$

Let $W _ { 2 }$ denote the quadratic Wasserstein distance induced by d, and define

$$
\begin{array} { r } { \varepsilon _ { t } ^ { 2 } : = \mathbb { E } [ W _ { 2 } ^ { 2 } ( \Pi _ { t } , \delta _ { \theta ^ { \star } } ) ] , \qquad \alpha _ { t } ^ { 2 } : = \mathbb { E } [ W _ { 2 } ^ { 2 } ( Q _ { t } , \Pi _ { t } ) ] . } \end{array}\tag{4}
$$

The two radii in the above display deliberately measure diferent objects. The statistical radius $\varepsilon _ { t }$ is a property of the exact posterior and the data-generating process. The computational radius $\alpha _ { t }$ is a property of the approximation algorithm relative to the exact update. Keeping them separate will allow us to reuse the same statistical analysis for computational procedures with diferent error mechanisms.

## 2.2 Regret of exact Gibbs posteriors

As we mentioned in the introduction, the exact Gibbs posterior admits a telescoping identity, which can lead to fast regret rates. To be concrete, let

$$
Z _ { t } ^ { \Pi } : = \int \mathrm { e } ^ { - \eta \sum _ { s = 1 } ^ { t } \ell _ { s } \left( \theta ; \left( X _ { s } , Y _ { s } \right) \right) } \Pi _ { 0 } ( \mathrm { d } \theta ) .
$$

Then the Gibbs recursion gives

$$
\int \mathrm { e } ^ { - \eta \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) } \Pi _ { t - 1 } ( \mathrm { d } \theta ) = \frac { Z _ { t } ^ { \Pi } } { Z _ { t - 1 } ^ { \Pi } } .
$$

Thus the predictive losses telescope as

$$
\sum _ { t = 1 } ^ { T } \mathbf { m } _ { t } ( \Pi _ { t - 1 } ; ( X _ { t } , Y _ { t } ) ) = - { \frac { 1 } { \eta } } \sum _ { t = 1 } ^ { T } \{ \log Z _ { t } ^ { \Pi } - \log Z _ { t - 1 } ^ { \Pi } \} = - { \frac { 1 } { \eta } } \log Z _ { T } ^ { \Pi } .
$$

Subtracting $\begin{array} { r l } { \sum _ { t } \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) } \end{array}$ yields

$$
\mathsf { R e g r e t } \left( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \right) = - \frac { 1 } { \eta } \log \int \exp \left[ - \eta \sum _ { t = 1 } ^ { T } \{ \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) \} \right] \Pi _ { 0 } ( \mathrm { d } \theta ) .\tag{5}
$$

To bound the expectation of the right-hand side, the following proposition can be used.

Proposition 2.1 (Gibbs regret bound). Define

$$
\mathfrak { B } _ { T } : = \operatorname* { i n f } _ { \rho \leqslant \mathrm { I I } _ { 0 } } \left[ \sum _ { t = 1 } ^ { T } \int \mathbb { E } [ \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) ] \rho ( \mathrm { d } \theta ) + \frac { 1 } { \eta } \mathrm { K L } ( \rho \| \Pi _ { 0 } ) \right] .\tag{6}
$$

Then $\Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \leq \mathfrak { B } _ { T }$

## 2.3 Main theorem

For a deterministic probability measure $Q$ on $\Theta ,$ define the conditional predictive risk as

$$
\mathcal { M } _ { t } ( Q ) : = \mathbb { E } [ \mathsf { m } _ { t } ( Q ; ( X _ { t } , Y _ { t } ) ) | \mathcal { G } _ { t - 1 } ] .\tag{7}
$$

For a $\mathcal { G } _ { t - 1 }$ -measurable random measure $Q , { \mathcal { M } } _ { t } ( Q )$ denotes the evaluation of this random functional at $Q .$ . Then the expected regret of $( Q _ { t } ) _ { t = 0 } ^ { T - 1 }$ can be expressed as

$$
\begin{array} { r l } & { \Re \big ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mathbb { E } \left[ \mathsf { m } _ { t } ( Q _ { t - 1 } ; ( X _ { t } , Y _ { t } ) ) - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) | \mathcal { G } _ { t - 1 } \right] \right] } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \mathcal { M } _ { t } ( Q _ { t - 1 } ) - \mathcal { M } _ { t } ( \delta _ { \theta ^ { \star } } ) \right] . } \end{array}
$$

By a similar argument, we also have

$$
\Re \big ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) - \Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) = \sum _ { t = 2 } ^ { T } \mathbb { E } \left[ \mathcal { M } _ { t } ( Q _ { t - 1 } ) - \mathcal { M } _ { t } ( \Pi _ { t - 1 } ) \right] ,\tag{8}
$$

where the comparator term $\boldsymbol { \mathcal { M } } _ { t } ( \boldsymbol { \delta \mathbf { \relax _ { \theta ^ { \star } } } } )$ cancels when the two regrets are subtracted.

Assumption 2.2 (Conditional second-order predictive stability). There is an absolute constant $L _ { \mathcal { M } } > 0$ such that

$$
\mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) \leq L _ { \mathcal { M } } \left\{ W _ { 2 } ( P , \delta _ { \theta ^ { \star } } ) W _ { 2 } ( P , Q ) + W _ { 2 } ^ { 2 } ( P , Q ) \right\}\tag{9}
$$

almost surely for every $t \in \mathbb { N }$ and for every pair of probability measures $P , Q$

Assumption 2.2 is a second-order continuity condition on the conditional predictive risk: both terms on the right-hand side of (9) are products. A first-order Lipschitz condition $\begin{array} { r } { \mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) \lesssim } \end{array}$ $W _ { 2 } ( P , Q )$ would be easier to verify but yields the cumulative penalty $\textstyle \sum _ { t = 1 } ^ { T - 1 } \alpha _ { t }$ , requiring increasingly accurate computation regardless of statistical concentration. Under second-order stability, however, the same perturbation becomes progressively less consequential as the exact posterior contracts. Thus posterior concentration plays a dual role: it improves statistical prediction and simultaneously reduces the sensitivity of prediction to computational approximation. The quadratic term $\alpha _ { t } ^ { 2 }$ is the residual cost of the approximation itself.

Theorem 2.3 (Predictive regret under posterior approximation). Under Assumption 2.2,

$$
\Re \big ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \leq \Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) + L _ { \mathcal { M } } \sum _ { t = 1 } ^ { T - 1 } \big \{ \varepsilon _ { t } \alpha _ { t } + \alpha _ { t } ^ { 2 } \big \} .\tag{10}
$$

## 2.4 Suficient conditions for second-order predictive stability

Assumption 2.2 is stated at the level of predictive distributions. The next result reduces it to conditions on the underlying loss. Smoothness controls the second-order interpolation remainder, the exponential envelope controls the change of measure induced by the predictive normalizer, and conditional stationarity removes a nonvanishing first-order term at $\theta ^ { \star }$ . Note that the condition is stated with random envelopes so that it also applies coordinatewise to the exponential family sequence model in Section 4.

Theorem 2.4 (Suficient conditions of Assumption 2.2). Let $\Theta \subset \mathbb { R } ^ { d }$ be convex and compact. Suppose that, for every t, almost surely, the map $\ell _ { t } ( \cdot ; ( x , y ) )$ is continuously diferentiable on a neighborhood of Θ for almost every conditional realization $\left( x , y \right) o f \left( X _ { t } , Y _ { t } \right)$ , and there are nonnegative G<sub>t</sub>-measurable envelopes $H _ { t } , G _ { t } , D _ { t }$ such that

$$
\| \nabla \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) - \nabla \ell _ { t } ( u ; ( X _ { t } , Y _ { t } ) ) \| \le H _ { t } \| \theta - u \| , \qquad \theta , u \in \Theta ,\tag{11}
$$

$$
\operatorname* { s u p } _ { \theta \in \Theta } \left\| \nabla \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) \right\| \le G _ { t } ,\tag{12}
$$

$$
\operatorname* { s u p } _ { \theta , u \in \Theta } \left| \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) - \ell _ { t } ( u ; ( X _ { t } , Y _ { t } ) ) \right| \leq D _ { t } .\tag{13}
$$

Assume that there exists an absolute constant $C _ { \mathrm { e n v } } > 0$ such that

$$
\begin{array} { r } { \mathbb { E } \left[ \mathbf { e } ^ { \eta D _ { t } } ( H _ { t } + \eta G _ { t } ^ { 2 } ) \ | \ \mathcal { G } _ { t - 1 } \right] \leq C _ { \mathrm { e n v } } \qquad a . s . \ f o r \ e v e r y \ t \in \mathbb { N } . } \end{array}\tag{14}
$$

Moreover, assume that there exists $\theta ^ { \star } \in \mathrm { i n t } ( \Theta )$ such that

$$
\mathbb { E } [ \nabla \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) \ | \ \mathcal { G } _ { t - 1 } ] = 0 \qquad a . s .\tag{15}
$$

Then Assumption 2.2 holds for all probability measures supported on $\Theta$ , with a deterministic constant depending only on $C _ { \mathrm { e n v } } , \eta .$ , and diam(Θ).

## 3 Moreau–Yosida Langevin online Bayes

In this section, we apply our general result to a regular parametric problem with a strongly convex loss function. It turns out that strong convexity yields $\varepsilon _ { t } \asymp t ^ { - 1 / 2 }$ . Theorem 2.3 then shows that any posterior approximation method with $\alpha _ { t } \lesssim t ^ { - 1 / 2 }$ has an approximation penalty of order log T. The sampling analysis below is used precisely to reach this threshold.

## 3.1 Linear Gibbs models

We consider the online data generating process with predictable bounded design.

Assumption 3.1 (Predictable design). At round t, a covariate $X _ { t } \in \mathbb { R } ^ { d }$ is $\mathcal { G } _ { t - 1 }$ -measurable and satisfies $\| X _ { t } \| \leq K _ { x }$ for some absolute constant $K _ { x } > 0$ almost surely.

We use the regularized loss

$$
\ell _ { \lambda , t } ( \theta ; Y _ { t } ) = \phi ( Y _ { t } , X _ { t } ^ { \top } \theta ) + \frac { \lambda } { 2 } \| \theta \| ^ { 2 } , \qquad \lambda > 0 .\tag{16}
$$

Note that the predictable covariate is part of the round-t loss function rather than the random covariate in the abstract setup of Section 2. We impose some regularity conditions on the loss function and the parameter space.

Assumption 3.2 (Loss and parameter space). The parameter space $\Theta$ is a compact convex set with nonempty interior, which satisfies $B ( 0 , r _ { \Theta } ) \subseteq \Theta \subseteq B ( 0 , R _ { \Theta } )$ for some fixed $0 < r _ { \Theta } \le R _ { \Theta } < \infty$ For every y, the function $u \mapsto \phi ( y , u )$ is convex and continuously diferentiable. There are absolute constants $B _ { 1 } > 0$ and $L _ { \phi } > 0$ such that

$$
| \partial _ { u } \phi ( y , u ) | \leq B _ { 1 } , \qquad | \partial _ { u } \phi ( y , u ) - \partial _ { u } \phi ( y , v ) | \leq L _ { \phi } | u - v |
$$

for all $u , v \in \mathbb { R }$

Example 3.3. We list several examples of a loss $\phi$ satisfying the above assumption.

(i) square loss $\phi ( y , u ) = ( y - u ) ^ { 2 }$ with bounded y and $u ;$

(ii) logistic loss $\phi ( y , u ) = \log ( 1 + \mathrm { e } ^ { - y u } )$ for $y \in \{ - 1 , 1 \}$ ;

(iii) Huber loss $\phi ( y , u ) = h _ { \delta } ( y - u )$ with

$$
h _ { \delta } ( z ) = \left\{ { z ^ { 2 } } / { 2 } \begin{array} { l l } { z ^ { 2 } / 2 } & { \mathrm { ~ f o r ~ } | z | \leq \delta , } \\ { \delta ( | z | - \delta / 2 ) } & { \mathrm { ~ o t h e r w i s e . } } \end{array} \right.
$$

(iv) smoothed hinge loss $\phi ( y , u ) = s _ { \delta } ( y u )$ for $y \in \{ - 1 , 1 \}$ with

$$
s _ { \delta } ( z ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { ~ f o r ~ } z > 1 } \\ { 1 - z - \delta / 2 } & { \mathrm { ~ f o r ~ } z < 1 - \delta } \\ { ( 1 - z ) ^ { 2 } / ( 2 \delta ) } & { \mathrm { ~ o t h e r w i s e . } } \end{array} \right.
$$

As we did in our general setup, we assume that there is a fixed target parameter in the data-generating process.

Assumption 3.4 (Conditional stationary point). There is a fixed $\theta ^ { \star } \in$ int(Θ) satisfying

$$
\mathbb { E } [ \nabla \ell _ { \lambda , t } ( \theta ^ { \star } ; Y _ { t } ) \mid \mathcal { G } _ { t - 1 } ] = 0 \qquad \mathrm { a . s . ~ f o r ~ e v e r y ~ } t .\tag{17}
$$

Under the assumptions we have made, the conditions of Theorem 2.4 are satisfied and thus we can obtain the next result.

Theorem 3.5. Under Assumptions 3.1, 3.2 and ${ 3 . 4 } ,$ Assumption 2.2 holds.

Due to Theorem 3.5, we can apply Theorem 2.3. We then establish upper bounds on the two quantities related only to the exact Gibbs posteriors. For this purpose, we additionally require that the prior distribution is log-concave.

Assumption 3.6 (Prior distribution). The prior has density

$$
\Pi _ { 0 } ( \mathrm { d } \theta ) \propto \mathrm { e } ^ { - V _ { 0 } ( \theta ) } { \mathbf { 1 } } _ { \Theta } ( \theta ) \mathrm { d } \theta ,
$$

where $V _ { 0 } : \mathbb { R } ^ { d } $ R is convex and continuously diferentiable with a globally Lipschitz gradient, and the prior density is bounded below on a neighborhood of $\theta ^ { \star }$

The exact Gibbs posterior is given by

$$
\Pi _ { t } ( \mathrm { d } \theta ) \propto \mathrm { e } ^ { - U _ { t } ( \theta ) } { \bf 1 } _ { \Theta } ( \theta ) \mathrm { d } \theta
$$

with the potential function

$$
U _ { t } ( \theta ) = V _ { 0 } ( \theta ) + \eta \sum _ { s = 1 } ^ { t } \ell _ { \lambda , s } ( \theta ; Y _ { s } ) .
$$

Theorem 3.7 (Statistical guarantees for the Gibbs posterior). Under Assumptions 3.1, 3.2, 3.4 and 3.6, the exact Gibbs posterior satisfies

$$
\mathbb { E } \left[ W _ { 2 } ^ { 2 } ( \Pi _ { t } , \delta _ { \theta ^ { \star } } ) \right] \lesssim t ^ { - 1 }\tag{18}
$$

and

$$
\Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim \log T + 1 .\tag{19}
$$

## 3.2 Moreau–Yosida Langevin sampling

For a computable approximating algorithm for the exact Gibbs posterior, we consider Moreau–Yosida unadjusted Langevin algorithm (MYULA) of Brosse et al. (2017). For a smoothing parameter $\varsigma > 0$ define the Moreau–Yosida regularization of the hard constraint,

$$
U _ { t , \varsigma } ( \theta ) = U _ { t } ( \theta ) + \frac { 1 } { 2 \varsigma } \| \theta - \mathrm { p r o j } _ { \Theta } ( \theta ) \| ^ { 2 } .\tag{20}
$$

A single Moreau–Yosida unadjusted Langevin step with a step size $\gamma > 0$ is

$$
\theta ^ { + } = \theta - \gamma \left[ \nabla U _ { t } ( \theta ) + \varsigma ^ { - 1 } \{ \theta - \mathrm { p r o j } _ { \Theta } ( \theta ) \} \right] + \sqrt { 2 \gamma } \xi , \qquad \xi \sim \mathtt { N } ( 0 , I _ { d } ) .\tag{21}
$$

The kernel $\boldsymbol { \mathcal { K } } _ { t , \varsigma , \gamma }$ of the Markov chain defined by (21) is given for $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ and $A \subseteq \mathbb { R } ^ { d }$ by

$$
\mathcal { K } _ { t , \varsigma , \gamma } ( \theta , A ) = ( 4 \pi \gamma ) ^ { - d / 2 } \int _ { A } \exp \left( - ( 4 \gamma ) ^ { - 1 } \| u - \theta + \gamma \nabla U _ { t , \varsigma } ( \theta ) \| ^ { 2 } \right) \mathrm { d } u .
$$

From Theorem 2 of Brosse et al. (2017), we have the following result on the approximation accuracy of MYULA.

Theorem 3.8 (Approximation accuracy of MYULA). Fix $t \geq 1$ and $\tilde { \alpha } \in ( 0 , 1 )$ . Under the same assumptions of Theorem 3.7, for every fixed initial point $\theta _ { 0 } \in \Theta$ , there exist $\varsigma _ { t } > 0 , \gamma _ { t } > 0$ , and $N _ { t } \in \mathbb { N }$ such that

$$
\left\| \delta _ { \theta _ { 0 } } \mathcal { K } _ { t , \varsigma _ { t } , \gamma _ { t } } ^ { N _ { t } } - \Pi _ { t } \right\| _ { \mathrm { T V } } \leq \tilde { \alpha }\tag{22}
$$

Remark 3.9. The chain in (21) can leave Θ. We therefore project its final state. This does not worsen total variation distance from $\Pi _ { t }$ , because $\Pi _ { t }$ is supported on Θ and the projection acts as the identity there.

Algorithm 1 (Online MYULA). Set $Q _ { 0 } = \Pi _ { 0 }$ and $\tilde { \alpha } _ { t } \asymp t ^ { - 1 }$ . After observing $Y _ { t } ,$ , form $U _ { t }$ . Choose $\left( \varsigma _ { t } , \gamma _ { t } , N _ { t } \right)$ satisfying the conclusion of Theorem 3.8 with accuracy $\tilde { \alpha } _ { t }$ . Initialize the chain at the fixed point $\theta _ { 0 } \in \Theta$ , run $N _ { t }$ transitions of (21), and project the final state onto Θ. Denote the law of that projected state by $Q _ { t }$ . A warm start from the preceding output may be used in practice, but is not needed for the stated guarantee.

Remark 3.10. Theorem 3.8 only requires the existence of a finite MYULA schedule attaining a prescribed total-variation accuracy. The explicit parameter choices and their quantitative complexity follow from the nonasymptotic bounds in Theorem 2, together with Propositions 4 and 6 of Brosse et al. (2017). In particular, when the geometry of Θ is fixed and the smoothness parameters of $U _ { t }$ grow at most polynomially in t, those bounds yield a choice of $N _ { t }$ that is polynomial in the relevant problem parameters, including t, d, and α˜. We do not reproduce the resulting constants, since only the existence of a computable schedule is needed for the regret analysis below.

Theorem 3.11 (Fast regret of online MYULA). Under the same assumptions of Theorem 3.7, the approximate posterior computed by Algorithm 1 satisfies

$$
\mathbb { E } \left[ W _ { 2 } ^ { 2 } ( Q _ { t } , \Pi _ { t } ) \right] \lesssim t ^ { - 1 } ,\tag{23}
$$

and thus

$$
\Re ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } ) \lesssim \log T + 1 .\tag{24}
$$

## 4 Truncated online posteriors for exponential family sequences

We now turn to a diferent computational regime. In Section 3.2, the model was finite-dimensional and the computational error was sampling error. Here the situation is reversed. Each coordinate posterior is available exactly in closed conjugate form, and the obstruction is representation: the posterior is a product of the coordinate posteriors, and its worst-case memory is $O ( T )$ . The approximation we consider here updates only the first m coordinates, thereby reducing the linear worst-case memory to a sublinear one.

## 4.1 Infinite-dimensional exponential family sequence models

For $\theta \in \mathbb { R }$ , let $\mathbb { P } _ { \theta }$ be an one-dimensional regular canonical exponential family distribution, which admits a density with respect to a dominating measure ν such that

$$
\mathsf { p } _ { \theta } ( y ) = \exp ( y \theta - A ( \theta ) ) .\tag{25}
$$

Let $X _ { t }$ be an i.i.d. positive integer-valued random variable such that $\Pr ( X _ { t } = j ) = p _ { j }$ for $j \in \mathbb N$ with $\begin{array} { r } { \sum _ { j = 1 } ^ { \infty } p _ { j } = 1 } \end{array}$ . Let $\theta ^ { \star } = ( \theta _ { j } ^ { \star } ) _ { j \geq 1 }$ be a true infinite-dimensional parameter. Conditional on $X _ { t } = j$ , we assume that $Y _ { t }$ follows a distribution $\mathbb { P } _ { \theta _ { j } ^ { \star } }$ . We impose the following smoothness conditions on the true data-generating process.

Assumption 4.1 (Sobolev sequence model). The probability vector $( p _ { j } ) _ { j \geq 1 }$ satisfies

$$
p _ { j } \asymp j ^ { - \bar { \alpha } }\tag{26}
$$

for some ${ \mathfrak { a } } > 1$ . The true parameter $\theta ^ { \star }$ belongs to the Sobolev ellipsoid

$$
\Theta _ { \mathfrak { s } } ( B , R ) = \left\{ ( \theta _ { j } ) _ { j \geq 1 } : | \theta _ { j } | < B , \quad \sum _ { j \geq 1 } j ^ { 2 \mathfrak { s } } \theta _ { j } ^ { 2 } \leq R ^ { 2 } \right\} .\tag{27}
$$

for some ${ \mathfrak { s } } > 0 , B > 0$ and $R > 0$ . The log partition function A is defined on the real line and satisfies

$$
0 < \underline { { I } } \le A ^ { \prime \prime } ( \theta ) \le \overline { { I } } < \infty\tag{28}
$$

for any $\theta \in [ - B , B ]$

We assume that we correctly specify the exponential family sequence model, which means that we consider a negative log-density loss given by

$$
\ell _ { t } ( \theta ; ( j , y ) ) = - \log { \mathsf { p } } _ { \theta _ { j } } ( y ) = A ( \theta _ { j } ) - y \theta _ { j }
$$

for $\theta = ( \theta _ { j } ) _ { j \geq 1 }$ , and the inverse temperature of $\eta = 1$ . Then the expected regret is represented as an expected KL risk. For a $\mathcal { G } _ { t - 1 }$ -measurable distribution $Q$ of $\theta ,$ let $\widehat { \mathbb { P } } _ { Q } ^ { \mathrm { p r e d } } ( \cdot | j )$ be the associated predictive distribution given $X _ { t } = j$ with density

$$
\widehat { \mathsf { p } } _ { Q } ^ { \mathrm { p r e d } } ( y | j ) = \int \mathsf { p } _ { \theta _ { j } } ( y ) Q ( \mathrm { d } \theta ) = \int \mathrm { e } ^ { - \ell _ { t } ( \theta ; ( j , y ) ) } Q ( \mathrm { d } \theta ) .
$$

Then the conditional expectation of the excess predictive risk is

$$
\begin{array} { r l } & { \mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( \delta _ { \theta ^ { \star } } ) = \mathbb { E } \left[ \log \frac { \mathrm { e } ^ { - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) } } { \int \mathrm { e } ^ { - \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) } Q ( \mathrm { d } \theta ) } \vert \mathcal { G } _ { t - 1 } \right] } \\ & { \quad \quad \quad \quad \quad = \displaystyle \sum _ { j \geq 1 } p _ { j } \int \log \frac { \mathsf { p } _ { \theta _ { j } ^ { \star } } ( y ) } { \widehat { \mathsf { p } } _ { Q } ^ { \mathrm { p r e d } } ( y | j ) } \mathbb { P } _ { \theta _ { j } ^ { \star } } ( \mathrm { d } y ) } \\ & { \quad \quad \quad \quad = \displaystyle \sum _ { j \geq 1 } p _ { j } \mathrm { K L } \big ( \mathbb { P } _ { \theta _ { j } ^ { \star } } \| \widehat { \mathbb { P } } _ { Q } ^ { \mathrm { p r e d } } ( \cdot \vert j ) \big ) , } \end{array}
$$

which is the expected KL divergence between the predictive distribution and the true conditional distribution.

We consider the prediction metric given by

$$
{ \sf d } _ { p } ^ { 2 } ( \theta , \theta ^ { \prime } ) = \sum _ { j \geq 1 } p _ { j } ( \theta _ { j } - \theta _ { j } ^ { \prime } ) ^ { 2 } ,\tag{29}
$$

and $W _ { 2 , p }$ denotes the associated Wasserstein distance. The next theorem shows that Assumption 2.2 is satisfied for this exponential family sequence model

Theorem 4.2. Under Assumption $4 . 1 ,$ there is an absolute constant $L _ { \mathcal { M } } > 0$ such that

$$
\mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) \leq L _ { \mathcal { M } } \left\{ W _ { 2 , p } ( P , \delta _ { \theta ^ { \star } } ) W _ { 2 , p } ( P , Q ) + W _ { 2 , p } ^ { 2 } ( P , Q ) \right\}
$$

almost surely for every $t \in \mathbb { N }$ and for any two product measures $P = \otimes _ { j \geq 1 } P _ { j }$ and $Q = \otimes _ { j \geq 1 } Q _ { j }$ on $\begin{array} { r } { \prod _ { j \geq 1 } [ - B , B ] } \end{array}$

## 4.2 Conjugate prior and posterior

We consider a Diaconis–Ylvisaker-form conjugate prior (Diaconis and Ylvisaker, 1979) but here we restrict it to the fixed common interval $[ - B , B ]$ . Note that because this indicator does not depend on the data, the coordinatewise posterior update remains in the same truncated family. Specifically, we consider the prior distribution $\Pi _ { 0 } = \otimes _ { j \geq 1 } \Pi _ { 0 , j }$ , where each coordinate prior is of the form

$$
\Pi _ { 0 , j } ( { \mathrm { d } } \theta ) \propto \exp ( \kappa _ { j } A ^ { \prime } ( 0 ) \theta - \kappa _ { j } A ( \theta ) ) \mathbf { 1 } _ { [ - B , B ] } ( \theta ) { \mathrm { d } } \theta\tag{30}
$$

for $\kappa _ { j } \geq 1$ . We assume the following oracle choice of the hyperparameter $\kappa _ { j }$ of the prior.

Assumption 4.3 (Diaconis–Ylvisaker conjugate prior). For every $j \in \mathbb { N } , \kappa _ { j } \asymp j ^ { 1 + 2 \mathfrak { s } }$

Due to conjugacy, the posterior distribution is given by $\Pi _ { t } = \otimes _ { j \geq 1 } \Pi _ { t , j }$ with

$$
\Pi _ { t , j } ( \mathrm { d } \theta ) \propto \exp \left( \{ \kappa _ { j } A ^ { \prime } ( 0 ) + { S } _ { t , j } \} \theta - \{ \kappa _ { j } + { N } _ { t , j } \} A ( \theta ) \right) \mathbf { 1 } _ { [ - B , B ] } ( \theta ) \mathrm { d } \theta .\tag{31}
$$

where we let

$$
N _ { t , j } = \sum _ { s = 1 } ^ { t } \mathbf { 1 } \{ X _ { s } = j \} , \qquad S _ { t , j } = \sum _ { s = 1 } ^ { t } Y _ { s } \mathbf { 1 } \{ X _ { s } = j \} .
$$

Example 4.4 (Beta–Bernoulli sequence model). Let $Y _ { t } | X _ { t } = j \sim$ Bernoulli $( q _ { j } )$ with $q _ { j } = \sigma ( \theta _ { j } )$ for $\theta _ { j } \in [ - B , B ]$ , where $\sigma ( \theta ) = ( 1 + \mathrm { e } ^ { - \theta } ) ^ { - 1 }$ denotes the logistic function. Then $A ( \theta ) = \log ( 1 + \mathrm { e } ^ { \theta } )$ and $A ^ { \prime \prime } ( \theta ) = \sigma ( \theta ) \{ 1 - \sigma ( \theta ) \}$ is bounded above and away from zero on $[ - B , B ]$ . So (28) is satisfied. Under the change of variable $q = \{ 1 + \mathrm { e } ^ { - \theta } \} ^ { - 1 }$ , the prior (30) is a Beta $( \kappa _ { j } / 2 , \kappa _ { j } / 2 )$ law truncated to $[ \sigma ( - B ) , \sigma ( B ) ]$ , and the posterior is the corresponding truncated Beta update.

To ease the notation, we set

$$
\zeta : = { \mathfrak { a } } + 2 { \mathfrak { s } } + 1\tag{32}
$$

throughout this section.

Theorem 4.5 (Statistical guarantees for the posterior). Under Assumptions $\it 4 . 1$ and $4 . 3 ,$ the exact posterior satisfies

$$
\mathbb { E } \left[ W _ { 2 , p } ^ { 2 } ( \Pi _ { t } , \delta _ { \theta ^ { \star } } ) \right] \lesssim t ^ { - ( \zeta - 1 ) / \zeta }\tag{33}
$$

and

$$
\begin{array} { r } { \mathfrak { R } \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim T ^ { 1 / \zeta } . } \end{array}\tag{34}
$$

## 4.3 Truncated posterior

Implementing the exact posterior requires storing suficient statistics and computing posterior distributions for every distinct coordinate observed so far. Under the assumed regime $p _ { j } \asymp j ^ { - \bar { \alpha } }$ the expected number of such coordinates is of order $T ^ { 1 / { \mathfrak { a } } }$ . To improve computational eficiency, here we propose an online approximate posterior that updates only the first m $\ll T ^ { 1 / { \mathfrak { a } } }$ coordinates (in our theoretical analysis, the oracle truncation level is of order $T ^ { 1 / ( { \mathfrak { a } } + 2 { \mathfrak { s } } + 1 ) } )$ , and preserves the original prior on the unresolved tail, which we call truncated posterior.

Algorithm 2 (Online truncated posterior). Fix a truncation level $m \in \mathbb { N } .$ . Store $( N _ { t , j } , S _ { t , j } )$ for $j \leq m$ . Before observing $Y _ { t } ,$ after seeing $X _ { t } = j$ , output the truncated posterior

$$
Q _ { t - 1 } ^ { ( m ) } = \bigotimes _ { j \leq m } \Pi _ { t - 1 , j } \otimes \bigotimes _ { j > m } \Pi _ { 0 , j }\tag{35}
$$

After observing $Y _ { t } .$ , update $( N _ { t , j } , S _ { t , j } )$ only ${ \mathrm { i f ~ } } j \leq m$

Algorithm 2 uses $O ( m )$ memory and $O ( 1 )$ state-update work per observation, aside from evaluation of a one-dimensional conjugate normalizer ratio.

Theorem 4.6 (Truncation error). Under the same assumptions of Theorem $4 . 5 ,$ the truncated posterior computed by Algorithm 2 satisfies

$$
\mathbb { E } \left[ W _ { 2 , p } ^ { 2 } ( Q _ { t } ^ { ( m ) } , \Pi _ { t } ) \right] \lesssim m ^ { 1 - \zeta } + t ^ { - ( \zeta - 1 ) / \zeta } .\tag{36}
$$

Theorem 4.7 (Fast regret of online truncated posterior). Under the same assumptions of Theorem $4 . 5 ,$ , the truncated posterior computed by Algorithm $\mathcal { Q }$ satisfies

$$
\Re \big ( ( Q _ { t } ^ { ( m ) } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim T ^ { 1 / \zeta } + T ^ { ( \zeta + 1 ) / ( 2 \zeta ) } m ^ { ( 1 - \zeta ) / 2 } + T m ^ { 1 - \zeta } .\tag{37}
$$

In particular, when $m \gtrsim T ^ { 1 / \zeta }$

$$
\begin{array} { r } { \mathfrak { R } \big ( ( Q _ { t } ^ { ( m ) } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim T ^ { 1 / \zeta } . } \end{array}\tag{38}
$$

It is worth emphasizing that the truncation here is computational rather than statistical. The last two rate terms in (37) monotonically decrease as the number of active coordinates m grows. Whenever the order of m is larger than $T ^ { 1 / \zeta }$ , the resulting regret is minimax optimal (see the next theorem). However, large m requires more memory. The choice $m \asymp T ^ { 1 / \zeta }$ balances the statistical rate and computational burden, reducing expected memory from $T ^ { 1 / { \mathfrak { a } } }$ to $T ^ { 1 / \zeta } = T ^ { 1 / ( { \mathfrak { a } } + 2 { \mathfrak { s } } + 1 ) }$

Theorem 4.8 (Minimax lower bound). $I f \widehat { p } _ { t } \big ( \cdot | X _ { t } , \mathcal { G } _ { t - 1 } \big )$ ranges over all sequential predictive densities that may use the revealed context $X _ { t }$ and past history $\mathcal { G } _ { t - 1 }$ , then

$$
\operatorname* { i n f } _ { ( \widehat { p } _ { t } ) } \operatorname* { s u p } _ { \theta ^ { \star } \in \Theta _ { \mathfrak { s } } ( B , R ) } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \log \frac { p _ { \star , X _ { t } } ( Y _ { t } ) } { \widehat { p } _ { t } ( Y _ { t } | X _ { t } , \mathcal { G } _ { t - 1 } ) } \right] \gtrsim T ^ { 1 / \zeta } .
$$

## 5 Online sparse variational Gaussian processes

The preceding sequence example reduces representation cost by coordinate truncation. We now consider a diferent posterior approximation approach based on a variational Bayes algorithm for online nonparametric regression.

We introduce additional notation used in this section. We write $\langle f , f ^ { \prime } \rangle : = \langle f , f ^ { \prime } \rangle _ { L ^ { 2 } ( \mathbb { P } _ { X } ) } : =$ $\begin{array} { r } { \int f ( x ) f ^ { \prime } ( x ) \mathrm { d } \mathbb { P } _ { X } ( x ) } \end{array}$ denote the inner product on $L ^ { 2 } ( \mathbb { P } _ { X } )$ , and $\| f \| : = \| f \| _ { L ^ { 2 } ( { \mathbb { P } } _ { X } ) } : = { \sqrt { \langle f , f \rangle } }$ denote the associated norm. Moreover, $W _ { 2 }$ and $\left\| \cdot \right\| _ { \mathrm { o p } }$ denote the Wasserstein distance and the operator norm, respectively, both induced by the norm $\| \cdot \|$

## 5.1 Gaussian process regression

Let $\mathbb { P } _ { X }$ be a probability measure on a measurable input space X . At round t, we observe $X _ { t } \overset { \mathrm { i i d } } { \sim } \mathbb { P } _ { X }$ and then observe

$$
Y _ { t } = f ^ { \star } ( X _ { t } ) + \xi _ { t } , \xi _ { t } \overset { \mathrm { i i d } } { \sim } \mathtt { N } ( 0 , \sigma _ { \xi } ^ { 2 } ) ,\tag{39}
$$

where the noise $\xi _ { t }$ is independent of $X _ { t }$ . To simplify the analysis, we assume the noise variance $\sigma _ { \xi } ^ { 2 }$ is known to us. We use a negative log-density loss given by

$$
\ell _ { t } ( f ; ( x , y ) ) = \frac { 1 } { 2 \sigma _ { \xi } ^ { 2 } } \{ y - f ( x ) \} ^ { 2 } + \frac { 1 } { 2 } \log ( 2 \pi \sigma _ { \xi } ^ { 2 } )\tag{40}
$$

and the inverse temperature of $\eta = 1$

We impose a centered Gaussian process (GP) prior Π<sub>0</sub> on f determined by its covariance kernel $\mathcal { K } _ { 0 } : \mathcal { X } \times \mathcal { X } \mapsto \mathbb { R }$ , which has a Mercer decomposition

$$
{ \sf K } _ { 0 } ( x , x ^ { \prime } ) = \sum _ { j \geq 1 } \lambda _ { j } \psi _ { j } ( x ) \psi _ { j } ( x ^ { \prime } ) ,
$$

where $( \psi _ { j } ) _ { j \geq 1 }$ is an orthonormal basis of $L _ { 2 } ( \mathbb { P } _ { X } )$ . This is equivalent to considering a random series prior $\begin{array} { r } { f = \sum _ { j \geq 1 } \sqrt { \lambda _ { j } } Z _ { j } \psi _ { j } } \end{array}$ with $Z _ { j } \stackrel { \mathrm { i i d } } { \sim } \mathtt { N } ( 0 , 1 )$ . The exact posterior is also a GP. Furthermore, the sparse GP posterior obtained using variational Bayes with inducing variables, which is our primary inference approach to be examined in the next subsection, is itself a GP.

For a $\mathrm { G P } \ Q$ , we let $m _ { Q } ( x )$ and $v _ { Q } ( x )$ denote the mean and variance of $f ( x )$ for $f \sim Q$ respectively. The associated predictive distribution $\widehat { \mathbb { P } } _ { Q } ^ { \mathrm { p r e d } } ( \cdot | x )$ given $x \in \mathcal { X }$ has a density

$$
\widehat { \mathsf { p } } _ { Q } ^ { \mathrm { p r e d } } ( y | x ) = \int \mathrm { e } ^ { - \ell _ { t } ( f ; ( x , y ) ) } Q ( \mathrm { d } f ) ,
$$

which is equal to the density of $\mathbb { N } ( m _ { Q } ( x ) , \sigma _ { \xi } ^ { 2 } + v _ { Q } ( x ) )$ . When $Q$ is $\mathcal { G } _ { t - 1 }$ -measurable, the excess conditional predictive risk is

$$
\mathcal M _ { t } ( Q ) - \mathcal M _ { t } ( \delta _ { f ^ { \star } } ) = \int \mathrm { K L } \big ( \mathtt { N } ( f ^ { \star } ( x ) , \sigma _ { \xi } ^ { 2 } ) \| \widehat { \mathbb { P } } _ { Q } ^ { \mathrm { p r e d } } ( \cdot | x ) \big ) \mathbb { P } _ { X } ( \mathrm { d } x ) ,
$$

which is the expected KL divergence between the predictive distribution and the true conditional distribution. For this Gaussian regression setup, we cannot apply Theorem 2.4 as the responses are unbounded. The next theorem overcomes this issue, which establishes second-order predictive stability with an additional remainder term.

Theorem 5.1 (Predictive stability for GPs). Let P and Q be Gaussian process laws on $L ^ { 2 } ( \mathbb { P } _ { X } )$ Then there exists an absolute constant $L _ { \mathcal { M } } > 0$ such that

$$
\mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) \leq L _ { { \mathcal M } } \left\{ W _ { 2 } ( P , \delta _ { f ^ { \star } } ) W _ { 2 } ( P , Q ) + W _ { 2 } ( P , Q ) ^ { 2 } + \overline { { v } } ( P , Q ) W _ { 2 } ( P , \delta _ { f ^ { \star } } ) ^ { 2 } \right\}\tag{41}
$$

with $\begin{array} { r } { \overline { { v } } ( P , Q ) : = \operatorname* { s u p } _ { x \in \mathcal { X } } \{ v _ { P } ( x ) + v _ { Q } ( x ) \} } \end{array}$

We assume that the basis functions are bounded and the true regression function belongs to a Sobolev space.

Assumption 5.2 (Sobolev regression function). There exist absolute constants $B _ { \psi } > 0 , { \mathfrak { s } } > d .$ and $R > 0$ such that the orthonormal basis $( \psi _ { j } ) _ { j \geq 1 }$ of ${ \mathcal { L } } ^ { 2 } ( \mathbb { P } _ { X } )$ satisfies

$$
\operatorname* { s u p } _ { j \geq 1 } \operatorname* { s u p } _ { x \in \mathcal { X } } | \psi _ { j } ( x ) | \leq B _ { \psi } < \infty ,\tag{42}
$$

and the true regression function $\begin{array} { r } { f ^ { \star } = \sum _ { j \geq 1 } f _ { j } ^ { \star } \psi _ { j } } \end{array}$ satisfies

$$
\sum _ { j \geq 1 } j ^ { 2 \Im / d } ( f _ { j } ^ { \star } ) ^ { 2 } \leq R ^ { 2 } .\tag{43}
$$

We further assume that the scale of the GP prior is suitably chosen according to the Sobolev smoothness s.

Assumption 5.3 (GP prior). For every $j \in \mathbb { N } , \lambda _ { j } \asymp j ^ { - 1 - 2 \mathfrak { s } / d }$

Under the assumptions we have made, we derive the fast regret bounds for the exact GP posterior.

Theorem 5.4 (Statistical guarantees for the exact GP posterior). Under Assumptions 5.2 and 5.3, the exact GP posterior satisfies

$$
\mathbb { E } \left[ W _ { 2 } ^ { 2 } ( \Pi _ { t } , \delta _ { f ^ { \star } } ) \right] \lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) }\tag{44}
$$

and

$$
\Re ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } ) \lesssim T ^ { d / ( d + 2 \mathfrak { s } ) } .\tag{45}
$$

## 5.2 Sparse variational Gaussian processes

In practice, computing the GP posterior exactly becomes impractical for large t, because the computational cost grows cubically in t. To overcome this issue, we propose to use a sparse variational GP approach of Titsias (2009), for online prediction. First, we choose the J-many inducing variables that are fixed for all $t \leq T$ . Define inducing variables as

$$
u _ { j } : = \langle f , \psi _ { j } \rangle , \qquad 1 \leq j \leq J ,\tag{46}
$$

so that $u _ { 1 : J } = ( u _ { 1 } , \ldots , u _ { J } ) ^ { \top } \sim \mathtt { N } ( 0 , \Lambda _ { J } )$ with $\Lambda _ { J } : = \operatorname { d i a g } ( \lambda _ { 1 } , . . . , \lambda _ { J } )$ . Following Titsias (2009), we consider a variational family $\mathcal { Q } _ { J }$ which is the set of laws of the form

$$
Q ( \mathrm { d } f , \mathrm { d } u ) = \Pi _ { 0 } ( \mathrm { d } f \mid u ) Q _ { u } ( \mathrm { d } u ) ,\tag{47}
$$

where $Q _ { u }$ is a Gaussian distribution on $\mathbb { R } ^ { J }$ . The sparse variational GP posterior is defined as

$$
Q _ { t } ^ { ( J ) } \in \operatorname * { a r g m i n } _ { Q \in \mathcal { Q } _ { J } } \mathrm { K L } ( Q \| \Pi _ { t } ) .\tag{48}
$$

The minimizer $Q _ { t } ^ { ( J ) }$ in (48) is the law of GP with mean and covariance functions given by (cf. see Titsias (2009); Nieman et al. (2023))

$$
\widehat { f } _ { t } ^ { ( J ) } ( x ) = \sigma _ { \xi } ^ { - 2 } \psi _ { 1 : J } ( x ) ^ { \top } V _ { t , J } \Psi _ { t , J } ^ { \top } Y _ { 1 : t } ,\tag{49}
$$

$$
\mathsf { K } _ { t } ^ { ( J ) } ( x , x ^ { \prime } ) = \mathsf { K } _ { 0 } ( x , x ^ { \prime } ) - \psi _ { 1 : J } ( x ) ^ { \top } \left( \Lambda _ { J } - V _ { t , J } \right) \psi _ { 1 : J } , ( x ^ { \prime } )\tag{50}
$$

where we write $Y _ { 1 : t } : = ( Y _ { 1 } , \ldots , Y _ { t } ) ^ { \top } , \psi _ { 1 : J } ( x ) : = ( \psi _ { 1 } ( x ) , \ldots , \psi _ { J } ( x ) ) ^ { \top }$ and

$$
\begin{array} { r } { V _ { t , J } : = ( \Lambda _ { J } ^ { - 1 } + \sigma _ { \xi } ^ { - 2 } \Psi _ { t , J } ^ { \top } \Psi _ { t , J } ) ^ { - 1 } , \quad \Psi _ { t , J } : = ( \psi _ { j } ( X _ { i } ) ) _ { i \leq t , j \leq J } . } \end{array}
$$

Let $\widetilde { \psi } _ { t + 1 } : = \psi _ { 1 : J } ( X _ { t + 1 } )$ . Then the predictive distribution for $Y _ { t + 1 }$ is given by

$$
\widehat { \mathbb { P } } _ { Q _ { t } ^ { ( J ) } } ^ { \mathrm { p r e d } } ( \cdot | X _ { t + 1 } ) = \mathbb { N } \left( \sigma _ { \xi } ^ { - 2 } \widetilde { \psi _ { t + 1 } ^ { \top } } V _ { t , J } \Psi _ { t , J } ^ { \top } Y _ { 1 : t } , \sigma _ { \xi } ^ { 2 } + \mathsf { K } ( X _ { t + 1 } , X _ { t + 1 } ) - \widetilde { \psi _ { t + 1 } ^ { \top } } \left( \Lambda _ { J } - V _ { t , J } \right) \widetilde { \psi } _ { t + 1 } \right) .\tag{51}
$$

Since $\Psi _ { t + 1 , J } ^ { \top } \Psi _ { t + 1 , J } ~ = ~ \Psi _ { t , J } ^ { \top } \Psi _ { t , J } + \widetilde { \psi } _ { t + 1 } \widetilde { \psi } _ { t + 1 } ^ { \top }$ , we can update $V _ { t + 1 , J }$ from $V _ { t , J }$ using Sherman–Morrison formula, as

$$
V _ { t + 1 , J } = V _ { t , J } - \frac { V _ { t , J } \widetilde { \psi } _ { t + 1 } \widetilde { \psi } _ { t + 1 } ^ { \top } V _ { t , J } } { \sigma _ { \xi } ^ { 2 } + \widetilde { \psi } _ { t + 1 } ^ { \top } V _ { t , J } \widetilde { \psi } _ { t + 1 } } ,\tag{52}
$$

which requires $O ( J ^ { 2 } )$ computation. Based on this elementary observation, we propose an online algorithm that produces the sparse GP posterior (48) exactly with $O ( J ^ { 2 } )$ computational cost at each round.

Algorithm 3 (Online population-spectral sparse variational GP). Fix J. For $t = 0 , 1 , \ldots , T$ we do:

(i) after observing $X _ { t + 1 }$ , predict $Y _ { t + 1 }$ using the predictive distribution $\widehat { \mathbb { P } } _ { Q _ { t } ^ { ( J ) } } ^ { \mathrm { p r e d } } ( \cdot | X _ { t + 1 } )$ in (51).

(ii) update $V _ { t + 1 , J }$ from $V _ { t , J }$ according to (52).

(iii) after observing $Y _ { t + 1 }$ , compute $\Psi _ { t + 1 , J } ^ { \top } Y _ { 1 : ( t + 1 ) } = \Psi _ { t , J } ^ { \top } Y _ { 1 : t } + \widetilde \psi _ { t + 1 } Y _ { t + 1 }$

Theorem 5.5 (Approximation accuracy of sparse variational GP). Under the same assumption of Theorem $5 . 4 ,$ the sparse variational GP computed by Algorithm 3 with $J \gtrsim T ^ { d / ( d + 2 5 ) }$ satisfies

$$
\begin{array} { r } { \mathbb { E } \big [ W _ { 2 } ^ { 2 } ( Q _ { t } ^ { ( J ) } , \Pi _ { t } ) \big ] \lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } . } \end{array}\tag{53}
$$

A notable feature of Theorem 5.5 is that the KL divergence between $Q _ { t } ^ { ( J ) }$ and $\Pi _ { t }$ itself need not vanish. Indeed, the available KL bound may grow with t, which is common in many applications (e.g., Zhang and Gao, 2020; Ohn and Lin, 2024; Nieman et al., 2022). What matters for online prediction is that this information discrepancy is converted into Wasserstein displacement through the shrinking operator norm of the exact posterior covariance:

$$
W _ { 2 } ^ { 2 } ( Q _ { t } ^ { ( J ) } , \Pi _ { t } ) \lesssim \lVert \Sigma _ { t } \rVert _ { \mathrm { o p } } \mathrm { K L } ( Q _ { t } ^ { ( J ) } \lVert \Pi _ { t } ) .
$$

with $\Sigma _ { t }$ denoting the covariance operator of the exact GP posterior. For the proof, see Lemma D.6 in the appendix. Thus the same variational discrepancy becomes less consequential as the exact posterior becomes geometrically more concentrated.

The exact and sparse posterior predictive variances are uniformly bounded; see Lemma D.9 in the appendix. Hence Theorem 5.1 yields the next result.

Theorem 5.6 (Fast regret of online sparse variational GP). Under the same assumption of Theorem $5 . 4 ,$ the sparse variational GP computed by Algorithm 3 with $J \gtrsim T ^ { d / ( d + 2 5 ) }$ satisfies

$$
\Re \big ( ( Q _ { t } ^ { ( J ) } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim T ^ { d / ( d + 2 \mathfrak { s } ) } .\tag{54}
$$

With the oracle number of inducing variables $J \asymp T ^ { d / ( d + 2 5 ) }$ , the total computational cost through horizon $T$ of the sparse variational GPs $( Q _ { t } ^ { ( J ) } ) _ { t = 0 } ^ { T - 1 }$ is

$$
O ( T J ^ { 2 } ) = O \big ( T ^ { ( 3 d + 2 \mathfrak { s } ) / ( d + 2 \mathfrak { s } ) } \big ) .\tag{55}
$$

By comparison, the exact GP posterior has $O ( t ^ { 2 } )$ update cost at round $t ,$ and thus total $O ( T ^ { 3 } )$ cost through horizon $T .$

## 6 Conclusion

This paper develops a comparison principle for Bayesian online prediction with approximate posteriors. The main theorem separates the expected regret of a computed predictor into an exact Gibbs benchmark and an approximation penalty governed by two scales: the contraction radius $\varepsilon _ { t }$ of the exact posterior and the tracking error $\alpha _ { t }$ of the computational approximation. The penalty $\textstyle \sum _ { t = 1 } ^ { T - 1 } ( \varepsilon _ { t } \alpha _ { t } + \alpha _ { t } ^ { 2 } )$ makes precise how statistical concentration reduces sensitivity to computation. Posterior approximation need not be uniformly negligible; its accuracy only needs to improve at a rate compatible with contraction.

The analysis suggests a design principle for computational Bayesian prediction: first identify the contraction scale and predictive geometry of the exact posterior, then allocate only enough computational accuracy that the induced predictive perturbations remainbelow the statistical resolution of exact Gibbs posterior. This turns posterior approximation from an external numerical concern into an explicit component of sequential regret analysis.

Several limitations mark useful directions for further work. The finite-dimensional result assumes compact support, smooth loss, and strong convexity supplied by regularization. Its cold-start mixing schedule is conservative, and warm starts or stochastic-gradient samplers may reduce its computational cost. The sequence result assumes known smoothness, a horizon-dependent rank, coordinatewise conjugacy, and tractable one-dimensional predictive normalizers. The sparse-GP result assumes known population spectral features, fixed kernel hyperparameters and inducing variables, polynomial spectral decay, and a known horizon. Extending the same direct comparison to fixed point-inducing schemes, moving inducing sets, adaptive ranks, or online hyperparameter learning requires new residual-kernel and sequential tracking arguments.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) funded by the Korea government (MSIT) (RS-2026-25486448) and INHA UNIVERSITY Research Grant.

## A Proofs for Section 2

## A.1 Proof of Theorem 2.3

Proof. Note that, at round $t , Q _ { t - 1 }$ and $\Pi _ { t - 1 }$ are $\mathcal { G } _ { t - 1 }$ -measurable and Assumption 2.2 holds almost surely conditional on $\mathcal { G } _ { t - 1 }$ . By the identity (8), Assumption 2.2 gives

$$
\Re \big ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) - \Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \le L _ { \ c } \sum _ { t = 2 } ^ { T } \mathbb { E } \left[ W _ { 2 } ( \Pi _ { t - 1 } , \delta _ { \theta ^ { \star } } ) W _ { 2 } ( Q _ { t - 1 } , \Pi _ { t - 1 } ) + W _ { 2 } ^ { 2 } ( Q _ { t - 1 } , \Pi _ { t - 1 } ) \right] .
$$

Therefore, by the Cauchy–Schwarz inequality, we have

$$
\begin{array} { r l } & { \Re \big ( ( Q _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) - \Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \le L _ { \mathcal { M } } \displaystyle \sum _ { t = 2 } ^ { T } \Big [ \mathbb { E } [ W _ { 2 } ^ { 2 } ( \Pi _ { t - 1 } , \delta _ { \theta ^ { * } } ] ] ^ { 1 / 2 } \{ \mathbb { E } [ W _ { 2 } ^ { 2 } ( Q _ { t - 1 } , \Pi _ { t - 1 } ) ] \} ^ { 1 / 2 } + \mathbb { E } [ W _ { 2 } ^ { 2 } ( Q _ { t - 1 } , \Pi _ { t - 1 } ) ] \Big ] } \\ & { \qquad = L _ { \mathcal { M } } \displaystyle \sum _ { t = 1 } ^ { T - 1 } ( \varepsilon _ { t } \alpha _ { t } + \alpha _ { t } ^ { 2 } ) , } \end{array}
$$

which gives the desired result.

## A.2 Proof of Proposition 2.1

Proof. By the Donsker-Varadhan variational formula, we have

$$
- \log \int \mathrm { e } ^ { - F ( \theta ) } \Pi _ { 0 } ( \mathrm { d } \theta ) \leq \int F ( \theta ) \mathrm { d } \rho ( \theta ) + \mathrm { K L } ( \rho \| \Pi _ { 0 } )
$$

for every $\rho \ll \Pi _ { 0 } . \mathrm { W e }$ then apply this inequality with

$$
F ( \theta ) = \eta \sum _ { t = 1 } ^ { T } \{ \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) \} .
$$

(5) identifies $\begin{array} { r } { - \eta ^ { - 1 } \log \int \mathrm { e } ^ { - F ( \theta ) } \Pi _ { 0 } ( \mathrm { d } \theta ) } \end{array}$ with ${ \mathsf { R e g r e t } } ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } )$ . Taking expectations and using Fubini under the stated integrability assumptions yields

$$
\Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \leq \sum _ { t = 1 } ^ { T } \int \mathbb { E } \{ \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) - \ell _ { t } ( \theta ^ { \star } ; ( X _ { t } , Y _ { t } ) ) \} \rho ( \mathrm { d } \theta ) + \frac { 1 } { \eta } \mathrm { K L } ( \rho \| \Pi _ { 0 } ) .
$$

Taking the infimum over deterministic $\rho$ proves the proposition.

## A.3 Proof of Theorem 2.4

Proof. Fix a round t. Let $Z _ { t } : = ( X _ { t } , Y _ { t } )$ In this proof, all expectations are conditional on $\mathcal { G } _ { t - 1 }$ , and we suppress this conditioning from the notation for notational brevity. The conditional stationarity condition (15) becomes $\mathbb { E } [ \nabla \ell _ { t } ( \theta ^ { \star } ; Z _ { t } ) ] = 0$ , while (14) supplies a deterministic upper bound $C _ { \mathrm { e n v } }$ for every conditional envelope moment used below.

Let $\gamma$ be an optimal coupling of $P$ and $Q _ { i }$ and write

$$
\theta _ { s } : = ( 1 - s ) \theta + s \theta ^ { \prime } , \qquad v = \theta ^ { \prime } - \theta ,
$$

for $( \theta , \theta ^ { \prime } ) \sim \gamma$ . Let $Q _ { s }$ be the law of $\theta _ { s }$ . For fixed $z : = ( x , y )$ , define

$$
f _ { z } ( s ) = - \frac { 1 } { \eta } \log \int \mathrm { e } ^ { - \eta \ell _ { t } ( \theta _ { s } ; z ) } \gamma ( \mathrm { d } \theta , \mathrm { d } \theta ^ { \prime } ) .
$$

Note that

$$
\mathcal { M } _ { t } ( Q ) = \mathbb { E } [ f _ { Z _ { t } } ( 1 ) \mid \mathcal { G } _ { t - 1 } ] , \quad \mathcal { M } _ { t } ( P ) = \mathbb { E } [ f _ { Z _ { t } } ( 0 ) \mid \mathcal { G } _ { t - 1 } ] ,
$$

Because $\nabla \ell _ { t } ( \cdot , z )$ is $H _ { t }$ -Lipschitz, $s \mapsto \ell _ { t } ( \theta _ { s } , z )$ has a Lipschitz derivative and is twice diferentiable for Lebesgue-a.e. s, with

$$
\left| \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } s ^ { 2 } } \ell _ { t } ( \theta _ { s } ; z ) \right| \le H _ { t } \| v \| ^ { 2 } .
$$

Let $G _ { s , z }$ be a probability measure on $\Theta \times \Theta$ defined as

$$
G _ { s , z } ( \mathrm { d } \theta , \mathrm { d } \theta ^ { \prime } ) = \frac { \mathrm { e } ^ { - \eta \ell _ { t } ( \theta _ { s } ; z ) } \gamma ( \mathrm { d } \theta , \mathrm { d } \theta ^ { \prime } ) } { \int \mathrm { e } ^ { - \eta \ell _ { t } ( u _ { s } ; z ) } \gamma ( \mathrm { d } u , \mathrm { d } u ^ { \prime } ) } , \quad u _ { s } : = ( 1 - s ) u + s u ^ { \prime } .
$$

Diferentiation of the log integral gives

$$
f _ { z } ^ { \prime } ( s ) = \frac { \int \frac { \mathrm { d } } { \mathrm { d } s } \ell _ { t } ( \theta _ { s } ; z ) \mathrm { e } ^ { - \eta \ell _ { t } ( \theta _ { s } ; z ) } \gamma ( \mathrm { d } \theta , \mathrm { d } \theta ^ { \prime } ) } { \int \mathrm { e } ^ { - \eta \ell _ { t } ( u _ { s } , z ) } \gamma ( \mathrm { d } u , \mathrm { d } u ^ { \prime } ) } = \mathbb { E } _ { G _ { s , z } } \left[ \frac { \mathrm { d } } { \mathrm { d } s } \ell _ { t } ( \theta _ { s } ; z ) \right]
$$

where $\mathbb { E } _ { G _ { s , z } }$ is expectation under the probability measure $\boldsymbol { G } _ { s , z }$ . Moreover, we have

$$
\begin{array} { l } { { \displaystyle f _ { z } ^ { \prime \prime } ( s ) = \mathbb { E } _ { G _ { s , z } } \left[ \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } s ^ { 2 } } \ell _ { t } ( \theta _ { s } ; z ) \right] - \eta \mathbb { E } _ { G _ { s , z } } \left[ \big ( \frac { \mathrm { d } } { \mathrm { d } s } \ell _ { t } ( \theta _ { s } ; z ) \big ) ^ { 2 } \right] + \eta \left( \mathbb { E } _ { G _ { s , z } } \left[ \frac { \mathrm { d } } { \mathrm { d } s } \ell _ { t } ( \theta _ { s } ; z ) \right] \right) ^ { 2 } } } \\ { { \displaystyle \quad \quad = \mathbb { E } _ { G _ { s , z } } \left[ \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } s ^ { 2 } } \ell _ { t } ( \theta _ { s } ; z ) \right] - \eta \mathrm { V a r } _ { G _ { s , z } } \left[ \frac { \mathrm { d } } { \mathrm { d } s } \ell _ { t } ( \theta _ { s } ; z ) \right] . } } \end{array}
$$

By assumption, we have

$$
\frac { \mathrm { e } ^ { - \eta \ell _ { t } ( \theta _ { s } ; z ) } } { \int \mathrm { e } ^ { - \eta \ell _ { t } ( u _ { s } ; z ) } \gamma ( \mathrm { d } u , \mathrm { d } u ^ { \prime } ) } = \frac { 1 } { \int \mathrm { e } ^ { \eta \{ \ell _ { t } ( \theta _ { s } ; z ) - \ell _ { t } ( u _ { s } ; z ) \} } \gamma ( \mathrm { d } u , \mathrm { d } u ^ { \prime } ) } \le \frac { 1 } { \mathrm { e } ^ { - \eta D _ { t } } }\tag{56}
$$

and therefore,

$$
f _ { z } ^ { \prime \prime } ( s ) \leq \mathbb { E } _ { G _ { s , z } } \left[ \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } s ^ { 2 } } \ell _ { t } ( \theta _ { s } ; z ) \right] \leq H _ { t } \mathrm { e } ^ { \eta D _ { t } } W _ { 2 } ^ { 2 } ( P , Q ) .
$$

Since $| f _ { z } ^ { \prime \prime } ( s ) | \lesssim \mathrm { e } ^ { \eta D _ { t } } ( H _ { t } + \eta G _ { t } ^ { 2 } ) D _ { \Theta } ^ { 2 }$ and this is conditionally integrable by (14), Fubini and the fundamental theorem for absolutely continuous functions therefore yield

$$
\mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) \leq \frac { \mathrm { d } } { \mathrm { d } s } \mathcal { M } _ { t } ( Q _ { s } ) \bigg \vert _ { s = 0 } + \frac { L _ { 1 } } { 2 } W _ { 2 } ^ { 2 } ( P , Q ) ,\tag{57}
$$

where $L _ { 1 } : = \mathbb { E } [ H _ { t } \mathrm { e } ^ { \eta D _ { t } } \mid \mathcal { G } _ { t - 1 } ] \leq C _ { \mathrm { e n v } }$

It remains to control the directional derivative. Define

$$
w _ { P } ( \theta , z ) = \frac { \mathrm { e } ^ { - \eta \ell _ { t } ( \theta ; z ) } } { \int \mathrm { e } ^ { - \eta \ell _ { t } ( u , z ) } P ( \mathrm { d } u ) } ,
$$

and

$$
b _ { P , t } ( \theta ) = \mathbb { E } [ w _ { P } ( \theta , Z _ { t } ) \nabla \ell _ { t } ( \theta ; Z _ { t } ) \mid \mathcal { G } _ { t - 1 } ] .
$$

Then we have

$$
\frac { \mathrm { d } } { \mathrm { d } s } \mathcal { M } _ { t } ( Q _ { s } ) \bigg | _ { s = 0 } = \int \langle b _ { P , t } ( \theta ) , \theta ^ { \prime } - \theta \rangle \gamma ( \mathrm { d } \theta , \mathrm { d } \theta ^ { \prime } ) .\tag{58}
$$

Let $g _ { \star , t } : = \nabla \ell _ { t } ( \theta ^ { \star } ; Z _ { t } )$ and $\begin{array} { r } { m _ { P } : = \int \| u - \theta ^ { \star } \| P ( \mathrm { d } u ) } \end{array}$ . The stationarity assumption gives $\mathbb { E } ( g _ { \star , t } \ |$ $\mathcal { G } _ { t - 1 } ) = 0$ , so

$$
b _ { P , t } ( \theta ) = \mathbb { E } \big [ w _ { P } ( \theta , Z _ { t } ) [ \nabla \ell _ { t } ( \theta ; Z _ { t } ) - g _ { \star , t } ] \mid { \mathcal G } _ { t - 1 } \big ] + \mathbb { E } \big [ \{ w _ { P } ( \theta , Z _ { t } ) - 1 \} g _ { \star , t } \mid { \mathcal G } _ { t - 1 } \big ] .
$$

Using a similar argument to (56), the first term is bounded by $\mathbb { E } ( \mathrm { e } ^ { \eta D _ { t } } H _ { t } \mid \mathcal { G } _ { t - 1 } ) \Vert \theta - \theta ^ { \star } \Vert$ . For the second, the mean-value theorem applied to $u \mapsto \mathrm { e } ^ { - \eta \ell _ { t } ( u ; z ) }$ gives

$$
\begin{array} { r l } & { | w _ { P } ( \theta , z ) - 1 | \leq \left| \frac { \int \exp { \eta _ { t } ( \theta , z ) } - \mathrm { e } ^ { - \eta \ell _ { t } ( u ; z ) } P ( \mathrm { d } u ) } { \int \mathrm { e } ^ { - \eta \ell _ { t } ( x ; z ) } P ( \mathrm { d } u ) } \right| } \\ & { \qquad \leq \left| \frac { \int \left. \operatorname* { s u p } _ { u ^ { \prime } \in \Theta } - \eta \mathrm { e } ^ { - \eta \ell _ { t } ( u ^ { \prime } ; z ) } \nabla \ell _ { t } ( u ^ { \prime } ; z ) , \theta - u \right. P ( \mathrm { d } u ) } { \int \mathrm { e } ^ { - \eta \ell _ { t } ( u ; z ) } P ( \mathrm { d } u ) } \right| } \\ & { \qquad \leq \left| \frac { \int \eta G _ { t } \| \theta - u \| P ( \mathrm { d } u ) } { \int \mathrm { e } ^ { \sin { \eta _ { t } } \omega } \mathrm { e } ^ { \ell _ { t } ( u ^ { \prime } ; z ) - \eta \ell _ { t } ( u ; z ) } P ( \mathrm { d } u ) } \right| } \\ & { \qquad \leq \eta \mathrm { e } ^ { \eta D } G _ { t } \{ \| \theta - \theta ^ { \star } \| + m P \} . } \end{array}
$$

Because $\| g _ { \star , t } \| \leq G _ { t }$ , by (14),

$$
\lVert b _ { P , t } ( \theta ) \rVert \leq C _ { \mathrm { e n v } } \{ \| \theta - \theta ^ { \star } \| + m _ { P } \} .
$$

a.s.. By Jensen’s inequality, we have $m _ { P } \leq W _ { 2 } ( P , \delta _ { \theta ^ { \star } } )$ , and hence

$$
\lVert b _ { P , t } \rVert _ { L ^ { 2 } ( P ) } \leq 4 C _ { \mathrm { e n v } } W _ { 2 } ( P , \delta _ { \theta ^ { \star } } ) .
$$

Combining this with (58) and using Cauchy–Schwarz under $\gamma ,$ we have

$$
\mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) \leq 4 C _ { \mathrm { e n v } } W _ { 2 } ( P , \delta _ { \theta ^ { \star } } ) W _ { 2 } ( P , Q ) + \frac { L _ { 1 } } { 2 } W _ { 2 } ^ { 2 } ( P , Q ) ,
$$

which completes the proof.

## B Proofs for Section 3

## B.1 Proof of Theorem 3.5

Proof. It is easy to see that the loss $\theta \mapsto \ell _ { \lambda , t } ( \theta ; Y _ { t } )$ satisfies (11), (12) and (13) with

$$
\begin{array} { r l } & { H _ { t } = L _ { \phi } K _ { x } ^ { 2 } + \lambda } \\ & { G _ { t } = B _ { 1 } K _ { x } + \lambda \dim ( \Theta ) } \\ & { D _ { t } = G _ { t } \dim ( \Theta ) , } \end{array}
$$

These constants are fixed in t, hence (14) is met. Assumption 3.4 is exactly (17). Thus Theorem 2.4 can be applied, which leads to Assumption 2.2. □

## B.2 Proof of Theorem 3.7

We first record a self-contained moment bound that will be used in both finite- and infinitedimensional applications.

Lemma B.1 (Second moment under a constrained strongly convex Gibbs law). Let $K \subset \mathbb { R } ^ { d }$ be a compact convex set with nonempty interior and let $\theta _ { 0 } \in \operatorname { i n t } ( K )$ . Suppose U is continuously diferentiable on a neighborhood of K and β-strongly convex on K. Then for a probability measure Πˇ $o n \mathbb { R } ^ { d }$ such that

$$
\check { \Pi } ( \mathrm { d } \theta ) = Z ^ { - 1 } \mathrm { e } ^ { - U ( \theta ) } \mathbf { 1 } _ { K } ( \theta ) \mathrm { d } \theta ,
$$

we have

$$
\int \| \theta - \theta _ { 0 } \| ^ { 2 } \check { \Pi } ( \mathrm { d } \theta ) \leq \frac { 2 \| \nabla U ( \theta _ { 0 } ) \| ^ { 2 } } { \beta ^ { 2 } } + \frac { 2 d } { \beta } .\tag{59}
$$

Proof. A compact convex body has Lipschitz boundary, so the divergence theorem applies. At surface-a.e. $x \in \partial K$ , the outer normal $n ( x )$ satisfies $\langle x - \theta _ { 0 } , n ( x ) \rangle \geq 0$ by convexity. Hence

$$
0 \leq \int _ { \partial K } \langle x - \theta _ { 0 } , n ( x ) \rangle \mathrm { e } ^ { - U ( x ) } \mathrm { d } S ( x ) = \int _ { K } \{ d - \langle x - \theta _ { 0 } , \nabla U ( x ) \rangle \} \mathrm { e } ^ { - U ( x ) } \mathrm { d } x .
$$

This implies that

$$
\mathbb { E } _ { \breve { \Pi } } [ \langle X - \theta _ { 0 } , \nabla U ( X ) \rangle ] \leq d .
$$

Strong convexity implies strong monotonicity of the gradient,

$$
\langle X - \theta _ { 0 } , \nabla U ( X ) - \nabla U ( \theta _ { 0 } ) \rangle \geq \beta \| X - \theta _ { 0 } \| ^ { 2 } .
$$

Let $g _ { 0 } : = \| \nabla U ( \theta _ { 0 } ) \|$ and $s ^ { 2 } : = \mathbb { E } _ { \breve { \Pi } } \| X - \theta _ { 0 } \| ^ { 2 }$ . Taking expectations and using Cauchy–Schwarz gives $\beta s ^ { 2 } \leq d + g _ { 0 } s$ . Solving this quadratic inequality yields $s \leq g _ { 0 } / \beta + \sqrt { d / \beta }$ , proving the desired result. □

Proof of Theorem 3.7. Let

$$
\Xi _ { s } : = \nabla \ell _ { \lambda , s } ( \theta ^ { \star } ; Y _ { s } ) .
$$

By (17), $\left( \Xi _ { s } \right)$ is a martingale-diference sequence with respect to the pre-prediction filtration. Assumptions 3.1 and 3.2 imply $\| \Xi _ { s } \| \le G _ { 0 }$ almost surely for some absolute constant $G _ { 0 } > 0$ . Due

to the $L ^ { 2 }$ regularization term $( \lambda / 2 ) \lVert \theta \rVert ^ { 2 }$ , the potential function is $\beta _ { t ^ { - } } \mathrm { s t r o n g l y }$ convex on $\Theta$ with $\beta _ { t } : = \eta \lambda t$ . The gradient of $\overline { { U } } _ { t }$ at the target is

$$
\nabla \overline { { U } } _ { t } ( \theta ^ { \star } ) = \nabla V _ { 0 } ( \theta ^ { \star } ) + \eta \sum _ { s = 1 } ^ { t } \Xi _ { s } .
$$

The martingale diferences are orthogonal in $L ^ { 2 } ;$ for $r < s .$

$$
\begin{array} { r } { \mathbb { E } [ \langle \Xi _ { r } , \Xi _ { s } \rangle ] = \mathbb { E } \left[ \langle \Xi _ { r } , \mathbb { E } ( \Xi _ { s } \mid \mathcal { G } _ { s - 1 } ) \rangle \right] = 0 . } \end{array}
$$

Consequently,

$$
\mathbb { E } \left\| \sum _ { s = 1 } ^ { t } \Xi _ { s } \right\| ^ { 2 } = \sum _ { s = 1 } ^ { t } \mathbb { E } [ \| \Xi _ { s } \| ^ { 2 } ] \leq G _ { 0 } ^ { 2 } t ,
$$

and hence

$$
\mathbb { E } \| \nabla \overline { { U } } _ { t } ( \theta ^ { \star } ) \| ^ { 2 } \lesssim t .
$$

By applying Lemma B.1 with $K = \Theta$ and $\theta _ { 0 } = \theta ^ { \star }$ , we have

$$
\mathbb { E } [ W _ { 2 } ^ { 2 } ( \Pi _ { t } , \delta _ { \theta ^ { \star } } ) ] = \mathbb { E } \left[ \int \| \theta - \theta ^ { \star } \| ^ { 2 } \Pi _ { t } ( \mathrm { d } \theta ) \right] \lesssim \frac { d } { t } ,
$$

which proves (18).

We now verify (19). Fix $c > 0$ suficiently small and let $r _ { T } = c T ^ { - 1 / 2 }$ . Because $\theta ^ { \star }$ is interior, $B ( \theta ^ { \star } , r _ { T } ) \subset \Theta$ for all suficiently large T. Let $\rho _ { T }$ be the uniform distribution on this ball. The prior density is bounded below by a positive constant on a fixed neighborhood of $\theta ^ { \star }$ , and hence

$$
\begin{array} { r } { \mathrm { K L } ( \rho _ { T } | | \Pi _ { 0 } ) \lesssim - \log \mathrm { V o l } ( B ( 0 , r _ { T } ) ) + 1 \lesssim \log T + 1 . } \end{array}
$$

Define the conditional risk

$$
R _ { \lambda , t } ( \theta ) = \mathbb { E } [ \ell _ { \lambda , t } ( \theta ; Y _ { t } ) \mid \mathcal { G } _ { t - 1 } ] .
$$

By assumption, this has L<sub>R</sub>-Lipschitz gradient with $L _ { R } : = L _ { \phi } K _ { x } ^ { 2 } + \lambda$ . Moreover, by $( 1 7 ) , \nabla R _ { \lambda , t } ( \theta ^ { \star } ) =$ 0. Therefore, we have

$$
R _ { \lambda , t } ( \theta ) - R _ { \lambda , t } ( \theta ^ { \star } ) \leq \frac { L _ { R } } { 2 } \| \theta - \theta ^ { \star } \| ^ { 2 }
$$

almost surely. Taking expectations and summing over t yields

$$
\sum _ { t = 1 } ^ { T } \int \mathbb { E } [ \ell _ { \lambda , t } ( \theta ; Y _ { t } ) - \ell _ { \lambda , t } ( \theta ^ { \star } ; Y _ { t } ) ] \rho _ { T } ( \mathrm { d } \theta ) \lesssim 1 ,
$$

where the finitely many remaining small values of $T$ are absorbed into the implicit constant. Proposition 2.1 proves the desired result. □

## B.3 Proof of Theorem 3.8

Proof. We apply Theorem 2 of Brosse et al. (2017) with $K = \Theta$ and $f = U _ { t }$ and so it sufices to check Assumptions H1 and H2 therein. Assumption 3.2 implies that $U _ { t }$ is convex and continuously diferentiable on $\mathbb { R } ^ { d }$ with $L _ { t } { \mathrm { - L i p s c h i t z } }$ gradient, where $L _ { t } \lesssim ( 1 + t )$ . These verify H1. The geometry assumption in H2 is already assumed in our Assumption 3.2. □

## B.4 Proof of Theorem 3.11

Proof. Let $\widetilde { Q } _ { t }$ be the law before the final projection in Algorithm 1. Theorem 3.8 gives

$$
\| \widetilde { Q } _ { t } - \Pi _ { t } \| _ { \mathrm { T V } } \leq \widetilde { \alpha } _ { t } .
$$

Because proj is measurable and $\mathrm { p r o j } _ { \Theta } \# \Pi _ { t } = \Pi _ { t }$ , total variation contracts under pushforward:

$$
\| Q _ { t } - \Pi _ { t } \| _ { \mathrm { T V } } = \| \operatorname { p r o j } _ { \Theta } \# \widetilde { Q } _ { t } - \operatorname { p r o j } _ { \Theta } \# \Pi _ { t } \| _ { \mathrm { T V } } \leq \tilde { \alpha } _ { t } .
$$

Both $Q _ { t }$ and $\Pi _ { t }$ are supported on Θ. Maximal coupling therefore yields

$$
W _ { 2 } ^ { 2 } ( Q _ { t } , \Pi _ { t } ) \leq \mathrm { d i a m } ( \Theta ) ^ { 2 } \lVert Q _ { t } - \Pi _ { t } \rVert _ { \mathrm { T V } } \lesssim t ^ { - 1 } .
$$

Taking expectations proves (23).

Theorem 3.7 gives $\varepsilon _ { t } \lesssim t ^ { - 1 / 2 }$ . Consequently,

$$
\sum _ { t \geq 1 } \{ \varepsilon _ { t } \alpha _ { t } + \alpha _ { t } ^ { 2 } \} \stackrel { < } { \sim } \sum _ { t \geq 1 } t ^ { - 1 } \stackrel { < } { \sim } \log T .
$$

Applying Theorem 2.3 with Theorem 3.7 completes the proof.

## C Proofs for Section 4

## C.1 Supporting lemmas

We first collect four elementary lemmas.

Lemma C.1 (Wasserstein distance for product measures). Let $( p _ { j } ) _ { j \geq 1 }$ satisfy $p _ { j } > 0$ for every $j$ and $\textstyle \sum _ { j } p _ { j } = 1$ . Let $\mu = \otimes _ { i > 1 } \mu _ { j }$ and $\nu = \otimes _ { i > 1 } \nu _ { j }$ be probability measures on the sequence space equipped with the cost $\begin{array} { r } { \mathsf { d } _ { p } ^ { 2 } ( x , \overset {  } { y } ) = \sum _ { j \ge 1 } p _ { j } ( x _ { j } - \overset {  } { y } _ { j } ) ^ { 2 } . ~ I f \sum _ { j } p _ { j } W _ { 2 } ^ { 2 } ( \mu _ { j } , \nu _ { j } ) < \infty } \end{array}$ , then

$$
W _ { 2 , p } ^ { 2 } ( \mu , \nu ) = \sum _ { j \geq 1 } p _ { j } W _ { 2 } ^ { 2 } ( \mu _ { j } , \nu _ { j } ) .\tag{60}
$$

Proof. Let Γ be any coupling of $\mu$ and $\nu ,$ and let $\Gamma _ { j }$ be its jth coordinate marginal. By Tonelli,

$$
\int \mathrm { d } _ { p } ^ { 2 } ( x , y ) \Gamma ( \mathrm { d } x , \mathrm { d } y ) = \sum _ { j \ge 1 } p _ { j } \int ( x _ { j } - y _ { j } ) ^ { 2 } \Gamma _ { j } ( \mathrm { d } x _ { j } , \mathrm { d } y _ { j } ) \ge \sum _ { j \ge 1 } p _ { j } W _ { 2 } ^ { 2 } ( \mu _ { j } , \nu _ { j } ) .
$$

Taking the infimum over Γ gives the lower bound. Conversely, for each $j ,$ choose an optimal quadraticcost coupling $\gamma _ { j }$ of $\mu _ { j }$ and $\nu _ { j } { \mathrm { : } }$ ; such a coupling exists on R whenever the displayed Wasserstein distance is finite. The countable product measure $\otimes _ { j } \gamma _ { j }$ is a coupling of $\mu$ and $\nu ,$ and Tonelli’s theorem gives its total cost as exactly $\begin{array} { r } { \sum _ { j } p _ { j } W _ { 2 } ^ { 2 } ( \mu _ { j } , \nu _ { j } ) } \end{array}$ . This proves the reverse inequality. □

Lemma C.2 (One-coordinate posterior risk). Under Assumption $4 . 1$ , there is an absolute constant $C > 0$ such that

$$
\mathbb { E } \left[ \int ( \theta - \theta _ { j } ^ { \star } ) ^ { 2 } \Pi _ { t , j } ( \mathrm { d } \theta ) \Bigg | N _ { t , j } \right] \leq C \left\{ \frac { 1 } { \kappa _ { j } + N _ { t , j } } + \frac { \kappa _ { j } ^ { 2 } ( \theta _ { j } ^ { \star } ) ^ { 2 } } { ( \kappa _ { j } + N _ { t , j } ) ^ { 2 } } \right\}\tag{61}
$$

for any $j \in \mathbb N$ and $t \in \mathbb { N }$

Proof. The potential function of the posterior $\Pi _ { t , j }$ on $[ - B , B ]$ is given by

$$
U _ { t , j } ( \theta ) = ( \kappa _ { j } + N _ { t , j } ) A ( \theta ) - \{ \kappa _ { j } A ^ { \prime } ( 0 ) + S _ { t , j } \} \theta ,
$$

which is β-strongly convex with $\beta = ( \kappa _ { j } + N _ { t , j } ) \underline { { I } } .$ . Since $| \theta _ { j } ^ { \star } | < B$ , we can apply Lemma B.1. Note that

$$
U _ { t , j } ^ { \prime } ( \theta _ { j } ^ { \star } ) = \kappa _ { j } \{ A ^ { \prime } ( \theta _ { j } ^ { \star } ) - A ^ { \prime } ( 0 ) \} - \{ S _ { t , j } - N _ { t , j } A ^ { \prime } ( \theta _ { j } ^ { \star } ) \} .
$$

By the well-known properties of an exponential family distribution, conditional on $N _ { t , j }$ , the mean and variance of the suficient statistics $S _ { t , j }$ is $N _ { t , j } A ^ { \prime } ( \theta _ { j } ^ { \star } )$ and $N _ { t , j } A ^ { \prime \prime } ( \theta _ { j } ^ { \star } )$ , respectively. Since $A ^ { \prime \prime } ( \theta _ { j } ^ { \star } ) \leq \overline { { I } }$ and $| A ^ { \prime } ( \theta _ { j } ^ { \star } ) ) - A ^ { \prime } ( 0 ) | \leq \overline { { I } } | \theta _ { j } ^ { \star } |$ , we have

$$
\mathbb { E } \left[ | U _ { t , j } ^ { \prime } ( \theta _ { j } ^ { \star } ) | ^ { 2 } | N _ { t , j } \right] \lesssim N _ { t , j } + \kappa _ { j } ^ { 2 } ( \theta _ { j } ^ { \star } ) ^ { 2 } .
$$

Taking expectation in (59) therefore gives the desired bound.

Lemma C.3 (Binomial resolvents). If $N \sim$ Binomia $. ( t , p )$ and $\kappa \geq 1$ , then

$$
\mathbb { E } \bigg [ \frac { 1 } { \kappa + N } \bigg ] \leq \frac { C } { \kappa + t p } , \qquad \mathbb { E } \bigg [ \frac { 1 } { ( \kappa + N ) ^ { 2 } } \bigg ] \leq \frac { C } { ( \kappa + t p ) ^ { 2 } }\tag{62}
$$

for some absolute constant $C > 0$

Proof. Let $\mu = t p$ . If $\mu < 2$ , then $\kappa + \mu \leq$ 3κ because $\kappa \geq 1$ , and the claim follows from

$$
( \kappa + N ) ^ { - q } \leq \kappa ^ { - q } \leq ( ( \kappa + \mu ) / 3 ) ^ { - q }
$$

for $q \in \{ 1 , 2 \}$ . Assume $\mu \geq 2$ and let $E = \{ N \geq \mu / 2 \}$ . On $E ,$

$$
( \kappa + N ) ^ { - q } \leq 2 ^ { q } ( \kappa + \mu ) ^ { - q } .
$$

For the complement, for any $s > 0$ , Markov’s inequality gives

$$
\begin{array} { r l } & { \mathrm { P r } ( N \leq \mu / 2 ) \leq \mathrm { e } ^ { s \mu / 2 } { \mathbb E } [ \mathrm { e } ^ { - s N } ] } \\ & { \qquad = \mathrm { e } ^ { s \mu / 2 } ( 1 - p + p \mathrm { e } ^ { - s } ) ^ { t } } \\ & { \qquad \leq \exp \{ \mu ( s / 2 + \mathrm { e } ^ { - s } - 1 ) \} . } \end{array}
$$

With s = log 2, this is at most $\mathrm { e } ^ { - c _ { 0 } \mu }$ , where $c _ { 0 } : = ( 1 - \log 2 ) / 2 > 0$ . Thus

$$
\begin{array} { r } { { \mathbb E } [ ( \kappa + N ) ^ { - q } \mathbf { 1 } _ { E ^ { c } } ] \le \kappa ^ { - q } \mathrm { e } ^ { - c _ { 0 } \mu } . } \end{array}
$$

Since $x ^ { q } \mathrm { e } ^ { - c _ { 0 } x }$ is bounded on $[ 0 , \infty )$ and $\kappa \geq 1 , \mathrm { e } ^ { - c _ { 0 } \mu } \lesssim \{ \kappa / ( \kappa + \mu ) \} ^ { q }$ . Combining the bounds on E and $E ^ { c }$ proves (62). □

Lemma C.4 (A localized comparator for a conjugate prior). Let Π be a distribution with $\Pi _ { \kappa } ( \mathrm { d } \theta ) \propto$ $\exp ( \kappa A ^ { \prime } ( 0 ) \theta - \kappa A ( \theta ) ) \mathbf { 1 } _ { [ - B , B ] } \mathrm { d } \theta$ with $\kappa \geq 1$ . Assume (28). For $| \theta ^ { \star } | < B$ and $n \in \mathbb { N }$ , there exists a probability measure ρ on $[ - \dot { B } , B ]$ such that

$$
\mathrm { K L } ( \rho \| \Pi ) \leq C \left\{ 1 + \log \left( 1 + \frac { n } { \kappa } \right) + \kappa ( \theta ^ { \star } ) ^ { 2 } \right\} ,\tag{63}
$$

$$
\int { \cal D } _ { A } ( \theta , \theta ^ { \star } ) \rho ( \mathrm { d } \theta ) \leq \frac { C } { \kappa + n } ,\tag{64}
$$

for some absolute constant $C > 0$ , where $D _ { A } ( \theta , \theta ^ { \prime } ) : = A ( \theta ) - A ( \theta ^ { \prime } ) - A ^ { \prime } ( \theta ^ { \prime } ) ( \theta - \theta ^ { \prime } )$

Proof. Let $I _ { \star } \subset [ - B , B ]$ be a small neighborhood of $\theta ^ { \star }$ . Put $v = c _ { 0 } / ( \kappa + n )$ with suficiently small $c _ { 0 } .$ , and let $\rho$ be the $\mathbb { N } ( \theta ^ { \star } , v )$ law conditioned on $I _ { \star }$ . Then

$$
\int ( \theta - \theta ^ { \star } ) ^ { 2 } \rho ( \mathrm { d } \theta ) \lesssim v .
$$

Moreover the conditional Gaussian density is bounded by $v ^ { - 1 / 2 }$ , up to a multiplicative constant. Therefore, the entropy of $\rho$ is bounded below as

$$
\begin{array} { r l } & { \displaystyle h ( \rho ) : = - \int \log ( \rho (  { \mathrm { d } } \theta ) /  { \mathrm { d } } \theta ) \rho (  { \mathrm { d } } \theta ) } \\ & { \qquad \geq - \log \| \rho (  { \mathrm { d } } \theta ) /  { \mathrm { d } } \theta \| _ { \infty } } \\ & { \qquad \geq - \displaystyle \frac { 1 } { 2 } \log ( v ) + C _ { 1 } ^ { \prime } } \\ & { \qquad \geq - \displaystyle \frac { 1 } { 2 } \log ( \kappa + n ) + C _ { 2 } ^ { \prime } } \end{array}
$$

for some absolute constants $C _ { 1 } ^ { \prime }$ and $C _ { 2 } ^ { \prime }$ . The prior can be written as

$$
\Pi ( \mathrm { d } \theta ) = Z _ { 0 } ( \kappa ) ^ { - 1 } \mathrm { e } ^ { - \kappa D _ { A } ( \theta , 0 ) } \mathbf { 1 } _ { [ - B , B ] } ( \theta ) \mathrm { d } \theta ,
$$

where $\begin{array} { r } { Z _ { 0 } ( \kappa ) = \int _ { - B } ^ { B } \mathrm { e } ^ { - \kappa D _ { A } ( u , 0 ) } \mathrm { d } u } \end{array}$ . The curvature bounds (28) on A imply

$$
\frac { I } { 2 } u ^ { 2 } \leq D _ { A } ( u , 0 ) \leq \frac { \overline { { I } } } { 2 } u ^ { 2 } , \qquad | u | \leq B .
$$

Integrating the lower quadratic bound gives log $Z _ { 0 } ( \kappa ) \leq C _ { 3 } ^ { \prime } - \textstyle { \frac { 1 } { 2 } }$ log κ for some absolute constants $C _ { 3 } ^ { \prime }$ . Also,

$$
\kappa \int D _ { A } ( \theta , 0 ) \rho ( \mathrm { d } \theta ) \lesssim \kappa \{ ( \theta ^ { \star } ) ^ { 2 } + v \} \lesssim \kappa ( \theta ^ { \star } ) ^ { 2 } + 1 .
$$

Therefore

$$
\begin{array} { l } { \displaystyle \mathrm { K L } ( \boldsymbol { \rho } \| \Pi ) = - h ( \boldsymbol { \rho } ) + \kappa \int D _ { { \cal A } } ( \boldsymbol { \theta } , 0 ) \boldsymbol { \rho } ( \mathrm { d } \boldsymbol { \theta } ) + \log Z _ { 0 } ( \kappa ) } \\ { \displaystyle \lesssim \left. 1 + \log \left( 1 + \frac { n } { \kappa } \right) + \kappa ( \boldsymbol { \theta } ^ { \star } ) ^ { 2 } \right. , } \end{array}
$$

which proves (63). Finally, $D _ { A } ( \theta , \theta ^ { \star } ) \leq ( \overline { { I } } / 2 ) ( \theta - \theta ^ { \star } ) ^ { 2 }$ , so

$$
\int { \cal D } _ { A } ( \theta , \theta ^ { \star } ) \rho ( \mathrm { d } \theta ) \lesssim v \lesssim \frac { 1 } { \kappa + n } ,
$$

proving (64).

## C.2 Proof of Theorem 4.2

Proof. For a $\mathcal { G } _ { t - 1 }$ -measurable distribution $Q ,$ since the data are i.i.d., we have

$$
\mathcal { M } _ { t } ( Q ) = \mathbb { E } \left[ - \log \int \mathrm { e } ^ { - \ell _ { t } ( \theta ; ( X _ { t } , Y _ { t } ) ) } Q ( \mathrm { d } \theta ) | \mathcal { G } _ { t - 1 } \right]
$$

$$
\begin{array} { r l } & { = \displaystyle \sum _ { j \geq 1 } p _ { j } \mathbb { E } \left[ - \log \int \mathrm { e } ^ { - \ell _ { t } ( \theta ; ( j , Y _ { t } ) ) } Q ( \mathrm { d } \theta ) \big | \mathcal { G } _ { t - 1 } , X _ { t } = j \right] } \\ & { = \displaystyle \sum _ { j \geq 1 } p _ { j } \widetilde { \mathcal { M } } _ { j } ( Q ) , } \end{array}
$$

where we define the per-coordinate conditional predictive risk

$$
\widetilde { \mathcal { M } } _ { j } ( Q ) : =  { \mathbb { E } } \left[ - \log \int \mathrm { e } ^ { - \ell _ { t } ( \theta ; ( j , Y _ { t } ) ) } Q ( \mathrm { d } \theta ) \big | \mathcal { G } _ { t - 1 } , X _ { t } = j \right] .
$$

Note that the gradient of $\ell _ { t } ( \theta ; ( j , y ) )$ with respect to the first argument is $A ^ { \prime } ( \theta _ { j } ) - y$ and its gradient-Lipschitz constant is at most I. Put $\begin{array} { r } { M _ { A } = \operatorname* { s u p } _ { | u | \leq B } | A ^ { \prime } ( u ) | < \infty } \end{array}$ . Then we can take the explicit envelopes

$$
\begin{array} { r l } & { H ( y ) = \overline { { I } } , } \\ & { G ( y ) = M _ { A } + | y | , } \\ & { D ( y ) = 2 B M _ { A } + 2 B | y | , } \end{array}
$$

where we use the mean-value theorem giving $| A ( \theta ) - A ( u ) | \leq M _ { A } | \theta - u |$ and $| \theta - u | \leq 2 B$ for the last line. The elementary bound $1 + y ^ { 2 } \leq \mathrm { e } ^ { | y | }$ implies

$$
\mathrm { e } ^ { D ( y ) } \{ H ( y ) + G ^ { 2 } ( y ) \} \lesssim \mathrm { e } ^ { ( 2 B + 1 ) | y | } .
$$

Since the log-partition function is assumed to exist on the whole real line, the right-hand side of the above display is integrable, which verifies (14). Hence, by Theorem 2.4, there is an absolute constant $C ^ { \prime } > 0$ such that

$$
\widetilde { \mathcal { M } } _ { j } ( Q _ { j } ) - \widetilde { \mathcal { M } } _ { j } ( P _ { j } ) \le C ^ { \prime } \{ W _ { 2 } ( P _ { j } , \delta _ { \theta _ { j } ^ { \star } } ) W _ { 2 } ( P _ { j } , Q _ { j } ) + W _ { 2 } ( P _ { j } , Q _ { j } ) ^ { 2 } \} ,
$$

for any $j \in \mathbb N .$ . In view of Lemma C.1, by multiplying by $p _ { j }$ , summing, and applying Cauchy–Schwarz, we have

$$
\begin{array} { r l } & { \mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) = \displaystyle \sum _ { j \geq 1 } p _ { j } \{ \widetilde { \mathcal { M } } _ { j } ( Q _ { j } ) - \widetilde { \mathcal { M } } _ { j } ( P _ { j } ) \} } \\ & { ~ \lesssim \left\{ W _ { 2 , p } ( P , \delta _ { \theta ^ { \star } } ) W _ { 2 , p } ( P , Q ) + W _ { 2 , p } ^ { 2 } ( P , Q ) \right\} , } \end{array}
$$

which completes the proof.

## C.3 Proof of Theorem 4.5

Proof. By Lemma C.2, followed by Lemma C.3, there exists an absolute constant $C _ { 1 } ^ { \prime } > 0$ such that

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \displaystyle \int ( \theta - \theta _ { j } ^ { \star } ) ^ { 2 } \Pi _ { t , j } ( \mathrm { d } \theta ) \right] \le C _ { 1 } ^ { \prime } \left\{ \frac { 1 } { \kappa _ { j } + t p _ { j } } + \frac { \kappa _ { j } ^ { 2 } ( \theta _ { j } ^ { \star } ) ^ { 2 } } { ( \kappa _ { j } + t p _ { j } ) ^ { 2 } } \right\} } \\ & { } & { = C _ { 1 } ^ { \prime } \left\{ \frac { q _ { j } } { 1 + t q _ { j } } + \frac { p _ { j } \theta _ { \star , j } ^ { 2 } } { ( 1 + t q _ { j } ) ^ { 2 } } \right\} } \end{array}\tag{65}
$$

with $q _ { j } : = p _ { j } / \kappa _ { j } \asymp j ^ { - \zeta }$ , for any $j \in \mathbb N$ and $t \in \mathbb { N } .$ . By Lemma C.1, squared Wasserstein distances for these product measures add coordinatewise under the cost (29). Hence,

$$
\varepsilon _ { t } ^ { 2 } = \mathbb { E } [ W _ { 2 , p } ^ { 2 } ( \Pi _ { t } , \delta _ { \theta ^ { \star } } ) ] \lesssim \sum _ { j \geq 1 } \left\{ \frac { q _ { j } } { 1 + t q _ { j } } + \frac { p _ { j } \theta _ { \star , j } ^ { 2 } } { ( 1 + t q _ { j } ) ^ { 2 } } \right\} .\tag{66}
$$

For $t \geq 1$ , let $j _ { t } : = \lceil t ^ { 1 / \zeta } \rceil$ . Splitting at $j _ { t }$ gives

$$
\sum _ { j \ge 1 } \frac { q _ { j } } { 1 + t q _ { j } } \lesssim \frac { j _ { t } } { t } + \sum _ { j > j _ { t } } j ^ { - \zeta } \lesssim t ^ { - ( \zeta - 1 ) / \zeta } .
$$

For the bias term, the Sobolev constraint yields

$$
\sum _ { j \geq 1 } \frac { p _ { j } \theta _ { \star , j } ^ { 2 } } { ( 1 + t q _ { j } ) ^ { 2 } } \leq R ^ { 2 } \operatorname* { s u p } _ { j \geq 1 } \frac { p _ { j } j ^ { - 2 \mathfrak { s } } } { ( 1 + t q _ { j } ) ^ { 2 } } .
$$

Because $p _ { j } j ^ { - 2  s } \asymp j ^ { - ( \zeta - 1 ) }$ and $q _ { j } \asymp j ^ { - \zeta }$ , the last supremum is bounded by $t ^ { - ( \zeta - 1 ) / \zeta }$ up to a multiplicative constant by considering separately $j \leq j _ { t }$ and $j > j _ { t }$ . This proves (33).

It remains to prove (34). Fix $m \in \mathbb { N } .$ For $j \leq m$ , use the comparator of Lemma C.4 with the deterministic scale $n = T p _ { j }$ and for $j > m$ , retain the prior. The product comparator difers from the prior in only finitely many coordinates, so its KL divergence is the sum of the coordinatewise KL divergences. Moreover

$$
\mathbb { E } [ \ell _ { t } ( \theta , Y _ { t } ) ] - \mathbb { E } [ \ell _ { t } ( \theta ^ { \star } , Y _ { t } ) ] = \sum _ { j \geq 1 } p _ { j } D _ { A } ( \theta _ { j } , \theta _ { \star , j } ) .
$$

For $j \ \leq \ m$ , Lemma C.4 gives a risk contribution $T p _ { j } / ( \kappa _ { j } + T p _ { j } ) \ \lesssim \ 1$ and KL contribution $\{ 1 + \log ( 1 + T q _ { j } ) + \kappa _ { j } \theta _ { \star , j } ^ { 2 } \}$ . For $j > m$ , the quadratic upper bound on $D _ { A }$ and Lemma B.1 applied to the prior at $\theta _ { 0 } = 0$ give

$$
\int { \cal D } _ { A } ( \theta , \theta _ { \star , j } ) \Pi _ { 0 , j } ( \mathrm { d } \theta ) \le C _ { 2 } ^ { \prime } \{ \kappa _ { j } ^ { - 1 } + \theta _ { \star , j } ^ { 2 } \} .
$$

for any $j \in \mathbb N$ for some absolute constant $C _ { 2 } ^ { \prime } > 0$ . Consequently, Proposition 2.1 yields

$$
\Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim m + \sum _ { j \leq m } \log ( 1 + T q _ { j } ) + \sum _ { j \leq m } \kappa _ { j } \theta _ { \star , j } ^ { 2 } + T \sum _ { j > m } p _ { j } \{ \theta _ { \star , j } ^ { 2 } + \kappa _ { j } ^ { - 1 } \} .
$$

Choose $m = \lceil T ^ { 1 / \zeta } \rceil$ . Since $T \leq C m ^ { \zeta }$

$$
\sum _ { j \leq m } \log ( 1 + T q _ { j } ) \lesssim \sum _ { j \leq m } \{ 1 + \zeta \log ( m / j ) \} \lesssim m ,
$$

where the last step follows from log $( m ! ) \geq$ m log m − m. Also,

$$
\sum _ { j \le m } \kappa _ { j } \theta _ { \star , j } ^ { 2 } \lesssim m \sum _ { j \le m } j ^ { 2 \mathfrak { s } } ( \theta _ { j } ^ { \star } ) ^ { 2 } \lesssim m R ^ { 2 } .
$$

Finally,

$$
T \sum _ { j > m } p _ { j } \theta _ { \star , j } ^ { 2 } \lesssim T R ^ { 2 } \operatorname* { s u p } _ { j > m } j ^ { - \mathfrak { a } - 2 \mathfrak { s } } \stackrel { } { \sim } T m ^ { 1 - \zeta } ,
$$

and

$$
T \sum _ { j > m } p _ { j } \kappa _ { j } ^ { - 1 } \lesssim T \sum _ { j > m } j ^ { - \zeta } \lesssim T m ^ { 1 - \zeta } .
$$

Since $m \asymp T ^ { 1 / \zeta }$ , all terms are $O ( T ^ { 1 / \zeta } )$ , proving (34).

## C.4 Proof of Theorem 4.6

Proof. By Lemma C.1,

$$
W _ { 2 , p } ^ { 2 } ( Q _ { t } ^ { ( m ) } , \Pi _ { t } ) = \sum _ { j > m } p _ { j } W _ { 2 } ^ { 2 } ( \Pi _ { 0 , j } , \Pi _ { t , j } ) .
$$

By the triangle inequality

$$
W _ { 2 } ^ { 2 } ( \Pi _ { 0 , j } , \Pi _ { t , j } ) \le 2 W _ { 2 } ^ { 2 } ( \Pi _ { 0 , j } , \delta _ { \theta _ { j } ^ { \star } } ) + 2 W _ { 2 } ^ { 2 } ( \delta _ { \theta _ { j } ^ { \star } } , \Pi _ { t , j } ) ,
$$

and thus,

$$
\begin{array} { r l } & { W _ { 2 , p } ^ { 2 } ( Q _ { t } ^ { ( m ) } , \Pi _ { t } ) \le 2 \displaystyle \sum _ { j > m } p _ { j } W _ { 2 } ^ { 2 } ( \Pi _ { 0 , j } , \delta _ { \theta _ { j } ^ { \star } } ) + 2 \displaystyle \sum _ { j > m } p _ { j } W _ { 2 } ^ { 2 } ( \delta _ { \theta _ { j } ^ { \star } } , \Pi _ { t , j } ) } \\ & { \qquad \le 2 \displaystyle \sum _ { j > m } p _ { j } W _ { 2 } ^ { 2 } ( \Pi _ { 0 , j } , \delta _ { \theta _ { j } ^ { \star } } ) + 2 W _ { 2 , p } ^ { 2 } ( \Pi _ { t } , \delta _ { \theta ^ { \star } } ) , } \end{array}
$$

where the second term in the last line is bounded by $t ^ { - ( \zeta - 1 ) / \zeta }$ by Theorem 4.5. Hence, it remains to bound the first term. Lemma B.1 applied to the prior at 0 gives

$$
\begin{array} { r l } & { W _ { 2 } ^ { 2 } ( \Pi _ { 0 , j } , \delta _ { \theta _ { j } ^ { \star } } ) \leq \displaystyle \int ( \theta - \theta _ { j } ^ { \star } ) ^ { 2 } \Pi _ { 0 , j } ( { \mathrm { d } } \theta ) } \\ & { \qquad \quad \leq 2 \displaystyle \int \theta ^ { 2 } \Pi _ { 0 , j } ( { \mathrm { d } } \theta ) + 2 ( \theta _ { j } ^ { \star } ) ^ { 2 } } \\ & { \qquad \quad \leq C ^ { \prime } \kappa _ { j } ^ { - 1 } + 2 ( \theta _ { j } ^ { \star } ) ^ { 2 } } \end{array}
$$

for some absolute constant $C ^ { \prime } > 0$ , which implies that

$$
\mathbb { E } [ W _ { 2 , p } ^ { 2 } ( Q _ { t } ^ { ( m ) } , \Pi _ { t } ) ] \lesssim \sum _ { j > m } p _ { j } \{ \kappa _ { j } ^ { - 1 } + ( \theta _ { j } ^ { \star } ) ^ { 2 } \} .
$$

The prior term is $O ( m ^ { 1 - \zeta } )$ because $p _ { j } \kappa _ { j } ^ { - 1 } \asymp j ^ { - \zeta }$ . For the truth term,

$$
\sum _ { j > m } p _ { j } ( \theta _ { j } ^ { \star } ) ^ { 2 } \leq R ^ { 2 } \operatorname* { s u p } _ { j > m } p _ { j } j ^ { - 2 \mathfrak { s } } \lesssim m ^ { - ( \mathfrak { a } + 2 \mathfrak { s } ) } = m ^ { 1 - \zeta } ,
$$

which completes the proof.

## C.5 Proof of Theorem 4.7

Proof. Theorem 4.5 and Theorem 4.6 give

$$
\begin{array} { r } { \varepsilon _ { t } \lesssim t ^ { - ( \zeta - 1 ) / ( 2 \zeta ) } , \qquad \alpha _ { t } \lesssim m ^ { ( 1 - \zeta ) / 2 } + t ^ { - ( \zeta - 1 ) / ( 2 \zeta ) } . } \end{array}
$$

Theorem 2.3 therefore yields

$$
\Re \bigl ( \{ Q _ { t } ^ { ( m ) } \} _ { t = 0 } ^ { T - 1 } \bigr ) - \Re \bigl ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \bigr ) \lesssim \left[ m ^ { ( 1 - \zeta ) / 2 } \sum _ { t = 1 } ^ { T - 1 } t ^ { - ( \zeta - 1 ) / ( 2 \zeta ) } + \sum _ { t = 1 } ^ { T - 1 } t ^ { - ( \zeta - 1 ) / \zeta } + T m ^ { 1 - \zeta } \right]
$$

Combining this with (34) proves (37); setting $m _ { T } = \lceil T ^ { 1 / \zeta } \rceil$ gives (38).

## C.6 Proof of Theorem 4.8

Proof. $\mathrm { A }$ predictor is allowed to observe $X _ { t }$ and the past history before choosing a conditional density $\widehat { p _ { t } } \big ( \cdot | X _ { t } , \mathcal { G } _ { t - 1 } \big )$ . Its regret is

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \log \frac { p _ { \star , X _ { t } } ( Y _ { t } ) } { \widehat { p _ { t } } ( Y _ { t } | X _ { t } , \mathcal { G } _ { t - 1 } ) } \right] .
$$

Let $m = \lfloor c T ^ { 1 / \zeta } \rfloor$ for a fixed small $c > 0$ and put $b _ { m } ^ { 2 } : = c _ { 0 } m ^ { - 2 s - 1 }$ . Consider the uniform prior on the hypercube

$$
\theta _ { j } = \omega _ { j } b _ { m } , \quad j = m + 1 , \ldots , 2 m , \qquad \omega _ { j } \in \{ - 1 , 1 \} ,
$$

with all other coordinates zero. If $c _ { 0 }$ is suficiently small, every vertex belongs to $\Theta _ { \mathfrak { s } } ( B , R )$ . Since the prior distribution $P _ { \omega }$ of $\omega$ does not depend on $X _ { 1 : T }$ , we have

$$
\begin{array} { r l } & { I ( \omega ; Y _ { 1 : T } \mid X _ { 1 : T } ) = \mathrm { K L } ( P _ { \omega , Y _ { 1 : T } | X _ { 1 : T } } \| P _ { \omega } P _ { Y _ { 1 : T } | X _ { 1 : T } } ) } \\ & { \qquad = \mathbb { E } _ { P _ { \omega } } \left[ \mathrm { K L } ( P _ { Y _ { 1 : T } | \omega , X _ { 1 : T } } \| P _ { Y _ { 1 : T } | X _ { 1 : T } } ) \right] . } \end{array}
$$

Let $\widehat { P } _ { Y _ { 1 : T } | X _ { 1 : T } }$ be the joint law of $Y _ { 1 : T }$ generated by the sequential predictive densities. Then we can see that the Bayes-average log-loss regret of the sequential predictor under the prior $P _ { \omega }$ is at least the conditional mutual information $I ( \omega ; Y _ { 1 : T } \mid X _ { 1 : T } )$ , as

$$
\begin{array} { r l } & { \mathbb { E } _ { P _ { \omega } } \left[ \mathrm { K L } ( P _ { Y _ { 1 : T } | \omega , X _ { 1 : T } } | | \widehat { P } _ { Y _ { 1 : T } | \omega , X _ { 1 : T } } ) \right] } \\ & { \ = \mathbb { E } _ { P _ { \omega } } \left[ \mathrm { K L } ( P _ { Y _ { 1 : T } | \omega , X _ { 1 : T } } | | P _ { Y _ { 1 : T } | X _ { 1 : T } } ) \right] + \mathrm { K L } ( P _ { Y _ { 1 : T } | X _ { 1 : T } } | \widehat { P } _ { Y _ { 1 : T } | X _ { 1 : T } } ) ) } \\ & { \ \geq I ( \omega ; Y _ { 1 : T } \mid X _ { 1 : T } ) . } \end{array}
$$

Conditional on $X _ { 1 : T }$ , the observations attached to diferent active coordinates depend on disjoint independent bits, hence

$$
I ( \omega ; Y _ { 1 : T } \mid X _ { 1 : T } ) = \sum _ { j = m + 1 } ^ { 2 m } I ( \omega _ { j } ; Y _ { 1 : N _ { j } } ^ { ( j ) } \mid N _ { j } ) ,
$$

where $N _ { j }$ is the number of occurrences of coordinate $j$ and $Y _ { 1 : N _ { i } } ^ { ( j ) }$ is the collection of observations attached to coordinate $j$ . For one active coordinate, let $P _ { + }$ and $P _ { - }$ denote the laws at natural parameters $b _ { m }$ and $- b _ { m }$ . Their Bhattacharyya afinity is

$$
\int \sqrt { \mathrm { d } P _ { + } \mathrm { d } P _ { - } } = \exp \left\{ A ( 0 ) - \frac { A ( b _ { m } ) + A ( - b _ { m } ) } { 2 } \right\} \leq \mathrm { e } ^ { - \frac { \cal I b _ { m } ^ { 2 } / 2 } { 2 } } ,
$$

by the lower bound on $A ^ { \prime \prime }$ . So for n observations the afinity is at most $\mathrm { e } ^ { - n \underline { { I } } b _ { m } ^ { 2 } / 2 }$ . Since $\mathrm { T V } ( P , Q ) =$ $1 - \int \operatorname* { m i n } ( \mathrm { d } P , \mathrm { d } Q )$ and min $( p , q ) \leq { \sqrt { p q } }$ pointwise, we have

$$
\mathrm { T V } ( P _ { + } ^ { \otimes n } , P _ { - } ^ { \otimes n } ) \geq 1 - \mathrm { e } ^ { - n \underline { { I } } b _ { m } ^ { 2 } / 2 } .
$$

Let $M : = ( P _ { + } ^ { \otimes n } + P _ { - } ^ { \otimes n } ) / 2$ . Conditional on $N _ { j } = n$ , Pinsker’s inequality (e.g. Tsybakov (2009, Section 2.4)) gives

$$
I ( \omega _ { j } ; Y _ { 1 : n } ^ { ( j ) } ) = \frac 1 2 \mathrm { K L } ( P _ { + } ^ { \otimes n } | | M ) + \frac 1 2 \mathrm { K L } ( P _ { - } ^ { \otimes n } | | M ) \geq \frac 1 2 \mathrm { T V } ( P _ { + } ^ { \otimes n } , P _ { - } ^ { \otimes n } ) ^ { 2 } .
$$

For $j \in [ m + 1 , 2 m ] , p _ { j } \asymp m ^ { - a }$ and $\mu _ { j } : = T p _ { j } \asymp T m ^ { - \mathfrak { a } }$ . Since $\mu _ { j } b _ { m } ^ { 2 } \times T m ^ { - \zeta } \asymp 1$ and $\mu _ { j } \asymp m ^ { 2 \mathfrak { s } + 1 } \to$ ∞, the same exponential-Markov calculation as in Lemma C.3 gives $\operatorname* { P r } ( N _ { j } < \mu _ { j } / 2 ) \le \mathrm { e } ^ { - c _ { 0 } \mu _ { j } }$ . Hence $\operatorname* { P r } ( N _ { j } \geq \mu _ { j } / 2 ) \gtrsim 1$ uniformly over the active coordinates for all suficiently large T. On this event the preceding mutual-information lower bound is bounded below by a positive constant depending only on the fixed model constants. Thus

$$
\mathbb { E } [ I ( \omega _ { j } ; Y _ { 1 : N _ { j } } ^ { ( j ) } \mid N _ { j } ) ] \gtrsim 1 ,
$$

uniformly over active $j .$ Summing over the m active coordinates yields Bayes-average regret at least $m \asymp T ^ { 1 / \zeta }$ up to a multiplicative constant. Since the maximum risk dominates Bayes average risk, the minimax regret is at least $T ^ { 1 / \zeta }$ up to a multiplicative constant. □

## D Proofs for Section 5

We introduce additional notation used in this section. For a Gaussian law P on $L ^ { 2 } ( \mathbb { P } _ { X } )$ with covariance kernel $\mathsf { K } _ { P }$ , the covariance operator $\Sigma _ { P }$ is defined as

$$
\Sigma _ { P } f ( x ^ { \prime } ) = \int { \mathsf K } _ { P } ( x , x ^ { \prime } ) f ( x ) \mathbb { P } _ { X } ( \mathrm { d } x ) .
$$

Let H be the reproducing kernel Hilbert space (RKHS) associated with the prior GP $\Pi _ { 0 }$ . Moreover, we define

$$
J _ { t } ^ { \dagger } : = \left\lceil t ^ { d / ( d + 2 \mathfrak { s } ) } \right\rceil , \qquad N _ { t } ^ { \dagger } : = \left\lceil t ^ { d / ( 2 \mathfrak { s } ) } \right\rceil\tag{67}
$$

## D.1 Regular spectral design

Define the event

$$
\mathcal { E } _ { t } : = \left\{ \frac { t } { 2 } I _ { N _ { t } ^ { \dagger } } \preceq \Psi _ { t , N _ { t } ^ { \dagger } } ^ { \top } \Psi _ { t , N _ { t } ^ { \dagger } } \preceq \frac { 3 t } { 2 } I _ { N _ { t } ^ { \dagger } } \right\} .\tag{68}
$$

Lemma D.1 (Regular spectral design). lem:gp-design Under Assumption 5.2, there are absolute constants $c > 0$ and $C > 0$ such that

$$
\begin{array} { r } { \operatorname* { P r } \left( \mathcal { E } _ { t } \right) \ge 1 - C N _ { t } ^ { \dagger } \exp \{ - c t / N _ { t } ^ { \dagger } \} . } \end{array}\tag{69}
$$

Since $N _ { t } ^ { \dagger } \asymp t ^ { d / ( 2 \varsigma ) }$ , the probability of the complement tends to 0 faster than $t ^ { - b }$ for any $b > 0$

Proof. Let $N : = N _ { t } ^ { \dagger }$ . For $\psi _ { 1 : N } ( x ) = ( \psi _ { 1 } ( x ) , \ldots , \psi _ { L } ( x ) ) ^ { \top }$ , orthonormality gives $\mathbb { E } [ \psi _ { 1 : N } ( X ) \psi _ { 1 : N } ( X ) ^ { \top } ] =$ $I _ { L }$ , while (42) gives $\| \psi _ { 1 : N } ( X ) \psi _ { 1 : N } ( X ) ^ { \top } \| _ { \mathrm { o p } } = \| \psi _ { 1 : N } ( X ) \| ^ { 2 } \leq B _ { \psi } ^ { 2 } L$ . The upper and lower matrix Chernof inequalities with relative deviation $1 / 2$ yield (69). □

## D.2 Proof of Theorem 5.1

Lemma D.2 (Gaussian predictive excess). Let P be a Gaussian law on $L ^ { 2 } ( \mathbb { P } _ { X } )$ . Let $m _ { P }$ be the mean function of P and $v _ { P }$ be the variance of $f ( x )$ for $f \sim P$ . Then

$$
\mathcal M _ { t } ( P ) - \mathcal M _ { t } ( \delta _ { f ^ { \star } } ) = \frac 1 2 \int \left[ g \bigg ( \frac { v _ { P } ( x ) } { \sigma _ { \xi } ^ { 2 } } \bigg ) + \frac { \{ m _ { P } ( x ) - f ^ { \star } ( x ) \} ^ { 2 } } { \sigma _ { \xi } ^ { 2 } + v _ { P } ( x ) } \right] \mathbb P _ { X } ( \mathrm { d } x ) ,\tag{70}
$$

where $g ( u ) = \log ( 1 + u ) - u / ( 1 + u )$ . In particular,

$$
0 \leq \mathcal { M } _ { t } ( P ) - \mathcal { M } _ { t } ( \delta _ { f ^ { \star } } ) \leq \frac { 1 } { 2 \sigma _ { \xi } ^ { 2 } } W _ { 2 } ^ { 2 } ( P , \delta _ { f ^ { \star } } ) .\tag{71}
$$

Proof. Conditionally on $X \ = \ x .$ the true response law is $\mathbb { N } ( f ^ { \star } ( x ) , \sigma _ { \xi } ^ { 2 } )$ and the predictive law is $\mathbb { N } ( m _ { P } ( x ) , \sigma _ { \xi } ^ { 2 } + v _ { P } ( x ) )$ . Their Gaussian KL divergence gives (70). Since $0 \leq g ( u ) \leq u$ and $( \sigma _ { \xi } ^ { 2 } + v _ { P } ) ^ { - 1 } \leq \sigma _ { \xi } ^ { - 2 }$ , the right-hand side of (70) is at most $\{ \| m _ { P } - f ^ { \star } \| ^ { 2 } + \| v _ { P } \| \} / ( 2 \sigma _ { \xi } ^ { 2 } )$ , which equals the right-hand side of (71). □

Lemma D.3 (Trace norm and Bures–Wasserstein distance). For positive trace-class operators A and B

$$
\| A - B \| _ { 1 } \leq \{ { \sqrt { \operatorname { t r } ( A ) } } + { \sqrt { \operatorname { t r } ( B ) } } \} { \mathsf { d } } _ { \mathrm { B W } } ( A , B ) ,\tag{72}
$$

where $\| \cdot \| _ { 1 }$ denotes the Schatten-1 norm and d<sub>BW</sub> does the Bures–Wasserstein distance defined as

$$
\begin{array} { r } { { \bf d } _ { \mathrm { B W } } ^ { 2 } ( A , B ) : = \mathrm { t r } ( A + B - 2 ( A ^ { 1 / 2 } B A ^ { 1 / 2 } ) ^ { 1 / 2 } ) . } \end{array}
$$

Proof. Use the Hilbert–Schmidt representation $\mathsf { d } _ { \mathrm { B W } } ( A , B ) = \operatorname* { i n f } _ { U } \left. A ^ { 1 / 2 } - B ^ { 1 / 2 } U \right. _ { \mathrm { H S } }$ . For $X = A ^ { 1 / 2 }$ and $Y = B ^ { 1 / 2 } U , A - B = ( X - Y ) X ^ { * } + Y ( X ^ { * } - Y ^ { * } )$ . By the H¨older inequality for the Schatten norm gives

$$
\left\| A - B \right\| _ { 1 } \leq \left\{ \left\| X \right\| _ { \mathrm { H S } } + \left\| Y \right\| _ { \mathrm { H S } } \right\} \left\| X - Y \right\| _ { \mathrm { H S } } .
$$

Taking the infimum over U proves the claim.

Proof of Theorem 5.1. Recall the notation used in Lemmas D.2 and D.3. Let $m _ { P }$ and $\Sigma _ { P }$ be the mean function and covariance operator of P. Define $m _ { Q }$ and $\Sigma _ { Q }$ likewise. We write

$$
\varepsilon : = W _ { 2 } ( P , \delta _ { f ^ { \star } } ) , \qquad \alpha : = W _ { 2 } ( P , Q ) ,
$$

and put $b _ { P } : = m _ { P } - f ^ { \star }$ and $b _ { Q } : = m _ { Q } - f ^ { \star }$ . By Lemma D.2, we have

$$
\begin{array} { l } { \displaystyle \mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( P ) = \mathcal { M } _ { t } ( Q ) - \mathcal { M } _ { t } ( \delta _ { f ^ { \star } } ) - \left\{ \mathcal { M } _ { t } ( P ) - \mathcal { M } _ { t } ( \delta _ { f ^ { \star } } ) \right\} } \\ { \displaystyle = \frac { 1 } { 2 } \int \left[ g \left( \frac { v _ { Q } ( x ) } { \sigma _ { \xi } ^ { 2 } } \right) - g \left( \frac { v _ { P } ( x ) } { \sigma _ { \xi } ^ { 2 } } \right) + \frac { b _ { Q } ( x ) ^ { 2 } } { \sigma _ { \xi } ^ { 2 } + v _ { Q } ( x ) } - \frac { b _ { P } ( x ) ^ { 2 } } { \sigma _ { \xi } ^ { 2 } + v _ { P } ( x ) } \right] \mathbb { P } _ { X } ( \mathrm { d } x ) , } \end{array}\tag{73}
$$

By the Gaussian Wasserstein formula $W _ { 2 } ^ { 2 } ( P , Q ) \ = \ \| m _ { p } - m _ { Q } \| ^ { 2 } + { \sf d } _ { \mathrm { B W } } ^ { 2 } ( \Sigma _ { Q } , \Sigma _ { P } )$ , we have d<sub>BW</sub> $\left( \Sigma _ { Q } , \Sigma _ { P } \right) \leq \alpha$ . Therefore, Lemma D.3 gives

$$
\int | v _ { Q } ( x ) - v _ { P } ( x ) | \mathbb { P } _ { X } ( \mathrm { d } x ) \leq \| \Sigma _ { Q } - \Sigma _ { P } \| _ { 1 } \leq ( 2 \varepsilon + \alpha ) \alpha ,
$$

because $\sqrt { \mathrm { t r } ( \Sigma _ { P } ) } \leq \varepsilon$ and $\sqrt { \mathrm { t r } ( \Sigma _ { Q } ) } \le \varepsilon + \alpha$ . In (70), $g ^ { \prime } ( u ) = u / ( 1 + u ) ^ { 2 } \leq 1 / 4$ , so the variance-only diference in (73) is bounded by a constant multiple of $( 2 \varepsilon + \alpha ) \alpha$ . For the mean-related diference in (73), we use the inequality

$$
\begin{array} { r l } & { \int \left| \frac { b _ { Q } ^ { 2 } ( x ) } { \sigma _ { \xi } ^ { 2 } + v _ { Q } ( x ) } - \frac { b _ { P } ^ { 2 } ( x ) } { \sigma _ { \xi } ^ { 2 } + v _ { P } ( x ) } \right| \mathbb { P } _ { X } ( \mathrm { d } x ) } \\ & { \leq \frac { 1 } { \sigma _ { \xi } ^ { 2 } } \int \left| b _ { Q } ^ { 2 } ( x ) - b _ { P } ^ { 2 } ( x ) \right| \mathbb { P } _ { X } ( \mathrm { d } x ) + \frac { 1 } { \sigma _ { \xi } ^ { 4 } } \int b _ { P } ^ { 2 } ( x ) | v _ { Q } ( x ) - v _ { P } ( x ) | \mathbb { P } _ { X } ( \mathrm { d } x ) } \\ & { \leq \frac { 1 } { \sigma _ { \xi } ^ { 2 } } \int \left| b _ { Q } ^ { 2 } ( x ) - b _ { P } ^ { 2 } ( x ) \right| \mathbb { P } _ { X } ( \mathrm { d } x ) + \frac { 1 } { \sigma _ { \xi } ^ { 4 } } \overline { { v } } ( P , Q ) \left\| b _ { P } \right\| ^ { 2 } } \end{array}
$$

For the first term, we use the Gaussian Wasserstein formula to have $\| b _ { Q } - b _ { P } \| = \| m _ { Q } - m _ { P } \| \leq \alpha$ and $\| b _ { P } \| \leq \varepsilon$ . Moreover, $\| b _ { Q } \| \leq \| b _ { Q } - b _ { P } \| + \| b _ { P } \| \leq \varepsilon + \alpha$ . Therefore, by the Cauchy–Schwarz inequality

$$
\int \left| b _ { Q } ^ { 2 } ( x ) - b _ { P } ^ { 2 } ( x ) \right| \mathbb { P } _ { X } ( \mathrm { d } x ) \leq \| b _ { Q } - b _ { P } \| \left( \| b _ { Q } \| + \| b _ { P } \| \right) \leq \alpha ( 2 \varepsilon + \alpha ) .\tag{74}
$$

Combining these bounds proves (41).

## D.3 Proof of Theorem 5.4

Proof. For a Gaussian process posterior $\Pi _ { t }$ , we have

$$
W _ { 2 } ^ { 2 } ( \Pi _ { t } , \delta _ { f ^ { \star } } ) = \| \widehat { f } _ { t } - f ^ { \star } \| ^ { 2 } + \operatorname { t r } ( \Sigma _ { t } ) .
$$

where $\widehat { f } _ { t }$ denotes the posterior mean. Taking expectations and applying Lemmas D.4 and D.5 below proves (44). Lemma D.2 gives

$$
\Re \big ( ( \Pi _ { t } ) _ { t = 0 } ^ { T - 1 } \big ) \le \frac { 1 } { 2 \sigma _ { \xi } ^ { 2 } } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ W _ { 2 } ^ { 2 } ( \Pi _ { t } , \delta _ { f ^ { \star } } ) ] .
$$

The $t = 0$ term is finite and the remaining sum is bounded by $\begin{array} { r } { \sum _ { t = 1 } ^ { T - 1 } t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } \lesssim T ^ { d / ( d + 2 \mathfrak { s } ) } } \end{array}$ up to a multiplicative constant. □

Lemma D.4 (Exact posterior covariance geometry). Let $\Sigma _ { t }$ be the covariance operator of the $G P$ posterior $\Pi _ { t }$ . Under Assumptions 5.2 and 5.3,

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert \Sigma _ { t } \Vert _ { \mathrm { o p } } \right] \lesssim t ^ { - 1 } , \qquad \mathbb { E } \left[ \mathrm { t r } ( \Sigma _ { t } ) \right] \lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } . } \end{array}\tag{75}
$$

Proof. Put $N ~ = ~ N _ { t } ^ { \dagger } ~ \asymp ~ t ^ { d / ( 2 \varsigma ) }$ and let $\mathsf { P } _ { N }$ denote the orthogonal projection onto $\mathcal { H } _ { N } : =$ span $\left( \psi _ { 1 } , \dots , \psi _ { N } \right)$ . Under the prior GP,

$$
\begin{array} { r } { u _ { 1 : N } : = \left( \left. f , \psi _ { 1 } \right. , \ldots , \left. f , \psi _ { N } \right. \right) ^ { \top } \sim \mathtt { N } ( 0 , \Lambda _ { N } ) , } \end{array}\tag{76}
$$

with $\Lambda _ { N } : = \mathrm { d i a g } ( \lambda _ { 1 } , . . . , \lambda _ { N } )$ . We first control the contribution of the coeficients above N. Define

$$
R _ { t , N } : = \sum _ { j > N } \lambda _ { j } \psi _ { j } ( X _ { 1 : t } ) \psi _ { j } ( X _ { 1 : t } ) ^ { \top } ,\tag{77}
$$

where $\psi _ { j } ( X _ { 1 : t } ) : = ( \psi _ { j } ( X _ { 1 } ) , \ldots , \psi _ { j } ( X _ { t } ) ) ^ { \intercal }$ . By uniform boundedness of the eigenfunctions in (42) and the eigenvalue decay in Assumption 5.3,

$$
\begin{array} { l } { \displaystyle \| R _ { t , N } \| _ { \mathrm { o p } } \leq \mathrm { t r } ( R _ { t , N } ) = \displaystyle \sum _ { i = 1 } ^ { t } \sum _ { j > N } \lambda _ { j } \psi _ { j } ( X _ { i } ) ^ { 2 } } \\ { \leq B _ { \psi } ^ { 2 } t \displaystyle \sum _ { j > N } \lambda _ { j } \lesssim t N ^ { - 2 \mathfrak { s } / d } \lesssim 1 . } \end{array}\tag{78}
$$

To identify the posterior covariance of the first N coeficients, write

$$
Y _ { 1 : t } = \Psi _ { t , N } u _ { 1 : N } + \eta _ { t , N } + \xi _ { 1 : t } , \quad \eta _ { t , N } : = \sum _ { j > N } u _ { j } \psi _ { j } ( X _ { 1 : t } ) .\tag{79}
$$

Note that

$$
\eta _ { t , N } | X _ { 1 : t } \sim \mathbb { N } ( 0 , R _ { t , N } ) , \qquad \xi _ { 1 : t } \sim \mathbb { N } ( 0 , \sigma _ { \xi } ^ { 2 } I _ { t } ) ,\tag{80}
$$

and both vectors are independent of $u 1 { : } N$ . Hence, after integrating out the coeficients above $N$ , we have

$$
Y _ { 1 : t } \mid u _ { 1 : N } , X _ { 1 : t } \sim \mathbb { N } ( \Psi _ { t , N } u _ { 1 : N } , \sigma _ { \xi } ^ { 2 } I _ { t } + R _ { t , N } ) .\tag{81}
$$

Gaussian conjugacy therefore shows that the posterior covariance matrix of $u _ { 1 : N }$ conditional on $X _ { 1 : t }$ , equivalently the matrix representation of $\mathsf { P } _ { N } \Sigma _ { t } \mathsf { P } _ { N }$ in the basis $( \psi _ { j } ) _ { j \leq N }$ , is

$$
\begin{array} { r } { \Sigma _ { t , N } = \big \{ \Lambda _ { N } ^ { - 1 } + \Psi _ { t , N } ^ { \top } \big ( \sigma _ { \xi } ^ { 2 } I _ { t } + R _ { t , N } \big ) ^ { - 1 } \Psi _ { t , N } \big \} ^ { - 1 } . } \end{array}\tag{82}
$$

By (78), there exists an absolute constant $C ^ { \prime } > 0$ such that

$$
\sigma _ { \xi } ^ { 2 } I _ { t } + R _ { t , N } \preceq ( \sigma _ { \xi } ^ { 2 } + C ^ { \prime } ) I _ { t } ,\tag{83}
$$

and hence

$$
( \sigma _ { \xi } ^ { 2 } I _ { t } + R _ { t , N } ) ^ { - 1 } \succeq \frac { 1 } { \sigma _ { \xi } ^ { 2 } + C ^ { \prime } } I _ { t } .\tag{84}
$$

Since $\begin{array} { r } { \Psi _ { t , N } ^ { \top } \Psi _ { t , N } \succeq \frac { t } { 2 } I _ { N } } \end{array}$ on $\mathcal { E } _ { t }$ define in (68), we have

$$
\Psi _ { t , N } ^ { \top } ( \sigma _ { \xi } ^ { 2 } I _ { t } + R _ { t , N } ) ^ { - 1 } \Psi _ { t , N } \succeq \frac { t } { 2 ( \sigma _ { \xi } ^ { 2 } + C ^ { \prime } ) } I _ { N }\tag{85}
$$

on $\mathcal { E } _ { t }$ . Since the prior precision ${ \Lambda } _ { N } ^ { - 1 }$ is positive semidefinite, (82) gives $\| \Sigma _ { t , N } \| _ { \mathrm { o p } } \mathbf { 1 } _ { \mathcal { E } _ { t } } \lesssim t ^ { - 1 }$

It remains to pass from the first N coeficient directions to the full covariance operator. Relative to the orthogonal decomposition $L ^ { 2 } ( \mathbb { P } _ { X } ) = \mathcal { H } _ { N } \oplus \mathcal { H } _ { N } ^ { \bot }$ , write

$$
\Sigma _ { t } = \binom { A } { B ^ { * } } \begin{array} { l } { B } \\ { D } \end{array} ) ,\tag{86}
$$

where $A = \mathsf { P } _ { N } \Sigma _ { t } \mathsf { P } _ { N }$ and $D = \mathsf { P } _ { > N } \Sigma _ { t } \mathsf { P } _ { > N }$ , with the orthogonal projection $\mathsf { P } _ { > N }$ onto span $( \psi _ { j }$ $j > N )$ . The matrix representation of A is $\Sigma _ { t , N }$ , so $\| A \| _ { \mathrm { o p } } \mathbf { 1 } _ { \mathcal { E } _ { t } } \lesssim t ^ { - 1 }$ . Moreover, since Gaussian conditioning reduces covariance, we have $0 \preceq \Sigma _ { t } \preceq \Sigma _ { 0 }$ and so

$$
0 \preceq D \preceq \mathsf { P } _ { > N } \Sigma _ { 0 } \mathsf { P } _ { > N } .\tag{87}
$$

Hence,

$$
\| D \| _ { \mathrm { o p } } \le \operatorname* { s u p } _ { j > N } \lambda _ { j } \lesssim N ^ { - 1 - 2 \mathfrak { s } / d } \lesssim t ^ { - 1 } .\tag{88}
$$

Because $\Sigma _ { t }$ is positive semidefinite, covariance Cauchy–Schwarz gives, for $x \in \mathcal H _ { N }$ and $y \in \mathcal { H } _ { N } ^ { \perp }$ ，

$$
| \left. x , B y \right. | \leq { \sqrt { \left. x , A x \right. } } { \sqrt { \left. y , D y \right. } } .\tag{89}
$$

Thus

$$
\langle ( x , y ) , \Sigma _ { t } ( x , y ) \rangle \leq \left\{ \sqrt { \| A \| _ { \mathrm { o p } } } \| x \| + \sqrt { \| D \| _ { \mathrm { o p } } } \| y \| \right\} ^ { 2 }
$$

$$
\leq \{ \| A \| _ { \mathrm { o p } } + \| D \| _ { \mathrm { o p } } \} ( \| x \| ^ { 2 } + \| y \| ^ { 2 } ) .
$$

Taking the supremum over unit vectors yields $\| \Sigma _ { t } \| _ { \mathrm { o p } } \leq \| A \| _ { \mathrm { o p } } + \| D \| _ { \mathrm { o p } }$ . Thus, by ??, Equation (88) and the fact that $0 \preceq \Sigma _ { t } \preceq \Sigma _ { 0 }$ , we have

$$
\mathbb { E } \left[ \Vert \Sigma _ { t } \Vert _ { \mathrm { o p } } \right] \leq \mathbb { E } \left[ \Vert \Sigma _ { t } \Vert _ { \mathrm { o p } } 1 _ { \mathcal { E } _ { t } } \right] + \mathbb { E } \left[ \Vert \Sigma _ { t } \Vert _ { \mathrm { o p } } \mathbf { 1 } _ { \mathcal { E } _ { t } ^ { \complement } } \right]\tag{90}
$$

$$
\leq \mathbb { E } \left[ \| A \| _ { \mathrm { o p } } \mathbf { 1 } _ { \mathcal { E } _ { t } } \right] + \mathbb { E } \left[ \| D \| _ { \mathrm { o p } } \right] + \lambda _ { 1 } \operatorname* { P r } ( \mathcal { E } _ { t } ^ { \complement } ) \lesssim t ^ { - 1 } .\tag{91}
$$

which proves the operator-norm bound.

We now bound the trace. Put $J = J _ { t } ^ { \dagger }$ . Since $\mathsf { P } _ { J } \boldsymbol { \Sigma } _ { t } \mathsf { P } _ { J }$ is positive semidefinite and has rank at most $J _ { i { \mathrm { : } } }$

$$
\mathrm { t r } ( \mathsf P _ { J } \Sigma _ { t } \mathsf P _ { J } ) \mathbf 1 _ { \mathscr E _ { t } } \leq J \| \Sigma _ { t } \| _ { \mathrm { o p } } \mathbf 1 _ { \mathscr E _ { t } } \lesssim \frac { J } { t } .\tag{92}
$$

For the remaining directions, using covariance domination $0 \preceq \Sigma _ { t } \preceq \Sigma _ { 0 }$ again, we have

$$
\mathrm { t r } ( \mathsf P _ { > J } \Sigma _ { t } \mathsf P _ { > J } ) \le \mathrm { t r } ( \mathsf P _ { > J } \Sigma _ { 0 } \mathsf P _ { > J } ) = \sum _ { j > J } \lambda _ { j } \lesssim J ^ { - 2 \varsigma / d } .
$$

Combining these two inequalities, for $J _ { t } ^ { \dagger } \asymp t ^ { d / ( d + 2 \varsigma ) }$ , we have

$$
\mathrm { t r } ( \Sigma _ { t } ) \mathbf { 1 } _ { \mathcal { E } _ { t } } \lesssim \left\{ \frac { J } { t } + J ^ { - 2 \mathfrak { s } / d } \right\} \asymp t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } .\tag{93}
$$

Therefore, we have

$$
\mathbb { E } \left[ \operatorname { t r } ( \Sigma _ { t } ) \right] \leq \mathbb { E } \left[ \operatorname { t r } ( \Sigma _ { t } ) \mathbf { 1 } _ { \mathcal { E } _ { t } } \right] + \mathbb { E } \left[ \operatorname { t r } ( \Sigma _ { t } ) \mathbf { 1 } _ { \mathcal { E } _ { t } ^ { \complement } } \right]\tag{94}
$$

$$
\lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } + \operatorname { t r } ( \Sigma _ { 0 } ) \operatorname* { P r } ( \mathcal { E } _ { t } ^ { \complement } ) \lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) }\tag{95}
$$

which completes the proof.

Lemma D.5 (Risk of the exact posterior mean). Let $\widehat { f } _ { t }$ denote the mean of the GP posterior $\Pi _ { t }$ Under Assumptions 5.2 and 5.3,

$$
\mathbb { E } \left[ \Vert \widehat { f } _ { t } - f ^ { \star } \Vert ^ { 2 } \right] \lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } .\tag{96}
$$

Proof. The exact GP posterior mean is the kernel-ridge regression estimator

$$
\widehat { f } _ { t } = \underset { f \in \mathbb { H } } { \mathrm { a r g m i n } } \left\{ \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \{ Y _ { i } - f ( X _ { i } ) \} ^ { 2 } + \rho _ { t } \Vert f \Vert _ { \mathbb { H } } ^ { 2 } \right\} , \qquad \rho _ { t } : = \frac { \sigma _ { \xi } ^ { 2 } } { t } .
$$

We apply Theorem 1(ii) of Fischer and Steinwart (2020). In their notation, set

$$
p : = \frac { d } { d + 2 \mathfrak { s } } , \qquad \beta : = \frac { 2 \mathfrak { s } } { d + 2 \mathfrak { s } } ,
$$

so that $p + \beta = 1$ . The eigenvalue condition in Assumption 5.3 is then written as $\lambda _ { j } \asymp j ^ { - 1 / p }$ Moreover,

$$
\sum _ { j \ge 1 } \lambda _ { j } ^ { - \beta } ( f _ { j } ^ { \star } ) ^ { 2 } \lesssim \sum _ { j \ge 1 } j ^ { 2 \mathfrak { s } / d } ( f _ { j } ^ { \star } ) ^ { 2 } < \infty ,
$$

so the source condition of Fischer and Steinwart (2020) holds with exponent $\beta .$ . The uniform bound on the eigenfunctions implies that, for every $q > p ,$

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } \sum _ { j \geq 1 } \lambda _ { j } ^ { q } \psi _ { j } ( x ) ^ { 2 } \leq B _ { \psi } ^ { 2 } \sum _ { j \geq 1 } \lambda _ { j } ^ { q } < \infty .
$$

which verifies their embedding condition with any fixed $q \in ( p , 1 )$ , and that

$$
\| f ^ { \star } \| _ { \infty } \leq B _ { \psi } \sum _ { j \geq 1 } | f _ { j } ^ { \star } | \leq B _ { \psi } R \Big ( \sum _ { j \geq 1 } j ^ { - 2 \lceil d \rceil } \Big ) ^ { 1 / 2 } < \infty .\tag{97}
$$

Moreover, Gaussian noise satisfies their moment condition. Since

$$
\beta + p = 1 > q \qquad \mathrm { a n d } \qquad \rho _ { t } \asymp t ^ { - 1 / ( \beta + p ) } = t ^ { - 1 } ,
$$

Theorem 1(ii) of Fischer and Steinwart (2020) implies that there is an absolute constant $C ^ { \prime } > 0$ such that

$$
\mathrm { P r } \left( \left\| \widehat { f } _ { t } - f ^ { \star } \right\| ^ { 2 } > C ^ { \prime } \tau ^ { 2 } t ^ { - \beta } \right) \leq 4 \mathrm { e } ^ { - \tau }\tag{98}
$$

for $\tau \geq 1$ . We now pass from (98) to an expectation bound. Inspection of the sample-size condition in Fischer and Steinwart (2020, Theorem 16 and the proof of Theorem 1) shows that (98) holds uniformly for

$$
1 \leq \tau \leq \bar { \tau } _ { t } , \qquad \bar { \tau } _ { t } : = c \frac { t ^ { 1 - q } } { \log t } ,
$$

for some suficiently small constant $c > 0$ and all suficiently large t. For simplicity, we let $W _ { t } : = \left\| \widehat { f } _ { t } - f ^ { \star } \right\| ^ { 2 }$ . By the change of variables $v ^ { 2 } = ( C ^ { \prime } t ^ { - \beta } ) ^ { - 1 } u .$ , the truncated expectation is bounded as

$$
\begin{array} { r l } & { \mathbb { E } \left[ W _ { t } \wedge C ^ { \prime } \bar { \tau } _ { t } ^ { 2 } t ^ { - \beta } \right] \leq C ^ { \prime } t ^ { - \beta } + \displaystyle \int _ { C ^ { \prime } t ^ { - \beta } } ^ { C ^ { \prime } \bar { \tau } _ { t } ^ { 2 } t ^ { - \beta } } \operatorname* { P r } ( W _ { t } > u ) \mathrm { d } u } \\ & { \qquad \leq C ^ { \prime } t ^ { - \beta } + 2 ( C ^ { \prime } t ^ { - \beta } ) \displaystyle \int _ { 1 } ^ { \bar { \tau } _ { t } } v \operatorname* { P r } ( W _ { t } > ( C ^ { \prime } t ^ { - \beta } ) v ^ { 2 } ) \mathrm { d } v } \\ & { \qquad \leq C ^ { \prime } t ^ { - \beta } + 8 ( C ^ { \prime } t ^ { - \beta } ) \displaystyle \int _ { 1 } ^ { \bar { \tau } _ { t } } v \mathrm { e } ^ { - v } \mathrm { d } v \lesssim t ^ { - \beta } . } \end{array}
$$

It remains to control the upper tail. By optimality of the kernel-ridge estimator and comparison with $f = 0$ 2

$$
\rho _ { t } \big \| \widehat { f } _ { t } \big \| _ { \mathbb { H } } ^ { 2 } \leq \frac { 1 } { t } \sum _ { i = 1 } ^ { t } Y _ { i } ^ { 2 } .
$$

Since $\rho _ { t } = \sigma _ { \xi } ^ { 2 } / t$ and $\left\| f \right\| ^ { 2 } \leq \lambda _ { 1 } \left\| f \right\| _ { \mathbb { H } } ^ { 2 }$ , Gaussian noise and the boundedness of $f ^ { \star }$ imply

$$
\mathbb { E } [ W _ { t } ^ { 2 } ] \lesssim t ^ { 2 } .
$$

Furthermore, (98) at $\tau = \bar { \tau } _ { t }$ gives

$$
\operatorname* { P r } \{ W _ { t } > C ^ { \prime } \bar { \tau } _ { t } ^ { 2 } t ^ { - \beta } \} \leq 4 \mathrm { e } ^ { - \bar { \tau } _ { t } } .
$$

Therefore, by the Cauchy–Schwarz inequality

$$
\begin{array} { r l r } & { } & { { \mathbb E } \left[ W _ { t } { \mathbf 1 } \{ W _ { t } > C ^ { \prime } \bar { \tau } _ { t } ^ { 2 } t ^ { - \beta } \} \right] \leq ( { \mathbb E } W _ { t } ^ { 2 } ) ^ { 1 / 2 } \operatorname* { P r } \{ W _ { t } > C ^ { \prime } \bar { \tau } _ { t } ^ { 2 } t ^ { - \beta } \} ^ { 1 / 2 } } \\ & { } & { \lesssim t \mathrm { e } ^ { - \bar { \tau } _ { t } / 2 } = o ( t ^ { - \beta } ) . } \end{array}
$$

Combining the truncated expectation and the upper-tail bound gives

$$
\begin{array} { r } { \mathbb { E } [ W _ { t } ] \lesssim t ^ { - \beta } = t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } . } \end{array}
$$

Finitely many small values of t are absorbed into the constant.

## D.4 Proof of Theorem 5.5

Proof. The variational process posterior is absolutely continuous with respect to the exact posterior and has finite reverse KL. Conditional on the data, Lemma D.6 below gives

$$
W _ { 2 } ^ { 2 } ( Q _ { t } ^ { ( J ) } , \Pi _ { t } ) \leq 2 \| \Sigma _ { t } \| _ { \mathrm { o p } } \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) .
$$

In the proof of Lemma D.4, we have shown that on the event $\mathcal { E } _ { t }$ in (68), $\| \Sigma _ { t } \| _ { \mathrm { o p } } \lesssim t ^ { - 1 }$ . Therefore, Lemma D.7 gives

$$
\begin{array} { r } { \mathbb { E } \left[ \| \Sigma _ { t } \| _ { \mathrm { o p } } \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) \mathbf { 1 } _ { \mathcal { E } _ { t } } \right] \lesssim t ^ { - 1 } \mathbb { E } \left[ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) \right] \lesssim t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } . } \end{array}
$$

On the complement $\mathcal { E } _ { t } ^ { \complement }$ , since $\| \Sigma _ { t } \| _ { \mathrm { o p } } \leq \| \Sigma _ { 0 } \| _ { \mathrm { o p } } \lesssim 1$ and the event $\mathcal { E } _ { t } ^ { \complement }$ does not depend on the outcome variables $Y _ { 1 : t }$ , we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \| \Sigma _ { t } \| _ { \mathrm { o p } } \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) \mathbf { 1 } _ { \mathcal { E } _ { t } ^ { \complement } } \right] \lesssim \mathbb { E } \left[ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) \mathbf { 1 } _ { \mathcal { E } _ { t } ^ { \complement } } \right] } \\ & { \qquad = \mathbb { E } \left[ \mathbb { E } \left[ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) \big | X _ { 1 : t } \right] \mathbf { 1 } _ { \mathcal { E } _ { t } ^ { \complement } } \right] } \\ & { \qquad \lesssim t \mathrm { P r } ( \mathcal { E } _ { t } ^ { \complement } ) = o ( t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } ) , } \end{array}
$$

where we use Lemma D.8 given below for the second inequality. Combining the derived inequalities, we get the desired result. □

Lemma D.6 (Gaussian transportation). Let $\gamma$ be a Gaussian law on $L ^ { 2 } ( \mathbb { P } _ { X } )$ with covariance operator Σ. If $\nu \ll \gamma$ and $\mathrm { K L } ( \nu \| \gamma ) < \infty$ , then

$$
W _ { 2 } ^ { 2 } ( \nu , \gamma ) \leq 2 \| \Sigma \| _ { \mathrm { o p } } \mathrm { K L } ( \nu \| \gamma ) .\tag{99}
$$

Proof. By translation invariance of both the Wasserstein distance and relative entropy, it is enough to consider the centered case. Let $( e _ { j } ) _ { j \geq 1 }$ be an orthonormal basis corresponding to the positive eigenvalues of Σ. Let $H _ { \Sigma } : = \overline { { \operatorname { s p a n } ( e _ { j } : j \geq 1 ) } }$ . Then the Gaussian measure $\gamma$ is supported on $H _ { \Sigma }$ and $\nu \ll \gamma$ implies that ν is supported on the same space. For $N \geq 1$ , let $\mathsf { P } _ { N }$ be the orthogonal projection onto span $( e _ { 1 } , \dots , e _ { N } )$ , and write

$$
\begin{array} { r } { \nu _ { N } : = ( \mathsf { P } _ { N } ) _ { \# } \nu , \qquad \gamma _ { N } : = ( \mathsf { P } _ { N } ) _ { \# } \gamma . } \end{array}
$$

Then

$$
\gamma _ { N } = \mathtt { N } ( 0 , \Sigma _ { N } ) , \qquad \Sigma _ { N } : = \mathsf { P } _ { N } \Sigma \mathsf { P } _ { N } .
$$

The finite-dimensional Gaussian transportation inequality (Talagrand, 1996), followed by linear rescaling, gives

$$
W _ { 2 } ^ { 2 } ( \nu _ { N } , \gamma _ { N } ) \leq 2 \| \Sigma _ { N } \| _ { \mathrm { o p } } \mathrm { K L } ( \nu _ { N } \| \gamma _ { N } ) .
$$

Since $\| \Sigma _ { N } \| _ { \mathrm { o p } } \leq \| \Sigma \| _ { \mathrm { o p } }$ and $\mathrm { K L } ( \nu _ { N } \| \gamma _ { N } ) \le \mathrm { K L } ( \nu \| \gamma )$ (because the KL divergence decreases under measurable transformations), we have, for every $N$

$$
W _ { 2 } ^ { 2 } ( \nu _ { N } , \gamma _ { N } ) \leq 2 \| \Sigma \| _ { \mathrm { o p } } \mathrm { K L } ( \nu \| \gamma ) .\tag{100}
$$

Since $\gamma$ is Gaussian, Fernique’s theorem gives some $\eta > 0$ such that

$$
\int \exp \{ \eta  x   ^ { 2 } \} \gamma ( \mathrm { d } x ) < \infty .
$$

The Donsker-Varadhan variational formula therefore yields

$$
\eta \int \| x \| ^ { 2 } \nu ( \mathrm { d } x ) \le \mathrm { K L } ( \nu \| \gamma ) + \log \int \exp ( \eta \| x \| ^ { 2 } ) \gamma ( \mathrm { d } x ) < \infty .
$$

Thus both ν and $\gamma$ have finite second moments. Using the coupling $x \mapsto \left( x , \mathsf { P } _ { N } x \right)$

$$
W _ { 2 } ^ { 2 } ( \nu , \nu _ { N } ) \leq \int \left\| ( I - \mathsf { P } _ { N } ) x \right\| ^ { 2 } \nu ( \mathrm { d } x ) \longrightarrow 0 ,
$$

due to dominated convergence, because $\mathsf P _ { N } x \to x \mathrm { ~ a s ~ } N \to \infty$ for every $x \in H _ { \Sigma }$ and the integrand is bounded by $\| { \boldsymbol { x } } \| ^ { 2 }$ . The same argument gives $W _ { 2 } ( \gamma , \gamma _ { N } ) \longrightarrow 0$ . Hence, by the triangle inequality,

$$
W _ { 2 } ( \nu _ { N } , \gamma _ { N } ) \longrightarrow W _ { 2 } ( \nu , \gamma ) .
$$

Letting $N \to \infty$ in (100) proves

$$
W _ { 2 } ^ { 2 } ( \nu , \gamma ) \leq 2 \| \Sigma \| _ { \mathrm { o p } } \mathrm { K L } ( \nu \| \gamma ) .
$$

$\operatorname { I f } { \boldsymbol { \Sigma } } = 0$ , then $\gamma = \delta _ { 0 }$ , and $\nu \ll \gamma$ implies $\nu = \gamma$ , so the conclusion is immediate.

Lemma D.7 (KL approximation error of sparse GP). Under Assumptions 5.2 and 5.3, if $J \geq J _ { t } ^ { \dagger }$ then

$$
\mathbb { E } [ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) ] \lesssim t ^ { d / ( d + 2 \mathfrak { s } ) } .\tag{101}
$$

Proof. We apply Lemma 3 of Nieman et al. (2022): for every $h \in \mathbb { H }$

$$
\mathbb { E } [ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) ] \le \frac { 1 } { \sigma _ { \xi } ^ { 2 } } \left[ t \| f ^ { \star } - h \| ^ { 2 } + \| h \| _ { \mathbb { H } } ^ { 2 } \mathbb { E } [ \| R _ { t , J } \| _ { \mathrm { o p } } ] + \mathbb { E } [ \mathrm { t r } ( R _ { t , J } ) ] \right]\tag{102}
$$

where $\begin{array} { r } { R _ { t , J } : = \sum _ { i > J } \lambda _ { j } \psi _ { j } ( X _ { 1 : t } ) \psi _ { j } ( X _ { 1 : t } ) ^ { \intercal } } \end{array}$ . By uniform boundedness of the eigenfunctions in (42) and the eigenvalue decay in Assumption 5.3, we have the trace bound

$$
\mathrm { t r } ( R _ { t , J } ) = \sum _ { i = 1 } ^ { t } \sum _ { j > J } \lambda _ { j } \psi _ { j } ( X _ { i } ) ^ { 2 } \le B _ { \psi } ^ { 2 } t \sum _ { j > J } \lambda _ { j } \lesssim t J ^ { - 2 \mathfrak { s } / d } .\tag{103}
$$

Moreover, by applying Lemma 5 of Nieman et al. (2022), we have

$$
\begin{array} { r } { \mathbb { E } [ \| R _ { t , J } \| _ { \mathrm { o p } } ] \lesssim 1 + t J ^ { - 1 - 2 { \mathfrak { s } } / d } + t ^ { d / ( 2 \mathfrak { s } ) } J ^ { - 2 { \mathfrak { s } } / d } \log t . } \end{array}
$$

At $J \geq J _ { t } ^ { \dagger }$ this is bounded by a constant because ${ \mathfrak { s } } > d .$ . Now we take $h = \mathsf { P } _ { J _ { t } ^ { \dagger } } f ^ { \star }$ , the orthogonal projection onto span $( \psi _ { 1 } , \dots , \psi _ { J } )$ . Then (43) in Assumption 5.2 gives

$$
\| f ^ { \star } - h \| ^ { 2 } \leq C ( J _ { t } ^ { \dagger } ) ^ { - 2 \varsigma / d } , \qquad \| h \| _ { \mathbb H } ^ { 2 } = \sum _ { j \leq J _ { t } ^ { \dagger } } \frac { ( f _ { j } ^ { \star } ) ^ { 2 } } { \lambda _ { j } } \lesssim J _ { t } ^ { \dagger } .
$$

Substitution of these bounds into (102) proves (101).

Lemma D.8 (Crude KL approximation error of sparse GP). Under Assumptions 5.2 and 5.3,

$$
\mathbb { E } [ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) | X _ { 1 : t } ] \lesssim t\tag{104}
$$

for any given covariates $X _ { 1 : t } : = ( X _ { 1 } , \ldots , X _ { t } )$

Proof. By inspecting the proof of Lemma 3 of Nieman et al. (2022), we have the conditional expectation version:

$$
\mathbb { E } [ \mathrm { K L } ( Q _ { t } ^ { ( J ) } \| \Pi _ { t } ) | X _ { 1 : t } ] \lesssim \sum _ { i = 1 } ^ { t } ( f ^ { \star } ( X _ { i } ) - h ( X _ { i } ) ) ^ { 2 } + \| h \| _ { \mathbb { H } } ^ { 2 } \| R _ { t , J } \| _ { \mathrm { o p } } + \mathrm { t r } ( R _ { t , J } )\tag{105}
$$

where $\begin{array} { r } { R _ { t , J } : = \sum _ { j > J } \lambda _ { j } \psi _ { j } ( X _ { 1 : t } ) \psi _ { j } ( X _ { 1 : t } ) ^ { \intercal } } \end{array}$ . Take a constant function $h = 0$ . Then the desired result follows from combining two observations: $\| f ^ { \star } \| _ { \infty }$ is bounded under our assumption as we have shown in (97), and the trace term is bounded as t $\cdot ( R _ { t , J } ) \lesssim t$ as we have shown in (103). □

## D.5 Proof of Theorem 5.6

Lemma D.9 (Uniformly bounded predictive variance). Assume (42). Then for every $t \geq 0$ and every $J \geq 1$

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } v _ { \Pi _ { t } } ( x ) + \operatorname* { s u p } _ { x \in \mathcal { X } } v _ { Q _ { t } ^ { ( J ) } } ( x ) \lesssim 1 .\tag{106}
$$

Proof. The exact posterior covariance kernel is

$$
\mathsf { K } _ { t } ( x , x ^ { \prime } ) = \mathsf { K } _ { 0 } ( x , x ^ { \prime } ) - \mathsf { K } _ { 0 } ( x , X _ { 1 : t } ) ( \sigma _ { \xi } ^ { 2 } I + \mathsf { K } _ { 0 } ( X _ { 1 : t } , X _ { 1 : t } ) ) ^ { - 1 } \mathsf { K } _ { 0 } ( X _ { 1 : t } , x ^ { \prime } ) .
$$

Consequently,

$$
0 \leq v _ { \Pi _ { t } } ( x ) = \mathsf { K } _ { t } ( x , x ) \leq \mathsf { K } _ { 0 } ( x , x ) .
$$

For the variational posterior, (50) gives

$$
v _ { Q _ { t } ^ { ( J ) } } ( x ) = \mathsf { K } _ { t } ^ { ( J ) } ( x , x ) = \mathsf { K } _ { 0 } ( x , x ) - \psi _ { 1 : J } ( x ) ^ { \top } \left( \Lambda _ { J } - V _ { t , J } \right) \psi _ { 1 : J } ( x )
$$

Since

$$
V _ { t , J } ^ { - 1 } = \Lambda _ { J } ^ { - 1 } + \sigma _ { \xi } ^ { - 2 } \Psi _ { t , J } ^ { \top } \Psi _ { t , J } \succeq \Lambda _ { J } ^ { - 1 } ,
$$

we have $V _ { t } \preceq \Lambda _ { J }$ . Hence,

$$
0 \leq v _ { Q _ { t } ^ { ( J ) } } ( x ) \leq \mathsf { K } _ { 0 } ( x , x )
$$

Finally, Assumption 5.2 implies

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } \mathsf { K } _ { 0 } ( x , x ) \le B _ { \psi } ^ { 2 } \sum _ { j \ge 1 } \lambda _ { j } < \infty .
$$

The conclusion follows.

Proof of Theorem 5.6. With $J \gtrsim J _ { t } ^ { \dagger }$ , Substitution of the upper bounds in Theorems 5.4 and 5.5 and Lemma D.9 into Theorem 5.1 gives

$$
\begin{array} { r l r } {  { \Re \big ( ( Q _ { t } ^ { ( J ) } ) _ { t = 0 } ^ { T - 1 } \big ) \lesssim \sum _ { t = 1 } ^ { T } \{ \varepsilon _ { t } \alpha _ { t } + \alpha _ { t } ^ { 2 } + \varepsilon _ { t } ^ { 2 } \} } } \\ & { } & { \lesssim \displaystyle \sum _ { t = 1 } ^ { T } t ^ { - 2 \mathfrak { s } / ( d + 2 \mathfrak { s } ) } \lesssim T ^ { d / ( d + 2 \mathfrak { s } ) } , } \end{array}
$$

which is the desired result.

## References

Pierre Alquier. Non-exponentially weighted aggregation: regret bounds for unbounded loss functions. In International Conference on Machine Learning, pages 207–218. PMLR, 2021.

Pierre Alquier and James Ridgway. Concentration of tempered posteriors and of their variational approximations. The Annals of Statistics, 48(3):1475–1497, 2020.

Pierre Alquier, James Ridgway, and Nicolas Chopin. On the properties of variational approximations of gibbs posteriors. Journal of Machine Learning Research, 17(1):8374–8414, 2016.

Tamara Broderick, Nicholas Boyd, Andre Wibisono, Ashia C. Wilson, and Michael I. Jordan. Streaming variational Bayes. In Advances in Neural Information Processing Systems 26, pages 1727–1735. Curran Associates, Inc., 2013.

Nicolas Brosse, Alain Durmus, Eric Moulines, and Marcelo Pereyra. Sampling from a log-concave <sup>´</sup> distribution with compact support with proximal langevin monte carlo. In Proceedings of the 2017 Conference on Learning Theory, volume 65 of Proceedings of Machine Learning Research, pages 319–342. PMLR, 2017.

S´ebastien Bubeck, Ronen Eldan, and Joseph Lehec. Sampling from a log-concave distribution with projected langevin monte carlo. Discrete & Computational Geometry, 59(4):757–783, 2018. doi: 10.1007/s00454-018-9992-1.

Thang D. Bui, Cuong Nguyen, and Richard E. Turner. Streaming sparse gaussian process approximations. In Advances in Neural Information Processing Systems 30, pages 3299–3307, 2017.

David Burt, Carl Edward Rasmussen, and Mark van der Wilk. Rates of convergence for sparse variational gaussian process regression. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 862–871, 2019.

David Burt, Carl Edward Rasmussen, and Mark van der Wilk. Convergence of sparse variational inference in gaussian processes regression. Journal of Machine Learning Research, 21(131):1–63, 2020.

Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006. ISBN 9780521841085.

Venkat Chandrasekaran and Michael I. Jordan. Computational and statistical tradeofs via convex relaxation. Proceedings of the National Academy of Sciences, 110(13):E1181–E1190, 2013. doi: 10.1073/pnas.1302293110.

Badr-Eddine Ch´erief-Abdellatif, Pierre Alquier, and Mohammad Emtiyaz Khan. A generalization bound for online variational inference. In Proceedings of The Eleventh Asian Conference on Machine Learning, volume 101 of Proceedings of Machine Learning Research, pages 662–677. PMLR, 2019.

Bertrand S. Clarke and Andrew R. Barron. Information-theoretic asymptotics of Bayes methods. IEEE Transactions on Information Theory, 36(3):453–471, 1990. doi: 10.1109/18.54897.

Bo Dai, Niao He, Hanjun Dai, and Le Song. Provable Bayesian inference via particle mirror descent. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages 985–994. PMLR, 2016.

Arnak S Dalalyan. Theoretical guarantees for approximate sampling from smooth and log-concave densities. Journal of the Royal Statistical Society Series B: Statistical Methodology, 79(3):651–676, 2017.

Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential Monte Carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006. doi: 10.1111/j.1467-9868.2006.00553.x.

Persi Diaconis and Donald Ylvisaker. Conjugate priors for exponential families. The Annals of Statistics, 7(2):269–281, 1979.

Alain Durmus and Eric Moulines. Nonasymptotic convergence analysis for the unadjusted langevin <sup>´</sup> algorithm. The Annals of Applied Probability, 27(3):1551–1587, 2017. doi: 10.1214/16-AAP1238.

Simon Fischer and Ingo Steinwart. Sobolev norm learning rates for regularized least-squares algorithms. Journal of Machine Learning Research, 21(205):1–38, 2020.

Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2–3):169–192, 2007. doi: 10.1007/s10994-007-5016-8.

James Hensman, Nicolas Durrande, and Arno Solin. Variational fourier features for gaussian processes. Journal of Machine Learning Research, 18(151):1–52, 2018.

Matt Jones, Peter Chang, and Kevin Murphy. Bayesian online natural gradient (BONG). In Advances in Neural Information Processing Systems 37, pages 131104–131153, 2024. doi: 10. 52202/079017-4167.

Jungbin Jun and Ilsang Ohn. Adaptive bayesian online learning via expert aggregation. arXiv preprint, 2026.

Sham M. Kakade and Andrew Y. Ng. Online bounds for Bayesian algorithms. In Advances in Neural Information Processing Systems 17, pages 641–648, 2004.

Mohammad Emtiyaz Khan and H˚avard Rue. The bayesian learning rule. Journal of Machine Learning Research, 24(281):1–46, 2023.

Holden Lee, Oren Mangoubi, and Nisheeth K. Vishnoi. Online sampling from log-concave distributions. In Advances in Neural Information Processing Systems 32, 2019.

Jeyong Lee, Junhyeok Choi, and Minwoo Chae. Online bernstein-von mises theorem. Journal of Machine Learning Research, 27(49):1–124, 2026a.

Jeyong Lee, Junhyeok Choi, Dongguen Kim, and Minwoo Chae. Bayesian online learning in the one-pass regime: Frequentist validity and uncertainty quantification. arXiv preprint, 2026b.

Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Structured logconcave sampling with a restricted gaussian oracle. In Conference on Learning Theory, pages 2993–3050, 2021.

Alexander G. de G. Matthews, James Hensman, Richard E. Turner, and Zoubin Ghahramani. On sparse variational methods and the kullback–leibler divergence between stochastic processes. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages 231–239, 2016.

Dennis Nieman and Botond Szab´o. Adaptive sparse variational approximations for gaussian process regression. Bayesian Analysis, 2025.

Dennis Nieman, Botond Szabo, and Harry van Zanten. Contraction rates for sparse variational approximations in gaussian process regression. Journal of Machine Learning Research, 23(205): 1–26, 2022.

Dennis Nieman, Botond Szabo, and Harry van Zanten. Uncertainty quantification for sparse spectral variational approximations in gaussian process regression. Electronic Journal of Statistics, 17(2): 2250–2288, 2023. doi: 10.1214/23-EJS2155.

Ilsang Ohn and Lizhen Lin. Adaptive variational Bayes: Optimality, computation and applications. The Annals of Statistics, 52(1):335–363, 2024.

Garvesh Raskutti, Martin J Wainwright, and Bin Yu. Early stopping and non-parametric regression: an optimal data-dependent stopping rule. Journal of Machine Learning Research, 15(1):335–366, 2014.

Alessandro Rudi, Rafaello Camoriano, and Lorenzo Rosasco. Less is more: Nystr¨om computational regularization. In Advances in Neural Information Processing Systems, volume 28, 2015.

Alessio Spantini, Antti Solonen, Tiangang Cui, James Martin, Luis Tenorio, and Youssef Marzouk. Optimal low-rank approximations of bayesian linear inverse problems. SIAM Journal on Scientific Computing, 37(6):A2451–A2487, 2015.

Michel Talagrand. Transportation cost for gaussian and other product measures. Geometric and Functional Analysis, 6(3):587–600, 1996.

Michalis K. Titsias. Variational learning of inducing variables in sparse gaussian processes. In Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, volume 5 of Proceedings of Machine Learning Research, pages 567–574, 2009.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, New York, 2009. doi: 10.1007/b13794.

Dirk van der Hoeven, Tim van Erven, and Wojciech Kot lowski. The many faces of exponential weights in online learning. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 2067–2092. PMLR, 2018.

Tim van Erven, Peter D. Gr¨unwald, Nishant A. Mehta, Mark D. Reid, and Robert C. Williamson. Fast rates in statistical and online learning. Journal of Machine Learning Research, 16(54): 1793–1861, 2015.

Volodimir G Vovk. Aggregating strategies. In Proceedings of the third annual workshop on Computational learning theory, pages 371–386, 1990.

Qun Xie and Andrew R. Barron. Asymptotic minimax regret for data compression, gambling, and prediction. IEEE Transactions on Information Theory, 46(2):431–445, 2000. doi: 10.1109/18. 825803.

Yifan Yang, Chang Liu, and Zheng Zhang. Particle-based online Bayesian sampling. arXiv preprint, 2023. doi: 10.48550/arXiv.2302.14796.

Fengshuo Zhang and Chao Gao. Convergence rates of variational posterior distributions. The Annals of Statistics, 48(4):2180 – 2207, 2020.

Linda H. Zhao. Bayesian aspects of some nonparametric problems. The Annals of Statistics, 28(2): 532–552, 2000. doi: 10.1214/aos/1016218229.