# SAUSS: Stochastic Approximation with Unbiased Simulated Scores for Limited Dependent Variable Models

Sokbae Lee<sup>∗</sup> Yuan Liao<sup>†</sup> Myung Hwan Seo<sup>‡</sup> Youngki Shin<sup>§</sup>

August 25, 2026

## Abstract

Multinomial choice models allow flexible substitution patterns but become computationally demanding with many alternatives or observations. With a fixed perobservation simulation budget, simulated maximum likelihood introduces simulation bias, while each optimization step requires a full-sample likelihood evaluation. We propose Stochastic Approximation with Unbiased Simulated Scores (SAUSS), an averaged stochastic approximation based on conditionally unbiased mini-batch score estimates. Each iteration uses a fixed mini-batch regardless of sample size. For multinomial probit, accept–reject sampling provides exact conditional draws and unbiased score estimates for any fixed number of accepted draws. Under local conditions, asymptotic theory for the averaged estimator and the partial-sum process of the SAUSS iterates incorporates mini-batch and simulation variability and supports random-scaling and plug-in inference. In simulations and an application, SAUSS gives comparable results in less than 1% of the computation time of simulated maximum likelihood. SAUSS extends to limited dependent variable models with conditional-expectation score representations and exact conditional sampling.

Keywords: Accept–reject simulation; limited dependent variable models; multinomial probit; simulated scores; stochastic gradient descent.

## 1 Introduction

Statistical modeling of choices among multiple alternatives across a variety of fields requires balancing flexible dependence structures against computational tractability. The multinomial logit model remains a workhorse in part because its choice probabilities are available in closed form (McFadden, 1974; Train, 2009). Its independence-of-irrelevantalternatives structure, however, restricts how substitution across alternatives can respond to unobserved heterogeneity. Multinomial probit (MNP) allows the latent utility shocks to have a flexible covariance structure and thereby accommodates richer substitution patterns. The price of this flexibility is computation: each likelihood contribution is a multivariate normal region probability, with dimension increasing in the number of alternatives. This integration problem has long limited routine likelihood estimation of richly parameterized MNP models (e.g., Geweke et al., 1994; Train, 2009).

The computational burden has two distinct dimensions. As the number of alternatives J grows, each observation requires a higher-dimensional region probability and, after scale normalization, a flexible covariance specification contains a growing number of parameters. As the sample size n grows, conventional likelihood optimization repeatedly evaluates these costly contributions for the full sample. Conventional full-sample simulated maximum likelihood mitigates the first problem by replacing each probability with a smooth simulator, most commonly one based on the Geweke–Hajivassiliou–Keane (GHK) construction (e.g., B¨orsch-Supan and Hajivassiliou, 1993; Hajivassiliou et al., 1996). It does not by itself address the second problem created by a large sample. Moreover, placing an estimated probability inside a logarithm creates finite-simulation bias even when the probability simulator itself is unbiased. Under conventional simulated-likelihood asymptotics, the number of draws per observation must therefore increase with n, while every optimization step continues to process all n observations. Simulation efort per contribution to the log likelihood and the number of contributions per step can thus grow together.

Classical simulation-based work addresses this integration problem in several ways. McFadden (1989) introduces the method of simulated moments for discrete-response models, avoiding direct numerical integration of choice probabilities. Keane (1994) develops a computationally practical extension for panel limited dependent variable (LDV) models by factorizing moment conditions in terms of simulated transition probabilities. Geweke et al. (1994) compare simulated maximum likelihood, simulated moments, and Bayesian data augmentation directly for MNP. Together with the development and evaluation of GHK and related multivariate-normal simulators (B¨orsch-Supan and Hajivassiliou, 1993; Hajivassiliou et al., 1996), these contributions make simulation-based estimation of flexible probit models practical at moderate sample sizes but leave a basic large-sample tension: accurate simulation and repeated full-sample computation remain costly as n grows.

The closest antecedent to our approach is the method of simulated scores (MSS) of Hajivassiliou and McFadden (1998). Rather than inserting an estimated probability into a criterion, MSS represents the likelihood score as a conditional expectation of a complete-data score. Exact conditional draws obtained by accept–reject simulation yield an unbiased simulated score for any fixed number of accepted draws. MSS also develops continuous score simulators based on recursive conditioning and Gibbs resampling. The unbiased accept–reject score simulator is generally discontinuous in the parameter, whereas the continuous constructions may require increasing simulation efort to eliminate approximation bias. A finite Gibbs chain provides an exact conditional draw if initialized in stationarity; from a nonstationary initialization, its distribution converges to the target as the chain length tends to infinity. Because MSS is formulated as a fullsample simulated-score estimating equation, conventional numerical solution methods face a tradeof between unbiased score simulation and smoothness.

This paper develops SAUSS—Stochastic Approximation with Unbiased Simulated Scores—to sidestep that tradeof. SAUSS uses exact conditional accept–reject draws to construct an unbiased, potentially discontinuous score estimate inside a stochastic approximation recursion. This supplies a conditionally unbiased update direction with suitable moment properties without requiring each realization of the simulator to be smooth in the parameter. At each iteration, SAUSS draws a mini-batch of m observations, where m can be much smaller than n and may equal one, generates fresh conditional score draws for those observations, and updates the parameter using the resulting average negative score. The proposed estimator averages the iterates in the manner of Polyak and Juditsky (1992). With m fixed, the number of observations processed per update does not grow with n. SAUSS thus retains the exact likelihood score as its target while avoiding both the logarithm of a simulated probability and repeated full-sample updates.

Relative to existing methods, SAUSS makes three contributions. First, it embeds the exact conditional simulation and unbiased score construction of MSS in mini-batch stochastic approximation with iterate averaging. Second, it gives an explicit MNP implementation with a scale-normalized, otherwise unrestricted covariance parameterization and evaluates its computational performance as the number of alternatives grows. Third, it develops inference that jointly accounts for mini-batch and score-simulation variation. The framework extends to other LDV models admitting conditional-expectation score representations and exact conditional simulation.

Under the reported configurations, SAUSS delivers results broadly comparable to GHK-based simulated maximum likelihood, while GHK requires more than 100 times as much computation time in both the Monte Carlo benchmark and the empirical application. With the same tuning constants, SAUSS remains computationally feasible as the choice set expands far beyond the benchmark design. Although the reported runtime ratios depend on the simulation budgets and implementations, their consistent pattern illustrates the practical value of the SAUSS architecture. The statistical theory treats the variation created by that architecture as part of the inferential problem. Under local regularity and initialization conditions, we establish a linear representation and functional central limit theorem for the partial-sum path in the following ϵ-sense: for any $\epsilon > 0$ 2 the learning-rate constant can be chosen so that the recursion remains in the local neighborhood with probability at least $1 - \epsilon - o ( 1 )$ , and the conclusions hold on that event. The endpoint yields, in the same ϵ-sense, the corresponding normal approximation for the averaged estimator. The limit separates variation generated by mini-batch sampling from variation generated by score simulation and supports random-scaling and plug-in procedures that account for both sources of algorithmic variation. The closest contemporaneous work is Frazier et al. (2026), who propose Stochastically Estimated Gradient Ascent (SEGA). Both SEGA and SAUSS combine conditional-score simulation and iterate averaging. Drawing on two simulation routes already developed in MSS, SEGA uses Gibbs-based score simulation in full-sample updates, whereas SAUSS uses score simulation based on exact accept–reject conditional draws in mini-batch updates. SEGA’s theory assumes conditional unbiasedness, which is satisfied by exact conditional draws or by a Gibbs chain initialized in stationarity. The SAUSS inferential limit retains variation from mini-batch sampling and score simulation. Under SEGA’s regularity conditions and an iteration schedule that makes optimization error negligible relative to n<sup>−1/2</sup> sampling error, its averaged estimator is first-order asymptotically equivalent to the infeasible maximum likelihood estimator (MLE) based on the exact likelihood and therefore inherits its limiting distribution. The two approaches thus use the same conditional-score lineage from MSS but adopt diferent computational regimes and asymptotic approximations.

A parallel Bayesian literature avoids direct likelihood evaluation through latent-data augmentation. Albert and Chib (1993) provide the foundational framework for binary and polychotomous probit models. For MNP, McCulloch and Rossi (1994) use latent Gaussian utilities in a Gibbs posterior simulator, and Imai and van Dyk (2005) develop marginal data augmentation. Recent work develops scalable Bayesian estimation using structured covariance models (Loaiza-Maya and Nibbering, 2022) and fast variational approximations (Loaiza-Maya and Nibbering, 2023). These methods target posterior inference, whereas SAUSS targets the exact likelihood score and delivers frequentist inference for the averaged stochastic approximation.

Other recent approaches modify the likelihood approximation or the computation of choice probabilities, including approximate composite marginal likelihood (Bhat, 2011), expectation propagation within an EM algorithm (Ding et al., 2024), and a neuralnetwork emulator for correlated discrete-choice probabilities (Huch and Keane, 2026). SAUSS instead retains the exact likelihood score as its target and changes the computational and inferential architecture in which unbiased simulated-score contributions are subsampled across observations, accumulated over iterations, and used for inference.

Finally, SAUSS connects the classical stochastic approximation recursion of Robbins and Monro (1951) and iterate averaging of Polyak and Juditsky (1992) to simulationbased likelihood estimation. Recent stochastic-approximation work provides scalable estimation and inference for smooth-loss models, with applications to linear mean and logistic regression (Lee et al., 2022); for large-scale linear quantile regression (Lee et al., 2025); for semiparametric models (Chen et al., 2026); and for overidentified moment condition models through SGMM (Chen et al., 2025) and SLIM (Chen et al., 2025). In SAUSS, however, each update contains additional conditional-simulation variation because the exact likelihood score is not directly available. The functional limit and variance procedures explicitly accommodate this additional source of variation without smoothing the accept–reject mechanism.

The rest of the paper is organized as follows. Section 2 presents the SAUSS framework, and Sections 3 and 4 develop the LDV score identity and its MNP implementation. Sections 5 and 6 establish the asymptotic theory and inference procedures. Sections 7 and 8 report the Monte Carlo evidence and empirical application, and Section 9 concludes. Proofs of the asymptotic results and the supporting lemmas are collected in Appendix A.

## 2 The SAUSS Framework

We propose estimation and inference using stochastic gradient descent, which recursively updates the estimator as follows:

$$
\theta _ { t } = \theta _ { t - 1 } - \gamma _ { t } \hat { G } _ { t } ( \theta _ { t - 1 } )
$$

with a suitably chosen learning rate $\gamma _ { t }$ . Here, $\hat { G } _ { t } ( \theta _ { t - 1 } )$ is the negative of a simulated likelihood score. Unlike the usual stochastic gradient descent inference framework, directly evaluating the score from the log-likelihood is a challenging task in limited dependent variable models. This section presents the conceptual framework for SAUSS. It first reviews why smooth simulated likelihoods are attractive but biased at a fixed simulation budget, then explains how MSS supplies unbiased simulated score contributions, and finally sets out the resulting stochastic approximation recursion in descent form.

## 2.1 Smooth Simulated Likelihood and Bias

Let $P _ { i } ( \theta )$ denote the probability of the observed outcome for observation $i ,$ and let

$$
\mathcal { L } _ { n } ( \theta ) = \sum _ { i = 1 } ^ { n } \log P _ { i } ( \theta )
$$

be the exact log-likelihood. Simulated maximum likelihood replaces $P _ { i } ( \theta )$ with a simulated probability estimator $\hat { P } _ { i } ( \theta )$ and maximizes

$$
{ \hat { \mathcal { L } } } _ { n } ( \theta ) = \sum _ { i = 1 } ^ { n } \log { \hat { P } } _ { i } ( \theta ) .
$$

GHK-based simulated maximum likelihood is a prominent implementation of this strategy for MNP (e.g., Bolduc, 1999). Much of the classical simulation literature therefore places heavy weight on simulators that are continuous, diferentiable, bounded away from zero, and suitable for full-sample numerical optimization $( \mathrm { e . g . }$ , Train, 2009). Smoothness is valuable because the simulated object is treated as an objective function to be maximized.

The dificulty is that the smooth simulated objective is generally not an unbiased version of the exact log-likelihood. Even when $\hat { P } _ { i } ( \theta )$ is unbiased for $P _ { i } ( \theta )$ , the nonlinear transformation log $\hat { P } _ { i } ( \theta )$ is generally biased. A related problem arises for the simulated score. If $\hat { P } _ { i } ( \theta ; { \mathbf { u } } )$ denotes a simulated probability computed from random draws u, then, for a fixed finite simulation budget,

$$
\mathbb { E } _ { \mathbf { u } } \left[ \nabla _ { \theta } \log \hat { P } _ { i } ( \theta ; \mathbf { u } ) \right] \neq \nabla _ { \theta } \log P _ { i } ( \theta ) \quad \quad \mathrm { i n ~ g e n e r a l } ,\tag{1}
$$

whenever the derivative and expectation are well defined. Here $\mathbb { E } _ { \mathbf { u } }$ denotes expectation over the simulation draws conditional on the observation.

In full-sample simulated maximum likelihood, this approximation error is controlled by increasing the simulation budget as the sample size grows. That strategy is less attractive for stochastic approximation because a growing simulation budget also raises the cost of every mini-batch evaluation. The alternative pursued here is to simulate an unbiased likelihood-score contribution directly. The number of simulated score draws used in each evaluation can then remain fixed. In accept–reject implementations, the number of proposals required to obtain those draws may nevertheless be random.

## 2.2 Unbiased Simulated Scores

The method of simulated scores (MSS) proposed by Hajivassiliou and McFadden (1998) changes the object being simulated. Instead of simulating $P _ { i } ( \theta )$ and then taking a logarithm, it simulates the score

$$
s _ { i } ( \theta ) = \nabla _ { \theta } \log P _ { i } ( \theta )
$$

directly. When the simulator supplies exact conditional draws, or otherwise provides an unbiased score construction, the resulting estimate can satisfy

$$
\mathbb { E } _ { \mathbf { u } } \{ \hat { s } _ { i } ( \theta ) \mid D _ { i } \} = s _ { i } ( \theta ) .
$$

In Hajivassiliou and McFadden (1998), the MSS estimator is formulated as a full-sample Z-estimator based on simulated scores, that is, as a solution to

$$
n ^ { - 1 } \sum _ { i = 1 } ^ { n } { \hat { s } } _ { i } ( \theta ) = 0 .\tag{2}
$$

This root-finding problem can be dificult when $\theta \mapsto \hat { s } _ { i } ( \theta )$ is discontinuous, as often occurs with accept–reject constructions.

SAUSS retains the unbiased simulated score but changes how it is used. At each

iteration, the negative simulated score provides a conditionally unbiased stochastic direction for the gradient of the negative log-likelihood. This direction need not itself be the derivative of a smooth simulated objective.

## 2.3 SAUSS as Stochastic Approximation

We write the recursion in descent form for the negative log-likelihood. Given the observed sample $D _ { 1 : n }$ , define the individual loss $\ell _ { i } ( \theta ) = - \log P _ { i } ( \theta )$ and the empirical criterion $\begin{array} { r } { Q _ { n } ( \theta ) = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \ell _ { i } ( \theta ) } \end{array}$ . Let $\hat { \theta } _ { n }$ denote a solution of the sample score equation $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } s _ { i } ( \hat { \theta } _ { n } ) = 0 } \end{array}$

Let $\psi _ { i } ( \boldsymbol \theta , { \mathbf { u } } )$ be a one-draw simulated score satisfying $\mathbb { E } _ { \mathbf { u } } \{ \psi _ { i } ( \boldsymbol { \theta } , \mathbf { u } ) \mid D _ { i } \} = s _ { i } ( \boldsymbol { \theta } )$ , and write $g _ { i } ( \theta , { \bf u } ) = - \psi _ { i } ( \theta , { \bf u } )$ . Thus $g _ { i }$ is a conditionally unbiased stochastic direction for the individual loss gradient $\nabla _ { \theta } \ell _ { i } ( \theta ) = - s _ { i } ( \theta )$

Set $\theta _ { 1 } = \theta _ { \mathrm { i n i t } }$ . For iterations $t = 2 , \ldots , T$ , draw ordered mini-batch indices $I _ { t 1 } , \ldots , I _ { t m }$ independently and uniformly from $\{ 1 , \ldots , n \}$ , and generate the random inputs $\mathbf { u } _ { t b r } , r =$ $1 , \ldots , R$ , independently for every batch position, score draw, and iteration. The minibatch direction and SAUSS update are

$$
\begin{array} { c } { { \displaystyle \hat { G } _ { t } ( \theta _ { t - 1 } ) = \frac { 1 } { m R } \sum _ { b = 1 } ^ { m } \sum _ { r = 1 } ^ { R } g _ { I _ { t b } } ( \theta _ { t - 1 } , { \bf u } _ { t b r } ) , } } \\ { { \displaystyle \theta _ { t } = \theta _ { t - 1 } - \gamma _ { t } \hat { G } _ { t } ( \theta _ { t - 1 } ) . } } \end{array}\tag{3}
$$

Here $\gamma _ { t }$ is a decreasing step size. If $\mathcal { F } _ { t - 1 }$ denotes the information available immediately before the draws at iteration $t ,$ then E $\{ \hat { G } _ { t } ( \theta _ { t - 1 } ) \mid D _ { 1 : n } , \mathcal { F } _ { t - 1 } \} = \nabla _ { \theta } Q _ { n } ( \theta _ { t - 1 } )$ . Conditional on the sample, the ideal recursion therefore targets $\hat { \theta } _ { n }$

The reported estimator is the Polyak–Ruppert average

$$
\bar { \theta } _ { T } = \frac { 1 } { T - T _ { 0 } } \sum _ { t = T _ { 0 } + 1 } ^ { T } \theta _ { t } ,\tag{4}
$$

where $T _ { 0 }$ is the averaging start index. Thus T is the final iterate index, the run contains $T$ stored iterates including the initial estimator, and the recursion performs $T - 1$ stochastic updates.

The key statistical implication is that R need not diverge in order to eliminate simulation bias from the ideal recursion: for every fixed $R \geq 1$ , the stochastic direction remains conditionally unbiased. Holding R fixed does not, however, eliminate simulation variance. The magnitude of the resulting algorithmic uncertainty depends jointly on $T _ { \ast }$ m, and R.

Remark 1 (Accepted draws and computational cost). Fixing R fixes the number of accepted score draws, not the number of proposals. For an accept–reject implementation with acceptance probability $a _ { i } ( \theta )$ , the expected number of proposals required to obtain R accepted draws is $R / a _ { i } ( \theta )$ . Thus the exact recursion has random computational cost even when R is fixed.

The next section states the LDV score identity that delivers the unbiased score contributions.

## 3 The LDV Score Identity

This section states the identity used in Hajivassiliou and McFadden (1998). The key point is to represent the observed outcome as a fixed latent-region event, so that diferentiating the log probability produces a conditional expectation of a complete-data score.

Write the observed data as $D _ { i } = \left( Y _ { i } , X _ { i } \right)$ , where $Y _ { i }$ is the limited dependent variable outcome and $X _ { i }$ collects the observed conditioning variables. Conditional on $X _ { i }$ , suppose that $Y _ { i }$ is generated by an underlying latent vector $Y _ { i } ^ { * }$ . For the realized outcome $Y _ { i } =$ $y _ { i }$ , choose latent coordinates in which $\{ Y _ { i } = y _ { i } \}$ is equivalent to the fixed-region event $Y _ { i } ^ { * } ~ \in ~ { \mathcal { D } } _ { i }$ , where ${ \mathcal { D } } _ { i } = { \mathcal { D } } ( y _ { i } , X _ { i } )$ . The region may depend on the observed outcome and conditioning variables, but it does not vary with θ. All probabilities and densities below are conditional on $X _ { i }$ , with this conditioning suppressed to simplify notation. The parameter enters through the conditional latent density $f _ { i } ( y ; \theta )$ , and the likelihood contribution is

$$
P _ { i } ( \theta ) = \int _ { { \mathcal { D } } _ { i } } f _ { i } ( y ; \theta ) d y .\tag{5}
$$

Lemma 1 (Hajivassiliou–McFadden identity). Fix $i ,$ condition on $X _ { i }$ , and fix θ in the interior of the parameter space. Assume that the observed outcome $Y _ { i } = y _ { i }$ is equivalent to $Y _ { i } ^ { * } \in { \mathcal { D } } _ { i }$ for a measurable set $\mathcal { D } _ { i }$ that does not depend on $\theta .$ . Assume also that $P _ { i } ( \theta ) > 0$ and that $f _ { i } ( y ; \theta ) > 0$ for almost every $y \in { \mathcal { D } } _ { i }$ . Finally, suppose that there is a neighborhood $\mathcal { N } _ { \theta }$ of θ such that, for almost every $y \in { \mathcal { D } } _ { i }$ , the map $\vartheta \mapsto f _ { i } ( y ; \vartheta )$ is diferentiable on $\mathcal { N } _ { \theta }$ and there exists an integrable envelope $M _ { i }$ such that

$$
\operatorname* { s u p } _ { \vartheta \in N _ { \theta } } \| \nabla _ { \vartheta } f _ { i } ( y ; \vartheta ) \| \leq M _ { i } ( y ) , \qquad \int _ { D _ { i } } M _ { i } ( y ) d y < \infty .
$$

Then

$$
\nabla _ { \theta } \log P _ { i } ( \theta ) = \mathbb { E } _ { \theta } \left[ \nabla _ { \theta } \log f _ { i } ( Y _ { i } ^ { * } ; \theta ) \mid Y _ { i } ^ { * } \in { \mathcal { D } } _ { i } \right] .\tag{6}
$$

Proof. By the stated assumptions,

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } P _ { i } ( \theta ) = \int _ { \mathcal { D } _ { i } } \nabla _ { \theta } f _ { i } ( y ; \theta ) d y } \\ { \displaystyle = \int _ { \mathcal { D } _ { i } } \left[ \nabla _ { \theta } \log f _ { i } ( y ; \theta ) \right] f _ { i } ( y ; \theta ) d y . } \end{array}\tag{7}
$$

Dividing (7) by $P _ { i } ( \theta )$ gives

$$
\nabla _ { \boldsymbol { \theta } } \log { P _ { i } ( \boldsymbol { \theta } ) } = \int _ { \mathcal { D } _ { i } } \nabla _ { \boldsymbol { \theta } } \log f _ { i } ( \boldsymbol { y } ; \boldsymbol { \theta } ) \frac { f _ { i } ( \boldsymbol { y } ; \boldsymbol { \theta } ) } { P _ { i } ( \boldsymbol { \theta } ) } d \boldsymbol { y } ,
$$

which is (6) because $f _ { i } ( y ; \theta ) / P _ { i } ( \theta )$ is the conditional density of $Y _ { i } ^ { * }$ given $Y _ { i } ^ { * } \in { \mathcal { D } } _ { i }$ □

Remark 2 (Binary probit and fixed regions). For binary probit, let $Y _ { i } ^ { * } = X _ { i } ^ { \top } \beta + \varepsilon _ { i }$ , with $\varepsilon _ { i } \sim \mathcal { N } ( 0 , 1 )$ , and let $Y _ { i } = \mathbb { I } \{ Y _ { i } ^ { * } > 0 \}$ . In latent-utility coordinates, the observed outcome corresponds to the fixed region $( 0 , \infty )$ when $Y _ { i } = 1$ and to $( - \infty , 0 ]$ when $Y _ { i } = 0$ . In error coordinates, the same event has a boundary that depends on $\beta .$ . The identity is therefore applied in latent-utility coordinates, where the region is fixed and the parameter enters through the density.

Lemma 1 identifies the simulation target. If we can draw independent exact realizations $Y _ { i r } ^ { * } ( \theta )$ from the conditional distribution of $Y _ { i } ^ { * }$ given $Y _ { i } ^ { * } \in { \mathcal { D } } _ { i }$ , then

$$
\hat { s } _ { i } ( \theta ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \nabla _ { \theta } \log f _ { i } ( y ; \theta ) | _ { y = Y _ { i r } ^ { * } ( \theta ) }\tag{8}
$$

is unbiased for $s _ { i } ( \theta )$ for any fixed $R \geq 1$ . For an accept–reject simulator, when the same underlying proposal sequence is used across parameter values, the simulated-score map may be discontinuous in θ because the identity of the accepted proposal can change discretely as the parameter moves. For full-sample numerical root finding, this discontinuity can create a dificult computational problem. For stochastic approximation, conditional unbiasedness is the central property, together with the moment and stability conditions stated later.

The identity applies directly to the region-probability contributions in canonical LDV settings emphasized by Hajivassiliou and McFadden (1998), including panel probit and multinomial choice models with correlated latent errors. In Tobit-type models, it applies to censored contributions, while uncensored density contributions can be scored directly.

We build on the Hajivassiliou–McFadden identity and focus on LDV models in which the score admits a conditional-expectation representation and for which either an exact unbiased score simulator is available or a practical simulator is available whose approximation error can be diagnosed. Section 4 verifies this construction explicitly for MNP under the covariance normalization used by the algorithm.

## 4 SAUSS for Multinomial Probit

This section specializes the LDV score identity to MNP and describes how SAUSS can be implemented for MNP.

## 4.1 Model and Choice Regions

Consider n agents choosing among J alternatives. The latent utility for agent i and alternative j is

$$
U _ { i j } = x _ { i j } ^ { \top } \beta + \varepsilon _ { i j } , \quad j = 1 , \ldots , J ,\tag{9}
$$

where $x _ { i j } \in \mathbb { R } ^ { K } , \beta \in \mathbb { R } ^ { K }$ , and $\varepsilon _ { i } : = ( \varepsilon _ { i 1 } , \ldots , \varepsilon _ { i J } ) ^ { \top }$ satisfies $\varepsilon _ { i } \sim \mathcal { N } ( 0 , \Sigma )$ . Agent i chooses $y _ { i } = j { \mathrm { ~ i f ~ } } U _ { i j } > U _ { i k }$ for all $k \neq j$

Utility levels are not identified, so we work in diferences relative to alternative 1. For $j = 2 , \ldots , J ,$ define

$$
\Delta U _ { i j } : = U _ { i j } - U _ { i 1 } = \left( x _ { i j } - x _ { i 1 } \right) ^ { \top } \beta + \nu _ { i j } , \qquad \nu _ { i j } : = \varepsilon _ { i j } - \varepsilon _ { i 1 } .\tag{10}
$$

Let $d : = \boldsymbol { J } - 1 , \Delta \boldsymbol { U } _ { i } : = ( \Delta U _ { i 2 } , \ldots , \Delta U _ { i J } ) ^ { \top } , \boldsymbol { \nu } _ { i } : = ( \nu _ { i 2 } , \ldots , \nu _ { i J } ) ^ { \top }$ , and let $\Delta X _ { i }$ be the $d \times K$ matrix whose $( j - 1 )$ th row, corresponding to alternative $j ,$ is $( x _ { i j } - x _ { i 1 } ) ^ { \top }$ . Then

$$
\Delta U _ { i } = \Delta X _ { i } \beta + \nu _ { i } , \qquad \nu _ { i } \sim \mathcal { N } ( 0 , \Omega ) .
$$

The covariance matrix of diferenced errors satisfies

$$
\Omega _ { j - 1 , k - 1 } = \Sigma _ { j k } + \Sigma _ { 1 1 } - \Sigma _ { j 1 } - \Sigma _ { 1 k } , \qquad j , k = 2 , \dots , J .
$$

Thus the first coordinate of $\nu _ { i }$ is $\nu _ { i 2 }$ , and the first row and column of Ω correspond to alternative 2 relative to the base alternative.

The base alternative is a labeling convention. It fixes the coordinate system for utility diferences but does not restrict substitution patterns.

Let $W _ { i } : = \Delta U _ { i }$ , let $w : = ( w _ { 2 } , \ldots , w _ { J } ) ^ { \intercal } \in \mathbb { R } ^ { d }$ denote a realization of $W _ { i }$ , and set $w _ { 1 } : = 0$ for the base alternative. The observed choice corresponds to a fixed region of the diferenced latent-utility space. If the base alternative is chosen,

$$
\mathcal D _ { 1 } : = \left\{ w \in \mathbb R ^ { d } \vert w _ { j } < 0 \mathrm { ~ f o r ~ a l l ~ } j = 2 , \dots , J \right\} .
$$

For a non-base alternative $j = 2 , \dots , J$

$$
\mathcal D _ { j } : = \left\{ w \in \mathbb R ^ { d } \ | \ w _ { j } > 0 \ \mathrm { a n d } \ w _ { j } > w _ { k } \ \mathrm { f o r ~ a l l } \ k = 2 , \dots , J , \ k \neq j \right\} .
$$

Thus the probability of choice j is the integral of the $\mathcal { N } ( \Delta X _ { i } \beta , \Omega )$ density of $\Delta U _ { i }$ over the fixed region $\mathcal { D } _ { j }$ :

$$
P _ { i j } ( \theta ) : = \int _ { \mathscr { D } _ { j } } \phi _ { d } ( w ; \Delta X _ { i } \beta , \Omega ) d w ,\tag{11}
$$

where θ collects $\beta$ and the free covariance parameters specified below, and $\phi _ { d } ( \cdot ; \mu , \Omega )$ denotes the d-variate normal density.

## 4.2 Covariance Normalization

The model is identified only up to scale. For any $c > 0$ , the transformation

$$
( \beta , \Omega ) \mapsto ( c \beta , c ^ { 2 } \Omega )
$$

leaves all choice probabilities unchanged. The main SAUSS implementation uses the covariance Cholesky normalization

$$
\Omega : = L L ^ { \top } , \qquad L : = \left( \begin{array} { c c c c } { { 1 } } & { { } } & { { } } & { { } } \\ { { a _ { 2 1 } } } & { { e ^ { \lambda _ { 2 } } } } & { { } } & { { } } \\ { { \vdots } } & { { \ddots } } & { { \ddots } } & { { } } \\ { { a _ { d 1 } } } & { { \ddots } } & { { a _ { d , d - 1 } } } & { { e ^ { \lambda _ { d } } } } \end{array} \right) .\tag{12}
$$

Thus $L _ { 1 1 } = 1$ fixes the standard deviation of $\nu _ { i 2 }$ at one and thereby fixes the scale; the remaining diagonal elements are positive by construction, and the strictly lower-triangular elements are unrestricted. When $d = 1$ , this normalization reduces to $\Omega = 1$ , as in binary probit.

Let

$$
\begin{array} { r l r } { \theta : = ( \beta ^ { \top } , \theta _ { L } ^ { \top } ) ^ { \top } , } & { { } } & { \theta _ { L } : = \operatorname { f r e e } ( L ) . } \end{array}
$$

Here $\mathrm { f r e e } ( L )$ stacks the strictly lower-triangular entries of $L$ and the log-diagonal coordinates $\lambda _ { k } : = \log L _ { k k } , k = 2 , \ldots , d .$ , excluding the fixed element $L _ { 1 1 }$

When $d = 1 , \theta _ { L }$ is empty and the MNP normalization reduces to the usual binaryprobit scale normalization.

## 4.3 Likelihood and Descent Target

Let $d _ { i j } : = \mathbb { I } \{ y _ { i } = j \}$ , where $\mathbb { I } \{ \cdot \}$ denotes the indicator function. The sample log-likelihood is

$$
\mathcal { L } _ { n } ( \theta ) : = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { J } d _ { i j } \log P _ { i j } ( \theta ) .\tag{13}
$$

Because $P _ { i j } ( \theta )$ lacks a closed form, simulated maximum likelihood replaces it by a simulated probability estimate. If $\tilde { P } _ { i j } ^ { ( r ) } ( \theta ; u _ { i } ^ { ( r ) } )$ is the contribution from the $r { \mathrm { - t h } }$ sequence of random draws, the standard simulated objective is

$$
\hat { \mathcal { L } } _ { n } ( \boldsymbol { \theta } ) : = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { J } d _ { i j } \log \left( \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \tilde { P } _ { i j } ^ { ( r ) } ( \boldsymbol { \theta } ; \boldsymbol { u } _ { i } ^ { ( r ) } ) \right) .\tag{14}
$$

SAUSS does not maximize (14). It targets the score of (13) directly through simulation and uses the negative simulated score as a descent direction for the negative log-likelihood.

## 4.4 Unbiased Simulated Scores

The score for one observed choice is

$$
s _ { i } ( \theta ) : = \nabla _ { \theta } \log P _ { i y _ { i } } ( \theta ) .
$$

By Lemma 1,

$$
s _ { i } ( \theta ) = \mathbb { E } _ { \theta } \left[ \nabla _ { \theta } \log \phi _ { d } ( W _ { i } ; \Delta X _ { i } \beta , \Omega ) \ | \ W _ { i } \in \mathcal { D } _ { y _ { i } } \right] .
$$

The accept–reject simulator draws $\nu _ { i } \sim \mathcal { N } ( 0 , \Omega )$ , forms $W _ { i } : = \Delta X _ { i } \beta + \nu _ { i }$ , and keeps the draw if $W _ { i } \in \mathcal { D } _ { y _ { i } }$ . Thus an accepted draw has distribution

$$
\nu _ { i } \mid \{ \Delta X _ { i } \beta + \nu _ { i } \in \mathcal { D } _ { y _ { i } } \} .
$$

Conditional on acceptance, $W _ { i }$ has the latent normal distribution restricted to the observed choice region, so the average latent-score contribution over accepted draws is an unbiased estimate of $s _ { i } ( \theta )$ . This statement is exact for any fixed number $R \geq 1$ of accepted draws, provided the simulator runs until those accepted draws are obtained.

In the one-draw score formulas below, $\nu _ { i }$ denotes a generic accepted draw from this observation-specific conditional distribution, equivalently $\nu _ { i } = W _ { i } - \Delta X _ { i } \beta$ . Let $Q : = \Omega ^ { - 1 }$ for notational convenience. We suppress the dependence of this accepted draw on $\beta$ and Ω when no confusion arises.

The one-draw complete-data score contribution for $\beta$ is

$$
s _ { \beta i } ^ { \mathrm { c } } : = \Delta X _ { i } ^ { \top } Q \nu _ { i } .\tag{15}
$$

The matrix score with respect to Ω is

$$
S _ { \Omega i } ^ { \mathrm { c } } : = \frac { 1 } { 2 } Q ( \nu _ { i } \nu _ { i } ^ { \top } - \Omega ) Q .\tag{16}
$$

Because $\Omega = L L ^ { \top }$ , the corresponding score for the Cholesky factor is

$$
\begin{array} { r } { S _ { L i } ^ { \mathrm { c } } : = 2 S _ { \Omega i } ^ { \mathrm { c } } L = Q ( \nu _ { i } \nu _ { i } ^ { \top } - \Omega ) Q L . } \end{array}\tag{17}
$$

The descent implementation uses the negative of these score contributions. For any $d \times d$ matrix A, write $\mathcal { C } _ { \theta _ { L } } ( A ; L )$ for the vector that keeps the free lower-triangular entries of A, replaces each free diagonal entry $A _ { k k } , k = 2 , \ldots , d ,$ by $A _ { k k } L _ { k k }$ to account for the log-diagonal parameterization $\lambda _ { k } : = \log L _ { k k } , k = 2 , \ldots , d .$ , and drops the fixed coordinate (1, 1).

Let $\nu _ { i r } , r = 1 , \ldots , R .$ , denote independent accepted draws from the conditional distribution above for observation i. Averaging the negative score contributions over these

draws gives

$$
\widehat { g } _ { \beta i } ( \theta ) : = - \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \Delta X _ { i } ^ { \top } Q \nu _ { i r } , \qquad \widehat { g } _ { L i } ( \theta ) : = - \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \mathcal { C } _ { \theta _ { L } } \{ Q ( \nu _ { i r } \nu _ { i r } ^ { \top } - \Omega ) Q L ; L \} .\tag{18}
$$

Write

$$
\widehat { g } _ { i } ( \boldsymbol { \theta } ) : = \left( \widehat { g } _ { \beta i } ( \boldsymbol { \theta } ) ^ { \top } , \widehat { g } _ { L i } ( \boldsymbol { \theta } ) ^ { \top } \right) ^ { \top }
$$

for the loss-gradient contribution in the full parameter vector.

## 4.5 SAUSS Algorithm

The main recursion uses a scalar decreasing learning rate,

$$
\gamma _ { t } : = \gamma _ { 0 } ( t - 1 ) ^ { - a } , \qquad a \in ( 1 / 2 , 1 ) , \qquad t \geq 2 .\tag{19}
$$

This is the standard decreasing-rate form used by the theory.

SAUSS resamples the simulation draws whenever an observation enters a score evaluation. This difers from simulated likelihood implementations that use common random numbers to stabilize a fixed simulated likelihood objective. Here the algorithm does not optimize such a fixed simulated objective. It uses simulation to produce an unbiased stochastic direction for the gradient of the exact average negative log-likelihood, so the simulation randomness is renewed along the stochastic approximation path.

Algorithm 1 takes as data input the diferenced covariates and observed choices $\{ ( \Delta X _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ . The algorithmic parameter vector is $\theta = ( \beta ^ { \top } , \theta _ { L } ^ { \top } ) ^ { \top }$ . Although the algorithm is written in terms of $\theta ,$ each score evaluation reconstructs $\beta$ and the Cholesky factor L from $\theta ,$ imposes $L _ { 1 1 } = 1$ , and sets $\Omega = L L ^ { \top }$ . The initial value $\theta _ { \mathrm { i n i t } }$ is therefore an input to the procedure.

The computational budget is summarized by T, m, and R. Here $T$ is the final iterate index, so the algorithm performs $T - 1$ stochastic approximation updates; m is the number of observations in each mini-batch, and R is the number of accepted simulation draws used to form each observation-level loss-gradient. With iid mini-batches sampled from the empirical distribution, $m ( T - 1 ) / n$ is the expected number of passes through the sample.

The averaging input $T _ { 0 }$ determines when Polyak–Ruppert averaging begins. Algorithm 1 reports the average of the iterates $\theta _ { T _ { 0 } + 1 } , \dots , \theta _ { T }$ , consistent with the framework in Section $2 ;$ when $T _ { 0 } = 0$ , this average includes the initial estimator $\theta _ { 1 }$ . The asymptotic results in Section 5 apply when $T _ { 0 } = T _ { 0 } ( T )$ satisfies $T _ { 0 } / T  0$

The learning-rate inputs in Algorithm 1 are $\gamma _ { 0 }$ and a. The exponent $a \in ( 1 / 2 , 1 )$ gives the decreasing-rate form used in the theory, while $\gamma _ { 0 }$ controls the overall scale of the stochastic approximation step. For simplicity, the main algorithm uses a single scalar learning rate. In computation, fixed block-specific scale constants for the coeficient and Cholesky components can be used as a diagonal preconditioner, as described in Remark 5.

Algorithm 1: SAUSS for Multinomial Probit   
Input: Data $\{ ( \Delta X _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } ; \theta _ { \mathrm { i n i t } } ;$ tuning parameters $\{ T , T _ { 0 } , m , R , \gamma _ { 0 } , a \}$ , with   
$T \geq 2$ and $T _ { 0 } \in \{ 0 , \ldots , T - 1 \}$   
Output: $\mathrm { A }$ veraged parameter $\bar { \theta } _ { T }$ , reported as $\hat { \beta }$ and $\hat { \Omega } = \hat { L } \hat { L } ^ { \top }$   
Set $\theta _ { 1 }  \theta _ { \mathrm { i n i t } } , N _ { \mathrm { a v g } }  0 _ { \cdot }$ , and $\bar { \theta }  0$   
if $T _ { 0 } = 0$ then   
$N _ { \mathrm { a v g } } \gets 1$ and $\bar { \theta }  \theta _ { 1 }$   
end   
for $t = 2 , \ldots , T$ do   
Draw $I _ { t 1 } , \ldots , I _ { t m }$ independently and uniformly from $\{ 1 , \ldots , n \}$ , and set   
$\boldsymbol { B } _ { t } \gets ( I _ { t 1 } , \ldots , I _ { t m } )$   
Construct $\beta$ and L from $\theta _ { t - 1 }$ , with $L _ { 1 1 } = 1$   
Set $\Omega  L L ^ { \top }$ and $Q  \Omega ^ { - }$ −1   
Set $\gamma _ { t }  \gamma _ { 0 } ( t - 1 ) ^ { - a }$   
for $b = 1 , \dots , m$ do   
Set $i \gets I _ { t b }$   
Run the accept–reject simulator until R accepted draws are obtained for   
observation i   
Compute $\widehat { g } _ { t b } ( \theta _ { t - 1 } )$ from (18) for observation $i ,$ using fresh accepted draws   
end   
$\begin{array} { r } { \bar { g } _ { t }  m ^ { - 1 } \sum _ { b = 1 } ^ { m } \widehat { g } _ { t b } ( \theta _ { t - 1 } ) } \end{array}$   
$\theta _ { t }  \theta _ { t - 1 } - \gamma _ { t } \bar { g } _ { t }$   
if $t > T _ { 0 }$ then   
$N _ { \mathrm { a v g } }  N _ { \mathrm { a v g } } + 1$   
$\bar { \theta }  \bar { \theta } + ( \theta _ { t } - \bar { \theta } ) / N _ { \mathrm { a v g } }$   
end   
end   
Set $\bar { \theta } _ { T }  \bar { \theta }$   
Construct $\hat { \beta }$ and $\hat { L }$ from $\bar { \theta } _ { T }$ , and set $\hat { \Omega }  \hat { L } \hat { L } ^ { \top }$

Algorithm 1 describes the exact recursion: each observation-level simulator runs until R accepted draws are obtained, so the simulated score contribution is unbiased for $s _ { i } ( \theta )$ for every fixed $R \geq 1 $ ; equivalently, ${ \widehat { g } } _ { i } ( \theta )$ is unbiased for $- s _ { i } ( \theta )$ . Thus the ideal recursion uses a fixed number R of accepted draws, but the number of proposal draws, and hence runtime, is random.

Remark 3 (Capped computation). In practice, we impose a large cap $K _ { \mathrm { m a x } }$ on the number of proposal draws. If the cap is reached after at least one acceptance but before R acceptances, averaging the available accepted draws remains centered on the exact lossgradient, but uses a random, smaller simulation size and has a diferent conditional variance. If no draw is accepted, returning a zero contribution may bias the capped score estimator. We therefore monitor trials per accepted draw and the frequencies of partial-cap and no-acceptance events.

## 5 Asymptotic Theory

This section states the asymptotic theory for the SAUSS recursion with the exact simulator, which runs until R accepted draws are obtained for each observation. Inference based on these results is developed in Section 6. Throughout, n denotes the sample size, $T$ is the final iterate index, m is the mini-batch size, and R is the number of accepted simulation draws per observation. The initial estimator is $\theta _ { 1 }$ , and updates produce $\theta _ { t }$ for $t = 2 , \ldots , T$ , so the run contains $T - 1$ stochastic approximation updates.

All statements indexed by the run length are taken along a sequence in which $T \to \infty$ and the pilot size $n _ { 0 } = n _ { 0 } ( T ) \to \infty ;$ ; no relative rate between $n _ { 0 }$ and $T$ is imposed. The mini-batch size $m ,$ accepted-draw count $R ,$ and learning-rate constants $( \gamma _ { 0 } , a )$ , as well as the parameter dimension $p$ (and hence J in the MNP specialization), are held fixed.

The recursion is (3) of Section 2.3, with mini-batch direction $\hat { G } _ { t } ( \theta )$ built from the one-draw loss-gradient simulator $g _ { i } ( \theta , U ) = - \psi _ { i } ( \theta , U )$ . Let $\mathbb { E } _ { t } ^ { * }$ denote expectation over the simulation draws in iteration t, conditional on the past and the selected mini-batch, and let $\mathcal { F } _ { t }$ be the sigma-field generated by $\theta _ { 1 }$ and all mini-batch and simulation draws through iteration t. For $t \geq 2$ , define

$$
G _ { t } ( \theta ) = \mathbb { E } _ { t } ^ { * } \{ \hat { G } _ { t } ( \theta ) \} , \qquad \mathbb { R } ( \theta ) = \mathbb { E } \{ G _ { t } ( \theta ) \mid \mathcal { F } _ { t - 1 } \} , \qquad H = \nabla _ { \theta } \mathcal { R } ( \theta _ { 0 } ) .
$$

Under population sampling, $\mathcal { R } ( \theta ) = \mathbb { E } \{ g _ { i } ( \theta ) \} = - \mathbb { E } \{ s _ { i } ( \theta ) \}$ , so the target $\theta _ { 0 }$ satisfies $\mathcal { R } ( \theta _ { 0 } ) = 0$ , and H is the negative Jacobian of the population score, that is, the population information matrix under correct specification.

## 5.1 Assumptions

The assumptions below are stated for the recursion with the exact simulator; the proof appendix uses them with $m = 1$ (see Remark 5). Throughout, $C < \infty$ and $c > 0$ denote generic constants.

Assumption 1 (Local identification and information). The normalized parameter space $\Theta \subset \mathbb { R } ^ { p }$ contains $\theta _ { 0 }$ as an interior point, and $\theta _ { 0 }$ is the only zero of ${ \mathcal { R } } ( \theta )$ in a neighborhood $\mathcal { N } _ { 0 }$ of $\theta _ { 0 }$ . The Jacobian $\begin{array} { r } { H ( \theta ) = \nabla _ { \theta } \mathcal { R } ( \theta ) } \end{array}$ exists, is Lipschitz continuous, and satisfies $\lVert H ( \theta ) \rVert \leq C \mathrm { o n } \mathcal { N } _ { 0 } ; H = H ( \theta _ { 0 } )$ is symmetric with $\lambda _ { \operatorname* { m i n } } ( H ) > c$

Assumption 1 gives the local drift condition used for consistency: by the integral form of the mean value theorem, there is a ball $B _ { \rho } = \{ \theta : \| \theta - \theta _ { 0 } \| \leq \rho \} \subset \mathcal { N } _ { 0 }$ on which

$$
\begin{array} { r } { ( \theta - \theta _ { 0 } ) ^ { \prime } \mathcal { R } ( \theta ) \geq c _ { 0 } \| \theta - \theta _ { 0 } \| ^ { 2 } \quad \mathrm { a n d } \quad \| \mathcal { R } ( \theta ) - H ( \theta - \theta _ { 0 } ) \| \leq C \| \theta - \theta _ { 0 } \| ^ { 2 } } \end{array}
$$

for some $c _ { 0 } > 0$ . This is the descent-form analogue of local concavity of the population log-likelihood; symmetry of H holds for likelihood problems, where H is the population information matrix.

Assumption 2 (Local initialization). The recursion is initialized by a preliminary estimator $\theta _ { 1 } = \tilde { \theta } _ { n _ { 0 } }$ satisfying $\tilde { \theta } _ { n _ { 0 } } \xrightarrow { p } \theta _ { 0 }$ as $n _ { 0 } \to \infty$

Assumption 3 (Smoothness of the exact loss-gradient). For each observation, the exact loss-gradient $G _ { t } ( \theta ) = \mathbb { E } _ { t } ^ { * } \{ \hat { G } _ { t } ( \theta ) \}$ is continuously diferentiable in $\theta _ { ; }$ and

$$
\operatorname* { s u p } _ { \theta \in { \mathcal { N } } _ { 0 } } \mathbb { E } \big ( \| \nabla _ { \theta } G _ { t } ( \theta ) \| ^ { 2 } \mid { \mathcal { F } } _ { t - 1 } \big ) \leq C .
$$

The simulated loss-gradient $g _ { i } ( \theta , U )$ may be discontinuous in $\theta ;$ Assumption 3 concerns only its conditional expectation given the data, which for LDV models is the exact score contribution and is smooth.

Assumption 4 (Unbiased simulated loss-gradients and moments). For every $\theta \in \mathcal { N } _ { 0 }$ the one-draw simulator satisfies

$$
{ \mathbb E } \{ g _ { i } ( \theta , U ) \mid D _ { i } \} = g _ { i } ( \theta ) = - s _ { i } ( \theta ) ,
$$

and the simulation draws are fresh across observations, accepted draws, and stochastic approximation iterations. Let $a \in ( 1 / 2 , 1 )$ be the learning-rate exponent of Assumption 6. For some $\delta > 0$ and some $p _ { 0 } > ( 1 - a ) ^ { - 1 }$ , with $q = \operatorname* { m a x } \{ 2 + \delta , 2 p _ { 0 } \}$ ,

$$
\operatorname* { s u p } _ { \theta \in \mathcal { N } _ { 0 } } \mathbb { E } \big ( \| g _ { i } ( \theta , U ) \| ^ { q } \mid \mathcal { F } _ { t - 1 } \big ) \leq C , \qquad \operatorname* { s u p } _ { \theta \in \mathcal { N } _ { 0 } } \mathbb { E } \big ( \| G _ { t } ( \theta ) \| ^ { q } \mid \mathcal { F } _ { t - 1 } \big ) \leq C .
$$

The $2 + \delta$ moments are used for the Lindeberg condition; the $2 p _ { 0 }$ moments are used to control the Polyak–Ruppert weighted remainder, and $p _ { 0 }$ grows as $a \to 1$

Assumption 5 (Continuity of the simulation variance). Let $A _ { t } ( \theta ) = \operatorname { V a r } \{ g _ { i } ( \theta , U ) \mid D _ { i } \}$ be the conditional covariance of the one-draw simulated loss-gradient given the observation. Then $\theta \mapsto \mathbb { E } \{ A _ { t } ( \theta ) \mid \mathcal { F } _ { t - 1 } \}$ is Lipschitz continuous on $\mathcal { N } _ { 0 }$ with a bounded Lipschitz constant.

Assumption 5 replaces a mean-square continuity condition on the simulator itself: it restricts only the conditional variance of the simulated loss-gradient, not a coupling of draws at diferent parameter values, and it is what the variance-consistency step of the martingale functional central limit theorem uses.

Assumption 6 (Mini-batches and learning rates). At each update $t \geq 2$ , the mini-batch indices $I _ { t 1 } , \ldots , I _ { t m }$ are sampled independently with replacement from the population distribution, or from the empirical distribution in the fixed-sample version, independently of the past conditional on the current iterate. The learning rate is

$$
\gamma _ { t } = \gamma _ { 0 } ( t - 1 ) ^ { - a } , \qquad a \in ( 1 / 2 , 1 ) , \qquad t \geq 2 .
$$

Hence $\begin{array} { r } { \sum _ { t \geq 2 } \gamma _ { t } = \infty , \sum _ { t \geq 2 } \gamma _ { t } ^ { 2 } < \infty } \end{array}$ , and $\textstyle \sum _ { t \geq 2 } \gamma _ { t } / { \sqrt { t } } < \infty$

Assumption 7 (Conditional variance limit). For $t \geq 2$ , let

$$
\zeta _ { t } ( \theta ) = \hat { G } _ { t } ( \theta ) - G _ { t } ( \theta ) , \qquad \xi _ { t } ( \theta ) = G _ { t } ( \theta ) - \mathcal { R } ( \theta ) ,
$$

and write $\varepsilon _ { t } = \zeta _ { t } ( \theta _ { t - 1 } ) + \xi _ { t } ( \theta _ { t - 1 } )$ for the noise evaluated at the current iterate, which is a martingale diference with respect to $\mathcal { F } _ { t }$ . The limit

$$
V _ { m , R } = \operatorname { p l i m } _ { t \to \infty } \Big [ \operatorname { V a r } \{ G _ { t } ( \theta _ { 0 } ) \mid \mathcal { F } _ { t - 1 } \} + \frac { 1 } { m R } \mathbb { E } \{ A _ { t } ( \theta _ { 0 } ) \mid \mathcal { F } _ { t - 1 } \} \Big ]
$$

exists and is positive definite.

For iid mini-batches with replacement and iid simulation draws, the variance in Assumption 7 has the explicit decomposition

$$
V _ { m , R } = { \frac { 1 } { m } } \operatorname { V a r } \{ g _ { i } ( \theta _ { 0 } ) \} + { \frac { 1 } { m R } } \mathbb { E } \left[ \operatorname { V a r } \{ g _ { i } ( \theta _ { 0 } , U ) \mid D _ { i } \} \right] .\tag{20}
$$

The first term is mini-batch sampling variation in the exact loss-gradient. The second term is simulation variation conditional on the observation. Since $g _ { i } ( \theta _ { 0 } ) = - s _ { i } ( \theta _ { 0 } )$ , the first term equals $\mathrm { V a r } \{ s _ { i } ( \theta _ { 0 } ) \} / m$

Assumption 8 (Smooth transformations). When inference concerns a scalar parameter $\kappa = h ( \theta _ { 0 } )$ , the map $h : \Theta \to \mathbb { R }$ is twice continuously diferentiable on $\mathcal { N } _ { 0 }$ and satisfies

$$
\operatorname* { s u p } _ { \theta \in { \mathcal { N } } _ { 0 } } \| \nabla ^ { 2 } h ( \theta ) \| + \| \nabla h ( \theta _ { 0 } ) \| \le C .
$$

## 5.2 Main stochastic approximation result

Theorem 1 (Consistency, linear representation, and FCLT). Let $m = 1$ , so that each update uses one observation and R accepted draws, and suppose Assumptions $1 - 7$ hold. Define the average and, for $r \in [ 0 , 1 ]$ , the partial-sum and limiting processes by

$$
\bar { \theta } _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \theta _ { t } ,
$$

$$
X _ { T } ( r ) = { \frac { 1 } { \sqrt { T } } } \sum _ { t = 1 } ^ { \lfloor T r \rfloor } ( \theta _ { t } - \theta _ { 0 } ) ,
$$

$$
X ( r ) = \big ( H ^ { - 1 } V _ { 1 , R } H ^ { - 1 } \big ) ^ { 1 / 2 } W ( r ) ,
$$

where W is a p-dimensional standard Wiener process. Let $\tau$ be the first exit time from $B _ { \rho }$ and set $\mathcal { E } : = \{ \tau = \infty \}$ . Then, for every $\epsilon > 0$ , there exists $\bar { \gamma } ( \epsilon ) > 0$ such that, if $\gamma _ { 0 } \leq \bar { \gamma } ( \epsilon ) , \mathbb { P } ( \mathcal { E } ^ { c } ) \leq \epsilon + o ( 1 )$ , and the following conclusions hold:

(i) (consistency) $\lVert \theta _ { t } - \theta _ { 0 } \rVert \mathbb { I } \{ \mathcal { E } \} \stackrel { p } { \to } 0 \ a s \ t \to \infty ,$

(ii) (linear representation)

$$
\sqrt { T } ( \bar { \theta } _ { T } - \theta _ { 0 } ) = - H ^ { - 1 } \frac { 1 } { \sqrt { T } } \sum _ { t = 2 } ^ { T } \{ \zeta _ { t } ( \theta _ { t - 1 } ) + \xi _ { t } ( \theta _ { t - 1 } ) \} + Z _ { T } , \qquad Z _ { T } \mathbb { I } \{ \mathcal { E } \} \stackrel { p } {  } 0 ,\tag{21}
$$

and the same representation holds uniformly in r for the partial sums $X _ { T } ( r )$

(iii) (FCLT) for every bounded continuous f on $D [ 0 , 1 ] ^ { p }$

$$
\operatorname* { l i m } _ { T \to \infty } \left| \mathbb { E } f ( X _ { T } ) - \mathbb { E } f ( X ) \right| \leq 2 \epsilon \operatorname* { s u p } | f | ;
$$

in particular $\sqrt { T } ( \bar { \theta } _ { T } - \theta _ { 0 } ) = X _ { T } ( 1 )$ is asymptotically $\mathcal { N } ( 0 , H ^ { - 1 } V _ { 1 , R } H ^ { - 1 } )$ up to the same ϵ-error.

Remark 4 (On the ϵ-formulation). The tolerance ϵ originates in a single step of the proof. For any $b \in ( 0 , \rho )$ , the stopped-process argument and a supermartingale maximal inequality give

$$
\mathbb { P } ( \tau < \infty ) \le \mathbb { P } ( \| \theta _ { 1 } - \theta _ { 0 } \| > b ) + \frac { b ^ { 2 } + C \gamma _ { 0 } ^ { 2 } \sum _ { s \ge 1 } s ^ { - 2 a } } { \rho ^ { 2 } } .
$$

For a given $\epsilon \in ( 0 , 1 )$ , set $b _ { \epsilon } = \rho \sqrt { \epsilon / 2 }$ and choose $\gamma _ { 0 }$ suficiently small that

$$
C \gamma _ { 0 } ^ { 2 } \sum _ { s \geq 1 } s ^ { - 2 a } \leq \frac { \rho ^ { 2 } \epsilon } { 2 } .
$$

Then

$$
\begin{array} { r } { \mathbb { P } ( \tau < \infty ) \le \mathbb { P } ( \| \theta _ { 1 } - \theta _ { 0 } \| > b _ { \epsilon } ) + \epsilon = \epsilon + o ( 1 ) , } \end{array}
$$

where the final equality follows from pilot consistency as $n _ { 0 } = n _ { 0 } ( T ) \to \infty$ . All conclusions of the theorem are established on the event $\{ \tau = \infty \}$ , and ϵ bounds the probability of its complement.

Ordinary weak convergence without the ϵ-error is not available under the stated local assumptions because the unbounded simulation noise gives the recursion a positive probability of leaving the neighborhood on which those assumptions hold. Establishing such convergence would require additional stability conditions controlling the recursion outside that neighborhood. These conditions are generally unsuitable for MNP because the population objective can flatten as latent variances increase. We therefore retain the local ϵ-formulation, which applies to the diminishing learning-rate schedule used in implementation, with $\gamma _ { 0 }$ and a held fixed as $T$ increases.

Remark 5 (Mini-batches, step-size shift, and burn-in). Theorem 1 is stated and proved for $m = 1$ . For any fixed $m _ { : }$ suppose that the mini-batch observations are sampled independently with replacement and that their simulation draws are conditionally independent across contributions. The mini-batch direction $\hat { G } _ { t } ( \theta )$ in (3) is then the average of $m$ conditionally independent copies of the one-observation direction. It has the same conditional mean, and its conditional variance is $1 / m$ times the one-observation variance, yielding $V _ { m , R } = V _ { 1 , R } / m$ . Its moment and Lipschitz bounds follow from those of the individual contributions with adjusted constants. The same argument therefore applies with $V _ { 1 , R }$ replaced by $V _ { m , R }$

The same result covers a fixed symmetric positive-definite preconditioner $P$ in the update

$$
\theta _ { t } = \theta _ { t - 1 } - \gamma _ { t } P \hat { G } _ { t } ( \theta _ { t - 1 } ) , \qquad \gamma _ { t } = \gamma _ { 0 } ( t - 1 ) ^ { - a } .
$$

In the centered coordinate $\phi = P ^ { - 1 / 2 } ( \theta - \theta _ { 0 } )$ , the drift and innovation covariance are $P ^ { 1 / 2 } H P ^ { 1 / 2 }$ and $P ^ { 1 / 2 } V _ { m , R } P ^ { 1 / 2 }$ , respectively. Applying the theorem in this coordinate and mapping back preserves the Polyak–Ruppert covariance $H ^ { - 1 } V _ { m , R } H ^ { - 1 }$ , with an adjusted small-step threshold.

A fixed shift $\gamma _ { t } = \gamma _ { 0 } ( t - 1 + t _ { 0 } ) ^ { - a }$ also leaves the limits unchanged. For the averaged estimator, let $T _ { 0 } = T _ { 0 } ( T )$ be deterministic and satisfy $T _ { 0 } / T  0$ . The functional limit theorem then gives $\{ \sum _ { t = 1 } ^ { T _ { 0 } } ( \theta _ { t } - \theta _ { 0 } ) \} \mathbb { I } \{ \mathcal { E } \} = o _ { \mathbb { P } } ( \sqrt { T } )$ , and $T / ( T - T _ { 0 } )  1$ , so the linear representation and functional limit remain unchanged in the same ϵ-sense as Theorem 1.

Many scalar parameters of interest are nonlinear transformations of $\theta _ { 0 }$ , for example a particular substitution elasticity or choice probability in MNP. Let h be such a transformation and define $\kappa _ { t } = h ( \theta _ { t } )$ and $\kappa _ { 0 } = h ( \theta _ { 0 } )$

Theorem 2 (Smooth transformations). Under the conditions of Theorem 1, suppose in addition that Assumption 8 holds. Then, in the same sense as in Theorem $1 ( i i i )$ 2

$$
\frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { \lfloor T r \rfloor } ( \kappa _ { t } - \kappa _ { 0 } ) \Rightarrow \nabla h ( \theta _ { 0 } ) ^ { \prime } \bigl ( H ^ { - 1 } V _ { 1 , R } H ^ { - 1 } \bigr ) ^ { 1 / 2 } W ( r ) , \qquad r \in [ 0 , 1 ] .
$$

Theorem 2 says that the scalar transformed path $\kappa _ { t }$ obeys the same functional limit as a linear functional of $\theta _ { t }$ , so the path-based inference of Section 6 applies to $\kappa _ { t }$ directly, without a delta-method variance estimator.

## 6 Inference

This section collects the inference consequences of Theorems 1 and 2. Three features of SAUSS shape it. First, the recursion has two sources of algorithmic variation, mini-batch selection and score simulation. Their magnitude relative to sampling uncertainty from the data depends on $n , T , m ,$ , and R. Second, random-scaling inference uses only the solution path, so it does not require estimates of H or of the simulation variance. Third, many scalar objects of interest, such as a particular substitution elasticity or choice probability, are nonlinear transformations of $\theta _ { 0 }$ , and the transformed path can be used directly.

## 6.1 Sampling and algorithmic uncertainty

For a sample of size n, the same recursion can be viewed as a randomized algorithm targeting the exact sample MLE. Let

$$
S _ { n } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } s _ { i } ( \theta ) ,
$$

and let $\hat { \theta } _ { n }$ solve $S _ { n } (  { \hat { \theta } } _ { n } ) = 0$ , suppressing standard existence qualifications. Under the usual likelihood expansion,

$$
\sqrt { n } ( \hat { \theta } _ { n } - \theta _ { 0 } ) = H ^ { - 1 } \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } s _ { i } ( \theta _ { 0 } ) + o _ { p } ( 1 ) .
$$

For a fixed sample, the exact sample estimator $\hat { \theta } _ { n }$ is the algorithmic target. The sampling and algorithmic errors satisfy the exact decomposition

$$
\sqrt { n } ( \bar { \theta } _ { T } - \theta _ { 0 } ) = \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { 0 } ) + \sqrt { n } ( \bar { \theta } _ { T } - \hat { \theta } _ { n } ) .
$$

Remark 6 (Finite-run variance approximation). For a finite sample and a finite number of stochastic approximation iterations, a useful first-order approximation is

$$
\mathrm { V a r } ( \bar { \theta } _ { T } - \theta _ { 0 } ) \approx \frac { 1 } { n } H ^ { - 1 } \Sigma _ { s } H ^ { - 1 } + \frac { 1 } { T } H ^ { - 1 } V _ { m , R } H ^ { - 1 } .
$$

Equivalently, this approximation suggests

$$
\sqrt { n } ( \bar { \theta } _ { T } - \theta _ { 0 } ) \stackrel { \mathrm { a p p r o x . } } { \sim } \mathcal { N } \left( 0 , H ^ { - 1 } \Sigma _ { s } H ^ { - 1 } + \frac { n } { T } H ^ { - 1 } V _ { m , R } H ^ { - 1 } \right) .\tag{22}
$$

Here $\Sigma _ { s } = \mathrm { V a r } \{ s _ { i } ( \theta _ { 0 } ) \}$ . The first term represents sampling uncertainty in the exact sample estimator, while the second represents the additional uncertainty from terminating the randomized algorithm after $T$ iterations. Thus, the sum can provide a more informative finite-run variance approximation than the conventional likelihood variance alone when T is not large relative to n. By Remark 5, $V _ { m , R } = V _ { 1 , R } / m$ , so the approximation also provides the reductions from larger mini-batches and additional accepted simulation draws.

A corresponding plug-in approximation is

$$
\widehat { \mathrm { V a r } } ( \bar { \theta } _ { T } ) : = \frac { 1 } { n } \hat { H } ^ { - 1 } \hat { \Sigma } _ { s } \hat { H } ^ { - 1 } + \frac { 1 } { T } \hat { H } ^ { - 1 } \hat { V } _ { m , R } \hat { H } ^ { - 1 } .
$$

The fixed-sample implementation underlying this approximation uses iid sampling with replacement from the empirical distribution and fresh simulation draws at every score evaluation. When $m ( T - 1 ) > n$ , observations are sampled multiple times on average, and $m ( T - 1 ) / n$ is the expected number of passes.

## 6.2 Random scaling and sandwich inference

The functional central limit theorem in Theorem 1 gives two routes to inference on the algorithmic target. Let A be an $\ell \times p$ matrix with full row rank. For random-scaling inference, consider the null restriction $A \theta _ { 0 } = a _ { 0 }$

Corollary 1 (Inference consequences of the FCLT). Under the FCLT in Theorem 1, with the fixed-m extension in Remark 5 when m $> 1$ , let ${ \bar { \theta } } _ { T } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } \theta _ { t }$ and define

$$
\widehat V _ { \mathrm { R S } } ( A ) = \frac { 1 } { T ^ { 2 } } \sum _ { t = 1 } ^ { T } \left\{ \sum _ { s = 1 } ^ { t } A ( \theta _ { s } - \bar { \theta } _ { T } ) \right\} \left\{ \sum _ { s = 1 } ^ { t } A ( \theta _ { s } - \bar { \theta } _ { T } ) \right\} ^ { \prime } .
$$

Under the null restriction $A \theta _ { 0 } = a _ { 0 }$ , when $\widehat { V } _ { \mathrm { R S } } ( A )$ is nonsingular, the random-scaling statistic

$$
T ( A \bar { \theta } _ { T } - a _ { 0 } ) ^ { \prime } \widehat { V } _ { \mathrm { R S } } ( A ) ^ { - 1 } ( A \bar { \theta } _ { T } - a _ { 0 } )
$$

converges (in the sense of Theorem $1 ( i i i ) )$ to the standard self-normalized Brownian func-

tional

$$
W _ { \ell } ( 1 ) ^ { \prime } \left\{ \int _ { 0 } ^ { 1 } \bar { W } _ { \ell } ( r ) \bar { W } _ { \ell } ( r ) ^ { \prime } d r \right\} ^ { - 1 } W _ { \ell } ( 1 ) , \qquad \bar { W } _ { \ell } ( r ) = W _ { \ell } ( r ) - r W _ { \ell } ( 1 ) .
$$

Alternatively, if H<sup>ˆ</sup> consistently estimates H and $\hat { V } _ { m , R }$ consistently estimates $V _ { m , R } ,$ then

$$
\hat { H } ^ { - 1 } \hat { V } _ { m , R } \hat { H } ^ { - 1 } \stackrel { p } {  } H ^ { - 1 } V _ { m , R } H ^ { - 1 } .
$$

Consequently,

$$
\frac { 1 } { T } \hat { H } ^ { - 1 } \hat { V } _ { m , R } \hat { H } ^ { - 1 }
$$

is a sandwich estimator for the first-order algorithmic covariance of $\bar { \theta } _ { T }$

Under population sampling, the target in Corollary 1 is $\theta _ { 0 }$ . Conditional on a fixed sample, path variation instead describes the remaining randomized-algorithm variation around the sample target $\hat { \theta } _ { n }$ and does not by itself account for sampling uncertainty. Moreover, $\widehat { V } _ { \mathrm { R S } } ( A )$ is a self-normalizer with a nondegenerate Brownian-functional limit, not a consistent estimator of $A H ^ { - 1 } V _ { m , R } H ^ { - 1 } A ^ { \prime }$

Random scaling uses the solution path and follows the online-inference logic developed in Lee et al. (2022) and Lee et al. (2025). Sandwich inference is closer to conventional MLE inference and is useful when the algorithm is run long enough that computational uncertainty is intended to be small; the MNP form of $V _ { m , R }$ and its plug-in estimator are given next.

The MNP form of $V _ { m , R }$ and a plug-in estimator. For the MNP specialization, $V _ { m , R }$ can be written directly in terms of accepted-draw loss-gradient contributions. For an accepted draw ν of the diferenced error vector defined relative to alternative 1, define

$$
\Gamma _ { i } ( \theta , \nu ) : = \left( \begin{array} { c } { { - \Delta X _ { i } ^ { \top } Q \nu } } \\ { { - \mathcal { C } _ { \theta _ { L } } \{ Q ( \nu \nu ^ { \top } - \Omega ) Q L ; L \} } } \end{array} \right) , \qquad \Omega : = L L ^ { \top } , \qquad Q : = \Omega ^ { - 1 } .
$$

Let $\nu _ { i } ^ { \star } ( \theta _ { 0 } )$ denote one accepted draw from the conditional distribution of the canonical error vector given $( \Delta X _ { i } , y _ { i } )$ at $\theta _ { 0 }$ . Then

$$
g _ { i } ( \theta _ { 0 } ) = \mathbb { E } \{ \Gamma _ { i } ( \theta _ { 0 } , \nu _ { i } ^ { \star } ( \theta _ { 0 } ) ) ~ | ~ \Delta X _ { i } , y _ { i } \} ,
$$

and

$$
V _ { m , R } ^ { \mathrm { M N P } } = \frac { 1 } { m } \operatorname { V a r } \{ g _ { i } ( \theta _ { 0 } ) \} + \frac { 1 } { m R } \mathbb { E } \left[ \operatorname { V a r } \{ \Gamma _ { i } ( \theta _ { 0 } , \nu _ { i } ^ { \star } ( \theta _ { 0 } ) ) \mid \Delta X _ { i } , y _ { i } \} \right] .\tag{23}
$$

Equivalently, if

$$
\widehat { g } _ { i } ^ { R } ( \theta _ { 0 } ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \Gamma _ { i } ( \theta _ { 0 } , \nu _ { i r } ^ { \star } ( \theta _ { 0 } ) ) ,
$$

then

$$
V _ { m , R } ^ { \mathrm { M N P } } = \frac { 1 } { m } \operatorname { V a r } \{ \widehat { g } _ { i } ^ { R } ( \theta _ { 0 } ) \} .\tag{24}
$$

A direct plug-in estimator can be built from independent accepted-draw loss-gradient calls at a consistent estimate ${ \hat { \theta } } ,$ where $\hat { \theta } \stackrel { p } {  } \theta _ { 0 }$ . Generate B independent R-accepted calls

$$
\widehat { g } _ { i b } ^ { R } ( \widehat { \theta } ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \Gamma _ { i } ( \widehat { \theta } , \widehat { \nu } _ { i b r } ^ { \star } ) , \qquad b = 1 , \ldots , B ,
$$

set

$$
\bar { g } = \frac { 1 } { n B } \sum _ { i = 1 } ^ { n } \sum _ { b = 1 } ^ { B } \widehat { g } _ { i b } ^ { R } ( \widehat { \theta } ) ,
$$

and use

$$
\widehat { V } _ { m , R } ^ { \mathrm { M N P } } = \frac { 1 } { m } \frac { 1 } { n B - 1 } \sum _ { i = 1 } ^ { n } \sum _ { b = 1 } ^ { B } \left\{ \widehat { g } _ { i b } ^ { R } ( \widehat { \theta } ) - \bar { g } \right\} \left\{ \widehat { g } _ { i b } ^ { R } ( \widehat { \theta } ) - \bar { g } \right\} ^ { \prime } .\tag{25}
$$

For fixed $m , R ,$ , and $B ,$ under iid sampling, fresh conditionally independent evaluation calls, and standard continuity and local uniform-law conditions at $\hat { \theta } \stackrel { p } {  } \theta _ { 0 }$ , the estimator in (25) is consistent for $V _ { m , R } ^ { \mathrm { M N P } }$ as $n  \infty$ . Calls sharing the same observation i are conditionally independent given $D _ { i }$ , rather than unconditionally independent. With $B =$ 1, this estimator uses one independent loss-gradient call per observation and estimates the total one-step variance in (24). Larger B reduces Monte Carlo noise in estimating the sandwich input, and $B \geq 2$ is required to report the within- and between-observation decomposition in (23).

## 6.3 Nonlinear transformations

Let the scalar parameter of interest be $\kappa _ { 0 } = h ( \theta _ { 0 } )$ for a smooth map h satisfying Assumption 8, and suppose $\nabla h ( \theta _ { 0 } ) \neq 0$ so that its first-order limit is nondegenerate. Rather than estimating $\nabla h ( { \hat { \theta } } )$ and applying the delta method, we track the transformed path $\kappa _ { t } = h ( \theta _ { t } )$ inside the recursion and apply random scaling to $\kappa _ { t }$ directly. Theorem 2 shows that the partial-sum process of $\kappa _ { t } - \kappa _ { 0 }$ has the limit $\nabla h ( \theta _ { 0 } ) ^ { \prime } ( H ^ { - 1 } V _ { 1 , R } H ^ { - 1 } ) ^ { 1 / 2 } W ( r )$ . Define

$$
\bar { \kappa } _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \kappa _ { t } , \qquad \widehat { V } _ { \mathrm { R S } } ^ { \kappa } = \frac { 1 } { T ^ { 2 } } \sum _ { t = 1 } ^ { T } \left\{ \sum _ { s = 1 } ^ { t } ( \kappa _ { s } - \bar { \kappa } _ { T } ) \right\} ^ { 2 } .
$$

The scalar version of Corollary 1 therefore gives the self-normalized statistic

$$
\frac { T ( \bar { \kappa } _ { T } - \kappa _ { 0 } ) ^ { 2 } } { \widehat { V } _ { \mathrm { R S } } ^ { \kappa } } .
$$

## 7 Monte Carlo Experiments

In this section we examine the finite-sample performance of the SAUSS estimator for the multinomial probit model. We use the accept–reject simulator of Section 4 with the covariance parameterization $\Omega = L L ^ { \prime }$ , identified by the normalization $L _ { 1 1 } = 1$ . A sequential accept–reject variant produced similar results and we do not report it separately.

## 7.1 Design and Implementation

The baseline design, labeled D1, is a multinomial probit model written in utility diferences relative to alternative 1. For individual $i = 1 , \ldots , n$ and alternatives $j = 2 , \dots , J$

$$
U _ { i j } - U _ { i 1 } = ( x _ { i j } - x _ { i 1 } ) ^ { \prime } \beta _ { 0 } + \nu _ { i j } , \qquad \nu _ { i } = ( \nu _ { i 2 } , \ldots , \nu _ { i J } ) ^ { \prime } \sim \mathcal { N } ( 0 , \Omega _ { 0 } ) ,\tag{26}
$$

and the observed choice $y _ { i }$ is the alternative with the largest latent utility. Each alternative has $K = 3$ covariates drawn independently from the standard normal distribution, and the true coeficient vector is $\beta _ { 0 } = ( - 0 . 5 , 1 , 1 ) ^ { \prime }$ . The diferenced errors are equicorrelated: $\Omega _ { 0 }$ has unit diagonal and all of-diagonal elements equal to 0.3. The number of alternatives varies over $J \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ , so that the dimension of $\Omega _ { 0 }$ ranges from $3 \times 3 ~ \mathrm { t o ~ 6 3 \times 6 3 }$ . The sample size is fixed at $n = 2 0 0 0$ throughout, and each design cell is replicated 1000 times. We therefore focus on computational scaling with the number of alternatives and on finite-sample performance across diferent Monte Carlo designs.

The implementation follows Section 4. At each iteration a mini-batch of $m = 1 0$ observations is drawn from the sample independently with replacement, and the accept– reject simulator is run until $R = 5$ accepted draws are obtained for each observation in the mini-batch, subject to a trial cap of $K _ { \operatorname* { m a x } } = 2 0 { , } 0 0 0$ . If the simulator obtains no accepted draw for an observation within the cap, that observation’s contribution to the mini-batch gradient is omitted from the update. Let $s = 1 , \ldots , 8 0 , 0 0 0$ index the stochastic updates. The block-specific learning rates are $\gamma _ { \beta , s } = 0 . 5 s ^ { - 0 . 5 0 1 }$ and $\gamma _ { L , s } = 0 . 2 s ^ { - 0 . 5 0 1 }$ ; the exponent lies in the interval $( 1 / 2 , 1 )$ required for the asymptotic optimality of the averaged iterate (Polyak and Juditsky, 1992). We use Polyak–Ruppert averaging to construct the reported estimates. The tuning constants were fixed in a small pilot study at $J = 4$ and then held constant across all designs and all values of $J ;$ a tuning sweep at $J = 6 4$ , not reported here, produced stable results in a neighborhood of these values. Table 1 summarizes the design.

We evaluate the estimates with three criteria. Although the normalization $L _ { 1 1 } =$ 1 fixes the scale of the parameters, we report criteria that are invariant to the scale normalization so that comparisons with implementations adopting other normalizations remain transparent. First, the coeficient criterion measures the recovery of the direction of $\beta _ { 0 }$ . Writing ${ \tilde { \beta } } = \beta / \| \beta \|$ and letting $\hat { \beta } _ { b }$ denote the estimate in replication $b = 1 , \dots , B$

Table 1: Monte Carlo design  
Model Multinomial probit in utility diferences, equation (26)   
Alternatives $J \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \} ~ ( \mathrm { D 1 } ) ; J = 4 ~ ( \mathrm { D 2 } , \mathrm { D 3 } )$   
Covariates $K = 3 , x _ { i j k } \stackrel { i i d } { \sim } \mathcal { N } ( 0 , 1 )$   
Coeficients D1: $\beta _ { 0 } = ( - 0 . 5 , 1 , 1 ) ^ { \prime } ; \mathrm { D } 2$ , D3: $\beta _ { 0 } = ( - 1 . 5 , 2 , 2 ) ^ { \prime }$   
Error covariance D1: unit diagonal, equicorrelation 0.3; D2, D3: equation (27)   
Identification $\Omega = L L ^ { \prime } , L _ { 1 1 } = 1$   
Sample size $n = 2 0 0 0$   
Replications 1000 per design cell   
Mini-batch size $m = 1 0 ,$ sampled i.i.d. with replacement   
Accepted draws R = 5 per observation, trial cap $K _ { \operatorname* { m a x } } = 2 0 { , } 0 0 0$   
Step size $0 . 5 s ^ { - 0 . 5 0 1 } ~ ( \beta ) , 0 . 2 s ^ { - 0 . 5 0 1 } ~ ( L )$   
Updates 80,000, with Polyak–Ruppert averaging

$$
\begin{array} { r } { \mathrm { R M S E } _ { \beta } = \Big \{ B ^ { - 1 } \sum _ { b = 1 } ^ { B } \| \tilde { \hat { \beta } } _ { b } - \tilde { \beta } _ { 0 } \| ^ { 2 } \Big \} ^ { 1 / 2 } . } \end{array}
$$

Second, the correlation criterion measures the recovery of the error dependence structure. Let $C ( \Omega )$ denote the correlation matrix implied by Ω and $d = J - 1$ ; then

$$
\begin{array} { r } { \mathrm { R M S E } _ { C } = \Big \{ \frac { 2 } { B d ( d - 1 ) } \sum _ { b = 1 } ^ { B } \sum _ { k < \ell } \big ( C _ { k \ell } ( \hat { \Omega } _ { b } ) - C _ { k \ell } ( \Omega _ { 0 } ) \big ) ^ { 2 } \Big \} ^ { 1 / 2 } . } \end{array}
$$

Third, the probability criterion measures out-of-sample fit of the implied choice probabilities. For each J we fix a test design of $N _ { \mathrm { t e s t } } = 5 0 0 0$ covariate draws from the distribution above, independent of all estimation samples, and approximate the choice probabilities $P _ { q j } ( \theta )$ at any parameter value by frequency simulation with $S _ { \mathrm { e v a l } } = 5 0 0 0$ common random draws. The criterion is

$$
\begin{array} { r } { \mathrm { R M S E } _ { P } = \Big \{ \frac { 1 } { B N _ { \mathrm { t e s t } } J } \sum _ { b = 1 } ^ { B } \sum _ { q = 1 } ^ { N _ { \mathrm { t e s t } } } \sum _ { j = 1 } ^ { J } \big ( \hat { P } _ { q j } ( \hat { { \theta } } _ { b } ) - \hat { P } _ { q j } ( { \theta } _ { 0 } ) \big ) ^ { 2 } \Big \} ^ { 1 / 2 } , } \end{array}
$$

where the same test design and the same evaluation draws are used for all replications and all estimators within a given $^ { J , }$ so that simulation noise in the evaluation step is common across the comparison. Because average choice probabilities scale as $1 / J _ { \cdot }$ , levels of ${ \mathrm { R M S E } } _ { P }$ are not comparable across diferent values of $J ;$ we therefore use ${ \mathrm { R M S E } } _ { P }$ only for comparisons within a given J.

## 7.2 Main Results

We benchmark SAUSS against simulated maximum likelihood based on the Geweke– Hajivassiliou–Keane (GHK) simulator at $J \ = \ 4$ , where the conventional estimator is computationally reliable (B¨orsch-Supan and Hajivassiliou, 1993; Hajivassiliou et al., 1996;

Table 2: SAUSS performance across J and comparison with GHK-based simulated maximum likelihood at $J = 4$ (design D1)
<table><tr><td></td><td>J Estimator</td><td> $\mathrm { R M S E } _ { \beta }$ </td><td> ${ \mathrm { R M S E } } _ { C }$ </td><td> ${ \mathrm { R M S E } } _ { P }$ </td><td>Median time (s)</td></tr><tr><td rowspan="2">4</td><td>SAUSS</td><td>0.0205</td><td>0.0997</td><td>0.0120</td><td>0.5</td></tr><tr><td>GHK-SML</td><td>0.0202</td><td>0.0939</td><td>0.0122</td><td>71.8</td></tr><tr><td>8</td><td>SAUSS</td><td>0.0171</td><td>0.2010</td><td>0.0121</td><td>2.6</td></tr><tr><td>16</td><td>SAUSS</td><td>0.0159</td><td>0.2864</td><td>0.0101</td><td>9.8</td></tr><tr><td>32</td><td>SAUSS</td><td>0.0148</td><td>0.3077</td><td>0.0075</td><td>43.4</td></tr><tr><td>64</td><td>SAUSS</td><td>0.0140</td><td>0.3091</td><td>0.0056</td><td>80.5</td></tr></table>

Notes: GHK-SML is mlogit with probit=TRUE and 100 GHK draws at default settings. At $J = 4 ,$ both estimators are applied to the same 1,000 simulated data sets.

Train, 2009). The simulated log-likelihood is maximized numerically. We use the implementation in the R package mlogit (Croissant, 2020), with 100 GHK draws per observation and the package’s default starting values, optimizer, and convergence tolerances. Both estimators are applied to the same 1,000 simulated data sets and use the same normalization $L _ { 1 1 } = 1$ . For the larger choice sets, we report SAUSS alone to examine its computational and statistical scaling as J increases.

Consider first the behavior of SAUSS across J. The coeficient criterion improves as J grows with n fixed, from 0.0205 at $J = 4 \tan { 0 . 0 1 4 0 }$ at $J = 6 4 \colon$ each observation contributes a comparison among J alternatives, so the information about the direction of $\beta _ { 0 }$ per observation increases with J. The correlation criterion moves in the opposite direction, consistent with the growing number of covariance parameters relative to the fixed sample size: the number of free correlation parameters, $d ( d \mathrm { - } 1 ) / 2$ , grows quadratically in J. The acceptance rate of the accept–reject simulator declines steeply with J, though somewhat more slowly than the rate 1/J at the larger values, from 0.281 at J = 4 to 0.033 at $J = 6 4$ as the observed-choice region occupies a shrinking share of the error space. The declining acceptance rate increases simulation efort, but no-acceptance events remain rare. Even at $J = 6 4$ , the simulator failed to produce an accepted draw in only 143 of the 800,000 observation-level evaluations per replication, less than 0.02 percent. Computation time increases with J, but the median is only 80.5 seconds per replication at $J = 6 4$

At J = 4, the two estimators have similar accuracy and the absolute diferences across the three criteria are at most 0.0058. Under the reported configurations, SAUSS has a median runtime of approximately 0.5 seconds, compared with 71.8 seconds for GHK-SML, a factor of about 140. In this small-J setting, SAUSS delivers similar aggregate accuracy to the conventional estimator. The larger choice sets provide computational stress tests with an unrestricted covariance parameterization. In particular, the $J = 6 4$ design demonstrates computational feasibility rather than precise recovery of the full covariance matrix, with the tuning constants held fixed across J.

Table 3: SAUSS performance under the baseline and stress designs at $J = 4$
<table><tr><td>Design</td><td> $\mathrm { R M S E } _ { \beta }$ </td><td> ${ \mathrm { R M S E } } _ { \delta }$ </td><td> ${ \mathrm { R M S E } } _ { C }$ </td><td> ${ \mathrm { R M S E } } _ { P }$ </td><td></td><td>Acceptance rate Median time (s)</td></tr><tr><td>D1</td><td>0.0205</td><td></td><td>0.0997</td><td>0.0120</td><td>0.281</td><td>0.5</td></tr><tr><td>D2</td><td>0.0138</td><td></td><td>0.1341</td><td>0.0157</td><td>0.414</td><td>0.6</td></tr><tr><td>D3</td><td>0.0144</td><td> $0 . 0 5 1 4$ </td><td>0.1586</td><td>0.0205</td><td>0.453</td><td>0.5</td></tr></table>

Notes: $J = 4 , n = 2 0 0 0$ , 1000 replications, tuning as in Table 1. D1 is the equicorrelated baseline. RMSE is the direction criterion for the intercept block (D3 only). The rare alternative in D3 has a 9.2 percent choice share.

## 7.3 Heterogeneous Covariance and a Rare Alternative

The equicorrelated design is favorable to the accept–reject simulator because no choice region is unusually small. We therefore consider two additional designs at $J = 4$ that probe robustness beyond this baseline. Design D2 increases the coeficients to $\beta _ { 0 } =$ (−1.5, 2, 2)<sup>′</sup>, making choices more deterministic conditional on covariates, and replaces $\Omega _ { 0 }$ with the matrix

$$
\Omega _ { 0 } ^ { \mathrm { D 2 } } = \left( \begin{array} { l l l } { { 1 . 0 0 } } & { { 0 . 8 5 } } & { { 0 . 7 5 } } \\ { { 0 . 8 5 } } & { { 1 . 4 0 } } & { { 1 . 0 5 } } \\ { { 0 . 7 5 } } & { { 1 . 0 5 } } & { { 2 . 0 0 } } \end{array} \right) ,\tag{27}
$$

which has unequal variances and large, uneven correlations. Design D3 augments D2 with alternative-specific intercepts $\delta _ { 0 } = ( 1 . 5 , 0 . 5 , - 2 . 0 ) ^ { \prime }$ in the utility diferences, so that alternative 4 is chosen with low probability: its population choice share is 9.2 percent, against 41 percent for the most popular alternative. The intercepts are estimated by absorbing them into the covariate matrix as alternative dummies, with no change to the estimator, and we report the direction criterion separately for the slope and intercept blocks.

Table 3 compares the two stress designs with the $J = 4$ baseline. The overall acceptance rates in D2 and D3, 0.414 and 0.453, exceed the baseline rate of 0.281. D3 nevertheless creates substantial heterogeneity in computational dificulty: the acceptance rate for observations choosing the rare alternative is 0.26, roughly two-fifths of the rate for the most frequently chosen alternative 2 (0.66). The simulator remains efective: on average about two observation evaluations per replication produced no accepted draw, out of 800,000, and the trial cap was reached fewer than five times per replication. The accuracy measures remain in the same broad range across the three designs. The coeficient criterion is smaller in D2 and D3 than at the baseline, whereas the correlation and probability criteria are somewhat larger. Because the data-generating processes difer, these comparisons should be interpreted as robustness diagnostics rather than as a ranking of the designs. Overall, SAUSS continues to perform well under the heterogeneous covariance structure and the rare-alternative design.

Table 4: Multinomial probit estimates, daycare scenario of Boneva et al. (2026), Table 4, column 2; $N = 2 { \mathrm { , } } 8 7 3 .$ Standard errors in parentheses. Times are one fit on a single core.
<table><tr><td>GHK-SML</td><td></td><td>SAUSS</td></tr><tr><td>Child skills</td><td>0.4553 (0.1684)</td><td>0.4641 (0.1809)</td></tr><tr><td>Family outcomes</td><td>1.1614 (0.1876)</td><td>1.2722 (0.1993)</td></tr><tr><td>Maternal earnings (1,000 EUR)</td><td>0.0033 (0.0014)</td><td>0.0039 (0.0016)</td></tr><tr><td>Family&#x27;s opinion</td><td>0.2689 (0.0365)</td><td>0.2872 (0.0378)</td></tr><tr><td>Friends&#x27; opinion</td><td>0.2616 (0.0388)</td><td>0.2843 (0.0399)</td></tr><tr><td> $\Omega _ { 2 1 }$ </td><td>-0.0022 (0.0946)</td><td>-0.0418 (0.1000)</td></tr><tr><td> $\Omega _ { 2 2 }$ </td><td>0.4494 (0.1379)</td><td>0.5840 (0.1764)</td></tr><tr><td>Computation time (s)</td><td>4180</td><td>33</td></tr></table>

## 8 Application: Maternal Labor-Supply Intentions

We revisit the maternal labor-supply model of Boneva et al. (2026), which asks whether respondents’ beliefs about the consequences of maternal employment predict the laborsupply choice that they or their partner would make. Column 2 of its Table 4 fits a multinomial probit to the stated choices of $N \mathrm { ~ = ~ } 2 { , } 8 7 3$ respondents among $J \ = \ 3$ alternatives, which are ‘not working’, ‘part-time’, and ‘full-time’, respectively. In this scenario, full-time daycare is available. Utility is linear in five alternative-specific belief and norm measures and six case-specific demographic controls, so the utility-diferenced model carries $K = 1 9$ parameters. The published estimates use GHK-based simulated maximum likelihood with 2,000 draws.

We re-estimate that column with SAUSS by setting 1,600 epochs, mini-batches of $m = 1 0$ , five accepted simulator draws per observation, Polyak–Ruppert averaging, and a $t ^ { - 0 . 5 0 1 }$ step-size decay. We compare it with GHK simulated maximum likelihood. Because the published cmmprobit estimates in Stata use the alternative normalization $L _ { 1 1 } = { \sqrt { 2 } }$ we divide their coeficients and standard errors by $\sqrt { 2 }$ to express them on the $L _ { 1 1 } = 1$ scale used by mlogit and SAUSS (both in R). On this common scale, our 2,000-draw mlogit GHK coeficients difer from the published cmmprobit coeficients by at most 0.02 published standard errors.

Table 4 reports the five alternative-specific coeficients emphasized in the original paper, together with the error covariance and computation time. The five reported coefficients are broadly similar across the two methods at the stated computational budgets. Each SAUSS estimate lies within one GHK standard error of the corresponding GHK estimate, and the signs and 5% significance classifications agree. The SAUSS coeficients are somewhat larger, accompanied by a fitted $\Omega _ { 2 2 }$ of 0.58, compared with 0.45 for GHK.

This covariance diference is less than one standard error under either fit, although the continued downward movement of the SAUSS estimate near the end of the run indicates some finite-run sensitivity at the reported budget.

The SAUSS standard errors use the finite-run plug-in variance approximation in Remark 6 and are computed from the reported run in 0.7 seconds. After 4.6 million observation-level gradient evaluations, the estimated algorithmic component accounts for 0.16% of the total variance in this calculation, so the estimated variance is dominated by sampling uncertainty. Across ten random seeds, the coeficient estimates vary by only three to five percent of one reported standard error, indicating limited seed sensitivity for $\beta$ at this budget. One caution concerns estimating H using the outer product of simulated scores. For fixed $R ,$ this outer product estimates $\Sigma _ { s }$ plus a simulation-variance component of order $1 / R .$ . Using it in place of H therefore understates the standard errors: in this application, the resulting standard errors are, in median, 0.56 times the simulation-adjusted plug-in standard errors at $R = 5$ and 0.96 times them at $R = 2 0 0$ After accounting for simulation variance, the plug-in standard errors are essentially stable across these values of R.

The methods difer sharply in cost under these reported configurations: 33 seconds for SAUSS compared with 4,180 seconds for the GHK likelihood. Some of that gap is implementation rather than method, and a practitioner content with 200 draws would face a much smaller one, since the GHK fit then takes 163 seconds and moves by at most 0.16 standard errors. Because J = 3, this application provides a comparison in a small-choice-set setting where both methods are computationally feasible.

## 9 Conclusion

This paper develops SAUSS, a stochastic approximation approach to likelihood-based estimation when unbiased simulated scores are available but simulated probabilities are costly or biased inside logarithms. The main argument is that the method-of-simulatedscores construction, which was dificult to combine with deterministic full-sample optimization because of discontinuity, is well suited to mini-batch stochastic approximation. The MNP implementation shows how the general LDV score identity can be turned into an explicit descent algorithm by diferencing relative to alternative 1 and imposing the $L _ { 1 1 } = 1$ scale normalization, with accept–reject diagnostics separating the ideal unbiased recursion from capped finite computation.

The numerical results show that SAUSS achieves accuracy comparable to GHK-based simulated maximum likelihood in the small-choice-set benchmark and remains computationally feasible as the number of alternatives increases. In the empirical application, SAUSS produces broadly similar estimates with substantially lower computation time under the reported implementations.

## A Proofs

## A.1 Setup and notation

For the proofs of Theorems 1 and 2 and Lemmas $2 { - } 6$ , the mini-batch size is $m = 1$ , so that the direction in (3) is

$$
\hat { G } _ { t } ( \theta ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } g ( Y _ { t , r } ^ { * } ( \theta ) , D _ { t } , \theta ) , \qquad \theta _ { t } = \theta _ { t - 1 } - \gamma _ { t } \hat { G } _ { t } ( \theta _ { t - 1 } ) , \qquad t \geq 2 ,\tag{28}
$$

where $D _ { t }$ is the observation drawn at iteration t and $Y _ { t , r } ^ { * } ( \theta )$ is the r-th accepted latent draw of the exact simulator at parameter θ (so that $g ( Y _ { t , r } ^ { * } ( \theta ) , D _ { t } , \theta )$ is the one-draw loss-gradient $g _ { i } ( \theta , U )$ of Section 5 for i the observation drawn at t). As in Section 5, $G _ { t } ( \theta ) = \mathbb { E } _ { t } ^ { * } \hat { G } _ { t } ( \theta )$ is the expectation over the simulation draws given $\mathcal { F } _ { t - 1 }$ and $D _ { t } , \mathcal { R } ( \theta ) = \mathbb { E } ( G _ { t } ( \theta ) | \mathcal { F } _ { t - 1 } )$ $H ( \theta ) = \nabla _ { \theta } \mathcal { R } ( \theta )$ ， $H = H ( \theta _ { 0 } )$ , and $\mathcal { R } ( \theta _ { 0 } ) = 0$ . Observations are drawn independently across iterations (streaming or with-replacement sampling), so $\mathcal { R }$ does not depend on t. The proof of Corollary 1 invokes the fixed-m extension in Remark 5.

For those proofs, the assumptions are those of Section 5 (Assumptions 1–7 for Theorem 1, plus Assumption 8 for Theorem 2) with $m = 1$ ; in particular $q = \operatorname* { m a x } \{ 2 +$ $\delta , 2 p _ { 0 } \}$ with $p _ { 0 } > ( 1 - a ) ^ { \cdot }$ <sup>−1</sup> is the moment order of Assumption 4, $A _ { t } ( \theta ) = A _ { t } ( \theta , D _ { t } ) =$ $V a r _ { | t } ^ { * } ( g ( Y _ { t , r } ^ { * } ( \theta ) , D _ { t } , \theta ) )$ is the conditional simulation covariance of Assumption 5, where $V a r _ { | t } ^ { * }$ denotes the covariance with respect to the simulation draws given $\mathcal { F } _ { t } .$ <sub>−1</sub> and $D _ { t }$ and $S = V _ { 1 , R }$ is the limit in Assumption 7.

As in the main text, all statements indexed by the run length are taken along a sequence in which $T \to \infty$ and $n _ { 0 } = n _ { 0 } ( T ) \to \infty$ , with no relative-rate restriction; p, R, $\gamma _ { 0 }$ , and a are held fixed.

To display the main argument before its technical components, Section A.2 gives the proofs of Theorems 1 and 2 and Corollary 1; Section A.3 then states and proves Lemmas 2–6, which those proofs invoke.

## A.2 Proofs of the main results

Proof of Theorem 1. For $t \geq 2$ , we have

$$
\theta _ { t } = \theta _ { t - 1 } - \gamma _ { t } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } g ( Y _ { t , r } ^ { * } ( \theta _ { t - 1 } ) , D _ { t } , \theta _ { t - 1 } ) .\tag{29}
$$

For $t \geq 2$ , define

$$
\zeta _ { t } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } g ( Y _ { t , r } ^ { * } ( \theta _ { t - 1 } ) , D _ { t } , \theta _ { t - 1 } ) - G _ { t } ( \theta _ { t - 1 } ) .\tag{30}
$$

The term $\zeta _ { t }$ is the simulation component of the martingale-diference innovation. In addition, let

$$
\xi _ { t } = G _ { t } ( \theta _ { t - 1 } ) - \mathbb { E } ( G _ { t } ( \theta _ { t - 1 } ) | \mathcal { F } _ { t - 1 } ) .\tag{31}
$$

The term $\xi _ { t }$ is the observation-sampling component of the martingale-diference innovation.

Recall that $H = \nabla _ { \boldsymbol { \theta } } \mathcal { R } ( \theta _ { 0 } )$

Let

$$
\eta _ { t } = \mathbb { E } ( G _ { t } ( \theta _ { t - 1 } ) | \mathcal { F } _ { t - 1 } ) - H ( \theta _ { t - 1 } - \theta _ { 0 } ) .\tag{32}
$$

The term $\eta _ { t }$ is the nonlinear remainder from linearizing the mean drift around $\theta _ { 0 }$

From (30)–(32), we have

$$
\begin{array} { r } { \theta _ { t } = \theta _ { t - 1 } - \gamma _ { t } H ( \theta _ { t - 1 } - \theta _ { 0 } ) - \gamma _ { t } ( \zeta _ { t } + \zeta _ { t } + \eta _ { t } ) . } \end{array}\tag{33}
$$

Let

$$
\Delta _ { t } = \theta _ { t } - \theta _ { 0 } .
$$

Then

$$
\begin{array} { r } { \Delta _ { t } = \Delta _ { t - 1 } - \gamma _ { t } H \Delta _ { t - 1 } - \gamma _ { t } \big ( \underbrace { \zeta _ { t } + \xi _ { t } } _ { \mathrm { M D S - C L T } } + \underbrace { \eta _ { t } } _ { \mathrm { n o n l i n e a r i t y } } \big ) . } \end{array}\tag{34}
$$

Let $\tau , \rho ,$ and $\mathcal { E } = \{ \tau = \infty \}$ be as in Lemma 2; by Lemma $2 ( \mathrm { i i } )$ , lim inf<sub>T</sub> $\mathbb { P } ( \mathcal { E } ) \geq 1 - \epsilon$ once $\gamma _ { 0 } \le \bar { \gamma } ( \epsilon )$

Let $\widetilde { \varepsilon } _ { t }$ be the continued innovation defined in Lemma 4. Set $\widetilde { \Delta } _ { 1 } ^ { 1 } = \Delta _ { \scriptscriptstyle \cdot }$ <sub>1</sub> and, for $t \geq 2$ define the auxiliary linear recursion

$$
\widetilde { \Delta } _ { t } ^ { 1 } = ( I - \gamma _ { t } H ) \widetilde { \Delta } _ { t - 1 } ^ { 1 } - \gamma _ { t } \widetilde { \varepsilon } _ { t } .\tag{35}
$$

The claimed statements follow from the three results below. The approximation remainders are established on E: they are of the form $Z _ { T } \mathbb { I } \{ \mathcal { E } \}$ with $Z _ { T } \mathbb { I } \{ \mathcal { E } \} \overset { p } { \to } 0$

Lemma 3:

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \left\| \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { \lfloor T r \rfloor } ( \Delta _ { t } - \widetilde { \Delta } _ { t } ^ { 1 } ) \right\| \mathbb { I } \{ \mathcal { E } \} = o _ { \mathbb { P } } ( 1 )
$$

Lemma 4:

$$
- H ^ { - 1 } \widetilde { M } _ { T } ( r ) \Rightarrow ( H ^ { - 1 } S H ^ { - 1 } ) ^ { 1 / 2 } W ( r )
$$

Lemma 5:

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \left\| \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { \lfloor T r \rfloor } \widetilde { \Delta } _ { t } ^ { 1 } + H ^ { - 1 } \widetilde { M } _ { T } ( r ) \right\| = o _ { \mathbb { P } } ( 1 ) .
$$

Let $\widetilde { X } _ { T } ( r ) = - H ^ { - 1 } \widetilde { M } _ { T } ( r )$ . By Lemma 4, ${ \widetilde { X } } _ { T } \Rightarrow X$ . On $\mathcal { E } .$ , the continued innovations used in $\widetilde { M } _ { T }$ equal the actual innovations $\zeta _ { t } + \xi _ { t }$ at every iteration. Lemmas 3 and 5 therefore give

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \Vert X _ { T } ( r ) - \widetilde { X } _ { T } ( r ) \Vert \mathbb { I } \{ \mathcal { E } \} = o _ { \mathbb { P } } ( 1 ) .
$$

Consequently, for every bounded continuous f on $D [ 0 , 1 ] ^ { p }$ 2

$$
| \mathbb { E } f ( X _ { T } ) - \mathbb { E } f ( \widetilde { X } _ { T } ) | \le \mathbb { E } \{ | f ( X _ { T } ) - f ( \widetilde { X } _ { T } ) | \mathbb { I } \{ \mathcal { E } \} \} + 2 \operatorname* { s u p } | f | \mathbb { P } ( \mathcal { E } ^ { c } ) ,
$$

whose lim sup is at most 2ϵ sup |f|. This proves the stated ϵ-form FCLT; consistency and the linear representation follow from Lemmas 2, 3, and 5. □

Proof of Theorem 2. Recall that $\kappa _ { t } = h ( \theta _ { t } )$ and $\kappa _ { 0 } = h ( \theta _ { 0 } )$ , where h is scalar-valued. Let $\tau , \ u _ { t }$ , and $\mathcal { E } = \{ \tau = \infty \}$ be as in Lemma 2. On E, $\theta _ { t } \in B _ { \rho } \subset \mathcal N _ { 0 }$ for every t. Because $B _ { \rho }$ is convex, the line segment joining $\theta _ { t }$ and $\theta _ { 0 }$ also lies in $B _ { \rho }$ on this event. Hence, for some $\tilde { \theta } _ { t }$ on that line segment,

$$
\begin{array} { r l r } {  {  \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { [ T r ] } ( \kappa _ { t } - \kappa _ { 0 } ) - \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { [ T r ] } \nabla h ( \theta _ { 0 } ) ^ { \prime } \Delta _ { t }   { \mathbb { I } } \{  { \mathcal { E } } \} } } \\ & { } & { \leq \frac { 1 } { 2 \sqrt { T } } \sum _ { t = 1 } ^ { [ T r ] } \| \Delta _ { t } \| ^ { 2 } \| \nabla ^ { 2 } h ( \widetilde { \theta } _ { t } ) \|  { \mathbb { I } } \{  { \mathcal { E } } \} \leq \frac { C } { \sqrt { T } } \sum _ { t = 1 } ^ { [ T r ] } \| \Delta _ { t } \| ^ { 2 }  { \mathbb { I } } \{ \tau > t \} . } \end{array}
$$

by Assumption 8. By Lemma 6,

$$
\begin{array} { r l r } {  { \mathbb { E } ( \operatorname* { s u p } _ { r \in [ 0 , 1 ] } \frac { C } { \sqrt { T } } \sum _ { t = 1 } ^ { [ T r ] } \| \Delta _ { t } \| ^ { 2 } \mathbb { I } \{ \tau > t \} ) } } \\ & { } & { \leq \frac { C } { \sqrt { T } } \sum _ { t = 1 } ^ { T } u _ { t } } \\ & { } & { \leq \frac { C } { \sqrt { T } } \sum _ { t = 1 } ^ { \infty } \exp ( - C _ { 6 } t ^ { 1 - \alpha } ) + \frac { C } { \sqrt { T } } \sum _ { t = 1 } ^ { T } t ^ { - \alpha } = O ( T ^ { - 1 / 2 } + T ^ { 1 / 2 - \alpha } ) . } \end{array}
$$

Let $R _ { T } ( r )$ denote the diference on the left-hand side of the preceding Taylor bound. Since $\mathbb { I } \{ \mathcal { E } \} \le \mathbb { I } \{ \tau > t \}$ for every t, Markov’s inequality gives

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } | R _ { T } ( r ) | \mathbb { I } \{ \mathcal { E } \} = o _ { \mathbb { P } } ( 1 ) , \qquad \operatorname* { l i m } _ { T \to \infty } \mathbb { P } \Bigg ( \operatorname* { s u p } _ { r \in [ 0 , 1 ] } | R _ { T } ( r ) | > \delta \Bigg ) \leq \epsilon
$$

for every $\delta > 0$ , where the second conclusion also uses $\mathbb { P } ( \mathcal { E } ^ { c } ) \le \epsilon + o ( 1 )$ from Lemma

2(ii). Combining the stopped remainder bound with Lemmas 3–5 yields

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \left. \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { [ T r ] } ( \kappa _ { t } - \kappa _ { 0 } ) + \frac { 1 } { \sqrt { T } } \sum _ { t = 2 } ^ { [ T r ] } \nabla h ( \theta _ { 0 } ) ^ { \prime } H ^ { - 1 } ( \zeta _ { t } + \xi _ { t } ) \right. \mathbb { I } \{ \mathcal { E } \} = o _ { \mathbb { P } } ( 1 ) .
$$

On $\mathcal { E } ,$ the actual innovations in the preceding display equal the continued innovations of Lemma 4. That lemma and $\mathbb { P } ( \mathcal { E } ^ { c } ) \le \epsilon + o ( 1 )$ therefore give the claimed weak convergence in the same ϵ-sense as Theorem 1. □

Proof of Corollary 1. Let

$$
\Sigma _ { \mathrm { a l g } } = H ^ { - 1 } V _ { m , R } H ^ { - 1 } , \qquad \Sigma _ { A } = A \Sigma _ { \mathrm { a l g } } A ^ { \prime } .
$$

The matrix $\Sigma _ { A }$ is positive definite because A has full row rank and $\Sigma _ { \mathrm { a l g } }$ is positive definite. The FCLT in Theorem 1, together with Remark 5 for fixed m, gives

$$
A X _ { T } ( \cdot ) \Rightarrow \Sigma _ { A } ^ { 1 / 2 } W _ { \ell } ( \cdot )
$$

in the same ϵ-sense as that theorem. For every $t = 1 , \dots , T$ 2

$$
\frac { 1 } { \sqrt { T } } \sum _ { s = 1 } ^ { t } A ( \theta _ { s } - \bar { \theta } _ { T } ) = A X _ { T } ( t / T ) - \frac { t } { T } A X _ { T } ( 1 ) .
$$

Because the limiting process has continuous paths, the continuous mapping theorem and a Riemann-sum argument yield

$$
\widehat { V } _ { \mathrm { R S } } ( A ) \Rightarrow \Sigma _ { A } ^ { 1 / 2 } \left\{ \int _ { 0 } ^ { 1 } \bar { W } _ { \ell } ( r ) \bar { W } _ { \ell } ( r ) ^ { \prime } d r \right\} \Sigma _ { A } ^ { 1 / 2 } .
$$

Under $A \theta _ { 0 } = a _ { 0 }$

$$
\sqrt { T } ( A \bar { \theta } _ { T } - a _ { 0 } ) = A X _ { T } ( 1 ) \Rightarrow \Sigma _ { A } ^ { 1 / 2 } W _ { \ell } ( 1 ) .
$$

The integrated Brownian-bridge matrix is positive definite almost surely. Another application of the continuous mapping theorem therefore gives the stated self-normalized limit after the factors $\Sigma _ { A } ^ { 1 / 2 }$ cancel. Finally, if ${ \hat { H } } \ { \xrightarrow { p } } \ H$ and $\hat { V } _ { m , R } \stackrel { p } {  } V _ { m , R }$ , then

$$
\hat { H } ^ { - 1 } \hat { V } _ { m , R } \hat { H } ^ { - 1 } \stackrel { p } {  } H ^ { - 1 } V _ { m , R } H ^ { - 1 }
$$

by the continuous mapping theorem, proving the stated sandwich convergence. □

## A.3 Supporting lemmas

Lemma 2. Suppose Assumptions 1–7 hold with $m = 1$ . There exist $\rho > 0$ and $c _ { 0 } > 0$ such that, with

$$
\tau = \operatorname* { i n f } \{ t \geq 1 : \| \Delta _ { t } \| > \rho \} , \qquad u _ { t } = \mathbb { E } \left( \| \Delta _ { t } \| ^ { 2 } \mathbb { I } \{ \tau > t \} \right) ,
$$

the following hold for all $\gamma _ { 0 } \le \bar { \gamma }$ , where $\bar { \gamma } > 0$ depends only on $\rho , \ c _ { 0 } .$ , and the constants in Assumptions 1–7:

(i) for all $t \geq 2$

$$
u _ { t } \leq ( 1 - 2 c _ { 0 } \gamma _ { t } + C \gamma _ { t } ^ { 2 } ) u _ { t - 1 } + C \gamma _ { t } ^ { 2 } ;\tag{36}
$$

(ii) for every $\epsilon > 0$ , if in addition $\gamma _ { 0 } \le \bar { \gamma } ( \epsilon )$ , then

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { s u p } _ { } \mathbb { P } ( \tau < \infty ) \leq \epsilon ,
$$

that is, $\{ \| \Delta _ { t } \| \leq \rho \forall t \geq 1 \}$ holds with probability at least $1 - \epsilon - o ( 1 )$

(iii) $u _ { 1 } \leq C$ , and, for all $t \geq 2$

$$
\mathbb { E } \left( \Vert \Delta _ { t } \Vert ^ { 2 } \mathbb { I } \{ \tau > t \} \right) \leq C \gamma _ { t } .\tag{37}
$$

Proof. Step 1: local drift and remainder bounds. Since $\mathcal { R } ( \theta _ { 0 } ) = 0$ and $H ( \cdot )$ is Lipschitz with constant L on a neighborhood of $\theta _ { 0 }$

$$
\mathcal { R } ( \theta ) = \int _ { 0 } ^ { 1 } H \big ( \theta _ { 0 } + s ( \theta - \theta _ { 0 } ) \big ) d s ( \theta - \theta _ { 0 } )
$$

for all $\theta$ in that neighborhood. Because H is symmetric with $\lambda _ { \operatorname* { m i n } } ( H ) > c ,$ choose $\rho > 0$ so small that $B _ { \rho } = \{ \theta : \| \theta - \theta _ { 0 } \| \leq \rho \}$ lies in the neighborhood and $L \rho \leq c / 2$ . Then for all $\theta \in B _ { \rho }$

$$
\begin{array} { r } { ( \theta - \theta _ { 0 } ) ^ { \prime } \mathcal { R } ( \theta ) \geq ( c - L \rho ) \| \theta - \theta _ { 0 } \| ^ { 2 } \geq c _ { 0 } \| \theta - \theta _ { 0 } \| ^ { 2 } , \qquad c _ { 0 } : = c / 2 , } \end{array}\tag{38}
$$

and, using $\| H ( \theta ) \| \leq C$ on $B _ { \rho }$ ,

$$
\begin{array} { r } { \| \mathcal { R } ( { \boldsymbol { \theta } } ) \| \leq C \| { \boldsymbol { \theta } } - { \boldsymbol { \theta } } _ { 0 } \| , \qquad \| \mathcal { R } ( { \boldsymbol { \theta } } ) - H ( { \boldsymbol { \theta } } - { \boldsymbol { \theta } } _ { 0 } ) \| \leq \frac { L } { 2 } \| { \boldsymbol { \theta } } - { \boldsymbol { \theta } } _ { 0 } \| ^ { 2 } . } \end{array}\tag{39}
$$

The second inequality in (39) is the bound $\| \eta _ { t } \| \leq C \| \Delta _ { t - 1 } \| ^ { 2 }$ used below whenever $\theta _ { t - 1 } \in$ $B _ { \rho }$

Step 2: one-step inequality. Recall (34): $\Delta _ { t } = \Delta _ { t - 1 } - \gamma _ { t } \{ \mathcal { R } ( \theta _ { t - 1 } ) + \zeta _ { t } + \xi _ { t } \}$ , with $\mathbb { E } ( \zeta _ { t } + \xi _ { t } \vert \mathcal { F } _ { t - 1 } ) = 0$ ; here $\begin{array} { r } { \zeta _ { t } + \xi _ { t } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } g ( Y _ { t , r } ^ { * } ( \theta _ { t - 1 } ) , D _ { t } , \theta _ { t - 1 } ) - \mathcal { R } ( \theta _ { t - 1 } ) } \end{array}$ , and the

conditional mean is zero because $( D _ { t } , Y _ { t , \cdot } ^ { * } )$ is generated independently of $\mathcal { F } _ { t - 1 }$ and $\theta _ { t - 1 }$ is $\mathcal { F } _ { t - 1 }$ -measurable. Squaring,

$$
\begin{array} { r } { \mathbb { E } ( \| \Delta _ { t } \| ^ { 2 } | \mathcal { F } _ { t - 1 } ) = \| \Delta _ { t - 1 } \| ^ { 2 } - 2 \gamma _ { t } \Delta _ { t - 1 } ^ { \prime } \mathcal { R } ( \theta _ { t - 1 } ) + \gamma _ { t } ^ { 2 } \mathbb { E } \big ( \| \mathcal { R } ( \theta _ { t - 1 } ) + \zeta _ { t } + \xi _ { t } \| ^ { 2 } \big | \mathcal { F } _ { t - 1 } \big ) . } \end{array}
$$

By Assumption 4 (with $q \geq 2 )$ and the substitution rule for conditional expectations,

$$
\mathbb { E } ( \Vert \zeta _ { t } + \xi _ { t } \Vert ^ { 2 } \mid \mathcal { F } _ { t - 1 } ) \leq 2 \mathbb { E } \left( \left. \frac { 1 } { R } \sum _ { r } g ( Y _ { t , r } ^ { * } ( \theta _ { t - 1 } ) , D _ { t } , \theta _ { t - 1 } ) \right. ^ { 2 } \Bigg | \mathcal { F } _ { t - 1 } \right) + 2 \Vert \mathcal { R } ( \theta _ { t - 1 } ) \Vert ^ { 2 }
$$

on $\{ \theta _ { t - 1 } \in B _ { \rho } \}$ , using (39). Hence, on the event $\{ \tau > t - 1 \} = \{ \theta _ { s } \in B _ { \rho } , \ s \leq t - 1 \}$ , (38) gives

$$
\begin{array} { r } { \mathbb { E } ( \| \Delta _ { t } \| ^ { 2 } | \mathcal { F } _ { t - 1 } ) \leq \| \Delta _ { t - 1 } \| ^ { 2 } ( 1 - 2 c _ { 0 } \gamma _ { t } + C \gamma _ { t } ^ { 2 } ) + C \gamma _ { t } ^ { 2 } \qquad \mathrm { o n ~ } \{ \tau > t - 1 \} . } \end{array}\tag{40}
$$

The event $\{ \tau > t - 1 \}$ is $\mathcal { F } _ { t - 1 } { \mathrm { - m e a s u r a b l e } }$ . Multiplying (40) by $\mathbb { I } \{ \tau > t - 1 \}$ , using $\mathbb { I } \{ \tau > t \} \le \mathbb { I } \{ \tau > t - 1 \}$ and taking expectations yields (36), since $\mathbb { E } ( \| \Delta _ { t } \| ^ { 2 } \mathbb { I } \{ \tau > t \} ) \le$ $\mathbb { E } \{ \mathbb { I } \{ \tau > t - 1 \} \mathbb { E } ( \| \Delta _ { t } \| ^ { 2 } | \mathcal { F } _ { t - 1 } ) \}$ and $\mathbb { P } ( \tau > t - 1 ) \le 1$ . This proves (i). Note that no conditioning on a trajectory-dependent event is involved: the drift inequality is applied only on the $\mathcal { F } _ { t - 1 } .$ -measurable event $\{ \tau > t - 1 \}$

Step 3: the iterates stay in $B _ { \rho }$ with high probability. Let $V _ { t } = \| \Delta _ { t \wedge \tau } \| ^ { 2 }$ and take $\bar { \gamma }$ such that $C \bar { \gamma } \leq 2 c _ { 0 }$ , so that $1 - 2 c _ { 0 } \gamma _ { t } + C \gamma _ { t } ^ { 2 } \leq 1$ for all $t \geq 2$ when $\gamma _ { 0 } \leq \bar { \gamma }$ . On $\{ \tau > t - 1 \}$ $V _ { t } = \| \Delta _ { t } \| ^ { 2 }$ and (40) gives $\mathbb { E } ( V _ { t } | \mathcal { F } _ { t - 1 } ) \le V _ { t - 1 } + C \gamma _ { t } ^ { 2 } ; \mathrm { ~ o n ~ } \{ \tau \le t - 1 \} , V _ { t } = V _ { t - 1 }$ . Hence

$$
N _ { t } : = V _ { t } + C \sum _ { s > t } \gamma _ { s } ^ { 2 }
$$

is a nonnegative supermartingale with respect to $\mathcal { F } _ { t }$ . Fix $\epsilon > 0$ and let $B = \{ \| \Delta _ { 1 } \| \leq$ $\rho \sqrt { \epsilon / 2 } \} \in { \mathcal { F } } _ { 1 } ; \mathbb { P } ( B ) \to 1 \mathrm { ~ a s ~ } n _ { 0 } = n _ { 0 } ( T ) \to \infty$ because $\theta _ { 1 } = \widetilde { \theta } _ { n _ { 0 } } \overset { p } {  } \theta _ { 0 }$ . On $B , N _ { 1 } \le \rho ^ { 2 } \epsilon / 2 +$ $\begin{array} { r } { C \gamma _ { 0 } ^ { 2 } \sum _ { s \ge 1 } s ^ { - 2 a } \le \rho ^ { 2 } \epsilon } \end{array}$ once $\gamma _ { 0 } \le \bar { \gamma } ( \epsilon )$ . Since $\{ \tau < \infty \} \subset \{ \operatorname { s u p } _ { t } V _ { t } > \rho ^ { 2 } \} \subset \{ \operatorname { s u p } _ { t } N _ { t } \geq \rho ^ { 2 } \}$ Ville’s maximal inequality for nonnegative supermartingales gives

$$
\mathbb { P } ( \tau < \infty | \mathcal { F } _ { 1 } ) \le \frac { N _ { 1 } } { \rho ^ { 2 } } \le \epsilon \qquad \mathrm { o n } \ B ,
$$

so $\mathbb { P } ( \tau < \infty ) \le \epsilon + \mathbb { P } ( B ^ { c } ) = \epsilon + o ( 1 )$ . This proves (ii).

Step 4. Part (iii) follows from (i) and Lemma 6 below, whose proof uses only the recursion (36), since $\exp ( - C _ { 6 } t ^ { 1 - a } ) \leq C t ^ { - a }$ □

Remark 7. The smallness condition on $\gamma _ { 0 }$ can instead be ensured by using $\gamma _ { t } = \gamma _ { 0 } ( t -$ $1 + t _ { 0 } ) ^ { - a }$ with $t _ { 0 }$ suficiently large. This shift makes $1 - 2 c _ { 0 } \gamma _ { t } + C \gamma _ { t } ^ { 2 } \leq 1$ from the first

stochastic update onward. Parts (i) and (iii) do not use the symmetry of H beyond (38);   
for likelihood problems H is the population information matrix and is symmetric.

Lemma 3. Let $\mathcal { E } = \{ \tau = \infty \}$ be as in Lemma ${ \it 2 } ,$ and let $\widetilde { \Delta } _ { t } ^ { 1 }$ satisfy (35). Then

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \left\| \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { \lfloor T r \rfloor } ( \Delta _ { t } - \widetilde { \Delta } _ { t } ^ { 1 } ) \right\| \mathbb { I } \{ \mathcal { E } \} = o _ { \mathbb { P } } ( 1 ) .
$$

Proof. For integers $j \le t$ , define the deterministic transition matrices

$$
\Phi _ { j , t } : = \prod _ { \ell = j } ^ { t } ( I - \gamma _ { \ell } H ) , \qquad \Phi _ { j , t } : = I \quad \mathrm { w h e n ~ } j > t .
$$

For $k \geq 1$ , set

$$
\begin{array} { r l r } & { \displaystyle { B _ { k } : = \left( I + \sum _ { t = 2 } ^ { k } \Phi _ { 2 , t } \right) \Delta _ { 1 } , } } & \\ & { \displaystyle { K _ { j , k } : = \gamma _ { j } \sum _ { t = j } ^ { k } \Phi _ { j + 1 , t } , \qquad w _ { j } ^ { k } : = K _ { j , k } - H ^ { - 1 } , \qquad 2 \le j \le k } , } & \end{array}\tag{41}
$$

and set $B _ { 0 } : = 0$ . The standard product bounds for $\gamma _ { t } = \gamma _ { 0 } ( t - 1 ) ^ { - a }$ and positive definite H give

$$
\operatorname* { s u p } _ { k \geq 1 } \| B _ { k } \| \leq C \| \Delta _ { 1 } \| , \qquad \operatorname* { s u p } _ { 2 \leq j \leq k } \| w _ { j } ^ { k } \| \leq C , \qquad \sum _ { j = 2 } ^ { k } \| w _ { j } ^ { k } \| \leq C k ^ { a } .\tag{42}
$$

These are the usual Polyak–Ruppert weight bounds; see Lemma 2 of Polyak and Juditsky (1992) and Lemma 2 of Zhu and Dong (2021).

On E, Lemma 4 gives $\widetilde { \varepsilon } _ { j } = \zeta _ { j } + \xi _ { j }$ for every $j$ . Iterating (34) and (35), which have the same initial condition, therefore yields the exact finite-horizon identity

$$
\sum _ { t = 1 } ^ { k } ( \Delta _ { t } - \widetilde { \Delta } _ { t } ^ { 1 } ) = - \sum _ { j = 2 } ^ { k } K _ { j , k } \eta _ { j } = - \sum _ { j = 2 } ^ { k } ( H ^ { - 1 } + w _ { j } ^ { k } ) \eta _ { j } \qquad \mathrm { o n } \ \mathcal E .\tag{43}
$$

On $\{ \tau > j - 1 \}$ , (39) gives $\| \eta _ { j } \| \le C \| \Delta _ { j - 1 } \| ^ { 2 }$ . Hence (42) and (43) imply

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \frac { 1 } { \sqrt { T } } \left\| \sum _ { t = 1 } ^ { \lfloor T r \rfloor } ( \Delta _ { t } - \widetilde { \Delta } _ { t } ^ { 1 } ) \right\| \mathbb { I } \{ \mathcal { E } \} \leq \frac { C } { \sqrt { T } } \sum _ { j = 2 } ^ { T } \| \Delta _ { j - 1 } \| ^ { 2 } \mathbb { I } \{ \tau > j - 1 \} .\tag{44}
$$

Because the initialization may depend on $T .$ , we use a finite-horizon probability bound.

By Lemma 2(iii),

$$
\begin{array} { l } { { \displaystyle \mathbb E \left[ \frac { C } { \sqrt T } \sum _ { j = 2 } ^ { T } \| \Delta _ { j - 1 } \| ^ { 2 } \mathbb I \{ \tau > j - 1 \} \right] } } \\ { { \displaystyle \quad \le \frac { C } { \sqrt T } \left( u _ { 1 } + C \sum _ { j = 3 } ^ { T } \gamma _ { j - 1 } \right) } } \\ { { \displaystyle \quad = O ( T ^ { - 1 / 2 } ) + O ( T ^ { 1 / 2 - a } ) = o ( 1 ) , } } \end{array}
$$

where the last equality uses ${ \textstyle \sum _ { j = 3 } ^ { T } \gamma _ { j - 1 } = O ( T ^ { 1 - a } ) }$ and $a > 1 / 2$ . Markov’s inequality therefore implies that the right-hand side of (44) converges to zero in probability. □

Lemma 4 (MDS-FCLT). For $t \geq 2$ , define

$$
\varepsilon _ { t } ( \theta ) : = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } g ( Y _ { t , r } ^ { * } ( \theta ) , D _ { t } , \theta ) - \mathbb { E } \{ G _ { t } ( \theta ) \mid \mathcal { F } _ { t - 1 } \} .
$$

On an enlarged probability space, let $\varepsilon _ { t } ^ { 0 }$ be a fresh innovation generated at $\theta _ { 0 }$ that, conditional on $\mathcal { F } _ { t - 1 }$ , has the same distribution as $\varepsilon _ { t } ( \theta _ { 0 } )$ and is independent of the actual iteration-t draws. Continue to denote the enlarged filtration by $\mathcal { F } _ { t }$ , and define

$$
\widetilde { \varepsilon } _ { t } : = \mathbb { I } \{ \tau > t - 1 \} \varepsilon _ { t } ( \theta _ { t - 1 } ) + \mathbb { I } \{ \tau \leq t - 1 \} \varepsilon _ { t } ^ { 0 } , \qquad \widetilde { M } _ { T } ( r ) : = \frac { 1 } { \sqrt { T } } \sum _ { t = 2 } ^ { \lfloor T r \rfloor } \widetilde { \varepsilon } _ { t } .
$$

Then

$$
- H ^ { - 1 } \widetilde { M } _ { T } ( r ) \Rightarrow ( H ^ { - 1 } S H ^ { - 1 } ) ^ { 1 / 2 } W ( r ) \quad i n D [ 0 , 1 ] ^ { p } ,
$$

where $S ~ = ~ V _ { 1 , R }$ and W is a p-dimensional standard Wiener process. In particular, $\widetilde { M } _ { T } ( 1 ) \stackrel { d } { \to } { \mathcal { N } } ( 0 , S )$ . Moreover, on $\mathcal { E } = \{ \tau = \infty \}$

$$
\widetilde { \varepsilon } _ { t } = \varepsilon _ { t } ( \theta _ { t - 1 } ) = \zeta _ { t } + \xi _ { t } \qquad f o r \ e v e r y \ t \geq 2 .
$$

Proof. Because $\{ \tau > t - 1 \} \in \mathcal { F } _ { t - 1 }$ and each component innovation has conditional mean zero,

$$
\mathbb { E } ( \widetilde { \varepsilon } _ { t } \mid \mathcal { F } _ { t - 1 } ) = 0 .
$$

Thus $\{ \widetilde { \varepsilon } _ { t } , \mathcal { F } _ { t } \}$ is a martingale-diference sequence. Assumption 4, Jensen’s inequality, and the fact that $\theta _ { t - 1 } \in B _ { \rho } \subset \mathcal { N } _ { 0 }$ on $\{ \tau > t - 1 \}$ give the uniform conditional moment bound

$$
\mathbb { E } ( \| \widetilde { \varepsilon } _ { t } \| ^ { 2 + \delta } \ | \ F _ { t - 1 } ) \le C .\tag{45}
$$

For the predictable quadratic variation, write

$$
C _ { t } ( \theta ) : = \mathbb { E } \{ \varepsilon _ { t } ( \theta ) \varepsilon _ { t } ( \theta ) ^ { \prime } \mid \mathcal { F } _ { t - 1 } \} = \operatorname { V a r } \{ G _ { t } ( \theta ) \mid \mathcal { F } _ { t - 1 } \} + \frac { 1 } { R } \mathbb { E } \{ A _ { t } ( \theta , D _ { t } ) \mid \mathcal { F } _ { t - 1 } \} ,
$$

where $A _ { t } ( \theta , D _ { t } )$ is the simulation covariance conditional on $\mathcal { F } _ { t - 1 }$ and $D _ { t }$ . By construction,

$$
\widetilde { C } _ { t } : = \mathbb { E } ( \widetilde { \varepsilon } _ { t } \widetilde { \varepsilon } _ { t } ^ { \prime } | \mathcal { F } _ { t - 1 } ) = \mathbb { I } \{ \tau > t - 1 \} C _ { t } ( \theta _ { t - 1 } ) + \mathbb { I } \{ \tau \leq t - 1 \} C _ { t } ( \theta _ { 0 } ) .
$$

Assumption 3 and the integral mean-value formula imply

$$
\mathbb { E } \{ \| G _ { t } ( \theta ) - G _ { t } ( \theta _ { 0 } ) \| ^ { 2 } | \mathcal { F } _ { t - 1 } \} \le C \| \theta - \theta _ { 0 } \| ^ { 2 } \qquad ( \theta \in B _ { \rho } ) .
$$

Together with the bounded second moments and Assumption 5, this yields

$$
\| C _ { t } ( \theta ) - C _ { t } ( \theta _ { 0 } ) \| \le C \{ \| \theta - \theta _ { 0 } \| + \| \theta - \theta _ { 0 } \| ^ { 2 } \} \qquad ( \theta \in B _ { \rho } ) .
$$

Consequently, Lemma 2(iii) and Cauchy–Schwarz give

$$
\begin{array} { r l } & { \mathbb { E } \| \widetilde { C } _ { t } - C _ { t } ( \theta _ { 0 } ) \| \le C \mathbb { E } \big [ \{ \| \Delta _ { t - 1 } \| + \| \Delta _ { t - 1 } \| ^ { 2 } \} \mathbb { I } \{ \tau > t - 1 \} \big ] } \\ & { \qquad \le C \{ \gamma _ { t - 1 } ^ { 1 / 2 } + \gamma _ { t - 1 } \} \longrightarrow 0 . } \end{array}
$$

Assumption $7$ gives $C _ { t } ( \theta _ { 0 } ) \stackrel { p } {  } S$ . The uniform conditional $( 2 + \delta )$ -moment bound implies uniform integrabili $\mathrm { t y , }$ so

$$
\mathbb { E } \Vert \widetilde { C } _ { t } - S \Vert \longrightarrow 0 .
$$

It follows by Ces\`aro summation that

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \| { \frac { 1 } { T } } \sum _ { t = 2 } ^ { \lfloor T r \rfloor } { \widetilde C } _ { t } - r S \| \overset { p } {  } 0 .\tag{46}
$$

For every $\eta > 0$ , (45) gives the conditional Lindeberg bound

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 2 } ^ { T } \mathbb { E } \bigg [ \| \widetilde { \varepsilon } _ { t } \| ^ { 2 } \mathbb { I } \{ \| \widetilde { \varepsilon } _ { t } \| > \eta \sqrt { T } \} \mid \mathcal { F } _ { t - 1 } \bigg ] } \\ & { \qquad \leq \frac { 1 } { \eta ^ { \delta } T ^ { 1 + \delta / 2 } } \displaystyle \sum _ { t = 2 } ^ { T } \mathbb { E } ( \| \widetilde { \varepsilon } _ { t } \| ^ { 2 + \delta } \mid \mathcal { F } _ { t - 1 } ) \leq \frac { C } { \eta ^ { \delta } T ^ { \delta / 2 } } \longrightarrow 0 . } \end{array}
$$

The martingale functional central limit theorem (see, e.g., Hall and Heyde, 1980, Theorem 4.2), applied using (46) and the conditional Lindeberg condition, gives

$$
\widetilde { M } _ { T } ( r ) \Rightarrow S ^ { 1 / 2 } W ( r ) .
$$

Premultiplication by $- H ^ { - 1 }$ gives the stated functional limit. The final equality on $\mathcal { E }$ follows immediately from the definition of $\widetilde { \varepsilon } _ { t }$ □

Lemma 5. Let $\widetilde { \Delta } _ { t } ^ { 1 }$ satisfy (35). Then

$$
\operatorname* { s u p } _ { r \in [ 0 , 1 ] } \left\| \frac { 1 } { \sqrt { T } } \sum _ { t = 1 } ^ { \lfloor T r \rfloor } \widetilde { \Delta } _ { t } ^ { 1 } + H ^ { - 1 } \widetilde { M } _ { T } ( r ) \right\| = o _ { \mathbb { P } } ( 1 ) .
$$

Proof. Iterating (35) and summing through an arbitrary integer $k \geq 0$ gives the exact finite-horizon representation

$$
\sum _ { t = 1 } ^ { k } \widetilde { \Delta } _ { t } ^ { 1 } = B _ { k } - \sum _ { j = 2 } ^ { k } K _ { j , k } \widetilde { \varepsilon } _ { j } = - H ^ { - 1 } \sum _ { j = 2 } ^ { k } \widetilde { \varepsilon } _ { j } + B _ { k } - \sum _ { j = 2 } ^ { k } w _ { j } ^ { k } \widetilde { \varepsilon } _ { j } ,\tag{47}
$$

where $B _ { k } , K _ { j , k }$ , and $\boldsymbol { w } _ { j } ^ { k }$ are defined in (41); all sums are empty when their upper limit is below their lower limit. By (42) and Assumption 2,

$$
\frac { 1 } { \sqrt { T } } \operatorname* { m a x } _ { 0 \le k \le T } \| B _ { k } \| \le \frac { C } { \sqrt { T } } \| \Delta _ { 1 } \| = o _ { \mathbb { P } } ( 1 ) .
$$

Because $q \geq 2 p _ { 0 }$ in Assumption 4, the same argument as for (45) gives

$$
\mathbb { E } ( \| \widetilde { \varepsilon } _ { j } \| ^ { 2 p _ { 0 } } \mid { \mathcal { F } } _ { j - 1 } ) \leq C .
$$

For each fixed k, Burkholder’s inequality and (42) therefore imply

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left\| \sum _ { j = 2 } ^ { k } w _ { j } ^ { k } \widetilde { \varepsilon } _ { j } \right\| ^ { 2 p _ { 0 } } \leq C \mathbb { E } \left( \sum _ { j = 2 } ^ { k } \| w _ { j } ^ { k } \| ^ { 2 } \| \widetilde { \varepsilon } _ { j } \| ^ { 2 } \right) ^ { p _ { 0 } } } & { } \\ & { \leq C \left( \displaystyle \sum _ { j = 2 } ^ { k } \| w _ { j } ^ { k } \| ^ { 2 } \right) ^ { p _ { 0 } } \leq C k ^ { a p _ { 0 } } . } \end{array}
$$

Hence, for every $\delta > 0$ , a union bound and Markov’s inequality give

$$
\begin{array} { r l } & { \mathbb { P } \left( \displaystyle \operatorname* { m a x } _ { 2 \leq k \leq T } \frac { 1 } { \sqrt { T } } \left\| \displaystyle \sum _ { j = 2 } ^ { k } w _ { j } ^ { k } \widetilde { \varepsilon } _ { j } \right\| > \delta \right) } \\ & { \qquad \leq \displaystyle \frac { C } { \delta ^ { 2 p _ { 0 } } T ^ { p _ { 0 } } } \displaystyle \sum _ { k = 2 } ^ { T } k ^ { a p _ { 0 } } = O \big ( T ^ { 1 - ( 1 - a ) p _ { 0 } } \big ) \longrightarrow 0 , } \end{array}
$$

because $p _ { 0 } > ( 1 - a ) ^ { - 1 }$ . Taking $\begin{array} { r } { k \ = \ \lfloor T r \rfloor } \end{array}$ in (47) proves the result uniformly over $r \in [ 0 , 1 ]$ □

Lemma 6. Suppose $\gamma _ { t } ~ = ~ \gamma _ { 0 } ( t - 1 ) ^ { - a }$ for $t \geq 2$ , with $a ~ \in ~ ( 1 / 2 , 1 )$ , and let $u _ { t } ~ =$ $\mathbb { E } ( \| \Delta _ { t } \| ^ { 2 } \mathbb { I } \{ \tau > t \} )$ be as in Lemma 2. There are constants $C _ { 5 } , C _ { 6 } > 0$ such that, for all $t \geq 1$

$$
u _ { t } \le C _ { 5 } \exp ( - C _ { 6 } t ^ { 1 - a } ) + C _ { 5 } t ^ { - a } .
$$

Consequently $\| \Delta _ { t } \| \mathbb { I } \{ \tau > t \} \stackrel { p } {  } 0$ . Under the additional small-step condition of Lemma $\mathcal { Q } ( i i )$ , for every $b > 0$ ，

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { s u p } _ { \mathbb { P } } \mathbb { P } ( \| \theta _ { t } - \theta _ { 0 } \| > b ) \leq \epsilon .
$$

Proof. By (36),

$$
u _ { t } \le ( 1 - 2 c _ { 0 } \gamma _ { t } + C \gamma _ { t } ^ { 2 } ) u _ { t - 1 } + C \gamma _ { t } ^ { 2 } , \qquad u _ { 1 } \le \rho ^ { 2 } .\tag{48}
$$

After decreasing the upper bound on $\gamma _ { 0 }$ in Lemma 2, if necessary, we may assume that $C \gamma _ { t } \leq c _ { 0 }$ and $c _ { 0 } \gamma _ { t } \leq 1$ for every $t \geq 2$ . Hence

$$
0 \leq 1 - c _ { 0 } \gamma _ { t } \leq 1 , \qquad 1 - 2 c _ { 0 } \gamma _ { t } + C \gamma _ { t } ^ { 2 } \leq 1 - c _ { 0 } \gamma _ { t } ,
$$

and (48) implies

$$
u _ { t } \leq ( 1 - c _ { 0 } \gamma _ { t } ) u _ { t - 1 } + C \gamma _ { t } ^ { 2 } .\tag{49}
$$

Iterating this recursion and using $1 - x \leq e ^ { - x }$ gives

$$
u _ { t } \leq u _ { 1 } \exp \left( - c _ { 0 } \sum _ { i = 2 } ^ { t } \gamma _ { i } \right) + C \sum _ { k = 2 } ^ { t } \gamma _ { k } ^ { 2 } \exp \left( - c _ { 0 } \sum _ { i = k + 1 } ^ { t } \gamma _ { i } \right) .\tag{50}
$$

Integral bounds for the learning-rate sequence give constants $c , C > 0$ such that

$$
c \{ t ^ { 1 - a } - s ^ { 1 - a } \} \leq \sum _ { i = s + 1 } ^ { t } \gamma _ { i } \leq C \{ t ^ { 1 - a } - s ^ { 1 - a } \} , \quad \quad 1 \leq s < t .\tag{51}
$$

The first term in (50) is therefore bounded by $C \exp ( - c t ^ { 1 - a } )$

For the second term, let $k _ { \star } = \lfloor t / 2 \rfloor$ and first consider $k \ \leq \ k _ { \star }$ . Because $a > 1 / 2$ $\textstyle \sum _ { k \geq 2 } \gamma _ { k } ^ { 2 } < \infty$ , and (51) gives

$$
\begin{array} { r l r } {  { \sum _ { k = 2 } ^ { k _ { * } } \gamma _ { k } ^ { 2 } \exp ( - c _ { 0 } \sum _ { i = k + 1 } ^ { t } \gamma _ { i } ) } } \\ & { } & { \leq \exp ( - c _ { 0 } \sum _ { i = k _ { * } + 1 } ^ { t } \gamma _ { i } ) \sum _ { k = 2 } ^ { \infty } \gamma _ { k } ^ { 2 } \leq C \exp ( - c t ^ { 1 - a } ) . } \end{array}
$$

For $k \geq k _ { \star } + 1$ , monotonicity of $\gamma _ { k }$ gives $\gamma _ { k } \le C \gamma _ { t }$ , and hence

$$
\begin{array} { r l r } {  { \sum _ { k = k _ { * } + 1 } ^ { t } \gamma _ { k } ^ { 2 } \exp ( - c _ { 0 } \sum _ { i = k + 1 } ^ { t } \gamma _ { i } ) } } \\ & { } & { \leq C \gamma _ { t } \sum _ { k = k _ { * } + 1 } ^ { t } \gamma _ { k } \exp ( - c _ { 0 } \sum _ { i = k + 1 } ^ { t } \gamma _ { i } ) . } \end{array}
$$

Set $\begin{array} { r } { E _ { k } : = \exp \{ - c _ { 0 } \sum _ { i = k + 1 } ^ { t } \gamma _ { i } \} } \end{array}$ . Since $c _ { 0 } \gamma _ { k } \leq 1$

$$
E _ { k } - E _ { k - 1 } = E _ { k } ( 1 - e ^ { - c _ { 0 } \gamma _ { k } } ) \geq \frac { c _ { 0 } } { 2 } \gamma _ { k } E _ { k } .
$$

The preceding sum therefore telescopes and is uniformly bounded:

$$
\sum _ { k = k _ { \star } + 1 } ^ { t } \gamma _ { k } E _ { k } \leq \frac { 2 } { c _ { 0 } } \sum _ { k = k _ { \star } + 1 } ^ { t } ( E _ { k } - E _ { k - 1 } ) \leq \frac { 2 } { c _ { 0 } } .
$$

Thus the recent part is bounded by $C \gamma _ { t } \leq C t ^ { - a }$ . Combining the transient, early, and recent bounds in (50), and enlarging the constants to cover the finitely many small values of t, yields

$$
u _ { t } \le C _ { 5 } \exp ( - C _ { 6 } t ^ { 1 - a } ) + C _ { 5 } t ^ { - a } .
$$

For every $b > 0$ , Markov’s inequality gives

$$
\mathbb { P } ( \| \Delta _ { t } \| \mathbb { I } \{ \tau > t \} > b ) \le \frac { u _ { t } } { b ^ { 2 } } \longrightarrow 0 ,
$$

which proves the stopped consistency claim. Moreover,

$$
\mathbb { P } ( \| \Delta _ { t } \| > b ) \le \mathbb { P } ( \tau \le t ) + \frac { u _ { t } } { b ^ { 2 } } .
$$

Lemma 2(ii) consequently gives

$$
\operatorname* { l i m } _ { t \to \infty } \mathbb { P } ( \| \Delta _ { t } \| > b ) \leq \epsilon .
$$

This is the asserted ϵ-form of consistency; it does not imply unconditional $O _ { \mathbb { P } } ( 1 )$ for a fixed step-size scale. □

## References

Albert, J. H. and S. Chib (1993). Bayesian analysis of binary and polychotomous response data. Journal of the American Statistical Association 88(422), 669–679. Available at https://doi.org/10.1080/01621459.1993.10476321.

Bhat, C. R. (2011). The maximum approximate composite marginal likelihood estimation of multinomial probit-based unordered response choice models. Transportation Research Part B: Methodological 45 (7), 923–939. Available at https://doi.org/10.1016/ j.trb.2011.04.005.

Bolduc, D. (1999). A practical technique to estimate multinomial probit models in trans-

portation. Transportation Research Part B: Methodological 33 (1), 63–79. Available at https://doi.org/10.1016/S0191-2615(98)00028-9.

Boneva, T., M. Golin, K. Kaufmann, and C. Rauh (2026). Beliefs about maternal labour supply. The Economic Journal 136, 373–401. Available at https://doi.org/10.1093/ej/ ueaf067. Replication package: https://doi.org/10.5281/zenodo.16631469.

B¨orsch-Supan, A. and V. A. Hajivassiliou (1993). Smooth unbiased multivariate probability simulators for maximum likelihood estimation of limited dependent variable models. Journal of Econometrics 58 (3), 347–368. Available at https://doi.org/10. 1016/0304-4076(93)90049-B.

Chen, X., M. S. Kim, S. Lee, M. H. Seo, and M. Song (2025). SLIM: Stochastic learning and inference in overidentified models. arXiv:2510.20996. Available at https://arxiv. org/abs/2510.20996.

Chen, X., S. Lee, Y. Liao, M. H. Seo, Y. Shin, and M. Song (2025). SGMM: Stochastic approximation to generalized method of moments. Journal of Financial Econometrics 23 (1), nbad027. Available at https://doi.org/10.1093/jjfinec/nbad027.

Chen, X., E. Tamer, and Q. Yao (2026). Fast online inference on semiparametric models. arXiv:2603.08614. Available at https://arxiv.org/abs/2603.08614.

Croissant, Y. (2020). Estimation of random utility models in R: The mlogit package. Journal of Statistical Software 95 (11), 1–41. Available at https://doi.org/10.18637/ jss.v095.i11.

Ding, P., G. Imbens, Z. Qu, and Y. Ye (2024). Computationally eficient estimation of large probit models. arXiv:2407.09371. Available at https://arxiv.org/abs/2407.09371.

Frazier, D. T., R. Loaiza-Maya, and D. Nibbering (2026). Scalable likelihood-based inference for limited dependent variable models. arXiv:2608.13851. Available at https: //arxiv.org/abs/2608.13851.

Geweke, J., M. P. Keane, and D. E. Runkle (1994). Alternative computational approaches to inference in the multinomial probit model. Review of Economics and Statistics 76 (4), 609–632. Available at https://doi.org/10.2307/2109766.

Hajivassiliou, V., D. McFadden, and P. Ruud (1996). Simulation of multivariate normal rectangle probabilities and their derivatives: Theoretical and computational results. Journal of Econometrics 72 (1), 85–134. Available at https://doi.org/10.1016/ 0304-4076(94)01716-6.

Hajivassiliou, V. A. and D. L. McFadden (1998). The method of simulated scores for the estimation of LDV models. Econometrica 66(4), 863–896. Available at https: //doi.org/10.2307/2999576.

Hall, P. and C. C. Heyde (1980). Martingale limit theory and its application. New York: Academic Press. Available at https://doi.org/10.1016/C2013-0-10818-5.

Huch, E. K. and M. P. Keane (2026). Amortized inference for correlated discrete choice models via equivariant neural networks. arXiv:2603.24705. Available at https://arxiv. org/abs/2603.24705.

Imai, K. and D. A. van Dyk (2005). A bayesian analysis of the multinomial probit model using marginal data augmentation. Journal of Econometrics 124 (2), 311–334. Available at https://doi.org/10.1016/j.jeconom.2004.02.002.

Keane, M. P. (1994). A computationally practical simulation estimator for panel data. Econometrica 62 (1), 95–116. Available at https://doi.org/10.2307/2951477.

Lee, S., Y. Liao, M. H. Seo, and Y. Shin (2022). Fast and robust online inference with stochastic gradient descent via random scaling. Proceedings of the AAAI Conference on Artificial Intelligence 36 (7), 7381–7389. Available at https://doi.org/10.1609/aaai. v36i7.20701.

Lee, S., Y. Liao, M. H. Seo, and Y. Shin (2025). Fast inference for quantile regression with tens of millions of observations. Journal of Econometrics 249, 105673. Available at https://doi.org/10.1016/j.jeconom.2024.105673.

Loaiza-Maya, R. and D. Nibbering (2022). Scalable Bayesian estimation in the multinomial probit model. Journal of Business & Economic Statistics 40 (4), 1678–1690. Available at https://doi.org/10.1080/07350015.2021.1961788.

Loaiza-Maya, R. and D. Nibbering (2023). Fast variational Bayes methods for multinomial probit models. Journal of Business & Economic Statistics 41 (4), 1352–1363. Available at https://doi.org/10.1080/07350015.2022.2139267.

McCulloch, R. E. and P. E. Rossi (1994). An exact likelihood analysis of the multinomial probit model. Journal of Econometrics 64 (1–2), 207–240. Available at https://doi. org/10.1016/0304-4076(94)90064-7.

McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), Frontiers in Econometrics, pp. 105–142. New York: Academic Press. Available at https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf.

McFadden, D. (1989). A method of simulated moments for estimation of discrete response models without numerical integration. Econometrica 57(5), 995–1026. Available at https://doi.org/10.2307/1913621.

Polyak, B. T. and A. B. Juditsky (1992). Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization 30 (4), 838–855. Available at https://doi.org/10.1137/0330046.

Robbins, H. and S. Monro (1951). A stochastic approximation method. Annals of Mathematical Statistics 22 (3), 400–407. Available at https://doi.org/10.1214/aoms/ 1177729586.

Train, K. E. (2009). Discrete Choice Methods with Simulation (2nd ed.). New York: Cambridge University Press. Available at https://eml.berkeley.edu/books/choice2.html.

Zhu, Y. and J. Dong (2021). On constructing confidence region for model parameters in stochastic gradient descent via batch means. In 2021 Winter Simulation Conference (WSC), pp. 1–12. IEEE. Available at https://doi.org/10.1109/WSC52266.2021. 9715437.