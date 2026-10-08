# Conformal Prediction for Spatially Dependent Data via Sequential Whitening

Ayush Baran Sen

Indian Statistical Institute, Kolkata, India

and

Arkajyoti Saha Department of Statistics, University of California, Irvine

## Abstract

Split conformal prediction uses prediction errors on held-out (calibration) data to determine how wide the prediction intervals should be. It guarantees distribution-free finite-sample coverage when these errors and the error at the target site are exchangeable. This assumption may fail under spatial dependence and nonrandom sampling geometry. Existing spatial methods use fitting residuals to remove the predictable part of spatial variation from calibration and target errors. However, the spatial variation that only the calibration residuals can predict remains in both the target and calibration errors, reducing the eficiency and stability of the interval. We address this by additionally conditioning on the calibration residuals sequentially, which scales to large networks through nearest-neighbour approximations. Under a correct working covariance and an elliptical residual law, the resulting interval has exact finite-sample coverage under any spatial design, and under further conditions it is asymptotically oracle eficient. We also bound coverage loss under covariance misspecification and develop a diagnostic that identifies regions at risk of undercoverage. In simulated data, our method produces narrower and more stable intervals than global and localized state-of-the-art alternatives. In a national $\mathrm { P M } _ { 2 . 5 }$ application, it produces narrower intervals within the network and identifies regions at risk of coverage failure.

Keywords: spatial conformal prediction; exchangeability; Gaussian process; kriging; covariance misspecification

## 1 Introduction

Traditional uncertainty quantification approaches often require the assumed parametric model to be correctly specified to guarantee validity of the inference. Conformal prediction uses held-out observations (calibration data) to calibrate the fitted model’s errors and uses that to build the prediction intervals. This guarantees finite-sample marginal coverage as long as the errors on these held-out observations and the target location are exchangeable, and specifically does not require the mean model or the error distribution to be correctly specified (Angelopoulos and Bates, 2023).

In this article, we focus on inference for point-referenced spatial data. Such data are common across scientific fields, including environmental science, agriculture, ecology, and epidemiology. Prediction at unobserved locations is crucial here for policy and resource decisions. Some examples are assessing pollution exceedances at unmonitored sites (Berrocal et al., 2010), allocating inputs across agricultural fields (Oliver, 2010), and deciding where to drill for mineral deposits (Matheron, 1963). The data-generating process in these applications is complex and typically unknown, and hence requires intervals that remain reliable even when the spatial model is imperfect.

Conformal inference may appear to ofer a solution, but it is natural to wonder what happens to its exchangeability requirement under spatial dependence. The answer depends on how the locations are sampled. Under random location sampling, the scores can be dependent yet exchangeable. However, ordinary split conformal is ineficient here. It does not use the residuals in the observed data to predict the spatial variation in the target error, resulting in a wider interval that has the same width at every target location. The errors in the held-out data are spatially dependent, which makes the calibration quantile, and the resulting interval, unstable.

Mao et al. (2024); Moon and Kim (2026) ofer principled ways to address these problems. For split conformal, they propose to predict the spatial variation in the calibration and target residuals from the fitting residuals. Their interval widths also adapt to the prediction uncertainty since they standardize the errors using their respective predictive standard deviations. However, conditioning only on the fitting set still leaves the spatial variation captured solely by the calibration residuals in both the calibration and target errors, ofering only a partial solution to the problems with naive split conformal.

When locations are not sampled at random, the spatial variation left in the calibration residuals can make the target and calibration scores incomparable. This violates the exchangeability assumption required for validity of conformal inference. Mao et al. (2024); Guan (2023); Jiang and Xie (2026) propose to bypass this problem by giving greater weight to calibration points near the target. This makes the threshold more representative of the target neighbourhood, even in the presence of correlation in calibration residuals. Localization relies on fewer, often strongly correlated, observations, which can make the resulting quantile unstable. Moreover, strong localization can also produce an unbounded interval if there is not enough data near the target. Additionally, this mechanism does not account for the correlation captured by the calibration residuals and faces the same issues as the global method.

In this article, we develop a spatial conformal procedure that uses the predictive information in both the fitting and calibration residuals. Our main contributions are:

1. Methodology. We introduce a way to use the predictive power of both the fitting and calibration residuals via sequential conditioning. The resulting interval adapts to prediction uncertainty at the target and is narrower and more stable than intervals from existing spatial conformal methods. When the working covariance adequately represents the joint residual structure, the transformation also restores comparability between the calibration and target scores.

2. Validity and optimality. Under a correct working covariance matrix and an elliptical residual distribution, we establish exact finite-sample marginal coverage under any spatial design. Under additional assumptions, the interval is asymptotically as short as the shortest conditionally valid interval.

3. Robustness and diagnostics. We bound coverage loss under covariance misspecification and characterize target-specific coverage by comparing the variation remaining in the target error with that in the calibration errors. This characterization yields a diagnostic that can identify regions at risk of coverage failure without knowing the true covariance.

The rest of the paper is organized as follows. Section 2 reviews conformal prediction for independent data and formulates the spatial prediction problem. Section 3 introduces our method, and Section 4 develops its theoretical guarantees. Sections 5 and 6 present the simulation study and an application to fine-particulate-matter monitoring, respectively. Section 7 concludes with a discussion and directions for future research. Proofs and additional results appear in the Supplementary Material.

## 2 Problem definition

## 2.1 Conformal prediction for independent data

Let the data be denoted as $( \pmb { x } _ { i } , y _ { i } ) \in \mathbb { R } ^ { p } \times \mathbb { R } , i = 1 , \ldots , n$ , which are independent and identically distributed observations from some distribution P. Assume that $f ( \pmb { x } ) = \mathbb { E } ( \pmb { y } \mid \pmb { x } )$ exists. For any given $\alpha \in ( 0 , 1 )$ and a new data point $( { \pmb x } _ { 0 } , y _ { 0 } ) \sim P$ , we want to compute a prediction interval $\widehat { C } _ { \alpha } ( \pmb { x } _ { 0 } )$ which satisfies $\mathbb { P } \{ y _ { 0 } \in { \widehat { C } } _ { \alpha } ( \mathbf { x } _ { 0 } ) \} \geq 1 - \alpha$ . The probability is over all the data and the new pair. Ideally, the interval should be valid in finite samples for every $P$ and every fitted ${ \widehat { f } } ,$ and be as narrow as possible. Split conformal prediction (Papadopoulos et al., 2002; Vovk et al., 2005; Le et al., 2018) meets the first validity requirement. The procedure is as follows:

(a) Partition $\{ 1 , \ldots , n \}$ independently of the data into a fitting set $\mathcal { F }$ and a calibration set C of size $m ,$ , and fit any $\widehat { f }$ using only $\mathcal { F }$

(b) For each $i \in \mathcal { C }$ , compute the residual $r _ { i } = y _ { i } - { \widehat { f } } ( \mathbf { x } _ { i } )$ , which measures how poorly $\widehat { f }$ predicts the ith calibration observation.

(c) At $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ , compute the conformal interval as

$$
\widehat { C } _ { \alpha } ( \pmb { x } _ { 0 } ) = \widehat { f } ( \pmb { x } _ { 0 } ) \pm \widehat { q } _ { 1 - \alpha } , \qquad \widehat { q } _ { 1 - \alpha } = | r | _ { ( \lceil ( 1 - \alpha ) ( m + 1 ) \rceil ) } ,\tag{1}
$$

where $| r | _ { ( 1 ) } \leq \cdots \leq | r | _ { ( m ) }$ are the ordered calibration scores and $| r | _ { ( m + 1 ) } : = + \infty$ . The interval is therefore unbounded when $\alpha < 1 / ( m + 1 )$

The following lemma (Papadopoulos et al., 2002; Lei et al., 2018) ensures finite-sample coverage, as long as the calibration and target scores are exchangeable, that is, their joint distribution is invariant to any permutation.

Lemma 1. Let $S _ { i } = \left| r _ { i } \right| f o r ~ i \in \mathcal { C }$ and $S _ { 0 } = | y _ { 0 } - { \widehat f } ( { \pmb x } _ { 0 } ) |$ . If these $m + 1$ scores are exchangeable, then the interval in (1) satisfies $\mathbb { P } \{ y _ { 0 } \in { \widehat { C } } _ { \alpha } ( \mathbf { x } _ { 0 } ) \} \geq 1 - \alpha$ . If the scores are also almost surely distinct, then

$$
1 - \alpha \leq \mathbb { P } \{ y _ { 0 } \in \widehat { C } _ { \alpha } ( { \pmb x } _ { 0 } ) \} \leq 1 - \alpha + \frac { 1 } { m + 1 } .\tag{2}
$$

When scores are distinct, exchangeability implies that the rank of the target score is uniformly distributed among the $m + 1$ possible ranks, giving the lower bound in (2). The upper bound comes from rounding the conformal quantile. Exchangeability is trivially satisfied under i.i.d. data. We next examine how this argument changes under spatial dependence.

## 2.2 What changes for spatial data

Suppose we observe $\mathcal { D } _ { n } = \{ ( \boldsymbol { x } _ { i } , \boldsymbol { s } _ { i } , y _ { i } ) : i = 1 , \ldots , n \}$ , where $\pmb { s } _ { i } \in \mathbb { R } ^ { 2 }$ is the spatial location, typically geographical coordinates. We want to predict at a new location $( \boldsymbol { \boldsymbol { x } } _ { 0 } , \boldsymbol { \boldsymbol { s } } _ { 0 } )$ . We assume the model

$$
y _ { i } = f ( { \pmb x } _ { i } ) + \eta ( { \pmb s } _ { i } ) + \varepsilon _ { i } , \qquad i = 0 , 1 , \ldots , n ,\tag{3}
$$

where $f ( { \pmb x } )$ is the covariate efect and η is the spatial structure the covariates do not capture. The process η has mean zero and covariance $C _ { \theta }$ , with $C _ { \theta } ( s , s ) = \sigma ^ { 2 }$ . The errors $\varepsilon _ { i }$ are i.i.d. with mean zero and variance $\tau ^ { 2 } > 0$ , and independent of η. The covariates may vary with location, but where the data are sampled is unrelated to η and the errors. We want a prediction interval $\widehat { C } _ { \alpha } ( \pmb { x } _ { 0 } , s _ { 0 } )$ which satisfies $\mathbb { P } \{ y _ { 0 } \in { \widehat { C } } _ { \alpha } ( \mathbf { x } _ { 0 } , \mathbf { s } _ { 0 } ) \} \geq 1 - \alpha$ . Throughout, we treat the locations, the covariates, and the fitted mean and covariance as fixed, and take probabilities over the responses alone. Under random sampling, they are taken over the sampled locations as well.

The residuals are spatially dependent. However, naive conformal inference may still be valid here, since Lemma 1 only requires exchangeability. Whether exchangeability holds depends on the sampling design. We consider the two cases in turn, starting with the one where it holds.

## 2.2.1 When random sampling preserves exchangeability

Suppose the labelled and target data are sampled independently from the same design, and the fitting– calibration split is independent of the data. Then calibration and target scores are exchangeable (Mao et al., 2024, Lemma 1), so Lemma 1 applies and naive split conformal is valid. However, it ignores the spatial correlation, and so is ineficient in three ways:

(a) The width is unstable. In spatially correlated data, residuals from nearby points are positively correlated. This makes the efective sample size of the calibration set smaller than m. As a result, the interval width varies more than it would with m independent scores.

(b) Predictable target variation remains. Observed residuals contain information about the spatial component of the target error. Naive split conformal prediction does not make use of the predictive power of these residuals. This leaves more unexplained variation for the interval to cover, resulting in a wider interval.

(c) Width does not adapt to location. Naive conformal intervals have the same width at every target. However, in spatial statistics, isolated locations are harder to predict than those surrounded by observations. Ideally, prediction uncertainty should be reflected in the interval width, which is not the case here.

The concerns are partially addressed by the global spatial conformal approach of Mao et al. (2024) (GSCP). Let $r _ { \mathcal { F } }$ be the vector of fitting residuals and $\Sigma _ { \mathcal { F } } ^ { \mathrm { w } }$ its working covariance. For a location s, write $\pmb { k } _ { \mathcal { F } } ( \pmb { s } )$ for its working covariance with the fitting residuals and $\Sigma ^ { \mathrm { w } } ( s , s )$ for its working variance. The working kriging predictor and prediction variance are

$$
\begin{array} { r } { \hat { \mu } _ { \mathcal { F } } ( \pmb { s } ) = \pmb { k } _ { \mathcal { F } } ( \pmb { s } ) ^ { \top } ( \pmb { \Sigma } _ { \mathcal { F } } ^ { \mathrm { w } } ) ^ { - 1 } \pmb { r } _ { \mathcal { F } } , \qquad \hat { \nu } _ { \mathcal { F } } ( \pmb { s } ) = \pmb { \Sigma } ^ { \mathrm { w } } ( \pmb { s } , \pmb { s } ) - \pmb { k } _ { \mathcal { F } } ( \pmb { s } ) ^ { \top } ( \pmb { \Sigma } _ { \mathcal { F } } ^ { \mathrm { w } } ) ^ { - 1 } \pmb { k } _ { \mathcal { F } } ( \pmb { s } ) . } \end{array}\tag{4}
$$

GSCP uses the calibration scores $| r _ { i } - { \widehat \mu } _ { \mathcal { F } } ( \pmb { s } _ { i } ) | / { \widehat \nu } _ { \mathcal { F } } ( \pmb { s } _ { i } ) ^ { 1 / 2 }$ . Let $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { G } }$ denote their $\left. \lceil ( 1 - \alpha ) ( m + 1 ) \right\rceil$ ⌉-th order statistic, using the same +∞ convention as in (1). The resulting interval is

$$
\widehat { C } _ { \alpha } ^ { \mathrm { G } } ( \pmb { x } _ { 0 } , \pmb { s } _ { 0 } ) = \widehat { f } ( \pmb { x } _ { 0 } ) + \widehat { \mu } _ { \mathcal { F } } ( \pmb { s } _ { 0 } ) \pm \widehat { q } _ { 1 - \alpha } ^ { \mathrm { G } } \widehat { \nu } _ { \mathcal { F } } ( \pmb { s } _ { 0 } ) ^ { 1 / 2 } .\tag{5}
$$

GSCP only accounts for the predictive information in the fitting residuals. It does not utilize the spatial information the calibration residuals carry on their own. Hence, the resulting intervals are wider and less stable than they need to be. Though the width of the GSCP intervals adapts to the prediction uncertainty, it applies the same rule to every calibration point and to the target. Therefore, it has the same coverage guarantee as naive split conformal, i.e., it is valid when the sampling preserves exchangeability.

## 2.2.2 When geometry breaks exchangeability

There are two separate ways in which exchangeability can fail. First, the target score may have a diferent marginal distribution from the calibration scores. This might happen under spatial extrapolation or if the locations of the target and calibration are in diferent regions with distinct residual behaviour (Sampson and Guttorp, 1992). Second, even with identical marginal distributions, the calibration scores may be more strongly or weakly related to one another than to the target score.

Mao et al. (2024); Guan (2023); Jiang and Xie (2026) propose to address these by giving greater weight to calibration observations near the target. This makes the calibration scores more representative of the target score. In this article, we use a split form of the localized GSCP procedure proposed by Mao et al. (2024), which we call LSCP. A bandwidth parameter h controls the degree of localization. Under strong localization there may be too few calibration observations with a relevant weight, which leads to an unbounded interval. For larger h, the weights become more uniform and LSCP approaches GSCP.

GSCP and LSCP both ignore the predictive information in the calibration scores, hence their performance depends on the fitting–calibration split. We see this in the following example. We consider 100 labelled observations in two well-separated clusters and place the target in one of them. The residuals follow a stationary Gaussian process with strong dependence within clusters and negligible dependence between them. We compare three splits: the calibration observations lie entirely in the target cluster, are divided equally between the clusters, or lie entirely in the distant cluster. We call these the nearby, mixed, and distant calibration designs. We compute the coverage, mean width, and width stability for naive split conformal, GSCP and LSCP under known mean and covariance in Figure 1. We measure stability using the coeficient-of-variation ratio (CVR). This measures how much larger the coeficient of variation of a method’s interval width is than that of naive conformal on an i.i.d. version of the data. If the CVR is one, it means that the dependence does not add any extra variability. If it is higher, it means that it does. The figure also shows the performance of the proposed whitened method introduced in Section 3.

Naive split conformal does not use fitting or calibration residuals for prediction, hence its width is not afected by the split and is comparable across the three splits. Its CVR remains near two because it does not account for the correlation among the calibration data. Under nearby calibration, fitting and target data lie in diferent clusters and are nearly independent. Hence conditioning on fitting data does not give GSCP any additional advantage and it performs similarly to naive conformal. For mixed calibration, since fitting data share a cluster with both the calibration and target data, conditioning on them can explain some of the predictable spatial variation in both the target and the calibration data, leading to a much narrower interval with a CVR that is closer to one. In distant calibration the fitting data share a cluster with the target, hence GSCP can explain a lot more of the predictable variation in the target residual, making the interval the narrowest. However, the fitting data are nearly independent of the calibration data, and cannot explain the correlation in the calibration residuals, which reintroduces instability in the width and the CVR jumps to near two.

As far as coverage is concerned, naive conformal and GSCP attain near-nominal coverage under both nearby and mixed calibration, where all or part of the calibration data lie in the same cluster as the target. Under distant calibration, the calibration data are in a diferent cluster from the target. The calibration scores are then strongly related to one another but nearly independent of the target score. This violates exchangeability, and both naive conformal and GSCP undercover.

To see the problem, Figure 2 plots the rank of the target score among the calibration scores under distant calibration and compares it with nearby calibration. For GSCP the ranks are nearly uniform under nearby calibration but bimodal under distant calibration. The bimodal case places excess probability in the noncoverage region, so GSCP undercovers.

![](images/ff8c85f26e4d971221049c35f34ab7ff9968c321236f0520e7de48a997a3e185.jpg)

![](images/37ba95f7ab9751e9fce89d46a81c8eff38a5952759dd0484c241b8a2c634c56a.jpg)

(c) Distant calibration  
![](images/6f7451b65699c11bf843990e65d30f0956f4462b34e2714771fd205aa25b8227.jpg)

<table><tr><td>Method</td><td>Nearby</td><td>Mixed</td><td>Distant</td></tr><tr><td>Naive</td><td>0.911; 3.32 (2.17)</td><td>0.908; 3.41 (1.89)</td><td>0.858; 3.30 (2.16)</td></tr><tr><td>GSCP</td><td>0.910; 3.30 (2.16)</td><td>0.904; 2.10 (1.10)</td><td>0.855; 1.74 (2.14)</td></tr><tr><td>LSCP, h = 0.05</td><td>1.000;∞</td><td>1.000; ∞</td><td>1.000; ∞</td></tr><tr><td>LSCP, h = 0.20</td><td>0.926; 3.44 (2.13)</td><td>0.926; 2.32 (1.50)</td><td>1.000; ∞</td></tr><tr><td>LSCP, h = 0.40</td><td>0.926; 3.43 (2.13)</td><td>0.921; 2.22 (1.17)</td><td>0.886; 1.90 (2.07)</td></tr><tr><td>LSCP, h = 1</td><td>0.926; 3.43 (2.13)</td><td>0.919; 2.20 (1.12)</td><td>0.873; 1.83 (2.09)</td></tr><tr><td>LSCP, h = 2</td><td>0.926; 3.43 (2.13)</td><td>0.919; 2.20 (1.12)</td><td>0.872; 1.82 (2.09)</td></tr><tr><td>Proposed (Whitened)</td><td>0.904; 1.87 (1.00)</td><td>0.902; 1.87 (1.01)</td><td>0.900; 1.87 (1.00)</td></tr></table>

Figure 1: The top panel shows the three fitting–calibration splits on fixed data in the two-cluster design. The table below reports sensitivity of naive split conformal, GSCP, LSCP (under diferent bandwidths h), and the proposed approach to the fitting–calibration split. Each entry reports coverage; mean width (CVR) over 20,000 datasets for nominal 90% intervals. Under strong localization, LSCP can produce infinite intervals.

Mao et al. (2024) propose to bypass this problem via localization in LSCP. However, the performance of LSCP depends on the bandwidth that controls the degree of localization. In Figure 1, very strong localization (small h) produces an unbounded interval. An intermediate h increases coverage, but only with a wider interval, and overcovers where GSCP was already calibrated. As h increases further, the interval narrows toward GSCP and still undercovers under distant calibration. The h that balances coverage and width is unknown in practice. The example shows that conditioning on the fitting residuals alone leads to an interval that is wider than necessary and less stable, and can cause undercoverage or unbounded intervals. In the next section we propose our method, which addresses all of these together.

## 3 Method

The previous section showed that the crux of the issue is that the predictive information in the calibration residuals goes unused. However, predicting a calibration residual from the entire calibration set makes it predict itself. Its predictive variance is then zero and its score degenerate, so it can no longer be compared with the target score.

We propose to bypass this by conditioning on the calibration residuals sequentially. Fixing an order that does not depend on the responses, we predict each calibration residual from the fitting residuals and the calibration residuals before it. We subtract this prediction and divide by its predictive standard deviation. We place the target last and predict it from both the fitting and calibration residuals. We formalize the method below.

![](images/6f029920ec45ac98abd7a7b73edf5a454f1a73e065f5fb014779a312a40d233c.jpg)  
Figure 2: Distribution of rank of target score in the two-cluster example. The top row uses nearby calibration, and the bottom row uses distant calibration. The left column shows the calibration and target locations; the middle and right columns show the distribution of the rank of the target score among calibration scores under GSCP and the proposed whitened method. Dashed lines mark the expected bin height under a uniform rank distribution, and shaded regions contain ranks producing noncoverage.

## 3.1 Sequential prediction and standardization

Let F and C be the fitting and calibration sets of sizes $n { \mathcal { F } }$ and $m ,$ respectively. We fit the mean model $\widehat { f }$ on ${ \mathcal F } ,$ and compute out-of-bag or cross-fitted residuals, $\boldsymbol { r } _ { \mathcal { F } }$ . For $i \in \mathcal { C }$ , let $r _ { i } = y _ { i } - { \widehat { f } } ( \mathbf { x } _ { i } )$ and $r _ { 0 } = y _ { 0 } - { \hat { f } } ( { \pmb x } _ { 0 } )$ be the calibration and target residuals, respectively. Here $i = 1 , 2 , \dots , m$ indexes any response-independent order, and we write $\boldsymbol { r } \boldsymbol { c } = ( r _ { 1 } , \ldots , r _ { m } ) ^ { \top }$ . Next, we stack the residuals as follows:

$$
\begin{array} { r } { r _ { + } = \left( r _ { \mathcal { F } } ^ { \top } , r _ { \mathcal { C } } ^ { \top } , r _ { m + 1 } \right) ^ { \top } \in \mathbb { R } ^ { N } , \qquad r _ { m + 1 } = r _ { 0 } , \quad N = n _ { \mathcal { F } } + m + 1 . } \end{array}\tag{6}
$$

Let $\pmb { \Sigma } _ { + } ^ { \mathrm { w } }$ be a positive-definite working covariance matrix for $r _ { + }$ . For A, $B \subseteq \{ 1 , \dots , N \}$ , write $r _ { + , A }$ and $\Sigma _ { A , B } ^ { \mathrm { w } } = ( \Sigma _ { + } ^ { \mathrm { w } } ) _ { A , B }$ for the corresponding subvector and submatrix. For each $i \in \{ 1 , 2 , \ldots , m + 1 \}$ 2 let $k _ { i } = n _ { \mathscr { F } } + i$ and $I _ { i } = \{ 1 , \ldots , k _ { i } - 1 \}$ . The standardized score of $r _ { i }$ is

$$
e _ { i } = \frac { r _ { i } - \widehat { \mu } _ { i } } { \widehat { \nu } _ { i } ^ { 1 / 2 } } ,\tag{7}
$$

where, for all $i \in \{ 1 , 2 , \ldots , m + 1 \}$ 了,

$$
\begin{array} { r } { \widehat { \mu } _ { i } = \Sigma _ { k _ { i } , I _ { i } } ^ { \mathrm { w } } \big ( \Sigma _ { I _ { i } , I _ { i } } ^ { \mathrm { w } } \big ) ^ { - 1 } r _ { + , I _ { i } } , \qquad \widehat { \nu } _ { i } = \Sigma _ { k _ { i } , k _ { i } } ^ { \mathrm { w } } - \Sigma _ { k _ { i } , I _ { i } } ^ { \mathrm { w } } \big ( \Sigma _ { I _ { i } , I _ { i } } ^ { \mathrm { w } } \big ) ^ { - 1 } \Sigma _ { I _ { i } , k _ { i } } ^ { \mathrm { w } } . } \end{array}\tag{8}
$$

## 3.2 Equivalence to triangular whitening

Write the Cholesky factorization of the working covariance as $\pmb { \Sigma } _ { + } ^ { \mathrm { w } } = \pmb { L } \pmb { L } ^ { \top }$ , where $\pmb { L } = ( \ell _ { i j } ) _ { i , j = 1 } ^ { N }$ is lower triangular with positive diagonal entries. Next, we connect the sequential construction to a single matrix transformation.

Lemma 2 (Sequential form of triangular whitening). For every $i = 1 , \ldots , m + 1$

$$
\begin{array} { r } { ( L ^ { - 1 } r _ { + } ) _ { n _ { \mathcal { F } } + i } = e _ { i } . } \end{array}\tag{9}
$$

The proof is given in Supplementary Material Section S1. This result shows that the sequential prediction and standardization are equivalent to the triangular whitening of the ordered residual vector.

## 3.3 The whitened conformal interval

We now construct the whitened conformal interval.

Theorem 1 (Finite-sample validity of the whitened interval). Fix $\alpha \in ( 0 , 1 )$ . Suppose that $r _ { + }$ has a centred, absolutely continuous elliptical distribution with scatter matrix $\Sigma _ { + } ^ { \mathrm { w } }$ . Let $j _ { \alpha } = \lceil ( 1 - \alpha ) ( m + 1 ) \rceil$ and let $\widehat { q } _ { 1 - c } ^ { \mathrm { W } }$ be the $j _ { \alpha } \mathrm { - } t h$ order statistic of $\{ | e _ { 1 } | , \ldots , | e _ { m } | , + \infty \}$ , and write $\widehat \mu _ { \mathcal I \cup \mathcal { C } } ( \pmb { \mathscr { s } } _ { 0 } ) = \widehat \mu _ { m + 1 }$ and $\widehat { \nu } _ { { \mathcal F } \cup { \mathcal C } } ( s _ { 0 } ) = \widehat { \nu } _ { m + 1 }$ from (8) at $i = m + 1$ . Then the whitened conformal interval

$$
\widehat { C } _ { \alpha } ^ { \mathrm { W } } ( \pmb { x } _ { 0 } , \pmb { s } _ { 0 } ) = \widehat { f } ( \pmb { x } _ { 0 } ) + \widehat { \mu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { s } _ { 0 } ) \pm \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { s } _ { 0 } ) ^ { 1 / 2 }\tag{10}
$$

satisfies $\begin{array} { r } { \mathbb { P } \{ y _ { 0 } \in \widehat { C } _ { \alpha } ^ { \mathrm { W } } ( { \pmb x } _ { 0 } , { \pmb s } _ { 0 } ) \} = \lceil ( 1 - \alpha ) ( m + 1 ) \rceil / ( m + 1 ) \geq 1 - \alpha . } \end{array}$

Refer to Supplementary Material Section S2 for the proof. The result also holds when $\widehat { f }$ and $\Sigma _ { + } ^ { \mathrm { w } }$ are estimated from the fitting data, provided that, given the fitting data, the calibration and target residuals have an absolutely continuous elliptical distribution with location equal to their working kriging predictors from $\boldsymbol { r } _ { \mathcal { F } }$ and scatter matrix equal to the corresponding working conditional covariance.

## 3.4 What whitening changes

The two-cluster example in Section 2.2.2 showed that the nonoptimality of GSCP and LSCP comes from conditioning on the fitting residuals alone. Whitening addresses this by conditioning on both the fitting and calibration residuals.

Restoring rank symmetry. Under a correct working covariance and an elliptical residual law, accounting for the correlation within calibration residuals makes calibration and target scores exchangeable. Figure 2 shows this. The GSCP ranks are bimodal under distant calibration, whereas they become nearly uniform under our proposed approach.

Reducing interval width. Since the whitened interval uses more information than GSCP to predict the target errors, intuitively our interval should be narrower. Proposition 1 makes this precise: the ratio of the whitened and GSCP interval widths is asymptotically equal to the ratio of their predictive standard deviations at the target. Since ${ \mathcal { F } } \subseteq { \mathcal { F } } \cup { \mathcal { C } }$ , conditioning on the calibration residuals cannot increase the prediction variance, therefore $\widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { \mathscr { s } } _ { 0 } ) \leq \widehat { \nu } _ { \mathcal { F } } ( \pmb { \mathscr { s } } _ { 0 } )$ . When calibration residuals contain information in addition to fitting data, this inequality is strict. Figure 1 shows this in the two-cluster case.

Algorithm 1 Whitened spatial conformal prediction   
Require: Labelled data $\mathcal { D } _ { n } = \{ ( \boldsymbol { x } _ { i } , \boldsymbol { s } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , target $( \boldsymbol { \boldsymbol { x } } _ { 0 } , \boldsymbol { \boldsymbol { s } } _ { 0 } )$ , fitting and calibration sets $( { \mathcal { F } } , { \mathcal { C } } )$   
miscoverage level α, mean estimator $\mathcal { A } ,$ working covariance model $C _ { \theta } ^ { \mathrm { w } }$ , covariance-parameter   
estimator $\mathcal { E } ,$ calibration-ordering rule ${ \mathcal { O } } ,$ and routine B that returns the inverse-Cholesky factor   
of the fitted working covariance at an ordered set of locations   
1: Fit ${ \widehat { f } } \gets A ( { \mathcal { F } } )$   
2: Compute fitting residuals $\boldsymbol { r } _ { \mathcal { F } }$ , using out-of-bag or cross-fitted predictions for flexible learners,   
and compute $r _ { i } = y _ { i } - { \widehat { f } } ( \mathbf { x } _ { i } )$ for $i \in \mathcal { C }$   
3: Estimate $\widehat { \theta }  \mathcal { E } ( \{ ( s _ { i } , ( r _ { \mathcal { F } } ) _ { i } ) : i \in \mathcal { F } \} ; C _ { \theta } ^ { \mathrm { w } } )$   
4: Set $\pi _ { \mathscr { C } } \gets \mathscr { O } ( \{ ( \pmb { x } _ { i } , \pmb { s } _ { i } ) : i \in \mathscr { C } \} )$ and form $\pmb { r } _ { \pmb { \mathscr { C } } } = \left( r _ { \pi _ { \pmb { \mathscr { C } } } ( 1 ) } , \ldots , r _ { \pi _ { \pmb { \mathscr { C } } } ( m ) } \right) ^ { \top }$   
5: Form the augmented residual vector $\pmb { r } _ { + } = ( \pmb { r } _ { \mathcal { F } } ^ { \top } , \pmb { r } _ { \mathcal { C } } ^ { \top } , \pmb { r } _ { m + 1 } ) ^ { \top }$ , where the final entry $r _ { m + 1 } =$   
$y _ { 0 } - { \widehat { f } } ( { \pmb x } _ { 0 } )$ is unknown   
6: Construct the inverse-Cholesky factor of the fitted working covariance of $r _ { + }$ as ${ \pmb L } ^ { - 1 } \gets$   
$B \Big ( C _ { \widehat { \theta } } ^ { \mathrm { w } } ; s _ { \mathcal { F } } , s _ { \pi _ { C } } , s _ { 0 } \Big )$   
7: Compute $( L ^ { - 1 } ) _ { 1 : n , 1 : n } { \pmb { r } } _ { + , 1 : n }$ and retain its final m coordinates as $e _ { 1 } , \ldots , e _ { m }$   
8: Set $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } }$ to the $\lceil ( 1 - \alpha ) ( m + 1 ) \rceil$ -th order statistic of $\{ | e _ { 1 } | , \ldots , | e _ { m } | , + \infty \}$   
9: Let $( \pmb { a } ^ { \top } , d )$ be the final row of ${ \cal L } ^ { - 1 }$ and compute   
$\widehat { \mu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { s } _ { 0 } ) = - \frac { \pmb { a } ^ { \top } \pmb { r } _ { + , 1 : n } } { d } , \qquad \widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { s } _ { 0 } ) = d ^ { - 2 }$   
10: return The whitened conformal interval   
$\widehat { C } _ { \alpha } ^ { \mathrm { W } } ( \pmb { x } _ { 0 } , \pmb { s } _ { 0 } ) = \widehat { f } ( \pmb { x } _ { 0 } ) + \widehat { \mu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { s } _ { 0 } ) \pm \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { s } _ { 0 } ) ^ { 1 / 2 } .$

Proposition 1 (Asymptotic width ratio). Let $W ^ { \mathrm { G } } ( \pmb { s } _ { 0 } )$ and $W ^ { \mathrm { W } } ( \pmb { s } _ { 0 } )$ be the widths of the GSCP and whitened intervals under the same working covariance. Suppose that, as $m \to \infty$ , the empirical distributions of the absolute calibration scores of the two methods converge in probability to the same limit $G _ { 0 }$ . Suppose also that $G _ { 0 }$ is continuous and strictly increasing at its $( 1 - \alpha )$ quantile $\zeta _ { 1 - \alpha } > 0$ Then

$$
\frac { W ^ { \mathrm { W } } ( s _ { 0 } ) } { W ^ { \mathrm { G } } ( s _ { 0 } ) } = \left\{ \frac { \widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( s _ { 0 } ) } { \widehat { \nu } _ { \mathcal { F } } ( s _ { 0 } ) } \right\} ^ { 1 / 2 } \{ 1 + o _ { \mathbb { P } } ( 1 ) \} .\tag{11}
$$

The proof is given in Supplementary Material Section S3.

Stabilizing the calibration threshold. Under a correct Gaussian working covariance, the sequential conditioning of whitening removes correlation within calibration residuals beyond what is removed by GSCP and LSCP. As a result, each score contributes separate information to the threshold. Empirically, this makes our method’s stability comparable to the independent-data case. Figure 1 demonstrates this in the two-cluster example.

## 3.5 Implementation

Algorithm 1 summarizes the complete construction. The construction allows flexibility in mean estimation, construction of the inverse-Cholesky factor of the working covariance, and calibration ordering. We describe the choices used in this article below.

Mean estimation. This approach works with any mean estimator. In-sample fitting residuals from flexible learners are artificially small. This distorts covariance estimation and whitening. Therefore, we build $r _ { \mathcal { F } }$ from out-of-bag or cross-fitted predictions. Here we use a random forest with out-of-bag residuals.

Working covariance and factorization. This approach works with any positive-definite covariance model and parameter estimator. We use an exponential covariance with a nugget throughout and estimate its parameters with the Nearest Neighbor Gaussian Process (NNGP) approximation (Datta et al., 2016), implemented in BRISC (Saha and Datta, 2018). We then use these to compute the dense working covariance matrix and its Cholesky factor. This costs cubic time and quadratic storage. For moderate to large samples, this can become computationally prohibitive. Hence we recommend using NNGP covariance there. In NNGP, each residual is conditioned on at most K preceding neighbours, hence the inverse-Cholesky factor has at most K + 1 nonzero entries per row and costs linear time and storage. The Supplementary Material Section S6.2.3 shows that the two implementations give practically indistinguishable results.

Ordering. The method works with any order that does not depend on the responses. We use the order supplied by the split, and the Supplementary Material Section S6.2.3 shows that the results are insensitive to this choice.

## 4 Theoretical guarantees

In this section, we demonstrate the asymptotic equality of the widths of the whitened interval and the shortest conditionally valid interval under oracle conditions. Next, we consider what happens under misspecification. First, we derive a global bound on the resulting coverage error and then develop a local diagnostic that detects regions at risk of undercoverage without knowing the true data-generating process. Throughout this section, we treat the locations, the fitted mean, and the working covariance as fixed.

## 4.1 Eficiency

Theorem 2 (Asymptotic oracle eficiency). Let $\mathcal { D } = \mathcal { F } \cup \mathcal { C }$ . Suppose the working covariance is correct and the coordinates of ${ \bf \dot { e } } _ { + } = { \bf L } ^ { - 1 } { \bf r } _ { + }$ are independent with a common distribution having mean zero, unit variance, and a continuous density that is symmetric about zero and strictly decreasing on $( 0 , \infty )$ . Suppose also that the density of |e| is positive at its $( 1 - \alpha )$ quantile. Let $W ^ { \mathrm { o } } ( \pmb { s } _ { 0 } )$ denote the width of the shortest interval, measurable with respect to $r _ { \mathcal { D } _ { . } }$ , having conditional coverage at least 1 − α given r<sub>D</sub> when the true joint law is known. $I f W ^ { \mathrm { W } } ( s _ { 0 } )$ denotes the width of the whitened interval in (10), then, as $m \to \infty$

$$
\frac { W ^ { \mathrm { W } } ( \pmb { s } _ { 0 } ) } { W ^ { \mathrm { o } } \big ( \pmb { s } _ { 0 } \big ) } = 1 + O _ { \mathbb { P } } ( m ^ { - 1 / 2 } ) .\tag{12}
$$

The oracle and whitened intervals share the same predictive standard deviation but have diferent multipliers. The whitened interval uses the empirical quantile of the whitened calibration scores, whereas the oracle uses the population quantile of $| e |$ . This empirical calibration quantile approaches the oracle quantile at the rate in (12) under the assumptions of the theorem. Supplementary Material Section S3 contains the proof.

## 4.2 Coverage under covariance misspecification

In reality, the true covariance and the covariance utilised for whitening are typically diferent. First, we use the diference between an exchangeable reference law and the actual law of the transformed calibration and target errors to bound the resulting coverage error.

Let $\pmb { e } = ( e _ { 1 } , \ldots , e _ { m } , e _ { m + 1 } ) ^ { \top }$ be the vector of transformed calibration and target errors. Let $Q _ { e }$ represent a reference law under which the absolute coordinates of e are exchangeable and almost surely distinct, and let $P _ { e }$ denote the actual law of $e .$

Theorem 3 (Coverage gap). If $P _ { e } \ll Q _ { e }$ , then

$$
\left| \mathbb { P } \{ y _ { 0 } \in { \widehat { C } } _ { \alpha } ^ { \mathrm { W } } ( { \pmb x } _ { 0 } , { \pmb s } _ { 0 } ) \} - ( 1 - \alpha ) \right| \leq \frac { 1 } { m + 1 } + \sqrt { \frac { \mathrm { K L } ( P _ { e } \| Q _ { e } ) } { 2 } } .\tag{13}
$$

Exchangeability places the coverage probability within $1 / ( m + 1 )$ of its nominal level under $Q _ { e }$ When we replace $Q _ { e }$ with the real law $P _ { e } ,$ , the probability changes by at most the total variation distance between the two laws. Applying Pinsker’s inequality then gives us (13). Supplementary Material Section S4 contains the proof.

Now assume that the true and working residual laws have positive-definite covariance matrices $\Sigma _ { + }$ and $\Sigma _ { + } ^ { \mathrm { w } } = { \pmb { L } } { \pmb { L } } ^ { \top }$ , respectively, and are centred Gaussian. Define $\Omega = ( L ^ { - 1 } \Sigma _ { + } L ^ { - \top } ) _ { J , J } .$ , the actual covariance matrix of the transformed calibration and target errors, where $J = \{ n _ { \mathcal { F } } { + } 1 , . . . , N \}$ Their covariance matrix under the working Gaussian reference is $I _ { m + 1 }$ . If $\rho _ { 1 } , \ldots , \rho _ { m + 1 }$ are Ω’s eigenvalues, then $\begin{array} { r } { \mathrm { K L } \{ N ( \mathbf { 0 } , \Omega ) \| N ( \mathbf { 0 } , I _ { m + 1 } ) \} = \frac { 1 } { 2 } \sum _ { j } ( \rho _ { j } - 1 - \log \rho _ { j } ) } \end{array}$ , and substituting into (13) yields

$$
\left| \mathbb { P } \{ y _ { 0 } \in \widehat { C } _ { \alpha } ^ { \mathrm { W } } ( { \boldsymbol x } _ { 0 } , { \boldsymbol s } _ { 0 } ) \} - ( 1 - \alpha ) \right| \leq \frac { 1 } { m + 1 } + \frac { 1 } { 2 } \left\{ \sum _ { j = 1 } ^ { m + 1 } \bigl ( \rho _ { j } - 1 - \log \rho _ { j } \bigr ) \right\} ^ { 1 / 2 } .\tag{14}
$$

## 4.3 A local coverage diagnostic

At a given target, (14) does not give us information about whether the coverage is higher or lower than nominal. This is determined by the ratio of the scale of the transformed target error to that of the calibration scores.

Theorem 4 (Relative-scale characterization). Suppose the residual process is centred and Gaussian. Write $z _ { 0 } = e _ { m + 1 }$ , let $\kappa _ { i } ^ { 2 } = \mathbb { E } ( e _ { i } ^ { 2 } )$ for $i = 1 , \ldots , r$ m and $\kappa ( \pmb { s } _ { 0 } ) ^ { 2 } = \mathbb { E } ( \ b { z } _ { 0 } ^ { 2 } )$ , and let G be the standard half-normal distribution function with $\xi _ { 1 - \alpha } = G ^ { - 1 } ( 1 - \alpha )$ . Define

$$
\overline { { F } } _ { m } ( t ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } G \Bigl ( \frac { t } { \kappa _ { i } } \Bigr ) , \qquad q _ { m } ^ { \star } = \overline { { F } } _ { m } ^ { - 1 } ( 1 - \alpha ) , \qquad \kappa _ { \mathrm { r e l } } ( s _ { 0 } ) = \frac { \xi _ { 1 - \alpha } \kappa ( s _ { 0 } ) } { q _ { m } ^ { \star } } ,\tag{15}
$$

and let $\begin{array} { r } { v _ { m } = m ^ { - 2 } \sum _ { i , j = 1 } ^ { m } \mathrm { C o r r } ( e _ { i } , e _ { j } ) ^ { 2 } } \end{array}$ . Suppose the calibration scales $\kappa _ { i }$ are bounded above and away from zero, $\kappa ( s _ { 0 } )$ is bounded away from zero, and $v _ { m } \to 0$ . Then, as $m \to \infty$

$$
\mathbb { P } \{ y _ { 0 } \in \widehat { C } _ { \alpha } ^ { \mathrm { W } } ( \boldsymbol { x } _ { 0 } , \boldsymbol { s } _ { 0 } ) \} = G \bigg \{ \frac { \xi _ { 1 - \alpha } } { \kappa _ { \mathrm { r e l } } ( \boldsymbol { s } _ { 0 } ) } \bigg \} + O \big ( v _ { m } ^ { 1 / 3 } \big ) .\tag{16}
$$

The result follows by replacing the conformal threshold with its large-sample value $q _ { m } ^ { \star }$ , which is the $( 1 - \alpha )$ quantile of the average distribution of the absolute calibration errors. The condition $v _ { m } \to 0$ requires enough calibration information to remain after whitening for the threshold to concentrate. It permits the target and calibration errors to be dependent, permits diferent calibration scales, and does not impose stationarity or ordering. Section S5 of the Supplementary Material contains the proof.

We call $\kappa _ { \mathrm { r e l } } ( s _ { 0 } )$ in (15) the relative predictive scale. Equation (16) implies approximately nominal coverage when $\kappa _ { \mathrm { r e l } } ( s _ { 0 } ) = 1$ , conservative coverage when $\kappa _ { \mathrm { r e l } } ( s _ { 0 } ) < 1$ , and undercoverage when $\kappa _ { \mathrm { r e l } } ( s _ { 0 } ) > 1$ . The pointwise relative scale is generally unknown. But held-out responses from the same region ofer a regional approximation. We use the root-mean-square transformed error among the regional held-out data to estimate $\kappa ( s _ { \mathrm { 0 } } )$ . We then compare it to the conformal threshold that the interval actually employs. This diagnostic identifies areas that may be undercovered or overcovered without requiring the true covariance matrix. For technical details of implementation, please refer to Supplementary Material Section S5.1.

## 5 Simulation study

The simulations address three questions. (Q1) Does whitening result in a narrower and more stable interval in an exchangeable regime when all methods retain nominal coverage? (Q2) Does the benefit of whitening increase when the calibration residuals provide more information about the target than the fitting residuals? (Q3) When the calibration and the target data come from diferent environments, does that lead to a coverage failure? If so, can the proposed prospective diagnostic identify this failure in advance?

## 5.1 Methods and designs

The three questions use diferent sampling designs and allocations of labelled observations between the fitting and calibration sets. Regime A addresses Q1 by allocating them at random on a uniform sampling design, which guarantees that the design maintains exchangeability.

One can use any response-independent allocation of the fitting and calibration sets, but random splitting does not cover all practically relevant instances. In many applications, the split is predetermined because a model has already been trained on past data and later observations are only available for calibration. Even when we can choose the split, a single fitting–calibration split may be used for multiple targets, and for some of them the points in the calibration set may be the most informative. As shown in the two-cluster example (Figure 1), the performance of GSCP and LSCP can depend strongly on this allocation, since they do not make use of the predictive information in the calibration data. Since whitening makes use of this information, under the oracle setup of the two-cluster example, it was shown to be largely unafected by the choice of the split, while also producing narrower and more stable intervals.

Regimes B and C investigate the efect on performance when the sampling and splitting are nonrandom. Regime B addresses Q2 by allocating the points most informative about the target to the calibration set to varying degrees, which isolates the gain of utilizing the predictive information in the calibration residuals in whitening. Regime C addresses Q3 by creating dissimilarity between the calibration and target prediction environments through nonrandom spatial geometry, extrapolation, and nonstationarity. It evaluates both the resulting coverage and the ability of the proposed diagnostic to anticipate target-specific failures.

We compare four methods: (1) naive split conformal prediction from (1); (2) the split version of GSCP in (5), with the covariance estimator of the scp package accompanying Mao et al. (2024); (3) its localized version, LSCP, with bandwidth $h = 0 . 4 0 ;$ and (4) the proposed whitened interval from (10), using Algorithm 1.

Table 1: Eficiency and stability under random sampling. Each entry reports coverage/mean width, with the CVR from fixed-design counterparts in parentheses. Nominal coverage is 0.90.
<table><tr><td>Design</td><td>Naive</td><td>GSCP</td><td>LSCP</td><td>Whitened</td></tr><tr><td>Exponential, φ = 0.5</td><td>.899/2.19 (2.33)</td><td>.902/1.75 (1.91)</td><td>.909/1.80 (1.89)</td><td>.900/1.66 (1.12)</td></tr><tr><td>Exponential, φ = 1</td><td>.907/2.59 (2.54)</td><td>.910/1.84 (2.00)</td><td>.918/1.90 (1.79)</td><td>.916/1.76 (1.10)</td></tr><tr><td>Exponential, φ = 2</td><td>.908/3.02 (2.25)</td><td>.918/2.01 (1.36)</td><td>.922/2.07 (1.47)</td><td>.907/1.93 (1.11)</td></tr><tr><td>Exponential, φ = 6</td><td>.910/3.59 (1.87)</td><td>.902/2.42 (1.21)</td><td>.911/2.50 (1.26)</td><td>.903/2.32 (1.16)</td></tr><tr><td>Exponential, φ = 8</td><td>.907/3.64 (1.54)</td><td>.906/2.55 (1.17)</td><td>.910/2.62 (1.12)</td><td>.903/2.43 (1.06)</td></tr><tr><td>Exponential, φ = 30</td><td>.906/3.74 (1.07)</td><td>.900/3.37 (1.05)</td><td>.911/3.47 7 (1.08)</td><td>.902/3.24 (1.05)</td></tr><tr><td>Matérn,  $\nu = 3 / 2$ </td><td>.909/3.56 (1.96)</td><td>.913/2.02 (1.61)</td><td>.919/2.08 3 (1.55)</td><td>.906/1.94 (1.31)</td></tr><tr><td>Matérn,  $\nu = 5 / 2$ </td><td>.913/3.55 (2.23)</td><td>.904/1.93 (1.86)</td><td>.909/1.99 (1.68)</td><td>.895/1.85 (1.30)</td></tr><tr><td>Gaussian</td><td>.906/3.63 (1.93)</td><td>.906/1.97 (1.64)</td><td>.920/2.03 (1.82)</td><td>.908/1.89 (1.24)</td></tr></table>

For each simulation setup, we use 500 independently generated data sets of 501 spatial sites. For each site, we draw five i.i.d. Unif(0, 1) covariates independently of location, and compute its response as $y ( s ) = \{ 1 0 \sin ( \pi x _ { 1 } x _ { 2 } ) + 2 0 ( x _ { 3 } - 0 . 5 ) ^ { 2 } + 1 0 x _ { 4 } + 5 x _ { 5 } \} / 6 + r ( s )$ , where $r ( s )$ is a mean-zero spatial field. Five sites are held out as prediction targets, leaving 496 labelled observations, of which, unless stated otherwise, 330 form the fitting set and $m = 1 6 6$ the calibration set.

Unless stated otherwise, $r ( s )$ is a Gaussian process with covariance

$$
\mathrm { C o v } \{ r ( s ) , r ( s ^ { \prime } ) \} = \sigma ^ { 2 } \rho ( \| s - s ^ { \prime } \| ) + \tau ^ { 2 } \mathbf { 1 } \{ s = s ^ { \prime } \} , \qquad \rho ( d ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \big ( \sqrt { 2 \nu } \phi d \big ) ^ { \nu } K _ { \nu } \big ( \sqrt { 2 \nu } \phi d \big ) ,\tag{17}
$$

with $\sigma ^ { 2 } = 1 { \mathrm { ~ a n d ~ } } \tau ^ { 2 } = 0 . 1$ , where $K _ { \nu }$ is the modified Bessel function of the second kind. Here $\nu$ controls the smoothness of the process, with larger values giving smoother sample paths, and $\phi$ is an inverse-range parameter, with larger values giving faster decay. The case $\nu = 1 / 2$ is the exponential correlation. We also use the Gaussian correlation $\rho ( d ) = e ^ { - ( \dot { \phi } d ) ^ { 2 } }$

Please see Supplementary Material Section S6 for more details, including the regime-specific data-generating processes, the detailed implementation of every method, and the sensitivity of the comparison to the implementation choices.

## 5.2 Results

Regime A: eficiency and stability under random sampling. Regime A considers nine diferent settings with stationary residual processes under random locations and random splits. The first six settings use an exponential covariance and vary the decay parameter $\phi .$ Smaller values of $\phi$ produce stronger and longer-range dependence. The final three settings hold $\phi = 6$ fixed and increase the smoothness, which makes the field smoother at short distances. We compute coverage and mean width by averaging over random designs (i.e., the locations are redrawn in each replicate). We measure stability using the CVR defined in Section 2.2.2. To isolate the natural width variation of the methods from design efects, we compute the CVR on a fixed-design counterpart of each setting, in which one configuration of locations and split is held fixed across replicates.

In all settings (Table 1), whitening produces the narrowest interval. Its mean width is approximately 4% and 7% smaller than that of GSCP and LSCP, respectively. In comparison to naive split conformal, whitened intervals are 14% to 48% narrower across designs, with an average of 35%. The widths of the whitened intervals are also much more stable than those of the competing methods. The whitened interval has the lowest CVR in every setting (tied with GSCP at ϕ = 30).

Table 2: Value of informative calibration sets. Method entries report coverage/mean width. The last two columns give the available gain and the achieved width reduction of whitening relative to GSCP, both in percent. Nominal coverage is 0.90.
<table><tr><td>Design</td><td>Naive</td><td>GSCP</td><td>LSCP</td><td></td><td>Whitened Available gain Achieved gain</td><td></td></tr><tr><td>Halo 10</td><td></td><td></td><td></td><td>.900/3.01.894/2.23 .907/2.30.898/1.92</td><td>20.2</td><td>14.0</td></tr><tr><td>Halo 20</td><td></td><td></td><td></td><td>.900/3.03 .899/2.35 .910/2.41 .909/1.91</td><td>27.1</td><td>18.7</td></tr><tr><td>Halo 40</td><td></td><td></td><td></td><td>.904/3.12.908/2.57.916/2.62.913/1.93</td><td>34.1</td><td>24.8</td></tr><tr><td>Transect</td><td></td><td></td><td></td><td>.904/2.95.902/2.25.912/2.30.903/1.92</td><td>17.0</td><td>14.6</td></tr><tr><td>Boundary</td><td></td><td>.890/3.27.894/2.48.910/2.57.899/1.97</td><td></td><td></td><td>26.5</td><td>20.3</td></tr><tr><td>Monitoring network .905/3.23 .899/3.06.917/3.14 .918/2.16</td><td></td><td></td><td></td><td></td><td>40.4</td><td>29.3</td></tr></table>

Its advantage over GSCP and LSCP is largest under strong dependence. Naive conformal has the highest CVR in the presence of moderate to strong spatial dependence. This is expected, since it does not account for any spatial information. The methods perform comparably in terms of stability under weak dependence (ϕ = 30), where all of them approach the independent-data benchmark of one. Therefore, Regime A shows that even under random sampling and splitting, where whitening has the least to gain, it produces the narrowest and most stable intervals.

We only use Gaussian processes as residuals here. However, Theorem 1 covers any elliptical residual law, and Supplementary Material Section S6.2.4 reports results under a spatial t process.

Regime B: the value of predictive calibration residuals. Regime B considers six settings in which the calibration set is, to diferent degrees, more informative about the target set than the fitting set. In the three halo designs shown in the upper row of Figure 3, the k nearest points to each target are assigned to calibration data. Increasing k therefore shifts more local information about the target from the fitting to the calibration set. The lower row considers three more structured sampling geometries: transect sampling, boundary prediction, and a cluster monitoring network. For each setting, we measure how much smaller the predictive standard deviation at the target is, under the true covariance, when conditioning on both the fitting and calibration sets rather than on the fitting set alone. This is given in Table 2 under Available gain, as it measures how much the additional conditioning on the calibration set helps whitening over GSCP.

In Table 2, all methods remain near nominal coverage, so their widths are directly comparable. As k increases in the halo designs, more observations near the targets are assigned to calibration. GSCP and LSCP widen because neither can use those observations to predict the target. The whitened width stays nearly unchanged, since the total information available from the labelled observations remains constant in these designs. Whitening also gives the narrowest interval in the other three designs. Across the six designs, the width reduction of whitening relative to GSCP (Achieved gain) has correlation 0.98 with the available gain. Regime B demonstrates that the advantage of whitening increases as the calibration data become more informative about the target than the fitting set.

Regime C: coverage beyond random sampling. Regime C distinguishes nonrandom spatial assignment from failures of calibration–target comparability. The upper row of Figure 4 assigns the fitting, calibration, and target observations to diferent parts of the domain under a stationary process (Table 3). All methods maintain near-nominal coverage here, and whitening produces the narrowest interval. It is roughly 40% narrower than naive split conformal and 6–39% narrower than GSCP and LSCP.

![](images/11476809446c9308c903ac065c1f083f2d30783b6ae12c38aa7d0101d81b304e.jpg)  
Figure 3: Spatial designs used to vary the predictive information contained in the calibration residuals. Blue circles denote fitting observations, orange triangles denote calibration observations, and red stars denote prediction sites. The halo designs assign progressively more observations near each target to calibration, while the lower panels represent transect sampling, boundary prediction, and a cluster monitoring network. Each title reports the available gain.

In the distant-patch design, the calibration observations are located in a small patch, far from the targets. GSCP undercovers here. LSCP does not solve this problem either, and reaches nominal coverage only when its unbounded intervals are counted as covering. The calibration errors in the patch are closely correlated with each other but nearly uncorrelated with the target error. Whitening explicitly accounts for this correlation, and hence attains nominal coverage with the narrowest interval.

Under block holdout the covariance family is correct, but the calibration observations have near neighbours in the fitting set and the targets do not. As a result, their predictive variances respond diferently to error in the estimated range, which can leave the transformed target errors larger than the calibration scores. Every method undercovers here.

Under varying range, every method undercovers except LSCP. Localization downweights calibration observations far from the target. The two halves are separated by a straight line, so this prioritises calibration sites in the target’s own half that share its dependence range. A stationary working covariance cannot capture the sudden jump in the range, which results in undercoverage for our proposed approach.

It is reasonable to wonder if such undercoverage can be predicted in the absence of knowledge about the actual data-generating process. Our proposed diagnostic in Section 4.3 is developed to answer this. We evaluate the diagnostic on Regime C, which includes the designs where coverage fails. In each replicate, $\widehat { \kappa _ { \mathrm { r e l } } }$ is estimated from 200 additional labelled observations drawn from the region the targets occupy, separate from the 501 sites used for fitting, calibration, and evaluation. The predicted coverage is $G ( \xi _ { 1 - \alpha } / \widehat { \kappa _ { \mathrm { r e l } } } )$ . Please refer to Supplementary Material Section S5.1 for implementation details.

![](images/87c64feda1aeb6bee0d653303f95b14eb2381dc5539d4afb668d49a7d5c7b778.jpg)  
Figure 4: Designs used to distinguish nonrandom spatial assignment from failures of calibration–target comparability. Symbols are as in Figure 3. The upper row shows disjoint-halves, quadrant, and clustered calibration designs under a stationary process. The lower row shows distant-patch calibration, block holdout, and varying dependence range.

In the four designs where whitening maintains coverage, the predicted and realized coverage difer by at most 0.006. Under block holdout and varying range, the diagnostic correctly predicts undercoverage, within 2 percentage points of the realized value. Regime C shows that the diagnostic separates the settings in which whitening remains reliable from those at risk of undercoverage, and correctly approximates the size of the loss.

Table 3: Coverage beyond random sampling. Method entries report coverage/mean width. The last column gives the coverage predicted for the whitened interval from 200 separate diagnostic observations in the target region. Under distant patch, LSCP is unbounded at 25% of the targets, which we count as covered; its finite intervals have coverage/mean width .876/2.14. Nominal coverage is 0.90.
<table><tr><td>Design</td><td>Naive</td><td>GSCP</td><td>LSCP</td><td></td><td>Whitened Predicted coverage</td></tr><tr><td colspan="6">Role-dependent sampling</td></tr><tr><td>Disjoint halves</td><td></td><td></td><td></td><td>.897/3.41 .900/3.22.910/3.24 .914/1.98</td><td>.908</td></tr><tr><td>Quadrant</td><td></td><td></td><td></td><td>.889/3.15.895/2.73.902/2.79.907/1.93</td><td>.905</td></tr><tr><td>Clustered calibration .903/3.21 .906/2.04 .923/2.18 .904/1.93</td><td></td><td></td><td></td><td></td><td>.901</td></tr><tr><td colspan="6">Coverage stress tests</td></tr><tr><td>Distant patch</td><td></td><td></td><td></td><td>.914/3.34.850/1.90.907/∞ .908/1.86</td><td>.911</td></tr><tr><td>Block holdout</td><td></td><td></td><td></td><td>.853/3.05.870/2.74.880/2.89.873/2.64</td><td>.886</td></tr><tr><td>Varying range</td><td></td><td></td><td></td><td>.866/3.39.876/2.39.902/2.59.878/2.29</td><td>.861</td></tr></table>

## 6 Fine particulate matter monitoring network

## 6.1 Data and evaluation

Air-quality assessments need estimates of annual $\mathrm { P M } _ { 2 . 5 }$ concentrations on a fine grid, but most grid cells have no monitor. Assessing exposure there requires predicting the concentration and knowing how reliable the prediction is. Environmental and geographic covariates are often used to predict the mean concentration, but their relationship with $\mathrm { P M } _ { 2 . 5 }$ is complex and unknown. Therefore, it is dificult to rely on inference from a parametric model, which often requires its assumptions to be right to guarantee the validity of the inference. Conformal prediction is a natural alternative; however, its naive form ignores the spatial correlation among nearby monitors. The monitoring network is thus precisely the kind of application for which spatial conformal inference is intended.

We analyse annual mean $\mathrm { P M } _ { 2 . 5 }$ concentrations in 2022 at 723 Environmental Protection Agency monitors covering the contiguous United States. A random forest that uses environmental covariates has an out-of-bag $R ^ { 2 }$ of 0.23. After fitting a spatial model to the residuals, we find that spatial dependence accounts for 67% of the residual variance. The practical range of the spatial dependence is around 460 kilometres. Nearby residuals are therefore likely to be informative about one another, as a typical monitor has around 69 neighbouring monitors within that range. For details on data sources and preparation, please see Supplementary Material Section S7.

To evaluate the methods, we hold out monitors and check if their intervals contain the recorded concentrations. The remaining monitors are split at random into fitting (two thirds) and calibration (one third) sets. We address two questions as follows:

1. Does whitening result in narrower intervals at comparable coverage inside the monitoring network? We split the monitors into twenty random folds and use each fold in turn as the test set.

2. Does coverage degrade when predicting in a location where there are no observed data? If so, can the diagnostic detect the regions at risk? We use whole-state holdout, where each of the 49 state-level regions is held out in turn and its monitors predicted from observations outside the state.

Table 4: Coverage and interval width in the $\mathrm { P M } _ { 2 . 5 }$ application. Each entry reports coverage/mean width in $\mu \mathrm { { g } / m ^ { 3 } }$ . Random-fold results are pooled across the twenty folds. Whole-state results are computed for each of the 49 regions and averaged with equal weight. Nominal coverage is 0.90.
<table><tr><td>Scheme</td><td>Naive</td><td>GSCP</td><td>LSCP</td><td>Whitened</td></tr><tr><td>Random folds</td><td>.903/5.94</td><td>.887/5.16</td><td>.892/5.07</td><td>.898/4.85</td></tr><tr><td>Whole-state holdout</td><td>.868/5.44</td><td>.873/5.23</td><td>.873/5.10</td><td>.879/5.13</td></tr></table>

![](images/84ec0e24c4fce447f615fda3a6b96072aa392974bd4326811e492b4af1d0309c.jpg)

![](images/331991e384307bb70730fba7973bd736fdb04894c8af13de619cda33d428d186.jpg)  
Figure 5: Empirical coverage and mean width by distance to the nearest fitting monitor under random folds at nominal coverage 0.90. GSCP and LSCP reach the naive width as distance increases. Their smaller widths in the nearest quintile are accompanied by coverage seven to eight percentage points below nominal.

## 6.2 Random folds

When we pool the results across the twenty folds, the four methods remain broadly close to nominal coverage (Table 4). The whitened method produces the narrowest interval, by approximately 6% and 4% against GSCP and LSCP, respectively, and 18% against naive split conformal. The prediction performance may depend on the distance between a target monitor and the nearest fitting monitor. Thus, Figure 5 groups target monitors into five quintiles according to their distances. The naive interval ignores location; thus, it is consistently wide across all five groups. The other three methods start about 26 to 29% narrower than naive in the nearest quintile and widen with distance. The whitened interval is the narrowest everywhere except in the nearest quintile, where GSCP and LSCP are narrower but undercover by seven to eight percentage points. The widths of the GSCP and LSCP intervals reach those of naive in the furthest quintile, but the whitened interval remains 8 to 10% below them.

## 6.3 Whole-state holdout

Whole-state holdout is much more dificult because each target monitor is in a region with no fitting or calibration observations. As a result, all four approaches fall short of nominal coverage. LSCP and the whitened interval are the two best-performing methods here, and are practically indistinguishable in both coverage and width. The general undercoverage is expected, since here the calibration scores are derived from regions that do not always mirror the one being predicted. The most pertinent question is whether this failure is predictable in each region, which is what the diagnostic of Section 4.3 answers.

Table 5: Regional diagnostic under whole-state holdout. Regions are grouped into terciles of $\widehat { \kappa _ { \mathrm { r e l } } }$ , estimated from a diagnostic half of each region’s monitors. Coverage is measured on the disjoint evaluation half, and estimated coverage is $G ( \xi _ { 1 - \alpha } / \widehat { \kappa _ { \mathrm { r e l } } } )$ . Estimated and realized coverage are averaged over 20 random splits within each region and then over regions with equal weight, and regions with fewer than sixteen monitors are excluded. Nominal coverage is 0.90.
<table><tr><td>Tercile of  $\widehat { \kappa _ { \mathrm { r e l } } }$ </td><td>Regions</td><td>Mean  $\widehat { \kappa _ { \mathrm { r e l } } }$ </td><td>Estimated coverage</td><td>Realized coverage</td></tr><tr><td>Lowest</td><td>7</td><td>0.60</td><td>0.987</td><td>0.979</td></tr><tr><td>Middle</td><td>6</td><td>0.93</td><td>0.915</td><td>0.909</td></tr><tr><td>Highest</td><td>6</td><td>1.38</td><td>0.769</td><td>0.744</td></tr></table>

We test this prospectively by randomly separating the monitors in each held-out region into a diagnostic half and a disjoint evaluation half. We use the responses in the diagnostic half to estimate the scale, and the evaluation half to measure the coverage. We repeat the split 20 times to reduce sensitivity to how the monitors in smaller regions are divided. To ensure there are enough monitors in both halves, we restrict the analysis to regions with at least sixteen monitors, which leaves us with 19 regions.

When we average over the splits, $\widehat { \kappa _ { \mathrm { r e l } } }$ has Spearman correlation −0.90 with realized coverage across the 19 regions. We group the regions into terciles of $\widehat { \kappa _ { \mathrm { r e l } } }$ (Table 5). Since a higher value of $\widehat { \kappa _ { \mathrm { r e l } } }$ indicates undercoverage, the highest tercile of $\widehat { \kappa _ { \mathrm { r e l } } }$ corresponds to an estimated coverage of 0.769, which increases to 0.987 in the lowest tercile. Realized coverage closely follows this trend and is within one percentage point of the estimate in the lower two terciles. In the highest, the estimate overstates it by about two and a half points.

## 7 Discussion

Existing spatial conformal methods do not use the predictive power of the calibration residuals, which leaves shared spatial information in both the calibration and the target residuals. Conditioning on them directly is not possible, since each calibration residual would then be conditioned on itself, resulting in a degenerate score. We propose to bypass this problem by conditioning on the calibration residuals sequentially. When the residuals are elliptical and the working covariance is correct, this restores score exchangeability. Under additional conditions, it also stabilizes the threshold and produces an interval asymptotically as short as the oracle.

When the working covariance captures the residual dependence suficiently well, the coverage is nominal and the whitened interval is shorter and more stable than alternatives. The advantage of this method is largest when the observations near the target are in the calibration set, which cannot be used for prediction in GSCP and LSCP. A similar pattern holds for the air-quality application. Under random folds, whitening produced the narrowest interval, with its margin over GSCP and LSCP largest at monitors far from the fitting set. Under whole-state holdout, it was comparable to LSCP.

Since we do not know the true data-generating process in practice, the working covariance may not be an adequate representation of the residual dependence structure. As a result, all methods that use a working covariance matrix, including GSCP, LSCP and our whitening approach, might sufer from undercoverage. We develop a diagnostic for our method that identifies the regions at risk without knowing the true data-generating process. In the simulations it captured the coverage loss, and in the application it identified the high-risk regions.

Although not entirely distribution-free, the approach is model-lean. An elliptical residual law and a correct working covariance are suficient for exact coverage, and further conditions are needed for eficiency and local-coverage outcomes. Additionally, the estimated covariance is treated as fixed in the eficiency and misspecification results. Future research should naturally focus on building a diagnostic-based adjustment with a formal coverage guarantee and extending the guarantees to approximated and nonstationary covariance models.

## Code availability

Code to reproduce all numerical results is available at https://github.com/ArkajyotiSaha/ spatial-conformal-whitening.

## Acknowledgements

We acknowledge the use of generative AI to help polish the manuscript and write code for the numerical experiments. The authors independently reviewed all work and take sole responsibility for the content.

## References

Angelopoulos, A. N. and Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591.

Berrocal, V. J., Gelfand, A. E., and Holland, D. M. (2010). A spatio-temporal downscaler for output from numerical models. Journal of Agricultural, Biological and Environmental Statistics, 15(2):176–197.

Datta, A., Banerjee, S., Finley, A. O., and Gelfand, A. E. (2016). Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets. Journal of the American Statistical Association, 111(514):800–812.

Fang, K.-T., Kotz, S., and Ng, K. W. (1990). Symmetric Multivariate and Related Distributions. Chapman and Hall.

Guan, L. (2023). Localized conformal prediction: A generalized inference framework for conformal prediction. Biometrika, 110(1):33–50.

Jiang, H. and Xie, Y. (2026). Spatial conformal inference through localized quantile regression. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 53181–53202. PMLR.

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523):1094– 1111.

Mao, H., Martin, R., and Reich, B. J. (2024). Valid model-free spatial prediction. Journal of the American Statistical Association, 119(546):904–914.

Matheron, G. (1963). Principles of geostatistics. Economic Geology, 58(8):1246–1266.

Moon, I. and Kim, Y. (2026). Conformal prediction for spatial data with GAM-trend adjustment. Communications for Statistical Applications and Methods, 33(4):463–474.

Oliver, M. A., editor (2010). Geostatistical Applications for Precision Agriculture. Springer.

Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. (2002). Inductive confidence machines for regression. In Proceedings of the 13th European Conference on Machine Learning, volume 2430 of Lecture Notes in Computer Science, pages 345–356. Springer.

Saha, A. and Datta, A. (2018). BRISC: Bootstrap for rapid inference on spatial covariances. Stat, 7(1):e184.

Sampson, P. D. and Guttorp, P. (1992). Nonparametric estimation of nonstationary spatial covariance structure. Journal of the American Statistical Association, 87(417):108–119.

Serfling, R. J. (1980). Approximation Theorems of Mathematical Statistics. Wiley.

Tsybakov, A. B. (2009). Introduction to Nonparametric Estimation. Springer.

Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

# Supplementary Material for

# “Conformal Prediction for Spatially Dependent Data via Sequential Whitening”

This Supplementary Material contains the technical proofs, implementation details, and supporting numerical results for the main text. Sections S1–S3 develop the exact-whitening theory, including finite-sample validity and oracle eficiency. Sections S4 and S5 study imperfect whitening and develop the prospective relative-scale diagnostic. Section S6 contains details of the simulation designs and competing methods, and examines sensitivity to the main implementation choices. Section S7 then describes the details of data processing steps for the $\mathrm { P M } _ { 2 . 5 }$ application.

## S1 Sequential prediction is triangular whitening

We prove Lemma 2 here. This links the implemented sequential predictors to the matrix transformation used in the following arguments.

Proof of Lemma 2. Fix i and recall $k _ { i } = n _ { \mathscr { F } } + i$ and $I _ { i } = \{ 1 , \ldots , k _ { i } - 1 \}$ . Let $\boldsymbol { \ell } = ( \ell _ { k _ { i } 1 } , \dots , \ell _ { k _ { i } , k _ { i } - 1 } ) ^ { \top }$ be the first $k _ { i } - 1$ entries of the $k _ { i } \mathrm { t h }$ row of L. The leading block of L and the corresponding blocks of $\pmb { \Sigma } _ { + } ^ { \mathrm { w } } = \pmb { L } \pmb { L } ^ { \top }$ satisfy

$$
\begin{array} { r } { L _ { 1 : k _ { i } , 1 : k _ { i } } = \left( \begin{array} { l l } { L _ { I _ { i } , I _ { i } } } & { \textbf { 0 } } \\ { \ell ^ { \top } } & { \ell _ { k _ { i } k _ { i } } } \end{array} \right) , \quad \Sigma _ { I _ { i } , I _ { i } } ^ { \mathrm { w } } = L _ { I _ { i } , I _ { i } } L _ { I _ { i } , I _ { i } } ^ { \top } , \quad \Sigma _ { k _ { i } , I _ { i } } ^ { \mathrm { w } } = \ell ^ { \top } L _ { I _ { i } , I _ { i } } ^ { \top } , \quad \Sigma _ { k _ { i } , k _ { i } } ^ { \mathrm { w } } = \ell ^ { \top } \ell + \ell _ { k _ { i } k _ { i } } ^ { 2 } . } \end{array}\tag{S1}
$$

Substituting into (8) gives $\widehat { \mu } _ { i } = \ell ^ { \top } L _ { I _ { i } , I _ { i } } ^ { - 1 } r _ { + , I _ { i } }$ and $\widehat { \nu } _ { i } = \ell _ { k _ { i } k _ { i } } ^ { 2 }$ . We can then use forward substitution in (S1) to get

$$
\left( { \cal L } ^ { - 1 } r _ { + } \right) _ { k _ { i } } = \frac { r _ { i } - \widehat \mu _ { i } } { \widehat \nu _ { i } ^ { 1 / 2 } } = e _ { i } ,
$$

which is (9).

## S2 Exact coverage after whitening

Theorem 1 demonstrates that the same transformation that enables us to utilise calibration residuals to predict the target also recovers the exchangeability needed for the conformal coverage guarantee.

Proof of Theorem 1. $r _ { + }$ is centred and elliptical with scatter matrix $\pmb { \Sigma } _ { + } ^ { \mathrm { w } } = \pmb { L } \pmb { L } ^ { \top }$ . Hence, we can write it as follows:

$$
\begin{array} { r } { \pmb { r } _ { + } \overset { d } { = } \boldsymbol { R } \pmb { L } \mathbf { U } , } \end{array}\tag{S2}
$$

where U is uniform on the unit sphere $\mathbb { S } ^ { N - 1 }$ and $R > 0$ is independent of U (Fang et al., 1990, Theorem 2.5). If $\mathbb { E } R ^ { 2 } < \infty$ , the covariance of $r _ { + }$ is $( \mathbb { E } R ^ { 2 } / N ) \Sigma _ { + } ^ { \mathrm { w } }$ . In the Gaussian case with covariance Σ<sup>w</sup><sub>+</sub>, $R ^ { 2 } \sim \chi _ { N } ^ { 2 }$

Step 1: whitening produces a spherical vector. Applying ${ \cal L } ^ { - 1 }$ to (S2) gives

$$
\pmb { e } _ { + } = \pmb { L } ^ { - 1 } \pmb { r } _ { + } \overset { d } { = } \boldsymbol { R } \mathbf { U } .\tag{S3}
$$

For every orthogonal $N \times N$ matrix $Q ,$ rotational invariance of the uniform distribution on the sphere gives $Q \mathbf { U } \overset { d } { = } \mathbf { U }$ . Independence of R and U therefore implies $\boldsymbol { Q } \boldsymbol { e } _ { + } \overset { d } { = } \boldsymbol { e } _ { + }$

Step 2: the transformed calibration and target errors are exchangeable. Let $Q _ { \pi }$ permute the last $m + 1$ coordinates and fix the initial $n { \mathcal { F } }$ coordinates. Since $Q _ { \pi }$ is orthogonal, (S3) implies $Q _ { \pi } e _ { + } \stackrel { d } { = } e _ { + }$ . Lemma 2 shows that the last $m + 1$ coordinates are $( e _ { 1 } , \ldots , e _ { m + 1 } )$ , which proves that these errors, and thus their absolute values, are exchangeable.

Step 3: coverage is a rank event. Lemma 2 at $i = m + 1$ gives

$$
y _ { 0 } \in \widehat { C } _ { \alpha } ^ { \mathrm { W } } ( { \pmb x } _ { 0 } , s _ { 0 } ) \quad \iff \quad | e _ { m + 1 } | \leq \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } .\tag{S4}
$$

The law of e is absolutely continuous on $\mathbb { R } ^ { m + 1 }$ , hence absolute transformed errors have no ties almost surely. Therefore, their exchangeability makes the rank of $| e _ { m + 1 } |$ among $| e _ { 1 } | , \ldots , | e _ { m + 1 } |$ uniform on $\{ 1 , \ldots , m + 1 \}$ . By construction, the event in (S4) occurs exactly when this rank is at most $j _ { \alpha }$ . Consequently,

$$
\mathbb { P } \{ y _ { 0 } \in \widehat { C } _ { \alpha } ^ { \mathrm { W } } ( \pmb { x } _ { 0 } , \pmb { s } _ { 0 } ) \} = \frac { j _ { \alpha } } { m + 1 } = \frac { \lceil ( 1 - \alpha ) ( m + 1 ) \rceil } { m + 1 } \geq 1 - \alpha .\tag{S5}
$$

When $j _ { \alpha } = m + 1$ , this probability becomes one and it results in an unbounded interval.

## S3 Interval width and asymptotic eficiency

The whitened and GSCP intervals difer in two ways. Whitening replaces the GSCP calibration quantile with a quantile of the whitened scores. It also replaces the predictive variance given the fitting set with the predictive variance given the full labelled sample. In Section S3.1, we prove that the predictive variances determine the asymptotic width ratio of the whitened and GSCP intervals. Section S3.2 then proves Theorem 2, which shows that the proposed whitened interval is asymptotically as short as the shortest conditionally valid interval based on the full labelled sample.

## S3.1 Proof of the width ratio

Proof of Proposition 1. The GSCP and whitened widths satisfy

$$
\frac { W ^ { \mathrm { W } } ( s _ { 0 } ) } { W ^ { \mathrm { G } } ( s _ { 0 } ) } = \frac { \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } } { \widehat { q } _ { 1 - \alpha } ^ { \mathrm { G } } } \left\{ \frac { \widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( s _ { 0 } ) } { \widehat { \nu } _ { \mathcal { F } } ( s _ { 0 } ) } \right\} ^ { 1 / 2 } .\tag{S6}
$$

By the assumption of Proposition 1, the empirical distributions of the two sets of absolute calibration scores converge in probability to a common limit $G _ { 0 }$ , which is continuous and strictly increasing at $\zeta _ { 1 - \alpha } = G _ { 0 } ^ { - 1 } ( 1 - \alpha ) > 0$ . Hence

$$
\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \stackrel { p } { \longrightarrow } \zeta _ { 1 - \alpha } , \qquad \widehat { q } _ { 1 - \alpha } ^ { \mathrm { G } } \stackrel { p } { \longrightarrow } \zeta _ { 1 - \alpha } .
$$

Their ratio is therefore $1 + o _ { \mathbb { P } } ( 1 )$ . Substitution into (S6) proves (11).

## S3.2 Proof of asymptotic oracle eficiency

Let $\mathcal { D } = \mathcal { F } \cup \mathcal { C }$ . The coordinates of $e _ { + } = { L } ^ { - 1 } r _ { + }$ are i.i.d. with distribution F according to the assumptions of Theorem 2. Here F has mean zero and unit variance, and its density $f _ { e }$ is continuous,

symmetric about zero, and strictly decreasing on $( 0 , \infty )$ . Let us define

$$
H ( t ) = \mathbb { P } _ { F } ( | e | \leq t ) , \qquad \xi _ { 1 - \alpha } ^ { H } = H ^ { - 1 } ( 1 - \alpha ) ,\tag{S7}
$$

and assume the density of H is positive and continuous at $\xi _ { 1 - \alpha } ^ { H }$ . Next, we prove Theorem 2 in two steps. First, Lemma S1 gives a closed form for the oracle width in terms of $H .$ , and then we prove that the whitened interval attains that width asymptotically.

Lemma S1 (Form of the oracle interval). Under the assumptions of Theorem ${ \mathcal { Q } } ,$ the shortest interval with conditional coverage at least $1 - \alpha$ given $r _ { \mathcal { D } }$ is

$$
\begin{array} { r } { I ^ { \circ } ( { s } _ { 0 } ) = \widehat f ( { x } _ { 0 } ) + \mathbb { E } ( r _ { 0 } \mid r _ { \mathcal { D } } ) \pm \xi _ { 1 - \alpha } ^ { H } \nu ( { s } _ { 0 } ) ^ { 1 / 2 } , \qquad \nu ( { s } _ { 0 } ) = \mathrm { V a r } ( r _ { 0 } \mid r _ { \mathcal { D } } ) , } \end{array}\tag{S8}
$$

and hence $W ^ { \mathrm { o } } ( s _ { 0 } ) = 2 \xi _ { 1 - \alpha } ^ { H } \nu ( s _ { 0 } ) ^ { 1 / 2 }$

Proof. Because L is lower triangular and invertible, the first $N - 1$ coordinates of $e _ { + }$ are a measurable bijection of $r _ { \mathcal { D } }$ . The final row of $\pmb { r } _ { + } = \pmb { L } \pmb { e } _ { + }$ gives

$$
r _ { 0 } = \sum _ { j = 1 } ^ { N - 1 } \ell _ { N j } ( e _ { + } ) _ { j } + \ell _ { N N } ( e _ { + } ) _ { N } .\tag{S9}
$$

The first term is measurable with respect to $r _ { \mathcal { D } }$ , while $( e _ { + } ) _ { N } = e _ { m + 1 }$ is independent of $\mathbf { \nabla } _ { \mathbf { r } _ { D } }$ with distribution F. Since F has mean zero and unit variance, independence gives $\mathbb { E } \{ ( e _ { + } ) _ { N } \mid r _ { \mathcal { D } } \} = 0$ and Var $\{ ( e _ { + } ) _ { N } \mid r _ { \mathcal { D } } \} = 1$ , so that

$$
\mathbb { E } ( r _ { 0 } \mid r _ { T } ) = \sum _ { j = 1 } ^ { N - 1 } \ell _ { N j } ( e _ { + } ) _ { j } , \qquad \nu ( s _ { 0 } ) = \ell _ { N N } ^ { 2 } .
$$

Using the symmetry of the density $f _ { e } ,$ , and the fact that it is strictly decreasing away from zero, we have that $[ - \xi _ { 1 - \alpha } ^ { H } , \bar { \xi } _ { 1 - \alpha } ^ { H } ]$ is the shortest interval with probability at least $1 - \alpha$ . The oracle intervals in (S8) are then obtained by shifting and scaling this interval by the conditional location and scale, followed by the addition of the estimated mean $\widehat { f } ( \pmb { x } _ { 0 } )$ □

Proof of Theorem 2. Lemma 2 and (S9) show that the working predictive variance used by the whitened interval is $\widehat { \nu } _ { \mathcal { F } \cup \mathcal { C } } ( \pmb { \mathscr { s } } _ { 0 } ) = \ell _ { N N } ^ { 2 } = \nu ( \pmb { \mathscr { s } } _ { 0 } )$ . Thus

$$
\frac { W ^ { \mathrm { W } } ( s _ { 0 } ) } { W ^ { \mathrm { o } } ( s _ { 0 } ) } = \frac { \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } } { \xi _ { 1 - \alpha } ^ { H } } .\tag{S10}
$$

By (S10) the theorem reduces to showing that $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } / \xi _ { 1 - \alpha } ^ { H } = 1 + O _ { \mathbb { P } } ( m ^ { - 1 / 2 } )$ , and since $\xi _ { 1 - \alpha } ^ { H } > 0$ is fixed, this is equivalent to proving $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - \xi _ { 1 - \alpha } ^ { \bar { H } ^ { - } } = O _ { \mathbb { P } } ( m ^ { - 1 / 2 } )$ . We note that for $p _ { m } = j _ { \alpha } / m$ $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - \xi _ { 1 - \alpha } ^ { H } = \left( \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - H ^ { - 1 } ( p _ { m } ) \right) + \left( H ^ { - 1 } ( p _ { m } ) - \xi _ { 1 - \alpha } ^ { H } \right)$ , and prove the result by controlling each term separately.

First, note that the transformed calibration scores $\vert e _ { 1 } \vert , \ldots , \vert e _ { m } \vert$ are independent with distribution function H, and $j _ { \alpha } \le m$ for all suficiently large $m ,$ so the adjoined value $+ \infty$ does not afect the $j _ { \alpha } \mathrm { \cdot }$ th order statistic. Hence $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } }$ is the empirical quantile of the sample at level $p _ { m } = j _ { \alpha } / m$ . Since $p _ { m } = 1 - \alpha + O ( m ^ { - 1 } )$ and H has a density that is positive and continuous at $\xi _ { 1 - \alpha } ^ { H } ,$ asymptotic normality of central order statistics (Serfling, 1980, Section 2.3.3) implies that $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - H ^ { - 1 } ( p _ { m } ) =$ $O _ { \mathbb { P } } ( m ^ { - 1 / 2 } )$

Next, notice that the conformal rank $j _ { \alpha }$ must be an integer. This implies that the level $p _ { m }$ difers from $1 - \alpha$ by at most order $1 / m$ . Since $H ^ { - 1 }$ is diferentiable at $1 - \alpha$ , it follows that $H ^ { - 1 } ( p _ { m } ) - \xi _ { 1 - \alpha } ^ { H } = O ( m ^ { - 1 } )$ . Combining both results, we have $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - \xi _ { 1 - \alpha } ^ { H } = O _ { \mathbb { P } } ( m ^ { - 1 / 2 } ) + O ( m ^ { - 1 } )$ Since the first quantity dominates the second, we have $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - \bar { \xi } _ { 1 - \alpha } ^ { H } = \bar { O } _ { \mathbb { P } } ( m ^ { - 1 / 2 } )$ , which proves (12). □

## S4 Coverage when whitening is imperfect

Next, we consider what happens when the whitening covariance is not correct.

Let $P _ { e }$ be the actual law of $\pmb { e } = ( e _ { 1 } , \ldots , e _ { m } , e _ { m + 1 } ) ^ { \top }$ and let $Q _ { e }$ be a reference law under which $| e _ { 1 } | , \ldots , | e _ { m + 1 } |$ are exchangeable and almost surely distinct.

Proof of Theorem 3. By (S4), coverage is given by the event

$$
E = \{ | e _ { m + 1 } | \leq \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \} ,
$$

which depends only on e. Under $Q _ { e }$ , the rank argument from the proof of Theorem 1 gives

$$
\left| Q _ { e } ( E ) - ( 1 - \alpha ) \right| \leq \frac { 1 } { m + 1 } .\tag{S11}
$$

Total variation distance is the supremum of $| P ( A ) - Q ( A ) |$ over all measurable A. Hence we have the following inequality: $| P _ { e } ( E ) - Q _ { e } ( E ) | \leq \mathrm { T V } ( P _ { e } , Q _ { e } )$ . Combining this inequality with (S11) we have

$$
| P _ { e } ( E ) - ( 1 - \alpha ) | \leq { \frac { 1 } { m + 1 } } + \mathrm { T V } ( P _ { e } , Q _ { e } ) .\tag{S12}
$$

Pinsker’s inequality (Tsybakov, 2009, Lemma 2.5) tells us that total variation distance is bounded above using the KL divergence as follows:

$$
\mathrm { T V } ( P _ { e } , Q _ { e } ) \leq \left\{ \frac { \mathrm { K L } ( P _ { e } \| Q _ { e } ) } { 2 } \right\} ^ { 1 / 2 } .
$$

Combining this with (S12) gives us (13).

## S5 Relative scale and the prospective coverage diagnostic

Theorem 3 bounds the coverage error, but to evaluate the bound, we need to know the true covariance. However, this information is not available in real data applications. Theorem 4 bypasses this problem. It gives a way to identify regions at risk of undercoverage that can be estimated from residuals alone.

Throughout this section, we assume the residual vector to be centred Gaussian with positivedefinite covariance, so the transformed errors $e _ { 1 } , \ldots , e _ { m }$ and $z _ { 0 } = e _ { m + 1 }$ are jointly Gaussian and nondegenerate, with $e _ { i } \sim N ( 0 , \kappa _ { i } ^ { 2 } )$ and $z _ { 0 } \sim N ( 0 , \kappa _ { 0 } ^ { 2 } )$ , where $\kappa _ { 0 } = \kappa ( s _ { 0 } )$ . Let $G ( t ) = 2 \Phi ( t ) - 1$ and $g ( t ) = 2 \phi ( t )$ denote the distribution function and density of a standard half-normal variable, and write $\xi _ { 1 - \alpha } = G ^ { - 1 } ( 1 - \alpha )$ . Let $\scriptstyle { R _ { m } }$ be the correlation matrix of $( e _ { 1 } , \ldots , e _ { m } )$ and

$$
v _ { m } = { \frac { \| R _ { m } \| _ { F } ^ { 2 } } { m ^ { 2 } } } = { \frac { 1 } { m ^ { 2 } } } \sum _ { i , j = 1 } ^ { m } \mathrm { C o r r } ( e _ { i } , e _ { j } ) ^ { 2 } .\tag{S13}
$$

Coverage is the event that $| z _ { 0 } |$ falls below the conformal threshold. The proof has three steps. First, we show that the threshold converges to $q _ { m } ^ { \star }$ . This uses the bounds on the calibration scales and requires $v _ { m } \to 0$ . Second, coverage at $q _ { m } ^ { \star }$ equals $G \{ \xi _ { 1 - \alpha } / \kappa _ { \mathrm { r e l } } ( s _ { 0 } ) \}$ . Third, we bound the error from replacing the random threshold by $q _ { m } ^ { \star }$ . This uses the lower bound on $\kappa ( s _ { 0 } )$ . We make the following assumptions:

Assumption S1 (Bounded calibration scales). There are constants $0 < \kappa _ { \mathrm { m i n } } \le \kappa _ { \mathrm { m a x } } < \infty$ such that $\kappa _ { \mathrm { m i n } } \le \kappa _ { i } \le \kappa _ { \mathrm { m a x } }$ for all i and m.

Assumption S2 (Target scale). There is a constant $\kappa _ { 0 , \mathrm { m i n } } > 0$ such that $\kappa _ { 0 } \geq \kappa _ { 0 , \mathrm { { m i n } } }$ along the asymptotic sequence.

Assumption S3 (Weak correlation after whitening). $v _ { m } \to 0$ as $m \to \infty$

Under a correct Gaussian working covariance, $R _ { m } = \mathbf { I }$ , so $v _ { m } = 1 / m$ , and $\kappa _ { i } = \kappa _ { 0 } = 1 ;$ ; all three assumptions then hold. Next, we define the empirical distribution function of the absolute calibration errors $\left( F _ { m } ( t ) \right)$ , its expectation $( \overline { { F } } _ { m } ( t ) )$ , and the fixed value the threshold targets $( q _ { m } ^ { \star } )$ as follows

$$
F _ { m } ( t ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbf { 1 } \{ | e _ { i } | \leq t \} , \qquad \overline { { F } } _ { m } ( t ) = \mathbb { E } F _ { m } ( t ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } G \left( \frac { t } { \kappa _ { i } } \right) , \qquad q _ { m } ^ { \star } = \overline { { F } } _ { m } ^ { - 1 } ( 1 - \alpha ) ,\tag{S14}
$$

where the second equality uses the fact that $| e _ { i } | / \kappa _ { i }$ is standard half-normal.

Below we first prove that the conformal threshold converges to $q _ { m } ^ { \star }$ . We do so with the help of three lemmas. Lemma S2 bounds the covariance between the indicators of two correlated Gaussian variables. Lemma S3 uses it to bound the variance of $F _ { m } ( t )$ by $v _ { m }$ . Lemma S4 turns that variance bound into concentration of the threshold around $q _ { m } ^ { \star }$

Lemma S2 (Correlated absolute-value indicators). Let $( X , Y )$ be centred and jointly Gaussian with positive variances and correlation $\rho .$ Then for all $a , b \geq 0$

$$
\big | \mathrm { C o v } \big ( \mathbf { 1 } \{ | X | \leq a \} , \mathbf { 1 } \{ | Y | \leq b \} \big ) \big | \leq \frac { \rho ^ { 2 } } { 4 } .
$$

Proof. We write $X = \sigma _ { X } U$ and $Y = \sigma _ { Y } V$ , where $( U , V )$ is bivariate normal, with Corr $( U , V ) = \rho$ We denote the indicators as $f ( U )$ and $h ( V )$ , where $f ( u ) = \mathbf { 1 } \{ | u | \leq a / \sigma _ { X } \}$ and $h ( v ) = \mathbf { 1 } \{ | v | \leq b / \sigma _ { Y } \}$ Both of these are square integrable under the standard normal law. Hence, we can expand them in the orthonormal Hermite polynomials as $\begin{array} { r } { f = \sum _ { k > 0 } c _ { k } H _ { k } } \end{array}$ and $\begin{array} { r } { h = \sum _ { k > 0 } d _ { k } H _ { k } } \end{array}$ . Next, we apply Mehler’s formula, which gives E $\{ H _ { j } ( U ) H _ { k } ( V ) \} = \rho ^ { k } \mathbf { 1 } \bar { \{ j = k \} }$ . This implies Cov $\begin{array} { r } { \{ f ( U ) , h ( V ) \} = \sum _ { k \geq 1 } c _ { k } d _ { k } \rho ^ { k } } \end{array}$ Since both indicators are even, we have $c _ { k } = d _ { k } = 0$ for odd k, so every remaining term has $k \geq 2$ and $| \rho | ^ { k } \le \rho ^ { 2 }$ . Next, we apply the Cauchy–Schwarz inequality to the coeficient sequences and obtain

$$
\vert \mathrm { C o v } \{ f ( U ) , h ( V ) \} \vert \leq \rho ^ { 2 } \Bigl ( \sum _ { k \geq 1 } c _ { k } ^ { 2 } \Bigr ) ^ { 1 / 2 } \Bigl ( \sum _ { k \geq 1 } d _ { k } ^ { 2 } \Bigr ) ^ { 1 / 2 } = \rho ^ { 2 } \big [ \mathrm { V a r } \{ f ( U ) \} \mathrm { V a r } \{ h ( V ) \} \big ] ^ { 1 / 2 } \leq \frac { \rho ^ { 2 } } { 4 } .
$$

Here the last inequality uses the fact that an indicator has variance at most $1 / 4$

Lemma S3 (Variance of the calibration distribution). For every $t \geq 0$ , Var $\{ F _ { m } ( t ) \} \leq v _ { m } / 4$ Proof. By Lemma S2,

$$
\mathrm { V a r } \{ F _ { m } ( t ) \} = \frac { 1 } { m ^ { 2 } } \sum _ { i , j = 1 } ^ { m } \mathrm { C o v } ( { \bf 1 } \{ | e _ { i } | \leq t \} , { \bf 1 } \{ | e _ { j } | \leq t \} ) \leq \frac { 1 } { 4 m ^ { 2 } } \sum _ { i , j = 1 } ^ { m } \mathrm { C o r r } ( e _ { i } , e _ { j } ) ^ { 2 } = \frac { v _ { m } } { 4 } .
$$

Lemma S4 (Concentration of the whitened conformal threshold). Under Assumption S1, let $c = \kappa _ { \mathrm { m a x } } ^ { - 1 } g ( 2 \kappa _ { \mathrm { m a x } } \xi _ { 1 - \alpha } / \kappa _ { \mathrm { m i n } } )$ and $\delta _ { 0 } = \kappa _ { \mathrm { m i n } } \xi _ { 1 - \alpha } / 2$ . For all suficiently large m and all δ with $4 / ( c m ) \leq \delta \leq \delta _ { 0 }$

$$
\mathbb { P } \big ( | \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - q _ { m } ^ { \star } | > \delta \big ) \leq \frac { 2 v _ { m } } { c ^ { 2 } \delta ^ { 2 } } .\tag{S15}
$$

Proof. Since $\kappa _ { \mathrm { m i n } } \le \kappa _ { i } \le \kappa _ { \mathrm { m a x } }$ , we have $\overline { { F } } _ { m } ( \kappa _ { \mathrm { m i n } } \xi _ { 1 - \alpha } ) \leq 1 - \alpha \leq \overline { { F } } _ { m } ( \kappa _ { \mathrm { m a x } } \xi _ { 1 - \alpha } )$ . This implies $q _ { m } ^ { \star } \in \left[ \kappa _ { \operatorname* { m i n } } \xi _ { 1 - \alpha } , \kappa _ { \operatorname* { m a x } } \xi _ { 1 - \alpha } \right]$ and $[ q _ { m } ^ { \star } - \delta _ { 0 } , q _ { m _ { \star } } ^ { \star } + \delta _ { 0 } ] \subseteq [ \kappa _ { \operatorname* { m i n } } \xi _ { 1 - \alpha } / 2 , 2 \kappa _ { \operatorname* { m a x } } \xi _ { 1 - \alpha } ]$ . Since $g$ is decreasing on $[ 0 , \infty )$ , each summand $\kappa _ { i } ^ { - 1 } g ( t / \kappa _ { i } )$ of $\begin{array} { r } { \overline { { F } } _ { m } ^ { \prime } ( t ) = m ^ { - 1 } \sum _ { i } \kappa _ { i } ^ { - 1 } g ( t / \kappa _ { i } ) } \end{array}$ is at least c on that interval. Hence, for $0 < \delta \leq \delta _ { 0 }$ ，

$$
\begin{array} { r } { \overline { { F } } _ { m } ( q _ { m } ^ { \star } + \delta ) \geq 1 - \alpha + c \delta , \qquad \overline { { F } } _ { m } ( q _ { m } ^ { \star } - \delta ) \leq 1 - \alpha - c \delta . } \end{array}\tag{S16}
$$

For fixed $\alpha , j _ { \alpha } \le m$ for all suficiently large $m ,$ , so the adjoined value $+ \infty$ does not afect the $j _ { \alpha } \mathrm { t h }$ order statistic. The absolute values of the calibration scores have no ties almost surely, so $F _ { m } ( \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } ) = j _ { \alpha } / m .$ , and $1 - \alpha \leq j _ { \alpha } / m < 1 - \alpha + 2 / m$

$\mathrm { I f } \ \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } > q _ { m } ^ { \star } + \delta$ , fewer than $j _ { \alpha }$ scores lie at or below $q _ { m } ^ { \star } + \delta _ { \colon }$ , so $F _ { m } ( q _ { m } ^ { \star } + \delta ) \leq ( j _ { \alpha } - 1 ) / m <$ $1 - \alpha + 2 / m$ . By (S16), $F _ { m } ( q _ { m } ^ { \star } + \delta ) - F _ { m } ( q _ { m } ^ { \star } + \delta ) < 2 / m - c \delta \leq - c \delta / 2$ whenever $\delta \ge 4 / ( c m )$ . If $\widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } < q _ { m } ^ { \star } - \delta$ , then $F _ { m } ( q _ { m } ^ { \star } - \delta ) \ge j _ { \alpha } / m \ge 1 - \alpha .$ , so $F _ { m } ( q _ { m } ^ { \star } - \delta ) - \overline { { F } } _ { m } ( q _ { m } ^ { \star } - \delta ) \geq c \delta$ . We can bound both probabilities using Chebyshev’s inequality and Lemma S3. The first one is bounded by $( v _ { m } / 4 ) / ( c \delta / 2 ) ^ { 2 } = v _ { m } / ( c ^ { 2 } \delta ^ { 2 } )$ and the second by $v _ { m } / ( 4 c ^ { 2 } \delta ^ { 2 } )$ . Combining them, we have (S15).

Lemma S4 shows that the conformal threshold is close to $q _ { m } ^ { \star }$ with high probability. We now use this to prove Theorem 4.

Proof of Theorem $\it 4 .$ By (S4), coverage is the event $\{ | z _ { 0 } | \leq \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \}$ . Since $z _ { 0 } \sim N ( 0 , \kappa _ { 0 } ^ { 2 } )$

$$
\mathbb { P } ( | z _ { 0 } | \leq q _ { m } ^ { \star } ) = G \biggl ( \frac { q _ { m } ^ { \star } } { \kappa _ { 0 } } \biggr ) = G \biggl \{ \frac { \xi _ { 1 - \alpha } } { \kappa _ { \mathrm { r e l } } ( s _ { 0 } ) } \biggr \} .\tag{S17}
$$

On the event $\{ | \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - q _ { m } ^ { \star } | \leq \delta \}$ , the events $\{ | z _ { 0 } | \le \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \}$ and $\{ | z _ { 0 } | \leq q _ { m } ^ { \star } \}$ can difer only if $| z _ { 0 } | \in ( q _ { m } ^ { \star } - \delta , q _ { m } ^ { \star } + \delta ]$ . The density of $| z _ { 0 } |$ is at most $( 2 / \pi ) ^ { 1 / 2 } / \kappa _ { 0 , \mathrm { { m i n } } }$ under Assumption S2. Hence

$$
\left| \mathbb { P } \{ | z _ { 0 } | \leq \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } \} - \mathbb { P } ( | z _ { 0 } | \leq q _ { m } ^ { \star } ) \right| \leq \frac { 2 \delta } { \kappa _ { 0 , \operatorname* { m i n } } } \left( \frac { 2 } { \pi } \right) ^ { 1 / 2 } + \mathbb { P } \big ( | \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } - q _ { m } ^ { \star } | > \delta \big ) .\tag{S18}
$$

Take $\delta = v _ { m } ^ { 1 / 3 }$ . Since $v _ { m } \geq 1 / m$ , this $\delta$ exceeds $4 / ( c m )$ for large m, and since $v _ { m } \to 0$ it falls below $\delta _ { 0 }$ . Lemma S4 bounds the last term by $2 v _ { m } ^ { 1 / 3 } / c ^ { 2 }$ , so the right side of (S18) is $O ( v _ { m } ^ { 1 / 3 } )$ . Combining with (S17) proves (16). □

$\kappa _ { \mathrm { r e l } } ( \pmb { s } _ { 0 } )$ in Theorem 4 characterizes the expected coverage. If it exceeds one, the transformed target error is large relative to the threshold, and that can lead to undercoverage. Similarly, for a value below one, it predicts conservative coverage. In the next subsection, we turn this characterization into a prospective diagnostic.

## S5.1 Prospective implementation of the diagnostic

Here, we develop a way to estimate $\kappa _ { \mathrm { r e l } }$ from the data. The denominator of $\kappa _ { \mathrm { r e l } }$ is a function of the calibration set, and is already computed as a part of the conformal inference. The numerator denotes the scale of the transformed target error. To estimate it, we hold out a set $\mathcal { G }$ of diagnostic

observations from the target region. We do not use them for fitting, calibration or evaluation. Let $\mathcal { D } = \mathcal { F } \cup \mathcal { C }$ . We transform each $j \in \mathcal G$ separately as a target given D:

$$
e _ { j } ^ { \mathrm { { d i a g } } } = \frac { r _ { j } - \pmb { \Sigma } _ { j , D } ^ { \mathrm { w } } ( \pmb { \Sigma } _ { \mathcal { D } , \mathcal { D } } ^ { \mathrm { w } } ) ^ { - 1 } \pmb { r } _ { \mathcal { D } } } { \left\{ \pmb { \Sigma } _ { j , j } ^ { \mathrm { w } } - \pmb { \Sigma } _ { j , D } ^ { \mathrm { w } } ( \pmb { \Sigma } _ { \mathcal { D } , \mathcal { D } } ^ { \mathrm { w } } ) ^ { - 1 } \pmb { \Sigma } _ { \mathcal { D } , j } ^ { \mathrm { w } } \right\} ^ { 1 / 2 } } .\tag{S19}
$$

No diagnostic observation conditions on another, so each $e _ { j } ^ { \mathrm { d i a g } }$ uses the same information as the error at a future target. With $n _ { \mathrm { d i a g } } = | \mathcal { G } |$ , we estimate $\kappa _ { \mathrm { r e l } }$ as

$$
\widehat { \kappa _ { \mathrm { r e l } } } = \frac { \xi _ { 1 - \alpha } \left\{ n _ { \mathrm { d i a g } } ^ { - 1 } \sum _ { j \in \mathcal { G } } ( e _ { j } ^ { \mathrm { d i a g } } ) ^ { 2 } \right\} ^ { 1 / 2 } } { \widehat { q } _ { 1 - \alpha } ^ { \mathrm { W } } } , \qquad \widehat { p } _ { \mathrm { c o v } } = G \left( \frac { \xi _ { 1 - \alpha } } { \widehat { \kappa _ { \mathrm { r e l } } } } \right) .\tag{S20}
$$

Suppose, in addition to the assumptions of Theorem 4, that the diagnostic and target errors share the same scale and the correlation matrix $R _ { \mathcal { G } }$ of the diagnostic errors satisfies $\| \pmb { R } _ { \mathcal { G } } \| _ { F } ^ { 2 } / n _ { \mathrm { d i a g } } ^ { 2 }  0$ Then $\begin{array} { r } { n _ { \mathrm { d i a g } } ^ { - 1 } \sum _ { j \in \mathcal { G } } ( e _ { j } ^ { \mathrm { d i a g } } ) ^ { 2 } / \kappa ( s _ { 0 } ) ^ { 2 } \to 1 } \end{array}$ in probability, which together with Lemma S4 and Theorem 4 shows that $\widehat { p } _ { \mathrm { c o v } }$ is consistent for the coverage. If the scales vary across the region, $\widehat { \kappa _ { \mathrm { r e l } } }$ uses their root-mean-square, and $\widehat { p } _ { \mathrm { c o v } }$ is a regional approximation to the coverage averaged over the region.

## S6 Simulation designs and implementation choices

## S6.1 Data-generating mechanisms

Section 5.1 gives the general data-generating mechanism, including the mean function, the covariance family (17) and the random-sampling designs of Regime A. Below we specify the nonrandom designs of Regimes B and C. Except under the varying range design, all of them use the exponential correlation with $\phi = 2$

## S6.1.1 Regime B: predictive information in calibration

Halo k, $k \in \{ 1 0 , 2 0 , 4 0 \}$ . For each target in turn, its k nearest locations enter the calibration set. In case of overlapping neighbourhoods, we count them once, and exclude any target that falls in this neighbourhood. If this neighbourhood set exceeds 166 locations, we keep the first 166 in that order. Otherwise, we randomly choose from the remaining locations to fill it to 166.

Transect. The 330 fitting locations have $s _ { 1 } \sim \mathrm { U n i f } ( 0 , 1 )$ and $s _ { 2 }$ cycling over {0.1, 0.3, 0.5, 0.7, 0.9} with independent $N ( 0 , 0 . 0 1 2 ^ { 2 } )$ jitter, which gives 66 observations per line. The remaining 171 locations have uniform s<sub>1</sub> and $s _ { 2 }$ cycling over $\{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 \}$ with independent $N ( 0 , 0 . 0 2 ^ { 2 } )$ jitter. Five of them are targets and the rest form the calibration set.

Boundary. The 330 fitting locations are uniform on $[ 0 . 1 2 , 0 . 8 8 ] ^ { 2 }$ . The remaining 171 are uniform on the rest of $[ 0 , 1 ] ^ { 2 }$ . Five of them are targets and the rest form the calibration set.

Monitoring network. The 330 fitting locations are split equally among discs of radius 0.07 centred at (0.22, 0.22), (0.78, 0.25) and (0.5, 0.8). The remaining 171 locations come from a uniform distribution on $[ 0 , 1 ] ^ { 2 }$ , and maintain at least 0.18 distance from every centre. The first five of these are targets and the rest form the calibration set.

## S6.1.2 Regime C: calibration–target comparability

Locations are uniform on $[ 0 , 1 ] ^ { 2 }$ under quadrant, clustered calibration, block holdout and varying range.

Disjoint halves. The 330 fitting locations are uniform on $[ 0 , 0 . 4 5 ] \times [ 0 , 1 ]$ . The remaining 171 are uniform on $[ 0 . 5 5 , 1 ] \times [ 0 , 1 ]$ , and five of them are targets.

Quadrant. The five targets are drawn from $\{ s _ { 1 } > 0 . 5 , s _ { 2 } > 0 . 5 \}$ . The other locations in this quadrant enter calibration, which is filled to 166 from outside the quadrant.

Clustered calibration. The calibration set contains the 166 locations nearest (0.3, 0.3). The five targets are drawn from the 120 locations farthest from that point.

Distant patch. The 166 calibration locations are uniform on the disc of radius 0.10 centred at (0.13, 0.13). The other 335 are uniform on $[ 0 . 4 2 , 1 ] ^ { 2 }$ , and five of them are targets.

Block holdout. The five targets are drawn from $\{ s _ { 1 } > 0 . 5 , s _ { 2 } > 0 . 5 \}$ , and all other locations in this quadrant are removed. Of the locations outside it, 166 chosen at random form the calibration set and the rest, about 210, form the fitting set.

Varying range. The five targets are drawn from $\{ s _ { 1 } \geq 0 . 5 \}$ , and the rest are split at random. The residual process consists of two independent exponential fields on either side of $s _ { 1 } = 0 . 5$ , with $\phi = 1$ on the left and $\phi = 1 0$ on the right.

## S6.2 Implementation choices and sensitivity

## S6.2.1 Software and covariance estimation

All methods fit the mean with ranger using 500 trees and default settings. For whitening, the parameters of the exponential working covariance are estimated with BRISC\_estimation, using an intercept-only mean and the package defaults, which use 15 nearest neighbours and locations ordered by the sum of their coordinates. GSCP and LSCP use the covariance-estimation routine supplied with scp.

## S6.2.2 Choice of LSCP bandwidth

LSCP uses a kernel with bandwidth h to weight each calibration observation based on its distance to the target. If h is too small, the calibration observations carry too little weight for the weighted quantile to be finite, and the interval is unbounded. If h is too large, LSCP behaves like GSCP. Mao et al. (2024) navigate this tradeof by choosing h to minimise the average interval score over validation observations. In the split version only the calibration observations carry weight, so we validate on calibration observations and weight each one by the others. Their candidate grid gives no bounded interval at our sampling density, so we extend it to half the largest distance between calibration observations.

However, in Regime C the selected h can be far too small. Since h is selected using the calibration observations alone, the problem worsens the farther the calibration observations lie from the targets. Table S1 shows that under block holdout, the selected h produces unbounded intervals at 8% of the targets. This rises to 76% under clustered calibration and to every target under distant patch. An unbounded interval says nothing about the target, and counting it as covered inflates the coverage, to 0.986 under clustered calibration. To bypass this problem, we use a fixed $h = 0 . 4 0$ throughout the simulation. This produces similar results to the selected h on Regimes A and B in terms of coverage and width. In Regime C, this choice of h produces bounded intervals in all but the distant patch design, where it is unbounded at only 25% of the targets. We can in theory reduce this by choosing an even larger bandwidth, but a larger h increasingly moves LSCP toward GSCP on every other design, so we keep $h = 0 . 4 0$

Table S1: LSCP with fixed bandwidth h = 0.40 and selected bandwidth. Coverage/mean width are computed at nominal 0.90. Unbounded intervals count as covering their target and are excluded from the mean width. h = 0.40 is unbounded only under distant patch, at 24.8% of targets, where its entry is computed on the bounded intervals.
<table><tr><td colspan="5">Design Fixed h = 0.40 Selected h Selected unbounded (%) Median selected h</td></tr><tr><td>Exponential, φ = 0.5</td><td>.909/1.80</td><td>.908/1.80</td><td>0.0</td><td>0.50</td></tr><tr><td>Exponential, φ = 1</td><td>.918/1.90</td><td>.915/1.90</td><td>0.0</td><td>0.52</td></tr><tr><td>Exponential, φ = 2</td><td>.922/2.07</td><td>.922/2.08</td><td>0.0</td><td>0.50</td></tr><tr><td>Exponential, φ = 6</td><td>.911/2.50</td><td>.908/2.50</td><td>0.0</td><td>0.48</td></tr><tr><td>Exponential, φ = 8</td><td>.910/2.62</td><td>.914/2.64</td><td>0.0</td><td>0.48</td></tr><tr><td>Exponential, φ = 30</td><td>.911/3.47</td><td>.910/3.47</td><td>0.0</td><td>0.50</td></tr><tr><td>Matérn, ν = 3/2</td><td>.919/2.08</td><td>.922/2.09</td><td>0.0</td><td>0.49</td></tr><tr><td>Matérn, ν = 5/2</td><td>.909/1.99</td><td>.904/1.98</td><td>0.0</td><td>0.51</td></tr><tr><td>Gaussian</td><td>.920/2.03</td><td>.917/2.03</td><td>0.0</td><td>0.52</td></tr><tr><td>Halo 10</td><td>.907/2.30</td><td>.907/2.30</td><td>0.0</td><td>0.50</td></tr><tr><td>Halo 20</td><td>.910/2.41</td><td>.913/2.40</td><td>0.0</td><td>0.44</td></tr><tr><td>Halo 40</td><td>.916/2.62</td><td>.925/2.64</td><td>0.0</td><td>0.34</td></tr><tr><td>Transect</td><td>.912/2.30</td><td>.917/2.32</td><td>0.0</td><td>0.36</td></tr><tr><td>Boundary</td><td>.910/2.57</td><td>.914/2.60</td><td>0.0</td><td>0.38</td></tr><tr><td>Monitoring network</td><td>.917/3.14</td><td>.921/3.14</td><td>0.0</td><td>0.28</td></tr><tr><td>Disjoint halves</td><td>.910/3.24</td><td>.928/3.28</td><td>0.0</td><td>0.18</td></tr><tr><td>Quadrant</td><td>.902/2.79</td><td>.902/2.79</td><td>0.0</td><td>0.38</td></tr><tr><td>Clustered calibration</td><td>.923/2.18</td><td>.986/2.46</td><td>75.8</td><td>0.17</td></tr><tr><td>Distant patch</td><td>.876/2.14</td><td>unbounded</td><td>100.0</td><td>0.04</td></tr><tr><td>Block holdout</td><td>.880/2.89</td><td>.899/2.91</td><td>7.9</td><td>0.44</td></tr><tr><td>Varying range</td><td>.902/2.59</td><td>.906/2.64</td><td>0.0</td><td>0.36</td></tr></table>

## S6.2.3 Factorization and ordering

Under exact whitening, the coverage and the distribution of the interval width do not depend on the ordering of the calibration locations or the factor used for whitening. In practice the covariance is estimated and the factor may be approximate, as with NNGP, so the interval may depend on both. The main results use the order supplied by the split, which does not depend on the responses. We compare it with a fresh random permutation and with two structured orderings, by coordinate sum and by maximin distance. We also replace the dense Cholesky factor with the NNGP factor with K = 25 under the split order. Table S2 shows that all five configurations give nearly identical coverage and width.

## S6.2.4 Departure from Gaussianity

We replace the Gaussian field by the multivariate t field $r = W ^ { 1 / 2 } z$ , where z is a Gaussian field with covariance (17) and $W \sim \mathrm { I G } ( \nu _ { t } / 2 , \nu _ { t } / 2 )$ is independent of z. We take $\nu _ { t } \in \{ 5 , 1 0 \}$ , with the

Table S2: Efect of factorization and ordering in whitening. Coverage and mean width are computed at nominal 0.90, and averaged over the nine Regime A designs.
<table><tr><td>Factor</td><td>Ordering</td><td>Coverage</td><td>Mean width</td></tr><tr><td>Dense Cholesky</td><td>as supplied by the split</td><td>0.904</td><td>2.113</td></tr><tr><td>Dense Cholesky</td><td>coordinate sum</td><td>0.906</td><td>2.115</td></tr><tr><td>Dense Cholesky</td><td>maximin</td><td>0.904</td><td>2.113</td></tr><tr><td>Dense Cholesky</td><td>random</td><td>0.905</td><td>2.113</td></tr><tr><td>NNGP, K = 25</td><td>as supplied by the split</td><td>0.905</td><td>2.116</td></tr></table>

Table S3: Coverage/mean width at nominal coverage 0.90 under a spatial t process.
<table><tr><td>Correlation</td><td> $\nu _ { t }$ </td><td>Naive</td><td>GSCP</td><td>LSCP</td><td>Whitened</td></tr><tr><td>Exponential,  $\phi = 2$ </td><td>10</td><td>.899/3.27</td><td>.903/2.13</td><td>.909/2.19</td><td>.903/2.04</td></tr><tr><td>Exponential,  $\phi = 2$ </td><td>5</td><td>.906/3.45</td><td>.906/2.26</td><td>.915/2.34</td><td>.912/2.18</td></tr><tr><td>Gaussian,  $\phi = 6$ </td><td>10</td><td>.906/3.95</td><td>.901/2.10</td><td>.909/2.18</td><td>.904/2.02</td></tr><tr><td>Gaussian,  $\phi = 6$ </td><td>5</td><td>.902/4.17</td><td>.909/2.20</td><td>.914/2.26</td><td>.906/2.12</td></tr></table>

exponential correlation at $\phi = 2$ and the Gaussian correlation at $\phi = 6 .$ . Table S3 shows that all methods retain near-nominal coverage and that whitening gives the narrowest interval in every setting.

## S7 Fine particulate matter monitoring network data processing

Data. We use 2022 annual mean $\mathrm { P M } _ { 2 . 5 }$ concentrations from the Environmental Protection Agency’s Air Quality System. We only consider monitors with at least 250 daily observations and keep one monitor per site. We keep monitors in the contiguous United States, together with the three monitors in Mexico that the agency reports under state code 80.

Covariates. The agency’s site listing provides elevation, land use and location setting for each monitor. Land use is a categorical variable that records the main use of the land around the monitor, such as commercial, residential, industrial, desert, forest or mobile. Location setting records whether the monitor is in an urban, suburban or rural area. We also include wind speed from TerraClimate (2022 annual mean) and population density from WorldPop (2020), both extracted at the monitor coordinates.

Coordinates. We cannot directly use the Euclidean distance between raw latitude and longitude, since the Earth’s curvature makes it a distorted measure of distance on the ground. We therefore project the coordinates using the Albers equal-area conic projection for the contiguous United States. This provides planar coordinates in kilometres with minimal domain-wide distance distortion. We divide both coordinates by their largest range, 4547 km, so that the monitors lie in the unit square. This puts the coordinates on the same scale as the simulations, and LSCP uses the same bandwidth $h = 0 . 4 0$