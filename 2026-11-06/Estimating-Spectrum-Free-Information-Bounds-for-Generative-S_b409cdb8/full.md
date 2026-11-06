# Estimating Spectrum-Free Information Bounds for Generative Surrogates Beyond Conditional Collapse

Chaitanya Dumbre School of Computer Science and Engineering MIT World Peace University, Pune chaidumb04@gmail.com

Sourabh Dube Department of Physics Indian Institute of Science Education and Research (IISER), Pune sdube@iiserpune.ac.in

## Abstract

Conditional generative surrogates in high-energy physics, such as models that predict reconstructed jets from partons, are often evaluated on 1D marginal divergences among other metrics. Taken alone these metrics can reward conditional collapse. On a CMS simulation benchmark, a sampler that ignores the conditioner beats a trained Mixture Density Network (MDN) on all four target marginals under both KL and Wasserstein-1, and cannot be stably separated from a Conditional Flow Matching (CFM) model on two of the four. We instead score the models with the Continuous Ranked Probability Skill Score (CRPSS) from forecast verification, taking the exact marginal as the reference. We show how global skill depends on the evaluation spectrum, and we also report it within bins of a conditioning variable, where that dependence largely goes away. We then introduce a data-driven k-Nearest-Neighbor (k-NN) extrapolation that estimates the best CRPS the condi tioning information allows. Both models agree with this estimate on all four targets, which suggests that current architectures have reached the information available in the conditioner, at least as far as these per-target scores can tell.

## 1 Introduction

Conditional generative models are increasingly explored as fast replacements for expensive scientific simulators such as GEANT4 [1] and multi-leg Monte Carlo generators [2], within the broader framework of simulation-based inference [3–9]. The same conditioning input c can lead to many different outcomes y, so such a model has to learn a conditional distribution p(y | c) [10]. Optimizing for mean squared error [11] instead collapses predictions to the conditional mean and compresses the generated marginal spectrum [12]. A common evaluation asks how well the marginal distribution of each output is reproduced, using histogram separation power, the Wasserstein-1 distance [13] or the Kullback–Leibler divergence [14]. Such comparisons appear throughout community benchmark suites [15, 16] and surrogate studies [17–19], usually alongside classifier two-sample tests [20, 17, 19] and Fréchet and kernel physics distances [21, 15]. Taken alone, a marginal comparison cannot show that a model uses c at all, and a model that ignores c is said to be conditionally collapsed.

We quantify this on the task of predicting reconstructed jet kinematics from parton kinematics in a CMS [22] QCD simulation dataset [23]. A condition-blind sampler, which reproduces the marginal by construction, beats a trained Mixture Density Network (MDN) [24] on 8 of 8 marginal-divergence comparisons across three data splits, by margins of 3.65–22.39×. A Conditional Flow Matching (CFM) surrogate [25] matches the marginals more closely, but the sampler still beats it at every split on two of the four targets, and on the other two the winner changes from split to split.

This work builds on Baattrup et al. [12], who show that pointwise metrics such as the root-meansquare error reward collapse onto the conditional mean in multimodal inverse problems. They propose evaluating such models with a per-event Continuous Ranked Probability Score (CRPS), a population-level test of spectrum fidelity and a coverage-based calibration check. Their setting is an inverse problem, but the argument holds for any model of $p ( y$ | c) learned from simulated pairs, including a forward surrogate like ours. Marginal divergences fail in the opposite way, because they reward collapse onto the marginal. A condition-blind sampler passes a spectrum test by construction, and its prediction sets also have the right coverage when averaged over all events. In that protocol, only the per-event score and calibration checked within bins of phase space can tell it apart from a model that uses c. On its own, however, a CRPS value is hard to read. We give it a scale by taking the exact marginal as the zero of skill and a data-driven estimate of the best achievable score as the ceiling, and we show how the resulting skill depends on the evaluation spectrum.

This paper makes the following points.

• The metric trap. A condition-blind sampler beats a trained MDN on all four target marginals under both KL and $W _ { 1 } .$ , by up to 22.4×, and cannot be stably separated from a CFM on two of four targets (Section 6.1).

• A reference from forecast verification. We carry over the CRPS skill score with the exact marginal reference, which is the uncertainty term of the Hersbach decomposition [26], and use it to fix the zero of the skill scale (Section 4).

• Spectrum dependence. It is known that a climatological reference depends on how it is stratified [27, 26]. We show what this does to global skill on our jet benchmark, and we report a differential version in bins of parton p alongside it (Sections 4 and 6.2).

• Data-driven limits. A model-free k-NN extrapolation estimates the best achievable score, and both surrogates agree with it (Sections 5 and 6.2).

• Likelihood and CRPS differ. Past the best epoch the likelihood degrades while the CRPS barely moves, so a flat CRPS alone does not rule out overfitting (Section 6.3).

## 2 Related Work

The weakness of marginal metrics is well known. In high-energy physics, Kansal et al. [21] point out that 1D binned projections lose sensitivity to correlations and conditional distributions, and they call for multivariate tests that include the conditioning features when a model is conditional. In image generation the same worry led to intra-class FID [28], the Fréchet Joint Distance [29], conditional FID [30] and the Classification Accuracy Score [31]. Classifier two-sample tests [32] can detect collapse when the conditioning variable is given to the classifier, as in several calorimeter studies [20, 16, 19]. Comparing generated quantities against the conditioning variable exposes it directly [17], and the CaloChallenge also compares histograms of the total deposited energy divided by the incident energy, which probe the calorimeter response [16]. Our momentum target is a ratio of the same kind, and the sampler still matches it, because a histogram pooled over all events tests the response averaged over c and says little about how it changes with c. Fréchet and kernel physics distances computed on pooled samples do not see which event a prediction was conditioned on, although they can be extended to conditional generation [21]. Simulation-based inference checks conditional calibration with rank and coverage tests [33, 34].

Forecast verification has used proper scoring rules [35] and skill scores against a climatological reference [27, 36] for decades, and most of our reference construction comes from that literature. Murphy [27] separates single-valued references from multiple-valued ones built on subsamples of the data, and shows that forecasts generally look less skillful against the latter. Hersbach [26] decomposes the CRPS into reliability, resolution and uncertainty, gives the sorted formula for the uncertainty, and notes that it falls when the climatology is stratified by location and season. Using the outcomes of similar past cases as a predictive ensemble is also old in weather forecasting, where it is known as an analog ensemble [37]. Our ceiling applies the same idea to the training set. The nearest training events of each test event serve as its ensemble, and extrapolating their score to zero neighbourhood radius estimates the best CRPS the conditioner allows, in the spirit of the nearest-neighbour bounds on the Bayes error of Cover and Hart [38].

Baattrup et al. [12] recently proposed the CRPS as the per-event part of an evaluation protocol for inverse problems in particle physics, alongside spectrum fidelity and coverage-based calibration, and showed on dileptonic top-quark reconstruction that model rankings reverse between pointwise and distributional metrics. Our floor and ceiling give that per-event score a reference scale.

## 3 Task and Data

The models are trained to predict the kinematic 4-vectors of reconstructed jets from the 4-vectors of the underlying partons. We use a public CMS QCD dataset [23]. Its events were generated with PYTHIA 8 [39] using the CMS CP5 tune [40], passed through the full GEANT4 [1] detector simulation and CMSSW reconstruction chain, and distributed as NanoAOD. The sample has a flat transverse momentum $\hat { p } _ { \mathrm { T } }$ spectrum over a broad kinematic range. We keep reconstructed jets that satisfy $p _ { \mathrm { T } } > 5 0$ GeV and with pseudorapidity $( \eta )$ satisfying $| \eta | < 2 . 4$ that pass the jet identification requirement $( \mathsf { j e t I d } \geq 2 )$ . The jets are also required to satisfy pileup jet identification $\mathrm { ( p u I d \geq 1 ) }$ On the parton side we take the two leading outgoing partons of the hard process and restrict their p<sub>T</sub> to the fiducial window [50, 1000) GeV. A parton and a jet form a pair when each is the other’s nearest neighbour in $\Delta R ^ { 1 }$ and $\Delta R < 0 . 2$ . After the filters, about 10.8 million parton-jet pairs survive. We call each pair an event in what follows, and the data splits keep all pairs from one collision event together.

## 3.1 The Inputs (The Conditioner c)

Partons are close to massless at this kinematic scale $( p _ { \mathrm { T } } \ge 5 0 \mathrm { G e V } )$ , so the conditioning state is fixed by a spatial 3-vector and a discrete flavor. The models receive a 26-dimensional conditioning vector made of four blocks.

• Standardized kinematics (3 dims). The parton’s transverse momentum $( \log p _ { \mathrm { T } } )$ , absolute pseudorapidity (|η|), and log(cosh η), all z-score standardized.

• Pileup (1 dim). The number of true interactions (nTrueInt), z-score standardized. All results reported here use the recorded nTrueInt of each test event. When jets are generated from parton inputs alone, the value is instead sampled from the empirical pileup profile of the training set.

• Flavor (6 dims). One-hot vectors for the parton’s identity $( d , u , s , c , b , g )$

• Fourier-expanded azimuth (16 dims). The periodic angle $\phi$ is expanded into 16 Fourier features, sin $( k \phi )$ and $\cos ( k \phi )$ for $k \in \{ 1 , \ldots , 8 \}$

## 3.2 The Outputs (The Targets y)

The models predict a 4-dimensional target $y$ that describes the residual between the parton and the reconstructed jet. It is built from the following transformations.

$$
\mathbf { M o m e n t u m \ r e s p o n s e { : \quad } } y _ { 0 } = \log ( p _ { \mathrm { T } } ^ { \mathrm { j e t } } / p _ { \mathrm { T } } ^ { \mathrm { p a r t o n } } )\tag{1}
$$

$$
\mathrm { P a r i t y – f o l d e d \ p s e u d o r a p i d i t y : } \quad y _ { 1 } = \mathrm { s i g n } ( \eta ^ { \mathrm { p a r t o n } } ) \times ( \eta ^ { \mathrm { j e t } } - \eta ^ { \mathrm { p a r t o n } } )\tag{2}
$$

$$
\mathrm { A z i m u t h a l \ r e s i d u a l : } \quad y _ { 2 } = \mathrm { w r a p } ( \phi ^ { \mathrm { j e t } } - \phi ^ { \mathrm { p a r t o n } } ) \in [ - \pi , \pi ]\tag{3}
$$

$$
\mathrm { J e t ~ m a s s ~ f r a c t i o n } ; \quad y _ { 3 } = \log ( m _ { \mathrm { j e t } } / p _ { \mathrm { T } } ^ { \mathrm { j e t } } )\tag{4}
$$

At inference the generated residuals are inverted back to $( p _ { \mathrm { T } } , \eta , \phi , m )$ , and a disk clamp keeps the generated jet inside the $\Delta R < 0 . 2$ matching cone of the source parton. The network does not predict the jet energy. It is reconstructed from the predicted kinematics via $E = \sqrt { p _ { x } ^ { 2 } + p _ { y } ^ { 2 } + p _ { z } ^ { 2 } + m ^ { 2 } }$

## 4 The Bounded CRPS Framework

A marginal comparison pools all events, so it cannot check a prediction against the event it was conditioned on. We therefore score each test event on its own with the CRPS. For a predictive distribution $F$ and an observed value $y$ it is

$$
\mathrm { C R P S } ( F , y ) = \int \big ( F ( x ) - \mathbf { 1 } \{ y \leq x \} \big ) ^ { 2 } \mathrm { d } x = \mathbb { E } _ { F } | X - y | - \frac { 1 } { 2 } \mathbb { E } _ { F } | X - X ^ { \prime } | ,\tag{5}
$$

where $X , X ^ { \prime } \sim F$ are independent and lower is better. The integral is, up to sign, the uniformly weighted quadratic rule of Matheson and Winkler [41], and the expectation form is given by Gneiting and Raftery [35]. The CRPS is strictly proper for distributions with a finite first moment [35], so it penalizes both a wrong location and an over-confident spread. Its absolute value is hard to read without a reference, so we use two. The floor is the mean CRPS of the unconditional marginal. This is the best prediction that ignores $c ,$ and it marks zero skill. The ceiling is the best mean CRPS achievable from c (Section $\bar { 5 } )$ . Because lower CRPS is better, the ceiling lies numerically below the floor on a CRPS axis and above it on a skill axis. A model can score worse than the floor, so skill can be negative.

For the marginal of the N test events the expectations in Equation $5$ can be evaluated exactly. With the test targets sorted as $y _ { ( 1 ) } \le \cdots \le y _ { ( N ) }$ , the all-pairs sum reduces to

$$
\mathrm { C R P S } _ { \mathrm { f l o o r } } = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } ( 2 i - N - 1 ) y _ { ( i ) } .\tag{6}
$$

This costs $O ( N \log N )$ and equals half the Gini mean difference [42] of the empirical distribution of the test targets, with all $N ^ { 2 ^ { \prime } }$ pairs counted. It is also exactly the uncertainty term $U$ in the CRPS decomposition of Hersbach [26], whose Eq. (20) is the same sum written in terms of spacings, so this reference is long established. We use it in place of a Monte Carlo estimate because a plug-in score computed from m marginal draws per event would inflate the floor by a factor $( m \bar { + } 1 ) \bar { / } m$ [36]. Normalizing a model’s mean CRPS, $\mathrm { C R P S } _ { \mathrm { c o n d } }$ , by the floor gives the usual skill-score form $1 - A / A _ { \mathrm { r e f } }$ for an error measure whose perfect score is zero [27], with the sample marginal as reference. Forecast verification calls this a climatological reference [27, 36], and the resulting quantity, $S = 1 - \mathrm { C R P S } _ { \mathrm { c o n d } } / \mathrm { C R P S } _ { \mathrm { f l o o r } }$ , is the Continuous Ranked Probability Skill Score (CRPSS). The Hersbach decomposition writes the CRPS as reliability minus resolution plus uncertainty, which gives $S \approx ( \mathrm { R e s o l - \dot { R e l i } } ) / U$ . For a calibrated model $s$ is therefore the fraction of the floor resolved by $c ,$ and miscalibration lowers it. The relation is approximate here only because we score samples with the fair estimator of Section 5.

Dependence on the evaluation spectrum. We use spectrum for the distribution $p ( c )$ of conditioning states in the test set. Global S depends on it, because the reference does. Hersbach [26] makes the point for weather, where a sample that pools Norway in January with Spain in March has a very broad climatology and a large uncertainty term. To see the effect on skill, consider a homoscedastic location family $y = \mu ( c ) + \epsilon$ . The conditional CRPS is set by the noise $\epsilon ,$ while the marginal floor also contains the spread of $\mu ( c ) . \operatorname { A s } p ( c )$ widens the floor grows without limit and global skill tends to 1, although the noise has not changed. That skill is real, since knowing c does help, but its size says as much about the width of the test set as about the model. We therefore also recompute both the CRPS and the floor within narrow bins of $c .$ This is the stratified climatology that Hersbach [26] describes for location and season, with bins of c in their place, and it is the distributional counterpart of the multiple-valued reference of Murphy [27]. Each bin’s reference is then a sampler that knows the bin, so differential skill measures what a model adds beyond the binning variable and no longer depends on how widely that variable is spread. Here we bin in parton $p _ { \mathrm { T } }$ , and the other components of c stay averaged over. Appendix F shows how the result changes with the number of bins and with the binning variable.

## 5 Estimating the Limit with k-Nearest-Neighbor Extrapolation

To estimate the ceiling directly from data, we use a differential k-nearest-neighbor (k-NN) [38] extrapolation with four steps. (1) Standardize the continuous conditioning variables and match the discrete flavour exactly. (2) For each query $c _ { i }$ , find its k nearest training neighbours for $k \in$ $\{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ and take their true targets as an empirical conditional ensemble. (3) Compute the fair, ensemble-size adjusted CRPS with an $m ( m - 1 )$ kernel [36, 43], with ensemble size $m = k ,$ along with the mean neighbourhood radius $r _ { k } . ~ ( 4 )$ Fit $\left( r _ { k } , \mathbf { C R P S } _ { k } \right)$ across all k by Ordinary Least Squares [44] and take the $r  0$ intercept. The pseudocode is in Appendix G.

![](images/52b3e445f82f978c42cdca5ea43a3e9b40f2a2b9ea996109d0cb1c79093c8c5c.jpg)  
Figure 1: Data-driven limits (Seed 1). Columns are the four targets. Top. k-NN CRPS against mean neighborhood radius $\bar { r } _ { k }$ for $k \in \{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ , with the OLS fit and its $r  0$ intercept (⋆). Bottom. The differential ceiling over 14 log-spaced parton-p bins, with every series (MDN, CFM, k-NN ceiling, lookup table and Marginal sampler) scored on the same 200,000 test events and read against the floor recomputed inside its own bin. Top-edge triangles mark bins where the Marginal sampler is off scale above. Averaged over the spectrum, both surrogates agree with the k-NN estimate to within $\pm 0 . 0 0 5$ (Section 6.2). In single bins the gaps are larger, up to about 0.016.

What the estimate guarantees. The finite-k scores have a clean interpretation. Given the distance to the $( k + 1 )$ -th neighbour, the k retrieved targets are independent draws from the neighbourhood distribution ${ \bar { F } } ,$ , the conditional distribution averaged over that ball. The fair kernel is therefore unbiased for the CRPS of $\bar { F } \ [ 4 3 ]$ . Since the CRPS is proper, $\bar { F }$ cannot beat the true conditional distribution $F$ in expectation, and the excess is $\begin{array} { r } { \int ( \bar { F } - \bar { F ^ { } } ) ^ { 2 } \mathrm { d } x \geq 0 [ 4 1 } \end{array}$ , 35]. So in expectation every finite-k score sits above the Bayes-optimal score for this conditioner. This needs the reference and query events to come from the same distribution, which holds here because both are random splits of one simulated sample.

No such guarantee holds for the $r  0$ intercept, which aims at the Bayes-optimal score itself. If the CRPS really rises with radius and the curve is convex, a straight-line fit lands below the Bayes-optimal score at $r = 0 ,$ , so the intercept errs on the low side. That needs both assumptions and is not a theorem. On this benchmark the curve is close to flat, so the intercept simply scatters around the Bayes-optimal score and a model at the limit can land on either side of it. The k-NN CRPS barely moves across the ladder (Figure 1, top row), the slopes in Table 1 are consistent with zero across splits, and a slope can change sign when the reference subset is redrawn (Appendix F). Neighbourhood blur is small over the whole ladder, so the answer hardly depends on k and the extrapolation changes it very little. We therefore treat the intercept as an estimate of the ceiling and as a consistency check on how much skill the conditioner supports, and we do not claim it as a strict limit.

Table 1: Extrapolated k-NN slopes, CRPSS limits, and metric trap inversions. Every entry is the mean ± standard deviation over three random data splits (seeds 1–3). The KL columns give the marginal KL divergence of each surrogate divided by that of the condition-blind Marginal sampler on the full test split, so a ratio above $\bar { 1 } . 0 \times$ means the surrogate is worse. dCRPS/dr is the OLS slope of the $k { \mathrm { - N N } }$ CRPS on neighbourhood radius, in target units per unit of standardized radius, and values consistent with zero mean the estimate barely depends on k. $S _ { \mathrm { g l o b a l } }$ and $S _ { \mathrm { d i f f } }$ are the skill of the $r \to 0$ ceiling against the global floor and against the floor recomputed within each of 14 parton-p bins (median over bins). LUT Rec. is $S _ { \mathrm { { L U T } } } { \bar { / } } S _ { \mathrm { { C F M } } }$ , the lookup-table skill as a fraction of the CFM skill. CRPS quantities use the first 200,000 test events of each split with $m = 2 0 0$ samples.
<table><tr><td>Target</td><td>KL MDN / Marginal</td><td>KL CFM / Marginal</td><td> $\mathrm { d C R P S } / \mathrm { d } r$   $( \times 1 0 ^ { - 3 } )$ </td><td> $S _ { \mathrm { g l o b a l } }$ </td><td> $S _ { \mathrm { d i f f } }$ </td><td>LUT Rec.</td></tr><tr><td> $\log p _ { \mathrm { T } }$  response</td><td> $5 . 5 \pm 1 . 7 \times$ </td><td> $0 . 9 0 \pm 0 . 2 1 \times$ </td><td>+0.51 ± 0.64</td><td> $3 . 5 \pm 0 . 1 \%$ </td><td> $2 . 3 \pm 0 . 1 \%$ </td><td> $9 0 . 8 \pm 0 . 5 \%$ </td></tr><tr><td>log m/pT</td><td> $4 . 3 \pm 0 . 6 \times$ </td><td> $1 . 1 1 \pm 0 . 1 9 \times$ </td><td>+0.12 ± 2.95</td><td> $2 8 . 7 \pm 0 . 1 \%$ </td><td> $1 0 . 1 \pm 0 . 4 \%$ </td><td> $9 8 . 8 \pm 0 . 1 \%$ </td></tr><tr><td> $\Delta \eta$ </td><td> $1 6 . 7 \pm 1 . 9 \times$ </td><td> $4 . 1 \pm 0 . 3 \times$ </td><td> $+ 0 . 5 7 \pm 0 . 6 9$ </td><td> $2 . 2 \pm 0 . 3 \%$ </td><td> $1 . 5 \pm 0 . 2 \%$ </td><td> $6 0 . 8 \pm 1 . 4 \%$ </td></tr><tr><td> $\Delta \phi$ </td><td> $1 9 . 0 \pm 2 . 9 \times$ </td><td> $6 . 4 \pm 0 . 7 \times$ </td><td> $+ 0 . 1 9 \pm 0 . 2 3$ </td><td> $1 . 3 \pm 0 . 1 \%$ </td><td> $0 . 4 \pm 0 . 2 \%$ </td><td> $8 4 . 6 \pm 1 . 1 \%$ </td></tr></table>

## 6 Empirical Audits

## 6.1 The Metric Trap

The condition-blind Marginal sampler defeats the correctly conditioned MDN on 8 of 8 targetdivergence combinations at each of three random data splits, by margins of 3.65–22.39× under KL and 3.75–18.02× under $W _ { 1 }$ . Table 1 gives the KL half per target as a mean over splits, while the ranges here are per-split extremes. The Marginal sampler’s KL is at the histogram sampling-noise level (about $2 \times 1 0 ^ { - 4 }$ , against $1 { - } 4 \times 1 0 ^ { - 3 }$ for the MDN), so the large ratios partly reflect how small the sampler’s KL is. The CFM is closer. On momentum response and jet mass it changes sides with the sampler between splits, while on $\Delta \eta$ and $\Delta \phi$ the sampler wins at every split. The MDN reproduces the marginals worse than the CFM on every target and split, under both divergences, yet the two are almost indistinguishable on the per-event CRPS, where their spectrum-averaged ratios differ by at most 0.0007 (Section 6.2). On this benchmark a mixture head therefore matches a flow event by event but not on the marginals, which bears on the open question raised by Baattrup et al. [12] of whether mixture heads can match flows in spectrum fidelity on realistic problems.

## 6.2 Target-Dependent Saturation

Measured against the k-NN ceiling (Figure 1), both surrogates agree with the estimated limit. On every target and data split, the spectrum-averaged CRPS ratios of both learned surrogates sit within ±0.005 of the global k-NN intercept ratio, scored on the same events, and they fall on both sides of it. This holds when the reference subset is redrawn and when the full training set is used as reference (Appendix F).

To test whether that limit reflects model capacity or the conditioning state itself, we replace the networks with a parameter-free lookup table. The conditioner is reduced to four binned axes, 40 bins in log $p _ { \mathrm { T } } .$ 12 in $| \eta | .$ 6 in nTrueInt, and the 6 exact-match flavour strata, giving $4 0 \times 1 2 \times 6 \times 6 = 1 7 { , } 2 8 0$ cells, and each query is answered by resampling the training targets stored in its cell. Bin edges are training-set quantiles, so each continuous axis is equally populated. We do not bin log(cosh η) because it is a deterministic function of |η|. Cells holding fewer than 20 training rows back off to the flavour marginal, and then to the global marginal. Averaged over the three data splits, the table recovers 98.8% of the CFM’s spectrum-averaged conditional skill on jet mass and 84.6% on $\Delta \phi .$ , though only 60.8% on $\Delta \eta .$ , with no gradients and no learned parameters. Capacity is therefore unlikely to be the binding constraint, except possibly on $\Delta \eta$

The gap between $S _ { \mathrm { g l o b a l } }$ and $S _ { \mathrm { d i f f } }$ in Table 1 shows why we also recompute the floor locally. If the $r \to 0 \ k { \mathrm { - } } \mathrm { N N }$ ceiling is scored first against the global floor and then against the floor recomputed inside each $p _ { \mathrm { T } }$ bin, its skill on jet mass falls from 28.7% to 10.1%, a relative drop of 64.7% (mean over splits, per-split range 63.6–66.1%). The difference is real information, since jet mass fraction changes strongly with $p _ { \mathrm { T } }$ across the evaluated range, but its size depends on the width of the spectrum.

![](images/58d3e5e9b0bc0d65b2346673eeb71d416e206bb77edf9a9ac2f2b2a84bcc2e9f.jpg)  
Figure 2: Likelihood and CRPS past the best epoch (Seed 1, MDN). Validation NLL (left axis) and validation CRPS averaged over the four targets (right axis). The rule marks epoch 38, the checkpointed best-validation-loss epoch. Past it, the NLL worsens by 1.53% and the CRPS by only 0.18%. The two axes are scaled independently, so the CRPS rise looks larger than it is. The CFM is omitted because it does not overfit.

## 6.3 Likelihood and CRPS Near the Limit

The two scores respond differently to overfitting (Figure 2). We anchor at the best validation Negative Log-Likelihood (NLL), which for seed 1 falls at epoch 38. Training the MDN for 62 further epochs worsens validation NLL by 0.046 nats (1.53%). Validation CRPS, tracked on a fixed 20,000-event subset of the same split with $m = 2 0$ samples, worsens by only 0.18% over the same epochs, and seeds 2 and 3 show the same pattern. The NLL is itself a strictly proper scoring rule [35], so neither score has failed here and the two simply weight errors differently. The percentages cannot be compared directly, because the NLL has no natural zero and because it scores the joint 4D density while the CRPS here is a mean of 1D scores. In practice a flat CRPS does not rule out overfitting, and we would monitor it alongside the likelihood or a calibration test. Baattrup et al. [12] find the same blind spot on top-quark reconstruction, where coverage separates two flows with comparable CRPS.

## 7 Limitations

The ceiling is an estimate and does not prove saturation. It is computed from the continuous conditioner and flavour, and Appendix F shows that the ±0.005 agreement holds when the reference subset is redrawn and when the full training set is used. The surrogates also receive $\phi ,$ which the neighbourhood search does not, and this explains the one consistent gap we see. On momentum response they sit about 0.0014 below the k-NN score averaged over $k \geq 2 0$ at every split, and adding ϕ to the search closes that gap. Their offset from the extrapolated intercept is noisier and changes sign between splits. On the angular targets ±0.005 is a sizeable fraction of the available skill, so saturation there is only weakly constrained. The OLS standard errors in Algorithm 1 treat the five ladder points as independent, which they are not, since they share query events and nested neighbourhoods. They should be read as indicative only, and the redraw scatter in Appendix F is a better guide.

All scores are per-target 1D scores and do not test correlations between targets. Multivariate proper scores such as the energy score [35] and joint tests such as conditional classifiers are still needed, as are marginal checks, because a per-event score does not measure the population-level spectrum [12].

Differential skill depends on what is binned. The number of parton $p _ { \mathrm { T } }$ bins hardly matters, but binning in $| \eta |$ or jointly in $p _ { \mathrm { T } }$ and flavour gives very different values (Appendix F), so a differential skill means little unless its binning is stated. Momentum response is also weakly conditional by design. We use calibrated jets, whose energy corrections are built to flatten the response, so low skill on that target is expected. Our numbers are specific to this task and should not be carried over to other surrogates without repeating the procedure.

## 8 Conclusion

Marginal divergences alone can reward conditional collapse, as a condition-blind sampler shows on this benchmark. A CRPS skill score with the exact marginal reference is easier to interpret, but global skill depends on the breadth of the evaluation spectrum, so it should be reported alongside a differential skill that names its binning. A k-NN extrapolation then estimates how much skill the conditioner supports. Two widely used conditional density estimators agree with that estimate in parton-to-jet simulation, and a binned lookup table with no learned parameters recovers most of the same skill on most targets. This suggests the limit here is set mainly by the conditioning state and not by model capacity.

## Acknowledgements

This work was carried out by C.D. as an undergraduate research project under the supervision of S.D. It uses simulated data released through the CERN Open Data Portal [23]. The authors would like to acknowledge the efforts put in by the CMS Collaboration to release the simulation samples used in this work.

## Description of AI Use

Generative AI was used as a collaborative programming assistant to help write code snippets, and debug scripts, as well as for LaTeX formatting and copy-editing. To verify the accuracy of any generated code or text, we established a fully reproducible, end-to-end evaluation pipeline with provenance checks and tests. All empirical metrics and physics boundaries were directly computed from raw data and manually validated to ensure they adhere to expected mathematical constraints (such as the exact CRPS floors and k-NN ceilings). The core scientific methodology, claims, physical interpretations, mathematical derivations, and final conclusions were the authors’ own and were rigorously verified.

## References

[1] J. Allison, K. Amako, J. Apostolakis, P. Arce, M. Asai, T. Aso, E. Bagli, A. Bagulya, S. Banerjee, G. Barrand, B.R. Beck, A.G. Bogdanov, D. Brandt, J.M.C. Brown, H. Burkhardt, Ph. Canal, D. Cano-Ott, S. Chauvie, K. Cho, G.A.P. Cirrone, G. Cooperman, M.A. Cortés-Giraldo, G. Cosmo, G. Cuttone, G. Depaola, L. Desorgher, X. Dong, A. Dotti, V.D. Elvira, G. Folger, Z. Francis, A. Galoyan, L. Garnier, M. Gayer, K.L. Genser, V.M. Grichine, S. Guatelli, P. Guèye, P. Gumplinger, A.S. Howard, I. Hˇrivnácová, S. Hwang, S. Incerti, A. Ivanchenko, V.N.ˇ Ivanchenko, F.W. Jones, S.Y. Jun, P. Kaitaniemi, N. Karakatsanis, M. Karamitros, M. Kelsey, A. Kimura, T. Koi, H. Kurashige, A. Lechner, S.B. Lee, F. Longo, M. Maire, D. Mancusi, A. Mantero, E. Mendoza, B. Morgan, K. Murakami, T. Nikitina, L. Pandola, P. Paprocki, J. Perl, I. Petrovic, M.G. Pia, W. Pokorski, J.M. Quesada, M. Raine, M.A. Reis, A. Ribon,´ A. Ristic Fira, F. Romano, G. Russo, G. Santin, T. Sasaki, D. Sawkey, J.I. Shin, I.I. Strakovsky,´ A. Taborda, S. Tanaka, B. Tomé, T. Toshito, H.N. Tran, P.R. Truscott, L. Urban, V. Uzhinsky, J.M. Verbeke, M. Verderi, B.L. Wendt, H. Wenzel, D.H. Wright, D.M. Wright, T. Yamashita, J. Yarba, and H. Yoshida. Recent developments in Geant4. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 835:186–225, November 2016. ISSN 01689002. doi: 10.1016/j.nima.2016.06.125. URL https://linkinghub.elsevier.com/retrieve/pii/S0168900216306957.

[2] J. Alwall, R. Frederix, S. Frixione, V. Hirschi, F. Maltoni, O. Mattelaer, H.-S. Shao, T. Stelzer, P. Torrielli, and M. Zaro. The automated computation of tree-level and next-to-leading order differential cross sections, and their matching to parton shower simulations. J. High Energ. Phys., 2014(7):79, July 2014. ISSN 1029-8479. doi: 10.1007/JHEP07(2014)079. URL http://link.springer.com/10.1007/JHEP07(2014)079.

[3] Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proc. Natl. Acad. Sci. U.S.A., 117(48):30055–30062, December 2020. ISSN 0027-8424, 1091- 6490. doi: 10.1073/pnas.1912789117. URL https://pnas.org/doi/full/10.1073/pnas .1912789117.

[4] Johann Brehmer, Gilles Louppe, Juan Pavez, and Kyle Cranmer. Mining gold from implicit models to improve likelihood-free inference. Proc. Natl. Acad. Sci. U.S.A., 117(10):5242– 5249, March 2020. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1915980117. URL https://pnas.org/doi/full/10.1073/pnas.1915980117.

[5] Michela Paganini, Luke De Oliveira, and Benjamin Nachman. CaloGAN: Simulating 3D high energy particle showers in multilayer electromagnetic calorimeters with generative adversarial networks. Phys. Rev. D, 97(1):014021, January 2018. ISSN 2470-0010, 2470-0029. doi: 10.1103/PhysRevD.97.014021. URL https://link.aps.org/doi/10.1103/PhysRevD. 97.014021.

[6] Andreas Adelmann, Walter Hopkins, Evangelos Kourlitis, Michael Kagan, Gregor Kasieczka, Claudius Krause, David Shih, Vinicius Mikuni, Benjamin Nachman, Kevin Pedro, and Daniel Winklehner. New directions for surrogate models and differentiable programming for High Energy Physics detector simulation, 2022. URL https://arxiv.org/abs/2203.08806.

[7] Francesco Vaselli, Andrea Rizzi, Filippo Cattafesta, and Gloria Cicconofri. FlashSim: Accelerating HEP simulation with an end-to-end Machine Learning framework. EPJ Web of Conf., 295:09020, 2024. ISSN 2100-014X. doi: 10.1051/epjconf/202429509020. URL https://www.epj-conferences.org/10.1051/epjconf/202429509020.

[8] Anja Butter, Tilman Plehn, and Ramon Winterhalder. How to GAN LHC events. SciPost Phys., 7(6):075, December 2019. ISSN 2542-4653. doi: 10.21468/SciPostPhys.7.6.075. URL https://scipost.org/10.21468/SciPostPhys.7.6.075.

[9] The ATLAS Collaboration. Deep Generative Models for Fast Photon Shower Simulation in ATLAS. Comput Softw Big Sci, 8(1):7, December 2024. ISSN 2510-2036, 2510-2044. doi: 10.1007/s41781-023-00106-9. URL https://link.springer.com/10.1007/s41781-0 23-00106-9.

[10] Albert Tarantola. Inverse problem theory and methods for model parameter estimation. Number 89 in Other titles in applied mathematics. Society for Industrial and Applied Mathematics, Philadelphia, Pa, 2005. ISBN 978-0-89871-792-1 978-0-89871-572-9. doi: 10.1137/1.9780898717921.

[11] Christopher M. Bishop. Pattern Recognition and Machine Learning. Information science and statistics. Springer New York, New York, NY, 2006. ISBN 978-0-387-31073-2.

[12] Mads H. Baattrup, Jörn Bach, Laurids Jeppe, Finn Labe, Alexander Grohsjean, Christian Schwanenberger, and Peer Stelldinger. Pointwise Metrics Mislead: An Evaluation Protocol for Multimodal Inverse Problems, May 2026. URL http://arxiv.org/abs/2605.22891. arXiv:2605.22891 [cs.LG].

[13] Leonid Kantorovich and Gennady S. Rubinstein. On a space of totally additive functions. Vestnik Leningrad. Univ., 13(7):52–59, 1958.

[14] S. Kullback and R. A. Leibler. On Information and Sufficiency. Ann. Math. Statist., 22 (1):79–86, March 1951. ISSN 0003-4851. doi: 10.1214/aoms/1177729694. URL http: //projecteuclid.org/euclid.aoms/1177729694.

[15] Raghav Kansal, Carlos Pareja, Zichun Hao, and Javier Duarte. JetNet: A Python package for accessing open datasetsand benchmarking machine learning methods in high energy physics. JOSS, 8(90):5789, October 2023. ISSN 2475-9066. doi: 10.21105/joss.05789. URL https://joss.theoj.org/papers/10.21105/joss.05789.

[16] Claudius Krause, Michele Faucci Giannelli, Gregor Kasieczka, Benjamin Nachman, Dalila Salamani, David Shih, Anna Zaborowska, Oz Amram, Kerstin Borras, Matthew R Buckley, Erik Buhmann, Thorsten Buss, Renato Paulo Da Costa Cardoso, Anthony L Caterini, Nadezda Chernyavskaya, Federico A G Corchia, Jesse C Cresswell, Sascha Diefenbacher, Etienne Dreyer, Vijay Ekambaram, Engin Eren, Florian Ernst, Luigi Favaro, Matteo Franchini, Frank Gaede, Eilam Gross, Shih-Chieh Hsu, Kristina Jaruskova, Benno Käch, Jayant Kalagnanam, Raghav Kansal, Taewoo Kim, Dmitrii Kobylianskii, Anatolii Korol, William Korcari, Dirk Krücker, Katja Krüger, Marco Letizia, Shu Li, Qibin Liu, Xiulong Liu, Gabriel Loaiza-Ganem, Thandikire Madula, Peter McKeown, Isabell-A Melzer-Pellmann, Vinicius Mikuni, Nam Nguyen, Ayodele Ore, Sofia Palacios Schweitzer, Ian Pang, Kevin Pedro, Tilman Plehn, Witold Pokorski, Huilin Qu, Piyush Raikwar, John A Raine, Humberto Reyes-Gonzalez, Lorenzo Rinaldi, Brendan Leigh Ross, Moritz A W Scham, Simon Schnake, Chase Shimmin, Eli Shlizerman, Nathalie Soybelman, Mudhakar Srivatsa, Kalliopi Tsolaki, Sofia Vallecorsa, Kyongmin Yeo, and Rui Zhang. CaloChallenge 2022: a community challenge for fast calorimeter simulation. Rep. Prog. Phys., 88(11):116201, November 2025. ISSN 0034-4885, 1361-6633. doi: 10.1088/1361-6633/ae1304. URL https://iopscience.iop.org/article/10.1088/1361-6633/ae1304.

[17] Vinicius Mikuni and Benjamin Nachman. Score-based generative models for calorimeter shower simulation. Phys. Rev. D, 106(9):092009, November 2022. ISSN 2470-0010, 2470-0029. doi: 10.1103/PhysRevD.106.092009. URL https://link.aps.org/doi/10.1103/PhysRevD. 106.092009.

[18] Erik Buhmann, Sascha Diefenbacher, Engin Eren, Frank Gaede, Gregor Kasicezka, Anatolii Korol, William Korcari, Katja Krüger, and Peter McKeown. CaloClouds: fast geometryindependent highly-granular calorimeter simulation. J. Inst., 18(11):P11025, November 2023. ISSN 1748-0221. doi: 10.1088/1748-0221/18/11/P11025. URL https://iopscience.iop .org/article/10.1088/1748-0221/18/11/P11025.

[19] Oz Amram and Kevin Pedro. Denoising diffusion models with geometry adaptation for high fidelity calorimeter simulation. Phys. Rev. D, 108(7):072014, October 2023. ISSN 2470-0010, 2470-0029. doi: 10.1103/PhysRevD.108.072014. URL https://link.aps.org/doi/10. 1103/PhysRevD.108.072014.

[20] Claudius Krause and David Shih. Fast and accurate simulations of calorimeter showers with normalizing flows. Phys. Rev. D, 107(11):113003, June 2023. ISSN 2470-0010, 2470-0029. doi: 10.1103/PhysRevD.107.113003. URL https://link.aps.org/doi/10.1103/PhysR evD.107.113003.

[21] Raghav Kansal, Anni Li, Javier Duarte, Nadezda Chernyavskaya, Maurizio Pierini, Breno Orzari, and Thiago Tomei. Evaluating generative models in high energy physics. Phys. Rev. D, 107(7):076017, April 2023. ISSN 2470-0010, 2470-0029. doi: 10.1103/PhysRevD.107.076017. URL https://link.aps.org/doi/10.1103/PhysRevD.107.076017.

[22] CMS Collaboration. The CMS experiment at the CERN LHC. Journal ofInstrumentation, 3 (08):S08004, August 2008. doi: 10.1088/1748-0221/3/08/S08004. URL https://doi.org/ 10.1088/1748-0221/3/08/S08004.

[23] CMS Collaboration. Simulated dataset QCD\_Pt-15to7000\_TuneCP5\_Flat2018\_13TeV\_pythia8 in NANOAODSIM format for 2016 collision data, 2024. URL http://opendata.cern.ch /record/63168. DOI: 10.7483/OPENDATA.CMS.CK4Y.83FS.

[24] Christopher M. Bishop. Mixture density networks. Technical Report NCRG/94/004, Aston University, Birmingham, 1994.

[25] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

[26] Hans Hersbach. Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems. Wea. Forecasting, 15(5):559–570, October 2000. ISSN 0882-8156, 1520- 0434. doi: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2. URL http://journa ls.ametsoc.org/doi/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2.

[27] Allan H. Murphy. Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. Mon. Wea. Rev., 116(12):2417–2424, December 1988. ISSN 0027-0644, 1520-0493. doi: 10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2. URL http://journals.ametsoc.org/doi/10.1175/1520-0493(1988)116<2417: SSBOTM>2.0.CO;2.

[28] Takeru Miyato and Masanori Koyama. cGANs with projection discriminator. In International Conference on Learning Representations, 2018. URL https://arxiv.org/abs/1802.056 37.

[29] Terrance DeVries, Adriana Romero, Luis Pineda, Graham W. Taylor, and Michal Drozdzal. On the evaluation of conditional GANs, 2019. URL https://arxiv.org/abs/1907.08175.

[30] Michael Soloveitchik, Tzvi Diskin, Efrat Morin, and Ami Wiesel. Conditional Frechet Inception Distance, 2021. URL https://arxiv.org/abs/2103.11521.

[31] Suman Ravuri and Oriol Vinyals. Classification accuracy score for conditional generative models. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://arxiv.org/abs/1905.10887.

[32] David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests, 2016. URL https://arxiv.org/abs/1610.06545.

[33] Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating Bayesian inference algorithms with simulation-based calibration, 2018. URL https://arxiv. org/abs/1804.06788.

[34] Pablo Lemos, Adam Coogan, Yashar Hezaveh, and Laurence Perreault-Levasseur. Samplingbased accuracy testing of posterior estimators for general inference. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of PMLR, pages 19256–19273, 2023. URL https://arxiv.org/abs/2302.03026.

[35] Tilmann Gneiting and Adrian E Raftery. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477):359–378, March 2007. ISSN 0162-1459, 1537-274X. doi: 10.1198/016214506000001437. URL http://www.tandfonline.com/doi/abs/10.1198/016214506000001437.

[36] Christopher A. T. Ferro, David S. Richardson, and Andreas P. Weigel. On the effect of ensemble size on the discrete and continuous ranked probability scores. Meteorological Applications, 15(1):19–24, March 2008. ISSN 1350-4827, 1469-8080. doi: 10.1002/met.45. URL https: //rmets.onlinelibrary.wiley.com/doi/10.1002/met.45.

[37] L. Delle Monache, F. A. Eckel, D. L. Rife, B. Nagarajan, and K. R. Searight. Probabilistic weather prediction with an analog ensemble. Mon. Wea. Rev., 141(10):3498–3516, October 2013. doi: 10.1175/MWR-D-12-00281.1.

[38] T. Cover and P. Hart. Nearest neighbor pattern classification. IEEE Trans. Inform. Theory, 13 (1):21–27, January 1967. ISSN 0018-9448, 1557-9654. doi: 10.1109/TIT.1967.1053964. URL https://ieeexplore.ieee.org/document/1053964/.

[39] Torbjörn Sjöstrand, Stefan Ask, Jesper R. Christiansen, Richard Corke, Nishita Desai, Philip Ilten, Stephen Mrenna, Stefan Prestel, Christine O. Rasmussen, and Peter Z. Skands. An introduction to PYTHIA 8.2. Computer Physics Communications, 191:159–177, June 2015. ISSN 00104655. doi: 10.1016/j.cpc.2015.01.024. URL https://linkinghub.elsevier. com/retrieve/pii/S0010465515000442.

[40] CMS Collaboration. Extraction and validation of a new set of CMS pythia8 tunes from underlying-event measurements. Eur. Phys. J. C, 80(1):4, January 2020. ISSN 1434-6044, 1434-6052. doi: 10.1140/epjc/s10052-019-7499-4. URL http://link.springer.com/10 .1140/epjc/s10052-019-7499-4.

[41] James E. Matheson and Robert L. Winkler. Scoring rules for continuous probability distributions. Management Science, 22(10):1087–1096, 1976. ISSN 00251909, 15265501. doi: 10.1287/mn sc.22.10.1087. URL http://www.jstor.org/stable/2629907.

[42] C. Gini. Variabilità e mutabilità: contributo allo studio delle distribuzioni e delle relazioni statistiche. [Fasc. I.]. Studi economico-giuridici pubblicati per cura della facoltà di Giurisprudenza della R. Università di Cagliari. Tipogr. di P. Cuppini, 1912. URL https://books.google.co.in/books?id=fqjaBPMxB9kC.

[43] C. A. T. Ferro. Fair scores for ensemble forecasts. Q.J.R. Meteorol. Soc., 140(683):1917–1923, July 2014. ISSN 00359009. doi: 10.1002/qj.2270. URL https://onlinelibrary.wiley. com/doi/10.1002/qj.2270.

[44] Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. Springer Texts in Statistics. Springer US, New York, NY, 2021. ISBN 978-1-0716-1417-4 978-1-0716-1418-1. doi: 10.1007/978-1-0716-1418-1. URL https://link.springer.com/10.1007/978-1-0716-1418-1.

[45] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018. ISSN 0893-6080. doi: 10.1016/j.neunet.2017.12.012. URL https://www.sciencedirect. com/science/article/pii/S0893608017302976. Special issue on deep reinforcement learning.

[46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?i d=Bkg6RiCqY7.

[47] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In International Conference on Learning Representations, 2017. URL https://arxiv.org/ab s/1608.03983.

[48] Chris J. Maddison, Daniel Tarlow, and Tom Minka. A\* Sampling. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips. cc/paper\_files/paper/2014/file/937debc749f041eb5700df7211ac795c-Paper.p df.

[49] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper /2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.

## A Model Architectures and Training Setup

For a like-for-like comparison, both the Mixture Density Network (MDN) [24] and Conditional Flow Matching (CFM) [25] surrogates are built on the same backbone, a 6-layer Multilayer Perceptron (MLP) with a hidden width of 512 dimensions and SiLU activations [45]. The models are not matched in total size. With its output heads the MDN has 1.95 million parameters, and the CFM has 1.43 million including its time embedding. Both models are trained for 100 epochs using a batch size of 4,096 on the same data splits, and the seed of a split also sets the initialization and the batch order. Optimization uses AdamW [46] (learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $1 \times 1 0 ^ { - 5 } )$ with a linear warmup over the first 2,000 steps and cosine annealing [47], and gradients are clipped at norm 1.0. Each model is evaluated at the epoch with the best validation loss, which is the NLL for the MDN and the velocity MSE for the CFM.

## A.1 Mixture Density Network (MDN)

The MDN outputs parameters for an $_ { 8 \cdot }$ -component multivariate Gaussian mixture. The shared MLP backbone branches into three separate 3-layer MLP heads (hidden width 256) that predict the mixture weights (π), means (µ), and full lower-triangular Cholesky covariance factors $( L )$ for each component. To prevent variance collapse during training, the Cholesky diagonal uses a softplus activation paired with a hard epsilon floor.

The MDN is optimized on the exact conditional Negative Log-Likelihood (NLL) with a variance calibration penalty. This penalty (weighted by $\lambda = 0 . 1 )$ acts on the π-weighted mixture moments, pulling the batch mean of the normalized squared residual toward the target dimensionality $( D = 4 )$ To stabilize early training, weight decay is disabled for the output heads, and a cosine temperature annealing schedule is applied to the π logits over the first half of training. During inference, the model selects the active mixture component for each event using the Gumbel-max trick [48], followed by standard reparameterization sampling using the selected component’s mean and Cholesky factor.

## A.2 Conditional Flow Matching (CFM)

The CFM model acts as a continuous normalizing flow [49], parameterized by a velocity field network $v _ { \theta } ( x _ { t } , t , c )$ . The integration time t ∼ Uniform(0, 1) is projected via a 128-dimensional sinusoidal time embedding network. This time embedding is concatenated directly with the noisy state $x _ { t }$ and the 26-dimensional conditioner c before passing through the MLP backbone.

The CFM is trained using an independent noise coupling strategy. The target state $x _ { 1 }$ is drawn from the dataset, and the base state $\mathbf { \bar { \boldsymbol { x } } } _ { 0 } \sim \mathcal { N } ( \mu , \sigma ^ { 2 } )$ is drawn from a Gaussian noise prior explicitly moment-matched to the mean and variance of the empirical training targets. The model is optimized using Mean Squared Error (MSE) between the network’s predicted velocity and the target velocity vector $v _ { \mathrm { t a r g e t } } = x _ { 1 } - x _ { 0 }$ . At evaluation, samples are drawn by integrating the learned velocity field with a midpoint solver over 100 steps, for every CFM score reported here. Only the per-epoch validation check during training uses 20 steps. Table 1, Figure 1 and Figure D.1 score both surrogates on the first 200,000 test events of each split with $m = 2 0 0$ samples per event. The conditioner ladder (Figure E.1) and the generator-level oracle (Figure C.2) compute the CRPS on the first 50,000 test events with $m = 5 0$

## B Robustness to the Matching Cone

The angular channels $( \Delta \eta$ and $\Delta \phi )$ have very little conditional skill (Section 6.2). Target jets are matched to partons with a $\Delta R < 0 . 2$ cone, which truncates the tails of the angular targets. One might therefore ask whether the low skill is a real property of the problem or an artifact of this selection boundary.

To test this, we evaluated the differential k-NN ceiling on narrower cone subsets $( \Delta R < 0 . 1 5$ and $\Delta R < 0 . 1 0 )$ at each of the three data splits. Halving the matching cone to $\Delta R < 0 . 1$ 0 narrowed the raw target standard deviations, averaged over the three splits, by $1 . 8 3 7 \pm 0 . 0 0 6 \times$ (from 0.0746 to 0.0406) for $\Delta \eta .$ , and by $1 . 7 9 9 \pm 0 . 0 0 \bar { 3 } \times$ (from 0.0677 to 0.0376) for $\Delta \phi .$ If the low conditional skill were a boundary artifact, tightening the boundary this much should change the normalized CRPS ratio.

![](images/03ff24dc5cd111cc7102297b988c8bd899eb0d5d46ddb25f40f2a5c1c8417895.jpg)  
Figure B.1: Cone-narrowing robustness check. The median differential k-NN ceiling ratio evaluated across tightening $\Delta R$ cuts. Faint lines show the three data splits and bold lines their mean. While tightening the cone to 0.10 drastically reduces the raw standard deviation of the angular targets, the normalized conditional skill remains nearly flat, so the low predictability of $\Delta \eta$ and $\Delta \phi$ is not a boundary artifact.

Instead, as shown in Figure B.1, the median k-NN ceiling ratios changed little, moving on average by $+ 0 . 0 0 4 9 \pm 0 . 0 0 1 3$ for $\bar { \Delta } \eta ( 0 . 9 8 5 2  0 . 9 9 0 1 )$ and $+ 0 . 0 0 1 3 \pm 0 . 0 0 2 3$ for $\Delta \phi ( 0 . 9 9 6 4 \stackrel { - } {  } 0 . 9 9 7 7 )$ . For $\Delta \eta$ this shift was toward less apparent skill at every split, and for $\Delta \phi$ the mean shift points the same way, though it is consistent with zero across splits. The ratios change little under a large reduction in raw variance, so the low angular skill is not an artifact of the cone boundary.

## C Stage-Specific Ceilings & The Generator-Level Oracle

Applying the differential k-NN ceiling across intermediate simulation stages compares the stochasticity of the stage before generator-level jet formation with that of detector response, without training a network.

The standard surrogate task (parton2reco) maps directly from the hard-scatter parton to the reconstructed detector-level jet. We compute the model-free k-NN ceiling independently for the parton2genjet stage (parton shower, hadronization and underlying event) and the genjet2reco stage (detector simulation) across common parton $p _ { T }$ bins. All three stages use the same common event set, for both the reference and the query events.

As shown in Figure C.1, the detector response stage (genjet2reco) has a lower differential ceiling ratio across all targets than the parton2genjet stage, which means more skill against its own floor. For example, on seed 1 (the split shown in Figure C.1) the median differential ceiling ratio for jet mass fraction sits at 0.871 for the detector stage, compared to 0.897 for parton2genjet. This comparison is descriptive only, for the reason given next.

Changing the conditioning stage, for example from a parton input to a genjet input, also redefines the target, and the exact unconditional floor shifts by up to 6.4× on angular targets. Floor-normalized ratios therefore cannot be compared across stages unless the target distribution is the same.

To compare stages without these floor shifts, we used a generator-level oracle. On the jet mass target, the exact unconditional floors for both tasks match to five decimal places at every split (0.22580 on average). There, replacing the parton conditioner with the kinematics and flavour of the unobservable generator-level jet gave a relative gain in skill of $6 . 4 \pm 0 . 5 \% \mathrm { ( M D N ) }$ and $6 . 6 \pm 0 . 5 \%$ (CFM) on a common event set, averaged over the three splits. The generator-level jet mass is not part of this conditioner, so the gain measures what its 3-vector and flavour add, and it does not measure the total information lost before the detector. Figure C.2 shows this difference directly.

![](images/d4a7bd97bf8fcdf481a5879ef69bf330109d0ff7da444c0363958d006e9a9237.jpg)  
Figure C.1: Model-free differential stage ceilings. Evaluated via k-NN extrapolation across common $p _ { \mathrm { T } }$ bins (Seed 1). By computing the ceiling for the direct mapping (parton → reco) versus its intermediate physical stages (parton → genjet, genjet → reco), we compare the stochasticity added before and after generator-level jet formation.

![](images/08515c860ce706187975387cc62fabcdbc78388d44000e8f1dbedd3df93e30ce.jpg)  
Figure C.2: Generator-level oracle, jet-mass channel only (seeds 1–3). Floor-normalized CRPS for the standard parton2reco surrogate and for the same architecture handed the unobservable generator-level ${ \mathrm { j e t } } ,$ both scored on the first 50,000 events of the identical common event set with $m = 5 0$ . Faint lines show the three data splits and bold lines their mean. log\_mass\_frac is the only target shown because it is the only one whose exact unconditional floor is identical (to five decimal places, 0.22580) for the two conditioning stages. On the angular targets the floor shifts by up to 6.4× and the two ratios share no axis. Lower is better, so the drop is the skill gained from the generator-level jet’s 3-vector and flavour.

## D Full 3-Seed k-NN Extrapolations

As outlined in Section 5, the $r  0$ intercept is an estimate of the best achievable score rather than a strict bound, and we use it as a consistency check.

To show the radius dependence behind the slopes in Table 1, Figure D.1 gives the full k-NN extrapolation across all three data splits (Seeds 1, 2, and 3). The slope is positive at all three splits for momentum response and $\Delta \eta ,$ and changes sign between splits for jet mass and $\Delta \phi .$ . These signs should not be over-read, because the curves are nearly flat relative to the floor in every case and many of the signs change when the reference subset is redrawn (Appendix F). The surrogates lie on both sides of the intercept, within ±0.005 in ratio.

![](images/16cec27bd2b403b88c56255d5f894a06dd706de09ef5bdfc1cd5aa7f8d467371.jpg)  
Figure D.1: Full 3-seed k-NN extrapolations. k-NN extrapolation to zero mean neighborhood radius (r → 0) evaluated across three random data splits (Seeds 1, 2, and 3, top to bottom). Columns are the four targets. The intercept (⋆) estimates the Bayes-optimal CRPS. The slope is positive at all three splits for momentum response and $\Delta \eta$ (the two leftmost columns). Each panel is scaled to its own fit. The exact unconditional marginal floor lies above every panel here and enters Table 1 as the denominator of $S _ { \mathrm { g l o b a l } }$

## E Conditioner Ablation Ladder

The 26-dimensional conditioning state of Section 3 is built from four blocks, and they do not contribute equally. Figure E.1 retrains both surrogates on each rung of the ladder, ordered by information content, and reports the mean conditional log-likelihood alongside the CRPS ratio on the jet-mass channel, the target with the most conditional information. Each rung is trained at three random data splits (seeds 1–3). Faint lines show the individual splits and bold lines their mean. The CRPS ratio is scored on the first 50,000 test events of each split with $m = 5 0$ samples. The log-likelihood is exact for the MDN and uses the full test split. For the CFM it is obtained by integrating the exact divergence along the flow, extrapolated from 100 and 200 midpoint steps, on the first 100,000 test events.

Discrete flavor identity dominates. Adding it is worth roughly an order of magnitude more likelihood than pileup and the Fourier azimuth combined, and it accounts for essentially the whole of the jet-mass CRPS improvement across the ladder. The two model families agree on the ordering of the rungs and on the size of the flavour step, which is 0.106 nats for both. The remaining steps are small for both models, so the ladder describes the conditioning state and not either density estimator.

## F Sensitivity Checks

These checks reuse the stored samples of the trained models and run on the same 200,000 test events as Table 1, at all three data splits.

![](images/69ef2b092c84c8ec79ebbe15ac3063a4f7f761ae2bd9a80c0ef5ad75af3b9b31.jpg)  
Figure E.1: Conditioner ablation ladder (seeds 1–3). Faint lines show individual data splits and bold lines their mean. Mean conditional log-likelihood (left) and the floor-normalized CRPS on log\_mass\_frac (right) against the conditioner blocks supplied to the network. kin. is the three standardized parton kinematics, pu the pileup scalar, flav. the six-way one-hot flavor identity, and ϕ the 16 Fourier azimuth features. Higher is better on the left, lower on the right. The MDN log-likelihood uses the full test split and the CFM log-likelihood the first 100,000 test events.

Binning. Table F.1 repeats the differential evaluation under five binnings. The number of parton $p _ { \mathrm { T } }$ bins has little effect. Going from 14 to 28 bins moves every entry by less than 0.2 percentage points, and at 7 bins only jet mass moves noticeably, by about half a percentage point, which is expected when the bins are this wide.

The binning variable matters much more. Binning in |η| leaves almost all of the global skill on jet mass in place (27.9% against $S _ { \mathrm { g l o b a l } } = 2 8 . 7 \% )$ , which shows that |η| carries little information about jet mass fraction. Binning jointly in p<sub>T</sub> and flavour removes most of it. The ceiling keeps about 3% of skill on jet mass and about 0.1% on momentum response and $\Delta \phi ,$ while $\Delta \eta$ keeps about 1%. Most of the conditional information in these targets is therefore carried by parton p<sub>T</sub> and flavour, in line with the ablation ladder of Appendix E. In every row of the table the surrogates stay within about 0.2 percentage points of the ceiling, so their agreement with it does not depend on this choice.

Table F.1: Differential skill under different binnings. Skill in percent, taken as the median over bins, with mean ± standard deviation over three data splits. The k-NN rows use the $r  0$ ceiling. Every bin holds at least 200 test events, and the median $p _ { \mathrm { T } } \times$ flavour bin holds about 750. Because those bins are small, the floor here is corrected from the $\mathrm { \dot { N } ^ { 2 } }$ divisor to $N ( N - 1 )$ in each bin. This moves the 14-bin values by about 0.01 percentage points, so they round to the $S _ { \mathrm { d i f f } }$ column of Table 1.
<table><tr><td></td><td>Binning</td><td>log pò response</td><td> $\log m / p _ { \mathrm { T } }$ </td><td> $\Delta \eta$ </td><td>∆φ</td></tr><tr><td>k-NN</td><td> $p _ { \mathrm { T } } , 7$  bins</td><td> $2 . 3 9 \pm 0 . 1 5$ </td><td> $1 0 . 6 8 \pm 0 . 2 3$ </td><td> $1 . 5 8 \pm 0 . 3 7$ </td><td> $0 . 4 2 \pm 0 . 0 9$ </td></tr><tr><td></td><td>pT, 14 bins</td><td> $2 . 3 2 \pm 0 . 0 9$ </td><td> $1 0 . 1 3 \pm 0 . 3 6$ </td><td> $1 . 4 9 \pm 0 . 2 1$ </td><td> $0 . 3 7 \pm 0 . 1 8$ </td></tr><tr><td></td><td>pT, 28 bins</td><td> $2 . 3 5 \pm 0 . 1 0$ </td><td> $1 0 . 1 6 \pm 0 . 4 4$ </td><td> $1 . 4 3 \pm 0 . 2 8$ </td><td> $0 . 3 3 \pm 0 . 1 2$ </td></tr><tr><td></td><td>|η|, 14 bins</td><td> $3 . 4 3 \pm 0 . 0 5$ </td><td> $2 7 . 8 5 \pm 0 . 1 4$ </td><td> $1 . 3 2 \pm 0 . 3 4$ </td><td> $1 . 2 4 \pm 0 . 1 5$ </td></tr><tr><td></td><td> $p _ { \mathrm { T } } \times$  flavour, 84 bins</td><td> $0 . 1 3 \pm 0 . 0 8$ </td><td> $2 . 9 5 \pm 0 . 1 8$ </td><td> $1 . 0 9 \pm 0 . 1 2$ </td><td> $0 . 1 3 \pm 0 . 1 8$ </td></tr><tr><td>MDN</td><td> $p _ { \mathrm { T } } ,$  7 bins</td><td> $2 . 5 4 \pm 0 . 0 6$ </td><td> $1 0 . 6 8 \pm 0 . 2 5$ </td><td> $1 . 4 3 \pm 0 . 0 5$ </td><td> $0 . 2 9 \pm 0 . 0 3$ </td></tr><tr><td></td><td> $p _ { \mathrm { T } } ,$  14 bins</td><td> $2 . 4 9 \pm 0 . 0 5$ </td><td> $1 0 . 1 7 \pm 0 . 0 9$ </td><td> $1 . 4 5 \pm 0 . 0 3$ </td><td> $0 . 2 8 \pm 0 . 0 3$ </td></tr><tr><td></td><td> $p _ { \mathrm { T } } ,$  28 bins</td><td> $2 . 4 8 \pm 0 . 0 4$ </td><td> $1 0 . 0 5 \pm 0 . 0 7$ </td><td> $1 . 4 4 \pm 0 . 0 2$ </td><td> $0 . 2 9 \pm 0 . 0 2$ </td></tr><tr><td></td><td>|η|, 14 bins</td><td> $3 . 3 2 \pm 0 . 0 6$ </td><td> $2 7 . 8 9 \pm 0 . 2 0$ </td><td> $1 . 2 0 \pm 0 . 1 1$ </td><td> $1 . 1 5 \pm 0 . 0 5$ </td></tr><tr><td></td><td> $p _ { \mathrm { T } } \times$  flavour, 84 bins</td><td> $0 . 3 0 \pm 0 . 0 3$ </td><td> $3 . 0 7 \pm 0 . 1 4$ </td><td> $0 . 9 1 \pm 0 . 1 2$ </td><td> $- 0 . 0 2 \pm 0 . 0 2$ </td></tr><tr><td>CFM</td><td> $p _ { \mathrm { T } } , 7$  bins</td><td> $2 . 5 4 \pm 0 . 0 9$ </td><td> $1 0 . 7 5 \pm 0 . 1 9$ </td><td> $1 . 5 1 \pm 0 . 0 4$ </td><td> $0 . 3 2 \pm 0 . 0 2$ </td></tr><tr><td></td><td>pT, 14 bins</td><td> $2 . 5 1 \pm 0 . 0 3$ </td><td> $1 0 . 2 9 \pm 0 . 0 7$ </td><td> $1 . 5 2 \pm 0 . 0 0$ </td><td> $0 . 3 3 \pm 0 . 0 1$ </td></tr><tr><td></td><td>pT, 28 bins</td><td> $2 . 4 9 \pm 0 . 0 6$ </td><td> $1 0 . 1 1 \pm 0 . 0 3$ </td><td> $1 . 4 6 \pm 0 . 0 2$ </td><td> $0 . 3 3 \pm 0 . 0 2$ </td></tr><tr><td></td><td>|η|, 14 bins</td><td> $3 . 3 3 \pm 0 . 0 5$ </td><td> $2 7 . 9 4 \pm 0 . 1 9$ </td><td> $1 . 2 4 \pm 0 . 0 9$ </td><td> $1 . 2 5 \pm 0 . 0 4$ </td></tr><tr><td></td><td>pT× flavour, 84 bins</td><td> $0 . 3 1 \pm 0 . 0 2$ </td><td> $3 . 1 6 \pm 0 . 0 8$ </td><td> $0 . 8 6 \pm 0 . 1 3$ </td><td> $0 . 0 3 \pm 0 . 0 3$ </td></tr></table>

Reference subset. The main results use a random 400,000 reference events per flavour. We redrew that subset 12 times at each split. The intercept ratio scatters with a standard deviation of 0.0008–0.0013, which has a median of 2.5 times the OLS standard error and reaches 24 times it. All 312 comparisons between a model and an intercept (13 draws, three splits, four targets, two models) stay within ±0.005, the largest difference is 0.0041, and the models fall below the intercept in 54% of them. The slope is not stable under redrawing. It is positive in 42–67% of redraws on momentum response, 25–58% on jet mass and 25–75% on $\Delta \phi .$ , depending on the split. Only $\Delta \eta$ leans clearly positive, at 58–100%.

Full reference. We also removed the cap and used all of the roughly 8.6 million training events as reference, which shrinks the neighbourhood radii by about a factor of two. The intercept ratio moves by at most 0.0027 from the mean of the redraws. All 24 comparisons stay within ±0.005, with a largest difference of 0.0022, so the cap does not limit the estimate.

Azimuth. The surrogates receive $\phi$ and the neighbourhood search does not. To see whether this matters we repeated the full-reference search with the first Fourier pair (sin ϕ, cos ϕ) added to the search space, scaled to unit variance. We compare the ratio averaged over $k \geq 2 0$ , because the noise on the intercept is as large as the effect we are looking for. On momentum response, including ϕ lowers the k-NN score by $0 . 0 0 1 7 \pm 0 . 0 0 0 6$ . The surrogates sit 0.0014 below the search without ϕ and 0.0003 above the search with it, so ϕ accounts for the gap. Using $k = 1 0 0$ alone gives the same picture (0.0015 lower, and the surrogates then 0.0002 above). On average, jet mass and $\Delta \phi$ move by less than 0.0005. On $\Delta \eta$ the k-NN score rises by 0.0014 ± 0.0008, which is consistent with the larger neighbourhood, since the radii in this search are three to five times wider at $k \geq 2 0$

## G k-NN Extrapolation Algorithm

Algorithm 1 gives the procedure used to compute the data-driven ceiling. The idea is simple. For a query event, the true targets of its k most similar training events give an empirical picture of the conditional distribution. That picture is blurred, because the k neighbours are not at identical conditions, so the score it earns is slightly too high. A smaller neighbourhood is sharper but has fewer events. Measuring the score at several k and extrapolating the trend to zero radius estimates the score at zero blur, which is the best any model with the same conditioning information could achieve.

Three details in the implementation matter enough to state explicitly.

• Neighbours are used as they are. The k retrieved targets are scored directly as a $k \mathrm { - }$ member ensemble. The expected value of the $m ( m - 1 )$ kernel [43] does not depend on ensemble size [36], so drawing extra copies with replacement adds nothing and does harm. A resampled ensemble makes the kernel score the k-atom empirical distribution instead of the neighbourhood distribution. This adds a positive bias of about $\mathbb { E } | X - X ^ { \prime } | / ( 2 k )$ , which is largest at small $k ,$ so it pushes the score up there and can flip the sign of the trend.

• A query is excluded from its own neighbourhood by event identity, not by distance. If a query is also in the reference set, it retrieves itself and one ensemble member is a perfect prediction. This lowers the score by 40% at $k = 5$ (measured by querying reference events against their own search structure), in exactly the direction that would fake a model beating the ceiling. Distance cannot be used for this test, because NanoAOD stores $p _ { \mathrm { T } }$ and η at reduced precision. Different jets can therefore sit at zero distance from one another (2% of queries against the capped reference set), and discarding them would throw away the best neighbours available.

• Flavour is matched exactly. A separate search structure is built per flavour, so no neighbour is ever borrowed across flavours. Distances are computed over the four standardized continuous features (the three kinematic ones and pileup) and not the full 26-D conditioner, so the azimuth features are not used in the neighbourhood search.

The second term of FAIRCRPS uses the same all-pairs identity as the exact floor in Equation 6. Using the same all-pairs estimator for the ceiling, the floor and every model score is what makes the ratios in Table 1 comparable. The only difference is that the floor divides by $N ^ { 2 }$ and not $N ( N - 1 )$ . This is a relative difference of $1 / ( N - 1 )$ , or $5 \times 1 0 ^ { - 6 }$ at $N = 2 0 0 { , } 0 0 0$ and about $1 0 ^ { - 4 }$ in the smallest of the 14 $p _ { \mathrm { T } }$ bins, which holds about 9,000 events. Mixing estimators would matter much more. If a model scored with the fair kernel were normalized by a floor estimated from an m-member marginal

```latex
Algorithm 1 Data-Driven Performance Ceiling via k-NN Extrapolation
Require: Disjoint reference and query sets of events, each carrying a standardized continuous
conditioner $z ,$ a flavour $f ,$ targets $\boldsymbol { y } \in \mathbb { R } ^ { D }$ , and a unique event id
Require: Neighbourhood ladder $\overline { { \mathcal { K } } } = \{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \} $
Ensure: Per-target ceiling CRPS<sup>⋆</sup>, slope dCRPS/dr, and their standard errors
Stage 1: Stratify and standardize
1: Standardize the continuous conditioner using training-set statistics.
2: for each flavour $f$ do
3: Collect the reference events of that flavour (capped at $M _ { f }$ events) and build a KD-tree on
their z.
4: end for
Stage 2: Sweep the neighbourhood ladder
5: for $\bar { k } \in \mathcal { K }$ do ▷ $k \geq 2$ is required by the fair estimator
6: for each query event i do
7: Retrieve its $k$ nearest reference events within its ownflavour, excluding any neighbour
whose event id equals $i ^ { \gamma } \mathbf { s } .$
8: Take their true targets as the conditional ensemble $S _ { i }$ (used as-is, never resampled).
9: $\mathrm { c r p s } _ { i } \gets \mathrm { F A I R C R P S } ( y _ { i } , S _ { i } )$ , computed per target dimension.
10: end for
11: $\mathrm { C R P S } _ { k } \gets$ mean of crps over queries.
12: $\bar { r } _ { k } \gets$ mean distance to the k-th (farthest retained) neighbour.
13: end for
Stage 3: Extrapolate to zero radius
14: for each target dimension do
15: Fit $\mathrm { C R } \bar { \mathrm { P S } } _ { k } = \beta \bar { r } _ { k } + \alpha$ by ordinary least squares over the $n = | { \cal { K } } |$ points.
16: $\mathrm { C R P S ^ { \star } }  \alpha$ (the $r  0$ intercept); $\mathrm { d } \mathrm { C R } \mathrm { \bar { P } S } / \mathrm { d } r \gets \beta .$
17: $\begin{array} { r } { s ^ { 2 } \gets \frac { 1 } { n - 2 } \sum _ { k } ( \mathrm { C R P S } _ { k } - \beta \bar { r } _ { k } \stackrel { \cdot } { - } \alpha ) ^ { 2 } , \quad S _ { r r } \stackrel { \cdot } { - } \sum _ { k } ( \bar { r } _ { k } - \bar { r } ) ^ { 2 } . } \end{array}$
18: s $\begin{array} { r } { \mathrm { s e } ( \beta )  \sqrt { s ^ { 2 } / S _ { r r } } , \quad \mathrm { s e } ( \alpha )  \sqrt { s ^ { 2 } ( 1 / n + \bar { r } ^ { 2 } / S _ { r r } ) } . } \end{array}$
19: Report α as the estimated ceiling, and $\dot { \boldsymbol \beta }$ as a check on how strongly the estimate depends on
$k .$
20: end for
21: function $\mathrm { F A I R C R P S } ( y , S = \{ s _ { 1 } , \ldots , s _ { m } \} )$ ▷ fair $m ( m - 1 )$ kernel
22: Sort $S$ into order statistics $s _ { ( 1 ) } \leq \cdot \cdot \cdot \leq s _ { ( m ) }$
23: return $\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \lvert y - s _ { ( i ) } \rvert ~ - ~ \frac { 1 } { m ( m - 1 ) } \sum _ { i = 1 } ^ { m } ( 2 i - m - 1 ) ~ s _ { ( i ) }$
24: end function
ensemble with the plug-in $m ^ { 2 }$ kernel, the reference would be inflated by $( m + 1 ) / m$ [36]. This adds
$( 1 - S ) / ( m + 1 )$ of apparent skill, which is up to about 2% at $m = 5 0$ . That is about as large as the
whole differential skill on momentum response and larger than it on $\Delta \eta$ and $\Delta \phi$
The differential ceiling in Figure 1 runs the same sweep on the query events of each of the 14
log-spaced $\mathsf { p a r t o n } { - } p _ { \mathrm { T } }$ bins, against the same per-flavour reference trees, and reads each result against
the floor recomputed inside that bin, which removes the spectrum dependence of Section 4. Every
sweep, global or per bin, uses the same neighbourhood ladder $\kappa$ on the first 200,000 test events of
each split, with at most 400,000 reference events per flavour. Bins holding fewer than 200 test events
are skipped. The whole procedure is one tree build per flavour plus $| { \boldsymbol { \kappa } } |$ query passes per sweep, with
no training and no gradients.
```