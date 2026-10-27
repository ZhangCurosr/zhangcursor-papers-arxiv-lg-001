# Beyond Wasserstein Proximity: Finding Beter Transfer Sources

Gordei Verbii   
School of Mathematical Sciences   
Shanghai Jiao Tong University   
Shanghai, China   
Stratwise   
Singapore   
Georgii Killinkarov   
Stratwise   
Singapore   
Vladimir Yafarov   
Stratwise   
Singapore

## Abstract

Source selection often favors similar data, yet marginal proximity need not identify the most useful pretraining source. We motivate a conditional alternative through domain-adaptation theory and a regression example in which broader source variation improves estimation despite increasing Wasserstein distance. We examine five-day cryptocurrency volatility using ten Binance spot assets, six forecaster families, three seeds, and shared temporal cutofs. Selecting the three farthest rather than nearest Gaussian-Wasserstein sources improves mean persistence-relative skill by 12.52 percentage points for MLP and 20.90 for Transformer, with positive post-hoc paired date-block intervals. Forward pilot selection improves mean skill over random transfer in all six families. Compact recurrent models achieve the lowest absolute mean errors, separating selection utility from forecasting quality. Retrospective five-candidate screening shows additional gains from farthest-first geometry before pilot ranking for MLP and Transformer. We distinguish across-target associations from within-target decisions and evaluate target heterogeneity, training seeds, and temporal-block sensitivity. Findings are exploratory for a fixed historical panel; the screening analysis measures no runtime saving.

## 1 Introduction

Newly listed cryptocurrency assets provide short and uneven histories for volatility forecasting. Pretraining on another asset can improve the initialization available to a target forecaster, but choosing a source is a consequential decision: transfer can also preserve irrelevant structure. The practical problem is to identify a useful source for a fixed target, learner, and adaptation budget. Windowed Gaussian-Wasserstein distance ofers a tractable summary of difer ences between time-series distributions [4]. It is natural to prefer the closest source. We ask when a more distant source can instead be useful.

Similarity has a sound theoretical rationale. Domain-adaptation bounds relate target risk to source risk, distribution discrepancy, and the error of a hypothesis that must work on both domains [3, 16]. Reducing discrepancy tightens these bounds when other terms are controlled; distance alone does not order sources with diferent predictive relationships or estimation quality. Matching feature marginals can also leave conditional diferences unresolved [24]. These results explain why similarity may be insuficient, but do not themselves predict a benefit from greater distance.

There is a constructive reason to investigate that benefit. A source with broader informative variation can estimate a shared response more accurately while being farther away in marginal geometry. Finite target adaptation can retain this estimation advantage. Section 3 demonstrates the tradeof in a simple regression model. For neural forecasting, a source with diferent activity or volatility scale might similarly expose transferable temporal structure. This is a mechanism to investigate, not one established by the present experiment. Evidence that initialization-dependent benefits can survive fine-tuning [21], and that learned data selection can combine similarity with diversity [17], makes the question credible without implying a universal preference for dissimilarity.

Forecasting across heterogeneous series provides further context. Global models need not require identical marginal distributions [13], and domain-adapted forecasting can share temporal components while retaining domain-specific ones [10]. Their benefits depend on what is shared and on the learning procedure. We therefore investigate the conditional hypothesis that for a fixed target and finite adaptation budget, more distant Gaussian-Wasserstein sources can yield lower forecast error than nearer sources, with the efect depending on the learner. A direct comparison of source-selection rules tests this hypothesis more closely than an association across targets of unequal dificulty.

We evaluate ten Binance spot assets, six model families, and three seeds under shared calendar boundaries. The contributions are: (i) relating a statistical mechanism to a learner-specific reversal of nearest-source selection; (ii) separating selection gains from absolute quality using target-only, self-continuation, random-transfer, and HAR-style controls; and (iii) testing geometric screening before predictive validation against a matched random-screening comparator. Screening is retrospective; findings have exploratory, fixed-panel scope.

## 2 Related work

## 2.1 Domain adaptation and transferability

Domain-adaptation theory measures mismatch relative to a hypothesis class or loss [3, 12]; regression-specific work studies regularization and sample reweighting [6]. Optimal-transport bounds retain source and joint-risk terms alongside Wasserstein discrepancy [16]. Their assumptions do not directly certify an overlapping-window Gaussian � estimate for neural regression. Conditional shifts can obstruct adaptation despite invariant features [24]; negative transfer depends on the procedure and available target information [20]. Transfer dificulty can also change with the hypothesis class [9]. These strands motivate learner-specific comparisons against targetonly training.

Feature-transfer experiments distinguish specialization from optimization efects [21]. Distant tasks can still help relative to random initialization, although that work also reports declining transferability with task distance. It therefore does not establish a farthest-source selection rule. Ruder and Plank [17] learn selection criteria combining similarity and diversity in language tasks. Task-aware transferability measures include LEEP, which scores classifier outputs using target labels [14], and LogME, which evaluates feature–label evidence and supports regression [22]. Our forward pilot selector instead directly measures adapted forecasting error on later target outcomes. It is a validation comparator, not a new transferability score.

## 2.2 Dataset geometry and predictive similarity

For Gaussian distributions, Wasserstein geometry compares means and covariances [7]. Chen et al. [4] apply it to time-series windows and find useful associations with forecasting loss. MMD supplies a kernel-based alternative for comparing distributions [8]. Marginal feature distances can omit predictive structure. Optimal transport dataset distance incorporates feature–label geometry [1], while Cross-Learning Score uses bidirectional predictive generalization [19]. Our one-direction forward validation is not the full Cross-Learning Score. These approaches provide relevant alternatives, but superiority over them is not tested here. The question is how a common marginal geometry relates to source choices for diferent learners.

## 2.3 Learning across time series

DeepAR shares a recurrent probabilistic forecaster across related series [18]. Montero-Manso and Hyndman [13] study global versus local forecasting and complexity advantages of global learning; their results do not guarantee transfer from an arbitrary source with a fixed network. Domain Adaptation Forecaster shares attention components while retaining domain-specific values [10]. We instead study separate single-source initializations followed by fixed target adaptation.

Architecture matters independently of selection. Zeng et al. [23] show that linear forecasters can outperform the Transformer variants examined on long-horizon benchmarks. This motivates linear models with decomposition and compact recurrent controls without predicting their ranking for five-day volatility.

## 2.4 Volatility forecasting and evaluation

Cryptocurrency volatility has an established econometric literature: Katsiampa [11] compares Bitcoin GARCH specifications, including short- and long-run variance components. Andersen et al. [2] con nect intraday observations with realized-volatility measurement and forecasting; Corsi [5] models persistence across daily, weekly, and monthly horizons. These works motivate temporal baselines and a precise target definition. We forecast a logarithmic five-day RMS of daily returns, with a HAR-style comparator adapted to daily-return predictors. Neither is the intraday realized-volatility quantity of those studies. Patton [15] shows how proxy noise and loss choice can afect forecast rankings. Our findings concern the specified log proxy, not an invariant ranking for latent variance or trading objectives.

## 3 Why a more distant source can help

Consider $X _ { s } \ \sim \ N ( 0 , a ^ { 2 } ) , X _ { t } \ \sim \ N ( 0 , 1 )$ , and a shared response $\begin{array} { r } { Y = \beta X + \epsilon , } \end{array}$ where $a > 0$ and the noise is independent, mean-zero, with variance $\sigma ^ { 2 } > 0$ . Fit a through-origin source least-squares slope ${ \widehat \beta } _ { s }$ using $n > 2$ independent observations. Initialize $\beta _ { 0 } = \widehat { \beta } _ { s }$ and apply � population-gradient steps on target half-squared loss, with $0 < \eta < 1 \AA$ . For target squared risk $R _ { t } { \mathrm { : } }$ , we obtain

$$
W _ { 2 } ( P _ { s } ^ { X } , P _ { t } ^ { X } ) = | a - 1 | ,\tag{1}
$$

$$
\mathbb { E } [ R _ { t } ( \beta _ { K } ) - R _ { t } ( \beta ) ] = \frac { ( 1 - \eta ) ^ { 2 K } \sigma ^ { 2 } } { a ^ { 2 } ( n - 2 ) } .\tag{2}
$$

Conditional source slope variance is $\sigma ^ { 2 } / \textstyle { \sum _ { i } X _ { s , i } ^ { 2 } }$ , and $\textstyle \sum _ { i } X _ { s , i } ^ { 2 } / a ^ { 2 }$ $\chi _ { n } ^ { 2 } .$ . The inverse chi-square expectation gives the factor $1 / ( n - 2 ) ;$ each target update multiplies slope error by $1 - \eta .$ . Target covariate variance is one, so squared slope error equals excess squared risk.

For $a \ge 1$ , greater distance accompanies smaller expected excess risk: broader source variation estimates the common slope more accurately, and finite adaptation retains the advantage. For $a < 1$ moving farther from the target instead reduces information and increases risk. This illustration establishes a conditional possibility. Shared linear responses, independent samples, and population updates are not assumptions asserted for cryptocurrency markets or neural fine-tuning.

## 3.1 The relevant conditioning direction

Let $W _ { s t }$ be distance and $L _ { s t } ^ { ( m ) }$ test transfer loss for source �, target �, and family �. Correlating them across targets for a fixed source describes a diferent question from varying sources for one target. For a uniformly sampled ordered pair,

$$
\begin{array} { r l } & { \operatorname { C o v } ( W , L ) = \operatorname { \mathbb { E } } _ { t } \left[ \operatorname { C o v } ( W , L \mid t ) \right] } \\ & { \qquad + \operatorname { C o v } _ { t } ( \operatorname { \mathbb { E } } [ W \mid t ] , \operatorname { \mathbb { E } } [ L \mid t ] ) . } \end{array}\tag{3}
$$

This is the law of total covariance, not a decomposition of average correlations. Between-target dificulty can contribute to pooled associations. We consequently use within-target policy losses as the operational evidence.

## 4 Experimental design

## 4.1 Data, target, and information availability

The frozen panel comprises daily Binance spot USDT bars for BTC, ETH, BNB, XRP, SOL, SEI, JTO, ONDO, SUI, and TIA, from 10 August 2022 through 16 September 2026 where listing histories permit. Public-archive checksums verify the data. We use closes and actual quote-asset volume, with no futures or alternative-provider substitutions. Listing-day bars are retained; rolling warm-up excludes early model origins.

Each timestamp is a UTC bar date. Forecasts are issued after that bar closes. Let $r _ { c , u } = \log ( { p _ { c , u } } / { p _ { c , u - 1 } } )$ . The forward proxy and standardized target are

$$
v _ { c , u } = \sqrt { \frac { 1 } { 5 } \sum _ { i = 1 } ^ { 5 } r _ { c , u + i } ^ { 2 } } ,\tag{4}
$$

$$
y _ { c , u } = \{ \log [ \operatorname* { m a x } ( 1 0 ^ { - 6 } , v _ { c , u } ) ] - \mu _ { y } \} / \sigma _ { y } .\tag{5}
$$

Returns are not clipped. Pooled eligible training labels give $\mu _ { y } =$ $^ { - 3 }$ .47025894 and $\sigma _ { y } = 0$ .65780408. We evaluate squared error on this standardized log scale; common scaling leaves loss ratios unchanged.

The five features are daily log return, absolute log return, trailing 14-day return standard deviation, centered/scaled RSI-14, and log quote volume floored at one before transformation. Predictors use a 60-day lookback. Feature means and standard deviations are fitted to pooled training inputs and frozen. Longer histories contribute more observations to this scaler.

![](images/dc95d28efec97c8ca6138cf5bc78b44f64717a6ad5297a9e56c420279866f341.jpg)

![](images/40c8245ea5d051e1ba1d5df60dc4658db758a42b8f703d18bf82341b21ec32fd.jpg)  
Figure 1: Selection efects for three of nine sources. Left: policies versus expected random transfer. Right: direct farthest-versusnearest contrasts. Positive skill-point efects favor the named policy. Bars are conditional, pointwise 95% date-block intervals; direct contrasts are post-hoc.

All assets share exclusive boundaries: TRAIN ends 1 October 2025, PILOT ends 1 February 2026, and TEST ends 17 September 2026. Every forward label must end before its stage boundary. Historical feature context may extend into an earlier stage because it is then available. Training uses daily eligible origins. A common five-day phase gives 24 pilot and 45 test origins for every asset; test origins run from 1 February to 9 September 2026, with the final label ending 14 September. There are 1,062 training windows for each of BTC, ETH, BNB, XRP, and SOL; 796 for SUI, 692 for SEI, 615 for TIA, 578 for JTO, and 87 for ONDO.

## 4.2 Geometry and training

For each asset, we flatten 30-day feature windows into $\mathbb { R } ^ { 1 5 0 }$ and estimate a Gaussian with Ledoit–Wolf covariance shrinkage. Common preselection feature dates, 3 May 2025 through 31 January 2026, yield 245 overlapping windows. Pilot features already avail able at selection time enter the score; test features and responses do not. The windows summarize a distribution and are not treated as independent samples for inference. The score is the Gaussian distance [7]

$$
\begin{array} { r l } & { W _ { s t } ^ { 2 } = \vert \vert \hat { \mu } _ { s } - \hat { \mu } _ { t } \vert \vert ^ { 2 } + \mathrm { t r } \big [ \hat { \Sigma } _ { s } + \hat { \Sigma } _ { t } } \\ & { ~ - 2 ( \hat { \Sigma } _ { s } ^ { 1 / 2 } \hat { \Sigma } _ { t } \hat { \Sigma } _ { s } ^ { 1 / 2 } ) ^ { 1 / 2 } \big ] . } \end{array}\tag{6}
$$

Distance describes common recent histories, whereas source pretraining uses all eligible training history.

Six scalar-output architectures are evaluated (Table 1). Each asset is trained for 40 epochs with Adam at learning rate $1 0 ^ { - 3 } .$ . These fits serve as target-only baselines and source initializations. Every ordered cross-asset transfer then uses ten target epochs at $5 \times 1 0 ^ { - 4 }$ with a fresh optimizer. Self-continuation applies the same ten-epoch adaptation to the target’s own 40-epoch fit. It controls extra target training, but is not uninterrupted 50-epoch optimization. Batch size is 64, weight decay $1 0 ^ { - 3 }$ , gradient-norm clipping 5, dropout zero, and seeds 42, 1234, and 31337. There is no test-driven stopping. The complete design has $6 \times 3 \times \left( 1 0 + 1 0 + 9 0 \right) = 1 ,$ 980 fits. Fixed epochs imply unequal optimizer steps across unequal source histories.

Baselines share the test origins: causal five-day RMS persistence, each target’s training mean, the pooled training mean, ridge regression on flattened inputs, and HAR-style regression. The latter predicts the log proxy from log absolute return and its trailing fiveand 22-day means. Its daily-return inputs difer from the realizedvolatility HAR specification [5].

## 4.3 Source choice, metrics, and uncertainty

For each target and architecture, choose three of nine sources by smallest �, largest �, or smallest three-seed mean forward pilot transfer MSE. Primary choices are fixed before test evaluation. Losses are averaged over the three separate transfers and seeds; predictions and source data are not combined. Expected random

## The same transfer matrix answers two diferent questions

Small dots: individual assets · Diamonds: arithmetic means · Nine partners per correlation

![](images/9aafefd97a872d31c9b3e649124bb7d37a5e90a49b593961f25477c2207a0b1d.jpg)

![](images/452489674a6fe9a825d817324f390bfe1615c91742905403feafa3bf6cbe4cbf.jpg)  
Figure 2: Two conditioning directions for the same distance and transfer-loss matrices. Faint dots are correlations across nine partners for each fixed asset; diamonds average over ten assets. These descriptive distributions are not confidence intervals.

Table 1: Scalar forecasters and trainable parameter counts. DLinear is a scalar-output adaptation with a learned crossfeature readout; recurrent controls are not parametermatched to it.
<table><tr><td>Family</td><td>Configuration</td><td>Parameters</td></tr><tr><td>DLinear</td><td>Decomposition-based linear</td><td>616</td></tr><tr><td>LSTM-16</td><td>One layer, width 16</td><td>1,489</td></tr><tr><td>GRU-32</td><td>One layer, width 32</td><td>3,777</td></tr><tr><td>LSTM</td><td>Two layers, width 64</td><td>51,521</td></tr><tr><td>MLP</td><td>Two hidden layers, width 128</td><td>55,169</td></tr><tr><td>Transformer</td><td>Two layers, width 64, 4 heads</td><td>75,009</td></tr></table>

transfer is the exact all-nine mean, also the expected loss of a uniform three-source subset.

For target �, let $A _ { c , m } ^ { \pi }$ denote selected mean loss, $R _ { c , m }$ randomtransfer loss, and $P _ { c }$ persistence MSE. Our primary efect is

$$
\Delta _ { m } ^ { \pi } = \frac { 1 } { 1 0 } \sum _ { c } \frac { R _ { c , m } - A _ { c , m } ^ { \pi } } { P _ { c } } .\tag{7}
$$

We report 100Δ in percentage points ofpersistence-relative skill, not percent reduction in reference MSE. Targets are equally weighted after normalization. Macro MSE, target medians, and win counts de scribe complementary aspects of performance. Paired comparisons with other controls replace � by their losses.

Intervals use 1,000 circular test-date block resamples, four origins per block, and seed 2718. Both arms and all targets use identical sampled dates within a contrast; persistence denominators are recomputed. Primary policy-versus-random summaries use separate draw streams across family–policy combinations; post-hoc paired contrasts share a draw matrix. Percentile 95% intervals condition on this panel, fitted predictors, averaged seeds, and fixed source choices. They are pointwise and unadjusted for multiplicity, with no retraining, reselection, or asset resampling.

## 4.4 Retrospective screening and diagnostics

We restrict candidates to $q \in \{ 3 , 5 , 7 , 9 \}$ , then choose the pilot-best three within each shortlist. Rules retain the nearest $q ,$ farthest �, or a balanced set of $\lceil q / 2 \rceil$ nearest and $\lfloor q / 2 \rfloor$ farthest distinct sources. Ties use source-symbol order. The comparator exactly averages over all uniform �-source subsets, applying the same pilot selection inside each. There are 84, 126, 36, and one subsets respectively. This comparison separates geometric screening from predictive validation.

All budgets and rules are reported. The $q = 5$ contrast is fixed before calculating screening outcomes, but the analysis is post-hoc on an already evaluated panel. Its intervals share four-origin block draws. Candidate counts exclude source pretraining and distance calculation; this retrospective use of all fitted candidates measures no runtime saving.

Further diagnostics compute within-target Spearman correlations of pilot and test source losses and regret to an infeasible test-best-three selector. That selector uses the evaluation outcomes and is an optimistically biased within-candidate reference. Sensitiv ity checks keep the three-seed pilot choices fixed while evaluating individual seeds and block lengths of two, four, and eight origins.

## 5 Results

## 5.1 Operational source-selection efects

Figure 1 compares each selector with expected random transfer and directly contrasts farthest with nearest selection. Farthest-� has positive mean gains for five families, led by MLP at 8.07 points

## Better selection and better forecasting are separate outcomes

![](images/c9247f8de2eef5fadf821f25cba650365f8c005b86606a62ebd2917a2af73a8d.jpg)

![](images/9570dab297753b4200f0d0f0ceea3c0733fde8d5ffe9c11ae5f564273dab9b31.jpg)  
Figure 3: Selection utility and forecasting accuracy. Left: macro MSE, where lower is better; scratch uses 40 target epochs and self-continuation adds ten with fresh Adam. Right: post-hoc paired skill gains of pilot-selected transfer over daily-return HAR, with conditional 95% intervals.

[3.30, 15.27]. Nearest selection is most adverse in Transformer at −15.08 points [−25.69, −10.15]. Recurrent families have smaller, mixed geometric efects.

The direct farthest-over-nearest efect is 12.52 points for MLP [4.47, 23.87] and 20.90 for Transformer [14.64, 35.21]. The other four intervals include zero: DLinear 8.70 [−0.48, 21.00], LSTM-16 0.94 [−6.27, 7.28], GRU-32 −0.90 [−6.16, 4.90], and LSTM −2.00 [−8.33, 3.25]. These are learner-specific outcomes; diferent interval signs alone are not a formal between-family interaction test.

Pilot selection improves mean skill over random transfer in every family: 4.97 points for DLinear, 4.49 for LSTM-16, 4.29 for GRU-32, 3.05 for LSTM, 10.73 for MLP, and 11.73 for Transformer. All six conditional intervals are positive. Target efects nevertheless difer. DLinear’s nearest rule helps eight targets, but losses on SEI and TIA dominate its negative mean. GRU-32’s pilot-over-nearest gain averages 2.81 points, whereas its median is −2.34, with three wins, six losses, and one tie. An average gain is not a probability of helping a typical asset.

## 5.2 Across-target association and fixed-target choice

Mean per-source Pearson distance–loss correlations are negative for every family: −0.059, −0.093, −0.057, −0.098, −0.192, and −0.288 in the order of Table 1. Yet within-target means are positive for DLinear, LSTM-16, and GRU-32 (Figure 2). MLP and Transformer retain negative means in both directions. The spread of asset-specific correlations is substantial. This contrast is consistent with target heterogeneity in Equation 3, without identifying a causal mechanism. Within-target policy comparisons connect geometry to the source-choice decision.

## 5.3 Absolute forecast quality and controls

Pilot-selected macro MSE is 0.7653 for GRU-32, 0.7689 for LSTM-16, 0.8102 for LSTM, 0.8217 for DLinear, 1.0808 for Transformer, and 1.2155 for MLP. HAR-style MSE is 0.8534 and persistence is 1.1110. Thus strong selection gains for the larger feed-forward and attention models do not make them the best forecasters (Figure 3).

Post-hoc pilot-versus-HAR skill efects favor GRU-32 by 7.74 points [2.06, 14.88], LSTM-16 by 7.03 [3.58, 11.57], and LSTM by 4.03 [0.47, 9.76], with nine, seven, and eight target wins respectively. DLinear’s 2.61-point mean efect has an interval spanning zero. MLP and Transformer remain worse than HAR in mean error despite benefiting from source selection. No pairwise ranking between the compact recurrent families is established.

Pilot transfer improves mean skill over the corresponding scratch and self-continuation controls in all families, with positive intervals for all twelve contrasts. This statement concerns the specified bud gets. DLinear versus scratch averages 11.18 points, but has median −2.16 and only four target wins; unequal magnitudes matter. Selfcontinuation controls additional target epochs but cannot equalize the information in diferent source histories.

# Wasserstein can screen candidates before pilot selection

Retrospective exploratory analysis · Select pilot-best 3 within every shortlist

![](images/186c2592198c043a9364208ad5ac2b7a31942845630584d280a9e799425f8212.jpg)  
Zero = exact expected random-q screening + identical pilot selection. Bars only at fixed q = 5.  
Figure 4: Retrospective screening at four fixed candidate budgets. Each shortlist is reduced to its pilot-best three. Zero denotes parity with exact expected random screening of the same size, followed by identical pilot selection. All rules share axes; bars only at � = 5 show post-hoc conditional 95% intervals.

## 5.4 Screening candidates before pilot selection

At � = 5, farthest screening followed by pilot selection exceeds the matched random-five procedure by 4.24 points for MLP [1.00, 9.06] and 6.05 for Transformer [3.27, 12.09], with six and eight positive targets. Because random screening also applies pilot-best-three selection, these contrasts measure the incremental contribution of geometry (Figure 4). Other positive five-candidate efects have intervals spanning zero. Nearest-five screening is adverse for DLinear and MLP, and balanced-five screening is adverse for DLinear under these conditional intervals.

All rules coincide with full pilot selection at � = 9; zero gain means parity of the compared procedures. Five candidates would require 5/9 of candidate-specific adaptation fits if source parameters were already available, excluding pretraining and distance costs. The analysis reuses the complete candidate sweep and demonstrates no actual compute reduction. It motivates prospective validation of a fixed screening rule, rather than choosing an optimal budget from these test curves.

## 5.5 Ranking stability and sensitivity

Mean within-target pilot–test Spearman correlation ranges from −0.010 for GRU-32 to 0.493 for Transformer. Useful top-three selection can therefore coexist with poor stability of the complete nine-source ranking. Pilot has the smallest mean regret to the infeasible test-best-three selector among the four feasible policies for every family, but residual gaps remain 3.17–11.01 skill points. Mean selected-set overlap is 1.0–1.6 of three; two independent random three-of-nine sets overlap by one source on average. These are descriptive selection gaps, not achievable performance guarantees.

Pilot-versus-random point gains are positive for every seed in every family with the same three-seed pilot choices. Intervals remain positive at block lengths of two, four, and eight origins (10, 20, and 40 calendar days). Farthest-over-nearest intervals remain positive for MLP and Transformer and continue to span zero for the other families. These checks support internal stability under the tested choices, not replication on new assets or regimes.

## 6 Discussion and limitations

The results support learner-specific use of Wasserstein geometry. The illustration explains how informative variation and marginal mismatch can move together; source-history length, response compatibility, optimization, and regularization remain possible contributors in the market experiment. Compact recurrent models combine low error with useful pilot selection, while farthest-first screening is promising for MLP and Transformer.

Ten assets and one test period limit generalization. Temporal information comes from 45 shared origins. Fixed epochs leave unequal data and optimizer budgets; recent-history geometry and full-history pretraining are not disentangled.

Intervals omit retraining and reselection uncertainty. Seed and block sensitivity do not address new-panel generalization. Paired diagnostics and screening are post-hoc, with pointwise intervals un adjusted for multiplicity. Log-proxy MSE does not establish variance calibration, trading profitability, or intraday-volatility accuracy.

Prospective studies should rotate regimes, hold out asset cohorts, match training budgets, and vary adaptation length. Task-aware distances, additional horizons, and tuned GARCH benchmarks would broaden the comparison. Partial-candidate training should measure actual screening costs; multi-source learning should test diversity beyond separate-transfer loss averages.

## 7 Conclusion

Marginal similarity is one component of transfer utility. A simple finite-adaptation example shows how a more distant but informative source can help, motivating a conditional hypothesis for cryptocurrency forecasting. In this fixed panel, farthest-source selection beats nearest selection for MLP and Transformer, forward pilot selection improves mean skill over random transfer in all six families, and compact recurrent models attain the lowest mean errors. Matched-budget screening suggests an incremental role for farthest-first geometry before predictive validation. These findings support learner-specific source selection and motivate prospective tests across assets, regimes, and computational budgets.

## References

[1] David Alvarez-Melis and Nicolo Fusi. 2020. Geometric Dataset Distances via Optimal Transport. In Advances in Neural Information Processing Systems, Vol. 33. Curran Associates, Inc., 21428–21439. https://proceedings.neurips.cc/paper/202 0/hash/f52a7b2610fb4d3f74b4106fb80b233d-Abstract.html

[2] Torben G. Andersen, Tim Bollerslev, Francis X. Diebold, and Paul Labys. 2003. Modeling and Forecasting Realized Volatility. Econometrica 71, 2 (2003), 579–625. doi:10.1111/1468-0262.00418

[3] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. 2010. A theory of learning from diferent domains. Machine Learning 79 (2010), 151–175. doi:10.1007/s10994-009-5152-4

[4] Hongjie Chen, Akshay Mehra, Josh Kimball, and Ryan A. Rossi. 2025. Measuring Time-Series Dataset Similarity using Wasserstein Distance. Version 1. arXiv:2507.22189 doi:10.48550/arXiv.2507.22189

[5] Fulvio Corsi. 2009. A Simple Approximate Long-Memory Model of Realized Volatility. Journal ofFinancial Econometrics 7, 2 (2009), 174–196. doi:10.1093/jjfin ec/nbp001

[6] Corinna Cortes and Mehryar Mohri. 2014. Domain adaptation and sample bias correction theory and algorithm for regression. Theoretical Computer Science 519 (2014), 103–126. doi:10.1016/j.tcs.2013.09.027

[7] Matthias Gelbrich. 1990. On a Formula for the �<sup>2</sup> Wasserstein Metric between Measures on Euclidean and Hilbert Spaces. Mathematische Nachrichten 147, 1

(1990), 185–203. doi:10.1002/mana.19901470121

[8] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. 2012. A Kernel Two-Sample Test. Journal ofMachine Learning Research 13, 25 (2012), 723–773. https://jmlr.org/papers/v13/gretton12a.html

[9] Steve Hanneke, Samory Kpotufe, and Yasaman Mahdaviyeh. 2023. Limits ofModel Selection under Transfer Learning. In Proceedings ofthe Thirty Sixth Conference on Learning Theory (Proceedings of Machine Learning Research, Vol. 195). PMLR, 5781–5812. https://proceedings.mlr.press/v195/hanneke23c.html

[10] Xiaoyong Jin, Youngsuk Park, Danielle Maddix, Hao Wang, and Yuyang Wang. 2022. Domain Adaptation for Time Series Forecasting via Attention Sharing. In Proceedings ofthe 39th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 162). PMLR, 10280–10297. https://proceeding s.mlr.press/v162/jin22d.html

[11] Paraskevi Katsiampa. 2017. Volatility estimation for Bitcoin: A comparison of GARCH models. Economics Letters 158 (2017), 3–6. doi:10.1016/j.econlet.2017.06. 023

[12] Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. 2009. Domain Adaptation: Learning Bounds and Algorithms. In Proceedings of the 22nd Annual Conference on Learning Theory. Omnipress. https://www.learningtheory.org/col t2009/papers/003.pdf

[13] Pablo Montero-Manso and Rob J. Hyndman. 2021. Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting 37, 4 (2021), 1632–1653. doi:10.1016/j.ijforecast.2021.03.004

[14] Cuong Nguyen, Tal Hassner, Matthias Seeger, and Cedric Archambeau. 2020. LEEP: A New Measure to Evaluate Transferability of Learned Representations. In Proceedings of the 37th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 119). PMLR, 7294–7305. https://proceedings. mlr.press/v119/nguyen20b.html

[15] Andrew J. Patton. 2011. Volatility forecast comparison using imperfect volatility proxies. Journal ofEconometrics 160, 1 (2011), 246–256. doi:10.1016/j.jeconom.20 10.03.034

[16] Ievgen Redko, Amaury Habrard, and Marc Sebban. 2017. Theoretical Analysis of Domain Adaptation with Optimal Transport. In Machine Learning and Knowledge Discovery in Databases (Lecture Notes in Computer Science, Vol. 10535). Springer, 737–753. doi:10.1007/978-3-319-71246-8\_45

[17] Sebastian Ruder and Barbara Plank. 2017. Learning to select data for transfer learning with Bayesian Optimization. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 372–382. doi:10.18653/v1/D17-1038

[18] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. 2020. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal ofForecasting 36, 3 (2020), 1181–1191. doi:10.1016/j.ijforecast.2 019.07.001

[19] Shudong Sun, Hao Helen Zhang, and Joseph C. Watkins. 2026. Quantifying Data Similarity Using Cross Learning. Version 3. arXiv:2510.10866 doi:10.48550/arXiv .2510.10866

[20] Zirui Wang, Zihang Dai, Barnabás Póczos, and Jaime Carbonell. 2019. Characterizing and Avoiding Negative Transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11293–11302. https: //openaccess.thecvf.com/content\_CVPR\_2019/html/Wang\_Characterizi ng\_and\_Avoiding\_Negative\_Transfer\_CVPR\_2019\_paper.html

[21] Jason Yosinski, Jef Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks?. In Advances in Neural Information Processing Systems, Vol. 27. Curran Associates, Inc., 3320–3328. https://proceeding s.neurips.cc/paper\_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330- Abstract.html

[22] Kaichao You, Yong Liu, Jianmin Wang, and Mingsheng Long. 2021. LogME: Practical Assessment of Pre-trained Models for Transfer Learning. In Proceedings ofthe 38th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 139). PMLR, 12133–12143. https://proceedings.mlr.press/ v139/you21b.html

[23] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. 2023. Are Transformers Efective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence 37, 9 (2023), 11121–11128. doi:10.1609/aaai.v37i9.26317

[24] Han Zhao, Remi Tachet des Combes, Kun Zhang, and Geofrey Gordon. 2019. On Learning Invariant Representations for Domain Adaptation. In Proceedings of the 36th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 97). PMLR, 7523–7532. https://proceedings.mlr.press/v9 7/zhao19a.html