# Beyond RMSE and Granger Gain: Multi-Horizon Forecast-Error Entropy for Nonlinear Predictive Coupling

Andrei Velichko

Institute of Physics and Technology, Petrozavodsk State University, Petrozavodsk, Russia Email: velichkogf@gmail.com Google Scholar profile

Preprint, October 2026

## Abstract

We investigate whether the past of an auxiliary time series provides predictive information about a target beyond that contained in the target’s own history. We compare matched out-of-sample target-only and target-plus-source forecasts and introduce the forecast-error distribution entropy (FEDE) predictive-causality framework, where predictive causality is understood in the Granger sense: auxiliary history is informative when it improves prediction beyond the target’s own past. FEDE quantifies how the full signed forecast-error distribution changes across prediction horizons. Controlled nonlinear benchmarks hide source–target dependence behind nonlinear transformations not supplied to the predictors. LightGBM recovers these hidden channels, whereas Ridge does not. Across four equal-power nonlinearities and four noise levels, the mean response-sensitivity statistic for FEDE remains 29.00, 27.47, 23.22, and 28.67, compared with 14.09, 10.68, 12.67, and 12.71 for mean log-RMSE gain. Directional and source controls separate the true source-totarget channel from the reverse direction and from independent periodic, quasiperiodic, chaotic, and autoregressive auxiliaries. Chronological Granger-style comparisons show strong model-form dependence: a degree-2 polynomial model is exceptionally sensitive to the matched quadratic channel but becomes weak or oppositely aligned for other nonlinearities, while basic convergent cross mapping is weak or directionally inconsistent in this stochastic, partly non-invertible setting. FEDE remains positively aligned across all 108 tested combinations of window length, history length, and maximum forecast horizon. Together, these results establish FEDE as a sensitive and robust horizon-resolved tool that extracts predictive information beyond conventional errormagnitude measures when the nonlinear interaction form is unknown. The implementation code and benchmark data are openly available in a persistent Zenodo repository.

Keywords: forecast-error distribution entropy; forecast-error growth; predictive causality; nonlinear coupling; Granger causality; time series; machine learning. Software and benchmark package: Zenodo – 10.5281/zenodo.23330544.

## 1 Introduction

Many scientific and engineering problems involve two or more simultaneously observed time series for which simple association is not the quantity of primary interest. A stronger and more operational question is whether the past of a candidate series contains information that improves prediction of another series beyond what is already contained in the target’s own history. This distinction separates contemporaneous association from incremental predictive contribution. It is central to directional time-series analysis in economics and finance, climate and hydrology, epidemiology, neuroscience and physiology, ecology, and complex dynamical systems, where strongly correlated signals may arise from synchronization, common forcing, seasonality, or shared trends without providing independent predictive information [9, 19, 13, 22, 23, 24].

Let $x _ { t }$ denote a target time series and y<sub>t</sub> a candidate auxiliary series. The question studied here is

Does adding the aligned history of y improve, or systematically alter, the out-of-sample prediction of future x relative to using the history of x alone?

This formulation naturally leads to a matched restricted-versus-augmented prediction experiment: the two branches forecast exactly the same future target values, use the same training/test protocol and model class, and difer only in whether the history of y is available. Pairwise Pearson or Spearman correlation remains useful descriptively, but it does not condition on the target history and does not directly answer this incremental forecasting question.

The canonical predictive framework is Granger causality. In its classical linear form, a restricted autoregression based on the past of x is compared with a full model that additionally includes the past of y; a reduction in residual uncertainty is interpreted as evidence that y is Granger-predictive for x [9, 15]. The same directional logic has been generalized far beyond linear autoregression. Kernel and nonlinear extensions, polynomial and Volterra expansions, neural networks, and other machine-learning predictors replace the fixed linear basis while retaining the restricted-versus-full comparison [10, 11, 12, 16]. These methods can detect nonlinear dependence that linear Granger models miss, but their response is inevitably tied to the selected predictor representation, lag structure, regularization, and model complexity. A correctly specified low-dimensional basis can be extremely sensitive, whereas a misspecified basis can fail even when a strong predictive channel is present.

A closely related but more model-agnostic strategy is to evaluate the forecasts directly on held-out data. Baseline and augmented models can be compared using changes in RMSE, MAE, MSE, forecast ratios, or out-of-sample $R ^ { 2 }$ . This produces an interpretable efect size—how much forecast accuracy is gained by adding the candidate source—without requiring the statistic itself to assume a particular linear or nonlinear dependence. Statistical forecast-comparison procedures, most prominently the Diebold–Mariano test and its extensions, address a complementary question: whether the observed loss diference is statistically distinguishable from zero rather than how the predictive contribution should be represented [17, 18]. Importantly, multi-horizon forecast evaluation is itself well established; the novelty of the present work is therefore not the use of several forecast horizons alone.

A second major family takes a state-space rather than restricted-versus-full forecasting view. Convergent cross mapping (CCM) asks whether information about a putative driver can be reconstructed from the delay-coordinate manifold of the afected variable and whether cross-map skill improves as the library size increases [13, 20]. Empirical dynamic modeling (EDM), simplex projection, and S-map methods extend this state-space viewpoint to nonlinear forecasting and state-dependent interaction strength [21, 20]. These tools have been influential in ecology, epidemiology, climate and hydrology, physiology, neuroscience, and, more recently, financial time series [13, 22, 23, 24]. Their physical question is nevertheless diferent from incremental forecast gain: reconstructability of a source from an observed manifold need not be equivalent to improvement of a target forecast after the source is supplied. Standard CCM is also known to require careful treatment of embedding, convergence, noise, synchronization, common forcing, and short or discontinuous records [14, 19].

Table 1 summarizes the principal method families relevant to the present study. The comparison is intentionally conceptual rather than hierarchical because the methods estimate diferent objects. Granger methods quantify restricted-versus-full predictive dependence; forecast-gain measures quantify held-out accuracy improvement; Diebold–Mariano-type procedures quantify evidence for a loss diference; CCM/EDM quantify state-space reconstruction or nonlinear state-dependent prediction. The proposed FEDE construction belongs to the predictive-comparison side of this landscape but uses a diferent response variable: the horizon evolution of the complete signed forecast-error distribution.

Table 1: Representative method families for directional or incremental dependence between time series. The table emphasizes the object each method measures rather than treating the methods as interchangeable causal estimators.
<table><tr><td rowspan=1 colspan=6>Method family      Core question           Typical output      Role of forecast  Main limitation for  Representativehorizon           the present problem    references</td></tr><tr><td rowspan=1 colspan=6>Linear Granger  Does past y reduce  F/Wald statistic, p  Usually one-step  Misses nonlinear in-  [9, 15]</td></tr><tr><td rowspan=1 colspan=4>causality            the prediction-error  value, GC index,  or   lag-model</td><td rowspan=1 colspan=1>teractions when the</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=5 colspan=3>variance of x beyond  variance ratiopast x in a linearmodel?Nonlinear / ML Same     restricted-</td><td rowspan=1 colspan=1>based</td><td rowspan=1 colspan=1>linear basis is inad-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>equate; sensitive to</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>lag/model specifica-tion</td><td rowspan=4 colspan=1>[10, 11, 12, 16]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>NonlinearGC</td><td rowspan=1 colspan=1>Usually   one-</td><td rowspan=1 colspan=1>Sensitivity depends on</td></tr><tr><td rowspan=1 colspan=1>Granger</td><td rowspan=1 colspan=1>versus-fullquestion</td><td rowspan=1 colspan=1>score, generalized</td><td rowspan=1 colspan=1>step/few-step;</td><td rowspan=1 colspan=1>architecture or chosen</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>with a nonlinear</td><td rowspan=1 colspan=1>variance, variable</td><td rowspan=1 colspan=1>model  depen-</td><td rowspan=1 colspan=1>basis; greater tuning</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>predictor or feature</td><td rowspan=1 colspan=1>importance,  test</td><td rowspan=1 colspan=1>dent</td><td rowspan=1 colspan=1>burden</td><td rowspan=3 colspan=1>[17, 18]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>basis</td><td rowspan=1 colspan=1>statistic</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Out-of-sample</td><td rowspan=1 colspan=1>How  muchdoes</td><td rowspan=1 colspan=1>∆RMSE, ∆MAE,</td><td rowspan=1 colspan=1>Chosen explic-</td><td rowspan=1 colspan=1>Collapses forecast be-</td></tr><tr><td rowspan=1 colspan=1>forecast gain</td><td rowspan=1 colspan=1>adding y improve</td><td rowspan=1 colspan=1>loss ratio, OOS R²</td><td rowspan=1 colspan=1>itly by analyst;</td><td rowspan=1 colspan=1>havior to an accu-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>held-out prediction of</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>may be single-</td><td rowspan=1 colspan=1>racy summary and</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>future x?</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>or multi-horizon</td><td rowspan=1 colspan=1>does not describe how</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>the error distribution</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>changes with h</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Diebold-Mariano-</td><td rowspan=1 colspan=1>Is the mean loss differ</td><td rowspan=1 colspan=1>-DM statistic and p</td><td rowspan=1 colspan=1>Horizon-specific</td><td rowspan=1 colspan=1>Primarily a signif-</td><td rowspan=1 colspan=1>[17, 18]</td></tr><tr><td rowspan=1 colspan=1>type     forecast</td><td rowspan=1 colspan=1>ence between two fore-</td><td rowspan=1 colspan=1>value</td><td rowspan=1 colspan=1>or    extended</td><td rowspan=1 colspan=1>icance test, not a</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>comparison</td><td rowspan=1 colspan=1>cast systems distin-</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>to     multiple</td><td rowspan=1 colspan=1>coupling-magnitude</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>guishable from zero?</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>horizons</td><td rowspan=1 colspan=1>or   error-structure</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>descriptor</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CCM / EDM</td><td rowspan=1 colspan=1>Is information about</td><td rowspan=1 colspan=1>Cross-map  skill</td><td rowspan=1 colspan=1>Library size and</td><td rowspan=1 colspan=1>Reconstruction</td><td rowspan=1 colspan=1>[13, 20, 21, 14]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>one variable recover-</td><td rowspan=1 colspan=1>ρ(L), convergence,</td><td rowspan=1 colspan=1>embedding are</td><td rowspan=1 colspan=1>and predictive con-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>able from the recon-</td><td rowspan=1 colspan=1>nonlinear forecast</td><td rowspan=1 colspan=1>central; forecast</td><td rowspan=1 colspan=1>tributionarenot</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>structed state of an-</td><td rowspan=1 colspan=1>skill, local coeffi-</td><td rowspan=1 colspan=1>horizon enters</td><td rowspan=1 colspan=1>equivalent;    noise,</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>other, and does recon-</td><td rowspan=1 colspan=1>cients</td><td rowspan=1 colspan=1>EDM variants</td><td rowspan=1 colspan=1>synchronization, em-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>struction converge?</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>bedding, and common</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>forcing matter</td></tr><tr><td rowspan=1 colspan=1>FEDE</td><td rowspan=1 colspan=1>How does adding y</td><td rowspan=1 colspan=1>DirectedFEDE</td><td rowspan=1 colspan=1>Explicitly</td><td rowspan=1 colspan=1>Model-relative;   re-</td><td rowspan=1 colspan=1>Present study</td></tr><tr><td rowspan=1 colspan=1>predictive-</td><td rowspan=1 colspan=1>change the distribu-</td><td rowspan=1 colspan=1>∆ response and</td><td rowspan=1 colspan=1>uses</td><td rowspan=1 colspan=1>quires multi-horizon</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>causality frame-</td><td rowspan=1 colspan=1>tional evolution of out-</td><td rowspan=1 colspan=1>early-horizon</td><td rowspan=1 colspan=1>h = 1, . . . , H as</td><td rowspan=1 colspan=1>forecasts and an</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>work      (this</td><td rowspan=1 colspan=1>of-sample forecast er-</td><td rowspan=1 colspan=1>entropy-slope re-</td><td rowspan=1 colspan=1>theanalysis</td><td rowspan=3 colspan=2>engine</td></tr><tr><td rowspan=1 colspan=1>work)</td><td rowspan=1 colspan=1>rors for x across hori-</td><td rowspan=1 colspan=1>sponse</td><td rowspan=1 colspan=1>axis</td></tr><tr><td rowspan=1 colspan=3>zons?</td><td></td></tr></table>

Forecasting in nonlinear dynamical systems provides a natural setting in which the horizon dependence itself carries information. Classical nonlinear prediction and state-space reconstruction showed that the degradation of forecast skill with lead time reflects underlying dynamical structure [1, 2, 3, 4]. Our earlier work used machine-learning forecast-error growth to estimate largest-Lyapunovexponent-related behavior from scalar chaotic time series [5]. That line of work suggests that a forecast should not always be reduced to a single error number: the shape of the error evolution over h can be an informative observable.

The gap addressed here is therefore narrower, but also more specific, than the general problem of detecting directionality. We do not propose a new definition of structural or physical causality and do not claim novelty for the restricted-versus-augmented forecast comparison. Instead, FEDE treats the distribution of signed out-of-sample forecast errors at each horizon as the primary object. For a target-only forecast and a matched target-plus-auxiliary forecast, Shannon entropy is computed from the signed error distributions over $h = 1 , \ldots , H ;$ the change of the resulting entropy profile between short and longer horizons becomes a scalar, directed, model-relative descriptor. We refer to the complete construction as the FEDE predictive-causality framework. Throughout this work, predictive causality is used strictly in the Granger operational sense: a candidate source is predictive-causal for a target only insofar as its past provides additional out-of-sample predictive information beyond the target’s own past, within the specified model and conditioning set. The question is therefore no longer only whether y decreases an average loss, but whether access to y changes the way predictive uncertainty is redistributed across the forecast horizon.

This distinction motivates the controlled benchmarks used in the present study. A hidden nonlinear source is first constructed so that the raw auxiliary series can have essentially zero linear association with the future target even though a genuine predictive channel is present. The analysis then tests whether a flexible predictor can extract this hidden information and whether FEDE gives a more reproducible coupling response than conventional mean forecast gain. Direction and source controls ask whether the response is specific to the true $Y  X$ channel rather than to the reverse direction or to unrelated structured auxiliaries. A second benchmark varies the hidden functional form while keeping its power normalized, allowing direct comparison with chronological out-of-sample linear and polynomial Granger-style scores and with basic CCM convergence. A final predefined parameter grid then varies the FEDE analysis scales W, m, and H at fixed noise to test whether the nonlinear dose response survives changes in the method’s own window, history, and horizon settings.

The main contributions of the study are fivefold. First, we formulate the FEDE predictivecausality framework as a matched multi-horizon construction in which the only information diference between two forecast branches is the availability of the candidate auxiliary history. Second, we introduce FEDE-based directed descriptors that quantify changes in the entropy structure of signed forecast errors across horizons rather than only changes in mean error magnitude. Third, we validate source/direction specificity and robustness to several hidden nonlinear functional forms, including cases in which raw linear association is deliberately suppressed. Fourth, we compare the FEDE response with conventional multi-horizon forecast gain, chronological out-of-sample linear and degree-2 polynomial Granger-style controls, and CCM, while separately reporting computational cost. Fifth, we test the stability of the FEDE dose response over a predefined $3 \times 3 \times 3$ grid of window lengths, history lengths, and maximum forecast horizons. The comparison is designed to identify scope rather than to assert universal superiority: a correctly specified parametric model is expected to outperform a flexible generic method when the true interaction basis is known a priori.

The remainder of the paper is organized as follows. Section 2 defines the matched forecasting construction, FEG/FEDE descriptors, dose–response evaluation, directed contribution, time-resolved evaluation, and the competing Granger/CCM estimators. Section 3 separates the controlled tests into a linear-reference benchmark family (L1–L2) and a nonlinear Gaussian-AR(1) benchmark family (N1– N5), and states explicitly how each benchmark maps to the corresponding results. Section 4 presents calibration, detection, nonlinear extraction, specificity, method-comparison, tracking, computational, and parameter-robustness results. Section 5 discusses interpretation, model dependence, comparison with established directional methods, computational cost, and limitations. Section 6 summarizes the conclusions.

## 2 Methodology

## 2.1 Matched predictive-contribution construction

For history length m, define

$$
\mathbf { x } _ { t } = [ x _ { t - m + 1 } , \ldots , x _ { t } ] ,\tag{1}
$$

$$
\mathbf { z } _ { t } = [ x _ { t - m + 1 } , \ldots , x _ { t } , y _ { t - m + 1 } , \ldots , y _ { t } ] .\tag{2}
$$

Two matched predictors are fit using the same model class, hyperparameters, training indices, test indices, and forecast horizons $h = 1 , \ldots , H$

$$
\widehat { x } _ { t + h } ^ { ( x ) } = F _ { h } ( \mathbf { x } _ { t } ) ,\tag{3}
$$

$$
\widehat { x } _ { t + h } ^ { ( x , y ) } = F _ { h } ( \mathbf { z } _ { t } ) .\tag{4}
$$

The corresponding signed errors are

$$
e _ { x } ( i , h ) = x _ { i + h } - \widehat { x } _ { i + h } ^ { ( x ) } ,\tag{5}
$$

$$
e _ { x y } ( i , h ) = x _ { i + h } - \widehat { x } _ { i + h } ^ { ( x , y ) } .\tag{6}
$$

Any systematic diference between these two matched error sets is therefore attributable to the availability of the auxiliary history within the chosen prediction model.

![](images/ee84a1c615b4da0ae7a78723cf771e1f2754215bc418efdbb20e1321c527a01b.jpg)

Figure 1: Core construction for auxiliary predictive contribution. The two branches forecast exactly the same future target values. The only information diference is the availability of the aligned auxiliary history y. Conventional accuracy gain and FEG/FEDE descriptors are all computed from this same matched pair of out-of-sample forecasts.

Within the FEDE predictive-causality framework, this comparison is model-relative: predictive causality refers only to additional out-of-sample predictive information supplied by the candidate source. It should not be interpreted as structural or physical causal influence without additional assumptions, conditioning, or controls.

## 2.2 Conventional predictive gain

For each horizon,

$$
\mathrm { R M S E } _ { x } ( h ) = \sqrt { \frac { 1 } { N _ { h } } \sum _ { i } e _ { x } ( i , h ) ^ { 2 } } ,\tag{7}
$$

$$
\mathrm { R M S E } _ { x y } ( h ) = \sqrt { \frac { 1 } { N _ { h } } \sum _ { i } e _ { x y } ( i , h ) ^ { 2 } } .\tag{8}
$$

The main conventional summary is the mean multi-horizon log-RMSE gain,

$$
G _ { \mathrm { l o g R M S E } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \log \frac { \mathrm { R M S E } _ { x } ( h ) + \varepsilon } { \mathrm { R M S E } _ { x y } ( h ) + \varepsilon } .\tag{9}
$$

Positive values indicate lower forecast error when the auxiliary history is included.

## 2.3 Forecast-error growth (FEG) profile

For either matched error matrix e(i, h), define

$$
g ( h ) = \frac { 1 } { N _ { h } } \sum _ { i = 1 } ^ { N _ { h } } \log \left( \operatorname* { m a x } ( | e ( i , h ) | , \varepsilon ) \right) .\tag{10}
$$

The target-only and target-plus-auxiliary profiles are denoted $g _ { x } ( h )$ and $g _ { x y } ( h )$ . Their mean separation is

$$
G _ { \mathrm { p r o f l e } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \left[ g _ { x } ( h ) - g _ { x y } ( h ) \right] .\tag{11}
$$

A linear fit $g ( h ) \approx c + \lambda h$ also gives the horizon-growth response

$$
\Delta \lambda = \lambda _ { x } - \lambda _ { x y } .\tag{12}
$$

## 2.4 Forecast-error distribution entropy (FEDE)

FEDE describes the distribution of signed errors at each forecast horizon. For every matched window, baseline and auxiliary-informed errors are pooled to define one common histogram range $[ e _ { \mathrm { m i n } } , e _ { \mathrm { m a x } } ]$ The same $B = 1 6$ bins are then used for both branches and all horizons. If $p _ { j } ( h )$ denotes the empirical probability in bin $j ,$

$$
\operatorname { F E D E } ( h ) = - { \frac { \sum _ { j = 1 } ^ { B } p _ { j } ( h ) \log p _ { j } ( h ) } { \log B } } , \qquad 0 \leq \operatorname { F E D E } ( h ) \leq 1 .\tag{13}
$$

We use two horizon-structure summaries:

$$
\Delta \mathrm { F E D E } = \mathrm { F E D E } ( H ) - \mathrm { F E D E } ( 1 ) ,\tag{14}
$$

$$
s _ { \mathrm { F E D E } } ^ { \mathrm { e a r l y } } = \operatorname { s l o p e } ( \mathrm { F E D E } ( h ) , h = 1 , \dots , 5 ) .\tag{15}
$$

Their matched responses are

$$
D _ { \mathrm { F E D E } } = \Delta \mathrm { F E D E } _ { x y } - \Delta \mathrm { F E D E } _ { x } ,\tag{16}
$$

$$
E _ { \mathrm { F E D E } } = s _ { \mathrm { F E D E } , x y } ^ { \mathrm { e a r l y } } - s _ { \mathrm { F E D E } , x } ^ { \mathrm { e a r l y } } .\tag{17}
$$

The second quantity, E<sub>FEDE</sub>, is the FEDE early-slope change. It is particularly useful in the present linear benchmark because it combines weak-coupling sensitivity with an approximately linear calibration to the known coupling strength.

## 2.5 Prediction engines

Ridge regression is used as the linear predictive reference with $\alpha = 1 0 ^ { - 6 } ~ [ 6 ]$ . LightGBM provides a nonlinear comparison with 50 estimators, 7 leaves, maximum depth 3, learning rate 0.05, minimum child size 5, and $\ell _ { 2 }$ regularization 0.1 [7]. LightGBM uses one internal thread. The main figures use history $m = 1 ; m = 5$ is retained in the software as a sensitivity setting. The forecast horizon is $H = 1 4$ , and the chronological train/test split is 60:40.

## 2.6 Dose–response evaluation: calibration, detection, and reproducibility

The controlled benchmarks impose a known coupling strength $\gamma ,$ which provides an external calibration axis for any scalar response statistic S. In this paper, $S$ may denote a conventional forecast-gain quantity or an FEG/FEDE-derived response such as $G _ { \mathrm { l o g } }$ <sub>RMSE</sub>, $G _ { \mathrm { p r o f i l e } } , \Delta \lambda , D _ { \mathrm { F E D E } } ,$ or $E _ { \mathrm { F E D E } }$ . For a fixed noise level $\sigma .$ , let $\overline { { S } } ( \gamma , \sigma )$ denote the mean of that statistic across the independent realizations at a given coupling level. Calibration linearity is assessed from

$$
\overline { { S } } ( \gamma , \sigma ) \approx \alpha + \beta \gamma .\tag{18}
$$

The coeficient of determination $R _ { S , \gamma } ^ { 2 }$ measures how closely the condition-mean response follows a linear scale in the known imposed coupling. This complements monotonicity: a statistic can preserve rank ordering while having a strongly nonlinear response to γ.

The exact null condition is $\gamma = 0$ . For each noise level and statistic, the $\gamma = 0$ realizations define a one-sided 95th-percentile null threshold after orienting the statistic according to its dose–response direction. The detection rate at $\gamma > 0$ is the fraction of realizations exceeding that threshold. For a required sustained detection probability $p ,$ the detection limit is

$$
\gamma _ { \operatorname* { m i n } } ^ { ( p ) } ( \sigma ; S ) = \operatorname* { m i n } \left\{ \gamma _ { j } : p _ { \mathrm { d e t } } ( \gamma _ { k } , \sigma ; S ) \geq p \forall k \geq j \right\} .\tag{19}
$$

We report both $p = 0 . 8$ and $p = 0 . 9$ where the tested coupling range permits it. If no tested value satisfies Eq. (19), the result is reported as exceeding the largest tested $\gamma$

Condition-mean calibration alone does not penalize a descriptor whose response varies strongly between independent realizations. We therefore also fit the dose–response relation separately in every realization $r = 1 , \ldots , R$

$$
M _ { r } ( \gamma ) = \alpha _ { r } + \beta _ { r } \gamma ,\tag{20}
$$

where $M _ { r }$ is the realization-specific scalar predictive response. Let $\bar { \beta }$ denote the mean slope across the R independent realizations and $s _ { \beta }$ the sample standard deviation of these realization-specific slopes (estimated with the usual $R - 1$ denominator). The standard error (SE) of the mean slope is

$$
\mathrm { S E } ( \bar { \beta } ) = \frac { s _ { \beta } } { \sqrt { R } } ,\tag{21}
$$

and the signed response-sensitivity statistic is

$$
T = \frac { \bar { \beta } } { \mathrm { S E } ( \bar { \beta } ) } = \frac { \bar { \beta } } { s _ { \beta } / \sqrt { R } } .\tag{22}
$$

Its magnitude |T| quantifies the strength and reproducibility of the realization-wise dose–response slope relative to its between-realization uncertainty, while sign(T) gives the orientation of that response. Thus, a large negative $T$ is not intrinsically a low-sensitivity result: it can represent a strong and reproducible inverse response. In the present metrics, however, the orientation is defined a priori so that positive log-RMSE gain and positive FEDE changes correspond to the expected direction of useful auxiliary predictive contribution. Hence a large positive $T$ is the desired response, $T \approx 0$ indicates little reproducible dose response, and $T < 0$ indicates an anti-aligned response. For metrics whose physically meaningful direction is not defined in advance, both $T$ and $| T |$ should be reported, or the metric orientation should be specified a priori before calculating $T ;$ the sign should not be discarded post hoc. Because both numerator and denominator scale identically, $T$ is invariant to a constant positive rescaling of the metric and can therefore compare log-RMSE gain with FEDE descriptors having diferent numerical units. The 95% confidence interval for the mean slope is

$$
\bar { \beta } \pm t _ { 0 . 9 7 5 , R - 1 } \mathrm { S E } ( \bar { \beta } ) .\tag{23}
$$

Equation (22) is algebraically the one-sample t statistic for the realization-wise slopes, but here it is used descriptively as a noise-normalized sensitivity and reproducibility index rather than as a stand-alone multiple-testing significance claim.

## 2.7 Directed predictive contribution and source specificity

The matched construction is naturally directional because direction is defined by the forecast target rather than by the sign of the response. For a generic target series $u _ { t }$ and a candidate source $v _ { t } .$ let $\Delta \mathrm { F E D E } _ { u  u }$ denote the FEDE delta obtained from a target-only forecast and let $\Delta \mathrm { F E D E } _ { ( u , v ) }$ →u denote the corresponding quantity after adding the candidate source. We define the directed FEDE contribution

$$
D _ { \mathrm { F E D E } } ^ { v \to u } = \Delta \mathrm { F E D E } _ { ( u , v ) \to u } - \Delta \mathrm { F E D E } _ { u \to u } .\tag{24}
$$

Thus, testing $Y  X$ compares $X  X$ with $( X , Y ) \to X$ , whereas testing the reverse direction $X  Y$ compares $Y  Y$ with $( Y , X ) \to Y$ . A positive or negative value of $D _ { \mathrm { F E D E } } ^ { v  u }$ describes the orientation of the predictive response for the chosen direction; the sign itself does not define the direction.

For controlled specificity tests, the same candidate-source comparison can be repeated with unrelated auxiliary signals while keeping the target, prediction engine, windowing, and coupling scan unchanged. If the response is source-specific, the true driver should show a reproducible dependence on the imposed coupling strength, whereas unrelated candidate sources should remain near their null response. For each candidate direction/source, realization-wise slopes of $D _ { \mathrm { F E D E } } ^ { v  u } ( \gamma )$ are summarized by the signed statistic $T$ in Eq. (22). The resulting quantity is a test of direction-specific incremental predictability relative to a specified model class. It is not, by itself, a claim of structural or interventional causality; common drivers, feedback, omitted state variables, or model misspecification can in general generate predictive asymmetries without establishing a causal mechanism.

## 2.8 Competing directional estimators

The comparative benchmark uses each competing method in a form natural to its own construction rather than forcing every method onto the FEDE horizon axis. This distinction is important: FEDE and mean log-RMSE use the full forecast range $h = 1 , \ldots , H$ , the Granger-style controls are one-step predictive comparisons, and CCM is summarized by convergence with library size.

Chronological out-of-sample Granger-style gain. Classical Granger analysis compares a restricted model of the target from its own past with a full model that also contains the candidate source past [9]. To obtain an efect-size comparator without giving the larger full model an automatic in-sample residual advantage, we use chronological held-out scoring rather than the conventional in-sample $F$ test. In every $W = 1 0 0$ window and for history $p = 1$ , the supervised rows are split chronologically into 60% training and 40% test data. The restricted and full linear models are

$$
x _ { t + 1 } = a _ { 0 } + a _ { 1 } x _ { t } + \varepsilon _ { t + 1 } ^ { ( R ) } ,\tag{25}
$$

$$
x _ { t + 1 } = c _ { 0 } + c _ { 1 } x _ { t } + d _ { 1 } y _ { t } + \varepsilon _ { t + 1 } ^ { ( F ) } .\tag{26}
$$

Standardization parameters are fitted on the training rows only. Both models are estimated by ordinary least squares on the training subset and evaluated on the same held-out target values. The reported directional efect size is

$$
G _ { \mathrm { G C , O O S } } ^ { y  x } = \log \frac { \mathrm { M S E } _ { R } ^ { \mathrm { t e s t } } } { \mathrm { M S E } _ { F } ^ { \mathrm { t e s t } } } .\tag{27}
$$

Thus $G _ { \mathrm { G C , O O S } } > 0$ means that adding $y _ { t }$ improves unseen one-step prediction, whereas a negative value means that the additional source degrades generalization.

The nonlinear Granger-style comparator retains exactly the same train/test protocol but applies a fixed degree-2 polynomial expansion. For $p = 1$ , the restricted model receives $x _ { t }$ and $x _ { t } ^ { 2 }$ , while the full model receives

$$
x _ { t } , \ y _ { t } , \ x _ { t } ^ { 2 } , \ x _ { t } y _ { t } , \ y _ { t } ^ { 2 } .\tag{28}
$$

The degree is deliberately fixed at two for all hidden-channel forms; it is not retuned after the true nonlinearity is changed. This makes the quadratic channel a matched-basis case and the remaining channels tests of representation mismatch, consistent with the broader nonlinear Granger literature in which polynomial/Volterra and machine-learning representations define the class of interactions that can be extracted [12, 11, 10].

Convergent cross mapping (CCM). For a candidate direction $y  x ,$ convergent cross mapping (CCM) does not forecast future x from past y. Instead, it asks whether the contemporaneous source value $y _ { t }$ can be reconstructed from nearby states on the delay-coordinate manifold built from x [13]. The apparent reversal is intrinsic to cross mapping: if Y contributes to the dynamics of X, information about $Y$ may be encoded in the reconstructed state of X. In each $W = 1 0 0$ window, the delay is fixed at $\tau = 1$ . The embedding dimension $E \in \{ 1 , 2 , 3 , 4 \}$ is selected by the one-step nearest-neighbor prediction skill of the target, and source reconstruction uses the $E + 1$ nearest neighbors in the target manifold. Cross-map skill is the Pearson correlation $\rho ( L )$ between the observed source values and their nearest-neighbor reconstructions from the target manifold using library size L. Three random library draws are averaged at

$$
L \in \{ 3 0 , 5 0 , 7 0 , 9 0 \} ,\tag{29}
$$

and the scalar convergence comparator is

$$
M _ { \mathrm { C C M } } = \Delta \rho = \rho ( 9 0 ) - \rho ( 3 0 ) .\tag{30}
$$

Thus CCM is a state-space cross-reconstruction test rather than a future-forecast comparison. The reverse direction exchanges source and target. This is a deliberately basic convergence implementation: it does not include lag scanning, surrogate testing, or multivariate conditioning, and its interpretation is therefore restricted to the present controlled benchmark [14].

Table 2: Scalar quantities used in the method-comparison benchmark. Each method is evaluated on the same $W = 1 0 0$ windows and the same controlled $\gamma$ scan, but its native within-window construction is preserved.
<table><tr><td>Method</td><td>Within-window object</td><td>Scalar response</td><td>Native axis</td></tr><tr><td>FEDE  $\Delta + \mathrm { L i g h t G B M }$ </td><td>signed-error entropy profile</td><td> $D _ { \mathrm { F E D E } }$ </td><td> $h = 1 , \ldots , 1 4$ </td></tr><tr><td>Mean log-RMSE + LightGBM</td><td>forecast-error magnitude</td><td> $G _ { \mathrm { l o g } \mathrm { R M S E } }$ </td><td> $h = 1 , \ldots , 1 4$ </td></tr><tr><td>Linear Granger-style OOS</td><td>restricted vs full OLS</td><td> $\mathrm { E q . } \stackrel { \sim } { ( 2 7 ) }$ </td><td>one step</td></tr><tr><td>Polynomial Granger-style OOS</td><td>restricted vs degree-2 full OLS</td><td> $\mathrm { E q . ~ } ( 2 7 )$ </td><td>one step</td></tr><tr><td>CCM</td><td>source reconstruction from target manifold</td><td> $\rho ( 9 0 ) - \rho ( 3 0 )$ </td><td>library size</td></tr></table>

## 2.9 Time-resolved evaluation

For the intermittent nonlinear benchmark N2, the same matched forecasts are evaluated in rolling windows. For a window ending at time $\tau , \bar { \gamma } ( \tau )$ is the mean imposed coupling inside that window. Because a finite rolling window overlaps a switching boundary, $\bar { \gamma } ( \tau )$ provides the appropriate windowlevel ground truth rather than the instantaneous value at the window end.

For presentation and comparison with the synthetic nonlinear oracle defined with the nonlinear benchmark family in Section 3.4, rolling metrics are averaged across the independent realizations at each window end. If $M _ { r } ( \tau )$ is a metric and $O _ { r } ( \tau )$ is the oracle in realization r, then

$$
\overline { { M } } ( \tau ) = R ^ { - 1 } \sum _ { r } M _ { r } ( \tau ) ,\tag{31}
$$

$$
\overline { { { O } } } ( \tau ) = R ^ { - 1 } \sum _ { r } O _ { r } ( \tau ) .\tag{32}
$$

Time-resolved tracking quality is quantified by both Spearman $\rho _ { S }$ and Pearson $r _ { P }$ . We report correlations of the unsmoothed ensemble-mean response $\overline { { M } } ( \tau )$ with (i) the ensemble-mean nonlinear oracle $\overline { { O } } ( \tau )$ and (ii) the known window-mean imposed coupling $\bar { \gamma } ( \tau )$ . The second comparison is the most direct benchmark of temporal tracking because $\bar { \gamma } ( \tau )$ is known exactly by construction. Negative predictive responses are retained numerically in all calculations; in the time-domain FEDE figures they are shown with lower opacity rather than clipped. The displayed N2 curves are unsmoothed ensemble means without confidence bands so that the window-induced temporal broadening remains visible directly.

## 3 Controlled synthetic experiments

The synthetic study is organized into two deliberately distinct benchmark families. The linearreference family L1–L2 uses separately initialized, decorrelated chaotic logistic-map series and asks whether the proposed descriptors recover a known direct linear predictive contribution. The nonlinear family N1–N5 uses Gaussian AR(1) series and deliberately hides the tested source contribution behind nonlinear transformations that are not supplied to the predictors. The symbols y and q denote the tested source and an unobserved background forcing, respectively, in both families, but their generating processes are deliberately diferent. Table 3 provides the roadmap from each benchmark to its scientific purpose and corresponding results.

Table 3: Roadmap of the controlled synthetic benchmarks. L1–L2 and N1–N5 are separate generating families; they share the same matched forecasting logic but not the same source/background process.
<table><tr><td>Benchmark</td><td>Generating family</td><td>Main purpose</td><td>Results</td></tr><tr><td>L1</td><td>logistic-map y, q</td><td>calibration to known direct linear coupling</td><td>Sec. 4.1</td></tr><tr><td>L2</td><td>logistic-map y, q</td><td>weak coupling, process noise, and null-relative detection</td><td>Secs. 4.2–4.4</td></tr><tr><td>N1</td><td>Gaussian AR(1) y, q</td><td>hidden nonlinear extraction and Ridge-LightGBM comparison</td><td>Secs. 4.5–4.6</td></tr><tr><td>N2</td><td>Gaussian AR(1) y, q</td><td>time-resolved tracking of intermittent coupling</td><td>Sec. 4.9</td></tr><tr><td>N3</td><td>Gaussian AR(1) y, q</td><td>direction and source specificity</td><td>Sec. 4.7</td></tr><tr><td>N4</td><td>Gaussian AR(1) y, q</td><td>robustness to hidden functional form; Granger/CCM comparison</td><td>Secs. 4.8, 4.10</td></tr><tr><td>N5</td><td>Gaussian AR(1) y, q</td><td>robustness to W, m, and H</td><td>Sec. 4.11</td></tr></table>

## 3.1 Linear-reference benchmark family: common generating system

Benchmarks L1 and L2 use two separately initialized deterministic chaotic drivers, $y _ { n }$ and $q _ { n }$ , generated from the fully chaotic logistic map

$$
z _ { n + 1 } = 4 z _ { n } ( 1 - z _ { n } ) ,\tag{33}
$$

where z denotes either driver and the two sequences use separate initial conditions. Both are standardized after a transient. Candidate realizations are accepted only when the absolute retained correlation between the two drivers is below 0.15. The target is

$$
\begin{array} { r } { x _ { n + 1 } = 0 . 5 5 x _ { n } + 0 . 2 0 q _ { n } + 0 . 6 0 \gamma y _ { n } + \sigma \epsilon _ { n } , \qquad \epsilon _ { n } \sim \mathcal { N } ( 0 , 1 ) . } \end{array}\tag{34}
$$

Here $y _ { n }$ is the tested auxiliary source available to the augmented predictor, whereas $q _ { n }$ is an independent unobserved background chaotic forcing that contributes target variability unavailable to either forecasting branch. The parameter γ is the known strength of the added direct linear predictive channel $y _ { n } \to x _ { n + 1 }$ . This family therefore provides a reference case in which the injected source contribution is linear and Ridge is a natural baseline. It is used to ask three elementary questions before introducing hidden nonlinear structure: whether a response is calibrated to known $\gamma ,$ how early weak coupling separates from the exact null $\gamma = 0$ , and how process noise changes those properties.

## 3.2 Benchmark L1: stationary linear calibration

Benchmark L1 uses $\sigma = 0$ , 5000 retained samples per realization, 10 independent realizations, $W = 1 0 0$ ， step 100, and

$$
\gamma \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 3 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 3 0 , 0 . 5 0 , 0 . 7 0 , 1 . 0 0 \} .\tag{35}
$$

Each run has a constant $\gamma .$ . The benchmark therefore asks how each predictive metric changes as the known linear coupling strength is increased across otherwise comparable runs. Calibration linearity is evaluated with Eq. (18). The corresponding results are reported in Section 4.1.

## 3.3 Benchmark L2: weak linear coupling under process noise

Benchmark L2 uses 30 independent realizations, 3000 retained samples per realization, $W = 1 0 0$ , step 100,

$$
\gamma \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 3 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 3 0 \} ,\tag{36}
$$

and

$$
\sigma \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 4 0 , 0 . 8 0 \} .\tag{37}
$$

For every fixed realization and noise level, the same logistic-map y, logistic-map $q ,$ and Gaussian process-noise vector are reused across all $\gamma$ values. This paired design reduces unrelated variation

in the dose–response comparison. Benchmark L2 is used for the weak-coupling zoom, the noisedependent calibration study, and the null-relative detection limit in Eq. (19). Results are reported in Sections 4.2–4.4.

Table 4: Main settings of the linear-reference benchmarks L1 and L2.
<table><tr><td>Setting</td><td>Benchmark L1</td><td>Benchmark L2</td></tr><tr><td>Purpose</td><td>calibration</td><td>weak coupling + noise</td></tr><tr><td>Driver family</td><td>logistic map</td><td>logistic map</td></tr><tr><td>Realizations</td><td>10</td><td>30</td></tr><tr><td>Retained samples</td><td>5000</td><td>3000</td></tr><tr><td>Window length W</td><td>100</td><td>100</td></tr><tr><td>Forecast horizon H</td><td>14</td><td>14</td></tr><tr><td>Main history m</td><td>1</td><td>1</td></tr><tr><td>Coupling γ</td><td>0-1.0</td><td>0-0.30</td></tr><tr><td>Noise σ</td><td>0</td><td>0-0.80</td></tr><tr><td>Predictors</td><td>Ridge, LightGBM</td><td>Ridge, LightGBM</td></tr></table>

## 3.4 Nonlinear benchmark family: common Gaussian-AR(1) generating system

Benchmarks N1–N5 use a diferent generating family from L1–L2. The tested auxiliary source $y _ { t }$ is a stationary zero-mean, unit-variance Gaussian $\mathrm { A R } ( 1 )$ process,

$$
y _ { t + 1 } = \varphi _ { y } y _ { t } + \sqrt { 1 - \varphi _ { y } ^ { 2 } } \eta _ { t } ^ { ( y ) } , \qquad \varphi _ { y } = 0 . 6 0 ,\tag{38}
$$

and the target is also driven by an independent unobserved background Gaussian AR(1) forcing,

$$
q _ { t + 1 } = \varphi _ { q } q _ { t } + \sqrt { 1 - \varphi _ { q } ^ { 2 } } \eta _ { t } ^ { ( q ) } , \qquad \varphi _ { q } = 0 . 3 5 ,\tag{39}
$$

with independent standard-normal innovations. The innovation sequences $\eta _ { t } ^ { ( y ) }$ and $\eta _ { t } ^ { \left( q \right) }$ generate the stochastic $\mathrm { A R } ( 1 )$ source and background processes themselves; they are distinct from the independent target process noise $\epsilon _ { t }$ in Eq. (41), whose contribution is explicitly controlled by $\sigma .$ . Realizations are regenerated until the retained absolute correlation $| r ( y , q ) |$ is at most 0.15.

The term nonlinear benchmark refers to the hidden $Y  X$ interaction, not to the marginal source processes: $y _ { t }$ and $q _ { t }$ are linear Gaussian $\mathrm { A R } ( 1 )$ processes, whereas the tested predictive channel enters through a nonlinear transformation of $y _ { t }$ . The tested contribution is intentionally hidden from ordinary raw linear association. Rather than entering the target through $y _ { t }$ itself, the source acts through the normalized even transformation

$$
\phi _ { 2 } ( y ) = \frac { y ^ { 2 } - 1 } { \sqrt { 2 } } .\tag{40}
$$

For a standard Gaussian variable, $\phi _ { 2 } ( y )$ has zero mean and unit variance and is orthogonal to the raw first-order component $y$ in the population, so $\mathrm { C o r r } [ y , \phi _ { 2 } ( y ) ] = 0$ theoretically. To preserve this property in finite records, candidate realizations are additionally required to satisfy $| r ( y , \phi _ { 2 } ( y ) ) | \leq 0 . 0 5$ The common stationary nonlinear target is

$$
x _ { t + 1 } = 0 . 5 5 x _ { t } + 0 . 2 0 q _ { t } + 0 . 6 0 \gamma \phi _ { 2 } ( y _ { t } ) + \sigma \epsilon _ { t } , \qquad \epsilon _ { t } \sim \mathcal { N } ( 0 , 1 ) .\tag{41}
$$

Thus the nonlinear benchmarks contain a genuine, controlled predictive channel $Y  X$ , but its imposed contribution is deliberately dificult to detect from the raw linear association between y<sub>t</sub> and future x. Crucially, neither predictor is given $\phi _ { 2 } ( y _ { t } )$ : Ridge and LightGBM receive only the raw history of $y _ { t }$ in the augmented branch. Neither branch receives $q _ { t } , \eta _ { t } ^ { ( y ) } , \eta _ { t } ^ { ( q ) }$ , or $\epsilon _ { t }$ . The comparison therefore tests whether the chosen prediction engine can extract the hidden nonlinear information from raw $y$ before FEDE summarizes how that information changes the forecast-error distribution across horizons. The background process $q _ { t }$ serves only as independent temporally structured forcing of the target.

Because the generating mechanism is known here, a diagnostic synthetic oracle is available:

$$
O = r ( \phi _ { 2 } ( y _ { t } ) , x _ { t + 1 } ) ,\tag{42}
$$

computed on the same window as the predictive metrics. The corresponding raw diagnostic is $r ( y _ { t } , x _ { t + 1 } )$ . The oracle is not an estimator intended for real data and is not a competitor to FEDE; it is a validation device that checks whether the imposed nonlinear source contribution is present even when raw linear association is suppressed.

## 3.5 Benchmark N1: stationary hidden nonlinear coupling

Benchmark N1 uses the common nonlinear system in Eqs. (38)–(41) with 20 independent realizations, 3000 retained samples, $W = 1 0 0$ , stationary step 100, $H = 1 4$ , and histories $m \in \{ 1 , 5 \}$ . The coupling grid is

$$
\gamma \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 3 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 3 0 , 0 . 5 0 , 0 . 7 0 , 1 . 0 0 \} ,\tag{43}
$$

and the noise grid is

$$
\sigma \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \} .\tag{44}
$$

For a fixed realization and noise level, the same AR(1) source $y ,$ background forcing $q ,$ and Gaussian noise realization are reused across all $\gamma$ values. Thus the dose–response comparison is paired with respect to the stochastic forcing. Benchmark N1 first verifies that the oracle grows with $\gamma$ while the raw correlation remains near zero, and then compares whether Ridge and LightGBM can extract the hidden channel from raw $y .$ Results are reported in Sections 4.5–4.6.

## 3.6 Benchmark N2: intermittent hidden nonlinear coupling

Benchmark N2 uses the same Gaussian-AR(1) generating system and hidden quadratic channel, but the true coupling follows the rectangular schedule

$$
\gamma ( t ) : \quad 0 \to 0 . 0 5 \to 0 \to 0 . 1 0 \to 0 \to 0 . 2 0 \to 0 \to 0 . 4 0 .\tag{45}
$$

Each of the eight blocks lasts 1000 samples, giving 8000 retained samples in total. Ten independent realizations are evaluated at $\sigma = 0 . 1 0$ , history $m = 1$ , maximum forecast horizon $H = 1 4$ , and rolling step 20. Two window lengths are analyzed on the same realizations: $W = 2 0 0$ and W = 400. The $W = 4 0 0$ result is used for the main-text visualization because the longer window gives a more stable response estimate, while the $W = 2 0 0$ result is retained in the Appendix as a highertemporal-resolution comparison. The exact rectangular schedule is shown as the ground truth in the time-domain figures; rolling oracle and predictive quantities remain window-based and therefore broaden transitions over the finite temporal support of each window. Benchmark N2 therefore tests not only whether the hidden nonlinear contribution can be followed in time, but also the practical stability–temporal-resolution trade-of of the rolling analysis. Results are reported in Section 4.9.

Table 5: Settings of the stationary and intermittent hidden nonlinear benchmarks N1 and N2.
<table><tr><td>Setting</td><td>N1 stationary</td><td>N2 intermittent</td></tr><tr><td>Tested source</td><td>Gaussian  $\mathrm { A R } ( 1 ) , \varphi _ { y } = 0 . 6 0$ </td><td>same</td></tr><tr><td>Background forcing</td><td>Gaussian  $\mathrm { A R } ( 1 ) , \varphi _ { q } = 0 . 3 5$ </td><td>same</td></tr><tr><td>Hidden transform</td><td> $\phi _ { 2 } ( y ) = ( y ^ { 2 } - 1 ) / \sqrt { 2 }$ </td><td>same</td></tr><tr><td>Predictor input</td><td> $\operatorname { r a w } y \ \mathrm { o n l y }$ </td><td>raw y only</td></tr><tr><td>Realizations</td><td>20</td><td>10</td></tr><tr><td>Retained samples</td><td>3000</td><td>8000</td></tr><tr><td>Window length W</td><td>100</td><td>200, 400</td></tr><tr><td>Rolling step</td><td>100</td><td>20</td></tr><tr><td>Forecast horizon H</td><td>14</td><td>14</td></tr><tr><td>Histories</td><td>1, 5</td><td>1</td></tr><tr><td>Noise σ</td><td>0, 0.05, 0.10, 0.20</td><td>0.10</td></tr><tr><td>Coupling</td><td>13 stationary levels, 0–1</td><td>Eq. (45); 1000 samples/level</td></tr></table>

## 3.7 Benchmark N3: directional and source specificity

Benchmark N3 retains the N1 stationary Gaussian-AR(1) generating system, the full 13-point γ grid, $\sigma \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \}$ , 20 independent realizations, 3000 retained samples, $W = 1 0 0$ , step 100, $H = 1 4$ , and history $m = 1$ . Only LightGBM is used because the preceding N1 analysis identifies the nonlinear model with FEDE ∆ as the strongest overall positively aligned response to the hidden channel.

The primary comparison is the true nonlinear direction $Y  X$ , evaluated by Eq. (24). The reverse test exchanges source and target and evaluates $X  Y$ with the same forecasting construction. Four independent candidate sources are additionally supplied to the X forecast: (i) a sinusoid with an independent random phase, (ii) a quasiperiodic signal formed from two incommensurate sinusoidal components, (iii) a deterministic logistic-map signal at $r = 4$ after a 500-sample burn-in, and (iv) an independent Gaussian AR(1) signal with coeficient 0.60, equal to the memory coeficient of the true source Y. Each control is standardized and generated independently for every realization, but is held fixed across the $\gamma$ and σ scans within that realization. This design makes the $\operatorname { A R } ( 1 )$ control particularly stringent: it matches the linear temporal memory of the true source while containing none of its imposed predictive contribution to X.

For every source/direction, $D _ { \mathrm { F E D E } } ^ { v  u }$ is averaged over the 30 stationary windows of each condition. A linear slope versus the complete γ grid is then fitted separately for each realization, and the 20 slopes are summarized by T and by a 95% confidence interval for the mean slope. Because positive and negative control responses could otherwise cancel across noise levels, source specificity is summarized across σ using mean |T| in addition to the signed T values. Benchmark N3 results are reported in Section 4.7.

## 3.8 Benchmark N4: functional-form method comparison

Benchmark N4 asks whether the response observed for N1 depends on the particular quadratic mapping ϕ<sub>2</sub>. The N1 quadratic channel in Eq. (40) is retained as one member of the benchmark, and three additional normalized channels are introduced:

$$
\begin{array} { l } { \displaystyle \phi _ { 3 } ( \boldsymbol { y } ) = \frac { y ^ { 3 } - 3 y } { \sqrt { 6 } } , } \\ { \displaystyle \phi _ { 4 } ( \boldsymbol { y } ) = \frac { y ^ { 4 } - 6 y ^ { 2 } + 3 } { \sqrt { 2 4 } } , } \\ { \displaystyle \phi _ { \mathrm { s i n } } ( \boldsymbol { y } ) = \mathrm { s t a n d a r d i z e } \big [ \mathrm { s i n } ( y ^ { 2 } ) \big ] . } \end{array}\tag{46}
$$

The three polynomial channels $\phi _ { 2 } , \phi _ { 3 }$ , and $\phi _ { 4 }$ are normalized probabilists’ Hermite components for a Gaussian variable and are orthogonal to lower Gaussian orders in the population; $\phi _ { \mathrm { s i n } }$ is nonpolynomial and even. For every retained realization, each channel is additionally centered and rescaled to exact sample mean zero and sample standard deviation one, so that $\gamma$ multiplies equal finite-record RMS channel power. Candidate realizations are rejected if $| r ( y , q ) | > 0 . 1 5$ or if the finite-sample raw correlation $| r ( y , \phi _ { k } ( y ) ) | > 0 . 0 5$ for any of the four channels. The same $y$ and $q$ trajectories, initial target state, and noise realization are reused across channel forms within a realization and noise level; only the hidden mapping changes. Predictors always receive raw $y _ { t }$ , never $\phi _ { k } ( y _ { t } )$

The remaining protocol preserves N1: 20 realizations, 3000 retained samples, $W = 1 0 0$ , step 100, history $m = 1$ , the 13-point coupling grid from 0 to 1, and

$$
\sigma \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \} .\tag{47}
$$

For FEDE and mean log-RMSE, LightGBM uses the same fixed hyperparameters as in N1 and produces matched forecasts for $h = 1 , \ldots , 1 4$ . Linear and degree-2 polynomial Granger-style scores use the chronological 60:40 out-of-sample protocol in Eq. (27). CCM uses Eq. (30). Both the true $Y  X$ and reverse $X  Y$ directions are calculated.

For every method, channel, direction, noise level, and realization, the window-level scalar is averaged over the 30 non-overlapping windows. A realization-wise slope across the same 13-point $\gamma \ { \mathrm { g r i d } }$ is then fitted and converted to the common external sensitivity statistic $T$ in Eq. (22). Consequently, $T$ is not specific to FEDE: it is the common calibration scale used to compare how reproducibly each method responds to the known change in coupling. Signed $T$ is retained for all methods. Positive orientation is defined a priori as increasing predictive contribution with increasing $\gamma ;$ a negative $T$ therefore denotes a reproducible anti-aligned response. Benchmark N4 method-comparison and timing results are reported in Sections 4.8 and 4.10.

## 3.9 Benchmark N5: FEDE parameter robustness

The final synthetic benchmark tests whether the FEDE response depends on the nominal analysis scales used in the preceding experiments. The four hidden channels—ϕ from Eq. (40) and the three alternatives in Eq. (46)—are evaluated at fixed $\sigma = 0 . 1 0$ , with 20 realizations, 3000 retained samples, the same 13-point coupling grid, the same LightGBM hyperparameters, and the true $Y  X$ direction only. The predefined grid is

$$
W \in \{ 1 0 0 , 2 0 0 , 4 0 0 \} , \qquad m \in \{ 1 , 3 , 5 \} , \qquad H \in \{ 5 , 1 4 , 2 0 \} ,\tag{48}
$$

which gives $2 7$ configurations per hidden channel and 108 FEDE $\Delta$ response estimates in total. Rolling step is set equal to $W .$ , so the local windows do not overlap. Each value of H is fitted and evaluated independently rather than being truncated from a longer precomputed forecast profile. Within every configuration, the realization-wise FEDE slope across $\gamma$ is converted to $T$ using Eq. (22). The origina setting W = 100, m = 1, H = 14 is retained as the primary reference and is not re-selected after inspecting the grid. Benchmark N5 results are reported in Section 4.11.

## 4 Results

## 4.1 Benchmark L1: calibration to the known coupling strength

Benchmark L1 (Section 3.2) provides the linear-reference calibration. Figure 2 compares the normalized dose–response shapes at zero process noise. The dashed diagonal represents an ideal linear response to $\gamma .$ Ridge provides a particularly clear linear reference. Among its FEG-based descriptors, FEDE delta and FEDE early-slope change follow the known coupling closely over most of the range, whereas FEG profile gain and ∆λ show stronger departures from linearity at weak coupling. LightGBM gives a smooth monotone response but is not expected to outperform a linear predictor in a system whose injected auxiliary term is itself linear.

![](images/66d65fb9cc8354872ab689f5531392abdedef0ba83c823cc4d97decf3f026f98.jpg)  
Figure 2: Stationary linear-coupling calibration at $\sigma = 0$ . Each metric is endpoint-normalized to emphasize response shape rather than numerical scale. The dashed line is the ideal linear response.

The calibration $R ^ { 2 }$ values in Table 6 make this comparison quantitative. At $\sigma = 0 ,$ , Ridge FEDE delta and FEDE early-slope changes both achieve $R ^ { 2 } \approx 0 . 9 9$ . As noise increases, the Ridge early-slope descriptor remains close to linear over a broad range: $R ^ { 2 } = 0 . 9 3 4$ at $\sigma = 0 . 2 0$ and 0.862 at $\sigma = 0 . 8 0$ Conventional Ridge log-RMSE gain is slightly less linear at low noise but more stable at the highest noise level. Thus FEDE early slope is not a universal high-noise replacement for conventional error gain; its main advantage is the combination of low-noise calibration and weak-coupling sensitivity.

Table 6: Linearity of the metric-versus-coupling calibration, measured by $R ^ { 2 }$ of a linear fit to the condition means. Selected history-1 metrics are shown.
<table><tr><td>Predictor</td><td>Metric</td><td> $\sigma = 0$ </td><td> $\sigma = 0 . 2 0$ </td><td> $\sigma = 0 . 8 0$ </td></tr><tr><td>Ridge</td><td>mean log-RMSE gain</td><td>0.952</td><td>0.941</td><td>0.920</td></tr><tr><td>Ridge</td><td>FEDE delta change</td><td>0.995</td><td>0.910</td><td>0.832</td></tr><tr><td>Ridge</td><td>FEDE early-slope change</td><td>0.994</td><td>0.934</td><td>0.862</td></tr><tr><td>LightGBM</td><td>mean log-RMSE gain</td><td>0.964</td><td>0.933</td><td>0.887</td></tr><tr><td>LightGBM</td><td>FEDE delta change</td><td>0.984</td><td>0.904</td><td>0.545</td></tr><tr><td>LightGBM</td><td>FEDE early-slope change</td><td>0.986</td><td>0.923</td><td>0.365</td></tr></table>

## 4.2 Benchmark L2: weak-coupling response at zero process noise

Benchmark L2 (Section 3.3) targets weak linear coupling under process noise. Figure 3 magnifies the regime $0 \leq \gamma \leq 0 . 2 0$ . This view is central to the present study because a useful predictive-coupling descriptor should react before the auxiliary contribution becomes large. For Ridge, both FEDE delta and FEDE early-slope changes depart from their null levels at smaller γ than the conventional mean log-RMSE gain. The FEDE delta response has the largest amplitude, while the early-slope response is smoother and remains close to a calibrated linear trend. By contrast, the Ridge FEG profile gain and $\Delta \lambda$ are weak in this low-coupling range.

![](images/7fe06c1cd63263d66ba71d7e6c64f24a89cf1f30f73d5e8ff22b6e9f720eaf49.jpg)  
Figure 3: Weak-coupling zoom at zero process noise. Lines show condition means and shaded bands show between-realization standard deviations. The Ridge FEDE descriptors respond to weak auxiliary coupling before a stable conventional log-RMSE gain appears.

## 4.3 Benchmark L2: efect of process noise on calibration linearity

Continuing benchmark L2, Fig. 4 shows how the linear calibration degrades as σ increases. For Ridge, FEDE early-slope change stays near or above $R ^ { 2 } \approx 0 . 9$ through $\sigma = 0 . 4 0$ , while FEG profile gain deteriorates much earlier. The conventional Ridge log-RMSE gain remains highly linear across the full noise grid. LightGBM FEDE responses lose calibration more rapidly at high noise, whereas its mean log-RMSE gain remains comparatively stable. These results support Ridge as the natural reference predictor for the present linear system and FEDE early-slope change as the preferred FEG-derived descriptor for subsequent comparison.

![](images/4972e19a7ee66dd461a2f9fe587c04b018f88ae63a6ffe4296882fd9ea34a904.jpg)  
Figure 4: Linearity of metric-versus-γ calibration as process noise increases. The dashed line marks $R ^ { 2 } = 0 . 9 5$

## 4.4 Benchmark L2: minimum detectable predictive coupling

The weak-coupling experiment provides a second criterion: not just whether a metric is linear in γ, but how small a coupling can be separated from the exact $\gamma = 0$ null. Figure 5 and Table 7 summarize the sustained 80% detection threshold.

At zero noise, Ridge mean log-RMSE gain requires $\gamma = 0 . 1 5$ , whereas both Ridge FEDE delta and FEDE early-slope change detect $\gamma = 0 . 0 5$ . At $\sigma = 0 . 0 5$ , the FEDE thresholds are 0.075, still a factor of two below the conventional Ridge threshold of 0.15. At $\sigma = 0 . 1 0$ the diference disappears: all three reach $\gamma = 0 . 1 5$ . For larger noise, conventional log-RMSE gain is equal or more sensitive. LightGBM does not improve this linear benchmark; its FEDE early-slope threshold is 0.10 at zero noise and 0.15 at $\sigma = 0 . 0 5$

![](images/478ca2e992ff6d3e5dcdecb4b56aa2ed19be5d78f817c116366cad456b36c675.jpg)  
Figure 5: Minimum coupling strength required for sustained 80% null-relative detection using predictive metrics only. Missing points indicate that the criterion was not reached within the tested range $\gamma \leq 0 . 3 0$

Table 7: Sustained 80% detection limit $\gamma _ { \mathrm { m i n } }$ for the main history-1 predictive metrics. $> 0 . 3 0$ means that the criterion was not reached on the tested grid.
<table><tr><td>Predictor</td><td>Metric</td><td>0</td><td>0.05</td><td>0.10</td><td>0.20</td><td>0.40</td><td>0.80</td></tr><tr><td>Ridge</td><td>mean log-RMSE gain</td><td>0.150</td><td>0.150</td><td>0.150</td><td>0.150</td><td>0.300</td><td>&gt; 0.30</td></tr><tr><td>Ridge</td><td>FEDE delta change</td><td>0.050</td><td>0.075</td><td>0.150</td><td>0.200</td><td>0.300</td><td>&gt; 0.30</td></tr><tr><td>Ridge</td><td>FEDE early-slope change</td><td>0.050</td><td>0.075</td><td>0.150</td><td>0.200</td><td>0.300</td><td>&gt; 0.30</td></tr><tr><td>LightGBM</td><td>mean log-RMSE gain</td><td>0.150</td><td>0.150</td><td>0.150</td><td>0.200</td><td>0.300</td><td>&gt; 0.30</td></tr><tr><td>LightGBM</td><td>FEDE early-slope change</td><td>0.100</td><td>0.150</td><td>0.200</td><td>0.300</td><td>&gt; 0.30</td><td>&gt; 0.30</td></tr></table>

The two result criteria therefore give a consistent picture. In the controlled linear regime, Ridge is the appropriate predictive reference. Within the Ridge error structure, FEDE early-slope change is a strong candidate descriptor because it is approximately proportional to coupling strength over a broad low-noise range and reaches the same favorable weak-coupling detection threshold as FEDE delta. FEDE delta remains an important companion descriptor because its low-coupling amplitude is larger. Neither descriptor should be interpreted as universally superior under strong process noise.

## 4.5 Benchmark N1: hidden nonlinear channel validation

Benchmark N1 (Section 3.5) was designed so that a direct raw linear association cannot reveal the imposed auxiliary channel. Figure 6 verifies this construction. Across all tested $\gamma ,$ the mean one-step raw Pearson correlation $r ( y _ { t } , x _ { t + 1 } )$ remains close to zero. At the strongest coupling $\gamma = 1$ , the mean raw correlation is only 0.0079 at $\sigma = 0$ and 0.0082 at $\sigma = 0 . 2 0$ . In contrast, the synthetic oracle $r ( \phi _ { 2 } ( y _ { t } ) , x _ { t + 1 } )$ rises monotonically with coupling and reaches 0.792 at $\sigma = 0 , 0 . 7 8 4$ at $\sigma = 0 . 1 0$ , and 0.761 at $\sigma = 0 . 2 0$ . Thus the predictive channel is strong in the generating coordinates while remaining almost invisible to ordinary linear association with raw y.

![](images/8d3a6ee407e44d53dc285a153bc7ab5e3075f0d847ba77f0978bdf935fede75e.jpg)

![](images/68338192ccb3ee68244372626a5431c42bdacf155409687b14371e523771d029.jpg)  
Figure 6: Validation of the hidden nonlinear channel. Lines are means over 20 realizations and shaded bands are 95% confidence intervals of the mean. (a) The raw one-step Pearson association between y<sub>t</sub> and $x _ { t + 1 }$ remains near zero. (b) The synthetic oracle based on the known hidden transform $\phi ( y _ { t } )$ increases strongly with the imposed coupling γ.

## 4.6 Benchmark N1: nonlinear extraction—LightGBM versus Ridge

Within benchmark N1 (Section 3.5), Fig. 7 compares the two predictor classes under moderate process noise, $\sigma = 0 . 1 0$ , using history $m = 1$ . This comparison is intentionally not made in the ideal noiseless case. The conventional mean log-RMSE gain improves gradually for LightGBM and becomes positive only at strong coupling, whereas the Ridge response becomes slightly more negative as $\gamma$ increases. The FEDE responses separate the predictor classes more clearly. LightGBM FEDE delta increases from $- 0 . 0 0 2 7$ at $\gamma = 0$ to 0.0905 at $\gamma = 1$ , while Ridge remains near zero over the same range. LightGBM FEDE early-slope change similarly rises from approximately zero to 0.0210, while Ridge fluctuates around zero. Because neither predictor receives $\phi _ { 2 } ( y )$ explicitly, the diference isolates the ability of the nonlinear prediction engine to extract the hidden even dependence from raw y.

Nonlinear specificity: LightGBM vs Ridge (σ = 0.1, history=1)  
![](images/66b1e76866d3c595ea4dad034fc8a5b25647be3750eab190e0517519cc652df6.jpg)

![](images/a1e4ccf686e90d299029915d51a20cb3a9c686abe6f271395e8cebfc24eabc05.jpg)

(c)  
![](images/8d3d14e62acc866d12c053a44d7f2d9bc632a0ba2b468f334ed4262684d31495.jpg)  
Figure 7: Nonlinear specificity at $\sigma = 0 . 1 0$ and history $m = 1$ . Each point is the mean of 20 realization-level values; each realization value is itself averaged over 30 non-overlapping $W = 1 0 0$ windows. Shaded bands are 95% confidence intervals of the mean across realizations. LightGBM develops a systematic coupling-dependent response, whereas Ridge remains close to its null FEDE response.

The response-sensitivity statistic T makes the diference between metric amplitude and metric reproducibility explicit. Table 8 reports $T$ for the three principal LightGBM descriptors over the complete stationary coupling grid. FEDE delta has the largest mean $T$ across the noise grid (26.71), closely followed by FEDE early-slope change (26.53); mean log-RMSE gain is substantially lower (14.92). $\mathrm { A t } ~ \sigma = 0 . 1 0$ , the realization-wise mean slopes and 95% confidence intervals are 0.0379 [0.0321, 0.0437] for log-RMSE gain, 0.0981 [0.0902, 0.1059] for FEDE delta, and 0.0230 [0.0210, 0.0250] for FEDE early slope. The corresponding T values are 13.74, 26.24, and 24.24. Thus the two FEDE descriptors provide roughly twice the standardized dose-response strength of mean log-RMSE gain, despite having diferent numerical scales.

Table 8: Noise-normalized response sensitivity $T$ for LightGBM, history $m = 1$ , in the stationary hidden nonlinear benchmark. For each realization, a linear slope is fitted across the full 13-point γ grid; T is then computed from Eq. (22). The second column gives the mean slope and its 95% confidence interval at $\underline { { \sigma = 0 . 1 0 } } .$
<table><tr><td>Metric</td><td> $\bar { \beta }$  at  $\sigma = 0 . 1 0$ </td><td>[95% CI]</td><td> $T _ { 0 }$ </td><td> $T _ { 0 . 0 5 }$ </td><td> $T _ { 0 . 1 0 }$ </td><td> $T _ { 0 . 2 0 }$ </td><td>mean  $_ T$ </td></tr><tr><td>Mean log-RMSE gain</td><td></td><td>0.0379 [0.0321, 0.0437]</td><td>17.84</td><td>15.58</td><td>13.74</td><td>12.52</td><td>14.92</td></tr><tr><td>FEDE Δ change</td><td></td><td>0.0981 [0.0902, 0.1059]</td><td>27.04</td><td>24.50</td><td>26.24</td><td>29.06</td><td>26.71</td></tr><tr><td>FEDE early-slope change</td><td></td><td>0.0230 [0.0210, 0.0250]</td><td>25.87</td><td>29.80</td><td>24.24</td><td>26.21</td><td>26.53</td></tr></table>

The same calculation applied to Ridge provides a direct specificity control (Table 9). All three Ridge responses are negative across the tested noise grid. The FEDE values have small magnitudes, with mean $T = - 0 . 7 7$ for FEDE delta and $T = - 1 . 1 4$ for FEDE early slope, consistent with little reproducible monotone sensitivity to the hidden nonlinear channel. Mean log-RMSE gain shows a more systematic inverse response, with mean $T = - 3 . 4 7$ . At $\sigma = 0 . 1 0$ , its realization-wise mean slope is $- 0 . 0 0 8 1 5$ with 95% CI $[ - 0 . 0 1 2 5 3 , - 0 . 0 0 3 7 7 ]$ and $T = - 3 . 6 5$ . Because positive log-RMSE gain is defined as improved prediction after adding y, this negative response means that the raw auxiliary variable increasingly worsens the linear Ridge forecast as the strength of the hidden quadratic channel increases. It is therefore a reproducible anti-aligned response, not successful extraction of the nonlinear predictive contribution. The two Ridge FEDE slopes at $\sigma = 0 . 1 0$ have confidence intervals spanning zero, in agreement with their |T| values near one.

Table 9: Signed response sensitivity $T$ for Ridge, history $m = 1$ , in the stationary hidden nonlinear benchmark. Negative $T$ denotes a dose response opposite to the predefined positive orientation of the metric. A large $| T |$ would still indicate reproducible sensitivity, but in the inverse direction; values near zero indicate little reproducible response.
<table><tr><td>Metric</td><td> $T _ { 0 }$ </td><td> $T _ { 0 . 0 5 }$ </td><td> $T _ { 0 . 1 0 }$ </td><td> $T _ { 0 . 2 0 }$ </td><td>mean  $_ T$ </td></tr><tr><td>Mean log-RMSE gain</td><td>-3.37</td><td>-3.56</td><td>-3.65</td><td>-3.32</td><td>-3.47</td></tr><tr><td>FEDE ∆ change</td><td>-0.07</td><td>-1.20</td><td>-0.97</td><td>-0.86</td><td>-0.77</td></tr><tr><td>FEDE early-slope change</td><td>-1.73</td><td>-1.34</td><td>-1.19</td><td>-0.29</td><td>-1.14</td></tr></table>

For comparison, the mean signed T values of LightGBM are 14.92, 26.71, and 26.53 for mean log-RMSE gain, FEDE delta, and FEDE early slope, respectively. Relative to Ridge, the nonlinear model therefore changes not only the magnitude but also the direction of the calibrated response. This distinction is important when interpreting T: |T| answers how reproducibly a metric changes with coupling, whereas the sign answers whether that change is aligned with the metric’s predefined interpretation.

The null-relative detection analysis gives a complementary criterion. At $\sigma = 0 . 1 0$ , sustained 90% detection is reached at $\gamma = 0 . 3 0$ for both FEDE delta and FEDE early slope, compared with $\gamma = 0 . 5 0$ for mean log-RMSE gain. $\mathrm { A t } ~ \sigma = 0$ , the corresponding thresholds are $0 . 3 0 , 0 . 3 0$ , and 0.50. $\mathrm { A t } ~ \sigma = 0 . 2 0$ , all three require stronger coupling, with the 90% criterion reached at $\gamma = 0 . 5 0$ . Figure 8 summarizes both the 80% and 90% criteria.

![](images/7de1305e207b5cb0227d9b9d20242b0a15407169fa982b3f75d5b50f956aa09c.jpg)

![](images/b8554558684c0f3f23610ed9e9c71288e6ab4f78b9285a3944d99f66d1779ff9.jpg)  
Figure 8: Minimum hidden nonlinear coupling required for sustained $8 0 \%$ and $9 0 \%$ null-relative detection with LightGBM, history $m = 1$ . The FEDE descriptors reduce the strict 90% threshold relative to mean log-RMSE gain in several low-to-moderate noise cases.

## 4.7 Benchmark N3: directional and source specificity

The preceding stationary results establish that LightGBM can extract the hidden nonlinear channel and that FEDE $\Delta$ provides the largest overall noise-normalized response among the tested scalar descriptors. Benchmark N3 asks a stricter question: does the same response identify the correct candidate source and forecast direction, or does it also increase when an unrelated structured series is supplied to the augmented predictor?

The separation is pronounced (Table 10). For the true $Y  X$ channel, T remains between 24.50 and 29.06 over all four noise levels, with mean $| T | = 2 6 . 7 1$ . Reversing the forecasting direction to $X  Y$ reduces the response to $T = 0 . 0 3 – 1 . 2 3$ and mean $| T | = 0 . 6 0 .$ The independent periodic, quasiperiodic, chaotic, and $\mathrm { A R } ( 1 )$ sources likewise remain close to the null response. Across all reverse and independent controls, the largest absolute value is $| T | = 2 . 4 9$ , observed for the sinusoidal control at $\sigma = 0 . 2 0$ . Consequently, even the smallest true-driver response, $T = 2 4 . 5 0$ , exceeds the largest individual control response by a factor of 9.8. On a noise-averaged absolute scale, the true-driver mean $| T | = 2 6 . 7 1$ is approximately 24.8 times the largest control mean, 1.08 for the sinusoidal source.

Table 10: Directional and source specificity of the LightGBM FEDE $\Delta$ response. $T$ is the signed response-sensitivity statistic obtained from 20 realization-wise slopes across the complete coupling grid. The final column reports mean $| T |$ across noise levels so that positive and negative control responses cannot cancel.
<table><tr><td>Candidate source / direction</td><td> $T _ { 0 }$ </td><td> $T _ { 0 . 0 5 }$ </td><td> $T _ { 0 . 1 0 }$ </td><td> $T _ { 0 . 2 0 }$ </td><td>mean  $| T |$ </td></tr><tr><td> $Y  X$  (true nonlinear driver)</td><td>27.04</td><td>24.50</td><td>26.24</td><td>29.06</td><td>26.71</td></tr><tr><td> $X  Y$  (reverse)</td><td>0.03</td><td>0.99</td><td>1.23</td><td>0.15</td><td>0.60</td></tr><tr><td>Sinusoid → X</td><td>0.34</td><td>0.50</td><td>0.98</td><td>2.49</td><td>1.08</td></tr><tr><td>Quasiperiodic  $ X$ </td><td>1.01</td><td>-0.07</td><td>0.26</td><td>0.59</td><td>0.48</td></tr><tr><td>Logistic chaos → X</td><td>-0.55</td><td>1.21</td><td>0.42</td><td>0.29</td><td>0.62</td></tr><tr><td>Independent  $\operatorname { A R } ( 1 ) \to X$ </td><td>0.20</td><td>-0.30</td><td>0.96</td><td>0.11</td><td>0.39</td></tr></table>

The realization-level slopes support the same interpretation. $\mathrm { A t } ~ \sigma = 0 . 1 0$ , the true $Y  X$ response has $\bar { \beta } = 0 . 0 9 8 1$ with 95% CI $[ 0 . 0 9 0 7 , 0 . 1 0 5 4 ]$ and $T = 2 6 . 2 4$ . For the reverse $X  Y$ test, the corresponding mean slope is only $\bar { \beta } = 0 . 0 0 2 5 5$ with 95% CI $[ - 0 . 0 0 1 5 1 , 0 . 0 0 6 6 2 ]$ and $T = 1 . 2 3 { \mathrm { . } }$ The largest isolated control statistic, $T = 2 . 4 9$ for the sinusoidal source at $\sigma = 0 . 2 0$ , is not reproduced across the remaining noise levels; moreover, its mean endpoint contrast between $\gamma = 1$ and $\gamma = 0$ is 0.00738 with 95% $\operatorname { C I } \ [ - 0 . 0 0 0 4 7 , 0 . 0 1 5 2 4 ]$ , which spans zero. The control response is therefore consistent with a finite-sample fluctuation rather than a stable alternative predictive channel.

In benchmark N3 (Section 3.7), the most stringent control is the independent $\mathrm { A R } ( 1 )$ source because it has the same first-order memory coeficient, 0.60, as the true driver. Its mean |T| is only 0.39. Likewise, the deterministic logistic and quasiperiodic controls do not acquire a systematic FEDE response as the true $Y  X$ coupling increases. These results indicate that the FEDE $\Delta$ statistic is not merely reacting to added temporal structure, target complexity, or autocorrelation in the auxiliary input. Within this controlled benchmark it selectively tracks the candidate series that actually carries the imposed nonlinear predictive information, and it does so in the correct forecast direction.

## 4.8 Benchmark N4: functional-form robustness and comparison with Granger and CCM

Benchmark N4 (Section 3.8) separates two diferent forms of performance: peak sensitivity when the comparator representation happens to match the hidden interaction, and robustness when the interaction form is unknown. Table 11 reports the signed $T$ values averaged over the four noise levels for the true $Y  X$ direction. The Granger entries are the chronological out-of-sample scores from Eq. (27); the earlier in-window residual-ratio diagnostic is not used in the final comparison.

Table 11: Signed response-sensitivity statistic $T ,$ averaged over $\sigma \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \}$ , for four equal-power hidden nonlinear channels. Positive values are aligned with increasing useful predictive contribution. The polynomial Granger basis is fixed at degree two for all channels.
<table><tr><td>Method</td><td> $H _ { 2 }$ </td><td> $H _ { 3 }$ </td><td> $H _ { 4 }$ </td><td> $\sin ( y ^ { 2 } )$ </td></tr><tr><td>FEDE  $\Delta + \mathrm { L i g h t G B M }$ </td><td>29.00</td><td>27.47</td><td>23.22</td><td>28.67</td></tr><tr><td>Mean  $\mathrm { l o g - R M S E } + \mathrm { L i g h t G B M }$ </td><td>14.09</td><td>10.68</td><td>12.67</td><td>12.71</td></tr><tr><td>Linear Granger-style OOS</td><td>-7.94</td><td>-3.08</td><td>-6.70</td><td>-3.56</td></tr><tr><td>Polynomial Granger-style OOS, degree 2</td><td>74.21</td><td>-6.77</td><td>3.53</td><td>-4.01</td></tr><tr><td>CCM convergence  $\Delta \rho$ </td><td>1.85</td><td>9.07</td><td>0.36</td><td>0.02</td></tr></table>

FEDE is the most uniform positively aligned response across the four nonlinear forms. Its noise-averaged $T$ varies only from 23.22 to 29.00, with mean $| T | = 2 7 . 0 9$ across channel forms and a max/min ratio of 1.25. Even the smallest single-noise FEDE response in this benchmark is $T = 2 0 . 2 9$ for $H _ { 4 }$ at $\sigma = 0 . 2 0$ . By contrast, mean log-RMSE remains positive but is consistently weaker, with noise-averaged $T = 1 0 . 6 8 – 1 4 . 0 9$ . Because the two quantities are computed from the same LightGBM forecasts, this diference isolates the scalar summary: the entropy-based horizon structure provides a more reproducible coupling response than averaging forecast-error magnitude alone in this benchmark.

The out-of-sample Granger controls clarify the role of prior model form. Linear Granger-style gain is negative for all four channels, $T = - 7 . 9 4 \ \mathrm { t o \ - 3 . 0 8 }$ , meaning that adding raw $y _ { t }$ increasingly worsens rather than improves held-out one-step prediction as the hidden nonlinear contribution grows. The degree-2 polynomial comparator behaves very diferently when its prescribed basis matches the data-generating mechanism. For $H _ { 2 }$ , where the true channel is quadratic in $y ,$ it reaches $T = 7 4 . 2 1$ and therefore exceeds FEDE by a wide margin. This is an expected advantage of a correctly specified low-dimensional parametric representation. However, the same fixed degree-2 model becomes antialigned for $H _ { 3 }$ and $\sin ( y ^ { 2 } ) ~ ( T = - 6 . 7 7 \mathrm { ~ a n d ~ } - 4 . 0 1 )$ and only weakly positive for $H _ { 4 } ~ ( T = 3 . 5 3 )$ . Thus the benchmark does not support a claim that FEDE universally outperforms a matched parametric model; it supports the more specific conclusion that FEDE remains sensitive without being told which of several substantially diferent nonlinear forms generated the predictive channel.

The chronological scoring control is methodologically important. At $\gamma = 0 .$ , the mean OOS log gain is approximately −0.03 for the linear full model and −0.11 for the degree-2 full model, reflecting the ordinary generalization cost of extra parameters when the source contains no useful information. This behavior contrasts with the positive in-window nested residual ratios obtained when the same observations are used both to fit and score the larger model. The main comparison therefore uses only held-out Granger-style efect sizes.

Basic CCM convergence measures a diferent property and does not align consistently with the imposed predictive direction in this stochastic benchmark. Its true-direction noise-averaged $T$ is 1.85 for $H _ { 2 } , 9 . 0 7$ for $H _ { 3 } , 0 . 3 6$ for $H _ { 4 }$ , and 0.02 for sin $( y ^ { 2 } )$ . Moreover, the reverse-direction mean $| T |$ values in the same benchmark are 18.27, 18.34, 15.00, and 24.12, respectively. This should not be interpreted as a general failure of CCM. Here the source is stochastic AR(1), and three of the four mappings are even or otherwise non-invertible with respect to raw $y ,$ so the target can contain predictive information about $\phi _ { 2 } ( y )$ without preserving enough state information to reconstruct the signed source itself. The result instead emphasizes that forecast improvement and state-space reconstructability are distinct notions of directional dependence [13, 14].

FEDE also preserves directionality across the functional-form scan. Its true/reverse mean- $\cdot \vert T \vert$ ratios are 30.2 for $H _ { 2 } , 6 7 . 6$ for $H _ { 3 }$ , 45.3 for $H _ { 4 }$ , and 21.1 for sin $( y ^ { 2 } )$ . Thus the stable true-direction response is not obtained by making the descriptor indiscriminately responsive to either forecasting direction.

## 4.9 Benchmark N2: intermittent hidden nonlinear coupling

Benchmark N2 (Section 3.6) tests whether the nonlinear contribution can be followed in time rather than only calibrated across stationary realizations. Figure 9 shows the exact rectangular ground-truth schedule, the raw and oracle rolling correlations, and the two principal FEDE responses for the main-text setting $W = 4 0 0 , \sigma = 0 . 1 0 , m = 1$ , and rolling step 20. Each imposed coupling level lasts 1000 samples, so the active plateaus are long enough to separate the response level from the transition region even with the larger window. No post-hoc temporal smoothing is applied: the displayed curves are unsmoothed ensemble means over the 10 independent realizations.

The nonlinear oracle follows the active blocks and increases strongly with their amplitude, whereas the raw $r ( y _ { t } , x _ { t + 1 } )$ diagnostic remains close to zero. LightGBM FEDE responses are small at the weakest levels but form progressively clearer plateaus at $\gamma = 0 . 2 0$ and 0.40; Ridge remains near its baseline and does not reproduce this level ordering. The finite $W = 4 0 0$ support visibly broadens the rising and falling edges relative to the exact rectangular schedule, which is the expected cost of the increased statistical averaging.

The quantitative tracking summary in Fig. 10 uses the known window-mean imposed coupling $\bar { \gamma } ( \tau )$ as the direct ground truth. For LightGBM at $W = 4 0 0$ , Pearson correlations are $r _ { P } = 0 . 8 7 0$ for mean log-RMSE gain, 0.911 for FEDE $\Delta ,$ and 0.924 for FEDE early slope; the corresponding Spearman values are $\rho _ { S } = 0 . 5 5 0 , 0 . 5 3 5$ , and 0.600. Against the nonlinear oracle, the same three LightGBM responses give Pearson correlations of 0.829, 0.867, and 0.883, respectively, and Spearman correlations of 0.551, 0.544, and 0.598. Thus FEDE early slope gives the strongest overall time-resolved tracking in this setting, with FEDE $\Delta$ close behind.

At $W = 2 0 0$ , the LightGBM Pearson correlations with $\bar { \gamma } ( \tau )$ are 0.781, 0.860, and 0.863 for log-RMSE gain, FEDE $\Delta$ , and FEDE early slope, respectively; their Spearman counterparts are 0.552, 0.590, and 0.582. Increasing the window to 400 therefore strengthens the linear alignment and stabilizes the visible plateau response, while the rank correlations remain of similar magnitude and the transition edges become broader. Ridge remains weak or anti-aligned at $W = 4 0 0$ : its Pearson correlations with $\bar { \gamma } ( \tau )$ are −0.246, −0.335, and −0.191, and its corresponding Spearman correlations are −0.231, −0.261, and −0.082.

(a)  
![](images/5559cbdee6c4692c4808b658d8b2991140807b9cd1bb571ecfdff696bb92c1c1.jpg)

(b)  
![](images/efb8edc7738bd11db52e5613304821fb4cccbc76698718d5cc9e2023d7cc623a.jpg)

(c)  
![](images/be084e0a864a9ed0ae38a7addca7ff6431b2f857dbcb3686e738a17e6fb27dd0.jpg)

(d)  
![](images/ab7f29eb1c1c553d74ffa79599256c92bc16465bebf1bf6501099f11d4861976.jpg)  
Figure 9: Benchmark N2: time-resolved hidden nonlinear coupling at $\sigma = 0 . 1 0$ , history $m = 1$ , and rolling-window length $W = 4 0 0$ . (a) Exact imposed coupling schedule with 1000 samples per block. (b) Rolling raw linear correlation $r ( y _ { t } , x _ { t + 1 } )$ and nonlinear-oracle correlation $r ( \phi _ { 2 } ( y _ { t } ) , x _ { t + 1 } )$ . (c) FEDE $\Delta$ change and (d) FEDE early-slope change for LightGBM and Ridge. Curves are unsmoothed ensemble means over 10 realizations with rolling step 20 and $H = 1 4$ . Negative FEDE values are retained numerically and shown with lower opacity rather than clipped.

![](images/fd5085dffca0e874e48e344ba33adad6d9e5f7677150f54cf80c11e6bdb54713.jpg)  
Figure 10: Benchmark N2 tracking quality against the known window-mean imposed coupling $\bar { \gamma } ( \tau )$ (a) Pearson and (b) Spearman correlations for mean log-RMSE gain, FEDE $\Delta ,$ and FEDE early-slope change. Solid curves use $W = 4 0 0$ and dashed curves use $W = 2 0 0 \colon$ circles denote LightGBM and squares denote Ridge. Correlations are computed from the unsmoothed ensemble-mean rolling responses.

The W = 200 time-domain curves are retained in Appendix A as a direct higher-resolution comparison. They show the complementary side of the trade-of: shorter windows localize switching more sharply but produce a visibly noisier response. The main-text $W = 4 0 0$ choice is therefore a presentation setting for the intermittent benchmark, not a universal recommendation for empirical data.

## 4.10 Benchmark N4: computational cost

Benchmark N4 (Section 3.8) was executed under Windows 11 with 16 logical CPUs, Python 3.14.5, scikit-learn 1.9.0, LightGBM 4.7.0, and 10 outer worker processes. The complete Program08 run, including four channels, four noise levels, 13 coupling levels, 20 realizations, true and reverse directions, LightGBM forecasting, Granger diagnostics, and CCM, required 104.6 minutes of wall time.

Table 12 separates the principal true-direction method costs per 100 window evaluations. FEDE and mean log-RMSE share exactly the same LightGBM forecasts and therefore should not be added as independent costs. The shared LightGBM forecasting core requires 28.81 s per 100 windows, whereas the complete FEDE calculation requires 29.45 s. Thus the entropy-profile calculation itself adds approximately 0.64 s per 100 windows, or 2.2% beyond the nonlinear forecasting core. The chronological OOS Granger controls are orders of magnitude cheaper, and basic CCM lies between OLS and LightGBM.

Table 12: Approximate computation time for 100 true-direction window evaluations. FEDE and mean log-RMSE share the same LightGBM forecast core. OOS Granger timings come from the dedicated chronological control run.
<table><tr><td>Method</td><td>seconds / 100 windows</td><td>relative to FEDE</td></tr><tr><td>FEDE  $\Delta + \mathrm { L i g h t G B M }$ </td><td>29.451</td><td>1.000</td></tr><tr><td>Mean log-RMSE + LightGBM</td><td>28.820</td><td>0.979</td></tr><tr><td>CCM convergence  $\Delta \rho$ </td><td>1.864</td><td>0.063</td></tr><tr><td>Linear Granger-style OOS</td><td>0.0158</td><td>0.00054</td></tr><tr><td>Polynomial Granger-style OOS, degree 2</td><td>0.0207</td><td>0.00070</td></tr></table>

The practical implication is that FEDE is not computationally cheap if a nonlinear forecast engine must first be fitted across many horizons. However, once those forecasts already exist, the entropy descriptor adds little additional cost. This distinction is relevant when FEDE is used as an extra diagnostic alongside an existing multi-horizon forecasting pipeline.

## 4.11 Benchmark N5: parameter robustness of the FEDE response

Benchmark N5 (Section 3.9) asks whether the positive FEDE dose response survives changes in the method’s own analysis scales. At $\sigma = 0 . 1 0$ , the complete grid in Eq. (48) contains 27 configurations for each of the four hidden nonlinear channels, giving 108 independently fitted FEDE $\Delta$ response statistics. Every value is positively aligned with the imposed coupling: all 108 values satisfy $T > 0$ the smallest observed value is $T = 1 6 . 4 9$ , and the 95% confidence interval of the realization-level coupling slope remains above zero for every tested grid point.

The original configuration used throughout the main nonlinear experiments, $W = 1 0 0 , m = 1$ , and $H = 1 4$ , remains strongly responsive without any post-hoc retuning. It gives $T = 3 2 . 9 5 , 2 4 . 3 1 , 2 0 . 3 5$ and 29.71 for $H _ { 2 } , H _ { 3 } , H _ { 4 }$ , and $\sin ( y ^ { 2 } )$ , respectively. Across the full 27-point grid for each channel, the corresponding mean $T$ values are 41.34, 28.32, 22.30, and 45.39, while even the channel-specific minima remain between 16.49 and 18.02 (Table 13). Thus the positive response is not confined to the nominal parameter triplet.

Table 13: FEDE parameter robustness at $\sigma = 0 . 1 0$ . “Primary” is the configuration $W = 1 0 0 , m = 1$ $H = 1 4$ used in the main nonlinear experiments. Grid statistics summarize all 27 combinations in Eq. (48) for each hidden channel.
<table><tr><td>Hidden channel</td><td>Primary  $T$ </td><td>Grid mean  $T$ </td><td>Grid min  $T$ </td><td>Grid max  $T$ </td></tr><tr><td> $H _ { 2 }$ </td><td>32.95</td><td>41.34</td><td>16.49</td><td>55.35</td></tr><tr><td> $H _ { 3 }$ </td><td>24.31</td><td>28.32</td><td>16.90</td><td>42.88</td></tr><tr><td> $H _ { 4 }$ </td><td>20.35</td><td>22.30</td><td>18.02</td><td>32.13</td></tr><tr><td> $\sin ( y ^ { 2 } )$ </td><td>29.71</td><td>45.39</td><td>17.01</td><td>73.78</td></tr></table>

Figure 11 separates the marginal efects of the three analysis parameters. Window length produces the clearest aggregate change: after averaging over hidden channel, history, and horizon, mean $T$ rises from 23.65 at $W = 1 0 0$ to 36.30 at $W = 2 0 0$ and 43.05 at $W = 4 0 0$ . The corresponding mean realization-level slope increases from 0.0836 to 0.1222 and 0.1406. History length has a smaller systematic efect, with mean $T = 3 7 . 7 6 , 3 3 . 0 3$ , and 32.23 for $m = 1 , 3$ , and $5 .$ Dependence on the maximum forecast horizon is weaker still over the tested range: mean $T = 3 5 . 9 6 , 3 3 . 4 0$ , and 33.65 for $H = 5 , 1 4$ , and 20, respectively. In the panels, each channel point is the marginal mean over the two remaining grid dimensions and its vertical bar is the corresponding min–max range, not a confidence interval.

(a)  
![](images/8d4e17657374e8af42ba1004d773350e327b512e7c4a503b3d50b726858e6782.jpg)

![](images/db8c2c80eb728f7913603e8320560e00cf7b830fe531e1a4acb1eff8b199313c.jpg)

(c)  
![](images/2d5eff126be20dd520a7132d7c6372a3661eea3d80deb171d45dee9dbb0714c2.jpg)  
Figure 11: Marginal FEDE response sensitivity over the predefined parameter grid at $\sigma = 0 . 1 0 . \mathrm { ( a ) }$ Window length W; (b) history length m; (c) maximum forecast horizon H. Curves show the four hidden nonlinear channels and their overall mean. For each channel, the marker is the mean T after averaging over the other two grid parameters; vertical bars span the minimum and maximum over those remaining configurations and are not confidence intervals.

The most robust single grid point under a worst-channel criterion is W = 200, m = 1, H = 20, for which the weakest of the four channel responses is still $T = 3 2 . 1 3$ and the mean across channels is 43.77. This point is reported descriptively rather than adopted as a replacement for the original setting. The sensitivity experiment is therefore interpreted as a robustness check over a predefined grid, not as a post-hoc search for a new optimum.

## 5 Discussion

## 5.1 FEDE as a horizon-resolved predictive observable

The central methodological result of this study is not that restricted-versus-augmented forecasting is new. That logic is already fundamental to Granger-style analysis, out-of-sample forecast comparison, and several nonlinear predictive frameworks [9, 10, 11, 17]. The distinct object introduced here is the horizon-resolved entropy evolution of the signed forecast-error distribution. We therefore use the term FEDE predictive-causality framework for the complete matched construction, with predictive causality understood only in the Granger sense of additional predictive information and not as structural or physical causal identification. FEDE sits downstream of a predictor: it asks whether adding a candidate source changes how predictive uncertainty is organized from short to longer horizons, rather than only whether a one-step residual variance or a horizon-averaged loss decreases. This distinction is summarized conceptually in Table 1 and operationally in Table 2.

The controlled experiments show why this extra structure can be informative. In the Gaussian-AR(1) nonlinear benchmark family N1–N5, raw $r ( y _ { t } , x _ { t + 1 } )$ remains close to zero even as the oracle association with the hidden transform increases (Fig. 6). This behavior is deliberate: unlike the logistic-map linear references L1–L2, the nonlinear family injects the source through a transform that is orthogonal to raw Gaussian y at first order. A nonlinear forecasting engine can nevertheless recover the predictive contribution, and FEDE converts that recovery into a substantially more reproducible dose response than the mean forecast-error magnitude. With LightGBM, the FEDE $\Delta$ and early-slope statistics reach mean signed sensitivities of 26.71 and 26.53, respectively, compared with 14.92 for mean log-RMSE gain (Table 8); the corresponding Ridge responses are weak or anti-aligned (Table 9 and Fig. 7). Thus FEDE does not create nonlinear information that is absent from the predictor. Rather, once a predictor extracts that information, the entropy profile provides a diferent and, in these benchmarks, more stable summary of how the added source reshapes the forecast-error field across lead times.

This distinction is also visible in the detection experiment. At $\sigma = 0 . 1 0$ , sustained 90% detection occurs at $\gamma = 0 . 3 0$ for both FEDE descriptors but only at $\gamma = 0 . 5 0$ for mean log-RMSE gain $\left( \mathrm { F i g . ~ } 8 \right)$ Because all three descriptors are computed from the same LightGBM forecasts, this improvement cannot be attributed to a stronger prediction model. It reflects the choice of response variable itself: the distributional reorganization of errors across horizons can be more reproducible than a scalar average of their magnitudes.

Benchmark N2 extends this observation from stationary dose–response curves to a time-varying setting. With 1000-sample coupling blocks, the $W = 4 0 0$ LightGBM FEDE early-slope response reaches $r _ { P } = 0 . 9 2 4$ with the known window-mean coupling and $r _ { P } = 0 . 8 8 3$ with the nonlinear oracle, while FEDE $\Delta$ reaches 0.911 and 0.867, respectively. The shorter $W = 2 0 0$ analysis retains sharper transition localization but lower Pearson alignment. This is not a contradiction: increasing W supplies more forecast errors for entropy estimation and stabilizes the response, but any rolling statistic necessarily averages over a wider temporal support. The intermittent benchmark therefore turns the window-size efect into an explicit resolution–stability trade-of rather than a tuning artifact.

## 5.2 Comparison with Granger-style and conventional forecast-gain approaches

The method-comparison benchmark in Table 11 provides the most direct context for interpreting FEDE relative to established predictive approaches. The literature already ofers a continuum from classical linear Granger causality to polynomial/Volterra, kernel, neural, and other machine-learning extensions [9, 10, 11, 12, 16]. Their common strength is a clear restricted-versus-full interpretation; their common dependence is that sensitivity is tied to the representation supplied to the model. Our results reproduce both sides of that trade-of.

A correctly specified parametric comparator can outperform FEDE by a large margin. For the quadratic $H _ { 2 }$ channel, the fixed degree-2 polynomial Granger-style model explicitly contains the relevant $y _ { t } ^ { 2 }$ term and reaches $T = 7 4 . 2 1$ , whereas FEDE reaches $T = 2 9 . 0 0$ . This is not a weakness to be hidden: when the interaction form is known a priori, a parsimonious matched model is preferable and can be far more sensitive. The more revealing result appears when the hidden mapping changes while the polynomial basis is left fixed. The same degree-2 comparator becomes anti-aligned for $H _ { 3 }$ and sin $( y ^ { 2 } )$ and only weakly positive for $H _ { 4 }$ , while FEDE remains positively aligned in the narrow range $T = 2 3 . 2 2 \ – 2 9 . 0 0$ across all four nonlinear forms. The advantage demonstrated here is therefore robustness to unknown interaction form, not universal superiority over a model supplied with the correct basis.

The linear Granger-style control further illustrates the importance of honest out-of-sample scoring. Using a chronological 60:40 split, the linear full model is anti-aligned for every hidden nonlinear channel $( T = - 7 . 9 4 \mathrm { t o } - 3 . 0 8 )$ , and the degree-2 model is strongly useful only in its matched quadratic case. In contrast, an in-window nested residual comparison gives the larger model an automatic training-error advantage. The OOS formulation in Eq. (27) removes that bias and better matches the operational question posed in this paper: does the candidate source improve unseen future prediction? This choice is consistent with the broader forecast-comparison literature, where predictive value is evaluated on held-out errors and formal procedures such as Diebold–Mariano tests address whether forecast-loss diferences are statistically distinguishable from zero [17, 18]. FEDE is complementary to such tests: it supplies a magnitude/structure descriptor, whereas a DM-type procedure supplies inferential evidence about a chosen loss diferential.

The comparison with mean log-RMSE is especially informative because the prediction engine is held fixed. Across $H _ { 2 } , H _ { 3 } , H _ { 4 }$ , and standardized $\sin ( y ^ { 2 } )$ , FEDE yields noise-averaged $T = 2 9 . 0 0$ 27.47, 23.22, and 28.67, whereas the corresponding mean log-RMSE values are 14.09, 10.68, 12.67, and 12.71 (Table 11). FEDE therefore does more than re-express forecast accuracy on a diferent scale. It uses the shape change of the signed-error distribution across $h = 1 , \ldots , H$ , which appears to preserve a stronger and more uniform coupling signature than the mean error magnitude in the present controlled setting.

A possible explanation for this sensitivity is that RMSE compresses each horizon-specific error distribution to its second moment, whereas FEDE responds to changes in how probability mass is distributed across the error range. An auxiliary source may therefore reorganize the forecast errors—by changing their concentration, tails, or other non-Gaussian distributional features—without producing a proportionally large change in mean squared magnitude. FEDE additionally tracks how this reorganization evolves with forecast horizon, allowing a weak but systematic distributional signature to accumulate into a stable horizon-resolved response. Thus, the observed gain arises not from changing the forecasts, but from changing what is measured in their errors. Importantly, no qualitatively new information should be expected if the only efect were a pure Gaussian scale change: for an ideal Gaussian scale family, diferential entropy is a monotonic function of variance. The stronger FEDE response observed here therefore suggests that the nonlinear predictive channel modifies the structure and horizon evolution of the forecast-error distribution rather than merely rescaling its variance. This interpretation is qualitative rather than a formal equivalence for the finite-bin entropy estimator used in the present implementation.

## 5.3 Relation to CCM and state-space directionality

CCM and EDM occupy a diferent conceptual position in the literature. CCM asks whether information about a putative driver is recoverable from the reconstructed manifold of the afected variable and whether this cross-map skill converges with increasing library size [13, 20]. EDM and S-map methods extend state-space reconstruction toward nonlinear forecasting and state-dependent interaction strengths [21, 20]. These are powerful tools, but they do not estimate the same object as incremental forecast contribution, and comparative reviews emphasize that no single directional measure is uniformly optimal across deterministic, stochastic, noisy, or synchronized regimes [19].

That distinction is central to the present CCM result. Basic CCM convergence is weak or directionally inconsistent for the stochastic AR(1) source and partly non-invertible hidden mappings in Table 11. For $H _ { 2 } , H _ { 4 }$ , and $\sin ( y ^ { 2 } )$ , opposite signs of raw Y can produce the same hidden contribution, so knowledge of Y can improve prediction of X even though the state of X cannot uniquely reconstruct the signed source. Process noise and external influence are also known to complicate cross-mapping behavior [14]. The observed weak or reversed CCM response should therefore be read as a scope result, not as evidence that CCM is inferior in the deterministic attractor regimes for which it was developed.

The directional/source-specificity experiment addresses a diferent possible criticism: that FEDE might respond simply because any structured auxiliary variable is added to the model. Table 10 argues against that interpretation. The true $Y  X$ channel gives mean $| T | = 2 6 . 7 1$ , while reverse direction and independent periodic, quasiperiodic, chaotic, and AR(1) controls remain at mean $| T | \leq 1 . 0 8 ;$ ; even the smallest true-driver response exceeds the largest individual control response by a factor of 9.8. The four-channel experiment then preserves large true/reverse separation under substantially diferent nonlinear forms. These results support directional predictive specificity within the controlled benchmark, while stopping short of structural causal identification. Common drivers, hidden confounders, feedback, and incomplete conditioning can still make predictive direction difer from physical causation, as is well recognized in both Granger and state-space analyses [9, 15, 14].

## 5.4 Computational cost and practical trade-ofs

The computational comparison in Table 12 makes clear that FEDE is not the fastest method considered. For 100 true-direction window evaluations, the full FEDE+LightGBM pipeline requires approximately 29.45 s, compared with 1.86 s for the present basic CCM implementation and only 0.0158–0.0207 s for the OOS linear and degree-2 polynomial Granger controls. If the only objective is a one-step restrictedversus-full test under a plausible low-dimensional parametric model, the OLS-based approaches are decisively cheaper.

The practical comparison changes, however, when multi-horizon nonlinear forecasts already form part of the analysis. Mean log-RMSE and FEDE share the same LightGBM forecast core, which costs approximately 28.81 s per 100 windows. The complete FEDE calculation costs 29.45 ${ \mathrm { s } } ,$ so the entropy-profile processing itself adds only about 0.64 s per 100 windows, or 2.2% beyond the shared forecast core. FEDE is therefore best viewed not as a computational replacement for Granger or CCM, but as a low-overhead additional diagnostic once a flexible multi-horizon forecast pipeline is already being run. The cost is paid mainly for the nonlinear forecasts; the distributional descriptor extracts extra information from forecast errors that have already been generated.

This trade-of also clarifies when FEDE is most attractive. A simple linear or polynomial model is preferable when its assumptions are credible and computational economy is paramount. A state-space method is more natural when manifold reconstruction and deterministic interaction geometry are the scientific object. FEDE becomes attractive when the forecasting system is already nonlinear, the functional form of the candidate interaction is not known in advance, and one wants a directional descriptor that uses the full horizon evolution rather than only a one-step loss or a single averaged accuracy score. This positioning is consistent with the method landscape summarized in Table 1: the methods are complementary because they answer related but non-identical questions.

## 5.5 Parameter robustness and the resolution–stability trade-of

The parameter-grid experiment closes an important internal robustness question that was not addressed by the original $W = 1 0 0 , m = 1 , H = 1 4 \mathrm { d e s i g n }$ . Across $W \in \{ 1 0 0 , 2 0 0 , 4 0 0 \} , m \in \{ 1 , 3 , 5 \}$ , and $H \in \{ 5 , 1 4 , 2 0 \}$ at $\sigma = 0 . 1 0$ , all 108 channel–configuration combinations remain positively aligned with the imposed coupling, with $T \geq 1 6 . 4 9$ and a positive realization-level slope confidence interval at every tested grid point (Table 13). The original configuration remains strongly responsive for all four hidden forms and was not replaced by a post-hoc optimum.

The three parameters play diferent roles (Fig. 11). Window length has the strongest marginal efect: mean $T$ rises from 23.65 at $W = 1 0 0$ to 36.30 at $W = 2 0 0$ and 43.05 at $W = 4 0 0$ . This gain is expected in a stationary benchmark because larger windows provide more training examples and more held-out errors from which to estimate a distribution. It should not be interpreted as a universal recommendation for $W = 4 0 0$ . The revised intermittent benchmark in Figs. 9 and 10 makes the complementary cost explicit: increasing the rolling window from 200 to 400 raises the LightGBM Pearson tracking correlations with the known window-mean coupling from 0.860 to 0.911 for FEDE $\Delta$ and from 0.863 to 0.924 for FEDE early slope, but broadens the transitions around each switch. In a nonstationary application, the useful W must therefore balance statistical stability against temporal resolution.

History length has a smaller systematic efect, with the highest mean $T$ at $m = 1$ and modest reductions at $m = 3$ and 5. This is consistent with the short-memory construction of the synthetic process and should not be generalized to empirical systems with longer memory. The maximum horizon is comparatively insensitive over the tested range: mean $T$ is 35.96, 33.40, and 33.65 for $H = 5 , 1 4$ , and 20. Thus the main result does not depend critically on the particular choice $H = 1 4$ , while still requiring a genuinely multi-horizon profile: for the endpoint definition $\Delta \mathrm { F E D E } = \mathrm { F E D E } ( H ) - \mathrm { F E D E } ( 1 )$ $H = 1$ would be degenerate by construction.

## 5.6 Scope, limitations, and future validation

The study remains deliberately controlled. The synthetic benchmarks use one stochastic driver family, one target architecture, a finite set of hidden nonlinear mappings, one nonlinear prediction engine, and a basic CCM implementation without lag scanning, surrogate testing, or multivariate conditioning. The parameter study establishes robustness only over the tested ranges, not universal invariance. FEDE also remains model-relative: if the prediction engine cannot extract a real interaction, the entropy descriptor cannot recover information that is absent from its forecast errors. Conversely, a highly flexible predictor can exploit spurious structure unless the validation design, source controls, and temporal split are chosen carefully.

Real observational data add challenges that are intentionally absent here: common drivers, delayed feedback, changing regimes, missing observations, unequal sampling, multiple candidate sources, and limited sample sizes. These issues motivate the next validation stage. The broader literature shows that directional time-series questions arise across ecological interactions, epidemiological surveillance, hydrology and climate, physiological/neuroscience signals, and financial/economic systems [13, 22, 23, 12, 24]. Future work should therefore test whether the synthetic advantages demonstrated here— functional-form robustness, source specificity, and multi-horizon sensitivity—survive on observational datasets where the ground truth is incomplete and competing explanations cannot be controlled by construction. Such studies should include richer comparators where appropriate, including lagged or surrogate-tested CCM/EDM, nonlinear or neural Granger variants, and formal forecast-loss tests [10, 16, 18]. An especially relevant extension is to replace the conventional residual-variance score in matched Granger-style models by residual-distribution entropy and test, on the same restricted/full residuals, whether the sensitivity observed for FEDE arises primarily from the distributional observable, from the multi-horizon construction, or from their combination.

The present evidence therefore supports a deliberately limited claim. The FEDE predictivecausality framework is not a replacement for Granger causality, CCM, or conventional forecast evaluation, and its use of the term predictive causality does not imply structural causal identification. Its contribution is a new horizon-resolved predictive observable: by comparing the entropy evolution of matched signed forecast-error distributions, it converts multi-horizon forecasts into a directional descriptor that remains sensitive across noise, source controls, nonlinear functional forms, and a broad internal parameter grid. The main practical value demonstrated here is the combination of representation flexibility, directional specificity, and robust multi-horizon response at little additional cost once nonlinear forecasts are already available.

## 6 Conclusions

This work introduced the forecast-error distribution entropy (FEDE) predictive-causality framework as a horizon-resolved approach to directional predictive contribution. Predictive causality is used here strictly in the Granger operational sense: the source past must provide additional out-of-sample predictive information beyond the target’s own past, within the specified model and conditioning set. The construction compares matched target-only and target-plus-source forecasts and measures how the entropy of the signed forecast-error distribution evolves from short to longer lead times. The method is therefore distinct from correlation, one-step residual comparison, and state-space reconstruction: it uses the geometry of the forecast-error distribution across horizon as the measured object.

The controlled experiments establish five complementary properties. First, FEDE detects a hidden nonlinear predictive channel even when raw linear association is near zero, provided that the forecasting engine can extract the underlying information. Second, its response is source- and direction-specific: the true Y → X channel is separated from the reverse direction and from independent periodic, quasiperiodic, chaotic, and AR(1) auxiliaries. Third, FEDE remains positively aligned across four substantially diferent hidden nonlinear mappings without changing its representation, whereas the OOS polynomial Granger comparator is extremely strong only when its prescribed degree matches the true quadratic form. Fourth, in the intermittent benchmark the LightGBM FEDE response tracks the time-varying imposed coupling, with the $W = 4 0 0$ early-slope statistic reaching Pearson $r _ { P } = 0 . 9 2 4$ against the known window-mean coupling and exposing the expected stability–temporalresolution trade-of relative to $W = 2 0 0$ . Fifth, the positive stationary response survives all 108 tested combinations of window length, history length, and maximum forecast horizon, with $T \geq 1 6 . 4 9$ throughout the predefined grid.

The comparative results also define where FEDE should and should not be used. A correctly specified low-dimensional parametric model can be both faster and more sensitive; in the matched $H _ { 2 }$ case, degree-2 polynomial Granger reaches $T = 7 4 . 2 1$ compared with FEDE’s $T = 2 9 . 0 0$ . Basic CCM addresses a diferent state-space reconstruction question and should not be judged by the stochastic non-invertible benchmark alone. The total FEDE+LightGBM pipeline is computationally much more expensive than OLS-based Granger analysis, but once the multi-horizon LightGBM forecasts are available the entropy calculation adds only about 2.2% to the shared forecast-core time. FEDE’s advantage is therefore not universal accuracy or speed, but a comparatively uniform response when the interaction form is unknown and multi-horizon nonlinear forecasts are already part of the workflow.

The next step is validation on observational time series. Candidate domains include financial and economic systems, climate and hydrology, epidemiological surveillance, physiology and neuroscience, and ecological interaction networks, all of which have established traditions of predictive or state-space directional analysis [13, 22, 23, 12, 24]. Such applications should test robustness to confounding, feedback, regime changes, missing data, multiple candidate sources, and uncertain time delays, and should compare FEDE with stronger domain-appropriate baselines rather than relying on the synthetic ranking reported here.

The main contribution of the study is consequently specific but generalizable: the FEDE predictivecausality framework turns the horizon evolution of forecast-error distribution entropy into a modelrelative directional predictive descriptor. Across the controlled tests performed here, this observable adds information beyond average forecast error, preserves specificity to the true source and direction, and remains stable across markedly diferent nonlinear forms and analysis scales. These properties establish FEDE as a sensitive and robust complementary tool for studying nonlinear predictive coupling when the interaction form is not known in advance, while preserving a clear distinction between predictive and structural causality.

## Funding

This research was supported by the Russian Science Foundation (grant no. 22-11-00055-P).

## Code and data availability

A persistent Zenodo record for the FEDE predictive-causality software and benchmark package is publicly available at DOI: 10.5281/zenodo.23330544. The software package includes the reproducible Python implementation, benchmark scripts, configuration files, reference results, and a researcherfacing application workflow. It also contains numerical result tables underlying the principal methodcomparison, parameter-robustness, and N2 time-resolved tracking analyses, together with the N2 settings used to generate the corresponding figures. The synthetic benchmark definitions, parameter settings, and analysis procedures required to interpret the results are described in the manuscript and accompanying repository.

## A Supplementary nonlinear diagnostics

Figure 12 shows the complete LightGBM stationary response across all four nonlinear noise levels. Figure 13 provides the $W = 2 0 0$ time-domain comparison for benchmark N2 and complements the $W = 4 0 0$ main-text visualization.

![](images/38a09e0869a9860e965d0f0cb652f8ee36093716d315a38cacabb9b1322095b2.jpg)

![](images/e1c54d13c5eb34424f6d7bdb60f5a0d332e1c623a1a5f60dcc7fb61318c04cf5.jpg)

![](images/cb7acccaf99f69393dbb6f9371af2587dc43253d5feace948ce02d53b2a0a40c.jpg)  
Figure 12: Stationary hidden nonlinear coupling response for LightGBM across the complete noise grid. Bands show 95% confidence intervals of the mean across 20 realizations.

(a)  
![](images/91b09b25311edce4736e6e437efa97bcd519edbcb73bc2f039c1d215b00de1b0.jpg)

(b)  
![](images/fa97c80388c8e9d673879ee91922c53c8291c39428b6697ce7f79126d8a2f295.jpg)

(c)  
![](images/97c00ea9b414d80774f927f8b936e7647b6e9ec8f807c024f8d9548e9eaabf2a.jpg)

(d)  
![](images/312ee0a3901cebbc033bdfd688069f90406b1be3db9eb2a8f40601cfdeca27c7.jpg)  
Figure 13: Benchmark N2 supplementary time-domain comparison at $W = 2 0 0 , \sigma = 0 . 1 0$ , history $m = 1$ , rolling step 20, and $H = 1 4$ . The imposed schedule and all stochastic realizations are the same as in the $W = 4 0 0$ main-text analysis. The shorter window preserves sharper temporal localization around switching boundaries but yields a visibly less stable FEDE response. Negative FEDE values are retained and displayed with lower opacity rather than clipped.

## B Supplementary calibration-range diagnostic

Figure 14 shows how the linear-fit $R ^ { 2 }$ changes as progressively larger maximum coupling values are included. It is retained as a diagnostic because very narrow low-γ ranges can produce unstable $R ^ { 2 }$ values when the response amplitude is comparable to realization variability.

![](images/5625facf39bbd3af153457029d8451b0a8b89c102328ee0fe658b8c44f2cc0dd.jpg)

Figure 14: Linearity diagnostic as the fitted coupling range is expanded. The main text therefore emphasizes the complete controlled calibration and the noise-dependent linearity analysis rather than any single very narrow range.

## References

[1] J. D. Farmer and J. J. Sidorowich, “Predicting Chaotic Time Series,” Physical Review Letters, vol. 59, no. 8, pp. 845–848, 1987. doi:10.1103/PhysRevLett.59.845.

[2] M. Casdagli, “Nonlinear Prediction of Chaotic Time Series,” Physica D: Nonlinear Phenomena, vol. 35, pp. 335–356, 1989. doi:10.1016/0167-2789(89)90074-2.

[3] G. Sugihara and R. M. May, “Nonlinear Forecasting as a Way of Distinguishing Chaos from Measurement Error in Time Series,” Nature, vol. 344, pp. 734–741, 1990. doi:10.1038/344734a0.

[4] F. Takens, “Detecting Strange Attractors in Turbulence,” Lecture Notes in Mathematics, vol. 898, pp. 366– 381, 1981. doi:10.1007/BFb0091924.

[5] A. Velichko, M. Belyaev, and P. Boriskov, “A Novel Approach for Estimating Largest Lyapunov Exponents in One-Dimensional Chaotic Time Series Using Machine Learning,” Chaos, vol. 35, no. 10, 101101, 2025. doi:10.1063/5.0289352.

[6] A. E. Hoerl and R. W. Kennard, “Ridge Regression: Biased Estimation for Nonorthogonal Problems,” Technometrics, vol. 12, no. 1, pp. 55–67, 1970. doi:10.1080/00401706.1970.10488634.

[7] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A Highly Eficient Gradient Boosting Decision Tree,” Advances in Neural Information Processing Systems 30, pp. 3146–3154, 2017.

[8] C. E. Shannon, “A Mathematical Theory of Communication,” The Bell System Technical Journal, vol. 27, no. 3, pp. 379–423, 1948.

[9] A. Shojaie and E. Fox, “Granger Causality: A Review and Recent Advances,” Annual Review of Statistics and Its Application, vol. 9, pp. 289–319, 2021. doi:10.1146/annurev-statistics-040120-010930.

[10] A. Tank, I. Covert, N. J. Foti, A. Shojaie, and E. Fox, “Neural Granger Causality,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, pp. 4267–4279, 2021. doi:10.1109/TPAMI.2021.3065601.

[11] M. Roso l, M. M ly´nczak, and G. Cybulski, “Granger causality test with nonlinear neural-network-based methods: Python package and simulation study,” Computer Methods and Programs in Biomedicine, vol. 216, 106669, 2022. doi:10.1016/j.cmpb.2022.106669.

[12] F. Hasanzadeh, M. Mohebbi, and R. Rostami, “A Nonlinear Efective Connectivity Measure Based on Granger Causality and Volterra Series,” IEEE Journal of Biomedical and Health Informatics, vol. 26, pp. 2299–2307, 2021. doi:10.1109/JBHI.2021.3138199.

[13] G. Sugihara, R. May, H. Ye, C.-H. Hsieh, E. Deyle, M. Fogarty, and S. Munch, “Detecting Causality in Complex Ecosystems,” Science, vol. 338, pp. 496–500, 2012. doi:10.1126/science.1227079.

[14] D. Mønster, R. Fusaroli, K. Tyl´en, A. Roepstorf, and J. Sherson, “Causal inference from noisy time-series data—Testing the Convergent Cross-Mapping algorithm in the presence of noise and external influence,” Future Generation Computer Systems, vol. 73, pp. 52–62, 2017. doi:10.1016/j.future.2016.12.009.

[15] P. Stokes and P. Purdon, “A study of problems encountered in Granger causality analysis from a neuroscience perspective,” Proceedings of the National Academy of Sciences, vol. 114, pp. E7063–E7072, 2017. doi:10.1073/pnas.1704663114.

[16] A. Wism¨uller, A. M. D’Souza, A. Vosoughi, and A. Z. Abidin, “Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data,” Scientific Reports, vol. 11, 2021. doi:10.1038/s41598-021-87316-6.

[17] F. X. Diebold and R. S. Mariano, “Comparing Predictive Accuracy,” Journal of Business & Economic Statistics, vol. 20, pp. 134–144. doi:10.1198/073500102753410444.

[18] A. Grant, M. Mrazik, and S. Satchell, “Evaluating Forecasts at Multiple Horizons: An Extension of the Diebold–Mariano Approach,” Journal of Forecasting, 2026. doi:10.1002/for.70150.

[19] T. Edinburgh, S. Eglen, and A. Ercole, “Causality indices for bivariate time series data: A comparative review of performance,” Chaos, vol. 31, 083111, 2021. doi:10.1063/5.0053519.

[20] C.-W. Chang, M. Ushio, and C.-H. Hsieh, “Empirical dynamic modeling for beginners,” Ecological Research, vol. 32, pp. 785–796, 2017. doi:10.1007/s11284-017-1469-9.

[21] E. Deyle, R. May, S. Munch, and G. Sugihara, “Tracking and forecasting ecosystem interactions in real time,” Proceedings of the Royal Society B: Biological Sciences, vol. 283, 2016. doi:10.1098/rspb.2015.2258.

[22] M. C. Maher, R. D. Hernandez, S. Basu, and G. Sugihara, “Global environmental drivers of influenza,” Proceedings of the National Academy of Sciences, vol. 113, pp. 13081–13086, 2016. doi:10.1073/pnas.1607747113.

[23] G. Bonotto, T. Peterson, K. Fowler, and A. Western, “Identifying Causal Interactions Between Groundwater and Streamflow Using Convergent Cross-Mapping,” Water Resources Research, vol. 58, 2022. doi:10.1029/2021WR030231

[24] A. Isufaj, C. De Castro Martins, M. Cavazza, and H. Prendinger, “Applying time delay convergent cross mapping to Bitcoin time series,” Expert Systems with Applications, vol. 277, 127125, 2025. doi:10.1016/j.eswa.2025.127125.