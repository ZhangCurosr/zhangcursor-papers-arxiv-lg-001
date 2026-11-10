# Forecast Value Add: Reward-Based Action Selection over Probabilistic Demand Forecasts

Ankur Verma

TrueGradient AI, ankur@truegradient.ai

Abstract. Strong machine-learning ensembles capture much of the trend, seasonality, and known events in demand, yet they can still understate peaks, miss moving events, or mistake temporary changes for shifts in demand level. We address this gap by framing forecast refinement as a reward-driven decision problem. For each forecast period, we let an agent choose among values already produced by the ensemble: its base forecast, distribution-free bounds, quantiles, and a biascorrected estimate. We evaluate these choices against context-matched historical periods, accounting for calendar position, recurring and moving events, trend, price, and promotions. We then apply a structural seasonal guard to constrain adjustments and record the model, agent, and planner contributions to every published value. We evaluate our approach across five retail, consumer-goods, and direct-to-consumer deployments over nine months. Our refined forecasts reduce mean WMAPE by 6.3% compared with the stacked machine-learning forecast for every customer, with gains reaching 16.7%, and achieve the lowest error in 35 of 45 customer-months. Compared with statistical forecasts, our approach reduces mean error by 18.7%. These results demonstrate that context-aware, auditable action selection can add forecast value beyond a strong ensemble.

Keywords: Probabilistic Machine Learning · Reinforcement Learning · Planner Agents · Forecast Value Addition

## 1 Introduction

Demand planning is a decision problem layered on a prediction problem: a statistical engine answers “what is the most likely quantity?”, a planner “what quantity should we commit against?”.

What the ensemble already learns. A well-engineered machine-learning stack is not blind to structure. Fourier terms, calendar ordinals and seasonal lags give it the seasonal profile; a trend index and rolling means give it the level and its drift; holiday dummies, promotion flags and price drivers give it recurring events and commercial efects. The question is whether a separate decision layer can add value on top ofit.

Where value remains. Three kinds of residual survive even a strong ensemble. (i) Seasonal amplitude: an average-loss objective learns the shape of the season but slightly compresses its peaks and troughs, most in the highest-margin periods. (ii) Rare and moving events: a one-of promotion or a Black Friday that moved into another reporting month has few examples and a flag collinear with calendar features, so its lift is only partly attributed. (iii) Level shifts: a soft period, stock-out or data gap is partly absorbed into trend. These residuals recur at the same calendar positions, which makes them correctable by a policy that looks at the same positions in the past.

Forecast Value Add as the objective. Forecasting research scores models on accuracy; planning research scores interventions against a baseline using Forecast Value Add (FVA) [18,19]. FVA asks whether a step in the process improved on what preceded it, which is exactly what an agent that modifies an existing forecast is responsible for. We pose the refinement as decision making over a pre-sanctioned, discrete action set drawn from the ensemble’s own probabilistic outputs, so each figure is traceable and the policy is inspectable as “which action, and why”.

Contributions. (1) A distribution-free action set from an AutoML stack and Chebyshev– Cantelli bounds (Sect. 4); (2) an attribution whose layers sum exactly to the model output (Sect. 5); (3) a per-step reward from context-matched past periods, guarded by a structural seasonal model (Sect. 6); (4) a worked episode for one peak row (Sect. 7); (5) an evaluation on five customers over nine months, including the months the agent does not win (Sect. 8).

## 2 Related Work

Forecasting. The M5 competition showed that boosted trees over engineered calendar and price features remain highly competitive and that combinations dominate single learners [1]; DeepAR [3], N-BEATS [4] and the Temporal Fusion Transformer [5] exploit cross-series structure. Probabilistic forecasting frames the task as distribution estimation [2,7], with quantiles estimated directly [8,6]. We instead derive distributionfree bounds from residual dispersion, trading sharpness for validity.

Adjustment, explanation and decisions. Reconciliation aligns forecasts across a hierarchy [14], while the judgemental-adjustment literature finds that planner overrides often destroy accuracy and increase bias [19], motivating our FVA framing and governance constraints. Unlike SHAP [15], our attribution is a deterministic counterfactual that reconciles exactly to the published number. Contextual bandits [16] and reinforcement learning [17] supply the vocabulary for choosing among finite actions under a reward.

## 3 Problem Formulation

Let � index a product–location series and � a period with seasonal cycle length � ∈ {12, 52, 365}; $a _ { i , t }$ is realised demand and $\hat { y } _ { i , t }$ the base ensemble forecast. Accuracy is measured by weighted absolute percentage error over an evaluation set $\varepsilon _ { \mathrm { { } ; } }$

$$
\operatorname { W M A P E } ( { \mathcal { E } } ) = { \frac { \sum _ { ( i , t ) \in { \mathcal { E } } } | f _ { i , t } - a _ { i , t } | } { \sum _ { ( i , t ) \in { \mathcal { E } } } a _ { i , t } } } ,\tag{1}
$$

![](images/b6c26cdc4c38c726b55df5a97cd8139029ac0ce329d0deb962260ce2a4263e13.jpg)  
Fig. 1. The ensemble learns the structure; the agent refines it. Each published value is one of the ensemble’s own probabilistic values, chosen per period and checked by a structural guard.

for a forecast $f .$ For a downstream step � and an upstream step �, FVA is the relative error reduction

$$
\mathrm { F V A } ( A \parallel B ) = 1 0 0 \cdot \frac { \mathrm { W M A P E } _ { B } - \mathrm { W M A P E } _ { A } } { \mathrm { W M A P E } _ { B } } ,\tag{2}
$$

positive when � improves on �. We report the RL policy against two upstream references, the statistical forecast and the stacked ML forecast it refines.

## 4 AutoML Stacking and Distribution-Free Action Set

Segmentation and cleaning. Series are segmented by ABC–XYZ classification (volume contribution and coeficient of variation) combined with a department descriptor; segments define donor pools for short histories and the granularity of decisions. For A and B clusters, one-of spikes are imputed to a robust centre and out-of-stock demand is reconstructed, since censoring otherwise teaches the model that demand fell when supply did.

Feature engineering. Ten feature families are generated and then screened: regressors and drivers (price, discount, spend, promotion and campaign flags, continuous drivers held apart from events), datetime and Fourier terms with a trend index, lags and ofsets, encoded dimensions, per-holiday dummies and exogenous series. The lag grid is a dense block $\{ 1 , \ldots , 6 \}$ for short-run momentum plus custom lags {10, 11, 12, 13, 17, 18, 21, 24} that straddle the seasonal lag $m = 1 2$ , the half-cycle and the two-cycle lag. Rolling means over {1, 2, 3, 12} periods add a short and a full-cycle level, and lags respect the forecast gap to prevent leakage. Time-interaction dimensions let the seasonal shape difer across the product hierarchy. Mutual information $I ( X _ { j } ; Y )$ is then estimated per candidate and the top 100 features are kept, using a fractional threshold of 0.2.

Learner pool. Seven learners span four inductive families: gradient-boosted trees (XG-Boost, LightGBM) [11,13], a linear-booster XGBoost that extrapolates where trees cannot, a random forest that adds variance reduction with a diferent bias, and three neural models (MLP, LSTM, GRU), combined by stacked generalisation as in [12]. Each is trained per segment and tuned with Tree-structured Parzen Estimators [9,10].

The neural models capture price–promotion–calendar interactions, and the recurrent ones read lags as a gated sequence that can skip a stock-out gap or carry last year’s peak, so their residuals are partly decorrelated from the trees’.

Stacking. Configurations are ranked on validation behaviour alone, so in-sample fit cannot win a place, and the best $k = 3$ are combined as a convex stack,

$$
\hat { y } _ { i , t } = \sum _ { j = 1 } ^ { k } \omega _ { i } ^ { ( j ) } \hat { y } _ { i , t } ^ { ( j ) } , \qquad \omega _ { i } ^ { ( j ) } \geq 0 , \quad \sum _ { j } \omega _ { i } ^ { ( j ) } = 1 .\tag{3}
$$

We use equal weights $\omega = 1 / k$ , since fitted weights over a short validation window are high-variance [1]. The probabilistic values below are derived from the stack’s residuals.

Chebyshev and Cantelli bounds. Let $\boldsymbol { r } _ { i , t } = a _ { i , t } - \boldsymbol { \hat { y } } _ { i , t }$ over the validation window $\mathcal { N } _ { i }$ with bias $b _ { i }$ (mean residual) and dispersion $\sigma _ { i } ^ { 2 }$ (residual variance about $b _ { i } )$ . Demand residuals are skewed and heavy-tailed, so we use two inequalities that hold for any law with finite variance. Chebyshev bounds two-sided deviation, $\operatorname* { P r } ( | Y - \mu | \geq \kappa \sigma ) \leq \kappa ^ { - 2 }$ , so a tolerance � gives the symmetric multiplier $\kappa _ { \alpha } = \alpha ^ { - 1 / 2 }$ . Cantelli’s one-sided refinement, $\operatorname* { P r } ( Y - \mu \geq \kappa \sigma ) \leq ( 1 + \kappa ^ { 2 } ) ^ { - 1 }$ , gives the tighter $\kappa _ { \alpha } ^ { C } = \sqrt { ( 1 - \alpha ) / \alpha }$ for a single tail, and lets the two tails carry diferent tolerances $\alpha _ { \ell } , \alpha _ { u }$ , which matches the asymmetric costs of overstock and stock-out. The lower and upper bounds are

$$
\ell _ { i , t } = \operatorname* { m a x } \bigl ( 0 , \hat { y } _ { i , t } - \kappa _ { \ell } \sigma _ { i } \bigr ) , \qquad u _ { i , t } = \hat { y } _ { i , t } + \kappa _ { u } \sigma _ { i } ,\tag{4}
$$

with Chebyshev multipliers by default and Cantelli multipliers when the tails are set separately.

Quantiles and ofset. Percentile actions are running quantiles over the interval between the bounds,

$$
\begin{array} { r } { P _ { q } ( i , t ) = \ell _ { i , t } + q \big ( u _ { i , t } - \ell _ { i , t } \big ) , \qquad q \in \{ 0 . 1 , \ldots , 0 . 9 , 0 . 9 9 \} , } \end{array}\tag{5}
$$

which are monotone in $q ,$ , an ordering the resolution step later relies on. A further action removes the measured bias rather than widening the interval, $o _ { i , t } = \operatorname* { m a x } ( 0 , \hat { y } _ { i , t } + b _ { i } )$

Action set. The thirteen sanctioned alternatives are

$$
{ \mathcal { A } } = \{ \ell , P _ { 1 0 } , \ldots , P _ { 9 0 } , P _ { 9 9 } , u , o \} ,\tag{6}
$$

and the base forecast is a fourteenth option. Write $v _ { a } ( i , t )$ for the value of action � at row $( i , t ) . \mathrm { A }$ policy is admissible if it publishes only some $v _ { a } ( i , t )$ , so every published figure is one the pipeline has already produced; this is relaxed in exactly one audited case (Sect. 6.3). Each action has a plain reading in terms of the residual it corrects: � and the upper rungs restore compressed peak amplitude or an under-weighted event, ℓ and the lower rungs deepen a compressed trough, and � removes a persistent trend or level bias.

Concentration bounds are conservative $( \kappa _ { 0 . 1 } ^ { C } = 3$ against 1.64 for a one-sided Gaussian 95% bound), which is tolerable because the agent selects by measured reward and needs a well-ordered, suficiently wide menu rather than calibrated quantiles.

## 5 Attribution: Why Is the Forecast What It Is?

To avoid double-counting efects the ensemble already carries, each forecast is decomposed before correction:

$$
\hat { y } ^ { \mathrm { M L } } = B + T + S + E + D + M ,\tag{7}
$$

with baseline $B ,$ trend $T ,$ seasonality �, events $E ,$ commercial drivers � and a residual $M .$ , all non-negative and summing exactly to the pre-correction output. The agent and the planner then appear as signed layers that are never clipped,

$$
\mathrm { p u b l i s h e d } = \hat { y } ^ { \mathrm { M L } } + \underbrace { ( \hat { y } ^ { \mathrm { R L } } - \hat { y } ^ { \mathrm { M L } } ) } _ { \mathrm { a g e n t i c } } + \underbrace { ( \mathrm { p r e d } - \hat { y } ^ { \mathrm { R L } } ) } _ { \mathrm { p l a n n e r } } .\tag{8}
$$

Layers come from a per-series surrogate: columns are routed to one bucket each (holidays always to events), a two-stage ridge fits events first, and credit is a counterfactual $\phi _ { j } ( t ) = ( x _ { j } ( t ) - x _ { i } ^ { 0 } ( t ) ) \beta _ { j }$ against a family-specific reference (start of history for trend, cycle mean for seasonality, 0 for a binary event, median for a continuous driver), rescaled to the model output. The decomposition enters the agent’s state: it shows how much seasonality, trend and event lift the ensemble already carries, so an under-expressed peak can be told from one already learnt.

## 6 Temporal Reinforcement Learning for Forecast Value Add

## 6.1 Decision Process

We model refinement as a finite-action process $\langle S , \mathcal { R } , R , \gamma \rangle$ that is temporal: a decision is taken for each segment $g$ at each calendar position $p$ of the horizon, not once per series. The state $s _ { g , p }$ holds the candidate values $\{ v _ { a } \}$ , the calendar position, the structural seasonal index, event flags, growth diagnostics, the attribution layers and, centrally, the actuals, action values and commercial context of comparable past periods. An action is a choice $a \in { \mathcal { A } }$ . In precise terms the selection is an ofline contextual bandit evaluated by the direct method, without temporal credit assignment; we do not claim to solve a multi-step MDP.

## 6.2 Per-Step Reward from Comparable Past Periods

Absolute error alone favours a low, cautious forecast on spiky history, and matching the total alone tolerates months that are individually wrong, so the reward combines three components and is evaluated at each time step, because the best action difers by period.

Context-matched replay. For horizon position $p ,$ reference periods are retrieved from the series’ own history. Calendar alignment matches date minus $k$ years with a frequencyaware tolerance (15 days monthly, 3 weekly, 1 daily), and moving events such as Black Friday, Easter or Lunar New Year are matched to their previous occurrence even in a diferent month. Each reference records the value $v _ { a } ( i , p _ { k } )$ every action would have published against the realised $a _ { i , p _ { k } }$ and is weighted by how comparable its circumstances were,

$$
\begin{array} { r } { \omega _ { k } ( i , p ) \propto \rho ^ { k - 1 } \exp \Bigl ( - \frac { 1 } { \tau } \Bigr \| c _ { i , p } - c _ { i , p _ { k } } \Bigr \| ^ { 2 } \Bigr ) , } \end{array}\tag{9}
$$

where $c _ { i , p }$ stacks the holiday and event flags, the structural seasonal index, local trend growth, price index and promotion depth, and $\rho$ is a recency decay.

Reward components. For segment � and position $p ,$ each component is higher-is-better:

$$
\operatorname { A c c } _ { g , p } ( a ) = 1 - \frac { \sum _ { k } \sum _ { i \in g } \omega _ { k } | v _ { a } ( i , p _ { k } ) - a _ { i , p _ { k } } | } { \sum _ { k } \sum _ { i \in g } \omega _ { k } a _ { i , p _ { k } } + \epsilon } ,\tag{10}
$$

$$
{ \mathrm { B i a s } } _ { g , p } ( a ) = 1 - { \frac { \left| \sum _ { k } \sum _ { i \in g } \omega _ { k } \left( v _ { a } ( i , p _ { k } ) - a _ { i , p _ { k } } \right) \right| } { \sum _ { k } \sum _ { i \in g } \omega _ { k } a _ { i , p _ { k } } + \epsilon } } ,\tag{11}
$$

and $\operatorname { A d j } _ { g , p } ( a )$ , which is (10) computed against de-spiked actuals, where observations outside an inter-quantile fence $( Q _ { 0 . 2 } - 1 . 5 \mathrm { I Q R } , Q _ { 0 . 8 } + 1 . 5 \mathrm { I Q R } )$ are replaced by the series median. The step reward is their convex combination, and the step’s winner is its maximiser:

$$
R _ { g , p } ( a ) = w _ { \mathrm { a c c } } \mathrm { A c c } _ { g , p } + w _ { \mathrm { b i a s } } \mathrm { B i a s } _ { g , p } + w _ { \mathrm { a d j } } \mathrm { A d j } _ { g , p } , \qquad a _ { g , p } ^ { \star } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } R _ { g , p } ( a ) ,\tag{12}
$$

with weights (0.34, 0.33, 0.33) in our deployments. The winner is then applied to the same position of the current horizon, taking the value $v _ { a ^ { \star } } ( i , p )$ of the current action set.

Because the reward is per step, the agent stays central where the ensemble already tracks the period and moves towards the correcting bound where the replay shows a residual, so a segment receives a profile of actions over the calendar instead of one uniform shift, which is how seasonal amplitude, recurring events and one-of events are refined separately.

## 6.3 Structural Guard on the Winning Action

A replayed winner is evidence from the past, so it is checked against an independent structural reconstruction. Series with under one cycle borrow seasonality from similar series (never making a row protected), and a series with no sales in the last three periods keeps the base forecast.

Structural seasonal profile. On a complete calendar grid, one seasonal shape is fitted across all cycles as a rank-one model $a _ { y , p } \approx \ell _ { y } S _ { p }$ by alternating ratio estimates,

$$
\ell _ { y } = \frac { \sum _ { p } a _ { y , p } } { \sum _ { p } S _ { p } } , \qquad S _ { p } = \frac { \sum _ { \mathbf { y } } w _ { \mathbf { y } } ( a _ { y , p } / \ell _ { y } ) } { \sum _ { \mathbf { y } } w _ { \mathbf { y } } } , \qquad w _ { \mathbf { y } } = \rho ^ { Y - \mathbf { y } } \operatorname* { m i n } \Bigl ( 1 , \frac { c _ { \mathbf { y } } } { \bar { c } } \Bigr ) ,\tag{13}
$$

where $c _ { y }$ is the coverage of cycle � and � the latest cycle; unlike the classical index, it stays unbiased when cycles difer in level or the live cycle is unfinished. Positions are classified from $S _ { p } { } _ { ; }$ , never from the base forecast: a peak has $S _ { p } \geq 1 . 3 5$ (or is among the top two with $S _ { p } > 1 . 2 0 )$ , a trough has $S _ { p } \leq 0 . 7 0$ (or is among the bottom two with

$S _ { p } < 0 . 8 0 \}$ ). A seasonality classifier (autocorrelation at the seasonal lag, peak-to-trough ratio, consistency of the last two cycles) sets how far the reconstruction leans on history: strong seasonal series are rebuilt from a best-of-years shape times growth, weak ones blend prior-cycle anchors with the base forecast, and non-seasonal ones follow recent level and trend, with trivial edits skipped.

Structural target. The structural estimate is $F _ { \mathrm { s t r u c t } } = L \cdot S _ { p } \cdot E$ , with a blended level � and an event multiplier � for an event recurring, moving in or moving away. A prior-cycle replay $F _ { \mathrm { l y } }$ (calendar-aligned as above) uses a best-of-years anchor in high season,

$$
\begin{array} { r } { A _ { \mathrm { p e a k } } = \operatorname* { m a x } \big ( \mathrm { L Y } _ { 1 } ( 1 + g _ { \mathrm { p e a k } } ) , \mathrm { L Y } _ { 2 } ( 1 + g _ { \mathrm { u p } } ) ^ { 2 } \theta _ { 2 } \big ) , \qquad g _ { \mathrm { u p } } = \operatorname* { m a x } ( g , 0 ) , } \end{array}\tag{14}
$$

with $\theta _ { 2 }$ a trust factor on the older cycle, so one soft year cannot erase a spike the series has shown it can reach. The two are blended and the base forecast enters with a quality-weighted share:

$$
\begin{array} { r } { F _ { \mathrm { h i s t } } = w _ { s } F _ { \mathrm { s t r u c t } } + ( 1 - w _ { s } ) F _ { \mathrm { l y } } , } \end{array}\tag{15}
$$

$$
\begin{array} { r } { \hat { y } ^ { \mathrm { u n c } } = ( 1 - w _ { \mathrm { m l } } ) F _ { \mathrm { h i s t } } + w _ { \mathrm { m l } } \hat { y } , \qquad w _ { \mathrm { m l } } = w _ { c } ^ { \mathrm { b a s e } } ( 0 . 4 + 0 . 6 \eta ) , } \end{array}\tag{16}
$$

with $\eta \in [ 0 .$ , 1] the base model’s in-sample quality and $w _ { c } ^ { \mathrm { b a s e } }$ set by seasonality class. On protected peaks the base share is at most 5%, and it is set to zero when $\hat { y } < 0 . 5 A _ { \mathrm { p e a k } }$ that is, when the base horizon has not expressed the spike at all.

Protection and constraints. A protected peak is floored at 95% of its anchor and a protected trough capped at 110%. Three constraints then apply: (i) annual reconciliation, scaling up uniformly any series implying more than 12% de-growth against the calendarmatched prior year unless a decline is confirmed (two consecutive negative cycles, confidence $\ge ~ 0 . 6 .$ , growth $\leq - 1 5 \% )$ , with unconfirmed negative growth floored at −10%; (ii) shape coherence, smoothing the log-ratio to a probabilistic spine so its shape is borrowed without its level; and (iii) a magnitude cap of 2.0 times the largest historical actual (1.25 for recent launches). The order is fixed, row targets, reconciliation, shape fit, cap, audits, resolution, guard-rail, and later steps override earlier ones. Beyond the first cycle, the corrected output becomes the synthetic prior year for the next, the one place where an action at ℎ alters the state at $h + m$

Resolution. If $v _ { a ^ { \star } }$ respects protection and the constraints it is published; otherwise the constrained target ${ \hat { y } } ^ { \mathrm { u n c } }$ is resolved to the nearest sanctioned action, once and without preference ordering,

$$
a ( i , t ) = \arg \operatorname* { m i n } _ { a \in \mathcal { A } } \big | \widehat { y } _ { i , t } ^ { \mathrm { u n c } } - v _ { a } ( i , t ) \big | ,\tag{17}
$$

with ${ \hat { y } } ^ { \mathrm { u n c } }$ and the distance exported on every row.

Bound guard-rail. One audited exception to admissibility is available and disabled by default. A row may exceed � only if the series is in the top 80% by horizon volume, the row is seasonal, the resolved value sits at � yet below the prior year, and the series is neither a trough row nor a confirmed decline. The target is min(1.05 LY, 2�, cap), and the action is labelled upper\_bound\_x% with $x = 1 0 0 ( \mathrm { t a r g e t } / u - 1 )$ to flag the breach.

Table 1. The December episode. Top: per-step reward of selected actions from the context-matched replay. Bottom: the guard on the winning action and the published value.
<table><tr><td></td><td>Value va Reward</td><td>Action a  $R _ { g , \mathrm { D e c } } ( a )$ </td><td>l 140 0.408</td><td> $\boldsymbol { \mathrm { b a s e } } \hat { y }$  160 0.466</td><td> $P _ { 5 0 }$  210 0.612</td><td>0 240 0.699</td><td> $P _ { 9 0 }$  266 0.775</td><td> $P _ { 9 9 }$  278.6 0.811</td><td>u 280 0.816</td></tr><tr><td>#</td><td>Step</td><td></td><td>Calculation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1</td><td>Best-of-years anchor (14)</td><td colspan="6"> $\operatorname* { m a x } ( 3 3 0 \times 1 . 0 5 , 3 6 0 \times 1 . 0 4 ^ { 2 } \times 0 . 9 0 )$ </td><td> $A = 3 5 0 . 4$ </td></tr><tr><td>2</td><td>Protection check  $\mathrm { o n } a ^ { \star }$ </td><td colspan="6"> $v _ { u } = 2 8 0 < 0 . 9 5 \times 3 5 0 . 4 = 3 3 2 . 9$ </td><td>fails</td></tr><tr><td>3</td><td>Peak target</td><td colspan="6"> $\hat { y } = 1 6 0 < 0 . 5 A = 1 7 5 . 2 \Rightarrow w _ { \mathrm { m l } } = 0$ </td><td>350.4</td></tr><tr><td>4</td><td>Reconcile, shape, cap</td><td colspan="6">s = 1; peak keeps max; 350.4 ≤ 2 × 400</td><td> $\hat { y } ^ { \mathrm { u n c } } = 3 5 0 . 4$ </td></tr><tr><td>5</td><td>Resolution (17)</td><td colspan="6">nearest rung to 350.4; distance 70.4</td><td> $u = 2 8 0$ </td></tr><tr><td>6</td><td>Guard-rail test</td><td colspan="6"> $\mathrm { t o p { - } 8 0 ^ { c } / o , p e a k , a t } u , \mathrm { Y o Y } = 2 8 0 / 3 3 0 - 1 = - 1 5 . 2 ^ { \circ } { \mathrm { o } } $ </td><td>fres</td></tr><tr><td>7</td><td>Guard-rail target</td><td colspan="6">min(346.5, 560, 800)</td><td>346.5</td></tr><tr><td>8</td><td>Label</td><td colspan="6"> $x = \left( 3 4 6 . 5 / 2 8 0 - 1 \right) \times 1 0 0$ </td><td>upper_bound_23.8%</td></tr></table>

## 6.4 Governance Properties

Nothing is invented: every figure is in A or the labelled exception. Nothing is absurd: the cap bounds each row by its own history. Nothing is silent: each row carries its action, unconstrained value, distance and flags. Nothing is unowned: a named planner accepts or overrules, recorded as the planner layer of (8).

## 7 Worked Episode: One December Peak Row

We follow one decision through the loop of Sect. 6, with illustrative parameters: a monthly, strong-seasonal SKU in the top 80% by volume, guard-rail enabled, forecast for December 2026.

State. The structural index gives $S _ { \mathrm { D e c } } = 1 . 4 2$ , so December is a protected peak. The matched references are December $2 0 2 5 \left( \mathrm { L Y } _ { 1 } = 3 3 0 \right.$ , soft after a stock-out) and December 2024 $( \mathrm { L Y } _ { 2 } = 3 6 0$ , the real spike). The ensemble forecast is $\hat { y } = 1 6 0 \colon$ it has learnt that December is high, but the soft last year has pulled its level down. The largest historical actual is 400, and growth parameters are $g _ { \mathrm { p e a k } } = 5 \% , g _ { \mathrm { u p } } = 4 \% , \theta _ { 2 } = 0 . 9 0$

Actions. The ensemble’s probabilistic values for the row span $\ell = 1 4 0 \mathrm { t o } u = 2 8 0 ; \mathrm { b y } ($ (5) the rungs are $P _ { 1 0 } = 1 5 4 , P _ { 5 0 } = 2 1 0 , P _ { 9 0 } = 2 6 6$ and $P _ { 9 9 } = 2 7 8 . 6 $ , and the bias-corrected value is $o = 2 4 0$

Reward. With recency decay $\rho = 0 . 8$ and equal context distance, (9) weights the two references $\omega = ( 0 . 5 5 6 , 0 . 4 4 4 )$ , a weighted actual of 343.3. For brevity we let each reference year’s backtested values coincide with the current ones. Every action then lies below both actuals, so all three components of (12) reduce to $v _ { a } / 3 4 3 . 3$ and the reward is monotone in the action value (Table 1, top). The replay shows the December forecast has been persistently short; the winner is $\boldsymbol { a } ^ { \star } = \boldsymbol { u }$ with $R = 0 . 8 1 6$

Guard andpublication. The winner moves in the right direction but not far enough: 280 would ship a December 15.2% below a stock-out year and violates peak protection. The guard recovers the spike from LY through (14) and resolves it back to the ladder, again �. Because the top of the credible range lies below the prior year, the guard-rail fires and publishes 346.5, 5% above last year, labelled upper\_bound\_23.8% so the breach is auditable (Table 1, bottom). In the layers of (8) the published value is $1 6 0 + 1 8 6 . 5 + 0 \colon$ the ensemble’s 160, an agentic layer of +186.5 that restores the peak amplitude, and no planner override. The value then becomes the synthetic prior year for December 2027.

In short, the replay detects a residual the ensemble left at a recurring calendar position and picks the direction; the guard sizes it against the series’ own structure; the action set and the label constrain it to something a reviewer can trace.

## 8 Experiments

Setup. We evaluate on five customer deployments: two retail (eggs; sweets), two consumer-packaged-goods (sports goods; cookware) and one direct-to-consumer (skincare). Dificulty ranges from sweets (WMAPE near 0.1) to intermittent, event-driven sports goods (near 0.7). Nine monthly forecasts, January to September 2026, are scored against realised demand for three methods: Stat., the statistical forecast; ML, the stacked ensemble of Sect. 4; and RL, the agent of Sect. 6 applied on top of ML with the same reward weights everywhere. The metric is WMAPE (1) for each customer-month. “Window” denotes the unweighted mean ofthe nine monthly WMAPEs, and the five-customer mean weights customers equally. FVA (2) is computed from these figures, and Win denotes the forecast with the lowest WMAPE in a customer-month.

## 8.1 Results

Tables 2 and 3 and Fig. 2 report the results.

Value added on top of the ensemble. Over the window, RL beats the stacked ML forecast for every customer, with FVA between 4.3% (sports goods) and 16.7% (sweets), and 6.3% on the five-customer mean (0.390 against 0.416). The ensemble is itself strong, 13.3% better than the statistical forecast, so these gains are earned on top of a model that already learnt most of the structure. RL has the lowest WMAPE in 35 of the 45 customer-months (78%) and the stacked ML forecast in the other 10.

When the agent does not win. The ten ML wins are concentrated: four of the five customers in January, four in April, and one each in February (skincare) and March (eggs). In these months the agent’s loss is small, between 1.6% and 5.9%, and the fivecustomer mean is at parity (−0.4% and −0.9%). Both follow a calendar break (year-end; an early Easter in 2026), where references are least comparable and the ensemble was already close. There a coarse menu can only cost resolution.

Where the value concentrates. From May to September RL wins all 25 customermonths, and its largest gains over ML on the mean fall in May (10.7%), June (9.4%) and August (9.2%). June and August are the highest-error months for all three forecasts (Fig. 2), consistent with the residual being largest in peaks and event-heavy periods.

Table 2. WMAPE by customer and month (2026), rounded to two decimals; lower is better. The best forecast in each customer and column is in bold, and values tied after rounding are both in bold. Window is the mean of the nine months.
<table><tr><td></td><td>Jan</td><td>Feb</td><td>Mar</td><td>Apr</td><td>May</td><td>Jun</td><td>Jul</td><td>Aug</td><td>Sep</td><td>Window</td></tr><tr><td colspan="9">Retail / Eggs</td></tr><tr><td>RL</td><td>0.27</td><td>0.24</td><td>0.26</td><td>0.23</td><td>0.22</td><td>0.29</td><td>0.25</td><td>0.28</td><td>0.21</td><td>0.25</td></tr><tr><td>ML</td><td>0.26</td><td>0.27</td><td>0.26</td><td>0.25</td><td>0.25</td><td>0.33</td><td>0.28</td><td>0.31</td><td>0.25</td><td>0.27</td></tr><tr><td>Stat.</td><td>0.32</td><td>0.28</td><td>0.30</td><td>0.29</td><td>0.31</td><td>0.39</td><td>0.33</td><td>0.38</td><td>0.28</td><td>0.32</td></tr><tr><td colspan="9">Retail / Sweets</td></tr><tr><td>RL</td><td>0.11</td><td>0.09</td><td>0.11</td><td>0.09</td><td>0.09</td><td>0.12</td><td>0.10</td><td>0.11</td><td>0.09</td><td>0.10</td></tr><tr><td>ML</td><td>0.10</td><td>0.11</td><td>0.14</td><td>0.09</td><td>0.12</td><td>0.16</td><td>0.12</td><td>0.14</td><td>0.11</td><td>0.12</td></tr><tr><td>Stat.</td><td>0.14</td><td>0.14</td><td>0.17</td><td>0.16</td><td>0.18</td><td>0.20</td><td>0.18</td><td>0.20</td><td>0.16</td><td>0.17</td></tr><tr><td colspan="9">CPG / Sports</td></tr><tr><td>RL</td><td>0.70</td><td>0.66</td><td>0.61</td><td>0.70</td><td>0.60</td><td>0.72</td><td>0.64</td><td>0.68</td><td>0.54</td><td>0.65</td></tr><tr><td>ML</td><td>0.68</td><td>0.70</td><td>0.64</td><td>0.69</td><td>0.66</td><td>0.77</td><td>0.68</td><td>0.73</td><td>0.57</td><td>0.68</td></tr><tr><td>Stat.</td><td>0.78</td><td>0.71</td><td>0.71</td><td>0.79</td><td>0.72</td><td>0.86</td><td>0.75</td><td>0.82</td><td>0.64</td><td>0.75</td></tr><tr><td colspan="9">CPG / Cookware</td><td></td></tr><tr><td>RL</td><td>0.49</td><td>0.43</td><td>0.46</td><td>0.39</td><td>0.42</td><td>0.51</td><td>0.44</td><td>0.48</td><td>0.43</td><td>0.45</td></tr><tr><td>ML</td><td>0.48</td><td>0.47</td><td>0.50</td><td>0.38</td><td>0.46</td><td>0.57</td><td>0.47</td><td>0.53</td><td>0.47</td><td>0.48</td></tr><tr><td>Stat.</td><td>0.56</td><td>0.51</td><td>0.56</td><td>0.48</td><td>0.54</td><td>0.64</td><td>0.55</td><td>0.60</td><td>0.52</td><td>0.55</td></tr><tr><td colspan="9">D2C / Skincare</td><td></td></tr><tr><td>RL</td><td>0.54</td><td>0.47</td><td>0.52</td><td>0.46</td><td>0.43</td><td>0.58</td><td>0.51</td><td>0.53</td><td>0.46</td><td>0.50</td></tr><tr><td>ML</td><td>0.58</td><td>0.46</td><td>0.55</td><td>0.45</td><td>0.48</td><td>0.62</td><td>0.54</td><td>0.58</td><td>0.50</td><td>0.53</td></tr><tr><td>Stat.</td><td>0.63</td><td>0.54</td><td>0.62</td><td>0.55</td><td>0.56</td><td>0.70</td><td>0.61</td><td>0.66</td><td>0.57</td><td>0.60</td></tr><tr><td colspan="9">Mean of the five customers</td><td></td></tr><tr><td>RL 0.42</td><td></td><td>0.38</td><td>0.39</td><td>0.37</td><td>0.35</td><td>0.44</td><td>0.39</td><td>0.42</td><td>0.35</td><td>0.39</td></tr><tr><td>ML</td><td>0.42</td><td>0.40</td><td>0.42</td><td>0.37</td><td>0.39</td><td>0.49</td><td>0.42</td><td>0.46</td><td>0.38</td><td>0.42</td></tr><tr><td>Stat.</td><td>0.49</td><td>0.44</td><td>0.47</td><td>0.45</td><td>0.46</td><td>0.56</td><td>0.48</td><td>0.53</td><td>0.43</td><td>0.48</td></tr></table>

Consistency. RL beats the statistical forecast in all 45 customer-months (window FVA 13.7–40.8%, 18.7% on the mean). Relative gains over ML are largest for the low-error sweets customer (16.7%) and smallest for the high-error sports customer (4.3%), but in absolute terms the order reverses: the reduction is 0.020 WMAPE points for sweets and 0.029 to 0.030 for sports goods and cookware (Table 2, Window). Relative FVA therefore understates the value on hard categories.

## 8.2 Limitations

Five customers over nine months give breadth, not generality, and at most one instance of each peak. The agent does not win every month, and the window figures weight customers and months equally rather than by volume. Only accuracy is reported; a bias evaluation on the same customers is a natural next step. Per-step winners over-fit at series level, so we decide at segment level. Listing changes, confirmed orders and supply constraints remain the planner’s responsibility.

## 9 Conclusion

We framed the refinement of a probabilistic ML forecast as reward-driven selection over the ensemble’s own distribution-free values. The ensemble learns seasonality, trend and events; the agent scores each value at every time step against comparable past periods and moves the published value where the past shows a residual, refining seasonal amplitude, recurring and one-of events separately. Across five customer deployments the agent added value on top of a strong stacked ensemble for every customer, 6.3% on average and up to 16.7%, was never worse than the statistical forecast, and was at parity with the ensemble in the months where it did not win. Future work includes a multi-step policy with temporal credit assignment and a learnt fallback to the central rung when replay evidence is weak.

Table 3. Forecast Value Add of RL, in percent: how much lower the RL error was than the reference forecast. Negative values mean RL was worse. FVA is computed from unrounded WMAPE; the mean row uses the five-customer mean WMAPE of Table 2.
<table><tr><td></td><td>Jan</td><td>Feb</td><td>Mar</td><td>Apr</td><td>May</td><td>Jun</td><td>Jul</td><td>Aug</td><td>Sep</td><td>Window</td></tr><tr><td colspan="9">FVA of RL versus the stacked ML forecast</td><td></td></tr><tr><td>Retail / Eggs</td><td>-3.8</td><td>11.1</td><td>-2.0</td><td>8.0</td><td>12.0</td><td>12.1</td><td>10.7</td><td>9.7</td><td>14.3</td><td>8.2</td></tr><tr><td>Retail / Sweets</td><td>-5.9</td><td>17.5</td><td>18.2</td><td>-4.7</td><td>26.1</td><td>25.6</td><td>16.8</td><td>21.4</td><td>22.5</td><td>16.7</td></tr><tr><td>CPG / Sports</td><td>-2.9</td><td>5.7</td><td>4.7</td><td>-2.2</td><td>9.1</td><td>6.5</td><td>5.9</td><td>6.8</td><td>5.3</td><td>4.3</td></tr><tr><td>CPG / Cookware</td><td>-2.5</td><td>7.5</td><td>8.0</td><td>-1.6</td><td>8.7</td><td>10.5</td><td>6.4</td><td>9.4</td><td>7.5</td><td>6.3</td></tr><tr><td>D2C / Skincare</td><td>6.9</td><td>-3.3</td><td>5.5</td><td>-2.7</td><td>10.4</td><td>6.5</td><td>5.6</td><td>8.6</td><td>8.0</td><td>5.3</td></tr><tr><td>Mean</td><td>-0.4</td><td>5.5</td><td>5.8</td><td>-0.9</td><td>10.7</td><td>9.4</td><td>7.2</td><td>9.2</td><td>8.7</td><td>6.3</td></tr><tr><td colspan="9">FVA of RL versus the statistical forecast</td><td></td><td></td></tr><tr><td>Retail / Eggs</td><td>15.6</td><td>14.3</td><td>13.3</td><td>20.7</td><td>29.0</td><td>25.6</td><td>24.2</td><td>26.3</td><td>25.0</td><td>21.9</td></tr><tr><td>Retail / Sweets</td><td>21.7</td><td>34.7</td><td>34.9</td><td>43.8</td><td>51.4</td><td>40.8</td><td>44.7</td><td>45.0</td><td>44.9</td><td>40.8</td></tr><tr><td>CPG / Sports</td><td>10.3</td><td>7.0</td><td>14.1</td><td>11.4</td><td>16.7</td><td>16.3</td><td>14.7</td><td>17.1</td><td>15.6</td><td>13.7</td></tr><tr><td>CPG / Cookware</td><td>12.5</td><td>15.7</td><td>17.9</td><td>18.7</td><td>22.2</td><td>20.3</td><td>20.0</td><td>20.0</td><td>17.3</td><td>18.3</td></tr><tr><td>D2C / Skincare</td><td>14.3</td><td>13.0</td><td>16.1</td><td>16.4</td><td>23.2</td><td>17.1</td><td>16.4</td><td>19.7</td><td>19.3</td><td>17.3</td></tr><tr><td>Mean</td><td>13.2</td><td>13.3</td><td>16.9</td><td>17.6</td><td>23.9</td><td>20.5</td><td>19.8</td><td>21.8</td><td>20.3</td><td>18.7</td></tr></table>

![](images/1fe69932f166bb4bd12777f487fc91135acac60c82cd88debca82c16d17593f9.jpg)  
Fig. 2. Five-customer mean WMAPE by month. RL and ML coincide in January and April; the gap opens in the higher-error months of June and August.

Acknowledgments. The authors would like to thank Ankit Raj for his contributions towards RL architecture improvements; Ankit Solanki and Vasu Bhanderi for front development; Siddharth Shahi, Namrata Gupta and Sharad LM for helpful discussions and insights over the methodology and implementation.

## References

1. Makridakis, S., Spiliotis, E., Assimakopoulos, V.: M5 accuracy competition: results, findings, and conclusions. International Journal of Forecasting 38(4), 1346–1364 (2022)

2. Makridakis, S., Spiliotis, E., Assimakopoulos, V., Chen, Z., Gaba, A., Tsetlin, I., Winkler, R.L.: The M5 uncertainty competition: results, findings and conclusions. International Journal of Forecasting 38(4), 1365–1385 (2022)

3. Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T.: DeepAR: probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting 36(3), 1181– 1191 (2020)

4. Oreshkin, B.N., Carpov, D., Chapados, N., Bengio, Y.: N-BEATS: neural basis expansion analysis for interpretable time series forecasting. In: International Conference on Learning Representations (2020)

5. Lim, B., Arık, S.Ö., Loef, N., Pfister, T.: Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting 37(4), 1748–1764 (2021)

6. Wen, R., Torkkola, K., Narayanaswamy, B., Madeka, D.: A multi-horizon quantile recurrent forecaster. arXiv:1711.11053 (2017)

7. Gneiting, T., Raftery, A.E.: Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378 (2007)

8. Koenker, R., Bassett, G.: Regression quantiles. Econometrica 46(1), 33–50 (1978)

9. Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B.: Algorithms for hyper-parameter optimization. In: Advances in Neural Information Processing Systems, vol. 24, pp. 2546–2554 (2011)

10. Bergstra, J., Yamins, D., Cox, D.D.: Making a science of model search: hyperparameter optimization in hundreds of dimensions for vision architectures. In: International Conference on Machine Learning, pp. 115–123 (2013)

11. Chen, T., Guestrin, C.: XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. ACM, New York (2016)

12. Verma, Ankur: Consumer behaviour in retail: next logical purchase using deep neural network. arXiv:2010.06952 (2020)

13. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y.: LightGBM: a highly eficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems, vol. 30, pp. 3146–3154 (2017)

14. Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., Shang, H.L.: Optimal combination forecasts for hierarchical time series. Computational Statistics and Data Analysis 55(9), 2579– 2589 (2011)

15. Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems, vol. 30, pp. 4765–4774 (2017)

16. Li, L., Chu, W., Langford, J., Schapire, R.E.: A contextual-bandit approach to personalized news article recommendation. In: Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, New York (2010)

17. Sutton, R.S., Barto, A.G.: Reinforcement Learning: An Introduction. 2nd edn. MIT Press, Cambridge (2018)

18. Gilliland, M.: The Business Forecasting Deal: Exposing Myths, Eliminating Bad Practices, Providing Practical Solutions. Wiley, Hoboken (2010)

19. Morlidge, S.: Forecast quality in the supply chain. Foresight: The International Journal of Applied Forecasting 33, 26–31 (2014)