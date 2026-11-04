# Decomposing Rank Reversals under Temporal Nonstationarity: Alignment, Aggressiveness, and Instability

Deying Song Carnegie Mellon University Pittsburgh, PA, USA

## Abstract

Under temporal nonstationarity, model performance rankings reverse across training horizons due to heterogeneous sensitivities to memory. While existing literature addresses adaptive lookbacks, efective model complexity, and empirical forecast evaluation in isolation, no unified framework explains the structural mechanics driving these rank reversals. We bridge this gap by deriving a modelagnostic decomposition of out-of-sample R<sup>2</sup> into target alignment, miscalibrated aggressiveness, and predictive instability. Within this framework, we mathematically characterize complexity phase boundaries, establish closed-form optimal-memory equations for ridge regression under local drift, and extend these guarantees to complex model architectures via surrogate bounds. We validate these theoretical dynamics by benchmarking 14 linear, tree-based, and neural architectures across large-scale financial panels. In doing so, our framework demonstrates that empirical model rank reversals are driven by predictable surges in miscalibrated aggressiveness, providing a rigorous mechanistic diagnosis of generalization under temporal shift.

## 1 INTRODUCTION

Reliable forecasting under temporal nonstationarity requires balancing the estimation variance of restricted training windows against the bias induced by obsolete historical data (Dahlhaus, 1997; Pesaran and Timmermann, 2007; Giraitis et al., 2013). This tradeof is exceptionally severe in return prediction, where predictive signals are notoriously weak and transient (Welch and Goyal, 2008; Campbell and Thompson, 2008; Smith and Timmermann, 2021). While extensive literature addresses adaptive memory selection, sys-

Xueying Ding Carnegie Mellon University Pittsburgh, PA, USA

tematically explaining why the relative performance of models reverses across diferent training horizons requires formally linking theoretical dynamics with empirical phenomena. To establish this mechanistic link, we address two fundamental questions: (1) Which specific forecast components drive the performance diferential between models at a fixed memory? and (2) Can we mathematically predict, directly from the underly ing statistical dynamics, the critical memory threshold at which performance rankings reverse?

Although the literature addresses facets of this problem, existing approaches remain disconnected. In nonstationary settings, dynamic-regret analyses balance adaptation against data eficiency via sliding windows and forgetting factors, typically assuming known variation budgets (Cheung et al., 2019; Zhao et al., 2020; Yuan and Lamperski, 2020; Yang et al., 2016; Zhao and Zhang, 2021; Baby and Wang, 2023). The constraint is recently relaxed by adaptive-lookback estimators (Huang and Wang, 2023; Han et al., 2024; Baby et al., 2025). Concurrently, statistical learning theory links generalization bounds to efective complexity (Caponnetto and De Vito, 2007; Hsu et al., 2012; Dobriban and Wager, 2018; Bousquet and Elisseef, 2002). Finally, empirical forecast evaluation provides diagnostics for bias and calibration (Mincer and Zarnowitz, 1969; Theil, 1966; Gneiting and Resin, 2023; Waghmare and Ziegel, 2026), alongside statistical tests for predictive ability (Diebold and Mariano, 1995; Giacomini and White, 2006; White, 2000; Hansen, 2005; Hansen et al., 2011; Hewamalage et al., 2021). However, these paradigms operate in isolation. Regret and stability bounds do not predict when highcapacity models will surpass simpler baselines under shifting histories, while empirical diagnostics quantify performance gaps without isolating their structural drivers. Consequently, no existing framework systematically links training memory and model complexity to explain the rank reversals observed in nonstationary environments.

In this work, we close that gap in three steps. First, we derive a model-agnostic decomposition of out-of sample $R ^ { 2 }$ that isolates target alignment from penalties for miscalibrated aggressiveness (excess forecast magnitude) and predictive instability (variance across replicates). Second, we analytically characterize how the training horizon governs each component, estab lishing a complexity phase boundary and a closedform optimal-memory equation for ridge regression under local drift, with extensions to nonlinear architectures via surrogate transfer bounds. Third, we validate these theoretical mechanics through controlled simulations and extensive empirical evaluation, benchmarking 14 linear, tree-based, and neural specifications across both macroeconomic industry portfolios (Fama and French, 1997) and a large-scale, out-ofdomain equity panel (Zhang et al., 2026).

In summary, our primary contributions are:

• A mechanistic theory of rank reversals. We provide a unified theoretical framework that mathematically links training memory and model capacity to out-of-sample generalization, formally proving why and when high-capacity models lose their predictive advantage under distribution shift.

• Empirical diagnosis of performance reversals. Across large-scale financial panels, we demonstrate that when high-capacity models underperform simpler baselines, it is driven by predictable surges in miscalibrated aggressiveness.

## 2 RELATED WORK

Learning under temporal variation. Learning under nonstationarity balances rapid adaptation against historical data eficiency. Sliding windows and restarts bound dynamic regret but typically assume known variation budgets (Cheung et al., 2019; Zhao et al., 2020), while forgetting factors trade static and dynamic regret (Yuan and Lamperski, 2020; Yang et al., 2016; Zhao and Zhang, 2021; Baby and Wang, 2023). Recent advances deploy adaptive lookbacks for unknown shifts (Huang and Wang, 2023; Han et al., 2024; Baby et al., 2025). Despite these methods, the conditions under which a complex model systematically outperforms a simpler baseline across varying histories remain theoretically underspecified. Our complexity phase boundary analytically identifies when a model’s approximation advantage ofsets its heightened vulnerability to memory-dependent variance and drift.

Efective complexity and forecast instability. Generalization under finite data depends on efective complexity and algorithmic stability. While spectral dimension and signal strength govern ridge prediction risk (Caponnetto and De Vito, 2007; Hsu et al., 2012; Dobriban and Wager, 2018), regularization also controls sensitivity, rendering raw parameter counts inadequate (Hsu et al., 2012; Bousquet and Elisseef, 2002).

Standard risk bounds, however, do not translate this sensitivity into the realized out-of-sample $R ^ { 2 }$ cost induced by variation across forecast replicates. Our finite-sample identity bridges this gap by isolating predictive instability as an explicit additive penalty, proving that replicate ensembling eliminates it entirely.

Local change and estimation memory. Timevarying regressions show that recent data capture drift while longer histories reduce variance (Dahlhaus, 1997; Vogt, 2012), framing window selection as an old-data bias versus sampling error tradeof (Pesaran and Timmermann, 2007; Giraitis et al., 2013). Forecasting under instability also calls for comparisons that allow relative model performance to change over time (Giacomini and Rossi, 2015, 2016). Although these frameworks formalize the bias-variance tension, standard heuristics overlook how regularization structurally filters temporal drift across predictor directions to shift the optimal memory horizon. Under a local-drift specification, we derive an exact optimal-memory equation that couples spectral efective dimension, filtered drift, and the bias-drift cross-term.

Calibration, scale, and combination. Classical diagnostics isolate bias and scale, while modern scoring rules separate calibration, discrimination, and uncertainty (Mincer and Zarnowitz, 1969; Theil, 1966; Gneiting and Resin, 2023; Waghmare and Ziegel, 2026). Forecast combinations similarly handle structural breaks across predictor sets (Diebold and Pauly, 1987; Elliott and Timmermann, 2004; Chen and Maung, 2020). Nevertheless, existing frameworks do not jointly isolate the impacts of weak target align ment, miscalibrated magnitude, and predictive instability on out-of-sample $\bar { R ^ { 2 } }$ . Our score identity achieves this tripartite separation analytically, quantifying the oracle $R ^ { 2 }$ gain available solely through scaling forecast aggressiveness.

Comparing and ranking forecasts. Accuracy tests evaluate unconditional loss diferences or map predictive ability to information sets (Diebold and Mariano, 1995; Giacomini and White, 2006). To counter data-snooping biases, specialized tests and confidence sets correct for multiple comparisons (White, 2000; Hansen, 2005; Hansen et al., 2011), while rankstability analyses reveal sensitivity to evaluation design (Hewamalage et al., 2021). While these frameworks rigorously quantify statistical uncertainty, they are diagnostic rather than explanatory: they detect performance gaps without isolating the driving forecast properties. We augment dependence-aware comparisons with our attribution identity—decomposing gaps into alignment, aggressiveness, and instability— to mechanistically explain memory-induced rank reversals.

Return prediction and model comparisons. Equity premium forecasts routinely fail to outperform historical means, though shrinkage reduces prediction error (Welch and Goyal, 2008; Campbell and Thompson, 2008). While nonlinear interactions isolate weak signals (Goktas et al., 2026; Capponi et al., 2025), structural breaks degrade accuracy (Gu et al., 2020; Smith and Timmermann, 2021). Broad benchmarks corroborate this predictability limit, showing that transaction costs and market states fundamentally reorder rankings (Bejarano et al., 2025; Saly-Kaufmann et al., 2026; Zhang et al., 2026). Because benchmarks often conflate model selection with disparate tuning budgets, it remains unresolved whether training history alone can invert rankings under strictly matched information sets. By modulating memory across linear, tree-based, and neural models, we demonstrate that history alone reverses rankings, tracing these shifts directly to target alignment, aggressiveness, and instability.

## 3 FORECAST COMPONENTS ANDCOMPLEXITY

To formalize how training history shapes prediction, we introduce a model-agnostic decomposition of out-of-sample $R ^ { 2 }$ to mathematically isolate the drivers of rank reversal (Sections 3.1 and 3.2), and establish a complexity phase boundary that yields an exact optimal-memory equation for ridge regression under local drift (Section 3.3) alongside explicit transfer bounds for nonlinear architectures (Section 3.4). Throughout this section, we quantify training history via the memory h, defined as the number of most recent observations used to train a model. Full derivations and proofs for all theoretical results are in Appendix A.

![](images/dd6ed480a9921815d7e01232e0105d5c40701854a656bd25b7201375d39d69c2.jpg)

![](images/73f5eeb696e85bf60a219cb4d740240d1f8b2935e7da7d39d2823dee48ce4383.jpg)  
Figure 1: Theoretical leading-order dynamics with normalized parameters. (A) The complex class achieves lower risk when its history-independent advantage $\Delta _ { 0 }$ exceeds the complexity cost $\phi ( h )$ ; bands denote remainder bounds. (B) Variance $V / h$ decays and the drift penalty $W h ^ { 2 }$ grows with memory $h ,$ creating an interior risk minimum at $h ^ { * }$

## 3.1 Forecast-score decomposition

Let $\textbf { y } \in \mathbb { R } ^ { n }$ represent the true outcomes on a common evaluation sample, assuming a non-zero magnitude $Y _ { 2 } = \| \mathbf { y } \| ^ { 2 } > 0$ . Suppose a forecasting procedure is fitted M times—varying only the random seeds or training data splits—to produce M replicate forecasts. Let $\widehat { \mathbf { y } } _ { m } \in \mathbb { R } ^ { n }$ denote the m-th replicate, with an ensemble mean $\begin{array} { r } { \overline { { \mathbf { y } } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \widehat { \mathbf { y } } _ { m } } \end{array}$

We evaluate performance using the out-of-sample $R ^ { 2 }$ relative to a naive zero prediction. The average score across all replicates is $\begin{array} { r } { \overline { { R ^ { 2 } } } = 1 - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \Vert \mathbf { y } - \mathbf { \phi } } \end{array}$ ${ \widehat { \bf y } } _ { m } \| ^ { 2 } / Y _ { 2 }$ . To isolate what drives this performance, we characterize the geometry of the forecasts through three components: target alignment $^ { a , }$ forecast aggressiveness $^ { g , }$ and predictive instability $u ,$ defined as

$$
a = { \frac { \left. \mathbf { y } , { \overline { { \mathbf { y } } } } \right. } { \left\| \mathbf { y } \right\| \left\| { \overline { { \mathbf { y } } } } \right\| } } , \quad g = { \frac { \left\| { \overline { { \mathbf { y } } } } \right\| } { \left\| \mathbf { y } \right\| } } , \quad u = { \frac { { \frac { 1 } { M } } \sum _ { m = 1 } ^ { M } \left\| { \widehat { \mathbf { y } } } _ { m } - { \overline { { \mathbf { y } } } } \right\| ^ { 2 } } { Y _ { 2 } } } .\tag{1}
$$

Conceptually, alignment a measures direction (the cosine similarity between the target and the mean forecast), aggressiveness $g$ measures relative scale (the overall magnitude of the mean forecast compared to the target), and instability u measures variance (the squared dispersion of the individual replicates around their mean, normalized by the target energy $Y _ { 2 } )$

Replicate-ensemble gain. Expanding the squared error and noting that replicate deviations sum to zero $\begin{array} { r } { \big ( \sum _ { m } ( \widehat { \mathbf { y } } _ { m } - \overline { { \mathbf { y } } } ) = 0 \big ) } \end{array}$ yields an exact decomposition:

$$
\overline { { { R ^ { 2 } } } } = 2 a g - g ^ { 2 } - u .\tag{2}
$$

If $\mathbf { \overline { { y } } } = 0$ , we set $a = 0$ by convention. With this convention, the identity $a g = \langle \mathbf { y } , { \overline { { \mathbf { y } } } } \rangle / Y _ { 2 }$ holds in all cases. Performance thus separates into a positive alignment reward $\left( 2 a g \right)$ , penalized by aggressiveness $( g ^ { 2 } )$ and instability (u) costs. The ensemble mean y achieves a score of $2 a g \ : - \ : g ^ { 2 }$ , exceeding the average replicate score by u. Ensembling thus eliminates the instability penalty while preserving the alignment and aggressiveness terms.

Proposition 1 (optimal common scalar). Rescaling forecasts $\mathbf { p } _ { m }$ by a factor γ yields a mean score $S ( \gamma ) = 2 \gamma c - \gamma ^ { 2 } q .$ , where $c = \mathrm { E } _ { m } \langle \mathbf { y } , \mathbf { p } _ { m } \rangle / Y _ { 2 }$ and $q = \mathrm { E } _ { m } \| \mathbf { p } _ { m } \| ^ { 2 } / Y _ { 2 } > 0$ . On any closed interval $\mathcal { T } \subseteq \mathbb { R }$ the unique maximizer is the projection of $c / q$ onto $\mathcal { T } ,$

$$
\gamma _ { \mathcal { T } } ^ { * } = \Pi _ { \mathcal { T } } ( c / q ) .\tag{3}
$$

In essence, this calculates the maximum potential score improvement if the overall magnitude of the predictions is perfectly corrected. Unconstrained scaling $( \mathcal { T } = \mathbb { R } )$ maximizes the score at $c ^ { 2 } / q$

For individual replicates, $\gamma ^ { * } = a g / ( g ^ { 2 } + u )$ . For the ensemble mean, $\gamma ^ { * } = a / g$

## 3.2 Complexity phase boundary

Rank reversal. Suppose we train two models, A and B, using the h most recent observations and evaluate them on the same target, y. For each model $k \in \{ A , B \}$ , we can break down its average prediction score, $\overline { { R _ { k } ^ { 2 } } } ( h )$ , into three intuitive components defined in (1): alignment $a _ { k } ( h )$ , aggressiveness $g _ { k } ( h )$ , and instability $u _ { k } ( h )$

Using the identity from (2), we can calculate the exact performance gap, $\Delta _ { A B } ( h )$ , between the two models:

$$
\begin{array} { r l } {  { \Delta _ { A B } ( h ) = \overline { { R _ { A } ^ { 2 } } } ( h ) - \overline { { R _ { B } ^ { 2 } } } ( h ) } } \\ & { = 2 \{ a _ { A } g _ { A } - a _ { B } g _ { B } \} - \{ g _ { A } ^ { 2 } - g _ { B } ^ { 2 } \} } \\ & { \phantom { = } - \{ u _ { A } - u _ { B } \} . } \end{array}\tag{4}
$$

Model A outperforms model B at memory h whenever this gap is positive $( \Delta _ { A B } ( h ) > 0 )$ . A rank reversal occurs exactly when the performance gap changes sign, meaning $\Delta _ { A B } ( h _ { 1 } ) \Delta _ { A B } ( h _ { 2 } ) < 0$

Complexity phase boundary. Let $\mathcal { M } _ { 0 }$ be a simple model class and $\mathcal { M } _ { 1 }$ a more complex class. Let $\mathcal { L } _ { k } ( h )$ denote the excess loss of class $k \in \{ 0 , 1 \}$ at memory $h ,$ that is, the expected squared error against the current conditional mean. We assume this loss expands as

$$
\mathcal { L } _ { k } ( h ) = \mathcal { A } _ { k } + B _ { k } + \frac { V _ { k } } { h } + K _ { k } h + W _ { k } h ^ { 2 } + e _ { k } ( h ) ,\tag{5}
$$

with a known remainder bound $| e _ { k } ( h ) | \leq r _ { k } ( h )$ . Here, $\mathcal { A } _ { k }$ is the optimal approximation error, and $B _ { k }$ is a static, memory-independent bias. The terms $V _ { k } / h$ $K _ { k } h$ , and $W _ { k } \dot { h ^ { 2 } }$ capture the core memory tradeof: estimation variance (which decays with h) versus linear and quadratic drift penalties (which grow as older data is included).

Setting $\Delta _ { 0 } = ( A _ { 0 } + B _ { 0 } ) - ( A _ { 1 } + B _ { 1 } )$ isolates the historyindependent advantage of the complex class. Defining the coeficient diferences $\Delta V = { \bar { V _ { 1 } } } - V _ { 0 } , \Delta K = K _ { 1 } -$ $K _ { 0 } .$ , and $\Delta W = W _ { 1 } - W _ { 0 }$ yields its history-dependent cost, $\phi ( h ) = \Delta V / h + \Delta K h + \Delta W h ^ { 2 } ~ \mathrm { ( F i g . ~ 1 A ) }$

Proposition 2 (phase boundary). The complex class has strictly lower loss at memory h if its historyindependent advantage overcomes the complexity cost and remainder bounds, meaning $\Delta _ { 0 } > \phi ( h ) + r _ { 0 } ( h ) +$ $r _ { 1 } ( h )$ . Conversely, the simple class performs better if $\Delta _ { 0 } < \phi ( h ) - r _ { 0 } ( h ) - r _ { 1 } ( h )$

Setting aside the remainders, suppose the complex class incurs higher variance $( \Delta V > 0 )$ and quadratic drift $\left( \Delta W \quad > \quad 0 \right)$ penalties. The cost function $\phi ( h )$ is then strictly convex (U-shaped), reaching a unique minimum $\phi _ { \mathrm { m i n } }$ at a memory $h _ { \mathrm { c o s t } }$ that solves

$2 \Delta W h _ { \mathrm { c o s t } } ^ { 3 } + \Delta K h _ { \mathrm { c o s t } } ^ { 2 } = \Delta V$ . If $\Delta _ { 0 } < \phi _ { \mathrm { m i n } }$ , the complex class never performs better. If $\Delta _ { 0 } ~ > ~ \phi _ { \mathrm { m i n } }$ , it outperforms the simple class within an optimal memory window bounded by the two positive roots of $\Delta W h ^ { 3 } + \Delta K h ^ { 2 } - \Delta _ { 0 } h + \Delta V = 0 .$

When linear drift diferences vanish $( \Delta K = 0 )$ , this memory scales as $h _ { \mathrm { c o s t } } = ( \Delta V / 2 \Delta W ) ^ { 1 / 3 }$ with minimum cost $\phi _ { \operatorname* { m i n } } = 3 \cdot 2 ^ { - 2 / 3 } ( \Delta V ) ^ { 2 / 3 } ( \Delta W ) ^ { 1 / 3 }$ . Because varying the training memory dynamically shifts this variance-drift tradeof, parameter count alone does not determine the prevailing model. In summary, the complex class only performs better within a specific window of training memory where its baseline advantage outweighs the escalating penalties of variance and data drift. (Detailed proofs and degenerate cases are in Propositions A.2 and A.3 of Appendix A.2).

## 3.3 Exact prediction risk and optimal memory for ridge regression

We first bound the prediction risk of ridge regression under coeficient drift. Consider a forecast date t and a training history of h observations $( x _ { t - j } , y _ { t - j } )$ for $j ~ = ~ 1 , \dots , h$ , with predictors $\begin{array} { r } { \boldsymbol { x } _ { t - j } \in \mathbb { R } ^ { p } . } \end{array}$ We model nonstationarity as a local linear drift: $y _ { t - j } =$ $x _ { t - j } ^ { \top } \beta _ { t - j } + \varepsilon _ { t - j } ,$ , where the historical coeficients evolve as $\beta _ { t - j } = \beta _ { t } - j { \dot { \beta } } _ { t } + r _ { t , j }$ . Here, $\beta _ { t }$ represents the current target coeficient, $\dot { \beta } _ { t }$ the constant per-period drift, $\boldsymbol { r } _ { t , j }$ the nonlinear drift remainder, and $\varepsilon _ { t - j }$ the observation noise.

Define the sample second-moment matrix $\begin{array} { r l } { S _ { h } } & { { } = } \end{array}$ $\begin{array} { r } { h ^ { - 1 } \sum _ { i } x _ { t - j } x _ { t - j } ^ { \top } , } \end{array}$ , the ridge inverse $A _ { h } = ( S _ { h } + \lambda I ) ^ { - 1 }$ for penalty $\lambda > \mathrm { 0 }$ , the ridge filter $G _ { h } = A _ { h } S _ { h }$ , and the mean temporal lag $\mu _ { h } = ( h + 1 ) / 2$ . We measure prediction diferences using the local design norm $\| v \| _ { S _ { h } } ^ { 2 } =$ $v ^ { \top } S _ { h } v$ . Conditioning on the realized training design $X _ { h }$ , we assume the score vector $\begin{array} { r } { z = h ^ { - 1 } \sum _ { j } x _ { t - j } \varepsilon _ { t - j } } \end{array}$ satisfies $\smash { \mathrm { V a r } ( z \mid X _ { h } ) \preceq \kappa _ { h } \sigma _ { t } ^ { 2 } S _ { h } / h }$ . Here $\sigma _ { t } ^ { 2 }$ is the error variance and $\kappa _ { h } \geq 1$ is a dependence factor, the assumption does not require z to have a strict conditional mean of zero. The conditional risk The conditional risk $\mathcal { E } _ { t } ^ { X } ( h ) = \mathrm { E } [ \| \widehat { \beta } - \beta _ { t } \| _ { S _ { h } } ^ { 2 } \ | \ X _ { h } ]$ of of the ridge estimator $\widehat { \beta }$ partitions into four interpretable components:

$$
\begin{array} { r l } & { B _ { \mathrm { r e g } } = \Vert ( I - G _ { h } ) \beta _ { t } \Vert _ { S _ { h } } ^ { 2 } , \quad D ^ { 2 } = \Vert G _ { h } \dot { \beta } _ { t } \Vert _ { S _ { h } } ^ { 2 } , } \\ & { d _ { 2 , h } = \mathrm { t r } [ S _ { h } ^ { 2 } ( S _ { h } + \lambda I ) ^ { - 2 } ] , } \end{array}
$$

and a remainder $\rho ^ { 2 }$ that captures coeficient curvature, time-imbalance in the design, and the conditional mean of z.

Applying an exact bias-variance identity, the risk satisfies the following bound for any $\eta , \zeta > 0$ (Proposition A.4 in Appendix A.3.1):

$$
\begin{array} { r l } & { \mathcal { E } _ { t } ^ { X } ( h ) \leq ( 1 + \eta ) ( 1 + \zeta ) B _ { \mathrm { r e g } } } \\ & { \quad \quad + ( 1 + \eta ) ( 1 + \zeta ^ { - 1 } ) \mu _ { h } ^ { 2 } D ^ { 2 } } \\ & { \quad \quad + ( 1 + \eta ^ { - 1 } ) \rho ^ { 2 } + \kappa _ { h } \sigma _ { t } ^ { 2 } d _ { 2 , h } / h . } \end{array}\tag{6}
$$

Crucially, this bound is direction-aware. The variance decays as $1 / h$ via the spectral dimension $d _ { 2 , h }$ , while the drift penalty grows as $h ^ { 2 }$ via $\mu _ { h } ^ { 2 } D ^ { 2 }$ . Because the temporal drift $\dot { \beta } _ { t }$ enters exclusively through the filter $G _ { h } ,$ , ridge regression naturally attenuates drift in lowvariance predictor directions.

To isolate the exact memory dependence, assume the local second-moment matrix remains constant across candidate histories $( S _ { h } = S )$ , the remainder term vanishes $( \rho = 0 )$ , and the conditional score variance scales inversely with memory as $\mathrm { V a r } ( z ~ \vert ~ X _ { h } ) ~ = ~ \kappa \sigma _ { t } ^ { 2 } S / h$ for a constant $\kappa > 0$ Define the stable ridge filter $G = ( S + \lambda I ) ^ { - 1 } S ,$ the efective spectral dimension $d _ { 2 } = \mathrm { t r } [ S ^ { 2 } ( S + \lambda I ) ^ { - 2 } ]$ , and the inner product $\langle u , v \rangle _ { S } =$ $u ^ { \top } S v$ . The conditional risk then reduces to four fundamental components: the regularization bias $B _ { t } \ =$ $\| ( I - G ) \beta _ { t } \| _ { S } ^ { 2 }$ , the filtered drift energy $Q _ { t } = \| G \dot { \boldsymbol { \beta } } _ { t } \| _ { S } ^ { 2 } ,$ the bias-drift cross-term $J _ { t } = \langle ( I - G ) \beta _ { t } , G \dot { \beta } _ { t } \rangle _ { S }$ , and the variance scale $V _ { t } = \kappa \sigma _ { t } ^ { 2 } d _ { 2 } \colon$

$$
\mathscr { E } _ { t } ( h ) = B _ { t } + 2 \mu _ { h } J _ { t } + \mu _ { h } ^ { 2 } Q _ { t } + V _ { t } / h .\tag{7}
$$

Equation (7) matches the form of Equation (5) with $V _ { k } = V _ { t } , K _ { k } = J _ { t } + Q _ { t } / 2$ , and $W _ { k } = Q _ { t } / 4 \ \mathrm { ( F i g . \ 1 B ) }$

Proposition 3 (optimal memory for ridge regression). Assume strictly positive variance and drift energy $( V _ { t } > 0$ and $Q _ { t } > 0 )$ , and extend Equation (7) to real $h > 0$ . The continuous risk function possesses a unique positive minimizer $h _ { t } ^ { * }$ , which solves the optimality condition:

$$
\begin{array} { r } { V _ { t } / ( h _ { t } ^ { * } ) ^ { 2 } = J _ { t } + \frac { h _ { t } ^ { * } + 1 } { 2 } Q _ { t } . } \end{array}\tag{8}
$$

When the bias-drift cross-term vanishes $( J _ { t } = 0 )$ , the solution satisfies $( h _ { t } ^ { \ast } ) ^ { 2 } ( h _ { t } ^ { \ast } + 1 ) = 2 V _ { t } / Q _ { t } ,$ , converging asymptotically to the cube-root rule $h _ { t } ^ { * } \approx ( 2 V _ { t } / Q _ { t } ) ^ { 1 / 3 }$ for large training windows. Nonzero cross-terms structurally shift the optimal window: a negative $J _ { t }$ lengthens the ideal memory, whereas a positive $J _ { t }$ shortens the ideal memory. (Detailed proofs appear in Appendices A.3.1 and A.3.2).

## 3.4 Surrogate approximations and transfer bounds for trees and neural networks

Because architectures like boosted trees, RNNs, LSTMs, and Transformers are highly nonlinear in the underlying parameters, analyzing the exact memory dynamics requires an intermediate linear surrogate. We construct the theoretical surrogate by freezing the feature map prior to fitting the final readout layer (Propositions A.6 and A.8 in Appendix A.4.1). For tree-based models, the frozen map corresponds to a fixed tree topology; for neural networks, the frozen map represents the encoder producing the final hidden state or the pooled token state. Applying a ridge regression readout to the fixed features ensures the surrogate risk follows the exact complexity expansion in Equation (5), governed by the variance component $V _ { k } / h$ and the quadratic drift component $W _ { k } h ^ { 2 }$

To quantify the performance gap between the theoretical surrogate and the actual model k at memory $h ,$ , let $g _ { k , h }$ denote the surrogate forecast and $f _ { k , h }$ denote the fully trained forecast. Define $\mathcal { L } _ { k } ^ { 0 } ( h )$ and $\mathcal { L } _ { k } ( h )$ as the respective prediction risks against the current conditional mean. Finally, let $\delta _ { k , h } ^ { 2 }$ represent the mean squared distance between the trained forecast $f _ { k , h }$ and the surrogate forecast $g _ { k , h }$ , measured under the evaluation norm.

Proposition 4 (transfer bounds). The prediction risks of the fully trained model and the linear surrogate difer by at most $\epsilon _ { k , h } = 2 \delta _ { k , h } \sqrt { \mathcal { L } _ { k } ^ { 0 } ( h ) } + \delta _ { k , h } ^ { 2 }$ . Consequently, the trained complex class achieves strictly lower risk than the trained simple class if the historyindependent advantage $\Delta _ { 0 }$ exceeds the combined com plexity cost and remainder bounds: $\Delta _ { 0 } ~ > ~ \phi ( h ) ~ + ~ $ $r _ { 0 } ( h ) + r _ { 1 } ( h ) + \epsilon _ { 1 , h } + \epsilon _ { 0 , h }$ . Conversely, the simple class achieves lower risk if $\Delta _ { 0 } < \phi ( h ) - r _ { 0 } ( h ) - r _ { 1 } ( h ) - \epsilon _ { 1 , h } -$ $\epsilon _ { 0 , h }$

Furthermore, assuming a uniform transfer error bound $\epsilon _ { k , h } \ \leq \ \epsilon$ over an interval of candidate histories, and strictly positive surrogate coeficients $V _ { k } , W _ { k } > 0$ , the optimal memory for the trained risk lies within a distance of $\sqrt { 2 \epsilon / W _ { k } }$ from the surrogate optimum. The surrogate optimum solves $2 W _ { k } h ^ { 3 } + K _ { k } h ^ { 2 } = V _ { k }$ , where $K _ { k }$ represents the linear drift coeficient from Equation (5) (Proposition A.11 and Corollary A.16 in Appendix A.4.3).

Because the transfer error ϵ enters additively and scales linearly with the perturbation δ (for small δ), the phase boundary and optimal memory expressions preserve the fundamental scaling dynamics of ridge regression: a $1 / h$ decay for estimation variance and an $\bar { h } ^ { 2 }$ growth for quadratic drift (Fig. 1). The various nonlinear architectures difer only in how the perturbation bound δ scales:

• Boosted trees. For greedy trees built on a finite threshold grid with a strictly positive split margin $\Delta _ { \mathrm { s p l i t } } > 0$ , the transfer error scales as $\epsilon \propto e ^ { - \alpha h }$ where α ∝ $\Delta _ { \mathrm { s p l i t } } ^ { 2 }$ (Proposition A.12). The selection penalty decays exponentially as history length h increases, shifting the optimal memory toward longer windows.

• RNN. Provided the recurrent weights maintain an operator norm $\chi < 1$ , the perturbation δ scales linearly with the movement of the network weights and readout. The sequence scaling factor $( 1 - \chi ^ { L } ) / ( 1 -$ $\chi ) \leq 1 / ( 1 - \chi )$ remains bounded, independent of the context length L (Proposition A.13).

• LSTM. For Long Short-Term Memory networks, the perturbation δ scales linearly with the movement of the gating mechanisms and the readout. The sequence scaling factor expands to $\scriptstyle \sum _ { k < L } \varrho ^ { k }$ , governed by the Lipschitz constant ϱ of the cell state update. The scaling factor remains bounded for $\varrho < 1$ , but grows exponentially with context length L if $\varrho \ge 1$ (Proposition A.14).

• Transformer. For a single attention block, the perturbation δ scales linearly with the movement of the attention logits, value matrices, output map, and readout. The scaling factor relies on $\sqrt { N _ { \mathrm { h e a d } } }$ and exhibits no explicit dependence on the context length L (Proposition A.15).

For exact-greedy trees, the transfer bound does not decay with history length h (Complete conditions and proofs appear in Appendix A.4.3).

## 4 EXPERIMENTS

## 4.1 Controlled simulation

Design. We generate eight Gaussian covariates, $x _ { 1 } , \ldots , x _ { 8 }$ , with variances $q ^ { \stackrel { \smile } { k } - 1 }$ for spectral decay $q \in$ {1, 0.45} (Appendix B.1, covariates). The response combines a linear signal, a nonlinear interaction (γ times the standardized product $x _ { 1 } x _ { 2 } )$ , and Gaussian noise, with interaction strength $\gamma \in \{ 0 , 0 . 4 , 0 . 8 \}$ and $\mathrm { S N R } \in \{ 0 . 2 5 , 2 \}$ (Appendix B.1, signal). To simulate nonstationarity, the first coeficient of an observation aged a periods shifts by −0.008a (gradual drift), −0.65 for $a > 4 8$ (abrupt break), or 0 (zero drift); test targets use unshifted current coeficients (Appendix B.1, drift). We compare linear ridge on the 8 base covariates against quadratic ridge on 44 expanded features (including $x _ { 1 } x _ { 2 } )$ , testing penalties $\lambda \in \{ 0 . 0 3 , 0 . 5 \}$ and memories $h ~ \in ~ \{ 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ (Appendix B.1, models). Each factorial condition uses 24 independent worlds, each containing 16 test targets and three training histories (Appendix B.1, replicates). We score out-of-sample $R ^ { 2 }$ against the zero forecast, analyzing the performance gap $\Delta R ^ { 2 } = R _ { \mathrm { q u a d } } ^ { 2 } - R _ { \mathrm { l i n } } ^ { 2 }$ with paired 95% intervals across worlds (Appendix B.1, score and contrasts).

Results. Across 360 comparisons, the quadratic model outperforms the linear baseline in 0 of 120 cases at $\gamma = 0$ , 46 at $\gamma = 0 . 4$ , and 78 at $\gamma = 0 . 8 \ ( \mathrm { A p } \cdot$ pendix B.2, Figs. 7, 8, and 9). Under gradual drift $( \gamma = 0 . 8 )$ 2 $\Delta R ^ { 2 }$ is positive for all memories at SNR 2; at SNR 0.25, it remains negative for $h \leq 6 4$ and turns positive only at $h \geq 1 2 8 .$ , corroborating the phase boundary of Section 3.2 (Fig. 2; Appendix B.2, Fig. 8). In a balanced linear design, optimal memory falls from 128 to 32 as drift rises from 0.002 to 0.016; the −0.67 log-log slope tightly matches the predicted $- 2 / 3$ from Section 3.3 (Appendix B.2, Fig. 4). At equal raw drift, linear ridge (λ = 0.03) $R ^ { 2 }$ drops from 0.57 (h = 16) to $- 2 . 3 7 \ ( h = 2 5 6 )$ if the drifting coordinate variance is 1, but rises from 0.60 to 0.67 if the variance is 0.004, confirming ridge attenuates drift in low-variance directions (Appendix B.2, Fig. 5).

![](images/e364840a0002b728c628e94d5f7c4b45042916ab46a6b21f589122a228797c56.jpg)  
Figure 2: Out-of-sample performance gap $\Delta R ^ { 2 }$ (quadratic minus linear) across training memories under gradual drift $( q = 0 . 4 5 , \lambda = 0 . 0 3 )$ . Error bars denote paired 95% intervals.

## 4.2 Empirical dynamics of forecast components

Design. Following established protocols (Capponi et al., 2025), we forecast daily excess returns for 17 U.S. industry portfolios (Fama and French, 1997). The dataset spans September 1987 to November 2016, ded icating January 1990 onwards strictly to out-of-sample evaluation. The daily information set comprises 129 predictors partitioned into four structural categories: (1) 94 feature-based portfolios capturing firm-level attributes such as price trends, liquidity, and volatility (Gu et al., 2020); (2) three standard market factors (Fama and French, 1993); (3) 15 monthly structural series, including macroeconomic hidden states extracted from 178 underlying economic indicators (Chen et al., 2024); and (4) the previous day’s 17 target returns to capture autoregressive dynamics.

Six tabular models, fixed- and tuned-penalty ridge, random forest, XGBoost, small MLP, and medium MLP, process single-day predictors. Seven temporal models ingest the trailing $L \in \{ 2 0 , 6 0 \}$ days: fixed- and tuned-penalty ridge, random forest, XGBoost, MLPs, an RNN, an LSTM, and a Transformer (Appendix C.1, Model specifications). Each model trains on a rolling 16-month, rolling 64-month, or expanding historical window, with hyperparameters tuned on immediately preceding data (Appendix C.1, Hyperparameter selection). We evaluate six random-seed replicates per configuration, scoring out-of-sample $R ^ { 2 }$ against the zero forecast and decomposing performance gaps via identity (2) (Appendix C.1, Replicate statistics).

![](images/f34770316ec8f68f73a6b92574b6f83409d0e04d5bde8fc2ef827a3eeabdf971.jpg)

![](images/f402195dfb7710e5c5937a470036dcade0335ad428b4500d9a0c6a84b34d6e1a.jpg)

![](images/944073e0cb16ac00a569be90f5b679b8375d0b1581fd9b025a3d39a6b76be464.jpg)

D  
![](images/7ea6f8a9af2fe967a12fcd7d8197776c76c67359da958040d61dcc6056c4963c.jpg)  
Figure 3: Empirical dynamics of forecast components. Model classes are color-coded: tabular (teal), 20- day temporal (blue), and 60-day temporal (brown). (A) Restricted training memory systematically exacerbates predictive instability u. (B) Stronger ridge regularization efectively attenuates the aggressiveness penalty g. (C) Ex-post scalar calibration preserves target alignment, failing to resolve fundamental performance gaps between models. (D) Elevated ambient noise systematically diminishes the out-of-sample advantage of short training memory.

Results. Evaluated by mean out-of-sample $R ^ { 2 }$ , random forest performance degrades from 0.3549 under a 16-month memory to 0.2953 under an expanding history, whereas tuned ridge regression improves from 0.3515 to 0.3602 (Table 1). The complex tree-based model performs better at short memories where its static approximation advantage dominates, but the simpler ridge model performs better as history expands and variance-drift penalties shift, consistent with results in Section 3.2. A parallel reversal emerges among the 60-day temporal architectures: flattened XGBoost achieves the highest mean $R ^ { 2 }$ at 16 and 64 months (0.2812 and 0.3223), but the small Transformer strictly overtakes it under expanding history (0.3195). Finally, quantifying the empirical magnitude of predictive instability, replicate ensembling recovers substantial performance costs, boosting the 16-month random forest $R ^ { 2 }$ to 0.3705 and the expanding-history Transformer to 0.3342 (Table 1; full results in $\mathrm { A p \mathrm { - } }$ pendix Table 8).

As predicted by the variance term $V / h$ (Eq. (7)), restricted memory exacerbates instability: u is strictly larger at 16 than 64 months in 18 of 20 architecture– context pairs (Fig. 3A; Appendix Table 8). Ridge systematically attenuates forecast magnitude and variance (Section 3.3). Compared to a penalty of 1, tuned ridge (penalty 100 or 10,000) lowers aggressiveness g in 78 of 81 year-window pairs and instability u in all 81 (Fig. 3B). Because scalar transformations preserve target alignment (Section 3.1), calibration cannot resolve fundamental disparities. A historically fitted [0, 1] scalar shifts the 20-day XGBoost–Transformer gap by at most 0.005 (Fig. 3C).

Optimal memory expands with noise and contracts with filtered drift (Eq. (8)). Empirically, elevated noise (recent ridge-fit squared error) lowers the 16-month advantage by 0.012 $R ^ { 2 }$ per standard deviation (95% interval $[ - 0 . 1 0 9 , - 0 . 0 0 1 ]$ ; Fig. 3D).

## 4.3 Generalization to independent data

Design. To evaluate firm-level prediction, we extend our framework to daily CRSP equity data spanning 2018–2024, forecasting next-day total returns using the stock universe and 24 feature representations established by (Zhang et al., 2026). We refit all models monthly from January 2021 through December 2024 across three training memories: a rolling 252-day window, a rolling 756-day window, and an expanding historical window. Three tabular models (ridge, XG-Boost, and an MLP) process single-day predictors, while six temporal architectures ingest historical contexts of the trailing 20 or 60 days. Hyperparameters remain strictly fixed across evaluations, yielding two paired-seed replicates per architecture–context pair.

Results. On the independent CRSP stock-return panel, XGBoost strictly dominates the fixed architectures across all nine context–memory combinations (Table 2). In a separate 2022–2024 evaluation, applying an ex-post scalar calibration (fitted over the preceding 12 months) collapses nine XGBoost–neural performance gaps from the 0.20–1.03 range to under 0.00015. Because this convergence is achieved almost entirely by attenuating forecast magnitudes toward zero, it confirms that the initial performance disparities stemmed from miscalibrated aggressiveness rather than fundamental diferences in target alignment.

## 5 CONCLUSION

This paper investigates the fundamental mechanisms governing the heterogeneous sensitivity of predictive architectures to training memory under distribution shift. To this end, we derive a model-agnostic decomposition of out-of-sample $R ^ { 2 }$ that partitions predictive performance into target alignment, an aggressiveness penalty, and finite-sample predictive instability. Building upon this identity, we formally characterize the complexity phase boundary—delineating the memory regimes where high-capacity models overcome simpler baselines—and establish a closed-form optimal memory equation for ridge regression under local drift. Controlled simulations rigorously validate both the predicted phase boundaries and the derived −2/3 asymptotic scaling of optimal memory with respect to drift velocity. Crucially, in empirical financial forecasting, our framework mechanically explains rank reversals across lookback windows. The identity isolates these inversions to shifting tensions between alignment and aggressiveness, while confirming that replicate ensembling cleanly recovers the precise theoretical penalty of predictive instability. While prior lit erature has traditionally treated the estimation–drift tradeof, model benchmarking, and adaptive memory selection as disparate domains, our primary contribution is structural: by unifying these fields through a single deterministic identity, we trace any macroscopic shift in model ranking directly to a quantified, mechanistic forecast property.

Table 1: Memory-induced rank reversals and the empirical recovery of predictive instability. Expanding the training horizon (16m to Exp.) systematically inverts model rankings, favoring heavily regularized or temporally structured architectures over static, complex baselines at longer histories. Additionally, replicate ensembling strictly improves upon mean predictive performance, quantifying the exact out-of-sample cost of finite-sample instability. Shading identifies the optimal architecture within the tabular and 60-day temporal branches; RMSE is reported in percentage points of daily excess returns.
<table><tr><td rowspan="2"></td><td colspan="6">Tabular</td><td colspan="7">Temporal, 60-day context</td></tr><tr><td>Ridge, α=1</td><td>ridge</td><td>forest</td><td>Tuned Random XGBoost</td><td>Small MLP</td><td>MLP</td><td>ridge</td><td>Medium Flattened Flattened XGBoost</td><td>Fixed flattened flattened MLP</td><td>Tuned MLP</td><td>Vanilla LSTM RNN</td><td></td><td>Small Transformer</td></tr><tr><td colspan="10">Mean  $R ^ { 2 } \uparrow$ </td><td></td><td></td><td></td><td></td></tr><tr><td>16m</td><td>-0.0909 0.3515</td><td></td><td>0.3549</td><td>0.3316</td><td>0.2869</td><td>0.2786</td><td>0.0772</td><td>0.2812</td><td>-0.8216</td><td>-0.2223</td><td>0.2776 0.1913</td><td></td><td>0.2712</td></tr><tr><td>64m</td><td>0.3256</td><td>0.3668</td><td>0.3374</td><td>0.3419</td><td>0.3231</td><td>0.3189</td><td>0.1078</td><td>0.3223</td><td>-0.0124</td><td>0.0002</td><td>0.2989</td><td>0.2472</td><td>0.2969</td></tr><tr><td>Exp.</td><td>0.3386</td><td>0.3602</td><td>0.2953</td><td>0.3232</td><td>0.3301</td><td>0.3270</td><td>-0.2835</td><td>0.3148</td><td>0.0881</td><td>0.1158</td><td>0.3169</td><td>0.2885</td><td>0.3195</td></tr><tr><td colspan="10">Ensemble R2 ↑</td><td colspan="5"></td></tr><tr><td>16m</td><td></td><td>0.0706 0.3671</td><td>0.3705</td><td>0.3621</td><td>0.3096</td><td>0.3018</td><td>0.1092</td><td>0.3161</td><td>-0.3891</td><td>-0.0970</td><td>0.3123</td><td>0.2153</td><td>0.3191</td></tr><tr><td>64m</td><td>0.3481</td><td>0.3767 0.3644</td><td>0.3494</td><td>0.3619</td><td>0.3414 0.3423</td><td>0.3383 0.3425</td><td>0.1654 -0.0295</td><td>0.3402 0.3253</td><td>0.0765</td><td>0.0804</td><td>0.3167</td><td>0.2661</td><td>0.3160 0.3342</td></tr><tr><td colspan="10">Exp. 0.3477</td><td colspan="3">0.1690 0.1822</td><td colspan="3">0.3276 0.3042</td></tr><tr><td>Mean</td><td>RMSE</td><td>(pp) ↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16m</td><td>1.4675</td><td>1.1315</td><td>1.1286</td><td>1.1487</td><td>1.1865</td><td>1.1934</td><td>1.3498</td><td>1.1913</td><td>1.8962</td><td>1.5534</td><td>1.1942 1.2636</td><td></td><td>1.1995</td></tr><tr><td>64m Exp.</td><td>1.1539 1.1427</td><td>1.1181 1.1239</td><td>1.1438 1.1795</td><td>1.1399 1.1559</td><td>1.1560 1.1500</td><td>1.1596 1.1527</td><td>1.3272 1.5918</td><td>1.1567 1.1631</td><td>1.4137 1.3418</td><td>1.4049 1.3212</td><td>1.1765 1.1613 1.1851</td><td>1.2191</td><td>1.1782 1.1591</td></tr><tr><td colspan="10">Ensemble RMSE (pp) ↓</td><td colspan="5"></td></tr><tr><td></td><td></td><td>1.1178</td><td>1.1148</td><td>1.1222</td><td>1.1675</td><td>1.1741</td><td>1.3262</td><td>1.1620</td><td>1.6560</td><td>1.4716</td><td>1.1652 1.2446</td><td></td><td>1.1594</td></tr><tr><td>16m 64m</td><td>1.3546 1.1345</td><td>1.1093</td><td>1.1334</td><td>1.1224</td><td>1.1403</td><td>1.1430</td><td>1.2836</td><td>1.1413</td><td>1.3503</td><td>1.3474</td><td>1.1615 1.2037</td><td></td><td>1.1621</td></tr><tr><td>Exp.</td><td>1.1348</td><td>1.1202</td><td>1.1690</td><td>1.1422</td><td>1.1395</td><td>1.1394</td><td>1.4256</td><td>1.1541</td><td>1.2808</td><td>1.2707</td><td>1.1522 1.1720</td><td></td><td>1.1465</td></tr></table>

<table><tr><td>Context</td><td>252d</td><td>756d</td><td>Exp.</td></tr><tr><td>Tabular</td><td>-0.00189</td><td>-0.00145</td><td>-0.00129</td></tr><tr><td>Temporal 20d</td><td>-0.00198</td><td>-0.00132</td><td>-0.00076</td></tr><tr><td>Temporal 60d</td><td>-0.00199</td><td>-0.00135</td><td>-0.00096</td></tr></table>

Table 2: Mean pooled out-of-sample $R ^ { 2 }$ against a zeroreturn baseline (January 2021–December 2024).

Computational and economic implications. Aggressively restricting the training window is an inadequate heuristic for nonstationarity; when ambient noise dominates filtered drift, heavily regularized linear models with expanding histories can match high capacity learners at a fraction of the cost. Our decomposition therefore serves as a practical diagnostic, allowing practitioners to evaluate the three forecast components before funding expensive architecture searches. Conversely, while replicate ensembling eliminates the instability penalty, it scales training costs linearly. Ultimately, the framework quantifies the exact marginal predictive return of each additional unit of compute.

Limitations and future work. While our closedform derivations apply exactly to ridge regression under local drift, theoretical guarantees for high-capacity architectures currently rely on surrogate transfer bounds. Additionally, because memory is parameterized strictly via uniform truncation windows, the framework does not address exponentially weighted or adaptive histories. Future research could generalize the theory to accommodate dynamic forgetting factors, leveraging the proposed three-component decomposition for online, adaptive memory selection. Further natural extensions include tightening the nonlinear transfer bounds, integrating compute-aware evaluation metrics, and validating these complexity phase boundaries across diverse forecasting horizons and alternative asset classes.

## AI USE STATEMENT

During the preparation of this work, the authors used OpenAI ChatGPT-6 Sol, Claude Sonnet 5.5, and Google Gemini-3.1 Pro to conduct literature searches, generate baseline code configurations, develop proofs and derivations, create visualizations, generate initial drafts of writing, improve grammatical clarity, check for technical consistency, and verify the correctness of mathematical derivations. Following the use of these tools, the authors thoroughly reviewed, edited, and verified all generated output. The authors take full responsibility for the technical correctness, integrity, and originality of the final manuscript content.

## REFERENCES

Baby, D., Tang, Y., Nguyen, H. D., Wang, Y.-X., and Pyati, R. (2025). Adaptive estimation and learning under temporal distribution shift. arXiv:2505.15803.

Baby, D. and Wang, Y.-X. (2023). Second order path variationals in non-stationary online learning. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 9024–9075.

Bejarano, J., Desai, V., Keshava, K., Kumar, A., Wang, Z., Xu, V. H., and Xu, Y. (2025). An open benchmark for evaluating time series forecasting methods across financial markets. Working paper.

Bousquet, O. and Elisseef, A. (2002). Stability and generalization. Journal of Machine Learning Research, 2:499–526.

Campbell, J. Y. and Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? Review of Financial Studies, 21(4):1509–1531.

Caponnetto, A. and De Vito, E. (2007). Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7:331–368.

Capponi, A., Huang, C., Sidaoui, J. A., Wang, K., and Zou, J. (2025). The nonstationarity-complexity tradeof in return prediction. arXiv:2512.23596.

Chen, B. and Maung, K. (2020). Time-varying forecast combination for high-dimensional data. arXiv:2010.10435.

Chen, L., Pelger, M., and Zhu, J. (2024). Deep learning in asset pricing. Management Science, 70:714– 750.

Chen, M., Li, X., and Zhao, T. (2020). On generalization bounds of a family of recurrent neural networks. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics,

volume 108 of Proceedings of Machine Learning Research, pages 1233–1243.

Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794.

Cheung, W. C., Simchi-Levi, D., and Zhu, R. (2019). Learning to optimize under non-stationarity. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics.

Cortes, C., Mohri, M., and Storcheus, D. (2019). Regularized gradient boosting. In Advances in Neural Information Processing Systems, volume 32.

Dahlhaus, R. (1997). Fitting time series models to nonstationary processes. Annals of Statistics, 25(1):1–37.

Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3):253–263.

Diebold, F. X. and Pauly, P. (1987). Structural change and the combination of forecasts. Journal of Forecasting, 6(1):21–40.

Dobriban, E. and Wager, S. (2018). High-dimensional asymptotics of prediction: Ridge regression and classification. Annals of Statistics, 46(1):247–279.

Elliott, G. and Timmermann, A. (2004). Optimal forecast combinations under general loss functions and forecast error distributions. Journal of Econometrics, 122(1):47–79.

Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1):3–56.

Fama, E. F. and French, K. R. (1997). Industry costs of equity. Journal of Financial Economics, 43(2):153– 193.

Giacomini, R. and Rossi, B. (2015). Forecasting in nonstationary environments: What works and what doesn’t in reduced-form and structural models. Annual Review of Economics, 7:207–229.

Giacomini, R. and Rossi, B. (2016). Model comparisons in unstable environments. International Economic Review, 57(2):369–392.

Giacomini, R. and White, H. (2006). Tests of conditional predictive ability. Econometrica, 74(6):1545– 1578.

Giraitis, L., Kapetanios, G., and Price, S. (2013). Adaptive forecasting in the presence of recent and ongoing structural change. Journal of Econometrics, 177(2):153–170.

Gneiting, T. and Resin, J. (2023). Regression diagnos tics meets forecast evaluation: Conditional calibration, reliability diagrams, and coeficient of determination. Electronic Journal of Statistics, 17(2):3226– 3286.

Goktas, D., Ria˜no-Brice˜no, G., Abdullah, A., et al. (2026). TempusBench: An evaluation framework for time-series forecasting. arXiv:2604.11529.

Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5):2223–2273.

Han, E., Huang, C., and Wang, K. (2024). Model assessment and selection under temporal distribution shift. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17374–17392.

Hansen, P. R. (2005). A test for superior predictive ability. Journal of Business & Economic Statistics, 23(4):365–380.

Hansen, P. R., Lunde, A., and Nason, J. M. (2011). The model confidence set. Econometrica, 79(2):453– 497.

Hewamalage, H., Montero-Manso, P., Bergmeir, C., and Hyndman, R. J. (2021). A look at the evaluation setup of the M5 forecasting competition. arXiv:2108.03588.

Hsu, D., Kakade, S. M., and Zhang, T. (2012). Random design analysis of ridge regression. In Proceedings of the 25th Annual Conference on Learning Theory, volume 23 of Proceedings of Machine Learning Research, pages 9.1–9.24.

Huang, C. and Wang, K. (2023). A stability principle for learning under non-stationarity. arXiv:2310.18304.

Mincer, J. A. and Zarnowitz, V. (1969). The evaluation of economic forecasts. In Economic Forecasts and Expectations: Analysis of Forecasting Behavior and Performance. National Bureau of Economic Research.

Pesaran, M. H. and Timmermann, A. (2007). Selection of estimation window in the presence of breaks. Journal of Econometrics, 137(1):134–161.

Saly-Kaufmann, A., Wood, K., Peter-Calliess, J., and Zohren, S. (2026). Deep learning for financial time series: A large-scale benchmark of risk-adjusted performance. arXiv:2603.01820.

Smith, S. C. and Timmermann, A. (2021). Break risk. Review of Financial Studies, 34(4):2045–2100.

Stambaugh, R. F. (1999). Predictive regressions. Journal of Financial Economics, 54(3):375–421.

Theil, H. (1966). Applied Economic Forecasting. North-Holland.

Trauger, J. and Tewari, A. (2024). Sequence length independent norm-based generalization bounds for transformers. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 1405–1413.

Vogt, M. (2012). Nonparametric regression for locally stationary time series. Annals of Statistics, 40(5):2601–2633.

Waghmare, K. and Ziegel, J. (2026). Proper scoring rules for estimation and forecast evaluation. Annual Review of Statistics and Its Application, 13:271–296.

Welch, I. and Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. Review of Financial Studies, 21(4):1455– 1508.

White, H. (2000). A reality check for data snooping. Econometrica, 68(5):1097–1126.

Yang, T., Zhang, L., Jin, R., and Yi, J. (2016). Tracking slowly moving clairvoyant: Optimal dynamic regret of online learning with true and noisy gradient. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 449–457.

Yuan, J. and Lamperski, A. (2020). Trading-of static and dynamic regret in online least-squares and beyond. Proceedings of the AAAI Conference on Artificial Intelligence, 34(4):6712–6719.

Zhang, A., Cheng, Y., and Leung, K. (2026). Benchmarking deep time series models for equity portfolios. arXiv:2606.09420.

Zhao, P. and Zhang, L. (2021). Improved analysis for dynamic regret of strongly convex and smooth functions. In Proceedings of the 3rd Conference on Learning for Dynamics and Control, volume 144 of Proceedings of Machine Learning Research, pages 48–59.

Zhao, P., Zhang, L., Jiang, Y., and Zhou, Z.-H. (2020). A simple approach for non-stationary linear bandits. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 746–755.

# Decomposing Rank Reversals under Temporal Nonstationarity: Alignment, Aggressiveness, and Instability

## A THEORETICAL RESULTS AND PROOFS

This appendix gives the derivations behind Section 3. It has four subsections. Each subsection expands the subsection of Section 3 with the same number, in the same order and with the same logic. We restate the claims of Section 3, add the definitions and assumptions that it omits, and prove every claim. We also include further results that Section 3 does not state.

In Section A.1, we expand Section 3.1. We prove Proposition 1 of Section 3.1 in Proposition A.1, together with the constrained scaling rules that Section 3.1 omits. These results hold in every finite sample. They need no stationarity assumption and no linear-model assumption

In Section A.2, we expand Section 3.2. We prove Proposition 2 of Section 3.2 in Propositions A.2 and A.3. The phase boundary needs the risk expansion (5).

In Section A.3, we expand Section 3.3. We bound the risk of ridge regression under coeficient drift and serial dependence, which proves (6) (Proposition A.4). We then solve for the memory that minimizes an exactly solvable version of this risk, which proves Proposition 3 of Section 3.3 (Corollary A.5). Each result states the conditions that it needs.

In Section A.4, we expand Section 3.4. We extend the ridge bound to the surrogate, that is, to frozen feature maps (Proposition A.6) and to boosted trees with fixed splits (Proposition A.8). We derive an exact coordinatewise phase boundary for the surrogate (Proposition A.9). We then bound the gap between the surrogate and the trained model. This proves Proposition 4 of Section 3.4 (Proposition A.11 and Corollary A.16), and it gives the conditions for boosted trees, RNNs, LSTMs, and Transformers (Propositions A.12–A.15).

Throughout, ∥ · ∥ and $\langle \cdot , \cdot \rangle$ are the Euclidean norm and inner product. For symmetric matrices, $A \preceq B$ means that $B - A$ is positive semidefinite. We write E and Var for expectation and variance, tr for the trace, and R for the real line. The index t is the forecast date, and the memory h is the number of most recent observations that train a model. We number the results of this appendix A.1, A.2, and so on. We cite the results of Section 3 by their number in that section.

## A.1 Forecast-score decomposition, ensembling, and scalar calibration

Proposition A.1 (optimal common scalar). Assume $q > 0$ . Then $\begin{array} { r } { S ( \gamma ) = 2 \gamma c - \gamma ^ { 2 } q } \end{array}$ . Let $\mathcal { T } \subseteq \mathrm { R }$ be any nonempty closed interval, bounded or unbounded. On I, the score has the unique maximizer

$$
\sqrt { \gamma _ { \mathcal { T } } ^ { * } = \Pi _ { \mathcal { T } } \bigg ( \frac { c } { q } \bigg ) \ , }\tag{9}
$$

where Π<sub>I</sub> is the projection onto the interval. The interval sets the calibration rule. Signed scaling uses ${ \mathcal { T } } = \mathrm { R }$ and gives $c / q$ . Sign-preserving scaling uses $\mathcal { I } = [ 0 , \infty )$ and gives max $\{ 0 , c / q \}$ . Shrinkage-only scaling uses $\mathcal { T } = [ 0 , 1 ]$ and gives min $\{ 1 , \operatorname* { m a x } \{ 0 , c / q \} \}$ . For unconstrained scaling, the maximum score and the gain over $\gamma = 1$ are

$$
\operatorname* { m a x } _ { \gamma } \mathcal { S } ( \gamma ) = \frac { c ^ { 2 } } { q } , \qquad \mathcal { S } ( \gamma ^ { * } ) - \mathcal { S } ( 1 ) = \frac { ( q - c ) ^ { 2 } } { q } .\tag{10}
$$

If $q = 0$ , every forecast is zero. Then the score is zero for every $\gamma ,$ and the optimizer is not identified.

Proof. Expanding the squared error gives $\begin{array} { r } { S ( \gamma ) = 2 \gamma c - \gamma ^ { 2 } q } \end{array}$ . For $q > 0$ , completing the square gives $S ( \gamma ) =$ $c ^ { 2 } / q - q ( \gamma - c / q ) ^ { 2 }$ . This is a concave parabola with vertex $c / q$ . Hence its maximizer on a closed interval is the projection of the vertex onto that interval. $\mathrm { A t } \ \gamma = 1$ the score is $2 c - q .$ . Subtracting it from $c ^ { 2 } / q$ gives (10). We now apply this result to three cases. First, consider a single nonzero replicate $\mathbf { p } _ { m }$ . Define $\begin{array} { r l } { a _ { m } } & { { } = } \end{array}$ $\langle \mathbf { y } , \mathbf { p } _ { m } \rangle / ( \| \mathbf { y } \| \| \mathbf { p } _ { m } \| )$ and $g _ { m } = \| \mathbf { p } _ { m } \| / \| \mathbf { y } \|$ . Then $c = a _ { m } g _ { m }$ and $q \ = \ g _ { m } ^ { 2 }$ , so the unconstrained optimum is $a _ { m } / g _ { m }$ . It follows that the optimal scale shrinks the forecast if $g _ { m } > a _ { m }$ and expands it if $g _ { m } < a _ { m }$ . Second, consider the average replicate score under one common scale. Here $c = a g$ and $q = g ^ { 2 } + u$ , which gives

$$
\gamma _ { \mathrm { r e p } } ^ { * } = \frac { a g } { g ^ { 2 } + u } , \qquad \operatorname * { m a x } _ { \gamma } \overline { { { R ^ { 2 } } } } ( \gamma ) = \frac { a ^ { 2 } g ^ { 2 } } { g ^ { 2 } + u } .\tag{11}
$$

Third, consider the ensemble mean with $g > 0$ . Here $c = a g$ and $q = g ^ { 2 }$ . Hence $\gamma _ { \mathrm { e n s } } ^ { \ast } = a / g$ , and the maximum score is $a ^ { 2 }$ . The gap between the two maxima is

$$
{ \frac { a ^ { 2 } u } { g ^ { 2 } + u } } .\tag{12}
$$

This gap is the loss that instability causes even after optimal rescaling.

The optima above are in-sample oracle diagnostics, because c and q use the evaluation outcomes. A deployable scaling must use only information that is available before each forecast date.

## A.2 Rank reversal and the complexity phase boundary

We now give the details of the phase boundary in Section 3.2. Let $\mathcal { M } _ { 0 }$ be a simple model class and $\mathcal { M } _ { 1 }$ a more complex class. Let $\mathcal { F } _ { k }$ be the set of predictors that class $k \in \{ 0 , 1 \}$ can fit. We call the classes nested if ${ \mathcal { F } } _ { 0 } \subseteq { \mathcal { F } } _ { 1 }$ Let $\mathcal { L } _ { k } ( h )$ be the excess loss of class k at memory h. This is the expected squared error of the fitted forecast against the current conditional mean, so the irreducible noise drops out. We assume the leading expansions

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { k } ( h ) = \mathcal { A } _ { k } + B _ { k } + \frac { V _ { k } } { h } + K _ { k } h + W _ { k } h ^ { 2 } + e _ { k } ( h ) , } \\ { \displaystyle | e _ { k } ( h ) | \le r _ { k } ( h ) , } \end{array}\tag{13}
$$

for $k \in \{ 0 , 1 \}$ . These are the expansions (5) of Section 3.2. The terms have the following meaning. The oracle approximation error $\mathcal { A } _ { k }$ is the smallest error that a predictor in $\mathcal { F } _ { k }$ can attain against the current conditional mean. The term $B _ { k }$ is a bias that does not depend on the history. The term $V _ { k }$ scales with the noise variance and the efective dimension. The terms $K _ { k }$ and $W _ { k }$ are the coeficients of the linear and quadratic drift terms after filtering. Regularization and model type can change both $K _ { k }$ and $W _ { k }$ , and $K _ { k }$ may be negative. The remainder $e _ { k } ( h )$ has the known bound $r _ { k } ( h )$ . We show in Section A.3.2 that the ridge risk has this form with explicit coeficients. In the nested case, ${ \mathcal { F } } _ { 0 } \subseteq { \mathcal { F } } _ { 1 }$ implies $\mathcal { A } _ { 1 } \leq \mathcal { A } _ { 0 }$ . It does not order the other terms. Set

$$
\begin{array} { r } { \Delta _ { 0 } = ( \mathcal { A } _ { 0 } + B _ { 0 } ) - ( \mathcal { A } _ { 1 } + B _ { 1 } ) , \qquad } \\ { \Delta V = V _ { 1 } - V _ { 0 } , \quad \Delta K = K _ { 1 } - K _ { 0 } , } \\ { \Delta W = W _ { 1 } - W _ { 0 } . \qquad } \end{array}\tag{14}
$$

The complex class gains $\Delta _ { 0 }$ in the history-independent loss. It pays $\Delta V / h + \Delta K h + \Delta W h ^ { 2 }$ in the historydependent terms.

We prove Proposition 2 of Section 3.2 in two parts. We first state the conditions under which each class wins (Proposition A.2). The remainder bounds in these conditions form the band of the illustration (Fig. 1A). We then analyze the leading approximation (Proposition A.3).

Proposition A.2 (complexity phase boundary). The complex class has strictly smaller loss whenever

$$
\boxed { \Delta _ { 0 } > \frac { \Delta V } { h } + \Delta K h + \Delta W h ^ { 2 } + r _ { 1 } ( h ) + r _ { 0 } ( h ) . }\tag{15}
$$

The simple class has strictly smaller loss whenever

$$
\Delta _ { 0 } < \frac { \Delta V } { h } + \Delta K h + \Delta W h ^ { 2 } - r _ { 1 } ( h ) - r _ { 0 } ( h ) .\tag{16}
$$

Between these two inequalities, the leading expansion alone does not determine the ranking.

Proof. We subtract (13) for k = 0 from (13) for $k = 1$

$$
\mathcal { L } _ { 1 } - \mathcal { L } _ { 0 } = - \Delta _ { 0 } + \frac { \Delta V } { h } + \Delta K h + \Delta W h ^ { 2 } + e _ { 1 } - e _ { 0 } .
$$

Since $| e _ { 1 } - e _ { 0 } | \le r _ { 1 } + r _ { 0 }$ , the diference is negative under (15) and positive under (16).

Proposition A.3 (leading-approximation phase boundary). We now ignore the remainders. Suppose $\Delta V > 0$ and $\Delta W > 0$ . The complexity cost $\phi ( h ) = \Delta V / h + \Delta K h + \Delta W h ^ { 2 }$ is then strictly convex in $h > 0$ . Its unique minimizer $h _ { \mathrm { c o s t } }$ and its minimum $\phi _ { \mathrm { m i n } }$ satisfy

$$
\begin{array} { c } { 2 \Delta W h _ { \mathrm { c o s t } } ^ { 3 } + \Delta K h _ { \mathrm { c o s t } } ^ { 2 } - \Delta V = 0 , } \\ { \phi _ { \mathrm { m i n } } = \phi ( h _ { \mathrm { c o s t } } ) . } \end{array}\tag{17}
$$

If $\Delta K = 0$ , these reduce to $h _ { \mathrm { c o s t } } = ( \Delta V / 2 \Delta W ) ^ { 1 / 3 }$ and $\phi _ { \mathrm { m i n } } ~ = ~ 3 \cdot 2 ^ { - 2 / 3 } ( \Delta V ) ^ { 2 / 3 } ( \Delta W ) ^ { 1 / 3 }$ . In this leading approximation, the complex class wins at memory h if and only if $\Delta _ { 0 } > \phi ( h ) \ ( \mathrm { F i g . \ 1 A } )$ . The phase boundaries are the positive roots of

$$
\Delta W h ^ { 3 } + \Delta K h ^ { 2 } - \Delta _ { 0 } h + \Delta V = 0 .\tag{18}
$$

Three cases follow. If $\Delta _ { 0 } < \phi _ { \mathrm { m i n } }$ , the complex class never wins. If $\Delta _ { 0 } = \phi _ { \mathrm { m i n } }$ , the two curves touch at $h _ { \mathrm { c o s t } }$ . If $\Delta _ { 0 } > \phi _ { \mathrm { m i n } }$ , the complex class wins on an interval between the two positive roots of (18). Only the part of this interval with admissible $h \geq 1$ is relevant.

Proof. The second derivative is $\phi ^ { \prime \prime } ( h ) = 2 \Delta V / h ^ { 3 } + 2 \Delta W > 0$ , so $\phi$ is strictly convex on $h > 0$ . Hence the first derivative $\phi ^ { \prime } ( h ) = - \Delta V / h ^ { 2 } + \Delta K + 2 \Delta W h$ is strictly increasing. It tends $\mathrm { t o \ - \infty }$ as $h \to 0 ^ { + }$ and to +∞ as $h  \infty$ , so it has exactly one zero $h _ { \mathrm { c o s t } }$ . Multiplying $\phi ^ { \prime } ( h _ { \mathrm { c o s t } } ) = 0$ by $h _ { \mathrm { c o s t } } ^ { 2 }$ gives the cubic in (17). Next, the complex class wins at memory h if and only if $\Delta _ { 0 } > \phi ( h )$ Multiplying this inequality by $h > 0$ gives $\Delta W h ^ { 3 } + \Delta K h ^ { 2 } - \Delta _ { 0 } h + \Delta V < 0$ . The left side equals $h ( \phi ( h ) - \Delta _ { 0 } )$ , so its zeros solve $\phi ( h ) = \Delta _ { 0 }$ , which is (18). Since ϕ is strictly convex and $\phi ( h )  \infty$ as $h \to 0 ^ { + }$ and as $h  \infty$ , the equation $\phi ( h ) = \Delta _ { 0 }$ has no solution if $\Delta _ { 0 } < \phi _ { \mathrm { m i n } }$ . It has the single solution $h _ { \mathrm { c o s t } }$ if $\Delta _ { 0 } = \phi _ { \mathrm { m i n } }$ , and two positive solutions if $\Delta _ { 0 } > \phi _ { \mathrm { m i n } }$ . In the last case, $\phi ( h ) < \Delta _ { 0 }$ holds exactly between the two solutions. This gives the three cases. Finally, let $\Delta K = 0$ Then $h _ { \mathrm { c o s t } } ^ { 3 } = \Delta V / ( 2 \Delta W )$ , so $\Delta V / h _ { \mathrm { c o s t } } = 2 \Delta W h _ { \mathrm { c o s t } } ^ { 2 }$ . Hence $\phi _ { \mathrm { m i n } } = 3 \Delta W h _ { \mathrm { c o s t } } ^ { 2 }$ , and $h _ { \mathrm { c o s t } } ^ { 2 } = ( \Delta V / 2 \Delta W ) ^ { 2 / 3 }$ gives $\phi _ { \operatorname* { m i n } } = 3 \cdot 2 ^ { - 2 / 3 } ( \Delta V ) ^ { 2 / 3 } ( \Delta W ) ^ { 1 / 3 }$

Two degenerate cases with $\Delta K = 0$ and $\Delta _ { 0 } > 0$ are simple. If $\Delta W = 0$ and $\Delta V > 0$ , the complex class wins for $h > \Delta V / \Delta _ { 0 }$ . If $\Delta V = 0$ and $\Delta W > 0 ,$ , it wins for $h < \sqrt { \Delta _ { 0 } / \Delta W }$ . In the first case, $\phi ( h ) = \Delta V / h$ , and in the second case, $\phi ( h ) = \Delta W h ^ { 2 }$ . The U-shaped pattern above requires $\Delta W > 0$ . However, ∆W can be negative if regularization or model type filters drifting directions more strongly in the complex class. In that case, the U-shaped characterization does not apply. Thus parameter count alone decides neither side of the comparison in (15) and (16).

## A.3 Exact prediction risk and optimal memory for ridge regression

We give the details of Section 3.3 here. We first bound the risk of ridge regression under drift and seria dependence (Section A.3.1). We then make the memory dependence exact and prove Proposition 3 of Section 3.3 (Section A.3.2).

## A.3.1 Direction-aware ridge risk under temporal dependence

We now study ridge regression when the coeficients drift, as in Section 3.3. Let t be the forecast date. The model trains on the h observations that precede t, indexed by $j = 1 , \dotsc , h$ . Observation $t - j$ has predictor vector $x _ { t - j } \in \mathrm { R } ^ { p }$ and outcome $y _ { t - j } \in \mathrm { R }$ . We assume

$$
y _ { t - j } = x _ { t - j } ^ { \top } \beta _ { t - j } + \varepsilon _ { t - j } , \qquad \beta _ { t - j } = \beta _ { t } - j { \dot { \beta } } _ { t } + r _ { t , j } .\tag{19}
$$

Here $\beta _ { s } \in \mathrm { R } ^ { p }$ is the coeficient vector at date $s ,$ and $\varepsilon _ { s }$ is the regression error. The vector $\dot { \beta } _ { t }$ is the per-period drift of the coeficients near date t. The term $\boldsymbol { r } _ { t , j }$ is the remainder that this linear drift leaves out. The estimation target is the current coeficient $\beta _ { t }$ . We condition on the realized training design $X _ { h } = ( x _ { t - 1 } , \dots , x _ { t - h } )$ . We treat the coeficient path in (19) as fixed, or as a function of $X _ { h }$

We define the sample second-moment matrix $S _ { h }$ , the ridge inverse $A _ { h , \lambda } .$ , the ridge filter $G _ { h , \lambda } ,$ , and the mean lag µ<sub>h</sub>:

$$
S _ { h } = { \frac { 1 } { h } } \sum _ { j = 1 } ^ { h } x _ { t - j } x _ { t - j } ^ { \top } , \quad A _ { h , \lambda } = ( S _ { h } + \lambda I ) ^ { - 1 } ,
$$

$$
G _ { h , \lambda } = A _ { h , \lambda } S _ { h } , \qquad \mu _ { h } = { \frac { h + 1 } { 2 } } ,\tag{20}
$$

where $\lambda > 0$ is the ridge penalty. The ridge estimator is

$$
\widehat { \beta } _ { t , h , \lambda } = \arg \operatorname* { m i n } _ { b } \left\{ \frac { 1 } { h } \sum _ { j = 1 } ^ { h } ( y _ { t - j } - x _ { t - j } ^ { \top } b ) ^ { 2 } + \lambda \| b \| ^ { 2 } \right\} .\tag{21}
$$

Three further quantities enter the estimation error:

$$
H _ { h } = \frac { 1 } { h } \sum _ { j = 1 } ^ { h } j x _ { t - j } x _ { t - j } ^ { \top } - \mu _ { h } S _ { h } ,\tag{22}
$$

$$
q _ { t , h } = \frac { 1 } { h } \sum _ { j = 1 } ^ { h } x _ { t - j } x _ { t - j } ^ { \top } r _ { t , j } , \qquad z _ { t , h } = \frac { 1 } { h } \sum _ { j = 1 } ^ { h } x _ { t - j } \varepsilon _ { t - j } .\tag{23}
$$

We call $H _ { h }$ the time-weighted design imbalance. It vanishes if every lag carries the same second-moment matrix. We call $q _ { t , h }$ the curvature aggregate, because it sums the design-weighted remainders $\boldsymbol { r } _ { t , j }$ . We call $z _ { t , h }$ the score. Its conditional mean is $\bar { z } _ { t , h } = \mathrm { E } [ z _ { t , h } \mid X _ { h } ]$

We assume that the centered conditional score variance obeys

$$
\operatorname { V a r } ( z _ { t , h } \mid X _ { h } ) \preceq \frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } S _ { h } .\tag{A1}
$$

Here $\sigma _ { t } ^ { 2 } > 0$ is the error variance near date t, and $\kappa _ { h } \geq 1$ is a dependence factor. We use (A1) to bound the centered long-run score variance. We do not set the conditional score mean to zero. The stronger condition $\bar { z } _ { t , h } = 0$ is strict score exogeneity at the window level. It is strong because $X _ { h }$ contains regressors dated after some of the errors. Predetermined predictors, lagged returns, and persistent predictors with return-correlated innovations need not satisfy it (Stambaugh, 1999). For this reason, we keep $\bar { z } _ { t , h }$ in the bound. If the errors are independent of the whole design, conditionally homoskedastic, and conditionally serially uncorrelated, then $\kappa _ { h } = 1$ . Serial dependence can require a bounded inflation factor $\kappa _ { h } > 1$

For a vector $v ,$ write $\| \boldsymbol { v } \| _ { S _ { h } } ^ { 2 } = \boldsymbol { v } ^ { \top } S _ { h } \boldsymbol { v }$ . This local design norm measures the squared prediction diference, averaged over the training rows. We split the estimation risk into four components:

$$
\begin{array} { r l } & { B _ { \mathrm { r e g } , t } ( h , \lambda ) = \| ( I - G _ { h , \lambda } ) \beta _ { t } \| _ { S _ { h } } ^ { 2 } , } \\ & { \qquad d _ { 2 , h } ( \lambda ) = \operatorname { t r } \left[ S _ { h } ^ { 2 } ( S _ { h } + \lambda I ) ^ { - 2 } \right] , } \\ & { \qquad D _ { t , h } ^ { 2 } ( \lambda ) = \| G _ { h , \lambda } \dot { \beta } _ { t } \| _ { S _ { h } } ^ { 2 } , } \\ & { \qquad \rho _ { t , h } ^ { 2 } ( \lambda ) = \| A _ { h , \lambda } ( q _ { t , h } - H _ { h } \dot { \beta } _ { t } + \bar { z } _ { t , h } ) \| _ { S _ { h } } ^ { 2 } . } \end{array}\tag{24}
$$

The term $B _ { \mathrm { r e g } , t }$ is the regularization bias, the part of $\beta _ { t }$ that ridge shrinks away. The term $d _ { 2 , h }$ is the ridgefiltered spectral dimension, which scales the variance. The term $D _ { t , h } ^ { 2 }$ is the drift energy that the ridge filter passes through. The term $\rho _ { t , h } ^ { 2 }$ is a remainder. It records three departures from the ideal case: curvature in the coeficient path, time imbalance in the realized design, and a nonzero conditional score mean. In Section 3.3, we suppress the penalty λ and the date t in the notation. We write $A _ { h } , G _ { h } , z , B _ { \mathrm { r e g } } , D ^ { 2 } , \rho ^ { 2 }$ , and $\mathcal { E } _ { t } ^ { X } ( h )$ there for $A _ { h , \lambda } , G _ { h , \lambda } , z _ { t , h } , B _ { \mathrm { r e g } , t } ( h , \lambda ) , D _ { t , h } ^ { 2 } ( \lambda ) , \rho _ { t , h } ^ { 2 } ( \lambda )$ , and $\mathcal { E } _ { t } ^ { X } ( h , \lambda )$ . We keep the full notation here.

Proposition A.4 (conditional ridge identity and bound). The conditional excess prediction risk in the local design norm, $\dot { \mathcal { E } } _ { t } ^ { X } ( h , \lambda )$ , satisfies the exact identity

$$
\begin{array} { r l } & { { \mathcal { E } _ { t } ^ { X } ( h , \lambda ) } : = { { \mathrm { E } } \Big [ \| \widehat \beta _ { t , h , \lambda } - \beta _ { t } \| _ { S _ { h } } ^ { 2 } \mid { X } _ { h } \Big ] } } \\ & { \qquad = { \Big \| ( I - G _ { h , \lambda } ) \beta _ { t } + \mu _ { h } G _ { h , \lambda } { { \dot { \beta } } _ { t } } } } \\ & { \qquad \quad - A _ { h , \lambda } { ( q _ { t , h } - H _ { h } { { \dot { \beta } } _ { t } } + { { \bar { z } } _ { t , h } } ) } { \Big \| } _ { S _ { h } } ^ { 2 } } \\ & { \qquad \quad + \operatorname { t r } [ S _ { h } A _ { h , \lambda } \operatorname { V a r } ( { { z } _ { t , h } } \mid { X } _ { h } ) A _ { h , \lambda } ] . } \end{array}\tag{25}
$$

Under (A1), it also satisfies, for every $\eta , \zeta > 0$

$$
\begin{array} { r l } & { \mathcal { E } _ { t } ^ { X } ( h , \lambda ) \leq ( 1 + \eta ) ( 1 + \zeta ) B _ { \mathrm { r e g } , t } ( h , \lambda ) } \\ & { \qquad + ( 1 + \eta ) ( 1 + \zeta ^ { - 1 } ) \mu _ { h } ^ { 2 } D _ { t , h } ^ { 2 } ( \lambda ) } \\ & { \qquad + ( 1 + \eta ^ { - 1 } ) \rho _ { t , h } ^ { 2 } ( \lambda ) + \displaystyle \frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } d _ { 2 , h } ( \lambda ) . } \end{array}\tag{26}
$$

Proof. First we expand the estimator. The ridge normal equations and (19) give

$$
\widehat { \beta } _ { t , h , \lambda } = A _ { h , \lambda } \big [ S _ { h } \beta _ { t } - ( \mu _ { h } S _ { h } + H _ { h } ) \dot { \beta } _ { t } + q _ { t , h } + z _ { t , h } \big ] .
$$

Subtracting $\beta _ { t }$ yields

$$
\begin{array} { r l } & { \widehat { \beta } _ { t , h , \lambda } - \beta _ { t } = - ( I - G _ { h , \lambda } ) \beta _ { t } - \mu _ { h } G _ { h , \lambda } \dot { \beta } _ { t } } \\ & { ~ + A _ { h , \lambda } ( q _ { t , h } - H _ { h } \dot { \beta } _ { t } + \bar { z } _ { t , h } ) } \\ & { ~ + A _ { h , \lambda } ( z _ { t , h } - \bar { z } _ { t , h } ) . } \end{array}
$$

The first three terms are functions of $X _ { h }$ . The last term has conditional mean zero given $X _ { h }$ . Hence the conditional risk equals the squared norm of the first three terms plus the variance of the last term. This is (25). The identity needs no bound on the variance. We use (A1) only in the final step.

To prove (26), we use the inequality $\| v + w \| ^ { 2 } \leq ( 1 + c ) \| v \| ^ { 2 } + ( 1 + c ^ { - 1 } ) \| w \| ^ { 2 }$ , which holds for every $c > 0$ We apply it twice. The first application, with $c = \eta .$ , separates the remainder from the leading bias. The second application, with $c = \zeta ,$ separates the regularization bias from the filtered drift. For the variance term, $A _ { h , \lambda } S _ { h } A _ { h , \lambda }$ is positive semidefinite. Hence (A1) and the cyclicity of the trace bound the variance term. Because $S _ { h }$ and $A _ { h , \lambda }$ commute, the bound equals

$$
\frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } \operatorname { t r } \left[ S _ { h } ^ { 2 } ( S _ { h } + \lambda I ) ^ { - 2 } \right] .
$$

Collecting the terms proves (26).

Two features of (26) matter for the choice of memory (Fig. 1B). First, the variance term scales with the ridgefiltered dimension $d _ { 2 , h } ( \lambda )$ and decays like $h ^ { - 1 }$ . If $\mathrm { s u p } _ { h } \kappa _ { h } < \infty$ , serial dependence preserves this $h ^ { - 1 }$ order. Second, the obsolete-data term scales with $\mu _ { h } ^ { 2 } D _ { t , h } ^ { 2 } ( \lambda )$ , which uses the filtered drift $G _ { h , \lambda } \dot { \beta } _ { t }$ . Ridge shrinks drift in heavily shrunk or low-variance directions. Thus such drift adds little to the risk in the local prediction norm. We now connect the local norm to next-period risk. Let $\Sigma _ { t }$ be the second-moment matrix of the predictor at the forecast date, conditional on $X _ { h }$ . We measure next-period excess prediction risk by $\operatorname { E } [ \| \widehat { \beta } _ { t , h , \lambda } - \widehat { \beta } _ { t } \| _ { \Sigma _ { t } } ^ { 2 } \mid X _ { h } ]$ Suppose $\Sigma _ { t } \preceq c _ { + } S _ { h }$ for a constant $c _ { + } > 0$ . Then this risk is at most $c _ { + } \mathcal { E } _ { t } ^ { X } ( h , \lambda )$ . Hence $c _ { + }$ times the right side of (26) also bounds the next-period risk. If the covariance is locally stable, one would expect $S _ { h } , d _ { 2 , h }$ , and $D _ { t , h }$ to approach their local population counterparts. Finally, exact local linearity, a time-balanced design, and $\bar { z } _ { t , h } = 0$ together imply $\rho _ { t , h } = 0$

## A.3.2 Optimal memory and minimized risk

The bound (26) controls the risk from above. It does not identify the risk-minimizing memory. An exact memory calculation needs stronger assumptions than (A1). Hence we study a simplified family of candidate histories. All histories share the same local second-moment matrix S, coeficient $\beta _ { t } .$ drift $\dot { \beta } _ { t }$ , and penalty λ. We also assume $\rho _ { t , h } = 0$ , that is, $q _ { t , h } - H _ { h } \dot { \beta } _ { t } + \bar { z } _ { t , h } = 0$ . This holds, for example, if $q _ { t , h } , H _ { h }$ , and $\bar { z } _ { t , h }$ all vanish. The centered score covariance equals exactly $\kappa \sigma _ { t } ^ { 2 } S / h$ for a constant $\kappa > 0$ . These assumptions define a simplified comparison. We do not claim them for arbitrary nested historical windows. Write $G = ( S + \lambda I ) ^ { - 1 } S$ and $d _ { 2 } ( \lambda ) = \mathrm { t r } [ S ^ { 2 } ( S + \lambda I ) ^ { - 2 } ]$ Let $\langle u , v \rangle _ { S } = u ^ { \top } S v$ . Define

$$
\begin{array} { r l } & { B _ { t } = \| ( I - G ) \beta _ { t } \| _ { S } ^ { 2 } , } \\ & { J _ { t } = \langle ( I - G ) \beta _ { t } , G \dot { \beta } _ { t } \rangle _ { S } , } \\ & { Q _ { t } = \| G \dot { \beta } _ { t } \| _ { S } ^ { 2 } , \qquad V _ { t } = \kappa \sigma _ { t } ^ { 2 } d _ { 2 } ( \lambda ) . } \end{array}\tag{27}
$$

Here $B _ { t }$ is the regularization bias, $Q _ { t }$ is the filtered drift energy, $J _ { t }$ is their cross term, and $V _ { t }$ is the variance scale. Under these assumptions, (25) reduces to the exact interpolation

$$
\mathcal { E } _ { t } ^ { \circ } ( h , \lambda ) = B _ { t } + 2 \mu _ { h } J _ { t } + \mu _ { h } ^ { 2 } Q _ { t } + \frac { V _ { t } } { h } , \qquad \mu _ { h } = \frac { h + 1 } { 2 } ,\tag{28}
$$

where $\mathcal { E } _ { t } ^ { \circ } ( h , \lambda )$ is the conditional risk under these assumptions. This is the risk (7) of Section 3.3, which we write there as $\mathcal { E } _ { t } ( h )$

The risk (28) has the form (13) of Section A.2. Indeed, $2 \mu _ { h } J _ { t } = ( h + 1 ) J _ { t }$ and $\mu _ { h } ^ { 2 } Q _ { t } = ( h + 1 ) ^ { 2 } Q _ { t } / 4$ . Collecting the terms in $h ,$ we can take

$$
\begin{array} { c c } { { B _ { k } = B _ { t } + J _ { t } + Q _ { t } / 4 , } } & { { V _ { k } = V _ { t } , } } \\ { { K _ { k } = J _ { t } + Q _ { t } / 2 , } } & { { W _ { k } = Q _ { t } / 4 , } } \end{array}\tag{29}
$$

and we treat any oracle approximation error separately in $\mathcal { A } _ { k }$ . These are the coeficients that Section 3.3 states. This choice keeps the known linear term $K _ { k } h$ explicit. A quadratic-only surrogate would move this term into $e _ { k }$ . It would then leave a remainder of order h, which can make a narrow ranking margin uninformative. We restate Proposition 3 of Section 3.3 and prove it.

Corollary A.5 (cross-term-aware interior memory). Assume $V _ { t } > 0$ and $Q _ { t } > 0$ . Extend (28) to real $h > 0$ . The extended risk is strictly convex and has a unique positive minimizer $h _ { t } ^ { * }$ (Fig. 1B). It satisfies

$$
\boxed { \frac { V _ { t } } { ( h _ { t } ^ { * } ) ^ { 2 } } = J _ { t } + \frac { h _ { t } ^ { * } + 1 } { 2 } Q _ { t } . }\tag{30}
$$

If $J _ { t } = 0$ , this condition reduces to the exact finite-h cubic

$$
( h _ { t } ^ { \ast } ) ^ { 2 } ( h _ { t } ^ { \ast } + 1 ) = \frac { 2 V _ { t } } { Q _ { t } } .\tag{31}
$$

If $h _ { t } ^ { * }$ is large and $J _ { t }$ is negligible relative to $h _ { t } ^ { * } Q _ { t }$ , then $h _ { t } ^ { * } \sim ( 2 V _ { t } / Q _ { t } ) ^ { 1 / 3 }$ . The sign of $J _ { t }$ shifts the optimum. A negative $J _ { t }$ lengthens the optimal history, and a positive $J _ { t }$ shortens it. The cube-root formula is only approximate. It fails for general ridge shrinkage and for short histories.

Proof. Diferentiating (28) gives $- V _ { t } / h ^ { 2 } + J _ { t } + ( h + 1 ) Q _ { t } / 2$ . Its derivative is $2 V _ { t } / h ^ { 3 } + Q _ { t } / 2 > 0$ , so the first derivative is strictly increasing. It tends to −∞ as $h \to 0 ^ { + }$ and to +∞ as $h \to \infty$ . Hence it crosses zero exactly once on $( 0 , \infty )$ , and this zero is the unique minimizer. Setting $J _ { t } = 0$ and multiplying the first-order condition by $2 h ^ { 2 } { \mathrm { ~ g i v e s ~ } } ( 3 1 )$

Integer histories need a final step. For admissible histories $h \geq 1$ , we clip the continuous optimum to the permitted interval before we compare neighboring integers. In particular, we use $h = 1$ if $h _ { t } ^ { * } < 1$ . If $Q _ { t } = 0 ;$ then $J _ { t } = 0$ by the Cauchy–Schwarz inequality in the S-inner product. In that case, any positive $V _ { t }$ favors the longest available history. Regularization changes $V _ { t } , Q _ { t }$ , and $J _ { t }$ jointly.

A simpler surrogate keeps only the leading terms. We drop $J _ { t }$ and replace $\mu _ { h } ^ { 2 }$ by $h ^ { 2 } / 4$ . The memory-dependent part then equals $V _ { t } / h + W _ { t } h ^ { 2 }$ with $W _ { t } = Q _ { t } / 4$ . Its minimum is

$$
\operatorname* { m i n } _ { h > 0 } \left\{ \frac { V _ { t } } { h } + W _ { t } h ^ { 2 } \right\} = 3 \cdot 2 ^ { - 2 / 3 } V _ { t } ^ { 2 / 3 } W _ { t } ^ { 1 / 3 } .\tag{32}
$$

This value is the minimum of the surrogate, not of (28).

## A.4 Surrogate approximations and transfer bounds for trees and neural networks

We give the details of Section 3.4 here. Boosted trees, RNNs, LSTMs, and Transformers are not linear in their parameters. Hence we analyze a surrogate that fixes the feature map before we fit the readout. We bound the risk of this surrogate for frozen feature maps and fixed-topology boosting (Section A.4.1), and we derive its exact spectral phase boundary (Section A.4.2). We then bound the gap between the surrogate and the trained mode (Section A.4.3).

## A.4.1 Frozen feature maps and fixed-tree boosting

The ridge argument of Section A.3.1 extends exactly to a feature map that we fix before we observe the outcomes that fit its readout. Here “fixed” means that the parameters of the map do not change during the readout fit. The output of the map may still depend on the input sequence. We need this condition because a feature map that we select on the same responses as its readout is not fixed. For trees, the feature map is the tree topology. For the neural models, it is the encoder that produces the final hidden state or the pooled token state.

Let $v _ { s }$ be the raw input at date s. It can be a vector or a sequence of length $L .$ Let $\varphi$ map an input into $\mathrm { R } ^ { p } { \mathrm { . } }$ and call $\varphi ( v _ { s } )$ the feature vector. We may choose $\varphi$ from an independent or an earlier sample. Let $\mathcal { G } _ { h }$ be a sigma-field that contains the chosen map $\varphi$ and the current design $( v _ { t - 1 } , \ldots , v _ { t - h } )$ . We condition on $\mathcal { G } _ { h }$ and write $z _ { j } = \varphi ( v _ { t - j } )$ . We assume that the coeficient path, the approximation residuals, and all matrices below are $\mathcal { G } _ { h }$ -measurable. We also assume

$$
\begin{array} { r l } & { y _ { t - j } = z _ { j } ^ { \top } \theta _ { t - j } + b _ { t , j } + \varepsilon _ { t - j } , } \\ & { \theta _ { t - j } = \theta _ { t } - j \dot { \theta } _ { t } + r _ { t , j } , \qquad j = 1 , \dotsc , h . } \end{array}\tag{33}
$$

Here $\theta _ { s } \in \mathrm { R } ^ { p }$ is the feature coeficient at date $s , \ { \dot { \theta } } _ { t }$ is its drift per period, and $\boldsymbol { r } _ { t , j }$ is the remainder of the linear drift. The term $b _ { t , j }$ is the approximation residual, the part of the conditional mean that the features cannot represent.

We now make $\theta _ { s }$ and $b _ { t , j }$ unique. Let $P _ { s }$ be the distribution of the input $v _ { s } ,$ and let $m _ { s } ( v ) = \operatorname { E } [ y _ { s } \mid v _ { s } = v ]$ be the conditional mean function at date s. Write $\mathrm { E } _ { s }$ for expectation under $P _ { s } ,$ , and assume $\mathrm { E } _ { s } [ \varphi \varphi ^ { \top } ] \succ 0$ . We define $\theta _ { s }$ as the unique $L _ { 2 } ( P _ { s } )$ projection coeficient of $m _ { s }$ onto $\varphi .$ . We define $b _ { s } ( v ) = m _ { s } ( v ) - \varphi ( v ) ^ { \top } \theta _ { s }$ . Then $\mathrm { E } _ { s } [ \varphi b _ { s } ] = 0$ and $b _ { t , j } = b _ { t - j } ( v _ { t - j } )$ . This orthogonality holds in the population. It need not hold in a realized historical design.

We define the feature second-moment matrix, the ridge inverse and filter, the time imbalance, the aggregate bias, and the score:

$$
\begin{array} { l l } { { S _ { h } ^ { \varphi } = \displaystyle \frac { 1 } { h } \sum _ { j } z _ { j } z _ { j } ^ { \top } , \quad A _ { h } ^ { \varphi } = ( S _ { h } ^ { \varphi } + \lambda I ) ^ { - 1 } , } } \\ { { \ } } \\ { { G _ { h } ^ { \varphi } = A _ { h } ^ { \varphi } S _ { h } ^ { \varphi } , \quad H _ { h } ^ { \varphi } = \displaystyle \frac { 1 } { h } \sum _ { j } j z _ { j } z _ { j } ^ { \top } - \mu _ { h } S _ { h } ^ { \varphi } , } } \\ { { \ } } \\ { { q _ { h } ^ { \varphi } = \displaystyle \frac { 1 } { h } \sum _ { j } z _ { j } ( z _ { j } ^ { \top } r _ { t , j } + b _ { t , j } ) , } } \\ { { \ } } \\ { { \displaystyle s _ { h } ^ { \varphi } = \displaystyle \frac { 1 } { h } \sum _ { j } z _ { j } \varepsilon _ { t - j } , \quad \quad \mu _ { h } = \displaystyle \frac { h + 1 } { 2 } , } } \\ { { \ } } \\ { { \tilde { s } _ { h } ^ { \varphi } = \displaystyle { { \mathrm E } [ s _ { h } ^ { \varphi } | \ \xi _ { h } ] } . } } \end{array}\tag{34}
$$

We fit the readout $\widehat { \theta } _ { h }$ by ridge regression of $y _ { t - j }$ on $z _ { j }$ with penalty $\lambda \| \theta \| ^ { 2 }$ and $\lambda > 0$ . That is, $\widehat { \theta } _ { h }$ minimizes $\begin{array} { r } { h ^ { - 1 } \sum _ { j = 1 } ^ { h } ( y _ { t - j } - z _ { j } ^ { \top } \theta ) ^ { 2 } + \lambda \| \theta \| ^ { 2 } } \end{array}$ . We assume, with $\kappa _ { h }$ and $\sigma _ { t } ^ { 2 }$ as in (A1),

$$
\operatorname { V a r } ( s _ { h } ^ { \varphi } \mid { \mathcal { G } } _ { h } ) \preceq { \frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } } S _ { h } ^ { \varphi } .\tag{35}
$$

We do not assume a zero conditional score mean. The term $\bar { s } _ { h } ^ { \varphi }$ carries any whole-window score bias, including the bias that persistent endogenous predictors induce. Let $M _ { t } \succeq 0$ be any $\mathcal { G } _ { h }$ -measurable matrix that defines the prediction norm, and write $\| \boldsymbol { v } \| _ { M _ { t } } ^ { 2 } = \boldsymbol { v } ^ { \top } M _ { t } \boldsymbol { v }$

Proposition A.6 (frozen-feature-map ridge bound). Assume the model (33), the projection identification above, and (35). The conditional risk of the fitted readout for the represented target $\varphi ( \cdot ) ^ { \top } \theta _ { t }$ is then exactly

$$
\begin{array} { r l } & { { \mathcal E } _ { t } ^ { \varphi } ( h , \lambda ; M _ { t } ) : = \mathrm { E } \Big [ \| \widehat \theta _ { h } - \theta _ { t } \| _ { M _ { t } } ^ { 2 } \mid { \mathcal G } _ { h } \Big ] } \\ & { \quad \quad \quad \quad = \Big \| ( I - G _ { h } ^ { \varphi } ) \theta _ { t } + \mu _ { h } G _ { h } ^ { \varphi } \dot { \theta } _ { t } } \\ & { \quad \quad \quad \quad - A _ { h } ^ { \varphi } ( q _ { h } ^ { \varphi } - H _ { h } ^ { \varphi } \dot { \theta } _ { t } + \bar { s } _ { h } ^ { \varphi } ) \Big \| _ { M _ { t } } ^ { 2 } } \\ & { \quad \quad \quad \quad + \mathrm { t r } [ M _ { t } A _ { h } ^ { \varphi } \mathrm { V a r } ( s _ { h } ^ { \varphi } \mid { \mathcal G } _ { h } ) A _ { h } ^ { \varphi } ] . } \end{array}\tag{36}
$$

Set

$$
\begin{array} { r l } & { \quad B _ { h } ^ { \varphi } = \| ( I - G _ { h } ^ { \varphi } ) \theta _ { t } \| _ { M _ { t } } ^ { 2 } , \quad D _ { h , \varphi } ^ { 2 } = \| G _ { h } ^ { \varphi } \dot { \theta } _ { t } \| _ { M _ { t } } ^ { 2 } , } \\ & { \quad \rho _ { h , \varphi } ^ { 2 } = \| A _ { h } ^ { \varphi } ( q _ { h } ^ { \varphi } - H _ { h } ^ { \varphi } \dot { \theta } _ { t } + \bar { s } _ { h } ^ { \varphi } ) \| _ { M _ { t } } ^ { 2 } , } \\ & { \quad d _ { h } ^ { \varphi } = \mathrm { t r } [ M _ { t } A _ { h } ^ { \varphi } { \cal S } _ { h } ^ { \varphi } A _ { h } ^ { \varphi } ] . } \end{array}\tag{37}
$$

Then, for every $\eta , \zeta > 0$

$$
\begin{array} { r l } & { \mathcal { E } _ { t } ^ { \varphi } ( h , \lambda ; M _ { t } ) \leq ( 1 + \eta ) ( 1 + \zeta ) B _ { h } ^ { \varphi } } \\ & { \qquad + ( 1 + \eta ) ( 1 + \zeta ^ { - 1 } ) \mu _ { h } ^ { 2 } D _ { h , \varphi } ^ { 2 } } \\ & { \qquad + ( 1 + \eta ^ { - 1 } ) \rho _ { h , \varphi } ^ { 2 } + \frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } d _ { h } ^ { \varphi } . } \end{array}\tag{38}
$$

Proof. The ridge normal equations give

$$
\begin{array} { r l } & { \widehat { \theta } _ { h } - \theta _ { t } = - ( I - G _ { h } ^ { \varphi } ) \theta _ { t } - \mu _ { h } G _ { h } ^ { \varphi } \dot { \theta } _ { t } } \\ & { \phantom { \frac { 1 } { 2 } } + A _ { h } ^ { \varphi } ( q _ { h } ^ { \varphi } - H _ { h } ^ { \varphi } \dot { \theta } _ { t } ) + A _ { h } ^ { \varphi } s _ { h } ^ { \varphi } . } \end{array}
$$

We center $s _ { h } ^ { \varphi }$ at $\bar { s } _ { h } ^ { \varphi }$ and take the conditional squared $M _ { t ^ { - } } \mathrm { n o r m }$ . This proves (36). Two applications of Young’s inequality bound the squared bias, as in the proof of the ridge bound in Section A.3.1. The covariance order in (35) bounds the trace by $\kappa _ { h } \sigma _ { t } ^ { 2 } d _ { h } ^ { \varphi } / h$ . This proves (38).

Two choices of $M _ { t }$ are useful. If $M _ { t } = S _ { h } ^ { \varphi }$ , the variance term becomes $\mathrm { t r } [ ( S _ { h } ^ { \varphi } ) ^ { 2 } ( S _ { h } ^ { \varphi } + \lambda I ) ^ { - 2 } ]$ . If $M _ { t }$ is the next-period feature covariance, the bound measures risk in the norm that matters for forecasting. The bound concerns the represented target only. A bound for the full conditional mean must also include the approximation residual at the forecast date. To see this, take a $\mathcal { G } _ { h }$ -measurable evaluation design $Z _ { * } \in \mathrm { R } ^ { n _ { * } \times p }$ with $n _ { * }$ rows of feature vectors. Let $b _ { * } \in \mathrm { R } ^ { n }$ <sup>∗</sup> be the approximation residuals at these rows, and set $M _ { t } = Z _ { * } ^ { \top } Z _ { * } / n _ { * }$ . For any $\tau > 0$ , the prediction error for the full conditional mean satisfies

$$
\begin{array} { r l } & { \mathrm { E } \Big [ \| Z _ { * } ( \widehat \theta _ { h } - \theta _ { t } ) - b _ { * } \| ^ { 2 } / n _ { * } \mid \mathcal G _ { h } \Big ] } \\ & { \quad \leq ( 1 + \tau ) \mathcal E _ { t } ^ { \varphi } ( h , \lambda ; M _ { t } ) + ( 1 + \tau ^ { - 1 } ) \| b _ { * } \| ^ { 2 } / n _ { * } . } \end{array}\tag{39}
$$

This follows from the same two-term norm inequality $\begin{array} { r } { \| v + w \| ^ { 2 } \leq ( 1 + \tau ) \| v \| ^ { 2 } + ( 1 + \tau ^ { - 1 } ) \| w \| ^ { 2 } . } \end{array}$

Corollary A.7 (tree and sequence feature maps). Assume that the feature map does not depend on the readout outcomes and that (35) holds. Then the bound (38) of Proposition A.6 holds for each of the following maps:

• Prechosen trees. Let $\varphi _ { \mathrm { t r e e } } ( x )$ concatenate the leaf indicators of prechosen tree partitions. Joint ridge fitting of the additive leaf coeficients then obeys (38). This fit is a fixed-tree surrogate. It difers from the usual greedy XGBoost fit.

• Frozen RNN or LSTM. Let $\varphi _ { \mathrm { R N N } } ( v )$ or $\varphi _ { \mathrm { L S T M } } ( v )$ be the final hidden state of the encoder. A ridge readout on this state obeys (38). It has its own covariance spectrum and coeficient drift.

• Frozen Transformer. Let $\varphi _ { \mathrm { T r } } ( v )$ be the final-token state or a pooled state of the encoder. A ridge readout on this state obeys the same bound. Attention can depend on the input sequence, but the fitted parameters stay frozen.

We distinguish two lengths. The memory h is the number of training dates that fit the readout. The context length L is the number of time steps in the input sequence that constructs each feature vector.

Suppose that, over the candidate histories, we can approximate $d _ { h } ^ { \varphi } , \ D _ { h , \varphi } ^ { 2 } , \ \kappa _ { h }$ , and the remaining terms by constants. The memory-dependent part of the bound then has the form $V _ { \varphi } / h + K _ { \varphi } h + W _ { \varphi } h ^ { 2 }$ . This is the form (13) that Section 3.4 uses for the surrogate risk. Here $V _ { \varphi }$ is proportional to $\kappa \sigma _ { t } ^ { 2 } d ^ { \varphi }$ , and $W _ { \varphi }$ is proportional to $D _ { h , \varphi } ^ { 2 } . \mathrm { ~ H ~ } \mu _ { h } ^ { 2 }$ is the only source of time dependence in the bound, then $K _ { \varphi } = 2 W _ { \varphi }$ exactly. For positive $V _ { \varphi }$ and $W _ { \varphi }$ , this form has a unique positive continuous minimizer. It solves

$$
\boxed { 2 W _ { \varphi } ( h _ { \varphi } ^ { * } ) ^ { 3 } + K _ { \varphi } ( h _ { \varphi } ^ { * } ) ^ { 2 } - V _ { \varphi } = 0 . }\tag{40}
$$

For admissible integer histories $h \geq 1$ , we clip the continuous solution to the history interval and compare the neighboring integers. This procedure minimizes the stated surrogate. It does not minimize the risk of a retrained encoder.

We can also analyze the stagewise squared-loss updates of boosting for fixed tree topologies. This avoids refitting all leaf coeficients jointly. Let $N _ { \mathrm { t r e e } }$ be the number of boosting stages. Tree k has $T _ { k }$ leaves, and $Z _ { k } \in \{ 0 , 1 \} ^ { h \times T _ { k } }$ is its matrix of leaf indicators on the h training rows. We treat each $Z _ { k }$ as fixed given $\mathcal { G } _ { h }$ . Let $\mathsf { W } _ { k }$ be a nonnegative diagonal matrix of fixed row-subsampling weights, and set $\mathsf { W } _ { k } = I$ for full-row fitting. Let $\lambda _ { k } > 0$ be the $\ell _ { 2 }$ leaf penalty of stage $k ,$ and let $0 < \nu \leq 1$ be the learning rate. We assume a fixed zero initial prediction and no $\ell _ { 1 }$ penalty. Define the stage-k leaf smoother $P _ { k }$ and the boosting smoother $T _ { h }$ by

$$
P _ { k } = Z _ { k } ( Z _ { k } ^ { \top } \mathsf { W } _ { k } Z _ { k } + \lambda _ { k } I ) ^ { - 1 } Z _ { k } ^ { \top } \mathsf { W } _ { k } ,\tag{41}
$$

$$
T _ { h } = I - \prod _ { k = N _ { \mathrm { t r e e } } } ^ { 1 } ( I - \nu P _ { k } ) .\tag{42}
$$

The product in (42) runs from $k = N _ { \mathrm { t r e e } }$ on the left to $k = 1$ on the right. Let $\mathbf y _ { h } = ( y _ { t - 1 } , \dots , y _ { t - h } ) ^ { \top }$ . The fitted values at the training inputs are $T _ { h } \mathbf { y } _ { h }$

Let $m _ { t } = ( m _ { t } ( v _ { t - 1 } ) , \dots , m _ { t } ( v _ { t - h } ) ) ^ { \intercal }$ be the current regression function evaluated at the historical inputs. Let $\mathsf { L } _ { h } = \mathrm { d i a g } ( 1 , \ldots , h )$ be the matrix of lags. We assume that the historical conditional means obey

$$
\operatorname { E } [ \mathbf { y } _ { h } \mid \mathcal { G } _ { h } ] = m _ { t } - \mathsf { L } _ { h } \dot { m } _ { t } + \boldsymbol { r } _ { h } ,\tag{43}
$$

where we treat the drift vector $\dot { m } _ { t }$ and the remainder $r _ { h }$ as constants given $\mathcal { G } _ { h }$ . The remainder $r _ { h }$ includes any distortion of the conditional mean that comes from conditioning on the whole historical design. Hence $r _ { h } = 0$ is a strong whole-window restriction. It can fail with persistent endogenous predictors or lagged returns. Define the centered residual $\boldsymbol { \varepsilon } _ { h } = \mathbf { y } _ { h } - \operatorname { E } [ \mathbf { y } _ { h } \mid \mathcal { G } _ { h } ]$ . We assume

$$
\operatorname { E } [ \varepsilon _ { h } \mid { \mathcal { G } } _ { h } ] = 0 , \qquad \Omega _ { h } : = \operatorname { V a r } ( \varepsilon _ { h } \mid { \mathcal { G } } _ { h } ) \preceq \kappa _ { h } \sigma _ { t } ^ { 2 } I .\tag{44}
$$

The mean-zero equality holds by definition of $\varepsilon _ { h }$ . It does not impose strict exogeneity on the original return disturbance. We can take $\kappa _ { h } = 1$ , for example, if the centered errors are conditionally serially uncorrelated and conditionally homoskedastic with variance at most $\sigma _ { t } ^ { 2 }$ given $\mathcal { G } _ { h }$ . Independence from the design and homoskedasticity alone do not sufice, because of-diagonal covariances can remain. We use the scaled norm $\| v \| _ { h } ^ { 2 } = h ^ { - 1 } \| v \| ^ { 2 }$

Proposition A.8 (fixed-topology boosting bound). Assume (41)–(44). The conditional local-design risk of the boosted fit is then exactly

$$
\begin{array} { r l } & { \mathcal { E } _ { t } ^ { \mathrm { b o o s t } } ( h ) : = \mathrm { E } \big [ \| T _ { h } \mathbf { y } _ { h } - m _ { t } \| _ { h } ^ { 2 } \mid \mathcal { G } _ { h } \big ] } \\ & { \quad \quad \quad = \| ( I - T _ { h } ) m _ { t } + T _ { h } \mathsf { L } _ { h } \dot { m } _ { t } - T _ { h } r _ { h } \| _ { h } ^ { 2 } } \\ & { \quad \quad \quad \quad + \frac { 1 } { h } \mathrm { t r } ( T _ { h } \Omega _ { h } T _ { h } ^ { \top } ) . } \end{array}\tag{45}
$$

Define

$$
\begin{array} { r l } & { \quad B _ { h } ^ { \mathrm { b o o s t } } = \| ( I - T _ { h } ) m _ { t } \| _ { h } ^ { 2 } , } \\ & { \quad D _ { h , \mathrm { b o o s t } } ^ { 2 } = \| T _ { h } \mathsf { L } _ { h } \dot { m } _ { t } / \mu _ { h } \| _ { h } ^ { 2 } , } \\ & { \quad \rho _ { h , \mathrm { b o o s t } } ^ { 2 } = \| T _ { h } r _ { h } \| _ { h } ^ { 2 } , \quad d _ { h } ^ { \mathrm { b o o s t } } = \mathrm { t r } ( T _ { h } T _ { h } ^ { \top } ) . } \end{array}\tag{46}
$$

Then, for every $\eta , \zeta > 0$

$$
\begin{array} { r l } {  { \mathcal { E } _ { t } ^ { \mathrm { b o o s t } } ( h ) \le ( 1 + \eta ) ( 1 + \zeta ) B _ { h } ^ { \mathrm { b o o s t } } } } \\ & { \quad \quad \quad \quad + ( 1 + \eta ) ( 1 + \zeta ^ { - 1 } ) \mu _ { h } ^ { 2 } D _ { h , \mathrm { b o o s t } } ^ { 2 } } \\ & { \quad \quad \quad \quad + ( 1 + \eta ^ { - 1 } ) \rho _ { h , \mathrm { b o o s t } } ^ { 2 } + \frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } d _ { h } ^ { \mathrm { b o o s t } } . } \end{array}\tag{47}
$$

Proof. Let $\widehat { m } _ { k }$ be the vector of fitted values after stage $k ,$ with $\widehat { \boldsymbol { m } } _ { 0 } = 0$ . At stage $k ,$ squared-loss minimization on the fixed leaves gives the fitted increment $\nu P _ { k } ( \mathbf { y } _ { h } - \widehat { m } _ { k - 1 } )$ . Hence the residual obeys

$$
\begin{array} { r } { \mathbf y _ { h } - \widehat m _ { k } = ( I - \nu P _ { k } ) ( \mathbf y _ { h } - \widehat m _ { k - 1 } ) . } \end{array}
$$

Iterating from $\widehat { m } _ { 0 } = 0$ yields $\widehat { m } _ { N _ { \mathrm { t r e e } } } = T _ { h } \mathbf { y } _ { h }$ . We substitute (43) and obtain

$$
T _ { h } \mathbf { y } _ { h } - m _ { t } = - ( I - T _ { h } ) m _ { t } - T _ { h } \mathbf { L } _ { h } \dot { m } _ { t } + T _ { h } r _ { h } + T _ { h } \varepsilon _ { h } .
$$

The noise term has conditional mean zero, which gives (45). Two Young inequalities bound the squared bias. The condition (44) bounds the variance by $\kappa _ { h } \sigma _ { t } ^ { 2 } d _ { h } ^ { \mathrm { b o o s t } } / h$ . This proves (47). ■ The bound above measures risk on the training inputs. Forecasting needs the risk on a fixed common evaluation design with $n _ { * }$ rows. Write $\| v \| _ { * } ^ { 2 } = \| v \| ^ { 2 } / n$ for the scaled norm on this design. Let $Z _ { * , k }$ be the matrix of leaf indicators that tree k assigns to the evaluation rows. Define

$$
\begin{array} { l } { { \displaystyle T _ { * , h } = \nu \sum _ { k = 1 } ^ { N _ { \mathrm { t r e e } } } Z _ { * , k } ( Z _ { k } ^ { \top } \mathsf { W } _ { k } Z _ { k } + \lambda _ { k } I ) ^ { - 1 } } } \\ { { \displaystyle \qquad \times \left. Z _ { k } ^ { \top } \mathsf { W } _ { k } \prod _ { \ell = k - 1 } ^ { 1 } ( I - \nu P _ { \ell } ) . \right. } } \end{array}\tag{48}
$$

Here the empty product equals I. This operator maps the training response to the forecast of the fixed-tree model on the evaluation design. We call this forecast the reference forecast. Let $m _ { t , * }$ be the current mean vector on this design. The same calculation as above gives the unclipped reference risk

$$
\begin{array} { r l } & { { \cal L } _ { * , h } ^ { 0 , \mathrm { u n c l i p } } = \mathrm { E } [ \| T _ { * , h } { \bf y } _ { h } - m _ { t , * } \| _ { * } ^ { 2 } \vert \mathcal { G } _ { h } ] } \\ & { \quad \quad \quad = \| T _ { * , h } ( m _ { t } - \mathsf { L } _ { h } \dot { m } _ { t } + r _ { h } ) - m _ { t , * } \| _ { * } ^ { 2 } } \\ & { \quad \quad \quad \quad + n _ { * } ^ { - 1 } \mathrm { t r } ( T _ { * , h } \Omega _ { h } T _ { * , h } ^ { \top } ) . } \end{array}\tag{49}
$$

Under (44), the variance term is at most $\kappa _ { h } \sigma _ { t } ^ { 2 } d _ { * , h } / h$ , where $d _ { * , h } = ( h / n _ { * } ) \operatorname { t r } ( T _ { * , h } T _ { * , h } ^ { \top } )$ . For any $\eta , \zeta > 0$ , two Young inequalities then give the explicit upper bound

$$
\begin{array} { r l r } {  { U _ { * , h } ^ { \mathrm { f i x e d } } : = ( 1 + \eta ) ( 1 + \zeta ) \| m _ { t , * } - T _ { * , h } m _ { t } \| _ { * } ^ { 2 } } } \\ & { } & { + ( 1 + \eta ) ( 1 + \zeta ^ { - 1 } ) \| T _ { * , h } \mathsf { L } _ { h } \dot { m } _ { t } \| _ { * } ^ { 2 } } \\ & { } & { + ( 1 + \eta ^ { - 1 } ) \| T _ { * , h } r _ { h } \| _ { * } ^ { 2 } + \frac { \kappa _ { h } \sigma _ { t } ^ { 2 } } { h } d _ { * , h } . } \end{array}\tag{50}
$$

Hence $L _ { * , h } ^ { 0 , \mathrm { u n c l i p } } \leq U _ { * , h } ^ { \mathrm { f i x e d } } .$

The boosting operator $T _ { h }$ filters the drift in (47). Consider a single fixed tree with $\mathsf { W } _ { 1 } = I$ and $\nu = 1$ . Then $\begin{array} { r } { d _ { h } ^ { \mathrm { b o o s t } } = \sum _ { \ell } [ n _ { \ell } / ( n _ { \ell } + \lambda _ { 1 } ) ] ^ { 2 } } \end{array}$ , where $n _ { \ell }$ is the number of training rows in leaf ℓ. The leaf penalty in (41) acts on the unnormalized squared-loss scale. It difers from the normalized ridge objective (21). Suppose we hold $\lambda _ { 1 }$ fixed and $n _ { \ell }$ grows in proportion to $h .$ . Then each shrinkage factor $n _ { \ell } / ( n _ { \ell } + \lambda _ { 1 } )$ tends to one as $h$ grows. Short histories thus receive more shrinkage, which makes boosting less aggressive there. Consequently, $T _ { h }$ and $d _ { h } ^ { \mathrm { b o o s t } }$ change with the memory. If we scale the penalty as $\lambda _ { k } = h \lambda _ { k }$ , the relative penalty stays stable once the leaf proportions stabilize. This scaling is an additional condition. It does not follow from a fixed tree topology. For several trees, $T _ { h }$ need not be symmetric. Hence the relevant variance dimension is t $\mathrm { r } ( T _ { h } T _ { h } ^ { \top } )$ , not a ridge eigenvalue sum. Suppose $d _ { h } ^ { \mathrm { b o o s t } }$ and $D _ { h , \mathrm { b o o s t } } ^ { 2 }$ stay stable over the candidate windows, and we control the other terms. Then the memory-dependent part of the bound has the form $V _ { \mathrm { b o o s t } } / h + K _ { \mathrm { b o o s t } } h + W _ { \mathrm { b o o s t } } h ^ { 2 }$ . If $\mu _ { h } ^ { 2 }$ is the only source of time dependence, then $K _ { \mathrm { b o o s t } } = 2 W _ { \mathrm { b o o s t } }$ , and the positive continuous minimizer solves

$$
2 W _ { \mathrm { b o o s t } } h ^ { 3 } + K _ { \mathrm { b o o s t } } h ^ { 2 } - V _ { \mathrm { b o o s t } } = 0 .\tag{51}
$$

This result does not cover outcome-dependent greedy splits, $\ell _ { 1 }$ leaf penalties, or an outcome-dependent initial prediction. A fully retrained XGBoost model has all three features. Analyzing them needs additional assumptions. We treat greedy splits in Section A.4.3.

## A.4.2 Spectral phase boundaries for fixed feature maps

The bounds above compare the memory-dependent costs in aggregate. We now compare models coordinate by coordinate in an orthogonal fixed-feature setting. This refines Proposition 2 of Section 3.2 for the surrogate. Let $\psi _ { 1 } , \ldots , \psi _ { p }$ be features that we fix before we use the readout outcomes. Let V be a generic input drawn from the current input distribution $P _ { t }$ , and write $\mathrm { E } _ { t }$ for expectation under $P _ { t }$ . We assume that the features are orthogonal under $P _ { t } \colon \mathrm { E } _ { t } [ \psi _ { i } \psi _ { k } ] = s _ { i } { \bf 1 } \{ i = k \}$ with $s _ { i } ~ > ~ 0$ . We write the current conditional mean as $\begin{array} { r } { m _ { t } = m _ { \perp , t } + \sum _ { i = 1 } ^ { p } \theta _ { i , t } \psi _ { i } } \end{array}$ . Here $\theta _ { i , t }$ is the coeficient of mode $i ,$ and $m _ { \perp , t }$ is orthogonal to every $\psi _ { i }$ . The prediction norm is $\| f \| _ { t } ^ { 2 } = \operatorname { E } _ { V \sim P _ { t } } [ f ( V ) ^ { 2 } ]$ . We assume that the training normal equations are diagonal with the same $s _ { i }$ . Let $\dot { \theta } _ { i , \cdot }$ <sub>t</sub> be the per-period drift of $\theta _ { i , t }$ . We define the training score of mode i and assume its conditional moments:

$$
\begin{array} { r l r } { \displaystyle } & { } & { U _ { i , h } = \frac { 1 } { h } \sum _ { j = 1 } ^ { h } \psi _ { i } ( v _ { t - j } ) y _ { t - j } , } \\ { \displaystyle } & { } & { \displaystyle \mathrm { E } [ U _ { i , h } \mid \mathcal { G } _ { h } ] = s _ { i } ( \theta _ { i , t } - \mu _ { h } \dot { \theta } _ { i , t } ) , } \\ { \displaystyle } & { } & { \mathrm { V a r } ( U _ { i , h } \mid \mathcal { G } _ { h } ) = \frac { \kappa \sigma _ { t } ^ { 2 } s _ { i } } { h } , \qquad \mu _ { h } = \frac { h + 1 } { 2 } . } \end{array}\tag{52}
$$

These assumptions encode four conditions: locally linear coeficient drift, temporal balance of the design, an exact coordinate score variance, and zero conditional score bias given the whole design. The last condition is strict score exogeneity. A long-run variance bound does not imply it. All four conditions are stronger than the inequalities that we used for the frozen-feature-map and boosting bounds.

For a subset I of modes, we fit the ridge coeficients $\widehat { \theta _ { i } } = U _ { i , h } / ( s _ { i } + \lambda )$ with $\lambda \geq 0$ for $i \in I ,$ and we set all other coeficients to zero. Put $\omega _ { i } = s _ { i } / { \left( s _ { i } + \lambda \right) }$ , the ridge gain of mode i. Put $b _ { i , h } = ( 1 - \omega _ { i } ) \theta _ { i , t } + \mu _ { h } \omega _ { i } { \dot { \theta } } _ { i , t }$ , the bias of coordinate i.

Proposition A.9 (exact spectral complexity boundary). Assume (52). The conditional squared error of the fitted forecast for the current conditional mean, excluding irreducible future noise, is

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { I } ( h ) = \| m _ { \perp , t } \| _ { t } ^ { 2 } + \sum _ { i \notin I } s _ { i } \theta _ { i , t } ^ { 2 } } \\ { \displaystyle + \sum _ { i \in I } \left[ s _ { i } b _ { i , h } ^ { 2 } + \frac { \kappa \sigma _ { t } ^ { 2 } } { h } \omega _ { i } ^ { 2 } \right] . } \end{array}\tag{53}
$$

Consider nested mode sets $I _ { S } \subset I _ { F }$ that we fit with the same $\lambda ,$ and let $\mathcal { N } = I _ { F } \ \backslash \ I _ { S }$ be the added modes. The richer set $I _ { F }$ has strictly smaller error if and only if

$$
\sum _ { i \in \mathcal { N } } s _ { i } \theta _ { i , t } ^ { 2 } > \sum _ { i \in \mathcal { N } } \left( s _ { i } b _ { i , h } ^ { 2 } + \frac { \kappa \sigma _ { t } ^ { 2 } } { h } \omega _ { i } ^ { 2 } \right) .\tag{54}
$$

Now fit all included modes without penalty, so that $\omega _ { i } = 1$ for $i \in \mathcal N$ . Define $\begin{array} { r } { d _ { N } = | \mathcal { N } | , C _ { N } = \sum _ { i \in \mathcal { N } } s _ { i } \theta _ { i , t } ^ { 2 } } \end{array}$ , and $\begin{array} { r } { Q = \sum _ { i \in \mathcal { N } } s _ { i } \dot { \theta } _ { i , t } ^ { 2 } } \end{array}$ . Then (54) reduces exactly to

$$
\boxed { C _ { \mathcal { N } } > \frac { \kappa \sigma _ { t } ^ { 2 } d _ { \mathcal { N } } } { h } + \mu _ { h } ^ { 2 } Q . }\tag{55}
$$

The left side $C _ { \mathcal { N } }$ is the approximation error that the added modes remove. The right side has two costs of fitting them. The estimation variance κ $\tau _ { t } ^ { 2 } d _ { \mathcal { N } } / h$ falls with $h .$ The drift bias $\mu _ { h } ^ { 2 } Q$ grows with h.

Proof. For an included mode, (52) gives $\begin{array} { r } { \mathrm { E } [ \widehat { \theta } _ { i } \ | \ \mathcal { G } _ { h } ] = \omega _ { i } ( \theta _ { i , t } - \mu _ { h } \dot { \theta } _ { i , t } ) } \end{array}$ and $s _ { i } \operatorname { V a r } ( \widehat { \theta _ { i } } \mid \mathcal { G } _ { h } ) = \kappa \sigma _ { t } ^ { 2 } \omega _ { i } ^ { 2 } / h$ . Orthogonality makes the squared prediction error a sum of coordinate errors, plus $\| m _ { \perp , t } \| _ { t } ^ { 2 }$ . An omitted mode contributes $s _ { i } \theta _ { i , t } ^ { 2 }$ . This proves (53). The common coordinates cancel in the nested comparison, which gives (54) and (55).

Corollary A.10 (exact interior-memory equation). Assume that $s _ { i } , \lambda , \kappa \sigma _ { t } ^ { 2 }$ , and the coeficient path in Proposition A.9 are the same across candidate histories. Define

$$
\begin{array} { l } { { \displaystyle V _ { I } = \kappa \sigma _ { t } ^ { 2 } \sum _ { i \in I } \omega _ { i } ^ { 2 } , } } \\ { { \displaystyle J _ { I } = \sum _ { i \in I } s _ { i } ( 1 - \omega _ { i } ) \omega _ { i } \theta _ { i , t } \dot { \theta } _ { i , t } , } } \\ { { \displaystyle Q _ { I } = \sum _ { i \in I } s _ { i } \omega _ { i } ^ { 2 } \dot { \theta } _ { i , t } ^ { 2 } . } } \end{array}\tag{56}
$$

The part of (53) that depends on h equals $V _ { I } / h + 2 \mu _ { h } J _ { I } + \mu _ { h } ^ { 2 } Q _ { I }$ . Assume $V _ { I } > 0$ and $Q _ { I } > 0$ . The continuous extension of this part has a unique positive minimizer. It solves

$$
\boxed { \frac { V _ { I } } { h ^ { 2 } } = J _ { I } + \frac { h + 1 } { 2 } Q _ { I } . }\tag{57}
$$

If we fit all modes in I without penalty, then $J _ { I } = 0$ and $\begin{array} { r } { Q _ { I } = \sum _ { i \in I } s _ { i } \dot { \theta } _ { i , t } ^ { 2 } } \end{array}$ . The minimizer then solves the cubic below, and it obeys the asymptotic form when it is large:

$$
\boxed { h ^ { 2 } ( h + 1 ) = \frac { 2 \kappa \sigma _ { t } ^ { 2 } | I | } { Q _ { I } } } , \qquad h ^ { * } \sim \left( \frac { 2 \kappa \sigma _ { t } ^ { 2 } | I | } { Q _ { I } } \right) ^ { 1 / 3 } .\tag{58}
$$

We now apply this result to the cost of the added modes. Suppose we fit the added modes in (55) without penalty, and suppose $d _ { \mathcal { N } } , Q > 0$ . Their complexity cost is $\kappa \sigma _ { t } ^ { 2 } d _ { \cal N } / h + \mu _ { h } ^ { 2 } Q$ . Its minimizer solves (58) with |I| replaced by $d _ { \mathcal { N } }$ and $Q _ { I }$ replaced by $Q .$ . The complex class wins on an interval of positive histories if and only if $C _ { \mathcal { N } }$ exceeds the minimum of this cost. If $Q = 0$ and $d _ { \mathcal { N } } , C _ { \mathcal { N } } > 0$ , the complex class wins for all $h > \kappa \sigma _ { t } ^ { 2 } d _ { \mathcal { N } } / C _ { \mathcal { N } }$ For integer histories, we must check the grid directly.

Proof. We expand the square in (53). Its h-dependent terms are $V _ { I } / h + 2 \mu _ { h } J _ { I } + \mu _ { h } ^ { 2 } Q _ { I }$ , up to a constant. Diferentiation gives $- V _ { I } / h ^ { 2 } + J _ { I } + ( h + 1 ) Q _ { I } / 2$ . Its derivative is $2 V _ { I } / h ^ { 3 } + Q _ { I } / 2 > 0$ , so it has exactly one zero, which is the minimum. The unpenalized formulas follow by setting $\omega _ { i } = 1$

The results above apply to boosted trees with fixed splits. Take one prechosen squared-loss tree with unpenalized leaf means. Refine it into a nested tree with more leaves. An orthogonal contrast basis for the refinement makes $d _ { \mathcal { N } }$ the number of added leaf contrasts. The quantity $C _ { \mathcal { N } }$ is the reduction in current approximation error, and $Q$ is the drift energy in these contrasts. Under the score and design conditions above, (55) is thus a boundary for fixed-partition XGBoost-type models.

For several fixed boosting trees, we use the general noncommuting operator bound of Proposition A.8. A spectral specialization needs three further conditions. First, $\mathsf { W } _ { k } = I ,$ so that every $P _ { k }$ is symmetric. Second, the $P _ { k }$ commute. Third, their common orthonormal eigenvectors equal the normalized training-sample evaluations of the modes $\psi _ { i }$ . Commutation alone does not identify these modes. Let $p _ { k , \ast }$ <sub>i</sub> be the joint eigenvalue of $P _ { k }$ for mode i. The boosting smoother then has the modal gains $\begin{array} { r } { \omega _ { i } ^ { \mathrm { b o o s t } } = 1 - \prod _ { k = 1 } ^ { N _ { \mathrm { t r e e } } } ( 1 - \nu p _ { k , i } ) } \end{array}$ . If the prediction norm and the score moments use this joint mode basis and satisfy (52), the same calculation of bias, drift, and variance applies to these modes.

The same results cover frozen neural encoders. For a frozen RNN or LSTM, we can orthogonalize the finalhidden-state features and use them as the $\psi _ { i }$ . For a frozen Transformer, we can orthogonalize the pooled-state or selected-token features and use them as the $\psi _ { i }$ . Attention may stay input-dependent. Each model family has its own $( s _ { i } , \theta _ { i , t } , \dot { \theta } _ { i , t } )$ . Hence each family has its own boundary (54) and its own optimum (57). The comparison covers nested sets of readout modes within one frozen family. It does not order diferent, nonnested encoders. We fix the input-context length of an encoder and vary only the training history h. Final ${ \mathrm { l y } } ,$ the cross term $J _ { I }$ explains why a simple cube-root rule is only approximate. Ridge shrinkage and partial boosting gains both give $0 < \omega _ { i } < 1$ , which can make $J _ { I } \neq 0$

## A.4.3 Transfer bounds for boosted trees and neural encoders

We now bound the gap between the surrogate and the trained model, as in Section 3.4. We prove Proposition 4 of Section 3.4 in two steps. First, we bound the risk gap without model-specific conditions (Proposition A.11). Second, we derive the phase boundary and the memory for trained models (Corollary A.16). Between these steps, we give suficient conditions for each model family.

Proposition A.11 (transfer under forecast perturbation). Let k index a model. Let $^ { g _ { k , h } }$ be its fixedfeature surrogate forecast and $f _ { k , h }$ its retrained forecast at memory h. We evaluate both against the same current conditional mean $m _ { t }$ in the evaluation norm $\| \cdot \| _ { t }$ of Section A.4.2. Let $\mathcal { G }$ be common pre-fit information. Define the surrogate risk, the retrained risk, and the squared perturbation by

$$
\begin{array} { r l } & { \mathcal { L } _ { k } ^ { 0 } ( h ) = \mathrm { E } [ \| g _ { k , h } - m _ { t } \| _ { t } ^ { 2 } \mid \mathcal { G } ] , } \\ & { \mathcal { L } _ { k } ( h ) = \mathrm { E } [ \| f _ { k , h } - m _ { t } \| _ { t } ^ { 2 } \mid \mathcal { G } ] , } \\ & { \quad \delta _ { k , h } ^ { 2 } = \mathrm { E } [ \| f _ { k , h } - g _ { k , h } \| _ { t } ^ { 2 } \mid \mathcal { G } ] . } \end{array}
$$

Then

$$
\begin{array} { r } { \left| \mathcal { L } _ { k } ( h ) - \mathcal { L } _ { k } ^ { 0 } ( h ) \right| \leq 2 \delta _ { k , h } \sqrt { \mathcal { L } _ { k } ^ { 0 } ( h ) } + \delta _ { k , h } ^ { 2 } = : \epsilon _ { k , h } . } \end{array}\tag{59}
$$

Consequently, if a surrogate ranking gap exceeds $\epsilon _ { 1 , h } + \epsilon _ { 0 , h }$ , the retrained forecasts have the same ranking sign. Next, consider the memory. Let $\mathcal { H } \subseteq [ 1 , \infty )$ be an admissible interval of histories. Suppose $| \mathcal { L } _ { k } ( h ) - F _ { 0 } ( h ) | \le \epsilon$ for all $h \in \mathcal { H } .$ , where $F _ { 0 } ( h ) = C _ { 0 } + V / \bar { h } + K h + W h ^ { 2 }$ with $V , W > 0$ . Let $h _ { \mathrm { a c t u a l } }$ minimize $\mathcal { L } _ { k }$ over $\mathcal { H } ,$ and let $h _ { 0 } ^ { * }$ minimize $F _ { 0 }$ over H. If $h _ { 0 } ^ { * }$ is interior, it is the unique positive root of $2 W h ^ { 3 } + K h ^ { 2 } - V = 0$ . Otherwise it is the corresponding endpoint of H. Then

$$
\lvert h _ { \mathrm { a c t u a l } } - h _ { 0 } ^ { * } \rvert \leq \sqrt { 2 \epsilon / W } .\tag{60}
$$

In Section 3.4, V, K, and W are the surrogate coeficients $V _ { k } , K _ { k } ,$ , and $W _ { k }$

Proof. We expand the diference of the squared norms and apply the Cauchy–Schwarz inequality. This proves (59). For the memory claim, we apply the uniform error bound twice. It shows that the surrogate risk $F _ { 0 }$ at $h _ { \mathrm { a c t u a l } }$ exceeds its minimum by at most 2ϵ. The surrogate has second derivative $2 V / h ^ { 3 } + 2 W \geq 2 W$ . Because $h _ { 0 } ^ { * }$ minimizes $F _ { 0 }$ over the interval H, strong convexity gives $F _ { 0 } ( h ) - F _ { 0 } ( h _ { 0 } ^ { * } ) \geq W ( h - h _ { 0 } ^ { * } ) ^ { 2 }$ for every $h \in \mathcal H$ . Hence $W ( h _ { \mathrm { a c t u a l } } - h _ { 0 } ^ { * } ) ^ { 2 } \leq 2 \epsilon$ , which is (60).

The transfer error $\epsilon _ { k , h }$ enters additively, and it grows linearly in $\delta _ { k , h }$ when $\delta _ { k , h }$ is small, because $\delta _ { k , h } ^ { 2 }$ is of second order. This transfer result is conditional. We do not show that $\delta _ { k , h }$ is small for XGBoost with outcome-selected splits. We also do not show it for recurrent or attention encoders whose parameters we train on the forecast loss. Before we can transfer the fixed-feature boundary to these algorithms, we need an independent bound on the change in splits, the movement of the features, or the efect of the training algorithm. We now state suficient conditions that bound the perturbation $\delta _ { k , h }$ of Proposition A.11. Each condition concerns a specific version of a training procedure. A model name alone does not imply any of them. Prior work gives learning bounds for regularized boosting, recurrent networks, and Transformers under diferent complexity restrictions (Cortes et al., 2019; Chen et al., 2020; Trauger and Tewari, 2024). We add explicit forecast-perturbation terms to the memory and phase comparisons of this appendix.

We start with greedy boosted trees, the first family in Section 3.4. We follow the split-and-leaf rules of Chen and Guestrin (2016). Consider squared-loss stagewise boosting with $N _ { \mathrm { t r e e } }$ greedy, axis-aligned trees. The model has a fixed learning rate, fixed row weights, $\ell _ { 2 }$ leaf penalties, a zero initial prediction, and no $\ell _ { 1 }$ penalty. Candidate thresholds lie on a prespecified finite grid, for example a fixed number of histogram bins per feature. Tree depth and $N _ { \mathrm { t r e e } }$ stay bounded across candidate histories. Hence the total number of candidate gains along a reference path has a uniform bound.

We define a reference tree sequence from population gains. Let $\mathcal { G } _ { h } ^ { 0 }$ contain the training and common evaluation designs, the candidate grid, the row weights, the penalty choices, and the conditional target law. It excludes the realized readout errors. Index the candidate gains at the reference decisions by s. At each reference decision, we compute every normalized empirical gain $\widehat { G } _ { s }$ from the random reference residual. This residual results from applying all earlier reference leaf updates to the training outcomes. We define the population gain $G _ { s } = \mathrm { E } [ \widehat { G } _ { s }$ $\mathcal { G } _ { h } ^ { 0 } ]$ . Hence later-stage population gains average over the noisy residuals that earlier reference updates generate. We choose recursively the candidate with the largest population gain, including the choice to stop. These choices define the reference topology, which is $\mathcal { G } _ { h } ^ { 0 } .$ -measurable. Let $\widetilde { f _ { h } }$ be the forecast that fits leaf updates on this topology. Let $f _ { h }$ be the forecast that uses greedy, outcome-dependent splits on the same grid. We normalize al gains by h. We express any minimum-split-loss threshold on this same scale, or we set it to zero.

Proposition A.12 (finite-grid greedy XGBoost with stable splits). Conditional on $\mathcal { G } _ { h } ^ { 0 }$ , assume the following.

(i) Margin. At each reference split, the best population gain exceeds every competing gain by at least $\Delta _ { \mathrm { s p l i t } } > 0$

(ii) Count. The reference decisions involve at most $N _ { \mathrm { g a i n } }$ candidate gains in total, including the stop choice.

(iii) Concentration. For each such gain and every $\vartheta > 0$

$$
\begin{array} { r } { \operatorname* { P r } \Bigl ( | \widehat { G } _ { s } - G _ { s } | \geq \vartheta \mid \mathcal { G } _ { h } ^ { 0 } \Bigr ) \leq 2 \exp ( - c h \vartheta ^ { 2 } / b _ { G } ^ { 2 } ) , } \end{array}\tag{61}
$$

where $c > 0$ and $b _ { G } > 0$ are constants that are common to all gains.

The event in (iii) covers every decision that the greedy algorithm reaches while it agrees with the reference path. We clip both final forecasts to $[ - B , B ]$ , and we assume that the current conditional mean on the fixed evaluation design lies in $[ - B , B ]$ . Let $L _ { * , h } ^ { 0 , \bar { \mathrm { c l i p } } }$ and $L _ { * , h } ^ { \mathrm { g r e e d y , c l i p } }$ be the conditional mean-squared risks of the clipped reference forecast and the clipped greedy forecast in the common norm $\| \cdot \| _ { * }$ of Section A.4.1. Then

$$
\begin{array} { r l } & { \left. L _ { * , h } ^ { \mathrm { g r e e d y , c l i p } } - L _ { * , h } ^ { 0 , \mathrm { c l i p } } \right. \le \epsilon _ { \mathrm { X G B } } ( h ) , } \\ & { \qquad \epsilon _ { \mathrm { X G B } } ( h ) : = 8 B ^ { 2 } N _ { \mathrm { g a i n } } \exp \left( - \frac { c h \Delta _ { \mathrm { s p l i t } } ^ { 2 } } { 4 b _ { G } ^ { 2 } } \right) . } \end{array}\tag{62}
$$

Clipping also contracts squared error to a target in $[ - B , B ]$ , so

$$
\begin{array} { r l } { L _ { * , h } ^ { \mathrm { g r e e d y , c l i p } } \leq L _ { * , h } ^ { 0 , \mathrm { c l i p } } + \epsilon _ { \mathrm { X G B } } ( h ) } & { } \\ { \leq U _ { * , h } ^ { \mathrm { f i x e d } } ( h ) + \epsilon _ { \mathrm { X G B } } ( h ) , } \end{array}\tag{63}
$$

where $U _ { * , h } ^ { \mathrm { f i x e d } }$ is the fixed-evaluation upper bound (50). The second inequality in (63) is one-sided, because clipping can only lower the reference risk.

Proof. By the margin condition, deviations below $\Delta _ { \mathrm { s p l i t } } / 2$ preserve the best split at every reference decision. Induction over nodes and boosting stages then makes the greedy and reference topologies, leaf updates, and forecasts identical. The union bound and (61) give

$$
\operatorname* { P r } ( f _ { h } \neq \widetilde { f } _ { h } \ \vert \ \mathcal { G } _ { h } ^ { 0 } ) \leq 2 N _ { \mathrm { g a i n } } \exp ( - c h \Delta _ { \mathrm { s p l i t } } ^ { 2 } / ( 4 b _ { G } ^ { 2 } ) ) .
$$

Clipped squared error against a target in $[ - B , B ]$ lies in $[ 0 , 4 B ^ { 2 } ]$ . Hence a disagreement changes the conditional risk by at most $4 B ^ { 2 }$ times this probability, in either direction. Clipping cannot increase the squared error to such a target. The bound (50) controls the unclipped reference risk. This proves both claims. We now study the memory implied by this bound. Suppose the stable part of $U _ { * , h } ^ { \mathrm { f i x e d } }$ equals $C _ { 0 } + V / h + K h + W h ^ { 2 }$ with $V , W > 0$ . Suppose also that the split-margin constants $B , N _ { \mathrm { g a i n } } , \Delta _ { \mathrm { s p l i t } } , c , b _ { G }$ do not change across candidate histories. Then the upper bound in (63) equals $C _ { 0 } + V / h + K h + { W h ^ { 2 } } + { C _ { \mathrm { s e l } } } { e ^ { - \alpha h } }$ , where $C _ { \mathrm { s e l } } = 8 B ^ { 2 } N _ { \mathrm { g a i n } }$ and $\alpha = c \Delta _ { \mathrm { s p l i t } } ^ { 2 } / ( 4 b _ { G } ^ { 2 } )$ . This function is strictly convex. Its unique continuous minimizer solves

$$
2 W h + K - \frac { V } { h ^ { 2 } } - \alpha C _ { \mathrm { s e l } } e ^ { - \alpha h } = 0\tag{64}
$$

and exceeds the positive root of $2 W h ^ { 3 } + K h ^ { 2 } - V = 0$ , because the derivative of the selection term is negative at that root. Thus a positive split-selection penalty moves the minimizer of this upper-bound surrogate toward longer memory. We do not claim that the minimizer of the actual greedy risk moves by the same amount. This is the boosted-tree statement of Section 3.4, where $\alpha \propto \Delta _ { \mathrm { s p l i t } } ^ { 2 }$

The result has limits. The gain-tail condition (61) is an assumption. It does not follow from the XGBoost objective. Bounded gains, a minimum leaf occupancy, and a suitable dependence condition may supply it, but we do not prove this. Exact-greedy trees that consider every observed threshold behave diferently. Their candidate count generally grows with $h ,$ and adjacent gains have vanishing margins. For them, $\epsilon _ { \mathrm { X G B } } ( h )$ need not decay, and the constant-parameter memory surrogate fails without new h-dependent control. To extend the local-design bound to next-period inputs, we need the fixed-evaluation operator (48) and corresponding design assumptions.

We next turn to recurrent networks, the second family in Section 3.4. Consider an input sequence $x _ { 1 } , \ldots , x _ { L }$ of fixed length $L$ . We fix the reference encoder using independent or earlier outcomes. Its hidden states are $s _ { \ell } ^ { 0 } = \sigma ( W _ { 0 } s _ { \ell - 1 } ^ { 0 } + U _ { 0 } x _ { \ell } )$ . The trained encoder has hidden states $s _ { \ell } = \sigma ( W s _ { \ell - 1 } + U x _ { \ell } )$ . Both encoders start at zero. Here $x _ { \ell }$ is the input at step $\ell , W _ { 0 }$ and $W$ are the recurrent weight matrices, and $U _ { 0 }$ and $U$ are the input weight matrices. The function $\sigma$ is the activation. We write $\| \cdot \| _ { \mathrm { o p } }$ for the operator norm. Suppose the activation $\sigma$ is 1-Lipschitz and $\sigma ( 0 ) = 0$

Proposition A.13 (contractive trained RNN). Assume the following bounds. They hold almost surely given the common pre-fit sigma-field G, uniformly over evaluation inputs. Their upper bounds are G-measurable.

$$
\begin{array} { c } { \left\| x _ { \ell } \right\| \leq B _ { x } , } \\ { \left\| W _ { 0 } \right\| _ { \mathrm { o p } } , \left\| W \right\| _ { \mathrm { o p } } \leq \chi _ { \mathrm { R N N } } < 1 , } \\ { \left\| U _ { 0 } \right\| _ { \mathrm { o p } } , \left\| U \right\| _ { \mathrm { o p } } \leq b , } \\ { \left\| W - W _ { 0 } \right\| _ { \mathrm { o p } } \leq e _ { W } , } \\ { \left\| U - U _ { 0 } \right\| _ { \mathrm { o p } } \leq e _ { U } . } \end{array}
$$

These bounds control the input norm $( B _ { x } )$ , the recurrent gain $\left( \chi _ { \mathrm { R N N } } \right)$ , the input weights (b), and the movement of the two weight matrices $( e _ { W }$ and $e _ { U } )$ . Let the reference forecast be $g _ { h } ( v ) = a _ { 0 } ^ { \top } s _ { L } ^ { 0 } ( v )$ , where v is the input sequence and $a _ { 0 }$ is the ridge readout fitted on the frozen features. Let the trained forecast be $f _ { h } ( v ) = a ^ { \top } s _ { \cal L } ( v )$ with readout a. Assume almost surely given G that $\| a - a _ { 0 } \| \le e _ { a }$ and $\| a \| \le R _ { \mathrm { o u t } }$ , with G-measurable upper bounds. Set Γ<sub>RNN</sub> $, \angle = ( 1 - \chi _ { \mathrm { R N N } } ^ { L } ) / ( 1 - \chi _ { \mathrm { R N N } } )$ and $R _ { L } = b B _ { x } \Gamma _ { \mathrm { R N N } , L }$ . Then, uniformly in $v ,$ ,

$$
\begin{array} { r l } { | f _ { h } ( v ) - g _ { h } ( v ) | \le \delta _ { \mathrm { R N N } } , } & { } \\ { \delta _ { \mathrm { R N N } } : = e _ { a } R _ { L } } & { } \\ { + \ : R _ { \mathrm { o u t } } \Gamma _ { \mathrm { R N N } , L } ( e _ { W } R _ { L } + e _ { U } B _ { x } ) . } \end{array}\tag{65}
$$

Let $U _ { \mathrm { R N N } } ^ { 0 } ( h )$ be the bound on the conditional risk of the reference forecast from Proposition A.6, including any approximation residual via (39). Then the conditional risk $\mathcal { L } _ { \mathrm { R N N } } ( h )$ of the trained forecast obeys

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R N N } } ( h ) \leq U _ { \mathrm { R N N } } ^ { 0 } ( h ) + 2 \delta _ { \mathrm { R N N } } \sqrt { U _ { \mathrm { R N N } } ^ { 0 } ( h ) + \delta _ { \mathrm { R N N } } ^ { 2 } } . } \end{array}\tag{66}
$$

Proof. The reference state obeys $\lVert s _ { \ell } ^ { 0 } \rVert \leq \chi _ { \mathrm { R N N } } \lVert s _ { \ell - 1 } ^ { 0 } \rVert + b B _ { x }$ , hence $\| s _ { \ell } ^ { 0 } \| \le b B _ { x } \Gamma _ { \mathrm { R N N } , \ell } \le R _ { L }$ . The state diference satisfies

$$
\begin{array} { r } { \| s _ { \ell } - s _ { \ell } ^ { 0 } \| \leq \chi _ { \mathrm { R N N } } \| s _ { \ell - 1 } - s _ { \ell - 1 } ^ { 0 } \| + e _ { W } R _ { L } + e _ { U } B _ { x } . } \end{array}
$$

Iteration gives $\begin{array} { r } { \| s _ { L } - s _ { L } ^ { 0 } \| \leq \Gamma _ { \mathrm { R N N } , L } \big ( e _ { W } R _ { L } + e _ { U } B _ { x } \big ) } \end{array}$ . We add and subtract $a ^ { \top } s _ { L } ^ { 0 }$ to obtain (65). The pointwise bound controls the conditional $L _ { 2 }$ perturbation in Proposition A.11. This gives (66). ■

The contraction $\chi _ { \mathrm { R N N } } < 1$ keeps the perturbation bounded as L grows, because $\Gamma _ { \mathrm { R N N } , L } \leq ( 1 - \chi _ { \mathrm { R N N } } ) ^ { - 1 }$ . The result stays conditional. It needs bounds on the movement of the trained weights and readout. We do not claim that gradient descent supplies such bounds. This is the RNN statement of Section 3.4, where we write $\chi$ for χ<sub>RNN</sub>.

We turn to LSTMs, the third family in Section 3.4. Consider two LSTM encoders on the same inputs $x _ { 1 } , \ldots , x _ { L }$ We fix the reference encoder using independent or earlier outcomes. The trained encoder has all its parameters trained on the forecast loss. For either encoder, let $s _ { \ell } \in \mathrm { R } ^ { p }$ be the hidden state and $c _ { \ell } \in \mathrm { R } ^ { p }$ the cell state. Write $\boldsymbol { z } _ { \ell } = ( s _ { \ell } , c _ { \ell } )$ for the augmented state. The updates are

$$
c _ { \ell } = f _ { \ell } \odot c _ { \ell - 1 } + i _ { \ell } \odot \widetilde { c } _ { \ell } , \qquad s _ { \ell } = o _ { \ell } \odot \operatorname { t a n h } ( c _ { \ell } ) ,
$$

where $\odot$ is the elementwise product, $f _ { \ell } , i _ { \ell } ,$ , and $o _ { \ell }$ are the forget, input, and output gates, and $\widetilde { c } _ { \ell }$ is the candidate cell state. The sigmoid gates lie in $[ 0 , 1 ] ^ { p }$ , and $\widetilde c _ { \ell } \in [ - 1 , 1 ] ^ { p }$ . Both encoders start at zero. We assume that the forget gates satisfy $\| f _ { \ell } \| _ { \infty } \le \rho _ { f } < 1$

Proposition A.14 (trained LSTM with controlled gates). Compare the two encoders at any common admissible state and input. Assume that the forget, input, candidate, and output gate outputs of the two encoders difer by at most $e _ { f } , e _ { i } , e _ { \mathrm { c a n d } } , e _ { o } $ respectively, in Euclidean norm. Assume also that the one-step map of the trained encoder, which sends $z _ { \ell - 1 }$ to $z _ { \ell }$ for a given input, is ϱ-Lipschitz in its state argument, with finite $\varrho \ge 0$ . These movement and Lipschitz bounds hold almost surely given $\mathcal { G } _ { : }$ , uniformly over evaluation inputs. Their upper bounds are G-measurable. Let $g _ { h } ( v ) = a _ { 0 } ^ { \top } s _ { L } ^ { 0 } ( v )$ use the frozen ridge readout, and let $f _ { h } ( v ) = a ^ { \top } s _ { \cal L } ( v )$ use the trained readout. Assume $\| a - a _ { 0 } \| \le e _ { a }$ and $\| a \| \le R _ { \mathrm { o u t } }$ almost surely given ${ \mathcal { G } } .$ with G-measurable upper bounds. Put

$$
\begin{array} { c } { { C _ { f } = ( 1 - \rho _ { f } ) ^ { - 1 } , } } \\ { { K _ { \mathrm { g a t e } } = 2 ( C _ { f } e _ { f } + e _ { i } + e _ { \mathrm { c a n d } } ) + e _ { o } , } } \\ { { { } } } \\ { { \Gamma _ { \mathrm { L S T M } , L } = \displaystyle \sum _ { k = 0 } ^ { L - 1 } \varrho ^ { k } . } } \end{array}
$$

Then

$$
\begin{array} { r l } & { | f _ { h } ( v ) - g _ { h } ( v ) | \leq \delta _ { \mathrm { L S T M } } , } \\ & { \qquad \delta _ { \mathrm { L S T M } } : = e _ { a } \sqrt { p } + R _ { \mathrm { o u t } } \Gamma _ { \mathrm { L S T M } , L } K _ { \mathrm { g a t e } } . } \end{array}\tag{67}
$$

Let $U _ { \mathrm { L S T M } } ^ { 0 } ( h )$ be the risk bound for the frozen readout from Proposition A.6 and (39). Consequently, the risk of the trained forecast is at most

$$
U _ { \mathrm { L S T M } } ^ { 0 } ( h ) + 2 \delta _ { \mathrm { L S T M } } \sqrt { U _ { \mathrm { L S T M } } ^ { 0 } ( h ) } + \delta _ { \mathrm { L S T M } } ^ { 2 } .\tag{68}
$$

Proof. The forget-gate condition and the bounded input gate and candidate give $\| c _ { \ell } \| _ { \infty } \leq C _ { f }$ . In addition, $\| s \ell \| \leq \sqrt { p }$ . At a common state, perturbing the gates changes the cell update by at most $E _ { c } = C _ { f } e _ { f } + e _ { i } + e _ { \mathrm { c a n d } } .$ Since tanh is 1-Lipschitz, the hidden update changes by at most $e _ { o } + E _ { c }$ . The Euclidean change in the augmented state is thus at most $2 E _ { c } + e _ { o } = K _ { \mathrm { g a t e } }$ . We insert and subtract one full state update at the same state and obtain

$$
\begin{array} { r } { \| z _ { \ell } - z _ { \ell } ^ { 0 } \| \le \varrho \| z _ { \ell - 1 } - z _ { \ell - 1 } ^ { 0 } \| + K _ { \mathrm { g a t e } } . } \end{array}
$$

We iterate this inequality, and then we add and subtract $a ^ { \top } s _ { L } ^ { 0 }$ to get (67). We then apply Proposition A.11 to obtain (68).

The forget-gate bound controls the size of the cell state. It does not by itself imply a contraction of the full state, which the special case $\varrho < 1$ requires. We must verify the full-state Lipschitz constant from all recurrent and gate weights. Alternatively, we can replace it by a Jacobian product of finite length L. If $\varrho \ge 1$ , the factor Γ<sub>LSTM,L</sub> and hence the bound can grow rapidly with the context length. This is the LSTM statement of Section 3.4, where ϱ is the Lipschitz constant of the state update.

We finish with attention, the fourth family in Section 3.4. Consider a one-block Transformer readout with $N _ { \mathrm { h e a d } }$ attention heads and a fixed input-context length $L .$ . We fix its reference attention map using independent or earlier outcomes. For head r and token $\ell ,$ let $\xi _ { r \ell } ^ { 0 }$ and $v _ { r \ell } ^ { 0 }$ be the reference attention logit and value vector. Let $\xi _ { r \ell }$ and $v _ { r \ell }$ be their trained counterparts. The value vector $v _ { r \ell }$ carries two subscripts, and the input v carries none. Define the head outputs

$$
z _ { r } ^ { 0 } = \sum _ { \ell = 1 } ^ { L } \mathrm { s o f t m a x } ( \xi _ { r } ^ { 0 } ) _ { \ell } v _ { r \ell } ^ { 0 } , \quad z _ { r } = \sum _ { \ell = 1 } ^ { L } \mathrm { s o f t m a x } ( \xi _ { r } ) _ { \ell } v _ { r \ell } .
$$

Proposition A.15 (trained multihead attention block). Assume the following bounds. They hold almost surely given ${ \mathcal { G } } ,$ uniformly over heads, tokens, and evaluation inputs. Their upper bounds are G-measurable.

$$
\| v _ { r \ell } ^ { 0 } \| \leq B _ { v } , \quad \| v _ { r \ell } - v _ { r \ell } ^ { 0 } \| \leq e _ { v } , \quad | \xi _ { r \ell } - \xi _ { r \ell } ^ { 0 } | \leq e _ { \xi } .
$$

Let $\varphi ^ { 0 } = O _ { 0 } [ z _ { 1 } ^ { 0 } ; \dots ; z _ { N _ { \mathrm { h e a d } } } ^ { 0 } ]$ and $\varphi = O [ z _ { 1 } ; \dots ; z _ { N _ { \mathrm { h e a d } } } ]$ be the feature vectors. Here $O _ { 0 }$ and $O$ are the output projection matrices of the reference and trained blocks, and $[ \cdot ; \cdot ]$ stacks the head outputs. Assume $\| O _ { 0 } \| _ { \mathrm { o p } } , \| O \| _ { \mathrm { o p } } \leq N _ { O }$ and $\| O - O _ { 0 } \| _ { \mathrm { o p } } ~ \le ~ e _ { O }$ almost surely given ${ \mathcal { G } } _ { : }$ , with G-measurable upper bounds. The frozen forecast is $g _ { h } ( v ) = a _ { 0 } ^ { \top } \varphi ^ { 0 } ( v )$ with ridge readout $a _ { 0 }$ . The trained forecast is $f _ { h } ( v ) = a ^ { \top } \varphi ( v )$ . Assume $\| a - a _ { 0 } \| \le e _ { a }$ and $\| a \| \leq R _ { \mathrm { o u t } }$ almost surely given ${ \mathcal { G } } _ { : }$ with G-measurable upper bounds. Then

$$
\begin{array} { r l } & { | f _ { h } ( v ) - g _ { h } ( v ) | \leq \delta _ { \mathrm { T r } } , } \\ & { \qquad \delta _ { \mathrm { T r } } : = e _ { a } N _ { O } \sqrt { N _ { \mathrm { h e a d } } } B _ { v } } \\ & { \qquad + R _ { \mathrm { o u t } } \sqrt { N _ { \mathrm { h e a d } } } e _ { O } B _ { v } } \\ & { \qquad + R _ { \mathrm { o u t } } \sqrt { N _ { \mathrm { h e a d } } } N _ { O } ( e _ { v } + B _ { v } e _ { \xi } ) . } \end{array}\tag{69}
$$

Let $U _ { \mathrm { T r } } ^ { 0 } ( h )$ be the risk bound for the frozen attention map from Proposition A.6 and (39). Then the risk of the trained one-block model is at most

$$
U _ { \mathrm { T r } } ^ { 0 } ( h ) + 2 \delta _ { \mathrm { T r } } \sqrt { U _ { \mathrm { T r } } ^ { 0 } ( h ) + \delta _ { \mathrm { T r } } ^ { 2 } } .\tag{70}
$$

The perturbation bound has no explicit factor of $L .$ It does need a uniform bound on the logit and value perturbations over all L tokens.

Proof. Let π be a probability vector on the L tokens, and let ${ \cal D } _ { \pi } ^ { \mathrm { s m } } = \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top }$ be the softmax derivative. For any vector $w \in \bar { \mathbf { R } } ^ { L }$ , we have $\| D _ { \pi } ^ { \mathrm { s m } } w \| _ { 1 } = \mathrm { E } _ { \pi } | w - \mathrm { E } _ { \pi } w | \leq \| \dot { w } \| _ { \infty }$ . This follows from the Cauchy–Schwarz inequality and $\operatorname { V a r } _ { \pi } ( w ) \leq \| w \| _ { \infty } ^ { 2 }$ . We integrate along the line segment between the two logit vectors and obtain

$$
\begin{array} { r } { \| \operatorname { s o f t m a x } ( \xi ) - \operatorname { s o f t m a x } ( \xi ^ { 0 } ) \| _ { 1 } \leq \| \xi - \xi ^ { 0 } \| _ { \infty } . } \end{array}
$$

Hence $\| z _ { r } ^ { 0 } \| \leq B _ { v }$ and $\| z _ { r } - z _ { r } ^ { 0 } \| \leq e _ { v } + B _ { v } e _ { \xi }$ . Stacking the heads multiplies these Euclidean bounds by at most $\sqrt { N _ { \mathrm { h e a d } } }$ . We perturb O first and the readout second, which proves (69). We then apply Proposition A.11 to obtain (70). 厂

Consider dot-product logits $\xi _ { r \ell } = q _ { r } ^ { \top } k _ { r \ell } / \sqrt { d }$ with query $q _ { r }$ and keys $k _ { r \ell } \in \mathrm { R } ^ { d }$ . Suppose the query and key are linear images of tokens with norm at most $B _ { x }$ . Suppose also that the query and key matrices of both the trained and reference blocks have operator norm at most $N _ { Q K }$ . If the query matrices difer by at most $e _ { Q }$ and the key matrices by at most $e _ { K }$ in operator norm, then $e _ { \xi } \le B _ { x } ^ { 2 } N _ { Q K } ( e _ { Q } + e _ { K } ) / \sqrt { d } .$ . If the value matrices difer by at most $e _ { V }$ in operator norm, then $e _ { v } \leq B _ { x } e _ { V }$ . For additional Transformer blocks, feedforward layers, and residual paths, we can compose their bounded Lipschitz and perturbation constants. The bound (69) itself holds for one block only. This is the Transformer statement of Section 3.4.

We can combine each of Propositions A.12–A.15 with Proposition A.11 in two ways. First, suppose a fixedsurrogate complexity margin is strict and exceeds the two relevant transfer errors. Then the model ranking does not change. Second, suppose the transfer error is uniform over the candidate histories and the fixed-surrogate risk has the strongly convex memory form above. Then (60) localizes the retrained optimum. Both implications are conditional on the model-specific constants above. We do not claim that an arbitrary trained model satisfies them.

Corollary A.16 (model-conditional phase boundary and memory). Fix one common evaluation target, one design or distribution, one prediction norm, and one pre-fit sigma-field G for all models and histories. If XGBoost enters the comparison, we take $\| \cdot \| ,$ <sub>∗</sub> on the fixed evaluation design as the prediction norm of every model. Let $\mathcal { L } _ { k } ( h )$ be the risk of the fully trained model $k ,$ and let $\mathcal { L } _ { k } ^ { 0 } ( h )$ be the risk of its fixed surrogate. Define the transfer error $\epsilon _ { k } ( h )$ as follows. For a finite-grid greedy boosted-tree model that satisfies Proposition A.12, set

$$
\epsilon _ { \mathrm { X G B } } ( h ) = 8 B ^ { 2 } N _ { \mathrm { g a i n } } \exp ( - c h \Delta _ { \mathrm { s p l i t } } ^ { 2 } / ( 4 b _ { G } ^ { 2 } ) ) .
$$

For a trained RNN, LSTM, or one-block Transformer, set, respectively,

$$
\epsilon _ { k } ( h ) = 2 \delta _ { k } \sqrt { U _ { k } ^ { 0 } ( h ) } + \delta _ { k } ^ { 2 } ,
$$

with $\delta _ { k }$ given by (65), (67), or (69), and with $U _ { k } ^ { 0 }$ the surrogate risk bound of the corresponding proposition. We assume that the gain-tail and movement bounds that define $\epsilon _ { k }$ also hold given this common ${ \mathcal { G } } .$ For XGBoost, we assume that the fixed-evaluation moment and gain-tail conditions hold with $\mathcal { G } _ { h } = \mathcal { G } _ { h } ^ { 0 } = \mathcal { G }$ for every candidate h. We also assume that the fixed-surrogate risks of classes $\mathcal { M } _ { 0 }$ and $\mathcal { M } _ { 1 }$ have expansions of the form (13), with $e _ { k }$ the surrogate remainder bounded by $r _ { k } ( h )$ . For XGBoost, the fixed-surrogate risk is the clipped reference risk $L _ { * , h } ^ { 0 , \mathrm { c l i p } }$ of (62). The unclipped risk (49) gives only a one-sided upper bound after clipping. Hence we cannot derive the two-sided expansion of $L _ { * , h } ^ { 0 , \mathrm { c l i p } }$ that we need from Proposition A.8. For XGBoost, this expansion is an additional assumption.

(a) The fully trained complex model $\mathcal { M } _ { 1 }$ has smaller risk than the fully trained simple model $\mathcal { M } _ { 0 }$ whenever

$$
\begin{array} { c } { \displaystyle { \Delta _ { 0 } > \frac { \Delta V } { h } + \Delta K h + \Delta W h ^ { 2 } } } \\ { \displaystyle { + r _ { 1 } ( h ) + r _ { 0 } ( h ) + \epsilon _ { 1 } ( h ) + \epsilon _ { 0 } ( h ) . } } \end{array}\tag{71}
$$

(b) The reverse ranking holds when $\Delta _ { 0 }$ is below the same leading boundary minus all four remainder terms, that is, when $\Delta _ { 0 } < \Delta V / h + \Delta K h + \Delta W h ^ { 2 } - r _ { 1 } ( h ) - r _ { 0 } ( h ) - \epsilon _ { 1 } ( h ) - \epsilon _ { 0 } ( h )$

(c) Fix one model k. Suppose $r _ { k } ( h ) + \epsilon _ { k } ( h ) \leq \bar { \epsilon } _ { k }$ uniformly over a continuous admissible history interval, that $V _ { k } , W _ { k } > 0$ , and that both minimizers below lie in this interval. Let $h _ { k } ^ { \mathrm { t r a i n e d } }$ minimize $\mathcal { L } _ { k } ( h )$ , and let $h _ { k } ^ { 0 * }$

minimize the surrogate expansion $\mathcal { A } _ { k } + B _ { k } + V _ { k } / h + K _ { k } h + W _ { k } h ^ { 2 }$ . Then

$$
\begin{array} { c } { { \vert h _ { k } ^ { \mathrm { t r a i n e d } } - h _ { k } ^ { 0 * } \vert \le \sqrt { 2 \bar { \epsilon } _ { k } / W _ { k } } , } } \\ { { 2 W _ { k } ( h _ { k } ^ { 0 * } ) ^ { 3 } + K _ { k } ( h _ { k } ^ { 0 * } ) ^ { 2 } - V _ { k } = 0 } } \end{array}\tag{72}
$$

if the surrogate minimum is interior. At a boundary, $h _ { k } ^ { 0 * }$ is the constrained minimum instead.

Because we keep the known linear term $K _ { k } h$ explicit, $\bar { \epsilon } _ { k }$ need not absorb it. The localization bound is informative only if the remaining uniform error is small on the chosen interval. Items (a) and (b) are the phase-boundary statement of Proposition 4 in Section 3.4, and item (c) is its memory statement.

Proof. We combine the transfer bound (62) for XGBoost with Propositions A.13–A.15 for the neural models. This gives $| \mathcal { L } _ { k } ( h ) - \mathcal { L } _ { k } ^ { 0 } ( h ) | \leq \epsilon _ { k } ( h )$ in the common evaluation norm. For the neural cases, $\epsilon _ { k }$ increases with $\mathcal { L } _ { k } ^ { 0 } ,$ so we can replace $\mathcal { L } _ { k } ^ { 0 }$ by its bound $U _ { k } ^ { 0 }$ . We then subtract the two fixed-surrogate expansions. Bounding their remainders and the transfer errors gives (71). For the reverse ranking in (b), we repeat the steps with the signs reversed. For (c), the trained risk difers from $\mathcal { A } _ { k } + B _ { k } + V _ { k } / h + K _ { k } h + W _ { k } h ^ { 2 }$ by at most $\bar { \epsilon } _ { k } .$ , uniformly on the interval. The strong-convexity argument in the proof of Proposition A.11 then gives (72). Together with Proposition A.11, this proves Proposition 4 of Section 3.4.

## B CONTROLLED SIMULATION RESULTS

This appendix gives the design and results for Section 4.1; Figure 2 presents the main-text gradual-drift comparison.

## B.1 Simulation method

Covariates and signal. Each observation has $p = 8$ covariates $x \in \mathbb { R } ^ { p }$ drawn from $N ( 0 , \Sigma )$ . The covariance is $\Sigma = \operatorname { d i a g } ( \kappa _ { 1 } , . . . , \kappa _ { p } )$ with $\kappa _ { k } = q ^ { k - 1 }$ . The decay parameter $q \in \{ 1 , 0 . 4 5 \}$ } sets the spectrum. We call $q = 1$ the flat spectrum and $q = 0 . 4 5$ the decaying spectrum. The conditional mean at drift state $\beta$ is

$$
m _ { \beta } ( x ) = \sum _ { k = 1 } ^ { p } \beta _ { k } x _ { k } + \gamma \frac { x _ { 1 } x _ { 2 } } { \sqrt { \kappa _ { 1 } \kappa _ { 2 } } } .
$$

The nonlinear strength $\gamma \in \{ 0 , 0 . 4 , 0 . 8 \}$ scales a standardized product feature with unit variance. The baseline coeficient vector $\beta ^ { 0 }$ has a single nonzero entry, $\beta _ { 2 } ^ { 0 } = 0 . 7 / \sqrt { \kappa _ { 2 } }$ , so the linear signal has variance $0 . 7 ^ { 2 }$ for every q. The response is $y = m _ { \beta } ( x ) + \varepsilon$ with $\varepsilon \sim N ( 0 , s ^ { 2 } )$ . The signal-to-noise ratio is $\bar { \mathrm { S N R } } = ( 0 . 7 ^ { 2 } + \gamma ^ { 2 } ) / s ^ { 2 }$ , which fixes the noise standard deviation $s = \sqrt { ( 0 . 7 ^ { 2 } + \gamma ^ { 2 } ) } ,$ /SNR for $\mathrm { S N R } \in \{ 0 . 2 5 , 2 \}$

Drift. A training observation of age $a \in \{ 1 , \ldots , 2 5 6 \}$ was recorded a observations before the prediction date. Its coeficient vector equals $\beta ^ { 0 }$ except in one drifting coordinate j (coordinate 1 in the factorial grid). Under gradual drift, $\beta _ { j } = \beta _ { j } ^ { 0 } - \rho a$ with rate $\rho = 0 . 0 0 8$ . Under an abrupt break, $\beta _ { j } = \beta _ { j } ^ { 0 } - 0 . 6 5 \cdot 1 \{ a > 4 8 \}$ . Under zero drift, $\beta = \beta ^ { 0 }$ . The test targets use $\beta ^ { 0 }$ (age zero). Thus drift measures how far older observations lie from the prediction-date relation. The nonlinear term does not drift.

Models. The linear model uses the $p = 8$ covariates. The quadratic model adds the 36 centered second-order features $x _ { i } x _ { j } - \kappa _ { i } { \bf 1 } \{ i = j \}$ for $i \leq j$ , which gives 44 features. Every feature has mean zero. The quadratic span contains the true nonlinear term, so the quadratic model is correctly specified up to drift. Both models are fitted without an intercept by ridge regression. For features $\phi ,$ the estimator minimizes $\begin{array} { r } { h ^ { - 1 } \sum _ { t } ( y _ { t } - w ^ { \top } \phi _ { t } ) ^ { 2 } + \lambda \| w \| ^ { 2 } } \end{array}$ over the h most recent observations, with penalty $\lambda \in \ \{ 0 . 0 3 , 0 . 5 \}$ on the raw feature scale. The memory $h \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ is the number of training observations used. The histories are nested, so every memory uses the last h rows of one 256-row history.

Worlds, replicates, and cells. A world fixes 16 test covariate vectors drawn from $N ( 0 , \Sigma )$ and their targets. Within a world, we draw $R = 3$ replicates. Each replicate is an independent 256-row training history. All models, penalties, and memories are fitted to the same replicates and scored on the same targets. Hence replicate variation measures the efect of training sampling at fixed dates. Each factorial condition uses $W = 2 4$ independent worlds. A cell is one combination of design, model, penalty, and memory. It pools $N = 1 6 W$ test targets, collected in the vector $y \in \mathbb { R } ^ { N }$ . Let $f _ { r } \in \mathbb { R } ^ { N }$ be the forecasts of replicate r and $\bar { f } = R ^ { - 1 } \sum _ { r } f _ { r }$ their average. Let $\boldsymbol { \mu } \in \mathbb { R } ^ { N }$ be the known conditional mean at the test points.

Score and exact decomposition. The score is the out-of-sample $R ^ { 2 }$ against the zero forecast,

$$
R ^ { 2 } = 1 - \frac { R ^ { - 1 } \sum _ { r } \| f _ { r } - y \| ^ { 2 } } { \| y \| ^ { 2 } } .
$$

Define the aggressiveness $g = \| { \bar { f } } \| / \| y \|$ , the alignment $a = y ^ { \top } { \bar { f } } / ( \| y \| \| { \bar { f } } \| )$ (the cosine between targets and the average forecast), and the instability $\begin{array} { r } { u = R ^ { - 1 } \sum _ { r } \| f _ { r } - \bar { f } \| ^ { 2 } / \| y \| ^ { 2 } } \end{array}$ . Expanding the square gives the exact identity

$$
R ^ { 2 } = 2 a g - g ^ { 2 } - u .
$$

We call 2ag the alignment reward, $g ^ { 2 }$ the aggressiveness cost, and u the instability cost. We compute all three from unrounded forecasts. The largest reconstruction error over all 844 cells is 8.88 $\times 1 0 ^ { - 1 6 }$ . We also report the excess risk $\begin{array} { r } { R ^ { - 1 } N ^ { - 1 } \sum _ { r } \| f _ { r } - \mu \| ^ { 2 } } \end{array}$ , which removes observation noise from the comparison.

Spectral proxy. For each cell we compute a population approximation to the excess risk. Let $v _ { k }$ be the variance of feature k. It equals $\kappa _ { k }$ for a linear feature and $\kappa _ { i } \kappa _ { j } ( 1 + { \bf 1 } \{ i = j \} )$ for a quadratic one. Define the ridge filter $\tau _ { k } = v _ { k } / ( v _ { k } + \lambda )$ and the variance-efective dimension $\begin{array} { r } { d _ { 2 } ( \lambda ) = \sum _ { k } \tau _ { k } ^ { 2 } } \end{array}$ . The shrinkage bias in feature k is $- \beta _ { k } ^ { 0 } ( 1 - \tau _ { k } )$ . The drift bias in the drifting coordinate is $- \frac { h + 1 } { 2 } \rho \tau _ { j }$ under gradual drift and $- \frac { \operatorname* { m a x } ( h - 4 8 , 0 ) } { h } 0 . 6 5 \tau _ { j }$ under the break. Let $B$ be the sum over features of $v _ { k }$ times the squared total bias. Let $A = \gamma ^ { 2 } \ \mathrm { f o r }$ the linear model, which omits the nonlinear term, and $A = 0$ for the quadratic model. The proxy excess risk is

$$
{ \widehat { \mathrm { E R } } } = A + B + \left( s ^ { 2 } + A \right) d _ { 2 } ( \lambda ) / h .
$$

The proxy $R ^ { 2 }$ is $1 - ( s ^ { 2 } + \widehat \mathrm { E R } ) / ( 0 . 7 ^ { 2 } + \gamma ^ { 2 } + s ^ { 2 } )$ . The proxy gain of the quadratic model is $\widehat { \mathrm { E R } } _ { \mathrm { l i n e a r } } - \widehat { \mathrm { E R } } _ { \mathrm { q u a d r a t i c } } .$ This approximation is exact in form for correctly specified gradual linear drift. For the break and for the omitted nonlinearity, it is a diagnostic approximation, because the omitted term enters only as extra residual variance. We assign no proxy to the null design.

Focused designs. Four further designs share the ridge fits and score definitions above. All use zero nonlinear strength unless noted. (i) Ridge scaling. We use the flat spectrum, $\mathrm { S N R } \quad = \quad 1$ gradual drift with $\rho ~ \in ~ \{ 0 . 0 0 2 , 0 . 0 0 4 , 0 . 0 0 8 , 0 . 0 1 6 \}$ , linear ridge with $\begin{array} { r l r } { \lambda } & { { } = } & { 0 . 0 3 , } \end{array}$ and 11 memories in {16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256}. The design has 512 worlds. Training covariates are deterministic and balanced. A block of $2 p = 1 6$ rows contains $\pm \sqrt { p \kappa _ { k } } e _ { k }$ for each coordinate $k ,$ placed at positions k and $2 p + 1 - k .$ . The block is tiled 16 times, and every memory is a whole number of blocks. Hence the training second-moment matrix equals $\Sigma$ exactly. Moreover, the age-weighted second-moment matrix equals the mean age times $\Sigma ,$ because each sign pair is symmetric in age. Only the noise is redrawn across replicates. The test covariates stay random. We take the observed optimal memory $h ^ { \star }$ as the grid memory with the smallest excess risk. We compare it with the proxy minimizer and regress log $h ^ { \star }$ on log $\rho .$ Balancing the squared drift bias, of order $( h \rho ) ^ { 2 }$ , against the variance, of order $1 / h ,$ , predicts a slope of $- 2 / 3 . ( i i )$ Directional drift. We use gradual drift with $\rho = 0 . 0 1 2 , \mathrm { S N R } = 2 , q = 0 . 4 5$ , and the drifting coordinate $j = 1$ (variance $\kappa _ { 1 } = 1 ) \ { \mathrm { o r } } \ j = 8$ (variance $\kappa _ { 8 } = 0 . 4 5 ^ { 7 } \approx 0 . 0 0 4 )$ . The raw drift is equal, but the filtered drift $\kappa _ { j } ( \tau _ { j } \rho ) ^ { 2 }$ difers. (iii) Stationary weak signal. We use zero drift, $q = 0 . 4 5 , \ \beta _ { 2 } ^ { 0 } = 0 . 1 5 / \sqrt { \kappa _ { 2 } } , \ \gamma = 0$ , noise standard deviation $s = 1$ , and 192 worlds. (iv) Permutation null. We use zero drift, $q = 0 . 4 5 , \mathrm { S N R } = 2 , \gamma = 0 . 4$ , and 96 worlds. In each training history and each test set, we generate the responses from an independent covariate draw and then permute them across rows. Hence responses are independent of the covariates. We set the conditional mean to zero. No forecast should beat zero systematically. The directional, weak-signal, and null designs each use both models, both penalties, and the five main-grid memories (the scaling design uses linear ridge only). In total, the grid has $3 6 \times 2 0 = 7 2 0$ factorial cells and $4 0 + 4 4 + 2 0 + 2 0 = 1 2 4$ focused cells, which gives 844.

Random numbers and pairing. We use NumPy’s default rng with base seed $s _ { 0 } = 2 0 2 6 1 0 0 3$ . The factorial seed is $s _ { 0 } + 1 0 0 0 i _ { \mathrm { S N R } } + i _ { q } ,$ where i and $i _ { q }$ are the zero-based positions of the SNR and decay in their lists. Hence all nine drift–γ conditions with the same SNR and spectrum reuse the same standard normal draws, and only the coeficients and noise scale change. These common random numbers pair the contrasts across drift and nonlinear strength. The directional designs use $s _ { 0 } + 2 0 0 0$ and the scaling designs use $s _ { 0 } + 3 0 0 0$ . The weak-signal and null designs use $s _ { 0 } + 4 0 0 0 + k$ , where $k = 3 8$ and $k = 3 9$ is the position of the design in the run order.

Paired contrasts and intervals. For each of the 360 factorial conditions (drift, SNR, $\gamma , q , \lambda , h )$ , we compare the two models fitted to the same worlds. Let $\Delta R ^ { 2 } = R _ { \mathrm { q u a d r a t i c } } ^ { 2 } { - } R _ { \mathrm { l i n e a r } } ^ { 2 }$ . For world $w ,$ let $\ell _ { w }$ be the mean squared forecast error averaged over replicates and the 16 targets. The paired diference is $d _ { w } = \ell _ { w } ^ { \mathrm { l i n e a r } } - \ell _ { w } ^ { \mathrm { q u a d r a t i c } }$ . The standard error of $\Delta R ^ { 2 }$ is $\mathrm { s d } ( d _ { w } ) / \sqrt { W }$ divided by the mean per-world target energy. The interval is $\Delta R ^ { 2 } \pm 1 . 9 6$ standard errors. The 24 worlds are independent, so this normal interval treats worlds as clusters. We call a contrast clear if its interval excludes zero. We call the proxy sign matching if $\Delta R ^ { 2 }$ and the proxy gain have the same sign. The intervals are descriptive. They do not adjust for multiplicity or for the search over specifications.

## B.2 Results

![](images/fdbf17f01d54842bc72ae0bf2f59a7729941074c14be42bfb837bafb56d34c46.jpg)

Figure 4: Ridge theorem check under a balanced linear design. Each panel varies memory at a fixed coeficientdrift rate. Solid curves are observed excess prediction risk and dashed curves are the spectral approximation. The vertical rule marks the observed optimum; all four agree with the proxy grid optimum, and the log-optimalmemory versus log-drift slope is -0.67.  
![](images/55472efbab30ff8e0a437e2779189af1cb43aa8d5311b703a5512885c3c0c6d7.jpg)  
Figure 5: Direction-aware drift. Rows put the same raw drift rate in a high-variance coordinate (top) or heavily filtered low-information coordinate (bottom); columns vary ridge penalty. Linear and quadratic scores are shown at every candidate memory. With light regularization, the best linear memory is 32 in the high-variance direction and 256 in the low-information direction.

![](images/723570c0fa3d2e1f2073ace6c962f75abe6b59094af32a592079ad042e1d2dcd.jpg)

Figure 6: Stationary weak-signal and permutation-null controls. Columns compare light and heavy ridge penalties; each panel shows both model classes over every memory. The weak-signal optimum is long-memory, heavily shrunk linear ridge. None of the 20 null specifications exceeds the zero forecast in this run.  
![](images/5c5c76da8e5b724b00095e0eb25695251f5ead7f29d8ff4002957406f678602c.jpg)  
Figure 7: Quadratic-minus-linear $R ^ { 2 }$ gap versus memory under zero drift. Rows fix SNR and columns fix covariance decay q and penalty λ. Solid curves with bars are the observed gap and its paired 95% interval; colors give nonlinear strength γ. Dotted curves are the leading spectral proxy. Positive values favor the quadratic model. The vertical range of each row covers the observed intervals, so a proxy curve that leaves the panel has missed the observed gap by a large margin. The intervals are descriptive and unadjusted for multiplicity.

![](images/d17bb1aedc897c0178c005a48cce20129934340c7ab79b4bc8135af37c189ccc.jpg)  
Figure 8: Quadratic-minus-linear $R ^ { 2 }$ gap versus memory under gradual drift. Rows fix SNR and columns fix covariance decay q and penalty λ. Solid curves with bars are the observed gap and its paired 95% interval; colors give nonlinear strength γ. Dotted curves are the leading spectral proxy. Positive values favor the quadratic model. The vertical range of each row covers the observed intervals, so a proxy curve that leaves the panel has missed the observed gap by a large margin. The intervals are descriptive and unadjusted for multiplicity.

![](images/207289fc92a24bad65456db08f65c2e6872cd981762354e671097d95b6e1872e.jpg)  
Figure 9: Quadratic-minus-linear $R ^ { 2 }$ gap versus memory under abrupt drift. Rows fix SNR and columns fix covariance decay q and penalty λ. Solid curves with bars are the observed gap and its paired 95% interval; colors give nonlinear strength γ. Dotted curves are the leading spectral proxy. Positive values favor the quadratic model. The vertical range of each row covers the observed intervals, so a proxy curve that leaves the panel has missed the observed gap by a large margin. The intervals are descriptive and unadjusted for multiplicity.

## C EMPIRICAL DYNAMICS OF FORECAST COMPONENTS

## C.1 Experimental method

Targets and sample. Let $t = 1 , \dots , T$ index the $T = 7 { , } 3 7 4$ trading days from 1 September 1987 through 30 November 2016, in the calendar of the Kenneth French daily files. Let $j = 1 , \ldots , 1 7$ index the portfolios of the Kenneth French 17-industry classification: Food, Mines, Oil, Clths, Durbl, Chems, Cnsum, Cnstr, Steel, FabPr, Machn, Cars, Trans, Utils, Rtail, Finan, and Other. Let $R _ { t j }$ be the value-weighted daily return of portfolio j in percent, and let $r _ { t }$ be the daily risk-free rate in percent. We define the target $y _ { t j } = ( R _ { t j } - r _ { t } ) / 1 0 0$ , the daily excess return in decimals. Both series come from the Kenneth French data library, in the vintage built from the 202607 CRSP database. This vintage can difer slightly from the one that the source paper used. We fit and score each industry separately, so no parameter is shared across industries.

Forecast origins and test days. Let m run over the 323 calendar months from January 1990 through November 2016. Let $\mathcal { D } _ { m }$ be the set of trading days in month m. Each $\mathcal { D } _ { m }$ holds 15 to 23 days, with a median of 21, and the 323 sets hold 6,784 days. We fit each model once at the start of month $m _ { ; }$ , using only days before month m. Preprocessing, validation, early stopping, and tuning also use only days before month m. We use that fit to forecast every day in $\mathcal { D } _ { m }$ . The 590 trading days before 1990 serve only as training days. Each configuration thus yields $6 , 7 8 4 \times 1 7 = 1 1 5 , 3 2 8$ out-of-sample forecasts per seed.

Benchmark and score. Let S be a set of (day, industry) pairs, and let $\mathbf { y } _ { S }$ be the vector of the targets y in $y _ { t j }$ S. Let $\hat { \mathbf { y } } _ { S }$ be a vector of forecasts for the same pairs. We score the forecasts against the zero forecast by

$$
R ^ { 2 } ( \hat { \mathbf { y } } _ { S } ) = 1 - \frac { \| \mathbf { y } _ { S } - \hat { \mathbf { y } } _ { S } \| ^ { 2 } } { \| \mathbf { y } _ { S } \| ^ { 2 } } .
$$

The score is 1 for a perfect forecast, 0 for the zero forecast, and negative for a forecast that is worse than zero. The norms pool all pairs in $S ,$ so $R ^ { 2 }$ is a ratio of sums and not an average of finer scores. The zero benchmark suits daily returns, because their mean is small. Over 1990–2016 the mean of $y _ { t j }$ is 0.03% per day, against a root mean square of 1.4%. The source paper uses the same benchmark.

Inputs of the two branches. In the tabular branch, a model maps the vector $x _ { t }$ to the target $y _ { t j }$ . In the temporal branch, a model maps the context $X _ { t } ^ { ( L ) } = ( x _ { t - L + 1 } , \ldots , x _ { t } ) \in \mathbb { R } ^ { L \times 1 2 9 }$ to $y _ { t j }$ . The context holds the predictors of the last L trading days, including day t. We use $L = 2 0$ as the primary context length and $L = 6 0$ as the secondary one, about one and three months of trading days. We fixed both lengths in advance. Ridge, XGBoost, and the feed-forward networks receive the flattened context, a vector with 129L entries (2,580 for $L = 2 0$ and 7,740 for $L = 6 0 )$ . The recurrent and attention networks receive the ordered context. Thus all temporal models use the same numbers within a context length, and their diferences reflect the model type. By contrast, a comparison across branches or across context lengths changes the information set, so we do not read it as an efect of the model alone.

Training windows. Let $w \in \{ 1 6 , 6 4 , \infty \}$ be the window length in months. We write $w = \infty$ for the expanding window. For forecast month m, the window holds the trading days of the w calendar months before m, or all days before m if $w = \infty$ . No window holds a day of month m or later. In the temporal branch, a day t may train a model only if all L days of its context lie inside the window. Hence a short window never uses predictors from before its start. Test contexts may reach back into the previous month, because those days are observed when we forecast. The numbers of days appear in Table 3. The sample starts in September 1987, so the 64-month window holds the same days as the expanding window for the $3 7$ forecast months from January 1990 through January 1993. The 16-month window is complete from the first forecast month. The memory h of the main text is the number of days in the window before subsampling.

Training subsamples and seeds. Let $s \in \{ 0 , 1 , . . . , 5 \}$ be the seed. Let $n _ { c }$ be the number of window days in calendar month c. For every seed s, forecast month m, and month c of the window, we draw $\left\lfloor 0 . 8 n _ { c } \right\rfloor$ of these days without replacement. The draw uses a $\mathrm { N u m P y }$ generator seeded by $( s , m , c )$ . We call the union of the draws over c the subsample of the fit. All models of a branch, and all context lengths, use the same draws for the same $( s , m , c )$ , up to the first month of a window, where the context rule removes some days. Thus the replicates are paired across models and windows. The seed s also determines, together with the model, the month, and the window, the random initialization, the minibatch order, and the dropout masks of a network. Ridge and XGBoost use no random numbers once the subsample is fixed, because we set the XGBoost row and column sampling rates to one. Hence their variation across seeds comes from the subsample alone. Random forest adds bootstrap randomness, and networks add their own training randomness. We use six seeds, fewer than the 20 repetitions of the source paper, to limit compute. Each seed defines one replicate in the sense of the main text.

<table><tr><td>Window</td><td>Months</td><td>Window days</td><td>Subsampled days</td><td>Network fit days</td><td>Validation days</td></tr><tr><td>16 months</td><td>16</td><td>328-340</td><td>256-265</td><td>192-199</td><td>79-87</td></tr><tr><td>64 months</td><td>28-64</td><td>590-1,352</td><td>460-1,052</td><td>263-856</td><td>246-255</td></tr><tr><td>Expanding</td><td>28-350</td><td>590-7,353</td><td>460–5,718</td><td>263-5,522</td><td>246-255</td></tr></table>

Table 3: Trading days per fit in the tabular branch, from the minimum to the maximum over the 323 forecast months. Window days are the days of the window before subsampling. Network fit days are the subsampled days before the validation block, which train the networks. The counts do not depend on the seed. A temporal fit has L − 1 fewer window days.

Preprocessing. Let I be the set of days that train a fit. For each predictor $i ,$ we compute its mean $\mu _ { i }$ and standard deviation $\sigma _ { i }$ over I, with divisor |I|. In the temporal branch, we compute them over all days of all contexts of I. We replace $\sigma _ { i }$ by 1 if $\sigma _ { i } < 1 0 ^ { - 8 }$ . We standardize every predictor to $z _ { i } = ( x _ { i } - \mu _ { i } ) / \sigma _ { i } .$ , and we give the same standardized values to all models, including the boosted trees. Networks also standardize each target with its training mean and standard deviation. We map their forecasts back to decimal returns before scoring. Ridge, XGBoost, and random forest fit the unstandardized targets. Hence every score and every validation criterion uses return units. We store the predictors and targets in single precision.

Linear models. Let $z _ { t }$ be the standardized input of day t. For industry j, ridge regression solves

$$
\operatorname* { m i n } _ { b _ { 0 } , b } \sum _ { t \in \mathbb { Z } } \left( y _ { t j } - b _ { 0 } - z _ { t } ^ { \top } b \right) ^ { 2 } + \alpha \| b \| ^ { 2 } .
$$

The intercept $b _ { 0 }$ is not penalized. The penalty multiplies the sum of squared errors and not their mean, so $\alpha = 1$ shrinks little when |I| is large. We use $\alpha = 1$ as a fixed anchor, as in the source paper, and we tune α on a grid as a second ridge model. The temporal ridge solves the same problem for the flattened context. Scikit-learn solves the tabular problem directly and the flattened problem with its LSQR solver.

Boosted trees. XGBoost fits gradient-boosted regression trees by squared-error loss. It provides a boostedtree comparison with the random forest of the source paper. For industry $j$ we fit 150 rounds with maximum depth D and learning rate $\eta .$ We fix the minimum child weight at $5 ,$ the $L _ { 2 }$ leaf penalty at 1, and the $L _ { 1 }$ penalty at 0. We use no row or column subsampling, and we find splits from histograms on the GPU. All other settings keep their library defaults. We tune D and $\eta .$ The temporal version uses the flattened context.

Random forest. The tabular forest fits a separate regressor for each of the 17 industries using cuML on an NVIDIA A100. Each regressor grows 200 bootstrap trees, with maximum depth 5, minimum leaf size 1, all 129 predictors eligible at a split, and 128 histogram bins. The forest settings are fixed across windows and years; there is no validation grid for this anchor. The same six subsample seeds and monthly forecast origins used by the other tabular models determine its training sets.

Feed-forward networks. For industry $j ,$ a network maps the standardized input to a scalar through one or two ReLU hidden layers. Dropout with rate δ follows each hidden layer. The small network has one hidden layer of width 32. The medium network has two hidden layers of widths 128 and 64. The 17 networks of a model train as one batched tensor program, but they share no parameters. We initialize them with the PyTorch defaults for linear layers. We keep the networks small, because the samples are small: a 16-month fit has about 260 subsampled days.

Recurrent and attention networks. These networks read the ordered context. The recurrent network (RNN) applies $h _ { \tau } = \mathrm { t a n h } ( W _ { x } z _ { \tau } + W _ { h } h _ { \tau - 1 } + b )$ for $\tau = t - L + 1 , \dots , t .$ , with $h _ { t - L } = 0$ . The long short-term memory network (LSTM) uses the standard input, forget, and output gates and a memory cell. Both predict from the last state through dropout and a linear map. The Transformer embeds the 129 predictors of each day linearly into width d and adds sinusoidal position codes. It applies two pre-normalization blocks. Each block has multi-head self-attention with 4 heads and a residual connection, then a feed-forward layer of width 2d with a GELU activation and a residual connection. Dropout acts on the attention weights and inside each residual branch. A final layer normalization and a linear map turn the state of day t into the forecast. Each industry again has its own network. We list the settings, the candidate grids, and the parameter counts of all models in Table 4.

Song and Ding
<table><tr><td>Model</td><td>Input</td><td>Fixed settings</td><td>Candidate grid (eight candidates)</td></tr><tr><td>Tabular branch</td><td></td><td></td><td></td></tr><tr><td>Ridge, fixed alpha</td><td>xt</td><td> $\alpha = 1$ </td><td>none</td></tr><tr><td>Tuned ridge</td><td> $x _ { t }$ </td><td>none</td><td> $\alpha \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } , 0 . 1 , 1 , 1 0 , 1 0 0 , 1 0 ^ { 4 } , 1 0 ^ { 6 } \}$ </td></tr><tr><td>XGBoost</td><td> $x _ { t }$ </td><td>150 rounds; min. child weight 5;  $L _ { 2 }$  penalty 1; no subsampling</td><td> $D \in \{ 2 , 3 , 5 , 7 \}$  and  $\eta \in \{ 0 . 0 3 , 0 . 1 \}$ </td></tr><tr><td>Random forest</td><td> $x _ { t }$ </td><td>cuML; 200 bootstrap trees; depth  $5 ;$  min. leaf 1; all features; 128 bins</td><td>none</td></tr><tr><td>Small MLP</td><td>xt</td><td>width (32); 4,193 parameters; batch 256; at most 30 epochs; patience 5</td><td>learning rate {0.001, 0.003}, weight decay {0, 0.01}, dropout {0, 0.1}</td></tr><tr><td>Medium MLP</td><td>xt</td><td>widths (128, 64); 24,961 parameters; batch 256; at most 30 epochs; patience 5</td><td>same as the small MLP</td></tr><tr><td>Temporal branch</td><td></td><td></td><td></td></tr><tr><td>Flattened ridge</td><td>flat</td><td> $\alpha = 1$ </td><td>none</td></tr><tr><td>Flattened XGBoost Small flattened MLP</td><td>flat flat</td><td>as XGBoost width (32); learning rate 0.002;</td><td>as XGBoost none</td></tr><tr><td></td><td></td><td>weight decay 0.01; dropout 0; batch 128; at most 20 epochs; patience 4; 82,625 parameters (L = 20), 247,745 (L = 60)</td><td></td></tr><tr><td>Tuned flattened MLP</td><td>flat</td><td>weight decay 0; batch 128; at most 20 epochs; patience 4; 61,969 or 82,625 parameters  $( L = 2 0 )$  185,809 or 247,745 (L = 60)</td><td>width {(24), (32)}, learning rate {0.001, 0.003}, dropout {0, 0.1}</td></tr><tr><td>Vanilla RNN</td><td>ordered</td><td>weight decay 0; batch 128; at most 20 epochs; patience 4; 3,721 or 5,217 parameters</td><td>hidden size {24, 32}, learning rate {0.001, 0.003}, dropout {0, 0.1}</td></tr><tr><td>LSTM</td><td>ordered</td><td>same training settings; 6,829 or 9,361 parameters</td><td>hidden size {12, 16}, learning rate {0.001, 0.003}, dropout {0, 0.1}</td></tr><tr><td>Small Transformer</td><td>ordered</td><td>2 blocks; 4 heads; feed-forward width 2d; same training settings; 6,577 or 12,937 parameters</td><td>width d ∈ {16, 24}, learning rate {0.001, 0.003}, dropout {0, 0.1}</td></tr></table>

Table 4: Model specifications. The input is the predictor vector $x _ { t } ,$ the flattened context (flat), or the ordered context. Each tuned model has eight candidates, which are all combinations of the listed values. Parameter counts are trainable parameters per industry. When a cell lists two counts, they belong to the two grid values in the listed order. All networks use AdamW and clip the gradient norm at 1.0.

Hyperparameter selection. Ridge with $\alpha = 1$ , the tabular random forest, and the small flattened network have fixed settings, so no tuning choice afects these anchors. We tune every other model once per calendar year, and separately for each context length. Each tuned model has eight candidates (Table 4). For window w and forecast month $m ,$ the validation block is the last $v _ { w }$ months before $m _ { ; }$ , with $v _ { 1 6 } = 4$ and $v _ { 6 4 } = v _ { \infty } = 1 2$ . Let $\nu _ { w }$ be the days of the block. We use all of them, without subsampling. Let $\mathcal { T } _ { w , s }$ be the subsampled days of seed s that precede the block. For the January refit of the year, we fit each candidate θ on $\mathcal { T } _ { w , s }$ for the three windows and the seeds $s \in \{ 0 , 1 \}$ , and we predict $\nu _ { w }$ . For each industry we compute the relative squared error

$$
\mathrm { R S E } _ { j } ( \theta , w , s ) = \frac { \sum _ { t \in \mathcal { V } _ { w } } \left( y _ { t j } - \hat { y } _ { t j } \right) ^ { 2 } } { \sum _ { t \in \mathcal { V } _ { w } } y _ { t j } ^ { 2 } } .
$$

The score of candidate $\theta$ is the mean of $\mathrm { R S E } _ { j } ( \theta , w , s )$ over the 17 industries, the three windows, and the two seeds. We select the candidate with the smallest score. The selection uses only days before 1 January of that year. It then holds for all monthly refits of the year (eleven in 2016), for all three windows, and for all seeds. Thus every window uses the same hyperparameters, and a contrast between windows holds the model capacity fixed. The division by $\textstyle \sum y ^ { 2 }$ keeps volatile industries from dominating the choice, and it keeps the zero benchmark. The validation days always follow the training days, so no step uses random cross-validation. The frozen configuration files fix the grids and the validation lengths for all runs.

Network training. We minimize the mean squared error of the standardized targets with AdamW. We clip the gradient norm at 1.0, and we use the batch sizes in Table 4. After each epoch we compute the validation score: the mean over industries of $\mathrm { R S E } _ { j }$ on $\nu _ { w }$ , in return units. Training stops after the maximum number of epochs, or when the validation score has not improved for the patience number of epochs. We keep the weights of the epoch with the best validation score. A forecast fit of a network trains on $\mathcal { T } _ { w , s } ,$ , and it uses the validation block only to pick the epoch. By contrast, ridge, XGBoost, and random forest train on all subsampled days of the window, including the last $v _ { w }$ months. Consequently, the gradient updates of a network see only 12 of the 16 months in the shortest window, 52 of 64 months in the 64-month window, and all but the last 12 months in the expanding window (Table 3).

Replicate statistics and exact decomposition. Fix a configuration (a model, a context length, and a window) and a set S. Let $\hat { \mathbf { y } } _ { S } ^ { ( s ) }$ be the forecasts of seed s for the pairs in S, and let $M = 6$ be the number of seeds. The score of seed s is $R _ { s } ^ { 2 } = R ^ { 2 } ( \hat { \mathbf { y } } _ { S } ^ { ( s ) } )$ . The mean seed score is $\overline { { R ^ { 2 } } } = M ^ { - 1 } \textstyle \sum _ { s } R _ { s } ^ { 2 }$ . We also report the sample standard deviation $\hat { \sigma }$ of the six scores, with divisor $M - 1$ , and the interval $\overline { { R ^ { 2 } } } \pm t _ { 0 . 9 7 5 , 5 } \hat { \sigma } / \sqrt { M }$ with $t _ { 0 . 9 7 5 , 5 } = 2 . 5 7 0 6$ . Let $\bar { \mathbf { y } } _ { S } = M ^ { - 1 } \sum _ { s } \hat { \mathbf { y } } _ { S } ^ { ( s ) }$ be the mean forecast. The ensemble score is $R ^ { 2 } ( \bar { \bf { y } } _ { S } )$ . Write $Y _ { 2 } = \| \mathbf { y } _ { S } \| ^ { 2 }$ and define

$$
a = \frac { \langle \mathbf { y } _ { S } , \bar { \mathbf { y } } _ { S } \rangle } { \| \mathbf { y } _ { S } \| \| \bar { \mathbf { y } } _ { S } \| } , \qquad g = \frac { \| \bar { \mathbf { y } } _ { S } \| } { \| \mathbf { y } _ { S } \| } , \qquad u = \frac { M ^ { - 1 } \sum _ { s } \| \hat { \mathbf { y } } _ { S } ^ { ( s ) } - \bar { \mathbf { y } } _ { S } \| ^ { 2 } } { Y _ { 2 } } .
$$

We call a the alignment, $g$ the aggressiveness, and u the instability. The alignment is the cosine between the targets and the mean forecast. The aggressiveness is the size of the mean forecast relative to the targets. The instability is the mean squared distance of the seed forecasts from their mean, relative to $Y _ { 2 }$ . Expanding the squares gives the exact identity

$$
\overline { { { R ^ { 2 } } } } = 2 a g - g ^ { 2 } - u , \qquad R ^ { 2 } ( \bar { \bf y } _ { S } ) = 2 a g - g ^ { 2 } = \overline { { { R ^ { 2 } } } } + u .
$$

We call 2ag the alignment reward, $g ^ { 2 }$ the aggressiveness cost, and u the instability cost. We compute all three from unrounded forecasts, and the identity holds to within $1 0 ^ { - 9 }$ in all 351,540 scope rows. Because the seeds vary both the subsample and the fitting randomness, u mixes both sources of variation.

Scopes and uncertainty. We compute these statistics for five scopes S. The overall scope holds all 6,784 days and 17 industries. The industry scope holds all days of one industry. The year scope holds all industries and days of one calendar year (2016 has 11 months). The month scope holds all industries and days of one forecast month. The month-by-industry scope holds the days of one forecast month for one industry. Each scope pools its sums of squares before it forms a score, so the score of a coarse scope is not the average of the scores of finer scopes. This appendix displays the overall scope in tables, and the month and month-by-industry scopes in curve plots. The industry and year scopes remain in the performance files. The interval above describes the variation of the training seeds and subsamples only. It ignores serial dependence across days, dependence across industries, changes of regime, and the choice of hyperparameters. This appendix reports no test of equal predictive accuracy. For the contrast between the two context lengths, we compute $d _ { s } = R _ { s } ^ { 2 } ( L = 2 0 ) - R _ { s } ^ { 2 } ( L = 6 0 )$ for each seed s of one model and window. We report the mean <sup>¯</sup>d of the six values and the interval $\bar { d } \pm t _ { 0 . 9 7 5 , 5 } \mathrm { s d } ( d _ { s } ) / \sqrt { M }$ . The two contexts share the seed indices, but their subsamples difer slightly at the start of each window.

Computation and reproducibility. We ran the experiment as resumable jobs on the Vessl cloud service, each with one NVIDIA A100 GPU. There is one fitting job for the classical models and one for the neural models of each branch, and one aggregation job per branch. Each job splits its work into resumable tasks by calendar year and model group. XGBoost uses CUDA, and ridge uses the CPU. The jobs install $\mathrm { P y }$ Torch 2.6.0 (CUDA 12.4), NumPy 2.2.6, pandas 2.2.3, scikit-learn 1.7.2, and XGBoost 3.0.5, and they request deterministic kernels where $\mathrm { P y }$ Torch has them. Before a job starts, it checks the predictor panel against a SHA-256 hash. Every fit writes a checkpoint whose hash covers the configuration, the design, the training, validation, and test rows, the seeds, and the forecasts. The aggregation step checks every hash and refuses to continue when a checkpoint is missing or does not match.

<table><tr><td>Model</td><td></td><td>16 months</td><td>64 months</td><td>Expanding</td></tr><tr><td rowspan="2">Ridge, fixed alpha</td><td>Ensemble</td><td>0.0706</td><td>0.3481</td><td>0.3477</td></tr><tr><td>Mean [95% CI]</td><td>-0.0909 [-0.1033, -0.0786]</td><td>0.3256 [0.3235, 0.3277]</td><td>0.3386 [0.3374, 0.3397]</td></tr><tr><td rowspan="2">Tuned ridge</td><td>Ensemble</td><td>0.3671</td><td>0.3767</td><td>0.3644</td></tr><tr><td>Mean [95% CI]</td><td>0.3515 [0.3487, 0.3543]</td><td>0.3668 [0.3658, 0.3678]</td><td>0.3602 [0.3593, 0.3611]</td></tr><tr><td rowspan="2">XGBoost</td><td>Ensemble</td><td>0.3621</td><td>0.3619</td><td>0.3392</td></tr><tr><td>Mean [95% CI]</td><td>0.3316 [0.3284, 0.3348]</td><td>0.3419 [0.3380, 0.3457]</td><td>0.3232 [0.3202, 0.3262]</td></tr><tr><td rowspan="2">Small MLP</td><td>Ensemble</td><td>0.3096</td><td>0.3414</td><td>0.3423</td></tr><tr><td>Mean [95% CI]</td><td>0.2869 [0.2841, 0.2898]</td><td>0.3231 [0.3212, 0.3251]</td><td>0.3301 [0.3274, 0.3328]</td></tr><tr><td rowspan="2">Medium MLP</td><td>Ensemble</td><td>0.3018</td><td>0.3383</td><td>0.3425</td></tr><tr><td>Mean [95% CI]</td><td>0.2786 [0.2756, 0.2816]</td><td>0.3189 [0.3151, 0.3228]</td><td>0.3270 [0.3232, 0.3308]</td></tr><tr><td rowspan="2">Random forest</td><td>Ensemble</td><td>0.3705</td><td>0.3494</td><td>0.3079</td></tr><tr><td>Mean [95% CI]</td><td>0.3549 [0.3514, 0.3583]</td><td>0.3374 [0.3341, 0.3407]</td><td>0.2953 [0.2826, 0.3080]</td></tr></table>

Table 5: Tabular: pooled out-of-sample $R ^ { 2 }$ for every reported model and training window. In each cell, the upper line gives the $R ^ { 2 }$ of the six-seed ensemble, which scores the mean forecast. The lower line gives the mean of the six seed scores, with its seed-only 95% t interval in brackets.

## C.2 Results of tabular models

The tabular random forest has mean seed $R ^ { 2 }$ of 0.3549, 0.3374, and 0.2953 for the 16-month, 64-month, and expanding windows, respectively. Its corresponding ensemble scores are 0.3705, 0.3494, and 0.3079. The 16- month window scores highest for this model on both measures.

## C.2.1 Month-by-industry curves

We show one model per figure, with one panel per training window: 16 months, 64 months, and expanding. Each panel draws 17 curves, one per industry. A curve gives the six-seed mean of the pooled $R ^ { 2 }$ for that industry in each month. All panels share the vertical axis [−2.0, 1.0], and each industry keeps one color throughout. Values below −2.0 fall outside the frame, and each caption counts them.

![](images/fd04f6d6b56f29bc906e0c0021ba600b491d3596c87849761573167c6a4e4c9e.jpg)  
Figure 10: Tabular, Ridge, fixed alpha: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 393, 83, and 90 of 5,491 in the three panels.

![](images/eedb0997ee84217df8c2e77ed0770b7db34d2e0f2b3930cb33d9c0f3a2a6eafa.jpg)  
Figure 11: Tabular, Tuned ridge: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 2, 51, and 52 of 5,491 in the three panels.

![](images/57cd644f8e47779c7cd9e748061a47fb6c6a50a0751c5513f46d854c8a05d89d.jpg)  
Figure 12: Tabular, XGBoost: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 0, 36, and 38 of 5,491 in the three panels.

![](images/aa113a0d9a630cfcedebe867615f8ea60f6171e924a1c06775c065088f958310.jpg)  
Figure 13: Tabular, Small MLP: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 1, 2, and 10 of 5,491 in the three panels.

![](images/5b17d039e4725c3c4705c6c86b20173aa518ace6af25f96ff2b9ae749046602e.jpg)  
Figure 14: Tabular, Medium MLP: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 6, 1, and 7 of $^ { 5 , 4 9 1 }$ in the three panels.

![](images/485339570210e32e593dda0b221779143e1cb4f6673938bd6b277024b57e7238.jpg)  
Figure 15: Tabular, Random forest: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16- month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 2, 55, and 63 of 5,491 in the three panels.

## C.3 Results of temporal models, 20-day context

<table><tr><td>Model</td><td></td><td>16 months</td><td>64 months</td><td>Expanding</td></tr><tr><td>Flattened ridge</td><td>Ensemble</td><td>0.1915</td><td>-0.0139</td><td>-0.3308</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.1397 [0.1346, 0.1447]</td><td>-0.2128 [-0.2208, -0.2049]</td><td>-1.1533 [-1.1871, -1.1194]</td></tr><tr><td>Flattened XGBoost</td><td>Ensemble</td><td>0.3341</td><td>0.3485</td><td>0.3379</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.3026 [0.2992, 0.3061]</td><td>0.3287 [0.3258, 0.3315]</td><td>0.3208 [0.3170, 0.3247]</td></tr><tr><td>Small flattened MLP</td><td>Ensemble</td><td>0.0200</td><td>0.1544</td><td>0.2485</td></tr><tr><td></td><td>Mean [95% CI]</td><td>-0.0841 [-0.0926, -0.0756]</td><td>0.0574 [0.0532, 0.0617]</td><td>0.1623 [0.1552, 0.1694]</td></tr><tr><td>Tuned flattened MLP</td><td>Ensemble</td><td>0.0406</td><td>0.1715</td><td>0.2607</td></tr><tr><td></td><td>Mean [95% CI]</td><td>-0.0350 [-0.0559, -0.0141]</td><td>0.1160 [0.1112, 0.1209]</td><td>0.2066 [0.2038, 0.2095]</td></tr><tr><td>Vanilla RNN</td><td>Ensemble</td><td>0.3166</td><td>0.3166</td><td>0.3315</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.2860 [0.2834, 0.2887]</td><td>0.2987 [0.2970, 0.3005]</td><td>0.3184 [0.3161, 0.3206]</td></tr><tr><td>LSTM</td><td>Ensemble</td><td>0.2317</td><td>0.2673</td><td>0.3071</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.2087 [0.2059, 0.2114]</td><td>0.2491 [0.2463, 0.2519]</td><td>0.2905 [0.2886, 0.2925]</td></tr><tr><td>Small Transformer</td><td>Ensemble</td><td>0.3218</td><td>0.3191</td><td>0.3361</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.2828 [0.2789, 0.2866]</td><td>0.2995 [0.2972, 0.3018]</td><td>0.3193 [0.3172, 0.3213]</td></tr></table>

Table 6: Temporal, 20-day context: pooled out-of-sample $R ^ { 2 }$ for every completed model and training window. In each cell, the upper line gives the ${ \bar { R } } ^ { 2 }$ of the six-seed ensemble, which scores the mean forecast. The lower line gives the mean of the six seed scores, with its seed-only 95% t interval in brackets.

## C.3.1 Month-by-industry curves

We show one model per figure, with one panel per training window: 16 months, 64 months, and expanding. Each panel draws 17 curves, one per industry. A curve gives the six-seed mean of the pooled $R ^ { 2 }$ for that industry in each month. All panels share the vertical axis [−2.0, 1.0], and each industry keeps one color throughout. Values below −2.0 fall outside the frame, and each caption counts them.

![](images/072075783ea59af8004c621c40f4a4350740c26f4499607f5bf460b39e480027.jpg)  
Figure 16: Temporal, 20-day context: monthly pooled $R ^ { 2 }$ of every completed model, one curve each, under the 16-month, 64-month, and expanding windows. Each month pools all 17 industries and the trading days of that month. Values below −2.0 fall outside the frame: 3, 11, and 112 of 2,261 model-months in the three panels.

![](images/dee2165652d1807baabe01e07f99a2ed2c4ae50af51d6ce88ef9e64d090311b9.jpg)  
Figure 17: Temporal, 20-day context, Flattened ridge: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 14, 552, and 1,962 of 5,491 in the three panels.

![](images/0f873cfa9d50b1017eeec21bc2943f836b7046ce646a8d99baec79bcfdc3e83e.jpg)  
Figure 18: Temporal, 20-day context, Flattened XGBoost: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 0, 29, and 29 of 5,491 in the three panels.

![](images/d86ee12a61fcee7d2d56140417703e8a8c4d5ae18ac32334f3b9040cfb1d7a63.jpg)  
Figure 19: Temporal, 20-day context, Small flattened MLP: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 31, 46, and 42 of 5,491 in the three panels.

![](images/4f738adf55ba73d0d0d63e2cfb4a4c959376f08c381d9a5a2a135ef0aa3ddefa.jpg)  
Figure 20: Temporal, 20-day context, Tuned flattened MLP: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 23, 2, and 9 of 5,491 in the three panels.

![](images/77fdde5406fc376faa4d0b4cc8c4e11f77d2f090a035c65c4ad2f953abdf0432.jpg)  
Figure 21: Temporal, 20-day context, Vanilla RNN: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 0, 9, and 9 of 5,491 in the three panels.

![](images/91c982229e22434db6720dc1377e356a4f2637c6d4e807a873514ffd2e53ffb4.jpg)  
Figure 22: Temporal, 20-day context, LSTM: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 2, 6, and 4 of 5,491 in the three panels.

![](images/7b4dd891f698f7fe5a861826c86ad1b224f934a24adaeedbe9185b9c67d2f577.jpg)  
Figure 23: Temporal, 20-day context, Small Transformer: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 2, 21, and 22 of 5,491 in the three panels.

## C.4 Results of temporal models, 60-day context

<table><tr><td>Model</td><td></td><td>16 months</td><td>64 months</td><td>Expanding</td></tr><tr><td>Flattened ridge</td><td>Ensemble</td><td>0.1092</td><td>0.1654</td><td>-0.0295</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.0772 [0.0712, 0.0831]</td><td>0.1078 [0.1018, 0.1138]</td><td>-0.2835 [-0.2946, -0.2724]</td></tr><tr><td>Flattened XGBoost</td><td>Ensemble</td><td>0.3161</td><td>0.3402</td><td>0.3253</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.2812 [0.2779, 0.2845]</td><td>0.3223 [0.3188, 0.3257]</td><td>0.3148 [0.3132, 0.3164]</td></tr><tr><td>Small flattened MLP</td><td>Ensemble</td><td>-0.3891</td><td>0.0765</td><td>0.1690</td></tr><tr><td></td><td>Mean [95% CI]</td><td>-0.8216 [-0.8845, -0.7588]</td><td>-0.0124 [−0.0180, -0.0068]</td><td>0.0881 [0.0846, 0.0916]</td></tr><tr><td>Tuned flattened MLP</td><td>Ensemble</td><td>-0.0970</td><td>0.0804</td><td>0.1822</td></tr><tr><td></td><td>Mean [95% CI]</td><td>-0.2223 [-0.2347, -0.2098]</td><td>0.0002 [-0.0034, 0.0039]</td><td>0.1158 [0.1121, 0.1196]</td></tr><tr><td>Vanilla RNN</td><td>Ensemble</td><td>0.3123</td><td>0.3167</td><td>0.3276</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.2776 [0.2747, 0.2806]</td><td>0.2989 [0.2969, 0.3009]</td><td>0.3169 [0.3144, 0.3194]</td></tr><tr><td>LSTM</td><td>Ensemble</td><td>0.2153</td><td>0.2661</td><td>0.3042</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.1913 [0.1873, 0.1952]</td><td>0.2472 [0.2455, 0.2489]</td><td>0.2885 [0.2876, 0.2895]</td></tr><tr><td>Small Transformer</td><td>Ensemble</td><td>0.3191</td><td>0.3160</td><td>0.3342</td></tr><tr><td></td><td>Mean [95% CI]</td><td>0.2712 [0.2658, 0.2766]</td><td>0.2969 [0.2961, 0.2976]</td><td>0.3195 [0.3178, 0.3212]</td></tr></table>

Table 7: Temporal, 60-day context: pooled out-of-sample $R ^ { 2 }$ for every completed model and training window. In each cell, the upper line gives the ${ \bar { R } } ^ { 2 }$ of the six-seed ensemble, which scores the mean forecast. The lower line gives the mean of the six seed scores, with its seed-only 95% t interval in brackets.

## C.4.1 Month-by-industry curves

We show one model per figure, with one panel per training window: 16 months, 64 months, and expanding. Each panel draws 17 curves, one per industry. A curve gives the six-seed mean of the pooled $R ^ { 2 }$ for that industry in each month. All panels share the vertical axis [−2.0, 1.0], and each industry keeps one color throughout. Values below −2.0 fall outside the frame, and each caption counts them.

![](images/81a227f2c18d485750e3d02ebc8ad27d79c174ca4a01c26402461e6058be96d6.jpg)  
Figure 24: Temporal, 60-day context: monthly pooled $R ^ { 2 }$ of every completed model, one curve each, under the 16-month, 64-month, and expanding windows. Each month pools all 17 industries and the trading days of that month. Values below −2.0 fall outside the frame: 32, 0, and 27 of 2,261 model-months in the three panels.

![](images/e1fb99e0ce2cf353430a30a1d558110dcfd49d3c394648e6a4455f794a4496bd.jpg)  
Figure 25: Temporal, 60-day context, Flattened ridge: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 1, 48, and 685 of 5,491 in the three panels.

![](images/ae107f023b028943f9be0f76339659b16ba677ec879f6fbddb29828877373dac.jpg)  
Figure 26: Temporal, 60-day context, Flattened XGBoost: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 0, 5, and 5 of 5,491 in the three panels.

![](images/2a47a3e5fabad412516cce48f79706759011b4760a68098c49abf6d77073b583.jpg)  
Figure 27: Temporal, 60-day context, Small flattened MLP: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 673, 55, and 53 of 5,491 in the three panels.

![](images/440e0e019f413221e5c1ef6660abc089ea55f758e5bd23f7892e2df936e1d9ee.jpg)  
Figure 28: Temporal, 60-day context, Tuned flattened MLP: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 88, 29, and 32 of 5,491 in the three panels.

![](images/6d5a8f7342c5899a79ea1cceebb77fe18a539739f7ebe1b886294097844f9fc2.jpg)  
Figure 29: Temporal, 60-day context, Vanilla RNN: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 2, 4, and 5 of 5,491 in the three panels.

![](images/18f1901b128bfa6fbea3ec469e20165fa6879853264091b7b612011e754ce20e.jpg)  
Figure 30: Temporal, 60-day context, LSTM: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 1, 4, and 3 of 5,491 in the three panels.

![](images/5e6368da1687ec019995c17f57f92850aa03d7a3558628bc36ff53736e7150f3.jpg)  
Figure 31: Temporal, 60-day context, Small Transformer: monthly pooled $R ^ { 2 }$ for the 17 industries, one curve each, under the 16-month, 64-month, and expanding windows. Values below −2.0 fall outside the frame: 1, 6, and 6 of 5,491 in the three panels.

## C.5 Configuration-level numerical results

Table 8: All 60 reported Stage 5 model-window-context cells. $R ^ { 2 }$ is pooled against zero over 6,784 dates and 17 industries. The interval uses six seed scores; ensemble $R ^ { 2 }$ scores their mean forecast. The final three columns obey $\overline { { R ^ { 2 } } } = 2 a g - g ^ { 2 } - u$ up to rounding.
<table><tr><td rowspan=1 colspan=10>Model                  WindowMean $R ^ { 2 }$     SD CI low CI highEnsemble     $2 a g$       $g ^ { 2 }$       u</td></tr><tr><td rowspan=1 colspan=10>TabularRidge, fixed alpha     16m      -0.09090.0118 -0.1033 -0.0786    0.07060.97260.90190.1615</td></tr><tr><td rowspan=1 colspan=2>Ridge, fixed alpha    64m        0.3256</td><td rowspan=1 colspan=1>0.0020</td><td rowspan=1 colspan=4>0.3235  0.3277    0.3481</td><td rowspan=1 colspan=1>0.9373</td><td rowspan=1 colspan=1>0.5892</td><td rowspan=1 colspan=1>0.0225</td></tr><tr><td rowspan=1 colspan=2>Ridge, fixed alpha    Exp.        0.3386</td><td rowspan=1 colspan=1>0.0011</td><td rowspan=1 colspan=1>0.3374</td><td rowspan=1 colspan=3>0.3397    0.3477</td><td rowspan=1 colspan=1>0.7620</td><td rowspan=1 colspan=1>0.4143</td><td rowspan=1 colspan=1>0.0091</td></tr><tr><td rowspan=1 colspan=2>Tuned ridge           16m        0.3515</td><td rowspan=1 colspan=1>0.0027</td><td rowspan=1 colspan=4>0.3487  0.3543    0.3671</td><td rowspan=1 colspan=1>0.8296</td><td rowspan=1 colspan=1>0.4625</td><td rowspan=1 colspan=1>0.0156</td></tr><tr><td rowspan=1 colspan=3>Tuned ridge           64m        0.36680.0010</td><td rowspan=1 colspan=1>0.3658</td><td rowspan=1 colspan=3>0.3678    0.3767</td><td rowspan=1 colspan=1>0.8571</td><td rowspan=1 colspan=1>0.4804</td><td rowspan=1 colspan=1>0.0099</td></tr><tr><td rowspan=1 colspan=2>Tuned ridge           Exp.       0.3602</td><td rowspan=1 colspan=1>0.0008</td><td rowspan=1 colspan=1>0.3593</td><td rowspan=1 colspan=3>0.3611    0.3644</td><td rowspan=1 colspan=1>0.7084</td><td rowspan=1 colspan=1>0.3440</td><td rowspan=1 colspan=1>0.0042</td></tr><tr><td rowspan=1 colspan=2>XGBoost              16m        0.3316</td><td rowspan=1 colspan=1>0.0031</td><td rowspan=1 colspan=1>0.3284</td><td rowspan=1 colspan=3>0.3348    0.3621</td><td rowspan=1 colspan=1>0.7180</td><td rowspan=1 colspan=1>0.3559</td><td rowspan=1 colspan=1>0.0305</td></tr><tr><td rowspan=1 colspan=2>XGBoost              64m       0.3419</td><td rowspan=1 colspan=1>0.0037</td><td rowspan=1 colspan=1>0.3380</td><td rowspan=1 colspan=3>0.3457    0.3619</td><td rowspan=1 colspan=1>0.7345</td><td rowspan=1 colspan=1>0.3725</td><td rowspan=1 colspan=1>0.0201</td></tr><tr><td rowspan=1 colspan=2>XGBoost              Exp.       0.3232</td><td rowspan=1 colspan=1>0.0029</td><td rowspan=1 colspan=1>0.3202</td><td rowspan=1 colspan=3>0.3262    0.3392</td><td rowspan=1 colspan=1>0.6439</td><td rowspan=1 colspan=1>0.3047</td><td rowspan=1 colspan=1>0.0160</td></tr><tr><td rowspan=1 colspan=2>Small MLP            16m        0.2869</td><td rowspan=1 colspan=1>0.0027</td><td rowspan=1 colspan=1>0.2841</td><td rowspan=1 colspan=3>0.2898    0.3096</td><td rowspan=1 colspan=1>0.6003</td><td rowspan=1 colspan=1>0.2907</td><td rowspan=1 colspan=1>0.0227</td></tr><tr><td rowspan=1 colspan=2>Small MLP            64m        0.3231</td><td rowspan=1 colspan=1>0.0018</td><td rowspan=1 colspan=1>0.3212</td><td rowspan=1 colspan=3>0.3251    0.3414</td><td rowspan=1 colspan=1>0.6778</td><td rowspan=1 colspan=1>0.3364</td><td rowspan=1 colspan=1>0.0183</td></tr><tr><td rowspan=1 colspan=1>Small MLP</td><td rowspan=1 colspan=1>Exp.       0.3301</td><td rowspan=1 colspan=1>0.0026</td><td rowspan=1 colspan=1>0.3274</td><td rowspan=1 colspan=3>0.3328    0.3423</td><td rowspan=1 colspan=1>0.6304</td><td rowspan=1 colspan=1>0.2881</td><td rowspan=1 colspan=1>0.0122</td></tr><tr><td rowspan=1 colspan=1>Medium MLP</td><td rowspan=1 colspan=1>16m        0.2786</td><td rowspan=1 colspan=1>0.0028</td><td rowspan=1 colspan=1>0.2756</td><td rowspan=1 colspan=3>0.2816    0.3018</td><td rowspan=1 colspan=1>0.6012</td><td rowspan=1 colspan=1>0.2994</td><td rowspan=1 colspan=1>0.0232</td></tr><tr><td rowspan=1 colspan=1>Medium MLP</td><td rowspan=1 colspan=1>64m        0.3189</td><td rowspan=1 colspan=1>0.0037</td><td rowspan=1 colspan=1>0.3151</td><td rowspan=1 colspan=3>0.3228    0.3383</td><td rowspan=1 colspan=1>0.6700</td><td rowspan=1 colspan=1>0.3317</td><td rowspan=1 colspan=1>0.0193</td></tr><tr><td rowspan=1 colspan=1>Medium MLP</td><td rowspan=1 colspan=1>Exp.       0.3270</td><td rowspan=1 colspan=1>0.0036</td><td rowspan=1 colspan=1>0.3232</td><td rowspan=1 colspan=3>0.3308    0.3425</td><td rowspan=1 colspan=1>0.6272</td><td rowspan=1 colspan=1>0.2847</td><td rowspan=1 colspan=1>0.0155</td></tr><tr><td rowspan=1 colspan=2>Random forest        16m        0.3549</td><td rowspan=1 colspan=1>0.0033</td><td rowspan=1 colspan=1>0.3514</td><td rowspan=1 colspan=3>0.3583    0.3705</td><td rowspan=1 colspan=1>0.7320</td><td rowspan=1 colspan=1>0.3615</td><td rowspan=1 colspan=1>0.0156</td></tr><tr><td rowspan=1 colspan=2>Random forest        64m        0.3374</td><td rowspan=1 colspan=1>0.0031</td><td rowspan=1 colspan=1>0.3341</td><td rowspan=1 colspan=3>0.3407    0.3494</td><td rowspan=1 colspan=1>0.7117</td><td rowspan=1 colspan=1>0.3623</td><td rowspan=1 colspan=1>0.0120</td></tr><tr><td rowspan=1 colspan=2>Random forest        Exp.       0.2953</td><td rowspan=1 colspan=1>0.0121</td><td rowspan=1 colspan=4>0.2826  0.3080   0.3079</td><td rowspan=1 colspan=1>0.6036</td><td rowspan=1 colspan=1>0.2957</td><td rowspan=1 colspan=1>0.0126</td></tr><tr><td rowspan=1 colspan=7>Temporal, 20-day context</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Flattened ridge</td><td rowspan=1 colspan=1>16m        0.1397</td><td rowspan=1 colspan=1>0.0048</td><td rowspan=1 colspan=1>0.1346</td><td rowspan=1 colspan=3>0.1447    0.1915</td><td rowspan=1 colspan=1>0.5584</td><td rowspan=1 colspan=1>0.3669</td><td rowspan=1 colspan=1>0.0518</td></tr><tr><td rowspan=1 colspan=1>Flattened ridge</td><td rowspan=1 colspan=1>64m       -0.2128</td><td rowspan=1 colspan=1>0.0076</td><td rowspan=1 colspan=1>-0.2208</td><td rowspan=1 colspan=3>-0.2049   -0.0139</td><td rowspan=1 colspan=1>0.8094</td><td rowspan=1 colspan=1>0.8233</td><td rowspan=1 colspan=1>0.1990</td></tr><tr><td rowspan=1 colspan=1>Flattened ridge</td><td rowspan=1 colspan=1>Exp.      -1.1533</td><td rowspan=1 colspan=1>0.0323</td><td rowspan=1 colspan=1>-1.1871</td><td rowspan=1 colspan=3>-1.1194   -0.3308</td><td rowspan=1 colspan=1>0.8015</td><td rowspan=1 colspan=1>1.1324</td><td rowspan=1 colspan=1>0.8225</td></tr><tr><td rowspan=1 colspan=1>Flattened XGBoost</td><td rowspan=1 colspan=1>16m        0.3026</td><td rowspan=1 colspan=1>0.0033</td><td rowspan=1 colspan=1>0.2992</td><td rowspan=1 colspan=3>0.3061    0.3341</td><td rowspan=1 colspan=1>0.6053</td><td rowspan=1 colspan=1>0.2712</td><td rowspan=1 colspan=1>0.0315</td></tr><tr><td rowspan=1 colspan=1>Flattened XGBoost</td><td rowspan=1 colspan=1>64m        0.3287</td><td rowspan=1 colspan=1>0.0027</td><td rowspan=1 colspan=1>0.3258</td><td rowspan=1 colspan=3>0.3315    0.3485</td><td rowspan=1 colspan=1>0.6602</td><td rowspan=1 colspan=1>0.3116</td><td rowspan=1 colspan=1>0.0198</td></tr><tr><td rowspan=1 colspan=1>Flattened XGBoost</td><td rowspan=1 colspan=1>Exp.       0.3208</td><td rowspan=1 colspan=1>0.0037</td><td rowspan=1 colspan=1>0.3170</td><td rowspan=1 colspan=2>0.3247</td><td rowspan=1 colspan=1>0.3379</td><td rowspan=1 colspan=1>0.6047</td><td rowspan=1 colspan=1>0.2668</td><td rowspan=1 colspan=1>0.0171</td></tr><tr><td rowspan=1 colspan=1>Small flattened MLP</td><td rowspan=1 colspan=1>16m       -0.0841</td><td rowspan=1 colspan=1>0.0081</td><td rowspan=1 colspan=1>-0.0926</td><td rowspan=1 colspan=1>-0.0756</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0200</td><td rowspan=1 colspan=1>0.2220</td><td rowspan=1 colspan=1>0.2020</td><td rowspan=1 colspan=1>0.1041</td></tr><tr><td rowspan=1 colspan=1>Small flattened MLP</td><td rowspan=1 colspan=1>64m        0.0574</td><td rowspan=1 colspan=1>0.0040</td><td rowspan=1 colspan=1>0.0532</td><td rowspan=1 colspan=1>0.0617</td><td rowspan=1 colspan=2>0.1544</td><td rowspan=1 colspan=1>0.4273</td><td rowspan=1 colspan=1>0.2729</td><td rowspan=1 colspan=1>0.0969</td></tr><tr><td rowspan=1 colspan=1>Small flattened MLP</td><td rowspan=1 colspan=1>Exp.       0.1623</td><td rowspan=1 colspan=1>0.0068</td><td rowspan=1 colspan=1>0.1552</td><td rowspan=1 colspan=1>0.1694</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.2485</td><td rowspan=1 colspan=1>0.5360</td><td rowspan=1 colspan=1>0.2875</td><td rowspan=1 colspan=1>0.0862</td></tr><tr><td rowspan=1 colspan=1>Tuned flattened MLP</td><td rowspan=1 colspan=1>16m       -0.0350</td><td rowspan=1 colspan=1>0.0199</td><td rowspan=1 colspan=1>-0.0559</td><td rowspan=1 colspan=1>-0.0141</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0406</td><td rowspan=1 colspan=1>0.1764</td><td rowspan=1 colspan=1>0.1357</td><td rowspan=1 colspan=1>0.0756</td></tr><tr><td rowspan=1 colspan=1>Tuned flattened MLP</td><td rowspan=1 colspan=1>64m        0.1160</td><td rowspan=1 colspan=1>0.0046</td><td rowspan=1 colspan=1>0.1112</td><td rowspan=1 colspan=1>0.1209</td><td rowspan=1 colspan=2>0.1715</td><td rowspan=1 colspan=1>0.3652</td><td rowspan=1 colspan=1>0.1937</td><td rowspan=1 colspan=1>0.0554</td></tr><tr><td rowspan=1 colspan=1>Tuned flattened MLP</td><td rowspan=1 colspan=1>Exp.        0.2066</td><td rowspan=1 colspan=1>0.0027</td><td rowspan=1 colspan=1>0.2038</td><td rowspan=1 colspan=1>0.2095</td><td rowspan=1 colspan=2>0.2607</td><td rowspan=1 colspan=1>0.5001</td><td rowspan=1 colspan=1>0.2394</td><td rowspan=1 colspan=1>0.0540</td></tr><tr><td rowspan=1 colspan=1>Vanilla RNN</td><td rowspan=1 colspan=1>16m        0.2860</td><td rowspan=1 colspan=1>0.0025</td><td rowspan=1 colspan=1>0.2834</td><td rowspan=1 colspan=1>0.2887</td><td rowspan=1 colspan=2>0.3166</td><td rowspan=1 colspan=1>0.5951</td><td rowspan=1 colspan=1>0.2784</td><td rowspan=1 colspan=1>0.0306</td></tr><tr><td rowspan=1 colspan=1>Vanilla RNN</td><td rowspan=1 colspan=1>64m       0.2987</td><td rowspan=1 colspan=1>0.0017</td><td rowspan=1 colspan=1>0.2970</td><td rowspan=1 colspan=1>0.3005</td><td rowspan=1 colspan=2>0.3166</td><td rowspan=1 colspan=1>0.6320</td><td rowspan=1 colspan=1>0.3154</td><td rowspan=1 colspan=1>0.0178</td></tr><tr><td rowspan=1 colspan=1>Vanilla RNN</td><td rowspan=1 colspan=1>Exp.       0.3184</td><td rowspan=1 colspan=1>0.0021</td><td rowspan=1 colspan=1>0.3161</td><td rowspan=1 colspan=1>0.3206</td><td rowspan=1 colspan=2>0.3315</td><td rowspan=1 colspan=1>0.6067</td><td rowspan=1 colspan=1>0.2752</td><td rowspan=1 colspan=1>0.0132</td></tr><tr><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>16m       0.2087</td><td rowspan=1 colspan=1>0.0026</td><td rowspan=1 colspan=1>0.2059</td><td rowspan=1 colspan=1>0.2114</td><td rowspan=1 colspan=2>0.2317</td><td rowspan=1 colspan=1>0.3958</td><td rowspan=1 colspan=1>0.1641</td><td rowspan=1 colspan=1>0.0230</td></tr><tr><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>64m        0.2491</td><td rowspan=1 colspan=1>0.0027</td><td rowspan=1 colspan=1>0.2463</td><td rowspan=1 colspan=1>0.2519</td><td rowspan=1 colspan=2>0.2673</td><td rowspan=1 colspan=1>0.5130</td><td rowspan=1 colspan=1>0.2457</td><td rowspan=1 colspan=1>0.0182</td></tr><tr><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>Exp.       0.2905</td><td rowspan=1 colspan=1>0.0019</td><td rowspan=1 colspan=1>0.2886</td><td rowspan=1 colspan=3>0.2925    0.3071</td><td rowspan=1 colspan=1>0.5439</td><td rowspan=1 colspan=1>0.2368</td><td rowspan=1 colspan=1>0.0166</td></tr><tr><td rowspan=1 colspan=1>Small Transformer</td><td rowspan=1 colspan=1>16m        0.2828</td><td rowspan=1 colspan=1>0.0037</td><td rowspan=1 colspan=1>0.2789</td><td rowspan=1 colspan=3>0.2866    0.3218</td><td rowspan=1 colspan=1>0.6275</td><td rowspan=1 colspan=1>0.3058</td><td rowspan=1 colspan=1>0.0390</td></tr><tr><td rowspan=1 colspan=1>Small Transformer</td><td rowspan=1 colspan=1>64m       0.2995</td><td rowspan=1 colspan=1>0.0022</td><td rowspan=1 colspan=1>0.2972</td><td rowspan=1 colspan=3>0.3018   0.3191</td><td rowspan=1 colspan=1>0.6437</td><td rowspan=1 colspan=1>0.3247</td><td rowspan=1 colspan=1>0.0196</td></tr><tr><td rowspan=1 colspan=1>Small Transformer</td><td rowspan=1 colspan=1>Exp.       0.3193</td><td rowspan=1 colspan=1>0.0020</td><td rowspan=1 colspan=1>0.3172</td><td rowspan=1 colspan=3>0.3213   0.3361</td><td rowspan=1 colspan=1>0.6132</td><td rowspan=1 colspan=1>0.2772</td><td rowspan=1 colspan=1>0.0168</td></tr><tr><td rowspan=1 colspan=2>Temporal, 60-day context</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Flattened ridge</td><td rowspan=1 colspan=1>16m        0.0772</td><td rowspan=1 colspan=1>0.0057</td><td rowspan=1 colspan=1>0.0712</td><td rowspan=1 colspan=3>0.0831    0.1092</td><td rowspan=1 colspan=1>0.2775</td><td rowspan=1 colspan=1>0.1684</td><td rowspan=1 colspan=1>0.0320</td></tr><tr><td rowspan=1 colspan=1>Flattened ridge</td><td rowspan=1 colspan=1>64m        0.1078</td><td rowspan=1 colspan=1>0.0057</td><td rowspan=1 colspan=1>0.1018</td><td rowspan=1 colspan=3>0.1138    0.1654</td><td rowspan=1 colspan=1>0.5507</td><td rowspan=1 colspan=1>0.3853</td><td rowspan=1 colspan=1>0.0576</td></tr><tr><td rowspan=1 colspan=1>Flattened ridge</td><td rowspan=1 colspan=1>Exp.      -0.2835</td><td rowspan=1 colspan=1>0.0106</td><td rowspan=1 colspan=1>-0.2946</td><td rowspan=1 colspan=3>-0.2724  -0.0295</td><td rowspan=1 colspan=1>0.6561</td><td rowspan=1 colspan=1>0.6856</td><td rowspan=1 colspan=1>0.2541</td></tr><tr><td rowspan=1 colspan=1>Flattened XGBoost</td><td rowspan=1 colspan=1>16m        0.2812</td><td rowspan=1 colspan=1>0.0031</td><td rowspan=1 colspan=1>0.2779</td><td rowspan=1 colspan=3>0.2845    0.3161</td><td rowspan=1 colspan=1>0.5640</td><td rowspan=1 colspan=1>0.2479</td><td rowspan=1 colspan=1>0.0350</td></tr><tr><td rowspan=1 colspan=1>Flattened XGBoost</td><td rowspan=1 colspan=1>64m        0.3223</td><td rowspan=1 colspan=1>0.0033</td><td rowspan=1 colspan=1>0.3188</td><td rowspan=1 colspan=3>0.3257    0.3402</td><td rowspan=1 colspan=1>0.6283</td><td rowspan=1 colspan=1>0.2880</td><td rowspan=1 colspan=1>0.0180</td></tr><tr><td rowspan=1 colspan=1>Flattened XGBoost</td><td rowspan=1 colspan=1>Exp.       0.3148</td><td rowspan=1 colspan=1>0.0015</td><td rowspan=1 colspan=1>0.3132</td><td rowspan=1 colspan=3>0.3164    0.3253</td><td rowspan=1 colspan=1>0.5388</td><td rowspan=1 colspan=1>0.2135</td><td rowspan=1 colspan=1>0.0105</td></tr><tr><td rowspan=1 colspan=1>Small flattened MLP</td><td rowspan=1 colspan=1>16m       -0.8216</td><td rowspan=1 colspan=1>0.0599</td><td rowspan=1 colspan=1>-0.8845</td><td rowspan=1 colspan=3>-0.7588   -0.3891</td><td rowspan=1 colspan=1>0.1340</td><td rowspan=1 colspan=1>0.5231</td><td rowspan=1 colspan=1>0.4325</td></tr><tr><td rowspan=1 colspan=1>Small flattened MLP</td><td rowspan=1 colspan=1>64m       -0.0124</td><td rowspan=1 colspan=1>0.0053</td><td rowspan=1 colspan=1>-0.0180</td><td rowspan=1 colspan=3>-0.0068    0.0765</td><td rowspan=1 colspan=1>0.2813</td><td rowspan=1 colspan=1>0.2048</td><td rowspan=1 colspan=1>0.0889</td></tr><tr><td rowspan=1 colspan=1>Small flattened MLP</td><td rowspan=1 colspan=1>Exp.       0.0881</td><td rowspan=1 colspan=1>0.0033</td><td rowspan=1 colspan=1>0.0846</td><td rowspan=1 colspan=3>0.0916    0.1690</td><td rowspan=1 colspan=1>0.4152</td><td rowspan=1 colspan=1>0.2462</td><td rowspan=1 colspan=1>0.0809</td></tr><tr><td rowspan=1 colspan=2>Tuned flattened MLP16m       -0.2223</td><td rowspan=1 colspan=1>0.0119</td><td rowspan=1 colspan=1>-0.2347</td><td rowspan=1 colspan=3>-0.2098   -0.0970</td><td rowspan=1 colspan=1>0.0760</td><td rowspan=1 colspan=1>0.1729</td><td rowspan=1 colspan=1>0.1253</td></tr></table>

Rank Reversals under Temporal Nonstationarity
<table><tr><td>Model</td><td>Window</td><td>Mean  $R ^ { 2 }$ </td><td>SD</td><td>CI low</td><td>CI high</td><td>Ensemble</td><td> $2 a g$ </td><td> $g ^ { 2 }$ </td><td>u</td></tr><tr><td>Tuned flattened MLP</td><td>64m</td><td>0.0002</td><td>0.0035</td><td>-0.0034</td><td>0.0039</td><td>0.0804</td><td>0.2552</td><td>0.1748</td><td>0.0802</td></tr><tr><td>Tuned flattened MLP</td><td>Exp.</td><td>0.1158</td><td>0.0036</td><td>0.1121</td><td>0.1196</td><td>0.1822</td><td>0.4150</td><td>0.2328</td><td>0.0663</td></tr><tr><td>Vanilla RNN</td><td>16m</td><td>0.2776</td><td>0.0028</td><td>0.2747</td><td>0.2806</td><td>0.3123</td><td>0.5788</td><td>0.2665</td><td>0.0347</td></tr><tr><td>Vanilla RNN</td><td>64m</td><td>0.2989</td><td>0.0019</td><td>0.2969</td><td>0.3009</td><td>0.3167</td><td>0.6275</td><td>0.3108</td><td>0.0178</td></tr><tr><td>Vanilla RNN</td><td>Exp.</td><td>0.3169</td><td>0.0024</td><td>0.3144</td><td>0.3194</td><td>0.3276</td><td>0.5841</td><td>0.2565</td><td>0.0107</td></tr><tr><td>LSTM</td><td>16m</td><td>0.1913</td><td>0.0038</td><td>0.1873</td><td>0.1952</td><td>0.2153</td><td>0.3651</td><td>0.1498</td><td>0.0241</td></tr><tr><td>LSTM</td><td>64m</td><td>0.2472</td><td>0.0016</td><td>0.2455</td><td>0.2489</td><td>0.2661</td><td>0.5065</td><td>0.2404</td><td>0.0189</td></tr><tr><td>LSTM</td><td>Exp.</td><td>0.2885</td><td>0.0009</td><td>0.2876</td><td>0.2895</td><td>0.3042</td><td>0.5301</td><td>0.2259</td><td>0.0156</td></tr><tr><td>Small Transformer</td><td>16m</td><td>0.2712</td><td>0.0052</td><td>0.2658</td><td>0.2766</td><td>0.3191</td><td>0.6203</td><td>0.3011</td><td>0.0479</td></tr><tr><td>Small Transformer</td><td>64m</td><td>0.2969</td><td>0.0007</td><td>0.2961</td><td>0.2976</td><td>0.3160</td><td>0.6315</td><td>0.3155</td><td>0.0191</td></tr><tr><td>Small Transformer</td><td>Exp.</td><td>0.3195</td><td>0.0016</td><td>0.3178</td><td>0.3212</td><td>0.3342</td><td>0.5948</td><td>0.2606</td><td>0.0147</td></tr></table>