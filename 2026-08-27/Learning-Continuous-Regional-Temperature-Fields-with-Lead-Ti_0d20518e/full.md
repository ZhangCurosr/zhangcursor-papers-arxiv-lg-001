# Learning Continuous Regional Temperature Fields with Lead-Time and Resolution Queries

Chunlei Shi, Jiong Wang, Yi-Lin Wei, Junming Hou, Jinjin Liu, Yecheng Zhang, and Dan Niu, Member, IEEE

Abstract—Accurate regional near-surface temperature forecasting is fundamental to short-range weather services and downstream risk assessment. Existing deep learning-based regional forecasters commonly produce a fixed set of future frames on a prescribed grid, limiting their use when forecast products must be evaluated at query-dependent lead times or display resolutions. To overcome these fixed-output constraints, we formulate regional T2M forecasting as query-conditioned continuous spatiotemporal temperature field evaluation and propose the Continuous Spatiotemporal Temperature Forecaster (CSTF), a neural field that turns forecast lead time and output resolution into explicit queries when evaluating 2-m temperature (T2M). Specifically, CSTF first encodes multi-variable ERA5 histories into latent meteorological states and then decodes T2M as a coordinate-based field. Accordingly, spatial location, forecast lead time, and output resolution are introduced as queries, enabling standard hourly forecasts, intermediate leadtime diagnostics, and resolution-controllable outputs within a unified field-evaluation framework. Furthermore, to maintain coherence across flexible field queries, we design spatial-gradient, temporal-difference, and scale-consistency objectives that regularize regional thermal structures, lead-wise evolution, and crossresolution agreement. Experiments on the Southeast China 0–6 h ERA5-Land benchmark demonstrate that CSTF achieves the best aggregate deterministic skill, including a 17.0% reduction in Bias, with global-scope diagnostics further illustrating flexible lead-time and resolution-controllable inference.

Index Terms—Near-surface temperature forecasting, continuous spatiotemporal fields, neural fields, query-conditioned forecasting, multi-resolution prediction, ERA5-Land.

## I. INTRODUCTION

Accurate regional near-surface temperature forecasting supports weather services and decisions in electricity demand management, agricultural planning, public health warning, and urban heat-risk assessment [1, 2]. Near-surface temperature evolves continuously under advection, radiation, cloud cover, land–sea contrast, and surface heterogeneity, so local gradients can be as important as domain-mean error for regional use [3–5]. Accordingly, practical forecast products are not limited to a fixed set of hourly maps on a single grid, but often involve intermediate horizons, coarse-resolution diagnostics, and denser-grid outputs for visualization or downstream coupling.

Data-driven meteorological models have made rapid progress in learning structured atmospheric fields from largescale gridded data [6–8]. Yet most regional neural forecasters still expose a discrete product interface, either emitting a fixed stack of future frames or advancing the state autoregressively at a predefined time step [6–8]. Such designs are natural for benchmark evaluation, but they make lead time and output resolution architectural choices rather than query variables. A fixed multi-head or sequence-to-sequence model can return the frames it was built to produce, while a request such as t + 0.5 h at 192 × 192 resolution lies outside its native output space (Fig. 1a).

Autoregressive rollout can extend the forecast horizon, but it does not by itself provide dense-time field evaluation and may accumulate errors as predictions are fed back into the model (Fig. 1b) [7, 9]. Likewise, post-hoc resizing or superresolution can change the apparent grid after a forecast is generated, but it does not ensure that products sampled at different resolutions describe the same underlying temperature state (Fig. 1c) [10–13]. In all these cases, the available lead times and output grid are fixed by the training target or by the final prediction head. Increasing the number of supervised frames or training separate models for different resolutions can enlarge the product catalogue, but the model still predicts among predefined outputs rather than evaluating a requested field.

This limitation motivates the central question of this work: can regional temperature forecasting be formulated as continuous spatiotemporal field evaluation [14, 15]? Instead of asking a network to emit a finite list of images, we seek a single field evaluator that maps the same encoded atmospheric state to T2M values at requested spatial coordinates, lead times, and output grids. Under this view, time and resolution are part of the model input, and cross-resolution consistency becomes a property that can be trained and diagnosed explicitly.

To this end, we propose the Continuous Spatiotemporal Temperature Forecaster (CSTF), a query-conditioned neural field for regional 2-m temperature prediction (Fig. 1d). CSTF first encodes multi-variable ERA5 histories into a shared latent meteorological state, so forecast products at different times and grids are derived from the same atmospheric context. A coordinate decoder then evaluates T2M fields from explicit spatial-coordinate, lead-time, and resolution queries, enabling hourly forecasts, intermediate lead-time queries, <sub>Lead-tim</sub>and resolution-controllable outputs within one field-evaluation framework. Unlike fixed prediction heads or separate scalespecific models, this design makes the requested lead time and grid part of the prediction process itself. To make these flexible queries meteorologically coherent rather than merely interpolated products, we regularize regional thermal gradients, leadwise temperature changes, and cross-resolution agreement during training. We evaluate CSTF on a Southeast China 0–6 h T2M forecasting benchmark and further use global-scope diagnostics to examine the same query interface beyond the regional test domain. Our contributions are summarized as follows:

![](images/fe3cea0f5bd8eb82e297c592d114651e34bd7e170f12819379e639bdcdee73b3.jpg)  
Continuous query planeFig. 1: Comparison between conventional fixed-output and proposed query-conditioned temperature forecasting paradigms. (a) 192× 192Fixed multi-head models predict predefined lead times and output resolutions. (b) Autoregressive rollout extends forecasts step by step but can accumulate errors across forecast steps. (c) Post-hoc resizing changes the display grid without enforcing 96× 96 <sup>160×</sup> <sup>160</sup>consistency across resolutions. (d) Our CSTF formulates regional T2M prediction as continuous field evaluation conditioned on spatial, lead-time, and resolution queries, enabling flexible forecast requests.

• We reformulate regional T2M forecasting as queryconditioned continuous spatiotemporal field evaluation, where lead time and output resolution are treated as inputs rather than fixed output dimensions.

• We propose CSTF, a shared-latent-state and coordinatedecoding architecture that directly evaluates regional temperature fields at requested spatial locations, forecast lead times, and output grids.

• We develop coherence-preserving supervision and diagnostics for query-conditioned temperature fields, validating fixed-lead skill, continuous lead-time queryability, resolution control, and cross-resolution consistency.

## II. RELATED WORK

a) Data-driven weather prediction.: Machine learning is now widely used for weather and Earth-system prediction. WeatherBench established a benchmark for data-driven global forecasting [16], while FourCastNet, Pangu-Weather, Graph-Cast, and FengWu demonstrated skilful high-dimensional forecasts at low inference cost [6–8, 17]. At shorter ranges and regional scales, EarthFormer, wavelet-driven radar retrieval, and attention-based convolutional models capture useful evolution patterns from multi-channel meteorological fields [18, 19]. However, most systems still use a fixed-output interface: input history is mapped to predefined variables, lead times, and grids. Our work addresses this limitation for regional T2M prediction by making lead time and output resolution query variables.

b) Regional temperature prediction.: Regional nearsurface temperature prediction is a structured short-range refinement problem governed by synoptic forcing, advection, radiation, cloud cover, land–sea contrast, topography, and surface heterogeneity [3, 4]. ERA5 provides dynamically coherent atmospheric context, whereas ERA5-Land resolves regional thermal gradients relevant to local decisions [20, 21]. Convolutional encoder–decoders, transformer forecasters, and super-resolution models map coarse or multi-variable inputs to regional targets, but their lead times and grids are usually fixed during training [5, 10, 11, 18]. Post-hoc interpolation changes the display grid without ensuring cross-resolution agreement for the same temperature state [12, 13]; CSTF instead keeps supervised ERA5-Land verification while adding inferencetime control over lead time and resolution.

c) Continuous and coordinate-based modelling.: Recent weather-AI models relax fixed temporal interfaces by treating evolution as continuous or adapting rollout intervals [9, 14, 22]. Temporal interpolation and temporal superresolution have also been explored for radar precipitation and atmospheric fields [23–25]. Coordinate-based neural fields and operator-learning methods further represent signals or solution maps as functions evaluable at requested locations or discretizations [26, 27]. Together, these ideas motivate querybased meteorological prediction, yet temporal flexibility can leave the grid fixed, and coordinate flexibility alone does not ensure cross-resolution coherence. CSTF couples leadtime, coordinate, and resolution conditioning for regional T2M forecasting, using scale-consistency training and diagnostics to separate controllable inference from post-processing.

## A. Problem definition

Given a regional domain Ω discretized on a base grid of size $H \times W$ , each forecast initialization provides a multi-variable

![](images/7b3f2af5f97f7ad14a028576cf1d302b5006f71013c41d134374e9b66ec23c76.jpg)  
Fig. 2: Overall architecture of the proposed CSTF framework. The model encodes ERA5 multi-variable input history into a latent regional weather state within a unified query-conditioned forecasting pipeline. Conditioned on this latent state, a coordinate decoder predicts regional T2M fields from lead-time, spatial-coordinate, and output-resolution queries. Training combines deterministic, gradient, temporal-difference, and scale-consistency objectives, so the same trained model supports continuous lead-time queries, resolution-controllable outputs, and scale-consistency evaluation.

ERA5 history

$$
\mathbf { X } \in \mathbb { R } ^ { B \times T _ { \mathrm { i n } } \times C _ { \mathrm { i n } } \times H \times W } ,\tag{1}
$$

where B is the batch size, $T _ { \mathrm { i n } }$ denotes the number of historical frames, and $C _ { \mathrm { i n } } = 9$ corresponds to

$$
\{ \mathrm { t 2 m , d 2 m , u 1 0 , v 1 0 , m s l , c p , l s p , t c c , s s r d } \} .\tag{2}
$$

The supervised target is the future ERA5-Land T2M sequence $\textbf { Y } \in \overset { \bullet } { \mathbb { R } } ^ { B \times T _ { \mathrm { o u t } } \times 1 \times H \times W }$ , defined at hourly lead times for training and standard evaluation.

The standard supervised formulation learns a fixed-output sequence mapping

$$
\mathbf { X } \mapsto \{ \mathbf { Y } _ { 1 } , \dots , \mathbf { Y } _ { T _ { \mathrm { o u t } } } \} ,\tag{3}
$$

where the available lead times and output grid are determined by the target sequence and prediction head. Such a predictor can only return members of its predefined product set, and therefore cannot natively evaluate the temperature field at an intermediate lead time or an alternative grid. To expose lead time and output resolution as query variables, we instead formulate regional near-surface temperature forecasting as query-conditioned spatiotemporal field evaluation:

$$
\widehat { \mathbf { Y } } _ { \tau } ^ { ( H _ { o } , W _ { o } ) } = \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { X } ; \tau , H _ { o } , W _ { o } ) ,\tag{4}
$$

where τ denotes the requested lead time and $( H _ { o } , W _ { o } )$ denotes the requested output grid. The objective is to learn a single function $\mathcal { F } _ { \theta }$ that remains accurate at supervised integer leads while supporting diagnostic queries at non-integer lead times and multiple output resolutions.

## B. Overview

To implement this field-evaluation formulation, merely appending lead-time and resolution inputs to a fixed-grid forecaster is insufficient. If forecast products at different horizons or grids are generated by independent heads, rollout steps, or post-processing, their differences need not correspond to evaluations of a common thermal evolution. Moreover, simply resizing a prediction can change its display grid without informing the model of the requested sampling density, and pointwise supervision at hourly base-grid targets does not constrain non-integer lead queries or cross-resolution agreement. Our key insight is that query flexibility must be coupled with state sharing and consistency regularization. Accordingly, CSTF uses three coordinated design choices: a shared latent meteorological state, an explicit query-conditioned coordinate decoder, and coherence-preserving objectives for space, time, and scale. As illustrated in Fig. 2, the framework follows the mapping $\textbf { X } \xrightarrow { \mathcal { E } _ { \theta } } \textbf { Z } \xrightarrow { \mathcal { D } _ { \theta } \left( \tau , \mathbf { P } _ { H _ { o } , W _ { o } } , \mathbf { R } _ { H _ { o } , W _ { o } } \right) } \widehat { \textbf { Y } } _ { \tau } ^ { \left( H _ { o } , W _ { o } \right) }$ . Specifically, a multi-scale meteorological encoder transforms the multi-variable ERA5 history X into a latent regional weather state Z. The requested lead time τ, output coordinates $\mathbf { P } _ { H _ { o } , W _ { o } } ,$ and grid resolution ${ \mathbf { R } } _ { H _ { o } , W _ { o } }$ then condition a coordinate decoder, so each forecast product is obtained by evaluating the same state under a specified query. Training combines deterministic, spatial-gradient, temporal-difference, and scaleconsistency objectives to anchor supervised accuracy while regularizing regional thermal contrast, lead-wise evolution, and cross-resolution agreement.

## C. Query-conditioned field model

We formalize CSTF as a neural field conditioned on both the encoded atmospheric state and the requested forecast product. This shared-state formulation prevents the query interface from degenerating into separate product-specific mappings. The encoder first summarizes the input history into a shared latent regional state, after which the decoder evaluates this state at each requested lead time and grid cell:

$$
\mathbf { Z } = { \mathcal { E } } _ { \theta } ( \mathbf { X } ) ,
$$

$$
\begin{array} { r } { \widehat { \mathbf { Y } } _ { \tau } ^ { ( H _ { o } , W _ { o } ) } ( i , j ) = \mathcal { D } _ { \boldsymbol { \theta } } ( \mathbf { Z } _ { H _ { o } , W _ { o } } ( i , j ) , \phi ( \tau ) , \mathbf { p } _ { i , j } , \mathbf { r } _ { H _ { o } , W _ { o } } ) , } \end{array}\tag{5}
$$

where Z denotes the latent regional weather state, $\mathbf { Z } _ { H _ { o } , W _ { o } }$ is its interpolation on the requested grid, $\phi ( \tau )$ is a sinusoidal lead-time embedding, $\mathbf { p } _ { i , j } ~ \in ~ [ - 1 , 1 ] ^ { 2 }$ is the normalized coordinate of grid cell $( i , j )$ , and $\mathbf { r } _ { H _ { o } , W _ { o } } = ( 1 / H _ { o } , 1 / W _ { o } )$ encodes the requested sampling density. This formulation makes the output grid an explicit part of the decoder input, enabling different spatial discretizations to be treated as direct evaluations of the same atmospheric state.

## D. Multi-scale meteorological encoder

The encoder serves as the state-construction component of CSTF, rather than a product-specific prediction head. Because this state is later queried across lead times and resolutions, it must preserve regional thermal gradients while aggregating broader wind, pressure, moisture, cloud, and radiation cues from the multi-variable ERA5 history. We instantiate this state encoder as an encoder–decoder backbone with skip aggregation, leaving Z independent of any fixed lead time or output resolution and available for query-specific field evaluation by the coordinate decoder.

## E. Lead-time and resolution queries

A forecast query in CSTF specifies when the temperature field is evaluated, where it is sampled, and at what output density it is requested. For the temporal component, CSTF normalizes the requested lead τ by the maximum training horizon and maps it to a learnable temporal code:

$$
\phi ( \tau ) = \mathrm { M L P } \left( \operatorname { S i n C o s } \left( \frac { \tau } { \tau _ { \operatorname* { m a x } } } \right) \right) .\tag{6}
$$

The sinusoidal basis provides a smooth temporal representation, and the MLP adapts it to short-range temperature evolution. The same embedding function is used for supervised integer leads and non-integer diagnostic queries such as 0.5 h or 4.5 h.

For the spatial component, coordinates alone identify where to evaluate the field, but they do not indicate whether the field is being sampled sparsely or densely. We therefore encode the requested grid by both sampling locations and sampling density. The first field is the normalized coordinate grid

$$
\mathbf { P } _ { H _ { o } , W _ { o } } = \{ \mathbf { p } _ { i , j } = ( x _ { j } , y _ { i } ) : x _ { j } , y _ { i } \in [ - 1 , 1 ] \} ,\tag{7}
$$

which specifies the location of each output cell within the regional domain. The resolution field is constant over the output image:

$$
{ \bf R } _ { H _ { o } , W _ { o } } ( i , j ) = ( 1 / H _ { o } , 1 / W _ { o } ) .\tag{8}
$$

Together, $\phi ( \tau )$ $\mathbf { P } _ { H _ { o } , W _ { o } }$ , and ${ \mathbf { R } } _ { H _ { o } , W _ { o } }$ define the requested forecast product passed to the decoder.

## F. Coordinate decoder

The coordinate decoder evaluates the shared weather state under the requested product query. A conventional convolutional prediction head maps latent features to a fixed grid and is unaware of absolute sampling location or requested grid density, making resolution changes indistinguishable from post-hoc resizing. We instead interpolate Z to the requested grid and condition the decoder on local latent features, spatial coordinates, resolution code, and lead-time embedding:

$$
\widehat { \mathbf { Y } } _ { \tau } ^ { ( H _ { o } , W _ { o } ) } = \mathcal { D } _ { \boldsymbol { \theta } } [ \mathbf { Z } _ { H _ { o } , W _ { o } } , \mathbf { P } _ { H _ { o } , W _ { o } } , \mathbf { R } _ { H _ { o } , W _ { o } } , \boldsymbol { \phi } ( \tau ) ] .\tag{9}
$$

In implementation, $\mathcal { D } _ { \theta }$ is a lightweight convolutional decoder applied on the requested grid. Because query channels are part of the decoder input, each predicted value depends on local meteorological features, spatial position, forecast lead time, and sampling scale. Thus, outputs such as $9 6 \times 9 6 , 1 2 8 \times 1 2 8$ and $1 9 2 \times 1 9 2$ fields are generated by direct field evaluation rather than by resizing a fixed-grid forecast.

## G. Training objective

Training uses hourly lead times $\{ \tau _ { k } \} _ { k = 1 } ^ { T _ { \mathrm { o u t } } }$ with available ERA5-Land references. The supervised forecast loss anchors CSTF to observed ERA5-Land targets:

$$
\mathcal { L } _ { \mathrm { f o r e c a s t } } = \frac { 1 } { T _ { \mathrm { o u t } } } \sum _ { k = 1 } ^ { T _ { \mathrm { o u t } } } \mathrm { M S E } \Big ( \widehat { \mathbf { Y } } _ { \tau _ { k } } ^ { ( H , W ) } , \mathbf { Y } _ { \tau _ { k } } ^ { ( H , W ) } \Big ) .\tag{10}
$$

However, pointwise hourly supervision alone does not guarantee coherent behavior when the same field evaluator is queried across resolutions or adjacent lead times. We therefore add three regularizers for scale agreement, regional thermal structure, and lead-wise evolution. The scale-consistency loss matches a direct low-resolution query with an areadownsampled base-grid query, encouraging different resolution evaluations of the same atmospheric state to agree after aggregation:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s c a l e } } = \displaystyle \frac { 1 } { T _ { \mathrm { o u t } } } \sum _ { k = 1 } ^ { T _ { \mathrm { o u t } } } \mathrm { M S E } \Big ( \mathcal { F } _ { \theta } ( \mathbf { X } ; \tau _ { k } , H / 2 , W / 2 ) , } \\ { \mathrm { A r e a D o w n } \left( \mathcal { F } _ { \theta } ( \mathbf { X } ; \tau _ { k } , H , W ) \right) \Big ) , } \end{array}\tag{11}
$$

where AreaDown denotes area downsampling from $( H , W )$ to $( H / 2 , W / 2 )$

The spatial-gradient loss discourages oversmoothing by matching finite-difference T2M gradients:

$$
\begin{array} { r l r }   { \mathcal { L } _ { \mathrm { g r a d } } = \frac { 1 } { T _ { \mathrm { o u t } } } \sum _ { k = 1 } ^ { T _ { \mathrm { o u t } } } \Bigl ( \| \nabla _ { x } \widehat { \mathbf { Y } } _ { \tau _ { k } } - \nabla _ { x } { \mathbf { Y } } _ { \tau _ { k } } \| _ { 1 } } \\ & { } & { + \| \nabla _ { y } \widehat { \mathbf { Y } } _ { \tau _ { k } } - \nabla _ { y } { \mathbf { Y } } _ { \tau _ { k } } \| _ { 1 } \Bigr ) . } \end{array}\tag{12}
$$

TABLE I: Lead-wise quantitative comparison for short-range T2M forecasting over Southeast China. RMSE and MAE are reported in degrees Celsius at each supervised forecast lead, where lower values indicate better accuracy. The best result in each column is highlighted in green, and the second-best result is underlined.
<table><tr><td rowspan="2">Model</td><td colspan="2">t+6 h</td><td colspan="2">t+5 h</td><td colspan="2">t+4 h</td><td colspan="2">t+3 h</td><td colspan="2"> $^ { \mathrm { t } + 2 \mathrm { ~ h ~ } }$ </td><td colspan="2">t+1 h</td></tr><tr><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td></tr><tr><td>Baseline</td><td>8.196</td><td>3.741</td><td>7.992</td><td>3.378</td><td>7.782</td><td>2.968</td><td>7.583</td><td>2.529</td><td>7.419</td><td>2.108</td><td>7.309</td><td>1.779</td></tr><tr><td>SMAAT-UNet [28]</td><td>1.989</td><td>1.425</td><td>1.785</td><td>1.270</td><td>1.581</td><td>1.141</td><td>1.347</td><td>1.000</td><td>1.106</td><td>0.840</td><td>1.000</td><td>0.765</td></tr><tr><td>EarthFormer [18]</td><td>2.784</td><td>1.254</td><td>2.668</td><td>1.154</td><td>2.600</td><td>1.064</td><td>2.590</td><td>0.990</td><td>2.600</td><td>0.939</td><td>2.615</td><td>0.902</td></tr><tr><td>ARROW [9]</td><td>1.625</td><td>1.131</td><td>1.444</td><td>1.015</td><td>1.246</td><td>0.892</td><td>1.051</td><td>0.768</td><td>0.870</td><td>0.655</td><td>0.739</td><td>0.567</td></tr><tr><td>Weather-RF [29]</td><td>1.609</td><td>1.112</td><td>1.446</td><td>1.008</td><td>1.284</td><td>0.908</td><td>1.126</td><td>0.812</td><td>0.970</td><td>0.720</td><td>0.853</td><td>0.647</td></tr><tr><td>Ours</td><td>1.414</td><td>1.014</td><td>1.284</td><td>0.925</td><td>1.162</td><td>0.845</td><td>1.036</td><td>0.765</td><td>0.909</td><td>0.685</td><td>0.823</td><td>0.631</td></tr></table>

TABLE II: Overall quantitative comparison for integer-lead T2M forecasting over Southeast China. Lower is better for RMSE, MAE, and bias magnitude, while higher is better for correlation.
<table><tr><td>Model</td><td>RMSE↓</td><td>MAE↓</td><td>Bias↓</td><td>Corr. ↑</td></tr><tr><td>Baseline</td><td>7.720</td><td>2.751</td><td>1.247</td><td>0.576</td></tr><tr><td>SMAAT-UNet [28]</td><td>1.510</td><td>1.074</td><td>0.329</td><td>0.987</td></tr><tr><td>EarthFormer [18]</td><td>2.644</td><td>1.051</td><td>0.598</td><td>0.963</td></tr><tr><td>ARROW [9]</td><td>1.203</td><td>0.838</td><td>0.389</td><td>0.992</td></tr><tr><td>Weather-RF [29]</td><td>1.243</td><td>0.868</td><td>0.446</td><td>0.992</td></tr><tr><td>Ours</td><td>1.123</td><td>0.811</td><td>0.273</td><td>0.993</td></tr><tr><td>Improv. vs. Second</td><td>+6.7%</td><td>+3.2%</td><td>+17.0%</td><td>+0.1%</td></tr></table>

The temporal-difference loss constrains short-range evolution by aligning frame-to-frame temperature changes:

$$
\mathcal { L } _ { \mathrm { t e m p } } = \frac { 1 } { T _ { \mathrm { o u t } } - 1 } \sum _ { k = 1 } ^ { T _ { \mathrm { o u t } } - 1 } \| ( \widehat { \mathbf { Y } } _ { \tau _ { k + 1 } } - \widehat { \mathbf { Y } } _ { \tau _ { k } } )\tag{13}
$$

The overall training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f o r e c a s t } } + \lambda _ { g } \mathcal { L } _ { \mathrm { g r a d } } + \lambda _ { t } \mathcal { L } _ { \mathrm { t e m p } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s c a l e } } .\tag{14}
$$

## H. Inference and diagnostic queries

After training, inference is carried out by querying the same evaluator $\mathcal { F } _ { \theta }$ with specified lead-time and grid-size arguments. Integer leads are verified against ERA5-Land using deterministic forecast metrics. Non-integer leads are reported as forecastonly diagnostics because the current hourly target split has no direct half-hour references. For resolution-controllable inference, predictions at multiple output sizes are downsampled to a common base grid to quantify scale consistency. This diagnostic does not replace verification against ERA5-Land; it tests whether the query-conditioned decoder returns mutually coherent fields when the same atmospheric state is sampled at different spatial resolutions.

## III. EXPERIMENTS

## A. Implementation Details

Experimental Setup. We evaluate CSTF on short-range 2-m temperature forecasting from ERA5 to ERA5-Land. Each sample uses a three-hour, nine-variable ERA5 history to predict six hourly future ERA5-Land T2M fields. The main benchmark is conducted over Southeast China at a base grid of $1 2 8 \times 1 2 8$ and a global-scope split is used only for qualitative query diagnostics. Detailed data splits, preprocessing, optimization, objective weights, checkpoint selection, and query-evaluation protocols are provided in Appendix.

Evaluation Metrics. Following meteorological-state evaluation protocols [30], we report MAE and RMSE in physical units for supervised integer leads. We also use bias and spatial correlation for regional temperature diagnostics, and crossresolution scale-consistency error for resolution-controllable prediction. We define the reported scale-consistency error in normalized model space as

$$
\begin{array} { r } { \pmb { \mathcal { E } } _ { \mathrm { s c a l e } } ( r ) = \mathrm { R M S E } \Big ( \mathrm { R e s a m p l e } _ { r  r _ { 0 } } ^ { \mathrm { a r e a } } ( \widehat { \mathbf { Y } } _ { \tau } ^ { ( r , r ) } ) , \widehat { \mathbf { Y } } _ { \tau } ^ { ( r _ { 0 } , r _ { 0 } ) } \Big ) , } \end{array}\tag{15}
$$

where $r _ { 0 } = 1 2 8$ is the base resolution and Resample<sup>area</sup><sub>r→r</sub> maps each query to the base grid. This diagnostic compares predictions across query grids rather than measuring physicalunit error against ERA5-Land references.

Baseline Comparisons. We compare CSTF with baseline, SMAAT-UNet [28], EarthFormer [18], ARROW-style forecasting [9], and Weather-RF/FREUD-style forecasting [29]. All learned baselines follow the same data, input–output, normalization, and evaluation protocol; controlled CSTF variants only remove selected objective terms.

## B. Regional deterministic forecast

Before analyzing flexible query behavior, we first verify the deterministic forecasting skill of CSTF at the supervised hourly leads on the Southeast China benchmark.

Table I reports the lead-wise comparison. Although AR-ROW attains slightly lower errors at the first two leads, CSTF achieves the best skill from t+3 h to t+6 h, indicating more stable performance as short-range evolution becomes less constrained by the input state.

The aggregate results in Table II further show that CSTF obtains the lowest average RMSE, MAE, and bias, together with the highest spatial correlation among the compared methods. Relative to the strongest fixed-grid baseline, ARROW-style forecasting, CSTF reduces RMSE from $1 . 2 0 3 ^ { \circ } \mathrm { C }$ to $1 . 1 2 3 ^ { \circ } \mathrm { C }$ and MAE from 0.838<sup>◦</sup>C to $0 . 8 1 1 ^ { \circ } \mathrm { C } .$ The representative case in Fig. 3 shows that this improvement is accompanied by coherent regional thermal structures, with larger errors mainly confined to localized areas.

![](images/e48b01a6f8af3f3a1a4fd7c087acaf91105dc31a3c51b9acd08399832d65c540.jpg)  
Fig. 3: Qualitative comparison of deterministic T2M forecasts over the Southeast China benchmark. Panels show the ERA5 input context, integer-lead CSTF predictions, ERA5-Land references, and absolute-error maps.

## C. Global diagnostic visualization

To assess whether the query formulation preserves coherent large-scale thermal structures beyond the regional benchmark, we further apply CSTF to the global-scope split as a qualitative diagnostic. Figure 4 presents the input context, CSTF forecasts, ERA5-Land references, and absolute-error maps under a Robinson projection with polar insets. This visualization is used to inspect large-scale thermal coherence and spatial error patterns, while all quantitative comparisons are still computed on the native model grid.

## D. Continuous lead-time query

Because CSTF represents lead time as an explicit query variable, we test whether the learned field evaluator can produce

![](images/8a47bb21d45345e13d7c2ecbe8fa7b5ab4e738940b14c9382516825a84989d9c.jpg)  
Fig. 4: Global-scope diagnostic visualization of T2M field prediction. Panels compare ERA5-Land references, CSTF predictions, and absolute-error maps under a Robinson projection with polar insets. Zoomed-in views are provided in Appendix.

temporally ordered forecasts beyond the hourly supervision grid. The temporal queries include both verified integer leads and diagnostic non-integer leads:

$$
\tau \in \{ 0 . 5 , 2 , 4 . 5 , 6 \} \ \mathrm { h o u r s } .\tag{16}
$$

The integer leads are compared with ERA5-Land targets, whereas the half-hour queries are reported as forecast-only diagnostics because the target split is hourly. Figure 5 shows that regional temperature structures evolve smoothly across the queried lead times, and Fig. 6 provides the corresponding global diagnostic view.

## E. Resolution-controllable prediction

We then assess spatial queryability by evaluating the learned field decoder on output grids that differ from the base training resolution:

$$
( H _ { o } , W _ { o } ) \in \{ 9 6 \times 9 6 , 1 2 8 \times 1 2 8 , 1 6 0 \times 1 6 0 , 1 9 2 \times 1 9 2 \} _ { \mathrm { t } }\tag{17}
$$

Figures 7 and 8 show that CSTF directly decodes temperature fields at the requested grids, rather than resizing a fixed-grid prediction. The resulting panels preserve the main thermal patterns across query resolutions, supporting the intended resolution-controllable inference behavior.

## F. Scale-consistency analysis

Resolution-controllable visualization alone does not ensure compatible temperature states across grids. We therefore aggregate non-base queries to the reference grid and compare them with the base-grid prediction. The resulting diagnostic, reported in Table IV, isolates the resolution-query training signal under controlled ablations.

## G. Ablation Studies

We evaluate two variants: w/o aux. losses keeps only $\mathcal { L } _ { \mathrm { f o r e c a s t } }$ , while w/o $\mathcal { L } _ { \mathrm { s c a l e } }$ retains $\mathcal { L } _ { \mathrm { g r a d } }$ and $\mathcal { L } _ { \mathrm { t e m p } }$ but removes the scale-consistency term. As shown in Table III, the full objective improves RMSE and MAE across all supervised leads, with larger gains at longer horizons, indicating

TABLE III: Extended lead-wise ablation of CSTF objective terms on the Southeast China test split. RMSE and MAE are reported in degrees Celsius at each supervised hourly lead; lower is better. This table extends the main-text lead-wise ablation by separately removing the spatial-gradient and temporal-difference regularizers. The best score in each column is highlighted in green, and the second-best score is underlined.
<table><tr><td rowspan="2">Variant</td><td colspan="2">t+6 h</td><td colspan="2">t+5 h</td><td colspan="2">t+4 h</td><td colspan="2">t+3 h</td><td colspan="2">t+2 h</td><td colspan="2">t+1 h</td><td colspan="2">Mean</td></tr><tr><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td></tr><tr><td>w/o aux. losses</td><td>1.555</td><td>1.085</td><td>1.403</td><td>0.984</td><td>1.248</td><td>0.890</td><td>1.096</td><td>0.800</td><td>0.960</td><td>0.720</td><td>0.871</td><td>0.668</td><td>1.213</td><td>0.858</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { s c a l e } }$ </td><td>1.575</td><td>1.096</td><td>1.422</td><td>0.996</td><td>1.263</td><td>0.898</td><td>1.100</td><td>0.802</td><td>0.949</td><td>0.712</td><td>0.852</td><td>0.654</td><td>1.220</td><td>0.860</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { g r a d } }$ </td><td>1.581</td><td>1.106</td><td>1.420</td><td>1.003</td><td>1.264</td><td>0.907</td><td>1.116</td><td>0.817</td><td>0.974</td><td>0.730</td><td>0.875</td><td>0.669</td><td>1.230</td><td>0.872</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { t e m p } }$ </td><td>1.476</td><td>1.043</td><td>1.337</td><td>0.948</td><td>1.203</td><td>0.861</td><td>1.059</td><td>0.773</td><td>0.921</td><td>0.689</td><td>0.831</td><td>0.636</td><td>1.160</td><td>0.825</td></tr><tr><td>Ours</td><td>1.414</td><td>1.014</td><td>1.284</td><td>0.925</td><td>1.162</td><td>0.845</td><td>1.036</td><td>0.765</td><td>0.909</td><td>0.685</td><td>0.823</td><td>0.631</td><td>1.123</td><td>0.811</td></tr></table>

![](images/e2ff83850a1562fadc1826069c62368f4612f0b46795fbbc974693161f440fbf.jpg)  
Fig. 5: Continuous lead-time query. The same trained model is queried at 0.5, 2, 4.5, and 6 hours. The 0.5 h and 4.5 h panels are non-integer forecast-only queries, while the 2 h and 6 h panels align with supervised hourly targets.

![](images/297b9be2319218d53c0dbea2b1d07c519dee92188e1fb7c4474e9d617fe49eaf.jpg)  
Fig. 7: Resolution-controllable prediction. Fixed input and lead-time query are decoded at 96×96, 128×128, 160×160, and 192×192 grids. Panels show direct resolution-conditioned decoding rather than post-hoc resizing.

![](images/0fa8b288357e33e276be3458a2bdef566152ba39779976b5473c88703afd715b.jpg)  
Fig. 6: Global continuous lead-time query on a Robinson projection with polar insets. The same checkpoint is queried at 0.5, 2, 4.5, and 6 hours, using the same lead-time set as the regional query visualization.

![](images/e0220b023eba846a54e3053edf0c5351996dae24418519bfc58b6f60bc2f26b8.jpg)  
Fig. 8: Global resolution-controllable prediction. Orthographic visualizations are generated for different output grid sizes from the same input state and lead-time query.

more stable lead-wise temperature evolution. Table IV shows that removing $\mathcal { L } _ { \mathrm { s c a l e } }$ increases cross-resolution error after aggregation, supporting its role in coherent resolution-query prediction. Table V further indicates that the full objective provides balanced gains in aggregate accuracy, thermal-structure fidelity, and scale coherence.  
TABLE IV: Scale-consistency ablation for resolutioncontrollable prediction. Values are normalized cross-resolution RMSE after resampling to $1 2 8 \times 1 2 8 ;$ lower is better. Best results are highlighted in green, and second-best results are underlined.
<table><tr><td>Variant</td><td>96 → 128</td><td>160 → 128</td><td>192 → 128</td><td>Mean</td></tr><tr><td>w/o aux. losses</td><td>1.396</td><td>1.165</td><td>1.233</td><td>1.265</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { s c a l e } }$ </td><td>1.449</td><td>1.171</td><td>1.266</td><td>1.295</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { g r a d } }$ </td><td>1.111</td><td>1.082</td><td>0.997</td><td>1.060</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { t e m p } }$ </td><td>1.099</td><td>1.131</td><td>0.988</td><td>1.072</td></tr><tr><td>Ours</td><td>1.098</td><td>1.077</td><td>0.982</td><td>1.052</td></tr></table>

TABLE V: Objective-term ablation on the Southeast China test split. All variants use the same nine ERA5 input variables and architecture. Scale error is the mean normalized crossresolution RMSE after resampling non-base queries to 128 × 128. Best results are highlighted in green, and second-best results are underlined.
<table><tr><td>Variant</td><td>RMSE↓</td><td>MAE ↓</td><td>Bias↓</td><td>Grad. err. ↓</td><td>Scale err. ↓</td></tr><tr><td>w/o aux. losses</td><td>1.213</td><td>0.858</td><td>0.339</td><td>0.468</td><td>1.265</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { s c a l e } }$ </td><td>1.220</td><td>0.860</td><td>0.350</td><td>0.382</td><td>1.295</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { g r a d } }$ </td><td>1.230</td><td>0.872</td><td>0.356</td><td>0.455</td><td>1.060</td></tr><tr><td>w/o Ltemp</td><td>1.160</td><td>0.825</td><td>0.281</td><td>0.376</td><td>1.038</td></tr><tr><td>Ours</td><td>1.123</td><td>0.811</td><td>0.273</td><td>0.361</td><td>1.052</td></tr></table>

## IV. CONCLUSION

In this paper, we address the fixed-output limitation of regional T2M forecasting by introducing CSTF, a spatiotemporal field forecaster. Based on the view that temperature products at varying lead times and grids are evaluations of a shared evolving atmospheric state, CSTF encodes multivariable ERA5 histories and decodes T2M fields with explicit lead-time, spatial-coordinate, and resolution queries. The proposed deterministic, spatial-gradient, temporal-difference, and scale-consistency objectives promote forecast accuracy, regional thermal-structure fidelity, and cross-resolution coherence. Experiments on the SE China benchmark show that CSTF improves fixed-lead deterministic skill while enabling non-integer lead-time and resolution-controllable queries.

## REFERENCES

[1] S. K. Mukkavilli, D. S. Civitarese, J. Schmude, J. Jakubik, A. Jones, N. Nguyen, C. Phillips, S. Roy, S. Singh, C. Watson et al., “Ai foundation models for weather and climate: Applications, design, and implementation,” arXiv preprint arXiv:2309.10808, 2023.

[2] J. Zhou, F. Chen, Y. Zhu, F. Xie, C. Qin, H. Tian, Y. Shen, X. Yang, Y. Duan, M. M. Afzal et al., “Reconstruction of temperature, precipitation, and identification of extreme climate events in high mountain asia over 500 years using multi-method enkf,” Scientific Reports, 2026.

[3] K. Arjdal, F. Driouech, S. Balhane, and R. Manzanas, “Compound heatwaves in north africa: Mechanisms and model capabilities,” Earth Systems and Environment, pp. 1–25, 2026.

[4] F. Chajaei and H. Bagheri, “Machine learning framework for high-resolution air temperature downscaling using lidar-derived urban morphological features,” Urban Climate, vol. 57, p. 102102, 2024.

[5] Z. Liu, H. Chen, L. Bai, W. Li, W. Ouyang, Z. Zou, and Z. Shi, “Mambads: Near-surface meteorological field downscaling with topography constrained selective state space modeling,” arXiv preprint arXiv:2408.10854, 2024.

[6] K. Bi, L. Xie, H. Zhang, X. Chen, X. Gu, and Q. Tian, “Accurate medium-range global weather forecasting with 3d neural networks,” Nature, vol. 619, no. 7970, pp. 533– 538, 2023.

[7] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu et al., “Learning skillful mediumrange global weather forecasting,” Science, vol. 382, no. 6677, pp. 1416–1421, 2023.

[8] K. Chen, T. Han, J. Gong, L. Bai, F. Ling, J.-J. Luo, X. Chen, L. Ma, T. Zhang, R. Su et al., “Fengwu: Pushing the skillful global medium-range weather forecast beyond 10 days lead,” arXiv preprint arXiv:2304.02948, 2023.

[9] J. Tian, Y. Ding, R. Xu, H. Miao, C. Guo, and B. Yang, “Arrow: An adaptive rollout and routing method for global weather forecasting,” arXiv preprint arXiv:2510.09734, 2025.

[10] Y. Sun, K. Deng, K. Ren, J. Liu, C. Deng, and Y. Jin, “Deep learning in statistical downscaling for deriving high spatial resolution gridded meteorological data: A systematic review,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 208, pp. 14–38, 2024.

[11] P. Hess, M. Aich, B. Pan, and N. Boers, “Fast, scaleadaptive, and uncertainty-aware downscaling of earth system model fields with generative foundation models,” arXiv preprint arXiv:2403.02774, 2024.

[12] Z. Liu, H. Chen, L. Bai, W. Li, K. Chen, Z. Wang, W. Ouyang, Z. Zou, and Z. Shi, “Deriving accurate surface meteorological states at arbitrary locations via observation-guided continous neural field modeling,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[13] S. Gao, X. Liu, B. Zeng, S. Xu, Y. Li, X. Luo, J. Liu, X. Zhen, and B. Zhang, “Implicit diffusion models for continuous super-resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10 021–10 030.

[14] Y. Verma, M. Heinonen, and V. Garg, “Climode: Climate and weather forecasting with physics-informed neural odes,” arXiv preprint arXiv:2404.10024, 2024.

[15] Y. Seong, D. Kim, M. Seo, and C. Kim, “Rainode: Continuous-time precipitation forecasting with latent neural odes,” arXiv preprint arXiv:2606.29855, 2026.

[16] S. Rasp, P. D. Dueben, S. Scher, J. A. Weyn, S. Mouatadid, and N. Thuerey, “Weatherbench: a benchmark data set for data-driven weather forecasting,” Journal of Advances in Modeling Earth Systems, vol. 12, no. 11, p. e2020MS002203, 2020.

[17] J. Pathak, S. Subramanian, P. Harrington, S. Raja, A. Chattopadhyay, M. Mardani, T. Kurth, D. Hall, Z. Li, K. Azizzadenesheli et al., “Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators,” arXiv preprint arXiv:2202.11214, 2022.

[18] Z. Gao, X. Shi, H. Wang, Y. Zhu, Y. B. Wang, M. Li, and D.-Y. Yeung, “Earthformer: Exploring space-time transformers for earth system forecasting,” Advances in Neural Information Processing Systems, vol. 35, pp. 25 390–25 403, 2022.

[19] C. Shi, H. Xu, Y. Li, Y.-L. Wei, Y. Feng, Y. Zhang, and D. Niu, “Wavec2r: Wavelet-driven coarse-to-refined

hierarchical learning for radar retrieval,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 40, no. 11, 2026, pp. 8951–8959.

[20] H. Hersbach, B. Bell, P. Berrisford, S. Hirahara, A. Horanyi, J. Mu ´ noz-Sabater, J. Nicolas, C. Peubey,˜ R. Radu, D. Schepers et al., “The era5 global reanalysis,” Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730, pp. 1999–2049, 2020.

[21] W. Wang, S. Feng, Y. Zhang, Z. Wei, J. Dong, L. Weihermuller, C.-Q. Liu, and H. Vereecken, “Fusing era5-¨ land and smap l4 for an improved global soil moisture product (1950–2025).” Earth System Science Data, vol. 18, no. 2, p. 1061, 2026.

[22] P. Liu, T. Zhou, L. Sun, and R. Jin, “Mitigating time discretization challenges with weatherode: A sandwich physics-driven neural ode for weather forecasting,” arXiv preprint arXiv:2410.06560, 2024.

[23] M. Tatsubori, T. Moriyama, T. Ishikawa, P. Fraccaro, A. Jones, B. Edwards, J. Kuehnert, and S. L. Remy, “Deep temporal interpolation of radar-based precipitation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 1685–1689.

[24] B. Z. Demiray, M. Sit, and I. Demir, “Efficienttempnet: Temporal super-resolution of radar rainfall,” International Conference on Learning Representations (ICLR) 2023 Workshop on Tackling Climate Change with Machine Learning, 2023.

[25] L. Wang, Q. Li, Q. Lv, X. Peng, and W. You, “Temdeep: a self-supervised framework for temporal downscaling of atmospheric fields at arbitrary time resolutions,” Geoscientific Model Development, vol. 18, no. 8, pp. 2427– 2442, 2025.

[26] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, “Fourier neural operator for parametric partial differential equations,” arXiv preprint arXiv:2010.08895, 2020.

[27] J. Guibas, M. Mardani, Z. Li, A. Tao, A. Anandkumar, and B. Catanzaro, “Adaptive fourier neural operators: Efficient token mixers for transformers,” arXiv preprint arXiv:2111.13587, 2021.

[28] K. Trebing, “Smaat-unet: Precipitation nowcasting using a small attention-unet architecture,” Pattern Recognition Letters, vol. 145, pp. 178–186, 2021.

[29] J. Schusterbauer, J. Wiese, N. Stracke, T. Phan, and B. Ommer, “Probabilistic precipitation nowcasting with rectified flow transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 25 742–25 756.

[30] S. Tu, B. Fei, W. Yang, F. Ling, H. Chen, Z. Liu, K. Chen, H. Fan, W. Ouyang, and L. Bai, “Satellite observations guided diffusion model for accurate meteorological states at arbitrary resolution,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 28 071–28 080.

## APPENDIX

This supplementary material provides additional implementation details in Sec. A, extended temperature forecast case studies in Sec. B, and query, training, benchmark, and ablation diagnostics in Sec. C.

## A. Additional Implementation Details

Data split and preprocessing. The Southeast China (SE China) domain spans $1 0 0 . 0 ^ { \circ } \mathrm { E - 1 2 0 . 0 ^ { \circ } E }$ and $2 0 . 0 ^ { \circ } \mathrm { N - 4 0 . 0 ^ { \circ } N } .$ The Southeast China and global-scope splits both contain 4,615 training samples, 511 validation samples, and 714 test samples. The data are chronologically split over 2024–2025, with validation drawn from the training period using a 0.1 fraction and a held-out test subset used only for final evaluation. Unless otherwise stated, fields are resized to $1 2 8 \times 1 2 8$ for training and supervised integer-lead evaluation.

Training protocol. All reported CSTF models are trained for T2M prediction with a three-frame input history and a sixframe hourly forecast target. The main CSTF model is trained for 50 epochs with batch size 8, AdamW optimization, an initial learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 4 }$ , cosine learning-rate annealing, gradient clipping at 1.0, and random seed 42. The final model is selected by validation loss and then evaluated on the held-out test split. For the full objective, the auxiliary loss weights are $\lambda _ { s } ~ = ~ 0 . 0 5 , ~ \lambda _ { g } ~ = ~ 0 . 0 5$ , and $\lambda _ { t } = 0 . 0 1$ . Test metrics are computed with batch size 32 using the same normalization and evaluation script across CSTF and all fixed-grid baselines.

Query diagnostics. The standard benchmark verifies supervised integer-lead T2M forecasts against ERA5-Land. Continuous lead-time diagnostics use a single learned parameterization at both integer and non-integer lead times; non-integer leads are shown as forecast-only examples because the current hourly target split has no direct half-hour references. Resolution diagnostics evaluate the same field parameterization at 96 × 96, 128 × 128, 160 × 160, and $1 9 2 \times 1 9 2$ output grids and assess cross-resolution consistency after resampling to the base grid. The scale-consistency values in the main tables are reported as normalized RMSE diagnostics rather than physical-unit forecast errors.

Ablation setting. All controlled CSTF variants keep the architecture, data split, input variables, input length, output horizon, training resolution, and model-selection rule fixed. The ablations therefore isolate objective terms rather than changes in model capacity or input information.

Training objective rationale. The forecast MSE provides the main deterministic supervision on the hourly ERA5-Land targets and is computed over all supervised leads, channels, and grid cells in each training batch. The scale-consistency term is designed for the queryable-resolution interface: for the same atmospheric input and lead time, a direct low-resolution query should agree with an area-downsampled high-resolution query after both are represented on the same grid. In the implementation, the downsampled high-resolution branch is detached in this term, so the auxiliary gradient primarily aligns the direct low-resolution query with the high-resolution prediction used as the reference. The spatial-gradient term compares finite differences along the two image axes and discourages overly smooth temperature maps by penalizing mismatches in local thermal contrasts. The temporal-difference term compares adjacent-lead changes rather than only absolute values, encouraging the predicted sequence to follow the observed short-range warming or cooling tendency across the six supervised lead times. Together, these auxiliary objectives are intended to support the two query diagnostics used in the experiments: coherent multi-resolution products and smooth intermediate lead-time behavior.

## B. Additional Temperature Forecast Case Studies

This section provides additional deterministic T2M forecast visualizations before the query-specific diagnostics in Sec. C. The purpose is to separate two forms of evidence: first, whether the learned forecaster produces meteorologically coherent temperature fields at the supervised hourly leads, and second, whether the same model can later be interrogated through lead-time and resolution queries. Figures 9 and 10 therefore focus on standard deterministic forecast behavior rather than on flexible queryability.

Figure 9 extends the global-scope qualitative diagnostic beyond the main-text initialization. Because the global split is used only for diagnostic visualization, the figure should be interpreted as a spatial plausibility check: it examines whether large-scale thermal patterns, land–sea contrasts, and error distributions remain coherent when the regional fieldevaluation formulation is applied under a global rendering protocol. The inclusion of input context, ERA5-Land references, CSTF predictions, and absolute-error maps allows the reader to distinguish forecast structure from residual error patterns.

Figure 10 complements the global case by focusing on the supervised Southeast China benchmark used for quantitative evaluation. Instead of summarizing skill with a single aggregate score, the figure reports absolute-error maps at each integer lead time, making it possible to inspect where regional discrepancies emerge and how their spatial distribution changes across the 0–6 h forecast horizon. The number overlaid on each panel is the corresponding spatial RMSE in degrees Celsius, computed between the model prediction and the ERA5-Land T2M reference over valid land pixels. This view is particularly relevant for T2M forecasting because localized thermal contrasts can be smoothed or displaced even when average error remains low.

## C. Additional Query and Quantitative Diagnostics

This section expands the evidence for the query-conditioned field interface beyond the main-text examples. Figures 11– 15 are organized to separate the two query modes and the two diagnostic domains. Specifically, Figs. 11 and 12 examine continuous lead-time and resolution-conditioned behavior over Southeast China $\mathrm { ( 1 0 0 . 0 ^ { \circ } E - 1 2 0 . 0 ^ { \circ } E , ~ 2 0 . 0 ^ { \circ } N - }$ $4 0 . 0 ^ { \circ } \mathrm { N } )$ , whereas Figs. 13 and 14 provide the corresponding global-domain diagnostics $( 1 8 0 . 0 ^ { \circ } \mathrm { W } \mathrm { - } 1 8 0 . 0 ^ { \circ } \mathrm { E } , ~ 9 0 . 0 ^ { \circ } \mathrm { S } -$ $9 0 . 0 ^ { \circ } \mathrm { N } )$ . Figure 15 then summarizes training, benchmark, scale-consistency, and ablation evidence to connect these qualitative diagnostics with the quantitative claims in the main text.

Global CSTF forecast case, idx=101, anchor=2025090505 UTC

![](images/4aee89b7d840184aa58dc107238fc034a42f25e676903a895ae30664ad29ccd9.jpg)  
Fig. 9: Additional deterministic forecast diagnostic for the global domain (180.0<sup>◦</sup>W–180.0<sup>◦</sup>E, 90.0<sup>◦</sup>S–90.0<sup>◦</sup>N). The case shows input context, ERA5-Land references, CSTF predictions, and absolute-error maps under the main-text global visualization protocol. It supplements the idx0 main-text case and assesses whether the qualitative global diagnostic remains consistent across forecast initializations.

Figure 11 extends the regional continuous-time analysis from a single main-text example to three independent Southeast China forecast initializations. The purpose of this figure is to test whether varying the lead-time query produces physically plausible, spatially coherent temperature evolution rather than isolated artifacts at non-integer query times.

Figure 12 evaluates the regional resolution-query interface under the same Southeast China setting. By holding the atmospheric input and forecast case fixed while changing only the requested output grid, the figure checks whether CSTF behaves as a direct field evaluator instead of a fixed-grid predictor followed by post-hoc resizing.

Figure 13 transfers the continuous lead-time diagnostic to the global domain. This figure is not intended as a separate global benchmark; rather, it examines whether the same query-conditioned model produces coherent large-scale thermal structures when lead time is varied under a global diagnostic view.

Figure 14 provides the complementary global-domain resolution-query diagnostic. It is used to inspect whether large-scale temperature patterns remain stable across requested output grids, which supports the interpretation of resolution control as query-conditioned field evaluation rather than a visualization-only operation.

Figure 15 collects the quantitative diagnostics that support the preceding visual evidence. The panels cover optimization behavior, deterministic forecast skill, lead-wise degradation, cross-resolution consistency, and objective-term ablations, providing a compact check that the queryable behavior is accompanied by stable training and controlled test performance.

## D. Extended Objective-Term Ablation

The main text reports the primary objective-term ablation while emphasizing the complete CSTF objective and the removal of the scale-consistency term. To make the contribution of the remaining auxiliary regularizers explicit, Table III adds two controlled variants that remove the spatial-gradient loss or the temporal-difference loss individually. All variants use the same architecture, nine-variable ERA5 input history, Southeast China test split, checkpoint-selection rule, and deterministic evaluation protocol.

The extended results show that removing either regularizer degrades lead-wise deterministic skill relative to the full objective. Removing $\mathcal { L } _ { \mathrm { g r a d } }$ gives the largest deterioration, especially at longer lead times, indicating that preserving regional thermal gradients is important for maintaining spatial temperature structure. Removing $\mathcal { L } _ { \mathrm { t e m p } }$ is less damaging than removing $\mathcal { L } _ { \mathrm { g r a d } }$ , but it still increases RMSE and MAE across all supervised leads, supporting the role of temporal-difference supervision in stabilizing short-range evolution. Together with the scale-consistency ablation in the main text, these results indicate that the auxiliary terms provide complementary regularization for spatial structure, lead-wise evolution, and crossresolution coherence.

![](images/4fda6a14122332abd3bcf0d7be06f78f58cdb0c7a91f596342c8fbed35d89359.jpg)  
Fig. 10: Lead-wise absolute-error maps for the Southeast China case study $( 1 0 0 . 0 ^ { \circ } \mathrm { E - 1 2 0 . 0 ^ { \circ } E , ~ 2 0 . 0 ^ { \circ } N - 4 0 . 0 ^ { \circ } N } )$ . Errors are shown over the six supervised integer lead times, providing a spatial diagnostic of where forecast discrepancies grow with lead time. The overlaid number on each panel reports the per-frame spatial RMSE (<sup>◦</sup>C) against the ERA5-Land T2M reference over valid land pixels.

idx=101 anchor=2025090505  
![](images/91709ca13e65c0fa4495d86b305057fcae63ce9780b6f0e5eea0e1f8dcfe6aab.jpg)  
Fig. 11: Continuous lead-time query examples over Southeast China (100.0<sup>◦</sup>E–120.0<sup>◦</sup>E, 20.0<sup>◦</sup>N–40.0<sup>◦</sup>N). The three rows are independent test initializations at 2025-09-05 05:00 UTC, 2025-09-13 17:00 UTC, and 2025-09-22 05:00 UTC. Within each row, the same atmospheric input is queried at multiple lead times, including supervised hourly leads and non-integer lead times. The figure is therefore used to inspect whether the learned field evaluator produces temporally smooth and spatially coherent T2M fields when the requested forecast time is varied continuously. Non-integer lead panels are forecast-only diagnostics because the current ERA5-Land target sequence is available at hourly intervals.

resolution query at 3 h idx=101 anchor=2025090505  
![](images/45b07ba129ee6de30864ef99ed4a772bede3fae6cee45955042430a4d2612a22.jpg)  
Fig. 12: Resolution-conditioned query examples over Southeast China (100.0<sup>◦</sup>E–120.0<sup>◦</sup>E, 20.0<sup>◦</sup>N–40.0<sup>◦</sup>N). The three rows are independent test initializations at 2025-09-05 05:00 UTC, 2025-09-13 17:00 UTC, and 2025-09-22 05:00 UTC. Within each row, CSTF directly evaluates the same forecast state at different requested output grids rather than resizing a single fixed grid prediction. The visual comparison checks whether regional thermal structures remain consistent as the output resolution changes, complementing the cross-resolution RMSE values reported in the main text.

![](images/3e4ca37b3db9369f09054e799766fcf7ae33c9a0659a040b64dc57b8e69481f3.jpg)  
Fig. 13: Continuous lead-time query diagnostics over the global domain (180.0<sup>◦</sup>W–180.0<sup>◦</sup>E, 90.0<sup>◦</sup>S–90.0<sup>◦</sup>N). The three row are independent test initializations at 2025-09-05 05:00 UTC, 2025-09-13 17:00 UTC, and 2025-09-22 05:00 UTC, rendered with the same globe-style projection as the main-text global diagnostic. Within each row, the queried lead time is varied while the input state is fixed, allowing visual inspection of large-scale thermal evolution under the continuous-time interface. Non-integer leads are forecast-only diagnostics because the current target split provides hourly references.

![](images/c7f3060af0d550c5102ae461349e7902e1b27343e0b494f1d9c767bcba111930.jpg)  
Fig. 14: Resolution-conditioned query diagnostics over the global domain (180.0<sup>◦</sup>W–180.0<sup>◦</sup>E, 90.0<sup>◦</sup>S–90.0<sup>◦</sup>N). The three rows are independent test initializations at 2025-09-05 05:00 UTC, 2025-09-13 17:00 UTC, and 2025-09-22 05:00 UTC. Within each row, the model is queried at multiple output grid sizes for the same forecast case, so changes across columns reflect direct resolution-conditioned field evaluation. These visualizations are used as qualitative checks of large-scale coherence under resolution queries, whereas quantitative cross-resolution consistency is reported separately in the main tables.

![](images/2344c8def703579527410c13ebb0ffd1d2d1c72fa0e90f326db7a16ecfa5c006.jpg)

![](images/3f52f59d41dfec30fd474e4e52261347d5ed1863bb6f9bb3801b4278584438a8.jpg)  
Loss values are parsed from the original training logs and shown on a logarithmic scale

![](images/887dc04ac57d335c20c3eb3ac10e7281a941c123f6ca2cd16d9c7233a5733a79.jpg)

Deterministic forecast skill on the Southeast China test split  
![](images/887c12b021f12d5385c23f0efbaffee25b88c8fa4e598bcc3cd5af5866c5eaf2.jpg)

![](images/386902129d9595f382cba4f3904e6f4e9e0bf2b1ce1c76d4b3395e5e7a281ba3.jpg)  
Cross-resolution consistency

![](images/83b71021ee01b08f6218d61001633587aa109961153047b430fc9da8291a159b.jpg)

![](images/3ea754b44741092e5dbe63cb25731e0df54591dac079b418726572b5e6348650.jpg)

Effect of CSTF objective terms  
![](images/30ceb850e6fe0b98675992c2c1e1a1a3b544f3b1b26aa5a0f687830945dda9ec.jpg)

![](images/5e64f56b4b440b84435f73178205886dcb50591312621903c784b821e3eda58c.jpg)

![](images/c4cb20e3647602e4843502dc537d36e40659122ea9bf9b6f966fe6552788ec1b.jpg)

![](images/c763753c88a11d4c93dcc5336588fbd4581cf547543caaa2d35d8a3ce0a03535.jpg)  
Fig. 15: Additional training, benchmark, and ablation diagnostics supporting the main quantitative results. From left to right and top to bottom, the panels show optimization curves, aggregate deterministic skill, lead-wise RMSE, cross-resolution scale consistency, objective-term ablation metrics, and lead-wise ablation behavior. The first panel checks whether model training is stable, the next two panels summarize forecast accuracy on the supervised hourly benchmark, the scale-consistency panel evaluates the resolution-query interface, and the ablation panels isolate the effects of the auxiliary objective terms. These plots are diagnostic summaries; the standardized test metrics and primary comparisons are reported in the main-text tables.