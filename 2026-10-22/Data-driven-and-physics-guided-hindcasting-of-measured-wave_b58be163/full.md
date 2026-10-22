# Data-driven and physics-guided hindcasting of measured wave spectra

Sander Rikka<sup>1,∗</sup>

<sup>1</sup>Department of Marine Systems, Tallinn University of Technology <sup>∗</sup>Corresponding author: sander.rikka@taltech.ee

## Abstract

Short-range hindcasts of measured wave spectra are evaluated using a recurrent model initialized by one hourly buoy spectrum and supplied with local wind, fetch and depth. Measurements from three Baltic Sea stations are combined for training in 2017–2020, validation in 2021 and testing in 2022. ERA5 reanalysis supplies winds throughout the twelve-hour prediction interval. The comparison examines supervision of spectral variance, peak frequency and peak amplitude, followed by empirical wind-sea bounds and a local relaxation constraint. Variance and peak-frequency supervision gives the lowest spectral error within the data-driven comparison, whereas additional peak-amplitude supervision gives the lowest wave-height error. In the matched physical comparison, relaxation reduces height RMSE by 5.01% to 19.26 cm. Extending this constraint through hours 13–24, without spectral targets at those hours, reduces primary spectral RMSE by 2.31% to 0.39 m<sup>2</sup> Hz<sup>−1</sup>. Upper-bound constraints increase errors. Wind-condition comparisons show spectral gains from extended relaxation in increasing, decreasing and steady strong winds, with little change in steady weak winds. The results support modest, objective-dependent benefits from local physical guidance. Waveheight scores concern the retained frequency band, and operational forecasting and transfer to independent stations remain to be evaluated.

## 1 Introduction

Reliable information on sea state is important for maritime safety, coastal management, ofshore engineering and the planning of marine operations. Wave measurements describe the conditions encountered at a site, numerical models extend this information in space and time, and forecasts support decisions before adverse conditions develop. Both typical conditions and the intensity and duration of energetic events matter for these applications (Rutgersson et al., 2022; Björkqvist et al., 2024). Improving the representation and short-term prediction of waves therefore remains relevant even in seas with a comparatively moderate mean wave climate.

The Baltic Sea is a semi-enclosed basin with an irregular coastline, distinct sub-basins and seasonal ice cover. Its wave climate is dominated by locally generated windsea, with limited penetration of ocean swell. The 41-year hindcast of Björkqvist et al. (2018) gives open-sea ice-free mean significant wave heights broadly within 0.5–1.5 m, with the highest values in the exposed Baltic Proper and lower values in sheltered sub-basins. Published buoy records include storm significant wave heights of 8.2 m in the northern Baltic Proper, 8.1 m in the Bothnian Sea and 5.2 m in the Gulf of Finland (Björkqvist et al., 2018, 2020). These are measured extremes reported in those studies; the 41-year hindcast gives a local maximum of 10.1 m in the southern Baltic Proper. Mean peak periods are approximately 5.0–6.0 s in the Baltic Proper and below 5.0 s in the Gulf of Finland, with similar statistics in the Bothnian Sea; peak periods during severe Baltic Proper storms can reach approximately 10.0–14.0 s (Björkqvist et al., 2018).

Wave growth depends on both the available fetch length and the duration of the wind forcing. The longest possible Baltic fetch is about 700 km, but coastlines, islands and ice strongly restrict the efective fetch for many wind directions (Björkqvist et al., 2018). Changes in wind direction can therefore alter wave conditions rapidly. Soomere and Räämet (2011) describe relatively short saturation and memory times for Baltic wave fields, while Björkqvist et al. (2021) show that locally generated waves may persist after the wind weakens. Their swell partition depends on wave speed and wind–wave direction, and does not necessarily represent long-period oceanic swell. These conditions motivate examining the measured initial spectrum together with evolving local winds over a short prediction horizon.

Spectral wave models describe variance distributed over frequency and direction and its evolution through propagation, wind input, dissipation and wave interactions. SWAN, for example, solves a spectral action balance incorporating these processes (Booij et al., 1999). Such models have been used extensively to characterize Baltic wave conditions (Björkqvist et al., 2018). For operational applications, the Finnish Meteorological Institute produces the Copernicus Marine Baltic Sea Wave Analysis and Forecast product (European Union-Copernicus Marine Service, 2020). The cited product manual specifies WAM cycle 4.7 on an approximately one-nautical-mile grid, with hourly output and twice-daily production. MetCoOp-HARMONIE supplies winds for the first 66 h, followed by ECMWF forcing; currents, sea level and ice information are supplied by the regional ocean forecast system (Tuomi and Kanarik, 2025). Aguiar et al. (2024) evaluated an earlier WAM configuration and reported Baltic operational wave-height errors, including the one-to-twelve-hour range. Significant wave height is an important integrated measure, but spectra with the same variance can difer in peak location, peak density and shape. Wave-height accuracy alone is therefore insuficient to assess spectrum prediction.

Machine learning provides a complementary approach to short-term wave prediction. LSTMbased methods have been applied to significant wave height using historical wave and meteorological measurements (Fan et al., 2020). Observation-driven short-range prediction has also been combined with spatial surrogates and evaluated against operational numerical forecasts (Chen et al., 2023). Nevertheless, greater model complexity does not ensure higher accuracy: Jiang et al. (2024) found that neural models did not outperform autoregression in their univariate wave-height comparison. Appropriate baselines and a clearly defined input protocol remain necessary.

Deep learning has also been used to predict the spectrum itself. Song and Jiang (2023) modeled single-point directional spectra from ERA5 wind fields, and Gao and Jiang (2025) combined local and remote wind inputs with numerical hindcast spectra as targets. Lu et al. (2026) developed a global directional-spectrum prediction framework using latent compression and reconstruction. Observed spectra have also been used as learning targets: Filoche et al. (2026) corrected operational directional forecasts to one-dimensional buoy spectra. These approaches difer in whether they emulate numerical spectra, post-process operational forecasts or predict the subsequent evolution of an observed spectrum. Similarly, reanalysis-conditioned experiments, including global wave-height modeling by Wang and Jiang (2024), have diferent information constraints from forecasts driven only by data available at issue time.

Physical knowledge can guide learning through observed wave descriptors or through constraints on wave evolution. Physics-informed neural networks include governing-equation residuals in the loss (Raissi et al., 2019); ocean-wave applications include phase-resolved reconstruction (Ehlers et al., 2025) and reduced energy balances for regional wave parameters (Wang et al., 2026). Other methods supervise spectral energy, peak frequency and shape when reconstructing spectra from ship motions (Sun et al., 2025). Such descriptor supervision constrains agreement with observations, whereas a dynamical residual constrains agreement with an assumed physical relation.

This study examines how these two forms of guidance afect short-range hindcasts of measured one-dimensional wave spectra. One buoy spectrum initializes a twelve-hour prediction, with local wind, directional fetch and depth as inputs. The comparison first adds supervision of variance, peak frequency and peak amplitude, and then tests empirical wind-sea bounds and a local relaxation equation. A further test extends physical guidance beyond the hours with spectral targets. ERA5 reanalysis supplies the winds over each prediction interval; the study therefore concerns retrospective hindcasting. Measurements from three Baltic Sea stations provide contrasting wave conditions within the same training and evaluation framework.

Section 2 describes the data, model and constraints. Section 3 presents the spectral and waveheight errors, their station and wind dependence, and the continuation test. Section 4 discusses the physical interpretation and application limits, followed by the conclusions in Section 5.

## 2 Data and methods

## 2.1 Measured spectra and hindcasting task

The study combines records from the Northern Baltic Proper (NBP), Bothnian Sea (BS) and Gulf of Finland (GoF) stations (Table 1). The instruments are Datawell Directional Waverider Mark 4 buoys operated by the Finnish Meteorological Institute (FMI). A single model is trained jointly on the three stations. At each hourly origin t, one measured variance-density spectrum $E _ { t } ( f )$ initializes predictions for $t + 1 , \ldots , t + 1 2 \mathrm { ~ h ~ }$ .

The spectra were resampled to hourly resolution during data preparation and aligned with hourly winds. Incomplete records and windows crossing time gaps are excluded. Spectra are linearly interpolated onto 52 frequencies between 0.085 and 0.595 Hz, at nominal spacing 0.01 Hz. This interpolation does not explicitly preserve integrated variance. The same grid is used for observations, predictions and persistence.

The retained variance and corresponding significant wave height are

$$
m _ { 0 } ^ { B } = \sum _ { k = 1 } ^ { F } E ( f _ { k } ) \Delta f _ { k } , \qquad H _ { s } ^ { B } = 4 \sqrt { m _ { 0 } ^ { B } } .\tag{1}
$$

Here B denotes the retained frequency grid. Endpoint widths equal the adjacent frequency diference; interior widths equal half the two-neighbour diference. Throughout the following text, $m _ { 0 }$ and $H _ { s }$ denote these band-limited quantities, $m _ { 0 } ^ { B }$ and $H _ { s } ^ { B }$ . The resulting variance and wave height can therefore difer from those calculated using the full frequency range of the original spectra.

Table 1: Buoy locations, water depths and aligned prediction-origin counts.
<table><tr><td>Station</td><td>Latitude (°N)</td><td>Longitude (°E)</td><td>Depth (m)</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>BS</td><td>61.8000</td><td>20.2333</td><td>112.2</td><td>23,844</td><td>8,515</td><td>8,633</td></tr><tr><td>GoF</td><td>59.9650</td><td>25.2350</td><td>49.7</td><td>22,264</td><td>6,794</td><td>6,851</td></tr><tr><td>NBP</td><td>59.2500</td><td>21.0000</td><td>107.0</td><td>27,685</td><td>8,553</td><td>7,695</td></tr></table>

Training targets fall in 2017–2020, validation targets in 2021 and test targets in 2022. Every twelve-hour target window must lie within its assigned period. The resulting counts are 73,793, 23,862 and 23,179 origins, respectively. All three stations contribute to each period. Successive hourly origins overlap and do not represent independent events.

## 2.2 Wind, fetch and depth

The ten-metre wind components $u _ { 1 0 } , v _ { 1 0 }$ are from ERA5 hourly single-level reanalysis (Hersbach et al., 2020; Copernicus Climate Change Service (C3S), 2018), extracted from the closest point on its 0.25<sup>◦</sup> grid without spatial interpolation. Winds at the origin and throughout the prediction interval are supplied to the model. These are retrospective reanalysis winds; operational windforecast errors are not represented.

Upwind fetch is measured along eight rays from each station to the first land intersection, using the full-resolution GSHHG 2.3.7 coastline (Wessel and Smith, 2017). The nearest $4 5 ^ { \circ }$ sector is selected from meteorological wind direction, $\theta = \mathrm { a t a n 2 } ( - u _ { 1 0 } , - v _ { 1 0 } )$ modulo $3 6 0 ^ { \circ }$ . Station depth is taken from the nearest coordinate in the 500-m Baltic Sea Bathymetry Database (BSBD) 0.9.3 grid, downloaded on 3 December 2025 (Baltic Sea Hydrographic Commission, 2013). Its supplied vertical reference is retained.

## 2.3 Model and comparison design

Table 2 summarizes the seven objectives. The data-driven comparisons add supervision of measured spectral descriptors. The physics-guided comparisons add empirical wind-sea bounds or a local relaxation ordinary diferential equation (ODE). These physical terms represent reduced local assumptions; the spatial wave-action partial diferential equation (PDE) and complete source balance are not imposed.

Table 2: Comparison objectives. All variants share the same inputs and network.
<table><tr><td>ID</td><td>Descriptive name</td><td>Objective</td></tr><tr><td>A0</td><td>Spectrum baseline</td><td>Log-spectrum reconstruction and temporal increments</td></tr><tr><td>A1</td><td>Variance supervision</td><td>A0 plus observed spectral variance</td></tr><tr><td>A2</td><td>Peak-frequency supervision</td><td>A1 plus observed soft peak frequency</td></tr><tr><td>A3</td><td>Peak-amplitude supervision</td><td>A2 plus observed peak spectral density</td></tr><tr><td>P1</td><td>Wind-sea bounds</td><td>A2 plus upper bounds on wind-sea variance and peak period</td></tr><tr><td>P2</td><td>Relaxation</td><td>A2 plus a local relaxation residual at hours 1–12</td></tr><tr><td>P3</td><td>Extended relaxation</td><td>P2 plus the residual at hours 13–24, without spectral targets</td></tr></table>

All objectives use the same encoder–decoder long short-term memory (LSTM) architecture (Hochreiter and Schmidhuber, 1997). The encoder has three layers and the decoder one layer, each with 128 hidden units. The encoder receives the initial spectrum, wind and fetch; the decoder receives the preceding spectrum, wind and fetch for the next hour, and station depth. Wind components, fetch in km and signed bathymetry in m are used without further standardization. Spectra are represented as $Y = \log _ { 1 0 } \operatorname* { m a x } ( E , \epsilon )$ , with $\epsilon = 1 0 ^ { - 8 }$ . At each step, a predicted increment updates the spectrum:

$$
\widehat { Y } _ { h } = \mathrm { c l i p } ( \widehat { Y } _ { h - 1 } + \Delta Y _ { h } , - 1 2 , 4 ) , \qquad \widehat { Y } _ { 0 } = Y _ { t } .\tag{2}
$$

The model feeds its own prediction into the next step during both training and evaluation. Linear spectra are recovered as $1 0 ^ { \widehat { Y } _ { h } }$ ; descriptor losses use densities floored at ϵ.

Let ⟨·⟩ denote a mean over the applicable samples, hours and frequencies. For $N = 1 2$ prediction hours, batch size B and F frequency bins, the baseline loss is

$$
L _ { \mathrm { r e c } } = \langle ( { \widehat { Y } } - Y ) ^ { 2 } \rangle ,\tag{3}
$$

$$
L _ { \mathrm { i n c r } } = \frac { 1 } { ( N - 1 ) B F } \sum _ { b = 1 } ^ { B } \sum _ { h = 2 } ^ { N } \sum _ { k = 1 } ^ { F } a _ { h } \left[ ( \widehat { Y } _ { b , h , k } - \widehat { Y } _ { b , h - 1 , k } ) - ( Y _ { b , h , k } - Y _ { b , h - 1 , k } ) \right] ^ { 2 } ,\tag{4}
$$

$$
L _ { \mathrm { { A 0 } } } = L _ { \mathrm { { r e c } } } + 0 . 3 L _ { \mathrm { { i n c r } } } .\tag{5}
$$

In $L _ { \mathrm { i n c r } } , \ a _ { h }$ weights the squared error in the change from lead $h - 1$ to h: $a _ { 2 } = 1 . 2 5$ for 1–2 h, $a _ { 3 } = 1 . 1 5$ for 2–3 h and $a _ { 4 } = 1 . 0 5$ for 3–4 h; $a _ { h } = 1$ for all later pairs.

Variance supervision uses

$$
L _ { m _ { 0 } } = \left. [ \log _ { 1 0 } \operatorname* { m a x } ( \widehat { m } _ { 0 } , \epsilon ) - \log _ { 1 0 } \operatorname* { m a x } ( m _ { 0 } , \epsilon ) ] ^ { 2 } \right. .\tag{6}
$$

A diferentiable peak frequency is extracted from both predicted and observed log spectra:

$$
\widetilde { f } _ { p } ( Y ) = \sum _ { k } f _ { k } \frac { \exp ( Y _ { k } / T ) } { \sum _ { j } \exp ( Y _ { j } / T ) } , \quad T = 0 . 1 5 , \qquad L _ { \mathrm { s o f t } } = \left. \left[ \frac { \widetilde { f } _ { p } ( \widehat { Y } ) - \widetilde { f } _ { p } ( Y ) } { f _ { F } - f _ { 1 } } \right] ^ { 2 } \right. .\tag{7}
$$

For peak amplitude $A _ { p } = \operatorname* { m a x } _ { k } E ( f _ { k } )$ , the loss is $L _ { A _ { p } } = \sqrt { \langle ( \widehat { A } _ { p } - A _ { p } ) ^ { 2 } \rangle + 1 0 ^ { - 1 2 } }$ . The objectives are

$$
L _ { \mathrm { A 1 } } = L _ { \mathrm { A 0 } } + 0 . 1 L _ { m _ { 0 } } , \quad L _ { \mathrm { A 2 } } = L _ { \mathrm { A 1 } } + 0 . 2 L _ { \mathrm { s o f t } } , \quad L _ { \mathrm { A 3 } } = L _ { \mathrm { A 2 } } + 0 . 0 0 1 L _ { A _ { p } } .\tag{8}
$$

These are observation-misfit terms. The amplitude term uses linear density, whereas variance and peak-frequency supervision use logarithmic or normalized quantities.

## 2.4 Local physical constraints

## 2.4.1 Equilibrium wind sea

For an hourly transition, let $U _ { h } ^ { * }$ and $X _ { h } ^ { * }$ be the means of the endpoint wind speeds and upwind fetch distances. With positive water depth $d ,$ define $\chi = g X _ { h } ^ { * } / ( U _ { h } ^ { * } ) ^ { 2 }$ and $\delta = g d / ( U _ { h } ^ { * } ) ^ { 2 }$ . The Young–Verhagen growth relations (Young and Verhagen, 1996; Karimpour and Chen, 2016) give equilibrium variance and peak frequency:

$$
a = \operatorname { t a n h } ( 0 . 4 9 3 \delta ^ { 0 . 7 5 } ) , \qquad b = \operatorname { t a n h } ( 0 . 3 3 1 \delta ^ { 1 . 0 1 } ) ,\tag{9}
$$

$$
\varepsilon _ { \mathrm { e q } } = 0 . 0 0 3 6 4 [ a \operatorname { t a n h } ( 0 . 0 0 3 1 3 \chi ^ { 0 . 5 7 } / a ) ] ^ { 1 . 7 4 } ,\tag{10}
$$

$$
\nu _ { \mathrm { e q } } = 0 . 1 3 3 [ b \operatorname { t a n h } ( 0 . 0 0 0 5 2 1 5 \chi ^ { 0 . 7 3 } / b ) ] ^ { - 0 . 3 7 } ,\tag{11}
$$

$$
\begin{array} { r } { m _ { \mathrm { e q } } = \varepsilon _ { \mathrm { e q } } ( U _ { h } ^ { * } ) ^ { 4 } / g ^ { 2 } , \qquad f _ { \mathrm { e q } } = \nu _ { \mathrm { e q } } g / U _ { h } ^ { * } . } \end{array}\tag{12}
$$

The represented wind-sea bins are $\mathcal { W } _ { h } = \{ f _ { k } \in \mathcal { B } : f _ { k } \ge 0 . 9 ( 0 . 1 3 g / U _ { h } ^ { * } ) \}$ . To project equilibrium variance onto these bins, a JONSWAP shape is used (Hasselmann et al., 1973):

$$
J ( q ) = q ^ { - 5 } \exp ( - 1 . 2 5 q ^ { - 4 } ) 3 . 3 ^ { \exp [ - ( q - 1 ) ^ { 2 } / ( 2 \sigma ^ { 2 } ) ] } ,\tag{13}
$$

$$
E _ { \mathrm { e q } } ( f ) = \frac { m _ { \mathrm { e q } } } { f _ { \mathrm { e q } } } \frac { J ( f / f _ { \mathrm { e q } } ) } { \int _ { 0 . 0 5 } ^ { 1 0 0 } J ( q ) d q } , \qquad m _ { \mathrm { e q } , h } ^ { w } = \sum _ { f _ { k } \in \mathcal { W } _ { h } } E _ { \mathrm { e q } } ( f _ { k } ) \Delta f _ { k } .\tag{14}
$$

Here $q = f / f _ { \mathrm { e q } }$ , and $\sigma = 0 . 0 7$ below the peak and 0.09 above it. The broad normalization range preserves the distinction between total equilibrium variance and its represented fraction.

## 2.4.2 Applicability and calibration

Physical penalties require $U _ { h } ^ { * } \geq 5 \mathrm { ~ m ~ s ~ } ^ { - 1 }$ , at least four wind-sea bins, endpoint wind-speed change no greater than $2 . 5 ~ \mathrm { m s ^ { - 1 } }$ and wind turn no greater than $3 0 ^ { \circ }$ . At least half the initial observed variance must lie in the step’s $\mathcal { W } _ { h }$ ; this condition uses the origin observation at every step. Fetch, depth and forcing must also be valid.

Calibration uses only the first observed transition at each training origin; 51,759 of 73,793 transitions meet these conditions. Stationwise 95th percentiles of observed/equilibrium variance and soft peak-period ratios, multiplied by 1.001, set upper-bound tolerances. A station is excluded from the corresponding bound if its ratio exceeds 4 for variance or 1.5 for period. The largest accepted ratio, with a minimum of one, gives $c _ { E } = 2 . 5 2 $ and $c _ { T } = 1 . 2 4$ . These bounds apply at BS and NBP; GoF remains in supervised training but is excluded from P1’s physical penalties.

The relaxation time is selected from {1, 2, 3, 6, 12, 24, 48} h. The smallest value satisfying a training-residual 90th absolute percentile of at most 0.15 log units and RMSE at most 1.10 times the observed persistence-increment RMSE is $\tau = 1 2 \ ]$ h. This empirical choice controls how strongly an hourly transition approaches equilibrium.

## 2.4.3 Bounds and relaxation

Both endpoints are integrated over the same $\mathcal { W } _ { h }$ , with the observed spectrum used at $h = 0$ . P1 penalizes excess wind-sea variance and excess soft peak period:

$$
{ r } _ { E , h } = [ \log _ { 1 0 } \widehat { m } _ { h } ^ { \mathcal { W } _ { h } } - \log _ { 1 0 } ( c _ { E } m _ { \mathrm { e q } , h } ^ { \mathcal { W } } ) ] _ { + } ,\tag{15}
$$

$$
r _ { T , h } = [ \log _ { 1 0 } ( f _ { \mathrm { e q } } / c _ { T } ) - \log _ { 1 0 } \tilde { f } _ { h } ^ { \mathcal { W } _ { h } } ] _ { + } ,\tag{16}
$$

$$
L _ { \mathrm { P 1 } } = L _ { \mathrm { A 2 } } + 0 . 1 \langle r _ { E } ^ { 2 } \rangle _ { \mathcal { G } _ { E } } + 0 . 1 \langle r _ { T } ^ { 2 } \rangle _ { \mathcal { G } _ { T } } .\tag{17}
$$

Here $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ , the soft peak is restricted to $\mathcal { W } _ { h }$ , and $\mathcal { G } _ { E } , \mathcal { G } _ { T }$ denote the applicable steps and station masks.

P2 and P3 instead constrain the local relaxation equation dm $/ d t = ( m _ { \mathrm { e q } } - m ) / \tau$ . Its exact solution over one hour, holding the equilibrium and frequency subset fixed, gives

$$
m _ { h } ^ { * } = m _ { \mathrm { e q } , h } ^ { \mathcal { W } } + ( \widehat { m } _ { h - 1 } ^ { \mathcal { W } _ { h } } - m _ { \mathrm { e q } , h } ^ { \mathcal { W } } ) e ^ { - \Delta t / \tau } ,\tag{18}
$$

$$
r _ { h } = \log _ { 1 0 } \widehat { m } _ { h } ^ { \mathcal { W } _ { h } } - \log _ { 1 0 } m _ { h } ^ { * } ,\tag{19}
$$

$$
L _ { \mathrm { P 2 } } = L _ { \mathrm { A 2 } } + 0 . 1 \langle r _ { h } ^ { 2 } \rangle _ { \mathcal { G } , 1 : 1 2 } ,\tag{20}
$$

$$
L _ { \mathrm { P 3 } } = L _ { \mathrm { A 2 } } + 0 . 1 \left( \langle r _ { h } ^ { 2 } \rangle _ { \mathcal { G } , 1 : 1 2 } + \langle r _ { h } ^ { 2 } \rangle _ { \mathcal { G } , 1 3 : 2 4 } \right) .\tag{21}
$$

Eligible steps at all three stations contribute to these losses. Each residual term is averaged separately over eligible steps, with zero contribution for an empty set. Variances are floored at ϵ before logarithms, and both predicted states receive gradients.

For P3, the model continues to hour 24 during training, while observed-spectrum targets stop at hour 12. Missing winds and continuation steps outside the training period are masked. All supervised training origins are retained, and validation remains a twelve-hour task.

## 2.5 Training and evaluation

AdamW uses batch size 256, initial learning rate $1 . 7 5 8 \times 1 0 ^ { - 3 }$ and weight decay 0.05. The learning rate halves every 50 epochs. Training runs for at most 90 epochs, with early stopping after 12 epochs without improvement in validation reconstruction loss. Checkpoints minimize 2021 $L _ { \mathrm { r e c } }$ . The architecture and optimization are fixed; the physical comparisons use a separately fitted A2 reference under the same protocol. This reference is reported distinctly from A2 in the descriptor comparison.

Persistence repeats the origin spectrum. Spectral RMSE pools squared errors across origins, leads and frequencies before taking the square root. Descriptor RMSE similarly pools origin–lead pairs. Evaluation uses the hard peak $f _ { p } = f _ { \mathrm { a r g m a x } _ { k } } { \cal E } ( f _ { k } )$ , peak amplitude and $H _ { s }$ . Wave-height bias is prediction minus observation; r is Pearson correlation and $\mathrm { S I } = \mathrm { R M S E } / \overline { { H _ { s } } }$ . The primary aggregate covers leads 1–12, with 278,148 overlapping origin–lead pairs.

Normalized spectral error divides RMSE by the root mean square (RMS) observed spectral density in the same subset. Normalized height error divides RMSE by mean observed $H _ { s }$ Persistence skill is the percentage RMSE reduction,

$$
S _ { q } = 1 0 0 \left( 1 - \frac { \mathrm { R M S E } _ { q , \mathrm { m o d e l } } } { \mathrm { R M S E } _ { q , \mathrm { p e r s i s t e n c e } } } \right) \% .\tag{22}
$$

For continuation, A2, P2 and P3 are evaluated without further fitting on 22,892 origins having complete 24-hour observations. Hours 1–12 and 13–24 are scored separately on this common subset.

## 2.6 Wind-condition analysis

Each twelve-hour test interval is classified from ERA5 speed $U _ { h } = \sqrt { u _ { h } ^ { 2 } + v _ { h } ^ { 2 } }$ , its mean ${ \overline { { U } } } ,$ and net gradient $G = ( U _ { 1 2 } - U _ { 1 } ) / ( 1 1 \ : \mathrm { h } )$ . Wind is increasing when $G > 0 . 1 \mathrm { \dot { m } s ^ { - 1 } h ^ { - 1 } }$ and decreasing when $G < - 0 . 1$ . Remaining intervals are divided into low, middle and high speed using the training 20th and 80th percentiles, 4.4 and $9 . 7 ~ \mathrm { m s ^ { - 1 } }$ . Classification uses the unrounded thresholds; equality enters the low or high group. Small net trend does not imply constant wind, and the increasing/decreasing classes need not be monotonic.

Errors and normalization denominators are calculated within each station and wind class. Two NBP examples are selected by proximity to the respective class medians of initial $H _ { s } ,$ mean wind speed and net gradient, scaling each variable by its interquartile range. Selection does not use prediction errors.

## 3 Results

## 3.1 Spectral and wave-height accuracy

All learned models improve pooled spectral, wave-height, peak-frequency and peak-amplitude RMSE relative to persistence (Table 3). Within the descriptor comparison, variance and peakfrequency supervision (A2) gives the lowest spectral error, $0 . 4 0 ~ \mathrm { m ^ { 2 } H z ^ { - 1 } }$ , and peak-frequency error, $4 . 8 0 \times 1 0 ^ { - 2 }$ Hz. These errors are 2.63% and 3.09% lower than the baseline. Adding peak-amplitude supervision (A3) instead gives the lowest height and amplitude errors, 18.78 cm and $1 . 3 3 ~ \mathrm { m ^ { 2 } H z ^ { - 1 } }$ , respectively. Its spectral error is higher. Spectral shape, integrated variance and peak amplitude therefore favour diferent objectives.

Table 3: Pooled errors over leads 1–12 h. The two A2 rows are separate fits of the same supervised objective, one for each comparison. The WAM row provides regional context from Aguiar et al. (2024), rather than a matched benchmark.
<table><tr><td rowspan="2">Model</td><td colspan="3">Spectral RMSE</td><td colspan="5">Wave-height metrics</td></tr><tr><td> $E \ ( \mathrm { m ^ { 2 } H z ^ { - 1 } } )$ </td><td> $f _ { p } \ ( \mathrm { H z } )$ </td><td> $A _ { p } \ ( \mathrm { m } ^ { 2 } \mathrm { H z } ^ { - 1 } )$ </td><td>RMSE (cm) Bias (cm)</td><td></td><td></td><td></td><td>r</td><td>SI</td></tr><tr><td colspan="2">Descriptor comparison</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A0</td><td></td><td>0.41</td><td> $4 . 9 5 \times 1 0 ^ { - 2 }$ </td><td>1.60</td><td>19.54</td><td></td><td>-6.26</td><td>0.97</td><td>0.20</td></tr><tr><td>A1</td><td></td><td>0.41</td><td> $4 . 9 0 \times 1 0 ^ { - 2 }$ </td><td>1.48</td><td>19.00</td><td></td><td>-3.98</td><td>0.97</td><td>0.19</td></tr><tr><td>A2</td><td></td><td>0.40</td><td> $4 . 8 0 \times 1 0 ^ { - 2 }$ </td><td>1.56</td><td>19.62</td><td></td><td>-5.08</td><td>0.97</td><td>0.20</td></tr><tr><td>A3</td><td></td><td>0.43</td><td> $4 . 8 8 \times 1 0 ^ { - 2 }$ </td><td>1.33</td><td>18.78</td><td></td><td>-4.32</td><td>0.97</td><td>0.19</td></tr><tr><td colspan="2">Physical comparison</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A2 (reference)</td><td></td><td>0.40</td><td> $4 . 8 0 \times 1 0 ^ { - 2 }$ </td><td>1.61</td><td>20.28</td><td></td><td>-6.64</td><td>0.97</td><td>0.21</td></tr><tr><td>P1</td><td></td><td>0.43</td><td> $4 . 8 3 \times 1 0 ^ { - 2 }$ </td><td>1.80</td><td>21.27</td><td></td><td>-8.02</td><td>0.96</td><td>0.22</td></tr><tr><td>P2</td><td></td><td>0.40</td><td> $4 . 8 2 \times 1 0 ^ { - 2 }$ </td><td>1.57</td><td>19.26</td><td></td><td>-5.71</td><td>0.97</td><td>0.20</td></tr><tr><td>P3</td><td></td><td>0.39</td><td> $4 . 8 0 \times 1 0 ^ { - 2 }$ </td><td>1.46</td><td>19.66</td><td></td><td>-6.08</td><td>0.97</td><td>0.20</td></tr><tr><td>Persistence</td><td></td><td>0.61</td><td> $6 . 6 1 \times 1 0 ^ { - 2 }$ </td><td>2.37</td><td>39.76</td><td></td><td>-0.11</td><td>0.85</td><td>0.40</td></tr><tr><td>Aguiar WAM</td><td></td><td></td><td></td><td></td><td>19.00</td><td></td><td>2.00</td><td>0.97</td><td>0.19</td></tr></table>

Physical constraints also produce mixed efects. Relative to the matched A2 reference, wind-sea bounds (P1) increase spectral RMSE by 8.44% and height RMSE by 4.89%. Relaxation over twelve hours (P2) reduces height RMSE by 5.01% to 19.26 cm, with a small increase in spectral error. Extended relaxation (P3) gives the lowest spectral RMSE in this comparison, $0 . 3 9 ~ \mathrm { m ^ { 2 } H z ^ { - 1 } }$ , reducing spectral and height errors by 2.31% and 3.04%. The A2 reference retains the lowest peak-frequency error.

P3 spectral skill relative to persistence is 36.35%, while P2 height skill is 51.55%. All learned models underestimate $H _ { s }$ on average. P2, for example, has a bias of -5.71 cm and correlation r = 0.97. Errors generally increase with lead time, but more slowly than persistence (Fig. 1).

(a) Descriptor supervision  
![](images/5982af7b80a3fe7f4b6bdd27739967caed23547cd951f55b023b8ee72f03fa72.jpg)

(b) Descriptor supervision  
![](images/36434dc10cd598100147a154c36c6b31e3a7d0efc40e0d9a14a808a67ed15a04.jpg)

(c) Physical constraints  
![](images/89aebb3c62b07be348d113b987621c12e34db7d14306b574217cfef0858fbe7e.jpg)

(d) Physical constraints  
![](images/9f795cc4f3dd48cc831088f41196829d1bde712d1569a77c8ceb1c95e5442871.jpg)  
Figure 1: Spectral and wave-height RMSE by lead time: descriptor objectives (a,b) and physical constraints $^ \mathrm { ( c , d ) }$ . The lower panels use the matched A2 reference. All curves use the same 23,179 test origins; persistence repeats the initial spectrum.

## 3.2 Station diferences

Table 4 compares A2 and P3 by station. GoF has the smallest absolute spectral error, but its normalized error is close to that at NBP. The absolute contrast partly reflects lower observed spectral density at GoF. P3 reduces both spectral and height errors at BS and NBP; at GoF, spectral error decreases slightly while height error increases. Both models retain positive persistence skill at all three stations.

## 3.3 Dependence on wind conditions

The test contains 7,587 increasing-wind, 7,614 decreasing-wind and 1,436, 5,241 and 1,301 low-, middle- and high-speed small-trend origins. Absolute spectral errors are lowest in the low-speed class. For the A2 reference they range from 0.08 $\mathrm { m ^ { 2 } H z ^ { - 1 } }$ in that class to $0 . 8 1 ~ \mathrm { m ^ { 2 } H z ^ { - 1 } }$ in the high-speed class, whereas normalized errors are 0.81 and 0.49. Small absolute errors in weak seas thus coexist with substantial relative errors (Fig. 2).

P3 reduces pooled spectral error relative to A2 by 2.79% in increasing winds, 1.59% in decreasing winds and 2.80% in the high-speed small-trend class. Its largest height reductions among these classes occur in decreasing and high-speed winds, at 4.85% and 5.65%. In low-speed small-trend conditions the spectral change is below 0.1% and height error increases slightly. P1 is less favourable: spectral error increases by 20.70% during increasing winds. The physical constraints therefore do not improve every wind class.

Table 4: Station errors for the matched A2 reference and extended relaxation (P3). S is RMSE reduction relative to persistence.
<table><tr><td colspan="2"></td><td colspan="3">Spectral density</td><td colspan="3">Wave height</td></tr><tr><td>Station</td><td>Model</td><td>RMSE  $\left( \mathrm { m ^ { 2 } H z ^ { - 1 } } \right)$ </td><td>RMSE/RMS</td><td>S (%)</td><td>RMSE (cm)</td><td>SI</td><td>S (%)</td></tr><tr><td>BS</td><td>A2</td><td>0.48</td><td>0.56</td><td>33.06</td><td>22.54</td><td>0.23</td><td>49.69</td></tr><tr><td>BS</td><td>P3</td><td>0.47</td><td>0.54</td><td>34.55</td><td>21.47</td><td>0.22</td><td>52.10</td></tr><tr><td>GoF</td><td>A2</td><td>0.14</td><td>0.47</td><td>37.77</td><td>13.87</td><td>0.18</td><td>50.11</td></tr><tr><td>GoF</td><td>P3</td><td>0.14</td><td>0.45</td><td>39.29</td><td>13.96</td><td>0.18</td><td>49.78</td></tr><tr><td>NBP</td><td>A2</td><td>0.45</td><td>0.47</td><td>36.65</td><td>22.30</td><td>0.19</td><td>47.74</td></tr><tr><td>NBP</td><td>P3</td><td>0.44</td><td>0.46</td><td>38.16</td><td>21.77</td><td>0.19</td><td>48.98</td></tr></table>

![](images/57a6021b54d1122168cebb68e87d502d58243f3e40496a344d9374557205cdd0.jpg)

(b) BS  
![](images/dce3d6c37ebb6ad688e39baa75ae4d735e0d277b7f4def1a6ab0c7a4233291b3.jpg)

![](images/817e1547efe48f9b18520fcff0819fbc75db710172c35b5f8539bc6c63f7d03f.jpg)

(d) GoF  
![](images/eb100401882b27826c7bc31af2af7b57e197f6ff0d69fc3d48ee87df0fc92494.jpg)

![](images/6a179da2046aed50eb15c4851c1410569bb8a6646ab6a36a9b328f29efec86a0.jpg)

(f) NBP  
![](images/5341c28b588f01ab7f3efff48a59b562e17528674b9034d1f8526462e6c341aa.jpg)  
Figure 2: Wind-conditioned normalized errors for the matched A2 reference, extended relaxation (P3) and persistence. Rows show BS, GoF and NBP. Each value pools leads 1–12 within its station and wind class. Steady denotes small net wind-speed change, subdivided by mean speed; denominators use observations within each subset.

Figure 3 shows the A2 reference and P3 at one, six and twelve hours for two NBP origins selected from the increasing- and decreasing-wind classes. The observed spectra, persistence and supplied winds show how peak placement and density evolve within each interval. These cases illustrate the predictions without establishing general event-level performance.

(a) Increasing, 1 h  
![](images/8eafb69d5659cf7c436bc6d09f36a13bbccef2f0ef51740529157fcb69020533.jpg)

(b) Increasing, 6 h  
![](images/71cbe802f7ae702a88d1b98ff67f029feeb3e5f81d710a7e96cfcbb5bc880eaf.jpg)

(c) Increasing, 12 h  
![](images/95e9acd11bfddb0fc7bb6a67e5f2670054b92b96550b308effc31e3a193f62d1.jpg)

![](images/3708a5e710d0f2a047c7ae336c7ea2f5240f9e11037dbe79967b4812a0b7083a.jpg)

(e) Decreasing, 1 h  
![](images/0387f33f64e6a5390b5e09707ba22f85248c5d8fe86011e7206ae9a58a090562.jpg)

(f) Decreasing, 6 h  
![](images/9c08b76f660c08bb4c30e50f7d1de2356f04114ef08eab7b07729f8cabd0c39b.jpg)

(g) Decreasing, 12 h  
![](images/094be6bbda14624b7278a9fea19fa2fef0248346a716a9438d598328823430ea.jpg)

![](images/7cf9837bc2a33ed245307de6d46b7b7d7bc8667ab7abe7db7beadf99c3fde958.jpg)  
Figure 3: NBP spectral hindcasts and supplied wind for two forcing-selected origins: 30 June 2022, 21:00 UTC (a–d), and 2 December 2022, 14:00 UTC (e–h). Spectral panels compare observations, the matched A2 reference, P3 and persistence at one, six and twelve hours. Cases are selected from wind-class medians without using prediction errors.

## 3.4 Continuation beyond twelve hours

On the common 24-hour subset, P3 gives spectral RMSE 0.45 $\mathrm { m ^ { 2 } H z ^ { - 1 } }$ and height RMSE 24.87 cm at hours 13–24 (Fig. 4). Compared with A2, these are reductions of 0.76% and 1.18%. Compared with P2, P3 reduces spectral error by 6.30%, while height error increases by 1.46%. All three models outperform persistence. Extending the physical loss beyond the spectral targets therefore provides a modest spectral benefit, with a height tradeof.

![](images/07bd36f99b83172d7f2050fe664bfe65a56db850394ee9223c4fcb127448df61.jpg)

![](images/316a4dee6ef552eebc57fcbb8054b86e3b3d09da631204417b0b6ef7bbc91612.jpg)  
Figure 4: Continuation to 24 hours for the A2 reference, relaxation (P2), extended relaxation (P3) and persistence on 22,892 common test origins. The dashed line marks the end of spectral supervision. P3 alone received a physical residual at hours 13–24 during training.

## 4 Discussion

## 4.1 What physical guidance contributes

The main result is a tradeof between the quantities constrained during training. Supervision of variance and peak frequency favours spectral accuracy, whereas peak-amplitude supervision favours height and amplitude. The physical comparisons give a similar distinction: local relaxation improves height, and its extension beyond labelled hours improves spectral accuracy. Empirical upper bounds perform worse overall. Additional physical guidance is therefore useful only to the extent that its assumptions and weighting suit the observed wave evolution.

This study connects spectrum-prediction research (Song and Jiang, 2023; Gao and Jiang, 2025; Lu et al., 2026) with learning from in situ spectra (Filoche et al., 2026). Its contribution is the controlled comparison of spectral descriptors and local physical constraints for the subsequent evolution of a measured buoy spectrum. Related descriptor supervision has been used in inverse spectrum reconstruction (Sun et al., 2025); here, it is tested alongside empirical growth relations and a dynamical relaxation penalty in a time-evolving application.

The equilibrium relation links wind, fetch and depth, but several assumptions separate it from the measured state. Local depth approximates the bathymetry along the fetch. An instantaneous fetch and a small endpoint wind change do not establish that forcing has persisted long enough for equilibrium. The frequency cutof and fixed JONSWAP projection cannot resolve crossing directional systems. These limitations matter in the Baltic Sea, where wind direction changes the efective fetch and waves may persist after the generating wind changes (Soomere and Räämet, 2011; Björkqvist et al., 2021). A swell label based on wave age and wind–wave direction should not be equated with long-period oceanic swell.

Calibration allows departures from the estimated equilibrium; it does not independently validate that equilibrium. The bound tolerances and station exclusions are derived from training observations, and the 12-hour relaxation time is selected by residual criteria rather than measured as a physical adjustment time. Indeed, the selected relaxation closure alone does not outperform hourly persistence for eligible training transitions. Its usefulness is as a constraint on the learned spectra. The efects of the calibration rules, station masks and loss weights have not been isolated, so the present results support this particular formulation rather than universal parameter choices.

## 4.2 Wind dependence and regional context

The wind-class results add physical context to the pooled errors. P3 improves spectral accuracy in increasing, decreasing and steady strong winds, while its efect in steady weak winds is very small. This pattern is consistent with stronger physical guidance during suficiently windy conditions, but the diagnostic classes are not the same as the stepwise applicability masks. Initial spectra, seasons and station mixtures also difer between classes. The comparison is descriptive and cannot identify the physical penalty as the cause of a regime-specific improvement.

Station normalization is equally important. Lower absolute errors at GoF partly reflect its lower spectral energy scale. Diferences in wave climate, sheltering and fetch may also afect prediction dificulty, but their individual roles are not isolated. Supplying depth and fetch can encode station diferences even without an explicit station identifier.

For regional context, Aguiar et al. (2024) report operational WAM height bias of 2.00 cm, RMSE 19.00 cm, r = 0.97 and SI 0.19 at leads 1–12 h. The present height RMSEs are 19.26 cm for P2, 19.66 cm for P3 and 18.78 cm for A3, although WAM has a smaller bias magnitude. Diferences in forcing, frequency support and evaluation datasets prevent a matched performance ranking. Nevertheless, the similar reported height errors are encouraging given the simpler representation: the local approach uses one measured one-dimensional spectrum and local forcing, while operational WAM resolves directional spectra, propagation and source processes over a spatial grid.

## 4.3 Scope and further application

The retrospective wind information is a central limitation for operational use. A forecasting evaluation would require winds available at issue time and the actual availability time of the initial measured spectrum. Persistence establishes improvement over an unchanged sea state; a matched numerical-wave-model comparison and a stronger statistical baseline would be needed to establish practical advantages (Jiang et al., 2024).

Across 121,800 distinct station-hours eligible for the study windows, trapezoidal integration of native 0.025–1.0 Hz spectra and their retained portion gave a median variance retention of 96.95% and a median $H _ { s }$ reduction of 1.27 cm (95th percentile: 2.65 cm). On the same retained interval, regridding changed variance by a median absolute 0.52%, while the full endpoint weighting in Eq. 1 increased variance by a median 0.14% relative to trapezoidal integration. The 52-frequency grid thus retains most measured variance in typical conditions, although larger fractional losses occur in weak seas and some energetic NBP cases.

Directional information and spatial propagation are also absent. The principal comparisons use one fitted realization per objective, and overlapping origins repeatedly score related sea states; small diferences therefore should not be interpreted as statistically established improvements. Although training and checkpoint selection use the stated earlier years, 2022 informed earlier development, so the physical comparisons are exploratory.

The same prediction structure could potentially accept spectra estimated by another instrument, satellite retrieval or numerical model. Diferences in frequency support, averaging and observation error would first need to be accommodated and tested. Independent locations and operational forcing would then determine where a local measured-spectrum approach can complement existing wave-modelling systems.

## 5 Conclusion

Measured wave spectra provide a useful initial state for twelve-hour hindcasting with local wind and station information. All tested objectives improve spectral and wave-height errors relative to persistence, but the most accurate objective depends on the quantity of interest. Variance and peak-frequency supervision favours spectral accuracy; peak-amplitude supervision favours wave height and peak density.

Physical guidance produces modest and mixed changes. Local relaxation improves wave-height accuracy, while extending the relaxation constraint beyond the labelled interval improves spectral accuracy. Empirical wind-sea bounds worsen the pooled results. Wind-condition comparisons show that these efects vary with the forcing and wave-energy scale.

The method represents a local, reanalysis-conditioned application of data-driven and physics guided learning. Its calibration assumptions, band-limited spectra and evaluation at the training locations constrain the interpretation. Operational winds, independent locations and matched numerical-model comparisons are needed before extending the conclusions to practical forecasting.

## Acknowledgements

The wave-spectrum data were acquired through the Estonian Research Council Proof-of-Concept Grant EAG249. We thank the Finnish Meteorological Institute for providing the buoy spectra.

## Data availability

The buoy spectra were provided by the Finnish Meteorological Institute through project EAG249. The authors do not have permission to redistribute these spectra; access must be arranged with FMI. ERA5 forcing is from the Copernicus Climate Data Store product cited in Copernicus Climate Change Service (C3S) (2018). Bathymetry is from the Baltic Sea Bathymetry Database version 0.9.3, 500-m grid, downloaded on 3 December 2025 (Baltic Sea Hydrographic Commission, 2013). The coastline is from GSHHG version 2.3.7 at full resolution (Wessel and Smith, 2017). The code used for data preprocessing, model training and evaluation in this study is available from the corresponding author upon reasonable request.

## Author contributions and competing interests

Sander Rikka: conceptualization, modelling, manuscript writing and review. The author declares that they have no competing interests.

## References

Deborah Aguiar, Hedi Kanarik, Laura Tuomi, and Anni Jokiniemi. Evaluating accuracy of Baltic Sea wave forecasts. Geophysica, 59(1):3–19, 2024. URL https://www.geophysica.fi/article/ id-59-1-003-aguiar/.

Baltic Sea Hydrographic Commission. Baltic Sea Bathymetry Database (BSBD), version 0.9.3, 2013. URL https://www.bshc.pro/data/. 500-m grid, EPSG:3035; Arc-ASCII dataset; downloaded 3 December 2025.

Jan-Victor Björkqvist, Ingvar Lukas, Victor Alari, Gerbrant Ph. van Vledder, Sander Hulst, Heidi Pettersson, Arno Behrens, and Aarne Männik. Comparing a 41-year model hindcast with decades of wave measurements from the Baltic Sea. Ocean Engineering, 152:57–71, 2018. doi: 10.1016/j.oceaneng. 2018.01.048. URL https://doi.org/10.1016/j.oceaneng.2018.01.048.

Jan-Victor Björkqvist, Sander Rikka, Victor Alari, Aarne Männik, Laura Tuomi, and Heidi Pettersson. Wave height return periods from combined measurement–model data: a Baltic Sea case study. Natural Hazards and Earth System Sciences, 20(12):3593–3609, 2020. doi: 10.5194/nhess-20-3593-2020. URL https://doi.org/10.5194/nhess-20-3593-2020.

Jan-Victor Björkqvist, Siim Pärt, Victor Alari, Sander Rikka, Elisa Lindgren, and Laura Tuomi. Swell hindcast statistics for the Baltic Sea. Ocean Science, 17(6):1815–1829, 2021. doi: 10.5194/os-17-1815-2021. URL https://doi.org/10.5194/os-17-1815-2021.

Jan-Victor Björkqvist, Hedi Kanarik, Laura Tuomi, Lauri Niskanen, and Markus Kankainen. Event-based wave statistics for the Baltic Sea. State of the Planet, 4-osr8(0):1–8, 2024. doi: 10.5194/sp-4-osr8-10-2024. URL https://doi.org/10.5194/sp-4-osr8-10-2024.

N. Booij, R. C. Ris, and L. H. Holthuijsen. A third-generation wave model for coastal regions: 1. Model description and validation. Journal of Geophysical Research: Oceans, 104(C4):7649–7666, 1999. doi: 10.1029/98JC02622. URL https://doi.org/10.1029/98JC02622.

Jiaxin Chen, Ian G. C. Ashton, Edward C. C. Steele, and Ajit C. Pillai. A Real-Time Spatiotemporal Machine Learning Framework for the Prediction of Nearshore Wave Conditions. Artificial Intelligence for the Earth Systems, 2(1), 2023. doi: 10.1175/AIES-D-22-0033.1. URL https://doi.org/10.1175/ AIES-D-22-0033.1.

Copernicus Climate Change Service (C3S). ERA5 hourly data on single levels from 1940 to present, 2018. URL https://doi.org/10.24381/cds.adbb2d47. Dataset.

Svenja Ehlers, Norbert Hofmann, Tianning Tang, Adrian H. Callaghan, Rui Cao, Enrique M. Padilla, Yuxin Fang, and Merten Stender. Physics-informed neural networks for phase-resolved data assimilation and prediction of nonlinear ocean waves. Physical Review Fluids, 10(9):094901, 2025. doi: 10.1103/ytyy-pvys. URL https://doi.org/10.1103/ytyy-pvys.

European Union-Copernicus Marine Service. Baltic Sea Wave Analysis and Forecast, 2020. URL https://doi.org/10.48670/moi-00011. Product BALTICSEA\_ANALYSISFORECAST\_WAV\_003\_010; documentation accessed 14 October 2026.

Shuntao Fan, Nianhao Xiao, and Sheng Dong. A novel model to predict significant wave height based on long short-term memory network. Ocean Engineering, 205:107298, 2020. doi: 10.1016/j.oceaneng.2020.107298. URL https://doi.org/10.1016/j.oceaneng.2020.107298.

Arthur Filoche, Jef Hansen, Kevin Vinsen, and Travis Dawson. Site-Specific Post-Processing of Spectral Wave Forecast by Learning From Buoy Measurements. Journal of Geophysical Research: Machine Learning and Computation, 3(5):e2026JH001454, 2026. doi: 10.1029/2026JH001454. URL https: //doi.org/10.1029/2026JH001454.

Tianxiang Gao and Haoyu Jiang. Physics-Guided Deep Learning for Modeling Single-Point Wave Spectra Using Wind Inputs of Two Resolutions. Journal of Geophysical Research: Machine Learning and Computation, 2(2):e2024JH000492, 2025. doi: 10.1029/2024JH000492. URL https://doi.org/10. 1029/2024JH000492.

K. Hasselmann, T. P. Barnett, E. Bouws, H. Carlson, D. E. Cartwright, K. Enke, J. A. Ewing, H. Gienapp, et al. Measurements of Wind-Wave Growth and Swell Decay during the Joint North Sea Wave Project (JONSWAP). Ergänzungsheft zur Deutschen Hydrographischen Zeitschrift, Reihe A (8°) 12, Deutsches Hydrographisches Institut, Hamburg, 1973. URL https://resolver.tudelft.nl/uuid: f204e188-13b9-49d8-a6dc-4fb7c20562fc.

Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, András Horányi, Joaquín Muñoz Sabater, Julien Nicolas, Carole Peubey, et al. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020. doi: 10.1002/qj.3803. URL https://doi.org/10.1002/qj.3803.

Sepp Hochreiter and Jürgen Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8):1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735. URL https://doi.org/10.1162/neco.1997.9.8.1735.

Haoyu Jiang, Yuan Zhang, Chengcheng Qian, and Xuan Wang. Comment on papers using machine learning for significant wave height time series prediction: Complex models do not outperform autoregression. Ocean Modelling, 189:102364, 2024. doi: 10.1016/j.ocemod.2024.102364. URL https: //doi.org/10.1016/j.ocemod.2024.102364.

Arash Karimpour and Qin Chen. A Simplified Parametric Model for Fetch-Limited Peak Wave Frequency in Shallow Estuaries. Journal of Coastal Research, 32(4):954–965, 2016. doi: 10.2112/ JCOASTRES-D-15-00145.1. URL https://doi.org/10.2112/JCOASTRES-D-15-00145.1.

Yuting Lu, Lotfi Aouf, and Jiuke Wang. Global directional wave spectra prediction based on energyconserving deep learning. Ocean Modelling, 204:102807, 2026. doi: 10.1016/j.ocemod.2026.102807. URL https://doi.org/10.1016/j.ocemod.2026.102807.

M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019. doi: 10.1016/j.jcp.2018.10.045. URL https: //doi.org/10.1016/j.jcp.2018.10.045.

Anna Rutgersson, Erik Kjellström, Jari Haapala, Martin Stendel, Irina Danilovich, Martin Drews, Kirsti Jylhä, Pentti Kujala, et al. Natural hazards and extreme events in the Baltic Sea region. Earth System Dynamics, 13(1):251–301, 2022. doi: 10.5194/esd-13-251-2022. URL https://doi.org/10.5194/ esd-13-251-2022.

Yuhao Song and Haoyu Jiang. A Deep Learning–Based Approach for Empirical Modeling of Single-Point Wave Spectra in Open Oceans. Journal of Physical Oceanography, 53(9):2089–2103, 2023. doi: 10.1175/JPO-D-22-0198.1. URL https://doi.org/10.1175/JPO-D-22-0198.1.

Tarmo Soomere and Andrus Räämet. Spatial patterns of the wave climate in the Baltic Proper and the Gulf of Finland. Oceanologia, 53:335–371, 2011. doi: 10.5697/oc.53-1-TI.335. URL https: //doi.org/10.5697/oc.53-1-TI.335.

Lei Sun, Jun Wang, Zihao Li, Zilu Jiao, and Yuxiang Ma. G-PGNN: A physics-guided generative neural network model for discrete spectrum prediction. Ocean Engineering, 340:122411, 2025. doi: 10.1016/j.oceaneng.2025.122411. URL https://doi.org/10.1016/j.oceaneng.2025.122411.

Laura Tuomi and Hedi Kanarik. Product User Manual for Baltic Sea Waves Analysis and Forecasting Product BALTICSEA\_ANALYSISFORECAST\_WAV\_003\_010. Technical Report CMEMS-BAL-PUM-003-010, issue 2.8, EU Copernicus Marine Service, 2025. URL https://documentation.marine. copernicus.eu/PUM/CMEMS-BAL-PUM-003-010.pdf. November 2025; accessed 14 October 2026.

Lei Wang, Kezhao Fang, Kun Zheng, Jiawen Sun, Ping Wang, and Zhongbo Liu. Hybrid data-driven and physics-informed neural network modeling for regional spatiotemporal wave parameter prediction. Ocean Engineering, 358:125898, 2026. doi: 10.1016/j.oceaneng.2026.125898. URL https://doi.org/ 10.1016/j.oceaneng.2026.125898.

Xinxin Wang and Haoyu Jiang. Physics-guided deep learning for skillful wind-wave modeling. Science Advances, 10(49):eadr3559, 2024. doi: 10.1126/sciadv.adr3559. URL https://doi.org/10.1126/ sciadv.adr3559.

Paul Wessel and Walter H. F. Smith. A Global Self-consistent, Hierarchical, High-resolution Geography Database (GSHHG), version 2.3.7, 2017. URL https://www.soest.hawaii.edu/pwessel/gshhg/. Dataset, full-resolution shoreline; released 15 June 2017.

I.R. Young and L.A. Verhagen. The growth of fetch limited waves in water of finite depth. Part 1. Total energy and peak frequency. Coastal Engineering, 29(1-2):47–78, 1996. doi: 10.1016/S0378-3839(96) 00006-3. URL https://doi.org/10.1016/S0378-3839(96)00006-3.