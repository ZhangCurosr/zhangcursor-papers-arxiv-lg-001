# Frequency-aware forecasting for short-term typhoon gust prediction

Xuefei Wang<sup>1</sup>, Tingyi Liu<sup>2</sup>, Heng Zhang<sup>3</sup>, Shengjun Zhang<sup>1</sup>

<sup>1</sup>School of Artificial Intelligence, Hubei University, Wuhan, China. <sup>2</sup>School of Economics and Management, Wuhan University, Wuhan, China.

<sup>3</sup>School of Electrical Engineering, Shanghai Jiao Tong University, Shanghai, China.

Contributing authors: xuefeiw99@gmail.com; tingyiL@whu.edu.cn; zhangheng sjtu@sjtu.edu.cn; sj.zhang@hubu.edu.cn;

## Abstract

Accurate gust forecasting under typhoon conditions remains challenging due to the highly non-stationary and multi-scale characteristics of extreme wind fluctuations. Existing deep learning models often struggle to simultaneously capture long-term trends and rapid local variations, resulting in degraded performance during extreme events. We propose WDANet, a frequency-aware forecasting framework that integrates stationary wavelet decomposition, a Feature-wise Linear Modulation (FiLM) strategy, and a dual-branch encoder–decoder architecture, enabling separate modeling of trend and fluctuation components. Taking the ofshore regions of the Western Pacific in China as an example, we conduct finegrid wind gust prediction research. The results demonstrate that WDANet shows advantages for short lead times under the experimental setting across a 24-h forecasting horizon and achieves higher prediction accuracy than ECMWF-HRES within the first 6 h. During extreme wind events, WDANet more accurately captures gust peaks and attains the best RMSE and MAE performance. These results highlight its potential for ofshore wind power operation, disaster warning, and risk mitigation.

## 1 Introduction

Ofshore wind energy has emerged as a key pathway toward global carbon neutrality [1], with cumulative installed capacity projected to increase from 63 GW in 2022 to nearly 494 GW by 2030 [2]. Despite its substantial ofshore wind potential [3], a large fraction of China’s ofshore wind farms are located in the Northwest Pacific[4, 5], one of the most active tropical cyclone basins worldwide. In recent years, the frequency and intensity of typhoons in this region have shown an increasing trend [6–8], posing increasing challenges to the climate resilience and safe operation of ofshore wind infrastructure.

During typhoon events, ofshore wind turbines are exposed not only to elevated mean wind speeds but also to intense gusts [9]. Unlike mean wind speed, which represents temporally averaged atmospheric conditions, gusts correspond to short-duration wind peaks that impose significantly larger aerodynamic loads on turbine blades, towers, and support structures[10, 11]. Excessive gusts may trigger emergency shutdown procedures, accelerate structural fatigue, and in severe cases cause catastrophic failures[12, 13]. Consequently, operational safety and risk management in ofshore wind farms depend more directly on gust characteristics than on mean wind speed alone.

However, accurate gust forecasting remains substantially more challenging than mean wind speed prediction. Mean wind speed evolves relatively smoothly and is dominated by large-scale atmospheric processes, whereas gusts arise from localized turbulence, convection, and rapid energy transfer across multiple temporal scales [14]. Under typhoon conditions, gust behavior exhibits strong intermittency, abrupt transitions, and pronounced non-stationarity [9]. These characteristics make gust dynamics considerably more dificult to represent and predict using conventional forecasting models.

Despite their operational importance, most existing forecasting studies focus primarily on mean wind speed rather than gusts [15, 16]. Conventional forecasting targets are typically temporally averaged quantities that suppress short-term extreme fluctuations. As a result, models optimized for mean wind speed often fail to accurately represent transient gust peaks, creating a mismatch between forecasting objectives and the variables that govern turbine safety under typhoon conditions.

Wind speed forecasting methods can be broadly categorized into physics-based and data-driven approaches [17]. Numerical weather prediction (NWP) models provide physically interpretable forecasts but sufer from high computational cost and limited spatial-temporal resolution [18]. Data-driven approaches, including statistical models and deep learning architectures such as LSTM [19], GRU [20], and Transformer-based models like Informer [21] and SIGMAformer [22], have significantly improved predictive performance by capturing temporal dependencies. Nevertheless, despite their improved capability in modeling long-term temporal dependencies, existing deep learning approaches still struggle to accurately represent sudden and high-frequency wind variations under highly non-stationary typhoon conditions. This limitation is primarily attributed to the complex multi-scale nature of typhoon-driven wind fields, where low-frequency atmospheric evolution and high-frequency turbulent gusts are strongly coupled and exhibit highly intermittent behavior [23]. As a result, existing models tend to smooth out extreme short-term fluctuations, leading to suboptimal representation of gust dynamics that are critical for ofshore wind turbine safety.

To address multi-scale dynamics, a common strategy is to apply signal decomposition techniques such as discrete wavelet transform (DWT) or empirical wavelet transform (EWT) prior to forecasting [24, 25]. These approaches assume that decomposed sub-series within specific frequency bands exhibit simpler temporal patterns [26]. However, many existing decomposition-based methods rely on downsampling operations or non-shift-invariant filtering, which may introduce two potential limitations: (i) reduced temporal resolution, and (ii) sensitivity to time shifts that can afect the alignment of transient gust events. These methods often treat decomposed frequency components as relatively independent forecasting targets, which may overlook the cross-scale interactions between low-frequency atmospheric evolution and high-frequency gust fluctuations.

To overcome these limitations, this work introduces a Wavelet Dual-branch Attention Network (WDANet) for gust forecasting. Unlike traditional wavelet-based methods, SWT is fully non-decimated and shift-invariant, preserving temporal resolution across all decomposition levels. This property ensures that transient gust structures remain temporally aligned throughout the decomposition process, which is essential for accurately capturing abrupt wind changes.

Within WDANet, wind speed series are decomposed into low-frequency components capturing smooth atmospheric evolution and high-frequency components representing rapid gust-induced fluctuations. Furthermore, we design a frequencyaware learning strategy. The low-frequency component is used to model long-term atmospheric trends, while the high-frequency component is reformulated as a residual dynamic process emphasizing transient gust deviations. Considering that high-frequency gust bursts are not independent stochastic noise but are physically modulated by the evolving low-frequency typhoon background field, we introduce Feature-wise Linear Modulation (FiLM) [27]. FiLM explicitly models the dependency between large-scale typhoon evolution and small-scale gust fluctuations by dynamically adjusting feature representations conditioned on the low-frequency atmospheric state. This allows a physically consistent integration of multi-scale wind dynamics.

The main contributions of this work are summarized as follows:

• We identify that temporal misalignment, loss of shift-invariance, and independence assumptions in decomposition-based methods are key bottlenecks for gust-level wind forecasting under typhoon conditions.

• We propose a shift-invariant SWT-based multi-resolution forecasting framework that preserves full temporal resolution and maintains temporal coherence of transient gust structures.

• We introduce a frequency-aware learning strategy that explicitly decouples smooth atmospheric evolution and high-frequency gust dynamics in a physically consistent manner.

• The proposed method achieves superior gust forecasting accuracy against other machine learning models and outperforms ECMWF-HRES in the short term.

## 2 Results

## 2.1 Evaluation metrics

To comprehensively evaluate the forecasting performance, three complementary metrics are adopted in this work, including the root mean squared error (RMSE), mean absolute error (MAE), and relative accuracy (RAcc). RMSE and MAE are commonly used to quantify the prediction errors in wind gust forecasting, and their definitions are given as follows:

$$
R M S E = \sqrt { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( G _ { j } - \hat { G } _ { j } ) ^ { 2 } }\tag{1}
$$

$$
M A E = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left| \boldsymbol { G } _ { j } - \hat { \boldsymbol { G } } _ { j } \right|\tag{2}
$$

where N denotes the number of samples in the test set, $G _ { j }$ represents the observed wind gust value of the j-th sample, and $\hat { G } _ { j }$ denotes the corresponding predicted wind gust value. RMSE assigns larger penalties to samples with greater prediction deviations, making it more sensitive to extreme forecasting errors and abrupt gust events. In contrast, MAE measures the average absolute deviation over all samples, providing a more robust and interpretable evaluation of the overall forecasting accuracy.

Although RMSE and MAE are efective for quantifying absolute prediction errors, they cannot reflect the relative forecasting accuracy under diferent wind speed conditions. Considering the significant variability of wind gust intensity during typhoon evolution, evaluating the normalized prediction error is essential for comparing forecasting performance across diferent wind speed levels. Therefore, the relative accuracy (RAcc), derived from the mean absolute percentage error (MAPE), is further introduced:

$$
R A c c = 1 - M A P E = 1 - \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left| \frac { G _ { j } - \hat { G } _ { j } } { G _ { j } } \right|\tag{3}
$$

RAcc evaluates the forecasting performance by measuring the prediction deviation relative to the ERA5 reference gust, thereby providing a normalized assessment across diferent wind speed regimes. A higher RAcc value indicates that the predicted wind gust is closer to the observation, demonstrating stronger relative forecasting capability and better model generalization.

Furthermore, to provide a more detailed evaluation of the model capability in capturing extreme wind gust events, two additional metrics, namely peak error (PE) and peak time error (PTE), are introduced. PE quantifies the amplitude discrepancy between the predicted and observed peak gust values, while PTE measures the temporal shift between the predicted and observed peak occurrence times. They are defined as follows:

$$
P E = \left| G _ { p e a k } - \hat { G } _ { p e a k } \right|\tag{4}
$$

$$
P T E = \left| t _ { p e a k } - \hat { t } _ { p e a k } \right|\tag{5}
$$

where $G _ { p e a k }$ and $\hat { G } _ { p e a k }$ represent the maximum wind gust values in the observed and predicted sequences, respectively. $t _ { p e a k }$ and $\hat { t } _ { p e a k }$ denote the corresponding time indices of the observed and predicted peak gusts. PE directly characterizes the accuracy of extreme gust intensity prediction, whereas PTE evaluates the capability of the model to accurately capture the arrival time and occurrence moment of sudden gust events. These two metrics complement the aforementioned global error measurements and provide essential evidence for assessing the practical applicability and reliability of the proposed model under extreme weather conditions.

## 2.2 Comparison methods analysis

To validate the efectiveness of the proposed WDANet framework for wind gust forecasting, several representative models with diferent architectures are selected for comparison on the test dataset. The details of these baseline methods are summarized as follows:

CNN-LSTM: A hybrid architecture that employs convolutional layers to extract local temporal features and integrates LSTM networks to model sequential dependencies, thereby capturing the temporal dynamics of wind gust evolution [28].

Autoformer: A Transformer-based architecture that utilizes series decomposition and an autocorrelation mechanism to capture long-term dependencies and periodic patterns in time series data [29].

DLinear: An MLP-based model [30] that decomposes the input sequence into trend and seasonal components and applies linear mappings for forecasting.

TimesNet: A CNN-based architecture [31] that transforms one-dimensional time series into two-dimensional representations through fast Fourier transform (FFT), enabling the modeling of multi-period temporal variations from multiple perspectives.

iTransformer: A Transformer-based time series forecasting architecture [32] that retains the original Transformer structure but modifies the roles of the attention mechanism and feed-forward network through an inverted data representation.

PatchTST: A Transformer-based framework employing patching and channel independence mechanisms [33]. It treats local temporal segments as input tokens, enabling efective extraction of long-term dependencies while reducing computational complexity.

ECMWF-HRES: A high-resolution operational numerical weather prediction system developed by the European Centre for Medium-Range Weather Forecasts (ECMWF), which integrates advanced atmospheric dynamical and physical models with data assimilation techniques and is widely recognized as one of the leading global weather forecasting systems [34–36].

All the experiments are conducted on the same equipment, and the relevant parameters are shown in Supplementary Table 1 and Supplementary Table 2. Among these methods, CNN-LSTM represents a classical deep learning approach for temporal sequence modeling, whereas Autoformer, DLinear, TimesNet, iTransformer and PatchTST are representative state-of-the-art (SOTA) forecasting algorithms widely adopted in recent time series forecasting studies. All data-driven models were trained on the same dataset using an identical batch size of 32 and trained for 20 epochs, ensuring a fair comparison. To further improve the reliability of the evaluation and reduce the influence of randomness, five random seeds (seed = 7, 42, 123, 999, and 2021) were employed for repeated experiments. The parameters for all machine learning models are provided in Supplementary Table 3.

The ECMWF-HRES (European Centre for Medium-Range Weather Forecasts High Resolution system) is one of the most accurate numerical weather prediction models worldwide. During the testing phase, forecasts from ECMWF-HRES were collected for the same locations and periods as the test dataset, covering all selected typhoon events. The forecasts were initialized at 00:00 UTC with matching spatial and hourly temporal resolutions, and the forecast horizons were aligned with those of WDANet. This evaluation strategy ensures consistency and fairness between the proposed model and the operational numerical weather prediction baseline under identical forecasting conditions.

Tables 1, 2, and 3 present the performance metrics of diferent models over the 24- hour forecasting horizon, where the RMSE, MAE, and RAcc are all calculated based on hourly forecast results. Figure 1 further illustrates the variation trends of these three metrics with respect to forecast lead time within the 24-hour prediction range. The overall performance metrics are presented in Supplementary Table 4.

As shown in Figure 1a, the RMSE values of all models exhibit an increasing trend with increasing forecasting lead time, which is mainly attributed to the accumulation of prediction uncertainty and error propagation during wind gust evolution. Compared with the data-driven baseline approaches, the proposed WDANet consistently achieves the lowest RMSE across all forecasting horizons. Specifically, the RMSE values at 1-h and 24-h lead times reach 0.84 m/s and 2.79 m/s, respectively. Compared with the best-performing baseline model iTransformer, WDANet reduces the forecasting error by approximately 5.6%–6.1% across diferent lead times. Compared with Autoformer, the error reduction reaches 18.9%–52.8%.

A similar performance trend is observed for the MAE metric, where WDANet maintains the lowest prediction errors among all data-driven models. At the final forecasting step, WDANet achieves an MAE of only 1.85 m/s, representing improvements of approximately 8.9% and 26.3% compared with iTransformer (2.03 m/s) and Autoformer (2.51 m/s), respectively. These results indicate that WDANet provides improved multi-step forecasting capability among data-driven approaches and efectively alleviates error accumulation during extended prediction horizons.

The RAcc results further characterize the relative forecasting capability of diferent models. CNN-LSTM achieves a slightly higher RAcc than the proposed model, indicating its advantage in preserving the overall temporal variation tendency. This phenomenon may be attributed to the smoothing characteristics of CNN-LSTM, which enable it to capture dominant temporal trends while suppressing certain high-frequency fluctuations. Nevertheless, WDANet still outperforms most other baseline models, with only a marginal decrease of approximately 0.01 compared with CNN-LSTM over the entire forecasting period. Meanwhile, WDANet achieves the lowest RMSE and MAE values, indicating that although its trend consistency is slightly diferent from CNN-LSTM, it provides more accurate predictions of wind gust intensity.

To further assess the statistical robustness of model performance, a block bootstrap analysis was conducted based on individual typhoon events rather than hourly samples, using a fixed random seed of 42 with 1,000 resampling iterations. This strategy preserves the temporal dependence within each typhoon process and provides more reliable uncertainty estimation. The 95% confidence intervals of RMSE, MAE, and RAcc are presented in Figure 2.

WDANet achieves the lowest RMSE (2.27 m/s) with a narrow confidence interval of [2.17, 2.37] m/s, which is consistently lower than all baseline models. Similar behavior is observed for MAE, where WDANet obtains the smallest mean value (1.51 m/s) and maintains stable uncertainty across diferent typhoon events. Although CNN-LSTM achieves a slightly higher RAcc, WDANet provides lower RMSE and MAE, indicating more accurate estimation of wind gust intensity. These results demonstrate that the improvement of WDANet is robust across diferent typhoon cases rather than being dominated by individual events.

The superior performance of WDANet can be attributed to its frequency-aware forecasting mechanism. By applying stationary wavelet transform, the proposed model decomposes the input sequence into low-frequency and high-frequency components and processes them through independent encoder-decoder pathways. Such a design enables the separation of temporal characteristics at diferent frequency levels and reduces the interference between components. Compared with treating the input sequence as a unified signal, this frequency decomposition strategy improves the stability of multi-step forecasting. Furthermore, the decoupled architecture may reduce error propagation across forecasting horizons by independently modeling high-frequency variations and low-frequency trends. Consequently, the prediction error increases more slowly over longer forecasting periods, which is consistent with the observed trends in RMSE, MAE, and RAcc.

Compared with ECMWF-HRES, WDANet demonstrates competitive short-term forecasting capability, particularly within the first 6h. At the 1-h lead time, ECMWF-HRES obtains an RMSE of 1.93 m/s, whereas WDANet achieves an RMSE of 0.84 m/s, corresponding to a reduction of 56.5%. Similar advantages are observed in MAE and RAcc during the early forecasting period. However, as the forecasting lead time increases, the performance degradation of WDANet becomes more significant than that of ECMWF-HRES, indicating that ECMWF-HRES possesses stronger capability in capturing long-term atmospheric evolution patterns. The comparison with the operational numerical weather prediction system demonstrates that WDANet provides competitive short-term wind gust forecasting performance within 6 h. Combined with its higher computational eficiency, this result provides confidence in the potential application of WDANet in wind power.

## 2.3 Ablation Study

To further investigate the contribution of individual components, we conducted ablation experiments under five diferent random seeds to ensure statistical robustness.

The results are summarized in Table 4 and Table 5, including the efects of wavelet configurations, frequency branches, and attention mechanisms.

The influence of wavelet basis and decomposition depth is reported in Table 4. Under one-level decomposition, sym4 achieved the best performance with an RMSE of 2.28 and MAE of 1.55, while db2 and db4 showed slightly higher errors. Haar obtained relatively worse results, possibly due to its discontinuous basis function, which is less suitable for representing smooth meteorological variations. Increasing the decomposition depth beyond one level did not further improve performance. For sym4, two- and three-level decompositions increased RMSE to 2.59 and 2.51, respectively, indicating that excessive decomposition may reduce the efective temporal information available for sequence modeling.

The role of SWT was examined by comparing the complete model with learnable SWT filters, the fixed-SWT variant, and the model without wavelet decomposition. Removing SWT increased the RMSE from 2.28 to 2.92 and MAE from 1.55 to 1.89, indicating that frequency decomposition provides efective multi-scale representations for wind gust forecasting. Although the RAcc values of the two configurations remain similar, the larger error metrics without SWT demonstrate that wavelet decomposition mainly contributes to reducing prediction deviations. The fixed-SWT variant achieved an RMSE of 2.38, which was better than removing SWT but still inferior to the learnable version. This result suggests that the performance improvement originates not only from frequency decomposition itself but also from the adaptive adjustment of frequency responses.

The contribution of the FiLM-based cross-frequency modulation was further evaluated by removing the FiLM module from the complete architecture. As shown in Table 5, the variant without FiLM achieved an RMSE of 2.30, compared with 2.28 obtained by the full model. Although the MAE value of the w/o FiLM variant was slightly lower, the decrease in RMSE and RAcc indicates that FiLM mainly improves the model’s ability to capture large-magnitude forecasting deviations rather than uniformly reducing average errors. This behavior is important for wind gust forecasting under typhoon conditions, where extreme fluctuations contribute disproportionately to prediction uncertainty. By adaptively generating modulation parameters from complementary frequency components, FiLM enables dynamic information exchange between low-frequency background trends and high-frequency transient variations. Without this cross-frequency interaction, the model relies on independently encoded frequency representations, limiting its capability to adjust feature responses according to rapidly changing atmospheric states.

The necessity of the dual-branch architecture was evaluated by retaining only the approximation branch or the detail branch. Both variants showed degraded performance compared with the complete model, with RMSE values of 2.33 and 2.32, respectively. The approximation branch mainly captures low-frequency trends, whereas the detail branch describes short-term fluctuations. The performance gap indicates that accurate wind gust forecasting requires the joint modeling of stable temporal evolution and rapid variations.

The efect of the hybrid attention mechanism was further analyzed by replacing both branches with only global attention or only local attention. The global-attentiononly and local-attention-only variants achieved RMSE values of 2.34 and 2.39, respectively, both higher than the complete model. Global attention provides stronger long-range dependency modeling but may weaken the representation of abrupt local changes, while local attention improves short-term feature extraction at the cost of global temporal consistency. The proposed hybrid attention mechanism efectively combines these complementary capabilities, leading to improved forecasting performance.

These results verify that the proposed architecture benefits from the collaborative efects of adaptive SWT, FiLM-based cross-frequency modulation, dual-frequency representations, and hybrid attention. Each component addresses a specific limitation in wind gust forecasting, and their combination enables more accurate and robust prediction under complex typhoon conditions.

## 2.4 Performance under extreme typhoon conditions

In the previous section, the average forecasting performance of diferent models was evaluated over the complete typhoon-active period. However, extreme wind gust events involve stronger nonlinear variations and larger uncertainties, providing a more challenging scenario for evaluating forecasting robustness. Therefore, based on the same test dataset used in Section 2.2, two subsets were further extracted to evaluate model performance under more severe conditions: samples with maximum wind speeds exceeding 15 m/s were classified as typhoon-afected cases, and samples whose maximum wind gust values fell within the top 10% of all typhoon cases were defined as extreme typhoon cases, with wind gust intensities ranging from 28.52 to 46.85 m/s.

Figure 3a presents the hourly MAE of all models over the 24-h forecasting horizon under extreme typhoon cases. With increasing forecasting lead time, all models exhibit a rapid increase in MAE, indicating that the strong nonlinear characteristics and increased uncertainty of extreme typhoon events significantly increase the dificulty of forecasting. During the early forecasting period, iTransformer achieves relatively better performance. However, after 12 h, WDANet consistently maintains the lowest MAE, demonstrating its superior robustness under extreme wind gust conditions.

Under extreme typhoon conditions, WDANet achieves the lowest 24-h averaged MAE of 7.02 m/s, outperforming CNN-LSTM $\left( 7 . 4 7 ~ \mathrm { m / s } \right)$ , iTransformer $\left( 7 . 1 0 ~ \mathrm { m / s } \right)$ PatchTST (8.21 m/s), TimesNet (8.28 m/s), DLinear $( 8 . 9 0 ~ \mathrm { m / s } )$ , and Autoformer $\left( 9 . 4 6 ~ \mathrm { m / s } \right)$ . Compared with Autoformer, WDANet reduces the error by 25.8%, while the improvements over TimesNet and DLinear reach 15.2% and 21.1%, respectively. This demonstrates that WDANet is more capable of estimating extreme wind gust intensity under highly variable typhoon conditions. Averaged over the entire 24-h forecasting horizon, WDANet also achieves a competitive overall MAE of 3.87 m/s for all typhoon-afected cases, confirming its applicability across diferent wind intensity regimes.

Beyond the overall average, the hour-by-hour MAE variation for all typhoonafected cases is further examined (Figure 3b). WDANet achieves the lowest MAE over most forecasting horizons, while CNN-LSTM shows slightly lower errors during the final few hours. This indicates that CNN-LSTM can efectively capture dominant temporal trends under moderate conditions, which is consistent with our observations in the overall performance comparison. In contrast, WDANet provides stronger advantages for rapidly varying and high-amplitude wind gust events. The standard deviation analysis in Supplementary Fig. 1 and Fig. 2 further shows that WDANet maintains relatively lower prediction variability across forecasting horizons, demonstrating improved consistency under diverse typhoon scenarios.

Overall, these results highlight the efectiveness of WDANet for extreme wind gust forecasting. By combining adaptive frequency decomposition and cross-frequency feature interaction, WDANet reduces error accumulation and provides reliable predictions under nonlinear and rapidly evolving typhoon conditions.

## 2.5 Case study of Typhoon Yagi

To evaluate the practical forecasting capability of WDANet under extreme typhoon conditions, Typhoon Yagi (2411) is selected as a representative case study. As the strongest autumn typhoon to make landfall in China since 1949, Typhoon Yagi caused the collapse of multiple wind turbines in Hainan Province during its passage [37], resulting in considerable economic losses and highlighting the importance of reliable gust forecasting for the safe operation of wind energy systems. Haikou, Hainan Province, China (20.0<sup>◦</sup>N, 110.25<sup>◦</sup>E), is selected as the target location. During this event, the wind field exhibits pronounced non-stationary characteristics, with irregular fluctuations before landfall, rapid intensification during the main impact period, and subsequent decay, resulting in substantial variations in gust intensity (Figure 4). Based on the experimental settings described above, model performance is evaluated from two complementary perspectives: forecasting accuracy over the entire month containing Typhoon Yagi and predictive capability during the extreme high-wind period.

Supplementary Fig. 3 summarizes the forecasting errors of diferent models over the entire monthly period associated with Typhoon Yagi. This monthly-scale evaluation reflects the capability of diferent models to continuously track both background wind variations and typhoon-induced disturbances. WDANet achieves the best overall performance among all compared models, with an MAE of 1.70 m/s and an RMSE of 2.50 m/s. Compared with the second-best model, iTransformer (MAE: 1.73 m/s, RMSE: 2.57 m/s), WDANet reduces the MAE and RMSE by approximately 1.7% and 2.7%, respectively. Compared with TimesNet (MAE: 1.82 m/s, RMSE: 2.61 m/s), WDANet further reduces the MAE and RMSE by approximately 6.6% and 4.2%, respectively. These results demonstrate that WDANet can efectively capture the complex temporal variations associated with typhoon-induced gusts and maintain stable forecasting performance over extended prediction periods.

To further investigate the model capability under extreme wind conditions, Figure 5 presents the forecasting results during the most intense 24-hour high-wind period of Typhoon Yagi. This period covers the rapid strengthening and decay stages of the typhoon, with peak gust intensity exceeding 31 m/s, representing a highly challenging scenario characterized by strong nonlinear fluctuations. WDANet achieves the lowest MAE of 6.82 m/s and the lowest RMSE of 8.14 m/s among all compared models. Compared with TimesNet, WDANet reduces the MAE by approximately 9.3% and achieves comparable RMSE performance (8.14 m/s versus 8.33 m/s). Compared with Autoformer, WDANet reduces the MAE and RMSE by approximately 37.6% and 36.0%, respectively. These results indicate that WDANet exhibits stronger robustness in responding to rapidly varying extreme gust conditions.

In addition to conventional error metrics, peak-related metrics are employed to further evaluate the capability of diferent models in characterizing extreme gust events. As shown in Figure 5, WDANet achieves the lowest peak time error (PTE) of 9 hours, indicating superior capability in identifying the occurrence timing of extreme gusts. The accurate peak timing prediction highlights the timeliness of WDANet in capturing the onset of hazardous wind conditions, while its lower sequence-level errors demonstrate reliable forecasting performance under rapidly evolving typhoon environments. Although TimesNet achieves a smaller peak error (PE), indicating better reconstruction of the maximum gust magnitude at a specific time point, its higher MAE suggests less consistent prediction performance throughout the entire extreme period. Therefore, WDANet provides a better balance between overall forecasting accuracy, temporal response capability, and extreme-event robustness.

The diferent rankings among MAE, RMSE, PE, and PTE reveal the complementary characteristics of these evaluation metrics. MAE and RMSE quantify the overall deviation across the forecasting sequence, whereas PE focuses only on the error at the maximum gust point and may be sensitive to isolated peak reconstruction. Therefore, a smaller PE does not necessarily indicate superior extreme-event forecasting capability. For Typhoon Yagi, TimesNet obtains the lowest PE, suggesting a more accurate estimation of the maximum gust amplitude, while WDANet achieves lower sequencelevel errors and more accurate peak timing. This advantage can be attributed to the dual-frequency decomposition and adaptive fusion strategy of WDANet, which enables simultaneous modeling of slowly varying wind evolution and high-frequency gust fluctuations. Consequently, WDANet provides more stable, timely, and reliable forecasts for extreme typhoon-induced gust events, demonstrating its potential for wind energy safety management and early-warning applications.

## 3 Discussion

Accurate and reliable forecasting of extreme typhoon events is of great importance for ofshore wind power development, particularly in regions frequently afected by severe weather conditions. In this study, we propose WDANet, an efective multivariate datadriven framework specifically designed for wind gust forecasting (WGF). Compared with existing deep learning models, WDANet achieves superior performance in terms of RMSE, MAE, and RAcc. Compared with the ECMWF-HRES numerical weather prediction system, WDANet demonstrates significant advantages in short-term forecasting, achieving reductions of more than 50% in RMSE and MAE under certain forecasting horizons. Extensive evaluations on typhoon cases, extreme typhoon events, and the representative Typhoon Yagi case demonstrate that WDANet can provide more accurate and timely wind gust forecasts for key ofshore wind energy regions.

In short-term forecasting, WDANet directly learns the nonlinear mapping relationship between historical wind gust data and future wind gust predictions, enabling superior predictive capability. In contrast, ECMWF-HRES relies on physical process parameterizations to represent atmospheric evolution, where parameterization schemes inherently introduce approximation errors [38], and uncertainties in the initial conditions are unavoidable. However, for longer forecasting horizons, ECMWF-HRES benefits from the physical constraints imposed by atmospheric governing equations and exhibits stronger long-term stability. Since WDANet does not incorporate explicit physical constraints, forecasting errors may be nonlinearly amplified during autoregressive prediction.

Despite these advantages, several limitations remain in this study. The experiments are conducted using ERA5 reanalysis data with a spatial resolution of $0 . 2 5 ^ { \circ } \times 0 . 2 5 ^ { \circ }$ Although this resolution is widely adopted in atmospheric studies and provides relatively high-quality meteorological information, it may still be insuficient to capture finer-scale wind variations. It should be noted that the release of ERA5 data is subject to significant latency. According to the oficial documentation provided by the Copernicus Climate Change Service, ERA5 data are typically available with a delay of approximately 5 days. Therefore, although this study demonstrates the technical feasibility and accuracy of grid-point wind gust forecasting based on historical reanalysis data, the proposed model cannot directly support real-time operational deployment. In practical operational scenarios, the model’s inputs must be replaced with near-realtime data streams. Future research should focus on evaluating the model’s performance after integrating real-time forecast inputs, in order to validate its true operational potential. Furthermore, spatial information is not incorporated into the current framework to avoid potential deployment challenges associated with cross-station data sharing and transmission constraints. The evaluation is currently limited to typhoon events in the western North Pacific, and the applicability of WDANet to other ocean basins requires further investigation. Furthermore, WDANet still exhibits limitations in reproducing extremely high wind gust peaks. Similar behavior is also observed in other deep learning models evaluated in this study, suggesting that accurate extreme peak reconstruction remains a challenging problem for data-driven approaches. This limitation may be partly associated with the limited availability of extreme wind events in the training dataset, which restricts the model’s ability to learn rare high-intensity gust patterns.

Future studies can be extended in several directions. First, numerical weather prediction outputs, such as ECMWF forecasts, can be incorporated as auxiliary inputs to combine physical constraints with data-driven representations, which may improve long-term forecasting and extreme peak prediction. Second, transfer learning strategies can be explored to adapt the proposed framework to diferent geographical regions and ofshore wind farm locations. Third, data augmentation and cost-sensitive learning approaches can be investigated to alleviate the impact of the limited availability of extreme event samples during training. As wind power generation continues to expand, public concern regarding energy stability and reliability has intensified. Accurate wind gust forecasts can strengthen public confidence in wind energy as a sustainable power source, positioning it as a more mature and competitive energy option.

## 4 Methods

## 4.1 Dataset and preprocessing

Meteorological data used in this study are derived from the ERA5 reanalysis dataset provided by the European Centre for Medium-Range Weather Forecasts (ECMWF) [39]. The spatial resolution of the selected data is $0 . 2 5 ^ { \circ } \times 0 . 2 5 ^ { \circ }$ and the temporal resolution is 1 h. The selected variables include sea level pressure, surface pressure, 2 m air temperature, surface temperature, dew point temperature, relative humidity, and 10 m wind gust since previous post-processing. Relative humidity is computed as the ratio of actual vapor pressure to saturation vapor pressure. The pressure field of a tropical cyclone plays a crucial role in determining its wind structure, as horizontal pressure gradients govern the intensity and organization of wind fields [40]. Meanwhile, temperature and moisture jointly influence the TC pressure field through their efects on air density[41]. These factors collectively afect the vertical distribution of pressure deficit and hence the resulting wind field within the TC boundary layer, and they also jointly influence the development of turbulence and the eficiency of vertical momentum transport within the boundary layer, which are the primary mechanisms responsible for the generation and amplification of wind gusts.

A total of 400 typhoon events afecting the study region during the period from 1960 to 2025 are identified. For each typhoon event, hourly ERA5 reference gust data at the grid point corresponding to the station are extracted for the entire month during which the typhoon occurs. This strategy ensures that each sample contains a continuous time series covering pre-typhoon, typhoon, and post-typhoon phases, rather than isolated extreme segments. As a result, the dataset captures both stable background conditions and highly non-stationary dynamics within a unified temporal context, allowing the model to learn transitions between normal and extreme regimes.

All samples were organized in chronological order to preserve the temporal structure of the data. Missing values were first handled using forward filling, while missing values at the beginning of sequences were filled using backward filling. Any remaining missing values were replaced with zeros. To determine an appropriate historical input length, the temporal dependency of wind gusts was investigated using autocorrelation analysis (Supplementary Fig. 4). The results show that gust autocorrelation remains significant within the first 24 h but decreases rapidly at longer time lags. Therefore, all models adopted a 24-hour sliding window to construct input sequences, with a sliding step of 1 hour. All variables were normalized using min-max normalization to ensure consistent feature scales and improve training stability. All preprocessing parameters, including statistics for missing-value imputation and normalization parameters, were calculated exclusively from the training set. The validation and test sets were transformed using the parameters derived from the training set to prevent any form of information leakage. The dataset was split chronologically into training (70%, including 280 typhoon events), validation (15%, including 60 typhoon events), and test (15%, including 60 typhoon events) sets. The split was performed at the file level to ensure that continuous time-series samples from the same file were not distributed across diferent subsets, thereby enabling a reliable evaluation of the model’s generalization capability. The spatial distribution of the 60 selected typhoon events in the test set and the corresponding ERA5 extraction locations is illustrated in Figure 6. The dataset covers the western North Pacific region $( 1 0 0 ^ { \circ } \mathrm { E - 1 5 0 ^ { \circ } E , 1 0 ^ { \circ } N - 4 0 ^ { \circ } N } )$ , where tropical cyclones frequently occur and significantly afect ofshore wind energy regions.

## 4.2 Task formulation

The wind gust forecasting (WGF) task under typhoon conditions is formulated as a multi-step time series forecasting problem. Given a sequence of historical meteorological reanalysis data over the past M time steps, the objective is to predict the future wind gust evolution over a forecasting horizon of length H.

At each time step i, the atmospheric state is represented as a vector:

$$
\begin{array} { r } { \mathbf { x } _ { i } = [ \mathrm { S l p } _ { i } , \mathrm { S p } _ { i } , \mathrm { T 2 m } _ { i } , \mathrm { S t } _ { i } , \mathrm { T d } _ { i } , \mathrm { R h } _ { i } , G _ { i } ] ^ { \mathrm { T } } \in \mathbb { R } ^ { d } , } \end{array}\tag{6}
$$

where Slp, Sp, T2m, St, Td, and Rh denote sea level pressure, surface pressure, 2-meter air temperature, surface temperature, dew point temperature, and relative humidity, respectively, while $G _ { i }$ represents the 10m wind gust since previous post-processing at time step i.

Given the historical sequence:

$$
{ \bf x } _ { N - M : N - 1 } = \left[ { \bf x } _ { N - M } , { \bf x } _ { N - M + 1 } , . . . , { \bf x } _ { N - 1 } \right] ,\tag{7}
$$

the goal is to learn a nonlinear mapping:

$$
\hat { \mathbf { G } } _ { N : N + H - 1 } = f \left( \mathbf { x } _ { N - M : N - 1 } \right) ,\tag{8}
$$

where $\hat { \mathbf { G } } _ { N : N + H - 1 } \in \mathbb { R } ^ { H }$ denotes the predicted wind gust sequence over the next H time steps.

## 4.3 Learnable stationary wavelet transform module

To capture the frequency-dependent characteristics of wind gust signals, we employ the Stationary Wavelet Transform (SWT) [42] to decompose the input multivariate sequence into frequency-specific components. The implementation follows the convolution-based formulation described in [43]. Unlike the conventional Discrete Wavelet Transform (DWT), SWT eliminates the downsampling operation at each decomposition level, thereby preserving all temporal samples and achieving shift invariance. This property avoids the shift sensitivity introduced by decimation in DWT, where small temporal displacements may lead to significant variations in wavelet coeficients [42, 44]. Consequently, SWT maintains the temporal correspondence between slowly varying background components and transient fluctuation components, which is particularly important for accurately localizing short-duration wind gust events.

Specifically, the SWT is implemented using one-dimensional convolutions with a pair of low-pass and high-pass filters, denoted as h and g. Following the \`a trous algorithm, the filters are progressively dilated at diferent decomposition levels. Let ${ \bf a } _ { t } ^ { ( 0 ) } = { \bf x } _ { t }$ denote the input sequence. At decomposition level s, the approximation and detail coeficients are obtained as

$$
\mathbf { a } _ { t } ^ { ( s + 1 ) } = \sum _ { k } h ^ { ( s ) } ( k ) \mathbf { a } _ { t + k } ^ { ( s ) } , \quad \mathbf { d } _ { t } ^ { ( s + 1 ) } = \sum _ { k } g ^ { ( s ) } ( k ) \mathbf { a } _ { t + k } ^ { ( s ) } ,\tag{9}
$$

where $h ^ { ( s ) }$ and $g ^ { ( s ) }$ denote the dilated filters obtained by inserting $2 ^ { s } \ - \ 1$ zeros between adjacent coeficients of the original filters. The filtering operation is implemented as one-dimensional convolution (equivalent to cross-correlation in deep learning frameworks), which preserves the sequence length and maintains temporal alignment between diferent frequency components.

Rather than directly adopting fixed wavelet filters, the proposed method initializes the low-pass and high-pass kernels using the Symlet-4 (sym4) wavelet basis and allows them to be optimized during model training. The sym4 initialization provides a physically meaningful frequency decomposition prior, while the subsequent optimization enables the filters to adapt their spectral selectivity according to the intrinsic characteristics of wind gust signals. As illustrated in Supplementary Fig. 5, the learned filters preserve the fundamental low-pass and high-pass characteristics of the initialized wavelet basis while adjusting their frequency responses and spectral weighting. This adaptive refinement allows the decomposition to better represent the distinct temporal-frequency behaviors of wind gust variations, where large-scale atmospheric evolution and short-lived gust fluctuations exhibit diferent characteristic variability patterns.

In this work, a single-level SWT decomposition is employed, producing a lowfrequency component $\mathbf { A } = \mathbf { a } ^ { ( 1 ) }$ and a high-frequency component $\mathbf { D } = \mathbf { d } ^ { ( 1 ) }$ . Unlike [43], where SWT is mainly used for multi-scale token generation, the decomposed components in this study are directly fed into two dedicated forecasting branches. This frequency-aware architecture enables diferent branches to specialize in modeling slowly varying atmospheric evolution and rapidly changing gust fluctuations.

To further investigate the impact of the learned wavelet decomposition on feature organization, the correlation matrices before and after SWT processing are visualized in Figure 7. The original meteorological variables exhibit relatively weak and difuse correlation patterns (Figure 7a), reflecting heterogeneous relationships among diferent atmospheric variables. In contrast, the representations obtained after the sym4-initialized adaptive SWT exhibit more structured block-like correlation patterns (Figure 7b). This observation suggests that the proposed decomposition facilitates the organization of frequency-dependent meteorological information while preserving the intrinsic relationships among input variables.

The selection of a single decomposition level is empirically validated through an ablation study (see Section 2.3). The results demonstrate that $m = 1$ consistently achieves the best forecasting performance. Increasing the decomposition depth does not provide additional improvements and may even degrade performance due to excessive frequency separation, information fragmentation, and the amplification of noise components in higher-frequency bands. Overall, the proposed sym4-initialized adaptive SWT module provides a physically motivated frequency decomposition mechanism that separates slowly varying atmospheric evolution from transient gust fluctuations, enabling subsequent forecasting components to efectively capture the multi-timescale characteristics of wind gust signals.

## 4.4 Frequency-aware encoder-decoder architecture

The overall architecture of WDANet is illustrated in Figure 8. After frequency decomposition by SWT, the input meteorological sequence is separated into a lowfrequency component A and a high-frequency component D. These two components are processed by two independent encoder-decoder branches to capture diferent temporal characteristics. Specifically, the low-frequency branch focuses on large-scale atmospheric evolution, while the high-frequency branch models transient wind gust fluctuations. To establish interactions between the two frequency components, a FiLMbased cross-frequency modulation module is introduced, where the low-frequency representation adaptively recalibrates the high-frequency features. Finally, the outputs from the two branches are reconstructed through additive fusion to obtain the final wind gust prediction.

## 4.4.1 Encoder

While the SWT-based decomposition captures multi-scale frequency characteristics of meteorological variables, it does not explicitly model temporal dependencies across time steps. To address this limitation, two structurally identical but parameterindependent bidirectional LSTM encoders are employed to independently process the trend component A and the fluctuation component D.

For each branch, let the input sequence be denoted as $\mathbf { X } \in \mathbb { R } ^ { L \times C }$ , where L is the sequence length and C is the number of variables. The sequence is first projected into a latent space through a linear embedding layer:

$$
\mathbf { Z } = \mathbf { X } \mathbf { W } _ { \mathrm { e m b } } ,\tag{10}
$$

where $\mathbf { W } _ { \mathrm { e m b } } \in \mathbb { R } ^ { C \times D }$ is the learnable projection matrix and D is the embedding dimension.

The embedded sequence is then processed by the BiLSTM:

$$
\mathbf { H } , ( \mathbf { h } _ { e } , \mathbf { c } _ { e } ) = \mathrm { B i L S T M } ( \mathbf { Z } ) ,\tag{11}
$$

where H $\in \mathbb { R } ^ { L \times 2 H }$ denotes the sequence of hidden states, and $\mathbf h _ { e } , \mathbf c _ { e } \in \mathbb { R } ^ { S _ { 1 } \times 2 H }$ are the final hidden and cell states, with $S _ { 1 }$ being the number of LSTM layers and H the hidden dimension of each direction.

The two encoders share an identical structure but do not share parameters, allowing each branch to specialize in modeling distinct temporal dynamics associated with diferent frequency bands. The encoded representations H are subsequently passed to the decoder through attention mechanisms for generating wind gust predictions.

## 4.4.2 Attention mechanisms

To efectively utilize the encoded temporal representations, two distinct attention mechanisms are designed for the trend and fluctuation branches, respectively. Both mechanisms compute a set of attention weights over the encoder outputs, which are subsequently used by the decoder to construct context vectors.

For the trend branch processing A, global attention is adopted to capture longrange dependencies. The attention weights are computed as

$$
\alpha _ { t , i } ^ { ( \mathbf { A } ) } = \mathrm { s o f t m a x } \Big ( \mathbf { v } ^ { \top } \operatorname { t a n h } \big ( \mathbf { W } [ \mathbf { s } _ { t - 1 } ^ { ( \mathbf { A } ) } ; \mathbf { H } _ { i } ^ { ( \mathbf { A } ) } ] \big ) \Big ) ,\tag{12}
$$

where ${ \bf s } _ { t - 1 } ^ { ( \mathbf { A } ) }$ denotes the decoder hidden state at the previous time step, $\mathbf { H } _ { i } ^ { ( \mathbf { A } ) }$ is the i-th encoder output, W and v are learnable parameters, and the softmax is taken over all encoder time steps $i = 1 , \ldots , L$ . This global formulation allows the model to attend to any position in the input sequence when generating trend predictions.

To better capture short-term and high-frequency fluctuations, a local attention mechanism with a learnable temporal prior is introduced in the fluctuation branch. Unlike global attention, which considers all time steps equally, this mechanism restricts the attention focus to a dynamically predicted local region, thereby improving sensitivity to transient variations.

Specifically, a soft alignment position is first predicted from the decoder hidden state:

$$
p _ { t } = L \cdot \sigma ( \mathbf { W } _ { p } \mathbf { s } _ { t - 1 } ^ { ( \mathbf { D } ) } ) ,\tag{13}
$$

where L denotes the sequence length and $\sigma ( \cdot )$ is the sigmoid function that ensures the predicted position lies within a valid temporal range.

Next, a content-based compatibility score between the decoder state and encoder outputs is computed using a dot-product form:

$$
e _ { t , i } = \langle \mathbf { W } \mathbf { s } _ { t - 1 } ^ { ( \mathbf { D } ) } , \mathbf { H } _ { i } ^ { ( \mathbf { D } ) } \rangle ,\tag{14}
$$

where $\mathbf { H } _ { i } ^ { ( \mathbf { D } ) }$ represents the encoder output at time step i.

To incorporate locality information, a Gaussian temporal prior centered at $p _ { t }$ defined as:

$$
G ( i , p _ { t } ) = \exp \left( - \frac { ( i - p _ { t } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{15}
$$

where σ controls the width of the local attention window.

The final attention distribution is obtained by combining content relevance and the temporal prior in a normalized form:

$$
\alpha _ { t , i } ^ { ( \mathbf { D } ) } = \frac { \exp ( \boldsymbol { e } _ { t , i } ) \cdot G ( i , p _ { t } ) } { \sum _ { j = 1 } ^ { L } \exp ( \boldsymbol { e } _ { t , j } ) \cdot G ( j , p _ { t } ) } .\tag{16}
$$

This formulation enables the model to focus on a temporally constrained neighborhood while preserving content-based relevance, which is particularly suitable for modeling high-frequency wind gust fluctuations.

## 4.4.3 Decoder and cross-frequency modulation

The decoder generates future wind gust values in an autoregressive manner. At each decoding step $t ,$ the context vector is first obtained by aggregating the encoder outputs with the attention weights:

$$
\mathbf { c } _ { t } ^ { ( b ) } = \sum _ { i = 1 } ^ { L } \alpha _ { t , i } ^ { ( b ) } \mathbf { H } _ { i } ^ { ( b ) } ,\tag{17}
$$

where $b \in \{ { \bf A } , { \bf D } \}$ denotes the frequency branch.

The decoder then updates its hidden state as

$$
\mathbf { s } _ { t } ^ { ( b ) } = \mathrm { L S T M } \big ( [ \phi _ { \mathrm { i n } } ( \hat { y } _ { t - 1 } ^ { ( b ) } ) ; \mathbf { c } _ { t } ^ { ( b ) } ] , \mathbf { s } _ { t - 1 } ^ { ( b ) } \big ) ,\tag{18}
$$

where $\hat { y } _ { t - 1 } ^ { ( b ) }$ denotes the previous prediction, $\phi _ { \mathrm { i n } } ( \cdot )$ is a linear embedding layer mapping the scalar input into a $2 H .$ -dimensional representation, and $[ \cdot ; \cdot ]$ denotes concatenation. The decoder hidden and cell states are initialized from the corresponding bidirectional encoder outputs.

To enable information exchange between frequency components, a FiLM-based cross-frequency modulation mechanism is introduced after the decoder state update. Specifically, the low-frequency branch generates adaptive modulation parameters for the high-frequency representation:

$$
[ \gamma _ { t } , \beta _ { t } ] = f _ { \theta } ( \mathbf { m } _ { t } ^ { ( \mathbf { A } ) } ) ,\tag{19}
$$

where $f _ { \theta } ( \cdot )$ denotes a multilayer perceptron, and $\mathbf { m } _ { t } ^ { ( \mathbf { A } ) }$ represents the intermediate decoder feature of the low-frequency branch.

The high-frequency feature is then recalibrated as

$$
\tilde { \mathbf { m } } _ { t } ^ { ( \mathbf { D } ) } = \gamma _ { t } \odot \mathbf { m } _ { t } ^ { ( \mathbf { D } ) } + \beta _ { t } ,\tag{20}
$$

where $\odot$ denotes element-wise multiplication. This modulation enables the highfrequency branch to adaptively adjust its response according to the low-frequency atmospheric evolution state.

Finally, the branch outputs are obtained through linear projections:

$$
\hat { y } _ { t } ^ { ( \mathbf { A } ) } = \mathbf { W } _ { \mathrm { o u t } } ^ { ( \mathbf { A } ) } [ \mathbf { s } _ { t } ^ { ( \mathbf { A } ) } ; \mathbf { c } _ { t } ^ { ( \mathbf { A } ) } ] ,\tag{21}
$$

$$
\hat { y } _ { t } ^ { ( \mathbf { D } ) } = \mathbf { W } _ { \mathrm { o u t } } ^ { ( \mathbf { D } ) } [ \tilde { \mathbf { m } } _ { t } ^ { ( \mathbf { D } ) } ; \mathbf { c } _ { t } ^ { ( \mathbf { D } ) } ] ,\tag{22}
$$

where $\mathbf { W } _ { \mathrm { o u t } } ^ { ( b ) } \in \mathbb { R } ^ { 1 \times 4 H }$

## 4.4.4 Output reconstruction

The final wind gust prediction is obtained by summing the outputs of the two branches:

$$
\hat { y } _ { t } = \hat { y } _ { t } ^ { ( \mathbf { A } ) } + \hat { y } _ { t } ^ { ( \mathbf { D } ) } .\tag{23}
$$

This additive formulation reflects the decomposition of wind gust dynamics into low-frequency trend and high-frequency fluctuation components via the stationary wavelet transform. By separately modeling each component and combining their predictions, the model captures both large-scale atmospheric evolution and transient local variations in a complementary manner.

## 5 Tables

Table 1 RMSE↓ results of diferent models for WGF
<table><tr><td>Models</td><td>0h</td><td>1h</td><td>2h</td><td>3h</td><td>6h</td><td>9h</td><td>12h</td><td>18h</td><td>21h</td><td>23h</td></tr><tr><td>CNN-LSTM</td><td>0.91</td><td>1.25</td><td>1.52</td><td>1.72</td><td>2.13</td><td>2.39</td><td>2.61</td><td>2.90</td><td>2.99</td><td>3.05</td></tr><tr><td>Autoformer</td><td>1.78</td><td>1.96</td><td>2.10</td><td>2.20</td><td>2.44</td><td>2.70</td><td>2.89</td><td>3.15</td><td>3.23</td><td>3.44</td></tr><tr><td>TimesNet</td><td>0.97</td><td>1.30</td><td>1.56</td><td>1.76</td><td>2.21</td><td>2.48</td><td>2.69</td><td>3.06</td><td>3.21</td><td>3.29</td></tr><tr><td>iTransformer</td><td>0.89</td><td>1.24</td><td>1.49</td><td>1.68</td><td>2.07</td><td>2.32</td><td>2.50</td><td>2.77</td><td>2.89</td><td>2.97</td></tr><tr><td>PatchTST</td><td>0.89</td><td>1.30</td><td>1.59</td><td>1.82</td><td>2.29</td><td>2.57</td><td>2.77</td><td>3.01</td><td>3.06</td><td>3.12</td></tr><tr><td>DLinear</td><td>0.88</td><td>1.32</td><td>1.64</td><td>1.88</td><td>2.31</td><td>2.54</td><td>2.71</td><td>2.87</td><td>2.90</td><td>2.95</td></tr><tr><td>ECMWF</td><td>1.93</td><td>1.81</td><td>1.83</td><td>1.94</td><td>2.07</td><td>1.72</td><td>1.78</td><td>1.67</td><td>1.85</td><td>1.79</td></tr><tr><td>Ours</td><td>0.84</td><td>1.21</td><td>1.46</td><td>1.64</td><td>1.98</td><td>2.21</td><td>2.39</td><td>2.63</td><td>2.72</td><td>2.79</td></tr></table>

0 h, 1 h to 23 h represent the forecast horizons, with 0 h indicating the first prediction step based on the previous 24-hour input sequence. ↓ indicates that lower values are better.

Table 2 MAE↓ results of diferent models for WGF
<table><tr><td>Models</td><td>0h</td><td>1h</td><td>2h</td><td>3h</td><td>6h</td><td>9h</td><td>12h</td><td>18h</td><td>21h</td><td>23h</td></tr><tr><td>CNN-LSTM</td><td>0.62</td><td>0.88</td><td>1.07</td><td>1.20</td><td>1.47</td><td>1.63</td><td>1.75</td><td>1.93</td><td>2.00</td><td>2.06</td></tr><tr><td>Autoformer</td><td>1.22</td><td>1.34</td><td>1.44</td><td>1.51</td><td>1.66</td><td>1.84</td><td>2.02</td><td>2.24</td><td>2.30</td><td>2.51</td></tr><tr><td>TimesNet</td><td>0.66</td><td>0.90</td><td>1.08</td><td>1.22</td><td>1.51</td><td>1.69</td><td>1.83</td><td>2.04</td><td>2.15</td><td>2.22</td></tr><tr><td>iTransformer</td><td>0.60</td><td>0.87</td><td>1.05</td><td>1.18</td><td>1.43</td><td>1.60</td><td>1.71</td><td>1.87</td><td>1.95</td><td>2.03</td></tr><tr><td>PatchTST</td><td>0.60</td><td>0.91</td><td>1.13</td><td>1.29</td><td>1.60</td><td>1.78</td><td>1.88</td><td>2.00</td><td>2.02</td><td>2.05</td></tr><tr><td>DLinear</td><td>0.57</td><td>0.92</td><td>1.17</td><td>1.35</td><td>1.67</td><td>1.82</td><td>1.92</td><td>2.01</td><td>2.01</td><td>2.04</td></tr><tr><td>ECMWF</td><td>1.52</td><td>1.36</td><td>1.35</td><td>1.43</td><td>1.50</td><td>1.27</td><td>1.31</td><td>1.24</td><td>1.36</td><td>1.29</td></tr><tr><td>Ours</td><td>0.56</td><td>0.85</td><td>1.03</td><td>1.16</td><td>1.39</td><td>1.53</td><td>1.63</td><td>1.76</td><td>1.81</td><td>1.85</td></tr></table>

0 h, 1 h to 23 h represent the forecast horizons, with 0 h indicating the first prediction step based on the previous 24-hour input sequence.

Table 3 Relative accuracy (RAcc)↑ results of diferent models for WGF
<table><tr><td>Models</td><td>0h</td><td>1h</td><td>2h</td><td>3h</td><td>6h</td><td>9h</td><td>12h</td><td>18h</td><td>21h</td><td>23h</td></tr><tr><td>CNN-LSTM</td><td>0.90</td><td>0.87</td><td>0.84</td><td>0.82</td><td>0.78</td><td>0.75</td><td>0.73</td><td>0.70</td><td>0.69</td><td>0.68</td></tr><tr><td>Autoformer</td><td>0.77</td><td>0.74</td><td>0.72</td><td>0.71</td><td>0.68</td><td>0.65</td><td>0.61</td><td>0.56</td><td>0.54</td><td>0.50</td></tr><tr><td>TimesNet</td><td>0.88</td><td>0.83</td><td>0.79</td><td>0.77</td><td>0.73</td><td>0.70</td><td>0.67</td><td>0.63</td><td>0.61</td><td>0.60</td></tr><tr><td>iTransformer</td><td>0.89</td><td>0.84</td><td>0.80</td><td>0.78</td><td>0.73</td><td>0.70</td><td>0.68</td><td>0.66</td><td>0.63</td><td>0.62</td></tr><tr><td>PatchTST</td><td>0.89</td><td>0.83</td><td>0.79</td><td>0.76</td><td>0.71</td><td>0.68</td><td>0.66</td><td>0.64</td><td>0.64</td><td>0.63</td></tr><tr><td>DLinear</td><td>0.90</td><td>0.83</td><td>0.78</td><td>0.73</td><td>0.66</td><td>0.62</td><td>0.59</td><td>0.58</td><td>0.59</td><td>0.58</td></tr><tr><td>ECMWF</td><td>0.74</td><td>0.80</td><td>0.81</td><td>0.81</td><td>0.81</td><td>0.80</td><td>0.73</td><td>0.74</td><td>0.71</td><td>0.76</td></tr><tr><td>Ours</td><td>0.90</td><td>0.84</td><td>0.81</td><td>0.78</td><td>0.74</td><td>0.72</td><td>0.71</td><td>0.69</td><td>0.68</td><td>0.67</td></tr></table>

0 h, 1 h to 23 h represent the hours from 00:00 to 23:00, with 0 h being the first hour predicted based on the input historical data. ↑ indicates that higher values are better.

Table 4 Sensitivity analysis of wavelet basis and decomposition level.
<table><tr><td>Metric</td><td> ${ \mathrm { H a a r ~ } } ( m { = } 1 )$ </td><td> $\mathrm { d b 2 ~ ( } m \mathrm { = } 1 )$ </td><td> $\mathrm { d b } 4 \ ( m { = } 1 )$ </td><td> $\mathrm { s y m } 4 \ ( m { = } 1 )$ </td><td> $\mathrm { s y m 4 ~ } ( m { = } 2 )$ </td><td> $\mathrm { s y m 4 ~ } ( m { = } 3 )$ </td></tr><tr><td>RMSE↓</td><td> $2 . 4 7 \pm 0 . 3 1$ </td><td> $2 . 3 8 \pm 0 . 1 1$ </td><td> $2 . 4 1 \pm 0 . 1 2$ </td><td> ${ \bf 2 . 2 8 \pm 0 . 0 3 }$ </td><td> $2 . 5 9 \pm 0 . 5 8$ </td><td> $2 . 5 1 \pm 0 . 4 2$ </td></tr><tr><td>MAE↓</td><td> $1 . 6 5 \pm 0 . 2 0$ </td><td> $1 . 5 9 \pm 0 . 0 5$ </td><td> $1 . 6 0 \pm 0 . 0 6$ </td><td> ${ \bf 1 . 5 5 \pm 0 . 0 4 }$ </td><td> $1 . 7 4 \pm 0 . 4 6$ </td><td> $1 . 6 9 \pm 0 . 3 1$ </td></tr><tr><td>RAcc↑</td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $0 . 7 2 \pm 0 . 0 3$ </td><td> $0 . 7 2 \pm 0 . 0 3$ </td><td> $\mathbf { 0 . 7 3 \ : \pm { \ : 0 . 0 1 } }$ </td><td> $0 . 7 1 \pm 0 . 0 4$ </td><td> $0 . 7 1 \pm 0 . 0 3$ </td></tr></table>

Diferent wavelet bases are compared under a fixed decomposition level (m = 1), while the efect of decomposition depth is evaluated using the sym4 wavelet.

Table 5 Ablation study on model components.
<table><tr><td>Metric</td><td> $\mathrm { w } / \mathrm { o } \ \mathrm { S W T }$ </td><td> $\mathrm { w } / \mathrm { o } \ \mathrm { F i L M }$ </td><td>fixed SWT</td><td>Only A</td><td>Only D</td><td></td><td> $\mathrm { O n l y ~ g l o b a l }$ </td><td>Only local</td><td>Ours</td></tr><tr><td>RMSE↓</td><td> $2 . 9 2 \pm 0 . 1 0$ </td><td> $2 . 3 0 \pm 0 . 0 6$ </td><td> $2 . 3 8 \pm 0 . 0 9$ </td><td> $2 . 3 3 \pm 0 . 0 4$ </td><td></td><td> $2 . 3 2 \pm 0 . 0 3$ </td><td> $2 . 3 4 \pm 0 . 0 5$ </td><td> $2 . 3 9 \pm 0 . 1 5$ </td><td> ${ \bf 2 . 2 8 \pm 0 . 0 3 }$ </td></tr><tr><td>MAE↓</td><td> $1 . 8 9 \pm 0 . 0 5$ </td><td> ${ \bf 1 . 5 3 \pm 0 . 0 4 }$ </td><td> $1 . 6 1 \pm 0 . 0 6$ </td><td> $1 . 5 6 \pm 0 . 0 3$ </td><td> $1 . 5 4 \pm 0 . 0 1$ </td><td> $1 . 6 0 \pm 0 . 0 9$ </td><td></td><td> $1 . 6 0 \pm 0 . 1 6$ </td><td> ${ \bf 1 . 5 5 \pm 0 . 0 4 }$ </td></tr><tr><td>RAcc↑</td><td> $0 . 7 3 \pm 0 . 0 2$ </td><td> $0 . 7 2 \pm 0 . 0 1$ </td><td> $0 . 7 1 \pm 0 . 0 4$ </td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $0 . 7 2 \pm 0 . 0 1$ </td><td> $0 . 6 9 \pm 0 . 0 4$ </td><td></td><td> $0 . 7 0 \pm 0 . 0 5$ </td><td> $\mathbf { 0 . 7 3 \ : \pm { \ : 0 . 0 1 } }$ </td></tr></table>

w/o SWT: without wavelet decomposition; w/o FiLM: without FiLM-based cross-frequency modulation; fixed SWT: stationary wavelet transform with fixed (non-trainable) filters; Only A: approximation (low-frequency) branch only; Only D: detail (highfrequency) branch only; Only global/local: using only global or local attention; Ours: full model with learnable SWT, FiLM-based cross-frequency modulation, dual-branch structure, and hybrid attention.

## 6 Figures

![](images/d89c37c9b5de20bb8ec025ea21490a54e1f4f16c53e9b53c3fb1dd49a45978c7.jpg)

![](images/bd5478b0e65b21d337b4ab09b951b57c5eb0d5566e845437c34cf3facf982317.jpg)

![](images/1a29e925b45b795369f4c49987b44564c4626d79286bed6da1200067e416e5ad.jpg)  
Fig. 1 Performance comparison of diferent models for 24h wind gust prediction. a is RMSE result. b is MAE result. c is RAcc result. Lower RMSE/MAE and higher RAcc indicate better performance. A broken x-axis is used to distinguish short-term (0–12 h) and long-term (12–24 h) prediction intervals. The proposed model demonstrates competitive and stable performance across diferent lead times.

![](images/3f7ae1210567eccb2606c461683899df8348e495ff8b59c24de9115cb92bb5f4.jpg)

![](images/1a6809657e75cc9f613c14b4099c769ec5317e91e0960145521eef15f7fb4845.jpg)

![](images/5b44c5dcb1007e11748e0e8b88a510bf4c9dd7a0d8a91b7eef66ee216ffb34d8.jpg)  
Fig. 2 95% confidence intervals of model performance obtained from typhoon-event block bootstrap. a is RMSE result. b is MAE result. c is RAcc result. The markers represent the mean values of each metric, and the error bars indicate the 95% confidence intervals.

a)  
![](images/3fff8edd29371eb8e58ed667ccc13cdb112d825e5934690af953f6f40d411443.jpg)

b)  
![](images/c0c08d24376a46e90281d45dfa7177539b2f9f670ef253b6f476def9dd1a2294.jpg)  
Fig. 3 Performance comparison of multi-step wind gust forecasting under diferent typhoon scenarios. a MAE under typhoon conditions with maximum wind gusts ranging from 28.52 m/s to 46.85 m/s. b MAE under typhoon conditions with maximum wind gusts exceeding 15 m/s. The horizontal axis denotes the forecast horizon, and the vertical axis represents the error.

![](images/f6f782e25a60aa32e732fd52f53c949299c1993092f644a9c7671d8635f4cf00.jpg)  
Fig. 4 Case study of Typhoon Yagi. Comparison between predicted results and ground truth in Haikou, Hainan Province, China (20.0° N, 110.25° E) under the input-24/predict-24 setting during Typhoon Yagi, from September 2 to September 30, 2024.  
WDANet CNN-LSTM Autoformer DLinear TimesNet iTransformer PatchTST

![](images/00f7d5bb4da9fb4498d167860f95867e038d8ca06c7b84d5f17a83710557cd95.jpg)

![](images/d966f4dbfb28b77e27efddb1563ba37766d09dc610fac40bdccc3e5611eeb42c.jpg)

![](images/9d35a0d01dfa991ddfe448b7b491ac96bf0f1d4588b4e089560eac088819fc44.jpg)

![](images/6e1577c9d3483bfd3ce5d5ae11e18ad8f8a63e86b8f348175433c5b8c69517c5.jpg)  
Fig. 5 Forecasting performance of diferent models during the most intense 24-h gust period of Typhoon Yagi. Performance comparison of diferent forecasting models during the most intense 24-h gust period of Typhoon Yagi. The selected period represents the consecutive 24 hours with the highest average gust intensity, covering the most challenging extreme wind conditions. The evaluation includes a RMSE, b MAE, c peak error (PE), and d peak time error (PTE). Lower values indicate better performance.

![](images/2ccb6080715424eedf35d56773e0c96840b6aecfde7dece6b30edafe0d0f3750.jpg)  
Fig. 6 Western North Pacific Typhoon Tracks (100°E 150°E, 10°N 40°N.) Wind-Speed Gradient and Intensity Classification Spatial distribution of typhoon tracks over the western North Pacific basin (100°E–150°E, 10°N–40°N). Tracks are color-coded according to tropical cyclone intensity classification, with the wind-speed gradient (units: $\mathrm { m } / \mathrm { s } )$ indicated by the vertical color bar on the right. Yellow triangles indicate the ERA5 reanalysis data extraction locations.

![](images/6a8ecb2c1a423ec98d5c49e6ec155d144a46f9af86c6a9cc0f2f72b793a04d2c.jpg)

![](images/ce8fbbed625fd0f204d92961e894d7f79f27213efe47a6025da797992b9a5414.jpg)  
Fig. 7 Correlation analysis of input features and learned representations based on the Symlet-4 wavelet decomposition. a Correlation matrix of the original features. b Correlation matrix of the learned channel representations after wavelet-based decomposition.

![](images/30043351e150fb3691cbcac9f3dad3cd895a738043fa78648622b50403f436bf.jpg)  
Fig. 8 Architecture of WDANet: an illustration of the dual-branch network with SWT decomposition, separate encoder-decoders (global/local attention), and FiLM-based cross-frequency modulation for final prediction.

Supplementary information. We have Supplementary information.

Acknowledgements. We extend our sincere gratitude to the researchers at ECMWF for their invaluable contributions to the collection, archival, dissemination, and maintenance of the ERA5 reanalysis dataset and ECMWF HRES forecast data.

## Declarations

• Funding

Not applicable.

• Competing interests

The authors declare no competing interests.

• Ethics approval and consent to participate Not applicable.

• Consent for publication

Not applicable.

• Data availability

ERA5 reanalysis data were obtained from the Copernicus Climate Data Store at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels? tab=download. ECMWF-HRES forecast data were obtained from the European Centre for Medium-Range Weather Forecasts at https://www.ecmwf.int/en/ forecasts/datasets/set-i. The tropical cyclone best-track data used for generating the typhoon track schematic map can be accessed at https://www.ncei.noaa.gov/ products/international-best-track-archive.

• Materials availability

Not applicable.

• Code availability

The code for the comparison algorithm was from public content in GitHub: https: //github.com/thuml/Time-Series-Library.

• Author contribution

X. W performed data curation, developed the methodology, conducted the formal analyses, and wrote the original draft of the manuscript. T. L, H. Z and S. Z conceptualized the study, supervised the research, and contributed to the review and editing of the manuscript. All authors discussed the results and reviewed and approved the final manuscript.

## References

[1] Gielen, D. et al. World energy transitions outlook: 1.5° c pathway (2021).

[2] Alliance, G. R., Presidency, C. et al. Tripling renewable power and doubling energy eficiency by 2030: Crucial steps towards 1.5° c (2023).

[3] Xu, S. et al. Substantially lower estimates in china’s ofshore wind potential using farm-scale spatial modeling and wake efects. Nature Communications (2026).

[4] Wang, K. et al. Spatial distribution and long-term trend of wind energy in the northwest pacific ocean. Water-Energy Nexus 7, 135–142 (2024).

[5] Ou, L. et al. Ofshore wind zoning in china: Method and experience. Ocean & Coastal Management 151, 99–108 (2018).

[6] Zhao, H., Duan, X., Raga, G. & Klotzbach, P. J. Changes in characteristics of rapidly intensifying western north pacific tropical cyclones related to climate regime shifts. Journal of Climate 31, 8163–8179 (2018).

[7] Li, Y. et al. Recent increases in tropical cyclone rapid intensification events in global ofshore regions. Nature Communications 14, 5167 (2023).

[8] Zhao, Y., Tao, Y., Chen, Y., Yan, J. & Zeng, Z. Increasing extreme winds challenge ofshore wind energy resilience. Nature Communications 16, 9529 (2025).

[9] Worsnop, R. P., Lundquist, J. K., Bryan, G. H., Damiani, R. & Musial, W. Gusts and shear within hurricane eyewalls can exceed ofshore wind turbine design standards. Geophysical Research Letters 44, 6413–6420 (2017).

[10] Onol, A. O. & Yesilyurt, S. Efects of wind gusts on a vertical axis wind turbine with high solidity. Journal of Wind Engineering and Industrial Aerodynamics 162, 1–11 (2017).

[11] Wu, Z., Bangga, G. & Cao, Y. Efects of lateral wind gusts on vertical axis wind turbines. Energy 167, 1212–1223 (2019).

[12] Hou, G., Xu, K. & Lian, J. A review on recent risk assessment methodologies of ofshore wind turbine foundations. Ocean Engineering 264, 112469 (2022).

[13] Ning, J., Shi, R., Xuan, S., Jiang, C. & Jia, L. Assessment of ofshore island wind energy potential in typhoon-prone regions with a kde-based probabilistic modeling approach. Energy 139092 (2025).

[14] Beljaars, A. The influence of sampling and filtering on measured wind gusts. Journal of Atmospheric and Oceanic Technology 4, 613–626 (1987).

[15] Wang, H., Zhang, Y.-M., Mao, J.-X. & Wan, H.-P. A probabilistic approach for short-term prediction of wind gust speed using ensemble learning. Journal of Wind Engineering and Industrial Aerodynamics 202, 104198 (2020).

[16] Zhang, Z. et al. Optimization control model for wind farms considering the spatiotemporal dynamics of gusts. Energy 139730 (2025).

[17] Wang, Y., Zou, R., Liu, F., Zhang, L. & Liu, Q. A review of wind speed and wind power forecasting with deep neural networks. Applied energy 304, 117766 (2021).

[18] Bauer, P., Thorpe, A. & Brunet, G. The quiet revolution of numerical weather prediction. Nature 525, 47–55 (2015).

[19] Graves, A. Long short-term memory. Supervised sequence labelling with recurrent neural networks 37–45 (2012).

[20] Cho, K. et al. Learning phrase representations using rnn encoder–decoder for statistical machine translation. Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 1724–1734 (2014).

[21] Zhou, H. et al. Informer: Beyond eficient transformer for long sequence timeseries forecasting. Proceedings of the AAAI conference on artificial intelligence, Vol. 35, 11106–11115 (2021).

[22] Kim, D.-Y. & Suk, H.-I. Sigmaformer: a spatiotemporal gaussian mixture correlation transformer for global weather forecasting. npj Climate and Atmospheric Science 9, 113 (2026).

[23] Gallego-Castillo, C., Cuerva-Tejero, A. & Lopez-Garcia, O. A review on the recent history of wind power ramp forecasting. Renewable and Sustainable Energy Reviews 52, 1148–1157 (2015).

[24] Liu, H., Mi, X.-w. & Li, Y.-f. Wind speed forecasting method based on deep learning strategy using empirical wavelet transform, long short term memory neural network and elman neural network. Energy conversion and management 156, 498–514 (2018).

[25] Jiajun, H., Chuanjin, Y., Yongle, L. & Huoyue, X. Ultra-short term wind prediction with wavelet transform, deep belief network and ensemble learning. Energy Conversion and Management 205, 112418 (2020).

[26] Stephane, M. A wavelet tour of signal processing (1999).

[27] Perez, E., Strub, F., De Vries, H., Dumoulin, V. & Courville, A. Film: Visual reasoning with a general conditioning layer. Proceedings of the AAAI conference on artificial intelligence, Vol. 32 (2018).

[28] Zhang, H., Zhao, L. & Du, Z. Wind power prediction based on cnn-lstm. 2021 IEEE 5th Conference on Energy Internet and Energy System Integration (EI2), 3097–3102 (IEEE, 2021).

[29] Wu, H., Xu, J., Wang, J. & Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems 34, 22419–22430 (2021).

[30] Zeng, A., Chen, M., Zhang, L. & Xu, Q. Are transformers efective for time series forecasting? Proceedings of the AAAI conference on artificial intelligence,

Vol. 37, 11121–11128 (2023).

[31] Wu, H. et al. Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186 (2022).

[32] Liu, Y. et al. itransformer: Inverted transformers are efective for time series forecasting. International conference on learning representations, Vol. 2024, 11116–11140 (2024).

[33] Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730 (2022).

[34] Chen, L. et al. A machine learning model that outperforms conventional global subseasonal forecast models. Nature Communications 15, 6425 (2024).

[35] Ling, F. et al. Multi-task machine learning improves multi-seasonal prediction of the indian ocean dipole. Nature Communications 13, 7681 (2022).

[36] Owens, R. & Hewson, T. Ecmwf forecast user guide. reading, ecmwf (2018).

[37] Dialogue Earth. Victims of super typhoon Yagi include Hainan wind farm (2024). URL https://dialogue.earth/en/digest/ victims-of-super-typhoon-yagi-include-hainan-wind-farm/.

[38] Brenowitz, N. D. & Bretherton, C. S. Prognostic validation of a neural network unified physics parameterization. Geophysical Research Letters 45, 6289–6298 (2018).

[39] Hersbach, H. et al. Era5 hourly data on single levels from 1940 to present. https://doi.org/10.24381/cds.adbb2d47 (2023). Accessed on 26-Apr-2026.

[40] He, Y., Li, Q., Chan, P., Wu, J. & Fu, J. Toward modeling the spatial pressure field of tropical cyclones: Insights from typhoon hato (1713). Journal of Wind Engineering and Industrial Aerodynamics 184, 378–390 (2019).

[41] Snaiki, R. & Wu, T. Modeling tropical cyclone boundary layer: Heightresolving pressure and wind fields. Journal of wind Engineering and Industrial aerodynamics 170, 18–27 (2017).

[42] Nason, G. P. & Silverman, B. W. in The stationary wavelet transform and some statistical applications Wavelets and statistics 281–299 (Springer, 1995).

[43] Chen, H., Luong, V., Mukherjee, L. & Singh, V. Simpletm: A simple baseline for multivariate time series forecasting. The Thirteenth International Conference on Learning Representations (2025).

[44] Coifman, R. R. & Donoho, D. L. in Translation-invariant de-noising Wavelets and statistics 125–150 (Springer, 1995).