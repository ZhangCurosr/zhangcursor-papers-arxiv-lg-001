# Low-Cost Sensor Calibration for Indoor Air Quality Monitoring: A Dataset, E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> S<sub>cenar</sub>i<sub>os, an</sub>d <sub>a</sub> Li<sub>g</sub>ht<sub>we</sub>i<sub>g</sub>ht M<sub>o</sub>d<sub>e</sub>l

<sub>J</sub>i<sub>nyong</sub> Y<sub>un</sub>1<sub>,</sub> S<sub>eo</sub>kh<sub>o</sub> Ah<sub>n</sub>1<sub>,</sub> H<sub>yungj</sub>i<sub>n</sub> Ki<sub>m</sub>1<sub>,</sub> S<sub>ung</sub>b<sub>o</sub>k Shi<sub>n</sub>2<sub>,</sub> Y<sub>oung-</sub>D<sub>u</sub>k S<sub>eo</sub>1<sup>∗</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, Inha University, Incheon 22212, South Korea <sup>2</sup> Department of Computer Science and Engineering, Sogang University, Seoul 04107, South Korea 12191632Y@inha.edu, sokho0514@inha.edu, flslzk@inha.edu, sbshin90@sogang.ac.kr, mysid88@inha.ac.kr

## Abstract

Low-cost sensors enable scalable indoor air quality monitoring but require calibration because of nonlinear distortions, noise, and temporal drift. The conventional strict pairwise calibration setting requires a co-located reference sensor at each deployment location and does not account for spatial and temporal heterogeneity. To address these limitations, we introduce a six-month dataset comprising multivariate indoor air-quality measurements from low-cost and reference sensors with contextual metadata collected at five locations. Using this dataset, we define four evaluation scenarios. The referenceeficient and location-transfer scenarios evaluate spatial generalization, whereas the long-term drift and event-conditioned scenarios assess robustness to gradual and abrupt distribution shifts. Based on these scenarios, we derive design requirements and propose a lightweight temporal model that combines input-window compression with residual temporal and feature fusion. Experiments show strong calibration performance across all four scenarios with low edge-inference cost.

## 1 Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

People spend about 90% of their time indoors, making indoor environments a major source of daily pollutant exposure and an important factor in human health (Jiang et al. 2011; Zhong, Lalanne, and Alavi 2021). Accurate measurement of indoor air quality is therefore essential for protecting people. However, indoor environments change continuously due to short-term events such as ventilation, changes in occupancy, and human activities (Wu et al. 2021). They also exhibit spatial heterogeneity, meaning that air quality can difer across locations even within the same indoor space (Li et al. 2018). Because of this temporal and spatial heterogeneity, a small number of sensors may fail to capture both local changes and the overall air quality distribution. Therefore, precise monitoring of indoor environments requires the dense deployment of high-accuracy sensors throughout the monitored space.

Deploying a dense network of high-accuracy reference sensors (i.e., high-cost sensors), however, is costly. Low-cost sensors can be deployed at lower cost, but their measurements are often afected by noise and drift, making them less accurate (Koziel et al. 2025). Hence, instead of installing reference sensors at every measurement point, calibrating low-cost sensors with a lightweight deep learning model to achieve near-reference accuracy can provide an alternative that balances cost and performance (Concas et al. 2021; Ahn et al. 2024; Dalbah, Worring, and Hsu 2026).

Low-cost sensor calibration models are trained using data collected from reference sensors. Under this conventional setting, termed strict pairwise calibration, at least one reference sensor is required at every location where a low-cost sensor is deployed (Yu et al. 2020; Ahn et al. 2025). However, deploying reference sensors at all locations makes efective indoor air quality monitoring prohibitively expensive.

Accordingly, practical indoor air quality monitoring requires a calibration method that can calibrate low-cost sensors across multiple locations using only a few reference sensors. Such a method must be robust to both rapid environmental changes and spatial heterogeneity. Addressing these practical requirements involves three key challenges: constructing a dataset representative of indoor environments, defining realistic evaluation scenarios, and designing an eficient and robust calibration model. Specifically, we focus on the following:

• Lon<sub>g</sub>-term<sub>,</sub> multi-location data: A sensor calibration dataset must capture spatial variations and temporal changes under limited reference-sensor availability.

• Realistic evaluation scenarios: Calibration must operate robustly under real-world deployment conditions, including at locations without reference sensors and under shortand long-term distribution shifts.

• Eficient and robust calibration: A lightweight calibration model must efectively generalize across locations and distribution shifts using limited reference data.

To address these challenges, we propose a benchmark for indoor sensor calibration. First, we construct a six-month, multi-location indoor air quality dataset comprising synchronized measurements from low-cost and reference sensors deployed at five locations in a university laboratory (Section 4). The dataset contains measurements of five environmental variables $( i . e . , \mathrm { C O _ { 2 } } .$ , CO, temperature, humidity, and PM<sub>1</sub>), along with metadata on door and window states, enabling the analysis of both gradual distribution shifts and rapid environmental changes. Based on this dataset, we define four evaluation scenarios that reflect limited reference-data availability, unseen-location generalization, and short- and long-term distribution shifts (Section 3). Guided by these scenarios, we derive five design requirements and propose a calibration model designed to satisfy these requirements (Section 5). Our main contributions are summarized as follows:

Table 1: Comparison of low-cost sensor calibration studies and datasets. T. and RH. denote temperature and relative humidity.
<table><tr><td rowspan="2">Dataset (Calibration Study)</td><td rowspan="2">Modalities</td><td colspan="3">Dataset Properties</td><td colspan="4">Deployment Scenarios</td></tr><tr><td>Indoor.</td><td>MultiLoc.</td><td>Event.</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td></tr><tr><td>UCI Air Quality (Aula et al. 2022)</td><td> $\mathrm { C O , N O _ { 2 } , T . , R H . }$ </td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Author-collected (Allka et al. 2023)</td><td> $\mathrm { { O _ { 3 } } , T . }$ </td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Author-collected (Ahn et al. 2024)</td><td> $\mathrm { P M } _ { 1 0 }$ </td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Author-collected (Bachechi, Rollo, and Po 2024)</td><td> $\mathrm { N O } , \mathrm { N O } _ { 2 } , \mathrm { O } _ { 3 }$ </td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>SensEURCity (Ahn et al. 2025)</td><td> $\mathrm { P M _ { 1 } , P M _ { 2 . 5 } , P M _ { 1 0 } }$ </td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>AQ-SDR (Dalbah, Worring, and Hsu 2026)</td><td> $\mathrm { P M } _ { 2 . 5 }$ </td><td>X</td><td>√</td><td>X</td><td>√</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Indoor Sensor Calibration (OuRs)</td><td> $\mathbf { C O } _ { 2 } , \mathbf { C O } , \mathbf { T } , \mathbf { R H } . , \mathbf { P M } _ { 1 }$ </td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Dataset properties. Indoor. (indoor setting); MultiLoc.(multi-location data); Event. (event metadata). Symbols. Supported (✓); not supported (×).

• Dataset: We construct a six-month, five-location indoor sensor calibration dataset with synchronized low-cost and reference sensor measurements and event metadata.

• Scenarios: We design four practical scenarios for efective indoor sensor calibration, in which many low-cost sensors across multiple locations are calibrated using limited reference data under spatial and temporal shifts.

• Model: We propose a lightweight temporal calibration model that robustly addresses scenario-driven design requirements with low edge-inference cost.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

We review prior work on low-cost sensor calibration and the datasets used in previous calibration studies.

## 2<sub>.</sub>1 S<sub>e</sub>n<sub>so</sub>r C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n D<sub>a</sub>t<sub>ase</sub>t<sub>s</sub>

Table 1 compares representative calibration datasets. UCI Air Quality (Vito 2008; Aula et al. 2022) contains long-term measurements from a single outdoor site. The datasets used by Allka et al. (2023) and HypeAIR (Bachechi, Rollo, and Po 2024) extend this setting to multiple outdoor locations or longer deployment periods, while SensEURCity (Van Poppel et al. 2023) and AQ-SDR (Dalbah, Worring, and Hsu 2026) provide large-scale outdoor measurements across multiple regions. Among indoor datasets, SenDaL (Ahn et al. 2024) includes measurements from three environments but is limited to deployment periods of 10 days. None of these datasets combines synchronized measurements across multiple indoor locations with long-term coverage and event metadata. Our contribution. Our six-month, five-location indoor dataset combines paired low-cost and reference sensor measurements with door, window, and occupancy metadata, supporting all four indoor sensor calibration scenarios.

## 2<sub>.</sub>2 S<sub>e</sub>n<sub>so</sub>r C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n M<sub>o</sub>d<sub>e</sub>l<sub>s</sub>

Time-series models can be adapted to sensor calibration by learning temporal patterns in low-cost measurements. Representative backbones include linear, Multilayer perceptron (MLP)-based models, and attention-based architectures (Zeng et al. 2023; Chen et al. 2026; Liu et al. 2024).

Existing calibration methods address diferent deployment challenges. Allka et al. (2023) and HypeAIR (Bachechi, Rollo, and Po 2024) focus on temporal patterns and longterm outdoor calibration. SenDaL (Ahn et al. 2024) and

TESLA (Ahn et al. 2025) target eficient edge deployment, while Veli (Dalbah, Worring, and Hsu 2026) performs unsupervised calibration without paired reference measurements. However, existing methods address these challenges separately rather than jointly considering limited reference supervision, unseen-location generalization, short- and long-term shifts, and eficiency.

Our contribution. Our model combines temporal compression and residual fusion for robust calibration with limited references, unseen locations, and distribution shifts, while retaining low edge inference cost.

## 3 Ind<sub>oo</sub>r S<sub>e</sub>n<sub>so</sub>r C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n S<sub>ce</sub>n<sub>a</sub>ri<sub>os</sub>

Figure 1 presents four evaluation scenarios that relax strict pairwise calibration, where each low-cost sensor is trained with synchronized, co-located reference sensor measurements. This conventional setting assumes:

• One reference <sub>p</sub>er location: Conventional sensor calibration requires a co-located reference sensor at each deployment location for model training. In indoor environments, 1) reference sensors are often placed only in selected zones, and 2) low-cost sensors may later be deployed at new zones not represented during training.

• Tem<sub>p</sub>orall<sub>y</sub> stable calibration relationshi<sub>p</sub>s: The relationship between low-cost and reference measurements is assumed to remain stable over time, overlooking 1) gradual sensor drift and 2) abrupt event-driven changes.

The four scenarios (S1-S4) directly implement the two cases under each assumption. First, relaxing reference coverage (S1 and S2) is evaluated in the semi-pairwise stage in Figure 1, distinguished by whether the target location is observed during training:

S1 Reference-eficient. Reference measurements are available only at reference-equipped (anchor) locations. Lowcost measurements from all locations are observed during training. The model has the limited reference information across the observed network rather than generalizing to an unseen location.

S2 Location-transfer. The target location is excluded from training. The model learns from paired measurements at anchor locations and calibrates the unseen target using its low-cost measurements. Target references are used only for evaluation. Note that this scenario requires stronger spatial generalization than S1.

Next, relaxing temporal stability (S3 and S4) is evaluated in the heterogeneous-pairwise stage, where each lowcost-reference sensor pair has a distinct calibration relationship. Its two scenarios distinguish gradual internal drift from abrupt, externally induced shifts.

![](images/506fd56347b22ca432656998fad11166aa99e6e638cff6e61eea0939b49e201b.jpg)  
Figure 1: Comparison of standard sensor calibration and four deployment-oriented scenarios in terms of relaxed assumptions, calibration regimes, and world settings: Scenarios 1-4: reference-eficient, location-transfer, long-term drift, and event-condition.

S3 Long-term Drift. The model is trained on an initial period and tested later, after sensor-specific responses have gradually changed. Because the shift occurs within the low-cost sensor network without abrupt external events, S3 is a closed-world setting.

S4 Event-condition. External events, such as changes in door or window states, cause abrupt distribution shifts with location-dependent magnitudes and delays. The model must handle both temporal changes and locationspecific responses. Because these shifts are externally induced, S4 is an open-world setting.

## 4 Ind<sub>oo</sub>r S<sub>e</sub>n<sub>so</sub>r C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n D<sub>a</sub>t<sub>ase</sub>t

To support the scenarios introduced in Section 3, we construct a six-month calibration dataset with synchronized measurements from multiple locations, paired low-cost and reference sensor measurements, and event metadata.

## 4<sub>.</sub>1 D<sub>a</sub>t<sub>a</sub> C<sub>o</sub>ll<sub>ec</sub>ti<sub>on</sub>

The data is collected in a university laboratory from January 6 to July 6, 2026. We deploy sensor nodes at five locations within the same indoor space: East, West, South, North, and Center. All sensor nodes are installed at the same height of 0.75 m above the floor to maintain consistent measurement conditions across locations. The South node is located near a window, whereas the North node is located near an entrance, where they are more directly afected by ventilation and outdoor-air inflow. Together, these placements capture spatial variation within the room.

Each node records five environmental variables at 1- second intervals: CO , CO, PM , temperature (T.), and relative humidity (RH.). For each variable, the low-cost and reference sensor are paired as listed in Table 2. We categorize the channels as low-cost and reference (high-cost) channels based on their market prices at the time of data collection.

After preprocessing, the dataset contains 261,597 1- minute location-timestamp observations. Because the collection periods difer across locations, individual tasks contain between 223,460 and 260,483 paired observations, yielding 6,102,195 paired observations across all 25 tasks.

Table 2: Mean channel measurements across five locations after sanity filtering and 1-minute resampling.
<table><tr><td>Measured</td><td>Channel</td><td>East</td><td>West</td><td>South</td><td>North</td><td>Center</td></tr><tr><td>PM1 (µg/m³)</td><td>L PPD42</td><td>22.2</td><td>21.9</td><td>23.6</td><td>25.7</td><td>23.4</td></tr><tr><td></td><td>R PMS7003</td><td>20.2</td><td>18.9</td><td>21.4</td><td>21.8</td><td>19.6</td></tr><tr><td>CO (ppm)</td><td>L MQ135 RMQ9</td><td>6.3 4.5</td><td>7.1 4.2</td><td>6.2 4.4</td><td>8.5 4.8</td><td>6.8 4.1</td></tr><tr><td>CO2</td><td>L MQ135</td><td>552.8</td><td></td><td></td><td></td><td></td></tr><tr><td>(ppm)</td><td>R MG811</td><td>532.4</td><td>540.0 512.0</td><td>548.0 538.0</td><td>584.0 548.0</td><td>556.2 522.6</td></tr><tr><td>T.</td><td>L DHT11</td><td>24.6</td><td></td><td></td><td></td><td></td></tr><tr><td>(°C)</td><td>R DHT22</td><td>24.1</td><td>25.8 24.2</td><td>26.0</td><td>25.7</td><td>25.5</td></tr><tr><td></td><td></td><td></td><td></td><td>24.4</td><td>24.1</td><td>23.2</td></tr><tr><td>RH.</td><td>L DHT11</td><td>32.7</td><td>34.1</td><td>35.5</td><td>33.8</td><td>33.1</td></tr><tr><td>(%)</td><td>R DHT22</td><td>33.8</td><td>34.0</td><td>34.5</td><td>33.1</td><td>32.9</td></tr></table>

L: low-cost sensor; R: reference sensor; T.: Temperature; RH.: Relative Humidity

## 4<sub>.</sub>2 M<sub>e</sub>t<sub>a</sub>d<sub>a</sub>t<sub>a</sub>

The dataset provides UTC and local timestamps, entrancedoor state, the number of open windows (maximum 4), occupant count (maximum 11) during the preceding 30 minutes, and business-hour status. Door state and the number of open windows are recorded at 30-minute intervals. The metadata covers 223,460 1-minute timestamps. At this 30-minute annotation resolution, we observe 4,118 door-state changes and 6,621 changes in the number of open windows. These values count state changes between consecutive valid annotations. The benchmark uses the door and window variables to identify event-conditioned evaluation periods, while the remaining metadata are provided for additional analysis.

## 4<sub>.</sub>3 Pr<sub>ep</sub>r<sub>ocess</sub>in<sub>g</sub>

We process the data in the following manner. To begin with, for each location, timestamps were converted to UTC and sorted chronologically. Measurements sharing the same timestamp were aggregated by averaging each numerical channel. Nonnumeric and infinite values were treated as missing. Negative measurements were also treated as missing, except for temperature, for which negative values may be physically valid. Extreme values in the low-cost channels were removed using a relaxed filter based on the 1st and 99th percentiles. The 1-second measurements were then resampled into 1-minute averages.

![](images/90b86526e258418e57706023c74b6941549ab96a874ce09886e9f46f324956d5.jpg)  
Figure 2: Architecture of proposed model with initial temporal compression, temporal and feature fusion, and output head.

Before imputation, valid-value ratios ranged from 82.9% to 89.2% across locations and channels. As summarized in Table 2, channel means difered across locations, including the window-side and entrance-side nodes. These diferences indicate that the calibration relationship depends on sensor placement and environmental conditions and cannot be represented by a single location-invariant ofset. Detailed missingness statistics and imputation procedures are provided in the Supplementary Material.

## 4<sub>.</sub>4 S<sub>uppo</sub>rt f<sub>o</sub>r D<sub>ep</sub>l<sub>oy</sub>m<sub>e</sub>nt S<sub>ce</sub>n<sub>a</sub>ri<sub>os</sub>

The dataset was designed to support S1-S4 along two deployment dimensions: spatial reference coverage and temporal stability. First, because synchronized low-cost and reference measurements are available at all five locations, we can simulate sparse-reference deployments by using reference measurements from only selected locations while retaining the same environmental data. Because the measurements share aligned timestamps and a common acquisition protocol, the dataset supports controlled comparisons between S1, which shares limited reference supervision across observed locations, and S2, which transfers calibration relationships to a held-out location. In addition, the location-specific channel statistics in Table 2, particularly at the South node near the window and the North node near the entrance, show that these scenarios capture genuine spatial heterogeneity rather than arbitrary partitions of otherwise homogeneous data.

Second, the continuous six-month time series preserves the temporal history of each sensor-location pair. Because each pair was continuously monitored for six months, we can track how the calibration performance of the same deployed sensor changes over time. This enables S3 to evaluate gradual sensor drift. For S4, the distinct placements of the South and North nodes provide clear exposure to window ventilation and entrance-door air inflow, respectively. Together with repeated records of door- and window-state changes, these placements provide multiple event episodes for evaluating abrupt, delayed, and location-dependent sensor responses.

## 5 M<sub>o</sub>d<sub>e</sub>l f<sub>o</sub>r Ind<sub>oo</sub>r S<sub>e</sub>n<sub>so</sub>r C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n

Based on our benchmark, this section defines key model design requirements for robust calibration across all scenarios (Section 5.1) and presents a model designed to satisfy them (Section 5.2).

## 5<sub>.</sub>1 D<sub>es</sub>i<sub>g</sub>n R<sub>equ</sub>ir<sub>e</sub>m<sub>e</sub>nt<sub>s</sub>

We define the following design requirements for robust indoor sensor calibration across the four scenarios.

R1 Reference-eficient learning (S1). The model should share limited reference supervision across multiple lowcost sensor locations.

R2 Location-agnostic generalization (S2). The model should learn transferable calibration patterns applicable to locations unseen during training.

R3 Drift-robust representation (S3). The model should remain robust to gradual distribution shifts in sensor baselines and responses over time.

R4 Event-robust modeling (S4). The model should robustly handle abrupt, delayed, and transient sensor responses to environmental events.

R5 Lightweight edge inference. The model should enable low inference speed and memory-eficient calibration on resource-constrained IoT devices.

## 5<sub>.</sub>2 M<sub>o</sub>d<sub>e</sub>l D<sub>es</sub>i<sub>g</sub>n

To satisfy R1-R5, we adopt an attention-free MLP architecture. Temporal compression and residual temporal-feature fusion support transferable and shift-robust calibration (R1- R4), while operating in a compact temporal space keeps inference and memory costs low for edge deployment (R5).

Initial Temporal Compression. Let $x _ { t }$ and $y _ { t }$ denote the measurements of a low-cost sensor and its reference sensor at time t, respectively. Given an input window of length L, the calibration input is defined as:

$$
\mathbf { X } _ { t } = \left[ \mathbf { s } _ { t - L + 1 } , \ldots , \mathbf { s } _ { t } \right] ^ { \top } \in \mathbb { R } ^ { L \times F } ,\tag{1}
$$

where $\mathbf { s } _ { t } ~ \in ~ \mathbb { R } ^ { F }$ has a low-cost sensor measurement and optional temporal features. The model estimates the reference $y _ { t }$ corresponding to the final time step of the input window. For notational simplicity, the time index is often omitted, $i . e . ,$ $\mathbf { X } = \mathbf { X } _ { t }$ . The model first projects the L input time steps into R compressed temporal representations:

$$
\mathbf { Z } = \left[ \operatorname { R e L U } \left( \mathbf { X } ^ { \top } \mathbf { W } _ { p } + \mathbf { b } _ { p } \right) \right] ^ { \top } \in \mathbb { R } ^ { R \times F }\tag{2}
$$

Here, $\mathbf { W } _ { p } \in \mathbb { R } ^ { L \times R }$ with $R \ll L$ . This projection reduces the cost of subsequent operations while retaining the dominant temporal patterns within the input window. Operating in this compact temporal space reduces sensitivity to short-lived, location-specific fluctuations and promotes more transferable temporal patterns. This design is intended to improve spatial generalization when the reference location changes (R1) and when the model is transferred to unseen locations (R2). This compression also reduces the cost of subsequent transformations, supporting lightweight edge inference (R5).

Table 3: Performance comparison on the representative outdoor calibration dataset SensEURCity and our indoor dataset. The best result for each metric is marked in bold, and the second-best result is underlined. Lower values indicate better performance.
<table><tr><td rowspan="3">Model</td><td colspan="3">Outdoor data</td><td colspan="7">Indoor data (Ours)</td></tr><tr><td colspan="3">RMSE↓</td><td colspan="5">RMSE↓</td><td rowspan="2">Inference↓ (ms)</td><td rowspan="2">Memory ↓ (KB)</td></tr><tr><td> $\mathrm { P M _ { 1 0 } }$   $\left( \mu \mathrm { g } / \mathrm { m } ^ { 3 } \right)$ </td><td> $\mathrm { P M } _ { 2 . 5 }$   $\left( \mu \mathrm { g } / \mathrm { m } ^ { 3 } \right)$ </td><td> $\mathrm { P M } _ { 1 }$   $\left( \mu \mathrm { g } / \mathrm { m } ^ { 3 } \right)$ </td><td> $\mathrm { C O _ { 2 } }$  (ppm)</td><td>CO (ppm)</td><td> $\mathrm { P M } _ { 1 }$   $\left( \mu \mathrm { g } / \mathrm { m } ^ { 3 } \right)$ </td><td>T. (°C)</td><td>RH. (%)</td></tr><tr><td>iTransformer</td><td>15.92</td><td>7.36</td><td>4.07</td><td>9.80</td><td>4.92</td><td>8.54</td><td>2.43</td><td>5.24</td><td>2.32</td><td>41.20</td></tr><tr><td>TimeXer</td><td>15.12</td><td>6.31</td><td>3.31</td><td>9.05</td><td>7.73</td><td>9.77</td><td>3.75</td><td>5.63</td><td>2.99</td><td>186.60</td></tr><tr><td>DLinear</td><td>17.81</td><td>8.41</td><td>4.49</td><td>13.70</td><td>5.93</td><td>9.21</td><td>3.14</td><td>5.44</td><td>1.29</td><td>28.00</td></tr><tr><td>XLinear</td><td>19.05</td><td>11.64</td><td>8.48</td><td>14.71</td><td>5.92</td><td>8.71</td><td>3.23</td><td>6.20</td><td>2.12</td><td>152.00</td></tr><tr><td>TimeMixer</td><td>16.75</td><td>8.72</td><td>4.69</td><td>19.67</td><td>6.84</td><td>10.07</td><td>3.27</td><td>8.46</td><td>5.84</td><td>384.00</td></tr><tr><td>SenDaL</td><td>15.76</td><td>6.87</td><td>3.14</td><td>11.25</td><td>6.25</td><td>9.28</td><td>2.81</td><td>6.72</td><td>30.14</td><td>9614.00</td></tr><tr><td>TESLA</td><td>14.97</td><td>6.55</td><td>3.06</td><td>9.06</td><td>5.36</td><td>8.50</td><td>1.96</td><td>5.07</td><td>2.06</td><td>121.00</td></tr><tr><td>OURS</td><td>13.34</td><td>5.19</td><td>2.50</td><td>8.65</td><td>4.80</td><td>8.34</td><td>2.11</td><td>4.89</td><td>1.27</td><td>42.60</td></tr></table>

Temporal and Feature Fusion. The representation Z is first transformed along the temporal dimension:

$$
\begin{array} { r } { \tilde { \mathbf { H } } = \mathrm { B N } \Big ( \mathbf { Z } + \big [ \mathrm { R e L U } \big ( \mathbf { Z } ^ { \top } \mathbf { W } + \mathbf { b } \big ) \big ] ^ { \top } \Big ) } \end{array}\tag{3}
$$

where $\mathbf { W } \in \mathbb { R } ^ { R \times R } .$ . The fusion captures relationships among latent time steps, including gradual trends, delayed sensor responses, and rapid changes. The model then applies a nonlinear transformation along the feature dimension D:

$$
\mathbf { H } { = } \mathrm { B N } \Big ( \tilde { \mathbf { H } } { + } \mathrm { R e L U } \Big ( \tilde { \mathbf { H } } \mathbf { W } _ { 1 } { + } \mathbf { b } _ { 1 } \Big ) \mathbf { W } _ { 2 } { + } \mathbf { b } _ { 2 } \Big ) \in \mathbb { R } ^ { R \times F }\tag{4}
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { F \times D }$ and $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { D \times F }$ . With multiple input features, this transformation learns nonlinear dependencies among sensor and temporal variables. With a single sensor channel, it acts as a lightweight nonlinear calibration.

Residual connections preserve the underlying sensor trajectory while allowing the nonlinear transformations to learn the required correction. Batch normalization stabilizes the scale of intermediate representations. These components are intended to maintain stable representations under the longterm baseline and response changes considered (R3).

The temporal fusion also processes the recent trajectory rather than relying only on the final measurement. When a door or window transition and its resulting sensor response are contained within the input window, the model can represent the direction and duration of the change. This property supports calibration under the abrupt events considered (R4).

Calibration Output Head. The final representation H is aggregated over the latent temporal and feature dimensions:

$$
\hat { y } = \mathbf { w } _ { f } ^ { \top } \left( \mathbf { H } ^ { \top } \mathbf { w } _ { o } + b _ { o } \mathbf { 1 } _ { F } \right) + b _ { f }\tag{5}
$$

Here, ${ \bf w } _ { o } \in { \mathbb { R } } ^ { R }$ aggregates the latent temporal representation of each feature, while $\mathbf { \bar { w } } _ { f } \in \mathbb { R } ^ { F }$ combines the feature-wise values. The scalar output yˆ is the calibrated estimate corresponding to the final time step of the input window.

The model does not use location-specific information including particular door or window states. It therefore learns sensor-response patterns shared across locations instead of explicitly memorizing individual environments. Initial compression supports R1 and R2, while the residual representation supports long-term drift (R3) and abrupt events (R4).

Moreover, initial compression before the fusion blocks and using small hidden dimensions keep the parameter count and inference cost low (R5). This makes it suitable for real-time calibration on resource-constrained edge devices.

## 6 E<sub>xpe</sub>rim<sub>e</sub>nt<sub>a</sub>l S<sub>e</sub>ttin<sub>gs</sub>

This section describes the evaluation protocols, baselines, metrics, and implementation details.

## 6.1 Evaluation Objectives

Our experiments evaluate whether the proposed dataset and four scenarios expose calibration challenges that are obscured by standard strict pairwise evaluation. We further examine whether our model consistently achieves strong calibration performance under limited reference setting, unseenlocation transfer, long-term drift, and event-conditioned shifts, while retaining low computational cost. Table 3 reports standard indoor and outdoor calibration performance and computational eficiency. Tables 4-7 report the results for S1-S4, and Figure 3 evaluates latency and model size on the target edge device.

## 6<sub>.</sub>2 D<sub>a</sub>t<sub>ase</sub>t U<sub>sage an</sub>d E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> P<sub>ro</sub>t<sub>oco</sub>l

Indoor experiments in Tables 4-7 use the dataset from Section 4, while the outdoor experiments in Table 3 follow settings (Ahn et al. 2025). For standard calibration results in Table 3, calibration is performed separately at each location using co-located low-cost and reference-sensor measurements, and the reported root mean squared error (RMSE) is averaged across all locations.

For S1 and S2, samples at each location are chronologically split into 70% training, 15% validation, and 15% test sets. In S1, each location serves as the comparison-sensorequipped anchor. Training uses low-cost measurements from all locations but reference-channel supervision only from the anchor. For each anchor, RMSE is macro-averaged across the four non-anchor locations and then averaged across all five anchors. In S2, a model is trained on each location and evaluated without adaptation on the other four. RMSE is macroaveraged across the target locations and then across the five source locations.

Table 4: S1 reports the average RMSE on non-anchor locations across the five anchor-location settings.
<table><tr><td rowspan="2">Model</td><td colspan="5">RMSE↓</td></tr><tr><td>CO</td><td> $\mathrm { C O _ { 2 } }$ </td><td>T.</td><td>RH.</td><td> $\mathrm { P M } _ { 1 }$ </td></tr><tr><td>iTransformer</td><td>6.26</td><td>9.45</td><td>4.11</td><td>7.25</td><td>9.34</td></tr><tr><td>TimeXer</td><td>7.92</td><td>10.29</td><td>9.18</td><td>9.31</td><td>12.95</td></tr><tr><td>DLinear</td><td>7.71</td><td>10.13</td><td>5.02</td><td>11.91</td><td>14.45</td></tr><tr><td>XLinear</td><td>6.93</td><td>10.77</td><td>6.59</td><td>10.80</td><td>11.69</td></tr><tr><td>TimeMixer</td><td>9.51</td><td>10.65</td><td>6.91</td><td>10.71</td><td>12.54</td></tr><tr><td>SenDaL</td><td>9.35</td><td>10.89</td><td>8.28</td><td>9.21</td><td>9.67</td></tr><tr><td>TESLA</td><td>6.19</td><td>10.13</td><td>4.36</td><td>7.96</td><td>9.17</td></tr><tr><td>Ours</td><td>4.69</td><td>8.45</td><td>4.30</td><td>6.78</td><td>8.74</td></tr></table>

Table 5: S2 evaluates transfer from each source location to four unseen target locations without adaptation, averaged across the five source-location settings.
<table><tr><td rowspan="2">Model</td><td colspan="4">RMSE↓</td></tr><tr><td>CO</td><td> $\mathrm { { C O } _ { 2 } }$  T.</td><td>RH.</td><td> $\mathrm { P M } _ { 1 }$ </td></tr><tr><td>iTransformer</td><td>7.63</td><td>10.45 4.79</td><td>9.02</td><td>10.27</td></tr><tr><td>TimeXer</td><td>9.35 11.18</td><td>6.66</td><td>10.34</td><td>11.88</td></tr><tr><td>DLinear</td><td>8.69 11.60</td><td>11.43</td><td>12.27</td><td>12.66</td></tr><tr><td>XLinear</td><td>8.44 11.81</td><td>7.62</td><td>10.06</td><td>12.56</td></tr><tr><td>TimeMixer</td><td>9.70</td><td>11.75 8.54</td><td>13.05</td><td>13.65</td></tr><tr><td>SenDaL</td><td>9.48</td><td>10.92 5.94</td><td>10.34</td><td>12.04</td></tr><tr><td>TESLA</td><td>7.44</td><td>10.43</td><td>5.15 8.58</td><td>9.17</td></tr><tr><td>OuRs</td><td>7.56</td><td>9.68</td><td>4.51 8.21</td><td>8.85</td></tr></table>

In S3, each location-sensor time series is chronologically split into 50% training, 10% validation, and 40% deployment evaluation data. The evaluation data are split into four periods, with drift measured by the relative RMSE increase from the first to the last. Results are averaged across the five locations. In S4, window-related periods at the South location and door-related periods at the North location are identified from the number of open windows and the entrance-door state, respectively. Using the same trained model, eventperiod RMSE is evaluated for $\mathrm { { C O } _ { 2 } }$ and temperature, which are sensitive to external environmental events.

## 6<sub>.</sub>3 B<sub>ase</sub>lin<sub>es</sub>

We compare the proposed method with representative attention-based, linear/MLP-based, and calibration-specific models. The baselines include iTransformer (Liu et al. 2024) and TimeXer (Wang et al. 2024b), followed by DLinear (Zeng et al. 2023), XLinear (Chen et al. 2026), and TimeMixer (Wang et al. 2024a). We further include SenDaL (Ahn et al. 2024) and TESLA (Ahn et al. 2025), which were specifically developed for eficient sensor calibration. Veli (Dalbah, Worring, and Hsu 2026) is excluded because it is unsupervised, unlike our supervised benchmark.

## 6<sub>.</sub>4 E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs</sub>

Calibration accuracy is measured using RMSE in the original physical unit of each sensor variable. RMSE penalizes large deviations, which are particularly relevant under long-term drift and abrupt environmental changes. Computational eficiency is evaluated using per-sample inference time and peak inference memory. Inference time is averaged over repeated runs after graph warm-up using a fixed benchmark batch. Peak memory is measured relative to the pre-inference process baseline.

Table 6: S3 average relative increase from the earliest to the latest deployment period across five locations.
<table><tr><td rowspan="2">Model</td><td colspan="4">Relative increase (%)↓</td></tr><tr><td>CO</td><td> $\mathrm { C O _ { 2 } }$  T.</td><td>RH.</td><td> $\mathrm { P M } _ { 1 }$ </td></tr><tr><td>iTransformer</td><td>28.6</td><td>56.0</td><td>32.3 47.6</td><td>67.8</td></tr><tr><td>TimeXer</td><td>64.4</td><td>42.2</td><td>53.7 67.2</td><td>84.5</td></tr><tr><td>DLinear</td><td>63.8</td><td>84.9</td><td>49.2 65.1</td><td>108.8</td></tr><tr><td>XLinear</td><td>43.3</td><td>82.4</td><td>56.8 75.2</td><td>77.6</td></tr><tr><td>TimeMixer</td><td>64.2</td><td>73.5</td><td>61.1 60.1</td><td>97.5</td></tr><tr><td>SenDaL</td><td>87.1</td><td>78.6</td><td>62.4 105.9</td><td>75.2</td></tr><tr><td>TESLA</td><td>43.4</td><td>56.8</td><td>59.2 46.5</td><td>64.6</td></tr><tr><td>Ours</td><td>24.6</td><td>37.0</td><td>32.2 30.6</td><td>57.2</td></tr></table>

Table 7: S4 event-conditioned calibration results at the window/door-side locations. Each value reports RMSE during the corresponding event periods.
<table><tr><td rowspan="2">Model</td><td colspan="2">South (Window)</td><td colspan="2">North (Door)</td><td rowspan="2">Avg.</td></tr><tr><td> $\mathrm { C O _ { 2 } }$ </td><td>T.</td><td> $\mathrm { C O _ { 2 } }$ </td><td>T.</td></tr><tr><td>iTransformer</td><td>18.48</td><td>7.97</td><td>19.26</td><td>7.29</td><td>13.25</td></tr><tr><td>TimeXer</td><td>21.69</td><td>9.49</td><td>24.33</td><td>10.68</td><td>16.55</td></tr><tr><td>DLinear</td><td>20.55</td><td>10.37</td><td>24.08</td><td>10.93</td><td>16.48</td></tr><tr><td>XLinear</td><td>23.74</td><td>8.05</td><td>26.55</td><td>10.48</td><td>17.20</td></tr><tr><td>TimeMixer</td><td>19.21</td><td>10.58</td><td>32.08</td><td>11.69</td><td>18.39</td></tr><tr><td>SenDaL</td><td>20.68</td><td>10.61</td><td>15.35</td><td>9.79</td><td>14.11</td></tr><tr><td>TESLA</td><td>17.70</td><td>5.54</td><td>21.07</td><td>6.27</td><td>12.64</td></tr><tr><td>Ours</td><td>17.60</td><td>5.55</td><td>15.51</td><td>5.78</td><td>11.11</td></tr></table>

## 6<sub>.</sub>5 I<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

All models are implemented in TensorFlow 2.14 and trained under a common protocol adopted from Ahn et al. (2025), following the iTransformer configuration (Liu et al. 2024). Eficiency is measured under identical conditions on an AMD EPYC 7513 32-Core Processor. For embedded evaluation, the trained models are converted to TensorFlow Lite Flat-Bufers and deployed on an Arduino Nano 33 BLE Sense.

## 7 E<sub>xper</sub>i<sub>men</sub>t<sub>a</sub>l R<sub>esu</sub>lt<sub>s</sub>

This section presents calibration and eficiency results under standard and four deployment-oriented scenario.

## 7<sub>.</sub>1 O<sub>ve</sub>r<sub>a</sub>ll C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub>

Table 3 compares calibration performance on SensEURCity and the proposed indoor dataset. The proposed model achieved the lowest RMSE for all three outdoor variables and four of the five indoor variables, while ranking second for indoor temperature. These results establish strong performance under co-located reference supervision.

![](images/bc8048c5f5923083ee808dae5891bbf654adbe9cf426202e0fb5e5b7ecb1b7c9.jpg)

![](images/4becd16fe73bc0a7af7349a5cb90f8922cdb953b06d8e462618f0847d819b070.jpg)  
Figure 3: Performance on Arduino Nano 33 BLE Sense. The graphs show mean inference latency and FlatBufer size across input windows of 15, 60, 360, 720, and 1440. Missing points indicate that on-device measurement was not possible.

## 7<sub>.</sub>2 R1<sub>.</sub> R<sub>e</sub>f<sub>e</sub>r<sub>e</sub>n<sub>ce-</sub>Efi<sub>c</sub>i<sub>e</sub>nt C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n

Table 4 shows that the proposed model achieved the lowest RMSE for $\mathrm { C O } , \mathrm { C O } _ { 2 } ,$ humidity, and $\mathrm { P M } _ { \mathrm { 1 } } .$ . TESLA outperformed iTransformer on four indoor variables under strict pairwise calibration, whereas iTransformer performed better for $\mathrm { { C O } _ { 2 } }$ , temperature, and humidity in S1. This rank reversal shows that overall calibration performance does not necessarily imply efective sharing of limited reference supervision.

## 7<sub>.</sub>3 R2<sub>.</sub> L<sub>oca</sub>ti<sub>o</sub>n<sub>-</sub>Tr<sub>a</sub>n<sub>s</sub>f<sub>e</sub>r C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n

Table 5 shows higher RMSE than S1 for all five variables of the proposed model, reflecting the dificulty of transfer to unseen locations. Nevertheless, the model achieved the best results for $\mathrm { C O _ { 2 } } .$ , temperature, humidity, and $\mathrm { P M _ { 1 } }$ , and ranked second for CO. The S1-to-S2 increase varied from 1.3% for $\mathrm { P M _ { 1 } }$ to 61.2% for CO, showing that reference-eficient learning and unseen-location transfer impose distinct challenges.

## 7<sub>.</sub>4 R3<sub>.</sub> L<sub>o</sub>n<sub>g-</sub>T<sub>e</sub>rm C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n Drift

Table 6 shows positive RMSE increases for all models and variables, demonstrating persistent temporal degradation over the six-month deployment. The proposed model achieved the lowest relative increase for all five variables, although its RMSE still rise by 24.6-57.2%. Despite outperforming iTransformer under standard calibration, TESLA showed larger degradation for CO, $\mathrm { C O _ { 2 } }$ , and temperature. Thus, standard accuracy does not necessarily indicate longterm robustness.

## 7<sub>.</sub>5 R4<sub>.</sub> E<sub>ve</sub>nt-C<sub>o</sub>nditi<sub>o</sub>n<sub>e</sub>d C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n

Table 7 shows that the best-performing model varied across event conditions. TESLA achieved the lowest RMSE for South-window temperature, whereas SenDaL performed best for North-door $\mathrm { C O _ { 2 } } .$ . The proposed model performed best for the other two conditions and ranked first or second in all four cases. These condition-specific rankings show that aggregate results can obscure location- and modalitydependent event responses.

## 7<sub>.</sub>6 R5<sub>.</sub> Efi<sub>c</sub>i<sub>e</sub>n<sub>cy</sub> <sub>a</sub>nd On<sub>-</sub>d<sub>ev</sub>i<sub>ce</sub> P<sub>e</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub>

Figure 3 shows that the proposed model achieved the fastest inference time with low peak memory. We compare DLinear, the fastest on-device linear baseline, and TESLA, a recent calibration transformer model (Ahn et al. 2025). On the

Arduino Nano 33 BLE Sense, it was faster than TESLA at all sequence lengths and than DLinear for long input windows. Its FlatBufer size was larger than DLinear but smaller than TESLA, which failed at the longest sequence length. These results demonstrate eficient long-window calibration on resource-constrained devices.

## 8 Di<sub>scuss</sub>i<sub>o</sub>n <sub>a</sub>nd Limit<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>

We discuss the implications of our work.

## 8<sub>.</sub>1 D<sub>a</sub>t<sub>ase</sub>t <sub>a</sub>nd S<sub>ce</sub>n<sub>a</sub>ri<sub>o</sub>

Model rankings changed with reference coverage, targetlocation exposure, deployment time, and event conditions. This shows that location-specific accuracy, referenceeficient learning, unseen-location transfer, long-term robustness, and event robustness are distinct calibration capabilities. The proposed scenarios therefore reveal deployment challenges hidden by standard aggregate evaluation. The synchronized multi-location measurements, six-month time series, and event metadata support complementary evaluation of spatial and temporal robustness.

## 8<sub>.</sub>2 M<sub>o</sub>d<sub>e</sub>l <sub>an</sub>d D<sub>ep</sub>l<sub>oymen</sub>t

Our model ranked first or second across the deployment scenarios while maintaining low edge-inference cost. These results indicate robust calibration across locations, time periods, and event conditions without sacrificing eficiency. However, the 24.6-57.2% degradation in S3 shows that static calibration cannot eliminate long-term drift and motivates periodic recalibration or online adaptation. The on-device results further demonstrate eficient inference with input windows of up to 1,440 time steps.

## 8<sub>.</sub>3 Sin<sub>g</sub>l<sub>e-</sub>En<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt S<sub>cope</sub> <sub>a</sub>nd D<sub>a</sub>t<sub>a</sub> Fid<sub>e</sub>lit<sub>y</sub>

The indoor dataset was collected at five locations within one building, limiting its coverage of other buildings, climates, and ventilation conditions. However, this controlled setting enabled dense, synchronized, and long-term measurements under a consistent hardware and sampling protocol. It also supported uniform annotation of door state, window status, occupancy estimates, and business hours. Because the metadata were recorded at fixed intervals rather than exact event times, S4 evaluates event-conditioned periods rather than precisely aligned event responses.

## 9 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

We presented a deployment-oriented benchmark for indoor low-cost sensor calibration. We constructed a six-month, five-location dataset and defined four scenarios covering limited reference availability, unseen locations, long-term drift, and event-conditioned shifts. Based on these scenarios, we derived design requirements and proposed a lightweight temporal model for robust and eficient calibration. Experimental results demonstrate strong performance across the four scenarios with low edge-inference cost. Future work will extend the dataset to diverse indoor environments and explore online adaptation for long-term drift.

Ahn, S.; Kim, H.; Lee, E.; and Seo, Y.-D. 2024. SenDaL: An Efective and Eficient Calibration Framework of Low-Cost Sensors for Daily Life. IEEE Internet of Things Journal, 11(11): 20619–20630.

Ahn, S.; Kim, H.; Shin, S.; and Seo, Y.-D. 2025. Real-Time Calibration Model for Low-Cost Sensor in Fine-Grained Time Series. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 3–11.

Allka, X.; Ferrer-Cid, P.; Barcelo-Ordinas, J. M.; and Garcia-Vidal, J. 2023. Temporal pattern-based denoising and calibration for low-cost sensors in iot monitoring platforms. IEEE Transactions on Instrumentation and Measurement, 72: 1–11.

Aula, K.; Lagerspetz, E.; Nurmi, P.; and Tarkoma, S. 2022. Evaluation of low-cost air quality sensor calibration models. ACM transactions on sensor networks, 18(4): 1–32.

Bachechi, C.; Rollo, F.; and Po, L. 2024. HypeAIR: A novel framework for real-time low-cost sensor calibration for air quality monitoring in smart cities. Ecological Informatics, 81: 102568.

Chen, X.; Jin, H.; Huang, Y.; and Feng, Z. 2026. XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs. Proceedings of the AAAI Conference on Artificial Intelligence, 40(24): 20325–20335.

Concas, F.; Mineraud, J.; Lagerspetz, E.; Varjonen, S.; Liu, X.; Puolamäki, K.; Nurmi, P.; and Tarkoma, S. 2021. Lowcost outdoor air quality monitoring and sensor calibration: A survey and critical analysis. ACM Transactions on Sensor Networks (TOSN), 17(2): 1–44.

Dalbah, Y.; Worring, M.; and Hsu, Y.-C. 2026. Veli: Unsupervised Method and Unified Benchmark for Low-Cost Air Quality Sensor Correction. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 20684–20692.

Jiang, Y.; Li, K.; Tian, L.; Piedrahita, R.; Yun, X.; Mansata, O.; Lv, Q.; Dick, R. P.; Hannigan, M.; and Shang, L. 2011. MAQS: a personalized mobile sensing system for indoor air quality monitoring. In Proceedings of the 13th international conference on Ubiquitous computing, 271–280.

Koziel, S.; Pietrenko-Dabrowska, A.; Wojcikowski, M.; and Pankiewicz, B. 2025. Eficient field correction of low-cost particulate matter sensors using machine learning, mixed multiplicative/additive scaling and extended calibration inputs. Scientific Reports, 15(1): 18573.

Li, J.; Li, H.; Ma, Y.; Wang, Y.; Abokifa, A. A.; Lu, C.; and Biswas, P. 2018. Spatiotemporal distribution of indoor particulate matter concentration with a low-cost sensor network. Building and Environment, 127: 138–147.

Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and Long, M. 2024. iTransformer: Inverted Transformers Are Efective for Time Series Forecasting. In International Conference on Learning Representations.

Van Poppel, M.; Schneider, P.; Peters, J.; Yatkin, S.; Gerboles, M.; Matheeussen, C.; Bartonova, A.; Davila, S.; Signorini, M.; Vogt, M.; Dauge, F. R.; Skaar, J. S.; and Haugen, R. 2023.

SensEURCity: A multi-city air quality dataset collected for 2020/2021 using open low-cost sensor systems. Scientific Data, 10(1): 322.

Vito, S. 2008. Air quality. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5060Z.

Wang, S.; Wu, H.; Shi, X.; Hu, T.; Luo, H.; Ma, L.; Zhang, J. Y.; and ZHOU, J. 2024a. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. In International Conference on Learning Representations (ICLR).

Wang, Y.; Wu, H.; Dong, J.; Liu, Y.; Qiu, Y.; Zhang, H.; Wang, J.; and Long, M. 2024b. Timexer: Empowering transformers for time series forecasting with exogenous variables. Advances in Neural Information Processing Systems.

Wu, T.; Tasoglou, A.; Huber, H.; Stevens, P. S.; and Boor, B. E. 2021. Influence of mechanical ventilation systems and human occupancy on time-resolved source rates of volatile skin oil ozonolysis products in a LEED-certified ofice building. Environmental Science & Technology, 55(24): 16477– 16488.

Yu, H.; Li, Q.; Geng, Y.-a.; Zhang, Y.; and Wei, Z. 2020. Airnet: A calibration model for low-cost air monitoring sensors using dual sequence encoder networks. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, 1129–1136.

Zeng, A.; Chen, M.; Zhang, L.; and Xu, Q. 2023. Are transformers efective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, 11121– 11128.

Zhong, S.; Lalanne, D.; and Alavi, H. 2021. The complexity of indoor air quality forecasting and the simplicity of interacting with it–a case study of 1007 ofice meetings. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, 1–19.