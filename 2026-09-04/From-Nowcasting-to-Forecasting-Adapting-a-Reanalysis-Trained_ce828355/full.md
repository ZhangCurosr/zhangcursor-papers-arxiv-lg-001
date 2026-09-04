# From Nowcasting to Forecasting: Adapting a Reanalysis-Trained Cloud Cover Model to Observations

Mikko Partio<sup>1,2∗</sup>

Leila Hieta<sup>1</sup>

Ossi Laine<sup>1</sup>

<sup>1</sup>Finnish Meteorological Institute <sup>2</sup>Aalto University

September 4, 2026

## Abstract

Accurate cloud-cover forecasts are important for temperature prediction, radiation forecasting, and solar-power operations. Short-range forecasting methods can preserve observed cloud placement during the first forecast hours, but their skill decreases when cloud fields evolve through formation, dissipation and deformation. Longer lead times require accounting for atmospheric evolution, but operational numerical weather prediction (NWP) forecasts may not accurately represent the satelliteobserved cloud state at initialization. We develop CloudCast v2, a machine-learning model for 12-hour cloud-cover forecasting from observation-based initial conditions. The model is first trained on the Copernicus European Regional Reanalysis (Ridal et al., 2024) to learn cloud-evolution dynamics, and is then adapted to satellite-derived cloud fields using conditional flow matching (Lipman et al., 2023), a generative method that transforms noise into cloud-cover forecasts conditioned on the observed initial cloud fields and NWP inputs. CloudCast v2 reduces mean absolute error by 10% relative to its predecessor, CloudCast v1 (Partio et al., 2025), over the 1–12 h range. It also overtakes CloudCast v1 in fractions skill score, a neighborhood-based measure of spatial agreement, after approximately 3–6 h, depending on the cloudiness category. These results show that observation-initialized machinelearning forecasts can extend beyond the usual 1–3-hour nowcasting range while retaining spatial detail from satellite cloud fields.

Keywords: cloud cover, nowcasting, forecasting, machine learning, conditional flow matching, reanalysis

## 1 Introduction

Cloud cover is a major regulator of the Earth’s radiation budget and an important driver of both weather and climate (Hughes et al., 2024). Accurate cloud-cover forecasts are therefore valuable not only for meteorological prediction, but also for applications such as solar-power management and energy-system operations. Errors in cloud prediction can propagate into substantial biases in near-surface temperature, radiation, and other forecast variables. At local scales, the errors increase uncertainty in solar energy generation and decrease the reliability of operational decision-making (Wang et al., 2022).

Short-range cloud forecasting sits between observation-based nowcasting and dynamical weather prediction. Across the 1–12 hour range considered here, forecasts commonly rely on either numerical weather prediction (NWP) or observation-based extrapolation. NWP models can represent atmospheric dynamics and are therefore able to predict evolving cloud structures beyond simple advection. Cloud cover remains dificult to predict because it depends strongly on moisture and vertical motion, as well as on subgrid-scale processes that must be parameterized. Moreover, initial-state errors can strongly afect forecast quality (Sun et al., 2014). Extrapolation methods, in contrast, start from the latest observations and estimate future cloud fields by moving the observed cloud patterns according to their recent motion. These methods are often efective at lead times of one to two hours, when much of the forecast skill comes from persistence and advection (Prudden et al., 2020). At longer lead times of three hours or more, their skill drops as cloud formation and dissipation become important (Partio et al., 2025).

In operational forecasting, users need a single forecast covering the full lead-time range rather than separate forecasts for diferent lead times. This requirement creates a dificult transition: extrapolationbased methods tend to perform best at the earliest lead times, while dynamical models become more useful as lead time increases. Previous work has blended the approaches in image space across diferent lead-time ranges (Hieta et al., 2021); however, producing one coherent forecast remains nontrivial and can introduce visible discontinuities when the underlying forecasts disagree strongly. This transition between nowcasting and forecasting remains a central challenge for operational cloud-cover prediction.

Research on observation-based short-range forecasting has traditionally focused on extrapolating observed fields, particularly radar-derived precipitation fields, while more recent machine-learning approaches learn their evolution from data (Prudden et al., 2020). Recurrent neural networks (RNNs) were among the early deep-learning approaches (Shi et al., 2015) and were followed by convolutional architectures (Ayzel et al., 2020; Ha and Lee, 2024; Ritvanen et al., 2023). While deep generative radar nowcasting has improved precipitation forecasts at lead times of 5–90 minutes (Ravuri et al., 2021), other machine-learning models have extended high-resolution precipitation forecasting to 12 hours (Espeholt et al., 2022). More recently, conditional flow matching, a generative method that transforms random noise into forecast fields conditioned on the input data, has been applied to probabilistic precipitation nowcasting (Ribeiro and Pucer, 2026). These studies have, however, mainly addressed radar-observed precipitation. Satellite-derived cloud fields represent a diferent prediction target, so the applicability of these methods to cloud-cover forecasting remains uncertain.

Cloud-cover forecasting has received less attention than precipitation nowcasting. It poses a related but distinct prediction problem: cloud cover is a bounded fractional field, with predominantly clear-sky and overcast regimes and complex intermediate structures associated with partial or broken cloud cover. Recent studies have begun to target cloud cover directly with dedicated machine-learning models. For example, Kellerhals et al. (2022) and Xia et al. (2024) developed RNNs to predict cloud cover, while Partio et al. (2025) introduced CloudCast v1, a U-Net-based model trained on satellite-derived efective cloudiness to forecast cloud cover up to five hours ahead, with a reported skillful lead time of about three hours. Generative approaches have also been applied to satellite imagery. Chase et al. (2026) investigated score-based difusion for GOES-16 nowcasting, while Chen et al. (2026) introduced SATcast, a cascade difusion model conditioned on atmospheric-model fields. Both methods predict brightness temperature rather than fractional cloud cover.

In parallel, machine-learning weather forecasting has shown that models trained on reanalysis data can learn to predict atmospheric evolution beyond the nowcasting range. At synoptic and global scales, this progress includes GraphCast (Lam et al., 2023), PanguWeather (Bi et al., 2023), and AIFS (Lang et al., 2024). Regional studies have extended the same idea to limited-area forecasting, including graph-based neural weather prediction (Oskarsson et al., 2023) and stretched-grid architectures (Nipen et al., 2026). These models target forecast lead times of several days, but they do not yet show how reanalysis-trained forecasting systems can be adapted to observation-based initial conditions for short-range prediction. Recent work has begun to connect these approaches by combining observations with NWP inputs in regional weather forecasting (Miralles et al., 2026), but adapting a reanalysis-trained model to satellitederived fractional cloud cover for 12-hour forecasting remains underexplored.

We therefore develop CloudCast v2, a unified machine-learning approach for 12-hour cloud-cover forecasting. First, we train a vision transformer-based model on reanalysis data to learn to predict cloud evolution on a 5 km grid covering Scandinavia. We then adapt the model to observation-based cloud fields using a clean-target formulation based on conditional flow matching (Lipman et al., 2023). The model also receives atmospheric fields from NWP forecasts at each forecast lead. This design allows the model to retain dynamics learned from reanalysis while aligning its forecasts with observed cloud fields. We evaluate CloudCast v2 against nowcasting and NWP baselines using pixel-wise error, neighborhood-based spatial verification, and skill scores for cases with large cloud-cover changes. We show that CloudCast v2 can be initialized from observed cloud fields while extending useful forecast skill to longer lead times than CloudCast v1.

The remainder of this paper is organized as follows. Section 2 describes the data and methods, including the model architecture and training setup. Section 3 presents the forecast evaluation results, and Section 4 discusses their implications and outlines directions for future research.

## 2 Data and methods

This section defines the data, forecasting task, model, training protocol, and evaluation framework used in the study. We first describe the regional domain and the three data sources: the Copernicus European Regional Reanalysis (CERRA) (Ridal et al., 2024) for the long reanalysis record, Nowcasting Satellite Application Facilities (NWCSAF) efective cloudiness (Kerdraon and Fontaine, 2021) as the observationspace target, and MetCoOp Ensemble Prediction System (MEPS) forecasts (Eresmaa et al., 2026), whose atmospheric fields provide time-varying inputs to CloudCast v2 and whose cloud-cover forecasts serve as the NWP baseline. We then describe the architecture of CloudCast $\mathrm { v 2 . }$ how the model is trained and adapted to observations, and how its forecasts are evaluated.

## 2.1 Study domain and datasets

We conduct the study on a regional domain covering Scandinavia, defined on a common 5 km Lambert conformal conic grid. Before training and evaluation, we reproject all cloud-cover and atmospheric fields from their native grids to this common grid so that the diferent data sources are spatially aligned. Figure 1 compares the CERRA, MEPS, and NWCSAF cloud fields on this grid. In the example shown, all three exhibit the same broad cloud pattern, but CERRA and MEPS have less small-scale variability and fewer values near 0 and 1 than NWCSAF.

We use CERRA reanalysis data for model pre-training. CERRA covers Europe at 5.5 km horizontal resolution and includes both surface-level and pressure-level variables from 1984 onward. CERRA analyses are available every three hours; we construct a uniform hourly dataset by filling the intervening hours with the corresponding one- and two-hour forecasts. This gives the pre-training stage a long record from which the model can learn cloud evolution before adaptation to observations.

To adapt the CERRA-pretrained model to satellite observations, we use NWCSAF efective cloudiness for the initial cloud fields and prediction target during fine-tuning. NWCSAF is an EUMETSAT programme that supports operational nowcasting with satellite-derived cloud data from geostationary platforms. The efective-cloudiness product is derived from SEVIRI observations, mainly using the 10.8 µm infrared window channel together with cloud fraction and cloud emissivity (Aminou et al., 1997; Ker draon and Fontaine, 2021). It ranges from 0 for clear sky to 1 for optically thick cloud, with intermediate values representing semi-transparent cloud layers. The efective-cloudiness fields have a spatial resolution of 3 km at nadir and are available every 15 min.

CERRA (reanalysis)  
![](images/3e8f9a0fe5ad0edffbcfd7ca5ee82d913af81fd57e65da481bdda526f0ae70b2.jpg)

MEPS (NWP forecast)  
![](images/7d44fc459713708cc852163eb87eba3fadabf70d179c424e177d4488e70aaee9.jpg)

NWCSAF (satellite)  
![](images/1454e7e6a4ebbbaeb542178b931f86674bd0ff97db8be69bdca4b0ec909ae077.jpg)  
Total cloud cover (0 1)  
Figure 1: Comparison of cloud-cover fields on the common 5 km grid for 3 June 2021 at 12 UTC. The panels show CERRA and MEPS total cloud cover, and NWCSAF efective cloudiness, for the same valid time. The persistent feature in the northeastern corner of the NWCSAF panel is a known retrieval artefact in the satellite data and is retained in the data used here.

Satellite observations provide the cloud-cover target, but they do not provide the future atmospheric state required for forecasting. We therefore use forecasts from MEPS, a regional numerical weather prediction system based on HARMONIE-AROME cycle 43h2.2 (Bengtsson et al., 2017), operated by the meteorological institutes of Estonia, Finland, Norway, and Sweden. It is initialized every three hours and outputs forecasts at hourly lead times up to 66 hours ahead on a 2.5 km grid. In this study, atmospheric fields from MEPS serve as time-varying inputs to CloudCast v2, while MEPS total-cloud-cover forecasts serve as the NWP baseline.

## 2.2 Forecasting task

Each forecast case is initialized at time t over the entire common 5 km grid. The initial-state input consists of two gridded cloud-cover fields valid at t − 1 and t. The model also receives static fields and atmospheric fields valid at the two history times and at each forecast lead. We refer to the time-varying inputs, including atmospheric variables, insolation, and temporal encodings, as dynamic forcings. The model predicts cloud-cover fields from t + 1 to t + 12. Depending on the training stage, these forecasts are generated either autoregressively or directly from the initial state. The atmospheric input fields must be available over the full forecast horizon; the model predicts only cloud cover, not the atmospheric variables themselves. During pre-training, CERRA supplies both the cloud-cover fields and dynamic forcings. During fine-tuning, NWCSAF supplies the cloud-cover history and prediction targets, while MEPS supplies the dynamic forcings; the same NWCSAF–MEPS input combination is used at inference.

The model inputs describe three aspects of the forecasting problem: the recent cloud field, the surrounding atmospheric state, and the spatial and temporal setting. The variables were selected based on their physical relevance to cloud evolution and prior modelling experience. Wind represents cloud transport, temperature and humidity describe the thermodynamic environment, and geopotential provides information about the large-scale atmospheric state. Insolation, temporal encodings, and static fields account for diurnal, seasonal, and geographical influences. We use atmospheric variables at 1000, 925, 850, 700, and 500 hPa to sample the lower and middle troposphere while keeping the number of input fields manageable. The variables are organized as anemoi-datasets (Lang et al., 2024) and listed in Table 1.

<table><tr><td>Variable</td><td>Field type</td><td>Model role</td></tr><tr><td>Cloud cover</td><td>Total-column field</td><td>Prognostic</td></tr><tr><td>U-component of wind</td><td>Pressure-level field</td><td>Dynamic forcing</td></tr><tr><td>V-component of wind</td><td>Pressure-level field</td><td>Dynamic forcing</td></tr><tr><td>Temperature</td><td>Pressure-level field</td><td>Dynamic forcing</td></tr><tr><td>Relative humidity</td><td>Pressure-level field</td><td>Dynamic forcing</td></tr><tr><td>Geopotential</td><td>Pressure-level field</td><td>Dynamic forcing</td></tr><tr><td>Insolation</td><td>Solar forcing field</td><td>Dynamic forcing</td></tr><tr><td>Julian day</td><td>Temporal encoding</td><td>Dynamic forcing</td></tr><tr><td>Local time</td><td>Temporal encoding</td><td>Dynamic forcing</td></tr><tr><td>Latitude</td><td>Coordinate field</td><td>Static forcing</td></tr><tr><td>Longitude</td><td>Coordinate field</td><td>Static forcing</td></tr><tr><td>Orography</td><td>Surface field</td><td></td></tr><tr><td></td><td></td><td>Static forcing</td></tr><tr><td>Land-sea mask</td><td>Surface field</td><td>Static forcing</td></tr></table>

Table 1: Variables used by the model. Pressure-level fields are included at 1000, 925, 850, 700, and 500 hPa; insolation and temporal encodings are precomputed using anemoi-datasets (Lang et al., 2024). Dynamic forcings vary over the forecast horizon, whereas static forcings are fixed in time.

## 2.3 Model architecture

Our model is a U-shaped vision transformer based on the Swin Transformer architecture (Liu et al., 2021). We represent the regional domain on a regular grid, allowing the model to treat the input fields as spatially ordered patches. Full self-attention (Vaswani et al., 2017; Dosovitskiy et al., 2021) scales quadratically with the number of patches and would exceed the memory available for the regional domain. Swin instead computes attention within local windows and shifts the window partition between successive blocks. This reduces memory requirements while allowing information to propagate across window boundaries.

The network follows an encoder–decoder structure with two encoder stages and two decoder stages, as shown in Figure 2. he two-step cloud-cover and forcing histories are embedded into 4 × 4-pixel patches before entering the encoder. The two resolution levels combine local cloud structure with broader atmospheric context: on the 5 km grid, the attention windows span approximately 120 km in the first stage and 400 km in the second. The decoder combines information from both resolutions and reconstructs a full-resolution cloud-cover tendency, which is added to the current cloud field. The core architecture remains unchanged during pre-training and fine-tuning; only the training data and adaptation objective change.

![](images/b92c92770f375c10e6042a7d855e396d1144dbc36af28cc7d380e077edcdcbbc.jpg)

![](images/4d0bfdc345f3c5c2c199c1e173d34ac699be0a5c003bd01c4dbdd8ffc8a192a9.jpg)  
Figure 2: Overview of the CloudCast v2 architecture for target lead time t + k. Cloud-cover history and concurrent forcing history are embedded into patch tokens and processed by two encoder stages. The forcing fields valid at t + k are embedded separately and added before the first decoder stage. Patch expansion and reconstruction restore the spatial resolution, and the refinement head predicts a cloudcover tendency, which is added to the current cloud cover to obtain the forecast. During direct prediction, the encoded history is shared and the decoder is applied independently for $k = 1 , \ldots , 1 2$ . The dashed arrow denotes the encoder–decoder skip connection. Tensor shapes omit the batch dimension. Here, $T = 2 , C _ { c } = 1 , C _ { f } = 3 6 , D = 2 5 6 , H = 5 3 5$ , and $W = 4 7 5$ . The two additional conditioning channels used during flow-matching adaptation are not shown.  
Figure 3: Encoder and decoder block structure. (a) The encoder block arranges the embedded patches on their spatial grid, partitions the grid into local windows for self-attention, and then applies a feed-forward layer and depth-wise convolution for additional local spatial mixing. (b) The decoder block uses the same self-attention and refinement operations, but also applies window cross-attention to incorporate context tokens from the encoded representation or skip connection. Tensor shapes omit the batch dimension. At stage $s ,$ the input, context, and output token sequences have shape $T _ { s } P _ { s } \times D _ { s }$ , where $\begin{array} { r } { P _ { s } = H _ { s } W _ { s } } \end{array}$ The decoder tokens provide the queries in cross-attention, while the context tokens provide the keys and values. The decoder input has already been conditioned on the future forcing shown in Figure 2. Normalization and residual connections are omitted for clarity.

Figure 3 summarizes the internal structure of the encoder and decoder blocks. Both combine windowed self-attention with depth-wise convolution, which adds local spatial mixing at relatively low computational cost. In the decoder, tokens conditioned on the forcing fields for the target lead time use cross-attention to incorporate context from the encoded representation and skip connection. During direct prediction, this allows the encoded initial state to be shared across lead times while each decoder output is conditioned

on the corresponding atmospheric state.

## 2.4 Observation-space adaptation with conditional flow matching

Long reanalysis records provide extensive training data for machine-learning weather models, but using satellite observations for initialization introduces a distribution shift. To transfer the CERRA-trained model to satellite observations, we first fine-tune it on NWCSAF efective cloudiness with a deterministic objective. This step adapts the model to observed cloud fields, but the resulting forecasts tend to lack finescale cloud structure: the adaptation remains dificult because the observational record is much shorter than the reanalysis record, and because CERRA total cloud cover and NWCSAF efective cloudiness represent related but diferent cloud quantities. To better preserve fine-scale cloud structure, we therefore add a generative adaptation stage based on conditional flow matching (CFM) (Lipman et al., 2023), with the clean-target parameterization described below.

Conditional flow matching is a generative modeling method that learns a transport from a simple source distribution to a target data distribution. Rather than producing only a deterministic point forecast, the model learns a conditional denoising process that can sample plausible fields from the target distribution. In our setting, the source distribution is Gaussian noise and the target is the observationspace distribution of cloud fields. During training, we sample a noise level $\alpha \in [ 0 , 1 ]$ , draw Gaussian noise $z ,$ and construct the intermediate field in Equation (1).

$$
x _ { \alpha } = ( 1 - \alpha ) y + \alpha z\tag{1}
$$

where $y$ is the NWCSAF efective-cloudiness target field. Unlike conventional CFM formulations, which train the model to predict the velocity along the noise-to-data path, we use a clean-target parameterization. This choice preserves the model’s original cloud-cover prediction task, allowing initialization from the deterministic weights and continued use of the cloud-specific loss functions. The model receives $x _ { \alpha }$ and $\alpha ,$ together with the encoded analysis fields and future forcings, and is trained to predict $y$ di rectly. At inference, sampling starts from Gaussian noise at $\alpha = 1$ . At each step, the model estimates the clean cloud field from the current intermediate field $x _ { \alpha }$ and uses this estimate to update $x _ { \alpha }$ at the next, lower value of $\alpha .$ . Repeating this process over decreasing noise levels progressively transforms the initial noise into a cloud-cover forecast.

We implement this clean-target formulation using the same CloudCast $\mathrm { v 2 }$ architecture as the deterministic fine-tuned model. We initialize it from the deterministic weights, but we do not pass the deterministic mean forecast to the sampler. Instead, we provide $x _ { \alpha }$ and α as additional future-forcing channels, while keeping the rest of the data encoding unchanged. This setup lets the model sample observation-space cloud fields while retaining the cloud-evolution dynamics learned during reanalysis pre-training.

## 2.5 Training protocol

We train the model in four sequential stages, moving from reanalysis pre-training to observation-space adaptation (Figure 4). All stages are run on AMD GPUs on the LUMI supercomputer, and their configurations are summarized in Table 2.

The first two stages pre-train the model on CERRA. Stage 1 uses one-step prediction with mean squared error (MSE) loss for 100,000 iterations, giving the model an initial representation of cloud evolution from the full CERRA record. Stage 2 continues CERRA pre-training using six-hour autoregressive rollouts. At each step, the cloud fields supplied as history for the next prediction are taken either from CERRA or from the model’s preceding forecasts. Scheduled sampling gradually increases the probability of using the model forecasts during training (Bengio et al., 2015). The loss initially emphasizes the first two forecast hours and gradually shifts toward equal weighting across all six hours.

In stage 3, we adapt the CERRA-trained model to satellite observations by fine-tuning it on NWCSAF efective cloudiness, which we treat as a bounded fractional target. We use direct prediction mode: the model predicts all 12 lead times from the same initial state instead of feeding its own predictions back autoregressively. Direct prediction avoids autoregressive error accumulation through the forecast sequence and lets each lead time use its corresponding MEPS forcing. The resulting model transfers the CERRAtrained dynamics to satellite-derived cloud fields and serves as the deterministic non-CFM ablation.

Stage 4 applies CFM as a second observation-space adaptation stage, initialized from the deterministic stage-3 model. We train the model with the noisy intermediate fields described in Section 2.4, while keeping the same direct prediction setup over the 12-hour horizon. This stage produces the final flowmatched CloudCast v2 model used in the main evaluation.

![](images/2f0c2d917597d39b0421fe26cc94d82386cfb2d47415db1070c7bea5870d1a85.jpg)  
Figure 4: Schematic overview of the four-stage CloudCast v2 training protocol. Stages 1–2 pre-train the model on CERRA reanalysis to learn cloud-evolution dynamics. Stages 3–4 adapt the model to NWCSAF efective cloudiness using MEPS forcings: stage 3 performs deterministic adaptation, and stage 4 applies conditional flow matching to produce the final observation-space forecast model. Arrow labels distinguish the training data supplied to each stage from the model weights passed forward through the training sequence.

In both observation-space adaptation stages, we combine binary cross-entropy (BCE) loss with a diferentiable approximation of the fractions skill score (FSS; Roberts and Lean, 2008) used in evaluation. We refer to this approximation as Soft-FSS. Soft-FSS replaces hard cloud-category thresholds with smooth thresholds and compares neighborhood cloud fractions over several spatial scales. We use the cloudcategory thresholds from Table 3 and neighborhood scales of 3, 6, and 12 grid cells.

## 2.6 Evaluation protocol and metrics

We evaluate forecasts on the fully held-out 2025 test year, with initializations every three hours. Verification covers lead times t + 1 to t + 12 against NWCSAF efective cloudiness on the common 5 km grid. After excluding initialization times with missing satellite scans or failed MEPS cycles, the common test set contains n = 2,855 initialization times.

We compare CloudCast v2 with two reference systems: CloudCast v1 (Partio et al., 2025) and the MEPS control forecast (Eresmaa et al., 2026). We also include a deterministic CloudCast v2 ablation trained with the same NWCSAF fine-tuning setup but without the CFM stage; this isolates the contribution of CFM to forecast skill. CloudCast v1 produces forecasts on its native 512 × 512 grid, whereas CloudCast v2 uses the 535×475 common verification grid. We therefore resample CloudCast v1 forecasts using nearest-neighbour interpolation. Consequently, CloudCast v1 has a small nonzero error at t + 0, despite using the same underlying observed initial field.

We evaluate forecast performance over the 1–12 h lead-time range from four complementary perspectives: (1) pixel-wise errors against NWCSAF efective cloudiness; (2) anomaly- and reference-based scores that allow comparison with MEPS despite cloud-variable diferences; (3) neighborhood-based scores of spatial cloud placement; and (4) skill in cases with substantial cloud-cover changes. Because the reference systems predict partly diferent cloud quantities, we apply each metric only where the comparison is meaningful.

We begin with bias and mean absolute error (MAE), which apply most directly to the observationspace forecasts. Bias measures the mean signed error and indicates whether forecasts systematically overpredict or underpredict cloud cover, as defined by Equation (2).

<table><tr><td>Configuration</td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td><td>Stage 4</td></tr><tr><td>Stage objective</td><td>CERRA one-step</td><td>CERRA rollout</td><td>NWCSAF deterministic</td><td>NWCSAF CFM</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>5e-4</td><td>2e-5</td><td>3e-5</td><td>5e-6</td></tr><tr><td>Scheduler</td><td>Cosine annealing</td><td>Cosine annealing</td><td>Cosine annealing</td><td>Cosine annealing</td></tr><tr><td>Steps</td><td>100,000</td><td>20,000</td><td>20,000</td><td>20,000</td></tr><tr><td>Warmup steps</td><td>5,000</td><td>1,000</td><td>500</td><td>500</td></tr><tr><td>Loss function</td><td>MSE</td><td>MSE</td><td>BCE + 0.2 × Soft-FSS</td><td>BCE + 0.2 × Soft-FSS</td></tr><tr><td>Rollout mode</td><td>One-step</td><td>Autoregressive</td><td>Direct</td><td>Direct</td></tr><tr><td>Rollout length</td><td>1</td><td>6</td><td>12</td><td>12</td></tr><tr><td>Scheduled sampling</td><td>No</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Effective batch size</td><td>128</td><td>128</td><td>32</td><td>32</td></tr><tr><td>Training data length</td><td>1984-2020</td><td>1984-2020</td><td>2020-2024</td><td>2020-2024</td></tr><tr><td>Training time (GPUh)</td><td>2,200</td><td>2,300</td><td>480</td><td>750</td></tr></table>

Table 2: Summary of the four-stage training protocol. Stages 1–2 pre-train the model on CERRA, stage 3 adapts it deterministically to NWCSAF efective cloudiness, and stage 4 applies CFM adaptation from the stage-3 checkpoint.

$$
\mathrm { { B i a s } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } ( f _ { i } - o _ { i } )\tag{2}
$$

where $f _ { i }$ and $o _ { i }$ are the forecast and observed efective-cloudiness values, respectively, and N is the number of verified grid-point values. With this convention, positive values indicate overprediction and negative values indicate underprediction. MAE measures the mean error magnitude regardless of sign, as defined by Equation (3).

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| f _ { i } - o _ { i } \right|\tag{3}
$$

We report bias and MAE only for systems that predict NWCSAF efective cloudiness directly. This includes CloudCast v1, CloudCast v2, the no-CFM ablation, and Eulerian persistence, which uses the initial observed cloud field as the forecast at every lead time. MEPS is excluded from bias and MAE because it forecasts cloud area fraction rather than efective cloudiness, creating a systematic ofset in direct comparison. We include MEPS only in the ACC and MSESS analyses described below.

Anomaly- and reference-based skill scores provide a way to compare forecast systems beyond direct pointwise error. We use two metrics: the anomaly correlation coeficient (ACC) and the mean squared error skill score (MSESS) (Murphy and Epstein, 1989). For each forecast system and the observations, we calculate a separate per-grid-point climatology for each valid calendar month and UTC hour by pooling all cases and lead times in the 2025 test set. ACC is then computed separately for each case and lead time using Equation (4).

$$
\mathrm { A C C } = \frac { \sum _ { i = 1 } ^ { N } \left( f _ { i } - c _ { i , m , h } ^ { ( f ) } \right) \left( o _ { i } - c _ { i , m , h } ^ { ( o ) } \right) } { \sqrt { \sum _ { i = 1 } ^ { N } \left( f _ { i } - c _ { i , m , h } ^ { ( f ) } \right) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { N } \left( o _ { i } - c _ { i , m , h } ^ { ( o ) } \right) ^ { 2 } } }\tag{4}
$$

where $f _ { i }$ and $o _ { i }$ are the forecast and observed values at grid point i, and $c _ { i , m , h } ^ { f }$ and $c _ { i , m , h } ^ { o }$ are their respective climatological means for valid month m and hour h. The score evaluates spatial agreement between forecast and observed anomalies without removing their spatial means again. It ranges from −1 to 1, with higher values indicating better anomaly-pattern agreement. Reported lead-time scores are averaged across cases.

MSESS compares the forecast MSE with that of Eulerian persistence, as defined by Equation (5). Both errors are calculated against NWCSAF efective cloudiness for all systems. The MEPS score therefore remains afected by the mismatch between cloud area fraction and efective cloudiness.

$$
\mathrm { M S E S S } = 1 - { \frac { \mathrm { M S E } _ { f c s t } } { \mathrm { M S E } _ { r e f } } }\tag{5}
$$

MSESS is 1 for a perfect forecast, 0 for skill equal to the reference forecast, and negative when the forecast is worse than the reference.

Pointwise and anomaly-based metrics do not account for small spatial displacement errors in coherent cloud structures. We therefore use FSS (Roberts and Lean, 2008), which evaluates spatial agreement by comparing the fraction of grid points exceeding a chosen threshold within a neighborhood around each point. The FSS is defined by Equation (6).

$$
\mathrm { F S S } = 1 - \frac { \sum _ { i = 1 } ^ { N } ( f _ { i } - o _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { N } f _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { N } o _ { i } ^ { 2 } }\tag{6}
$$

where $f _ { i }$ and $o _ { i }$ are the forecast and observed fractions of grid points exceeding the threshold in the neighborhood around point i. The FSS ranges from 0 to 1, with higher values indicating better spatial agreement. In this study, FSS is computed over neighborhood scales from 25 to 110 km and reported separately for each cloudiness category, allowing us to assess whether the models place cloud structures correctly even when exact grid-point agreement is imperfect.

Because MEPS predicts cloud area fraction rather than NWCSAF efective cloudiness, we exclude it from the category-based FSS comparison. Following Partio et al. (2025), we divide efective cloudiness into four cloudiness categories (Table 3).

<table><tr><td>Description</td><td>Cloudiness value</td><td>Octas</td><td>Grid-point fraction</td></tr><tr><td>Clear sky</td><td>[0, 0.0625]</td><td>0</td><td>0.24</td></tr><tr><td>Partly cloudy</td><td>(0.0625, 0.5625]</td><td>1-4</td><td>0.07</td></tr><tr><td>Mostly cloudy</td><td>(0.5625, 0.9375]</td><td>5-7</td><td>0.14</td></tr><tr><td>Overcast</td><td>(0.9375, 1.0]</td><td>8</td><td>0.55</td></tr></table>

Table 3: Cloudiness categories used for threshold-based verification. The cloudiness value gives the NWCSAF efective-cloudiness interval defining each category. Octas give the corresponding traditional cloud-cover category in eighths, and grid-point fraction gives the proportion of grid points in the tes dataset that fall into each category.

Finally, we evaluate cases in which the observed cloud field changes substantially during the forecast window. For each forecast, we measure the domain-mean squared 12-hour change in observed efective cloudiness using Equation (7).

$$
D = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( o _ { i , t + 1 2 } - o _ { i , t } \right) ^ { 2 }\tag{7}
$$

where $o _ { i , t }$ is the observed efective cloudiness at grid point i and time t. We rank all forecasts from the 2025 test year by D and retain the top 20% as the dynamic subset, corresponding to 570 forecasts.

For these rapidly changing cases, horizontal displacement of the initial cloud field provides a more relevant baseline than simple persistence. We therefore construct a Lagrangian persistence baseline by advecting the initial cloud-cover field with Färneback optical flow (Farnebäck, 2003), estimated from pre-forecast frames only, avoiding information from the verification period. Skill on the dynamic subset is reported as MSESS relative to this advected-persistence baseline, testing whether the model adds skill beyond displacement of the initial cloud field.

## 3 Results

This section compares CloudCast v2 with CloudCast v1, the CloudCast v2 no-CFM ablation, Eulerian persistence, and MEPS on the 2025 test set using the applicable metrics defined in Section 2.6.

## 3.1 Pixel-wise verification

We first assess pointwise forecast errors using bias and MAE (Fig. 5). Panel (a) shows that both CloudCast v2 variants remain close to zero bias throughout the forecast window. CloudCast v1 has a positive bias that peaks near t + 3 and subsequently decreases. Because its MAE continues to increase over the same period, the decreasing bias indicates greater cancellation between positive and negative errors rather than improving pointwise accuracy.

![](images/fd1943f8ea7a57ad80284e159266296457f3a8e0fc6928abf86de7b52c918555.jpg)

![](images/8e07b75a103af9d80af7a5e46b4a0c0052f66b0fe933807d847739a01726e01f.jpg)  
Figure 5: Pixel-wise forecast errors relative to NWCSAF efective cloudiness over the 2025 test set. (a) Bias, with positive values indicating overprediction, and (b) MAE as a function of lead time. Curves show CloudCast $\mathrm { v 2 , }$ CloudCast v2 without CFM, CloudCast v1, and Eulerian persistence. Lead time $t + 0$ is included as an initialization check; the nonzero CloudCast v1 error reflects resampling from its native grid.

Panel (b) shows that CloudCast v1 has the lowest MAE during the first two forecast hours. From approximately $t + 3$ , both CloudCast v2 variants have lower MAE than CloudCast v1 and Eulerian persistence. Their MAE growth also slows after about t + 5, whereas the errors of CloudCast v1 and persistence continue to increase, widening the advantage of CloudCast v2 at longer lead times. The no-CFM model has consistently slightly lower MAE than the CFM model, indicating that the benefit of flow matching is not primarily expressed in pixel-wise absolute error.

## 3.2 Pattern and reference-based skill

![](images/12b63ecb80cb6b6975088eedbeff76d704be5a8f3fb891361ac5a1e13aa512e7.jpg)

![](images/0de05ee41d6702607255a128dc15cfb072dc58b6d3d8db841eda134c744863e9.jpg)  
Figure 6: Anomaly and reference-based skill on the 2025 test set: (a) ACC against NWCSAF efective cloudiness and (b) MSESS relative to Eulerian persistence. ACC includes t + 0 as an initialization check, whereas MSESS begins at t + 1 because it is undefined at t + 0. Panel (a) includes all five systems; in panel (b), the horizontal line at zero denotes equal skill to Eulerian persistence.

Both CloudCast v2 variants maintain higher ACC than CloudCast v1, MEPS, and Eulerian persistence through most of the forecast window (Fig. 6a). The no-CFM variant has the highest ACC at all lead times, while the full CFM model remains close to it and clearly above CloudCast v1 after the first few hours. CloudCast v1 falls below the nearly constant MEPS ACC at about t + 5, whereas both CloudCast v2 variants remain above MEPS through t + 12. Eulerian persistence decreases most rapidly as the initial observed cloud field becomes outdated. MEPS, in contrast, has nearly constant ACC with lead time. Because it is not initialized from the observed NWCSAF cloud field, its ACC is already relatively low at short lead times; the nearly flat curve indicates that this initial disagreement changes little over the forecast window.

MSESS is positive for both CloudCast v2 variants throughout the forecast window and increases with lead time as their advantage over persistence widens (Fig. 6b). The no-CFM variant has the highest MSESS at every lead time; the full CFM model is below CloudCast v1 at t+1 but exceeds both CloudCast v1 and MEPS thereafter. MEPS becomes skillful relative to persistence at t + 3 and overtakes CloudCast v1 at about t + 7, but remains below both CloudCast v2 variants through t + 12.

## 3.3 Spatial skill across cloudiness regimes

To assess neighborhood-scale spatial agreement, Figure 7 shows FSS across the four cloudiness categories defined in Table 3, averaged over neighborhood scales from 25 to 110 km.

![](images/311ca78507f8ce6dc2888e9f2a24d39cc68258a9b607542dd4624fb6ffd2f681.jpg)  
Figure 7: FSS by cloudiness category as a function of forecast lead time, averaged over neighborhood scales from 25 to 110 km. Scores are computed against NWCSAF efective cloudiness for (a) clear sky, (b) partly cloudy, (c) mostly cloudy, and (d) overcast conditions, as defined in Table 3. All panels use the same y-axis scale. Curves show CloudCast v2 with and without CFM and CloudCast v1; higher values indicate better neighborhood-scale spatial agreement.

Across the forecast window, FSS remains substantially higher for clear-sky and overcast conditions than for the two intermediate categories. The lower skill for partly and mostly cloudy conditions is consistent with the greater spatial complexity of intermediate cloud fields.

CloudCast v1 performs best at t + 1 in every category, but its skill declines more rapidly. The full CloudCast v2 overtakes it at approximately t + 2 for mostly cloudy, t + 4 for clear-sky and overcast, and t + 6 for partly cloudy conditions, remaining more skillful thereafter. In contrast, the no-CFM variant remains below CloudCast v1 over most of the forecast window and only matches or exceeds it at longer lead times for mostly cloudy conditions. The full model achieves higher FSS than the no-CFM variant across nearly all lead times and categories, with the diference increasing most clearly with lead time under clear-sky conditions. These comparisons indicate that the CFM stage accounts for most of CloudCast v2’s long-lead FSS advantage over CloudCast v1.

## 3.4 Skill during large cloud-cover changes

Figure 8 shows forecast skill on the top-quintile dynamic subset defined in Section 2.6, using Lagrangian persistence as the reference.

![](images/db54e87e197ece7ff069e4a3c666b6712db39f1c3ef0c5c11dc10bd1869c3ea8.jpg)

![](images/d2495f4c528a74239da64a1700e230d12c5cb29e547d266a66e7133ba66254b7.jpg)  
Figure 8: Skill on the top-quintile dynamic cases, evaluated with MSESS relative to Lagrangian persistence. (a) Mean MSESS as a function of forecast lead time. (b) Per-case MSESS at t + 12 for CloudCast v2 and CloudCast v1, with points colored by season. Points above the diagonal indicate higher skill for CloudCast v2.

In the lead-time comparison (panel a), both CloudCast v2 variants add skill beyond advecting the initial observed cloud field throughout the 12-hour forecast window. The no-CFM variant has the highest MSESS at every lead time, while the full model remains clearly above CloudCast v1. The relative skill of both variants increases with lead time as Lagrangian persistence becomes a weaker forecast. CloudCast v1 remains skillful but nearly constant, whereas Eulerian persistence performs worse than Lagrangian persistence throughout the forecast window.

In the case-by-case comparison at t + 12 (panel b), CloudCast v2 outperforms CloudCast v1 for most high-change cases, with most points lying above the diagonal in every season. CloudCast v2 also retains positive MSESS in nearly all cases, including many in which CloudCast v1 has little or negative skill. Because t + 12 lies beyond CloudCast v1’s five-hour training horizon, we also performed the comparison at t + 5. CloudCast v2 achieved higher MSESS in 96.3% of the dynamic cases (not shown), indicating that its advantage is not confined to lead times outside the CloudCast v1 training range.

## 3.5 Overall model performance

The verification metrics capture diferent aspects of forecast quality, and no single model performs best across all scores. Table 4 summarizes the quantitative results.

The ablation comparison isolates the efect of CFM. The no-CFM model gives the lowest MAE and highest ACC and MSESS, whereas the full model gives higher FSS across all cloudiness categories. Thus, CFM slightly reduces performance according to conventional error-based metrics while improving neighborhood-scale spatial agreement.

Relative to CloudCast v1, both CloudCast v2 variants become more advantageous as lead time increases and perform better in cases with large observed cloud-cover changes. CloudCast v1 remains competitive at the earliest lead times, but its skill deteriorates more rapidly. CloudCast v2 therefore provides more sustained skill across the 12-hour forecast window, including when the observed cloud field changes substantially.

Table 4: Summary of verification metrics on the 2025 test set. MAE, bias, and FSS are averaged over $t + 1 { - } t + 1 2$ , and FSS is additionally averaged over neighborhood scales from 25 to 110 km. Dynamic MSESS is computed on the top-quintile dynamic subset relative to Lagrangian persistence. Arrows indicate whether lower or higher values are better; for signed bias, values closest to zero are best. Bold values mark the best-performing forecast system or systems in each column.
<table><tr><td></td><td colspan="2">Pixel-wise</td><td colspan="2">ACC</td><td colspan="2">MSESS</td><td colspan="2">Dynamic MSESS</td><td colspan="4">FSS</td></tr><tr><td>Model</td><td>MAE↓</td><td>Bias</td><td>t+1↑</td><td>t+12↑</td><td>t+1↑</td><td>t+12↑</td><td>t+4↑</td><td>t+12↑</td><td>Clear↑</td><td>Partly↑</td><td>Mostly↑</td><td>Overc.↑</td></tr><tr><td>CloudCast v2</td><td>0.213</td><td>+0.001</td><td>0.686</td><td>0.483</td><td>0.335</td><td>0.479</td><td>+0.336</td><td>+0.471</td><td>0.668</td><td>0.325</td><td>0.500</td><td>0.785</td></tr><tr><td>CloudCast v2 (no CFM)</td><td>0.203</td><td>+0.002</td><td>0.743</td><td>0.589</td><td>0.462</td><td>0.600</td><td>+0.485</td><td>+0.588</td><td>0.520</td><td>0.270</td><td>0.416</td><td>0.708</td></tr><tr><td>CloudCast v1</td><td>0.238</td><td>+0.005</td><td>0.710</td><td>0.276</td><td>0.398</td><td>0.260</td><td>+0.202</td><td>+0.205</td><td>0.623</td><td>0.323</td><td>0.452</td><td>0.760</td></tr><tr><td>MEPS (NWP)</td><td></td><td></td><td>0.469</td><td>0.461</td><td>-0.282</td><td>0.368</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 3.6 Illustrative case studies

In addition to the quantitative evaluation, we examine two forecast cases to make the main model diferences visible. The first case illustrates how CloudCast v2 captures regional cloud-cover evolution, and how CFM can sharpen cloud-edge structure. The second case shows a failure mode in which the same sharpening produces a confident false cloud structure.

Figure 9 shows an extensive cloud field over western and northern Scandinavia at initialization, with comparatively clear conditions over Finland and the Baltic region. Over the following 12 hours, the domain-mean cloud cover changes little, but this masks opposing regional changes: clearing over western Scandinavia and increasing cloud cover near the western Finnish coast. CloudCast v1 fails to capture both signals, retaining too much cloud over western Scandinavia and missing the cloud increase near Finland. Both CloudCast v2 variants capture the broader evolution, but the no-CFM version produces smoother cloud edges and underestimates the spatial extent of the clearing. The full CFM model better preserves the sharp cloud-edge structure in both regions, even when pixel-wise errors are similar.

Figure 10 begins with fragmented cloud cover over western Scandinavia and along the southern part of the domain, while much of Finland and the Baltic region is comparatively clear. Over the following 12 hours, the observed cloud cover expands across the southern and eastern parts of the domain. CloudCast v2 anticipates this development too early, producing a coherent cloud mass at t + 4 h and t + 8 h that is not yet present in the NWCSAF observations. By t + 12 h, the forecast more closely resembles the observed cloudier pattern, suggesting that the error is partly a timing error rather than a wholly spurious structure. The no-CFM version avoids the early false positive but remains smoother, while CloudCast v1 misses much of the subsequent cloud development over the Baltic countries. This case shows that the sharper structures introduced by CFM can occur at the wrong time.

t+0 h  
t+4 h  
t+8 h  
t+12 h  
![](images/69cc89e14d55005a31073d0b237be433c2127a765108f59c1d8fb96c7dadc219.jpg)  
Figure 9: Illustrative cloud-cover case initialized on 1 July 2025 at 03 UTC. Columns show the observed initial state and forecast lead times t + 4, t + 8, and t + 12 h; rows show NWCSAF observations and forecasts from CloudCast v2, CloudCast v2 without CFM, and CloudCast v1. Forecast rows begin at t + 4 because all models share the observed t + 0 initial field. The case has a near-neutral domain-mean cloud-cover change, but opposing regional signals: clearing over western Scandinavia and increasing cloud cover near the western Finnish coast.

NWCSAF (observed)  
t+0 h  
t+4 h  
t+8 h  
t+12 h  
![](images/0af7c724677a279013d347119f99052427321a286edda7dd5354648bd4205fe7.jpg)  
Figure 10: Illustrative winter cloud-cover case initialized on 18 January 2025 at 21 UTC, showing winter cloud-cover evolution over Scandinavia. Columns show the observed initial state and forecast lead times $t + 4 \mathrm { h } , t + 8 \mathrm { h }$ , and t + 12 h; forecast rows are shown only for the lead times because the models use the observed t + 0 cloud field as the initial state. Rows compare NWCSAF observations with CloudCast v2, CloudCast v2 without CFM, and CloudCast v1. The case shows a premature cloud structure produced by the CFM model over the southern part of the domain.

## 4 Discussion and Conclusions

In this study, we develop CloudCast v2, a machine-learning model that produces cloud-cover forecasts up to 12 hours from satellite-derived initial conditions and NWP atmospheric forcings. The model first learns cloud-evolution dynamics from a long reanalysis record and is then adapted to satellite-derived efective cloudiness. CloudCast v1, the earlier U-Net-based model, remains competitive at the earliest lead times. CloudCast v2, however, maintains higher skill later in the forecast window, achieves greater neighborhoodscale spatial agreement, and performs better in cases with large observed cloud-cover changes.

CloudCast v2 connects observation-based nowcasting with NWP-informed forecasting. Satellite fields anchor the forecast to the observed cloud distribution at initialization, while reanalysis pre-training provides a long record from which to learn cloud evolution. NWP atmospheric forcings then describe how the surrounding atmospheric conditions develop over the forecast period. This combination allows the cloud field to evolve beyond persistence or advection without losing its connection to the observed initial state. Its benefit becomes most apparent after the first few hours, as the skill of persistence and CloudCast v1 declines.

The benefit of CFM is not uniform across all metrics. Deterministic regression models tend toward conditional-mean forecasts, which are favored by pixel-wise losses such as MSE. The flow-matching stage produces sharper and more spatially coherent cloud structures, as reflected in its higher FSS, but it can also increase local errors when sharpened features are displaced or timed incorrectly. This trade-of is important operationally: sharper forecasts may better represent cloud-field morphology, but they can also produce confident local false positives or false negatives.

The dynamic-case evaluation provides a complementary view of forecast skill. In the first forecast hours, much of the skill comes from starting with the observed cloud field, but longer lead times require the model to predict changes that are not explained by horizontal transport alone. We therefore compared the models against a Lagrangian persistence baseline that advects the initial cloud field using optical flow. Positive skill relative to this baseline shows that CloudCast v2 improves on simply displacing the initial cloud field. This does not prove that the model explicitly represents cloud formation or dissipation; rather, it indicates skill in cases where formation, dissipation, or deformation likely contribute to the observed cloud-cover change.

The comparison with MEPS, used here as the NWP baseline, requires caution: MEPS forecasts cloud area fraction, whereas CloudCast v2 forecasts NWCSAF efective cloudiness, which also serves as the verification field. ACC reduces sensitivity to the resulting systematic ofset by evaluating anomalies after separate month-and-hour climatologies are removed from each system and the observations. CloudCast v2 maintains higher ACC than MEPS throughout the 12-hour forecast window, indicating closer agreement with the observed large-scale anomaly pattern. MSESS also favors CloudCast v2, but because it is computed against NWCSAF efective cloudiness, the comparison remains afected by the diference between the predicted cloud quantities. MEPS nevertheless becomes more competitive as lead time increases, consistent with the diminishing value of the observed initial cloud field.

MEPS also has an operational role in CloudCast v2 by supplying its atmospheric forcings. The latest NWCSAF fields provide the initial cloud state, and once both inputs are available, CloudCast v2 generates the complete 12-hour forecast in approximately 90 seconds on an NVIDIA H100 GPU. This short inference time adds little computational delay before forecast dissemination compared with producing the underlying NWP forecast. CloudCast v2 cannot be issued before the required MEPS fields become available, however, and therefore does not replace the NWP production cycle. Instead, it rapidly combines available NWP guidance with the latest satellite-observed cloud field.

Several limitations should be considered when interpreting these results. First, the verification uses NWCSAF efective cloudiness as the reference field. This provides a consistent observation-space target, but it does not capture all aspects of cloud cover that may matter for downstream applications such as radiation or solar-power forecasting. Second, the model is trained and evaluated on a single regional domain, and the evaluation covers only one held-out year. The model therefore cannot be assumed to transfer directly to other regions without domain-specific adaptation; similarly, the test set may not represent the full range of weather regimes encountered in other years.

Future work should extend both the forecasting framework and the evaluation. Because the CFM formulation is stochastic, it could be developed into probabilistic or ensemble forecasts that quantify uncertainty in cloud evolution. Evaluation over additional years and geographic domains is also needed to assess the model’s robustness and transferability. The direct-prediction setup could also support sub-hourly forecasts, provided that suitable forcing fields are available at the target times. Finally, the atmospheric forcing set was not extensively optimized in this study; a broader or more carefully selected set of predictors may further improve forecast quality. Together, these directions would test whether the same observation-initialized framework can support more flexible and user-relevant cloud forecasts.

The results indicate that machine-learning models initialized from satellite observations can extend cloud-cover prediction beyond short-range extrapolation when supplied with NWP atmospheric fields. CloudCast v2 complements rather than replaces NWP by combining its atmospheric forecasts with the latest observed cloud field at a small additional computational cost.

## Acknowledgements

We acknowledge CSC – IT Center for Science, Finland for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through the Finnish LUMI Extreme Scale call.

We acknowledge the computational resources provided by the Aalto Science-IT project.

## Additional information

Competing interests: The authors declare no competing interests.

## Data Availability Statement

The source code used in this study is available at https://github.com/fmidev/cloudcast. CERRA data are publicly available from the Copernicus Climate Data Store at https://doi.org/10.24381/ cds.622a565a. MEPS forecast data are openly available through the MET Norway THREDDS archive at https://thredds.met.no/thredds/catalog/metno.html. The NWCSAF efective-cloudiness data and the processed training and evaluation datasets are not publicly archived because of their volume and associated storage and distribution requirements.

## References

Aminou, D. M. A., Jacquet, B., and Pasternak, F. (1997). Characteristics of the meteosat second generation (msg) radiometer/imager: Seviri. In Sensors, Systems, and Next-Generation Satellites, volume 3221.

Ayzel, G., Schefer, T., and Heistermann, M. (2020). Rainnet v1.0: A convolutional neural network for radar-based precipitation nowcasting. Geoscientific Model Development, 13.

Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. Technical report.

Bengtsson, L., Andrae, U., Aspelien, T., Batrak, Y., Calvo, J., de Rooy, W., Gleeson, E., Hansen-Sass, B., Homleid, M., Hortal, M., Ivarsson, K. I., Lenderink, G., Niemelä, S., Nielsen, K. P., Onvlee, J., Rontu, L., Samuelsson, P., Muñoz, D. S., Subias, A., Tijm, S., Toll, V., Yang, X., and Ødegaard Køltzow, M. (2017). The harmonie-arome model configuration in the aladin-hirlam nwp system. Monthly Weather Review, 145.

Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., and Tian, Q. (2023). Accurate medium-range global weather forecasting with 3d neural networks. Nature, 619:533–538.

Chase, R. J., Haynes, K., Hoef, L. V., and Ebert-Uphof, I. (2026). How to use score-based difusion in earth system science: A satellite nowcasting example. Artificial Intelligence for the Earth Systems, 5(3).

Chen, H., Zhong, X., Zhai, Q., Li, X., Chan, Y. W., Chan, P. W., Yang, M., Huang, Y., Li, H., and Shi, X. (2026). Skillful short-term forecasting of clouds with a cascade difusion model. Journal of Geophysical Research: Machine Learning and Computation, 3.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale.

Eresmaa, R., Andrae, U., Berggren, L., Bremnes, J. B., Fortelius, C., Frogner, I.-L., Gregow, E., Grote, R., Ridal, M., Vignes, O., Ansper, I., Azad, R., Begens, J., Bergholt, L., Blyverket, J., Daniel, L., Engdahl, B. J., Erlandsen, H., Eshagh, M., Hasu, M., Homleid, M., Ideström, P., Ivarsson, K.-I., Lerner-Vilu, A., Mets, A., Moldengauere, S., Noer, G., Nurmi, E., Partio, M., Pried¯ıtis, G., Saarikalle, E., Schönach, D., Shapkalijevski, M., Sild, K., Sitčihins, V., Sokka, N., Spjelkavik, S., Tack, A., Vaštšenko, A., Wettergren, A., Ylinen, K., and Zandovskis, U. (2026). The operational forecast process at metcoop. Bulletin of the American Meteorological Society.

Espeholt, L., Agrawal, S., Sønderby, C., Kumar, M., Heek, J., Bromberg, C., Gazen, C., Carver, R., Andrychowicz, M., Hickey, J., Bell, A., and Kalchbrenner, N. (2022). Deep learning for twelve hour precipitation forecasts. Nature Communications, 13.

Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion. In Image Analysis: 13th Scandinavian Conference, SCIA 2003, volume 2749, pages 363–370. Springer.

Ha, J. H. and Lee, H. (2024). A deep learning model for precipitation nowcasting using multiple optical flow algorithms. Weather and Forecasting, 39:41–53.

Hieta, L., Partio, M., Laine, M., Tuomola, M. L., Hohti, H., Perttula, T., Gregow, E., and Ylhäisi, J. S. (2021). Smartmet nowcast – rapidly updating nowcasting system at finnish meteorological institute. Meteorologische Zeitschrift, 30.

Hughes, N. M., Sanchez, A., Berry, Z. C., and Smith, W. K. (2024). Clouds and plant ecophysiology: missing links for understanding climate change impacts. Frontiers in Forests and Global Change, 7.

Kellerhals, S. A., Leeuw, F. D., and Rivero, C. R. (2022). Cloud nowcasting with structure-preserving convolutional gated recurrent units. Atmosphere, 13.

Kerdraon, G. and Fontaine, E. (2021). Algorithm theoretical basis document for the cloud product processors of the nwc/geo. Technical report, EUMETSAT NWC SAF.

Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. (2023). Graphcast: Learning skillful medium-range global weather forecasting. Science, 382(6677):1416–1421.

Lang, S., Alexe, M., Chantry, M., Dramsch, J., Pinault, F., Raoult, B., Clare, M. C. A., Lessig, C., Maier-Gerber, M., Magnusson, L., Bouallègue, Z. B., Nemesio, A. P., Dueben, P. D., Brown, A., Pappenberger, F., and Rabier, F. (2024). Aifs – ecmwf’s data-driven forecasting system.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). Flow matching for generative modeling.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows . In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9992–10002, Los Alamitos, CA, USA. IEEE Computer Society.

Miralles, O., Nerini, D., Bhend, J., Raoult, B., and Spirig, C. (2026). Observation-guided interpolation using graph neural networks for high-resolution operational nowcasting in switzerland. Artificial Intelligence for the Earth Systems, 5.

Murphy, A. and Epstein, E. (1989). Skill scores and correlation coeficients in model verification. Monthly Weather Review, 117:572–581.

Nipen, T. N., Haugen, H. H., Ingstad, M. S., Nordhagen, E. M., Salihi, A. F. S., Tedesco, P., Seierstad, I. A., Kristiansen, J., Lang, S., Alexe, M., Dramsch, J., Raoult, B., Mertes, G., and Chantry, M. (2026). Regional data-driven weather modeling with a global stretched grid. Artificial Intelligence for the Earth Systems, 5(2):250001.

Oskarsson, J., Landelius, T., and Lindsten, F. (2023). Graph-based neural weather prediction for limited area modeling.

Partio, M., Hieta, L., and Kokkonen, A. (2025). Cloudcast—total cloud cover nowcasting with machine learning. Artificial Intelligence for the Earth Systems, 4(3):e240104.

Prudden, R., Adams, S., Kangin, D., Robinson, N., Ravuri, S., Mohamed, S., and Arribas, A. (2020). A review of radar-based nowcasting of precipitation and applicable machine learning techniques.

Ravuri, S., Lenc, K., Willson, M., Kangin, D., Lam, R., Mirowski, P., Fitzsimons, M., Athanassiadou, M., Kashem, S., Madge, S., Prudden, R., Mandhane, A., Clark, A., Brock, A., Simonyan, K., Hadsell, R., Robinson, N., Clancy, E., Arribas, A., and Mohamed, S. (2021). Skilful precipitation nowcasting using deep generative models of radar. Nature, 597:672–677.

Ribeiro, B. P. and Pucer, J. F. (2026). Flowcast: Advancing precipitation nowcasting with conditional flow matching. In International Conference on Learning Representations, pages 156863–156883.

Ridal, M., Bazile, E., Moigne, P. L., Randriamampianina, R., Schimanke, S., Andrae, U., Berggren, L., Brousseau, P., Dahlgren, P., Edvinsson, L., El-Said, A., Glinton, M., Hagelin, S., Hopsch, S., Isaksson, L., Medeiros, P., Olsson, E., Unden, P., and Wang, Z. Q. (2024). Cerra, the copernicus european regional reanalysis system. Quarterly Journal of the Royal Meteorological Society, 150:3385–3411.

Ritvanen, J., Harnist, B., Aldana, M., Makinen, T., and Pulkkinen, S. (2023). Advection-free convolutional neural network for convective rainfall nowcasting. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 16.

Roberts, N. M. and Lean, H. W. (2008). Scale-selective verification of rainfall accumulations from highresolution forecasts of convective events. Monthly Weather Review, 136.

Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., and Woo, W. C. (2015). Convolutional lstm network: A machine learning approach for precipitation nowcasting. In Advances in Neural Information Processing Systems, volume 2015-January.

Sun, J., Xue, M., Wilson, J. W., Zawadzki, I., Ballard, S. P., Onvlee-Hooimeyer, J., Joe, P., Barker, D. M., Li, P. W., Golding, B., Xu, M., and Pinto, J. (2014). Use of nwp for nowcasting convective precipitation: Recent progress and challenges. Bulletin of the American Meteorological Society, 95.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 6000–6010, Red Hook, NY, USA. Curran Associates Inc.

Wang, Y., Millstein, D., Mills, A. D., Jeong, S., and Ancell, A. (2022). The cost of day-ahead solar forecasting errors in the united states. Solar Energy, 231:846–856.

Xia, P., Zhang, L., Min, M., Li, J., Wang, Y., Yu, Y., and Jia, S. (2024). Accurate nowcasting of cloud cover at solar photovoltaic plants using geostationary satellite images. Nature Communications, 15.