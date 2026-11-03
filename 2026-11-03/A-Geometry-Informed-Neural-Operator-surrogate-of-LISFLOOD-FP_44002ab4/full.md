# A Geometry-Informed Neural Operator surrogate of LISFLOOD-FP for rapid pluvial–fluvial flood forecasting: the case of Forlì, Italy

Samuele Maria Garofalo<sup>a,∗</sup>, Fabio Limardo<sup>a</sup>, Simonetta Bodojra<sup>a</sup>, Mohammad Amin Nabian<sup>b</sup>, Mehdi Taghizadeh<sup>c</sup>

<sup>a</sup>Data Reply, Turin, Italy

<sup>b</sup>NVIDIA, Santa Clara, CA, USA

<sup>c</sup>University of Virginia, USA

## Abstract

Physically based two-dimensional hydraulic models are too slow for real-time inundation forecasting and large ensembles. We<sup>6</sup> present a time-dependent geometry-informed neural operator (GINO) surrogate of LISFLOOD-FP for the municipality of Forlì (Italy), severely afected by the May 2023 Emilia-Romagna floods. The surrogate is designed for an operational setting in which only river stage gauges and rain gauges are available. A reproducible pipeline turns public regional observations into a physically based training set: each river inflow is driven by the first stage gauge downstream of it through a calibrated water-surface boundar condition, and rain enters as SCS-CN efective rainfall. The surrogate is an autoregressive GINO trained on 216 simulations of 24 observed events with scaled forcing. It predicts volume-conserving polygon-mean depths on an exposure-based Voronoi mesh of 659 nodes, with a 20 min step over 72 h. The surrogate is tested on 38 independent observed events that it never saw during training.<sup>2</sup> On these events it reaches a critical success index (CSI) of 0.934 at 0 05 m and 0.844 at 0 3 m, a depth RMSE of 0 008 m and an R<sup>2</sup> of 0.984 for maximum-depth maps. On one NVIDIA A10G GPU a 72-h forecast takes 1.7 s instead of 55 s with LISFLOOD-FP, and a 50-member ensemble, computed in a single batched rollout, takes 6.2 s instead of about 46 min, which makes probabilistic flood forecasting practical. These results show that neural-operator surrogates can deliver accurate pluvial–fluvial flood forecasts. driven by river stage for operational early warning.<sub>c</sub>

Keywords: Flood forecasting, Geometry-informed neural operator, Surrogate modelling, LISFLOOD-FP, Pluvial–fluvial flooding, Ensemble forecasting

## 1. Introduction

Floods are among the most damaging natural hazards in Europe. In Italy, Emilia-Romagna is one of the most exposed re-<sup>.</sup> gions. In May 2023 two sequences of persistent, intense rainfall struck the Romagna foothills within a few weeks. Many rivers overtopped or breached their embankments and flooded towns and farmland across the plain (Dorrington et al., 2024). Forlì, crossed by the Montone and Ronco rivers and located just downstream of their Apennine catchments, was among the most afected cities. Events of this kind combine fluvial flood-<sub>a</sub> ing from the rivers with pluvial flooding from local rain on the urban area and the plain. They develop over one to three days, and their timing and extent depend on how river stage and local rain evolve together. Because both the meteorological forcing and the river response are uncertain, a single deterministic run is not enough: an ensemble of plausible scenarios is needed.

Physically based two-dimensional hydraulic models answer these questions by solving the shallow water equations or simplified forms of them. LISFLOOD-FP, with its local-inertial formulation (Bates and De Roo, 2000; Bates et al., 2010) and GPU solver (Shaw et al., 2021), is one of the most eficient such codes. Even so, running a 72-h simulation over a whole municipality for every forecast update, and for every member of an ensemble, is still expensive. For this reason, machinelearning (ML) surrogates that learn the response of a hydraulic model from a library of its simulations, and then reproduce it at a small fraction of the cost, have become an active area of research (Bentivoglio et al., 2022; Mosavi et al., 2018).

Early surrogates relied on convolutional neural networks (CNNs) operating on regular rasters (Kabir et al., 2020; Guo et al., 2021; Löwe et al., 2021). CNNs are efective on grids, but they are tied to a fixed structured discretisation and cannot easily represent irregular domains or meshes. Graph neural networks (GNNs) remove this constraint by passing messages between the nodes of an unstructured mesh (Pfaf et al., 2021). They have been used for spatio-temporal flood modelling with hydraulics-based architectures (Bentivoglio et al., 2023) and, in HydroGraphNet, with a physics-informed loss and the pushforward trick (Brandstetter et al., 2022) for stable autoregressive forecasts of pluvial and fluvial events (Taghizadeh et al., 2025). However, local message passing needs many layers to propagate information across large domains. Neural operators address this by learning maps between function spaces (Li et al., 2020; Kovachki et al., 2023). The Fourier neural operator (FNO) captures long-range interactions eficiently on regular grids (Li et al., 2021), and FNO-type models have been applied to flood prediction (Kumar et al., 2024; Xu et al., 2025). The geometryinformed neural operator (GINO) combines a graph neural operator encoder and decoder, which handle arbitrary point clouds or meshes, with an FNO on a regular latent grid (Li et al., 2023). FloodForecaster (Taghizadeh et al., 2026) introduced the first time-dependent, autoregressive GINO for flood simulation. Operational flood forecasting in a city such as Forlì (228 36 km<sup>2</sup>) raises further questions:

• Multiple rivers. The presence of several rivers (Montone, Rabbi and Bidente) makes modeling a surrogate more complex.

• No discharge data. The regional network measures river stage and rain in near real time. Discharge would require rating curves, which are uncertain and often unavailable at the gauges that matter. A surrogate for operational use should be driven by the quantities that are actually observed.

• Mixed pluvial–fluvial forcing. River inflows and local rain act together and are correlated in real events. The surrogate must respond to both.

• Product-oriented meshes. Risk products are often delivered on administrative or exposure-based units rather than on the hydraulic grid. A surrogate that predicts directly on these units must preserve the flood volume when it aggregates the simulator output.

In this paper we adopt the time-dependent GINO model family implemented with the neuraloperator library (Kossaifi et al., 2024). We extend them to an operational setting driven only by stage and rain gauges and test the surrogate on observed events that it has never seen. The main contributions are:

1. A complete, reproducible pipeline from public regional observations (ARPAE stage and rain gauges, Arpae Emilia-Romagna – Servizio IdroMeteoClima (SIMC); no discharge data) to a physically based LISFLOOD-FP training set. The pipeline includes a first-downstream-gauge rule for driving each river inflow and a library-anchored calibration of the stage boundary conditions (Sections 2.2– 3.3).

2. A time-dependent GINO surrogate for mixed pluvial– fluvial forcing over a whole municipality. It is trained on 24 observed events with scaled forcing. It reaches CSI of 0 934 at 0 05 m, $R ^ { 2 }$ of the maximum depth of 0 984 on 38 independent observed events never seen in training (Section 4.2).

3. Volume-conserving polygon-mean targets on a productoriented, exposure-based Voronoi mesh of 659 nodes, which keep the product’s spatial units and leave the inference cost unchanged, combined with a loss weight on predicted wet nodes and a long push-forward curriculum for stable 72 h rollouts (Sections 3.4 and 3.6).

4. A check of the simulator’s credibility: the LISFLOOD-FP set-up is compared with the oficial perimeter of the areas flooded in May 2023 (Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025) and reaches a CSI of 0.6 (Section 4.1).

5. An evaluation that complements depth and extent scores with node-level flood metrics on the anomaly above each node’s base level, and a 50-member ensemble for probabilistic forecasting (Sections 4.3 and 4.5).

The paper is organised as follows. Section 2 describes the study area, the observations and the LISFLOOD-FP set-up. Section 3 presents the event dataset, the mesh, the GINO surrogate, its training and the evaluation metrics. Section 4 reports the results, from the check of the simulator to the ensemble forecasts. Section 5 discusses their implications and limitations, and Section 6 concludes.

## 2. Study area and data

## 2.1. Study area

The study area is the municipality of Forlì (Emilia-Romagna, northern Italy), located where the rivers draining the Romagna Apennines enter the Po plain (Fig. 1). Three rivers enter the simulation domain from the south-west and south: the Montone, its tributary the Rabbi, and the Bidente, which is named Ronco downstream of Meldola, about 2 6 km inside the domain, and flows east of the city. Downstream of the Apennine foothills the rivers are confined between levees whose crests stand several metres above the surrounding plain, so that flooding is controlled both by the river stage and by local rainfall that cannot drain to the embanked channels. The surrogate is built for the whole municipality, an area of 228 36 km<sup>2</sup> discretised by the mesh of Section 3.4; the LISFLOOD-FP domain extends beyond it to include the river inflows and the outflow edges.

Forlì was among the municipalities most severely hit by the floods of May 2023 in Emilia-Romagna (Dorrington et al., 2024; Arrighi and Domeneghetti, 2024). On 16–17 May 2023, after a first event at the beginning of the month, persistent rainfall over already wet soils led to river overtopping and levee breaches and to the inundation of urban districts and of large parts of the plain around the city. This event provides the observed flood extent used to check the simulator (Section 3.9) and is one of the base events of the training library (Section 3.3).

## 2.2. Observations and geospatial data

Hydro-meteorological observations. All forcing data are public observations of the regional environmental agency ARPAE, retrieved from its Dext3r/open-data archive (Arpae Emilia-Romagna

– Servizio IdroMeteoClima (SIMC)): river stage (metres above the gauge zero) and rainfall, for the years 2006–2026. No discharge measurements or rating curves are used. Each river inflow is driven by the stage of the first ARPAE gauge downstream of the inflow, before any confluence: Castrocaro on the Montone, Ponte Calanca on the Rabbi and Meldola on the Bidente– Ronco. For each driver gauge ARPAE defines three alert thresholds $H _ { 1 } < H _ { 2 } < H _ { 3 }$ (Castrocaro 1.3/2.0/3 0 m; Ponte Calanca 0.9/1.3/2 0 m; Meldola 0.6/1.1/2 1 m), where $H _ { 3 }$ approximates

![](images/d258135296a73ddaa2fdc678d18dc12f01eab8dd626debf2c1f68292999d896f.jpg)  
Figure 1: Study area: LISFLOOD-FP domain on the conditioned DEM, with the municipality of Forlì (dashed; extent of the GINO mesh, $2 2 8 . 3 6 \mathrm { k m } ^ { 2 } ) .$ , the Montone, Rabbi and Bidente–Ronco rivers, the three river inflow cells (stars), the ARPAE stage gauges (red: driver gauges of the three inflows), the 19 rain gauges inside the rain box (dotted) and, shaded, the areas flooded on 16–17 May 2023 (Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025). Inset: location in Italy, with Emilia-Romagna in blue.

bankfull conditions; they are used to grade event severity and to normalise the stage rise (Section 3.1). Two upstream gauges, Rocca San Casciano (Montone) and S. Zeno (Rabbi), serve only as substitutes to fill the shape of long gaps in the driver records. Rainfall is taken from all ARPAE rain gauges within the box 44 08<sup>◦</sup>–44 37<sup>◦</sup> N, 11 88<sup>◦</sup>–12 30<sup>◦</sup> E and averaged into a domainmean hourly series. Further stage gauges inside the domain (Forlì and Ponte Braldo on the Montone, Ronco and Coccolia on the Ronco) are not used as forcing.

Terrain, river network and land cover. The 100 m model grid is defined on the national bare-earth DTM TINITALY (10 m, version 1.1) (Tarquini et al., 2007, 2023), which also fills the cells without LiDAR coverage and provides the upstream drainage areas. Elevations and channel geometry are derived from the Li-DAR DTM 2023/24 of Regione Emilia-Romagna (0 5 m cells, nominal vertical accuracy 8 cm, accessed at 2 m) (Regione Emilia-Romagna, 2024). River centrelines are taken from the regional topographic database (DBTR) (Regione Emilia-Romagna). Land cover is the ESA WorldCover 2021 map at 10 m (Zanaga et al., 2022), used both for Manning’s roughness and for the SCS curve numbers. All fine-resolution products are used only to derive 100 m values (Section 3.1).

Observed flood extent. To check the simulator against reality we use the regional perimeter of the areas flooded on 16–17 May 2023 (Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025), produced by the regional agency for territorial safety and civil protection.

It was obtained by photo-interpretation of several sources, including contributions of local authorities and land-reclamation consortia, at a nominal scale of 1:5000, and maps the flooding caused by river breaches and overtopping.

## 3. Methodology

The methodology chains four components (Fig. 2): (i) a LISFLOOD-FP set-up of the domain driven only by observed river stage and rain (Section 3.1); (ii) a library of historical events and a training set built scaling known hydrographs and hyetograph, with an independent set of observed test events (Sections 3.2–3.3); (iii) a product-oriented Voronoi mesh with volume-conserving node targets (Section 3.4); and (iv) a time dependent GINO trained to emulate the simulator autoregressively (Sections 3.5–3.6). The surrogate is evaluated with depth, extent and node-level flood metrics (Section 3.7) and in ensemble mode (Section 3.8); the simulator itself is checked against the observed May 2023 flood (Section 3.9).

## 3.1. LISFLOOD-FP set-up

Solver and grid. The training data are generated with LISFLOOD-FP 8.2 (Bates and De Roo, 2000; Shaw et al., 2021), using its GPU implementation of the local-inertial (acceleration) formulation of the shallow-water equations (Bates et al., 2010) on a regular 100 m grid. The grid covers the bounding box of the municipality plus a 5 km bufer: 329 × 328 cells (107 912 cells, about 32 9 km × 32 8 km, $1 0 7 9 \mathrm { k m } ^ { 2 } )$ , of which the municipality covers $2 2 8 \mathrm { k m } ^ { 2 }$ . The resolution was deliberately kept at 100 m so that several hundred event simulations remain afordable. The GPU solver has neither subgrid channels nor infiltration, which motivates the channel conditioning and the efective-rain formulation described below. Water depth is saved every 20 min over a 72 h horizon, giving 217 depth frames $( t = 0 , \ldots , 2 1 6 )$ per run.

DEM conditioning. Cell elevations are the means of the Li-DAR DTM over each 100 m cell, with the national DTM where LiDAR is missing, followed by a D8 priority-flood depression filling. Because rivers exist in the model only as DEM cells, the channel and levees are imposed explicitly: for every channel cell of the DBTR network, three LiDAR cross-sections perpendicular to the river give the bed, the bank tops (levee crests where a narrow ridge stands above the land behind) and the channel width W. The channel cell is set to the lowest bank top minus an equivalent depth $d _ { \mathrm { e q } } = ( z _ { \mathrm { b a n k } } - z _ { \mathrm { b e d } } )$ · min(1 $W / \Delta x ) ^ { 0 . 6 }$ $\Delta x = 1 0 0 \mathrm { m }$ , which gives the 100 m-wide cell the same Manning conveyance as the real W-wide channel; cells flanking a levee are raised to its crest so that levees keep acting as barriers. Since LISFLOOD-FP exchanges water only across cell faces, a D4 conditioning step finally carves a face-connected path from each inflow to the domain edge along the channel, opens diagonal corners and fills residual pits, while protecting channel and levee cells. One inflow cell per river is placed on the valley bottom where the river enters the domain; all other edges are free outflow boundaries.

![](images/a61917a61deb952cf1425d2176ee3ae78f40c7d702ed8843af88630343eaddcd.jpg)  
Figure 2: Overview of the workflow, illustrated with the 16 May 2023 event. (a) Observed stage at the three driver gauges and domain-mean rain (bars, inverted axis), which define the stage boundary conditions and the efective rain. (b) LISFLOOD-FP maximum water depth on the 100 m grid. (c) Polygon-mean maximum depth on the 659-node Voronoi mesh, the training target of the surrogate. (d) Time-dependent GINO, applied autoregressively. (e) Ensemble forcing: 50 perturbed Montone stage series (grey) around the observed one (black), propagated by the surrogate in a single batched rollout.

Roughness. Manning’s n is assigned from the ESA WorldCover classes (Zanaga et al., 2022) resampled to the grid: tree cover 0.07, shrubland 0.06, grassland 0.035, cropland 0.050, builtup 0.100, bare/sparse vegetation 0.030, snow and ice 0.035, permanent water 0.025, herbaceous wetland 0.040, mangroves 0.080, moss and lichen 0.070 (unclassified cells 0.035).

Stage boundary conditions (HVAR). Without discharge data, each inflow cell receives a time-varying water-surface elevation (HVAR, Dirichlet) derived from the stage H(t) of its driver gauge (Section 2.2):

$$
\begin{array} { l } { \displaystyle \mathbf { W S E } ( t ) = z + f \left[ h _ { 0 } + s d _ { \mathrm { b f } } \frac { H ( t ) - H _ { b } } { H _ { 3 } - H _ { b } } \right] , } \\ { \displaystyle f = \left( \frac { W } { 1 0 0 \mathrm { m } } \right) ^ { 0 . 6 } , } \end{array}\tag{1}
$$

bounded below by z, where z is the elevation of the inflow cell, $h _ { 0 } = 0 . 2$ m a low-flow depth, $d _ { \mathrm { b f } }$ and W the bankfull depth and width of the channel at the inflow (from the LiDAR sections), $H _ { 3 }$ the third ARPAE threshold of the gauge (about bankfull), $H _ { b }$ the per-event base level, taken as the 5th percentile of the stage within ±30 days of the event, and s a per-river scale. The stage rise is thus applied as a fraction of bankfull, the scale on which the model channel is built, rather than in raw metres, which depend on each gauge’s cross-section; the width factor f maps the depth of a W-wide channel to the equivalent depth of a 100 m cell. Stage records are despiked, smoothed over 60 min and interpolated to 20 min; each series starts from the spin-up level z + f h<sub>0</sub> and ramps linearly to the event values over the first 3 h.

Calibration. The scale s of each river is calibrated once on four anchor events of the library with diferent severities (16 May 2023, 13 May 2019, 2 February 2018, 20 November 2024). For each anchor, river-only runs (observed stages, no rain) are performed with all scales multiplied by $m \in \{ 0 . 5 , 0 . 7 , 1 . 0 , 1 . 4 , 2 . 0 \}$ ; for each river, the multiplier at which the simulated stage rise at the driver gauge equals the observed rise above $H _ { b }$ is interpolated in log–log space, and the new scale is the previous one times the median over the anchors. This yields s = 1 196 (Montone), 1 04 (Rabbi) and 0 833 (Bidente). The per-anchor multipliers spread by no more than about ±10%, which supports a single stage-to-inflow mapping per river across event magnitudes.

Efective rainfall. As the GPU solver ignores infiltration, the rainfall input is the SCS curve-number runof (USDA Soil Conservation Service, 1986) of each cell. With P(t) the observed domain-mean rain cumulated from the start of the event,

$$
\begin{array} { r l r } & { } & { S = \displaystyle \frac { 2 5 4 0 0 } { \mathrm { C N } } - 2 5 4 , \qquad I _ { a } = 0 . 2 S , } \\ & { } & { Q ( t ) = \displaystyle \frac { ( P ( t ) - I _ { a } ) ^ { 2 } } { P ( t ) - I _ { a } + S } \quad \mathrm { f o r } P ( t ) > I _ { a } , } \end{array}\tag{2}
$$

(S , I<sub>a</sub>, P, Q in mm; $Q = 0$ otherwise), and the efective rain of each 20 min step is the increment of $Q .$ . The curve number (AMC II) is assigned per cell from WorldCover and the TR-55 tables for hydrologic soil group C, in the absence of a soil map, and converted to the event’s antecedent moisture class, determined from the 5-day antecedent rain with the TR-55 sea sonal thresholds. The efective rain is written as a spatially distributed NetCDF forcing; rainfall itself is spatially uniform (domain mean of the gauges).

Spin-up. A single spin-up run of 240 h, without rain and with every inflow held at its low-flow level, establishes base flow in the channels. Its final state is the initial condition of every event simulation, and the same state initialises the surrogate (Section 3.5).

## 3.2. Event library

Historical events are extracted from the ARPAE archives for 2006–2026, restricted to January–May and September–December Events are the peaks of a flood index $I ( t ) = \mathrm { m a x } _ { r } H _ { r } ( t ) / H _ { 1 , r }$ over the three rivers, with $I \geq 1$ and at least 72 h between peaks; severity is 3 if a gauge exceeds its threshold $H _ { 3 } , 2$ if it exceeds $H _ { 2 } ,$ , and 1 otherwise. Base events comprise all severity-3 events plus the most diverse severity-2 and severity-1 events, chosen by greedy farthest-point sampling on the per-river peak $/ H _ { 3 }$ ratios and the logarithm of the event rain, so that diferent dominant rivers and rain amounts are represented. All remaining severity $\geq 2$ events, and a set of severity-1 events spread over the years, form the test set.

## 3.3. Dataset design

For the dataset construction, a limited number of base events is expanded by random scaling, and the resulting simulations are split at random into training and validation sets. Each of the 24 base events is simulated unscaled and in 8 variants in which the river-stage rise above $H _ { b }$ and the rain are multiplied by independent factors drawn from $\mathcal { U } ( 0 . 8 , 1 . 2 )$ . The resulting 216 simulations are split at random into 194 training and 22 validation simulations.

We additionally test on 38 independent observed events, simulated once with their observed forcing, never seen in training either as such or as variants. They span 2006-03-05 to 2026- 02-04 and include the two largest floods of the test set, 2019- 11-17 and 2024-09-18.

## 3.4. Voronoi mesh and volume-conserving targets

The surrogate operates on an unstructured mesh of 659 Voronoi polygons covering the municipality (Fig. 3). The seeds are placed according to exposure, i.e., counts of buildings, transport infrastructure, geological and water features, so that the mesh is finer where assets are concentrated. The same polygons are the aggregation unit of the risk product, so the surrogate predicts directly at the scale at which flood information is consumed.

The target depth of a node is the mean depth over the 100 m LISFLOOD-FP cells whose centres fall inside its polygon, i.e., the polygon water volume divided by its area; polygons smaller than a cell take the cell containing their centroid. This aggregation conserves the simulated flood volume. For consistency, the static features Elevation, Slope, Curvature and Manning are also polygon means of the DEM and roughness map used by the simulations. The aggregation is performed once during data preparation and leaves the inference cost of the surrogate unchanged.

## 3.5. Time-dependent GINO

We adopt the geometry-informed neural operator (Li et al., 2023) as implemented in the neuraloperator library, version 2.0.0 (Kossaifi et al., 2024; NeuralOperator developers), in the time-dependent autoregressive formulation of Taghizadeh et al. (2026) (Fig. 4). A graph-neural-operator (GNO) encoder (Li et al., 2020) maps the node features from the irregular mesh to a regular $3 2 \times 3 2$ latent grid, by kernel integration over all mesh nodes within a radius of 0.15 in coordinates normalised to the unit square; a Fourier neural operator (Li et al., 2021; Kovachki et al., 2023) with 16 × 16 retained modes propagates information globally on the latent $\mathrm { g r i d } ;$ and a GNO decoder with the same radius maps the latent field back to the mesh nodes. Other architectural hyper-parameters are the defaults of neuraloperator version 2.0.0 (four FNO layers with 64 hidden channels; Table 2), for a total of about 2.7 million trainable parameters.

![](images/a9cce540aa3d787572315046410088e8f3e0bdf085aea9858f244f5e7a0e2e3d.jpg)  
Figure 3: Exposure-based Voronoi mesh of the municipality of Forlì (659 polygons; median area 25 ha), with the main rivers.

At each step, the model $\mathcal { G } _ { \theta }$ receives the input x<sup>t</sup> (Section 3.5.1) and predicts the normalised depth increment, which is added to the current state:

$$
{ \bf d } ^ { t + 1 } = \mathrm { R e L U } \big ( { \bf d } ^ { t } + \sigma _ { \Delta } \mathcal { G } _ { \theta } ( { \bf x } ^ { t } ) + \mu _ { \Delta } \big ) , \qquad t = 0 , \ldots , 2 1 5 ,\tag{3}
$$

where $\mathbf { d } ^ { t }$ is the vector of node depths, $\mu _ { \Delta }$ and $\sigma _ { \Delta }$ are the mean and standard deviation of the one-step increments over the training set, and the ReLU enforces non-negative depths. The time step is 20 min, and a 72 h forecast requires 216 steps.

## 3.5.1. Input features

Each node carries 12 input channels (Table 1). The six static channels describe the terrain, the roughness, the polygon area and the proximity to the river inflows. The five dynamic channels encode the forcing at the current step: rain enters as the domain-mean efective rain of Eq. (2) together with its cumulative value and a short causal moving mean; the river forcing enters through the HVAR water-surface elevations of the three inflow points, each projected onto its nearest mesh node (anchor) and spread to the other nodes with a Gaussian kernel $( \sigma = 1 0 0 0 \mathrm { m } )$ , the rivers being combined by a kernel-weighted mean. The resulting field is expressed as an anomaly above a per-node reference level, the minimum over the training simulations, so that it is zero at low flow; a 6 h causal moving mean adds memory of the hydrograph. The last channel is the current depth. Static and rain channels are standardised, and the river channels are only rescaled so that zero remains zero; all normalisation statistics are computed on the training simulations.

![](images/a6c96043d8a01c18ab404a8d47da9fbdffa89b56e8d92683d574f326e3d5a240.jpg)  
Figure 4: Architecture of the time-dependent GINO surrogate. The node features on the Voronoi mesh are encoded onto a regular 32 × 32 latent grid by a GNO, processed globally by four Fourier layers and decoded back to the nodes by a second GNO, which predicts the depth increment of Eq. (3); the new depth is fed back as input for the next 20 min step.

Table 1: Input channels of the GINO surrogate (per node and time step).
<table><tr><td>Channel</td><td>Description</td></tr><tr><td>Static</td><td></td></tr><tr><td>Elevation</td><td>Polygon mean of the conditioned DEM (m)</td></tr><tr><td>Slope</td><td>Polygon mean of the terrain slope</td></tr><tr><td>Curvature</td><td>Polygon mean of the terrain curvature</td></tr><tr><td>Manning</td><td>Polygon mean of Manning&#x27;s n</td></tr><tr><td>Area SDF</td><td>Polygon area  $\exp ( - d / 1 0 0 0 \mathrm { m } )$  , d = distance to the nearest in-</td></tr><tr><td>flow anchor node</td><td></td></tr><tr><td colspan="2">Dynamic (forcing)</td></tr><tr><td>Precip</td><td>Domain-mean effective (SCS-CN) rain at t</td></tr><tr><td>Rain_cum</td><td>Effective rain cumulated up to t</td></tr><tr><td>Rain_win</td><td>Causal mean over 6 steps (2 h)</td></tr><tr><td>BC_anom</td><td>Kernel-propagated inflow WSE above the node</td></tr><tr><td>BC_win</td><td>reference level (m) Causal mean of BC_anom over 18 steps (6 h)</td></tr><tr><td colspan="2">State</td></tr><tr><td>Depth</td><td>Node depth at t (teacher-forced in training, fed back at inference)</td></tr></table>

## 3.6. Training

Loss. Training samples are pairs (x<sup>t</sup> ∆<sup>t</sup>) drawn from all steps of the 194 training simulations, with ∆<sup>t</sup> the normalised true increment. Since most nodes are dry most of the time, a plain MSE would be dominated by trivial zeros. We therefore use a

wet-weighted MSE:

$$
\begin{array} { r l r } & { \mathcal { L } = \frac { \sum _ { n } w _ { n } ( \hat { \Delta } _ { n } - \Delta _ { n } ) ^ { 2 } } { \sum _ { n } w _ { n } } , } & \\ & { w _ { n } = 1 + \lambda \mathbb { 1 } \Big [ \operatorname* { m a x } \big ( d _ { n } ^ { t } , d _ { n } ^ { t + 1 } , \hat { d } _ { n } ^ { t + 1 } \big ) > h _ { w } \Big ] , } & \end{array}\tag{4}
$$

with $\lambda = 5$ and $h _ { w } = 0 . 0 5$ m, where d is the true depth and ${ \hat { d } } ^ { t + 1 }$ the depth predicted by Eq. (3). The predicted-wet term gives the same weight to spurious thin water produced by the model as to a missed wet node, which discourages the slow accumulation of false water during long rollouts.

Stabilisation ofthe rollout. Two techniques counter the drift of autoregressive rollouts. First, Gaussian noise with standard deviation 0 1 $\sigma _ { d } ~ ( \sigma _ { d } =$ standard deviation of the training depths) is added to the input depth of one-step samples, with the target increment recomputed from the noisy input. Second, a pushforward curriculum (Brandstetter et al., 2022): with probability 0.3, a batch is unrolled for K steps, the first K−1 without gradient on the model’s own predictions, and the last one is penalised against the true depth at t + K. The unroll length grows with the epochs: K = 1 (epochs 1–20), 2 (21–40), 8 (41–60), 16 (61–80) and 32 (81–100).

Optimisation and model selection. The model is trained for 100 epochs with AdamW (Loshchilov and Hutter, 2019) (learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 } .$ , 5 epochs of linear warm-up followed by cosine annealing), batch size 16 and gradient-norm clipping at 1. Every 10 epochs the model is run in free-running mode over the full 72 h of all validation simulations, starting from the spin-up state, and the checkpoint with the lowest areaweighted mean absolute error on wet nodes (node-level definitions of Section 3.7) is retained. Training used PyTorch 2.6 (Paszke et al., 2019) on an AWS SageMaker ml.g5.2xlarge instance with one NVIDIA A10G GPU (24 GB), in 4 47 h. Hyperparameters are summarised in Table 2.

Table 2: Hyper-parameters of the GINO surrogate and of its training.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Input / output channels</td><td>12 / 1 (depth increment)</td></tr><tr><td>GNO radius (encoder, decoder)</td><td>0.15 (normalised coordi-</td></tr><tr><td>Latent grid</td><td>nates) 32 × 32</td></tr><tr><td>FNO modes</td><td>16 × 16</td></tr><tr><td>FNO layers / hidden channels GNO kernel MLPs (encoder; de-</td><td>4 / 64 (GELU) [80,80,80]; [512,256],</td></tr><tr><td>coder) Positional embedding channels</td><td>linear transform</td></tr><tr><td>Other architecture settings</td><td>32 neuraloperator 2.0.0</td></tr><tr><td></td><td>defaults</td></tr><tr><td>Trainable parameters Optimiser</td><td>2,733,629 AdamW, lr  $1 0 ^ { - 4 }$  weight</td></tr><tr><td>Schedule</td><td>decay  $1 0 ^ { - 5 }$  5-epoch warm-up, cosine</td></tr><tr><td>Epochs / batch size</td><td>annealing</td></tr><tr><td>Gradient clipping</td><td>100 / 16 1.0 (norm)</td></tr><tr><td>Wet weight λ / threshold  $h _ { w }$ </td><td>5 / 0.05 m, incl. predicted</td></tr><tr><td></td><td>wet</td></tr><tr><td>Input-depth noise Push-forward probability</td><td>0.1 σd (K = 1 batches)</td></tr><tr><td>Unroll schedule K</td><td>0.3 1, 2, 8, 16, 32 (20 epochs</td></tr><tr><td></td><td>each)</td></tr><tr><td>Rollout validation Rain_win / BC_win windows</td><td>every 10 epochs</td></tr><tr><td>BC kernel σ</td><td>6 / 18 steps 1000 m</td></tr><tr><td>Hardware</td><td>1x NVIDIA A10G</td></tr><tr><td></td><td>(24 GB)</td></tr><tr><td>Training time</td><td>4.47 h</td></tr></table>

## 3.7. Evaluation metrics

All metrics are computed on free-running rollouts of 216 steps (72 h) started from the spin-up state, for the 22 validation simulations (scaled variants of the training events) and the 38 test events (independent observed events).

Depth and extent metrics. Following Taghizadeh et al. (2026), accuracy is measured on the absolute depth at all $N ~ = ~ 6 5 9$ nodes and all time steps. For a simulation with predicted and true depths $\hat { d } _ { n } ^ { t }$ and $d _ { n } ^ { t }$

$$
\mathrm { R M S E } _ { t } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { \sum _ { n = 1 } ^ { N } \omega _ { n } \left( \hat { d } _ { n } ^ { t } - d _ { n } ^ { t } \right) ^ { 2 } } ,\tag{5}
$$

$$
\mathrm { C S I } _ { \mathrm { t h r } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { H _ { \mathrm { t h r } } ^ { t } } { H _ { \mathrm { t h r } } ^ { t } + M _ { \mathrm { t h r } } ^ { t } + F _ { \mathrm { t h r } } ^ { t } } ,\tag{6}
$$

where $T = 2 1 6 , \omega _ { n } = 1 / N$ , and H, M and F are the (weighted) numbers of hits, misses and false alarms of the wet maps $\hat { d } >$ thr and d > thr, for thr = 0 05 and 0 3 m. Each simulation thus yields one value, and we report the mean ± standard deviation over simulations. Because Voronoi polygons difer in area, we also report area-weighted versions $\textstyle ( \omega _ { n } = A _ { n } / \sum _ { m } A _ { m }$ and H, M,

F summed in area). To characterise maximum values and hazard timing, we further compute, per simulation and over nodes, the coeficient of determination $R ^ { 2 }$ between true and predicted maximum-depth maps, between arrival times (first step with depth > 0 1 m, on nodes wet in both) and between inundation durations (time with depth > 0 1 m, on nodes wet in either).

Node-level flood metrics. To focus on the flood signal we define, for each node, a base level $b _ { n }$ as the minimum over the training simulations of the per-simulation maximum depth, and the anomaly

$$
a _ { n } ^ { t } = \operatorname* { m a x } \bigl ( d _ { n } ^ { t } - b _ { n } , 0 \bigr ) .\tag{7}
$$

Nodes that are deeper than 1 m even in the weakest event (permanently deep channel nodes) and isolated deep pits are excluded; a node is wet when $a _ { n } ^ { t } > 5 \mathrm { c m }$ , and the flood volume is $\textstyle \sum _ { n } a _ { n } ^ { t } A _ { n } .$ . Because node depths are polygon means, polygons crossed by a river also contain the in-channel stage rise, so these metrics reflect near-channel as well as overbank dynamics. For each test event we compute, at the step of the true volume peak, the area-weighted CSI and false-alarm ratio (FAR) and the ratio of predicted to true volume; the area-weighted MAE over nodes wet in the truth or in the prediction over the whole 72 h; and the timing error of the predicted volume peak; these scores are averaged over all test events. Between-event diagnostics complement these scores: the correlation between true and predicted peak volumes, their coeficients of variation, and the one-step (teacher-forced) skill relative to persistence in the rising, peak and recession phases.

## 3.8. Ensemble forecasting

Because a rollout costs a small fraction of a LISFLOOD-FP run (Section 4.6), the surrogate can propagate forcing uncertainty by ensembles. For each test event we generate 50 members by perturbing the river forcing, with a lognormal amplitude factor $( \sigma = 0 . 3 0 )$ applied to the anomaly above the event minimum and a time shift drawn from a normal distribution with standard deviation 12 steps (4 h), and the rain, with a lognormal amplitude factor (σ = 0 15) and a time shift with standard deviation 3 steps (1 h). All members are run in one batched rollout. At the true peak we evaluate the ensemble spread on wet nodes, the ratio of spread to the error of the ensemble mean (1 for a consistent ensemble) and the fraction of nodes with an informative flood probability (anomaly > 10 cm with probability between 5% and 95%). Since the error is measured against the unperturbed LISFLOOD-FP run, the spread–error ratio is a consistency check rather than a calibration measure.

## 3.9. Comparison of LISFLOOD-FP with the observed flood extent

The credibility of the training data is assessed by comparing the LISFLOOD-FP simulation of the 16 May 2023 event, run with the same set-up as the dataset, with the regional perimeter of the areas flooded on 16–17 May 2023 (Section 2.2). The comparison protocol is as follows. (i) The simulated flood map is the maximum depth over the 72 h horizon, thresholded at 0 10 m, as in previous binary-pattern evaluations (Aronica et al.,

![](images/01ef7e251cd8f891398f61a3bef08797e34e39127fb2079ba8808deaa6f27b82.jpg)  
Figure 5: LISFLOOD-FP maximum water depth above 0 10 m over 72 h for the 16 May 2023 event compared, cell by cell, with the regional perimeter of the areas flooded on 16–17 May 2023 (Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025), municipality of Forlì.

2002; Travert et al., 2024). (ii) The main channel itself is excluded. (iii) The comparison is restricted to the municipality of Forlì and carried out on the 100 m cells, a cell being observed as flooded when at least 30% of its area is covered by the perimeter. Agreement is summarised by the critical success index $\mathrm { C S I } = H / ( H + M + F )$ (Wing et al., 2017; Stephens et al., 2014), with H, M and F the numbers of hit, missed and false-alarm cells. The perimeter is an envelope of the event without timing information, so the comparison checks the simulated maximum extent only.

## 4. Results

## 4.1. Credibility of the LISFLOOD-FP training data

Our training set is driven by observed gauge records, which allows the simulator set-up itself to be checked against an observed flood. Following the protocol of Section 3.9, the maximum water depth simulated by LISFLOOD-FP for the 16 May 2023 event was compared with the regional perimeter of the areas flooded on 16–17 May 2023 (Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025), using the 0 10 m wet threshold adopted in previous binary-pattern evaluations (Aronica et al., 2002; Travert et al., 2024). Over the municipality of Forlì, LISFLOOD-FP reaches a critical success index of $\mathrm { C S I } ~ = ~ 0 . 6$ . For a 100 m model without subgrid channels, driven only by stage and rain gauges, this indicates a credible reproduction of the observed flood pattern, supporting the use of these simulations as training data.

Figure 5 shows the spatial pattern behind the score. The bulk of the observed flooding in the plain north-west of the city, between the Montone and the north-western municipal boundary, is reproduced as hits over a large contiguous area. Misses concentrate along the margins of the observed polygons and in a band on the eastern side of the municipality, while many of the model-only cells are small, scattered patches of water away from the main rivers. Figure 6 shows the evolution in time of the same flood simulated by LISFLOOD-FP.

Figure 7 illustrates the range of hydraulic conditions covered by the simulations. The largest events, such as the May 2023 training event and the September 2024 test event, flood extensive areas of the plain downstream of the city, with maximum depths of 0.5–2 m along the Montone corridor and north of the municipality, whereas low-forcing events keep the water within the channel network. This spread, from in-channel conditions to municipality-scale inundation, is what the surrogate has to reproduce from the same set of inputs.

## 4.2. Accuracy of the surrogate

Table 3 reports the depth and extent metrics, computed on absolute depth at all nodes and over all 216 autoregressive steps (72 h), for the 22 validation simulations and for the 38 independent test events. On the test events the GINO surrogate achieves a time-averaged depth RMSE of $0 . 0 0 8 \pm 0 . 0 0 9$ m and a CSI of $0 . 9 3 4 \pm 0 . 0 3 9$ at 0 05 m and $0 . 8 4 4 \pm 0 . 0 8 4 \mathrm { a t } 0 . 3 \mathrm { m }$ . Maximumdepth maps are reproduced with $R ^ { 2 } = 0 . 9 8 4 \pm 0 . 0 1 7$ , and the hazard metrics derived from the full time series, arrival time and inundation duration, reach $R ^ { 2 } = 0 . 8 5 6 { \pm } 0 . 2 1 9$ and $0 . 8 7 6 { \scriptstyle \pm 0 . 0 9 6 }$ Area-weighted scores, which give each node a weight proportional to its polygon area, are consistent with the node-averaged ones (test CSI of $0 . 9 3 6 \pm 0 . 0 2 7$ at 0 05 m and $0 . 8 3 6 \pm 0 . 0 8 9$ at 0 3 m).

Validation and test scores are very close (CSI at 0 05 m of 0 936 and $0 . 9 3 4 ; R ^ { 2 }$ of the maximum depth of 0 988 and 0 984). The validation simulations are scaled variants of the training events, whereas the test events are observed events with their own hydrograph shapes, rain patterns and timing, never seen in any form during training. The surrogate therefore generalises from 24 base events to new observed events with no loss of skill in the detection of inundated nodes.

## 4.3. Node-level flood metrics

To isolate the flood signal we also evaluate node-level flood metrics, defined on the anomaly above each node’s base level (Section 3.7). Averaged over the 38 test events, the surrogate reaches an area-weighted CSI at the true volume peak of 0 840 (90% bootstrap confidence interval [0 792 0 885]; median over events 0 904), with a false-alarm ratio of 0 072 ([0 040 0 112]). It reproduces the peak flood volume with a mean ratio of 0 971 ([0 954 0 988]) and has a mean absolute error on wet nodes of 0 020 m ([0 017 0 025]) over the full 72 h. The peak is predicted on average 0 73 h late, close to two output intervals.

The surrogate also ranks events correctly by magnitude, which is essential for warning. Across test events, the correlation between true and predicted peak flood volume is $r = 0 . 9 9 5$ . The predicted between-event variability is slightly compressed (coeficient of variation of the peak volume 0 799 against 0 864 in the simulations), and the dependence of the flood volume on rainfall is preserved $( r = 0 . 8 7 3$ predicted against 0 877 true).

![](images/182825c01d41cadc230d2e8e280ee157807fb7289e988760343bbf84022ed2f6.jpg)  
Figure 6: Water depth simulated by LISFLOOD-FP for the 16 May 2023 event, from 12 h to 72 h after the start of the simulation window.

Table 3: Accuracy of the GINO surrogate on the 22 validation simulations and on the 38 independent test events, mean ± standard deviation over simulations.
<table><tr><td>Metric</td><td>Validation (n = 22)</td><td> $\mathrm { T e s t } \left( n = 3 8 \right)$ </td></tr><tr><td>RMSE (m)</td><td> $0 . 0 0 6 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 9$ </td></tr><tr><td>CSI (0.05 m)</td><td> $0 . 9 3 6 \pm 0 . 0 4 5$ </td><td> $0 . 9 3 4 \pm 0 . 0 3 9$ </td></tr><tr><td>CSI (0.3 m)</td><td> $0 . 9 0 1 \pm 0 . 0 8 1$ </td><td> $0 . 8 4 4 \pm 0 . 0 8 4$ </td></tr><tr><td>RMSE, area-weighted (m)</td><td> $0 . 0 0 7 \pm 0 . 0 0 8$ </td><td> $0 . 0 0 9 \pm 0 . 0 1 0$ </td></tr><tr><td>CSI (0.05 m), area-weighted</td><td> $0 . 9 4 2 \pm 0 . 0 3 9$ </td><td> $0 . 9 3 6 \pm 0 . 0 2 7$ </td></tr><tr><td>CSI (0.3 m), area-weighted</td><td> $0 . 8 9 9 \pm 0 . 0 7 9$ </td><td> $0 . 8 3 6 \pm 0 . 0 8 9$ </td></tr><tr><td> $R ^ { 2 }$  maximum depth</td><td> $0 . 9 8 8 \pm 0 . 0 1 4$ </td><td> $0 . 9 8 4 \pm 0 . 0 1 7$ </td></tr><tr><td> $R ^ { 2 }$  arrival time</td><td> $0 . 9 2 1 \pm 0 . 0 9 1$ </td><td> $0 . 8 5 6 \pm 0 . 2 1 9$ </td></tr><tr><td> $R ^ { 2 }$  inundation duration</td><td> $0 . 9 2 9 \pm 0 . 0 5 9$ </td><td> $0 . 8 7 6 \pm 0 . 0 9 6$ </td></tr></table>

Figure 8 shows the free-running trajectories of the flood volume for all test events. The surrogate reproduces the timing, the shape of the rising limb and the recession of each event over the full 72 h, and the ordering of events by magnitude. Figure 9 shows how the errors evolve with lead time. The median volume error stays within 1% of the peak volume up to about 42 h and grows to about 4% at the end of the recession, where the surrogate retains slightly too much water; there is no sign of the compounding error growth typical of unstable autoregressive models. The CSI is low only in the first hours, when very few nodes are wet and the score is dominated by small absolute diferences; it exceeds 0 8 from about 24 h, peaks at about 0 9 around 42 h and remains above 0 8 until the end of the horizon. The MAE on wet nodes drops after the first hours, shows a temporary increase around 30 h, when the flood is spreading fastest, and settles at about 0 012 m during the recession.

![](images/8b7c2bd979803bcfd2e51180ec784ae902fe8158c33a6d47591ab8674297d914.jpg)  
Figure 7: Maximum water depth simulated by LISFLOOD-FP over 72 h for a selection of runs of the final set-up, ordered from the most to the least flooded.

![](images/202de653bd8231d4540c210fa48b0e30878f8fa3cc3df59c465cc7f4e8078826.jpg)

![](images/cb3dd17863846d9340032223c24be66f4846c69ebf2fd379e8dd18875f9b7687.jpg)  
Figure 8: Flood volume (sum over nodes of the depth anomaly times the polygon area, m<sup>3</sup>) for all 38 test events: LISFLOOD-FP (left) and free-running GINO surrogate rollout over 72 h (right). Line colour: total efective rainfall of the event (mm).

## 4.4. Representative and challenging events

Figure 10 shows two test events of diferent magnitude in detail. For the test event of 2006-03-05 the surrogate tracks the inundated fraction of the municipality, the flood volume and the depth anomaly at the most afected node almost exactly through the rising limb and the peak, including the two-stage rise of the node hydrograph; in the recession it slightly overestimates the remaining volume and wet area. For the much smaller event of 2016-03-18 the flood volume is again reproduced closely, with a second, slightly overestimated peak around 55 h; the very small wet area (below 2% of the evaluated area) is underestimated, and at the most afected node the timing of the rise is correct but its amplitude is underestimated (about 0 10 m against 0 14 m).

The spatial pattern at the peak of the 2006-03-05 event is shown in Fig. 11. The predicted map reproduces the location and depth of the flooded nodes, from the deepest nodes at the northern tip of the municipality to the flooded nodes west and south of the city. Errors are within a few centimetres almost

Error as a function of lead time (median over the test events)

![](images/dec00be854ccfbbb94c9d7f92d72f3abb5c54cd358b4954f40c00553cd2f102e.jpg)  
Figure 10: Two test events of diferent magnitude: 2016-03-18 (top) and 2006-03-05 (bottom). Left: wet fraction of the municipality (% of the area with anomaly above 5 cm); centre: flood volume (sum of anomaly times area, m<sup>3</sup>); right: depth anomaly at the node with the largest anomaly, with domain-mean rainfall (grey bars, right axis). Solid blue: LISFLOOD-FP; dashed red: GINO surrogate (free-running rollout, 72 h).

![](images/ec503714ef6b9b5ffbab5d2db81ff467f8b5b6b87eee4da96fb865edd5a7a7c9.jpg)

![](images/a064ac5638d378c5763b6f5a2674b5e2b0e9a27aca924f0756133663cb85c1fc.jpg)

![](images/8633d435715216242d4ceeed767778fd761b0ce4ae6697c6a01811f52a7558e5.jpg)  
Figure 9: Error as a function of lead time, median over the 38 test events in 6 h bins: (left) volume error relative to the true peak volume; (centre) area-weighted CSI; (right) MAE on nodes wet in truth or prediction (m). Node-level flood metrics on the anomaly above each node’s base level.  
LISFLOOD-FP vs GINO surrogate: wet area, flood volume and most flooded node

everywhere (at most about 0 08 m), with the largest local underestimation at a few nodes along the eastern river corridor and at the south-western edge of the city.

The most challenging test cases are the two largest floods of the test set, 2019-11-17 and 2024-09-18, with CSI at 0 05 m of 0 808 and 0 820, respectively. For 2019-11-17 the RMSE is 0 020 m and the maximum-depth map still reaches $R ^ { 2 } = 0 . 9 7 1 ;$ 2024-09-18 is the most demanding event overall (RMSE 0 054 m, $R ^ { 2 } = 0 . 8 9 9$ for maximum depth).

## 4.5. Ensemble forecasting

Because one rollout takes a single batched forward pass per time step, the surrogate can turn the uncertainty of the forcing into a probabilistic flood forecast. For each test event we generated 50 members by perturbing the amplitude and timing of the river hydrographs and of the rainfall (Section 3.8) and ran them in one batched rollout. Figure 12 shows the result for the 2006-03-05 event. The ensemble mean reproduces the true peak map, and the ensemble spread and the probability of an anomaly above 0 10 m concentrate on the flood-prone nodes near the rivers and in the north of the municipality, while the rest of the domain is confidently dry. Over all test events, the mean spread on wet nodes is 0 082 m and the median ratio between ensemble spread and error of the ensemble mean is 0 72, i.e. the ensemble is moderately under-dispersive but of the right order of magnitude (a ratio of 1 indicates a consistent ensemble). Only 8.0% of the nodes have a flood probability between 5% and 95%: the forecast is sharp, and the uncertainty is confined to a small set of nodes where it is informative for decision makers.

![](images/4d36ab614a01986549a0e0214c0591a691cbb3d5094960837b862ca9e3085598.jpg)

![](images/204b9a17d4917466d4e5dffad1a81c9e2c6d26cce84565be3785dcabafc7e71a.jpg)

![](images/92710580109316aefd82248e44f98a84750b0e1a008dfb8dc79c9988fb3a7f10.jpg)

Figure 11: Node depth anomaly (m) at the time of the true volume peak (34 7 h) for the test event 2006-03-05: LISFLOOD-FP (left), GINO surrogate (centre) and error, prediction minus truth (right; red = overestimation, blue = underestimation). Each dot is a node of the 659-node Voronoi mesh; coordinates in UTM 32N (m).  
![](images/c429889d1b80f5fd741618df0380c37d35863be05afd73ededa07b8f145045e6.jpg)  
Figure 12: 50-member ensemble forecast with the GINO surrogate for a test event, at the time of the true volume peak. Top: LISFLOOD-FP depth anomaly (m), ensemble-mean anomaly (m) and probability of an anomaly above 0 10 m (%). Bottom: ensemble spread (standard deviation, m); histogram of the flood probability over the evaluated nodes; perturbed boundary water surface elevation of the 50 members (grey) and observed boundary condition (red) at the node with the largest stage range, with the time of the true peak (dotted line).

## 4.6. Computational eficiency

We compare the wall-clock time needed to produce a 72 h forecast for the municipality on the same hardware, an AWS g5.2xlarge instance with one NVIDIA A10G GPU (24 GB). LISFLOOD-FP must simulate its whole computational domain, which is required to resolve the flows entering and leaving the municipality, whereas the surrogate predicts directly the 659 polygon depths of the risk product. For LISFLOOD-FP we report the duration of the time-stepping loop printed by the solver (loop time, snapshot output included); for the surrogate, the end-to-end rollout time, including the construction of the forcing channels, with data already in memory and after one warmup run.

Over the 254 event simulations of the dataset, a 72 h LISFLOOD-FP run takes a median of 55 s (range 51–76 s; 57 s for the 38 test events). The GINO surrogate produces the same 72 h forecast, 216 autoregressive steps, in 1 72 s, about 32 times faster. The gain is much larger for ensembles: the 50-member ensemble

of Section 4.5 is computed in a single batched rollout in 6 19 s, about 0 12 s per member, whereas 50 LISFLOOD-FP runs on the same GPU would take about 46 min $( 5 0 \times 5 5 \mathrm { s } )$ , a speed-up of about 440.

## 5. Discussion

## 5.1. Accuracy on independent observed events

The main result of this study is that a time-dependent GINO keeps high accuracy when it is tested on observed events that are independent of the training set. In fact, none of the 38 test events, nor any scaled variant of them, was used for training or model selection. Their hydrograph shapes, rain patterns and timing therefore difer from those of the 24 training events. Accuracy on the test set is nonetheless close to the validation accuracy. CSI at 0 05 m is $0 . 9 3 4 \pm 0 . 0 3 9$ on the test set against $0 . 9 3 6 \pm 0 . 0 4 5$ on validation, and the $R ^ { 2 }$ of the maximum depth is $0 . 9 8 4 \pm 0 . 0 1 7$ against $0 . 9 8 8 \pm 0 . 0 1 4$ (Table 3). The drop is larger for the deeper threshold (CSI at 0 3 m $0 . 8 4 4 \pm 0 . 0 8 4$ against $0 . 9 0 1 \pm 0 . 0 8 1 )$ and for the timing metrics, which are the hardest to transfer to new events. The surrogate also orders events correctly by magnitude: the correlation between true and predicted peak flood volume across test events is 0.995. This is the information an early-warning system needs to tell a minor event from a major one.

The node-level metrics confirm this picture (Section 4.3). Over the 38 test events, the surrogate captures the flooded area at the volume peak with a CSI of 0.840, reproduces the peak flood volume with a mean ratio of 0.971 and has a wet-node MAE of 0 020 m over the full 72 h. Since none of these events, nor any variant of them, was available in training, the surrogate does more than reproduce known scenarios: it composes the response to a new combination of river and rain forcing.

## 5.2. Operational value of stage-driven forcing

Driving the surrogate with river stage instead of discharge is a deliberate design choice. Stage is what the regional network measures and transmits in real time (Arpae Emilia-Romagna – Servizio IdroMeteoClima (SIMC)). Converting it to discharge would add a rating-curve error that is largest exactly during floods, when the curves are extrapolated and the channel geometry may change. The first-downstream-gauge rule ties each inflow to a single, well-defined observation, and the libraryanchored calibration of the stage boundary conditions changed the per-river scales by moderate factors (Section 3.1). As a result, the full chain—observed gauges, boundary conditions, LISFLOOD-FP, surrogate—can be run in the same way in hindcast and in forecast mode, using stage forecasts or observed stages as they arrive. The same pipeline also produced the test set: 38 of the 43 candidate events passed the automatic quality control of the driver gauges.

## 5.3. Product-oriented mesh and volume-conserving targets

The Voronoi mesh was designed for the risk product, with seeds placed according to exposure (buildings, transport, geology, water), not for the hydraulics. Predicting directly on it means that the surrogate output is already in the units used to report risk per area, with no post-processing. Polygon-mean depths conserve the flood volume of the LISFLOOD-FP solution by construction. Because the number of nodes and the architecture are unchanged, this choice does not add to the inference cost. On the training side, weighting the loss on predicted wet nodes and extending the push-forward curriculum to 32 unrolled steps target spurious thin water and long-horizon drift, the two typical failure modes of autoregressive surrogates (Brandstetter et al., 2022; Taghizadeh et al., 2025); both act only on training and leave the inference cost unchanged.

## 5.4. Ensembles and simulator credibility

Because all ensemble members share the static inputs and the mesh, a 50-member ensemble is computed as a single batched rollout per event (Section 4.5). The ensemble produces floodprobability maps in which 8.0% of the nodes carry informative probabilities (between 5% and 95%). These nodes mark the uncertain fringe of the flood, which is where forecasters should focus. The median spread-to-error ratio of 0.72 shows that the current perturbations of river stage and rain are somewhat under-dispersive. Calibrating the perturbation amplitudes against forecast errors of the stage and rain predictions is a natural next step.

Finally, a surrogate can only be as good as its simulator. Surrogate studies usually treat the hydraulic simulations as ground truth; we additionally compared the LISFLOOD-FP set-up with the regional perimeter of the May 2023 flooding (Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025) and obtained a CSI of 0.6 (Section 4.1). We used the regional perimeter rather than the Copernicus EMS radar delineation (Copernicus Emergency Management Service, 2023), because the radar misses much of the flooding in builtup areas. Binary skill scores depend strongly on flood magnitude, resolution and the reference data (Stephens et al., 2014; Wing et al., 2017). For a 100 m model without subgrid channels or levee and drainage detail, compared with a 1:5000 photointerpreted envelope, this agreement indicates that the simulator captures the main flooded areas of the largest recent event. It thus provides a credible physical basis for the surrogate.

## 5.5. Limitations and future work

Several limitations point to future work. (i) The surrogate was trained and tested on a single municipality. Training on several cities is the natural route towards a future implementation. (ii) The surrogate inherits the limitations of its simulator: a 100 m LISFLOOD-FP set-up without subgrid channels, in which levees are represented only as raised DEM cells at crest elevation (overtopping only, no breaching), and without the urban drainage network, which limits the detail of urban flow paths. (iii) Results are from a single training seed.

## 6. Conclusions

This study presented a time-dependent geometry-informed neural operator surrogate of LISFLOOD-FP for rapid pluvial– fluvial flood forecasting in the municipality of Forlì, Italy. The surrogate is driven only by the observations available in real time: river stage and rain gauges. Building on the time-dependent GINO formulation of Taghizadeh et al. (2026), we developed a reproducible pipeline that turns public regional observations into a physically based training set. In this pipeline each river inflow is driven by its first downstream stage gauge through calibrated boundary conditions, and rain enters as SCS-CN effective rainfall. An autoregressive GINO, trained on 216 simulations of 24 observed events with scaled forcing, predicts volume-conserving polygon-mean depths on an exposure-based Voronoi mesh of 659 nodes every 20 min over a 72 h horizon.

Tested on 38 independent observed events that were never seen in training, the surrogate reaches a CSI of 0.934 at 0 05 m and 0.844 at 0 3 m, a depth RMSE of 0 008 m and an R<sup>2</sup> of 0.984 for the maximum-depth maps. These values are close to its accuracy on the validation simulations. At node level, it captures the flooded area at the volume peak with a CSI of 0.840 and the peak volume with a mean ratio of 0.971. Volumeconserving targets preserve the flood volume of the simulator on the product mesh at no additional inference cost. On the same GPU, a 72-h forecast takes 1 7 s against 55 s for LISFLOOD FP, and a 50-member ensemble, computed in one batched rollout, takes 6 2 s against about 46 min, and produces flood-probabil maps for early warning. The underlying LISFLOOD-FP set-up reproduces the observed May 2023 flood extent with a CSI of 0.6.

Together, these results show that neural-operator surrogates can move from controlled experiments on synthetic scenarios to an operational setting: driven by stage rather than discharge, forced by both rivers and rain, delivered on the units of the risk product, and tested on real events. Future work will extend the approach to several cities through multi-domain training and domain adaptation, refine the mesh near the rivers, enrich the training set with more extreme events, and couple the surrogate with operational stage and rain forecasts. These steps lead towards fast ensemble-based flood early warning for the Emilia-Romagna region.

## References

Aronica, G., Bates, P.D., Horritt, M.S., 2002. Assessing the uncertainty in distributed model predictions using observed binary pattern information within GLUE. Hydrological Processes 16, 2001–2016. doi:10.1002/hyp.398.

Arpae Emilia-Romagna – Servizio IdroMeteoClima (SIMC), . Dext3r: hydro-meteorological data extraction web application. Arpae Emilia-Romagna. URL: https://simc. arpae.it/dext3r/. rain-gauge and river-stage observations. Accessed 2026-10-23.

Arrighi, C., Domeneghetti, A., 2024. Brief communication: On the environmental impacts of the 2023 floods in Emilia-Romagna (Italy). Natural Hazards and Earth System Sciences 24, 673–679. doi:10.5194/nhess-24-673-2024.

Bates, P.D., De Roo, A.P.J., 2000. A simple raster-based model for flood inundation simulation. Journal of Hydrology 236, 54–77. doi:10.1016/S0022-1694(00)00278-X.

Bates, P.D., Horritt, M.S., Fewtrell, T.J., 2010. A simple inertial formulation of the shallow water equations for eficient twodimensional flood inundation modelling. Journal of Hydrology 387, 33–45. doi:10.1016/j.jhydrol.2010.03.027.

Bentivoglio, R., Isufi, E., Jonkman, S.N., Taormina, R., 2022. Deep learning methods for flood mapping: a review of existing applications and future research directions. Hydrology and Earth System Sciences 26, 4345–4378. doi:10.5194/ hess-26-4345-2022.

Bentivoglio, R., Isufi, E., Jonkman, S.N., Taormina, R., 2023. Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks. Hydrology and Earth System Sciences 27, 4227–4246. doi:10.5194/hess-27-4227-2023.

Brandstetter, J., Worrall, D., Welling, M., 2022. Message passing neural PDE solvers, in: International Conference on Learning Representations (ICLR). URL: https://arxiv. org/abs/2202.03376, arXiv:2202.03376.

Copernicus Emergency Management Service, 2023. EMSR664: Rapid mapping activation for the floods in Emilia-Romagna, Italy (May 2023). Copernicus EMS Rapid Mapping, European Commission. URL: https://mapping.emergency.copernicus.eu/ activations/EMSR664. activated 16 May 2023; AOI01 Forlì delineation from COSMO-SkyMed imagery of 17 May 2023. Accessed 2026-10-23.

Dorrington, J., Wenta, M., Grazzini, F., Magnusson, L., Vitart, F., Grams, C.M., 2024. Precursors and pathways: dynamically informed extreme event forecasting demonstrated on the historic Emilia-Romagna 2023 flood. Natural Hazards and Earth System Sciences 24, 2995–3012. doi:10.5194/ nhess-24-2995-2024.

Guo, Z., Leitão, J.P., Simões, N.E., Moosavi, V., 2021. Datadriven flood emulation: Speeding up urban flood predictions by deep convolutional neural networks. Journal of Flood Risk Management 14, e12684. doi:10.1111/jfr3.12684.

Kabir, S., Patidar, S., Xia, X., Liang, Q., Neal, J., Pender, G., 2020. A deep convolutional neural network model for rapid prediction of fluvial flood inundation. Journal of Hydrology 590, 125481. doi:10.1016/j.jhydrol.2020.125481.

Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiafini, M., George, R.J., Bonev, B., Azizzadenesheli, K., Berner, J., Duruisseaux, V., Anandkumar, A., 2024. A library for learning neural operators. arXiv preprint arXiv:2412.10354. URL: https://arxiv.org/abs/2412.10354, doi:10.48550/ arXiv.2412.10354, arXiv:2412.10354.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural

operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1–97. URL: https://www.jmlr.org/papers/v24/ 21-1524.html.

Kumar, C., McSpadden, D., Goldenberg, S., Schram, M., Richter, H., Wang, Y., Roy, B., Goodall, J.L., 2024. Location invariant flood prediction using Fourier neural operator, in: 2024 International Conference on Machine Learning and Applications (ICMLA), IEEE. pp. 1105–1110. doi:10. 1109/ICMLA61862.2024.00168.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2020. Neural operator: Graph kernel network for partial diferential equations. arXiv preprint arXiv:2003.03485. URL: https: //arxiv.org/abs/2003.03485, doi:10.48550/arXiv. 2003.03485, arXiv:2003.03485.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial diferential equations, in: International Conference on Learning Representations (ICLR). URL: https://arxiv.org/abs/2010.08895, arXiv:2010.08895.

Li, Z., Kovachki, N., Choy, C., Li, B., Kossaifi, J., Otta, S., Nabian, M.A., Stadler, M., Hundt, C., Azizzadenesheli, K., Anandkumar, A., 2023. Geometry-informed neural operator for large-scale 3D PDEs, in: Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pp. 35836–35854. doi:10.52202/075280-1556.

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations (ICLR). URL: https://arxiv.org/abs/ 1711.05101, arXiv:1711.05101.

Löwe, R., Böhm, J., Jensen, D.G., Leandro, J., Rasmussen, S.H., 2021. U-FLOOD – topographic deep learning for predicting urban pluvial flood water depth. Journal of Hydrology 603, 126898. doi:10.1016/j.jhydrol.2021.126898.

Mosavi, A., Ozturk, P., Chau, K.w., 2018. Flood prediction using machine learning models: Literature review. Water 10, 1536. doi:10.3390/w10111536.

NeuralOperator developers, . neuraloperator: Learning in infinite dimension with neural operators. GitHub repository. URL: https://github.com/neuraloperator/ neuraloperator. accessed 2026-10-23.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., De-Vito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S., 2019. Py-Torch: An imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019). URL: https:

//proceedings.neurips.cc/paper/2019/hash/ bdbca288fee7f92f2bfa9f7012727740-Abstract. html.

Pfaf, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P.W., 2021. Learning mesh-based simulation with graph networks, in: International Conference on Learning Representations (ICLR). URL: https://arxiv.org/abs/2010.03409, arXiv:2010.03409.

Regione Emilia-Romagna, Database topografico regionale (DBTR): natural watercourses layer (FIU\_GLI), scale 1:5000. Geoportale Regione Emilia-Romagna. URL: https://geoportale. regione.emilia-romagna.it/catalogo/ dati-cartografici/cartografia-di-base/ database-topografico-regionale. licence CC BY 4.0. Accessed 2026-10-26.

Regione Emilia-Romagna, 2024. DTM 50 cm – RER 2023- 24: digital terrain model from airborne LiDAR. ArcGIS ImageServer DtmRER2023\_24, Regione Emilia-Romagna, Settore Innovazione Digitale, Dati, Tecnologia e Polo Archivistico. URL: https://servizigis.regione. emilia-romagna.it/arcgis/rest/services/public/ DtmRER2023\_24/ImageServer. accessed 2026-10-19.

Regione Emilia-Romagna – Agenzia regionale per la sicurezza territoriale e la protezione civile, 2025. Relazione attività perimetrazione aree allagate evento maggio 2023 – V7 del 16/09/2025. Geoportale della Regione Emilia-Romagna. URL: https://geoportale.regione.emilia-romagna.it/ approfondimenti/emergenza-maggio-23/report\_ perimetrazioni\_maggio\_2023\_v7\_dsg\_88\_2025.pdf. report V7 of 16 September 2025, adopted by Decree of the Secretary General of the Po River District Basin Authority no. 88/2025; nominal scale 1:5000. In Italian. Accessed 2026-10-23.

Shaw, J., Kesserwani, G., Neal, J., Bates, P., Sharifian, M.K., 2021. LISFLOOD-FP 8.0: the new discontinuous Galerkin shallow-water solver for multi-core CPUs and GPUs. Geoscientific Model Development 14, 3577–3602. doi:10. 5194/gmd-14-3577-2021.

Stephens, E., Schumann, G., Bates, P., 2014. Problems with binary pattern measures for flood model evaluation. Hydrological Processes 28, 4928–4937. doi:10.1002/hyp.9979.

Taghizadeh, M., Zandsalimi, Z., Nabian, M.A., Goodall, J.L., Alemazkoor, N., 2026. FloodForecaster: A domainadaptive geometry-informed neural operator framework for rapid flood forecasting. Journal of Hydrology 664, 134512. doi:10.1016/j.jhydrol.2025.134512.

Taghizadeh, M., Zandsalimi, Z., Nabian, M.A., Shafiee-Jood, M., Alemazkoor, N., 2025. Interpretable physics-informed

graph neural networks for flood forecasting. Computer-Aided Civil and Infrastructure Engineering 40, 2629–2649. doi:10.1111/mice.13484.

Tarquini, S., Isola, I., Favalli, M., Battistini, A., Dotta, G., 2023. TINITALY, a digital elevation model of Italy with a 10 meters cell size (Version 1.1). Istituto Nazionale di Geofisica e Vulcanologia (INGV). URL: https://doi.org/10.13127/ tinitaly/1.1, doi:10.13127/tinitaly/1.1.

Tarquini, S., Isola, I., Favalli, M., Mazzarini, F., Bisson, M., Pareschi, M.T., Boschi, E., 2007. TINITALY/01: a new triangular irregular network of Italy. Annals of Geophysics 50, 407–425. doi:10.4401/ag-4424.

Travert, J.P., Boyaval, S., Goeury, C., Bacchi, V., Zaoui, F., 2024. Evaluation of performance measures for qualifying flood models with satellite observations. arXiv preprint arXiv:2408.00571. URL: https: //arxiv.org/abs/2408.00571, doi:10.48550/arXiv. 2408.00571, arXiv:2408.00571.

USDA Soil Conservation Service, 1986. Urban Hydrology for Small Watersheds. Technical Release 55 (TR-55), 2nd edition. United States Department of Agriculture, Soil Conservation Service. Washington, DC. URL: https://www. govinfo.gov/content/pkg/CZIC-gb980-u73-1986/ html/CZIC-gb980-u73-1986.htm. accessed 2026-10-23.

Wing, O.E.J., Bates, P.D., Sampson, C.C., Smith, A.M., Johnson, K.A., Erickson, T.A., 2017. Validation of a 30 m resolution flood hazard model of the conterminous United States. Water Resources Research 53, 7968–7986. doi:10.1002/ 2017WR020917.

Xu, Q., De Vos, L.F., Shi, Y., Rüther, N., Bronstert, A., Zhu, X.X., 2025. Urban flood modeling and forecasting with deep neural operator and transfer learning. Journal of Hydrology 661, 133705. doi:10.1016/j.jhydrol.2025.133705.

Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, F., Arino, O., 2022. ESA World-Cover 10 m 2021 v200. URL: https://doi.org/10. 5281/zenodo.7254221, doi:10.5281/zenodo.7254221. dataset, version v200. Accessed 2026-10-23.