# A Dataset and Model for Imputing Water Surface Elevation on a Large and Extremely Sparse Spatiotemporal Graph

Ruben Cartuyvels<sup>1∗</sup>, Karim Douch<sup>2,</sup> <sup>7</sup>, Gabriele Bertoli<sup>3,</sup> <sup>4</sup>, Mounia El Baz<sup>1</sup>, Artemis Vrettou<sup>7</sup>, Sébastien Lefèvre<sup>5,</sup> <sup>6</sup>, Diego Fernandez Prieto<sup>2</sup>

<sup>1</sup>European Space Agency, Φ-lab, <sup>2</sup>European Space Agency, Science Hub, <sup>3</sup>University of Florence, <sup>4</sup>Imperial College London, <sup>5</sup>Université Bretagne Sud, IRISA, <sup>6</sup>University of Tromsø, <sup>7</sup>Serco Italia SpA, Rome, Italy

## Abstract

Continuous monitoring of water surface elevation across river networks is critical for flood forecasting, water resource management, and understanding the global water cycle. Yet, the scarcity of in situ gauges across much of the globe constrains the development of reliable modeling frameworks. Satellite altimetry has the potential to alleviate this problem but its use is currently hindered by sparse temporal coverage. To this end, we introduce AmazonWSE, a dataset for training and evaluating large-scale spatiotemporal graph imputation methods that integrates processed satellite altimetry measurements from a range of sources, including the recent wide-swath SWOT sensor. The dataset covers approximately 19K river sections in the Amazon river basin over 10 years (2016–2026), with in situ gauges held out for evaluation. Besides contributing a novel real-world use case with the potential for societal impact, AmazonWSE introduces significant technical challenges: with fewer than 1% of sections observed per day, the dataset is far sparser than existing imputation benchmarks, and its directed acyclic river topology is both structurally diferent from and larger than graphs in existing datasets. We show that prior spatiotemporal graph imputation methods are not adapted to this topology, scale and sparsity, and propose a simple bidirectional selective state space model that outperforms them by sampling connected subgraphs and flattening space and time into a single token sequence with topology-aware positional encodings. Compared to the state-of-the-art published method for SWOT-based WSE densification, which integrates statistics with physical modeling, our model reduces RMSE against in situ gauges by 18-39%, while producing predictions for every river section rather than only those with suficient nearby satellite coverage.

## Introduction

Monitoring water surface elevation (WSE) across river networks is essential for flood forecasting, water resource management, and understanding the global water cycle. The Amazon basin, the largest river system on Earth, is of particular importance because of its role in global climate regulation and biodiversity (Fassoni-Andrade et al. 2021).

Observations of WSE come from diverse sensors that each cover the spatiotemporal domain only partially. In situ flow gauges, such as those operated by Brazil’s National Water and Sanitation agency (ANA), provide sub-daily measurements at fixed locations but cover unevenly a small fraction of the river network. Satellite altimetry missions have measured WSE globally since the 1990s, but these instruments observe a given location only every 10–91 days along narrow ground tracks (Abdalati et al. 2010; Normandin et al. 2018). The Surface Water and Ocean Topography (SWOT) mission, launched in 2022, provides for the first time wideswath observations of river WSE with a 21-day repeat cycle (Biancamaria, Lettenmaier, and Pavelsky 2016), yielding unprecedented spatial coverage but a short temporal record.

![](images/4be51d4f89733ba975a4ac62b3c3a311e1f81c54029950ed7042fbbc25cdc9ca.jpg)

![](images/3477427dbd2e048ea7aa27074fb5658c38dec1fa04ef2338d83f94c7cc7f1fde.jpg)  
Figure 1: Example time series and local graph topology in AmazonWSE. Locations can be observed by zero, one (such as the location observed by SWOT) or more (such as the location observed by ICESat-2 and S3A) sources. The satellite sources measure with reference to the approximate sea level, while the in situ gauges measure deviations from a chosen location-specific reference level.

We synthesize these sensor observations into a benchmark dataset for spatiotemporal graph imputation of daily WSE that observes the Amazon river network. The spatiotemporal graph formed by the Amazon and its tributaries is extremely large with over 19K nodes (or reaches; river sections of ∼8- 12 km as defined in (Altenau et al. 2021)) and highly sparse: on any given day, fewer than 1% of nodes carry an observation. Spatiotemporal graph neural networks (STGNN) for imputation often take full graphs as input, which for large graphs is resource intensive. Consequently, current methods are evaluated on small graphs of ∼1,000 nodes or less. Furthermore, under extreme sparsity, same-day spatial neighborhoods contain almost no observed nodes, either starving the message-passing mechanism or introducing prohibitive amounts of virtual tokens. Finally, we incorporate multiple sources that may observe the same node on the same day, yet due to diferent sensor characteristics, yield a diferent measurement. Figure 1 shows example time series and a fragment of the underlying topology.

<table><tr><td>Dataset</td><td>Nodes</td><td>Obs.</td><td>Missing</td><td>Graph</td></tr><tr><td>AQI-36</td><td>36</td><td>274K</td><td>13.24%</td><td>Cities</td></tr><tr><td>METR-LA</td><td>207</td><td>6.52M</td><td>8.11%</td><td>Road</td></tr><tr><td>PEMS-BAY</td><td>325</td><td>16.94M</td><td>0.003%</td><td>Road</td></tr><tr><td>CausalRivers</td><td>666</td><td>107M</td><td>~8%</td><td>River</td></tr><tr><td>LamaH-CE</td><td>859</td><td>11B</td><td>&lt;20%</td><td>River</td></tr><tr><td>LargeST-CA</td><td>8.6K</td><td>4.52B</td><td>n.r.</td><td>Road</td></tr><tr><td>AmazonWSE</td><td>19.2K</td><td>1.9M</td><td>99%</td><td>River</td></tr></table>

Table 1: Spatiotemporal dataset comparison. Obs.: measurements, reported or estimated from time steps and sparsity.

We show that recent transductive and inductive graph or recurrence based imputation methods do not perform well in our setting. We introduce a simple sequence model which flattens space and time into a single token sequence of observed measurements. This model handles the sparsity more naturally than grid-based STGNNs and its inductive bias fits the sequential upstream→downstream and temporal order inherent to river systems. Our results support recent findings that GNNs applied to river networks for discharge prediction show limited benefit from graph topology (Kirschstein and Sun 2024). Importantly, we find that river topology does inform better predictions, but is better exploited through subgraph sampling and metadata encodings than through explicit graph convolutions.

In summary, this paper makes the following contributions:

1. We introduce a dataset of Amazon WSE time series on an underlying river graph that observes 19K river sections with a daily frequency over 2016–2026 and serves as a benchmark for imputation under extreme sparsity.

2. We propose a Mamba-based sequence model as baseline that we train for masked reconstruction on the heterogeneous observations with topology-aware subgraph samples (Gu and Dao 2023).

3. We show that this model outperforms interpolation and existing transductive and inductive neural methods, and provide ablation studies. We demonstrate improvements in coverage and accuracy against Reach-Reg (Halicki et al. 2026), a current state-of-the-art method for densification of altimetry-derived WSE from SWOT data.

## Related Work

Satellite Altimetry for River Monitoring. Classical altimetry satellites provide an estimate of the WSE along their ground-track by measuring the distance between the satellite and the water surface with a radar, and have provided observations since 1991. The HydroWeb database aggregates measurements from ERS-1/2, Topex/Poseidon, Jason-1/2/3, Envisat, Saral/AltiKa, Sentinel-3A/B, and Sentinel-6A into time series at locations where the satellite ground track crosses a river (Santos da Silva et al. 2010; Normandin et al. 2018). While other altimeters observe only along their precise ground track, the novel wide-swath InSAR sensor of the SWOT satellite observes entire contiguous river segments rather than point crossings (Biancamaria, Lettenmaier, and Pavelsky 2016). Reach-Reg (Halicki et al. 2026) exploits the SWOT measurement geometry for spatiotemporal WSE densification by chaining linear regressions between simultaneously observed reaches and by modeling water velocity. They achieve the best accuracy with re-processed SWOT data which is not available at scale (Schwatke et al. 2015) but show that their method also works on public SWOT RiverSP data. However, like other works (Tourian et al. 2016; Nielsen et al. 2022), they consider only large and well observed rivers without complex topology.

<table><tr><td>Source</td><td>Nodes</td><td># obs.</td><td>Period</td><td>% obs.</td></tr><tr><td>SWORD reaches</td><td>19,172</td><td>− (static)</td><td></td><td></td></tr><tr><td>SWOT RiverSP</td><td>10K</td><td>353K</td><td>2023-26</td><td>1.69%</td></tr><tr><td>HydroWeb</td><td>3.8K</td><td>391K</td><td>2016-26</td><td>0.50%</td></tr><tr><td>ICESat-2</td><td>18K</td><td>208K</td><td>2018-26</td><td>0.32%</td></tr><tr><td>Total observed</td><td>18.5K</td><td>1.9M</td><td>2016-26</td><td>0.97%</td></tr><tr><td>In situ (eval)</td><td>375</td><td>955K</td><td>2016-26</td><td>1.32%</td></tr></table>

Table 2: Dataset statistics for the Amazon basin. Percentage observed (% obs.) is calculated over the respective periods of operation of the sources.

Spatiotemporal Imputation. Early work focused on modeling the temporal dimension (Yi et al. 2016; Cao et al. 2018). Subsequent works model spatial interactions more explicitly and can be categorized into transductive methods, that assume to see every to-be-predicted node during training (Cini, Marisca, and Alippi 2022; Marisca, Cini, and Alippi 2022; Liu et al. 2023a; Cheng et al. 2024; De Felice et al. 2024; Nie et al. 2024; Yang et al. 2025b) and inductive methods, that transfer to entirely unseen nodes (Wu et al. 2021; Zheng et al. 2023; Li et al. 2025; Xu et al. 2025; Ren et al. 2026; Liang et al. 2026). Even though some methods introduce mitigations for the cost of processing large graphs, none of these has been evaluated on graphs with more than 1,200 nodes. SPIN (Marisca, Cini, and Alippi 2022) and IGNNK (Wu et al. 2021) support subgraph sampling, similar to the model proposed here. Only SPIN (Marisca, Cini, and Alippi 2022) and ImputeFormer (Nie et al. 2024) have been shown to work (transductively) at sparsity levels exceeding 90%, but not up to 99% (Marisca, Cini, and Alippi 2022; Nie et al. 2024). Forecasting methods exist for larger but denser graphs (Cini et al. 2023; Liu et al. 2023b).

Existing Datasets. Benchmarks used for spatiotemporal imputation, though having more observations along their temporal axis, are substantially smaller in their spatial extent than AmazonWSE (Table 1). AQI-36 contains 36 hourly airquality series, METR-LA and PEMS-BAY contain 207 and

![](images/4766270fd72c2f8c2e55cd6d733c263135b627dfc9c035fbf8ca8b8960a7097c.jpg)  
Figure 2: Spatial distribution of data sources in AmazonWSE, after quality filters, with SWORD river topology background. Left to right: SWORD reaches, SWOT-observed reaches, HydroWeb virtual stations, ICESat-2 transects, ANA in situ gauges.

325 five-minute trafic series, respectively (Zheng, Liu, and Hsieh 2013; Li et al. 2018). LargeST scales trafic forecasting to 8,600 sensors over five years, and CER-E and PV-US contain energy grid time series over ∼5,000-6,000 nodes, but none of these has been used for imputation (Commission for Energy Regulation 2012; Hummon et al. 2012; Liu et al. 2023b). Table 1 shows that these datasets ofer denser data, which hampers the comparability of imputation studies that report incompatible artificial sparsification scenarios, from random missing entries (Nie et al. 2024) and temporal/spatial blocks (Liu et al. 2023a) to entirely withheld target nodes (Yang et al. 2025a), removed context nodes (Zheng et al. 2023), and held-out variable channels (De Felice et al. 2024). AmazonWSE ofers a consistent benchmark for extreme sparsity; by combining 1.9M irregular observations at 19K nodes over 2016–2026, fewer than 1% of reaches are observed on any day. Its directed acyclic topology is also novel with respect to the cyclical graphs of existing datasets.

Hydrology datasets with discharge or water level time series aggregate data spatially into catchments and thereby erase the topology (Caravan; Kratzert et al. 2023), or only make available in situ gauge data, no satellite altimetry: LamaH-CE; (Klingler, Schulz, and Herrnegger 2021) and CausalRivers (Stein et al. 2025).

Rivers and GNNs. Acosta et al. (2025); Taghizadeh et al. (2025) propose GNNs for spatial flood forecasting. Dufourg et al. (2024) find that a GNN slightly outperforms (Conv)LSTMs when forecasting a water index from satellite image time series. Kirschstein and Sun (2024), on the other hand, report that reflecting river network topology through node adjacency in GNNs does not improve discharge forecasting over using isolated node-level MLPs. Here, we find that topology information does improve WSE reconstruction if used to sample context and encoded through positional encodings rather than through GNN message passing.

## Dataset

We construct a multi-source dataset of water surface elevation time series for the Amazon river by integrating five data sources, each with diferent spatial and temporal characteristics. Figure 2 shows the spatial distribution of the data sources. Table 2 summarizes the resulting dataset.

## SWORD and Topology

The SWORD database (v17b) was designed as a complementary static dataset to SWOT products and provides a topological graph of 19,172 river reaches in the Amazon basin (Altenau et al. 2021). Each reach is characterized by a unique ID, geographic coordinates, and upstream/downstream connectivity. SWORD provides the graph structure, shown in Figures 1-3, on which our model operates and the static metadata used for spatial encodings.

The river topology defines a directed graph $\mathcal { G } = ( \nu , \mathcal { E } )$ where V is the set of SWORD reaches and $( u , v ) \in \mathcal { E }$ indicates that v is immediately downstream of u. The graph is acyclic, v ̸⇝ v, and approximately but not strictly a tree, since reaches sometimes have multiple parents.

## Observation Sources

SWOT RiverSP. The SWOT RiverSP product provides WSE observations mapped to SWORD reaches, from the wide-swath InSAR sensor (SWOT 2025). We use data from the science cycle that started in July, 2023. Of the 19,172 SWORD reaches, approximately 17K are observed by SWOT, and 10K are retained by quality filtering. The repeat cycle takes 21 days, but due to the wide-swath sensor some reaches are observed several times during this period.

HydroWeb.next. The HydroWeb database provides WSE time series at 7.3K river crossings in the Amazon basin, derived from altimetry missions (including S3A, shown in Figure 1) spanning 1993–2026 (Crétaux and Calmant 2015). We use both the operational and research collections and retain 3.7K locations with data between 2016–2026 matched to a nearby reach. Each virtual station has a measurement every 10–35 days depending on the altimetry mission.

ANA In Situ Gauges. The Brazilian National Water Agency (ANA) operates flow gauges across the Amazon basin. We retrieve WSE records via their API (Agencia Nacional de Aguas e Saneamento Basico (ANA) 2026). We retain 375 gauges with data between 2016–2026 after quality filtering, manual inspection and matching to SWORD reaches. Following standard practice in hydrology, we provide the gauge data as ground truth. They provide dense temporal sampling (with an aggregated daily observation during periods of operation) but are geographically sparse.

ICESat-2. The NASA Ice, Cloud and land Elevation Satellite-2 (ICESat-2) mission carries a laser altimetry instrument, which measures along three pairs of narrow laser beams (Abdalati et al. 2010). We use the mean along-track surface elevation for each beam for each transect across a water body from the Level3B ATL22 Mean Inland Surface

Water Data product (Jasinski et al. 2025). Observations span Oct. 2018–Mar. 2026; with a nominal repeat cycle of 91 days and a large spatial coverage. After filtering and matching, time series remain for >90% of SWORD reaches.

## Processing and Quality Filtering

HydroWeb, ANA, and ICESat-2 locations are assigned to the nearest SWORD reach, retaining only matches within 10 km (unmatched locations are discarded, imposing a stricter limit than the 20 km limit used by Halicki et al. (2026)). SWOT RiverSP observations are already mapped to SWORD reaches. HydroWeb, SWOT, and ICESat-2 report elevations referenced to the EGM2008 geoid (Pavlis et al. 2012), which provides a gravity-adjusted estimate of “elevation above sea level”. ANA records are in a local gauge datum; for evaluation, each gauge is therefore mapped to the prediction datum by a fitted linear transformation, following Halicki et al. (2026). Every source is placed on the common daily grid from 2016-01-01 to 2026-05-01 (3,774 days).

Further processing is source-specific and according to expert hydrology standards. For SWOT, we screen product quality fields including reach quality, width, crosstrack distance, crossover calibration, random WSE uncertainty, and severe bit flags. These criteria combine filters adapted from Andreadis et al. (2025) and Halicki et al. (2026) with additional harmonic-residual, observed-pixel precision and ∆width/∆WSE consistency checks we introduce. HydroWeb text products are parsed from both operational and research collections; invalid fill values and the short Jason-2 interleaved (J2N) record are removed before daily alignment. ICESat-2 transects over reservoirs and transects shorter than 50 m are discarded, followed by a per-reach harmonic-residual filter. ANA sub-daily measurements are reduced to a daily median, and filtered using seasonal-trend residuals, followed by manual inspection by an expert. More details are given in Appendix A.2.

All data (filtered and original measurements, uncertainties, quality flags, metadata) is provided in h5netcdf format and can easily be read by xarray. Each dataset may be used freely for any purpose under its provider’s terms; some providers require attribution. Appendix A.3 explains the storage format. All data export, processing, loading and model code will be made public upon acceptance.<sup>1</sup>

## Task: Spatiotemporal Imputation

Given whichever altimetry measurements are available, the task is to reconstruct a daily WSE at each SWORD reach. The evaluation is broken down in two tracks: reconstruction during the SWOT era (2023-07 to 2026-05), and a pre-SWOT hindcast. ANA gauges are not used as model inputs or as training targets but are reserved only for evaluation.

## Model

We formulate WSE densification as masked reconstruction on the directed SWORD graph $\mathcal { G } = ( \nu , \mathcal { E } )$ . An observation is a tuple $( j , i , t , r , h _ { j , t } ^ { ( r ) } )$ , where source location $j$ is mapped to SWORD reach $i \in \mathcal V$ by $i = \nu ( j )$ , t is a daily date bin, r identifies the observing source, and $h _ { j , t } ^ { ( r ) }$ is the measured WSE. Multiple sources may therefore produce distinct tokens for the same reach and day.

![](images/bddd348f76d6562e82c7d017617b357ef5957f1bda728b8e674c78201739847b.jpg)  
Figure 3: Example of sampled subgraph (local edges in grey) with river topology (SWORD reaches in blue).

## Sample Construction

A sample is defined by an anchor location $^ { a , }$ a contiguous time interval I, and a connected local neighborhood $\mathcal { V } _ { a } ^ { \mathrm { ~ \tiny ~ - ~ } } \mathcal { V }$ obtained by following the SWORD topology upstream and downstream from the anchor (Example in Figure 3). Let $\mathcal { I } _ { a }$ be the selected source locations mapped to these reaches. The observed part of the sample is

$$
\begin{array} { c } { { \mathcal { D } _ { a , I } = \left\{ ( j , i , t , r ) : j \in \mathcal { I } _ { a } , i = \nu ( j ) , t \in I , \right. } } \\ { { \left. h _ { j , t } ^ { ( r ) } \mathrm { ~ i s ~ o b s e r v e d } \right\} . } } \end{array}\tag{1}
$$

We create tokens only for available measurements rather than constructing a dense reach–time grid. Thus, sparsity reduces sequence length instead of filling the input with missingvalue tokens.

Token Representation. For token ℓ, let $z _ { \ell }$ be the WSE normalized by the mean/std of the entire time series at source location j (diferent sources observing the same location are normalized separately). The embedded token $\mathbf { e } _ { \ell }$ that is input to the model is a linear projection of $z _ { \ell }$ if the token is not masked and a learned mask token embedding $\mathbf { e } _ { \mathrm { m a s k } }$ if it is masked (for training) or a query token (for inference).

Node-Independent Metadata Encodings. Each dynamic token (masked and non-masked) receives temporal, source, geographic, and topological metadata:

$$
\begin{array} { r l } & { \mathbf { p } _ { \ell } = \mathbf { E } _ { \mathrm { m o n t h } } [ m _ { \ell } ] + \mathrm { P E } _ { \mathrm { d a y } } ( t _ { \ell } - t _ { \mathrm { m i n } } ) + \mathbf { E } _ { \mathrm { s r c } } [ r _ { \ell } ] } \\ & { \qquad + \mathbf { W } _ { a } \mathbf { c } _ { i _ { \ell } } ^ { a } + \mathbf { W } _ { g } \mathbf { c } _ { i _ { \ell } } ^ { g } + \mathrm { T r e e } \mathrm { P E } ( \mathbf { b } _ { i _ { \ell } } ) . } \end{array}\tag{2}
$$

Here, $m _ { \ell }$ is the month, $\mathbf { E } _ { \mathrm { m o n t h } }$ and $\mathbf { E } _ { \mathrm { s r c } }$ are month and source embedding tables, $\mathrm { P E _ { d a y } }$ is a sine/cosine position encoding, $t _ { \ell } - t _ { \mathrm { m i n } }$ is the token’s date index ofset to the earliest token in the sample, $\mathbf { c } _ { i } ^ { a }$ contains normalized coordinates relative to the most-downstream sampled reach, $\mathbf { c } _ { i } ^ { g }$ contains absolute geographic coordinates, and $\mathbf { b } _ { i }$ is the local branch path from reach i to the sampled downstream root. The source embedding distinguishes measurements produced by diferent altimeters. Metadata are added to model inputs before every layer: $\mathbf { e } _ { \ell } \gets \mathbf { e } _ { \ell } + \mathbf { p } _ { \ell }$

![](images/d3bdba3d8d29d23b2c97a37e93456396b57879b9ba2fb30fe1daa61441b06909.jpg)  
Figure 4: Model architecture . During training, sparse WSE observations (colored) are sampled and part of these are masked. Learned query tokens (grey) and temporal, satellite, coordinate, and river-tree metadata are added. Tokens are processed by N bidirectional selective-SSM layers, and mapped to normalized WSE by satellite-specific heads. In inference, the masking is replaced by the introduction of query tokens carrying the metadata of what is to be predicted.

Each sampled location additionally contributes a static token encoding its mean WSE w.r.t. the most-downstream reach in the sample: $v _ { i } ^ { \mathrm { m e a n } } = \mathrm { N o r m } _ { \beta } ( \mu _ { i } - \mu _ { i _ { * } } )$ , where $i _ { * }$ is the most-downstream sampled reach. Static metadata tokens form a prefix. Dynamic tokens are then sorted so that they are ordered first by day and then along the river topology according to downstream flow rank.

We approximate a local subgraph with a tree and encode the position ofreach i by the sequence ofbranch choices $\mathbf { b } _ { i } =$ $\left( b _ { i , 0 } , \ldots , b _ { i , K - 1 } \right)$ that separate i from the local downstream root following Shiv and Quirk (2019). Let $\mathbf { u } _ { i , k } \in \{ 0 , 1 \} ^ { B }$ be the one-hot encoding of branch choice $b _ { i , k } ,$ , where $B = 3$ is the branching factor and $\mathbf { u } _ { i , k } = \mathbf { 0 }$ for padded path positions. $\mathbf { A }$ decay rate applied to depth k is computed from learned weights $w \in \mathbb { R } ^ { \mathbf { \hat { F } } }$ , for channel $f = 1 , \ldots , F \colon$

$$
[ \mathbf { g } _ { k } ] _ { f } = \rho _ { f } ^ { k } \sqrt { \frac { F } { 2 } \left( 1 - \rho _ { f } ^ { 2 } \right) } , \quad \rho _ { f } = \operatorname { t a n h } ( w _ { f } )\tag{3}
$$

The tree encoding is obtained by assigning this weight vector to the branch selected at each depth:

$$
\mathrm { T r e e P E } ( { \bf b } _ { i } ) = { \bf W } _ { \mathrm { t r e e } } \mathrm { v e c } \left( \mathrm { c o n c a t } _ { k = 0 } ^ { K - 1 } { \bf u } _ { i , k } { \bf g } _ { k } ^ { \top } \right) .\tag{4}
$$

Each path depth therefore activates one branch-specific block of $F$ components. The learned geometric decay rates allow diferent channels to emphasize diferent parts of the path, while shared path segments receive the same encoding. Learned weights $\mathbf { W } _ { \mathrm { t r e e } }$ project the encoding from $K \cdot B \cdot { \bar { F } }$ dimensions to the model dimension.

These encodings contain no learned lookup indexed by node index (SWORD reach). The same functions encode coordinates, relative branch paths, source identities, and scalar

![](images/740197b7b2c0cd3e540b349f699bc8860a43066c0c72a2b1aef7f2347e5ce9cc.jpg)  
Figure 5: Construction of inference sample that is input to our model (top) vs. to graph based methods for imputation like SPIN (Marisca, Cini, and Alippi 2022) or KITS (Xu et al. 2025). Learned query tokens are added to sparse conditioning inputs to inform the model of what to decode. Graph based methods add large amounts of query tokens (grey) to obtain a full spatiotemporal grid, while we add tokens only for a single location, but still using context from diferent locations.

WSE metadata at every location. The model is therefore inductive to reaches not observed during training, provided their SWORD topology and static metadata are available.

## Bidirectional Mamba

The ordered sequence is processed using standard Mamba blocks (Gu and Dao 2023). At its core, Mamba applies an input-dependent state-space recurrence

$$
\begin{array} { r } { \mathbf s _ { \ell } = \overline { { \mathbf { A } } } _ { \ell } \mathbf s _ { \ell - 1 } + \overline { { \mathbf { B } } } _ { \ell } \mathbf { x } _ { \ell } , } \end{array}\tag{5}
$$

$$
\mathbf { y } _ { \ell } = \mathbf { C } _ { \ell } \mathbf { s } _ { \ell } + \mathbf { D } \mathbf { x } _ { \ell } ,\tag{6}
$$

where the discretization and the maps $\overline { { \mathbf { B } } } _ { \ell }$ and $\mathbf { C } _ { \ell }$ depend on the current input $\mathbf { x } _ { \ell }$ . This selectivity allows the model to retain or discard information as it scans the sequence.

Each residual block has the form

$$
\mathrm { M B l o c k } ( \mathbf { X } ) = \mathbf { X } + \mathrm { M a m b a } ( \mathrm { L N } ( \mathbf { X } ) ) .\tag{7}
$$

The standard formulation of Mamba is 1-directional, but we combine a forward and backward scan, as proposed by Zhu et al. (2024), but in our case to integrate past information from upstream nodes with future information from downstream nodes. After restoring the reverse output to the original order, both directions are fused:

$$
\mathbf { H } ^ { ( k ) } = \mathbf { W } _ { \mathrm { b i } } \left[ \mathbf { H } _ { \right. } ^ { ( k ) } \parallel \mathrm { r e v } _ { \mathrm { d y n } } \left( \mathbf { H } _ { \left. \right)} ^ { ( k ) }  \right] .\tag{8}
$$

Only the dynamic sequence is reversed; the static metadata prefix remains fixed. The resulting model uses observations on both sides of a query and is therefore a non-causal reconstruction model rather than a forecaster. A source-specific linear head maps each final token representation to normalized WSE prediction $\widehat { z } _ { \ell }$

<table><tr><td colspan="3">RMSE↓</td></tr><tr><td>Model</td><td>&gt;2023/07</td><td>&lt;2022/06</td></tr><tr><td>Transductive</td><td></td><td></td></tr><tr><td>Temp. LSTM</td><td>sub 1.47</td><td>2.47</td></tr><tr><td>GRIN</td><td>full 0.88</td><td>1.28</td></tr><tr><td>SPIN-H</td><td>sub 2.52</td><td>2.25</td></tr><tr><td>full sub</td><td>0.93</td><td>1.58</td></tr><tr><td>full</td><td>0.74</td><td>0.94</td></tr><tr><td>ImputeFormer</td><td>1.69</td><td>1.56</td></tr><tr><td>sub</td><td>0.67</td><td>0.91</td></tr><tr><td>Inductive</td><td></td><td></td></tr><tr><td>kNN</td><td>full 1.77</td><td>2.24</td></tr><tr><td>IGNNK</td><td>full 2.33</td><td>2.38</td></tr><tr><td></td><td>sub 1.11</td><td>1.34</td></tr><tr><td>KITS full</td><td>2.38</td><td>2.40</td></tr><tr><td>sub</td><td>1.23</td><td>1.61</td></tr><tr><td>Ours 20</td><td>sub 0.62</td><td>0.84</td></tr></table>

Table 3: Baseline comparison on held-out in situ gauges for the 2.7% (>2023/07) and 0.6% (<2022/06) sparsity regimes. sub: subgraph sampling andfull: full graph in each sample.

## Training

Training combines location masking, which hides complete source-location time series to teach spatial reconstruction at unseen locations, and random masking, which hides individual observations to teach temporal densification. Static metadata remain visible. If Ω is the set of masked, non-padding observations in a minibatch, we minimize

$$
\mathcal { L } ( \theta ) = \frac { 1 } { \left| \Omega \right| } \sum _ { \ell \in \Omega } \left( \widehat { z } _ { \ell } - z _ { \ell } \right) ^ { 2 } .\tag{9}
$$

## Inference

To predict reach i over interval I, we use it as anchor a = i to construct a subgraph sample as described above, and we add i as virtual location containing exactly one masked query token (i, t) for each day t ∈ I. No daily queries are introduced for the other reaches in the sampled neighborhood: their sparse satellite measurements remain the observed context,

$$
\mathcal { D } _ { i , I } ^ { \mathrm { i n f } } = \mathcal { D } _ { i , I } \cup \left\{ ( i , t , \mathrm { m a s k e d } ) : t \in I \right\} .\tag{10}
$$

Predicting only the target reach prevents the sequence from being dominated by virtual tokens and maintains a favorable ratio of observed context to queries even under extreme graph sparsity (illustrated in Figure 5). Each window is reconstructed independently, without feeding predictions back into the model, and we average overlapping windows.

## Experiments

Hyperparameters are listed in Appendix A.4. Our model has 3.3M parameters and takes 2 h to train with less than 4 GB of GPU memory on one H100, predicting all 19K nodes takes another 1,5 h. Training data spans 2023-07–2026-05 and combines SWOT with HydroWeb and ICESat-2 observations. For our training and for baselines, time series with less than 10 (for ICESat-2) or 20 (for the remaining sources) values are discarded, reducing the number of observed ICESat-2 locations from 18.5K to 10K (the data are kept in the published dataset for subsequent works to use as they see fit). With ∼75% of reaches observed by the remaining time series, our task involves both transduction and induction, even though 95% of the evaluation gauges are located on a reach that is observed (this can be expected to slightly disadvantage transductive methods). We evaluate in two temporal settings:

![](images/6d6916d2d168d32abc41e82138b06cfb0b44693def9103117fe2776cf2def9d6.jpg)  
Figure 6: Ours vs. baseline RMSE as a function of the number of observations in the subgraph neighborhood per month.

1. SWOT period (2023-07 to 2026-05): The sparsity of this period is lower due to the presence of SWOT measurements (in both training and inference): 2.7% observed.

2. Hindcast (2016-01 to 2022-06-30): the model must extrapolate backward in time using only classical altimetry from HydroWeb and ICESat-2 observing 0.6% of day– reach slots, without any SWOT data.

We keep gauge data from the year 2022-07 to 2023-07 as evaluation setting for ablations (not for early stopping). The model never sees in situ data during training, preventing leakage also when the training and inference periods coincide.

Baselines. We select baselines to cover complementary settings and model families. A per-reach bidirectional LSTM inspired by BRITS-I (Cao et al. 2018) and spatial kNN isolate temporal modeling and non-parametric spatiotemporal interpolation. Among transductive imputers, GRIN (Cini, Marisca, and Alippi 2022) represents recurrent graph message passing, while SPIN-H (Marisca, Cini, and Alippi 2022) and ImputeFormer (Nie et al. 2024) represent attention-based approaches and demonstrated high (95%) point sparsity. IGNNK (Wu et al. 2021) and KITS (Xu et al. 2025) test inductive reconstruction at unseen reaches; KITS is especially relevant because it addresses the gap between sparse training graphs and virtual targets at inference. Some baselines natively support multivariate inputs; for those that do not, we add separate source channels, compare them with merging observations into one WSE series, and report whichever performs best. We run graph baselines both on the full graph and on the same sampled subgraphs as our model. We also compare with Reach-Reg (Halicki et al. 2026), the domainspecific baseline for SWOT WSE densification, which we fit on SWOT RiverSP data to make predictions for our gauges. Appendix A.5 describes baseline details.

<table><tr><td colspan="2">Configuration</td><td>RMSE↓</td></tr><tr><td>Base</td><td></td><td>0.56</td></tr><tr><td rowspan="3">Metadata enc</td><td>No tree encoding</td><td>0.58</td></tr><tr><td>No satellite enc</td><td>0.58</td></tr><tr><td>No mean WSE tokens</td><td>0.58</td></tr><tr><td>Data sources</td><td>No ICESat-2</td><td>0.59</td></tr><tr><td rowspan="2">Token order</td><td>No SWOT &amp; ICEsat-2</td><td>0.60</td></tr><tr><td>Flow</td><td>0.57</td></tr><tr><td></td><td>Random</td><td>0.64</td></tr><tr><td>Architecture</td><td>1-directional</td><td>0.72</td></tr><tr><td>&amp; inputs</td><td>Isolated</td><td>2.58</td></tr></table>

Table 4: Ablation study (validation period, mean over heldout gauges). Base has all metadata encodings and tokens, uses all 3 data sources in training and inference (when available), sees input sequences sorted by time, and is 2-directional.

## Results

Baseline Comparison. Table 3 compares our model with existing ML methods; we outperform all methods in all settings. It also shows that baselines have a larger gap between performance on SWOT period imputation where 2.7% of days have an input measurement and on the Hindcast setting where only $0 . { \dot { 6 } } \%$ is observed. SPIN-H and ImputeFormer with subgraphs reach good accuracy, which is in line with their reported improved results on sparse settings compared to other models. Except for GRIN, subgraph sampling outperforms handling the full graph at once, we attribute this to our larger graph compared to those of existing datasets. The transductive baselines do better than inductive baselines (but not than our model which is also inductive). Figure 6 shows performance of our model vs. ImputeFormer and SPIN-H, in function of number of the average amount of context tokens available in the subgraph (colored tokens in Figure 5), where baseline performance deteriorates faster when less conditioning observations are available. These results support the hypothesis that sequence models outperform GNN-based approaches under extreme spatiotemporal sparsity.

Ablations. Table 4 presents ablation experiments on metadata encoding and data source inclusion. In contrast to Kirschstein and Sun (2024), we find that graph topology and spatial context do help: the metadata encodings as wel as tokens being ordered by time or flow improve predictions slightly and the subgraph model (Base) significantly outperforms the Isolated model (that sees only 1 node timeseries). All data sources (SWOT, ICESat-2 and HydroWeb) contribute to better performance. Figure 7 shows a comparison of time window size and the spatial neighborhood size that the subgraph is sampled from: a neighborhood size of 300- 1000 km works best and performance is robust to temporal window size, with 3 months giving the lowest RMSE.

Comparison with Reach-Reg. Table 5 compares our model against Reach-Reg and ImputeFormer on overlapping in situ gauges across both evaluation periods. Kling– Gupta Eficiency (Gupta et al. 2009) is defined as KGE =

<table><tr><td rowspan="2">Period</td><td rowspan="2"></td><td colspan="3">RMSE↓</td><td colspan="3">KGE↑</td></tr><tr><td>Cov.</td><td>RR IF</td><td>20</td><td>RR</td><td>IF</td><td>2</td></tr><tr><td>SWOT era</td><td>184/284</td><td>0.90</td><td>0.61</td><td>0.55</td><td>0.85</td><td>0.92</td><td>0.94</td></tr><tr><td>Hindcast</td><td>149/250</td><td>0.85</td><td>0.80</td><td>0.70</td><td>0.88</td><td>0.88</td><td>0.91</td></tr></table>

Table 5: Comparison with Reach-Reg (RR) and Impute-Former (IF). Mean metrics are computed on gauges overlapping with Reach-Reg; Cov. reports RR/total gauge coverage. Scores difer from other tables because we evaluate only on gauges for which Reach-Reg makes a prediction.

![](images/2614e0c1f71719ebd1b6b8db538937e5374902da8320ad4baf0e08f3c7fb71c1.jpg)  
Figure 7: Ablation of sampled subgraph size limits.

$1 - \sqrt { ( r - 1 ) ^ { 2 } + ( \alpha - 1 ) ^ { 2 } + ( \beta - 1 ) ^ { 2 } }$ , where r is the Pearson correlation between predictions and observations, α = $\sigma _ { \mathrm { p r e d } } / \sigma _ { \mathrm { o b s } }$ measures variability, and $\beta = \mu _ { \mathrm { p r e d } } / \mu _ { \mathrm { o b s } }$ measures bias. It is widely used in Hydrology and jointly evaluates correlation, variability, and mean agreement, with a perfect score of 1 and higher values indicating better performance. Our model outperforms both methods on both periods and provides greater coverage than Reach-Reg. We evaluate Reach-Reg for making hindcasts, even though it was originally developed only to make predictions for the SWOT-era (details in Appendix A.5). As a state-of-the-art hydrology method, it sets a level of accuracy along with a coverage baseline that new methods should aim to meet or surpass on the AmazonWSE benchmark. Visualizations and comparisons of predictions are provided in Appendix A.6.

## Conclusion

We introduced AmazonWSE, a benchmark dataset for spatiotemporal graph imputation on time series of water levels observed satellite altimetry, covering the Amazon river basin, with 19K river reaches as nodes, 99% sparsity, and a directed acyclic graph topology. We showed that prior spatiotemporal graph imputation methods are not adapted to this scale and sparsity, and proposed a bidirectional selective state space model that outperforms them by sampling connected subgraphs and flattening space and time into a single token sequence. Our model also outperforms a state-of-theart non-neural baseline for SWOT-based WSE densification and produces denser reconstructions. We hope that these results inspire the community to develop improved methods for spatiotemporal imputation in extremely sparse settings such as river monitoring from satellite altimetry.

## References

Abdalati, W.; Zwally, H. J.; Bindschadler, R.; Csatho, B.; Farrell, S. L.; Fricker, H. A.; Harding, D.; Kwok, R.; Lefsky, M.; Markus, T.; Marshak, A.; Neumann, T.; Palm, S.; Schutz, B.; Smith, B.; Spinhirne, J.; and Webb, C. 2010. The ICESat-2 Laser Altimetry Mission. Proceedings ofthe IEEE, 98(5): 735–751.

Acosta, C. M.; Herath, H. M. V. V.; Lim, J. Y.; Saha, A.; Rasnayaka, S.; and Marshall, L. 2025. DUALFloodGNN: Physics-informed Graph Neural Network for Operational Flood Modeling. arXiv:2512.23964.

Agencia Nacional de Aguas e Saneamento Basico (ANA). 2026. HidroWeb Web Service API. https://www.ana.gov.br/ hidrowebservice/swagger-ui. Accessed: February 5, 2026.

Altenau, E. H.; Pavelsky, T. M.; Durand, M. T.; Yang, X.; Frasson, R. P. D. M.; and Bendezu, L. 2021. The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A Global River Network for Satellite Data Products. Water Resources Research, 57(7): e2021WR030054.

Andreadis, K. M.; Coss, S. P.; Durand, M.; Gleason, C. J.; Simmons, T. T.; Tebaldi, N.; et al. 2025. A First Look at River Discharge Estimation from SWOT Satellite Observations. Geophysical Research Letters, 52(9): e2024GL114185.

Biancamaria, S.; Lettenmaier, D. P.; and Pavelsky, T. M. 2016. The SWOT Mission and Its Capabilities for Land Hydrology. Surveys in Geophysics, 37: 307–337.

Cao, W.; Wang, D.; Li, J.; Zhou, H.; Li, L.; and Li, Y. 2018. BRITS: Bidirectional Recurrent Imputation for Time Series. In Advances in Neural Information Processing Systems, volume 31, 6775–6785.

Cheng, S.; Osman, N.; Qu, S.; and Ballan, L. 2024. Fast-STI: A fast conditional pseudo numerical difusion model for spatio-temporal trafic data imputation. IEEE Transactions on Intelligent Transportation Systems, 25(12): 20547–20560.

Cini, A.; Marisca, I.; and Alippi, C. 2022. Filling the G\_ap\_s: Multivariate Time Series Imputation by Graph Neural Networks. In International Conference on Learning Representations.

Cini, A.; Marisca, I.; Bianchi, F. M.; and Alippi, C. 2023. Scalable spatiotemporal graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 7218–7226.

Commission for Energy Regulation. 2012. CER smart metering project-electricity customer behaviour trial, 2009-2010 [Dataset]. https://github.com/wwzjustin/CER-Smart-Meter-Project-by-Irish-Social-Science-Data-Archive. Accessed: February 5, 2026.

Crétaux, J.-F.; and Calmant, S. 2015. Hauteur des lacs et des rivières (HYDROWEB) [Dataset]. https://doi.org/ 10.24400/329360/HYDROWEB\_WATER\_LEVEL. Accessed: February 5, 2026.

De Felice, G.; Cini, A.; Zambon, D.; Gusev, V.; and Alippi, C. 2024. Graph-based virtual sensing from sparse and partial multivariate observations. In International Conference on Learning Representations, 17111–17132.

Dufourg, C.; Pelletier, C.; May, S.; and Lefèvre, S. 2024. Forecasting water resources from satellite image time series using a graph-based learning strategy. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48: 81–88.

Fassoni-Andrade, A. C.; Fleischmann, A. S.; Papa, F.; Paiva, R. C. D.; Wongchuig, S.; Melack, J. M.; et al. 2021. Amazon Hydrology from Space: Scientific Advances and Future Challenges. Reviews ofGeophysics, 59(4): e2020RG000728.

Gu, A.; and Dao, T. 2023. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

Gupta, H. V.; Kling, H.; Yilmaz, K. K.; and Martinez, G. F. 2009. Decomposition of the Mean Squared Error and NSE Performance Criteria: Implications for Improving Hydrological Modelling. Journal ofHydrology, 377(1–2): 80–91.

Halicki, M.; Niedzielski, T.; Schwatke, C.; Scherer, D.; and Dettmering, D. 2026. Daily River Water Levels from Multi-Mission Altimetry: A Reach-Based Regression Method Using the Unique SWOT Data Geometry. Journal ofHydrology, 673: 135367.

Hummon, M.; Ibanez, E.; Brinkman, G.; and Lew, D. 2012. Sub-hour solar data for power system modeling from static spatial variability analysis. Technical report, National Renewable Energy Laboratory (NREL), Golden, CO (United States).

Jasinski, M.; Stoll, J.; Hancock, D.; Robbins, J.; and Nattala, J. 2025. ATLAS/ICESat-2 L3B Mean Inland Surface Water Data, Version 4 [Dataset]. https://doi.org/10.5067/ATLAS/ ATL22.004. Accessed: February 5, 2026.

Kirschstein, N.; and Sun, Y. 2024. The Merit of River Network Topology for Neural Flood Forecasting. In Proceedings of the 41st International Conference on Machine Learning, 24713–24725.

Klingler, C.; Schulz, K.; and Herrnegger, M. 2021. LamaH-CE: LArge-SaMple DAta for hydrology and environmental sciences for central Europe. Earth System Science Data, 13(9): 4529–4565.

Kratzert, F.; Nearing, G.; Addor, N.; Erickson, T.; Gauch, M.; Gilon, O.; Gudmundsson, L.; Hassidim, A.; Klotz, D.; Nevo, S.; et al. 2023. Caravan-A global community dataset for large-sample hydrology. Scientific Data, 10(1): 61.

Li, Y.; Yu, R.; Shahabi, C.; and Liu, Y. 2018. Difusion Convolutional Recurrent Neural Network: Data-Driven Trafic Forecasting. In International Conference on Learning Representations.

Li, Y.; Zezhi, S.; Yu, C.; Qian, T.; Zhang, Z.; Du, Y.; He, S.; Wang, F.; and Xu, Y. 2025. Sta-gann: A valid and generalizable spatio-temporal kriging approach. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 1726–1736.

Liang, Z.; Li, W.; Zhang, D.; Jia, Z.; Chen, Y.; Wang, Z.; Zheng, X.; and Youssef, M. 2026. DarkFarseer: Robust Spatio-Temporal Kriging Under Graph Sparsity and Noise. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 23451–23459.

Liu, M.; Huang, H.; Feng, H.; Sun, L.; Du, B.; and Fu, Y. 2023a. Pristi: A conditional difusion framework for spatiotemporal imputation. In Proceedings ofthe IEEE International Conference on Data Engineering (ICDE), 1927–1939.

Liu, X.; Xia, Y.; Liang, Y.; Hu, J.; Wang, Y.; Bai, L.; Huang, C.; Liu, Z.; Hooi, B.; and Zimmermann, R. 2023b. LargeST: A Benchmark Dataset for Large-Scale Trafic Forecasting. In Advances in Neural Information Processing Systems, volume 36, 75354–75371.

Marisca, I.; Cini, A.; and Alippi, C. 2022. Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations. In Advances in Neural Information Processing Systems, volume 35.

Nie, T.; Qin, G.; Ma, W.; Mei, Y.; and Sun, J. 2024. ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2260–2271.

Nielsen, K.; Zakharova, E.; Tarpanelli, A.; Andersen, O. B.; and Benveniste, J. 2022. River levels from multi mission altimetry, a statistical approach. Remote Sensing of Environment, 270: 112876.

Normandin, C.; Frappart, F.; Diepkilé, A. T.; Marieu, V.; Mougin, E.; Blarel, F.; et al. 2018. Evolution of the Performances of Radar Altimetry Missions from ERS-2 to Sentinel-3A over the Inner Niger Delta. Remote Sensing, 10(6): 833.

Pavlis, N. K.; Holmes, S. A.; Kenyon, S. C.; and Factor, J. K. 2012. The Development and Evaluation of the Earth Gravitational Model 2008 (EGM2008). Journal of Geophysical Research: Solid Earth, 117(B4): B04406.

Ren, X.; Zhao, K.; Taškova, K.; and Riddle, P. 2026. AnchorGK: Anchor-based Incremental and Stratified Graph Learning Framework for Inductive Spatio-Temporal Kriging. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1251–1262.

Santos da Silva, J.; Calmant, S.; Seyler, F.; Rotunno Filho, O. C.; Cochonneau, G.; and Mansur, W. J. 2010. Water Levels in the Amazon Basin Derived from the ERS-2 and ENVISAT Radar Altimetry Missions. Remote Sensing of Environment, 114(10): 2160–2181.

Schwatke, C.; Dettmering, D.; Bosch, W.; and Seitz, F. 2015. DAHITI–an innovative approach for estimating water level time series over inland waters using multi-mission satellite altimetry. Hydrology and Earth System Sciences, 19(10): 4345–4364.

Shiv, V.; and Quirk, C. 2019. Novel Positional Encodings to Enable Tree-Based Transformers. In Advances in Neural Information Processing Systems, volume 32.

Stein, G.; Shadaydeh, M.; Blunk, J.; Penzel, N.; and Denzler, J. 2025. CausalRivers – Scaling Up Benchmarking of Causal Discovery for Real-World Time-Series. In International Conference on Learning Representations.

SWOT. 2025. SWOT Level 2 River Single-Pass Vector Data Product [Dataset]. https://doi.org/10.5067/SWOT-RIVERSP-D. Accessed: February 5, 2026.

Taghizadeh, M.; Zandsalimi, Z.; Nabian, M. A.; Shafiee-Jood, M.; and Alemazkoor, N. 2025. Interpretable physics-informed graph neural networks for flood forecasting. Computer-Aided Civil and Infrastructure Engineering, 40(18): 2629–2649.

Tourian, M.; Tarpanelli, A.; Elmi, O.; Qin, T.; Brocca, L.; Moramarco, T.; and Sneeuw, N. 2016. Spatiotemporal densification of river water level time series by multimission satellite altimetry. Water Resources Research, 52(2): 1140– 1159.

Wu, Y.; Zhuang, D.; Labbe, A.; and Sun, L. 2021. Inductive graph neural networks for spatiotemporal kriging. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 4478–4485.

Xu, Q.; Long, C.; Li, Z.; Ruan, S.; Zhao, R.; and Li, Z. 2025. Kits: Inductive spatio-temporal kriging with increment training strategy. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, 12945–12953.

Yang, C.; Zhao, C.; Wang, C.; and Fan, J. 2025a. DRIK: Distribution-Robust Inductive Kriging without Information Leakage. arXiv:2509.23631.

Yang, X.; Sun, Y.; Chen, X.; Zhang, Y.; and Yuan, X. 2025b. Graph structure learning for spatial-temporal imputation: Adapting to node and feature scales. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 959– 967.

Yi, X.; Zheng, Y.; Zhang, J.; and Li, T. 2016. ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data. In Proceedings of the International Joint Conference on Artificial Intelligence, 2704–2710.

Zheng, C.; Fan, X.; Wang, C.; Qi, J.; Chen, C.; and Chen, L. 2023. Increase: Inductive graph representation learning for spatio-temporal kriging. In Proceedings ofthe ACM Web Conference, 673–683.

Zheng, Y.; Liu, F.; and Hsieh, H.-P. 2013. U-air: When urban air quality inference meets big data. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1436–1444.

Zhu, L.; Liao, B.; Zhang, Q.; Wang, X.; Liu, W.; and Wang, X. 2024. Vision Mamba: Eficient Visual Representation Learning with Bidirectional State Space Model. In International Conference on Machine Learning, 62429–62442. PMLR.

## A Appendix

## Overview of Supplementary Material

• This Appendix (Technical Supplement):

1. Dataset Sources

2. Data Preprocessing Details

3. Storage Format

4. Hyperparameters and Model Details

5. Baselines

6. Predictions

• Media Supplement: animated .gif files of 1) sparse ICESat-2 and HydroWeb inputs and 2) imputation prediction of daily WSE for all SWORD reaches by our model.

• Code and Data Supplement:

– Netcdf files for each source in the AmazonWSE dataset with a tiny amount of measurements: as an example of the storage format.

– Code of data export and preprocessing scripts.

– Code to run training and evaluation of our model and the baselines on the tiny included dataset.

## A.1 Dataset Sources

Figure 8 shows the temporal coverage of diferent sources in AmazonWSE and in HydroWeb. The next paragraphs provide more details about the sources when relevant.

SWORD. The Surface Water and Ocean Topography (SWOT) Mission River Database (v17b) was designed as a complementary static dataset to SWOT products and provides a topological graph of 19,172 river reaches in the Amazon basin (Altenau et al. 2021).<sup>2</sup>

ICESat-2. The NASA Ice, Cloud and land Elevation Satellite-2 (ICESat-2) mission carries the ATLAS (Advance Topographic Laser Altimeter System) instrument. Data is collected along three pairs of narrow laser beams that operate at 532nm wavelength. The beams in the pair are separated by a distance of around 90m, while each pair is separated by the next by a distance of 3 km. The data in AmazonWSE are from the Level3B ATL22 Mean Inland Surface Water Data product version 4 (Jasinski et al. 2025), which reports mean along-track surface elevation for each beam for each transect across a water body. Our observations span Oct. 2018–Mar. 2026 because the product has been released up until March 2026 at the time of writing.

SWOT RiverSP. We use data from the “SWOT Level 2 River Single-Pass Vector Reach Data Product, Version D”, and of the science cycle that started in July, 2023 (SWOT 2025).

HydroWeb.next. We use the operational and research collections (Crétaux and Calmant 2015) and include data from the following satellites: the SWOT nadir altimeter (i.e., a diferent sensor on the same satellite than the In-SAR swath altimetry sensor used for the SWOT data source in AmazonWSE), Sentinel-3A/B, Sentinel-6A, JASON-2/3, SARAL.

## A.2 Data Preprocessing Details

This section describes the data processing in more detail. All timestamps are first converted from local time to UTC (if not already in UTC). Subsequently, all measurements are assigned to daily bins on the shared temporal grid.

A quality flag in the shared data files records missing, rejected (by quality filters), and retained observations.

SWOT. Our product-level screening draws on the criteria used by Andreadis et al. (2025) and Halicki et al. (2026), with additional criteria added by ourselves. RiverSP records are aggregated into 24-hour bins. WSE and width use the median within each reach–day, provided measurement standard deviations are aggregated accordingly, and quality bit fields use the maximum so that severe flags are preserved. Many filters use variables shared as part of the SWOT RiverSP products, the names of those variables are marked here by the monospaced font, like wse\_r\_u.

Altogether filters take away about 50% of measurements. This may seem like a lot, but examples shown in Figure 11 of unfiltered RiverSP SWOT time series along with collocated HydroWeb (“classical” altimetry) and in situ gauge time series can hopefully motivate the need for this extensive filtering.

## Filters that Apply to Reaches.

• River Width. We remove reaches with a static (mean) width in SWORD lower than 50 m. Andreadis et al. (2025) used an 80 m cutof for their early discharge validation; we instead use SWOT’s 50 m river-observation goal (Biancamaria, Lettenmaier, and Pavelsky 2016; Andreadis et al. 2025) to retain the mission’s intended lower-width range.

• Reach Identifier. SWORD assigns identifiers to each reach of the form “CBBBBBRRRRT”, where the ending “T” encodes a reach type. The entire identifier follows the Pfafstetter coding system where the characters encode the topology of the river network (Verdin and Verdin 1999). We remove reaches with $T \in \{ 4 , 5 , 6 \}$ , since they encode “dam or waterfall”, “unreliable topology”, or “ghost reach/node”.

## Filters that Apply to Individual Measurements.

• Cross-track Distance. Observations within 15 km crosstrack distance (x\_trk\_dist) of nadir ground track are removed, following Andreadis et al. (2025).

• Crossover Calibration and WSE Uncertainty. Observations with crossover-calibration quality flag xovr\_cal\_q > 1 or with the random component of their reported WSE uncertainty wse\_r\_u > 0.5 m are removed, following Andreadis et al. (2025). The wse\_u field provided in the RiverSP products (which estimates the standard deviation of the WSE measurement error of the SWOT sensor) is stored in our dataset’s netcdf files (after root-sum-square aggregation if a reach is observed more than once on a single day).

• Reach Quality Flags. Observations with reach quality flag reach\_q above 2 are removed, following Halicki et al. (2026). Unlike Halicki et al. (2026), who remove bitwise quality flags $\mathtt { r e a c h \_ q \_ b } > 2 0 9 7 1 5 2 ,$ we remove only reach\_q\_b > 8388608 which additionally allows observations flagged as coming from a lake, but still removes severe flags like outliers.

![](images/71805b0421438c00b1a2469d549f050fc8893586ba1f53cab5d40e8d19eba680.jpg)

![](images/9d1e853d62c3475e1f942657290b1d1b52236b3e5c39dfc507d1a17abcdb4f38.jpg)  
Figure 8: Left: temporal coverage ofsources in AmazonWSE. Right: temporal coverage ofsatellite sensors included in HydroWeb: the SWOT nadir altimeter (i.e., a diferent sensor than the swath altimetry sensor used for the SWOT data source in AmazonWSE), Sentinel-3A/B, Sentinel-6A, JASON-2/3, SARAL.

![](images/b8f63ebb2dda7706cf516846db0ffc92589cf58629050e02fa9ab3433c22a981.jpg)

![](images/67d172a614e582089f8558f95e7a66a3ba0021ada9c1f87db9c44c0fca83a77b.jpg)  
Figure 9: Examples of SWOT observations retained and rejected by the harmonic-residual (left). Observations retained and discarded by the rating curve filter that considers $\Delta \mathrm { w i d t h } / \Delta \mathrm { W S E }$ (right).

• Observed Precision. Instead ofapplying a fixed threshold on the fraction of dark pixels (pixels for which no or not enough backscattered signal was measured and for which hence no measurement could be derived that is fed into the RiverSP measurement calculations) like (Andreadis et al. 2025; Halicki et al. 2026), we estimate whether the observed water pixels support a target precision $p = 0 . 1$ m. We approximate observed-pixel errors to be unbiased, independent, and approximately equal-variance, and normally distributed. If $\mathbf { \dot { \phi } } _ { N _ { i , t } ^ { \mathrm { o b s } } } ^ { \mathrm { o b s } }$ is the number of observed water pixels and $u _ { i , t }$ is reported WSE uncertainty, an observa-

tion is retained only if:

$$
2 \frac { u _ { i , t } } { \sqrt { N _ { i , t } ^ { \mathrm { o b s } } } } \leq p
$$

$$
\begin{array} { c } { { N _ { i , t } ^ { \mathrm { o b s } } \geq 4 \left( \displaystyle \frac { u _ { i , t } } { p } \right) ^ { 2 } , } } \\ { { p = 0 . 1 \mathrm { m } . } } \end{array}
$$

Thus a larger reported uncertainty requires support from more observed pixels, where the factor of 2 approximates a two-sided 95% normal confidence interval. We crudely estimate $N _ { i , t } ^ { \mathrm { o b s } }$ using the resolution of the InSAR sensor $( \sim 7 . 5 \mathrm { ~ m ~ } \times 2 0 $ m and the static mean river width and reach length reported in SWORD (thereby assuming that the entire reach falls within one of the two 50 km wide observed swaths by the sensor).

• Deviation from Harmonics. For reaches with suficient observations, a two-harmonic seasonal model is fitted and WSE residuals beyond 4.5 standard deviations from the mean (residual) are removed (Figure 9, left).

• Rating Curve. A robust delta rating-curve filter additionally identifies inconsistent changes in river width and WSE as observed by SWOT (Figure 9, right; river width and the water level are bound by a monotonously increasing relationship determined physically by the shape of the river bed). We fit a regression line on pairs of ∆width, ∆WSE between subsequent observations of the same reach, then remove observations of which the ∆width, ∆WSE deviates more than 4.5 standard deviations from the regressed line, or observations of which the ∆width and ∆WSE have non-matching signs (allowing a noise bufer).

![](images/a62eab80a989b8eb812ba838fa882e9bc9d80634e1bc532227449033519c00d5.jpg)

![](images/8fcf2ee844f5982e11263816ba188c7b84f2b7948feb4196e685a4da6fbfd095.jpg)  
Figure 10: Interpolated SWOT mean and std fields that are used to de-normalize predictions.

• Pass Correction. We investigate the bias between overpasses with diferent IDs that observe the same reach and found that the bias between passes is minimal. We try to correct it by shifting the mean WSE observed by the same pass ID towards each other, but found that that it makes negligible diference.

• Mean/Std Field Interpolation. Because SWOT provides the densest coverage, we compute its per-location mean and standard deviation, and interpolate this linearly and spatially to obtain dense fields as shown in Figure 10. The interpolation uses inverse distance weighting and averages upstream branch contributions if there are several. We use these fields for de-normalization of predictions (since predictions may have to be made for locations where no measurements are available).

HydroWeb. The HydroWeb database already cleans and homogenizes measurements as described in (Santos da Silva et al. 2010; Normandin et al. 2018).

• Text files returned from the HydroWeb.next API<sup>3</sup> (for the operational and research collections, or HYDROWEB\_RIVERS\_OPE and

HYDROWEB\_RIVERS\_RESEARCH) with time series are parsed, heights encoded as missing or fill values are converted to consistent missing data indicators; otherwise, HydroWeb’s supplied measurements and uncertainties are preserved without an additional value-level outlier filter.

• Jason-2 Interleaved (J2N) is excluded because its Oct. 2016–May 2017 record is too short for stable statistics. SARAL and Jason-2 are retained, with normalization statistics estimated from 2012 onward rather than 2016 to use their longer records.

ICESat-2. The filters remove about 4% of measurements (of the subset of measurements that has been matched to a SWORD reach).

• Transect Matching. The ATL22 contains measurements per transect, which is a contiguous portion of one ICESat-2 beam crossing a water body; land interruptions such as islands can split one beam pass into several transects. Transects arejoined to their nearest SWORD reach. Reservoir observations and transects shorter than 50 m are excluded.

• Deviation from Harmonics. For reaches with at least 10 transects, a two-harmonic model is fitted to transect orthometric heights and observations with residual modified Z-scores above 4.5 are rejected.

• Transect Aggregation. Remaining transects are first aggregated within each reach-day and beam pass (source file and beam ID) using their median orthometric height. Reach-day WSE is the median of these beam-pass values, so a beam split into several transects does not receive extra weight.

• Uncertainties. We compute wse\_u as the leave-onebeam-pass-out jackknife standard uncertainty of the reach-day median. The ATL22 ht\_stdev field is provided as the median within-transect standard deviation of filtered short-segment heights, but it should not be interpreted as a standard error of the transect mean. The output also retains the slope-fit RMSE, transect counts, beam-pass counts, and distinct-beam counts.

![](images/be2061330854af36abe746a711b43116e33233e350f569afe4e9d7432bc56658.jpg)

![](images/4c99bab4ff7f9bfe7819c77505e4fec940474079f655b2417af32e7d01ee28f9.jpg)

![](images/9a15c405ce8edaaa4512f06adbff455e17d14dd2a6e6769bd477f9324816a1d2.jpg)

![](images/0d267b74ac455cda1893e2f8bd2ad83ff4ca4eaf70fc239cba6ef1432522a6c5.jpg)  
Figure 11: Unfiltered SWOT RiverSP measurements (in red) with time series of collocated in situ gauges (top) and HydroWeb locations (bottom). While for some reaches SWOT measurements follow HydroWeb/in situ gauges closely (left), for other reaches, SWOT measurements are very noisy or low quality (right).

![](images/62027bf632cf58733cecf0ef64976aec46d1c8728ad8ad3457d93e9a8c67b4ab.jpg)

![](images/256ac40edce66b2dd553340926fdca4b2f6ea0868ed86ce8ba73f94c7d915f40.jpg)  
Figure 12: ICESat-2 harmonics filter (left) and aggregation of transect measurements per reach per day with accepted and rejected measurements (right).

ANA in situ gauges Figure 13 show examples or in situ gauges before and after quality filters. 21% of the 1000 timeseries are rejected automatically (10% of all measurements), ofthe remaining 810, 589 have a SWORD reach match within 10 km, and of those, we retain 375 series.

• ANA Timestamps. are converted from Brazilian standard time (UTC-3), to UTC and binned to calendar days. Multiple observations within a day are represented by their median and converted from centimetres to meters.

• Uncertainty. As ANA provides no per-observation measurement uncertainty, we use the within-day population standard deviation of the contributing readings as a dataderived uncertainty measure.

• Minimum Amount and Quality. Stations must span at least two annual cycles (730 days) and contain at least 100 finite daily observations before filtering. Repeatedvalue runs longer than three observations are removed. Negative observations are retained because gauge stage is referenced to a local datum.

• Seasonality. Daily series are linearly interpolated only to fit an STL seasonal–trend decomposition; the original observations are retained for filtering. Outliers are identified iteratively, for up to five STL fits, using a modified z-score of the residuals with a threshold of 4.5.

• Manual Inspection. An expert hydrology scientist manually inspects the remaining 581 time series and discards those that contain clear faulty measurements: 375 time series remain.

• Datum Alignment. Satellite sources report WSE w.r.t. a reference geoid (Pavlis et al. 2012), i.e., the elevation w.r.t. an approximate sea level. On the other hand, the in situ time series are measured in reference to a local gauge datum, for instance w.r.t. the local river bed bottom, which does not include the terrain elevation. Hence, before comparison, in all evaluations, we follow common practice in hydrology (Halicki et al. 2026) to project the in situ measurements with a per-location fitted linear regression to the reference scale of the satellite data and/or predictions. This operation is considered part of our evaluation benchmark, will be part of published evaluation code, and should be replicated by future authors that use this benchmark dataset.

Matching to SWORD Reaches. Figure 14 shows matched and unmatched locations per source.

• HydroWeb virtual stations, ICESat-2 transects, and ANA gauges are matched to their nearest SWORD reach using a 10 km distance threshold (adapted from the 20 km threshold Halicki et al. (2026) uses for the Solimoes, which is part of the Amazon basin). Time series that are further than 10 km from the nearest SWORD reach are not used (though they are still included in the data files).

• SWOT is linked directly by the RiverSP reach identifier.

## A.3 Storage Format

Each source is stored in a netcdf file with h5netcdf as a CF 1.13 indexed-ragged time series (Eaton et al. 2025), which means they are stored along a location dimension and an observation dimension (rather than a time dimension), where time is an additional variable. This means only measured location–time pairs are stored, and missing (unobserved) location–time pairs do not take up storage space. Table 6 summarizes the shared schema. An example netcdf file per source is included in the Code and Data Supplement along with the submission.

Indexing and Validity. The index variable declares instance\_dimension=location. Locations are unique and sorted by location\_id; observations are sorted first by location and then by time. Duplicate (location\_id, time) pairs are forbidden. A quality flag of 0 marks a recorded but rejected value, a flag of 1 marks an accepted value, and a missing observation has no corresponding row.

Global Metadata. Each file records schema\_version=1.0, featureType=timeSeries, a non-empty source, and quality\_convention="quality\_flag: 0=rejected, 1=accepted".

Source-specific extensions. Sources may add variables without changing the two-dimensional organization:

• SWOT adds observation-level width, width\_u, slope, and reach\_q\_b, together with int32 cycle\_id and pass\_id.

• ICESat-2 adds float32 ht\_stdev and slope\_cross\_error (m), as well as int32 counts of transects, beams, and their quality categories.

• HydroWeb adds observation-level satellite, orbit, and retracking\_algorithm metadata.

• ANA adds the location-level string river\_name.

## A.4 Hyperparameters and Model Details

Table 7 gives an overview of the used hyperparameters. Importantly, since none of the models sees in situ data during training, and since our model makes source-aware predictions (i.e., with a separate linear decoding head per source), we make predictions as if for one of the training sources, also when we compare to in situ data. We de-normalize the prediction with the interpolated statistics from Figure 10: since the source that is being decoded might not have measurements hence also no statistics on the reach that is being predicted and statistics of the in situ gauge series should not be used.

We find that decoding as HydroWeb works best when comparing to in situ gauge data, presumably since this source has long records of clean data. This creates a discrepancy: we decode normalized HydroWeb predictions, and de-normalize them with interpolated SWOT statistics, but of all the tested combinations, this gave best results.

## A.5 Baselines

All baselines use the splits, masking conventions, and sampled-subgraph settings, and hyperparameters from Table 7 unless stated otherwise.

Subgraphs vs full graph. We distinguish fixed-graph and sampled-subgraph settings. Fixed graphs use a basin-wide time–node grid with stable reach identities, following the transductive protocols of GRIN, SPIN-H, and ImputeFormer (Cini, Marisca, and Alippi 2022; Marisca, Cini, and Alippi 2022; Nie et al. 2024). Sampled sub-graph windows instead contain the same local river neighborhoods as our method (cfr. Table 7 and Figure 3). SPIN-H and IGNNK explicitly support subgraph sampling (Marisca, Cini, and Alippi 2022; Wu et al. 2021), while for GRIN and ImputeFormer we run the subgraph version as adaptations rather than replications of their published protocols. KITS is closest to its published increment-training setting when run on a fixed graph with virtual nodes (Xu et al. 2025).

Source representation. We consider two representations of co-located sources (SWOT, HydroWeb, ICESat-2): the channels mode assigns each source a separate value (in a separate input channel), input mask, and target mask, preserving source identity and allowing one source to remain visible while another is held out. The merged mode first normalizes sources with common SWOT-derived reach statistics; after masking, visible source values are averaged into one scalar. Thus, held-out observations never enter the aggregate, but source identity and source-specific biases are removed.

Models are never trained on in situ data. At in situ gauges, we use the training source predictions and dense SWOTderived reach statistics. IGNNK and KITS use merged inputs because they are scalar-WSE models (Wu et al. 2021; Xu et al. 2025). For ImputeFormer, GRIN, and SPIN-H we implement channel variants, even though these methods were primarily evaluated with scalar targets (Nie et al. 2024; Cini, Marisca, and Alippi 2022; Marisca, Cini, and Alippi 2022). We also tested representing each source–location pair as a separate node (allowing thus several nodes for the same reach), but it gave worse performance.

Non-graph controls. The non-parametric kNN baseline uses 91-day windows, ten spatial neighbors, and equal spatial and temporal weights. It fits a dense field separately per input source and outputs the mean of the per-source predictions.

![](images/fd76bc7fce1ad93ee6e015f8da57001600524ae87c4ce23ea3a56acf250aba7c.jpg)  
Figure 13: Filtered (blue) and unfiltered (orange) in situ gauge time series: a clean time series (top, left), a faulty time series that shows sudden jumps that clearly do not represent natural variations in river WSE (top, right), and two time series where automatic filters removed outliers (bottom, left and right).

A temporal-only, one-layer bidirectional LSTM inspired by BRITS-I (Cao et al. 2018) uses the channels mode and 365- day per-reach windows, without graph information.

Graph-imputation models. For fixed-graph runs we compared 7-, 14-, and 28-day fixed windows (smaller than subgraph runs to fit in memory), as well as merged vs. channels mode, and chose the setting that gave best RMSE on validation samples. We had to reduce the hidden dimension size of some baselines to fit their training in our GPU memory. We reuse settings from the original publications and/or default settings in the published repositories where applicable.

GRIN (Cini, Marisca, and Alippi 2022) uses one recurrent graph-imputation layer, hidden size 64, kernel size 2, and decoder order 1. Its fixed and sampled variants use channels and windows of 28 and 91 days and feed-forward widths of 128 and 64, respectively. SPIN-H (Marisca, Cini, and Alippi 2022) uses five layers, hidden size 32, η = 3, and two message-passing layers. Its fixed variant uses 28-day merged inputs, latent size 128, and four heads; its sampled variant uses 91-day channels, latent size 64, and two heads. IGNNK (Wu et al. 2021) uses hidden size 192, first-order difusion, and merged windows of 28 days (fixed graph) or 91 days (sampled subgraph). Its graph convolutions reconstruct each time step from visible neighbors without explicit temporal dynamics. KITS (Xu et al. 2025) uses hidden size 64, a 0.3 virtual-node ratio, unit cycle-consistency weight, and merged windows of 7 days (fixed) or 91 days (sampled). Training-time virtual nodes are connected around random one-hop neighborhoods and removed before evaluation. ImputeFormer (Nie et al. 2024) uses input, node-embedding, and feed-forward dimensions of 64, 96, and 256; three layers; four temporal heads; rank 8; dropout 0.1; and Fourier-loss weight 0.01. Its fixed variant uses a 14-day window and its sampled subgraph variant uses a 91-day window, both with source channels.

Repositories. For GRIN, SPIN-H and the BRITS-I inspired bidirectional LSTM we use the implementations of torch-spatiotemporal.<sup>4</sup> ImputeFormer<sup>5</sup>, IGNNK<sup>6</sup> and KITS<sup>7</sup> are adapted from their published repositories.

Baselines that were not included. GgNet (De Felice et al. 2024) is excluded because it assumes dynamic covariates available at target reaches, unlike our sparse observations of the WSE target itself. Difusion imputers such as PriSTI and FastSTI (Liu et al. 2023a; Cheng et al. 2024) are deferred pending a matched, deterministic extreme-sparsity protocol. Forecasting-only methods are excluded because they provide neither an imputation objective nor a corresponding masking protocol.

Reach-Reg. Reach-Reg is a recent method that leverages the SWOT measurement geometry to fit a chain of linear regressions (orthogonal distance regressions) to propagate measurements between consecutive reaches. Subsequently, a time lag is estimated with the Manning formula for river velocity, and aggregation and interpolation then yield a daily

![](images/792bcaea44ced16470f7ea0153a29821e3c3b996236aa0250702fba5635bc72c.jpg)  
Figure 14: Spatial distribution of data sources in AmazonWSE, matched (kept) and unmatched (discarded) to SWORD reaches within 10 km (before quality filtering). From left to right: HydroWeb virtual stations, ICESat-2 transects, and ANA in situ gauges.

<table><tr><td>Variable</td><td>Type</td><td>Description</td></tr><tr><td colspan="3">Location variables (location)</td></tr><tr><td>location_id</td><td>int64</td><td>Unique site identifier (cf_role=timeseries_id)</td></tr><tr><td>location_quality_flag</td><td>int8</td><td>Site validity: 0 rejected, 1 accepted</td></tr><tr><td>latitude</td><td>float32</td><td>Latitude (degrees_north)</td></tr><tr><td>longitude</td><td>float32</td><td>Longitude (degrees_east)</td></tr><tr><td>sword_reach_id</td><td>int64</td><td>Associated SWORD reach identifier</td></tr><tr><td>reach_distance_m</td><td>float32</td><td>Distance to the associated reach (m)</td></tr><tr><td colspan="3">Observation variables (observation)</td></tr><tr><td>observation_location_index</td><td>int32</td><td>Index into the location dimension</td></tr><tr><td>time</td><td>datetime64[ns]</td><td>Observation time (CF convention)</td></tr><tr><td>wse</td><td>float32</td><td>Water-surface elevation (m)</td></tr><tr><td>wse_u</td><td>float32</td><td>Water-surface elevation uncertainty (m)</td></tr><tr><td>quality_flag</td><td>int8</td><td>Observation validity: 0 rejected, 1 accepted</td></tr></table>

Table 6: NetCDF schema shared by all data sources. Variables are grouped by their associated dimension.

WSE. The approach is applied either to combined SWOT, S3 and S6 measurements from the DAHITI dataset (Schwatke et al. 2015) or to SWOT RiverSP measurements only. We use the oficial Reach-Reg implementation.<sup>8</sup>

We group the 375 gauges into 107 unique river segments (sequences of river reaches formed by traversing downstream and upstream from the target gauge, choosing the upstream branch with highest accumulated flow). We run Reach-Reg with the settings of Halicki et al. (2026) on SWOT RiverSP data (version D, as our model and baselines) retrieved and filtered from the HydroCron API with the settings and filters proposed by Reach-Reg. We compare linear with Akima interpolation and find that it does not make a significant diference. A prediction is not made for all gauges, since Reach-Reg does not make a prediction when quality filters leave insuficient neighboring stations for propagation or when propagation and path-error filtering leave no valid predictions. The authors achieve a better accuracy when running Reach-Reg on timeseries from the DAHITI dataset, combining S3, S6 and SWOT (Schwatke et al. 2015), rather than on SWOT only, but since SWOT timeseries for only a small part of reaches in the Amazon have been included in DAHITI as of yet, this further reduces coverage by over %50.

We next adapt Reach-Reg to make hindcast predictions for the available gauges (i.e., for the period from 2016-01 to 2022-06-30 for which no SWOT measurements are available). We reuse the fitted orthogonal distance regression weights and Manning parameters that resulted from the prediction that was just described for SWOT RiverSP data. We then inject historical DAHITI observations into the matching (and fitted) SWOT-era station slots and propagate them according the fitted and chosen parameters. We increase the maximum cumulative ODR path-error, defined as the sum of the ODR RMSEs along a propagation path, from 10 to 50 m. We also increase the per-link ODR-RMSE threshold used to choose a direct link versus an alternative path from 0.2 to 1.0 m. We retain a propagated observation only when its cumulative path-error is below max(0.2 · WSE amplitude, 1.5 · median(cumulative path-error)). These relaxations accommodate the longer regression chains needed in the much sparser hindcast setting. It should be noted that this evaluation takes Reach-Reg out of the context it was developed for, and that results are hence only indicative.

<table><tr><td>Group</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Architecture</td><td>Temporal model</td><td>Mamba-1</td></tr><tr><td></td><td>Bidirectional; decoding order</td><td>true; time then flow</td></tr><tr><td></td><td>Embedding / SSM state dimension</td><td>192 / 16</td></tr><tr><td></td><td>Bidirectional SSM layers</td><td>3</td></tr><tr><td></td><td>Step rank; convolution width</td><td>12;4</td></tr><tr><td></td><td>Dropout rate</td><td>0.5</td></tr><tr><td></td><td>Metadata combination; reinjection</td><td>elementwise addition; true</td></tr><tr><td></td><td>Output heads</td><td>satellite-specific</td></tr><tr><td></td><td>Tree embedding dimension F (Shiv and Quirk 2019)</td><td>4</td></tr><tr><td>Optimization</td><td>Loss</td><td>MSE</td></tr><tr><td></td><td>Optimizer</td><td>AdamW (Loshchilov and Hutter 2019)</td></tr><tr><td></td><td>Learning rate; weight decay</td><td>10−4; 0.05</td></tr><tr><td></td><td>Batch size; maximum steps</td><td>16; 100,000</td></tr><tr><td></td><td>Warmup steps; gradient clipping</td><td>1,000; 1.0</td></tr><tr><td></td><td>Learning-rate schedule</td><td>reduce on plateau (factor 0.2, patience 5)</td></tr><tr><td></td><td>Validation interval; early-stopping patience</td><td>500 steps; 20 checks</td></tr><tr><td></td><td>Early-stopping metric</td><td>RMSE on location masked HydroWeb</td></tr><tr><td></td><td>Seed</td><td>43</td></tr><tr><td>Input subgraphs</td><td>Bin size; temporal bins</td><td>24 hours; 91 (days)</td></tr><tr><td></td><td>Minimum / maximum measurements</td><td>15 (training), 0 (eval) / 500</td></tr><tr><td></td><td>Maximum spatial distance</td><td>300 km</td></tr><tr><td></td><td>Upstream / downstream sampling</td><td>0.75 / 0.25</td></tr><tr><td></td><td>Main-upstream trunk proportion</td><td>0.33</td></tr><tr><td></td><td>Maximum upstream / downstream hops</td><td>30 /30</td></tr><tr><td></td><td>WSE normalization</td><td>per-location mean and standard deviation</td></tr><tr><td>Masking</td><td>Training strategies (probabilities)</td><td>location (0.9), random (0.1)</td></tr><tr><td></td><td>Mask ratio</td><td>0.66</td></tr></table>

Table 7: Hyperparameters used for the SSM model.

## A.6 Predictions

Figure 15 shows examples of predicted time series.

![](images/919026648d86339dcb8a734985bf57458b5b44e39314c3599b2793930ef7a8f4.jpg)

Station 965005 -- Reach 62265800021 [2016-01-01 -> 2022-06-30]  
![](images/1a9ace517df95ed97602b2f362aa195634a746041e68a6b2c6800089804f8613.jpg)  
Figure 15: Example predictions of our method compared to Reach-Reg on SWOT-era (top) and on hindcast (bottom, where Reach-Reg does not have enough input observations to predict the whole series). The dotted lines represent the in situ gauge series linearly projected to the datum of our predictions and of Reach-Reg.

## Appendix References

Eaton, B.; Gregory, J.; Drach, B.; Taylor, K.; Hankin, S.;

Caron, J.; Signell, R.; Bentley, P.; Rappa, G.; Höck, H.; Pam-

ment, A.; Juckes, M.; Raspaud, M.; Blower, J.; Horne, R.;

Whiteaker, T.; Blodgett, D.; Zender, C.; Lee, D.; Hassell, D.;

Snow, A. D.; Kölling, T.; Allured, D.; Jelenak, A.; Soerensen,

A. M.; Gaultier, L.; Herlédan, S.; Manzano, F.; Bärring, L.;

Barker, C.; Bartholomew, S. L.; Lavergne, T.; Lawrence, B.;

Massey, N.; Cofiño, A. S.; McGinnis, S.; and Laake, P. V. 2025. NetCDF Climate and Forecast (CF) Metadata Conventions. https://doi.org/10.5281/zenodo.14274886.

Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations.

Verdin, K. L.; and Verdin, J. P. 1999. A topological system for delineation and codification of the Earth’s river basins. Journal ofHydrology, 218(1-2): 1–12.