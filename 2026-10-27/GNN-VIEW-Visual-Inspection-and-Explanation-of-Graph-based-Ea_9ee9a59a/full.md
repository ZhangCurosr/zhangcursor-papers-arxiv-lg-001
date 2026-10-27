# GNN-VIEW: Visual Inspection and Explanation of Graph-based Earth-System Models

Alessandro Lucatelli<sup>∗</sup> Ferdinand Unterhuber Katharina Krenzer Mennatallah El-Assady Benedikt Soja Christina Humer<sup>∗</sup> Joel Oskarsson<sup>∗</sup> ETH Zürich, Switzerland <sup>∗</sup>{alucatelli|humerc|ojoel}@ethz.ch

## Abstract

Machine learning Earth-system models are becoming increasingly important, and so is their development. Graph-based models, one of the most prominent approaches, allow developers to constrain the learning process before training via graph connectivity. Yet, developing these models remains challenging: the graph is built algorithmically with little intuition about the final result beyond static plotting, which requires extra work and is often not detailed enough. Even after successful training, how the model arrives at forecasts remains obscure. We present GNN-VIEW, an interactive browser-based tool that supports the development of such systems. GNN-VIEW exposes the graph’s information pathways, allowing model developers to debug the graph structure before training. After training, GNN-VIEW uses attribution-based explainability to reveal which input features are used by the trained model for a given forecast. This helps developers understand why a model fails. We showcase the usefulness of GNN-VIEW in the construction stage of a graph for an ocean forecasting model, and in explaining poorly predicted precipitation of a trained weather forecasting model during Hurricane Laura.

## 1 Introduction

Machine learning (ML) has opened up new possibilities for making good use of Earth Observation (EO) data in Earth-system modeling. ML models can learn an internal representation of the Earth system either from reanalysis datasets [1, 2, 3] derived from EO data, or by directly ingesting observations to produce forecasts [4]. A particularly effective and popular modeling paradigm has been graph-based Earth-system models using Graph Neural Networks (GNNs). Their graph structure encodes the spatial connectivity of the Earth system and creates a spatial inductive bias that constrains the representational capability of the model before training. This is essential for model generalization, and improper graph initialization can degrade forecast capabilities [3, 2]. These models follow an encoder-processor-decoder structure: physical states live on an input grid, while the computation happens on a coarser mesh graph. Three sets of edges tie these together: grid-to-mesh edges (G2M) encode inputs into the mesh, mesh-to-mesh edges (M2M) propagate information through message passing, and mesh-to-grid edges (M2G) decode the state back into the grid points (Fig. 1b). GNNbased weather models have matched and often surpassed the forecast ability of operational numerical prediction at a fraction of the inference cost [3, 5], and are increasingly adopted operationally [6]. The same design and development challenges extend beyond atmospheric modeling to domains such as ocean [2, 7], wildfire [8], hydrology [9] and ice sheet [10] modeling.

Despite its key role in these models, validation of graph construction remains challenging. Developers algorithmically construct large-scale graphs in which structural anomalies hide in raw tensor files and can silently degrade performance. To assess graph construction, developers currently rely on custom static plotting of specific graph regions [11, 12, 2], which is not scalable and domain-dependent.

![](images/1eedfc78fcd9c5e9f2c987de76ad93f266a29698451e9256a68f3dae6c4e4608.jpg)  
Figure 1: GNN-VIEW follows a developer’s workflow. (a) Before training, it allows (b) exploration of the graph structure and its components (grid points, mesh nodes, edges), followed by (c) detailed inspection of the graph structure via (d-e) manual thresholding of graph statistics. (f) After model training, developers can get (i-k) attribution-based explanations for (g,h) predictions of interest to detect what lies behind model failure.

After successful graph initialization, the challenges continue in post-training. Assessing why a model produced a specific forecast is hard: aggregate metrics such as RMSE over a held-out period report how much and where a model fails, but not why the model made the inaccurate forecast. Understanding this is crucial for debugging the model and uncovering potential shortcut learning [13], ultimately enabling the development of more robust and trustworthy models.

In both development challenges, developers need the same: to see what aggregate metrics and raw tensors hide. Visual analytics is an established paradigm for gaining a better understanding of deep learning [14, 15, 16], and GNNs specifically [17]. These systems are model-agnostic and could in theory be applied to Earth-system models. Yet, they do not incorporate the geographic semantics needed here. The two closest works differ significantly: GNNLens [17] is for node classification tasks only, and CloudNine [18] addresses the data assimilation stage, rather than the forecast. Within the Earth-system GNN community, libraries such as Neural-LAM [19] and Anemoi [20] provide basic plotting utilities, yet they are insufficient to effectively inspect dense, multi-layer large-scale graphs.

Our contribution addresses this gap. We present GNN-VIEW<sup>1</sup>: an interactive visual tool that supports the full development pipeline of Earth-system GNNs, from graph inspection to forecast explanation, across domains. We demonstrate its usefulness in two scenarios: identifying construction flaws in an ocean forecasting graph and explaining a mispredicted field during Hurricane Laura.

## 2 GNN-VIEW

GNN-VIEW is organized around three modes that follow a developer’s workflow (Fig. 1) and analyzes model representational capabilities from different angles: Explore and Inspect describe the information pathways available in the graph; Explain measures how the trained model uses them.

Graph structure exploration GNN-VIEW allows exploration of the complete encoder-processordecoder graph over a projected global map via two coordinated modes: Explore and Inspect (Fig. 1a-e). The Explore mode shows the full graph (Fig. 1b), letting the developer build an initial understanding of the graph structure the model operates on. The Inspect mode enables precise navigation to find mis-specified graph elements. We characterize the local structure of a selected subgraph with three quantities: node in- and out-degree, and edge length (Fig. 1d-e). Developers can specify upper and lower thresholds for these statistics to highlight elements of potential interest in the map (Fig. 1c). Coupling statistical selection with geographic patterns enables efficient graph diagnostics.

Explaining model predictions GNN-VIEW supports post-training explainability through its Explain mode (Fig. 1f-k), with a two-stage workflow: the user first identifies a target prediction (time, variable, grid-point) of interest, then the tool attributes it back to its inputs (timesteps, variables, grid-points). Complementary error metrics resolved over time, space, and variables (Fig. 1g,h) guide developers toward periods, regions, and variables of poor performance to ultimately help with model debugging. When a target is selected, the user can explore which inputs contribute to the prediction with attribution-based XAI [21] (Fig. 1i). To trade off latency against exactness, two attribution methods are available: Input × Gradient (IxG) [22] for a fast qualitative overview and Integrated Gradients (IG) [23] for slower and more accurate attributions. Both methods produce signed scores per input feature: positive values indicate features that increase the model’s prediction above a baseline value, while negative values indicate those that decrease it. For a more detailed description of IG completeness and sanity checks [24], see App. D and App. C, respectively. Summing absolute attribution scores over space and normalizing by the total gives the attribution share of each input field: how much each input (timestep, variable) contributes to the total attribution.

degree  
81 of 286 nodes flagged  
![](images/6cc875bfabd93fa94f5e3aafbdc2431d0c6b61aa653324d4aed5b3c31642c5c3.jpg)

![](images/ab13787537dd8a212c38f96a4e7f81a748c9765a2d7e5348ce304e003ec8da04.jpg)

![](images/6b32adf2d62cccbaeaa308f22697d14d6768442c5e0931fbb5c80a3bc60baa4f.jpg)

![](images/2b1d67ec45f90e19d140e331339b68f0fdb4eb52c463755bc606cabe4b7b42a2.jpg)

![](images/20ca17d7ce678d752dbe82db4ff7e62f2bb7dfa0ec043e4e1558c25388219f28.jpg)  
Figure 2: GNN-VIEW Inspect mode, on the Njord ocean graph. (a-b) Grid points thresholded on out-degree in the encoder subgraph (G2M). (c-e) Mesh edges thresholded on length and mesh nodes thresholded on degree. All distributions span the full graph, not just the viewport in the figure.

Resulting attributions are extremely high-dimensional — one tensor (timesteps, variables, grid-points) for each target — and presenting them to the user in a meaningful way is challenging. We tackle this as follows: Attribution scores are shown on a heatmap (Fig. 1j) to highlight their spatial distribution. We show two of these side by side to compare spatial structures across input timesteps or variables. Additionally, a ranked list of input variables, ordered by attribution share (Fig. 1k), averages the spatial dimension out and informs about high-level model behavior over time and variables.

Technical setup GNN-VIEW operates on three independent entities, each loaded separately: Explore and Inspect only need the graph topology (Fig. 1a). Explain additionally needs the trained model and dataset (Fig. 1f). The graph topology can be uploaded in either Neural-LAM [19] or Anemoi [6, 20] format, for both flat [3, 6] and hierarchical graphs [25, 2]. No assumptions about mesh generation are made: any graph can be visualized regardless of how it was built. Tool latency scales linearly with the number of graph elements (App. B). The model is uploaded as a TorchScript .pt file. The tool ships with two default datasets that users can select: ERA5 [26] at 1.5° and 6-hourly resolution [5] for weather modeling, and an adaptation of GLORYS12 [27], re-gridded to 0.25° and with sea-ice added, for oceanography modeling. Local deployment is also possible (App. A.2).

## 3 Use cases: GNN-VIEW in action

Spotting flaws in a graph We showcase the use of GNN-VIEW for validating graph structures on the graph used for the Njord ocean forecasting model [2]. Its mesh is generated by spatially clustering sea surface grid points, so the number of clusters (i.e., mesh nodes) is a free parameter. Ocean graphs are challenging to build because of the irregular domain they operate on. This implies several, often competing, requirements: edges should not cross major land masses, yet still cover all water basins, and every sea grid point should be connected to the mesh [2, 28]. To this end, Njord’s developers cap edge lengths above a free threshold parameter and reconnect isolated elements to the mesh via a repair pass [2]. Setting the free parameters is iterative, time-consuming, and error-prone.

Fig. 2 illustrates how Inspect mode can help inform this process. By setting the threshold for outdegree to 0 in the G2M subgraph (Fig. 2b), developers can surface isolated points whose states are not encoded in the model (Fig. 2a), here undermining the correct representation of the Kara Sea. This leads to a more informed decision on whether to run a disconnection repair algorithm or increase the number of mesh nodes. In addition, by thresholding the mesh edge length and mesh node degree, two other failure modes can be seen: a mesh edge crosses over Wales, connecting water masses separated by land, and mesh nodes on the western shores of Denmark are isolated, leaving the Baltic Sea disconnected from the rest of the ocean (Fig. 2c-e). This informs developers about the edge length cap to use and the need to reconnect the nodes to the mesh. Fig. 2c contrasts two cutoffs: 250 km isolates the land-crossing edge, while 200 km also captures legitimate long-range links. GNN-VIEW makes this sensitivity visible to calibrate the threshold effectively.

(a)  
![](images/1b1151f94cad1b5349025e2f9289249fc20c5df15fecb6778868bc68d14e0559.jpg)

![](images/91c6c8df1ea72aa3b112d534f0b8a02ce6cca32f090bf5168b26679b86f3e2da.jpg)

(c)  
![](images/68b0596e53abe7bbdfe49f6838140ea6b7ee46649851f6ef10531f42666d1d9f.jpg)

(d)  
![](images/f2ae88b4aa76c486eef5b81269558b0f2beca079caab7ce8e71a9a1d4083f47a.jpg)

![](images/9c6f349c0ab8057f8d7fa4a7d09175eb3e4ff274b3e71babed1b2b3688b31f1f.jpg)

![](images/f063ccf29f9914c160d4b0fccdedd5baa40eb61f05e79af6e46fbe1dbc256c70.jpg)  
Figure 3: GNN-VIEW Explain mode on a GraphCast-style GNN. (Black square) Target node 31.5° N 93.0° W. XAI target field is (Left) Temperature at 925 hPa and (Right) Total 6-hourly Precipitation. (a,c) Target error: absolute standardized error heatmap of the target field. (b,d) Input attribution: signed IG attribution of the target node’s prediction w.r.t. that field at t−6 h, scaled to each panel’s own largest absolute attribution excluding the target node. (e,f) Ranked input fields: top-10 input features by IG attribution share.

Explaining a hurricane mis-prediction We illustrate the Explain mode using a GraphCast-style GNN [3, 25] trained on 1.5° ERA5 data, focusing on forecasts of Hurricane Laura in 2020. The target time is 2020-08-27 12:00 UTC, i.e. six hours after Hurricane Laura made landfall at Category 4 intensity. The model takes as input two past atmospheric states (00:00 and 06:00 UTC), for which the storm is at peak intensity. Hence it is asked to predict its collapse as it traverses from ocean to land.

In Fig. 3 we compare the Explain mode attribution patterns of the accurately predicted temperature at 925 hPa (Fig. 3a) with the poorly predicted 6-hour precipitation (Fig. 3c). For temperature, the attribution ranking (Fig. 3e) is dominated by a single input: the same field at t−6 h; and the attribution is compact and centered around the target point (Fig. 3b). The model uses the recent local state for the prediction. For precipitation, the pattern is qualitatively different. The attribution ranking is relatively flat (Fig. 3f), led by geopotential and temperature at higher altitudes, which characterize the surroundings rather than the storm itself. Known physical drivers of precipitation, such as moisture and winds, enter the ranking marginally or not at all. Precipitation itself at t−6 h does not enter the top ten, and its attribution map is weak and mixed in sign (Fig. 3d). One hypothesis consistent with this pattern is that the model at this coarse resolution falls back to correlating precipitation with larger-scale circulation patterns rather than drawing on relevant physical processes. This is exactly the kind of testable hypothesis GNN-VIEW is built to generate.

## 4 Summary and future work

Developing accurate graph-based Earth-system models with good representational capabilities requires the validation of the graph structure before training, and debugging failure modes of model predictions after training. GNN-VIEW is an interactive specialized visualization tool that brings these into developers’ view. Through two scenarios, a poorly connected ocean graph and a mispredicted precipitation field, we have showcased how this tool can give Earth-model developers actionable insights to debug and improve their models. In both use cases GNN-VIEW returns interactive evidence, not verdicts, enabling an interactive loop with the tool (Fig. 1a,f) that ultimately allows for a better and more transparent development. Some design choices behind GNN-VIEW come with limitations that can be explored in future work. Structure-aware explainers [29] could connect model behavior closer to the graph structure. Explainability of forecast rollouts [30] and probabilistic model ensembles would also broaden the tool’s impact in the Earth-system modeling community. Further evaluation of the tool’s impact on developers’ workflow through user studies would also be valuable.

## Acknowledgments and Disclosure of Funding

This research was supported by the ETH AI Center through ETH AI Center postdoctoral fellowships to Christina Humer and Joel Oskarsson and by the International Geneva AI Innovation Programme (IGAIP) with funding from the Swiss Foreign Ministry.

## References

[1] Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Allen, A., Brandstetter, J., Garvan, P., Riechert, M., Weyn, J. A., Dong, H., Gupta, J. K., Thambiratnam, K., Archibald, A. T., Wu, C.-C., Heider, E., et al.. “A foundation model for the Earth system”. Nature. 2025. DOI: 10.1038/s41586-025-09005-y.

[2] Holmberg, D., Oskarsson, J., Wikingsson, E., Lindsten, F., and Roos, T.. Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting. 2026. DOI: 10.48550/arXiv.2605.15470. http://arxiv.org/abs/2605.15470.

[3] Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., et al.. “Learning skillful medium-range global weather forecasting”. Science. 2023. DOI: 10.1126/science.adi2336.

[4] Alexe, M., Boucher, E., Lean, P., Pinnington, E., Laloyaux, P., McNally, A., Lang, S., Chantry, M., Burrows, C., Chrust, M., Pinault, F., Villeneuve, E., Bormann, N., and Healy, S.. Graph-DOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations. 2024. DOI: 10.48550/arXiv.2412.15687. http://arxiv.org/ abs/2412.15687.

[5] Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., Russell, T., Sanchez-Gonzalez, A., Yang, V., Carver, R., Agrawal, S., Chantry, M., Ben Bouallegue, Z., Dueben, P., Bromberg, C., Sisk, J., et al.. “WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models”. Journal of Advances in Modeling Earth Systems. 2024. DOI: 10.1029/2023MS004019.

[6] Lang, S., Alexe, M., Chantry, M., Dramsch, J., Pinault, F., Raoult, B., Clare, M. C. A., Lessig, C., Maier-Gerber, M., Magnusson, L., Bouallègue, Z. B., Nemesio, A. P., Dueben, P. D., Brown, A., Pappenberger, F., et al.. AIFS – ECMWF’s data-drivenforecasting system. 2024. DOI: 10.48550/arXiv.2406.01465. http://arxiv.org/abs/2406.01465.

[7] Holmberg, D., Clementi, E., Epicoco, I., and Roos, T.. “Accurate Mediterranean Sea forecasting via graph-based deep learning”. Scientific Reports. 2025. DOI: 10.1038/s41598-025-31177-w.

[8] Michail, D., Davalas, C., Chafis, K., et al.. “FireCastNet: earth-as-a-graph for seasonal fire prediction”. Scientific Reports. 2026. DOI: 10.1038/s41598-025-30645-7.

[9] Bentivoglio, R., Isufi, E., Jonkman, S. N., and Taormina, R.. “Multi-scale hydraulic graph neural networks for flood modelling”. Natural Hazards and Earth System Sciences. 2025. DOI: 10.5194/nhess-25-335-2025.

[10] Koo, Y. and Rahnemoonfar, M.. “Graph convolutional network as a fast statistical emulator for numerical ice sheet modeling”. Journal ofGlaciology. 2025. DOI: 10.1017/jog.2024.93.

[11] Graph Creator — Anemoi Graphs 0.1.dev103+gccae142 documentation. https://anemoi. readthedocs.io/projects/graphs/en/stable/modules/graph\_creator.html.

[12] Creating the graph — Graphs for data-driven weather models. https://mllam.github. io/weather-model-graphs/creating\_the\_graph.html.

[13] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A.. “Shortcut learning in deep neural networks”. Nature Machine Intelligence. 2020. DOI: 10.1038/s42256-020-00257-z.

[14] Hohman, F., Park, H., Robinson, C., and Chau, D. H.. “Summit: Scaling Deep Learning Interpretability by Visualizing Activation and Attribution Summarizations”. IEEE Transactions on Visualization and Computer Graphics. 2020. DOI: 10.1109/TVCG.2019.2934659.

[15] Kahng, M., Andrews, P. Y., Kalro, A., and Chau, D. H.. “ActiVis: Visual Exploration of Industry-Scale Deep Neural Network Models”. IEEE Transactions on Visualization and Computer Graphics. 2018. DOI: 10.1109/TVCG.2017.2744718.

[16] Spinner, T., Fürst, D., and El-Assady, M.. iNNspector: Visual, Interactive Deep Model Debugging. 2024. DOI: 10.48550/arXiv.2407.17998. http://arxiv.org/abs/2407.17998.

[17] Jin, Z., Wang, Y., Wang, Q., Ming, Y., Ma, T., and Qu, H.. “GNNLens: A Visual Analytics Approach for Prediction Error Diagnosis of Graph Neural Networks”. IEEE Transactions on Visualization and Computer Graphics. 2023. DOI: 10.1109/TVCG.2022.3148107.

[18] Jeon, H.-J., Kang, J.-h., Kwon, I.-H., and Lee, O.-J.. “Observation impact explanation in atmospheric state estimation using hierarchical message-passing graph neural networks”. Mach. Learn.: Sci. Technol.. 2024. DOI: 10.1088/2632-2153/ad8981.

[19] Oskarsson, J., Landelius, T., and Lindsten, F.. “Graph-based Neural Weather Prediction for Limited Area Modeling”. NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning. 2023. .

[20] ECMWF. anemoi-graphs: Graph construction for data-driven weather models. 2024. . https: //github.com/ecmwf/anemoi-core.

[21] Wang, Y., Zhang, T., Guo, X., and Shen, Z.. Gradient based Feature Attribution in Explainable AI: A Technical Review. 2024. DOI: 10.48550/arXiv.2403.10415. http://arxiv.org/abs/ 2403.10415.

[22] Shrikumar, A., Greenside, P., and Kundaje, A.. “Learning important features through propagating activation differences”. Proceedings ofthe 34th International Conference on Machine Learning - Volume 70. 2017. .

[23] Sundararajan, M., Taly, A., and Yan, Q.. “Axiomatic Attribution for Deep Networks”. Proceed ings ofthe 34th International Conference on Machine Learning. 2017. .

[24] Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B.. “Sanity checks for saliency maps”. Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018. .

[25] Oskarsson, J., Landelius, T., Deisenroth, M. P., and Lindsten, F.. “Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks”. Advances in Neural Information Processing Systems. 2024. .

[26] Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., et al.. “The ERA5 global reanalysis”. Quarterly Journal of the Royal Meteorological Society. 2020. DOI: 10.1002/qj.3803.

[27] El Aouni, A., Gaudel, Q., Johnson, J. E., Charly, R., Le Sommer, J., Gennip, S. van, Fablet, R., Drevillon, M., Drillet, Y., and Le Traon, P.. “OceanBench: A Benchmark for Data-Driven Global Ocean Forecasting systems”. Advances in Neural Information Processing Systems. 2026. DOI: 10.52202/085713-0303.

[28] Cuervo-Londoño, G. A., Reyes, J. G., Rodríguez-Santana, Á., and Sánchez, J.. “Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware Meshes in Graph Neural Networks for Sea Surface Temperature Forecasting”. Electronics. 2025. DOI: 10.3390/elec tronics14244841.

[29] Luo, D., Cheng, W., Xu, D., Yu, W., Zong, B., Chen, H., and Zhang, X.. “Parameterized explainer for graph neural network”. Proceedings of the 34th International Conference on Neural Information Processing Systems. 2020. .

[30] Essafouri, Y., Raynaud, L., Drozda, L., and Risser, L.. Explanation of Dynamic Physical Field Predictions using WassersteinGrad: Application to Autoregressive Weather Forecasting. 2026. DOI: 10.48550/arXiv.2604.22580. http://arxiv.org/abs/2604.22580.

[31] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., and Samek, W.. “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation”. PLOS ONE. 2015. DOI: 10.1371/journal.pone.0130140.

[32] Samek, W., Binder, A., Montavon, G., Lapuschkin, S., and Müller, K.-R.. “Evaluating the Visualization of What a Deep Neural Network Has Learned”. IEEE Transactions on Neural Networks and Learning Systems. 2017. DOI: 10.1109/TNNLS.2016.2599820.

[33] Petsiuk, V., Das, A., and Saenko, K.. RISE: Randomized Input Samplingfor Explanation of Black-box Models. 2018. DOI: 10.48550/arXiv.1806.07421. http://arxiv.org/abs/1806. 07421.

[34] Hedström, A., Weber, L., Lapuschkin, S., and Höhne, M. “A Fresh Look at Sanity Checks for Saliency Maps”. In: Explainable Artificial Intelligence. Ed. by L. Longo, S. Lapuschkin, and C. Seifert. Cham: Springer, 2024. DOI: 10.1007/978-3-031-63787-2\_21. https://doi.org/ 10.1007/978-3-031-63787-2\_21.

[35] Wickstrøm, K., Höhne, M., and Hedström, A. “From Flexibility to Manipulation: The Slippery Slope of XAI Evaluation”. In: Computer Vision – ECCV 2024 Workshops. Ed. by A. Del Bue, C. Canton, J. Pont-Tuset, and T. Tommasi. Cham: Springer, 2025. DOI: 10.1007/978-3-031- 92648-8\_15. https://doi.org/10.1007/978-3-031-92648-8\_15.

[36] Bhatt, U., Weller, A., and Moura, J. M. F.. “Evaluating and aggregating feature-based model explanations”. Proceedings ofthe Twenty-Ninth International Joint Conference on Artificial Intelligence. 2020. DOI: 10.24963/ijcai.2020/417.

[37] Bommer, P. L., Kretschmer, M., Hedström, A., Bareeva, D., and Höhne, M. M.-C.. “Finding the Right XAI Method—A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science”. Artificial Intelligencefor the Earth Systems. 2024. DOI: 10.1175/AIES-D-23-0074.1.

## A GNN-VIEW details

## A.1 Single-step inference and evaluation metric

Given two consecutive normalized input states (t−12 h and t−6 h) and a forcing tensor, the model produces the state at t. The per-grid-point prediction error is computed on demand for the selected timestep and is rendered as the error heatmap. The single-step timeline summaries (mean and maximum RMSE, plus error concentration) are precomputed over the full test period, so the overview is available without re-running inference.

## A.2 Local deployment

The tool can be deployed locally, in which case users can point GNN-VIEW at their own datasets. Inference, error and attribution pipelines stay unchanged, since nothing downstream of loading assumes a particular dataset.

## B GNN-VIEW latency

We measured server latency on synthetic (Fig. 4a-c), as well as on real graphs (Fig. 4d), on a standard consumer laptop. Synthetic graphs have been generated by plotting an arbitrary number of grid-points, mesh nodes and edges. To measure individual contributions to latency, we increase the number of each one of these elements at a time while keeping the other two fixed (i.e. total edges increase in Fig. 4c, while grid points and mesh nodes are fixed).

Fig. 4a-c shows that latency scales linearly with the number of components and that it is dominated by edge rendering: 5.73 µs/element against the 3.71 and 4.19 µs/element of grid-points and mesh nodes. Edges dominate latency also because they are typically far more numerous than grid-points or mesh nodes in real graphs. For this reason Fig. 4d compares real graphs based on their total number of edges.

Fig. 4d shows the server latency for rendering the graph provided as defaults in GNN-VIEW, taken from recent literature in graph-based weather [25, 3] and ocean [2, 7] modeling. Latency scales linearly for real graphs too (Fig. 4d): an oceanography graph with approximately 22× more edges requires almost 20× more time.

Latency measured in Fig. 4 is the total latency to render the full graph on screen at once. Typically GNN-VIEW operates on much finer zoom levels where only a portion of interest of the graph is rendered at a time. Given the linear scaling of the latency, this results in a considerably lower latency in common use cases of GNN-VIEW.

## C Sanity Checks

We performed two types of sanity checks to quantify the relationship between explanations and the model: randomization of model parameters [24] and deletion tests [31, 32, 33]. This allows us to assess the “sensitivity” [34] of explanations to the model’s learned parameters and their “faithfulness” [35] to the model’s input-output behavior. While some literature refers to both of these properties as “faithfulness”, here we follow the naming proposed by [34, 35]. Both tests have been adapted to the graph-based regression task of interest in this paper, where inputs have a much larger extent than typical computer vision tasks, and the model has direct input-output skip-connections for the target’s own field at t−6 h. Following [34], we also compared all results against reference control condition rather than relying on the inspection of raw results.

All sanity checks have been performed for the Hurricane Laura use case (see Sec. 3) and the following target predictions:

S0 A pool of 50 targets chosen at random from the ERA5 test set (1 Jan 2020 to 10 Jan 2021).

S1 The 50 targets for which the model has the highest error over the ERA5 test set.

Due to normalization, the highest errors often occur for Vertical Velocity at 50 hPa. To have a representative pool of high-error targets across different variables, a capping mechanism was put

measured server latency linear fit

(a) 8k mesh nodes 300k edges  
![](images/97357182dd651a2549fa717e8273ff52008b8ea6051bcad910bd2f4e5879abd9.jpg)

![](images/eb73cfc6011025583fe064bf8ae2a166009d860d92794075ae62cb3f5aadd008.jpg)  
GraphCast-style (flat icosahedral)

![](images/589e39808b92ee222eaea5efda02a78e32bf39c115e6e9df78b2fdf830ebf21e.jpg)  
(d) GNN-VIEW default graphs  
Graph-EFM (hierarchical icosahedral)

![](images/fc5ec5cf1d2f34705190d590e99e9590cfdfebde022c4e4e946ff9449e7df89f.jpg)  
SeaCast-style (hierarchical icosahedral-masked)  
Njord (hierarchical cluster-based)

Figure 4: Server latency measured for (a-c) one component at a time, and (d) for real graphs. (a) Latency of rendering an increasing number of grid points; mesh nodes and edges count kept fixed. (b) Latency of rendering an increasing number of mesh nodes; grid-points and edge count kept fixed. (c) Latency of rendering an increasing number of edges; grid-point and mesh-node count kept fixed. (d) Latency of rendering the default graphs available in GNN-VIEW: GraphCast-style and Graph-EFM graphs for weather modeling from [3, 25] and SeaCast-style and Njord graphs for ocean modeling from [7, 2].

in place: if a maximum of 10 high-error targets is reached for a given variable family, all the other high-error targets for the same variable family are ignored. The Hurricane Laura use case provides a direct sanity check of what is shown in the main text; the other two provide more robust results. This approach is replicated for IG completeness experiments in App. D. For all of these experiments we used the GraphCast-like trained model from Sec. 3.

Most of the discussion in this appendix regarding randomization and deletion sanity checks focuses on IG explanations (Fig. 5, Fig. 7). However, sanity checks for IxG explanations have been performed as well (Fig. 6, Fig. 8).

## C.1 Randomization tests

We randomized all weights at once, rather than layer-by-layer as in [24]. The reason is to provide a general outlook of the sensitivity of the XAI techniques implemented in the tool, rather than assessing its layer-wise granularity. Model weights are randomized with 5 different seeds. For each target of interest, IG input attribution maps and rankings are computed for the GraphCast-like trained model (same as in Sec. 3) and the 5 randomized models.

We then define 4 paired tests to compare explanations: T-R) explanations of trained model against those of randomized model. R-R) explanations of random models compared across different seeds. T-X) Explanation of trained model against the input. R-X) Explanation of randomized model against the input. The first and third (T-R and T-X) answer the questions we care about, i.e., explanation sensitivity to model parameters (T-R) and their difference from inputs (T-X), while the other two (R-R and R-X) serve as a control. For each of these pairs two quantities are computed: Pearson’s correlation (ρ) between the attribution tensors, and Kendall’s correlation (τ) between the attribution rankings. Pearson’s correlation was chosen in place of Spearman correlation [24] because the large number of almost-zero entries in the attribution tensor, which is caused by the localized receptive fields of the graph-based model, would raise Spearman correlation in a meaningless way.

Fig. 5 shows randomization tests for the use case in Fig. 3. While previous literature [24] marks IG as poorly performing under randomization tests, it does perform fairly well in this specific context. Correlation of trained model’s explanation with the randomized model’s explanation (T-R in Fig. 5ad) is always lower than the similarity of explanations within different randomized models (R-R in Fig. 5a-d). This holds for both input attribution maps (Fig. 5a,c) and input rankings (Fig. 5b,d).

![](images/c6ea3ed437bcbe38aa9a1247f3bda791d9e8a259b95ecb8fa9da73da345745fd.jpg)

![](images/8e2664afa89b0d94d87d825b3240763b6e30398d31e3fe65ff97cc3368425903.jpg)

![](images/e6b2e1327cb1464dc5c5daa8dcce920060d4b339d563c502dbe8c36f5a9c97b6.jpg)

![](images/33f033ff0ec087e099e87aad1e95baba2f874bfd5462ea983d2b0671305ae6fb.jpg)

(e)  
![](images/18d1f899dd114196547623970fdc3671b1ee6af9718122834d21ebf2c327e052.jpg)

![](images/fe1459eb29e9b6bf843dbe10a66f4f285df60010c82095f656cffda8f23b743e.jpg)

(g)  
![](images/d2f24ac0edb7e6a733ed4b28306fac903364811d1a584aed48a6b20f67a1add6.jpg)

![](images/96e6351f7ea9168b43e1992c1062b257a422b67b2006685a664f87bd3e1bd10f.jpg)

![](images/ef6a1b94b4f187518c0764a72a3ac352504fbb23e3878658e59104ca58aa287a.jpg)

![](images/539c5b7035745b180a3a617527550a73c2104b66b0eb918767e71ffd47232bcb.jpg)

![](images/726d8cb6ad6529e547f6f86c1f377de750cc768410236870a7bce95a544aa78e.jpg)

![](images/e2d145dbfe36fe6f25c51efcf3097569570979fcb701c60a9a1d4990408be7c9.jpg)

![](images/1004a43e5c2062aff3ff297ce1336941a2fcd10ca1493a5612a7545eb498eda5.jpg)

![](images/d73a5583cf136f335bcd7f87a7736bb101fbaa498b138cf97bac411b63878beb.jpg)

![](images/061e949c1041bbbfb8898b04f268ea63a64db597ce764a7441387a35d035efe2.jpg)

(p)  
![](images/4b238f2c77bf8edaea5a95237fd0bc2dae323047f0acd4a0efe1ae0831143f50.jpg)  
Figure 5: Randomization tests for Hurricane Laura use case in Sec. 3. Target XAI field is (Left) Temperature at 925 hPa, (Right) Precipitation. (a,c) Similarity of IG input attribution maps for the 4 paired tests described in App. C.1 measured as Pearson’s correlation. (b,d) Similarity of IG rankings measured as Kendall’s correlation. (Second row) IG explanations for the Laura use case. (Third row) IG explanations for a model with randomized weights. For each target the seed whose IG attribution has the highest correlation with the trained model’s IG attribution is shown. (Fourth row) Model inputs for the target’s own field at t−6 h. Correlations with the corresponding entries in the Second row are reported.

![](images/c2a0d49a3251bd1131ced7366837ce95e5edddeb29148af048a1dc8fa4c7bd3a.jpg)  
Figure 6: Randomization test for (Left) 50 random targets (S0), and (Right) 50 targets with highest prediction error (S1). (First row) IG. (Second row) IxG. (Brackets) Two-sided Wilcoxon signed-rank test, paired by target $( \mathtt { n s } > 0 . 0 5 , \mathtt { * } \le 0 . 0 5 , \mathtt { * * } \le 0 . 0 1 , \mathtt { * * * } \le 1 0 ^ { - 3 } , \mathtt { * * * } \le 1 0 ^ { - 4 } )$ . p-values are Bonferroni-corrected.

The similarity between explanations and inputs is almost null if we compare the spatial patterns (Fig. 5a,c), yet present if we compare the per-variable magnitudes (Fig. 5b,d). Input field magnitude (Fig. 5n,p) share is computed by summing per-variable absolute inputs over space and normalizing by the total over variables.

Fig. 6a-d shows that IG randomization test results are stable across explanations for different target predictions, and agree with what is described above for the Hurricane Laura use case. IxG also performs well under randomization tests (Fig. 6e-h), with similar sensitivity to IG, making it a good explanation technique for fast input attributions. Nonetheless, it lacks completeness by design.

## C.2 Deletion tests

Deletion tests ask whether the inputs that an explanation ranks highest actually drive the prediction: if they do, removing them first should change the model’s output more, and sooner, than removing inputs in any other order [31, 32, 33].

We adapted the test to our regression setting. An input is removed by setting it to zero in normalized space, i.e. to its training mean. Forcing and static features are never modified. Unlike classification tasks, the model has no class score. Hence, to quantify how much the prediction moves we measure the absolute change of the trained model’s prediction at the target, $| f ( x ) - f ( x _ { \mathrm { r e m o v e d } } )$ |, in units of the target variable’s standard deviation. Also, the target’s own field at t−6 h at the target node is never removed: the skip-connection copies it directly to the output, so removing it would make any ranking that places it first look faithful.

For each target prediction we ran three tests, each against a control condition:

Single-entry deletion [32] The k input entries with the largest |IG| are progressively removed. This is compared with two controls: removal in random order (3 orders, averaged), and removal in order of the fields’ input magnitude share (each field’s sum of |x| over all grid nodes, normalized by the total over the 166 fields).

Field deletion The k input fields (one variable at one input step over all grid nodes) with the largest attribution share are progressively removed. This is compared with their removal in random order (3 orders, averaged). This tests the field ranking (Fig. 3e-f) that is actually read in the tool, rather than single entries.

Precipitation, 6 h total

Faithfulness correlation [36, 37] Each of the 166 fields alone is removed, and the resulting prediction change is recorded. Pearson’s ρ is then computed between these and the fields’ IG shares. As a control, the same prediction changes are correlated with each field’s input magnitude share, which asks whether the size of the input alone predicts which field matters.

For the single-entry and field deletion tests, the prediction change plotted against k forms a mostrelevant-first (MoRF) perturbation curve [32]. We summarize each per-target curve by its mean over the removal steps, i.e. the Area Over the Perturbation Curve (AOPC) [32]. The approach is adapted to our spatio-temporal regression task: the output is a regression target, so we plot the absolute change rather than the output. In this setting, AOPC measures the area under our curve. Also, since the input has ≈ 4.8 M entries, k is log-spaced, so the AOPC is the normalized area on a log-k axis. $k = 0$ and full removal are excluded from AOPC computation, since all orderings coincide there.

Temperature at 925 hPa

![](images/1b41466228ce1ee7aa067399c9a2c69b10f9cca8c5e5041c2cf269655a4f895c.jpg)

![](images/9bc62ce34db597e0b9d8e4289517abe223bae9ab8d4b5757ccb5c70b604b0269.jpg)

![](images/e4e1d184be6139007e2e1816022c6f3a2c033f3eb7d0c6425c0a79ad9325cd36.jpg)

![](images/3b510463792df2736fbddbc59bb3ed9cc298e5d2cd63444f2c7aebd723990b08.jpg)

![](images/6f87c53e2971d23a590f0ffa698cfbea2786bc60ad0982a80967b2d21fb785be.jpg)

![](images/4084705af95db5bc7d6ec94fb3191f972a8797de63e822d424b4b3299eb85c93.jpg)  
Figure 7: Deletion tests for the Hurricane Laura use case in Sec. 3. Target XAI field is (Left) Temperature at 925 hPa, (Right) Precipitation. (a,c) Perturbation curves when deleting single entries, ranked by (green) trained model’s IG attributions, (gray) input magnitude, and (light gray) ranked at random. (b,d) Perturbation curves when deleting whole input fields, ranked by (green) trained model’s IG attribution share, (gray) input field magnitude share, and (light gray) at random. (e,f) Faithfulness correlations.

Fig. 7 shows the deletion tests described above for the use case in Fig. 3. The MoRF perturbation curves for single entries (Fig. 7a,c) and whole fields (Fig. 7b,d) suggest that IG explanations well capture the inputs that drive model prediction. There seems to be some overlap between the input fields with the highest magnitude and the IG attributions (Fig. 7b,d and Fig. 5b,d). Faithfulness correlations in Fig. 7e-f settle this: IG reliably identifies the input fields that change model prediction the most, rather than those with the highest absolute magnitude.

Similarly to what was done for the sanity checks, the deletion test has been conducted for the targets in S0 and S1, defined in Sec. C.1. The results in Fig. 8 provide even stronger support to the claims in the paragraph above: IG correctly identifies the fields that drive predictions rather than relying on high-magnitude fields. IxG $( \mathrm { F i g . \dot { 8 } e { \cdot } h , j , l } )$ also performs well under deletion tests, supporting its use in the tool as a fast attribution method.

## D Integrated Gradients

IG attribution scores are computed by integrating the gradient along the path from a reference state to the actual input. The reference state is chosen to be the climatological mean of each field. With this choice the results have to be interpreted as follows: positive values indicate features that increase the model’s prediction above a baseline output value, while negative values indicate features that decrease it below the baseline output. The baseline output is the model output when its input is the climatological mean of each field, i.e. the reference input state. IG guarantees that the sum of attribution scores accounts in full for the prediction shift caused by the input moving from the baseline to the actual value of the field (this property is called completeness).

Deletion of single entries<sup>(a)</sup>  
![](images/ce5c431f5c2b419257850178ad7dd0711d3493de618f8f3441c9f033becaafec.jpg)  
Deletion of single entries<sup>(e)</sup>

![](images/1923a4b3511594cf45503ec1fa19adfb71e726a53df5b7c599fe76844218875a.jpg)  
Deletion of whole fields<sup>(f)</sup>

![](images/dbe703ddf86f5d222b821c8c5403aa7115ed7439bf5855ffbfd85010d367e2c0.jpg)

Deletion of whole fields<sup>(d)</sup>  
![](images/4f10dd31e795983eed347fd62e375a6a0e8a1f256e626fa8fa407fdab37133d6.jpg)

![](images/1d09a6b27fd6bda20563bfb49038560f9e83434daeea8bd10b2b9a49c65ac51f.jpg)

Deletion of whole fields<sup>(h)</sup>  
![](images/cfc0d75c9842317518d393a5ea128e261fe6f64ed3c5be32761ea18df160d247.jpg)

Deletion of single entries<sup>(g)</sup>  
![](images/44046e010b4b5a4cf727841a32ae15331bad69c93bbf2c24e718524de8e4f2b4.jpg)

![](images/bbba6e9741f584822281d835ddbc13884d20709463b6a4ba8ddeb97e7be60eac.jpg)

![](images/4ecd924661f1a63d9b88bf2d7e6f6994703e2f36653ff7e66c04478477ab339d.jpg)

![](images/daccade0f2503607887f66dfae6a64b381ad5554a0be9b32a6e4642ba1f778b7.jpg)

![](images/2ac472d9925731e0c9834165f011862bd8887331c8c0f1a425caa1c5fd076856.jpg)

![](images/2266d33852882fd71f8d5890cc788fe9b6633039d240af54f97eb34588110c3d.jpg)  
Figure 8: Deletion tests for (Left) 50 random targets (S0), and (Right) 50 targets with highest prediction error (S1). (First row) IG attributions (Second row) IxG. (a,c,e,g) Perturbation curves when deleting single entries, ranked by (green) trained model’s IG/IxG attributions, (gray) input magnitude, and (light gray) ranked at random. (b,d,f,h) Perturbation curves when deleting whole input fields, ranked by (green) trained model’s IG/IxG attribution share, (gray) input field magnitude share, and (light gray) at random. Lines are medians over targets, shaded areas show the range from the 25th to the 75th percentiles. (Third row) Faithfulness correlations for (i,k) IG and (j,l) IxG. (Brackets) Two-sided Wilcoxon signed-rank test of trained vs. input-only $\rho ,$ paired by target $( \mathrm { n s } > 0 . 0 5 ,$ \* $\leq 0 . 0 5 , * * \leq 0 . 0 1 , * * * \leq 1 0 ^ { - 3 } , * * * * \leq 1 0 ^ { - 4 } )$ . p-values are Bonferroni-corrected.

We approximate this integral with a Riemann sum, recovering a fraction (Fig. 9c,f) of this shift depending on the integration steps m. A high number of integration steps ensures higher completeness at the cost of higher latency. To quantify the effects of the integration steps on the resulting attributions we computed two quantities: 1) the completeness gap (%), expressed as the difference between the prediction shift explained by IG and the real prediction shift, divided by the latter. 2) Pearson’s correlation between IG attributions computed with m integration steps and twice as much (2m), with increasing number of integration steps m.

Fig. 9 shows the results for the S0 and S1 target pools from App. C. IG with $m = 4$ steps recovers 96% of the prediction shift at the median for the worst-points pool (S1, Fig. 9f). Input attributions seem to stabilize after $m = 4 \colon$ IG input attributions computed with $m = 8$ integration steps are well correlated with the attributions computed with $m = 4 ,$ and the same holds for all larger m (Fig. 9).

![](images/4c19bb26c6b8009ac4d029ba620133bbd771283b84a6c615b78570c15c652278.jpg)

![](images/6618017f7987e6f621c60b6779742dd7a8891285fa732efd211a8ab9730596be.jpg)

![](images/8c7b12a30c09f783f26b082650418435aa4a09302b2eab98b0574527532bbd11.jpg)

![](images/c5a93eba17183031c957ec730280f78b6d92d482f782e86e2f7147662a6a6c09.jpg)

![](images/e731b5d8eb28572754cce1992c1f81c961958da7c478dd3d6bf18f5fc2262d81.jpg)

![](images/65acd6fce3f43a0e837112f9b54c6de1aa3c5f09b61a1cfc6efff238cd2c2245.jpg)  
Figure 9: IG completeness tests for (First row) 50 random targets (S0), and (Second row) 50 targets with highest prediction error (S1). (a,d) Convergence of IG input attribution maps. (b,e) Convergence of IG input rankings. (c,f) Completeness gap with increasing number of integration steps (m).

This holds for both attribution maps (Fig. 9a,d) and rankings (Fig. 9b,e). For this reason, we choose m = 4 as the default in the tool, to trade off latency against exactness. Nonetheless, completeness for m = 4 still varies significantly (Fig. 9c,f), at times exceeding the 5% limit proposed by [23]. For this reason, we suggest using $m = 8$ integration steps if deploying locally with GPU acceleration.

As for the Hurricane Laura use case discussed in Sec. 3, Fig. 10a-d reports the same attribution panels as in Fig. 3b,d,e,f computed with $m = 6 4$ integration steps instead of $m = 4$ , as well as the convergence curves (Fig. 10e-h). Using m = 64 integration steps, IG attributions account for 99.99% of the prediction shift, hence we can consider this as the ground truth for IG explanations. The comparison is positive: IG with m = 4 integration steps recovers 99.42% of the temperature prediction shift (Fig. 10e,f) and 98.8% of the precipitation one (Fig. 10g,h). The input attributions in Fig. 3 also closely resemble those in Fig. 10, except for a minor reduction in magnitude of a few IG attributions in the precipitation panel (Fig. 3d vs Fig. 10c). The rankings match too, with only a change in relative position between Specific Humidity at 1000 hPa (t−6 h) and Temperature at 925 hPa (t−12 h) when going from m = 4 to m = 64 (Fig. 3e vs Fig. 10b). This is, however, minor if we compare their attribution share.

![](images/c65d9192e86401f901b0b6e862cae890452c9f71e17a047fcb78ea912c573af5.jpg)

![](images/daaa0497cceed84cde9d35ddb75d936ed4708e74577726ac4dcd9720aff5630c.jpg)

![](images/2354046f896bb18ef28a2bea408b2b00867621ef2c096456ebac534a8a005dee.jpg)

![](images/cbd5a41a3cc96b02cff735a8e4dc3cc682c4241eb75397889a606fd4c3450014.jpg)

![](images/ff9e7a650592b576696dc4e6782b254bbf58c9ad1c4a209028460c41ffa41d82.jpg)

![](images/f0f9fef0d6491789a678e7b6c6d8b25946da5487c8c04977862f50d5d54f09c0.jpg)

![](images/d223df8021d15f5e66fbef974b287fffde6a043e5037b44a241f1166e8484281.jpg)

![](images/c1dae00abcdec458abc5ba279b2af737e53b08b148951d5a3b0e85bceb07b71c.jpg)  
Figure 10: IG completeness tests for the Hurricane Laura use case in Sec. 3. Target XAI field is (Left) Temperature at 925 hPa, (Right) Precipitation. (First row) IG explanations for the Laura use case computed with m = 64 integration steps. (e,g) Convergence of IG input attribution maps. (f,h) Convergence of IG input rankings. In all four plots the completeness gap for using a given m is reported on the horizontal axis. The vertical axis reports correlations between attributions computed with m and 2m integration steps.